# Stage 8 — SciPy parametric load model — codebase map

**Audience:** Stage 8 implementer, before opening any file.
**Status:** Research note — immutable after Stage 8 ships.
**Date:** 2026-04-21
**Baseline SHA:** `50b1970`

---

## §1 Touchpoint inventory

### NEW files

| Path | Purpose |
|------|---------|
| `src/bristol_ml/models/scipy_parametric.py` | `ScipyParametricModel` — parametric temperature-response + Fourier model behind the `Model` protocol |
| `conf/model/scipy_parametric.yaml` | Hydra group file, `# @package model`, `type: scipy_parametric` and config defaults |
| `conf/model/scipy_parametric.yaml` | mirrors `conf/model/sarimax.yaml` in shape |
| `tests/unit/models/test_scipy_parametric.py` | Spec-derived test suite |
| `scripts/_build_notebook_08.py` | Deterministic notebook generator (mirrors Stage 7 template — see §7) |
| `notebooks/08_scipy_parametric.ipynb` | Demo notebook — fitted temperature-response curve, parameter table with CIs |
| `docs/lld/stages/08-scipy-parametric.md` | Stage retro (hygiene, written at close) |

### MODIFIED files

| Path | Anchor | What changes |
|------|--------|--------------|
| `conf/_schemas.py:545` | `ModelConfig = NaiveConfig \| LinearConfig \| SarimaxConfig` | Extend union to `… \| ScipyParametricConfig`; add `ScipyParametricConfig` class above the union line |
| `src/bristol_ml/evaluation/harness.py:475-491` | `_build_model_from_config` three-branch `isinstance` ladder | Add a fourth `isinstance(model_cfg, ScipyParametricConfig)` branch returning `ScipyParametricModel(model_cfg)` |
| `src/bristol_ml/train.py:218-252` | Inline `isinstance` ladder in `_cli_main` | Add a fourth `elif isinstance(model_cfg, ScipyParametricConfig)` branch (mirrors the `SarimaxConfig` branch exactly, including the `feature_columns` promotion) |
| `src/bristol_ml/models/__init__.py` | Lazy re-export surface | Add `ScipyParametricModel` and `ScipyParametricConfig` |
| `pyproject.toml:13-30` | `[project].dependencies` | Add `scipy>=1.13,<2` — **scipy is NOT in the current dependency list** (confirmed at baseline SHA) |
| `CHANGELOG.md` | `[Unreleased]` section | Stage 8 bullets under `### Added` and `### Changed` |
| `src/bristol_ml/models/CLAUDE.md` | Current surface section | Add `ScipyParametricModel` entry; document the module-level function constraint and covariance serialisation |
| `docs/stages/README.md` | Stage 8 row | Flip status to `shipped`; link intent, plan, layer, LLD, retro |
| `docs/architecture/layers/models.md:1` | Status line | Update from "extended by Stage 7" to include Stage 8 |
| `docs/plans/active/08-*.md` | Active plan | Move to `docs/plans/completed/` |
| `uv.lock` | Lock file | Regenerated after adding `scipy` — run `uv sync --group dev` |

### REFERENCED (read-only)

| Path | Why |
|------|-----|
| `src/bristol_ml/models/protocol.py:42-89` | The five-member `Model` protocol; runtime-checkable caveat |
| `src/bristol_ml/models/io.py:32-68` | `save_joblib` / `load_joblib` — atomic joblib round-trip |
| `src/bristol_ml/models/sarimax.py` | Implementation template for constructor, state, `fit`/`predict`/`save`/`load`/`metadata` |
| `src/bristol_ml/features/fourier.py:56-152` | `append_weekly_fourier` — reused for both diurnal and weekly Fourier terms |
| `src/bristol_ml/features/assembler.py:74-144` | `OUTPUT_SCHEMA` and `CALENDAR_OUTPUT_SCHEMA` — confirms `temperature_2m` column name |
| `conf/_schemas.py:494-539` | `SarimaxConfig` and `SarimaxKwargs` — template for `ScipyParametricConfig` |
| `conf/_schemas.py:548-589` | `ModelMetadata` — `hyperparameters` bag for parameter names, values, std errors |
| `src/bristol_ml/evaluation/harness.py:206-259` | Harness `predict(X_test)` call contract |
| `tests/unit/models/test_sarimax.py` | Test-naming and helper conventions to mirror |
| `pyproject.toml:57-68` | `@pytest.mark.slow` registration, `addopts = "-m 'not slow'"` |
| `docs/intent/08-scipy-parametric.md` | Acceptance criteria (authoritative) |

---

## §2 Protocol conformance contract

### Constructor

Pattern established by `SarimaxModel.__init__(config: SarimaxConfig)` at `sarimax.py:91-107`. The Stage 8 constructor should be:

```python
def __init__(self, config: ScipyParametricConfig) -> None:
    self._config: ScipyParametricConfig = config
    self._popt: np.ndarray | None = None         # shape (n_params,)
    self._pcov: np.ndarray | None = None         # shape (n_params, n_params)
    self._feature_columns: tuple[str, ...] = ()
    self._fit_utc: datetime | None = None
    self._param_names: tuple[str, ...] = ()      # ordered names matching _popt
```

`_popt` and `_pcov` are the direct outputs of `scipy.optimize.curve_fit`. Unlike `SARIMAXResultsWrapper` (an opaque object with internal state), both are plain `np.ndarray` — joblib handles these cleanly with no special treatment.

### `fit` signature and internal state

```python
def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
```

Mirrors `sarimax.py:113-217` in structure:
- Length-parity and UTC-index guards (same `_require_utc_datetimeindex` pattern).
- Feature-column resolution (same `_resolve_feature_columns` pattern — `feature_columns=None` uses all columns).
- Fourier append via `append_weekly_fourier` (see §4).
- Call `scipy.optimize.curve_fit(f, xdata, ydata, p0=..., bounds=..., maxfev=...)`.
- Store `_popt`, `_pcov`, `_feature_columns`, `_fit_utc`, `_param_names`.
- Re-entrant: a second call overwrites all five fields unconditionally.

The parametric function `f` must be a **module-level pure function**, not a lambda or bound method (see §6 and §10).

### `predict` signature

```python
def predict(self, features: pd.DataFrame) -> pd.Series: ...
```

Raises `RuntimeError` if `_popt is None`. Evaluates `f(xdata, *self._popt)` on the test feature matrix and returns `pd.Series(values, index=features.index, name=self._config.target_column)`. No re-indexing surprise applies here (unlike SARIMAX) — `curve_fit` is a regression over an arbitrary array, not a time-series state-space model; the output length matches `len(features)` exactly.

### `save` / `load`

Both `_popt` (shape `(n_params,)`) and `_pcov` (shape `(n_params, n_params)`) are plain `np.ndarray`. joblib serialises ndarrays natively and efficiently — no issue (confirmed by Stage 7's pattern of saving the entire `SarimaxModel` instance, which itself carries ndarrays). The entire `ScipyParametricModel` instance is passed to `save_joblib(self, path)` exactly as in `sarimax.py:263-288`.

```python
def save(self, path: Path) -> None:
    if self._popt is None:
        raise RuntimeError("Cannot save unfitted ScipyParametricModel")
    save_joblib(self, path)

@classmethod
def load(cls, path: Path) -> ScipyParametricModel:
    obj = load_joblib(path)
    if not isinstance(obj, cls):
        raise TypeError(...)
    return obj
```

### `metadata` — `hyperparameters` dict content

The `hyperparameters` bag at `conf/_schemas.py:580` is `dict[str, Any]`. For the parametric model it should carry:

```python
hyperparameters = {
    "target_column": self._config.target_column,
    "temperature_harmonics": self._config.temperature_harmonics,
    "diurnal_harmonics": self._config.diurnal_harmonics,
    "weekly_harmonics": self._config.weekly_harmonics,
    # After fit:
    "param_names": list(self._param_names),
    "param_values": self._popt.tolist(),           # list[float], JSON-safe
    "param_std_errors": np.sqrt(np.diag(self._pcov)).tolist(),
    "covariance_matrix": self._pcov.tolist(),      # nested list[list[float]]
}
```

`np.ndarray.tolist()` produces a plain Python `list[list[float]]` that is JSON-serialisable and survives the `dict[str, Any]` bag without pickling. Storing as a nested list (rather than a base64 blob or a re-serialised ndarray) is the simplest option that satisfies AC-5 (save/load preserves CI information) and keeps `metadata` inspectable in notebooks without additional unpacking. When `_pcov` has `inf` entries (near-singular fit — see §6), `np.inf` does not serialise to JSON but `float('inf')` does; use `np.where(np.isinf(self._pcov), float('inf'), self._pcov).tolist()`.

The `ModelMetadata.name` regex is `^[a-z][a-z0-9_.-]*$` (`conf/_schemas.py:567`). A helper like `_build_metadata_name(config)` → `"scipy-parametric-t{t}-d{d}-w{w}"` (temperature/diurnal/weekly harmonics counts) satisfies it.

---

## §3 Feature-table shape at Stage 8

After Stages 3, 5, and 7 the feature table has two schemas.

**`weather_only`** — 10 columns (`assembler.py:91-100`):

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_utc` | `timestamp[us, tz=UTC]` | index after `set_index` in `train.py:217` |
| `nd_mw` | `int32` | target |
| `tsd_mw` | `int32` | |
| `temperature_2m` | `float32` | **Stage 8 temperature term** |
| `dew_point_2m` | `float32` | |
| `wind_speed_10m` | `float32` | |
| `cloud_cover` | `float32` | |
| `shortwave_radiation` | `float32` | |
| `neso_retrieved_at_utc` | `timestamp[us, tz=UTC]` | provenance |
| `weather_retrieved_at_utc` | `timestamp[us, tz=UTC]` | provenance |

**`weather_calendar`** — 55 columns = `weather_only` (10) + 44 calendar `int8` + `holidays_retrieved_at_utc` (`assembler.py:118-144`).

`temperature_2m` is at column position 3 in both schemas and is guaranteed `float32`. The assembler's `build()` contract guarantees no `NaN` values in the output. Stage 8's temperature-response term consumes `temperature_2m` directly — it is available by name on both schemas; no extra ingestion is needed.

The index is `DatetimeIndex` tz-aware UTC after `set_index("timestamp_utc")` is called by `train.py:217`. The assembler does **not** set `df.index.freq` — this is irrelevant to `curve_fit` (which treats its inputs as plain arrays, not time-series) but relevant if the implementer also attaches a `freq` for a future SARIMAX-style surrogate call. In practice: `curve_fit` does not inspect `index.freq`, so the assembler's omission is a non-issue for Stage 8.

---

## §4 Fourier helper reuse

`append_weekly_fourier` is at `src/bristol_ml/features/fourier.py:56-152`. Its signature:

```python
def append_weekly_fourier(
    df: pd.DataFrame,
    *,
    period_hours: int = 168,
    harmonics: int = 3,
    column_prefix: str = "week",
) -> pd.DataFrame:
```

The integer-hour derivation that makes it DST-insensitive is at lines 134-135:

```python
hours_since_epoch = df.index.view("int64") // _NANOSECONDS_PER_HOUR
t = np.asarray(hours_since_epoch, dtype=np.float64)
```

where `_NANOSECONDS_PER_HOUR = 3_600_000_000_000` (line 53).

**Calling it with `period_hours=24` for the diurnal term works correctly.** The calculation at line 142 is:

```python
angular = 2.0 * np.pi * float(k) * t / float(period_hours)
```

Substituting `period_hours=24` gives `2πk · t / 24` where `t` is the integer hour since the Unix epoch (UTC-anchored). This is exactly the correct diurnal Fourier basis because:

1. UTC is continuous — DST does not cause a phase jump. The London clock may gain or lose an hour on DST-change Sundays, but UTC hours are unaffected, so the diurnal signal at `t mod 24` is well-defined.
2. The assembler contract guarantees exactly 24 UTC rows per calendar day (including DST-change days), so `period_hours=24` maps exactly to one calendar day in UTC.

**There is no subtle DST-drift issue with `period_hours=24`**. The Stage 7 SARIMAX used `period_hours=168` (weekly) but the helper is fully general. A Stage 8 call would be:

```python
df = append_weekly_fourier(df, period_hours=24, harmonics=k_diurnal, column_prefix="diurnal")
df = append_weekly_fourier(df, period_hours=168, harmonics=k_weekly, column_prefix="weekly")
```

Both calls return new DataFrames (no mutation) and both require a tz-aware DatetimeIndex (same as SARIMAX). The resulting columns are named `diurnal_sin_k1`, `diurnal_cos_k1`, … and `weekly_sin_k1`, `weekly_cos_k1`, … — no name collisions.

---

## §5 Harness and train-CLI dispatchers

### `harness.py:475-491` — standalone `_build_model_from_config`

Current three-branch ladder:

```python
def _build_model_from_config(model_cfg: object) -> Model | None:
    from conf._schemas import LinearConfig, NaiveConfig, SarimaxConfig

    if isinstance(model_cfg, NaiveConfig):
        from bristol_ml.models.naive import NaiveModel
        return NaiveModel(model_cfg)
    if isinstance(model_cfg, LinearConfig):
        from bristol_ml.models.linear import LinearModel
        return LinearModel(model_cfg)
    if isinstance(model_cfg, SarimaxConfig):
        from bristol_ml.models.sarimax import SarimaxModel
        return SarimaxModel(model_cfg)
    return None
```

Stage 8 adds a fourth branch before `return None`:

```python
    if isinstance(model_cfg, ScipyParametricConfig):
        from bristol_ml.models.scipy_parametric import ScipyParametricModel
        return ScipyParametricModel(model_cfg)
```

The local import keeps harness.py lazy with respect to scipy (same pattern as the SARIMAX branch keeps it lazy with respect to statsmodels — documented in the Stage 7 CHANGELOG at the `### Changed` bullet).

### `train.py:218-262` — inline dispatcher in `_cli_main`

Current three-branch ladder (lines 221-261):

```python
if isinstance(model_cfg, NaiveConfig):
    primary = NaiveModel(model_cfg)
    primary_kind = "naive"
elif isinstance(model_cfg, LinearConfig):
    ...
    primary = _NamedLinearModel(linear_cfg, ...)
    primary_kind = "linear"
elif isinstance(model_cfg, SarimaxConfig):
    ...
    primary = SarimaxModel(sarimax_cfg)
    primary_kind = "sarimax"
else:  # pragma: no cover
    print(f"No harness factory for ...", file=sys.stderr)
    return 3
```

Stage 8 adds a fourth `elif` before the `else` clause, and the `else` branch moves down one. The pattern mirrors the `SarimaxConfig` branch exactly: promote `feature_columns=None` to the resolved column tuple, log if pinned, instantiate `ScipyParametricModel(scipy_cfg)`. Import: `from bristol_ml.models.scipy_parametric import ScipyParametricModel`.

**Decision point for the plan to surface:** The Stage 7 Phase 3 review flagged this two-site duplication as a candidate for an ADR (noted in the Stage 7 retro as "B1, not filed"). Stage 8 will be the **second** successive model addition that requires both sites to be updated in lockstep. If the Stage 8 plan does not file an ADR or propose a consolidation, a Stage 9 or Stage 10 implementer will face a triple-site update. The plan must explicitly surface whether to file ADR-B1 ("keep two sites as documented tech debt") or refactor `_build_model_from_config` in `train.py` into a shared helper that both `harness.py` and `train.py` call.

---

## §6 `curve_fit` integration surface

### Signature

```python
scipy.optimize.curve_fit(
    f,           # callable: f(xdata, *params) -> ndarray
    xdata,       # array-like, shape (n_features, n_obs) or (n_obs,)
    ydata,       # array-like, shape (n_obs,)
    p0=None,     # initial guess, shape (n_params,)
    sigma=None,  # weights
    absolute_sigma=False,
    check_finite=True,
    bounds=(-np.inf, np.inf),
    method=None, # 'lm', 'trf', 'dogbox'
    jac=None,
    **kwargs,    # passed to the underlying minimiser
) -> tuple[ndarray, ndarray]  # (popt, pcov)
```

### Return shape and `pcov` hazards

- `popt`: shape `(n_params,)` — the optimal parameter vector.
- `pcov`: shape `(n_params, n_params)` — the estimated covariance matrix of `popt`, derived from the Jacobian at the solution. Standard errors are `np.sqrt(np.diag(pcov))`.

When the fit is near-singular (e.g. poorly conditioned Jacobian, inadequate data coverage of the temperature range), `pcov` will contain `inf` values for the affected rows/columns. `scipy` emits an `OptimizeWarning` in this case rather than raising. The `inf`-detection pattern:

```python
if np.any(np.isinf(pcov)):
    logger.warning("ScipyParametricModel.fit: covariance matrix contains inf entries — "
                   "parameter identifiability may be poor; check initial guesses and data range.")
```

### Failure modes observable from test code

| Exception / Warning | Trigger | How to handle |
|--------------------|---------|---------------|
| `scipy.optimize.OptimizeWarning` | Covariance estimation failed; `pcov` → `inf` | Catch with `warnings.catch_warnings(record=True)` and re-emit at loguru WARNING; same pattern as `SarimaxModel.fit` for `ConvergenceWarning` (`sarimax.py:201-211`) |
| `RuntimeError("Optimal parameters not found")` | Iteration limit exceeded (`maxfev`) | Re-raise as `RuntimeError` with a message naming the model and suggesting increasing `maxfev` or improving `p0` |
| `np.linalg.LinAlgError` | Jacobian is exactly singular (rarer) | Re-raise |
| `ValueError` | `p0` shape mismatch, `bounds` shape mismatch | Surface immediately; these indicate a bug in the parametric-function definition |

Suppression idiom:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    popt, pcov = curve_fit(f, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)
for w in caught:
    if issubclass(w.category, OptimizeWarning):
        logger.warning("ScipyParametricModel.fit: OptimizeWarning: {}", str(w.message))
```

### `least_squares` vs `curve_fit`

`curve_fit` wraps `scipy.optimize.least_squares` when `method="trf"` or `method="dogbox"` (passing `loss="linear"` by default). Using `least_squares` directly enables robust loss functions (`loss="huber"`, `loss="soft_l1"`) that down-weight demand outliers. The intent document notes this as a valid choice ("Robustness to outliers — Huber, soft L1 change the answer"). If Stage 8 wants to support robust losses, `least_squares` is the primitive; `curve_fit` does not expose `loss`. For the initial implementation, `curve_fit` is simpler and consistent with the intent document's framing ("numerical optimiser"). A config field `loss: Literal["linear", "huber", "soft_l1"] = "linear"` can gate the two code paths.

### Module-level function constraint

The parametric function `f(xdata, *params)` **must be a module-level function** (`scipy_parametric.py` top-level `def`), not a lambda, a locally-defined function, or a bound method. Reason: `save_joblib` uses `joblib.dump` which uses pickle; lambdas and local functions are not pickleable. The entire `ScipyParametricModel` instance is pickled, including any reference to `f`. If `f` is a module-level function, pickle stores only its qualified name (`bristol_ml.models.scipy_parametric._parametric_model_fn`) and reconstructs on load. A lambda would cause `_pickle.PicklingError: Can't pickle <function <lambda>>`.

Pattern: define one or more private module-level functions at the top of `scipy_parametric.py`:

```python
def _parametric_fn(X: np.ndarray, *params: float) -> np.ndarray:
    """Module-level; must remain pickleable. Called by curve_fit and predict."""
    ...
```

Then pass `_parametric_fn` (not an instance attribute holding a lambda) to `curve_fit`. If the functional form is selected by config, use a module-level registry dict `_FUNCTIONAL_FORMS: dict[str, Callable] = {"piecewise_linear": _piecewise_linear_fn, ...}` with each entry a module-level function.

---

## §7 Notebook and generator pattern

`scripts/_build_notebook_07.py` is 617 lines. Structure:

- Lines 1-33: module docstring explaining the three-step regeneration flow (`_build_notebook_07.py` → `nbconvert --execute` → `ruff format`).
- Lines 35-62: two helper functions `md(source) -> dict` and `code(source) -> dict` that build cell dicts; a global `_CELL_COUNTER` with `_next_id(prefix)` for deterministic per-cell IDs.
- Lines 64-577: thirteen named cell variables (`cell_0` through `cell_12`), each either `md(...)` or `code(...)`.
- Lines 580-616: a `notebook` dict assembled from the cell list plus `metadata` (kernelspec, language_info, `nbformat=4`, `nbformat_minor=5`); `OUT.write_text(json.dumps(notebook, indent=1) + "\n")`.

**The script is directly reusable as a template for `scripts/_build_notebook_08.py`.** The two helper functions (`md`, `code`, `_next_id`) are entirely generic. The only Stage-7-specific content is the cell bodies and the `OUT` path. A Stage 8 generator copies the preamble, changes `OUT` to `notebooks/08_scipy_parametric.ipynb`, and replaces the cell bodies.

The regeneration flow is:

```
uv run python scripts/_build_notebook_08.py
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/08_scipy_parametric.ipynb
uv run ruff format notebooks/08_scipy_parametric.ipynb
```

The Stage 7 notebook comment (line 23) notes: "The generator's cell-source strings are not pre-formatted to ruff's line-wrapping conventions … the final `ruff format` step is mandatory." This applies equally to Stage 8.

---

## §8 Test conventions

`tests/unit/models/test_sarimax.py` contains **16 tests** across three task groups (T3, T4, T5) plus a `@pytest.mark.slow` benchmark. Structure:

- **Module-level helpers** (lines 50-103):
  - `_synthetic_utc_frame(n_rows)` → `tuple[pd.DataFrame, pd.Series]`: tz-aware UTC hourly index, two `float64` exog columns (`temp_c`, `cloud_cover`), AR(1) + daily + weekly sine target scaled to ~10 000 MW, reproducible via `np.random.default_rng(0)`.
  - `_synthetic_utc_frame_4col(n_rows)` → `tuple[pd.DataFrame, pd.Series]`: four-column variant used in T5 save/load tests.
  - `_FAST_CONFIG`: `SarimaxConfig(order=(1,0,0), seasonal_order=(0,0,0,24), weekly_fourier_harmonics=2)` — a minimal fast config for non-order-sensitive tests.
  - `_FAST_CONFIG_NO_FOURIER`: same with `weekly_fourier_harmonics=0`.

Stage 8 should mirror this with a `_synthetic_utc_frame_with_temperature(n_rows)` helper that includes a `temperature_2m` column with realistic variation (e.g. `rng.normal(loc=10.0, scale=6.0)`) and a target that includes a genuine temperature-response component (quadratic or piecewise) so the parametric fit has signal to find.

**`@pytest.mark.slow` machinery** is confirmed in place at `pyproject.toml:63-66`:

```toml
addopts = "-ra --strict-markers --record-mode=none -m 'not slow'"
markers = [
    "slow: benchmark/performance guards excluded from the default run; opt in with -m slow.",
]
```

Stage 8 should add a `@pytest.mark.slow` benchmark test (analogous to `test_sarimax_fit_single_fold_completes_under_60_seconds` at `test_sarimax.py:1196-1267`) that asserts `curve_fit` on a full-year dataset completes under a reasonable wall-clock budget. The `curve_fit` call on `n_obs ≈ 8760` rows with a ~10-parameter model should be much faster than SARIMAX (seconds, not tens of seconds), so the time budget can be tighter (e.g. 10 s).

---

## §9 Stage hygiene files

Stage 7 Task T8 (retro `docs/lld/stages/07-sarimax.md` section "T8 — Stage hygiene") updated exactly these files:

1. `CHANGELOG.md` — `[Unreleased]` bullets under `### Added` and `### Changed`.
2. `docs/stages/README.md` — Stage 7 row flipped to `shipped` with five links.
3. `docs/architecture/layers/models.md` — inventory row for `sarimax.py` flipped from `(planned)` to `Shipped`.
4. `src/bristol_ml/models/CLAUDE.md` — SARIMAX specifics subsection.
5. `docs/lld/stages/07-sarimax.md` — this retro (written as the final deliverable).
6. `docs/plans/active/07-sarimax.md` → `docs/plans/completed/07-sarimax.md` — plan archival.

Stage 8's checklist is identical in shape:

1. `CHANGELOG.md` — Stage 8 bullets.
2. `docs/stages/README.md` — Stage 8 row.
3. `docs/architecture/layers/models.md` — Stage 8 row.
4. `src/bristol_ml/models/CLAUDE.md` — `ScipyParametricModel` entry + pickleability note.
5. `docs/lld/stages/08-scipy-parametric.md` — retro.
6. `docs/plans/active/08-*.md` → `docs/plans/completed/08-*.md`.

One additional item unique to Stage 8: `pyproject.toml` and `uv.lock` change because `scipy` is a new dependency. The CHANGELOG `### Added` bullet must name the dependency pin (e.g. `scipy>=1.13,<2`).

---

## §10 Codebase surprises

**S1 — `scipy` is not in `pyproject.toml`.** This is the single largest first-attempt trap. `scipy.optimize.curve_fit` will import cleanly inside the Docker container only if `scipy` is in `[project].dependencies`. The baseline has `statsmodels>=0.14,<1` which transitively pulls in `scipy` as a statsmodels dependency, so `import scipy` does not fail at the container's current state — but relying on transitive availability is fragile. Stage 8 must add `scipy>=1.13,<2` (or similar) explicitly and regenerate `uv.lock`. The `pyproject.toml:13-30` dependency block is the target; it currently ends at `seaborn>=0.13,<1`.

**S2 — The parametric function `f` must be module-level for pickle.** Covered in §6 but worth repeating as a first-attempt stub-your-toe hazard: defining `f` as a local function inside `fit()` or as a `lambda` is the most natural first attempt, and it will fail at `model.save(path)` with `_pickle.PicklingError`. The entire model instance is pickled by `save_joblib(self, path)`. Module-level pure functions survive pickle; closures and lambdas do not.

**S3 — `pcov` can contain `inf`; `np.sqrt(np.diag(pcov))` will produce `nan` or `inf` std errors.** `curve_fit` does not raise on a near-singular Jacobian; it fills `pcov` with `inf` and emits `OptimizeWarning`. Downstream code that calls `np.sqrt(np.diag(pcov))` for the standard errors will get `nan` for any affected parameter. The `metadata.hyperparameters` serialisation must guard against this (e.g. `float('nan')` is JSON-serialisable; `np.nan` in a nested list becomes `null` in `json.dumps` by default). Test code should include a test for `_pcov` containing `inf` on a pathologically underdetermined fit.

**S4 — Diurnal Fourier call is `period_hours=24`, not `period_hours=168`.** The helper's default is 168 (weekly). A Stage 8 implementer adding both diurnal and weekly Fourier terms makes two calls. The `period_hours=24` call for the diurnal term is easy to misread as "period 24 harmonics" rather than "period 24 hours". The `column_prefix` argument must differ between the two calls (`"diurnal"` and `"weekly"` or similar) to avoid column-name collisions.

**S5 — `features.index.freq` is irrelevant but tempting to set.** A first-attempt implementer familiar with the SARIMAX `freq="h"` surprise (Stage 7 plan surprise 2) may attempt to set `features.index.freq = "h"` in `ScipyParametricModel.fit` by habit. This is harmless but unnecessary — `curve_fit` treats its `xdata` as a plain ndarray and does not inspect `DatetimeIndex.freq` at all. The UTC-index guard remains necessary (the `append_weekly_fourier` call requires a tz-aware index), but the `freq` attribute is irrelevant.

**S6 — `p0` (initial parameter guess) must be documented.** The intent document states: "Documenting sensible starting values is important" (`docs/intent/08-scipy-parametric.md` §"Points for consideration"). Without reasonable `p0`, `curve_fit` may converge to a local minimum or fail entirely. `ScipyParametricConfig` should carry a `p0: tuple[float, ...] | None = None` field (analogous to `SarimaxConfig.feature_columns: tuple[str, ...] | None`), with `None` meaning "use a heuristic from the data" (e.g. mean demand as the base level, zero for all harmonic amplitudes). This must be settled in the config schema before implementation begins.

---

## Relevant file index

| Path | Purpose |
|------|---------|
| `src/bristol_ml/models/protocol.py` | `Model` protocol + `ModelMetadata` re-export |
| `src/bristol_ml/models/io.py` | Atomic joblib save/load helpers |
| `src/bristol_ml/models/sarimax.py` | Implementation template for every protocol member |
| `src/bristol_ml/features/fourier.py:56-152` | `append_weekly_fourier` — reused for diurnal and weekly Fourier terms |
| `src/bristol_ml/features/assembler.py:74-144` | `OUTPUT_SCHEMA`, `CALENDAR_OUTPUT_SCHEMA` — confirms `temperature_2m` column |
| `conf/_schemas.py:494-545` | `SarimaxConfig`, `SarimaxKwargs`, `ModelConfig` union — template for Stage 8 config |
| `conf/_schemas.py:548-589` | `ModelMetadata` — `hyperparameters` bag |
| `conf/model/sarimax.yaml` | Template Hydra group file |
| `src/bristol_ml/evaluation/harness.py:475-491` | `_build_model_from_config` — fourth branch insertion point |
| `src/bristol_ml/train.py:218-262` | Inline dispatcher — fourth `elif` insertion point |
| `tests/unit/models/test_sarimax.py` | Test naming, helper structure, `@pytest.mark.slow` pattern |
| `pyproject.toml:57-68` | `@pytest.mark.slow` registration; scipy dependency target at lines 13-30 |
| `scripts/_build_notebook_07.py` | Notebook generator template (617 lines, directly reusable) |
| `docs/intent/08-scipy-parametric.md` | Acceptance criteria (authoritative) |
| `docs/lld/stages/07-sarimax.md` | T8 hygiene checklist precedent |
