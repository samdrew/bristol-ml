# Stage 7 — SARIMAX codebase map

**Audience:** Stage 7 implementer, before opening any file.  
**Status:** Research note — immutable after Stage 7 ships.  
**Date:** 2026-04-21

---

## 1. The `Model` protocol contract

**File:** `src/bristol_ml/models/protocol.py`

```python
@runtime_checkable
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Model: ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

`ModelMetadata` fields (`conf/_schemas.py:475–507`):

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str` matching `^[a-z][a-z0-9_.-]*$` | e.g. `"sarimax-d1-d1-s168"` |
| `feature_columns` | `tuple[str, ...]` | ordered; exact columns `fit` trained on |
| `fit_utc` | `datetime \| None` (tz-aware UTC) | `None` before fit; naive rejected by validator |
| `git_sha` | `str \| None` | outside a git tree: `None` |
| `hyperparameters` | `dict[str, Any]` | free-form; by convention matches config field names |

**Invariants enforced by tests** (`tests/unit/models/test_protocol.py`):

- `isinstance(m, Model)` checks attribute presence only — not signatures (PEP 544 caveat; `test_model_protocol_caveat_signatures_not_checked`).
- A class missing any one of the five members fails `isinstance` (`test_model_protocol_rejects_missing_method`).
- `ModelMetadata` must be importable from `bristol_ml.models.protocol` (it re-exports from `conf._schemas`).

**Invariants in `protocol.py:54–65` docstring (not in tests; load-bearing):**

- `fit` must be **re-entrant**: a second call discards the previous fit entirely.
- `predict` before `fit` must raise `RuntimeError` (not return stale data).
- `metadata` is callable before `fit`; `metadata.fit_utc is None` signals unfitted.
- Cross-version load compatibility is an **explicit non-goal** (Stage 9 registry owns it).

**Stage 5 `_NamedLinearModel` wrapper pattern** (`src/bristol_ml/train.py:337–385`): `train.py` wraps `LinearModel` in a thin composition class to override `metadata.name` without touching the model itself. Stage 7 does not need this pattern if `SarimaxModel.metadata` returns the name directly from config — but the pattern shows how to satisfy the protocol structurally without inheritance.

---

## 2. `LinearModel` and `NaiveModel` structure — annotated map

**`LinearModel`** (`src/bristol_ml/models/linear.py`):

- Constructor (`:67`): stores `LinearConfig`; sets `_results: RegressionResultsWrapper | None = None`, `_feature_columns: tuple = ()`, `_fit_utc: datetime | None = None`.
- `fit` (`:79`): validates length parity; resolves feature columns; casts to `float64`; optionally calls `sm.add_constant`; calls `sm.OLS(y, X).fit()`; stores `_results`, `_feature_columns`, `_fit_utc`. **Discards prior state on re-call.**
- `predict` (`:121`): raises `RuntimeError` if unfitted; reconstructs `X` from `_feature_columns`; calls `self._results.predict(X)`; returns `pd.Series` with `name=config.target_column`, indexed to `features.index`.
- `save`/`load` (`:152–176`): thin delegation to `save_joblib(self, path)` / `load_joblib(path)`. `load` does `isinstance(obj, cls)` guard and raises `TypeError` on mismatch.
- `metadata` property (`:178`): constructs `ModelMetadata` fresh on each call; includes `rsquared`, `nobs`, serialised `coefficients` in `hyperparameters`.
- Public `.results` property (`:204`): exposes `RegressionResultsWrapper` for notebook `summary()`.

**SARIMAX parallels and divergences:**

- **Same:** joblib save/load; re-entrant `fit`; `predict` returns `Series` indexed to `features.index`; `metadata` from config fields; `RuntimeError` before fit; standalone CLI.
- **Different:** SARIMAX holds a `statsmodels.tsa.statespace.sarimax.SARIMAXResults` object (not `RegressionResultsWrapper`). `SARIMAXResults` **is** picklable via joblib (it inherits from statsmodels' `MLEResults` which delegates pickle to `__getstate__`/`__setstate__`). However, fitted state includes a Kalman filter cache; confirm round-trip in a test.
- **Different:** `predict` must pass `exog` (the feature columns) for the test period and specify `start`/`end` or `steps` — the call signature of `SARIMAXResults.get_forecast` / `.predict` differs from `RegressionResultsWrapper.predict`. See Section 3.

---

## 3. Harness assumptions about `predict`

Relevant lines from `src/bristol_ml/evaluation/harness.py:206–259`:

```python
for fold_index, (train_idx, test_idx) in enumerate(
    rolling_origin_split_from_config(len(df), splitter_cfg)
):
    X_train = features_frame.iloc[train_idx]   # line 209
    y_train = target_series.iloc[train_idx]    # line 210
    X_test  = features_frame.iloc[test_idx]    # line 211
    y_test  = target_series.iloc[test_idx]     # line 212

    model.fit(X_train, y_train)                # line 214
    y_pred = model.predict(X_test)             # line 215
```

Then at line 234–237:
```python
y_pred_arr = np.asarray(
    y_pred.to_numpy() if isinstance(y_pred, pd.Series) else y_pred,
    dtype=np.float64,
)
```

**Key facts:**

- The harness calls `model.predict(X_test)` with the **full test DataFrame** — not row-by-row.
- It passes `X_test` as a positional argument; `predict` must accept `(self, features: pd.DataFrame) -> pd.Series`.
- It does **not** pass the training data or any index information to `predict`. SARIMAX's native `get_forecast(steps=N, exog=exog_test)` will work, **but** the returned series must be re-indexed to `features.index` before returning — otherwise the harness's `y_true_arr - y_pred_arr` subtraction is over mismatched indices.
- Returned `pd.Series` must have the same length as `X_test` (`len(test_idx)` rows). Any shape mismatch will produce a silent or noisy broadcasting error in the metric calls.
- `SARIMAXResults.predict` returns **in-sample** predictions; `SARIMAXResults.get_forecast` returns out-of-sample. For rolling-origin CV, every test fold is out-of-sample relative to the training window, so `get_forecast(steps=len(test_idx), exog=X_test_values)` is the correct call.
- The harness does not call `predict` with the training data, so SARIMAX cannot use the "append training data to get forecasts" pattern.

---

## 4. Feature table Stage 7 will consume

**Weather-only schema** (`src/bristol_ml/features/assembler.py:74–105`): 10 columns.

- `timestamp_utc`: `timestamp[us, tz=UTC]` — UTC-aware, strictly monotonic, no NaN.
- `nd_mw`: `int32` — national demand in MW; the target column.
- `tsd_mw`: `int32`.
- `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`: all `float32`.
- `neso_retrieved_at_utc`, `weather_retrieved_at_utc`: provenance timestamps.

**Calendar schema** (`src/bristol_ml/features/assembler.py:118–144`): 55 columns = weather-only (10) + 44 calendar int8 columns + `holidays_retrieved_at_utc`. Calendar columns are one-hots: 23 hour-of-day, 6 day-of-week, 11 month, 4 holiday flags.

**Harness indexing:** `df.set_index("timestamp_utc")` is called in `train.py:216` before the frame reaches `evaluate()`. The index is therefore `DatetimeIndex` tz-aware UTC. **Timestamps are guaranteed regular (hourly, no gaps)** by the assembler's `build()` contract — NaN rows are dropped; forward-fill covers gaps up to `forward_fill_hours` (default 3).

**Target:** `nd_mw`, `int32`. SARIMAX fits on float; cast with `y.astype("float64")`. The target series index is `DatetimeIndex[UTC]`, strictly ascending, frequency effectively 1h (no explicit `freq` attribute is set — the assembler stores UTC timestamps, which on DST-change Sundays are 23 or 25 rows but always 1-hour-apart in UTC). SARIMAX requires a regular frequency; pass `freq="h"` to the `SARIMAX` constructor or the model will warn.

---

## 5. Statsmodels import surface

Current `statsmodels` imports in `src/`:

| File | Import |
|------|--------|
| `models/linear.py:42–44` | `import statsmodels.api as sm` + `from statsmodels.regression.linear_model import RegressionResultsWrapper` |
| `evaluation/plots.py:91` | `from statsmodels.graphics.tsaplots import plot_acf` |

The Stage 2 `lowess` claim in the prompt was checked — **no** `statsmodels.nonparametric` import exists in the current codebase.

**`statsmodels.tsa.statespace.SARIMAX`:** this sub-package (in `statsmodels>=0.14`) does pull in `scipy.optimize` (for the MLE maximisation) and `scipy.linalg` (for the Kalman filter). Both are transitive dependencies of the existing `statsmodels` install — `scipy` is already a statsmodels dependency and is already present in the venv. No new top-level dependency is required.

**Current pin** (`pyproject.toml:21`): `statsmodels>=0.14,<1`.

---

## 6. Config schema and Hydra groups

**Existing model configs** (`conf/_schemas.py:417–472`):

```python
class NaiveConfig(BaseModel):
    type: Literal["naive"] = "naive"
    strategy: Literal["same_hour_yesterday", "same_hour_last_week", "same_hour_same_weekday"]
    target_column: str = "nd_mw"

class LinearConfig(BaseModel):
    type: Literal["linear"] = "linear"
    target_column: str = "nd_mw"
    feature_columns: tuple[str, ...] | None = None
    fit_intercept: bool = True

ModelConfig = NaiveConfig | LinearConfig  # line 472
AppConfig.model: ModelConfig | None = Field(default=None, discriminator="type")
```

**Where `SarimaxConfig` slots in:**

- New sibling: `class SarimaxConfig(BaseModel)` in `conf/_schemas.py`, with `type: Literal["sarimax"] = "sarimax"`.
- Extend the union: `ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig`.
- New file: `conf/model/sarimax.yaml` with `# @package model` header, `type: sarimax`, and ARIMA order fields.
- `conf/config.yaml` defaults list: no edit needed unless Stage 7 wants SARIMAX as the default model (unlikely). To run SARIMAX: `python -m bristol_ml.train model=sarimax`.

**Proposed `SarimaxConfig` fields** (for the plan author to confirm):
- `target_column: str = "nd_mw"`
- `feature_columns: tuple[str, ...] | None = None` (consistent with `LinearConfig`)
- `order: tuple[int, int, int]` — `(p, d, q)` ARIMA order.
- `seasonal_order: tuple[int, int, int, int]` — `(P, D, Q, S)`.
- `trend: str | None = None` — statsmodels `trend` argument (`"n"`, `"c"`, `"t"`, `"ct"`).

**Harness CLI** (`src/bristol_ml/evaluation/harness.py:475–487`) uses a `_build_model_from_config` dispatcher that checks `isinstance(model_cfg, NaiveConfig)` / `isinstance(model_cfg, LinearConfig)`. Stage 7 must add a branch here (or in `train.py`'s `_cli_main`) to instantiate `SarimaxModel`. Both places need updating.

---

## 7. Save/load directory conventions

**Stage 4/5 train CLI** (`src/bristol_ml/train.py`): does **not** call `model.save()` at all. The rolling-origin harness trains and evaluates but does not persist the fitted model. There is no `data/models/` directory convention yet.

**`save_joblib`** (`src/bristol_ml/models/io.py:32–53`): creates the parent directory on write; `.tmp` + `os.replace` atomic idiom. The suffix `.joblib` is conventional but not enforced.

**Registry (Stage 9):** not yet shipped. No strong opinion on path. Safe default for Stage 7: `data/models/sarimax.joblib` or let the caller specify the path. The `SarimaxModel.save(path)` signature takes an explicit `Path`, so the caller owns the directory — consistent with `LinearModel.save`.

**Hazard:** if Stage 7 adds a `python -m bristol_ml.models.sarimax` CLI that calls `save()`, it must not hard-code a path — use a CLI argument. The `linear.py` and `naive.py` standalone CLIs intentionally do **not** call `save()`; they only print config. Stage 7 can follow the same pattern.

---

## 8. Notebook conventions

`docs/stages/README.md:36` shows Stage 7 at `planning` status with no notebook column. The stage table shows:

- Stage 4 and 5 each got a new `notebooks/0N_<slug>.ipynb`.
- Stage 6 appended to `notebooks/04_linear_baseline.ipynb` (D11 "append-only, byte-preserving" policy for that notebook; its own cells were added at the end).

The intent doc (`docs/intent/07-sarimax.md:AC-4`) mandates: "the notebook renders a seasonal decomposition, a fit diagnostic, and a forecast comparison." This implies a new notebook — `notebooks/07_sarimax.ipynb` — following the Stage 5 pattern of owning a new file, not appending to the Stage 4/6 notebook.

---

## 9. Open questions in architecture docs relevant to Stage 7

**`docs/architecture/layers/models.md:118–121` Open questions:**

- *"Per-model CLI parity"* — "whether every family's CLI should converge on a common shape… is up to Stage 7." Stage 7 must decide.
- *"Hyperparameter search composition"* — deferred to Stage 10; not Stage 7's problem.
- *"Cross-version load compatibility"* — deferred to Stage 9; not Stage 7's problem.

**`docs/architecture/layers/evaluation.md:165` Open questions:**

- *"Multi-horizon fold structure"* — "Week-ahead evaluation needs either a horizon column on the metrics output or a separate splitter flavour." The intent doc (`07-sarimax.md`) notes the evaluator "needs to be happy" with multi-step native forecasts but defers refactors. Stage 7 must fit within the existing 24-row test-fold contract, not redesign the harness.

---

## 10. Integration with Stage 6 plot helpers

`src/bristol_ml/evaluation/plots.py` helpers accept plain `pd.Series` / `pd.DataFrame`:

- `acf_residuals(residuals, *, lags=168, ...)` — takes a `pd.Series`. SARIMAX residuals from `results.resid` are a `pd.Series`; feed directly.
- `residuals_vs_time(residuals, ...)` — same.
- `error_heatmap_hour_weekday(residuals, ...)` — requires a `DatetimeIndex` on the residuals series. `results.resid` carries the training index; after harness evaluation, compute `y_test - y_pred` (the harness does this in `predictions_df["error"]` when `return_predictions=True`).
- `forecast_overlay_with_band(actual, point_prediction, per_fold_errors, ...)` — `per_fold_errors` is sourced from `predictions_df["error"]`; no SARIMAX-specific plumbing needed.
- `predicted_vs_actual(y_true, y_pred)` — both plain `pd.Series`.

**No shape mismatch.** All helpers are model-agnostic by design (evaluation `CLAUDE.md:AC-3`). SARIMAX residuals and predictions feed them cleanly provided `predict()` returns a correctly-indexed `pd.Series`.

---

## Relevant file index

| Path | Purpose |
|------|---------|
| `src/bristol_ml/models/protocol.py` | `Model` protocol + `ModelMetadata` re-export |
| `src/bristol_ml/models/io.py` | Atomic joblib save/load helpers |
| `src/bristol_ml/models/linear.py` | Template model implementation |
| `src/bristol_ml/models/naive.py` | Minimal-lines protocol proof |
| `src/bristol_ml/train.py` | CLI orchestrator; `_NamedLinearModel` wrapper; model dispatcher |
| `src/bristol_ml/evaluation/harness.py` | Rolling-origin loop; `predict(X_test)` call site |
| `src/bristol_ml/features/assembler.py` | `WEATHER_VARIABLE_COLUMNS`, `CALENDAR_VARIABLE_COLUMNS`, `OUTPUT_SCHEMA`, `CALENDAR_OUTPUT_SCHEMA` |
| `src/bristol_ml/evaluation/plots.py` | Stage 6 diagnostic helpers (model-agnostic) |
| `conf/_schemas.py` | `NaiveConfig`, `LinearConfig`, `ModelMetadata`, `AppConfig.model` union |
| `conf/model/linear.yaml` | Template Hydra group file |
| `conf/config.yaml` | Defaults list; `model: linear` default |
| `tests/unit/models/test_protocol.py` | Protocol-conformance tests Stage 7 must also pass |
| `tests/unit/models/test_linear.py` | Model test pattern to follow |
| `docs/architecture/layers/models.md` | Layer contract, open questions for Stage 7 |
| `docs/architecture/layers/evaluation.md` | Harness contract, multi-horizon open question |
| `docs/intent/07-sarimax.md` | Stage 7 acceptance criteria (authoritative) |
| `pyproject.toml` | `statsmodels>=0.14,<1` pin |
