# Stage 6 — Enhanced Evaluation: Codebase Map

**Purpose.** Read-only survey of what exists. No proposals. The planner
decides what to build; this document tells the planner what it must conform to.

---

## 1. Entry points and current evaluation surface

All symbols lazy-re-exported from `src/bristol_ml/evaluation/__init__.py`
via `__getattr__` dispatch. No eager imports — `python -m bristol_ml` stays
cheap.

### `bristol_ml.evaluation.splitter`

- `rolling_origin_split(n_rows, *, min_train, test_len, step, gap=0, fixed_window=False) -> Iterator[tuple[np.ndarray, np.ndarray]]` — pure integer-index generator; data-structure-agnostic (AC-4).
- `rolling_origin_split_from_config(n_rows, config: SplitterConfig) -> Iterator[...]` — Pydantic-unpacking sugar over the kernel.

### `bristol_ml.evaluation.metrics`

- `mae(y_true, y_pred) -> float`
- `mape(y_true, y_pred) -> float` — raises on any zero in `y_true` (D8)
- `rmse(y_true, y_pred) -> float`
- `wape(y_true, y_pred) -> float` — raises when `sum(|y_true|) == 0` (D9)
- `METRIC_REGISTRY: dict[str, MetricFn]` — name → callable; keyed by lowercase `__name__`.
- `MetricFn` — type alias `Callable[[ArrayLike, ArrayLike], float]`.

All metrics return `float` (not `np.floating`); inputs are coerced via
`np.asarray(…, dtype="float64")` at `_coerce_and_validate`. Values are
fractions, not percentages.

### `bristol_ml.evaluation.harness`

```python
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
```

**Return shape.** One row per fold; columns (in this exact order, tested in
`test_harness_column_order_and_dtypes`):

| Column | dtype | Description |
|--------|-------|-------------|
| `fold_index` | `int64` | Zero-based fold counter |
| `train_end` | `datetime64[ns, UTC]` | Last timestamp in training slice |
| `test_start` | `datetime64[ns, UTC]` | First timestamp in test slice |
| `test_end` | `datetime64[ns, UTC]` | Last timestamp in test slice |
| `<metric.__name__>` … | `float64` | One column per metric callable |

Timestamps are tz-naive if the input index was tz-naive; UTC-aware if input
was UTC-aware. Column order is `["fold_index", "train_end", "test_start",
"test_end", *(metric.__name__ for metric in metrics)]`.

### `bristol_ml.evaluation.benchmarks`

- `align_half_hourly_to_hourly(df, *, aggregation="mean", value_columns=("demand_forecast_mw", "demand_outturn_mw")) -> pd.DataFrame` — collapses NESO half-hourly to hourly UTC index.
- `compare_on_holdout(models, df, neso_forecast, splitter_cfg, metrics, *, aggregation="mean", target_column="nd_mw", feature_columns=None) -> pd.DataFrame` — indexed by `[*sorted(models), "neso"]`; one float column per metric; values are fold-mean.

### `__all__` (full list)

```python
["METRIC_REGISTRY", "MetricFn", "align_half_hourly_to_hourly",
 "compare_on_holdout", "evaluate", "mae", "mape", "rmse",
 "rolling_origin_split", "rolling_origin_split_from_config", "wape"]
```

---

## 2. Existing plotting idioms across notebooks 01–05

### 2a. By library

**matplotlib only** (no seaborn, no plotly): all five notebooks use only
`matplotlib.pyplot` + `matplotlib.dates` (`mdates`). `statsmodels.nonparametric.smoothers_lowess` is used as a smoothing computation in notebook 02, with matplotlib rendering the result — it is not a plotting library call.

**statsmodels** — used only for `lowess()` computation (notebook 02) and `results.summary()` tabular print (notebooks 04, 05).

### 2b. By chart type and data shape

**Notebook 01 — line + daily-peak line**

- Cell 4: line plot of hourly `nd_mw` resampled from half-hourly, over one week. Consumes `pd.Series` with `DatetimeIndex` (UTC). `ax.plot(hourly.index, hourly.values)`. No `mdates` formatter; plain UTC tick labels.
- Cell 6: line plot of `daily_peak = df["nd_mw"].resample("D").max()` over the full window. Same shape. No formatter.

**Notebook 02 — scatter + LOWESS overlay**

- Cell 6: scatter of temperature vs demand (`ax.scatter(temp, nd_mw, s=2, alpha=0.15)`), LOWESS curve (`ax.plot(smooth[:,0], smooth[:,1])`), two `axvline` reference lines at CIBSE thresholds. Consumes two aligned `np.ndarray` columns from a joined DataFrame. Key: sample cap (20 000 rows), `random_state=0`. Reusable idiom: scatter + smooth curve + vertical reference.

**Notebook 03 — fold-boundary shading + multi-fold span**

- Cell (fold overview): full-year `nd_mw` line (`linewidth=0.4`), `axvspan` for the first fold's training window, two `axvline` fold-origin markers. Consumes `display_ts` (`features["timestamp_utc"].dt.tz_convert("Europe/London")`), `features["nd_mw"]`. `mdates.MonthLocator(interval=2)` + `mdates.DateFormatter("%b %Y")`.
- Cell (fold anatomy): zoomed 7-fold view. Line plot over a masked slice; `axvspan` per fold from `display_ts.iloc[test[0]]` to `display_ts.iloc[test[-1]]`. `mdates.DateFormatter("%d %b\n%H:%M")`.

**Notebook 04 — residual-vs-time + 48-hour forecast overlay**

- Cell (residuals top panel): line plot of `residuals = features["nd_mw"] - linear.results.fittedvalues` against `display_ts` (`features.index.tz_convert("Europe/London")`). `linewidth=0.4`, `color="C1"`, `axhline(0.0)`. `mdates.MonthLocator(interval=2)` + `mdates.DateFormatter("%b %Y")`. Consumes `linear.results.fittedvalues` (statsmodels `RegressionResultsWrapper` attribute; a `pd.Series` indexed by the training DataFrame's index).
- Cell (48 h overlay bottom panel): three-line overlay (`ax.plot`) of actual, naive prediction, and linear prediction over a 48-hour window. Optional fourth line for NESO if cache is warm. Consumes `window["nd_mw"]` (actual), `naive.predict(window)` (`pd.Series`), `linear.predict(window)` (`pd.Series`), `neso_slice["demand_forecast_mw"]` (`pd.Series`). `mdates.DateFormatter("%d %b\n%H:%M")`. Legend at lower right.

**Notebook 05 — dual residual comparison + (same 48-h idiom as 04)**

- Cell (residual comparison): two residual lines (`weather_only` and `weather_calendar`) on a single axis, one test week. Consumes `residual_wonly = window_wonly["nd_mw"] - linear_wonly_full.predict(window_wonly)` and `residual_wcal`. Both are `pd.Series` indexed by `window_wcal.index.tz_convert("Europe/London")`. `std()` in legend label. `axhline(0.0)`. `mdates.DateFormatter("%a %d %b")`.

### 2c. Idioms flagged as helper-function candidates

| Idiom | Notebooks | Inputs |
|-------|-----------|--------|
| Residual-vs-time (full window, monthly x-axis) | 04 | `residuals: pd.Series`, UTC/London index, title str |
| Forecast overlay (short window, N models) | 04, 05 | `window: pd.DataFrame`, dict of predictions, `pd.Series` optional NESO |
| Weekly-ripple comparison (two residual series, shared axis) | 05 | `residual_a: pd.Series`, `residual_b: pd.Series`, labels |
| Fold-boundary shading (demand series + axvspan per fold) | 03 | `series: pd.Series`, `folds: list[tuple[np.ndarray, np.ndarray]]` |
| Scatter + LOWESS + vline references | 02 | `x: np.ndarray`, `y: np.ndarray`, threshold values |

All five idioms use the same underlying pattern: matplotlib figure + axes, UTC → `Europe/London` tz_convert on the display axis, `mdates` formatter for the x-axis, `ax.grid(True, alpha=0.3)`, `plt.tight_layout()`, `plt.show()`.

---

## 3. Existing dependency graph

From `pyproject.toml` `[project.dependencies]` and `[dependency-groups].dev`:

### Runtime dependencies (always installed)

| Package | Version pin | Directly reachable for Stage 6 diagnostics? |
|---------|------------|----------------------------------------------|
| `statsmodels>=0.14,<1` | runtime | Yes — `RegressionResultsWrapper` (`.fittedvalues`, `.resid`, `.params`, `.rsquared`, `.summary()`); `statsmodels.graphics.tsaplots.plot_acf` / `plot_pacf` are in the installed package |
| `pandas>=2.2,<3` | runtime | Yes — all notebooks and eval code already use it |
| `numpy` | transitive via pandas/statsmodels | Yes |
| `pyarrow>=16,<22` | runtime | Indirect (storage boundary only) |
| `loguru>=0.7,<1` | runtime | Yes (structured logging) |
| `hydra-core>=1.3,<2` | runtime | Config only |
| `pydantic>=2.7,<3` | runtime | Schema validation only |

### Dev-only dependencies (installed with `--group dev`)

| Package | Version pin | Note |
|---------|------------|------|
| `matplotlib>=3.8,<4` | dev | All five notebooks draw with it; **not a runtime dep** |
| `jupyter>=1,<2` | dev | Notebook execution |
| `nbconvert>=7,<8` | dev | Available for CI notebook smoke tests — but CI does **not** currently use it (see §9) |

**Key finding.** `matplotlib` is a **dev-only** dependency, not a runtime one. A `plots.py` or `diagnostics.py` module inside `bristol_ml.evaluation` would import `matplotlib` — this is currently only valid inside the dev group. If Stage 6 wants the helpers importable in non-dev environments, the planner must decide whether to promote `matplotlib` to a runtime dependency or keep the helpers dev-only.

`statsmodels.graphics.tsaplots` (`plot_acf`, `plot_pacf`) is reachable without any new dependency — statsmodels is already a runtime dep. `statsmodels.nonparametric.smoothers_lowess` is reachable for the same reason.

---

## 4. Stage 4 notebook cell-by-cell structure

File: `/workspace/notebooks/04_linear_baseline.ipynb`

| # | Cell type | Purpose | Output |
|---|-----------|---------|--------|
| 0 | Markdown | Intro: what the notebook does; why losing to NESO is expected | None |
| 1 | Code | `sys.path` bootstrap; imports (`matplotlib.dates`, `matplotlib.pyplot`, `pandas`, `load_config`, `compare_on_holdout`, `evaluate`, `METRIC_REGISTRY`, `assembler`, `neso_forecast_mod`, `LinearModel`, `NaiveModel`); load config with overrides `min_train_periods=720, step=168`; load feature table; print shape | stdout: feature table rows/columns/window |
| 2 | Markdown | Seasonal-naive baseline explanation; `ModelMetadata` rationale | None |
| 3 | Code | Construct `NaiveModel`, fit on full feature table, print `naive.metadata.model_dump_json(indent=2)` | JSON metadata block |
| 4 | Markdown | Linear regression (OLS) rationale; statsmodels choice justification | None |
| 5 | Code | Construct `LinearModel`, fit on full feature table, print `linear.results.summary()` | statsmodels OLS summary table |
| 6 | Markdown | Rolling-origin evaluation explanation; metric fractions convention | None |
| 7 | Code | Get metric names from config; fresh model instances; `evaluate()` both models; print mean per metric; `linear_per_fold.head()` | stdout + DataFrame head |
| 8 | Markdown | Three-way benchmark explanation; D4 aggregation note | None |
| 9 | Code | Check forecast cache; if warm: `compare_on_holdout`; else print "cache not populated" message | Three-way metric table or skip note |
| 10 | Markdown | Residual + 48-h overlay explanation; weekly-ripple hook for Stage 5 | None |
| 11 | Code | **THE MAIN PLOT CELL**: 2-panel `plt.subplots(2, 1, figsize=(10,7))`; top = residual-vs-time (full year) + `axhline(0)`; bottom = 48-h overlay (actual + naive + linear + optional NESO); `mdates` formatters; `legend`; `plt.tight_layout()`; `plt.show()` | Two-panel figure |
| 12 | Markdown | What this sets up; live-demo suggestions (model=naive, strategy=same_hour_yesterday, autumn clock-change window, NESO cache) | None |

Cell 11 is the primary Stage 6 update target. It draws `residuals = features["nd_mw"] - linear.results.fittedvalues` (the full-sample residuals from the Cell 5 global fit, not per-fold residuals). The `display_ts` variable is `features.index.tz_convert("Europe/London")`.

---

## 5. Evaluation layer contract + module docstrings

### From `docs/architecture/layers/evaluation.md`

> The evaluation layer sits between features and models: it is the **library** (per DESIGN §3.2 — explicitly not a service) that every modelling stage reaches into for time-series cross-validation primitives. It owns: splitters — enumerating train/test folds under rolling-origin or related rules. Metric functions (Stage 4+) — pure `(y_true, y_pred) -> float` callables. A fold-level harness (Stage 4+) — the loop that runs a model across folds and collects per-fold metrics. Benchmark comparison (Stage 6+) — reporting a candidate model against the NESO day-ahead forecast.

The layer doc explicitly anticipates Stage 6 under "Stage 6 will add visualisation and richer-diagnostics primitives".

### From `src/bristol_ml/evaluation/CLAUDE.md`

> This module is the **evaluation layer**: splitters, metrics, evaluators, and benchmark comparisons that sit between the features layer (feature tables in) and the models layer (models out). Stage 3 introduced it with the rolling-origin splitter; Stage 4 extends it with metric functions, the fold-level evaluator harness, and the three-way NESO benchmark.

The `evaluation/CLAUDE.md` also notes:

> **Placement note (plan D5).** The benchmark helper lives in `evaluation/` rather than a peer `benchmarks/` module. … a future Stage 6 refactor into a peer module stays cheap because the public surface is two pure functions.

### Open questions from `evaluation.md` that Stage 6 may resolve

1. **`NesoBenchmarkConfig.holdout_start/_end` consumers.** From the layer doc: "Stage 4's `compare_on_holdout` derives the holdout window from fold test-periods and never reads them. A Stage 6 richer-diagnostics flow (fixed-window retrospective comparison outside the rolling-origin harness) is the natural consumer." This field is currently latent — Stage 6 is explicitly named as the consumer.

2. **Benchmark placement.** Resolved at Stage 4 (evaluation-layer sibling). The open question notes Stage 6 could move it; the planner should decide whether Stage 6 triggers that refactor.

3. **Multi-horizon fold structure** (still open — deferred until week-ahead model is in scope).

4. **Registry integration** (still open — deferred to Stage 9).

5. **Drift monitoring placement** (still open — deferred to Stage 18).

---

## 6. Model protocol touchpoints for diagnostics

### `Model` protocol (`src/bristol_ml/models/protocol.py`)

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

`predict` returns `pd.Series` indexed to `features.index`. `metadata` returns a fully-populated `ModelMetadata` after `fit`.

### `LinearModel` — extra surface available to diagnostics

`linear.results` is a `statsmodels.regression.linear_model.RegressionResultsWrapper`. After `fit()` it exposes:

| Attribute | Type | Usage |
|-----------|------|-------|
| `.fittedvalues` | `pd.Series` | In-sample predictions (index = training DataFrame's index) |
| `.resid` | `pd.Series` | In-sample residuals (`y - fittedvalues`; same index) |
| `.params` | `pd.Series` | Coefficient estimates (index = column names incl. `"const"`) |
| `.rsquared` | `float` | R² on training data |
| `.summary()` | `str`-like | Full OLS summary table |
| `.aic`, `.bic` | `float` | Information criteria |

Notebook 04 uses `linear.results.fittedvalues` directly (Cell 11: `residuals = features["nd_mw"] - linear.results.fittedvalues`). Notebook 05 calls `linear_wonly_full.predict(window_wonly)` (the `predict` protocol method, not `.results.fittedvalues`) because it needs a subset window.

**`NaiveModel` does not expose `.results`.** It has no statsmodels wrapper; `predict` returns a shifted target series. Any diagnostic helper that branches on `hasattr(model, "results")` or accepts `Optional[RegressionResultsWrapper]` will be the pattern the notebooks already use implicitly. A helper that unconditionally calls `model.results` will fail for `NaiveModel`.

---

## 7. Integration points for Stage 6 helpers

### (a) Stage 4 notebook — `notebooks/04_linear_baseline.ipynb`

Cell 11 is the update target. Currently it draws the two panels inline with ~40 lines of matplotlib boilerplate. Stage 6 replaces (or wraps) this with calls to Stage 6 helper functions.

What needs to change in the notebook:
- Import the new helper(s) from `bristol_ml.evaluation.<new_module>` (or whatever the planner names it).
- Replace the inline matplotlib code in Cell 11 with the helper call(s).
- The data already available at Cell 11: `features` (full feature DataFrame, UTC index), `linear` (fitted `LinearModel`), `naive` (fitted `NaiveModel`), `linear_per_fold` and `naive_per_fold` (per-fold metric DataFrames from `evaluate()`), `forecast_cache` / `neso_df_local` (optional, NESO-dependent).
- Cell 9 (`benchmark_table`) is a `pd.DataFrame | None` that a richer-diagnostics helper could also consume for a metric-bar chart or similar.

The notebook must stay under the 120-second wall-clock budget (plan D7 from Stage 4; `step=168` config override already handles the fold-count constraint).

### (b) Future notebooks (Stages 7, 8, 10, 11)

These notebooks will fit models that implement `Model` but may not be `LinearModel`. Constraints on helpers for future-proofing:

- Accept `Model` (the protocol), not `LinearModel` concretely.
- Accept `per_fold: pd.DataFrame` (the `evaluate()` return shape) as the primary diagnostic input — this is the model-agnostic interface.
- Degrade gracefully when `hasattr(model, "results")` is `False` (e.g., tree models, neural nets).
- Accept the `features` DataFrame with a UTC `DatetimeIndex` and handle `tz_convert("Europe/London")` internally for display.
- The planner may want helpers that accept either (a) `model + features` for in-sample residual plots or (b) `per_fold DataFrame` for fold-level metric plots — these are different data shapes.

---

## 8. CLAUDE.md conventions relevant to Stage 6

From `/workspace/CLAUDE.md`:

**Every module runs standalone (§2.1.1):**
> Every module runs standalone via `python -m bristol_ml.<module>` (§2.1.1).

Any `plots.py` or `diagnostics.py` added to `bristol_ml.evaluation` must implement a `_cli_main()` function and `if __name__ == "__main__": raise SystemExit(_cli_main())`. The CLI contract is: prints something useful (e.g. lists available helpers and their signatures) against the resolved Hydra config with warm caches.

**Notebooks are thin (§2.1.8):**
> Notebooks are thin (§2.1.8) — they import from `src/bristol_ml/`, they do not reimplement logic.

The refactored Cell 11 in notebook 04 must import the diagnostic function and call it — not copy-paste the matplotlib boilerplate. No business logic in notebook cells.

**British English in documentation and user-facing strings:**
> British English in documentation and user-facing strings.

Docstrings and axis labels must use British English: "colour" not "color" in docstrings; "visualisation" not "visualization"; though matplotlib keyword arguments use American spellings (`color=`, `facecolor=`) because those are library API names.

**Typed narrow interfaces (§2.1.2):**
> Type hints on all public function signatures; never `# type: ignore` without a comment explaining why (§2.1.2). … Downstream code never sees raw `DictConfig` — convert via `bristol_ml.config.validate` at the CLI boundary and pass the Pydantic model onward.

Every helper function must carry full type annotations on its public signature. Return type for plot helpers is typically `matplotlib.figure.Figure` or `tuple[matplotlib.figure.Figure, ...]`. Accepting `matplotlib.axes.Axes` as an optional parameter (for composability) is the existing notebook pattern's spirit even though no existing production module does it yet.

**Stub-first for expensive/flaky external dependencies (§2.1.3):**
Applies if any helper makes HTTP calls — irrelevant for pure plotting. Does not constrain the diagnostics module.

**Configuration outside code (§2.1.4):**
Any configurable knobs (e.g., figure size, DPI, colour palette) belong in a `conf/evaluation/plots.yaml` if they are user-configurable. Hard-coded figure defaults (`figsize=(10, 7)`) that match the existing notebook convention are acceptable as defaults.

---

## 9. Test conventions

### Directory layout

Test files mirror source:
`tests/unit/evaluation/` ↔ `src/bristol_ml/evaluation/`

Existing test files in `tests/unit/evaluation/`:
- `test_splitter.py` — splitter unit tests and namespace re-export tests.
- `test_metrics.py` — metric function unit tests.
- `test_metrics_cli.py` — standalone CLI smoke test.
- `test_harness.py` — harness unit tests (the most detailed; see §1 for full column-order contract tested here).
- `test_benchmarks.py` — benchmark helper tests.
- `test_init_reexports.py` — `__all__` and lazy `__getattr__` contract.

A Stage 6 `plots.py` or `diagnostics.py` module would correspond to `tests/unit/evaluation/test_plots.py` (or `test_diagnostics.py`).

### Fixture patterns reusable by Stage 6 tests

**`loguru_caplog`** (`tests/conftest.py`): routes loguru records into pytest `caplog`; needed if helpers emit INFO logs. Already used by `test_harness.py`.

**`_make_df(n, tz, seed)`** pattern (defined in `test_harness.py`): builds a synthetic hourly DataFrame with the five weather columns and `nd_mw`. Stage 6 test helpers can replicate this pattern inline (single-file fixture, not shared fixture — that is the existing convention; only repo-wide fixtures go in `conftest.py`).

**`_make_neso_df`** pattern (defined in `test_benchmarks.py`): builds a synthetic NESO half-hourly forecast frame. Reusable as a template.

**`SplitterConfig` direct construction** (seen in `test_harness.py`):
```python
_SPLIT_CFG = SplitterConfig(min_train_periods=200, test_len=48, step=48, gap=0, fixed_window=False)
```
No Hydra round-trip needed for unit tests — the Pydantic model is constructable directly.

### How tests are written (style)

- `np.random.default_rng(seed=42)` for all synthetic data.
- `pytest.approx` for float comparisons.
- Each test docstring cites the plan clause or spec section it guards.
- British English in all docstrings.
- Tests at boundaries, not everywhere (§2.1.7) — test the helper's public surface, not internal implementation.
- No `xfail` without a linked issue; no skipped tests.
- `# type: ignore` only with an explanatory comment.

### Notebook smoke tests

**There are none in CI.** The `.github/workflows/ci.yml` runs only `ruff format --check`, `ruff check`, and `pytest`. `nbconvert` is installed as a dev dependency but is not invoked by CI. There is no `--nbval`, `nbmake`, or `nbconvert --execute` step in the pipeline.

Pre-commit hooks (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`, `ruff-format`, `ruff-check`) do not execute notebooks either.

The notebooks are run manually (or in a dev session). Cell outputs are committed into the `.ipynb` files (the Stage 4 notebook has cell outputs present). This is the current convention; Stage 6 does not need to add a notebook smoke test unless the planner explicitly chooses to.

---

## Cross-references

- `src/bristol_ml/evaluation/__init__.py` — lazy re-export namespace
- `src/bristol_ml/evaluation/harness.py` — `evaluate()` return-type definition
- `src/bristol_ml/evaluation/benchmarks.py` — `compare_on_holdout()` return-type
- `src/bristol_ml/evaluation/CLAUDE.md` — module invariants
- `docs/architecture/layers/evaluation.md` — layer contract; open questions
- `src/bristol_ml/models/protocol.py` — `Model` protocol (five members)
- `src/bristol_ml/models/linear.py` — `LinearModel.results` property
- `src/bristol_ml/models/naive.py` — `NaiveModel` (no `.results`)
- `notebooks/04_linear_baseline.ipynb` — Cell 11: primary notebook update target
- `notebooks/05_calendar_features.ipynb` — Cell 11: weekly-ripple idiom (reusable pattern)
- `tests/unit/evaluation/test_harness.py` — per-fold DataFrame column-order contract test
- `tests/conftest.py` — `loguru_caplog` fixture
- `pyproject.toml` — `matplotlib` is dev-only; `statsmodels` is runtime
