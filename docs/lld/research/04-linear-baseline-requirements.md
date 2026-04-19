# Stage 4 — Linear regression baseline + evaluation harness: requirements

> Produced by `requirements-analyst` during Phase 1 discovery.  Becomes structured input to `docs/plans/active/04-linear-baseline.md`.

## 1. Goal

Land the first working models (seasonal-naive and linear regression), the metric functions, the fold-level evaluation harness, and a three-way CLI comparison against the NESO benchmark, establishing the `Model` protocol that every subsequent modelling stage implements.

## 2. User stories (Given/When/Then)

**US-1 — CLI training and evaluation**
Given a warm feature-table cache (`data/features/weather_only.parquet`) and a resolved `AppConfig` with `model=linear`, when the user runs `python -m bristol_ml train model=linear`, then a metric table (MAE, MAPE, RMSE, WAPE, mean and spread across folds) is printed to stdout and the process exits 0.

**US-2 — Three-way benchmark comparison**
Given the NESO day-ahead forecast archive is fetched and cached, when the user runs the CLI with the benchmark flag (or a dedicated command), then a single metric table comparing seasonal-naive, linear regression, and the NESO forecast on the same held-out period is printed, showing whether the linear model beats, ties, or loses to NESO.

**US-3 — One-word model swap**
Given two YAML group files `conf/model/naive.yaml` and `conf/model/linear.yaml` exist, when the user changes `model=naive` to `model=linear` in the CLI invocation (or `conf/config.yaml` default), then a different model trains and evaluates without any code change, demonstrating the Hydra `_target_` pattern (DESIGN §7.3, §7.5).

**US-4 — Save, reload, identical predictions**
Given a fitted linear model saved to `data/models/linear.joblib` (or equivalent), when the model is loaded via `LinearModel.load(path)` and `predict()` is called on the same feature rows, then the returned `pd.Series` is numerically identical to the predictions made before saving.

**US-5 — Demo notebook**
Given Stages 0–3 caches are warm, when the user runs `notebooks/04_linear_baseline.ipynb` top to bottom, then all cells execute without error, a residual plot and forecast overlay are rendered, and the three-way metric table is displayed, in a total wall-clock time within the notebook runtime budget (see NFR-5).

## 3. Functional requirements

**F-1 — `Model` protocol (DESIGN §7.3)**
Introduce `src/bristol_ml/models/base.py` containing a `Model` `Protocol` (or ABC — see D3) with exactly the surface defined in DESIGN §7.3:

```python
def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
def predict(self, features: pd.DataFrame) -> pd.Series: ...
def save(self, path: Path) -> None: ...
@classmethod
def load(cls, path: Path) -> "Model": ...
@property
def metadata(self) -> ModelMetadata: ...
```

`ModelMetadata` must carry at minimum: model name, feature column names used at fit-time, fit timestamp (UTC), git SHA, and any scalar hyperparameters. It must be serialisable alongside the model artefact and readable without re-fitting.

**F-2 — Seasonal-naive model**
`src/bristol_ml/models/naive.py` implements the `Model` protocol. The concrete naive definition is an open decision (D1). The naive model must train in negligible time (no iterative loop) to prove the protocol works without a real training loop (intent AC-2). `fit()` records which training rows underpin the look-back; `predict()` looks up the appropriate historical hour(s) from the training set.

**F-3 — Linear regression model**
`src/bristol_ml/models/linear.py` implements the `Model` protocol using OLS on the Stage 3 `weather_only` feature table. Estimator library is an open decision (D2); intent leans statsmodels. All weather columns from `assembler.OUTPUT_SCHEMA` are eligible as regressors; the intercept is included. The model must expose fitted coefficients through `metadata` so a meetup facilitator can print them.

**F-4 — Metric functions (DESIGN §5.3)**
`src/bristol_ml/evaluation/metrics.py` exposes pure functions with signatures:

```python
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
```

These are the four metrics named explicitly in DESIGN §5.3. MAPE zero-denominator behaviour and WAPE formula are open decisions (D8, D9). Functions must accept `np.ndarray`, `pd.Series`, and `list[float]` identically, and must reject length-mismatched or NaN-containing inputs with a named `ValueError`.

**F-5 — Fold-level evaluation harness**
`src/bristol_ml/evaluation/harness.py` exposes:

```python
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
) -> pd.DataFrame: ...
```

The harness iterates folds from `rolling_origin_split_from_config`, calls `model.fit(train_features, train_target)` and `model.predict(test_features)` per fold, and returns a long-form `pd.DataFrame` with one row per fold carrying columns: `fold_index`, `train_end`, `test_start`, `test_end`, and one column per named metric. The harness is the only function in the evaluation layer that touches a `Model` instance (evaluation layer contract, `docs/architecture/layers/evaluation.md`).

**F-6 — NESO forecast ingestion extension**
A new function in `src/bristol_ml/ingestion/neso.py` (or a new peer `src/bristol_ml/ingestion/neso_forecast.py`) fetches and caches the NESO Historic Day-Ahead Demand Forecast. Output is parquet under `data/raw/` following ingestion-layer conventions. The NESO forecast is half-hourly; alignment to hourly is an open decision (D4).

**F-7 — Benchmark comparison helper**
`src/bristol_ml/evaluation/benchmarks.py` provides a function that aligns model predictions with NESO forecast and actuals on a held-out period and returns a metric table.  Module placement is an open decision (D5).

**F-8 — Metric table output**
The CLI prints the metric table to stdout in a human-readable tabular format (e.g. `tabulate` or a manual f-string grid). The table must show model name, MAE, MAPE, RMSE, WAPE (mean across folds), and the NESO MAE ratio (DESIGN §5.3). Columns are labelled in British English. No file is written to disk by the print path.

**F-9 — CLI path via Hydra config group**
`conf/model/naive.yaml` and `conf/model/linear.yaml` set `_target_` per the DESIGN §7.5 pattern. `conf/config.yaml` defaults include `model: linear`. Switching to `model=naive` at the CLI requires no code change. A new `conf/model/` group is added; `AppConfig` gains a `model` field (Pydantic schema to be defined — see D3).

**F-10 — Model serialisation**
`save()` and `load()` use joblib (see D6). Artefacts write to a configurable path (e.g. `data/models/<name>.joblib`). Reloading a saved model and predicting on the same inputs produces a numerically identical `pd.Series` (intent AC-3).

**F-11 — Standalone module CLIs**
`python -m bristol_ml.models.naive`, `python -m bristol_ml.models.linear`, `python -m bristol_ml.evaluation.metrics`, and `python -m bristol_ml.evaluation.harness` each exit 0 with a help string (DESIGN §2.1.1). A training/evaluation CLI under `python -m bristol_ml train` (or a new `src/bristol_ml/train.py`) wires the full pipeline.

**F-12 — Demo notebook**
`notebooks/04_linear_baseline.ipynb` is thin (DESIGN §2.1.8): imports from `src/bristol_ml/`, no reimplemented logic. Cells: load feature table, define a reduced-fold config (see D7), fit and evaluate both models, plot residuals and a 48-hour forecast overlay, print the three-way metric table. A markdown cell explains whether the linear model beats the NESO benchmark and why that outcome is pedagogically useful regardless of direction.

**F-13 — Protocol-conformance tests**
A test for each model (`tests/unit/models/test_naive.py`, `tests/unit/models/test_linear.py`) verifies: `fit()` runs without error on a small synthetic feature frame; `predict()` returns a `pd.Series` of the same length as the test input; `save()` writes a file to `tmp_path`; `load()` of that file returns identical predictions; `metadata` is a populated `ModelMetadata` instance. These tests are spec-derived and exist independently of the implementation.

**F-14 — Per-metric unit tests**
`tests/unit/evaluation/test_metrics.py` verifies each of MAE, MAPE, RMSE, WAPE against hand-computed fixtures (small arrays where the expected value is derivable by inspection). Tests cover: perfect predictions (all metrics zero), constant offset, MAPE zero-denominator guard, WAPE formula edge case.

## 4. Non-functional requirements

**NFR-1 — British English** in all docstrings, log messages, CLI help text, and notebook prose.

**NFR-2 — Type hints and no silent ignores.** All public function signatures carry type hints. Any `# type: ignore` comment must include an inline explanation of why it cannot be avoided (DESIGN §2.1.2). Mypy (if wired in CI) and ruff must pass clean.

**NFR-3 — Parquet at storage boundaries.** The NESO forecast cache writes parquet with a declared `OUTPUT_SCHEMA` (ingestion-layer convention). Model artefacts are serialised via joblib (D6). Whether fold-level prediction outputs are written as parquet for downstream Stage 6 diagnostics is an open decision (D10).

**NFR-4 — Reproducibility.** `cfg.project.seed` must be passed to any stochastic component. OLS is deterministic; the seasonal-naive has no randomness. The seed exists for future stochastic models sharing this interface.  Models trained on identical data with the same config must produce identical artefacts.

**NFR-5 — Notebook runtime budget.** The notebook must run top-to-bottom in under 120 s on a laptop with warm caches (matching Stage 3 D7, per DESIGN §11 OQ-1). A reduced-fold override (D7) is the mechanism; the full-fold mode is reserved for the CLI path.

**NFR-6 — Loguru house style.** INFO-level narration for run-level events (model name, fold count, total elapsed time, metric table). DEBUG-level detail for per-fold events. No `print()` in library code; only CLI and notebook output use `print()`.

**NFR-7 — Pre-commit and lint gates.** All new code passes `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pre-commit run --all-files` before the stage is marked complete.

**NFR-8 — DESIGN §2.1.6 provenance.** `ModelMetadata` records fit timestamp (UTC), git SHA at fit time, and the feature column names used.  Every persisted artefact is traceable to its inputs and code state.

## 5. Acceptance criteria

| # | Intent AC | Satisfied by |
|---|-----------|-------------|
| AC-1 | Both models train, evaluate, and print a metric table from the CLI. | F-3, F-4, F-5, F-8, F-9, F-11 |
| AC-2 | The model interface is implementable in very few lines — the naive model proves this. | F-1, F-2 |
| AC-3 | Saving a fitted model and reloading it produces identical predictions. | F-10, F-13 |
| AC-4 | Metric functions produce mathematically correct values on hand-computed fixtures. | F-4, F-14 |
| AC-5 | The benchmark comparison produces a three-way metric table on the held-out period. | F-6, F-7, F-8 |
| AC-6 | The notebook runs top-to-bottom in a reasonable time on a laptop. | F-12, NFR-5 (ceiling: 120 s) |
| AC-7 | A protocol-conformance test exists for both models; metric functions have their own unit tests. | F-13, F-14 |

**Implicit ACs from DESIGN §9 (stage DoD checklist):**

| # | DoD requirement | Notes |
|---|-----------------|-------|
| AC-8 | CI green, all tests pass, no `xfail`, no skips. | Standard gate. |
| AC-9 | Module CLAUDE.md added or updated for every touched module. | `models/`, `evaluation/` both touched. |
| AC-10 | `README.md` updated with any new entry point. | `python -m bristol_ml train` is new. |
| AC-11 | `docs/lld/stages/04-linear-baseline.md` retrospective filed. | Stage hygiene. |
| AC-12 | `CHANGELOG.md` entry under `[Unreleased]`. | Stage hygiene. |
| AC-13 | Notebook demonstrates the output. | F-12. |
| AC-14 | DESIGN §6 layout updated to reflect new modules. | Deny-tier for lead; human approval required. |

## 6. Open questions / decisions the plan must pin down

**D1 — Seasonal-naive definition.**  Options: same-hour-yesterday (`y_{t-24}`), same-hour-last-week (`y_{t-168}`), same-hour-same-weekday-most-recent.  The latter is marginally more complex without material gain over `y_{t-168}`.  Recommendation: `y_{t-168}` (captures dominant weekly seasonality; credible-but-beatable floor).  Encode as `Literal["same_hour_yesterday", "same_hour_last_week", "same_hour_same_weekday"]` on `NaiveConfig.strategy`.

**D2 — OLS library.**  statsmodels vs scikit-learn.  statsmodels gives `.summary()`, coefficients with SE/t/p, and native `.get_prediction()` for intervals.  DESIGN §8 already names statsmodels.  Recommendation: statsmodels `OLS`.  Wrap in `Model` protocol interface; sklearn-Pipeline compatibility deferred.

**D3 — Protocol vs ABC.**  Intent flags both.  DESIGN §7.3 uses `Protocol` syntax.  Recommendation: `typing.Protocol` decorated `@runtime_checkable`; record as ADR-0003 so later stages know it is decided.

**D4 — NESO half-hourly → hourly alignment.**  Options: mean, sum, take-one.  Sum is unit-wrong for MW.  Recommendation: mean (consistent with Stage 3 D1 which aggregates the demand actuals identically).  Encode as `NesoBenchmarkConfig.aggregation: Literal["mean", "first"]`.

**D5 — Benchmark module placement.**  Intent puts it in scope for Stage 4; evaluation layer doc flags it as open.  Recommendation: `src/bristol_ml/evaluation/benchmarks.py`.  Stage 6 may refactor.

**D6 — Serialisation format.**  joblib vs pickle.  Recommendation: joblib (sklearn ecosystem default, handles numpy efficiently, better security posture than bare pickle).  Note skops upgrade path for Stage 9.

**D7 — Reduced-fold notebook mode.**  Intent flags fold-count runtime cost.  Recommendation: notebook sets `evaluation.rolling_origin.step=168` (52 weekly folds); CLI default remains `step=24` (~335 daily folds).  OLS fit cost is negligible, so 335 folds is fine for CLI; the override preserves notebook narrative pacing.

**D8 — MAPE zero-denominator behaviour.**  GB demand never approaches zero in practice.  Recommendation: raise `ValueError("MAPE is undefined when y_true contains zeros")` rather than silently returning `inf` or dropping rows.  Document in docstring.

**D9 — WAPE formula.**  Kolassa & Schütz (2007) / Hyndman form: `Σ|y − ŷ| / Σ|y|`.  Recommendation: implement as stated, guard against `Σ|y| == 0` with `ValueError`.  Document formula in docstring.

**D10 — Fold-level prediction artefacts.**  Whether to persist per-fold predictions as parquet for Stage 6.  Recommendation: do not persist at Stage 4; the harness returns metric DataFrame in memory.  Add a `predictions_path: Path | None = None` kwarg later without breaking callers.  Flag as a seam in harness docstring.

**D11 — Whether beating NESO is a stage goal.**  Intent explicit: losing cleanly has pedagogical value.  Recommendation: beating NESO is NOT a success criterion.  Notebook markdown frames expected loss as correct outcome and names Stage 5 as the intervention that will close the gap.  No CI assertion on relative performance.

## 7. Stage 3 housekeeping to fold in

Three items deferred from Stage 3 Phase 3 review must land in Stage 4 as a housekeeping sub-task:

**H-1 — UTC validation in the evaluation harness (R3).**  `rolling_origin_split_from_config` receives only `n_rows`, so UTC validation cannot live there.  The harness (`evaluate`) receives the full DataFrame and must assert `df.index.tz == ZoneInfo("UTC")` (or the project's canonical UTC reference), raising `ValueError` on non-UTC tz-aware input.  Add a unit test in `tests/unit/evaluation/test_harness.py`.

**H-2 — `SplitterConfig` field comments (R5).**  `min_train_periods`, `test_len`, `step` in `conf/_schemas.py` have `Field(ge=1)` validators without explanatory comments.  Add one-line British-English comments per field, consistent with surrounding style.

**H-3 — `evaluation/__init__.__all__` re-export (D1).**  Add `rolling_origin_split_from_config` to `__init__.__all__` and the lazy `__getattr__` loader; add a test that `from bristol_ml.evaluation import rolling_origin_split_from_config` resolves.

These three items form the first commit (T0) of Stage 4 to keep them clearly separated from the main implementation.

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `statsmodels` absent from `pyproject.toml`; adding it triggers `uv lock` churn mid-stage | Low | Low | Codebase-explorer confirmed it is already declared (line 22 of `pyproject.toml`). |
| Seasonal-naive look-back reaches outside training window on short fixtures (<168 rows) | Medium | Medium | Guard in `predict()`: raise if any required look-back row is absent; add unit test on a 48-row fixture. |
| NESO forecast archive has a different schema than actuals fetcher; ingestion extension breaks module boundary | Medium | High | Research found two candidate resources (`9847e7bb...` cardinal-point 2018+, `08e41551...` half-hourly 2021+).  Plan must pick one and enumerate schema before T-impl. |
| Harness re-fits in place; save/load round-trip test flakes because subsequent `fit()` mutates state | Low | Medium | Specify `fit()` must be re-entrant; `save()` captures most recent fit.  Test explicitly in F-13. |
| Notebook exceeds 120 s on a slow laptop even with weekly-step override | Low | Low | Set `min_train_periods=720` in notebook as fallback; document as demo artefact. |
| `ModelMetadata` over-specified; conflicts with Stage 9 registry design | Medium | Medium | Keep minimal (`name`, `columns`, `fit_utc`, `git_sha`, `hyperparameters: dict[str, Any]`).  Dict field absorbs future extensions. |
| `Model` name collision with Pydantic `BaseModel` in import surface | Low | Low | If collisions surface, rename internal `ModelProtocol`; export as `Model` in `__all__`. |
| NESO forecast resource has pre-2021 coverage gap | Medium | Medium | Restrict three-way comparison to 2021–present test window; flag gap in notebook. |

## Source files cited

- `/workspace/docs/intent/04-linear-baseline.md` — stage intent
- `/workspace/docs/intent/DESIGN.md` — §§2.1, 5.3, 7.3, 8, 9
- `/workspace/docs/architecture/layers/evaluation.md` — evaluation layer contract
- `/workspace/docs/architecture/layers/features.md` — features layer contract
- `/workspace/docs/plans/completed/03-feature-assembler.md` — Stage 3 decisions D1–D8
- `/workspace/docs/lld/stages/03-feature-assembler.md` — Stage 3 retrospective
- `/workspace/src/bristol_ml/evaluation/splitter.py` — `rolling_origin_split_from_config` signature
- `/workspace/src/bristol_ml/evaluation/__init__.py` — missing re-export (H-3)
- `/workspace/conf/_schemas.py` — `SplitterConfig` fields (H-2), `AppConfig` to extend
- `/workspace/src/bristol_ml/features/assembler.py` — `OUTPUT_SCHEMA` and column list
