# Plan — Stage 4: Linear regression baseline + evaluation harness

**Status:** `shipped` — all eleven tasks (T0-T10) landed; all decisions (D1-D11) and housekeeping carry-overs (H-1-H-3) accepted by human 2026-04-19; retrospective at [`docs/lld/stages/04-linear-baseline.md`](../../lld/stages/04-linear-baseline.md).
**Intent:** [`docs/intent/04-linear-baseline.md`](../../intent/04-linear-baseline.md)
**Upstream stages shipped:** Stage 0 (foundation), Stage 1 (NESO demand), Stage 2 (weather + national aggregate), Stage 3 (feature assembler + splitter).
**Downstream consumers:** every subsequent modelling stage (5, 7, 8, 10, 11) implements the `Model` protocol established here. Stage 9 (registry) retrofits these models. Stage 12 (serving) loads them.
**Baseline SHA:** `bd66ebc` (tip of `reclaude` at plan time).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/04-linear-baseline-requirements.md`](../../lld/research/04-linear-baseline-requirements.md)
- Codebase map — [`docs/lld/research/04-linear-baseline-codebase-map.md`](../../lld/research/04-linear-baseline-codebase-map.md)
- External research — [`docs/lld/research/04-linear-baseline-external-research.md`](../../lld/research/04-linear-baseline-external-research.md)

**Architectural weight.** This stage defines the `Model` protocol every future modelling stage conforms to, the metric formulae every future model is judged by, and the rolling-origin evaluation convention used across the project. Decisions here are load-bearing downstream; over-correcting later will mean retrofitting every model.

---

## 1. Decisions for the human (resolve before Phase 2)

Eleven decision points plus three housekeeping carry-overs. For each I propose a default that honours the simplicity bias and cite the evidence. Mark each `ACCEPT` / `OVERRIDE: <alt>` in your reply; I'll update the plan before Phase 2.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Seasonal-naive definition | **`y_{t-168}`** (same hour last week). Encode as `Literal["same_hour_yesterday", "same_hour_last_week", "same_hour_same_weekday"]` on `NaiveConfig.strategy`; default `"same_hour_last_week"`. | Captures dominant weekly seasonality; simple one-index subtraction; credible floor that a temperature-only OLS can still beat — so Stage 5's calendar-feature story still lands. Same-hour-yesterday is too easy; same-weekday-most-recent adds logic without material gain. | External research R3; Ziel & Weron 2018; Dancker 2023. |
| **D2** | OLS library | **statsmodels** `OLS`. sklearn is not a declared dependency; do not introduce it. | `pyproject.toml` already ships `statsmodels>=0.14,<1` (line 22). `.summary()` is a ready-made notebook demo artefact: coefficients, SEs, t-stats, R², AIC/BIC. DESIGN §8 mandates statsmodels. | Codebase map §C, `pyproject.toml:22`; external research R4; DESIGN §8. |
| **D3** | `Model` interface mechanism | **`typing.Protocol`** decorated `@runtime_checkable`. Record as **ADR-0003**. | Matches the DESIGN §7.3 sketch verbatim; avoids forcing all future models to inherit a base; `isinstance(m, Model)` still works for the protocol-conformance test (AC-7). The caveat that `@runtime_checkable` only checks attribute presence (not signatures) is itself a worthwhile teaching moment. | External research R1; DESIGN §7.3 sketch; PEP 544. |
| **D4** | NESO forecast half-hourly → hourly alignment | **`mean`** of the two half-hours per UTC hour. Encode as `Literal["mean", "first"]` on `NesoBenchmarkConfig.aggregation`; default `"mean"`. Omit `sum` (unit-wrong for MW). | NESO ND is a power rate (MW), average over the settlement period. Mean preserves the MW scale and matches Stage 3 D1 (the assembler aggregates ND outturn identically). Sum would produce MWh. Take-one aliases. | External research R7; Stage 3 D1; ScienceDirect 2021. |
| **D5** | Benchmark module placement | **`src/bristol_ml/evaluation/benchmarks.py`**. | Intent scopes it here; evaluation layer doc accepts it. A future Stage 6 refactor into a peer `benchmarks/` module is cheap. Record the placement as a note in the evaluation layer doc, not an ADR. | Intent §Scope; `docs/architecture/layers/evaluation.md` §Open questions. |
| **D6** | Model serialisation format | **joblib** (`joblib.dump` / `joblib.load`) for any numpy-heavy objects; wrap statsmodels `OLSResults` using `joblib` to keep one call path. Add a code comment in `models/io.py` naming **skops** as the Stage 9+ upgrade path. | Intent says "joblib/pickle is fine". sklearn ecosystem default; handles numpy efficiently; simpler than pickle for our purposes. skops adds audit-step friction disproportionate to the stage's demo focus. | Intent §Points for consideration; external research R8. |
| **D7** | Notebook fold count vs CLI fold count | **CLI**: `step=24` (daily) — ~335 folds over a year; full evaluation. **Notebook**: in-cell Hydra override `evaluation.rolling_origin.step=168` (weekly, ~52 folds) for demo pacing. Both run under the 120 s notebook budget; this override is about *narrative* pacing, not runtime. | OLS fit at n=8760, p=4 is <1 ms; 335 folds is trivially fast. The override makes the per-fold table a manageable size to display in a meetup. Stage 3 established the 120 s ceiling (D7 Stage 3). | External research R9; Stage 3 D7. |
| **D8** | MAPE zero-denominator policy | **Raise** `ValueError("MAPE is undefined when y_true contains zeros")`. GB demand never approaches zero in practice, so this guard should never fire for in-scope data; it protects callers from silent `inf` poisoning. Document in docstring. | External research R2 (Amperon, Hyndman); DESIGN §5.3. |
| **D9** | WAPE formula | **`Σ|y_true − y_pred| / Σ|y_true|`** (Kolassa & Schütz 2007 / Hyndman form). Guard against `Σ|y_true| == 0` with `ValueError`. Document formula verbatim in docstring and a one-line inline comment. | Standard industry form; matches Amazon Forecast, Hyndman, Wikipedia WMAPE; distinct from `mean(|err|/|y|)` MAPE-mean. | External research R2. |
| **D10** | Fold-level prediction persistence | **Do not persist** per-fold `y_pred` at Stage 4. Harness returns only a metric DataFrame in memory. Document a seam: a future `predictions_path: Path \| None = None` kwarg on `evaluate()` will land in Stage 6. | Intent is silent; avoid premature storage contract. Stage 6 (richer diagnostics) is the correct inflection point. | Intent §Out of scope; evaluation layer doc. |
| **D11** | "Beat NESO" as success criterion | **No** — not a Stage 4 goal. Notebook markdown frames expected loss to NESO as the correct outcome and explicitly names Stage 5 calendar features as the intervention that will close the gap. No CI or test assertion on relative performance. | Intent explicit: "there is an argument that [beating NESO] should not be [a goal] — that the linear baseline is meant to lose cleanly, so Stage 5's without/with comparison lands with force." | Intent §Points for consideration item 1. |

**Blocking note:** D1, D2, and D3 are load-bearing. D2 governs whether `pyproject.toml` stays untouched (it does, under the proposal); D3 governs the interface every subsequent modelling stage implements; D1 governs the quality of the baseline against which every future improvement is judged. Please resolve all three before Phase 2.

### Housekeeping carry-overs from Stage 3 Phase 3 review

Three items deferred in the Stage 3 review must land in Stage 4 as **Task T0**:

| # | Item | Resolution |
|---|---|---|
| **H-1** | R3: `rolling_origin_split_from_config` does not validate non-UTC tz-aware indices. | Check lives in the harness (`evaluate`), not the splitter — splitter receives only `n_rows`, so UTC check cannot be there without breaking its data-structure-agnosticism. Raise `ValueError` on non-UTC tz-aware index in `evaluate()`. Test in `tests/unit/evaluation/test_harness.py`. |
| **H-2** | R5: `SplitterConfig` fields without defaults lack explanatory comments. | Add one-line British-English comments above `min_train_periods`, `test_len`, `step` in `conf/_schemas.py`. Python comments, not `Field(description=...)` (avoids ABI churn on `frozen=True` model). |
| **H-3** | D1: `evaluation/__init__.__all__` missing `rolling_origin_split_from_config`. | Add to `__all__` and the lazy `__getattr__` loader. Add a test that the symbol imports from the module namespace. |

---

## 2. Scope

### In scope

- A new `src/bristol_ml/models/` module with `protocol.py` (the `Model` Protocol + `ModelMetadata`), `naive.py` (seasonal-naive), `linear.py` (statsmodels OLS), `io.py` (joblib save/load helper), `__init__.py` (lazy re-exports), and `CLAUDE.md`.
- Extensions to `src/bristol_ml/evaluation/`: `metrics.py` (MAE, MAPE, RMSE, WAPE as pure functions) and `harness.py` (`evaluate(model, df, splitter_cfg, metrics)` fold-level loop). The harness holds the **UTC-index check** from H-1.
- A new `src/bristol_ml/evaluation/benchmarks.py` that aligns NESO day-ahead forecast with model predictions and actuals on a held-out period.
- A new `src/bristol_ml/ingestion/neso_forecast.py` (copy-and-adapt of `neso.py`) that fetches and caches the NESO Day-Ahead Half-Hourly Demand Forecast Performance archive (resource `08e41551-80f8-4e28-a416-ea473a695db9`).
- New Pydantic schemas in `conf/_schemas.py`: `ModelMetadata` (immutable provenance), `NaiveConfig`, `LinearConfig`, `ModelsGroup`, `MetricsConfig`, `NesoBenchmarkConfig`, `NesoForecastIngestionConfig`. `AppConfig` gains a `model: ModelsGroup` field and the `IngestionGroup` gains `neso_forecast`.
- New Hydra config groups: `conf/model/naive.yaml`, `conf/model/linear.yaml`, `conf/ingestion/neso_forecast.yaml`, and `conf/evaluation/metrics.yaml`. `conf/config.yaml` defaults list gains these entries; `model: linear` is the default.
- A new `src/bristol_ml/train.py` (or `models/_cli.py`) that wires `load_config → assembler.load → harness.evaluate → print metric table`, invoked as `python -m bristol_ml.train` (or equivalent standalone module path).
- A new `notebooks/04_linear_baseline.ipynb` demonstrating both models, residuals, a 48-hour forecast overlay, and the three-way NESO benchmark table.
- Unit tests for each metric (hand-computed fixtures), protocol conformance tests for both models, harness tests (UTC guard + fold semantics), and VCR cassette tests for the new NESO forecast ingester.
- Stage-hygiene updates: module CLAUDE.md files, retrospective, CHANGELOG entry, stages index status cell, evaluation + models layer docs.
- **Task T0 (housekeeping carry-overs from Stage 3 review):** H-1 (UTC guard), H-2 (SplitterConfig comments), H-3 (`__all__` re-export).

### Out of scope (do not accidentally implement)

From Intent §Out of scope: calendar features (Stage 5), lag features, model registry (Stage 9), richer diagnostics (Stage 6), hyperparameter search, any model beyond naive and linear.

Also out of scope for this plan:
- `scikit-learn` as a dependency — DESIGN §8 and D2 keep it out until a stage surfaces a clear need.
- `skops` secure serialisation — code comment signposts Stage 9+ adoption (D6).
- Persisting per-fold predictions to disk — D10 defers.
- Pre-2021 NESO forecast coverage — resource `08e41551...` starts Apr 2021; comparison restricted to 2021+.
- Changes to `src/bristol_ml/cli.py` or `__main__.py` — model selection is a config group override, not a subcommand.

---

## 3. Reading order for the implementer

Read top-to-bottom before opening code:

1. `docs/intent/04-linear-baseline.md` — the spec. Where this plan disagrees, the spec wins.
2. `docs/lld/research/04-linear-baseline-requirements.md` — Given/When/Then acceptance criteria and the full F-number / D-number / H-number reference.
3. `docs/lld/research/04-linear-baseline-codebase-map.md` §A (consumable Stage 3 surfaces), §C (patterns to follow), §D (Stage 3 deferred items — the H-1/H-2/H-3 targets), §E (NESO forecast resource), §F (integration points).
4. `docs/lld/research/04-linear-baseline-external-research.md` R2, R3, R7 (rigorous metric + naive + HH→hourly definitions).
5. `docs/intent/DESIGN.md` §2.1 (principles), §5.3 (metrics), §7.3 (Model protocol), §8 (tech choices), §9 (DoD).
6. `docs/architecture/layers/evaluation.md` — contract + extension plan for `metrics.py` and `harness.py`.
7. `docs/plans/completed/03-feature-assembler.md` — D1–D8 tell you the exact shape of the feature table this stage consumes.
8. `src/bristol_ml/features/assembler.py` — `OUTPUT_SCHEMA` is the load-bearing column contract. Code to what's in the file.
9. `src/bristol_ml/evaluation/splitter.py` — `rolling_origin_split_from_config` signature and return shape.
10. `src/bristol_ml/ingestion/neso.py` + `_common.py` — template for `neso_forecast.py`; helpers to reuse verbatim.
11. `conf/_schemas.py` — the `ConfigDict(extra="forbid", frozen=True)` pattern and the `*Group | None = None` idiom to mirror for `ModelsGroup`.

CLAUDE.md + `.claude/playbook/` are read once for process, not per-stage.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

1. **Both models train, evaluate, and print a metric table from the CLI.** *(Satisfied by F-3, F-4, F-5, F-8, F-9, F-11; tasks T3, T4, T5, T8.)*
2. **The model interface is implementable in very few lines of code** — the naive model proves this. *(F-1, F-2; tasks T2, T3.)*
3. **Saving a fitted model and reloading it produces identical predictions.** *(F-10; T3, T4, T6.)*
4. **Metric functions produce mathematically correct values on hand-computed fixtures.** *(F-4, F-14; T5.)*
5. **The benchmark comparison produces a three-way metric table on the held-out period.** *(F-6, F-7, F-8; T7, T8.)*
6. **The notebook runs top-to-bottom in a reasonable time on a laptop.** *(F-12, NFR-5; T9. Concrete ceiling per D7: 120 s.)*
7. **A protocol-conformance test exists for both models; metric functions have their own unit tests.** *(F-13, F-14; T3, T4, T5.)*

Implicit DoD ACs (DESIGN §9): CI green, module CLAUDE.md updated, README entry-point listed, retrospective filed, CHANGELOG entry, notebook demonstrates the output. §6 repo-layout tree update is deny-tier for the lead; surface to the human if structural changes demand it.

Mapping of ACs to tasks is in §6.

---

## 5. Architecture summary (no surprises)

Data flow — end-to-end for the `train` action:

```
load_config() → AppConfig
├── .features.weather_only        → assembler.load(path) → pd.DataFrame (OUTPUT_SCHEMA)
├── .evaluation.rolling_origin    → (passed to harness)
├── .evaluation.metrics           → list[MetricFn]
├── .model.linear / .model.naive  → Model instance via Hydra _target_
└── (optional) .ingestion.neso_forecast → neso_forecast.fetch/load for benchmark

harness.evaluate(model, df, splitter_cfg, metrics) →
    for train_idx, test_idx in rolling_origin_split_from_config(len(df), splitter_cfg):
        model.fit(df.iloc[train_idx][features], df.iloc[train_idx]["nd_mw"])
        y_pred = model.predict(df.iloc[test_idx][features])
        record per-fold metrics → pd.DataFrame

benchmarks.compare(linear_preds, naive_preds, neso_forecast, actuals) →
    three-way pd.DataFrame (rows: model, cols: metric)

train._cli_main() → print both tables to stdout
```

Public API surface (Stage 4 adds only these):

```python
# models/protocol.py
@runtime_checkable
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
    @property
    def metadata(self) -> ModelMetadata: ...

class ModelMetadata(BaseModel):
    name: str
    feature_columns: tuple[str, ...]
    fit_utc: datetime | None
    git_sha: str | None
    hyperparameters: dict[str, Any]

# models/naive.py
class NaiveModel:
    def __init__(self, config: NaiveConfig): ...
    # implements Model

# models/linear.py
class LinearModel:
    def __init__(self, config: LinearConfig): ...
    # implements Model

# models/io.py
def save_joblib(obj: Any, path: Path) -> None: ...
def load_joblib(path: Path) -> Any: ...

# evaluation/metrics.py
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...

# evaluation/harness.py
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame: ...

# evaluation/benchmarks.py
def compare_on_holdout(
    models: Mapping[str, Model],
    df: pd.DataFrame,
    neso_forecast: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
) -> pd.DataFrame: ...

# ingestion/neso_forecast.py
OUTPUT_SCHEMA: pa.Schema
def fetch(config: NesoForecastIngestionConfig, cache: CachePolicy = OFFLINE) -> Path: ...
def load(path: Path) -> pd.DataFrame: ...
```

No change to `src/bristol_ml/cli.py`, `__main__.py`, or `load_config()` signature. The `train` action is a new standalone module entry point.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T0 — Stage 3 review housekeeping
*(Independent of everything else; first commit to keep carry-overs cleanly separated.)*

- [ ] H-2: add one-line British-English comments above `min_train_periods`, `test_len`, `step` in `conf/_schemas.py::SplitterConfig`. Example style: `# Minimum training rows before the first test origin — e.g. 8760 = one year of hourly data.`
- [ ] H-3: add `rolling_origin_split_from_config` to `evaluation/__init__.__all__` and the lazy `__getattr__` loader.
- [ ] Test: extend `tests/unit/evaluation/test_splitter.py` (or add a new `test_evaluation_init.py`) with `test_rolling_origin_split_from_config_importable_from_namespace`.
- H-1 is NOT in T0 — it lives in T6 (harness implementation) because `evaluate()` does not exist until then.
- **Acceptance:** contributes to AC-8 (CI green; review carry-overs closed).
- **Command:** `uv run pytest tests/unit/evaluation/ -q && uv run pytest tests/unit/test_config.py -q`.

### Task T1 — Config schemas and Hydra groups
*(Unblocks T2–T9; no downstream data dependency.)*

- [ ] Add to `conf/_schemas.py`:
  - `ModelMetadata` — frozen immutable provenance (not a group; used as a field on fitted models).  Fields: `name: str`, `feature_columns: tuple[str, ...]`, `fit_utc: datetime | None`, `git_sha: str | None`, `hyperparameters: dict[str, Any] = {}`.
  - `NaiveConfig` — `strategy: Literal["same_hour_yesterday", "same_hour_last_week", "same_hour_same_weekday"] = "same_hour_last_week"` (per **D1**), `target_column: str = "nd_mw"`.
  - `LinearConfig` — `target_column: str = "nd_mw"`, `feature_columns: tuple[str, ...] | None = None` (None means "all weather columns from assembler schema"), `fit_intercept: bool = True`.
  - `ModelsGroup` — `naive: NaiveConfig | None = None`, `linear: LinearConfig | None = None`.  `AppConfig.model: ModelsGroup = Field(default_factory=ModelsGroup)`.
  - `MetricsConfig` — `names: tuple[Literal["mae", "mape", "rmse", "wape"], ...] = ("mae", "mape", "rmse", "wape")`.  Extend `EvaluationGroup` with `metrics: MetricsConfig | None = None`.
  - `NesoForecastIngestionConfig` — mirrors `NesoIngestionConfig` structurally (same rate-limit / retry / cache fields) per codebase map §E.  Add `resource_id: str` field pinned to `"08e41551-80f8-4e28-a416-ea473a695db9"` (per **D4** research R6).
  - `NesoBenchmarkConfig` — `aggregation: Literal["mean", "first"] = "mean"` (per **D4**), `holdout_start: datetime`, `holdout_end: datetime`.  Consider placement under `EvaluationGroup.benchmark` or a new `BenchmarksGroup` — decide at implementation time.
  - Extend `IngestionGroup` with `neso_forecast: NesoForecastIngestionConfig | None = None`.
- [ ] Create `conf/model/naive.yaml` with `# @package model.naive`, `_target_: bristol_ml.models.naive.NaiveModel`, and the strategy field. Use the Hydra `_target_` pattern so `model=naive` instantiates the class directly.
- [ ] Create `conf/model/linear.yaml` — same shape, `_target_: bristol_ml.models.linear.LinearModel`.
- [ ] Create `conf/ingestion/neso_forecast.yaml` with `# @package ingestion.neso_forecast`, same shape as `conf/ingestion/neso.yaml` with the pinned resource ID.
- [ ] Create `conf/evaluation/metrics.yaml` with `# @package evaluation.metrics` — default names list.
- [ ] Create `conf/evaluation/benchmark.yaml` with `# @package evaluation.benchmark` — default aggregation + holdout window (start=`2023-01-01T00:00Z`, end=`2023-12-31T23:59Z`, to match the shipped weather fixture).
- [ ] Add the new entries to `conf/config.yaml` `defaults:` list. **Default `model: linear` per intent "Demo moment".**
- **Acceptance:** contributes to AC-1, AC-9, AC-14 (test below).
- **Tests (spec-derived, written by `@test-author`):**
  - `test_app_config_populates_models_linear_from_defaults` — `AppConfig.model.linear` is a populated `LinearConfig` after defaults-only `load_config()`.
  - `test_model_swap_via_override` — `load_config(overrides=["model=naive"])` yields `AppConfig.model.naive` populated and `AppConfig.model.linear is None` (or both populated — depending on the `model:` group wiring; document which).
  - `test_naive_config_rejects_unknown_strategy` — Literal narrowing.
  - `test_linear_config_rejects_extra_keys` — `extra="forbid"`.
  - `test_metrics_config_defaults` — all four metric names present by default.
- **Command:** `uv run pytest tests/unit/test_config.py -q`.

### Task T2 — `Model` protocol + `ModelMetadata` + io helpers
*(Depends on T1 for `ModelMetadata` Pydantic model; no data dependency.)*

- [ ] Create `src/bristol_ml/models/__init__.py` with lazy re-exports.
- [ ] Create `src/bristol_ml/models/protocol.py`:
  - `@runtime_checkable class Model(Protocol)` matching the DESIGN §7.3 signature (per **D3**).
  - Re-export `ModelMetadata` from `conf._schemas` for convenience.
  - Docstring explains the caveat: `isinstance(m, Model)` only checks attribute presence, not signatures.
- [ ] Create `src/bristol_ml/models/io.py`:
  - `save_joblib(obj: Any, path: Path) -> None` — writes atomically via `path.with_suffix(path.suffix + ".tmp")` + `os.replace` (mirrors `ingestion._common._atomic_write` idiom).
  - `load_joblib(path: Path) -> Any`.
  - One-line code comment: `# Upgrade path: skops.io for secure artefacts once Stage 9 registry lands.`
- [ ] Create `src/bristol_ml/models/CLAUDE.md` — layer surface + invariants + standalone CLI note + cross-refs (template: `src/bristol_ml/evaluation/CLAUDE.md`).
- **Acceptance:** AC-2 (interface implementable in very few lines).
- **Tests (spec-derived):**
  - `test_model_protocol_is_runtime_checkable` — dummy class with all five methods passes `isinstance(x, Model)`; class missing one fails.
  - `test_joblib_round_trip_atomic` — writes a dict, checks no `.tmp` sibling, loads and compares.
- **Command:** `uv run pytest tests/unit/models/ -q`.

### Task T3 — Seasonal-naive model
*(Depends on T1 for `NaiveConfig`; T2 for `Model` protocol.)*

- [ ] Create `src/bristol_ml/models/naive.py`:
  - `class NaiveModel` implementing the `Model` protocol.
  - `__init__(self, config: NaiveConfig)` stores config + empty state.
  - `fit(features, target)` — records a mapping from `(hour_of_day, day_of_week)` (or simpler, depending on strategy) to the last observed target value per key. For `same_hour_last_week`: store the training-end 168 hours of target plus index.
  - `predict(features)` — for each test timestamp, look up the appropriate historical value. **Guard**: raise `ValueError` if the required look-back row is missing from training data; instruct caller to supply more training history.
  - `save(path)` / `load(path)` via `io.save_joblib` / `io.load_joblib`.
  - `metadata` property returns a `ModelMetadata` with `name="naive-same-hour-last-week"` (or parametrised on strategy), `fit_utc`, and `hyperparameters={"strategy": config.strategy}`.
  - `_cli_main(argv=None) -> int` printing strategy + help (DESIGN §2.1.1).
- **Acceptance:** AC-2, AC-3, AC-7.
- **Tests (spec-derived; see F-13):**
  - `test_naive_fit_stores_lookup_table`.
  - `test_naive_predict_same_hour_last_week` — synthetic 336-row (14-day) fixture with a known pattern; predict reproduces `y_{t-168}`.
  - `test_naive_predict_raises_when_lookback_missing` — fit on 72 rows, predict on hours that need 168h lookback → `ValueError`.
  - `test_naive_save_load_round_trip` — save, load, assert predictions identical.
  - `test_naive_conforms_to_model_protocol` — `isinstance(NaiveModel(cfg), Model)`.
- **Command:** `uv run pytest tests/unit/models/test_naive.py -q`.

### Task T4 — Linear regression model (statsmodels OLS)
*(Depends on T1 for `LinearConfig`; T2 for `Model` protocol.)*

- [ ] Create `src/bristol_ml/models/linear.py`:
  - `class LinearModel` implementing the `Model` protocol, using `statsmodels.regression.linear_model.OLS`.
  - `fit(features, target)`:
    - Resolve `feature_columns`: if `None`, use all weather float32 columns from `assembler.OUTPUT_SCHEMA` (enumerate at runtime to stay in sync).
    - Add intercept column if `fit_intercept=True` (statsmodels does not add one by default).
    - Fit with `sm.OLS(y, X).fit()`; store the `RegressionResultsWrapper`.
    - Populate `metadata` including coefficients via `hyperparameters={"coefficients": {...}, "rsquared": results.rsquared, ...}`.
  - `predict(features)` — apply intercept addition if needed; call `results.predict(X)`; return as `pd.Series` indexed to the input.
  - `save(path)` / `load(path)` via `io.save_joblib` for the wrapping object (the `RegressionResultsWrapper` itself pickles fine through joblib).
  - `_cli_main(argv=None) -> int` printing `.summary()` if fitted, else help.
- **Acceptance:** AC-1, AC-3, AC-7.
- **Tests (spec-derived):**
  - `test_linear_fit_recovers_known_coefficients` — synthetic `y = 2*x1 + 3*x2 + 1 + N(0, 0.01)`; fitted coefficients within tolerance.
  - `test_linear_predict_shape` — predict on 24-row test set returns 24-row Series.
  - `test_linear_save_load_identical_predictions` — AC-3.
  - `test_linear_conforms_to_model_protocol` — `isinstance(LinearModel(cfg), Model)`.
  - `test_linear_metadata_includes_coefficients` — post-fit `metadata.hyperparameters["coefficients"]` is populated.
- **Command:** `uv run pytest tests/unit/models/test_linear.py -q`.

### Task T5 — Metric functions
*(Independent of other tasks; can run in parallel with T2–T4 if needed.)*

- [ ] Create `src/bristol_ml/evaluation/metrics.py`:
  - Four pure functions: `mae`, `mape`, `rmse`, `wape`.
  - Accept `np.ndarray | pd.Series | list[float]` via `np.asarray` coercion at entry.
  - Reject length mismatches, NaN values, zero-denominators (D8 MAPE, D9 WAPE) with named `ValueError`.
  - Module-level `METRIC_REGISTRY: dict[str, MetricFn]` mapping name → function, used by the harness when `MetricsConfig.names` drives selection.
  - Docstrings cite formulae verbatim (per D8, D9).
- [ ] Extend `evaluation/__init__.py` with metric re-exports.
- **Acceptance:** AC-4, AC-7 (per-metric unit tests).
- **Tests (spec-derived; see F-14):**
  - `test_mae_hand_computed` — `([1, 2, 3], [2, 2, 2]) → (1 + 0 + 1)/3 = 0.667`.
  - `test_rmse_hand_computed` — `([1, 2, 3], [2, 2, 2]) → sqrt(2/3)`.
  - `test_mape_hand_computed` — small fixture with obvious values.
  - `test_mape_raises_on_zero_target` — `y_true` contains 0 → `ValueError` (D8).
  - `test_wape_hand_computed` — verify `Σ|err|/Σ|y|` exactly (D9).
  - `test_wape_raises_on_zero_sum_target` — all zeros → `ValueError` (D9).
  - `test_metric_rejects_nan`.
  - `test_metric_rejects_length_mismatch`.
  - `test_perfect_prediction_all_zero` — `(y, y) → 0.0` for all four.
- **Command:** `uv run pytest tests/unit/evaluation/test_metrics.py -q`.

### Task T6 — Evaluation harness
*(Depends on T1, T2, T5; also depends on Stage 3 `rolling_origin_split_from_config`.)*

- [ ] Create `src/bristol_ml/evaluation/harness.py`:
  - `evaluate(model, df, splitter_cfg, metrics, *, target_column="nd_mw", feature_columns=None)`.
  - **H-1 UTC guard**: raise `ValueError("DataFrame index must be UTC-aware; got {df.index.tz}")` if `df.index.tz` is not None and not UTC.
  - If `feature_columns` is `None`, pick all float32 columns from `assembler.OUTPUT_SCHEMA`.
  - Iterate folds via `rolling_origin_split_from_config(len(df), splitter_cfg)`.
  - Per fold: call `model.fit(X_train, y_train)`, `model.predict(X_test)`, compute each metric; collect results into one row per fold.
  - Return a `pd.DataFrame` with columns `fold_index`, `train_end`, `test_start`, `test_end`, plus one column per named metric.
  - Emit `INFO`-level structured log per fold: `fold_index`, `train_len`, `test_len`, per-metric values (loguru house style per NFR-6).
  - Emit one `INFO`-level summary log on completion: total folds, elapsed time, per-metric mean ± std.
- [ ] Extend `evaluation/__init__.py` with `evaluate` re-export.
- **Acceptance:** AC-1, AC-5, AC-7. **H-1 lands here.**
- **Tests (spec-derived):**
  - `test_harness_rejects_non_utc_index` — H-1.
  - `test_harness_accepts_naive_index` — tz-None is allowed (splitter is tz-agnostic); add this test only if the spec is naive-friendly; otherwise assert the reverse. Decide at implementation.
  - `test_harness_returns_one_row_per_fold` — on a 336-row fixture, assert `len(result) == fold_count`.
  - `test_harness_per_fold_metrics_match_direct_computation` — for one fold, manually compute metrics and assert they match the harness output row.
  - `test_harness_logs_summary` — uses `loguru_caplog`; asserts the completion summary line carries `total_folds=` and per-metric mean.
- **Command:** `uv run pytest tests/unit/evaluation/test_harness.py -q`.

### Task T7 — NESO forecast ingestion extension
*(Depends on T1 for `NesoForecastIngestionConfig`; independent of T2–T6.  Can run in parallel.)*

- [ ] Create `src/bristol_ml/ingestion/neso_forecast.py`:
  - Copy-and-adapt of `neso.py`; target resource `08e41551-80f8-4e28-a416-ea473a695db9`.
  - `OUTPUT_SCHEMA: pa.Schema` — half-hourly columns per the resource (`TARGETDATE`, `FORECASTDEMAND`, etc.) + `retrieved_at_utc` scalar-per-fetch.
  - `fetch(config, cache=OFFLINE)` — uses `_common._retrying_get`, `_common._respect_rate_limit`, `_common._atomic_write`.  Two-req-per-minute NESO rate limit applies.
  - `load(path)` — schema-validating parquet reader.
  - `_cli_main(argv=None) -> int`.
- [ ] Extend `src/bristol_ml/ingestion/CLAUDE.md` with the new module surface.
- [ ] VCR cassette under `tests/fixtures/neso_forecast/cassettes/` for a representative request (CKAN `datastore_search` with `limit` pagination).  Cassette coverage: one month of 2023 Q4 (per research R6 "2023-Q4 sample into `data/raw/`").
- **Acceptance:** AC-5 (benchmark data source); NFR-3.
- **Tests (spec-derived + cassette-backed):**
  - `test_neso_forecast_fetch_writes_parquet` — VCR cassette playback; asserts parquet exists, schema passes.
  - `test_neso_forecast_load_schema_enforced` — reading a file with missing column raises.
  - `test_neso_forecast_provenance_column_populated` — `retrieved_at_utc` is a single scalar across all rows of one fetch.
- **Command:** `uv run pytest tests/unit/ingestion/test_neso_forecast.py -q && uv run python -m bristol_ml.ingestion.neso_forecast --help`.

### Task T8 — Benchmark comparison helper + `train` CLI
*(Depends on T3, T4, T6, T7.)*

- [ ] Create `src/bristol_ml/evaluation/benchmarks.py`:
  - `compare_on_holdout(models, df, neso_forecast, splitter_cfg, metrics)`:
    - Align NESO half-hourly forecast to hourly via **D4** aggregation (`mean`).
    - Restrict comparison to the intersection of model test period and NESO forecast coverage (Apr 2021+).
    - For each named model, run the harness; for NESO, compute metrics against its own aligned forecast vs actuals over the same periods.
    - Return a `pd.DataFrame` indexed by model name, columns = metrics.
- [ ] Create `src/bristol_ml/train.py` (or `models/_cli.py`):
  - Single entry point `_cli_main(argv=None) -> int`.
  - Resolves config via `load_config(overrides=argv)`; loads feature table via `assembler.load(cfg.features.weather_only.cache_dir / cfg.features.weather_only.cache_filename)`; instantiates the model(s) selected by the Hydra group; runs the harness; prints the metric table.
  - If `cfg.ingestion.neso_forecast` is populated and cache is warm, also runs `benchmarks.compare_on_holdout` and prints the three-way table.
  - Table printing: a small helper `_print_metric_table(df: pd.DataFrame) -> None` that formats floats to 2 dp and respects British-English column labels.
- [ ] Update `src/bristol_ml/evaluation/CLAUDE.md` with harness + metrics + benchmarks surface.
- **Acceptance:** AC-1, AC-5, AC-10.
- **Tests (spec + implementation-derived):**
  - `test_benchmarks_aligns_half_hourly_to_hourly` — synthetic HH → hourly mean matches hand computation.
  - `test_benchmarks_three_way_table_shape` — returns a 3-row DataFrame with 4 metric columns.
  - `test_train_cli_prints_metric_table` — run `python -m bristol_ml.train` via `subprocess.run`; parse stdout; assert a metric row is present.
  - `test_train_cli_model_swap` — run with `model=naive` override; assert naive-specific stdout marker.
- **Command:** `uv run pytest tests/unit/evaluation/test_benchmarks.py tests/unit/test_train_cli.py -q && uv run python -m bristol_ml.train --help`.

### Task T9 — Demo notebook
*(Depends on T1–T8.)*

- [ ] Create `notebooks/04_linear_baseline.ipynb`:
  - Thin cells per DESIGN §2.1.8: import from `bristol_ml.models`, `bristol_ml.evaluation`, call `load_config()` with the notebook reduced-fold override `evaluation.rolling_origin.step=168` (per **D7**).
  - Cell 1 (md): stage goal, note on why losing to NESO is the expected outcome.
  - Cell 2 (code): `load_config`, `assembler.load`.
  - Cell 3 (code): fit seasonal-naive; print its `ModelMetadata`.
  - Cell 4 (code): fit linear model; print `model.results.summary()` as the notebook demo payoff.
  - Cell 5 (code): run `harness.evaluate` for both models; display the per-fold metric table.
  - Cell 6 (code): load NESO forecast cache (or skip with a markdown note if not populated); run `benchmarks.compare_on_holdout`; display the three-way table.
  - Cell 7 (code): residual plot (linear model residuals vs time, as line) and 48-hour forecast overlay (actuals, naive, linear, NESO on one axes).
  - Cell 8 (md): closing narrative — weather-only captures temperature sensitivity but misses day-of-week and calendar effects; Stage 5 adds those.
- [ ] Smoke check: `uv run jupyter nbconvert --to notebook --execute notebooks/04_linear_baseline.ipynb --output /tmp/04_test_run.ipynb` finishes **under 120 s** with warm caches (per **D7**).
- **Acceptance:** AC-6, AC-11 (notebook is the demo artefact), AC-13.
- **No new tests.** nbconvert smoke is the gate.

### Task T10 — Stage hygiene
*(Depends on T0–T9.)*

- [ ] `CHANGELOG.md` under `[Unreleased]`: `### Added` bullets for `models/` module, `evaluation.metrics`, `evaluation.harness`, `evaluation.benchmarks`, `ingestion.neso_forecast`, `Model` protocol, notebook, config groups, tests.  `### Changed` bullet noting `evaluation/__init__.__all__` re-export gained `rolling_origin_split_from_config` (H-3). `### Fixed` for H-1 UTC guard.
- [ ] `docs/lld/stages/04-linear-baseline.md` — retrospective following `docs/lld/stages/00-foundation.md` template.  Must document any deviations from this plan; cite ADR-0003 if D3 accepted.
- [ ] `docs/stages/README.md` — flip Stage 4 status cell to `shipped`, link brief = plan, link retro.
- [ ] `docs/architecture/layers/models.md` — **new** layer doc, warn-tier (Contract + Internals).  Template: `docs/architecture/layers/evaluation.md`.  Open question: whether hyperparameter search gets a sub-module (defer to Stage 10+).
- [ ] `docs/architecture/layers/evaluation.md` — extend to cover metrics, harness, benchmarks; update "open questions" section to mark benchmark placement (D5) as resolved with a back-reference.
- [ ] `docs/architecture/README.md` — layers table gains `models` row; **warn-tier**.
- [ ] `docs/architecture/decisions/0003-protocol-for-model-interface.md` — MADR ADR recording D3 (Protocol vs ABC).  Append-only.
- [ ] `docs/architecture/ROADMAP.md` — update "Models" row status.
- [ ] Move this plan from `docs/plans/active/` to `docs/plans/completed/` **as part of the final commit only**.
- [ ] **Not** touching `docs/intent/DESIGN.md` §6 unless a structural change requires it; deny-tier for the lead.  If touched: surface to human first.
- **Acceptance:** AC-9, AC-11, AC-12, AC-14.

---

## 7. Files expected to change

### New
- `src/bristol_ml/models/__init__.py`
- `src/bristol_ml/models/CLAUDE.md`
- `src/bristol_ml/models/protocol.py`
- `src/bristol_ml/models/naive.py`
- `src/bristol_ml/models/linear.py`
- `src/bristol_ml/models/io.py`
- `src/bristol_ml/evaluation/metrics.py`
- `src/bristol_ml/evaluation/harness.py`
- `src/bristol_ml/evaluation/benchmarks.py`
- `src/bristol_ml/ingestion/neso_forecast.py`
- `src/bristol_ml/train.py`  *(or `models/_cli.py`; decide at T8)*
- `conf/model/naive.yaml`
- `conf/model/linear.yaml`
- `conf/ingestion/neso_forecast.yaml`
- `conf/evaluation/metrics.yaml`
- `conf/evaluation/benchmark.yaml`
- `tests/unit/models/__init__.py`
- `tests/unit/models/test_protocol.py`
- `tests/unit/models/test_naive.py`
- `tests/unit/models/test_linear.py`
- `tests/unit/models/test_io.py`
- `tests/unit/evaluation/test_metrics.py`
- `tests/unit/evaluation/test_harness.py`
- `tests/unit/evaluation/test_benchmarks.py`
- `tests/unit/ingestion/test_neso_forecast.py`
- `tests/unit/test_train_cli.py`
- `tests/fixtures/neso_forecast/cassettes/neso_forecast_2023_q4.yaml` (VCR)
- `notebooks/04_linear_baseline.ipynb`
- `docs/lld/stages/04-linear-baseline.md`
- `docs/architecture/layers/models.md`
- `docs/architecture/decisions/0003-protocol-for-model-interface.md`

### Modified
- `conf/_schemas.py` — `ModelMetadata`, `NaiveConfig`, `LinearConfig`, `ModelsGroup`, `MetricsConfig`, `NesoForecastIngestionConfig`, `NesoBenchmarkConfig`; new fields on `AppConfig` / `IngestionGroup` / `EvaluationGroup`; H-2 inline comments on `SplitterConfig`.
- `conf/config.yaml` — new `defaults:` entries (`model: linear`, `ingestion/neso_forecast`, `evaluation/metrics`, `evaluation/benchmark`).
- `src/bristol_ml/evaluation/__init__.py` — H-3 re-export; new re-exports for `metrics`, `harness`, `benchmarks`.
- `src/bristol_ml/evaluation/CLAUDE.md` — extend with metrics/harness/benchmarks surface; mark D5 resolved.
- `src/bristol_ml/ingestion/CLAUDE.md` — add `neso_forecast` to surface list.
- `CHANGELOG.md` — `[Unreleased]` bullets.
- `docs/stages/README.md` — Stage 4 status cell to `shipped`, links.
- `docs/architecture/README.md` — layers table gains `models`.
- `docs/architecture/layers/evaluation.md` — extend for metrics + harness + benchmarks.
- `docs/architecture/ROADMAP.md` — Models row updated.
- `README.md` — add `python -m bristol_ml.train` entry-point mention.

### Intentionally not modified
- `docs/intent/**` — deny-tier. Intent is immutable.
- `src/bristol_ml/cli.py`, `__main__.py` — model selection is a config override, not a subcommand (codebase map §F).
- `src/bristol_ml/config.py` — `load_config` signature unchanged.
- `src/bristol_ml/features/**` — no Stage 4 reason to touch.
- `pyproject.toml` / `uv.lock` — `statsmodels` already declared; no new runtime deps. If T5 needs `tabulate` for the metric table, reject — use f-strings. If `joblib` is not yet a declared dependency, surface before Phase 2.
- `src/bristol_ml/evaluation/splitter.py` — UTC check belongs in harness (H-1), not splitter (codebase map §D).
- `docs/intent/DESIGN.md` §6 — deny-tier; only touch with human approval if structural change demands.

**Dependency audit note:** `joblib` is NOT in `pyproject.toml` as of baseline. statsmodels does NOT depend on joblib transitively in a way guaranteed to expose it. **The plan assumes joblib will need adding to `pyproject.toml` at T2.** Confirm during the Phase 2 kickoff; if joblib is absent, add it and `uv lock` as the first act of T2.

---

## 8. Exit criteria (definition of done per DESIGN §9)

- All tests pass: `uv run pytest -q` green; no `xfail`, no skipped.
- Lint/format clean: `uv run ruff check .`, `uv run ruff format --check .`.
- Pre-commit clean: `uv run pre-commit run --all-files`.
- Standalone CLIs exit 0: `python -m bristol_ml.models.naive --help`, `python -m bristol_ml.models.linear --help`, `python -m bristol_ml.evaluation.metrics --help`, `python -m bristol_ml.evaluation.harness --help`, `python -m bristol_ml.ingestion.neso_forecast --help`, `python -m bristol_ml.train --help`.
- Hydra model swap works end-to-end: `python -m bristol_ml.train model=naive` and `python -m bristol_ml.train model=linear` both exit 0 and print a metric table.
- Notebook runs top-to-bottom under **120 s** with warm caches (per D7).
- Every new public symbol has a British-English docstring.
- CHANGELOG bullets present under `[Unreleased]`.
- Retrospective at `docs/lld/stages/04-linear-baseline.md`.
- ADR-0003 filed.
- Stages index status cell updated.
- This plan moved from `docs/plans/active/` to `docs/plans/completed/` in the final commit.
- **Stage 3 deferred items H-1, H-2, H-3 landed** (T0 + T6).
- Pre-existing `dependabot` / CI invocation paths unchanged.

---

## 9. Team-shape recommendation

**Sequential single-session** work by the lead (me), following the orchestrator playbook — Phase 2 task-by-task, spawning `@test-author` after each code task to write spec-derived tests before declaring the task complete.

Rationale:
- T1, T2, T5, T7 have minimal data dependencies on each other and could theoretically run in parallel; in practice all four are small (1–3 files each) and the cost of coordinating worktrees outweighs the ~2 hours saved.
- T3 and T4 share the `Model` protocol (T2) but otherwise diverge; both need to exist before T6 (harness) can run end-to-end tests.
- T8 (train CLI + benchmarks) is the most integration-heavy task and must be serial.
- No new model family — statsmodels OLS is well-understood.  No further research contemplated beyond the R1–R9 already landed.

Escalate to `@reframer` only if a task fails three times with the same framing (per CLAUDE.md §Escalation ladder).  Early candidates for fragility:
- T4 (linear model) — statsmodels intercept handling is a common pitfall; expect one revision.
- T7 (NESO forecast VCR) — CKAN pagination semantics may surprise; allow budget for one cassette re-record.

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `joblib` absent from `pyproject.toml`; adding it triggers `uv lock` churn mid-T2 | **High** | Low | Audit at Phase 2 kickoff; add + lock as first act of T2 if absent. |
| NESO forecast resource schema differs from expected (columns, date format) | Medium | High | Research fixed on `08e41551...` (codebase map §E); record real response in the first VCR cassette at T7 and derive `OUTPUT_SCHEMA` from that cassette, not from guesswork. |
| NESO forecast rate limit (2 req/min) slows cassette recording | Low | Low | Use `_respect_rate_limit` via `_common.py`; single-shot fetch for the month-sized sample is well under the limit. |
| Seasonal-naive look-back misses first 168 hours of any fold → NaN predictions on Fold 0 | Medium | Medium | T3 guard raises `ValueError`; the harness in T6 handles by passing `min_train_periods >= 168` in notebook + CLI (the default 8760 already satisfies). Test: `test_naive_predict_raises_when_lookback_missing`. |
| statsmodels intercept handling silently drops the constant (common pitfall) | Medium | Medium | `LinearConfig.fit_intercept=True` by default; explicit `sm.add_constant` call in `fit()`; test: `test_linear_fit_recovers_known_coefficients` with non-zero intercept. |
| Harness re-fits in place; save/load round-trip flakes because subsequent `fit()` mutates state | Low | Medium | Specify `fit()` must be re-entrant; `save()` captures most recent fit.  Test explicitly in T3/T4. |
| `ModelMetadata` shape over-specified here; Stage 9 registry needs different fields | Medium | Medium | `hyperparameters: dict[str, Any]` absorbs future extensions. Keep the other fields minimal. |
| Notebook exceeds 120 s on a slow laptop even with weekly-step override | Low | Low | Drop to `min_train_periods=720` (30 days) in notebook as fallback; document as demo artefact, not evaluation. |
| `Model` name collides with Pydantic `BaseModel` or other convention | Low | Low | If collisions surface, rename internal symbol `ModelProtocol`; export as `Model` in `__all__`. |
| NESO forecast pre-2021 coverage gap surprises users in the notebook | Medium | Low | Restrict three-way comparison window to 2021+; flag gap in notebook markdown cell. |
| ADR-0003 wording contentious (Protocol vs ABC has religious debates) | Low | Low | Focus ADR on trade-offs; cite DESIGN §7.3 sketch as the tie-breaker.  Stage 9 can supersede with another ADR if registry design demands. |

---

## Human sign-off (2026-04-19)

All decisions accepted as proposed.

- D1 (seasonal-naive definition = `y_{t-168}`): **ACCEPT**
- D2 (OLS library = statsmodels): **ACCEPT**
- D3 (Protocol + `@runtime_checkable`; ADR-0003): **ACCEPT**
- D4 (HH → hourly aggregation = mean): **ACCEPT**
- D5 (benchmark placement = `evaluation/benchmarks.py`): **ACCEPT**
- D6 (serialisation = joblib; skops noted as Stage 9 upgrade): **ACCEPT**
- D7 (notebook `step=168` override; CLI `step=24`): **ACCEPT**
- D8 (MAPE zero-denominator = raise `ValueError`): **ACCEPT**
- D9 (WAPE = `Σ|y−ŷ|/Σ|y|`): **ACCEPT**
- D10 (fold-level predictions not persisted at Stage 4): **ACCEPT**
- D11 ("beat NESO" not a stage goal): **ACCEPT**

Housekeeping:
- H-1 (UTC guard in harness): **ACCEPT**
- H-2 (`SplitterConfig` inline comments): **ACCEPT**
- H-3 (`__all__` re-export of `rolling_origin_split_from_config`): **ACCEPT**
