# Evaluation — layer architecture

- **Status:** Provisional — realised by Stage 3 (rolling-origin splitter, shipped), Stage 4 (metrics + fold-level harness + three-way NESO benchmark, shipped), Stage 6 (diagnostic-plot helper library + fixed-window NESO benchmark bar chart, shipped), Stage 9 (`evaluate_and_keep_final_model` for registry integration, shipped), and Stage 10 (`plots.loss_curve` for the NN live-demo surface, shipped). Revisit at Stage 18 (drift monitoring placement).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (evaluation paragraph) and [§5.1](../../intent/DESIGN.md#51-evaluation) (rolling-origin evaluator).
- **Concrete instances:** [Stage 3 retro](../../lld/stages/03-feature-assembler.md) (the splitter); [Stage 4 retro](../../lld/stages/04-linear-baseline.md) (metrics, harness, benchmarks); [Stage 6 retro](../../lld/stages/06-enhanced-evaluation.md) (diagnostic-plot helper library); [Stage 10 retro](../../lld/stages/10-simple-nn.md) (`loss_curve`).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.7 (tests at boundaries).

---

## Why this layer exists

The evaluation layer sits between features and models: it is the **library** (per DESIGN §3.2 — explicitly not a service) that every modelling stage reaches into for time-series cross-validation primitives. It owns:

- Splitters — enumerating train/test folds under rolling-origin or related rules.
- Metric functions (Stage 4+) — pure `(y_true, y_pred) -> float` callables.
- A fold-level harness (Stage 4+) — the loop that runs a model across folds and collects per-fold metrics.
- Benchmark comparison (Stage 6+) — reporting a candidate model against the NESO day-ahead forecast.

It is deliberately data-structure-agnostic. Splitters yield integer-array indices into a caller-held DataFrame; metric functions take array-like `y_true` / `y_pred` without touching a DataFrame at all. This is load-bearing for the layer: models and feature sets change shape across the stage plan, but the index-array + metric-function contract does not.

The layer was new at Stage 3, expanded at Stage 4 with the metric functions, the evaluator harness, and the three-way NESO benchmark comparison, and extended again at Stage 6 with a diagnostic-plot helper library (residuals, ACF, forecast overlays, empirical uncertainty band, fixed-window NESO bar chart). Stage 18 (drift monitoring) may extend it again with post-deployment scoring primitives.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| Rolling-origin splitter (expanding + fixed window) | ✓ | — |
| Metric functions (MAE, MAPE, RMSE, WAPE) | ✓ | — |
| Fold-level evaluation harness | ✓ | — |
| Three-way benchmark comparison against NESO day-ahead | ✓ (Stage 4; see D5 below) | — |
| Fixed-window retrospective NESO bar chart | ✓ (Stage 6, `plots.benchmark_holdout_bar` — consumes `NesoBenchmarkConfig.holdout_start/_end`) | — |
| Diagnostic-plot helper library (residuals, ACF, overlays, uncertainty band) | ✓ (Stage 6, `evaluation/plots.py`) | — |
| Drift-detection primitives | ✓ (Stage 18, if scoped here) | — |
| Training a model | — | models layer |
| Saving an evaluation run | — | registry layer (Stage 9) |
| Serving predictions | — | serving layer |
| Plotting evaluations | ✓ (Stage 6, bare helpers returning `matplotlib.figure.Figure`) | interactive dashboards / web UIs stay out |

The split is enforced by the layer's public surface: every evaluation primitive is a pure function or an iterator yielding plain arrays / floats / simple records. A primitive that needs a fitted model object, a registry handle, or a live HTTP client is doing something that does not belong here.

## Cross-module conventions

The layer holds four primitive families at Stage 4; these conventions bind all of them.

### 1. Module shape

- `src/bristol_ml/evaluation/<module>.py` — one module per primitive family. Stage 3 shipped `splitter.py`; Stage 4 added `metrics.py`, `harness.py`, and `benchmarks.py`; Stage 6 added `plots.py` (diagnostic-plot helpers).
- `conf/evaluation/<group>.yaml` — Hydra group file per primitive with configurable knobs. Stage 3 shipped `rolling_origin.yaml`; Stage 4 added `metrics.yaml` (selectable metric list) and `benchmark.yaml` (three-way NESO benchmark window + HH → hourly aggregation rule); Stage 6 added `plots.yaml` (figsize, DPI, display timezone, ACF default lags).
- No dedicated fixtures directory — evaluation primitives operate on integers, array-likes, and pandas frames; tests build inputs inline.

### 2. Public interface

Evaluation primitives follow one of three shapes:

**Splitters** — iterators yielding `(train_idx, test_idx)` index-array pairs:

```python
def rolling_origin_split(n_rows: int, *, min_train: int, test_len: int,
                         step: int, gap: int = 0,
                         fixed_window: bool = False
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
def rolling_origin_split_from_config(n_rows: int,
                                     config: SplitterConfig
                                    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
```

**Metric functions** — pure `(y_true, y_pred) -> float`:

```python
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
METRIC_REGISTRY: dict[str, MetricFn]
```

**Fold-level harness** — loop that calls a `Model` across a splitter's folds:

```python
def evaluate(model: Model, df: pd.DataFrame, splitter_cfg: SplitterConfig,
             metrics: Sequence[MetricFn], *,
             target_column: str = "nd_mw",
             feature_columns: Sequence[str] | None = None,
             return_predictions: bool = False,
            ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: ...
```

**Benchmark helper** — three-way NESO comparison:

```python
def align_half_hourly_to_hourly(df: pd.DataFrame, *,
                                aggregation: Literal["mean", "first"] = "mean",
                                value_columns: Sequence[str] = ...) -> pd.DataFrame: ...
def compare_on_holdout(models: Mapping[str, Model], df: pd.DataFrame,
                       neso_forecast: pd.DataFrame, splitter_cfg: SplitterConfig,
                       metrics: Sequence[MetricFn], *,
                       aggregation: Literal["mean", "first"] = "mean",
                       target_column: str = "nd_mw",
                       feature_columns: Sequence[str] | None = None
                       ) -> pd.DataFrame: ...
```

- Splitters take primitive kwargs (not a config object) so bare unit tests can call them without instantiating Pydantic models. A `_from_config` wrapper is sugar for the call site.
- Metric functions are array-in-float-out, broadcast-friendly, and must handle `np.ndarray`, `pd.Series`, and `list[float]` identically. MAPE raises on any zero in `y_true` (D8); WAPE raises on `Σ|y_true| == 0` (D9).
- The harness is the single orchestration point — it holds the loop, collects per-fold metrics into a long-form `pd.DataFrame`, and is the only function in the layer that touches a `Model` instance. The **H-1 UTC guard** lives here: tz-aware non-UTC indices raise `ValueError` before any fold is enumerated.
- `compare_on_holdout` runs every model through `evaluate`, aggregates NESO HH → hourly, and scores the aligned NESO series on the intersection of fold test-periods and forecast coverage. Returns a DataFrame indexed `[*sorted(models), "neso"]`; empty intersections raise.
- `python -m bristol_ml.evaluation.<module>` is every module's CLI (§2.1.1). The splitter CLI prints the fold count and first/last-fold heads against a synthetic `--n-rows`; the metrics CLI prints the registered + selected metric names; the harness CLI runs the resolved model against the warm feature-table cache; the benchmarks CLI prints the three-way comparison when both the assembler and NESO forecast caches are warm.

### 3. Data-structure agnosticism

Splitter output is `tuple[np.ndarray, np.ndarray]` — integer index arrays, not DataFrame slices. Callers slice their own frame via `df.iloc[train_idx]`. This is a non-negotiable design decision (intent AC-4) because:

- The same splitter works for a `weather_only` feature set at Stage 3, a `weather_calendar` set at Stage 5, and a REMIT-augmented set at Stage 16 — all with different column shapes.
- Integer arrays are trivially serialisable (log a fold; replay a fold) in a way that pandas slices are not.
- `sklearn`-style APIs return integer arrays; newcomers familiar with `TimeSeriesSplit` get the same shape.

A splitter that returned `(train_df, test_df)` tuples would couple the evaluation layer to pandas and to a specific feature-table schema. The slicing cost is one line at every call site and belongs there, not here.

### 4. Fail-loudly parameter validation

Every primitive validates its parameters up front and raises `ValueError` with a named-parameter message:

- Splitters reject non-positive `min_train`, `test_len`, `step`; negative `gap`; and infeasible configs (`min_train + gap + test_len > n_rows` — the first fold cannot exist).
- Metric functions reject length-mismatched `y_true` / `y_pred`, NaN in either, and the D8 MAPE / D9 WAPE zero-denominator cases.
- The harness rejects tz-aware non-UTC indices (**H-1**) before any fold is enumerated.
- `compare_on_holdout` rejects empty `models` / `metrics` and empty fold-test ∩ NESO-coverage intersections.

Silent fall-throughs on infeasibility — e.g. yielding zero folds when the caller asked for "all folds from this config", or returning an empty three-way table when NESO has no coverage over the holdout — are actively avoided.

### 5. Configuration

Primitives with runtime-configurable knobs (the splitter's window type, train length, horizon) ride the Hydra + Pydantic pipeline:

- `conf/evaluation/<name>.yaml` declares the knob defaults; the `# @package evaluation.<name>` header slots the group under `AppConfig.evaluation`.
- `conf/_schemas.py` declares the Pydantic type (`SplitterConfig` for Stage 3); the `EvaluationGroup` aggregates siblings.
- The primitive itself takes primitive kwargs (see above); the Pydantic model is consumed in the `_from_config` wrapper at the call-site boundary.

## Upgrade seams

Each of these is swappable without touching downstream code. The index-array + pure-function interface is what's load-bearing.

| Swappable | Load-bearing |
|-----------|--------------|
| Splitter window type (expanding ↔ sliding) — `SplitterConfig.fixed_window` | Splitter yielding `tuple[np.ndarray, np.ndarray]` |
| Gap / embargo size — `SplitterConfig.gap` | Splitter invariants: no overlap, monotonic, fixed test width |
| Test-fold width — `SplitterConfig.test_len` | Metric functions as pure `(y_true, y_pred) -> float` |
| Metric function implementation (numpy → torch at serving time) | `evaluate()` return shape (long-form per-fold DataFrame) |
| Benchmark source (NESO forecast → alt. vendor) | Hydra + Pydantic config surface (group under `evaluation.<name>`) |
| Diagnostic-plot palette (Okabe-Ito → alt.) via `plt.rcParams["axes.prop_cycle"]` override (Stage 6) | Model-agnostic signature — helpers take `pd.Series` / `pd.DataFrame`, never a `Model` instance |

## Module inventory

| Module | Function | Stage | Status | Notes |
|--------|---------|-------|--------|-----|
| `evaluation/splitter.py` | Rolling-origin splitter (expanding + fixed) | 3 | Shipped | First layer member; sets the index-array contract. |
| `evaluation/metrics.py` | MAE, MAPE, RMSE, WAPE + `METRIC_REGISTRY` | 4 | Shipped | Pure `(y_true, y_pred) -> float`; fractions per DESIGN §5.3. D8/D9 zero-denominator policies raise. |
| `evaluation/harness.py` | Fold-level evaluator `evaluate()` | 4 | Shipped | Holds the H-1 UTC guard; per-fold + summary loguru INFO logs. |
| `evaluation/benchmarks.py` | NESO three-way `compare_on_holdout` + HH→hourly aligner | 4 | Shipped | D5 resolved in favour of an evaluation-layer sibling; a future peer-module refactor stays cheap because the public surface is two pure functions. |
| `evaluation/plots.py` | Diagnostic-plot helpers: `residuals_vs_time`, `predicted_vs_actual`, `acf_residuals`, `error_heatmap_hour_weekday`, `forecast_overlay`, `forecast_overlay_with_band`, `benchmark_holdout_bar`, `loss_curve`; plus `apply_plots_config(PlotsConfig)` to wire Hydra overrides | 6, 10 | Shipped | Model-agnostic (AC-3) — inputs are `pd.Series` / `pd.DataFrame`, never a `Model` object. Okabe-Ito palette injected into `plt.rcParams` at import time (D2); `apply_plots_config` propagates `evaluation.plots.figsize` / `dpi` from a loaded `PlotsConfig` (D5; wired up at Phase 3 review N2). `benchmark_holdout_bar` consumes `NesoBenchmarkConfig.holdout_start/_end` via a lazy `compare_on_holdout` import (D10). `loss_curve` added at Stage 10 (plan D6) — renders per-epoch train + validation loss from `NnMlpModel.loss_history_`; model-agnostic (accepts any list of `{"epoch", "train_loss", "val_loss"}` dicts). |
| `evaluation/drift.py` (tentative) | Post-deployment drift primitives | 18 | Planning | May live here or under `monitoring/`; decide at Stage 18. |

## Open questions

- **Benchmark comparison — evaluation layer or a `benchmarks` peer?** *Resolved at Stage 4.* Per plan [04-linear-baseline.md D5](../../plans/completed/04-linear-baseline.md), the helper lives in `evaluation/benchmarks.py` as a layer sibling. A future Stage 6 refactor into a peer `benchmarks/` module is cheap because the public surface is two pure functions (`align_half_hourly_to_hourly`, `compare_on_holdout`).
- **Multi-horizon fold structure.** The Stage 3 splitter yields fixed `test_len`-hour horizons. Week-ahead evaluation needs either a horizon column on the metrics output or a separate splitter flavour. Deferred until a week-ahead model is in scope.
- **`NesoBenchmarkConfig.holdout_start/_end` consumers.** *Resolved at Stage 6.* Per plan [06-enhanced-evaluation.md D10](../../plans/completed/06-enhanced-evaluation.md), `evaluation/plots.benchmark_holdout_bar` is the fixed-window consumer — it constructs a single-fold `SplitterConfig(fixed_window=True, ...)` from the two `holdout_*` fields and dispatches through `compare_on_holdout` for the retrospective metric table. Stage 4's `compare_on_holdout` continues to derive the holdout window from rolling-origin fold test-periods; the two consumers coexist without sharing state.
- **Drift monitoring placement.** Stage 18's drift detection primitives could live here (as an extension of the evaluation library for post-deployment evaluation) or in a peer `monitoring/` module. DESIGN §3.2 puts monitoring as its own layer; the exact split of primitives across the two at Stage 18 is undecided.
- **Registry integration.** Stage 9 introduces a model registry with metadata sidecars. Whether an evaluation run should be identified by a registry entry ID, a content hash, or a free-form label is not yet drawn. Decide at Stage 9.
- **`sklearn.model_selection.TimeSeriesSplit` parity.** The Stage 3 splitter deliberately does not depend on scikit-learn. A future deep-learning stage may want the same iterator shape from scikit-learn for compatibility with its own utilities; the current primitive is close enough that a shim would be mechanical. Worth building when a caller asks.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (layer responsibilities), [§5.1](../../intent/DESIGN.md#51-evaluation) (rolling-origin evaluator), [§5.3](../../intent/DESIGN.md#53-reporting) (per-fold mean-and-spread reporting).
- [`docs/intent/03-feature-assembler.md`](../../intent/03-feature-assembler.md) — Stage 3 intent (the splitter half).
- [`docs/intent/04-linear-baseline.md`](../../intent/04-linear-baseline.md) — Stage 4 intent (the metrics, harness, and three-way benchmark).
- [`docs/lld/stages/03-feature-assembler.md`](../../lld/stages/03-feature-assembler.md), [`docs/lld/stages/04-linear-baseline.md`](../../lld/stages/04-linear-baseline.md) — retrospectives applying this architecture.
- [`docs/lld/research/03-feature-assembler.md`](../../lld/research/03-feature-assembler.md) §2 — rolling-origin vocabulary and fold-count arithmetic.
- [`src/bristol_ml/evaluation/CLAUDE.md`](../../../src/bristol_ml/evaluation/CLAUDE.md) — module-local guide; splitter + metrics + harness + benchmarks surface with invariants.
- Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*. International Journal of Forecasting 16, 437-450.
- Kolassa, S., & Schütz, W. (2007). *Advantages of the MAD/Mean ratio over the MAPE*. Foresight 6, 40-43. (WAPE formula pinned at D9.)
