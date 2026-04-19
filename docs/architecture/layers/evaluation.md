# Evaluation — layer architecture

- **Status:** Provisional — realised by Stage 3 (rolling-origin splitter, shipped). Revisit at Stage 4 (metric functions, fold-level harness), Stage 6 (enhanced evaluation + visualisation), and Stage 9 (registry integration — which may force evaluation to quote a registry entry by ID).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (evaluation paragraph) and [§5.1](../../intent/DESIGN.md#51-evaluation) (rolling-origin evaluator).
- **Concrete instances:** [Stage 3 retro](../../lld/stages/03-feature-assembler.md) (the splitter).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.7 (tests at boundaries).

---

## Why this layer exists

The evaluation layer sits between features and models: it is the **library** (per DESIGN §3.2 — explicitly not a service) that every modelling stage reaches into for time-series cross-validation primitives. It owns:

- Splitters — enumerating train/test folds under rolling-origin or related rules.
- Metric functions (Stage 4+) — pure `(y_true, y_pred) -> float` callables.
- A fold-level harness (Stage 4+) — the loop that runs a model across folds and collects per-fold metrics.
- Benchmark comparison (Stage 6+) — reporting a candidate model against the NESO day-ahead forecast.

It is deliberately data-structure-agnostic. Splitters yield integer-array indices into a caller-held DataFrame; metric functions take array-like `y_true` / `y_pred` without touching a DataFrame at all. This is load-bearing for the layer: models and feature sets change shape across the stage plan, but the index-array + metric-function contract does not.

The layer is new at Stage 3. Stage 4 fills in the metrics and the harness, Stage 6 adds visualisation and benchmark comparison, and Stage 18 (drift monitoring) may extend it again with post-deployment scoring primitives.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| Rolling-origin splitter (expanding + fixed window) | ✓ | — |
| Metric functions (MAE, MAPE, RMSE, WAPE) | ✓ (Stage 4) | — |
| Fold-level evaluation harness | ✓ (Stage 4) | — |
| Benchmark-comparison helpers | ✓ (Stage 6) | — |
| Drift-detection primitives | ✓ (Stage 18, if scoped here) | — |
| Training a model | — | models layer |
| Saving an evaluation run | — | registry layer (Stage 9) |
| Serving predictions | — | serving layer |
| Plotting evaluations | — (bare helpers) | notebooks + docs/lld narrative |

The split is enforced by the layer's public surface: every evaluation primitive is a pure function or an iterator yielding plain arrays / floats / simple records. A primitive that needs a fitted model object, a registry handle, or a live HTTP client is doing something that does not belong here.

## Cross-module conventions

The layer is small today; these conventions will be retrofitted onto the metrics and harness modules as they land in Stage 4.

### 1. Module shape

- `src/bristol_ml/evaluation/<module>.py` — one module per primitive family. Stage 3 ships `splitter.py`; Stage 4 adds `metrics.py` and `harness.py`.
- `conf/evaluation/<group>.yaml` — Hydra group file per primitive with configurable knobs. Stage 3 ships `rolling_origin.yaml`; Stage 4 may add `metrics.yaml` if metric selection becomes runtime-configurable (weak signal so far).
- No dedicated fixtures directory — evaluation primitives operate on integers and array-likes; tests build inputs inline.

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

**Metric functions** (Stage 4) — pure `(y_true, y_pred) -> float`:

```python
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
```

**Fold-level harness** (Stage 4) — loop that calls a `Model` across a splitter's folds:

```python
def evaluate(model: Model, df: pd.DataFrame, splitter_cfg: SplitterConfig,
             metrics: Sequence[MetricFn]) -> pd.DataFrame: ...
```

- Splitters take primitive kwargs (not a config object) so bare unit tests can call them without instantiating Pydantic models. A `_from_config` wrapper is sugar for the call site.
- Metric functions are array-in-float-out, broadcast-friendly, and must handle `np.ndarray`, `pd.Series`, and `list[float]` identically.
- The harness is the single orchestration point — it holds the loop, collects per-fold metrics into a long-form `pd.DataFrame`, and is the only function in the layer that touches a `Model` instance.
- `python -m bristol_ml.evaluation.<module>` is every module's CLI (§2.1.1). The splitter CLI prints the fold count and first/last-fold heads against a synthetic `--n-rows`.

### 3. Data-structure agnosticism

Splitter output is `tuple[np.ndarray, np.ndarray]` — integer index arrays, not DataFrame slices. Callers slice their own frame via `df.iloc[train_idx]`. This is a non-negotiable design decision (intent AC-4) because:

- The same splitter works for a `weather_only` feature set at Stage 3, a `weather_calendar` set at Stage 5, and a REMIT-augmented set at Stage 16 — all with different column shapes.
- Integer arrays are trivially serialisable (log a fold; replay a fold) in a way that pandas slices are not.
- `sklearn`-style APIs return integer arrays; newcomers familiar with `TimeSeriesSplit` get the same shape.

A splitter that returned `(train_df, test_df)` tuples would couple the evaluation layer to pandas and to a specific feature-table schema. The slicing cost is one line at every call site and belongs there, not here.

### 4. Fail-loudly parameter validation

Every primitive validates its parameters up front and raises `ValueError` with a named-parameter message:

- Splitters reject non-positive `min_train`, `test_len`, `step`; negative `gap`; and infeasible configs (`min_train + gap + test_len > n_rows` — the first fold cannot exist).
- Metric functions reject length-mismatched `y_true` / `y_pred` and NaN in either (though the harness is where the NaN-policy is decided — the metric is the primitive).

Silent fall-throughs on infeasibility — e.g. yielding zero folds when the caller asked for "all folds from this config" — are actively avoided. A Stage 4 user running a metric on an infeasible split should see an error, not an empty result DataFrame.

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
| Benchmark source (NESO forecast → alt. vendor) at Stage 6 | Hydra + Pydantic config surface (group under `evaluation.<name>`) |

## Module inventory

| Module | Function | Stage | Status | Notes |
|--------|---------|-------|--------|-----|
| `evaluation/splitter.py` | Rolling-origin splitter (expanding + fixed) | 3 | Shipped | First layer member; sets the index-array contract. |
| `evaluation/metrics.py` (planned) | MAE, MAPE, RMSE, WAPE | 4 | Planning | Pure `(y_true, y_pred) -> float`. |
| `evaluation/harness.py` (planned) | Fold-level evaluation loop | 4 | Planning | Consumes a `Model` and returns per-fold metrics. |
| `evaluation/benchmarks.py` (tentative) | NESO-forecast comparison helpers | 6 | Planning | Whether this lives here or in a peer `benchmarks/` module is an open question. |
| `evaluation/drift.py` (tentative) | Post-deployment drift primitives | 18 | Planning | May live here or under `monitoring/`; decide at Stage 18. |

## Open questions

- **Benchmark comparison — evaluation layer or a `benchmarks` peer?** DESIGN §3.2 is explicit that the rolling-origin evaluator is a library; whether the NESO-forecast comparison is a sibling library in the same layer or a peer module is undecided. Decide at Stage 6 when the comparison actually ships.
- **Harness output shape.** Stage 4 will return a per-fold metric DataFrame; the exact long-vs-wide shape, which fold-identifier columns to include (`fold_index`, `train_end`, `test_start`, `test_end`), and whether to carry the fold's index arrays on the result record is undesigned. Sketched in the Stage 4 plan when it lands.
- **Multi-horizon fold structure.** The Stage 3 splitter yields fixed `test_len`-hour horizons. Week-ahead evaluation needs either a horizon column on the metrics output or a separate splitter flavour. Deferred until a week-ahead model is in scope.
- **Drift monitoring placement.** Stage 18's drift detection primitives could live here (as an extension of the evaluation library for post-deployment evaluation) or in a peer `monitoring/` module. DESIGN §3.2 puts monitoring as its own layer; the exact split of primitives across the two at Stage 18 is undecided.
- **Registry integration.** Stage 9 introduces a model registry with metadata sidecars. Whether an evaluation run should be identified by a registry entry ID, a content hash, or a free-form label is not yet drawn. Decide at Stage 9.
- **`sklearn.model_selection.TimeSeriesSplit` parity.** The Stage 3 splitter deliberately does not depend on scikit-learn. A future deep-learning stage may want the same iterator shape from scikit-learn for compatibility with its own utilities; the current primitive is close enough that a shim would be mechanical. Worth building when a caller asks.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (layer responsibilities), [§5.1](../../intent/DESIGN.md#51-evaluation) (rolling-origin evaluator), [§5.3](../../intent/DESIGN.md#53-reporting) (per-fold mean-and-spread reporting).
- [`docs/intent/03-feature-assembler.md`](../../intent/03-feature-assembler.md) — Stage 3 intent (the splitter half).
- [`docs/lld/stages/03-feature-assembler.md`](../../lld/stages/03-feature-assembler.md) — retrospective applying this architecture.
- [`docs/lld/research/03-feature-assembler.md`](../../lld/research/03-feature-assembler.md) §2 — rolling-origin vocabulary and fold-count arithmetic.
- [`src/bristol_ml/evaluation/CLAUDE.md`](../../../src/bristol_ml/evaluation/CLAUDE.md) — module-local guide, invariants, Stage 4 expected additions.
- Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*. International Journal of Forecasting 16, 437-450.
