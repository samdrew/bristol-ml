# `bristol_ml.evaluation` — module guide

This module is the **evaluation layer**: splitters, metrics, evaluators,
and benchmark comparisons that sit between the features layer (feature
tables in) and the models layer (models out). Stage 3 introduced it with
the rolling-origin splitter; Stage 4 extends it with metric functions,
the fold-level evaluator harness, and the three-way NESO benchmark.

## Current surface (Stage 4)

### Splitter (Stage 3)

- `bristol_ml.evaluation.splitter.rolling_origin_split(n_rows, *, min_train,
  test_len, step, gap=0, fixed_window=False)` — yields
  `(train_idx, test_idx)` integer-array pairs in chronological order. The
  kernel function is data-structure-agnostic; callers hold their own
  DataFrame and slice it via `df.iloc[train_idx]` (Intent AC-4).
- `bristol_ml.evaluation.splitter.rolling_origin_split_from_config(n_rows,
  config)` — thin wrapper that unpacks a `SplitterConfig` (see
  `conf/_schemas.py`) into the kernel function's kwargs. Keeps the kernel
  free of Pydantic dependencies so it is usable in bare unit tests.

### Metrics (Stage 4)

- `bristol_ml.evaluation.metrics.{mae, mape, rmse, wape}` — pure
  `(y_true, y_pred) -> float` callables. MAPE raises on any zero in
  `y_true`; WAPE raises on `sum(|y_true|) == 0`. The fractions (not
  percentages) convention matches DESIGN §5.3 so downstream formatting
  owns the "×100" decision.
- `bristol_ml.evaluation.metrics.METRIC_REGISTRY` — lowercase-name →
  callable mapping consumed by `MetricsConfig.names` and by the harness
  / benchmark CLIs. Adding a new metric is a two-step: pure function
  here + extend the `MetricsConfig.names` Literal.

### Harness (Stage 4)

- `bristol_ml.evaluation.harness.evaluate(model, df, splitter_cfg,
  metrics, *, target_column="nd_mw", feature_columns=None) ->
  pd.DataFrame` — iterates the rolling-origin folds described by
  `splitter_cfg`, calls `model.fit` / `model.predict` per fold, and
  returns a long-form per-fold DataFrame. Columns: `fold_index`
  (`int`), `train_end` / `test_start` / `test_end` (timestamps from
  `df.index`), plus one float column per metric keyed by the metric
  callable's `__name__`.
- **H-1 guard (Stage 3 carry-over, implemented here in Stage 4):**
  `df.index` must be a `pandas.DatetimeIndex`; tz-naive is permitted,
  UTC-aware is permitted, any other timezone is rejected.
- `feature_columns=None` falls back to every weather column in
  `bristol_ml.features.assembler.WEATHER_VARIABLE_COLUMNS` — the same
  rule `LinearModel` uses, so the harness and the default linear model
  never disagree on "what are the features?".
- One structured loguru `INFO` log per fold plus one summary line on
  completion (total folds, elapsed wall time, per-metric mean ± std).

### Benchmarks (Stage 4)

- `bristol_ml.evaluation.benchmarks.align_half_hourly_to_hourly(df, *,
  aggregation="mean", value_columns=("demand_forecast_mw",
  "demand_outturn_mw")) -> pd.DataFrame` — D4 aggregation rule (default
  `"mean"`; alternative `"first"`) that collapses the half-hourly NESO
  forecast to hourly. The output is indexed by hourly UTC
  `DatetimeIndex`; values are `float64` (MW scale preserved).
- `bristol_ml.evaluation.benchmarks.compare_on_holdout(models, df,
  neso_forecast, splitter_cfg, metrics, *, aggregation="mean",
  target_column="nd_mw", feature_columns=None) -> pd.DataFrame` —
  three-way metric table. Runs every `models` entry through
  `harness.evaluate`, aggregates the NESO forecast to hourly, and
  scores the aligned NESO series against its own outturn on the
  intersection of fold test-periods ∩ forecast coverage. Returns a
  DataFrame indexed by `[*sorted(models), "neso"]` with one column per
  metric; per-model values are the mean across folds.
- **Placement note (plan D5).** The benchmark helper lives in
  `evaluation/` rather than a peer `benchmarks/` module. The evaluation
  layer doc's open question on placement is marked resolved with a
  back-reference; a future Stage 6 refactor into a peer module stays
  cheap because the public surface is two pure functions.

## Invariants (load-bearing for every downstream modelling stage)

### Splitter

The splitter **guarantees** that for every yielded `(train_idx, test_idx)`
pair:

- `max(train_idx) < min(test_idx)` — no train/test leakage within a fold.
- `train_idx` and `test_idx` are each monotonically ascending.
- If `gap > 0`, then `max(train_idx) + gap < min(test_idx)` — the embargo
  is honoured.
- `len(test_idx) == test_len` — every fold has a fixed-width forecast
  horizon.
- Under `fixed_window=False` (default), `train_idx` always starts at 0 and
  grows with each fold (expanding window).
- Under `fixed_window=True`, `len(train_idx) == min_train` for every fold
  (sliding window of fixed width).
- Arrays are `numpy.int64` and compatible with `DataFrame.iloc`.

### Metrics

- Pure functions; no side effects; deterministic.
- Length-mismatched or NaN-containing inputs raise `ValueError` at the
  shared `_coerce_and_validate` gate.
- MAPE and WAPE zero-denominator policies raise (plan D8 / D9).

### Harness

- Per-fold metric values are `float` (not `np.floating`) — the DataFrame
  dtype story stays stable across numpy version bumps.
- Re-calling `fit` between folds discards prior state (the Model protocol
  guarantees re-entrancy; plan §10 risk row).

### Benchmarks

- The returned DataFrame always carries a `"neso"` row and one row per
  model in `sorted(models)`; empty intersections raise rather than
  returning an empty frame.
- All metrics are evaluated on the **same** hourly grid — the union of
  per-model fold test-periods intersected with NESO hourly coverage.

These invariants are tested at `tests/unit/evaluation/`. If a change here
breaks any of them, fix the test only if the invariant itself is wrong —
do not weaken the test to make it pass (`CLAUDE.md` §"Quality gates and
debugging").

## Running standalone

    python -m bristol_ml.evaluation.splitter [--n-rows N] [overrides ...]
    python -m bristol_ml.evaluation.metrics [overrides ...]
    python -m bristol_ml.evaluation.harness [overrides ...]
    python -m bristol_ml.evaluation.benchmarks [overrides ...]

The splitter CLI prints the fold count and the first/last fold's train /
test index heads, backed by the resolved `evaluation.rolling_origin` Hydra
config. The metrics CLI prints the registered + selected metric names.
The harness CLI runs the resolved model against the warm feature-table
cache and prints the per-fold table. The benchmark CLI prints the
three-way comparison on the resolved holdout window — it expects both
the assembler cache and the NESO forecast cache to be warm; see
`python -m bristol_ml.train` for the full end-to-end pipeline.

## Cross-references

- Layer contract — `docs/architecture/layers/evaluation.md` (Stage 3
  lands the initial version; Stage 4 extends for metrics + harness +
  benchmarks).
- Stage 4 plan — `docs/plans/active/04-linear-baseline.md` §5 (public
  surface) and §6 Tasks T5–T8.
- Stage 3 plan — `docs/plans/completed/03-feature-assembler.md` §5 (the
  splitter's public signature).
- Rolling-origin research — `docs/lld/research/03-feature-assembler.md`
  §2 (vocabulary, fold-count arithmetic, the argument against
  `sklearn.TimeSeriesSplit`).
- Metric formulae — `DESIGN` §5.3; Kolassa & Schütz (2007) for WAPE.
