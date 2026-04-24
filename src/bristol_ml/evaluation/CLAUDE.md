# `bristol_ml.evaluation` — module guide

This module is the **evaluation layer**: splitters, metrics, evaluators,
benchmark comparisons, and diagnostic-plot helpers that sit between the
features layer (feature tables in) and the models layer (models out).
Stage 3 introduced it with the rolling-origin splitter; Stage 4 extends
it with metric functions, the fold-level evaluator harness, and the
three-way NESO benchmark; Stage 6 adds ``plots`` — a colourblind-safe
diagnostic-plot helper library (residuals, ACF, forecast overlays,
empirical uncertainty bands, and a fixed-window NESO bar chart).

## Current surface (Stages 3–6, 9)

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
  metrics, *, target_column="nd_mw", feature_columns=None,
  return_predictions=False) -> pd.DataFrame | tuple[pd.DataFrame,
  pd.DataFrame]` — iterates the rolling-origin folds described by
  `splitter_cfg`, calls `model.fit` / `model.predict` per fold, and
  returns a long-form per-fold DataFrame. Columns: `fold_index`
  (`int`), `train_end` / `test_start` / `test_end` (timestamps from
  `df.index`), plus one float column per metric keyed by the metric
  callable's `__name__`. When `return_predictions=True`, returns a
  `(metrics_df, predictions_df)` tuple; `predictions_df` has one row
  per (fold, horizon-index) with columns `["fold_index", "test_start",
  "test_end", "horizon_h", "y_true", "y_pred", "error"]` (Stage 6 D9
  — see the API growth trigger note below).
- `bristol_ml.evaluation.harness.evaluate_and_keep_final_model(model,
  df, splitter_cfg, metrics, *, target_column="nd_mw",
  feature_columns=None) -> tuple[pd.DataFrame, Model]` — Stage 9
  (plan D17) companion to `evaluate`.  Delegates to `evaluate()` and
  additionally returns the model instance after the harness has left it
  fitted on the final fold.  Introduced to give the registry
  (`registry.save`) access to the fitted artefact without adding a
  second boolean flag to `evaluate()` (H5 API-growth rule — see
  "Harness output — API growth trigger" below).  The metrics DataFrame
  is identical to the one `evaluate()` returns; the returned `model`
  is the same object passed in, now carrying final-fold state.  A
  zero-fold configuration leaves `model` unfitted; callers should
  guard on `model.metadata.fit_utc is not None` before registering.
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

### Plots (Stage 6)

- `bristol_ml.evaluation.plots.residuals_vs_time(residuals, *,
  display_tz="Europe/London", ax=None) -> Figure` — line plot with a
  thin zero line; x-axis in local display timezone.
- `bristol_ml.evaluation.plots.predicted_vs_actual(y_true, y_pred, *,
  ax=None) -> Figure` — scatter with a 45-degree reference line;
  Gelman-convention axes (x=predicted, y=actual).
- `bristol_ml.evaluation.plots.acf_residuals(residuals, *, lags=168,
  alpha=0.05, reference_lags=(24, 168), ax=None) -> Figure` — wraps
  `statsmodels.graphics.tsaplots.plot_acf`; default `lags=168` (one
  week of hourly data) so the weekly spike that motivates Stage 7's
  SARIMAX is visible. Annotates two labelled vertical markers at the
  daily (lag 24) and weekly (lag 168) reference points by default
  (Stage 6 D7 reinforcement). Pass `reference_lags=()` to disable
  markers.
- `bristol_ml.evaluation.plots.error_heatmap_hour_weekday(residuals,
  *, display_tz="Europe/London", ax=None) -> Figure` — 24x7
  seaborn heatmap (weekday × hour-of-day) of mean signed residual,
  centred at zero using the `RdBu_r` diverging colormap.
- `bristol_ml.evaluation.plots.forecast_overlay(actual,
  predictions_by_name, *, display_tz="Europe/London", ax=None) ->
  Figure` — actual line plus one line per named prediction.
- `bristol_ml.evaluation.plots.forecast_overlay_with_band(actual,
  point_prediction, per_fold_errors, *, quantiles=(0.1, 0.9),
  ax=None) -> Figure` — forecast overlay plus an empirical q10–q90
  uncertainty band derived from rolling-origin per-fold errors
  (Stage 6 D8 — non-parametric, model-agnostic).
- `bristol_ml.evaluation.plots.benchmark_holdout_bar(candidates,
  neso_forecast, features, metrics, *, holdout_start, holdout_end,
  ax=None) -> Figure` — fixed-window bar chart for the NESO three-way
  benchmark comparison (Stage 6 D10 — wires up the
  `NesoBenchmarkConfig.holdout_start/_end` consumer added at Stage 4).
- `bristol_ml.evaluation.plots.loss_curve(history, *, title=..., ax=None)
  -> Figure` — Stage 10 D6 demo-moment helper: renders train + validation
  loss vs epoch from the `NnMlpModel.loss_history_` shape (list of
  `{"epoch", "train_loss", "val_loss"}` dicts).  Uses Okabe-Ito colours
  (`OKABE_ITO[1]` orange for train, `OKABE_ITO[2]` sky-blue for
  validation) so the NN-training plot matches the rest of the Stage 6
  diagnostic surface.  Model-agnostic: accepts any sequence of dicts
  that carry the three keys (no `NnMlpModel` import).
- `bristol_ml.evaluation.plots.apply_plots_config(config: PlotsConfig)
  -> None` — re-apply the rcParams overlay with ``figure.figsize`` and
  ``figure.dpi`` sourced from a loaded ``PlotsConfig``. Call this from
  a notebook or CLI after ``load_config()`` if you want Hydra overrides
  of ``evaluation.plots.figsize`` / ``dpi`` to take effect; without
  this call, the module-default values written at import time stay in
  force (Stage 6 Phase 3 review N2 wired this up — D5's "Hydra-
  configurable" promise was decorative before the fix).

#### Plotting conventions

- **Palette policy (Stage 6 D2).** The module exposes three constants —
  `OKABE_ITO` (8 hex colours, Wong 2011 *Nature Methods*), `SEQUENTIAL_CMAP
  = "cividis"`, and `DIVERGING_CMAP = "RdBu_r"`. All three are formally
  colourblind-safe for deuteranopia, protanopia, and tritanopia.
  `tab10` (matplotlib's default) is explicitly rejected. At import time
  the module writes an Okabe-Ito `axes.prop_cycle` into `plt.rcParams`
  alongside a 12x8 default figsize, 110 DPI, and 12/14/11 pt
  axis/title/legend font sizes (Stage 6 plan D5 human mandate,
  2026-04-20).
- **CVD-safety opt-out idiom.** Facilitators who want a bespoke palette
  call `plt.rcdefaults()` or `plt.rcParams.update({"axes.prop_cycle":
  cycler(color=[...])})` after importing `plots`. The helpers honour
  the active rcParams — no hard-coded colour list inside a helper body.
- **`ax=` composability contract.** Every helper accepts an optional
  `ax: matplotlib.axes.Axes | None`. When `ax is None` the helper mints
  a new figure sized from `plt.rcParams["figure.figsize"]`; otherwise
  the helper draws onto the supplied axes and returns the owning
  figure. This lets facilitators compose helpers into a single
  `plt.subplots(2, 2)` grid without the helpers owning figure
  lifetime.
- **British English in docstrings and labels.** Axis labels use
  "colour" / "behaviour"; weekday abbreviations are `Mon`/`Tue`/…/`Sun`
  per the Stage 5 calendar convention.
- **Model-agnosticism (AC-3).** Helpers take `pd.Series` /
  `pd.DataFrame` inputs — never a `Model` object. `isinstance(model,
  LinearModel)` branches are banned from this module.

#### Harness output — API growth trigger

`evaluate(..., return_predictions: bool = False)` is a single-flag
concession at Stage 6 for the uncertainty-band helper.
**Do not add a second boolean flag to `evaluate()` for any future output extension.**
If Stage 9 (registry) needs a run-id column, Stage 18 (drift) needs drift
scores, a multi-horizon model needs per-horizon predictions with a
different shape, or a probabilistic model needs quantiles — propose a
first-class `EvaluationResult` dataclass (metrics + optional predictions
+ optional extras) returned unconditionally, or a typed `evaluate_v2`
alongside a deprecation of the boolean form. The re-engineering trigger
is the **second** ask; Stage 6 is the first.

This rule is a Stage 6 plan D9 concession (2026-04-20 human mandate,
recorded verbatim here so future implementers find it without needing
to re-read the plan). See `docs/plans/completed/06-enhanced-evaluation.md`
§1 D9 for the original wording and rationale.

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
    python -m bristol_ml.evaluation.plots [--help]

The splitter CLI prints the fold count and the first/last fold's train /
test index heads, backed by the resolved `evaluation.rolling_origin` Hydra
config. The metrics CLI prints the registered + selected metric names.
The harness CLI runs the resolved model against the warm feature-table
cache and prints the per-fold table. The benchmark CLI prints the
three-way comparison on the resolved holdout window — it expects both
the assembler cache and the NESO forecast cache to be warm; see
`python -m bristol_ml.train` for the full end-to-end pipeline. The
plots CLI prints the exported helper surface, the active Okabe-Ito
palette, and the rcParams written at import time — useful for a
live-demo sanity check and to satisfy DESIGN §2.1.1.

## Cross-references

- Layer contract — `docs/architecture/layers/evaluation.md` (Stage 3
  lands the initial version; Stage 4 extends for metrics + harness +
  benchmarks).
- Stage 4 plan — `docs/plans/completed/04-linear-baseline.md` §5 (public
  surface) and §6 Tasks T5–T8.
- Stage 3 plan — `docs/plans/completed/03-feature-assembler.md` §5 (the
  splitter's public signature).
- Rolling-origin research — `docs/lld/research/03-feature-assembler.md`
  §2 (vocabulary, fold-count arithmetic, the argument against
  `sklearn.TimeSeriesSplit`).
- Metric formulae — `DESIGN` §5.3; Kolassa & Schütz (2007) for WAPE.
