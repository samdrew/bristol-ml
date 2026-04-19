# `bristol_ml.evaluation` — module guide

This module is the **evaluation layer**: splitters, metrics, and evaluators
that sit between the features layer (feature tables in) and the models
layer (models out). Stage 3 introduces it with the rolling-origin splitter;
Stage 4 adds metric functions and the fold-level evaluator harness.

## Current surface (Stage 3)

- `bristol_ml.evaluation.splitter.rolling_origin_split(n_rows, *, min_train,
  test_len, step, gap=0, fixed_window=False)` — yields
  `(train_idx, test_idx)` integer-array pairs in chronological order. The
  kernel function is data-structure-agnostic; callers hold their own
  DataFrame and slice it via `df.iloc[train_idx]` (Intent AC-4).
- `bristol_ml.evaluation.splitter.rolling_origin_split_from_config(n_rows,
  config)` — thin wrapper that unpacks a `SplitterConfig` (see
  `conf/_schemas.py`) into the kernel function's kwargs. Keeps the kernel
  free of Pydantic dependencies so it is usable in bare unit tests.

## Invariants (load-bearing for every downstream modelling stage)

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

These invariants are tested at `tests/unit/evaluation/test_splitter.py`.
If a change here breaks any of them, fix the test only if the invariant
itself is wrong — do not weaken the test to make it pass
(`CLAUDE.md` §"Quality gates and debugging").

## Expected additions (Stage 4)

- `bristol_ml.evaluation.metrics.{mae, mape, rmse, wape}` — pure functions
  on `(y_true, y_pred)`.
- `bristol_ml.evaluation.harness.evaluate(model, df, splitter_cfg, metrics)`
  — fold-level evaluation loop that returns a per-fold metric DataFrame.

## Running standalone

    python -m bristol_ml.evaluation.splitter [--n-rows N] [overrides ...]

Prints the fold count and the first/last fold's train/test index heads,
backed by the resolved `evaluation.rolling_origin` Hydra config.

## Cross-references

- Layer contract — `docs/architecture/layers/evaluation.md` (Stage 3 lands
  the initial version; Stage 4/6 extend it).
- Stage 3 plan — `docs/plans/active/03-feature-assembler.md` §5 (the
  public surface signature) and §6 Task T2.
- Rolling-origin research — `docs/lld/research/03-feature-assembler.md`
  §2 (vocabulary, fold-count arithmetic, the argument against
  `sklearn.TimeSeriesSplit`).
