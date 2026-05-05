"""Fold-level evaluation harness — glues the splitter, metrics, and model.

Stage 4 Task T6.  Given a :class:`bristol_ml.models.Model` implementor and
a Stage-3 feature table, :func:`evaluate` iterates rolling-origin folds
and returns a per-fold metric DataFrame ready for notebook display or
subsequent aggregation.  The harness is the integration seam between
the four earlier Stage-4 artefacts (protocol, naive, linear, metrics)
and the Stage-3 splitter.

Contract highlights:

- The feature DataFrame ``df`` must have a :class:`pandas.DatetimeIndex`.
  The index may be tz-naive or UTC-aware; **any other timezone is
  rejected** via the H-1 guard inherited from the Stage 3 review.  The
  guard lives here rather than in the splitter because
  ``rolling_origin_split_from_config`` receives only ``n_rows`` — it
  would be forced to care about a data structure it otherwise never
  touches (codebase-map §D).
- If ``feature_columns`` is ``None`` the harness falls back to every
  float32 weather column declared in
  :data:`bristol_ml.features.assembler.WEATHER_VARIABLE_COLUMNS` — the
  same rule :class:`bristol_ml.models.linear.LinearModel` uses, so the
  harness and the default linear model never disagree on "what are the
  features?".
- Per-fold timestamps (``train_end``, ``test_start``, ``test_end``) are
  read from ``df.index`` so the returned DataFrame carries a
  human-legible audit trail for notebook tables.  On a tz-naive input
  the timestamps are tz-naive; on a UTC-aware input they are UTC-aware.

One structured loguru ``INFO`` log per fold (``fold_index``, ``train_len``,
``test_len``, per-metric values); one summary ``INFO`` line on completion
(total folds, elapsed wall time, per-metric mean ± std).  No DEBUG chatter
so the notebook demo stays clean.

Running standalone::

    python -m bristol_ml.evaluation.harness [--help]

The CLI resolves the defaults-only config, loads the feature table via
:func:`bristol_ml.features.assembler.load`, runs the resolved model
through :func:`evaluate`, and prints the per-fold table — a live-demo
sanity check without the train-CLI ceremony.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from bristol_ml.evaluation.metrics import METRIC_REGISTRY, MetricFn
from bristol_ml.evaluation.splitter import rolling_origin_split_from_config
from bristol_ml.features.assembler import WEATHER_VARIABLE_COLUMNS

if TYPE_CHECKING:  # pragma: no cover — typing-only imports
    from bristol_ml.models.protocol import Model
    from conf._schemas import SplitterConfig

__all__ = ["evaluate", "evaluate_and_keep_final_model"]


#: Column order of the per-fold predictions frame returned by
#: :func:`evaluate` when ``return_predictions=True``.  Pinned by the
#: Stage 6 T5 contract; consumers (``forecast_overlay_with_band``) rely
#: on ``horizon_h``, ``y_true``, ``y_pred``, and ``error``.
_PREDICTIONS_COLUMN_ORDER: tuple[str, ...] = (
    "fold_index",
    "test_start",
    "test_end",
    "horizon_h",
    "y_true",
    "y_pred",
    "error",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = ...,
    feature_columns: Sequence[str] | None = ...,
    return_predictions: Literal[False] = ...,
    n_jobs: int = ...,
) -> pd.DataFrame: ...


@overload
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = ...,
    feature_columns: Sequence[str] | None = ...,
    return_predictions: Literal[True],
    n_jobs: int = ...,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
    return_predictions: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``model`` through the rolling-origin folds described by ``splitter_cfg``.

    Parameters
    ----------
    model:
        Any :class:`~bristol_ml.models.protocol.Model` implementor.  The
        harness calls ``model.fit`` and ``model.predict`` once per fold;
        the protocol guarantees that re-calling ``fit`` discards prior
        state (plan §10 risk row).
    df:
        Feature table — typically the Stage 3 assembler output with
        ``timestamp_utc`` promoted to the index.  Must have a
        :class:`pandas.DatetimeIndex`; tz-naive or UTC-aware only.
    splitter_cfg:
        The validated :class:`~conf._schemas.SplitterConfig` driving the
        fold layout.
    metrics:
        A sequence of :data:`~bristol_ml.evaluation.metrics.MetricFn`
        callables.  Column names in the returned DataFrame come from the
        ``__name__`` attribute of each callable (so the four registry
        entries produce ``"mae"``, ``"mape"``, ``"rmse"``, ``"wape"``).
    target_column:
        Column name holding the target series; defaults to
        ``"nd_mw"`` — the Stage 3 assembler's national-demand column.
    feature_columns:
        Columns to pass to ``model.fit`` / ``model.predict``.  ``None``
        (the default) falls back to every float32 weather column in
        :data:`~bristol_ml.features.assembler.WEATHER_VARIABLE_COLUMNS`.
    return_predictions:
        When ``False`` (default), the harness returns only the per-fold
        metric DataFrame — the Stage 4 Stage behaviour.  When ``True``,
        the harness returns a 2-tuple ``(metrics_df, predictions_df)``
        where ``predictions_df`` is a long-form frame with one row per
        forecast-hour across all folds and columns
        ``["fold_index", "test_start", "test_end", "horizon_h",
        "y_true", "y_pred", "error"]``.  Consumed by
        :func:`bristol_ml.evaluation.plots.forecast_overlay_with_band`
        to build the empirical q10-q90 uncertainty band (Stage 6 D9).
    n_jobs:
        Number of worker processes to dispatch fold work across.
        ``1`` (default) runs every fold in the calling process — the
        original Stage 4 behaviour, byte-for-byte preserved.  Values
        ``> 1`` use :class:`joblib.Parallel` with the ``loky`` backend
        to fit folds in parallel.  Folds are mathematically independent
        (the ``Model`` protocol's ``fit`` re-entrancy contract makes
        each fold a pure function of ``(train_idx, test_idx)`` and a
        fresh model instance), so the parallel result matches the
        serial result on the metrics frame and the predictions frame
        after both have been ordered by ``fold_index``.

        Two kinds of equality apply by model class:

        - **Closed-form models** (``NaiveModel``, ``LinearModel``,
          ``ScipyParametricModel``): byte-for-byte identical between
          ``n_jobs=1`` and ``n_jobs>1``; the parallel-vs-serial test
          asserts ``check_exact=True``.
        - **Iterative-MLE models** (``SarimaxModel``, the NN families):
          metrics agree to ``rtol≈1e-6`` — well below practical
          interpretation — but tiny float-level drift is possible
          because statsmodels' Kalman filter + ``scipy.optimize``
          internals are not bit-deterministic across separate
          processes.  Operators should compare on the metric scale
          (MW or fraction), not bit-by-bit.

        Each worker sets ``OMP_NUM_THREADS=1`` and ``MKL_NUM_THREADS=1``
        to prevent BLAS thread oversubscription — without this, ``N``
        parallel jobs each spawning ``N`` BLAS threads would total
        ``N**2`` threads and slow down rather than speed up.  Useful
        primarily for the Stage 7 SARIMAX path where each fold's MLE
        fit is ~7-8 s on a 30-day training window (typical 4-fold
        speedup ~2.2x, scaling toward the core count for larger fold
        counts); the Stage 8 parametric fit is ~4 ms per fold, where
        pickle overhead exceeds the work and ``n_jobs=1`` is best.
        ``< 1`` raises ``ValueError``.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        When ``return_predictions=False`` (default): one row per fold
        with columns ``fold_index``, ``train_end``, ``test_start``,
        ``test_end``, plus one float column per metric.  Ordered by
        ``fold_index``.

        When ``return_predictions=True``: a 2-tuple
        ``(metrics_df, predictions_df)`` where ``metrics_df`` is the
        same frame as above and ``predictions_df`` is a long-form frame
        with dtypes ``fold_index: int64``, ``test_start/test_end:
        datetime64[ns, UTC]`` (inherits input tz; tz-naive input
        produces tz-naive timestamps), ``horizon_h: int64``, and
        ``y_true/y_pred/error: float64``.

    Notes
    -----
    The ``return_predictions`` flag is a **single-flag concession** for
    Stage 6.  Do not add a second boolean flag for any future output
    extension — see ``src/bristol_ml/evaluation/CLAUDE.md``
    'Harness output - API growth trigger' (Stage 6 D9 architectural-debt
    note).

    Raises
    ------
    TypeError
        If ``df.index`` is not a :class:`pandas.DatetimeIndex`.
    ValueError
        If ``df.index`` is tz-aware but not UTC (plan H-1); if the
        target column or any feature column is missing from ``df``;
        if ``metrics`` is empty.
    """
    if n_jobs < 1:
        raise ValueError(f"evaluate() requires n_jobs >= 1; got {n_jobs}.")
    _validate_inputs(df, metrics, target_column)
    columns = _resolve_feature_columns(df, feature_columns)

    target_series = df[target_column]
    features_frame = df[list(columns)]

    started = time.monotonic()

    fold_specs = list(enumerate(rolling_origin_split_from_config(len(df), splitter_cfg)))

    fold_records: list[dict[str, object]] = []
    predictions_frames: list[pd.DataFrame] = []

    if n_jobs == 1:
        # Serial path — preserves the byte-for-byte behaviour of the
        # original Stage 4 harness.  Re-uses the caller's ``model``
        # instance (the ``fit`` re-entrancy contract makes this safe).
        for fold_index, (train_idx, test_idx) in fold_specs:
            record, fold_preds = _run_fold(
                fold_index=fold_index,
                train_idx=train_idx,
                test_idx=test_idx,
                model=model,
                features_frame=features_frame,
                target_series=target_series,
                index=df.index,
                metrics=metrics,
                return_predictions=return_predictions,
            )
            fold_records.append(record)
            if fold_preds is not None:
                predictions_frames.append(fold_preds)
    else:
        # Parallel path — dispatch fold work across ``n_jobs`` worker
        # processes via joblib's loky backend.  Each worker receives a
        # *deep copy* of the caller's ``model`` so its ``fit`` does not
        # race against any sibling worker's state; this is cheap because
        # the project's models carry only their config + numpy arrays
        # before fit.  The parent's ``model`` is therefore unchanged on
        # return — callers that need the final-fold fitted estimator
        # use :func:`evaluate_and_keep_final_model`, which re-runs the
        # final fold serially in the parent.
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            initializer=_init_worker,
        )(
            delayed(_run_fold)(
                fold_index=fold_index,
                train_idx=train_idx,
                test_idx=test_idx,
                model=copy.deepcopy(model),
                features_frame=features_frame,
                target_series=target_series,
                index=df.index,
                metrics=metrics,
                return_predictions=return_predictions,
            )
            for fold_index, (train_idx, test_idx) in fold_specs
        )
        # joblib.Parallel preserves submission order, but sort
        # defensively so a future backend swap does not break ordering.
        results = sorted(results, key=lambda r: int(r[0]["fold_index"]))
        for record, fold_preds in results:
            fold_records.append(record)
            if fold_preds is not None:
                predictions_frames.append(fold_preds)

    # ``_run_fold`` emits the per-fold INFO log itself, once per fold,
    # so both serial (n_jobs=1) and parallel (n_jobs>1) paths produce
    # identical structured records.  Order is deterministic for the
    # serial path and may interleave for the parallel path.

    elapsed = time.monotonic() - started

    result = pd.DataFrame.from_records(
        fold_records,
        columns=[
            "fold_index",
            "train_end",
            "test_start",
            "test_end",
            *(metric.__name__ for metric in metrics),
        ],
    )

    if fold_records:
        summary = {
            metric.__name__: {
                "mean": float(result[metric.__name__].mean()),
                "std": float(result[metric.__name__].std(ddof=0)),
            }
            for metric in metrics
        }
    else:
        summary = {
            metric.__name__: {"mean": float("nan"), "std": float("nan")} for metric in metrics
        }

    logger.info(
        "Evaluator complete: total_folds={} elapsed_seconds={:.3f} summary={}",
        len(fold_records),
        elapsed,
        summary,
    )

    if return_predictions:
        if predictions_frames:
            predictions_df = pd.concat(predictions_frames, ignore_index=True)[
                list(_PREDICTIONS_COLUMN_ORDER)
            ]
        else:
            predictions_df = pd.DataFrame(
                {
                    "fold_index": pd.Series([], dtype=np.int64),
                    "test_start": pd.Series([], dtype=df.index.dtype),
                    "test_end": pd.Series([], dtype=df.index.dtype),
                    "horizon_h": pd.Series([], dtype=np.int64),
                    "y_true": pd.Series([], dtype=np.float64),
                    "y_pred": pd.Series([], dtype=np.float64),
                    "error": pd.Series([], dtype=np.float64),
                }
            )[list(_PREDICTIONS_COLUMN_ORDER)]
        return result, predictions_df

    return result


def evaluate_and_keep_final_model(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, Model]:
    """Run rolling-origin evaluation and return the final-fold fitted model.

    Identical to :func:`evaluate` in every respect except that the
    ``model`` returned is the one the harness left in its fitted state
    after the final fold — the Stage 9 registry (plan D17 / AC-2) saves
    this model rather than re-fitting on the full training set.

    Rationale (plan R2).  The leaderboard metric row is the cross-fold
    mean; the final-fold fitted model is an honest representative of the
    estimator the metrics describe.  Re-fitting on the full training
    set would introduce a second fitted object whose own metrics are not
    the ones in the sidecar.

    Parameters
    ----------
    model:
        See :func:`evaluate`.  On return the same instance carries the
        state of its final-fold ``fit``.
    df, splitter_cfg, metrics, target_column, feature_columns, n_jobs:
        See :func:`evaluate`.  When ``n_jobs > 1`` every fold is fit in
        a worker process; the parent's ``model`` is therefore unchanged
        by the parallel evaluation, so this function performs **one
        additional serial fit on the final fold's training data** to
        leave the caller-supplied estimator in the documented final-fold
        state.  That extra fit costs at most one fold's wall time
        (~7-8 s for SARIMAX, ~4 ms for ScipyParametric) — negligible
        beside the savings from parallelising the rest of the loop.

    Returns
    -------
    tuple[pd.DataFrame, Model]
        ``(metrics_df, model)`` where ``metrics_df`` is the same
        per-fold frame :func:`evaluate` returns and ``model`` is the
        same fitted estimator passed in.  The tuple shape keeps this
        function independent of :func:`evaluate`'s ``return_predictions``
        overload — do not add a second boolean flag to ``evaluate``
        (API-growth rule; see module ``CLAUDE.md``).

    Raises
    ------
    See :func:`evaluate`.  Additionally, a configuration that produces
    zero folds leaves ``model`` unfitted; downstream consumers should
    check ``model.metadata.fit_utc is not None`` before registering.
    """
    # ``return_predictions=False`` narrows ``evaluate``'s return type to
    # ``pd.DataFrame`` via the overload — no runtime assertion needed; a
    # bare ``assert`` would be stripped under ``python -O``.
    metrics_df = evaluate(
        model,
        df,
        splitter_cfg,
        metrics,
        target_column=target_column,
        feature_columns=feature_columns,
        return_predictions=False,
        n_jobs=n_jobs,
    )

    # Under n_jobs>1 the workers each held a deepcopy; the caller's
    # ``model`` is unchanged.  Re-run the final fold's fit serially on
    # the original instance so the registry sees a fitted artefact
    # whose metrics row in ``metrics_df`` matches ground truth.  No-op
    # when there are zero folds (a degenerate config) — caller checks
    # ``model.metadata.fit_utc`` before registering, per the docstring.
    if n_jobs > 1 and len(metrics_df) > 0:
        _validate_inputs(df, metrics, target_column)
        columns = _resolve_feature_columns(df, feature_columns)
        target_series = df[target_column]
        features_frame = df[list(columns)]
        fold_specs = list(rolling_origin_split_from_config(len(df), splitter_cfg))
        train_idx, _test_idx = fold_specs[-1]
        model.fit(features_frame.iloc[train_idx], target_series.iloc[train_idx])

    return metrics_df, model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_worker() -> None:
    """Cap BLAS threads inside each parallel-fold worker process.

    statsmodels SARIMAX and ``scipy.optimize.curve_fit`` both call into
    NumPy/SciPy linear-algebra primitives, which by default use one
    thread per physical core via OpenBLAS / MKL.  When the harness
    dispatches ``N`` worker processes each spawning ``N`` BLAS threads
    the system runs ``N**2`` threads — heavy context switching that
    typically makes a parallel run *slower* than the serial baseline.

    Setting ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` to ``1`` per
    worker gives each fit a single dedicated thread; the OS then
    schedules ``N`` workers cleanly across ``N`` cores.  We use
    ``setdefault`` so a caller who has explicitly tuned thread counts
    upstream is not overridden.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _run_fold(
    *,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model: Model,
    features_frame: pd.DataFrame,
    target_series: pd.Series,
    index: pd.Index,
    metrics: Sequence[MetricFn],
    return_predictions: bool,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Fit ``model`` on one fold and return ``(record, predictions_or_None)``.

    Module-level so :class:`joblib.Parallel`'s loky backend can pickle
    it.  Caller is responsible for passing a model instance that does
    not need to be shared across folds — the serial path passes the
    caller's ``model`` directly (re-entrant ``fit`` discards prior
    state); the parallel path passes a ``copy.deepcopy(model)`` per
    worker so concurrent fits do not race.
    """
    X_train = features_frame.iloc[train_idx]
    y_train = target_series.iloc[train_idx]
    X_test = features_frame.iloc[test_idx]
    y_test = target_series.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metric_values: dict[str, float] = {
        metric.__name__: float(metric(y_test, y_pred)) for metric in metrics
    }

    test_start_ts = index[test_idx[0]]
    test_end_ts = index[test_idx[-1]]

    record: dict[str, Any] = {
        "fold_index": fold_index,
        "train_end": index[train_idx[-1]],
        "test_start": test_start_ts,
        "test_end": test_end_ts,
        **metric_values,
    }

    fold_preds: pd.DataFrame | None = None
    if return_predictions:
        y_true_arr = np.asarray(y_test.to_numpy(), dtype=np.float64)
        y_pred_arr = np.asarray(
            y_pred.to_numpy() if isinstance(y_pred, pd.Series) else y_pred,
            dtype=np.float64,
        )
        n = len(test_idx)
        fold_preds = pd.DataFrame(
            {
                "fold_index": np.full(n, fold_index, dtype=np.int64),
                "test_start": pd.Series([test_start_ts] * n).astype(index.dtype),
                "test_end": pd.Series([test_end_ts] * n).astype(index.dtype),
                "horizon_h": np.arange(n, dtype=np.int64),
                "y_true": y_true_arr,
                "y_pred": y_pred_arr,
                "error": y_true_arr - y_pred_arr,
            }
        )

    logger.info(
        "Evaluator fold {} train_len={} test_len={} metrics={}",
        fold_index,
        len(train_idx),
        len(test_idx),
        metric_values,
    )
    return record, fold_preds


def _validate_inputs(
    df: pd.DataFrame,
    metrics: Sequence[MetricFn],
    target_column: str,
) -> None:
    """Guard the harness-level invariants before any fold work begins.

    - H-1: ``df.index`` must be a :class:`pandas.DatetimeIndex`; tz-naive
      is permitted, but a tz-aware non-UTC index is rejected.  Naive is
      allowed on purpose: the splitter is data-structure-agnostic, so
      requiring UTC awareness for this check alone would force callers
      to localise synthetic test fixtures.
    - ``metrics`` must be non-empty — an evaluation run with zero
      metrics is a configuration error, not a reasonable input.
    - ``target_column`` must exist on ``df``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "evaluate() requires df to carry a pandas DatetimeIndex "
            f"(set_index('timestamp_utc')); got {type(df.index).__name__}."
        )
    if df.index.tz is not None and str(df.index.tz) != "UTC":
        raise ValueError(
            "DataFrame index must be UTC-aware (or tz-naive); "
            f"got df.index.tz={df.index.tz} (plan H-1)."
        )
    if not metrics:
        raise ValueError("evaluate() requires at least one metric; got an empty sequence.")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} is missing from df; "
            f"available columns: {list(df.columns)!r}."
        )


def _resolve_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve the regressor set; fall back to the weather defaults.

    Mirrors :class:`bristol_ml.models.linear.LinearModel._resolve_feature_columns`
    so harness + default model never disagree on "the features".  Any
    missing column raises ``ValueError`` with the missing list named.
    """
    if feature_columns is None:
        resolved = tuple(name for name, _dtype in WEATHER_VARIABLE_COLUMNS)
    else:
        resolved = tuple(feature_columns)

    missing = [c for c in resolved if c not in df.columns]
    if missing:
        raise ValueError(
            f"evaluate(): feature column(s) missing from df: {missing}. "
            f"Supply explicit feature_columns or ensure df carries the "
            f"Stage 3 assembler schema."
        )
    return resolved


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.evaluation.harness``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.evaluation.harness",
        description=(
            "Run the rolling-origin evaluator against the resolved Hydra "
            "config: load the Stage 3 feature-table cache, instantiate the "
            "resolved model, compute the selected metrics, and print the "
            "per-fold table.  Expects warm caches; for the end-to-end "
            "train + benchmark pipeline see `python -m bristol_ml.train`."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluation.rolling_origin.step=168",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Returns ``0`` on success; ``2`` on missing config / cache; ``3`` if
    the resolved ``model`` variant has no factory wired into the harness
    CLI (remote possibility — the config's discriminator constrains this
    to ``naive`` or ``linear``).
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local imports keep ``--help`` lightweight — avoid pulling statsmodels
    # into the import chain when the user just wants usage text.
    from bristol_ml.config import load_config
    from bristol_ml.features import assembler

    cfg = load_config(overrides=list(args.overrides))
    if cfg.features.weather_only is None or cfg.evaluation.rolling_origin is None:
        print(
            "Required config missing: features.weather_only and "
            "evaluation.rolling_origin must both be populated.",
            file=sys.stderr,
        )
        return 2
    if cfg.model is None:
        print(
            "No model resolved — pass `model=linear` or `model=naive` on the CLI.",
            file=sys.stderr,
        )
        return 2

    fset = cfg.features.weather_only
    cache_path = fset.cache_dir / fset.cache_filename
    if not cache_path.exists():
        print(
            f"Feature-table cache missing at {cache_path}. Run "
            "`python -m bristol_ml.features.assembler` first.",
            file=sys.stderr,
        )
        return 2

    df = assembler.load(cache_path)
    df = df.set_index("timestamp_utc")

    model = _build_model_from_config(cfg.model)
    if model is None:
        print(
            f"No harness factory for model type {type(cfg.model).__name__!r}.",
            file=sys.stderr,
        )
        return 3

    selected_metrics = (
        cfg.evaluation.metrics.names
        if cfg.evaluation.metrics is not None
        else tuple(METRIC_REGISTRY)
    )
    metric_fns = [METRIC_REGISTRY[name] for name in selected_metrics]

    result = evaluate(
        model,
        df,
        cfg.evaluation.rolling_origin,
        metric_fns,
        target_column=_target_column(cfg.model),
    )
    print(result.to_string(index=False))
    return 0


def _build_model_from_config(model_cfg: object) -> Model | None:
    """Instantiate the concrete model class named by the discriminated union.

    Stage 11 D13 clause iii adds the :class:`NnTemporalConfig` branch;
    Stage 11 D14 (housekeeping catch-up) adds the missing
    :class:`NnMlpConfig` branch Stage 10 shipped without.  Both land in
    the same commit as T6 — a one-line ``isinstance`` gap is not worth a
    separate Stage-10-hotfix PR, and a named ``Stage 10 catch-up``
    commit trailer keeps the audit trail honest.
    """
    from conf._schemas import (
        LinearConfig,
        NaiveConfig,
        NnMlpConfig,
        NnTemporalConfig,
        SarimaxConfig,
        ScipyParametricConfig,
    )

    if isinstance(model_cfg, NaiveConfig):
        from bristol_ml.models.naive import NaiveModel

        return NaiveModel(model_cfg)
    if isinstance(model_cfg, LinearConfig):
        from bristol_ml.models.linear import LinearModel

        return LinearModel(model_cfg)
    if isinstance(model_cfg, SarimaxConfig):
        from bristol_ml.models.sarimax import SarimaxModel

        return SarimaxModel(model_cfg)
    if isinstance(model_cfg, ScipyParametricConfig):
        from bristol_ml.models.scipy_parametric import ScipyParametricModel

        return ScipyParametricModel(model_cfg)
    if isinstance(model_cfg, NnMlpConfig):
        # Stage 11 D14 — Stage 10 catch-up.  The train CLI already
        # wires NnMlpConfig; the harness factory did not, a latent gap
        # codebase-map §1.4 surfaced.  Fixed in-commit with T6.
        from bristol_ml.models.nn.mlp import NnMlpModel

        return NnMlpModel(model_cfg)
    if isinstance(model_cfg, NnTemporalConfig):
        # Stage 11 D13 clause iii.
        from bristol_ml.models.nn.temporal import NnTemporalModel

        return NnTemporalModel(model_cfg)
    return None


def _target_column(model_cfg: object) -> str:
    """Return the target column name declared by the resolved model config."""
    from conf._schemas import (
        LinearConfig,
        NaiveConfig,
        NnMlpConfig,
        NnTemporalConfig,
        SarimaxConfig,
        ScipyParametricConfig,
    )

    if isinstance(
        model_cfg,
        (
            NaiveConfig,
            LinearConfig,
            SarimaxConfig,
            ScipyParametricConfig,
            NnMlpConfig,
            NnTemporalConfig,
        ),
    ):
        return model_cfg.target_column
    return "nd_mw"


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
