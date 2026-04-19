"""Three-way benchmark comparison against the NESO day-ahead forecast.

Stage 4 Task T8.  The benchmark helper aligns the half-hourly NESO
day-ahead demand-forecast archive (``bristol_ml.ingestion.neso_forecast``)
with the hourly feature table the Stage 4 models consume and produces a
compact per-model metric table on a shared holdout window.

Why this lives in ``bristol_ml.evaluation.benchmarks`` rather than a peer
module: plan D5 records the placement decision — the intent scopes the
helper here and the evaluation layer doc accepts it; a future Stage 6
refactor into a peer ``benchmarks/`` module is cheap when the helper
grows past one function.

Hourly alignment (D4)
---------------------

The NESO forecast is published at half-hourly cadence keyed by
``timestamp_utc``.  The assembler feature-table is hourly.  Alignment
is a :meth:`pandas.DataFrame.resample` on the UTC index under one of
two rules:

- ``"mean"`` — the D4 default.  Averages the two settlement periods
  inside each UTC hour.  Preserves MW scale and mirrors the Stage 3
  D1 convention for the ND outturn.
- ``"first"`` — retains only the first settlement period of each UTC
  hour.  Loses information and is kept for ablation only.

On clock-change Sundays the UTC timeline is still regular (the NESO
ingester has unwound the DST algebra).  A spring-forward day collapses
to 23 hourly aggregate rows and an autumn-fallback day to 25 — matching
the assembler's behaviour.

Holdout semantics
-----------------

``compare_on_holdout`` restricts every series (model test-period
predictions, NESO forecast, NESO outturn) to the **intersection** of:

- ``splitter_cfg``'s test period (the last fold's ``test_start`` through
  ``test_end`` inclusive), and
- the NESO forecast archive's coverage (which begins April 2021).

If that intersection is empty the function raises :class:`ValueError`
rather than returning an empty table — silent zero-length benchmark
tables would be a confusing failure mode during a live demo.

Output
------

A :class:`pandas.DataFrame` indexed by model name with one column per
metric.  The ``"neso"`` row is always present (it is the benchmark); the
remaining rows come from ``models``.  Metric column names are taken
from each callable's ``__name__`` so they match the ``evaluate``
harness's per-fold DataFrame.

Running standalone::

    python -m bristol_ml.evaluation.benchmarks --help
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

from bristol_ml.evaluation.harness import evaluate
from bristol_ml.evaluation.metrics import METRIC_REGISTRY, MetricFn

if TYPE_CHECKING:  # pragma: no cover — typing-only imports
    from bristol_ml.models.protocol import Model
    from conf._schemas import SplitterConfig


__all__ = [
    "align_half_hourly_to_hourly",
    "compare_on_holdout",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_half_hourly_to_hourly(
    df: pd.DataFrame,
    *,
    aggregation: Literal["mean", "first"] = "mean",
    value_columns: Sequence[str] = ("demand_forecast_mw", "demand_outturn_mw"),
) -> pd.DataFrame:
    """Collapse a half-hourly NESO frame to hourly under the D4 rule.

    Parameters
    ----------
    df:
        Frame produced by :func:`bristol_ml.ingestion.neso_forecast.load`.
        Must carry a tz-aware UTC ``timestamp_utc`` column and every
        name in ``value_columns``.
    aggregation:
        ``"mean"`` (default, plan D4) averages the two settlement periods
        landing in each UTC hour.  ``"first"`` keeps the first settlement
        period per hour and drops the second.
    value_columns:
        Columns to carry through the resample.  Defaults to the forecast
        and outturn MW series consumed by :func:`compare_on_holdout`.

    Returns
    -------
    pd.DataFrame
        Indexed by hourly UTC :class:`pandas.DatetimeIndex`; columns
        match ``value_columns``.  Values are ``float64`` — the MW scale
        is preserved but integer-valued rows may carry a ``.0`` after
        averaging.

    Raises
    ------
    ValueError
        If ``timestamp_utc`` is missing, not tz-aware, not UTC, or if
        any requested value column is missing from ``df``.  Also raised
        if ``aggregation`` is not in ``{"mean", "first"}`` — mirrors the
        ``NesoBenchmarkConfig.aggregation`` literal.
    """
    if aggregation not in ("mean", "first"):
        raise ValueError(
            f"aggregation must be 'mean' or 'first'; got {aggregation!r} "
            "(plan D4 Literal contract)."
        )
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            "align_half_hourly_to_hourly: 'timestamp_utc' column missing "
            f"from frame with columns {list(df.columns)!r}."
        )
    ts = df["timestamp_utc"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise ValueError(
            "align_half_hourly_to_hourly requires a UTC-aware 'timestamp_utc' "
            f"column; got tz={ts.dt.tz}."
        )
    missing = [c for c in value_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"align_half_hourly_to_hourly: value column(s) missing from frame: {missing}."
        )

    selected = df[["timestamp_utc", *value_columns]].copy()
    # Cast the value columns to float64 so the aggregation path does not
    # silently truncate int MW values under the ``"mean"`` rule.
    for col in value_columns:
        selected[col] = selected[col].astype("float64")
    selected = selected.set_index("timestamp_utc").sort_index()

    resampled = selected.resample("1h")
    hourly = resampled.mean() if aggregation == "mean" else resampled.first()

    # ``resample`` inserts NaN rows for any empty hours.  A clock-change
    # day legitimately has 23 (spring) or 25 (autumn) hours; we keep the
    # NaN handling upstream of the caller — ``compare_on_holdout`` drops
    # NaN rows explicitly after the holdout intersection.
    return hourly


def compare_on_holdout(
    models: Mapping[str, Model],
    df: pd.DataFrame,
    neso_forecast: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    aggregation: Literal["mean", "first"] = "mean",
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Three-way metric table on the intersection of model holdout ∩ NESO coverage.

    The helper runs every ``models`` entry through
    :func:`bristol_ml.evaluation.harness.evaluate` and computes the same
    metric set on the NESO day-ahead forecast aggregated to hourly.  All
    scores are computed on the **same** hourly timestamps — the
    intersection of the model test periods (the timestamps yielded by
    ``splitter_cfg`` over ``df``) and the NESO forecast archive's hourly
    coverage.

    Parameters
    ----------
    models:
        Name → :class:`~bristol_ml.models.protocol.Model` mapping.  Every
        value is fit and predicted via the harness; callers should pass
        a freshly-constructed model per run because the harness calls
        ``fit`` once per fold.
    df:
        Feature table (hourly; Stage 3 assembler shape).  Must carry a
        ``timestamp_utc`` column OR a UTC-aware
        :class:`pandas.DatetimeIndex`.  The function promotes the column
        to an index when needed so callers can pass either shape.
    neso_forecast:
        Half-hourly NESO forecast frame
        (:func:`bristol_ml.ingestion.neso_forecast.load` output).
    splitter_cfg:
        Rolling-origin :class:`~conf._schemas.SplitterConfig` used to
        define the per-model fold layout.
    metrics:
        Metric callables (see :mod:`bristol_ml.evaluation.metrics`).  The
        NESO row uses the same set.
    aggregation:
        Half-hourly → hourly rule (plan D4 default ``"mean"``).
    target_column:
        Target column on ``df`` — defaults to ``"nd_mw"``.  The NESO
        benchmark's comparable actual is taken from the forecast
        archive's own ``demand_outturn_mw`` column, **not** from
        ``df[target_column]``, because the two sources may disagree on
        MW per hour even after alignment (different rounding in NESO's
        internal preparation).
    feature_columns:
        Forwarded to :func:`evaluate`; see that function's docstring.

    Returns
    -------
    pd.DataFrame
        Index = ``[*sorted(models), "neso"]`` (benchmark row last);
        columns = ``[metric.__name__ for metric in metrics]``.  Every
        value is a ``float``; the harness scores are the mean across
        folds (matching the per-fold DataFrame's column semantics).

    Raises
    ------
    ValueError
        If ``models`` is empty; if ``metrics`` is empty; if the holdout
        intersection is empty; if the NESO frame is missing a required
        column.
    """
    if not models:
        raise ValueError("compare_on_holdout: 'models' is empty; pass at least one model.")
    if not metrics:
        raise ValueError("compare_on_holdout: 'metrics' is empty; pass at least one callable.")

    df = _ensure_utc_index(df)

    # Align the NESO archive to hourly up-front — one pass, reused by both
    # the intersection computation and the NESO metric row.
    neso_hourly = align_half_hourly_to_hourly(
        neso_forecast,
        aggregation=aggregation,
        value_columns=("demand_forecast_mw", "demand_outturn_mw"),
    )
    # Drop rows where either side is NaN — the resample inserts one per
    # empty hour and an incomplete source row would poison the metric.
    neso_hourly = neso_hourly.dropna(how="any")

    # Per-model scores: run the harness, collapse per-fold metrics to
    # one scalar per metric (mean across folds — matches the DESIGN §5.3
    # "per-fold mean and spread" aggregation), restricted to the folds'
    # union test-period intersected with NESO coverage.
    model_rows: dict[str, dict[str, float]] = {}
    per_fold_tables: dict[str, pd.DataFrame] = {}
    for name in sorted(models):
        model = models[name]
        per_fold = evaluate(
            model,
            df,
            splitter_cfg,
            metrics,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        per_fold_tables[name] = per_fold
        # Fold-level mean over each metric column; these are the scalars
        # reported in the benchmark table.
        model_rows[name] = {
            metric.__name__: float(per_fold[metric.__name__].mean()) for metric in metrics
        }

    # Holdout-window intersection: span every fold's test period and
    # intersect with the NESO hourly coverage.
    if not per_fold_tables:
        raise ValueError("compare_on_holdout: evaluator returned no folds; widen the config.")
    fold_spans = pd.concat(per_fold_tables.values(), ignore_index=True)
    holdout_start = pd.Timestamp(fold_spans["test_start"].min())
    holdout_end = pd.Timestamp(fold_spans["test_end"].max())

    mask = (neso_hourly.index >= holdout_start) & (neso_hourly.index <= holdout_end)
    benchmark_slice = neso_hourly.loc[mask]
    if benchmark_slice.empty:
        raise ValueError(
            "compare_on_holdout: intersection of model holdout "
            f"[{holdout_start}, {holdout_end}] with NESO forecast coverage is empty. "
            "Widen the holdout window or extend the NESO forecast cache."
        )

    neso_row = {
        metric.__name__: float(
            metric(
                benchmark_slice["demand_outturn_mw"].to_numpy(),
                benchmark_slice["demand_forecast_mw"].to_numpy(),
            )
        )
        for metric in metrics
    }

    logger.info(
        "Benchmark comparison: models={} neso_rows={} holdout=[{}, {}]",
        list(model_rows),
        int(benchmark_slice.shape[0]),
        holdout_start,
        holdout_end,
    )

    table = pd.DataFrame.from_records(
        [
            *[{"model": name, **row} for name, row in model_rows.items()],
            {"model": "neso", **neso_row},
        ]
    ).set_index("model")
    # Preserve column order matching the ``metrics`` sequence — the
    # ``from_records`` + dict path is order-preserving in Python 3.7+,
    # but re-indexing here makes the contract explicit.
    return table[[metric.__name__ for metric in metrics]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a UTC-aware :class:`pandas.DatetimeIndex`.

    Accepts either an already-indexed frame (UTC-aware or tz-naive) or a
    frame carrying a ``timestamp_utc`` column.  A tz-aware non-UTC index
    is rejected (plan H-1 propagates from the harness).
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None and str(df.index.tz) != "UTC":
            raise ValueError(
                "compare_on_holdout: df.index is tz-aware but not UTC "
                f"(got {df.index.tz}); plan H-1."
            )
        return df
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            "compare_on_holdout: df must carry a 'timestamp_utc' column or a pandas DatetimeIndex."
        )
    ts = df["timestamp_utc"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise ValueError(
            f"compare_on_holdout: 'timestamp_utc' must be UTC-aware; got tz={ts.dt.tz}."
        )
    return df.set_index("timestamp_utc")


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.evaluation.benchmarks``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.evaluation.benchmarks",
        description=(
            "Print the three-way benchmark metric table (naive, linear, "
            "NESO day-ahead) on the resolved holdout window.  Expects warm "
            "caches (assembler output + ingestion/neso_forecast output); "
            "for the full train pipeline see `python -m bristol_ml.train`."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluation.benchmark.aggregation=first",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI — DESIGN §2.1.1 compliance.

    Returns ``0`` on success; ``2`` on missing config / cache.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local imports so ``--help`` stays lightweight.
    from bristol_ml.config import load_config
    from bristol_ml.features import assembler
    from bristol_ml.ingestion import neso_forecast
    from bristol_ml.models.linear import LinearModel
    from bristol_ml.models.naive import NaiveModel
    from conf._schemas import LinearConfig, NaiveConfig

    cfg = load_config(overrides=list(args.overrides))
    fset = cfg.features.weather_only
    nfore_cfg = cfg.ingestion.neso_forecast
    split_cfg = cfg.evaluation.rolling_origin
    if fset is None or nfore_cfg is None or split_cfg is None:
        print(
            "Benchmark CLI requires features.weather_only, "
            "ingestion.neso_forecast, and evaluation.rolling_origin "
            "to be populated; one is None in the resolved config.",
            file=sys.stderr,
        )
        return 2

    feature_cache = fset.cache_dir / fset.cache_filename
    forecast_cache = nfore_cfg.cache_dir / nfore_cfg.cache_filename
    for path in (feature_cache, forecast_cache):
        if not path.exists():
            print(
                f"Required cache missing at {path}. Run the corresponding "
                "ingester with `--cache auto` first.",
                file=sys.stderr,
            )
            return 2

    df = assembler.load(feature_cache).set_index("timestamp_utc")
    neso_df = neso_forecast.load(forecast_cache)

    # Instantiate both baseline models with defaults — the CLI is a
    # smoke check; overrides are Hydra-style on the underlying configs.
    models: dict[str, Model] = {
        "naive": NaiveModel(NaiveConfig()),
        "linear": LinearModel(LinearConfig()),
    }

    selected = (
        cfg.evaluation.metrics.names
        if cfg.evaluation.metrics is not None
        else tuple(METRIC_REGISTRY)
    )
    metric_fns = [METRIC_REGISTRY[name] for name in selected]

    table = compare_on_holdout(
        models,
        df,
        neso_df,
        split_cfg,
        metric_fns,
        aggregation=(
            cfg.evaluation.benchmark.aggregation if cfg.evaluation.benchmark is not None else "mean"
        ),
    )
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
