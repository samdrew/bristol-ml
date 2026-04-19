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
import sys
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from bristol_ml.evaluation.metrics import METRIC_REGISTRY, MetricFn
from bristol_ml.evaluation.splitter import rolling_origin_split_from_config
from bristol_ml.features.assembler import WEATHER_VARIABLE_COLUMNS

if TYPE_CHECKING:  # pragma: no cover — typing-only imports
    from bristol_ml.models.protocol import Model
    from conf._schemas import SplitterConfig

__all__ = ["evaluate"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        One row per fold, with columns ``fold_index`` (``int``),
        ``train_end`` / ``test_start`` / ``test_end`` (timestamps from
        ``df.index``), plus one float column per metric keyed by the
        metric callable's ``__name__``.  Ordered by ``fold_index``.

    Raises
    ------
    TypeError
        If ``df.index`` is not a :class:`pandas.DatetimeIndex`.
    ValueError
        If ``df.index`` is tz-aware but not UTC (plan H-1); if the
        target column or any feature column is missing from ``df``;
        if ``metrics`` is empty.
    """
    _validate_inputs(df, metrics, target_column)
    columns = _resolve_feature_columns(df, feature_columns)

    target_series = df[target_column]
    features_frame = df[list(columns)]

    fold_records: list[dict[str, object]] = []
    started = time.monotonic()

    for fold_index, (train_idx, test_idx) in enumerate(
        rolling_origin_split_from_config(len(df), splitter_cfg)
    ):
        X_train = features_frame.iloc[train_idx]
        y_train = target_series.iloc[train_idx]
        X_test = features_frame.iloc[test_idx]
        y_test = target_series.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_values: dict[str, float] = {
            metric.__name__: float(metric(y_test, y_pred)) for metric in metrics
        }

        record: dict[str, object] = {
            "fold_index": fold_index,
            "train_end": df.index[train_idx[-1]],
            "test_start": df.index[test_idx[0]],
            "test_end": df.index[test_idx[-1]],
            **metric_values,
        }
        fold_records.append(record)

        logger.info(
            "Evaluator fold {} train_len={} test_len={} metrics={}",
            fold_index,
            len(train_idx),
            len(test_idx),
            metric_values,
        )

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

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    """Instantiate the concrete model class named by the discriminated union."""
    from conf._schemas import LinearConfig, NaiveConfig

    if isinstance(model_cfg, NaiveConfig):
        from bristol_ml.models.naive import NaiveModel

        return NaiveModel(model_cfg)
    if isinstance(model_cfg, LinearConfig):
        from bristol_ml.models.linear import LinearModel

        return LinearModel(model_cfg)
    return None


def _target_column(model_cfg: object) -> str:
    """Return the target column name declared by the resolved model config."""
    from conf._schemas import LinearConfig, NaiveConfig

    if isinstance(model_cfg, (NaiveConfig, LinearConfig)):
        return model_cfg.target_column
    return "nd_mw"


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
