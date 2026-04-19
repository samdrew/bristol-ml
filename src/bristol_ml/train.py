"""End-to-end train + evaluate CLI — Stage 4 demo entry point.

Runnable as ``python -m bristol_ml.train [overrides ...]``.  Wires the
four Stage 4 subsystems into a single invocation:

1. Resolve + validate the Hydra config via
   :func:`bristol_ml.config.load_config`.
2. Load the Stage 3 feature table from its warm parquet cache
   (:func:`bristol_ml.features.assembler.load`).
3. Instantiate the resolved model variant — plan D10 wires
   :class:`~bristol_ml.models.naive.NaiveModel` and
   :class:`~bristol_ml.models.linear.LinearModel` behind the Hydra
   ``model=`` group discriminator.
4. Run :func:`bristol_ml.evaluation.harness.evaluate` against the
   rolling-origin splitter config; print the per-fold metric table.
5. If ``ingestion.neso_forecast`` is populated and its cache is warm,
   also run :func:`bristol_ml.evaluation.benchmarks.compare_on_holdout`
   for the three-way NESO comparison and print the resulting table.

The CLI is the demo-moment surface of the stage (intent "Demo moment"):
a single invocation that produces both the harness output and the
three-way benchmark table, with model swap via ``model=naive`` /
``model=linear``.  No training loop is reimplemented here — the work
lives in the models, harness, and benchmarks modules; this file is
purely orchestration.

Exit codes::

    0 — success (per-fold table printed; benchmark table printed iff
        the forecast cache was warm).
    2 — required config group or cache missing.
    3 — the resolved model variant has no harness factory.

Running standalone::

    python -m bristol_ml.train --help
    python -m bristol_ml.train                           # default (linear)
    python -m bristol_ml.train model=naive               # swap model
    python -m bristol_ml.train evaluation.rolling_origin.step=168
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

import pandas as pd
from loguru import logger

from bristol_ml.evaluation.metrics import METRIC_REGISTRY

__all__ = ["_cli_main"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.train",
        description=(
            "Train and evaluate the Stage 4 model resolved by the Hydra "
            "`model=` group against the rolling-origin splits from Stage 3, "
            "and print the per-fold metric table.  If the NESO day-ahead "
            "forecast cache is warm, also print the three-way benchmark "
            "table (naive, linear, NESO)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model=naive evaluation.rolling_origin.step=168",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1.

    Parameters
    ----------
    argv:
        Optional override for ``sys.argv[1:]``.  Passing an explicit list
        lets tests drive the CLI via :func:`subprocess.run` *or* via a
        direct in-process call for speed.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local imports keep ``--help`` lightweight — statsmodels + Hydra are
    # comparatively heavy.
    from bristol_ml.config import load_config
    from bristol_ml.evaluation.benchmarks import compare_on_holdout
    from bristol_ml.evaluation.harness import evaluate
    from bristol_ml.features import assembler
    from bristol_ml.ingestion import neso_forecast as neso_forecast_mod
    from bristol_ml.models.linear import LinearModel
    from bristol_ml.models.naive import NaiveModel
    from conf._schemas import LinearConfig, NaiveConfig

    cfg = load_config(overrides=list(args.overrides))

    fset = cfg.features.weather_only
    split_cfg = cfg.evaluation.rolling_origin
    if fset is None or split_cfg is None:
        print(
            "Required config missing: features.weather_only and "
            "evaluation.rolling_origin must both be populated.",
            file=sys.stderr,
        )
        return 2
    if cfg.model is None:
        print(
            "No model resolved. Ensure `model=linear` or `model=naive` is "
            "in the resolved Hydra composition.",
            file=sys.stderr,
        )
        return 2

    feature_cache = fset.cache_dir / fset.cache_filename
    if not feature_cache.exists():
        print(
            f"Feature-table cache missing at {feature_cache}. Run "
            "`python -m bristol_ml.features.assembler` first.",
            file=sys.stderr,
        )
        return 2

    df = assembler.load(feature_cache).set_index("timestamp_utc")

    model_cfg = cfg.model
    target_column = _target_column(model_cfg)
    if isinstance(model_cfg, NaiveConfig):
        primary = NaiveModel(model_cfg)
        primary_name = "naive"
    elif isinstance(model_cfg, LinearConfig):
        primary = LinearModel(model_cfg)
        primary_name = "linear"
    else:  # pragma: no cover — the discriminated union is exhaustive
        print(
            f"No harness factory for model type {type(model_cfg).__name__!r}.",
            file=sys.stderr,
        )
        return 3

    selected_metric_names = (
        cfg.evaluation.metrics.names
        if cfg.evaluation.metrics is not None
        else tuple(METRIC_REGISTRY)
    )
    metric_fns = [METRIC_REGISTRY[name] for name in selected_metric_names]

    logger.info(
        "Training {}: splits={} target={} metrics={}",
        primary_name,
        split_cfg.model_dump(),
        target_column,
        [m.__name__ for m in metric_fns],
    )

    per_fold = evaluate(
        primary,
        df,
        split_cfg,
        metric_fns,
        target_column=target_column,
    )

    print(f"Per-fold metrics for model={primary_name}:")
    _print_metric_table(per_fold)

    # Three-way benchmark — only if the NESO forecast config *and* its
    # cache are both present.  This keeps the CLI useful offline without
    # a pre-populated forecast cache.
    nfore_cfg = cfg.ingestion.neso_forecast
    if nfore_cfg is None:
        logger.info("NESO forecast config unresolved — skipping benchmark table.")
        return 0

    forecast_cache = nfore_cfg.cache_dir / nfore_cfg.cache_filename
    if not forecast_cache.exists():
        logger.info(
            "NESO forecast cache missing at {} — skipping benchmark table.",
            forecast_cache,
        )
        return 0

    neso_df = neso_forecast_mod.load(forecast_cache)
    # Always instantiate both baseline models for the benchmark table so
    # the three-way comparison is complete regardless of the ``model=``
    # selection.  Per the Stage 4 demo moment: the pedagogical payoff is
    # the full three-way table.
    benchmark_models = {
        "naive": NaiveModel(NaiveConfig()),
        "linear": LinearModel(LinearConfig()),
    }
    aggregation = (
        cfg.evaluation.benchmark.aggregation if cfg.evaluation.benchmark is not None else "mean"
    )
    table = compare_on_holdout(
        benchmark_models,
        df,
        neso_df,
        split_cfg,
        metric_fns,
        aggregation=aggregation,
        target_column=target_column,
    )
    print()
    print("Benchmark comparison (mean across folds; NESO row scored on the same hourly grid):")
    _print_metric_table(table)
    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_metric_table(df: pd.DataFrame) -> None:
    """Print ``df`` to stdout with floats rounded to two decimal places.

    The helper stays deliberately small — the Stage 4 demo moment calls
    for legibility rather than aligned-column heroics; pandas' default
    :meth:`DataFrame.to_string` is already column-aligned.  Using
    :class:`str.format` on the floats via ``float_format`` keeps the
    output free of NumPy repr quirks (scientific notation on very small
    residuals, ``None``-as-``NaN`` formatting).
    """
    print(df.to_string(float_format=lambda v: f"{v:.2f}"))


def _target_column(model_cfg: object) -> str:
    """Return the resolved model's target column (``"nd_mw"`` default)."""
    from conf._schemas import LinearConfig, NaiveConfig

    if isinstance(model_cfg, (NaiveConfig, LinearConfig)):
        return model_cfg.target_column
    return "nd_mw"


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
