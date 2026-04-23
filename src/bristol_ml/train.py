"""End-to-end train + evaluate CLI — Stage 4 demo entry point (Stage 5 extended).

Runnable as ``python -m bristol_ml.train [overrides ...]``.  Wires the
four Stage 4 subsystems into a single invocation:

1. Resolve + validate the Hydra config via
   :func:`bristol_ml.config.load_config`.
2. Load the feature table from its warm parquet cache — either the
   Stage 3 weather-only schema (:func:`bristol_ml.features.assembler.load`)
   or the Stage 5 weather+calendar schema
   (:func:`bristol_ml.features.assembler.load_calendar`), selected by
   ``_resolve_feature_set(cfg)``.
3. Instantiate the resolved model variant — plan D10 wires
   :class:`~bristol_ml.models.naive.NaiveModel` and
   :class:`~bristol_ml.models.linear.LinearModel` behind the Hydra
   ``model=`` group discriminator.
4. Run :func:`bristol_ml.evaluation.harness.evaluate` against the
   rolling-origin splitter config with the resolved feature-column set;
   print the per-fold metric table (header includes the model's
   ``metadata.name`` so the feature-set choice is visible in stdout).
5. If ``ingestion.neso_forecast`` is populated and its cache is warm,
   also run :func:`bristol_ml.evaluation.benchmarks.compare_on_holdout`
   for the three-way NESO comparison and print the resulting table.

The CLI is the demo-moment surface of the stage (intent "Demo moment"):
a single invocation that produces both the harness output and the
three-way benchmark table, with model swap via ``model=naive`` /
``model=linear`` and feature-set swap via ``features=weather_only`` /
``features=weather_calendar`` (Stage 5 T5).  No training loop is
reimplemented here — the work lives in the models, harness, and
benchmarks modules; this file is purely orchestration.

Exit codes::

    0 — success (per-fold table printed; benchmark table printed iff
        the forecast cache was warm).
    2 — required config group or cache missing.
    3 — the resolved model variant has no harness factory.

Running standalone::

    python -m bristol_ml.train --help
    python -m bristol_ml.train                           # default (linear + weather_only)
    python -m bristol_ml.train model=naive               # swap model
    python -m bristol_ml.train features=weather_calendar # swap feature set
    python -m bristol_ml.train evaluation.rolling_origin.step=168
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd
from loguru import logger

from bristol_ml.evaluation.metrics import METRIC_REGISTRY

if TYPE_CHECKING:  # pragma: no cover — typing-only imports
    from conf._schemas import AppConfig, FeatureSetConfig

__all__ = ["_cli_main", "_resolve_feature_set"]


@runtime_checkable
class _LoadFn(Protocol):
    """Structural protocol for a feature-table loader.

    Satisfied by both :func:`bristol_ml.features.assembler.load` and
    :func:`bristol_ml.features.assembler.load_calendar`.  Kept as a
    ``@runtime_checkable`` ``Protocol`` (plan T5 line 322) so the
    resolver's return type narrows statically without inheriting a
    common base.
    """

    def __call__(self, path: Path) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Feature-set resolver (Stage 5 T5)
# ---------------------------------------------------------------------------


def _resolve_feature_set(
    cfg: AppConfig,
) -> tuple[FeatureSetConfig, _LoadFn, tuple[str, ...]]:
    """Pick the populated feature-set config and its loader + column list.

    Plan T5 contract: exactly one of ``cfg.features.weather_only`` /
    ``cfg.features.weather_calendar`` must be populated per run — the
    Hydra group-swap refactor (Stage 5 T1) arranges this at config-
    resolution time.  This function enforces the mutual-exclusivity
    invariant at runtime and returns a tuple of:

    1. The populated :class:`FeatureSetConfig`.
    2. The matching loader — :func:`assembler.load` for the weather-only
       schema, :func:`assembler.load_calendar` for the 55-column
       calendar schema.
    3. The ordered tuple of feature-column names for the set — the five
       Stage 3 weather columns, or the ten-prefix weather columns
       excluding the three provenance scalars and the two
       demand columns, plus the 44 calendar columns.

    Raises
    ------
    ValueError
        If both or neither feature set is populated (the two degenerate
        cases).  The message names the Hydra override the user should
        use to recover.
    """
    from bristol_ml.features import assembler
    from bristol_ml.features.calendar import CALENDAR_VARIABLE_COLUMNS

    weather_only = cfg.features.weather_only
    weather_calendar = cfg.features.weather_calendar

    weather_names = tuple(name for name, _ in assembler.WEATHER_VARIABLE_COLUMNS)
    calendar_names = tuple(name for name, _ in CALENDAR_VARIABLE_COLUMNS)

    if weather_only is not None and weather_calendar is None:
        return (weather_only, assembler.load, weather_names)
    if weather_calendar is not None and weather_only is None:
        return (weather_calendar, assembler.load_calendar, weather_names + calendar_names)
    raise ValueError(
        "Exactly one of features.weather_only or features.weather_calendar must be set; "
        "use 'features=<name>' CLI override (e.g. features=weather_calendar). "
        f"Got: weather_only={'set' if weather_only is not None else 'None'}, "
        f"weather_calendar={'set' if weather_calendar is not None else 'None'}."
    )


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
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help=(
            "Override the Stage 9 registry root (default: data/registry). "
            "Each invocation registers the final-fold fitted model here."
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help=(
            "Skip registry.save after evaluation — handy for notebook "
            "experiments that do not want to accumulate runs on disk."
        ),
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
    from bristol_ml import registry
    from bristol_ml.config import load_config
    from bristol_ml.evaluation.benchmarks import compare_on_holdout
    from bristol_ml.evaluation.harness import evaluate_and_keep_final_model
    from bristol_ml.ingestion import neso_forecast as neso_forecast_mod
    from bristol_ml.models.linear import LinearModel
    from bristol_ml.models.naive import NaiveModel
    from bristol_ml.models.sarimax import SarimaxModel
    from bristol_ml.models.scipy_parametric import ScipyParametricModel
    from conf._schemas import LinearConfig, NaiveConfig, SarimaxConfig, ScipyParametricConfig

    cfg = load_config(overrides=list(args.overrides))

    split_cfg = cfg.evaluation.rolling_origin
    if split_cfg is None:
        print(
            "Required config missing: evaluation.rolling_origin must be populated.",
            file=sys.stderr,
        )
        return 2

    try:
        fset, load_fn, feature_column_names = _resolve_feature_set(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if cfg.model is None:
        print(
            "No model resolved. Ensure `model=linear`, `model=naive`, "
            "`model=sarimax`, or `model=scipy_parametric` is in the "
            "resolved Hydra composition.",
            file=sys.stderr,
        )
        return 2

    feature_cache = fset.cache_dir / fset.cache_filename
    if not feature_cache.exists():
        print(
            f"Feature-table cache missing at {feature_cache}. Run "
            f"`python -m bristol_ml.features.assembler` (or "
            f"`assemble_calendar` for the calendar set) first.",
            file=sys.stderr,
        )
        return 2

    df = load_fn(feature_cache).set_index("timestamp_utc")

    model_cfg = cfg.model
    target_column = _target_column(model_cfg)
    if isinstance(model_cfg, NaiveConfig):
        primary = NaiveModel(model_cfg)
        primary_kind = "naive"
    elif isinstance(model_cfg, LinearConfig):
        # Override LinearConfig.feature_columns to carry the resolved set so
        # LinearModel fits the full weather+calendar regressor stack when
        # features=weather_calendar.  If the user already pinned
        # feature_columns via Hydra, respect their choice and log the
        # override (plan T5 line 325).
        if model_cfg.feature_columns is None:
            linear_cfg = model_cfg.model_copy(update={"feature_columns": feature_column_names})
        else:
            logger.info(
                "LinearConfig.feature_columns explicitly set via Hydra "
                "({} columns); not overriding with resolved feature-set columns.",
                len(model_cfg.feature_columns),
            )
            linear_cfg = model_cfg
        primary = _NamedLinearModel(
            linear_cfg,
            metadata_name=f"linear-ols-{fset.name.replace('_', '-')}",
        )
        primary_kind = "linear"
    elif isinstance(model_cfg, SarimaxConfig):
        # Mirror the LinearConfig branch: promote feature_columns=None to
        # the resolved feature-set tuple so ``SarimaxModel._config.feature_columns``
        # records exactly which columns the fit saw.  Without this promotion
        # the stored config says ``None`` while the fitted metadata carries
        # a resolved tuple — reproducibility from config alone is lost
        # (Stage 7 Phase 3 review R3).
        if model_cfg.feature_columns is None:
            sarimax_cfg = model_cfg.model_copy(update={"feature_columns": feature_column_names})
        else:
            logger.info(
                "SarimaxConfig.feature_columns explicitly set via Hydra "
                "({} columns); not overriding with resolved feature-set columns.",
                len(model_cfg.feature_columns),
            )
            sarimax_cfg = model_cfg
        primary = SarimaxModel(sarimax_cfg)
        primary_kind = "sarimax"
    elif isinstance(model_cfg, ScipyParametricConfig):
        # Stage 8: the parametric model's ``feature_columns`` field
        # constrains *Fourier* columns generated inside the model, not
        # the raw input columns (unlike Linear / SARIMAX).  So there is
        # no resolved-feature-set promotion to mirror here — leave the
        # Hydra-resolved config unmodified.  The harness's
        # ``feature_columns=`` kwarg below still slices the raw frame
        # down to ``feature_column_names`` (weather or weather+calendar
        # per the active feature set), and the parametric model reads
        # its temperature column from that slice.
        primary = ScipyParametricModel(model_cfg)
        primary_kind = "scipy_parametric"
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
        "Training {}: feature_set={} splits={} target={} metrics={}",
        primary_kind,
        fset.name,
        split_cfg.model_dump(),
        target_column,
        [m.__name__ for m in metric_fns],
    )

    # Stage 9 D17: keep the final-fold fitted model so we can register it
    # without re-fitting on the full training set (AC-2 + plan R2).  The
    # harness wraps ``evaluate`` and returns (metrics_df, fitted_model) in
    # one call — no second boolean flag was added to ``evaluate`` (H5
    # API-growth rule in ``evaluation/CLAUDE.md``).
    per_fold, primary = evaluate_and_keep_final_model(
        primary,
        df,
        split_cfg,
        metric_fns,
        target_column=target_column,
        feature_columns=feature_column_names,
    )

    # Header carries both the coarse model kind and the full metadata.name
    # so the feature-set selection is greppable in the demo-moment stdout
    # (plan T5 test ``test_train_cli_features_override_swaps_feature_set``
    # asserts on ``"weather-calendar"`` appearing here).
    print(
        f"Per-fold metrics for model={primary_kind} "
        f"({primary.metadata.name}, feature_set={fset.name}):"
    )
    _print_metric_table(per_fold)

    # Stage 9 D17: register the final-fold model.  Gated behind
    # ``--no-register`` for notebook workflows that don't want to
    # accumulate runs on disk.  The run_id is printed so the facilitator
    # can feed it straight into ``python -m bristol_ml.registry describe``.
    #
    # Stage 9 Phase 3 review B1: failure to save is treated as
    # catastrophic — the whole point of the Demo moment is an on-disk
    # artefact, so a silent warning here would let a CI pipeline pass
    # while the registry is empty.  We let the exception propagate so
    # ``_cli_main`` returns a non-zero exit code.
    if not args.no_register and primary.metadata.fit_utc is not None:
        run_id = registry.save(
            primary,
            per_fold,
            feature_set=fset.name,
            target=target_column,
            registry_dir=args.registry_dir,
        )
        print(f"Registered run_id: {run_id}")

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
    # selection.  Under the calendar feature set the linear benchmark
    # model also uses the calendar regressors — otherwise the comparison
    # would silently fall back to five weather columns on a 55-column
    # frame (plan T5 line 324).
    benchmark_linear_cfg = LinearConfig(feature_columns=feature_column_names)
    benchmark_models = {
        "naive": NaiveModel(NaiveConfig()),
        "linear": LinearModel(benchmark_linear_cfg),
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
        feature_columns=feature_column_names,
    )
    print()
    print("Benchmark comparison (mean across folds; NESO row scored on the same hourly grid):")
    _print_metric_table(table)
    return 0


# ---------------------------------------------------------------------------
# Internal: named LinearModel subclass (Stage 5 T5)
# ---------------------------------------------------------------------------


class _NamedLinearModel:
    """Thin wrapper that makes ``metadata.name`` reflect the feature set.

    Plan T5 line 325 says the fitted model's ``metadata.name`` should read
    ``"linear-ols-weather-only"`` vs ``"linear-ols-weather-calendar"``.
    :class:`~bristol_ml.models.linear.LinearModel` hard-codes the former;
    the plan directs us to "keep ``LinearModel`` unchanged".  This wrapper
    composes a :class:`LinearModel` and forwards every protocol member
    while overriding :attr:`metadata` to substitute the dynamic name.

    Implements the :class:`~bristol_ml.models.protocol.Model` protocol
    structurally — ``@runtime_checkable`` ``isinstance`` checks pass.
    """

    def __init__(self, config: object, *, metadata_name: str) -> None:
        from bristol_ml.models.linear import LinearModel

        self._inner = LinearModel(config)  # type: ignore[arg-type]
        self._metadata_name = metadata_name

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _NamedLinearModel:
        self._inner.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self._inner.predict(features)

    def save(self, path: Path) -> None:
        self._inner.save(path)

    @classmethod
    def load(cls, path: Path) -> _NamedLinearModel:
        # Never part of the train-CLI path; the harness only calls
        # fit/predict.  A direct load would lose the metadata name
        # override — surfaced here for protocol completeness.
        raise NotImplementedError(
            "_NamedLinearModel.load is not supported; load via LinearModel.load "
            "and re-wrap with the appropriate metadata_name."
        )

    @property
    def metadata(self):  # type: ignore[no-untyped-def]
        base = self._inner.metadata
        return base.model_copy(update={"name": self._metadata_name})

    @property
    def results(self):  # type: ignore[no-untyped-def]
        """Expose the inner LinearModel's statsmodels results wrapper."""
        return self._inner.results


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
    from conf._schemas import LinearConfig, NaiveConfig, SarimaxConfig, ScipyParametricConfig

    if isinstance(model_cfg, (NaiveConfig, LinearConfig, SarimaxConfig, ScipyParametricConfig)):
        return model_cfg.target_column
    return "nd_mw"


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
