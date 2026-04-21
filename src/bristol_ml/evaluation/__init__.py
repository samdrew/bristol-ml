"""Evaluation layer — splitters, metrics, evaluators, benchmarks, and plots.

Stage 3 shipped this module with the rolling-origin splitter; Stage 4
extended it with metric functions (MAE, MAPE, RMSE, WAPE), the evaluator
harness that consumes the splitter's fold indices, and the three-way
NESO day-ahead benchmark comparison.  Stage 6 adds ``plots`` — a library
of colourblind-safe diagnostic-plot helpers (residuals, ACF, forecast
overlays, uncertainty bands, and a fixed-window NESO bar chart).

Submodules are not imported eagerly so ``python -m bristol_ml`` (scaffold
invocation) stays cheap. Import by name::

    from bristol_ml.evaluation import splitter, metrics, harness, benchmarks, plots

or resolve a top-level alias lazily via ``__getattr__``::

    from bristol_ml.evaluation import rolling_origin_split
    from bristol_ml.evaluation import rolling_origin_split_from_config
    from bristol_ml.evaluation import mae, mape, rmse, wape, METRIC_REGISTRY
    from bristol_ml.evaluation import evaluate
    from bristol_ml.evaluation import align_half_hourly_to_hourly, compare_on_holdout
    from bristol_ml.evaluation import (
        residuals_vs_time,
        predicted_vs_actual,
        acf_residuals,
        error_heatmap_hour_weekday,
        forecast_overlay,
        forecast_overlay_with_band,
        benchmark_holdout_bar,
        OKABE_ITO,
        SEQUENTIAL_CMAP,
        DIVERGING_CMAP,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.benchmarks import (
        align_half_hourly_to_hourly,
        compare_on_holdout,
    )
    from bristol_ml.evaluation.harness import evaluate
    from bristol_ml.evaluation.metrics import (
        METRIC_REGISTRY,
        MetricFn,
        mae,
        mape,
        rmse,
        wape,
    )
    from bristol_ml.evaluation.plots import (
        DIVERGING_CMAP,
        OKABE_ITO,
        SEQUENTIAL_CMAP,
        acf_residuals,
        benchmark_holdout_bar,
        error_heatmap_hour_weekday,
        forecast_overlay,
        forecast_overlay_with_band,
        predicted_vs_actual,
        residuals_vs_time,
    )
    from bristol_ml.evaluation.splitter import (
        rolling_origin_split,
        rolling_origin_split_from_config,
    )

__all__ = [
    "DIVERGING_CMAP",
    "METRIC_REGISTRY",
    "OKABE_ITO",
    "SEQUENTIAL_CMAP",
    "MetricFn",
    "acf_residuals",
    "align_half_hourly_to_hourly",
    "benchmark_holdout_bar",
    "compare_on_holdout",
    "error_heatmap_hour_weekday",
    "evaluate",
    "forecast_overlay",
    "forecast_overlay_with_band",
    "mae",
    "mape",
    "predicted_vs_actual",
    "residuals_vs_time",
    "rmse",
    "rolling_origin_split",
    "rolling_origin_split_from_config",
    "wape",
]

_SPLITTER_NAMES = frozenset({"rolling_origin_split", "rolling_origin_split_from_config"})
_METRIC_NAMES = frozenset({"METRIC_REGISTRY", "MetricFn", "mae", "mape", "rmse", "wape"})
_HARNESS_NAMES = frozenset({"evaluate"})
_BENCHMARK_NAMES = frozenset({"align_half_hourly_to_hourly", "compare_on_holdout"})
_PLOTS_NAMES = frozenset(
    {
        "DIVERGING_CMAP",
        "OKABE_ITO",
        "SEQUENTIAL_CMAP",
        "acf_residuals",
        "benchmark_holdout_bar",
        "error_heatmap_hour_weekday",
        "forecast_overlay",
        "forecast_overlay_with_band",
        "predicted_vs_actual",
        "residuals_vs_time",
    }
)


def __getattr__(name: str) -> object:
    """Lazy re-export of public splitter, metric, harness, benchmark, and plots symbols."""
    if name in _SPLITTER_NAMES:
        from bristol_ml.evaluation import splitter as _splitter

        return getattr(_splitter, name)
    if name in _METRIC_NAMES:
        from bristol_ml.evaluation import metrics as _metrics

        return getattr(_metrics, name)
    if name in _HARNESS_NAMES:
        from bristol_ml.evaluation import harness as _harness

        return getattr(_harness, name)
    if name in _BENCHMARK_NAMES:
        from bristol_ml.evaluation import benchmarks as _benchmarks

        return getattr(_benchmarks, name)
    if name in _PLOTS_NAMES:
        from bristol_ml.evaluation import plots as _plots

        return getattr(_plots, name)
    raise AttributeError(f"module 'bristol_ml.evaluation' has no attribute {name!r}")
