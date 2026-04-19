"""Evaluation layer — splitters, metrics, and evaluators.

Stage 3 shipped this module with the rolling-origin splitter; Stage 4
extends it with metric functions (MAE, MAPE, RMSE, WAPE) and the
evaluator harness that consumes the splitter's fold indices.

Submodules are not imported eagerly so ``python -m bristol_ml`` (scaffold
invocation) stays cheap. Import by name::

    from bristol_ml.evaluation import splitter, metrics, harness

or resolve a top-level alias lazily via ``__getattr__``::

    from bristol_ml.evaluation import rolling_origin_split
    from bristol_ml.evaluation import rolling_origin_split_from_config
    from bristol_ml.evaluation import mae, mape, rmse, wape, METRIC_REGISTRY
    from bristol_ml.evaluation import evaluate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.harness import evaluate
    from bristol_ml.evaluation.metrics import (
        METRIC_REGISTRY,
        MetricFn,
        mae,
        mape,
        rmse,
        wape,
    )
    from bristol_ml.evaluation.splitter import (
        rolling_origin_split,
        rolling_origin_split_from_config,
    )

__all__ = [
    "METRIC_REGISTRY",
    "MetricFn",
    "evaluate",
    "mae",
    "mape",
    "rmse",
    "rolling_origin_split",
    "rolling_origin_split_from_config",
    "wape",
]

_SPLITTER_NAMES = frozenset({"rolling_origin_split", "rolling_origin_split_from_config"})
_METRIC_NAMES = frozenset({"METRIC_REGISTRY", "MetricFn", "mae", "mape", "rmse", "wape"})
_HARNESS_NAMES = frozenset({"evaluate"})


def __getattr__(name: str) -> object:
    """Lazy re-export of public splitter, metric, and harness symbols."""
    if name in _SPLITTER_NAMES:
        from bristol_ml.evaluation import splitter as _splitter

        return getattr(_splitter, name)
    if name in _METRIC_NAMES:
        from bristol_ml.evaluation import metrics as _metrics

        return getattr(_metrics, name)
    if name in _HARNESS_NAMES:
        from bristol_ml.evaluation import harness as _harness

        return getattr(_harness, name)
    raise AttributeError(f"module 'bristol_ml.evaluation' has no attribute {name!r}")
