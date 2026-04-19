"""Evaluation layer — splitters, metrics, and evaluators.

Stage 3 ships this module with the rolling-origin splitter only; Stage 4
extends it with metric functions (MAE, MAPE, RMSE, WAPE) and the evaluator
harness that consumes the splitter's fold indices.

Submodules are not imported eagerly so ``python -m bristol_ml`` (scaffold
invocation) stays cheap. Import by name::

    from bristol_ml.evaluation import splitter

or resolve a top-level alias lazily via ``__getattr__``::

    from bristol_ml.evaluation import rolling_origin_split
    from bristol_ml.evaluation import rolling_origin_split_from_config
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.splitter import (
        rolling_origin_split,
        rolling_origin_split_from_config,
    )

__all__ = ["rolling_origin_split", "rolling_origin_split_from_config"]


def __getattr__(name: str) -> object:
    """Lazy re-export of public splitter symbols from the splitter submodule."""
    if name in {"rolling_origin_split", "rolling_origin_split_from_config"}:
        from bristol_ml.evaluation import splitter as _splitter

        return getattr(_splitter, name)
    raise AttributeError(f"module 'bristol_ml.evaluation' has no attribute {name!r}")
