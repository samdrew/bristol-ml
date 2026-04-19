"""Evaluation layer — splitters, metrics, and evaluators.

Stage 3 ships this module with the rolling-origin splitter only; Stage 4
extends it with metric functions (MAE, MAPE, RMSE, WAPE) and the evaluator
harness that consumes the splitter's fold indices.

Submodules are not imported eagerly so ``python -m bristol_ml`` (scaffold
invocation) stays cheap. Import by name::

    from bristol_ml.evaluation import splitter

or resolve the top-level alias lazily via ``__getattr__``::

    from bristol_ml.evaluation import rolling_origin_split
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.splitter import rolling_origin_split

__all__ = ["rolling_origin_split"]


def __getattr__(name: str) -> object:
    """Lazy re-export of ``rolling_origin_split`` from the splitter submodule."""
    if name == "rolling_origin_split":
        from bristol_ml.evaluation import splitter as _splitter

        return _splitter.rolling_origin_split
    raise AttributeError(f"module 'bristol_ml.evaluation' has no attribute {name!r}")
