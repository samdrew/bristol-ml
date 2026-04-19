"""Models layer — the ``Model`` protocol, provenance metadata, and IO helpers.

Stage 4 introduces this module with the :class:`~bristol_ml.models.protocol.Model`
runtime-checkable protocol (the contract every subsequent modelling stage
implements) plus joblib-backed save/load helpers.  Stages 4+ add concrete
models (``NaiveModel``, ``LinearModel``) in their own submodules.

Submodules are not imported eagerly so ``python -m bristol_ml`` (scaffold
invocation) stays cheap — matches the ``bristol_ml.evaluation`` pattern.
Import by name::

    from bristol_ml.models import protocol
    from bristol_ml.models.naive import NaiveModel

or resolve a top-level alias lazily via ``__getattr__``::

    from bristol_ml.models import Model, ModelMetadata

The lazy-re-export list mirrors what notebooks and the ``train`` CLI need at
their fingertips; concrete model classes are imported from their submodules to
avoid pulling heavy dependencies (statsmodels) into every ``bristol_ml``
import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.models.io import load_joblib, save_joblib
    from bristol_ml.models.protocol import Model, ModelMetadata

__all__ = [
    "Model",
    "ModelMetadata",
    "load_joblib",
    "save_joblib",
]


def __getattr__(name: str) -> object:
    """Lazy re-export of the protocol + metadata + io helpers."""
    if name in {"Model", "ModelMetadata"}:
        from bristol_ml.models import protocol as _protocol

        return getattr(_protocol, name)
    if name in {"load_joblib", "save_joblib"}:
        from bristol_ml.models import io as _io

        return getattr(_io, name)
    raise AttributeError(f"module 'bristol_ml.models' has no attribute {name!r}")
