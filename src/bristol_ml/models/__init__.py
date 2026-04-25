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
    from bristol_ml.models.io import (
        UntrustedTypeError,
        load_joblib,
        load_skops,
        register_safe_types,
        save_joblib,
        save_skops,
    )
    from bristol_ml.models.linear import LinearModel
    from bristol_ml.models.naive import NaiveModel
    from bristol_ml.models.nn.mlp import NnMlpModel
    from bristol_ml.models.protocol import Model, ModelMetadata
    from bristol_ml.models.sarimax import SarimaxModel
    from bristol_ml.models.scipy_parametric import ScipyParametricModel
    from conf._schemas import (
        LinearConfig,
        NaiveConfig,
        NnMlpConfig,
        SarimaxConfig,
        ScipyParametricConfig,
    )

__all__ = [
    "LinearConfig",
    "LinearModel",
    "Model",
    "ModelMetadata",
    "NaiveConfig",
    "NaiveModel",
    "NnMlpConfig",
    "NnMlpModel",
    "SarimaxConfig",
    "SarimaxModel",
    "ScipyParametricConfig",
    "ScipyParametricModel",
    "UntrustedTypeError",
    "load_joblib",
    "load_skops",
    "register_safe_types",
    "save_joblib",
    "save_skops",
]


def __getattr__(name: str) -> object:
    """Lazy re-export of the protocol + metadata + io helpers + concrete models.

    Concrete model classes are pulled from their submodules on first
    access so ``python -m bristol_ml`` (scaffold invocation) does not
    pay the statsmodels import cost just to print version info.
    """
    if name in {"Model", "ModelMetadata"}:
        from bristol_ml.models import protocol as _protocol

        return getattr(_protocol, name)
    if name in {
        "UntrustedTypeError",
        "load_joblib",
        "load_skops",
        "register_safe_types",
        "save_joblib",
        "save_skops",
    }:
        from bristol_ml.models import io as _io

        return getattr(_io, name)
    if name == "NaiveModel":
        from bristol_ml.models.naive import NaiveModel

        return NaiveModel
    if name == "LinearModel":
        from bristol_ml.models.linear import LinearModel

        return LinearModel
    if name == "SarimaxModel":
        from bristol_ml.models.sarimax import SarimaxModel

        return SarimaxModel
    if name == "ScipyParametricModel":
        from bristol_ml.models.scipy_parametric import ScipyParametricModel

        return ScipyParametricModel
    if name == "NnMlpModel":
        # Stage 10: the ``nn`` sub-package's lazy re-export keeps ``torch``
        # out of the import graph until this attribute is actually resolved.
        from bristol_ml.models.nn.mlp import NnMlpModel

        return NnMlpModel
    if name in {
        "NaiveConfig",
        "LinearConfig",
        "SarimaxConfig",
        "ScipyParametricConfig",
        "NnMlpConfig",
    }:
        from conf import _schemas

        return getattr(_schemas, name)
    raise AttributeError(f"module 'bristol_ml.models' has no attribute {name!r}")
