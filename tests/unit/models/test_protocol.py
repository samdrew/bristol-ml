"""Spec-derived tests for ``bristol_ml.models.protocol`` and the lazy re-export layer.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T2 (acceptance criteria
  and named test list).
- ``docs/plans/active/04-linear-baseline.md`` §4 AC-2: the interface must be
  implementable in very few lines of code.
- ``docs/plans/active/04-linear-baseline.md`` §1 D3: ``@runtime_checkable``
  ``typing.Protocol``; the PEP 544 caveat is a documented teaching moment.
- ``src/bristol_ml/models/protocol.py`` module docstring (caveat + re-export note).
- ``src/bristol_ml/models/__init__.py`` lazy re-export contract.

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan AC or D-number it guards.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bristol_ml.models.protocol import Model, ModelMetadata

# ---------------------------------------------------------------------------
# Helpers — minimal dummy classes
# ---------------------------------------------------------------------------


class _FullDummy:
    """A minimal dummy that exposes all five required ``Model`` members.

    Used to verify that ``@runtime_checkable`` attribute-presence checks pass
    for any object that provides the requisite names.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(dtype="float64")

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> _FullDummy:
        return cls()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="dummy", feature_columns=())


class _MissingPredict:
    """A dummy that exposes four of the five required ``Model`` members,
    omitting ``predict``; used to verify the rejection path.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> _MissingPredict:
        return cls()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="broken", feature_columns=())


class _WrongArityFit:
    """A dummy whose ``fit`` takes wrong arity (only ``self``) but still
    exposes all five attribute names.

    Used for the PEP 544 teaching test: ``isinstance`` passes despite the
    signature mismatch.
    """

    def fit(self) -> None:  # type: ignore[override]  # intentionally wrong arity for caveat test
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(dtype="float64")

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> _WrongArityFit:
        return cls()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="wrong-arity", feature_columns=())


# ---------------------------------------------------------------------------
# AC-2 / AC-7 — ``@runtime_checkable`` positive path
# ---------------------------------------------------------------------------


def test_model_protocol_is_runtime_checkable() -> None:
    """Guards T2 AC-2 / AC-7: a complete implementor passes ``isinstance(x, Model)``.

    Verifies that the ``@runtime_checkable`` decorator on ``Model`` enables
    ``isinstance`` to return ``True`` for any object that exposes all five
    required attributes (``fit``, ``predict``, ``save``, ``load``,
    ``metadata``).

    The check is performed on an *instance*, not the class itself — the
    runtime-checkable protocol inspects attribute presence on the object,
    including the ``load`` classmethod and the ``metadata`` property, both
    of which are accessible on an instance.

    Guards T2 AC-2 / AC-7.
    """
    instance = _FullDummy()
    assert isinstance(instance, Model), (
        "A class with all five Model members (fit, predict, save, load, metadata) "
        "must pass isinstance(x, Model) when the protocol is @runtime_checkable "
        "(T2 AC-2 / AC-7 / D3)."
    )


# ---------------------------------------------------------------------------
# AC-7 — rejection of incomplete implementors
# ---------------------------------------------------------------------------


def test_model_protocol_rejects_missing_method() -> None:
    """Guards T2 AC-7: a class missing ``predict`` fails ``isinstance``.

    The ``@runtime_checkable`` check verifies *all* declared protocol members
    are present on the instance.  Removing even one (here: ``predict``) must
    cause ``isinstance`` to return ``False``.

    Guards T2 AC-7 (protocol-conformance test exists for both models).
    """
    instance = _MissingPredict()
    assert not isinstance(instance, Model), (
        "A class missing 'predict' must fail isinstance(x, Model) (T2 AC-7 / D3 negative path)."
    )


# ---------------------------------------------------------------------------
# D3 — PEP 544 caveat (teaching test — pin documented behaviour, do not hide it)
# ---------------------------------------------------------------------------


def test_model_protocol_caveat_signatures_not_checked() -> None:
    """Guards D3 PEP 544 caveat: wrong-arity ``fit`` still passes ``isinstance``.

    This is a deliberate *teaching test*, as noted in
    ``src/bristol_ml/models/protocol.py``'s module docstring.

    ``@runtime_checkable`` only checks that the named attributes *exist* on the
    object — it does **not** verify their call signatures.  A ``fit`` that
    accepts only ``(self,)`` instead of the required
    ``(self, features, target)`` will still satisfy ``isinstance(x, Model)``
    at runtime.  Static type checkers (mypy / pyright) catch this at
    development time; runtime ``isinstance`` alone does not.

    This test pins the documented behaviour rather than hiding it, because the
    plan (D3) explicitly calls the caveat "a worthwhile teaching moment".
    Removing or weakening this test would obscure a real limitation that every
    Stage 4+ implementor should understand.

    Guards D3 (PEP 544 structural subtyping caveat).
    """
    instance = _WrongArityFit()
    # This MUST pass isinstance — the caveat is that signatures are not checked.
    assert isinstance(instance, Model), (
        "PEP 544 caveat: @runtime_checkable checks attribute *presence* only, "
        "not signatures.  A wrong-arity fit must still pass isinstance(x, Model). "
        "This is the documented teaching moment from D3 / protocol.py module docstring."
    )


# ---------------------------------------------------------------------------
# ModelMetadata re-export from protocol module
# ---------------------------------------------------------------------------


def test_model_metadata_is_reexported_from_protocol_module() -> None:
    """Guards T2: ``ModelMetadata`` is re-exported from ``bristol_ml.models.protocol``.

    The ``protocol.py`` module docstring states that ``ModelMetadata`` is
    re-exported there "for ergonomic notebook/CLI use".  This confirms that
    the import path ``from bristol_ml.models.protocol import ModelMetadata``
    resolves to the canonical ``conf._schemas.ModelMetadata`` so callers do
    not need to import from two different places.

    Guards T2 (``ModelMetadata`` re-export contract).
    """
    from bristol_ml.models.protocol import ModelMetadata as _ProtocolMeta
    from conf._schemas import ModelMetadata as _SchemaMeta

    assert _ProtocolMeta is _SchemaMeta, (
        "bristol_ml.models.protocol.ModelMetadata must be the same object as "
        "conf._schemas.ModelMetadata (re-export contract, T2)."
    )


# ---------------------------------------------------------------------------
# Lazy re-export from the package namespace
# ---------------------------------------------------------------------------


def test_model_and_metadata_available_via_lazy_reexport() -> None:
    """Guards T2: ``from bristol_ml.models import Model, ModelMetadata`` resolves.

    The lazy ``__getattr__`` in ``bristol_ml.models.__init__`` must make both
    ``Model`` and ``ModelMetadata`` importable at the package level and must
    return the same objects as their submodule originals.

    Guards T2 (lazy re-export contract; ``__init__.__all__`` surface).
    """
    from bristol_ml.models import Model as _PackageModel
    from bristol_ml.models import ModelMetadata as _PackageMeta
    from bristol_ml.models.protocol import Model as _ProtoModel
    from bristol_ml.models.protocol import ModelMetadata as _ProtoMeta

    assert _PackageModel is _ProtoModel, (
        "bristol_ml.models.Model must be the same object as "
        "bristol_ml.models.protocol.Model (lazy re-export, T2)."
    )
    assert _PackageMeta is _ProtoMeta, (
        "bristol_ml.models.ModelMetadata must be the same object as "
        "bristol_ml.models.protocol.ModelMetadata (lazy re-export, T2)."
    )


def test_lazy_reexport_raises_attribute_error_for_unknown_name() -> None:
    """Guards T2: the lazy ``__getattr__`` raises ``AttributeError`` for unknown names.

    The ``__getattr__`` in ``bristol_ml.models.__init__`` must not silently
    return ``None`` or swallow unknown attribute lookups.  Any name not in
    the declared ``__all__`` must raise ``AttributeError`` with a message
    that names the attribute, matching standard Python module behaviour.

    Guards T2 (``__getattr__`` fail-closed contract).
    """
    import bristol_ml.models as _models

    with pytest.raises(AttributeError, match="definitely_not_exported"):
        getattr(_models, "definitely_not_exported")  # noqa: B009 — exercise module __getattr__
