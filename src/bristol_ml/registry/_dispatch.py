"""Local class ↔ type-string dispatch for the registry (Stage 9 D9).

The registry needs two inverse look-ups on the four Stage-4/7/8 model
classes:

- **Save time.**  Map a ``Model`` instance to the sidecar ``type`` string.
- **Load time.**  Map a sidecar ``type`` string back to the concrete
  class's ``load`` classmethod.

Both live here as module-level dicts.  Codebase-map hazard H4 warns
against adding a *third* dispatcher site alongside the two in
``train.py`` and the benchmark harness; this module is a *local*
dispatcher internal to the registry, not a shared helper.  If either
direction grows new keys in later stages, promote both pairs to a
shared ADR (Housekeeping carry-over H-4) rather than accreting a
third dispatcher.

``_NamedLinearModel`` — the ``train.py`` internal wrapper introduced in
Stage 5 to override ``metadata.name`` — maps to ``"linear"`` for save
purposes.  Per plan D16 (cut) the registry's ``load()`` returns a base
``LinearModel``; the sidecar's ``name`` field preserves the dynamic
name for reading but is not re-applied to the loaded instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from bristol_ml.models.nn.mlp import NnMlpModel
from bristol_ml.models.sarimax import SarimaxModel
from bristol_ml.models.scipy_parametric import ScipyParametricModel

if TYPE_CHECKING:  # pragma: no cover — typing only
    from bristol_ml.models.protocol import Model


#: Sidecar ``type`` string → concrete ``Model`` class.  Consumed by
#: :func:`bristol_ml.registry.load` (T3).  Keys are the canonical short
#: names used in both the sidecar JSON and the CLI filters.
_TYPE_TO_CLASS: dict[str, type] = {
    "naive": NaiveModel,
    "linear": LinearModel,
    "sarimax": SarimaxModel,
    "scipy_parametric": ScipyParametricModel,
    "nn_mlp": NnMlpModel,
}

#: Class name → sidecar ``type`` string.  Keyed on ``type(model).__name__``
#: rather than on the class objects themselves so ``_NamedLinearModel``
#: (defined in ``train.py`` and not importable here without a
#: circular-import hazard) can be handled via its class name.
_CLASS_NAME_TO_TYPE: dict[str, str] = {
    "NaiveModel": "naive",
    "LinearModel": "linear",
    # train.py internal wrapper (Stage 5 T5).  Registry load returns base
    # LinearModel — plan D16 cut the dynamic-name round-trip.
    "_NamedLinearModel": "linear",
    "SarimaxModel": "sarimax",
    "ScipyParametricModel": "scipy_parametric",
    "NnMlpModel": "nn_mlp",
}


def model_type(model: Model) -> str:
    """Return the sidecar ``type`` string for a fitted ``Model`` instance.

    Raises
    ------
    TypeError
        If ``model``'s class is not recognised.  Adding a new model
        family means adding one key to
        :data:`_CLASS_NAME_TO_TYPE` and one key to
        :data:`_TYPE_TO_CLASS` — fail loudly rather than silently
        registering a run under an unknown type.
    """
    name = type(model).__name__
    try:
        return _CLASS_NAME_TO_TYPE[name]
    except KeyError as exc:
        raise TypeError(
            f"Cannot register a {name} via the registry; expected one of "
            f"{sorted(_CLASS_NAME_TO_TYPE)}. Register new model classes in "
            "bristol_ml.registry._dispatch before calling save()."
        ) from exc


def class_for_type(type_str: str) -> type:
    """Return the concrete class for a sidecar ``type`` string.

    Raises
    ------
    ValueError
        If ``type_str`` is not one of the four registered model types.
    """
    try:
        return _TYPE_TO_CLASS[type_str]
    except KeyError as exc:
        raise ValueError(
            f"Unknown registry run type {type_str!r}; expected one of {sorted(_TYPE_TO_CLASS)}."
        ) from exc
