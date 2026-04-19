"""The ``Model`` protocol every Stage 4+ estimator implements.

Per plan D3 the interface mechanism is :class:`typing.Protocol` decorated
``@runtime_checkable``, matching the ``DESIGN`` §7.3 sketch verbatim.  A
``Protocol`` rather than an abstract base class means:

- Concrete models are plain classes; no mandatory inheritance chain.  This
  keeps the naive model, in particular, implementable in very few lines of
  code (plan AC-2 / intent AC-2).
- ``isinstance(m, Model)`` still evaluates (the ``@runtime_checkable``
  decorator wires up attribute-presence checks).  The protocol-conformance
  tests in ``tests/unit/models/test_protocol.py`` rely on this.

**Caveat (teaching moment).**  ``@runtime_checkable`` only verifies that the
object exposes the named attributes — it does *not* verify callable
signatures.  A mis-typed ``fit`` (say, taking only one argument) will still
satisfy ``isinstance(...)``.  Static type checkers (mypy / pyright) catch
the signature mismatch; runtime ``isinstance`` alone does not.  This is the
price of PEP 544 structural subtyping; keep it visible rather than hidden.

The matching ``ModelMetadata`` Pydantic model lives in ``conf._schemas`` so
that Hydra's resolve-then-validate pipeline can (eventually, Stage 9) load
and validate a serialised metadata record without circular-import gymnastics.
It is re-exported here for ergonomic notebook/CLI use.

See :mod:`bristol_ml.models.io` for the joblib save/load helpers the
protocol's ``save``/``load`` methods delegate to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from conf._schemas import ModelMetadata

__all__ = ["Model", "ModelMetadata"]


@runtime_checkable
class Model(Protocol):
    """Structural protocol for every Stage 4+ estimator.

    Every implementor must expose the five members below.  The protocol is
    deliberately small: it mirrors ``DESIGN`` §7.3 and keeps the naive
    baseline implementable in a handful of lines.

    Implementors:

    - Store any config (``NaiveConfig``, ``LinearConfig``) on ``__init__``;
      do not mutate it.
    - ``fit`` must be re-entrant — calling it a second time discards the
      previous fit rather than layering state on top (plan §10 risk row
      "save/load round-trip flakes because subsequent ``fit()`` mutates
      state").
    - ``predict`` must accept any DataFrame whose columns are a superset of
      the features the model was fit on, and return a ``pd.Series`` indexed
      to the input's index.
    - ``save``/``load`` delegate to ``bristol_ml.models.io`` (joblib, atomic
      write) per plan D6.  A model saved with version *X* should load with
      the same version; cross-version compatibility is an explicit non-goal
      at Stage 4 (Stage 9 registry owns that story).
    - ``metadata`` returns a fully-populated :class:`ModelMetadata` after
      ``fit``.  Before ``fit`` callers may observe ``fit_utc is None``.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit the estimator on the aligned ``(features, target)`` pair."""
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return a ``pd.Series`` of predictions indexed to ``features.index``."""
        ...

    def save(self, path: Path) -> None:
        """Serialise the fitted model to ``path`` atomically."""
        ...

    @classmethod
    def load(cls, path: Path) -> Model:
        """Load a previously-saved model instance from ``path``."""
        ...

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit."""
        ...
