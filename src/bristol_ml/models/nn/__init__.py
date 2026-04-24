"""Neural-network models sub-layer — Stage 10 onwards.

Stage 10 introduces this sub-package with a single concrete model
(:class:`~bristol_ml.models.nn.mlp.NnMlpModel`) — a small MLP that
conforms to the Stage 4 :class:`bristol_ml.models.Model` protocol.
Stage 11's temporal architectures (RNN / CNN / transformer) will land
here alongside it, sharing the hand-rolled training-loop conventions
Stage 10 establishes (plan D10 extraction seam).

The sub-package is structured so ``python -m bristol_ml.models.nn.mlp``
prints the config schema (DESIGN §2.1.1 / plan NFR-6) and so that
``from bristol_ml.models.nn import NnMlpModel`` works without pulling
``torch`` into the import graph of modules that do not need it — the
concrete class is resolved lazily through ``__getattr__`` below,
mirroring the top-level :mod:`bristol_ml.models` re-export pattern.

Cross-references:

- Layer doc: ``docs/architecture/layers/models-nn.md`` (Stage 10 T6).
- Module guide: ``src/bristol_ml/models/nn/CLAUDE.md`` (Stage 10 T6).
- Plan: ``docs/plans/active/10-simple-nn.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.models.nn.mlp import NnMlpModel

__all__ = ["NnMlpModel"]


def __getattr__(name: str) -> object:
    """Lazy re-export of :class:`NnMlpModel` to keep ``torch`` off cheap paths.

    Importing :mod:`bristol_ml.models.nn` does not pull ``torch`` into the
    import graph; the concrete class is only materialised on first attribute
    access (e.g. ``bristol_ml.models.nn.NnMlpModel``).  Matches the lazy
    ``__getattr__`` idiom in :mod:`bristol_ml.models.__init__`.
    """
    if name == "NnMlpModel":
        from bristol_ml.models.nn.mlp import NnMlpModel

        return NnMlpModel
    raise AttributeError(f"module 'bristol_ml.models.nn' has no attribute {name!r}")
