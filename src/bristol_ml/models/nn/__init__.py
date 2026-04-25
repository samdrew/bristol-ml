"""Neural-network models sub-layer — Stage 10 onwards.

Stage 10 introduced this sub-package with a single concrete model
(:class:`~bristol_ml.models.nn.mlp.NnMlpModel`) — a small MLP that
conforms to the Stage 4 :class:`bristol_ml.models.Model` protocol.
Stage 11 adds :class:`~bristol_ml.models.nn.temporal.NnTemporalModel`
(a dilated-causal Temporal Convolutional Network — the Bai et al. 2018
recipe) alongside it, sharing the hand-rolled training loop extracted
at Stage 11 T1 to :mod:`bristol_ml.models.nn._training` per the
Stage 10 D10 extraction seam.

The sub-package is structured so each model's standalone CLI
(``python -m bristol_ml.models.nn.mlp`` / ``python -m
bristol_ml.models.nn.temporal``) prints the config schema
(DESIGN §2.1.1 / plan NFR-6) and so that
``from bristol_ml.models.nn import NnMlpModel`` /
``from bristol_ml.models.nn import NnTemporalModel`` work without
pulling ``torch`` into the import graph of modules that do not need
it — the concrete classes are resolved lazily through ``__getattr__``
below, mirroring the top-level :mod:`bristol_ml.models` re-export
pattern.  The same mechanism keeps ``python -m bristol_ml.models.nn
--help`` torch-free (Stage 11 NFR-5 / T3 lazy-import guard).

Cross-references:

- Layer doc: ``docs/architecture/layers/models-nn.md`` (Stage 10 T6;
  Stage 11 extends it with the TCN contract).
- Module guide: ``src/bristol_ml/models/nn/CLAUDE.md`` (Stage 10 T6).
- Plans: ``docs/plans/completed/10-simple-nn.md``,
  ``docs/plans/completed/11-complex-nn.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.models.nn.mlp import NnMlpModel
    from bristol_ml.models.nn.temporal import NnTemporalModel

__all__ = ["NnMlpModel", "NnTemporalModel"]


def __getattr__(name: str) -> object:
    """Lazy re-export of concrete NN model classes to keep ``torch`` off cheap paths.

    Importing :mod:`bristol_ml.models.nn` does not pull ``torch`` into the
    import graph; the concrete class is only materialised on first attribute
    access (e.g. ``bristol_ml.models.nn.NnMlpModel`` or
    ``bristol_ml.models.nn.NnTemporalModel``).  Matches the lazy
    ``__getattr__`` idiom in :mod:`bristol_ml.models.__init__`.
    """
    if name == "NnMlpModel":
        from bristol_ml.models.nn.mlp import NnMlpModel

        return NnMlpModel
    if name == "NnTemporalModel":
        from bristol_ml.models.nn.temporal import NnTemporalModel

        return NnTemporalModel
    raise AttributeError(f"module 'bristol_ml.models.nn' has no attribute {name!r}")
