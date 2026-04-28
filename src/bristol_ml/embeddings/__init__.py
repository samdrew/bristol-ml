"""Stage 15 — Embedding index over REMIT: typed boundary.

This package exposes the **public boundary** of the Stage 15 embedding
index: the :class:`Embedder` and :class:`VectorIndex` Protocols, plus
their factory functions and the env-var constants that gate
stub-vs-live dispatch.

Plan §1 D2 + D3: the two Protocols are
:func:`~typing.runtime_checkable` so unit tests can assert structural
conformance with ``isinstance(_, Embedder)`` / ``isinstance(_,
VectorIndex)`` — ADR-0003 sets the precedent for Protocol-over-ABC for
swappable interfaces. The ``Embedder`` carries two methods (single +
batch); the ``VectorIndex`` carries four (``add`` / ``query`` / ``save``
/ ``load``). NFR-4 caps the surface at those numbers; growth is a
contract change to be discussed before doing it.

Plan AC-1 sub-criterion: importing ``bristol_ml.embeddings`` must not
drag in ``sentence-transformers`` or ``torch`` — Stage 16 callers
holding only the Protocol type must not pay the ML-stack import cost.
The split between this ``__init__`` (boundary types + factory) and
``_embedder`` / ``_index`` / ``_cache`` (implementations) is the
load-bearing mechanism. The factory deferred-imports the heavy
backends (T4 path) only when the ``"sentence_transformers"``
discriminator is hit.

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md`` (Stage 15 T10).
- Stage 15 plan — ``docs/plans/completed/15-embedding-index.md``.
- Intent — ``docs/intent/15-embedding-index.md``.
- ADR-0008 — Embedder + VectorIndex Protocols (Stage 15 T10).
"""

from __future__ import annotations

from bristol_ml.embeddings._cache import EmbeddingCache, EmbeddingCacheMetadata
from bristol_ml.embeddings._factory import (
    MODEL_PATH_ENV_VAR,
    STUB_ENV_VAR,
    build_embedder,
    build_index,
    embed_corpus,
)
from bristol_ml.embeddings._protocols import (
    Embedder,
    NearestNeighbour,
    VectorIndex,
)
from bristol_ml.embeddings._text import synthesise_embeddable_text

__all__ = [
    "MODEL_PATH_ENV_VAR",
    "STUB_ENV_VAR",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingCacheMetadata",
    "NearestNeighbour",
    "VectorIndex",
    "build_embedder",
    "build_index",
    "embed_corpus",
    "synthesise_embeddable_text",
]
