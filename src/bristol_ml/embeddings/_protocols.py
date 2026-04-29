"""Stage 15 — :class:`Embedder` and :class:`VectorIndex` Protocols.

Plan §1 D2 + D3 + NFR-4: two narrow, swap-safe Protocols at the layer
boundary. AC-5 (intent line 34): *"the index's interface is small
enough that swapping the vector store implementation later (for
example, FAISS for a toy numpy store, or vice-versa) is a mechanical
change"*.

Both Protocols are :func:`~typing.runtime_checkable` per the
ADR-0003 precedent (Protocol over ``abc.ABC`` for swappable
interfaces). Concrete implementations live in ``_embedder.py``
(``StubEmbedder`` / ``SentenceTransformerEmbedder``) and ``_index.py``
(``StubIndex`` / ``NumpyIndex``).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "NearestNeighbour",
    "VectorIndex",
]


class NearestNeighbour(NamedTuple):
    """One ``(id, score)`` pair returned by :meth:`VectorIndex.query`.

    A :class:`~typing.NamedTuple` (rather than a Pydantic model) keeps
    the boundary lightweight — the index is queried in tight loops
    where the per-row Pydantic validation cost is wasted: the
    :class:`VectorIndex` only ever returns pairs it just constructed,
    so the values are already typed.

    The pair shape is plan §1 D3:
    ``query(vec, k) -> list[(id, score)]``. The ``id`` is whatever
    string the caller passed to :meth:`VectorIndex.add` (Stage 15
    pairs ``(mrid, revision_number)`` into a single string id —
    ``"<mrid>::<revision_number>"`` — at the corpus-loading site,
    not in the index itself).

    ``score`` is a cosine similarity in the closed range ``[-1, 1]``;
    larger means more similar. The :class:`VectorIndex.query`
    contract guarantees the returned list is sorted by ``score``
    descending.
    """

    id: str
    score: float


@runtime_checkable
class Embedder(Protocol):
    """Stage 15 sentence-embedding contract — two methods only.

    Plan §1 D2 / NFR-4: the Protocol carries one single-text method
    and one batch method, mirroring the Stage 14 :class:`Extractor`
    shape exactly.

    **Prefix asymmetry (gte-modernbert-base / E5 convention).**  The
    plan-bound default checkpoint is
    ``Alibaba-NLP/gte-modernbert-base``, which uses the E5 family's
    asymmetric prefix convention: queries are prefixed with
    ``"query: "``; documents take no prefix. The
    :class:`SentenceTransformerEmbedder` honours this — :meth:`embed`
    is the *query* path (prepends ``"query: "``); :meth:`embed_batch`
    is the *document* path (no prefix). The :class:`StubEmbedder`
    mirrors the contract by hashing the prefixed-vs-unprefixed text,
    so a unit test can detect a regression where a downstream caller
    accidentally calls ``embed_batch`` with query-shaped input.

    **Output shape.**  :meth:`embed` returns a 1-D ``np.ndarray`` of
    ``float32``; :meth:`embed_batch` returns a 2-D ``np.ndarray`` of
    shape ``(len(texts), dim)``. Both are L2-normalised — pre-
    normalisation at the embedding layer simplifies the cosine
    computation in :class:`VectorIndex.query` to a plain matmul.

    Implementations:

    - :class:`bristol_ml.embeddings._embedder.StubEmbedder` — the
      offline-by-default path. Deterministic SHA-256-derived vectors;
      no network, no model download, no tokeniser.
    - :class:`bristol_ml.embeddings._embedder.SentenceTransformerEmbedder`
      — the live path. Loads
      ``Alibaba-NLP/gte-modernbert-base`` (768-dim) by default;
      ``HF_HUB_OFFLINE=1`` is asserted in the project conftest so a
      missing local cache fails loudly rather than triggering a
      download.
    """

    @property
    def dim(self) -> int:
        """The embedding dimensionality.  Constant per :class:`Embedder` instance."""
        ...

    @property
    def model_id(self) -> str:
        """The model identifier (e.g. ``"Alibaba-NLP/gte-modernbert-base"``).

        Stamped into :class:`EmbeddingCacheMetadata.model_id` for
        provenance / cache-invalidation (plan §1 D7 + D13 + NFR-5).
        For the stub, the id is a documented sentinel
        (``"stub-sha256-<dim>"``).
        """
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed a single (query-shaped) text.

        Returns a 1-D ``np.ndarray`` of shape ``(dim,)``, dtype
        ``float32``, L2-normalised.

        For the live :class:`SentenceTransformerEmbedder`, the input
        is prepended with ``"query: "`` per the E5 / GTE-modernbert
        convention. The stub mirrors the same prepend so its hash
        signature differs between :meth:`embed` and
        :meth:`embed_batch` — a regression test catches a downstream
        caller that mistakenly routes queries through the document
        path.
        """
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of (document-shaped) texts.

        Returns a 2-D ``np.ndarray`` of shape ``(len(texts), dim)``,
        dtype ``float32``, L2-normalised row-wise. Order is preserved.

        Documents take no prefix. For the live embedder the batch
        path uses the sentence-transformers encoder's native
        ``encode(..., batch_size=...)`` call so the tokeniser is
        amortised across the input list.
        """
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Stage 15 nearest-neighbour vector store — four methods.

    Plan §1 D3 / NFR-4: AC-5 caps the public surface at four methods.
    A future FAISS / hnswlib implementation slots in by extending the
    :class:`~conf._schemas.EmbeddingConfig` ``backend`` literal and
    adding a dispatch branch in
    :func:`bristol_ml.embeddings.build_index`; the Protocol does not
    change.

    Implementations:

    - :class:`bristol_ml.embeddings._index.NumpyIndex` — plain numpy
      backend. Pre-normalised float32 corpus matrix; cosine query is
      a single matmul. Sufficient for the few-thousand-event REMIT
      corpus the meetup demo runs against.
    - :class:`bristol_ml.embeddings._index.StubIndex` — deterministic
      tiny in-memory implementation used by the unit tests; satisfies
      the Protocol structurally without the numpy matmul path.

    **Empty index discipline.**  :meth:`query` on an empty index
    returns ``[]``; it does not raise. This matches the
    :class:`StubExtractor`'s miss-path discipline (Stage 14 D16) —
    callers receive a typed-empty result they can branch on.
    """

    @property
    def dim(self) -> int:
        """The embedding dimensionality the index was built for."""
        ...

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Append rows to the index.

        ``ids`` and ``vectors`` must have the same length; ``vectors``
        is ``(n, dim)`` dtype ``float32`` and L2-normalised
        row-wise (the :class:`Embedder` contract above guarantees
        this).

        Re-adding an id is the caller's responsibility to police —
        the index does not de-duplicate. Stage 15's only producer of
        ids is :func:`bristol_ml.embeddings.embed_corpus`, which
        builds them from the corpus's
        ``(mrid, revision_number)`` primary key (unique by Stage 13
        ``OUTPUT_SCHEMA``).
        """
        ...

    def query(self, vector: np.ndarray, k: int) -> list[NearestNeighbour]:
        """Return the top-``k`` nearest neighbours, sorted by score descending.

        ``vector`` is the *query* embedding (1-D, ``float32``,
        L2-normalised). ``k`` is the requested neighbour count;
        ``k`` larger than the index size is clipped silently to the
        index size (no exception, no zero-padding).

        Score is cosine similarity in ``[-1, 1]`` — guaranteed by the
        normalised-vectors contract. The returned list is length
        ``min(k, len(self))`` and is sorted by ``score`` descending.
        """
        ...

    def save(self, path: Path) -> None:
        """Persist the index to ``path`` atomically.

        The on-disk layout is implementation-specific; the only
        contract is that :meth:`load` can reconstruct an equivalent
        index from the same path. :class:`NumpyIndex` writes a single
        ``.npz`` file (corpus matrix + ids array) via the parquet
        atomic-write idiom (``.tmp`` sibling + ``os.replace``).
        """
        ...

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        """Reconstruct an index previously persisted via :meth:`save`."""
        ...
