"""Stage 15 — concrete :class:`VectorIndex` implementations.

Two implementations selected by
:class:`conf._schemas.EmbeddingConfig.vector_backend` (plan §1 D3 + D5):

- :class:`StubIndex` — in-memory list-of-tuples backend used by the
  unit tests. Satisfies the :class:`~bristol_ml.embeddings.VectorIndex`
  Protocol structurally; the :meth:`StubIndex.query` path uses an
  explicit Python loop so the test failure mode is obvious when the
  Protocol contract changes.
- :class:`NumpyIndex` — production backend. Stage 15 D5 binds the
  vector store to plain numpy: pre-normalised float32 corpus matrix +
  matmul query. Zero new deps. The few-thousand-event REMIT corpus
  fits comfortably in laptop RAM (R-2). A future FAISS / hnswlib swap
  is one new branch in :func:`bristol_ml.embeddings.build_index`; the
  Protocol does not change.

Both classes pre-normalise on :meth:`add` (defensively — the
:class:`Embedder` contract already L2-normalises, but a hand-built
fixture might not) and persist to disk via the parquet-tested atomic-
write idiom (``.tmp`` sibling + :func:`os.replace`).

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md``.
- Stage 15 plan — ``docs/plans/active/15-embedding-index.md`` §6 T2 + T3.
- Atomic-write prior art — :func:`bristol_ml.ingestion._common._atomic_write`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger

from bristol_ml.embeddings._protocols import NearestNeighbour

__all__ = [
    "NumpyIndex",
    "StubIndex",
]


# Floating-point tolerance for the "is this vector L2-normalised?" check.
# A normalised float32 vector's norm is 1.0 ± a few ULPs; 1e-3 is loose
# enough to absorb fp16 inference round-trip jitter (R-3) without
# masking a genuinely non-normalised input.
_NORM_TOLERANCE: float = 1e-3


def _validate_vectors(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Coerce ``vectors`` to a float32 (n, dim) array; raise on shape mismatch.

    ``add`` and the index loaders both funnel through this helper so a
    bad shape fails at the storage boundary, not in the matmul path
    where the error is harder to read.
    """
    if vectors.ndim != 2:
        raise ValueError(
            f"vectors must be 2-D (n, dim); got ndim={vectors.ndim} shape={vectors.shape}."
        )
    if vectors.shape[1] != dim:
        raise ValueError(
            f"vectors second dimension must match index dim={dim}; got shape={vectors.shape}."
        )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _renormalise(vectors: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalised copy of ``vectors``.

    Defensive — the :class:`Embedder` contract (D2) already guarantees
    the inputs are normalised, but a hand-rolled test fixture or a
    future custom embedder might not. Renormalising at the index
    boundary keeps the cosine-as-matmul invariant load-bearing.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Replace zero norms with 1.0 to avoid divide-by-zero; a zero
    # vector has no defined cosine direction, so the normalised
    # output stays zero and contributes 0 to every cosine score.
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------
# In-memory stub — minimal Protocol-satisfying backend for unit tests
# ---------------------------------------------------------------------


class StubIndex:
    """Tiny list-backed :class:`VectorIndex` for unit tests.

    Plan §1 D3 / D18: the stub satisfies the Protocol without a numpy
    matmul path. The :meth:`query` loop is explicit so a Protocol
    drift (e.g. a method rename) reads as a test failure, not as a
    silent dispatch onto the numpy backend.

    Empty-index discipline: :meth:`query` returns ``[]`` on an empty
    index (mirrors :class:`StubExtractor`'s miss-path discipline,
    Stage 14 D16).
    """

    def __init__(self, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"StubIndex dim must be positive; got {dim}.")
        self._dim = dim
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids and vectors must have the same length; got "
                f"len(ids)={len(ids)} and vectors.shape[0]={vectors.shape[0]}."
            )
        validated = _validate_vectors(vectors, self._dim)
        renormalised = _renormalise(validated)
        self._ids.extend(ids)
        self._vectors.extend(row for row in renormalised)

    def query(self, vector: np.ndarray, k: int) -> list[NearestNeighbour]:
        if not self._ids:
            return []
        if vector.ndim != 1 or vector.shape[0] != self._dim:
            raise ValueError(
                f"query vector must be 1-D with dim={self._dim}; got shape={vector.shape}."
            )
        v = np.ascontiguousarray(vector, dtype=np.float32)
        # Renormalise defensively (cheap for a single vector).
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        scores = [float(np.dot(v, row)) for row in self._vectors]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        clipped = min(k, len(self._ids))
        return [NearestNeighbour(id=self._ids[i], score=scores[i]) for i in order[:clipped]]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        matrix = (
            np.stack(self._vectors).astype(np.float32)
            if self._vectors
            else np.zeros((0, self._dim), dtype=np.float32)
        )
        np.savez_compressed(
            tmp,
            ids=np.asarray(self._ids, dtype=object),
            vectors=matrix,
            dim=np.asarray([self._dim], dtype=np.int32),
        )
        # ``np.savez_compressed`` may add a ``.npz`` suffix if the
        # source path lacks one; normalise so ``os.replace`` lands on
        # the caller-named file.
        actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
        os.replace(actual_tmp, path)
        logger.debug("StubIndex saved to {} (n={}, dim={})", path, len(self._ids), self._dim)

    @classmethod
    def load(cls, path: Path) -> StubIndex:
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            dim = int(npz["dim"][0])
            ids = list(npz["ids"].tolist())
            vectors = np.asarray(npz["vectors"], dtype=np.float32)
        index = cls(dim=dim)
        if ids:
            index.add(ids, vectors)
        logger.debug("StubIndex loaded from {} (n={}, dim={})", path, len(ids), dim)
        return index


# ---------------------------------------------------------------------
# Numpy production backend — pre-normalised matmul query
# ---------------------------------------------------------------------


class NumpyIndex:
    """Plain-numpy :class:`VectorIndex`. Pre-normalised float32 corpus matrix.

    Plan §1 D5: cosine similarity is ``corpus @ query.T`` because both
    operands are L2-normalised at :meth:`add` time. Top-``k`` is a
    single :func:`numpy.argpartition` + a sort over the partitioned
    slice — O(n + k log k) per query, fast enough for the
    few-thousand-event REMIT corpus (NFR-6).

    Append-only: :meth:`add` extends the corpus matrix; the index
    does not de-duplicate ids (the upstream
    :func:`bristol_ml.embeddings.embed_corpus` pairs ``mrid`` with
    ``revision_number`` so the storage grain is unique by Stage 13's
    ``OUTPUT_SCHEMA`` primary key).

    Persistence: a single ``.npz`` file written via the parquet-tested
    atomic-write idiom (``.tmp`` sibling + :func:`os.replace`). The
    on-disk layout — ``ids`` (object array of strings), ``vectors``
    (float32 (n, dim)), ``dim`` (int32 scalar in a length-1 array) —
    is private; only :meth:`save` / :meth:`load` round-trip it.
    """

    def __init__(self, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"NumpyIndex dim must be positive; got {dim}.")
        self._dim = dim
        self._ids: list[str] = []
        # Start as an empty (0, dim) matrix so the first ``add`` can
        # vstack without a special case.
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids and vectors must have the same length; got "
                f"len(ids)={len(ids)} and vectors.shape[0]={vectors.shape[0]}."
            )
        validated = _validate_vectors(vectors, self._dim)
        # Defensive renormalisation — the Embedder contract already
        # normalises, but a downstream caller passing a hand-built
        # matrix would otherwise silently degrade query scores.
        renormalised = _renormalise(validated)
        self._ids.extend(ids)
        if self._matrix.shape[0] == 0:
            self._matrix = renormalised
        else:
            self._matrix = np.vstack([self._matrix, renormalised])

    def query(self, vector: np.ndarray, k: int) -> list[NearestNeighbour]:
        if self._matrix.shape[0] == 0:
            return []
        if vector.ndim != 1 or vector.shape[0] != self._dim:
            raise ValueError(
                f"query vector must be 1-D with dim={self._dim}; got shape={vector.shape}."
            )
        v = np.ascontiguousarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0:
            # An all-zero query has no defined cosine direction; return
            # every id with score 0.0 to keep the contract total.
            clipped = min(k, len(self._ids))
            return [NearestNeighbour(id=self._ids[i], score=0.0) for i in range(clipped)]
        if not (1.0 - _NORM_TOLERANCE <= norm <= 1.0 + _NORM_TOLERANCE):
            v = v / norm
        # Cosine = matmul because both operands are normalised.
        scores = self._matrix @ v
        n = scores.shape[0]
        clipped = min(k, n)
        if clipped == n:
            order = np.argsort(-scores)
        else:
            # ``argpartition`` puts the top-``clipped`` indices at the
            # front in arbitrary order; sort just that slice.
            partitioned = np.argpartition(-scores, clipped - 1)[:clipped]
            order = partitioned[np.argsort(-scores[partitioned])]
        return [NearestNeighbour(id=self._ids[int(i)], score=float(scores[int(i)])) for i in order]

    def save(self, path: Path) -> None:
        """Persist the index atomically (``.tmp`` sibling + ``os.replace``).

        Mirrors :func:`bristol_ml.ingestion._common._atomic_write`'s
        contract — a partial write leaves the previous file intact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        np.savez_compressed(
            tmp,
            ids=np.asarray(self._ids, dtype=object),
            vectors=self._matrix,
            dim=np.asarray([self._dim], dtype=np.int32),
        )
        # ``np.savez_compressed`` appends ``.npz`` if the path has no
        # extension; we wrote ``<path>.tmp`` which already has one, so
        # numpy leaves it alone — but be defensive in case a caller
        # passes a path without an extension at all.
        actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
        os.replace(actual_tmp, path)
        logger.info(
            "NumpyIndex saved to {} (n={}, dim={})",
            path,
            self._matrix.shape[0],
            self._dim,
        )

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        """Reconstruct an index from a previous :meth:`save`."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            dim = int(npz["dim"][0])
            ids = list(npz["ids"].tolist())
            vectors = np.asarray(npz["vectors"], dtype=np.float32)
        index = cls(dim=dim)
        if ids:
            index.add(ids, vectors)
        logger.info("NumpyIndex loaded from {} (n={}, dim={})", path, len(ids), dim)
        return index
