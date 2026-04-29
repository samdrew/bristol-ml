"""Stage 15 — content-addressed embedding cache.

Plan §1 D7 + D13 + D14: the cache is a single Parquet file
(``data/embeddings/<model_id_sanitised>.parquet``) with a vector
list-column (one row per REMIT event) and three provenance fields
stamped into Parquet ``custom_metadata``:

- ``embedded_at_utc`` — ISO-8601 UTC string, when the cache was built.
- ``corpus_sha256`` — full 64-char hex SHA-256 of the deterministic
  serialisation of the embeddable-text column. Stored long; the public
  :class:`EmbeddingCacheMetadata` dataclass exposes a 12-char prefix
  alongside the full hash for log-line ergonomics.
- ``model_id`` — the configured embedder ``model_id`` (e.g.
  ``"Alibaba-NLP/gte-modernbert-base"`` or ``"stub-sha256-8"``).

NFR-2 + AC-3: re-reading a fresh cache is a no-op (read-only). Stale
cache (corpus hash changed *or* model_id changed) triggers an automatic
rebuild via :meth:`EmbeddingCache.load_or_build`. AC-8: the rebuild is
loud (``loguru`` WARNING with the offending field) — never silent.

This is the project's first content-addressed cache. The lessons it
teaches — corpus-hash + model-id as the invalidation key, Parquet
``custom_metadata`` as the provenance carrier, deterministic recompute
on mismatch — generalise beyond embeddings to any cached derivation
downstream of any ingested corpus.

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md``.
- Stage 15 plan — ``docs/plans/completed/15-embedding-index.md`` §6 T5.
- Atomic-write prior art — :func:`bristol_ml.ingestion._common._atomic_write`.
- DESIGN.md §2.1.6 (provenance discipline).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from bristol_ml.embeddings._protocols import Embedder

__all__ = [
    "CORPUS_HASH_PREFIX_LEN",
    "EmbeddingCache",
    "EmbeddingCacheMetadata",
    "compute_corpus_sha256",
]


# Plan §1 D13: ``corpus_sha256`` is stored full-length on the parquet
# but the public dataclass exposes a 12-char prefix for log-line
# ergonomics — long enough to distinguish corpus revisions in practice
# while staying readable in WARNING messages on a 132-column terminal.
CORPUS_HASH_PREFIX_LEN: int = 12

# Parquet ``custom_metadata`` keys are bytes by pyarrow convention.
_META_KEY_EMBEDDED_AT_UTC = b"embedded_at_utc"
_META_KEY_CORPUS_SHA256 = b"corpus_sha256"
_META_KEY_MODEL_ID = b"model_id"
_META_KEY_DIM = b"dim"


@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    """Provenance fields read from / written to Parquet ``custom_metadata``.

    Plan §1 D13: three fields. ``git_sha`` was dropped per the Scope
    Diff (no AC; duplicates the registry-layer convention for a cache
    file).

    ``corpus_sha256`` is the full 64-char hex; :attr:`corpus_sha256_prefix`
    returns the first :data:`CORPUS_HASH_PREFIX_LEN` chars for log
    messages (mirrors :func:`bristol_ml.llm._prompts.prompt_sha256_prefix`).

    ``embedded_at_utc`` is a UTC-aware :class:`~datetime.datetime` after
    parsing; on disk it is the ISO-8601 string. Naive timestamps round-
    tripping through this dataclass is a hard error.
    """

    embedded_at_utc: datetime
    corpus_sha256: str
    model_id: str
    dim: int

    def __post_init__(self) -> None:
        if self.embedded_at_utc.tzinfo is None:
            raise ValueError(
                "EmbeddingCacheMetadata.embedded_at_utc must be timezone-aware "
                "(UTC); got naive datetime."
            )
        if len(self.corpus_sha256) != 64:
            raise ValueError(
                f"corpus_sha256 must be the full 64-char hex digest; "
                f"got len={len(self.corpus_sha256)}."
            )
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string.")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive; got {self.dim}.")

    @property
    def corpus_sha256_prefix(self) -> str:
        """First :data:`CORPUS_HASH_PREFIX_LEN` chars of the corpus hash."""
        return self.corpus_sha256[:CORPUS_HASH_PREFIX_LEN]

    def to_parquet_metadata(self) -> dict[bytes, bytes]:
        """Encode this dataclass as ``custom_metadata`` bytes for Parquet write."""
        return {
            _META_KEY_EMBEDDED_AT_UTC: self.embedded_at_utc.isoformat().encode("utf-8"),
            _META_KEY_CORPUS_SHA256: self.corpus_sha256.encode("ascii"),
            _META_KEY_MODEL_ID: self.model_id.encode("utf-8"),
            _META_KEY_DIM: str(self.dim).encode("ascii"),
        }

    @classmethod
    def from_parquet_metadata(cls, raw: dict[bytes, bytes]) -> EmbeddingCacheMetadata:
        """Decode parquet ``custom_metadata`` back into an instance.

        Missing keys raise :class:`KeyError` with the offending key
        named — there's no sensible default for any of these (they
        define the cache identity).
        """
        for key in (
            _META_KEY_EMBEDDED_AT_UTC,
            _META_KEY_CORPUS_SHA256,
            _META_KEY_MODEL_ID,
            _META_KEY_DIM,
        ):
            if key not in raw:
                raise KeyError(
                    f"EmbeddingCacheMetadata: missing key {key.decode('ascii')!r} "
                    f"in parquet custom_metadata. Cache file is corrupt or "
                    f"written by an older Stage 15 schema."
                )
        return cls(
            embedded_at_utc=datetime.fromisoformat(raw[_META_KEY_EMBEDDED_AT_UTC].decode("utf-8")),
            corpus_sha256=raw[_META_KEY_CORPUS_SHA256].decode("ascii"),
            model_id=raw[_META_KEY_MODEL_ID].decode("utf-8"),
            dim=int(raw[_META_KEY_DIM].decode("ascii")),
        )


# ---------------------------------------------------------------------
# Corpus hashing — D7
# ---------------------------------------------------------------------


def compute_corpus_sha256(texts: Iterable[str]) -> str:
    """Deterministic SHA-256 over an ordered iterable of corpus texts.

    Plan §1 D7: the cache invalidation key is SHA-256(corpus + model_id).
    This function computes the corpus half. The serialisation is
    ``"\\n".join(texts).encode("utf-8")`` — order-sensitive (a row
    reorder is, semantically, a different corpus from the index's
    perspective because ids align by row position), null-safe (the
    caller is responsible for collapsing NULLs to a synthesised string
    via :func:`~bristol_ml.embeddings.synthesise_embeddable_text`).

    A trailing newline ensures a single-row corpus and the same row
    pre-pended with an empty string hash to different values:
    ``"foo\\n"`` vs ``"\\nfoo\\n"``.

    Returns the full 64-char lowercase hex digest. The 12-char prefix
    used in log messages is :data:`CORPUS_HASH_PREFIX_LEN`.
    """
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------
# EmbeddingCache — write, read, and freshness check
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingCache:
    """A loaded embedding cache: ids + vectors + metadata.

    Plan §1 D14: vectors live as a Parquet ``list<float32>`` column
    (one list per row); ``ids`` is a parallel ``string`` column;
    metadata sits on the schema's ``custom_metadata``. A single file
    keeps the cache atomic to write / read; no sidecar JSON.

    The class is constructed by :meth:`load_or_build`, which is the
    single entrypoint Stage 15 callers reach for. Direct
    :meth:`write` / :meth:`read` are exposed for tests.
    """

    ids: list[str]
    vectors: np.ndarray  # shape (n, dim), dtype float32
    metadata: EmbeddingCacheMetadata

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def write(self, path: Path) -> None:
        """Persist the cache to ``path`` atomically (``.tmp`` + ``os.replace``).

        Mirrors the parquet-tested atomic-write idiom from
        :func:`bristol_ml.ingestion._common._atomic_write`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n, dim = self.vectors.shape
        if dim != self.metadata.dim:
            raise ValueError(
                f"vectors second dim {dim} disagrees with metadata.dim "
                f"{self.metadata.dim}; refusing to write inconsistent cache."
            )
        if len(self.ids) != n:
            raise ValueError(f"ids length {len(self.ids)} disagrees with vectors row count {n}.")
        # Build the table. Vectors stored as a list column to keep
        # row-major retrieval cheap and avoid pyarrow's fixed-size-list
        # roundtripping quirks across versions.
        vector_lists = [vec.tolist() for vec in self.vectors]
        table = pa.table(
            {
                "id": pa.array(self.ids, type=pa.string()),
                "vector": pa.array(vector_lists, type=pa.list_(pa.float32())),
            }
        )
        schema = table.schema.with_metadata(self.metadata.to_parquet_metadata())
        table = table.replace_schema_metadata(schema.metadata)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        logger.info(
            "EmbeddingCache written to {} (n={}, dim={}, model_id={}, corpus_sha256={})",
            path,
            n,
            dim,
            self.metadata.model_id,
            self.metadata.corpus_sha256_prefix,
        )

    @classmethod
    def read(cls, path: Path) -> EmbeddingCache:
        """Read a cache file written by :meth:`write`.

        Raises :class:`FileNotFoundError` when ``path`` is absent;
        callers wanting a "build if absent" path should use
        :meth:`load_or_build` instead.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EmbeddingCache: no file at {path}.")
        table = pq.read_table(path)
        raw_metadata = table.schema.metadata or {}
        metadata = EmbeddingCacheMetadata.from_parquet_metadata(dict(raw_metadata))
        ids = table.column("id").to_pylist()
        vector_lists = table.column("vector").to_pylist()
        if vector_lists:
            vectors = np.asarray(vector_lists, dtype=np.float32)
        else:
            vectors = np.zeros((0, metadata.dim), dtype=np.float32)
        if vectors.ndim == 2 and vectors.shape[1] != metadata.dim:
            raise ValueError(
                f"EmbeddingCache: parquet vector dim {vectors.shape[1]} disagrees "
                f"with custom_metadata.dim {metadata.dim} at {path}."
            )
        logger.debug(
            "EmbeddingCache read from {} (n={}, dim={}, model_id={}, corpus_sha256={})",
            path,
            len(ids),
            metadata.dim,
            metadata.model_id,
            metadata.corpus_sha256_prefix,
        )
        return cls(ids=ids, vectors=vectors, metadata=metadata)

    # ------------------------------------------------------------------
    # Public freshness + build entrypoint
    # ------------------------------------------------------------------

    @classmethod
    def load_or_build(
        cls,
        *,
        path: Path,
        ids: list[str],
        texts: list[str],
        embedder: Embedder,
        force_rebuild: bool = False,
    ) -> EmbeddingCache:
        """Return a fresh cache, building from scratch if stale or missing.

        Plan §1 D7: the freshness key is
        ``(compute_corpus_sha256(texts), embedder.model_id)``. A
        mismatch on either field triggers an automatic rebuild
        (AC-8 — loud, not silent: WARNING log naming the field).
        ``force_rebuild=True`` skips the freshness check entirely
        (D17 escape hatch).

        Parameters
        ----------
        path
            Cache file location. Created on first build; rewritten on
            stale-cache rebuild.
        ids
            One id per text — the index will use these as the
            :class:`~bristol_ml.embeddings.NearestNeighbour.id` field.
        texts
            One embeddable text per row, in alignment with ``ids``.
            Callers should pre-coerce NULL ``message_description`` via
            :func:`~bristol_ml.embeddings.synthesise_embeddable_text`
            (plan §1 D9).
        embedder
            The :class:`~bristol_ml.embeddings.Embedder` instance —
            its ``model_id`` is the second freshness component.
            ``embed_batch`` is the document-path call.
        force_rebuild
            If True, bypass the freshness check and rebuild
            unconditionally (plan §1 D17).
        """
        if len(ids) != len(texts):
            raise ValueError(
                f"ids and texts must align; got len(ids)={len(ids)} and len(texts)={len(texts)}."
            )
        path = Path(path)
        corpus_sha256 = compute_corpus_sha256(texts)

        if path.exists() and not force_rebuild:
            try:
                cached = cls.read(path)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "EmbeddingCache: existing file at {} is unreadable ({}); rebuilding.",
                    path,
                    exc,
                )
            else:
                stale_reason = _stale_reason(
                    cached=cached.metadata,
                    expected_corpus_sha256=corpus_sha256,
                    expected_model_id=embedder.model_id,
                )
                if stale_reason is None:
                    logger.info(
                        "EmbeddingCache hit at {} (n={}, model_id={}, corpus_sha256={})",
                        path,
                        len(cached.ids),
                        cached.metadata.model_id,
                        cached.metadata.corpus_sha256_prefix,
                    )
                    return cached
                logger.warning(
                    "EmbeddingCache stale at {} ({}); rebuilding (model_id={}, corpus_sha256={}).",
                    path,
                    stale_reason,
                    embedder.model_id,
                    corpus_sha256[:CORPUS_HASH_PREFIX_LEN],
                )

        # Build path — either no file, force_rebuild, unreadable file,
        # or stale freshness key.
        return _build_and_write(
            path=path,
            ids=ids,
            texts=texts,
            embedder=embedder,
            corpus_sha256=corpus_sha256,
        )


def _stale_reason(
    *,
    cached: EmbeddingCacheMetadata,
    expected_corpus_sha256: str,
    expected_model_id: str,
) -> str | None:
    """Return a human-readable stale-reason, or None if the cache is fresh."""
    if cached.model_id != expected_model_id:
        return f"model_id mismatch (cached={cached.model_id!r}, expected={expected_model_id!r})"
    if cached.corpus_sha256 != expected_corpus_sha256:
        return (
            f"corpus_sha256 mismatch (cached={cached.corpus_sha256_prefix}, "
            f"expected={expected_corpus_sha256[:CORPUS_HASH_PREFIX_LEN]})"
        )
    return None


def _build_and_write(
    *,
    path: Path,
    ids: list[str],
    texts: list[str],
    embedder: Embedder,
    corpus_sha256: str,
) -> EmbeddingCache:
    """Embed ``texts`` and persist the resulting cache. Helper for testability."""
    if texts:
        vectors = embedder.embed_batch(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedder returned vectors of shape {vectors.shape}; expected ({len(texts)}, dim)."
            )
    else:
        vectors = np.zeros((0, embedder.dim), dtype=np.float32)
    metadata = EmbeddingCacheMetadata(
        embedded_at_utc=datetime.now(UTC),
        corpus_sha256=corpus_sha256,
        model_id=embedder.model_id,
        dim=embedder.dim,
    )
    cache = EmbeddingCache(ids=list(ids), vectors=vectors, metadata=metadata)
    cache.write(path)
    return cache


# ---------------------------------------------------------------------
# Factory hook for build_index — used by _factory.embed_corpus
# ---------------------------------------------------------------------


def build_cache(
    *,
    path: Path,
    ids: list[str],
    texts: list[str],
    embedder: Embedder,
    force_rebuild: bool = False,
) -> EmbeddingCache:
    """Module-level alias for :meth:`EmbeddingCache.load_or_build`.

    Exposed so :func:`bristol_ml.embeddings.embed_corpus` can call a
    plain function without instantiating the dataclass first. The
    behaviour is identical.
    """
    return EmbeddingCache.load_or_build(
        path=path,
        ids=ids,
        texts=texts,
        embedder=embedder,
        force_rebuild=force_rebuild,
    )
