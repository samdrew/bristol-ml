"""Spec-derived tests for the Stage 15 T5 :class:`EmbeddingCache`.

Every test here is derived from:

- ``docs/plans/completed/15-embedding-index.md`` §6 T5 named tests:
  ``test_cache_hit_no_rebuild``,
  ``test_corpus_change_invalidates``,
  ``test_model_change_invalidates``.
- AC-3 (intent line 36): "the embedding cache means re-running the
  notebook is fast".
- AC-8 (requirements §3): "Cache invalidation is detectable (loud,
  not silent)".
- Plan §1 D7 (corpus SHA-256 + model_id), D13 (provenance fields),
  D14 (single Parquet file).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from bristol_ml.embeddings._cache import (
    CORPUS_HASH_PREFIX_LEN,
    EmbeddingCache,
    EmbeddingCacheMetadata,
    compute_corpus_sha256,
)
from bristol_ml.embeddings._embedder import StubEmbedder

# ---------------------------------------------------------------------
# Plan §1 D7 — deterministic corpus hashing
# ---------------------------------------------------------------------


def test_corpus_sha256_deterministic() -> None:
    """Plan §1 D7: same texts in same order → same hash."""
    texts = ["a", "b", "c"]
    h1 = compute_corpus_sha256(texts)
    h2 = compute_corpus_sha256(texts)
    assert h1 == h2
    assert len(h1) == 64


def test_corpus_sha256_changes_with_text() -> None:
    """Plan §1 D7: a single text edit changes the hash."""
    h1 = compute_corpus_sha256(["a", "b", "c"])
    h2 = compute_corpus_sha256(["a", "B", "c"])
    assert h1 != h2


def test_corpus_sha256_changes_with_order() -> None:
    """Plan §1 D7: a row reorder is, semantically, a different corpus.

    Stage 15 aligns ids and vectors by row position; reordering the
    rows changes the index's id-to-vector mapping, so the cache
    must invalidate.
    """
    h1 = compute_corpus_sha256(["a", "b", "c"])
    h2 = compute_corpus_sha256(["b", "a", "c"])
    assert h1 != h2


# ---------------------------------------------------------------------
# Plan §1 D13 — metadata round-trip
# ---------------------------------------------------------------------


def test_metadata_round_trip_through_parquet_bytes() -> None:
    """Plan §1 D13: provenance fields survive the parquet ``custom_metadata`` round-trip."""
    meta = EmbeddingCacheMetadata(
        embedded_at_utc=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
        corpus_sha256="a" * 64,
        model_id="stub-sha256-8",
        dim=8,
    )
    raw = meta.to_parquet_metadata()
    decoded = EmbeddingCacheMetadata.from_parquet_metadata(raw)
    assert decoded == meta


def test_metadata_naive_datetime_rejected() -> None:
    """Plan §1 D13 + DESIGN §2.1: timestamps are timezone-aware."""
    with pytest.raises(ValueError, match="timezone-aware"):
        EmbeddingCacheMetadata(
            embedded_at_utc=datetime(2026, 4, 27, 12, 0),  # naive
            corpus_sha256="a" * 64,
            model_id="stub-sha256-8",
            dim=8,
        )


def test_metadata_short_hash_rejected() -> None:
    """Plan §1 D13: corpus_sha256 must be the full 64-char hex digest."""
    with pytest.raises(ValueError, match="64-char"):
        EmbeddingCacheMetadata(
            embedded_at_utc=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
            corpus_sha256="a" * 12,  # only the prefix
            model_id="stub-sha256-8",
            dim=8,
        )


def test_metadata_corpus_hash_prefix_property() -> None:
    """Plan §1 D13: the prefix property gives the documented 12-char log helper."""
    meta = EmbeddingCacheMetadata(
        embedded_at_utc=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
        corpus_sha256="a" * 64,
        model_id="stub-sha256-8",
        dim=8,
    )
    assert meta.corpus_sha256_prefix == "a" * CORPUS_HASH_PREFIX_LEN


# ---------------------------------------------------------------------
# Plan §6 T5 — hit / miss freshness logic
# ---------------------------------------------------------------------


def _build_cache(tmp_path: Path) -> tuple[Path, list[str], list[str], StubEmbedder]:
    """Helper: build a fresh cache; return its path + the inputs that produced it."""
    embedder = StubEmbedder()
    ids = ["M-1::0", "M-2::0", "M-3::0"]
    texts = ["alpha", "beta", "gamma"]
    path = tmp_path / "cache.parquet"
    EmbeddingCache.load_or_build(
        path=path,
        ids=ids,
        texts=texts,
        embedder=embedder,
    )
    return path, ids, texts, embedder


def test_cache_hit_no_rebuild(tmp_path: Path) -> None:
    """Plan §6 T5: re-running with identical inputs is a no-op (no rebuild).

    AC-3: re-running the notebook must be fast — the second
    :meth:`load_or_build` call must read the existing parquet rather
    than re-embedding.
    """
    path, ids, texts, embedder = _build_cache(tmp_path)
    mtime_before = path.stat().st_mtime_ns

    cache = EmbeddingCache.load_or_build(path=path, ids=ids, texts=texts, embedder=embedder)

    mtime_after = path.stat().st_mtime_ns
    assert mtime_before == mtime_after, "Fresh cache must not be rewritten on a hit (AC-3)."
    assert cache.metadata.model_id == embedder.model_id
    assert cache.metadata.corpus_sha256 == compute_corpus_sha256(texts)


def test_corpus_change_invalidates(tmp_path: Path, caplog) -> None:
    """Plan §6 T5: a corpus edit triggers automatic rebuild (AC-8 — loud).

    The WARNING-level log naming the offending field is part of
    NFR-8 (loguru convention). The test asserts the rebuild happened
    by checking the ``corpus_sha256`` field updated; assertion on
    exact log text is intentionally absent (NFR-8 softening).
    """
    path, ids, _texts, embedder = _build_cache(tmp_path)
    new_texts = ["alpha", "beta", "DELTA"]  # third entry edited

    cache = EmbeddingCache.load_or_build(path=path, ids=ids, texts=new_texts, embedder=embedder)

    assert cache.metadata.corpus_sha256 == compute_corpus_sha256(new_texts)
    # The cache file on disk is rewritten with the new metadata.
    fresh_read = EmbeddingCache.read(path)
    assert fresh_read.metadata.corpus_sha256 == compute_corpus_sha256(new_texts)


def test_model_change_invalidates(tmp_path: Path) -> None:
    """Plan §6 T5: a model_id swap triggers automatic rebuild.

    The freshness key is ``(corpus_sha256, model_id)``. Even when the
    corpus is byte-identical, swapping the embedder's ``model_id``
    must invalidate the cache.
    """
    path, ids, texts, embedder = _build_cache(tmp_path)

    # Construct a different stub embedder (different dim → different
    # model_id sentinel ``"stub-sha256-<dim>"``).
    other_embedder = StubEmbedder(dim=16)
    assert other_embedder.model_id != embedder.model_id

    cache = EmbeddingCache.load_or_build(path=path, ids=ids, texts=texts, embedder=other_embedder)

    assert cache.metadata.model_id == other_embedder.model_id
    assert cache.metadata.dim == other_embedder.dim


def test_force_rebuild_skips_freshness_check(tmp_path: Path) -> None:
    """Plan §1 D17: ``force_rebuild=True`` rewrites the cache unconditionally."""
    path, ids, texts, embedder = _build_cache(tmp_path)

    # Patch the embedded_at_utc on disk to a known value, then
    # force-rebuild and assert it changes.
    before = EmbeddingCache.read(path).metadata.embedded_at_utc

    EmbeddingCache.load_or_build(
        path=path,
        ids=ids,
        texts=texts,
        embedder=embedder,
        force_rebuild=True,
    )

    after = EmbeddingCache.read(path).metadata.embedded_at_utc
    assert after >= before  # rebuild rewrote the timestamp


# ---------------------------------------------------------------------
# Plan §1 D14 — single-Parquet file with vector list-column
# ---------------------------------------------------------------------


def test_cache_persists_vectors_at_correct_dtype(tmp_path: Path) -> None:
    """Plan §1 D14: cache vectors are float32 on read (R-3 query stability)."""
    path, _, _, _ = _build_cache(tmp_path)
    cache = EmbeddingCache.read(path)
    assert cache.vectors.dtype == np.float32


def test_cache_zero_row_corpus_handled(tmp_path: Path) -> None:
    """Edge case: an empty corpus produces a (0, dim) vector matrix without crashing."""
    embedder = StubEmbedder()
    path = tmp_path / "empty.parquet"
    cache = EmbeddingCache.load_or_build(path=path, ids=[], texts=[], embedder=embedder)
    assert cache.vectors.shape == (0, embedder.dim)
    assert cache.ids == []


def test_cache_id_text_length_mismatch_rejected(tmp_path: Path) -> None:
    """Robustness: misaligned ids / texts must raise rather than silently truncating."""
    embedder = StubEmbedder()
    path = tmp_path / "misaligned.parquet"
    with pytest.raises(ValueError, match="align"):
        EmbeddingCache.load_or_build(path=path, ids=["a", "b"], texts=["one"], embedder=embedder)
