"""Spec-derived tests for the Stage 15 T3 :class:`NumpyIndex`.

Every test here is derived from:

- ``docs/plans/completed/15-embedding-index.md`` §6 T3 named test
  ``test_query_returns_sorted_top_k`` plus
- AC-2 (intent line 35): "nearest-neighbour queries return in well
  under a second".
- AC-6 (requirements §3): "Similarity scores rendered alongside
  neighbours" — cosine score in ``[-1, 1]``.
- Plan §1 D3 (sorted descending), D5 (numpy matmul backend, save/load
  round-trip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bristol_ml.embeddings import NearestNeighbour
from bristol_ml.embeddings._embedder import StubEmbedder
from bristol_ml.embeddings._index import NumpyIndex, StubIndex

# ---------------------------------------------------------------------
# Plan §6 T3 — top-k sorted descending
# ---------------------------------------------------------------------


def _build_index_with_5_docs(index_cls) -> tuple[object, list[str]]:
    """Helper: populate a tiny stub-embedded corpus for the query tests."""
    embedder = StubEmbedder()
    texts = [
        "Outage Planned Nuclear T_HARTLEPOOL-1",
        "Restriction Forced Gas T_PEMBROKE-1",
        "Outage Unplanned Coal RATCLIFFE-1",
        "Outage Planned Wind GORDONBUSH-1",
        "Outage Planned Solar SOLARFARM-2",
    ]
    ids = [f"M-{i}::0" for i in range(len(texts))]
    vectors = embedder.embed_batch(texts)
    index = index_cls(dim=embedder.dim)
    index.add(ids, vectors)
    return index, ids


@pytest.mark.parametrize("index_cls", [NumpyIndex, StubIndex])
def test_query_returns_sorted_top_k(index_cls) -> None:
    """Plan §6 T3 / §1 D3: query returns top-k in sorted-descending order.

    The contract: :meth:`VectorIndex.query` returns
    ``list[NearestNeighbour]`` of length ``min(k, len(self))``,
    ordered by ``score`` descending.
    """
    index, _ = _build_index_with_5_docs(index_cls)
    embedder = StubEmbedder()
    query = embedder.embed("planned nuclear outage")
    neighbours = index.query(query, k=3)

    assert len(neighbours) == 3
    assert all(isinstance(nn, NearestNeighbour) for nn in neighbours)
    scores = [nn.score for nn in neighbours]
    assert scores == sorted(scores, reverse=True), (
        f"query must return scores in descending order; got {scores}."
    )


@pytest.mark.parametrize("index_cls", [NumpyIndex, StubIndex])
def test_query_clips_k_to_index_size(index_cls) -> None:
    """Plan §1 D3: ``k > len(index)`` clips silently to ``len(index)``."""
    index, ids = _build_index_with_5_docs(index_cls)
    embedder = StubEmbedder()
    query = embedder.embed("any")
    neighbours = index.query(query, k=100)
    assert len(neighbours) == len(ids)


@pytest.mark.parametrize("index_cls", [NumpyIndex, StubIndex])
def test_query_scores_in_cosine_range(index_cls) -> None:
    """Plan §1 D3 / AC-6: cosine score is in ``[-1, 1]`` for normalised vectors."""
    index, _ = _build_index_with_5_docs(index_cls)
    embedder = StubEmbedder()
    query = embedder.embed("any text")
    neighbours = index.query(query, k=5)
    for nn in neighbours:
        # Tolerance for fp32 round-trip jitter — a normalised matmul
        # can land at 1.0 + a few ULPs on the matched-self path.
        assert -1.0 - 1e-5 <= nn.score <= 1.0 + 1e-5, (
            f"score {nn.score} outside cosine range [-1, 1] for id {nn.id}."
        )


@pytest.mark.parametrize("index_cls", [NumpyIndex, StubIndex])
def test_query_self_match_scores_one(index_cls) -> None:
    """Sanity: querying with a doc vector should put that doc top with score≈1.0.

    Note: the stub embedder's ``embed`` and ``embed_batch`` paths
    differ by the QUERY_PREFIX, so this test queries by re-using a
    document vector directly (not via ``embed``) to test the index's
    cosine identity rather than the prefix asymmetry.
    """
    index, ids = _build_index_with_5_docs(index_cls)
    # Reach in for a known document vector. Both index types store
    # vectors row-aligned with ``ids``; we reconstruct the first one
    # from the embedder rather than reaching into private state.
    embedder = StubEmbedder()
    doc_texts = [
        "Outage Planned Nuclear T_HARTLEPOOL-1",
        "Restriction Forced Gas T_PEMBROKE-1",
        "Outage Unplanned Coal RATCLIFFE-1",
        "Outage Planned Wind GORDONBUSH-1",
        "Outage Planned Solar SOLARFARM-2",
    ]
    target_vec = embedder.embed_batch([doc_texts[0]])[0]
    neighbours = index.query(target_vec, k=1)
    assert neighbours[0].id == ids[0]
    assert abs(neighbours[0].score - 1.0) < 1e-4, (
        f"self-match score must be ~1.0; got {neighbours[0].score}."
    )


# ---------------------------------------------------------------------
# AC-2 — well under a second
# ---------------------------------------------------------------------


def test_query_at_1k_corpus_runs_under_a_second() -> None:
    """Plan AC-2: nearest-neighbour queries return in well under a second.

    This is a smoke test, not a benchmark — it asserts orders of
    magnitude. The 1000-row stub-corpus query lands in milliseconds
    on any laptop CPU; ``< 1.0 s`` is a generous ceiling.
    """
    import time

    embedder = StubEmbedder()
    n = 1_000
    ids = [f"M-{i}::0" for i in range(n)]
    texts = [f"document number {i}" for i in range(n)]
    vectors = embedder.embed_batch(texts)
    index = NumpyIndex(dim=embedder.dim)
    index.add(ids, vectors)

    query = embedder.embed("query text")
    t0 = time.perf_counter()
    neighbours = index.query(query, k=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"query took {elapsed:.3f}s on n={n}; AC-2 violated."
    assert len(neighbours) == 10


# ---------------------------------------------------------------------
# Plan §1 D5 — round-trip save/load
# ---------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", [NumpyIndex, StubIndex])
def test_index_save_load_round_trip(index_cls, tmp_path: Path) -> None:
    """Plan §1 D5: ``save`` + ``load`` reconstructs an equivalent index."""
    index, _ids = _build_index_with_5_docs(index_cls)
    embedder = StubEmbedder()
    query = embedder.embed("planned nuclear outage")
    neighbours_before = index.query(query, k=3)

    path = tmp_path / "index.npz"
    index.save(path)
    assert path.exists()

    reloaded = index_cls.load(path)
    assert reloaded.dim == index.dim
    neighbours_after = reloaded.query(query, k=3)
    # Same ids, same scores (modulo fp32 round-trip).
    assert [nn.id for nn in neighbours_after] == [nn.id for nn in neighbours_before]
    np.testing.assert_allclose(
        [nn.score for nn in neighbours_after],
        [nn.score for nn in neighbours_before],
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Plan §1 D5 — defensive renormalisation
# ---------------------------------------------------------------------


def test_index_renormalises_on_add() -> None:
    """Plan §1 D5: even if a caller passes non-normalised vectors, queries still work.

    The Embedder contract guarantees normalised inputs, but a hand-
    rolled fixture might not honour it. Defensive renormalisation at
    :meth:`VectorIndex.add` keeps cosine-as-matmul load-bearing.
    """
    n, dim = 4, 8
    rng = np.random.default_rng(seed=42)
    raw = rng.normal(size=(n, dim)).astype(np.float32)  # NOT normalised
    raw *= 5.0  # blow them up further
    ids = [f"M-{i}::0" for i in range(n)]

    index = NumpyIndex(dim=dim)
    index.add(ids, raw)

    # Build a query that is exactly the renormalised first row.
    expected_top = raw[0] / np.linalg.norm(raw[0])
    neighbours = index.query(expected_top, k=1)
    assert neighbours[0].id == ids[0]
    assert abs(neighbours[0].score - 1.0) < 1e-4
