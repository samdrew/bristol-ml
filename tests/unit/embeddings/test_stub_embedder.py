"""Spec-derived tests for the Stage 15 T2 :class:`StubEmbedder`.

Every test here is derived from:

- ``docs/plans/active/15-embedding-index.md`` §6 T2 named tests:
  ``test_stub_default_when_env_var_set``,
  ``test_stub_returns_same_vector_per_text``.
- AC-4 (intent line 33): "the embedding model has no external API
  dependency — it runs locally".
- Plan §1 D8 (triple-gated stub-first) + D9 (prefix-asymmetry contract
  the stub mirrors so a regression in either direction is catchable).
"""

from __future__ import annotations

import numpy as np

from bristol_ml.embeddings._embedder import (
    DEFAULT_STUB_DIM,
    QUERY_PREFIX,
    StubEmbedder,
)
from bristol_ml.embeddings._factory import STUB_ENV_VAR, build_embedder

# ---------------------------------------------------------------------
# Plan §6 T2 — env-var triple-gate
# ---------------------------------------------------------------------


def test_stub_default_when_env_var_set(monkeypatch) -> None:
    """Plan §1 D8: ``BRISTOL_ML_EMBEDDING_STUB=1`` forces stub regardless of YAML.

    The env-var override outranks YAML so CI / offline runs can keep
    network-free behaviour even when the YAML in a feature branch
    has flipped to ``type: sentence_transformers``.
    """
    from conf._schemas import EmbeddingConfig

    monkeypatch.setenv(STUB_ENV_VAR, "1")
    # YAML claims the live path; env-var must override.
    cfg = EmbeddingConfig(type="sentence_transformers", model_id="some/model")
    embedder = build_embedder(cfg)
    assert isinstance(embedder, StubEmbedder), (
        "build_embedder must honour BRISTOL_ML_EMBEDDING_STUB=1 even when "
        "YAML config selects the live path."
    )


def test_stub_default_when_yaml_type_is_stub() -> None:
    """Plan §1 D8: YAML ``type: stub`` returns the stub embedder."""
    from conf._schemas import EmbeddingConfig

    cfg = EmbeddingConfig(type="stub")
    embedder = build_embedder(cfg)
    assert isinstance(embedder, StubEmbedder)
    assert embedder.dim == DEFAULT_STUB_DIM
    assert embedder.model_id == f"stub-sha256-{DEFAULT_STUB_DIM}"


# ---------------------------------------------------------------------
# Plan §6 T2 — determinism + L2 normalisation
# ---------------------------------------------------------------------


def test_stub_returns_same_vector_per_text() -> None:
    """Plan §6 T2: the stub is deterministic — same text → identical vector."""
    embedder = StubEmbedder()
    a1 = embedder.embed("nuclear outage at Hartlepool")
    a2 = embedder.embed("nuclear outage at Hartlepool")
    np.testing.assert_array_equal(a1, a2)


def test_stub_returns_distinct_vectors_per_distinct_text() -> None:
    """Plan §6 T2: the stub distinguishes texts.

    SHA-256 produces vectors that are near-orthogonal in expectation
    so any two distinct strings get distinguishable embeddings; this
    is load-bearing for the cosine-distance test in
    ``test_index_query.py``.
    """
    embedder = StubEmbedder()
    a = embedder.embed("nuclear outage at Hartlepool")
    b = embedder.embed("solar restriction at SOLARFARM-2")
    assert not np.array_equal(a, b)


def test_stub_vectors_are_l2_normalised() -> None:
    """Plan §1 D2: the embedder returns L2-normalised float32 vectors."""
    embedder = StubEmbedder()
    vec = embedder.embed("any text")
    assert vec.dtype == np.float32
    assert vec.shape == (embedder.dim,)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Stub vector must be L2-normalised; got norm={norm}."


def test_stub_batch_returns_2d_normalised_matrix() -> None:
    """Plan §1 D2: ``embed_batch`` returns ``(n, dim)`` float32, row-normalised."""
    embedder = StubEmbedder()
    matrix = embedder.embed_batch(["a", "b", "c"])
    assert matrix.dtype == np.float32
    assert matrix.shape == (3, embedder.dim)
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_stub_empty_batch_returns_zero_rows() -> None:
    """Plan §1 D2: empty batch is a valid call returning shape (0, dim)."""
    embedder = StubEmbedder()
    matrix = embedder.embed_batch([])
    assert matrix.shape == (0, embedder.dim)
    assert matrix.dtype == np.float32


# ---------------------------------------------------------------------
# Plan §1 D9 / T4 — prefix asymmetry mirrored on the stub
# ---------------------------------------------------------------------


def test_stub_query_and_document_paths_diverge() -> None:
    """Plan §1 D9 / T4: ``embed(text)`` (query) ≠ ``embed_batch([text])[0]`` (doc).

    The prefix asymmetry on the live :class:`SentenceTransformerEmbedder`
    is mirrored by the stub: :meth:`embed` prepends
    :data:`QUERY_PREFIX`; :meth:`embed_batch` does not. A unit test
    asserting non-equality catches a downstream regression where a
    caller routes a query through the document path or vice-versa.
    """
    embedder = StubEmbedder()
    text = "planned nuclear outage"
    query_vec = embedder.embed(text)
    doc_vec = embedder.embed_batch([text])[0]
    assert not np.array_equal(query_vec, doc_vec), (
        "Stub must mirror the live embedder's prefix asymmetry: "
        "embed(text) (query path) and embed_batch([text])[0] (doc path) "
        "must hash differently because QUERY_PREFIX is prepended on the "
        "query path only."
    )


def test_query_prefix_constant_matches_e5_gte_convention() -> None:
    """Plan §1 D4 / D9 — pin the documented E5 / GTE-modernbert prefix string."""
    assert QUERY_PREFIX == "query: ", (
        "QUERY_PREFIX must be 'query: ' per the E5 / GTE-modernbert "
        "asymmetric prefix convention (plan §1 D4 + D9)."
    )
