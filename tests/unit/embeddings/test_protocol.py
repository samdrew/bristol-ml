"""Spec-derived tests for the Stage 15 T1 typed boundary.

Every test here is derived from:

- ``docs/plans/completed/15-embedding-index.md`` §6 T1 named test
  ``test_embedder_and_index_satisfy_protocol_structurally`` plus
  AC-5 (intent line 34): "the index's interface is small enough that
  swapping the vector store implementation later is a mechanical
  change". NFR-4 caps the surface at two methods on
  :class:`Embedder` and four on :class:`VectorIndex`.
- Plan §1 D2 + D3 + D16: ``runtime_checkable`` Protocols, ADR-0008.
- Plan §1 D2 prefix-asymmetry contract (E5/GTE-modernbert).

No production code is modified here; if a test below fails, the
failure points at a deviation from the plan.
"""

from __future__ import annotations

import importlib
import sys
from typing import Protocol

import numpy as np

from bristol_ml.embeddings import (
    Embedder,
    NearestNeighbour,
    VectorIndex,
)
from bristol_ml.embeddings._embedder import StubEmbedder
from bristol_ml.embeddings._index import NumpyIndex, StubIndex

# ---------------------------------------------------------------------
# AC-5 / NFR-4 — small interface
# ---------------------------------------------------------------------


def test_embedder_protocol_has_two_methods() -> None:
    """Plan AC-5 / NFR-4 / D2: the Embedder Protocol exposes exactly two methods.

    AC-5 caps the public surface so adding a method becomes a plan-
    edit conversation rather than a quiet code edit. The two methods
    are :meth:`Embedder.embed` (single, query-shaped) and
    :meth:`Embedder.embed_batch` (batch, document-shaped).
    """
    assert issubclass(Embedder, Protocol), (
        "Embedder must be a typing.Protocol so a third implementation "
        "can satisfy it without inheriting (NFR-4)."
    )
    method_names = {
        name
        for name, value in vars(Embedder).items()
        if callable(value) and not name.startswith("_")
    }
    assert method_names == {"embed", "embed_batch"}, (
        "Embedder Protocol must expose exactly embed + embed_batch "
        f"(plan D2 / NFR-4); got {sorted(method_names)!r}."
    )


def test_vector_index_protocol_has_four_methods() -> None:
    """Plan AC-5 / NFR-4 / D3: the VectorIndex Protocol exposes exactly four methods.

    The four methods are :meth:`VectorIndex.add`,
    :meth:`VectorIndex.query`, :meth:`VectorIndex.save`,
    :meth:`VectorIndex.load`. Growth past four is a plan-edit
    conversation.
    """
    assert issubclass(VectorIndex, Protocol), (
        "VectorIndex must be a typing.Protocol so a future FAISS "
        "implementation can satisfy it without inheriting (NFR-4)."
    )
    method_names = {
        name
        for name, value in vars(VectorIndex).items()
        if callable(value) and not name.startswith("_") and not isinstance(value, classmethod)
    }
    # ``load`` is declared as a classmethod — strip the descriptor and
    # add by name so the count is exact regardless of the descriptor
    # surface.
    classmethod_names = {
        name
        for name, value in vars(VectorIndex).items()
        if isinstance(value, classmethod) and not name.startswith("_")
    }
    all_methods = method_names | classmethod_names
    assert all_methods == {"add", "query", "save", "load"}, (
        "VectorIndex Protocol must expose exactly add + query + save + load "
        f"(plan D3 / NFR-4); got {sorted(all_methods)!r}."
    )


def test_embedder_and_index_satisfy_protocol_structurally() -> None:
    """Plan §6 T1 named test — runtime_checkable conformance for both stubs.

    The stub implementations are the unit-test default. They must
    structurally satisfy the Protocols without inheritance — this is
    ADR-0008's load-bearing claim about Protocol-over-ABC swap-safety.
    """
    stub_embedder = StubEmbedder()
    assert isinstance(stub_embedder, Embedder), (
        "StubEmbedder must satisfy the Embedder Protocol via runtime_checkable (ADR-0008)."
    )

    stub_index = StubIndex(dim=stub_embedder.dim)
    assert isinstance(stub_index, VectorIndex), (
        "StubIndex must satisfy the VectorIndex Protocol via runtime_checkable (ADR-0008)."
    )

    numpy_index = NumpyIndex(dim=stub_embedder.dim)
    assert isinstance(numpy_index, VectorIndex), (
        "NumpyIndex must satisfy the VectorIndex Protocol via runtime_checkable (ADR-0008)."
    )


# ---------------------------------------------------------------------
# AC-1 sub-criterion — boundary import is light
# ---------------------------------------------------------------------


def test_boundary_import_does_not_pull_sentence_transformers() -> None:
    """Plan AC-1 sub-criterion: importing ``bristol_ml.embeddings`` is light.

    Stage 16 callers holding only the Protocol type must not pay the
    sentence-transformers import cost. The split between the package
    ``__init__`` (boundary types + factory) and ``_embedder`` (heavy
    live backend, with deferred imports) is the load-bearing mechanism.

    Implementation note (cross-suite hygiene): we deliberately do NOT
    evict ``torch`` from ``sys.modules`` here.  ``torch.__init__``
    contains a module-level ``torch.library.Library("triton", "DEF")``
    registration; popping torch and letting a sibling test re-import
    it triggers a global-state ``RuntimeError`` ("Only a single
    TORCH_LIBRARY can be used to register the namespace triton")
    that cascades into every subsequent torch-using test in the
    suite (Stage 10/11 nn-MLP / nn-temporal, Stage 9 registry
    dispatch).  Torch is allowed to be pre-loaded; the load-bearing
    invariant for *this* test is only that
    ``sentence_transformers`` stays out of the boundary import.
    """
    # Evict the embeddings package + every submodule; any cached
    # ``sentence_transformers`` import from a sibling test would
    # confound this test's premise.
    for name in list(sys.modules):
        if name == "bristol_ml.embeddings" or name.startswith("bristol_ml.embeddings."):
            sys.modules.pop(name, None)

    sys.modules.pop("sentence_transformers", None)

    pkg = importlib.import_module("bristol_ml.embeddings")
    assert hasattr(pkg, "Embedder")
    assert hasattr(pkg, "VectorIndex")

    # The cold-path import alone must not have triggered the heavy
    # SDKs. ``sentence_transformers`` lives behind the live-path lazy
    # import in :class:`SentenceTransformerEmbedder.__init__`.
    assert "sentence_transformers" not in sys.modules, (
        "Importing ``bristol_ml.embeddings`` pulled in "
        "``sentence_transformers``. The boundary/implementation split "
        "has been broken; Stage 16 would inherit the heavy SDK as a "
        "transitive dep."
    )


# ---------------------------------------------------------------------
# NearestNeighbour — the typed return shape
# ---------------------------------------------------------------------


def test_nearest_neighbour_is_named_tuple_with_id_and_score() -> None:
    """Plan §1 D3: ``query`` returns ``list[(id, score)]`` as named pairs.

    A :class:`~typing.NamedTuple` is lighter than a Pydantic model at
    the per-row tight-loop call site (the index returns its own
    constructed pairs, so per-row validation is wasted).
    """
    nn = NearestNeighbour(id="M-X::0", score=0.42)
    # Tuple-shaped (positional + named access).
    assert nn[0] == "M-X::0"
    assert nn[1] == 0.42
    assert nn.id == "M-X::0"
    assert nn.score == 0.42
    # Field set is exactly two — no growth without a plan edit.
    assert NearestNeighbour._fields == ("id", "score")


# ---------------------------------------------------------------------
# Empty-index discipline — D16
# ---------------------------------------------------------------------


def test_empty_index_query_returns_empty_list() -> None:
    """Plan §1 D3: ``query`` on an empty index returns ``[]``, not raises.

    Mirrors :class:`StubExtractor`'s miss-path discipline (Stage 14
    D16) — callers receive a typed-empty result they can branch on.
    """
    stub_embedder = StubEmbedder()
    query_vec = stub_embedder.embed("any text")
    for empty in (StubIndex(dim=stub_embedder.dim), NumpyIndex(dim=stub_embedder.dim)):
        assert empty.query(query_vec, k=10) == []


# ---------------------------------------------------------------------
# Pydantic schema round-trip — EmbeddingConfig
# ---------------------------------------------------------------------


def test_embedding_config_field_set_matches_plan_schema() -> None:
    """Plan §1 D2 + D5 + D6 + D8 + D14 + D17: EmbeddingConfig field set is the contract.

    Drift on this field set silently changes the YAML schema callers
    rely on; the test pins it.
    """
    from conf._schemas import EmbeddingConfig

    expected = {
        "type",
        "model_id",
        "cache_path",
        "vector_backend",
        "default_top_k",
        "projection_type",
        "force_rebuild",
        "fp16",
    }
    actual = set(EmbeddingConfig.model_fields.keys())
    assert actual == expected, (
        "EmbeddingConfig field set has drifted from plan §1 D-rows; "
        f"expected {sorted(expected)!r}, got {sorted(actual)!r}."
    )


def test_embedding_config_extra_keys_forbidden() -> None:
    """Plan NFR-3: extra keys at the YAML boundary raise ValidationError."""
    import pytest as _pytest
    from pydantic import ValidationError

    from conf._schemas import EmbeddingConfig

    with _pytest.raises(ValidationError):
        EmbeddingConfig(type="stub", typo_field="oops")  # type: ignore[call-arg]


def test_embedding_config_is_frozen() -> None:
    """Plan NFR-3: frozen=True — config is immutable post-construction."""
    import pytest as _pytest
    from pydantic import ValidationError

    from conf._schemas import EmbeddingConfig

    cfg = EmbeddingConfig(type="stub")
    with _pytest.raises(ValidationError):
        cfg.type = "sentence_transformers"  # type: ignore[misc]


# Silence unused-import lint warning for numpy — it's used via the
# typing import path in :class:`Embedder`'s annotations indirectly.
_ = np
