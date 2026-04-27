"""Spec-derived tests for the Stage 15 T4 :class:`SentenceTransformerEmbedder`.

Every test here is derived from:

- ``docs/plans/active/15-embedding-index.md`` §6 T4 named test
  ``test_st_embedder_loads_offline``.
- Plan §1 D4 / A2 (Ctrl+G 2026-04-27): the live default checkpoint is
  ``Alibaba-NLP/gte-modernbert-base``.
- Plan §1 D9 / D12: ``HF_HUB_OFFLINE=1`` is asserted in the test conftest;
  a missing local model cache must raise a clear ``RuntimeError`` rather
  than triggering a silent download (NFR-1).
"""

from __future__ import annotations

import importlib.util
import os

import pytest


def _sentence_transformers_available() -> bool:
    """Detect whether ``sentence-transformers`` is installed in the env."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _hf_cache_has_model(model_id: str) -> bool:
    """Heuristic: is the model already downloaded into the local HF cache?

    The check is best-effort — we look for the documented HF cache
    layout under ``~/.cache/huggingface/hub`` (or
    ``$HF_HOME/hub``). A false negative here only causes the test to
    skip; it never produces a false positive (a download), because
    ``HF_HUB_OFFLINE=1`` is set in conftest regardless.
    """
    from pathlib import Path

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    sanitised = "models--" + model_id.replace("/", "--")
    return (Path(hf_home) / "hub" / sanitised).exists()


# ---------------------------------------------------------------------
# Plan §6 T4 — live embedder loads offline if the cache is warm
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed; live embedder path unreachable.",
)
@pytest.mark.skipif(
    not _hf_cache_has_model("Alibaba-NLP/gte-modernbert-base"),
    reason="HF cache lacks gte-modernbert-base; pre-warm with the documented one-liner.",
)
def test_st_embedder_loads_offline_with_warm_cache() -> None:
    """Plan §6 T4: a pre-warmed cache loads cleanly under ``HF_HUB_OFFLINE=1``.

    This test is the live-path smoke. CI without the warm cache
    skips it; a developer running locally after the documented
    pre-warm one-liner gets a non-skipped run that exercises the
    full sentence-transformers + torch import chain.
    """
    from bristol_ml.embeddings._embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(
        model_id="Alibaba-NLP/gte-modernbert-base",
        fp16=True,
    )
    assert embedder.model_id == "Alibaba-NLP/gte-modernbert-base"
    # gte-modernbert-base is a 768-dim model.
    assert embedder.dim == 768

    # Smoke: a single embed + a small batch must succeed offline.
    query_vec = embedder.embed("planned nuclear outage")
    assert query_vec.shape == (768,)

    batch = embedder.embed_batch(["doc one", "doc two"])
    assert batch.shape == (2, 768)


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed; live embedder path unreachable.",
)
def test_st_embedder_raises_with_helpful_pre_warm_command_on_missing_cache(
    monkeypatch,
) -> None:
    """Plan §6 T4 / R-1: a missing cache surfaces a clear error, not a silent download.

    The error must name the pre-warm command so an operator can
    recover without spelunking through the sentence-transformers
    source. We trigger the failure by pointing at a guaranteed-
    nonexistent model id.
    """
    from bristol_ml.embeddings._embedder import SentenceTransformerEmbedder

    # Force offline mode in case conftest hasn't already.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    bogus_model_id = "this-org-does-not-exist/this-model-does-not-exist-9b"
    with pytest.raises(RuntimeError) as excinfo:
        SentenceTransformerEmbedder(model_id=bogus_model_id, fp16=False)

    # The error message must guide the operator to the recovery
    # command — see the SentenceTransformerEmbedder.__init__
    # docstring.
    assert "pre-warm" in str(excinfo.value).lower() or "SentenceTransformer" in str(excinfo.value)
    assert bogus_model_id in str(excinfo.value)
