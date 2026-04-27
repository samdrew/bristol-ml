"""Stage 15 — factory functions and env-var constants.

Plan §1 D8: triple-gated stub-first dispatch.

- :data:`STUB_ENV_VAR` (``BRISTOL_ML_EMBEDDING_STUB``) — when set to
  ``"1"``, every :func:`build_embedder` call returns a
  :class:`StubEmbedder` regardless of YAML configuration. Mirrors
  Stage 14's ``BRISTOL_ML_LLM_STUB`` discipline: load-bearing for CI
  safety and offline runs.
- :data:`MODEL_PATH_ENV_VAR` (``BRISTOL_ML_EMBEDDING_MODEL_PATH``) —
  optional override for the live model id. Lets a developer point at
  a local cache directory copy without editing YAML.

The two factory functions :func:`build_embedder` and
:func:`build_index` are the single dispatch points the public
boundary exposes. :func:`embed_corpus` glues them together with
:class:`bristol_ml.embeddings.EmbeddingCache` so a notebook /
standalone CLI caller need only pass an
:class:`~conf._schemas.EmbeddingConfig` and a corpus DataFrame.

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md``.
- Stage 15 plan — ``docs/plans/active/15-embedding-index.md`` §6 T7.
- Stage 14 prior art — :func:`bristol_ml.llm.extractor.build_extractor`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger

from bristol_ml.embeddings._cache import EmbeddingCache
from bristol_ml.embeddings._embedder import StubEmbedder
from bristol_ml.embeddings._index import NumpyIndex, StubIndex
from bristol_ml.embeddings._text import synthesise_embeddable_text

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    import pandas as pd

    from bristol_ml.embeddings._protocols import Embedder, VectorIndex
    from conf._schemas import EmbeddingConfig

__all__ = [
    "MODEL_PATH_ENV_VAR",
    "STUB_ENV_VAR",
    "build_embedder",
    "build_index",
    "embed_corpus",
]


# Plan §1 D8: env-var triple-gate. Public so tests can monkeypatch one
# canonical name instead of the literal string.
STUB_ENV_VAR: Final[str] = "BRISTOL_ML_EMBEDDING_STUB"
MODEL_PATH_ENV_VAR: Final[str] = "BRISTOL_ML_EMBEDDING_MODEL_PATH"


# ---------------------------------------------------------------------
# build_embedder — discriminator + env-var dispatch
# ---------------------------------------------------------------------


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Return an :class:`Embedder` instance per ``config`` and env state.

    Plan §1 D8 (triple-gated stub-first):

    1. If ``BRISTOL_ML_EMBEDDING_STUB=1``, return :class:`StubEmbedder`.
       This wins over YAML to keep CI / offline runs safe.
    2. Else if ``config.type == "stub"``, return :class:`StubEmbedder`.
    3. Else (``config.type == "sentence_transformers"``), import the
       heavy backend lazily and load the model.

    The lazy import in step 3 means a stub-only import of
    ``bristol_ml.embeddings`` does *not* drag in
    ``sentence-transformers`` / ``torch`` (AC-1 sub-criterion at
    plan §6 T7).

    The optional ``BRISTOL_ML_EMBEDDING_MODEL_PATH`` env var, when
    set, replaces ``config.model_id`` for the live path — useful when
    a developer wants to point at a side-loaded local model without
    editing YAML.
    """
    if os.environ.get(STUB_ENV_VAR) == "1":
        logger.info(
            "build_embedder: {}=1 — returning StubEmbedder (overrides YAML type={!r}).",
            STUB_ENV_VAR,
            config.type,
        )
        return StubEmbedder()

    if config.type == "stub":
        logger.debug("build_embedder: YAML type='stub' — returning StubEmbedder.")
        return StubEmbedder()

    # Live path. Defer the import so a stub-only test environment
    # doesn't pay the sentence-transformers + torch import cost.
    from bristol_ml.embeddings._embedder import SentenceTransformerEmbedder

    model_id = os.environ.get(MODEL_PATH_ENV_VAR) or config.model_id
    if not model_id:
        raise RuntimeError(
            "EmbeddingConfig.model_id must be set when type='sentence_transformers' "
            f"(or pass an override via {MODEL_PATH_ENV_VAR}). "
            "Plan §1 A2 default: 'Alibaba-NLP/gte-modernbert-base'."
        )
    logger.info(
        "build_embedder: loading SentenceTransformerEmbedder (model_id={}, fp16={}).",
        model_id,
        config.fp16,
    )
    return SentenceTransformerEmbedder(model_id=model_id, fp16=config.fp16)


# ---------------------------------------------------------------------
# build_index — discriminator dispatch
# ---------------------------------------------------------------------


def build_index(config: EmbeddingConfig, *, dim: int) -> VectorIndex:
    """Return a :class:`VectorIndex` per ``config.vector_backend``.

    Plan §1 D5: the two production dispatches are ``"numpy"`` (the
    default — :class:`NumpyIndex`) and ``"stub"`` (unit-test only —
    :class:`StubIndex`). A future FAISS / hnswlib swap is one new
    branch here; the :class:`VectorIndex` Protocol does not change.

    ``dim`` must come from the constructed
    :class:`Embedder.dim` — the index is dimensionally tied to the
    embedder. A mismatch is a programming error and is rejected at
    :meth:`VectorIndex.add` time anyway.
    """
    if config.vector_backend == "stub":
        return StubIndex(dim=dim)
    if config.vector_backend == "numpy":
        return NumpyIndex(dim=dim)
    # Pydantic Literal already prevents this; pragma for safety.
    raise ValueError(  # pragma: no cover
        f"Unknown vector_backend {config.vector_backend!r}. Expected one of {{'numpy', 'stub'}}."
    )


# ---------------------------------------------------------------------
# embed_corpus — glue: cache build + index hydration
# ---------------------------------------------------------------------


# Plan §1 D14: cache file path is
# ``data/embeddings/<model_id_sanitised>.parquet``. The sanitiser
# replaces non-filesystem-safe characters with ``_`` so a model id
# like ``"Alibaba-NLP/gte-modernbert-base"`` becomes
# ``"Alibaba-NLP_gte-modernbert-base"``.
_FILESYSTEM_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitised_model_id(model_id: str) -> str:
    """Return a filesystem-safe version of ``model_id`` for the cache path."""
    return _FILESYSTEM_SAFE.sub("_", model_id).strip("_") or "unnamed_model"


def _default_cache_path(model_id: str) -> Path:
    """Project-default cache path: ``data/embeddings/<sanitised>.parquet``.

    Anchored to the repo root via ``Path(__file__).resolve().parents[3]``
    (Stage 14 ``LlmExtractor`` precedent) so the path is the same
    regardless of the caller's CWD.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "embeddings" / f"{_sanitised_model_id(model_id)}.parquet"


def embed_corpus(
    *,
    config: EmbeddingConfig,
    corpus: pd.DataFrame,
    id_columns: tuple[str, str] = ("mrid", "revision_number"),
) -> tuple[VectorIndex, EmbeddingCache]:
    """End-to-end: build / load the cache, return a populated index + cache.

    This is the function the Stage 15 notebook (T9) and the standalone
    module (T8) both call. It glues:

    1. :func:`bristol_ml.embeddings.synthesise_embeddable_text` — the
       NULL-aware text-coercion pass over the corpus rows (plan §1 D9).
    2. :func:`build_embedder` — env-var-aware dispatch.
    3. :meth:`EmbeddingCache.load_or_build` — content-addressed
       freshness check, rebuild on stale.
    4. :func:`build_index` + :meth:`VectorIndex.add` — populate the
       in-memory index with the cached vectors.

    Parameters
    ----------
    config
        The validated :class:`~conf._schemas.EmbeddingConfig`.
    corpus
        A pandas DataFrame matching the Stage 13
        ``OUTPUT_SCHEMA``. Must carry the ``id_columns`` plus the
        ``message_description`` / ``event_type`` / ``cause`` /
        ``fuel_type`` / ``affected_unit`` columns the synthesiser
        reads from.
    id_columns
        The two-column tuple Stage 15 uses as the row id grain. The
        default is the Stage 13 primary key ``(mrid, revision_number)``;
        the index id is the joined string ``"<mrid>::<revision_number>"``.

    Returns
    -------
    (VectorIndex, EmbeddingCache)
        The populated nearest-neighbour index plus the cache it was
        built from. The cache exposes the provenance fields (D13) for
        notebook display and log records.
    """
    embedder = build_embedder(config)

    # --- text synthesis pass (plan §1 D9) -----------------------------
    texts = [synthesise_embeddable_text(row) for _, row in corpus.iterrows()]
    ids = [
        f"{_coerce_id_part(row[id_columns[0]])}::{_coerce_id_part(row[id_columns[1]])}"
        for _, row in corpus.iterrows()
    ]

    # --- cache resolution (plan §1 D14) --------------------------------
    cache_path = config.cache_path or _default_cache_path(embedder.model_id)
    cache = EmbeddingCache.load_or_build(
        path=cache_path,
        ids=ids,
        texts=texts,
        embedder=embedder,
        force_rebuild=config.force_rebuild,
    )

    # --- index hydration (plan §1 D5) ----------------------------------
    index = build_index(config, dim=cache.metadata.dim)
    if cache.ids:
        index.add(cache.ids, cache.vectors)
    logger.info(
        "embed_corpus: index hydrated (n={}, dim={}, backend={}).",
        len(cache.ids),
        cache.metadata.dim,
        config.vector_backend,
    )
    return index, cache


def _coerce_id_part(value: object) -> str:
    """Render an id column cell to a stable string key.

    pandas may surface integer ``revision_number`` as a numpy scalar;
    the cast through ``str()`` is consistent across pandas versions.
    """
    if value is None:
        return "NULL"
    return str(value)
