"""Stage 15 — concrete :class:`Embedder` implementations.

Two implementations selected by
:class:`conf._schemas.EmbeddingConfig`'s ``type`` discriminator
(plan §1 D2 + D8) plus the ``BRISTOL_ML_EMBEDDING_STUB`` env var
(plan §1 D8 — triple-gated stub-first):

- :class:`StubEmbedder` — the offline-by-default path (AC-4). Produces
  deterministic SHA-256-derived float32 vectors; no network, no model
  download, no tokeniser. ``dim`` defaults to 8 — small enough to keep
  the unit tests fast, large enough to give the cosine-similarity
  contract teeth.
- :class:`SentenceTransformerEmbedder` — the live path. Loads
  ``Alibaba-NLP/gte-modernbert-base`` (768-dim) by default with
  ``model_kwargs={"torch_dtype": torch.float16}`` so RAM stays under
  ~150 MB. ``HF_HUB_OFFLINE=1`` is set in the project ``conftest.py``
  to assert no network call escapes; a missing local model cache
  raises a clear ``RuntimeError`` rather than triggering a silent
  download.

Both implementations satisfy the
:class:`bristol_ml.embeddings.Embedder` Protocol structurally —
:func:`runtime_checkable` lets the unit tests assert this without
inheritance (ADR-0003 / ADR-0008 precedent).

The :func:`bristol_ml.embeddings.build_embedder` factory in
``_factory.py`` is the single dispatch point so callers (the
notebook, the standalone CLI, Stage 16 tests) need only pass an
:class:`EmbeddingConfig`. The factory honours
``BRISTOL_ML_EMBEDDING_STUB=1`` regardless of YAML — load-bearing for
CI safety (plan §1 D8).

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md``.
- Stage 15 plan — ``docs/plans/active/15-embedding-index.md`` §6 T2 + T4.
- Boundary types — :mod:`bristol_ml.embeddings` (``__init__.py``).
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
from loguru import logger

__all__ = [
    "DEFAULT_STUB_DIM",
    "QUERY_PREFIX",
    "SentenceTransformerEmbedder",
    "StubEmbedder",
]


# Plan §1 D9 + A2: the live default checkpoint
# (``Alibaba-NLP/gte-modernbert-base``) follows the E5 / GTE-modernbert
# convention of prefixing query texts with ``"query: "``; documents
# (the corpus rows) take no prefix. The asymmetry matters: a downstream
# caller that accidentally routes queries through the document path
# (or vice-versa) silently degrades retrieval quality with no error.
# Centralising the constant here lets the stub mirror the same
# behaviour, so a unit test that compares ``embed(text)`` against
# ``embed_batch([text])[0]`` detects the regression.
QUERY_PREFIX: Final[str] = "query: "

# Plan §6 T2: stub embedder dim. Small enough to keep tests fast;
# large enough that the cosine-distance test has meaningful signal.
DEFAULT_STUB_DIM: Final[int] = 8


# ---------------------------------------------------------------------
# Stub implementation — offline by default (AC-4)
# ---------------------------------------------------------------------


def _stub_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic SHA-256-derived L2-normalised float32 vector.

    Used by :class:`StubEmbedder` as a hash-then-normalise content
    embedding. Two distinct strings produce vectors that are
    near-orthogonal in expectation (because SHA-256 output bytes are
    indistinguishable from uniform random); identical strings produce
    bit-identical vectors. Both properties are load-bearing for the
    StubEmbedder's "deterministic but distinguishable" contract.

    The vector is built by repeatedly hashing the digest until enough
    bytes accumulate to fill ``dim`` float32 entries. ``-128.0`` shifts
    the unsigned bytes into a centred range so the L2-normalised
    output is not biased toward the all-positive quadrant.
    """
    needed_bytes = dim * 4  # float32 stride
    accumulator = bytearray()
    seed = text.encode("utf-8")
    while len(accumulator) < needed_bytes:
        seed = hashlib.sha256(seed).digest()
        accumulator.extend(seed)
    raw = np.frombuffer(bytes(accumulator[:needed_bytes]), dtype=np.uint8)
    # Centre the unsigned-byte range (~[0, 255]) around zero so the
    # normalised vector spans both halves of the unit sphere.
    centred = raw.astype(np.float32) - 128.0
    # Reshape from a 1-D byte stream to a 1-D float vector by viewing
    # as 4-byte groups -- but we already broadcast to float32 above
    # via the byte-by-byte centring, so just take ``dim`` floats from
    # what we have. The astype is on a uint8-byte buffer; truncate to
    # ``dim`` entries.
    floats = centred[:dim]
    norm = np.linalg.norm(floats)
    if norm == 0:  # pragma: no cover — astronomically unlikely from SHA-256
        return floats
    return (floats / norm).astype(np.float32)


class StubEmbedder:
    """Offline deterministic embedder backed by SHA-256.

    AC-4 (intent line 33): *"the embedding model has no external API
    dependency — it runs locally"*. The stub takes the discipline one
    step further: it has no model dependency at all. SHA-256 provides
    the determinism (same text → same vector across processes /
    Python versions / OS); ``dim`` is small enough that the unit
    tests run in milliseconds.

    Plan §6 T2: the stub mirrors the live :class:`Embedder` contract
    exactly — :meth:`embed` prepends :data:`QUERY_PREFIX`,
    :meth:`embed_batch` does not. A unit test asserting
    ``embed(text) != embed_batch([text])[0]`` catches a downstream
    regression where the prefix asymmetry is dropped.

    The class satisfies :class:`bristol_ml.embeddings.Embedder`
    structurally via :func:`runtime_checkable`; no inheritance.
    Construction is constant-time; the stub is thread-safe (no
    shared state).
    """

    def __init__(self, *, dim: int = DEFAULT_STUB_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"StubEmbedder dim must be positive; got {dim}.")
        self._dim = dim
        self._model_id = f"stub-sha256-{dim}"
        logger.debug("StubEmbedder constructed (dim={}, model_id={})", dim, self._model_id)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, text: str) -> np.ndarray:
        return _stub_vector(QUERY_PREFIX + text, self._dim)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # Documents take no prefix; preserve order.
        return np.stack([_stub_vector(t, self._dim) for t in texts]).astype(np.float32)


# ---------------------------------------------------------------------
# Live implementation — sentence-transformers + HF_HUB_OFFLINE
# ---------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Live :class:`Embedder` backed by ``sentence-transformers``.

    Plan §1 D4 / A2 binding: the default checkpoint is
    ``Alibaba-NLP/gte-modernbert-base`` (149 M params, 768-dim,
    MTEB-56 avg 64.38, Apache 2.0, no ``trust_remote_code``,
    max_seq=8192). ``model_kwargs={"torch_dtype": torch.float16}``
    halves the inference RAM footprint to ~149 MB.

    Plan §1 D12 / NFR-1: ``HF_HUB_OFFLINE=1`` is set in
    ``tests/conftest.py`` to assert no network I/O escapes during the
    unit / integration suite. If the model is not in the local
    HuggingFace cache, the underlying ``SentenceTransformer(...)``
    call raises ``OSError`` (or similar) and this constructor
    re-raises with a message naming the pre-warm command so an
    operator can recover without spelunking through the
    ``sentence-transformers`` source.

    Prefix asymmetry (plan §6 T4): :meth:`embed` prepends
    :data:`QUERY_PREFIX` (``"query: "``) — the gte-modernbert / E5
    convention; :meth:`embed_batch` does not. The stub mirrors this
    so a regression in either direction is caught by a single unit
    test.

    Construction is *not* cheap (loads the model into CPU RAM); a
    Stage 16 caller should construct once and reuse the instance.
    Thread-safe after construction (the underlying model is read-only
    at inference time).
    """

    def __init__(self, *, model_id: str, fp16: bool = True) -> None:
        # Defer the import so a stub-only test environment (or a CI
        # run without sentence-transformers in the lock) still imports
        # this module. Mirrors the OpenAI-SDK lazy-import discipline
        # in the Stage 14 ``LlmExtractor``.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover — CI installs the dep
            raise RuntimeError(
                "sentence-transformers is not installed. Add it via "
                "`uv sync --group dev` or set "
                "BRISTOL_ML_EMBEDDING_STUB=1 to use the offline stub "
                "(plan §1 D8 — triple-gated for CI safety)."
            ) from exc

        # ``fp16=True`` halves the model's RAM footprint; the live
        # default at A2 is 298 MB at fp32 → ~149 MB at fp16. The
        # cache vector matrix stays float32 (set in :meth:`embed_batch`
        # below) for query stability.
        model_kwargs: dict[str, object] | None = None
        if fp16:
            try:
                import torch

                model_kwargs = {"torch_dtype": torch.float16}
            except ImportError:  # pragma: no cover — torch is a runtime dep
                model_kwargs = None
                logger.warning(
                    "torch not importable; falling back to fp32 model load. "
                    "RAM footprint will be ~2x the fp16 path."
                )

        try:
            self._model = SentenceTransformer(
                model_id,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:  # pragma: no cover — exercised in T4 e2e
            raise RuntimeError(
                f"Failed to load sentence-transformer model {model_id!r}. "
                "If this is the first run, pre-warm the local HF cache with: "
                f'\n  python -c "from sentence_transformers import '
                f'SentenceTransformer; SentenceTransformer({model_id!r})"'
                "\nThen re-run with HF_HUB_OFFLINE=1. Underlying error: "
                f"{type(exc).__name__}."
            ) from exc

        self._model_id = model_id
        # The model exposes its embedding dim via
        # ``get_sentence_embedding_dimension()``; we cache it once.
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover — every published ST model returns int
            raise RuntimeError(
                f"sentence-transformers model {model_id!r} did not expose a "
                "sentence-embedding dimension. Check the model card."
            )
        self._dim = int(dim)
        logger.info(
            "SentenceTransformerEmbedder loaded (model_id={}, dim={}, fp16={})",
            model_id,
            self._dim,
            fp16,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, text: str) -> np.ndarray:
        # Query path — gte-modernbert prefix.
        return self._encode_one(QUERY_PREFIX + text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        # Document path — no prefix.
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # ``normalize_embeddings=True`` makes cosine similarity a
        # plain matmul in the index; ``convert_to_numpy=True`` returns
        # a numpy array directly so we can pin the dtype downstream.
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)
