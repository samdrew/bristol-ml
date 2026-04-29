# ADR 0008 — Embedder + VectorIndex are runtime-checkable Protocols, not ABCs

- **Status:** Accepted — 2026-04-27.
- **Deciders:** Project author; Stage 15 lead agent.
- **Related:** [`layers/embeddings.md`](../layers/embeddings.md), [`src/bristol_ml/embeddings/_protocols.py`](../../../src/bristol_ml/embeddings/_protocols.py), [`docs/plans/completed/15-embedding-index.md`](../../plans/completed/15-embedding-index.md) §1 D2 + D3 + D16, ADR-0003 (`Model` Protocol precedent).

## Context

Stage 15 introduces a new `bristol_ml/embeddings/` layer with two
boundary types that downstream stages will consume:

- `Embedder` — turns text into a vector (`embed(str) -> np.ndarray`,
  `embed_batch(list[str]) -> np.ndarray`).
- `VectorIndex` — stores (id, vector) pairs and serves nearest-
  neighbour queries (`add(ids, vectors)`, `query(q, k) -> list[(id,
  score)]`, `save(path)`, `load(path)`).

The layer ships **two implementations of each at Stage 15** — `StubEmbedder` and `SentenceTransformerEmbedder` for `Embedder`; `StubIndex` and `NumpyIndex` for `VectorIndex`. The stub-paths are the offline-by-default discipline (intent AC-4 / NFR-1, plan §1 D8); the live paths are the demo-quality production paths backed by `Alibaba-NLP/gte-modernbert-base` (plan §1 D4 / A2) and a pre-normalised numpy matrix.

A future RAG stage may swap `NumpyIndex` for FAISS / Qdrant / a remote vector DB without touching the embedder; a future quantised-model experiment may swap the embedder without touching the index. The boundary types are the load-bearing contract that makes those swaps **mechanical** (intent AC-5, plan NFR-4).

Three project constraints narrow the choice of how to express the boundary in code:

1. **DESIGN §2.1.2 (typed narrow interfaces)** — the boundary types are imported by Stage 16 and any future RAG stage. They must be importable without dragging the heavy live backend (`sentence-transformers`, `transformers`, `torch.float16`) into the consumer's import graph.
2. **DESIGN §2.1.3 (stub-first for expensive / flaky external dependencies)** — the stub implementations must satisfy the *same* boundary as the live ones. A unit test that depends only on the boundary must run without `sentence-transformers` installed; the test asserts conformance via `isinstance(_, Embedder)`.
3. **ADR-0003 precedent** — the `Model` Protocol uses `typing.Protocol` + `runtime_checkable` for exactly this swap-safety property. Stage 14's `Extractor` Protocol followed suit. A third Protocol layer would establish the pattern as the project's house style for swappable interfaces.

The decision shapes the boundary's import cost (cheap or heavy?), the test contract (`isinstance` or `issubclass`?), and the rule a future implementation author follows when deciding whether their new vector store satisfies the contract.

## Decision

`Embedder` and `VectorIndex` are **`@runtime_checkable` `typing.Protocol` types** declared in `src/bristol_ml/embeddings/_protocols.py`. Concrete implementations satisfy them **structurally — no inheritance, no `abc.ABC`**.

The Protocols are re-exported from `bristol_ml.embeddings` (the package `__init__`); concrete implementations live in private modules (`_embedder.py`, `_index.py`) and are imported only via the factory (`build_embedder`, `build_index`) or the standalone CLI. A consumer holding only the Protocol type pays no import cost beyond `numpy` + `pydantic`; in particular, **`sentence-transformers` does not enter the import graph** unless the consumer actually constructs the live backend.

```python
# src/bristol_ml/embeddings/_protocols.py
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class VectorIndex(Protocol):
    @property
    def dim(self) -> int: ...
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, k: int) -> list[NearestNeighbour]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "VectorIndex": ...
```

Concretely:

- **Two methods on `Embedder`**, **four on `VectorIndex`** — pinned by `tests/unit/embeddings/test_protocol.py::test_embedder_protocol_has_two_methods` and `test_vector_index_protocol_has_four_methods`. Adding a method to either Protocol is a plan-edit conversation (NFR-4).
- **`runtime_checkable`** lets the unit tests assert `isinstance(StubEmbedder(), Embedder)` and `isinstance(NumpyIndex(dim=8), VectorIndex)` without any production code inheriting from the Protocol. A third implementation (e.g. `FaissIndex`, `BedrockEmbedder`) becomes a *new file* — no edits to the Protocol, no edits to existing implementations.
- **Stage 16 imports `Embedder` and `VectorIndex` only.** The factory functions are how anyone *constructs* an instance; everywhere else the Protocol is the type that crosses module boundaries.
- **Empty-index discipline (plan §1 D16).** `query` on an empty index returns `[]`, never raises. Mirrors the Stage 14 `StubExtractor`'s miss-path discipline and is part of the contract a future implementation must honour.

## Consequences

- **Layer doc edits.** [`layers/embeddings.md`](../layers/embeddings.md) §"Public interface" cites this ADR and lists the Protocol method-counts as the contract surface. The Stage 16 plan's Reading-order section will name the layer doc; the layer doc points here for the rationale.
- **Stage 16 / future RAG stage are unaffected by backend swaps.** They depend on the Protocol type and on `bristol_ml.embeddings.build_embedder` / `build_index`. A swap from `NumpyIndex` to a hypothetical `FaissIndex` is one new file plus one new dispatch branch in `build_index`; no Stage 16 / RAG code changes.
- **`sentence-transformers` stays out of the cold-import path.** The boundary import asserts this in `tests/unit/embeddings/test_protocol.py::test_boundary_import_does_not_pull_sentence_transformers`. A future regression that pulls `sentence_transformers` into `bristol_ml/embeddings/__init__.py` is caught by that test before it reaches main.
- **The pattern is now established.** Three swappable interfaces (`Model`, `Extractor`, `Embedder`/`VectorIndex`) all use `runtime_checkable` Protocol. A fourth swappable interface arriving in a later stage follows the same shape; a deviation is a discussion-worthy event, not a quiet alternative.
- **No retroactive change to ADR-0003.** ADR-0003 covers `Model`. This ADR extends the same rationale to a new layer; ADR-0003 is unaltered.

## Alternatives considered

- **Use `abc.ABC` instead of `Protocol`.** Rejected. Inheritance forces every concrete implementation to import the abstract base — a future implementation written in a separate package would either inherit transitively (coupling its dependency graph to ours) or reimplement the abstract class header (re-introducing the swap-fragility we are trying to avoid). `Protocol` makes "satisfies the contract" structural — a concrete class satisfies the Protocol by exposing the right method names and signatures, with no import of the Protocol module required at definition time.
- **Combine `Embedder` and `VectorIndex` into a single `EmbeddingStore` interface.** Rejected. The two have distinct lifecycles: an `Embedder` is constructed once per process (heavy, model-loaded, thread-safe-after-construction); a `VectorIndex` is constructed per corpus build (light, mutable, populated via `add`). Combining them would pull `model_id` / `embed` / `embed_batch` onto the same surface as `add` / `query` / `save` / `load`, doubling the swap cost — replacing the index also forces replacing the embedder.
- **Skip Protocols, use duck typing with type aliases.** Rejected. `EmbedderLike = Any` would technically work but provides zero static-analysis surface and zero runtime conformance check. The unit tests assert `isinstance(_, Embedder)` because a regression that drops `embed_batch` would otherwise surface only in an integration test (or worse, in production). Static + runtime conformance is the lesson the `Model` Protocol taught at Stage 4 — re-deriving it here would be a regression.
- **Wrap in a single `bristol_ml.boundaries` module.** Rejected. Stage 14's `Extractor` Protocol lives next to `bristol_ml.llm`'s implementations; Stage 4's `Model` Protocol lives next to `bristol_ml.models`. Co-locating the Protocol with the layer that owns it keeps the discovery story consistent — a Stage 16 author looking for the embedding contract finds it in `bristol_ml.embeddings`, not in a sibling-of-everything `boundaries` module that mixes contracts from unrelated layers.

## Supersession

If a future stage requires the boundary to grow a method (e.g. `embed_async` for an async serving path, or `query_with_metadata` for a richer return shape), this ADR should be superseded by a new one recording the expanded surface and the migration path for existing implementations. The two-method / four-method shape is correct for the Stage 15 demo + Stage 16 join + future-RAG triple; any growth is a real architectural decision worth its own ADR.

If a third swappable interface arrives without using `runtime_checkable` Protocol (e.g. an interface that has no obvious structural surface, or one whose conformance must be verified via behavioural tests rather than method-presence), this ADR's "house style" claim should be re-examined. The current decision is correct for interfaces whose conformance is a method-shape question, not a behavioural-property question.

## References

- [`docs/plans/completed/15-embedding-index.md`](../../plans/completed/15-embedding-index.md) §1 D2 (Embedder Protocol shape), §1 D3 (VectorIndex Protocol shape), §1 D16 (empty-index returns `[]`).
- [`docs/lld/research/15-embedding-index-scope-diff.md`](../../lld/research/15-embedding-index-scope-diff.md) — `@minimalist` Phase-1 critique; the Protocol method-counts were a `RESTATES INTENT` row (kept).
- [`docs/architecture/layers/embeddings.md`](../layers/embeddings.md) §"Public interface" + §"Why two Protocols, not one".
- [`src/bristol_ml/embeddings/_protocols.py`](../../../src/bristol_ml/embeddings/_protocols.py) — implementation.
- ADR-0003 (`decisions/0003-protocol-for-model-interface.md`) — the precedent for `runtime_checkable` Protocol over ABC.
- DESIGN §2.1.2 (typed narrow interfaces), §2.1.3 (stub-first), §2.1.7 (tests at boundaries) — the principles this decision honours.
