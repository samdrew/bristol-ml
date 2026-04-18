# Stage 15 — Embedding index for REMIT

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 13 (and Stage 14 conceptually, though not mechanically)
**Enables:** richer exploratory tools for REMIT; a precursor to any RAG-style extension

## Purpose

Add a semantic index over the REMIT event corpus, so that a facilitator can ask "show me outages semantically similar to this one" and get an answer that keyword search would not produce. The index itself is architectural exercise: it introduces embeddings, a vector store, and nearest-neighbour retrieval — all components a richer LLM-driven pipeline would depend on. This stage keeps it simple and local; no API calls.

## Scope

In scope:
- A module that produces embeddings for REMIT event descriptions using a local sentence-transformer model.
- A vector index covering the embedded corpus, supporting nearest-neighbour search.
- A notebook that demonstrates the index by taking a sample event, finding its nearest neighbours, and visualising the cluster it sits in (a 2D projection is enough for demo purposes).
- Caching of embeddings so the index does not have to be rebuilt from scratch on every run.

Out of scope:
- Retrieval-augmented generation using this index (a separate richer stage, not in the current plan).
- Online embedding as messages arrive (the index is rebuilt in bulk for now).
- Hybrid keyword-plus-semantic search.

## Demo moment

A facilitator picks a REMIT event — say, a planned nuclear outage — and the notebook returns its ten nearest neighbours. The neighbours are clearly semantically related (other nuclear outages, other planned events of similar durations), not just keyword-matched. The 2D projection shows the corpus falling into visible clusters by event type and fuel.

## Acceptance criteria

1. Embedding the corpus runs end-to-end on a laptop CPU in a reasonable time.
2. Nearest-neighbour queries return in well under a second.
3. The embedding cache means re-running the notebook is fast.
4. The embedding model has no external API dependency — it runs locally.
5. The index's interface is small enough that swapping the vector store implementation later (for example, FAISS for a toy numpy store, or vice-versa) is a mechanical change.

## Points for consideration

- Choice of embedding model. A general-purpose sentence transformer is the default; a domain-tuned model would probably be better but is out of scope to train. The general model is unlikely to distinguish fuel types as well as a human reader would.
- Vector store. For a corpus of a few thousand events, a plain numpy array with cosine similarity is enough. For a larger corpus, FAISS or a similar structure scales better. Starting simple and growing is defensible.
- Cache invalidation. The embedding cache becomes stale if the REMIT corpus changes or the embedding model changes. A content hash of both makes invalidation automatic; a manual rebuild command is simpler.
- 2D projection for visualisation — UMAP, t-SNE, PCA. Each has trade-offs; the purpose here is illustrative, so any of them is fine.
- Whether the index is worth rebuilding if the REMIT ingestion updates. For a mostly-static corpus, the answer is "occasionally." For a live-feed scenario, the answer becomes "incrementally." The current project is in the former regime.
- How neighbours are scored and displayed. Cosine similarity is the usual default; the notebook should print the similarity alongside each neighbour so a reader can judge.
- Interaction with Stage 14's extracted structured features. Semantic search over free text complements structured filters but does not replace them; a notebook that combines both is more powerful than either alone.
- Whether to expose the index as a CLI command in addition to the notebook. Useful for meetups where a facilitator wants to ask ad-hoc questions without opening Jupyter.

## Dependencies

Upstream: Stage 13 (REMIT corpus).

Downstream: any future retrieval-augmented extension would build on this index; not in the current plan.

## Out of scope, explicitly deferred

- Retrieval-augmented generation.
- Fine-tuned embeddings.
- Incremental index maintenance under a live feed.
- Hybrid keyword-plus-semantic search.
