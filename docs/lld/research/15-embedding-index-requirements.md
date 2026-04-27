# Stage 15 — Embedding index for REMIT: requirements

**Source intent:** `docs/intent/15-embedding-index.md`
**Artefact role:** Phase 1 research deliverable (requirements analyst).
**Audience:** plan author (lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## 1. Goal

Add a local, offline semantic index over the REMIT event corpus — using a sentence-transformer model and a simple vector store — so that a meetup facilitator can retrieve the ten nearest neighbours of any chosen event and visualise where it sits in the embedded corpus, with no API calls and no model re-embedding on repeated runs.

---

## 2. User stories

**US-1 — Meetup facilitator (demo path).**
Given the REMIT corpus has been ingested (Stage 13) and the embedding index has been built,
when the facilitator picks a sample event (e.g. a planned nuclear outage) in the notebook,
then the notebook returns its ten nearest neighbours with cosine-similarity scores alongside each, and renders a 2D projection showing the event's position in the embedded corpus — all without a network call or a long wait.
(Intent lines 26–27; AC-1, AC-2, AC-3.)

**US-2 — Attendee re-running the notebook (warm-cache path).**
Given the embedding cache is already populated from a prior run,
when the attendee executes the notebook top-to-bottom,
then no embeddings are recomputed — the cache is read instead — and the notebook completes in well under a minute.
(Intent lines 17, 43; AC-3.)

**US-3 — CI / offline clone (no-download path).**
Given the sentence-transformer model weights are present in the local model cache (pre-warmed) and the REMIT parquet fixture is present,
when CI runs the module's smoke test,
then the embedder initialises, produces a small batch of embeddings, and the nearest-neighbour query returns a typed result — without any network I/O.
(DESIGN.md §2.1.3; AC-4.)

**US-4 — Future implementer swapping the vector backend (interface-stability path).**
Given the vector store is accessed only through the module's narrow interface,
when the implementer replaces the numpy cosine-similarity store with FAISS (or vice-versa),
then no calling code outside `bristol_ml/llm/embeddings.py` requires change.
(Intent line 34; AC-5.)

**US-5 — Stage 16 / future RAG stage (downstream consumer).**
Given Stage 15 exposes a typed query interface,
when a downstream stage passes an event description and requests k neighbours,
then it receives a typed list of (event_id, similarity_score) pairs without depending on any vector-store internals.
(Intent line 5; DESIGN.md §2.1.2.)

---

## 3. Acceptance criteria

The following five are restated verbatim from the intent (lines 31–34):

**AC-1.** Embedding the corpus runs end-to-end on a laptop CPU in a reasonable time.

**AC-2.** Nearest-neighbour queries return in well under a second.

**AC-3.** The embedding cache means re-running the notebook is fast.

**AC-4.** The embedding model has no external API dependency — it runs locally.

**AC-5.** The index's interface is small enough that swapping the vector store implementation later (for example, FAISS for a toy numpy store, or vice-versa) is a mechanical change.

**Implicit AC-6 — Similarity scores are displayed alongside neighbours.**
The notebook prints cosine-similarity scores next to each returned neighbour so a reader can judge result quality (intent line 43: "the notebook should print the similarity alongside each neighbour").

**Implicit AC-7 — 2D projection renders visible clusters.**
The notebook includes a 2D projection of the embedded corpus that visually separates events by type or fuel — sufficient for the demo moment described at intent lines 26–27.

**Implicit AC-8 — Cache invalidation is detectable.**
There is a documented mechanism (manual rebuild command or automatic content-hash check) to detect and rebuild a stale cache when the corpus or model changes (intent lines 40–41).

**Implicit AC-9 — Module is runnable standalone.**
`python -m bristol_ml.llm.embeddings` (or an equivalent entry point) executes without error, prints the active configuration, and emits a sample query result (DESIGN.md §2.1.1).

**Implicit AC-10 — At least one test exercises the public interface.**
The public interface has at minimum one smoke test that builds a small index from fixture events and asserts a nearest-neighbour query returns a typed, non-empty result (DESIGN.md §2.1.7).

---

## 4. Non-functional requirements

**NFR-1 — Offline by default (DESIGN.md §2.1.3; AC-4).**
The sentence-transformer model must be loaded from a local path (huggingface `cache_dir` or a committed fixture for small test models). The embedder must not initiate any HTTP call during initialisation or inference. This is enforced by at least one test that asserts no network I/O occurs when the pre-cached model path is set (pattern from Stage 14 NFR-1).

**NFR-2 — Idempotent cache / cache invalidation (DESIGN.md §2.1.5; intent lines 40–41).**
Re-running the embedding step when a valid cache exists must be a no-op: the cache is read and the corpus is not re-embedded. If the corpus changes or the model identifier changes, the stale cache must be detectable. The chosen invalidation strategy (content-hash or manual rebuild flag) must be stated in the module's `CLAUDE.md` and in the YAML config. Silent use of a stale cache is not acceptable.

**NFR-3 — Configuration in YAML (DESIGN.md §2.1.4; intent line 14).**
Embedding model name/path, cache directory, vector store backend selector, number of neighbours returned by default, and 2D projection technique are all YAML fields under `conf/llm/embeddings.yaml`. No values are hard-coded in the module.

**NFR-4 — Narrow, swap-safe vector store interface (AC-5; DESIGN.md §2.1.2).**
The vector store is accessed through a runtime-checkable `Protocol` (or abstract base class) with at most three methods: `add(ids, vectors)`, `query(vector, k) -> list[tuple[id, score]]`, and `save(path) / load(path)`. The numpy and FAISS implementations each satisfy this protocol. Callers hold the protocol type, not the concrete type.

**NFR-5 — Provenance (DESIGN.md §2.1.6).**
The embedding cache artefact records: the embedding model identifier, the corpus parquet path or content hash, the git SHA at build time, and the build timestamp. These fields must be queryable without re-loading the full index.

**NFR-6 — CPU-affordable runtime (AC-1; DESIGN.md §1.1 quality bar).**
Embedding a corpus of a few thousand events on a mid-range laptop CPU must complete in a time that does not require the facilitator to leave the stage. This is not a hard number in the intent, but "reasonable" in context of a meetup demo means below ~5 minutes for a first run and under 10 seconds for a cached run. The implementer must benchmark on the target corpus size and record the result in the stage LLD.

**NFR-7 — Notebooks are thin (DESIGN.md §2.1.8).**
The demonstration notebook imports from `bristol_ml.llm.embeddings`; it does not reimplement embedding, similarity, or projection logic. Every substantive computation lives in the module. The notebook executes top-to-bottom without errors on a machine with a warm model cache and a warm embedding cache.

**NFR-8 — Observability (DESIGN.md §2.1 cross-cutting; loguru convention from Stage 14).**
The embedder emits structured `loguru` log lines at INFO for: cache hit (with event count), cache miss (triggering rebuild, with corpus size and model name), and query latency. It emits at WARNING if the cache is detected stale. DEBUG level logs individual embedding calls if batch size is 1.

---

## 5. Open questions

**OQ-1 — Which sentence-transformer model?**
The intent says "a general-purpose sentence transformer is the default" (line 38) but does not name one. Options:
- `all-MiniLM-L6-v2` (~80 MB, fast on CPU, widely used as a default): lowest friction, weights pre-cached easily, short context window (256 tokens) sufficient for REMIT descriptions.
- `all-mpnet-base-v2` (~420 MB, higher quality): better semantic accuracy at ~5x the size and ~3x the CPU latency.
- `paraphrase-multilingual-MiniLM-L12-v2`: only relevant if non-English REMIT messages exist; they do not.
- A domain-fine-tuned energy model: out of scope per intent line 38.

Recommended default: `all-MiniLM-L6-v2`. It fits in under 100 MB, runs a 3,000-event corpus in a few minutes on a laptop CPU, and is the canonical benchmark model for sentence-transformers demos. Record the choice as a config value so switching costs only a YAML edit.

**OQ-2 — Vector store backend: numpy cosine similarity vs FAISS?**
The intent says "for a corpus of a few thousand events, a plain numpy array with cosine similarity is enough" (line 39) and names FAISS as the scale option. Options:
- numpy: no extra dependency, transparent code, sufficient for <50,000 events, O(n) query but n is small.
- FAISS: fast approximate nearest-neighbour, but an extra C++ dependency and overkill for a demo corpus.

Recommended default: numpy store behind the Protocol interface (OQ-4 below). FAISS is a config-selectable second implementation. This satisfies AC-5 and avoids a heavy optional dependency in CI.

**OQ-3 — Cache invalidation strategy: content-hash vs manual rebuild?**
The intent presents both (line 40–41). Options:
- Content-hash: hash the corpus parquet file and the model identifier; compare to stored hash in cache metadata; auto-rebuild on mismatch. Automatic and correct, slightly more complex.
- Manual rebuild flag: `--rebuild` CLI argument or a config field `force_rebuild: true`; simpler but requires the user to remember to do it.

Recommended default: content-hash of both corpus file and model identifier, stored in the cache provenance record (NFR-5). The hash computation adds negligible overhead compared to re-embedding. A `--rebuild` flag is also exposed as an escape hatch.

**OQ-4 — 2D projection technique: UMAP, t-SNE, or PCA?**
The intent says "any of them is fine" (line 41) for illustrative purposes. Options:
- PCA: zero extra dependency (sklearn), deterministic, fast; clusters are less visually separated.
- t-SNE: sklearn, non-deterministic by default (fixable with `random_state`), good cluster separation, slow on >10,000 points.
- UMAP: best visual cluster separation, but an extra dependency (`umap-learn`), non-deterministic without seed.

Recommended default: t-SNE with `random_state=42`, using sklearn (already likely in the dependency tree). This is deterministic, produces a compelling demo plot, and adds no new dependencies. Projection technique is a config field so a facilitator can switch at demo time.

**OQ-5 — CLI surface: yes or no?**
The intent says "useful for meetups where a facilitator wants to ask ad-hoc questions without opening Jupyter" (line 45) but does not mandate it. Options:
- No CLI: notebook only. Simpler, stays within the demo surface.
- Minimal CLI: `python -m bristol_ml.llm.embeddings query --event-id <ID> --k 10`. Satisfies NFR-8 (standalone module) and enables ad-hoc demos without Jupyter.

Recommended default: minimal CLI, limited to two subcommands — `build` (build/rebuild the index) and `query` (nearest-neighbour lookup by event ID or free text). This is consistent with the standalone-module principle (DESIGN.md §2.1.1) and is low implementation cost.

**OQ-6 — Interaction with Stage 14 structured features?**
The intent notes that "a notebook that combines both is more powerful than either alone" (line 44). Options:
- Stage 15 notebook is purely semantic (no Stage 14 dependency): simpler, separates concerns, avoids a hard dependency on Stage 14 being complete.
- Stage 15 notebook combines semantic similarity with Stage 14 structured filters (e.g. "nearest neighbours of this event, filtered to the same fuel type"): more powerful demo but couples the notebook to Stage 14's schema.

Recommended default: Stage 15 has no hard mechanical dependency on Stage 14 (consistent with intent line 4: "Stage 14 conceptually, though not mechanically"). The notebook may include an optional cell that joins structured features when available, but this cell must be clearly labelled optional and must not break top-to-bottom execution when Stage 14 output is absent.

**OQ-7 — What text field is embedded?**
The intent says "embeddings for REMIT event descriptions" (intent line 14) but Stage 13 stores both structured fields and a free-text description. Options:
- Free-text description only: purest semantic signal; what Stage 14 also reads.
- Concatenated structured fields + description: richer input but conflates structured and textual signals; structured fields can dominate.
- Description + asset name: a pragmatic middle ground.

Recommended default: the free-text description field only, consistent with Stage 14's scope and the intent's phrase "REMIT event descriptions." The field name must match the Stage 13 parquet schema; confirm before implementation.

---

## 6. Explicit non-goals

Restated verbatim from intent lines 55–59:

- Retrieval-augmented generation.
- Fine-tuned embeddings.
- Incremental index maintenance under a live feed.
- Hybrid keyword-plus-semantic search.

Additionally, from intent lines 19–22:

- Online embedding as messages arrive (the index is rebuilt in bulk for now).

---

## 7. Risks

**R-1 — Model weight download fails or is slow at first run.**
`sentence-transformers` downloads model weights from Hugging Face Hub on first use (~80 MB for `all-MiniLM-L6-v2`). At a meetup with unreliable Wi-Fi this will stall the demo. Mitigation: document a pre-warm step (`python -m bristol_ml.llm.embeddings build --dry-run` or equivalent) that downloads weights without processing data; include it in the "before you come" attendee notes. CI must pre-cache the model or use a tiny fixture model that requires no download.

**R-2 — Corpus size vs available RAM.**
A full REMIT archive (potentially tens of thousands of events) stored as float32 embeddings at 384 dimensions occupies roughly 384 × 4 × N bytes. At N = 50,000 that is ~75 MB — fine. At N = 500,000 it is ~750 MB, which can exhaust a constrained laptop. Mitigation: document the tested corpus size in the stage LLD; add a config guard `max_corpus_size` that truncates with a warning rather than OOM-crashing.

**R-3 — Non-deterministic similarity scores across reruns.**
If the embedding model or projection uses randomness without a seed, two runs over the same corpus will produce different nearest-neighbour rankings or a different-looking 2D plot. This undermines demo reproducibility. Mitigation: sentence-transformer inference is deterministic given the same model and hardware. The projection technique (OQ-4) must use a fixed `random_state`. Record this as a CI assertion: same fixture, same top-1 neighbour.

**R-4 — Stage 13 parquet schema drift.**
Stage 15 depends on a specific free-text field from Stage 13's output (OQ-7). If Stage 13 renames that field, the embedder silently produces empty strings or raises a KeyError. Mitigation: an integration test reads the Stage 13 fixture parquet and asserts the expected field is present before passing it to the embedder (same pattern as Stage 14 R-4).

**R-5 — Cache grows stale silently after a corpus update.**
If the Stage 13 ingestion adds new events and the content-hash invalidation is not implemented or is bypassed, the index will not include the new events. The demo will then return neighbours that exclude recent outages without any warning. Mitigation: NFR-2 requires the stale-cache detection to emit a WARNING log; the notebook should surface this warning in a prominent cell output.

**R-6 — Projection reveals no visible clusters.**
If the REMIT corpus is too small or too homogeneous, the 2D projection may show an undifferentiated blob — undermining the demo moment. Mitigation: test the projection on the actual corpus before the stage ships; if clusters are not visible, try a different projection technique (see OQ-4) or increase perplexity (t-SNE) / n_neighbours (UMAP). Document the chosen parameters in the notebook.

---

*This artefact is one of the Phase-1 research inputs for Stage 15. It covers requirements only. Codebase patterns, domain research, and scope-diff analysis are in companion artefacts.*

---

**Relevant file paths:**
- `/workspace/docs/intent/15-embedding-index.md` — primary intent document
- `/workspace/docs/intent/DESIGN.md` — project spec (§2.1.1–2.1.8, §3.2, §8, §9)
- `/workspace/docs/intent/13-remit-ingestion.md` — upstream corpus dependency
- `/workspace/docs/intent/14-llm-extractor.md` — sibling stage; stub-first pattern and Protocol shape
- `/workspace/docs/lld/research/14-llm-extractor-requirements.md` — structural template followed above
