# Plan - Stage 15: Embedding index over REMIT

**Status:** `approved` - Ctrl+G 2026-04-27 resolved A1, A3, A4, A5. A2 bound 2026-04-27 to `Alibaba-NLP/gte-modernbert-base` after the SOTA-verification follow-up returned. Phase 2 implementation unblocked.

**Ctrl+G dispositions (2026-04-27):**
- **A1 (D1) — bound:** new sibling `bristol_ml/embeddings/` layer + ADR-0008.
- **A2 (D4) — bound:** default checkpoint `Alibaba-NLP/gte-modernbert-base` (149 M params, 768-dim, MTEB-56 avg **64.38**, +2.21 over BGE-small; ~298 MB safetensors fp32 / ~149 MB at fp16; Apache 2.0; no `trust_remote_code`; max_seq=8192). The 2022-2023 alternatives surveyed in the original domain artefact are superseded by ModernBERT-architecture (Dec 2024) checkpoints. Evidence: [`docs/lld/research/15-embedding-index-domain-update.md`](../../lld/research/15-embedding-index-domain-update.md).
- **A3 (D6) — flipped:** UMAP (`umap-learn`, `random_state=42`) replaces PCA. The user's argument: PCA's two principal components do not preserve neighbourhood structure in high-dim embedding space, which undermines the demo moment. Dependency cost (~200 MB numba/llvmlite) accepted.
- **A4 (D10) — bound:** standalone-module entry only (`python -m bristol_ml.embeddings`); no `build`/`query` subcommands.
- **A5 (D11) — flipped:** include the optional Stage 14 cross-stage cell. Notebook gains one cell that joins Stage 14 structured features when present, skipping cleanly with a printed banner when absent.
**Intent:** [`docs/intent/15-embedding-index.md`](../../intent/15-embedding-index.md)
**Upstream stages shipped:** Stages 0-14 (foundation -> ingestion -> features -> six model families -> enhanced evaluation -> registry -> MLP -> TCN -> serving -> REMIT bi-temporal store -> LLM feature extractor).
**Downstream consumers:** future RAG-style stage (out of scope here); Stage 16 (model with REMIT) may optionally consume nearest-neighbour features.
**Baseline SHA:** `74c8d03` (tip of `main` after Stage 14 merge, PR #15).

**Discovery artefacts produced in Phase 1:**

- Requirements - [`docs/lld/research/15-embedding-index-requirements.md`](../../lld/research/15-embedding-index-requirements.md)
- Codebase map - [`docs/lld/research/15-embedding-index-codebase.md`](../../lld/research/15-embedding-index-codebase.md)
- Domain research - [`docs/lld/research/15-embedding-index-domain.md`](../../lld/research/15-embedding-index-domain.md)
- Domain research (SOTA-verification follow-up, A2) - [`docs/lld/research/15-embedding-index-domain-update.md`](../../lld/research/15-embedding-index-domain-update.md)
- Scope Diff - [`docs/lld/research/15-embedding-index-scope-diff.md`](../../lld/research/15-embedding-index-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §"Demo moment" names a single moment: a facilitator picks a planned nuclear outage, the notebook returns ten semantically-similar neighbours with cosine scores, and a 2D projection shows visible clusters. Everything else - the Protocol, the cache-hash invalidation, the stub-first triple gate - is plumbing in service of that moment, plus the contract a future RAG stage will consume.

**Architectural weight.** This is the project's first content-addressed cache. The lessons it teaches - corpus-hash + model-id as the invalidation key, Parquet `custom_metadata` as the provenance carrier, a deterministic recompute on mismatch - generalise beyond embeddings to any cached derivation downstream of REMIT (or any other ingested corpus). The `Embedder` + `VectorIndex` Protocol pair is the third runtime-checkable Protocol layer in the project (after `Model` and `Extractor`); ADR-0003 binds.

**Upstream-data sharp edge.** The Stage 13 stream endpoint frequently leaves `message_description` NULL on live responses (`src/bristol_ml/ingestion/remit.py:431-436`; Stage 14 already addressed this). Stage 15 inherits the constraint: the embedder must accept NULL `message_description` and synthesise embeddable text from the structured fields. **This is mechanical reuse of Stage 14's OQ-B resolution**, not a new decision.

---

## 1. Decisions

The decision set is twenty rows. To respect reviewer time, the lead split them into:

- **§1.A - five rows that genuinely shape the architecture** (engaged with at Ctrl+G 2026-04-27).
- **§1.B - fifteen rows whose defaults bind on the strength of intent + research convergence**.

The full Scope Diff is at [`docs/lld/research/15-embedding-index-scope-diff.md`](../../lld/research/15-embedding-index-scope-diff.md). The `@minimalist` flagged six rows for cuts/softening. After Ctrl+G review:

- **Cuts kept:** D11 (cross-stage cell — *flipped back to keep* per A5), D13 (`git_sha` dropped), D17 (no CLI flag, just YAML field per A4 narrowing), CLI tests (per A4).
- **Cuts overturned by Ctrl+G:** D6 (PCA → UMAP per A3 — user judged PCA inadequate for high-dim neighbourhood preservation), D11 (the optional cross-stage cell *is* in scope per A5).
- **Softened:** NFR-8 (loguru convention without exact-message-text test assertions).

### §1.A - Five decisions, Ctrl+G-resolved

| # | Decision | Final disposition | Resolution rationale |
|---|---|---|---|
| **A1 (=D1)** | **Layer placement.** | **New sibling layer.** `src/bristol_ml/embeddings/` + `docs/architecture/layers/embeddings.md` + ADR-0008. | The existing `llm/` layer is built around the `Extractor` Protocol's free-text → structured-fields contract; semantic search is a different abstraction. Two short layer docs read better than one long one. |
| **A2 (=D4)** | **Embedding model checkpoint default.** | **`Alibaba-NLP/gte-modernbert-base`** (149 M params, 768-dim, MTEB-56 avg **64.38**, ~298 MB safetensors fp32 / ~149 MB at fp16; Apache 2.0; no `trust_remote_code`; max_seq=8192; requires `transformers >= 4.48.0`). Query-side prefix `"query: "`; documents have no prefix (E5/GTE convention). Inference uses `model_kwargs={"torch_dtype": torch.float16}` for half-precision RAM. | SOTA-verification follow-up (2026-04-27) confirmed the user's instinct that the originally-recommended 2022-2023 checkpoints (BGE-small / GTE-small / MiniLM-L6) are superseded. ModernBERT (Dec 2024 — RoPE, Flash Attention 2, alternating local/global) lifts MTEB-56 by +2.21 over BGE-small while keeping deps minimal (no `trust_remote_code`, Apache 2.0). 768-dim doubles the cache footprint vs 384-dim but stays inside laptop-RAM bounds (R-2). Evidence: [`docs/lld/research/15-embedding-index-domain-update.md`](../../lld/research/15-embedding-index-domain-update.md). |
| **A3 (=D6)** | **2D projection.** | **UMAP** (`umap-learn`, `random_state=42`, `n_components=2`). Adds `umap-learn` + numba + llvmlite ~200 MB compiled stack. | User's argument: PCA's two principal components do not preserve local neighbourhood structure in 768-dim space, which undermines the demo moment ("the 2D projection shows the corpus falling into visible clusters by event type and fuel"). Dependency cost accepted. The Scope Diff's headline cut is overturned for visual-quality reasons. |
| **A4 (=D10)** | **CLI surface.** | **Standalone-module entry only.** `python -m bristol_ml.embeddings` prints active config + a sample query and exits 0 (mirrors Stage 14's `extractor.py` `__main__`). No `build`/`query` subcommands. | Intent line 45 phrases the full CLI as "useful, not mandatory". The notebook plus the standalone entry covers both demo paths. Saves two implementation surfaces and two CLI tests. |
| **A5 (=D11)** | **Stage 14 cross-stage cell.** | **Include as optional cell.** Notebook gets one extra cell that joins Stage 14 structured-extraction features when their parquet output is present, skipping cleanly with a printed banner when absent. The skip-path is exercised in CI (Stage 14 output is *not* present in stub-mode CI). | The intent's demo arc benefits from the joined view (intent line 44: "a notebook that combines both is more powerful than either alone"). The skip-clean guard is one branch with a clear test; not enough complexity to defer to Stage 16. |

### §1.B - Fifteen decisions that bind on default

The Evidence column cites the artefact that resolved each decision. Tags from the Scope Diff: most are `RESTATES INTENT`, two are `HOUSEKEEPING`. No engagement needed unless something looks wrong.

| # | Decision | Default | Tag | Evidence |
|---|---|---|---|---|
| **D2** | Embedder Protocol shape | `runtime_checkable` Protocol with `embed(text: str) -> np.ndarray` and `embed_batch(texts: list[str]) -> np.ndarray`. Mirrors Stage 14's `Extractor`. | RESTATES INTENT | AC-5; ADR-0003. |
| **D3** | VectorIndex Protocol + implementations | `runtime_checkable` Protocol with `add(ids, vecs)`, `query(vec, k) -> list[(id, score)]`, `save(path)`, `load(path)`. Two implementations: `NumpyIndex`, `StubIndex`. | RESTATES INTENT | AC-5; intent line 39. |
| **D5** | Vector-store backend | Plain numpy: `(corpus_matrix @ query.T)` on pre-normalised float32. Zero new deps. FAISS swap path documented in module CLAUDE.md. | RESTATES INTENT | Intent line 39; domain research §2; Codebase §7 confirms numpy is already a transitive dep. |
| **D7** | Cache invalidation | SHA-256 of the deterministic serialisation of the corpus text column + the configured `model_id` string. Both stored in Parquet `custom_metadata`. Recompute & compare on load; rebuild on mismatch. `--rebuild` escape hatch (D17). | RESTATES INTENT | Intent lines 40-41; domain research §4. |
| **D8** | Stub-first triple gate | YAML default `embedding.type: stub` + `BRISTOL_ML_EMBEDDING_STUB=1` env-var override + model-availability check at init. Mirrors Stage 14 D4. | RESTATES INTENT | AC-4; DESIGN.md §2.1.3; Stage 14 precedent. |
| **D9** | Text field embedded | `message_description` only. NULL fallback synthesises embeddable text from `event_type + cause + fuel_type + affected_unit` (mirrors Stage 14 OQ-B resolution). | RESTATES INTENT | Intent line 14; remit.py:431-436; Stage 14 OQ-B (Ctrl+G 2026-04-27). |
| **D12** | Offline guarantee in CI | `HF_HUB_OFFLINE=1` set in `tests/conftest.py` (or `tests/embedding/conftest.py`) before any HF import. `StubEmbedder` is the test default. No HTTP calls. | RESTATES INTENT | AC-4; DESIGN.md §2.1.3; domain research §5. |
| **D13** | Provenance fields on cache | Three fields in Parquet `custom_metadata`: `embedded_at_utc` (UTC-aware datetime, ISO-8601 string), `corpus_sha256` (12-char hex prefix), `model_id` (the configured model name). **Dropped:** `git_sha` per Scope Diff (no AC; duplicates registry-layer convention for a cache file). | RESTATES INTENT (softened by Scope Diff) | DESIGN.md §2.1.6; intent lines 40-41; Scope Diff D13. |
| **D14** | Cache file format | Single Parquet file with vector list-column (one row per event), `custom_metadata` carrying the D13 fields. Not split `.npy` + sidecar JSON. Path: `data/embeddings/<model_id_sanitised>.parquet` (gitignored). | RESTATES INTENT | Intent line 17; domain research §4. |
| **D15** | Dependency additions | Runtime: `sentence-transformers` (live embedder) and `umap-learn` (notebook 2D projection — restored per A3 flip). Numba + llvmlite arrive as transitive deps. **No FAISS, no LanceDB, no chromadb.** | RESTATES INTENT | AC-4 live path; A3 disposition; Scope Diff D15. |
| **D16** | Layer doc + ADR | New `docs/architecture/layers/embeddings.md` + new `docs/architecture/decisions/0008-embedding-index-protocol.md` recording the Protocol-pair design (binds D2 + D3). Indexed in `docs/architecture/README.md`. | HOUSEKEEPING | Cross-stage convention; codebase map §2 confirms layer-doc pattern. |
| **D17** | Rebuild escape hatch | `force_rebuild: bool = False` in the YAML schema. Surfaced as a kwarg on the public build entrypoint. The §2.1.1 standalone-module entry (A4) does not need a CLI flag because the YAML override covers it. | RESTATES INTENT (narrowed by A4) | Intent line 41. |
| **D18** | Test fixtures | `tests/fixtures/embedding/tiny_corpus.parquet` - 5-10 REMIT rows covering NULL and non-NULL `message_description`. `StubEmbedder` returns deterministic fixed-dim vectors (small `n=8`-dim is fine for tests). No external test fixtures needed. | RESTATES INTENT | DESIGN.md §2.1.7; codebase map §6. |
| **D19** | Notebook builder | `scripts/_build_notebook_15.py` mirroring `_build_notebook_14.py`. Notebook executes top-to-bottom in CI under `BRISTOL_ML_EMBEDDING_STUB=1` (or against pre-warmed model cache + `HF_HUB_OFFLINE=1`). Five cells: bootstrap, gold-set load, build/load index, top-k query, 2D projection plot. | RESTATES INTENT | Intent §Scope; DESIGN.md §2.1.8; codebase map §5. |
| **D20** | Live-path cassette | None. The embedding model is local; there are no HTTP calls to record once the model is cached. `HF_HUB_OFFLINE=1` covers the live path in CI. | RESTATES INTENT | Domain research §5; supersedes Stage 14's cassette pattern (which addressed an authenticated remote endpoint). |

### Non-functional requirements

| # | NFR | Source |
|---|---|---|
| **NFR-1** | Offline by default. No HTTP at init or inference; `HF_HUB_OFFLINE=1` in test config. | AC-4; DESIGN.md §2.1.3. |
| **NFR-2** | Idempotent cache. Re-run with valid cache is a no-op (read-only). Stale cache (corpus or model change) triggers automatic rebuild. | AC-3; DESIGN.md §2.1.5. |
| **NFR-3** | YAML configuration. `model_id`, `cache_path`, `vector_backend`, `default_top_k`, `projection.type`, `force_rebuild` all in YAML. Pydantic model with `extra="forbid"`, `frozen=True`. | DESIGN.md §2.1.4. |
| **NFR-4** | Narrow swap-safe Protocols. Embedder = 2 methods; VectorIndex = 4 methods. Callers hold the Protocol type. | AC-5. |
| **NFR-5** | Provenance. Cache carries `embedded_at_utc`, `corpus_sha256`, `model_id` (D13). | DESIGN.md §2.1.6; intent lines 40-41. |
| **NFR-6** | CPU-affordable runtime. First-run embed of a few-thousand-event corpus completes in under ~5 minutes on a laptop CPU; cached re-run completes in seconds. Benchmark on actual corpus and record in stage LLD. | AC-1. |
| **NFR-7** | Thin notebook. All compute in the module; notebook is import + display only. | DESIGN.md §2.1.8. |
| **NFR-8** | Loguru observability (convention only). Use `loguru` per project convention; emit structured records on cache hit, cache rebuild, and stale-cache detection. **Tests do not assert exact message text** - softened from the draft NFR-8 per Scope Diff. | DESIGN.md §2.1; Scope Diff NFR-8 (PLAN POLISH softened). |

---

## 2. Architecture sketch

```
src/bristol_ml/embeddings/
+- __init__.py             # Public boundary: Embedder Protocol, VectorIndex Protocol,
|                          # build_embedder factory, build_index factory, EmbeddingCache
|                          # dataclass, ENV_VAR constants. Re-exports from submodules.
+- _embedder.py            # StubEmbedder, SentenceTransformerEmbedder.
|                          # ST embedder: model_kwargs={"torch_dtype": torch.float16};
|                          # embed(text) prepends "query: "; embed_batch(texts) does not
|                          # (E5/GTE-modernbert prefix asymmetry).
+- _index.py               # NumpyIndex, StubIndex
+- _cache.py               # EmbeddingCache: read/write Parquet with custom_metadata,
|                          # SHA-256 corpus hashing, freshness check, rebuild dispatch.
|                          # Vector matrix kept as float32 even when inference is fp16.
+- _projection.py          # UMAP wrapper used by the notebook (random_state=42, n_components=2,
|                          # n_jobs=1 for coordinate-exact reproducibility per R-3)
+- _factory.py             # build_embedder(config), build_index(config, dim)
+- __main__.py             # python -m bristol_ml.embeddings: prints active config +
                           # sample top-k for a fixed gold event id; mirrors Stage 14
                           # extractor.py __main__ shape

conf/
+- embedding/              # singular — matches AppConfig.embedding field +
                           # `+embedding=default` Hydra override + package directive
   +- default.yaml         # type: stub by default; model_id, cache_path, top_k,
                           # projection.type, force_rebuild

tests/
+- unit/embeddings/
|  +- test_protocol.py             # Embedder + VectorIndex isinstance(_, Protocol)
|  +- test_stub_embedder.py        # stub default, env-var gate, deterministic vectors
|  +- test_index_query.py          # top-k sorted, cosine score in [-1, 1]
|  +- test_cache_invalidation.py   # corpus-hash mismatch -> rebuild; model-id mismatch
|  |                               # -> rebuild; valid cache -> no rebuild
|  +- test_text_synthesis.py       # NULL message_description -> synthesised text
|  +- test_module_runs_standalone.py  # python -m bristol_ml.embeddings exits 0
+- integration/
   +- test_notebook_15.py    # nbconvert --execute under BRISTOL_ML_EMBEDDING_STUB=1
+- fixtures/embedding/
   +- tiny_corpus.parquet    # 5-10 rows; NULL and non-NULL message_description
```

The runtime gates from Stage 14 transplant 1:1 with renamed env vars:
`BRISTOL_ML_EMBEDDING_STUB=1` and `BRISTOL_ML_EMBEDDING_MODEL_PATH` (optional override).

---

## 3. Risks

| # | Risk | Mitigation |
|---|---|---|
| **R-1** | Model weight download fails / slow at meetup over hotel Wi-Fi. | Document a one-line pre-warm command in `src/bristol_ml/embeddings/CLAUDE.md`: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"` (~298 MB safetensors fp32, one-time). Add to README's "Before the meetup" section. CI uses StubEmbedder; weights never download in CI. |
| **R-2** | Corpus size vs available RAM at scale. | At 768-dim float32, 5k events ≈ 15 MB; 50k ≈ 150 MB; 500k ≈ 1.5 GB. fp16 inference (`torch_dtype=torch.float16`) halves the *model* footprint to ~149 MB but the cache vector matrix stays float32 for query stability. Document tested corpus size in `docs/lld/stages/15-embedding-index.md`. No config-level `max_corpus_size` guard (would be `PREMATURE OPTIMISATION` per Scope Diff). |
| **R-3** | Non-deterministic similarity / projection across reruns. | Sentence-transformer inference is deterministic given identical model + hardware. fp16 inference can introduce minor numeric jitter vs fp32 — tests that assert exact cosine values run under StubEmbedder, which is fp32. UMAP with `random_state=42` is coordinate-exact-reproducible only when single-threaded (per umap-learn docs); the wrapper sets `n_jobs=1` to guarantee this. CI test asserts top-1 neighbour stable for fixture event; projection-coordinate determinism is asserted on the small fixture corpus. |
| **R-4** | Stage 13 schema drift renames the `message_description` column. | Integration test reads the fixture parquet and asserts the column is present before passing to the embedder (mirrors Stage 14 R-4). |
| **R-5** | Cache stale silently after corpus update. | NFR-2 + D7 content-hash invalidation makes this loud (WARNING log + automatic rebuild). |
| **R-6** | UMAP install footprint blocks a constrained CI runner (numba + llvmlite ~200 MB, first-import compile time non-trivial). | Test on the existing CI runner before merging Phase 2. If first-import compile dominates total CI runtime, set `NUMBA_CACHE_DIR` to a workspace-local path so the JIT cache persists across CI runs. Fallback (only if blocked): switch the projection to t-SNE (sklearn, no extra dep) — pre-clear with the human before flipping. |

---

## 4. Acceptance criteria (intent restatement)

- **AC-1.** Embedding the corpus runs end-to-end on a laptop CPU in a reasonable time. (NFR-6 quantifies "reasonable" as < 5 min for a first-run of the live demo corpus.)
- **AC-2.** Nearest-neighbour queries return in well under a second.
- **AC-3.** The embedding cache means re-running the notebook is fast.
- **AC-4.** The embedding model has no external API dependency - it runs locally.
- **AC-5.** The index's interface is small enough that swapping the vector store is a mechanical change.

Plus implicit ACs from the requirements artefact §3:
- **AC-6.** Similarity scores rendered alongside neighbours in the notebook.
- **AC-7.** 2D projection produces a plot.
- **AC-8.** Cache invalidation is detectable (loud, not silent).
- **AC-9.** Module runs standalone via `python -m bristol_ml.embeddings`.
- **AC-10.** Public interface has at least one smoke test.

---

## 5. Notebook structure

`notebooks/15_embedding_index.ipynb` - six executable cells plus a leading title-markdown cell:

1. **Bootstrap.** Print active embedding config (model_id, cache path, projection type). Marker: `T5 Cell 1`.
2. **Gold set load.** Load the REMIT fixture parquet; print row count and NULL `message_description` ratio. Marker: `T5 Cell 2`.
3. **Build / load index.** Triggered build under stub mode; print cache hit/miss + provenance fields. Marker: `T5 Cell 3`.
4. **Top-k query.** Pick a fixed gold event id; print the ten nearest neighbours with cosine scores. Marker: `T5 Cell 4`.
5. **2D projection.** UMAP on the corpus matrix (`random_state=42`, `n_components=2`, `n_jobs=1`); matplotlib scatter coloured by event type or fuel. Marker: `T5 Cell 5`.
6. **Optional cross-stage join (per A5).** If the Stage 14 extracted-features parquet is present at the documented path, join structured features (`fuel_type`, `event_type` from extraction) onto the nearest-neighbour result and re-print the top-k with the structured columns alongside cosine score. If absent, print a single-line banner ("Stage 14 output not found at <path> — skipping cross-stage join.") and continue. Cell must execute cleanly under both branches in CI. Marker: `T5 Cell 6`.

Total: 8 cells (1 title-markdown + 6 code + 1 closing-discussion markdown), one cell richer than Stage 14's 7-cell layout — the trailing discussion cell narrates the cross-stage join's optionality so a facilitator presenting from the notebook does not need to ad-lib.

---

## 6. Task list (for Phase 2)

| Task | Description | Acceptance test |
|---|---|---|
| **T1** | Public boundary types: `Embedder` Protocol, `VectorIndex` Protocol, `EmbeddingConfig` Pydantic schema, env-var constants. | `tests/unit/embeddings/test_protocol.py::test_embedder_and_index_satisfy_protocol_structurally` |
| **T2** | `StubEmbedder` + `StubIndex`: deterministic fixed-dim vectors, in-memory query. | `tests/unit/embeddings/test_stub_embedder.py::test_stub_default_when_env_var_set`, `test_stub_returns_same_vector_per_text` |
| **T3** | `NumpyIndex`: pre-normalise + matmul query; sorted top-k; save/load via `np.save`. | `tests/unit/embeddings/test_index_query.py::test_query_returns_sorted_top_k` |
| **T4** | `SentenceTransformerEmbedder`: HF_HUB_OFFLINE-respecting init, `model_kwargs={"torch_dtype": torch.float16}` for half-precision RAM, `encode()` wrapper, batch path. **Query-side prefix:** `embed(text)` prepends `"query: "` to the input; `embed_batch(texts)` does *not* (documents take no prefix per E5/GTE-modernbert convention). The prefix asymmetry is documented on the public method docstrings and asserted in T2's stub-embedder test where the stub mirrors the prefix contract for parity. | `tests/unit/embeddings/test_st_embedder.py::test_st_embedder_loads_offline` (requires pre-warmed cache; xfail/skip if absent in CI). |
| **T5** | `EmbeddingCache`: corpus SHA-256, Parquet write + read with `custom_metadata`, freshness check, rebuild dispatch. | `tests/unit/embeddings/test_cache_invalidation.py::test_cache_hit_no_rebuild`, `test_corpus_change_invalidates`, `test_model_change_invalidates` |
| **T6** | NULL `message_description` synthesis (mirrors Stage 14 OQ-B). | `tests/unit/embeddings/test_text_synthesis.py::test_null_message_uses_structured_fallback` |
| **T7** | `build_embedder` + `build_index` factory functions; YAML schema in `conf/_schemas.py`; `conf/embeddings/default.yaml`; round-trip test. | `tests/unit/test_config.py` extension. |
| **T8** | `python -m bristol_ml.embeddings` standalone-module entry: prints config + sample top-k, exits 0. | `tests/unit/embeddings/test_module_runs_standalone.py::test_module_runs_under_stub` |
| **T9** | `notebooks/15_embedding_index.ipynb` via `scripts/_build_notebook_15.py`; nbconvert smoke test. Includes the Stage 14 optional-join cell with skip-clean-on-absent guard (per A5). | `tests/integration/test_notebook_15.py::test_notebook_15_executes_top_to_bottom`; the absent-Stage-14-output skip path is the default CI codepath under stub mode. |
| **T10** | Module `CLAUDE.md`; layer doc `docs/architecture/layers/embeddings.md`; ADR-0008; `docs/lld/stages/15-embedding-index.md`; CHANGELOG entry; README "Before the meetup" pre-warm note. | docs review at Phase 3. |

---

## 7. Dependencies

**New runtime:** `sentence-transformers` (live embedder; brings `transformers >= 4.48.0` — required for ModernBERT support per A2 binding —, `tokenizers`, `huggingface_hub`; `torch` already in lock) and `umap-learn` (notebook 2D projection per A3; brings `numba` + `llvmlite` + `pynndescent`). No `faiss-cpu`, no `chromadb`, no `lancedb`.

**Version pin note (A2).** `Alibaba-NLP/gte-modernbert-base` requires `transformers >= 4.48.0` for the ModernBERT model class; record the resolved version in the lockfile in T7's PR. If the project's pinned `transformers` is older, T4 will fail at `SentenceTransformer(...)` with an "unrecognized model" error before any inference happens — surface to the human at that point rather than silently bumping.

**No new dev deps.** `numpy`, `scikit-learn`, `pyarrow` already present.

---

## 8. Exit checklist

- [ ] `uv run pytest -q` clean (target: ~12-13 new tests in `tests/unit/embeddings/` + 1 integration test).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `python -m bristol_ml.embeddings` exits 0 under stub mode.
- [ ] `notebooks/15_embedding_index.ipynb` executes top-to-bottom via nbconvert.
- [ ] `docs/architecture/layers/embeddings.md` written; ADR-0008 written; both indexed in `docs/architecture/README.md`.
- [ ] `docs/lld/stages/15-embedding-index.md` retrospective written.
- [ ] `CHANGELOG.md` entry under `[Unreleased]`.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/` in the final commit.
- [ ] Three Phase-3 reviewers (`arch-reviewer`, `code-reviewer`, `docs-writer`) run; blocking findings addressed; PR description drafted from synthesis.
