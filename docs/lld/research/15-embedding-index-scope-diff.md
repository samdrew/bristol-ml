# Stage 15 - Embedding index: Scope Diff

**Role:** fourth Phase-1 research artefact (`@minimalist` pre-synthesis critic).
**Inputs read:** `docs/intent/15-embedding-index.md`, the three Phase-1 research artefacts (`15-embedding-index-{requirements,codebase,domain}.md`), and the lead's draft decision set.
**Output:** tag every D-line / NFR / test / dep as `RESTATES INTENT`, `PLAN POLISH`, `PREMATURE OPTIMISATION`, or `HOUSEKEEPING`; close with the single highest-leverage cut.

---

## Tag table

| Item | Summary | Tag | Justification |
|------|---------|-----|---------------|
| D1 - `bristol_ml/embeddings/` module + layer doc | New top-level module, sibling to `llm/` | RESTATES INTENT | Intent §Scope "A module that produces embeddings"; DESIGN.md §module boundaries lists `models/` stage entries - a new sibling module is the correct placement. |
| D2 - Embedder Protocol | `runtime_checkable`, `embed` + `embed_batch` | RESTATES INTENT | AC-5 (swap-safe interface); ADR-0003 precedent. |
| D3 - VectorIndex Protocol + NumpyIndex + StubIndex | Four methods; two implementations | RESTATES INTENT | AC-5; intent §Points "plain numpy array with cosine similarity is enough." |
| D4 - `BAAI/bge-small-en-v1.5` default, `all-MiniLM-L6-v2` fallback | Model selection | PLAN POLISH | Intent §Points says "general-purpose sentence transformer is the default" without naming one; picking a specific model checkpoint is a plan-stage choice, not a contract requirement - forces the implementer to pre-warm a specific model in CI docs. |
| D5 - Plain numpy cosine on pre-normalised float32; zero new deps | Vector backend | RESTATES INTENT | Intent §Points "plain numpy array with cosine similarity is enough"; AC-5. |
| D6 - UMAP `random_state=42`; adds `umap-learn` + numba ~200 MB | 2D projection technique | PLAN POLISH | Intent §Scope "a 2D projection is enough for demo purposes"; §Points "any of them is fine." Choosing UMAP over PCA (zero extra deps) adds ~200 MB compiled stack and forces a new runtime dep for a cell whose visual output the intent calls illustrative - adds 1 dep, ~1 notebook cell of complexity. |
| D7 - SHA-256 of corpus text + model_id; `--rebuild` escape hatch | Cache invalidation | RESTATES INTENT | Intent §Points "a content hash of both makes invalidation automatic; a manual rebuild command is simpler" - both paths are explicitly named in the intent. |
| D8 - Triple-gate (YAML discriminator + env-var + availability check) | Stub-first enforcement | RESTATES INTENT | AC-4; DESIGN.md §2.1.3 stub-first; established pattern from Stage 14. |
| D9 - `message_description` only; NULL fallback synthesises from structured fields | Text field selection | RESTATES INTENT | Intent §Scope "embeddings for REMIT event descriptions"; NULL fallback mirrors Stage 14 OQ-B, which is established codebase pattern (remit.py:431-436). |
| D10 - Two CLI subcommands: `build` and `query` | CLI surface | PLAN POLISH | Intent §Points "useful for meetups... not mandatory" - phrased as a consideration, not a requirement. Forces 2 new tests (CLI build, CLI query) and additional implementation surface that the notebook alone would satisfy. |
| D11 - Optional notebook cell joining Stage 14 features | Stage 14 interaction | PLAN POLISH | Intent §Points "a notebook that combines both is more powerful" - phrased as an aspiration; intent line 4 says "conceptually, though not mechanically." An optional cell adds notebook-execution complexity with no AC coverage, and the guard logic (`if stage14_output.exists()`) must be tested for the skip path. |
| D12 - `HF_HUB_OFFLINE=1` in `conftest.py` + StubEmbedder as CI default | Offline CI guarantee | RESTATES INTENT | AC-4; DESIGN.md §2.1.3. |
| D13 - Four provenance fields in Parquet custom_metadata | `embedded_at_utc`, `corpus_sha256`, `model_id`, `git_sha` | PLAN POLISH | Intent §Points names content-hash invalidation; DESIGN.md §2.1.6 is the source. `embedded_at_utc` + `corpus_sha256` + `model_id` are load-bearing for cache invalidation and NFR-5. `git_sha` is not named by any AC and duplicates registry-layer provenance convention for a cache file rather than a model artefact - adds no invalidation capability, forces `registry._git` import into the embeddings module. |
| D14 - Single Parquet file with vector list-column + custom_metadata | Cache file format | RESTATES INTENT | Intent §Scope "Caching of embeddings"; domain research §4 recommends this pattern; pyarrow already a project dep. |
| D15 - `sentence-transformers` runtime; `umap-learn` optional | Dependency additions | PLAN POLISH | `sentence-transformers` is load-bearing (AC-4 live path). `umap-learn` is load-bearing only if D6 stays; it is a consequence of D6, not an independent decision - tag rides on D6's resolution. |
| D16 - `docs/architecture/layers/embeddings.md` + ADR-0008 | Layer doc + ADR | HOUSEKEEPING | Cross-stage hygiene; every new module layer gets a layer doc and ADR per convention. |
| D17 - `--rebuild` CLI flag + `force_rebuild: bool` config field | Rebuild escape hatch | RESTATES INTENT | Intent §Points "a manual rebuild command is simpler" - explicitly named. |
| D18 - `tests/fixtures/embedding/` 5-10-row parquet; StubEmbedder deterministic | Test fixtures | RESTATES INTENT | DESIGN.md §2.1.7; implicit AC-10; established fixture pattern. |
| D19 - `scripts/_build_notebook_15.py` + CI execution under stub | Notebook builder | RESTATES INTENT | Intent §Scope "A notebook that demonstrates the index"; DESIGN.md §2.1.8 thin-notebook convention; Stage 14 HOUSEKEEPING precedent. |
| D20 - No cassette; `HF_HUB_OFFLINE=1` covers live path | Live-path cassette decision | RESTATES INTENT | No HTTP calls to record; the offline mechanism (D12) is the correct substitute. |
| NFR-1 - Offline by default | No network calls | RESTATES INTENT | AC-4; DESIGN.md §2.1.3. |
| NFR-2 - Idempotent cache | Re-run is a no-op when cache valid | RESTATES INTENT | AC-3; DESIGN.md §2.1.5. |
| NFR-3 - YAML config | All tunables in YAML | RESTATES INTENT | DESIGN.md §2.1.4. |
| NFR-4 - Narrow swap-safe protocols | Protocol with <=4 methods | RESTATES INTENT | AC-5. |
| NFR-5 - Provenance | Cache metadata carries model+corpus identity | RESTATES INTENT | DESIGN.md §2.1.6; intent §Points lines 40-41. |
| NFR-6 - CPU-affordable runtime | First-run under ~5 min on laptop | RESTATES INTENT | AC-1. |
| NFR-7 - Thin notebook | No reimplemented logic in notebook | RESTATES INTENT | DESIGN.md §2.1.8. |
| NFR-8 - Loguru observability | INFO/WARNING/DEBUG log lines | PLAN POLISH | DESIGN.md §2.1 names loguru as the convention; specifying the exact log-line inventory (cache hit count, corpus size, query latency, stale WARNING) is a plan-stage prescription not tied to any AC - adds ~4 assertion points in tests to verify specific message text. |
| test - protocol structural conformance | `isinstance` checks against Protocol | RESTATES INTENT | AC-5; ADR-0003. |
| test - stub embedder default | YAML `type: stub` produces StubEmbedder | RESTATES INTENT | AC-4; DESIGN.md §2.1.3. |
| test - env-var gate | `BRISTOL_ML_EMBEDDING_STUB=1` overrides config | RESTATES INTENT | AC-4; Stage 14 D4 pattern. |
| test - cache hit no-rebuild | Second call reads cache | RESTATES INTENT | AC-3; NFR-2. |
| test - cache miss rebuilds | Cold start triggers embed | RESTATES INTENT | AC-3. |
| test - corpus-change invalidates | New corpus hash triggers rebuild | RESTATES INTENT | NFR-2; intent §Points line 40. |
| test - model-change invalidates | Different model_id triggers rebuild | RESTATES INTENT | NFR-2. |
| test - query returns sorted top-k | Scores descending, length = k | RESTATES INTENT | AC-2; intent §Demo moment. |
| test - NULL message synthesis fallback | NULL `message_description` uses structured fields | RESTATES INTENT | D9; AC-1 (corpus runs end-to-end including NULL rows). |
| test - CLI build subcommand | `python -m bristol_ml.embeddings build` exits 0 | PLAN POLISH | CLI is itself D10 (PLAN POLISH); if D10 is cut, this test vanishes - adds 1 test with no AC of its own. |
| test - CLI query subcommand | `python -m bristol_ml.embeddings query` exits 0 | PLAN POLISH | Same as above - rides on D10; adds 1 test. |
| test - notebook executes top-to-bottom | nbconvert passes | RESTATES INTENT | DESIGN.md §2.1.8 convention; Stage convention. |
| dep - `sentence-transformers` | Live embedder | RESTATES INTENT | AC-4 live path requires a local sentence-transformer model. |
| dep - `umap-learn` | 2D projection | PREMATURE OPTIMISATION | `umap-learn` is needed only for D6 (UMAP); intent §Points says "any of them is fine" including PCA, which adds zero deps. Numba + llvmlite bring ~200 MB compiled stack into the dev and optional-runtime dependency graph - a dependency whose sole purpose is visual polish for one notebook cell. |

---

## Single highest-leverage cut

**If you cut one item to halve this plan's scope, cut D6 (UMAP with `umap-learn`) and replace it with `sklearn.decomposition.PCA`, because doing so eliminates the only new heavyweight transitive dependency (~200 MB numba/llvmlite stack), removes `umap-learn` from D15, drops the two CLI tests that ride on D10 from needing a projection concern, and leaves the demo moment - visible clusters in 2D - fully intact using a zero-dep alternative the intent explicitly sanctions.**
