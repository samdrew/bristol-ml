# Stage 15 — Embedding index for REMIT: domain research

**Date:** 2026-04-27
**Intent source:** `docs/intent/15-embedding-index.md`
**Baseline SHA:** main @ `74c8d03` (Stage 14 merged)

**Scope:** External landscape for six technical questions that shape Stage 15 design choices. British English. No interface sketches — that is the lead's job.

---

## 1. Sentence-transformer model selection for short technical English on CPU

### Canonical sources

| Source | Summary |
|---|---|
| [BAAI/bge-small-en-v1.5 — HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33.4 M params, 384-dim, MTEB avg 62.17 across 56 datasets; STS 81.59 |
| [thenlper/gte-small — HuggingFace](https://huggingface.co/thenlper/gte-small) | 33.4 M params, 384-dim, 70 MB, MTEB avg 61.36; STS 82.07 |
| [all-MiniLM-L6-v2 — HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22.7 M params, 384-dim, ~91 MB; 215 M monthly downloads; 256-token limit |
| [all-mpnet-base-v2 — Zilliz guide](https://zilliz.com/ai-models/all-mpnet-base-v2) | 110 M params, 768-dim, ~420 MB; MTEB 57.78; CPU ~170 sent/sec |
| [SBERT pretrained models — sbert.net](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) | Speed table: MiniLM-L6 ~750 sent/sec CPU; mpnet-base ~170 sent/sec CPU |
| [MTEB Leaderboard — HuggingFace Spaces](https://huggingface.co/spaces/mteb/leaderboard) | Live benchmark — rankings shift; verify before pinning |
| [Model comparison FAQ — Milvus](https://milvus.io/ai-quick-reference/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2) | MiniLM-L6 is 4-5x faster than mpnet-base on CPU |
| [BGE model memory — RunThisModel](https://runthismodel.com/models/bge-small-en-v1.5) | fp16 download: 63 MB |

### Findings

| Model | Params | Dim | MTEB avg (56 datasets) | Download | CPU throughput (approx) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7 M | 384 | ~57-58 (older subset) | ~91 MB | ~750 sent/sec |
| all-mpnet-base-v2 | 110 M | 768 | 57.78 | ~420 MB | ~170 sent/sec |
| BGE-small-en-v1.5 | 33.4 M | 384 | **62.17** | 63 MB (fp16) | comparable to MiniLM |
| GTE-small | 33.4 M | 384 | 61.36 | 70 MB | comparable to MiniLM |

BGE-small-en-v1.5 and GTE-small both outperform all-MiniLM-L6-v2 on the full 56-dataset MTEB average at the same size and CPU cost. The STS sub-task (most relevant for "find semantically similar outage events") shows BGE-small at 81.59 and GTE-small at 82.07 - a meaningful margin above older MiniLM benchmarks. all-mpnet-base-v2 is 4-5x slower on CPU and ~420 MB with no MTEB advantage in the 384-dim tier; unjustifiable for a laptop demo.

REMIT texts are short (<50 words). all-MiniLM-L6-v2 has a 256-token training limit (truncation at 512), whereas BGE-small and GTE-small accept 512 - more headroom, not critical for short descriptions. No domain-tuned energy/REMIT embedding model exists (see Section 6).

MTEB rankings are volatile - new 384-dim models appear frequently. The scores cited are from model cards indexed before April 2026; verify at the live leaderboard before finalising the plan.

**Recommended default for Stage 15:** `BAAI/bge-small-en-v1.5`. Highest MTEB-56 average in the sub-100 MB / 384-dim tier, Apache 2.0, 512-token limit. Fall back to `all-MiniLM-L6-v2` only if hub-availability friction arises.

---

## 2. Vector-store backends

### Canonical sources

| Source | Summary |
|---|---|
| [FAISS GitHub - facebookresearch/faiss](https://github.com/facebookresearch/faiss) | IndexFlatIP for cosine (after L2-normalise); write_index/read_index serialisation |
| [faiss-cpu - PyPI](https://pypi.org/project/faiss-cpu/) | v1.13.2 (Dec 2025); wheel 18-24 MB Linux; Python 3.10-3.14; no Python-level deps |
| [FAISS vs HNSWlib - Zilliz](https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search) | FAISS: multi-algorithm library; HNSWlib: fast ANN, simpler API |
| [hnswlib - PyPI](https://pypi.org/project/hnswlib/) | v0.8.0 (Dec 2023, dormant 16+ months); cosine native; save_index/load_index |
| [sklearn NearestNeighbors - sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) | brute metric=cosine; already a project dependency; serialises via pickle |
| [LanceDB vs Qdrant - Zilliz](https://zilliz.com/comparison/qdrant-vs-lancedb) | LanceDB embedded no-server, Rust-backed; Qdrant service-oriented, heavier |
| [Embeddings in Parquet - Max Woolf](https://minimaxir.com/2025/02/embeddings-parquet/) | Numpy arrays stored as Parquet list-column with metadata |
| [FAISS cosine pattern - myscale](https://www.myscale.com/blog/faiss-cosine-similarity-enhances-search-efficiency/) | `faiss.normalize_L2(vecs)` then IndexFlatIP = cosine |

### Findings

At 5 000 vectors, every backend trivially satisfies AC-2 (sub-second queries). The decision is about install footprint and interface simplicity.

**Plain numpy (`query @ corpus_matrix.T` on pre-normalised float32):** The 5 000 x 384 corpus matrix is ~7 MB in RAM. A BLAS-backed matmul returns all cosine scores in well under 1 ms. Serialisation: `np.save` / `np.load`. No index build step, no opaque binary format, zero new dependencies. Scales as O(n) in query time - acceptable up to ~100 k vectors at which point approximate indexing becomes worthwhile. That is outside Stage 15 scope.

**scikit-learn `NearestNeighbors(algorithm='brute', metric='cosine')`:** Wraps the same BLAS matmul with a familiar API, returns sorted neighbour indices directly. Already a project dependency. Serialises via pickle. Negligible overhead at 5 k vectors.

**FAISS `IndexFlatIP`:** `faiss-cpu` v1.13.2 wheel is 18-24 MB (Linux). Cosine requires pre-normalising vectors; IndexFlatIP computes inner product equivalent to cosine on unit vectors. `faiss.write_index` / `faiss.read_index` for binary serialisation. Low-level C-style API (add/search). Justified when corpus grows past ~100 k and approximate IVF/HNSW indexes are needed - not yet.

**hnswlib:** Last release December 2023; dormant. Approximate ANN. No advantage over numpy at 5 k vectors. Release cadence is a mild concern for dependency pinning.

**LanceDB / ChromaDB / Qdrant-local:** All three are substantially heavier installs (Rust extensions, transitive deps). ChromaDB has a 2025 Rust rewrite. None add anything for 5 k vectors that the two-line numpy approach does not already provide.

**Recommended default for Stage 15:** Plain numpy matrix (pre-normalised float32, `.npy` file). Satisfies AC-2 and AC-5. Zero new dependencies. The swap path to `faiss.IndexFlatIP` is one function replacement - document it in the module `CLAUDE.md`.

---

## 3. 2D projection for visualisation

### Canonical sources

| Source | Summary |
|---|---|
| [UMAP reproducibility - umap-learn docs](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) | `random_state` forces single-threaded mode (~2x slower) but guarantees coordinate-exact repeatability |
| [UMAP benchmarking - umap-learn docs](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) | At 3 200-6 400 samples UMAP outperforms sklearn t-SNE substantially in speed; runtime seconds |
| [umap-learn - PyPI](https://pypi.org/project/umap-learn/) | v0.5.12 (Apr 2026); depends on numba + llvmlite + pynndescent; wheel 92 KB but numba pulls ~200 MB compiled stack |
| [PCA vs t-SNE vs UMAP - biostatsquid](https://biostatsquid.com/pca-umap-tsne-comparison/) | PCA deterministic and fastest; both t-SNE and UMAP stochastic but seed-stabilisable |
| [UMAP O(n log n) vs t-SNE O(n^2) - MCP Analytics](https://mcpanalytics.ai/whitepapers/whitepaper-umap) | UMAP dramatically faster for >=6 400 samples |
| [PCA vs UMAP on short news text - GDELT blog](https://blog.gdeltproject.org/visualizing-an-entire-day-of-global-news-coverage-technical-experiments-pca-vs-umap-for-hdbscan-t-sne-dimensionality-reduction/) | UMAP produces more separated, interpretable clusters on short news text than PCA |

### Findings

**PCA:** Deterministic, milliseconds at 5 k x 384, no extra dependencies. Global variance preserved but two principal components explain a small fraction of variance in 384-dim space - clusters are blurred and unimpressive as a demo artefact.

**t-SNE (sklearn):** Stochastic; `random_state` makes it repeatable. Good local cluster preservation. sklearn's O(n^2) implementation is slow at 5 k points (order of minutes). `openTSNE` reduces this to seconds. A 2025 single-cell study found t-SNE had *higher* run-to-run reproducibility than UMAP, but that context may not generalise to text embeddings.

**UMAP:** Stochastic; `random_state=42` gives coordinate-exact repeatability but forces single-threaded optimisation (~2x slower). At 5 k vectors even single-threaded UMAP runs in seconds. Cluster separation is visually superior to PCA and competitive with t-SNE for short text (GDELT study). Install cost: numba + llvmlite ~200 MB of compiled stack. umap-learn v0.5.12 is Python >=3.9.

A notebook re-run with `random_state=42` will produce an identical picture each time - the UMAP docs confirm this to coordinate-exact precision when single-threaded.

**Recommended default for Stage 15:** UMAP (`umap-learn`, `random_state=42`, `n_components=2`). Accept the numba install cost for the visual quality of the demo. Fall back to PCA if umap-learn is excluded from the dependency set (deterministic, no new deps, weaker cluster signal).

---

## 4. Cache-invalidation patterns for embedding caches

### Canonical sources

| Source | Summary |
|---|---|
| [HuggingFace datasets cache - huggingface.co](https://huggingface.co/docs/datasets/en/about_cache) | Fingerprint = hash of Arrow table + dill-pickle hash of each transform; chain invalidates automatically |
| [SentenceTransformer API - sbert.net](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) | No built-in embedding-result cache; model weights cached via HF hub; embedding cache is user responsibility |
| [Embeddings in Parquet - Max Woolf](https://minimaxir.com/2025/02/embeddings-parquet/) | Stores embedding vectors as Parquet list-column alongside text metadata; no invalidation discussed |
| [HF hub env vars - huggingface.co](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) | `HF_HOME` / `HF_HUB_CACHE` control model weight cache root; separate from user embedding cache |

### Findings

`sentence-transformers` provides no embedding-result cache. The cache referred to in AC-3 is a user-level Parquet (or `.npy`) file produced after the first `model.encode()` run. Three patterns surveyed:

1. **Content-hash sidecar:** SHA-256 of the concatenated corpus text bytes + model identifier string stored in Parquet custom metadata (`pyarrow.parquet.write_table(..., custom_metadata={"corpus_sha256": ..., "model_id": ...})`). On load: recompute hash; rebuild if mismatch. This is minimal-but-correct: catches both corpus mutations and model swaps. Uses only pyarrow, already a project dependency.

2. **HuggingFace `datasets` fingerprint:** Dill-pickle hash of the transform function + parameters, chained across `.map()` calls. Elegant for pipeline transforms but brings in `datasets` as a dependency if not already present. Overkill for a single `model.encode()` call over a flat Parquet corpus.

3. **Manual rebuild command:** Delete and recreate the cache on demand. Zero automatic invalidation. The intent doc notes this as the simpler option; the hash pattern is safer for a corpus that will be updated between sessions.

The `datasets` fingerprint approach (dill-hash of function + args) is conceptually closest to what Stage 14 would have used for LLM API response caching. For Stage 15 the relevant state is simpler: which rows are in the corpus, and which model produced the embeddings. SHA-256 of a deterministic serialisation of the text column (e.g., SHA-256 of sorted text values joined with `\n`) plus the model name string covers both.

**Recommended default for Stage 15:** Store `{"corpus_sha256": <hex>, "model_id": <str>}` in the Parquet file's `custom_metadata` field via pyarrow at write time. Recompute and compare on load; rebuild if mismatched. No new dependencies; satisfies §2.1.6 provenance.

---

## 5. Sentence-transformers offline behaviour

### Canonical sources

| Source | Summary |
|---|---|
| [HF hub env vars - huggingface.co](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) | `HF_HUB_OFFLINE=1` blocks all HTTP to hub; raises `OfflineModeIsEnabled` if model not cached; must be set before import |
| [SentenceTransformer API - sbert.net](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) | `local_files_only=True` constructor arg prevents any download; `cache_folder` or `SENTENCE_TRANSFORMERS_HOME` overrides weight cache path |
| [HF transformers offline - huggingface.co](https://huggingface.co/docs/transformers/main/en/installation) | `TRANSFORMERS_OFFLINE=1` is legacy; `HF_HUB_OFFLINE=1` is the current canonical var |
| [Offline mode issue - sentence-transformers GitHub #1725](https://github.com/huggingface/sentence-transformers/issues/1725) | Older versions did not fully respect `HF_HUB_OFFLINE`; `local_files_only=True` is more reliable across versions |

### Findings

`SentenceTransformer("BAAI/bge-small-en-v1.5")` makes at least one HTTP request even after the model is cached - an ETag check to detect newer revisions. Setting `HF_HUB_OFFLINE=1` before importing `huggingface_hub` skips this check entirely, using only cached files. If the model is not cached and offline mode is active, the library raises `OfflineModeIsEnabled` (a clean error, not a network timeout).

Three complementary mechanisms:

1. **`HF_HUB_OFFLINE=1` env var:** Set at process start, before `import huggingface_hub`. The canonical way to enforce no-network-call in CI. Blocks all hub HTTP.
2. **`local_files_only=True` constructor arg:** Per-instance control; more reliable than the env var in some older sentence-transformers versions (GitHub issue #1725).
3. **Explicit local path:** `SentenceTransformer("/abs/path/to/model")` bypasses hub routing entirely. Useful in strict air-gap scenarios.

**Conventional pattern for "no API call" tests:**
- Download the model once in the dev environment; it lands at `~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<rev>/`.
- In `conftest.py` set `os.environ["HF_HUB_OFFLINE"] = "1"` (before any HF import), or use `monkeypatch.setenv`.
- Provide a `StubEmbedder` that returns deterministic random or zero vectors without loading any model, as the CI default (satisfies §2.1.3 stub-first).

`SENTENCE_TRANSFORMERS_HOME` controls the weight cache folder separately from `HF_HUB_CACHE`. In sentence-transformers v3.x, model loading delegates entirely to `huggingface_hub`, so `HF_HUB_CACHE` is the authoritative cache root.

**Recommended default for Stage 15:** `HF_HUB_OFFLINE=1` in `conftest.py` (or CI env). A `StubEmbedder` (deterministic fixed-dim vectors) as the test default. Document the one-time `SentenceTransformer('BAAI/bge-small-en-v1.5')` download command in the module `CLAUDE.md`.

---

## 6. REMIT-text-specific considerations

### Canonical sources

| Source | Summary |
|---|---|
| [Aviation domain adaptation - AIAA SciTech 2024 / arXiv:2305.09556](https://doi.org/10.2514/6.2024-2702) | General sentence-transformers underperform on abbreviation-heavy technical short text; TSDAE + fine-tuning required for material improvement |
| [Sentence transformer length handling - Zilliz FAQ](https://zilliz.com/ai-faq/how-do-sentence-transformers-handle-different-lengths-of-input-text-and-does-sentence-length-affect-the-resulting-embedding) | Very short inputs (2-3 words) produce shallower embeddings; 200-400 token texts optimal |
| [MTEB Leaderboard - HuggingFace Spaces](https://huggingface.co/spaces/mteb/leaderboard) | No energy, REMIT, or power-system domain in MTEB task taxonomy |
| [ML for electricity market agents - arXiv:2206.02196](https://arxiv.org/pdf/2206.02196) | RL/numeric methods; no NLP embedding component |
| [LLMs in building energy - ScienceDirect 2025](https://www.sciencedirect.com/article/pii/S0378778825015300) | Survey of LLM use in energy; no REMIT outage-text embedding prior art |

### Findings

**Prior art:** No published work combining REMIT outage notification text with sentence-transformer embeddings, semantic search, or clustering was found as of April 2026. Searches on "REMIT embedding", "outage notification clustering", "energy market event embedding", and "power system event NLP" returned numeric forecasting models, RL market agents, and general energy LLM surveys - nothing on semantic search over REMIT-style free text. This is genuinely uncharted territory at the meetup-demo scale.

**Tokenisation of technical abbreviations:** REMIT descriptions contain abbreviations ("NG", "CCGT", "MW", "kV", "TSO") and unit strings ("300 MW"). General BERT-style WordPiece tokenisers split unknown abbreviations into subword fragments (e.g. "CCGT" -> "CC" + "##GT") and do not preserve numeric-unit relationships. This degrades embedding quality relative to a domain-tuned vocabulary. However, surrounding natural-language words ("planned outage", "maintenance", "nuclear", "gas") carry most of the semantic signal. Semantically similar events cluster together because those words are well-represented in the model's training corpus.

The aviation domain adaptation study (AIAA 2024) is directly analogous: short technical English, abbreviation-heavy, domain vocabulary. They found general sentence-transformers produce adequate but sub-optimal retrieval, and that TSDAE pretraining on domain text followed by fine-tuning gave material improvement. Fine-tuning is explicitly out of scope for Stage 15 (intent doc §"Out of scope").

**Short-text depth:** REMIT descriptions are typically 10-40 words. At this length embeddings are adequate, though shorter inputs carry less context. The demo moment (planned nuclear outage -> neighbours that are also nuclear outages) is achievable because "nuclear" maps unambiguously. Distinguishing subtypes (planned vs. unplanned, partial vs. full outage) will be less reliable without the fuel-type flag from Stage 14's structured extraction.

**Optional mitigation:** BGE-small-en-v1.5 supports an optional query-instruction prefix (`"Represent this sentence for searching relevant passages:"`). This may improve short-text retrieval slightly. Worth demonstrating in the notebook as a commentary note, not a module requirement.

**Recommended default for Stage 15:** Accept a general-purpose model. Document the fuel-type limitation in the notebook commentary - this is honest pedagogy, not a gap. No domain fine-tuning.

---

## Summary table

| Decision | Recommended default | Key reason | Fallback |
|---|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` | Best MTEB-56 avg in sub-100 MB / 384-dim tier | `all-MiniLM-L6-v2` |
| Vector backend | Plain numpy matrix (`.npy`) | Zero new deps; sub-ms at 5 k; trivially swappable | `faiss.IndexFlatIP` |
| 2D projection | UMAP `random_state=42` | Best visual cluster separation for meetup demo | PCA (deterministic, no numba) |
| Embedding cache | SHA-256(corpus text) + model_id in Parquet custom_metadata | Provenance §2.1.6; pyarrow already in project | Manual rebuild CLI command |
| Offline / CI | `HF_HUB_OFFLINE=1` + `StubEmbedder` | No network calls in CI; satisfies §2.1.3 stub-first | `local_files_only=True` constructor arg |
| REMIT domain | General model; document fuel-type limitation | Fine-tuning out of scope; honest pedagogy | None (deferred) |

*MTEB scores cited from model cards indexed before April 2026. The live leaderboard at [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) should be verified before the plan is finalised - the 384-dim tier is competitive and new models appear frequently.*

---

## Sources

- [BAAI/bge-small-en-v1.5 - HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [thenlper/gte-small - HuggingFace](https://huggingface.co/thenlper/gte-small)
- [sentence-transformers/all-MiniLM-L6-v2 - HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-mpnet-base-v2 model guide - Zilliz](https://zilliz.com/ai-models/all-mpnet-base-v2)
- [MTEB Leaderboard - HuggingFace Spaces](https://huggingface.co/spaces/mteb/leaderboard)
- [SBERT pretrained models - sbert.net](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Model comparison FAQ - Milvus](https://milvus.io/ai-quick-reference/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2)
- [BGE model memory - RunThisModel](https://runthismodel.com/models/bge-small-en-v1.5)
- [FAISS GitHub - facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- [faiss-cpu - PyPI](https://pypi.org/project/faiss-cpu/)
- [FAISS vs HNSWlib - Zilliz](https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search)
- [hnswlib - PyPI](https://pypi.org/project/hnswlib/)
- [sklearn NearestNeighbors - sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
- [LanceDB vs Qdrant - Zilliz](https://zilliz.com/comparison/qdrant-vs-lancedb)
- [Embeddings in Parquet - Max Woolf](https://minimaxir.com/2025/02/embeddings-parquet/)
- [FAISS cosine pattern - myscale](https://www.myscale.com/blog/faiss-cosine-similarity-enhances-search-efficiency/)
- [UMAP reproducibility - umap-learn docs](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)
- [UMAP benchmarking - umap-learn docs](https://umap-learn.readthedocs.io/en/latest/benchmarking.html)
- [umap-learn - PyPI](https://pypi.org/project/umap-learn/)
- [PCA vs t-SNE vs UMAP - biostatsquid](https://biostatsquid.com/pca-umap-tsne-comparison/)
- [UMAP O(n log n) vs t-SNE O(n^2) - MCP Analytics](https://mcpanalytics.ai/whitepapers/whitepaper-umap)
- [PCA vs UMAP on short news text - GDELT blog](https://blog.gdeltproject.org/visualizing-an-entire-day-of-global-news-coverage-technical-experiments-pca-vs-umap-for-hdbscan-t-sne-dimensionality-reduction/)
- [HuggingFace datasets cache - huggingface.co](https://huggingface.co/docs/datasets/en/about_cache)
- [SentenceTransformer API - sbert.net](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
- [HF hub env vars - huggingface.co](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- [HF transformers offline - huggingface.co](https://huggingface.co/docs/transformers/main/en/installation)
- [Offline mode issue - sentence-transformers GitHub #1725](https://github.com/huggingface/sentence-transformers/issues/1725)
- [Aviation domain adaptation - AIAA SciTech 2024 / arXiv:2305.09556](https://doi.org/10.2514/6.2024-2702)
- [Sentence transformer length handling - Zilliz FAQ](https://zilliz.com/ai-faq/how-do-sentence-transformers-handle-different-lengths-of-input-text-and-does-sentence-length-affect-the-resulting-embedding)
- [ML for electricity market agents - arXiv:2206.02196](https://arxiv.org/pdf/2206.02196)
- [LLMs in building energy - ScienceDirect 2025](https://www.sciencedirect.com/article/pii/S0378778825015300)
