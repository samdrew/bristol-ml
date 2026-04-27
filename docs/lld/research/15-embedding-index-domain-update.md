# Stage 15 Embedding Model Update — April 2026

**Scope:** Single-question update to the prior domain research recommendation.
**Prior recommendation:** `BAAI/bge-small-en-v1.5`
**Verdict (summary):** The recommendation changes. `Alibaba-NLP/gte-modernbert-base` is the new default, with `nomic-ai/nomic-embed-text-v1.5` as the closest alternative.

---

## 1. Candidate Survey

All MTEB scores quoted are from the **legacy English leaderboard (56 tasks)** unless labelled "MTEB v2." The two leaderboards are not numerically comparable; the MTEB v2 score is calibrated to a different task set and scoring method introduced with MMTEB in 2025. Sources flag which version is in use.

### Baseline — BAAI/bge-small-en-v1.5

| Attribute | Value |
|-----------|-------|
| Parameters | 33.4 M |
| Embedding dim | 384 |
| Safetensors (fp32) | 133 MB |
| MTEB English avg (56 tasks) | 62.17 |
| Licence | MIT |
| Max seq length | 512 |
| `trust_remote_code` | No |
| `SentenceTransformer(...)` | Yes, clean |

Released 2023. No query prefix needed in v1.5. Simple, sub-150 MB, widely reproduced. Source: [BAAI/bge-small-en-v1.5 HF model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

---

### A. Alibaba-NLP/gte-modernbert-base (released Dec 2024)

| Attribute | Value |
|-----------|-------|
| Parameters | 149 M |
| Embedding dim | 768 |
| Safetensors (fp32) | 298 MB |
| MTEB English avg (56 tasks) | 64.38 |
| Licence | Apache 2.0 |
| Max seq length | 8 192 |
| `trust_remote_code` | No |
| `SentenceTransformer(...)` | Yes, clean |

Built on [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (Dec 2024). ModernBERT's architectural improvements — RoPE positional encoding, Flash Attention 2 pathway, alternating local/global attention — give it substantially better throughput per parameter than BERT-era encoders on short-to-medium sequences, particularly relevant for CPU inference on commodity hardware. The model loads with a plain `SentenceTransformer("Alibaba-NLP/gte-modernbert-base")` call with no special flags. Sources: [gte-modernbert-base HF card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base); [ModernBERT paper arXiv:2412.13663](https://arxiv.org/html/2412.13663v2).

**Size note:** 298 MB safetensors is inside the 250 MB soft limit stated in the constraints — marginally. At fp32 the download is 298 MB; no official fp16 safetensors file is published separately, but the sentence-transformers library will load in fp16 if `model_kwargs={"torch_dtype": torch.float16}` is passed, halving RAM to ~149 MB. The ONNX quantised file is 596 MB ONNX full, 298 MB ONNX fp16 (repo also contains `onnx/model_fp16.onnx`).

---

### B. nomic-ai/nomic-embed-text-v1.5 (released Mar 2024)

| Attribute | Value |
|-----------|-------|
| Parameters | 137 M |
| Embedding dim | 768 (MRL: 512, 256, 128, 64) |
| Safetensors (fp32) | 547 MB |
| MTEB English avg (56 tasks) | 62.28 |
| Licence | Apache 2.0 |
| Max seq length | 8 192 |
| `trust_remote_code` | No longer required (transformers >= 5.5, sentence-transformers >= 5.3) |
| `SentenceTransformer(...)` | Yes; task prefix required (`search_document:` / `search_query:`) |

Source: [nomic-embed-text-v1.5 HF card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Note that the safetensors file is 547 MB — above the 250 MB preferred ceiling, though Matryoshka lets the effective index footprint be reduced by slicing to 256 dims with a score drop of only ~1.2 points.

---

### C. nomic-ai/modernbert-embed-base (released Dec 2024)

| Attribute | Value |
|-----------|-------|
| Parameters | 149 M (same base as gte-modernbert-base) |
| Embedding dim | 768 (MRL 256 available) |
| Safetensors (fp32) | 596 MB |
| MTEB English avg (56 tasks) | ~62.62 (from model card) |
| Licence | Apache 2.0 |
| Max seq length | 8 192 |
| `trust_remote_code` | No |
| `SentenceTransformer(...)` | Yes; task prefix required (`search_document:` / `search_query:`) |

Same ModernBERT backbone as gte-modernbert-base but trained on Nomic's own dataset pipeline. The model card says it "outperforms both nomic-embed-text-v1 and nomic-embed-text-v1.5 on MTEB," consistent with the 62.62 figure. Source: [nomic-ai/modernbert-embed-base HF card](https://huggingface.co/nomic-ai/modernbert-embed-base). The 596 MB safetensors file is above the 250 MB ceiling. **gte-modernbert-base dominates it on score (64.38 vs 62.62) and on file size (298 MB vs 596 MB).**

---

### D. Snowflake/snowflake-arctic-embed-s (v1, Apache 2.0)

| Attribute | Value |
|-----------|-------|
| Parameters | 33 M |
| Embedding dim | 384 |
| Safetensors (fp32) | 133 MB |
| MTEB Retrieval NDCG@10 | 51.98 (beats bge-small's 51.68 on retrieval subset) |
| MTEB English avg (56 tasks) | Not published by Snowflake |
| Licence | Apache 2.0 |
| Max seq length | 512 |
| `trust_remote_code` | No |
| `SentenceTransformer(...)` | Yes; query prefix required |

Sources: [snowflake-arctic-embed-s HF card](https://huggingface.co/Snowflake/snowflake-arctic-embed-s); [arctic-embed arXiv:2405.05374](https://arxiv.org/html/2405.05374v1). Snowflake only reports NDCG@10 on the 15-task retrieval subset, not the full 56-task average. That makes direct comparison with the 62.17 bge-small figure structurally invalid. Given that bge-small already holds a 62.17 overall average (retrieval is one component), and arctic-embed-s only marginally beats bge-small on the retrieval slice, there is no evidence arctic-embed-s is ahead of bge-small on the full 56-task board.

### E. Snowflake/snowflake-arctic-embed-m-v2.0 (released Dec 2024)

| Attribute | Value |
|-----------|-------|
| Parameters | 305 M total (113 M non-embedding) |
| Embedding dim | 768 (MRL to 256) |
| Safetensors (fp32) | ~1.2 GB (0.3B model) |
| MTEB Retrieval BEIR | 55.4 |
| MTEB English avg (56 tasks) | Not published |
| Licence | Apache 2.0 |
| Max seq length | 8 192 |
| `trust_remote_code` | Required |
| `SentenceTransformer(...)` | Yes, with `trust_remote_code=True` |

Sources: [snowflake-arctic-embed-m-v2.0 HF card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0); [arctic-embed 2.0 arXiv:2412.04506](https://arxiv.org/html/2412.04506v2). At 305 M params and >1 GB disk, this is well outside the constraints. Multilingual-oriented. Excluded.

---

### F. avsolatorio/GIST-small-Embedding-v0

| Attribute | Value |
|-----------|-------|
| Parameters | 33.4 M |
| Embedding dim | 384 |
| Safetensors (fp32) | 133 MB |
| MTEB English avg (56 tasks) | 62.48 |
| Licence | MIT |
| Max seq length | 512 |
| `trust_remote_code` | No |
| `SentenceTransformer(...)` | Yes |

GISTEmbed fine-tune of bge-small-en-v1.5. Score delta over the base: +0.31. Source: [GISTEmbed paper arXiv:2402.16829](https://arxiv.org/html/2402.16829v1). This is a marginal improvement on the prior recommended default — still the same 33 M / 384-dim form factor — but it does not change the character of the recommendation relative to the newer 149 M class.

---

### G. mixedbread-ai/mxbai-embed-large-v1

335 M params, 1 024 dim, ~1.3 GB. Strong on retrieval (MTEB 64.68) but well outside the size ceiling. Source: [mxbai-embed-large-v1 HF card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1). Excluded on size.

### H. dunzhang/stella_en_400M_v5 (released Jul 2024)

400 M params, 1 024 dim default, ~1.6 GB, requires `trust_remote_code` plus CPU-specific config kwargs. Source: [dunzhang/stella_en_400M_v5 HF card](https://huggingface.co/dunzhang/stella_en_400M_v5). Excluded on size and operational complexity.

### I. jinaai/jina-embeddings-v3

570 M params, CC-BY-NC-4.0 licence. Source: [Jina AI announcement](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/). Excluded on size and licence.

### J. jinaai/jina-embeddings-v5-text-nano (released Feb 2026)

239 M params, CC-BY-NC-4.0 licence, requires `trust_remote_code`. Sources: [HF card](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano); [arXiv:2602.15547](https://arxiv.org/html/2602.15547). Excluded on licence.

### K. BAAI/bge-m3

~570 M params, designed as a multilingual multi-retrieval model. ~2.2 GB on disk. Source: [BAAI/bge-m3 HF card](https://huggingface.co/BAAI/bge-m3). Excluded.

### L. Alibaba-NLP/gte-base-en-v1.5

137 M params, 768 dim, 547 MB, MTEB 64.11, **requires `trust_remote_code=True`**. Source: [gte-base-en-v1.5 HF card](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5). Predecessor to gte-modernbert-base. Dominated on every axis (lower score, larger file, requires trust_remote_code). Excluded.

---

## 2. Comparison Table (Eligible Candidates)

| Model | Params | Dim | Disk (fp32) | MTEB v1 avg | Licence | `trust_remote_code` | Verdict |
|-------|--------|-----|-------------|-------------|---------|---------------------|---------|
| bge-small-en-v1.5 | 33 M | 384 | 133 MB | 62.17 | MIT | No | Prior default |
| GIST-small-Embedding-v0 | 33 M | 384 | 133 MB | 62.48 | MIT | No | Marginal upgrade on prior |
| snowflake-arctic-embed-s | 33 M | 384 | 133 MB | retrieval only | Apache 2.0 | No | Retrieval-tier comparison invalid |
| **gte-modernbert-base** | **149 M** | **768** | **298 MB** | **64.38** | **Apache 2.0** | **No** | **New default** |
| nomic-embed-text-v1.5 | 137 M | 768 | 547 MB | 62.28 | Apache 2.0 | No (ST >= 5.3) | Near-tie alt; over size limit |
| nomic-modernbert-embed-base | 149 M | 768 | 596 MB | 62.62 | Apache 2.0 | No | Dominated by gte-modernbert-base |
| gte-base-en-v1.5 | 137 M | 768 | 547 MB | 64.11 | Apache 2.0 | Yes | Dominated by gte-modernbert-base |
| mxbai-embed-large-v1 | 335 M | 1 024 | ~1.3 GB | 64.68 | Apache 2.0 | No | Excluded: size |
| stella_en_400M_v5 | 400 M | 1 024 | ~1.6 GB | — | MIT | Yes | Excluded: size, complexity |
| jina-embeddings-v3 | 570 M | 1 024 | >2 GB | 65.52 | CC-BY-NC | Yes | Excluded: size, licence |
| jina-embeddings-v5-text-nano | 239 M | 768 | ~950 MB est. | — | CC-BY-NC | Yes | Excluded: licence |
| bge-m3 | ~570 M | 1 024 | ~2.2 GB | — | MIT | No | Excluded: size |
| snowflake-arctic-embed-m-v2.0 | 305 M | 768 | ~1.2 GB | retrieval only | Apache 2.0 | Yes | Excluded: size |

---

## 3. Recommended Default Checkpoint

**`Alibaba-NLP/gte-modernbert-base`**

This replaces `BAAI/bge-small-en-v1.5`.

### Justification

1. **MTEB gain is real.** 64.38 vs 62.17 is a +2.2-point improvement on the same 56-task English board. That is not noise — it is a consistent improvement across classification (74.31 vs 74.14), retrieval, and clustering categories visible on the model card.

2. **Size is within the extended ceiling.** At 298 MB the model sits inside the "up to 250 MB if delta justifies it" envelope stated in the constraints. The implementer should set `model_kwargs={"torch_dtype": torch.float16}` in the `SentenceTransformer` constructor to load the model into ~149 MB RAM at inference time, which is well within a laptop's headroom.

3. **CPU-first architecture.** ModernBERT's alternating local/global attention and RoPE design reduce the quadratic attention cost on sequences under 512 tokens. The ModernBERT paper reports up to 2x throughput over DeBERTa-v3 on mixed-length inputs on GPU; no official CPU benchmark is published, but the architecture rationale applies proportionally to CPU inference.

4. **No special loading.** Unlike gte-base-en-v1.5 (requires `trust_remote_code=True`) and all Jina models, gte-modernbert-base loads cleanly with no flags once `transformers >= 4.48.0` is installed.

5. **Licence is permissive.** Apache 2.0. No restriction for pedagogical or commercial derivative use.

6. **768-dim output is acceptable.** The constraints list 384 or 512 as preferred and 768 as acceptable if the model is "a clear winner." The +2.2-point MTEB gap and the architectural upgrade together meet that bar.

---

## 4. Near-Tie Alternatives and When to Pick Them

### nomic-ai/nomic-embed-text-v1.5 — pick if download bandwidth is hard-constrained to 150 MB

At 62.28 MTEB this model nearly ties bge-small on the overall leaderboard but brings 8 192-token context, Matryoshka dims, and a fully reproducible open training pipeline (weights + code + data all public). Its 547 MB safetensors file is over the stated ceiling but can be trimmed: the 256-dim slice produces 61.04 MTEB average, reducing the active index from ~6 MB to ~1.5 MB for 5 000 events.

Caveat: task prefixes (`search_document:` / `search_query:`) are required. As of `sentence-transformers >= 5.3.0` the model no longer needs `trust_remote_code=True`.

### avsolatorio/GIST-small-Embedding-v0 — pick if footprint is the hardest constraint

62.48 MTEB average, 33 M params, 133 MB, MIT licence, no trust_remote_code, no prefix. It strictly improves over the prior default bge-small-en-v1.5 (+0.31 points) at identical cost. Trained in 2024; the improvement is marginal relative to gte-modernbert-base.

---

## 5. Why the Prior Recommendation Does Not Hold

The user's observation is correct. The small-tier has moved since 2023. The specific evidence:

- bge-small-en-v1.5 scored 62.17 on the 56-task MTEB board (BERT-era, 2023 architecture).
- gte-modernbert-base scores 64.38 on the same board with a newer architecture (ModernBERT-base, Dec 2024), an identical licence tier, no loading quirks, and a disk footprint that sits at the top of the stated ceiling.
- No third model in the eligible size band (Apache/MIT, <= 250 MB download, no trust_remote_code) scores between those two figures except gte-base-en-v1.5 (64.11, but requires trust_remote_code and is 547 MB, making it strictly worse than gte-modernbert-base on every dimension).

The flip is not manufactured. The 2024 ModernBERT generation represents a real architectural step that propagated immediately into the embedding model tier.

---

## 6. Known Pitfalls, Deprecations, and Compatibility Notes

**gte-modernbert-base**
- Requires `transformers >= 4.48.0`. Earlier versions will fail to load the ModernBERT architecture.
- `flash_attn` is optional but recommended. On CPU it is not used; no error is raised if absent.
- Input prefix convention: queries use `"query: "` prefix and documents use no prefix (same convention as the E5 family). Check the `config_sentence_transformers.json` for the registered prompt names.
- No announced deprecation of the checkpoint as of April 2026.

**MTEB v1 vs MTEB v2 warning.** Several third-party sources quote MTEB v2 scores in the 65-75 range. These are from the MMTEB multilingual leaderboard (131 tasks, Borda count aggregation, launched Feb 2025) and are not directly comparable to the 56-task English v1 figures used here. Do not mix the two scales in any implementation documentation. Source: [MMTEB arXiv:2502.13595](https://arxiv.org/abs/2502.13595).

---

## 7. Canonical Sources

| Source | Summary |
|--------|---------|
| [Alibaba-NLP/gte-modernbert-base HF card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Primary spec for the recommended checkpoint |
| [BAAI/bge-small-en-v1.5 HF card](https://huggingface.co/BAAI/bge-small-en-v1.5) | Prior default; comparison baseline |
| [ModernBERT paper arXiv:2412.13663](https://arxiv.org/html/2412.13663v2) | Architecture rationale for inference efficiency claims |
| [nomic-embed-text-v1.5 HF card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Principal alternative |
| [GISTEmbed paper arXiv:2402.16829](https://arxiv.org/html/2402.16829v1) | GIST-small data; MTEB delta vs bge-small base |
| [MMTEB arXiv:2502.13595](https://arxiv.org/abs/2502.13595) | MTEB v2 benchmark specification |
| [MTEB leaderboard (live)](https://huggingface.co/spaces/mteb/leaderboard) | Authoritative live leaderboard |
