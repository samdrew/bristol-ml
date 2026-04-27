# `bristol_ml.embeddings` — module guide

This module is the **semantic-search layer**: a typed boundary
(`Embedder` → vector; `VectorIndex` → nearest neighbours) plus two
interchangeable backends for each, content-addressed cache, and a
UMAP projection wrapper for the notebook. Stage 15 introduced the
layer; a future RAG-style stage and Stage 16's optional cross-stage
join are the downstream consumers.

The architectural narrative — why the layer exists, the stub-first
discipline, the Protocol contract, the cache-invalidation rationale,
the demo moment — lives in
[`docs/architecture/layers/embeddings.md`](../../../docs/architecture/layers/embeddings.md).
The file you are reading is the **module guide**: schema reference,
the model pre-warm ritual, the cache layout on disk, and how to add
a third backend.

## Public surface

```python
from bristol_ml.embeddings import (
    Embedder,                  # Protocol — 2 methods + 2 properties
    VectorIndex,               # Protocol — 4 methods + 1 property
    NearestNeighbour,          # NamedTuple(id: str, score: float)
    EmbeddingCache,            # dataclass with classmethod load_or_build
    EmbeddingCacheMetadata,    # provenance fields
    build_embedder,            # EmbeddingConfig -> Embedder
    build_index,               # EmbeddingConfig, dim -> VectorIndex
    embed_corpus,              # config, DataFrame -> (VectorIndex, EmbeddingCache)
    synthesise_embeddable_text,
    STUB_ENV_VAR,              # "BRISTOL_ML_EMBEDDING_STUB"
    MODEL_PATH_ENV_VAR,        # "BRISTOL_ML_EMBEDDING_MODEL_PATH"
)
```

The `__init__.py` re-exports only the boundary types + factories.
Concrete implementations live in private modules (`_embedder.py`,
`_index.py`, `_cache.py`, `_text.py`, `_factory.py`,
`_projection.py`) so callers (Stage 16, future RAG) can import the
schema without dragging `sentence-transformers` / `torch` into their
import graph (ADR-0008). The factory deferred-imports the heavy
backend only when the live discriminator is hit.

## `Embedder` Protocol

`runtime_checkable`; two methods + two properties; no other public
attributes:

```python
@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...
```

Adding a method (e.g. `embed_async`) is a Stage-16 / future-RAG
contract change — discuss before doing it; supersede ADR-0008 when
making it. The two-method shape is the AC-5 cap.

**Output shape contract.** `embed` returns 1-D `np.ndarray`,
`(dim,)`, `float32`, L2-normalised. `embed_batch` returns 2-D
`np.ndarray`, `(len(texts), dim)`, `float32`, L2-normalised
row-wise. Pre-normalisation simplifies cosine query to a matmul.

**Prefix asymmetry (E5 / GTE convention).** `embed` is the *query*
path — for `SentenceTransformerEmbedder` it prepends `"query: "` to
the input. `embed_batch` is the *document* path — no prefix. The
`StubEmbedder` mirrors the asymmetry by hashing the prefixed-vs-
unprefixed text. A regression that routes queries through
`embed_batch` (or vice-versa) surfaces in
`tests/unit/embeddings/test_stub_embedder.py` rather than as silent
retrieval-quality degradation.

## `VectorIndex` Protocol

`runtime_checkable`; four methods + one property:

```python
@runtime_checkable
class VectorIndex(Protocol):
    @property
    def dim(self) -> int: ...
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, k: int) -> list[NearestNeighbour]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> VectorIndex: ...
```

**Empty-index discipline.** `query` on an empty index returns `[]`,
never raises. Mirrors Stage 14's `StubExtractor` miss-path
discipline.

**`k`-clipping.** `k` larger than the index size is silently clipped
to the index size — no exception, no zero-padding.

**`add` does not de-duplicate.** Stage 15's only producer of ids is
`embed_corpus`, which builds them from `(mrid, revision_number)` —
unique by Stage 13's `OUTPUT_SCHEMA`. A future caller adding
duplicate ids is the caller's responsibility to police.

## `EmbeddingCache` — content-addressed cache

```python
@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    embedded_at_utc: datetime
    corpus_sha256: str            # full 64-char hex
    corpus_sha256_prefix: str     # 12-char prefix (log-line ergonomics)
    model_id: str
    dim: int

@dataclass(frozen=True)
class EmbeddingCache:
    ids: list[str]
    vectors: np.ndarray           # (n, dim) float32, L2-normalised
    metadata: EmbeddingCacheMetadata

    @classmethod
    def load_or_build(
        cls,
        path: Path,
        ids: list[str],
        texts: list[str],
        embedder: Embedder,
        force_rebuild: bool = False,
    ) -> EmbeddingCache: ...
```

`load_or_build` is the cache's only public entry. It:

1. Tries to load the parquet at `path`.
2. If the file is absent, or `force_rebuild=True`, or the recomputed
   `corpus_sha256` differs from the stored value, or the supplied
   `embedder.model_id` differs from the stored value — rebuilds via
   `embedder.embed_batch(texts)` and atomically writes the parquet.
3. Otherwise returns the loaded cache as a no-op.

A stale-cache rebuild emits a `loguru` WARNING naming the offending
field (`corpus_sha256` or `model_id`). A successful no-op emits an
INFO record. Tests assert the WARNING fires (NFR-8 — convention,
not exact-message-text matching).

## Cache file layout

A single Parquet file per `(model_id_sanitised)` at
`data/embeddings/<sanitised>.parquet` (gitignored). Schema:

| Column | Type | Notes |
|--------|------|-------|
| `id` | `string` | The caller-supplied id (Stage 15: `"<mrid>::<revision_number>"`). |
| `vector` | `list<float32>` | Length = `dim` for every row; L2-normalised. |

`custom_metadata` (Parquet table-level):

| Key | Value |
|-----|-------|
| `embedded_at_utc` | UTC ISO-8601 string. |
| `corpus_sha256` | Full 64-char hex SHA-256 of `\n`-joined texts. |
| `model_id` | The configured embedder `model_id`. |
| `dim` | The vector dimensionality (decoded as int). |

The sanitiser replaces non-`[A-Za-z0-9._-]` with `_`; e.g.
`Alibaba-NLP/gte-modernbert-base` becomes
`Alibaba-NLP_gte-modernbert-base.parquet`. The cache file is written
via `np.savez_compressed` to a `.tmp` sibling then `os.replace`-d
into place — a crash mid-write leaves the previous-good cache
intact.

## Quick recipes

### Run the stub against a synthetic corpus (offline)

```python
import pandas as pd
from bristol_ml.embeddings import build_embedder, build_index, NearestNeighbour
from conf._schemas import EmbeddingConfig

cfg = EmbeddingConfig()  # all defaults — type='stub', vector_backend='numpy'
embedder = build_embedder(cfg)
index = build_index(cfg, dim=embedder.dim)
docs = ["planned outage on Hartlepool", "wind-farm restriction at Burbo"]
index.add(["A", "B"], embedder.embed_batch(docs))

print(index.query(embedder.embed("nuclear refuelling outage"), k=2))
# [NearestNeighbour(id='A', score=0.0734), NearestNeighbour(id='B', score=0.0211)]
```

### Run the standalone CLI

```bash
uv run python -m bristol_ml.embeddings                         # stub by default
uv run python -m bristol_ml.embeddings embedding.type=sentence_transformers
uv run python -m bristol_ml.embeddings embedding.force_rebuild=true
```

The CLI prints the active config + a one-shot top-3 query against a
hand-picked 5-row synthetic corpus, then exits 0. Output is
byte-deterministic in stub mode.

### Build / load a real REMIT corpus cache

```python
import pandas as pd
from bristol_ml.config import load_config
from bristol_ml.embeddings import embed_corpus

cfg = load_config(overrides=["+embedding=default"])
corpus = pd.read_parquet("tests/fixtures/embedding/tiny_corpus.parquet")
index, cache = embed_corpus(config=cfg.embedding, corpus=corpus)

print(f"corpus_sha256={cache.metadata.corpus_sha256_prefix}")
print(f"model_id     ={cache.metadata.model_id}")
print(f"embedded_at  ={cache.metadata.embedded_at_utc.isoformat()}")
print(f"n            ={len(cache.ids)}, dim={cache.metadata.dim}")
```

A second call with the same corpus + config is the no-op cache-hit
path (~milliseconds). Mutate one row in `corpus` and the next call
rebuilds with a WARNING naming `corpus_sha256`.

## Pre-warming the live model (`Alibaba-NLP/gte-modernbert-base`)

The live default checkpoint is **`Alibaba-NLP/gte-modernbert-base`**
(149 M params, 768-dim, ~298 MB safetensors fp32, Apache 2.0). First-
run download is ~298 MB; the README's "Before the meetup" section
names the one-line pre-warm:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"
```

The model lands in HuggingFace's cache (`~/.cache/huggingface/hub/`
by default; `HF_HOME` overrides). A subsequent live-path embedder
load is ~5 s of cold-start latency without a download.

CI never downloads — `BRISTOL_ML_EMBEDDING_STUB=1` plus
`HF_HUB_OFFLINE=1` in `tests/conftest.py` keep the live path off the
CI runner entirely. The integration test
`tests/integration/test_notebook_15.py` sets both env-vars in its
`subprocess.run` call.

## Adding a third backend

A future implementation (e.g. `OpenAIEmbedder` for `text-embedding-3-large`,
or `FaissIndex` for a multi-million-row index) is a one-file plus
one-dispatch-branch change.

### For a new `Embedder`

1. Add a literal value to `EmbeddingConfig.type`:
   `Literal["stub", "sentence_transformers", "openai"]`.
2. Add a class to `_embedder.py` that satisfies the `Embedder`
   Protocol structurally (no inheritance — `runtime_checkable` makes
   conformance a method-shape question).
3. Add a dispatch branch in `build_embedder` (`_factory.py`). Defer
   the heavy import to inside the branch so the boundary stays
   light.
4. Add unit tests under `tests/unit/embeddings/`. The Protocol
   conformance test
   (`test_protocol.py::test_embedder_protocol_has_two_methods`) is
   shared.

### For a new `VectorIndex`

1. Add a literal value to `EmbeddingConfig.vector_backend`:
   `Literal["numpy", "stub", "faiss"]`.
2. Add a class to `_index.py` that satisfies the `VectorIndex`
   Protocol structurally.
3. Add a dispatch branch in `build_index` (`_factory.py`).
4. Add a `save` / `load` round-trip test under `tests/unit/embeddings/`.

The cache layer (`_cache.py`) is **unchanged** in either case — the
cache stores the embeddings, not the index. A new `Embedder` may
require a fresh cache file if its `model_id` differs from the
existing one (the `corpus_sha256 + model_id` invalidation key
handles this automatically — the new `model_id` triggers an
automatic rebuild on first run).

If the new interface needs a *behaviour* outside the four-method
shape (e.g. async `embed_async`, or `query_with_metadata` returning
richer payloads), it is no longer a structural extension and ADR-0008
should be superseded.

## Notebook

`notebooks/15_embedding_index.ipynb` is the demo surface — six code
cells plus title-markdown leading and discussion-markdown trailing
(8 cells total):

1. **Bootstrap** — repo-root walk + `os.environ["BRISTOL_ML_EMBEDDING_STUB"]="1"`
   + `+embedding=default` Hydra compose. Marker: `T5 Cell 1`.
2. **Gold-set load** — read `tests/fixtures/embedding/tiny_corpus.parquet`;
   print row count + NULL `message_description` ratio. Marker: `T5 Cell 2`.
3. **Build / load index** — `embed_corpus(config=..., corpus=df)`
   into a per-run `tempfile.mkdtemp()` cache; print provenance
   fields. Marker: `T5 Cell 3`.
4. **Top-k query** — pick a fixed gold event id; print 10 nearest
   neighbours with cosine scores. Marker: `T5 Cell 4`.
5. **2D projection** — `_projection.umap_project(cache.vectors, ...)`;
   matplotlib scatter coloured by `fuel_type`. Marker: `T5 Cell 5`.
6. **Optional cross-stage join (A5)** — when
   `data/features/remit_extracted.parquet` is present, join Stage 14
   `event_type` / `fuel_type` onto the top-k. When absent, print a
   single-line skip banner. Marker: `T5 Cell 6`.

Regenerate via the three-step ritual (mirrors Stage 14):

```bash
uv run python scripts/_build_notebook_15.py
BRISTOL_ML_EMBEDDING_STUB=1 \
uv run jupyter nbconvert --execute --to notebook --inplace \
    notebooks/15_embedding_index.ipynb
uv run ruff format notebooks/15_embedding_index.ipynb
```

## Tests

Located alongside the production code:

- `tests/unit/embeddings/test_protocol.py` — Protocol shape
  (`isinstance(_, Embedder)` / `isinstance(_, VectorIndex)`),
  method-count pinning, boundary-import asserts no
  `sentence_transformers` in the import graph.
- `tests/unit/embeddings/test_stub_embedder.py` — env-var triple-gate,
  determinism (same text → same vector), prefix asymmetry, dim
  contract.
- `tests/unit/embeddings/test_st_embedder.py` — live embedder
  conformance against pre-warmed local cache; **xfail / skip when
  the cache is absent** (CI default).
- `tests/unit/embeddings/test_index_query.py` — top-k sorted, cosine
  in `[-1, 1]`, save/load round-trip, empty-index returns `[]`.
- `tests/unit/embeddings/test_cache_invalidation.py` — content-hash
  change → rebuild + WARNING; model-id change → rebuild + WARNING;
  fresh cache → no rebuild.
- `tests/unit/embeddings/test_text_synthesis.py` — NULL
  `message_description` falls back to structured fields; the
  documented sentinel fires when every field is also NULL.
- `tests/unit/embeddings/test_module_runs_standalone.py` —
  `python -m bristol_ml.embeddings` exits 0 under stub mode +
  prints expected lines.
- `tests/integration/test_notebook_15.py` — `nbconvert --execute`
  on the demo notebook under `BRISTOL_ML_EMBEDDING_STUB=1` +
  `HF_HUB_OFFLINE=1`.

Run the embeddings-only suite:

```bash
uv run pytest tests/unit/embeddings/ \
              tests/integration/test_notebook_15.py -v
```

## Cross-references

- Layer doc — [`docs/architecture/layers/embeddings.md`](../../../docs/architecture/layers/embeddings.md).
- Stage 15 retro — [`docs/lld/stages/15-embedding-index.md`](../../../docs/lld/stages/15-embedding-index.md).
- Intent — [`docs/intent/15-embedding-index.md`](../../../docs/intent/15-embedding-index.md).
- Plan — [`docs/plans/completed/15-embedding-index.md`](../../../docs/plans/completed/15-embedding-index.md).
- ADR-0008 — Embedder + VectorIndex `runtime_checkable` Protocols.
- ADR-0003 — the precedent `Model` Protocol over ABC.
- Sibling boundary — `src/bristol_ml/llm/CLAUDE.md` (parallel
  thread on the same upstream data; cross-stage join cell consumes
  its parquet output).
- README §"Before the meetup" — operator-facing model pre-warm.
