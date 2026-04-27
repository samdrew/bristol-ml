# Embeddings — layer architecture

- **Status:** Provisional — first realised by Stage 15 (embedding
  index over REMIT: `Embedder` + `VectorIndex` Protocols, two
  implementations of each, content-addressed cache, UMAP projection,
  optional cross-stage join with Stage 14 features).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities);
  [`docs/intent/15-embedding-index.md`](../../intent/15-embedding-index.md)
  (Stage 15 intent — the demo moment + offline-default discipline).
- **Concrete instances:** [Stage 15 retro](../../lld/stages/15-embedding-index.md) — corpus-size benchmark, cache-hit timings, deviations.
- **Related principles:** §2.1.1 (standalone module), §2.1.2 (typed
  narrow interfaces), §2.1.3 (stub-first for expensive / flaky external
  dependencies — the load-bearing principle for this layer), §2.1.4
  (config outside code), §2.1.5 (idempotent ingestion — the cache
  generalises this to "idempotent derivation"), §2.1.6 (provenance —
  every cache file carries `corpus_sha256` + `model_id` +
  `embedded_at_utc`), §2.1.7 (tests at boundaries), §2.1.8 (thin
  notebooks).
- **Upstream layer:** [Ingestion](ingestion.md) — Stage 13's REMIT
  bi-temporal store. The embedder consumes a DataFrame matching
  `OUTPUT_SCHEMA`; it does not write back to the parquet.
- **Downstream consumers:** Stage 16 (feature-table join — the
  notebook may consume nearest-neighbour features); a future RAG-style
  stage that swaps `NumpyIndex` for FAISS / Qdrant without touching
  the embedder side.
- **Sibling layer:** [LLM](llm.md). Stage 14 + Stage 15 are parallel
  threads on the same upstream data: the `Extractor` extracts
  *structured* features from REMIT free-text; the `Embedder` produces
  *semantic* features from the same rows. The notebook's optional
  cross-stage cell (plan A5) joins them at the demo surface.

---

## Why this layer exists

The embeddings layer is the project's **content-addressed-cache
boundary** and its first **semantic-search surface**. REMIT messages
carry free-text descriptions whose semantic neighbourhoods (events
"like this nuclear outage") downstream stages can exploit for
retrieval, weak labels, or RAG-style joins. A sentence-embedding
model is the cheapest mechanism that produces those neighbourhoods
without per-query LLM cost — but two properties of an embedding
model make it incompatible with naïve once-per-call use:

1. **Loading the model is expensive.** A 149 MB safetensors file
   plus the `transformers` import path is ~5–10 s of cold-start
   latency before a single text is encoded. Embedding a few-thousand-
   event corpus is then a few minutes of CPU.
2. **The result is invariant under repeated calls.** Identical text
   plus identical model produces identical vectors — cosine
   similarity rounds to bit-equality on fp32 hardware.

Stage 15 draws a typed boundary that hides both of those properties
from downstream callers. The two-method `Embedder` Protocol takes a
text and returns a vector; the four-method `VectorIndex` Protocol
serves nearest-neighbour queries. Whether the vectors came from a
hand-deterministic stub, the live `Alibaba-NLP/gte-modernbert-base`
checkpoint, or a future quantised model is invisible to the caller.
The pedagogical payoff is the **stub-first triple gate**: every CI
run, every notebook bootstrap, and every unit test runs the stub
path with no model download and no network call (intent AC-4). The
architectural payoff is the **content-hash cache**: a corpus rebuild
is automatic-and-loud when the corpus or the model changes, and a
no-op when neither changed (intent AC-3 / AC-8).

The load-bearing design constraint is intent line 33: *"the
embedding model has no external API dependency — it runs locally"*,
plus line 34: *"the index's interface is small enough that swapping
the vector store is a mechanical change"*. Every decision in this
layer flows from those two.

---

## Stub-first discipline (offline-by-default — AC-4)

The Stage 15 embedder is **triple-gated** so that no path through
the project can fire a live model download by accident:

1. **Config discriminator (plan §1 D8).** `EmbeddingConfig.type:
   Literal["stub", "sentence_transformers"]` defaults to `stub` in
   `conf/embedding/default.yaml`. A YAML override
   (`embedding.type=sentence_transformers`) is required to opt in.
2. **Env-var override (plan §1 D8).** `BRISTOL_ML_EMBEDDING_STUB=1`
   forces `StubEmbedder` regardless of `config.type`. The notebook
   sets it explicitly in Cell 1 (before any embeddings import); CI
   sets it in `tests/integration/test_notebook_15.py` and
   `tests/conftest.py`. The override is the load-bearing fallback
   when the YAML is misconfigured — a misconfigured live `type` plus
   a missing local model cache would otherwise raise at init; the
   env-var short-circuits before that.
3. **Model-availability check at init (plan §1 D8).** The live path
   is gated by `HF_HUB_OFFLINE=1` (set in `tests/conftest.py` and
   the notebook subprocess env). If the configured `model_id` is not
   present in the local HuggingFace cache, `SentenceTransformerEmbedder`
   raises `RuntimeError` at construction time naming the model id
   *and* the `BRISTOL_ML_EMBEDDING_STUB=1` escape hatch — an operator
   hitting the error sees the offline path immediately.

The dispatch happens in a single function
(`bristol_ml.embeddings.build_embedder`); the env-var check fires
*before* the YAML branch, which fires before the model-availability
branch. The Stage 14 `BRISTOL_ML_LLM_STUB` precedent set the env-var
naming convention; Stage 15 reuses it 1:1 with the layer prefix
swap. A future authenticated-or-flaky-dependency layer copies the
same pattern.

---

## Public surface

```python
# src/bristol_ml/embeddings/__init__.py
from bristol_ml.embeddings import (
    Embedder,                  # Protocol — 2 methods (plan §1 D2)
    VectorIndex,               # Protocol — 4 methods (plan §1 D3)
    NearestNeighbour,          # NamedTuple(id: str, score: float)
    EmbeddingCache,            # dataclass: ids, vectors, metadata
    EmbeddingCacheMetadata,    # dataclass: corpus_sha256, model_id, ...
    build_embedder,            # factory: EmbeddingConfig -> Embedder
    build_index,               # factory: EmbeddingConfig, dim -> VectorIndex
    embed_corpus,              # glue: cache + embedder + index
    synthesise_embeddable_text,
    STUB_ENV_VAR,              # "BRISTOL_ML_EMBEDDING_STUB"
    MODEL_PATH_ENV_VAR,        # "BRISTOL_ML_EMBEDDING_MODEL_PATH"
)

# src/bristol_ml/embeddings/__main__.py
# python -m bristol_ml.embeddings [overrides...]
```

### `Embedder` Protocol (in `bristol_ml/embeddings/_protocols.py`, re-exported from `__init__.py`)

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

Two methods + two properties is the AC-5 cap. ADR-0008 records the
Protocol-over-ABC choice; ADR-0003 set the precedent for `Model`,
Stage 14 reused it for `Extractor`, Stage 15 makes it the project's
house style for swappable interfaces. Adding a third method (e.g.
`embed_async` for an async serving path) is a plan-edit conversation
and warrants superseding ADR-0008.

**Output shape contract.** `embed` returns a 1-D `np.ndarray` of
shape `(dim,)` dtype `float32`, L2-normalised. `embed_batch` returns
a 2-D `np.ndarray` of shape `(len(texts), dim)` dtype `float32`,
L2-normalised row-wise. Pre-normalisation at the embedder simplifies
`VectorIndex.query` to a plain matmul — there is no normalisation
hop inside the index.

**Prefix asymmetry (E5 / GTE convention).** The plan-bound default
checkpoint `Alibaba-NLP/gte-modernbert-base` follows the E5 family's
asymmetric prefix: queries take `"query: "`, documents take no
prefix. `embed` is the *query* path (prepends `"query: "`);
`embed_batch` is the *document* path (no prefix). The
`StubEmbedder` mirrors the asymmetry by hashing the
prefixed-vs-unprefixed text — a regression where a downstream caller
accidentally routes queries through `embed_batch` (or vice-versa)
surfaces in a unit test rather than as silent retrieval-quality
degradation.

### `VectorIndex` Protocol

```python
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

Four methods + one property is the AC-5 cap. The Protocol does *not*
own the cache — `save` / `load` persist the in-memory index for
fast restart, while the `EmbeddingCache` (next section) carries the
provenance metadata for invalidation. Keeping the two concerns
distinct is what makes the swap mechanical: a future `FaissIndex`
needs to satisfy these four methods only; the cache layer is
unchanged.

**Empty-index discipline (plan §1 D16).** `query` on an empty index
returns `[]`, never raises. Mirrors Stage 14's `StubExtractor`
miss-path discipline — callers receive a typed-empty result they
can branch on without `try`/`except`.

**`k` larger than the index size** is silently clipped to the index
size (no exception, no zero-padding).

### `EmbeddingCache` + `EmbeddingCacheMetadata`

```python
@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    embedded_at_utc: datetime         # UTC-aware
    corpus_sha256: str                # full 64-char hex
    corpus_sha256_prefix: str         # 12-char prefix for log lines
    model_id: str
    dim: int

@dataclass(frozen=True)
class EmbeddingCache:
    ids: list[str]
    vectors: np.ndarray               # (n, dim) float32, L2-normalised
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

`EmbeddingCache` is the project's **first content-addressed cache**.
The lessons it teaches generalise beyond embeddings: *corpus-hash +
model-id as the invalidation key, Parquet `custom_metadata` as the
provenance carrier, deterministic recompute on mismatch*. A future
cached derivation downstream of REMIT (or any other ingested corpus)
copies the same shape.

The metadata pair `(corpus_sha256, model_id)` is the cache key.
Either changing triggers an automatic rebuild with a `loguru`
WARNING naming the offending field. `force_rebuild=True` skips the
freshness check and rebuilds unconditionally (plan §1 D17, surfaced
as the `force_rebuild` YAML field, not a CLI flag, per A4).

---

## Content-addressed cache invalidation (NFR-2 + NFR-5 + AC-3 + AC-8)

The cache discipline is the load-bearing architectural lesson of
the layer. Three fields stamped into Parquet `custom_metadata`:

| Field | Source | Purpose |
|-------|--------|---------|
| `corpus_sha256` | `hashlib.sha256` over the deterministic serialisation of the embeddable-text column (one row per line, `\n`-joined) | Detects corpus changes — a row added, removed, or edited produces a different hash. |
| `model_id` | `Embedder.model_id` at build time | Detects backend swaps — replacing the live checkpoint with the stub (or vice-versa) invalidates the cache. |
| `embedded_at_utc` | `datetime.now(UTC)` at build time | Provenance only; not part of the cache key. Lets a notebook display "last embedded at …". |

A fresh load reads the parquet, recomputes the corpus hash from the
caller-supplied texts, and compares it against the stored hash *and*
the supplied `embedder.model_id` against the stored `model_id`.
Either mismatch → WARNING log + automatic rebuild. Both match →
no-op (read-only).

**Why a hash, not a timestamp or a row count?** Timestamps drift
silently when the upstream parquet is touched without content
changes (e.g. an `os.utime` from a backup tool); row counts collide
trivially when one row is replaced with another. A bytes hash is
the cheapest content-derived identity that survives both.

**Why store the full 64-char SHA-256 but expose a 12-char prefix?**
The full hash lets a future debugging session join the cache file
back onto a corpus archive that recorded the same hash; the 12-char
prefix is what the notebook + WARNING lines display, because a
132-column terminal renders 64 hex chars badly. Collisions over the
project lifetime are vanishing (12 hex chars = 48 bits = 1 in 281
trillion).

**Atomic-write idiom.** The cache file is written via
`np.savez_compressed` to a `.tmp` sibling then `os.replace`-d into
place — the parquet atomic-write idiom from
`bristol_ml.ingestion._common._atomic_write`. A crash mid-write
leaves the previous-good cache intact; a successful write is
visible atomically.

---

## NULL `message_description` synthesis (plan §1 D9)

The Stage 13 stream endpoint frequently leaves `message_description`
NULL on live responses (`src/bristol_ml/ingestion/remit.py:431-436`;
Stage 14 already addressed this). Stage 15 inherits the constraint
mechanically. `synthesise_embeddable_text(row)` returns:

- `row.message_description` if it is non-NULL and non-empty after
  strip; or
- a synthesised string from `event_type`, `cause`, `fuel_type`, and
  `affected_unit` joined as `"<event_type> <cause> <fuel_type>
  <affected_unit>"`; or
- a documented sentinel (`"REMIT event (details unavailable)"`) when
  every structured field is also NULL — extreme case the live data
  occasionally produces.

This is **mechanical reuse of Stage 14's OQ-B resolution**, not a
new decision. A row that the extractor handled by structural-field
synthesis is embedded from the same fields here, so a downstream
join (Stage 16, plan A5 cross-stage cell) sees a self-consistent
provenance story.

---

## Optional cross-stage join (plan A5)

The Stage 15 notebook's final code cell (`T5 Cell 6`) is the
cross-stage demo — when Stage 14's structured-extraction parquet is
present at the documented path, the cell joins the extracted
`event_type` / `fuel_type` columns onto the nearest-neighbour result
and re-prints the top-k with the structured columns alongside
cosine score. When Stage 14's output is *absent* (the default in
stub-mode CI), the cell prints a single-line banner and exits
cleanly:

```
Stage 14 output not found at <path> — skipping cross-stage join.
```

The skip-clean guard is one branch with a clear test (the CI codepath
under stub mode exercises the absent branch). The plan A5 disposition
overturned the Scope-Diff cut here for one-cell pedagogical reasons:
the notebook's join is what makes the demo arc coherent.

---

## Live path: `Alibaba-NLP/gte-modernbert-base` (plan §1 A2)

The plan-bound default checkpoint is **`Alibaba-NLP/gte-modernbert-base`**
(149 M params, 768-dim, MTEB-56 avg 64.38, ~298 MB safetensors fp32 /
~149 MB at fp16, Apache 2.0, no `trust_remote_code`, max_seq=8192,
requires `transformers >= 4.48.0`).

`SentenceTransformerEmbedder` loads the model with
`model_kwargs={"torch_dtype": torch.float16}` so the live RAM
footprint is ~149 MB instead of ~298 MB. The cache vector matrix
stays float32 for query stability — fp16 inference can introduce
minor numeric jitter vs fp32, which would surface as non-deterministic
top-k ordering when ties are close. Tests that assert exact cosine
values therefore run under `StubEmbedder`, which is fp32 throughout.

**Pre-warming the model cache.** First-run download is ~298 MB. The
README's "Before the meetup" section names the one-line pre-warm:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"
```

CI never downloads — `BRISTOL_ML_EMBEDDING_STUB=1` plus
`HF_HUB_OFFLINE=1` keep the live path off the CI runner entirely.

---

## 2D projection — UMAP (plan §1 A3)

The notebook's projection cell uses **UMAP** (`umap-learn`,
`random_state=42`, `n_components=2`, `n_jobs=1`, `metric="cosine"`),
not PCA. The Scope-Diff originally proposed cutting UMAP for PCA on
zero-extra-deps grounds; A3 overturned the cut because PCA's two
principal components do not preserve neighbourhood structure in
768-dim space, which undermines the demo moment ("the 2D projection
shows the corpus falling into visible clusters by event type and
fuel"). Dependency cost (~200 MB compiled `numba` + `llvmlite`)
accepted.

**Why `n_jobs=1`?** UMAP's `random_state` only delivers
coordinate-exact reproducibility when single-threaded — the
`umap-learn` docs are explicit on this. The wrapper sets `n_jobs=1`
so the notebook's projection scatter is byte-identical across re-runs
(R-3); a regression test on the small fixture corpus pins the
coordinates.

**Tiny-corpus guard.** The notebook computes
`n_neighbors_eff = max(2, min(15, n_samples - 1))` so the fixture's
8-row corpus does not trip UMAP's `n_neighbors >= n_samples` assertion.

---

## Internals

### `build_embedder` factory

```python
def build_embedder(config: EmbeddingConfig) -> Embedder:
```

Single dispatch point. Returns `StubEmbedder()` when
`BRISTOL_ML_EMBEDDING_STUB=1` is set in the environment (regardless
of YAML), when `config.type == "stub"`, or when `config.model_id` is
absent on a live config (defensive guard — Pydantic's discriminator
catches the typed case). Returns `SentenceTransformerEmbedder(...)`
when `config.type == "sentence_transformers"` and the env-var is not
set. The live import is **deferred to inside the function body** so
a stub-only import of `bristol_ml.embeddings` does not pay the
`sentence-transformers` + `torch` import cost — Stage 16 callers
holding only the `Embedder` Protocol type stay lightweight (AC-1
sub-criterion).

The optional `BRISTOL_ML_EMBEDDING_MODEL_PATH` env-var, when set,
replaces `config.model_id` for the live path — useful when a
developer wants to point at a side-loaded local model snapshot
without editing YAML.

### `build_index` factory

```python
def build_index(config: EmbeddingConfig, *, dim: int) -> VectorIndex:
```

Single dispatch point. Returns `NumpyIndex(dim=dim)` for the
production binding (`config.vector_backend == "numpy"`); returns
`StubIndex(dim=dim)` for unit-test paths. `dim` must come from the
constructed `Embedder.dim` — the index is dimensionally tied to the
embedder. A future `FaissIndex` is one new branch here; the
`VectorIndex` Protocol is unchanged.

### `embed_corpus` glue

```python
def embed_corpus(
    *,
    config: EmbeddingConfig,
    corpus: pd.DataFrame,
    id_columns: tuple[str, str] = ("mrid", "revision_number"),
) -> tuple[VectorIndex, EmbeddingCache]:
```

The function the Stage 15 notebook (T9) and the standalone module
(T8) both call. Glues:

1. `synthesise_embeddable_text` over each row (NULL-aware text
   coercion).
2. `build_embedder` (env-var-aware dispatch).
3. `EmbeddingCache.load_or_build` (content-addressed freshness check,
   rebuild on stale).
4. `build_index` + `VectorIndex.add` (populate the in-memory index
   with the cached vectors).

The id grain is the Stage 13 primary key `(mrid, revision_number)`;
the index id is the joined string `"<mrid>::<revision_number>"`.
Stage 13's `OUTPUT_SCHEMA` guarantees uniqueness of the pair, so the
joined string is unique per row — `VectorIndex.add` does not
de-duplicate.

### Module structure

```
src/bristol_ml/embeddings/
├── __init__.py      # Public boundary: re-exports from _protocols, _factory, _cache, _text
├── _protocols.py    # @runtime_checkable Protocols + NearestNeighbour NamedTuple
├── _embedder.py     # StubEmbedder + SentenceTransformerEmbedder
├── _index.py        # StubIndex + NumpyIndex (np.savez_compressed persistence)
├── _cache.py        # EmbeddingCache + EmbeddingCacheMetadata + Parquet I/O
├── _factory.py      # build_embedder + build_index + embed_corpus + env-var constants
├── _text.py         # synthesise_embeddable_text (NULL-aware row -> text)
├── _projection.py   # UMAP wrapper (random_state=42, n_jobs=1, metric="cosine")
├── __main__.py      # `python -m bristol_ml.embeddings` standalone CLI (A4)
└── CLAUDE.md        # module guide; pre-warm ritual; cache layout
```

The split between `__init__.py` (public re-exports) and the leading-
underscore submodules is the load-bearing mechanism for ADR-0008's
import-cost claim: Stage 16 imports `Embedder` + `VectorIndex` only;
nothing in the public re-exports drags `sentence-transformers` /
`torch` into the consumer's import graph. The
`test_boundary_import_does_not_pull_sentence_transformers` test pins
this — a future regression that pulls the heavy backend into
`__init__.py` is caught before merge.

---

## Standalone CLI

```bash
uv run python -m bristol_ml.embeddings --help
uv run python -m bristol_ml.embeddings                       # stub by default
uv run python -m bristol_ml.embeddings embedding.type=sentence_transformers
uv run python -m bristol_ml.embeddings embedding.force_rebuild=true
```

The CLI composes `+embedding=default` automatically (the `embedding`
group is *not* in `conf/config.yaml`'s defaults — parallel to `llm`
and `serving`); explicit Hydra overrides flow through. The CLI
prints the active config, builds a synthetic 5-row index from
hand-picked REMIT-flavoured texts, runs a one-shot top-3 query, and
exits 0. Output is byte-deterministic in stub mode so the smoke
test (`tests/unit/embeddings/test_module_runs_standalone.py`) can
pin expected lines.

A4 narrowed the CLI to the standalone-module entry only — no
`build` / `query` subcommands. The notebook plus the standalone
entry covers both demo paths; subcommands would have been two more
implementation surfaces and two more CLI tests for no marginal
value beyond the notebook (intent line 45: *"useful, not
mandatory"*).

DESIGN §2.1.1 (every module runs standalone) is enforced by
`tests/unit/embeddings/test_module_runs_standalone.py::test_module_runs_under_stub`.

---

## Config

```python
# conf/_schemas.py
class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["stub", "sentence_transformers"] = "stub"
    model_id: str | None = None
    cache_path: Path | None = None
    vector_backend: Literal["numpy", "stub"] = "numpy"
    default_top_k: int = Field(default=10, ge=1, le=1000)
    projection_type: Literal["umap"] = "umap"
    force_rebuild: bool = False
    fp16: bool = True

# AppConfig
embedding: EmbeddingConfig | None = None
```

```yaml
# conf/embedding/default.yaml
# @package embedding
type: stub
model_id: Alibaba-NLP/gte-modernbert-base
cache_path: null               # null = derive from model_id at embed_corpus call time
vector_backend: numpy
default_top_k: 10
projection_type: umap
force_rebuild: false
fp16: true
```

The `embedding` group is *not* listed in `conf/config.yaml`'s
defaults — entry points (the embeddings CLI, the Stage 15 notebook)
compose it explicitly via `+embedding=default`. The `None` default
on `AppConfig.embedding` keeps every prior stage's CLI / config-smoke
test unaffected (the train pipeline never resolves `cfg.embedding`).

`cache_path: null` means *"derive from `model_id` at `embed_corpus`
call time"*. Resolution happens in the factory rather than the
schema because Pydantic models are frozen and the sanitised
filename depends on `model_id` — `data/embeddings/<model_id_sanitised>.parquet`,
where the sanitiser replaces non-filesystem-safe characters with
`_` (so `Alibaba-NLP/gte-modernbert-base` becomes
`Alibaba-NLP_gte-modernbert-base.parquet`). The `data/embeddings/`
directory is gitignored.

---

## Cross-references

- [`src/bristol_ml/embeddings/CLAUDE.md`](../../../src/bristol_ml/embeddings/CLAUDE.md) — concrete module guide; pre-warm ritual; cache layout; how to add a third backend.
- [Stage 15 retro](../../lld/stages/15-embedding-index.md) — observed corpus-size benchmark, cache-hit timings, deviations.
- [`docs/intent/15-embedding-index.md`](../../intent/15-embedding-index.md) — the contract (5 ACs + 6 points for consideration).
- [`docs/plans/completed/15-embedding-index.md`](../../plans/completed/15-embedding-index.md) — Stage 15 plan including the 20-decision table + Ctrl+G resolution log.
- [`docs/lld/research/15-embedding-index-scope-diff.md`](../../lld/research/15-embedding-index-scope-diff.md) — `@minimalist` Phase-1 critique. The `@arch-reviewer` at Phase 3 applies the same four-tag taxonomy to the implementation diff.
- [`docs/architecture/layers/llm.md`](llm.md) — sibling layer; the cross-stage join cell (plan A5) consumes its parquet output when present.
- [`docs/architecture/layers/ingestion.md`](ingestion.md) — Stage 13's bi-temporal store; `OUTPUT_SCHEMA` is the upstream contract `embed_corpus` consumes.
- ADR-0008 (`decisions/0008-embedding-index-protocol.md`) — Embedder + VectorIndex as `runtime_checkable` Protocols.
- ADR-0003 (`decisions/0003-protocol-for-model-interface.md`) — `typing.Protocol` over `abc.ABC` for swappable interfaces; the `Embedder` + `VectorIndex` Protocols apply the same pattern.
- README.md §"Before the meetup" — operator-facing model pre-warm walkthrough.
