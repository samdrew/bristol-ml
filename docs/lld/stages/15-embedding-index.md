# Stage 15 — Embedding index over REMIT

## Goal

Land the project's first **content-addressed cache** behind a typed,
swappable boundary, and ship a notebook-grade semantic search demo
over the Stage 13 REMIT corpus. Three targets, in priority order:

1. Ship the `Embedder` + `VectorIndex` Protocols (plan §1 D2 + D3) so
   a future RAG-style stage and Stage 16's optional cross-stage join
   can depend on the boundary without depending on `sentence-
   transformers` or any specific vector store.
2. Ship two interchangeable backends for each Protocol — `StubEmbedder`
   + `SentenceTransformerEmbedder` for `Embedder`, `StubIndex` +
   `NumpyIndex` for `VectorIndex` — selected by config + env-var,
   triple-gated for CI safety (plan §1 D8).
3. Ship the demo notebook — `notebooks/15_embedding_index.ipynb` —
   that takes a fixed REMIT event id, returns ten semantically-similar
   neighbours with cosine scores, and renders a UMAP scatter coloured
   by fuel type. The scatter is the demo moment (intent §"Demo
   moment"); the cache-hash invalidation is the architectural lesson
   (intent line 40).

## What was built

- `src/bristol_ml/embeddings/__init__.py` — public boundary re-export
  (Protocols, factories, env-var constants, dataclasses). Stays
  light: importing this package does *not* drag
  `sentence-transformers` / `torch` into the consumer's import graph
  (asserted by `tests/unit/embeddings/test_protocol.py::test_boundary_import_does_not_pull_sentence_transformers`).
- `src/bristol_ml/embeddings/_protocols.py` — `Embedder` (2 methods +
  2 properties) and `VectorIndex` (4 methods + 1 property), both
  `runtime_checkable`. `NearestNeighbour` `NamedTuple` for the
  `query` return shape. ADR-0008 records the Protocol-over-ABC
  rationale.
- `src/bristol_ml/embeddings/_embedder.py` — `StubEmbedder` (8-dim
  SHA-256-derived L2-normalised float32 vectors; deterministic;
  mirrors the `"query: "` prefix asymmetry); `SentenceTransformerEmbedder`
  (loads `Alibaba-NLP/gte-modernbert-base` with
  `model_kwargs={"torch_dtype": torch.float16}`; deferred import so
  the boundary stays light; `HF_HUB_OFFLINE=1` enforced via
  `tests/conftest.py`).
- `src/bristol_ml/embeddings/_index.py` — `StubIndex` (in-memory
  list-of-pairs; minimum viable conformance) and `NumpyIndex`
  (pre-normalised float32 corpus matrix; cosine query is a single
  `corpus @ query.T` matmul; `np.savez_compressed` persistence with
  the `.tmp` + `os.replace` atomic-write idiom).
- `src/bristol_ml/embeddings/_cache.py` — `EmbeddingCache` +
  `EmbeddingCacheMetadata`. `load_or_build` is the cache's only
  public entry: tries to read the parquet, recomputes
  `corpus_sha256` from the supplied texts, compares against the
  stored hash *and* the supplied `embedder.model_id`, rebuilds on
  mismatch with a `loguru` WARNING naming the offending field.
  Three provenance fields stored in Parquet `custom_metadata`:
  `embedded_at_utc`, `corpus_sha256` (full 64-char hex),
  `model_id`. `git_sha` was the Scope-Diff cut (no AC; duplicates
  registry-layer convention for a cache file).
- `src/bristol_ml/embeddings/_factory.py` — `build_embedder`,
  `build_index`, `embed_corpus` (the high-level glue). Triple-gated
  dispatch: `BRISTOL_ML_EMBEDDING_STUB=1` env-var beats YAML
  discriminator beats explicit live config. `embed_corpus` ties
  text-synthesis + cache + factory + index hydration into one
  call site that the notebook and the standalone CLI both use.
- `src/bristol_ml/embeddings/_text.py` — `synthesise_embeddable_text`
  for NULL `message_description` rows. Falls back to
  `"<event_type> <cause> <fuel_type> <affected_unit>"` joined; the
  documented sentinel `"REMIT event (details unavailable)"` fires
  when every structured field is also NULL. Mechanical reuse of
  Stage 14's OQ-B resolution.
- `src/bristol_ml/embeddings/_projection.py` — `project_to_2d`
  UMAP wrapper (`random_state=42`, `n_jobs=1`, `metric="cosine"`,
  `n_components=2`). Lazy import keeps the boundary import light.
- `src/bristol_ml/embeddings/__main__.py` —
  `python -m bristol_ml.embeddings`. Composes `+embedding=default`
  automatically, prints the resolved config, builds a synthetic
  5-row index, runs a one-shot top-3 query, exits 0. Output is
  byte-deterministic in stub mode.
- `conf/_schemas.py` — new `EmbeddingConfig` Pydantic model (`type:
  Literal["stub", "sentence_transformers"]` discriminator,
  `model_id`, `cache_path`, `vector_backend`, `default_top_k`,
  `projection_type`, `force_rebuild`, `fp16`).
  `AppConfig.embedding: EmbeddingConfig | None = None` slot — the
  `None` default keeps every prior stage's CLI / config-smoke test
  unaffected.
- `conf/embedding/default.yaml` — Hydra group file mirroring the
  schema defaults; `type: stub` so the offline path is the runtime
  default. Group is *not* in `conf/config.yaml`'s defaults — entry
  points compose it via `+embedding=default` (parallel to `llm`).
- `tests/fixtures/embedding/tiny_corpus.parquet` — eight-row REMIT-
  shaped corpus spanning four fuel types (Nuclear, Gas, Coal, Wind
  ×2 each) with three NULL `message_description` rows (M-AA, M-C,
  M-D). Built by `scripts/_build_embedding_fixture.py` against the
  Stage 13 `OUTPUT_SCHEMA` so a drift between the fixture and
  upstream is caught at the next plan-edit conversation.
- `notebooks/15_embedding_index.ipynb` — eight-cell demo notebook (1
  title-markdown + 6 plan-§5 code cells + 1 closing-discussion
  markdown). Generated programmatically from
  `scripts/_build_notebook_15.py`. Cell 1 sets
  `os.environ["BRISTOL_ML_EMBEDDING_STUB"]="1"` *before* importing
  any embeddings code; Cell 3 builds the index into a per-run
  `tempfile.mkdtemp()` cache; Cell 6 is the optional cross-stage
  join (skip-clean banner when Stage 14 output is absent — the CI
  default codepath).
- Tests:
  - `tests/unit/embeddings/test_protocol.py` — Protocol shape (2 +
    4 method counts pinned), `isinstance(_, Embedder)` /
    `isinstance(_, VectorIndex)` for both stubs, boundary-import
    asserts no `sentence_transformers` in the import graph,
    `NearestNeighbour` namedtuple shape.
  - `tests/unit/embeddings/test_stub_embedder.py` — env-var triple-
    gate dispatch, determinism (same text → bit-identical vector),
    prefix asymmetry between `embed` (query path) and `embed_batch`
    (document path).
  - `tests/unit/embeddings/test_st_embedder.py` — live embedder
    conformance against pre-warmed local cache; xfail / skip when
    cache absent (CI default — the test does not download).
  - `tests/unit/embeddings/test_index_query.py` — top-k sorted
    descending, cosine scores in `[-1, 1]`, save/load round-trip
    (NumpyIndex), empty-index returns `[]`, k > index-size silently
    clipped.
  - `tests/unit/embeddings/test_cache_invalidation.py` — fresh cache
    is no-op; corpus-hash mismatch triggers rebuild + WARNING;
    model-id mismatch triggers rebuild + WARNING; `force_rebuild`
    overrides; atomic-write leaves no `.tmp` artefacts on success.
  - `tests/unit/embeddings/test_text_synthesis.py` — non-NULL
    message passes through; NULL message falls back to structured
    fields; every-field-NULL row hits the documented sentinel;
    pandas NULL flavours (`pd.NA`, `np.nan`, `None`) all coerce
    correctly.
  - `tests/unit/embeddings/test_module_runs_standalone.py` —
    `python -m bristol_ml.embeddings` exits 0 under stub mode +
    prints expected config / neighbour lines.
  - `tests/unit/test_config.py::test_embedding_config_round_trips_through_hydra`
    — Hydra-YAML-to-Pydantic round trip (T7 acceptance).
  - `tests/integration/test_notebook_15.py` — `nbconvert --execute`
    on the demo notebook under `BRISTOL_ML_EMBEDDING_STUB=1` +
    `HF_HUB_OFFLINE=1` (180 s timeout); asserts T5 Cell 1 / 3 / 4 /
    5 each produce output; cell-count sanity check at 8.
- `docs/architecture/decisions/0008-embedding-index-protocol.md` —
  new ADR; `Embedder` + `VectorIndex` are `runtime_checkable`
  Protocols, not ABCs. Three swappable interfaces (`Model`,
  `Extractor`, `Embedder`/`VectorIndex`) now use the same pattern;
  ADR-0008 establishes it as the project's house style.
- `docs/architecture/layers/embeddings.md` — new layer doc capturing
  the Protocol contract, the env-var triple-gate, the content-hash
  cache discipline, the prefix asymmetry, the live-checkpoint
  binding, the UMAP projection rationale, and the optional cross-
  stage cell. Status: Provisional (first realised by Stage 15).
- `docs/architecture/README.md` — layer index extended with the
  Embeddings row; ADR-0008 added to the decisions list.
- `src/bristol_ml/embeddings/CLAUDE.md` — module guide: schema
  reference, cache file layout on disk, model pre-warm ritual,
  how-to-add-a-third-backend recipe.
- `README.md` — new "Before the meetup" section linking the model
  pre-warm one-liner.
- `CHANGELOG.md` — `### Added` entry under `[Unreleased]`.

## Design choices made here

- **Two `runtime_checkable` Protocols, four implementations.** ADR-0008.
  Three swappable interfaces (`Model`, `Extractor`, `Embedder` +
  `VectorIndex`) all use `runtime_checkable` Protocol; a fourth
  swappable interface arriving in a later stage follows the same
  shape. Adding a fifth method to either Protocol is a plan-edit
  conversation, pinned by the `test_*_protocol_has_*_methods` tests.
- **Triple-gated stub-first.** Plan §1 D8.
  `EmbeddingConfig.type` discriminator *and*
  `BRISTOL_ML_EMBEDDING_STUB` env-var *and* `HF_HUB_OFFLINE=1` for
  the model-availability check. The env-var beats YAML so a
  misconfigured live `type` plus a missing local model cache
  cannot trigger a download in CI. Mirror of Stage 14's
  `BRISTOL_ML_LLM_STUB` discipline.
- **Content-addressed cache key = `(corpus_sha256, model_id)`.**
  Plan §1 D7 + D13. Either changing triggers an automatic rebuild
  with a WARNING. The lessons generalise beyond embeddings: any
  cached derivation downstream of an ingested corpus copies the
  same shape — content-hash + backend-id as the invalidation key,
  Parquet `custom_metadata` as the provenance carrier, deterministic
  recompute on mismatch.
- **Cache vectors stay float32 even when inference is fp16.**
  Plan §1 D4 / R-3. `model_kwargs={"torch_dtype": torch.float16}`
  halves the live model RAM footprint (~298 MB → ~149 MB) but the
  cache vector matrix and the index matmul stay float32 — fp16
  inference can introduce minor numeric jitter that would surface
  as non-deterministic top-k ordering when ties are close. Tests
  asserting exact cosine values therefore run under `StubEmbedder`,
  which is fp32 throughout.
- **Pre-normalise at the embedder boundary.** Both `embed` and
  `embed_batch` return L2-normalised vectors. The
  `NumpyIndex.query` matmul is then just `corpus @ query.T` — no
  normalisation hop in the index. The `_embedder` test pins the
  `np.linalg.norm(vec) ≈ 1.0` invariant for both stubs and the live
  embedder.
- **Prefix asymmetry mirrored in the stub.** Plan §1 D9 + A2. The
  live `Alibaba-NLP/gte-modernbert-base` checkpoint follows the E5
  convention: queries take `"query: "`, documents take no prefix.
  The `StubEmbedder` mirrors the asymmetry by hashing the
  prefixed-vs-unprefixed text — a regression where a downstream
  caller routes queries through `embed_batch` (or vice-versa)
  surfaces in `test_stub_embedder.py` rather than as silent
  retrieval-quality degradation.
- **UMAP, not PCA, for the 2D projection.** Plan A3 (Ctrl+G flip).
  PCA's two principal components do not preserve local neighbourhood
  structure in 768-dim space, which undermines the demo moment
  ("the 2D projection shows the corpus falling into visible
  clusters by event type and fuel"). Dependency cost (~200 MB
  compiled `numba` + `llvmlite`) accepted. UMAP wrapper pins
  `n_jobs=1` because UMAP's `random_state` is coordinate-exact-
  reproducible only when single-threaded (per the umap-learn
  docs).
- **Tiny-corpus guard for UMAP at fixture size.** Stage 15's tiny
  corpus (8 rows) cannot use UMAP's default `n_neighbors=15` — the
  library asserts `n_neighbors < n_samples`. The notebook computes
  `n_neighbors_eff = max(2, min(15, n_samples - 1))` so the
  fixture path executes cleanly while the production path
  (~thousand-row corpus) keeps the default.
- **Standalone-module entry only; no `build` / `query`
  subcommands.** Plan A4 (Ctrl+G narrowing).
  `python -m bristol_ml.embeddings` mirrors the Stage 14 extractor
  CLI shape: prints the active config + a one-shot top-3 query and
  exits 0. The notebook plus the standalone entry covers both demo
  paths; subcommands would have been two more implementation
  surfaces and two more CLI tests for no marginal value.
- **Optional cross-stage join cell, skip-clean when Stage 14 output
  absent.** Plan A5 (Ctrl+G overturn of the Scope-Diff cut). The
  notebook's final code cell joins Stage 14's structured-extraction
  parquet onto the nearest-neighbour result *when present*; when
  absent (the default in stub-mode CI), prints a single-line skip
  banner and continues. The CI codepath under stub mode exercises
  the absent branch.
- **`save` / `load` on the index, not on the cache.** Plan §1 D3.
  The `VectorIndex.save` / `load` methods persist the in-memory
  index for fast restart; the `EmbeddingCache` is what carries the
  provenance metadata for invalidation. Keeping the two concerns
  distinct is what makes the swap mechanical: a future `FaissIndex`
  needs to satisfy the four methods only; the cache layer is
  unchanged.
- **Provenance via 12-char SHA-256 prefix in log lines, full hash
  in parquet.** Plan §1 D13. Full 64-char hex stored long for
  archival joinability; 12-char prefix exposed to log lines and
  notebook display because a 132-column terminal renders 64 hex
  chars badly. Collisions over the project lifetime are vanishing
  (12 hex chars = 48 bits = 1 in 281 trillion).
- **Cache file path derived from `model_id` at call time, not
  Pydantic validation time.** Plan §1 D14. Pydantic models are
  frozen, and the sanitised filename depends on a runtime value;
  resolving in the factory (`embed_corpus`) keeps the schema
  compositional. The sanitiser replaces non-`[A-Za-z0-9._-]`
  characters with `_` and strips trailing `_`s, with an
  `"unnamed_model"` fallback if every character was rejected.
- **Atomic-write idiom borrowed from ingestion.** Plan §1 D14.
  `np.savez_compressed` to a `.tmp` sibling then `os.replace` —
  the same idiom Stage 13's `bristol_ml.ingestion._common._atomic_write`
  established. A crash mid-write leaves the previous-good cache
  intact; a successful write is visible atomically.
- **`NearestNeighbour` is a `NamedTuple`, not a Pydantic model.**
  The `VectorIndex.query` return shape crosses the boundary in
  tight loops; per-row Pydantic validation is wasted because the
  index only ever returns pairs it just constructed (the values
  are already typed at the source). Keeps the boundary lightweight
  without sacrificing static type info.
- **No formal cassette pattern at this layer.** Plan §1 D20.
  Stage 14's cassette pattern addressed an authenticated remote
  endpoint (OpenAI Chat Completions); the Stage 15 live path is a
  *local* model load — there are no HTTP calls to record once the
  model cache is warm. `HF_HUB_OFFLINE=1` covers the live path in
  CI by causing any network call to surface as `OSError` rather
  than a silent download.
- **Loguru convention, no exact-message-text test assertions.**
  Plan NFR-8 (softened from the draft). The cache emits structured
  records on hit / rebuild / stale-detected; tests assert *that* a
  record at the expected level fires, not the exact wording. Keeps
  the docstring ergonomics free of test-coupling.

## Demo moment

From a clean clone (Stages 0–14 already built):

```bash
uv sync --group dev                                                   # installs sentence-transformers + umap-learn
uv run pytest -q                                                      # all green, including the new 13-test embeddings suite + nbconvert smoke

uv run python -m bristol_ml.embeddings --help                         # offline; CLI usage
uv run python -m bristol_ml.embeddings                                # offline stub; prints config + sample top-3
uv run python -m bristol_ml.embeddings embedding.force_rebuild=true   # rebuild path

# Live model pre-warm (one-time, ~298 MB):
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"

# Notebook (offline under stub, ~20 s wall-clock):
BRISTOL_ML_EMBEDDING_STUB=1 \
uv run jupyter nbconvert --to notebook --execute \
    notebooks/15_embedding_index.ipynb --output /tmp/15_test_run.ipynb
```

The notebook's Cell 4 (top-k query) is the demo moment proper: a
fixed gold event id, ten semantically-similar neighbours with
cosine scores rendered as a pandas frame. Cell 5's UMAP scatter
puts the visual cluster lesson alongside the numeric one — the
corpus falls into visible clusters by fuel type, which a
facilitator can pause on to ask: *"these two events are nearest
neighbours despite having different fuel types — what does that
say about how the embedding model sees REMIT text?"*. Cell 6's
optional cross-stage join is the bridge to Stage 16: when Stage
14's structured features are present, the top-k display gains
`event_type` / `fuel_type` columns alongside the cosine score.

## Observations from execution

- **Live model not pre-warmed in dev environment.** No
  HuggingFace cache available in the worktree at Stage 15
  build-up. The live-path test
  (`tests/unit/embeddings/test_st_embedder.py::test_st_embedder_loads_offline`)
  honours the `xfail`/`skip` semantics when the cache is absent —
  same shape as `test_remit_cassettes` and
  `test_llm_extractor_cassette`. The pre-warm is a one-line ritual
  documented in `src/bristol_ml/embeddings/CLAUDE.md` and
  `README.md`'s "Before the meetup" section; an operator with
  network access produces the cache and unblocks the live-path
  test.
- **Final fixture corpus: 8 rows, 4 fuel types, 3 NULL
  `message_description`.** Plan §1 D18 named "5–10 rows" as the
  range; landed at 8 to give the UMAP scatter just enough cluster
  signal to demonstrate the projection's visual lesson without the
  fixture turning into a curation artefact. Distribution:
  Nuclear ×2 (M-A, M-AA), Gas ×2 (M-B, M-BB), Coal ×2 (M-C, M-CC),
  Wind ×2 (M-D, M-DD); three NULL `message_description` rows
  (M-AA, M-C, M-D) exercise the structural-fallback path.
- **One pre-emptive deviation: PyTorch `_TritonLibrary` cross-suite
  hygiene.** During the boundary-import test
  (`test_boundary_import_does_not_pull_sentence_transformers`),
  the first attempt evicted `torch` along with `sentence_transformers`
  via `sys.modules.pop`. When pytest later collected
  `test_sequence_dataset.py` (which has `import torch` at module
  level), torch's `__init__.py` re-ran its module-level
  `torch.library.Library('triton', 'DEF')` registration and raised
  `RuntimeError: Only a single TORCH_LIBRARY can be used to register
  the namespace triton`, cascading into 49 unrelated test failures.
  Fix: the test pops only `sentence_transformers` (the actual
  load-bearing assertion target); a 12-line implementation note in
  the test docstring records the cross-suite hygiene constraint so
  a future author does not re-introduce the regression.
- **`embed_corpus` returns a 2-tuple `(VectorIndex, EmbeddingCache)`,
  not a single result object.** First-attempt notebook code
  referenced `result.embedder` / `result.vectors` / `result.cache_hit`
  — none of which exist. The shipped factory is a thin wrapper:
  `EmbeddingCache` carries the metadata, `VectorIndex` is the
  query surface, `embedder` is reconstructed independently for the
  query path (so the notebook's cell ordering matches the
  factory's call ordering). The 2-tuple shape is asserted in
  `test_factory.py` (T7 round-trip).

## Phase-3 review fixes

*To be filled in after the three reviewers (`arch-reviewer`,
`code-reviewer`, `docs-writer`) run.*

## Deferred

- **Live HuggingFace model pre-warm in CI.** Build-up phase
  decision; documented ritual lives in the module guide and
  README. Pre-warming the model in CI would add ~298 MB to every
  run for no marginal value beyond the stub path; the live-path
  unit test honours xfail/skip when the cache is absent.
- **FAISS / Qdrant / quantised-index backends.** Plan §1 D5 leaves
  the `Literal["numpy", "stub", ...]` slot extensible. Stage 15
  ships only `numpy` and `stub`; a future RAG-style stage (or a
  concrete need to scale past ~50k rows) is the trigger to add a
  third backend.
- **`embed_async` / streaming embeddings.** Out of scope at this
  stage. The Protocol's two synchronous methods are sufficient
  for the demo + Stage 16 join. A future async serving path would
  supersede ADR-0008.
- **Calibrated retrieval scores.** Cosine similarity in `[-1, 1]`
  is the documented score; it is *not* a probability. Calibrating
  retrieval scores into a probability of relevance would be a
  Stage-17-or-later conversation if observed downstream usage
  needs it.
- **Hydration of NULL `message_description` via
  `GET /remit/{messageId}`.** Inherited from Stage 14's deferral.
  ~45,000 calls per archive run, not justified by the embedding
  layer's demo scope. The structured-fallback path (D9) is the
  shipped solution.
- **`bristol_ml.embeddings.cli` build / query subcommands.** Plan
  A4 narrowed the CLI to the standalone-module entry only. A
  future stage that needs a CLI build / query workflow (e.g. for
  a non-notebook batch flow) is the trigger to add subcommands.
- **PCA fallback path.** Plan A3 overturned the PCA fallback; the
  layer is UMAP-only at Stage 15. R-6 documents the fallback
  posture: if `umap-learn` install footprint blocks a constrained
  CI runner, switching the projection to t-SNE (sklearn, no extra
  dep) is a one-PR pre-cleared change.

## Next

→ Stage 16: feature-table join — both Stage 14 (extracted features)
and Stage 15 (embeddings) flow into the modelling feature table.
The cross-stage cell already shipped here (plan A5) is the
notebook-level demo of the same join; Stage 16's task is to
crystallise that join into a feature-table layer the model can
train against, using Stage 13's `as_of` mechanic to guarantee no
training-time leakage.
