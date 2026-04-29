# Stage 15 — Embedding index over REMIT corpus: codebase map

Artefact type: researcher output. Audience: implementing team.
Baseline SHA: 6267cc0. Date: 2026-04-27.

---

## 1. REMIT corpus shape

**File:** `/workspace/src/bristol_ml/ingestion/remit.py`
**Layer contract:** `/workspace/docs/architecture/layers/ingestion.md` §"remit.py — output schema"

### `load()` return

```python
def load(path: Path) -> pd.DataFrame:
```

Returns a pandas DataFrame matching `OUTPUT_SCHEMA` (remit.py:247–272). All four
timestamp columns are `timestamp[us, tz=UTC]`; `effective_to` is `pd.NaT` for
open-ended events. Schema is asserted on read; missing column or type mismatch
raises `ValueError` naming the offending column.

### `OUTPUT_SCHEMA` columns (remit.py:148–173)

| Column | Parquet type | Notes |
|--------|-------------|-------|
| `mrid` | `string`, non-null | Primary key half |
| `revision_number` | `int32`, non-null | Primary key half |
| `message_type` | `string`, non-null | |
| `message_status` | `string`, non-null | Active / Inactive / Cancelled / Withdrawn / Dismissed |
| `published_at` | `timestamp[us, tz=UTC]`, non-null | Transaction-time axis |
| `effective_from` | `timestamp[us, tz=UTC]`, non-null | Valid-time start |
| `effective_to` | `timestamp[us, tz=UTC]`, **nullable** | `pd.NaT` = open-ended |
| `retrieved_at_utc` | `timestamp[us, tz=UTC]`, non-null | Per-fetch provenance |
| `affected_unit` | `string`, nullable | BMU id |
| `asset_id` | `string`, nullable | |
| `fuel_type` | `string`, nullable | |
| `affected_mw` | `float64`, nullable | |
| `normal_capacity_mw` | `float64`, nullable | |
| `event_type` | `string`, nullable | |
| `cause` | `string`, nullable | |
| `message_description` | `string`, nullable | **The canonical text field for embedding** (comment at remit.py:170: "Free text — Stage 14 will read this") |

### Canonical text field

`message_description` is **frequently NULL** on the live Elexon stream endpoint
(remit.py:431–436, `_parse_message` comment; also llm.md §"Live path" and
llm/__init__.py:58–60). The Stage 14 pattern — synthesise from structured fields
when NULL — is the established fallback. Stage 15 will need the same strategy:
embed from `message_description` when present; fall back to a concatenation of
`event_type`, `cause`, `fuel_type`, `affected_unit` when not.

### How the Stage 14 harness pulls text

`tests/fixtures/llm/hand_labelled.json` encodes `RemitEvent` objects that mirror
`OUTPUT_SCHEMA` rows. Stage 14's `StubExtractor._load_gold_set` indexes by
`(mrid, revision_number)` (extractor.py:158–163). In the notebook (Cell 3,
`_build_notebook_14.py:189`):

```python
sample_events = [RemitEvent(**r["event"]) for r in records[:3]]
```

So iteration is: load the JSON, iterate `.records`, construct `RemitEvent(**r["event"])`,
then call `.message_description` off the model. For a parquet-backed corpus the
equivalent is `load(path)` → iterate rows → access `row["message_description"]`.

### `as_of` — the bi-temporal primitive

`as_of(df, t)` (remit.py:275–354) applies the transaction-time filter before
embedding; Stage 15 should embed the `as_of`-filtered view so vectors correspond
to the corpus as it was known at a given point in time.

---

## 2. LLM layer's offline-by-default + Protocol pattern

**Files:** `/workspace/src/bristol_ml/llm/__init__.py`,
`/workspace/src/bristol_ml/llm/extractor.py`,
`/workspace/docs/architecture/layers/llm.md`
**ADR:** `/workspace/docs/architecture/decisions/0003-protocol-for-model-interface.md`

### The `Extractor` Protocol shape (llm/__init__.py:150–189)

```python
@runtime_checkable
class Extractor(Protocol):
    def extract(self, event: RemitEvent) -> ExtractionResult: ...
    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]: ...
```

Stage 15 must mirror this exactly: `@runtime_checkable`, two methods, no extra
public attributes. ADR-0003 (decisions/0003-protocol-for-model-interface.md) is
the authoritative precedent — structural subtyping, no ABC inheritance.

### Triple-gate pattern (extractor.py:596–614)

Stage 14 uses three gates to keep CI safe:

1. **Config discriminator** — `LlmExtractorConfig.type: Literal["stub", "openai"]` defaults
   to `"stub"` in `conf/llm/extractor.yaml` (line 28: `type: stub`).
2. **Env-var override** — `BRISTOL_ML_LLM_STUB=1` forces stub regardless of config.
3. **API-key gate** — `LlmExtractor.__init__` reads `config.api_key_env_var` and raises
   `RuntimeError` if empty (extractor.py:398–405), naming both env-vars in the error message.

Stage 15 needs an equivalent `BRISTOL_ML_EMBEDDING_STUB=1` env var (following the pattern
at remit.py:76, `_STUB_ENV_VAR: Final[str] = "BRISTOL_ML_REMIT_STUB"`).

### `build_extractor` factory pattern (extractor.py:576–614)

```python
def build_extractor(
    config: LlmExtractorConfig | None,
    *,
    gold_set_path: Path | None = None,
) -> Extractor:
```

Single dispatch point; supports `config=None` returning the stub. Stage 15 should
implement `build_embedder(config: EmbeddingConfig | None, ...) -> Embedder` with
identical `None`-tolerance and env-var short-circuit.

### Provenance shape (ExtractionResult fields, llm/__init__.py:103–134)

Stage 14 stamps `prompt_hash` (12-char SHA-256 prefix of the prompt bytes) and
`model_id` on every `ExtractionResult`. Stage 15's cached embedding must carry the
equivalent: `model_name` (the sentence-transformer checkpoint name) and a
`corpus_hash` (content-hash of the embedded text corpus, not a per-row hash).

The prompt-hashing utility lives at `/workspace/src/bristol_ml/llm/_prompts.py` —
`load_prompt(path) -> (text, prompt_hash)` where `prompt_hash` is
`hashlib.sha256(text.encode()).hexdigest()[:12]`. The same SHA-256-then-truncate
pattern is the right precedent for content-hashing the corpus.

---

## 3. Configuration patterns

**Files:** `/workspace/conf/_schemas.py`, `/workspace/conf/llm/extractor.yaml`,
`/workspace/conf/config.yaml`

### How Stage 14 wired its config

`LlmExtractorConfig` (conf/_schemas.py:1024–1091):

```python
class LlmExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["stub", "openai"] = "stub"
    model_name: str | None = None
    api_key_env_var: str = "BRISTOL_ML_LLM_API_KEY"
    prompt_file: Path | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0)
```

`AppConfig.llm: LlmExtractorConfig | None = None` (conf/_schemas.py:1117). The
field is `None`-defaulted so prior stages' CLIs and smoke tests remain unaffected.

The YAML is not listed in `conf/config.yaml`'s defaults; entry points compose it
with `+llm=extractor` (extractor.py:708, notebook cell 1). This is the "serving"
precedent — `ServingConfig` follows the same pattern (conf/_schemas.py:1111).

### For Stage 15

Stage 15 needs `conf/embedding/` (or `conf/embedding.yaml`) plus:

```python
class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["stub", "sentence_transformers"] = "stub"
    model_name: str | None = None           # e.g. "all-MiniLM-L6-v2"
    cache_dir: Path                          # where to persist the vector index
    cache_filename: str = "embedding.index" # matches ingestion cache_filename pattern
    backend: Literal["flat", "hnsw"] = "flat"
    # ... batch_size, device, etc.
```

`AppConfig` gets `embedding: EmbeddingConfig | None = None`, mirroring the `llm`
and `serving` precedents. The Pydantic constraint style (`extra="forbid"`,
`frozen=True`, `Field(...)` validators) is mandatory (conf/_schemas.py:18–20 sets
the template).

---

## 4. Caching / artefact-on-disk patterns

### Ingestion layer: `_atomic_write` (ingestion/_common.py:201–210)

```python
def _atomic_write(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
```

Pattern: write to `.tmp` sibling, then `os.replace`. Used by every ingester for
their parquet outputs (remit.py:237; neso.py; weather.py).

### Registry layer: `_atomic_write_run` (registry/_fs.py:69–122)

Same principle but for a directory: stage into `.tmp_{uuid}/` then `os.replace`
to the final run directory. Staging dir cleaned up on any exception.

### Content-hash invalidation

No existing prior art for *content-hash-keyed* cache invalidation in the codebase.
The closest analogue is the prompt hash in `_prompts.py` (SHA-256 of file bytes,
first 12 hex chars). The embedding cache must be keyed on `(corpus_hash,
model_name)` — the `corpus_hash` being a SHA-256 of the sorted `(mrid,
revision_number, message_description)` tuples for the as-of-filtered corpus slice.

### `retrieved_at_utc` convention

Every ingester stamps `retrieved_at_utc` on every row at fetch time (ingestion
CLAUDE.md: "equal across all rows of a single `fetch` call"). The registry sidecar
(`registry/_schema.py`) carries `fit_utc`. Stage 15 should stamp
`embedded_at_utc: timestamp[us, tz=UTC]` on the embedding cache metadata —
consistent with the `retrieved_at_utc` / `fit_utc` naming pattern across layers.

---

## 5. Notebook scaffolding patterns

**Files:** `/workspace/scripts/_build_notebook_14.py`,
`/workspace/tests/integration/test_notebook_14.py`

### Build script pattern (`scripts/_build_notebook_14.py`)

- `REPO_ROOT = Path(__file__).resolve().parents[1]` (line 29) anchors the output path.
- `OUT = REPO_ROOT / "notebooks" / "14_llm_extractor.ipynb"` (line 30).
- Two helper functions: `md(source: str) -> dict` and `code(source: str) -> dict`
  (lines 42–59) build notebook cells as plain dicts; no `nbformat` import needed.
- Global `_CELL_COUNTER` + `_next_id(prefix)` generates stable cell ids.
- Notebook dict assembled manually and written with `json.dumps(notebook, indent=1)`.
- Three-step regeneration ritual (build script → `nbconvert --execute --inplace` →
  `ruff format`), documented in the module docstring (lines 13–14).

Stage 15 copies this pattern verbatim; output file would be
`notebooks/15_embedding_index.ipynb`.

### Integration test pattern (`tests/integration/test_notebook_14.py`)

Key elements (lines 37–141):

- `NOTEBOOK_PATH` constant; asserts notebook exists before running.
- Copies source notebook to `.pytest-exec-<tmp_path.name>.ipynb` sibling; deletes
  in `try/finally`.
- Injects `BRISTOL_ML_LLM_STUB=1` via `env` dict (line 74) — Stage 15 injects
  `BRISTOL_ML_EMBEDDING_STUB=1`.
- Sets `PYTHONPATH` to include both repo root and `src/` (lines 69–72).
- `subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--execute", ...],
  cwd=REPO_ROOT)`.
- Reads executed notebook JSON back; asserts each marker-tagged cell produced
  non-empty `outputs` list.
- Session-scoped `autouse` fixture sweeps orphaned `.pytest-exec-*.ipynb` files.
- Parametrised cell-count sanity check (line 127: `@pytest.mark.parametrize("expected_count", [7])`).

---

## 6. Test fixture patterns

**Dir:** `/workspace/tests/fixtures/`

### Stage 14 gold-set fixture (`tests/fixtures/llm/hand_labelled.json`)

- Top-level `schema_version: 1` field (asserted by `_load_gold_set`; extractor.py:152–157).
- Top-level `records` list (extractor.py:144–147).
- Each record: `{"event": {RemitEvent fields}, "expected": {ExtractionResult minus provenance}}`.
- Indexed by `(mrid, revision_number)` at load time.
- 76 records; stratified across fuel types and event types.

### Where Stage 15 fixtures would live

`tests/fixtures/embedding/` following the `tests/fixtures/llm/` and
`tests/fixtures/remit/` conventions. A tiny REMIT parquet (5–10 rows, covering
NULL and non-NULL `message_description`) would be the fast-test corpus. The stub
`VectorIndex` returns a fixed list of neighbour `(mrid, revision_number, score)` tuples
for any query — mirrors `StubExtractor`'s gold-set lookup pattern.

No cassette is needed for a sentence-transformers model in offline-stub mode
(no network call), but if the live path calls a remote embedding API, a VCR
cassette would go in `tests/fixtures/embedding/cassettes/` (matching the
`tests/fixtures/llm/cassettes/` shape).

---

## 7. Dependencies already in `pyproject.toml`

### Present and directly useful for Stage 15

| Package | Version pin | Relevance |
|---------|-------------|-----------|
| `numpy` | transitive via torch/scipy | Array operations, cosine similarity |
| `scikit-learn` | transitive via statsmodels; 1.8.0 in lock | `sklearn.metrics.pairwise.cosine_similarity`, `NearestNeighbors` for flat index |
| `torch>=2.7,<3` | runtime dep (pyproject.toml:44) | GPU embedding inference; `torch.nn.functional.normalize` |
| `scipy>=1.13,<2` | runtime dep (pyproject.toml:26) | `cdist` for brute-force cosine |
| `pyarrow>=16,<22` | runtime dep | Cache parquet storage |
| `pandas>=2.2,<3` | runtime dep | DataFrame interface with REMIT corpus |
| `loguru>=0.7,<1` | runtime dep | Standard project logger |
| `pydantic>=2.7,<3` | runtime dep | Config schema + boundary types |
| `joblib>=1.4,<2` | runtime dep | Parallel batch inference (pickle of numpy arrays is safe here) |

### Absent from both `pyproject.toml` and `uv.lock`

Confirmed by `grep` against `uv.lock` (0 matches):

- `sentence-transformers` — not present. Required for the live embedding path.
- `faiss-cpu` / `faiss-gpu` — not present. Required for an HNSW/IVF vector index.
- `umap-learn` — not present. Would be needed only for a 2D projection notebook cell; out of scope for Stage 15 core.
- `chromadb`, `hnswlib`, `annoy` — not present.

`scikit-learn` 1.8.0 **is** transitively present in `uv.lock`. Its
`sklearn.neighbors.NearestNeighbors` with `metric="cosine"` is sufficient for a
flat brute-force index at corpus sizes up to ~50k rows — the Stage 15 "flat" backend
can be implemented with zero new runtime deps. The `sentence-transformers` dep is
the only mandatory addition for a non-stub embedding path.

---

## 8. Provenance + `retrieved_at_utc` convention

### Ingestion layer stamp (CLAUDE.md §"Storage conventions")

```
retrieved_at_utc (timestamp[us, tz=UTC]) is written on every row as
per-fetch provenance — equal across all rows of a single fetch call.
```

Implemented in remit.py at `_to_arrow` (line 767):
```python
frame["retrieved_at_utc"] = retrieved_at   # datetime.now(UTC) from fetch()
```

### Registry sidecar stamp

`registry/_schema.py` carries `fit_utc` (and `git_sha`). The sidecar is written
once per `save()` call.

### Stage 15 must do both

The embedding cache metadata must carry:

- `embedded_at_utc: datetime` — when this embedding run was executed (matches
  `retrieved_at_utc` / `fit_utc` conventions; tz-aware UTC; reject naive values
  same as the RemitEvent validator at llm/__init__.py:81–99).
- `corpus_hash: str` — SHA-256 prefix of the embedded corpus slice (content-based
  invalidation key; same truncation as `prompt_hash` in `_prompts.py`).
- `model_name: str` — the embedding model checkpoint name (matches `model_id` in
  `ExtractionResult`; must be a config value, never hard-coded per §2.1.4).
- `git_sha: str | None` — sourced from `registry/_git.py::_git_sha_or_none()` at
  embed time, mirroring the registry's precedent.

The `(corpus_hash, model_name)` pair is the invalidation key: if either changes,
the cached index must be rebuilt. The `embedded_at_utc` + `git_sha` fields are
provenance only (not cache keys).

---

## Cross-references

- `/workspace/docs/architecture/decisions/0003-protocol-for-model-interface.md` — the Protocol-over-ABC precedent Stage 15 must follow for `Embedder` and `VectorIndex`.
- `/workspace/docs/architecture/layers/llm.md` — the stub-first discipline and triple-gate pattern Stage 15 mirrors.
- `/workspace/src/bristol_ml/ingestion/remit.py` — `OUTPUT_SCHEMA`, `load()`, `as_of()`.
- `/workspace/src/bristol_ml/llm/__init__.py` — `Extractor` Protocol shape to copy.
- `/workspace/src/bristol_ml/llm/extractor.py` — `build_extractor` factory, `StubExtractor`, triple-gate, `STUB_ENV_VAR`.
- `/workspace/src/bristol_ml/llm/_prompts.py` — SHA-256 truncation for content-hash provenance.
- `/workspace/src/bristol_ml/ingestion/_common.py` — `_atomic_write`, `CachePolicy`, structural `Protocol` config pattern.
- `/workspace/src/bristol_ml/registry/_fs.py` — `_atomic_write_run`, directory-level atomic write with `.tmp_{uuid}/` staging.
- `/workspace/src/bristol_ml/registry/_git.py` — `_git_sha_or_none()` for provenance.
- `/workspace/conf/_schemas.py` — `LlmExtractorConfig` (lines 1024–1091) as the config schema template; `AppConfig` (lines 1093–1117) for the new optional field.
- `/workspace/conf/llm/extractor.yaml` — YAML template; `@package llm`, `+llm=extractor` compose pattern.
- `/workspace/scripts/_build_notebook_14.py` — notebook build script pattern.
- `/workspace/tests/integration/test_notebook_14.py` — notebook integration test pattern.
- `/workspace/tests/fixtures/llm/hand_labelled.json` — fixture file shape (`schema_version` guard, `records` list).
