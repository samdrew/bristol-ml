# Stage 14 — LLM feature extractor: codebase map

**Artefact type:** researcher output. Audience: implementing team.
**Date:** 2026-04-26
**Baseline SHA:** `6267cc0` (Stage 13 merged; current branch `stage-10-simple-nn`
carries the Stage 13 production files but not yet a Stage 14 plan).
**Intent source:** `docs/intent/14-llm-extractor.md`

---

## 1. The upstream REMIT data shape

### File

`/workspace/src/bristol_ml/ingestion/remit.py`

### `OUTPUT_SCHEMA` — 16 columns

```
mrid                  pa.string()                  not null  — identifier
revision_number       pa.int32()                   not null  — identifier
message_type          pa.string()                  not null  — identifier
message_status        pa.string()                  not null  — identifier
published_at          pa.timestamp("us", tz="UTC") not null  — transaction-time
effective_from        pa.timestamp("us", tz="UTC") not null  — valid-time start
effective_to          pa.timestamp("us", tz="UTC") nullable  — valid-time end (NaT = open-ended)
retrieved_at_utc      pa.timestamp("us", tz="UTC") not null  — provenance
affected_unit         pa.string()                  nullable  — BMU id e.g. "WBURB-1"
asset_id              pa.string()                  nullable  — prefixed BMU e.g. "T_WBURB-1"
fuel_type             pa.string()                  nullable
affected_mw           pa.float64()                 nullable  — unavailable capacity MW
normal_capacity_mw    pa.float64()                 nullable
event_type            pa.string()                  nullable
cause                 pa.string()                  nullable
message_description   pa.string()                  nullable  — free-text; Stage 14 extraction target
```

**Storage grain:** `(mrid, revision_number)` — unique, append-only.
The parquet at `data/raw/remit/remit.parquet` is sorted by
`(published_at ASC, mrid ASC, revision_number ASC)`.

### Free-text field: `message_description`

The column is populated from the live API field `messageDescription`
(in `_parse_message`, line 436: `record.get("messageDescription")`).

The domain research (`docs/lld/research/13-remit-ingestion-domain.md`
§"Structured versus free text") names the actual Elexon API field as
`relatedInformation`, not `messageDescription`. Observed examples from
that research include:

- `"Status changed to Current"`
- `"GT1 issue"`
- `"~/E_DERBY-1,2024-01-16 23:30:00,2024-01-17 06:00:00,0"` (informal encoded schedule)
- Longer prose: outage reason, duration uncertainty notes, revision rationale.

**The cassette at `tests/fixtures/remit/cassettes/remit_2024_01_01.yaml`
has a gzip-compressed body** — the raw field values are not human-readable
from the YAML file. The stub fixture in `_stub_records()` (lines 568–746 of
`remit.py`) contains the only committed plain-text examples:

| mrid | message_description |
|------|---------------------|
| M-A  | `"Stub: planned nuclear outage for refuelling."` |
| M-B rev0 | `"Stub: gas unit unplanned outage."` |
| M-B rev1 | `"Stub: extended end time after diagnostics."` |
| M-B rev2 | `"Stub: derate revised slightly downward."` |
| M-C rev0 | `"Stub: coal unit outage — later withdrawn."` |
| M-C rev1 | `"Stub: message withdrawn by participant."` |
| M-D  | `"Stub: open-ended wind farm restriction."` |
| M-E  | `"Stub: planned nuclear maintenance."` |
| M-F  | `"Stub: open-ended hydro restriction."` |
| M-G  | `"Stub: solar inverter fault."` |

For Stage 14's hand-labelled set the implementer should supplement
these ten stubs with real messages pulled from the cassette
(decompress with `python -c "import zlib, base64, yaml;
d=yaml.safe_load(open('tests/fixtures/remit/cassettes/remit_2024_01_01.yaml'))"`)
or from a fresh live fetch.

**Critical implementation note:** The stream endpoint (`/datasets/REMIT/stream`)
did not return `messageDescription` as of the Stage 13 live response observed
2026-04 (domain research §R6, and the `_parse_message` comment at line 431:
*"The stream endpoint does not return a long-form message description today…
kept on the schema so Stage 14 can populate it from a follow-up `/remit/{mrid}`
call without a schema migration."*). Stage 14 must decide whether to:
(a) call `GET /remit/{messageId}` per mRID to fill `message_description`, or
(b) read from the opinionated `GET /remit/search?mrid=…` endpoint.
The intent §Scope is silent on this; the implementer must state their choice.

### `MESSAGE_STATUSES` constant

```python
("Active", "Inactive", "Cancelled", "Withdrawn", "Dismissed")
```

### `FUEL_TYPES` constant

```python
("Coal", "Gas", "Nuclear", "Oil", "Wind", "Solar", "Hydro",
 "Pumped Storage", "Biomass", "Other", "Interconnector", "Battery")
```

Note: the live API's canonical fuel-type strings confirmed in domain research
§R2 use a different vocabulary (`"Fossil Gas"`, `"Wind Offshore"`, etc.).
`FUEL_TYPES` uses the project-normalised taxonomy; the extractor's output
should align to this tuple.

### `as_of` function

```python
def as_of(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
```

Returns one row per active mRID as known at `t` (transaction-time only).
Stage 14 does not need to call `as_of` internally but the evaluation
harness will — the comparison is "what did the LLM extract from messages
known at training time?".

---

## 2. Stub-first patterns already in the codebase

### Env-var discriminator pattern

Every stub-capable module declares a module-level constant:

```python
_STUB_ENV_VAR: Final[str] = "BRISTOL_ML_REMIT_STUB"
```

and branches on it inside `fetch`:

```python
use_stub = os.environ.get(_STUB_ENV_VAR) == "1"
if use_stub:
    records = _stub_records()
else:
    ...live path...
```

Stage 14 follows the same pattern. Suggested constant name:
`_STUB_ENV_VAR: Final[str] = "BRISTOL_ML_LLM_STUB"`.
CI sets it to `"1"` so the hand-labelled set is used by default.

**The stub is the default at runtime** — the real LLM path is gated by
both the env var and a config switch (intent AC-2, AC-3).

### Discriminated-union pattern for swapping implementations

`conf/_schemas.py` lines 931–941 implement the `ModelConfig` discriminated union:

```python
ModelConfig = (
    NaiveConfig | LinearConfig | SarimaxConfig
    | ScipyParametricConfig | NnMlpConfig | NnTemporalConfig
)
# AppConfig.model is:
model: ModelConfig | None = Field(default=None, discriminator="type")
```

Each concrete config carries `type: Literal["naive"]` / `Literal["linear"]` etc.
Stage 14 `LlmExtractorConfig` should follow the same `Literal["stub"]` /
`Literal["openai"]` pattern on a `type` field so a future third implementation
slot in without breaking callers.

---

## 3. Ingestion-adjacent dependencies: `_common.py` API-client wrapper

**File:** `/workspace/src/bristol_ml/ingestion/_common.py`

### What Stage 14 needs from `_common.py`

Stage 14 will make outbound HTTP calls to an LLM API (and potentially to
`GET /remit/{messageId}` to fill `message_description`). The relevant helpers:

| Helper | Signature | What it needs from config |
|--------|-----------|--------------------------|
| `_retrying_get` | `(client, url, params, config: RetryConfig) → httpx.Response` | `max_attempts`, `backoff_base_seconds`, `backoff_cap_seconds`, `request_timeout_seconds` |
| `_respect_rate_limit` | `(last_request_at, min_gap_seconds) → float` | `min_inter_request_seconds` |
| `CachePolicy` / `CacheMissingError` | StrEnum / Exception | — |

`_retrying_get` retries on `httpx.ConnectError`, `httpx.ReadTimeout`, 5xx, and
429. For LLM APIs, 429 (rate-limit) and 5xx (server errors) are both plausible;
the helper covers both automatically.

### Adding a new HTTP client backend

The pattern (from `remit.py` lines 231–234):
```python
with httpx.Client(timeout=config.request_timeout_seconds) as client:
    _respect_rate_limit(None, config.min_inter_request_seconds)
    records = _live_fetch(config, client=client)
```

The `LlmExtractorConfig` Pydantic model must structurally satisfy all three
protocols (`RetryConfig`, `RateLimitConfig`, `CachePathConfig`) if it caches
extraction results — no inheritance needed, just the same field names.

### API key env var pattern

There are **no existing API-key env-var reads** in the codebase (the Elexon
API is unauthenticated; the NESO API is also unauthenticated). Stage 14
introduces this pattern for the first time. The intent §AC-3 says:
*"guarded by a configuration switch and uses an environment variable for the
API key"*. The natural place is:

```python
_LLM_API_KEY_ENV_VAR: Final[str] = "BRISTOL_ML_LLM_API_KEY"
api_key = os.environ.get(_LLM_API_KEY_ENV_VAR)
if api_key is None and use_live:
    raise RuntimeError(
        f"{_LLM_API_KEY_ENV_VAR} must be set when the LLM extractor is live; "
        "set it to your provider API key or set BRISTOL_ML_LLM_STUB=1 for offline mode."
    )
```

No existing precedent — Stage 14 sets it.

---

## 4. Configuration surface

### Where `LlmExtractorConfig` belongs

`conf/_schemas.py` has `AppConfig` (lines 1024–1042). The current top-level
groups are: `ingestion`, `features`, `evaluation`, `model`, `serving`.
Stage 14 adds an `llm` group:

```python
# conf/_schemas.py — add alongside ServingConfig:
class LlmExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["stub", "openai"]  # discriminator
    ...

# AppConfig — add field:
llm: LlmExtractorConfig | None = None
```

**Hydra wiring:** Following the `ingestion/remit@ingestion.remit` pattern:

- Add `conf/llm/stub.yaml` and `conf/llm/openai.yaml`.
- Add `- llm: stub@llm` to `conf/config.yaml` defaults (or leave absent with
  the `None` default, as `serving` does — the serving pattern is the right
  precedent since the LLM extractor is not part of the training pipeline by
  default).

The `ServingConfig` precedent (lines 988–1022) is the closest model — a
self-contained block that is `None` in the default train-pipeline config and
composed in only by the consuming entry point.

### `IngestionGroup` — no change needed

`conf/_schemas.py` line 267–281 defines `IngestionGroup`; `remit` is already
present at line 281. Stage 14 does not add anything to `IngestionGroup`.

---

## 5. Model registry pattern

**Registry is not directly relevant to Stage 14.**

The registry (`bristol_ml.registry`, `/workspace/src/bristol_ml/registry/__init__.py`)
stores fitted `Model` protocol implementors with a `metrics_df`. The LLM
extractor is a transformation step, not a forecasting model — it does not
fit a `Model` and does not produce `metrics_df` in the registry sense.

However, the intent §AC-4 says *"any registered feature extraction should
record which prompt produced it"*. The natural storage is a small JSON sidecar
alongside the extraction cache (not the model registry). The hand-labelled
set itself is a versioned file in-repo.

The only registry touchpoint is if Stage 16 (feature-table join) saves a
model trained on LLM-extracted features — that is Stage 16's concern.

---

## 6. Notebook conventions

### Naming

`notebooks/14_llm_extractor.ipynb` — follows the two-digit prefix + underscore convention.

### Bootstrap cell (load-bearing)

All notebooks open identically (from `notebooks/13_remit_ingestion.ipynb` T5 Cell 1):

```python
REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # cache_dir in conf/*.yaml resolves against cwd
```

`os.chdir(REPO_ROOT)` is load-bearing: YAML `cache_dir` values use
`${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/…}` which resolves relative to cwd.

### Cell marker convention

`notebooks/13_remit_ingestion.ipynb` uses `# T5 Cell N —` in code-cell
comments (T5 = task 5, referring to the plan). Stage 14 should use the same
`# T5 Cell N —` style.

### Config loading

```python
cfg = load_config(config_path=REPO_ROOT / "conf")
assert cfg.llm is not None, "LLM extractor config not resolved"
```

### CI execution

CI runs the Stage 13 notebook against `BRISTOL_ML_REMIT_STUB=1`. Stage 14
must similarly gate: `BRISTOL_ML_LLM_STUB=1` for CI, no API key needed.

---

## 7. Testing conventions

### `pytest.importorskip` guard

Every test file begins:
```python
module = pytest.importorskip("bristol_ml.llm.extractor")
```
so the suite stays green while the implementer is still building.

### Cassette fixture pattern (for the per-message `/remit/{mrid}` call, if used)

```
tests/fixtures/llm/cassettes/<cassette_name>.yaml
```

Fixture structure matches the REMIT pattern in
`tests/integration/ingestion/test_remit_cassettes.py`:

```python
@pytest.fixture(scope="module")
def _cassette_present_or_skip(request):
    if request.config.getoption("--record-mode", default="none") != "none":
        return
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip("No cassette at …")

@pytest.fixture
def vcr_config(request):
    record_mode = request.config.getoption("--record-mode", default="none")
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": record_mode,
        "allow_playback_repeats": True,
    }
```

`filter_headers` already includes `authorization` and `x-api-key` — Stage 14's
LLM API key (typically in `Authorization: Bearer …`) is automatically filtered.
No change to the `vcr_config` template needed.

### `_build_config` helper

Every unit-test file has a local helper constructing the module-specific Pydantic
config pointing at `tmp_path`:

```python
def _build_extractor_config(tmp_path: Path) -> LlmExtractorConfig:
    return LlmExtractorConfig(
        type="stub",
        cache_dir=tmp_path,
        cache_filename="llm_features.parquet",
    )
```

### `loguru_caplog` shared fixture

`/workspace/tests/conftest.py` lines 16–34 provides `loguru_caplog` — use for
any test asserting on structured log lines from the extractor.

### `monkeypatch.setenv` for stub trigger

```python
def test_stub_extract_returns_fixed_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRISTOL_ML_LLM_STUB", "1")
    ...
```

### CI config (pyproject.toml line 119)

```toml
addopts = "-ra --strict-markers --record-mode=none -m 'not slow and not gpu'"
```

`--record-mode=none` is already global. No change needed.

---

## 8. Relevant files

| File | Purpose |
|------|---------|
| `/workspace/src/bristol_ml/ingestion/remit.py` | Upstream data source; read `OUTPUT_SCHEMA`, `MESSAGE_STATUSES`, `FUEL_TYPES`, `as_of`, `_stub_records`, `_parse_message` |
| `/workspace/src/bristol_ml/ingestion/_common.py` | HTTP retry helper, cache helpers, structural protocols; reuse `_retrying_get`, `_respect_rate_limit`, `CachePolicy`, `CacheMissingError` |
| `/workspace/conf/_schemas.py` | Add `LlmExtractorConfig` here; wire into `AppConfig`; follow `ServingConfig` (lines 988–1022) and `RemitIngestionConfig` (lines 211–264) as templates |
| `/workspace/conf/config.yaml` | May add `- llm: stub` to defaults list (or leave absent like `serving`) |
| `/workspace/src/bristol_ml/ingestion/CLAUDE.md` | Schema table format to follow when documenting the new module's output schema |
| `/workspace/tests/conftest.py` | `loguru_caplog` fixture (lines 16–34) |
| `/workspace/tests/integration/ingestion/test_remit_cassettes.py` | VCR cassette fixture pattern to replicate |
| `/workspace/tests/unit/ingestion/test_remit.py` | `_build_config` helper pattern; `_make_row` / `_make_df` in-memory test helpers; monkeypatch stub trigger pattern |
| `/workspace/notebooks/13_remit_ingestion.ipynb` | T5 Cell marker convention; bootstrap cell; config loading pattern |
| `/workspace/pyproject.toml` lines 112–126 | pytest markers, `--record-mode=none` global flag, `filterwarnings` |

**New files Stage 14 will create:**

| File | Purpose |
|------|---------|
| `src/bristol_ml/llm/__init__.py` | Module entry point |
| `src/bristol_ml/llm/extractor.py` | Main module: stub + real implementation, evaluation harness, CLI |
| `src/bristol_ml/llm/CLAUDE.md` | Module-local guide |
| `conf/llm/stub.yaml` | Hydra config for stub implementation |
| `conf/llm/openai.yaml` | Hydra config for real LLM implementation |
| `tests/unit/llm/test_extractor.py` | In-memory unit tests |
| `tests/integration/llm/test_extractor_cassettes.py` | VCR cassette tests (if calling Elexon per-mRID) |
| `tests/fixtures/llm/cassettes/` | Cassette directory |
| `notebooks/14_llm_extractor.ipynb` | Demo notebook |

---

## 9. Hazards and fragile areas

### `message_description` is NULL in the current stub and likely NULL in the stream endpoint

The stream endpoint (`/datasets/REMIT/stream`) did not return `messageDescription`
in the live response observed 2026-04 (domain research §R6, `_parse_message` line
431 comment). The stub fixture populates it with short synthetic strings. The actual
free text is in the Elexon API's `relatedInformation` field on the opinionated
`GET /remit/{messageId}` endpoint, not `messageDescription`. Stage 14 must resolve
whether to:
(a) call the per-mRID endpoint to hydrate `message_description` (adds network cost,
    needs its own cassette or stub), or
(b) work with whatever the stream returns (may be all-NULL in production), or
(c) use the `relatedInformation` field from a supplementary fetch pass and write
    it to `message_description` in a pre-processing step.
**This is the single highest-risk gap in the upstream data contract.**

### `FUEL_TYPES` vocabulary mismatch vs live API

The project `FUEL_TYPES` tuple uses normalised short names (`"Gas"`, `"Wind"`,
`"Nuclear"`) while the live API returns EU-REMIT vocabulary (`"Fossil Gas"`,
`"Wind Offshore"`, `"Wind Onshore"`, etc.). The `_parse_message` function passes
`fuelType` through verbatim (line 421), so the `fuel_type` column on disk may carry
`"Fossil Gas"` from live data but `"Gas"` in the stub fixture. The extractor's
hand-labelled set and evaluation harness must decide which vocabulary is ground truth.
This was an acknowledged wart in Stage 13; the domain research §R2 documents the full
live vocabulary.

### No existing API-key env-var pattern to copy

Stage 14 introduces the first authenticated outbound call in the project. There is no
precedent for the env-var name, the error message style, or the test that verifies
the guard raises when the key is absent but live mode is requested.

### Notebook `message_description` will be null for stub runs

The Stage 13 notebook (`13_remit_ingestion.ipynb`) uses stub records that all have
short `message_description` strings. Stage 14's demo notebook should handle the
case where the real parquet has NULLs in that column, and show the demo from the
stub's synthetic text — the demo moment is the LLM extraction, not the data.

### `IngestionGroup` has no `llm` field — do not add one

`IngestionGroup` (lines 267–281 of `conf/_schemas.py`) is for ingestion sources.
The LLM extractor is not an ingestion source. Its config goes under a new top-level
`llm` field on `AppConfig`, not under `ingestion`.

### `RemitIngestionConfig.window_end: date | None` defaults to `None` (today UTC)

If Stage 14 calls `GET /remit/{messageId}` to hydrate descriptions, it needs the
integer message ID from the revisions endpoint. The stream endpoint returns mRID
strings, not integer IDs. The mapping from mRID to integer ID requires calling
`GET /remit/revisions?mrid=<mrid>`. This is a two-step lookup the Stage 13 ingester
does not do. The implementer must decide upfront whether to add this to Stage 13's
fetch (out of scope per the Stage 13 intent) or handle it entirely in Stage 14.

### Prompt versioning TODO in the intent

The intent §Points says *"any registered feature extraction should record which
prompt produced it"*. There is no existing pattern for prompt versioning in the
codebase. The natural approach is a string field in the extraction sidecar (not the
model registry). This is a design decision the implementer must make — the codebase
offers no precedent.
