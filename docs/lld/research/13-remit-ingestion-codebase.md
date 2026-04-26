# Stage 13 — REMIT ingestion: codebase map

**Purpose:** Name every surface, pattern, and constraint the Stage 13 implementer will touch or depend on.
**Date:** 2026-04-25
**Baseline SHA:** `git log --oneline -1` — current branch `stage-10-simple-nn`; all ingestion modules from Stages 1, 2, 4, 5 are present.

---

## Existing ingestion module shape

### Submodules in `src/bristol_ml/ingestion/`

| File | Source | Stage | Public surface | Cache file |
|------|--------|-------|----------------|------------|
| `neso.py` | NESO CKAN paginated API | 1 | `fetch`, `load`, `OUTPUT_SCHEMA`, `REQUIRED_RAW_COLUMNS`, `CachePolicy`, `CacheMissingError` | `neso_demand.parquet` |
| `neso_forecast.py` | NESO CKAN single-resource | 4 | `fetch`, `load`, `OUTPUT_SCHEMA`, `IDENTITY_RAW_COLUMNS`, `CachePolicy`, `CacheMissingError` | `neso_forecast.parquet` |
| `weather.py` | Open-Meteo archive API | 2 | `fetch`, `load`, `OUTPUT_SCHEMA`, `KNOWN_VARIABLES`, `CachePolicy`, `CacheMissingError` | `weather.parquet` |
| `holidays.py` | gov.uk JSON | 5 | `fetch`, `load`, `OUTPUT_SCHEMA`, `KNOWN_DIVISIONS`, `CachePolicy`, `CacheMissingError` | `holidays.parquet` |
| `_common.py` | (shared helpers) | 2 | `CachePolicy`, `CacheMissingError`, `_atomic_write`, `_cache_path`, `_respect_rate_limit`, `_retrying_get`, `_RetryableStatusError`, `RetryConfig`, `RateLimitConfig`, `CachePathConfig` | — |
| `__init__.py` | (lazy re-exports) | 1 | `CachePolicy`, `CacheMissingError` via `__getattr__` | — |

**Public contract invariant.** Every module exposes exactly:

```python
def fetch(config: SourceConfig, *, cache: CachePolicy = CachePolicy.AUTO) -> Path: ...
def load(path: Path) -> pd.DataFrame: ...
```

and is invocable as `python -m bristol_ml.ingestion.<source>`.

**Cache discipline (all modules).** The `fetch` body follows the identical three-branch pattern:
1. `OFFLINE` + no file → raise `CacheMissingError` naming the expected path.
2. `AUTO` + file exists → return path immediately (zero network contact).
3. Otherwise (`REFRESH`, or `AUTO` + file absent) → fetch, transform, `_atomic_write`, log row count + path.

`retrieved_at_utc` is stamped once per `fetch` call (`datetime.now(UTC).replace(microsecond=0)`) and written on every row. Re-running `REFRESH` produces identical payload rows; only `retrieved_at_utc` changes — this is the idempotence contract every integration test verifies.

**CLI shape.** Every module has a `_build_cli_parser() → argparse.ArgumentParser` and a `_cli_main(argv) → int`. The `if __name__ == "__main__"` block calls `raise SystemExit(_cli_main())`. Hydra config is loaded inside `_cli_main` via a local `from bristol_ml.config import load_config` import (deferred so `--help` does not pull Hydra into the import chain). The CLI checks that its specific config slot (e.g. `cfg.ingestion.holidays is None`) and prints a helpful message if the YAML default was not loaded.

**`OUTPUT_SCHEMA` constant.** Each module declares a module-level `pa.Schema` named `OUTPUT_SCHEMA`. The `load` function iterates over `OUTPUT_SCHEMA` fields and raises `ValueError` naming the offending column on type mismatch or absence. No permissiveness in `load`; schema assertion is strict.

**`_to_arrow` boundary function.** Each module has a private `_to_arrow(df) → pa.Table` that applies `table.cast(OUTPUT_SCHEMA, safe=True)` as the final type-coercion step. This is the storage boundary: upstream code keeps source-native column names; `_to_arrow` renames and reorders.

**Schema assertion on raw data.** On ingest, required columns missing → hard `KeyError` naming the column. Unknown columns → `UserWarning` + drop. This runs *before* `_to_arrow`. See `_assert_schema` in `neso.py` (lines 245–280) and `weather.py` (lines 283–321).

---

## Shared utilities and reuse candidates

All of the following live in `/workspace/src/bristol_ml/ingestion/_common.py`.

### `CachePolicy` (lines 60–70)
`StrEnum` with values `AUTO = "auto"`, `REFRESH = "refresh"`, `OFFLINE = "offline"`. Stage 13 imports from `_common` directly; re-exports it for notebook ergonomics matching prior ingesters.

### `CacheMissingError` (lines 74–75)
`class CacheMissingError(FileNotFoundError)`. Stage 13 raises this on `OFFLINE` + absent cache.

### `_cache_path(config: CachePathConfig) → Path` (lines 118–122)
Resolves `config.cache_dir` + `config.cache_filename`, calling `.mkdir(parents=True, exist_ok=True)` on the parent. Stage 13's config Pydantic model must expose `cache_dir: Path` and `cache_filename: str` to satisfy the `CachePathConfig` protocol.

### `_atomic_write(table: pa.Table, path: Path) → None` (lines 201–211)
Writes to `path.with_suffix(path.suffix + ".tmp")`, then `os.replace(tmp, path)`.

### `_retrying_get(client, url, params, config: RetryConfig) → httpx.Response` (lines 157–193)
Tenacity-wrapped GET. Retries on `httpx.ConnectError`, `httpx.ReadTimeout`, `_RetryableStatusError` (5xx, 429). Never retries other 4xx. On final failure raises `RuntimeError` naming URL + attempt count. Stage 13's config must expose `max_attempts: int`, `backoff_base_seconds: float`, `backoff_cap_seconds: float`, `request_timeout_seconds: float` to satisfy `RetryConfig`.

### `_respect_rate_limit(last_request_at, min_gap_seconds) → float` (lines 130–145)
Returns `time.monotonic()` after sleeping if needed. Stage 13 passes `None` for the first call (single GET or first in a loop).

### Structural `Protocol`s (lines 83–109)
`RetryConfig`, `RateLimitConfig`, `CachePathConfig` are `@runtime_checkable`. Stage 13's `RemitIngestionConfig` Pydantic model must satisfy all three structurally — no inheritance required. The required field names are: `max_attempts`, `backoff_base_seconds`, `backoff_cap_seconds`, `request_timeout_seconds` (RetryConfig); `min_inter_request_seconds` (RateLimitConfig); `cache_dir`, `cache_filename` (CachePathConfig).

### Known gap — DST helpers not in `_common`
`_autumn_fallback_dates`, `_spring_forward_dates`, `_to_utc`, `_parse_settlement_date` remain duplicated between `neso.py` (lines 304–416) and `neso_forecast.py` (lines 339–432). The CLAUDE.md at `/workspace/src/bristol_ml/ingestion/CLAUDE.md` (lines 119–130) documents this explicitly and names the candidate extraction (`bristol_ml.ingestion._neso_dst`). REMIT has no settlement periods, so Stage 13 does **not** need these helpers and should not move them.

---

## Downstream consumer contract

### Feature assembler (`features/assembler.py`)
The assembler (`/workspace/src/bristol_ml/features/assembler.py`) consumes ingestion output via `load(path) → pd.DataFrame`. It never calls `fetch` directly; the calling convention is: notebooks and the CLI call `fetch` to warm the cache, then `load` to read it.

**`OUTPUT_SCHEMA` for the weather-only feature table** (assembler.py lines 91–105):
- `timestamp_utc`, `nd_mw`, `tsd_mw`, `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`, `neso_retrieved_at_utc`, `weather_retrieved_at_utc`
- Primary key: `timestamp_utc` unique, ascending, tz-aware UTC.

**`CALENDAR_OUTPUT_SCHEMA`** (assembler.py lines 118–136) extends the weather-only schema with 44 calendar columns and `holidays_retrieved_at_utc`. The weather-only schema is an exact prefix of the calendar schema.

Stage 13 does NOT join REMIT into the feature table (that is Stage 16). The REMIT parquet is consumed directly by Stage 14 (LLM extractor) as a text stream, not joined into the assembler pipeline. This means Stage 13's parquet schema is completely independent of `assembler.OUTPUT_SCHEMA`.

**Index / join convention Stage 16 will eventually require.** Looking ahead (informing schema design now without specifying it): the assembler joins on `timestamp_utc` as the common key. Stage 16's as-of join from REMIT onto the feature table will need to filter REMIT rows where `published_at <= T` and `effective_from <= T <= effective_to` for any forecast horizon T. The REMIT parquet must therefore store all three times as `timestamp[us, tz=UTC]` columns so the as-of query can run without post-processing.

**`IngestionGroup` in `conf/_schemas.py`** (lines 211–225):
```python
class IngestionGroup(BaseModel):
    neso: NesoIngestionConfig | None = None
    weather: WeatherIngestionConfig | None = None
    neso_forecast: NesoForecastIngestionConfig | None = None
    holidays: HolidaysIngestionConfig | None = None
```
Stage 13 adds `remit: RemitIngestionConfig | None = None` here.

**`conf/config.yaml` `defaults:` list** (lines 1–22): Stage 13 adds `- ingestion/remit@ingestion.remit` to the defaults list, matching the pattern for every prior ingestion group. The CLI guard pattern (`if cfg.ingestion.remit is None: print(...); return 2`) follows identically from `holidays.py` lines 357–369.

---

## Fixture and stub conventions

### pytest-recording / vcrpy
All cassette-backed tests use `pytest-recording` (vcrpy). CI runs with `--record-mode=none` (set in `/workspace/pyproject.toml` line 119: `addopts = "-ra --strict-markers --record-mode=none ..."`). The implementer records once with `pytest --record-mode=once` after deleting the cassette directory.

### Cassette directory layout
```
tests/fixtures/<source>/cassettes/<cassette_name>.yaml
```
Existing: `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml`, `tests/fixtures/weather/cassettes/weather_2023_01.yaml`, `tests/fixtures/holidays/cassettes/holidays_refresh.yaml`, `tests/fixtures/neso_forecast/cassettes/neso_forecast_refresh.yaml`.

Stage 13 creates: `tests/fixtures/remit/cassettes/remit_<slug>.yaml` (one or more cassettes for a curated sample — the intent §"Points for consideration" notes that a full REMIT archive is too large to commit).

### Cassette-skip guard
Every integration test file uses a module-scoped fixture that skips the whole module when the cassette is absent (pattern from `/workspace/tests/integration/ingestion/test_neso_cassettes.py` lines 57–69 and `/workspace/tests/unit/ingestion/test_holidays.py` lines 47–59):

```python
@pytest.fixture(scope="module")
def _cassettes_present_or_skip() -> None:
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip("No cassette at …; record once via pytest --record-mode=once")
```

### `vcr_config` fixture
Every cassette test file declares a `vcr_config` fixture filtering sensitive headers:
```python
return {
    "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
    "record_mode": "none",
    "allow_playback_repeats": True,
}
```
Elexon Insights API **requires authentication** (API key). The `x-api-key` filter is already in the filter list for future-proofing; Stage 13 must confirm the exact Elexon header name and add it if different. This is the first real exercise of the `x-api-key` filter.

### `vcr_cassette_dir` and `default_cassette_name` fixtures
Per-file fixtures point pytest-recording at the source-specific cassette directory and share a single bulk cassette across all VCR-marked tests in the file (pattern from `test_neso_cassettes.py` lines 73–87). Stage 13 follows the same pattern.

### Unit-test fixture files
Alongside cassettes, `neso` has a hand-crafted `tests/fixtures/neso/clock_change_rows.csv` for DST edge cases. `weather` has `tests/fixtures/weather/toy_stations.csv` for aggregator unit tests. Stage 13 will need at least one hand-crafted fixture — a minimal REMIT-style JSON payload covering the bi-temporal query discipline (AC-5).

### `pytest.importorskip` guard
Every test file begins `module = pytest.importorskip("bristol_ml.ingestion.<source>")` so the suite stays green while the implementer is still building. See `test_neso.py` line 37, `test_weather.py` line 31, `test_holidays.py` line 29.

### `_build_config(tmp_path, **overrides)` helper
Every unit-test file has a local `_build_config` helper that constructs the source-specific Pydantic config pointing at pytest's `tmp_path`. This avoids touching the real cache dir in tests. Stage 13 follows the same pattern.

### `loguru_caplog` shared fixture
`/workspace/tests/conftest.py` (lines 16–34) provides `loguru_caplog` for tests asserting on structured log lines. Stage 13 tests that assert on the `logger.info("REMIT cache written: …")` line should use this fixture.

---

## Notebook conventions

### REPO_ROOT resolution cell (cell 1 in every notebook)
All three ingestion notebooks (`01_neso_demand.ipynb`, `02_weather_demand.ipynb`, `05_calendar_features.ipynb`) open with:
```python
REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # cache_dir in conf/ingestion/*.yaml resolves against cwd
```
Stage 13 notebook must open identically. `os.chdir(REPO_ROOT)` is load-bearing: the YAML `cache_dir` values use `${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/…}` which resolves relative to cwd.

### Config loading pattern
After the REPO_ROOT cell, every notebook calls:
```python
cfg = load_config(config_path=REPO_ROOT / "conf")
assert cfg.ingestion.<source> is not None, "<Source> ingestion config not resolved"
path = <source>.fetch(cfg.ingestion.<source>, cache=CachePolicy.AUTO)
```
Stage 13 follows the same `cfg.ingestion.remit` access.

### Thin notebook rule (§2.1.8)
Notebooks import from `src/bristol_ml/`; they do not reimplement logic. The Stage 13 notebook's visualisation (aggregate unavailable capacity over time by fuel type) must call `remit.load(path)` and build the chart inline — no analytical logic reimplemented inside the notebook.

### Markdown prose structure
- Cell 0: `# Stage N — <title>` with a 1-2 paragraph framing + caveats (API choice rationale, data model notes). See `02_weather_demand.ipynb` cell 0 for the template.
- Subsequent markdown cells explain each chart before or after it.
- A final "What this shows / what to try" cell names interactive demo extensions.

### Naming convention
`notebooks/NN_<slug>.ipynb` using two-digit stage number and underscores: `01_neso_demand.ipynb`, `02_weather_demand.ipynb`, `05_calendar_features.ipynb`. Stage 13 → `notebooks/13_remit_ingestion.ipynb`.

---

## Friction points to avoid replicating

### 1. Dead `columns` field on `NesoIngestionConfig`
`/workspace/conf/_schemas.py` lines 51–52 and `/workspace/src/bristol_ml/ingestion/neso.py` (the field exists on the config but is never consumed inside `fetch` — it was scaffolded for potential column-selection behaviour that was never wired). Documented as a known wart in the Stage 2 retro (`/workspace/docs/lld/stages/02-weather-ingestion.md` line 70) and Stage 3 retro (`/workspace/docs/lld/stages/03-feature-assembler.md` line 62). Stage 13 should **not** add a similarly dangling field; if `RemitIngestionConfig` carries configurable column selection, wire it or omit it.

### 2. Duplicated DST helpers across `neso.py` and `neso_forecast.py`
`_autumn_fallback_dates`, `_spring_forward_dates`, `_to_utc` are copy-pasted between the two NESO ingesters (`neso.py` lines 304–416; `neso_forecast.py` lines 339–432). The CLAUDE.md (lines 119–130) documents the duplication and names the candidate extraction. REMIT has no settlement periods so Stage 13 does not need these helpers; it should not extend the duplication by copying them into `remit.py` "just in case."

### 3. `assemble_calendar` duplicates the NESO/weather composition
`/workspace/src/bristol_ml/features/assembler.py` duplicates ~25 lines of the NESO/weather/resample/`build` pipeline inline rather than delegating to `assemble()`, because the mutual-exclusivity invariant of the Hydra group-swap makes delegation impossible at that design. Stage 13 has no assembler concern (Stage 16 owns the REMIT feature join), but the assembler friction is a signal: if Stage 13 introduces any helper that the Stage 16 implementer will need, expose it as a pure function in `remit.py` rather than burying it in the CLI's local scope.

### 4. `weather.py`'s `_assert_schema` takes a `str` station name with a `getattr` fallback
`/workspace/src/bristol_ml/ingestion/weather.py` lines 311–312 carry `station_name = getattr(station, "name", station)` — a cosmetic wart from test ergonomics that crept into production code. Stage 13 should keep assertion helpers accepting plain scalar identifiers (string or int), without accepting domain objects with a `.name` attribute via `getattr`.

### 5. Cache staleness for multi-item sources
`weather.py` caches all stations in one flat parquet. Adding a station to the YAML does **not** trigger a re-fetch on the next `AUTO` call — the cache is treated as all-or-nothing. This is documented in CLAUDE.md (lines 163–167). REMIT is also a multi-message source with potential incremental append semantics; Stage 13 must decide explicitly whether the cache is all-or-nothing (safest and consistent with prior art) or supports partial refresh. If incremental, a new `CachePolicy` value or a separate `--since` flag would be needed — the intent §"Points for consideration" does not require incremental, and adding a new `CachePolicy` value would break existing tests.

### 6. Layer architecture doc is marked "Provisional — revisit after Stage 13"
`/workspace/docs/architecture/layers/ingestion.md` line 3 reads: `Revisit again after Stage 13 (REMIT), which brings bi-temporal storage.` The bi-temporal storage section (lines 123–124) explicitly defers the design decision. Stage 13 must resolve it and update the layer architecture doc (WARN-tier, legitimate Stage 13 activity). The three options named are: extend the base schema with three nullable timestamp columns; isolate REMIT in a separate layer; treat bi-temporality as a features-layer concern.

---

## Module-local CLAUDE.md conventions

**File:** `/workspace/src/bristol_ml/ingestion/CLAUDE.md`

This is the single module-local guide for the entire ingestion layer (all submodules). Stage 13 extends it, not creates a new one. Existing structure:

- **§ "Public contract (every ingester)"** — `fetch` / `load` signatures + `CachePolicy` values. Stage 13 adds a `remit.py` row but does not change this section's contract.
- **§ "Storage conventions"** — `timestamp[us, tz=UTC]`, `int32`/`int8` sizing, atomic writes, `retrieved_at_utc`. Stage 13 **adds** the bi-temporal columns here — this is the first schema with more than one canonical timestamp per row. The CLAUDE.md must document `published_at` / `effective_from` / `effective_to` alongside `retrieved_at_utc` and explain the as-of query semantics.
- **§ "Schema assertion (at ingest)"** — required missing → hard error; unknown → warning + drop. No change for Stage 13.
- **Per-source schema tables** — `neso.py`, `weather.py`, `holidays.py`. Stage 13 adds a `remit.py` table after the `holidays.py` table, following the same Markdown-table format with `Column`, `Parquet type`, `Notes` columns.
- **§ "Shared helpers — `_common.py`"** — documents the `Protocol`-based generality. If Stage 13 adds any helper to `_common.py` (unlikely but possible if an Elexon pagination pattern is general enough), document it here.
- **§ "Stage 4 follow-up"** (lines 119–130) — the pending DST-helper refactor note. Stage 13 must leave this paragraph intact (it does not own that refactor).
- **§ "Fixtures"** (lines 213–225) — documents cassette locations and pytest-recording setup. Stage 13 extends this section with the REMIT cassette path and a note that the cassette is a curated sample (not the full archive) because of size constraints noted in the Stage 13 intent.
- **§ "Licence"** — OGL / CC BY 4.0 / OGL v3.0 acknowledgements. Stage 13 adds the Elexon Insights API data licence acknowledgement (likely "Elexon REMIT data: © Elexon; terms at developer.elexon.co.uk").

**Pattern for schema table entry:**
```markdown
## `remit.py` — output schema

File path: `<cache_dir>/remit.parquet` where `<cache_dir>` defaults to
`${BRISTOL_ML_CACHE_DIR:-./data/raw/remit}`.

| Column              | Parquet type                          | Notes                            |
|---------------------|---------------------------------------|----------------------------------|
| `message_id`        | `string`                              | Elexon REMIT message identifier. |
| …                   | …                                     | …                                |
| `retrieved_at_utc`  | `timestamp[us, tz=UTC]`               | Per-fetch provenance (§2.1.6).   |
```
