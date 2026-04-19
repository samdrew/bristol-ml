# Ingestion — layer architecture

- **Status:** Provisional — exercised by Stage 1 (NESO demand, shipped) and Stage 2 (weather, designed). Revisit again after Stage 13 (REMIT), which brings bi-temporal storage.
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (ingestion paragraph).
- **Concrete instances:** [Stage 1 LLD](../../lld/ingestion/neso.md), [Stage 2 LLD](../../lld/ingestion/weather.md). Research: [NESO](../../lld/research/01-neso-ingestion.md), [weather](../../lld/research/02-weather-ingestion.md).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow interfaces), §2.1.3 (stub-first), §2.1.5 (idempotence), §2.1.6 (provenance), §2.1.7 (tests at boundaries).

---

## Why this layer exists

The ingestion layer is the edge of the system. It is the only place that:

- Holds an HTTP client or opens a network socket.
- Knows the shape of a foreign API (CKAN actions, Open-Meteo query params, Elexon BMRS routes, gov.uk JSON).
- Writes parquet to `data/raw/`.

It owns translation from "whatever the remote gave us" into "typed parquet with a documented schema" — nothing more. Transformation, joining, resampling, feature derivation are all downstream (features layer, §3.2). Ingestion is deliberately thin; the cost of a fat ingestion layer is that every downstream module accumulates assumptions about the source's shape.

Seven concrete ingestion modules land across the stage plan: NESO demand, NESO day-ahead forecasts, weather, bank holidays, Elexon prices, Elexon generation, REMIT. All seven share the conventions below.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| HTTP calls, retries, timeouts | ✓ | — |
| Parsing CSV / JSON / XML into parquet | ✓ | — |
| Provenance metadata (`retrieved_at_utc`, `source_year`) | ✓ | — |
| Schema assertion on raw records | ✓ | — |
| Settlement-period → UTC conversion | ✓ | — |
| Calendar features, lags, joins | — | features layer |
| Derived aggregates (hourly from half-hourly, population-weighted national weather) | — | features layer (Stage 2 seeds `features/weather.py`) |
| Holding state beyond the cache file | — | not anywhere |

The split is enforced by the `fetch`/`load` public interface (below): a module that reaches outside those two callables is doing something that does not belong here.

## Cross-module conventions

Every ingestion module follows the same five-part shape. Stage 1 is the worked template; each subsequent feed is a copy-and-adapt, not a rewrite.

### 1. Module shape

- `src/bristol_ml/ingestion/<source>.py` — fetcher + loader.
- `conf/ingestion/<source>.yaml` — Hydra group file, selectable via `ingestion=<source>` or composable under `defaults`.
- `tests/fixtures/<source>/` — recorded HTTP cassettes plus any hand-crafted edge-case fixtures.

No dispatcher, no registry of ingestion names. Hydra's `_target_` pattern is the only factory. Adding a feed is one Python file, one YAML file, one fixture directory.

### 2. Public interface

Every module exposes exactly two public callables:

```python
def fetch(config: SourceConfig, *, cache: CachePolicy = CachePolicy.AUTO) -> Path: ...
def load(path: Path) -> pd.DataFrame: ...
```

- `fetch` is the side-effectful entry point; it returns the path to the cached parquet.
- `load` is pure and cheap; it validates the on-disk schema and returns a typed dataframe.
- `CachePolicy` is a three-valued enum: `AUTO | REFRESH | OFFLINE`.
  - `AUTO` — use cache if present, fetch if not. Notebook-friendly default.
  - `REFRESH` — always fetch, overwrite cache. Explicit.
  - `OFFLINE` — never touch the network; fail loudly if cache is missing. CI default.
- `python -m bristol_ml.ingestion.<source>` is every module's CLI; it calls `fetch(config, cache=CachePolicy.AUTO)` and prints the resulting path. Satisfies §2.1.1.

The enum is tri-valued deliberately — a boolean `force_refresh` collapses two distinct situations (cold-start-with-network and cold-start-without) into one codepath, and the collapsed version silently fetches in CI if someone ships a misconfiguration.

### 3. Storage conventions

- **Root:** `data/raw/<source>/`, configurable via `config.cache_dir`. Defaults to `${BRISTOL_ML_CACHE_DIR:-data/raw/<source>}`. Gitignored.
- **Format:** Parquet via `pyarrow`.
- **Timestamps:** `pa.timestamp('us', tz='UTC')` for the canonical time column. Microseconds are the portable default; nanoseconds require Parquet v2.6 and break older readers. Local-time columns may be retained alongside for demo legibility but never as the canonical timestamp.
- **Integer types:** sized to the data — `int32` for MW values (GB peak ≈ 60 000), `int8` for settlement period (1–50), `int16` for year. No `int64` by default.
- **Atomic writes:** `tmp = path.with_suffix(path.suffix + ".tmp"); pq.write_table(table, tmp); os.replace(tmp, path)`. PyArrow has no built-in atomic mode; `os.replace` is the portable Python-3.3+ primitive (atomic on POSIX and NTFS).
- **Partitioning:** flat single file until the dataset crosses ~1 GB, then partition by year via `pyarrow.dataset.write_dataset`. Retrofittable without changing the public interface.
- **Multi-endpoint sources:** a single source with a list of endpoints (Stage 2's ten weather stations; Stage 17's multiple Elexon routes) still writes one canonical parquet under `<source>/`. A `station` / `resource` discriminator column identifies rows; raw per-endpoint files, if kept, are an internal staging concern and never the `fetch` return value. `load(path)` always returns the combined frame.
- **Provenance:** every written row carries `retrieved_at_utc` (§2.1.6). Per-fetch, not per-row, so byte-equal idempotence is achievable within a single run.

### 4. Schema assertion at ingest

Every module declares its raw-column expectation explicitly — either as a Pydantic model or as a `pa.schema`. On ingest:

- **Required columns missing** → hard error naming the offending column. No fallback parsing.
- **Unknown columns present** → warning and drop. Upstream schema drift (new interconnector columns, renamed fields) surfaces in logs but does not stop the demo.
- **Type mismatches** → hard error with the column name and both types.

Silent schema drift is the specific failure mode this rule prevents. NESO columns have demonstrably drifted across years (interconnectors added post-2023); Elexon and REMIT schemas drift on API version bumps. The warn-and-drop rule is the cheapest way to keep a live demo running while still leaving a trail.

### 5. Retries and fixtures

- **Retry library:** `tenacity`. Three attempts, exponential backoff (base 1s, cap 10s), retries only on `ConnectError | ReadTimeout | HTTPStatusError(5xx | 429)`. Never retries other 4xx.
- **Failure message:** on final failure, the error names the URL, the attempt count, and the last-seen status. No silent resilience — a facilitator on a flaky wifi sees what broke.
- **Fixture library:** `pytest-recording` (vcrpy under the hood). Cassettes in `tests/fixtures/<source>/cassettes/`. `--record-mode=none` in CI; `--record-mode=once` for first recording; re-recording is a deliberate developer action.
- **Sensitive headers** filtered at cassette-write time — set up now even for unauthenticated feeds so a future auth-requiring source does not leak through a re-record.

## Upgrade seams

Each of these is swappable without touching downstream code. The `fetch`/`load`/`CachePolicy` interface is what's load-bearing.

| Swappable | Load-bearing |
|-----------|--------------|
| Parquet partition layout (flat → year-partitioned) | `fetch` signature and return type (`Path`) |
| Retry library (`tenacity` → `httpx.HTTPTransport(retries=)` if it catches up) | `CachePolicy` semantics (the three values, not the class name) |
| Fixture library (`pytest-recording` → `responses` if explicit mocks become cleaner) | On-disk parquet schema (documented in each module's `CLAUDE.md`) |
| HTTP client (`httpx` → `requests`) | `load(path)` returning a tz-aware dataframe |
| `tenacity` backoff parameters | `retrieved_at_utc` provenance column |

## Module inventory

| Module | Source | Stage | Target column(s) | LLD | Status | Notes |
|--------|--------|-------|------------------|-----|--------|-----|
| `neso.py` (demand) | NESO CKAN | 1 | `ND`, `TSD` | [`lld/ingestion/neso.md`](../../lld/ingestion/neso.md) | Shipped | Sets the template. |
| `weather.py` | Open-Meteo | 2 | `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation` | [`lld/ingestion/weather.md`](../../lld/ingestion/weather.md) | Designed | First multi-endpoint ingester (ten UK stations); seeds the features layer for the national weighted aggregate. |
| `neso_forecast.py` | NESO CKAN | 4 | `FORECASTDEMAND` | — | Planning | Reuses Stage 1 scaffolding. |
| `holidays.py` | gov.uk | 5 | three-division bank holidays | — | Planning | Annual refresh cadence. |
| `remit.py` | Elexon BMRS | 13 | bi-temporal event stream | — | Planning | **Forces revisit of storage conventions** — first module with `published_at`/`effective_from`/`effective_to`. |
| `elexon_prices.py` | Elexon BMRS | 17 | MID price | — | Planning | — |
| `elexon_generation.py` | Elexon BMRS | 17 | generation by fuel | — | Planning | — |

## Open questions

- **Shared `ingestion/_common.py` — trigger now met.** Stage 1 shipped with its atomic-write, retry wrapper, rate-limit helper, and cassette harness inlined. Stage 2 repeats all four. The extraction should happen *during* Stage 2 rather than speculatively before — two concrete callers is the threshold, and the candidate symbols are visible (`_atomic_write`, `_retrying_get`, `_RetryableStatusError`, `_respect_rate_limit`, a pytest-recording cassette fixture). Stage 2 LLD §11 records the extraction plan; Stage 13 is where bi-temporal storage may force further reshaping.
- **Bi-temporal storage (Stage 13).** REMIT events carry (published_at, effective_from, effective_to); the single-timestamp convention above is insufficient. Options: extend the base schema with three nullable timestamp columns; isolate REMIT in a separate `remit` layer; treat bi-temporality as a features-layer concern. Decide at Stage 13, not now.
- **Schema-discovery mode.** CKAN `package_show` and Elexon swagger both let a client enumerate resources at runtime. Today, year → UUID mapping for NESO is hand-maintained in `conf/ingestion/neso.yaml`. An opt-in discovery mode would remove the maintenance toil but adds complexity and runtime network dependence on metadata endpoints — not currently earned.
- **Client-side throttling.** NESO's advertised 2 req/min on the datastore is unusually strict; whether enforced server-side is unclear. Stage 1 inserts a conservative inter-request delay. Open-Meteo's 600/min free-tier limit is never approached at Stage 2's ten-station scale. Whether this becomes a shared concern depends on Stage 17 (Elexon) — Elexon has no comparable documented limit.
- **Multi-endpoint fetch concurrency.** Stage 2 issues ~10 independent station requests. Serial httpx calls are fine at this scale and keep retry accounting simple; `httpx.AsyncClient` gains concurrency at the cost of an async call tree. Deferred until a source has many endpoints and a live-demo delay is observable.
- **Cache portability.** A portable cache archive (one machine seeds another) is flagged in the Stage 1 intent as a plausible follow-up but not a requirement. If it ever lands, the `CachePolicy` enum is the natural place to extend.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (layer responsibilities), [§4](../../intent/DESIGN.md#4-data-sources) (sources), [§7](../../intent/DESIGN.md#7-configuration-and-extensibility) (configuration).
- [`docs/intent/01-neso-demand-ingestion.md`](../../intent/01-neso-demand-ingestion.md), [`docs/intent/02-weather-ingestion.md`](../../intent/02-weather-ingestion.md) — concrete intents.
- [`docs/lld/ingestion/neso.md`](../../lld/ingestion/neso.md), [`docs/lld/ingestion/weather.md`](../../lld/ingestion/weather.md) — per-stage LLDs applying this architecture.
- [`docs/lld/research/01-neso-ingestion.md`](../../lld/research/01-neso-ingestion.md), [`docs/lld/research/02-weather-ingestion.md`](../../lld/research/02-weather-ingestion.md) — empirical inputs behind the claims above.
- [CKAN datastore docs](https://docs.ckan.org/en/2.9/maintaining/datastore.html); [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
- [PyArrow Parquet docs](https://arrow.apache.org/docs/python/parquet.html).
- [tenacity](https://github.com/jd/tenacity); [pytest-recording](https://pypi.org/project/pytest-recording/).
