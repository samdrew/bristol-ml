# Ingestion — layer architecture

- **Status:** Provisional — exercised by Stage 1 only. Revisit after Stage 2 (weather) introduces a second concrete caller, and again after Stage 13 (REMIT) which brings bi-temporal storage.
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (ingestion paragraph).
- **First concrete instance:** [Stage 1 LLD](../../lld/ingestion/neso.md); [research note](../../lld/research/01-neso-ingestion.md).
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
| Derived aggregates (hourly from half-hourly, weighted weather mean) | — | features layer |
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

| Module | Source | Stage | Target column(s) | LLD | Notes |
|--------|--------|-------|------------------|-----|-----|
| `neso.py` (demand) | NESO CKAN | 1 | `ND`, `TSD` | [`lld/ingestion/neso.md`](../../lld/ingestion/neso.md) | Sets the template. |
| `neso_forecast.py` | NESO CKAN | 4 | `FORECASTDEMAND` | — | Reuses Stage 1 scaffolding. |
| `weather.py` | Open-Meteo | 2 | multi-station `temperature_2m` etc. | — | Adds weighted aggregation in features layer. |
| `holidays.py` | gov.uk | 5 | three-division bank holidays | — | Annual refresh cadence. |
| `elexon_prices.py` | Elexon BMRS | 17 | MID price | — | — |
| `elexon_generation.py` | Elexon BMRS | 17 | generation by fuel | — | — |
| `remit.py` | Elexon BMRS | 13 | bi-temporal event stream | — | **Forces revisit of storage conventions** — first module with `published_at`/`effective_from`/`effective_to`. |

## Open questions

- **Shared `ingestion/_common.py`.** Seven modules will each repeat the atomic-write, retry wrapper, and cassette-fixture harness. A shared helper arrives when there are two concrete callers (Stage 2), not one. Extracting too early risks an abstraction that fits the first feed and fights every other.
- **Bi-temporal storage (Stage 13).** REMIT events carry (published_at, effective_from, effective_to); the single-timestamp convention above is insufficient. Options: extend the base schema with three nullable timestamp columns; isolate REMIT in a separate `remit` layer; treat bi-temporality as a features-layer concern. Decide at Stage 13, not now.
- **Schema-discovery mode.** CKAN `package_show` and Elexon swagger both let a client enumerate resources at runtime. Today, year → UUID mapping for NESO is hand-maintained in `conf/ingestion/neso.yaml`. An opt-in discovery mode would remove the maintenance toil but adds complexity and runtime network dependence on metadata endpoints — not currently earned.
- **Client-side throttling.** NESO's advertised 2 req/min on the datastore is unusually strict; whether enforced server-side is unclear. Stage 1 inserts a conservative inter-request delay. Whether this becomes a shared concern depends on Stage 17 (Elexon) — Elexon has no comparable documented limit.
- **Cache portability.** A portable cache archive (one machine seeds another) is flagged in the Stage 1 intent as a plausible follow-up but not a requirement. If it ever lands, the `CachePolicy` enum is the natural place to extend.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (layer responsibilities), [§4](../../intent/DESIGN.md#4-data-sources) (sources), [§7](../../intent/DESIGN.md#7-configuration-and-extensibility) (configuration).
- [`docs/intent/01-neso-demand-ingestion.md`](../../intent/01-neso-demand-ingestion.md) — the first concrete intent.
- [`docs/lld/ingestion/neso.md`](../../lld/ingestion/neso.md) — Stage 1 LLD applying this architecture.
- [`docs/lld/research/01-neso-ingestion.md`](../../lld/research/01-neso-ingestion.md) — empirical inputs behind the claims above.
- [CKAN datastore docs](https://docs.ckan.org/en/2.9/maintaining/datastore.html).
- [PyArrow Parquet docs](https://arrow.apache.org/docs/python/parquet.html).
- [tenacity](https://github.com/jd/tenacity); [pytest-recording](https://pypi.org/project/pytest-recording/).
