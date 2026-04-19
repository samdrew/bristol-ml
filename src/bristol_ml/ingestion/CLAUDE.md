# `bristol_ml.ingestion` — module guide

This module implements the ingestion layer contract. Conventions live in
[`docs/architecture/layers/ingestion.md`](../../../docs/architecture/layers/ingestion.md);
this file documents the concrete schema for each ingester under
`src/bristol_ml/ingestion/`.

## Public contract (every ingester)

```python
def fetch(config: SourceConfig, *, cache: CachePolicy = CachePolicy.AUTO) -> Path: ...
def load(path: Path) -> pd.DataFrame: ...
```

`CachePolicy` values:

- `AUTO` — use the cache if present; fetch if not. Notebook-friendly default.
- `REFRESH` — always fetch; overwrite cache atomically.
- `OFFLINE` — never touch the network; raise `CacheMissingError` if the cache
  is absent. The CI-safe choice.

Each ingester runs standalone: `python -m bristol_ml.ingestion.<source>`.

## Storage conventions

- Parquet via pyarrow; one file per source until a dataset outgrows ~1 GB.
- Timestamps stored as `timestamp[us, tz=UTC]`; local-time columns may
  accompany for demo legibility but never replace the canonical UTC column.
- Integer widths sized to the data: `int32` for MW (GB peak ~60 000),
  `int8` for settlement period (1–50), `int16` for year.
- Atomic writes via `os.replace`; a partial write leaves the previous file
  intact.
- `retrieved_at_utc` (`timestamp[us, tz=UTC]`) is written on every row as
  per-fetch provenance — equal across all rows of a single `fetch` call,
  so a refresh run is byte-equal modulo this column.

## Schema assertion (at ingest)

- Required columns missing → hard error naming the column.
- Unknown columns present → warning (UserWarning); the column is dropped
  from the persisted frame.
- Type mismatches → hard error with column name and both types.

Rationale: NESO has demonstrably added interconnector columns across years;
Elexon API versions drift. Warn-and-drop keeps the live demo running while
leaving an audit trail.

## `neso.py` — output schema

File path: `<cache_dir>/neso_demand.parquet` where
`<cache_dir>` defaults to `${BRISTOL_ML_CACHE_DIR:-./data/raw/neso}`.

| Column              | Parquet type                          | Notes                                                                   |
|---------------------|---------------------------------------|-------------------------------------------------------------------------|
| `timestamp_utc`     | `timestamp[us, tz=UTC]`               | Canonical. Start of the half-hour period, in UTC. Primary sort key.     |
| `timestamp_local`   | `timestamp[us, tz=Europe/London]`     | For demo legibility only; never used for arithmetic or joins.           |
| `settlement_date`   | `date32`                              | As reported by NESO (local settlement date).                            |
| `settlement_period` | `int8`                                | 1–50. 46 on spring-forward Sundays; 50 on autumn-fallback Sundays.      |
| `nd_mw`             | `int32`                               | National Demand in MW; excludes station load, pump storage, exports.   |
| `tsd_mw`            | `int32`                               | Transmission System Demand in MW; includes the above.                   |
| `source_year`       | `int16`                               | Which NESO annual resource supplied the row.                            |
| `retrieved_at_utc`  | `timestamp[us, tz=UTC]`               | Per-fetch provenance (§2.1.6).                                          |

Primary key: `timestamp_utc` is unique and the rows are sorted ascending by it.

### Dropped NESO columns

Stage 1 intent §"Out of scope" explicitly defers embedded generation and
interconnector flows. The following are present in the raw CKAN response
and **not** written to the parquet file; they surface a `UserWarning` the
first time a new one appears for a given year:

- `ENGLAND_WALES_DEMAND`
- `EMBEDDED_WIND_GENERATION`, `EMBEDDED_WIND_CAPACITY`
- `EMBEDDED_SOLAR_GENERATION`, `EMBEDDED_SOLAR_CAPACITY`
- `NON_BM_STOR`, `PUMP_STORAGE_PUMPING`, `SCOTTISH_TRANSFER`
- Interconnector flows (IFA, IFA2, BRITNED, NEMO, MOYLE, EAST_WEST, NSL,
  ELECLINK, VIKING, GREENLINK, …) — the exact list changes year-on-year.

The `_id` column that CKAN injects on every datastore table is suppressed
silently.

## Settlement period → UTC

Built from `(SETTLEMENT_DATE + (SETTLEMENT_PERIOD - 1) × 30 min)` then
`tz_localize("Europe/London", ambiguous=<bool>, nonexistent="raise")`:

- **Spring-forward** (last Sunday in March): the 01:00–02:00 local hour is
  skipped; the day has 46 periods. Periods 3 and 4 must not appear — if
  they do, `nonexistent="raise"` surfaces it as an error (corrupt data).
- **Autumn-fallback** (last Sunday in October): the 01:00–02:00 local hour
  occurs twice; the day has 50 periods. Period 3 is the first occurrence
  (BST, +01:00); period 4 is the second (GMT, +00:00). The `ambiguous=`
  array encodes this deterministically from the date — never from
  `"infer"` (pandas issue #47398 makes inference brittle near gaps).

## Retry + rate limit

- `tenacity`, 3 attempts, exponential backoff (base 1 s, cap 10 s).
- Retries only `httpx.ConnectError`, `httpx.ReadTimeout`, and HTTP 5xx/429.
- NESO advertises "two requests per minute" on the datastore; a
  conservative `min_inter_request_seconds` delay (default 30 s) is inserted
  between paginated calls.
- On final failure the error names the URL and attempt count.

## Shared helpers — `_common.py`

Stage 2 extracted `CachePolicy`, `CacheMissingError`, `_atomic_write`,
`_cache_path`, `_respect_rate_limit`, `_retrying_get`, and
`_RetryableStatusError` from `neso.py` into `_common.py`. The retry /
rate-limit / cache-path helpers accept any config that satisfies the
structural `Protocol` types declared in `_common` (`RetryConfig`,
`RateLimitConfig`, `CachePathConfig`) — so `NesoIngestionConfig`,
`NesoForecastIngestionConfig`, and `WeatherIngestionConfig` all work
without a shared base class. Per-source modules (`neso.py`,
`neso_forecast.py`, `weather.py`) import the helpers and re-export
`CachePolicy` / `CacheMissingError` for notebook ergonomics.

### Stage 4 follow-up

Stage 4 added `neso_forecast.py`, a second NESO-flavoured ingester
that duplicates `neso.py`'s settlement-period → UTC algebra (the
autumn-fallback / spring-forward shifts plus the ambiguity mask for
`tz_localize`). The duplication is deliberate for Stage 4 — lifting
the helpers into `_common.py` would touch the Stage 1 public surface
and is out of scope for a baseline-modelling stage. The second-caller
trigger is acknowledged here so a future refactor can factor it out
(candidate: `bristol_ml.ingestion._neso_dst` with three functions —
`_autumn_fallback_dates`, `_spring_forward_dates`, `_settlement_to_utc`).

## `weather.py` — output schema

File path: `<cache_dir>/weather.parquet` where `<cache_dir>` defaults to
`${BRISTOL_ML_CACHE_DIR:-./data/raw/weather}`. **Long-form**: one row per
station × hour. The national aggregate is computed downstream in
`bristol_ml.features.weather.national_aggregate` — never persisted by
the ingester.

| Column                | Parquet type               | Notes                                                             |
|-----------------------|----------------------------|-------------------------------------------------------------------|
| `timestamp_utc`       | `timestamp[us, tz=UTC]`    | Canonical. Open-Meteo returns UTC natively; no DST algebra.       |
| `station`             | `string`                   | Lowercase snake-case name, matches `WeatherStation.name`.         |
| `temperature_2m`      | `float32`                  | °C. Open-Meteo reports to 0.1 °C; float32 is ample.               |
| `dew_point_2m`        | `float32`                  | °C.                                                               |
| `wind_speed_10m`      | `float32`                  | km/h (API default unit).                                          |
| `cloud_cover`         | `int8`                     | %, 0–100.                                                         |
| `shortwave_radiation` | `float32`                  | W/m².                                                             |
| `retrieved_at_utc`    | `timestamp[us, tz=UTC]`    | Per-fetch provenance (§2.1.6).                                    |

Primary key: `(timestamp_utc, station)` unique; sorted by
`timestamp_utc ASC, station ASC`.

### Data model caveat

Open-Meteo's archive endpoint serves **ERA5 / ERA5-Land / CERRA** reanalyses
at ~9–11 km, **not** the UKMO UKV 2 km model claimed in DESIGN §4.2. The UKV
archive is reachable only via the separate `historical-forecast-api` and
only from 2022-03-01 — incompatible with the Stage 1 training window.
Stage 2 pins `era5_seamless` (the archive default) and defers the DESIGN
§4.2 correction to a human-approved main-session edit.

### Cache staleness

A station added to `conf/ingestion/weather.yaml` does **not** trigger a
partial re-fetch on the next `AUTO` call — the user must re-run with
`REFRESH` to rebuild the cache from all configured stations. Staleness
detection is a Stage 19 (orchestration) concern.

## Fixtures

Cassettes live at `tests/fixtures/<source>/cassettes/` via
[`pytest-recording`](https://pypi.org/project/pytest-recording/). CI runs
with `--record-mode=none`; developers re-record with `--record-mode=once`
after deleting the cassette directory. Sensitive headers
(`Authorization`, `Cookie`, `X-Api-Key`) are filtered at cassette-write
time even though none are currently required — this sets the precedent
for future authenticated feeds.

## Licence

NESO Historic Demand Data is published under NESO Open Data / OGL v3. The
licence is acknowledged in the project `README.md`; we do not replicate
licence metadata into the parquet file. Open-Meteo historical data is
released under CC BY 4.0 — citation lives in the project `README.md`
alongside the NESO acknowledgement.
