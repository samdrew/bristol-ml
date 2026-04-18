# Stage 1 research — NESO demand ingestion

Findings for the implementers of Stage 1 (`ingestion/neso.py`). Format: **Question → Finding → Citation → Implication**. Access date for all URLs: 2026-04-18.

---

## 1. NESO CKAN shape, URL pattern, columns, resource-id enumeration

**Finding.** Base URL is `https://api.neso.energy/api/3/action/`. The canonical call is `datastore_search?resource_id=<UUID>`. Each historic year is a distinct resource; 2023's UUID is `bf5ab335-9b40-4ea4-b93a-ab4af7bce003`. The 2023 CSV has 22 columns: `SETTLEMENT_DATE` (date), `SETTLEMENT_PERIOD` (int), `ND`, `TSD`, `ENGLAND_WALES_DEMAND`, `EMBEDDED_WIND_GENERATION`, `EMBEDDED_WIND_CAPACITY`, `EMBEDDED_SOLAR_GENERATION`, `EMBEDDED_SOLAR_CAPACITY`, `NON_BM_STOR`, `PUMP_STORAGE_PUMPING`, `SCOTTISH_TRANSFER`, plus nine interconnector flow columns. A direct CSV download URL also exists (`/dataset/<pkg>/resource/<uuid>/download/demanddata_YYYY.csv`). NESO's API guidance page lists `package_show` and `resource_search` as discovery endpoints, so year→UUID mapping **can** be discovered at runtime but is not published as a stable machine-readable index. The portal rate-limits: "maximum of two requests per minute" on the datastore. Columns have drifted historically (earlier years lack the newer interconnector flows); schema is not guaranteed stable across resources. Licence: NEDL / OGL v3 (stated on the data portal landing page; the individual resource pages do not repeat it).

**Citations.** NESO Historic Demand Data 2023 page, <https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2023>; NESO API guidance, <https://www.neso.energy/data-portal/api-guidance>.

**Implication.** Year→UUID mapping belongs in `conf/ingestion/neso.yaml`, not hardcoded. Schema must be asserted per-year on ingest (reject unknown columns loudly). Stage 1 loads ND and SETTLEMENT_{DATE,PERIOD} only; the interconnector flows are deferred but should not cause a crash.

---

## 2. CKAN `datastore_search` quirks (limit, offset, totals, `_id`)

**Finding.** Default `limit=100`, upper bound `ckan.datastore.search.rows_max=32000`. If a client requests more than the max, the server silently clamps and echoes the clamped value in the response `limit` field. Response shape: `{fields, records, total, total_was_estimated, limit, _links: {start, next}}`. `include_total=True` is the default; `total_estimation_threshold` lets the server return an estimated total above a threshold. `_id` is a reserved system column (auto-incremented int) present on every datastore table. Sources disagree on whether `limit=0` returns schema-only — CKAN's action code accepts `limit=0` and returns `records: []` with `fields` populated, which is the practical schema-probe idiom, but the docs do not promise this contract.

**Citations.** CKAN 2.9 datastore docs, <https://docs.ckan.org/en/2.9/maintaining/datastore.html>; CKAN action source, <https://github.com/ckan/ckan/blob/master/ckanext/datastore/logic/action.py>.

**Implication.** Paginate with `offset` until `len(records) < limit` or cumulative count reaches `total`; do not trust a single call. Use `total` (not `len(records)`) as the authoritative stop condition. Page size 32000 minimises round-trips and respects the 2-req/min limit. Do not rely on `limit=0` schema-probe; use `resource_show` for metadata instead.

---

## 3. GB settlement periods on BST transition days

**Finding.** Confirmed: 46 periods on spring-forward (the 01:00–02:00 hour is skipped), 50 on autumn-fallback (the 01:00–02:00 hour occurs twice). Elexon official guidance for 2025 is explicit: "the Settlement Periods in the contract notification should be numbered from 1 to 46" (spring) and "1 to 50" (autumn). For 2026 the transitions fall on Sunday 29 March and Sunday 25 October (last Sundays; confirmed by Wikipedia BST article against UK Summer Time Act 1972). Mapping to wall-clock: period N covers the half-hour beginning at `00:00 + 30min*(N-1)` in local time. Converting (date, period) → UTC requires localising to `Europe/London` with `nonexistent='raise'` (for data integrity — no valid data should carry spring-forward periods) and `ambiguous` handled via the natural sequencing of periods (period 3 = first 01:00 BST hour; period 4 = second 01:00 GMT hour on autumn day).

**Citations.** Elexon spring guidance 2025, <https://www.elexon.co.uk/2025/03/28/treatment-of-volume-notifications-on-the-short-clock-change-day/>; Elexon autumn guidance 2025, <https://www.elexon.co.uk/bsc/event/volume-notification-on-long-clock-change-day/>; pandas `tz_localize` reference, <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.tz_localize.html>.

**Implication.** In pandas 2.x, the robust pattern is: build naive local timestamps from `(SETTLEMENT_DATE + (SETTLEMENT_PERIOD-1) * 30min)`, then `tz_localize("Europe/London", ambiguous=<bool_array>, nonexistent='raise')`, then `tz_convert("UTC")`. The `ambiguous` bool array is deterministic: `period in (3, 4)` on autumn day, else False. `ambiguous='infer'` is brittle near missing data (pandas issue #47398) — do not use it. Spring-forward periods 3–4 should never appear in real NESO data; assert this in the loader.

---

## 4. Parquet conventions for tz-aware timestamps, IDs, demand

**Finding.** Parquet stores timestamps as INT64 plus a logical type carrying `isAdjustedToUTC` and `timeUnit`. PyArrow's idiomatic type is `pa.timestamp('us', tz='UTC')` — this sets `isAdjustedToUTC=true` and gives microsecond resolution. Nanosecond resolution requires Parquet v2.6 and breaks older readers; microseconds are the portable default. Integer demand in MW fits in int32 (GB peaks ~60 GW = 60000 MW). `os.replace` is the Python-3.3+ atomic rename primitive; the canonical pattern is to write to a temp path in the same directory then `os.replace(tmp, final)`. PyArrow's `write_table` has no built-in atomic mode. Sources agree partitioning by year is the idiomatic Hive-style layout for time-series (`year=YYYY/part-0.parquet`) via `pyarrow.dataset.write_dataset`, and agree that for a single-digit-GB dataset (25 years × 17520 half-hours × ~20 cols is tiny) a single file is also defensible; they disagree on whether partitioning earns its keep below ~1 GB.

**Citations.** PyArrow Parquet docs, <https://arrow.apache.org/docs/python/parquet.html>; PyArrow timestamps, <https://arrow.apache.org/docs/python/timestamps.html>; `pyarrow.parquet.write_table`, <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html>; Python `os.replace`, <https://docs.python.org/3/library/os.html#os.replace>.

**Implication.** Use `pa.timestamp('us', tz='UTC')` for timestamp columns and int32 for MW values. Implement atomic-write as `tmp = path.with_suffix(".parquet.tmp"); pq.write_table(table, tmp); os.replace(tmp, path)`. For Stage 1 a single `data/raw/neso_demand.parquet` is sufficient; partitioning by year is a cheap retrofit if Stage 13+ adds write volume.

---

## 5. HTTP fixture recording library

**Finding.** Four live options: (a) `vcrpy` — records all supported HTTP libs (requests, urllib3, httpx, aiohttp, tornado) to YAML cassettes; filters sensitive headers/params; ~3 MiB wheel. (b) `pytest-recording` — thin pytest wrapper around vcrpy, exposes `@pytest.mark.vcr` and `--record-mode=<none|once|new_episodes|all>`. (c) `responses` — requests-only, explicit mock definitions, no recording. (d) hand-rolled JSON fixtures — zero deps, maximal control, highest maintenance. Consensus: vcrpy+pytest-recording is the idiomatic pytest-native recording story; `responses` is the idiom if you want explicit mocks without touching the network. Known trade-off: YAML cassettes for large responses bloat the repo (a full year of NESO CSV is ~1.5 MiB JSON-encoded); vcrpy supports response-body filtering and gzip to mitigate.

**Citations.** vcrpy project, <https://github.com/kevin1024/vcrpy>; pytest-recording, <https://pypi.org/project/pytest-recording/>; responses, <https://github.com/getsentry/responses>.

**Implication.** `pytest-recording` fits Stage 1's shape (pytest-native, httpx support, record-once ergonomics via `--record-mode=once`). Record a narrow slice (e.g. one year, one offset page, plus `package_show` for schema discovery) to keep cassette size under a few hundred kB; filter query params that would otherwise churn the cassette.

---

## 6. Retry strategy

**Finding.** httpx's built-in `HTTPTransport(retries=N)` retries **only** `ConnectError` and `ConnectTimeout`. It does not retry read/write errors, does not retry 429/503, and does not implement backoff or jitter. For those, the primary-source guidance in the httpx docs themselves points users at `tenacity`. Third-party middlewares (`httpx-retries`, successor to the now-archived `httpx-retry`) add backoff+status-retry at the transport layer, but they are not part of httpx proper. Sources agree tenacity is still the default for anything beyond connection flakes; sources disagree on whether transport-layer retry or decorator-style retry is more idiomatic for small codebases.

**Citations.** HTTPX transports docs, <https://www.python-httpx.org/advanced/transports/>; httpx PR #778 adding retries, <https://github.com/encode/httpx/pull/778>; tenacity, <https://github.com/jd/tenacity>.

**Implication.** For a demo context, "fail loudly with a clear message" beats silent resilience. A bounded tenacity wrapper (3 attempts, exponential backoff base 1s cap 10s, retry on `ConnectError|ReadTimeout|HTTPStatusError(5xx|429)` only) is conventional and keeps the failure mode visible. Do not retry 4xx other than 429.

---

## 7. ND vs TSD — benchmark target

**Finding.** The NESO Day Ahead National Demand Forecast resource page is unambiguous: its `FORECASTDEMAND` column predicts National Demand (ND), defined as "the Great Britain generation requirement" excluding station load, pump storage pumping, and interconnector exports. The separate "Day Ahead Demand Forecast" (TSD-based) product exists but is not the primary day-ahead demand-forecast product linked from the 1-day-ahead landing page. Sources agree ND is the benchmark target; no source contradicts this.

**Citations.** NESO Day Ahead National Demand Forecast, <https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/day_ahead_national_demand_forecast>; NESO Demand Data Update (ND/TSD definitions), <https://www.neso.energy/data-portal/daily-demand-update/demand_data_update>.

**Implication.** Stage 1 must load ND as the primary target column (matches DESIGN.md §5.1 and Stage 4 comparability). TSD should be loaded alongside but marked secondary. Embedded wind/solar columns are not loaded in Stage 1 (deferred per stage-intent §"Out of scope"), but the loader must tolerate their presence/absence across years.

---

## Areas where sources are sparse or disagree

1. **`limit=0` as a schema probe.** Works in practice; not documented as contract. Use `resource_show` instead.
2. **Partition granularity for small datasets.** Arrow docs lean "partition when it pays"; community blog posts lean "partition by year always". For <1 GB this is aesthetic.
3. **Schema stability across NESO year resources.** No NESO-published changelog. Columns demonstrably changed as interconnectors came online (e.g. VIKING, GREENLINK post-2023). Stage 1 must assert-and-surface unknown columns, not silently drop them.
4. **Rate limit interpretation.** The API guidance says "two requests per minute" for the datastore, which is unusually strict. Sources do not clarify whether this is advisory or enforced. Conservative client-side throttling is safer than assuming a 429 will come back cleanly.
