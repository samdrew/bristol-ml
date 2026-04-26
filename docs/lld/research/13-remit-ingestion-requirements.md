# Stage 13 — REMIT ingestion: requirements

## Goal

Ingest REMIT regulatory-disclosure messages from the Elexon Insights API, persist them in a bi-temporal store that supports correct "as-of" queries, and produce a notebook visualisation of unavailable-capacity density over time by fuel type.

## User stories

**US-1 — Notebook facilitator**
Given a warm local cache of REMIT messages,
When the facilitator opens the visualisation notebook,
Then they see a chart of aggregate unavailable capacity over time, coloured by fuel type, with visible spikes at planned outage seasons and identifiable step-changes at unplanned events, without any network call.

**US-2 — Backtest engineer**
Given REMIT messages ingested with bi-temporal metadata,
When the engineer queries "what did the market know about outages at time T?",
Then the module returns exactly the set of messages that had been published on or before T and whose event window overlapped T, honouring all revisions and withdrawals that were known at T.

**US-3 — Stage 14 LLM extractor**
Given Stage 13 has run and populated its cache,
When Stage 14 reads the persisted store,
Then it can iterate over raw REMIT records including the free-text description field without re-fetching from Elexon, and each record carries the three temporal fields (`published_at`, `effective_from`, `effective_to`) alongside the structured fields (asset, type, MW).

**US-4 — Stage 16 feature-join consumer**
Given Stage 13 has ingested a range covering the training window,
When Stage 16 joins REMIT features onto the hourly feature table,
Then it can resolve aggregate unavailable MW at any hour in the training window using a bi-temporal index, without ambiguity about which message revision was active at that hour.

**US-5 — Developer / CI runner**
Given a recorded API fixture is present,
When the test suite runs with no network access,
Then all REMIT ingestion and bi-temporal query tests pass using the cassette, and the stub path never calls the Elexon API.

**US-6 — First-time attendee (offline clone)**
Given a curated sample cache is seeded alongside the repo (or via a documented one-off script),
When the attendee runs the notebook without internet access,
Then the visualisation renders correctly from the sample cache without modification.

## Acceptance criteria

**AC-1 — As-of query correctness.**
The module exposes a query that accepts a timestamp T and returns all REMIT events whose `published_at` ≤ T, including the correct revision state of any message that was revised or withdrawn before T. A test using recorded fixture data must cover: (a) a message with no revisions, (b) a message that was revised after an initial publication, and (c) a message that was withdrawn.

**AC-2 — Offline-first with cache present.**
Running the ingestion module with a warm cache completes without any network call. The `CachePolicy.OFFLINE` path must be exercised by at least one test.

**AC-3 — Fetch-and-persist without cache.**
Running without a cache fetches from the Elexon Insights `/remit` endpoint and persists the result locally. Subsequent runs in `OFFLINE` mode succeed and produce the same data.

**AC-4 — Idempotence.**
Running the ingestion twice in a row over the same date range produces the same on-disk result. Rerunning must not duplicate records.

**AC-5 — Three temporal fields per record.**
Every persisted record carries at minimum: `published_at` (when the market participant published the message), `effective_from` (start of the event window), `effective_to` (end of the event window), and `retrieved_at_utc` (when this project fetched the message). All four are timezone-aware UTC timestamps.

**AC-6 — Revision and supersede handling.**
The store preserves all revisions of a message (not only the latest). A "latest as-of T" view can be derived from the store; the store itself is append-only for received messages.

**AC-7 — Structured fields preserved.**
Each record preserves the structured Elexon fields — at minimum: asset identifier, event type, affected capacity in MW, `effective_from`, `effective_to`, `published_at` — plus the raw free-text description (unmodified) for Stage 14 consumption.

**AC-8 — Schema documented.**
The output parquet schema (column names, types, and semantics) is documented in the module's `CLAUDE.md`, following the pattern established in `src/bristol_ml/ingestion/CLAUDE.md`.

**AC-9 — Visualisation notebook.**
The notebook renders a time-series chart of aggregate unavailable capacity coloured by fuel type. It runs top-to-bottom from a warm cache in a time consistent with the project's notebook runtime budget, without reimplementing logic from the module (§2.1.8).

**AC-10 — Tests use recorded fixtures.**
All tests that exercise fetch behaviour use recorded cassettes, not live Elexon calls. At minimum: one cassette covering a small date window with at least one revised message.

**AC-11 — Module standalone.**
`python -m bristol_ml.ingestion.remit` runs without error and prints a meaningful summary (e.g. record count and date range from the cache), consistent with the `CachePolicy`-based CLI pattern in Stages 1, 2, and 5.

## Non-functional requirements

**NFR-1 — Idempotent ingestion (§2.1.5).**
Re-running over an already-cached window must not produce duplicate rows. The append-only revision store must be written atomically (following the `_atomic_write` pattern from `ingestion/_common.py`).

**NFR-2 — Stub-first (§2.1.3).**
A stub path must be available for CI and for attendees without Elexon access. The stub must satisfy the same public interface as the live fetcher, returning a structurally valid parquet with the same schema. The stub is the CI default.

**NFR-3 — Paging and rate-limit discipline.**
The Elexon `/remit` endpoint paginates. The fetcher must handle paging reliably, with configurable inter-request throttling using the shared `_respect_rate_limit` / `_retrying_get` helpers from `ingestion/_common.py`. Reliability is more important than speed; a slow, complete fetch is better than a fast, partial one.

**NFR-4 — Parquet at the storage boundary (§2.1.2).**
Persisted data must be Parquet (pyarrow), consistent with every other ingestion stage. Column types must be pinned; timestamps must be `timestamp[us, tz=UTC]`.

**NFR-5 — Cache outside the repo.**
REMIT history may be large. The cache directory must default to a path outside the repo (consistent with the `${oc.env:BRISTOL_ML_CACHE_DIR,...}` pattern from `conf/ingestion/neso.yaml`) and must be documented as gitignored. A curated sample fixture for tests and demos must be small enough to commit.

**NFR-6 — Offline-after-first-fetch.**
Once the cache is populated, every downstream consumer (notebook, Stage 14, Stage 16) must be able to run without network access. `CachePolicy.OFFLINE` must raise `CacheMissingError` (from `_common`) rather than silently returning empty data.

**NFR-7 — UTC timestamps throughout.**
All timestamps stored and returned by the module are timezone-aware UTC, following the convention established at Stage 1. Local-time representations are not stored.

**NFR-8 — Configuration in YAML (§2.1.4).**
API base URL, default date range, paging parameters, rate-limit settings, and cache directory must all live in `conf/ingestion/remit.yaml`, validated by a new Pydantic schema. No values hard-coded in the module.

**NFR-9 — Provenance (§2.1.6).**
Each persisted batch must record a `retrieved_at_utc` scalar alongside the data, consistent with the pattern in `neso.py` and `weather.py`.

**NFR-10 — Observability.**
The module must emit structured `loguru` log lines at INFO level for each paging step (record count, window, page number) and at WARNING level for any revision collision or unexpected message state, consistent with the logging style in `features/calendar.py`.

## Open questions

**OQ-1 — What constitutes "one row" in the store: one row per message-ID or one row per (message-ID, revision)?**
The intent notes that a single outage may generate an initial message plus revisions plus a withdrawal. Treating each API response as a new row (append-only, each with its own `retrieved_at_utc`) supports as-of queries naturally but produces a wide revision fan. Treating only the latest revision as a row gives a simpler store but loses historical revision state needed for correct backtesting.
*Default if unresolved:* one row per fetched message occurrence (append-only). Each row has a unique `(message_id, published_at)` key. A derived "latest as-of" view is a query, not a storage design.

**OQ-2 — What archive depth should the default fetch cover?**
REMIT goes back to 2014. The primary demand model uses 2018 onwards; price modelling (Stage 17) might benefit from earlier history. A full archive fetch for CI or demo setup is impractical.
*Default if unresolved:* default window matches the demand training window: 2018-01-01 to present. Older history is fetchable via a Hydra CLI override; no automated backfill.

**OQ-3 — How should the test cassette be scoped to remain small and cover revision semantics?**
A cassette covering the full archive is too large to commit. A narrow window may not contain any revised messages naturally.
*Default if unresolved:* one cassette covering a single week known (from manual inspection) to contain at least one revised message, plus one synthetic fixture record injected at test-setup time to cover the withdrawal case deterministically.

**OQ-4 — Should asset identifiers be normalised against a plant master-data source?**
The intent explicitly defers plant-master normalisation to "out of scope." However, Stage 16's feature join and the notebook's "show me Hinkley Point B" use case both benefit from normalised IDs.
*Default if unresolved:* store the raw Elexon asset identifier string as-is. No normalisation at Stage 13. A `plant_id_normalised` column may be added by Stage 16 if needed.

**OQ-5 — Where does the bi-temporal index live — embedded in the parquet schema or as a separate index structure?**
Parquet supports predicate pushdown on timestamp columns, which handles as-of filtering efficiently without a separate index. A separate index (e.g. an interval tree in memory) is faster for repeated queries but adds complexity.
*Default if unresolved:* bi-temporal metadata lives as plain timestamp columns in parquet (`published_at`, `effective_from`, `effective_to`, `retrieved_at_utc`). As-of filtering is done via pandas predicates at query time. No separate index structure at Stage 13.

**OQ-6 — Should "effective_to" be nullable, and what sentinel represents an open-ended event?**
Some REMIT messages describe ongoing outages with no declared end date. The column must handle this without silently excluding the record from as-of queries that would otherwise include it.
*Default if unresolved:* `effective_to` is nullable (`timestamp[us, tz=UTC]`, `None` = open-ended). As-of query logic treats `NULL` effective_to as "still active." A test must cover this case explicitly.

**OQ-7 — What Hydra group structure should the REMIT config occupy — under `ingestion/` alongside `neso.yaml` and `weather.yaml`, or a standalone group?**
All prior ingestion stages place their config under `conf/ingestion/`. A separate group (e.g. `conf/remit/`) is possible but breaks the established layer symmetry.
*Default if unresolved:* `conf/ingestion/remit.yaml` under the existing `ingestion/` group, with a matching `IngestionGroup.remit: RemitIngestionConfig | None = None` field on the Pydantic schema, consistent with the `holidays` field added at Stage 5.

**OQ-8 — Should the module emit a separate "events" view (one row per event window, de-duplicated) alongside the raw message store, or leave that entirely to Stage 16?**
A de-duplicated events view is useful for the notebook and for Stage 14's text extraction, but implementing it at Stage 13 risks pre-empting Stage 16's join design.
*Default if unresolved:* Stage 13 persists only the raw message log. No derived events view. The notebook queries the raw store directly with a simple "latest published_at per message_id" groupby for visualisation purposes.

---

*This artefact is one of four Phase-1 research inputs for Stage 13. It covers requirements only; API surface detail is in the domain-researcher artefact; existing code patterns are in the codebase-explorer artefact; scope boundaries are in the scope-diff artefact.*
