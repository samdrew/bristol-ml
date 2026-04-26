# Stage 13 — REMIT bi-temporal ingestion

## Goal

Land the project's first **bi-temporal** ingester: REMIT (Remit Energy
Market Integrity and Transparency) outage and unavailability messages
from the Elexon Insights API. Three targets, in priority order:

1. Persist REMIT messages with all three publish-axis times preserved
   (`published_at` for transaction-time, `effective_from` /
   `effective_to` for valid-time) plus the standard `retrieved_at_utc`
   project-axis provenance scalar.
2. Ship the new public primitive `as_of(df, t)` — the transaction-time
   query that answers "what did the market know at time `t`?". This is
   the contract Stages 14 and 16 will consume.
3. Demo the storage-shape mechanic on a single chart: declared
   unavailable capacity by fuel type, month-over-month, where a
   facilitator can point at a spike and say "that was a nuclear unit
   going offline on this date" (intent §Demo moment).

## What was built

- `src/bristol_ml/ingestion/remit.py` — single-file ingester following
  the `holidays.py` structural template (single endpoint, single
  parquet file, simple parser). Public surface: `OUTPUT_SCHEMA`
  (16-column pyarrow schema), `MESSAGE_STATUSES` / `FUEL_TYPES`
  constants, `fetch(config, *, cache=CachePolicy.AUTO) -> Path`,
  `load(path) -> pd.DataFrame`, and the new primitive
  `as_of(df, t) -> pd.DataFrame`. `_common.py` helpers
  (`_atomic_write`, `_cache_path`, `_respect_rate_limit`,
  `_retrying_get`, `CachePolicy`, `CacheMissingError`) reused verbatim
  via the structural-Protocol contract on `RemitIngestionConfig`. Live
  fetch path issues a single GET against
  `/datasets/REMIT/stream` (no paging — Elexon serves the whole
  window in one response under the streaming endpoint; domain research
  §R3) and parses each message via a 15-field camelCase →
  snake_case mapping (`eventStatus → message_status`,
  `unavailableCapacity → affected_mw`, etc.). loguru INFO record per
  fetch carries the record count and window slice (NFR-10); WARNING
  on unknown `message_status` values. Module CLI
  `python -m bristol_ml.ingestion.remit --help` prints the resolved
  `RemitIngestionConfig` schema.
- `as_of(df, t)` algorithm: filter `published_at <= t`, group by
  `mrid` and keep the row with the maximum `revision_number`, drop
  rows whose `message_status == "Withdrawn"`. Naive timestamps raise
  `ValueError`. Strictly transaction-time — callers wanting "events
  active at `t`" (valid-time) chain a second filter on
  `effective_from` / `effective_to`. The two-step decomposition is
  the standard bi-temporal pattern and keeps each filter testable in
  isolation.
- `conf/_schemas.py` — new `RemitIngestionConfig` Pydantic model (12
  fields covering API endpoint, window, retry / rate-limit Protocol
  fields, and cache location). `IngestionGroup.remit:
  RemitIngestionConfig | None = None` slot added alongside the four
  existing ingester slots — the `None` default means every prior
  ingester continues to validate without change.
- `conf/ingestion/remit.yaml` — Hydra group file mirroring the schema
  defaults; `conf/config.yaml` `defaults:` list extended with
  `- ingestion/remit@ingestion.remit`.
- Stub mode (`BRISTOL_ML_REMIT_STUB=1`) — `_stub_records()` produces
  10 hand-crafted records spanning seven mRIDs across 2024-01-01 …
  2024-07-01, exercising all four AC-1 cases (fresh / revised /
  withdrawn / open-ended `effective_to`). The stub casts through the
  same `OUTPUT_SCHEMA` as a live fetch, so the notebook + tests run an
  identical code path with deterministic offline data.
- `tests/fixtures/remit/cassettes/remit_2024_01_01.yaml` — single VCR
  cassette covering one day (2024-01-01 → 2024-01-02) of
  `/datasets/REMIT/stream`. Observed payload: ~125 records / 70 mRIDs
  / 31 in-window revision chains. **Cassette size: 20 220 bytes
  (~20 kB)**, comfortably inside the plan's 100 kB budget. The
  one-day window was chosen over the plan's "one-week window"
  drafting (D11) because the live API's record density (~125
  records/day) would have produced a ~1.8 MB raw payload over a week,
  exceeding the budget.
- Tests:
  - 13 unit tests at `tests/unit/ingestion/test_remit.py` — eight
    T2 tests (the four AC-1 `as_of` cases + naive-timestamp guard +
    `OUTPUT_SCHEMA` pinning + module/standalone-CLI smoke), plus
    five T3 tests (stub-fetch round-trip, four-timestamp `tz=UTC`
    round-trip, open-ended `effective_to` as `pd.NaT`,
    `CachePolicy.OFFLINE` raises `CacheMissingError`, and
    `python -m bristol_ml.ingestion.remit` exits 0).
  - Four integration tests at
    `tests/integration/ingestion/test_remit_cassettes.py` — cassette
    populates cache (AC-3), idempotent re-fetch row-identical
    modulo `retrieved_at_utc` (AC-4 / NFR-1), `as_of` invariants at
    three sample times across the cassette window (AC-1 at
    integration), and loguru INFO record carrying the record count
    + window slice (NFR-10).
  - One notebook smoke test at `tests/integration/test_notebook_13.py`
    — `nbconvert --execute` round-trip under
    `BRISTOL_ML_REMIT_STUB=1` asserting cells T5 Cell 1 / 3 / 5 each
    produce non-empty output (AC-9).
- `notebooks/13_remit_ingestion.ipynb` — seven-cell demo notebook
  (4 markdown + 3 code cells; plan §5 specified a three-cell minimum
  for the load / compute / plot core, the markdown cells provide
  narrative coherence). Generated programmatically from
  `scripts/_build_notebook_13.py` to keep cell source under version
  control as readable text. End-to-end execution under stub mode
  (`BRISTOL_ML_REMIT_STUB=1`): ~5 seconds — well inside any informal
  budget. The chart at Cell 5 is the demo moment: stacked-area MW by
  fuel type, month-start as-of.
- `docs/architecture/layers/ingestion.md` — flipped from "Provisional
  — revisit after Stage 13" to "Stable for two storage shapes". New
  §"Bi-temporal storage shape" section documents the four-timestamp
  model, the append-only revision log, the `as_of` algorithm, and a
  worked "published / revised / withdrawn" example. Module inventory
  row for `remit.py` flipped from Planning to Shipped. The "bi-temporal
  storage (Stage 13)" open question marked resolved with a
  back-reference to the new section.
- `src/bristol_ml/ingestion/CLAUDE.md` — REMIT row added to the
  schema-table family alongside `neso.py` / `weather.py` /
  `holidays.py`; new "as-of query" subsection explains the
  primitive's contract and points at the layer doc for the worked
  example. Stub-mode and cassette-scope notes record the
  `BRISTOL_ML_REMIT_STUB=1` env-var contract and the 20 kB cassette
  size with rationale.
- `README.md` — new "Worked example: REMIT bi-temporal ingestion"
  paragraph between Stage 12 and Stage 5, linking the layer doc, the
  module guide, and this retrospective.
- `CHANGELOG.md` — `### Added` bullet under `[Unreleased]` covering
  the new module, the `as_of` primitive, the cassette + stub fixtures,
  the notebook, and the layer-doc resolution.
- Plan moved from `docs/plans/active/13-remit-ingestion.md` to
  `docs/plans/completed/13-remit-ingestion.md` in the final hygiene
  commit.

## Design choices made here

- **Append-only `(mrid, revision_number)` storage, not "latest
  wins".** The intent's hardest constraint is bi-temporal correctness
  in the face of revisions and withdrawals. A "latest revision wins"
  ingest-time overwrite would silently leak future information at any
  query whose `t` is earlier than the latest revision's
  `published_at`. Append-only storage with a transaction-time
  predicate at query time is the smallest representation that gets
  this right; the cost is one column (`revision_number`) on disk, and
  the win is that the failure mode is impossible by construction.
- **`as_of` is a pure pandas function, not a class or an indexed
  store.** Plan D13. The full algorithm fits in three lines of pandas
  (transaction-time filter + groupby max-revision + Withdrawn drop),
  has no state to carry between calls, and is trivially testable on
  in-memory frames without any fixture machinery. The unit tests
  cover all four AC-1 cases plus the naive-timestamp guard before any
  fetch path is wired (T2 ships before T3/T4) — the bi-temporal
  teaching point lands first.
- **Single endpoint (`/datasets/REMIT/stream`); no paged fallback, no
  per-mRID `revisions/{mrid}` call.** Plan D4 (Scope Diff). The
  streaming endpoint has no observed window cap and returns
  `revisionNumber` per row, so revision chains are reconstructable
  from a single GET. Adding a second endpoint would mean a second
  cassette and a second test path for no acceptance criterion the
  stream cannot already cover.
- **Row-level `fuel_type`, no separate BMU reference table.** Plan
  D15 (Scope Diff) — the single highest-leverage cut from the
  `@minimalist` critique. Domain research §R6 confirmed `fuelType` is
  a first-class field on every `/datasets/REMIT/stream` row, so the
  notebook's "MW by fuel type" aggregation reads the row's own column
  rather than joining against a reference fetch. Cut eliminates a new
  module (`_bmu_reference.py`), a new cache file, a new Hydra config
  slot, a new CI stub, and at least two tests — all in one decision.
- **Cassette window: one day, not one week.** Plan D11 drafted a
  one-week cassette targeting ≤ 100 kB compressed. The live API's
  observed record density (~125 records/day, confirmed during
  recording) would have produced a ~1.8 MB raw payload across a week
  even after VCR's body filtering — well over budget. A one-day
  window still exhibits the necessary revision behaviour (31 multi-revision
  mRIDs in the recorded day) so the cassette tests
  `as_of` against genuine revision chains rather than synthetic ones.
  The withdrawal case stays at the unit level via
  `test_as_of_withdrawn_message_excludes_row`.
- **No formal ADR for bi-temporal storage.** Plan D17 (Scope Diff —
  defer). The design is captured in the layer doc and the
  requirements artefact; an ADR would add maintenance burden and
  signal that the choice was a coin-flip when in fact the four-column
  append-only model is the simplest thing that satisfies the intent.
  Promote to a formal ADR only if a future stage finds the choice
  contested.
- **Stub-and-cassette stay decoupled.** Plan OQ-B. The stub's 10
  hand-crafted records cover the four AC-1 cases (fresh / revised /
  withdrawn / open-ended); the cassette covers the live HTTP /
  parser / cache-write path. Trying to unify them — e.g. by deriving
  the stub records from the cassette payload — would couple two
  orthogonal concerns and make the stub records harder to reason
  about as exhaustive AC-1 coverage.

## Demo moment

From a clean clone (Stages 0–12 already built):

```bash
uv sync --group dev
uv run pytest -q                                                 # all green
uv run python -m bristol_ml.ingestion.remit --help               # prints RemitIngestionConfig
BRISTOL_ML_REMIT_STUB=1 uv run jupyter nbconvert --to notebook \
    --execute notebooks/13_remit_ingestion.ipynb \
    --output /tmp/13_test_run.ipynb                              # ~5 s, offline
```

The notebook's Cell 5 stacked-area chart is the demo moment: monthly
declared unavailable MW by fuel type, with each spike traceable to a
specific outage event in the underlying frame. A facilitator can point
at the chart and say "that was a nuclear unit going offline on this
date", then drop into the frame in Cell 3 and confirm via
`monthly_long.query(...)`.

A live (non-stub) run against the warm cassette / archive cache
produces a far denser chart over the same code path.

## Observations from execution

- **Live API field naming differs from domain research draft.**
  Domain research §R2 transcribed the field as `messageStatus` based
  on swagger documentation; the live `/datasets/REMIT/stream` payload
  emits `eventStatus` (and the values are `Active` / `Inactive` /
  `Dismissed`, not `Active` / `Withdrawn`). The implementation maps
  `eventStatus → message_status` and accepts the four observed status
  strings; `Withdrawn` is reachable only via the synthetic test fixture
  but is necessary because the algorithm contract drops it.
- **`datetime.fromisoformat` handles the `Z` suffix natively in Python
  3.12.** The Elexon timestamps come back as `2024-01-01T23:54:02Z`;
  no manual `replace("Z", "+00:00")` is required.
- **31 in-window revision chains in a single day.** Higher than
  expected — the cassette therefore exercises `as_of` against real
  revision behaviour, not just degenerate single-revision rows.
- **`cache_dir` default deviates from the plan, in line with the
  layer convention.** Plan §NFR-5 wrote
  `${oc.env:BRISTOL_ML_CACHE_DIR,~/.cache/bristol_ml}`; the shipped
  `conf/ingestion/remit.yaml` writes
  `${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/remit}`, matching the
  pattern every other ingester (NESO, weather, holidays,
  neso_forecast) already uses.  The plan clause was a one-off
  carry-over from a generic codebase-map example, not a deliberate
  choice to break the layer convention; the implementation honours
  the convention so cassette-vs-fresh-fetch behaviour is uniform
  across ingesters and the developer cache stays project-scoped
  (alongside `data/raw/neso/`, etc.).  `data/` is already gitignored
  with a pinned `.gitkeep`, so NFR-5 ("cache outside the repo's
  tracked files") is preserved.  The Phase 3 `arch-reviewer` flagged
  the divergence as M-3; on inspection it is a deliberate
  good-deviation and the plan's clause is what should have read
  `data/raw/remit`.  No code change required.
- **Phase 3 review fixes landed in-branch.**  `arch-reviewer` and
  `code-reviewer` together raised 13 items; on a fix-or-reject
  triage (no silent deferral), the three Blocking items (B-1
  data-shape ADR, B-2 offline-cache test, B-3 plan-mismatch parse
  count) and four Major code defects (R1 `date.today()`-vs-UTC
  default, R2 missing-required-field `KeyError` regression, R3
  unsafe URL string concatenation, N2 `RuntimeError`-vs-typed
  payload error) were fixed against tightened structural guards.
  M-1 was strengthened to assert exact `(9, 64, 70)` mRID counts
  with a strict-subset chain across the three sample times.  M-2
  was the missing-from-`load`-section `as_of` definition, fixed by
  reordering its source-file position so the algorithm sits next
  to its consumer.  N1 was a notebook-test cleanup hygiene gap —
  fixed via a `try/finally`-guarded executed-copy with a
  session-scoped sweeper for orphaned artefacts.  Items rejected
  with reasoning rather than fixed are recorded in the PR
  description.

## Deferred

- **Asset-identifier normalisation against a plant master.** Intent
  §Out of scope. The persisted `affected_unit` / `asset_id` columns
  carry whatever Elexon publishes; downstream stages that want
  station-level aggregation will have to thread their own join.
- **Free-text extraction from `message_description`.** Stage 14 owns
  this — the column is persisted intact for the LLM extractor to
  consume.
- **REMIT in the feature table.** Stage 16 owns the join from this
  bi-temporal frame into the modelling feature table; the `as_of`
  primitive is exactly what guarantees that join uses only information
  available at training time, no leakage.
- **Embedding / semantic search across REMIT messages.** Stage 15
  owns this — consumes Stage 14, not Stage 13 directly.
- **Derived "events" view (one row per event window, de-duplicated).**
  Plan OQ-8 default — deferred to Stage 16's join design. The current
  storage shape preserves every revision and the notebook composes
  the event view at read time; persisting a second derived view
  would split the source of truth.
- **Plant-station-level grouping.** Domain research §R6 confirmed the
  API has no station-level grouping primitive; deferred indefinitely.

## Next

→ Stage 14: free-text extraction from REMIT `message_description` via
LLM, producing a structured-fields companion frame keyed on
`(mrid, revision_number)`.
