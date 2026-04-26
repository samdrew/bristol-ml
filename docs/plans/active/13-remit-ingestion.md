# Plan — Stage 13: REMIT ingestion

**Status:** `approved` — Ctrl+G review on 2026-04-26 accepted all 21 decisions and all four open-question defaults; OQ-D (DESIGN §6 flip) explicitly removed from scope. Ready for Phase 2.
**Intent:** [`docs/intent/13-remit-ingestion.md`](../../intent/13-remit-ingestion.md)
**Upstream stages shipped:** Stages 0–12 (foundation → ingestion → features → six model families → enhanced evaluation → registry → MLP → TCN → serving). Stage 13 depends only on the ingestion-layer conventions established at Stages 1, 2, and 5; it does not touch any modelling stage.
**Downstream consumers:** Stage 14 (LLM extractor — consumes the persisted free-text descriptions), Stage 16 (model with REMIT — joins bi-temporal events into the feature table). Stage 15 (embedding index) consumes Stage 14, not Stage 13 directly.
**Baseline SHA:** `b947006` (tip of `main` after Stage 12 merge).

**Discovery artefacts produced in Phase 1:**

- Requirements — [`docs/lld/research/13-remit-ingestion-requirements.md`](../../lld/research/13-remit-ingestion-requirements.md)
- Codebase map — [`docs/lld/research/13-remit-ingestion-codebase.md`](../../lld/research/13-remit-ingestion-codebase.md)
- Domain research — [`docs/lld/research/13-remit-ingestion-domain.md`](../../lld/research/13-remit-ingestion-domain.md)
- Scope Diff — [`docs/lld/research/13-remit-ingestion-scope-diff.md`](../../lld/research/13-remit-ingestion-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §Demo moment names a single moment: a chart of aggregate unavailable capacity over time, by fuel type, where a facilitator can point at a spike and say "that was a nuclear unit going offline on this date." Everything else — the bi-temporal storage, the as-of query API, the offline cache discipline — is plumbing in service of that one moment, plus the contract Stages 14 and 16 will consume. Every decision below is filtered through that lens.

**Bi-temporal weight.** The intent's hardest constraint is that the module must answer "what did the market know at time T?" *correctly*, including revisions and withdrawals. This is not a "log the latest message" pipeline — it is an append-only bi-temporal store with three publish-axis timestamps (`published_at`, `effective_from`, `effective_to`) plus a project-axis provenance scalar (`retrieved_at_utc`). The `as_of(df, t)` query is the one new user-facing primitive Stage 13 introduces.

---

## 1. Decisions for the human (drafted by lead, filtered through `@minimalist`, awaiting Ctrl+G)

Twenty-one decision points drawn from the three research artefacts and filtered through the [`Scope Diff`](../../lld/research/13-remit-ingestion-scope-diff.md) `@minimalist` critique. Three rows were flipped from the lead's draft framing:

- **D4** — drop the separate `/remit/revisions/{mrid}` test path; the stream cassette covers revision chains.
- **D15 — CUT.** The `_bmu_reference.py` helper is unnecessary because `fuelType` is already a first-class field on every REMIT row from the stream endpoint. **This is the single highest-leverage cut from the Scope Diff.** Eliminates a new module, a new cache file, a new Hydra config slot, a new CI stub, and at least two tests in one cut.
- **D17 — DEFER.** The bi-temporal-storage ADR is not load-bearing for Stage 13; capture the design in the layer doc instead and promote to an ADR only if Stage 16 finds the choice contested.
- **D18e — CUT.** The `(mrid, revisionNumber)` unique-invariant test duplicates the idempotent-re-fetch test (D18c) and AC-6 does not require uniqueness as a stored constraint, only revision preservation.

The Evidence column cites the artefact that *resolved* each decision.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Module location | **`src/bristol_ml/ingestion/remit.py`** mirroring `neso.py`/`weather.py`/`holidays.py`. | Pattern exact-match across four prior ingestion modules — no new layout to invent. | Codebase map §1; Scope Diff D1 (RESTATES INTENT). |
| **D2** | Public surface | **`fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`**, both with the structural-Protocol contract over `RemitIngestionConfig`. | Implements AC-2/AC-3/AC-11; the load-bearing contract every prior ingester exposes; codebase map §1 confirms downstream consumers (Stage 14, Stage 16) all reach in via `load`. | Codebase map §1, §3; requirements AC-11, US-3, US-4. |
| **D3** | Hydra config + Pydantic schema | **New `conf/ingestion/remit.yaml` + `RemitIngestionConfig` in `conf/_schemas.py` + `IngestionGroup.remit: RemitIngestionConfig \| None = None` slot.** | Implements NFR-8; structural symmetry with the four sibling slots; codebase map flags this as non-negotiable for `cfg.ingestion.remit` to resolve. | Codebase map §3; requirements NFR-8, OQ-7. |
| **D4** | Fetch endpoint | **`GET /datasets/REMIT/stream` only.** No paged fallback. The streaming endpoint has no observed window cap (domain research §R3); the non-streaming endpoint caps at 24h and is unsuitable for backfill. The separate `GET /remit/revisions/{mrid}` endpoint is **not** used at Stage 13 — the stream return already carries `revisionNumber` per row, so revision chains are recoverable via `groupby(mrid)` without a second endpoint. | Single endpoint, single cassette. Adding a per-mRID revision call would mean a second cassette and a second test path for no AC the stream cannot already cover. | Domain research §R2, §R3, §R6; Scope Diff D4 (PLAN POLISH → cut second endpoint). |
| **D5** | Authentication | **None.** Base URL `https://data.elexon.co.uk/bmrs/api/v1/`. No API key, no headers, no secrets. | Domain research §R1 confirms the Insights API is unauthenticated. Reduces scope (no credential stub, no `${oc.env:...}` for keys). | Domain research §R1. |
| **D6** | Default ingestion window | **2018-01-01 → today.** Override via Hydra CLI for a deeper backfill (`+ingestion.remit.window_start=2014-01-01`). | Matches the demand-model training window referenced in the intent §Points; the requirements OQ-2 default. Older history is retrievable but not automatically pulled. | Requirements OQ-2; intent §Points. |
| **D7** | Cache layout | **Single `remit.parquet` under `${BRISTOL_ML_CACHE_DIR}` with no partitioning.** Uncompressed-equivalent payload is ~50–100 MB compressed for the full archive (domain research §R6); pyarrow handles single-file reads fine at this size. | Mirrors `neso.py` / `weather.py` / `holidays.py` exactly. Partitioning is `PREMATURE OPTIMISATION` until the file exceeds working memory. | Domain research §R6; codebase map §1; Scope Diff D7 (RESTATES INTENT). |
| **D8** | Bi-temporal model | **Four timestamp columns, all `timestamp[us, tz=UTC]`:** `published_at` (transaction-time axis: when the participant disclosed), `effective_from` / `effective_to` (valid-time axis: the event window; `effective_to` nullable for open-ended events), `retrieved_at_utc` (project-axis provenance: when this run fetched). As-of filtering is a pandas predicate on `published_at <= t` followed by `groupby(mrid).last()` within the predicate frame; no separate index structure. | Implements AC-1, AC-5, AC-6 directly. The four-column model is the simplest representation that supports all three temporal queries (transaction-time as-of, valid-time overlap, provenance audit). Predicate-pushdown on parquet timestamp columns is fast enough at this scale (~250k rows for the full archive). | Domain research §R4, §R5; requirements AC-1/AC-5/AC-6, OQ-5/OQ-6. |
| **D9** | Row granularity | **One row per `(mrid, revisionNumber)`. Append-only.** No "latest revision wins" overwrite. | Implements AC-6 ("the store preserves all revisions"); the requirements OQ-1 default. The "latest as-of T" view is a query (`as_of(df, t)`), not a storage shape. | Requirements OQ-1; intent §Points "message revisions". |
| **D10** | `effective_to` nullability | **Nullable `timestamp[us, tz=UTC]`. `None` = open-ended event.** As-of query treats `NULL` `effective_to` as "still active." Explicit test for the case. | Some REMIT messages describe ongoing outages with no declared end (domain research §R5). Using `None` rather than a sentinel value (e.g. far-future timestamp) keeps the parquet schema honest and the as-of predicate explicit. | Domain research §R5; requirements OQ-6. |
| **D11** | Test cassette scope | **One VCR cassette covering a one-week window manually selected to contain at least one revised message** (target ≤ 100 kB compressed; the holidays cassette is 31 kB and the NESO is 94.5 kB, so we're in budget). The withdrawn-message case is covered by **one synthetic fixture record** injected at test setup (a bare dict whose `messageStatus="Withdrawn"`); does not exist in any natural one-week slice we can record. | Implements AC-10 + AC-1(b/c); honours the cassette-size budget the codebase map flags. The synthetic withdrawal fixture is a deliberate trade-off: a cassette covering a withdrawal would balloon. | Codebase map §4; requirements OQ-3, AC-1, AC-10. |
| **D12** | Stub mode | **`_stub_*` helpers in `remit.py` produce a structurally valid parquet from a small in-memory record list.** CI default = stub via `BRISTOL_ML_REMIT_STUB=1` env var. The stub's data shape exactly matches the live fetch's `OUTPUT_SCHEMA`. | Implements NFR-2 (stub-first §2.1.3) and the offline-CI requirement. The pattern matches `weather.py`'s stub convention exactly. | Codebase map §4; requirements NFR-2; DESIGN §2.1.3. |
| **D13** | As-of query API | **A single module-level function `as_of(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame`** in `remit.py`. Filters `published_at <= t`, groups by `mrid`, takes the row with the maximum `revisionNumber`, then filters out rows whose `messageStatus == "Withdrawn"`. Returns the active-state frame. | Implements AC-1 directly. A pure-pandas function has no state to carry between calls and is trivially testable. No interval-tree index, no cached state. | Domain research §R4; requirements US-2/AC-1; Scope Diff D13 (RESTATES INTENT). |
| **D14** | Notebook | **`notebooks/13-remit-ingestion.ipynb`** — three cells: (i) `load(path)` the warm cache; (ii) for each month from Jan 2018 to today, compute aggregate active unavailable MW via `as_of(df, t)` at the start of the month, sum by `fuelType` (the row's own `fuelType` field, not a reference-table join — see D15 cut); (iii) plot the resulting `(month, fuel_type, mw)` table as a stacked area chart. The notebook reimplements no logic from the module (§2.1.8). | Implements AC-9 + intent §Demo moment. Three cells is the minimum that surfaces the bi-temporal query mechanic to the audience: `load → as_of → plot`. | Intent §Demo moment; requirements AC-9, US-1; Scope Diff D14 (RESTATES INTENT). |
| ~~**D15**~~ | ~~BMU reference helper~~ | **CUT** per Scope Diff. `fuelType` is already a first-class field on every REMIT row returned by `/datasets/REMIT/stream` (domain research §R2 / §R6); no separate reference fetch is needed. | The notebook joins on the row's own `fuelType` column. Removing this cut removes a new module, a new cache file, a new Hydra slot, a new CI stub, and at least two tests. | **Single highest-leverage cut.** Domain research §R2/§R6; Scope Diff D15. |
| **D16** | Architecture-doc revision | **Update `docs/architecture/layers/ingestion.md`** (currently flagged "Provisional — revisit after Stage 13") with the bi-temporal storage shape: the four-timestamp model, the append-only revision log, the `as_of` query semantics, and a worked example of "message published, revised, withdrawn — what does as-of(t) return?". | The codebase map names this file as the canonical home for the cross-stage ingestion-layer story. The bi-temporal shape is genuinely new ground in the layer — no prior ingester has more than a single `timestamp_utc` column. | Codebase map §3, §7; Scope Diff D16 (HOUSEKEEPING). |
| ~~**D17**~~ | ~~ADR for bi-temporal storage~~ | **DEFER.** The design is captured in the layer doc (D16) and the requirements artefact's OQ-5 default. Promote to a formal ADR only if a future stage (Stage 16's feature join, or Stage 17's price model) finds the choice contested and warrants the ceremony. | An ADR adds maintenance burden and signals that the choice was a coin-flip when in fact the four-column append-only model is the simplest thing that satisfies the intent. Not load-bearing for Stage 13 shipping. | Scope Diff D17 (PLAN POLISH → defer). |
| **D18a** | `as_of` four-scenario unit tests | **Required.** fresh / revised / withdrawn / open-ended; each case asserts the returned frame is exactly the expected mRID set. | Implements AC-1(a/b/c) + the open-ended case under AC-5. | Requirements AC-1, AC-5. |
| **D18b** | Schema validation tests | **Required.** `OUTPUT_SCHEMA` column names and pyarrow types pinned; `load(path)` round-trip yields identical schema. | Implements AC-5 + NFR-4. Pattern carried from every prior ingester. | Requirements NFR-4, AC-5. |
| **D18c** | Idempotent re-fetch test | **Required.** `fetch` twice over the same window from a recorded cassette; assert the on-disk parquet is byte-identical (or row-identical after sort, depending on `_atomic_write` semantics — codebase map §2 confirms `_atomic_write` writes deterministically given a sorted input). | Implements AC-4, NFR-1. | Requirements AC-4, NFR-1. |
| **D18d** | `CachePolicy.OFFLINE` raises `CacheMissingError` | **Required.** Empty cache directory + `CachePolicy.OFFLINE` raises. | Implements AC-2, NFR-6. | Requirements AC-2, NFR-6. |
| ~~**D18e**~~ | ~~`(mrid, revisionNumber)` unique-invariant test~~ | **CUT** per Scope Diff. AC-6 requires all revisions preserved; it does not require a uniqueness invariant. The idempotent re-fetch test (D18c) covers the de-duplication path. | Duplicate safety net. If implementation later reveals a way for duplicates to slip in, add the test then. | Scope Diff D18e (PLAN POLISH → cut). |
| **D18f** | Cassette-driven fetch integration test | **Required.** End-to-end `fetch → on-disk parquet matches the canonical fixture` from a VCR cassette. | Implements AC-10, AC-3. | Requirements AC-3, AC-10. |
| **D18g** | `load` round-trip integration test | **Required.** All four timestamp fields survive parquet round-trip with `tz=UTC`; `effective_to`'s nullability is preserved on the open-ended case. | Implements AC-5. | Requirements AC-5. |
| **D18h** | `as_of` integration test against canonical fixture | **Required.** Three sample times across the cassette's window; assert the returned active-state frame matches a hand-computed expected count + mRID set. | Implements AC-1 at integration level. | Requirements AC-1. |
| **D18i** | Notebook smoke test | **Required.** `tests/integration/notebooks/test_13_remit_ingestion.py` executes the notebook top-to-bottom against the cassette + stub fixture. | Implements AC-9 + project notebook-CI convention. | Codebase map §5; requirements AC-9. |
| **D19** | CHANGELOG | **`### Added` bullet under `[Unreleased]`** for the new ingestion module + the as-of query API. | Stage hygiene. | CLAUDE.md §Stage hygiene. |
| **D20** | Stage retrospective | **`docs/lld/stages/13-remit-ingestion.md`** following the existing template. | Stage hygiene. | CLAUDE.md §Stage hygiene. |
| **D21** | Module CLAUDE.md update | **Update `src/bristol_ml/ingestion/CLAUDE.md`** with the REMIT row in the schema table and a paragraph on the `as_of` query semantics. | Implements AC-8 (the AC names the module CLAUDE.md as the schema's home) + NFR-8. | Requirements AC-8, NFR-8; codebase map §7. |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-1** | Idempotent ingestion (§2.1.5) | Re-running over an already-cached window must not produce duplicate rows. `_atomic_write` writes a deterministic-ordered parquet; cassette-driven test asserts byte-identical output across two consecutive runs. | Requirements NFR-1; DESIGN §2.1.5. |
| **NFR-2** | Stub-first (§2.1.3) | `_stub_*` helpers in `remit.py`; `BRISTOL_ML_REMIT_STUB=1` env var triggers stub path. CI default = stub. | Requirements NFR-2; DESIGN §2.1.3. |
| **NFR-3** | Paging + rate-limit | Use `_retrying_get` and `_respect_rate_limit` from `_common.py` exactly; `RemitIngestionConfig` exposes the seven structural-Protocol fields (`max_attempts`, `backoff_base_seconds`, `backoff_cap_seconds`, `request_timeout_seconds`, `min_inter_request_seconds`, `cache_dir`, `cache_filename`). | Codebase map §2; requirements NFR-3. |
| **NFR-4** | Parquet at boundary (§2.1.2) | Pyarrow parquet with `OUTPUT_SCHEMA` pinning column types. Timestamps `timestamp[us, tz=UTC]`. | Requirements NFR-4; DESIGN §2.1.2. |
| **NFR-5** | Cache outside the repo | `${oc.env:BRISTOL_ML_CACHE_DIR,~/.cache/bristol_ml}` pattern; `data/` gitignored. The cassette fixture lives under `tests/fixtures/` and is committed (≤ 100 kB). | Codebase map §1, §4; requirements NFR-5. |
| **NFR-6** | Offline-after-first-fetch | `CachePolicy.OFFLINE` + missing cache → `CacheMissingError`. | Requirements NFR-6; codebase map §2. |
| **NFR-7** | UTC timestamps | All persisted timestamps are `timestamp[us, tz=UTC]`. No naive datetimes anywhere. | Requirements NFR-7. |
| **NFR-8** | Config in YAML (§2.1.4) | All knobs in `conf/ingestion/remit.yaml`, validated by `RemitIngestionConfig`. No hard-coded values. | Requirements NFR-8; DESIGN §2.1.4. |
| **NFR-9** | Provenance | `retrieved_at_utc` per row (not per batch — every revision must record when *this run* fetched it; consistent with the bi-temporal model). | Requirements NFR-9. |
| **NFR-10** | Observability | `loguru` INFO at each paging step (record count, window slice); WARNING on unexpected `messageStatus` values not in the known set. | Requirements NFR-10; codebase map §1. |

### Decisions and artefacts explicitly **not** in Stage 13

- **D15** — BMU reference helper. Cut per Scope Diff; row-level `fuelType` is sufficient for the notebook.
- **D17** — Bi-temporal-storage ADR. Deferred per Scope Diff; design captured in layer doc (D16) instead.
- **D18e** — `(mrid, revisionNumber)` unique-invariant test. Cut per Scope Diff; duplicates D18c.
- **A separate paged fetcher.** Stream endpoint has no window cap; paging is unnecessary at V1.
- **A `revisions/{mrid}` endpoint integration.** Stream returns `revisionNumber` per row; revision chains are reconstructable.
- **Asset-identifier normalisation against a plant master.** Intent §Out of scope.
- **Free-text extraction from `messageDescription`.** Intent §Out of scope (Stage 14).
- **Joining REMIT into the feature table.** Intent §Out of scope (Stage 16).
- **Embedding / semantic search.** Intent §Out of scope (Stage 15).
- **A derived "events" view** (one row per event window, de-duplicated). Requirements OQ-8 default; deferred to Stage 16's join design.
- **Plant-station-level grouping** (e.g. "show me Hinkley Point B"). Domain research §R6 confirms there is no station-level grouping in the API; deferred indefinitely.

### Open questions for Ctrl+G review (resolved)

All four open-question defaults were accepted at Ctrl+G review on 2026-04-26.

- **OQ-A — fetch granularity.** **Resolved at default.** Single end-of-window cassette plus on-demand backfill via Hydra CLI override. The "weekly refresh" entry point alternative is deferred to Stage 19 (orchestration) where a refresh schedule is the topic of the stage rather than a side decision.
- **OQ-B — stub data realism.** **Resolved at default.** ~10 hand-crafted stub records exercising fresh + revised + withdrawn + open-ended; cassette covers the fetch path; stub and cassette stay decoupled.
- **OQ-C — `effective_to` for instantaneous events.** **Resolved at default.** `effective_from == effective_to` is treated as a normal closed interval; no special-casing in `as_of` or in the notebook's valid-time filter.
- **OQ-D — DESIGN.md §6 status flip.** **Removed from scope** at Ctrl+G review. No §6 edit will be made as part of Stage 13; the existing line "Stages 1, 2, 5, 13, 17" already references Stage 13 and the stage map remains accurate without a per-stage status flip. The PR description will not surface this item.

### Resolution log

- **Drafted 2026-04-25** — pre-Ctrl+G. All 21 decisions proposed; D15, D17, D18e cut/deferred per `@minimalist` Scope Diff. The four open questions OQ-A/B/C/D were unresolved at draft time.
- **Ctrl+G review 2026-04-26** — human accepted all 21 decisions and the default disposition for OQ-A, OQ-B, OQ-C; OQ-D explicitly removed from scope ("Ignore DESIGN §6 changes"). Status flipped `draft → approved` and ready for Phase 2.

---

## 2. Scope

### In scope

Transcribed from `docs/intent/13-remit-ingestion.md §Scope`:

- **A module that retrieves REMIT messages from the Elexon Insights API and persists them locally** — `src/bristol_ml/ingestion/remit.py` (D1) with `fetch` / `load` / `as_of` / `OUTPUT_SCHEMA` (D2/D13).
- **A data model that preserves three times per event** — `published_at`, `effective_from`, `effective_to`, plus the project-axis `retrieved_at_utc` (D8/D10).
- **A mechanism for handling message revisions and supersedes** — append-only `(mrid, revisionNumber)` row granularity (D9); `as_of(df, t)` query takes the latest revision with `published_at <= t`, dropping withdrawn messages (D13).
- **A notebook that visualises REMIT event density over time, coloured by fuel type or event type** — `notebooks/13-remit-ingestion.ipynb`, three cells, fuel-type stacked area chart (D14).
- **Tests against recorded API fixtures** — VCR cassette + synthetic withdrawal fixture (D11); D18a–D18i tests.

Additionally in scope as direct consequences of the above:

- **Hydra config** — `conf/ingestion/remit.yaml` + `RemitIngestionConfig` Pydantic model + `IngestionGroup.remit` slot (D3).
- **Layer-doc revision** — `docs/architecture/layers/ingestion.md` updated with the bi-temporal storage shape (D16).
- **Module guide** — `src/bristol_ml/ingestion/CLAUDE.md` gains the REMIT row + as-of paragraph (D21).
- **Stage retro + CHANGELOG** (D19/D20).

### Explicit out-of-scope

(See §1 "Decisions explicitly not in Stage 13".)

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/13-remit-ingestion.md`](../../intent/13-remit-ingestion.md) — the contract; 5 ACs and 7 "Points for consideration".
2. [`docs/lld/research/13-remit-ingestion-requirements.md`](../../lld/research/13-remit-ingestion-requirements.md) — US-1..US-6, AC-1..AC-11, NFR-1..NFR-10, OQ-1..OQ-8. OQ-1 through OQ-8 resolved by the decisions above (OQ-1=row-per-revision per D9; OQ-2=2018+ per D6; OQ-3=cassette+synthetic per D11; OQ-4=raw asset id per intent OOS; OQ-5=plain timestamp columns per D8; OQ-6=nullable per D10; OQ-7=`conf/ingestion/` per D3; OQ-8=raw log only per D14).
3. [`docs/lld/research/13-remit-ingestion-codebase.md`](../../lld/research/13-remit-ingestion-codebase.md) — `_common.py` Protocol shapes (§2), `IngestionGroup` slot pattern (§3), VCR cassette convention (§4), notebook CI smoke-test pattern (§5), the layer-doc "provisional pending Stage 13" call-out (§7).
4. [`docs/lld/research/13-remit-ingestion-domain.md`](../../lld/research/13-remit-ingestion-domain.md) — §R1 (no-auth Insights API), §R2 (REMIT endpoint family + response schema), §R3 (24h cap on `/datasets/REMIT`, no cap on `/datasets/REMIT/stream`), §R4 (bi-temporal as-of mechanics), §R5 (open-ended `effective_to`), §R6 (volume estimate + `fuelType` on every row + no station-level grouping).
5. [`docs/lld/research/13-remit-ingestion-scope-diff.md`](../../lld/research/13-remit-ingestion-scope-diff.md) — `@minimalist` critique; every cut and retention is listed there.
6. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
7. `docs/architecture/layers/ingestion.md` — the layer-level contract; the file Stage 13 will revise (D16).
8. `src/bristol_ml/ingestion/CLAUDE.md` — concrete ingestion-layer surface; the schema table is the file Stage 13 will extend (D21).
9. `src/bristol_ml/ingestion/_common.py` — the four-callable `_atomic_write` / `_cache_path` / `_respect_rate_limit` / `_retrying_get` Stage 13 reuses verbatim.
10. `src/bristol_ml/ingestion/holidays.py` — closest structural sibling (single endpoint, single parquet file, simple parser); the file the Stage 13 implementer should pattern after for layout.
11. `tests/integration/ingestion/test_neso_holidays_*.py` — the cassette + `pytest-recording` convention to mirror.
12. `tests/conftest.py` — the `loguru_caplog` fixture for asserting on bound log fields (NFR-10).

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five intent-ACs are copied verbatim from `docs/intent/13-remit-ingestion.md §Acceptance criteria`, then grounded in one or more named tests.

- **AC-1 (intent).** "The module can answer the question 'what REMIT events were known to the market at time T?' correctly for any T within the cached period, including revisions."
  - Tests:
    - `test_as_of_fresh_message_returns_active_state` (D18a) — single mRID, single revision, `published_at = t-1h`; `as_of(df, t)` returns that row.
    - `test_as_of_revised_message_returns_latest_revision` (D18a) — single mRID with revisions 0/1/2 published at t-3h/t-2h/t-1h; `as_of(df, t-1.5h)` returns rev 1 (the latest revision visible at that point in time, since rev 2 has not yet been published), `as_of(df, t)` returns rev 2.
    - `test_as_of_withdrawn_message_excludes_row` (D18a) — single mRID with rev 0 published at t-2h then `messageStatus="Withdrawn"` at t-1h; `as_of(df, t)` excludes this mRID.
    - `test_as_of_open_ended_effective_to_treated_as_active` (D18a) — `effective_to=None`, `effective_from < t`; `as_of(df, t)` includes the row.
    - `test_as_of_against_cassette_fixture_at_three_sample_times` (D18h, integration) — full cassette load + three timestamp slices; mRID-set + count assertions against hand-computed expected.

- **AC-2 (intent).** "Running the ingestion with a cache present completes offline."
  - Tests:
    - `test_fetch_offline_with_warm_cache_returns_path_without_network` (D18d) — cassette absent (or VCR set to `record_mode=none`); `CachePolicy.OFFLINE` returns the cache path.
    - `test_fetch_offline_without_cache_raises_cache_missing` (D18d) — empty cache dir; `CachePolicy.OFFLINE` raises `CacheMissingError` whose message names the cache path.

- **AC-3 (intent).** "Running the ingestion without a cache fetches from Elexon and populates the cache."
  - Tests:
    - `test_fetch_against_cassette_populates_cache` (D18f, integration) — empty cache dir + cassette; `CachePolicy.AUTO` results in a populated cache file.
    - `test_fetch_idempotent_against_cassette` (D18c) — fetch twice; on-disk parquet is byte-identical (or row-identical after sort).

- **AC-4 (intent).** "The visualisation notebook renders a meaningful summary of events over time."
  - Tests:
    - `test_notebook_13_remit_executes_top_to_bottom` (D18i, integration) — `nbclient` + cassette + stub.

- **AC-5 (intent).** "Tests cover the bi-temporal query discipline, not just the fetch."
  - Tests:
    - This AC is met by AC-1's five `as_of` tests being a strict majority of the named tests under the AC family.
    - `test_load_round_trips_all_four_timestamps_with_tz_utc` (D18g) — load yields columns with `pd.api.types.is_datetime64tz_dtype` true and `tz=='UTC'` for all four temporal columns.
    - `test_load_round_trips_open_ended_effective_to_as_pd_nat` (D18g) — `effective_to` is `pd.NaT` for the open-ended row.
    - `test_output_schema_columns_and_types_pinned` (D18b) — `OUTPUT_SCHEMA` matches the documented schema in `ingestion/CLAUDE.md` exactly.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_remit_config_round_trips_through_hydra` — `conf/ingestion/remit.yaml` defaults match `RemitIngestionConfig(...)` exactly.
- `test_app_config_ingestion_remit_default_is_none_so_existing_callers_unaffected` — every prior ingestion module continues to validate.
- `test_remit_module_runs_standalone` — `python -m bristol_ml.ingestion.remit --help` exits 0 (AC-11/NFR-2 pattern from prior ingesters).
- `test_remit_logs_paging_step_at_info` — `loguru_caplog` asserts an INFO record per paging step (NFR-10).

**Total shipped tests: ~14** — five AC-1 `as_of` tests, two AC-2 cache-policy tests, two AC-3 fetch tests, one AC-4 notebook test, three AC-5 schema/round-trip tests, plus four D-derived (Hydra round-trip, ingestion-group default, standalone CLI, log-record).

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/ingestion/
├── _common.py           # (existing) — reused via Protocol-typed helpers
├── neso.py              # (existing)
├── neso_forecast.py     # (existing)
├── weather.py           # (existing)
├── holidays.py          # (existing)
├── remit.py             # NEW — Stage 13's only new source file under src/
└── CLAUDE.md            # (existing) — extended with the REMIT row + as_of paragraph
```

`remit.py`'s public surface:

```python
__all__ = [
    "OUTPUT_SCHEMA",
    "MESSAGE_STATUSES",      # known {"Active", "Cancelled", "Withdrawn", ...}
    "FUEL_TYPES",            # known {"Coal", "Gas", "Nuclear", ...} for stub
    "CacheMissingError",     # re-exported from _common
    "CachePolicy",           # re-exported from _common
    "fetch",                 # (config, *, cache=CachePolicy.AUTO) -> Path
    "load",                  # (path) -> pd.DataFrame
    "as_of",                 # (df, t) -> pd.DataFrame  ← the new primitive
]
```

### `OUTPUT_SCHEMA`

```python
import pyarrow as pa

OUTPUT_SCHEMA: pa.Schema = pa.schema([
    # Identifier axis
    pa.field("mrid", pa.string(), nullable=False),                       # Elexon message-id
    pa.field("revision_number", pa.int32(), nullable=False),             # 0-indexed
    pa.field("message_type", pa.string(), nullable=False),               # "Production", "Consumption", ...
    pa.field("message_status", pa.string(), nullable=False),             # "Active", "Withdrawn", ...

    # Bi-temporal axis
    pa.field("published_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("effective_from", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("effective_to", pa.timestamp("us", tz="UTC"), nullable=True),
    pa.field("retrieved_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),

    # Asset axis (raw — no normalisation per intent OOS)
    pa.field("affected_unit", pa.string(), nullable=True),               # BMU id, e.g. "WBURB-1"
    pa.field("asset_id", pa.string(), nullable=True),                    # prefixed BMU, e.g. "T_WBURB-1"
    pa.field("fuel_type", pa.string(), nullable=True),                   # "Nuclear", "Gas", ... (D15: row-level, no reference join)

    # Capacity axis
    pa.field("affected_mw", pa.float64(), nullable=True),                # the unavailable-capacity headline number
    pa.field("normal_capacity_mw", pa.float64(), nullable=True),

    # Free text — Stage 14 will read this
    pa.field("event_type", pa.string(), nullable=True),                  # "Outage", "Restriction", ...
    pa.field("cause", pa.string(), nullable=True),                       # "Planned", "Unplanned", "Forced", ...
    pa.field("message_description", pa.string(), nullable=True),         # the unstructured payload
])
```

The exact field set is taken from the domain research §R2's confirmed live-call response. Field-name casing snake-cases the API's camelCase. `nullable=True` on `effective_to` is load-bearing for the open-ended case (D10).

### `as_of` semantics

```python
def as_of(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return the active-state frame as known to the market at time t.

    Implements the intent's central question: "what REMIT events were known
    to the market at time T?".

    Algorithm:
      1. Filter df to rows with published_at <= t (transaction-time as-of).
      2. Within that filter, group by mrid; keep the row with max revision_number.
      3. Drop rows whose message_status == "Withdrawn".

    Notes:
      - effective_to is *not* part of the as-of filter — that is a valid-time
        join, separate from the transaction-time as-of. Callers who want
        "active at t" (valid-time) chain a second filter: df = as_of(df, t);
        df = df[(df.effective_from <= t) & (df.effective_to.isna() | (df.effective_to > t))].
      - Returns a copy. Does not mutate the input.

    Raises:
      ValueError: if t is naive (i.e. t.tzinfo is None).
    """
    ...
```

The two-step "as-of then optionally valid-time-active" decomposition is the standard bi-temporal pattern (domain research §R4). Keeping `as_of` strictly transaction-time means the function has one job and the caller composes for valid-time. The notebook's monthly aggregation does compose both filters.

### Hydra config

```python
# conf/_schemas.py (addition)

class RemitIngestionConfig(BaseModel):
    """Stage 13 — REMIT ingestion.

    Structurally satisfies the three Protocol types in
    bristol_ml.ingestion._common (RetryConfig, RateLimitConfig, CachePathConfig).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    # API
    base_url: HttpUrl = "https://data.elexon.co.uk/bmrs/api/v1/"
    endpoint_path: str = "datasets/REMIT/stream"
    window_start: date = date(2018, 1, 1)
    window_end: date | None = None  # None = today

    # Retry / rate-limit (Protocol structural fields)
    max_attempts: int = 5
    backoff_base_seconds: float = 1.0
    backoff_cap_seconds: float = 30.0
    request_timeout_seconds: float = 30.0
    min_inter_request_seconds: float = 0.5

    # Cache (Protocol structural fields)
    cache_dir: Path = Path("${oc.env:BRISTOL_ML_CACHE_DIR,~/.cache/bristol_ml}")
    cache_filename: str = "remit.parquet"

# conf/_schemas.py — IngestionGroup gains:
#     remit: RemitIngestionConfig | None = None
```

```yaml
# conf/ingestion/remit.yaml
# @package ingestion.remit
base_url: https://data.elexon.co.uk/bmrs/api/v1/
endpoint_path: datasets/REMIT/stream
window_start: 2018-01-01
window_end: null
max_attempts: 5
backoff_base_seconds: 1.0
backoff_cap_seconds: 30.0
request_timeout_seconds: 30.0
min_inter_request_seconds: 0.5
cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,~/.cache/bristol_ml}
cache_filename: remit.parquet
```

`conf/config.yaml` defaults gain:

```yaml
- ingestion/remit@ingestion.remit
```

### Notebook structure (`notebooks/13-remit-ingestion.ipynb`)

Three cells (no setup boilerplate beyond `import`):

1. **Load** — `df = remit.load(remit.fetch(cfg.ingestion.remit, cache=CachePolicy.AUTO))`. Cell prints record count + window summary.
2. **Compute** — for each month-start `t` in `[2018-01-01, today]`, run `as_of(df, t)` then valid-time filter (`effective_from <= t < effective_to.fillna(far-future)`), sum `affected_mw` by `fuel_type`. Build a long-form `(month, fuel_type, total_mw)` frame.
3. **Plot** — stacked area chart with `fuel_type` colour layer; title + x/y labels per the project's plotting conventions.

### Standalone CLI

```
$ uv run python -m bristol_ml.ingestion.remit --help
usage: remit [-h] [--cache {auto,refresh,offline}] [overrides ...]

Fetch and persist REMIT messages from the Elexon Insights API.

options:
  -h, --help                       show this help and exit
  --cache {auto,refresh,offline}   cache policy (default: auto)

positional arguments:
  overrides                        Hydra-style config overrides
```

`__main__.py` (or `_cli_main(argv)` inline in `remit.py`, matching the holidays/weather pattern) — argparse + `bristol_ml.config.load_config()` + `fetch(cfg.ingestion.remit, cache=...)` + summary print.

---

## 6. Tasks (sequential — see CLAUDE.md §Phase 2 for sequencing rules)

Each task ends with one or more pytest invocations and a single git commit citing this plan task number. The `@tester` is spawned alongside or before each task per CLAUDE.md §Tester timing.

### T1 — `RemitIngestionConfig` + Hydra config + `IngestionGroup.remit` slot.
1. Add `RemitIngestionConfig` to `conf/_schemas.py` per §5.
2. Add `remit: RemitIngestionConfig | None = None` to `IngestionGroup`.
3. Create `conf/ingestion/remit.yaml` per §5.
4. Add `- ingestion/remit@ingestion.remit` to `conf/config.yaml` defaults list.
- **Tests:** `test_remit_config_round_trips_through_hydra`, `test_app_config_ingestion_remit_default_is_none_so_existing_callers_unaffected`.
- **Commit:** `Stage 13 T1: RemitIngestionConfig + Hydra config + IngestionGroup slot`.

### T2 — `OUTPUT_SCHEMA` + `as_of` + module skeleton.
1. Create `src/bristol_ml/ingestion/remit.py` with module docstring, `__all__`, `OUTPUT_SCHEMA`, `MESSAGE_STATUSES` / `FUEL_TYPES` constants, and `as_of(df, t)` per §5.
2. `fetch` / `load` are `NotImplementedError` stubs at this point — wired in T3/T4.
3. The trick: `as_of` is testable purely against in-memory frames, no fixtures, so it can ship before the fetch path. This lands the bi-temporal teaching point first.
- **Tests (D18a, D18b):** `test_as_of_fresh_message_returns_active_state`, `test_as_of_revised_message_returns_latest_revision`, `test_as_of_withdrawn_message_excludes_row`, `test_as_of_open_ended_effective_to_treated_as_active`, `test_as_of_raises_on_naive_timestamp`, `test_output_schema_columns_and_types_pinned`.
- **Commit:** `Stage 13 T2: remit.py skeleton + OUTPUT_SCHEMA + as_of() with unit tests`.

### T3 — Stub fetch path + `load` + standalone CLI.
1. Implement `_stub_*` helpers producing ~10 hand-crafted records covering all four AC-1 cases (fresh / revised / withdrawn / open-ended).
2. Implement `fetch` with the `BRISTOL_ML_REMIT_STUB=1` branch wired to `_stub_*`. Live branch is `NotImplementedError` — wired in T4.
3. Implement `load(path)` — pyarrow read + schema cast + return as pandas frame with the four temporal columns having `tz=UTC`.
4. Implement `_cli_main(argv)` per §5.
- **Tests (D18d, D18g, NFR-2):** `test_fetch_with_stub_env_var_writes_canonical_parquet`, `test_load_round_trips_all_four_timestamps_with_tz_utc`, `test_load_round_trips_open_ended_effective_to_as_pd_nat`, `test_fetch_offline_without_cache_raises_cache_missing`, `test_remit_module_runs_standalone`.
- **Commit:** `Stage 13 T3: stub fetch + load + standalone CLI`.

### T4 — Live fetch + VCR cassette + integration test.
1. Implement the live fetch path against `/datasets/REMIT/stream` using `_retrying_get` and `_respect_rate_limit`. Iterate the stream, parse each message into the canonical row shape, append to a list, write via `_atomic_write` once the stream completes.
2. Record one VCR cassette covering a manually selected one-week window known to contain at least one revised message. Place under `tests/fixtures/cassettes/test_remit_fetch_against_cassette_populates_cache.yaml`. Target ≤ 100 kB.
3. Inject one synthetic withdrawn-message record in the test setup for the as-of-withdrawn integration test (the cassette window is unlikely to contain a withdrawal).
- **Tests (D18c, D18f, D18h, NFR-1, NFR-10):** `test_fetch_against_cassette_populates_cache`, `test_fetch_idempotent_against_cassette`, `test_as_of_against_cassette_fixture_at_three_sample_times`, `test_remit_logs_paging_step_at_info`.
- **Commit:** `Stage 13 T4: live fetch + VCR cassette + integration tests`.

### T5 — Notebook + notebook smoke test.
1. Create `notebooks/13-remit-ingestion.ipynb` with the three cells per §5.
2. Use the project's notebook-CI pattern (Stages 1, 2, 5) — the smoke test runs `nbclient` against the cassette + stub fixture; CI default is `BRISTOL_ML_REMIT_STUB=1`.
- **Tests (D18i, AC-9):** `test_notebook_13_remit_executes_top_to_bottom`.
- **Commit:** `Stage 13 T5: visualisation notebook + smoke test`.

### T6 — Documentation.
1. `docs/architecture/layers/ingestion.md` — flip "Provisional — revisit after Stage 13" → resolved; add a §"Bi-temporal storage shape" section with the four-column model, the `as_of` semantics, and the worked "published / revised / withdrawn — what does as-of(t) return?" example.
2. `src/bristol_ml/ingestion/CLAUDE.md` — add the REMIT row to the schema table; add an "as-of query" paragraph.
3. README — short bullet under the "ingestion" section linking to the new layer doc and the notebook.
4. `docs/lld/stages/13-remit-ingestion.md` — retro skeleton; observed cassette size + record count recorded here.
5. `CHANGELOG.md` — `### Added` bullet under `[Unreleased]`.
- **Tests:** none (doc edits).
- **Commit:** `Stage 13 T6: layer doc + module guide + README + retro skeleton + CHANGELOG`.

### T7 — Stage hygiene + plan move.
1. `git mv docs/plans/active/13-remit-ingestion.md docs/plans/completed/13-remit-ingestion.md`.
2. Final retro updates: actual cassette size, actual full-archive record count if a backfill was run locally, any decisions deviated from in-stage.
3. `uv run pytest -q` clean; `uv run ruff check .` clean; `uv run ruff format --check .` clean; `uv run pre-commit run --all-files` clean.
- **Commit:** `Stage 13 T7: stage hygiene + retro + plan moved to completed/`.

### T8 — Phase 3 review.
Spawn `arch-reviewer` (conformance to plan + intent), `code-reviewer` (code quality + security — particular focus on the bi-temporal predicate correctness in `as_of`), `docs-writer` (user + developer docs sweep) in parallel. Synthesise findings, address Blocking items in-branch, surface Major+Minor in the PR description.

---

## 7. Exit checklist

Before opening the PR:

- [ ] All ~14 named tests in §4 pass; full `uv run pytest -q` is clean.
- [ ] All five `as_of` unit tests pass (T2 — covers AC-1 + AC-5).
- [ ] VCR cassette ≤ 100 kB; integration test asserts byte-identical idempotent re-fetch.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `uv run python -m bristol_ml.ingestion.remit --help` exits 0 with the resolved `RemitIngestionConfig` schema.
- [ ] Layer doc `docs/architecture/layers/ingestion.md` no longer carries the "provisional" flag; the bi-temporal section is present.
- [ ] Module guide `src/bristol_ml/ingestion/CLAUDE.md` carries the REMIT row + as-of paragraph.
- [ ] README has a brief reference to the new ingestion source + the notebook.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`: REMIT module + `as_of` listed under `### Added`.
- [ ] Retro at `docs/lld/stages/13-remit-ingestion.md` carries the observed cassette size + any deviations.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/`.
- [ ] PR description surfaces:
  - The bi-temporal contract for Stage 14 (column names + nullability of `effective_to`).
  - The cassette scope decision (D11) and the synthetic-withdrawal-fixture trade-off — for the next ingestion stage's plan author.
  - The Scope-Diff cuts (D15 = single highest-leverage cut, D17 deferred, D18e cut) — for the stage retro narrative.
