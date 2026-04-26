# Stage 13 — REMIT ingestion: scope diff

**Role:** Phase-1 scope critic, run after the three research artefacts and the lead's draft decision set, before the plan is written.
**Inputs:** `docs/intent/13-remit-ingestion.md`, the requirements / codebase / domain artefacts in this directory, and the lead's 21-item draft decision set.
**Tag taxonomy:** `RESTATES INTENT` (keep) · `PLAN POLISH` (reconsider) · `PREMATURE OPTIMISATION` (default cut) · `HOUSEKEEPING` (keep — required by project conventions).

## Scope diff table

| Item | Summary | Tag | Justification | Recommended action |
|------|---------|-----|---------------|--------------------|
| D1 | Module location `src/bristol_ml/ingestion/remit.py` | `RESTATES INTENT` | Directly implements "A module that retrieves REMIT messages…and persists them locally." | Keep. |
| D2 | Public surface `fetch` / `load` | `RESTATES INTENT` | Implements AC-2, AC-3, the §2.1.1 standalone-run contract; load-bearing for Stage 14. | Keep. |
| D3 | Hydra config + Pydantic schema + `IngestionGroup.remit` slot | `RESTATES INTENT` | AC-11 (standalone CLI), NFR-8 (config in YAML); structural hook every prior ingester adds. | Keep. |
| D4 | `GET /datasets/REMIT/stream` for bulk; `GET /remit/revisions/{mrid}` only for tests | `PLAN POLISH` | The streaming endpoint is the only viable bulk path. The separate revisions call is optional — the stream cassette can cover revision chains without a second endpoint, saving one test path and one cassette. | **Cut the second endpoint** unless cassette inspection shows the stream alone cannot deliver a multi-revision example. |
| D5 | No authentication | `RESTATES INTENT` | Domain research confirms Elexon Insights is unauthenticated. Reduces scope (no credential stub). | Keep. |
| D6 | Default ingestion window 2018-01-01 → today | `RESTATES INTENT` | Implements OQ-2 resolution; matches the demand training window referenced in the intent. | Keep. |
| D7 | Single `remit.parquet`, no partitioning at V1 | `RESTATES INTENT` | Implements OQ-5 resolution; 50–100 MB ceiling is manageable; consistent with prior ingesters. | Keep. |
| D8 | Bi-temporal four-column model | `RESTATES INTENT` | Directly implements the bi-temporal §Scope paragraph and AC-5, AC-1. | Keep. |
| D9 | Row granularity `(mrid, revisionNumber)`, append-only | `RESTATES INTENT` | Implements AC-6, OQ-1 resolution, the "message revisions" §Points concern. | Keep. |
| D10 | `effective_to` nullable | `RESTATES INTENT` | Implements OQ-6; covered by AC-5 and tested under AC-1(b/c). | Keep. |
| D11 | VCR cassette + synthetic withdrawn-message fixture | `RESTATES INTENT` | Implements AC-10, AC-1(c), OQ-3 resolution. | Keep. |
| D12 | Stub mode for CI | `RESTATES INTENT` | Implements NFR-2 (stub-first §2.1.3), AC-2. | Keep. |
| D13 | `as_of(df, t)` public function | `RESTATES INTENT` | Single most direct operationalisation of AC-1. | Keep. |
| D14 | Visualisation notebook | `RESTATES INTENT` | Matches §Demo moment and AC-9. | Keep. |
| **D15** | **`ingestion/_bmu_reference.py` helper for fuel-type join** | **`PLAN POLISH`** | **`fuelType` is already a first-class field on every REMIT row from the stream endpoint (per the domain artefact). A BMU reference join is unnecessary for the notebook's fuel-type colour layer. A separate module adds a public surface, a new cache file, a new Hydra slot, a new CI stub, and ~2 tests.** | **Cut.** Use the row's `fuelType` field directly in the notebook. |
| D16 | Update `docs/architecture/layers/ingestion.md` | `HOUSEKEEPING` | The codebase artefact notes the layer doc is flagged "Revisit after Stage 13"; resolving it is the same hygiene as CHANGELOG. | Keep. |
| D17 | New ADR "Bi-temporal storage for REMIT" | `PLAN POLISH` | Intent does not name an ADR; the bi-temporal approach is resolved by OQ-5; an ADR adds maintenance burden and sets precedent for follow-on ADRs. Worth doing eventually — not load-bearing for shipping Stage 13. | **Defer.** Capture the design in `layers/ingestion.md` under D16; promote to an ADR only if Stage 16 finds the choice contested. |
| D18a | `as_of` four-scenario unit tests | `RESTATES INTENT` | Implements AC-1(a/b/c) and the open-ended case under AC-5. | Keep. |
| D18b | Schema validation tests | `RESTATES INTENT` | Implements AC-5 and NFR-4. | Keep. |
| D18c | Idempotent re-fetch test | `RESTATES INTENT` | Implements AC-4, NFR-1. | Keep. |
| D18d | `CachePolicy.OFFLINE` raises `CacheMissingError` | `RESTATES INTENT` | Implements AC-2, NFR-6. | Keep. |
| D18e | `(mrid, revisionNumber)` unique-invariant test | `PLAN POLISH` | AC-6 requires all revisions preserved; it does not name a uniqueness constraint. If the idempotent re-fetch test (D18c) covers the de-duplication path, this is a duplicate safety net. | **Cut** unless implementation reveals a non-trivial way for duplicates to slip in. |
| D18f | Cassette-driven integration fetch test | `RESTATES INTENT` | Implements AC-10, AC-3. | Keep. |
| D18g | `load` round-trip integration test | `RESTATES INTENT` | Implements AC-5. | Keep. |
| D18h | `as_of` integration at three sample times | `RESTATES INTENT` | Implements AC-1 at integration level. | Keep. |
| D18i | Notebook smoke test | `RESTATES INTENT` | Implements AC-9 and project notebook-CI convention. | Keep. |
| D19 | CHANGELOG bullet | `HOUSEKEEPING` | Required by stage hygiene. | Keep. |
| D20 | Stage retrospective | `HOUSEKEEPING` | Required by stage hygiene. | Keep. |
| D21 | Module CLAUDE.md update | `RESTATES INTENT` | Directly implements AC-8 ("schema documented in module CLAUDE.md") and NFR-8. | Keep. |

## Single highest-leverage cut

Cut **D15 (`ingestion/_bmu_reference.py`)** because `fuelType` is already a first-class field on every REMIT row from the stream endpoint, making the BMU reference join entirely unnecessary for the notebook's fuel-type colour layer, and removing it eliminates a new module, a new cache file, a new Hydra config slot, a new CI stub, and at least two tests in one cut.
