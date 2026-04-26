# Stage 13 — REMIT ingestion: domain research

**Date:** 2026-04-25
**Target plan:** `docs/plans/active/13-remit-ingestion.md` (not yet created)
**Intent source:** `docs/intent/13-remit-ingestion.md`
**Baseline SHA:** main @ `6267cc0` (Stage 9 merged; Stage 10 branch active)

**Scope:** External technical context for the lead to synthesise into the Stage 13 plan. Citations-grounded. Not a design document — findings only. British English throughout.

---

## REMIT in the GB context

### What REMIT is

REMIT (Regulation on Wholesale Energy Market Integrity and Transparency) prohibits insider trading and market manipulation in wholesale energy markets and requires market participants to publish "inside information" before they may trade on it. For electricity, inside information is any precise, non-public information about a generation, transmission or consumption facility that, if made public, would be likely to significantly affect wholesale energy prices. In practice this means outage and availability events — when a nuclear unit trips or a CCGT undertakes planned maintenance, the operator must submit an Urgent Market Message (UMM) to a designated publication platform.

### Classes of events reported

The REMIT schema for electricity uses the `UnavailabilitiesOfElectricityFacilities` message type. Two types of unavailability exist:

- **Planned** — scheduled maintenance or repair works with advance notice.
- **Unplanned** — unforeseen technical problems; the initial message may have `durationUncertainty: "Unknown"`.

A third message type, `Curtailment`, exists for wind/solar output limitations but is less common in the Elexon dataset. Ofgem has specifically warned against rigid 100 MW capacity thresholds when determining whether information is "inside information" — participants must assess price sensitivity based on market context.

### GB versus EU REMIT post-Brexit

REMIT was originally EU Regulation 1227/2011. It was incorporated into UK law at Brexit and is now enforced by Ofgem rather than ACER.

| Dimension | EU REMIT | GB REMIT |
|---|---|---|
| Enforcer | ACER (cross-border) | Ofgem (GB only) |
| Trade data reporting to regulator | Via Registered Reporting Mechanisms (RRMs) to ACER | No equivalent regime; Ofgem uses broker/exchange data |
| Inside information guidance | ACER comprehensive guidance (regularly updated) | Ofgem uses ACER guidance; UK-specific guidance not yet issued |
| Revised REMIT (EU) 2024 | In force 7 May 2024; expands scope, ACER on-site inspection powers, algorithmic trading notification | **Does not apply to GB.** UK regime is frozen at Brexit-day state. Ofgem paused review work pending EU-UK energy reset. |
| Publication platform | EMFIP / ACER-authorised IIPs | Elexon Insights Solution (electricity); National Gas Transmission (gas) |

**For Stage 13 (GB electricity only), the authoritative publication platform is Elexon Insights Solution.** The old BMRS REMIT portal was decommissioned on 31 May 2024.

GB electricity unavailability data stopped appearing on the ENTSO-E Transparency Platform on 15 June 2021 as a consequence of the Trade and Cooperation Agreement; Northern Ireland continues under the Ireland/Northern Ireland Protocol.

### What a structured REMIT message contains

A confirmed live response from `GET /datasets/REMIT` (2024-01-15 window, verified directly against the API) contains the following fields per row:

| Field | Type | Semantics |
|---|---|---|
| `dataset` | string | Always `"REMIT"` |
| `mrid` | string | Market Resource ID — unique per event chain; format `<registrationCode>-RMT-<sequence>` |
| `revisionNumber` | integer | Starts at 1; increments with each revision |
| `publishTime` | ISO 8601 UTC | When Elexon published the message (transaction time) |
| `createdTime` | ISO 8601 UTC | When the submitter created the message (seconds before publishTime) |
| `messageType` | string | e.g. `"UnavailabilitiesOfElectricityFacilities"` |
| `messageHeading` | string | Short system-generated descriptor |
| `eventType` | string | e.g. `"Production unavailability"` |
| `unavailabilityType` | string | `"Planned"` or `"Unplanned"` |
| `participantId` | string | Elexon short code e.g. `"DRAX"`, `"WESTBURB"` |
| `registrationCode` | string | ACER/Elexon EIC registration code e.g. `"48X000000000045Z"` |
| `assetId` | string | Elexon asset ID with prefix e.g. `"T_WBURB-1"` (T_ = transmission) |
| `assetType` | string | e.g. `"Production"` |
| `affectedUnit` | string | Unit short name without prefix e.g. `"WBURB-1"` — maps to `nationalGridBmUnit` |
| `affectedUnitEIC` | string | ENTSO-E EIC code e.g. `"48W00000WBURB-19"` |
| `affectedArea` | string | Grid Supply Point group e.g. `"B1"` |
| `biddingZone` | string | Always `"10YGB----------A"` for GB generation |
| `fuelType` | string | One of 12 values (see below) |
| `normalCapacity` | float (MW) | Registered capacity |
| `availableCapacity` | float (MW) | Capacity available during event |
| `unavailableCapacity` | float (MW) | `normalCapacity − availableCapacity` |
| `eventStatus` | string | `"Active"` or `"Dismissed"` |
| `eventStartTime` | ISO 8601 UTC | Unavailability start — valid time start |
| `eventEndTime` | ISO 8601 UTC | Unavailability end — valid time end |
| `durationUncertainty` | string | Optional; e.g. `"Unknown"`, `"+- 1 day"` |
| `cause` | string | e.g. `"Mechanical"`, `"Boiler / Fuel Supply"`, `"Under investigation"` |
| `relatedInformation` | string | **Free-text field** — Stage 14's extraction target |
| `outageProfile` | array | Optional nested `[{startTime, endTime, capacity}]` for sub-period capacity changes |

**Confirmed fuel types** (from `GET /reference/remit/fueltypes/all`): Biomass, Fossil Brown coal/Lignite, Fossil Gas, Fossil Hard coal, Fossil Oil, Hydro Pumped Storage, Hydro Water Reservoir, Nuclear, Other, Other renewable, Wind Offshore, Wind Onshore.

### Structured versus free text

- **Structured**: all fields except `relatedInformation`, and to a lesser extent `messageHeading`, `cause`, and `durationUncertainty`.
- **Free text / semi-structured**: `relatedInformation` contains human-written notes such as `"Status changed to Current"`, `"GT1 issue"`, or informal encoded schedules like `"~/E_DERBY-1,2024-01-16 23:30:00,2024-01-17 06:00:00,0"`. Stage 13 preserves it verbatim; Stage 14 extracts from it.

### Revisions and supersedes in the regulation

REMIT messages are routinely corrected, prolonged, and withdrawn. Elexon's model in the Insights API:

- Each revision shares the same **mRID** and increments `revisionNumber`.
- There is **no `supersedes` or `supersededBy` field** in the API response — revision linkage is solely via mRID grouping plus revisionNumber ordering.
- A **withdrawn/cancelled** message receives `eventStatus: "Dismissed"` rather than being deleted.
- The `/remit/revisions` endpoint returns minimal metadata (id, mrid, revisionNumber, publishTime, createdTime) for all revisions of an mRID — not the full message body.
- The full body of any revision is available at `GET /remit/{id}` (where `id` is the integer identifier in the revisions list).

---

## Elexon Insights API surface

### Base URL and authentication

**Base URL:** `https://data.elexon.co.uk/bmrs/api/v1/`

**Authentication:** None. All Insights Solution APIs are public. No API key is needed. This is explicitly confirmed in the developer portal and the `insights-docs` GitHub repository. The old BMRS API at `api.bmreports.com` required a scripting key and was decommissioned 31 May 2024. **Any code or tutorial referencing `api.bmreports.com` or BMRS scripting keys is obsolete.**

### Full REMIT endpoint family

Confirmed from `elexonpy` Swagger-generated client and direct API calls:

| Endpoint | Type | Description |
|---|---|---|
| `GET /datasets/REMIT` | Dataset | All messages in a publish-time window; JSON/CSV/XML |
| `GET /datasets/REMIT/stream` | Stream | Same data, streaming JSON — preferred for bulk fetches |
| `GET /remit` | Opinionated | Bulk fetch by list of integer message IDs |
| `GET /remit/{messageId}` | Opinionated | Single full message record by integer `id` |
| `GET /remit/search` | Opinionated | Fetch by mRID string; returns full record |
| `GET /remit/revisions` | Opinionated | All revision stubs for a given mRID |
| `GET /remit/list/by-publish` | Opinionated | List (not full records) in publish-time window |
| `GET /remit/list/by-event` | Opinionated | List (not full records) in event-time window |
| `GET /remit/list/by-publish/stream` | Stream | Streaming list/by-publish |
| `GET /remit/list/by-event/stream` | Stream | Streaming list/by-event |
| `GET /reference/remit/assets/all` | Reference | All asset IDs (plain string array) |
| `GET /reference/remit/fueltypes/all` | Reference | All fuel type strings |
| `GET /reference/remit/participants/all` | Reference | All participant IDs |

### Pagination, rate limiting, and query windows

**`GET /datasets/REMIT` has a 1-day maximum query window.** A two-day window returns HTTP 400 (confirmed empirically). A single-day query returns approximately 100–150 rows including multiple revisions.

**`GET /datasets/REMIT/stream` has no observed window constraint** and no row cap. It is the correct endpoint for historical backfills.

**The opinionated `list/by-publish` and `list/by-event` endpoints** also appear to have window constraints; streaming variants should be preferred for bulk work.

**Pagination:** The `{"data": [...]}` response contains no `totalCount`, `hasMore`, or cursor fields. Response size is controlled by the time window rather than page parameters.

**Rate limits:** Not publicly documented. Elexon states "API requests may be capped as Elexon deems appropriate." Implement conservative exponential back-off and at minimum 1 second between consecutive calls.

**Response format:** JSON (default), CSV, XML for dataset endpoints. Stream endpoints are JSON-only.

### Time field semantics

All timestamps in the API use **ISO 8601 with UTC (`Z` suffix)**. There are no local-time fields in the JSON response. The four time fields per message map as follows:

| API field | Bi-temporal axis | Notes |
|---|---|---|
| `eventStartTime` | Valid time start | When unavailability begins |
| `eventEndTime` | Valid time end | When unavailability ends |
| `publishTime` | Transaction time | When market learnt of the message |
| `createdTime` | Near-transaction time | Seconds before publishTime; use publishTime for as-of queries |
| *(not in API)* | Ingestion time | Must be stamped by the pipeline at fetch time |

### Asset identifiers and BMU linkage

`assetId` uses the Elexon BMU classification prefix: `T_` (transmission-connected), `E_` (embedded), `2__` (interconnector). The `affectedUnit` field is the same identifier without the prefix, and corresponds to `nationalGridBmUnit` in the BMU reference data (`GET /reference/bmunits/all`). The BMU reference table also carries `elexonBmUnit` (the prefixed form), EIC code, capacity, and fuel type. The mapping is stable across revisions for the same physical unit.

There is no station-level identifier — a multi-unit plant (e.g. Drax) appears as separate `assetId` values per unit (`T_DRAX-1` through `T_DRAX-6`).

### Archive depth

REMIT data is available back to approximately 2014–2015. For a project training on 2018+ data, a 7-year archive (2018–2025) contains roughly 365 × 7 × 125 = ~320,000 rows. As Parquet with compression, well under 100 MB. A full backfill at 1 call per day requires approximately 2,555 API calls — allow 1–2 hours at a polite rate.

### Smallest sensible fixture query

```
GET https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT
    ?publishDateTimeFrom=2024-01-15T10:00:00Z
    &publishDateTimeTo=2024-01-15T12:00:00Z
```

This 2-hour window returned 15 records in live testing — small enough to commit as a test fixture, large enough to cover multiple fuel types and at least one revision chain.

---

## Bi-temporal modelling patterns

### Terminology mapping for REMIT

The intent document names three times. Their mapping to standard vocabulary:

| Intent phrasing | Standard term | Fowler term | SQL:2011 term | Field |
|---|---|---|---|---|
| When the event is scheduled to occur | Valid time | Actual time | `PERIOD FOR VALID_TIME` | `event_start`, `event_end` |
| When the message was published | Transaction time | Record time | `PERIOD FOR SYSTEM_TIME` | `publish_time` |
| When we retrieved the message | Ingestion time | *(not in SQL:2011)* | *(project-specific)* | `ingested_at` |

The "as-of" query — "what did the market know at time T?" — operates on **transaction time (`publish_time`)**: find all records where `publish_time <= T`, group by `mrid`, and take the highest `revisionNumber`.

### As-of query in a Parquet-backed store

The recommended pattern is **append-only storage with query-time materialisation**:

```
Schema (one row per API record):
  mrid               string         -- event chain grouping key
  revision_number    int32          -- ordering within chain
  publish_time       timestamp[UTC] -- transaction time
  event_start        timestamp[UTC] -- valid time start
  event_end          timestamp[UTC] -- valid time end
  ingested_at        timestamp[UTC] -- pipeline stamp at fetch
  event_status       string         -- "Active" | "Dismissed"
  fuel_type          string
  normal_capacity    float64
  available_capacity float64
  unavailable_capacity float64
  ... (all other structured fields)
  related_information string        -- free text, verbatim
```

As-of query in pandas:

```python
known = df[df["publish_time"] <= T]
latest = (
    known
    .sort_values("revision_number")
    .groupby("mrid")
    .last()
    .reset_index()
)
active = latest[latest["event_status"] == "Active"]
```

For large stores, partition the Parquet by `publish_time` date and maintain an incremental "latest-per-mrid" materialised view alongside the raw store.

### Worked example: revision chain with as-of queries

Consider a gas unit with three messages for the same event:

| publish_time | mrid | revision_number | event_start | event_end | event_status | available_capacity |
|---|---|---|---|---|---|---|
| 2024-03-01T08:00Z | ABC-001 | 1 | 2024-03-02T06:00Z | 2024-03-04T06:00Z | Active | 0 MW |
| 2024-03-01T10:00Z | ABC-001 | 2 | 2024-03-02T06:00Z | 2024-03-05T06:00Z | Active | 0 MW |
| 2024-03-04T14:00Z | ABC-001 | 3 | 2024-03-02T06:00Z | 2024-03-05T06:00Z | Dismissed | 435 MW |

- **as-of(2024-03-01T09:00Z):** only revision 1 known → unit offline 2–4 March.
- **as-of(2024-03-01T11:00Z):** revisions 1 and 2 known → latest is r2 → unit offline 2–5 March (prolongation).
- **as-of(2024-03-05T00:00Z):** all revisions known → latest is r3, Dismissed → no active outage.

A "latest-wins overwrite" store answers the present-day query correctly but **cannot answer the 2024-03-01T09:00Z query** — it would incorrectly report the unit offline until 5 March. This breaks any backtest re-simulating the market's knowledge at a historical date.

### "One event" across a chain: project judgement call

The mRID groups all revisions of a single original notification. There is no settled industry answer to whether multiple revisions of the same physical event constitute "one event" for downstream modelling. The reasonable project choices are:

- **Stage 13 storage**: store all rows; mRID is the grouping key.
- **Stage 14 (LLM extraction)**: extract from each revision's `relatedInformation` separately; track which extraction belongs to which revision.
- **Stage 16 (feature table)**: the as-of view collapses to one row per mRID at query time; treat mRID as the "event" unit for features.

Treating the mRID chain as "one event" with an append-only raw store is consistent with event sourcing practice and the most auditable choice.

---

## Reference implementations and prior art

### ElexonDataPortal (OSUKED/ElexonDataPortal)

- **Language:** Python. **Licence:** MIT.
- **Scope:** Wraps the **old BMRS API** (decommissioned May 2024). Includes `MessageListRetrieval` and `MessageDetailRetrieval` for REMIT, with parameters: `EventStart`, `EventEnd`, `PublicationFrom`, `PublicationTo`, `ParticipantId`, `MessageID`, `AssetID`, `EventType`, `FuelType`, `MessageType`, `UnavailabilityType`, `AffectedUnitID`, `ActiveFlag`.
- **Cache discipline / schema:** None built-in; raw DataFrames.
- **Status for Stage 13:** Targets a dead API. Valuable as vocabulary reference for filter parameters but not suitable for direct use.

### elexonpy (openclimatefix-archives/Elexonpy)

- **Language:** Python (Swagger Codegen). **Licence:** MIT.
- **Scope:** Full Insights Solution API (new API). Confirms base URL `https://data.elexon.co.uk/bmrs/api/v1` and the complete REMIT endpoint family.
- **Cache discipline / schema:** Auto-generated models; no Pydantic or type-safety beyond generated stubs.
- **Status for Stage 13:** Useful for endpoint and parameter discovery; not maintained as a first-class library.

### IRIS clients (elexon-data/iris-clients)

- **Language:** Python 3.11+, Node.js 20+, C#/.NET 10+.
- **Scope:** Near-real-time AMQP push service (Azure Service Bus). Requires credentials (Client ID, Client Secret, Queue Name).
- **Status for Stage 13:** Out of scope. Relevant for a future incremental update pattern (Stage 19 or later); Stage 13 targets batch pull.

### ENTSO-E Transparency Platform

- **URL:** https://transparency.entsoe.eu/
- **Scope / schema:** EU-wide generation unavailability data. Key schema parallels with Elexon REMIT: `StartTS`/`EndTS` ≈ `eventStartTime`/`eventEndTime`; `AvailableCapacity` ≈ `availableCapacity`; `PowerResourceEIC` ≈ `affectedUnitEIC`; `Version` ≈ `revisionNumber`; `UpdateTime` ≈ `publishTime`.
- **Key difference:** ENTSO-E uses a `Status` field with enumerated values (`Active`, `Cancelled`, `Withdrawn`) where Elexon uses `eventStatus: "Active"/"Dismissed"`.
- **GB availability:** GB data ceased 15 June 2021. Not a usable source for Stage 13.

---

## Risks and sharp edges

### The old BMRS API is dead

`api.bmreports.com` was decommissioned 31 May 2024. Any tutorial, Stack Overflow answer, or library targeting that URL is stale. The implementer must use `data.elexon.co.uk/bmrs/api/v1`.

### Query window constraint on `/datasets/REMIT`

The non-streaming endpoint enforces a **1-day maximum window** (two-day windows return HTTP 400 — confirmed empirically). A 7-year backfill requires iterating day-by-day. Use `/datasets/REMIT/stream` for all bulk fetches; it showed no window constraint and no row cap in testing.

### Volume and backfill time

A single day yields approximately 100–150 rows. Full 7-year archive (2018–2025): ~320,000 rows, ~400–640 MB raw JSON, well under 100 MB as compressed Parquet. A daily-iteration backfill (2,555 API calls) takes 1–2 hours at a polite rate. Commit a curated 30-day slice as a test fixture; the full archive populates on first live run.

### Timezone handling

All API timestamps are **UTC with `Z` suffix** — no local-time fields. However, `eventStartTime` and `eventEndTime` often align to UTC midnight or UTC 22:00/23:00, reflecting the GB settlement day boundary (which shifts with BST/GMT). Store all timestamps as `datetime64[ns, UTC]`; conversion to `Europe/London` is a feature-engineering concern (Stage 16), not an ingestion concern.

### No explicit supersedes link

The API contains no `supersedes` or `supersededBy` field. Revision linkage is solely via shared `mrid` + incrementing `revisionNumber`. The as-of view must be derived by the consumer.

### Free-text encoding in `relatedInformation`

Some records contain an informal schedule encoding in the `relatedInformation` field: `"~/E_DERBY-1,2024-01-16 23:30:00,2024-01-17 06:00:00,0"`. This is not documented and may vary by participant system. Stage 13 must store it verbatim; Stage 14 must handle it gracefully.

### Two API round-trips to reconstruct a full revision chain

`/remit/revisions?mrid=...` returns only stub metadata (id, mrid, revisionNumber, publishTime, createdTime), not the full message. A full chain reconstruction requires fetching each revision from `/remit/{id}`. For bulk backfills, `/datasets/REMIT/stream` already includes all revisions in the time window — prefer it.

### Authentication future risk

The API is currently unauthenticated. Elexon previously required scripting keys (old BMRS). If authentication is introduced, the project's `BRISTOL_ML_<SCREAMING_SNAKE>` environment variable pattern will accommodate it without code changes — but any hardcoded "no-key required" assumption will need revisiting.

### Deprecated endpoints and libraries

- `ElexonDataPortal`'s `Client(api_key)` signature and BMRS URL are dead.
- `api.bmreports.com` is decommissioned.
- Old REMIT parameters `ActiveFlag`, `SequenceId`, `AffectedUnitID` (from BMRS) do not map directly to Insights API query parameters.

---

## Citations

| Source | URL | One-line summary |
|---|---|---|
| Elexon — Insights API Developer Portal | https://developer.data.elexon.co.uk/ | Official portal; confirms no auth required; base URL |
| Elexon — REMIT endpoint docs | https://bmrs.elexon.co.uk/api-documentation/endpoint/remit | Opinionated REMIT endpoint family |
| Elexon — GET /datasets/REMIT | https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/REMIT | Dataset endpoint documentation |
| Elexon — REMIT revisions endpoint | https://bmrs.elexon.co.uk/api-documentation/endpoint/remit/revisions | All revisions for a given mRID |
| Elexon — list/by-publish | https://bmrs.elexon.co.uk/api-documentation/endpoint/remit/list/by-publish | List messages by publish time |
| Elexon — list/by-event | https://bmrs.elexon.co.uk/api-documentation/endpoint/remit/list/by-event | List messages by event time |
| Elexon — New REMIT portal announcement | https://www.elexon.co.uk/bsc/article/new-and-improved-remit-portal-live-on-the-insights-solution/ | BMRS decommissioned 31 May 2024; new portal launched |
| Elexon — Live /datasets/REMIT response | https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT?publishDateTimeFrom=2024-01-15T10:00:00Z&publishDateTimeTo=2024-01-15T12:00:00Z | Confirmed full response schema in live call |
| Elexon — Live /remit/search response | https://data.elexon.co.uk/bmrs/api/v1/remit/search?mrid=48X000000000045Z-NGET-RMT-00131535 | Confirmed full field set |
| Elexon — Live fuel types | https://data.elexon.co.uk/bmrs/api/v1/reference/remit/fueltypes/all | Confirmed 12 fuel type values |
| elexonpy on PyPI | https://pypi.org/project/elexonpy/0.0.4/ | Swagger-codegen client; confirms base URL and endpoint list |
| Ofgem — REMIT and wholesale market integrity | https://www.ofgem.gov.uk/energy-policy-and-regulation/policy-and-regulatory-programmes/remit-and-wholesale-market-integrity | Ofgem role as GB REMIT enforcer post-Brexit |
| Ofgem — Publishing inside information | https://www.ofgem.gov.uk/policy/publishing-inside-information-under-remit-article-4 | 100 MW threshold concerns; Article 4 obligations |
| Norton Rose Fulbright — REMIT after Brexit | https://www.nortonrosefulbright.com/en/knowledge/publications/5ba53626/remit-after-brexit | GB vs EU REMIT differences post-Brexit |
| Shakespeare Martineau — Revised REMIT 2024 | https://www.shma.co.uk/our-thoughts/revised-remit-what-you-need-to-know/ | Revised EU REMIT May 2024; does not affect GB |
| Emissions-EUETS — Inside information | https://www.emissions-euets.com/obligation-to-publish-inside-information | Full list of REMIT UMM required fields (ACER schema) |
| Martin Fowler — Bitemporal History | https://martinfowler.com/articles/bitemporal-history.html | Actual vs record time; retroactive correction examples |
| Wikipedia — Bitemporal modelling | https://en.wikipedia.org/wiki/Bitemporal_modeling | Valid time / transaction time; SQL:2011 |
| OSUKED — ElexonDataPortal | https://github.com/OSUKED/ElexonDataPortal | Python client for old BMRS API; MIT; REMIT parameter vocabulary |
| MichaelKavanagh/elexon — methods.py | https://github.com/MichaelKavanagh/elexon/blob/master/elexon/methods.py | REMIT method parameters from old BMRS API |
| elexon-data — IRIS clients | https://github.com/elexon-data/iris-clients | Real-time AMQP push clients; Python/Node/C# |
| elexon-data — insights-docs | https://github.com/elexon-data/insights-docs | Developer-facing docs; confirms public API |
| ENTSO-E — Unavailability of Production Units | https://transparency.entsoe.eu/content/static_content/Static%20content/knowledge%20base/data-views/outage-domain/Data-view%20Unavailability%20of%20Production%20and%20Generation%20Units.html | ENTSO-E schema; comparison basis |
| OpenMod — UK data leaving ENTSO-E | https://groups.google.com/g/openmod-initiative/c/1JGZYhuV76s | GB ENTSO-E data ceased 15 June 2021 |
| GB REMIT Central Collection Service | https://www.remit.gb.net/ | Gas REMIT publication (electricity via Elexon) |
| Elexon — REMIT XML Implementation Guide v1.2 | https://assets.elexon.co.uk/wp-content/uploads/sites/11/2014/08/28164207/REMIT-XML-Implementation-Guide-v1.2.pdf | Original GB REMIT XSD; mRID definition |

---
