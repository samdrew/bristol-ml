# Stage 13 — REMIT ingestion

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 5
**Enables:** Stage 14 (LLM extractor), and through it Stages 15 and 16

## Purpose

Bring the textual disclosure stream into the project. REMIT is the regulatory mechanism by which market participants must publish events that affect supply or demand — plant outages, maintenance windows, availability changes — before trading on them. For a forecasting project, REMIT is the richest source of "things the market knows that plain time-series history does not encode." This stage handles ingestion, persistence, and the bi-temporal storage discipline that any event-based data source needs.

## Scope

In scope:
- A module that retrieves REMIT messages from the Elexon Insights API and persists them locally.
- A data model that preserves three times per event: when the event is scheduled to occur (effective-from / effective-to), when the message was published, and when the message was retrieved by this project. These three times support correct "as-of" queries — what did the market know at time T? — which will be needed for forecasting backtests later.
- A mechanism for handling message revisions and supersedes, because REMIT messages are routinely updated and the latest-published message is not always the one that was active at a prior moment.
- A notebook that visualises REMIT event density over time, coloured by fuel type or event type, so a facilitator can see when the market has been signalled about large unavailable capacity.
- Tests against recorded API fixtures.

Out of scope:
- Any extraction from the free-text description — that's Stage 14.
- Any joining of REMIT features into the feature table — that's Stage 16.
- Embedding or semantic search — that's Stage 15.
- The Stage 15 vector index.

## Demo moment

A chart of aggregate unavailable capacity over time, by fuel type. Clear spikes around planned outage seasons; large downward steps around unplanned events. A facilitator can point at a specific spike and say "that was a nuclear unit going offline on this date."

## Acceptance criteria

1. The module can answer the question "what REMIT events were known to the market at time T?" correctly for any T within the cached period, including revisions.
2. Running the ingestion with a cache present completes offline.
3. Running the ingestion without a cache fetches from Elexon and populates the cache.
4. The visualisation notebook renders a meaningful summary of events over time.
5. Tests cover the bi-temporal query discipline, not just the fetch.

## Points for consideration

- REMIT messages are numerous. A full historical archive may be large enough that committing it to the repo is a bad idea; a curated sample for testing is more portable. Caching elsewhere on the machine, with a documented way to populate it, is the likely pattern.
- The bi-temporal model is the hardest design decision in this stage. At minimum, the storage needs to distinguish "when did the market learn" from "when is the event effective" from "when did we retrieve the message." An "as-of" query needs all three.
- Message revisions. REMIT messages can be corrected, withdrawn, or superseded. A naive "latest message wins" approach gives the wrong answer for any historical query. A bi-temporal store handles this naturally but adds complexity.
- Elexon's API returns messages structured with fields (asset, type, MW, times) plus a free-text description. The structured fields are what the ingestion stage preserves; the free text is what Stage 14 will extract from.
- What counts as "one event" across messages. A single outage may generate an initial message, a revision with better MW estimates, a prolongation notice, and a final "event concluded" notice. Whether to treat these as one row or many is a design call with consequences for Stage 14 and Stage 16.
- Archive depth. REMIT goes back to 2014 or so in Elexon's archive. For a demand model trained on 2018 onwards, the older history is not directly useful; for a price model (Stage 17) it might be.
- The Elexon Insights API has rate limits and paging. A slow but reliable bulk-fetch is better than a fast-but-flaky one.
- Whether to normalise asset identifiers against a plant database so a user can ask "show me all REMIT events for Hinkley Point B." This is a richer feature but adds a dependency on a master-data source.

## Dependencies

Upstream: Stage 5 (the ingestion pattern and cache discipline; Stage 13 does not depend on modelling stages).

Downstream: Stage 14 (LLM extractor consumes REMIT free text), Stage 16 (model with REMIT features).

## Out of scope, explicitly deferred

- LLM-based extraction (Stage 14).
- Embedding index (Stage 15).
- Feature joins into the model (Stage 16).
- Plant master-data normalisation.
