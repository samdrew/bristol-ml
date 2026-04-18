# Stage 1 — NESO demand ingestion + first plot

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 0
**Enables:** Stage 3 (feature assembler)

## Purpose

Bring real GB electricity demand data into the project and produce the first plot. This is the shortest path from "empty repo" to "something visible on screen," and it establishes the ingestion pattern every subsequent feed will follow.

## Scope

In scope:
- A module that retrieves the NESO Historic Demand Data resource, parses it, and persists it locally in a typed columnar form.
- Configuration describing which years to retrieve and how the fetch-versus-cache choice is expressed.
- A module-level Claude Code guide describing the ingestion conventions so later ingestion stages can follow the same shape.
- A notebook that loads the cached demand data, aggregates to hourly, and plots a human-legible view (something like a week of hourly demand plus a year of daily peaks).
- Tests against recorded API fixtures so the module can be exercised offline.

Out of scope:
- Weather data (Stage 2).
- Any feature engineering.
- The day-ahead forecast archive (arrives with Stage 4 when it becomes the benchmark).
- Intraday or real-time feeds.

## Demo moment

From a clean clone, a facilitator runs the notebook and sees real GB demand plotted within a small number of minutes. The plot shows the twin-peak shape of daily electricity demand, and a day's worth of settlement periods is legible. If the cache is present, no network call happens.

## Acceptance criteria

1. Running the ingestion with a cache present completes offline.
2. Running the ingestion without a cache fetches from the NESO CKAN API and writes a local copy for subsequent runs.
3. Running the ingestion twice in a row produces the same on-disk result.
4. The output schema is documented in the module's Claude Code guide.
5. The notebook runs top-to-bottom quickly on a laptop.
6. Tests exercise the public interface of the module using recorded fixtures.

## Points for consideration

- NESO publishes one CSV resource per year, each identified by a UUID. The mapping from year to resource ID is not derivable — it's listed on the dataset page. How that mapping is captured in configuration affects how painful adding a new year is.
- Half-hourly data contains settlement periods 1-46 or 1-50 on clock-change days, not 1-48. Any aggregation to hourly has to handle this. The spring-forward day in March is a useful test case.
- `ND` (National Demand) excludes station load, pump storage, and interconnector exports; `TSD` (Transmission System Demand) includes them. `ND` is the standard day-ahead forecasting target and matches the NESO benchmark. Either can be loaded, but the framing from DESIGN §5 picks one.
- The NESO columns `EMBEDDEDWINDGENERATION` and `EMBEDDEDSOLARGENERATION` estimate generation invisible to transmission metering. As rooftop solar grows, `ND` understates true consumption by that amount. This is not something to fix in Stage 1, but a facilitator demoing the data will likely be asked about it; a note in the notebook commentary is cheap.
- Settlement times are in UK local time, so there is a choice to make about whether to store timestamps as tz-aware UTC, as local time, or both. Downstream stages will want unambiguity.
- Caching stance: cache on first fetch to a location outside the repo, so that the repo itself stays small and clones are fast. A portable cache archive (so one machine can seed another) is a plausible follow-up but not a requirement.
- The CKAN API is usually reliable but occasionally slow. For a live-demo context, a bounded retry with a clear failure message is probably more valuable than silent resilience.
- Whether the module should default to "cache if present, fetch if not" or require an explicit choice. The former is friendlier for notebooks; the latter is more honest about what's happening.

## Dependencies

Upstream: Stage 0.

Downstream: Stage 3 consumes the cached demand data as the target time series. Stage 4 re-uses this module's scaffolding when it adds the day-ahead forecast archive.

## Out of scope, explicitly deferred

- Day-ahead forecast archive (Stage 4).
- Settlement prices or any other NESO feed beyond demand.
- Embedded generation as a modelling input.
