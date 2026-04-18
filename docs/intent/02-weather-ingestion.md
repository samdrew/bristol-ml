# Stage 2 — Weather ingestion + joined plot

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 0, Stage 1
**Enables:** Stage 3 (feature assembler joins weather to demand)

## Purpose

Fetch hourly historical weather for a configurable list of UK population centres, aggregate those point observations into a single national weather signal, and produce the joined temperature-against-demand scatter. This stage proves the ingestion pattern generalises beyond a single source and lands the first analytical insight: the non-linear relationship between temperature and electricity demand.

## Scope

In scope:
- A module that retrieves hourly historical weather from Open-Meteo for a configured list of stations and persists it locally per station.
- A transformation that composes per-station hourly data into a national weighted signal, with weights expressible in configuration.
- Configuration naming the stations, their weights, and the weather variables of interest.
- A notebook that joins national demand (from Stage 1) with the national weather aggregate, produces the temperature-versus-demand scatter, and overlays a smooth fit.
- Tests against recorded API fixtures.

Out of scope:
- Forecast weather data (deferred until a serving stage needs it).
- Derived weather features like cooling / heating degree days.
- Grid-point weather or any attempt at spatial modelling.

## Demo moment

A scatter of hourly national temperature against national hourly demand, one year of data, with a smooth fit overlaid. The non-linear V-shape is visible: demand rises at both cold and hot extremes, with a minimum in the mild-temperature range. This is the pedagogical hook for the linear regression stage.

## Acceptance criteria

1. Running the ingestion with a cache present completes offline.
2. Running the ingestion without a cache fetches all configured stations.
3. The national aggregation accepts any subset of the configured station list, so a demo can run with fewer stations to show the effect.
4. The notebook runs top-to-bottom quickly on a laptop.
5. The notebook's commentary motivates the choice of Open-Meteo over Met Office DataHub briefly.
6. Smoke test for the fetcher; a test for the aggregation that asserts equal weights on identical inputs yield the identity.

## Points for consideration

- Station selection. A handful of population centres is enough to proxy a national weighted temperature; the alternative (one point per local authority, say) is expensive and probably unrewarding. What counts as "enough" is a judgement call.
- Weighting rationale. Population weighting is defensible for a demand model because demand is dominated by population centres. Uniform weighting is less defensible but simpler. Whatever the choice, the rationale and the source of the weights should live alongside them.
- Variables to fetch. Temperature is the headline driver; wind speed, cloud cover, and solar irradiance become relevant once the project reaches the price tier (Stage 17) or tries to model embedded generation. Fetching them now is cheaper than re-fetching later.
- Dew point vs relative humidity. Dew point aggregates better across stations than relative humidity does. Either is fine for modelling demand, but the choice has consequences for Stage 3.
- Open-Meteo composes multiple underlying weather models. Older historical data (pre-UKV) is at coarser resolution than recent data. Worth flagging in module commentary so a facilitator isn't surprised.
- Time zones. Open-Meteo returns UTC by default. Matching the convention chosen in Stage 1 for NESO data keeps joins simple.
- Aggregation order. Population-weighted mean of per-station temperatures is defensible for demand forecasting. Temperature-weighted mean of populations is not. The pedagogical value of getting this right is that it comes up naturally in the notebook.
- Rate limits. The free tier is generous for this volume but documented. Worth a look before heavy fetches.
- Caching stance. Cached per station per year on first fetch, outside the repo. Same shape as Stage 1.

## Dependencies

Upstream: Stage 0, Stage 1 (the ingestion pattern this stage follows).

Downstream: Stage 3 (feature assembler), and every modelling stage that uses weather as an exogenous driver.

## Out of scope, explicitly deferred

- Forecast weather (later, when needed for serving).
- Sub-hourly interpolation.
- Met Office DataHub (rejected: the 48-hour free-tier historical window rules it out for training).
