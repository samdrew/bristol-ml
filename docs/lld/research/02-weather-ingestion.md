# Stage 2 — Weather ingestion research

**Access date:** 2026-04-18
**Author:** `@researcher` (facts only; no recommendation)
**Format:** Question → Finding → Citation → Implication

This note gathers facts behind Stage 2 (Open-Meteo ingestion + national aggregation).
Each question maps to a bullet in the spawn brief. Where sources disagree, the
disagreement is surfaced in-line rather than smoothed over.

---

## 1. Archive API shape

**Finding.** Endpoint: `https://archive-api.open-meteo.com/v1/archive`. Required
parameters: `latitude`, `longitude`, `start_date` (ISO `yyyy-mm-dd`), `end_date`.
Key optional parameters: `hourly` (array of variable names, repeated or
comma-separated), `daily`, `timezone` (default `GMT`; accepts `auto` and IANA
names), `temperature_unit`, `wind_speed_unit`, `timeformat` (`iso8601` |
`unixtime`), `cell_selection` (`land` | `sea` | `nearest`), `models` (to pin a
reanalysis model — `era5`, `era5_land`, `era5_seamless`, `cerra`, `ecmwf_ifs`),
`apikey` (commercial only), and vectorised lists of `latitude`/`longitude`
(batching).

The response is a single JSON object, not paginated. Shape:

```json
{
  "latitude": …, "longitude": …, "elevation": …,
  "generationtime_ms": …, "utc_offset_seconds": 0,
  "timezone": "GMT", "timezone_abbreviation": "GMT",
  "hourly": {
    "time": ["2023-01-01T00:00", "2023-01-01T01:00", …],
    "temperature_2m": [4.2, 4.1, …],
    "cloud_cover": [100, 93, …]
  },
  "hourly_units": {"temperature_2m": "°C", …}
}
```

The `hourly` arrays are flat and index-aligned with `time`. Docs specify no
maximum date range per call, but the rate-limit accountant uses a quadratic
weight, `weight = n_locations × (n_days / 14) × (n_variables / 10)`, so a 5-year
window of 5 variables for one station ≈ `1 × (1826/14) × (5/10)` ≈ 65 "calls".

**Citation.**
- [Historical Weather API docs](https://open-meteo.com/en/docs/historical-weather-api) (accessed 2026-04-18).
- [Weather data for multiple locations at once (Open-Meteo blog)](https://openmeteo.substack.com/p/weather-data-for-multiple-locations) (accessed 2026-04-18).

**Implication.** A single call per station covering the full 2018-onwards
training window is well within the per-call contract, but needs to be costed
against the rate-limit accountant (≥2 weeks × >10 variables triggers the
fractional-call charge). Pagination logic is not required; schema-assertion
logic is.

---

## 2. Rate limits and auth

**Finding.** Free-tier caps on `open-meteo.com` non-commercial endpoints:
**600 calls/minute, 5 000/hour, 10 000/day, 300 000/month**. No API key, no
registration, no credit card; throttling is IP-based. Requests for `>10`
variables or `>2 weeks` of data at one location count as fractional multiples
rather than one. Commercial traffic uses a separate host
(`customer-api.open-meteo.com`) with `&apikey=` — this project does not need
that tier.

**Citation.**
- [Open-Meteo Terms](https://open-meteo.com/en/terms) (accessed 2026-04-18).
- [Open-Meteo Pricing](https://open-meteo.com/en/pricing) (accessed 2026-04-18).
- [GitHub discussion #853 on IP-based rate limiting](https://github.com/open-meteo/open-meteo/discussions/853) (accessed 2026-04-18).

**Implication.** Ten stations × ~7 years × ~5 variables is roughly 10 × (2555/14)
× (5/10) ≈ 910 weighted calls — one cold fetch, comfortably inside the 10 000/day
cap. Subsequent runs hit local cache. The CI-default `CachePolicy.OFFLINE`
already guards against accidental refetches in automation.

---

## 3. Variable names

**Finding.** Open-Meteo uses consistent snake_case variable names across archive
and forecast endpoints. The five names for the spawn brief's headline variables
are:

| Intent | API name | Units |
|---|---|---|
| 2 m air temperature | `temperature_2m` | °C |
| 2 m dew point | `dew_point_2m` | °C |
| 10 m wind speed | `wind_speed_10m` | km/h (default; settable) |
| Cloud cover (total) | `cloud_cover` | % |
| Shortwave solar irradiance | `shortwave_radiation` | W/m² |

**Note of drift.** Historically Open-Meteo used `dewpoint_2m`, `windspeed_10m`,
`cloudcover`; these were deprecated in favour of the snake-case spellings above
around v1.2 of the API (2023). Both forms are still accepted server-side but the
snake-case forms are authoritative per current docs. Also relevant but not in
the brief: `direct_radiation`, `diffuse_radiation`, `direct_normal_irradiance`,
`relative_humidity_2m`, `surface_pressure`, `precipitation`, `wind_gusts_10m`.

**Citation.**
- [Historical Weather API — variable list](https://open-meteo.com/en/docs/historical-weather-api) (accessed 2026-04-18).

**Implication.** The brief's phrasing "dew point, wind speed" maps to concrete
tokens unambiguously. Stage 2's config YAML should pin the modern spellings;
fixtures recorded against older spellings will need re-recording.

---

## 4. Underlying data model for UK archive

**Finding — and a correction to DESIGN §4.2.** The archive endpoint
(`archive-api.open-meteo.com/v1/archive`) serves ERA5 / ERA5-Land / CERRA / ECMWF
IFS reanalyses — **not** the UKMO UKV 2 km model. Resolutions:

- ERA5: global, ~0.25° (~25 km), hourly, 1940–present, ~5-day delay.
- ERA5-Land: global land, ~0.1° (~11 km), hourly, 1950–present, ~5-day delay.
- CERRA: Europe only, 5 km, hourly, **1985 – June 2021** (archived, no updates).
- ECMWF IFS: 9 km, 2017–present. Higher-accuracy but inconsistent with earlier
  years.
- Default model (`era5_seamless`) blends ERA5-Land surface fields with ERA5
  upper-air/solar/wind, auto-falling-back to ERA5 when ERA5-Land lacks a variable.

The UKMO UKV 2 km model **is** exposed by Open-Meteo, but through the separate
[historical-forecast-api](https://open-meteo.com/en/docs/historical-forecast-api)
(which concatenates past initialisations of live NWP models) and
[ukmo-api](https://open-meteo.com/en/docs/ukmo-api). That archive **begins
2022-03-01** and carries a rolling ~2-year window — so it cannot supply the
2018-onwards training period that Stage 1's NESO data covers.

**Resolution drift over time** (worth flagging in the notebook):

- Pre-1950: ERA5 only, ~25 km.
- 1950–1984: ERA5-Land available, ~11 km over land.
- 1985–June 2021: CERRA available for Europe at 5 km (if explicitly requested).
- 2017–present: IFS 9 km operational archive.
- 2022-03-01–present: UKV 2 km via historical-forecast-api only.

**Citation.**
- [Historical Weather API docs](https://open-meteo.com/en/docs/historical-weather-api) (accessed 2026-04-18).
- [Historical Forecast API docs](https://open-meteo.com/en/docs/historical-forecast-api) (accessed 2026-04-18).
- [UK Met Office API docs (Open-Meteo)](https://open-meteo.com/en/docs/ukmo-api) (accessed 2026-04-18).
- [Open-Meteo blog: Processing 90 TB historical weather data](https://openmeteo.substack.com/p/processing-90-tb-historical-weather) (accessed 2026-04-18).
- [Met Office UKV on AWS registry](https://registry.opendata.aws/met-office-uk-deterministic/) (accessed 2026-04-18).

**Disagreement note.** DESIGN.md §4.2 claims "~10 km via UKMO UKV 2 km model".
That is incorrect on both counts for the archive endpoint: the archive uses
ERA5/ERA5-Land/CERRA, not UKV; the spatial resolution for the 2018-present UK
archive is ~9–11 km via ERA5-Land/IFS, not "UKV 2 km". A spec-drift note
should surface this before implementation begins (per CLAUDE.md spec-drift rule).

**Implication.** Two live choices for Stage 2, both documented:
- Pin `models=era5_seamless` (or leave unset) for 1940-compatible, consistent,
  coarser (~11 km) data across the whole training window.
- Pin `models=cerra` to get 5 km Europe-only data but lose everything after
  June 2021 — incompatible with a training window extending to today.
Either way, the spec line should be corrected. UKV 2 km is not reachable from
the archive endpoint for the training window.

---

## 5. Station list and population weights

**Finding — coordinates.** No single ONS publication gives "city-centre lat/lon"
for the ten population centres. The pragmatic primary sources are OSM-derived
city-centre points or ONS Built-Up Area centroids. Representative city-centre
coordinates (verify against chosen source before committing):

| City | Lat | Lon |
|---|---|---|
| London (Charing Cross ref.) | 51.5074 | -0.1278 |
| Birmingham | 52.4862 | -1.8904 |
| Manchester | 53.4808 | -2.2426 |
| Glasgow | 55.8642 | -4.2518 |
| Leeds | 53.8008 | -1.5491 |
| Bristol | 51.4545 | -2.5879 |
| Cardiff | 51.4816 | -3.1791 |
| Belfast | 54.5973 | -5.9301 |
| Edinburgh | 55.9533 | -3.1883 |
| Newcastle upon Tyne | 54.9784 | -1.6174 |

These match Wikipedia/OSM city infoboxes to ~0.01°, which is well inside the
9–11 km ERA5/ERA5-Land cell used by Open-Meteo — so ±0.05° of jitter on the
chosen point is analytically irrelevant. Open-Meteo's `cell_selection=land`
default snaps to the nearest land cell when coordinates fall near water.

**Finding — populations.** The ONS 2021 Census for England & Wales reports
figures by Built-Up Area, but — as of 2025 and still 2026 — equivalent
UK-wide urban-area tables comparable to the 2011 Census have not been published;
Scotland's Census was held in 2022, Northern Ireland's in 2021. Wikipedia's
[list of UK urban areas](https://en.wikipedia.org/wiki/List_of_urban_areas_in_the_United_Kingdom)
still cites 2011 Census figures. Approximate 2011 Census built-up-area
populations:

| Area (2011 BUA) | Population |
|---|---|
| Greater London | 9 787 426 |
| West Midlands (Birmingham) | 2 440 986 |
| Greater Manchester | 2 553 379 |
| West Yorkshire (Leeds) | 1 777 934 |
| Greater Glasgow | 957 620 |
| Tyneside (Newcastle) | 774 891 |
| Bristol | 617 280 |
| Belfast | 595 879 |
| Edinburgh | 482 270 |
| Cardiff | 447 287 |

**Citation.**
- [ONS: Towns and cities, Census 2021](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/townsandcitiescharacteristicsofbuiltupareasenglandandwalescensus2021) (accessed 2026-04-18).
- [List of urban areas in the United Kingdom — Wikipedia](https://en.wikipedia.org/wiki/List_of_urban_areas_in_the_United_Kingdom) (accessed 2026-04-18; 2011 Census figures).
- [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) (accessed 2026-04-18).

**Implication.** Wikipedia's 2011 urban-area table is reputable-enough and
citeable. A Stage 2 config should record the exact source and year of each
figure in the YAML alongside the weight, so a future refresh to 2021/2022
figures is mechanical.

---

## 6. Time-zone handling

**Finding.** With `timezone=GMT` (the API default) or `timezone=UTC`, the
response is in UTC. The `time` array is one ISO-8601 string per hour with no
DST jump — UTC has no DST. The array is flat (not per-day-paginated) and
contiguous from `start_date` 00:00 to `end_date` 23:00 inclusive. Setting
`timezone=Europe/London` instead returns local-time strings with DST
discontinuities (the 01:00 on the spring-forward Sunday is absent; the fallback
Sunday has duplicate 01:00 entries). Open-Meteo's own docs state "If timezone
is set, all timestamps are returned as local-time and data is returned starting
at 00:00 local-time."

**Citation.**
- [Historical Weather API docs — timezone section](https://open-meteo.com/en/docs/historical-weather-api) (accessed 2026-04-18).
- [Open-Meteo issue #850 on timezone behaviour](https://github.com/open-meteo/open-meteo/issues/850) (accessed 2026-04-18).

**Implication.** Requesting `timezone=UTC` (or omitting it) keeps Stage 2 in
lockstep with Stage 1's canonical `timestamp_utc` column and removes all DST
edge-case handling from the ingester. Stage 1's settlement-period DST algebra
has no analogue here — Open-Meteo weather is hourly and tz-naïve in UTC.

---

## 7. Caching granularity

**Finding.** Open-Meteo's own Python-client examples use an HTTP-level cache
(`requests-cache`) that stores response bytes keyed by URL, rather than a
domain-level parquet cache. There is no published guidance on "parquet per
station per year" versus "one parquet per station" versus "one combined
parquet". Existing third-party patterns split across the three options:

- **Per (station, year)** — mirrors ERA5-CDS download conventions and makes
  incremental fetches trivial (one file per year as data extends). Cost:
  10 stations × 8 years = 80 small files; more filesystem chatter.
- **Per station, flat file** — matches Stage 1's single-parquet pattern for
  NESO demand. One file per station, overwritten on `REFRESH`. Cost: entire
  file rewrites even when only the latest month has changed.
- **One combined parquet with a `station` column** — queries by station need a
  `.filter`, but joins to the national aggregate are one-shot. Cost: a single
  ~50 MB file for 10 stations × 8 years × 24 h × 5 vars, comfortably under the
  layer's 1 GB partition threshold.

The Stage 2 intent line "cached per station per year on first fetch" is
ambiguous between options (1) and (2) — it could mean physical per-year files
**or** per-station files whose contents span multiple years accreted on first
fetch. Both Stage-1-style single-file and CDS-style per-year layouts satisfy
the letter of the intent.

**Citation.**
- [openmeteo-requests README — request-caching example](https://github.com/open-meteo/python-requests/blob/main/README.md) (accessed 2026-04-18).
- [Copernicus ERA5-Land time-series product page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries) (accessed 2026-04-18).
- [Ingestion layer architecture note — `docs/architecture/layers/ingestion.md`](/workspace/docs/architecture/layers/ingestion.md) — "flat single file until the dataset crosses ~1 GB".

**Implication (no recommendation).** The trade-off is incremental-refresh cost
vs file-count overhead vs join ergonomics. The layer note's 1 GB partition
threshold is not approached at Stage 2's scale, so all three options are
compatible with the layer contract. Decision belongs to the implementer in
consultation with the lead.

---

## 8. Existing Python clients

**Finding.** Open-Meteo publishes a supported Python client,
[`openmeteo-requests`](https://pypi.org/project/openmeteo-requests/) (v1.7.5 as
of Jan 2026). Architecture:

- Built on `niquests` (a `requests` fork), **not** `httpx`. It is **not**
  drop-in replaceable against an `httpx.Client`. Stage 1 uses `httpx`; adopting
  the official client therefore introduces a second HTTP stack.
- Transport uses FlatBuffers (host-negotiated via `Accept` header), with
  zero-copy numpy array extraction. For a handful of stations the zero-copy
  advantage is immaterial; it matters at multi-gigabyte scale.
- Does **not** provide a one-line "response → DataFrame". Caller assembles a
  `pd.date_range` from `time_start/time_end/time_step` and zips in each
  variable's numpy array.
- Requires the caller to bring their own `requests-cache` and `retry-requests`
  sessions (both external packages). That is the pattern Open-Meteo's docs
  recommend explicitly.

**Citation.**
- [openmeteo-requests on PyPI](https://pypi.org/project/openmeteo-requests/) (accessed 2026-04-18).
- [open-meteo/python-requests README](https://github.com/open-meteo/python-requests) (accessed 2026-04-18).
- [DeepWiki user guide — open-meteo/python-requests](https://deepwiki.com/open-meteo/python-requests/2-user-guide) (accessed 2026-04-18).

**Implication (no recommendation).** Two live paths:

- Adopt `openmeteo-requests` + `requests-cache` + `retry-requests`. Matches
  upstream's recommended pattern; adds three deps; introduces `niquests`
  alongside Stage 1's `httpx`.
- Hit the REST endpoint directly with `httpx` + `tenacity` (Stage 1's existing
  stack). Parses plain JSON; zero new deps; ~40 lines of retry glue already
  exist in `src/bristol_ml/ingestion/neso.py` and can be copy-adapted.

The HTTP-client choice is called out as a "swappable" item in the ingestion
layer architecture table — neither choice violates the contract.

---

## 9. Weighted-mean aggregation idioms

**Finding — pandas.** The canonical idiom for a grouped weighted mean is a
custom function fed to `groupby().apply()`:

```python
def wavg(group, value_col, weight_col):
    v = group[value_col]
    w = group[weight_col]
    mask = v.notna() & w.notna()
    return (v[mask] * w[mask]).sum() / w[mask].sum()
```

`numpy.average(values, weights=weights)` raises on NaN inputs, so the mask
must be applied before the `np.average` call. pandas has no built-in
`weighted_mean` aggregation (open since
[pandas#10030](https://github.com/pandas-dev/pandas/issues/10030), 2015).

**Finding — polars.** Polars also lacks a native weighted-mean aggregation
(see [valves#8](https://github.com/pola-rs/valves/issues/8)); the idiomatic
expression is `((pl.col("value") * pl.col("weight")).sum() /
pl.col("weight").sum())`, wrapped in `.group_by().agg(...)`.

**Finding — energy-forecasting precedent.** The closest formalised GB
population-or-demand-weighted weather variable is the National Gas
**Composite Weather Variable (CWV)**. It is a per-LDZ weighted combination of
temperature, wind speed and solar irradiation; the national CWV is a weighted
average of the 13 LDZ CWVs, with weights reflecting **gas demand** by LDZ (not
strictly population). CWV is the canonical weather input for **gas**
day-ahead demand forecasting in GB. For **electricity**, Thornton, Hoskins &
Scaife (2016) use the Central England Temperature (CET) record as a proxy
because "population and demand are weighted to the south of GB"; they do not
apply a formal population weighting to the temperature series. NESO's own
forecasting documentation is sparse on the exact weighting scheme used for
electricity, stating only that "temperature" and ~80 other inputs feed the
day-ahead model.

**Citation.**
- [Practical Business Python — "Learn More About Pandas By Building and Using a Weighted Average Function"](https://pbpython.com/weighted-average.html) (accessed 2026-04-18).
- [pandas-dev/pandas#10030 — "weighted mean" feature request](https://github.com/pandas-dev/pandas/issues/10030) (accessed 2026-04-18).
- [pola-rs/valves#8 — "Weighted aggregation"](https://github.com/pola-rs/valves/issues/8) (accessed 2026-04-18).
- [National Gas / Xoserve — Composite Weather Variable reference](https://umbraco.xoserve.com/media/4uvo4hpj/composite-weather-variable.pdf) (accessed 2026-04-18; binary PDF — content summarised from cross-references).
- [Gas Governance — CWV temperature & wind-speed analysis (2024)](https://www.gasgovernance.co.uk/sites/default/files/related-files/2024-01/CWV%20-%20Temp%20and%20Wind%20Speed%20Analysis.pdf) (accessed 2026-04-18).
- [Thornton, Hoskins, Scaife (2016), *Environmental Research Letters* 11:114015, "The role of temperature in the variability and extremes of electricity and gas demand in Great Britain"](https://iopscience.iop.org/article/10.1088/1748-9326/11/11/114015) (accessed 2026-04-18).
- [NESO — Day-Ahead National Demand Forecast](https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/day_ahead_national_demand_forecast) (accessed 2026-04-18).

**Disagreement note.** CWV uses **gas-demand weighting** across LDZs, not
population weighting. Thornton et al. use **no explicit weighting** (single
CET series). The stage-intent line "Population weighting is defensible for a
demand model because demand is dominated by population centres" is defensible
but not an industry-verified standard for **electricity** demand in GB. It is
best framed pedagogically (a sensible default) rather than as matching ESO
practice.

**Implication.** Stage 2 should ship the weighting scheme and the population
source inline in the YAML — not hard-coded — and the notebook narrative should
acknowledge that the industry precedent for a national weather signal is
CWV (gas-demand weighted), not population-weighted.

---

## 10. Non-linear temperature-vs-demand relationship

**Finding.** Thornton, Hoskins & Scaife (2016), analysing GB daily electricity
demand 1975–2013 against the CET record, report a **slightly non-linear**
relationship: strong linear anti-correlation below ~17 °C, levelling-off above
17 °C — a hockey-stick rather than a symmetric V. For winter months the
detrended correlation is *r* ≈ −0.81 (electricity) and −0.90 (gas). This is
historically asymmetric because GB has little residential air-conditioning;
recent work (Tandfonline 2025; Exeter 2022) flags that the hot-side arm of
the V is **emerging** as cooling-demand rises, particularly in London. CIBSE's
degree-day methodology (HDD base 15.5 °C historically, CDD base 22 °C) provides
the standard parametric form and base-temperature convention.

**Citation.**
- [Thornton, Hoskins & Scaife (2016), Env. Res. Lett. 11:114015](https://iopscience.iop.org/article/10.1088/1748-9326/11/11/114015) (accessed 2026-04-18).
- [Carbon Trust / Sustainability Exchange — "Degree days for energy management" (CIBSE-aligned guide)](https://www.sustainabilityexchange.ac.uk/files/degree_days_for_energy_management_carbon_trust.pdf) (accessed 2026-04-18).
- [Vesma — Standard heating/cooling degree days for the UK](https://vesma.com/ddd/std-year.htm) (accessed 2026-04-18).
- ["Hourly cooling demand prediction through a bottom-up model in London" — Int. J. Green Energy (2025)](https://www.tandfonline.com/doi/full/10.1080/15435075.2025.2452220) (accessed 2026-04-18).
- [University of Exeter — "Cooling, a blind spot in UK energy policy" (2022)](https://blogs.exeter.ac.uk/energy/2022/03/25/cooling-a-blind-spot-in-uk-energy-policy/) (accessed 2026-04-18).

**Implication.** The Stage 2 notebook's demo moment — "V-shape visible" — is
**historically weaker than a true V**: on 2018-2024 GB data the cold arm will
be pronounced, the hot arm shallow and noisy. The demo will land more honestly
as "anti-correlation below ~17 °C with a flat-to-slightly-rising arm above",
not as a symmetric V. CIBSE degree-day thresholds (15.5/22) are cite-able
reference points.

---

## Summary of disagreements surfaced

1. **DESIGN.md §4.2** says "~10 km via UKMO UKV 2 km model" — **incorrect** for
   `archive-api.open-meteo.com`. Archive is ERA5/ERA5-Land/CERRA-backed
   (~11 km). UKV 2 km is only available via `historical-forecast-api` and only
   from 2022-03-01.
2. **Stage 2 intent** asserts population weighting as defensible for electricity
   demand. The industry-adopted GB precedent (CWV) uses **demand weighting**
   across distribution zones, not population. Thornton et al. use CET unweighted.
   Not wrong, but not "standard practice".
3. **Stage 2 intent** invokes a "V-shape" as the demo moment. On historical GB
   data the curve is a hockey-stick (strong cold arm, flat warm arm), not a
   symmetric V. The V is only beginning to emerge post-2020 with rising
   residential cooling.

---

## Out of scope for this note

- Recommending which data model to pin (`era5_seamless` vs `era5` vs `cerra`).
- Recommending a caching granularity.
- Recommending adoption/rejection of `openmeteo-requests`.
- Writing any production code.

Per the researcher role contract, those decisions belong to the
implementer/lead in the Stage 2 work session, with this document as input.
