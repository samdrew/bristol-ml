# Stage 2 — Weather ingestion + national aggregation

## Goal

Bring hourly historical weather into the project, compose ten UK population centres into a single national signal, and land the first analytical plot — the temperature-vs-demand curve. Second concrete ingester: the point at which the layer architecture's "two-caller" trigger for shared helpers is finally met, and the point at which the features layer earns its first member.

## What was built

- `src/bristol_ml/ingestion/_common.py` — shared ingestion helpers, lifted from `neso.py` in a zero-behaviour-change refactor. Public-for-the-layer: `CachePolicy` (`AUTO | REFRESH | OFFLINE`) and `CacheMissingError`. Module-private helpers: `_atomic_write(table, path)`, `_cache_path(config)`, `_respect_rate_limit(last, gap)`, `_retrying_get(client, url, params, config)`, `_RetryableStatusError`. Generic across sources via three `@runtime_checkable` structural `Protocol`s — `RetryConfig`, `RateLimitConfig`, `CachePathConfig` — so `NesoIngestionConfig` and `WeatherIngestionConfig` both satisfy the helper contract without a shared base class.
- `src/bristol_ml/ingestion/neso.py` — refactored to import the six helpers from `_common`; public surface unchanged; Stage 1's 35 tests continue to pass unmodified.
- `src/bristol_ml/ingestion/weather.py` — the module. Public surface: `fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`, plus lazy re-exports of `CachePolicy` and `CacheMissingError` for notebook ergonomics. Module CLI `python -m bristol_ml.ingestion.weather [--cache auto|refresh|offline] [overrides ...]`. Long-form parquet output at `<cache_dir>/weather.parquet`; schema documented in `src/bristol_ml/ingestion/CLAUDE.md`.
- `src/bristol_ml/features/__init__.py` and `src/bristol_ml/features/weather.py` — the features layer is introduced one stage earlier than DESIGN §9 implies, because Stage 2's notebook needs a weighted-mean helper and notebooks are thin (§2.1.8). Lazy re-export of `national_aggregate` from the submodule keeps `python -m bristol_ml` cheap. Module CLI `python -m bristol_ml.features.weather [--head N] [overrides ...]` fetches + loads + aggregates + prints a head of the wide-form national frame.
- `src/bristol_ml/features/weather.py::national_aggregate(df, weights)` — NaN-safe, renormalising, population-weighted mean. One row per `timestamp_utc` in the returned wide-form frame; one column per weather variable. Subset semantics documented on the function docstring (see "Design choices" below).
- `conf/_schemas.py` — new types `WeatherStation` and `WeatherIngestionConfig`; `IngestionGroup` gains an optional `weather: WeatherIngestionConfig | None` sibling of `neso`. `WeatherIngestionConfig` carries a `model_validator` that rejects `end_date < start_date` and duplicate / empty station lists.
- `conf/ingestion/weather.yaml` — Hydra group file for ten UK population centres (London, Birmingham, Manchester, Leeds, Newcastle, Sheffield, Bristol, Liverpool, Glasgow, Edinburgh) with ONS 2011 BUA population weights; five default variables (`temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`); `start_date: 2018-01-01`; `end_date` omitted (defaults to today at fetch time); `min_inter_request_seconds: 0.25`; `timezone: UTC`.
- `src/bristol_ml/ingestion/CLAUDE.md` — extended with a second `weather.py` output-schema table alongside the existing `neso.py` table, plus a "Shared helpers — `_common.py`" section documenting the `Protocol`-based generality, a "Data model caveat" on the DESIGN §4.2 drift, and a "Cache staleness" note. Open-Meteo CC BY 4.0 acknowledgement added to the licence section.
- `src/bristol_ml/features/CLAUDE.md` — new module guide, scoped to Stage 2's single function, with signposts for the Stage 3 assembler that will join demand + weather + calendar.
- `notebooks/02_weather_demand.ipynb` — thin demo. Primes the NESO cache and the weather cache, composes the national aggregate via `national_aggregate`, joins onto hourly demand, and plots temperature-vs-demand with a LOWESS overlay. Prose commentary frames the curve as a **hockey-stick** rather than a symmetric V (research §10; Thornton et al. 2016; Exeter 2022), motivates Open-Meteo over Met Office DataHub (48-hour DataHub free-tier window; AC5), and marks the CIBSE HDD 15.5 °C / CDD 22 °C reference lines.
- `scripts/record_weather_cassette.py` — one-off recorder mirroring `record_neso_cassette.py`; outside the `bristol_ml` package so it is excluded from the wheel.
- `tests/fixtures/weather/cassettes/weather_2023_01.yaml` (~64 kB, London + Bristol × January 2023 × five variables) plus `tests/fixtures/weather/toy_stations.csv` for aggregator unit tests.
- 58 new tests (93 total across the repo, all green):
  - 12 implementer-derived weather-helper tests covering `_fetch_station`, `_parse_station_payload`, `_assert_schema`, `_to_arrow`, `_resolve_end_date`, and the CLI wrapper.
  - 10 implementer-derived aggregate-helper tests exercising the NaN-safe mask, renormalisation, and the `_NON_VARIABLE_COLUMNS` guard.
  - 10 spec-derived unit tests covering the intent's AC1-AC6 on the ingester surface.
  - 15 spec-derived aggregate tests covering the documented subset semantics, identity-under-equal-weights, and invariants from intent §Points-for-consideration.
  - 8 integration tests via cassette replay (`REFRESH` writes parquet, `AUTO` reuses, `OFFLINE` raises when cache is absent, byte-equality modulo `retrieved_at_utc` across successive runs).
  - 3 smoke tests (`--help` for both CLIs; module import under `python -m bristol_ml`).
- `CHANGELOG.md` — `### Added` bullets under `[Unreleased]` for `bristol_ml.ingestion.weather`, `bristol_ml.ingestion._common`, `bristol_ml.features.national_aggregate`, the Hydra group, both CLIs, the notebook, the cassette, and the `statsmodels` runtime dep. `### Changed` bullet for the `neso.py` refactor.
- `README.md` — Stage 2 entry-point paragraph referencing `notebooks/02_weather_demand.ipynb`, the Open-Meteo CC BY 4.0 attribution, the population-weighted aggregate, and the hockey-stick framing.

## Design choices made here

- **`_common.py` extraction via structural `Protocol`s, not a shared base class.** The helpers were lifted in a zero-behaviour-change refactor (commit `a5aba89`). The retry / rate-limit / cache-path knob surfaces are declared as three `@runtime_checkable` `Protocol` types (`RetryConfig`, `RateLimitConfig`, `CachePathConfig`). Both `NesoIngestionConfig` and `WeatherIngestionConfig` satisfy them structurally — no inheritance coupling, no `Union` type in the helper signatures, and a future fourth ingester gets the same treatment without touching `_common.py`. `isinstance(cfg, RetryConfig)` works at runtime thanks to `@runtime_checkable` — used in one of the Stage 2 unit tests as a smoke check that the two configs stay compatible.
- **`national_aggregate(df, weights: Mapping[str, float])` — deliberate drift from LLD §2's config-based sketch.** The LLD proposed `national_aggregate(df, config, *, stations=None, variables=None)`, reading weights off `WeatherIngestionConfig.stations`. The implementation takes a plain `Mapping[str, float]` of station-name to weight. Rationale: the features layer must not import from `conf._schemas` — ingestion-layer config shapes are an ingestion concern, and wiring them into features creates the same coupling the Stage 1 `_common.py` split was meant to avoid. Callers (notebook, Stage 3 assembler, CLI) extract `{s.name: s.weight for s in cfg.stations}` at the boundary. Documented as a deviation on this retrospective; the LLD text stays as historical record per the tier conventions.
- **Subset-weight semantics — loud on stations-in-weights-missing-from-frame, silent on stations-in-frame-missing-from-weights.** Two failure modes look similar and carry opposite intents:
  - A caller naming a station that the frame lacks is asking for a weight on a signal that does not exist — the returned weighted mean would silently differ from what the caller intended. `national_aggregate` raises `ValueError` naming the missing stations.
  - A frame containing a station the caller has not weighted is the expected shape of the acceptance-criterion-3 subset case (demo runs with fewer stations). `national_aggregate` excludes those rows and proceeds.
  NaN at an individual station drops that station from that `(hour, variable)` group; remaining weights renormalise to sum-to-one on the intersection. All-NaN groups propagate NaN rather than producing 0/0. Documented on the function docstring.
- **`_assert_schema(df, station: str, requested_variables)` — `station` is a plain `str`, with a `getattr(station, "name", station)` fallback.** The signature parallels Stage 1's `(df, year: int)` — assertion helpers take a scalar identifier, not a typed domain object. The fallback accepts a `WeatherStation` passed from implementer-derived unit tests so those tests do not have to know about the flattening. Cosmetic wart; a Stage 3 clean-up could remove the fallback once the last internal caller is migrated.
- **Long-form parquet, not wide-per-station.** One row per `(timestamp_utc, station)`, one column per weather variable. Scales to N stations without schema changes. Adding a station re-fetches everything on the next `REFRESH` (no partial updates; see "Cache staleness" in `src/bristol_ml/ingestion/CLAUDE.md`) and appends rows on the `station` axis rather than adding columns.
- **Serial httpx across stations with a shared pooled client.** Ten stations × one window is ~5-15 seconds cold; well inside Open-Meteo's 600-per-minute free-tier cap. A shared `httpx.Client` reuses TCP connections across the loop. `httpx.AsyncClient` is a defensible upgrade once a live demo feels the latency; deferred to when measured.
- **`min_inter_request_seconds: 0.25` default** — conservative pacing only; Open-Meteo's rate limit is never the binding constraint at Stage 2 scale. Mirrors the Stage 1 ergonomic of an explicit knob rather than an implicit "as fast as possible".
- **Ten stations, ONS 2011 BUAs.** London, Birmingham, Manchester, Leeds, Newcastle, Sheffield, Bristol, Liverpool, Glasgow, Edinburgh. Deviation from the LLD §3.2 list (which named Belfast and Cardiff in place of Sheffield and Liverpool) — the shipped set weights the English conurbations more heavily, which matches GB demand (Northern Ireland is not on the GB grid; Cardiff's BUA is dwarfed by Sheffield). Not analytically important at this weighting precision, but noted for future harmonisation with the 2021 Census refresh.
- **Cassette scope deliberately tight.** Two stations (London + Bristol — ~15× weight ratio) × one month × five variables = ~64 kB. Keeps CI fast and the repo slim; a replay covers the full parquet schema, the atomic-write path, and the `REFRESH`/`AUTO`/`OFFLINE` semantics. Wider coverage is a developer action via `record_weather_cassette.py`, not a test dependency.
- **Notebook reframes the intent's "V-shape" as a hockey-stick.** Research §10 documents the GB historical curve as strong cold arm, flat-to-noisy warm arm. The notebook's prose cites Thornton et al. (2016) on CET-based GB demand sensitivity and the University of Exeter's 2022 summer-peak analysis, and marks the CIBSE HDD 15.5 °C / CDD 22 °C reference lines. The intent's §Demo-moment text stays immutable per the tier policy; the framing drift is surfaced to the lead (see "Deferred / known drift" below).
- **Features layer lands at Stage 2.** DESIGN §9 implies Stage 3, but Stage 2 needs a weighted-mean helper and §2.1.8 forbids reimplementing it in the notebook. `bristol_ml.features` ships with one function; Stage 3's assembler becomes the second. The proper architecture doc for the features layer (`docs/architecture/layers/features.md`) lands at Stage 3 when there is more than one function to describe.

## Demo moment

From a clean clone (Stages 0 and 1 already built):

```
uv sync --group dev
uv run pytest                                                               # 93 passed
uv run python -m bristol_ml.ingestion.weather --help                        # CLI help
uv run python -m bristol_ml.ingestion.weather --cache refresh               # first fetch (~5-15 s)
uv run python -m bristol_ml.ingestion.weather --cache offline               # subsequent runs
uv run python -m bristol_ml.features.weather --head 10                      # aggregate preview
uv run jupyter nbconvert --to notebook --execute notebooks/02_weather_demand.ipynb
```

The notebook renders the hockey-stick temperature-vs-demand scatter with a LOWESS overlay — the first analytical plot in the repo. Cold-arm anti-correlation is unambiguous; the warm arm is flat pre-2020 and faintly upward-sloping post-2020.

## Deferred

- **REMIT-style bi-temporal storage.** The single-timestamp convention used here is insufficient for REMIT events (`published_at`, `effective_from`, `effective_to`). Decision deferred to Stage 13 — the layer architecture flags it as the trigger to revisit the storage section.
- **Discovery mode for station coordinates.** Today the ten station lat/lons are inline in `conf/ingestion/weather.yaml`. A discovery mode (e.g. GeoNames lookup) is plausible but unearned.
- **2021 ONS BUA refresh.** The 2011 figures ship here because the 2022 Scotland / 2021 Northern Ireland / 2021 England & Wales data are not yet published as a harmonised urban-area table (as of 2026-04-18). Mechanical refresh when that publication lands.
- **`cerra` / `era5_land` model switch.** The `WeatherIngestionConfig` schema does not yet expose a `model` field — `era5_seamless` is implicitly used (the API's archive default). A validator that rejects `model=cerra` with `end_date > 2021-06-30` is deferred to when a caller actually wants to pin a model.
- **Wiring the dead `columns` field on `NesoIngestionConfig`.** Carried forward unaddressed from Stage 1 — the field is present on the Pydantic model but not consumed anywhere. Pure clean-up when someone touches Stage 1 again.
- **Commercial `customer-api.open-meteo.com` tier support.** The `tenacity` retry predicate and the `httpx` client already route any `base_url`; only the `&apikey=` parameter is missing. Deferred until someone hits the free-tier cap, which is well beyond Stage 2's volume.
- **Concurrency (`httpx.AsyncClient`).** See "Design choices" above.
- **Full features-layer architecture doc.** Written at Stage 3, when there is more than one function to describe.
- **Known drift, surfaced for the lead (do not fix from within Stage 2):**
  - DESIGN §4.2 still claims "UKV 2 km" for Open-Meteo's archive; correct shape is ERA5 / ERA5-Land / CERRA at ~9-11 km. Deny-tier for the lead; human main-session edit.
  - The intent's "V-shape" framing is historically imprecise for GB demand; a hockey-stick is correct. Intent is immutable; the notebook reframes.
  - DESIGN §6 layout tree missing: Stage 1's ingestion additions (`ingestion/neso.py`, `conf/ingestion/neso.yaml`) and Stage 2's additions (`ingestion/weather.py`, `ingestion/_common.py`, `features/`, `conf/ingestion/weather.yaml`). Deny-tier for the lead; batched human edit.
  - LLD-to-implementation drifts for Stage 2: `national_aggregate` signature (LLD used `config`; shipped takes `weights: Mapping[str, float]`) and station list (LLD had Belfast / Cardiff; shipped has Sheffield / Liverpool). LLD left as historical record per tier convention; retrospective is the source of truth.
  - Stage 1 LLD `neso.md` minor drifts (§§ 1-5 / 7-13) also still unaddressed.

## Next

→ Stage 3 — Feature assembler + train/test split. First consumer of the features layer this stage introduced; `features.assembler.build` will call `national_aggregate` (or a thin wrapper) when composing the canonical hourly frame.
