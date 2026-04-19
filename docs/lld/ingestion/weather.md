# LLD — Stage 2, weather ingestion + national aggregate

- **Status:** First-pass design, pre-implementation — 2026-04-18.
- **Scope:** `src/bristol_ml/ingestion/weather.py`, `src/bristol_ml/ingestion/_common.py` (extracted from Stage 1 this stage), `src/bristol_ml/features/__init__.py`, `src/bristol_ml/features/weather.py`, `conf/ingestion/weather.yaml`, `tests/unit/ingestion/test_weather.py`, `tests/unit/features/test_weather_aggregate.py`, `tests/integration/ingestion/test_weather_cassettes.py`, `tests/fixtures/weather/`.
- **Authoritative inputs:** [`docs/intent/02-weather-ingestion.md`](../../intent/02-weather-ingestion.md) (intent); [`docs/architecture/layers/ingestion.md`](../../architecture/layers/ingestion.md) (layer architecture); [`docs/lld/research/02-weather-ingestion.md`](../research/02-weather-ingestion.md) (facts).
- **Spec-drift rule:** where this LLD and the stage intent disagree, the intent wins; surface the drift, do not silently rewrite.

## 0. Spec drift flagged before implementation

Three drifts to surface to the human before this stage begins. None is a blocker; each is a decision the implementer should not silently resolve.

1. **DESIGN §4.2 data model.** The spec says "~10 km via UKMO UKV 2 km model" for Open-Meteo's archive. Research §4 shows the archive is ERA5 / ERA5-Land / CERRA reanalyses at ~9–11 km, **not** UKV. UKV 2 km is only reachable via `historical-forecast-api` and only from 2022-03-01 (incompatible with our 2018-onwards training window). Corrective edit belongs in DESIGN §4.2, main session, human approval.
2. **"V-shape" demo framing.** Stage 2 intent §"Demo moment" asserts a non-linear V-shape visible in one year of GB data. Research §10 shows the historical curve is a hockey-stick (strong cold arm, flat-to-noisy warm arm). Notebook narrative should frame the demo as "anti-correlation below ~17 °C with a flat warm arm, with the hot arm emerging post-2020" — not a symmetric V.
3. **Population-weighted national temperature.** Stage 2 intent §"Points for consideration" offers population-weighting as defensible. Research §9 shows the GB industry precedent (National Gas CWV) uses gas-demand weighting across 13 LDZs, not population; Thornton et al. (2016) use CET with no weighting. Population-weighting is a defensible pedagogical default; the notebook should acknowledge it is not an industry-verified standard for electricity.

The LLD below proceeds on the research-corrected data model and on the pedagogical framing. The intent file stays immutable.

## 1. Module layout

```
src/bristol_ml/ingestion/
├── CLAUDE.md                   # updated: adds weather.py schema table
├── _common.py                  # NEW: shared helpers extracted from neso.py
├── neso.py                     # existing; refactored to import from _common
└── weather.py                  # NEW

src/bristol_ml/features/
├── __init__.py                 # NEW — features module introduced at Stage 2
└── weather.py                  # NEW — population-weighted national aggregate

conf/ingestion/
├── neso.yaml                   # existing
└── weather.yaml                # NEW

tests/
├── unit/ingestion/
│   ├── test_neso.py            # existing
│   └── test_weather.py         # NEW
├── unit/features/
│   ├── __init__.py             # NEW
│   └── test_weather_aggregate.py   # NEW
├── integration/ingestion/
│   ├── test_neso_cassettes.py  # existing
│   └── test_weather_cassettes.py   # NEW
└── fixtures/weather/
    ├── cassettes/              # pytest-recording YAML; narrow slice
    └── station_subset.csv      # hand-crafted rows for aggregator tests
```

The extraction of `_common.py` is driven by the ingestion-layer arch's "Open questions" — the two-caller trigger is now met. What moves:

- `_atomic_write(table, path)` — unchanged.
- `_retrying_get(client, url, params, config)` — parameterised on the config's retry knobs, not on a specific source's schema.
- `_RetryableStatusError` — internal signal class.
- `_respect_rate_limit(last, gap)` — unchanged.
- `pytest-recording` fixtures filter-list — centralised so future feeds do not re-derive the auth-header scrubber.

Not moved (stay in `neso.py`): `_to_utc`, `_parse_settlement_date`, `_autumn_fallback_dates`, `_spring_forward_dates`, `_assert_schema`. These are NESO-specific. If a future settlement-period-carrying source emerges (it will not — Stage 17 prices are half-hourly but dated differently), the extraction is a follow-up.

## 2. Public interface

```python
# src/bristol_ml/ingestion/weather.py
from pathlib import Path
import pandas as pd
from bristol_ml.ingestion._common import CachePolicy, CacheMissingError  # re-exported
from conf._schemas import WeatherIngestionConfig

def fetch(
    config: WeatherIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch hourly historical weather for all configured stations; return cache path."""

def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet, assert the schema, return a long-form tz-aware frame."""
```

```python
# src/bristol_ml/features/weather.py
import pandas as pd
from conf._schemas import WeatherIngestionConfig  # weights live in ingestion config

def national_aggregate(
    df: pd.DataFrame,
    config: WeatherIngestionConfig,
    *,
    stations: list[str] | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Collapse per-station hourly weather to a national weighted signal.

    Weights are read from `config.stations[*].weight`, restricted to the subset
    in `stations` (or all configured stations if None). Returns a wide-form
    frame with one row per UTC hour and one column per requested variable.
    NaNs at individual stations are dropped before re-normalising the weights.
    """
```

Both callables are re-exported: `from bristol_ml.ingestion import weather`, `from bristol_ml.features import weather`. Module CLIs — `python -m bristol_ml.ingestion.weather` and `python -m bristol_ml.features.weather` (the latter prints the national aggregate for the default station list, reading from the cached weather parquet).

`CachePolicy` reuses Stage 1's `StrEnum` values (`AUTO | REFRESH | OFFLINE`) without redefinition — the shared enum lives in `_common.py` after the extraction.

## 3. Config shape

### 3.1 Pydantic schema (`conf/_schemas.py`)

```python
class WeatherStation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    latitude: float = Field(ge=49.0, le=61.0)    # UK envelope
    longitude: float = Field(ge=-8.5, le=2.5)
    weight: float = Field(gt=0)                   # population (2011 urban-area)
    weight_source: str                            # e.g. "ONS 2011 Census, BUA"

class WeatherIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: HttpUrl = HttpUrl("https://archive-api.open-meteo.com/v1/archive")
    model: Literal["era5_seamless", "era5", "era5_land", "cerra"] = "era5_seamless"
    stations: list[WeatherStation]                # ten UK population centres
    variables: list[str] = [
        "temperature_2m", "dew_point_2m",
        "wind_speed_10m", "cloud_cover", "shortwave_radiation",
    ]
    start_date: date                              # e.g. 2018-01-01
    end_date: date | None = None                  # None → today
    cache_dir: Path
    cache_filename: str = "weather_hourly.parquet"
    request_timeout_seconds: float = 30.0
    max_attempts: int = 3
    backoff_base_seconds: float = 1.0
    backoff_cap_seconds: float = 10.0
    min_inter_request_seconds: float = 0.0        # Open-Meteo's 600/min is not a constraint
```

`IngestionGroup` (added in Stage 1) gains an optional `weather: WeatherIngestionConfig` sibling of `neso`.

### 3.2 YAML (`conf/ingestion/weather.yaml`)

```yaml
# @package ingestion.weather
base_url: https://archive-api.open-meteo.com/v1/archive
model: era5_seamless
cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,${hydra:runtime.cwd}/data/raw/weather}
start_date: 2018-01-01
# end_date omitted: defaults to today at fetch time
variables: [temperature_2m, dew_point_2m, wind_speed_10m, cloud_cover, shortwave_radiation]
stations:
  - { name: london,      latitude: 51.5074, longitude: -0.1278, weight: 9787426, weight_source: "ONS 2011 Census, Greater London BUA" }
  - { name: birmingham,  latitude: 52.4862, longitude: -1.8904, weight: 2440986, weight_source: "ONS 2011 Census, West Midlands BUA" }
  - { name: manchester,  latitude: 53.4808, longitude: -2.2426, weight: 2553379, weight_source: "ONS 2011 Census, Greater Manchester BUA" }
  - { name: leeds,       latitude: 53.8008, longitude: -1.5491, weight: 1777934, weight_source: "ONS 2011 Census, West Yorkshire BUA" }
  - { name: glasgow,     latitude: 55.8642, longitude: -4.2518, weight:  957620, weight_source: "ONS 2011 Census, Greater Glasgow BUA" }
  - { name: newcastle,   latitude: 54.9784, longitude: -1.6174, weight:  774891, weight_source: "ONS 2011 Census, Tyneside BUA" }
  - { name: bristol,     latitude: 51.4545, longitude: -2.5879, weight:  617280, weight_source: "ONS 2011 Census, Bristol BUA" }
  - { name: belfast,     latitude: 54.5973, longitude: -5.9301, weight:  595879, weight_source: "ONS 2011 Census, Belfast BUA" }
  - { name: edinburgh,   latitude: 55.9533, longitude: -3.1883, weight:  482270, weight_source: "ONS 2011 Census, Edinburgh BUA" }
  - { name: cardiff,     latitude: 51.4816, longitude: -3.1791, weight:  447287, weight_source: "ONS 2011 Census, Cardiff BUA" }
```

Top `conf/config.yaml` gains `ingestion/weather@ingestion.weather` in its defaults list, sibling to the existing NESO default.

Rationale for ten stations: research §5 shows the ERA5-Land cell size (~11 km) makes ±0.05° of coordinate jitter analytically irrelevant, so "city-hall lat/lon to three decimals" is sufficient precision. Weights are inlined with their provenance so a future refresh to 2021/2022 Census figures is mechanical.

## 4. Output parquet schema

Stored at `<cache_dir>/<cache_filename>` — default `data/raw/weather/weather_hourly.parquet`. **Long-form** (one row per station × hour); the national aggregate is computed downstream by `features/weather.py`, never persisted by the ingester.

| Column               | Parquet type              | Unit / notes                                            |
|----------------------|---------------------------|---------------------------------------------------------|
| `timestamp_utc`      | `timestamp[us, tz=UTC]`   | Canonical. Open-Meteo returns UTC natively; no DST alg. |
| `station`            | `string` (dictionary)     | Lowercase snake-case name, matches `WeatherStation.name`. |
| `temperature_2m`     | `float32`                 | °C. Open-Meteo reports to 0.1 °C; float32 is ample.     |
| `dew_point_2m`       | `float32`                 | °C.                                                     |
| `wind_speed_10m`     | `float32`                 | km/h (API default).                                     |
| `cloud_cover`        | `int8`                    | %, 0–100.                                               |
| `shortwave_radiation`| `float32`                 | W/m².                                                   |
| `retrieved_at_utc`   | `timestamp[us, tz=UTC]`   | Per-fetch provenance (§2.1.6).                          |
| `source_model`       | `string` (dictionary)     | `era5_seamless` by default; records which reanalysis was used. |

Primary key: `(timestamp_utc, station)` unique. Sorted by `timestamp_utc ASC, station ASC`.

Schema is documented in `src/bristol_ml/ingestion/CLAUDE.md` alongside `neso.py`'s table.

## 5. Data flow

```
  ┌───────────────────────────────────────────────────────────────┐
  │ fetch(config, cache)                                          │
  │                                                               │
  │   1. resolve cache_path                                       │
  │   2. if cache == OFFLINE and not exists: raise                │
  │   3. if cache == AUTO and exists: return cache_path           │
  │   4. for each station in config.stations:                     │
  │        payload = _fetch_station(station, config, client)      │
  │        raw_df = _parse_station_payload(payload, station)      │
  │        _assert_schema(raw_df, station, config.variables)      │
  │        frames.append(raw_df)                                  │
  │   5. combined = concat(frames).sort(timestamp_utc, station)   │
  │   6. table = _to_arrow(combined, config.model, retrieved_at)  │
  │   7. atomic_write(table, cache_path)                          │
  │   8. return cache_path                                        │
  └───────────────────────────────────────────────────────────────┘
```

A single shared `httpx.Client` across all stations (connection reuse). Requests are serial; concurrency is deferred (layer arch "Open questions"). The rate-limit helper is called with `min_inter_request_seconds=0` by default — Open-Meteo's 600/min is not a binding constraint.

Internal helpers:

- `_fetch_station(station, config, client) -> dict` — one GET with all variables and full date range; returns parsed JSON.
- `_parse_station_payload(payload, station) -> pd.DataFrame` — zips `hourly.time` with each variable array; returns a long-form frame.
- `_assert_schema(df, station, requested_vars) -> pd.DataFrame` — required columns = `time` + requested variables; unknown → warn-and-drop; missing → `KeyError` naming the variable.
- `_to_arrow(df, model, retrieved_at) -> pa.Table` — adds `source_model` and `retrieved_at_utc`, casts to `OUTPUT_SCHEMA`.

`load(path)` iterates the documented `OUTPUT_SCHEMA` exactly as Stage 1 does.

## 6. National aggregate (`features/weather.py`)

The function body applies the pandas weighted-mean idiom from research §9 — there is no pandas built-in (`pandas#10030`, 2015).

```python
def national_aggregate(df, config, *, stations=None, variables=None):
    subset = df if stations is None else df[df["station"].isin(stations)]
    weights = {s.name: s.weight for s in config.stations
               if stations is None or s.name in stations}
    if not weights:
        raise ValueError("No stations selected for aggregation.")
    vars_out = variables or [v for v in subset.columns
                             if v not in {"timestamp_utc", "station",
                                          "retrieved_at_utc", "source_model"}]
    long = subset.melt(
        id_vars=["timestamp_utc", "station"],
        value_vars=vars_out, var_name="variable", value_name="value",
    )
    long["weight"] = long["station"].map(weights)
    # NaN-safe weighted mean: drop NaN values, renormalise per (hour, variable) group.
    long = long.dropna(subset=["value"])
    agg = (
        long.groupby(["timestamp_utc", "variable"], observed=True)
        .apply(lambda g: (g["value"] * g["weight"]).sum() / g["weight"].sum())
        .rename("value").reset_index()
    )
    return agg.pivot(index="timestamp_utc", columns="variable", values="value")
```

Stage 2's notebook uses `national_aggregate` to produce the scatter plot. Stage 3's `features/assembler.py` will call the same function (or a thin wrapper) when composing the canonical feature table.

Intent acceptance criterion 3 — "accepts any subset of the configured station list" — is exercised by a unit test that passes `stations=["london", "manchester"]`. Acceptance criterion 6 — "equal weights on identical inputs yield the identity" — is the aggregator's smoke test.

## 7. Cache semantics

Identical to Stage 1's `CachePolicy`:

- **AUTO**: return cache if present, fetch all stations if not.
- **REFRESH**: fetch all stations, overwrite atomically.
- **OFFLINE**: return cache if present, raise `CacheMissingError` if not.

No per-station-partial-refresh. A station added to `conf/ingestion/weather.yaml` triggers a full re-fetch on the next `AUTO` if the cache predates the config change — but Stage 2 does not implement staleness detection, so the user must re-run with `REFRESH` when they change the station list. Documented loudly in `ingestion/CLAUDE.md`; staleness logic is a Stage 19 (orchestration) concern.

## 8. Retry policy

Reuses `_common.py`'s `_retrying_get` verbatim. Open-Meteo's failure modes are mostly `ConnectError` (intermittent DNS) and `ReadTimeout` (the 5-year queries can take ~2 seconds). Retry on those, on 5xx, and on 429; fail loudly on 4xx-non-429. Three attempts, exponential backoff base 1s cap 10s — same defaults as NESO.

No documented Open-Meteo error-response JSON shape; treat `{"error": true, "reason": "..."}` (observed in practice) as a 4xx failure, not retried.

## 9. Fixture strategy

- **Library:** `pytest-recording` + vcrpy, already a dev dep from Stage 1.
- **Cassette scope:** two stations (London + Bristol — geographically distinct, weights span 15×) × one month (Jan 2023) × five variables. Keeps the cassette under ~50 kB.
- **Filter:** no auth today; the header-scrub list lives in `_common.py` so future feeds inherit it.
- **Hand-crafted fixture:** `tests/fixtures/weather/station_subset.csv` — four stations × one UTC hour with hand-picked values. Used by unit tests for `national_aggregate` so test assertions can be arithmetic identities.

## 10. Tests

| Test                                                             | Level       | Asserts                                                                        |
|------------------------------------------------------------------|-------------|--------------------------------------------------------------------------------|
| `test_parse_station_payload_shape`                               | Unit        | hourly.time zipped with variable arrays produces expected long-form columns.   |
| `test_assert_schema_missing_variable_raises`                     | Unit        | Requested variable missing → `KeyError` naming the variable and station.       |
| `test_assert_schema_unknown_variable_warns_and_drops`            | Unit        | Unknown variable → warning; not persisted.                                     |
| `test_output_schema_types_match`                                 | Unit        | `_to_arrow` produces a table whose schema == the documented `OUTPUT_SCHEMA`.   |
| `test_fetch_offline_raises_when_cache_missing`                   | Unit        | `OFFLINE` + no cache → `CacheMissingError`.                                    |
| `test_fetch_auto_returns_cached_path_no_network`                 | Unit        | `AUTO` + cache present performs zero HTTP calls.                               |
| `test_fetch_refresh_end_to_end`                                  | Integration | `REFRESH` + cassette → parquet written; schema matches §4; rows sorted.        |
| `test_fetch_idempotent`                                          | Integration | Two `REFRESH` runs produce identical rows (modulo `retrieved_at_utc`).         |
| `test_aggregate_equal_weights_identity`                          | Unit        | Equal weights on identical station values yield the identity (intent AC 6).    |
| `test_aggregate_station_subset`                                  | Unit        | Passing `stations=[...]` restricts + renormalises (intent AC 3).               |
| `test_aggregate_nan_handling`                                    | Unit        | A NaN at one station drops that station from that hour's weighting.            |
| `test_aggregate_missing_station_raises`                          | Unit        | `stations=["mars"]` raises with a clear message.                               |
| `test_cli_help_weather`                                          | Smoke       | `python -m bristol_ml.ingestion.weather --help` exits 0.                       |

Acceptance-criteria trace:

| Intent criterion                                                                   | Covered by                                          |
|-----------------------------------------------------------------------------------|-----------------------------------------------------|
| 1. Cache present → completes offline                                              | `test_fetch_auto_returns_cached_path_no_network`    |
| 2. No cache → fetches all configured stations                                     | `test_fetch_refresh_end_to_end`                     |
| 3. Aggregation accepts subsets of station list                                    | `test_aggregate_station_subset`                     |
| 4. Notebook runs top-to-bottom quickly                                            | Manual; see §12.                                    |
| 5. Notebook commentary motivates Open-Meteo over Met Office DataHub briefly       | Manual; covered in notebook prose.                  |
| 6. Smoke test for fetcher; aggregation identity under equal weights               | `test_fetch_refresh_end_to_end`, `test_aggregate_equal_weights_identity` |

## 11. `_common.py` extraction plan

The extraction happens **before** `weather.py` is written, so Stage 2's ingester imports from `_common` from the first commit rather than being refactored mid-stage.

Step 1 (one PR or one commit): move the following from `neso.py` to `_common.py`, with `neso.py` importing them:
- `CachePolicy`, `CacheMissingError`.
- `_atomic_write`.
- `_retrying_get`, `_RetryableStatusError`.
- `_respect_rate_limit`.

Step 2 (same branch): generalise the retry signature to read `max_attempts`, `backoff_base_seconds`, `backoff_cap_seconds`, `request_timeout_seconds` from the config — already true; no generalisation needed, only re-parameterisation.

Step 3 (same branch): add `weather.py` against the new `_common.py`.

Step 4: regression-test `neso.py`. Existing tests should all pass without modification (the extraction is a pure refactor). If any test references a now-moved symbol, update the import.

The extraction does not touch the public interface of `neso` or add a public `_common` surface — `_` prefix convention continues, consumers go through the ingester's own module.

## 12. Notebook

`notebooks/02_weather_joined_plot.ipynb` — thin (§2.1.8):

1. `from bristol_ml import load_config; from bristol_ml.ingestion import weather, neso; from bristol_ml.features import weather as feat_weather`
2. `cfg = load_config(overrides=["+ingestion=[neso,weather]"])`
3. `demand_path = neso.fetch(cfg.ingestion.neso); weather_path = weather.fetch(cfg.ingestion.weather)`
4. `demand = neso.load(demand_path); stations = weather.load(weather_path)`
5. `national = feat_weather.national_aggregate(stations, cfg.ingestion.weather)`
6. Hour-align demand to UTC, resample to hourly from half-hourly (inline — Stage 3 re-homes this to `features/assembler.py`).
7. Scatter `national["temperature_2m"]` against hourly `nd_mw` — one year, point size modulated by hour-of-day.
8. Overlay a smoothed fit (LOWESS via `statsmodels`) — the curve to interpret.
9. Prose commentary:
   - One paragraph on the hockey-stick shape vs the "V-shape" the intent invokes — research §10 framing.
   - One paragraph on the weighting choice — pedagogical default, acknowledging CWV precedent from research §9.
   - Two-line note on Open-Meteo's archive data model (ERA5-seamless, ~11 km) — intent AC 5.
   - One line on the Met Office DataHub rejection (48-hour free-tier window) — intent AC 5.

Target runtime: under 45 seconds when the weather cache is warm and the NESO cache is warm.

## 13. Risks, open questions, deferred items

- **DESIGN §4.2 drift** — see §0. Must surface before implementation; a corrective edit belongs in the same PR as the Stage 2 merge (main session).
- **V-shape vs hockey-stick** — see §0. Addressed by notebook commentary, not by code changes.
- **Industry weighting vs population weighting** — see §0. Config records the weight source; notebook acknowledges the CWV precedent.
- **CERRA cut-off.** If a facilitator switches `model` to `cerra` (5 km Europe), end_date after 2021-06 will silently return partial data. Config validation should `ValueError` on `model=cerra` with `end_date` > 2021-06-30. Implementer decision — the validator is one `@model_validator` on `WeatherIngestionConfig`.
- **`openmeteo-requests` SDK not adopted.** Research §8 offers two paths. Sticking with `httpx` + `tenacity` matches Stage 1 and avoids a second HTTP stack (`niquests` vs `httpx`). Trade-off documented; swap is an LLD-level decision, not an architecture-level one.
- **Subset-of-stations-cached**. A partially-fetched cache (say, a prior REFRESH failed after station 7) leaves the old cache intact (atomic write) — the next AUTO re-returns the stale cache. Acceptable: a REFRESH failure is loud, the human re-runs. Documented in `ingestion/CLAUDE.md`.
- **Aggregator location**. `features/weather.py` introduces `features/` at Stage 2 rather than Stage 3. The single function here is the whole module; Stage 3's `features/assembler.py` joins weather + demand. No premature abstraction — if Stage 3 wants to consume the aggregator differently, it can import or rewrite without affecting Stage 2's notebook.
- **Deferred**: forecast weather (serving stages), derived weather features (Stage 3), sub-hourly interpolation, spatial modelling.

## 14. Traceability

- Intent → **Stage 2 intent** (`docs/intent/02-weather-ingestion.md`).
- Layer architecture → **Ingestion layer** (`docs/architecture/layers/ingestion.md`) — now revisits Stage 2's multi-endpoint pattern and the `_common.py` extraction.
- Facts → **Research note** (`docs/lld/research/02-weather-ingestion.md`).
- Principles → **DESIGN.md** §2.1.1–§2.1.8, §3.2 (ingestion + features), §4.2 (source — flagged for correction), §7 (config framework).
