# Exploration — Stage 3: Feature assembler + train/test split

- **Written:** 2026-04-19
- **Status:** Pre-implementation exploration; feeds the Stage 3 brief and implementer spawn prompt.
- **Sources verified against:** commit `8e6b902` (Stage 2: Weather ingestion + national aggregation).

---

## 1. Entry points and boundaries

### Stage 1 output — NESO demand

**Path:** `${BRISTOL_ML_CACHE_DIR:-data/raw/neso}/neso_demand.parquet`
Config source: `conf/ingestion/neso.yaml` line 15 — `cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/neso}`.

**How to read:**
```python
from bristol_ml.ingestion import neso
path = neso.fetch(cfg.ingestion.neso, cache=CachePolicy.OFFLINE)
df = neso.load(path)   # asserts OUTPUT_SCHEMA, returns pd.DataFrame
```
`neso.load` at `src/bristol_ml/ingestion/neso.py:161` calls `pq.read_table` then validates every field in `OUTPUT_SCHEMA` by name and type. Hard error on mismatch.

**On-disk schema** (`neso.OUTPUT_SCHEMA`, `src/bristol_ml/ingestion/neso.py:82`):

| Column | Parquet type | Notes |
|---|---|---|
| `timestamp_utc` | `timestamp[us, tz=UTC]` | Start of half-hour period. Primary sort key. **Half-hourly cadence.** |
| `timestamp_local` | `timestamp[us, tz=Europe/London]` | Demo legibility only; never use for joins. |
| `settlement_date` | `date32` | NESO local settlement date. |
| `settlement_period` | `int8` | 1–46 (spring-forward), 1–48 (normal), 1–50 (autumn-fallback). |
| `nd_mw` | `int32` | National Demand in MW. |
| `tsd_mw` | `int32` | Transmission System Demand in MW. |
| `source_year` | `int16` | NESO annual resource. |
| `retrieved_at_utc` | `timestamp[us, tz=UTC]` | Per-fetch provenance. |

Primary key: `timestamp_utc` unique, sorted ascending.
Date range: 2018-01-01 onwards (from `conf/ingestion/neso.yaml`).

**Cadence note:** half-hourly (48 rows per normal day; 46 on spring-forward, 50 on autumn-fallback). The assembler must aggregate to hourly.

### Stage 2 output — weather

**Path:** `${BRISTOL_ML_CACHE_DIR:-data/raw/weather}/weather.parquet`
Config source: `conf/ingestion/weather.yaml` line 22.

**How to read:**
```python
from bristol_ml.ingestion import weather
path = weather.fetch(cfg.ingestion.weather, cache=CachePolicy.OFFLINE)
df = weather.load(path)   # long-form; asserts OUTPUT_SCHEMA
```
`weather.load` at `src/bristol_ml/ingestion/weather.py:164` follows the same pattern as `neso.load`.

**On-disk schema** (`weather.OUTPUT_SCHEMA`, `src/bristol_ml/ingestion/weather.py:84`):

| Column | Parquet type | Notes |
|---|---|---|
| `timestamp_utc` | `timestamp[us, tz=UTC]` | Hourly. Open-Meteo returns UTC natively; no DST algebra needed. |
| `station` | `string` | Lowercase snake-case, e.g. `london`. |
| `temperature_2m` | `float32` | °C |
| `dew_point_2m` | `float32` | °C |
| `wind_speed_10m` | `float32` | km/h |
| `cloud_cover` | `int8` | %, 0–100 |
| `shortwave_radiation` | `float32` | W/m² |
| `retrieved_at_utc` | `timestamp[us, tz=UTC]` | Per-fetch provenance. |

Primary key: `(timestamp_utc, station)` unique, sorted `timestamp_utc ASC, station ASC`.
Date range: `start_date: 2018-01-01` (`conf/ingestion/weather.yaml` line 26), end omitted → today.
Ten stations; long-form (one row per station per hour).

**National aggregate intermediate** (not persisted, computed on demand):
`bristol_ml.features.weather.national_aggregate(df, weights) -> pd.DataFrame`
Returns wide-form frame, index = `timestamp_utc`, one column per weather variable.
Source: `src/bristol_ml/features/weather.py:39`.
Weights from: `{s.name: s.weight for s in cfg.ingestion.weather.stations}` — see CLI at `src/bristol_ml/features/weather.py:192`.

**NaN policy:** NaN at one station drops that station from that `(hour, variable)` group; remaining weights renormalise. All-NaN propagates NaN. Documented at `src/bristol_ml/features/weather.py:64`.

### `bristol_ml.config.validate` and Hydra CLI boundary

Config is loaded via `bristol_ml.config.load_config` (`src/bristol_ml/config.py:41`) or via `@hydra.main` in `src/bristol_ml/cli.py:16`. Both paths call `validate(cfg)` which returns an `AppConfig` Pydantic object. Downstream code never touches raw `DictConfig`.

The Stage 3 CLI entry point pattern to follow:
```python
# inside _cli_main:
from bristol_ml.config import load_config
cfg = load_config(overrides=list(args.overrides))
# cfg.ingestion.neso, cfg.ingestion.weather, cfg.features (once added)
```
Both Stage 1 (`src/bristol_ml/ingestion/neso.py:491`) and Stage 2 (`src/bristol_ml/ingestion/weather.py:392`) import `load_config` locally inside `_cli_main` to keep the `--help` path cheap.

### Provenance columns

Both parquets carry `retrieved_at_utc: timestamp[us, tz=UTC]` equal across all rows of a single fetch call (`src/bristol_ml/ingestion/neso.py:139`, `src/bristol_ml/ingestion/weather.py:141`). The assembler's output parquet should either propagate these as separate columns (`neso_retrieved_at_utc`, `weather_retrieved_at_utc`) or document explicitly that they are dropped; neither convention is established yet.

---

## 2. Patterns to follow

### Ingestion module file layout

Both existing ingesters follow an identical five-part layout. Stage 3 modules should mirror it exactly:

```
src/bristol_ml/features/
├── __init__.py          # lazy re-exports via __getattr__; no eager pandas import
├── weather.py           # already exists (Stage 2)
├── assembler.py         # NEW Stage 3 — join + resample
└── CLAUDE.md            # update to document assembler

src/bristol_ml/evaluation/
├── __init__.py          # NEW Stage 3
├── splitter.py          # NEW Stage 3 — rolling-origin split utility
└── CLAUDE.md            # NEW Stage 3

conf/
├── features/            # NEW group directory
│   └── weather_only.yaml    # feature set config (name chosen per §"Feature-set naming")
└── evaluation/          # NEW group directory
    └── rolling_origin.yaml  # splitter config

tests/unit/features/     # already exists; add test_assembler.py
tests/unit/evaluation/   # NEW __init__.py + test_splitter.py
tests/fixtures/features/ # NEW — small demand + weather fixtures for assembler smoke test
```

Reference: `docs/architecture/layers/ingestion.md:43` for the five-part description.

### Real/stub behind the same interface

Stages 1 and 2 achieve this via `CachePolicy.OFFLINE` — tests pass a `tmp_path`-based `cache_dir` and use cassette replay. There is no explicit stub class. The assembler should follow the same pattern: tests supply small fixture parquets (not a real internet call) and the `load_config` path stays exercised via `tmp_path` overrides. See `tests/unit/ingestion/test_neso.py:77` (`_build_config` helper) for the pattern.

### `python -m bristol_ml.<module>` wiring

Every module has a `_cli_main(argv=None) -> int` and a guard:
```python
if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
```
The `__main__.py` for sub-modules follows the ingestion pattern: `_build_cli_parser()` + `_cli_main()` with local `load_config` import. See `src/bristol_ml/ingestion/neso.py:458` and `src/bristol_ml/ingestion/weather.py:362`.

### Existing Pydantic schemas (`conf/_schemas.py`)

Current models (all at `conf/_schemas.py`):

| Model | Key fields |
|---|---|
| `ProjectConfig` | `name: str` (pattern `^[a-z][a-z0-9_]*$`), `seed: int` |
| `NesoYearResource` | `year: int`, `resource_id: UUID` |
| `NesoIngestionConfig` | `base_url`, `resources`, `cache_dir: Path`, `cache_filename`, page/retry/rate knobs |
| `WeatherStation` | `name` (same regex), `latitude`, `longitude`, `weight`, `weight_source` |
| `WeatherIngestionConfig` | `base_url`, `stations`, `variables`, `start_date`, `end_date`, `cache_dir`, retry/rate knobs, `timezone`; `model_validator` for date order + unique names |
| `IngestionGroup` | `neso: NesoIngestionConfig | None`, `weather: WeatherIngestionConfig | None` |
| `AppConfig` | `project: ProjectConfig`, `ingestion: IngestionGroup` |

All models use `model_config = ConfigDict(extra="forbid", frozen=True)`.
`IngestionGroup` makes each field `| None = None` so Stage 0 configs still validate — Stage 3 should add `FeaturesGroup` and `EvaluationGroup` with the same optional-defaulting pattern.
`AppConfig` adds `features: FeaturesGroup = Field(default_factory=FeaturesGroup)` and `evaluation: EvaluationGroup = Field(default_factory=EvaluationGroup)`.

### Parquet schema enforcement

Both modules define an `OUTPUT_SCHEMA: pa.Schema` constant and validate it in `load()` by iterating fields and comparing `field.name` + `field.type`. There is **no pandera** in the project — only `pyarrow` schemas. `pandera` is not in `pyproject.toml` dependencies. The assembler's output should define its own `OUTPUT_SCHEMA: pa.Schema` and enforce it in its `load()` function using the same loop pattern.

### Atomic writes

`bristol_ml.ingestion._common._atomic_write(table, path)` (`src/bristol_ml/ingestion/_common.py:201`) — tmp file + `os.replace`. Stage 3 assembler output should use the same helper (it is already in `_common`; the assembler can import it directly).

### Tests — style and location

- Tests under `tests/unit/<module>/` (filename mirrors the source file: `test_assembler.py`, `test_splitter.py`).
- Tests under `tests/integration/<module>/` only when network or parquet round-trip is needed.
- `pytest.importorskip("bristol_ml.features.assembler")` at module top — see `tests/unit/features/test_weather_aggregate.py:55`.
- Fixture helpers follow the `_build_config(tmp_path, ...)` pattern in `tests/unit/ingestion/test_neso.py:77`.
- Hand-crafted fixture CSVs (not cassettes) for pure-logic tests, following `tests/fixtures/weather/toy_stations.csv`.
- No `xfail`, no skipped tests, no `# type: ignore` without explanation.

---

## 3. Integration points where Stage 3 must plug in

### Hydra config tree (`conf/`)

Existing config groups:
```
conf/
├── config.yaml           # defaults: [ingestion/neso@ingestion.neso, ingestion/weather@ingestion.weather]
└── ingestion/
    ├── neso.yaml         # @package ingestion.neso
    └── weather.yaml      # @package ingestion.weather
```

Stage 3 adds two new group directories:
```
conf/features/
    weather_only.yaml     # @package features.weather_only (or features.v1)
conf/evaluation/
    rolling_origin.yaml   # @package evaluation.rolling_origin
```

`conf/config.yaml` `defaults:` list gains two more entries:
```yaml
- features/weather_only@features.weather_only
- evaluation/rolling_origin@evaluation.rolling_origin
```

The `@package` directive pattern is established at `conf/ingestion/neso.yaml:1` (`# @package ingestion.neso`) and `conf/ingestion/weather.yaml:1` (`# @package ingestion.weather`).

### `src/bristol_ml/__init__.py`

Currently re-exports: `__version__`, `load_config`, `CachePolicy` (lazy), `CacheMissingError` (lazy).
Source: `src/bristol_ml/__init__.py:18`.

Stage 3 expected additions (follow the existing `__getattr__` lazy pattern): none are strictly mandated by convention, but if the assembler or splitter are likely to be imported from notebooks, add lazy re-exports in the same style. The features layer already has its own `__getattr__` in `src/bristol_ml/features/__init__.py:25`.

### `src/bristol_ml/__main__.py`

No change needed. It calls `bristol_ml.cli.main` which is Hydra's entry point and will pick up new config groups automatically once they are wired into `conf/config.yaml`.

### `CHANGELOG.md`

Current structure: `## [Unreleased]` with `### Added` and `### Changed` sub-sections. Stage 3 adds bullets in the same format as the Stage 2 entries (one bullet per shipped artefact: module, config group, CLI, notebook, tests, fixtures).
See `CHANGELOG.md:9`.

### Stage hygiene checklist (from `CLAUDE.md`)

On PR:
- `src/bristol_ml/features/CLAUDE.md` — update with assembler surface.
- `src/bristol_ml/evaluation/CLAUDE.md` — new file.
- `docs/lld/stages/03-feature-assembler.md` — retrospective following template at `docs/lld/stages/00-foundation.md`.
- `CHANGELOG.md` under `[Unreleased]`.
- If structural: `docs/intent/DESIGN.md` §6 (deny-tier, human approval) and `README.md`.

---

## 4. Existing tests and fixtures

### Test directory layout

```
tests/
├── __init__.py
├── unit/
│   ├── test_config.py                        # Stage 0; 3 tests
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── test_neso.py                      # Stage 1; 31 tests
│   │   ├── test_neso_helpers.py              # Stage 1 implementer helpers
│   │   ├── test_weather.py                   # Stage 2
│   │   └── test_weather_helpers.py           # Stage 2 implementer helpers
│   └── features/
│       ├── __init__.py
│       ├── test_weather_aggregate.py         # Stage 2 spec-derived (15 tests)
│       └── test_weather_aggregate_helpers.py # Stage 2 implementer-derived (10 tests)
├── integration/
│   ├── __init__.py
│   └── ingestion/
│       ├── __init__.py
│       ├── test_neso_cassettes.py            # Stage 1 cassette replay
│       └── test_weather_cassettes.py         # Stage 2 cassette replay
└── fixtures/
    ├── __init__.py
    ├── neso/
    │   ├── clock_change_rows.csv             # 12 rows: autumn-fallback 2024-10-27 + spring-forward 2024-03-31
    │   └── cassettes/neso_2023_refresh.yaml
    └── weather/
        ├── toy_stations.csv                  # 8 rows: 4 stations × 2 hours
        └── cassettes/weather_2023_01.yaml
```

### Fixture schemas available for assembler smoke tests

**`tests/fixtures/neso/clock_change_rows.csv`** — columns: `SETTLEMENT_DATE`, `SETTLEMENT_PERIOD`, `ND`, `TSD`. Raw (pre-`_to_utc`) format. Covers autumn-fallback (periods 1–7 on 2024-10-27) and spring-forward (periods 1–3, 46, 47 on 2024-03-31). Not in the parquet output format — the assembler smoke test needs a parquet fixture, not this CSV.

**`tests/fixtures/weather/toy_stations.csv`** — columns: `timestamp_utc`, `station`, `latitude`, `longitude`, `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`. Four stations (`london`, `bristol`, `manchester`, `glasgow`) × 2 hours (`2023-01-01T00:00:00+00:00`, `2023-01-01T01:00:00+00:00`). This is in long-form weather format — **directly usable** as input to `national_aggregate` and hence to the assembler.

**Stage 3 fixture needs:** The assembler smoke test needs a small demand parquet (half-hourly, a handful of hours) that aligns with the weather fixture timestamps. The implementer should create `tests/fixtures/features/toy_demand.parquet` (or CSV converted at test time) covering at least `2023-01-01T00:00` – `2023-01-01T01:30` in 30-min increments to give 2 hourly buckets when resampled, matching the weather fixture's 2 UTC hours. The cassette pattern (`pytest-recording` VCR) is **not** needed for the assembler — there is no HTTP call.

---

## 5. Gotchas worth flagging

### Gotcha 1: Cadence mismatch — half-hourly demand vs hourly weather

**The single largest design decision in Stage 3.** NESO demand is half-hourly (`timestamp_utc` at :00 and :30); weather is hourly (:00 only). The assembler must aggregate demand to hourly before joining.

The aggregation function is unspecified by the intent; the intent explicitly flags it as a decision that "affects all downstream modelling" (`docs/intent/03-feature-assembler.md:43`). Options: `mean`, `sum`, `peak` (max). The decision must be documented in the assembler's docstring (not left implicit) because every model inherits it.

After aggregation the join is a straightforward `timestamp_utc` inner join — both sides are UTC, both are `timestamp[us, tz=UTC]`.

**Clock-change days:** On autumn-fallback (50 periods), the naive hour-bucket `timestamp_utc.dt.floor("h")` produces two rows labelled `01:00 UTC` — the 00:30 UTC row and the 01:00 UTC row both floor to `01:00 UTC` (or rather the five rows from 00:00–02:00 UTC collapse normally since everything is already in UTC). There is no ambiguity once in UTC — the danger is only if anyone tries to aggregate in local time before converting. Since Stage 1 converts to UTC before persisting, the assembler works in UTC throughout and clock-change days are handled correctly by `timestamp_utc.dt.floor("h")`.

**Spring-forward (46 periods):** The UTC hour `01:00–02:00` (local non-existent hour) still exists in UTC — there are simply no NESO rows for it because the NESO day has 46 periods (the 01:00–02:00 BST window is absent). After flooring to hours, this UTC hour will produce NaN demand in the join. The missing-data policy must cover this explicitly.

Cited: `src/bristol_ml/ingestion/neso.py:327` (`_to_utc` — the DST arithmetic), `docs/intent/03-feature-assembler.md:43`.

### Gotcha 2: Timezone consistency — both UTC, but `timestamp_local` must not be used for joins

Both parquets store `timestamp_utc` as `timestamp[us, tz=UTC]`. The NESO parquet also carries `timestamp_local` as `timestamp[us, tz=Europe/London]`. The weather parquet has no local-time column at all.

**The trap:** joining on `timestamp_local` (or flooring in local time) breaks on clock-change days. The ingestion CLAUDE.md at `src/bristol_ml/ingestion/CLAUDE.md:27` states explicitly: "`timestamp_local`... For demo legibility only; **never used for arithmetic or joins**."

The assembler must join exclusively on `timestamp_utc`. If the assembler creates any local-time column for demo legibility (a reasonable choice given the notebook purpose), it must not use it as the join key.

### Gotcha 3: `pandera` not in the dependency tree — schema enforcement is pyarrow-only

The intent document (`docs/intent/03-feature-assembler.md:47`) explicitly raises `pandera` as an option: "Schema enforcement. `pandera` is idiomatic for DataFrame schemas; Pydantic is already present from Stage 0. Either works. The choice probably warrants a small ADR."

`pandera` is **not** in `pyproject.toml` (confirmed at `/workspace/pyproject.toml`). Adding it requires a `pyproject.toml` change plus a `uv lock` update (never use `pip`). The existing modules use only `pyarrow` schemas — defining `OUTPUT_SCHEMA: pa.Schema` and asserting in `load()`. An implementer who reaches for `pandera` needs to add the dependency first and produce an ADR; an implementer who follows the existing pyarrow pattern can proceed without touching `pyproject.toml`.

This is a branch-point that should be resolved in the stage brief or by the lead before the implementer starts.

Cited: `pyproject.toml:16` (dependencies), `src/bristol_ml/ingestion/neso.py:82` (`OUTPUT_SCHEMA`), `docs/intent/03-feature-assembler.md:47`.

### Gotcha 4: Where the assembled feature table lives on disk

The intent notes: "Since it's regenerable, a location outside the repo is reasonable." (`docs/intent/03-feature-assembler.md:48`). No `conf/features/` group exists yet. The implementer must decide on `cache_dir` + `cache_filename` pattern (mirroring `NesoIngestionConfig` and `WeatherIngestionConfig`) and wire it through a new `FeaturesConfig` / `FeatureSetConfig` Pydantic model and a `conf/features/weather_only.yaml` Hydra group.

The `cache_dir` YAML value should use the `${oc.env:BRISTOL_ML_CACHE_DIR,...}` pattern used by both existing ingesters (`conf/ingestion/neso.yaml:15`, `conf/ingestion/weather.yaml:22`). A natural default: `${oc.env:BRISTOL_ML_CACHE_DIR,data/features}`.

### Gotcha 5: Feature-set naming — "weather_only" vs "default"

The intent warns: "Stage 3's set is weather-only; Stage 5 will extend it with calendar features. Naming them distinctly now (rather than calling both 'default') keeps the without/with comparison clean later." (`docs/intent/03-feature-assembler.md:49`).

The Hydra group file should be `conf/features/weather_only.yaml` (not `default.yaml`). The `@package` directive must also use a non-clashing name. This is a naming-at-origin decision; changing it after Stage 5 lands requires touching many files.

### Gotcha 6: The `columns` field on `NesoIngestionConfig` is dead weight

`NesoIngestionConfig.columns: list[str]` exists in `conf/_schemas.py:51` and `conf/ingestion/neso.yaml:23`, but is never consumed in `neso.py`. Noted as deferred cleanup in `docs/lld/stages/02-weather-ingestion.md:71`. The assembler should **not** try to use this field as a mechanism to select which demand columns to include — it is inert. Select columns explicitly in the assembler.

---

## 6. Data flow / call graph

```
conf/config.yaml
  └── load_config() → AppConfig
        ├── .ingestion.neso  → neso.fetch() → neso_demand.parquet
        │                       neso.load()  → pd.DataFrame (half-hourly)
        │                                         │
        │                     assembler._resample_demand_to_hourly()
        │                                         ↓
        ├── .ingestion.weather → weather.fetch() → weather.parquet
        │                          weather.load() → long-form pd.DataFrame
        │                                           │
        │                        features.weather.national_aggregate(df, weights)
        │                                           ↓ wide hourly pd.DataFrame
        │
        └── .features.weather_only → assembler.build(demand_hourly, weather_national, config)
                                              ↓
                                       feature_table.parquet   (OUTPUT_SCHEMA)
                                              ↓
              .evaluation.rolling_origin → splitter.rolling_origin_split(df, config)
                                              ↓
                                        [(train_idx, test_idx), ...]
```

The assembler's `build()` function signature (to be designed):
```python
def build(
    demand: pd.DataFrame,   # hourly demand (after resample)
    weather: pd.DataFrame,  # wide-form national aggregate (index=timestamp_utc)
    config: FeatureSetConfig,
) -> pd.DataFrame: ...
```
It joins on `timestamp_utc`, drops `retrieved_at_utc` columns (or propagates them with source-prefixed names), enforces `OUTPUT_SCHEMA`, and returns the feature table.

The splitter:
```python
def rolling_origin_split(
    df: pd.DataFrame,
    config: SplitterConfig,
) -> list[tuple[np.ndarray, np.ndarray]]: ...
```
Returns integer index arrays (not boolean masks), per intent AC4.

---

## Open questions for the lead / brief

1. **Demand aggregation function for hourly resampling** — `mean`, `sum`, or `peak`? Document choice in the assembler docstring and the stage brief. `mean` is the most natural pedagogical choice (it preserves the scale roughly); `sum` is what a total-energy metric would want.

2. **`pandera` vs pyarrow schema for the feature table output** — follow existing pyarrow pattern (no new dependency) or introduce pandera with an ADR? Recommend pyarrow for consistency unless there is a specific pedagogical reason to show pandera.

3. **Provenance handling in the assembled output** — propagate `neso_retrieved_at_utc` + `weather_retrieved_at_utc` as two columns, one column (max/latest), or drop both? Existing convention (one `retrieved_at_utc` per parquet) does not cover a joined table.

4. **Missing data policy** — demand NaN on spring-forward hours (no NESO rows), weather NaN for recent days (Open-Meteo ~5-day delay), weather NaN when all stations missing for a variable. The intent says document once at the assembler's docstring level. Dropping demand NaN rows is safest for modelling; forward-filling weather NaN (up to a configurable cap) is the convention the intent suggests.

5. **Fixture format** — create `tests/fixtures/features/toy_demand.parquet` (preferred; matches storage convention) or a CSV that is read and converted in the test? The ingestion tests use CSVs for the pre-parquet raw format and cassettes for the full round-trip. A small hand-crafted parquet is the cleanest option for the assembler smoke test since it exercises the `neso.load`-compatible schema directly.
