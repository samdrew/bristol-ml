# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Stage 3: `bristol_ml.features.assembler` — feature-table assembler joining hourly-aggregated NESO demand with the population-weighted national weather aggregate. Public interface `OUTPUT_SCHEMA: pa.Schema`, `build(demand_hourly, weather_national, config, *, neso_retrieved_at_utc=None, weather_retrieved_at_utc=None) -> pd.DataFrame`, `load(path) -> pd.DataFrame`, `assemble(cfg, cache="offline") -> Path`. Writes atomic parquet to `${BRISTOL_ML_CACHE_DIR:-data/features}/<feature_set>.parquet` via `ingestion._common._atomic_write`. Emits one structured loguru INFO line per `build()` call covering demand-NaN rows dropped, per-variable forward-fill counts, and weather-NaN rows dropped after the fill cap. Module CLI `python -m bristol_ml.features.assembler [--cache auto|refresh|offline] [overrides ...]`. On-disk schema (ten columns: `timestamp_utc`, `nd_mw`, `tsd_mw`, five weather variables, two provenance columns) documented in [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md).
- Stage 3: `bristol_ml.evaluation` — new layer introduced to host time-series cross-validation primitives. First member `bristol_ml.evaluation.splitter.rolling_origin_split(n_rows, *, min_train, test_len, step, gap=0, fixed_window=False) -> Iterator[tuple[np.ndarray, np.ndarray]]` plus a `rolling_origin_split_from_config(n_rows, SplitterConfig)` sugar wrapper. Expanding-window default; `fixed_window=True` slides a constant-size training window. Module CLI `python -m bristol_ml.evaluation.splitter [--n-rows N] [overrides ...]`.
- Stage 3: Hydra config groups `conf/features/weather_only.yaml` (feature-set name, demand aggregation `mean` or `max` per `Literal`, cache location, forward-fill cap) and `conf/evaluation/rolling_origin.yaml` (one-year training window, 24-h day-ahead horizon, expanding window default). Wired through `AppConfig.features: FeaturesGroup` and `AppConfig.evaluation: EvaluationGroup` via four new Pydantic models in `conf/_schemas.py`.
- Stage 3: `notebooks/03_feature_assembler.ipynb` — thin demo notebook that primes the Stage 1 and Stage 2 caches, assembles the `weather_only` feature table, round-trips through `assembler.load`, enumerates rolling-origin folds, and plots fold boundaries overlaid on the GB demand series (year-long view plus a seven-fold zoom). Runs end-to-end in ~6 s on warm caches.
- Stage 3: spec-derived test suite — 17 assembler tests (determinism modulo provenance, column-order and dtype conformance, clock-change autumn / spring, missing-data policy including the D5 structured log), 52 splitter tests (no train/test overlap, fold-count arithmetic, gap handling, expanding vs fixed window, all four parameter boundaries, infeasibility surface), 7 implementer-derived CLI tests (atomic persistence, idempotence, missing-config refusal). `tests/conftest.py` gains a loguru-to-caplog adapter fixture so INFO-line assertions pass without plumbing changes in production code.
- Stage 3: retrospective at [`docs/lld/stages/03-feature-assembler.md`](docs/lld/stages/03-feature-assembler.md); layer docs [`docs/architecture/layers/features.md`](docs/architecture/layers/features.md) and [`docs/architecture/layers/evaluation.md`](docs/architecture/layers/evaluation.md); architecture-index layers table extended.

- Stage 2: `bristol_ml.ingestion.weather` — Open-Meteo historical-weather archive ingestion. Public interface `fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`; writes long-form hourly parquet (one row per station × hour) to `${BRISTOL_ML_CACHE_DIR:-data/raw/weather}/weather.parquet`. On-disk schema (`timestamp_utc`, `station`, five weather variables, `retrieved_at_utc`) documented in [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md).
- Stage 2: `bristol_ml.ingestion._common` — shared helpers extracted from `neso.py` (`CachePolicy`, `CacheMissingError`, `_atomic_write`, `_cache_path`, `_respect_rate_limit`, `_retrying_get`, `_RetryableStatusError`). Retry / rate-limit / cache-path helpers accept any config satisfying the structural `Protocol` types (`RetryConfig`, `RateLimitConfig`, `CachePathConfig`) — zero-behaviour-change refactor for Stage 1.
- Stage 2: `bristol_ml.features` — features layer introduced one stage earlier than DESIGN §9 implies so the Stage 2 notebook can import a weighted-mean helper instead of reimplementing it. `bristol_ml.features.weather.national_aggregate(df, weights)` collapses long-form per-station hourly weather into a wide-form national signal; subset-of-stations and NaN-safe weight renormalisation documented in the module docstring.
- Stage 2: Hydra config group `conf/ingestion/weather.yaml` wired through `AppConfig.ingestion.weather` (`WeatherStation`, `WeatherIngestionConfig` in `conf/_schemas.py`). Ten UK population centres with ONS 2011 Census BUA weights; default variables `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`.
- Stage 2: module CLIs `python -m bristol_ml.ingestion.weather` and `python -m bristol_ml.features.weather` — standalone Hydra-driven entry points per principle §2.1.1.
- Stage 2: `notebooks/02_weather_demand.ipynb` — thin demo notebook that primes both caches, joins hourly demand to the national weather aggregate, and plots temperature vs demand with a LOWESS fit; narrative motivates Open-Meteo over Met Office DataHub (AC5) and frames the curve as a hockey-stick rather than a symmetric V.
- Stage 2: recorded HTTP cassette `tests/fixtures/weather/cassettes/weather_2023_01.yaml` (London + Bristol, January 2023, five variables, ~64 kB) plus the one-off recorder at `scripts/record_weather_cassette.py`; `statsmodels` added as a runtime dependency for LOWESS.

- Stage 1: `bristol_ml.ingestion.neso` — NESO historic demand ingestion. Public interface `fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`, with `CachePolicy` values `AUTO | REFRESH | OFFLINE` and a `CacheMissingError` raised when `OFFLINE` is requested without a cache. Writes half-hourly parquet to `${BRISTOL_ML_CACHE_DIR:-data/raw/neso}/neso_demand.parquet`; on-disk schema (canonical `timestamp_utc` in UTC, plus `timestamp_local`, `settlement_date`, `settlement_period`, `nd_mw`, `tsd_mw`, `source_year`, `retrieved_at_utc`) documented in [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md).
- Stage 1: Hydra config group `conf/ingestion/neso.yaml` wired through `AppConfig.ingestion.neso` (`IngestionGroup` + `NesoIngestionConfig` in `conf/_schemas.py`). Covers the NESO annual resources for 2018–2025; adding a year is one YAML list entry.
- Stage 1: module CLI `python -m bristol_ml.ingestion.neso [--cache auto|refresh|offline] [overrides ...]` — calls `fetch` with the composed Hydra config and prints the resulting cache path.
- Stage 1: `notebooks/01_neso_demand.ipynb` — thin demo notebook that loads the cached parquet via `neso.load`, resamples to hourly, and plots one representative week of `nd_mw` plus daily peaks across the cached window.
- Stage 1: recorded HTTP cassette `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml` (two paginator pages of 500 rows for resource `bf5ab335-9b40-4ea4-b93a-ab4af7bce003`) plus the one-off recorder at `scripts/record_neso_cassette.py`, so integration tests replay offline under `--record-mode=none`.
- Stage 1: retrospective at [`docs/lld/stages/01-neso-demand-ingestion.md`](docs/lld/stages/01-neso-demand-ingestion.md).

### Changed

- Stage 2: `bristol_ml.ingestion.neso` refactored to import `CachePolicy`, `CacheMissingError`, `_atomic_write`, `_cache_path`, `_respect_rate_limit`, `_retrying_get`, and `_RetryableStatusError` from `bristol_ml.ingestion._common`. Public interface unchanged; Stage 1's 35 tests continue to pass without modification.
- Reorganised `docs/` into a tiered `intent/` / `architecture/` / `lld/` layout to support `.claude/hooks/tiered-write-paths.sh` from the lead agent. `DESIGN.md` moved to `docs/intent/DESIGN.md`; ADRs to `docs/architecture/decisions/`; stage retrospectives to `docs/lld/stages/`. Lead agent frontmatter wired to the tiered hook.

## [0.0.0] — 2026-04-18

### Added

- Project scaffold: `pyproject.toml` (hatchling, Python 3.12), `.gitignore`, `README.md`, `CLAUDE.md`.
- Hydra + Pydantic config pipeline: `conf/config.yaml`, `conf/_schemas.py`, `src/bristol_ml/config.py`, `src/bristol_ml/cli.py`.
- `python -m bristol_ml` entry point via `src/bristol_ml/__main__.py`.
- `py.typed` marker.
- Test suite (`tests/unit/test_config.py`) covering config load, `extra="forbid"` rejection, and the `--help` smoke.
- CI workflow (`.github/workflows/ci.yml`) — ruff format check, ruff lint, pytest via uv.
- Pre-commit hooks (`.pre-commit-config.yaml`).
- Docs: `docs/architecture.md` (pointer), ADRs 0001 and 0002, `docs/stages/00-foundation.md`.
