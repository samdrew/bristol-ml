# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Stage 1: `bristol_ml.ingestion.neso` — NESO historic demand ingestion. Public interface `fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`, with `CachePolicy` values `AUTO | REFRESH | OFFLINE` and a `CacheMissingError` raised when `OFFLINE` is requested without a cache. Writes half-hourly parquet to `${BRISTOL_ML_CACHE_DIR:-data/raw/neso}/neso_demand.parquet`; on-disk schema (canonical `timestamp_utc` in UTC, plus `timestamp_local`, `settlement_date`, `settlement_period`, `nd_mw`, `tsd_mw`, `source_year`, `retrieved_at_utc`) documented in [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md).
- Stage 1: Hydra config group `conf/ingestion/neso.yaml` wired through `AppConfig.ingestion.neso` (`IngestionGroup` + `NesoIngestionConfig` in `conf/_schemas.py`). Covers the NESO annual resources for 2018–2025; adding a year is one YAML list entry.
- Stage 1: module CLI `python -m bristol_ml.ingestion.neso [--cache auto|refresh|offline] [overrides ...]` — calls `fetch` with the composed Hydra config and prints the resulting cache path.
- Stage 1: `notebooks/01_neso_demand.ipynb` — thin demo notebook that loads the cached parquet via `neso.load`, resamples to hourly, and plots one representative week of `nd_mw` plus daily peaks across the cached window.
- Stage 1: recorded HTTP cassette `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml` (two paginator pages of 500 rows for resource `bf5ab335-9b40-4ea4-b93a-ab4af7bce003`) plus the one-off recorder at `scripts/record_neso_cassette.py`, so integration tests replay offline under `--record-mode=none`.
- Stage 1: retrospective at [`docs/lld/stages/01-neso-demand-ingestion.md`](docs/lld/stages/01-neso-demand-ingestion.md).

### Changed

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
