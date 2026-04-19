# bristol_ml

Reference ML/DS architecture with a GB day-ahead electricity demand forecaster as the worked example. **The project spec lives at [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md)** — start there for the full scope.

## Quick start

```
git clone <repo>
cd bristol-ml-reference
uv sync --group dev
uv run python -m bristol_ml --help
uv run pytest
```

## What is this?

- A pedagogical reference repo for a generic ML / data-science architecture.
- The worked example is GB national electricity demand, day-ahead.
- [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md) has the full scope. Each stage lands with a demoable artefact.

See [`CLAUDE.md`](CLAUDE.md) for Claude Code guidance, [`docs/architecture/`](docs/architecture/) for architectural decisions, and [`docs/lld/stages/`](docs/lld/stages/) for stage retrospectives.

## Worked example: NESO demand (Stage 1)

`notebooks/01_neso_demand.ipynb` loads cached GB half-hourly demand (sourced from the NESO Data Portal under OGL v3), resamples to hourly, and plots a representative week plus daily peaks across the cached window. The first run populates the local cache from the NESO CKAN API; subsequent runs work offline. See [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md) for the on-disk parquet schema, and the [Stage 1 retrospective](docs/lld/stages/01-neso-demand-ingestion.md) for the design choices.

## Worked example: weather + demand (Stage 2)

`notebooks/02_weather_demand.ipynb` primes a second ingestion cache from the [Open-Meteo](https://open-meteo.com) historical archive (CC BY 4.0), composes ten UK population centres into a population-weighted national temperature signal via `bristol_ml.features.weather.national_aggregate`, joins onto hourly demand, and plots the non-linear temperature-vs-demand curve with a LOWESS overlay. The curve is a **hockey-stick** on historical GB data — strong cold arm, flat warm arm — not a symmetric V; the hot-side response is only beginning to emerge post-2020 as residential cooling grows.

## Worked example: feature assembler + rolling-origin folds (Stage 3)

`notebooks/03_feature_assembler.ipynb` assembles the first end-to-end hourly feature table — demand joined with the population-weighted national weather aggregate — and enumerates rolling-origin train/test folds against it. The notebook calls `assembler.assemble(cfg, cache="auto")` to write `data/features/weather_only.parquet`, round-trips it through `assembler.load`, then plots fold boundaries overlaid on the GB demand series. Runs end-to-end in approximately 6 seconds on warm caches (D7). Entry points: `python -m bristol_ml.features.assembler --help` and `python -m bristol_ml.evaluation.splitter --help`. See [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md), [`src/bristol_ml/evaluation/CLAUDE.md`](src/bristol_ml/evaluation/CLAUDE.md), and the [Stage 3 retrospective](docs/lld/stages/03-feature-assembler.md).

## Worked example: linear baseline + evaluation harness (Stage 4)

`notebooks/04_linear_baseline.ipynb` fits a seasonal-naive floor and a statsmodels OLS weather-only regression behind the shared five-member `Model` protocol, runs both through the fold-level `evaluate` harness, and scores the pair against the NESO day-ahead forecast on the period they overlap. The `python -m bristol_ml.train` CLI is the single-invocation equivalent: `uv run python -m bristol_ml.train` runs the default linear model; `uv run python -m bristol_ml.train model=naive` swaps via Hydra; `uv run python -m bristol_ml.train --help` lists every override. Linear loses cleanly to NESO — that's the pedagogical setup for Stage 5 calendar features. Notebook runs end-to-end in ~8 seconds on warm caches (D7). Entry points: `python -m bristol_ml.train --help`, `python -m bristol_ml.models.linear --help`, `python -m bristol_ml.models.naive --help`, `python -m bristol_ml.evaluation.metrics --help`, `python -m bristol_ml.evaluation.harness --help`, `python -m bristol_ml.evaluation.benchmarks --help`, `python -m bristol_ml.ingestion.neso_forecast --help`. See [`src/bristol_ml/models/CLAUDE.md`](src/bristol_ml/models/CLAUDE.md), [`docs/architecture/layers/models.md`](docs/architecture/layers/models.md), and the [Stage 4 retrospective](docs/lld/stages/04-linear-baseline.md).
