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
