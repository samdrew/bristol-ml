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
