# Stage 0 — Project foundation

## Goal

Install the scaffolding — build system, config pipeline, CI, docs surface — so every subsequent stage has a single-command entry point and a consistent home for its module, tests, and retrospective.

## What was built

- `pyproject.toml` — hatchling, Python 3.12, runtime deps `hydra-core` + `omegaconf` + `pydantic`, dev group with `pytest` + `ruff` + `pre-commit`.
- `src/bristol_ml/` — package with `__init__.py`, `__main__.py`, `cli.py` (Hydra entry), `config.py` (resolve-then-validate), `py.typed`.
- `conf/` — `config.yaml` (project block + Hydra run/sweep dir overrides) and `_schemas.py` (Pydantic `AppConfig` with `extra="forbid"`).
- `tests/` — unit / integration / fixtures tree; three tests covering `load_config`, unknown-key rejection, and the `python -m bristol_ml --help` smoke.
- `.pre-commit-config.yaml` — pre-commit-hooks plus ruff-format + ruff-check.
- `.github/workflows/ci.yml` — checkout, setup-uv, uv sync, ruff format check, ruff lint, pytest. Targets `main`.
- `docs/architecture.md` (pointer into `DESIGN.md` §3), ADRs 0001 and 0002, this retrospective.
- `README.md`, `CLAUDE.md`, `CHANGELOG.md`.

## Design choices made here

- **Build backend**: hatchling. uv's default; needs no extra config for `src/` layout.
- **Module invocation**: `python -m bristol_ml` via `__main__.py` rather than a `[project.scripts]` console script, keeping a single canonical invocation pattern (principle 2.1.1).
- **Hydra working directories**: `hydra.run.dir` and `hydra.sweep.dir` redirected to `data/_runs/` (already gitignored) so Hydra's per-invocation artefacts do not litter the repo root.
- **`conf/__init__.py`**: added so `conf._schemas` is importable. Not listed in `DESIGN.md` §6's original tree; §6 updated in this same PR.
- **`py.typed`**: shipped so downstream consumers (and any future type-checker in CI) treat the package as typed (principle 2.1.2).
- **ruff `known-first-party = ["bristol_ml", "conf"]`**: keeps the schema import in the first-party block with the rest of the package.

## Demo moment

From a clean clone:

```
uv sync --group dev
uv run pytest                           # 3 tests pass
uv run python -m bristol_ml --help      # Hydra CLI help
uv run python -m bristol_ml             # prints validated JSON
tree -L 2                                # matches DESIGN.md §6
```

## Deferred

- The `Model` protocol (`src/bristol_ml/models/base.py`) — arrives at Stage 4 when the first model needs it. The pattern is committed here in ADR 0001 and `DESIGN.md` §7.3.
- Logging framework (`loguru` per `DESIGN.md` §8) — no module has anything to log yet; `print` is deliberate at Stage 0.
- Static type checker (mypy / pyright) — Stage 0's typed surface is small enough to be verified by use in tests. Add when a module genuinely benefits.
- MkDocs site (`DESIGN.md` §8) — defer until `docs/` has enough content to render.

## Next

→ Stage 1: NESO demand ingestion and the first plot.
