# TEMPLATE_PROJECT

A Python 3.12 project scaffold extracted from the
[bristol_ml](https://github.com/) reference architecture.  Ships
with:

- A worked example (text-statistics service) demonstrating the
  `core/` + `services/` layer split and the Hydra+Pydantic config
  wiring.
- A four-tier documentation methodology (intent / architecture /
  plans / lld) and a stage-driven development workflow.
- A Claude Code agent roster with path-tier write hooks (see
  `.claude/`).

**The project spec lives at
[`docs/intent/DESIGN.md`](docs/intent/DESIGN.md)** — start there for
the full scope.

## Quick start

```bash
uv sync --group dev
uv run python -m TEMPLATE_PROJECT --help
uv run python -m TEMPLATE_PROJECT
uv run pytest
```

## Worked example

`src/TEMPLATE_PROJECT/services/text_stats_service.py` reads a UTF-8
text file pointed at by `cfg.services.text_stats.input_path` and
prints character / word / line counts.  Run it directly:

```bash
# Run with the default fixture
uv run python -m TEMPLATE_PROJECT.services.text_stats_service

# Override the input path via Hydra
uv run python -m TEMPLATE_PROJECT.services.text_stats_service \
    services.text_stats.input_path=/path/to/your/file.txt

# Switch to a human-readable output
uv run python -m TEMPLATE_PROJECT.services.text_stats_service \
    services.text_stats.output_format=human
```

The service composes:

- `src/TEMPLATE_PROJECT/core/text_stats.py` — pure function with no
  IO and no Hydra dependency.
- `src/TEMPLATE_PROJECT/services/text_stats_service.py` —
  config-driven IO wrapper.
- `conf/services/text_stats.yaml` — Hydra group definition.
- `conf/_schemas.py::TextStatsConfig` — Pydantic schema.
- `tests/unit/core/test_text_stats.py` (7 tests) +
  `tests/integration/test_text_stats_service.py` (4 tests) —
  spec-pinning regression guard.

The example exercises every load-bearing piece of the template
(layer separation, config wiring, CLI, unit + integration tests,
docs); strip it when you're ready to add your own domain.

## Adapting the template

See [`TEMPLATE_USAGE.md`](TEMPLATE_USAGE.md) for the full rename-
and-customise guide.  Quick summary:

1. Rename the package directory: `git mv src/TEMPLATE_PROJECT src/<your_name>`.
2. Update `pyproject.toml`'s `[project].name`, hatch wheel
   `packages`, and `[tool.ruff.lint.isort].known-first-party`.
3. Update imports in `src/<your_name>/{__init__,__main__,cli}.py`
   and any test that references `TEMPLATE_PROJECT`.
4. Replace the worked example with your domain logic.
5. Edit `docs/intent/DESIGN.md` to describe your project.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — Claude Code routing and agent
  conventions.
- [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md) — project spec.
- [`docs/architecture/`](docs/architecture/) — layer contracts and
  ADRs.
- [`docs/lld/stages/`](docs/lld/stages/) — stage retrospectives
  (stage 0 retro is the worked-example shipping record).
