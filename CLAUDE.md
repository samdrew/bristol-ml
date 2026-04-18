# Claude Code guidance for `bristol_ml`

This file is guidance for navigating and modifying the codebase. It is not the source of truth for intent — `DESIGN.md` is. Keep this file short; link out rather than duplicate.

## Canonical references

- `DESIGN.md` — scope, principles, stage plan. Read §2 (principles) and §6 (current layout) before any non-trivial change.
- `docs/architecture.md` — findability shim pointing at `DESIGN.md` §3.
- `docs/adrs/` — decisions that seem obvious in retrospect but weren't.

## Invocation pattern

Every module runs standalone (principle 2.1.1):

```
uv run python -m bristol_ml              # resolves + validates + prints config
uv run python -m bristol_ml.<module>     # stage-specific entry point
```

Config flows through `bristol_ml.load_config()` for programmatic callers (tests, notebooks).

## Configuration

Hydra composes YAML from `conf/`; `conf/_schemas.py` defines the Pydantic schemas the resolved config is validated against. Never accept a raw `DictConfig` in function signatures past the CLI boundary — call `bristol_ml.config.validate()` and pass the Pydantic model onward.

Adding a new config group is one sub-model in `_schemas.py`, one entry on `AppConfig`, and one subdirectory under `conf/`.

## Notebook policy

Notebooks import from `src/bristol_ml/`; they do not reimplement logic (principle 2.1.8). If you catch yourself writing logic in a `.ipynb`, move it into a module first.

## Stage hygiene

When a stage touches a module, the same PR must update:

- The module's own `CLAUDE.md` (added in the stage that first creates the module).
- `docs/stages/NN-name.md` with a retrospective.
- `CHANGELOG.md`.
- `DESIGN.md` §6 if the tree structure changed.
- `README.md` if a new entry point was added.

## What Stage 0 provides

Scaffold only — build system, Hydra+Pydantic config pipeline, CI, pre-commit, docs surface. No ingestion, models, or features yet. See `docs/stages/00-foundation.md`.

## Test surface

```
uv run pytest
```

CI runs the same command. Coverage is not a goal (principle 2.1.7).
