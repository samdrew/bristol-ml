# `TEMPLATE_PROJECT.core` — module guide

The **core layer** holds pure, dependency-free domain logic.  Modules
here:

- Take primitive values or small Pydantic / dataclass inputs.
- Return primitive values or small Pydantic / dataclass outputs.
- Do **no** IO, hold **no** global state, **never** import Hydra or
  read configuration.
- Are unit-testable in isolation.

Services (`TEMPLATE_PROJECT.services`) wrap core functions with
Hydra-driven IO, logging, and CLI plumbing — but the core itself
stays clean.

## Current surface

- `text_stats.compute_text_statistics(text: str) -> TextStatistics`
  — counts characters, non-whitespace characters, words, and lines.
  The shipped worked example.

## When adding a new core module

1. Drop the module under `src/TEMPLATE_PROJECT/core/<name>.py`.
2. Make every public symbol importable from the module's `__all__`.
3. Add a unit test under `tests/unit/core/test_<name>.py`.
4. If a service wraps it, see `services/CLAUDE.md`.
5. If the module is non-trivial, document it in
   `docs/architecture/layers/core.md`.

## Cross-references

- Layer contract: [`docs/architecture/layers/core.md`](../../../docs/architecture/layers/core.md).
- Worked example retro: [`docs/lld/stages/00-foundation.md`](../../../docs/lld/stages/00-foundation.md).
