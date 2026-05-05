# Core layer

- **Status:** Stub — shipped at Stage 0 with the
  text-statistics worked example.  Rename, replace, or remove as
  your project's domain demands.
- **Canonical overview:**
  [`DESIGN.md` §3.1](../../intent/DESIGN.md#31-layer-responsibilities).
- **Module guide:**
  [`src/TEMPLATE_PROJECT/core/CLAUDE.md`](../../../src/TEMPLATE_PROJECT/core/CLAUDE.md).

## Why this layer exists

The core layer holds **pure, dependency-free domain logic**.
Modules under `core/` take primitive values or small Pydantic /
dataclass inputs, return primitive values or small Pydantic /
dataclass outputs, do **no** IO, hold **no** global state, and
**never** import Hydra or read configuration.  This keeps the core
unit-testable in isolation and reusable across services.

The layer is deliberately stateless at the package level —
`TEMPLATE_PROJECT.core` does not register handlers, does not
dispatch by name, and does not import heavy dependencies eagerly.
A user wanting `text_stats` imports
`from TEMPLATE_PROJECT.core.text_stats import compute_text_statistics`;
the package's `__init__.py` is intentionally near-empty.

## Contract

Every core module MUST:

- Be importable without side effects (no IO, no env-var reads, no
  filesystem touches at import time).
- Have a module-level `__all__` declaring its public surface.
- Accept primitive inputs (str, int, float, bytes, datetime,
  Pydantic / dataclass models).
- Return primitive outputs.
- Be safe to call concurrently (no shared mutable state).

A core module MUST NOT:

- Import Hydra, OmegaConf, or `TEMPLATE_PROJECT.config`.
- Read environment variables.
- Open network connections, files, or sockets.
- Import optional dependencies that are not declared in
  `pyproject.toml`'s runtime dep list.

## Internals

Current surface (as shipped):

- `text_stats.compute_text_statistics(text: str) -> TextStatistics`
  — counts characters, non-whitespace characters, words, and lines.
  `TextStatistics` is a frozen Pydantic model.

This is the worked example.  Replace it with your project's
domain logic.

## Cross-references

- Module guide:
  [`src/TEMPLATE_PROJECT/core/CLAUDE.md`](../../../src/TEMPLATE_PROJECT/core/CLAUDE.md).
- Worked-example retro:
  [`docs/lld/stages/00-foundation.md`](../../lld/stages/00-foundation.md).
- Stage 0 plan:
  [`docs/plans/completed/00-foundation.md`](../../plans/completed/00-foundation.md).
