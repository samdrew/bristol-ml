# Services layer

- **Status:** Stub — shipped at Stage 0 with the
  text-statistics worked example.  Rename, replace, or remove as
  your project's domain demands.
- **Canonical overview:**
  [`DESIGN.md` §3.1](../../intent/DESIGN.md#31-layer-responsibilities).
- **Module guide:**
  [`src/TEMPLATE_PROJECT/services/CLAUDE.md`](../../../src/TEMPLATE_PROJECT/services/CLAUDE.md).

## Why this layer exists

The services layer wraps the pure :mod:`TEMPLATE_PROJECT.core`
modules with **Hydra-driven IO, logging, and CLI plumbing**.  Each
service is a thin module that:

1. Takes a validated Pydantic config (loaded via
   `TEMPLATE_PROJECT.config.load_config`).
2. Reads inputs from disk / network / stdin per the config.
3. Calls one or more pure `core/` functions.
4. Renders the output (JSON to stdout, file write, etc.).
5. Optionally exposes a `__main__` block for `python -m
   TEMPLATE_PROJECT.services.<name>` per DESIGN §2.1.1.

This separation keeps the core trivially unit-testable while
letting services own the side-effecting concerns.

## Contract

Every service module SHOULD:

- Take its config as the first argument to a `run()` function (or
  `main()` for the CLI entry point).
- Resolve the Hydra config via `TEMPLATE_PROJECT.config.load_config`
  exactly once at the entry-point boundary; pass the validated
  Pydantic model onward (downstream code never sees `DictConfig`).
- Log structured events through `loguru.logger`, not `print`.
- Return rendered output strings (not write directly to stdout)
  where possible; the caller decides what to do with them.
- Add `from __future__ import annotations` and pass type hints on
  every public signature.

A service module MAY:

- Open files, network connections, sockets.
- Call multiple `core/` functions and orchestrate their output.
- Define its own subcommand structure if the Hydra-only CLI is
  insufficient.

A service module MUST NOT:

- Reimplement logic that belongs in `core/`.  If you find yourself
  writing pure-function logic inside a service, extract it.
- Mutate the config.

## Internals

Current surface (as shipped):

- `text_stats_service.run(config: TextStatsConfig) -> str` — reads
  `config.input_path`, calls
  `core.text_stats.compute_text_statistics`, renders the result as
  JSON or as a small aligned text table per
  `config.output_format`.
- `text_stats_service.main(argv) -> int` — CLI entry point;
  resolves the Hydra config (with optional override args) and
  prints the rendered result.

This is the worked example.  Replace it with your project's
services.

## Cross-references

- Module guide:
  [`src/TEMPLATE_PROJECT/services/CLAUDE.md`](../../../src/TEMPLATE_PROJECT/services/CLAUDE.md).
- Core layer (composed by every service):
  [`docs/architecture/layers/core.md`](./core.md).
- Worked-example retro:
  [`docs/lld/stages/00-foundation.md`](../../lld/stages/00-foundation.md).
- Stage 0 plan:
  [`docs/plans/completed/00-foundation.md`](../../plans/completed/00-foundation.md).
