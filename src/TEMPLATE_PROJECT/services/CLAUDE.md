# `TEMPLATE_PROJECT.services` — module guide

The **services layer** wraps the pure
:mod:`TEMPLATE_PROJECT.core` modules with Hydra-driven IO, logging,
and CLI plumbing.

Each service:

1. Takes a validated Pydantic config (loaded via
   :func:`TEMPLATE_PROJECT.config.load_config`).
2. Reads inputs from disk / network / stdin per the config.
3. Calls one or more pure ``core`` functions.
4. Renders the output (JSON to stdout, file write, etc.).
5. Optionally exposes a `__main__` block for `python -m
   TEMPLATE_PROJECT.services.<name>` (per DESIGN §2.1.1).

This separation keeps the core trivially unit-testable while letting
services own the side-effecting concerns.

## Current surface

- `text_stats_service` — reads a UTF-8 text file specified by
  ``config.services.text_stats.input_path``, calls
  :func:`core.text_stats.compute_text_statistics`, and prints the
  result as JSON or a small human-readable table.

## When adding a new service

1. Drop a Pydantic schema for the service config in
   `conf/_schemas.py` (frozen + `extra="forbid"`).
2. Add a Hydra group YAML under `conf/services/<name>.yaml` with
   `# @package services.<name>`.
3. List the variant in `conf/config.yaml` under `defaults:` if the
   service runs by default; otherwise compose at the entry point
   with `+services=<name>`.
4. Add a field to `ServicesGroup` (in `conf/_schemas.py`) typed by
   the new schema.
5. Implement the service module under
   `src/TEMPLATE_PROJECT/services/<name>_service.py` with a `run()`
   function and a `main()` standalone entry point.
6. Add an integration test under
   `tests/integration/test_<name>_service.py`.

## Cross-references

- Layer contract: [`docs/architecture/layers/services.md`](../../../docs/architecture/layers/services.md).
- Worked example retro: [`docs/lld/stages/00-foundation.md`](../../../docs/lld/stages/00-foundation.md).
