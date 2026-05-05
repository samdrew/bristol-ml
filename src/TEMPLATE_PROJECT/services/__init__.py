"""Services layer — Hydra-driven wrappers around the core layer.

Each service is a thin module that:

1. Takes a validated Pydantic config (loaded via
   :func:`TEMPLATE_PROJECT.config.load_config`) instead of raw
   ``DictConfig``.
2. Reads inputs from disk / network / stdin per the config.
3. Calls one or more pure :mod:`TEMPLATE_PROJECT.core` functions.
4. Renders the output (JSON to stdout, file write, etc.).

This separation keeps the core trivially unit-testable while letting
services own the IO, the logging, and the CLI plumbing.

Replace the worked-example ``text_stats_service`` module when
adapting the template.  See
``docs/architecture/layers/services.md`` for the layer contract.
"""

from __future__ import annotations
