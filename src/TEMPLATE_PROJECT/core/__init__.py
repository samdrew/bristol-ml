"""Core layer — pure, dependency-free domain logic.

Modules under ``core/`` are intentionally pure: they take primitive
values or dataclasses, return primitive values or dataclasses, and
have no IO, no global state, no Hydra dependency.  This keeps the
core unit-testable in isolation and reusable across services.

Replace the worked-example ``text_stats`` module when adapting the
template to your project.  See ``docs/architecture/layers/core.md``
for the layer contract.
"""

from __future__ import annotations
