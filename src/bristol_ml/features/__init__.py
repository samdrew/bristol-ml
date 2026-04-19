"""Features layer — functions that compose cleaned per-source data into model inputs.

See ``docs/architecture/layers/ingestion.md`` and Stage 2's LLD for how
the features layer composes per-station weather into a national aggregate.

At Stage 2 this layer contains ``weather.national_aggregate`` only; Stage 3
adds the feature assembler that joins demand + weather + calendar.

This package does **not** eagerly import its submodules so that
``python -m bristol_ml`` (scaffold invocation) stays cheap. Import submodules
by name (``from bristol_ml.features import weather``) or resolve the top-level
alias lazily via ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.features.weather import national_aggregate

__all__ = ["national_aggregate"]


def __getattr__(name: str) -> object:
    """Lazy re-export of ``national_aggregate`` from the weather submodule."""
    if name == "national_aggregate":
        from bristol_ml.features import weather as _weather

        return _weather.national_aggregate
    raise AttributeError(f"module 'bristol_ml.features' has no attribute {name!r}")
