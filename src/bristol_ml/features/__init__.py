"""Features layer — functions that compose cleaned per-source data into model inputs.

See ``docs/architecture/layers/ingestion.md`` and Stage 2's LLD for how
the features layer composes per-station weather into a national aggregate.

At Stage 2 this layer contains ``weather.national_aggregate`` only; Stage 3
adds the feature assembler that joins demand + weather; Stage 5 adds
``calendar.derive_calendar`` and the ``weather_calendar`` schema
(``assemble_calendar`` / ``load_calendar`` / ``CALENDAR_OUTPUT_SCHEMA``)
alongside the Stage 3 surface.

This package does **not** eagerly import its submodules so that
``python -m bristol_ml`` (scaffold invocation) stays cheap. Import submodules
by name (``from bristol_ml.features import weather``) or resolve the top-level
alias lazily via ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.features.calendar import derive_calendar
    from bristol_ml.features.weather import national_aggregate

__all__ = ["derive_calendar", "national_aggregate"]


def __getattr__(name: str) -> object:
    """Lazy re-export of top-level feature functions.

    The assembler module (Stage 3 / Stage 5) has a wider public surface
    (``build``, ``load``, ``assemble``, ``OUTPUT_SCHEMA``; plus
    ``assemble_calendar`` / ``load_calendar`` / ``CALENDAR_OUTPUT_SCHEMA`` /
    ``CALENDAR_VARIABLE_COLUMNS`` from Stage 5) so is accessed by submodule
    import rather than a single top-level alias — ``from bristol_ml.features
    import assembler`` — matching the ``bristol_ml.ingestion`` convention.
    """
    if name == "national_aggregate":
        from bristol_ml.features import weather as _weather

        return _weather.national_aggregate
    if name == "derive_calendar":
        from bristol_ml.features import calendar as _calendar

        return _calendar.derive_calendar
    raise AttributeError(f"module 'bristol_ml.features' has no attribute {name!r}")
