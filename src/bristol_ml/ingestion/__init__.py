"""Ingestion layer — fetchers + typed parquet loaders, one module per source.

See ``docs/architecture/layers/ingestion.md`` for the shared contract:
``fetch(config) -> Path``, ``load(path) -> DataFrame``, and ``CachePolicy``.

This package intentionally does **not** eagerly import its per-source
submodules (``neso``, ``weather``), so that
``python -m bristol_ml.ingestion.<source>`` does not trigger runpy's
"module found in sys.modules before execution" warning. Import submodules
by name instead (``from bristol_ml.ingestion import neso``) or import the
public types directly (``from bristol_ml.ingestion._common import CachePolicy``).

``CachePolicy`` and ``CacheMissingError`` are resolved lazily via
``__getattr__`` from ``bristol_ml.ingestion._common`` — the shared
extraction point introduced at Stage 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.ingestion._common import CacheMissingError, CachePolicy

__all__ = ["CacheMissingError", "CachePolicy"]


def __getattr__(name: str) -> object:
    """Lazy re-export of ``CachePolicy`` and ``CacheMissingError`` from ``_common``.

    Keeping the re-export behind ``__getattr__`` means that simply importing
    ``bristol_ml.ingestion`` does not force the per-source modules to load —
    which both avoids the ``runpy`` warning on CLI invocation and keeps a
    ``python -m bristol_ml`` (scaffold-only workflow) free of pandas/pyarrow
    import cost.
    """
    if name in {"CachePolicy", "CacheMissingError"}:
        from bristol_ml.ingestion import _common

        return getattr(_common, name)
    raise AttributeError(f"module 'bristol_ml.ingestion' has no attribute {name!r}")
