"""bristol_ml — reference ML architecture. See docs/intent/DESIGN.md."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from bristol_ml.config import load_config

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.ingestion.neso import CacheMissingError, CachePolicy

try:
    __version__ = version("bristol_ml")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["CacheMissingError", "CachePolicy", "__version__", "load_config"]


def __getattr__(name: str) -> object:
    """Lazy re-export so importing ``bristol_ml`` does not pull in pandas.

    The ingestion types are nice for notebook ergonomics
    (``from bristol_ml import CachePolicy``) but pulling them eagerly would
    make ``python -m bristol_ml`` pay a pandas-import cost on every Hydra
    invocation and would re-introduce the runpy "module-already-loaded"
    warning on ``python -m bristol_ml.ingestion.neso``.
    """
    if name in {"CachePolicy", "CacheMissingError"}:
        from bristol_ml.ingestion import neso as _neso

        return getattr(_neso, name)
    raise AttributeError(f"module 'bristol_ml' has no attribute {name!r}")
