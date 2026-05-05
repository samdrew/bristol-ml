"""TEMPLATE_PROJECT — a Python project template scaffold.

See ``docs/intent/00-foundation.md`` for the worked example shipped
with the template, and ``TEMPLATE_USAGE.md`` for the
instantiation guide.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from TEMPLATE_PROJECT.config import load_config

try:
    __version__ = version("TEMPLATE_PROJECT")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__", "load_config"]
