"""bristol_ml — reference ML architecture. See DESIGN.md."""

from importlib.metadata import PackageNotFoundError, version

from bristol_ml.config import load_config

try:
    __version__ = version("bristol_ml")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__", "load_config"]
