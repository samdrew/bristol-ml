"""Backend factory.

Selects the light backend from the LIGHT_BACKEND environment variable
(default "virtual"). The Hue backend is imported lazily so its Bluetooth
dependencies are only required when actually requested.
"""

from __future__ import annotations

import os

from app.backends.base import LightBackend


def make_backend() -> LightBackend:
    """Return a LightBackend chosen by the LIGHT_BACKEND env var."""
    kind = os.getenv("LIGHT_BACKEND", "virtual").lower()

    if kind in ("virtual", "mock"):
        from app.backends.virtual import VirtualBackend

        return VirtualBackend()

    if kind == "hue":
        from app.backends.hue import HueBackend

        return HueBackend()

    raise ValueError(
        f"Unknown LIGHT_BACKEND {kind!r}; expected 'virtual' or 'hue'."
    )


__all__ = ["make_backend"]
