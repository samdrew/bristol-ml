"""The contract every light backend implements.

The REST layer only ever talks to this interface, so swapping the real HueBLE
light for the in-memory mock is a one-line change at startup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import LightState


class LightBackend(ABC):
    #: Short identifier surfaced in LightState.backend (e.g. "virtual", "hue").
    name: str = "base"

    async def connect(self) -> None:
        """Establish any connection the backend needs. No-op by default."""

    async def close(self) -> None:
        """Tear down. No-op by default."""

    @abstractmethod
    async def get_state(self) -> LightState:
        ...

    @abstractmethod
    async def set_power(self, on: bool) -> None:
        ...

    @abstractmethod
    async def set_brightness(self, level: int) -> None:
        """Set brightness from a 0-100 percentage."""

    @abstractmethod
    async def set_color(self, hex_color: str) -> None:
        """Set colour from a #RRGGBB hex string."""
