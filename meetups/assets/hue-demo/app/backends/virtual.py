"""In-memory light. No hardware, no Bluetooth, no setup.

This is the default backend so anyone at the meetup can drive the API and watch
the circle move without owning a Hue Go. State lives in process memory and is
read straight back out, so the visualiser reflects changes instantly.
"""

from __future__ import annotations

from app.backends.base import LightBackend
from app.models import LightState


class VirtualBackend(LightBackend):
    name = "virtual"

    def __init__(self, color: str = "#ffd6aa", brightness: int = 80, on: bool = True):
        self._on = on
        self._brightness = brightness
        self._color = color

    async def get_state(self) -> LightState:
        return LightState(
            on=self._on,
            brightness=self._brightness,
            color=self._color,
            reachable=True,
            backend=self.name,
        )

    async def set_power(self, on: bool) -> None:
        self._on = on

    async def set_brightness(self, level: int) -> None:
        self._brightness = max(0, min(100, level))

    async def set_color(self, hex_color: str) -> None:
        # Validate/normalise via the colour module so bad input fails loudly.
        from app.color import hex_to_rgb, rgb_to_hex

        self._color = rgb_to_hex(*hex_to_rgb(hex_color))
