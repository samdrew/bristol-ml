"""Real light backend, driving a Bluetooth Hue Go via the HueBLE library.

Reads come from HueBLE's pushed cache (it subscribes to GATT notifications on
connect), so get_state() is cheap and we don't hammer the radio. Writes map the
API's percentage/hex units onto HueBLE's 0-255 brightness and CIE xy colour.

Selected only when LIGHT_BACKEND=hue. The bleak/HueBLE imports are deferred so
the virtual demo has zero BLE dependencies.
"""

from __future__ import annotations

import logging
import os

from app.backends.base import LightBackend
from app.color import hex_to_xy, xy_to_hex
from app.models import LightState

logger = logging.getLogger(__name__)


def _patch_effect_type_tolerance() -> None:
    """Make HueBLE's EffectType tolerate effect codes it doesn't know about.

    The library hardcodes a small EffectType enum (values up to 0x11), but real
    lights report codes outside that set — e.g. a Hue Go sends 248 (0xF8). Its
    poll_effects() does a bare EffectType(raw), which raises ValueError on any
    unknown code and takes down connect(). We can't fix the library in place
    (it lives in site-packages and reinstalls would clobber it), so we install
    an enum _missing_ hook that maps unknown codes to NONE instead of crashing.
    """
    import HueBLE

    effect_type = HueBLE.EffectType
    if getattr(effect_type, "_huedemo_tolerant", False):
        return

    @classmethod
    def _missing_(cls, value):  # type: ignore[no-untyped-def]
        logger.warning("Unknown Hue effect code %r; treating as NONE", value)
        return cls.NONE

    effect_type._missing_ = _missing_
    effect_type._huedemo_tolerant = True


def _pct_to_255(level: int) -> int:
    # Hue brightness is 1-254 (0 would be off); clamp into that range.
    return max(1, min(254, round(level / 100 * 254)))


def _255_to_pct(value: int) -> int:
    return max(0, min(100, round(value / 254 * 100)))


class HueBackend(LightBackend):
    name = "hue"

    def __init__(self, address: str | None = None):
        # A CoreBluetooth UUID on macOS, a MAC on Linux. If unset we discover.
        self._address = address or os.getenv("HUE_ADDRESS")
        self._light = None  # type: ignore[var-annotated]
        self._last_color = "#ffffff"  # fallback when the light is in temp mode

    async def connect(self) -> None:
        from bleak import BleakScanner  # deferred
        import HueBLE

        _patch_effect_type_tolerance()

        device = None
        if self._address:
            device = await BleakScanner.find_device_by_address(self._address)
        else:
            found = await HueBLE.discover_lights()
            if found:
                device = found[0]

        if device is None:
            raise RuntimeError(
                "No Hue light found. Put it in pairing mode and set HUE_ADDRESS, "
                "or rely on discover_lights()."
            )

        self._light = HueBLE.HueBleLight(device)
        # First connection on macOS triggers the OS pairing prompt — accept it.
        await self._light.connect()
        await self._light.poll_state()

    async def close(self) -> None:
        if self._light is not None:
            await self._light.disconnect()

    async def get_state(self) -> LightState:
        light = self._require_light()
        xy = light.colour_xy
        brightness = light.brightness or 0

        if xy is not None:
            color = xy_to_hex(xy[0], xy[1], 1.0)
            self._last_color = color
        else:
            color = self._last_color  # light is in colour-temperature mode

        return LightState(
            on=bool(light.power_state),
            brightness=_255_to_pct(brightness),
            color=color,
            reachable=light.available,
            backend=self.name,
        )

    async def set_power(self, on: bool) -> None:
        await self._require_light().set_power(on)

    async def set_brightness(self, level: int) -> None:
        await self._require_light().set_brightness(_pct_to_255(level))

    async def set_color(self, hex_color: str) -> None:
        x, y = hex_to_xy(hex_color)
        await self._require_light().set_colour_xy(x, y)
        self._last_color = hex_color

    def _require_light(self):
        if self._light is None:
            raise RuntimeError("Hue backend not connected; call connect() first.")
        return self._light
