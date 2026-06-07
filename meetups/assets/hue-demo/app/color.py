"""Colour conversion between sRGB hex strings and CIE 1931 xy coordinates.

HueBLE speaks CIE xy (chromaticity only, brightness is separate). The REST API
and the visualiser speak hex. These helpers bridge the two. The matrices are the
ones Philips/Signify publish for the wide colour gamut; round-trips are close but
not exact because of gamut clamping, which is fine for a demo.
"""

from __future__ import annotations


def _linear(c: float) -> float:
    """sRGB component (0-1) -> linear light."""
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _srgb(c: float) -> float:
    """Linear light -> sRGB component (0-1)."""
    c = max(c, 0.0)
    return 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"invalid hex colour: {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(v))) for v in (r, g, b)))


def hex_to_xy(value: str) -> tuple[float, float]:
    r, g, b = (v / 255 for v in hex_to_rgb(value))
    r, g, b = _linear(r), _linear(g), _linear(b)

    X = r * 0.649926 + g * 0.103455 + b * 0.197109
    Y = r * 0.234327 + g * 0.743075 + b * 0.022598
    Z = r * 0.000000 + g * 0.053077 + b * 1.035763

    total = X + Y + Z
    if total == 0:
        return (0.0, 0.0)
    return (round(X / total, 4), round(Y / total, 4))


def xy_to_hex(x: float, y: float, brightness: float = 1.0) -> str:
    """Convert xy (+ optional luminance 0-1) back to an sRGB hex string."""
    if y <= 0:
        return "#000000"

    z = 1.0 - x - y
    Y = brightness
    X = (Y / y) * x
    Z = (Y / y) * z

    r = X * 1.612 - Y * 0.203 - Z * 0.302
    g = -X * 0.509 + Y * 1.412 + Z * 0.066
    b = X * 0.026 - Y * 0.072 + Z * 0.962

    r, g, b = _srgb(r), _srgb(g), _srgb(b)

    # Desaturate by scaling down if any channel clips above 1.
    peak = max(r, g, b)
    if peak > 1:
        r, g, b = r / peak, g / peak, b / peak

    return rgb_to_hex(*(max(0.0, min(1.0, v)) * 255 for v in (r, g, b)))
