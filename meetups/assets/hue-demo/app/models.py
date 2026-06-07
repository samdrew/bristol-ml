"""API data models, shared by the REST layer and the backends."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LightState(BaseModel):
    """Normalised view of a light, backend-agnostic.

    Brightness is a percentage and colour is a hex string deliberately: these are
    the units an MCP tool (and a human) reason about. Backends translate to/from
    whatever their hardware actually wants.
    """

    on: bool = Field(description="Whether the light is emitting.")
    brightness: int = Field(ge=0, le=100, description="Brightness, 0-100%.")
    color: str = Field(description="Colour as #RRGGBB hex.")
    reachable: bool = Field(default=True, description="Can the backend talk to the light?")
    backend: str = Field(description="Which backend is serving this state.")


class PowerRequest(BaseModel):
    on: bool


class BrightnessRequest(BaseModel):
    level: int = Field(ge=0, le=100, description="Brightness, 0-100%.")


class ColorRequest(BaseModel):
    hex: str = Field(description="Colour as #RRGGBB (or #RGB) hex.")

    @field_validator("hex")
    @classmethod
    def _valid_hex(cls, value: str) -> str:
        from app.color import hex_to_rgb, rgb_to_hex

        try:
            return rgb_to_hex(*hex_to_rgb(value))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
