"""REST interface in front of a swappable light backend.

Endpoints are intentionally small and well-typed so the MCP tool (next stage)
maps straight onto them, and FastAPI's generated OpenAPI at /docs is the schema
that MCP layer can be built from.

    GET  /api/state        -> LightState
    POST /api/power        {on: bool}
    POST /api/brightness   {level: 0-100}
    POST /api/color        {hex: "#RRGGBB"}
    GET  /api/healthz
    GET  /                 -> the coloured-circle visualiser
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backends import make_backend
from app.models import (
    BrightnessRequest,
    ColorRequest,
    LightState,
    PowerRequest,
)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = make_backend()
    await backend.connect()
    app.state.backend = backend
    try:
        yield
    finally:
        await backend.close()


app = FastAPI(title="Hue Light REST", version="0.1.0", lifespan=lifespan)


@app.get("/api/state", response_model=LightState, tags=["light"])
async def get_state() -> LightState:
    """Return the light's current state."""
    return await app.state.backend.get_state()


@app.post("/api/power", response_model=LightState, tags=["light"])
async def set_power(req: PowerRequest) -> LightState:
    """Turn the light on or off."""
    await app.state.backend.set_power(req.on)
    return await app.state.backend.get_state()


@app.post("/api/brightness", response_model=LightState, tags=["light"])
async def set_brightness(req: BrightnessRequest) -> LightState:
    """Set brightness as a percentage (0-100)."""
    await app.state.backend.set_brightness(req.level)
    return await app.state.backend.get_state()


@app.post("/api/color", response_model=LightState, tags=["light"])
async def set_color(req: ColorRequest) -> LightState:
    """Set the colour from a #RRGGBB hex string."""
    await app.state.backend.set_color(req.hex)
    return await app.state.backend.get_state()


@app.get("/api/healthz", tags=["meta"])
async def healthz() -> dict:
    state = await app.state.backend.get_state()
    return {"ok": state.reachable, "backend": state.backend}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
