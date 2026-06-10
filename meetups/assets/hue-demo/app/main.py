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

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.backends import make_backend
from app.models import (
    BrightnessRequest,
    ColorRequest,
    LightState,
    PowerRequest,
)

STATIC_DIR = Path(__file__).parent / "static"

# Paths whose access logs are pure noise on a polling demo: the visualiser hits
# /api/state several times a second, and browsers probe for icons that don't exist.
_QUIET_PATHS = ("/api/state", "/favicon.ico", "/apple-touch-icon")

# Mutating endpoints we *do* want to see in detail, so the request -> response is
# legible on the CLI during a live demo.
_VERBOSE_PATHS = ("/api/power", "/api/brightness", "/api/color")

logger = logging.getLogger("hue")


class _QuietAccessFilter(logging.Filter):
    """Drop uvicorn access-log records for high-frequency, low-interest paths."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access formats with args = (client, method, path, http_version, status)
        if isinstance(record.args, tuple) and len(record.args) >= 3:
            path = str(record.args[2])
            return not path.startswith(_QUIET_PATHS)
        return True


def _configure_logging() -> None:
    """Quieten polling noise and route our verbose logs through uvicorn's handler."""
    logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())

    # Reuse uvicorn's *default* handler (plain "INFO:" formatter) so our lines match
    # the house style. The access handler is no good here: its formatter expects the
    # 5-tuple access record args and would crash on our messages.
    default = logging.getLogger("uvicorn")
    if default.handlers and not logger.handlers:
        logger.handlers = default.handlers
        logger.setLevel(logging.INFO)
        logger.propagate = False


_configure_logging()


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


class VerboseRequestLogging(BaseHTTPMiddleware):
    """Log the full request body and response for the mutating endpoints."""

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith(_VERBOSE_PATHS):
            return await call_next(request)

        body = (await request.body()).decode("utf-8", "replace").strip()
        logger.info(
            "→ %s %s from %s  %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "?",
            body or "(no body)",
        )

        response = await call_next(request)

        # BaseHTTPMiddleware hands back a streaming response; drain it so we can both
        # log the payload and replay it to the client.
        chunks = [chunk async for chunk in response.body_iterator]  # type: ignore[attr-defined]
        payload = b"".join(chunks)
        logger.info(
            "← %s %s  %s",
            response.status_code,
            request.url.path,
            payload.decode("utf-8", "replace").strip(),
        )
        # body_iterator is now exhausted; hand the buffered payload back to the client.
        return Response(
            content=payload,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


app.add_middleware(VerboseRequestLogging)


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
