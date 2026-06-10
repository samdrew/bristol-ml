"""A minimal MCP server over the Hue Light REST API.

Design note
-----------
The REST backend exposes four write endpoints (/api/power, /api/brightness,
/api/color) plus /api/state and /api/healthz. We deliberately do *not* mirror
those one-to-one. The MCP surface is an LLM-facing API: it is shaped around the
intents an agent expresses ("dim it and make it warm"), not around the HTTP
verbs that happen to implement them. So the four mutating endpoints collapse
into a single ``set_light`` tool, and reads are offered as both a tool (for
reach) and a resource (for correctness).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

# --- Configuration (never hard-code; never accept as a tool argument) --------
API_BASE = os.environ.get("HUE_API_BASE", "http://127.0.0.1:8000")
TIMEOUT = float(os.environ.get("HUE_API_TIMEOUT", "5.0"))


# --- Types -------------------------------------------------------------------
class LightState(BaseModel):
    """Normalised view of the light. Mirrors the REST API's LightState so the
    return type annotation drives FastMCP's generated output schema."""

    on: bool
    brightness: int = Field(ge=0, le=100, description="Brightness percentage.")
    color: str = Field(description="Colour as #RRGGBB hex.")
    reachable: bool = Field(default=True, description="Backend can talk to the light.")
    backend: str = Field(description="Which backend served this state.")


# Constrained parameter types. FastMCP turns these into JSON Schema that the
# model sees, so they both validate input and *inform* the model of the rules.
Brightness = Annotated[
    int, Field(ge=0, le=100, description="Brightness percentage, 0-100 (not 0-99).")
]
HexColour = Annotated[
    str,
    Field(
        pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$",
        description="Colour as #RRGGBB or #RGB hex, e.g. '#ff8800' for warm orange.",
    ),
]


# --- HTTP client lifecycle ---------------------------------------------------
def build_client() -> httpx.AsyncClient:
    """Factory for the shared client. Pulled out so tests can swap in an
    httpx.MockTransport without touching the rest of the code."""
    return httpx.AsyncClient(base_url=API_BASE, timeout=TIMEOUT)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Create one AsyncClient for the server's lifetime rather than per call,
    avoiding a fresh TCP/TLS handshake on every tool invocation."""
    async with build_client() as client:
        yield {"client": client}


mcp = FastMCP(
    name="Hue Light",
    instructions=(
        "Control a single smart light through a REST backend. Brightness is an "
        "integer percentage 0-100; colour is a #RRGGBB hex string. Call "
        "get_state to read the authoritative state before assuming what the "
        "light is doing."
    ),
    lifespan=lifespan,
)


def _client(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_context["client"]


async def _request(
    ctx: Context,
    method: str,
    path: str,
    json: dict[str, Any] | None = None,
    *,
    require_reachable: bool = True,
) -> LightState:
    """Make one request and return validated state.

    Two failure modes are kept distinct on purpose:
      * transport / HTTP errors  -> we never reached or were rejected;
      * reachable == false       -> the call succeeded but the *device* didn't,
                                     so a 200 does not mean the change took.
    Both are surfaced as ToolError with an actionable message instead of being
    swallowed into a success-looking return value.
    """
    try:
        resp = await _client(ctx).request(method, path, json=json)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = f" - {e.response.json()}"
        except Exception:
            pass
        raise ToolError(
            f"Light service returned HTTP {e.response.status_code}{detail}"
        ) from e
    except httpx.HTTPError as e:
        raise ToolError(
            f"Couldn't reach the light service at {API_BASE}: {e}"
        ) from e

    state = LightState.model_validate(resp.json())
    if require_reachable and not state.reachable:
        raise ToolError(
            "The service responded but reports the light is unreachable; "
            "the change may not have taken effect."
        )
    return state


# --- Tools -------------------------------------------------------------------
@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True, idempotentHint=True, openWorldHint=True
    )
)
async def get_state(ctx: Context) -> LightState:
    """Return the light's current power, brightness, colour and reachability."""
    # require_reachable=False: an unreachable light is still useful to report.
    return await _request(ctx, "GET", "/api/state", require_reachable=False)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,  # setting on-when-on (etc.) is a no-op; safe to retry
        destructiveHint=False,
        openWorldHint=True,  # drives a physical device beyond this server
    )
)
async def set_light(
    ctx: Context,
    on: bool | None = None,
    brightness: Brightness | None = None,
    color: HexColour | None = None,
) -> LightState:
    """Set any combination of power, brightness and colour in one call.

    Omitted fields are left unchanged. Use this for requests like
    "dim to 20% and make it warm orange" or "turn the light off".
    Returns the resulting state as reported by the device.
    """
    if on is None and brightness is None and color is None:
        raise ToolError("Specify at least one of: on, brightness, color.")

    # Order matters: turn on first so brightness/colour apply to a lit light,
    # and turn off last so we don't visibly flicker settings on the way out.
    if on is True:
        await _request(ctx, "POST", "/api/power", {"on": True})
    if brightness is not None:
        await _request(ctx, "POST", "/api/brightness", {"level": brightness})
    if color is not None:
        await _request(ctx, "POST", "/api/color", {"hex": color})
    if on is False:
        await _request(ctx, "POST", "/api/power", {"on": False})

    # Read-after-write: report what the device actually shows now, which can
    # differ from what we sent.
    return await _request(ctx, "GET", "/api/state")


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True, idempotentHint=True, openWorldHint=True
    )
)
async def health(ctx: Context) -> dict[str, Any]:
    """Report whether the light service itself is up (independent of the light)."""
    try:
        resp = await _client(ctx).get("/api/healthz")
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise ToolError(f"Health check failed: {e}") from e
    return resp.json()


# --- Resource ----------------------------------------------------------------
# Semantically a GET belongs as a resource (read context, no side effects).
# We expose it both ways because client support for resources is uneven and
# many agent runtimes consume only tools.
@mcp.resource("light://state")
async def state_resource(ctx: Context) -> LightState:
    """Current light state."""
    return await _request(ctx, "GET", "/api/state", require_reachable=False)


def main() -> None:
    """Console-script entry point. Defaults to stdio transport.

    For an HTTP server instead, run:  fastmcp run hue_mcp.server:mcp --transport http
    """
    mcp.run()


if __name__ == "__main__":
    main()
