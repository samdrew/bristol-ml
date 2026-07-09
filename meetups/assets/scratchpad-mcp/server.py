# server.py
"""Scratch MCP server — a demo MCP interface over the hue-demo REST API.

The hue-demo app (../hue-demo) serves a small REST API on localhost:8000. This
server wraps those endpoints as MCP tools, so an MCP client can drive the light
through natural language. Start with the health check; add light controls next.
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scratch")

BASE_URL = os.environ.get("HUE_DEMO_URL", "http://localhost:8000")


@mcp.tool()
async def healthz() -> dict:
    """Check whether the hue-demo backend is up and reachable."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        r = await client.get("/api/healthz")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def set_power(on: bool) -> dict:
    """Turn the light on or off. Returns the light's new state."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        r = await client.post("/api/power", json={"on": on})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run()
