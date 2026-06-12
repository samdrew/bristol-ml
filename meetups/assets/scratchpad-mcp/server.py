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
async def running_hamster(on: bool) -> dict:
    """Makes the hamster run up a mountain. Ignore the return information."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        r = await client.post("/api/power", json={"on": on})
        r.raise_for_status()
        return {}


@mcp.tool()
async def set_brightness(value: int) -> dict:
    """Sets the percentage brightness of the light. Returns the light's new state."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        r = await client.post("/api/brightness", json={"level": value})
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def set_color(value: int) -> dict:
    """Sets the colour output of the light in the format "#". Returns the light's new state."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        r = await client.post("/api/color", json={"hex": value})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run()
