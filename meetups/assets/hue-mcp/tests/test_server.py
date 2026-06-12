"""Tests against a mocked backend, using FastMCP's in-memory client.

No network and no real light required: httpx.MockTransport stands in for the
REST API, and Client(mcp) talks to the server in-process.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp import Client

from hue_mcp import server


@pytest.fixture
def backend_state() -> dict:
    return {
        "on": True,
        "brightness": 80,
        "color": "#ffd6aa",
        "reachable": True,
        "backend": "virtual",
    }


@pytest.fixture(autouse=True)
def mock_backend(monkeypatch, backend_state):
    """Swap the client factory for one backed by an in-memory fake API."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/api/state":
            return httpx.Response(200, json=backend_state)
        if request.method == "GET" and path == "/api/healthz":
            return httpx.Response(200, json={"status": "ok"})

        body = json.loads(request.content or b"{}")
        if path == "/api/power":
            backend_state["on"] = body["on"]
        elif path == "/api/brightness":
            if not 0 <= body["level"] <= 100:
                return httpx.Response(422, json={"detail": "out of range"})
            backend_state["brightness"] = body["level"]
        elif path == "/api/color":
            backend_state["color"] = body["hex"]
        else:
            return httpx.Response(404)
        return httpx.Response(200, json=backend_state)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        server,
        "build_client",
        lambda: httpx.AsyncClient(base_url=server.API_BASE, timeout=5.0, transport=transport),
    )


async def test_get_state():
    async with Client(server.mcp) as c:
        res = await c.call_tool("get_state", {})
    assert res.data.on is True
    assert res.data.brightness == 80
    assert res.data.backend == "virtual"


async def test_set_light_combined():
    async with Client(server.mcp) as c:
        res = await c.call_tool("set_light", {"brightness": 20, "color": "#ff8800"})
    assert res.data.brightness == 20
    assert res.data.color == "#ff8800"


async def test_set_light_power_off():
    async with Client(server.mcp) as c:
        res = await c.call_tool("set_light", {"on": False})
    assert res.data.on is False


async def test_set_light_requires_an_argument():
    async with Client(server.mcp) as c:
        with pytest.raises(Exception):
            await c.call_tool("set_light", {})


async def test_bad_colour_rejected_before_call():
    async with Client(server.mcp) as c:
        with pytest.raises(Exception):
            await c.call_tool("set_light", {"color": "not-a-hex"})


async def test_health():
    async with Client(server.mcp) as c:
        res = await c.call_tool("health", {})
    assert res.data["status"] == "ok"
