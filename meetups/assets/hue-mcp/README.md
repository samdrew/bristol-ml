# hue-mcp

A minimal [MCP](https://modelcontextprotocol.io) server, built with
[FastMCP](https://gofastmcp.com), that wraps the **Hue Light REST API**
(`/api/state`, `/api/power`, `/api/brightness`, `/api/color`, `/api/healthz`).

It is a teaching demo: the code favours clarity and explains its design choices
in comments.

## Design in one line

The MCP surface is **not** a one-to-one mirror of the REST API. The four
mutating endpoints collapse into a single intent-shaped `set_light` tool,
because an LLM expresses intent ("dim it and make it warm"), not HTTP verbs.

| MCP component | Kind | Maps to |
|---|---|---|
| `get_state` | tool (read-only) | `GET /api/state` |
| `light://state` | resource | `GET /api/state` |
| `set_light(on?, brightness?, color?)` | tool (write) | `POST /api/power` + `/brightness` + `/color` |
| `health` | tool (read-only) | `GET /api/healthz` |

## Quick start

```bash
uv sync                 # create .venv and install
cp .env.example .env    # point HUE_API_BASE at your backend if not localhost:8000

# Run over stdio (what Claude Desktop and most local clients expect):
uv run hue-mcp

# Or run as an HTTP server:
uv run fastmcp run hue_mcp.server:mcp --transport http
```

Development with hot reload:

```bash
uv run fastmcp dev hue_mcp.server:mcp
```

## Configuration

Read from the environment (never passed as tool arguments):

| Variable | Default | Meaning |
|---|---|---|
| `HUE_API_BASE` | `http://127.0.0.1:8000` | Base URL of the REST API |
| `HUE_API_TIMEOUT` | `5.0` | Per-request timeout, seconds |

## Tests

```bash
uv run pytest
```

The suite uses `httpx.MockTransport` as a fake backend and FastMCP's in-memory
`Client`, so it needs neither the network nor a real light.

## Notes worth keeping

- **`reachable` is checked on writes.** A `200 OK` means the *service* answered,
  not that the *device* did. `set_light` raises if the backend reports the light
  unreachable, rather than reporting a phantom success.
- **Read-after-write.** `set_light` finishes with a `GET /api/state` and returns
  what the device actually reports, not what was sent.
- **Brightness is 0–100**, stated in the tool description and enforced by the
  schema, because a model's prior is 0–100 and it will guess accordingly.
- **Atomic tools** (`set_power`, `set_brightness`, `set_color`) are intentionally
  omitted to keep the surface small; adding them is a few lines each if a client
  benefits from finer-grained, individually annotated operations.
