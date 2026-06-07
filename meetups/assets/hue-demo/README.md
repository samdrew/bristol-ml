# Hue Light REST demo

A small REST service in front of a swappable "light" backend, with a live
coloured-circle visualiser. Built as the first stage of a meetup demo whose
second stage wraps an MCP tool around this REST API.

```
            ┌─────────────┐     HTTP/JSON     ┌──────────────────┐
   (later)  │  MCP tool   │ ────────────────▶ │   REST API       │
            └─────────────┘                   │   (FastAPI)      │
                                              └────────┬─────────┘
   browser ──── GET /api/state (poll) ────────────────┤  swappable backend
   (circle)                                            ├── VirtualBackend (in-memory)
                                                       └── HueBackend (HueBLE / Bluetooth)
```

The webpage is a **view**, not a backend: it renders whatever `GET /api/state`
returns, so it works identically against the mock and the real light.

## Run the mock (no hardware)

```bash
uv run uvicorn app.main:app --reload
```

`uv run` creates the virtual environment and installs dependencies from
`pyproject.toml` on first invocation, so there's no separate install step. Open
http://127.0.0.1:8000 for the visualiser, and http://127.0.0.1:8000/docs for the
OpenAPI explorer.

If you'd rather provision the environment up front, `uv sync` does that.

## Run against a real Hue Go

```bash
uv sync --extra hue
# Put the light in pairing mode first (Hue app -> Settings -> Voice Assistants
# -> Make discoverable, or factory reset). On macOS, accept the OS pairing
# prompt on first connect.
LIGHT_BACKEND=hue uv run uvicorn app.main:app
# Optional: pin a specific light (CoreBluetooth UUID on macOS, MAC on Linux)
LIGHT_BACKEND=hue HUE_ADDRESS=XX:XX:XX:XX:XX:XX uv run uvicorn app.main:app
```

## REST API

| Method | Path              | Body                  | Returns      |
|--------|-------------------|-----------------------|--------------|
| GET    | `/api/state`      | —                     | `LightState` |
| POST   | `/api/power`      | `{"on": bool}`        | `LightState` |
| POST   | `/api/brightness` | `{"level": 0-100}`    | `LightState` |
| POST   | `/api/color`      | `{"hex": "#RRGGBB"}`  | `LightState` |
| GET    | `/api/healthz`    | —                     | status       |

`LightState`: `{on, brightness (0-100), color (#hex), reachable, backend}`.

Every mutation returns the resulting state, so the MCP tool gets immediate
confirmation without a follow-up `GET`.

## Layout

```
app/
  main.py            FastAPI routes + static serving
  models.py          Pydantic request/response models (+ hex validation)
  color.py           sRGB hex <-> CIE xy (Hue gamut)
  backends/
    base.py          LightBackend abstract interface
    virtual.py       in-memory mock
    hue.py           HueBLE-backed real light
    __init__.py      make_backend() factory (reads LIGHT_BACKEND)
  static/index.html  the visualiser
```