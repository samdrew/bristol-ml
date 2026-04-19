"""One-off cassette recorder for Open-Meteo's archive API (London + Bristol, Jan 2023).

Run from the repo root::

    uv run python scripts/record_weather_cassette.py

Records a vcrpy cassette under ``tests/fixtures/weather/cassettes/`` covering
exactly two stations (London + Bristol, geographically distinct, weights span
~15x) over a single month (January 2023) at five variables. The cassette
lives at ``tests/fixtures/weather/cassettes/weather_2023_01.yaml`` and is
replayed by the Stage 2 integration tests under ``--record-mode=none``.

Sensitive headers are filtered even though the archive endpoint is
unauthenticated today — this sets the precedent for future authenticated
feeds (e.g. the commercial ``customer-api.open-meteo.com`` host).

This script is a development utility, not production code, and is
excluded from the wheel build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import vcr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CASSETTE_DIR = ROOT / "tests" / "fixtures" / "weather" / "cassettes"
CASSETTE_PATH = CASSETTE_DIR / "weather_2023_01.yaml"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

STATIONS: list[dict[str, float | str]] = [
    {"name": "london", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "bristol", "latitude": 51.4545, "longitude": -2.5879},
]

VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]

START_DATE = "2023-01-01"
END_DATE = "2023-01-31"


def main() -> int:
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    if CASSETTE_PATH.exists():
        print(f"Cassette already present at {CASSETTE_PATH}; delete to re-record.")
        return 0

    my_vcr = vcr.VCR(
        record_mode="once",
        filter_headers=[
            ("Authorization", "REDACTED"),
            ("Cookie", "REDACTED"),
            ("X-Api-Key", "REDACTED"),
            ("set-cookie", "REDACTED"),
        ],
        decode_compressed_response=True,
    )

    with my_vcr.use_cassette(str(CASSETTE_PATH)), httpx.Client(timeout=30.0) as client:
        for station in STATIONS:
            params = {
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "start_date": START_DATE,
                "end_date": END_DATE,
                "hourly": ",".join(VARIABLES),
                "timezone": "UTC",
            }
            r = client.get(BASE_URL, params=params)
            r.raise_for_status()
            payload = r.json()
            hours = len(payload.get("hourly", {}).get("time", []) or [])
            print(f"{station['name']}: {hours} hourly rows")

    size_kb = CASSETTE_PATH.stat().st_size / 1024
    print(f"Cassette recorded at {CASSETTE_PATH}")
    print(f"Cassette size: {size_kb:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
