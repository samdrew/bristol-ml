"""One-off cassette recorder for the gov.uk bank-holidays feed.

Run from the repo root::

    uv run python scripts/record_holidays_cassette.py

Records a vcrpy cassette under ``tests/fixtures/holidays/cassettes/``
covering one real GET of ``https://www.gov.uk/bank-holidays.json`` —
the feed is a single small JSON object (< 30 kB) carrying events for
all three UK divisions, so no pagination and no body-trimming is
required.

Sensitive headers are filtered even though gov.uk's bank-holidays
endpoint is unauthenticated today — this sets the precedent for future
authenticated feeds and matches the convention in
``scripts/record_neso_cassette.py`` / ``scripts/record_weather_cassette.py``.

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

CASSETTE_DIR = ROOT / "tests" / "fixtures" / "holidays" / "cassettes"
CASSETTE_PATH = CASSETTE_DIR / "holidays_refresh.yaml"

URL = "https://www.gov.uk/bank-holidays.json"


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
        r = client.get(URL)
        r.raise_for_status()
        payload = r.json()
        totals = {division: len(block.get("events", [])) for division, block in payload.items()}
        print(f"Divisions: {sorted(payload.keys())}")
        for division, n in totals.items():
            print(f"  {division}: {n} events")

    size_kb = CASSETTE_PATH.stat().st_size / 1024
    print(f"Cassette recorded at {CASSETTE_PATH}")
    print(f"Cassette size: {size_kb:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
