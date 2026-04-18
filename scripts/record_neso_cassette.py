"""One-off cassette recorder for the NESO 2023 historic demand resource.

Run from the repo root::

    uv run python scripts/record_neso_cassette.py

Records a vcrpy cassette under ``tests/fixtures/neso/cassettes/`` that
covers exactly two paginator pages at ``page_size=500`` for the 2023
resource. Before persisting the cassette, each response body is
rewritten to keep only the columns used by the ingester
(``SETTLEMENT_DATE``, ``SETTLEMENT_PERIOD``, ``ND``, ``TSD``, ``_id``)
and to cap the ``result.total`` value so replay halts after page 2.

The recorder drives the HTTP calls directly rather than going through
``fetch()`` — ``fetch()`` would paginate to completion (17 520 rows for
2023) at the real server, which blows the ~200 kB cassette budget.

This script is a development utility, not production code, and is
excluded from the wheel build.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx
import vcr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CASSETTE_DIR = ROOT / "tests" / "fixtures" / "neso" / "cassettes"
CASSETTE_PATH = CASSETTE_DIR / "neso_2023_refresh.yaml"

BASE_URL = "https://api.neso.energy/api/3/action/datastore_search"
RESOURCE_ID = "bf5ab335-9b40-4ea4-b93a-ab4af7bce003"

KEEP_COLUMNS = {"_id", "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "TSD"}
PAGE_SIZE = 500
# Two pages x 500 rows = 1000 records total. The second page's response
# will report ``total=1000`` so the paginator in
# ``bristol_ml.ingestion.neso._fetch_year`` stops cleanly after page 2.
FAKE_TOTAL = PAGE_SIZE * 2


def _filter_response(response: dict) -> dict:
    """Strip columns + cap ``total`` in the persisted CKAN response body."""
    body = response.get("body", {})
    string = body.get("string")
    if string is None:
        return response
    text = string.decode("utf-8") if isinstance(string, bytes) else string
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return response
    result = payload.get("result")
    if result and "records" in result:
        records = result["records"][:PAGE_SIZE]
        result["records"] = [{k: v for k, v in rec.items() if k in KEEP_COLUMNS} for rec in records]
        result["total"] = FAKE_TOTAL
        if "fields" in result:
            result["fields"] = [f for f in result["fields"] if f.get("id") in KEEP_COLUMNS]
    new_text = json.dumps(payload, separators=(",", ":"))
    new = copy.deepcopy(response)
    new["body"]["string"] = new_text
    headers = new.get("headers") or {}
    for hname in list(headers.keys()):
        if hname.lower() == "content-length":
            headers[hname] = [str(len(new_text.encode("utf-8")))]
    return new


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
        before_record_response=_filter_response,
        decode_compressed_response=True,
    )

    with my_vcr.use_cassette(str(CASSETTE_PATH)), httpx.Client(timeout=30.0) as client:
        for page, offset in enumerate([0, PAGE_SIZE], start=1):
            r = client.get(
                BASE_URL,
                params={
                    "resource_id": RESOURCE_ID,
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )
            r.raise_for_status()
            body = r.json()
            server_total = body.get("result", {}).get("total")
            returned = len(body.get("result", {}).get("records", []))
            print(
                f"Page {page}: offset={offset} returned {returned} rows "
                f"(server total={server_total})"
            )

    size_kb = CASSETTE_PATH.stat().st_size / 1024
    print(f"Cassette recorded at {CASSETTE_PATH}")
    print(f"Cassette size: {size_kb:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
