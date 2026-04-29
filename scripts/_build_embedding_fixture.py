"""Build the Stage 15 tiny REMIT corpus fixture used by the notebook + tests.

Plan §1 D18: ``tests/fixtures/embedding/tiny_corpus.parquet`` carries
**8 REMIT-shaped rows** spanning four fuel types (Nuclear, Gas, Coal,
Wind), with three rows whose ``message_description`` is ``None`` so
:func:`bristol_ml.embeddings.synthesise_embeddable_text` is exercised
against the structured-fallback path on the fixture itself.

Run once and commit the output:

    uv run python scripts/_build_embedding_fixture.py

The schema mirrors :data:`bristol_ml.ingestion.remit.OUTPUT_SCHEMA`
exactly — the notebook reads the parquet through pandas and passes
the rows straight to the embedder, so a drift between this fixture
and the upstream schema is a regression worth catching at the next
plan-edit conversation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "tests" / "fixtures" / "embedding" / "tiny_corpus.parquet"


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


# Eight rows.  Five carry a non-NULL ``message_description``; three
# carry ``None`` and exercise the structural-fallback path in
# :func:`bristol_ml.embeddings.synthesise_embeddable_text`.  Fuel-type
# spread (Nuclear x2, Gas x2, Coal x2, Wind x2) keeps the 2D-projection
# scatter readable even at this small size.
_RECORDS: list[dict[str, Any]] = [
    {
        "mrid": "M-A",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 1, 1, 9),
        "effective_from": _utc(2024, 1, 15),
        "effective_to": _utc(2024, 1, 20),
        "retrieved_at_utc": _utc(2024, 1, 1, 9),
        "affected_unit": "T_HARTLEPOOL-1",
        "asset_id": "T_HARTLEPOOL-1",
        "fuel_type": "Nuclear",
        "affected_mw": 600.0,
        "normal_capacity_mw": 1180.0,
        "event_type": "Outage",
        "cause": "Planned",
        "message_description": (
            "Planned nuclear outage at Hartlepool-1 for refuelling and statutory inspection."
        ),
    },
    {
        "mrid": "M-AA",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 1, 5, 9),
        "effective_from": _utc(2024, 2, 1),
        "effective_to": _utc(2024, 2, 28),
        "retrieved_at_utc": _utc(2024, 1, 5, 9),
        "affected_unit": "T_HEYSHAM2-2",
        "asset_id": "T_HEYSHAM2-2",
        "fuel_type": "Nuclear",
        "affected_mw": 615.0,
        "normal_capacity_mw": 1255.0,
        "event_type": "Outage",
        "cause": "Planned",
        # NULL message — exercises the structural-fallback path.
        "message_description": None,
    },
    {
        "mrid": "M-B",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 2, 1, 10),
        "effective_from": _utc(2024, 2, 10),
        "effective_to": _utc(2024, 2, 15),
        "retrieved_at_utc": _utc(2024, 2, 1, 10),
        "affected_unit": "T_PEMBROKE-1",
        "asset_id": "T_PEMBROKE-1",
        "fuel_type": "Gas",
        "affected_mw": 400.0,
        "normal_capacity_mw": 540.0,
        "event_type": "Outage",
        "cause": "Unplanned",
        "message_description": "Gas unit Pembroke-1 forced outage following compressor trip.",
    },
    {
        "mrid": "M-BB",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 2, 5, 10),
        "effective_from": _utc(2024, 2, 12),
        "effective_to": _utc(2024, 2, 14),
        "retrieved_at_utc": _utc(2024, 2, 5, 10),
        "affected_unit": "T_GRAIN-7",
        "asset_id": "T_GRAIN-7",
        "fuel_type": "Gas",
        "affected_mw": 380.0,
        "normal_capacity_mw": 540.0,
        "event_type": "Restriction",
        "cause": "Planned",
        "message_description": (
            "Capacity restriction on Grain-7 during scheduled maintenance window."
        ),
    },
    {
        "mrid": "M-C",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 3, 1, 8),
        "effective_from": _utc(2024, 3, 5),
        "effective_to": _utc(2024, 3, 8),
        "retrieved_at_utc": _utc(2024, 3, 1, 8),
        "affected_unit": "T_RATCLIFFE-1",
        "asset_id": "T_RATCLIFFE-1",
        "fuel_type": "Coal",
        "affected_mw": 500.0,
        "normal_capacity_mw": 500.0,
        "event_type": "Outage",
        "cause": "Planned",
        # NULL — structural fallback synthesises text from event/cause/fuel/unit.
        "message_description": None,
    },
    {
        "mrid": "M-CC",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 3, 4, 8),
        "effective_from": _utc(2024, 3, 10),
        "effective_to": _utc(2024, 3, 12),
        "retrieved_at_utc": _utc(2024, 3, 4, 8),
        "affected_unit": "T_RATCLIFFE-2",
        "asset_id": "T_RATCLIFFE-2",
        "fuel_type": "Coal",
        "affected_mw": 480.0,
        "normal_capacity_mw": 500.0,
        "event_type": "Outage",
        "cause": "Unplanned",
        "message_description": "Coal unit Ratcliffe-2 forced outage — boiler tube leak.",
    },
    {
        "mrid": "M-D",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 4, 1, 7),
        "effective_from": _utc(2024, 4, 2),
        "effective_to": None,  # open-ended event
        "retrieved_at_utc": _utc(2024, 4, 1, 7),
        "affected_unit": "T_GORDONS-1",
        "asset_id": "T_GORDONS-1",
        "fuel_type": "Wind",
        "affected_mw": 300.0,
        "normal_capacity_mw": 300.0,
        "event_type": "Outage",
        "cause": "Unplanned",
        # NULL — third structural-fallback exemplar (open-ended).
        "message_description": None,
    },
    {
        "mrid": "M-DD",
        "revision_number": 0,
        "message_type": "Production",
        "message_status": "Active",
        "published_at": _utc(2024, 4, 5, 7),
        "effective_from": _utc(2024, 4, 7),
        "effective_to": _utc(2024, 4, 9),
        "retrieved_at_utc": _utc(2024, 4, 5, 7),
        "affected_unit": "T_BURBO-2",
        "asset_id": "T_BURBO-2",
        "fuel_type": "Wind",
        "affected_mw": 250.0,
        "normal_capacity_mw": 350.0,
        "event_type": "Restriction",
        "cause": "Planned",
        "message_description": "Offshore wind Burbo-2 cable maintenance reduces export capacity.",
    },
]


def _build_frame() -> pd.DataFrame:
    """Return the fixture frame in canonical column order."""
    return pd.DataFrame(_RECORDS)


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write the frame to parquet using the Stage 13 schema."""
    from bristol_ml.ingestion.remit import OUTPUT_SCHEMA

    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


if __name__ == "__main__":
    df = _build_frame()
    _write(df, OUT)
    print(
        f"Wrote {OUT} ({len(df)} rows; "
        f"{df['message_description'].isna().sum()} with NULL message_description)"
    )
