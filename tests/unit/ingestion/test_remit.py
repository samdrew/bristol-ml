"""Unit tests for ``bristol_ml.ingestion.remit`` — T2 surface.

Covers Stage 13 Task T2: ``OUTPUT_SCHEMA``, ``as_of``, ``MESSAGE_STATUSES``,
and ``FUEL_TYPES``.  ``fetch`` and ``load`` are ``NotImplementedError`` stubs
at T2 and are not tested here.

Test list is drawn from:

- ``docs/plans/active/13-remit-ingestion.md`` §4 (AC-1 / AC-5 named tests)
  and §6 T2 (six plan-named tests under D18a / D18b).
- ``docs/lld/research/13-remit-ingestion-requirements.md`` AC-1 four cases
  (fresh / revised / withdrawn / open-ended) and AC-5.
- ``docs/plans/active/13-remit-ingestion.md`` §1 D8 / D9 / D10 / D13 for
  the as_of contract; D18a / D18b for the test surface.

All tests are pure in-memory — no disk I/O, no mocks, no cassettes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from bristol_ml.ingestion.remit import (
    FUEL_TYPES,
    MESSAGE_STATUSES,
    OUTPUT_SCHEMA,
    as_of,
)

# ---------------------------------------------------------------------------
# Canonical "now" anchor used across all tests (UTC-aware, readable value).
# ---------------------------------------------------------------------------

T = pd.Timestamp("2024-06-15T12:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Small helper — builds a one-row dict with sensible defaults for the nine
# columns not under test, then promotes a list of those dicts to a DataFrame.
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "message_type": "Production",
    "message_status": "Active",
    "effective_from": T - pd.Timedelta(hours=2),
    "effective_to": T + pd.Timedelta(hours=10),
    "retrieved_at_utc": T - pd.Timedelta(minutes=5),
    "affected_unit": "WBURB-1",
    "asset_id": "T_WBURB-1",
    "fuel_type": "Gas",
    "affected_mw": 100.0,
    "normal_capacity_mw": 500.0,
    "event_type": "Outage",
    "cause": "Planned",
    "message_description": "Planned maintenance.",
}


def _make_row(**kwargs: Any) -> dict[str, Any]:
    """Return a single-row dict with caller overrides applied on top of defaults."""
    row = dict(_DEFAULTS)
    row.update(kwargs)
    return row


def _make_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts, matching OUTPUT_SCHEMA column order."""
    df = pd.DataFrame(rows)
    # Ensure the four temporal columns are tz-aware UTC so the published_at
    # comparison in as_of does not raise a tz-naive comparison error.
    for col in ("published_at", "effective_from", "effective_to", "retrieved_at_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


# ---------------------------------------------------------------------------
# Test 1 — AC-1(a): fresh message (single mRID, single revision)
# ---------------------------------------------------------------------------


def test_as_of_fresh_message_returns_active_state() -> None:
    """Pins D8 + D13 + AC-1(a): a single un-revised message published before t is returned.

    A single mRID with revision_number=0 published at t-1h is the simplest
    possible REMIT event.  as_of(df, t) must return exactly that one row
    with the message_status "Active".
    """
    df = _make_df(
        [
            _make_row(
                mrid="MSG-001",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=1),
                message_status="Active",
                affected_mw=200.0,
            )
        ]
    )

    result = as_of(df, T)

    assert len(result) == 1, f"Expected 1 active row; got {len(result)}."
    assert result.iloc[0]["mrid"] == "MSG-001"
    assert result.iloc[0]["revision_number"] == 0
    assert result.iloc[0]["affected_mw"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Test 2 — AC-1(b): revised message — latest revision known at t is returned
# ---------------------------------------------------------------------------


def test_as_of_revised_message_returns_latest_revision() -> None:
    """Pins D9 + D13 + AC-1(b): as_of returns the max revision_number known by t.

    Three revisions (0 / 1 / 2) for the same mRID are published at
    t-3h / t-2h / t-1h.  The affected_mw differs per revision so any
    assertion on that column is meaningful (it disambiguates which row
    was actually selected).

    - as_of(df, t - 1.5h) sees rev 0 (t-3h ≤ t-1.5h) and rev 1 (t-2h ≤ t-1.5h),
      but NOT rev 2 (t-1h > t-1.5h) → returns rev 1.
    - as_of(df, t) sees all three (t-3h, t-2h, t-1h all ≤ t) → returns rev 2.

    Note: the plan §4 prose says "as_of(df, t-2.5h) returns rev 1" but the
    published times are t-3h/t-2h/t-1h.  t-2h > t-2.5h so rev 1 is not yet
    visible at t-2.5h.  The correct midpoint is t-1.5h (between t-2h and
    t-1h), which makes both rev 0 and rev 1 visible while rev 2 is not.
    """
    df = _make_df(
        [
            _make_row(
                mrid="MSG-002",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=3),
                affected_mw=100.0,
            ),
            _make_row(
                mrid="MSG-002",
                revision_number=1,
                published_at=T - pd.Timedelta(hours=2),
                affected_mw=200.0,
            ),
            _make_row(
                mrid="MSG-002",
                revision_number=2,
                published_at=T - pd.Timedelta(hours=1),
                affected_mw=300.0,
            ),
        ]
    )

    # At t - 1.5h revisions 0 and 1 are known (published at t-3h and t-2h ≤ t-1.5h);
    # rev 2 (published at t-1h) has not yet been disclosed.
    result_mid = as_of(df, T - pd.Timedelta(hours=1.5))
    assert len(result_mid) == 1, f"Expected 1 row at t-1.5h; got {len(result_mid)}."
    assert result_mid.iloc[0]["revision_number"] == 1, (
        f"Expected revision 1 at t-1.5h; got {result_mid.iloc[0]['revision_number']}."
    )
    assert result_mid.iloc[0]["affected_mw"] == pytest.approx(200.0)

    # At t all three revisions are visible; rev 2 must win.
    result_now = as_of(df, T)
    assert len(result_now) == 1, f"Expected 1 row at t; got {len(result_now)}."
    assert result_now.iloc[0]["revision_number"] == 2, (
        f"Expected revision 2 at t; got {result_now.iloc[0]['revision_number']}."
    )
    assert result_now.iloc[0]["affected_mw"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Test 3 — AC-1(c): withdrawn message is excluded from the active-state frame
# ---------------------------------------------------------------------------


def test_as_of_withdrawn_message_excludes_row() -> None:
    """Pins D13 + AC-1(c): a withdrawn latest revision removes the mRID from the result.

    Rev 0 is published at t-2h with status "Active".
    Rev 1 is published at t-1h with status "Withdrawn".

    - as_of(df, t) — both revisions visible; rev 1 is latest and is "Withdrawn"
      → the mRID must be absent from the result (empty frame).
    - as_of(df, t - 1.5h) — only rev 0 is visible (published at t-2h ≤ t-1.5h;
      rev 1 at t-1h is not yet known) → the mRID must appear as "Active".
    """
    df = _make_df(
        [
            _make_row(
                mrid="MSG-003",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=2),
                message_status="Active",
                affected_mw=150.0,
            ),
            _make_row(
                mrid="MSG-003",
                revision_number=1,
                published_at=T - pd.Timedelta(hours=1),
                message_status="Withdrawn",
                affected_mw=0.0,
            ),
        ]
    )

    # At t the withdrawal is known → mRID must be absent.
    result_after = as_of(df, T)
    assert len(result_after) == 0, (
        f"Withdrawn mRID must not appear in the active-state frame at t; "
        f"got {len(result_after)} row(s)."
    )

    # At t - 1.5h only rev 0 is visible → mRID must appear as Active.
    result_before = as_of(df, T - pd.Timedelta(hours=1.5))
    assert len(result_before) == 1, (
        f"Before the withdrawal rev 0 must appear; got {len(result_before)} row(s)."
    )
    assert result_before.iloc[0]["revision_number"] == 0
    assert result_before.iloc[0]["message_status"] == "Active"
    assert result_before.iloc[0]["affected_mw"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Test 4 — AC-5 / D10: open-ended effective_to (NaT) is treated as active
# ---------------------------------------------------------------------------


def test_as_of_open_ended_effective_to_treated_as_active() -> None:
    """Pins D10 + AC-5: effective_to=NaT (open-ended) does not suppress the row.

    The as_of function is a transaction-time filter (published_at ≤ t) —
    effective_to is NOT part of that filter.  A row with effective_to=NaT
    published before t must appear in the result regardless of its null
    effective_to value.  This pins the design note in the as_of docstring.
    """
    df = _make_df(
        [
            _make_row(
                mrid="MSG-004",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=1),
                effective_from=T - pd.Timedelta(minutes=30),
                effective_to=pd.NaT,  # open-ended — no declared end
                message_status="Active",
                affected_mw=75.0,
            )
        ]
    )

    result = as_of(df, T)

    assert len(result) == 1, (
        f"Open-ended (effective_to=NaT) row must appear in as_of result; "
        f"got {len(result)} row(s)."
    )
    assert result.iloc[0]["mrid"] == "MSG-004"
    assert pd.isna(result.iloc[0]["effective_to"]), (
        "effective_to must remain NaT in the returned frame."
    )


# ---------------------------------------------------------------------------
# Test 5 — NFR-7 / D8: naive timestamp raises ValueError
# ---------------------------------------------------------------------------


def test_as_of_raises_on_naive_timestamp() -> None:
    """Pins D8 + NFR-7: as_of(df, naive_t) raises ValueError mentioning 'naive' or 'timezone'.

    REMIT bi-temporal queries are meaningless without a timezone reference.
    The as_of docstring documents that a naive timestamp raises ValueError.
    The error message must contain the word "naive" or "timezone" so the
    caller understands what went wrong.
    """
    df = _make_df(
        [
            _make_row(
                mrid="MSG-005",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=1),
            )
        ]
    )

    naive_t = pd.Timestamp("2024-06-15T12:00:00")  # no tz — intentionally naive
    assert naive_t.tzinfo is None, "Sanity check: timestamp must be naive for this test."

    with pytest.raises(ValueError) as exc_info:
        as_of(df, naive_t)

    error_text = str(exc_info.value).lower()
    assert "naive" in error_text or "timezone" in error_text, (
        f"ValueError must mention 'naive' or 'timezone'; got: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# Test 6 — AC-5 / D18b: OUTPUT_SCHEMA column count, types, and nullability
# ---------------------------------------------------------------------------


def test_output_schema_columns_and_types_pinned() -> None:
    """Pins D8 + D10 + AC-5 + D18b: OUTPUT_SCHEMA has exactly 16 fields with correct types.

    Asserts:
    - Exactly 16 fields (the four-axis layout documented in the plan §5).
    - The four bi-temporal timestamp columns are timestamp[us, tz=UTC].
    - effective_to is nullable (open-ended events — D10).
    - published_at, effective_from, and retrieved_at_utc are non-nullable.
    - The identifier axis fields (mrid, revision_number, message_type,
      message_status) are non-nullable.
    """
    assert len(OUTPUT_SCHEMA) == 16, (
        f"OUTPUT_SCHEMA must have exactly 16 fields; got {len(OUTPUT_SCHEMA)}."
    )

    utc_timestamp_type = pa.timestamp("us", tz="UTC")

    # --- Bi-temporal axis: check all four columns exist and have correct type ---
    temporal_columns = {
        "published_at": {"type": utc_timestamp_type, "nullable": False},
        "effective_from": {"type": utc_timestamp_type, "nullable": False},
        "effective_to": {"type": utc_timestamp_type, "nullable": True},   # D10 load-bearing
        "retrieved_at_utc": {"type": utc_timestamp_type, "nullable": False},
    }

    schema_names = OUTPUT_SCHEMA.names
    for col, spec in temporal_columns.items():
        assert col in schema_names, f"Temporal column {col!r} missing from OUTPUT_SCHEMA."
        field = OUTPUT_SCHEMA.field(col)
        assert field.type == spec["type"], (
            f"Column {col!r}: expected type {spec['type']}; got {field.type}."
        )
        assert field.nullable == spec["nullable"], (
            f"Column {col!r}: expected nullable={spec['nullable']}; got {field.nullable}."
        )

    # --- Identifier axis: non-nullable ---
    non_nullable_identifiers = ("mrid", "revision_number", "message_type", "message_status")
    for col in non_nullable_identifiers:
        assert col in schema_names, f"Identifier column {col!r} missing from OUTPUT_SCHEMA."
        field = OUTPUT_SCHEMA.field(col)
        assert not field.nullable, (
            f"Identifier column {col!r} must be non-nullable; got nullable=True."
        )

    # --- Spot-check a few nullable columns (asset / capacity axis) ---
    nullable_columns = ("affected_unit", "asset_id", "fuel_type", "affected_mw",
                        "normal_capacity_mw", "event_type", "cause", "message_description")
    for col in nullable_columns:
        assert col in schema_names, f"Column {col!r} missing from OUTPUT_SCHEMA."
        field = OUTPUT_SCHEMA.field(col)
        assert field.nullable, (
            f"Column {col!r} must be nullable; got nullable=False."
        )


# ---------------------------------------------------------------------------
# Smoke tests on MODULE_STATUSES and FUEL_TYPES constants
# ---------------------------------------------------------------------------


def test_message_statuses_includes_withdrawn() -> None:
    """Pins D13: MESSAGE_STATUSES must include 'Withdrawn' — used by as_of step 3."""
    assert "Withdrawn" in MESSAGE_STATUSES, (
        "'Withdrawn' must be in MESSAGE_STATUSES; as_of step 3 filters on this value."
    )
    assert "Active" in MESSAGE_STATUSES, (
        "'Active' must be in MESSAGE_STATUSES; the primary live-event status."
    )


def test_fuel_types_is_non_empty_closed_set() -> None:
    """Pins D13 / stub semantics: FUEL_TYPES is a non-empty tuple of strings."""
    assert isinstance(FUEL_TYPES, tuple), "FUEL_TYPES must be a tuple."
    assert len(FUEL_TYPES) > 0, "FUEL_TYPES must be non-empty."
    for item in FUEL_TYPES:
        assert isinstance(item, str), f"Every FUEL_TYPES entry must be a str; got {type(item)}."
    # Nuclear is the pedagogically prominent fuel type named in the intent §Demo moment.
    assert "Nuclear" in FUEL_TYPES, "'Nuclear' must be in FUEL_TYPES (intent §Demo moment)."
