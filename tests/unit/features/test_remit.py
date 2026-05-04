"""Spec-derived tests for ``bristol_ml.features.remit`` — Stage 16 T2 surface.

Tests pin the acceptance criteria from
``docs/plans/active/16-model-with-remit.md`` §4 (AC-1, AC-7, AC-8, AC-9)
and §6 T2 (six required tests: leakage, schema conformance, zero-handling,
forward-looking, no-NaN invariant, multi-revision Withdrawn-truncates-prior).

Nine tests total (six required + three implicit):

1. ``test_bitemporal_leakage_rev1_not_visible_before_publication``  — AC-1
2. ``test_schema_conformance_column_names_dtypes_and_constant``     — AC-9 + schema
3. ``test_zero_event_hour_handling``                                 — AC-7
4. ``test_forward_looking_column_semantics``                         — AC-6
5. ``test_no_nan_in_remit_columns_with_mixed_nulls``                — AC-7 cont.
6. ``test_withdrawn_revision_truncates_prior_active_revision``       — multi-revision
7. ``test_cli_main_standalone_exits_zero_and_prints_columns``        — AC-8
8. ``test_public_surface_importable``                                — AC-9
9. ``test_validation_guards_raise_valueerror``                       — invariant

Approach
--------
Mirrors the programmatic-fixture pattern from ``test_assembler_calendar.py``:
all data is built in-memory from primitive dicts, then assembled into
DataFrames; no filesystem reads (except the CLI / stub-mode tests that invoke
the REMIT ingestion stub via ``BRISTOL_ML_REMIT_STUB=1``).

All timestamps are tz-aware UTC; naive timestamps are only constructed
explicitly for the validation-guard test.

UTC discipline: the module requires a tz-aware UTC ``hourly_index``; all test
grids use ``pd.date_range(... tz='UTC')``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from bristol_ml.features.remit import REMIT_VARIABLE_COLUMNS, derive_remit_features

# ---------------------------------------------------------------------------
# Canonical "now" anchor — mirrors the pattern in tests/unit/ingestion/test_remit.py
# ---------------------------------------------------------------------------

T = pd.Timestamp("2024-06-01T12:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Helper: build a synthetic REMIT DataFrame from a list of row dicts.
#
# The Row builder mirrors the approach in test_remit.py (ingestion layer):
# a default-dict is used for all columns not under test, so callers only
# override what they need to control.  The REMIT ingestion OUTPUT_SCHEMA
# (16 columns) is used as the column set; ``derive_remit_features`` needs a
# subset of those columns (``_REQUIRED_COLUMNS`` in remit.py).
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "message_type": "Production",
    "message_status": "Active",
    "effective_from": T - pd.Timedelta(hours=2),
    "effective_to": T + pd.Timedelta(hours=10),
    "retrieved_at_utc": T - pd.Timedelta(minutes=5),
    "affected_unit": "T_UNIT-1",
    "asset_id": "T_UNIT-1",
    "fuel_type": "Gas",
    "affected_mw": 100.0,
    "normal_capacity_mw": 500.0,
    "event_type": "Outage",
    "cause": "Planned",
    "message_description": "Test event.",
}


def _make_row(**kwargs: Any) -> dict[str, Any]:
    """Return a single-row dict with caller overrides applied on top of defaults."""
    row = dict(_DEFAULTS)
    row.update(kwargs)
    return row


def _make_remit_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a REMIT-schema DataFrame from a list of row dicts.

    Ensures all four temporal columns are tz-aware UTC so the
    ``derive_remit_features`` validation does not reject them as naive.
    """
    df = pd.DataFrame(rows)
    for col in ("published_at", "effective_from", "effective_to", "retrieved_at_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _empty_remit_df() -> pd.DataFrame:
    """Return a zero-row REMIT DataFrame with the required column structure."""
    cols = [
        "mrid",
        "revision_number",
        "message_type",
        "message_status",
        "published_at",
        "effective_from",
        "effective_to",
        "retrieved_at_utc",
        "affected_unit",
        "asset_id",
        "fuel_type",
        "affected_mw",
        "normal_capacity_mw",
        "event_type",
        "cause",
        "message_description",
    ]
    df = pd.DataFrame(columns=cols)
    for col in ("published_at", "effective_from", "effective_to", "retrieved_at_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


# ---------------------------------------------------------------------------
# Test 1 — Bi-temporal correctness / leakage test (AC-1)
#
# Construct a two-revision REMIT log for the same mrid:
#   - rev 0 published at T-2h: effective_from=T+1h, effective_to=T+24h, mw=100
#   - rev 1 published at T+1h: effective_from=T+1h, effective_to=T+48h, mw=500
#
# At hour T   → only rev 0 is known (published T-2h ≤ T; rev 1 at T+1h not yet)
#             → 0 MW active (T+1h has not started yet at T)
# At hour T+2h → both revisions known; rev 1 is latest
#             → 500 MW (event is active: T+1h ≤ T+2h < T+48h)
#
# Critical anti-leakage assertion: no row with timestamp_utc < T+1h may carry
# 500 MW.  This pins the bi-temporal correctness invariant from the module
# docstring and the plan AC-1.
# ---------------------------------------------------------------------------


def test_bitemporal_leakage_rev1_not_visible_before_publication() -> None:
    """Pins AC-1: rev-1 (500 MW) must NOT appear at any hour before its publication.

    Revision 0 (100 MW) is published at T-2h with effective_from=T+1h.
    Revision 1 (500 MW) is published at T+1h with the same effective_from.

    At hour T: rev 0 is the latest known → only 100 MW *if* the event window
    had started, but T+1h has not yet arrived at T, so the active count is 0
    (effective_from=T+1h > T).

    At hour T+2h: rev 1 is now the latest known → 500 MW active.

    The 500 MW value must NOT appear at any hour whose timestamp_utc < T+1h.
    """
    mrid = "MRID-LEAKAGE-TEST"
    remit_df = _make_remit_df(
        [
            _make_row(
                mrid=mrid,
                revision_number=0,
                published_at=T - pd.Timedelta(hours=2),
                effective_from=T + pd.Timedelta(hours=1),
                effective_to=T + pd.Timedelta(hours=24),
                affected_mw=100.0,
                cause="Planned",
            ),
            _make_row(
                mrid=mrid,
                revision_number=1,
                published_at=T + pd.Timedelta(hours=1),
                effective_from=T + pd.Timedelta(hours=1),
                effective_to=T + pd.Timedelta(hours=48),
                affected_mw=500.0,
                cause="Planned",
            ),
        ]
    )

    # Grid from T-3h through T+50h (inclusive) — hourly
    hourly_index = pd.date_range(
        start=T - pd.Timedelta(hours=3),
        end=T + pd.Timedelta(hours=50),
        freq="h",
        tz="UTC",
    )
    result = derive_remit_features(remit_df, hourly_index)

    # Locate the rows at T and T+2h
    ts_at_T = T
    ts_at_T2h = T + pd.Timedelta(hours=2)
    row_at_T = result.loc[result["timestamp_utc"] == ts_at_T]
    row_at_T2h = result.loc[result["timestamp_utc"] == ts_at_T2h]

    # At T: rev 1 is not yet published → 100 MW if the event window started,
    # but effective_from = T+1h > T so the event is not yet active.
    assert len(row_at_T) == 1, f"Expected exactly one row at T; got {len(row_at_T)}."
    mw_at_T = float(row_at_T.iloc[0]["remit_unavail_mw_total"])
    assert mw_at_T == pytest.approx(0.0), (
        f"At T, rev-0 event has not started (effective_from=T+1h > T); "
        f"expected 0 MW but got {mw_at_T}."
    )

    # At T+2h: rev 1 is known (published T+1h ≤ T+2h); event is active.
    assert len(row_at_T2h) == 1, f"Expected exactly one row at T+2h; got {len(row_at_T2h)}."
    mw_at_T2h = float(row_at_T2h.iloc[0]["remit_unavail_mw_total"])
    assert mw_at_T2h == pytest.approx(500.0), (
        f"At T+2h, rev-1 (500 MW) should be the active revision; got {mw_at_T2h}."
    )

    # Anti-leakage: 500 MW must NOT appear at any row whose timestamp_utc < T+1h
    early_rows = result.loc[result["timestamp_utc"] < T + pd.Timedelta(hours=1)]
    leakage_rows = early_rows.loc[early_rows["remit_unavail_mw_total"] >= 500.0]
    assert len(leakage_rows) == 0, (
        f"Leakage detected: {len(leakage_rows)} row(s) with timestamp_utc < T+1h "
        f"carry ≥500 MW (rev-1 value).  Timestamps: "
        f"{list(leakage_rows['timestamp_utc'])}."
    )


# ---------------------------------------------------------------------------
# Test 2 — Schema conformance (AC-9 + structural invariant)
#
# The derived frame must have columns exactly equal to:
#   ["timestamp_utc", "remit_unavail_mw_total",
#    "remit_active_unplanned_count", "remit_unavail_mw_next_24h"]
# in that order.  Dtypes: float32 / int32 / float32 for the REMIT columns.
# The REMIT_VARIABLE_COLUMNS constant has length 3 and matches column order.
# ---------------------------------------------------------------------------


def test_schema_conformance_column_names_dtypes_and_constant() -> None:
    """Pins AC-9 + schema invariant: column names, order, dtypes, and constant alignment.

    Asserts:
    - REMIT_VARIABLE_COLUMNS has length 3.
    - Derived frame has exactly 4 columns in contractual order.
    - Column names match REMIT_VARIABLE_COLUMNS name list.
    - remit_unavail_mw_total and remit_unavail_mw_next_24h are float32.
    - remit_active_unplanned_count is int32.
    - REMIT_VARIABLE_COLUMNS names are in the same order as the derived columns[1:].
    """
    # Schema check does not depend on REMIT events — use a minimal one-event frame
    remit_df = _make_remit_df(
        [
            _make_row(
                mrid="SCHEMA-TEST",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=1),
                effective_from=T - pd.Timedelta(hours=2),
                effective_to=T + pd.Timedelta(hours=4),
                affected_mw=200.0,
                cause="Planned",
            )
        ]
    )
    hourly_index = pd.date_range(start=T - pd.Timedelta(hours=3), periods=8, freq="h", tz="UTC")
    result = derive_remit_features(remit_df, hourly_index)

    # REMIT_VARIABLE_COLUMNS length
    assert len(REMIT_VARIABLE_COLUMNS) == 3, (
        f"REMIT_VARIABLE_COLUMNS must have exactly 3 entries; got {len(REMIT_VARIABLE_COLUMNS)}."
    )

    # Column count
    expected_columns = [
        "timestamp_utc",
        "remit_unavail_mw_total",
        "remit_active_unplanned_count",
        "remit_unavail_mw_next_24h",
    ]
    assert list(result.columns) == expected_columns, (
        f"derive_remit_features must return columns {expected_columns!r} in order; "
        f"got {list(result.columns)!r}."
    )

    # REMIT_VARIABLE_COLUMNS names match the three REMIT columns in order
    constant_names = [name for name, _ in REMIT_VARIABLE_COLUMNS]
    assert constant_names == expected_columns[1:], (
        f"REMIT_VARIABLE_COLUMNS names must match the three REMIT column names "
        f"in order.  Constant gives {constant_names!r}; frame has {expected_columns[1:]!r}."
    )

    # REMIT_VARIABLE_COLUMNS pyarrow types
    constant_types = {name: dtype for name, dtype in REMIT_VARIABLE_COLUMNS}
    assert constant_types["remit_unavail_mw_total"] == pa.float32(), (
        f"REMIT_VARIABLE_COLUMNS: remit_unavail_mw_total must be pa.float32(); "
        f"got {constant_types['remit_unavail_mw_total']}."
    )
    assert constant_types["remit_active_unplanned_count"] == pa.int32(), (
        f"REMIT_VARIABLE_COLUMNS: remit_active_unplanned_count must be pa.int32(); "
        f"got {constant_types['remit_active_unplanned_count']}."
    )
    assert constant_types["remit_unavail_mw_next_24h"] == pa.float32(), (
        f"REMIT_VARIABLE_COLUMNS: remit_unavail_mw_next_24h must be pa.float32(); "
        f"got {constant_types['remit_unavail_mw_next_24h']}."
    )

    # DataFrame dtypes
    assert str(result["remit_unavail_mw_total"].dtype) == "float32", (
        f"remit_unavail_mw_total must be float32; got {result['remit_unavail_mw_total'].dtype}."
    )
    assert str(result["remit_active_unplanned_count"].dtype) == "int32", (
        f"remit_active_unplanned_count must be int32; "
        f"got {result['remit_active_unplanned_count'].dtype}."
    )
    assert str(result["remit_unavail_mw_next_24h"].dtype) == "float32", (
        f"remit_unavail_mw_next_24h must be float32; "
        f"got {result['remit_unavail_mw_next_24h'].dtype}."
    )


# ---------------------------------------------------------------------------
# Test 3 — Zero-event hour handling (AC-7)
#
# Pass an empty REMIT frame (zero rows, correct schema) and a 48-hour grid.
# Assert: shape is (48, 4), all REMIT columns are zero (not NaN), no raise.
# ---------------------------------------------------------------------------


def test_zero_event_hour_handling() -> None:
    """Pins AC-7: an empty REMIT log produces an all-zero, NaN-free derived frame.

    Asserts:
    - Shape is (48, 4) — exactly one row per hourly grid point.
    - No NaN values in any REMIT column.
    - All three REMIT columns contain only zeros.
    - The function does not raise.
    """
    remit_df = _empty_remit_df()
    hourly_index = pd.date_range(
        start=pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        periods=48,
        freq="h",
        tz="UTC",
    )

    result = derive_remit_features(remit_df, hourly_index)

    assert result.shape == (48, 4), (
        f"Empty-REMIT derived frame must have shape (48, 4); got {result.shape}."
    )

    remit_cols = [
        "remit_unavail_mw_total",
        "remit_active_unplanned_count",
        "remit_unavail_mw_next_24h",
    ]
    for col in remit_cols:
        nan_count = result[col].isna().sum()
        assert nan_count == 0, (
            f"AC-7: column {col!r} must contain no NaN; found {nan_count} NaN value(s)."
        )
        nonzero_count = (result[col] != 0).sum()
        assert nonzero_count == 0, (
            f"AC-7: empty-log column {col!r} must be all-zero; "
            f"found {nonzero_count} non-zero value(s)."
        )


# ---------------------------------------------------------------------------
# Test 4 — Forward-looking column semantics (AC-6)
#
# Single event: published before the window, effective_from=2024-01-15 12:00,
# effective_to=2024-01-20 12:00, affected_mw=200, cause="Planned".
# forward_lookahead_hours=24.
#
# Assertions:
#   - At 2024-01-14 12:00 UTC (24h before start): remit_unavail_mw_next_24h == 200
#   - At 2024-01-15 11:00 UTC (1h before start):  remit_unavail_mw_next_24h == 200
#   - At 2024-01-15 12:00 UTC (the start itself):  remit_unavail_mw_next_24h == 0
#                                                   (event is now "current", not "future")
#   - At 2024-01-14 11:00 UTC (25h before start):  remit_unavail_mw_next_24h == 0
#                                                   (outside the 24h lookahead window)
#   - With forward_lookahead_hours=0: all-zeros for that column.
# ---------------------------------------------------------------------------


def test_forward_looking_column_semantics() -> None:
    """Pins AC-6: remit_unavail_mw_next_24h is non-zero only in the lookahead window.

    The forward-looking column fires when effective_from ∈ [t, t + lookahead_h).
    Events whose effective_from has already arrived (effective_from ≤ t) do not
    fire the forward column — they fire remit_unavail_mw_total instead.

    Full boundary assertions are included to pin the open/closed interval
    semantics defined in the module docstring algorithm §Step 2.
    """
    published = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    effective_from = pd.Timestamp("2024-01-15 12:00", tz="UTC")
    effective_to = pd.Timestamp("2024-01-20 12:00", tz="UTC")

    remit_df = _make_remit_df(
        [
            _make_row(
                mrid="FWD-TEST",
                revision_number=0,
                published_at=published,
                effective_from=effective_from,
                effective_to=effective_to,
                affected_mw=200.0,
                cause="Planned",
            )
        ]
    )

    # Build a grid covering the region of interest: 2024-01-14 10:00 to 2024-01-16 00:00
    hourly_index = pd.date_range(
        start=pd.Timestamp("2024-01-14 10:00", tz="UTC"),
        end=pd.Timestamp("2024-01-16 00:00", tz="UTC"),
        freq="h",
        tz="UTC",
    )
    result = derive_remit_features(remit_df, hourly_index, forward_lookahead_hours=24)

    def _lookup(ts: pd.Timestamp) -> tuple[float, float]:
        """Return (remit_unavail_mw_next_24h, remit_unavail_mw_total) at ts."""
        row = result.loc[result["timestamp_utc"] == ts]
        assert len(row) == 1, f"Expected exactly 1 row at {ts}; got {len(row)}."
        return (
            float(row.iloc[0]["remit_unavail_mw_next_24h"]),
            float(row.iloc[0]["remit_unavail_mw_total"]),
        )

    # 2024-01-14 11:00 UTC: 25h before start → outside the 24h lookahead window → 0
    t_25h_before = pd.Timestamp("2024-01-14 11:00", tz="UTC")
    fwd_25h, _ = _lookup(t_25h_before)
    assert fwd_25h == pytest.approx(0.0), (
        f"At {t_25h_before} (25h before start), remit_unavail_mw_next_24h must be 0; got {fwd_25h}."
    )

    # 2024-01-14 12:00 UTC: exactly 24h before start → boundary, within [t, t+24h) → 200
    t_24h_before = pd.Timestamp("2024-01-14 12:00", tz="UTC")
    fwd_24h, _ = _lookup(t_24h_before)
    assert fwd_24h == pytest.approx(200.0), (
        f"At {t_24h_before} (24h before start), remit_unavail_mw_next_24h must be 200 MW; "
        f"got {fwd_24h}."
    )

    # 2024-01-15 11:00 UTC: 1h before start → inside [t, t+24h) → 200
    t_1h_before = pd.Timestamp("2024-01-15 11:00", tz="UTC")
    fwd_1h, active_1h = _lookup(t_1h_before)
    assert fwd_1h == pytest.approx(200.0), (
        f"At {t_1h_before} (1h before start), remit_unavail_mw_next_24h must be 200 MW; "
        f"got {fwd_1h}."
    )
    # The event has not started yet — active total should be 0 at T-1h
    assert active_1h == pytest.approx(0.0), (
        f"At {t_1h_before} (1h before start), remit_unavail_mw_total must be 0 "
        f"(event not started); got {active_1h}."
    )

    # 2024-01-15 12:00 UTC: the moment effective_from arrives → "current" not "future"
    # The forward window is [t, t+24h); t == effective_from so forward fires if and
    # only if effective_from is strictly after t.  Per the open interval, this is 0.
    t_start = pd.Timestamp("2024-01-15 12:00", tz="UTC")
    fwd_at_start, active_at_start = _lookup(t_start)
    assert fwd_at_start == pytest.approx(0.0), (
        f"At effective_from={t_start}, remit_unavail_mw_next_24h must be 0 "
        f"(event is now 'current', not 'future'); got {fwd_at_start}."
    )
    # At the start moment the event IS active
    assert active_at_start == pytest.approx(200.0), (
        f"At effective_from={t_start}, remit_unavail_mw_total must be 200 MW; "
        f"got {active_at_start}."
    )

    # With forward_lookahead_hours=0 the column is always zero
    result_zero_lookahead = derive_remit_features(remit_df, hourly_index, forward_lookahead_hours=0)
    all_zero = (result_zero_lookahead["remit_unavail_mw_next_24h"] == 0).all()
    assert all_zero, "With forward_lookahead_hours=0, remit_unavail_mw_next_24h must be all-zero."


# ---------------------------------------------------------------------------
# Test 5 — No-NaN invariant (AC-7 cont.) — stub corpus
#
# For the real stub fixture (BRISTOL_ML_REMIT_STUB=1, 10 rows), which has
# open-ended effective_to (NaT), planned/unplanned/forced causes, and the
# Withdrawn case, the derived frame must contain no NaN in any REMIT column.
# ---------------------------------------------------------------------------


def test_no_nan_in_remit_columns_with_mixed_nulls(
    tmp_path: pytest.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins AC-7: no NaN in any REMIT column even with open-ended / null rows.

    Uses the canonical stub fixture (BRISTOL_ML_REMIT_STUB=1) from Stage 13,
    which contains:
    - Two open-ended rows (effective_to=NaT): M-D (Wind), M-F (Hydro)
    - Planned, Unplanned, and Forced causes
    - A Withdrawn revision (M-C, rev 1)
    - Rows with null affected_mw (via the Withdrawn case)
    - A multi-revision chain (M-B, revisions 0/1/2)

    Asserts that the derived frame has no NaN in any of the three REMIT columns
    across a 200-hour grid spanning the stub corpus.
    """
    from bristol_ml.ingestion import remit as remit_ingest
    from conf._schemas import RemitIngestionConfig

    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    cfg = RemitIngestionConfig(cache_dir=tmp_path, cache_filename="remit.parquet")
    stub_path = remit_ingest.fetch(cfg, cache=remit_ingest.CachePolicy.REFRESH)
    remit_df = remit_ingest.load(stub_path)

    # Grid spanning the stub corpus: 2024-01-01 through roughly 2024-08-20 (200 rows)
    hourly_index = pd.date_range(
        start=pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        periods=200,
        freq="h",
        tz="UTC",
    )

    result = derive_remit_features(remit_df, hourly_index)

    remit_cols = [
        "remit_unavail_mw_total",
        "remit_active_unplanned_count",
        "remit_unavail_mw_next_24h",
    ]
    for col in remit_cols:
        nan_count = int(result[col].isna().sum())
        assert nan_count == 0, (
            f"AC-7: column {col!r} must contain no NaN values even with "
            f"mixed-null REMIT log; found {nan_count} NaN(s)."
        )


# ---------------------------------------------------------------------------
# Test 6 — Multi-revision-per-mrid: Withdrawn truncates prior Active (invariant)
#
# Three rows for the same mrid:
#   - rev 0 Active, published 2024-03-01 08:00, effective 2024-03-05 .. 2024-03-10, mw=500
#   - rev 1 Withdrawn, published 2024-03-02 09:00, effective 2024-03-05 .. 2024-03-10, mw=500
#
# Because rev 1 (Withdrawn) is published BEFORE the event window opens, rev 0
# is truncated in the transaction-time dimension to [2024-03-01 08:00, 2024-03-02 09:00).
# The event window [2024-03-05 .. 2024-03-10] lies entirely after 2024-03-02 09:00,
# so there is NO overlap between rev 0's transaction-time validity and the event window.
# Hours 2024-03-05 00:00 .. 2024-03-09 23:00 must therefore have 0 MW.
# ---------------------------------------------------------------------------


def test_withdrawn_revision_truncates_prior_active_revision() -> None:
    """Pins the Withdrawn-truncates-prior semantic from ``_per_mrid_validity``.

    A Withdrawn revision must shorten the transaction-time validity of the
    preceding Active revision.  If the Withdrawal is published before the
    event window opens, the event window sees zero unavailability.

    This test guards the fix described in the plan D17 note referencing
    Stage 13 commit message semantics.
    """
    mrid = "MRID-WITHDRAWN-TRUNC"
    remit_df = _make_remit_df(
        [
            _make_row(
                mrid=mrid,
                revision_number=0,
                message_status="Active",
                published_at=pd.Timestamp("2024-03-01 08:00", tz="UTC"),
                effective_from=pd.Timestamp("2024-03-05 00:00", tz="UTC"),
                effective_to=pd.Timestamp("2024-03-10 00:00", tz="UTC"),
                affected_mw=500.0,
                cause="Planned",
            ),
            _make_row(
                mrid=mrid,
                revision_number=1,
                message_status="Withdrawn",
                published_at=pd.Timestamp("2024-03-02 09:00", tz="UTC"),
                effective_from=pd.Timestamp("2024-03-05 00:00", tz="UTC"),
                effective_to=pd.Timestamp("2024-03-10 00:00", tz="UTC"),
                affected_mw=500.0,
                cause="Planned",
            ),
        ]
    )

    # Grid covering the full event window
    hourly_index = pd.date_range(
        start=pd.Timestamp("2024-03-05 00:00", tz="UTC"),
        end=pd.Timestamp("2024-03-09 23:00", tz="UTC"),
        freq="h",
        tz="UTC",
    )

    result = derive_remit_features(remit_df, hourly_index)

    # All hours in the event window must see 0 MW (Withdrawn truncated rev-0)
    nonzero_rows = result.loc[result["remit_unavail_mw_total"] > 0]
    assert len(nonzero_rows) == 0, (
        f"Withdrawn revision must truncate prior Active revision's transaction-time "
        f"validity.  Expected 0 MW across {len(hourly_index)} grid hours but found "
        f"{len(nonzero_rows)} nonzero row(s).  "
        f"First nonzero: {nonzero_rows.head(3).to_dict('records')}."
    )


# ---------------------------------------------------------------------------
# Test 7 — AC-8: module runs standalone
#
# ``python -m bristol_ml.features.remit`` under ``BRISTOL_ML_REMIT_STUB=1``
# should exit 0 and print the three column names.  Using ``_cli_main`` directly
# (matching the test_assembler_calendar.py precedent) is faster than subprocess.
# ---------------------------------------------------------------------------


def test_cli_main_standalone_exits_zero_and_prints_columns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: pytest.Path,
) -> None:
    """Pins AC-8: ``_cli_main([])`` exits 0 and prints the three REMIT column names.

    Uses monkeypatch to set BRISTOL_ML_REMIT_STUB=1 so the CLI finds the stub
    REMIT cache without touching the network.  The warm REMIT cache is pre-built
    via the ingestion stub, so _cli_main can load it and emit the sample table.
    """
    from bristol_ml.features.remit import _cli_main
    from bristol_ml.ingestion import remit as remit_ingest
    from conf._schemas import RemitIngestionConfig

    # Prime the stub cache in tmp_path so the CLI finds it
    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    cfg = RemitIngestionConfig(cache_dir=tmp_path, cache_filename="remit.parquet")
    remit_ingest.fetch(cfg, cache=remit_ingest.CachePolicy.REFRESH)

    # Point the CLI's REMIT cache dir to tmp_path via the env var used by Hydra config
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))

    rc = _cli_main([])

    assert rc == 0, f"_cli_main must exit 0; got {rc}."

    stdout = capsys.readouterr().out
    # All three column names must appear in the printed output
    for col_name, _ in REMIT_VARIABLE_COLUMNS:
        assert col_name in stdout, (
            f"Column name {col_name!r} must appear in _cli_main output; got stdout:\n{stdout!r}"
        )


# ---------------------------------------------------------------------------
# Test 8 — AC-9: public surface importable
# ---------------------------------------------------------------------------


def test_public_surface_importable() -> None:
    """Pins AC-9: REMIT_VARIABLE_COLUMNS and derive_remit_features are importable.

    The import is already at the top of this file; this test pins the contract
    explicitly so a future rename / removal would fail here with a clear message.
    """
    # Both names are imported at module level — verify they are not None
    # and have the expected types.
    assert REMIT_VARIABLE_COLUMNS is not None, (
        "REMIT_VARIABLE_COLUMNS must be importable from bristol_ml.features.remit."
    )
    assert callable(derive_remit_features), (
        f"derive_remit_features must be callable; got {type(derive_remit_features).__name__}."
    )
    # Verify via __all__
    import bristol_ml.features.remit as remit_mod

    assert "REMIT_VARIABLE_COLUMNS" in remit_mod.__all__, (
        "REMIT_VARIABLE_COLUMNS must be listed in bristol_ml.features.remit.__all__."
    )
    assert "derive_remit_features" in remit_mod.__all__, (
        "derive_remit_features must be listed in bristol_ml.features.remit.__all__."
    )


# ---------------------------------------------------------------------------
# Test 9 — Validation guards (invariant)
#
# a) tz-naive hourly_index raises ValueError
# b) empty hourly_index raises ValueError
# c) missing required column in remit_df raises ValueError naming the column
# ---------------------------------------------------------------------------


class TestValidationGuards:
    """Invariant tests for the input-validation layer of derive_remit_features.

    Each sub-test verifies that an illegal input surfaces as a ValueError
    (not a cryptic downstream error) with a message that names the problem.
    """

    def test_tz_naive_hourly_index_raises_valueerror(self) -> None:
        """A tz-naive hourly_index must raise ValueError mentioning 'tz' or 'naive'."""
        remit_df = _empty_remit_df()
        naive_index = pd.date_range(
            start=pd.Timestamp("2024-01-01"),
            periods=24,
            freq="h",
            # intentionally no tz= argument → naive
        )
        assert naive_index.tz is None, "Sanity: test index must be naive."

        with pytest.raises(ValueError) as exc_info:
            derive_remit_features(remit_df, naive_index)

        error_text = str(exc_info.value).lower()
        assert "tz" in error_text or "naive" in error_text or "utc" in error_text, (
            f"ValueError for tz-naive index must mention 'tz', 'naive', or 'utc'; "
            f"got: {exc_info.value!r}"
        )

    def test_empty_hourly_index_raises_valueerror(self) -> None:
        """An empty hourly_index must raise ValueError."""
        remit_df = _empty_remit_df()
        empty_index = pd.DatetimeIndex([], tz="UTC")

        with pytest.raises(ValueError):
            derive_remit_features(remit_df, empty_index)

    def test_missing_required_column_raises_valueerror_naming_column(self) -> None:
        """A remit_df missing a required column must raise ValueError naming the column.

        The error must include the missing column name so the caller can
        diagnose the mismatch without inspecting the source code.
        """
        # Build a valid remit_df then drop a required column
        remit_df = _make_remit_df(
            [
                _make_row(
                    mrid="VAL-TEST",
                    revision_number=0,
                    published_at=T - pd.Timedelta(hours=1),
                )
            ]
        )
        # Drop a column that is definitely required by the function
        remit_df_missing = remit_df.drop(columns=["mrid"])
        hourly_index = pd.date_range(start=T, periods=4, freq="h", tz="UTC")

        with pytest.raises(ValueError) as exc_info:
            derive_remit_features(remit_df_missing, hourly_index)

        error_text = str(exc_info.value)
        assert "mrid" in error_text, (
            f"ValueError for missing 'mrid' column must name the column; got: {error_text!r}"
        )


# ---------------------------------------------------------------------------
# Test 10 — Unplanned-vs-Forced count discrimination (reviewer R3)
#
# The module docstring says "Forced" is a distinct category from "Unplanned"
# and is NOT counted by remit_active_unplanned_count.  Pin the discrimination
# at the value level: construct one Unplanned event and one Forced event with
# the same active window and assert the count is 1 (Unplanned only), not 2.
# ---------------------------------------------------------------------------


def test_unplanned_count_excludes_forced_cause() -> None:
    """``remit_active_unplanned_count`` counts Unplanned only; Forced is excluded.

    Constructs two events whose active windows overlap exactly at the same
    hours: one with cause='Unplanned', one with cause='Forced'.  At any hour
    in the overlap, the count must be 1 (the Unplanned event), not 2.

    Pins the load-bearing cause-vocabulary discrimination from the module's
    ``_UNPLANNED_TAG`` constant + comment (Stage 16 plan; reviewer R3).
    """
    remit_df = _make_remit_df(
        [
            _make_row(
                mrid="UNPLANNED-EVENT",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=2),
                effective_from=T,
                effective_to=T + pd.Timedelta(hours=4),
                affected_mw=200.0,
                cause="Unplanned",
            ),
            _make_row(
                mrid="FORCED-EVENT",
                revision_number=0,
                published_at=T - pd.Timedelta(hours=2),
                effective_from=T,
                effective_to=T + pd.Timedelta(hours=4),
                affected_mw=300.0,
                cause="Forced",
            ),
        ]
    )
    hourly_index = pd.date_range(start=T - pd.Timedelta(hours=1), periods=8, freq="h", tz="UTC")

    derived = derive_remit_features(remit_df, hourly_index)
    overlap = derived[
        (derived["timestamp_utc"] >= T) & (derived["timestamp_utc"] < T + pd.Timedelta(hours=4))
    ]

    # Both events contribute to remit_unavail_mw_total during the overlap.
    assert (overlap["remit_unavail_mw_total"] == 500.0).all(), (
        "Both Unplanned and Forced events should contribute to the MW sum; "
        f"got {overlap['remit_unavail_mw_total'].tolist()!r}"
    )
    # Only the Unplanned event contributes to the count signal — Forced is
    # a distinct upstream category.
    assert (overlap["remit_active_unplanned_count"] == 1).all(), (
        "remit_active_unplanned_count must equal 1 during the overlap (Unplanned only, "
        f"NOT Forced); got {overlap['remit_active_unplanned_count'].tolist()!r}"
    )
