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

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.ingestion import remit
from bristol_ml.ingestion.remit import (
    FUEL_TYPES,
    MESSAGE_STATUSES,
    OUTPUT_SCHEMA,
    as_of,
)
from conf._schemas import RemitIngestionConfig

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
        f"Open-ended (effective_to=NaT) row must appear in as_of result; got {len(result)} row(s)."
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
        "effective_to": {"type": utc_timestamp_type, "nullable": True},  # D10 load-bearing
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
    nullable_columns = (
        "affected_unit",
        "asset_id",
        "fuel_type",
        "affected_mw",
        "normal_capacity_mw",
        "event_type",
        "cause",
        "message_description",
    )
    for col in nullable_columns:
        assert col in schema_names, f"Column {col!r} missing from OUTPUT_SCHEMA."
        field = OUTPUT_SCHEMA.field(col)
        assert field.nullable, f"Column {col!r} must be nullable; got nullable=False."


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


# ---------------------------------------------------------------------------
# T3 — stub fetch + load + CLI
# ---------------------------------------------------------------------------
#
# Tests in this section exercise the paths wired at Stage 13 Task T3:
#   - fetch(config, cache=REFRESH) with BRISTOL_ML_REMIT_STUB=1
#   - load(path) round-trip (schema, timestamp tz, NaT open-ended rows)
#   - CacheMissingError from CachePolicy.OFFLINE with no warm cache
#   - Standalone CLI (python -m bristol_ml.ingestion.remit --help)
#
# All five tests are drawn from plan §6 T3 and §4 acceptance criteria
# (AC-2/AC-5/NFR-2/AC-11).  No cassette or live network needed.
# ---------------------------------------------------------------------------


def _build_remit_config(tmp_path: Path) -> RemitIngestionConfig:
    """Return a ``RemitIngestionConfig`` pointing at ``tmp_path``.

    All protocol fields take their declared defaults; only ``cache_dir``
    and ``cache_filename`` are overridden so the cache lands under the
    test's temporary directory.
    """
    return RemitIngestionConfig(cache_dir=tmp_path, cache_filename="remit.parquet")


def test_fetch_with_stub_env_var_writes_canonical_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins NFR-2 + D12 + AC-3: stub path writes a valid parquet with 10 rows.

    When ``BRISTOL_ML_REMIT_STUB=1`` is set, ``fetch`` with
    ``CachePolicy.REFRESH`` must:

    1. Return a path equal to ``cache_dir / cache_filename`` and that path
       must exist on disk.
    2. Produce a parquet file whose arrow schema equals ``OUTPUT_SCHEMA``
       exactly (16 columns, correct types, correct nullability).
    3. Contain exactly 10 rows — the ten hand-crafted records in
       ``_stub_records()``.
    """
    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    cfg = _build_remit_config(tmp_path)

    result_path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)

    expected_path = tmp_path / "remit.parquet"
    assert result_path == expected_path, (
        f"fetch must return cache_dir / cache_filename; got {result_path}."
    )
    assert result_path.exists(), f"Parquet file must exist at {result_path}."

    table = pq.read_table(result_path)
    assert table.schema == remit.OUTPUT_SCHEMA, (
        f"On-disk schema does not match OUTPUT_SCHEMA.\n"
        f"Got:      {table.schema}\n"
        f"Expected: {remit.OUTPUT_SCHEMA}"
    )
    assert table.num_rows == 10, f"Stub fixture must write exactly 10 rows; got {table.num_rows}."


def test_load_round_trips_all_four_timestamps_with_tz_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins D8 + D18g + AC-5: all four temporal columns round-trip with tz=UTC.

    After a stub fetch and a ``load``, each of the four temporal columns
    must carry a ``pd.DatetimeTZDtype`` whose ``tz`` attribute resolves to
    ``'UTC'``.  This pins the plan §5 requirement that no naive datetimes
    survive the parquet round-trip (NFR-7).
    """
    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    cfg = _build_remit_config(tmp_path)
    path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)

    df = remit.load(path)

    temporal_columns = ("published_at", "effective_from", "effective_to", "retrieved_at_utc")
    for col in temporal_columns:
        assert col in df.columns, f"Column {col!r} missing from loaded frame."
        dtype = df[col].dtype
        assert isinstance(dtype, pd.DatetimeTZDtype), (
            f"Column {col!r} must have pd.DatetimeTZDtype; got {dtype!r}."
        )
        assert str(dtype.tz) == "UTC", f"Column {col!r} must have tz='UTC'; got tz={dtype.tz!r}."


def test_load_round_trips_open_ended_effective_to_as_pd_nat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins D10 + D18g + AC-5: open-ended rows have effective_to == pd.NaT.

    The stub fixture contains exactly two open-ended records: ``M-D``
    (Wind) and ``M-F`` (Hydro), both with ``effective_to=None``.
    After a parquet round-trip via ``load``, those two rows must carry
    ``pd.NaT`` in ``effective_to`` and all other rows must be non-null.
    """
    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    cfg = _build_remit_config(tmp_path)
    path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)

    df = remit.load(path)

    null_count = df["effective_to"].isna().sum()
    assert null_count == 2, f"Exactly 2 open-ended rows expected (M-D + M-F); got {null_count}."
    open_ended_mrids = set(df.loc[df["effective_to"].isna(), "mrid"])
    assert open_ended_mrids == {"M-D", "M-F"}, (
        f"Open-ended mRIDs must be exactly {{'M-D', 'M-F'}}; got {open_ended_mrids}."
    )


def test_fetch_offline_with_warm_cache_returns_path_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins AC-2 + D18d: ``CachePolicy.OFFLINE`` returns the cache path on a warm cache.

    The warm-cache offline contract is half of AC-2: ``OFFLINE`` raises
    when the cache is missing (covered by the sibling
    ``test_fetch_offline_without_cache_raises_cache_missing``) and
    returns the path **without touching the network** when it is warm.

    The test exercises the warm path by:

    1. Populating the cache via a stub-mode REFRESH so the parquet
       lands at ``tmp_path / remit.parquet``.
    2. Removing the stub env-var so any code path that *does* reach the
       network (or the stub's in-memory fallback) would do unintended
       work — neither should happen on a warm OFFLINE call.
    3. Monkeypatching ``httpx.Client`` to a sentinel that raises on any
       attribute access; if ``fetch`` opens an HTTP client under
       ``OFFLINE``, the access fault surfaces as a clear test failure.
    4. Calling ``fetch(cfg, cache=OFFLINE)`` and asserting it returns
       the expected path.

    Together with the sibling ``raises`` test, this pins both branches
    of the AC-2 acceptance criterion: warm-OFFLINE returns the path,
    cold-OFFLINE raises ``CacheMissingError``.
    """
    cfg = _build_remit_config(tmp_path)

    # Step 1: warm the cache via stub mode.
    monkeypatch.setenv("BRISTOL_ML_REMIT_STUB", "1")
    warm_path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)
    assert warm_path.exists(), "Pre-condition: stub fetch must populate the cache."

    # Step 2: clear the stub trigger so the OFFLINE path is exercised
    # without any in-memory fallback waiting in the wings.
    monkeypatch.delenv("BRISTOL_ML_REMIT_STUB", raising=False)

    # Step 3: poison ``httpx.Client`` so any HTTP-client construction
    # attempt during the OFFLINE call surfaces as a loud test failure
    # rather than silently doing network work.
    class _NoNetworkClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "OFFLINE cache hit must not construct an httpx.Client; got call with "
                f"args={args!r} kwargs={kwargs!r}."
            )

    monkeypatch.setattr("bristol_ml.ingestion.remit.httpx.Client", _NoNetworkClient)

    # Step 4: warm-OFFLINE call returns the same path without network I/O.
    offline_path = remit.fetch(cfg, cache=remit.CachePolicy.OFFLINE)

    assert offline_path == warm_path, (
        f"Warm OFFLINE fetch must return the cached path {warm_path}; got {offline_path}."
    )
    assert offline_path.exists(), "Cache path must still exist after the OFFLINE call."


def test_fetch_offline_without_cache_raises_cache_missing(tmp_path: Path) -> None:
    """Pins AC-2 + D18d + NFR-6: CachePolicy.OFFLINE raises when the cache is absent.

    An empty ``cache_dir`` with ``CachePolicy.OFFLINE`` must raise
    ``CacheMissingError``.  The stub env var need not be set — the
    ``OFFLINE`` branch checks for the cache file before deciding whether
    to reach the network (or the stub), so the test exercises the guard
    path regardless.

    The raised error message must contain the absolute cache path so the
    user knows where to place the missing file.
    """
    cfg = _build_remit_config(tmp_path)
    expected_cache = tmp_path / "remit.parquet"
    assert not expected_cache.exists(), "Pre-condition: cache must be absent before the call."

    with pytest.raises(remit.CacheMissingError) as exc_info:
        remit.fetch(cfg, cache=remit.CachePolicy.OFFLINE)

    error_text = str(exc_info.value)
    assert str(expected_cache) in error_text, (
        f"CacheMissingError message must contain the absolute cache path "
        f"{str(expected_cache)!r}; got: {error_text!r}"
    )


# ---------------------------------------------------------------------------
# Phase 3 review-finding regression tests (R1 / R2 / R3 / N2)
# ---------------------------------------------------------------------------
#
# These tests pin behavioural fixes raised by the Phase-3 code-reviewer.
# They exist to prevent the exact bug from regressing — each docstring
# names the original failure mode.
# ---------------------------------------------------------------------------


def test_parse_message_missing_required_field_raises_typed_parse_error() -> None:
    """Pins R2: missing required field surfaces as ``RemitParseError``, not bare ``KeyError``.

    The previous implementation raised ``KeyError('mrid')`` from a bare
    ``record["mrid"]`` lookup, leaving the operator with no context
    about which record was malformed or what other fields it carried.
    The fix wraps the lookup in an explicit ``required_fields`` loop
    and raises a typed ``RemitParseError`` carrying the field name and
    the available keys.

    The test feeds a record missing ``mrid`` and asserts:

    1. The exception type is :class:`remit.RemitParseError`
       (and therefore a :class:`ValueError` for callers that catch the
       broader category).
    2. The error message names the missing field (``'mrid'``).
    3. The error message lists the available keys so the operator can
       see *which* malformed record triggered the failure.
    """
    record = {
        "revisionNumber": 1,
        "publishTime": "2024-01-01T09:00:00Z",
        "eventStartTime": "2024-01-02T00:00:00Z",
        "eventStatus": "Active",
    }

    with pytest.raises(remit.RemitParseError) as exc_info:
        remit._parse_message(record)

    error_text = str(exc_info.value)
    assert "'mrid'" in error_text, (
        f"RemitParseError must name the missing field 'mrid'; got: {error_text!r}"
    )
    assert "revisionNumber" in error_text, (
        f"RemitParseError must list the available keys for context; got: {error_text!r}"
    )
    # Also verify it's a ValueError so existing 'except ValueError' callers still work.
    assert isinstance(exc_info.value, ValueError), (
        "RemitParseError must inherit from ValueError for catch-broader compatibility."
    )


def test_live_fetch_url_construction_handles_slash_variations() -> None:
    """Pins R3: URL composition produces a single separator under any slash combination.

    The previous implementation was ``f"{config.base_url}{config.endpoint_path}"``
    which silently produced ``"https://api/example.compath/foo"`` on
    no-trailing/no-leading and ``"https://api/example.com//path/foo"``
    on trailing/leading.  The fix mirrors ``neso.py`` /
    ``neso_forecast.py``: ``rstrip('/')`` the base, ``lstrip('/')`` the
    endpoint, join with a single ``"/"``.

    The test inspects the source of ``_live_fetch`` to confirm the
    rstrip/lstrip pattern is in place — a structural guard against
    regressing to the f-string concat.  This is a source-level test
    rather than a behavioural one because exercising the live URL
    construction without hitting the network would require an httpx
    mock that defeats the purpose; the structural guard catches the
    intended regression cheaply.
    """
    import inspect

    source = inspect.getsource(remit._live_fetch)
    assert '.rstrip("/")' in source and '.lstrip("/")' in source, (
        f"_live_fetch must use rstrip/lstrip URL composition (R3 fix); "
        f"got source without the pattern. Excerpt:\n{source[:400]}"
    )
    # And confirm the bare f-string concat is gone.
    assert 'f"{config.base_url}{config.endpoint_path}"' not in source, (
        f"_live_fetch must not use the bare f-string concat that R3 fixed; "
        f"got source with regression: {source[:400]}"
    )


def test_live_fetch_default_window_end_uses_utc_not_local_time() -> None:
    """Pins R1: ``window_end`` defaults to UTC date, not local-time date.

    The previous implementation was ``config.window_end or date.today()``
    which uses the host's local timezone — a Bristol host at 23:30
    local on 30 June would resolve ``date.today()`` to 30 June, while
    ``datetime.now(UTC).date()`` would (on BST) resolve to 30 June and
    (on GMT) to 30 June; the failure mode bites at midnight crossings,
    where the local clock can show a different calendar date from UTC.

    The same UTC-discipline failure mode surfaced and was patched on
    NESO ingestion at Stage 1; this test makes the discipline explicit
    for REMIT.

    Test approach: source-level inspection of the *call expression*
    (not the docstring/comment text — the explanatory comment for the
    fix necessarily mentions ``date.today()``, so the regression check
    targets the actual statement ``= config.window_end or ...``).
    """
    import inspect
    import re

    source = inspect.getsource(remit._live_fetch)
    # Strip comments and docstrings — leave just the code statements.
    code_only_lines = [
        line for line in source.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    # Drop the docstring block by finding the first triple-quote pair.
    in_docstring = False
    code_only: list[str] = []
    for line in code_only_lines:
        triple = line.count('"""')
        if in_docstring:
            if triple >= 1:
                in_docstring = False
            continue
        if triple >= 2:
            # Single-line docstring; skip
            continue
        if triple == 1:
            in_docstring = True
            continue
        code_only.append(line)
    code_text = "\n".join(code_only)

    # The fixed assignment must be present; the previous bug must not.
    assert re.search(r"window_end\s*=.*datetime\.now\(UTC\)\.date\(\)", code_text), (
        f"_live_fetch must assign window_end via datetime.now(UTC).date() "
        f"(R1 fix). Code-only source:\n{code_text[:600]}"
    )
    assert not re.search(r"window_end\s*=.*\bdate\.today\(\)", code_text), (
        f"_live_fetch must not assign window_end from date.today() "
        f"(R1 regression). Code-only source:\n{code_text[:600]}"
    )


def test_to_arrow_or_live_fetch_uses_remit_parse_error_for_payload_shape() -> None:
    """Pins N2: payload-shape errors raise ``RemitParseError``, not bare ``RuntimeError``.

    The previous implementation raised ``RuntimeError`` for two
    payload-shape failures in ``_live_fetch``:

    - top-level payload not a list
    - per-record payload not a dict

    Both are now ``RemitParseError(ValueError)`` so callers can
    distinguish parse errors from generic runtime issues without
    string-matching.  Source-level guard.
    """
    import inspect

    source = inspect.getsource(remit._live_fetch)
    # Both payload-shape errors should now be RemitParseError, not RuntimeError.
    assert "RemitParseError" in source, (
        f"_live_fetch must raise RemitParseError for payload-shape errors (N2 fix). "
        f"Source excerpt:\n{source[:400]}"
    )
    # The empty-records guard in _to_arrow legitimately stays as RuntimeError
    # (it's a fetch-result sanity, not a parse error); the targeted N2 fix is
    # in _live_fetch only. Confirm _live_fetch no longer raises bare RuntimeError
    # for the two payload-shape paths.
    assert 'raise RuntimeError(\n            f"REMIT stream' not in source, (
        f"_live_fetch must not raise RuntimeError for payload-shape errors "
        f"(N2 regression). Source excerpt:\n{source[:400]}"
    )


def test_remit_module_runs_standalone() -> None:
    """Pins AC-11 + NFR-2: ``python -m bristol_ml.ingestion.remit --help`` exits 0.

    Every ingestion module must be runnable as ``python -m <module>``
    (DESIGN §2.1.1).  The ``--help`` flag exercises the CLI entry point
    without performing any network access or requiring a warm cache,
    and provides a smoke test that the module imports cleanly and the
    argparse surface is wired.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.ingestion.remit", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help must exit 0; got {result.returncode}.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "usage:" in result.stdout.lower(), (
        f"--help output must contain 'usage:'; got stdout: {result.stdout!r}"
    )
