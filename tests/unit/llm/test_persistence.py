"""Spec-derived tests for ``bristol_ml.llm.persistence`` — Stage 16 T3.

Every test here is derived from the Stage 16 plan T3 acceptance criteria:

    ``tests/unit/llm/test_extractor_persistence.py``: stub-mode parquet
    round-trip; idempotent re-write.

Extended per the tester brief with:

- Empty corpus.
- ``load_extracted`` schema rejection (missing column, extra column).
- Missing required REMIT column raises ValueError.

All tests use the stub extractor (``StubExtractor()``) and build the REMIT
frame in-memory using the same helpers as ``tests/unit/ingestion/test_remit.py``
and ``tests/unit/features/test_remit.py`` — no live OpenAI path, no cassette,
no network.

``BRISTOL_ML_REMIT_STUB=1`` is set where tests need a real parquet from the
REMIT ingestion stub (test 1). The other tests build frames directly in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.llm.extractor import StubExtractor
from bristol_ml.llm.persistence import (
    EXTRACTED_OUTPUT_SCHEMA,
    extract_and_persist,
    load_extracted,
)

# ---------------------------------------------------------------------------
# Helpers — build REMIT frames in-memory
#
# Mirrors the row-builder pattern from tests/unit/ingestion/test_remit.py and
# tests/unit/features/test_remit.py.  All required columns for
# ``extract_and_persist`` come from the ``required`` tuple in the production
# code, plus additional ingestion-layer columns that define the full schema.
# ---------------------------------------------------------------------------

_T = pd.Timestamp("2024-03-01T10:00:00", tz="UTC")

_REMIT_DEFAULTS: dict[str, Any] = {
    "message_type": "Production",
    "message_status": "Active",
    "published_at": _T - pd.Timedelta(hours=1),
    "effective_from": _T,
    "effective_to": _T + pd.Timedelta(hours=8),
    "retrieved_at_utc": _T - pd.Timedelta(minutes=5),
    "affected_unit": "T_UNIT-1",
    "asset_id": "T_UNIT-1",
    "fuel_type": "Gas",
    "affected_mw": 200.0,
    "normal_capacity_mw": 500.0,
    "event_type": "Outage",
    "cause": "Planned",
    "message_description": "Test event for persistence tests.",
}

# Full column set that extract_and_persist requires (from the ``required``
# tuple in persistence.py — these are the eleven columns it checks).
_REQUIRED_COLUMNS = (
    "mrid",
    "revision_number",
    "message_status",
    "published_at",
    "effective_from",
    "effective_to",
    "fuel_type",
    "affected_mw",
    "event_type",
    "cause",
    "message_description",
)


def _make_remit_row(mrid: str, revision_number: int, **kwargs: Any) -> dict[str, Any]:
    """Return a single REMIT row dict with caller overrides on top of defaults."""
    row = dict(_REMIT_DEFAULTS)
    row["mrid"] = mrid
    row["revision_number"] = revision_number
    row.update(kwargs)
    return row


def _make_remit_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a REMIT-schema DataFrame with UTC-aware timestamps."""
    df = pd.DataFrame(rows)
    for col in ("published_at", "effective_from", "effective_to", "retrieved_at_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _stub_remit_df() -> pd.DataFrame:
    """Return a 10-row REMIT DataFrame from the canonical stub records.

    Builds the same rows as ``bristol_ml.ingestion.remit._stub_records()``
    and returns them with UTC-aware timestamps.  Used by test_1 and
    test_2 to avoid needing to monkeypatch fetch() and call disk I/O.
    """
    # Import directly from the internal helper — we are in a unit test; the
    # stub fixture is canonical and stable.  The ingestion module makes this
    # public-enough that Stage 13 and Stage 16 both reference it.
    from bristol_ml.ingestion.remit import _stub_records  # type: ignore[attr-defined]

    records = _stub_records()
    df = pd.DataFrame(records)
    for col in ("published_at", "effective_from", "effective_to"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    # ``_stub_records`` does not include ``retrieved_at_utc`` (the per-fetch
    # provenance scalar); add a constant so the DataFrame is complete for
    # ``extract_and_persist`` (which only requires the eleven _REQUIRED_COLUMNS,
    # not retrieved_at_utc).
    if "retrieved_at_utc" not in df.columns:
        df["retrieved_at_utc"] = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    return df


def _empty_remit_df() -> pd.DataFrame:
    """Return a zero-row REMIT DataFrame carrying the required column structure.

    Mirrors the ``_empty_remit_df`` helper in
    ``tests/unit/features/test_remit.py``.
    """
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
# Test 1 — Stub-mode parquet round-trip
#
# Plan T3 acceptance criterion:
#   "stub-mode parquet round-trip: … assert the path exists and
#    load_extracted(path) returns a DataFrame with the correct columns +
#    dtypes (assert against EXTRACTED_OUTPUT_SCHEMA). Assert len(df) == 10
#    and that (mrid, revision_number) matches the input keys."
# ---------------------------------------------------------------------------


def test_stub_mode_parquet_round_trip(tmp_path: Path) -> None:
    """Pins plan T3 AC: stub-mode produces a schema-conformant 10-row parquet.

    Uses the canonical 10-row REMIT stub fixture (the same records that
    ``BRISTOL_ML_REMIT_STUB=1`` causes ``remit.fetch`` to write).  The
    ``StubExtractor`` makes zero network calls.

    Asserts:
    - The output path exists after the call.
    - ``load_extracted(path)`` returns a DataFrame.
    - The schema equals ``EXTRACTED_OUTPUT_SCHEMA`` (11 columns, exact types).
    - Exactly 10 rows — one per input row.
    - The ``(mrid, revision_number)`` values in the output match the input.
    """
    remit_df = _stub_remit_df()
    assert len(remit_df) == 10, "Stub fixture must have 10 rows."

    output_path = tmp_path / "out.parquet"
    extractor = StubExtractor()

    returned_path = extract_and_persist(extractor, remit_df, output_path=output_path)

    # Path contract.
    assert returned_path == output_path, (
        f"extract_and_persist must return the output path; "
        f"got {returned_path}, expected {output_path}."
    )
    assert output_path.exists(), f"Output parquet not written to {output_path}."

    # Round-trip via load_extracted.
    df = load_extracted(output_path)
    assert isinstance(df, pd.DataFrame)

    # Row count.
    assert len(df) == 10, f"Expected 10 rows; got {len(df)}."

    # Schema conformance against the canonical constant.
    table = pq.read_table(output_path)
    assert table.schema == EXTRACTED_OUTPUT_SCHEMA, (
        f"On-disk schema does not match EXTRACTED_OUTPUT_SCHEMA.\n"
        f"Got:      {table.schema}\n"
        f"Expected: {EXTRACTED_OUTPUT_SCHEMA}"
    )

    # (mrid, revision_number) keys must match the input.
    input_keys = set(
        zip(
            remit_df["mrid"].astype(str),
            remit_df["revision_number"].astype(int),
            strict=True,
        )
    )
    output_keys = set(zip(df["mrid"].astype(str), df["revision_number"].astype(int), strict=True))
    assert output_keys == input_keys, (
        f"Output (mrid, revision_number) keys do not match input.\n"
        f"Missing from output: {input_keys - output_keys}\n"
        f"Extra in output:     {output_keys - input_keys}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Idempotent re-write
#
# Plan T3 acceptance criterion (NFR-3):
#   "Both calls succeed; the file is overwritten cleanly (no stray .tmp
#    file in the parent directory after the second call)."
# ---------------------------------------------------------------------------


def test_idempotent_rewrite_no_stray_tmp(tmp_path: Path) -> None:
    """Pins plan NFR-3: calling extract_and_persist twice produces no stray .tmp file.

    Atomic writes use ``path.with_suffix('.parquet.tmp')`` as the temporary
    file; after a successful ``os.replace`` the ``.tmp`` is gone.  A second
    call against the same output path must also leave no ``.tmp`` artefact.
    """
    remit_df = _stub_remit_df()
    output_path = tmp_path / "out.parquet"
    extractor = StubExtractor()

    # First call.
    extract_and_persist(extractor, remit_df, output_path=output_path)
    assert output_path.exists()

    # Second call — idempotent overwrite.
    extract_and_persist(extractor, remit_df, output_path=output_path)
    assert output_path.exists()

    # No stray .tmp files anywhere in the parent directory.
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert not tmp_files, (
        f"Stray .tmp file(s) left after second call: {tmp_files}. "
        "The atomic write helper must clean up unconditionally."
    )

    # The parquet is still valid after the re-write.
    df = load_extracted(output_path)
    assert len(df) == 10


# ---------------------------------------------------------------------------
# Test 3 — Empty corpus
#
# Plan T3 criterion:
#   "file exists; load_extracted(path) returns a zero-row frame with the
#    correct 11-column schema; the function does not raise."
# ---------------------------------------------------------------------------


def test_empty_corpus_writes_zero_row_schema_conformant_parquet(
    tmp_path: Path,
) -> None:
    """Pins plan T3 AC: empty REMIT input produces a zero-row, schema-conformant parquet.

    An empty corpus is a valid input — CI dry-runs and the first build
    before any REMIT data exists.  The function must not raise and must
    produce a parquet that passes ``load_extracted``.
    """
    remit_df = _empty_remit_df()
    assert len(remit_df) == 0, "Fixture must be zero-row."

    output_path = tmp_path / "empty.parquet"
    extractor = StubExtractor()

    # Must not raise.
    extract_and_persist(extractor, remit_df, output_path=output_path)

    assert output_path.exists(), f"Output parquet not written to {output_path}."

    df = load_extracted(output_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0, f"Expected zero rows; got {len(df)}."

    # Schema must still be the 11-column EXTRACTED_OUTPUT_SCHEMA.
    table = pq.read_table(output_path)
    assert table.schema == EXTRACTED_OUTPUT_SCHEMA, (
        f"Zero-row parquet schema does not match EXTRACTED_OUTPUT_SCHEMA.\n"
        f"Got:      {table.schema}\n"
        f"Expected: {EXTRACTED_OUTPUT_SCHEMA}"
    )
    assert len(df.columns) == 11, (
        f"Schema must have 11 columns; got {len(df.columns)}: {list(df.columns)}."
    )


# ---------------------------------------------------------------------------
# Test 4 — load_extracted schema rejection
#
# Plan T3 criterion:
#   "Write an arbitrary parquet with the wrong shape (e.g. one column
#    wrong: int32) to a temp path, then call load_extracted(path) and assert
#    ValueError with a message naming the missing column. Also test the
#    extra-column case."
# ---------------------------------------------------------------------------


def test_load_extracted_rejects_missing_column(tmp_path: Path) -> None:
    """Pins plan T3 AC: load_extracted raises ValueError naming a missing column.

    Writes a one-column parquet with ``wrong: int32`` — missing all 11
    columns of EXTRACTED_OUTPUT_SCHEMA.  Expects a ValueError that names
    at least one of the missing columns.
    """
    bad_path = tmp_path / "bad.parquet"
    bad_table = pa.table({"wrong": pa.array([1, 2, 3], type=pa.int32())})
    pq.write_table(bad_table, bad_path)

    with pytest.raises(ValueError) as excinfo:
        load_extracted(bad_path)

    # The error message must name at least one of the expected columns.
    # The implementation names the *first* missing column it encounters.
    err_msg = str(excinfo.value)
    expected_columns = {field.name for field in EXTRACTED_OUTPUT_SCHEMA}
    named_in_msg = any(col in err_msg for col in expected_columns)
    assert named_in_msg, (
        f"ValueError message must name a missing column from EXTRACTED_OUTPUT_SCHEMA; "
        f"got: {err_msg!r}. Expected one of: {sorted(expected_columns)}."
    )


def test_load_extracted_rejects_extra_column(tmp_path: Path) -> None:
    """Pins plan T3 AC: load_extracted raises ValueError on extra columns.

    Takes the round-trip output (which passes load_extracted), appends an
    extra column, writes back, and asserts that load_extracted rejects it.
    The schema is exact — downstream joins rely on this invariant.
    """
    remit_df = _stub_remit_df()
    output_path = tmp_path / "round_trip.parquet"
    extractor = StubExtractor()
    extract_and_persist(extractor, remit_df, output_path=output_path)

    # Verify the clean round-trip passes.
    _ = load_extracted(output_path)

    # Inject an extra column.
    table = pq.read_table(output_path)
    extra_col = pa.array([0] * table.num_rows, type=pa.int32())
    wide_table = table.append_column(pa.field("extra_intruder", pa.int32()), extra_col)
    extra_path = tmp_path / "extra_column.parquet"
    pq.write_table(wide_table, extra_path)

    with pytest.raises(ValueError) as excinfo:
        load_extracted(extra_path)

    err_msg = str(excinfo.value)
    assert "extra_intruder" in err_msg, (
        f"ValueError for extra columns must name the offending column(s); got: {err_msg!r}."
    )


# ---------------------------------------------------------------------------
# Test 5 — Missing required REMIT column raises ValueError
#
# Plan T3 criterion:
#   "Pass a REMIT frame missing one of the required columns (e.g. drop
#    ``cause``); assert ValueError naming the missing column."
# ---------------------------------------------------------------------------


def test_missing_required_remit_column_raises_valueerror(tmp_path: Path) -> None:
    """Pins plan T3 AC: REMIT frame missing a required column raises ValueError.

    ``extract_and_persist`` validates the input frame against the eleven
    required columns before calling the extractor.  A frame that lacks
    ``cause`` (one of those columns) must produce a ValueError that names
    the missing column so the caller can diagnose the problem immediately.

    This test also covers the case where the frame looks structurally valid
    (correct number of rows, other columns present) but is missing exactly
    one required column — the most likely accidental scenario for a caller
    that builds frames from partial data.
    """
    remit_df = _stub_remit_df()
    assert "cause" in remit_df.columns, "Sanity: stub fixture must have 'cause' column."

    # Drop the required column.
    remit_df_missing = remit_df.drop(columns=["cause"])
    output_path = tmp_path / "should_not_exist.parquet"
    extractor = StubExtractor()

    with pytest.raises(ValueError) as excinfo:
        extract_and_persist(extractor, remit_df_missing, output_path=output_path)

    err_msg = str(excinfo.value)
    assert "cause" in err_msg, f"ValueError must name the missing column 'cause'; got: {err_msg!r}."

    # The output file must not have been written — a clean failure.
    assert not output_path.exists(), (
        "extract_and_persist must not write any output when the input frame "
        "is invalid; partial writes violate NFR-3."
    )
