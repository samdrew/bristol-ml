"""Boundary tests for the Stage 16 ``assemble_with_remit`` / ``load_with_remit`` pair.

Mirrors :mod:`tests.unit.features.test_assembler_calendar` on the 59-column
:data:`bristol_ml.features.assembler.WITH_REMIT_OUTPUT_SCHEMA`.  Stage 16
plan §6 T4 + reviewer T-1: the assembler ships two new public entry points
(``assemble_with_remit``, ``load_with_remit``) plus the schema constant; the
load surface is the boundary at which the cross-feature-set mutual
exclusivity is enforced (a ``weather_calendar`` parquet must be rejected by
``load_with_remit`` because its REMIT columns are absent, and vice versa).

This file deliberately covers only the **load + schema** boundary.  The full
``assemble_with_remit`` orchestrator is exercised end-to-end by the
``test_notebook_04_executes_top_to_bottom`` integration test under stub mode
(via the auto-run-extractor fallback in the assembler), and would otherwise
require a synthetic NESO + weather + holidays + REMIT cache stack — which is
the domain of the test_assembler_calendar module already.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.features import assembler as assembler_mod

_START_UTC = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
_HOURS = 48


def _make_with_remit_frame() -> pd.DataFrame:
    """Return a 48-row DataFrame conforming to ``WITH_REMIT_OUTPUT_SCHEMA``.

    Synthetic but schema-conformant: two days of hourly UTC timestamps,
    constant demand / weather / calendar columns, and zero-valued REMIT
    columns (the typical no-event hour case which AC-7 mandates is a
    valid output state).
    """
    timestamps = pd.date_range(start=_START_UTC, periods=_HOURS, freq="h", tz="UTC")
    retrieved = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")

    rows: dict[str, object] = {
        "timestamp_utc": timestamps,
        "nd_mw": pd.Series([30_000] * _HOURS, dtype="int32"),
        "tsd_mw": pd.Series([31_000] * _HOURS, dtype="int32"),
        "temperature_2m": pd.Series([5.0] * _HOURS, dtype="float32"),
        "dew_point_2m": pd.Series([3.0] * _HOURS, dtype="float32"),
        "wind_speed_10m": pd.Series([10.0] * _HOURS, dtype="float32"),
        "cloud_cover": pd.Series([50.0] * _HOURS, dtype="float32"),
        "shortwave_radiation": pd.Series([100.0] * _HOURS, dtype="float32"),
        "neso_retrieved_at_utc": [retrieved] * _HOURS,
        "weather_retrieved_at_utc": [retrieved] * _HOURS,
    }
    # 44 calendar columns (int8); zero-fill the constant case.
    for name, _dtype in assembler_mod.CALENDAR_VARIABLE_COLUMNS:
        rows[name] = pd.Series([0] * _HOURS, dtype="int8")
    rows["holidays_retrieved_at_utc"] = [retrieved] * _HOURS
    # Three REMIT columns + provenance (zero-event hours, AC-7 invariant).
    rows["remit_unavail_mw_total"] = pd.Series([0.0] * _HOURS, dtype="float32")
    rows["remit_active_unplanned_count"] = pd.Series([0] * _HOURS, dtype="int32")
    rows["remit_unavail_mw_next_24h"] = pd.Series([0.0] * _HOURS, dtype="float32")
    rows["remit_retrieved_at_utc"] = [retrieved] * _HOURS

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — load_with_remit round-trips a schema-compliant parquet
# ---------------------------------------------------------------------------


def test_load_with_remit_round_trips_compliant_parquet(tmp_path: Path) -> None:
    """A schema-compliant 59-column parquet round-trips through load_with_remit.

    Pins the AC-2 boundary contract: a parquet written under
    ``WITH_REMIT_OUTPUT_SCHEMA`` is loadable verbatim, with the column order
    + dtypes preserved.
    """
    frame = _make_with_remit_frame()
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(
        assembler_mod.WITH_REMIT_OUTPUT_SCHEMA, safe=True
    )
    out_path = tmp_path / "with_remit.parquet"
    pq.write_table(table, out_path)

    loaded = assembler_mod.load_with_remit(out_path)
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded.columns) == assembler_mod.WITH_REMIT_OUTPUT_SCHEMA.names, (
        "Loaded with_remit frame columns must exactly match "
        "WITH_REMIT_OUTPUT_SCHEMA.names in order."
    )
    assert len(loaded) == _HOURS, f"Expected {_HOURS} rows; got {len(loaded)}."


# ---------------------------------------------------------------------------
# Test 2 — load_with_remit rejects a weather_calendar parquet
# ---------------------------------------------------------------------------


def test_load_with_remit_rejects_weather_calendar_schema(tmp_path: Path) -> None:
    """load_with_remit must raise ValueError naming a missing REMIT column.

    A parquet written under CALENDAR_OUTPUT_SCHEMA (55 columns) is missing
    every REMIT column; load_with_remit's exact-schema check must reject it
    and name at least one missing column so the caller can diagnose the
    mismatch without inspecting source.
    """
    timestamps = pd.date_range(start=_START_UTC, periods=_HOURS, freq="h", tz="UTC")
    retrieved = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    rows: dict[str, object] = {
        "timestamp_utc": timestamps,
        "nd_mw": pd.Series([30_000] * _HOURS, dtype="int32"),
        "tsd_mw": pd.Series([31_000] * _HOURS, dtype="int32"),
        "temperature_2m": pd.Series([5.0] * _HOURS, dtype="float32"),
        "dew_point_2m": pd.Series([3.0] * _HOURS, dtype="float32"),
        "wind_speed_10m": pd.Series([10.0] * _HOURS, dtype="float32"),
        "cloud_cover": pd.Series([50.0] * _HOURS, dtype="float32"),
        "shortwave_radiation": pd.Series([100.0] * _HOURS, dtype="float32"),
        "neso_retrieved_at_utc": [retrieved] * _HOURS,
        "weather_retrieved_at_utc": [retrieved] * _HOURS,
    }
    for name, _dtype in assembler_mod.CALENDAR_VARIABLE_COLUMNS:
        rows[name] = pd.Series([0] * _HOURS, dtype="int8")
    rows["holidays_retrieved_at_utc"] = [retrieved] * _HOURS

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False).cast(
        assembler_mod.CALENDAR_OUTPUT_SCHEMA, safe=True
    )
    wc_path = tmp_path / "weather_calendar.parquet"
    pq.write_table(table, wc_path)

    with pytest.raises(ValueError) as exc_info:
        assembler_mod.load_with_remit(wc_path)

    error_msg = str(exc_info.value)
    sentinel_columns = (
        "remit_unavail_mw_total",
        "remit_active_unplanned_count",
        "remit_unavail_mw_next_24h",
        "remit_retrieved_at_utc",
    )
    assert any(sentinel in error_msg for sentinel in sentinel_columns), (
        "ValueError message must name at least one missing REMIT column "
        f"({sentinel_columns!r}). Got: {error_msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — load_calendar rejects a with_remit parquet (the inverse direction)
# ---------------------------------------------------------------------------


def test_load_calendar_rejects_with_remit_schema(tmp_path: Path) -> None:
    """load_calendar must raise ValueError when given a with_remit parquet.

    Pins the inverse contract: a parquet written under
    WITH_REMIT_OUTPUT_SCHEMA carries four extra columns (the three REMIT
    columns plus ``remit_retrieved_at_utc``) that load_calendar's
    exact-schema check must surface as 'unexpected column(s)'.
    """
    frame = _make_with_remit_frame()
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(
        assembler_mod.WITH_REMIT_OUTPUT_SCHEMA, safe=True
    )
    wr_path = tmp_path / "with_remit.parquet"
    pq.write_table(table, wr_path)

    with pytest.raises(ValueError) as exc_info:
        assembler_mod.load_calendar(wr_path)

    error_msg = str(exc_info.value)
    sentinel_columns = ("remit_unavail_mw_total", "remit_retrieved_at_utc")
    assert any(sentinel in error_msg for sentinel in sentinel_columns), (
        "ValueError message must name at least one extra REMIT-side column "
        f"({sentinel_columns!r}). Got: {error_msg!r}"
    )
