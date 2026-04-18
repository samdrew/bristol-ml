"""Implementer-derived unit tests for bristol_ml.ingestion.neso private helpers.

These tests cover the structural choices the implementer made and are not
a substitute for the spec-derived acceptance tests (those live in
``tests/integration/ingestion/`` and ``tests/unit/ingestion/`` where the
tester places them). Keep this file narrow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.ingestion.neso import (
    OUTPUT_SCHEMA,
    _assert_schema,
    _atomic_write,
    _autumn_fallback_dates,
    _parse_settlement_date,
    _spring_forward_dates,
    _to_utc,
)


def test_autumn_fallback_date_2024() -> None:
    assert _autumn_fallback_dates([2024])[2024] == pd.Timestamp("2024-10-27")


def test_spring_forward_date_2024() -> None:
    assert _spring_forward_dates([2024])[2024] == pd.Timestamp("2024-03-31")


def test_parse_settlement_date_nesco_dd_mmm_yy() -> None:
    parsed = _parse_settlement_date(pd.Series(["01-Jan-23", "15-Dec-23"]))
    assert list(parsed) == [pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-15").date()]


def test_parse_settlement_date_iso() -> None:
    parsed = _parse_settlement_date(pd.Series(["2023-01-01", "2023-12-15"]))
    assert list(parsed) == [pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-15").date()]


def test_parse_settlement_date_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        _parse_settlement_date(pd.Series(["nonsense-date"]))


def _make_raw(date: str, periods: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SETTLEMENT_DATE": date, "SETTLEMENT_PERIOD": p, "ND": 20_000 + p, "TSD": 21_000 + p}
            for p in periods
        ]
    )


def test_to_utc_autumn_fallback_produces_contiguous_30min_steps() -> None:
    df = _make_raw("2024-10-27", list(range(1, 51)))
    out = _to_utc(_assert_schema(df, year=2024))
    deltas = out["timestamp_utc"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=30)


def test_to_utc_spring_forward_produces_contiguous_30min_steps() -> None:
    df = _make_raw("2024-03-31", list(range(1, 47)))
    out = _to_utc(_assert_schema(df, year=2024))
    deltas = out["timestamp_utc"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=30)


def test_to_utc_autumn_periods_3_and_5_resolve_to_distinct_utc() -> None:
    df = _make_raw("2024-10-27", [3, 5])
    out = _to_utc(_assert_schema(df, year=2024))
    # Period 3 = first 01:00 BST = 00:00 UTC
    # Period 5 = second 01:00 GMT = 01:00 UTC
    delta = out.iloc[1]["timestamp_utc"] - out.iloc[0]["timestamp_utc"]
    assert delta == pd.Timedelta(hours=1)


def test_to_utc_rejects_period_47_on_spring_forward_day() -> None:
    df = _make_raw("2024-03-31", [47])
    with pytest.raises(ValueError, match="out of the valid per-day range"):
        _to_utc(_assert_schema(df, year=2024))


def test_to_utc_rejects_period_51_on_autumn_fallback_day() -> None:
    df = _make_raw("2024-10-27", [51])
    with pytest.raises(ValueError, match="out of the valid per-day range"):
        _to_utc(_assert_schema(df, year=2024))


def test_assert_schema_rejects_missing_required_column() -> None:
    df = pd.DataFrame({"SETTLEMENT_DATE": ["2024-06-15"], "SETTLEMENT_PERIOD": [1], "ND": [100]})
    with pytest.raises(KeyError, match="TSD"):
        _assert_schema(df, year=2024)


def test_assert_schema_warns_on_unknown_column_and_drops_it() -> None:
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": ["2024-06-15"],
            "SETTLEMENT_PERIOD": [1],
            "ND": [100],
            "TSD": [110],
            "EMBEDDED_SOLAR_GENERATION": [50],
        }
    )
    with pytest.warns(UserWarning, match="EMBEDDED_SOLAR_GENERATION"):
        out = _assert_schema(df, year=2024)
    assert "EMBEDDED_SOLAR_GENERATION" not in out.columns


def test_atomic_write_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int32())})
    target = tmp_path / "out.parquet"
    _atomic_write(table, target)
    assert target.exists()
    # The .tmp sibling must be cleaned up.
    assert not (tmp_path / "out.parquet.tmp").exists()
    # Round-trip.
    assert pq.read_table(target).column("x").to_pylist() == [1, 2, 3]


def test_atomic_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    target.write_bytes(b"old content not a real parquet")
    table = pa.table({"x": pa.array([42], type=pa.int32())})
    _atomic_write(table, target)
    assert pq.read_table(target).column("x").to_pylist() == [42]


def test_output_schema_is_fully_specified() -> None:
    """Sanity: every column has a concrete (non-null) arrow type."""
    for field in OUTPUT_SCHEMA:
        assert field.type is not None
    assert OUTPUT_SCHEMA.field("timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert OUTPUT_SCHEMA.field("nd_mw").type == pa.int32()
    assert OUTPUT_SCHEMA.field("settlement_period").type == pa.int8()
