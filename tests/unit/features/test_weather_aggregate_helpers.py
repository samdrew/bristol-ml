"""Implementer-derived unit tests for bristol_ml.features.weather helpers.

These tests cover the implementer's choices around subset semantics and
NaN handling. The spec-derived acceptance tests — equal-weight identity
(AC6) and station-subset (AC3) — are the tester's responsibility and
live in a separate file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bristol_ml.features.weather import national_aggregate


def _long_frame(
    timestamps: list[str],
    stations: list[str],
    temperatures: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []
    for station in stations:
        for t, temp in zip(timestamps, temperatures[station], strict=True):
            rows.append(
                {
                    "timestamp_utc": pd.Timestamp(t, tz="UTC"),
                    "station": station,
                    "temperature_2m": temp,
                    "dew_point_2m": temp - 2.0,
                    "wind_speed_10m": 5.0,
                    "cloud_cover": 50,
                    "shortwave_radiation": 0.0,
                    "retrieved_at_utc": pd.Timestamp("2026-04-18T00:00:00", tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def test_missing_station_in_frame_raises_with_clear_message() -> None:
    df = _long_frame(
        ["2023-01-01T00:00", "2023-01-01T01:00"],
        ["london", "bristol"],
        {"london": [4.0, 4.5], "bristol": [6.0, 6.2]},
    )
    with pytest.raises(ValueError, match="mars"):
        national_aggregate(df, {"london": 1.0, "mars": 1.0})


def test_empty_weights_raises() -> None:
    df = _long_frame(["2023-01-01T00:00"], ["london"], {"london": [4.0]})
    with pytest.raises(ValueError, match="non-empty"):
        national_aggregate(df, {})


def test_negative_weight_raises() -> None:
    df = _long_frame(["2023-01-01T00:00"], ["london"], {"london": [4.0]})
    with pytest.raises(ValueError, match="positive"):
        national_aggregate(df, {"london": -1.0})


def test_missing_station_column_raises() -> None:
    df = pd.DataFrame(
        {"timestamp_utc": [pd.Timestamp("2023-01-01T00:00", tz="UTC")], "temperature_2m": [4.0]}
    )
    with pytest.raises(ValueError, match="'station'"):
        national_aggregate(df, {"london": 1.0})


def test_missing_timestamp_column_raises() -> None:
    df = pd.DataFrame({"station": ["london"], "temperature_2m": [4.0]})
    with pytest.raises(ValueError, match="timestamp_utc"):
        national_aggregate(df, {"london": 1.0})


def test_nan_at_one_station_drops_that_station_from_that_hour() -> None:
    df = _long_frame(
        ["2023-01-01T00:00"],
        ["london", "bristol"],
        {"london": [np.nan], "bristol": [10.0]},
    )
    out = national_aggregate(df, {"london": 2.0, "bristol": 1.0})
    # London NaN drops; bristol alone contributes, renormalised to weight 1.
    assert out.loc[pd.Timestamp("2023-01-01T00:00", tz="UTC"), "temperature_2m"] == pytest.approx(
        10.0
    )


def test_all_nan_at_hour_propagates_nan() -> None:
    df = _long_frame(
        ["2023-01-01T00:00"],
        ["london", "bristol"],
        {"london": [np.nan], "bristol": [np.nan]},
    )
    out = national_aggregate(df, {"london": 1.0, "bristol": 1.0})
    result = out.loc[pd.Timestamp("2023-01-01T00:00", tz="UTC"), "temperature_2m"]
    assert pd.isna(result)


def test_station_in_frame_absent_from_weights_is_silently_excluded() -> None:
    # Three stations in the frame, only two weighted. The weighted mean must
    # reflect the two passed weights, not all three.
    df = _long_frame(
        ["2023-01-01T00:00"],
        ["london", "bristol", "edinburgh"],
        {"london": [10.0], "bristol": [20.0], "edinburgh": [999.0]},
    )
    out = national_aggregate(df, {"london": 1.0, "bristol": 3.0})
    expected = (10.0 * 1.0 + 20.0 * 3.0) / 4.0
    got = out.loc[pd.Timestamp("2023-01-01T00:00", tz="UTC"), "temperature_2m"]
    assert got == pytest.approx(expected)


def test_wide_output_has_timestamp_index_and_variable_columns() -> None:
    df = _long_frame(
        ["2023-01-01T00:00", "2023-01-01T01:00"],
        ["london", "bristol"],
        {"london": [4.0, 4.5], "bristol": [6.0, 6.5]},
    )
    out = national_aggregate(df, {"london": 1.0, "bristol": 1.0})
    assert out.index.name == "timestamp_utc"
    # Variable columns — cloud_cover included, retrieved_at_utc excluded.
    assert "temperature_2m" in out.columns
    assert "cloud_cover" in out.columns
    assert "retrieved_at_utc" not in out.columns
    assert "station" not in out.columns
    # Sorted ascending.
    assert out.index.is_monotonic_increasing


def test_renormalisation_independent_of_weight_magnitude() -> None:
    df = _long_frame(
        ["2023-01-01T00:00"],
        ["london", "bristol"],
        {"london": [10.0], "bristol": [20.0]},
    )
    a = national_aggregate(df, {"london": 1.0, "bristol": 1.0})
    b = national_aggregate(df, {"london": 1_000_000.0, "bristol": 1_000_000.0})
    pd.testing.assert_frame_equal(a, b)
