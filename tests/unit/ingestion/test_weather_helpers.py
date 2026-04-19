"""Implementer-derived unit tests for bristol_ml.ingestion.weather private helpers.

These tests cover the structural choices the implementer made — payload
parsing, schema-assertion behaviour, arrow cast rules — and are not a
substitute for the spec-derived acceptance tests (those live alongside in
``tests/`` under the tester's ownership). Keep this file narrow.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from bristol_ml.ingestion import weather as weather_mod
from bristol_ml.ingestion._common import (
    CacheMissingError,
    CachePathConfig,
    RateLimitConfig,
    RetryConfig,
)
from conf._schemas import WeatherIngestionConfig, WeatherStation

STATION = WeatherStation(
    name="london",
    latitude=51.5074,
    longitude=-0.1278,
    weight=9_787_426,
    weight_source="test",
)


def _mini_payload(hours: int = 3) -> dict:
    return {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "GMT",
        "hourly": {
            "time": [f"2023-01-01T{h:02d}:00" for h in range(hours)],
            "temperature_2m": [4.0 + h * 0.1 for h in range(hours)],
            "dew_point_2m": [2.0 + h * 0.1 for h in range(hours)],
            "wind_speed_10m": [5.0 + h * 0.2 for h in range(hours)],
            "cloud_cover": [90 - h * 5 for h in range(hours)],
            "shortwave_radiation": [0.0 + h for h in range(hours)],
        },
    }


def _build_config(tmp_path: Path) -> WeatherIngestionConfig:
    return WeatherIngestionConfig(
        stations=[STATION],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 31),
        cache_dir=tmp_path,
    )


# --------------------------------------------------------------------------- #
# _parse_station_payload
# --------------------------------------------------------------------------- #


def test_parse_station_payload_zips_time_with_variable_arrays() -> None:
    df = weather_mod._parse_station_payload(_mini_payload(hours=4), STATION)
    assert list(df.columns).count("time") == 1
    assert list(df["station"].unique()) == ["london"]
    assert len(df) == 4
    assert list(df["temperature_2m"]) == [4.0, 4.1, 4.2, 4.3]


def test_parse_station_payload_raises_when_hourly_missing() -> None:
    payload: dict = {"latitude": 51.5, "longitude": -0.1}
    with pytest.raises((RuntimeError, KeyError)):
        weather_mod._parse_station_payload(payload, STATION)


def test_parse_station_payload_raises_on_length_mismatch() -> None:
    payload = _mini_payload(hours=3)
    payload["hourly"]["temperature_2m"] = [1.0, 2.0]  # one short
    with pytest.raises(RuntimeError, match="temperature_2m"):
        weather_mod._parse_station_payload(payload, STATION)


# --------------------------------------------------------------------------- #
# _assert_schema
# --------------------------------------------------------------------------- #


def test_assert_schema_missing_requested_variable_raises() -> None:
    payload = _mini_payload(hours=3)
    payload["hourly"].pop("shortwave_radiation")
    df = weather_mod._parse_station_payload(payload, STATION)
    with pytest.raises(KeyError, match="shortwave_radiation"):
        weather_mod._assert_schema(
            df,
            STATION,
            [
                "temperature_2m",
                "dew_point_2m",
                "wind_speed_10m",
                "cloud_cover",
                "shortwave_radiation",
            ],
        )


def test_assert_schema_unknown_variable_warns_and_drops_it() -> None:
    payload = _mini_payload(hours=3)
    payload["hourly"]["relative_humidity_2m"] = [90, 88, 85]  # unknown to us
    df = weather_mod._parse_station_payload(payload, STATION)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = weather_mod._assert_schema(
            df,
            STATION,
            [
                "temperature_2m",
                "dew_point_2m",
                "wind_speed_10m",
                "cloud_cover",
                "shortwave_radiation",
            ],
        )
    assert any("relative_humidity_2m" in str(w.message) for w in caught), (
        f"Expected a warning naming 'relative_humidity_2m'; got {[str(w.message) for w in caught]}"
    )
    assert "relative_humidity_2m" not in out.columns


def test_assert_schema_materialises_tz_aware_utc_column() -> None:
    df = weather_mod._parse_station_payload(_mini_payload(hours=3), STATION)
    out = weather_mod._assert_schema(
        df,
        STATION,
        [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ],
    )
    assert "timestamp_utc" in out.columns
    tz = out["timestamp_utc"].dt.tz
    assert tz is not None
    assert str(tz) in {"UTC", "utc"}


# --------------------------------------------------------------------------- #
# _to_arrow
# --------------------------------------------------------------------------- #


def test_to_arrow_output_schema_matches_declared() -> None:
    df = weather_mod._parse_station_payload(_mini_payload(hours=2), STATION)
    cleaned = weather_mod._assert_schema(
        df,
        STATION,
        [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ],
    )
    cleaned["retrieved_at_utc"] = pd.Timestamp("2026-04-18T00:00:00", tz="UTC")
    table = weather_mod._to_arrow(cleaned)
    assert table.schema == weather_mod.OUTPUT_SCHEMA


# --------------------------------------------------------------------------- #
# fetch / OFFLINE
# --------------------------------------------------------------------------- #


def test_fetch_offline_raises_cache_missing_error(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    with pytest.raises(CacheMissingError):
        weather_mod.fetch(cfg, cache=weather_mod.CachePolicy.OFFLINE)


def test_fetch_auto_returns_cache_path_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _build_config(tmp_path)
    cache_path = tmp_path / cfg.cache_filename
    # Build a minimal valid long-form parquet via _to_arrow so the schema check passes.
    stub_df = pd.DataFrame(
        {
            "time": ["2023-01-01T00:00"],
            "station": ["london"],
            "temperature_2m": [1.0],
            "dew_point_2m": [0.0],
            "wind_speed_10m": [5.0],
            "cloud_cover": [50],
            "shortwave_radiation": [0.0],
        }
    )
    cleaned = weather_mod._assert_schema(
        stub_df,
        STATION,
        [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ],
    )
    cleaned["retrieved_at_utc"] = pd.Timestamp("2026-04-18T00:00:00", tz="UTC")
    import pyarrow.parquet as pq

    pq.write_table(weather_mod._to_arrow(cleaned), cache_path)

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("AUTO + cache present must not touch the network")

    import httpx

    monkeypatch.setattr(httpx.Client, "get", _explode, raising=False)
    monkeypatch.setattr(weather_mod, "_fetch_station", _explode)
    returned = weather_mod.fetch(cfg, cache=weather_mod.CachePolicy.AUTO)
    assert Path(returned) == cache_path


def test_fetch_refresh_persists_with_monkeypatched_station_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _build_config(tmp_path)

    def _fake_fetch_station(
        station: WeatherStation, config: WeatherIngestionConfig, *, client: object = None
    ) -> dict:
        return _mini_payload(hours=4)

    monkeypatch.setattr(weather_mod, "_fetch_station", _fake_fetch_station)
    path = weather_mod.fetch(cfg, cache=weather_mod.CachePolicy.REFRESH)
    assert path.exists()
    loaded = weather_mod.load(path)
    # Exactly one station x four hours.
    assert len(loaded) == 4
    assert set(loaded["station"].unique()) == {"london"}
    assert loaded["timestamp_utc"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# _common Protocol structural satisfaction
# --------------------------------------------------------------------------- #


def test_weather_config_satisfies_common_protocols(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    assert isinstance(cfg, RetryConfig)
    assert isinstance(cfg, RateLimitConfig)
    assert isinstance(cfg, CachePathConfig)


# --------------------------------------------------------------------------- #
# OUTPUT_SCHEMA surface
# --------------------------------------------------------------------------- #


def test_output_schema_is_fully_specified() -> None:
    for field in weather_mod.OUTPUT_SCHEMA:
        assert field.type is not None
    assert weather_mod.OUTPUT_SCHEMA.field("timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert weather_mod.OUTPUT_SCHEMA.field("cloud_cover").type == pa.int8()
    assert weather_mod.OUTPUT_SCHEMA.field("temperature_2m").type == pa.float32()
