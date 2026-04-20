"""Spec-derived tests for the Stage 5 T4 assembler extension.

Tests pin the acceptance criteria from
``docs/plans/active/05-calendar-features.md`` §"Task T4 - Assembler
extension", lines 305-311.

Six tests:

1. ``test_calendar_output_schema_is_weather_schema_plus_calendar_plus_provenance``
   — structural invariant, no filesystem.
2. ``test_assemble_calendar_writes_parquet``          — AC-7
3. ``test_load_calendar_rejects_weather_only_schema`` — cross-schema rejection
4. ``test_load_rejects_weather_calendar_schema``      — reverse rejection
5. ``test_assemble_calendar_is_idempotent``           — §2.1.5
6. ``test_assemble_calendar_provenance_scalars_preserved`` — DESIGN §2.1.6

Approach
--------
Mirrors the ``primed_caches`` / ``_make_app_config`` pattern from
``test_assembler_cli.py``, extended with a holidays cache primed
programmatically from ``bristol_ml.ingestion.holidays.OUTPUT_SCHEMA``.

All timestamps are pinned to two full UTC days in 2023-01:
  2023-01-01 00:00 UTC → 2023-01-02 23:00 UTC (48 hours)

2023-01-02 is the observed bank holiday "New Year's Day (substitute
day)" for England & Wales — so ``is_bank_holiday_ew`` fires on at least
one row, exercising the holiday-column logic end-to-end.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conf._schemas import (
    AppConfig,
    EvaluationGroup,
    FeatureSetConfig,
    FeaturesGroup,
    HolidaysIngestionConfig,
    IngestionGroup,
    NesoIngestionConfig,
    NesoYearResource,
    ProjectConfig,
    WeatherIngestionConfig,
    WeatherStation,
)

assembler_mod = pytest.importorskip("bristol_ml.features.assembler")
neso_mod = pytest.importorskip("bristol_ml.ingestion.neso")
weather_mod = pytest.importorskip("bristol_ml.ingestion.weather")
holidays_mod = pytest.importorskip("bristol_ml.ingestion.holidays")


# ---------------------------------------------------------------------------
# UTC window used across all fixtures
# The 48-hour window 2023-01-01..2023-01-02 is pinned so:
#   - 2023-01-02 is "New Year's Day (substitute)" for England & Wales →
#     is_bank_holiday_ew fires on the 24 UTC hours that land on that date.
#   - 2023-01-02 is NOT a Scottish bank holiday (Scotland's New Year bank
#     holiday is 01-Jan and 02-Jan is the "2 January" Scottish holiday,
#     which is scotland-only) — so the fixture exercises both EW and SCO
#     paths with the same date range.
# ---------------------------------------------------------------------------

_START_UTC = pd.Timestamp("2023-01-01 00:00", tz="UTC")
_HOURS = 48  # exactly two full UTC days


# ---------------------------------------------------------------------------
# Cache-building helpers
# ---------------------------------------------------------------------------


def _make_neso_cache(path: Path) -> None:
    """Build a 96-row half-hourly NESO parquet spanning 2023-01-01/02 UTC."""
    periods = _HOURS * 2  # 48 half-hours per UTC day
    timestamps = pd.date_range(start=_START_UTC, periods=periods, freq="30min", tz="UTC")
    local = timestamps.tz_convert("Europe/London")
    retrieved = pd.Timestamp("2023-02-01 12:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "timestamp_local": local,
            "settlement_date": [ts.date() for ts in local],
            "settlement_period": [((i % 48) + 1) for i in range(periods)],
            "nd_mw": [30_000 + 10 * (i % 48) for i in range(periods)],
            "tsd_mw": [31_000 + 10 * (i % 48) for i in range(periods)],
            "source_year": [2023] * periods,
            "retrieved_at_utc": [retrieved] * periods,
        }
    )
    df["settlement_period"] = df["settlement_period"].astype("int8")
    df["nd_mw"] = df["nd_mw"].astype("int32")
    df["tsd_mw"] = df["tsd_mw"].astype("int32")
    df["source_year"] = df["source_year"].astype("int16")
    table = pa.Table.from_pandas(df, preserve_index=False).cast(neso_mod.OUTPUT_SCHEMA, safe=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_weather_cache(path: Path) -> None:
    """Build an hourly weather parquet for two stations, 2023-01-01/02 UTC."""
    timestamps = pd.date_range(start=_START_UTC, periods=_HOURS, freq="h", tz="UTC")
    stations = ("london", "bristol")
    retrieved = pd.Timestamp("2023-02-01 13:00:00", tz="UTC")
    rows: list[dict[str, object]] = []
    for station_idx, station in enumerate(stations):
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "station": station,
                    "temperature_2m": float(5.0 + 0.1 * i + station_idx),
                    "dew_point_2m": float(3.0 + 0.05 * i + station_idx),
                    "wind_speed_10m": float(10.0 + 0.2 * i),
                    "cloud_cover": int((80 + i) % 101),
                    "shortwave_radiation": float(max(0.0, 100.0 - abs(i - 12) * 8.0)),
                    "retrieved_at_utc": retrieved,
                }
            )
    df = pd.DataFrame(rows)
    df["temperature_2m"] = df["temperature_2m"].astype("float32")
    df["dew_point_2m"] = df["dew_point_2m"].astype("float32")
    df["wind_speed_10m"] = df["wind_speed_10m"].astype("float32")
    df["cloud_cover"] = df["cloud_cover"].astype("int8")
    df["shortwave_radiation"] = df["shortwave_radiation"].astype("float32")
    table = pa.Table.from_pandas(df, preserve_index=False).cast(
        weather_mod.OUTPUT_SCHEMA, safe=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_holidays_cache(path: Path) -> None:
    """Build a synthetic holidays parquet covering dates in the 2023-01 window.

    Rows:
    - 2023-01-02 england-and-wales  "New Year's Day (substitute day)"
    - 2023-01-02 scotland           "2nd January"  (2 January is Scotland-only)
    - 2023-01-02 northern-ireland   "New Year's Day (substitute day)"
    - 2022-12-27 england-and-wales  "Christmas Day (substitute day)" — pre-window
      entry exercises the D-6 proximity logic (day *after* this is 2022-12-28,
      not in our 2023-01 window — included so derive_calendar sees a full picture)

    Using ``bristol_ml.ingestion.holidays.OUTPUT_SCHEMA`` as the cast target
    ensures the fixture is schema-correct without hitting the network.
    """
    retrieved = pd.Timestamp("2023-02-01 09:00:00", tz="UTC")
    rows = [
        {
            "date": date(2023, 1, 2),
            "division": "england-and-wales",
            "title": "New Year's Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
        {
            "date": date(2023, 1, 2),
            "division": "scotland",
            "title": "2nd January",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
        {
            "date": date(2023, 1, 2),
            "division": "northern-ireland",
            "title": "New Year's Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
        {
            "date": date(2022, 12, 27),
            "division": "england-and-wales",
            "title": "Christmas Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
        {
            "date": date(2022, 12, 27),
            "division": "scotland",
            "title": "Christmas Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
        {
            "date": date(2022, 12, 27),
            "division": "northern-ireland",
            "title": "Christmas Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": retrieved,
        },
    ]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False).cast(
        holidays_mod.OUTPUT_SCHEMA, safe=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def primed_calendar_caches(tmp_path: Path) -> dict[str, Path]:
    """Return tmp paths for all four caches, NESO + weather + holidays pre-populated."""
    neso_dir = tmp_path / "raw" / "neso"
    weather_dir = tmp_path / "raw" / "weather"
    holidays_dir = tmp_path / "raw" / "holidays"
    features_dir = tmp_path / "features"

    neso_path = neso_dir / "neso_demand.parquet"
    weather_path = weather_dir / "weather.parquet"
    holidays_path = holidays_dir / "holidays.parquet"

    _make_neso_cache(neso_path)
    _make_weather_cache(weather_path)
    _make_holidays_cache(holidays_path)

    return {
        "neso_dir": neso_dir,
        "weather_dir": weather_dir,
        "holidays_dir": holidays_dir,
        "features_dir": features_dir,
        "neso_path": neso_path,
        "weather_path": weather_path,
        "holidays_path": holidays_path,
        "calendar_path": features_dir / "weather_calendar.parquet",
    }


def _make_calendar_app_config(paths: dict[str, Path]) -> AppConfig:
    """Construct an AppConfig with weather_calendar populated and weather_only=None.

    Mirrors the Hydra group-swap runtime state described in the plan's
    implementation finding (lines 484-488): under ``features=weather_calendar``
    exactly one of cfg.features.weather_only / cfg.features.weather_calendar
    is populated; the other is None.
    """
    return AppConfig(
        project=ProjectConfig(name="bristol_ml_calendar_test", seed=0),
        ingestion=IngestionGroup(
            neso=NesoIngestionConfig(
                resources=[NesoYearResource(year=2023, resource_id=uuid4())],
                cache_dir=paths["neso_dir"],
            ),
            weather=WeatherIngestionConfig(
                stations=[
                    WeatherStation(
                        name="london",
                        latitude=51.5074,
                        longitude=-0.1278,
                        weight=1.0,
                    ),
                    WeatherStation(
                        name="bristol",
                        latitude=51.4545,
                        longitude=-2.5879,
                        weight=1.0,
                    ),
                ],
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 2),
                cache_dir=paths["weather_dir"],
            ),
            holidays=HolidaysIngestionConfig(
                cache_dir=paths["holidays_dir"],
            ),
        ),
        # weather_only is intentionally None — mirrors Hydra group-swap state.
        features=FeaturesGroup(
            weather_calendar=FeatureSetConfig(
                name="weather_calendar",
                demand_aggregation="mean",
                cache_dir=paths["features_dir"],
                cache_filename="weather_calendar.parquet",
                forward_fill_hours=3,
            ),
        ),
        evaluation=EvaluationGroup(),
    )


# ---------------------------------------------------------------------------
# Test 1 — structural schema invariant (no filesystem)
# Plan T4 ref: lines 305-306
# ---------------------------------------------------------------------------


class TestCalendarOutputSchemaStructure:
    """Plan T4 line 306: structural invariant on CALENDAR_OUTPUT_SCHEMA."""

    def test_calendar_output_schema_is_weather_schema_plus_calendar_plus_provenance(
        self,
    ) -> None:
        """CALENDAR_OUTPUT_SCHEMA = OUTPUT_SCHEMA prefix + CALENDAR_VARIABLE_COLUMNS + provenance.

        Plan T4 test list (line 306):
        - CALENDAR_OUTPUT_SCHEMA.names[:10] == OUTPUT_SCHEMA.names
        - columns 10..53 match CALENDAR_VARIABLE_COLUMNS name + type pairwise
        - column 54 is holidays_retrieved_at_utc with timestamp[us, tz=UTC]
        - total column count is exactly 55
        """
        cal_schema = assembler_mod.CALENDAR_OUTPUT_SCHEMA
        wo_schema = assembler_mod.OUTPUT_SCHEMA
        cal_var_cols = assembler_mod.CALENDAR_VARIABLE_COLUMNS

        # Total column count
        assert len(cal_schema) == 55, (
            f"CALENDAR_OUTPUT_SCHEMA must have exactly 55 columns; got {len(cal_schema)}."
        )

        # weather-only is an exact prefix (first 10 columns)
        assert cal_schema.names[:10] == wo_schema.names, (
            "CALENDAR_OUTPUT_SCHEMA.names[:10] must exactly equal OUTPUT_SCHEMA.names. "
            f"Got {cal_schema.names[:10]!r}, expected {wo_schema.names!r}."
        )

        # Pairwise: columns 10..53 match CALENDAR_VARIABLE_COLUMNS
        assert len(cal_var_cols) == 44, (
            f"CALENDAR_VARIABLE_COLUMNS must contain 44 entries; got {len(cal_var_cols)}."
        )
        for i, (expected_name, expected_type) in enumerate(cal_var_cols):
            actual_field = cal_schema.field(10 + i)
            assert actual_field.name == expected_name, (
                f"CALENDAR_OUTPUT_SCHEMA column {10 + i} name mismatch: "
                f"got {actual_field.name!r}, expected {expected_name!r} "
                f"(CALENDAR_VARIABLE_COLUMNS index {i})."
            )
            assert actual_field.type == expected_type, (
                f"CALENDAR_OUTPUT_SCHEMA column {10 + i} ({expected_name!r}) type mismatch: "
                f"got {actual_field.type}, expected {expected_type}."
            )

        # Last column: holidays_retrieved_at_utc
        last_field = cal_schema.field(54)
        assert last_field.name == "holidays_retrieved_at_utc", (
            f"Column 54 must be 'holidays_retrieved_at_utc'; got {last_field.name!r}."
        )
        assert last_field.type == pa.timestamp("us", tz="UTC"), (
            f"'holidays_retrieved_at_utc' must be timestamp[us, tz=UTC]; got {last_field.type}."
        )


# ---------------------------------------------------------------------------
# Test 2 — assemble_calendar writes a parquet loadable by load_calendar (AC-7)
# Plan T4 ref: lines 307
# ---------------------------------------------------------------------------


class TestAssembleCalendarWritesParquet:
    """Plan T4 line 307 / AC-7: assemble_calendar writes a schema-correct parquet."""

    def test_assemble_calendar_writes_parquet(
        self, primed_calendar_caches: dict[str, Path]
    ) -> None:
        """assemble_calendar returns a Path that exists and passes load_calendar.

        Asserts:
        - Return value is a Path that exists on disk.
        - load_calendar(path) succeeds (no ValueError).
        - DataFrame has exactly CALENDAR_OUTPUT_SCHEMA.names as its column list.
        - Row count is positive.
        """
        cfg = _make_calendar_app_config(primed_calendar_caches)
        out_path = assembler_mod.assemble_calendar(cfg, cache="offline")

        assert isinstance(out_path, Path), (
            f"assemble_calendar must return a Path; got {type(out_path)!r}."
        )
        assert out_path.exists(), f"Expected calendar feature parquet at {out_path}."

        df = assembler_mod.load_calendar(out_path)
        assert list(df.columns) == assembler_mod.CALENDAR_OUTPUT_SCHEMA.names, (
            "Loaded calendar frame columns must exactly match CALENDAR_OUTPUT_SCHEMA.names."
        )
        assert len(df) > 0, "Assembled calendar feature table must have at least one row."


# ---------------------------------------------------------------------------
# Test 3 — load_calendar rejects a weather-only parquet
# Plan T4 ref: line 308
# ---------------------------------------------------------------------------


class TestLoadCalendarRejectsWeatherOnlySchema:
    """Plan T4 line 308: loading a weather_only parquet via load_calendar raises ValueError."""

    def test_load_calendar_rejects_weather_only_schema(self, tmp_path: Path) -> None:
        """load_calendar must raise ValueError for a parquet with OUTPUT_SCHEMA only.

        The error message must name at least one missing calendar column
        (e.g. 'hour_of_day_01', 'is_bank_holiday_ew', or
        'holidays_retrieved_at_utc') so the caller can diagnose the mismatch.
        """
        # Write a parquet that conforms to the weather-only OUTPUT_SCHEMA
        timestamps = pd.date_range(start=_START_UTC, periods=_HOURS, freq="h", tz="UTC")
        retrieved = pd.Timestamp("2023-02-01 12:00:00", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                "nd_mw": [30_000] * _HOURS,
                "tsd_mw": [31_000] * _HOURS,
                "temperature_2m": [5.0] * _HOURS,
                "dew_point_2m": [3.0] * _HOURS,
                "wind_speed_10m": [10.0] * _HOURS,
                "cloud_cover": [50.0] * _HOURS,
                "shortwave_radiation": [100.0] * _HOURS,
                "neso_retrieved_at_utc": [retrieved] * _HOURS,
                "weather_retrieved_at_utc": [retrieved] * _HOURS,
            }
        )
        df["nd_mw"] = df["nd_mw"].astype("int32")
        df["tsd_mw"] = df["tsd_mw"].astype("int32")
        for col in (
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ):
            df[col] = df[col].astype("float32")

        wo_path = tmp_path / "weather_only.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False).cast(
            assembler_mod.OUTPUT_SCHEMA, safe=True
        )
        pq.write_table(table, wo_path)

        # load_calendar must reject this — the 44 calendar columns are absent
        with pytest.raises(ValueError) as exc_info:
            assembler_mod.load_calendar(wo_path)

        error_msg = str(exc_info.value)
        sentinel_columns = ("hour_of_day_01", "is_bank_holiday_ew", "holidays_retrieved_at_utc")
        assert any(sentinel in error_msg for sentinel in sentinel_columns), (
            f"ValueError message must name at least one missing calendar column "
            f"({sentinel_columns!r}). Got: {error_msg!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — load() (weather-only reader) rejects a calendar-schema parquet
# Plan T4 ref: line 309
# ---------------------------------------------------------------------------


class TestLoadRejectsWeatherCalendarSchema:
    """Plan T4 line 309: the existing load() rejects a weather_calendar parquet."""

    def test_load_rejects_weather_calendar_schema(
        self, primed_calendar_caches: dict[str, Path]
    ) -> None:
        """assembler.load must raise ValueError when called on a 55-column parquet.

        The extra calendar columns violate the exact-schema contract documented
        on load() (Plan AC-2): downstream models may select columns positionally.
        """
        cfg = _make_calendar_app_config(primed_calendar_caches)
        cal_path = assembler_mod.assemble_calendar(cfg, cache="offline")

        with pytest.raises(ValueError):
            assembler_mod.load(cal_path)


# ---------------------------------------------------------------------------
# Test 5 — idempotence (§2.1.5)
# Plan T4 ref: line 310
# ---------------------------------------------------------------------------


class TestAssembleCalendarIsIdempotent:
    """Plan T4 line 310 / §2.1.5: second call with same caches yields equal parquet."""

    def test_assemble_calendar_is_idempotent(self, primed_calendar_caches: dict[str, Path]) -> None:
        """Two consecutive assemble_calendar calls return the same path + equal DataFrames.

        Non-provenance columns must be byte-equal (same demand, weather, calendar
        dummies). Provenance columns are expected equal too (same cached
        retrieved_at_utc scalars), but the comparison is done on the non-provenance
        subset to be conservative, matching the idiom in test_assembler_cli.py.
        """
        cfg = _make_calendar_app_config(primed_calendar_caches)

        path_first = assembler_mod.assemble_calendar(cfg, cache="offline")
        df_first = assembler_mod.load_calendar(path_first)

        path_second = assembler_mod.assemble_calendar(cfg, cache="offline")
        df_second = assembler_mod.load_calendar(path_second)

        assert path_first == path_second, "Both assemble_calendar calls must return the same path."

        provenance_cols = [
            c for c in assembler_mod.CALENDAR_OUTPUT_SCHEMA.names if c.endswith("_retrieved_at_utc")
        ]
        non_provenance = [
            c for c in assembler_mod.CALENDAR_OUTPUT_SCHEMA.names if c not in provenance_cols
        ]

        pd.testing.assert_frame_equal(
            df_first[non_provenance].reset_index(drop=True),
            df_second[non_provenance].reset_index(drop=True),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# Test 6 — provenance scalars preserved (DESIGN §2.1.6)
# Plan T4 ref: line 311
# ---------------------------------------------------------------------------


class TestAssembleCalendarProvenanceScalarsPreserved:
    """Plan T4 line 311 / DESIGN §2.1.6: provenance columns are single scalars."""

    def test_assemble_calendar_provenance_scalars_preserved(
        self, primed_calendar_caches: dict[str, Path]
    ) -> None:
        """After assemble_calendar, all three *_retrieved_at_utc columns have nunique()==1.

        DESIGN §2.1.6 requires provenance stamps to be per-run scalars
        repeated across every row — cheap and greppable. Also asserts that
        all three columns are tz-aware UTC timestamps.
        """
        cfg = _make_calendar_app_config(primed_calendar_caches)
        out_path = assembler_mod.assemble_calendar(cfg, cache="offline")
        df = assembler_mod.load_calendar(out_path)

        provenance_cols = [
            "neso_retrieved_at_utc",
            "weather_retrieved_at_utc",
            "holidays_retrieved_at_utc",
        ]

        for col in provenance_cols:
            assert col in df.columns, f"Provenance column {col!r} missing from calendar DataFrame."
            unique_count = df[col].nunique()
            assert unique_count == 1, (
                f"DESIGN §2.1.6: provenance column {col!r} must be a single "
                f"scalar repeated across all rows; found {unique_count} distinct values."
            )
            # tz-aware UTC check
            dtype = df[col].dtype
            assert hasattr(dtype, "tz"), (
                f"Provenance column {col!r} must be a tz-aware dtype; got {dtype!r}."
            )
            assert str(dtype.tz) == "UTC", (
                f"Provenance column {col!r} must be UTC; got tz={dtype.tz!r}."
            )
