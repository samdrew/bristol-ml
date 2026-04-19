"""CLI + orchestrator integration tests for ``bristol_ml.features.assembler``.

These are **implementation-derived** — they exercise the
``assemble`` orchestrator and ``_cli_main`` wiring rather than the
pure ``build`` function. The spec-derived behavioural tests for
``build`` / ``load`` / ``_resample_demand_hourly`` live in
``test_assembler.py``.

Plan cross-reference: ``docs/plans/active/03-feature-assembler.md``
§6 Task T4.

Approach
--------
The orchestrator fans out to ``ingestion.neso.fetch``,
``ingestion.weather.fetch`` and the atomic-write helper. Rather than
stand up full CKAN / Open-Meteo cassettes just to test the CLI shape,
these tests pre-populate both ingestion caches on disk (built
programmatically from the declared ``OUTPUT_SCHEMA`` of each ingester)
and construct an :class:`AppConfig` in Python pointing at the tmp
paths. ``assemble()`` is then called with ``cache="offline"``. This
exercises:

- The ``assemble()`` orchestrator's fetch-load-resample-aggregate-build
  chain.
- The atomic write (``_atomic_write``) at the feature-set cache path:
  parquet exists, no lingering ``.tmp`` sibling.
- ``load()`` round-trip against the persisted parquet.

The ``--help`` smoke is a subprocess test for principle §2.1.1 parity
with the ingestion CLIs.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime
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


# ---------------------------------------------------------------------------
# Fixtures: build NESO and weather caches programmatically from their
# published OUTPUT_SCHEMAs (plan §10 Risk register mitigation).
# ---------------------------------------------------------------------------


def _make_neso_cache(path: Path) -> None:
    """Build a small NESO-shaped parquet at ``path``.

    Two full UTC days of half-hourly rows. Matches ``neso.OUTPUT_SCHEMA``
    exactly so ``neso.load`` succeeds without schema complaints.
    """
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    periods = 96  # 2 days x 48 half-hours
    timestamps = pd.date_range(start=start, periods=periods, freq="30min", tz="UTC")
    local = timestamps.tz_convert("Europe/London")
    retrieved = pd.Timestamp("2024-02-01 12:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "timestamp_local": local,
            "settlement_date": [ts.date() for ts in local],
            "settlement_period": [((i % 48) + 1) for i in range(periods)],
            "nd_mw": [30_000 + 10 * (i % 48) for i in range(periods)],
            "tsd_mw": [31_000 + 10 * (i % 48) for i in range(periods)],
            "source_year": [2024] * periods,
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
    """Build a small weather-shaped parquet at ``path``.

    Two full UTC days of hourly rows for two stations (london, bristol)
    — matches ``weather.OUTPUT_SCHEMA``.
    """
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    hours = 48
    timestamps = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    stations = ("london", "bristol")
    retrieved = pd.Timestamp("2024-02-01 13:00:00", tz="UTC")
    rows: list[dict[str, object]] = []
    for station_idx, station in enumerate(stations):
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "station": station,
                    "temperature_2m": 5.0 + 0.1 * i + station_idx,
                    "dew_point_2m": 3.0 + 0.05 * i + station_idx,
                    "wind_speed_10m": 10.0 + 0.2 * i,
                    "cloud_cover": (80 + i) % 101,
                    "shortwave_radiation": max(0.0, 100.0 - abs(i - 12) * 8.0),
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


@pytest.fixture()
def primed_caches(tmp_path: Path) -> dict[str, Path]:
    """Return tmp paths for all three caches, with NESO + weather pre-populated."""
    neso_dir = tmp_path / "raw" / "neso"
    weather_dir = tmp_path / "raw" / "weather"
    features_dir = tmp_path / "features"
    neso_path = neso_dir / "neso_demand.parquet"
    weather_path = weather_dir / "weather.parquet"
    _make_neso_cache(neso_path)
    _make_weather_cache(weather_path)
    return {
        "neso_dir": neso_dir,
        "weather_dir": weather_dir,
        "features_dir": features_dir,
        "neso_path": neso_path,
        "weather_path": weather_path,
        "features_path": features_dir / "weather_only.parquet",
    }


def _make_app_config(paths: dict[str, Path]) -> AppConfig:
    """Construct an :class:`AppConfig` pointing at the tmp cache dirs.

    Bypasses Hydra entirely — the ``assemble()`` orchestrator consumes an
    ``AppConfig`` directly; Hydra is only the resolution mechanism at the
    CLI boundary. Building the config in Python lets us assert exactly
    what ``assemble`` does with it, independent of Hydra override syntax.
    """
    return AppConfig(
        project=ProjectConfig(name="bristol_ml_cli_test", seed=0),
        ingestion=IngestionGroup(
            neso=NesoIngestionConfig(
                resources=[NesoYearResource(year=2024, resource_id=uuid4())],
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
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 2),
                cache_dir=paths["weather_dir"],
            ),
        ),
        features=FeaturesGroup(
            weather_only=FeatureSetConfig(
                name="weather_only",
                demand_aggregation="mean",
                cache_dir=paths["features_dir"],
                cache_filename="weather_only.parquet",
                forward_fill_hours=3,
            ),
        ),
        evaluation=EvaluationGroup(),
    )


# ---------------------------------------------------------------------------
# Tests — CLI shape
# ---------------------------------------------------------------------------


class TestAssemblerCliHelp:
    """``python -m bristol_ml.features.assembler --help`` exits 0 (§2.1.1)."""

    def test_cli_help_exits_zero(self) -> None:
        """Acceptance: standalone invocation honours principle §2.1.1."""
        result = subprocess.run(
            [sys.executable, "-m", "bristol_ml.features.assembler", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"CLI --help must exit 0; stdout={result.stdout!r} stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Tests — assemble() orchestrator + atomic persistence (plan T4)
# ---------------------------------------------------------------------------


class TestAssembleOrchestratorAtomicWrite:
    """Plan T4: ``assemble()`` writes the feature-set parquet atomically."""

    def test_assembler_writes_parquet_atomically(self, primed_caches: dict[str, Path]) -> None:
        """Guards Plan T4 acceptance.

        Asserts:
        - ``assemble`` returns the feature-set cache path.
        - The parquet exists at that path.
        - No leftover ``.tmp`` sibling (atomic-write invariant).
        - The written parquet passes ``assembler.load`` (OUTPUT_SCHEMA-correct).
        - Row count is positive and every row is non-null.
        """
        cfg = _make_app_config(primed_caches)
        out_path = assembler_mod.assemble(cfg, cache="offline")

        assert out_path == primed_caches["features_path"], (
            f"assemble must return the resolved feature-set path; "
            f"got {out_path}, expected {primed_caches['features_path']}."
        )
        assert out_path.exists(), f"Expected feature-set parquet at {out_path}."
        tmp_sibling = out_path.with_suffix(out_path.suffix + ".tmp")
        assert not tmp_sibling.exists(), (
            f"Atomic-write invariant: no leftover .tmp sibling at {tmp_sibling}."
        )

        df = assembler_mod.load(out_path)
        assert len(df) > 0, "Assembled feature table must have at least one row."
        assert list(df.columns) == assembler_mod.OUTPUT_SCHEMA.names, (
            "Loaded feature table must match OUTPUT_SCHEMA column order."
        )
        assert df.isna().sum().sum() == 0, (
            "CLAUDE.md invariant: no NaN values in the persisted feature table."
        )

    def test_assembler_is_idempotent(self, primed_caches: dict[str, Path]) -> None:
        """Guards §2.1.5: re-running produces an equivalent parquet with no debris."""
        cfg = _make_app_config(primed_caches)
        out_path = assembler_mod.assemble(cfg, cache="offline")
        first = assembler_mod.load(out_path)

        assembler_mod.assemble(cfg, cache="offline")
        second = assembler_mod.load(out_path)

        tmp_sibling = out_path.with_suffix(out_path.suffix + ".tmp")
        assert not tmp_sibling.exists(), "No .tmp sibling after a second run."

        # Both runs must have equal row counts and equal demand + weather
        # columns. The two provenance columns are allowed to differ (D8 —
        # per-run scalars fall back to `Timestamp.utcnow()` when the source
        # frames' `retrieved_at_utc` scalar was the same, which it is here,
        # so they should actually match — but we compare conservatively).
        non_provenance = [
            c for c in assembler_mod.OUTPUT_SCHEMA.names if not c.endswith("_retrieved_at_utc")
        ]
        pd.testing.assert_frame_equal(
            first[non_provenance].reset_index(drop=True),
            second[non_provenance].reset_index(drop=True),
            check_exact=True,
        )

    def test_assembler_refuses_missing_feature_config(self, primed_caches: dict[str, Path]) -> None:
        """``assemble`` raises on ``features.weather_only is None`` rather than
        producing a silent no-op."""
        cfg = _make_app_config(primed_caches).model_copy(update={"features": FeaturesGroup()})
        with pytest.raises(ValueError, match="feature-set config"):
            assembler_mod.assemble(cfg, cache="offline")

    def test_assembler_refuses_missing_ingestion_config(
        self, primed_caches: dict[str, Path]
    ) -> None:
        """``assemble`` raises if either ingestion config is not resolved."""
        cfg = _make_app_config(primed_caches).model_copy(update={"ingestion": IngestionGroup()})
        with pytest.raises(ValueError, match="ingestion"):
            assembler_mod.assemble(cfg, cache="offline")


# ---------------------------------------------------------------------------
# Regression guards on the primed-cache fixtures themselves
# ---------------------------------------------------------------------------


class TestPrimedCacheShapes:
    """Sanity checks on ``_make_neso_cache`` / ``_make_weather_cache``."""

    def test_neso_cache_round_trips(self, tmp_path: Path) -> None:
        """``_make_neso_cache`` produces a parquet ``neso.load`` accepts."""
        path = tmp_path / "neso_demand.parquet"
        _make_neso_cache(path)
        df = neso_mod.load(path)
        assert len(df) == 96, f"Expected 96 half-hourly rows; got {len(df)}."
        assert df["timestamp_utc"].is_monotonic_increasing, (
            "NESO fixture must be sorted by timestamp_utc."
        )

    def test_weather_cache_round_trips(self, tmp_path: Path) -> None:
        """``_make_weather_cache`` produces a parquet ``weather.load`` accepts."""
        path = tmp_path / "weather.parquet"
        _make_weather_cache(path)
        df = weather_mod.load(path)
        assert len(df) == 96, f"Expected 48 hours x 2 stations = 96 rows; got {len(df)}."
        assert set(df["station"].unique()) == {"london", "bristol"}, (
            "Weather fixture must contain the two stations weighted in _make_app_config."
        )


# ---------------------------------------------------------------------------
# Trivial smoke ensuring test-file imports stay live
# ---------------------------------------------------------------------------


def test_datetime_utc_import_is_available() -> None:
    """Guard the `datetime` import used in the fixtures above."""
    now = datetime.now(UTC)
    assert now.tzinfo is UTC
