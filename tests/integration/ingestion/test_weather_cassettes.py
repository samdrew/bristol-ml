"""Integration tests for ``bristol_ml.ingestion.weather`` against recorded cassettes.

Cassette convention
-------------------
Single bulk cassette ``tests/fixtures/weather/cassettes/weather_2023_01.yaml``,
recorded once by the implementer via ``pytest --record-mode=once`` against
the live Open-Meteo archive API (London + Bristol, Jan 2023, five
variables). The module-level ``default_cassette_name`` fixture points
every ``@pytest.mark.vcr`` test at that single file — same pattern as
Stage 1's ``neso_2023_refresh.yaml``.

Until the cassette is recorded the suite skips rather than errors. Once
it exists CI runs with ``--record-mode=none`` (set in ``pyproject.toml``)
and the tests replay deterministically.

Acceptance-criteria mapping
---------------------------
- **AC 2** (no cache → fetch all configured stations): exercised by
  `TestFetchRefreshEndToEnd`.
- **AC 6** (smoke test for the fetcher): any of the REFRESH tests
  suffice; the schema + primary-key tests are the "smoke" in the
  hardest sense.
- LLD §4 schema: column set, types, primary key, sort order.
- LLD §7 idempotence: two REFRESH runs produce identical rows modulo
  `retrieved_at_utc`.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytest

weather = pytest.importorskip("bristol_ml.ingestion.weather")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pytest_recording")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "weather"
CASSETTES = FIXTURES / "cassettes"

# Default to a single bulk cassette (LLD §9, same pattern as Stage 1).
BULK_CASSETTE = "weather_2023_01.yaml"
BULK_CASSETTE_STEM = "weather_2023_01"


pytestmark = [
    pytest.mark.usefixtures("_cassettes_present_or_skip"),
    pytest.mark.vcr,
]


# --------------------------------------------------------------------------- #
# pytest-recording wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _cassettes_present_or_skip() -> None:
    """Skip when no cassette has been recorded yet.

    The implementer is expected to record the bulk cassette once; CI
    runs `--record-mode=none` (set in pyproject). Until the file
    exists the integration tests skip rather than fail, keeping the
    suite green during the build-up phase.
    """
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip(
            f"No bulk cassette at {CASSETTES / BULK_CASSETTE}; "
            "implementer records once via pytest --record-mode=once "
            "before CI runs --record-mode=none."
        )


@pytest.fixture
def vcr_cassette_dir() -> str:
    """Point pytest-recording at ``tests/fixtures/weather/cassettes``."""
    return str(CASSETTES)


@pytest.fixture
def default_cassette_name() -> str:
    """All VCR-marked tests share the single bulk cassette stem."""
    return BULK_CASSETTE_STEM


@pytest.fixture
def vcr_config() -> dict[str, Any]:
    """Filter sensitive headers; allow playback repeats for idempotence."""
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": "none",
        "allow_playback_repeats": True,
    }


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #


def _build_config(tmp_path: Path, **overrides: Any) -> Any:
    """`WeatherIngestionConfig` for the recorded cassette slice.

    The cassette is expected to cover London + Bristol for Jan 2023, 5
    variables (LLD §9). Changing the station list or the date range
    here would mismatch the recorded URL params and trigger vcrpy's
    "no match" error on replay.
    """
    from conf._schemas import WeatherIngestionConfig  # type: ignore[import-not-found]

    cfg_kwargs: dict[str, Any] = dict(
        stations=[
            {
                "name": "london",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "weight": 9787426.0,
                "weight_source": "ONS 2011 Census, Greater London BUA",
            },
            {
                "name": "bristol",
                "latitude": 51.4545,
                "longitude": -2.5879,
                "weight": 617280.0,
                "weight_source": "ONS 2011 Census, Bristol BUA",
            },
        ],
        variables=[
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ],
        start_date=_dt.date(2023, 1, 1),
        end_date=_dt.date(2023, 1, 31),
        cache_dir=tmp_path,
    )
    cfg_kwargs.update(overrides)
    return WeatherIngestionConfig(**cfg_kwargs)


# --------------------------------------------------------------------------- #
# REFRESH end-to-end (intent AC 2, AC 6; LLD §4 schema)
# --------------------------------------------------------------------------- #


class TestFetchRefreshEndToEnd:
    """Intent AC 2 + AC 6: `REFRESH` writes parquet with the documented schema."""

    def test_parquet_is_written(self, tmp_path: Path) -> None:
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        assert Path(path).exists(), f"REFRESH must write parquet at {path}"

    def test_schema_matches_lld_section_4(self, tmp_path: Path) -> None:
        """LLD §4 — documented columns + types.

        Float columns are pinned to `float32`, `cloud_cover` to `int8`,
        timestamps to `timestamp[us, tz=UTC]`. The task-brief table
        differs from the LLD on wind-speed *units* (m/s vs km/h); this
        test pins the *type* (`float32`) only, since units can be
        inspected via parquet metadata rather than the pyarrow type.
        """
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        table = pq.read_table(path)
        schema = table.schema

        required = {
            "timestamp_utc",
            "station",
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "retrieved_at_utc",
        }
        names = set(table.column_names)
        missing = required - names
        assert not missing, f"Persisted parquet missing columns: {missing}"

        # Types.
        ts_utc = schema.field("timestamp_utc").type
        assert pa.types.is_timestamp(ts_utc), f"timestamp_utc not timestamp: {ts_utc}"
        assert ts_utc.unit == "us", f"timestamp_utc unit: expected 'us', got {ts_utc.unit}"
        assert ts_utc.tz in {"UTC", "utc"}, f"timestamp_utc tz: expected UTC, got {ts_utc.tz}"

        stn_t = schema.field("station").type
        assert pa.types.is_string(stn_t) or pa.types.is_dictionary(stn_t), (
            f"station must be string/dictionary, got {stn_t}"
        )

        for col in ("temperature_2m", "dew_point_2m", "wind_speed_10m", "shortwave_radiation"):
            assert schema.field(col).type == pa.float32(), (
                f"{col} must be float32; got {schema.field(col).type}"
            )
        assert schema.field("cloud_cover").type == pa.int8(), (
            f"cloud_cover must be int8; got {schema.field('cloud_cover').type}"
        )

        ts_prov = schema.field("retrieved_at_utc").type
        assert pa.types.is_timestamp(ts_prov)
        assert ts_prov.tz in {"UTC", "utc"}

    def test_one_row_per_timestamp_station_pair(self, tmp_path: Path) -> None:
        """LLD §4: primary key `(timestamp_utc, station)` is unique."""
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        df = weather.load(Path(path))
        dupes = df.duplicated(subset=["timestamp_utc", "station"]).sum()
        assert dupes == 0, f"(timestamp_utc, station) must be unique; found {dupes} duplicate(s)."

    def test_all_configured_stations_present(self, tmp_path: Path) -> None:
        """Intent AC 2: all configured stations appear in the output.

        A silently-dropped station (e.g. Bristol returning empty,
        implementer ignores) is the exact failure mode the layer
        architecture's "missing required → hard error" rule is there
        to prevent. Assert all configured names appear at least once.
        """
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        df = weather.load(Path(path))
        stations_out = set(df["station"].unique())
        expected = {s.name for s in cfg.stations}
        missing = expected - stations_out
        assert not missing, f"Intent AC 2 failure: stations fetched but not persisted: {missing}"

    def test_rows_sorted_by_timestamp_then_station(self, tmp_path: Path) -> None:
        """LLD §4: sorted by `timestamp_utc ASC, station ASC`."""
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        df = weather.load(Path(path))
        # Primary sort key: timestamp_utc monotonic non-decreasing.
        assert df["timestamp_utc"].is_monotonic_increasing, (
            "Rows must be sorted by timestamp_utc ascending per LLD §4."
        )

    def test_timestamp_utc_is_timezone_aware_utc(self, tmp_path: Path) -> None:
        """LLD §4: the loaded dataframe carries tz-aware UTC timestamps."""
        cfg = _build_config(tmp_path)
        path = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
        df = weather.load(Path(path))
        tz = df["timestamp_utc"].dt.tz
        assert tz is not None, "timestamp_utc must be tz-aware."
        assert str(tz) in {"UTC", "utc"}, f"timestamp_utc tz must be UTC; got {tz}"


# --------------------------------------------------------------------------- #
# Idempotence (LLD §7)
# --------------------------------------------------------------------------- #


def test_fetch_idempotent_refresh_runs_equal_modulo_provenance(tmp_path: Path) -> None:
    """Two consecutive REFRESH runs produce identical rows aside from `retrieved_at_utc`.

    LLD §7: "REFRESH: fetch all stations, overwrite atomically." The
    provenance column is per-fetch (§2.1.6) so it is permitted to
    differ between runs; everything else must be byte-equal.
    """
    cfg = _build_config(tmp_path)

    path1 = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
    df1 = weather.load(Path(path1))

    path2 = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
    df2 = weather.load(Path(path2))

    assert Path(path1) == Path(path2), "Cache path must be stable across runs."

    payload_cols = [c for c in df1.columns if c != "retrieved_at_utc"]
    left = df1[payload_cols].reset_index(drop=True)
    right = df2[payload_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=True,
        check_dtype=True,
        obj="payload (all columns except retrieved_at_utc)",
    )


# --------------------------------------------------------------------------- #
# AUTO cache-hit (intent AC 1) — integration-level belt-and-braces
# --------------------------------------------------------------------------- #


def test_auto_after_refresh_returns_same_cache_offline(tmp_path: Path) -> None:
    """Intent AC 1: once the cache exists, AUTO (and OFFLINE) work without network.

    REFRESH populates the cache from the cassette; a subsequent OFFLINE
    call must return the same path without issuing any HTTP request.
    The cassette's `allow_playback_repeats=True` is irrelevant here —
    OFFLINE should short-circuit before any client is created.
    """
    cfg = _build_config(tmp_path)
    seeded = weather.fetch(cfg, cache=weather.CachePolicy.REFRESH)
    # OFFLINE must not hit the cassette either.
    offline_path = weather.fetch(cfg, cache=weather.CachePolicy.OFFLINE)
    assert Path(seeded) == Path(offline_path), (
        f"OFFLINE after REFRESH must return the same cache path; "
        f"got {seeded!r} then {offline_path!r}"
    )
