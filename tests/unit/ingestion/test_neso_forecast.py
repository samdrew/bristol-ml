"""Unit tests for ``bristol_ml.ingestion.neso_forecast``.

Covers Stage 4 Task T7 — the NESO Day-Ahead Half-Hourly Demand Forecast
Performance archive ingester.  The test list is drawn from:

- ``docs/plans/active/04-linear-baseline.md`` §6 T7 (the three
  plan-named tests: parquet write, schema enforcement, provenance).
- ``docs/intent/04-linear-baseline.md`` AC-5 (benchmark data source).
- The layer contract in ``docs/architecture/layers/ingestion.md``
  (schema assertion, atomic writes, cache policy semantics).

Cassette: ``tests/fixtures/neso_forecast/cassettes/neso_forecast_refresh.yaml``
— one paginator page covering 2023-10-28 (normal) and 2023-10-29
(autumn-fallback Sunday, 50 settlement periods).  The cassette was
hand-crafted from live NESO CKAN responses with ``filters={"Date":
...}`` then rewritten to expose ``total=98`` so the pagination loop
terminates after a single request.

Cassette playback runs the ingester against the recorded response;
CI executes with ``--record-mode=none`` (set in ``pyproject.toml``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

neso_forecast = pytest.importorskip("bristol_ml.ingestion.neso_forecast")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pytest_recording")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "neso_forecast"
CASSETTES = FIXTURES / "cassettes"
BULK_CASSETTE = "neso_forecast_refresh.yaml"
BULK_CASSETTE_STEM = "neso_forecast_refresh"


# --------------------------------------------------------------------------- #
# pytest-recording wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _cassette_present_or_skip() -> None:
    """Skip the cassette-backed tests when no recording exists.

    Mirrors the pattern used by the Stage 1 cassette suite: during the
    build-up phase the cassette may be absent; once recorded, every
    ``@pytest.mark.vcr`` test replays it.
    """
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip(
            f"No cassette at {CASSETTES / BULK_CASSETTE}; record once via "
            "pytest --record-mode=once before CI."
        )


@pytest.fixture
def vcr_cassette_dir() -> str:
    """Point pytest-recording at this ingester's cassette dir."""
    return str(CASSETTES)


@pytest.fixture
def default_cassette_name() -> str:
    """Share the single bulk cassette across every VCR-marked test."""
    return BULK_CASSETTE_STEM


@pytest.fixture
def vcr_config() -> dict[str, Any]:
    """Filter sensitive headers; allow replay repeats for idempotence checks."""
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": "none",
        "allow_playback_repeats": True,
    }


def _build_config(tmp_path: Path, **overrides: Any) -> Any:
    """Build a ``NesoForecastIngestionConfig`` pointing at ``tmp_path``.

    ``page_size`` defaults to 500 to match the cassette recording.
    """
    from conf._schemas import NesoForecastIngestionConfig

    kwargs: dict[str, Any] = dict(
        resource_id=UUID("08e41551-80f8-4e28-a416-ea473a695db9"),
        cache_dir=tmp_path,
        page_size=500,
    )
    kwargs.update(overrides)
    return NesoForecastIngestionConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Plan-named tests (T7 "Tests" block)
# --------------------------------------------------------------------------- #


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_neso_forecast_fetch_writes_parquet(tmp_path: Path) -> None:
    """Plan T7: ``fetch`` writes a parquet whose schema matches ``OUTPUT_SCHEMA``.

    Playback is against the 2023-Q4 cassette (2 days incl. autumn
    fallback).  The test asserts both the file's existence and that the
    persisted arrow schema equals the one declared by the module.
    """
    cfg = _build_config(tmp_path)
    path = neso_forecast.fetch(cfg, cache=neso_forecast.CachePolicy.REFRESH)

    assert Path(path).exists(), f"REFRESH must write parquet at {path}"

    table = pq.read_table(path)
    schema = table.schema
    expected = neso_forecast.OUTPUT_SCHEMA
    for field in expected:
        assert field.name in schema.names, f"Missing column {field.name!r} in persisted schema."
        assert schema.field(field.name).type == field.type, (
            f"Column {field.name!r} type={schema.field(field.name).type}; expected {field.type}."
        )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_neso_forecast_load_schema_enforced(tmp_path: Path) -> None:
    """Plan T7: reading a parquet with a missing required column raises.

    Write a valid parquet via ``fetch``, then rewrite it without the
    ``demand_forecast_mw`` column and confirm ``load`` raises a
    ``ValueError`` naming the offender.
    """
    cfg = _build_config(tmp_path)
    path = neso_forecast.fetch(cfg, cache=neso_forecast.CachePolicy.REFRESH)

    table = pq.read_table(path)
    stripped = table.drop_columns(["demand_forecast_mw"])
    broken_path = Path(path).with_name("broken.parquet")
    pq.write_table(stripped, broken_path)

    with pytest.raises(ValueError) as exc:
        neso_forecast.load(broken_path)
    assert "demand_forecast_mw" in str(exc.value), (
        "load() must name the missing column; got: " + str(exc.value)
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_neso_forecast_provenance_column_populated(tmp_path: Path) -> None:
    """Plan T7: ``retrieved_at_utc`` is a single scalar across all rows.

    Per the layer contract the provenance column is per-fetch, not
    per-row; two runs will write different values but within one run
    every row shares the same timestamp.
    """
    cfg = _build_config(tmp_path)
    path = neso_forecast.fetch(cfg, cache=neso_forecast.CachePolicy.REFRESH)
    df = neso_forecast.load(Path(path))

    assert "retrieved_at_utc" in df.columns, "Provenance column must be present on load."
    assert len(df) > 0, "Cassette must produce at least one row."
    unique_stamps = df["retrieved_at_utc"].unique()
    assert len(unique_stamps) == 1, (
        f"retrieved_at_utc must be constant per fetch; got {len(unique_stamps)} distinct values."
    )
    stamp = pd.Timestamp(unique_stamps[0])
    assert stamp.tz is not None and str(stamp.tz) in {"UTC", "utc"}, (
        f"retrieved_at_utc must be tz=UTC; got tz={stamp.tz!r}."
    )


# --------------------------------------------------------------------------- #
# Contract corners — schema assertion, DST arithmetic, load() validation
# --------------------------------------------------------------------------- #


def test_assert_schema_missing_identity_raises() -> None:
    """``_assert_schema`` raises ``KeyError`` naming a missing identity column."""
    df = pd.DataFrame(
        {
            # ``Date`` missing.
            "Settlement_Period": [1],
            "Demand_Forecast": [21000],
            "Demand_Outturn": [20500],
            "APE": [2.4],
        }
    )
    with pytest.raises(KeyError) as exc:
        neso_forecast._assert_schema(df, ("Demand_Forecast", "Demand_Outturn", "APE"))
    assert "Date" in str(exc.value), "Missing-required error must name the offending column."


def test_assert_schema_missing_measurement_raises() -> None:
    """A caller-requested measurement missing upstream raises ``KeyError``."""
    df = pd.DataFrame(
        {
            "Date": ["2023-10-28"],
            "Settlement_Period": [1],
            "Demand_Forecast": [21000],
            # ``Demand_Outturn`` missing.
            "APE": [2.4],
        }
    )
    with pytest.raises(KeyError) as exc:
        neso_forecast._assert_schema(df, ("Demand_Forecast", "Demand_Outturn", "APE"))
    assert "Demand_Outturn" in str(exc.value)


def test_assert_schema_unknown_columns_warn_and_drop() -> None:
    """Layer contract: unknown columns produce ``UserWarning`` and are dropped."""
    df = pd.DataFrame(
        {
            "Date": ["2023-10-28"],
            "Settlement_Period": [1],
            "Demand_Forecast": [21000],
            "Demand_Outturn": [20500],
            "APE": [2.4],
            "_id": [42],  # suppressed silently.
            "TRIAD_Avoidance_Estimate": [0],
            "Publish_Datetime": ["2023-10-27T08:45:00"],
        }
    )
    with pytest.warns(UserWarning) as record:
        cleaned = neso_forecast._assert_schema(df, ("Demand_Forecast", "Demand_Outturn", "APE"))
    messages = " ".join(str(r.message) for r in record)
    assert "TRIAD_Avoidance_Estimate" in messages
    assert "Publish_Datetime" in messages
    assert "TRIAD_Avoidance_Estimate" not in cleaned.columns, "Unknown col must be dropped."
    assert "_id" not in cleaned.columns, "Sentinel _id must be suppressed."


def test_to_utc_autumn_fallback_periods_3_and_5_one_hour_apart() -> None:
    """DST algebra: on an autumn-fallback day, periods 3 (BST) and 5 (GMT) are 1h apart."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-10-29"] * 6).date,
            "Settlement_Period": [1, 2, 3, 4, 5, 6],
        }
    )
    out = neso_forecast._to_utc(df)
    period3 = out[out["Settlement_Period"] == 3].iloc[0]
    period5 = out[out["Settlement_Period"] == 5].iloc[0]
    ts3 = pd.Timestamp(period3["timestamp_utc"])
    ts5 = pd.Timestamp(period5["timestamp_utc"])
    delta = (ts5 - ts3).total_seconds()
    assert delta == 3600, (
        f"Autumn-fallback period 5 must sit 1h after period 3 in UTC; got {delta}s."
    )


def test_to_utc_spring_forward_period_out_of_range_raises() -> None:
    """Period > 46 on a spring-forward Sunday is corrupt data and must raise."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-03-26"] * 2).date,
            "Settlement_Period": [46, 47],  # 47 is out of range on a 46-period day.
        }
    )
    with pytest.raises(ValueError) as exc:
        neso_forecast._to_utc(df)
    msg = str(exc.value)
    assert "47" in msg or "2023-03-26" in msg or "settlement period" in msg.lower()


def test_load_rejects_wrong_type(tmp_path: Path) -> None:
    """``load`` raises when a column's arrow type disagrees with ``OUTPUT_SCHEMA``.

    Build a frame conformant to ``OUTPUT_SCHEMA`` except that
    ``settlement_period`` is widened to ``int16``; the schema check
    must fire on that column with both the expected and actual types
    quoted in the error.
    """
    # Start from the declared OUTPUT_SCHEMA and swap one field so the
    # mismatch is localised to a single column.  Using the declared
    # schema as the source of truth avoids stray drift (e.g. ns vs us
    # on timestamp columns) tripping the check before we reach the
    # column under test.
    good_schema = neso_forecast.OUTPUT_SCHEMA
    broken_schema = pa.schema(
        [(f.name, pa.int16() if f.name == "settlement_period" else f.type) for f in good_schema]
    )
    ts = pd.Timestamp("2023-10-28T00:00:00Z").to_pydatetime()
    rows = {
        "timestamp_utc": [ts],
        "timestamp_local": [ts],
        "settlement_date": [ts.date()],
        "settlement_period": [1],
        "demand_forecast_mw": [21000],
        "demand_outturn_mw": [20500],
        "ape_percent": [2.4],
        "retrieved_at_utc": [ts],
    }
    broken = tmp_path / "bad_types.parquet"
    pq.write_table(pa.Table.from_pydict(rows, schema=broken_schema), broken)
    with pytest.raises(ValueError) as exc:
        neso_forecast.load(broken)
    assert "settlement_period" in str(exc.value), (
        f"load() must name the mismatched column; got: {exc.value!r}"
    )


def test_offline_cache_missing_raises(tmp_path: Path) -> None:
    """``CachePolicy.OFFLINE`` against an absent cache raises ``CacheMissingError``."""
    cfg = _build_config(tmp_path)
    with pytest.raises(neso_forecast.CacheMissingError):
        neso_forecast.fetch(cfg, cache=neso_forecast.CachePolicy.OFFLINE)


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #


def test_cli_help_exits_zero() -> None:
    """``python -m bristol_ml.ingestion.neso_forecast --help`` exits 0."""
    completed = subprocess.run(
        [sys.executable, "-m", "bristol_ml.ingestion.neso_forecast", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"--help must exit 0; stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert "NESO Day-Ahead" in completed.stdout or "neso_forecast" in completed.stdout


# --------------------------------------------------------------------------- #
# Cassette sanity — the hand-crafted cassette must parse as JSON in its body.
# --------------------------------------------------------------------------- #


def test_cassette_body_is_valid_json() -> None:
    """Guard rail: the cassette's response body must be valid JSON.

    A malformed cassette silently mis-replays and produces hard-to-trace
    test failures downstream.  Parsing the body eagerly catches
    escape-character slip-ups at the fixture-author level.
    """
    cassette_path = CASSETTES / BULK_CASSETTE
    if not cassette_path.exists():
        pytest.skip("Cassette not recorded yet.")
    text = cassette_path.read_text()
    # Locate the 'string:' line and parse from there.
    import re

    match = re.search(r"body:\s*\n\s*string:\s*'(.*?)'\n", text, re.DOTALL)
    assert match is not None, "Cassette layout unexpectedly changed; body string not found."
    # YAML doubles single quotes inside single-quoted strings.
    body = match.group(1).replace("''", "'")
    payload = json.loads(body)
    assert payload["success"] is True
    assert payload["result"]["total"] == len(payload["result"]["records"])
