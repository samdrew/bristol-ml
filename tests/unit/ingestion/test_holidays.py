"""Unit tests for ``bristol_ml.ingestion.holidays``.

Covers Stage 5 Task T2 — GB bank-holidays ingester.  The test list is
drawn from:

- ``docs/plans/active/05-calendar-features.md`` §6 T2 (the six plan-named
  tests: parquet write, schema enforcement, provenance, idempotent offline,
  offline-raises-on-missing, all-three-divisions).
- AC-6: bank-holiday ingestion is idempotent and offline-first.
- The layer contract in ``docs/architecture/layers/ingestion.md``
  (schema assertion, atomic writes, cache policy semantics).

Cassette: ``tests/fixtures/holidays/cassettes/holidays_refresh.yaml``
— one GET of ``https://www.gov.uk/bank-holidays.json``; events from
2019-01-01 → 2028-12-26; 280 total events across three divisions.
Christmas Day 2019-12-25 appears under all three divisions.

Cassette playback runs the ingester against the recorded response;
CI executes with ``--record-mode=none`` (set in ``pyproject.toml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

holidays = pytest.importorskip("bristol_ml.ingestion.holidays")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pytest_recording")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "holidays"
CASSETTES = FIXTURES / "cassettes"
BULK_CASSETTE = "holidays_refresh.yaml"
BULK_CASSETTE_STEM = "holidays_refresh"


# --------------------------------------------------------------------------- #
# pytest-recording wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _cassette_present_or_skip() -> None:
    """Skip the cassette-backed tests when no recording exists.

    Mirrors the pattern used by the NESO forecast cassette suite: during
    the build-up phase the cassette may be absent; once recorded, every
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
    """Build a ``HolidaysIngestionConfig`` pointing at ``tmp_path``."""
    from conf._schemas import HolidaysIngestionConfig

    kwargs: dict[str, Any] = dict(cache_dir=tmp_path)
    kwargs.update(overrides)
    return HolidaysIngestionConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Plan-named tests (T2 "Tests" block)
# --------------------------------------------------------------------------- #


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_holidays_fetch_writes_parquet(tmp_path: Path) -> None:
    """Plan T2: ``fetch`` writes a parquet whose schema matches ``OUTPUT_SCHEMA``.

    Playback is against the 2019-2028 cassette (all three UK divisions).
    The test asserts both the file's existence and that the persisted arrow
    schema equals the one declared by the module, column-by-column.
    """
    cfg = _build_config(tmp_path)
    path = holidays.fetch(cfg, cache=holidays.CachePolicy.REFRESH)

    assert Path(path).exists(), f"REFRESH must write parquet at {path}"

    table = pq.read_table(path)
    schema = table.schema
    expected = holidays.OUTPUT_SCHEMA
    for field in expected:
        assert field.name in schema.names, f"Missing column {field.name!r} in persisted schema."
        assert schema.field(field.name).type == field.type, (
            f"Column {field.name!r} type={schema.field(field.name).type}; expected {field.type}."
        )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_holidays_load_schema_enforced(tmp_path: Path) -> None:
    """Plan T2: reading a parquet with a missing required column raises.

    Write a valid parquet via ``fetch``, then rewrite it without the
    ``title`` column and confirm ``load`` raises a ``ValueError`` naming
    the offender.
    """
    cfg = _build_config(tmp_path)
    path = holidays.fetch(cfg, cache=holidays.CachePolicy.REFRESH)

    table = pq.read_table(path)
    stripped = table.drop_columns(["title"])
    broken_path = Path(path).with_name("broken.parquet")
    pq.write_table(stripped, broken_path)

    with pytest.raises(ValueError) as exc:
        holidays.load(broken_path)
    assert "title" in str(exc.value), "load() must name the missing column; got: " + str(exc.value)


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_holidays_provenance_column_populated(tmp_path: Path) -> None:
    """Plan T2: ``retrieved_at_utc`` is a single scalar across all rows.

    Per the layer contract the provenance column is per-fetch, not
    per-row; two runs will write different values but within one run
    every row shares the same timestamp, and that timestamp must be
    tz=UTC.
    """
    cfg = _build_config(tmp_path)
    path = holidays.fetch(cfg, cache=holidays.CachePolicy.REFRESH)
    df = holidays.load(Path(path))

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


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_holidays_fetch_idempotent_offline(tmp_path: Path) -> None:
    """Plan T2 / AC-6: a second ``OFFLINE`` call returns the same path unchanged.

    Steps:
    1. ``REFRESH`` under the cassette to populate the cache.
    2. Record the file's mtime (nanosecond precision).
    3. Call ``OFFLINE`` — no cassette wrap; an HTTP call would blow up VCR.
    4. Assert path identity and that mtime_ns is unchanged (no re-write).
    """
    cfg = _build_config(tmp_path)
    first_path = holidays.fetch(cfg, cache=holidays.CachePolicy.REFRESH)
    mtime_before = Path(first_path).stat().st_mtime_ns

    second_path = holidays.fetch(cfg, cache=holidays.CachePolicy.OFFLINE)

    assert Path(second_path) == Path(first_path), (
        "OFFLINE call must return the same path as the initial REFRESH."
    )
    mtime_after = Path(second_path).stat().st_mtime_ns
    assert mtime_after == mtime_before, (
        f"OFFLINE call must not re-write the cache file; "
        f"mtime changed from {mtime_before} to {mtime_after}."
    )


def test_holidays_fetch_offline_raises_on_missing_cache(tmp_path: Path) -> None:
    """Plan T2: ``CacheMissingError`` is raised when the cache is absent.

    No cassette needed — the call must raise before any network contact.
    The error message must mention the missing cache path.
    """
    cfg = _build_config(tmp_path)

    with pytest.raises(holidays.CacheMissingError) as exc:
        holidays.fetch(cfg, cache=holidays.CachePolicy.OFFLINE)

    error_text = str(exc.value)
    # The error must mention the path so the user knows where to look.
    assert str(tmp_path) in error_text or "holidays.parquet" in error_text, (
        f"CacheMissingError message does not mention the missing path; got: {error_text!r}"
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_holidays_all_three_divisions_present(tmp_path: Path) -> None:
    """Plan T2 / D-1: the parquet contains events for all three UK divisions.

    The cache is policy-agnostic — even though the feature derivation only
    encodes E&W and Scotland (plan D-2), the ingester persists Northern
    Ireland too so a future regional stage does not need to re-ingest.

    Additionally asserts that 2019-12-25 (Christmas Day, UK-wide) has
    exactly three rows — one per division — each with title "Christmas Day".
    """
    cfg = _build_config(tmp_path)
    path = holidays.fetch(cfg, cache=holidays.CachePolicy.REFRESH)
    df = holidays.load(Path(path))

    assert set(df["division"].unique()) == {
        "england-and-wales",
        "scotland",
        "northern-ireland",
    }, f"Expected all three UK divisions; got: {sorted(df['division'].unique())}"

    christmas_2019 = df[df["date"] == pd.Timestamp("2019-12-25").date()]
    assert len(christmas_2019) == 3, (
        f"Christmas Day 2019-12-25 must appear once per division (3 rows); "
        f"got {len(christmas_2019)} rows."
    )
    assert set(christmas_2019["title"].unique()) == {"Christmas Day"}, (
        f"All three rows for 2019-12-25 must have title 'Christmas Day'; "
        f"got: {sorted(christmas_2019['title'].unique())}"
    )
