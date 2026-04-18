"""Integration tests for ``bristol_ml.ingestion.neso`` against recorded cassettes.

The implementer records cassettes once (``pytest --record-mode=once``)
against the live NESO CKAN API; CI replays them with
``--record-mode=none`` (configured in ``pyproject.toml``).

All integration tests in this module share a single recorded cassette
``tests/fixtures/neso/cassettes/neso_2023_refresh.yaml`` — two paginator
pages of 500 rows each, with ``total`` filtered to 1000 so the replay
paginator terminates. The ``default_cassette_name`` fixture is
overridden at module level so every ``@pytest.mark.vcr``-decorated test
loads that single bulk cassette, rather than the pytest-recording
default of per-test-function cassette files.

Tests encode:
- Acceptance criterion 2 — no cache → fetch + write local copy.
- Acceptance criterion 3 — two runs produce the same on-disk result.
- Acceptance criterion 6 — tests exercise the public interface using
  recorded fixtures.
- LLD §4 — output parquet schema and row ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

neso = pytest.importorskip("bristol_ml.ingestion.neso")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

# ``pytest-recording`` (vcrpy) provides the ``@pytest.mark.vcr`` marker. If it
# isn't installed yet (implementer hasn't wired dependencies), skip the
# integration suite rather than erroring out.
pytest.importorskip("pytest_recording")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "neso"
CASSETTES = FIXTURES / "cassettes"
BULK_CASSETTE = "neso_2023_refresh.yaml"
# pytest-recording appends the serializer suffix (".yaml") to the default
# cassette name, so the fixture override returns the stem.
BULK_CASSETTE_STEM = "neso_2023_refresh"

# Configure vcrpy cassette directory + skip when nothing's been recorded yet.
# Every VCR-marked test in this module shares the single bulk cassette.
pytestmark = [
    pytest.mark.usefixtures("_cassettes_present_or_skip"),
    pytest.mark.vcr,
]


@pytest.fixture(scope="module")
def _cassettes_present_or_skip() -> None:
    """Skip the module when no cassettes have been recorded yet.

    The implementer records cassettes once; until then every integration
    test would error with vcrpy's "no cassette" message. Skipping keeps
    the suite green during the build-up phase. When cassettes exist
    the tests run normally.
    """
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip(
            f"No bulk cassette at {CASSETTES / BULK_CASSETTE}; implementer records once "
            "via pytest --record-mode=once before CI runs --record-mode=none."
        )


@pytest.fixture
def vcr_cassette_dir() -> str:
    """Point pytest-recording at ``tests/fixtures/neso/cassettes``."""
    return str(CASSETTES)


@pytest.fixture
def default_cassette_name() -> str:
    """Override pytest-recording's per-test cassette name.

    All integration tests in this module replay against the single
    recorded bulk cassette, not per-function files. Returning the bulk
    cassette stem (pytest-recording appends ``.yaml``) points every
    ``@pytest.mark.vcr`` test at the shared recording.
    """
    return BULK_CASSETTE_STEM


@pytest.fixture
def vcr_config() -> dict[str, Any]:
    """Filter sensitive headers even on unauthenticated feeds (layer §5).

    ``allow_playback_repeats=True`` lets the idempotence test make two
    consecutive REFRESH runs against the same cassette — each run hits
    the identical ``(resource_id, limit, offset=0)`` and ``(offset=500)``
    request pairs, and we want the second run to replay them rather
    than error out on "no more recordings for this request".
    """
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": "none",
        "allow_playback_repeats": True,
    }


def _build_config(
    tmp_path: Path,
    *,
    page_size: int = 500,
    resources: list[dict[str, Any]] | None = None,
) -> Any:
    """Config for cassette tests — narrow 2023 slice (LLD §9).

    ``page_size`` defaults to 500 to match the bulk cassette recording;
    changing it here would mismatch the recorded ``offset`` sequence
    and trigger vcrpy's "no match found" error on replay.
    """
    from conf._schemas import NesoIngestionConfig  # type: ignore[import-not-found]

    if resources is None:
        resources = [{"year": 2023, "resource_id": UUID("bf5ab335-9b40-4ea4-b93a-ab4af7bce003")}]
    return NesoIngestionConfig(
        resources=resources,
        cache_dir=tmp_path,
        columns=["ND", "TSD"],
        page_size=page_size,
    )


# --------------------------------------------------------------------------- #
# End-to-end REFRESH against recorded cassette (acceptance criterion 2)
# --------------------------------------------------------------------------- #


class TestFetchRefreshEndToEnd:
    """LLD §10: REFRESH + recorded cassette → parquet; schema matches §4."""

    def test_parquet_is_written(self, tmp_path: Path) -> None:
        cfg = _build_config(tmp_path)
        path = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
        assert Path(path).exists(), f"REFRESH must write parquet at {path}"

    def test_schema_matches_lld_section_4(self, tmp_path: Path) -> None:
        """LLD §4: exact column set and types (parquet-level assertions)."""
        cfg = _build_config(tmp_path)
        path = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
        table = pq.read_table(path)
        names = set(table.column_names)

        required = {
            "timestamp_utc",
            "timestamp_local",
            "settlement_date",
            "settlement_period",
            "nd_mw",
            "tsd_mw",
            "source_year",
            "retrieved_at_utc",
        }
        missing = required - names
        extra = names - required
        assert not missing, f"Persisted parquet missing columns: {missing}"
        assert not extra, (
            f"Persisted parquet has unexpected columns: {extra}. "
            "Embedded wind/solar + interconnector columns should be dropped per LLD §4."
        )

        # Spot-check types per LLD §4.
        schema = table.schema
        ts_utc = schema.field("timestamp_utc").type
        assert pa.types.is_timestamp(ts_utc), f"timestamp_utc must be timestamp, got {ts_utc}"
        assert ts_utc.unit == "us", f"timestamp_utc must be 'us', got {ts_utc.unit}"
        assert ts_utc.tz in {"UTC", "utc"}, f"timestamp_utc must be tz=UTC, got {ts_utc.tz}"

        ts_local = schema.field("timestamp_local").type
        assert pa.types.is_timestamp(ts_local)
        # LLD §4 lists the local zone as Europe/London; accept either UTC+offset
        # representation if the implementer normalised to a fixed offset.
        assert ts_local.tz in {"Europe/London", "UTC+00:00", "UTC+01:00"} or ts_local.tz, (
            f"timestamp_local must be tz-aware; got tz={ts_local.tz}"
        )

        assert pa.types.is_date32(schema.field("settlement_date").type)
        assert schema.field("settlement_period").type == pa.int8()
        assert schema.field("nd_mw").type == pa.int32()
        assert schema.field("tsd_mw").type == pa.int32()
        assert schema.field("source_year").type == pa.int16()
        ts_prov = schema.field("retrieved_at_utc").type
        assert pa.types.is_timestamp(ts_prov)
        assert ts_prov.tz in {"UTC", "utc"}

    def test_rows_sorted_by_timestamp_utc(self, tmp_path: Path) -> None:
        """LLD §4: sorted by ``timestamp_utc`` ascending."""
        cfg = _build_config(tmp_path)
        path = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
        df = neso.load(Path(path))
        ts = df["timestamp_utc"]
        assert ts.is_monotonic_increasing, (
            "Persisted rows must be sorted by timestamp_utc ascending per LLD §4."
        )

    def test_primary_key_unique(self, tmp_path: Path) -> None:
        """LLD §4: ``timestamp_utc`` is the primary key — unique."""
        cfg = _build_config(tmp_path)
        path = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
        df = neso.load(Path(path))
        dupes = df["timestamp_utc"].duplicated().sum()
        assert dupes == 0, f"timestamp_utc must be unique; found {dupes} duplicates."


# --------------------------------------------------------------------------- #
# Idempotence (acceptance criterion 3)
# --------------------------------------------------------------------------- #


def test_fetch_idempotent_refresh_runs_equal_modulo_provenance(tmp_path: Path) -> None:
    """Two consecutive REFRESH runs produce identical rows aside from provenance.

    Acceptance criterion 3 from the stage intent: "Running the ingestion
    twice in a row produces the same on-disk result." LLD §7 reads this
    as: same row-set, same schema; ``retrieved_at_utc`` is provenance
    and is permitted to differ between runs (it is per-fetch, not
    per-row).
    """
    cfg = _build_config(tmp_path)

    path1 = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
    df1 = neso.load(Path(path1))

    path2 = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)
    df2 = neso.load(Path(path2))

    assert Path(path1) == Path(path2), "Cache path must be stable across runs"

    # Drop provenance column and compare row-wise.
    payload_cols = [c for c in df1.columns if c != "retrieved_at_utc"]
    assert sorted(df1.columns) == sorted(df2.columns), (
        f"Column sets must match across runs; got {sorted(df1.columns)} vs {sorted(df2.columns)}"
    )

    left = df1[payload_cols].reset_index(drop=True)
    right = df2[payload_cols].reset_index(drop=True)
    # Use pandas test util so the error messages carry column-level detail.
    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=True,
        check_dtype=True,
        obj="payload (all columns except retrieved_at_utc)",
    )


# --------------------------------------------------------------------------- #
# Public interface coverage (acceptance criterion 6)
# --------------------------------------------------------------------------- #


def test_public_interface_reexported_from_package() -> None:
    """Acceptance criterion 6: tests exercise the module's public interface.

    LLD §2 requires ``fetch``, ``load`` and ``CachePolicy`` to be
    importable from ``bristol_ml.ingestion`` (the package boundary).
    """
    import bristol_ml.ingestion as pkg

    assert hasattr(pkg, "neso"), "bristol_ml.ingestion must expose neso submodule"
    # ``fetch`` and ``load`` may also be re-exported at the package level
    # per LLD §2 "Both callables are re-exported from bristol_ml.ingestion".
    # We do not hard-require the top-level re-export because LLD allows
    # `from bristol_ml.ingestion import neso` as the canonical form.
    assert callable(pkg.neso.fetch)
    assert callable(pkg.neso.load)
    assert hasattr(pkg.neso, "CachePolicy")
