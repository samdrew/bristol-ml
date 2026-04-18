"""Unit tests for ``bristol_ml.ingestion.neso``.

These tests encode Stage 1 behavioural requirements drawn from:

- ``docs/intent/01-neso-demand-ingestion.md`` (authoritative intent).
- ``docs/architecture/layers/ingestion.md`` (layer-wide contract).
- ``docs/lld/ingestion/neso.md`` §2 (interface), §4 (parquet schema),
  §6 (DST conversion), §7 (cache semantics), §10 (test list).

The DST tests in ``TestToUtcDstBoundaries`` track the lead-accepted
spec-drift from the original LLD §6 sketch. Real Elexon / NESO
settlement periods are numbered contiguously 1..46 on spring-forward
Sundays and 1..50 on autumn-fallback Sundays — there is no "skip period
3-4" on spring-forward, and on autumn-fallback periods 3 and 4 are both
in BST with period 5 being the first GMT (second-occurrence) period.
The implementer's ``_to_utc`` applies the appropriate +/- 60 minute
shifts; these tests assert the shifted mapping rather than the original
LLD code sketch.

Tests use ``pytest.importorskip`` so the suite is green while the
implementer is still working; once the module lands the tests start
running real assertions.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

# Skip the whole module while the implementer hasn't landed yet.
neso = pytest.importorskip("bristol_ml.ingestion.neso")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "neso"
CLOCK_CHANGE_CSV = FIXTURES / "clock_change_rows.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_clock_change_rows(
    *,
    exclude_spring_out_of_range: bool = True,
):
    # Return type intentionally untyped: pandas is imported lazily via
    # importorskip, so we cannot annotate with pd.DataFrame at module load.
    """Load the hand-crafted DST fixture rows.

    ``exclude_spring_out_of_range=True`` (default) drops the adversarial
    spring-forward period-47 row on 2024-03-31 — this is the "valid NESO
    data" shape used by most tests. The spring-forward range-check test
    sets this to ``False`` to feed the adversarial row through
    ``_to_utc`` and assert the raise.

    Under contiguous Elexon numbering, spring-forward days run 1..46
    (not 1..48); a row carrying period 47 on 2024-03-31 is corrupt
    upstream data and is expected to raise.
    """
    df = pd.read_csv(CLOCK_CHANGE_CSV)
    if exclude_spring_out_of_range:
        invalid_spring = (df["SETTLEMENT_DATE"] == "2024-03-31") & (df["SETTLEMENT_PERIOD"] > 46)
        df = df.loc[~invalid_spring].reset_index(drop=True)
    return df


def _build_config(
    tmp_path: Path,
    *,
    resources: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    **overrides: Any,
) -> Any:
    """Build a ``NesoIngestionConfig`` pointing at ``tmp_path`` for cache.

    All tests use ephemeral caches under pytest's tmp_path to avoid
    bleed between runs. ``resources`` defaults to a single 2023 entry
    so tests remain independent of the YAML catalogue.
    """
    from conf._schemas import NesoIngestionConfig  # type: ignore[import-not-found]

    if resources is None:
        resources = [{"year": 2023, "resource_id": UUID("bf5ab335-9b40-4ea4-b93a-ab4af7bce003")}]
    if columns is None:
        columns = ["ND", "TSD"]
    cfg_kwargs: dict[str, Any] = dict(
        resources=resources,
        cache_dir=tmp_path,
        columns=columns,
    )
    cfg_kwargs.update(overrides)
    return NesoIngestionConfig(**cfg_kwargs)


# --------------------------------------------------------------------------- #
# _to_utc — settlement-period → UTC conversion (LLD §6, lead-accepted drift)
# --------------------------------------------------------------------------- #


class TestToUtcDstBoundaries:
    """Cover LLD §6 — BST transition handling under contiguous numbering.

    The original LLD §6 sketch assumed periods on clock-change Sundays
    re-use the 1..48 numbering with period 3 being the missing/ambiguous
    01:00 slot. Real Elexon numbering is contiguous: spring-forward is
    1..46 (the 01:00-02:00 local gap is simply absent from the
    numbering) and autumn-fallback is 1..50 (periods 3-4 cover the
    first 01:00 hour in BST; periods 5-6 cover the repeated 01:00 hour
    in GMT). The implementer's ``_to_utc`` applies the corresponding
    minute shifts; these tests pin that contract.
    """

    def test_spring_forward_period_3_maps_to_02_00_local_after_shift(self) -> None:
        """Period 3 on a spring-forward Sunday is VALID and lands at 02:00 BST.

        Under contiguous 1..46 numbering the 01:00-02:00 local gap does
        not appear in the numbering — period 2 ends at 01:00 GMT /
        02:00 BST, period 3 begins at 02:00 BST. The implementer adds
        a +60 min shift on periods >=3 so the naive local stamp skips
        the vanished hour; ``tz_localize(..., nonexistent='raise')``
        therefore does NOT fire here.
        """
        df = _load_clock_change_rows(exclude_spring_out_of_range=True)
        df = df[df["SETTLEMENT_DATE"] == "2024-03-31"].reset_index(drop=True)
        assert 3 in df["SETTLEMENT_PERIOD"].tolist(), (
            "Fixture must contain a period-3 row on 2024-03-31; "
            "otherwise the spring-forward shift path is not exercised."
        )

        out = neso._to_utc(df.copy())
        spring = out[out["SETTLEMENT_DATE"].astype(str).str.startswith("2024-03-31")]
        period3 = spring[spring["SETTLEMENT_PERIOD"] == 3]
        assert len(period3) == 1, "Expected exactly one period-3 row on 2024-03-31."

        ts_local = pd.Timestamp(period3["timestamp_local"].iloc[0])
        ts_utc = pd.Timestamp(period3["timestamp_utc"].iloc[0])

        # Local wall-clock should be 02:00 on 2024-03-31 in BST (+01:00).
        assert ts_local.tz is not None and "London" in str(ts_local.tz), (
            f"timestamp_local must be tz=Europe/London; got {ts_local.tz!r}"
        )
        assert ts_local.strftime("%Y-%m-%d %H:%M") == "2024-03-31 02:00", (
            f"Spring-forward period 3 must land at 02:00 local (BST); got {ts_local!r}"
        )
        # UTC equivalent is 01:00 on 2024-03-31.
        assert ts_utc.tz_convert("UTC").strftime("%Y-%m-%d %H:%M") == "2024-03-31 01:00", (
            f"Spring-forward period 3 must be 01:00 UTC; got {ts_utc!r}"
        )

    def test_spring_forward_period_47_or_higher_raises(self) -> None:
        """Period > 46 on a spring-forward Sunday is corrupt data and must raise.

        The implementer's explicit per-day range check (spring-forward
        day has 46 periods) rejects any row with period >46 on that
        date. The error must name the offending row so a facilitator
        can diagnose without source-diving.
        """
        df = _load_clock_change_rows(exclude_spring_out_of_range=False)
        df = df[df["SETTLEMENT_DATE"] == "2024-03-31"].reset_index(drop=True)
        assert 47 in df["SETTLEMENT_PERIOD"].tolist(), (
            "Fixture must contain a period-47 row on 2024-03-31 to exercise "
            "the spring-forward range check."
        )

        with pytest.raises(Exception) as exc_info:
            neso._to_utc(df.copy())

        msg = str(exc_info.value)
        # The raise must identify the offending row: either the date, the
        # period number, or a 'settlement period' range-error phrase.
        assert "2024-03-31" in msg or "47" in msg or "settlement period" in msg.lower(), (
            "Spring-forward range-check raise must identify the offending row; "
            f"got: {exc_info.value!r}"
        )

    def test_autumn_fallback_periods_3_and_5_one_hour_apart_in_utc(self) -> None:
        """Period 3 (BST 01:00) and period 5 (GMT 01:00) are 1h apart in UTC.

        Under contiguous 1..50 numbering on autumn-fallback day, both
        period 3 and period 5 carry the *local wall-clock* label 01:00,
        but period 3 is still in BST (UTC 00:00) and period 5 is the
        first GMT slot (UTC 01:00). The UTC delta is exactly one hour.
        """
        df = _load_clock_change_rows()
        out = neso._to_utc(df.copy())

        autumn = out[out["SETTLEMENT_DATE"].astype(str).str.startswith("2024-10-27")]
        period3 = autumn[autumn["SETTLEMENT_PERIOD"] == 3]
        period5 = autumn[autumn["SETTLEMENT_PERIOD"] == 5]
        assert len(period3) == 1 and len(period5) == 1, (
            "Fixture must contain exactly one row each for autumn-fallback periods 3 and 5."
        )

        ts3_utc = pd.Timestamp(period3["timestamp_utc"].iloc[0])
        ts5_utc = pd.Timestamp(period5["timestamp_utc"].iloc[0])
        delta = (ts5_utc - ts3_utc).total_seconds()
        assert delta == 3600, (
            f"Autumn-fallback periods 3 and 5 must be exactly one hour apart in UTC; "
            f"got {delta}s between {ts3_utc!r} and {ts5_utc!r}."
        )

        # Both rows carry the 01:00 local wall-clock label, but the tz differs.
        ts3_local = pd.Timestamp(period3["timestamp_local"].iloc[0])
        ts5_local = pd.Timestamp(period5["timestamp_local"].iloc[0])
        assert ts3_local.strftime("%H:%M") == "01:00", (
            f"Autumn-fallback period 3 local wall-clock must be 01:00; got {ts3_local!r}"
        )
        assert ts5_local.strftime("%H:%M") == "01:00", (
            f"Autumn-fallback period 5 local wall-clock must be 01:00; got {ts5_local!r}"
        )

    def test_autumn_fallback_periods_3_and_4_are_thirty_minutes_apart_in_utc(self) -> None:
        """Period 3 and period 4 are both BST; their UTC timestamps differ by 30 min.

        Pins the contract that the two consecutive periods inside the
        first (BST) 01:00-02:00 hour are simply 30 minutes apart in UTC,
        i.e. no ambiguity resolution kicks in between them.
        """
        df = _load_clock_change_rows()
        out = neso._to_utc(df.copy())

        autumn = out[out["SETTLEMENT_DATE"].astype(str).str.startswith("2024-10-27")]
        period3 = autumn[autumn["SETTLEMENT_PERIOD"] == 3]
        period4 = autumn[autumn["SETTLEMENT_PERIOD"] == 4]
        assert len(period3) == 1 and len(period4) == 1, (
            "Fixture must contain exactly one row each for autumn-fallback periods 3 and 4."
        )

        ts3_utc = pd.Timestamp(period3["timestamp_utc"].iloc[0])
        ts4_utc = pd.Timestamp(period4["timestamp_utc"].iloc[0])
        delta = (ts4_utc - ts3_utc).total_seconds()
        assert delta == 1800, (
            f"Autumn-fallback periods 3 and 4 must be exactly 30 minutes apart in UTC; "
            f"got {delta}s between {ts3_utc!r} and {ts4_utc!r}."
        )

    def test_to_utc_output_is_utc_tz_aware(self) -> None:
        """``timestamp_utc`` must be tz-aware and anchored to UTC (LLD §4)."""
        df = _load_clock_change_rows()
        out = neso._to_utc(df.copy())
        assert "timestamp_utc" in out.columns
        series = out["timestamp_utc"]
        # pandas exposes tz via .dt.tz for Series of Timestamp
        tz = series.dt.tz
        assert tz is not None, "timestamp_utc must be tz-aware"
        assert str(tz) in {"UTC", "utc"}, f"timestamp_utc tz must be UTC, got {tz}"


# --------------------------------------------------------------------------- #
# schema assertion (LLD §5 helper, layer architecture "Schema assertion at ingest")
# --------------------------------------------------------------------------- #


class TestAssertSchema:
    def test_missing_required_column_raises_and_names_it(self) -> None:
        """Missing ``ND`` or ``SETTLEMENT_PERIOD`` raises a named error.

        Per layer architecture: "Required columns missing → hard error
        naming the offending column. No fallback parsing."
        """
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": ["2024-04-01"],
                # SETTLEMENT_PERIOD missing
                "ND": [32000],
                "TSD": [33500],
            }
        )
        with pytest.raises(Exception) as exc_info:
            neso._assert_schema(df, year=2024)
        assert "SETTLEMENT_PERIOD" in str(exc_info.value), (
            f"Missing-required error must name the offending column, got: {exc_info.value!r}"
        )

        df2 = pd.DataFrame(
            {
                "SETTLEMENT_DATE": ["2024-04-01"],
                "SETTLEMENT_PERIOD": [1],
                # ND missing
                "TSD": [33500],
            }
        )
        with pytest.raises(Exception) as exc_info:
            neso._assert_schema(df2, year=2024)
        assert "ND" in str(exc_info.value), (
            f"Missing-required error must name the offending column, got: {exc_info.value!r}"
        )

    def test_unknown_column_warns_and_is_dropped_from_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown NESO column produces a warning and is NOT persisted.

        Layer architecture "Schema assertion at ingest": "Unknown columns
        present → warning and drop." LLD §4: embedded wind/solar and
        interconnector columns are warned-and-dropped, not written.

        We assert both legs: a warning is emitted *and* the column is
        absent from the output parquet.
        """
        # Build a synthetic year dataframe with an unknown column, plumb it
        # through `fetch` via monkeypatched `_fetch_year` so the test does
        # not touch the network.
        raw = pd.DataFrame(
            {
                "SETTLEMENT_DATE": ["2024-04-01", "2024-04-01"],
                "SETTLEMENT_PERIOD": [1, 2],
                "ND": [32000, 31800],
                "TSD": [33500, 33300],
                # Post-2023 interconnector example — not in our retained set.
                "VIKING_FLOW": [0, 0],
            }
        )

        def _fake_fetch_year(*args: Any, **kwargs: Any):
            # Accept whichever positional signature the implementer settles on
            # (e.g. (client, year, resource_id, config) or (year, resource_id, config)).
            return raw.copy()

        # Preferred seam: monkeypatch the private helper.
        if hasattr(neso, "_fetch_year"):
            monkeypatch.setattr(neso, "_fetch_year", _fake_fetch_year)
        else:
            pytest.skip("neso._fetch_year not present; cannot exercise warn-and-drop end-to-end")

        cfg = _build_config(tmp_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = neso.fetch(cfg, cache=neso.CachePolicy.REFRESH)

        assert any("VIKING_FLOW" in str(w.message) for w in caught), (
            "Unknown column must produce a warning naming it; "
            f"got warnings: {[str(w.message) for w in caught]}"
        )

        # Persisted parquet must not carry the unknown column.
        assert path.exists(), f"REFRESH must write parquet, missing at {path}"
        table = pq.read_table(path)
        assert "VIKING_FLOW" not in table.column_names, (
            "Unknown columns must be dropped from the persisted parquet; "
            f"got columns: {table.column_names}"
        )


# --------------------------------------------------------------------------- #
# CachePolicy semantics (LLD §7)
# --------------------------------------------------------------------------- #


class TestCachePolicy:
    def test_offline_raises_when_cache_missing_and_names_path(self, tmp_path: Path) -> None:
        """``CachePolicy.OFFLINE`` + no cache → error naming the expected path.

        Maps to acceptance criterion 1 (ensures offline stance is correct
        when cache is absent) and LLD §7.
        """
        cfg = _build_config(tmp_path)
        expected_path = tmp_path / "neso_demand.parquet"
        assert not expected_path.exists(), "Precondition: cache must be absent"

        with pytest.raises(Exception) as exc_info:
            neso.fetch(cfg, cache=neso.CachePolicy.OFFLINE)

        # The error must name the expected path so a user can see what's missing.
        assert str(expected_path) in str(exc_info.value) or expected_path.name in str(
            exc_info.value
        ), f"OFFLINE + missing cache error must name the expected path, got: {exc_info.value!r}"
        # LLD §7 names the error type as CacheMissingError; enforce it if
        # exposed, otherwise accept any raise whose message names the path.
        cache_missing_error = getattr(neso, "CacheMissingError", None)
        if cache_missing_error is not None:
            assert isinstance(exc_info.value, cache_missing_error), (
                f"Expected CacheMissingError, got {type(exc_info.value).__name__}"
            )

    def test_auto_returns_cached_path_without_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CachePolicy.AUTO`` + cache present → zero HTTP calls.

        Acceptance criterion 1: "Running the ingestion with a cache
        present completes offline." We prove this by patching the
        network seam(s) to explode on any call.
        """
        cfg = _build_config(tmp_path)
        cache_path = tmp_path / cfg.cache_filename
        # Write a minimal, well-formed parquet so `fetch` has something
        # to return. The round-trip is what we care about (no network),
        # not the payload shape.
        schema = pa.schema(
            [
                pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
                pa.field("timestamp_local", pa.timestamp("us", tz="Europe/London")),
                pa.field("settlement_date", pa.date32()),
                pa.field("settlement_period", pa.int8()),
                pa.field("nd_mw", pa.int32()),
                pa.field("tsd_mw", pa.int32()),
                pa.field("source_year", pa.int16()),
                pa.field("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
            ]
        )
        empty_table = pa.Table.from_pylist([], schema=schema)
        pq.write_table(empty_table, cache_path)
        assert cache_path.exists()

        # Network tripwire: any HTTP seam blows up.
        def _explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "AUTO + cache present must not touch the network; "
                f"called with args={args!r} kwargs={kwargs!r}"
            )

        # Patch every plausible network entry point.
        httpx = pytest.importorskip("httpx")
        monkeypatch.setattr(httpx, "get", _explode, raising=False)
        monkeypatch.setattr(httpx.Client, "get", _explode, raising=False)
        monkeypatch.setattr(httpx.Client, "send", _explode, raising=False)
        if hasattr(neso, "_retrying_get"):
            monkeypatch.setattr(neso, "_retrying_get", _explode)
        if hasattr(neso, "_fetch_year"):
            monkeypatch.setattr(neso, "_fetch_year", _explode)

        returned = neso.fetch(cfg, cache=neso.CachePolicy.AUTO)
        assert Path(returned) == cache_path, (
            f"AUTO must return the existing cache path; got {returned!r}, expected {cache_path!r}"
        )


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #


def test_cli_help_exits_zero() -> None:
    """``python -m bristol_ml.ingestion.neso --help`` exits 0.

    LLD §10 smoke test; principle §2.1.1 (every module runs standalone).
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.ingestion.neso", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI --help must exit 0; stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# --------------------------------------------------------------------------- #
# Module-local CLAUDE.md documents the schema (acceptance criterion 4)
# --------------------------------------------------------------------------- #


def test_module_claude_md_documents_schema() -> None:
    """Acceptance criterion 4: output schema documented in module's CLAUDE.md.

    Not a runtime assertion of behaviour but a regression guard that the
    file exists and lists the columns. Per the brief, this is a
    lightweight check to prevent the stage from shipping with the
    schema undocumented.
    """
    repo_root = Path(__file__).resolve().parents[3]
    module_claude = repo_root / "src" / "bristol_ml" / "ingestion" / "CLAUDE.md"
    assert module_claude.exists(), (
        f"Module CLAUDE.md must exist at {module_claude} "
        "(acceptance criterion 4: output schema documented there)."
    )
    text = module_claude.read_text(encoding="utf-8")
    # A very loose markdown-table guard: require a header containing 'Column'.
    # Avoids locking the exact formatting but catches "file is empty".
    assert "Column" in text, (
        "Module CLAUDE.md should contain a schema column table (look for a 'Column' header)."
    )
    # At minimum the canonical column names are mentioned.
    for col in (
        "timestamp_utc",
        "settlement_period",
        "nd_mw",
        "tsd_mw",
        "source_year",
        "retrieved_at_utc",
    ):
        assert col in text, f"Schema doc must mention column {col!r}."
