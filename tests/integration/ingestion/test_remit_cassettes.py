"""Integration tests for ``bristol_ml.ingestion.remit`` against a recorded cassette.

The implementer records the cassette once
(``uv run pytest tests/integration/ingestion/test_remit_cassettes.py
--record-mode=once``) against the live Elexon Insights API; CI replays
it with ``--record-mode=none`` (configured in ``pyproject.toml``).

All integration tests in this module share a single recorded cassette
``tests/fixtures/remit/cassettes/remit_2024_01_01.yaml`` — a one-day
window of ``/datasets/REMIT/stream`` carrying ~125 messages across 70
mRIDs and exhibiting at least 31 in-window revision chains.  Cassette
size target ≤ 100 kB compressed (plan §1 D11; observed 2026-04 raw JSON
~140 kB before VCR's body filtering).

Tests encode:

- AC-3 — no cache → fetch + write local copy (T4 / D18f).
- AC-4 / NFR-1 — two REFRESH runs produce row-identical content modulo
  the per-fetch ``retrieved_at_utc`` provenance column (T4 / D18c).
- AC-1 — ``as_of(df, t)`` returns invariants-respecting active state at
  three sample times spanning the cassette window (T4 / D18h).
- NFR-10 — ``loguru`` INFO record with the live-fetch step + record
  count (T4).

The synthetic withdrawal fixture (plan D11) lives in the unit-test file
``tests/unit/ingestion/test_remit.py`` because the live Insights API
emits ``Active`` / ``Inactive`` / ``Dismissed`` only — a
``Withdrawn`` row is unlikely to appear in any naturally selected
window and would balloon the cassette if forced.  The unit-level
``test_as_of_withdrawn_message_excludes_row`` covers AC-1(c).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

remit = pytest.importorskip("bristol_ml.ingestion.remit")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pytest_recording")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "remit"
CASSETTES = FIXTURES / "cassettes"
BULK_CASSETTE = "remit_2024_01_01.yaml"
BULK_CASSETTE_STEM = "remit_2024_01_01"


# --------------------------------------------------------------------------- #
# pytest-recording wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _cassette_present_or_skip(request: pytest.FixtureRequest) -> None:
    """Skip the suite when no cassette exists yet (build-up phase).

    Honoured under ``--record-mode=none`` (the CI default).  When the
    implementer is recording (``--record-mode=once`` / ``--record-mode=new_episodes``)
    the skip is bypassed so VCR can populate the cassette against the
    live API.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    if record_mode != "none":
        return
    if not (CASSETTES / BULK_CASSETTE).exists():
        pytest.skip(
            f"No cassette at {CASSETTES / BULK_CASSETTE}; record once via "
            "`uv run pytest --record-mode=once` before CI runs."
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
def vcr_config(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Filter sensitive headers; allow replay repeats for idempotence checks.

    ``allow_playback_repeats=True`` lets the idempotent re-fetch test
    issue two consecutive REFRESH calls against the same cassette — VCR
    would otherwise raise "no more recordings for this request" on the
    second match.

    The ``record_mode`` defaults to ``"none"`` (CI replay) and is taken
    from the ``--record-mode`` CLI flag when the implementer is
    recording locally.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": record_mode,
        "allow_playback_repeats": True,
    }


def _build_config(tmp_path: Path, **overrides: Any) -> Any:
    """Build a ``RemitIngestionConfig`` for a one-day window.

    The window matches the recorded cassette exactly; changing it here
    would mismatch the recorded ``publishDateTimeFrom`` /
    ``publishDateTimeTo`` query parameters and trigger VCR's
    "no match found" error on replay.
    """
    from datetime import date

    from conf._schemas import RemitIngestionConfig

    kwargs: dict[str, Any] = {
        "cache_dir": tmp_path,
        "window_start": date(2024, 1, 1),
        "window_end": date(2024, 1, 2),
        "min_inter_request_seconds": 0.0,  # speed up the test; VCR is local I/O
    }
    kwargs.update(overrides)
    return RemitIngestionConfig(**kwargs)


# --------------------------------------------------------------------------- #
# T4 plan tests
# --------------------------------------------------------------------------- #


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_fetch_against_cassette_populates_cache(tmp_path: Path) -> None:
    """T4 / AC-3 / D18f: REFRESH against cassette writes a canonical parquet.

    Asserts (a) the cache file is created at the configured path, (b) its
    on-disk schema equals :data:`remit.OUTPUT_SCHEMA`, and (c) the
    record count is non-trivially large (≥ 100) — the cassette covers a
    full day at the live record density (~125 records observed).
    """
    cfg = _build_config(tmp_path)
    path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)

    assert Path(path).exists(), f"REFRESH must write parquet at {path}"

    table = pq.read_table(path)
    assert table.schema == remit.OUTPUT_SCHEMA, (
        f"Persisted schema must match OUTPUT_SCHEMA exactly.\n"
        f"actual={table.schema}\nexpected={remit.OUTPUT_SCHEMA}"
    )
    assert table.num_rows >= 100, (
        f"Cassette is expected to carry ≥ 100 records (one day at live density); "
        f"got {table.num_rows}."
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_fetch_idempotent_against_cassette(tmp_path: Path) -> None:
    """T4 / AC-4 / NFR-1 / D18c: two REFRESH runs produce row-identical output.

    The ``retrieved_at_utc`` column is a per-fetch provenance scalar and
    will differ between consecutive runs by design (NFR-9); every other
    column must match row-for-row after stable sort.  This is the
    plan's "row-identical after sort" form of the byte-identical
    invariant — the only thing that changes between runs is the
    provenance stamp.
    """
    cfg = _build_config(tmp_path / "first")
    cfg2 = _build_config(tmp_path / "second")

    path_a = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)
    path_b = remit.fetch(cfg2, cache=remit.CachePolicy.REFRESH)

    df_a = remit.load(path_a).drop(columns=["retrieved_at_utc"])
    df_b = remit.load(path_b).drop(columns=["retrieved_at_utc"])

    # Sort both frames identically before comparing — the on-disk order
    # is already deterministic via _to_arrow's stable sort, but reset
    # the index so frame-equality compares values not positions.
    df_a_sorted = df_a.sort_values(
        ["published_at", "mrid", "revision_number"], kind="stable"
    ).reset_index(drop=True)
    df_b_sorted = df_b.sort_values(
        ["published_at", "mrid", "revision_number"], kind="stable"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_a_sorted,
        df_b_sorted,
        check_dtype=True,
        check_exact=True,
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_as_of_against_cassette_fixture_at_three_sample_times(tmp_path: Path) -> None:
    """T4 / AC-1 / D18h: ``as_of`` produces the exact expected state at three sample times.

    Sample times across the cassette window:

    - ``t_early``  — 06:00 on the cassette day; partial set of messages
      have been published.
    - ``t_late``   — 23:30 on the cassette day; nearly all messages
      have been published.
    - ``t_after``  — 02:00 on the day after; every message in the
      cassette has been published.

    The test asserts both **structural invariants** (the algorithm's
    contract) and **exact-count expectations** (deterministic numbers
    derived from the committed cassette).  The exact counts pin
    ``as_of`` against silent algorithmic regression — any change in
    the transaction-time filter, the ``idxmax`` revision selection, or
    the ``Withdrawn`` exclusion would shift at least one of the three
    numbers.  A re-record of the cassette would also trigger this
    test, forcing the implementer to re-derive and re-pin the counts
    deliberately rather than weaken them silently.

    Counts derived against the committed cassette
    ``remit_2024_01_01.yaml`` (125 records, 70 mRIDs, 31 in-window
    revision chains):

    - ``len(early) == 9``  — only the earliest-published 9 mRIDs are
      visible at 06:00 UTC.
    - ``len(late) == 64``  — most mRIDs published by end-of-day, six
      remaining late-night disclosures arrive between 23:30 and EOD.
    - ``len(after) == 70`` — the full mRID set, matching the cassette's
      total ``df['mrid'].nunique()``.

    Structural invariants asserted in addition to the exact counts:

    1. The result has unique ``mrid`` per the plan §5 ``as_of`` contract.
    2. No row carries ``message_status == "Withdrawn"``.
    3. Every returned row's ``published_at`` is ``<= t``.
    4. The mRID set at ``t_early`` is a strict subset of ``t_late``,
       which is a strict subset of ``t_after``.
    5. ``len(after)`` equals the cassette's total ``df['mrid'].nunique()``
       — a "did we lose any mRID?" cross-check.
    """
    cfg = _build_config(tmp_path)
    path = remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)
    df = remit.load(path)

    t_early = pd.Timestamp("2024-01-01T06:00:00", tz="UTC")
    t_late = pd.Timestamp("2024-01-01T23:30:00", tz="UTC")
    t_after = pd.Timestamp("2024-01-02T02:00:00", tz="UTC")

    early = remit.as_of(df, t_early)
    late = remit.as_of(df, t_late)
    after = remit.as_of(df, t_after)

    # Exact-count expectations — pin the algorithm against silent regression.
    # See docstring; counts derived from remit_2024_01_01.yaml.
    assert len(early) == 9, f"as_of(t_early=06:00 UTC) must return 9 mRIDs; got {len(early)}."
    assert len(late) == 64, f"as_of(t_late=23:30 UTC) must return 64 mRIDs; got {len(late)}."
    assert len(after) == 70, f"as_of(t_after=02:00 UTC+1) must return 70 mRIDs; got {len(after)}."

    for label, frame, t in (
        ("early", early, t_early),
        ("late", late, t_late),
        ("after", after, t_after),
    ):
        # Invariant 1: unique mrid per result.
        assert frame["mrid"].is_unique, (
            f"as_of({label}) must return one row per mrid; got duplicates."
        )
        # Invariant 2: no withdrawn rows survive.
        assert (frame["message_status"] != "Withdrawn").all(), (
            f"as_of({label}) must drop Withdrawn rows; got "
            f"{(frame['message_status'] == 'Withdrawn').sum()} withdrawal(s)."
        )
        # Invariant 3: every row's transaction-time is <= t.
        if not frame.empty:
            assert (frame["published_at"] <= t).all(), (
                f"as_of({label}) must filter published_at <= t={t}; got "
                f"{(frame['published_at'] > t).sum()} row(s) violating it."
            )

    # Invariant 4: strict subset chain — every mRID known at t_early is
    # known at t_late, every mRID known at t_late is known at t_after.
    early_mrids = set(early["mrid"])
    late_mrids = set(late["mrid"])
    after_mrids = set(after["mrid"])
    assert early_mrids.issubset(late_mrids), (
        f"mRID set at t_early must be a subset of t_late; orphans: {early_mrids - late_mrids}"
    )
    assert late_mrids.issubset(after_mrids), (
        f"mRID set at t_late must be a subset of t_after; orphans: {late_mrids - after_mrids}"
    )

    # Invariant 5: t_after returns the cassette's full mRID set (no row
    # silently dropped by the algorithm — the only Withdrawn-filtered
    # mRIDs would surface here as missing-from-after, and the cassette
    # carries no Withdrawn rows in the recorded window).
    assert after_mrids == set(df["mrid"]), (
        f"as_of(t_after) must return every mRID in the cassette; "
        f"missing: {set(df['mrid']) - after_mrids}, "
        f"extra: {after_mrids - set(df['mrid'])}."
    )


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip")
def test_remit_logs_paging_step_at_info(
    tmp_path: Path,
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """T4 / NFR-10: live fetch emits an INFO record with record count + window.

    The `/datasets/REMIT/stream` endpoint is a single GET (no paging),
    so the "paging step" log line is the same as the response-summary
    line.  Per NFR-10 the line must carry the record count and the
    window slice so an operator running with ``loguru`` at INFO can
    audit the fetch.
    """
    cfg = _build_config(tmp_path)
    remit.fetch(cfg, cache=remit.CachePolicy.REFRESH)

    matches = [
        record
        for record in loguru_caplog.records
        if "REMIT live fetch" in record.getMessage() and "record(s)" in record.getMessage()
    ]
    assert matches, (
        "NFR-10 requires an INFO log carrying 'REMIT live fetch' + record count; "
        f"saw {[r.getMessage() for r in loguru_caplog.records]}"
    )
    summary = matches[-1].getMessage()
    assert "2024-01-01" in summary and "2024-01-02" in summary, (
        f"INFO summary must include the window slice; got {summary!r}."
    )
