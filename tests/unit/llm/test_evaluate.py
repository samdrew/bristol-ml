"""Spec-derived tests for the Stage 14 T5 evaluation harness.

Plan §6 T5 named tests:

- ``test_evaluate_harness_runs_against_stub`` — AC-4: harness exits 0
  against the stub, stdout carries the per-field summary table.
- ``test_evaluate_harness_output_is_deterministic_modulo_timestamp`` —
  AC-4 / NFR-3: two harness runs produce byte-identical stdout when
  the ``--redact-timestamps`` flag is set.
- ``test_evaluate_harness_records_provenance`` — AC-4: stdout header
  contains the prompt hash, model id, gold-set hash, gold-set size,
  and implementation name.

Plan §1 D11 / D12 tolerance metric tests round out the suite — the
disagreement listing is the demo punch line, so a mismatch within
tolerance must register as a tolerance hit + an exact miss.

The harness imports ``StubExtractor``-internal helpers (`_GoldSetExpected`,
`_GoldSetRecord`, `_load_gold_set`) so the tests are coupled to the
internal contract; that's intentional — the harness lives in the same
package and the contract is more stable than the public stub
constructor signature.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bristol_ml.llm import ExtractionResult, RemitEvent
from bristol_ml.llm.evaluate import (
    REDACTED_TIMESTAMP_SENTINEL,
    TOLERANCE_CAPACITY_MW,
    _cli_main,
    evaluate,
    format_report,
)
from conf._schemas import LlmExtractorConfig

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _make_minimal_gold_fixture(tmp_path: Path) -> Path:
    """Write a 2-record gold-set JSON fixture and return its path.

    Two records is the smallest size that still exercises the
    iteration order (sorted by mrid) and the divisor of the per-field
    fractions. Larger fixtures are unnecessary for the harness's
    behavioural contract.
    """
    payload = {
        "schema_version": 1,
        "description": "test fixture",
        "records": [
            {
                "event": {
                    "mrid": "T-A",
                    "revision_number": 0,
                    "message_status": "Active",
                    "published_at": "2024-01-01T00:00:00Z",
                    "effective_from": "2024-01-15T00:00:00Z",
                    "effective_to": "2024-01-20T00:00:00Z",
                    "fuel_type": "Gas",
                    "affected_mw": 100.0,
                    "event_type": "Outage",
                    "cause": "Planned",
                    "message_description": "test gas outage",
                },
                "expected": {
                    "event_type": "Outage",
                    "fuel_type": "Gas",
                    "affected_capacity_mw": 100.0,
                    "effective_from": "2024-01-15T00:00:00Z",
                    "effective_to": "2024-01-20T00:00:00Z",
                    "confidence": 1.0,
                },
            },
            {
                "event": {
                    "mrid": "T-B",
                    "revision_number": 0,
                    "message_status": "Active",
                    "published_at": "2024-02-01T00:00:00Z",
                    "effective_from": "2024-02-10T00:00:00Z",
                    "effective_to": None,
                    "fuel_type": "Wind",
                    "affected_mw": 200.0,
                    "event_type": "Restriction",
                    "cause": "Forced",
                    "message_description": "test wind restriction",
                },
                "expected": {
                    "event_type": "Restriction",
                    "fuel_type": "Wind",
                    "affected_capacity_mw": 200.0,
                    "effective_from": "2024-02-10T00:00:00Z",
                    "effective_to": None,
                    "confidence": 1.0,
                },
            },
        ],
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# AC-4: ``test_evaluate_harness_runs_against_stub``
# --------------------------------------------------------------------------- #


def test_evaluate_harness_runs_against_stub(tmp_path: Path) -> None:
    """Plan §6 T5 / AC-4: harness runs against stub and reports 100 % agreement.

    The stub on its own gold set is the trivial sanity case — every
    field's exact + tolerance counts equal the gold-set size. A
    failure here means either the harness is mis-counting or the
    stub's ``extract`` no longer matches its own labels.
    """
    fixture = _make_minimal_gold_fixture(tmp_path)
    config = LlmExtractorConfig(type="stub")

    report = evaluate(config, gold_set_path=fixture)

    assert report.implementation == "StubExtractor"
    assert report.gold_set_size == 2
    assert report.gold_set_path == fixture
    for name in (
        "event_type",
        "fuel_type",
        "affected_capacity_mw",
        "effective_from",
        "effective_to",
    ):
        agreement = report.per_field[name]
        assert agreement.exact_match == 2, (
            f"AC-4: stub on its own gold set must agree on {name}; "
            f"got exact_match={agreement.exact_match}/2."
        )
        assert agreement.tolerance_match == 2
    assert report.disagreements == []


# --------------------------------------------------------------------------- #
# AC-4 / NFR-3: deterministic output modulo timestamp
# --------------------------------------------------------------------------- #


def test_evaluate_harness_output_is_deterministic_modulo_timestamp(
    tmp_path: Path,
) -> None:
    """Plan §6 T5 / NFR-3: two runs produce byte-identical stdout under redaction.

    The only varying dimension between two harness invocations on the
    same gold set is the wall-clock ``generated_at`` header line. The
    ``--redact-timestamps`` flag (and ``redact_timestamps=True`` on
    :func:`format_report`) swaps that line for a fixed sentinel so
    test asserts on byte-equality.
    """
    fixture = _make_minimal_gold_fixture(tmp_path)
    config = LlmExtractorConfig(type="stub")

    # Two runs at *different* wall-clock times — the test would
    # otherwise pass trivially if the second run reused the first's
    # timestamp.
    report_a = evaluate(
        config,
        gold_set_path=fixture,
        now=datetime(2026, 4, 27, 9, 0, 0, tzinfo=UTC),
    )
    report_b = evaluate(
        config,
        gold_set_path=fixture,
        now=datetime(2026, 4, 27, 9, 0, 5, tzinfo=UTC),
    )

    text_a = format_report(report_a, redact_timestamps=True)
    text_b = format_report(report_b, redact_timestamps=True)

    assert text_a == text_b, (
        "NFR-3: two harness runs with --redact-timestamps must produce byte-identical stdout."
    )
    # Sanity: redacted line is present + carries the sentinel.
    assert REDACTED_TIMESTAMP_SENTINEL in text_a

    # And without the redaction the two outputs differ only on the
    # generated_at line — every other line is byte-identical, so the
    # diff is exactly two lines (one per side).
    text_a_unredacted = format_report(report_a)
    text_b_unredacted = format_report(report_b)
    diff_lines_a = [
        line
        for line in text_a_unredacted.splitlines()
        if line not in text_b_unredacted.splitlines()
    ]
    assert len(diff_lines_a) == 1 and "generated_at" in diff_lines_a[0]


# --------------------------------------------------------------------------- #
# AC-4: provenance recording
# --------------------------------------------------------------------------- #


def test_evaluate_harness_records_provenance(tmp_path: Path) -> None:
    """Plan §6 T5 / AC-4: every dimension that would change the result is named.

    The plan §5 header layout is binding: implementation name, prompt
    hash (or the literal ``none``), model id (or ``none``), gold-set
    path + sha256 prefix + size, and the generated-at timestamp. A
    missing field would mean an operator could swap something
    invisibly between two reports.
    """
    fixture = _make_minimal_gold_fixture(tmp_path)
    config = LlmExtractorConfig(type="stub")

    report = evaluate(config, gold_set_path=fixture)
    text = format_report(report)

    # Implementation name
    assert "implementation: StubExtractor" in text
    # Stub path: prompt_hash and model_id are 'none'
    assert "prompt_hash:    none" in text
    assert "model_id:       none" in text
    # Gold-set provenance
    assert str(fixture) in text
    assert f"n={report.gold_set_size}" in text
    # Sha256 prefix should be 12 hex chars
    assert re.search(r"sha256: [0-9a-f]{12}", text), (
        f"Provenance: header must carry a 12-char SHA-256 hex prefix; saw {text.splitlines()[4]!r}"
    )
    # generated_at line is present
    assert "generated_at:" in text


# --------------------------------------------------------------------------- #
# Plan §1 D11/D12: two-metric design — tolerance counts when exact misses
# --------------------------------------------------------------------------- #


def test_capacity_within_tolerance_passes_for_small_drift(tmp_path: Path) -> None:
    """Plan §1 D12: a ±2 MW drift counts as exact miss + tolerance hit.

    Builds a gold set where the stub's *expected* capacity differs
    from the *actual* extracted capacity by 2 MW (within the ±5 MW
    tolerance). The exact column should report 0/1, the tolerance
    column 1/1 — the demo's whole point.
    """
    # Construct an extraction whose actual diverges from expected by
    # 2 MW. We do this via a custom Extractor that returns a fixed
    # ExtractionResult regardless of input — bypasses the stub and
    # makes the test a pure harness-counting check.
    fixture_path = _make_minimal_gold_fixture(tmp_path)

    class _DriftExtractor:
        """Returns an ExtractionResult whose capacity is 2 MW off."""

        def extract(self, event: RemitEvent) -> ExtractionResult:
            assert event.affected_mw is not None
            return ExtractionResult(
                event_type=event.event_type or "Other",
                fuel_type=event.fuel_type or "Other",
                affected_capacity_mw=event.affected_mw + 2.0,
                effective_from=event.effective_from,
                effective_to=event.effective_to,
                confidence=1.0,
                prompt_hash="abcdef012345",
                model_id="test-model",
            )

        def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
            return [self.extract(e) for e in events]

    # Monkey-patch build_extractor in evaluate's namespace so the test
    # uses the drift extractor without going through the factory.
    import bristol_ml.llm.evaluate as eval_mod

    real_build = eval_mod.build_extractor
    eval_mod.build_extractor = lambda cfg, gold_set_path=None: _DriftExtractor()
    try:
        report = evaluate(
            LlmExtractorConfig(type="openai", model_name="test"),
            gold_set_path=fixture_path,
        )
    finally:
        eval_mod.build_extractor = real_build

    capacity = report.per_field["affected_capacity_mw"]
    assert capacity.exact_match == 0, (
        f"D11: exact column must reject ±2 MW drift; got {capacity.exact_match}/{capacity.total}."
    )
    assert capacity.tolerance_match == 2, (
        f"D12: tolerance column (±{TOLERANCE_CAPACITY_MW} MW) must "
        f"accept ±2 MW drift; got {capacity.tolerance_match}/{capacity.total}."
    )


def test_datetime_tolerance_accepts_30_minute_drift(tmp_path: Path) -> None:
    """Plan §1 D12: a 30-minute drift counts as exact miss + tolerance hit on
    ``effective_from``.

    ±1 hour is the documented threshold; 30 minutes is half-way. The
    test checks the harness wires the tolerance check rather than the
    exact one for time fields.
    """
    fixture_path = _make_minimal_gold_fixture(tmp_path)

    class _TimeDriftExtractor:
        def extract(self, event: RemitEvent) -> ExtractionResult:
            return ExtractionResult(
                event_type=event.event_type or "Other",
                fuel_type=event.fuel_type or "Other",
                affected_capacity_mw=event.affected_mw,
                effective_from=event.effective_from + timedelta(minutes=30),
                effective_to=event.effective_to,
                confidence=1.0,
                prompt_hash="abcdef012345",
                model_id="test-model",
            )

        def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
            return [self.extract(e) for e in events]

    import bristol_ml.llm.evaluate as eval_mod

    real_build = eval_mod.build_extractor
    eval_mod.build_extractor = lambda cfg, gold_set_path=None: _TimeDriftExtractor()
    try:
        report = evaluate(
            LlmExtractorConfig(type="openai", model_name="test"),
            gold_set_path=fixture_path,
        )
    finally:
        eval_mod.build_extractor = real_build

    eff_from = report.per_field["effective_from"]
    assert eff_from.exact_match == 0
    assert eff_from.tolerance_match == 2, (
        f"D12: ±1 h tolerance must accept 30-minute drift; got "
        f"{eff_from.tolerance_match}/{eff_from.total}."
    )


def test_disagreements_listed_with_mrid_and_field(tmp_path: Path) -> None:
    """Plan §5: disagreement listing reads ``mrid.field expected=... got=...``.

    A non-empty disagreement set must surface with both the mrid and
    the field name so the demo can talk about *where* the LLM gets it
    wrong, not just *that* it does.
    """
    fixture_path = _make_minimal_gold_fixture(tmp_path)

    class _WrongFuelExtractor:
        def extract(self, event: RemitEvent) -> ExtractionResult:
            return ExtractionResult(
                event_type=event.event_type or "Other",
                fuel_type="Other",  # Always wrong
                affected_capacity_mw=event.affected_mw,
                effective_from=event.effective_from,
                effective_to=event.effective_to,
                confidence=1.0,
                prompt_hash="deadbeefcafe",
                model_id="test-model",
            )

        def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
            return [self.extract(e) for e in events]

    import bristol_ml.llm.evaluate as eval_mod

    real_build = eval_mod.build_extractor
    eval_mod.build_extractor = lambda cfg, gold_set_path=None: _WrongFuelExtractor()
    try:
        report = evaluate(
            LlmExtractorConfig(type="openai", model_name="test"),
            gold_set_path=fixture_path,
        )
    finally:
        eval_mod.build_extractor = real_build

    # Two records, fuel_type wrong on both
    text = format_report(report)
    assert "T-A.fuel_type" in text
    assert "T-B.fuel_type" in text
    assert "expected='Gas'" in text
    assert "expected='Wind'" in text
    assert "got='Other'" in text
    # Header should not redact provenance
    assert "model_id:       test-model" in text


def test_max_disagreements_truncates_and_appends_count(tmp_path: Path) -> None:
    """Plan §5: disagreement listing is capped; tail records are summarised.

    The default cap is 10; passing ``max_disagreements=1`` against a
    fixture with two disagreements must show one and append
    ``... and 1 more``.
    """
    fixture_path = _make_minimal_gold_fixture(tmp_path)

    class _WrongFuelExtractor:
        def extract(self, event: RemitEvent) -> ExtractionResult:
            return ExtractionResult(
                event_type=event.event_type or "Other",
                fuel_type="Other",
                affected_capacity_mw=event.affected_mw,
                effective_from=event.effective_from,
                effective_to=event.effective_to,
                confidence=1.0,
            )

        def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
            return [self.extract(e) for e in events]

    import bristol_ml.llm.evaluate as eval_mod

    real_build = eval_mod.build_extractor
    eval_mod.build_extractor = lambda cfg, gold_set_path=None: _WrongFuelExtractor()
    try:
        report = evaluate(
            LlmExtractorConfig(type="openai", model_name="test"),
            gold_set_path=fixture_path,
            max_disagreements=1,
        )
    finally:
        eval_mod.build_extractor = real_build

    text = format_report(report, max_disagreements=1)
    assert "... and 1 more" in text


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_evaluate_module_runs_standalone_help() -> None:
    """Plan NFR-8: ``python -m bristol_ml.llm.evaluate --help`` exits 0.

    The CLI must surface its docstring without instantiating an
    extractor (no API key, no gold-set load), so ``--help`` short-circuits
    before either path matters.
    """
    with pytest.raises(SystemExit) as exc_info:
        _cli_main(["--help"])
    assert exc_info.value.code == 0


def test_evaluate_cli_runs_against_stub_via_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-4: invoking ``_cli_main([])`` exits 0 + prints the report.

    Mirrors the in-process invocation Stage 6's evaluation harness CLI
    test uses. The default config selects the stub (NFR-1) so this
    test runs without an API key.
    """
    # Force the stub regardless of any ambient API key in the env
    monkeypatch.setenv("BRISTOL_ML_LLM_STUB", "1")

    rc = _cli_main(["--redact-timestamps"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "=== Stage 14 LLM extractor evaluation ===" in captured.out
    assert "implementation: StubExtractor" in captured.out
    assert "=== end ===" in captured.out
    # Reproducibility flag swapped in the sentinel
    assert REDACTED_TIMESTAMP_SENTINEL in captured.out


def test_loguru_summary_record_emitted_on_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """Plan §1 D15 / Ctrl+G OQ-D: harness emits a structured INFO record.

    The non-notebook (CLI / orchestration) path captures the report
    into the standard log stream as one ``loguru.logger.info`` call
    with ``extra={"summary": {...}}``. The test asserts the record
    is emitted at INFO and carries the summary payload.
    """
    monkeypatch.setenv("BRISTOL_ML_LLM_STUB", "1")
    fixture = _make_minimal_gold_fixture(tmp_path)

    # Patch DEFAULT_GOLD_SET_PATH lookup — the CLI uses the default
    # path; for the test we want the small fixture.  Pass-through:
    # call the public evaluate() with the path explicitly and emit
    # the loguru summary as the CLI would.
    from bristol_ml.llm.evaluate import _emit_loguru_summary

    config = LlmExtractorConfig(type="stub")
    report = evaluate(config, gold_set_path=fixture)
    _emit_loguru_summary(report)

    # The bridge fixture surfaces loguru records as logging.LogRecord
    # objects; we look for the one whose message names the harness.
    matching = [rec for rec in loguru_caplog.records if "Stage 14 evaluation" in rec.getMessage()]
    assert matching, (
        "OQ-D: harness must emit a structured INFO record naming the "
        f"evaluation; saw {[r.getMessage() for r in loguru_caplog.records]}."
    )
    msg = matching[-1].getMessage()
    assert "implementation=StubExtractor" in msg
    assert "gold_set_size=2" in msg
    assert "disagreements=0" in msg
