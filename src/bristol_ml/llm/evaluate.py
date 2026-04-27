"""Stage 14 — evaluation harness for the LLM feature extractor.

Plan §6 T5: runs the active extractor over the hand-labelled gold set
and prints a deterministic side-by-side accuracy report — the demo
moment named in intent §"Demo moment". The same content is emitted
as a single structured ``loguru`` INFO record (plan §1 D15 / OQ-D)
so a non-notebook CLI run leaves a captured trace in the standard log
stream rather than disappearing into stdout.

Output format (plan §5):

::

    === Stage 14 LLM extractor evaluation ===
    implementation: stub
    prompt_hash:    none
    model_id:       none
    gold_set:       tests/fixtures/llm/hand_labelled.json (sha256: a1b2c3..., n=76)
    generated_at:   2026-04-27T14:23:11Z   (← redacted in tests)

    per-field agreement:
      field                   exact   tolerance
      event_type              76/76    76/76
      fuel_type               76/76    76/76
      affected_capacity_mw    76/76    76/76    (tolerance: ±5 MW)
      effective_from          76/76    76/76    (tolerance: ±1 h)
      effective_to            76/76    76/76    (tolerance: ±1 h)

    disagreements (first 10):
      (none — stub on its own gold set always agrees)

    === end ===

Two-metric design (plan §1 D11): exact-match-per-field *and*
tolerance/F1 metric on the same fields, side-by-side. The tolerance
column carries representative thresholds (plan §1 D12 — ±5 MW for
``affected_capacity_mw``, ±1 h for the two timestamps); these are
documented choices, not magic numbers, and the implementer may
revise based on observed gold-set distribution.

Reproducibility (plan NFR-3): two runs with the same inputs produce
byte-identical stdout *modulo* the single ``generated_at`` header
line. ``--redact-timestamps`` (plan §6 T5 step 3) replaces that line
with a fixed sentinel for the test that asserts byte-equality.

Provenance (plan AC-4 ``test_evaluate_harness_records_provenance``):
header carries the prompt hash, model id, gold-set hash, gold-set
size, and implementation name — every dimension the operator might
swap that would change the result is named in the output.

CLI: ``python -m bristol_ml.llm.evaluate [overrides ...]
[--redact-timestamps] [--max-disagreements N]``. Overrides flow
through to :func:`bristol_ml.config.load_config` so
``llm.type=openai`` switches to the live path against the recorded
VCR cassette. Default is the stub (NFR-1 / AC-2).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from bristol_ml.config import load_config
from bristol_ml.llm import ExtractionResult
from bristol_ml.llm.extractor import (
    DEFAULT_GOLD_SET_PATH,
    _GoldSetExpected,
    _GoldSetRecord,
    _load_gold_set,
    build_extractor,
)
from conf._schemas import LlmExtractorConfig

__all__ = [
    "DEFAULT_MAX_DISAGREEMENTS",
    "REDACTED_TIMESTAMP_SENTINEL",
    "TOLERANCE_CAPACITY_MW",
    "TOLERANCE_TIME",
    "EvaluationReport",
    "FieldAgreement",
    "evaluate",
    "format_report",
]


# Plan §1 D12: representative tolerance thresholds. Documented here
# (module docstring) rather than embedded as magic numbers in the
# diff loop; a future revision can change these without grepping
# the codebase.
TOLERANCE_CAPACITY_MW = 5.0
TOLERANCE_TIME = timedelta(hours=1)

# Plan §5 disagreement listing limit — keeps the output legible at the
# demo while preserving the demo punch line. Configurable via CLI flag.
DEFAULT_MAX_DISAGREEMENTS = 10

# Plan NFR-3 / §6 T5 step 3: deterministic sentinel for byte-equality
# tests under ``--redact-timestamps``.
REDACTED_TIMESTAMP_SENTINEL = "REDACTED"


# ---------------------------------------------------------------------
# Per-field agreement bookkeeping
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldAgreement:
    """Counts of exact + tolerance matches for one extraction field.

    The harness reports both metrics side-by-side (plan §1 D11). The
    ``tolerance_match`` count is always ``>=`` the ``exact_match``
    count: every exact match is also a tolerance match. For
    categorical fields (``event_type``, ``fuel_type``) the two counts
    are identical because there is no notion of tolerance — a string
    matches or it does not. The pair is still printed so the
    columnar layout is uniform across all five fields.
    """

    field: str
    total: int
    exact_match: int
    tolerance_match: int
    tolerance_label: str

    def exact_fraction(self) -> str:
        return f"{self.exact_match}/{self.total}"

    def tolerance_fraction(self) -> str:
        return f"{self.tolerance_match}/{self.total}"


@dataclass(frozen=True, slots=True)
class _Disagreement:
    """One mrid-keyed mismatch for the disagreement listing.

    Carries (mrid, field, expected, actual) so the output reads as
    *"M-DC.fuel_type expected=Solar got=Other"* — readable inline at
    the demo without further parsing.
    """

    mrid: str
    field: str
    expected: str
    actual: str


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Structured result of one harness run.

    Carries everything the formatter needs plus the structured fields
    the loguru summary record echoes (plan §1 D15 / OQ-D). The
    ``per_field`` mapping is keyed by the field-name string so the
    output table preserves the AC-4 / D11 ordering.
    """

    implementation: str
    prompt_hash: str | None
    model_id: str | None
    gold_set_path: Path
    gold_set_sha256: str
    gold_set_size: int
    per_field: dict[str, FieldAgreement]
    disagreements: list[_Disagreement]
    generated_at: datetime


def _sha256_prefix_of_file(path: Path, chars: int = 12) -> str:
    """Return the first ``chars`` of the SHA-256 hex digest of ``path``.

    Plan §5 header line: ``gold_set: tests/fixtures/.../hand_labelled.json
    (sha256: a1b2c3..., n=76)``. The hash pins the gold-set bytes; a
    silent edit changes the hash in the output, surfacing in the same
    line where the row count is reported.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:chars]


def _capacity_within_tolerance(expected: float | None, actual: float | None) -> bool:
    """``True`` when both are None or both are within ±:data:`TOLERANCE_CAPACITY_MW`.

    Plan §1 D12: ±5 MW threshold. ``None`` matches ``None`` only — a
    missing value vs a populated value is a mismatch (the demo punch
    line: the LLM hallucinated a number).
    """
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    return math.fabs(expected - actual) <= TOLERANCE_CAPACITY_MW


def _datetime_within_tolerance(expected: datetime | None, actual: datetime | None) -> bool:
    """``True`` when both are None or both are within ±:data:`TOLERANCE_TIME`.

    Plan §1 D12: ±1 h threshold. Used for both ``effective_from`` and
    ``effective_to``. ``None`` ↔ populated mismatch is a mismatch by
    the same logic as :func:`_capacity_within_tolerance`.
    """
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    return abs(expected - actual) <= TOLERANCE_TIME


def _stringify(value: object) -> str:
    """Stringify a field value for the disagreement listing."""
    if value is None:
        return "None"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _accumulate_per_field(
    expected: _GoldSetExpected,
    actual: ExtractionResult,
    mrid: str,
    counts: dict[str, dict[str, int]],
    disagreements: list[_Disagreement],
    *,
    capture_disagreements: bool,
) -> None:
    """Update per-field exact + tolerance counters for one (expected, actual) pair.

    The helper is the only place where the field list lives; the
    formatter reads :data:`_FIELDS` to keep the column order stable.
    Categorical fields treat exact == tolerance because there is no
    notion of "close enough" for strings.
    """
    # event_type — categorical
    exact = expected.event_type == actual.event_type
    counts["event_type"]["exact"] += int(exact)
    counts["event_type"]["tolerance"] += int(exact)
    if not exact and capture_disagreements:
        disagreements.append(
            _Disagreement(
                mrid=mrid,
                field="event_type",
                expected=_stringify(expected.event_type),
                actual=_stringify(actual.event_type),
            )
        )

    # fuel_type — categorical
    exact = expected.fuel_type == actual.fuel_type
    counts["fuel_type"]["exact"] += int(exact)
    counts["fuel_type"]["tolerance"] += int(exact)
    if not exact and capture_disagreements:
        disagreements.append(
            _Disagreement(
                mrid=mrid,
                field="fuel_type",
                expected=_stringify(expected.fuel_type),
                actual=_stringify(actual.fuel_type),
            )
        )

    # affected_capacity_mw — numeric, ±5 MW tolerance
    exact = expected.affected_capacity_mw == actual.affected_capacity_mw
    tolerant = _capacity_within_tolerance(
        expected.affected_capacity_mw, actual.affected_capacity_mw
    )
    counts["affected_capacity_mw"]["exact"] += int(exact)
    counts["affected_capacity_mw"]["tolerance"] += int(tolerant)
    if not tolerant and capture_disagreements:
        disagreements.append(
            _Disagreement(
                mrid=mrid,
                field="affected_capacity_mw",
                expected=_stringify(expected.affected_capacity_mw),
                actual=_stringify(actual.affected_capacity_mw),
            )
        )

    # effective_from — datetime, ±1 h tolerance
    exact = expected.effective_from == actual.effective_from
    tolerant = _datetime_within_tolerance(expected.effective_from, actual.effective_from)
    counts["effective_from"]["exact"] += int(exact)
    counts["effective_from"]["tolerance"] += int(tolerant)
    if not tolerant and capture_disagreements:
        disagreements.append(
            _Disagreement(
                mrid=mrid,
                field="effective_from",
                expected=_stringify(expected.effective_from),
                actual=_stringify(actual.effective_from),
            )
        )

    # effective_to — datetime, ±1 h tolerance
    exact = expected.effective_to == actual.effective_to
    tolerant = _datetime_within_tolerance(expected.effective_to, actual.effective_to)
    counts["effective_to"]["exact"] += int(exact)
    counts["effective_to"]["tolerance"] += int(tolerant)
    if not tolerant and capture_disagreements:
        disagreements.append(
            _Disagreement(
                mrid=mrid,
                field="effective_to",
                expected=_stringify(expected.effective_to),
                actual=_stringify(actual.effective_to),
            )
        )


# Order of fields in the per-field summary table (plan §5 layout).
_FIELDS: tuple[tuple[str, str], ...] = (
    ("event_type", "—"),
    ("fuel_type", "—"),
    ("affected_capacity_mw", f"±{int(TOLERANCE_CAPACITY_MW)} MW"),
    ("effective_from", f"±{int(TOLERANCE_TIME.total_seconds() // 3600)} h"),
    ("effective_to", f"±{int(TOLERANCE_TIME.total_seconds() // 3600)} h"),
)


def evaluate(
    config: LlmExtractorConfig | None,
    *,
    gold_set_path: Path | None = None,
    max_disagreements: int = DEFAULT_MAX_DISAGREEMENTS,
    now: datetime | None = None,
) -> EvaluationReport:
    """Run the active extractor over the gold set and return a report.

    The factory (:func:`bristol_ml.llm.extractor.build_extractor`) is
    the dispatch point: ``BRISTOL_ML_LLM_STUB=1`` overrides
    ``config.type`` per plan §1 D4 (load-bearing for CI safety). For
    the stub path the report records ``prompt_hash=None`` and
    ``model_id=None``; for the OpenAI path both are populated from
    the live extractor.

    The ``now`` argument is the harness's wall-clock at run time —
    captured here rather than read inside :func:`format_report` so a
    test can pin it deterministically. The ``--redact-timestamps``
    CLI flag suppresses the field on the formatted output for byte
    equality.
    """
    path = gold_set_path if gold_set_path is not None else DEFAULT_GOLD_SET_PATH
    extractor = build_extractor(config, gold_set_path=gold_set_path)
    gold_set = _load_gold_set(path)
    sha256_prefix = _sha256_prefix_of_file(path)

    # Sort by mrid for deterministic iteration order — the dict is
    # already insertion-ordered in Python 3.7+, but the gold-set file's
    # record order is the implementer's call; sorting by mrid pins
    # the output regardless of file ordering.
    keys = sorted(gold_set.keys())

    counts: dict[str, dict[str, int]] = {
        field: {"exact": 0, "tolerance": 0} for field, _ in _FIELDS
    }
    disagreements: list[_Disagreement] = []

    for key in keys:
        record: _GoldSetRecord = gold_set[key]
        actual = extractor.extract(record.event)
        _accumulate_per_field(
            expected=record.expected,
            actual=actual,
            mrid=record.event.mrid,
            counts=counts,
            disagreements=disagreements,
            # Stop appending after we have ``max_disagreements + 1`` so
            # we know whether to print "...and N more". A small overhead
            # on a tiny list; not worth a counter.
            capture_disagreements=len(disagreements) <= max_disagreements,
        )

    per_field: dict[str, FieldAgreement] = {}
    n = len(keys)
    for field, tolerance_label in _FIELDS:
        per_field[field] = FieldAgreement(
            field=field,
            total=n,
            exact_match=counts[field]["exact"],
            tolerance_match=counts[field]["tolerance"],
            tolerance_label=tolerance_label,
        )

    # Provenance: stub returns prompt_hash=None / model_id=None on every
    # row by construction; live extractor stamps both. Read from the
    # first extracted result for the live path so the report header
    # matches what's on each row.
    sample_result: ExtractionResult | None = None
    if keys:
        sample_record = gold_set[keys[0]]
        sample_result = extractor.extract(sample_record.event)
    prompt_hash = sample_result.prompt_hash if sample_result else None
    model_id = sample_result.model_id if sample_result else None

    return EvaluationReport(
        implementation=type(extractor).__name__,
        prompt_hash=prompt_hash,
        model_id=model_id,
        gold_set_path=path,
        gold_set_sha256=sha256_prefix,
        gold_set_size=n,
        per_field=per_field,
        disagreements=disagreements,
        generated_at=now if now is not None else datetime.now(UTC),
    )


# ---------------------------------------------------------------------
# Formatting (deterministic stdout — plan NFR-3 / AC-4)
# ---------------------------------------------------------------------


def _format_header(report: EvaluationReport, *, redact_timestamps: bool) -> list[str]:
    """Format the five header lines (provenance + reproducibility line).

    Plan §5: every dimension that would change the result is named.
    The ``generated_at`` line is replaced by :data:`REDACTED_TIMESTAMP_SENTINEL`
    when ``redact_timestamps`` is set, so two harness runs produce
    byte-identical output (NFR-3).
    """
    if redact_timestamps:
        ts = REDACTED_TIMESTAMP_SENTINEL
    else:
        # Strict ISO-8601 UTC with the trailing ``Z``; matches the
        # plan's example exactly.
        ts = report.generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        "=== Stage 14 LLM extractor evaluation ===",
        f"implementation: {report.implementation}",
        f"prompt_hash:    {report.prompt_hash or 'none'}",
        f"model_id:       {report.model_id or 'none'}",
        (
            f"gold_set:       {report.gold_set_path} "
            f"(sha256: {report.gold_set_sha256}, n={report.gold_set_size})"
        ),
        f"generated_at:   {ts}",
    ]


def _format_per_field(report: EvaluationReport) -> list[str]:
    """Format the per-field agreement table.

    The columnar layout matches plan §5: a fixed-width field column,
    then ``exact`` and ``tolerance`` count columns, with the tolerance
    label appended in parentheses for fields where it applies.
    """
    lines = ["", "per-field agreement:"]
    # Fixed-width column for readability; the longest field name is
    # ``affected_capacity_mw`` at 20 chars, padded to 22 for breathing
    # room. Column positions are intentionally hard-coded — the
    # alignment is the demo output's whole point.
    header = f"  {'field':<22}  {'exact':>7}  {'tolerance':>9}"
    lines.append(header)
    for field, _ in _FIELDS:
        agreement = report.per_field[field]
        suffix = (
            f"    (tolerance: {agreement.tolerance_label})"
            if agreement.tolerance_label != "—"
            else ""
        )
        lines.append(
            f"  {field:<22}  {agreement.exact_fraction():>7}  "
            f"{agreement.tolerance_fraction():>9}{suffix}"
        )
    return lines


def _format_disagreements(report: EvaluationReport, *, max_disagreements: int) -> list[str]:
    """Format the disagreement listing — the demo punch line.

    Truncates at ``max_disagreements`` (plan §5 caps it at 10 by
    default) and appends an ellipsis line if more were observed. The
    "(none — ...)" line is intentional reading for the stub-on-its-own-
    gold-set case so the absence of disagreements is explicit.
    """
    lines = ["", f"disagreements (first {max_disagreements}):"]
    if not report.disagreements:
        lines.append("  (none — extractor agrees on every field)")
        return lines
    shown = report.disagreements[:max_disagreements]
    for d in shown:
        lines.append(f"  {d.mrid}.{d.field}: expected={d.expected!r} got={d.actual!r}")
    remaining = len(report.disagreements) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return lines


def format_report(
    report: EvaluationReport,
    *,
    redact_timestamps: bool = False,
    max_disagreements: int = DEFAULT_MAX_DISAGREEMENTS,
) -> str:
    """Format ``report`` as deterministic stdout text per plan §5.

    With ``redact_timestamps=True`` the output is byte-identical
    across two runs of :func:`evaluate` (the only varying dimension
    is the wall-clock header line), satisfying plan NFR-3 /
    ``test_evaluate_harness_output_is_deterministic_modulo_timestamp``.
    """
    parts: list[str] = []
    parts.extend(_format_header(report, redact_timestamps=redact_timestamps))
    parts.extend(_format_per_field(report))
    parts.extend(_format_disagreements(report, max_disagreements=max_disagreements))
    parts.extend(["", "=== end ==="])
    return "\n".join(parts)


def _emit_loguru_summary(report: EvaluationReport) -> None:
    """Emit the report as one structured ``loguru`` INFO record.

    Plan §1 D15 / Ctrl+G OQ-D ("stdout, or captured in logging in the
    non-notebook case"): the harness prints the report to stdout *and*
    leaves a structured trace at INFO so an orchestration / CLI run
    captures it without a markdown destination.

    The ``extra={"summary": {...}}`` payload is the same five header
    fields plus a ``per_field`` dict mirroring the table — enough that
    a downstream log consumer can recover the whole report from one
    record without re-parsing the formatted text.
    """
    summary_payload = {
        "implementation": report.implementation,
        "prompt_hash": report.prompt_hash,
        "model_id": report.model_id,
        "gold_set_path": str(report.gold_set_path),
        "gold_set_sha256": report.gold_set_sha256,
        "gold_set_size": report.gold_set_size,
        "per_field": {
            name: {
                "total": ag.total,
                "exact_match": ag.exact_match,
                "tolerance_match": ag.tolerance_match,
                "tolerance_label": ag.tolerance_label,
            }
            for name, ag in report.per_field.items()
        },
        "disagreement_count": len(report.disagreements),
    }
    logger.bind(summary=summary_payload).info(
        "Stage 14 evaluation: implementation={} gold_set_size={} disagreements={}",
        report.implementation,
        report.gold_set_size,
        len(report.disagreements),
    )


# ---------------------------------------------------------------------
# Standalone CLI (NFR-8 / DESIGN §2.1.1)
# ---------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the ``python -m bristol_ml.llm.evaluate`` parser."""
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.llm.evaluate",
        description=(
            "Run the active LLM extractor over the hand-labelled gold "
            "set and print a side-by-side accuracy report. Hydra-style "
            "overrides are accepted; by default the offline stub path "
            "is selected. Composes the +llm=extractor config group "
            "automatically."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides applied on top of conf/config.yaml + "
            "+llm=extractor (e.g. llm.type=openai)."
        ),
    )
    parser.add_argument(
        "--redact-timestamps",
        action="store_true",
        help=(
            "Replace the 'generated_at' header with a fixed sentinel — "
            "used by the reproducibility test (NFR-3) so two runs "
            "produce byte-identical output."
        ),
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=DEFAULT_MAX_DISAGREEMENTS,
        help=(
            f"Maximum number of disagreements to list "
            f"(default {DEFAULT_MAX_DISAGREEMENTS}); plan §5 caps the "
            "demo output at 10 to keep it legible."
        ),
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry — DESIGN §2.1.1, plan AC-4 / NFR-8.

    Resolves the active :class:`LlmExtractorConfig` via Hydra, runs
    :func:`evaluate`, prints :func:`format_report`, and emits the
    loguru summary record. Returns 0 on success, 2 on a config or
    gold-set error.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = ["+llm=extractor", *args.overrides]
    try:
        cfg = load_config(overrides=overrides)
    except (ValidationError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    try:
        report = evaluate(
            cfg.llm,
            max_disagreements=args.max_disagreements,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2

    print(
        format_report(
            report,
            redact_timestamps=args.redact_timestamps,
            max_disagreements=args.max_disagreements,
        )
    )
    _emit_loguru_summary(report)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(_cli_main())
