"""Spec-derived tests for the Stage 14 T2 typed boundary.

Every test here is derived from:

- ``docs/plans/completed/14-llm-extractor.md`` §6 T2 named tests:
  ``test_extractor_protocol_has_two_methods``,
  ``test_extraction_result_pydantic_validation_on_bad_types``,
  ``test_extraction_result_datetimes_are_utc_aware``,
  ``test_extraction_result_confidence_in_unit_interval``,
  ``test_schema_importable_without_concrete_implementations``.
- Plan §5 schema sketch (the ``Extractor``, ``ExtractionResult``,
  ``RemitEvent`` shapes).
- Plan D2 (two-method Protocol), D14 (ExtractionResult fields,
  ``confidence`` in [0.0, 1.0], UTC-aware datetimes), AC-1
  ("interface small enough that writing a third implementation in
  the future is plausible") and AC-5 ("schema typed and validated at
  the interface boundary").

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test;
surface the failure to the implementer.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from typing import Protocol, get_type_hints

import pytest
from pydantic import ValidationError

from bristol_ml.llm import ExtractionResult, Extractor, RemitEvent

# ---------------------------------------------------------------------
# AC-1 — small interface
# ---------------------------------------------------------------------


def test_extractor_protocol_has_two_methods() -> None:
    """Guards plan AC-1 / D2: the Protocol has exactly two methods.

    AC-1 (intent line 32): *"the interface is small enough that
    writing a third implementation in the future is plausible"*.
    Plan D2 binds this to two methods: ``extract`` (single event) and
    ``extract_batch`` (batch).  Any growth on the public surface
    erodes AC-1; the test catches it at CI time.

    The check is structural — we read the Protocol body's own
    callable members rather than ``dir(Extractor)`` (which
    includes ``__init_subclass__`` etc.) — so adding a new method
    fails loudly while adding a docstring or a type alias does not.

    Cited criterion: plan §6 T2 named test
    ``test_extractor_protocol_has_two_methods`` (also AC-1).
    """
    # ``Extractor`` is itself a Protocol — confirm we kept the
    # structural-typing path open for third implementations.
    assert issubclass(Extractor, Protocol), (
        "Extractor must be a typing.Protocol so a third implementation "
        "can satisfy it without inheriting (AC-1)."
    )

    # The Protocol's own annotated callables — the methods declared
    # in the class body.  Anything inherited from the Protocol
    # machinery (``__class_getitem__``, ``_is_protocol``, etc.) is
    # filtered out by the underscore prefix or by being absent from
    # ``__annotations__``.
    method_names = {
        name
        for name, value in vars(Extractor).items()
        if callable(value) and not name.startswith("_")
    }
    assert method_names == {"extract", "extract_batch"}, (
        "Extractor Protocol must expose exactly extract + extract_batch "
        f"(plan D2 / AC-1); got {sorted(method_names)!r}."
    )


# ---------------------------------------------------------------------
# AC-5 — typed boundary
# ---------------------------------------------------------------------


def _valid_extraction_kwargs() -> dict[str, object]:
    """Minimal field set that constructs a valid ExtractionResult.

    Used by the negative tests below — each test perturbs one field
    away from this baseline and asserts the perturbation raises.
    """
    return {
        "event_type": "Outage",
        "fuel_type": "Gas",
        "affected_capacity_mw": 200.0,
        "effective_from": datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        "effective_to": datetime(2026, 4, 2, 12, 0, tzinfo=UTC),
        "confidence": 0.5,
        "prompt_hash": "abc123def456",
        "model_id": "gpt-4o-mini",
    }


def test_extraction_result_pydantic_validation_on_bad_types() -> None:
    """Guards plan AC-5 / D14: invalid types at the boundary raise ValidationError.

    AC-5 (intent line 37): *"the extracted feature schema is typed
    and validated at the interface boundary"*.  Constructing
    ``ExtractionResult(affected_capacity_mw="not a number")`` must
    fail with a Pydantic ``ValidationError`` — not silently coerce, not
    pass through.  The ``extra="forbid"`` ConfigDict and the field
    types together carry this guarantee.

    Cited criterion: plan §6 T2 named test
    ``test_extraction_result_pydantic_validation_on_bad_types``
    (also AC-5).
    """
    bad_kwargs = _valid_extraction_kwargs() | {"affected_capacity_mw": "not a number"}

    with pytest.raises(ValidationError) as excinfo:
        ExtractionResult(**bad_kwargs)  # type: ignore[arg-type]  # testing the validator

    assert "affected_capacity_mw" in str(excinfo.value), (
        "ValidationError must name the offending field "
        f"``affected_capacity_mw``; got: {excinfo.value!s}"
    )


def test_extraction_result_datetimes_are_utc_aware() -> None:
    """Guards plan D14: naive datetimes on ``effective_from`` / ``effective_to`` raise.

    Plan D14: *"All datetime fields are timezone-aware UTC consistent
    with Stage 13"*.  Stage 13's ``OUTPUT_SCHEMA`` pins every
    timestamp to ``timestamp[us, tz=UTC]``; passing a naive datetime
    through the Stage 14 boundary would silently break that contract,
    so the validator catches it at construction time.

    Cited criterion: plan §6 T2 named test
    ``test_extraction_result_datetimes_are_utc_aware`` (also AC-5).
    """
    naive_dt = datetime(2026, 4, 1, 12, 0)  # no tzinfo
    assert naive_dt.tzinfo is None  # sanity — the test premise

    bad_kwargs_from = _valid_extraction_kwargs() | {"effective_from": naive_dt}
    with pytest.raises(ValidationError) as excinfo:
        ExtractionResult(**bad_kwargs_from)  # type: ignore[arg-type]
    assert "effective_from" in str(excinfo.value), (
        f"Naive ``effective_from`` must be flagged by name; got: {excinfo.value!s}"
    )

    bad_kwargs_to = _valid_extraction_kwargs() | {"effective_to": naive_dt}
    with pytest.raises(ValidationError) as excinfo:
        ExtractionResult(**bad_kwargs_to)  # type: ignore[arg-type]
    assert "effective_to" in str(excinfo.value), (
        f"Naive ``effective_to`` must be flagged by name; got: {excinfo.value!s}"
    )

    # The same discipline applies to RemitEvent so callers cannot
    # smuggle naive timestamps in via the input boundary either.
    naive_remit_kwargs = {
        "mrid": "MRID-1",
        "revision_number": 0,
        "message_status": "Active",
        "published_at": naive_dt,
        "effective_from": datetime(2026, 4, 1, tzinfo=UTC),
        "effective_to": None,
    }
    with pytest.raises(ValidationError) as excinfo:
        RemitEvent(**naive_remit_kwargs)  # type: ignore[arg-type]
    assert "published_at" in str(excinfo.value), (
        f"Naive RemitEvent ``published_at`` must be flagged by name; got: {excinfo.value!s}"
    )


def test_extraction_result_confidence_in_unit_interval() -> None:
    """Guards plan D14: ``confidence`` outside [0.0, 1.0] raises ValidationError.

    Plan D14: ``confidence`` is constrained to the closed unit
    interval — ``Field(ge=0.0, le=1.0)``.  The sentinel values 0.0
    (stub default fallback) and 1.0 (stub hand-labelled hit, live
    OpenAI) are both included.  Anything outside the range is a
    schema bug; the validator catches it.

    Cited criterion: plan §6 T2 named test
    ``test_extraction_result_confidence_in_unit_interval`` (also
    AC-5).
    """
    # Boundary values must be accepted.
    for ok in (0.0, 0.5, 1.0):
        ExtractionResult(**(_valid_extraction_kwargs() | {"confidence": ok}))

    # Out-of-range values must be rejected.
    for bad in (-0.0001, 1.0001, -1.0, 2.0):
        with pytest.raises(ValidationError) as excinfo:
            ExtractionResult(**(_valid_extraction_kwargs() | {"confidence": bad}))
        assert "confidence" in str(excinfo.value), (
            f"Out-of-range confidence={bad} must be flagged by field name; got: {excinfo.value!s}"
        )


def test_extraction_result_extra_keys_forbidden() -> None:
    """Guards plan §5 ConfigDict: ``extra="forbid"`` traps caller typos.

    ``model_config = ConfigDict(extra="forbid", frozen=True)`` is
    load-bearing — a typo in a downstream ``ExtractionResult(...)``
    construction must raise rather than silently drop the field.
    The plan §5 schema sketch declares it; this test enforces it
    survives refactors.
    """
    bad_kwargs = _valid_extraction_kwargs() | {"event_typo": "Outage"}
    with pytest.raises(ValidationError) as excinfo:
        ExtractionResult(**bad_kwargs)  # type: ignore[arg-type]
    assert "event_typo" in str(excinfo.value) or "extra" in str(excinfo.value).lower(), (
        f"extra='forbid' must surface unknown fields; got: {excinfo.value!s}"
    )


def test_extraction_result_is_frozen() -> None:
    """Guards plan §5 ConfigDict: ``frozen=True`` makes results immutable.

    Frozen Pydantic models hash by value and reject attribute
    assignment; AC-5's *"validated at the interface boundary"*
    promise is undermined if a caller can mutate a result after
    construction.  The plan §5 schema sketch declares it; this test
    enforces it survives refactors.
    """
    result = ExtractionResult(**_valid_extraction_kwargs())
    with pytest.raises(ValidationError):
        result.event_type = "Restriction"  # type: ignore[misc]


# ---------------------------------------------------------------------
# AC-1 — boundary is importable without the concrete implementations
# ---------------------------------------------------------------------


def test_schema_importable_without_concrete_implementations() -> None:
    """Guards plan AC-1: ``bristol_ml.llm`` is importable without ``extractor``.

    AC-1 sub-criterion (plan §4 / §5): *"the schema is importable
    from ``bristol_ml.llm`` without importing either concrete
    implementation"*.  This is the load-bearing mechanism that lets
    Stage 15 (embedding index) and Stage 16 (feature-table join)
    depend on the boundary types without dragging in either backend
    or its transitive dependencies (e.g. the ``openai`` SDK).

    The test re-imports the package after evicting any cached
    ``bristol_ml.llm.extractor`` module from ``sys.modules``, then
    asserts that the boundary types resolve and that the concrete
    module is NOT pulled in by the import.

    Cited criterion: plan §6 T2 named test
    ``test_schema_importable_without_concrete_implementations``
    (also AC-1).
    """
    # Evict the concrete-implementation module + its parent so the
    # re-import below is genuinely cold for that path.
    for name in list(sys.modules):
        if name == "bristol_ml.llm" or name.startswith("bristol_ml.llm."):
            sys.modules.pop(name, None)

    pkg = importlib.import_module("bristol_ml.llm")

    # The three boundary symbols must resolve from the package
    # __init__ alone.
    assert hasattr(pkg, "Extractor")
    assert hasattr(pkg, "ExtractionResult")
    assert hasattr(pkg, "RemitEvent")

    # And — critically — the package import alone must not have
    # caused the concrete-implementation module to load.  If it has,
    # the boundary/implementation split has been broken.
    assert "bristol_ml.llm.extractor" not in sys.modules, (
        "Importing ``bristol_ml.llm`` pulled in "
        "``bristol_ml.llm.extractor`` — the boundary/implementation "
        "split required by AC-1 has been broken; Stage 15 and Stage "
        "16 would inherit the OpenAI SDK as a transitive dep."
    )


# ---------------------------------------------------------------------
# Field hygiene — the schema is the contract Stage 15/16 will consume
# ---------------------------------------------------------------------


def test_extraction_result_field_set_matches_plan_schema() -> None:
    """Guards plan §5 + D14: the field set is exactly the plan-declared shape.

    Stage 15 (embedding index) and Stage 16 (feature-table join)
    will be written against this exact field set; adding or removing
    a field is a downstream-affecting change that must be a plan
    edit, not a quiet code edit.  The plan §5 schema sketch enumerates
    the eight fields; this test holds them in place.
    """
    expected = {
        "event_type",
        "fuel_type",
        "affected_capacity_mw",
        "effective_from",
        "effective_to",
        "confidence",
        "prompt_hash",
        "model_id",
    }
    actual = set(ExtractionResult.model_fields.keys())
    assert actual == expected, (
        "ExtractionResult field set has drifted from plan §5; "
        f"expected {sorted(expected)!r}, got {sorted(actual)!r}."
    )


def test_remit_event_field_set_matches_plan_schema() -> None:
    """Guards plan §5: the RemitEvent field set is the extraction-relevant subset.

    RemitEvent is the typed mirror of Stage 13's ``OUTPUT_SCHEMA``
    extraction-relevant subset (plan §5).  Drift here would silently
    break the StubExtractor (T3) and LlmExtractor (T4) inputs.
    """
    expected = {
        "mrid",
        "revision_number",
        "message_status",
        "published_at",
        "effective_from",
        "effective_to",
        "fuel_type",
        "affected_mw",
        "event_type",
        "cause",
        "message_description",
    }
    actual = set(RemitEvent.model_fields.keys())
    assert actual == expected, (
        "RemitEvent field set has drifted from plan §5; "
        f"expected {sorted(expected)!r}, got {sorted(actual)!r}."
    )


def test_extractor_protocol_method_signatures() -> None:
    """Guards plan §5: the Protocol method signatures match the published shape.

    Stage 15 / Stage 16 call sites are written against
    ``extract(event: RemitEvent) -> ExtractionResult`` and
    ``extract_batch(events: list[RemitEvent]) -> list[ExtractionResult]``.
    A type-hint regression on either method is a contract break;
    this test pins the signatures structurally.
    """
    extract_hints = get_type_hints(Extractor.extract)
    assert extract_hints.get("event") is RemitEvent, (
        f"extract(event: RemitEvent); got {extract_hints!r}."
    )
    assert extract_hints.get("return") is ExtractionResult, (
        f"extract -> ExtractionResult; got {extract_hints!r}."
    )

    batch_hints = get_type_hints(Extractor.extract_batch)
    assert batch_hints.get("events") == list[RemitEvent], (
        f"extract_batch(events: list[RemitEvent]); got {batch_hints!r}."
    )
    assert batch_hints.get("return") == list[ExtractionResult], (
        f"extract_batch -> list[ExtractionResult]; got {batch_hints!r}."
    )
