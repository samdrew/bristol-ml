"""Stage 14 — LLM feature extractor: typed boundary.

This package exposes the **public boundary** of the Stage 14 extractor:
the :class:`Extractor` :class:`~typing.Protocol`, the
:class:`ExtractionResult` Pydantic model that crosses every call, and
the :class:`RemitEvent` Pydantic model that callers wrap a Stage 13
parquet row in.  The two concrete implementations
(:class:`~bristol_ml.llm.extractor.StubExtractor` and
:class:`~bristol_ml.llm.extractor.LlmExtractor`) live in
``bristol_ml.llm.extractor`` so that callers — Stage 15 (embedding
index) and Stage 16 (feature-table join) — can depend on this module
without importing either backend.

Plan AC-1 sub-criterion: *"the schema is importable from
``bristol_ml.llm`` without importing either concrete implementation."*
The split between this ``__init__`` (boundary types) and ``extractor``
(implementations + factory) is the load-bearing mechanism.

Plan AC-5: every value crossing the public interface is a Pydantic
model instance, not a raw ``dict``.  The two models below carry the
enforcement; the Protocol is structurally typed so a third
implementation slots in without changes here (plan D3).

Cross-references:

- Layer contract — ``docs/architecture/layers/llm.md`` (Stage 14 T7).
- Stage 14 plan — ``docs/plans/completed/14-llm-extractor.md``.
- Intent — ``docs/intent/14-llm-extractor.md``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ExtractionResult",
    "Extractor",
    "RemitEvent",
]


class RemitEvent(BaseModel):
    """One row of Stage 13's REMIT parquet, lifted into a typed boundary.

    Stage 13's ``OUTPUT_SCHEMA`` is a pyarrow schema; downstream consumers
    (Stage 14, Stage 15, Stage 16) need the same data as a typed Python
    object so the LLM extractor's interface is not implicitly
    ``dict``-shaped.  ``RemitEvent`` is the typed mirror of the
    extraction-relevant subset.

    The free-text ``message_description`` field is **frequently NULL**
    on the live Elexon stream endpoint — the Stage 13 implementation
    (``remit.py:431``) keeps the column nullable for exactly this case
    and Stage 14 plan §1 D6 (Ctrl+G OQ-B) elects to *accept the NULL
    and synthesise a prompt input from the structured fields*, rather
    than issue follow-up ``GET /remit/{messageId}`` calls.  Callers do
    not need to filter NULL descriptions; the extractor handles them.

    Frozen + extra="forbid" keeps the boundary inspectable: a typo in
    a caller's row dict raises a :class:`pydantic.ValidationError` at
    construction rather than silently dropping the field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mrid: str
    revision_number: int = Field(ge=0)
    message_status: str
    published_at: datetime
    effective_from: datetime
    effective_to: datetime | None = None
    fuel_type: str | None = None
    affected_mw: float | None = None
    event_type: str | None = None
    cause: str | None = None
    message_description: str | None = None

    @field_validator("published_at", "effective_from", "effective_to")
    @classmethod
    def _require_utc_aware(cls, value: datetime | None) -> datetime | None:
        """Reject naive datetimes; mirror the Stage 13 timezone discipline.

        Stage 13's ``OUTPUT_SCHEMA`` pins every timestamp to
        ``timestamp[us, tz=UTC]``; passing a naive datetime through the
        Stage 14 boundary would silently break that contract.  The
        validator catches it at construction time.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "RemitEvent timestamps must be timezone-aware UTC "
                "(consistent with Stage 13 OUTPUT_SCHEMA); "
                f"got naive datetime {value!r}."
            )
        return value


class ExtractionResult(BaseModel):
    """Structured features extracted from a single :class:`RemitEvent`.

    AC-5 (intent line 37): every value crossing the extractor interface
    boundary is an instance of this model, never a raw ``dict``.  The
    field set is the intent §Scope line 14 enumeration plus a
    provenance pair (``prompt_hash`` / ``model_id``) so each result is
    traceable back to the prompt + model that produced it (plan §1 D10
    + NFR-5).

    ``confidence`` is a documented sentinel:
    1.0 for a stub hand-labelled hit, 0.0 for the stub default
    fallback, and a fixed 1.0 for the live OpenAI path (plan §1 D14:
    OpenAI strict mode does not return per-token logprobs by default
    and the schema does not require calibrated probabilities).  Stage
    16 callers must treat this as a pedagogical sentinel, not a
    calibrated probability — see the module ``CLAUDE.md`` for the
    downstream-consumer warning.

    The provenance fields are ``None`` for the stub path (no prompt,
    no model) and populated for the live path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str
    fuel_type: str
    affected_capacity_mw: float | None = None
    effective_from: datetime
    effective_to: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    prompt_hash: str | None = None
    model_id: str | None = None

    @field_validator("effective_from", "effective_to")
    @classmethod
    def _require_utc_aware(cls, value: datetime | None) -> datetime | None:
        """Reject naive datetimes (mirror :class:`RemitEvent` discipline)."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "ExtractionResult timestamps must be timezone-aware UTC; "
                f"got naive datetime {value!r}."
            )
        return value


@runtime_checkable
class Extractor(Protocol):
    """The Stage 14 extraction contract — small, implementation-agnostic.

    AC-1 (intent line 32): *"The interface is small enough that
    writing a third implementation in the future is plausible."*  The
    Protocol carries exactly two methods — single-event and batch —
    and refers to neither the LLM, the prompt, nor the stub data
    store.  Stage 15 (embedding index) and Stage 16 (feature-table
    join) depend only on this Protocol, not on either concrete class.

    Implementations:

    - :class:`bristol_ml.llm.extractor.StubExtractor` — the default
      offline path.  Reads ``tests/fixtures/llm/hand_labelled.json``;
      returns a documented sentinel (``confidence=0.0``) for unknown
      events.
    - :class:`bristol_ml.llm.extractor.LlmExtractor` — the live
      OpenAI path.  Activated by ``llm.type=openai`` *and* a populated
      ``BRISTOL_ML_LLM_API_KEY`` *and* ``BRISTOL_ML_LLM_STUB`` not set
      to ``"1"`` (plan §1 D4 — triple-gated for safety).

    A future implementation slots in by adding a new ``type`` literal
    to :class:`~conf._schemas.LlmExtractorConfig` and a new
    ``Extractor``-shaped class to the factory in
    ``bristol_ml.llm.extractor``; nothing in this module changes.

    The Protocol is :func:`~typing.runtime_checkable` so unit tests
    can assert structural conformance with ``isinstance(_, Extractor)``;
    ADR-0003 sets the precedent for runtime-checkable Protocol over
    ``abc.ABC`` for swappable interfaces.
    """

    def extract(self, event: RemitEvent) -> ExtractionResult:
        """Extract structured features from a single REMIT event."""
        ...

    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
        """Extract features from a batch — returned in input order."""
        ...
