"""Stage 14 — concrete extractor implementations + factory.

Two implementations selected by :class:`conf._schemas.LlmExtractorConfig`'s
``type`` discriminator (plan §1 D3) plus the ``BRISTOL_ML_LLM_STUB`` env
var (plan §1 D4):

- :class:`StubExtractor` — the offline-by-default path (AC-2). Reads a
  hand-labelled JSON gold set at construction; ``extract`` returns the
  labelled record when ``(mrid, revision_number)`` matches and the
  documented default (plan §1 D16) otherwise. Makes zero network
  calls (NFR-1).
- :class:`LlmExtractor` — the live OpenAI Chat Completions path (T4).
  Triple-gated: ``type == "openai"`` *and* ``BRISTOL_ML_LLM_STUB`` not
  set to ``"1"`` *and* the configured API-key env var populated. The
  T3 placeholder lives here; the OpenAI client wiring lands at T4.

Both implementations satisfy the :class:`bristol_ml.llm.Extractor`
``Protocol`` structurally — :func:`runtime_checkable` lets the unit
tests assert this without inheritance (ADR-0003 precedent).

The :func:`build_extractor` factory is the single dispatch point so
callers (Stage 15 embedding index, Stage 16 feature-table join, the
evaluation harness) need only pass an :class:`LlmExtractorConfig`.
The factory honours ``BRISTOL_ML_LLM_STUB=1`` regardless of config
``type`` — load-bearing for CI safety (plan §1 D4).

Cross-references:

- Layer contract — ``docs/architecture/layers/llm.md`` (lands at T7).
- Stage 14 plan — ``docs/plans/active/14-llm-extractor.md`` §5, §6.
- Boundary types — :mod:`bristol_ml.llm` (`__init__.py`).

Standalone CLI (``python -m bristol_ml.llm.extractor``) prints the
active implementation, the loaded gold-set size, and one sample
extraction; exits 0 (NFR-8).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from bristol_ml.config import load_config
from bristol_ml.llm import ExtractionResult, Extractor, RemitEvent
from bristol_ml.llm._prompts import load_prompt
from conf._schemas import LlmExtractorConfig

__all__ = [
    "DEFAULT_GOLD_SET_PATH",
    "STUB_ENV_VAR",
    "LlmExtractor",
    "StubExtractor",
    "build_extractor",
]


# Plan §1 D4: ``BRISTOL_ML_LLM_STUB=1`` activates the stub regardless
# of config ``type``. Mirrors the ingestion-layer ``BRISTOL_ML_REMIT_STUB``
# precedent so the project's offline-by-default discipline is uniform
# across modules.
STUB_ENV_VAR = "BRISTOL_ML_LLM_STUB"

# Plan §1 D9: the default gold-set path. Versioned in-repo with the
# code; the ``schema_version`` field at the top of the file guards
# additions. Callers can override via ``StubExtractor(gold_set_path=...)``
# (e.g. for unit tests that point at a smaller fixture).
DEFAULT_GOLD_SET_PATH = Path("tests/fixtures/llm/hand_labelled.json")


# ---------------------------------------------------------------------
# Gold-set fixture schema
# ---------------------------------------------------------------------


class _GoldSetExpected(BaseModel):
    """The expected-extraction half of a gold-set record.

    Mirrors :class:`ExtractionResult` but without ``prompt_hash`` /
    ``model_id`` — those are provenance the gold set doesn't dictate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str
    fuel_type: str
    affected_capacity_mw: float | None = None
    effective_from: datetime
    effective_to: datetime | None = None
    confidence: float


class _GoldSetRecord(BaseModel):
    """One ``(event, expected)`` pair from the hand-labelled gold set.

    Internal — the only callers are the stub loader and the (T5)
    evaluation harness. The ``expected`` payload is shaped like an
    :class:`ExtractionResult` minus the provenance pair (which the
    stub fills with ``None``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: RemitEvent
    expected: _GoldSetExpected


def _load_gold_set(path: Path) -> dict[tuple[str, int], _GoldSetRecord]:
    """Read the gold-set JSON and index by ``(mrid, revision_number)``.

    The lookup key matches the storage grain of Stage 13's REMIT
    parquet — ``(mrid, revision_number)`` is the primary key in
    :data:`bristol_ml.ingestion.remit.OUTPUT_SCHEMA` — so every
    distinct revision can carry its own labelled extraction.

    Raises :class:`FileNotFoundError` if ``path`` is missing and
    :class:`pydantic.ValidationError` if any record fails the
    :class:`_GoldSetRecord` shape.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 14 gold-set fixture not found at {path}. "
            "The default location is "
            f"{DEFAULT_GOLD_SET_PATH}; pass an alternative path to "
            "StubExtractor(gold_set_path=...) for tests."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(
            f"Gold-set fixture {path} must be an object with a "
            "'records' list; see tests/fixtures/llm/hand_labelled.json "
            "for the canonical shape."
        )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Gold-set fixture {path} has schema_version="
            f"{schema_version!r}; this StubExtractor only understands "
            "schema_version=1. Update bristol_ml.llm.extractor when the "
            "fixture format evolves."
        )
    indexed: dict[tuple[str, int], _GoldSetRecord] = {}
    for raw in payload["records"]:
        record = _GoldSetRecord.model_validate(raw)
        key = (record.event.mrid, record.event.revision_number)
        indexed[key] = record
    return indexed


# ---------------------------------------------------------------------
# Stub implementation — offline by default (AC-2)
# ---------------------------------------------------------------------


class StubExtractor:
    """Offline extractor backed by a hand-labelled gold set.

    AC-2 (intent line 34): *"the stub is the default; running anything
    that consumes the extractor works offline with no API key"*. The
    stub loads a JSON fixture at construction, indexes it by
    ``(mrid, revision_number)``, and returns the labelled extraction
    on a hit; on a miss it returns the documented default (plan §1
    D16) — ``confidence=0.0``, fields synthesised from the input's
    structured columns where possible.

    Plan §1 D14: the stub uses ``confidence=1.0`` for hand-labelled
    hits and ``confidence=0.0`` for misses. Both are documented
    sentinels — Stage 16 callers must not treat them as calibrated
    probabilities.

    The class satisfies :class:`bristol_ml.llm.Extractor` structurally
    via :func:`runtime_checkable`; no inheritance needed.

    Construction is cheap (one JSON read + Pydantic validation) so
    re-instantiating per call site is fine. The stub is thread-safe
    after construction — the lookup map is built once and read-only
    thereafter.
    """

    def __init__(
        self,
        gold_set_path: Path | None = None,
    ) -> None:
        path = gold_set_path if gold_set_path is not None else DEFAULT_GOLD_SET_PATH
        self._gold_set_path = path
        self._lookup = _load_gold_set(path)
        logger.debug(
            "StubExtractor loaded {} record(s) from {}",
            len(self._lookup),
            path,
        )

    @property
    def gold_set_path(self) -> Path:
        """The path the gold set was loaded from (read-only)."""
        return self._gold_set_path

    @property
    def gold_set_size(self) -> int:
        """The number of indexed gold-set records (read-only)."""
        return len(self._lookup)

    def extract(self, event: RemitEvent) -> ExtractionResult:
        """Return the labelled extraction for ``event`` or the documented default.

        On a ``(mrid, revision_number)`` hit, returns an
        :class:`ExtractionResult` populated from the gold-set
        ``expected`` payload with ``prompt_hash=None``, ``model_id=None``
        (the stub has neither). On a miss, returns the plan-§1-D16
        default: structural-field-derived event_type / fuel_type when
        non-NULL, capacity ``None``, times mirrored from input,
        ``confidence=0.0``.
        """
        key = (event.mrid, event.revision_number)
        record = self._lookup.get(key)
        if record is not None:
            expected = record.expected
            return ExtractionResult(
                event_type=expected.event_type,
                fuel_type=expected.fuel_type,
                affected_capacity_mw=expected.affected_capacity_mw,
                effective_from=expected.effective_from,
                effective_to=expected.effective_to,
                confidence=expected.confidence,
                prompt_hash=None,
                model_id=None,
            )
        # Plan §1 D16 documented default for unknown events.
        return ExtractionResult(
            event_type=event.event_type if event.event_type is not None else "Other",
            fuel_type=event.fuel_type if event.fuel_type is not None else "Other",
            affected_capacity_mw=None,
            effective_from=event.effective_from,
            effective_to=event.effective_to,
            confidence=0.0,
            prompt_hash=None,
            model_id=None,
        )

    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
        """Extract a batch — identical to mapping :meth:`extract` over ``events``.

        The Stage 14 stub has no batch optimisation (no network call
        to amortise); the live :class:`LlmExtractor` will. Order
        is preserved per plan D2.
        """
        return [self.extract(event) for event in events]


# ---------------------------------------------------------------------
# Live implementation — placeholder; T4 wires the OpenAI client
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Strict-mode JSON schema (plan §1 D6 — OpenAI ``json_schema`` response_format)
# ---------------------------------------------------------------------


# OpenAI strict mode requires every object to declare
# ``additionalProperties: false`` and to list every property in
# ``required`` (no implicit-optional via ``default``). Pydantic's
# ``model_json_schema()`` emits ``additionalProperties: false`` for
# ``extra="forbid"`` models but it also emits ``default: null`` on
# ``Optional[...]`` fields and omits them from ``required`` — both
# would be rejected by the OpenAI API at first call.
#
# We therefore hand-author the schema for the LLM-populated subset.
# The provenance fields (``prompt_hash`` + ``model_id``) are stamped
# by the extractor *after* the LLM returns and so are not part of
# what the model populates.
#
# Reference: https://platform.openai.com/docs/guides/structured-outputs
# (Strict mode, "additionalProperties:false on every object" + "all
# properties in required").
_OPENAI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_type",
        "fuel_type",
        "affected_capacity_mw",
        "effective_from",
        "effective_to",
        "confidence",
    ],
    "properties": {
        "event_type": {
            "type": "string",
            "description": (
                "One of 'Outage', 'Restriction', 'Withdrawn', 'Other'. "
                "Prefer the input's structured event_type when present."
            ),
        },
        "fuel_type": {
            "type": "string",
            "description": (
                "Canonical Elexon fuel-type vocabulary "
                "('Coal', 'Gas', 'Nuclear', 'Oil', 'Wind', 'Solar', "
                "'Hydro', 'Pumped Storage', 'Biomass', 'Other', "
                "'Interconnector', 'Battery'). Match the input's "
                "fuel_type when present; otherwise infer from "
                "the unit name or description."
            ),
        },
        "affected_capacity_mw": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": (
                "Unavailable capacity in megawatts; null if no number "
                "is recoverable from the input."
            ),
        },
        "effective_from": {
            "type": "string",
            "description": "UTC ISO-8601 timestamp marking the event start.",
        },
        "effective_to": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "UTC ISO-8601 timestamp marking the event end; null for open-ended events."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "1.0 when every output field is grounded in an explicit "
                "input field; 0.0 when no field could be recovered."
            ),
        },
    },
}


class LlmExtractor:
    """Live OpenAI Chat Completions extractor (plan §1 D6).

    Calls ``openai.chat.completions.create`` with a strict
    ``response_format`` of type ``"json_schema"`` so the model is
    constrained at decode time to emit valid JSON conforming to the
    schema (CFG token masking; OpenAI strict mode, GA since
    August 2024).

    Plan §1 D5: ``__init__`` reads the API key from the configured env
    var. If absent and ``BRISTOL_ML_LLM_STUB`` is not set, raise
    :class:`RuntimeError` at init time with a message naming both
    env-vars (so the operator sees the offline escape hatch).

    Plan §1 D14 + D16: a parse / validation failure logs WARNING with
    the event id and returns an :class:`ExtractionResult` populated
    with the documented default (``confidence=0.0``, structural fields
    mirrored from the input, capacity ``None``, provenance recorded).
    ``extract`` never raises an unhandled exception (NFR-6).

    The triple-gating is enforced by :func:`build_extractor`, not by
    the class itself — a caller can construct ``LlmExtractor`` directly
    in a test, but the factory is the production path and short-circuits
    to the stub when the env var or API key is missing.
    """

    # Class-level reference to the strict schema so tests can assert
    # its shape without importing the module-private constant.
    RESPONSE_SCHEMA = _OPENAI_RESPONSE_SCHEMA

    def __init__(self, config: LlmExtractorConfig) -> None:
        if config.type != "openai":
            raise ValueError(
                f"LlmExtractor requires config.type='openai'; got {config.type!r}. "
                "Use build_extractor() for the type-dispatched factory."
            )
        if config.model_name is None:
            raise ValueError(
                "LlmExtractorConfig.model_name must be set when type='openai' "
                "(e.g. 'gpt-4o-mini'); got None."
            )
        if config.prompt_file is None:
            raise ValueError(
                "LlmExtractorConfig.prompt_file must be set when type='openai' "
                "(e.g. conf/llm/prompts/extract_v1.txt); got None."
            )
        api_key = os.environ.get(config.api_key_env_var, "").strip()
        if not api_key:
            raise RuntimeError(
                f"LLM API key not found in environment variable "
                f"{config.api_key_env_var!r}. Either populate that variable "
                f"or set {STUB_ENV_VAR}=1 to force the offline stub path "
                "(plan §1 D4 — triple-gated for CI safety)."
            )
        # Plan §1 D7 / NFR-5: load prompt + compute hash at init
        # so every call shares the same prompt-bytes-derived identity
        # and a deploy-time stale prompt fails loudly here rather than
        # at the first call.
        prompt_text, prompt_hash = load_prompt(config.prompt_file)
        # Defer the openai import so a stub-only test environment
        # (or a CI run without the SDK) still imports this module.
        from openai import OpenAI

        self._config = config
        self._prompt_text = prompt_text
        self._prompt_hash = prompt_hash
        self._client = OpenAI(api_key=api_key, timeout=config.request_timeout_seconds)
        logger.info(
            "LlmExtractor initialised (model={}, prompt_hash={})",
            config.model_name,
            prompt_hash,
        )

    @property
    def prompt_hash(self) -> str:
        """The 12-char SHA-256 prefix of the active prompt (read-only)."""
        return self._prompt_hash

    @property
    def model_id(self) -> str:
        """The configured OpenAI model id (read-only)."""
        # ``__init__`` validates ``model_name is not None``.
        assert self._config.model_name is not None
        return self._config.model_name

    def extract(self, event: RemitEvent) -> ExtractionResult:
        """Extract structured features via one OpenAI call.

        Plan §1 D6 + D14 + D16: builds a user message with the event
        rendered as JSON (per the prompt's ``{event_json}`` placeholder),
        calls ``chat.completions.create`` with strict
        ``response_format``, parses the JSON, and constructs an
        :class:`ExtractionResult` stamped with ``prompt_hash`` and
        ``model_id`` provenance. On any failure (network, parse,
        Pydantic validation), logs WARNING and returns the documented
        default.
        """
        try:
            user_message = self._render_user_message(event)
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self._prompt_text},
                    {"role": "user", "content": user_message},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_result",
                        "strict": True,
                        "schema": self.RESPONSE_SCHEMA,
                    },
                },
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(
                    f"OpenAI returned empty content for mrid={event.mrid!r} "
                    f"rev={event.revision_number!r}; expected JSON string."
                )
            payload = json.loads(content)
            return self._build_extraction_result(payload)
        except Exception as exc:  # NFR-6: never raise from extract.
            logger.warning(
                "LlmExtractor.extract failed for mrid={} rev={}: {}; "
                "returning documented default (confidence=0.0).",
                event.mrid,
                event.revision_number,
                exc,
            )
            logger.debug("Failure traceback (mrid={}): {!r}", event.mrid, exc)
            return self._fallback_result(event)

    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
        """Extract a batch — one call per event, order preserved.

        Stage 14 ships a sequential implementation; the OpenAI Chat
        Completions API does not have a free batch primitive (the
        ``/v1/batches`` endpoint is async and adds operational
        complexity that is not justified at the demo scale). The
        Protocol is shaped so a future stage can swap in a parallel
        implementation without changing callers.
        """
        return [self.extract(event) for event in events]

    # ------------------------------------------------------------------
    # Internal helpers — kept module-private; tests use them via the
    # public extract() path.
    # ------------------------------------------------------------------

    @staticmethod
    def _render_user_message(event: RemitEvent) -> str:
        """Render the RemitEvent as a JSON object for the prompt's user message.

        The prompt body refers to ``{event_json}`` — the user message
        substitutes that placeholder by being the JSON object directly.
        ``mode='json'`` ensures datetimes are serialised as ISO strings.
        """
        return event.model_dump_json(indent=2)

    def _build_extraction_result(self, payload: dict[str, Any]) -> ExtractionResult:
        """Construct an ExtractionResult from an OpenAI strict-mode payload.

        Strict mode guarantees every required field is present with the
        right type, but Pydantic still validates because the schema's
        ``confidence`` constraint (``[0, 1]``) and the datetime-aware
        validators (UTC required) are stricter than what the LLM
        contract enforces.
        """
        return ExtractionResult(
            event_type=payload["event_type"],
            fuel_type=payload["fuel_type"],
            affected_capacity_mw=payload.get("affected_capacity_mw"),
            effective_from=payload["effective_from"],
            effective_to=payload.get("effective_to"),
            confidence=payload["confidence"],
            prompt_hash=self._prompt_hash,
            model_id=self.model_id,
        )

    def _fallback_result(self, event: RemitEvent) -> ExtractionResult:
        """Documented default per plan §1 D16 — provenance recorded.

        Mirrors the StubExtractor's miss-path defaults for parity, plus
        the live extractor's ``prompt_hash`` and ``model_id`` so a
        downstream consumer can still see "this came from prompt X
        on model Y" even on a fallback row.
        """
        return ExtractionResult(
            event_type=event.event_type if event.event_type is not None else "Other",
            fuel_type=event.fuel_type if event.fuel_type is not None else "Other",
            affected_capacity_mw=None,
            effective_from=event.effective_from,
            effective_to=event.effective_to,
            confidence=0.0,
            prompt_hash=self._prompt_hash,
            model_id=self.model_id,
        )


# ---------------------------------------------------------------------
# Factory — single dispatch point; honours BRISTOL_ML_LLM_STUB
# ---------------------------------------------------------------------


def build_extractor(
    config: LlmExtractorConfig | None,
    *,
    gold_set_path: Path | None = None,
) -> Extractor:
    """Construct the active :class:`Extractor` for ``config``.

    Plan §1 D3 / D4: dispatch on ``config.type`` *and* the
    ``BRISTOL_ML_LLM_STUB`` env var. The env var is load-bearing for
    CI safety — it forces the stub regardless of YAML so a misconfigured
    test cannot fire the live API.

    Calling with ``config=None`` is supported and returns a
    :class:`StubExtractor` — useful for code paths that want a usable
    extractor even when the config tree did not compose ``llm`` (e.g.
    Stage 15 / Stage 16 unit tests).

    ``gold_set_path`` is forwarded to :class:`StubExtractor`; ignored
    on the live path.
    """
    stub_env_set = os.environ.get(STUB_ENV_VAR) == "1"
    if config is None or config.type == "stub" or stub_env_set:
        if stub_env_set and config is not None and config.type != "stub":
            logger.info(
                "{}=1 overrides config.type={!r}; using StubExtractor.",
                STUB_ENV_VAR,
                config.type,
            )
        return StubExtractor(gold_set_path=gold_set_path)
    if config.type == "openai":
        return LlmExtractor(config)
    # Defensive: a future ``Literal["stub", "openai", "<future>"]``
    # extension that forgets to update this branch should fail loudly
    # rather than silently returning the stub.
    raise ValueError(
        f"build_extractor: unsupported config.type={config.type!r}. "
        "Add a dispatch branch in bristol_ml.llm.extractor when "
        "extending the LlmExtractorConfig.type literal."
    )


# ---------------------------------------------------------------------
# Standalone CLI (NFR-8 / DESIGN §2.1.1)
# ---------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the ``python -m bristol_ml.llm.extractor`` parser."""
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.llm.extractor",
        description=(
            "Run a single extraction against the active configuration "
            "and print the result. Hydra-style overrides are accepted; "
            "by default the offline stub path is selected. "
            "Composes the +llm=extractor config group automatically."
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
    return parser


def _format_extraction_summary(
    result: ExtractionResult,
    *,
    implementation: str,
    gold_set_size: int | None,
) -> str:
    """Format a one-extraction summary for the CLI's stdout output.

    Deterministic to support the standalone smoke test (NFR-8).
    """
    lines = [
        "=== Stage 14 LLM extractor ===",
        f"implementation: {implementation}",
        f"gold_set_size:  {gold_set_size if gold_set_size is not None else '—'}",
        "sample extraction:",
        f"  event_type:           {result.event_type}",
        f"  fuel_type:            {result.fuel_type}",
        f"  affected_capacity_mw: {result.affected_capacity_mw}",
        f"  effective_from:       {result.effective_from.isoformat()}",
        f"  effective_to:         "
        f"{result.effective_to.isoformat() if result.effective_to else '—'}",
        f"  confidence:           {result.confidence}",
        f"  prompt_hash:          {result.prompt_hash or 'none'}",
        f"  model_id:             {result.model_id or 'none'}",
        "=== end ===",
    ]
    return "\n".join(lines)


def _sample_event() -> RemitEvent:
    """Return a deterministic ``RemitEvent`` for the CLI smoke output.

    Matches gold-set record M-A so the stub returns its labelled
    extraction (``confidence=1.0``); the example reads as
    ``planned nuclear outage`` rather than the documented default.
    """
    from datetime import UTC, datetime  # local import; standalone CLI

    return RemitEvent(
        mrid="M-A",
        revision_number=0,
        message_status="Active",
        published_at=datetime(2024, 1, 1, 9, tzinfo=UTC),
        effective_from=datetime(2024, 1, 15, tzinfo=UTC),
        effective_to=datetime(2024, 1, 20, tzinfo=UTC),
        fuel_type="Nuclear",
        affected_mw=600.0,
        event_type="Outage",
        cause="Planned",
        message_description="Stub: planned nuclear outage for refuelling.",
    )


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry — DESIGN §2.1.1, plan NFR-8.

    Resolves the active :class:`LlmExtractorConfig` via Hydra
    (composing ``+llm=extractor`` automatically) and prints a
    deterministic summary. Returns 0 on success, 2 on a config or
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
        extractor = build_extractor(cfg.llm)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Extractor build error: {exc}", file=sys.stderr)
        return 2

    implementation = type(extractor).__name__
    gold_set_size: int | None = None
    if isinstance(extractor, StubExtractor):
        gold_set_size = extractor.gold_set_size

    sample = _sample_event()
    try:
        result = extractor.extract(sample)
    except NotImplementedError as exc:
        # T3: the live path is a placeholder; surface that cleanly.
        print(f"Live extractor not yet wired (T4): {exc}", file=sys.stderr)
        return 2

    print(
        _format_extraction_summary(
            result,
            implementation=implementation,
            gold_set_size=gold_set_size,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(_cli_main())
