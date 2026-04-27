"""Spec-derived tests for the Stage 14 T4 LlmExtractor (OpenAI strict mode).

Every test here is derived from:

- ``docs/plans/completed/14-llm-extractor.md`` §6 T4 named tests:
  ``test_stub_and_llm_extractors_satisfy_protocol_structurally``
  (LLM half), ``test_llm_extractor_config_discriminator_supports_third_literal_slot``,
  ``test_model_name_and_endpoint_are_in_yaml_not_code``,
  ``test_malformed_llm_response_logs_warning_and_returns_default``,
  ``test_prompt_hash_changes_when_prompt_file_changes``.
- Plan §1 D5 (env-var API key gate), D6 (OpenAI strict-mode schema —
  ``additionalProperties: false`` on every object + every property
  ``required``), D14 (provenance pair on every result), D16 (graceful
  default on parse failure), NFR-5 (prompt versioning via SHA-256
  hash), NFR-6 (graceful degradation — never raise).
- AC-1 (interface small enough that adding a third implementation is
  plausible), AC-3 (real implementation guarded by config switch +
  env var), AC-5 (typed boundary).

The cassette-backed integration test lives at
``tests/integration/llm/test_llm_extractor_cassette.py``; this file
covers the OpenAI-call wiring with a mocked client so it runs in any
environment without an API key.

No production code is modified here.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from bristol_ml.llm import ExtractionResult, Extractor, RemitEvent
from bristol_ml.llm._prompts import (
    PROMPT_HASH_PREFIX_CHARS,
    load_prompt,
    prompt_sha256_prefix,
)
from bristol_ml.llm.extractor import (
    STUB_ENV_VAR,
    LlmExtractor,
)
from conf._schemas import LlmExtractorConfig

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def openai_config(monkeypatch: pytest.MonkeyPatch) -> LlmExtractorConfig:
    """A populated openai-typed config with a real API-key env var.

    Uses the canonical ``conf/llm/prompts/extract_v1.txt`` shipped at
    T1 so prompt-hash assertions reference the actual repo file.
    """
    monkeypatch.setenv("BRISTOL_ML_LLM_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv(STUB_ENV_VAR, raising=False)
    return LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="BRISTOL_ML_LLM_API_KEY",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
        request_timeout_seconds=30.0,
    )


@pytest.fixture
def sample_event() -> RemitEvent:
    """One representative event for ``LlmExtractor.extract`` calls."""
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


def _patch_openai_client(
    extractor: LlmExtractor,
    *,
    response_content: str | None,
) -> MagicMock:
    """Replace the extractor's OpenAI client with a MagicMock returning content.

    Returns the chat-completions ``create`` mock so the test can assert
    on the kwargs passed (model, messages, response_format) when the
    structural shape is the load-bearing claim.
    """
    create_mock = MagicMock()
    if response_content is not None:
        choice = MagicMock()
        choice.message.content = response_content
        completion = MagicMock()
        completion.choices = [choice]
        create_mock.return_value = completion
    extractor._client = MagicMock()  # type: ignore[attr-defined]
    extractor._client.chat.completions.create = create_mock
    return create_mock


# ---------------------------------------------------------------------
# AC-1 — protocol satisfaction (LLM half)
# ---------------------------------------------------------------------


def test_llm_extractor_satisfies_protocol_structurally(
    openai_config: LlmExtractorConfig,
) -> None:
    """Guards plan §6 T4 named test (LLM half) / AC-1.

    The Protocol's promise — *"a third implementation in the future
    is plausible"* — is hollow if the live extractor doesn't satisfy
    it structurally.  Pairs with the stub-half assertion in
    ``test_stub_extractor.py``.
    """
    extractor = LlmExtractor(openai_config)
    assert isinstance(extractor, Extractor)


def test_llm_extractor_config_discriminator_supports_third_literal_slot() -> None:
    """Guards plan §6 T4 / AC-1: the discriminator literal is extensible.

    AC-1 sub-criterion: adding a future ``"future"`` slot to
    ``LlmExtractorConfig.type`` should be a one-line schema change
    (plus a dispatch branch in ``build_extractor``).  This test
    captures the principle by asserting both currently allowed values
    parse and that the ``type`` field's JSON schema enumerates exactly
    them — so a future literal must appear deliberately, not as a
    silent regression.
    """
    # Both currently allowed values resolve.
    LlmExtractorConfig(type="stub")
    LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
    )
    # A deliberately-wrong value is rejected at the discriminator.
    with pytest.raises(ValidationError):
        LlmExtractorConfig(type="anthropic")  # type: ignore[arg-type]
    # The JSON schema enumerates exactly the two allowed values, so
    # adding a third is a deliberate plan edit.
    schema = LlmExtractorConfig.model_json_schema()
    type_enum = schema["properties"]["type"]["enum"]
    assert sorted(type_enum) == ["openai", "stub"], (
        "LlmExtractorConfig.type literal must be exactly "
        "{'stub', 'openai'} until a plan edit widens it; "
        f"got {type_enum!r}."
    )


# ---------------------------------------------------------------------
# AC-3 — real implementation guarded by config + env var
# ---------------------------------------------------------------------


def test_llm_extractor_init_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan §6 T4 / D5 / AC-3.

    With ``type='openai'`` and no API key in the env var (and the
    stub-override env var unset), ``__init__`` must raise
    :class:`RuntimeError` naming both env-vars so the operator sees
    the offline escape hatch.
    """
    monkeypatch.delenv(STUB_ENV_VAR, raising=False)
    monkeypatch.delenv("BRISTOL_ML_LLM_API_KEY", raising=False)
    config = LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        LlmExtractor(config)
    msg = str(excinfo.value)
    assert "BRISTOL_ML_LLM_API_KEY" in msg
    assert STUB_ENV_VAR in msg


def test_llm_extractor_init_requires_prompt_file() -> None:
    """Guards plan §1 D7 / NFR-5: ``prompt_file`` must be set when live.

    The prompt is the source of the SHA-256 provenance hash; an init
    with ``prompt_file=None`` would produce results without the
    hash, undermining NFR-5 silently.  Surface it loudly at construction.
    """
    config = LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        prompt_file=None,
    )
    with pytest.raises(ValueError, match=r"prompt_file"):
        LlmExtractor(config)


def test_model_name_and_endpoint_are_in_yaml_not_code() -> None:
    """Guards plan §6 T4 / NFR-7: no model literals hard-coded in src/.

    NFR-7 (DESIGN §2.1.4): configuration lives outside code.  A
    string literal starting ``gpt-`` or ``claude-`` in
    ``src/bristol_ml/llm/`` is a config-leak smell; the test
    forecloses on it via grep.

    Allowed exceptions: docstrings reference model families
    illustratively (e.g. *"e.g. 'gpt-4o-mini'"* in a class docstring).
    The test searches for non-comment, non-docstring occurrences by
    excluding lines whose stripped form starts with ``"`` (docstring
    body) or ``#`` (comment).
    """
    src_dir = Path("src/bristol_ml/llm")
    pattern = re.compile(r"\b(gpt-|claude-)[a-z0-9._-]+\b", re.IGNORECASE)
    offenders: list[tuple[Path, int, str]] = []

    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            # Skip comment-only lines and lines that are clearly
            # inside a docstring (the project's style is """\n...\n""").
            if stripped.startswith("#"):
                continue
            if pattern.search(stripped) and not (
                stripped.startswith('"""')
                or stripped.startswith("'''")
                or '"""' in stripped[: stripped.find(":")]  # def line
                or stripped.startswith('"')
                or stripped.startswith("'")
            ):
                offenders.append((py_file, lineno, line))

    # The remaining offenders must be inside docstrings — check by
    # eyeballing the file region; the heuristic above admits some
    # docstring lines.  Filter out lines that appear to be inside a
    # triple-quoted block.
    confirmed: list[tuple[Path, int, str]] = []
    for path, lineno, line in offenders:
        # If the surrounding region looks like a docstring (lines
        # bracketed by triple quotes), skip; otherwise it's a code-line
        # leak.
        all_lines = path.read_text(encoding="utf-8").splitlines()
        in_docstring = False
        triple_count = 0
        for idx in range(lineno):
            if '"""' in all_lines[idx] or "'''" in all_lines[idx]:
                triple_count += all_lines[idx].count('"""')
                triple_count += all_lines[idx].count("'''")
        in_docstring = triple_count % 2 == 1
        if not in_docstring:
            confirmed.append((path, lineno, line))

    assert not confirmed, (
        "NFR-7 violation: model-name string literal in src/bristol_ml/llm/. "
        "Move to conf/llm/extractor.yaml.\n"
        + "\n".join(f"  {p}:{ln}: {line!r}" for p, ln, line in confirmed)
    )


# ---------------------------------------------------------------------
# AC-5 + D14 — provenance + typed boundary
# ---------------------------------------------------------------------


def test_extract_payload_round_trips_through_extraction_result(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
) -> None:
    """Guards plan §5 / D14: a strict-mode response yields a valid ExtractionResult.

    The extractor must accept a payload conforming to its declared
    schema and produce a fully-typed :class:`ExtractionResult` with
    ``prompt_hash`` and ``model_id`` provenance stamped from the
    extractor's own state — not from the LLM payload (which doesn't
    carry them).
    """
    extractor = LlmExtractor(openai_config)
    payload = """
    {
        "event_type": "Outage",
        "fuel_type": "Nuclear",
        "affected_capacity_mw": 600.0,
        "effective_from": "2024-01-15T00:00:00Z",
        "effective_to": "2024-01-20T00:00:00Z",
        "confidence": 1.0
    }
    """
    create_mock = _patch_openai_client(extractor, response_content=payload)
    result = extractor.extract(sample_event)

    assert isinstance(result, ExtractionResult)
    assert result.event_type == "Outage"
    assert result.fuel_type == "Nuclear"
    assert result.affected_capacity_mw == pytest.approx(600.0)
    assert result.confidence == 1.0
    # Provenance is stamped by the extractor, not echoed from the LLM.
    assert result.prompt_hash == extractor.prompt_hash
    assert result.model_id == "gpt-4o-mini"
    # The strict response_format is the load-bearing constraint —
    # check it was actually passed.
    create_mock.assert_called_once()
    kwargs = create_mock.call_args.kwargs
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False, (
        "OpenAI strict mode requires additionalProperties=False on the root "
        "object (plan §1 D6 schema-shape note)."
    )
    assert set(schema["required"]) == {
        "event_type",
        "fuel_type",
        "affected_capacity_mw",
        "effective_from",
        "effective_to",
        "confidence",
    }, "OpenAI strict mode requires every property in 'required' (plan §1 D6 schema-shape note)."


def test_response_schema_is_strict_mode_compliant() -> None:
    """Guards plan §1 D6: the response schema satisfies OpenAI's strict-mode rules.

    The two strict-mode rules are:
    - ``additionalProperties: false`` on every object.
    - Every property listed in ``required``.

    A failure here is the kind of silent regression that the OpenAI
    API would reject at first call with ``Invalid schema``; pinning
    it as a unit-level invariant catches the bug before any network
    call.
    """
    schema = LlmExtractor.RESPONSE_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys()), (
        "Every property must be in 'required' under OpenAI strict mode."
    )


# ---------------------------------------------------------------------
# NFR-5 — prompt versioning via SHA-256 hash
# ---------------------------------------------------------------------


def test_prompt_hash_changes_when_prompt_file_changes(tmp_path: Path) -> None:
    """Guards plan §6 T4 / NFR-5: a one-byte edit produces a different hash.

    The hash is the load-bearing identity that lets a downstream
    consumer say "we changed the prompt and the extractions
    changed because of it" — vs. some upstream-data drift.  If the
    hash is stable across edits, NFR-5 is hollow.
    """
    p1 = tmp_path / "prompt_v1.txt"
    p2 = tmp_path / "prompt_v2.txt"
    p1.write_text("Extract the structured features from the event.\n")
    p2.write_text("Extract the structured features from the event.")  # no trailing newline

    _, hash1 = load_prompt(p1)
    _, hash2 = load_prompt(p2)
    assert hash1 != hash2, (
        "A one-byte edit must produce a different prompt_hash (NFR-5); "
        f"got hash1={hash1} hash2={hash2}."
    )
    assert len(hash1) == PROMPT_HASH_PREFIX_CHARS
    assert len(hash2) == PROMPT_HASH_PREFIX_CHARS


def test_prompt_hash_prefix_matches_sha256() -> None:
    """Guards plan §5 / NFR-5: the prefix is the SHA-256 truncated to 12 chars.

    The ``ExtractionResult.prompt_hash`` field is documented as the
    first 12 hex chars of the SHA-256 digest.  This test pins the
    algorithm so a future reimplementation cannot silently use a
    different hash function.
    """
    import hashlib

    prompt_bytes = b"hello, prompt"
    expected = hashlib.sha256(prompt_bytes).hexdigest()[:PROMPT_HASH_PREFIX_CHARS]
    assert prompt_sha256_prefix(prompt_bytes) == expected


def test_prompt_hash_is_recorded_in_extraction_result(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
) -> None:
    """Guards plan §1 D10 / NFR-5: every ExtractionResult carries the active hash.

    The extractor stamps the active prompt's SHA-256 prefix into every
    result it produces — happy path AND fallback path — so a
    persisted column "this row came from prompt X" is always
    answerable.
    """
    extractor = LlmExtractor(openai_config)
    payload = """
    {
        "event_type": "Outage",
        "fuel_type": "Nuclear",
        "affected_capacity_mw": 600.0,
        "effective_from": "2024-01-15T00:00:00Z",
        "effective_to": "2024-01-20T00:00:00Z",
        "confidence": 1.0
    }
    """
    _patch_openai_client(extractor, response_content=payload)
    result = extractor.extract(sample_event)
    assert result.prompt_hash is not None
    assert len(result.prompt_hash) == PROMPT_HASH_PREFIX_CHARS
    # And it matches the computed hash of the canonical prompt file.
    # Anchor on the source-tree root (same rationale as the
    # source-side fix to ``DEFAULT_GOLD_SET_PATH``) so the test does
    # not silently rely on the pytest cwd being the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    canonical_bytes = (repo_root / "conf/llm/prompts/extract_v1.txt").read_bytes()
    assert result.prompt_hash == prompt_sha256_prefix(canonical_bytes)


# ---------------------------------------------------------------------
# NFR-6 / D16 — graceful degradation
# ---------------------------------------------------------------------


def test_malformed_llm_response_logs_warning_and_returns_default(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards plan §6 T4 / NFR-6 / D16.

    A malformed OpenAI response (unparseable JSON) must:
    1. NOT raise from ``extract`` (NFR-6).
    2. Emit a WARNING naming the event id (operability).
    3. Return an :class:`ExtractionResult` with the documented default
       (D16) — ``confidence=0.0``, structural fields mirrored from
       input, capacity ``None``, provenance recorded.
    """
    extractor = LlmExtractor(openai_config)
    _patch_openai_client(extractor, response_content="not valid json {{")

    # Bridge loguru → caplog so caplog.records sees the WARNING.
    from loguru import logger

    handler_id = logger.add(
        lambda message: caplog.handler.handle(_loguru_to_logging(message)),
        format="{message}",
        level="WARNING",
    )
    try:
        result = extractor.extract(sample_event)
    finally:
        logger.remove(handler_id)

    assert isinstance(result, ExtractionResult)
    assert result.confidence == 0.0
    # Structural fields mirror input.
    assert result.event_type == sample_event.event_type
    assert result.fuel_type == sample_event.fuel_type
    assert result.effective_from == sample_event.effective_from
    assert result.effective_to == sample_event.effective_to
    assert result.affected_capacity_mw is None
    # Provenance is still recorded so the failure is diagnosable.
    assert result.prompt_hash is not None
    assert result.model_id == "gpt-4o-mini"
    # The WARNING names the event mrid.
    warnings = [r for r in caplog.records if r.levelno >= 30]
    assert any("M-A" in r.getMessage() for r in warnings), (
        f"WARNING must name the failing event mrid; got {[r.getMessage() for r in warnings]!r}."
    )


def test_validation_error_path_returns_default(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
) -> None:
    """Guards plan D16: a Pydantic-validation failure also returns the default.

    The strict response_format pins the JSON shape, but Pydantic's
    further constraints (UTC-aware datetimes, confidence ∈ [0, 1])
    are stricter than what OpenAI's schema enforces.  A response
    that satisfies the JSON schema but violates Pydantic must hit
    the same fallback path.
    """
    extractor = LlmExtractor(openai_config)
    # Confidence > 1.0 — JSON-valid, schema-tolerated (no max enforced
    # by OpenAI for strict mode), Pydantic-rejected.
    payload = """
    {
        "event_type": "Outage",
        "fuel_type": "Nuclear",
        "affected_capacity_mw": 600.0,
        "effective_from": "2024-01-15T00:00:00Z",
        "effective_to": "2024-01-20T00:00:00Z",
        "confidence": 1.5
    }
    """
    _patch_openai_client(extractor, response_content=payload)
    result = extractor.extract(sample_event)
    assert result.confidence == 0.0  # fallback


def test_network_error_path_returns_default(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
) -> None:
    """Guards plan D16 / NFR-6: a network exception is also caught.

    The OpenAI SDK raises various exceptions for HTTP errors.  The
    extractor must catch any of them — never propagate.
    """
    extractor = LlmExtractor(openai_config)
    create_mock = MagicMock(side_effect=RuntimeError("Simulated 503 from OpenAI"))
    extractor._client = MagicMock()  # type: ignore[attr-defined]
    extractor._client.chat.completions.create = create_mock
    result = extractor.extract(sample_event)
    assert isinstance(result, ExtractionResult)
    assert result.confidence == 0.0


def test_failure_log_does_not_leak_exception_message_or_repr(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards Phase-3 code-review finding: API-key partial leak in logs.

    OpenAI's ``AuthenticationError`` echoes a partial key
    (e.g. ``"Incorrect API key provided: sk-real*****CDEF"``) in the
    server-supplied message; logging the exception's ``str`` or
    ``repr`` would land that fragment in operator logs / shared demo
    output. The fallback path must log only ``type(exc).__name__``
    and identifying event metadata — never the exception itself.

    DESIGN §7: secrets must not appear in code, config, or logs.
    """
    extractor = LlmExtractor(openai_config)
    # Simulate the worst-case AuthenticationError-style message: an
    # exception whose str / repr contains an API-key fragment.
    leaky_secret = "sk-realk1234EXAMPLECDEF"
    create_mock = MagicMock(
        side_effect=RuntimeError(f"Incorrect API key provided: {leaky_secret}.")
    )
    extractor._client = MagicMock()  # type: ignore[attr-defined]
    extractor._client.chat.completions.create = create_mock

    # Bridge loguru → caplog so caplog.records sees both WARNING + DEBUG.
    from loguru import logger

    handler_id = logger.add(
        lambda message: caplog.handler.handle(_loguru_to_logging(message)),
        format="{message}",
        level="DEBUG",
    )
    try:
        result = extractor.extract(sample_event)
    finally:
        logger.remove(handler_id)

    assert result.confidence == 0.0  # fallback path, as expected
    # The leaky fragment must not appear in any captured log record.
    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert leaky_secret not in rendered, (
        "API-key fragment leaked into log output; the failure-path "
        f"logger must not include the exception message/repr. "
        f"Records: {rendered!r}."
    )
    # And the fallback warning must still name the event so the
    # operator can identify which call failed.
    assert any("M-A" in r.getMessage() for r in caplog.records), (
        "WARNING must still name the failing event mrid for operability; "
        f"got {[r.getMessage() for r in caplog.records]!r}."
    )


# ---------------------------------------------------------------------
# extract_batch — order preserved
# ---------------------------------------------------------------------


def test_llm_extract_batch_preserves_input_order(
    openai_config: LlmExtractorConfig,
    sample_event: RemitEvent,
) -> None:
    """Guards plan §5: ``extract_batch`` returns results in input order.

    Stage 16 (feature-table join) zips inputs to extractions; an
    out-of-order return would silently misalign the join.
    """
    extractor = LlmExtractor(openai_config)
    payload_1 = """
    {
        "event_type": "Outage",
        "fuel_type": "Nuclear",
        "affected_capacity_mw": 600.0,
        "effective_from": "2024-01-15T00:00:00Z",
        "effective_to": "2024-01-20T00:00:00Z",
        "confidence": 1.0
    }
    """
    payload_2 = """
    {
        "event_type": "Restriction",
        "fuel_type": "Wind",
        "affected_capacity_mw": null,
        "effective_from": "2024-04-10T00:00:00Z",
        "effective_to": null,
        "confidence": 0.5
    }
    """
    create_mock = MagicMock()
    completion_1 = MagicMock(choices=[MagicMock(message=MagicMock(content=payload_1))])
    completion_2 = MagicMock(choices=[MagicMock(message=MagicMock(content=payload_2))])
    create_mock.side_effect = [completion_1, completion_2]
    extractor._client = MagicMock()  # type: ignore[attr-defined]
    extractor._client.chat.completions.create = create_mock

    second_event = sample_event.model_copy(update={"mrid": "M-D", "fuel_type": "Wind"})
    results = extractor.extract_batch([sample_event, second_event])
    assert len(results) == 2
    assert results[0].event_type == "Outage"
    assert results[1].event_type == "Restriction"


# ---------------------------------------------------------------------
# Helpers used above
# ---------------------------------------------------------------------


def _loguru_to_logging(message: Any) -> Any:
    """Bridge a loguru record into a stdlib ``logging.LogRecord``.

    ``caplog`` is the stdlib-logging fixture; the project uses loguru
    via :func:`loguru.logger`.  The bridge turns a loguru INFO/WARNING
    into a stdlib record so :attr:`caplog.records` sees it.
    """
    import logging

    record = message.record
    log_record = logging.LogRecord(
        name=record["name"],
        level=record["level"].no,
        pathname=record["file"].path,
        lineno=record["line"],
        msg=record["message"],
        args=(),
        exc_info=None,
    )
    return log_record
