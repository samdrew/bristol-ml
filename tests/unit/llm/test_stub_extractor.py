"""Spec-derived tests for the Stage 14 T3 StubExtractor + factory + CLI.

Every test here is derived from:

- ``docs/plans/completed/14-llm-extractor.md`` §6 T3 named tests:
  ``test_stub_and_llm_extractors_satisfy_protocol_structurally`` (stub
  half), ``test_stub_active_when_env_var_set``,
  ``test_stub_default_in_yaml_config``, ``test_stub_makes_no_network_call``,
  ``test_stub_returns_default_for_unknown_event``,
  ``test_extractor_module_runs_standalone``.
- Plan §1 D3 (factory dispatches on ``config.type``), D4 (env-var double
  gate), D9 (gold-set fixture path + JSON shape), D14 (confidence
  sentinels), D16 (documented default for unknown events).
- AC-1 ("interface small enough … third implementation plausible"),
  AC-2 ("stub is the default; runs offline with no API key"), AC-5
  ("schema typed and validated at the interface boundary").

No production code is modified here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from bristol_ml.llm import ExtractionResult, Extractor, RemitEvent
from bristol_ml.llm.extractor import (
    DEFAULT_GOLD_SET_PATH,
    STUB_ENV_VAR,
    LlmExtractor,
    StubExtractor,
    build_extractor,
)
from conf._schemas import LlmExtractorConfig

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _known_event() -> RemitEvent:
    """Return a RemitEvent whose ``(mrid, revision_number)`` is in the gold set.

    Must stay in sync with ``tests/fixtures/llm/hand_labelled.json`` —
    M-A is the canonical "fresh single-revision Active" record (Stage
    13 ``_stub_records``).
    """
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


def _unknown_event() -> RemitEvent:
    """A RemitEvent whose key is NOT in the gold set — exercises the default path."""
    return RemitEvent(
        mrid="MRID-NOT-IN-GOLD-SET",
        revision_number=42,
        message_status="Active",
        published_at=datetime(2025, 7, 1, 12, tzinfo=UTC),
        effective_from=datetime(2025, 8, 1, tzinfo=UTC),
        effective_to=datetime(2025, 8, 2, tzinfo=UTC),
        fuel_type="Battery",
        affected_mw=20.0,
        event_type="Restriction",
        cause="Forced",
        message_description=None,
    )


# ---------------------------------------------------------------------
# AC-1 — structural protocol satisfaction
# ---------------------------------------------------------------------


def test_stub_extractor_satisfies_protocol_structurally() -> None:
    """Guards plan §6 T3 named test (stub half) / AC-1.

    AC-1: a third implementation must be plausible without inheritance.
    The stub is the load-bearing first proof — if ``runtime_checkable``
    on the Protocol does not accept the stub structurally, AC-1's
    promise is hollow.
    """
    stub = StubExtractor()
    assert isinstance(stub, Extractor), (
        "StubExtractor must satisfy Extractor structurally (AC-1); "
        "isinstance via runtime_checkable Protocol failed."
    )


# ---------------------------------------------------------------------
# AC-2 — offline by default
# ---------------------------------------------------------------------


def test_stub_default_in_yaml_config() -> None:
    """Guards plan AC-2 / §6 T3: ``conf/llm/extractor.yaml`` defaults to ``type: stub``.

    The YAML default + the Pydantic discriminator together carry the
    offline-by-default invariant. If the YAML drifts, every CI run
    that does not explicitly opt in starts firing the live path.
    """
    from bristol_ml.config import load_config

    cfg = load_config(overrides=["+llm=extractor"])
    assert cfg.llm is not None
    assert cfg.llm.type == "stub", (
        "conf/llm/extractor.yaml must default to type='stub' (AC-2). "
        f"Got cfg.llm.type={cfg.llm.type!r}."
    )


def test_stub_active_when_env_var_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards plan AC-2 / D4: ``BRISTOL_ML_LLM_STUB=1`` overrides config.type.

    The env-var is the load-bearing CI safety mechanism — even if a
    YAML override accidentally selects ``type=openai``, setting
    ``BRISTOL_ML_LLM_STUB=1`` forces the stub. Without this guard a
    misconfigured CI run could fire the live API.
    """
    monkeypatch.setenv(STUB_ENV_VAR, "1")
    # Construct a config whose ``type`` is openai but missing all
    # secrets — the env var must short-circuit before any of that
    # matters.
    config = LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="BRISTOL_ML_LLM_API_KEY",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
        request_timeout_seconds=30.0,
    )
    extractor = build_extractor(config)
    assert isinstance(extractor, StubExtractor), (
        f"BRISTOL_ML_LLM_STUB=1 must force StubExtractor regardless of "
        f"config.type; got {type(extractor).__name__}."
    )


def test_build_extractor_with_none_config_returns_stub() -> None:
    """Guards plan §5: ``build_extractor(None)`` is a usable code path.

    Stage 15 + Stage 16 unit tests need a working extractor without
    composing the ``llm`` config group. Returning the stub on ``None``
    keeps those test surfaces ergonomic without weakening the
    discriminator elsewhere.
    """
    extractor = build_extractor(None)
    assert isinstance(extractor, StubExtractor)


def test_stub_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards plan AC-2 / NFR-1: the stub path makes zero HTTP calls.

    The mechanism: monkeypatch ``httpx.Client.send`` to raise on any
    call; the stub's ``extract`` must succeed against a known event.
    If the stub ever takes a network dependency, this test fails with
    the patched RuntimeError pointing at the offending call site.
    """

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            f"Stub path attempted a network call (NFR-1 violated): args={args!r} kwargs={kwargs!r}"
        )

    monkeypatch.setattr(httpx.Client, "send", _explode)
    extractor = StubExtractor()
    result = extractor.extract(_known_event())
    assert result.event_type == "Outage"
    assert result.confidence == 1.0


# ---------------------------------------------------------------------
# AC-5 — typed return; documented default
# ---------------------------------------------------------------------


def test_stub_returns_extraction_result_for_known_event() -> None:
    """Guards plan §6 T3 / D14: known events return ``confidence=1.0``.

    M-A is the canonical "fresh single-revision Active" record; the
    gold set labels its extraction with ``confidence=1.0`` (plan D14
    sentinel for hand-labelled hits).
    """
    extractor = StubExtractor()
    result = extractor.extract(_known_event())
    assert isinstance(result, ExtractionResult)
    assert result.event_type == "Outage"
    assert result.fuel_type == "Nuclear"
    assert result.affected_capacity_mw == pytest.approx(600.0)
    assert result.confidence == 1.0
    # Provenance is None for the stub — plan §5 schema sketch.
    assert result.prompt_hash is None
    assert result.model_id is None


def test_stub_returns_default_for_unknown_event() -> None:
    """Guards plan §6 T3 / D14 / D16: unknown events get the documented default.

    Plan D16: on miss, ``confidence=0.0``, structural fields synthesised
    from the input where non-NULL, capacity ``None``, times mirrored
    from input. The structural-field passthrough is what makes the
    stub useful even on misses — Stage 15 / Stage 16 callers always
    receive a fully-populated ``ExtractionResult``.
    """
    extractor = StubExtractor()
    event = _unknown_event()
    result = extractor.extract(event)
    assert isinstance(result, ExtractionResult)
    assert result.confidence == 0.0
    assert result.event_type == "Restriction", (
        "Unknown-event default must mirror the input's event_type when non-NULL."
    )
    assert result.fuel_type == "Battery", (
        "Unknown-event default must mirror the input's fuel_type when non-NULL."
    )
    assert result.affected_capacity_mw is None, (
        "Unknown-event default must set affected_capacity_mw=None per plan D16."
    )
    assert result.effective_from == event.effective_from
    assert result.effective_to == event.effective_to


def test_stub_unknown_event_with_null_structural_fields_uses_other() -> None:
    """Guards plan §1 D16: NULL structural fields fall back to ``"Other"``.

    Plan §1 D16 documents the unknown-event default; when both
    ``event_type`` and ``fuel_type`` are NULL on the input, the stub
    must still return a fully-typed ``ExtractionResult`` — the only
    sensible default is the open-vocab ``"Other"`` value (matching
    Elexon's own ``FUEL_TYPES`` vocab + the prompt's ``event_type``
    enumeration).
    """
    event = RemitEvent(
        mrid="MRID-NULL-FIELDS",
        revision_number=0,
        message_status="Active",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
        effective_from=datetime(2025, 2, 1, tzinfo=UTC),
        effective_to=None,
        fuel_type=None,
        affected_mw=None,
        event_type=None,
        cause=None,
        message_description=None,
    )
    extractor = StubExtractor()
    result = extractor.extract(event)
    assert result.event_type == "Other"
    assert result.fuel_type == "Other"
    assert result.confidence == 0.0


def test_stub_extract_batch_preserves_input_order() -> None:
    """Guards plan §5: ``extract_batch`` returns results in input order.

    The Protocol says results are returned in input order — Stage 16
    (feature-table join) will rely on this when zipping inputs to
    extractions. Order-preservation is cheap on the stub (no
    parallelism); this test pins the contract.
    """
    extractor = StubExtractor()
    events = [_known_event(), _unknown_event(), _known_event()]
    results = extractor.extract_batch(events)
    assert len(results) == 3
    assert results[0].confidence == 1.0  # known
    assert results[1].confidence == 0.0  # unknown
    assert results[2].confidence == 1.0  # known


def test_stub_loads_canonical_gold_set_path() -> None:
    """Guards plan D9: the default gold-set path matches the documented location.

    Plan §1 D9: ``tests/fixtures/llm/hand_labelled.json``. The constant
    is anchored on the source-file location (Phase-3 review fix) so
    callers from any cwd can load it; the test asserts the path tail
    rather than the absolute prefix so a repo move does not break it.
    """
    expected_tail = Path("tests") / "fixtures" / "llm" / "hand_labelled.json"
    assert DEFAULT_GOLD_SET_PATH.is_absolute(), (
        f"DEFAULT_GOLD_SET_PATH must be absolute (anchored on source "
        f"location) so notebook / library callers from any cwd can "
        f"load it; got {DEFAULT_GOLD_SET_PATH}."
    )
    assert DEFAULT_GOLD_SET_PATH.parts[-4:] == expected_tail.parts, (
        f"DEFAULT_GOLD_SET_PATH tail must equal "
        f"tests/fixtures/llm/hand_labelled.json; got "
        f"{DEFAULT_GOLD_SET_PATH}."
    )
    assert DEFAULT_GOLD_SET_PATH.exists(), (
        f"Canonical gold-set path {DEFAULT_GOLD_SET_PATH} is missing — "
        "T3 fixture or path constant has drifted."
    )


def test_stub_loads_canonical_gold_set_from_alternate_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-3 review fix: the default path resolves regardless of cwd.

    The previous ``Path("tests/fixtures/llm/hand_labelled.json")``
    constant was cwd-relative — a notebook in ``notebooks/`` or any
    Stage 15/16 caller from elsewhere on the filesystem hit
    ``FileNotFoundError`` at the default-path resolve. Anchoring on
    ``__file__`` fixes that; this test guards against a regression by
    chdir-ing to a temp directory before constructing the stub.
    """
    monkeypatch.chdir(tmp_path)
    # No gold_set_path override — must resolve via DEFAULT_GOLD_SET_PATH.
    stub = StubExtractor()
    assert stub.gold_set_size > 0, (
        "StubExtractor must load the canonical gold set even when the "
        "process working directory is not the repo root."
    )


def test_stub_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Guards plan D9: ``schema_version != 1`` raises with a useful message.

    Plan §1 D9: the gold-set JSON carries a ``schema_version: int``
    field at the top level so additions are detectable. The loader
    must reject unknown versions loudly; otherwise a future schema
    bump silently breaks the stub.
    """
    bad_path = tmp_path / "future_schema.json"
    bad_path.write_text(json.dumps({"schema_version": 99, "records": []}))
    with pytest.raises(ValueError, match="schema_version"):
        StubExtractor(gold_set_path=bad_path)


def test_stub_rejects_missing_gold_set(tmp_path: Path) -> None:
    """Guards plan §6 T3: missing fixture raises ``FileNotFoundError``.

    The default location is in-repo, so a missing file at construction
    is unambiguously a bug — the loader must surface it immediately
    with the path named in the message.
    """
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.json"):
        StubExtractor(gold_set_path=missing)


# ---------------------------------------------------------------------
# build_extractor — discriminator dispatch
# ---------------------------------------------------------------------


def test_build_extractor_stub_branch() -> None:
    """Guards plan D3 / D4: ``type='stub'`` returns a StubExtractor.

    The factory is the single dispatch point; production callers
    (notebook, harness, Stage 16 join) use it. This test pins the
    happy-path stub branch.
    """
    config = LlmExtractorConfig(type="stub")
    extractor = build_extractor(config)
    assert isinstance(extractor, StubExtractor)


def test_build_extractor_openai_branch_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan D5: ``type='openai'`` without API key raises RuntimeError.

    The error must name both env-vars (the configured API-key var and
    the ``BRISTOL_ML_LLM_STUB`` escape hatch) so the operator sees the
    offline path immediately.
    """
    monkeypatch.delenv(STUB_ENV_VAR, raising=False)
    monkeypatch.delenv("BRISTOL_ML_LLM_API_KEY", raising=False)
    config = LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="BRISTOL_ML_LLM_API_KEY",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
        request_timeout_seconds=30.0,
    )
    with pytest.raises(RuntimeError) as excinfo:
        build_extractor(config)
    msg = str(excinfo.value)
    assert "BRISTOL_ML_LLM_API_KEY" in msg
    assert STUB_ENV_VAR in msg, (
        f"RuntimeError must name {STUB_ENV_VAR} as the offline escape hatch (D5); got: {msg!r}."
    )


def test_llm_extractor_init_succeeds_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan D5: with a populated API key, init does not raise.

    T3 covers the construction guard rails only; the actual OpenAI
    call lands at T4. This test pins the no-raise path so a future
    init-time regression is caught even before T4 wires the call.
    """
    monkeypatch.setenv("BRISTOL_ML_LLM_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv(STUB_ENV_VAR, raising=False)
    config = LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="BRISTOL_ML_LLM_API_KEY",
        prompt_file=Path("conf/llm/prompts/extract_v1.txt"),
        request_timeout_seconds=30.0,
    )
    # Should not raise; T3 stops here.
    extractor = LlmExtractor(config)
    assert isinstance(extractor, Extractor)


def test_llm_extractor_init_rejects_non_openai_type() -> None:
    """Guards plan §5: direct ``LlmExtractor`` construction asserts type.

    The factory dispatches on ``type``; constructing the class directly
    with the wrong type is a programming error and must surface
    immediately rather than later when the OpenAI client tries to
    use a stub-shaped config.
    """
    config = LlmExtractorConfig(type="stub")
    with pytest.raises(ValueError, match=r"config\.type='openai'"):
        LlmExtractor(config)


# ---------------------------------------------------------------------
# NFR-8 — module runs standalone
# ---------------------------------------------------------------------


def test_extractor_module_runs_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards plan §6 T3 / NFR-8 / DESIGN §2.1.1.

    ``python -m bristol_ml.llm.extractor`` exits 0 with the offline
    stub path active. The smoke output names the implementation, the
    gold-set size, and a sample extraction — enough that a developer
    running the module by hand sees the surface working without
    needing the README open.
    """
    # Use subprocess so we exercise the same path a developer / CI
    # would; the env-var stub guard is set explicitly to remove any
    # ambient API key from the host shell.
    env = {
        "BRISTOL_ML_LLM_STUB": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
    }
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.llm.extractor"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/workspace",
        check=False,
    )
    assert result.returncode == 0, (
        f"python -m bristol_ml.llm.extractor exited {result.returncode}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}."
    )
    assert "Stage 14 LLM extractor" in result.stdout
    assert "implementation: StubExtractor" in result.stdout
    assert "gold_set_size:" in result.stdout
