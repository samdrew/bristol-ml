"""Cassette-backed integration tests for :class:`LlmExtractor` (Stage 14 T4).

Plan §6 T4: *"Record one VCR cassette under
``tests/fixtures/llm/cassettes/test_llm_extractor_against_cassette.yaml``
covering ~5 events spanning the gold set's strata; replay-only by
default per ``record-mode=none`` in pyproject.toml."*

The cassette records actual OpenAI Chat Completions API responses and
is replayed under ``--record-mode=none`` so CI never makes a network
call.  Recording requires an :env:`BRISTOL_ML_LLM_API_KEY` populated
against an OpenAI account and is performed once locally:

.. code-block:: shell

    BRISTOL_ML_LLM_API_KEY=sk-... uv run pytest \\
        tests/integration/llm/test_llm_extractor_cassette.py \\
        --record-mode=once

VCR is configured to filter ``authorization``, ``x-api-key``,
``cookie`` and ``set-cookie`` headers (mirrors the precedent at
``tests/integration/ingestion/test_remit_cassettes.py``); the body
of the request still contains the prompt + event JSON, which is fine
because the prompt is open-source in this repo.

Until the cassette is recorded the test skips under the CI default
record mode — same shape as :mod:`test_remit_cassettes`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pytest_recording")
pytest.importorskip("openai")

from bristol_ml.llm import ExtractionResult, RemitEvent
from bristol_ml.llm.extractor import LlmExtractor
from conf._schemas import LlmExtractorConfig

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "llm"
CASSETTES = FIXTURES / "cassettes"
CASSETTE_STEM = "test_llm_extractor_against_cassette"
CASSETTE_FILE = f"{CASSETTE_STEM}.yaml"


# --------------------------------------------------------------------------- #
# pytest-recording wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _cassette_present_or_skip(request: pytest.FixtureRequest) -> None:
    """Skip when no cassette exists yet (build-up phase).

    Honoured under ``--record-mode=none`` (the CI default).  Recording
    bypasses the skip so VCR can populate the cassette against the live
    OpenAI API.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    if record_mode != "none":
        return
    if not (CASSETTES / CASSETTE_FILE).exists():
        pytest.skip(
            f"No cassette at {CASSETTES / CASSETTE_FILE}; record once via "
            "`BRISTOL_ML_LLM_API_KEY=... uv run pytest "
            "tests/integration/llm/test_llm_extractor_cassette.py "
            "--record-mode=once` before CI runs."
        )


@pytest.fixture
def vcr_cassette_dir() -> str:
    """Point pytest-recording at this layer's cassette dir."""
    return str(CASSETTES)


@pytest.fixture
def default_cassette_name() -> str:
    """Share the single integration cassette across every VCR-marked test."""
    return CASSETTE_STEM


@pytest.fixture
def vcr_config(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Filter sensitive headers; replay-only on the CI default.

    ``authorization`` carries the bearer token; ``x-api-key`` is the
    OpenAI legacy header on some endpoints; ``cookie`` /
    ``set-cookie`` are belt-and-braces against any session cookie a
    proxy might inject.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    return {
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "record_mode": record_mode,
    }


@pytest.fixture
def _api_key_for_recording(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide a placeholder API key on replay so ``__init__`` does not raise.

    On replay (``--record-mode=none``), VCR intercepts the HTTP request
    and the API key is never sent over the wire — but ``LlmExtractor``
    still requires it at init time (plan §1 D5).  We set a placeholder
    when no real key is present.

    On record-mode (``once`` / ``new_episodes``), the operator must have
    a real API key in :envvar:`BRISTOL_ML_LLM_API_KEY` already; we leave
    that environment alone so the live API receives it.
    """
    record_mode = request.config.getoption("--record-mode", default="none")
    if record_mode == "none" and not os.environ.get("BRISTOL_ML_LLM_API_KEY"):
        monkeypatch.setenv("BRISTOL_ML_LLM_API_KEY", "sk-placeholder-for-replay")


def _build_config(tmp_path: Path) -> LlmExtractorConfig:
    """Build the canonical Stage 14 OpenAI config for the cassette test.

    The model name + prompt path here must match the values that were
    in effect when the cassette was recorded.  Changing either silently
    would mismatch the recorded request body and trigger VCR's
    "no match found" error on replay — so they are pinned in the test.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return LlmExtractorConfig(
        type="openai",
        model_name="gpt-4o-mini",
        api_key_env_var="BRISTOL_ML_LLM_API_KEY",
        prompt_file=repo_root / "conf" / "llm" / "prompts" / "extract_v1.txt",
        request_timeout_seconds=30.0,
    )


def _sample_events() -> list[RemitEvent]:
    """Five strata-spanning events drawn from the gold set.

    One Nuclear, one Gas, one Coal, one Wind (with no message
    description — exercises the synthesise-on-NULL path, plan OQ-B),
    one Solar.  Events match gold-set entries M-A, M-B, M-C, M-H, M-I
    so the LLM has explicit context to extract from in four of five
    cases and the fifth probes the more-fragile path.
    """
    from datetime import UTC, datetime

    return [
        RemitEvent(
            mrid="M-A",
            revision_number=0,
            message_status="Active",
            published_at=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            effective_from=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            effective_to=datetime(2024, 1, 20, 0, 0, tzinfo=UTC),
            fuel_type="Nuclear",
            affected_mw=600.0,
            event_type="Outage",
            cause="Planned",
            message_description="Stub: planned nuclear outage for refuelling.",
        ),
        RemitEvent(
            mrid="M-B",
            revision_number=2,
            message_status="Active",
            published_at=datetime(2024, 2, 3, 12, 0, tzinfo=UTC),
            effective_from=datetime(2024, 2, 10, 0, 0, tzinfo=UTC),
            effective_to=datetime(2024, 2, 18, 0, 0, tzinfo=UTC),
            fuel_type="Gas",
            affected_mw=380.0,
            event_type="Outage",
            cause="Unplanned",
            message_description="Stub: derate revised slightly downward.",
        ),
        RemitEvent(
            mrid="M-C",
            revision_number=0,
            message_status="Active",
            published_at=datetime(2024, 3, 1, 8, 0, tzinfo=UTC),
            effective_from=datetime(2024, 3, 5, 0, 0, tzinfo=UTC),
            effective_to=datetime(2024, 3, 8, 0, 0, tzinfo=UTC),
            fuel_type="Coal",
            affected_mw=500.0,
            event_type="Outage",
            cause="Planned",
            message_description="Stub: coal unit outage — later withdrawn.",
        ),
        RemitEvent(
            mrid="M-H",
            revision_number=0,
            message_status="Active",
            published_at=datetime(2024, 5, 1, 10, 0, tzinfo=UTC),
            effective_from=datetime(2024, 5, 10, 0, 0, tzinfo=UTC),
            effective_to=datetime(2024, 5, 11, 0, 0, tzinfo=UTC),
            fuel_type=None,
            affected_mw=None,
            event_type=None,
            cause=None,
            message_description=None,
        ),
        RemitEvent(
            mrid="M-I",
            revision_number=0,
            message_status="Active",
            published_at=datetime(2024, 6, 1, 7, 30, tzinfo=UTC),
            effective_from=datetime(2024, 6, 7, 0, 0, tzinfo=UTC),
            effective_to=datetime(2024, 6, 7, 18, 0, tzinfo=UTC),
            fuel_type="Solar",
            affected_mw=120.0,
            event_type="Restriction",
            cause="Planned",
            message_description="Stub: solar farm reduced output for grid maintenance.",
        ),
    ]


# --------------------------------------------------------------------------- #
# T4 plan tests
# --------------------------------------------------------------------------- #


@pytest.mark.vcr
@pytest.mark.usefixtures("_cassette_present_or_skip", "_api_key_for_recording")
def test_llm_extractor_against_cassette(tmp_path: Path) -> None:
    """T4 / AC-3 / AC-5: live extractor against the recorded cassette returns
    structurally valid :class:`ExtractionResult` rows for every event.

    Asserts (a) the extractor returns one row per input event, (b) every
    row is an :class:`ExtractionResult` (Pydantic-validated), and (c)
    every row carries provenance — non-empty ``prompt_hash`` (12 hex
    chars) and ``model_id`` matching the configured model.

    Per plan §1 D14, ``confidence`` is in the closed unit interval; we
    assert that as well — Pydantic enforces it at construction so a
    failure here would surface as a ValidationError caught by the
    extractor and return the documented default with confidence 0.0
    (still in [0, 1]).

    The test does **not** assert specific extracted values — those are
    LLM-dependent and would tighten the cassette to one model version.
    The :func:`bristol_ml.llm.evaluate` harness (T5) is the place for
    accuracy assertions against the gold set.
    """
    config = _build_config(tmp_path)
    extractor = LlmExtractor(config)

    events = _sample_events()
    results = extractor.extract_batch(events)

    assert len(results) == len(events), (
        f"extract_batch must preserve cardinality; got {len(results)} for "
        f"{len(events)} input events."
    )
    for event, result in zip(events, results, strict=True):
        assert isinstance(result, ExtractionResult), (
            f"Each result must be an ExtractionResult; got {type(result).__name__} "
            f"for mrid={event.mrid!r}."
        )
        assert result.prompt_hash is not None and len(result.prompt_hash) == 12, (
            f"Provenance: prompt_hash must be the 12-char SHA-256 prefix; got "
            f"{result.prompt_hash!r} for mrid={event.mrid!r}."
        )
        assert result.model_id == config.model_name, (
            f"Provenance: model_id must equal config.model_name="
            f"{config.model_name!r}; got {result.model_id!r} for mrid={event.mrid!r}."
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"D14: confidence must be in [0, 1]; got {result.confidence} for mrid={event.mrid!r}."
        )
