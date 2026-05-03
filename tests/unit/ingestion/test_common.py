"""Tests for the ingestion-layer shared helpers in ``bristol_ml.ingestion._common``.

Focused on the recently-added diagnostic + retry behaviour following a
live HTTP 429 from the Open-Meteo archive endpoint:

- ``_RetryableStatusError`` carries the status code, the parsed
  ``Retry-After`` value, and a snippet of the response body.
- ``_parse_retry_after`` correctly handles the two RFC 7231 forms
  (delta-seconds and HTTP-date), unparseable input, and missing input.
- ``_safe_body_snippet`` truncates and collapses whitespace.
- ``_retrying_get`` honours a server-advertised ``Retry-After`` header
  in preference to the configured exponential backoff, surfaces the
  body in the final raise, and emits a WARNING log per retry attempt.

Tests use an in-memory :class:`httpx.MockTransport` — no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from bristol_ml.ingestion._common import (
    _parse_retry_after,
    _RetryableStatusError,
    _retrying_get,
    _safe_body_snippet,
)


@dataclass(frozen=True)
class _StubRetryConfig:
    """Satisfies the ``RetryConfig`` Protocol with caller-controlled values."""

    max_attempts: int = 3
    backoff_base_seconds: float = 0.001  # tiny so tests run fast
    backoff_cap_seconds: float = 5.0
    request_timeout_seconds: float = 1.0


# ---------------------------------------------------------------------------
# _parse_retry_after — RFC 7231 cases
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_missing_or_empty_returns_zero(self) -> None:
        assert _parse_retry_after(None) == 0.0
        assert _parse_retry_after("") == 0.0
        assert _parse_retry_after("   ") == 0.0

    def test_delta_seconds_integer(self) -> None:
        assert _parse_retry_after("60") == 60.0
        assert _parse_retry_after("0") == 0.0
        assert _parse_retry_after("  120  ") == 120.0

    def test_delta_seconds_negative_clamps_to_zero(self) -> None:
        assert _parse_retry_after("-30") == 0.0

    def test_http_date_returns_positive_delta(self) -> None:
        future = datetime.now(tz=UTC) + timedelta(seconds=45)
        header = format_datetime(future, usegmt=True)
        delta = _parse_retry_after(header)
        assert 40 <= delta <= 50, f"Expected ~45s; got {delta:.2f}s"

    def test_http_date_in_past_clamps_to_zero(self) -> None:
        past = datetime.now(tz=UTC) - timedelta(hours=1)
        header = format_datetime(past, usegmt=True)
        assert _parse_retry_after(header) == 0.0

    def test_unparseable_returns_zero(self) -> None:
        assert _parse_retry_after("soon") == 0.0
        assert _parse_retry_after("not-a-date-or-int") == 0.0


# ---------------------------------------------------------------------------
# _safe_body_snippet — truncation + whitespace collapse
# ---------------------------------------------------------------------------


def test_safe_body_snippet_collapses_whitespace_and_truncates() -> None:
    """Multi-line bodies collapse to a single line; oversize bodies truncate."""
    response = httpx.Response(429, text="line one\n\nline two   with   spaces")
    snippet = _safe_body_snippet(response, limit=500)
    assert snippet == "line one line two with spaces"

    big = "x" * 2000
    response = httpx.Response(429, text=big)
    snippet = _safe_body_snippet(response, limit=500)
    assert len(snippet) <= 500
    # Truncation marker present at the tail.
    assert snippet.endswith("…")


def test_safe_body_snippet_handles_unread_body_gracefully() -> None:
    """If the body cannot be read (e.g. binary streamed), return empty string."""
    # An httpx.Response with no content has empty body — easy case.
    empty = httpx.Response(429, content=b"")
    assert _safe_body_snippet(empty) == ""


# ---------------------------------------------------------------------------
# _RetryableStatusError — public attribute surface
# ---------------------------------------------------------------------------


def test_retryable_status_error_carries_diagnostic_attributes() -> None:
    exc = _RetryableStatusError(
        "boom",
        status_code=429,
        retry_after_seconds=42.0,
        body_snippet="rate limit hit",
    )
    assert "boom" in str(exc)
    assert exc.status_code == 429
    assert exc.retry_after_seconds == 42.0
    assert exc.body_snippet == "rate limit hit"

    # Defaults preserve back-compat for callers that only care about the message.
    default = _RetryableStatusError("plain", status_code=503)
    assert default.retry_after_seconds == 0.0
    assert default.body_snippet == ""


# ---------------------------------------------------------------------------
# _retrying_get — Retry-After contract + body propagation + before_sleep log
# ---------------------------------------------------------------------------


def test_retrying_get_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 with ``Retry-After: 2`` should sleep at least 2s before retrying."""
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="rate limit")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=3, backoff_cap_seconds=10.0)

    response = _retrying_get(client, "https://example.test/x", {}, config)

    assert response.status_code == 200
    assert state["calls"] == 2, "Expected one retry after the 429."
    assert any(s >= 2.0 for s in sleep_calls), (
        f"Expected at least one sleep >= 2s (Retry-After value); got {sleep_calls!r}"
    )


def test_retrying_get_falls_back_to_exponential_when_retry_after_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 without ``Retry-After`` falls back to exponential backoff."""
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, text="no retry-after here")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=3, backoff_cap_seconds=5.0)

    _retrying_get(client, "https://example.test/y", {}, config)

    assert sleep_calls and max(sleep_calls) < 1.0, (
        f"Without Retry-After we should fall back to exponential backoff "
        f"(<1s with stub config); got sleeps {sleep_calls!r}"
    )


def test_retrying_get_clamps_retry_after_at_backoff_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving server cannot lock the client up indefinitely."""
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "9999"}, text="locked out")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=3, backoff_cap_seconds=5.0)

    _retrying_get(client, "https://example.test/z", {}, config)

    assert sleep_calls and max(sleep_calls) <= 5.0, (
        f"Sleep must clamp at backoff_cap_seconds (5s); got {sleep_calls!r}"
    )


def test_retrying_get_preserves_response_body_in_final_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all retries exhaust, the raised exception names the response body.

    This is the load-bearing diagnostic added in this fix: previously
    a 429 left the operator guessing which limit was hit because the
    response body was discarded.
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)

    body = "Minutely API request limit exceeded. Please try again in one minute."

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text=body)

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=2)

    with pytest.raises(_RetryableStatusError) as exc_info:
        _retrying_get(client, "https://example.test/q", {}, config)

    # The exception attributes carry the diagnostic; the str carries it too
    # so a traceback paste from the operator is self-contained.
    assert exc_info.value.status_code == 429
    assert "Minutely API request limit" in exc_info.value.body_snippet
    assert "Minutely API request limit" in str(exc_info.value)


def test_retrying_get_emits_warning_log_per_retry_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every retry attempt must emit a WARNING so the operator sees the loop running.

    Pre-fix, tenacity slept silently and the operator saw a long wall-clock gap
    with no output before the final raise — making it indistinguishable from a
    hang.  This test pins the visible-retry contract.
    """
    import logging

    # Bridge loguru -> caplog so pytest's caplog sees loguru records.
    from loguru import logger as loguru_logger

    class _PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(
        _PropagateHandler(level=logging.WARNING), level="WARNING", format="{message}"
    )
    try:
        monkeypatch.setattr("time.sleep", lambda _s: None)

        state = {"calls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] < 3:
                return httpx.Response(429, text="rate limit")
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(_handler)
        client = httpx.Client(transport=transport)
        config = _StubRetryConfig(max_attempts=3)

        with caplog.at_level(logging.WARNING):
            response = _retrying_get(client, "https://example.test/r", {}, config)
        assert response.status_code == 200

        # We expect at least one "Ingestion retry" WARNING per failed attempt
        # (2 failures → 2 retry log lines).
        retry_warnings = [r for r in caplog.records if "Ingestion retry" in r.getMessage()]
        assert len(retry_warnings) >= 2, (
            f"Expected at least 2 'Ingestion retry' WARNING log lines (one per failed "
            f"attempt); got {[r.getMessage() for r in caplog.records]!r}"
        )
    finally:
        loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# WeatherIngestionConfig — defaults bumped in this fix
# ---------------------------------------------------------------------------


def test_weather_config_defaults_are_polite_for_open_meteo(tmp_path: Path) -> None:
    """The Stage 2 defaults must respect Open-Meteo's per-IP token bucket.

    Pins the values bumped in this fix so a future "tighten the
    defaults" regression surfaces here rather than in a live 429.
    """
    from datetime import date

    from conf._schemas import WeatherIngestionConfig, WeatherStation

    cfg = WeatherIngestionConfig(
        stations=[WeatherStation(name="x", latitude=51.0, longitude=0.0, weight=1.0)],
        start_date=date(2018, 1, 1),
        cache_dir=tmp_path,
    )
    assert cfg.max_attempts == 5, "Expected max_attempts bumped to 5 (was 3)."
    assert cfg.backoff_cap_seconds == 60.0, (
        "Expected backoff_cap_seconds bumped to 60s (was 10s) so the "
        "exponential ceiling matches Open-Meteo's typical Retry-After window."
    )
    assert cfg.min_inter_request_seconds == 5.0, (
        "Expected min_inter_request_seconds bumped to 5.0s (was 0.25s) so "
        "multi-station refreshes do not exhaust Open-Meteo's per-minute "
        "token bucket on the 7th-of-10 station."
    )
