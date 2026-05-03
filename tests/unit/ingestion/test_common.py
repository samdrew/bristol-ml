"""Tests for the ingestion-layer shared helpers in ``bristol_ml.ingestion._common``.

Stage 2 introduced the helpers; this file pins the recently-added
behaviour added in response to a live 429 from the Open-Meteo archive
endpoint:

- :func:`_parse_retry_after` correctly handles the two RFC 7231 forms
  (delta-seconds and HTTP-date), unparseable input, and missing input.
- :func:`_retrying_get` honours a server-advertised ``Retry-After``
  header in preference to the configured exponential backoff.

The tests are minimal and self-contained — they construct an
``httpx.Client`` against an in-memory :class:`httpx.MockTransport` so
no network is touched.
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
)

# ---------------------------------------------------------------------------
# Helper: a minimal RetryConfig — the structural Protocol in _common.py only
# reads four attributes, so a frozen dataclass is sufficient.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubRetryConfig:
    """Satisfies the ``RetryConfig`` Protocol with caller-controlled values."""

    max_attempts: int = 3
    backoff_base_seconds: float = 0.001  # tiny so tests are fast
    backoff_cap_seconds: float = 5.0
    request_timeout_seconds: float = 1.0


# ---------------------------------------------------------------------------
# _parse_retry_after — the four RFC 7231 cases
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_missing_or_empty_returns_zero(self) -> None:
        assert _parse_retry_after(None) == 0.0
        assert _parse_retry_after("") == 0.0
        assert _parse_retry_after("   ") == 0.0

    def test_delta_seconds_integer(self) -> None:
        assert _parse_retry_after("60") == 60.0
        assert _parse_retry_after("0") == 0.0
        # A surrounding whitespace pad is stripped.
        assert _parse_retry_after("  120  ") == 120.0

    def test_delta_seconds_negative_clamps_to_zero(self) -> None:
        # The header is documented as non-negative, but a buggy server
        # might emit a negative value.  Clamp rather than wait backwards.
        assert _parse_retry_after("-30") == 0.0

    def test_http_date_returns_positive_delta(self) -> None:
        future = datetime.now(tz=UTC) + timedelta(seconds=45)
        header = format_datetime(future, usegmt=True)
        delta = _parse_retry_after(header)
        # Allow a small tolerance for clock drift between header build
        # and parse.
        assert 40 <= delta <= 50, f"Expected ~45s; got {delta:.2f}s"

    def test_http_date_in_past_clamps_to_zero(self) -> None:
        past = datetime.now(tz=UTC) - timedelta(hours=1)
        header = format_datetime(past, usegmt=True)
        assert _parse_retry_after(header) == 0.0

    def test_unparseable_returns_zero(self) -> None:
        # Neither a number nor an RFC 7231 date.
        assert _parse_retry_after("soon") == 0.0
        assert _parse_retry_after("not-a-date-or-int") == 0.0


# ---------------------------------------------------------------------------
# _retrying_get — Retry-After contract integration
# ---------------------------------------------------------------------------


def test_retrying_get_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 with ``Retry-After: 2`` should sleep at least 2s before retrying.

    Uses an in-memory MockTransport that returns 429 once then 200 OK,
    and patches ``time.sleep`` so the test does not actually wait — but
    asserts that the sleep duration tenacity requested was at least the
    server's advertised value.
    """
    sleep_calls: list[float] = []

    def _record_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    monkeypatch.setattr("time.sleep", _record_sleep)

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
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
    """A 429 without a ``Retry-After`` header uses the exponential-backoff path.

    With ``backoff_base_seconds=0.001`` the first wait is ~1 ms; the test
    asserts that the recorded sleep is below 1 second (i.e. did not pick
    up a phantom Retry-After value).
    """
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            # No Retry-After header.
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=3, backoff_cap_seconds=5.0)

    _retrying_get(client, "https://example.test/y", {}, config)

    # The recorded sleep should be the exponential-backoff first step
    # (~1 ms with the stub config), not several seconds.
    assert sleep_calls and max(sleep_calls) < 1.0, (
        f"Without Retry-After we should fall back to exponential backoff "
        f"(<1s with stub config); got sleeps {sleep_calls!r}"
    )


def test_retrying_get_clamps_retry_after_at_backoff_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving server cannot lock the client up indefinitely.

    Server advertises ``Retry-After: 9999`` but the configured cap is 5s;
    the recorded sleep must not exceed the cap.
    """
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    state = {"calls": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "9999"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    config = _StubRetryConfig(max_attempts=3, backoff_cap_seconds=5.0)

    _retrying_get(client, "https://example.test/z", {}, config)

    assert sleep_calls and max(sleep_calls) <= 5.0, (
        f"Sleep must clamp at backoff_cap_seconds (5s); got {sleep_calls!r}"
    )


def test_retryable_status_error_carries_retry_after_attribute() -> None:
    """The exception's public attribute round-trips for downstream consumers."""
    exc = _RetryableStatusError("boom", retry_after_seconds=42.0)
    assert exc.retry_after_seconds == 42.0
    assert "boom" in str(exc)

    default = _RetryableStatusError("plain")
    assert default.retry_after_seconds == 0.0


# ---------------------------------------------------------------------------
# WeatherIngestionConfig — defaults bumped in this fix
# ---------------------------------------------------------------------------


def test_weather_config_defaults_match_polite_envelope(tmp_path: Path) -> None:
    """The Stage 2 defaults must be polite to Open-Meteo's per-IP token bucket.

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
        "Expected backoff_cap_seconds bumped to 60s so the exponential ceiling "
        "matches Open-Meteo's typical Retry-After window (30-60s)."
    )
    assert cfg.min_inter_request_seconds == 1.0, (
        "Expected min_inter_request_seconds bumped to 1.0s so multi-station "
        "refreshes do not burst above Open-Meteo's per-minute token bucket."
    )
