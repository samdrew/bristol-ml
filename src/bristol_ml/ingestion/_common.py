"""Shared helpers for the ingestion layer.

Extracted from ``bristol_ml.ingestion.neso`` at Stage 2 when the second
caller (``weather.py``) landed and the layer architecture's two-caller
trigger for ``_common.py`` was met.

Everything here is module-private to the layer: the ``_`` prefix convention
continues, and consumers reach the helpers via their own ingester
module (``bristol_ml.ingestion.neso`` / ``bristol_ml.ingestion.weather``).

What is lifted:

- ``CachePolicy`` — the three-valued cache enum (public name, private module).
- ``CacheMissingError`` — raised by ``OFFLINE`` when the cache is absent.
- ``_atomic_write(table, path)`` — tmp file + ``os.replace`` for partial-write
  safety.
- ``_cache_path(config)`` — resolve the absolute cache path; ensures the parent
  directory exists.
- ``_respect_rate_limit(last, gap)`` — sleep so successive calls are separated
  by at least ``gap`` seconds.
- ``_retrying_get(client, url, params, config)`` — GET with tenacity retry on
  transient errors only.
- ``_RetryableStatusError`` — internal signal class used by the retry predicate.

What is **not** lifted (stays per-ingester): settlement-period arithmetic,
schema assertion, date parsing — these are NESO-specific and have no
analogue in the weather module.

The retry / rate-limit / cache-path helpers are generic across ingesters via
structural ``Protocol`` types — any config whose public read-only attributes
cover the named knobs is accepted. This keeps ``_common.py`` decoupled from
the ``NesoIngestionConfig`` / ``WeatherIngestionConfig`` Pydantic models.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Cache policy + error
# ---------------------------------------------------------------------------


class CachePolicy(StrEnum):
    """How ``fetch`` treats the local cache.

    - ``AUTO``: use cache if present, fetch if not. Notebook-friendly default.
    - ``REFRESH``: always fetch; overwrite cache atomically.
    - ``OFFLINE``: never touch the network; raise ``CacheMissingError`` if the
      cache file is absent. The CI-safe choice.
    """

    AUTO = "auto"
    REFRESH = "refresh"
    OFFLINE = "offline"


class CacheMissingError(FileNotFoundError):
    """Raised when ``CachePolicy.OFFLINE`` is requested without a cache file."""


# ---------------------------------------------------------------------------
# Structural Protocols for config knobs
# ---------------------------------------------------------------------------


@runtime_checkable
class RetryConfig(Protocol):
    """The retry-knob surface that ``_retrying_get`` reads from a config.

    Any object exposing these four attributes is accepted, so Pydantic models
    for different sources (``NesoIngestionConfig``, ``WeatherIngestionConfig``)
    satisfy the contract structurally without inheriting a common base.
    """

    max_attempts: int
    backoff_base_seconds: float
    backoff_cap_seconds: float
    request_timeout_seconds: float


@runtime_checkable
class RateLimitConfig(Protocol):
    """The rate-limit-knob surface that ``_respect_rate_limit`` expects."""

    min_inter_request_seconds: float


@runtime_checkable
class CachePathConfig(Protocol):
    """The cache-path-knob surface that ``_cache_path`` expects."""

    cache_dir: Path
    cache_filename: str


# ---------------------------------------------------------------------------
# Cache path
# ---------------------------------------------------------------------------


def _cache_path(config: CachePathConfig) -> Path:
    """Resolve the absolute cache path and ensure the parent directory exists."""
    cache_dir = Path(config.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / config.cache_filename


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def _respect_rate_limit(last_request_at: float | None, min_gap_seconds: float) -> float:
    """Sleep so that calls are separated by at least ``min_gap_seconds``.

    Returns the wall-clock timestamp (from ``time.monotonic``) at which the
    caller is now free to issue a request. Takes a ``None`` previous stamp on
    the first call and skips the sleep.
    """
    now = time.monotonic()
    if last_request_at is not None and min_gap_seconds > 0:
        elapsed = now - last_request_at
        remaining = min_gap_seconds - elapsed
        if remaining > 0:
            logger.debug("Ingestion rate-limit sleep: {:.2f}s", remaining)
            time.sleep(remaining)
            now = time.monotonic()
    return now


# ---------------------------------------------------------------------------
# Retry GET
# ---------------------------------------------------------------------------


class _RetryableStatusError(Exception):
    """Internal signal that a 5xx or 429 response should be retried.

    Carries an optional ``retry_after_seconds`` value extracted from the
    ``Retry-After`` HTTP header (RFC 7231 §7.1.3).  When present and
    positive, ``_retrying_get`` honours it as the next sleep duration in
    preference to the configured exponential backoff — servers that
    explicitly advertise a recovery interval should be respected over
    our blind exponential.
    """

    def __init__(self, message: str, *, retry_after_seconds: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _parse_retry_after(value: str | None) -> float:
    """Parse the ``Retry-After`` header value into a non-negative seconds delay.

    RFC 7231 §7.1.3 allows two forms:

    - **delta-seconds** — a non-negative decimal integer (e.g. ``"60"``).
    - **HTTP-date** — an absolute timestamp (e.g.
      ``"Wed, 21 Oct 2026 07:28:00 GMT"``).  The seconds returned is the
      delta from "now" (UTC) to that timestamp.

    Returns ``0.0`` when the header is missing, blank, or unparseable —
    callers fall back to the configured exponential backoff in that case.
    Negative deltas (an HTTP-date already in the past) clamp to ``0.0``.
    """
    if not value:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    # Form 1: integer seconds.
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    # Form 2: RFC 7231 HTTP-date.
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delta = (target - datetime.now(tz=UTC)).total_seconds()
    return max(0.0, delta)


def _retrying_get(
    client: httpx.Client,
    url: str,
    params: dict[str, object],
    config: RetryConfig,
) -> httpx.Response:
    """GET with tenacity retry on transient errors only.

    Retries on ``httpx.ConnectError``, ``httpx.ReadTimeout``, and HTTP 5xx/429.
    Never retries non-429 4xx — those are caller errors and should fail loudly.
    On final failure the raised error names the URL and attempt count.

    The wait between retries is ``max(server-advertised Retry-After,
    exponential-backoff)``: a server that responds 429 with a
    ``Retry-After`` header gets its requested cooldown honoured rather
    than overridden by our blind exponential.  This is the documented
    Open-Meteo recovery contract and the expected RFC 7231 behaviour for
    polite clients.
    """
    base_wait = wait_exponential(
        multiplier=config.backoff_base_seconds,
        max=config.backoff_cap_seconds,
    )

    def _wait_with_retry_after(retry_state):  # type: ignore[no-untyped-def]
        # tenacity passes a ``RetryCallState``; the failed attempt's
        # exception lives on ``outcome``.  When the failure carries a
        # positive ``retry_after_seconds`` we honour it (clamped to the
        # configured cap so a misbehaving server cannot lock us up
        # indefinitely); otherwise we fall back to exponential backoff.
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        backoff = base_wait(retry_state)
        if isinstance(exc, _RetryableStatusError) and exc.retry_after_seconds > 0:
            advertised = min(exc.retry_after_seconds, config.backoff_cap_seconds)
            chosen = max(advertised, backoff)
            logger.info(
                "Honouring Retry-After: server requested {:.1f}s; sleeping {:.1f}s "
                "before retry attempt {}.",
                exc.retry_after_seconds,
                chosen,
                retry_state.attempt_number + 1,
            )
            return chosen
        return backoff

    @retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=_wait_with_retry_after,
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ReadTimeout, _RetryableStatusError)
        ),
        reraise=True,
    )
    def _do_get() -> httpx.Response:
        response = client.get(url, params=params)
        if response.status_code >= 500 or response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise _RetryableStatusError(
                f"Upstream returned {response.status_code} for {response.request.url}",
                retry_after_seconds=retry_after,
            )
        response.raise_for_status()
        return response

    try:
        return _do_get()
    except RetryError as exc:  # pragma: no cover — reraise=True replaces this path
        raise RuntimeError(f"GET {url} failed after {config.max_attempts} attempts: {exc}") from exc


# ---------------------------------------------------------------------------
# Atomic parquet write
# ---------------------------------------------------------------------------


def _atomic_write(table: pa.Table, path: Path) -> None:
    """Write ``table`` to ``path`` via a tmp file + ``os.replace``.

    ``os.replace`` is the portable Python atomic-rename primitive (atomic on
    POSIX and NTFS). PyArrow has no built-in atomic mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


__all__ = [
    "CacheMissingError",
    "CachePathConfig",
    "CachePolicy",
    "RateLimitConfig",
    "RetryConfig",
    "_RetryableStatusError",
    "_atomic_write",
    "_cache_path",
    "_respect_rate_limit",
    "_retrying_get",
]
