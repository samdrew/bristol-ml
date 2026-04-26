"""Elexon REMIT ingestion (Stage 13).

Public interface (per the ingestion-layer contract):

- ``fetch(config, *, cache=CachePolicy.AUTO) -> Path`` — fetch REMIT
  messages from the Elexon Insights ``/datasets/REMIT/stream`` endpoint
  or reuse a local cache; returns the path to the consolidated parquet
  file.  *Stubbed at T2; wired in T3 (stub path) and T4 (live path).*
- ``load(path) -> pd.DataFrame`` — cheap, pure read of the cached
  parquet with a schema assertion.  Returns a frame with all four
  temporal columns typed as UTC-aware ``datetime64[us, UTC]``.
  *Stubbed at T2; wired in T3.*
- ``as_of(df, t) -> pd.DataFrame`` — bi-temporal "what did the market
  know at time T?" query.  Filters to ``published_at <= t``, groups by
  ``mrid`` taking the max ``revision_number``, drops withdrawn rows.
  Returns a copy.  This is the new public primitive Stage 13 introduces.
- ``OUTPUT_SCHEMA`` — the pyarrow schema pinning the parquet layout.
- ``MESSAGE_STATUSES`` — the closed set of valid ``message_status``
  values; an unknown status emits a WARNING log line at fetch time
  rather than failing.
- ``FUEL_TYPES`` — the canonical Elexon fuel-type vocabulary; used by
  the stub fixture and the notebook.
- ``CachePolicy`` / ``CacheMissingError`` — re-exported from ``_common``.

Source: ``https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT/stream``,
the Elexon Insights API (no authentication, public endpoint).  See
`docs/lld/research/13-remit-ingestion-domain.md` for the API surface.

Bi-temporal storage shape (Stage 13 D8):

- ``published_at`` — transaction-time axis: when the participant
  disclosed the message to the market.
- ``effective_from`` / ``effective_to`` — valid-time axis: the event
  window.  ``effective_to`` is nullable; ``None`` means "open-ended".
- ``retrieved_at_utc`` — project-axis provenance: when *this run*
  fetched the row.

Storage is append-only on the ``(mrid, revision_number)`` key — every
revision is preserved so historical as-of queries are correct.
A "latest as-of T" view is a query (:func:`as_of`), not a storage shape.

Run standalone (principle §2.1.1)::

    python -m bristol_ml.ingestion.remit [--help]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from bristol_ml.ingestion._common import (
    CacheMissingError,
    CachePolicy,
    _atomic_write,
    _cache_path,
    _respect_rate_limit,
    _retrying_get,
)
from conf._schemas import RemitIngestionConfig

# Env-var trigger for the stub fetch path (NFR-2 / DESIGN §2.1.3).  The
# CI default is ``BRISTOL_ML_REMIT_STUB=1``; setting it to anything other
# than ``"1"`` (or leaving it unset) takes the live fetch path.
_STUB_ENV_VAR: Final[str] = "BRISTOL_ML_REMIT_STUB"

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "FUEL_TYPES",
    "MESSAGE_STATUSES",
    "OUTPUT_SCHEMA",
    "CacheMissingError",
    "CachePolicy",
    "RemitParseError",
    "as_of",
    "fetch",
    "load",
]


class RemitParseError(ValueError):
    """A REMIT record (or stream payload) could not be parsed.

    Raised when the live Insights API returns a payload whose shape does
    not match the documented schema — a missing required field on a
    record, a non-array top-level payload, a record that is not a JSON
    object.  Inherits from :class:`ValueError` so callers that already
    catch ``ValueError`` for parse-level issues continue to work; the
    typed subclass lets new callers distinguish parse errors from other
    value errors (e.g. the naive-timestamp guard on :func:`as_of`).

    Carries enough context in ``str(exc)`` to identify the offending
    record without having to re-fetch — the field name, the available
    keys, or the offending JSON shape, depending on the case.
    """


# Canonical Elexon REMIT message statuses (per domain research §R2).  An
# unknown status at fetch time emits a WARNING log line; the row is still
# persisted so a downstream consumer can investigate.
MESSAGE_STATUSES: Final[tuple[str, ...]] = (
    "Active",
    "Inactive",
    "Cancelled",
    "Withdrawn",
    "Dismissed",
)


# Canonical Elexon REMIT fuel-type vocabulary (per the live
# ``/reference/remit/fueltypes/all`` endpoint, observed 2026-04 in the
# domain research).  Used by the stub fixture and the notebook colour
# layer; an unknown value at fetch time emits a WARNING but is preserved.
FUEL_TYPES: Final[tuple[str, ...]] = (
    "Coal",
    "Gas",
    "Nuclear",
    "Oil",
    "Wind",
    "Solar",
    "Hydro",
    "Pumped Storage",
    "Biomass",
    "Other",
    "Interconnector",
    "Battery",
)


# Storage schema (Stage 13 D8 + D10 — four UTC-aware timestamp axes,
# ``effective_to`` nullable for open-ended events).  Field-name casing
# snake-cases the Elexon camelCase.
OUTPUT_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        # Identifier axis
        pa.field("mrid", pa.string(), nullable=False),
        pa.field("revision_number", pa.int32(), nullable=False),
        pa.field("message_type", pa.string(), nullable=False),
        pa.field("message_status", pa.string(), nullable=False),
        # Bi-temporal axis (D8 / D10)
        pa.field("published_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("effective_from", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("effective_to", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("retrieved_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        # Asset axis (raw — no master-data normalisation per intent OOS)
        pa.field("affected_unit", pa.string(), nullable=True),
        pa.field("asset_id", pa.string(), nullable=True),
        pa.field("fuel_type", pa.string(), nullable=True),
        # Capacity axis
        pa.field("affected_mw", pa.float64(), nullable=True),
        pa.field("normal_capacity_mw", pa.float64(), nullable=True),
        # Event metadata
        pa.field("event_type", pa.string(), nullable=True),
        pa.field("cause", pa.string(), nullable=True),
        # Free text — Stage 14 will read this
        pa.field("message_description", pa.string(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Public functions — fetch / load are stubbed at T2; as_of is fully wired.
# ---------------------------------------------------------------------------


def fetch(
    config: RemitIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch REMIT messages from the Insights API or reuse a local cache.

    Behaviour matches the layer contract:

    - ``AUTO``: if the cache file exists, return it without touching the
      network.  Otherwise fetch and persist.
    - ``REFRESH``: fetch and overwrite the cache atomically.
    - ``OFFLINE``: return the cache if present; raise
      :class:`CacheMissingError` if not.

    Stub-first: when ``BRISTOL_ML_REMIT_STUB=1`` is set in the
    environment, the "fetch" step builds a small in-memory record set
    (:func:`_stub_records`) instead of calling the live Insights API.
    The on-disk schema is identical, so notebook + CI exercise the same
    `load` / `as_of` code paths against deterministic fixture data
    (NFR-2 / DESIGN §2.1.3).  The live path lands at T4.
    """
    cache_path = _cache_path(config)

    if cache is CachePolicy.OFFLINE:
        if not cache_path.exists():
            raise CacheMissingError(
                f"REMIT cache not found at {cache_path}. "
                "Re-run with CachePolicy.AUTO (or REFRESH) to populate it."
            )
        logger.info("REMIT cache hit (offline) at {}", cache_path)
        return cache_path

    if cache is CachePolicy.AUTO and cache_path.exists():
        logger.info("REMIT cache hit (auto) at {}", cache_path)
        return cache_path

    use_stub = os.environ.get(_STUB_ENV_VAR) == "1"
    logger.info(
        "REMIT fetch: {}{} → {} (policy={})",
        config.base_url,
        config.endpoint_path,
        cache_path,
        cache.value,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0)

    if use_stub:
        logger.info("REMIT fetch via stub fixture ({}=1)", _STUB_ENV_VAR)
        records = _stub_records()
    else:
        with httpx.Client(timeout=config.request_timeout_seconds) as client:
            _respect_rate_limit(None, config.min_inter_request_seconds)
            records = _live_fetch(config, client=client)

    table = _to_arrow(records, retrieved_at=retrieved_at)
    _atomic_write(table, cache_path)
    logger.info(
        "REMIT cache written: {} row(s) covering {} mRID(s) → {}",
        table.num_rows,
        len({r["mrid"] for r in records}),
        cache_path,
    )
    return cache_path


def load(path: Path) -> pd.DataFrame:
    """Read the cached REMIT parquet; assert the schema; return a frame.

    The returned dataframe carries the canonical schema declared on
    :data:`OUTPUT_SCHEMA`.  All four temporal columns
    (``published_at``, ``effective_from``, ``effective_to``,
    ``retrieved_at_utc``) round-trip with ``tz=UTC``; ``effective_to``
    is ``pd.NaT`` for open-ended events (D10).

    A schema mismatch raises :class:`ValueError` naming the offending
    column — the on-disk type or the missing-column case.
    """
    table = pq.read_table(path)
    actual = table.schema
    for field in OUTPUT_SCHEMA:
        if field.name not in actual.names:
            raise ValueError(
                f"Cached REMIT parquet at {path} is missing required column {field.name!r}"
            )
        actual_field = actual.field(field.name)
        if actual_field.type != field.type:
            raise ValueError(
                f"Column {field.name!r} in {path} has type {actual_field.type}; "
                f"expected {field.type}"
            )
    return table.to_pandas(types_mapper=None)


def as_of(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return the active-state frame as known to the market at time ``t``.

    Implements the central question from ``docs/intent/13-remit-ingestion.md``:
    "what REMIT events were known to the market at time T?".  This is a
    **transaction-time** query: it filters by ``published_at`` (when the
    participant disclosed the message), not by the event window.

    Algorithm:

    1. Filter ``df`` to rows with ``published_at <= t`` — only messages
       the market had seen by time ``t``.
    2. Within that filter, group by ``mrid``; keep the row with the
       maximum ``revision_number`` — the latest revision known by ``t``.
    3. Drop rows whose ``message_status == "Withdrawn"`` — withdrawn
       messages were retracted by their publisher and should not appear
       in the active state.

    The ``effective_from`` / ``effective_to`` columns are *not* part of
    this filter — that is a valid-time join, separate from the
    transaction-time as-of.  Callers who want "active at ``t``"
    (valid-time) chain a second filter::

        df_known = as_of(df, t)
        df_active = df_known[
            (df_known.effective_from <= t)
            & (df_known.effective_to.isna() | (df_known.effective_to > t))
        ]

    The two-step decomposition is the standard bi-temporal pattern (see
    ``docs/lld/research/13-remit-ingestion-domain.md`` §R4) — keeping
    ``as_of`` strictly transaction-time means the function has one job
    and the caller composes for valid-time when wanted.

    Args:
        df: A REMIT frame matching :data:`OUTPUT_SCHEMA` (typically the
            return value of :func:`load`).
        t: A timezone-aware ``pd.Timestamp``.  A naive timestamp raises
            ``ValueError`` — REMIT bi-temporal queries are nonsense
            without a timezone reference.

    Returns:
        A new ``pd.DataFrame`` (a copy; ``df`` is not mutated) carrying
        one row per ``mrid`` that was active at time ``t`` — i.e. its
        latest pre-``t`` revision was not withdrawn.  Columns and dtypes
        match :data:`OUTPUT_SCHEMA`; the row index is reset to a fresh
        ``RangeIndex`` so callers do not inherit stale positional state.

    Raises:
        ValueError: if ``t`` is naive (i.e. ``t.tzinfo is None``).
    """
    if t.tzinfo is None:
        raise ValueError(
            f"as_of requires a timezone-aware timestamp; got naive {t!r}. "
            "REMIT bi-temporal queries are nonsense without a timezone "
            "reference — pass e.g. pd.Timestamp(..., tz='UTC')."
        )

    # Step 1: transaction-time filter.  Use a strictly-typed comparison
    # so a frame with a tz-naive published_at column raises rather than
    # silently producing wrong answers.
    known = df[df["published_at"] <= t]

    if known.empty:
        # Preserve schema by returning an empty copy with reset index.
        return known.reset_index(drop=True).copy()

    # Step 2: keep the latest revision per mrid that was known by t.
    # idxmax over revision_number returns the positional index of the
    # winning row in each group; using .loc[...] preserves all columns.
    latest_idx = known.groupby("mrid", sort=False)["revision_number"].idxmax()
    latest = known.loc[latest_idx]

    # Step 3: drop withdrawn rows.  The check is on the latest revision
    # only — a previously-active mRID withdrawn before ``t`` is correctly
    # excluded; a previously-withdrawn mRID re-issued before ``t`` (rare
    # but possible) is correctly included via its newer revision_number.
    active = latest[latest["message_status"] != "Withdrawn"]

    return active.reset_index(drop=True).copy()


# ---------------------------------------------------------------------------
# Live fetch — Elexon Insights API ``/datasets/REMIT/stream``.  Single GET
# (the stream endpoint returns the full window in one response, no
# pagination cap — domain research §R3).  The Elexon JSON shape is
# normalised to the canonical column set by :func:`_parse_message`; the
# field-name mapping snake-cases the API's camelCase and renames a few
# Elexon idioms to project conventions (``unavailableCapacity`` →
# ``affected_mw``; ``eventStatus`` → ``message_status``).
# ---------------------------------------------------------------------------


# Elexon timestamp parser.  The Insights API emits ISO-8601 strings with
# the ``Z`` suffix (e.g. ``"2024-01-01T23:54:02Z"``); ``datetime.fromisoformat``
# handles ``Z`` natively from Python 3.11 onward.  Returns a tz-aware
# ``datetime`` so the arrow cast picks it up as ``timestamp[us, tz=UTC]``.
def _parse_elexon_timestamp(value: str | None) -> datetime | None:
    """Parse an Elexon ISO-8601 timestamp; ``None`` passes through."""
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(value)
    # Elexon always emits UTC.  Defensive: if a future field lacks tzinfo,
    # raise rather than silently lose the offset.
    if parsed.tzinfo is None:
        raise ValueError(f"Elexon REMIT timestamp {value!r} is naive; expected UTC offset (Z).")
    return parsed.astimezone(UTC)


def _parse_message(record: dict[str, Any]) -> dict[str, Any]:
    """Map one Elexon REMIT JSON record to the canonical row dict.

    The mapping is deliberately conservative: every column on
    :data:`OUTPUT_SCHEMA` is populated (using ``None`` when the API
    omits the source field), so the downstream ``_to_arrow`` cast does
    not need to invent missing columns.  Field-name correspondences
    (Elexon → project) are recorded inline.

    Required fields (``mrid``, ``revisionNumber``, ``publishTime``,
    ``eventStartTime``) raise :class:`RemitParseError` when missing,
    naming the offending field plus the available keys on the record so
    the failure is debuggable from the message alone.  This is more
    diagnostic than the bare ``KeyError`` Python's dict access raises by
    default.
    """
    required_fields = ("mrid", "revisionNumber", "publishTime", "eventStartTime")
    missing = [name for name in required_fields if name not in record]
    if missing:
        raise RemitParseError(
            f"REMIT record is missing required field(s) {missing!r}; "
            f"available keys: {sorted(record.keys())!r}"
        )
    return {
        # Identifier axis
        "mrid": record["mrid"],
        "revision_number": int(record["revisionNumber"]),
        "message_type": record.get("messageType", "Unknown"),
        # Elexon's ``eventStatus`` carries the Active / Inactive /
        # Cancelled / Withdrawn / Dismissed vocabulary documented in
        # MESSAGE_STATUSES; rename to message_status for the project.
        "message_status": record.get("eventStatus", "Unknown"),
        # Bi-temporal axis
        "published_at": _parse_elexon_timestamp(record["publishTime"]),
        "effective_from": _parse_elexon_timestamp(record["eventStartTime"]),
        "effective_to": _parse_elexon_timestamp(record.get("eventEndTime")),
        # Asset axis (raw — no master-data normalisation per intent OOS)
        "affected_unit": record.get("affectedUnit"),
        "asset_id": record.get("assetId"),
        "fuel_type": record.get("fuelType"),
        # Capacity axis — Elexon's ``unavailableCapacity`` is the
        # headline "MW down" number we use for the demo aggregate.
        "affected_mw": _coerce_optional_float(record.get("unavailableCapacity")),
        "normal_capacity_mw": _coerce_optional_float(record.get("normalCapacity")),
        # Event metadata
        "event_type": record.get("eventType"),
        "cause": record.get("cause"),
        # Free text — the stream endpoint does not return a long-form
        # message description today (per the live response observed
        # 2026-04 in domain research §R6); kept on the schema so Stage 14
        # can populate it from a follow-up ``/remit/{mrid}`` call without
        # a schema migration.
        "message_description": record.get("messageDescription"),
    }


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce ``int`` / ``float`` / ``None`` → ``float | None``.

    Elexon's JSON sometimes ships capacity as an int, sometimes a float;
    pyarrow's float64 cast accepts both but pandas' dtype-inference does
    not, so normalise here.
    """
    if value is None:
        return None
    return float(value)


def _live_fetch(
    config: RemitIngestionConfig,
    *,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    """Fetch the REMIT window from the Insights API and return canonical rows.

    Issues a single GET against ``/datasets/REMIT/stream`` with
    ``publishDateTimeFrom`` / ``publishDateTimeTo`` query parameters
    spanning ``config.window_start`` to ``config.window_end`` (defaulting
    to today's UTC date when ``window_end is None``).  The endpoint has
    no documented page cap (domain research §R3) so a single request
    suffices.

    Logs INFO with the record count + window slice (NFR-10 observability)
    and emits a WARNING for any ``message_status`` outside
    :data:`MESSAGE_STATUSES` — the row is still persisted so a downstream
    consumer can investigate.
    """
    # ``datetime.now(UTC).date()`` (not ``date.today()``): the latter
    # uses the *local* timezone, so a host clock crossing midnight UTC
    # before the local date rolls would silently fetch the wrong
    # cassette window.  Same UTC-discipline failure mode the NESO
    # ingester patched after Stage 1.
    window_end = config.window_end or datetime.now(UTC).date()
    window_start = config.window_start
    if window_end < window_start:
        # The Pydantic ``model_validator`` already catches this when the
        # config is built; defensive duplicate to keep this helper safe
        # if it is ever called with a hand-crafted config.
        raise ValueError(
            f"REMIT fetch window_end ({window_end}) is before window_start ({window_start})."
        )

    # URL composition mirrors ``neso.py`` / ``neso_forecast.py``:
    # rstrip a trailing slash from ``base_url`` and lstrip a leading
    # slash from ``endpoint_path`` so a Hydra-overridden config that
    # supplies either with or without slashes still produces a single
    # well-formed URL (no missing or duplicated separator).
    url = str(config.base_url).rstrip("/") + "/" + config.endpoint_path.lstrip("/")
    params: dict[str, object] = {
        "publishDateTimeFrom": f"{window_start.isoformat()}T00:00Z",
        "publishDateTimeTo": f"{window_end.isoformat()}T00:00Z",
    }

    logger.info(
        "REMIT live fetch: GET {} window={}..{}",
        url,
        window_start,
        window_end,
    )
    response = _retrying_get(client, url, params, config)
    payload = response.json()
    if not isinstance(payload, list):
        raise RemitParseError(
            f"REMIT stream at {url} returned non-array payload: type={type(payload).__name__}"
        )

    records: list[dict[str, Any]] = []
    unknown_statuses: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise RemitParseError(
                f"REMIT stream record is not an object: {raw!r}; expected JSON object."
            )
        parsed = _parse_message(raw)
        status = parsed["message_status"]
        if status not in MESSAGE_STATUSES:
            unknown_statuses.add(status)
        records.append(parsed)

    logger.info(
        "REMIT live fetch: {} record(s) covering window {}..{}",
        len(records),
        window_start,
        window_end,
    )
    if unknown_statuses:
        logger.warning(
            "REMIT response carried unknown message_status value(s): {}; "
            "rows preserved on disk for downstream investigation.",
            sorted(unknown_statuses),
        )
    return records


# ---------------------------------------------------------------------------
# Stub fixture (NFR-2 / DESIGN §2.1.3).  Ten hand-crafted records covering
# all four AC-1 cases (fresh / revised / withdrawn / open-ended).  The
# shape exactly matches the live ``OUTPUT_SCHEMA`` so notebook + CI
# exercise the same load/as_of code paths against deterministic data.
# ---------------------------------------------------------------------------


def _stub_records() -> list[dict[str, Any]]:
    """Return the canonical stub fixture for offline / CI use.

    Ten records spanning seven mRIDs:

    - ``M-A`` — fresh single-revision Active (Nuclear, closed window).
    - ``M-B`` — three revisions 0/1/2, all Active (Gas, end-time extended
      across revisions; tests "latest revision wins").
    - ``M-C`` — two revisions: rev 0 Active, rev 1 Withdrawn (Coal;
      tests as-of withdrawal exclusion).
    - ``M-D`` — single revision, ``effective_to=None`` (Wind; open-ended).
    - ``M-E`` — fresh single-revision Active (Nuclear, larger MW).
    - ``M-F`` — single revision, ``effective_to=None`` (Hydro; open-ended).
    - ``M-G`` — fresh single-revision Active (Solar, closed window).

    All times are UTC-aware ``datetime`` objects so the arrow cast picks
    them up as ``timestamp[us, tz=UTC]`` directly.
    """

    def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
        return datetime(year, month, day, hour, tzinfo=UTC)

    return [
        # M-A: fresh, single revision, closed window
        {
            "mrid": "M-A",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 1, 1, 9),
            "effective_from": utc(2024, 1, 15),
            "effective_to": utc(2024, 1, 20),
            "affected_unit": "T_HARTLEPOOL-1",
            "asset_id": "T_HARTLEPOOL-1",
            "fuel_type": "Nuclear",
            "affected_mw": 600.0,
            "normal_capacity_mw": 1180.0,
            "event_type": "Outage",
            "cause": "Planned",
            "message_description": "Stub: planned nuclear outage for refuelling.",
        },
        # M-B: revised three times — all Active, latest revision wins
        {
            "mrid": "M-B",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 2, 1, 10),
            "effective_from": utc(2024, 2, 10),
            "effective_to": utc(2024, 2, 15),
            "affected_unit": "T_PEMBROKE-1",
            "asset_id": "T_PEMBROKE-1",
            "fuel_type": "Gas",
            "affected_mw": 400.0,
            "normal_capacity_mw": 540.0,
            "event_type": "Outage",
            "cause": "Unplanned",
            "message_description": "Stub: gas unit unplanned outage.",
        },
        {
            "mrid": "M-B",
            "revision_number": 1,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 2, 2, 11),
            "effective_from": utc(2024, 2, 10),
            "effective_to": utc(2024, 2, 16),
            "affected_unit": "T_PEMBROKE-1",
            "asset_id": "T_PEMBROKE-1",
            "fuel_type": "Gas",
            "affected_mw": 400.0,
            "normal_capacity_mw": 540.0,
            "event_type": "Outage",
            "cause": "Unplanned",
            "message_description": "Stub: extended end time after diagnostics.",
        },
        {
            "mrid": "M-B",
            "revision_number": 2,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 2, 3, 12),
            "effective_from": utc(2024, 2, 10),
            "effective_to": utc(2024, 2, 18),
            "affected_unit": "T_PEMBROKE-1",
            "asset_id": "T_PEMBROKE-1",
            "fuel_type": "Gas",
            "affected_mw": 380.0,
            "normal_capacity_mw": 540.0,
            "event_type": "Outage",
            "cause": "Unplanned",
            "message_description": "Stub: derate revised slightly downward.",
        },
        # M-C: rev 0 Active, rev 1 Withdrawn (excluded from as-of after t-1h)
        {
            "mrid": "M-C",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 3, 1, 8),
            "effective_from": utc(2024, 3, 5),
            "effective_to": utc(2024, 3, 8),
            "affected_unit": "T_RATCLIFFE-1",
            "asset_id": "T_RATCLIFFE-1",
            "fuel_type": "Coal",
            "affected_mw": 500.0,
            "normal_capacity_mw": 500.0,
            "event_type": "Outage",
            "cause": "Planned",
            "message_description": "Stub: coal unit outage — later withdrawn.",
        },
        {
            "mrid": "M-C",
            "revision_number": 1,
            "message_type": "Production",
            "message_status": "Withdrawn",
            "published_at": utc(2024, 3, 2, 9),
            "effective_from": utc(2024, 3, 5),
            "effective_to": utc(2024, 3, 8),
            "affected_unit": "T_RATCLIFFE-1",
            "asset_id": "T_RATCLIFFE-1",
            "fuel_type": "Coal",
            "affected_mw": 500.0,
            "normal_capacity_mw": 500.0,
            "event_type": "Outage",
            "cause": "Planned",
            "message_description": "Stub: message withdrawn by participant.",
        },
        # M-D: open-ended event (effective_to=None)
        {
            "mrid": "M-D",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 4, 1, 14),
            "effective_from": utc(2024, 4, 10),
            "effective_to": None,
            "affected_unit": "T_HOWA-1",
            "asset_id": "T_HOWA-1",
            "fuel_type": "Wind",
            "affected_mw": 250.0,
            "normal_capacity_mw": 600.0,
            "event_type": "Restriction",
            "cause": "Forced",
            "message_description": "Stub: open-ended wind farm restriction.",
        },
        # M-E: fresh, single revision, closed window — second nuclear datapoint
        {
            "mrid": "M-E",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 5, 1, 7),
            "effective_from": utc(2024, 5, 15),
            "effective_to": utc(2024, 5, 20),
            "affected_unit": "T_HEYSHAM-1",
            "asset_id": "T_HEYSHAM-1",
            "fuel_type": "Nuclear",
            "affected_mw": 1100.0,
            "normal_capacity_mw": 1100.0,
            "event_type": "Outage",
            "cause": "Planned",
            "message_description": "Stub: planned nuclear maintenance.",
        },
        # M-F: open-ended hydro restriction
        {
            "mrid": "M-F",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 6, 1, 9),
            "effective_from": utc(2024, 6, 5),
            "effective_to": None,
            "affected_unit": "T_DINORWIG-1",
            "asset_id": "T_DINORWIG-1",
            "fuel_type": "Hydro",
            "affected_mw": 100.0,
            "normal_capacity_mw": 288.0,
            "event_type": "Restriction",
            "cause": "Planned",
            "message_description": "Stub: open-ended hydro restriction.",
        },
        # M-G: fresh, single revision, closed window — solar
        {
            "mrid": "M-G",
            "revision_number": 0,
            "message_type": "Production",
            "message_status": "Active",
            "published_at": utc(2024, 7, 1, 10),
            "effective_from": utc(2024, 7, 5),
            "effective_to": utc(2024, 7, 12),
            "affected_unit": "T_CLEVE-1",
            "asset_id": "T_CLEVE-1",
            "fuel_type": "Solar",
            "affected_mw": 150.0,
            "normal_capacity_mw": 200.0,
            "event_type": "Outage",
            "cause": "Forced",
            "message_description": "Stub: solar inverter fault.",
        },
    ]


def _to_arrow(
    records: Sequence[dict[str, Any]],
    *,
    retrieved_at: datetime,
) -> pa.Table:
    """Cast the canonical record list to an arrow table matching ``OUTPUT_SCHEMA``.

    ``records`` carry every column on the schema except
    ``retrieved_at_utc`` — that is project-axis provenance the ingester
    stamps once per fetch (NFR-9).

    Sorted by ``(published_at ASC, mrid ASC, revision_number ASC)`` so
    the on-disk parquet is deterministic for byte-identical idempotent
    re-fetch (D18c, AC-4 / NFR-1).
    """
    if not records:
        raise RuntimeError("REMIT fetch produced zero records; refusing to persist an empty cache.")
    frame = pd.DataFrame.from_records(list(records))
    frame["retrieved_at_utc"] = retrieved_at
    frame = frame.sort_values(
        ["published_at", "mrid", "revision_number"], kind="stable"
    ).reset_index(drop=True)

    # Project to the canonical column order before casting so the arrow
    # table's field order matches OUTPUT_SCHEMA exactly.
    frame = frame[[field.name for field in OUTPUT_SCHEMA]]
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.cast(OUTPUT_SCHEMA, safe=True)


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.ingestion.remit``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.ingestion.remit",
        description=(
            "Fetch and persist REMIT messages from the Elexon Insights API. "
            "Reads `conf/config.yaml` via Hydra; prints the resulting cache path. "
            f"Set {_STUB_ENV_VAR}=1 in the environment to use the stub fixture."
        ),
    )
    parser.add_argument(
        "--cache",
        choices=[p.value for p in CachePolicy],
        default=CachePolicy.AUTO.value,
        help="Cache policy: auto (default), refresh (force re-fetch), offline (cache only).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. ingestion.remit.cache_dir=/tmp/remit",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Imported locally so ``--help`` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.remit is None:
        print(
            "No REMIT ingestion config resolved. Ensure "
            "`ingestion/remit@ingestion.remit` is in "
            "`conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    path = fetch(cfg.ingestion.remit, cache=CachePolicy(args.cache))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
