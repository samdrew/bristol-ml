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

from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa

from bristol_ml.ingestion._common import (
    CacheMissingError,
    CachePolicy,
)
from conf._schemas import RemitIngestionConfig

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "FUEL_TYPES",
    "MESSAGE_STATUSES",
    "OUTPUT_SCHEMA",
    "CacheMissingError",
    "CachePolicy",
    "as_of",
    "fetch",
    "load",
]


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

    *Stub at T2 — implementation lands in T3 (stub path) and T4 (live path).*
    """
    raise NotImplementedError(
        "remit.fetch is stubbed at T2; T3 wires the stub path and T4 the "
        "live path against /datasets/REMIT/stream."
    )


def load(path: Path) -> pd.DataFrame:
    """Read the cached REMIT parquet and assert the schema.

    *Stub at T2 — implementation lands in T3.*
    """
    raise NotImplementedError(
        "remit.load is stubbed at T2; T3 wires the parquet read."
    )


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
