"""NESO Day-Ahead Half-Hourly Demand Forecast Performance ingestion.

Public interface (per the ingestion layer contract):

- ``fetch(config, *, cache=CachePolicy.AUTO) -> Path`` — fetch from the
  CKAN datastore or reuse a local cache; returns the path to the
  consolidated parquet file.
- ``load(path) -> pd.DataFrame`` — cheap, pure read of the cached parquet
  with a schema assertion; returns a tz-aware dataframe.
- ``CachePolicy`` / ``CacheMissingError`` — re-exported from ``_common``.

Target CKAN resource (research R6, plan D4):
``08e41551-80f8-4e28-a416-ea473a695db9`` — *Day Ahead Half Hourly Demand
Forecast Performance*, published from April 2021 onwards.  The resource
is a single archive table (contrast with ``neso.py``, which paginates
across per-year resources) so the fetch loop is simpler: one
``_fetch_resource`` call with CKAN ``limit`` / ``offset`` pagination.

Source schema (CKAN ``datastore_search`` fields, verified 2026-04):

- ``Date``              — ISO date (``2021-04-01``).
- ``Settlement_Period`` — 1-50 (46 spring-forward, 50 autumn-fallback,
  mirroring the outturn resource).
- ``Datetime``          — local wall-clock (BST / GMT depending on
  date); not used for timestamp construction because it carries no
  timezone offset and is ambiguous on autumn-fallback Sundays.
- ``Demand_Forecast``   — int MW (day-ahead forecast).
- ``Demand_Outturn``    — int MW (realised).
- ``APE``               — float % (NESO's own accuracy metric).
- plus ``Month``, ``Absolute_Error``, ``TRIAD_Avoidance_*``,
  ``Publish_Datetime``, ``_id`` which are tolerated and dropped.

Canonical UTC timestamps are built from ``(Date, Settlement_Period)``
using the same DST-shift algebra as :mod:`bristol_ml.ingestion.neso` —
see :func:`_to_utc` for the rationale.  Those helpers are duplicated
here rather than lifted to :mod:`bristol_ml.ingestion._common` because
the lift touches the Stage 1 surface; a follow-up refactor is noted in
the ingestion layer doc.

Storage schema is documented on :data:`OUTPUT_SCHEMA` and in
``src/bristol_ml/ingestion/CLAUDE.md``.

Run standalone (principle §2.1.1)::

    python -m bristol_ml.ingestion.neso_forecast [--help]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

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
from conf._schemas import NesoForecastIngestionConfig

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "IDENTITY_RAW_COLUMNS",
    "OUTPUT_SCHEMA",
    "CacheMissingError",
    "CachePolicy",
    "fetch",
    "load",
]


# Identity columns the ingester always fetches and converts, regardless
# of ``config.columns``.  Missing any of these is a hard error: the
# schema has fundamentally changed upstream.
IDENTITY_RAW_COLUMNS: tuple[str, ...] = (
    "Date",
    "Settlement_Period",
)
"""Minimum set of columns the CKAN response must carry for the
timestamp construction to succeed."""


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("timestamp_local", pa.timestamp("us", tz="Europe/London")),
        ("settlement_date", pa.date32()),
        ("settlement_period", pa.int8()),
        ("demand_forecast_mw", pa.int32()),
        ("demand_outturn_mw", pa.int32()),
        ("ape_percent", pa.float32()),
        ("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk parquet schema.  Documented in ``ingestion/CLAUDE.md``.

Raw-to-canonical renames applied at the storage boundary (mirrors the
pattern on :mod:`bristol_ml.ingestion.neso`):

    Date                → settlement_date
    Settlement_Period   → settlement_period
    Demand_Forecast     → demand_forecast_mw
    Demand_Outturn      → demand_outturn_mw
    APE                 → ape_percent
"""


# Column names persisted under the canonical lowercase schema.  Used by
# ``_to_arrow`` to project + cast; kept as a module-private constant so
# the rename map has a single point of truth.
_RAW_TO_CANONICAL: dict[str, str] = {
    "Date": "settlement_date",
    "Settlement_Period": "settlement_period",
    "Demand_Forecast": "demand_forecast_mw",
    "Demand_Outturn": "demand_outturn_mw",
    "APE": "ape_percent",
}


# ---------------------------------------------------------------------------
# fetch / load
# ---------------------------------------------------------------------------


def fetch(
    config: NesoForecastIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch (or reuse) the NESO day-ahead forecast archive; return cache path.

    Behaviour matches the layer contract:

    - ``AUTO``: if the cache file exists, return it without touching the
      network.  Otherwise fetch the full resource and persist.
    - ``REFRESH``: fetch and overwrite the cache atomically.
    - ``OFFLINE``: return the cache if present; raise
      :class:`CacheMissingError` if not.

    Returns the absolute path to the parquet file.
    """
    cache_path = _cache_path(config)

    if cache is CachePolicy.OFFLINE:
        if not cache_path.exists():
            raise CacheMissingError(
                f"NESO forecast cache not found at {cache_path}. "
                "Re-run with CachePolicy.AUTO (or REFRESH) to populate it."
            )
        logger.info("NESO forecast cache hit (offline) at {}", cache_path)
        return cache_path

    if cache is CachePolicy.AUTO and cache_path.exists():
        logger.info("NESO forecast cache hit (auto) at {}", cache_path)
        return cache_path

    logger.info(
        "NESO forecast fetch: resource={} → {} (policy={})",
        config.resource_id,
        cache_path,
        cache.value,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    with httpx.Client(timeout=config.request_timeout_seconds) as client:
        _respect_rate_limit(None, config.min_inter_request_seconds)
        raw = _fetch_resource(str(config.resource_id), config, client=client)

    required_measurements = tuple(config.columns)
    cleaned = _assert_schema(raw, required_measurements)
    tidied = _to_utc(cleaned)
    tidied["retrieved_at_utc"] = retrieved_at

    tidied = tidied.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)

    table = _to_arrow(tidied, required_measurements)
    _atomic_write(table, cache_path)
    logger.info("NESO forecast cache written: {} rows → {}", len(tidied), cache_path)
    return cache_path


def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet; assert the persisted schema; return a frame.

    The returned dataframe has a tz-aware ``timestamp_utc`` column (UTC)
    and the canonical lowercase schema (``demand_forecast_mw``,
    ``demand_outturn_mw``, ``ape_percent``) — independent of whatever
    casing the upstream CKAN resource used at ingest time.
    """
    table = pq.read_table(path)
    actual = table.schema
    for field in OUTPUT_SCHEMA:
        if field.name not in actual.names:
            raise ValueError(f"Cached parquet at {path} is missing required column {field.name!r}")
        actual_field = actual.field(field.name)
        if actual_field.type != field.type:
            raise ValueError(
                f"Column {field.name!r} in {path} has type {actual_field.type}; "
                f"expected {field.type}"
            )
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Private helpers — NESO-forecast-specific.  Generic retry/rate-limit/
# atomic-write plumbing lives in ``bristol_ml.ingestion._common``.
# ---------------------------------------------------------------------------


def _fetch_resource(
    resource_id: str,
    config: NesoForecastIngestionConfig,
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Paginate the CKAN datastore for the forecast archive.

    The two positional arguments ``(resource_id, config)`` mirror
    :func:`bristol_ml.ingestion.neso._fetch_year` minus the ``year`` key
    — the forecast archive is a single resource.  The optional
    keyword-only ``client`` lets callers share a connection-pooled
    :class:`httpx.Client`; an owned short-lived client is created when
    omitted.
    """
    if client is None:
        with httpx.Client(timeout=config.request_timeout_seconds) as owned_client:
            return _fetch_resource(resource_id, config, client=owned_client)

    url = str(config.base_url).rstrip("/") + "/datastore_search"
    collected: list[dict[str, object]] = []
    offset = 0
    total: int | None = None

    while True:
        params = {
            "resource_id": resource_id,
            "limit": config.page_size,
            "offset": offset,
        }
        response = _retrying_get(client, url, params, config)
        payload = response.json()
        if not payload.get("success", False):
            raise RuntimeError(
                f"NESO forecast CKAN call ({resource_id}) returned success=false: {payload}"
            )
        result = payload["result"]
        records = result.get("records", []) or []
        if total is None:
            total = int(result.get("total", 0))
        collected.extend(records)
        if not records:
            break
        offset += len(records)
        if offset >= total:
            break

    if not collected:
        raise RuntimeError(f"NESO forecast CKAN returned no records for {resource_id}.")

    df = pd.DataFrame.from_records(collected)
    logger.info("NESO forecast: fetched {} rows from {}", len(df), resource_id)
    return df


def _assert_schema(df: pd.DataFrame, required_measurements: tuple[str, ...]) -> pd.DataFrame:
    """Validate the raw dataframe: required columns present; unknown warn + drop.

    Required columns: ``IDENTITY_RAW_COLUMNS`` (the timestamp basis)
    plus every name in ``required_measurements`` (from
    :attr:`NesoForecastIngestionConfig.columns`).  Missing any of these
    raises :class:`KeyError` naming the offender.

    Unknown columns (not required, not ``_id``) surface a
    :class:`UserWarning` and are dropped.  Layer architecture "Schema
    assertion at ingest".

    Returns a frame restricted to the identity + required measurement
    columns, with strict dtypes enforced on the survivors.
    """
    required = (*IDENTITY_RAW_COLUMNS, *required_measurements)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"NESO forecast: required column(s) missing from response: {missing}")

    unknown = [c for c in df.columns if c not in required and c != "_id"]
    if unknown:
        warnings.warn(
            f"NESO forecast: {len(unknown)} unknown column(s) ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    subset = df[list(required)].copy()
    subset["Settlement_Period"] = pd.to_numeric(subset["Settlement_Period"], errors="raise").astype(
        "int8"
    )
    subset["Date"] = _parse_forecast_date(subset["Date"])
    for name in required_measurements:
        # The three documented measurements are ints (Demand_Forecast,
        # Demand_Outturn, Absolute_Error) and a float (APE).  Coerce via
        # ``to_numeric`` so a string "NaN" does not silently poison the
        # cast; the persisted dtypes are fixed in ``_to_arrow``.
        subset[name] = pd.to_numeric(subset[name], errors="raise")
    return subset


def _parse_forecast_date(raw: pd.Series) -> pd.Series:
    """Parse NESO forecast ``Date`` values to :class:`datetime.date`.

    The forecast resource uses ISO ``YYYY-MM-DD``; older exports of
    sibling resources use ``DD-MMM-YY``.  Accept both and raise on
    anything else so upstream drift surfaces immediately.
    """
    sample = next((v for v in raw if pd.notna(v)), None)
    if sample is None:
        return raw
    for fmt in ("%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.to_datetime(raw, format=fmt).dt.date
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Unrecognised forecast Date format (sample={sample!r})")


def _autumn_fallback_dates(years: Iterable[int]) -> dict[int, pd.Timestamp]:
    """{year: last-Sunday-of-October}.  Mirrors ``neso._autumn_fallback_dates``.

    Duplicated rather than imported from :mod:`bristol_ml.ingestion.neso`
    to keep the two NESO ingesters independently replaceable; a future
    refactor could lift the three clock-change helpers to ``_common.py``.
    """
    out: dict[int, pd.Timestamp] = {}
    for year in set(years):
        days = pd.date_range(
            start=pd.Timestamp(year=year, month=10, day=25),
            end=pd.Timestamp(year=year, month=10, day=31),
            freq="D",
        )
        sundays = [d for d in days if d.weekday() == 6]
        if sundays:
            out[year] = sundays[-1]
    return out


def _spring_forward_dates(years: Iterable[int]) -> dict[int, pd.Timestamp]:
    """{year: last-Sunday-of-March}.  Mirrors ``neso._spring_forward_dates``."""
    out: dict[int, pd.Timestamp] = {}
    for year in set(years):
        days = pd.date_range(
            start=pd.Timestamp(year=year, month=3, day=25),
            end=pd.Timestamp(year=year, month=3, day=31),
            freq="D",
        )
        sundays = [d for d in days if d.weekday() == 6]
        if sundays:
            out[year] = sundays[-1]
    return out


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Build tz-aware UTC + Europe/London timestamps from (Date, Settlement_Period).

    Reuses the Stage 1 settlement-period algebra (see
    :func:`bristol_ml.ingestion.neso._to_utc` for the full rationale):

    - Normal day (48 periods, 1-48): naive local = ``(p - 1) * 30 min``.
    - Spring-forward day (46 periods, 1-46): periods >= 3 shifted +60
      min to skip the vanished 01:00-02:00 local hour.
    - Autumn-fallback day (50 periods, 1-50): periods >= 5 shifted -60
      min to re-live the ambiguous 01:00-02:00 hour in GMT.  The
      ``ambiguous=`` mask for :meth:`Series.dt.tz_localize` resolves
      periods 3-4 to BST (first occurrence) and periods 5-6 to GMT
      (second occurrence).

    A period number outside the per-day valid range is corrupt upstream
    data and raises :class:`ValueError` naming the offending row.
    """
    settlement_date = pd.to_datetime(df["Date"])
    period = df["Settlement_Period"].astype("int64")

    years = settlement_date.dt.year.unique().tolist()
    autumn_dates = _autumn_fallback_dates(years)
    spring_dates = _spring_forward_dates(years)

    date_only = settlement_date.dt.normalize()
    is_autumn = pd.Series(False, index=df.index)
    for fallback_ts in autumn_dates.values():
        is_autumn = is_autumn | (date_only == fallback_ts)
    is_spring = pd.Series(False, index=df.index)
    for spring_ts in spring_dates.values():
        is_spring = is_spring | (date_only == spring_ts)

    invalid_spring = is_spring & (period > 46)
    invalid_autumn = is_autumn & (period > 50)
    invalid_normal = (~is_spring) & (~is_autumn) & (period > 48)
    invalid = invalid_spring | invalid_autumn | invalid_normal
    if invalid.any():
        bad = df.loc[invalid, ["Date", "Settlement_Period"]].to_dict(orient="records")
        raise ValueError(
            f"NESO forecast carries settlement periods out of the valid per-day range: {bad}"
        )

    minutes = (period - 1) * 30
    minutes = minutes.where(~(is_autumn & (period >= 5)), minutes - 60)
    minutes = minutes.where(~(is_spring & (period >= 3)), minutes + 60)

    naive_local = settlement_date + pd.to_timedelta(minutes, unit="m")

    ambiguous = is_autumn & period.isin([3, 4])
    local = naive_local.dt.tz_localize(
        "Europe/London",
        ambiguous=ambiguous.to_numpy(),
        nonexistent="raise",
    )
    out = df.copy()
    out["timestamp_utc"] = local.dt.tz_convert("UTC")
    out["timestamp_local"] = local
    return out


def _to_arrow(df: pd.DataFrame, required_measurements: tuple[str, ...]) -> pa.Table:
    """Cast the cleaned frame to the canonical parquet schema.

    Applies the raw-to-canonical column rename (``Demand_Forecast`` →
    ``demand_forecast_mw`` et al.) and backfills any documented
    measurement the caller has not requested with a null column so the
    persisted :data:`OUTPUT_SCHEMA` stays fixed regardless of which
    subset of the three documented measurements appears in
    ``config.columns``.
    """
    del required_measurements  # used only by the _assert_schema caller
    renamed = df.rename(columns=_RAW_TO_CANONICAL)

    # Backfill any documented measurement the caller did not request
    # so the cast to OUTPUT_SCHEMA below succeeds.  Extending the
    # canonical schema with a new measurement (e.g. Absolute_Error)
    # requires bumping OUTPUT_SCHEMA and this backfill list together.
    for canonical_name in ("demand_forecast_mw", "demand_outturn_mw", "ape_percent"):
        if canonical_name not in renamed.columns:
            renamed[canonical_name] = pd.Series([pd.NA] * len(renamed), dtype="Int64")

    projected = renamed[
        [
            "timestamp_utc",
            "timestamp_local",
            "settlement_date",
            "settlement_period",
            "demand_forecast_mw",
            "demand_outturn_mw",
            "ape_percent",
            "retrieved_at_utc",
        ]
    ].copy()
    projected["settlement_period"] = projected["settlement_period"].astype("int8")
    projected["demand_forecast_mw"] = pd.to_numeric(
        projected["demand_forecast_mw"], errors="coerce"
    ).astype("Int32")
    projected["demand_outturn_mw"] = pd.to_numeric(
        projected["demand_outturn_mw"], errors="coerce"
    ).astype("Int32")
    projected["ape_percent"] = pd.to_numeric(projected["ape_percent"], errors="coerce").astype(
        "float32"
    )
    table = pa.Table.from_pandas(projected, preserve_index=False)
    return table.cast(OUTPUT_SCHEMA, safe=True)


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.ingestion.neso_forecast``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.ingestion.neso_forecast",
        description=(
            "Fetch the NESO Day-Ahead Half-Hourly Demand Forecast Performance "
            "archive via CKAN and persist to parquet. Reads `conf/config.yaml` "
            "via Hydra; prints the resulting cache path."
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
        help="Hydra overrides, e.g. ingestion.neso_forecast.page_size=10000",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Imported locally so ``--help`` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.neso_forecast is None:
        print(
            "No NESO forecast ingestion config resolved. Ensure "
            "`ingestion/neso_forecast@ingestion.neso_forecast` is in "
            "`conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    path = fetch(cfg.ingestion.neso_forecast, cache=CachePolicy(args.cache))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
