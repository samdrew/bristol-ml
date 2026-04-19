"""NESO Historic Demand Data ingestion.

Public interface (per the ingestion layer contract):

- ``fetch(config, *, cache=CachePolicy.AUTO) -> Path`` — fetch from CKAN or
  reuse a local cache; returns the path to the consolidated parquet file.
- ``load(path) -> pd.DataFrame`` — cheap, pure read of the cached parquet
  with a schema assertion; returns a tz-aware dataframe.
- ``CachePolicy`` — three-valued enum: ``AUTO | REFRESH | OFFLINE``.

Storage schema is documented in ``src/bristol_ml/ingestion/CLAUDE.md`` and
reproduced on the ``OUTPUT_SCHEMA`` constant below.

Run standalone (principle §2.1.1)::

    python -m bristol_ml.ingestion.neso [--help]

CKAN notes:

- Endpoint: ``GET {base_url}datastore_search?resource_id=<UUID>&limit=<N>&offset=<M>``.
- Pagination: advance ``offset`` by ``len(records)`` until the cumulative count
  reaches the server-reported ``total``. Default server ``limit`` is 100, the
  upper bound is 32 000; the server silently clamps over-sized requests.
- Rate limit: NESO's portal advises two requests per minute on the datastore.
  Respected via ``min_inter_request_seconds``.
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
from conf._schemas import NesoIngestionConfig

# ---------------------------------------------------------------------------
# Public surface — ``CachePolicy`` and ``CacheMissingError`` are re-exported
# from ``_common`` so ``from bristol_ml.ingestion.neso import CachePolicy``
# (the Stage 1 public API) keeps working unchanged after the extraction.
# ---------------------------------------------------------------------------


__all__ = [
    "OUTPUT_SCHEMA",
    "REQUIRED_RAW_COLUMNS",
    "CacheMissingError",
    "CachePolicy",
    "fetch",
    "load",
]


# Canonical ordered raw-column expectations ----------------------------------

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "SETTLEMENT_DATE",
    "SETTLEMENT_PERIOD",
    "ND",
    "TSD",
)
"""Columns that must be present on every NESO year resource. Missing any of
these is a hard error: the schema has fundamentally changed upstream."""


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("timestamp_local", pa.timestamp("us", tz="Europe/London")),
        ("settlement_date", pa.date32()),
        ("settlement_period", pa.int8()),
        ("nd_mw", pa.int32()),
        ("tsd_mw", pa.int32()),
        ("source_year", pa.int16()),
        ("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk parquet schema. Documented in ``ingestion/CLAUDE.md``."""


# ---------------------------------------------------------------------------
# fetch / load
# ---------------------------------------------------------------------------


def fetch(
    config: NesoIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch (or reuse) NESO historic demand; return the cache path.

    Behaviour:

    - ``AUTO``: if the cache file exists, return it without touching the
      network. Otherwise fetch all configured years and persist.
    - ``REFRESH``: fetch all configured years and overwrite the cache atomically.
    - ``OFFLINE``: return the cache if present; raise ``CacheMissingError`` if not.

    Returns the absolute path to the parquet file.
    """
    cache_path = _cache_path(config)

    if cache is CachePolicy.OFFLINE:
        if not cache_path.exists():
            raise CacheMissingError(
                f"NESO cache not found at {cache_path}. "
                "Re-run with CachePolicy.AUTO (or REFRESH) to populate it."
            )
        logger.info("NESO cache hit (offline) at {}", cache_path)
        return cache_path

    if cache is CachePolicy.AUTO and cache_path.exists():
        logger.info("NESO cache hit (auto) at {}", cache_path)
        return cache_path

    logger.info(
        "NESO fetch: {} year(s) → {} (policy={})",
        len(config.resources),
        cache_path,
        cache.value,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=config.request_timeout_seconds) as client:
        last_request_at: float | None = None
        for res in config.resources:
            last_request_at = _respect_rate_limit(last_request_at, config.min_inter_request_seconds)
            raw = _fetch_year(res.year, str(res.resource_id), config, client=client)
            cleaned = _assert_schema(raw, res.year)
            tidied = _to_utc(cleaned)
            tidied["source_year"] = res.year
            frames.append(tidied)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    combined["retrieved_at_utc"] = retrieved_at

    table = _to_arrow(combined)
    _atomic_write(table, cache_path)
    logger.info("NESO cache written: {} rows → {}", len(combined), cache_path)
    return cache_path


def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet; assert the persisted schema; return a dataframe.

    The returned dataframe has a tz-aware ``timestamp_utc`` column (UTC) and a
    tz-aware ``timestamp_local`` column (Europe/London) — per the schema
    documented in ``ingestion/CLAUDE.md``.
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
# Private helpers — NESO-specific. Generic retry/rate-limit/atomic-write
# helpers (``_atomic_write``, ``_cache_path``, ``_respect_rate_limit``,
# ``_retrying_get``) live in ``bristol_ml.ingestion._common``; this module
# keeps only the settlement-period and CKAN-schema logic.
# ---------------------------------------------------------------------------


def _fetch_year(
    year: int,
    resource_id: str,
    config: NesoIngestionConfig,
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Paginate the CKAN datastore for one year and return a raw dataframe.

    The three positional arguments ``(year, resource_id, config)`` form the
    public contract (per the LLD §5). The optional keyword-only ``client``
    lets the outer ``fetch`` loop share a single connection-pooled
    ``httpx.Client`` across years; when omitted, a short-lived client is
    created and closed inside this call.
    """
    if client is None:
        with httpx.Client(timeout=config.request_timeout_seconds) as owned_client:
            return _fetch_year(year, resource_id, config, client=owned_client)

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
                f"NESO CKAN call for {year} ({resource_id}) returned success=false: {payload}"
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
        raise RuntimeError(f"NESO CKAN returned no records for {year} ({resource_id}).")

    df = pd.DataFrame.from_records(collected)
    logger.info("NESO year {}: fetched {} rows", year, len(df))
    return df


def _assert_schema(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Validate the raw dataframe: required columns present; unknown columns warned + dropped.

    Signature per LLD §5: ``_assert_schema(df, year) -> pd.DataFrame``. ``year``
    is threaded through so errors and warnings can cite which NESO annual
    resource triggered them.

    Rule (layer architecture, "Schema assertion at ingest"):

    - Required missing  → ``KeyError`` naming the column.
    - Unknown columns   → ``UserWarning`` and dropped from the returned frame.
    - Known-but-extra   → silently kept only if they appear in ``REQUIRED_RAW_COLUMNS``.

    Returns a frame restricted to ``REQUIRED_RAW_COLUMNS`` only.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"NESO year {year}: required column(s) missing from response: {missing}")

    unknown = [c for c in df.columns if c not in REQUIRED_RAW_COLUMNS and c != "_id"]
    if unknown:
        warnings.warn(
            f"NESO year {year}: {len(unknown)} unknown column(s) ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    # Enforce dtypes of the surviving required columns. A silent string → int
    # mismatch on SETTLEMENT_PERIOD would corrupt the DST conversion.
    subset = df[list(REQUIRED_RAW_COLUMNS)].copy()
    subset["SETTLEMENT_PERIOD"] = pd.to_numeric(subset["SETTLEMENT_PERIOD"], errors="raise").astype(
        "int8"
    )
    subset["ND"] = pd.to_numeric(subset["ND"], errors="raise").astype("int32")
    subset["TSD"] = pd.to_numeric(subset["TSD"], errors="raise").astype("int32")
    subset["SETTLEMENT_DATE"] = _parse_settlement_date(subset["SETTLEMENT_DATE"])
    return subset


def _parse_settlement_date(raw: pd.Series) -> pd.Series:
    """Parse NESO ``SETTLEMENT_DATE`` values to Python ``date`` objects.

    NESO's CKAN responses for historic demand use ``DD-MMM-YY`` (e.g.
    ``01-Jan-23``). Older CSV exports use ISO ``YYYY-MM-DD``. Both are
    accepted; anything else fails loudly so upstream format drift surfaces
    immediately instead of being silently mis-parsed.
    """
    # Fast path: first non-null value tells us which format to use.
    sample = next((v for v in raw if pd.notna(v)), None)
    if sample is None:
        return raw
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = pd.to_datetime(raw, format=fmt).dt.date
            return parsed
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Unrecognised SETTLEMENT_DATE format (sample={sample!r})")


def _autumn_fallback_dates(years: Iterable[int]) -> dict[int, pd.Timestamp]:
    """Return {year: last-Sunday-of-October} for the given years.

    GB clock-change Sundays are defined by the 1972 Summer Time Act as the
    last Sunday of October (fallback) and last Sunday of March (spring
    forward). Returning ``pd.Timestamp`` so callers can compare against
    ``pd.to_datetime`` output.
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


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Build tz-aware UTC and Europe/London timestamps from (date, period).

    Per Elexon's numbering convention, clock-change Sundays carry a
    non-standard number of settlement periods:

    - Normal day (48 periods, 1..48): naive local = ``(p - 1) * 30 min``.
    - Spring-forward day (46 periods, 1..46): periods 1-2 follow the normal
      formula; periods 3-46 skip the non-existent 01:00-02:00 local hour,
      so naive local = ``(p - 1) * 30 + 60 min``.
    - Autumn-fallback day (50 periods, 1..50): periods 1-4 follow the normal
      formula and cover the first 00:00-02:00 BST hours; periods 5-50
      re-live the ambiguous 01:00-02:00 hour in GMT, so naive local =
      ``(p - 1) * 30 - 60 min``.

    ``tz_localize`` then resolves the autumn-fallback ambiguity via a
    deterministic boolean mask (periods 3-4 → True / first / BST; periods
    5-6 → False / second / GMT). The spring-forward gap is enforced
    structurally by the shifted formula plus ``nonexistent="raise"`` on
    data that would map to the vanished 01:00-02:00 hour. An explicit
    period-range check catches any upstream row whose period number
    exceeds the day's valid range.
    """
    settlement_date = pd.to_datetime(df["SETTLEMENT_DATE"])
    period = df["SETTLEMENT_PERIOD"].astype("int64")

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

    # Sanity checks on settlement-period ranges:
    # - Spring-forward day has 46 periods (1..46).
    # - Autumn-fallback day has 50 periods (1..50).
    # - Normal day has 48 periods (1..48).
    # Anything outside these bounds is corrupt upstream data and must raise.
    invalid_spring = is_spring & (period > 46)
    invalid_autumn = is_autumn & (period > 50)
    invalid_normal = (~is_spring) & (~is_autumn) & (period > 48)
    invalid = invalid_spring | invalid_autumn | invalid_normal
    if invalid.any():
        bad = df.loc[invalid, ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]].to_dict(orient="records")
        raise ValueError(
            f"NESO data carries settlement periods out of the valid per-day range: {bad}"
        )

    # Naive local offset in minutes.
    minutes = (period - 1) * 30
    # Autumn fallback: shift periods >= 5 back by one hour (they re-live 01:00).
    minutes = minutes.where(~(is_autumn & (period >= 5)), minutes - 60)
    # Spring forward: shift periods >= 3 forward by one hour (01:00-02:00 is skipped).
    minutes = minutes.where(~(is_spring & (period >= 3)), minutes + 60)

    naive_local = settlement_date + pd.to_timedelta(minutes, unit="m")

    # Ambiguity mask: True for period 3-4 (first / BST); False for period 5-6
    # (second / GMT). Periods on non-fallback days are unambiguous.
    ambiguous = is_autumn & period.isin([3, 4])
    local = naive_local.dt.tz_localize(
        "Europe/London",
        ambiguous=ambiguous.to_numpy(),
        nonexistent="raise",
    )
    # Preserve the source-case raw columns (SETTLEMENT_DATE, SETTLEMENT_PERIOD,
    # ND, TSD, ...) per LLD §6; the rename-to-canonical-lowercase schema
    # happens downstream in ``_to_arrow``.
    out = df.copy()
    out["timestamp_utc"] = local.dt.tz_convert("UTC")
    out["timestamp_local"] = local
    return out


def _spring_forward_dates(years: Iterable[int]) -> dict[int, pd.Timestamp]:
    """Return {year: last-Sunday-of-March} for the given years."""
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


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    """Materialise the cleaned dataframe to the canonical parquet schema.

    Applies the source-case → canonical-lowercase column rename
    (``SETTLEMENT_DATE`` → ``settlement_date``, ``ND`` → ``nd_mw``, …) at the
    storage boundary. Upstream (``_to_utc``, ``_assert_schema``) keep the
    uppercase NESO-native column names per LLD §6.
    """
    renamed = df.rename(
        columns={
            "SETTLEMENT_DATE": "settlement_date",
            "SETTLEMENT_PERIOD": "settlement_period",
            "ND": "nd_mw",
            "TSD": "tsd_mw",
        }
    )
    projected = renamed[
        [
            "timestamp_utc",
            "timestamp_local",
            "settlement_date",
            "settlement_period",
            "nd_mw",
            "tsd_mw",
            "source_year",
            "retrieved_at_utc",
        ]
    ].copy()
    # Normalise dtypes before handing to arrow so integer widths match.
    projected["settlement_period"] = projected["settlement_period"].astype("int8")
    projected["nd_mw"] = projected["nd_mw"].astype("int32")
    projected["tsd_mw"] = projected["tsd_mw"].astype("int32")
    table = pa.Table.from_pandas(projected, preserve_index=False)
    # Cast each column to the declared schema so integer widths and tz metadata
    # match the documented contract, regardless of pandas' inference.
    cast = table.cast(OUTPUT_SCHEMA, safe=True)
    return cast


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.ingestion.neso`
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.ingestion.neso",
        description=(
            "Fetch NESO historic demand via CKAN and persist to parquet. "
            "Reads `conf/config.yaml` via Hydra; prints the resulting cache path."
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
        help="Hydra overrides, e.g. ingestion.neso.page_size=10000",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Imported locally so `--help` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.neso is None:
        print(
            "No NESO ingestion config resolved. Ensure `ingestion/neso@ingestion.neso` "
            "is in `conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    path = fetch(cfg.ingestion.neso, cache=CachePolicy(args.cache))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
