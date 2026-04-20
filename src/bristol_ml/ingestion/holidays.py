"""GB bank-holidays ingestion (gov.uk/bank-holidays.json).

Public interface (per the ingestion layer contract):

- ``fetch(config, *, cache=CachePolicy.AUTO) -> Path`` — fetch the
  gov.uk bank-holidays feed or reuse a local cache; returns the path
  to the consolidated parquet file.
- ``load(path) -> pd.DataFrame`` — cheap, pure read of the cached
  parquet with a schema assertion; returns a frame with a ``date``
  column typed as ``datetime64[ns]`` (pyarrow ``date32``) and string
  division / title / notes columns.
- ``CachePolicy`` / ``CacheMissingError`` — re-exported from ``_common``.

Source: ``https://www.gov.uk/bank-holidays.json`` under the UK
**Open Government Licence v3.0** (crown copyright).  The payload is a
single JSON object keyed by the three UK divisions
(``england-and-wales``, ``scotland``, ``northern-ireland``); each
division carries an ``events`` array whose elements are
``{title, date, notes, bunting}`` records.

**Coverage.**  As observed 2026-04 the feed carries events from
2019-01-01 onwards and rolls forward as gov.uk publishes future years.
Stage 5 research §R1 / §R10 cites 2012-01-02 as the historical lower
bound; in practice the feed window has narrowed since that research
was captured.  The ingester tolerates earlier config windows because
rows with no matching holiday simply never match the feature-layer
lookup (plan D-6 handles pre-window rows by zero-filling the holiday
columns and logging a WARNING).

The ingester persists **every** division returned by the feed, even
though the Stage 5 feature derivation only encodes England & Wales and
Scotland (plan **D-2**).  Keeping the cache policy-agnostic means a
future regional stage does not need to re-ingest.

Storage schema is documented on :data:`OUTPUT_SCHEMA` and in
``src/bristol_ml/ingestion/CLAUDE.md``.

Run standalone (principle §2.1.1)::

    python -m bristol_ml.ingestion.holidays [--help]
"""

from __future__ import annotations

import argparse
import sys
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
from conf._schemas import HolidaysIngestionConfig

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "KNOWN_DIVISIONS",
    "OUTPUT_SCHEMA",
    "CacheMissingError",
    "CachePolicy",
    "fetch",
    "load",
]


KNOWN_DIVISIONS: tuple[str, ...] = (
    "england-and-wales",
    "scotland",
    "northern-ireland",
)
"""The three UK divisions the gov.uk feed exposes.  An unknown division
key in the payload is a hard error — the upstream schema has drifted."""


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        ("date", pa.date32()),
        ("division", pa.string()),
        ("title", pa.string()),
        ("notes", pa.string()),
        ("bunting", pa.bool_()),
        ("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk parquet schema.  Documented in ``ingestion/CLAUDE.md``.

Primary key: ``(date, division)`` unique; sorted by ``date ASC,
division ASC``.  The upstream feed has never (as of 2026-04) carried
two events with the same ``(date, division)`` tuple — a duplicate would
indicate upstream drift and raises at fetch time.
"""


# ---------------------------------------------------------------------------
# fetch / load
# ---------------------------------------------------------------------------


def fetch(
    config: HolidaysIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch (or reuse) the GB bank-holidays feed; return the cache path.

    Behaviour matches the layer contract:

    - ``AUTO``: if the cache file exists, return it without touching the
      network.  Otherwise fetch and persist.
    - ``REFRESH``: fetch and overwrite the cache atomically.
    - ``OFFLINE``: return the cache if present; raise
      :class:`CacheMissingError` if not.

    The persisted parquet contains every division listed in
    ``config.divisions`` (defaults to all three).  Filtering to a
    narrower subset is the feature layer's job.

    Returns the absolute path to the parquet file.
    """
    cache_path = _cache_path(config)

    if cache is CachePolicy.OFFLINE:
        if not cache_path.exists():
            raise CacheMissingError(
                f"Holidays cache not found at {cache_path}. "
                "Re-run with CachePolicy.AUTO (or REFRESH) to populate it."
            )
        logger.info("Holidays cache hit (offline) at {}", cache_path)
        return cache_path

    if cache is CachePolicy.AUTO and cache_path.exists():
        logger.info("Holidays cache hit (auto) at {}", cache_path)
        return cache_path

    logger.info(
        "Holidays fetch: {} → {} (policy={})",
        config.url,
        cache_path,
        cache.value,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    with httpx.Client(timeout=config.request_timeout_seconds) as client:
        _respect_rate_limit(None, config.min_inter_request_seconds)
        payload = _fetch_feed(config, client=client)

    rows = _parse_feed(payload, tuple(config.divisions))
    frame = pd.DataFrame.from_records(
        rows,
        columns=["date", "division", "title", "notes", "bunting"],
    )
    if frame.empty:
        raise RuntimeError(
            f"Holidays feed at {config.url} returned no events across divisions "
            f"{list(config.divisions)}. Treated as fatal — refusing to persist "
            "an empty cache."
        )

    frame = frame.sort_values(["date", "division"], kind="stable").reset_index(drop=True)
    _assert_no_duplicate_keys(frame)
    frame["retrieved_at_utc"] = retrieved_at

    table = _to_arrow(frame)
    _atomic_write(table, cache_path)
    logger.info(
        "Holidays cache written: {} rows across {} division(s) → {}",
        len(frame),
        frame["division"].nunique(),
        cache_path,
    )
    return cache_path


def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet; assert the persisted schema; return a frame.

    The returned dataframe carries the canonical lowercase schema:
    ``date`` (as ``datetime64[ns]`` after pyarrow ``date32`` → pandas
    conversion), ``division``, ``title``, ``notes``, ``bunting``,
    ``retrieved_at_utc``.  A schema mismatch raises :class:`ValueError`
    naming the offending column.
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
# Private helpers — gov.uk-specific.  Generic retry / rate-limit /
# atomic-write plumbing lives in ``bristol_ml.ingestion._common``.
# ---------------------------------------------------------------------------


def _fetch_feed(
    config: HolidaysIngestionConfig,
    *,
    client: httpx.Client,
) -> dict[str, object]:
    """GET the gov.uk feed and return the parsed top-level JSON object."""
    response = _retrying_get(client, str(config.url), {}, config)
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Holidays feed at {config.url} returned non-object payload: {payload!r}"
        )
    return payload


def _parse_feed(
    payload: dict[str, object],
    wanted_divisions: tuple[str, ...],
) -> list[dict[str, object]]:
    """Flatten the nested ``{division: {events: [...]}}`` payload to rows.

    - Unknown division keys (not in :data:`KNOWN_DIVISIONS`) raise — upstream
      drift that the schema-assertion contract treats as fatal.
    - Divisions listed in :data:`KNOWN_DIVISIONS` but absent from the
      payload raise — the feed is expected to cover all three.
    - Only divisions named in ``wanted_divisions`` are flattened into the
      output rows; the others are ignored at this layer.
    - A missing ``events`` list on any division raises.
    - Per-event fields missing a value fall back to sensible defaults:
      ``notes`` → empty string, ``bunting`` → False.
    """
    unknown = [k for k in payload if k not in KNOWN_DIVISIONS]
    if unknown:
        raise KeyError(
            f"Holidays feed carries unknown division key(s): {sorted(unknown)}; "
            f"expected a subset of {list(KNOWN_DIVISIONS)}"
        )
    missing = [d for d in KNOWN_DIVISIONS if d not in payload]
    if missing:
        raise KeyError(
            f"Holidays feed is missing expected division(s): {missing}; "
            f"payload keys={sorted(payload.keys())}"
        )

    rows: list[dict[str, object]] = []
    for division in wanted_divisions:
        block = payload.get(division)
        if not isinstance(block, dict):
            raise RuntimeError(f"Holidays division {division!r} block is not an object: {block!r}")
        events = block.get("events")
        if not isinstance(events, list):
            raise RuntimeError(
                f"Holidays division {division!r} has no ``events`` list; "
                f"keys={sorted(block.keys())}"
            )
        for event in events:
            if not isinstance(event, dict):
                raise RuntimeError(
                    f"Holidays event for division {division!r} is not an object: {event!r}"
                )
            if "date" not in event:
                raise KeyError(
                    f"Holidays event for division {division!r} missing 'date': {event!r}"
                )
            if "title" not in event:
                raise KeyError(
                    f"Holidays event for division {division!r} missing 'title': {event!r}"
                )
            rows.append(
                {
                    "date": event["date"],
                    "division": division,
                    "title": event["title"],
                    "notes": event.get("notes", "") or "",
                    "bunting": bool(event.get("bunting", False)),
                }
            )
    return rows


def _assert_no_duplicate_keys(frame: pd.DataFrame) -> None:
    """Raise if any ``(date, division)`` tuple repeats — the primary key."""
    duplicated = frame.duplicated(subset=["date", "division"], keep=False)
    if duplicated.any():
        offenders = (
            frame.loc[duplicated, ["date", "division", "title"]]
            .sort_values(["date", "division"])
            .to_dict(orient="records")
        )
        raise RuntimeError(f"Holidays feed carries duplicate (date, division) keys: {offenders}")


def _to_arrow(frame: pd.DataFrame) -> pa.Table:
    """Cast the cleaned frame to the canonical parquet schema.

    Parses ``date`` strings (ISO ``YYYY-MM-DD``) once at the cast
    boundary; everything upstream carries them as strings.  The
    ``bunting`` column is stored as ``bool_`` per the schema; ``notes``
    is always a string, never ``None``.
    """
    projected = frame.copy()
    projected["date"] = pd.to_datetime(projected["date"], format="%Y-%m-%d").dt.date
    projected["division"] = projected["division"].astype("string")
    projected["title"] = projected["title"].astype("string")
    projected["notes"] = projected["notes"].fillna("").astype("string")
    projected["bunting"] = projected["bunting"].astype("bool")
    projected = projected[["date", "division", "title", "notes", "bunting", "retrieved_at_utc"]]
    table = pa.Table.from_pandas(projected, preserve_index=False)
    return table.cast(OUTPUT_SCHEMA, safe=True)


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.ingestion.holidays``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.ingestion.holidays",
        description=(
            "Fetch the gov.uk GB bank-holidays feed and persist to parquet. "
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
        help="Hydra overrides, e.g. ingestion.holidays.cache_dir=/tmp/holidays",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Imported locally so ``--help`` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.holidays is None:
        print(
            "No holidays ingestion config resolved. Ensure "
            "`ingestion/holidays@ingestion.holidays` is in "
            "`conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    path = fetch(cfg.ingestion.holidays, cache=CachePolicy(args.cache))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
