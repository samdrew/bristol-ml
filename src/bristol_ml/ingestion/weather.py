"""Open-Meteo historical weather archive ingestion.

Public interface (per the ingestion layer contract):

- ``fetch(config, *, cache=CachePolicy.AUTO) -> Path`` — fetch from
  ``archive-api.open-meteo.com/v1/archive`` or reuse a local cache;
  returns the path to the consolidated long-form parquet file.
- ``load(path) -> pd.DataFrame`` — cheap, pure read of the cached parquet
  with a schema assertion; returns a tz-aware long-form dataframe.
- ``CachePolicy`` — re-exported from ``_common``.

Storage schema is documented in ``src/bristol_ml/ingestion/CLAUDE.md`` and
reproduced on the ``OUTPUT_SCHEMA`` constant below.

Run standalone (principle §2.1.1)::

    python -m bristol_ml.ingestion.weather [--help]

Open-Meteo notes:

- Endpoint: single GET per station covering the full date window; the API
  returns a flat JSON object with an ``hourly`` payload containing a ``time``
  array and one array per requested variable, all index-aligned. No pagination.
- Data model: ERA5 / ERA5-Land / CERRA reanalyses at ~9-11 km (not UKMO UKV 2 km
  as stated in DESIGN §4.2 — flagged for correction). Timezone UTC; no DST.
- Rate limit: 600/min, 5 000/h, 10 000/day, 300 000/month IP-based for the
  non-commercial free tier. Ten stations x one window is comfortably inside.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from datetime import UTC, date, datetime
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
from conf._schemas import WeatherIngestionConfig, WeatherStation

# ---------------------------------------------------------------------------
# Public surface — ``CachePolicy``/``CacheMissingError`` are re-exported from
# ``_common`` so ``from bristol_ml.ingestion.weather import CachePolicy`` works
# in notebooks (matches the Stage 1 NESO ergonomic).
# ---------------------------------------------------------------------------


__all__ = [
    "OUTPUT_SCHEMA",
    "CacheMissingError",
    "CachePolicy",
    "fetch",
    "load",
]


# Declared variable names that the ingester knows about. Anything requested
# beyond this list still flows through (warn-and-drop on assert), but the
# parquet schema pins these five — the default YAML set.
KNOWN_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
)
"""Variables with an arrow type baked into ``OUTPUT_SCHEMA``."""


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("station", pa.string()),
        ("temperature_2m", pa.float32()),
        ("dew_point_2m", pa.float32()),
        ("wind_speed_10m", pa.float32()),
        ("cloud_cover", pa.int8()),
        ("shortwave_radiation", pa.float32()),
        ("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk long-form parquet schema. Documented in ``ingestion/CLAUDE.md``."""


# ---------------------------------------------------------------------------
# fetch / load
# ---------------------------------------------------------------------------


def fetch(
    config: WeatherIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch (or reuse) Open-Meteo weather; return the consolidated cache path.

    Behaviour matches the layer contract:

    - ``AUTO``: if the cache file exists, return it without touching the
      network. Otherwise fetch all configured stations and persist.
    - ``REFRESH``: fetch all configured stations and overwrite atomically.
    - ``OFFLINE``: return the cache if present; raise ``CacheMissingError`` if not.

    Returns the absolute path to the parquet file.
    """
    cache_path = _cache_path(config)

    if cache is CachePolicy.OFFLINE:
        if not cache_path.exists():
            raise CacheMissingError(
                f"Weather cache not found at {cache_path}. "
                "Re-run with CachePolicy.AUTO (or REFRESH) to populate it."
            )
        logger.info("Weather cache hit (offline) at {}", cache_path)
        return cache_path

    if cache is CachePolicy.AUTO and cache_path.exists():
        logger.info("Weather cache hit (auto) at {}", cache_path)
        return cache_path

    logger.info(
        "Weather fetch: {} station(s) → {} (policy={})",
        len(config.stations),
        cache_path,
        cache.value,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=config.request_timeout_seconds) as client:
        last_request_at: float | None = None
        for station in config.stations:
            last_request_at = _respect_rate_limit(last_request_at, config.min_inter_request_seconds)
            payload = _fetch_station(station, config, client=client)
            raw = _parse_station_payload(payload, station)
            cleaned = _assert_schema(raw, station.name, config.variables)
            frames.append(cleaned)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp_utc", "station"], kind="stable").reset_index(
        drop=True
    )
    combined["retrieved_at_utc"] = retrieved_at

    table = _to_arrow(combined)
    _atomic_write(table, cache_path)
    logger.info("Weather cache written: {} rows → {}", len(combined), cache_path)
    return cache_path


def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet; assert the persisted schema; return a long-form frame.

    The returned dataframe has a tz-aware ``timestamp_utc`` column (UTC), a
    ``station`` column (one row per station and hour), one column per weather
    variable, and a ``retrieved_at_utc`` provenance column.
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
# Private helpers — Open-Meteo-specific. Generic retry/rate-limit/atomic-write
# helpers live in ``bristol_ml.ingestion._common``; this module keeps only
# the HTTP-payload parsing and per-station schema assertion.
# ---------------------------------------------------------------------------


def _fetch_station(
    station: WeatherStation,
    config: WeatherIngestionConfig,
    *,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    """Fetch the full date range for one station as a parsed JSON payload.

    Open-Meteo returns the entire hourly series in one response — there is no
    pagination. A short-lived client is created if the caller does not pass
    one; the outer ``fetch`` loop shares a single connection-pooled client
    across stations.
    """
    if client is None:
        with httpx.Client(timeout=config.request_timeout_seconds) as owned_client:
            return _fetch_station(station, config, client=owned_client)

    url = str(config.base_url)
    params: dict[str, object] = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "start_date": config.start_date.isoformat(),
        "end_date": _resolve_end_date(config).isoformat(),
        "hourly": ",".join(config.variables),
        "timezone": config.timezone,
    }
    response = _retrying_get(client, url, params, config)
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Open-Meteo returned non-object payload for station {station.name!r}: {payload!r}"
        )
    if payload.get("error"):
        raise RuntimeError(
            f"Open-Meteo error for station {station.name!r}: {payload.get('reason', payload)}"
        )
    if "hourly" not in payload:
        raise RuntimeError(
            f"Open-Meteo response for station {station.name!r} missing 'hourly' block; "
            f"keys={sorted(payload.keys())}"
        )
    logger.info(
        "Weather station {}: fetched {} hourly rows",
        station.name,
        len(payload["hourly"].get("time", []) or []),
    )
    return payload


def _resolve_end_date(config: WeatherIngestionConfig) -> date:
    """Resolve ``config.end_date`` or fall back to today (UTC).

    Open-Meteo's archive is delayed ~5 days. A user who asks for "today"
    gets what the API can serve; a missing variable on a recent day surfaces
    as NaN in the parsed payload rather than as an HTTP error.
    """
    if config.end_date is not None:
        return config.end_date
    return datetime.now(UTC).date()


def _parse_station_payload(payload: dict[str, object], station: WeatherStation) -> pd.DataFrame:
    """Zip ``hourly.time`` with each variable array into a long-form frame.

    Adds the ``station`` column so concatenated frames are self-identifying.
    """
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError(
            f"Open-Meteo 'hourly' block for station {station.name!r} is not an object: {hourly!r}"
        )
    time_values = hourly.get("time")
    if not isinstance(time_values, list):
        raise RuntimeError(f"Open-Meteo 'hourly.time' for station {station.name!r} is not a list")

    data: dict[str, object] = {"time": time_values, "station": [station.name] * len(time_values)}
    for key, values in hourly.items():
        if key == "time":
            continue
        if not isinstance(values, list):
            continue
        if len(values) != len(time_values):
            raise RuntimeError(
                f"Open-Meteo variable {key!r} for station {station.name!r} has "
                f"{len(values)} rows; expected {len(time_values)}"
            )
        data[key] = values

    return pd.DataFrame(data)


def _assert_schema(df: pd.DataFrame, station: str, requested_variables: list[str]) -> pd.DataFrame:
    """Validate the parsed station frame: required columns present; unknown warn+drop.

    Required: the ``time`` column (for timestamp conversion) plus every
    variable the caller requested. Missing a requested variable is a hard
    error — the schema has drifted upstream. Unknown variables (not in
    ``KNOWN_VARIABLES`` and not requested) produce a ``UserWarning`` and are
    dropped before the frame is concatenated.

    Returns a frame with canonical ``timestamp_utc`` (tz-aware UTC) plus the
    requested variables plus ``station``; the raw ``time`` column is dropped.
    """
    if "time" not in df.columns:
        raise KeyError(f"Weather station {station!r}: response payload missing 'time' array")
    missing = [v for v in requested_variables if v not in df.columns]
    if missing:
        raise KeyError(
            f"Weather station {station!r}: requested variable(s) missing from response: {missing}"
        )

    acceptable = {"time", "station", *requested_variables, *KNOWN_VARIABLES}
    unknown = [c for c in df.columns if c not in acceptable]
    if unknown:
        warnings.warn(
            f"Weather station {station!r}: {len(unknown)} unknown column(s) "
            f"ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    station_name = getattr(station, "name", station)
    subset = df[["time", *requested_variables]].copy()
    subset.insert(1, "station", station_name)
    # Open-Meteo returns ISO strings when ``timeformat=iso8601`` (default). With
    # ``timezone=UTC``/``GMT`` they carry no offset; we localise to UTC.
    ts = pd.to_datetime(subset["time"], utc=False)
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    subset["timestamp_utc"] = ts
    subset = subset.drop(columns=["time"])
    return subset


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    """Cast the combined long-form frame to the canonical parquet schema.

    Variable columns that happen to be absent (e.g. the caller requested a
    subset smaller than ``KNOWN_VARIABLES``) are backfilled with NaN so the
    persisted schema stays fixed. ``cloud_cover`` is cast to int8 with a
    sentinel of 0 for missing rows — this only fires for the subset path.
    """
    projected = df.copy()
    # Backfill missing known variables so the schema cast succeeds.
    for name in ("temperature_2m", "dew_point_2m", "wind_speed_10m", "shortwave_radiation"):
        if name not in projected.columns:
            projected[name] = pd.Series([float("nan")] * len(projected), dtype="float32")
        else:
            projected[name] = pd.to_numeric(projected[name], errors="coerce").astype("float32")
    if "cloud_cover" not in projected.columns:
        projected["cloud_cover"] = pd.Series([0] * len(projected), dtype="int8")
    else:
        projected["cloud_cover"] = (
            pd.to_numeric(projected["cloud_cover"], errors="coerce").fillna(0).astype("int8")
        )
    projected = projected[
        [
            "timestamp_utc",
            "station",
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "retrieved_at_utc",
        ]
    ]
    table = pa.Table.from_pandas(projected, preserve_index=False)
    return table.cast(OUTPUT_SCHEMA, safe=True)


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.ingestion.weather`
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.ingestion.weather",
        description=(
            "Fetch Open-Meteo historical weather for the configured stations "
            "and persist to parquet. Reads `conf/config.yaml` via Hydra; "
            "prints the resulting cache path."
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
        help="Hydra overrides, e.g. ingestion.weather.start_date=2023-01-01",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Imported locally so `--help` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.weather is None:
        print(
            "No weather ingestion config resolved. Ensure "
            "`ingestion/weather@ingestion.weather` is in `conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    path = fetch(cfg.ingestion.weather, cache=CachePolicy(args.cache))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
