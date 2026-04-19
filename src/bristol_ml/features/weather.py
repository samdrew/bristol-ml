"""Weather-derived features — population-weighted national aggregate.

At Stage 2 this module contains a single function, ``national_aggregate``,
which collapses per-station hourly weather (the long-form output of
``bristol_ml.ingestion.weather``) into a wide-form national signal using a
caller-supplied mapping of station weights.

The weighted mean idiom follows research §9: pandas has no built-in
``weighted_mean`` aggregation (open since pandas#10030, 2015). We use a
NaN-safe mask so a missing value at one station does not poison the whole
hour — missing stations drop out, remaining weights are renormalised by
``weight.sum()`` per ``(hour, variable)`` group.

Run standalone::

    python -m bristol_ml.features.weather [--help]

The CLI prints the national aggregate for the default ten stations against
the locally cached weather parquet. Useful for sanity-checking a live demo.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

__all__ = ["national_aggregate"]


_NON_VARIABLE_COLUMNS: frozenset[str] = frozenset({"timestamp_utc", "station", "retrieved_at_utc"})
"""Columns in the long-form weather frame that are **not** weather variables."""


def national_aggregate(
    df: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Collapse per-station hourly weather to a weighted national signal.

    Parameters
    ----------
    df
        Long-form weather frame as produced by
        ``bristol_ml.ingestion.weather.load`` — one row per ``(timestamp_utc,
        station)``, with one column per weather variable plus the provenance
        columns (``retrieved_at_utc``). A ``station`` column is required.
    weights
        Mapping of ``station_name -> weight`` (positive floats). A subset of
        ``df["station"].unique()`` is acceptable; subset behaviour is
        documented below.

    Returns
    -------
    pandas.DataFrame
        Wide-form hourly frame: one row per ``timestamp_utc``, one column per
        weather variable. The index is the sorted unique ``timestamp_utc``.

    Subset semantics (acceptance criterion 3)
    -----------------------------------------
    - Stations in ``weights`` that are **absent from ``df``** raise ``ValueError``
      naming the missing station(s). The caller is asking for a weight on a
      signal that is not available — silent dropping would produce a subtly
      different weighted mean than the caller intended.
    - Stations in ``df`` that are **absent from ``weights``** are silently
      excluded (treated as zero weight). This lets a demo restrict the
      aggregation to a subset of the configured stations simply by passing
      a narrower mapping.
    - Per-``(hour, variable)`` NaN values drop that station from that group;
      the remaining weights are **renormalised** to sum to 1 before the
      weighted mean. If all stations are NaN for a given hour the variable
      is NaN for that hour.
    - Equal weights on identical station inputs yield the identity
      (acceptance criterion 6): the renormalised weighted mean of a constant
      is the constant.
    """
    if "station" not in df.columns:
        raise ValueError(
            "national_aggregate expects a long-form frame with a 'station' column; "
            f"got columns={list(df.columns)}"
        )
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"national_aggregate expects a 'timestamp_utc' column; got columns={list(df.columns)}"
        )
    if not weights:
        raise ValueError("national_aggregate requires a non-empty weights mapping.")
    if any(w <= 0 for w in weights.values()):
        raise ValueError(
            f"national_aggregate weights must all be strictly positive; got {dict(weights)!r}"
        )

    available = set(df["station"].unique())
    missing_from_frame = [s for s in weights if s not in available]
    if missing_from_frame:
        raise ValueError(
            "national_aggregate: station(s) named in weights are absent from the "
            f"weather frame: {sorted(missing_from_frame)}. Fetch a wider station "
            "list or pass a narrower weights mapping."
        )

    # Restrict the frame to weighted stations and tag each row with its weight.
    weighted = df.loc[df["station"].isin(weights.keys())].copy()
    if weighted.empty:
        raise ValueError(
            "national_aggregate: no rows match the supplied weights mapping; "
            f"weighted stations={sorted(weights.keys())}"
        )
    weighted["weight"] = weighted["station"].map(weights).astype("float64")

    variable_columns = [c for c in df.columns if c not in _NON_VARIABLE_COLUMNS | {"weight"}]
    if not variable_columns:
        raise ValueError(
            "national_aggregate: no weather variables in the frame "
            f"(non-variable columns: {sorted(_NON_VARIABLE_COLUMNS)})"
        )

    results: dict[str, pd.Series] = {}
    for variable in variable_columns:
        values = pd.to_numeric(weighted[variable], errors="coerce")
        # Mask out NaNs so a missing observation at one station does not
        # contaminate the weighted mean — and renormalise the surviving weights.
        mask = values.notna()
        numer = (
            (values.where(mask) * weighted["weight"].where(mask))
            .groupby(weighted["timestamp_utc"], observed=True)
            .sum(min_count=1)
        )
        denom = (
            weighted["weight"]
            .where(mask)
            .groupby(weighted["timestamp_utc"], observed=True)
            .sum(min_count=1)
        )
        # sum(min_count=1) returns NaN when every input in the group is masked —
        # we keep that NaN rather than producing 0/0.
        results[variable] = numer / denom

    wide = pd.DataFrame(results)
    wide.index.name = "timestamp_utc"
    return wide.sort_index()


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.features.weather`
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.features.weather",
        description=(
            "Compute the population-weighted national weather aggregate from the "
            "cached weather parquet and print a summary to stdout."
        ),
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print (default: 5).",
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

    from bristol_ml.config import load_config
    from bristol_ml.ingestion import weather as weather_ingester

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.weather is None:
        print(
            "No weather ingestion config resolved. Ensure "
            "`ingestion/weather@ingestion.weather` is in `conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2
    cache_path: Path = weather_ingester.fetch(cfg.ingestion.weather)
    long_form = weather_ingester.load(cache_path)
    weights = {s.name: s.weight for s in cfg.ingestion.weather.stations}
    national = national_aggregate(long_form, weights)
    logger.info("Weather aggregate: {} hours x {} variables", len(national), national.shape[1])
    print(national.head(args.head))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
