"""Calendar features for the GB demand forecaster (Stage 5).

Pure-function derivation of the 44 calendar columns required by the
Stage 5 ``weather_calendar`` feature set.  Input: an hourly dataframe
with a tz-aware UTC ``timestamp_utc`` column and the gov.uk-derived
``holidays_df`` as produced by :func:`bristol_ml.ingestion.holidays.load`.
Output: the input frame with 44 new ``int8`` columns appended.  No I/O,
no global state, no side effects beyond a single structured ``loguru``
``INFO`` line per call (matching the Stage 3 D-5 convention) plus an
optional ``WARNING`` line when the D-6 historical-depth fallback fires.

## Column catalogue (44 columns total)

1. **``hour_of_day_01`` … ``hour_of_day_23``** (23 cols) — one-hot over
   the ``Europe/London`` local hour of day.  Hour 0 is the dropped
   reference category (per plan **D-3**; external research §R4).
   Local, not UTC: GB demand patterns track the clock humans
   experience, not UTC; aligning with ``day_of_week`` and ``month``
   (both local per **D-7**) means all three cyclical encodings move
   together under DST.
2. **``day_of_week_1`` … ``day_of_week_6``** (6 cols) — one-hot over
   the ``Europe/London`` day of week.  **Monday (pandas weekday = 0) is
   the dropped reference category** (plan **D-4** human mandate,
   2026-04-20).  The remaining six columns map as:

   ====================  ===================
   Column                Europe/London day
   ====================  ===================
   ``day_of_week_1``     Tuesday
   ``day_of_week_2``     Wednesday
   ``day_of_week_3``     Thursday
   ``day_of_week_4``     Friday
   ``day_of_week_5``     Saturday
   ``day_of_week_6``     Sunday
   ====================  ===================

   Pandas, ISO-8601, and US conventions each number weekdays
   differently; OLS coefficient readers need the mapping spelled out
   verbatim.  This docstring is the canonical source.

3. **``month_02`` … ``month_12``** (11 cols) — one-hot over the
   ``Europe/London`` calendar month.  January (``month_01``) is the
   dropped reference category (plan **D-4**; external research §R4).
4. **``is_bank_holiday_ew``** (1 col) — fires on rows whose
   ``Europe/London`` local date is in the ``england-and-wales``
   division's holiday-date set (plan **D-2**).
5. **``is_bank_holiday_sco``** (1 col) — fires on rows whose
   ``Europe/London`` local date is in the ``scotland`` division's
   holiday-date set (plan **D-2**).
6. **``is_day_before_holiday``** (1 col) — fires on rows whose next
   calendar date (``Europe/London``) is in the **intersection** of the
   ``england-and-wales`` and ``scotland`` holiday-date sets, i.e. a
   UK-wide statutory holiday (plan **D-5**).
7. **``is_day_after_holiday``** (1 col) — fires on rows whose previous
   calendar date is in the same intersection set (plan **D-5**).

The ``northern-ireland`` division is *explicitly not encoded* (plan
**D-2**); the ingester still persists it so a future regional stage
can encode it without re-ingesting.

**``is_weekend`` is deliberately NOT emitted** (collinearity guard;
external research §R5): the one-hot ``day_of_week`` columns already
encode weekend information and adding ``is_weekend`` would introduce
perfect multicollinearity into the OLS design matrix.  A runtime
``assert`` guards against future accidental extensions.

## DST rule (plan D-7)

All calendar features are computed against the **``Europe/London``
local date** of the row's ``timestamp_utc``.  Converting from UTC via
``dt.tz_convert("Europe/London").dt.date`` is the single source of
truth:

- On spring-forward days (last Sunday in March) the 01:00-02:00 local
  hour is skipped; the 23 UTC hours of that calendar day all share one
  local date (the Sunday).
- On autumn-fallback days (last Sunday in October) the 01:00-02:00
  local hour repeats; the 25 UTC hours of that calendar day all share
  one local date (the Sunday).
- Adjacent-day UTC hours whose local date differs (e.g. 23:00 UTC on
  2023-12-25 is 23:00 local in winter but 00:00 local on 2023-12-26 in
  summer under BST) resolve to the **local** date, not the UTC date —
  this is the whole point of the tz-convert.

## Historical-depth fallback (plan D-6)

If any input row has a local date earlier than the earliest holiday in
``holidays_df``, :func:`derive_calendar` zero-fills the four holiday
columns on those rows (``is_bank_holiday_ew``, ``is_bank_holiday_sco``,
``is_day_before_holiday``, ``is_day_after_holiday``) and logs a single
``WARNING`` per call naming the affected row count.  The function does
**not** raise — the Stage 5 feature derivation must degrade gracefully
on pre-window data so longer historical back-tests remain possible.

## Run standalone (principle §2.1.1)

::

    python -m bristol_ml.features.calendar [--help]

The CLI prints the expected output schema (44 columns) and exits.  If
the ``weather_calendar`` + ``holidays`` caches are both warm it also
loads them and prints the first few rows of the derivation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
from loguru import logger

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "CALENDAR_VARIABLE_COLUMNS",
    "derive_calendar",
]


# Hour-of-day one-hot: drop hour 0 (reference).  Zero-padded to two digits.
_HOUR_COLUMNS: tuple[tuple[str, pa.DataType], ...] = tuple(
    (f"hour_of_day_{h:02d}", pa.int8()) for h in range(1, 24)
)

# Day-of-week one-hot: Monday=0 reference, drop. Remaining 1..6 map to
# Tue..Sun (see module docstring table).
_DAY_OF_WEEK_COLUMNS: tuple[tuple[str, pa.DataType], ...] = tuple(
    (f"day_of_week_{d}", pa.int8()) for d in range(1, 7)
)

# Month one-hot: January=1 reference, drop. Zero-padded to two digits.
_MONTH_COLUMNS: tuple[tuple[str, pa.DataType], ...] = tuple(
    (f"month_{m:02d}", pa.int8()) for m in range(2, 13)
)

# Holiday flags and proximity.  Kept as int8 (not bool) for consistency
# with the one-hot columns; downstream linear models consume them as
# numeric regressors anyway.
_HOLIDAY_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("is_bank_holiday_ew", pa.int8()),
    ("is_bank_holiday_sco", pa.int8()),
    ("is_day_before_holiday", pa.int8()),
    ("is_day_after_holiday", pa.int8()),
)


CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    *_HOUR_COLUMNS,
    *_DAY_OF_WEEK_COLUMNS,
    *_MONTH_COLUMNS,
    *_HOLIDAY_COLUMNS,
)
"""The 44 calendar columns (23 + 6 + 11 + 4), in the exact order
appended by :func:`derive_calendar`.  Pinned as an ordered constant so
downstream harnesses (``LinearConfig.feature_columns``) and the
assembler's ``CALENDAR_OUTPUT_SCHEMA`` have a single source of truth.

The split is:

- 23 ``hour_of_day_{01..23}`` (int8)
- 6 ``day_of_week_{1..6}`` (int8)
- 11 ``month_{02..12}`` (int8)
- 4 holiday flags (int8): ``is_bank_holiday_ew``,
  ``is_bank_holiday_sco``, ``is_day_before_holiday``,
  ``is_day_after_holiday``
"""


# Module-level collinearity guard — if anyone ever appends ``is_weekend``
# to the output column set (either here or by mutating this constant),
# this assertion fires at import time.
_FORBIDDEN_OUTPUT_COLUMNS = frozenset({"is_weekend"})
assert not any(name in _FORBIDDEN_OUTPUT_COLUMNS for name, _ in CALENDAR_VARIABLE_COLUMNS), (
    "Calendar output schema must not contain is_weekend; day_of_week one-hot "
    "already encodes weekend information (external research §R5)."
)


# ---------------------------------------------------------------------------
# Public: derive_calendar
# ---------------------------------------------------------------------------


def derive_calendar(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Append the 44 calendar columns to ``df``.

    Parameters
    ----------
    df
        Hourly input frame.  MUST carry a tz-aware UTC
        ``timestamp_utc`` column (not an index).  Arbitrary other
        columns are preserved unchanged and placed **before** the
        calendar columns in the output.
    holidays_df
        GB bank-holidays frame as returned by
        :func:`bristol_ml.ingestion.holidays.load`.  MUST carry
        ``date`` (``datetime.date`` or equivalent) and ``division``
        (``string``) columns.  Divisions other than
        ``england-and-wales`` and ``scotland`` are ignored (plan
        **D-2**); their presence is not an error.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the 44 calendar columns appended in the order given
        by :data:`CALENDAR_VARIABLE_COLUMNS`.  All 44 are ``int8``.
        The row count and row order are preserved exactly.

    Raises
    ------
    ValueError
        If ``timestamp_utc`` is missing, not tz-aware, or not
        ``Europe/London``-convertible.  If ``holidays_df`` is missing
        a required column.
    """
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"derive_calendar expects a 'timestamp_utc' column; got {list(df.columns)!r}."
        )
    timestamps = df["timestamp_utc"]
    if not hasattr(timestamps, "dt") or timestamps.dt.tz is None:
        raise ValueError(
            "derive_calendar requires a tz-aware 'timestamp_utc'; got tz-naive. "
            "Upstream layer (ingestion / assembler) must emit UTC tz-aware timestamps."
        )

    for required in ("date", "division"):
        if required not in holidays_df.columns:
            raise ValueError(
                f"derive_calendar holidays_df is missing {required!r}; got "
                f"{list(holidays_df.columns)!r}."
            )

    # --- UTC → Europe/London local components (the D-7 contract) ----------
    # All cyclical encodings (hour, day-of-week, month) and the holiday
    # lookup use the Europe/London local component — GB demand follows the
    # local clock, not UTC, and aligning the three encodings keeps them
    # consistent across DST transitions.
    local = timestamps.dt.tz_convert("Europe/London")
    local_dates = local.dt.date  # Series[datetime.date]
    local_hours = local.dt.hour
    local_weekday = local.dt.weekday  # Monday=0..Sunday=6
    local_month = local.dt.month

    # --- Holiday-date sets per division (plan D-2) ------------------------
    ew_dates = _holiday_dates_for(holidays_df, "england-and-wales")
    sco_dates = _holiday_dates_for(holidays_df, "scotland")
    uk_wide_dates = ew_dates & sco_dates  # plan D-5 intersection rule

    # Earliest-holiday floor for the D-6 fallback.  Taken as the minimum
    # across all three divisions in the cache so a row that pre-dates the
    # cache for ANY division gets flagged, not just ew/sco.
    earliest_cached_date = _earliest_holiday_date(holidays_df)

    # --- Build the 44 columns --------------------------------------------
    derived = df.copy()

    # Hour-of-day one-hot (Europe/London — hour 0 is the dropped reference).
    for h in range(1, 24):
        derived[f"hour_of_day_{h:02d}"] = (local_hours == h).astype("int8")

    # Day-of-week one-hot (Europe/London — Monday = 0 is the dropped reference).
    for d in range(1, 7):
        derived[f"day_of_week_{d}"] = (local_weekday == d).astype("int8")

    # Month one-hot (Europe/London — January = 1 is the dropped reference).
    for m in range(2, 13):
        derived[f"month_{m:02d}"] = (local_month == m).astype("int8")

    # Holiday flags.  pandas Series.map over a set returns bool; cast to int8.
    derived["is_bank_holiday_ew"] = local_dates.map(lambda d: d in ew_dates).astype("int8")
    derived["is_bank_holiday_sco"] = local_dates.map(lambda d: d in sco_dates).astype("int8")

    # Proximity uses the intersection set (plan D-5).  The next/previous
    # calendar date is computed from the local date by adding / subtracting
    # a 1-day offset; this respects month and year boundaries cleanly.
    next_dates = local_dates.map(lambda d: d + pd.Timedelta(days=1).to_pytimedelta())
    prev_dates = local_dates.map(lambda d: d - pd.Timedelta(days=1).to_pytimedelta())
    derived["is_day_before_holiday"] = next_dates.map(lambda d: d in uk_wide_dates).astype("int8")
    derived["is_day_after_holiday"] = prev_dates.map(lambda d: d in uk_wide_dates).astype("int8")

    # --- D-6 historical-depth fallback ------------------------------------
    pre_window_rows = 0
    if earliest_cached_date is not None:
        pre_window_mask = local_dates.map(lambda d: d < earliest_cached_date)
        pre_window_rows = int(pre_window_mask.sum())
        if pre_window_rows:
            logger.warning(
                "derive_calendar: {} row(s) have local dates before the earliest "
                "cached holiday ({}); zero-filling the four holiday columns for "
                "those rows (plan D-6).",
                pre_window_rows,
                earliest_cached_date.isoformat(),
            )
            for col in (
                "is_bank_holiday_ew",
                "is_bank_holiday_sco",
                "is_day_before_holiday",
                "is_day_after_holiday",
            ):
                derived.loc[pre_window_mask, col] = 0
                derived[col] = derived[col].astype("int8")

    # --- Collinearity guard (runtime mirror of the module-level assert) --
    assert "is_weekend" not in derived.columns, (
        "derive_calendar must not emit an 'is_weekend' column (external research §R5 — "
        "perfect collinearity with the day_of_week one-hot)."
    )

    # --- Single structured INFO log (Stage 3 D-5 convention) --------------
    logger.info(
        "derive_calendar: row_count={} holiday_dates_ew={} holiday_dates_sco={} "
        "uk_wide_dates={} pre_window_rows_zero_filled={}",
        len(derived),
        len(ew_dates),
        len(sco_dates),
        len(uk_wide_dates),
        pre_window_rows,
    )
    return derived


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _holiday_dates_for(holidays_df: pd.DataFrame, division: str) -> frozenset:
    """Return the set of ``datetime.date`` values for one division.

    The set is ``frozen`` because the public ``derive_calendar`` uses it
    only for membership checks — frozenset signals the immutability.
    ``date`` values are normalised via ``pd.to_datetime(...).dt.date`` so
    inputs carrying ``datetime64[ns]`` (e.g. from a raw parquet read)
    and native ``datetime.date`` (e.g. from a pyarrow date32 column)
    compare equal.
    """
    mask = holidays_df["division"].astype("string") == division
    raw = holidays_df.loc[mask, "date"]
    if raw.empty:
        return frozenset()
    normalised = pd.to_datetime(raw).dt.date
    return frozenset(normalised.tolist())


def _earliest_holiday_date(holidays_df: pd.DataFrame):
    """Return the earliest ``datetime.date`` across all divisions, or ``None``."""
    if holidays_df.empty:
        return None
    return pd.to_datetime(holidays_df["date"]).dt.date.min()


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.features.calendar``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.features.calendar",
        description=(
            "Print the 44-column calendar feature schema produced by "
            "derive_calendar. If the weather_calendar + holidays caches are "
            "warm, also prints the first few rows of the derivation applied "
            "to the weather-only feature cache."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of derived rows to print (default: 5).",
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

    # Imported locally so ``--help`` is cheap.
    from bristol_ml.config import load_config

    print(f"CALENDAR_VARIABLE_COLUMNS ({len(CALENDAR_VARIABLE_COLUMNS)} columns):")
    for name, dtype in CALENDAR_VARIABLE_COLUMNS:
        print(f"  {name:<24} {dtype}")

    cfg = load_config(overrides=list(args.overrides))
    if cfg.ingestion.holidays is None:
        print(
            "\nNo holidays ingestion config resolved; skipping live-data preview.",
            file=sys.stderr,
        )
        return 0
    if cfg.features.weather_calendar is None:
        print(
            "\nNo weather_calendar feature set in config; skipping live-data preview. "
            "Run with `features=weather_calendar` to enable.",
            file=sys.stderr,
        )
        return 0

    holidays_cache = Path(cfg.ingestion.holidays.cache_dir) / cfg.ingestion.holidays.cache_filename
    weather_cache = (
        Path(cfg.features.weather_calendar.cache_dir) / cfg.features.weather_calendar.cache_filename
    )
    if not (holidays_cache.exists() and weather_cache.exists()):
        print(
            f"\nCaches not warm ({holidays_cache.exists()=}, {weather_cache.exists()=}); "
            "skipping live-data preview.",
            file=sys.stderr,
        )
        return 0

    from bristol_ml.features import assembler as _assembler
    from bristol_ml.ingestion import holidays as _holidays

    weather_df = _assembler.load(weather_cache)
    holidays_df = _holidays.load(holidays_cache)
    derived = derive_calendar(weather_df, holidays_df)
    print(f"\nDerived frame head({args.rows}):")
    print(derived.head(args.rows).to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
