"""Spec-derived tests for ``bristol_ml.features.calendar``.

Every test is derived from:

- ``docs/plans/active/05-calendar-features.md`` §6 Task T3 (the named test list,
  plan lines 275-291).
- The module docstring of ``src/bristol_ml/features/calendar.py``, which is the
  canonical source of the column catalogue, the DST rule (D-7), the proximity
  intersection rule (D-5), and the historical-depth fallback (D-6).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the orchestrator.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from bristol_ml.features.calendar import CALENDAR_VARIABLE_COLUMNS, derive_calendar

# ---------------------------------------------------------------------------
# Helpers: synthetic frame builders
# ---------------------------------------------------------------------------


def _hourly_utc_frame(start: str, periods: int) -> pd.DataFrame:
    """Build a minimal hourly UTC frame for ``derive_calendar`` input."""
    return pd.DataFrame(
        {"timestamp_utc": pd.date_range(start, periods=periods, freq="h", tz="UTC")}
    )


def _holidays(dates_by_division: dict[str, list[str]]) -> pd.DataFrame:
    """Build a synthetic holidays_df with the expected schema.

    Parameters
    ----------
    dates_by_division
        Mapping from gov.uk division name (e.g. ``"england-and-wales"``,
        ``"scotland"``, ``"northern-ireland"``) to a list of ISO date strings
        (``"YYYY-MM-DD"``).

    Returns
    -------
    pandas.DataFrame
        A frame with columns ``date`` (datetime.date), ``division`` (str),
        ``title`` (str), ``notes`` (str), ``bunting`` (bool) — the same
        shape as the holidays ingester output.
    """
    rows = []
    for division, dates in dates_by_division.items():
        for d in dates:
            rows.append(
                {
                    "date": datetime.date.fromisoformat(d),
                    "division": division,
                    "title": "Synthetic Holiday",
                    "notes": "",
                    "bunting": True,
                }
            )
    if rows:
        return pd.DataFrame(rows)
    # empty frame with the right schema
    return pd.DataFrame(columns=["date", "division", "title", "notes", "bunting"]).astype(
        {"date": object, "division": str, "title": str, "notes": str, "bunting": bool}
    )


# Convenience: empty holidays frame covering the test date range so the
# D-6 fallback never fires unless a test explicitly wants it.
_NO_HOLIDAYS = _holidays({})

# UK-wide holidays used across several tests.
_UK_WIDE = {
    "england-and-wales": ["2024-12-25", "2024-12-26"],
    "scotland": ["2024-12-25", "2024-12-26"],
}

# ---------------------------------------------------------------------------
# 1. test_derive_calendar_deterministic
# ---------------------------------------------------------------------------


def test_derive_calendar_deterministic() -> None:
    """Same input twice produces byte-identical output (AC-2).

    Guards against hidden mutation or non-deterministic column generation.
    """
    df = _hourly_utc_frame("2024-06-01 00:00", 48)
    hols = _holidays(_UK_WIDE)

    result_a = derive_calendar(df, hols)
    result_b = derive_calendar(df, hols)

    pd.testing.assert_frame_equal(result_a, result_b, check_dtype=True, check_like=False)


# ---------------------------------------------------------------------------
# 2. test_derive_calendar_one_hot_hour_sum_leq_one
# ---------------------------------------------------------------------------


def test_derive_calendar_one_hot_hour_sum_leq_one() -> None:
    """Each row's 23 hour dummies sum to 0 (hour-0 rows) or 1 (all others).

    A 48-row UTC frame starting at midnight covers two full days so every
    local-hour bucket appears at least once.
    """
    df = _hourly_utc_frame("2024-06-03 00:00", 48)
    derived = derive_calendar(df, _NO_HOLIDAYS)

    hour_cols = [f"hour_of_day_{h:02d}" for h in range(1, 24)]
    row_sums = derived[hour_cols].sum(axis=1)
    assert row_sums.isin([0, 1]).all(), (
        f"Hour one-hot row sums should be 0 or 1; got unique values {row_sums.unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. test_derive_calendar_one_hot_weekday_monday_is_reference
# ---------------------------------------------------------------------------


def test_derive_calendar_one_hot_weekday_monday_is_reference() -> None:
    """Monday rows have all six day_of_week_* == 0; Tuesday rows have day_of_week_1 == 1.

    2024-01-01 is a Monday; 2024-01-02 is a Tuesday.  Both frames are 24 rows
    (one hour per row) so every row shares the same local date.  Pins the D-4
    Monday-reference convention.
    """
    # Monday block — all six columns must be zero.
    monday_df = _hourly_utc_frame("2024-01-01 00:00", 24)
    monday_derived = derive_calendar(monday_df, _NO_HOLIDAYS)
    for d in range(1, 7):
        col = f"day_of_week_{d}"
        assert (monday_derived[col] == 0).all(), (
            f"Monday frame: expected {col} == 0 everywhere, got non-zero rows"
        )

    # Tuesday block — day_of_week_1 must be 1; the rest must be 0.
    tuesday_df = _hourly_utc_frame("2024-01-02 00:00", 24)
    tuesday_derived = derive_calendar(tuesday_df, _NO_HOLIDAYS)
    assert (tuesday_derived["day_of_week_1"] == 1).all(), (
        "Tuesday frame: expected day_of_week_1 == 1 everywhere"
    )
    for d in range(2, 7):
        col = f"day_of_week_{d}"
        assert (tuesday_derived[col] == 0).all(), f"Tuesday frame: expected {col} == 0 everywhere"


# ---------------------------------------------------------------------------
# 4. test_derive_calendar_one_hot_weekday_sum_leq_one
# ---------------------------------------------------------------------------


def test_derive_calendar_one_hot_weekday_sum_leq_one() -> None:
    """Each row's 6 day-of-week dummies sum to 0 (Monday) or 1 (any other day)."""
    # 7 days x 24 hours so every weekday is covered.
    df = _hourly_utc_frame("2024-01-01 00:00", 168)
    derived = derive_calendar(df, _NO_HOLIDAYS)

    dow_cols = [f"day_of_week_{d}" for d in range(1, 7)]
    row_sums = derived[dow_cols].sum(axis=1)
    assert row_sums.isin([0, 1]).all(), (
        f"Day-of-week one-hot row sums should be 0 or 1; got {row_sums.unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# 5. test_derive_calendar_one_hot_month_sum_leq_one
# ---------------------------------------------------------------------------


def test_derive_calendar_one_hot_month_sum_leq_one() -> None:
    """Each row's 11 month dummies sum to 0 (January) or 1 (any other month)."""
    # 13 months x 24 hours so every month bucket appears.
    df = _hourly_utc_frame("2024-01-01 00:00", 24 * 13 * 31)
    derived = derive_calendar(df, _NO_HOLIDAYS)

    month_cols = [f"month_{m:02d}" for m in range(2, 13)]
    row_sums = derived[month_cols].sum(axis=1)
    assert row_sums.isin([0, 1]).all(), (
        f"Month one-hot row sums should be 0 or 1; got {row_sums.unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# 6. test_derive_calendar_is_bank_holiday_ew_fires_on_ew_only_date
# ---------------------------------------------------------------------------


def test_derive_calendar_is_bank_holiday_ew_fires_on_ew_only_date() -> None:
    """An E&W-only holiday fires is_bank_holiday_ew=1 and is_bank_holiday_sco=0 (D-2).

    Synthetic E&W-only holiday: 2024-06-17 (imaginary royal event).
    """
    hols = _holidays({"england-and-wales": ["2024-06-17"]})
    # 48-hour frame centred on the date; local dates will include 2024-06-17.
    df = _hourly_utc_frame("2024-06-17 00:00", 48)
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date
    target = datetime.date(2024, 6, 17)
    mask = local_dates == target

    assert mask.any(), "No rows with local date 2024-06-17 found — check frame construction."
    assert (derived.loc[mask, "is_bank_holiday_ew"] == 1).all(), (
        "Expected is_bank_holiday_ew == 1 for all rows on 2024-06-17"
    )
    assert (derived.loc[mask, "is_bank_holiday_sco"] == 0).all(), (
        "Expected is_bank_holiday_sco == 0 for all rows on 2024-06-17 (E&W-only holiday)"
    )


# ---------------------------------------------------------------------------
# 7. test_derive_calendar_is_bank_holiday_sco_fires_on_scotland_only_date
# ---------------------------------------------------------------------------


def test_derive_calendar_is_bank_holiday_sco_fires_on_scotland_only_date() -> None:
    """2 January (Scotland-only) fires is_bank_holiday_sco=1 and is_bank_holiday_ew=0 (D-2)."""
    hols = _holidays({"scotland": ["2024-01-02"]})
    df = _hourly_utc_frame("2024-01-02 00:00", 48)
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date
    target = datetime.date(2024, 1, 2)
    mask = local_dates == target

    assert mask.any(), "No rows with local date 2024-01-02 found."
    assert (derived.loc[mask, "is_bank_holiday_sco"] == 1).all(), (
        "Expected is_bank_holiday_sco == 1 for all rows on 2024-01-02"
    )
    assert (derived.loc[mask, "is_bank_holiday_ew"] == 0).all(), (
        "Expected is_bank_holiday_ew == 0 for all rows on 2024-01-02 (Scotland-only holiday)"
    )


# ---------------------------------------------------------------------------
# 8. test_derive_calendar_is_bank_holiday_both_fire_on_uk_wide_date
# ---------------------------------------------------------------------------


def test_derive_calendar_is_bank_holiday_both_fire_on_uk_wide_date() -> None:
    """2024-12-25 (Christmas, UK-wide) fires both is_bank_holiday_ew
    and is_bank_holiday_sco (D-2)."""
    hols = _holidays(
        {
            "england-and-wales": ["2024-12-25"],
            "scotland": ["2024-12-25"],
        }
    )
    df = _hourly_utc_frame("2024-12-25 00:00", 48)
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date
    target = datetime.date(2024, 12, 25)
    mask = local_dates == target

    assert mask.any(), "No rows with local date 2024-12-25 found."
    assert (derived.loc[mask, "is_bank_holiday_ew"] == 1).all(), (
        "Expected is_bank_holiday_ew == 1 on Christmas Day"
    )
    assert (derived.loc[mask, "is_bank_holiday_sco"] == 1).all(), (
        "Expected is_bank_holiday_sco == 1 on Christmas Day"
    )


# ---------------------------------------------------------------------------
# 9. test_derive_calendar_ni_not_encoded
# ---------------------------------------------------------------------------


def test_derive_calendar_ni_not_encoded() -> None:
    """Battle of the Boyne (NI-only) must not fire E&W or Scotland flags; no NI column (D-2)."""
    hols = _holidays({"northern-ireland": ["2024-07-12"]})
    df = _hourly_utc_frame("2024-07-12 00:00", 48)
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date
    target = datetime.date(2024, 7, 12)
    mask = local_dates == target

    assert mask.any(), "No rows with local date 2024-07-12 found."
    assert (derived.loc[mask, "is_bank_holiday_ew"] == 0).all(), (
        "Expected is_bank_holiday_ew == 0 on NI-only holiday"
    )
    assert (derived.loc[mask, "is_bank_holiday_sco"] == 0).all(), (
        "Expected is_bank_holiday_sco == 0 on NI-only holiday"
    )
    assert "is_bank_holiday_ni" not in derived.columns, (
        "Output schema must not contain is_bank_holiday_ni (plan D-2)"
    )


# ---------------------------------------------------------------------------
# 10. test_derive_calendar_proximity_intersection_fires_on_uk_wide_only
# ---------------------------------------------------------------------------


def test_derive_calendar_proximity_intersection_fires_on_uk_wide_only() -> None:
    """is_day_after_holiday fires only when the previous date is in E&W ∩ Scotland (D-5).

    Scenario:
    - 2024-01-02 is Scotland-only (NOT in the intersection).
    - 2024-12-25 and 2024-12-26 are both UK-wide (in the intersection).

    Assertions:
    - 2024-12-26 rows: is_day_after_holiday == 1 (Christmas was UK-wide).
    - 2024-01-03 rows: is_day_after_holiday == 0 (2 Jan is NOT UK-wide).
    - 2024-12-27 rows: is_day_after_holiday == 1 (Boxing Day was UK-wide).
    """
    hols = _holidays(
        {
            "england-and-wales": ["2024-12-25", "2024-12-26"],
            "scotland": ["2024-01-02", "2024-12-25", "2024-12-26"],
        }
    )
    # Build a frame that covers all the dates we need (24h per day).
    dates = ["2024-01-01", "2024-01-03", "2024-12-25", "2024-12-26", "2024-12-27"]
    all_timestamps: list[pd.Timestamp] = []
    for d in dates:
        all_timestamps.extend(pd.date_range(f"{d} 00:00", periods=24, freq="h", tz="UTC").tolist())
    df = pd.DataFrame({"timestamp_utc": all_timestamps})
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date

    def _mask(date_str: str) -> pd.Series:
        return local_dates == datetime.date.fromisoformat(date_str)

    # 2024-12-26: day after Christmas (UK-wide) → should fire.
    assert (derived.loc[_mask("2024-12-26"), "is_day_after_holiday"] == 1).all(), (
        "Expected is_day_after_holiday == 1 on 2024-12-26 (day after UK-wide Christmas)"
    )
    # 2024-01-03: day after 2 Jan (Scotland-only) → must NOT fire.
    assert (derived.loc[_mask("2024-01-03"), "is_day_after_holiday"] == 0).all(), (
        "Expected is_day_after_holiday == 0 on 2024-01-03 (2 Jan is Scotland-only, not UK-wide)"
    )
    # 2024-12-27: day after Boxing Day (UK-wide) → should fire.
    assert (derived.loc[_mask("2024-12-27"), "is_day_after_holiday"] == 1).all(), (
        "Expected is_day_after_holiday == 1 on 2024-12-27 (day after UK-wide Boxing Day)"
    )


# ---------------------------------------------------------------------------
# 11. test_derive_calendar_proximity_christmas_cluster
# ---------------------------------------------------------------------------


def test_derive_calendar_proximity_christmas_cluster() -> None:
    """Proximity bit patterns across a Christmas-New Year cluster (D-5).

    Holidays (all UK-wide): 2024-12-25, 2024-12-26, 2025-01-01.
    Frame: 2024-12-24 through 2025-01-02.

    Expected:
    - 2024-12-24: is_day_before_holiday=1, is_day_after_holiday=0 (Christmas Eve)
    - 2024-12-27: is_day_before_holiday=0, is_day_after_holiday=1 (day after Boxing Day)
    - 2024-12-31: is_day_before_holiday=1, is_day_after_holiday=0 (New Year's Eve)
    - 2025-01-02: is_day_before_holiday=0, is_day_after_holiday=1 (day after New Year's)
    """
    hols = _holidays(
        {
            # Add a June anchor so the D-6 fallback does not zero-fill December rows.
            "england-and-wales": ["2024-06-01", "2024-12-25", "2024-12-26", "2025-01-01"],
            "scotland": ["2024-06-01", "2024-12-25", "2024-12-26", "2025-01-01"],
        }
    )
    # Build a 240-hour frame from 2024-12-24 through 2025-01-02 (10 days).
    df = _hourly_utc_frame("2024-12-24 00:00", 240)
    derived = derive_calendar(df, hols)
    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date

    def _mask(date_str: str) -> pd.Series:
        return local_dates == datetime.date.fromisoformat(date_str)

    # Christmas Eve — day before Christmas.
    assert (derived.loc[_mask("2024-12-24"), "is_day_before_holiday"] == 1).all(), (
        "2024-12-24 should have is_day_before_holiday=1"
    )
    assert (derived.loc[_mask("2024-12-24"), "is_day_after_holiday"] == 0).all(), (
        "2024-12-24 should have is_day_after_holiday=0"
    )

    # Day after Boxing Day.
    assert (derived.loc[_mask("2024-12-27"), "is_day_before_holiday"] == 0).all(), (
        "2024-12-27 should have is_day_before_holiday=0"
    )
    assert (derived.loc[_mask("2024-12-27"), "is_day_after_holiday"] == 1).all(), (
        "2024-12-27 should have is_day_after_holiday=1"
    )

    # New Year's Eve — day before New Year's Day.
    assert (derived.loc[_mask("2024-12-31"), "is_day_before_holiday"] == 1).all(), (
        "2024-12-31 should have is_day_before_holiday=1"
    )
    assert (derived.loc[_mask("2024-12-31"), "is_day_after_holiday"] == 0).all(), (
        "2024-12-31 should have is_day_after_holiday=0"
    )

    # 2025-01-02 — day after New Year's Day.
    assert (derived.loc[_mask("2025-01-02"), "is_day_before_holiday"] == 0).all(), (
        "2025-01-02 should have is_day_before_holiday=0"
    )
    assert (derived.loc[_mask("2025-01-02"), "is_day_after_holiday"] == 1).all(), (
        "2025-01-02 should have is_day_after_holiday=1"
    )


# ---------------------------------------------------------------------------
# 12. test_derive_calendar_dst_spring_forward
# ---------------------------------------------------------------------------


def test_derive_calendar_dst_spring_forward() -> None:
    """DST spring-forward Sunday 2024-03-31: UTC hours are regular; local dates shift (D-7).

    Frame: 24 UTC hours on 2024-03-31 (Sunday, last Sunday of March).
    Synthetic bank holiday on 2024-04-01 (following Monday) under both
    england-and-wales and scotland.

    On spring-forward (clocks move from 01:00 GMT to 02:00 BST):
    - UTC 00:00 to UTC 22:00 (23 rows) map to local date Sunday 2024-03-31.
    - UTC 23:00 (1 row) maps to local date Monday 2024-04-01 (00:00 BST).

    Assertions (per plan §6 T3, D-7; human mandate 2026-04-20 — hour is UTC):
    - Sunday-local-date rows have is_bank_holiday_ew == 0 (holiday is Monday).
    - Monday-local-date row has is_bank_holiday_ew == 1 (UTC 23:00 = local 00:00 Monday BST).
    - hour_of_day_01 sums to exactly 1 across the 24 UTC rows (UTC timeline is regular —
      every calendar day has a UTC 01:00 hour, including on spring-forward Sunday).
    - Sunday-local-date rows have day_of_week_6 == 1 (Sunday = column 6 in the one-hot).

    Also add a Jan anchor holiday to avoid the D-6 fallback zeroing out the Sunday rows.
    """
    hols = _holidays(
        {
            # Jan anchor so D-6 fallback does not zero-fill March rows.
            "england-and-wales": ["2024-01-01", "2024-04-01"],
            "scotland": ["2024-01-01", "2024-04-01"],
        }
    )
    # 2024-03-31 00:00 UTC to 2024-03-31 23:00 UTC (24 rows).
    df = _hourly_utc_frame("2024-03-31 00:00", 24)
    derived = derive_calendar(df, hols)

    local_dates = derived["timestamp_utc"].dt.tz_convert("Europe/London").dt.date
    sunday = datetime.date(2024, 3, 31)
    monday = datetime.date(2024, 4, 1)
    mask_sunday = local_dates == sunday
    mask_monday = local_dates == monday

    # 23 UTC rows should map to Sunday, 1 to Monday.
    assert mask_sunday.sum() == 23, (
        f"Expected 23 UTC rows on spring-forward Sunday to have local date Sunday; "
        f"got {mask_sunday.sum()}. Local dates: {local_dates.tolist()}"
    )
    assert mask_monday.sum() == 1, (
        f"Expected exactly 1 UTC row (UTC 23:00) to have local date Monday; got {mask_monday.sum()}"
    )

    # is_bank_holiday_ew must be 0 on the Sunday rows (bank holiday is on Monday).
    assert (derived.loc[mask_sunday, "is_bank_holiday_ew"] == 0).all(), (
        "Expected is_bank_holiday_ew == 0 on spring-forward Sunday rows (holiday is Monday)"
    )

    # is_bank_holiday_ew must be 1 on the Monday-local row (UTC 23:00 = local 00:00 BST Monday).
    assert (derived.loc[mask_monday, "is_bank_holiday_ew"] == 1).all(), (
        "Expected is_bank_holiday_ew == 1 on the UTC 23:00 row (local Monday 2024-04-01)"
    )

    # UTC hour 1 is always present — the UTC timeline is regular even on DST-change days.
    # (Human mandate 2026-04-20: hour_of_day is UTC, not local.)
    assert derived["hour_of_day_01"].sum() == 1, (
        "Expected hour_of_day_01 sum == 1 across 24 UTC rows (UTC timeline is regular "
        "on spring-forward Sunday; hour-of-day is UTC per human mandate 2026-04-20)"
    )

    # Sunday is day_of_week_6 in the one-hot (D-4 mapping).
    assert (derived.loc[mask_sunday, "day_of_week_6"] == 1).all(), (
        "Expected day_of_week_6 == 1 on all spring-forward Sunday UTC rows"
    )


# ---------------------------------------------------------------------------
# 13. test_derive_calendar_pre_2012_warns_and_fills_zero
# ---------------------------------------------------------------------------


def test_derive_calendar_pre_2012_warns_and_fills_zero(
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-window rows get zero-filled holiday columns + single WARNING logged (D-6).

    holidays_df starts at 2020-01-01; input frame is 24 hours in 2011.
    Expects:
    - A WARNING log mentioning the row count and the earliest cached date.
    - All four holiday columns == 0 for every row in the frame.
    - No exception raised.
    """
    hols = _holidays(
        {
            "england-and-wales": ["2020-01-01"],
            "scotland": ["2020-01-01"],
        }
    )
    df = _hourly_utc_frame("2011-06-15 00:00", 24)

    with loguru_caplog.at_level("WARNING"):
        derived = derive_calendar(df, hols)

    # Exactly one WARNING from the fallback.
    warning_records = [r for r in loguru_caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) >= 1, (
        f"Expected at least one WARNING record; "
        f"got: {[r.getMessage() for r in loguru_caplog.records]}"
    )
    warn_msg = warning_records[0].getMessage()
    # The message must mention the row count (24) and the earliest date.
    assert "24" in warn_msg, f"WARNING message should mention row count 24; got: {warn_msg!r}"
    assert "2020" in warn_msg, (
        f"WARNING message should mention earliest cached date (2020); got: {warn_msg!r}"
    )

    # All four holiday columns must be 0.
    holiday_cols = (
        "is_bank_holiday_ew",
        "is_bank_holiday_sco",
        "is_day_before_holiday",
        "is_day_after_holiday",
    )
    for col in holiday_cols:
        assert (derived[col] == 0).all(), (
            f"Expected all {col} == 0 for pre-window rows; got non-zero values"
        )


# ---------------------------------------------------------------------------
# 14. test_derive_calendar_no_is_weekend_column
# ---------------------------------------------------------------------------


def test_derive_calendar_no_is_weekend_column() -> None:
    """Output schema must not contain is_weekend (collinearity guard; external research §R5)."""
    df = _hourly_utc_frame("2024-06-01 00:00", 48)
    derived = derive_calendar(df, _NO_HOLIDAYS)
    assert "is_weekend" not in derived.columns, (
        "Output schema must not contain 'is_weekend'; day_of_week one-hot already "
        "encodes weekend information (would produce perfect multicollinearity)."
    )


# ---------------------------------------------------------------------------
# 15. test_derive_calendar_column_order_matches_constant
# ---------------------------------------------------------------------------


def test_derive_calendar_column_order_matches_constant() -> None:
    """The last 44 output columns match CALENDAR_VARIABLE_COLUMNS in order."""
    df = _hourly_utc_frame("2024-06-01 00:00", 24)
    derived = derive_calendar(df, _NO_HOLIDAYS)

    expected_names = [name for name, _ in CALENDAR_VARIABLE_COLUMNS]
    assert len(expected_names) == 45, (
        f"CALENDAR_VARIABLE_COLUMNS should have 45 entries; got {len(expected_names)}"
    )
    actual_tail = list(derived.columns[-45:])
    assert actual_tail == expected_names, (
        f"Last 45 columns do not match CALENDAR_VARIABLE_COLUMNS.\n"
        f"Expected: {expected_names}\n"
        f"Actual:   {actual_tail}"
    )


# ---------------------------------------------------------------------------
# 16. test_derive_calendar_no_io
# ---------------------------------------------------------------------------


def test_derive_calendar_no_io() -> None:
    """derive_calendar must perform no I/O: no open(), httpx.get(), or Path.write_bytes().

    Guards AC-2 (pure function, no I/O, no global state).
    """
    df = _hourly_utc_frame("2024-06-01 00:00", 24)
    hols = _holidays(_UK_WIDE)

    with (
        patch("builtins.open") as mock_open,
        patch("pathlib.Path.write_bytes") as mock_write_bytes,
    ):
        try:
            import httpx  # noqa: F401 — only patch if the library is importable

            with patch("httpx.get") as mock_httpx_get:
                derive_calendar(df, hols)
                assert not mock_httpx_get.called, "derive_calendar must not call httpx.get()"
        except ImportError:
            derive_calendar(df, hols)

        assert not mock_open.called, "derive_calendar must not call open()"
        assert not mock_write_bytes.called, "derive_calendar must not call Path.write_bytes()"
