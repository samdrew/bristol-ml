"""Spec-derived tests for ``bristol_ml.features.assembler``.

Every test is derived from:

- ``docs/plans/active/03-feature-assembler.md`` §4 (Acceptance Criteria) and
  §6 Task T3 (the named test list).
- ``docs/plans/active/03-feature-assembler.md`` §1 (the eight human-ratified
  decisions: D1 demand aggregation, D2 pyarrow schema, D5 missing-data policy,
  D8 provenance columns).
- ``src/bristol_ml/features/CLAUDE.md`` "Invariants" section — the load-bearing
  guarantees every downstream stage (Stage 4 onwards) relies on.
- ``docs/lld/exploration/03-feature-assembler.md`` §5 Gotchas (esp. Gotcha 2:
  tz-naive join trap; Gotcha 1: clock-change aggregation in UTC).
- ``docs/lld/research/03-feature-assembler.md`` §3 (clock-change UTC row counts).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the orchestrator.

Fixture strategy (plan §10 Risk register)
-----------------------------------------
Demand fixtures are generated programmatically from ``neso.OUTPUT_SCHEMA``
inside pytest fixtures so they cannot drift from the ingestion schema.
Weather fixtures are derived from ``national_aggregate`` on a small in-process
frame.  Neither fixture is a committed binary parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger

# Skip the entire module until the implementation lands.
assembler_mod = pytest.importorskip("bristol_ml.features.assembler")

OUTPUT_SCHEMA = assembler_mod.OUTPUT_SCHEMA
build = assembler_mod.build
load = assembler_mod.load
_resample_demand_hourly = assembler_mod._resample_demand_hourly
DEMAND_COLUMNS = assembler_mod.DEMAND_COLUMNS
WEATHER_VARIABLE_COLUMNS = assembler_mod.WEATHER_VARIABLE_COLUMNS

_WEATHER_NAMES = [name for name, _dtype in WEATHER_VARIABLE_COLUMNS]


# ---------------------------------------------------------------------------
# Loguru → caplog adapter (plan §6 T3, D5)
# ---------------------------------------------------------------------------


@pytest.fixture()
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Route loguru INFO output into pytest's caplog fixture.

    The assembler uses loguru (house style).  Stdlib ``caplog`` does not
    capture loguru records without this adapter.  Pattern taken from plan §6.
    """
    handler_id = logger.add(caplog.handler, format="{message}", level="INFO")
    yield caplog
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_feature_set_config(
    tmp_path: Path,
    *,
    name: str = "weather_only",
    demand_aggregation: str = "mean",
    forward_fill_hours: int = 3,
) -> Any:
    """Construct a minimal ``FeatureSetConfig`` for use in tests.

    Uses a local import so this file does not import ``conf`` at module level,
    keeping the ``--help`` path cheap (mirrors ingestion-test convention).
    """
    from conf._schemas import FeatureSetConfig  # type: ignore[import-not-found]

    return FeatureSetConfig(
        name=name,
        demand_aggregation=demand_aggregation,  # type: ignore[arg-type]
        cache_dir=tmp_path,
        cache_filename="test_features.parquet",
        forward_fill_hours=forward_fill_hours,
    )


def _make_half_hourly_demand(
    start_utc: str,
    n_periods: int,
    nd_mw: int = 30_000,
    tsd_mw: int = 32_000,
) -> pd.DataFrame:
    """Build a NESO-style half-hourly demand frame for ``_resample_demand_hourly``.

    The returned frame has ``timestamp_utc`` (tz-aware UTC) at 30-minute
    intervals plus ``nd_mw`` and ``tsd_mw`` int32 columns, mirroring
    ``neso.OUTPUT_SCHEMA``.
    """
    timestamps = pd.date_range(start=start_utc, periods=n_periods, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "nd_mw": pd.array([nd_mw] * n_periods, dtype="int32"),
            "tsd_mw": pd.array([tsd_mw] * n_periods, dtype="int32"),
        }
    )


def _make_wide_weather(
    hourly_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a minimal wide-form national weather frame for ``build()``.

    Columns are the five canonical weather variables.  The frame uses the
    index-as-timestamp convention that ``national_aggregate`` returns.
    """
    n = len(hourly_timestamps)
    data: dict[str, Any] = {"timestamp_utc": hourly_timestamps}
    for name, dtype in WEATHER_VARIABLE_COLUMNS:
        if dtype == pa.float32():
            data[name] = pd.array([10.0] * n, dtype="float32")
        else:
            data[name] = pd.array([50] * n, dtype="float32")
    return pd.DataFrame(data).set_index("timestamp_utc")


def _smoke_inputs(
    tmp_path: Path,
    n_hours: int = 4,
    start_utc: str = "2023-01-01T00:00:00+00:00",
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Return (demand_hourly, weather_national, config) for a standard smoke test."""
    demand_half = _make_half_hourly_demand(start_utc, n_periods=n_hours * 2)
    demand_hourly = _resample_demand_hourly(demand_half, agg="mean")

    weather_idx = pd.date_range(start=start_utc, periods=n_hours, freq="h", tz="UTC")
    weather_national = _make_wide_weather(weather_idx)

    config = _make_feature_set_config(tmp_path)
    return demand_hourly, weather_national, config


# ---------------------------------------------------------------------------
# Smoke test: AC-6 — a small fixture in, schema-correct DataFrame out
# ---------------------------------------------------------------------------


class TestAssemblerSmoke:
    """Guards AC-1 (deterministic), AC-2 (schema), AC-6 (smoke test).

    A small fixture of hourly demand + wide weather passes through ``build()``
    and the result round-trips through ``pa.Table.from_pandas(...).cast(
    OUTPUT_SCHEMA)`` and a temp-file ``load()`` without error.
    """

    def test_assembler_smoke(self, tmp_path: Path) -> None:
        """Smoke test: fixture hourly demand + wide weather → schema-correct DataFrame.

        Asserts:
        - Row count equals the number of overlapping hours between demand and weather.
        - Column names are exactly ``OUTPUT_SCHEMA.names``.
        - Arrow round-trip via ``pa.Table.from_pandas`` + ``.cast(OUTPUT_SCHEMA)``
          raises no exception.
        - ``load()`` of the written parquet raises no exception and returns the
          same column set.

        Guards AC-6 (smoke test) and AC-2 (schema conformance).
        """
        n_hours = 4
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=n_hours)

        result = build(demand_hourly, weather_national, config)

        # Row count and column names.
        assert len(result) == n_hours, (
            f"Expected {n_hours} rows from {n_hours}-hour demand + weather; "
            f"got {len(result)} (AC-6 smoke test)."
        )
        assert list(result.columns) == OUTPUT_SCHEMA.names, (
            f"Column list mismatch: {list(result.columns)} != {OUTPUT_SCHEMA.names} (AC-2)."
        )

        # Arrow round-trip — cast enforces exact schema.
        table = pa.Table.from_pandas(result, preserve_index=False).cast(OUTPUT_SCHEMA, safe=True)
        assert table.num_rows == n_hours

        # Write and reload.
        out_path = tmp_path / "smoke_features.parquet"
        pq.write_table(table, out_path)
        reloaded = load(out_path)
        assert list(reloaded.columns) == OUTPUT_SCHEMA.names, (
            "load() must return columns matching OUTPUT_SCHEMA.names."
        )
        assert len(reloaded) == n_hours


# ---------------------------------------------------------------------------
# Determinism: AC-1 — identical inputs → identical output (modulo provenance)
# ---------------------------------------------------------------------------


class TestAssemblerDeterministic:
    """Guards AC-1: the assembler is deterministic.

    ``CLAUDE.md`` invariant: identical inputs produce identical output.
    D8 explicitly permits the two ``*_retrieved_at_utc`` provenance columns
    to differ between runs (they record wall-clock time); the deep equality
    assertion drops those two columns.
    """

    def test_assembler_deterministic(self, tmp_path: Path) -> None:
        """Two invocations of ``build()`` on identical inputs produce byte-identical
        output, excluding the two ``*_retrieved_at_utc`` provenance columns (D8).

        Guards AC-1 and CLAUDE.md invariant.
        """
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=6)
        provenance_cols = ["neso_retrieved_at_utc", "weather_retrieved_at_utc"]

        result_a = build(demand_hourly, weather_national, config)
        result_b = build(demand_hourly, weather_national, config)

        pd.testing.assert_frame_equal(
            result_a.drop(columns=provenance_cols),
            result_b.drop(columns=provenance_cols),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# Clock-change: autumn fallback produces 25 hourly UTC rows
# ---------------------------------------------------------------------------


class TestClockChangeAutumn:
    """Guards research §3: autumn-fallback day produces 25 UTC-hour rows.

    Exploration Gotcha 1: once in UTC the clock-change days are well-behaved
    — a 50-period autumn day has 25 UTC hours because the UTC timeline is
    regular and the NESO ingester has already unwound the DST algebra.
    """

    def test_assembler_clock_change_autumn(self, tmp_path: Path) -> None:
        """50-period autumn-fallback day → exactly 25 UTC-hour rows after ``build()``.

        Construction: 50 half-hourly rows anchored at UTC midnight of
        2024-10-27 (autumn-fallback Sunday), giving 25 UTC hours (00:00-24:00
        exclusive) when floored and grouped. A matching 25-row weather frame
        ensures no NaN filtering occurs.

        Guards plan Task T3 ``test_assembler_clock_change_autumn``,
        research §3.
        """
        # Autumn fallback 2024: 50 half-hourly periods.
        # Simply start at 00:00 UTC and emit 50 rows at 30-min intervals.
        # All land in UTC so grouping by floor("h") gives 25 distinct buckets.
        n_periods = 50
        demand_half = _make_half_hourly_demand("2024-10-27T00:00:00+00:00", n_periods=n_periods)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")
        assert len(demand_hourly) == 25, (
            f"_resample_demand_hourly on a 50-period autumn day must yield 25 hourly rows; "
            f"got {len(demand_hourly)}."
        )

        # Matching 25-row weather frame — no NaN so no rows are dropped.
        weather_idx = pd.date_range("2024-10-27T00:00:00+00:00", periods=25, freq="h", tz="UTC")
        weather_national = _make_wide_weather(weather_idx)
        config = _make_feature_set_config(tmp_path)

        result = build(demand_hourly, weather_national, config)
        assert len(result) == 25, (
            f"build() on a 50-period autumn-fallback day must produce 25 UTC-hour rows; "
            f"got {len(result)} (research §3 clock-change invariant)."
        )
        # Confirm no NaN remains.
        assert not result.isnull().any().any(), (
            "No NaN values expected when every weather row is present (CLAUDE.md no-NaN invariant)."
        )


# ---------------------------------------------------------------------------
# Clock-change: spring forward produces 23 hourly UTC rows
# ---------------------------------------------------------------------------


class TestClockChangeSpring:
    """Guards research §3: spring-forward day produces 23 UTC-hour rows.

    Exploration Gotcha 1: a 46-period spring-forward day has only 46 half-hour
    slots, which collapse to 23 UTC hours (00:00-23:00 exclusive) after
    flooring.  There is no "missing UTC hour" — the NESO day simply has no
    data for the clocks-forward window — so a matching 23-row weather frame
    produces a clean 23-row result with no NaN.
    """

    def test_assembler_clock_change_spring(self, tmp_path: Path) -> None:
        """46-period spring-forward day → exactly 23 UTC-hour rows after ``build()``.

        Construction: 46 half-hourly rows anchored at UTC midnight of
        2024-03-31 (spring-forward Sunday).  On a spring-forward day NESO
        emits 46 periods (periods 3 and 4 correspond to the non-existent
        01:00-02:00 BST hour, which NESO skips).  In UTC the 46 periods
        floor to 23 distinct hours (the 01:00 UTC bucket has only one period,
        but it still exists — the "missing" clock-change hour is 01:00 *BST*,
        which maps to 00:00 UTC — still present in UTC).

        Simulating via sequential 30-min UTC timestamps from midnight:
        46 rows → 23 hourly buckets (each hour except the last gets 2 rows
        because 46 / 2 = 23 exactly).

        Guards plan Task T3 ``test_assembler_clock_change_spring``.
        """
        n_periods = 46
        demand_half = _make_half_hourly_demand("2024-03-31T00:00:00+00:00", n_periods=n_periods)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")
        assert len(demand_hourly) == 23, (
            f"_resample_demand_hourly on a 46-period spring-forward day must yield 23 hourly rows; "
            f"got {len(demand_hourly)}."
        )

        # Matching 23-row weather frame — no NaN, no rows dropped by D5.
        weather_idx = pd.date_range("2024-03-31T00:00:00+00:00", periods=23, freq="h", tz="UTC")
        weather_national = _make_wide_weather(weather_idx)
        config = _make_feature_set_config(tmp_path)

        result = build(demand_hourly, weather_national, config)
        assert len(result) == 23, (
            f"build() on a 46-period spring-forward day must produce 23 UTC-hour rows; "
            f"got {len(result)} (research §3 clock-change invariant)."
        )
        # D5: no NaN should remain in the output.
        assert not result.isnull().any().any(), (
            "No NaN values expected when demand + weather are complete (D5 policy, "
            "CLAUDE.md no-NaN invariant)."
        )


# ---------------------------------------------------------------------------
# Gotcha 2: reject tz-naive join inputs
# ---------------------------------------------------------------------------


class TestRejectsLocalTimeJoin:
    """Guards Exploration Gotcha 2: joining on tz-naive timestamps is refused.

    The assembler must raise ``ValueError`` (with a message referencing UTC)
    if the demand input's ``timestamp_utc`` is tz-naive — which is what a
    caller accidentally gets if they strip tz or pass ``timestamp_local``.

    CLAUDE.md invariant: tz-aware UTC ``timestamp_utc`` is required on both
    inputs.
    """

    def test_assembler_rejects_local_time_join(self, tmp_path: Path) -> None:
        """``build()`` raises ``ValueError`` when demand ``timestamp_utc`` is tz-naive.

        Simulates the Gotcha 2 scenario: caller strips the UTC timezone from
        ``timestamp_utc`` (or accidentally passes ``timestamp_local`` renamed
        to ``timestamp_utc``).

        Guards Exploration Gotcha 2 and CLAUDE.md tz-aware invariant.
        """
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=4)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")

        # Strip timezone — simulates a caller who accidentally passed a naive series.
        demand_hourly = demand_hourly.copy()
        demand_hourly["timestamp_utc"] = demand_hourly["timestamp_utc"].dt.tz_localize(None)

        weather_idx = pd.date_range("2023-01-01T00:00:00+00:00", periods=2, freq="h", tz="UTC")
        weather_national = _make_wide_weather(weather_idx)
        config = _make_feature_set_config(tmp_path)

        with pytest.raises(ValueError, match=r"(?i)utc|tz-naive|timezone"):
            build(demand_hourly, weather_national, config)


# ---------------------------------------------------------------------------
# Schema enforcement: load() rejects extra columns
# ---------------------------------------------------------------------------


class TestOutputSchemaForbidsExtraColumns:
    """Guards AC-2: the feature-table schema is exact (Plan AC-2).

    ``load()`` must raise ``ValueError`` naming the extra column when a
    parquet is written with one column beyond ``OUTPUT_SCHEMA``.  Downstream
    models may select columns positionally for speed, so extras are unsafe.
    """

    def test_assembler_output_schema_forbids_extra_columns(self, tmp_path: Path) -> None:
        """``load()`` raises ``ValueError`` mentioning the extra column.

        A parquet written with an extra column (beyond ``OUTPUT_SCHEMA``) must
        be rejected by ``load()`` with an error message that names the offender,
        making it easy to diagnose schema drift.

        Guards AC-2 and CLAUDE.md exact-schema invariant.
        """
        # Build a valid feature table first.
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=3)
        result = build(demand_hourly, weather_national, config)

        # Cast to the canonical schema so all dtypes (including timestamp[us])
        # are correct before we add the spurious column.  Without this, pandas
        # writes timestamps as timestamp[ns] and load() trips on the type check
        # before it reaches the extra-column check.
        valid_table = pa.Table.from_pandas(result, preserve_index=False).cast(
            OUTPUT_SCHEMA, safe=True
        )

        # Append the spurious column with an explicit arrow type.
        spurious = pa.array([42] * len(result), type=pa.int64())
        table_with_extra = valid_table.append_column(
            pa.field("spurious_column", pa.int64()), spurious
        )

        out_path = tmp_path / "extra_col.parquet"
        pq.write_table(table_with_extra, out_path)

        with pytest.raises(ValueError, match="spurious_column"):
            load(out_path)


# ---------------------------------------------------------------------------
# Structured INFO log on every build() call (D5)
# ---------------------------------------------------------------------------


class TestAssemblerLogsStructuredSummary:
    """Guards D5: every ``build()`` call emits exactly one structured INFO log.

    D5 (plan §1): 'log a structured summary on every ``build()`` call:
    rows dropped (demand NaN), rows forward-filled per weather variable,
    rows dropped (weather NaN after fill cap). Log at INFO.'

    The log line must contain the four named keys:
    ``demand_nan_rows_dropped=``, ``weather_forward_filled_rows=``,
    ``weather_nan_rows_dropped_after_fill=``, and ``row_count=``,
    with numeric values that match what the test constructed.
    """

    def test_assembler_logs_missing_data_summary(
        self, tmp_path: Path, loguru_caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single INFO record is emitted per ``build()`` call with structured counts.

        Scenario constructed:
        - One demand row carries NaN on ``nd_mw`` → dropped (demand_nan=1).
        - Two weather rows have NaN on one variable, within the 3-hour fill cap
          → forward-filled (weather_filled >= 1).
        - One weather row is beyond the fill cap AND has NaN → dropped
          (weather_nan_after_fill=1).

        Asserts that the single log line contains all four expected substrings
        with correct numeric values.

        Guards D5 (plan §1) and CLAUDE.md structured-log invariant.
        """
        import logging

        # Build a demand frame where hour-index 2 has NaN nd_mw.
        n_hours = 8
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=n_hours * 2)
        # Introduce NaN demand: zero out one half-hour row so the hourly mean
        # rounds to a meaningful value; instead we inject NaN explicitly on the
        # already-resampled hourly frame.
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")
        # Set hour 2 demand to NaN (will be dropped — demand_nan_rows_dropped=1).
        demand_hourly = demand_hourly.copy()
        demand_hourly.loc[demand_hourly.index[2], "nd_mw"] = float("nan")

        # Build a weather frame for n_hours rows, then inject NaN gaps.
        weather_idx = pd.date_range(
            "2023-01-01T00:00:00+00:00", periods=n_hours, freq="h", tz="UTC"
        )
        weather_data: dict[str, Any] = {"timestamp_utc": weather_idx}
        for name, _dtype in WEATHER_VARIABLE_COLUMNS:
            weather_data[name] = pd.array([10.0] * n_hours, dtype="float32")
        weather_df = pd.DataFrame(weather_data).set_index("timestamp_utc")

        # Forward-fillable gap: set hours 4 and 5 on temperature_2m to NaN
        # (2 consecutive NaN rows, within the default fill cap of 3).
        weather_df.iloc[4, weather_df.columns.get_loc("temperature_2m")] = float("nan")
        weather_df.iloc[5, weather_df.columns.get_loc("temperature_2m")] = float("nan")

        # Unfillable gap: set hours 4-7 on dew_point_2m to NaN
        # (4 consecutive NaN rows, cap=3 -> row 7 still NaN after fill -> dropped).
        for idx in [4, 5, 6, 7]:
            weather_df.iloc[idx, weather_df.columns.get_loc("dew_point_2m")] = float("nan")

        config = _make_feature_set_config(tmp_path, forward_fill_hours=3)

        with loguru_caplog.at_level(logging.INFO):
            result = build(demand_hourly, weather_df, config)

        # Exactly one INFO record from the assembler.
        assembler_records = [
            r for r in loguru_caplog.records if "demand_nan_rows_dropped" in r.getMessage()
        ]
        assert len(assembler_records) == 1, (
            f"Expected exactly one INFO log record containing 'demand_nan_rows_dropped'; "
            f"got {len(assembler_records)}. "
            f"Records: {[r.getMessage() for r in loguru_caplog.records]}"
        )

        msg = assembler_records[0].getMessage()

        # All four required keys must appear.
        for key in (
            "demand_nan_rows_dropped=",
            "weather_forward_filled_rows=",
            "weather_nan_rows_dropped_after_fill=",
            "row_count=",
        ):
            assert key in msg, (
                f"Expected structured key {key!r} in log message; got: {msg!r} (D5 invariant)."
            )

        # demand_nan_rows_dropped must be 1 (hour-index 2 was set to NaN).
        assert "demand_nan_rows_dropped=1" in msg, (
            f"demand_nan_rows_dropped must be 1; log message: {msg!r}"
        )

        # The final row_count must match len(result).
        assert f"row_count={len(result)}" in msg, (
            f"row_count in log must equal actual result length {len(result)}; log message: {msg!r}"
        )


# ---------------------------------------------------------------------------
# _resample_demand_hourly: mean vs max per D1
# ---------------------------------------------------------------------------


class TestResampleDemandHourly:
    """Guards D1 (plan §1): ``_resample_demand_hourly`` aggregates to hourly
    via ``mean`` or ``max``; the Literal contract is enforced.

    CLAUDE.md invariant: ``int32`` demand columns are preserved after resample.
    """

    @pytest.mark.parametrize("agg", ["mean", "max"])
    def test_resample_demand_hourly_mean_and_max(self, agg: str) -> None:
        """Mean and max aggregation produce the expected per-hour value.

        Constructs two half-hourly rows per UTC hour with known values (100
        and 200 MW) and asserts:
        - ``agg="mean"`` → 150 MW per hour.
        - ``agg="max"``  → 200 MW per hour (the half-hourly peak, not the mean).

        Guards D1 (plan §1) and CLAUDE.md ``int32`` dtype invariant.
        """
        n_hours = 3
        # First half-hour of each hour: 100 MW; second: 200 MW.
        timestamps = pd.date_range(
            "2023-01-01T00:00:00+00:00", periods=n_hours * 2, freq="30min", tz="UTC"
        )
        nd_values = [100, 200] * n_hours
        demand_half = pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                "nd_mw": pd.array(nd_values, dtype="int32"),
                "tsd_mw": pd.array(nd_values, dtype="int32"),
            }
        )

        result = _resample_demand_hourly(demand_half, agg=agg)

        assert len(result) == n_hours, f"Expected {n_hours} hourly rows; got {len(result)}."
        expected = 150 if agg == "mean" else 200
        assert (result["nd_mw"] == expected).all(), (
            f"agg={agg!r}: expected nd_mw={expected} for all hours; "
            f"got {result['nd_mw'].tolist()} (D1 aggregation invariant)."
        )
        # Dtypes must remain int32 after aggregation (CLAUDE.md invariant).
        assert result["nd_mw"].dtype == "int32", (
            f"nd_mw must stay int32 after resample; got {result['nd_mw'].dtype}."
        )

    def test_resample_demand_hourly_rejects_bad_agg(self) -> None:
        """``agg='sum'`` raises ``ValueError`` citing the Literal contract.

        D1 explicitly restricts the aggregation to ``Literal["mean", "max"]``.
        Any other string — including ``'sum'`` — must fail fast so a config
        typo surfaces immediately.

        Guards D1 Literal contract.
        """
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=4)
        with pytest.raises(ValueError, match=r"(?i)mean|max|agg|Literal"):
            _resample_demand_hourly(demand_half, agg="sum")  # type: ignore[arg-type]

    def test_resample_demand_hourly_requires_tz_aware(self) -> None:
        """Tz-naive ``timestamp_utc`` raises ``ValueError``.

        The NESO ingester emits tz-aware UTC; if the upstream layer has
        regressed (tz stripped in transit), the resampler must refuse rather
        than silently mis-aggregate clock-change days.

        Guards CLAUDE.md tz-aware invariant and ``_resample_demand_hourly``
        docstring contract.
        """
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=4)
        # Strip timezone.
        demand_naive = demand_half.copy()
        demand_naive["timestamp_utc"] = demand_naive["timestamp_utc"].dt.tz_localize(None)

        with pytest.raises(ValueError, match=r"(?i)tz-aware|tz.aware|timezone|utc"):
            _resample_demand_hourly(demand_naive, agg="mean")


# ---------------------------------------------------------------------------
# CLAUDE.md invariants on build() output
# ---------------------------------------------------------------------------


class TestBuildColumnOrder:
    """Guards CLAUDE.md invariant: columns are exactly ``OUTPUT_SCHEMA.names``
    in the same order.

    Column order is contractual because downstream models may select columns
    positionally for speed.
    """

    def test_assembler_build_columns_match_schema_order(self, tmp_path: Path) -> None:
        """``list(result.columns) == OUTPUT_SCHEMA.names`` (column order is contractual).

        Guards CLAUDE.md exact-column-order invariant.
        """
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=4)
        result = build(demand_hourly, weather_national, config)

        assert list(result.columns) == OUTPUT_SCHEMA.names, (
            f"Column order mismatch: {list(result.columns)} != {OUTPUT_SCHEMA.names} "
            f"(CLAUDE.md column-order invariant)."
        )


class TestBuildDtypes:
    """Guards CLAUDE.md dtype invariants: int32 demand, float32 weather,
    tz-aware UTC provenance timestamps.
    """

    def test_assembler_build_dtypes_match_schema(self, tmp_path: Path) -> None:
        """``nd_mw``/``tsd_mw`` are ``int32``; weather vars are ``float32``;
        provenance columns are tz-aware UTC timestamps.

        Guards CLAUDE.md dtype invariants: these are the dtypes every
        downstream model (Stage 4+) relies on without further casting.
        """
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=4)
        result = build(demand_hourly, weather_national, config)

        # Demand columns: int32.
        for col in DEMAND_COLUMNS:
            assert result[col].dtype == "int32", (
                f"Column {col!r} must be int32; got {result[col].dtype} "
                f"(CLAUDE.md int32 demand invariant)."
            )

        # Weather columns: float32.
        for name, _dtype in WEATHER_VARIABLE_COLUMNS:
            assert result[name].dtype == "float32", (
                f"Column {name!r} must be float32; got {result[name].dtype} "
                f"(CLAUDE.md float32 weather invariant)."
            )

        # Provenance columns: tz-aware UTC.
        for col in ("neso_retrieved_at_utc", "weather_retrieved_at_utc"):
            series = result[col]
            assert hasattr(series.dt, "tz"), (
                f"Column {col!r} must be a datetime series; got {series.dtype}."
            )
            assert series.dt.tz is not None, (
                f"Column {col!r} must be tz-aware; got tz=None "
                f"(D8 / CLAUDE.md provenance invariant)."
            )
            assert str(series.dt.tz) == "UTC", (
                f"Column {col!r} must be UTC; got tz={series.dt.tz!r} (D8 invariant)."
            )


class TestBuildNoNaN:
    """Guards CLAUDE.md invariant: no NaN values anywhere in the ``build()`` output.

    The assembler guarantees that demand-NaN rows are dropped, weather gaps
    shorter than ``forward_fill_hours`` are filled, and longer gaps drop the
    row — so the caller never sees NaN in the result.
    """

    def test_assembler_build_no_nan_values(self, tmp_path: Path) -> None:
        """The result of ``build()`` contains no NaN values anywhere.

        Guards CLAUDE.md no-NaN invariant (load-bearing for Stage 4+).
        """
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=6)
        result = build(demand_hourly, weather_national, config)

        nan_counts = result.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        assert nan_columns.empty, (
            f"build() output must contain no NaN values; "
            f"columns with NaN: {nan_columns.to_dict()} (CLAUDE.md no-NaN invariant)."
        )


class TestBuildMonotonicUniqueTimestamps:
    """Guards CLAUDE.md invariant: ``timestamp_utc`` is strictly monotonically
    ascending and unique.

    This is the primary sort key contract; every downstream model assumes
    the feature table is sorted by time with no duplicates.
    """

    def test_assembler_build_monotonic_unique_timestamps(self, tmp_path: Path) -> None:
        """``timestamp_utc`` is strictly monotonically ascending and unique.

        Guards CLAUDE.md timestamp-ordering invariant (load-bearing for
        Stage 4+ models and the rolling-origin splitter).
        """
        demand_hourly, weather_national, config = _smoke_inputs(tmp_path, n_hours=8)
        result = build(demand_hourly, weather_national, config)

        ts = result["timestamp_utc"]

        # Unique.
        assert ts.is_unique, (
            f"timestamp_utc must be unique; found {ts.duplicated().sum()} duplicate(s) "
            f"(CLAUDE.md monotonic-unique-timestamps invariant)."
        )

        # Strictly ascending (each diff is positive).
        diffs = ts.diff().dropna()
        assert (diffs > pd.Timedelta(0)).all(), (
            f"timestamp_utc must be strictly monotonically ascending; "
            f"got non-positive diffs at positions {diffs[diffs <= pd.Timedelta(0)].index.tolist()} "
            f"(CLAUDE.md monotonic-unique-timestamps invariant)."
        )


# ---------------------------------------------------------------------------
# D5 missing-data policy: forward-fill capped at forward_fill_hours
# ---------------------------------------------------------------------------


class TestMissingDataPolicy:
    """Guards D5 (plan §1): the assembler's missing-data policy.

    - Demand NaN rows are dropped.
    - Weather NaN gaps ≤ ``forward_fill_hours`` are forward-filled.
    - Weather NaN gaps > ``forward_fill_hours`` drop the row.
    - Result contains no NaN (see TestBuildNoNaN for the no-NaN invariant).
    """

    def test_demand_nan_rows_are_dropped(self, tmp_path: Path) -> None:
        """A demand row with NaN ``nd_mw`` is removed from the output.

        Guards D5: demand NaN is not recoverable (e.g. spring-forward dropped
        hours, upstream corruption) — the row is silently dropped.
        """
        n_hours = 5
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=n_hours * 2)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")
        # Inject a NaN demand on hour 2 of the hourly frame.
        demand_hourly = demand_hourly.copy()
        demand_hourly.loc[demand_hourly.index[2], "nd_mw"] = float("nan")

        weather_idx = pd.date_range(
            "2023-01-01T00:00:00+00:00", periods=n_hours, freq="h", tz="UTC"
        )
        weather_national = _make_wide_weather(weather_idx)
        config = _make_feature_set_config(tmp_path, forward_fill_hours=3)

        result = build(demand_hourly, weather_national, config)

        # One row was dropped.
        assert len(result) == n_hours - 1, (
            f"Expected {n_hours - 1} rows after dropping one NaN demand row; "
            f"got {len(result)} (D5 demand-NaN-drop policy)."
        )
        # No NaN in result.
        assert not result.isnull().any().any(), "Result must contain no NaN after D5 policy."

    def test_weather_nan_within_fill_cap_is_filled(self, tmp_path: Path) -> None:
        """Weather NaN gaps within ``forward_fill_hours`` are forward-filled.

        Constructs a single-NaN gap on temperature_2m at hour 2 (preceded by
        a known value at hour 1) and confirms the row survives in the output
        with a non-NaN temperature.

        Guards D5: short weather gaps are a data-quality artefact, not a real
        signal; forward-filling closes them.
        """
        n_hours = 5
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=n_hours * 2)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")

        weather_idx = pd.date_range(
            "2023-01-01T00:00:00+00:00", periods=n_hours, freq="h", tz="UTC"
        )
        weather_data: dict[str, Any] = {"timestamp_utc": weather_idx}
        for name, _dtype in WEATHER_VARIABLE_COLUMNS:
            weather_data[name] = pd.array([15.0] * n_hours, dtype="float32")
        weather_df = pd.DataFrame(weather_data).set_index("timestamp_utc")
        # One-row NaN gap on temperature_2m at hour 2 (well within cap=3).
        weather_df.iloc[2, weather_df.columns.get_loc("temperature_2m")] = float("nan")

        config = _make_feature_set_config(tmp_path, forward_fill_hours=3)
        result = build(demand_hourly, weather_national=weather_df, config=config)

        # Row is NOT dropped — it was filled.
        assert len(result) == n_hours, (
            f"Expected {n_hours} rows (NaN within fill cap should be filled, not dropped); "
            f"got {len(result)} (D5 forward-fill policy)."
        )
        assert not result.isnull().any().any(), "Result must contain no NaN after forward-fill."

    def test_weather_nan_beyond_fill_cap_drops_row(self, tmp_path: Path) -> None:
        """Weather NaN gaps beyond ``forward_fill_hours`` cause the rows to be dropped.

        Constructs a gap of 5 consecutive NaN rows on temperature_2m with
        fill cap = 2.  The rows at positions 3+ remain NaN after fill and are
        dropped.

        Guards D5 forward-fill cap and drop policy.
        """
        n_hours = 8
        demand_half = _make_half_hourly_demand("2023-01-01T00:00:00+00:00", n_periods=n_hours * 2)
        demand_hourly = _resample_demand_hourly(demand_half, agg="mean")

        weather_idx = pd.date_range(
            "2023-01-01T00:00:00+00:00", periods=n_hours, freq="h", tz="UTC"
        )
        weather_data: dict[str, Any] = {"timestamp_utc": weather_idx}
        for name, _dtype in WEATHER_VARIABLE_COLUMNS:
            weather_data[name] = pd.array([15.0] * n_hours, dtype="float32")
        weather_df = pd.DataFrame(weather_data).set_index("timestamp_utc")

        # 5-row NaN gap starting at position 1 on temperature_2m.
        # With fill cap=2, positions 1+2 are filled; positions 3,4,5 remain NaN → 3 rows dropped.
        for i in range(1, 6):
            weather_df.iloc[i, weather_df.columns.get_loc("temperature_2m")] = float("nan")

        config = _make_feature_set_config(tmp_path, forward_fill_hours=2)
        result = build(demand_hourly, weather_national=weather_df, config=config)

        # 3 rows should be dropped (positions 3, 4, 5 still NaN after fill cap=2).
        assert len(result) == n_hours - 3, (
            f"Expected {n_hours - 3} rows after dropping 3 weather-NaN rows (cap=2); "
            f"got {len(result)} (D5 cap-and-drop policy)."
        )
        assert not result.isnull().any().any(), "Result must contain no NaN after D5 policy."
