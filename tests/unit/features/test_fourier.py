"""Spec-derived tests for ``bristol_ml.features.fourier``.

Every test is derived from:

- ``docs/plans/active/07-sarimax.md`` §Task T2 (plan lines 278-300).
- The module docstring of ``src/bristol_ml/features/fourier.py``, which
  documents the UTC-anchored integer-hour computation, the column-naming
  scheme, the DST-insensitive guarantee, and the copy-on-write contract.

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bristol_ml.features.fourier import _cli_main, append_weekly_fourier

# ---------------------------------------------------------------------------
# Helper: build a minimal tz-aware UTC DatetimeIndex frame
# ---------------------------------------------------------------------------

_DEFAULT_COLS = [
    "week_sin_k1",
    "week_cos_k1",
    "week_sin_k2",
    "week_cos_k2",
    "week_sin_k3",
    "week_cos_k3",
]


def _utc_frame(start: str, periods: int, *, extra_col: bool = False) -> pd.DataFrame:
    """Build a minimal UTC-indexed DataFrame for use in tests.

    Parameters
    ----------
    start
        ISO-format start timestamp (may include timezone; UTC assumed when
        no zone is given, via the ``tz="UTC"`` argument below).
    periods
        Number of hourly rows.
    extra_col
        When ``True``, include a ``value`` column of integer zeros so tests
        that inspect original-column preservation have something concrete to
        check.
    """
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    data: dict[str, object] = {}
    if extra_col:
        data["value"] = range(periods)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# 1. Column-names contract
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_column_names_contract() -> None:
    """Plan §Task T2 test 1 — six columns in exact order for default harmonics=3."""
    df = _utc_frame("2024-01-01", 24, extra_col=True)
    out = append_weekly_fourier(df)

    # Exactly six new columns were appended.
    new_cols = list(out.columns[len(df.columns) :])
    assert new_cols == _DEFAULT_COLS, f"Expected new columns {_DEFAULT_COLS!r}; got {new_cols!r}"

    # The original columns must still be present at the front.
    original_cols = list(df.columns)
    assert list(out.columns[: len(original_cols)]) == original_cols, (
        "Original columns must appear before the Fourier columns in the output"
    )


# ---------------------------------------------------------------------------
# 2. Deterministic value at a fixed timestamp
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_deterministic_at_fixed_timestamp() -> None:
    """Plan §Task T2 test 2 — closed-form check at 2024-01-01 00:00:00+00:00.

    t_hours = 1704067200 // 3600 = 473352.
    """
    target = pd.Timestamp("2024-01-01 00:00:00+00:00")
    df = pd.DataFrame(index=pd.DatetimeIndex([target]))

    out = append_weekly_fourier(df)

    t_hours = 473352  # 1704067200 // 3600
    period_hours = 168

    for k in range(1, 4):
        angular = 2.0 * np.pi * k * t_hours / period_hours
        expected_sin = np.sin(angular)
        expected_cos = np.cos(angular)

        np.testing.assert_allclose(
            out.iloc[0][f"week_sin_k{k}"],
            expected_sin,
            atol=1e-12,
            err_msg=f"week_sin_k{k}: expected {expected_sin!r}",
        )
        np.testing.assert_allclose(
            out.iloc[0][f"week_cos_k{k}"],
            expected_cos,
            atol=1e-12,
            err_msg=f"week_cos_k{k}: expected {expected_cos!r}",
        )


# ---------------------------------------------------------------------------
# 3. Period identity at lag 168
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_168_period_identity() -> None:
    """Plan §Task T2 test 3 — period identity at lag 168.

    For every row i and every default column col,
    out.iloc[i][col] must equal out.iloc[i+168][col] within atol=1e-10.
    """
    # 336 hours = 2 * 168, so valid indices run 0..167.
    df = _utc_frame("2024-01-01", 336)
    out = append_weekly_fourier(df)

    for col in _DEFAULT_COLS:
        for i in range(168):
            val_i = out.iloc[i][col]
            val_i168 = out.iloc[i + 168][col]
            np.testing.assert_allclose(
                val_i,
                val_i168,
                atol=1e-10,
                err_msg=(
                    f"Period identity failed for column {col!r} at row {i}: "
                    f"iloc[{i}]={val_i!r} vs iloc[{i + 168}]={val_i168!r}"
                ),
            )


# ---------------------------------------------------------------------------
# 4. harmonics=0 is a no-op
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_harmonics_zero_noop() -> None:
    """Plan §Task T2 test 4 — harmonics=0 returns the frame unchanged (new object)."""
    df = _utc_frame("2024-01-01", 24, extra_col=True)
    out = append_weekly_fourier(df, harmonics=0)

    # Must be a new object, not the same reference.
    assert out is not df, "harmonics=0 must return a new object, not the input frame"

    # Columns must be identical to the input.
    assert list(out.columns) == list(df.columns), (
        f"harmonics=0: expected columns {list(df.columns)!r}; got {list(out.columns)!r}"
    )

    # Values must be identical.
    pd.testing.assert_frame_equal(out, df, check_dtype=True, check_like=False)

    # Index must be preserved.
    pd.testing.assert_index_equal(out.index, df.index)


# ---------------------------------------------------------------------------
# 5. Rejects tz-naive and non-DatetimeIndex
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_rejects_tz_naive_index() -> None:
    """Plan §Task T2 test 5 — ValueError on tz-naive index and on non-DatetimeIndex."""
    # tz-naive DatetimeIndex raises ValueError referencing tz-awareness.
    naive_index = pd.date_range("2024-01-01", periods=24, freq="h")
    df_naive = pd.DataFrame(index=naive_index)

    with pytest.raises(ValueError, match=r"(?i)tz") as exc_info:
        append_weekly_fourier(df_naive)
    # The message must reference tz-awareness in some form.
    assert "tz" in str(exc_info.value).lower(), (
        f"Expected ValueError message to reference tz; got: {exc_info.value!r}"
    )

    # Non-DatetimeIndex (RangeIndex) also raises ValueError.
    df_range = pd.DataFrame({"x": range(24)})
    assert isinstance(df_range.index, pd.RangeIndex)

    with pytest.raises(ValueError):
        append_weekly_fourier(df_range)


# ---------------------------------------------------------------------------
# 6. Does not mutate input
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_does_not_mutate_input() -> None:
    """Plan §Task T2 test 6 — caller's frame is unmodified after the call."""
    df = _utc_frame("2024-01-01", 24, extra_col=True)
    snapshot = df.copy()

    append_weekly_fourier(df)

    pd.testing.assert_frame_equal(
        df,
        snapshot,
        check_dtype=True,
        check_like=False,
        obj="input DataFrame after append_weekly_fourier call",
    )


# ---------------------------------------------------------------------------
# 7. DST insensitivity — no jump across DST transitions
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_output_is_not_dst_sensitive() -> None:
    """Plan §Task T2 test 7 — Fourier columns are continuous across DST transitions.

    Spans 48 hours around each of the two 2024 DST transition Sundays:
    - Spring: 2024-03-31 (clocks forward GMT→BST)
    - Autumn: 2024-10-27 (clocks back BST→GMT)

    The first-difference of each column across adjacent UTC hours must be
    bounded by 0.15 (the theoretical max |dsin/dt| for k=3 is ≈0.112 per
    hour; 0.15 gives a safe but tight margin).  A literal DST re-anchoring
    would produce a one-hour jump of up to 2.0 in the sin/cos output.
    """
    # 48 hours centred on each DST boundary (24 before, 24 after).
    spring_idx = pd.date_range("2024-03-30 00:00", periods=48, freq="h", tz="UTC")
    autumn_idx = pd.date_range("2024-10-26 00:00", periods=48, freq="h", tz="UTC")

    for label, idx in [("spring DST", spring_idx), ("autumn DST", autumn_idx)]:
        df = pd.DataFrame(index=idx)
        out = append_weekly_fourier(df)

        for col in _DEFAULT_COLS:
            diffs = out[col].diff().dropna().abs()
            max_diff = diffs.max()
            assert max_diff < 0.15, (
                f"{label}: column {col!r} has a first-difference of {max_diff:.6f}, "
                f"exceeding the 0.15 continuity bound — possible DST jump in the Fourier output"
            )


# ---------------------------------------------------------------------------
# 7b. REGRESSION GUARD — microsecond-precision DatetimeIndex
#
# Bug fixed 2026-05-04 ("fourier-microsecond-precision" branch).  The
# Stage 3 / Stage 5 assembler writes parquet with
# ``timestamp[us, tz=UTC]``; when ``pyarrow`` round-trips that into
# pandas the resulting ``DatetimeIndex`` has *microsecond* precision,
# not nanosecond.  The pre-fix implementation used
# ``df.index.view("int64") // _NANOSECONDS_PER_HOUR`` which assumed
# nanosecond precision and silently divided microsecond timestamps by
# a constant 1000x too large — collapsing a year of hourly data onto
# just ~10 distinct integer ``t`` values, making every sin/cos column
# nearly constant.  Stage 8 ``ScipyParametricModel.fit`` (and any other
# caller of ``append_weekly_fourier`` against a parquet-loaded frame)
# saw a rank-deficient design matrix, ``curve_fit`` failed convergence
# with ``alpha`` in the millions and infinite covariance diagonals,
# and the user-facing scatter plot showed forecasts at -7 000 000 MW.
#
# All previous tests in this file used ``pd.date_range(...)`` which
# defaults to nanosecond precision — the bug was invisible to the
# suite.  The two tests below are the load-bearing regression guards:
# they exercise the precision the assembler actually produces.
# ---------------------------------------------------------------------------


def test_append_weekly_fourier_microsecond_precision_index() -> None:
    """Sin/cos columns must take many distinct values on a microsecond-precision index.

    A year of hourly data has 8760 rows.  A correctly-implemented
    weekly Fourier basis sees those 8760 distinct UTC timestamps and
    produces a sinusoid that, modulo period 168, samples ~52 full
    cycles — yielding ``len(np.unique(...)) ≈ 84``.  The pre-fix
    implementation collapsed the year onto only 10 distinct integer
    ``t`` values and therefore only 10 distinct sin/cos values.  The
    threshold in this test (>= 50 distinct rounded sin values) catches
    that collapse with margin to spare without being brittle on minor
    numerical-rounding shifts.
    """
    # Match what ``assembler.load_calendar`` returns: microsecond-
    # precision tz-aware UTC index.
    idx_us = pd.date_range("2024-01-01", periods=8760, freq="h", tz="UTC").as_unit("us")
    assert idx_us.dtype == "datetime64[us, UTC]", (
        f"Test fixture must be microsecond-precision; got {idx_us.dtype}"
    )
    df = pd.DataFrame({"value": np.arange(8760)}, index=idx_us)

    out = append_weekly_fourier(df, period_hours=168, harmonics=1, column_prefix="weekly")

    sin_arr = np.asarray(out["weekly_sin_k1"])
    distinct = len(np.unique(np.round(sin_arr, 8)))
    assert distinct >= 50, (
        f"weekly_sin_k1 produced only {distinct} distinct values across 8760 hours; "
        f"a correct sinusoid samples ~84 distinct values.  This is the "
        f"microsecond-precision regression — see fourier.py docstring."
    )
    # And the squared norm should be ~n/2 = 4380 for a properly-shaped sinusoid.
    sq_norm = float((sin_arr**2).sum())
    assert 4000.0 <= sq_norm <= 4800.0, (
        f"||weekly_sin_k1||² = {sq_norm:.1f}; expected ~4380 (= n/2 for a unit-amplitude "
        f"sinusoid over a year of hourly data).  Norm well outside that range indicates "
        f"the sin column has collapsed to near-constant values — the pre-fix "
        f"microsecond-precision regression."
    )


def test_append_weekly_fourier_precision_independent() -> None:
    """Same UTC timestamps -> same Fourier values regardless of index precision.

    pandas exposes 4 datetime resolutions (s / ms / us / ns).  The
    helper must compute identical sin/cos columns at every resolution
    because the underlying timestamps are identical.  This test pins
    that contract — a future regression that re-introduces a
    precision-dependent path (e.g. another ``view("int64")``) fails
    here loudly.
    """
    base = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")  # ns by default
    columns = ("weekly_sin_k1", "weekly_cos_k1", "weekly_sin_k2", "weekly_cos_k2")
    outs: dict[str, np.ndarray] = {}
    for unit in ("s", "ms", "us", "ns"):
        idx = base.as_unit(unit)
        df = pd.DataFrame({"x": 0.0}, index=idx)
        out = append_weekly_fourier(df, period_hours=168, harmonics=2, column_prefix="weekly")
        outs[unit] = np.column_stack([out[c].to_numpy() for c in columns])
    # Cross-precision equality: every cell within 1e-12 of the ns reference.
    reference = outs["ns"]
    for unit, arr in outs.items():
        np.testing.assert_allclose(
            arr,
            reference,
            atol=1e-12,
            err_msg=(
                f"Fourier columns at unit={unit!r} differ from nanosecond reference "
                f"by more than 1e-12.  The helper must be precision-independent."
            ),
        )


# ---------------------------------------------------------------------------
# 8. CLI smoke test
# ---------------------------------------------------------------------------


def test_fourier_cli_main_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Plan §Task T2 — _cli_main([]) returns 0 and mentions append_weekly_fourier."""
    return_code = _cli_main([])

    assert return_code == 0, f"Expected _cli_main([]) to return 0; got {return_code!r}"

    captured = capsys.readouterr()
    assert "append_weekly_fourier" in captured.out, (
        f"Expected _cli_main([]) stdout to mention 'append_weekly_fourier'; got: {captured.out!r}"
    )
