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
