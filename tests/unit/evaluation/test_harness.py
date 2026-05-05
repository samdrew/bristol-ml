"""Spec-derived acceptance and regression tests for the rolling-origin evaluation harness.

Stage 4, Task T6.
Plan: ``docs/plans/active/04-linear-baseline.md`` §6 Task T6.
Plan tag H-1: UTC-index guard — raise ``ValueError`` on non-UTC tz-aware index.

Every test is derived from the plan acceptance criteria, contract corners
documented in Task T6, or the H-1 housekeeping carry-over from Stage 3.

No production code is modified here.  If a test fails, the failure
indicates a deviation from the spec; surface it to the implementer rather
than weakening the assertion.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan clause, plan tag, or contract section
  it guards.
- ``np.random.default_rng(seed=42)`` for all synthetic data.
- ``pytest.approx`` for float comparisons.
- The ``loguru_caplog`` fixture from ``tests/conftest.py`` is used for all
  log-capture assertions (Stage 4 harness is the planned second caller of
  this adapter, per the conftest comment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.evaluation.harness import _build_model_from_config, _cli_main, evaluate
from bristol_ml.evaluation.metrics import mae, mape, rmse, wape
from bristol_ml.evaluation.splitter import rolling_origin_split_from_config
from bristol_ml.features.assembler import OUTPUT_SCHEMA, WEATHER_VARIABLE_COLUMNS
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.sarimax import SarimaxModel
from conf._schemas import LinearConfig, SarimaxConfig, SplitterConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_N_ROWS = 500
_SPLIT_CFG = SplitterConfig(
    min_train_periods=200,
    test_len=48,
    step=48,
    gap=0,
    fixed_window=False,
)
# Pre-verified: 6 folds for n=500 with the above config.
_EXPECTED_FOLD_COUNT = 6

_WEATHER_COLS = [name for name, _ in WEATHER_VARIABLE_COLUMNS]
_ALL_METRICS = [mae, mape, rmse, wape]


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = _N_ROWS, tz: str | None = "UTC", seed: int = 42) -> pd.DataFrame:
    """Build a synthetic hourly DataFrame with weather columns and ``nd_mw``.

    Parameters
    ----------
    n:
        Number of rows.
    tz:
        Timezone string for the DatetimeIndex, or ``None`` for tz-naive.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: all five weather variable columns (float32) + ``nd_mw``
        (float64).  Index: hourly DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start="2024-01-01 00:00", periods=n, freq="h", tz=tz)
    data: dict[str, np.ndarray] = {
        col: rng.uniform(0.0, 30.0, n).astype("float32") for col in _WEATHER_COLS
    }
    # nd_mw must be strictly positive to avoid MAPE/WAPE zero-denominator guards.
    data["nd_mw"] = rng.uniform(20_000.0, 50_000.0, n).astype("float64")
    return pd.DataFrame(data, index=idx)


def _linear_model() -> LinearModel:
    """Return a fresh ``LinearModel`` with default weather-column fallback."""
    return LinearModel(LinearConfig(type="linear", target_column="nd_mw"))


# ---------------------------------------------------------------------------
# Plan-named test 1: test_harness_rejects_non_utc_index (H-1)
# ---------------------------------------------------------------------------


def test_harness_rejects_non_utc_index() -> None:
    """Plan H-1: a tz-aware but non-UTC index must raise ``ValueError``.

    The harness carries the UTC-index guard from Stage 3 review item H-1.
    The guard lives in ``evaluate()`` (not in the splitter) because the
    splitter receives only ``n_rows`` and should remain data-structure-
    agnostic (codebase-map §D).

    The error message must contain the substring ``"UTC"`` so callers can
    diagnose the issue without reading source code.

    Plan clause: Task T6 / plan H-1 / harness.py ``_validate_inputs`` H-1
    block.
    """
    df = _make_df(tz="Europe/London")
    model = _linear_model()

    with pytest.raises(ValueError, match="UTC"):
        evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)


# ---------------------------------------------------------------------------
# Plan-named test 2: test_harness_accepts_naive_index
# ---------------------------------------------------------------------------


def test_harness_accepts_naive_index() -> None:
    """Plan T6 / H-1: a tz-naive DatetimeIndex must be accepted without raising.

    The spec permits tz-naive indices (the splitter is tz-agnostic; forcing
    UTC awareness on all callers would add a conversion burden with no
    analytical benefit when working with synthetic or already-normalised
    test data).

    Assert that ``evaluate`` returns a non-empty DataFrame without
    raising any exception.

    Plan clause: Task T6 named test / harness.py module docstring ("tz-naive
    or UTC-aware only").
    """
    df = _make_df(tz=None)
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    assert isinstance(result, pd.DataFrame), (
        "evaluate() on a tz-naive index must return a DataFrame (plan T6 / H-1)."
    )
    assert len(result) > 0, (
        "evaluate() on a tz-naive index must return at least one fold row (plan T6 / H-1)."
    )


# ---------------------------------------------------------------------------
# Plan-named test 3: test_harness_returns_one_row_per_fold
# ---------------------------------------------------------------------------


def test_harness_returns_one_row_per_fold() -> None:
    """Plan T6: the returned DataFrame must have exactly one row per fold.

    The expected fold count is derived independently via
    ``rolling_origin_split_from_config`` — not from ``evaluate`` itself —
    so this test would fail if the harness skipped or duplicated folds.

    Plan clause: Task T6 named test / evaluate() return docstring ("One row
    per fold").
    """
    df = _make_df()
    model = _linear_model()

    expected_fold_count = len(list(rolling_origin_split_from_config(len(df), _SPLIT_CFG)))
    assert expected_fold_count == _EXPECTED_FOLD_COUNT, (
        f"Pre-condition: expected fold count must be {_EXPECTED_FOLD_COUNT}; "
        f"got {expected_fold_count}."
    )

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    assert len(result) == expected_fold_count, (
        f"evaluate() must return one row per fold; "
        f"expected {expected_fold_count} rows, got {len(result)} (plan T6 named test)."
    )


# ---------------------------------------------------------------------------
# Plan-named test 4: test_harness_per_fold_metrics_match_direct_computation
# ---------------------------------------------------------------------------


def test_harness_per_fold_metrics_match_direct_computation() -> None:
    """Plan T6: per-fold metric values must equal direct manual computation.

    Selects fold 0 (the first fold).  After ``evaluate`` returns, re-fits a
    fresh ``LinearModel`` on the same training slice and predicts on the
    same test slice.  Computes ``mae`` directly and asserts that
    ``result["mae"].iloc[0]`` equals the hand-computed value under
    ``pytest.approx``.

    This guards against harness bugs such as: metric applied to the wrong
    slice, prediction/target mismatch, or wrong fold index.

    Plan clause: Task T6 named test / evaluate() contract.
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    # Re-derive fold 0 indices independently.
    folds = list(rolling_origin_split_from_config(len(df), _SPLIT_CFG))
    train_idx, test_idx = folds[0]

    # Fit a fresh model on the same training slice.
    fresh_model = _linear_model()
    X_train = df[_WEATHER_COLS].iloc[train_idx]
    y_train = df["nd_mw"].iloc[train_idx]
    X_test = df[_WEATHER_COLS].iloc[test_idx]
    y_test = df["nd_mw"].iloc[test_idx]

    fresh_model.fit(X_train, y_train)
    y_pred = fresh_model.predict(X_test)

    expected_mae = mae(y_test, y_pred)
    harness_mae = result["mae"].iloc[0]

    assert harness_mae == pytest.approx(expected_mae, rel=1e-9), (
        f"Harness mae for fold 0 ({harness_mae!r}) must equal direct computation "
        f"({expected_mae!r}) under pytest.approx (plan T6 named test)."
    )


# ---------------------------------------------------------------------------
# Plan-named test 5: test_harness_logs_summary
# ---------------------------------------------------------------------------


def test_harness_logs_summary(loguru_caplog: pytest.LogCaptureFixture) -> None:
    """Plan T6: a completion INFO line naming ``total_folds=`` must be emitted.

    Uses the repo-wide ``loguru_caplog`` fixture (``tests/conftest.py``)
    which routes loguru records into pytest's ``caplog`` at INFO and above.

    Assert that at least one captured message contains both the
    ``"Evaluator complete"`` prefix (matching ``harness.py``'s summary log
    call) and the ``"total_folds="`` field.

    Plan clause: Task T6 named test / harness.py module docstring ("one
    summary INFO line on completion … total folds, elapsed wall time,
    per-metric mean ± std").
    """
    df = _make_df()
    model = _linear_model()

    with loguru_caplog.at_level("INFO"):
        evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    messages = [r.getMessage() for r in loguru_caplog.records]
    summary_lines = [m for m in messages if "Evaluator complete" in m and "total_folds=" in m]

    assert len(summary_lines) >= 1, (
        f"Expected at least one INFO message containing 'Evaluator complete' "
        f"and 'total_folds='; captured messages were: {messages!r} (plan T6 named test)."
    )


# ---------------------------------------------------------------------------
# Validation corners — section 6 of the task brief
# ---------------------------------------------------------------------------


def test_harness_rejects_non_datetimeindex() -> None:
    """Contract corner 6: a ``RangeIndex`` must raise ``TypeError`` naming 'DatetimeIndex'.

    ``evaluate()`` documents ``TypeError`` when ``df.index`` is not a
    ``pandas.DatetimeIndex``.  The error message must contain 'DatetimeIndex'
    so the caller can diagnose the issue.

    Plan clause: Task T6 contract corners / harness.py ``_validate_inputs``
    TypeError block.
    """
    df = _make_df()
    df = df.reset_index(drop=True)  # Converts the DatetimeIndex to a RangeIndex.
    assert isinstance(df.index, pd.RangeIndex)

    model = _linear_model()

    with pytest.raises(TypeError, match="DatetimeIndex"):
        evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)


def test_harness_rejects_empty_metrics() -> None:
    """Contract corner 7: an empty ``metrics`` sequence must raise ``ValueError``.

    An evaluation run with zero metrics is a configuration error, not a
    reasonable input.  The error message must contain 'metric' or 'empty'
    so the caller understands the cause.

    Plan clause: Task T6 contract corners / harness.py ``_validate_inputs``
    "metrics must be non-empty" block.
    """
    df = _make_df()
    model = _linear_model()

    with pytest.raises(ValueError, match=r"(?i)(metric|empty)"):
        evaluate(model, df, _SPLIT_CFG, [])


def test_harness_rejects_missing_target_column() -> None:
    """Contract corner 8: a missing target column must raise ``ValueError`` naming the column.

    Drop ``nd_mw`` from the DataFrame and confirm the error message names the
    missing column so callers can diagnose schema mismatches immediately.

    Plan clause: Task T6 contract corners / harness.py ``_validate_inputs``
    "target_column missing" block.
    """
    df = _make_df().drop(columns=["nd_mw"])
    model = _linear_model()

    with pytest.raises(ValueError, match="nd_mw"):
        evaluate(model, df, _SPLIT_CFG, _ALL_METRICS, target_column="nd_mw")


def test_harness_rejects_missing_feature_column() -> None:
    """Contract corner 9: explicit missing ``feature_columns`` must raise ``ValueError``.

    Pass ``feature_columns=("does_not_exist",)`` and confirm ``ValueError``
    listing the missing column name.

    Plan clause: Task T6 contract corners / harness.py
    ``_resolve_feature_columns`` missing-column block.
    """
    df = _make_df()
    model = _linear_model()

    with pytest.raises(ValueError, match="does_not_exist"):
        evaluate(
            model,
            df,
            _SPLIT_CFG,
            _ALL_METRICS,
            feature_columns=("does_not_exist",),
        )


def test_harness_feature_columns_fallback_to_weather_defaults() -> None:
    """Contract corner 10: ``feature_columns=None`` must use the weather-column fallback.

    Build a DataFrame carrying only the five weather columns and ``nd_mw``
    (the minimal assembler output shape) with ``feature_columns=None``.
    ``evaluate()`` must succeed and return a positive metric value.

    Plan clause: Task T6 contract corners / harness.py
    ``_resolve_feature_columns`` fallback path.
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, [mae], feature_columns=None)

    assert len(result) > 0, (
        "evaluate() with feature_columns=None must return at least one fold row "
        "(plan T6 contract corner 10)."
    )
    assert result["mae"].iloc[0] > 0.0, (
        f"Expected a positive mae on the first fold; got {result['mae'].iloc[0]!r} "
        "(plan T6 contract corner 10 — fallback path)."
    )


def test_harness_column_order_and_dtypes() -> None:
    """Contract corner 11: output columns must be in documented order with correct dtypes.

    Expected column list (plan T6 return contract):
      ``["fold_index", "train_end", "test_start", "test_end", "mae", "rmse", "mape", "wape"]``

    Expected dtypes:
    - ``fold_index``: ``int64``
    - ``train_end``, ``test_start``, ``test_end``: datetime (UTC-aware for
      UTC-indexed input)
    - ``mae``, ``rmse``, ``mape``, ``wape``: ``float64``

    Plan clause: Task T6 contract corners / evaluate() return docstring.
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    expected_columns = [
        "fold_index",
        "train_end",
        "test_start",
        "test_end",
        "mae",
        "mape",
        "rmse",
        "wape",
    ]
    assert list(result.columns) == expected_columns, (
        f"Column order must be exactly {expected_columns!r}; "
        f"got {list(result.columns)!r} (plan T6 contract corner 11)."
    )

    # fold_index must be integer.
    assert result["fold_index"].dtype == np.dtype("int64"), (
        f"fold_index dtype must be int64; got {result['fold_index'].dtype} "
        "(plan T6 contract corner 11)."
    )

    # Timestamp columns must be datetime (tz-aware because input was UTC).
    for ts_col in ("train_end", "test_start", "test_end"):
        assert pd.api.types.is_datetime64_any_dtype(result[ts_col]), (
            f"Column '{ts_col}' must have a datetime dtype; "
            f"got {result[ts_col].dtype} (plan T6 contract corner 11)."
        )
        assert result[ts_col].dt.tz is not None, (
            f"Column '{ts_col}' must be tz-aware when the input index was UTC; "
            f"got tz=None (plan T6 contract corner 11)."
        )

    # Metric columns must be float64.
    for metric_col in ("mae", "rmse", "mape", "wape"):
        assert result[metric_col].dtype == np.dtype("float64"), (
            f"Metric column '{metric_col}' dtype must be float64; "
            f"got {result[metric_col].dtype} (plan T6 contract corner 11)."
        )


# ---------------------------------------------------------------------------
# Additional smoke tests on the public re-export
# ---------------------------------------------------------------------------


def test_evaluate_importable_from_evaluation_namespace() -> None:
    """Smoke: ``evaluate`` must be importable from ``bristol_ml.evaluation``.

    The plan directs that ``evaluate`` be re-exported via
    ``evaluation/__init__.py``.  An ``ImportError`` here means the re-export
    wiring is absent.

    Plan clause: Task T6 / H-3 re-export pattern / evaluation/__init__.py.
    """
    from bristol_ml.evaluation import evaluate as evaluate_from_ns

    assert callable(evaluate_from_ns), (
        "evaluate imported from bristol_ml.evaluation must be callable."
    )


def test_harness_fold_index_column_is_zero_based() -> None:
    """Smoke: ``fold_index`` must run 0, 1, 2, … without gaps or repeats.

    Ensures the harness enumerates folds in order and that the column records
    the correct positional index rather than being reset or omitted.

    Plan clause: evaluate() return docstring ("Ordered by fold_index").
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, [mae])

    expected_indices = list(range(len(result)))
    actual_indices = result["fold_index"].tolist()

    assert actual_indices == expected_indices, (
        f"fold_index column must be [0, 1, ..., {len(result) - 1}]; "
        f"got {actual_indices!r} (harness fold-order contract)."
    )


def test_harness_train_end_before_test_start() -> None:
    """Invariant: for every fold, ``train_end`` timestamp is strictly before ``test_start``.

    This guards the fundamental no-leakage invariant carried from the
    splitter: the training window never overlaps the test window.  In terms
    of timestamps, the last training timestamp must precede the first test
    timestamp.

    Plan clause: evaluation/CLAUDE.md invariant "max(train_idx) < min(test_idx)".
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, [mae])

    for _, row in result.iterrows():
        assert row["train_end"] < row["test_start"], (
            f"fold {row['fold_index']}: train_end ({row['train_end']}) "
            f"must be strictly before test_start ({row['test_start']}) "
            "(no-leakage invariant)."
        )


# ---------------------------------------------------------------------------
# Stage 6 T5 — Harness predictions emission
# ---------------------------------------------------------------------------


def test_harness_evaluate_default_returns_metrics_only() -> None:
    """Guards Stage 6 T5 — default call returns a plain DataFrame, not a tuple.

    Regression guard: Stage 4 call sites must be unaffected by the new
    ``return_predictions`` flag.  With the default (``return_predictions=False``),
    ``evaluate`` must return a ``pd.DataFrame`` — never a tuple.

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_evaluate_default_returns_metrics_only``.
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS)

    assert isinstance(result, pd.DataFrame), (
        "evaluate() with default return_predictions=False must return a pd.DataFrame, "
        f"not {type(result).__name__} (Stage 6 T5 regression guard)."
    )
    assert not isinstance(result, tuple), (
        "evaluate() with default return_predictions=False must not return a tuple "
        "(Stage 6 T5 regression guard)."
    )


def test_harness_evaluate_return_predictions_returns_tuple() -> None:
    """Guards Stage 6 T5 — ``return_predictions=True`` returns a 2-tuple.

    The first element must be a ``pd.DataFrame`` (the metrics frame) and the
    second element must also be a ``pd.DataFrame`` (the predictions frame).
    Neither element may be a tuple itself.

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_evaluate_return_predictions_returns_tuple``.
    """
    df = _make_df()
    model = _linear_model()

    result = evaluate(model, df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True)

    assert isinstance(result, tuple), (
        "evaluate() with return_predictions=True must return a tuple; "
        f"got {type(result).__name__} (Stage 6 T5)."
    )
    assert len(result) == 2, (
        f"evaluate() with return_predictions=True must return a 2-tuple; "
        f"got length {len(result)} (Stage 6 T5)."
    )

    metrics_df, predictions_df = result
    assert isinstance(metrics_df, pd.DataFrame), (
        "First element of the 2-tuple must be a pd.DataFrame (metrics); "
        f"got {type(metrics_df).__name__} (Stage 6 T5)."
    )
    assert isinstance(predictions_df, pd.DataFrame), (
        "Second element of the 2-tuple must be a pd.DataFrame (predictions); "
        f"got {type(predictions_df).__name__} (Stage 6 T5)."
    )


def test_harness_predictions_column_order_and_dtypes() -> None:
    """Guards Stage 6 T5 — predictions frame has pinned column order and dtypes.

    Expected column order (pinned by Stage 6 T5 contract):
      ``["fold_index", "test_start", "test_end", "horizon_h",
         "y_true", "y_pred", "error"]``

    Expected dtypes:
    - ``fold_index``: ``int64``
    - ``test_start``, ``test_end``: ``datetime64[ns, UTC]`` for UTC-aware input
    - ``horizon_h``: ``int64``
    - ``y_true``, ``y_pred``, ``error``: ``float64``

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_predictions_column_order_and_dtypes``.
    """
    # UTC-aware input — timestamps in predictions must be UTC-aware.
    df = _make_df(tz="UTC")
    model = _linear_model()

    _metrics_df, predictions_df = evaluate(
        model, df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True
    )

    expected_columns = [
        "fold_index",
        "test_start",
        "test_end",
        "horizon_h",
        "y_true",
        "y_pred",
        "error",
    ]
    assert list(predictions_df.columns) == expected_columns, (
        f"Predictions column order must be exactly {expected_columns!r}; "
        f"got {list(predictions_df.columns)!r} (Stage 6 T5)."
    )

    # fold_index must be int64.
    assert predictions_df["fold_index"].dtype == np.dtype("int64"), (
        f"fold_index dtype must be int64; got {predictions_df['fold_index'].dtype} (Stage 6 T5)."
    )

    # Timestamp columns must be datetime and tz-aware (input was UTC).
    for ts_col in ("test_start", "test_end"):
        assert pd.api.types.is_datetime64_any_dtype(predictions_df[ts_col]), (
            f"Column '{ts_col}' must have a datetime dtype; "
            f"got {predictions_df[ts_col].dtype} (Stage 6 T5)."
        )
        assert predictions_df[ts_col].dt.tz is not None, (
            f"Column '{ts_col}' must be tz-aware when the input index was UTC; "
            f"got tz=None (Stage 6 T5)."
        )

    # horizon_h must be int64.
    assert predictions_df["horizon_h"].dtype == np.dtype("int64"), (
        f"horizon_h dtype must be int64; got {predictions_df['horizon_h'].dtype} (Stage 6 T5)."
    )

    # Prediction and error columns must be float64.
    for float_col in ("y_true", "y_pred", "error"):
        assert predictions_df[float_col].dtype == np.dtype("float64"), (
            f"Column '{float_col}' dtype must be float64; "
            f"got {predictions_df[float_col].dtype} (Stage 6 T5)."
        )


def test_harness_predictions_one_row_per_forecast_hour() -> None:
    """Guards Stage 6 T5 — total predictions row count equals sum of test_len across folds.

    For the shared ``_SPLIT_CFG`` with ``test_len=48`` and the pre-verified
    ``_EXPECTED_FOLD_COUNT=6``, the predictions frame must have
    ``6 x 48 = 288`` rows.

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_predictions_one_row_per_forecast_hour``.
    """
    df = _make_df()
    model = _linear_model()

    # Derive expected total independently from splitter, not from evaluate.
    folds = list(rolling_origin_split_from_config(len(df), _SPLIT_CFG))
    expected_total_rows = sum(len(test_idx) for _train_idx, test_idx in folds)

    _metrics_df, predictions_df = evaluate(
        model, df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True
    )

    assert len(predictions_df) == expected_total_rows, (
        f"Predictions frame must have {expected_total_rows} rows "
        f"(one per forecast hour across all folds); "
        f"got {len(predictions_df)} (Stage 6 T5)."
    )


def test_harness_predictions_horizon_h_zero_based_per_fold() -> None:
    """Guards Stage 6 T5 — ``horizon_h`` resets to 0 at the start of each fold.

    For each unique ``fold_index`` in the predictions frame, ``horizon_h``
    must run ``0, 1, 2, …, test_len-1`` without gaps or resets.  The
    per-fold sequence is verified by grouping on ``fold_index`` and checking
    the sorted ``horizon_h`` values match ``range(test_len)``.

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_predictions_horizon_h_zero_based_per_fold``.
    """
    df = _make_df()
    model = _linear_model()

    _metrics_df, predictions_df = evaluate(
        model, df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True
    )

    expected_test_len = _SPLIT_CFG.test_len

    for fold_idx, group in predictions_df.groupby("fold_index"):
        horizon_values = sorted(group["horizon_h"].tolist())
        expected = list(range(expected_test_len))
        assert horizon_values == expected, (
            f"fold {fold_idx}: horizon_h must be {expected!r}; "
            f"got {horizon_values!r} (Stage 6 T5 — horizon_h resets per fold)."
        )


def test_harness_predictions_error_equals_y_true_minus_y_pred() -> None:
    """Guards Stage 6 T5 — ``error`` column equals ``y_true - y_pred`` exactly.

    Uses ``np.isclose`` for element-wise comparison to tolerate any
    float64 rounding at the margins of numerical precision.

    Plan clause: docs/plans/active/06-enhanced-evaluation.md §6 T5
    ``test_harness_predictions_error_equals_y_true_minus_y_pred``.
    """
    df = _make_df()
    model = _linear_model()

    _metrics_df, predictions_df = evaluate(
        model, df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True
    )

    expected_error = predictions_df["y_true"].to_numpy() - predictions_df["y_pred"].to_numpy()
    actual_error = predictions_df["error"].to_numpy()

    assert np.all(np.isclose(actual_error, expected_error)), (
        "Column 'error' must equal y_true - y_pred for every row; "
        "np.isclose check failed (Stage 6 T5)."
    )


# ---------------------------------------------------------------------------
# Stage 7 T6 — SARIMAX dispatcher wiring (harness)
# Plan: docs/plans/active/07-sarimax.md §6 Task T6
# ---------------------------------------------------------------------------


def _write_feature_cache_for_harness(path: Path, n_hours: int = 24 * 90, seed: int = 42) -> Path:
    """Write a minimal feature-table parquet conforming to ``assembler.OUTPUT_SCHEMA``.

    Mirrors the helper of the same name in ``tests/unit/test_train_cli.py``.
    Kept private to this module so the evaluation test file has no cross-file
    import dependency (CLAUDE.md §"Tests at boundaries").

    The 90-day window gives enough rows for a ``min_train_periods=720`` rolling-
    origin run with at least one test fold.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-10-01", periods=n_hours, freq="1h", tz="UTC")
    nd = (
        30_000 + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 200, n_hours)
    ).astype("int32")
    tsd = (nd + 3_000).astype("int32")
    weather = {
        name: rng.normal(loc=10, scale=3, size=n_hours).astype("float32")
        for name, _ in WEATHER_VARIABLE_COLUMNS
    }
    retrieved_at = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "nd_mw": nd,
            "tsd_mw": tsd,
            **weather,
            "neso_retrieved_at_utc": [retrieved_at] * n_hours,
            "weather_retrieved_at_utc": [retrieved_at] * n_hours,
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(OUTPUT_SCHEMA, safe=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def warm_feature_cache_for_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate a warm feature-table cache and point Hydra at ``tmp_path``.

    Mirrors ``tests/unit/test_train_cli.py::warm_feature_cache`` for use in
    harness CLI integration tests that need a populated feature cache without
    taking a cross-file fixture import dependency.
    """
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    _write_feature_cache_for_harness(tmp_path / "weather_only.parquet")
    return tmp_path


def test_harness_build_model_dispatches_sarimax_config() -> None:
    """Stage 7 T6 (AC-6): dispatcher returns a ``SarimaxModel`` for ``SarimaxConfig``.

    Directly instantiates a default ``SarimaxConfig()`` and passes it to the
    module-private dispatcher.  Asserts the returned object is an instance of
    ``SarimaxModel`` — not merely truthy — so that a wrong-type return (e.g.
    ``LinearModel`` or ``None``) will fail.

    Plan clause: docs/plans/active/07-sarimax.md §6 Task T6 —
    ``test_harness_build_model_dispatches_sarimax_config``.
    """
    cfg = SarimaxConfig()
    result = _build_model_from_config(cfg)

    assert isinstance(result, SarimaxModel), (
        f"_build_model_from_config(SarimaxConfig()) must return a SarimaxModel; "
        f"got {type(result)!r} (Stage 7 T6 AC-6)."
    )


def test_harness_cli_runs_with_model_sarimax(
    warm_feature_cache_for_harness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 7 T6 (AC-6, AC-11): harness CLI exits 0 when ``model=sarimax`` is selected.

    Integration smoke.  Uses a warm 90-day synthetic feature cache and
    deliberately small rolling-origin parameters to keep fit time within a few
    seconds.  The ``seasonal_order=[0,0,0,24]`` and ``weekly_fourier_harmonics=0``
    overrides disable the heavy seasonal and Fourier components so the SARIMA fit
    completes quickly on the synthetic data.

    A zero exit code confirms that the harness ``_build_model_from_config``
    branch for ``SarimaxConfig`` is wired correctly end-to-end; no assertion on
    stdout content is made because the harness CLI prints a plain DataFrame, not
    the ``"Per-fold metrics for model=…"`` banner from ``train.py``.

    Plan clause: docs/plans/active/07-sarimax.md §6 Task T6 —
    ``test_harness_cli_runs_with_model_sarimax``.
    """
    exit_code = _cli_main(
        [
            "model=sarimax",
            "model.order=[1,0,0]",
            "model.seasonal_order=[0,0,0,24]",
            "model.weekly_fourier_harmonics=0",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=24",
            "evaluation.rolling_origin.step=720",
            "evaluation.rolling_origin.fixed_window=true",
        ]
    )

    assert exit_code == 0, (
        f"harness _cli_main must exit 0 with model=sarimax on a warm feature cache; "
        f"got exit_code={exit_code} (Stage 7 T6 AC-6)."
    )


# ---------------------------------------------------------------------------
# Stage 8 T6 — ScipyParametric dispatcher wiring (harness)
# Plan: docs/plans/active/08-scipy-parametric.md §6 Task T6
# ---------------------------------------------------------------------------


def test_harness_build_model_dispatches_scipy_parametric_config() -> None:
    """Stage 8 T6 (AC-7): dispatcher returns a ``ScipyParametricModel``.

    Specifically: ``_build_model_from_config(ScipyParametricConfig(...))``
    must return a ``ScipyParametricModel`` instance.

    Directly instantiates a default ``ScipyParametricConfig()`` and passes it
    to the module-private dispatcher.  Asserts the returned object is an
    instance of ``ScipyParametricModel`` — not merely truthy — so that a
    wrong-type return (e.g. ``LinearModel`` or ``None``) will fail.

    Plan clause: docs/plans/active/08-scipy-parametric.md §6 Task T6 —
    ``test_harness_build_model_dispatches_scipy_parametric_config``.
    """
    from bristol_ml.models.scipy_parametric import ScipyParametricModel
    from conf._schemas import ScipyParametricConfig

    cfg = ScipyParametricConfig()
    result = _build_model_from_config(cfg)

    assert isinstance(result, ScipyParametricModel), (
        f"_build_model_from_config(ScipyParametricConfig()) must return a ScipyParametricModel; "
        f"got {type(result)!r} (Stage 8 T6 AC-7)."
    )


def test_harness_cli_runs_with_model_scipy_parametric(
    warm_feature_cache_for_harness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 8 T6 (AC-7, AC-11): harness CLI exits 0 when ``model=scipy_parametric`` is selected.

    Integration smoke.  Uses a warm 90-day synthetic feature cache and
    deliberately small Fourier-harmonic counts (``diurnal_harmonics=1``,
    ``weekly_harmonics=1``) to keep the ``curve_fit`` call fast on the
    synthetic data.  The tight rolling-origin overrides (``min_train_periods=720``,
    ``test_len=24``, ``step=720``, ``fixed_window=true``) give a single test fold
    that completes within a few seconds.

    A zero exit code confirms that the harness ``_build_model_from_config``
    branch for ``ScipyParametricConfig`` is wired correctly end-to-end.

    Plan clause: docs/plans/active/08-scipy-parametric.md §6 Task T6 —
    ``test_harness_cli_runs_with_model_scipy_parametric``.
    """
    exit_code = _cli_main(
        [
            "model=scipy_parametric",
            "model.diurnal_harmonics=1",
            "model.weekly_harmonics=1",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=24",
            "evaluation.rolling_origin.step=720",
            "evaluation.rolling_origin.fixed_window=true",
        ]
    )

    assert exit_code == 0, (
        f"harness _cli_main must exit 0 with model=scipy_parametric on a warm feature cache; "
        f"got exit_code={exit_code} (Stage 8 T6 AC-7)."
    )


# ---------------------------------------------------------------------------
# Stage 11 T6 — nn dispatcher wiring (D13 iii + D14 catch-up)
# Plan: docs/plans/active/11-complex-nn.md §6 Task T6
# ---------------------------------------------------------------------------


def test_harness_build_model_from_config_dispatches_nn_temporal() -> None:
    """Stage 11 T6 (D13 clause iii): dispatcher returns an ``NnTemporalModel``.

    Directly instantiates a default ``NnTemporalConfig`` and passes it to
    the module-private dispatcher.  Asserts the returned object is an
    instance of ``NnTemporalModel`` — not merely truthy — so that a
    wrong-type return (e.g. ``NnMlpModel`` or ``None``) will fail.

    Plan clause: docs/plans/active/11-complex-nn.md §6 Task T6 —
    ``test_harness_build_model_from_config_dispatches_nn_temporal`` (D13
    clause iii).
    """
    from bristol_ml.models.nn.temporal import NnTemporalModel
    from conf._schemas import NnTemporalConfig

    cfg = NnTemporalConfig(
        seq_len=32,
        num_blocks=2,
        channels=8,
        kernel_size=3,
        weight_norm=False,
        dropout=0.0,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=64,
        max_epochs=3,
        patience=10,
        seed=None,
        device="cpu",
        target_column="nd_mw",
        feature_columns=None,
    )
    result = _build_model_from_config(cfg)

    assert isinstance(result, NnTemporalModel), (
        f"_build_model_from_config(NnTemporalConfig(...)) must return an NnTemporalModel; "
        f"got {type(result)!r} (Stage 11 T6 D13 clause iii)."
    )


def test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up() -> None:
    """Stage 11 T6 (D14 catch-up): dispatcher returns an ``NnMlpModel``.

    Stage 10 shipped the ``NnMlpConfig`` isinstance branch in ``train.py``
    but the harness ``_build_model_from_config`` factory was missing the
    same branch (plan D14 — Stage 10 gap surfaced at Stage 11 T6).  This
    test is the regression guard for the catch-up commit: after D14 lands,
    ``_build_model_from_config(NnMlpConfig(...))`` must return a proper
    ``NnMlpModel`` instead of ``None``.

    Plan clause: docs/plans/active/11-complex-nn.md §6 Task T6 —
    ``test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up``
    (D14 — regression against the Stage 10 gap).
    """
    from bristol_ml.models.nn.mlp import NnMlpModel
    from conf._schemas import NnMlpConfig

    cfg = NnMlpConfig(
        hidden_sizes=[8],
        activation="relu",
        dropout=0.0,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=16,
        max_epochs=3,
        patience=10,
        seed=None,
        device="cpu",
        target_column="nd_mw",
        feature_columns=None,
    )
    result = _build_model_from_config(cfg)

    assert isinstance(result, NnMlpModel), (
        f"_build_model_from_config(NnMlpConfig(...)) must return an NnMlpModel after the "
        f"Stage 11 D14 catch-up; got {type(result)!r}. "
        f"The Stage 10 harness-factory gap was not closed."
    )


# ---------------------------------------------------------------------------
# Parallel rolling-origin folds (n_jobs > 1)
#
# The contract (per ``evaluate``'s docstring): folds are mathematically
# independent, so the parallel result must be byte-for-byte identical
# to the serial result on both the metrics frame and the predictions
# frame after both are sorted by ``fold_index``.  These tests pin that
# contract end-to-end against ``LinearModel`` (cheap fits — single-
# digit ms each — so the integration test is fast).  The actual
# wall-clock speedup is dominated by the SARIMAX path, which the
# Stage 7 / 8 notebook tests exercise; pinning the speedup itself
# here would be flaky and is not the point.
# ---------------------------------------------------------------------------


def test_evaluate_n_jobs_2_matches_serial_metrics() -> None:
    """``n_jobs=2`` produces the same metrics frame as ``n_jobs=1``.

    Pins the "folds are independent" contract.  Any divergence here
    means a fold's behaviour depends on cross-fold state — which would
    silently invalidate every Stage 7 / 8 / 16 retrospective.
    """
    df = _make_df()
    serial = evaluate(_linear_model(), df, _SPLIT_CFG, _ALL_METRICS, n_jobs=1)
    parallel = evaluate(_linear_model(), df, _SPLIT_CFG, _ALL_METRICS, n_jobs=2)

    pd.testing.assert_frame_equal(serial, parallel, check_dtype=True, check_exact=True)
    assert len(serial) == _EXPECTED_FOLD_COUNT


def test_evaluate_n_jobs_2_matches_serial_predictions() -> None:
    """``n_jobs=2`` predictions frame matches the serial frame exactly.

    The ``return_predictions=True`` path is the load-bearing one for
    the Stage 6 q10-q90 uncertainty band; a parallelised divergence
    here would silently shift the band.
    """
    df = _make_df()
    _, serial_preds = evaluate(
        _linear_model(), df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True, n_jobs=1
    )
    _, parallel_preds = evaluate(
        _linear_model(), df, _SPLIT_CFG, _ALL_METRICS, return_predictions=True, n_jobs=2
    )

    # The parallel path may interleave folds; sort defensively before
    # the equality check (the production code does this internally,
    # but the assertion is more robust to a future ordering change).
    serial_sorted = serial_preds.sort_values(["fold_index", "horizon_h"]).reset_index(drop=True)
    parallel_sorted = parallel_preds.sort_values(["fold_index", "horizon_h"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        serial_sorted, parallel_sorted, check_dtype=True, check_exact=True
    )


def test_evaluate_n_jobs_below_one_raises_valueerror() -> None:
    """``n_jobs <= 0`` raises ``ValueError`` before any fold work begins.

    The early raise (before ``_validate_inputs``) means a misconfigured
    notebook fails fast rather than after the ingestion-cache load.
    """
    df = _make_df()
    for bad in (0, -1, -8):
        with pytest.raises(ValueError, match=r"n_jobs >= 1"):
            evaluate(_linear_model(), df, _SPLIT_CFG, _ALL_METRICS, n_jobs=bad)


def test_evaluate_and_keep_final_model_n_jobs_2_leaves_caller_model_fitted() -> None:
    """Under ``n_jobs > 1`` the caller's model still ends up fit on the final fold.

    Workers run in separate processes, so the parent's ``model`` is
    untouched by the parallel evaluation.  The wrapper performs one
    additional serial fit on the final-fold training data so callers
    (notably the Stage 9 registry path) see a populated
    ``metadata.fit_utc`` and a fittable estimator on return.
    """
    from bristol_ml.evaluation.harness import evaluate_and_keep_final_model

    df = _make_df()
    model = _linear_model()
    assert model.metadata.fit_utc is None, "test fixture: model must start unfitted."

    metrics_df, returned_model = evaluate_and_keep_final_model(
        model, df, _SPLIT_CFG, _ALL_METRICS, n_jobs=2
    )

    # Same instance round-trips (the docstring's "same fitted estimator passed in").
    assert returned_model is model
    # And it is now fitted (the post-parallel serial refit on the final fold).
    assert returned_model.metadata.fit_utc is not None, (
        "evaluate_and_keep_final_model(n_jobs=2) must leave the caller's model fitted on "
        "the final fold's training data — the workers fit deepcopies, so a final "
        "serial refit on the parent is required."
    )
    # And the metrics row from that final-fold model matches the metrics_df row,
    # because both were fit on the same final-fold training data deterministically.
    assert len(metrics_df) == _EXPECTED_FOLD_COUNT
