"""Spec-derived tests for ``bristol_ml.evaluation.metrics``.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T5 (named tests 1-9
  and contract-corner tests 10-16).
- Plan decisions D8 (MAPE zero-denominator = raise ``ValueError``) and D9
  (WAPE formula = Σ|y-ŷ|/Σ|y|, Kolassa & Schütz 2007 / Hyndman form).
- ``docs/intent/DESIGN.md`` §5.3 (metric definitions and fraction convention).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each docstring cites the plan clause, AC, D-number, or F-number it guards.
- ``pytest.parametrize`` is used for tests covering all four metrics and
  for input-type variants; parametrised by ``(name, fn)`` pairs so test
  output is readable.
- ``pytest.approx(…, rel=1e-12)`` is used only where ``sqrt`` or other
  transcendental arithmetic genuinely introduces float error.  For
  MAE / MAPE / WAPE on small integer fixtures the result is exactly
  representable in IEEE 754 and strict ``==`` is used.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

metrics_mod = pytest.importorskip("bristol_ml.evaluation.metrics")

mae = metrics_mod.mae
mape = metrics_mod.mape
rmse = metrics_mod.rmse
wape = metrics_mod.wape
METRIC_REGISTRY = metrics_mod.METRIC_REGISTRY
MetricFn = metrics_mod.MetricFn


# ---------------------------------------------------------------------------
# Convenience fixture: (name, fn) pairs for all four metrics
# ---------------------------------------------------------------------------

_ALL_METRICS: list[tuple[str, MetricFn]] = [
    ("mae", mae),
    ("rmse", rmse),
    ("mape", mape),
    ("wape", wape),
]

# Perfect-prediction fixtures that keep all denominators non-zero.
# y_pred == y_true so every metric must return exactly 0.0.
_PERFECT_Y: list[float] = [10.0, 20.0, 30.0]
_PERFECT_P: list[float] = [10.0, 20.0, 30.0]

# Generic valid fixture re-used across input-type and registry tests.
# mae([10,20,30],[10,20,30]) == 0.0 (trivially correct).
_GENERIC_Y: list[float] = [10.0, 20.0, 30.0]
_GENERIC_P: list[float] = [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Test 1 — MAE hand-computed (Plan T5)
# ---------------------------------------------------------------------------


def test_mae_hand_computed() -> None:
    """Guards plan T5 test 1 / AC-4 / F-14: MAE formula.

    Contract: MAE = mean(|y_true - y_pred|).
    Fixture:  y_true=[1,2,3], y_pred=[2,2,2].
    Errors:   |1-2|=1, |2-2|=0, |3-2|=1.
    Expected: (1 + 0 + 1) / 3 = 2/3.

    The result is exactly representable in IEEE 754 double precision (two
    of the three error terms are exact integers, and the mean of 2 over 3
    is a recurring decimal).  Use ``pytest.approx`` because floating-point
    division produces a non-terminating binary fraction.
    """
    result = mae([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert result == pytest.approx(2 / 3, rel=1e-12), (
        f"mae([1,2,3],[2,2,2]) must equal 2/3 ≈ {2 / 3:.17f}; got {result!r} (plan T5 test 1)."
    )


# ---------------------------------------------------------------------------
# Test 2 — RMSE hand-computed (Plan T5)
# ---------------------------------------------------------------------------


def test_rmse_hand_computed() -> None:
    """Guards plan T5 test 2 / AC-4 / F-14: RMSE formula.

    Contract: RMSE = sqrt(mean((y_true - y_pred)**2)).
    Fixture:  y_true=[1,2,3], y_pred=[2,2,2].
    Squared errors: 1, 0, 1.  Mean = 2/3.
    Expected: sqrt(2/3).

    ``sqrt`` is a transcendental function; ``pytest.approx`` with a tight
    relative tolerance is appropriate here.
    """
    expected = math.sqrt(2 / 3)
    result = rmse([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert result == pytest.approx(expected, rel=1e-12), (
        f"rmse([1,2,3],[2,2,2]) must equal sqrt(2/3) ≈ {expected:.17f}; "
        f"got {result!r} (plan T5 test 2)."
    )


# ---------------------------------------------------------------------------
# Test 3 — MAPE hand-computed (Plan T5)
# ---------------------------------------------------------------------------


def test_mape_hand_computed() -> None:
    """Guards plan T5 test 3 / AC-4 / F-14 / D8: MAPE formula (fraction form).

    Contract: MAPE = mean(|(y_true - y_pred) / y_true|).
    Returned as a FRACTION, not a percentage (plan D8 / DESIGN §5.3).

    Fixture:  y_true=[2,4,10], y_pred=[1,2,5].
    Per-element: |2-1|/2 = 0.5, |4-2|/4 = 0.5, |10-5|/10 = 0.5.
    Expected: mean([0.5, 0.5, 0.5]) = 0.5.

    Each element is exactly representable; mean of identical values is
    exact; strict equality is appropriate.
    """
    result = mape([2.0, 4.0, 10.0], [1.0, 2.0, 5.0])
    assert result == 0.5, (
        f"mape([2,4,10],[1,2,5]) must equal 0.5 (fraction, not %); "
        f"got {result!r} (plan T5 test 3, D8 fraction convention)."
    )


# ---------------------------------------------------------------------------
# Test 4 — MAPE raises on zero target (Plan T5 / D8)
# ---------------------------------------------------------------------------


def test_mape_raises_on_zero_target() -> None:
    """Guards plan T5 test 4 / D8: MAPE must raise ValueError for zero y_true.

    Per plan D8: ``ValueError("MAPE is undefined when y_true contains zeros")``.
    GB national demand never approaches zero in practice; the guard protects
    callers from silent ``inf`` poisoning when the function is applied to
    out-of-scope data.

    Tests both a single zero in an otherwise non-zero array and an array
    that is entirely zero.
    """
    # Single zero among non-zero values.
    with pytest.raises(ValueError, match=r"(?i)(MAPE is undefined|y_true contains zeros)"):
        mape([1.0, 0.0, 3.0], [1.0, 1.0, 3.0])

    # Entire array of zeros.
    with pytest.raises(ValueError, match=r"(?i)(MAPE is undefined|y_true contains zeros)"):
        mape([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Test 5 — WAPE hand-computed (Plan T5 / D9)
# ---------------------------------------------------------------------------


def test_wape_hand_computed() -> None:
    """Guards plan T5 test 5 / D9 / AC-4 / F-14: WAPE Kolassa-form formula.

    Contract (plan D9): WAPE = sum(|y_true - y_pred|) / sum(|y_true|).
    This is the Kolassa & Schütz (2007) / Hyndman form; it is NOT the
    mean form (mean(|err|/|y|)), which is just MAPE under a different name.

    Fixture:  y_true=[1,2,3], y_pred=[2,2,2].
    Numerator:   |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2.
    Denominator: |1| + |2| + |3| = 6.
    Expected: 2 / 6 = 1/3.

    Proof that this differs from the mean form:
      mean(|err|/|y|) = mean([1, 0, 1/3]) = (1 + 0 + 1/3)/3 = 4/9 ≠ 1/3.

    1/3 is a recurring decimal; ``pytest.approx`` is appropriate.
    """
    result = wape([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert result == pytest.approx(1 / 3, rel=1e-12), (
        f"wape([1,2,3],[2,2,2]) must equal 1/3 ≈ {1 / 3:.17f} (Kolassa Σ-form); "
        f"got {result!r}.  "
        "If this returns 4/9 the implementation is using the mean form, not D9."
    )


# ---------------------------------------------------------------------------
# Test 6 — WAPE raises on zero-sum target (Plan T5 / D9)
# ---------------------------------------------------------------------------


def test_wape_raises_on_zero_sum_target() -> None:
    """Guards plan T5 test 6 / D9: WAPE must raise ValueError when sum(|y_true|)==0.

    Per plan D9 the denominator guard fires when the aggregate magnitude of
    the actuals is zero.  The canonical case is all-zero y_true.
    """
    with pytest.raises(ValueError, match=r"(?i)(WAPE is undefined|sum.*y_true.*== 0)"):
        wape([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Test 7 — NaN inputs rejected (Plan T5) — parametrised over all four metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metric_rejects_nan(name: str, fn: MetricFn) -> None:
    """Guards plan T5 test 7: NaN in either input array raises ValueError.

    The shared guard ``_coerce_and_validate`` must catch NaN in both
    ``y_true`` and ``y_pred`` before any metric kernel executes; silent
    ``nan`` propagation would poison fold-level metric tables.

    Parametrised over all four metrics (plan T5 names the guard as shared).
    """
    # NaN in y_true.
    with pytest.raises(ValueError):
        fn([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])

    # NaN in y_pred.
    with pytest.raises(ValueError):
        fn([1.0, 2.0, 3.0], [1.0, float("nan"), 3.0])


# ---------------------------------------------------------------------------
# Test 8 — Length mismatch rejected (Plan T5) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metric_rejects_length_mismatch(name: str, fn: MetricFn) -> None:
    """Guards plan T5 test 8: mismatched y_true / y_pred lengths raise ValueError.

    The shared guard ``_coerce_and_validate`` must detect the shape
    discrepancy before the kernel runs; a silently truncated computation
    would produce a metric for a different problem size.
    """
    with pytest.raises(ValueError, match=r"(?i)(length|shape|same)"):
        fn([1.0, 2.0, 3.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Test 9 — Perfect prediction returns 0.0 (Plan T5) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_perfect_prediction_all_zero(name: str, fn: MetricFn) -> None:
    """Guards plan T5 test 9 / AC-4: y_pred == y_true yields metric == 0.0.

    Uses y=[10,20,30] so that MAPE and WAPE denominators are non-zero.
    When y_pred equals y_true exactly, every error term is zero regardless
    of the formula; strict equality is required.
    """
    result = fn(_PERFECT_Y, _PERFECT_P)
    assert result == 0.0, (
        f"{name}(y, y) must be exactly 0.0 when y_pred == y_true; got {result!r} (plan T5 test 9)."
    )


# ---------------------------------------------------------------------------
# Test 10 — list[float] input accepted (Contract corner) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metrics_accept_list_input(name: str, fn: MetricFn) -> None:
    """Guards contract: plain list[float] input accepted; return type is float.

    The contract states each function accepts ``np.ndarray | pd.Series | list[float]``
    via ``np.asarray`` coercion at entry.  This test passes raw Python lists
    with no numpy involvement at the call site.
    """
    result = fn([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert isinstance(result, float), (
        f"{name} with list input must return Python float; got {type(result).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Test 11 — numpy array input accepted (Contract corner) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metrics_accept_numpy_array_input(name: str, fn: MetricFn) -> None:
    """Guards contract: np.ndarray input accepted; return type is float.

    The contract explicitly names np.ndarray as one of the three accepted
    input types.  Passing a pre-coerced array must not fail or change the
    result type.
    """
    y = np.asarray([10.0, 20.0, 30.0])
    p = np.asarray([10.0, 20.0, 30.0])
    result = fn(y, p)
    assert isinstance(result, float), (
        f"{name} with np.ndarray input must return Python float; got {type(result).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Test 12 — pd.Series input accepted (Contract corner) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metrics_accept_pandas_series_input(name: str, fn: MetricFn) -> None:
    """Guards contract: pd.Series input accepted; return type is float.

    The contract explicitly names pd.Series as one of the three accepted
    input types.  pandas Series are the natural output of model.predict()
    and the natural container for y_test slices from DataFrame.
    """
    y = pd.Series([10.0, 20.0, 30.0])
    p = pd.Series([10.0, 20.0, 30.0])
    result = fn(y, p)
    assert isinstance(result, float), (
        f"{name} with pd.Series input must return Python float; got {type(result).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Test 13 — Empty arrays rejected (Contract corner) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metrics_reject_empty_arrays(name: str, fn: MetricFn) -> None:
    """Guards contract: empty y_true and y_pred raise ValueError naming 'empty' or 'zero-length'.

    The shared guard ``_coerce_and_validate`` must reject zero-length inputs
    before the kernel executes; an empty mean or sum is undefined.
    """
    with pytest.raises(ValueError, match=r"(?i)(empty|zero.length|non.empty)"):
        fn([], [])


# ---------------------------------------------------------------------------
# Test 14 — Return type is Python float, not numpy float (Contract corner) — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metrics_return_python_float_not_numpy_float(name: str, fn: MetricFn) -> None:
    """Guards contract: each metric returns Python ``float``, not ``np.floating``.

    Plan T5 / harness contract: the evaluation harness assembles a DataFrame
    from per-metric return values; numpy-float values produce surprising dtypes
    in that DataFrame (e.g. ``object`` column when mixing numpy scalars with
    Python floats in a list).  Enforcing ``type(result) is float`` eliminates
    this class of bug.

    Note: ``isinstance(np.float64(1.0), float)`` is ``True`` on CPython due to
    numpy's subclassing, which is why we use strict ``type(…) is float``.
    """
    result = fn([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert type(result) is float, (
        f"{name} must return Python ``float`` (type is float), not "
        f"``{type(result).__name__}`` — see plan T5 harness DataFrame constraint."
    )


# ---------------------------------------------------------------------------
# Test 15 — METRIC_REGISTRY contains all four and maps to correct functions
# ---------------------------------------------------------------------------


def test_metric_registry_contains_all_four() -> None:
    """Guards plan T5 test 15 / METRIC_REGISTRY contract.

    ``METRIC_REGISTRY`` must have exactly the four keys ``{"mae", "mape",
    "rmse", "wape"}`` and each value must be the corresponding module-level
    function (identity check via ``is``).

    The registry is consumed by the Task T6 harness via
    ``MetricsConfig.names``; a missing or mis-mapped entry would silently
    produce wrong metrics.
    """
    assert set(METRIC_REGISTRY) == {"mae", "mape", "rmse", "wape"}, (
        f"METRIC_REGISTRY keys must be exactly {{mae, mape, rmse, wape}}; "
        f"got {set(METRIC_REGISTRY)!r}."
    )
    assert METRIC_REGISTRY["mae"] is mae, "METRIC_REGISTRY['mae'] must be the mae function."
    assert METRIC_REGISTRY["mape"] is mape, "METRIC_REGISTRY['mape'] must be the mape function."
    assert METRIC_REGISTRY["rmse"] is rmse, "METRIC_REGISTRY['rmse'] must be the rmse function."
    assert METRIC_REGISTRY["wape"] is wape, "METRIC_REGISTRY['wape'] must be the wape function."


# ---------------------------------------------------------------------------
# Test 16 — Registry functions produce same result as direct call — parametrised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL_METRICS, ids=[n for n, _ in _ALL_METRICS])
def test_metric_registry_functions_produce_same_result_as_direct_call(
    name: str, fn: MetricFn
) -> None:
    """Guards plan T5 test 16 / METRIC_REGISTRY contract.

    ``METRIC_REGISTRY[name](y, p)`` must produce the same value as a
    direct call to the named function.  This guards against a registry
    that wraps or transforms the function rather than re-exporting it.
    """
    y = [10.0, 20.0, 30.0]
    p = [10.0, 20.0, 30.0]
    registry_result = METRIC_REGISTRY[name](y, p)
    direct_result = fn(y, p)
    assert registry_result == direct_result, (
        f"METRIC_REGISTRY['{name}'](y, p) == {registry_result!r} but direct call "
        f"returned {direct_result!r} — registry must be an identity mapping."
    )
