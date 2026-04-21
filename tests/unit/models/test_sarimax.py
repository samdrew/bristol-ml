"""Spec-derived tests for ``bristol_ml.models.sarimax.SarimaxModel`` scaffold.

Every test is derived from:

- ``docs/plans/active/07-sarimax.md`` §Task T3 (lines 302-322): named test list,
  acceptance criteria AC-1, AC-6, AC-10, AC-11.
- ``docs/plans/active/07-sarimax.md`` §Task T4 (lines 324-360): fit/predict tests,
  acceptance criteria AC-1, AC-2, AC-3, AC-6, AC-7, AC-8, AC-11.
- ``src/bristol_ml/models/sarimax.py`` inline contracts (constructor, ``metadata``,
  ``results``, ``_cli_main``).
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section ("metadata before
  fit", "Predict-before-fit" guard convention).
- ``conf/_schemas.py`` ``SarimaxConfig`` defaults: ``order=(1,0,1)``,
  ``seasonal_order=(1,1,1,24)``.

No production code is modified here.  If a test below fails the failure indicates
a deviation from the spec — do not weaken the test; surface the failure to the
implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- ``SarimaxConfig()`` default construction throughout (no extra kwargs needed).
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX as _StatsmodelsSARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from bristol_ml.models import Model, SarimaxModel
from bristol_ml.models.sarimax import _cli_main
from conf._schemas import SarimaxConfig

# ---------------------------------------------------------------------------
# Module-level synthetic data helper
# ---------------------------------------------------------------------------


def _synthetic_utc_frame(n_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features_df, target_series)`` for use in T4 tests.

    - DatetimeIndex: tz-aware UTC, hourly, starting 2024-01-01.
    - Two ``float64`` exog columns: ``temp_c``, ``cloud_cover``.
    - Target: AR(1)-ish process plus mild daily + weekly sine components,
      scaled to roughly 10 000 MW, reproducible via ``numpy.random.default_rng(0)``.
    - ``n_rows`` should be small (256-500) so individual tests stay fast.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")

    # Exog columns
    temp_c = rng.normal(loc=10.0, scale=5.0, size=n_rows)
    cloud_cover = rng.uniform(0.0, 1.0, size=n_rows)

    # Target: AR(1) + daily + weekly seasonality + noise, scaled to ~10 000
    ar_coef = 0.7
    noise = rng.normal(scale=200.0, size=n_rows)
    t = np.arange(n_rows, dtype=np.float64)
    daily = 500.0 * np.sin(2.0 * np.pi * t / 24.0)
    weekly = 300.0 * np.sin(2.0 * np.pi * t / 168.0)
    target_vals = np.zeros(n_rows, dtype=np.float64)
    target_vals[0] = 10_000.0
    for i in range(1, n_rows):
        target_vals[i] = (
            ar_coef * target_vals[i - 1]
            + (1.0 - ar_coef) * 10_000.0
            + daily[i]
            + weekly[i]
            + noise[i]
        )

    features_df = pd.DataFrame(
        {"temp_c": temp_c, "cloud_cover": cloud_cover},
        index=index,
    )
    target_series = pd.Series(target_vals, index=index, name="nd_mw")
    return features_df, target_series


# Small, fast SARIMAX config — used throughout T4 unless the test is
# specifically exercising the order/seasonal parameters themselves.
_FAST_CONFIG = SarimaxConfig(
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 24),
    weekly_fourier_harmonics=2,
)

_FAST_CONFIG_NO_FOURIER = SarimaxConfig(
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 24),
    weekly_fourier_harmonics=0,
)

# ---------------------------------------------------------------------------
# 1. test_sarimax_model_conforms_to_model_protocol (T3 plan §Task T3, AC-1, AC-5)
# ---------------------------------------------------------------------------


def test_sarimax_model_conforms_to_model_protocol() -> None:
    """Guards T3 named test and AC-1/AC-5: isinstance check against Model protocol.

    ``@runtime_checkable`` structural-subtype check confirms that all five
    required protocol members (``fit``, ``predict``, ``save``, ``load``,
    ``metadata``) are present on ``SarimaxModel``.

    Plan clause: T3 plan §Task T3 named test / AC-1 / AC-5.
    """
    config = SarimaxConfig()
    model = SarimaxModel(config)
    assert isinstance(model, Model), (
        "SarimaxModel(SarimaxConfig()) must pass isinstance(model, Model); "
        "the @runtime_checkable protocol check requires all five members: "
        "fit, predict, save, load, metadata (T3 plan / AC-1 / AC-5)."
    )


# ---------------------------------------------------------------------------
# 2. test_sarimax_metadata_name_matches_regex (T3 plan §Task T3, AC-6, AC-10)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_name_matches_regex() -> None:
    """Guards T3 named test: metadata name matches regex and equals the expected value.

    Two assertions are required by the plan:
    1. The ``name`` field matches the ``ModelMetadata`` constraint regex
       ``^[a-z][a-z0-9_.-]*$``.
    2. With default ``order=(1,0,1)`` and ``seasonal_order=(1,1,1,24)`` the
       name must equal ``"sarimax-1-0-1-1-1-1-24"`` (the format produced by
       ``_build_metadata_name``).

    Plan clause: T3 plan §Task T3 named test / AC-6 / AC-10.
    """
    config = SarimaxConfig()  # defaults: order=(1,0,1), seasonal_order=(1,1,1,24)
    model = SarimaxModel(config)
    name = model.metadata.name

    assert re.match(r"^[a-z][a-z0-9_.-]*$", name), (
        f"metadata.name must match ^[a-z][a-z0-9_.-]*$; got {name!r} (T3 plan §Task T3 / AC-6)."
    )
    expected = "sarimax-1-0-1-1-1-1-24"
    assert name == expected, (
        f"metadata.name must equal {expected!r} for default config "
        f"order=(1,0,1) seasonal_order=(1,1,1,24); got {name!r} "
        "(T3 plan §Task T3 / _build_metadata_name contract)."
    )


# ---------------------------------------------------------------------------
# 3. test_sarimax_metadata_fit_utc_none_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_fit_utc_none_before_fit() -> None:
    """Guards T3 named test: unfitted model's metadata.fit_utc is None.

    Before any call to ``fit()`` the ``fit_utc`` field must be ``None``,
    matching the Stage 4 protocol convention documented in models CLAUDE.md
    ("metadata before fit").

    Plan clause: T3 plan §Task T3 named test / models CLAUDE.md
    "metadata before fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    assert model.metadata.fit_utc is None, (
        f"metadata.fit_utc must be None before fit(); "
        f"got {model.metadata.fit_utc!r} (T3 plan §Task T3)."
    )


# ---------------------------------------------------------------------------
# 4. test_sarimax_metadata_feature_columns_empty_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_feature_columns_empty_before_fit() -> None:
    """Guards T3 named test: unfitted model's metadata.feature_columns is empty tuple.

    Before any call to ``fit()`` the ``feature_columns`` field must be an
    empty tuple ``()``.  Matching the Stage 4 protocol convention documented
    in models CLAUDE.md ("metadata before fit").

    Plan clause: T3 plan §Task T3 named test / models CLAUDE.md
    "metadata before fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    assert model.metadata.feature_columns == (), (
        f"metadata.feature_columns must be () before fit(); "
        f"got {model.metadata.feature_columns!r} (T3 plan §Task T3)."
    )


# ---------------------------------------------------------------------------
# 5. test_sarimax_results_property_raises_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_results_property_raises_before_fit() -> None:
    """Guards T3 named test: accessing .results before fit raises RuntimeError.

    The ``results`` property must raise ``RuntimeError`` (not return ``None``
    or raise ``AttributeError``) when the model has not yet been fit.  The
    error message must mention "fit" so the user understands the pre-condition.

    Plan clause: T3 plan §Task T3 named test / sarimax.py ``results`` property
    docstring / models CLAUDE.md "Predict-before-fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    with pytest.raises(RuntimeError) as exc_info:
        _ = model.results
    assert "fit" in str(exc_info.value).lower(), (
        f"RuntimeError message must mention 'fit'; "
        f"got {str(exc_info.value)!r} (T3 plan §Task T3 / sarimax.py results guard)."
    )


# ---------------------------------------------------------------------------
# 6. test_sarimax_cli_main_returns_zero (T3 plan §Task T3, AC-11)
# ---------------------------------------------------------------------------


def test_sarimax_cli_main_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Guards T3 named test and AC-11: _cli_main([]) returns 0 and prints config schema.

    ``_cli_main([])`` must:
    - Return the integer 0 (DESIGN §2.1.1 standalone module contract).
    - Print text to stdout that contains "SarimaxConfig" (the JSON schema
      header written by the implementation).

    The call resolves the real Hydra config (``model=sarimax`` override) so
    this test validates the full config-resolution path, not just the return value.

    Plan clause: T3 plan §Task T3 named test / DESIGN §2.1.1 / AC-11.
    """
    result = _cli_main([])
    captured = capsys.readouterr()
    assert result == 0, (
        f"_cli_main([]) must return 0; got {result!r} (T3 plan §Task T3 / DESIGN §2.1.1)."
    )
    assert "SarimaxConfig" in captured.out, (
        f"_cli_main([]) stdout must contain 'SarimaxConfig'; "
        f"got {captured.out[:200]!r} (T3 plan §Task T3 / AC-11)."
    )


# ===========================================================================
# Task T4 — SarimaxModel.fit and .predict (plan §Task T4, lines 324-360)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. test_sarimax_fit_stores_results
# ---------------------------------------------------------------------------


def test_sarimax_fit_stores_results() -> None:
    """Post-fit ``_results`` is a ``SARIMAXResultsWrapper`` instance.

    Plan clause: T4 §Task T4 named test / AC-1, AC-6.
    """
    features, target = _synthetic_utc_frame(256)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    model.fit(features, target)
    assert isinstance(model._results, SARIMAXResultsWrapper), (
        f"After fit(), model._results must be a SARIMAXResultsWrapper; "
        f"got {type(model._results).__name__!r} (T4 plan §Task T4)."
    )


# ---------------------------------------------------------------------------
# 2. test_sarimax_fit_rejects_length_mismatch
# ---------------------------------------------------------------------------


def test_sarimax_fit_rejects_length_mismatch() -> None:
    """``ValueError`` raised when ``len(features) != len(target)``; message mentions "len".

    Plan clause: T4 §Task T4 named test / AC-8.
    """
    features, target = _synthetic_utc_frame(256)
    # Trim target to create a mismatch
    short_target = target.iloc[:200]
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    with pytest.raises(ValueError) as exc_info:
        model.fit(features, short_target)
    assert "len" in str(exc_info.value).lower(), (
        f"ValueError message must reference 'len'; got {str(exc_info.value)!r} (T4 plan §Task T4)."
    )


# ---------------------------------------------------------------------------
# 3. test_sarimax_fit_rejects_non_utc_index
# ---------------------------------------------------------------------------


def test_sarimax_fit_rejects_non_utc_index() -> None:
    """``ValueError`` raised for tz-naive and non-UTC tz-aware DatetimeIndex.

    Two cases exercised:
    (a) tz-naive DatetimeIndex.
    (b) Non-UTC tz-aware DatetimeIndex (``Europe/London``).

    Both error messages must reference "UTC" or "tz".

    Plan clause: T4 §Task T4 named test / AC-7.
    """
    features, target = _synthetic_utc_frame(256)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)

    # (a) tz-naive
    naive_features = features.copy()
    naive_features.index = features.index.tz_localize(None)
    naive_target = target.copy()
    naive_target.index = target.index.tz_localize(None)
    with pytest.raises(ValueError) as exc_info_naive:
        model.fit(naive_features, naive_target)
    msg_naive = str(exc_info_naive.value).lower()
    assert "utc" in msg_naive or "tz" in msg_naive, (
        f"ValueError for tz-naive index must reference 'UTC' or 'tz'; "
        f"got {str(exc_info_naive.value)!r} (T4 plan §Task T4 / AC-7)."
    )

    # (b) non-UTC tz-aware — Europe/London
    london_features = features.copy()
    london_features.index = features.index.tz_convert("Europe/London")
    london_target = target.copy()
    london_target.index = target.index.tz_convert("Europe/London")
    with pytest.raises(ValueError) as exc_info_london:
        model.fit(london_features, london_target)
    msg_london = str(exc_info_london.value).lower()
    assert "utc" in msg_london or "tz" in msg_london, (
        f"ValueError for non-UTC tz-aware index must reference 'UTC' or 'tz'; "
        f"got {str(exc_info_london.value)!r} (T4 plan §Task T4 / AC-7)."
    )


# ---------------------------------------------------------------------------
# 4. test_sarimax_fit_emits_freq_hourly_to_sarimax_constructor
# ---------------------------------------------------------------------------


def test_sarimax_fit_emits_freq_hourly_to_sarimax_constructor() -> None:
    """The SARIMAX constructor is called with ``freq="h"`` (surprise 2 regression guard).

    Monkeypatches ``bristol_ml.models.sarimax.SARIMAX`` with a ``wraps``-based
    spy so the real fitting still runs, then inspects the captured ``call_args``
    to assert ``freq="h"`` was supplied.

    Plan clause: T4 §Task T4 named test (surprise 2 regression guard) / AC-6.
    """
    features, target = _synthetic_utc_frame(256)
    config = _FAST_CONFIG_NO_FOURIER

    with patch(
        "bristol_ml.models.sarimax.SARIMAX",
        wraps=_StatsmodelsSARIMAX,
    ) as mock_sarimax:
        model = SarimaxModel(config)
        model.fit(features, target)

    assert mock_sarimax.called, (
        "SARIMAX constructor must be called during fit() (T4 plan §Task T4 / surprise 2)."
    )
    call_kwargs = mock_sarimax.call_args.kwargs
    assert call_kwargs.get("freq") == "h", (
        f"SARIMAX constructor must receive freq='h'; "
        f"got freq={call_kwargs.get('freq')!r} (T4 plan §Task T4 / surprise 2)."
    )


# ---------------------------------------------------------------------------
# 5. test_sarimax_fit_is_reentrant
# ---------------------------------------------------------------------------


def test_sarimax_fit_is_reentrant() -> None:
    """Second ``fit()`` discards prior state; ``_feature_columns`` reflects only the latest call.

    Strategy: first fit on a frame with columns ``["temp_c", "cloud_cover"]``
    (config.feature_columns=None, so all columns are taken), then fit again on a
    frame with only ``["temp_c"]``.  Assert that after the second fit:

    - ``m._feature_columns`` contains only ``"temp_c"`` (no union with the first fit).
    - ``m._results.nobs`` equals the second dataset's row count (256).

    Plan clause: T4 §Task T4 named test (NFR-5 re-entrancy) / AC-3, AC-6.
    """
    rng = np.random.default_rng(42)
    # First fit: two-column frame
    idx1 = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    features1 = pd.DataFrame(
        {"temp_c": rng.normal(10, 5, 300), "cloud_cover": rng.uniform(0, 1, 300)},
        index=idx1,
    )
    target1 = pd.Series(rng.normal(10_000, 500, 300), index=idx1, name="nd_mw")

    config = SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 24),
        weekly_fourier_harmonics=0,
    )
    model = SarimaxModel(config)
    model.fit(features1, target1)
    assert "cloud_cover" in model._feature_columns, (
        "After first fit cloud_cover should be in _feature_columns (sanity check)."
    )

    # Second fit: one-column frame of different length
    idx2 = pd.date_range("2024-06-01", periods=256, freq="h", tz="UTC")
    features2 = pd.DataFrame(
        {"temp_c": rng.normal(18, 4, 256)},
        index=idx2,
    )
    target2 = pd.Series(rng.normal(9_000, 400, 256), index=idx2, name="nd_mw")

    model.fit(features2, target2)

    assert model._feature_columns == ("temp_c",), (
        f"After second fit, _feature_columns must reflect only the second call; "
        f"got {model._feature_columns!r} (T4 plan §Task T4 / NFR-5)."
    )
    assert int(model._results.nobs) == 256, (
        f"After second fit, nobs must equal the second dataset's row count (256); "
        f"got {model._results.nobs!r} (T4 plan §Task T4 / NFR-5)."
    )


# ---------------------------------------------------------------------------
# 6. test_sarimax_fit_appends_weekly_fourier_when_harmonics_gt_zero
# ---------------------------------------------------------------------------


def test_sarimax_fit_appends_weekly_fourier_when_harmonics_gt_zero() -> None:
    """With ``weekly_fourier_harmonics=3``, ``feature_columns`` includes 6 Fourier columns.

    The six expected columns are ``week_sin_k1``, ``week_cos_k1``,
    ``week_sin_k2``, ``week_cos_k2``, ``week_sin_k3``, ``week_cos_k3``.

    Plan clause: T4 §Task T4 named test / AC-6 (plan D1/D3 DHR path).
    """
    features, target = _synthetic_utc_frame(256)
    config = SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 24),
        weekly_fourier_harmonics=3,
    )
    model = SarimaxModel(config)
    model.fit(features, target)

    feature_cols = model.metadata.feature_columns
    expected_fourier = (
        "week_sin_k1",
        "week_cos_k1",
        "week_sin_k2",
        "week_cos_k2",
        "week_sin_k3",
        "week_cos_k3",
    )
    for col in expected_fourier:
        assert col in feature_cols, (
            f"Expected Fourier column {col!r} missing from metadata.feature_columns; "
            f"got {feature_cols!r} (T4 plan §Task T4 / AC-6 DHR path)."
        )


# ---------------------------------------------------------------------------
# 7. test_sarimax_fit_harmonics_zero_skips_fourier
# ---------------------------------------------------------------------------


def test_sarimax_fit_harmonics_zero_skips_fourier() -> None:
    """With ``weekly_fourier_harmonics=0``, no ``week_`` column appears in feature_columns.

    Plan clause: T4 §Task T4 named test / AC-6 (plan D1/D3 Fourier disable path).
    """
    features, target = _synthetic_utc_frame(256)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    model.fit(features, target)

    feature_cols = model.metadata.feature_columns
    fourier_cols = [c for c in feature_cols if c.startswith("week_")]
    assert fourier_cols == [], (
        f"With weekly_fourier_harmonics=0, no 'week_' column should appear in "
        f"metadata.feature_columns; got {fourier_cols!r} (T4 plan §Task T4 / AC-6)."
    )


# ---------------------------------------------------------------------------
# 8. test_sarimax_predict_returns_series_indexed_to_features_index
# ---------------------------------------------------------------------------


def test_sarimax_predict_returns_series_indexed_to_features_index() -> None:
    """Predicted Series index equals ``features.index`` exactly (surprise 1 regression guard).

    Fit on 500 rows; predict on the last 48.  The statsmodels
    ``get_forecast(...).predicted_mean`` does NOT preserve ``features.index``
    — this is the "surprise 1" described in sarimax.py and models CLAUDE.md.
    ``SarimaxModel.predict`` must re-index before returning.

    Plan clause: T4 §Task T4 named test (surprise 1 regression guard) / AC-6, AC-7.
    """
    features, target = _synthetic_utc_frame(500)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    model.fit(features, target)

    test_features = features.iloc[-48:]
    pred = model.predict(test_features)

    assert (pred.index == test_features.index).all(), (
        "pred.index must equal features.iloc[-48:].index exactly "
        "(surprise 1 re-indexing guard; T4 plan §Task T4)."
    )
    assert pred.index.tz is not None, "pred.index must be tz-aware (T4 plan §Task T4 / surprise 1)."
    assert str(pred.index.tz) == "UTC", (
        f"pred.index timezone must be UTC; got {pred.index.tz!r} (T4 plan §Task T4)."
    )


# ---------------------------------------------------------------------------
# 9. test_sarimax_predict_returns_series_with_target_column_name
# ---------------------------------------------------------------------------


def test_sarimax_predict_returns_series_with_target_column_name() -> None:
    """Predicted Series ``name`` equals ``config.target_column`` (default ``"nd_mw"``).

    Plan clause: T4 §Task T4 named test / AC-6.
    """
    features, target = _synthetic_utc_frame(256)
    config = _FAST_CONFIG_NO_FOURIER  # target_column defaults to "nd_mw"
    model = SarimaxModel(config)
    model.fit(features, target)

    pred = model.predict(features.iloc[-24:])
    assert pred.name == config.target_column, (
        f"pred.name must equal config.target_column ({config.target_column!r}); "
        f"got {pred.name!r} (T4 plan §Task T4 / AC-6)."
    )


# ---------------------------------------------------------------------------
# 10. test_sarimax_predict_before_fit_raises_runtime_error
# ---------------------------------------------------------------------------


def test_sarimax_predict_before_fit_raises_runtime_error() -> None:
    """Calling ``predict()`` on an unfitted model raises ``RuntimeError``; message mentions "fit".

    Plan clause: T4 §Task T4 named test / models CLAUDE.md "Predict-before-fit" guard.
    """
    features, _ = _synthetic_utc_frame(48)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    with pytest.raises(RuntimeError) as exc_info:
        model.predict(features)
    assert "fit" in str(exc_info.value).lower(), (
        f"RuntimeError message must mention 'fit'; "
        f"got {str(exc_info.value)!r} (T4 plan §Task T4 / predict-before-fit guard)."
    )


# ---------------------------------------------------------------------------
# 11. test_sarimax_predict_length_matches_features
# ---------------------------------------------------------------------------


def test_sarimax_predict_length_matches_features() -> None:
    """``len(pred) == len(features)`` passed to predict.

    Plan clause: T4 §Task T4 named test / AC-6.
    """
    features, target = _synthetic_utc_frame(300)
    model = SarimaxModel(_FAST_CONFIG_NO_FOURIER)
    model.fit(features, target)

    # Predict on a window of 36 rows
    test_window = features.iloc[-36:]
    pred = model.predict(test_window)
    assert len(pred) == len(test_window), (
        f"len(pred) must equal len(features) passed to predict; "
        f"expected 36, got {len(pred)} (T4 plan §Task T4 / AC-6)."
    )


# ---------------------------------------------------------------------------
# 12. test_sarimax_predict_raises_on_missing_feature_column
# ---------------------------------------------------------------------------


def test_sarimax_predict_raises_on_missing_feature_column() -> None:
    """``KeyError`` raised when predict features are missing a fit-time column.

    Fit on a frame with columns ``["a", "b"]``, predict on a frame with only ``["a"]``.

    Plan clause: T4 §Task T4 named test / AC-8.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=256, freq="h", tz="UTC")
    train_features = pd.DataFrame(
        {"a": rng.normal(0, 1, 256), "b": rng.normal(0, 1, 256)},
        index=idx,
    )
    target = pd.Series(rng.normal(10_000, 500, 256), index=idx, name="nd_mw")

    config = SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 24),
        weekly_fourier_harmonics=0,
        feature_columns=("a", "b"),
    )
    model = SarimaxModel(config)
    model.fit(train_features, target)

    # Predict frame missing column "b"
    idx_test = pd.date_range("2024-02-01", periods=24, freq="h", tz="UTC")
    test_features = pd.DataFrame(
        {"a": rng.normal(0, 1, 24)},
        index=idx_test,
    )
    with pytest.raises(KeyError):
        model.predict(test_features)
