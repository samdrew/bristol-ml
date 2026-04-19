"""Spec-derived tests for ``bristol_ml.models.linear.LinearModel``.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T4 (acceptance
  criteria and named test list).
- ``docs/plans/active/04-linear-baseline.md`` §1 D1 (statsmodels mandate,
  intercept handling).
- ``docs/plans/active/04-linear-baseline.md`` §4 AC-3 (save/load identical
  predictions), AC-7 (protocol conformance).
- ``docs/plans/active/04-linear-baseline.md`` §10 risk register (statsmodels
  intercept-handling pitfall, re-entrancy).
- ``src/bristol_ml/models/linear.py`` inline contracts.
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section.

No production code is modified here.  If any test below fails the failure
indicates a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan clause, AC, or F-number it guards.
- ``tmp_path`` (pytest built-in) for filesystem operations.
- ``pd.date_range(..., tz="UTC")`` for all timestamp indices.
- ``np.random.seed(0)`` at the top of every test that draws random data.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bristol_ml.features.assembler import WEATHER_VARIABLE_COLUMNS
from bristol_ml.models.io import save_joblib
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.protocol import Model
from conf._schemas import LinearConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _linear_cfg(
    feature_columns: tuple[str, ...] | None = ("x1", "x2"),
    target_column: str = "nd_mw",
    fit_intercept: bool = True,
) -> LinearConfig:
    """Return a ``LinearConfig`` wired to explicit feature columns by default."""
    return LinearConfig(
        feature_columns=feature_columns,
        target_column=target_column,
        fit_intercept=fit_intercept,
    )


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """Return a UTC-aware hourly DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _make_xy(
    n: int,
    *,
    coef_x1: float = 2.0,
    coef_x2: float = 3.0,
    intercept: float = 1.0,
    noise_std: float = 0.01,
    start: str = "2024-01-01 00:00",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target) for a clean two-predictor OLS fixture.

    ``y = coef_x1*x1 + coef_x2*x2 + intercept + N(0, noise_std)``.
    """
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n, start=start)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = coef_x1 * x1 + coef_x2 * x2 + intercept + rng.normal(0, noise_std, n)
    features = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    return features, target


# ---------------------------------------------------------------------------
# 1. test_linear_fit_recovers_known_coefficients (T4 plan named test, §10 risk)
# ---------------------------------------------------------------------------


def test_linear_fit_recovers_known_coefficients() -> None:
    """Guards T4 (plan §6 named test) and §10 risk register: intercept-handling pitfall.

    Generates ``y = 2*x1 + 3*x2 + 1 + N(0, 0.01)`` with n=500 rows and fits
    with explicit ``feature_columns=("x1", "x2")``.  The fitted coefficients
    must be within ``1e-2`` of ``{"const": 1.0, "x1": 2.0, "x2": 3.0}``.

    With ``fit_intercept=True`` (the default), ``sm.add_constant`` must add the
    constant column so OLS recovers the true intercept.  If it were silently
    dropped (§10 risk register), the recovered intercept would be zero and the
    test would fail.

    Plan clause: T4 plan §6 named test / §10 risk register "statsmodels intercept
    handling silently drops the constant".
    """
    np.random.seed(0)
    n = 500
    cfg = _linear_cfg(feature_columns=("x1", "x2"), fit_intercept=True)
    model = LinearModel(cfg)
    features, target = _make_xy(n, coef_x1=2.0, coef_x2=3.0, intercept=1.0, noise_std=0.01, seed=0)

    model.fit(features, target)

    coefs = model.metadata.hyperparameters["coefficients"]
    assert isinstance(coefs, dict), (
        f"metadata.hyperparameters['coefficients'] must be a dict; got {type(coefs)} (T4 plan)."
    )

    true_values = {"const": 1.0, "x1": 2.0, "x2": 3.0}
    for key, true_val in true_values.items():
        assert key in coefs, (
            f"Expected key '{key}' in coefficients dict; got keys {list(coefs.keys())} "
            f"(T4 plan §6 named test / §10 risk register)."
        )
        diff = abs(coefs[key] - true_val)
        assert diff < 1e-2, (
            f"Coefficient '{key}' = {coefs[key]:.6f} must be within 1e-2 of {true_val}; "
            f"diff={diff:.6f} (T4 plan §6 named test / §10 risk register)."
        )


# ---------------------------------------------------------------------------
# 2. test_linear_predict_shape (T4 plan named test)
# ---------------------------------------------------------------------------


def test_linear_predict_shape() -> None:
    """Guards T4 (plan §6 named test): predict on 24 rows returns 24-element Series.

    Fit on 500 rows; predict on 24 rows; assert length 24, type ``pd.Series``,
    and name ``"nd_mw"`` (default target_column).

    Plan clause: T4 plan §6 named test / linear.py ``predict()`` return shape contract.
    """
    np.random.seed(0)
    n_train = 500
    n_test = 24
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, target = _make_xy(n_train, seed=0)
    model.fit(features, target)

    test_features, _ = _make_xy(n_test, start="2024-01-22 00:00", seed=1)
    preds = model.predict(test_features)

    assert len(preds) == n_test, (
        f"predict() on {n_test} rows must return {n_test} predictions; "
        f"got {len(preds)} (T4 plan §6 named test)."
    )
    assert isinstance(preds, pd.Series), (
        f"predict() must return pd.Series; got {type(preds)} (T4 plan §6 named test)."
    )
    assert preds.name == "nd_mw", (
        f"Series name must equal 'nd_mw' (default target_column); "
        f"got {preds.name!r} (T4 plan §6 named test)."
    )


# ---------------------------------------------------------------------------
# 3. test_linear_save_load_identical_predictions (T4 plan named test, AC-3)
# ---------------------------------------------------------------------------


def test_linear_save_load_identical_predictions(tmp_path: Path) -> None:
    """Guards T4 (plan §6 named test) and AC-3: joblib round-trip is bit-identical.

    Fit, save to ``tmp_path / "linear.joblib"``, load, predict on same test
    frame; assert predictions are ``np.allclose`` with ``atol=0, rtol=0``
    (exact equality — joblib round-trip must be bit-identical).

    Plan clause: T4 plan §6 named test / AC-3 / F-10.
    """
    np.random.seed(0)
    n = 500
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, target = _make_xy(n, seed=0)
    model.fit(features, target)

    path = tmp_path / "linear.joblib"
    model.save(path)
    assert path.exists(), f"save() must create the artefact at {path} (AC-3)."

    loaded = LinearModel.load(path)
    assert isinstance(loaded, Model), (
        "Loaded LinearModel must satisfy isinstance(loaded, Model) (AC-3 / AC-7)."
    )

    test_features, _ = _make_xy(24, start="2024-01-22 00:00", seed=2)
    original_preds = model.predict(test_features)
    loaded_preds = loaded.predict(test_features)

    assert np.allclose(original_preds.values, loaded_preds.values, atol=0, rtol=0), (
        "Predictions after joblib round-trip must be bit-identical (T4 plan §6 / AC-3 / F-10).\n"
        f"Max abs diff: {np.abs(original_preds.values - loaded_preds.values).max()}"
    )


# ---------------------------------------------------------------------------
# 4. test_linear_conforms_to_model_protocol (T4 plan named test, AC-7)
# ---------------------------------------------------------------------------


def test_linear_conforms_to_model_protocol() -> None:
    """Guards T4 (plan §6 named test) and AC-7: ``isinstance(LinearModel(cfg), Model)`` is True.

    The ``@runtime_checkable`` protocol check verifies attribute presence for
    all five required members: ``fit``, ``predict``, ``save``, ``load``,
    ``metadata``.

    Plan clause: T4 plan §6 named test / AC-7 / D3.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    assert isinstance(model, Model), (
        "LinearModel must pass isinstance(model, Model) (T4 plan §6 named test / AC-7)."
    )


# ---------------------------------------------------------------------------
# 5. test_linear_metadata_includes_coefficients (T4 plan named test)
# ---------------------------------------------------------------------------


def test_linear_metadata_includes_coefficients() -> None:
    """Guards T4 (plan §6 named test): post-fit metadata carries coefficients, rsquared, nobs.

    After fitting with ``fit_intercept=True``:
    - ``metadata.hyperparameters["coefficients"]`` is a dict containing ``"const"``
      and every feature column name.  Values are ``float``.
    - ``metadata.hyperparameters["rsquared"]`` is a ``float``.
    - ``metadata.hyperparameters["nobs"]`` is an ``int``.

    Plan clause: T4 plan §6 named test / linear.py ``metadata`` property docstring.
    """
    np.random.seed(0)
    n = 200
    cfg = _linear_cfg(feature_columns=("x1", "x2"), fit_intercept=True)
    model = LinearModel(cfg)
    features, target = _make_xy(n, seed=0)
    model.fit(features, target)

    meta = model.metadata
    hp = meta.hyperparameters

    assert "coefficients" in hp, (
        f"'coefficients' must be in metadata.hyperparameters after fit; "
        f"got keys {list(hp.keys())} (T4 plan §6 named test)."
    )
    coefs = hp["coefficients"]
    assert isinstance(coefs, dict), (
        f"coefficients must be a dict; got {type(coefs)} (T4 plan §6 named test)."
    )

    # The constant column must be present (fit_intercept=True).
    assert "const" in coefs, (
        f"'const' must be in coefficients when fit_intercept=True; "
        f"got {list(coefs.keys())} (T4 plan §6 named test / §10 risk register)."
    )
    for col in ("x1", "x2"):
        assert col in coefs, (
            f"Feature column '{col}' must appear in coefficients; "
            f"got {list(coefs.keys())} (T4 plan §6 named test)."
        )
    for key, val in coefs.items():
        assert isinstance(val, float), (
            f"Each coefficient value must be float; got {type(val)} for key '{key}' "
            "(T4 plan §6 named test)."
        )

    assert "rsquared" in hp, (
        f"'rsquared' must be in metadata.hyperparameters after fit; "
        f"got keys {list(hp.keys())} (T4 plan §6 named test)."
    )
    assert isinstance(hp["rsquared"], float), (
        f"rsquared must be a float; got {type(hp['rsquared'])} (T4 plan §6 named test)."
    )

    assert "nobs" in hp, (
        f"'nobs' must be in metadata.hyperparameters after fit; "
        f"got keys {list(hp.keys())} (T4 plan §6 named test)."
    )
    assert isinstance(hp["nobs"], int), (
        f"nobs must be an int; got {type(hp['nobs'])} (T4 plan §6 named test)."
    )


# ---------------------------------------------------------------------------
# 6. test_linear_fit_without_intercept_excludes_const
# ---------------------------------------------------------------------------


def test_linear_fit_without_intercept_excludes_const() -> None:
    """Guards linear.py intercept-handling contract: no ``"const"`` when fit_intercept=False.

    With ``fit_intercept=False``, ``sm.add_constant`` is not called; the design
    matrix contains only the explicit regressors.  After fitting,
    ``metadata.hyperparameters["coefficients"]`` must NOT contain ``"const"``
    and must contain exactly the two requested feature columns.

    Plan clause: T4 / linear.py ``fit()`` docstring / §10 risk register
    "statsmodels intercept handling".
    """
    np.random.seed(0)
    n = 200
    cfg = _linear_cfg(feature_columns=("x1", "x2"), fit_intercept=False)
    model = LinearModel(cfg)
    features, target = _make_xy(n, seed=0)
    model.fit(features, target)

    coefs = model.metadata.hyperparameters["coefficients"]
    assert "const" not in coefs, (
        f"'const' must NOT be in coefficients when fit_intercept=False; "
        f"got {list(coefs.keys())} (T4 / linear.py intercept contract)."
    )
    assert set(coefs.keys()) == {"x1", "x2"}, (
        f"coefficients must contain exactly the two regressors when fit_intercept=False; "
        f"got {set(coefs.keys())} (T4 / linear.py intercept contract)."
    )


# ---------------------------------------------------------------------------
# 7. test_linear_fit_resolves_weather_columns_when_feature_columns_none
# ---------------------------------------------------------------------------


def test_linear_fit_resolves_weather_columns_when_feature_columns_none() -> None:
    """Guards linear.py feature-column resolution: ``None`` triggers weather-column fallback.

    ``LinearConfig()`` default has ``feature_columns=None``.  Build a DataFrame
    containing all five weather columns from ``WEATHER_VARIABLE_COLUMNS`` plus
    a noise target and fit.  After fitting:
    - fit succeeds (no exception).
    - ``metadata.feature_columns`` equals the five weather column names in order.

    Plan clause: T4 / linear.py ``_resolve_feature_columns`` / D2 (assembler
    feature-table contract).
    """
    np.random.seed(0)
    n = 200
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    weather_col_names = [name for name, _ in WEATHER_VARIABLE_COLUMNS]

    feature_data = {col: rng.standard_normal(n).astype("float32") for col in weather_col_names}
    features = pd.DataFrame(feature_data, index=idx)
    # Target is a simple linear combination of the weather columns.
    target_vals = sum(features[col].astype("float64") for col in weather_col_names)
    target = pd.Series(target_vals, index=idx, name="nd_mw")  # type: ignore[arg-type]

    cfg = LinearConfig(feature_columns=None, target_column="nd_mw", fit_intercept=True)
    model = LinearModel(cfg)
    model.fit(features, target)

    expected_cols = tuple(weather_col_names)
    assert model.metadata.feature_columns == expected_cols, (
        f"metadata.feature_columns must equal {expected_cols!r} when feature_columns=None; "
        f"got {model.metadata.feature_columns!r} (T4 / linear.py _resolve_feature_columns)."
    )


# ---------------------------------------------------------------------------
# 8. test_linear_fit_rejects_length_mismatch
# ---------------------------------------------------------------------------


def test_linear_fit_rejects_length_mismatch() -> None:
    """Guards linear.py ``fit()`` contract: mismatched feature/target lengths raise ``ValueError``.

    ``features`` with 100 rows and ``target`` with 99 rows is a data-alignment
    error.  ``fit()`` must raise ``ValueError`` rather than silently ignoring
    the extra row or propagating an opaque numpy broadcast error.

    Plan clause: T4 / linear.py ``fit()`` ``ValueError`` docstring.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, target = _make_xy(100, seed=0)
    target_short = target.iloc[:99]

    with pytest.raises(ValueError):
        model.fit(features, target_short)


# ---------------------------------------------------------------------------
# 9. test_linear_fit_rejects_missing_feature_columns
# ---------------------------------------------------------------------------


def test_linear_fit_rejects_missing_feature_columns() -> None:
    """Guards linear.py ``fit()`` contract: missing columns raise ``ValueError`` naming them.

    Config names ``("missing_col",)``; features have only ``("t2m",)``.
    ``fit()`` must raise ``ValueError`` and the message must name ``"missing_col"``.

    Plan clause: T4 / linear.py ``fit()`` ``ValueError`` docstring.
    """
    cfg = LinearConfig(feature_columns=("missing_col",), target_column="nd_mw", fit_intercept=True)
    model = LinearModel(cfg)

    idx = _hourly_index(50)
    features = pd.DataFrame({"t2m": np.random.default_rng(0).standard_normal(50)}, index=idx)
    target = pd.Series(np.zeros(50), index=idx, name="nd_mw")

    with pytest.raises(ValueError) as exc_info:
        model.fit(features, target)

    assert "missing_col" in str(exc_info.value), (
        f"ValueError message must name 'missing_col'; got {str(exc_info.value)!r} "
        "(T4 / linear.py fit() error contract)."
    )


# ---------------------------------------------------------------------------
# 10. test_linear_predict_before_fit_raises_runtime_error
# ---------------------------------------------------------------------------


def test_linear_predict_before_fit_raises_runtime_error() -> None:
    """Guards linear.py ``predict()`` contract: predict before fit raises ``RuntimeError``.

    A freshly constructed ``LinearModel`` has no fitted state.  Any call to
    ``predict()`` must raise ``RuntimeError`` rather than returning stale or
    incorrect output.

    Plan clause: T4 / linear.py ``predict()`` docstring / models CLAUDE.md
    "Predict-before-fit" protocol semantic.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, _ = _make_xy(10, seed=0)

    with pytest.raises(RuntimeError):
        model.predict(features)


# ---------------------------------------------------------------------------
# 11. test_linear_predict_rejects_missing_feature_columns
# ---------------------------------------------------------------------------


def test_linear_predict_rejects_missing_feature_columns() -> None:
    """Guards linear.py ``predict()`` contract: missing columns raise ``ValueError``.

    Fit on ``("x1", "x2")``; predict on a DataFrame containing only ``("x1",)``.
    ``predict()`` must raise ``ValueError`` naming ``"x2"``.

    Plan clause: T4 / linear.py ``predict()`` ``ValueError`` docstring.
    """
    np.random.seed(0)
    cfg = _linear_cfg(feature_columns=("x1", "x2"))
    model = LinearModel(cfg)
    features, target = _make_xy(200, seed=0)
    model.fit(features, target)

    idx_test = _hourly_index(10, start="2024-02-01 00:00")
    partial_features = pd.DataFrame({"x1": np.ones(10)}, index=idx_test)

    with pytest.raises(ValueError) as exc_info:
        model.predict(partial_features)

    assert "x2" in str(exc_info.value), (
        f"ValueError message must name the missing column 'x2'; "
        f"got {str(exc_info.value)!r} (T4 / linear.py predict() error contract)."
    )


# ---------------------------------------------------------------------------
# 12. test_linear_predict_series_name_matches_target_column
# ---------------------------------------------------------------------------


def test_linear_predict_series_name_matches_target_column() -> None:
    """Guards linear.py ``predict()`` return-shape contract: Series name equals target_column.

    Configure ``target_column="custom_target"``; after fitting and predicting,
    the returned Series must have ``name == "custom_target"``.

    Plan clause: T4 / linear.py ``predict()`` return docstring.
    """
    np.random.seed(0)
    cfg = LinearConfig(
        feature_columns=("x1", "x2"), target_column="custom_target", fit_intercept=True
    )
    model = LinearModel(cfg)

    idx = _hourly_index(200)
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"x1": rng.standard_normal(200), "x2": rng.standard_normal(200)},
        index=idx,
    )
    target = pd.Series(np.zeros(200), index=idx, name="custom_target")
    model.fit(features, target)

    test_idx = _hourly_index(5, start="2024-01-10 00:00")
    test_features = pd.DataFrame({"x1": np.ones(5), "x2": np.ones(5)}, index=test_idx)
    preds = model.predict(test_features)

    assert preds.name == "custom_target", (
        f"Returned Series name must equal config.target_column='custom_target'; "
        f"got {preds.name!r} (T4 / linear.py predict() return docstring)."
    )


# ---------------------------------------------------------------------------
# 13. test_linear_predict_series_index_matches_features
# ---------------------------------------------------------------------------


def test_linear_predict_series_index_matches_features() -> None:
    """Guards linear.py ``predict()`` return-shape contract: index equals features.index.

    The returned Series must share the exact index of the input features
    DataFrame so the harness can align predictions with actuals.

    Plan clause: T4 / linear.py ``predict()`` return docstring / harness
    alignment contract.
    """
    np.random.seed(0)
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, target = _make_xy(200, seed=0)
    model.fit(features, target)

    test_features, _ = _make_xy(24, start="2024-02-01 00:00", seed=3)
    preds = model.predict(test_features)

    pd.testing.assert_index_equal(
        preds.index,
        test_features.index,
        obj="predict() return Series index vs features.index",
    )


# ---------------------------------------------------------------------------
# 14. test_linear_save_before_fit_raises_runtime_error
# ---------------------------------------------------------------------------


def test_linear_save_before_fit_raises_runtime_error(tmp_path: Path) -> None:
    """Guards linear.py ``save()`` contract: save before fit raises ``RuntimeError``.

    Persisting unfitted state must be refused; the implementation raises
    ``RuntimeError`` to prevent an empty-state artefact being written to disk.

    Plan clause: T4 / linear.py ``save()`` ``RuntimeError`` docstring.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)

    with pytest.raises(RuntimeError):
        model.save(tmp_path / "x.joblib")


# ---------------------------------------------------------------------------
# 15. test_linear_load_rejects_wrong_artefact_type
# ---------------------------------------------------------------------------


def test_linear_load_rejects_wrong_artefact_type(tmp_path: Path) -> None:
    """Guards linear.py ``load()`` contract: wrong artefact type raises ``TypeError``.

    A plain dict written via ``save_joblib`` is not a ``LinearModel``.
    ``LinearModel.load(path)`` must raise ``TypeError`` rather than silently
    returning the wrong class.

    Plan clause: T4 / linear.py ``load()`` ``TypeError`` docstring.
    """
    path = tmp_path / "wrong.joblib"
    save_joblib({"not": "a model"}, path)

    with pytest.raises(TypeError):
        LinearModel.load(path)


# ---------------------------------------------------------------------------
# 16. test_linear_results_property_raises_before_fit
# ---------------------------------------------------------------------------


def test_linear_results_property_raises_before_fit() -> None:
    """Guards linear.py ``results`` property contract: raises ``RuntimeError`` before fit.

    A freshly constructed ``LinearModel`` must raise ``RuntimeError`` when
    ``.results`` is accessed, rather than returning ``None`` or crashing with an
    ``AttributeError``.

    Plan clause: T4 / linear.py ``results`` property docstring / models
    CLAUDE.md "Predict-before-fit" protocol semantic.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)

    with pytest.raises(RuntimeError):
        _ = model.results


# ---------------------------------------------------------------------------
# 17. test_linear_results_summary_after_fit_contains_ols
# ---------------------------------------------------------------------------


def test_linear_results_summary_after_fit_contains_ols() -> None:
    """Guards T4 demo-moment contract: ``str(model.results.summary())`` contains OLS header.

    The Stage 4 notebook demo payoff is ``print(model.results.summary())``.
    After fitting, calling ``.summary()`` on the ``RegressionResultsWrapper``
    must produce a string that contains "OLS Regression Results" — the
    standard statsmodels header confirming the notebook demo works end-to-end.

    Plan clause: T4 / linear.py ``results`` property docstring "Stage 4 demo moment".
    """
    np.random.seed(0)
    cfg = _linear_cfg()
    model = LinearModel(cfg)
    features, target = _make_xy(200, seed=0)
    model.fit(features, target)

    summary_text = str(model.results.summary())
    assert "OLS Regression Results" in summary_text, (
        f"model.results.summary() must contain 'OLS Regression Results'; "
        f"got {summary_text[:200]!r} (T4 / demo-moment contract)."
    )


# ---------------------------------------------------------------------------
# 18. test_linear_refit_is_re_entrant (plan §10 risk register)
# ---------------------------------------------------------------------------


def test_linear_refit_is_re_entrant() -> None:
    """Guards plan §10 risk register: second ``fit()`` discards prior state.

    Fit once on dataset A (x1 coefficient ≈ 10.0), then fit again on dataset B
    (x1 coefficient ≈ 1.0).  Predictions and coefficients after the second fit
    must reflect dataset B only.  The x1 coefficient from the two fits must differ
    by more than ``0.5`` to confirm state replacement rather than accumulation.

    Plan clause: T4 / §10 risk register row "fit() must be re-entrant" / models
    CLAUDE.md "Re-entrancy" protocol semantic.
    """
    np.random.seed(0)
    n = 300
    cfg = _linear_cfg(feature_columns=("x1", "x2"))
    model = LinearModel(cfg)

    # Dataset A: x1 coefficient = 10.0.
    features_a, target_a = _make_xy(
        n, coef_x1=10.0, coef_x2=1.0, intercept=0.0, noise_std=0.01, seed=0
    )
    model.fit(features_a, target_a)
    coef_x1_after_a = model.metadata.hyperparameters["coefficients"]["x1"]

    # Dataset B: x1 coefficient = 1.0.
    features_b, target_b = _make_xy(
        n, coef_x1=1.0, coef_x2=1.0, intercept=0.0, noise_std=0.01, start="2024-02-01 00:00", seed=1
    )
    model.fit(features_b, target_b)
    coef_x1_after_b = model.metadata.hyperparameters["coefficients"]["x1"]

    diff = abs(coef_x1_after_a - coef_x1_after_b)
    assert diff > 0.5, (
        f"Re-fit must discard prior state: x1 coefficient after dataset A is "
        f"{coef_x1_after_a:.4f}, after dataset B is {coef_x1_after_b:.4f}; "
        f"diff={diff:.4f} must be > 0.5 (§10 risk register re-entrancy)."
    )


# ---------------------------------------------------------------------------
# 19. test_linear_metadata_before_fit_has_config_only
# ---------------------------------------------------------------------------


def test_linear_metadata_before_fit_has_config_only() -> None:
    """Guards linear.py ``metadata`` property: pre-fit metadata has config-only fields.

    Before any ``fit()`` call:
    - ``metadata.fit_utc is None``
    - ``metadata.feature_columns == ()``
    - ``metadata.hyperparameters == {"target_column": "nd_mw", "fit_intercept": True}``
      (no ``"coefficients"``, ``"rsquared"``, or ``"nobs"`` keys).

    Plan clause: T4 / linear.py ``metadata`` property docstring / models CLAUDE.md
    "metadata before fit" protocol semantic.
    """
    cfg = LinearConfig(feature_columns=("x1",), target_column="nd_mw", fit_intercept=True)
    model = LinearModel(cfg)
    meta = model.metadata

    assert meta.fit_utc is None, (
        f"metadata.fit_utc must be None before fit(); got {meta.fit_utc!r} (T4 / linear.py)."
    )
    assert meta.feature_columns == (), (
        f"metadata.feature_columns must be () before fit(); "
        f"got {meta.feature_columns!r} (T4 / linear.py)."
    )
    assert meta.hyperparameters == {"target_column": "nd_mw", "fit_intercept": True}, (
        f"metadata.hyperparameters must contain only config keys before fit(); "
        f"got {meta.hyperparameters!r} (T4 / linear.py metadata property)."
    )
    for forbidden_key in ("coefficients", "rsquared", "nobs"):
        assert forbidden_key not in meta.hyperparameters, (
            f"'{forbidden_key}' must NOT be in metadata.hyperparameters before fit(); "
            f"got {meta.hyperparameters!r} (T4 / linear.py metadata property)."
        )


# ---------------------------------------------------------------------------
# 20. test_linear_metadata_name_is_fixed
# ---------------------------------------------------------------------------


def test_linear_metadata_name_is_fixed() -> None:
    """Guards linear.py ``metadata`` property: name is always ``"linear-ols-weather-only"``.

    The model name is a fixed constant regardless of fit state or config variant.
    It must equal ``"linear-ols-weather-only"`` both before and after fitting.

    Plan clause: T4 / linear.py ``metadata`` property.
    """
    cfg = _linear_cfg()
    model = LinearModel(cfg)

    assert model.metadata.name == "linear-ols-weather-only", (
        f"metadata.name before fit must be 'linear-ols-weather-only'; "
        f"got {model.metadata.name!r} (T4 / linear.py metadata property)."
    )

    np.random.seed(0)
    features, target = _make_xy(100, seed=0)
    model.fit(features, target)

    assert model.metadata.name == "linear-ols-weather-only", (
        f"metadata.name after fit must be 'linear-ols-weather-only'; "
        f"got {model.metadata.name!r} (T4 / linear.py metadata property)."
    )


# ---------------------------------------------------------------------------
# 21. test_linear_fit_ignores_extra_feature_columns
# ---------------------------------------------------------------------------


def test_linear_fit_ignores_extra_feature_columns() -> None:
    """Guards linear.py ``fit()`` tolerates extra columns in features DataFrame.

    When ``features`` contains ``("x1", "x2", "noise")`` but config names only
    ``("x1", "x2")``, fit must succeed and ``metadata.feature_columns`` must
    equal exactly ``("x1", "x2")`` — the extra column is silently ignored.

    Plan clause: T4 / linear.py ``fit()`` docstring "extra columns are tolerated
    and ignored".
    """
    np.random.seed(0)
    n = 200
    cfg = _linear_cfg(feature_columns=("x1", "x2"))
    model = LinearModel(cfg)

    idx = _hourly_index(n)
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "noise": rng.standard_normal(n),  # Extra column.
        },
        index=idx,
    )
    target = pd.Series(features["x1"] * 2 + features["x2"] * 3, index=idx, name="nd_mw")

    model.fit(features, target)

    assert model.metadata.feature_columns == ("x1", "x2"), (
        f"metadata.feature_columns must equal ('x1', 'x2') when config names only those "
        f"two columns; extra 'noise' column must be ignored. "
        f"Got {model.metadata.feature_columns!r} (T4 / linear.py fit() docstring)."
    )
