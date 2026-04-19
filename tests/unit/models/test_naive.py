"""Spec-derived tests for ``bristol_ml.models.naive.NaiveModel``.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T3 (acceptance criteria
  and named test list).
- ``docs/plans/active/04-linear-baseline.md`` §1 D1 (three strategies:
  ``same_hour_yesterday``, ``same_hour_last_week``,
  ``same_hour_same_weekday``).
- ``docs/plans/active/04-linear-baseline.md`` §10 risk register (re-entrancy /
  second ``fit()`` discards prior state).
- ``docs/plans/active/04-linear-baseline.md`` §4 AC-2 (interface implementable
  in very few lines), AC-3 (save/load reproduces identical predictions),
  AC-7 (protocol conformance test exists).
- ``src/bristol_ml/models/naive.py`` module docstring + inline contracts.
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section.

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan clause, AC, or F-number it guards.
- ``tmp_path`` (pytest built-in) for filesystem operations.
- ``pd.date_range(..., tz="UTC")`` for all timestamp indices.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

from bristol_ml.models.io import save_joblib
from bristol_ml.models.naive import NaiveModel
from bristol_ml.models.protocol import Model
from conf._schemas import NaiveConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _naive_cfg(strategy: str = "same_hour_last_week", target_column: str = "nd_mw") -> NaiveConfig:
    """Return a ``NaiveConfig`` with the given strategy, defaulting to ``same_hour_last_week``."""
    return NaiveConfig(strategy=strategy, target_column=target_column)  # type: ignore[arg-type]


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """Return a UTC-aware hourly DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _target_series(n: int, start: str = "2024-01-01 00:00", name: str = "nd_mw") -> pd.Series:
    """Return a target series ``y[i] = float(i)`` with a UTC hourly index."""
    idx = _hourly_index(n, start=start)
    return pd.Series([float(i) for i in range(n)], index=idx, name=name)


def _features_df(n: int, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """Return a single-column feature DataFrame with a UTC hourly index."""
    idx = _hourly_index(n, start=start)
    return pd.DataFrame({"t2m": [float(i) * 0.1 for i in range(n)]}, index=idx)


# ---------------------------------------------------------------------------
# 1. test_naive_fit_stores_lookup_table (T3 plan named test)
# ---------------------------------------------------------------------------


def test_naive_fit_stores_lookup_table() -> None:
    """Guards T3 (plan §6 named test): fit on 200-row hourly fixture populates metadata.

    After a successful ``fit()``:
    - ``metadata.feature_columns`` reflects the features DataFrame columns.
    - ``metadata.fit_utc`` is not ``None`` and is tz-aware (UTC).
    - ``metadata.name`` encodes the strategy.

    Plan clause: T3 acceptance / AC-7.
    """
    n = 200
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = _features_df(n)
    target = _target_series(n)

    model.fit(features, target)

    meta = model.metadata
    assert meta.feature_columns == ("t2m",), (
        f"metadata.feature_columns must equal ('t2m',) after fitting on a single-column "
        f"features DataFrame; got {meta.feature_columns!r} (T3 plan named test)."
    )
    assert meta.fit_utc is not None, (
        "metadata.fit_utc must be populated after fit() (T3 plan named test)."
    )
    assert meta.fit_utc.tzinfo is not None, (
        "metadata.fit_utc must be tz-aware (T3 plan named test)."
    )
    assert meta.fit_utc.tzinfo == UTC or str(meta.fit_utc.tzinfo) in ("UTC", "utc"), (
        f"metadata.fit_utc must be UTC-aware; got tzinfo={meta.fit_utc.tzinfo!r} "
        "(T3 plan named test)."
    )


# ---------------------------------------------------------------------------
# 2. test_naive_predict_same_hour_last_week (T3 plan named test, D1 default)
# ---------------------------------------------------------------------------


def test_naive_predict_same_hour_last_week() -> None:
    """Guards T3 (plan §6 named test): predictions equal ``y[t - 168h]`` exactly.

    A 336-row (14-day) hourly UTC-aware fixture with ``y[i] = float(i)``.
    Fit on the full 336 rows; predict on ``idx[168:200]``.  Each prediction
    must equal ``y[t - 168h]`` — i.e. for ``i`` in ``[168, 200)``,
    ``p[i - 168] == float(i - 168)``.

    Plan clause: T3 plan §6 named test / D1 / AC-2.
    """
    n = 336  # 14 days * 24 h
    cfg = _naive_cfg(strategy="same_hour_last_week")
    model = NaiveModel(cfg)

    features = _features_df(n)
    target = _target_series(n)

    model.fit(features, target)

    predict_features = features.iloc[168:200]
    preds = model.predict(predict_features)

    assert len(preds) == 32, f"Expected 32 predictions; got {len(preds)}."
    for loc, idx_val in enumerate(predict_features.index):
        expected_val = float(168 + loc - 168)  # == float(loc)
        # y[t - 168h]: t is idx[168 + loc], look-back is idx[loc]
        expected_val = float(loc)
        assert preds.iloc[loc] == expected_val, (
            f"Prediction at position {loc} (t={idx_val}) must equal y[t-168h] = "
            f"{expected_val}; got {preds.iloc[loc]} (T3 plan named test / D1)."
        )


# ---------------------------------------------------------------------------
# 3. test_naive_predict_raises_when_lookback_missing (T3 plan named test)
# ---------------------------------------------------------------------------


def test_naive_predict_raises_when_lookback_missing() -> None:
    """Guards T3 (plan §6 named test): missing look-back rows raise ``ValueError``.

    Fit on the first 72 rows of a 200-row series.  Predict on the first 10
    rows — each of these requires a target at ``t - 168h``, which precedes the
    training window.  The ``ValueError`` message must name the strategy and the
    missing-row count.

    Plan clause: T3 plan §6 named test / §10 risk register.
    """
    n = 200
    cfg = _naive_cfg(strategy="same_hour_last_week")
    model = NaiveModel(cfg)

    features = _features_df(n)
    target = _target_series(n)

    model.fit(features.iloc[:72], target.iloc[:72])

    with pytest.raises(ValueError) as exc_info:
        model.predict(features.iloc[:10])

    msg = str(exc_info.value)
    assert "same_hour_last_week" in msg, (
        f"ValueError message must name the strategy 'same_hour_last_week'; got {msg!r} "
        "(T3 plan §6 named test)."
    )
    # The message must include a numeric count of missing rows.
    import re

    assert re.search(r"\d+", msg), (
        f"ValueError message must include the missing-row count; got {msg!r} "
        "(T3 plan §6 named test)."
    )


# ---------------------------------------------------------------------------
# 4. test_naive_save_load_round_trip (T3 plan named test, AC-3)
# ---------------------------------------------------------------------------


def test_naive_save_load_round_trip(tmp_path: Path) -> None:
    """Guards T3 (plan §6 named test) and AC-3: save/load reproduces identical predictions.

    Fit a model, save it to ``tmp_path / "naive.joblib"``, load it back, and
    assert that predictions on the same test frame are element-wise identical.
    The loaded object must also conform to the ``Model`` protocol.

    Plan clause: T3 plan §6 named test / AC-3 / F-10.
    """
    n = 336
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = _features_df(n)
    target = _target_series(n)
    model.fit(features, target)

    path = tmp_path / "naive.joblib"
    model.save(path)

    assert path.exists(), f"save() must create the artefact at {path}."

    loaded = NaiveModel.load(path)

    # Protocol conformance of the loaded object.
    assert isinstance(loaded, Model), (
        "Loaded NaiveModel must satisfy isinstance(loaded, Model) (AC-3 / AC-7)."
    )

    test_features = features.iloc[168:200]
    original_preds = model.predict(test_features)
    loaded_preds = loaded.predict(test_features)

    pd.testing.assert_series_equal(
        original_preds,
        loaded_preds,
        check_exact=True,
        obj="save/load predictions",
    )


# ---------------------------------------------------------------------------
# 5. test_naive_conforms_to_model_protocol (T3 plan named test, AC-7)
# ---------------------------------------------------------------------------


def test_naive_conforms_to_model_protocol() -> None:
    """Guards T3 (plan §6 named test) and AC-7: ``isinstance(NaiveModel(cfg), Model)`` is True.

    The ``@runtime_checkable`` protocol check verifies attribute presence for
    all five required members: ``fit``, ``predict``, ``save``, ``load``,
    ``metadata``.

    Plan clause: T3 plan §6 named test / AC-7 / D3.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    assert isinstance(model, Model), (
        "NaiveModel must pass isinstance(model, Model) (T3 plan §6 named test / AC-7)."
    )


# ---------------------------------------------------------------------------
# 6. test_naive_predict_same_hour_yesterday
# ---------------------------------------------------------------------------


def test_naive_predict_same_hour_yesterday() -> None:
    """Guards D1: ``same_hour_yesterday`` strategy returns ``y[t - 24h]`` exactly.

    A 72-row (3-day) hourly fixture with ``y[i] = float(i)``.
    Predict on rows ``idx[24:48]`` — each must equal ``y[t - 24h]``,
    i.e. ``float(i - 24)`` for ``i`` in ``[24, 48)``.

    Plan clause: T3 / D1 (strategy ablation) / AC-2.
    """
    n = 72
    cfg = _naive_cfg(strategy="same_hour_yesterday")
    model = NaiveModel(cfg)
    features = _features_df(n)
    target = _target_series(n)

    model.fit(features, target)

    predict_features = features.iloc[24:48]
    preds = model.predict(predict_features)

    assert len(preds) == 24
    for loc in range(24):
        expected = float(loc)  # y[t - 24h] = y[(24 + loc) - 24] = y[loc]
        assert preds.iloc[loc] == expected, (
            f"same_hour_yesterday prediction at loc={loc} must equal y[t-24h]={expected}; "
            f"got {preds.iloc[loc]} (D1 same_hour_yesterday)."
        )


# ---------------------------------------------------------------------------
# 7. test_naive_predict_same_hour_same_weekday
# ---------------------------------------------------------------------------


def test_naive_predict_same_hour_same_weekday() -> None:
    """Guards D1: ``same_hour_same_weekday`` returns the latest matching (weekday, hour) target.

    A fixture spanning 3 full weeks (504 rows = 21 days * 24 h).
    Fit on the first 2 weeks (rows 0-335, i.e. 14 days * 24 h).
    Predict on the first 24 rows of week 3 (idx[336:360]).
    For each prediction timestamp ``t`` in week 3, the most-recent training
    row with matching (weekday, hour) must be in week 2 (offset 168 hours
    behind), so the expected look-up value is ``y[t - 168h]``.

    Plan clause: T3 / D1 (same_hour_same_weekday strategy) / AC-2.
    """
    n = 504  # 21 days
    cfg = _naive_cfg(strategy="same_hour_same_weekday")
    model = NaiveModel(cfg)
    features = _features_df(n)
    target = _target_series(n)

    # Fit on the first two weeks.
    model.fit(features.iloc[:336], target.iloc[:336])

    # Predict on the first 24 rows of week 3.
    predict_features = features.iloc[336:360]
    preds = model.predict(predict_features)

    assert len(preds) == 24
    for loc in range(24):
        # Week 3 hour `loc` corresponds to week 2 hour `loc` (exactly 168 h earlier)
        # because the fixture is contiguous and ``y[i] = float(i)``.
        t = predict_features.index[loc]
        expected_t = t - pd.Timedelta(hours=168)
        expected_val = target.loc[expected_t]
        assert preds.iloc[loc] == expected_val, (
            f"same_hour_same_weekday at loc={loc} (t={t}) must look up "
            f"y[{expected_t}]={expected_val}; got {preds.iloc[loc]} "
            "(D1 same_hour_same_weekday)."
        )


# ---------------------------------------------------------------------------
# 8. test_naive_predict_before_fit_raises_runtime_error
# ---------------------------------------------------------------------------


def test_naive_predict_before_fit_raises_runtime_error() -> None:
    """Guards naive.py contract: ``predict()`` before ``fit()`` raises ``RuntimeError``.

    A freshly constructed ``NaiveModel`` has no fitted state.  Any call to
    ``predict()`` before ``fit()`` must raise ``RuntimeError`` rather than
    returning stale or silently incorrect output.

    Plan clause: T3 / models CLAUDE.md "Predict-before-fit" protocol semantic.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = _features_df(10)

    with pytest.raises(RuntimeError):
        model.predict(features)


# ---------------------------------------------------------------------------
# 9. test_naive_save_before_fit_raises_runtime_error
# ---------------------------------------------------------------------------


def test_naive_save_before_fit_raises_runtime_error(tmp_path: Path) -> None:
    """Guards naive.py contract: ``save()`` before ``fit()`` raises ``RuntimeError``.

    Persisting unfitted state must be refused; the implementation raises
    ``RuntimeError`` to prevent an empty-state artefact being written to disk.

    Plan clause: T3 / naive.py ``save()`` docstring.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)

    with pytest.raises(RuntimeError):
        model.save(tmp_path / "x.joblib")


# ---------------------------------------------------------------------------
# 10. test_naive_load_rejects_wrong_artefact_type
# ---------------------------------------------------------------------------


def test_naive_load_rejects_wrong_artefact_type(tmp_path: Path) -> None:
    """Guards naive.py ``load()`` docstring: wrong artefact type raises ``TypeError``.

    A plain dict ``{"not": "a model"}`` written via ``save_joblib`` is not a
    ``NaiveModel``.  ``NaiveModel.load(path)`` must raise ``TypeError`` rather
    than silently returning the wrong class.

    Plan clause: T3 / naive.py ``load()`` ``TypeError`` contract.
    """
    path = tmp_path / "wrong.joblib"
    save_joblib({"not": "a model"}, path)

    with pytest.raises(TypeError):
        NaiveModel.load(path)


# ---------------------------------------------------------------------------
# 11. test_naive_fit_rejects_non_datetime_index
# ---------------------------------------------------------------------------


def test_naive_fit_rejects_non_datetime_index() -> None:
    """Guards naive.py ``fit()`` contract: ``RangeIndex`` on target raises ``TypeError``.

    The seasonal-naive look-up is a time-offset arithmetic operation; a plain
    integer index makes look-ups meaningless.  ``fit()`` must reject non-
    ``DatetimeIndex`` targets with ``TypeError``.

    Plan clause: T3 / naive.py ``fit()`` ``TypeError`` docstring.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = pd.DataFrame({"t2m": [1.0, 2.0, 3.0]})  # RangeIndex
    target = pd.Series([1.0, 2.0, 3.0])  # RangeIndex

    with pytest.raises(TypeError):
        model.fit(features, target)


# ---------------------------------------------------------------------------
# 12. test_naive_fit_rejects_length_mismatch
# ---------------------------------------------------------------------------


def test_naive_fit_rejects_length_mismatch() -> None:
    """Guards naive.py ``fit()`` contract: mismatched feature/target lengths raise ``ValueError``.

    ``features`` with 100 rows and ``target`` with 99 rows represents a data
    alignment error.  ``fit()`` must detect and raise ``ValueError`` rather
    than silently ignoring the extra row.

    Plan clause: T3 / naive.py ``fit()`` ``ValueError`` docstring.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = _features_df(100)
    target = _target_series(99)

    with pytest.raises(ValueError):
        model.fit(features, target)


# ---------------------------------------------------------------------------
# 13. test_naive_fit_rejects_non_monotonic_index
# ---------------------------------------------------------------------------


def test_naive_fit_rejects_non_monotonic_index() -> None:
    """Guards naive.py ``fit()`` contract: reversed DatetimeIndex raises ``ValueError``.

    The fixed-lag look-up requires a strictly ascending index so that
    ``reindex`` on ``t - lag`` produces unambiguous matches.  A reversed
    (descending) index must be rejected.

    Plan clause: T3 / naive.py ``fit()`` ``ValueError`` docstring.
    """
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    n = 50
    idx = _hourly_index(n)[::-1]  # Reversed — not monotonic increasing.
    features = pd.DataFrame({"t2m": list(range(n))}, index=idx)
    target = pd.Series(list(range(n)), index=idx, dtype="float64")

    with pytest.raises(ValueError):
        model.fit(features, target)


# ---------------------------------------------------------------------------
# 14. test_naive_predict_rejects_non_datetime_index
# ---------------------------------------------------------------------------


def test_naive_predict_rejects_non_datetime_index() -> None:
    """Guards naive.py ``predict()`` contract: ``RangeIndex`` on features raises ``TypeError``.

    After a valid ``fit()``, the model must still reject features whose index
    is not a ``DatetimeIndex``, because the time-offset arithmetic would be
    undefined.

    Plan clause: T3 / naive.py ``predict()`` ``TypeError`` docstring.
    """
    n = 336
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    model.fit(_features_df(n), _target_series(n))

    bad_features = pd.DataFrame({"t2m": [1.0, 2.0, 3.0]})  # RangeIndex

    with pytest.raises(TypeError):
        model.predict(bad_features)


# ---------------------------------------------------------------------------
# 15. test_naive_metadata_before_fit
# ---------------------------------------------------------------------------


def test_naive_metadata_before_fit() -> None:
    """Guards models CLAUDE.md "metadata before fit" semantic.

    Before any ``fit()`` call, ``metadata`` must be observable with:
    - ``fit_utc is None``
    - ``feature_columns == ()``
    - ``name.startswith("naive-")``
    - ``hyperparameters["strategy"] == "same_hour_last_week"`` (the default).

    Plan clause: T3 / models CLAUDE.md protocol-semantics / naive.py metadata
    property docstring.
    """
    cfg = _naive_cfg()  # default strategy = same_hour_last_week
    model = NaiveModel(cfg)
    meta = model.metadata

    assert meta.fit_utc is None, (
        f"metadata.fit_utc must be None before fit(); got {meta.fit_utc!r} "
        "(models CLAUDE.md / naive.py metadata contract)."
    )
    assert meta.feature_columns == (), (
        f"metadata.feature_columns must be () before fit(); got {meta.feature_columns!r} "
        "(models CLAUDE.md / naive.py metadata contract)."
    )
    assert meta.name.startswith("naive-"), (
        f"metadata.name must start with 'naive-'; got {meta.name!r} (naive.py metadata property)."
    )
    assert meta.hyperparameters["strategy"] == "same_hour_last_week", (
        f"metadata.hyperparameters['strategy'] must equal 'same_hour_last_week' (the default); "
        f"got {meta.hyperparameters['strategy']!r} (naive.py metadata property)."
    )


# ---------------------------------------------------------------------------
# 16. test_naive_metadata_name_hyphenated
# ---------------------------------------------------------------------------


def test_naive_metadata_name_hyphenated() -> None:
    """Guards naive.py ``metadata`` property: underscores in strategy become hyphens in name.

    ``strategy="same_hour_last_week"`` must produce
    ``metadata.name == "naive-same-hour-last-week"``.

    Plan clause: T3 / naive.py metadata property comment ("convert underscores … to hyphens").
    """
    cfg = _naive_cfg(strategy="same_hour_last_week")
    model = NaiveModel(cfg)
    assert model.metadata.name == "naive-same-hour-last-week", (
        f"metadata.name for strategy='same_hour_last_week' must be "
        f"'naive-same-hour-last-week'; got {model.metadata.name!r} "
        "(naive.py metadata property)."
    )


# ---------------------------------------------------------------------------
# 17. test_naive_refit_is_re_entrant  (plan §10 risk register)
# ---------------------------------------------------------------------------


def test_naive_refit_is_re_entrant() -> None:
    """Guards plan §10 risk register: second ``fit()`` discards previous state.

    Fit once on a dataset where all values are 0.0; fit again on a dataset
    where all values are 1.0.  Predictions after the second fit must reflect
    the second dataset only.

    Plan clause: T3 / §10 risk register row "fit() must be re-entrant" /
    models CLAUDE.md "Re-entrancy" protocol semantic.
    """
    n = 336
    cfg = _naive_cfg()

    features_a = _features_df(n)
    target_a = pd.Series([0.0] * n, index=_hourly_index(n), name="nd_mw")

    features_b = _features_df(n, start="2024-02-01 00:00")
    target_b = pd.Series([1.0] * n, index=_hourly_index(n, start="2024-02-01 00:00"), name="nd_mw")

    model = NaiveModel(cfg)
    model.fit(features_a, target_a)
    model.fit(features_b, target_b)

    # Predict on a slice of the second dataset.
    test_features = features_b.iloc[168:200]
    preds = model.predict(test_features)

    assert (preds == 1.0).all(), (
        "After re-fit on target_b (all 1.0), predictions must reflect target_b, "
        f"not target_a (all 0.0); got min={preds.min()} max={preds.max()} "
        "(plan §10 risk register / models CLAUDE.md re-entrancy)."
    )


# ---------------------------------------------------------------------------
# 18. test_naive_predict_series_name_matches_target_column
# ---------------------------------------------------------------------------


def test_naive_predict_series_name_matches_target_column() -> None:
    """Guards naive.py contract: returned Series ``name`` matches ``config.target_column``.

    The plan states the returned series carries ``name=config.target_column``.
    Override ``target_column`` to ``"target_col"`` and assert the returned
    series has the same name.

    Plan clause: T3 / naive.py ``predict()`` return shape contract /
    naive.py class docstring.
    """
    n = 336
    cfg = _naive_cfg(target_column="target_col")
    model = NaiveModel(cfg)

    features = _features_df(n)
    target = pd.Series([float(i) for i in range(n)], index=_hourly_index(n), name="target_col")
    model.fit(features, target)

    test_features = features.iloc[168:200]
    preds = model.predict(test_features)

    assert preds.name == "target_col", (
        f"Returned Series name must equal config.target_column='target_col'; "
        f"got {preds.name!r} (naive.py class docstring / predict contract)."
    )


# ---------------------------------------------------------------------------
# 19. test_naive_predict_series_index_matches_features
# ---------------------------------------------------------------------------


def test_naive_predict_series_index_matches_features() -> None:
    """Guards naive.py contract: returned Series index equals ``features.index`` element-wise.

    The plan states predictions are "indexed to the ``features`` DataFrame
    passed to ``predict``" so the harness can align them with the target slice.

    Plan clause: T3 / naive.py class docstring / harness alignment contract.
    """
    n = 336
    cfg = _naive_cfg()
    model = NaiveModel(cfg)
    features = _features_df(n)
    target = _target_series(n)
    model.fit(features, target)

    test_features = features.iloc[168:200]
    preds = model.predict(test_features)

    pd.testing.assert_index_equal(
        preds.index,
        test_features.index,
        obj="predict() return Series index vs features.index",
    )
