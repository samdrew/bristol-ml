"""Spec-derived tests for ``bristol_ml.registry.save`` — Stage 9 T2.

Every test is derived from:

- ``docs/plans/active/09-model-registry.md`` §6 Task T2 (the four AC-2 save
  tests + the two AC-3 tests).
- ``docs/plans/active/09-model-registry.md`` §4 AC-2 ("every model shipped
  before this stage can save through the registry without code changes to
  the model itself").
- ``docs/plans/active/09-model-registry.md`` §4 AC-3 (automatic git-SHA
  capture; explicit kwargs for ``feature_set`` and ``target``).
- ``docs/plans/active/09-model-registry.md`` §1 D9 (the registry consumes
  ``Model.save`` via the protocol; no model-code changes).
- ``docs/plans/active/09-model-registry.md`` §5 (sidecar JSON schema).

The four AC-2 round-trip agreement checks (``save`` + ``load`` +
``predict`` at ``atol=1e-10``) land in T3 when ``registry.load`` is
implemented; T2 only proves the save side produces the on-disk layout and
sidecar content required by §5.

No production code is modified here.  If any test below fails, the
failure points at a deviation from the spec — do not weaken the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bristol_ml import registry
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from bristol_ml.models.sarimax import SarimaxModel
from bristol_ml.models.scipy_parametric import ScipyParametricModel
from conf._schemas import (
    LinearConfig,
    NaiveConfig,
    SarimaxConfig,
    ScipyParametricConfig,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """UTC-aware hourly DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _fake_metrics_df(n_folds: int = 3, *, with_nan: bool = False) -> pd.DataFrame:
    """Return a Stage-6-harness-shaped per-fold metrics DataFrame.

    Columns follow ``evaluation.harness.evaluate``: ``fold_index``,
    ``train_end``, ``test_start``, ``test_end``, plus ``mae``, ``rmse``,
    ``mape``, ``wape`` (the four Stage 6 metrics).  Set ``with_nan=True``
    to put a ``NaN`` into the last fold's ``mape`` — a regression guard
    for the plan's ``allow_nan=True`` JSON serialisation rule.
    """
    rows = []
    for i in range(n_folds):
        row = {
            "fold_index": i,
            "train_end": pd.Timestamp("2024-01-10 00:00", tz="UTC") + pd.Timedelta(hours=i * 24),
            "test_start": pd.Timestamp("2024-01-10 01:00", tz="UTC") + pd.Timedelta(hours=i * 24),
            "test_end": pd.Timestamp("2024-01-11 00:00", tz="UTC") + pd.Timedelta(hours=i * 24),
            "mae": 100.0 + i,
            "rmse": 150.0 + i,
            "mape": 0.03 + 0.001 * i,
            "wape": 0.028 + 0.001 * i,
        }
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    if with_nan:
        df.loc[df.index[-1], "mape"] = float("nan")
    return df


def _read_sidecar(run_dir: Path) -> dict:
    """Read and JSON-decode the ``run.json`` sidecar from a run directory."""
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _assert_run_layout(tmp_path: Path, run_id: str) -> Path:
    """Assert the D1 on-disk layout and return the run directory."""
    run_dir = tmp_path / run_id
    assert run_dir.is_dir(), f"expected run directory {run_dir} to exist"
    assert (run_dir / "artefact" / "model.joblib").is_file(), (
        "expected artefact/model.joblib to be written by the model protocol"
    )
    assert (run_dir / "run.json").is_file(), "expected run.json sidecar to be written"
    # Atomic-write (D5) invariant: no staging directories left behind.
    assert not list(tmp_path.glob(".tmp_*")), (
        "expected staging directories to have been renamed away"
    )
    return run_dir


def _assert_sidecar_fields(
    sidecar: dict,
    *,
    expected_type: str,
    expected_name: str,
    expected_feature_set: str,
    expected_target: str,
) -> None:
    """Assert the §5 sidecar schema is present with the expected values."""
    required_keys = {
        "run_id",
        "name",
        "type",
        "feature_set",
        "target",
        "feature_columns",
        "fit_utc",
        "git_sha",
        "hyperparameters",
        "metrics",
        "registered_at_utc",
    }
    assert set(sidecar.keys()) == required_keys, (
        f"sidecar keys must match the §5 schema exactly; "
        f"missing={required_keys - set(sidecar.keys())!r}, "
        f"unexpected={set(sidecar.keys()) - required_keys!r}"
    )
    assert sidecar["type"] == expected_type
    assert sidecar["name"] == expected_name
    assert sidecar["feature_set"] == expected_feature_set
    assert sidecar["target"] == expected_target
    # Per-metric summary shape: mean + std + per_fold.
    for metric_name, summary in sidecar["metrics"].items():
        assert set(summary.keys()) == {"mean", "std", "per_fold"}, (
            f"metric {metric_name!r} summary must be {{mean, std, per_fold}}; "
            f"got {set(summary.keys())!r}"
        )


# ---------------------------------------------------------------------------
# AC-2 — NaiveModel save through the registry (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_naive_model_via_protocol(tmp_path: Path) -> None:
    """``registry.save`` accepts a fitted ``NaiveModel`` via the protocol (AC-2)."""
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)

    n = 400
    idx = _hourly_index(n)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    model.fit(features, target)

    metrics_df = _fake_metrics_df()
    run_id = registry.save(
        model,
        metrics_df,
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    assert run_id.startswith("naive-"), (
        f"run_id must begin with the naive metadata-name prefix; got {run_id!r}"
    )
    run_dir = _assert_run_layout(tmp_path, run_id)
    sidecar = _read_sidecar(run_dir)
    _assert_sidecar_fields(
        sidecar,
        expected_type="naive",
        expected_name=model.metadata.name,
        expected_feature_set="weather_only",
        expected_target="nd_mw",
    )
    # Metrics summary is populated from the per-fold DataFrame.
    assert set(sidecar["metrics"].keys()) == {"mae", "rmse", "mape", "wape"}
    mae_per_fold = sidecar["metrics"]["mae"]["per_fold"]
    assert mae_per_fold == [100.0, 101.0, 102.0], (
        f"per_fold list must echo the metrics DataFrame; got {mae_per_fold!r}"
    )
    assert sidecar["metrics"]["mae"]["mean"] == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# AC-2 — LinearModel save through the registry (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_linear_model_via_protocol(tmp_path: Path) -> None:
    """``registry.save`` accepts a fitted ``LinearModel`` via the protocol (AC-2)."""
    cfg = LinearConfig(feature_columns=("x1", "x2"), target_column="nd_mw", fit_intercept=True)
    model = LinearModel(cfg)

    n = 200
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * x1 + 3.0 * x2 + 1.0 + rng.normal(0.0, 0.01, n)
    features = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model.fit(features, target)

    metrics_df = _fake_metrics_df()
    run_id = registry.save(
        model,
        metrics_df,
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    run_dir = _assert_run_layout(tmp_path, run_id)
    sidecar = _read_sidecar(run_dir)
    _assert_sidecar_fields(
        sidecar,
        expected_type="linear",
        expected_name=model.metadata.name,
        expected_feature_set="weather_only",
        expected_target="nd_mw",
    )
    # feature_columns on the sidecar is a *list* (JSON has no tuple type).
    assert sidecar["feature_columns"] == list(model.metadata.feature_columns)
    # hyperparameters survive a JSON round-trip and include the fitted coefficients.
    assert "coefficients" in sidecar["hyperparameters"]
    coef_dict = sidecar["hyperparameters"]["coefficients"]
    assert set(coef_dict.keys()) >= {"const", "x1", "x2"}


# ---------------------------------------------------------------------------
# AC-2 — SarimaxModel save through the registry (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_sarimax_model_via_protocol(tmp_path: Path) -> None:
    """``registry.save`` accepts a fitted ``SarimaxModel`` via the protocol (AC-2)."""
    # Small, fast SARIMAX — matches the _FAST_CONFIG idiom in Stage 7 tests.
    cfg = SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 24),
        weekly_fourier_harmonics=2,
    )
    model = SarimaxModel(cfg)

    n = 200
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    temp_c = rng.normal(10.0, 5.0, n)
    cloud_cover = rng.uniform(0.0, 1.0, n)
    t = np.arange(n, dtype=np.float64)
    target_vals = np.zeros(n)
    target_vals[0] = 10_000.0
    for i in range(1, n):
        target_vals[i] = (
            0.7 * target_vals[i - 1]
            + 0.3 * 10_000.0
            + 500.0 * np.sin(2.0 * np.pi * t[i] / 24.0)
            + rng.normal(0.0, 200.0)
        )
    features = pd.DataFrame({"temp_c": temp_c, "cloud_cover": cloud_cover}, index=idx)
    target = pd.Series(target_vals, index=idx, name="nd_mw")
    model.fit(features, target)

    metrics_df = _fake_metrics_df()
    run_id = registry.save(
        model,
        metrics_df,
        feature_set="weather_calendar",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    run_dir = _assert_run_layout(tmp_path, run_id)
    sidecar = _read_sidecar(run_dir)
    _assert_sidecar_fields(
        sidecar,
        expected_type="sarimax",
        expected_name=model.metadata.name,
        expected_feature_set="weather_calendar",
        expected_target="nd_mw",
    )


# ---------------------------------------------------------------------------
# AC-2 — ScipyParametricModel save through the registry (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_scipy_parametric_model_via_protocol(tmp_path: Path) -> None:
    """``registry.save`` accepts a fitted ``ScipyParametricModel`` via the protocol (AC-2).

    Guards plan §8 R3 — ``ScipyParametricModel.metadata.hyperparameters``
    may contain ``float("inf")`` entries in the ``param_std_errors`` /
    ``covariance_matrix`` fields when ``pcov`` has non-finite diagonals.
    The registry's ``json.dumps(..., allow_nan=True)`` lets these round-trip
    (as the non-strict-JSON ``Infinity`` token) rather than raising.
    """
    cfg = ScipyParametricConfig()
    model = ScipyParametricModel(cfg)

    n = 500
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    temperature = rng.uniform(5.0, 20.0, n)
    target_vals = 10_000.0 + rng.normal(0.0, 200.0, n)
    features = pd.DataFrame({"temperature_2m": temperature}, index=idx)
    target = pd.Series(target_vals, index=idx, name="nd_mw")
    model.fit(features, target)

    metrics_df = _fake_metrics_df()
    run_id = registry.save(
        model,
        metrics_df,
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    run_dir = _assert_run_layout(tmp_path, run_id)
    sidecar = _read_sidecar(run_dir)
    _assert_sidecar_fields(
        sidecar,
        expected_type="scipy_parametric",
        expected_name=model.metadata.name,
        expected_feature_set="weather_only",
        expected_target="nd_mw",
    )
    # Covariance matrix survives a JSON round-trip (may contain Infinity).
    assert "covariance_matrix" in sidecar["hyperparameters"], (
        "ScipyParametricModel's hyperparameters must include covariance_matrix "
        "(plan §8 R3 — round-trip with allow_nan=True)"
    )


# ---------------------------------------------------------------------------
# AC-3 — git SHA auto-capture (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_captures_git_sha_automatically(tmp_path: Path) -> None:
    """``save`` stores a non-empty hex git SHA without the caller supplying it (AC-3).

    CI runs inside the project working tree, so the helper returns a
    populated SHA.  The sidecar field is a plain ``str`` — ``None`` is a
    legitimate registry state outside a git tree (tested separately in
    ``test_registry_fs.py``) but must not appear inside the tree.
    """
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 200
    idx = _hourly_index(n)
    features = pd.DataFrame({"t2m": np.zeros(n)}, index=idx)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    model.fit(features, target)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )
    sidecar = _read_sidecar(tmp_path / run_id)
    sha = sidecar["git_sha"]
    assert isinstance(sha, str), (
        f"git_sha must be a str inside the project working tree; got {sha!r} (AC-3)"
    )
    assert len(sha) >= 7, f"git_sha should be at least 7 hex chars; got {sha!r}"
    assert all(c in "0123456789abcdef" for c in sha.lower()), (
        f"git_sha should be lowercase hex; got {sha!r}"
    )


# ---------------------------------------------------------------------------
# AC-3 — missing required explicit kwarg (plan T2 named test)
# ---------------------------------------------------------------------------


def test_registry_save_raises_on_missing_required_explicit_field(tmp_path: Path) -> None:
    """``save()`` without the ``feature_set`` kwarg raises ``TypeError`` (AC-3).

    ``feature_set`` and ``target`` are declared after ``*`` so they are
    keyword-only; Python raises ``TypeError`` when either is omitted, so
    the registry does not need an explicit guard.
    """
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 200
    idx = _hourly_index(n)
    features = pd.DataFrame({"t2m": np.zeros(n)}, index=idx)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    model.fit(features, target)

    metrics_df = _fake_metrics_df()

    # Missing feature_set.
    with pytest.raises(TypeError):
        registry.save(  # type: ignore[call-arg]
            model,
            metrics_df,
            target="nd_mw",
            registry_dir=tmp_path,
        )

    # Missing target.
    with pytest.raises(TypeError):
        registry.save(  # type: ignore[call-arg]
            model,
            metrics_df,
            feature_set="weather_only",
            registry_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Defensive guard — unfitted model is rejected
# ---------------------------------------------------------------------------


def test_registry_save_rejects_unfitted_model(tmp_path: Path) -> None:
    """``save()`` on a model with ``metadata.fit_utc is None`` raises ``RuntimeError``.

    Ensures the registry does not silently write a "run" whose artefact is
    an unfitted estimator — the Stage 6 harness guarantees a fitted model
    on the save path, but the public API is defensive anyway.
    """
    model = NaiveModel(NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw"))
    assert model.metadata.fit_utc is None

    with pytest.raises(RuntimeError, match="fitted"):
        registry.save(
            model,
            _fake_metrics_df(),
            feature_set="weather_only",
            target="nd_mw",
            registry_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Defensive guard — unknown model class is rejected
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T3 — AC-2 round-trip: save + load + predict agreement to atol=1e-10
# ---------------------------------------------------------------------------


def _assert_round_trip_predict(
    original: object,
    loaded: object,
    features: pd.DataFrame,
    *,
    atol: float = 1e-10,
) -> None:
    """Assert ``predict(features)`` agrees between original and loaded to ``atol``."""
    predicted_original = original.predict(features)  # type: ignore[attr-defined]
    predicted_loaded = loaded.predict(features)  # type: ignore[attr-defined]
    np.testing.assert_allclose(
        predicted_loaded.to_numpy(),
        predicted_original.to_numpy(),
        atol=atol,
        err_msg=(
            "registry.load round-trip broke predict() numerical equivalence; "
            "the model artefact or dispatcher is dropping state."
        ),
    )


def test_registry_load_round_trips_naive_model(tmp_path: Path) -> None:
    """save + load + predict on ``NaiveModel`` agrees to atol=1e-10 (AC-2 round-trip)."""
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 400
    idx = _hourly_index(n)
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    model.fit(features, target)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )
    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, NaiveModel)
    # Use the tail of the fitted range for predictions (naive lookback=168).
    predict_features = features.iloc[200:]
    _assert_round_trip_predict(model, loaded, predict_features)


def test_registry_load_round_trips_linear_model(tmp_path: Path) -> None:
    """save + load + predict on ``LinearModel`` agrees to atol=1e-10 (AC-2 round-trip)."""
    cfg = LinearConfig(feature_columns=("x1", "x2"), target_column="nd_mw", fit_intercept=True)
    model = LinearModel(cfg)
    n = 200
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * x1 + 3.0 * x2 + 1.0 + rng.normal(0.0, 0.01, n)
    features = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model.fit(features, target)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )
    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, LinearModel)
    _assert_round_trip_predict(model, loaded, features)


def test_registry_load_round_trips_sarimax_model(tmp_path: Path) -> None:
    """save + load + predict on ``SarimaxModel`` agrees to atol=1e-10 (AC-2 round-trip)."""
    cfg = SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 24),
        weekly_fourier_harmonics=2,
    )
    model = SarimaxModel(cfg)
    n = 200
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    temp_c = rng.normal(10.0, 5.0, n)
    cloud_cover = rng.uniform(0.0, 1.0, n)
    t = np.arange(n, dtype=np.float64)
    target_vals = np.zeros(n)
    target_vals[0] = 10_000.0
    for i in range(1, n):
        target_vals[i] = (
            0.7 * target_vals[i - 1]
            + 0.3 * 10_000.0
            + 500.0 * np.sin(2.0 * np.pi * t[i] / 24.0)
            + rng.normal(0.0, 200.0)
        )
    features = pd.DataFrame({"temp_c": temp_c, "cloud_cover": cloud_cover}, index=idx)
    target = pd.Series(target_vals, index=idx, name="nd_mw")
    model.fit(features, target)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_calendar",
        target="nd_mw",
        registry_dir=tmp_path,
    )
    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, SarimaxModel)
    _assert_round_trip_predict(model, loaded, features)


def test_registry_load_round_trips_scipy_parametric_model(tmp_path: Path) -> None:
    """save + load + predict on ``ScipyParametricModel`` agrees to atol=1e-10 (AC-2 round-trip)."""
    cfg = ScipyParametricConfig()
    model = ScipyParametricModel(cfg)
    n = 500
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    temperature = rng.uniform(5.0, 20.0, n)
    target_vals = 10_000.0 + rng.normal(0.0, 200.0, n)
    features = pd.DataFrame({"temperature_2m": temperature}, index=idx)
    target = pd.Series(target_vals, index=idx, name="nd_mw")
    model.fit(features, target)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )
    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, ScipyParametricModel)
    _assert_round_trip_predict(model, loaded, features)


# ---------------------------------------------------------------------------
# T3 — load error branches
# ---------------------------------------------------------------------------


def test_registry_load_raises_on_missing_run_id(tmp_path: Path) -> None:
    """``load`` on a non-existent ``run_id`` raises ``FileNotFoundError`` (plan T3)."""
    with pytest.raises(FileNotFoundError, match="No registered run"):
        registry.load("does_not_exist_20260101T0000", registry_dir=tmp_path)


def test_registry_load_named_linear_returns_base_class(tmp_path: Path) -> None:
    """Loading a ``_NamedLinearModel`` run returns a base ``LinearModel`` (plan D16, T3).

    Guards the plan D16 cut — the dynamic name is preserved on the sidecar
    for reading but is not re-applied to the loaded instance.  ``_NamedLinearModel.save``
    delegates to the inner ``LinearModel.save``, so the on-disk artefact is
    already a plain ``LinearModel``; the registry loads the base class and
    the sidecar's ``name`` field carries the dynamic name forward.
    """
    # Import the wrapper from the train module (where it lives).
    from bristol_ml.train import _NamedLinearModel

    cfg = LinearConfig(feature_columns=("x1",), target_column="nd_mw", fit_intercept=True)
    wrapper = _NamedLinearModel(cfg, metadata_name="linear-ols-weather-only")

    n = 50
    rng = np.random.default_rng(0)
    idx = _hourly_index(n)
    x1 = rng.standard_normal(n)
    y = 2.0 * x1 + rng.normal(0.0, 0.01, n)
    features = pd.DataFrame({"x1": x1}, index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    wrapper.fit(features, target)

    run_id = registry.save(
        wrapper,  # type: ignore[arg-type]
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    # Sidecar records the dynamic name …
    sidecar = _read_sidecar(tmp_path / run_id)
    assert sidecar["name"] == "linear-ols-weather-only"
    assert sidecar["type"] == "linear"  # dispatched via class name, D16

    # … but load returns the base LinearModel (not the wrapper).
    loaded = registry.load(run_id, registry_dir=tmp_path)
    assert isinstance(loaded, LinearModel)
    assert not isinstance(loaded, _NamedLinearModel)


def test_registry_save_rejects_unknown_model_class(tmp_path: Path) -> None:
    """``save()`` raises ``TypeError`` on a model whose class is not registered.

    Guards codebase hazard H4 — new model families must be added to
    :mod:`bristol_ml.registry._dispatch` before they can be registered.
    A stand-in class mimicking the ``Model`` protocol surface triggers
    the dispatcher's unknown-class branch.
    """

    from datetime import UTC, datetime

    from bristol_ml.models.protocol import ModelMetadata

    class _UnregisteredModel:
        """A protocol-conformant shape the dispatcher does not know about."""

        @property
        def metadata(self) -> ModelMetadata:
            return ModelMetadata(
                name="unregistered-sham",
                feature_columns=("t2m",),
                fit_utc=datetime.now(UTC),
                git_sha=None,
                hyperparameters={},
            )

        def save(self, path: Path) -> None:  # pragma: no cover — never reached
            path.write_bytes(b"")

    with pytest.raises(TypeError, match="Cannot register"):
        registry.save(
            _UnregisteredModel(),  # type: ignore[arg-type]
            _fake_metrics_df(),
            feature_set="weather_only",
            target="nd_mw",
            registry_dir=tmp_path,
        )
