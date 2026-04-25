"""Local pytest fixtures + helpers for Stage 12 serving integration tests.

The helpers below build small per-family fitted-model fixtures and
register them into a tmp registry, so the AC-3 prediction-parity test
can parametrise over all six model families without each test having
to re-derive the fitting recipe.

The factories mirror the per-family helpers in
``tests/unit/registry/test_registry_skops_migration.py`` deliberately:
both files exist so the integration suite is self-contained and does
not cross-import from the unit-test tree (tests/integration must not
depend on tests/unit).  If a factory drifts between the two files the
divergence is intentional — the serving tests live their own life
and the unit-test factories may evolve under different pressure.

The fixtures here are session-scoped where it cuts wall-clock time
materially (the NN families' fits dominate); the tmp registry itself
is per-test (``tmp_path``) so test isolation is preserved.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bristol_ml import registry
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from bristol_ml.models.nn.mlp import NnMlpModel
from bristol_ml.models.nn.temporal import NnTemporalModel
from bristol_ml.models.protocol import Model
from bristol_ml.models.sarimax import SarimaxModel
from bristol_ml.models.scipy_parametric import ScipyParametricModel
from conf._schemas import (
    LinearConfig,
    NaiveConfig,
    NnMlpConfig,
    NnTemporalConfig,
    SarimaxConfig,
    ScipyParametricConfig,
)


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """UTC-aware hourly DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _fake_metrics_df(*, mae: float = 100.0, n_folds: int = 3) -> pd.DataFrame:
    """Stage-6-harness-shaped per-fold metrics DataFrame.

    The ``mae`` argument controls the *mean* MAE across folds — the
    AC-3 / D6 test fixtures lean on this so a registry containing
    multiple runs can be ranked deterministically by lowest MAE.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n_folds):
        rows.append(
            {
                "fold_index": i,
                "train_end": pd.Timestamp("2024-01-10 00:00", tz="UTC")
                + pd.Timedelta(hours=i * 24),
                "test_start": pd.Timestamp("2024-01-10 01:00", tz="UTC")
                + pd.Timedelta(hours=i * 24),
                "test_end": pd.Timestamp("2024-01-11 00:00", tz="UTC") + pd.Timedelta(hours=i * 24),
                "mae": mae + i,
                "rmse": (mae * 1.5) + i,
                "mape": 0.03 + 0.001 * i,
                "wape": 0.028 + 0.001 * i,
            }
        )
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Per-family fitted-model factories — each returns
# ``(fitted_model, predict_features)``.
# ---------------------------------------------------------------------------


def fit_naive() -> tuple[NaiveModel, pd.DataFrame]:
    """Fit a tiny ``NaiveModel`` on a synthetic hourly series."""
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 400
    idx = _hourly_index(n)
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    model.fit(features, target)
    return model, features.iloc[200:]


def fit_linear() -> tuple[LinearModel, pd.DataFrame]:
    """Fit a tiny ``LinearModel`` on a synthetic hourly series."""
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
    return model, features.iloc[-24:]


def fit_sarimax() -> tuple[SarimaxModel, pd.DataFrame]:
    """Fit a tiny ``SarimaxModel`` (SARIMAX with weekly Fourier exog)."""
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
    return model, features.iloc[-24:]


def fit_scipy_parametric() -> tuple[ScipyParametricModel, pd.DataFrame]:
    """Fit a tiny ``ScipyParametricModel`` on a synthetic hourly series."""
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
    return model, features.iloc[-24:]


def fit_nn_mlp() -> tuple[NnMlpModel, pd.DataFrame]:
    """Fit a tiny ``NnMlpModel`` (single hidden layer, few epochs)."""
    cfg = NnMlpConfig(
        hidden_sizes=[8],
        activation="relu",
        dropout=0.0,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=16,
        max_epochs=5,
        patience=10,
        seed=None,
        device="cpu",
        target_column="nd_mw",
        feature_columns=None,
    )
    n = 60
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(n, 3)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(n)
    idx = _hourly_index(n)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)], index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model = NnMlpModel(cfg)
    model.fit(features, target, seed=17)
    return model, features.iloc[-12:]


def fit_nn_temporal() -> tuple[NnTemporalModel, pd.DataFrame]:
    """Fit a tiny ``NnTemporalModel`` (TCN, 3 epochs, 2 dilation blocks)."""
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
    n = 400
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(n, 3)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(n)
    idx = _hourly_index(n)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)], index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model = NnTemporalModel(cfg)
    model.fit(features, target, seed=17)
    # NnTemporalModel ships with a baked warmup so single-row predict
    # works through the serving boundary; the AC-3 parity guard uses a
    # post-train-tail row that matches the warmup envelope.
    return model, features.iloc[-1:]


#: Per-family factory table.  Each entry is
#: ``(family_id, factory, expected_class)``; the family_id is used as
#: the pytest parametrise id so failures name the offending family.
FAMILY_FACTORIES: list[tuple[str, Callable[[], tuple[Model, pd.DataFrame]], type[Any]]] = [
    ("naive", fit_naive, NaiveModel),
    ("linear", fit_linear, LinearModel),
    ("sarimax", fit_sarimax, SarimaxModel),
    ("scipy_parametric", fit_scipy_parametric, ScipyParametricModel),
    ("nn_mlp", fit_nn_mlp, NnMlpModel),
    ("nn_temporal", fit_nn_temporal, NnTemporalModel),
]


def register_run(
    model: Model,
    *,
    registry_dir: Path,
    mae: float = 100.0,
) -> str:
    """Save ``model`` to ``registry_dir`` and return the new ``run_id``.

    Wraps :func:`bristol_ml.registry.save` with a synthetic Stage-6-shaped
    metrics DataFrame so the test does not need to reproduce the
    harness's data layout.  ``mae`` controls the per-fold MAE so tests
    that need a deterministic ranking across multiple runs can
    distinguish them.
    """
    return registry.save(
        model,
        _fake_metrics_df(mae=mae),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=registry_dir,
    )
