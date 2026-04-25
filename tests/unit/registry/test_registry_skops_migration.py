"""Stage 12 T5 — registry filesystem layout migration to ``.skops``.

Three named tests guard the Stage 12 D10 (Ctrl+G reversal) migration of
the registry's canonical artefact filename from ``model.joblib`` to
``model.skops``:

- :func:`test_registry_save_writes_skops_artefact_only` — after
  ``registry.save`` the run directory contains ``artefact/model.skops``
  and *not* ``artefact/model.joblib``.  The atomic-write contract from
  Stage 9 D5 carries over unchanged across the serialiser flip; the
  rename invariant is preserved.
- :func:`test_load_rejects_joblib_artefact_in_registry` — a
  pre-Stage-12 run directory carrying a ``model.joblib`` artefact is
  rejected by ``registry.load`` with a clear ``RuntimeError`` whose
  message points the operator at the retraining migration path.
  The error message is part of the contract: Stage 12 D10 demands a
  *visible* failure, not a silent fallthrough into joblib's
  unrestricted unpickler.
- :func:`test_registry_roundtrip_per_family` — parametrised over all
  six model families (``naive``, ``linear``, ``sarimax``,
  ``scipy_parametric``, ``nn_mlp``, ``nn_temporal``).  Each family
  fits a small fixture, saves through the registry, loads it back,
  and asserts the round-trip ``predict`` agrees on a held-out window.
  Locks down the cross-family migration as a single grep-able guard.

Plan clause: ``docs/plans/active/12-serving.md`` §Task T5 named-test
list / D10 (Ctrl+G reversal).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bristol_ml import registry
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from bristol_ml.models.nn.mlp import NnMlpModel
from bristol_ml.models.nn.temporal import NnTemporalModel
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

# ---------------------------------------------------------------------------
# Shared fixture helpers — kept local so this file is self-contained.
# ---------------------------------------------------------------------------


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _fake_metrics_df(n_folds: int = 3) -> pd.DataFrame:
    """Stage-6-harness-shaped per-fold metrics DataFrame."""
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
                "mae": 100.0 + i,
                "rmse": 150.0 + i,
                "mape": 0.03 + 0.001 * i,
                "wape": 0.028 + 0.001 * i,
            }
        )
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Per-family fitted-model factories — each returns
# ``(fitted_model, predict_features)``.
# ---------------------------------------------------------------------------


def _fit_naive() -> tuple[NaiveModel, pd.DataFrame]:
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 400
    idx = _hourly_index(n)
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    model.fit(features, target)
    return model, features.iloc[200:]


def _fit_linear() -> tuple[LinearModel, pd.DataFrame]:
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


def _fit_sarimax() -> tuple[SarimaxModel, pd.DataFrame]:
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


def _fit_scipy_parametric() -> tuple[ScipyParametricModel, pd.DataFrame]:
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


def _fit_nn_mlp() -> tuple[NnMlpModel, pd.DataFrame]:
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


def _fit_nn_temporal() -> tuple[NnTemporalModel, pd.DataFrame]:
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
    # works through the serving boundary; the round-trip guard uses a
    # post-train-tail window that matches the existing dispatch tests.
    return model, features.iloc[-32:]


# ---------------------------------------------------------------------------
# 1. test_registry_save_writes_skops_artefact_only
# ---------------------------------------------------------------------------


def test_registry_save_writes_skops_artefact_only(tmp_path: Path) -> None:
    """Stage 12 T5 named test (D10) — registry save writes ``model.skops``, not ``model.joblib``.

    After ``registry.save`` the run directory under ``registry_dir``
    must contain exactly one artefact file at
    ``artefact/model.skops``.  No ``model.joblib`` may exist alongside
    it; no staging ``.tmp_*`` directories may remain.  The atomic
    single-file save contract (Stage 9 D5) is preserved across the
    Stage 12 serialiser flip.

    Plan clause: Stage 12 plan §Task T5 named-test list / D10 /
    Stage 9 D5.
    """
    model, _predict_features = _fit_naive()
    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    run_dir = tmp_path / run_id
    artefact_dir = run_dir / "artefact"

    siblings = sorted(p.name for p in artefact_dir.iterdir())
    assert siblings == ["model.skops"], (
        "registry.save must write exactly one artefact file at "
        f"artefact/model.skops; got siblings {siblings!r} (Stage 12 T5 / D10)."
    )
    assert (artefact_dir / "model.skops").is_file()
    assert not (artefact_dir / "model.joblib").exists(), (
        "no .joblib artefact may remain alongside .skops — Stage 12 D10 "
        "disabled joblib at the registry boundary for security."
    )
    # Atomic-write invariant: no leftover staging directories.
    assert not list(tmp_path.glob(".tmp_*")), (
        "registry.save must rename staging directories away atomically "
        "(Stage 9 D5 atomic-write contract carries over across the "
        "Stage 12 serialiser flip)."
    )


# ---------------------------------------------------------------------------
# 2. test_load_rejects_joblib_artefact_in_registry
# ---------------------------------------------------------------------------


def test_load_rejects_joblib_artefact_in_registry(tmp_path: Path) -> None:
    """Stage 12 T5 named test (D10) — pre-Stage-12 joblib artefact is rejected with a clear error.

    Hand-build a registry run directory that mimics the pre-Stage-12
    layout — ``run.json`` sidecar plus ``artefact/model.joblib``,
    *no* ``model.skops`` — and assert that ``registry.load`` raises a
    :class:`RuntimeError` whose message:

    1. names the offending ``model.joblib`` path so the operator can
       locate it on disk;
    2. mentions Stage 12 D10 / the security rationale ("joblib loads
       are disabled at the registry boundary");
    3. points the operator at the retraining migration path.

    A silent fallthrough into joblib's unrestricted unpickler would
    re-introduce the exact RCE vector D10 exists to close.

    Plan clause: Stage 12 plan §Task T5 named-test list / D10 / the
    `registry.load` ``RuntimeError`` docstring.
    """
    # Build a pre-Stage-12-shaped run directory: sidecar present,
    # artefact is the legacy joblib file, no model.skops in sight.
    run_id = "linear-ols-weather-only_20260423T1430"
    run_dir = tmp_path / run_id
    artefact_dir = run_dir / "artefact"
    artefact_dir.mkdir(parents=True)
    (artefact_dir / "model.joblib").write_bytes(b"legacy-joblib-artefact-bytes")
    (run_dir / "run.json").write_text(
        '{"run_id": "linear-ols-weather-only_20260423T1430", '
        '"name": "linear-ols-weather-only", "type": "linear", '
        '"feature_set": "weather_only", "target": "nd_mw", '
        '"feature_columns": [], "fit_utc": "2026-04-23T14:30:00+00:00", '
        '"git_sha": null, "hyperparameters": {}, "metrics": {}, '
        '"registered_at_utc": "2026-04-23T14:30:18+00:00"}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc_info:
        registry.load(run_id, registry_dir=tmp_path)

    msg = str(exc_info.value)
    assert "model.joblib" in msg, (
        f"RuntimeError message must name the offending model.joblib "
        f"path; got {msg!r} (Stage 12 T5 / D10)."
    )
    assert "joblib" in msg.lower() and "skops" in msg.lower(), (
        f"RuntimeError message must reference both joblib (the rejected "
        f"format) and skops (the expected migration target); "
        f"got {msg!r} (Stage 12 T5 / D10)."
    )
    assert "retrain" in msg.lower(), (
        f"RuntimeError message must point the operator at the retraining "
        f"migration path; got {msg!r} (Stage 12 T5 / D10)."
    )


# ---------------------------------------------------------------------------
# 3. test_registry_roundtrip_per_family — parametrised over all six families
# ---------------------------------------------------------------------------


_FAMILY_FACTORIES: list[tuple[str, Callable[[], tuple[Any, pd.DataFrame]], type[Any]]] = [
    ("naive", _fit_naive, NaiveModel),
    ("linear", _fit_linear, LinearModel),
    ("sarimax", _fit_sarimax, SarimaxModel),
    ("scipy_parametric", _fit_scipy_parametric, ScipyParametricModel),
    ("nn_mlp", _fit_nn_mlp, NnMlpModel),
    ("nn_temporal", _fit_nn_temporal, NnTemporalModel),
]


@pytest.mark.parametrize(
    ("family", "factory", "expected_class"),
    _FAMILY_FACTORIES,
    ids=[name for name, _factory, _cls in _FAMILY_FACTORIES],
)
def test_registry_roundtrip_per_family(
    tmp_path: Path,
    family: str,
    factory: Callable[[], tuple[Any, pd.DataFrame]],
    expected_class: type[Any],
) -> None:
    """Stage 12 T5 named test (D10) — every model family round-trips through the .skops registry.

    Six-family parametrised guard for the Stage 12 D10 migration:
    fit a small per-family fixture, ``registry.save`` it, then
    ``registry.load`` and assert (a) the loaded model is the expected
    concrete class and (b) ``predict`` on a held-out window matches
    the pre-save prediction under ``rtol=0, atol=1e-8`` (the per-family
    inner-blob round-trips bit-exactly inside its skops envelope; a
    tiny tolerance accommodates float-formatting noise that the NN
    families exhibit on rare CPU paths).

    Locking the migration down as a single grep-able test prevents a
    future model family from regressing on the registry boundary
    silently.

    Plan clause: Stage 12 plan §Task T5 named-test list ("parametrised
    over all six families end-to-end") / D10 / D9 (nn_temporal
    first-class, Ctrl+G reversal).
    """
    model, predict_features = factory()
    pred_before = model.predict(predict_features)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    # The on-disk artefact is a .skops file (not .joblib).
    artefact_dir = tmp_path / run_id / "artefact"
    assert (artefact_dir / "model.skops").is_file(), (
        f"family={family!r}: registry must write artefact/model.skops "
        "(Stage 12 D10 — registry artefact filename migrated from .joblib)."
    )

    loaded = registry.load(run_id, registry_dir=tmp_path)
    assert isinstance(loaded, expected_class), (
        f"family={family!r}: loaded artefact must be a "
        f"{expected_class.__name__} instance; got {type(loaded).__name__}."
    )

    pred_after = loaded.predict(predict_features)
    np.testing.assert_allclose(
        pred_before.to_numpy(),
        pred_after.to_numpy(),
        rtol=0.0,
        atol=1e-8,
        err_msg=(
            f"family={family!r}: predictions after registry round-trip "
            "must match pre-save predictions under rtol=0, atol=1e-8. "
            "Stage 12 T5 / D10 — the inner per-family blob must be "
            "bit-exact through the .skops envelope."
        ),
    )
    assert pred_after.index.equals(pred_before.index), (
        f"family={family!r}: predict.index must round-trip exactly "
        "(load-bearing for the Stage 12 serving boundary)."
    )
