"""D10 round-trip test — registry → MLflow PyFunc → load → predict.

Derived from `docs/plans/active/09-model-registry.md` §6 Task T7 and §1
D10.  The test packages a registered ``NaiveModel`` through the test-only
:class:`tests.integration.mlflow_adapter.RegistryPyfuncAdapter` and
asserts that loading it back via :func:`mlflow.pyfunc.load_model`
produces numerically-identical predictions to the original model
(``numpy.allclose(..., atol=1e-10)``).

This is the *falsifier* for the plan's "mechanical migration" claim:
if a future MLflow release changes the PyFunc save/load contract in a
breaking way, this test goes red and the drift surfaces at the
version-bump boundary rather than silently in production.  If MLflow is
unavailable (e.g. lean install without ``--group dev``) the module is
skipped via :func:`pytest.importorskip` rather than failing.

NaiveModel is the cheapest fit in the Stage 4/7/8 roster; the D10 proof
is adapter-shaped, not model-family-specific.  Broader coverage of the
adapter against the other three families is a deliberate Stage 9 cut
(plan scope diff / R6 low-likelihood MLflow version drift).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip(
    "mlflow",
    reason="D10 adapter test requires the `dev` dependency group (MLflow installed).",
)
mlflow_pyfunc = pytest.importorskip("mlflow.pyfunc")

from bristol_ml import registry  # noqa: E402 — after importorskip gate
from bristol_ml.models.naive import NaiveModel  # noqa: E402
from conf._schemas import NaiveConfig  # noqa: E402
from tests.integration.mlflow_adapter import package_run_as_pyfunc  # noqa: E402


def _hourly_index(n: int) -> pd.DatetimeIndex:
    """UTC-aware hourly index helper local to this module."""
    return pd.date_range("2024-01-01 00:00", periods=n, freq="h", tz="UTC")


def _minimal_metrics_df() -> pd.DataFrame:
    """Two-fold Stage-6-harness-shaped DataFrame — fixture for save()."""
    return pd.DataFrame(
        {
            "fold_index": [0, 1],
            "train_end": [
                pd.Timestamp("2024-01-10 00:00", tz="UTC"),
                pd.Timestamp("2024-01-11 00:00", tz="UTC"),
            ],
            "test_start": [
                pd.Timestamp("2024-01-10 01:00", tz="UTC"),
                pd.Timestamp("2024-01-11 01:00", tz="UTC"),
            ],
            "test_end": [
                pd.Timestamp("2024-01-11 00:00", tz="UTC"),
                pd.Timestamp("2024-01-12 00:00", tz="UTC"),
            ],
            "mae": [100.0, 101.0],
            "rmse": [150.0, 151.0],
        }
    )


def test_registry_run_is_loadable_via_mlflow_pyfunc_adapter(tmp_path: Path) -> None:
    """Round-trip: save → package → load → predict matches the original (D10).

    Arrange
    -------
    Fit a ``NaiveModel`` on a small synthetic series and register it
    via :func:`bristol_ml.registry.save` under a temp registry dir.
    Capture a reference prediction on a held-out feature frame.

    Act
    ---
    Package the run as an MLflow PyFunc via
    :func:`tests.integration.mlflow_adapter.package_run_as_pyfunc`, then
    load it back with :func:`mlflow.pyfunc.load_model` and predict on
    the same held-out frame.

    Assert
    ------
    The MLflow PyFunc ``predict`` output matches the reference prediction
    element-wise to ``atol=1e-10``.  Any non-trivial difference
    indicates the PyFunc adapter contract has drifted — do not weaken
    the tolerance.
    """
    # Arrange — fit and register.  Redirect MLflow tracking URI into
    # ``tmp_path`` so ``mlflow.pyfunc.save_model`` / ``load_model`` do not
    # spawn a default ``./mlruns/0/`` experiment bucket in the repo root
    # (MLflow's implicit behaviour whenever tracking is unset).
    mlflow.set_tracking_uri(str(tmp_path / "mlflow_tracking"))
    registry_dir = tmp_path / "registry"
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 400
    idx = _hourly_index(n)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    model.fit(features, target)

    run_id = registry.save(
        model,
        _minimal_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=registry_dir,
    )

    # Hold-out frame — NaiveModel same_hour_last_week needs the training
    # history appended so it can look up the previous week's value.
    holdout_idx = pd.date_range("2024-01-18 00:00", periods=24, freq="h", tz="UTC")
    holdout_features = pd.DataFrame(
        {"t2m": np.arange(24, dtype=float) * 0.1},
        index=holdout_idx,
    )
    reference = np.asarray(model.predict(holdout_features))

    # Act — package as PyFunc and load back.
    pyfunc_dst = tmp_path / "mlflow_pyfunc_artefact"
    package_run_as_pyfunc(run_id, pyfunc_dst, registry_dir=registry_dir)
    loaded = mlflow.pyfunc.load_model(str(pyfunc_dst))
    via_mlflow = np.asarray(loaded.predict(holdout_features))

    # Assert — shape and numerical agreement.
    assert via_mlflow.shape == reference.shape, (
        f"MLflow PyFunc output shape drifted: reference={reference.shape}, "
        f"mlflow={via_mlflow.shape}"
    )
    assert np.allclose(via_mlflow, reference, atol=1e-10, equal_nan=True), (
        "MLflow PyFunc predictions must match the wrapped Model.predict output "
        "to atol=1e-10 — divergence signals adapter-contract drift."
    )
