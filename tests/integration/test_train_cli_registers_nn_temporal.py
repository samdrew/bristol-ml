"""Integration test — ``python -m bristol_ml.train model=nn_temporal`` round-trip — T6.

Derived from:

- ``docs/plans/active/11-complex-nn.md`` §6 Task T6
  (``test_train_cli_registers_nn_temporal_final_fold_model`` — full pipeline
  integration; parallels the Stage 10
  ``test_train_cli_registers_nn_mlp_final_fold_model``).
- ``docs/plans/active/11-complex-nn.md`` §4 AC-1 / AC-4 (Model protocol
  conformance + registry round-trip).
- ``docs/plans/active/11-complex-nn.md`` §1 D13 clause ii (the
  ``NnTemporalConfig`` isinstance branch in ``train.py``).
- ``docs/plans/active/11-complex-nn.md`` §1 NFR-1 / D11 (``device=cpu``
  pin so the CI grid runs the test deterministically without a GPU).

The test runs the train CLI end-to-end against a synthetic warm feature
cache and asserts:

1. Exit code 0 (the ``NnTemporalConfig`` dispatcher branch ran without error).
2. A single run ends up in the tmp registry dir.
3. The sidecar's ``type`` is ``"nn_temporal"`` and the metrics round-trip.

This is the Stage 11 parallel of Stage 10's
``test_train_cli_registers_nn_mlp_final_fold_model`` — one of the load-bearing
demo-moment checks for AC-1 + AC-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml import registry
from bristol_ml.features.assembler import OUTPUT_SCHEMA, WEATHER_VARIABLE_COLUMNS
from bristol_ml.train import _cli_main as _train_cli_main


def _write_feature_cache(tmp_path: Path, n_hours: int = 24 * 60) -> Path:
    """Write a minimal synthetic weather-only feature parquet.

    Mirrors the fixture shape in
    ``tests/integration/test_train_cli_registers_nn_mlp.py::_write_feature_cache``
    so the Stage 11 integration stays strictly parallel.  A 60-day
    hourly frame is short enough to fit a 2-block TCN in a handful of
    seconds on CI CPU.

    The rolling-origin overrides used in the test
    (``min_train_periods=720, test_len=168, step=168``) require at least
    ``720 + 168 = 888`` hours of data; ``n_hours=24*60=1440`` gives a
    comfortable margin.  The TCN's internal val tail eats
    ``max(seq_len + 1, fold_train // 10)`` rows; with ``seq_len=48``
    and a 720-row initial fold the val tail is ``max(49, 72) = 72`` rows,
    leaving 648 training rows with ``648 - 48 = 600`` usable windows.
    """
    rng = np.random.default_rng(42)
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
    out_path = tmp_path / "weather_only.parquet"
    pq.write_table(table, out_path)
    return out_path


# ===========================================================================
# Stage 11 T6 — train CLI end-to-end with model=nn_temporal
# ===========================================================================


def test_train_cli_registers_nn_temporal_final_fold_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: ``model=nn_temporal`` on a warm cache registers one run (AC-1 + AC-4 + T6).

    Pipeline exercised:

    1. Hydra resolves ``model=nn_temporal`` → ``NnTemporalConfig`` +
       ``NnTemporalModel``.
    2. ``train.py``'s ``isinstance(model_cfg, NnTemporalConfig)`` branch picks
       ``primary = NnTemporalModel(model_cfg); primary_kind = "nn_temporal"``.
    3. The harness runs the TCN across the rolling-origin folds, keeping
       the final-fold fitted model (Stage 9 D17 contract).
    4. ``registry.save`` stores the TCN artefact + sidecar through
       ``Model.save`` (Stage 9 D9 / Stage 11 D13 clause i).
    5. ``list_runs`` finds the new run, and the sidecar's ``type`` is
       ``"nn_temporal"``.

    Architecture is deliberately tiny (``num_blocks=2``, ``channels=8``,
    ``seq_len=48``, ``max_epochs=3``, ``batch_size=64``, ``device=cpu``)
    so the three rolling-origin folds complete within a few seconds on
    CI CPU.  This is the CLI integration smoke-test — unit-level
    fit/predict coverage lives at T4.

    Plan clause: T6 §Task T6 — structural parallel of
    ``test_train_cli_registers_nn_mlp_final_fold_model``.
    """
    _write_feature_cache(tmp_path)
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    registry_dir = tmp_path / "registry"

    exit_code = _train_cli_main(
        [
            "model=nn_temporal",
            # Minimal, CPU-pinned architecture so CI fits in a few seconds.
            # 2 blocks of 8 channels, kernel=3 — receptive field = 13 steps.
            # seq_len=48 is above the min receptive field and short enough
            # to produce many windows on the 720-row training fold.
            "model.num_blocks=2",
            "model.channels=8",
            "model.kernel_size=3",
            "model.seq_len=48",
            "model.max_epochs=3",
            "model.batch_size=64",
            "model.device=cpu",
            # Rolling-origin config: one fold of width 168, so the final-fold
            # fitted model exists and gets registered (Stage 9 D17 gate).
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
            "--registry-dir",
            str(registry_dir),
        ]
    )
    assert exit_code == 0, (
        f"_cli_main must exit 0 when model=nn_temporal is selected; got {exit_code} "
        f"(plan T6 / AC-1). stderr+stdout:\n{capsys.readouterr().out}"
    )
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=nn_temporal" in out, (
        f"stdout must contain 'Per-fold metrics for model=nn_temporal' confirming "
        f"the inline NnTemporalConfig branch was taken; stdout was:\n{out}"
    )
    assert "Registered run_id:" in out, (
        "Stage 9 D17 contract — the train CLI must print the registered "
        "run_id; missing means registry.save was skipped."
    )

    runs = registry.list_runs(registry_dir=registry_dir)
    assert len(runs) == 1, (
        f"expected exactly one registered run after model=nn_temporal; got {len(runs)}"
    )
    sole = runs[0]
    assert sole["type"] == "nn_temporal", (
        f"sidecar 'type' must be 'nn_temporal' (plan D13 clause i); got {sole['type']!r}"
    )
    assert sole["feature_set"] == "weather_only"
    assert sole["target"] == "nd_mw"
    for metric in ("mae", "mape", "rmse", "wape"):
        assert metric in sole["metrics"], (
            f"sidecar metrics must carry {metric!r}; got keys {sorted(sole['metrics'].keys())!r}"
        )
