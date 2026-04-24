"""Integration test — ``python -m bristol_ml.train model=nn_mlp`` round-trip — T5.

Derived from:

- ``docs/plans/active/10-simple-nn.md`` §6 Task T5
  (``test_train_cli_registers_nn_mlp_final_fold_model`` — full pipeline
  integration; parallels the Stage 9
  ``test_train_cli_registers_final_fold_model``).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-1 / AC-4 (Model protocol
  conformance + registry round-trip).
- ``docs/plans/active/10-simple-nn.md`` §1 D2 clause iii (the ``NnMlpConfig``
  isinstance branch in ``train.py``).
- ``docs/plans/active/10-simple-nn.md`` §1 D11 (``device=cpu`` pin so the
  CI grid runs the test deterministically without a GPU).

The test runs the train CLI end-to-end against a synthetic warm feature
cache and asserts:

1. Exit code 0 (the ``NnMlpConfig`` dispatcher branch ran without error).
2. A single run ends up in the tmp registry dir.
3. The sidecar's ``type`` is ``"nn_mlp"`` and the metrics round-trip.

This is the Stage 10 parallel of Stage 9's
``test_train_cli_registers_final_fold_model`` — one of the load-bearing
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
    ``tests/unit/registry/test_registry_cli.py::test_train_cli_registers_final_fold_model``
    so the Stage 10 integration stays strictly parallel.  A 60-day
    hourly frame is short enough to fit a 1-hidden-layer MLP in a
    handful of seconds on CI CPU.
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
# Stage 10 T5 — train CLI end-to-end with model=nn_mlp
# ===========================================================================


def test_train_cli_registers_nn_mlp_final_fold_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: ``model=nn_mlp`` on a warm cache registers one run (AC-1 + AC-4 + T5).

    Pipeline exercised:

    1. Hydra resolves ``model=nn_mlp`` → ``NnMlpConfig`` + ``NnMlpModel``.
    2. ``train.py``'s ``isinstance(model_cfg, NnMlpConfig)`` branch picks
       ``primary = NnMlpModel(model_cfg); primary_kind = "nn_mlp"``.
    3. The harness runs the NN across the rolling-origin folds, keeping
       the final-fold fitted model (Stage 9 D17 contract).
    4. ``registry.save`` stores the NN artefact + sidecar through
       ``Model.save`` (Stage 9 D9 / Stage 10 D5).
    5. ``list_runs`` finds the new run, and the sidecar's ``type`` is
       ``"nn_mlp"``.

    Architecture is deliberately tiny (``hidden_sizes=[4]``,
    ``max_epochs=3``, ``batch_size=64``, ``device=cpu``) so the three
    rolling-origin folds complete within a few seconds on CI CPU.  This
    is the CLI integration smoke-test — unit-level fit/predict coverage
    lives at T2.

    Plan clause: T5 §Task T5 — structural parallel of
    ``test_train_cli_registers_final_fold_model``.
    """
    _write_feature_cache(tmp_path)
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    registry_dir = tmp_path / "registry"

    exit_code = _train_cli_main(
        [
            "model=nn_mlp",
            # Minimal, CPU-pinned architecture so CI fits in a few seconds.
            # Hidden layer of 4 units + 3 epochs is enough to exercise the
            # dispatch + registry round-trip without a training-loop budget
            # test — that belongs to T2 / NFR-1 (CPU bit-identity).
            "model.hidden_sizes=[4]",
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
        f"_cli_main must exit 0 when model=nn_mlp is selected; got {exit_code} "
        f"(plan T5 / AC-1). stderr+stdout:\n{capsys.readouterr().out}"
    )
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=nn_mlp" in out, (
        f"stdout must contain 'Per-fold metrics for model=nn_mlp' confirming "
        f"the inline NnMlpConfig branch was taken; stdout was:\n{out}"
    )
    assert "Registered run_id:" in out, (
        "Stage 9 D17 contract — the train CLI must print the registered "
        "run_id; missing means registry.save was skipped."
    )

    runs = registry.list_runs(registry_dir=registry_dir)
    assert len(runs) == 1, (
        f"expected exactly one registered run after model=nn_mlp; got {len(runs)}"
    )
    sole = runs[0]
    assert sole["type"] == "nn_mlp", (
        f"sidecar 'type' must be 'nn_mlp' (plan D2 clause v); got {sole['type']!r}"
    )
    assert sole["feature_set"] == "weather_only"
    assert sole["target"] == "nd_mw"
    for metric in ("mae", "mape", "rmse", "wape"):
        assert metric in sole["metrics"], (
            f"sidecar metrics must carry {metric!r}; got keys {sorted(sole['metrics'].keys())!r}"
        )
