"""Spec-derived tests for the Stage 9 registry CLI + harness extension.

Every test is derived from:

- ``docs/plans/active/09-model-registry.md`` §6 Task T5 named test list.
- ``docs/plans/active/09-model-registry.md`` §1 D16 (CLI entry —
  ``python -m bristol_ml.registry list|describe``).
- ``docs/plans/active/09-model-registry.md`` §1 D17 (train-CLI wiring:
  final-fold fitted model saved via ``registry.save``, no re-fit).
- ``docs/plans/active/09-model-registry.md`` §4 AC-1 (the CLI surface is
  the "list" / "describe" pair that keeps the public-verb cap at four).

No production code is modified here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from bristol_ml import registry
from bristol_ml.evaluation.harness import evaluate_and_keep_final_model
from bristol_ml.models.naive import NaiveModel
from bristol_ml.registry.__main__ import _cli_main as _registry_cli_main
from bristol_ml.registry._fs import _atomic_write_run
from bristol_ml.registry._schema import SidecarFields
from conf._schemas import NaiveConfig, SplitterConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_sidecar(
    run_id: str,
    *,
    model_type: str = "naive",
    feature_set: str = "weather_only",
    target: str = "nd_mw",
    mae_mean: float = 100.0,
) -> SidecarFields:
    """Minimal well-typed sidecar for CLI round-trip tests."""
    return SidecarFields(
        run_id=run_id,
        name="demo",
        type=model_type,
        feature_set=feature_set,
        target=target,
        feature_columns=[],
        fit_utc="2026-04-23T14:30:17+00:00",
        git_sha=None,
        hyperparameters={},
        metrics={
            "mae": {"mean": mae_mean, "std": 1.0, "per_fold": [mae_mean]},
            "rmse": {"mean": mae_mean * 1.5, "std": 1.0, "per_fold": [mae_mean * 1.5]},
        },
        registered_at_utc="2026-04-23T14:30:18+00:00",
    )


def _write_run(registry_dir: Path, sidecar: SidecarFields) -> None:
    _atomic_write_run(
        registry_dir,
        sidecar["run_id"],
        artefact_writer=lambda p: p.write_bytes(b""),
        sidecar=sidecar,
    )


# ---------------------------------------------------------------------------
# Registry CLI — plan T5 named tests
# ---------------------------------------------------------------------------


def test_registry_cli_list_prints_leaderboard_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m bristol_ml.registry list`` prints a table with run_ids and metrics."""
    _write_run(tmp_path, _make_sidecar("best_20260423T1430", mae_mean=100.0))
    _write_run(tmp_path, _make_sidecar("worst_20260423T1430", mae_mean=300.0))

    exit_code = _registry_cli_main(["list", "--registry-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    # Table header + both runs are visible; best is above worst (ascending sort).
    assert "run_id" in out
    assert "mae" in out
    assert "best_20260423T1430" in out
    assert "worst_20260423T1430" in out
    best_idx = out.index("best_20260423T1430")
    worst_idx = out.index("worst_20260423T1430")
    assert best_idx < worst_idx, "default sort is MAE-ascending — best above worst"


def test_registry_cli_list_empty_registry_prints_placeholder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``list`` on an empty directory prints a placeholder and exits 0."""
    exit_code = _registry_cli_main(["list", "--registry-dir", str(tmp_path)])
    assert exit_code == 0
    assert "no registered runs" in capsys.readouterr().out


def test_registry_cli_list_filter_by_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``--target`` flag restricts the leaderboard to matching runs (D7)."""
    _write_run(tmp_path, _make_sidecar("demand_20260423T1430", target="nd_mw"))
    _write_run(tmp_path, _make_sidecar("price_20260423T1430", target="day_ahead_price"))

    _registry_cli_main(["list", "--registry-dir", str(tmp_path), "--target", "day_ahead_price"])
    out = capsys.readouterr().out
    assert "price_20260423T1430" in out
    assert "demand_20260423T1430" not in out


def test_registry_cli_describe_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``describe`` pretty-prints one sidecar as valid JSON to stdout (plan T5)."""
    _write_run(tmp_path, _make_sidecar("solo_20260423T1430", mae_mean=42.0))

    exit_code = _registry_cli_main(
        ["describe", "solo_20260423T1430", "--registry-dir", str(tmp_path)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["run_id"] == "solo_20260423T1430"
    assert parsed["metrics"]["mae"]["mean"] == pytest.approx(42.0)


def test_registry_cli_describe_missing_run_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``describe`` exits 2 and prints the missing-run message to stderr."""
    exit_code = _registry_cli_main(
        ["describe", "does_not_exist_20260101T0000", "--registry-dir", str(tmp_path)]
    )
    assert exit_code == 2
    assert "No registered run" in capsys.readouterr().err


def test_registry_cli_help_exits_zero_via_subprocess() -> None:
    """``python -m bristol_ml.registry --help`` exits 0 with usage text."""
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.registry", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "list" in result.stdout
    assert "describe" in result.stdout


# ---------------------------------------------------------------------------
# Harness extension — evaluate_and_keep_final_model (plan T5 named test)
# ---------------------------------------------------------------------------


def test_harness_evaluate_and_keep_final_model_returns_tuple() -> None:
    """Plan T5: ``evaluate_and_keep_final_model`` returns ``(metrics_df, fitted_model)``.

    A naive model on a 500-row hourly fixture over a minimal splitter
    produces at least one fold; the returned model must be the same
    instance that was passed in, and its ``metadata.fit_utc`` must be
    populated (proof of final-fold fit).
    """
    from bristol_ml.evaluation.metrics import mae

    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "nd_mw": [float(i) for i in range(n)],
            "t2m": [float(i) * 0.1 for i in range(n)],
        },
        index=idx,
    )
    model = NaiveModel(NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw"))
    splitter_cfg = SplitterConfig(min_train_periods=200, test_len=48, step=48, gap=0)

    metrics_df, fitted = evaluate_and_keep_final_model(
        model,
        df,
        splitter_cfg,
        [mae],
        target_column="nd_mw",
        feature_columns=("t2m",),
    )
    assert fitted is model, "final-fold fitted instance must be the same object passed in"
    assert fitted.metadata.fit_utc is not None, "final-fold fit must populate fit_utc"
    # Metrics frame is the same shape ``evaluate`` returns.
    assert {"fold_index", "train_end", "test_start", "test_end", "mae"}.issubset(metrics_df.columns)


# ---------------------------------------------------------------------------
# Train-CLI wiring — plan T5 / D17 named test
# ---------------------------------------------------------------------------


def test_train_cli_registers_final_fold_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running ``python -m bristol_ml.train model=naive`` leaves one new run in the registry (D17).

    End-to-end on the synthetic feature fixture:

    1. CLI runs, prints the metric table, calls ``registry.save`` on the
       final-fold fitted model.
    2. A single run_id appears in the tmp registry dir.
    3. The sidecar's ``type`` is ``"naive"``; ``feature_set`` is
       ``"weather_only"``; ``target`` is ``"nd_mw"``; the metrics round-trip.
    """
    # Lazy import so the registry test module does not pull pyarrow at
    # collection time when the train-cli test module is absent.
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bristol_ml.features.assembler import OUTPUT_SCHEMA, WEATHER_VARIABLE_COLUMNS
    from bristol_ml.train import _cli_main as _train_cli_main

    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    registry_dir = tmp_path / "registry"

    # Minimal feature-table parquet (mirrors the fixture in tests/unit/test_train_cli.py).
    n_hours = 24 * 90
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
    pq.write_table(table, tmp_path / "weather_only.parquet")

    exit_code = _train_cli_main(
        [
            "model=naive",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
            "--registry-dir",
            str(registry_dir),
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Registered run_id:" in out

    runs = registry.list_runs(registry_dir=registry_dir)
    assert len(runs) == 1, f"expected exactly one registered run; got {len(runs)}"
    sole = runs[0]
    assert sole["type"] == "naive"
    assert sole["feature_set"] == "weather_only"
    assert sole["target"] == "nd_mw"
    # Metrics round-trip via the per-fold summary.
    for metric in ("mae", "mape", "rmse", "wape"):
        assert metric in sole["metrics"]


def test_train_cli_no_register_flag_skips_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--no-register`` leaves the registry dir empty and omits the log line."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bristol_ml.features.assembler import OUTPUT_SCHEMA, WEATHER_VARIABLE_COLUMNS
    from bristol_ml.train import _cli_main as _train_cli_main

    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    registry_dir = tmp_path / "registry"

    n_hours = 24 * 60
    rng = np.random.default_rng(0)
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
    pq.write_table(table, tmp_path / "weather_only.parquet")

    exit_code = _train_cli_main(
        [
            "model=naive",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
            "--registry-dir",
            str(registry_dir),
            "--no-register",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Registered run_id:" not in out
    # registry_dir was never created because nothing was saved.
    assert not registry_dir.exists() or not list(registry_dir.iterdir())
