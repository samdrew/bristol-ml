"""Acceptance tests for the Stage 4 ``python -m bristol_ml.train`` CLI.

Plan: ``docs/plans/active/04-linear-baseline.md`` §6 Task T8.

The CLI wires the harness, metrics, benchmark, and model registry into
a single entry point.  The tests here exercise it primarily in-process
via :func:`bristol_ml.train._cli_main` (fast; lets the test capture
stdout with :class:`pytest.capsys`), plus one subprocess smoke that
confirms the module-level ``__main__`` wiring works under
``python -m bristol_ml.train``.

Every test writes a synthetic feature-table parquet to a ``tmp_path``
location and drives the CLI's Hydra overrides so the run is
hermetic — no reliance on ``data/features/`` on the host machine.

Conventions
-----------
- British English in docstrings.
- ``np.random.default_rng(seed=42)`` for reproducibility.
- ``pytest.MonkeyPatch`` sets ``BRISTOL_ML_CACHE_DIR`` because multiple
  config groups resolve cache paths from it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.features.assembler import OUTPUT_SCHEMA, WEATHER_VARIABLE_COLUMNS
from bristol_ml.train import _cli_main

# ---------------------------------------------------------------------------
# Shared synthetic feature-table fixture
# ---------------------------------------------------------------------------


def _write_feature_cache(path: Path, n_hours: int = 24 * 90, seed: int = 42) -> Path:
    """Write a minimal feature-table parquet matching ``assembler.OUTPUT_SCHEMA``.

    The table is synthetic but schema-conformant: 90 days of hourly UTC
    timestamps, demand with a daily sine + noise, Gaussian weather.
    """
    rng = np.random.default_rng(seed)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def warm_feature_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate a warm feature-table cache and point Hydra at it.

    ``BRISTOL_ML_CACHE_DIR`` drives the ``${oc.env:...}`` interpolation in
    every cache-owning YAML, so pointing it at ``tmp_path`` gives the CLI
    a writable sandbox.  The feature-table parquet lands at
    ``tmp_path/weather_only.parquet`` (matching the default
    ``cache_filename``).
    """
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    _write_feature_cache(tmp_path / "weather_only.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# In-process acceptance tests (plan T8 acceptance set)
# ---------------------------------------------------------------------------


def test_train_cli_prints_metric_table(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T8 acceptance: the CLI prints a metric table on stdout."""
    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    # The per-fold table header comes from _cli_main; metric-column names
    # from METRIC_REGISTRY.
    assert "Per-fold metrics for model=linear" in out
    for metric in ("mae", "mape", "rmse", "wape"):
        assert metric in out
    # At least one fold row present — the per-fold DataFrame always has
    # a ``fold_index`` column.
    assert "fold_index" in out


def test_train_cli_model_swap(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T8 acceptance: ``model=naive`` override produces naive-specific output."""
    exit_code = _cli_main(
        [
            "model=naive",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=naive" in out


def test_train_cli_exits_2_when_feature_cache_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing feature-table cache produces exit code 2 with a named-path error."""
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    # Deliberately do NOT call _write_feature_cache — the cache is absent.

    exit_code = _cli_main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Feature-table cache missing" in err


def test_train_cli_skips_benchmark_when_forecast_cache_missing(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No forecast cache → harness table printed, benchmark table omitted.

    This is the expected behaviour during a fresh-clone live demo: the
    assembler cache is warm but the slow NESO forecast ingest has not
    been run yet.  The CLI must still exit 0 and print the per-fold
    table; the benchmark section is elided.
    """
    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics" in out
    assert "Benchmark comparison" not in out


def test_train_cli_runs_benchmark_when_forecast_cache_warm(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Warm forecast cache → three-way benchmark table printed after the harness output."""
    # Synthesise a minimal NESO forecast parquet at the default cache
    # path — ``${oc.env:BRISTOL_ML_CACHE_DIR}/neso_forecast.parquet``.
    from bristol_ml.ingestion.neso_forecast import OUTPUT_SCHEMA as NESO_SCHEMA

    n_hours = 24 * 90
    hh = pd.date_range("2023-10-01", periods=2 * n_hours, freq="30min", tz="UTC")
    base = np.full(2 * n_hours, 30_000.0)
    rng = np.random.default_rng(7)
    local = hh.tz_convert("Europe/London")
    retrieved = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_utc": hh,
            "timestamp_local": local,
            "settlement_date": local.normalize().date,
            "settlement_period": np.tile(np.arange(1, 49, dtype="int8"), n_hours // 24),
            "demand_forecast_mw": (base + rng.normal(0, 200, 2 * n_hours)).astype("int32"),
            "demand_outturn_mw": (base + rng.normal(0, 50, 2 * n_hours)).astype("int32"),
            "ape_percent": rng.normal(0.0, 0.5, 2 * n_hours).astype("float32"),
            "retrieved_at_utc": [retrieved] * (2 * n_hours),
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(NESO_SCHEMA, safe=True)
    pq.write_table(table, tmp_path / "neso_forecast.parquet")

    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics" in out
    assert "Benchmark comparison" in out
    # Three-way row identifiers should appear in the benchmark section.
    for label in ("naive", "linear", "neso"):
        assert label in out


# ---------------------------------------------------------------------------
# Subprocess smoke — confirms ``python -m bristol_ml.train`` wiring works
# ---------------------------------------------------------------------------


def test_train_cli_help_exits_zero_via_subprocess() -> None:
    """``python -m bristol_ml.train --help`` exits 0 with usage text on stdout.

    This covers the ``__main__`` wiring in ``train.py`` — the in-process
    tests above call ``_cli_main`` directly and do not exercise the
    ``if __name__ == "__main__"`` branch.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Train and evaluate" in result.stdout
