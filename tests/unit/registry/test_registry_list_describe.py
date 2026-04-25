"""Spec-derived tests for ``bristol_ml.registry.list_runs`` + ``describe`` — Stage 9 T4.

Every test is derived from:

- ``docs/plans/active/09-model-registry.md`` §6 Task T4 named test list.
- ``docs/plans/active/09-model-registry.md`` §1 D7 (exact-match filters).
- ``docs/plans/active/09-model-registry.md`` §1 D8 (MAE-ascending default sort).
- ``docs/plans/active/09-model-registry.md`` §4 AC-4 (instantaneous listing
  of 100 runs; NFR-speed 1 s ceiling).
- ``docs/plans/active/09-model-registry.md`` §8 R5 (``.tmp_*`` staging dirs
  must be ignored by the leaderboard scanner).

No production code is modified here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bristol_ml import registry
from bristol_ml.registry._fs import _atomic_write_run
from bristol_ml.registry._schema import SidecarFields

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_sidecar(
    run_id: str,
    *,
    name: str = "demo",
    model_type: str = "naive",
    feature_set: str = "weather_only",
    target: str = "nd_mw",
    mae_mean: float = 100.0,
    rmse_mean: float = 150.0,
) -> SidecarFields:
    """Return a minimal but schema-complete sidecar for list/describe tests."""
    return SidecarFields(
        run_id=run_id,
        name=name,
        type=model_type,
        feature_set=feature_set,
        target=target,
        feature_columns=[],
        fit_utc="2026-04-23T14:30:17+00:00",
        git_sha=None,
        hyperparameters={},
        metrics={
            "mae": {"mean": mae_mean, "std": 1.0, "per_fold": [mae_mean]},
            "rmse": {"mean": rmse_mean, "std": 1.0, "per_fold": [rmse_mean]},
        },
        registered_at_utc="2026-04-23T14:30:18+00:00",
    )


def _write_run(registry_dir: Path, sidecar: SidecarFields) -> None:
    """Write a fake run with an empty artefact payload."""
    _atomic_write_run(
        registry_dir,
        sidecar["run_id"],
        artefact_writer=lambda p: p.write_bytes(b""),
        sidecar=sidecar,
    )


# ---------------------------------------------------------------------------
# AC-4 — 100-run leaderboard is instantaneous (< 1 s; plan T4 named test)
# ---------------------------------------------------------------------------


def test_registry_list_hundred_entries_is_fast(tmp_path: Path) -> None:
    """``list_runs`` over 100 runs completes in under 1 s on CI (AC-4 / NFR-speed).

    Populates a fresh registry root with 100 synthetic sidecars and asserts
    the wall-clock of a single ``list_runs()`` is under 1.0 s.  Expected
    local runtime is on the order of tens of milliseconds; the 1 s gate is
    order-of-magnitude headroom for slow CI (plan R4).
    """
    for i in range(100):
        _write_run(
            tmp_path,
            _make_sidecar(run_id=f"demo_{i:03d}_20260423T1430", mae_mean=float(i)),
        )

    start = time.monotonic()
    runs = registry.list_runs(registry_dir=tmp_path)
    elapsed = time.monotonic() - start

    assert len(runs) == 100, f"expected 100 runs; got {len(runs)}"
    assert elapsed < 1.0, (
        f"list_runs(100 runs) took {elapsed:.3f}s; NFR-speed ceiling is 1.0 s (AC-4)"
    )


# ---------------------------------------------------------------------------
# D7 — exact-match filters (plan T4 named test)
# ---------------------------------------------------------------------------


def test_registry_list_filter_by_target(tmp_path: Path) -> None:
    """``list_runs(target=...)`` returns only runs whose sidecar target matches (D7)."""
    _write_run(tmp_path, _make_sidecar("demand_run_20260423T1430", target="nd_mw"))
    _write_run(tmp_path, _make_sidecar("price_run_20260423T1430", target="day_ahead_price"))
    _write_run(tmp_path, _make_sidecar("other_run_20260423T1430", target="nd_mw"))

    runs = registry.list_runs(target="nd_mw", registry_dir=tmp_path)
    assert len(runs) == 2
    assert {r["run_id"] for r in runs} == {
        "demand_run_20260423T1430",
        "other_run_20260423T1430",
    }


def test_registry_list_filter_by_model_type(tmp_path: Path) -> None:
    """``list_runs(model_type=...)`` filters on the sidecar ``type`` field (D7)."""
    _write_run(tmp_path, _make_sidecar("linear_run_20260423T1430", model_type="linear"))
    _write_run(tmp_path, _make_sidecar("sarimax_run_20260423T1430", model_type="sarimax"))

    runs = registry.list_runs(model_type="sarimax", registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == ["sarimax_run_20260423T1430"]


def test_registry_list_filter_by_feature_set(tmp_path: Path) -> None:
    """``list_runs(feature_set=...)`` filters on the sidecar ``feature_set`` field (D7)."""
    _write_run(
        tmp_path,
        _make_sidecar("a_20260423T1430", feature_set="weather_only"),
    )
    _write_run(
        tmp_path,
        _make_sidecar("b_20260423T1430", feature_set="weather_calendar"),
    )

    runs = registry.list_runs(feature_set="weather_calendar", registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == ["b_20260423T1430"]


# ---------------------------------------------------------------------------
# D8 — MAE-ascending default sort (plan T4 named test)
# ---------------------------------------------------------------------------


def test_registry_list_default_sort_is_mae_ascending(tmp_path: Path) -> None:
    """Default sort is MAE ascending — best (lowest) MAE first (D8 / Demo moment)."""
    _write_run(tmp_path, _make_sidecar("worst_20260423T1430", mae_mean=300.0))
    _write_run(tmp_path, _make_sidecar("best_20260423T1430", mae_mean=100.0))
    _write_run(tmp_path, _make_sidecar("mid_20260423T1430", mae_mean=200.0))

    runs = registry.list_runs(registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == [
        "best_20260423T1430",
        "mid_20260423T1430",
        "worst_20260423T1430",
    ]


def test_registry_list_descending_sort(tmp_path: Path) -> None:
    """``ascending=False`` returns largest metric first."""
    _write_run(tmp_path, _make_sidecar("worst_20260423T1430", mae_mean=300.0))
    _write_run(tmp_path, _make_sidecar("best_20260423T1430", mae_mean=100.0))

    runs = registry.list_runs(ascending=False, registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == [
        "worst_20260423T1430",
        "best_20260423T1430",
    ]


def test_registry_list_sort_by_alternative_metric(tmp_path: Path) -> None:
    """``sort_by="rmse"`` orders by RMSE regardless of default direction."""
    _write_run(tmp_path, _make_sidecar("a_20260423T1430", mae_mean=100.0, rmse_mean=500.0))
    _write_run(tmp_path, _make_sidecar("b_20260423T1430", mae_mean=200.0, rmse_mean=300.0))

    runs = registry.list_runs(sort_by="rmse", registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == ["b_20260423T1430", "a_20260423T1430"]


def test_registry_list_missing_metric_sorts_last(tmp_path: Path) -> None:
    """Runs missing the ``sort_by`` metric go to the end regardless of direction."""
    # Hand-craft one sidecar that's missing "mae".
    sidecar_no_mae: SidecarFields = _make_sidecar("no_mae_20260423T1430")
    sidecar_no_mae["metrics"].pop("mae")
    _write_run(tmp_path, sidecar_no_mae)

    _write_run(tmp_path, _make_sidecar("has_mae_20260423T1430", mae_mean=100.0))

    runs = registry.list_runs(registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == [
        "has_mae_20260423T1430",
        "no_mae_20260423T1430",
    ]
    runs_desc = registry.list_runs(ascending=False, registry_dir=tmp_path)
    # Missing-last holds under descending too.
    assert runs_desc[-1]["run_id"] == "no_mae_20260423T1430"


# ---------------------------------------------------------------------------
# R5 — staging directories are ignored
# ---------------------------------------------------------------------------


def test_registry_list_ignores_staging_directories(tmp_path: Path) -> None:
    """Directories starting with ``.tmp_`` are skipped by the scanner (plan R5)."""
    _write_run(tmp_path, _make_sidecar("real_20260423T1430", mae_mean=100.0))
    # Simulate a crash during save — a staging directory left behind.
    staging = tmp_path / ".tmp_deadbeef"
    staging.mkdir()
    (staging / "run.json").write_text("{}", encoding="utf-8")

    runs = registry.list_runs(registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == ["real_20260423T1430"]


def test_registry_list_returns_empty_on_nonexistent_registry(tmp_path: Path) -> None:
    """Pointing at a missing registry root returns ``[]`` rather than raising."""
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()
    assert registry.list_runs(registry_dir=missing) == []


# ---------------------------------------------------------------------------
# Stage 12 D10 — legacy joblib runs are filtered out of the leaderboard
# ---------------------------------------------------------------------------


def test_registry_list_skips_legacy_joblib_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A legacy ``model.joblib``-only run is excluded from ``list_runs``.

    Stage 12 D10 disabled joblib loads at the registry boundary
    (:func:`registry.load` raises ``RuntimeError``).  ``list_runs`` is
    the discovery surface that feeds the Stage 11 ablation notebook and
    every harness-driven workflow; if a legacy run is reported as
    "registered" the caller's subsequent :func:`load` will die.  This
    regression guard asserts the filter:

    1. A legacy joblib-only run is **not** returned.
    2. A healthy skops sibling run **is** returned.
    3. A loguru warning is emitted naming the skipped ``run_id``.
    """
    # Healthy skops run — written through the standard atomic-write path.
    _write_run(tmp_path, _make_sidecar("skops_run_20260424T2331", mae_mean=42.0))

    # Legacy joblib run — synthesise the on-disk shape directly so we
    # do not have to reach back into pre-Stage-12 IO helpers.
    legacy_dir = tmp_path / "joblib_run_20260424T2331"
    (legacy_dir / "artefact").mkdir(parents=True)
    (legacy_dir / "artefact" / "model.joblib").write_bytes(b"")
    legacy_sidecar = _make_sidecar("joblib_run_20260424T2331", mae_mean=10.0)
    (legacy_dir / "run.json").write_text(
        json.dumps(dict(legacy_sidecar), indent=2),
        encoding="utf-8",
    )

    from loguru import logger as _loguru_logger

    sink_id = _loguru_logger.add(caplog.handler, format="{message}")
    try:
        runs = registry.list_runs(registry_dir=tmp_path)
    finally:
        _loguru_logger.remove(sink_id)

    assert [r["run_id"] for r in runs] == ["skops_run_20260424T2331"], (
        f"list_runs must filter out the legacy joblib run; saw {[r['run_id'] for r in runs]!r}."
    )
    assert any("joblib_run_20260424T2331" in record.getMessage() for record in caplog.records), (
        "list_runs must emit a warning naming the skipped legacy run."
    )


# ---------------------------------------------------------------------------
# describe — plan T4 named test
# ---------------------------------------------------------------------------


def test_registry_describe_returns_full_sidecar(tmp_path: Path) -> None:
    """``describe`` returns the full parsed sidecar for a single run (plan T4)."""
    sidecar_in = _make_sidecar("demo_20260423T1430", mae_mean=42.0)
    _write_run(tmp_path, sidecar_in)

    got = registry.describe("demo_20260423T1430", registry_dir=tmp_path)
    assert got["run_id"] == "demo_20260423T1430"
    assert got["metrics"]["mae"]["mean"] == pytest.approx(42.0)
    # Every required sidecar field survives the JSON round-trip.
    assert set(got.keys()) == set(json.loads(json.dumps(dict(sidecar_in))).keys())


def test_registry_describe_raises_on_missing_run(tmp_path: Path) -> None:
    """``describe`` raises ``FileNotFoundError`` for an unregistered ``run_id``."""
    with pytest.raises(FileNotFoundError):
        registry.describe("nope_20260423T1430", registry_dir=tmp_path)
