"""Spec-derived tests for the Stage 11 ``nn_temporal`` registry dispatch — Task T6.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §6 Task T6 (the three T6 named
  tests: registry save via protocol, ``list_runs`` includes the new
  ``type``, and the structural-parallel isinstance-branch check).
- ``docs/plans/active/11-complex-nn.md`` §4 AC-4 (registry round-trip at
  ``atol=1e-10``).
- ``docs/plans/active/11-complex-nn.md`` §1 D13 clauses i, ii (one new
  ``isinstance`` branch in ``train.py``; one new entry in each of the
  two registry dispatch dicts).

The structural isinstance-branch test below is the Stage-11 parallel of
Stage 10's ``test_nn_mlp_is_dispatched_by_train_cli_isinstance_branch``
— recast against ``bristol_ml.train`` because ``train.py``'s dispatcher
is inlined inside ``_cli_main`` (no standalone ``_build_model_from_config``
helper).  A full end-to-end CLI exercise of the ``model=nn_temporal`` path
lives at T6 as ``test_train_cli_registers_nn_temporal_final_fold_model`` —
this file keeps its unit focus on the dispatch wiring itself.

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test;
surface the failure to the implementer.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bristol_ml import registry, train
from bristol_ml.models.nn.temporal import NnTemporalModel
from bristol_ml.registry._dispatch import _CLASS_NAME_TO_TYPE, _TYPE_TO_CLASS
from conf._schemas import NnTemporalConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers — mirror test_registry_nn_mlp_dispatch.py so the
# Stage 11 entries align with the Stage 10 suite shape (plan T6: "extends
# the Stage 10 AC-4 test suite with the sixth model family").
# ---------------------------------------------------------------------------


def _hourly_index(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """UTC-aware hourly DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="h", tz="UTC")


def _fake_metrics_df(n_folds: int = 3) -> pd.DataFrame:
    """Stage-6-harness-shaped per-fold metrics DataFrame."""
    rows = []
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


def _read_sidecar(run_dir: Path) -> dict:
    """Read and JSON-decode the ``run.json`` sidecar."""
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _cpu_temporal_config() -> NnTemporalConfig:
    """Return a CPU-pinned short-budget ``NnTemporalConfig``.

    The T6 tests all exercise the registry + dispatch layer, not the
    training loop itself — a handful of epochs on a minimal TCN is enough
    to surface save-time and load-time wiring bugs.  CPU is pinned so the
    tests are reproducible inside CI without a GPU.

    Minimum viable settings per the plan T6 fixture guidance:
    ``seq_len=32, num_blocks=2, channels=8, kernel_size=3,
    weight_norm=False, dropout=0.0, max_epochs=3, batch_size=64``.
    """
    return NnTemporalConfig(
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


def _fit_tiny_temporal(
    n: int = 400, n_features: int = 3, seed: int = 17
) -> tuple[NnTemporalModel, pd.DataFrame, pd.Series]:
    """Fit an ``NnTemporalModel`` on a small deterministic frame.

    Uses ``n=400`` rows — larger than the plan's suggested 200 to ensure
    the internal val split (``n_val = max(seq_len + 1, n // 10)``) leaves
    enough training windows.  At ``seq_len=32`` and ``n=400`` the val
    tail is ``max(33, 40) = 40`` rows, leaving 360 training rows with
    ``360 - 32 = 328`` usable sequence windows — plenty for 3 epochs.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(n)
    idx = _hourly_index(n)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model = NnTemporalModel(_cpu_temporal_config())
    model.fit(features, target, seed=seed)
    return model, features, target


# ===========================================================================
# 1. test_registry_save_nn_temporal_model_via_protocol
# ===========================================================================


def test_registry_save_nn_temporal_model_via_protocol(tmp_path: Path) -> None:
    """Guards plan T6 / AC-4 — registry save accepts ``NnTemporalModel`` via the protocol.

    The Stage 9 registry consumes ``Model.save`` via the protocol (plan
    D9 in Stage 9); adding the sixth model family means (a) registering
    the ``NnTemporalModel`` class in ``_dispatch.py`` and (b) routing
    through ``registry.save`` with no model-class changes.  A failure
    here points at a missing key in ``_CLASS_NAME_TO_TYPE`` or a
    missing branch in ``train.py``.

    Plan clause: T6 §Task T6 / AC-4.
    """
    model, _features, _target = _fit_tiny_temporal()

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    run_dir = tmp_path / run_id
    assert run_dir.is_dir(), f"expected registry run directory {run_dir} to exist"
    assert (run_dir / "artefact" / "model.joblib").is_file(), (
        "Stage 11 D13 mandates the single joblib envelope at artefact/model.joblib; "
        "missing file indicates NnTemporalModel.save did not route through Model.save."
    )
    assert (run_dir / "run.json").is_file(), "expected run.json sidecar to be written"

    sidecar = _read_sidecar(run_dir)
    assert sidecar["type"] == "nn_temporal", (
        f"sidecar 'type' must be 'nn_temporal' for the TCN family (D13 clause i); "
        f"got {sidecar['type']!r}"
    )
    assert sidecar["name"] == model.metadata.name
    assert sidecar["feature_set"] == "weather_only"
    assert sidecar["target"] == "nd_mw"
    # Metrics summary is populated from the per-fold DataFrame.
    assert set(sidecar["metrics"].keys()) == {"mae", "rmse", "mape", "wape"}


# ===========================================================================
# 2. test_registry_load_round_trips_nn_temporal_model  (AC-4 round-trip)
# ===========================================================================


def test_registry_load_round_trips_nn_temporal_model(tmp_path: Path) -> None:
    """Guards AC-4 — save + load + predict agrees to ``atol=1e-10``.

    The stronger AC-4 check: after a full registry round-trip the
    loaded model's ``predict()`` output matches the original to
    ``atol=1e-10``.  This is the same bar the five Stage-4/7/8/10 models
    are held to in ``test_registry_save_load.py`` and
    ``test_registry_nn_mlp_dispatch.py`` — the sixth family joins
    the same suite.

    Plan clause: T6 §Task T6 / AC-4 / NFR-3.
    """
    model, features, _target = _fit_tiny_temporal()
    predicted_before = model.predict(features)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, NnTemporalModel), (
        f"registry.load must return an NnTemporalModel for a 'nn_temporal' run; "
        f"got {type(loaded).__name__!r} — check _TYPE_TO_CLASS."
    )
    predicted_after = loaded.predict(features)

    np.testing.assert_allclose(
        predicted_after.to_numpy(),
        predicted_before.to_numpy(),
        atol=1e-10,
        err_msg=(
            "registry.save + registry.load broke predict() numerical equivalence; "
            "the NnTemporalModel artefact or registry dispatcher is dropping state "
            "(plan AC-4 / NFR-3)."
        ),
    )
    # Index and name round-trip — inherits the Stage 4 Model protocol invariants.
    assert predicted_after.index.equals(predicted_before.index)
    assert predicted_after.name == predicted_before.name


# ===========================================================================
# 3. test_registry_list_runs_includes_nn_temporal_type
# ===========================================================================


def test_registry_list_runs_includes_nn_temporal_type(tmp_path: Path) -> None:
    """Guards plan T6 — ``list_runs(model_type='nn_temporal')`` returns the TCN run.

    The Stage 9 ``list_runs`` filter is exact-match on the sidecar's
    ``type`` field (plan D7 in Stage 9).  For the Stage 11 leaderboard
    demo to surface the new family, a saved ``NnTemporalModel`` must be
    reachable via ``list_runs(model_type='nn_temporal')``.

    Plan clause: T6 §Task T6.
    """
    model, _features, _target = _fit_tiny_temporal()

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    runs = registry.list_runs(model_type="nn_temporal", registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == [run_id], (
        f"list_runs(model_type='nn_temporal') must return exactly the one nn_temporal run; "
        f"got {[r['run_id'] for r in runs]!r}"
    )
    assert runs[0]["type"] == "nn_temporal"


# ===========================================================================
# 4. test_nn_temporal_is_dispatched_by_train_cli_isinstance_branch  (structural)
# ===========================================================================


def test_nn_temporal_is_dispatched_by_train_cli_isinstance_branch() -> None:
    """Guards plan T6 / D13 clause ii — ``train.py`` carries the TCN branch.

    Structural parallel of the Stage 10 T4 dispatch test.  ``train.py``
    does not expose a standalone ``_build_model_from_config`` helper —
    the dispatch is inlined inside ``_cli_main`` as an ``isinstance``
    cascade.  A full CLI exercise of the branch lives at T6
    (``test_train_cli_registers_nn_temporal_final_fold_model``); this unit
    test asserts the branch *exists* in the cascade, because forgetting
    a dispatch site was the codebase-map S3 hazard flagged at plan-time.

    The assertion is source-level rather than behavioural: we inspect
    ``_cli_main``'s source and check that it references both
    ``NnTemporalConfig`` (the discriminator) and ``NnTemporalModel`` (the
    constructor).  Either missing means the sixth family is not
    reachable via the train CLI.
    """
    src = inspect.getsource(train._cli_main)
    assert "NnTemporalConfig" in src, (
        "train._cli_main must branch on NnTemporalConfig (plan D13 clause ii); "
        "missing from source — the nn_temporal family is not wired into the train CLI."
    )
    assert "NnTemporalModel" in src, (
        "train._cli_main must instantiate NnTemporalModel inside the NnTemporalConfig branch "
        "(plan D13 clause ii); missing from source."
    )
    # The target-column resolver must also know about the TCN family, otherwise
    # the harness is driven with the wrong target column.
    target_src = inspect.getsource(train._target_column)
    assert "NnTemporalConfig" in target_src, (
        "train._target_column must include NnTemporalConfig in its isinstance tuple "
        "(D13 clause ii); otherwise the harness defaults to 'nd_mw' even when "
        "a different target is configured."
    )


# ===========================================================================
# 5. test_registry_dispatch_maps_nn_temporal_symmetrically  (D13 clause i guard)
# ===========================================================================


def test_registry_dispatch_maps_nn_temporal_symmetrically() -> None:
    """Guards plan D13 clause i — both dispatch dicts have the ``nn_temporal`` entry.

    ``registry/_dispatch.py`` carries two inverse look-ups:

    - ``_TYPE_TO_CLASS["nn_temporal"] == NnTemporalModel`` — used by ``load``.
    - ``_CLASS_NAME_TO_TYPE["NnTemporalModel"] == "nn_temporal"`` — used by
      ``save``.

    Forgetting one of the two dicts is the failure mode that produces
    a run that can be saved but not loaded (or vice versa).  This test
    pins both sides explicitly.

    Plan clause: T6 §Task T6 / D13 clause i.
    """
    assert _TYPE_TO_CLASS.get("nn_temporal") is NnTemporalModel, (
        "_TYPE_TO_CLASS must map 'nn_temporal' -> NnTemporalModel (D13 clause i); "
        f"got {_TYPE_TO_CLASS.get('nn_temporal')!r}."
    )
    assert _CLASS_NAME_TO_TYPE.get("NnTemporalModel") == "nn_temporal", (
        "_CLASS_NAME_TO_TYPE must map 'NnTemporalModel' -> 'nn_temporal' (D13 clause i); "
        f"got {_CLASS_NAME_TO_TYPE.get('NnTemporalModel')!r}."
    )
