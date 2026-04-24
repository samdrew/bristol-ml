"""Spec-derived tests for the Stage 10 ``nn_mlp`` registry dispatch — Task T4.

Every test is derived from:

- ``docs/plans/active/10-simple-nn.md`` §6 Task T4 (the three T4 named
  tests: registry save via protocol, ``list_runs`` includes the new
  ``type``, and the structural-parallel isinstance-branch check).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-4 (registry round-trip at
  ``atol=1e-10``).
- ``docs/plans/active/10-simple-nn.md`` §1 D2 clauses iii + v (one new
  ``isinstance`` branch in ``train.py``; one new entry in each of the
  two registry dispatch dicts).
- ``docs/plans/active/10-simple-nn.md`` §1 D5 (single joblib envelope —
  the registry's ``artefact/model.joblib`` file-path contract remains
  unchanged; the NN's save routes through ``Model.save`` exactly like
  the four Stage-4/7/8 families).

The structural isinstance-branch test below is the Stage-10 parallel of
Stage 8's ``test_harness_build_model_dispatches_scipy_parametric_config``
— recast against ``bristol_ml.train`` because ``train.py``'s dispatcher
is inlined inside ``_cli_main`` (no standalone ``_build_model_from_config``
helper).  A full end-to-end CLI exercise of the ``model=nn_mlp`` path
lives at T5 as ``test_train_cli_registers_nn_mlp_final_fold_model`` —
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
from bristol_ml.models.nn.mlp import NnMlpModel
from bristol_ml.registry._dispatch import _CLASS_NAME_TO_TYPE, _TYPE_TO_CLASS
from conf._schemas import NnMlpConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers — mirror test_registry_save_load.py so the Stage 10
# entries align with the Stage 9 AC-2 suite (plan T4: "extends the Stage 9
# AC-2 test suite with a fifth model family").
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


def _cpu_config() -> NnMlpConfig:
    """Return a CPU-pinned short-budget ``NnMlpConfig``.

    The three T4 tests all exercise the registry + dispatch layer, not
    the training loop itself — a handful of epochs on a single-layer
    MLP is enough to surface save-time and load-time wiring bugs.  CPU
    is pinned so the tests are reproducible inside CI without a GPU.
    """
    return NnMlpConfig(
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


def _fit_tiny_nn(
    n: int = 60, n_features: int = 3, seed: int = 17
) -> tuple[NnMlpModel, pd.DataFrame, pd.Series]:
    """Fit an ``NnMlpModel`` on a small deterministic frame.

    Mirrors the tiny-fixture shape used by ``test_nn_mlp_fit_predict.py``
    and ``test_nn_mlp_save_load.py`` so the round-trip test here agrees
    numerically with the layer-level tests without re-deriving the
    reference output.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(n)
    idx = _hourly_index(n)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=idx)
    target = pd.Series(y, index=idx, name="nd_mw")
    model = NnMlpModel(_cpu_config())
    model.fit(features, target, seed=seed)
    return model, features, target


# ===========================================================================
# 1. test_registry_save_nn_mlp_model_via_protocol
# ===========================================================================


def test_registry_save_nn_mlp_model_via_protocol(tmp_path: Path) -> None:
    """Guards plan T4 / AC-4 — registry save accepts ``NnMlpModel`` via the protocol.

    The Stage 9 registry consumes ``Model.save`` via the protocol (plan
    D9 in Stage 9); adding the fifth model family means (a) registering
    the ``NnMlpModel`` class in ``_dispatch.py`` and (b) routing
    through ``registry.save`` with no model-class changes.  A failure
    here points at a missing key in ``_CLASS_NAME_TO_TYPE`` or a
    missing branch in ``train.py``.

    Plan clause: T4 §Task T4 / AC-4.
    """
    model, _features, _target = _fit_tiny_nn()

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
        "Stage 10 D5 mandates the single joblib envelope at artefact/model.joblib; "
        "missing file indicates NnMlpModel.save did not route through Model.save."
    )
    assert (run_dir / "run.json").is_file(), "expected run.json sidecar to be written"

    sidecar = _read_sidecar(run_dir)
    assert sidecar["type"] == "nn_mlp", (
        f"sidecar 'type' must be 'nn_mlp' for the NN family (D2 clause v); got {sidecar['type']!r}"
    )
    assert sidecar["name"] == model.metadata.name
    assert sidecar["feature_set"] == "weather_only"
    assert sidecar["target"] == "nd_mw"
    # Metrics summary is populated from the per-fold DataFrame.
    assert set(sidecar["metrics"].keys()) == {"mae", "rmse", "mape", "wape"}


# ===========================================================================
# 2. test_registry_load_round_trips_nn_mlp_model  (AC-4 round-trip)
# ===========================================================================


def test_registry_load_round_trips_nn_mlp_model(tmp_path: Path) -> None:
    """Guards AC-4 — save + load + predict agrees to ``atol=1e-10``.

    The stronger AC-4 check: after a full registry round-trip the
    loaded model's ``predict()`` output matches the original to
    ``atol=1e-10``.  This is the same bar the four Stage-4/7/8 models
    are held to in ``test_registry_save_load.py`` — the fifth family
    joins the same suite.

    Plan clause: T4 §Task T4 / AC-4 / NFR-3.
    """
    model, features, _target = _fit_tiny_nn()
    predicted_before = model.predict(features)

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    loaded = registry.load(run_id, registry_dir=tmp_path)

    assert isinstance(loaded, NnMlpModel), (
        f"registry.load must return an NnMlpModel for a 'nn_mlp' run; "
        f"got {type(loaded).__name__!r} — check _TYPE_TO_CLASS."
    )
    predicted_after = loaded.predict(features)

    np.testing.assert_allclose(
        predicted_after.to_numpy(),
        predicted_before.to_numpy(),
        atol=1e-10,
        err_msg=(
            "registry.save + registry.load broke predict() numerical equivalence; "
            "the NnMlpModel artefact or registry dispatcher is dropping state "
            "(plan AC-4 / NFR-3)."
        ),
    )
    # Index and name round-trip — inherits the Stage 4 Model protocol invariants.
    assert predicted_after.index.equals(predicted_before.index)
    assert predicted_after.name == predicted_before.name


# ===========================================================================
# 3. test_registry_list_runs_includes_nn_mlp_type
# ===========================================================================


def test_registry_list_runs_includes_nn_mlp_type(tmp_path: Path) -> None:
    """Guards plan T4 — ``list_runs(model_type='nn_mlp')`` returns the NN run.

    The Stage 9 ``list_runs`` filter is exact-match on the sidecar's
    ``type`` field (plan D7 in Stage 9).  For the Stage 10 leaderboard
    demo to surface the new family, a saved ``NnMlpModel`` must be
    reachable via ``list_runs(model_type='nn_mlp')``.

    Plan clause: T4 §Task T4.
    """
    model, _features, _target = _fit_tiny_nn()

    run_id = registry.save(
        model,
        _fake_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    runs = registry.list_runs(model_type="nn_mlp", registry_dir=tmp_path)
    assert [r["run_id"] for r in runs] == [run_id], (
        f"list_runs(model_type='nn_mlp') must return exactly the one nn_mlp run; "
        f"got {[r['run_id'] for r in runs]!r}"
    )
    assert runs[0]["type"] == "nn_mlp"


# ===========================================================================
# 4. test_nn_mlp_is_dispatched_by_train_cli_isinstance_branch  (structural)
# ===========================================================================


def test_nn_mlp_is_dispatched_by_train_cli_isinstance_branch() -> None:
    """Guards plan T4 / D2 clause iii — ``train.py`` carries the NN branch.

    Structural parallel of the Stage 8 T6 dispatch test.  ``train.py``
    does not expose a standalone ``_build_model_from_config`` helper —
    the dispatch is inlined inside ``_cli_main`` as an ``isinstance``
    cascade.  A full CLI exercise of the branch lives at T5
    (``test_train_cli_registers_nn_mlp_final_fold_model``); this unit
    test asserts the branch *exists* in the cascade, because forgetting
    a dispatch site was the codebase-map S3 hazard flagged at plan-time.

    The assertion is source-level rather than behavioural: we inspect
    ``_cli_main``'s source and check that it references both
    ``NnMlpConfig`` (the discriminator) and ``NnMlpModel`` (the
    constructor).  Either missing means the fifth family is not
    reachable via the train CLI.
    """
    src = inspect.getsource(train._cli_main)
    assert "NnMlpConfig" in src, (
        "train._cli_main must branch on NnMlpConfig (plan D2 clause iii); "
        "missing from source — the nn_mlp family is not wired into the train CLI."
    )
    assert "NnMlpModel" in src, (
        "train._cli_main must instantiate NnMlpModel inside the NnMlpConfig branch "
        "(plan D2 clause iii); missing from source."
    )
    # The target-column resolver must also know about the NN family, otherwise
    # the harness is driven with the wrong target column.
    target_src = inspect.getsource(train._target_column)
    assert "NnMlpConfig" in target_src, (
        "train._target_column must include NnMlpConfig in its isinstance tuple "
        "(D2 clause iii); otherwise the harness defaults to 'nd_mw' even when "
        "a different target is configured."
    )


# ===========================================================================
# 5. test_registry_dispatch_maps_nn_mlp_symmetrically  (D2 clause v guard)
# ===========================================================================


def test_registry_dispatch_maps_nn_mlp_symmetrically() -> None:
    """Guards plan D2 clause v — both dispatch dicts have the ``nn_mlp`` entry.

    ``registry/_dispatch.py`` carries two inverse look-ups:

    - ``_TYPE_TO_CLASS["nn_mlp"] == NnMlpModel`` — used by ``load``.
    - ``_CLASS_NAME_TO_TYPE["NnMlpModel"] == "nn_mlp"`` — used by
      ``save``.

    Forgetting one of the two dicts is the failure mode that produces
    a run that can be saved but not loaded (or vice versa).  This test
    pins both sides explicitly.
    """
    assert _TYPE_TO_CLASS.get("nn_mlp") is NnMlpModel, (
        "_TYPE_TO_CLASS must map 'nn_mlp' -> NnMlpModel (D2 clause v); "
        f"got {_TYPE_TO_CLASS.get('nn_mlp')!r}."
    )
    assert _CLASS_NAME_TO_TYPE.get("NnMlpModel") == "nn_mlp", (
        "_CLASS_NAME_TO_TYPE must map 'NnMlpModel' -> 'nn_mlp' (D2 clause v); "
        f"got {_CLASS_NAME_TO_TYPE.get('NnMlpModel')!r}."
    )
