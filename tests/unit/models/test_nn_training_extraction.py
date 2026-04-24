"""Structural regression guards for the Stage 11 T1 training-loop extraction.

Guards derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T1 (D4 / AC-2).
- ``src/bristol_ml/models/nn/CLAUDE.md`` lines 141-154 ("Extraction seam for Stage 11").
- Stage 10 plan D10 — the extraction seam flagged inside
  ``NnMlpModel._run_training_loop`` / ``NnMlpModel.fit``.

These tests exercise *structure*, not semantics.  The semantic coverage is
already provided by ``tests/unit/models/test_nn_mlp_fit_predict.py`` (24
tests, all passing after the refactor).  The structural tests here fail
loudly if:

1. Someone deletes ``_training.py`` and moves the body back into ``mlp.py``
   (test 1), or
2. ``NnMlpModel.fit`` is accidentally reverted to call an in-module loop
   rather than the shared helper — a copy-paste regression the end-to-end
   tests would not catch if the body were duplicated faithfully (test 2).

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause (T1 / D10) it guards.
- ``device="cpu"`` throughout — NFR-1 bit-identity holds only on CPU.
- No ``@pytest.mark.gpu``, no ``skip``, no ``xfail``.
- Both tests run in well under a second on any CI host.
"""

from __future__ import annotations

import unittest.mock

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports-under-test — keep them tuple-form to mirror the pattern in
# ``test_nn_mlp_fit_predict.py`` (single import per module).
# ---------------------------------------------------------------------------
from bristol_ml.models.nn._training import _seed_four_streams, run_training_loop
from bristol_ml.models.nn.mlp import NnMlpModel
from conf._schemas import NnMlpConfig

# ---------------------------------------------------------------------------
# Shared fixture helpers — copied from ``test_nn_mlp_fit_predict.py``
# so this file is self-contained and no import coupling is introduced
# between test modules.
# ---------------------------------------------------------------------------


def _tiny_fixture(
    n: int = 60, n_features: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair for structural fit tests.

    60 rows x 3 features satisfies the 10 % val-tail split (n_val = 6,
    n_train = 54) with comfortable room, and keeps each test well inside
    a one-second budget.  The data function is immaterial to the structural
    assertions.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(size=n)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


def _cpu_config(**overrides: object) -> NnMlpConfig:
    """Return a CPU-pinned, minimal-budget ``NnMlpConfig`` for structural tests."""
    kwargs: dict[str, object] = dict(
        hidden_sizes=[8],
        activation="relu",
        dropout=0.0,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=2,
        patience=5,
        seed=None,
        device="cpu",
        target_column="nd_mw",
        feature_columns=None,
    )
    kwargs.update(overrides)
    return NnMlpConfig(**kwargs)  # type: ignore[arg-type]


# ===========================================================================
# 1. test_nn_training_helpers_are_importable_from_shared_module
#
# Plan clause: Stage 11 T1 / D4 / Stage 10 D10.
# ===========================================================================


def test_nn_training_helpers_are_importable_from_shared_module() -> None:
    """Guards that ``_training.py`` exists and exports both callables.

    The extraction seam (Stage 10 D10 / Stage 11 T1 / CLAUDE.md lines
    141-154) mandates that the two helpers — ``_seed_four_streams`` and
    ``run_training_loop`` — live in
    ``bristol_ml.models.nn._training``.  If someone re-inlines the loop
    into ``mlp.py`` and deletes ``_training.py``, this import will raise
    ``ImportError`` and this test will fail immediately, surfacing the
    regression before any semantic test runs.

    The assertion is minimal: both names are callable.  Behavioural
    correctness is the province of ``test_nn_mlp_fit_predict.py``; this
    test only verifies the structural contract on the extraction module.

    Plan clause: Stage 11 T1 (D4 / Stage 10 D10).
    """
    assert callable(_seed_four_streams), (
        "_seed_four_streams must be a callable exported from "
        "bristol_ml.models.nn._training (Stage 11 T1 / Stage 10 D10 extraction seam)."
    )
    assert callable(run_training_loop), (
        "run_training_loop must be a callable exported from "
        "bristol_ml.models.nn._training (Stage 11 T1 / Stage 10 D10 extraction seam)."
    )


# ===========================================================================
# 2. test_nn_mlp_fit_calls_shared_run_training_loop
#
# Plan clause: Stage 11 T1 / D4 / AC-2 regression guard named
# ``test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction``.
# ===========================================================================


def test_nn_mlp_fit_calls_shared_run_training_loop() -> None:
    """Guards that ``NnMlpModel.fit`` calls the shared helper, not an in-module copy.

    Monkeypatches ``bristol_ml.models.nn.mlp.run_training_loop`` with a spy
    (a :class:`unittest.mock.MagicMock` that delegates to the real
    implementation via ``side_effect``).  After running a minimal
    ``NnMlpModel.fit`` the spy must have been called exactly once and the
    call must have included the keyword arguments ``max_epochs``,
    ``patience``, ``loss_history``, and ``criterion`` — the full signature of
    the shared helper.

    Why this test catches copy-paste regressions that the end-to-end tests
    miss: if a future author copies the loop body back into ``fit`` verbatim,
    the semantic output is identical (same loop logic) so the 24 existing
    ``test_nn_mlp_fit_predict.py`` tests all pass.  But the name
    ``bristol_ml.models.nn.mlp.run_training_loop`` would no longer refer to
    the import site inside ``fit`` — it would refer to either a local or a
    module-level name that is never called — so the spy would record zero
    calls, and this test would fail.

    Plan clause: Stage 11 T1 / D4 / AC-2
    (``test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction``).
    """
    # Import the real implementation so the spy can delegate to it.
    real_run_training_loop = run_training_loop

    spy = unittest.mock.MagicMock(side_effect=real_run_training_loop)

    features, target = _tiny_fixture()
    cfg = _cpu_config(batch_size=8, max_epochs=2, patience=5)

    with unittest.mock.patch(
        "bristol_ml.models.nn.mlp.run_training_loop",
        spy,
    ):
        model = NnMlpModel(cfg)
        model.fit(features, target, seed=0)

    # --- Structural assertion 1: called exactly once per fit() -----------
    assert spy.call_count == 1, (
        f"run_training_loop must be called exactly once per NnMlpModel.fit(); "
        f"got {spy.call_count} calls.  If the loop was re-inlined into mlp.py, "
        f"the spy at the import-name site records zero calls (Stage 11 T1 / D4)."
    )

    # --- Structural assertion 2: called with the expected keyword args ---
    # Retrieve the keyword arguments from the single call.
    _, kwargs = spy.call_args
    expected_kwargs = {"max_epochs", "patience", "loss_history", "criterion"}
    missing_kwargs = expected_kwargs - set(kwargs)
    assert not missing_kwargs, (
        f"run_training_loop was called but is missing keyword arguments "
        f"{missing_kwargs!r}.  The full shared-helper signature requires "
        f"max_epochs, patience, loss_history, and criterion (Stage 11 T1 / D4)."
    )
