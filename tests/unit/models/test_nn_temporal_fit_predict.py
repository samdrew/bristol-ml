"""Spec-derived tests for the Stage 11 ``NnTemporalModel`` fit / predict path — Task T4.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T4 (lines 494-503): the 9
  T4 named tests (round-trip, seeded-identity CPU, seeded-close CUDA,
  seed-diversity, loss-history, epoch-callback, cold-start,
  shared-training-loop, causal-padding).
- ``docs/plans/active/11-complex-nn.md`` §4 AC-1, AC-2 and NFR-1
  (reproducibility and loss-history provenance).
- ``src/bristol_ml/models/nn/temporal.py`` public surface and invariants:
  ``fit``, ``predict``, ``_build_temporal_module_class``,
  ``_TemporalBlockImpl``, ``_NnTemporalModuleImpl``, ``_make_tcn``.
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section
  (re-entrancy / cold-start, predict-before-fit RuntimeError).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- All tests pin ``device="cpu"`` unless explicitly exercising the CUDA
  branch — NFR-1 bit-identity only holds on CPU; the CUDA marker test
  uses the NFR-1 close-match tolerance (``atol=1e-5, rtol=1e-4``).
- No ``xfail``, no ``skip`` (the one ``pytest.mark.gpu`` test is opted
  out of the default run via ``addopts``, not skipped at import time).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bristol_ml.models.nn.temporal import (
    NnTemporalModel,
    _build_temporal_module_class,
)
from conf._schemas import NnTemporalConfig

# ---------------------------------------------------------------------------
# Shared tiny fixture — deterministic, CPU-friendly, sized so that at
# seq_len=48 we have enough training + val windows inside the internal
# train/val split.  n=500 gives ~500-48=452 usable windows on the
# training slice (after the internal val tail is removed) — sufficient
# for a 3-epoch convergence check.
# ---------------------------------------------------------------------------


def _tiny_temporal_fixture(
    n: int = 500, n_features: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair for TCN fit tests.

    The target is a linear + mild sinusoid function of the features plus
    Gaussian noise, so a TCN with sensible defaults can fit it in a
    handful of epochs on CPU.  The seed argument controls the numpy
    RNG used to *draw the data* — it is not the fit seed.

    The index is UTC-aware hourly to match the harness convention (plan
    §4 AC-1).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(size=n)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


def _cpu_temporal_config(**overrides: object) -> NnTemporalConfig:
    """Return a CPU-pinned, short-budget ``NnTemporalConfig`` for CI-friendly fits.

    Architecture choices:
    - ``seq_len=48`` — small enough for fast windowing; large enough for
      sensible TCN behaviour.
    - ``num_blocks=2, kernel_size=3`` — receptive field =
      1 + 2*(3-1)*(2**2-1) = 13; min seq_len = max(6, 1) = 6, well
      within 48.
    - ``weight_norm=False`` — removes weight-norm extra state_dict keys
      that add noise to bit-identity checks without changing the test's
      focus.
    - ``dropout=0.0`` — needed for the causal-padding test (dropout
      introduces stochasticity; ``eval()`` turns it off, but being
      explicit is safer).
    - ``max_epochs=3, patience=3`` — fast; patience > max_epochs so
      early stopping does not interfere with history-length assertions.
    """
    kwargs: dict[str, object] = dict(
        seq_len=48,
        num_blocks=2,
        channels=8,
        kernel_size=3,
        dropout=0.0,
        weight_norm=False,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=64,
        max_epochs=3,
        patience=3,
        seed=None,
        device="cpu",
        target_column="nd_mw",
        feature_columns=None,
    )
    kwargs.update(overrides)
    return NnTemporalConfig(**kwargs)  # type: ignore[arg-type]


# ===========================================================================
# 1. test_nn_temporal_fit_predict_round_trip_on_tiny_fixture  (AC-1)
# ===========================================================================


def test_nn_temporal_fit_predict_round_trip_on_tiny_fixture() -> None:
    """Guards AC-1: ``fit`` then ``predict`` returns a finite Series on CPU.

    Pins ``device="cpu"`` because NFR-1 bit-identity is only guaranteed
    on CPU; pins ``max_epochs=3`` so the test budget is short.  Also
    checks the warmup-prefix design invariant: ``len(predict) == len(features)``
    with no seq_len truncation visible to the caller, per the plan T4
    content note on predict returning one row per input row.

    Plan clause: T4 §Task T4 named test / AC-1.
    """
    features, target = _tiny_temporal_fixture()
    model = NnTemporalModel(_cpu_temporal_config())
    model.fit(features, target, seed=123)

    # Predict on an out-of-sample slice (last 50 rows).
    features_test = features.iloc[-50:]
    y_pred = model.predict(features_test)

    assert isinstance(y_pred, pd.Series), (
        f"NnTemporalModel.predict must return a pd.Series; got {type(y_pred).__name__}."
    )
    assert len(y_pred) == len(features_test), (
        f"predict output length {len(y_pred)} must equal len(features_test)={len(features_test)}; "
        "the warmup-prefix design must deliver one prediction per input row with no "
        "seq_len truncation visible to the caller (plan T4 content note)."
    )
    assert y_pred.index.equals(features_test.index), (
        "predict output must be indexed on features_test.index "
        "(Model-protocol convention / plan T4 AC-1)."
    )
    assert np.all(np.isfinite(y_pred.to_numpy())), (
        f"predict output must be finite; got {y_pred.to_numpy()!r}."
    )

    md = model.metadata
    assert md.fit_utc is not None, "metadata.fit_utc must be set after fit() (plan T4 AC-1)."
    assert md.feature_columns == tuple(features.columns), (
        f"metadata.feature_columns must match fitted columns; got {md.feature_columns!r} "
        f"vs {tuple(features.columns)!r}."
    )


# ===========================================================================
# 2. test_nn_temporal_seeded_runs_produce_identical_state_dicts_on_cpu
#    (NFR-1 CPU bit-identity)
# ===========================================================================


def test_nn_temporal_seeded_runs_produce_identical_state_dicts_on_cpu() -> None:
    """Guards NFR-1 CPU half: two seeded fits produce byte-identical params.

    Uses ``torch.equal`` on every parameter tensor and registered buffer
    (scaler buffers ``feature_mean`` / ``feature_std`` / ``target_mean`` /
    ``target_std`` are fitted from data so they must also match
    bit-for-bit on CPU).  ``predict`` output is additionally byte-compared.

    Plan clause: T4 §Task T4 named test / NFR-1 CPU half.
    """
    import torch

    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config()

    model_a = NnTemporalModel(cfg)
    model_a.fit(features, target, seed=42)

    model_b = NnTemporalModel(cfg)
    model_b.fit(features, target, seed=42)

    assert model_a._module is not None and model_b._module is not None
    sd_a = model_a._module.state_dict()
    sd_b = model_b._module.state_dict()
    assert sd_a.keys() == sd_b.keys(), (
        f"state_dict key sets must match; got {set(sd_a)} vs {set(sd_b)}."
    )
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), (
            f"state_dict[{key!r}] must be torch.equal across two seeded "
            f"fits on CPU (NFR-1 bit-identity)."
        )

    # predict() must also be bit-identical.
    pred_a = model_a.predict(features)
    pred_b = model_b.predict(features)
    pd.testing.assert_series_equal(pred_a, pred_b, check_exact=True)


# ===========================================================================
# 3. test_nn_temporal_seeded_runs_match_on_cuda_within_tolerance
#    (NFR-1 GPU) — @pytest.mark.gpu
# ===========================================================================


@pytest.mark.gpu
def test_nn_temporal_seeded_runs_match_on_cuda_within_tolerance() -> None:
    """Guards NFR-1 GPU half: seeded fits close-match under atol/rtol on CUDA.

    The ``@pytest.mark.gpu`` marker puts this test behind ``-m gpu`` in
    the pyproject addopts selector; inside the test we still guard
    ``torch.cuda.is_available()`` — the marker gates selection, the
    guard here gates execution (same pattern as Stage 10 precedent
    ``test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance``).

    Tolerances (``atol=1e-5, rtol=1e-4``) come from plan NFR-1: cuBLAS /
    cuDNN nondeterminism on reductions is real even with
    ``cudnn.deterministic=True``; bit-identity is not achievable on CUDA.

    Plan clause: T4 §Task T4 named test / NFR-1 GPU half.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host.")

    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config(device="cuda")

    model_a = NnTemporalModel(cfg)
    model_a.fit(features, target, seed=7)

    model_b = NnTemporalModel(cfg)
    model_b.fit(features, target, seed=7)

    pred_a = torch.from_numpy(model_a.predict(features).to_numpy())
    pred_b = torch.from_numpy(model_b.predict(features).to_numpy())
    assert torch.allclose(pred_a, pred_b, atol=1e-5, rtol=1e-4), (
        "Two seeded fits on CUDA must close-match within atol=1e-5 rtol=1e-4 (NFR-1 GPU half)."
    )


# ===========================================================================
# 4. test_nn_temporal_different_seeds_produce_different_state_dicts
# ===========================================================================


def test_nn_temporal_different_seeds_produce_different_state_dicts() -> None:
    """Guards the seed-diversity half of the reproducibility story.

    If two *different* seeds produced identical params, the four-stream
    seed helper (``_seed_four_streams``) would be broken or the network
    initialisation wouldn't depend on the seed at all.  The assertion is
    that at least one parameter tensor (not buffer) differs — scaler
    buffers are fitted from the same data and therefore seed-invariant.

    Plan clause: T4 §Task T4 named test.
    """
    import torch

    features, target = _tiny_temporal_fixture()

    model_a = NnTemporalModel(_cpu_temporal_config())
    model_a.fit(features, target, seed=0)

    model_b = NnTemporalModel(_cpu_temporal_config())
    model_b.fit(features, target, seed=1)

    assert model_a._module is not None and model_b._module is not None
    sd_a = model_a._module.state_dict()
    sd_b = model_b._module.state_dict()

    # At least one *parameter* (not buffer) tensor must differ.
    param_names = {name for name, _ in model_a._module.named_parameters()}
    assert param_names, "The TCN module must expose at least one named parameter."
    any_diff = False
    for key in sd_a:
        if key not in param_names:
            continue  # buffers are seed-invariant
        if not torch.equal(sd_a[key], sd_b[key]):
            any_diff = True
            break
    assert any_diff, (
        "At least one parameter tensor must differ between seed=0 and seed=1 fits "
        "(otherwise the seed is not wired into initialisation / training)."
    )


# ===========================================================================
# 5. test_nn_temporal_fit_populates_loss_history_per_epoch  (AC-2)
# ===========================================================================


def test_nn_temporal_fit_populates_loss_history_per_epoch() -> None:
    """Guards AC-2: ``loss_history_`` grows by one dict per epoch.

    Each entry is a dict with keys ``{"epoch", "train_loss", "val_loss"}``.
    The ``epoch`` value is an ``int`` (1-indexed, strictly increasing).
    ``train_loss`` and ``val_loss`` are finite ``float``.  Length is
    <= ``cfg.max_epochs``.

    Plan clause: T4 §Task T4 named test / AC-2.
    """
    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config(max_epochs=3, patience=3)
    model = NnTemporalModel(cfg)
    model.fit(features, target, seed=0)

    history = model.loss_history_
    assert isinstance(history, list), f"loss_history_ must be a list; got {type(history).__name__}."
    assert len(history) > 0, "loss_history_ must be non-empty after fit()."
    assert len(history) <= cfg.max_epochs, (
        f"loss_history_ length {len(history)} must be <= max_epochs={cfg.max_epochs}."
    )

    prev_epoch = 0
    for i, entry in enumerate(history):
        assert set(entry.keys()) == {"epoch", "train_loss", "val_loss"}, (
            f"loss_history_[{i}] keys must be {{'epoch','train_loss','val_loss'}}; "
            f"got {set(entry.keys())!r}."
        )
        # ``epoch`` must be int per plan (not float, not bool).
        assert isinstance(entry["epoch"], int), (
            f"loss_history_[{i}]['epoch'] must be int; got {type(entry['epoch']).__name__}."
        )
        assert entry["epoch"] > prev_epoch, (
            f"loss_history_ epochs must be strictly increasing; "
            f"got {entry['epoch']} after {prev_epoch} at index {i}."
        )
        # First epoch must be 1-based.
        if i == 0:
            assert entry["epoch"] == 1, (
                f"loss_history_[0]['epoch'] must be 1 (1-based); got {entry['epoch']!r}."
            )
        prev_epoch = entry["epoch"]
        for key in ("train_loss", "val_loss"):
            assert isinstance(entry[key], float), (
                f"loss_history_[{i}][{key!r}] must be float; got {type(entry[key]).__name__}."
            )
            assert math.isfinite(entry[key]), (
                f"loss_history_[{i}][{key!r}] must be finite; got {entry[key]!r}."
            )


# ===========================================================================
# 6. test_nn_temporal_fit_invokes_epoch_callback_when_provided  (AC-2)
# ===========================================================================


def test_nn_temporal_fit_invokes_epoch_callback_when_provided() -> None:
    """Guards AC-2: ``epoch_callback`` receives a defensive copy per epoch.

    Passes ``epoch_callback=lambda d: captured.append(d); d["epoch"] = -999``.
    After fit, ``model.loss_history_[0]["epoch"]`` must still be ``1`` because
    the callback receives a ``dict(entry)`` copy — external mutation of the
    callback dict must not corrupt the canonical history entry (Stage 10
    Phase 3 lesson inherited).

    Also asserts ``len(captured) == len(model.loss_history_)`` — the callback
    is invoked exactly once per appended history entry.

    Plan clause: T4 §Task T4 named test / AC-2 / defensive-copy semantics.
    """
    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config(max_epochs=3, patience=3)

    captured: list[dict] = []

    def cb(entry: dict) -> None:
        captured.append(entry)
        # Deliberately mutate the received dict.  A defensive copy on the
        # production side means this must not corrupt the stored history.
        entry["epoch"] = -999

    model = NnTemporalModel(cfg)
    model.fit(features, target, seed=0, epoch_callback=cb)

    assert len(captured) == len(model.loss_history_), (
        f"epoch_callback must be invoked once per appended history entry; "
        f"got {len(captured)} calls vs {len(model.loss_history_)} entries."
    )
    # The canonical stored history must not have been corrupted by the
    # external mutation of the callback argument.
    assert model.loss_history_[0]["epoch"] == 1, (
        "loss_history_[0]['epoch'] must be 1 after fit() even though the "
        "epoch_callback mutated its copy to -999; the production code must "
        "pass a defensive dict(entry) copy to the callback, not the "
        "stored dict itself (Stage 10 Phase 3 lesson inherited)."
    )
    for rx, stored in zip(captured, model.loss_history_, strict=True):
        assert rx is not stored, (
            "epoch_callback must receive a defensive copy, not the same dict "
            "object that is appended to loss_history_ (Stage 10 Phase 3 review)."
        )


# ===========================================================================
# 7. test_nn_temporal_fit_uses_cold_start_per_fold_when_called_repeatedly
#    (D8 cold-start pattern)
# ===========================================================================


def test_nn_temporal_fit_uses_cold_start_per_fold_when_called_repeatedly() -> None:
    """Guards D8: a second ``fit()`` cold-starts — no carry-over from the first.

    Calls ``fit`` twice on the *same* ``(features, target)`` with the *same*
    seed; the post-second-fit ``loss_history_`` must match the post-first-fit
    ``loss_history_`` exactly.  Also verifies that ``len(loss_history_)``
    after the second fit equals only the second fit's epochs (no
    accumulation from the first run).

    Plan clause: T4 §Task T4 named test / Stage 10 D8 cold-start pattern
    inherited / bristol_ml.models.CLAUDE.md re-entrancy contract.
    """
    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config(max_epochs=3, patience=3)

    model = NnTemporalModel(cfg)
    model.fit(features, target, seed=11)
    first_history = [dict(e) for e in model.loss_history_]
    first_len = len(model.loss_history_)

    model.fit(features, target, seed=11)
    second_history = [dict(e) for e in model.loss_history_]

    assert len(model.loss_history_) == first_len, (
        "loss_history_ must not accumulate across two fit() calls; "
        f"first fit had {first_len} epochs, second fit appended to "
        f"produce {len(model.loss_history_)} entries — cold-start "
        "must reset the list (Stage 10 D8 pattern)."
    )
    assert first_history == second_history, (
        "Two fit() calls with the same seed must produce identical "
        "loss_history_ entries — the second fit must cold-start, not "
        "warm-start (D8).\n"
        f"first:  {first_history!r}\n"
        f"second: {second_history!r}"
    )


# ===========================================================================
# 8. test_nn_temporal_fit_uses_shared_training_loop  (AC-2 / D4 seam)
# ===========================================================================


def test_nn_temporal_fit_uses_shared_training_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards AC-2 and D4: ``run_training_loop`` is called exactly once per fit.

    Monkeypatches ``bristol_ml.models.nn.temporal.run_training_loop`` with
    a spy that records calls and returns a minimal valid result.  After a
    fit on a tiny fixture, the spy must have been called exactly once —
    structurally verifying that the D4 extraction seam (Stage 11 T1) is
    wired correctly and that ``NnTemporalModel.fit`` delegates to the shared
    helper rather than implementing its own loop.

    The spy returns ``(dict(module.state_dict()), 1)`` so the ``fit`` path
    can load_state_dict and complete normally.

    Plan clause: T4 §Task T4 named test / AC-2 / Stage 11 T1 D4 seam.
    """
    import bristol_ml.models.nn.temporal as _temporal_mod

    spy_calls: list[dict] = []
    _real_run_training_loop = _temporal_mod.run_training_loop

    def _spy_run_training_loop(
        module,
        train_loader,
        val_loader,
        *,
        optimiser,
        criterion,
        device,
        max_epochs,
        patience,
        loss_history,
        epoch_callback=None,
    ):
        # Record the call.
        spy_calls.append({"max_epochs": max_epochs, "patience": patience})
        # Populate loss_history with one minimal entry so downstream
        # assertions on loss_history_ (e.g. predict paths) hold.
        loss_history.append({"epoch": 1, "train_loss": 1.0, "val_loss": 1.0})
        if epoch_callback is not None:
            epoch_callback({"epoch": 1, "train_loss": 1.0, "val_loss": 1.0})
        # Return CPU state_dict clone + best_epoch.
        best_state_dict = {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}
        return best_state_dict, 1

    monkeypatch.setattr(_temporal_mod, "run_training_loop", _spy_run_training_loop)

    features, target = _tiny_temporal_fixture()
    cfg = _cpu_temporal_config()
    model = NnTemporalModel(cfg)
    model.fit(features, target, seed=0)

    assert len(spy_calls) == 1, (
        f"run_training_loop must be called exactly once per fit(); "
        f"spy recorded {len(spy_calls)} calls.  "
        "If this fails, NnTemporalModel.fit is not delegating to the "
        "shared helper (D4 seam broken)."
    )


# ===========================================================================
# 9. test_nn_temporal_causal_padding_does_not_leak_future
# ===========================================================================


def test_nn_temporal_causal_padding_does_not_leak_future() -> None:
    """Guards the causal-padding invariant: future positions must not influence
    earlier outputs inside ``_TemporalBlockImpl``.

    The test extracts the first ``_TemporalBlockImpl`` from a freshly-built
    ``_NnTemporalModuleImpl`` and feeds it two ``(1, channels, seq_len)``
    tensors that agree on positions ``[0..t]`` and differ on positions
    ``(t..seq_len)``.  The block's output at position ``t`` must be
    identical across the two inputs under ``torch.equal`` — if causal
    padding pads right instead of left (i.e. the block leaks future
    context), the output at position ``t`` will see the future-differing
    positions and the assertion fails loudly.

    ``t = seq_len // 2`` is the mid-sequence split that maximises the
    gap between agreement and disagreement without hitting boundary effects.

    Dropout is set to ``0.0`` and ``block.eval()`` is called so
    stochastic dropout cannot produce a false positive (different outputs
    due to mask randomness rather than padding direction).

    Plan clause: T4 §Task T4 named test / causal-padding structural guard.
    """
    import torch

    seq_len = 24  # small enough to be fast; large enough to have a meaningful t
    n_features = 4
    channels = 8
    kernel_size = 3
    num_blocks = 2

    # Build the module class lazily and instantiate the TCN directly.
    # No ``NnTemporalConfig`` instance is needed — we call the module
    # class constructor directly with the architecture kwargs so we can
    # probe ``_TemporalBlockImpl`` in isolation without a full
    # ``NnTemporalModel.fit`` path.
    module_cls = _build_temporal_module_class()
    module = module_cls(
        input_dim=n_features,
        seq_len=seq_len,
        num_blocks=num_blocks,
        channels=channels,
        kernel_size=kernel_size,
        dropout=0.0,
        weight_norm=False,
    )
    module.eval()

    # Extract the first temporal block (dilation=1).
    block = next(iter(module.blocks))
    block.eval()

    # The block expects input of shape (B, C_in, L); the first block's
    # C_in is n_features (input_dim) — see _NnTemporalModuleImpl.__init__
    # where in_ch starts at input_dim and advances to channels after block 0.
    t = seq_len // 2

    rng = torch.Generator()
    rng.manual_seed(7)
    # Base tensor: random values everywhere.
    x_base = torch.randn(1, n_features, seq_len, generator=rng)
    # Future-differing tensor: same as base on [0..t], different on (t..seq_len).
    x_future_diff = x_base.clone()
    x_future_diff[:, :, t + 1 :] = torch.randn(1, n_features, seq_len - t - 1, generator=rng)

    with torch.no_grad():
        out_base = block(x_base)
        out_future_diff = block(x_future_diff)

    # The outputs at position t must be identical: the two inputs agree
    # on all positions [0..t] so a causally-correct left-pad cannot
    # produce different outputs at t.  Right-pad would allow position t
    # to see positions [t+1..t+pad], which differ — test fails loudly.
    assert torch.equal(out_base[:, :, t], out_future_diff[:, :, t]), (
        "Causal padding test failed: _TemporalBlockImpl output at position t "
        f"(t={t}, seq_len={seq_len}) differs between two inputs that agree on "
        f"positions [0..{t}] and differ only on positions [{t + 1}..{seq_len - 1}]. "
        "This means the block is leaking future context (right-pad instead of "
        "left-pad), violating the Bai et al. 2018 causal-TCN design "
        "(plan T4 causal-padding structural guard)."
    )
