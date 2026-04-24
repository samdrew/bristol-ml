"""Spec-derived tests for the Stage 10 ``NnMlpModel`` fit / predict path — Task T2.

Every test is derived from:

- ``docs/plans/active/10-simple-nn.md`` §Task T2 (lines 364-385): the 10
  T2 named tests (round-trip, seeded-identity CPU, seeded-close CUDA,
  seed-diversity, loss-history, epoch-callback, early-stopping,
  cold-start, device-auto precedence, device-pin & invalid).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-1, AC-2, AC-3 and NFR-1,
  NFR-3 (reproducibility and loss-history provenance).
- ``src/bristol_ml/models/nn/mlp.py`` module-level helpers
  (``_select_device``, ``_seed_four_streams``, ``_make_mlp``,
  ``_ALLOWED_DEVICES``) and :class:`NnMlpModel` public surface.
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section
  (re-entrancy / cold-start, predict-before-fit RuntimeError).

No production code is modified here.  If any test below fails the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- All tests pin ``device="cpu"`` unless explicitly exercising the auto /
  CUDA branch — NFR-1 bit-identity only holds on CPU; the CUDA marker
  test uses the NFR-1 close-match tolerance (``atol=1e-5, rtol=1e-4``).
- No ``xfail``, no ``skip`` (the one ``pytest.mark.gpu`` test is opted
  out of the default run via ``addopts``, not skipped at import time).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bristol_ml.models.nn.mlp import (
    _ALLOWED_DEVICES,
    NnMlpModel,
    _select_device,
)
from conf._schemas import NnMlpConfig

# ---------------------------------------------------------------------------
# Shared tiny fixture — deterministic, CPU-friendly, small enough to fit
# in a unit-test budget but large enough to cover a 10% val tail (>=10
# rows required so n_val = max(1, n//10) = n//10 and n_train >= 9).
# ---------------------------------------------------------------------------


def _tiny_fixture(
    n: int = 60, n_features: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair for fit tests.

    The target is a linear + mild sinusoid function of the features plus
    Gaussian noise, so an MLP with sensible defaults can fit it in a
    handful of epochs on CPU.  The seed argument controls the numpy
    RNG used to *draw the data* — it is not the fit seed.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(size=n)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


def _cpu_config(**overrides: object) -> NnMlpConfig:
    """Return a CPU-pinned, short-budget ``NnMlpConfig`` for CI-friendly fits."""
    kwargs: dict[str, object] = dict(
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
    kwargs.update(overrides)
    return NnMlpConfig(**kwargs)  # type: ignore[arg-type]


# ===========================================================================
# 1. test_nn_mlp_fit_predict_round_trip_on_tiny_fixture  (AC-1)
# ===========================================================================


def test_nn_mlp_fit_predict_round_trip_on_tiny_fixture() -> None:
    """Guards AC-1: ``fit`` then ``predict`` returns a finite Series on CPU.

    Pins ``device="cpu"`` because NFR-1 bit-identity is only guaranteed
    on CPU; also pins ``max_epochs=5`` so the test budget is short.

    Plan clause: T2 §Task T2 named test / AC-1.
    """
    features, target = _tiny_fixture()
    model = NnMlpModel(_cpu_config())
    model.fit(features, target, seed=123)

    pred = model.predict(features)

    assert isinstance(pred, pd.Series), (
        f"NnMlpModel.predict must return a pd.Series; got {type(pred).__name__}."
    )
    assert pred.index.equals(features.index), (
        "predict output must be indexed on features.index (Model-protocol convention)."
    )
    assert len(pred) == len(features), (
        f"predict output length {len(pred)} must equal len(features)={len(features)}."
    )
    assert np.all(np.isfinite(pred.to_numpy())), (
        f"predict output must be finite; got {pred.to_numpy()!r}."
    )
    assert pred.name == "nd_mw", (
        f"predict output name must equal config.target_column='nd_mw'; got {pred.name!r}."
    )

    # ``metadata.fit_utc`` flips from None to a datetime (protocol contract).
    md = model.metadata
    assert md.fit_utc is not None, "metadata.fit_utc must be set after fit()."
    assert md.feature_columns == tuple(features.columns), (
        f"metadata.feature_columns must match fitted columns; got {md.feature_columns!r}."
    )


# ===========================================================================
# 2. test_nn_mlp_seeded_runs_produce_identical_state_dicts  (AC-2 / NFR-1 CPU)
# ===========================================================================


def test_nn_mlp_seeded_runs_produce_identical_state_dicts() -> None:
    """Guards AC-2 / NFR-1 CPU: two seeded fits produce byte-identical params.

    Uses ``torch.equal`` on every parameter tensor (parameters *and*
    registered buffers) — NFR-1 explicitly requires bit-identity on CPU
    for the same seed.  ``predict`` output is additionally byte-compared.

    Plan clause: T2 §Task T2 named test / AC-2 / NFR-1 CPU half.
    """
    import torch

    features, target = _tiny_fixture()

    model_a = NnMlpModel(_cpu_config())
    model_a.fit(features, target, seed=42)

    model_b = NnMlpModel(_cpu_config())
    model_b.fit(features, target, seed=42)

    # ``model._module`` is the fitted torch.nn.Module; compare every
    # state_dict entry (both parameters and buffers — scaler buffers
    # are fitted from data so they must also match bit-for-bit).
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
    np.testing.assert_array_equal(
        pred_a.to_numpy(),
        pred_b.to_numpy(),
        err_msg="predict outputs must be byte-identical across two seeded CPU fits.",
    )


# ===========================================================================
# 3. test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance  (NFR-1 GPU)
# ===========================================================================


@pytest.mark.gpu
def test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance() -> None:
    """Guards NFR-1 GPU half: seeded fits close-match under atol/rtol.

    The ``@pytest.mark.gpu`` marker puts this test behind ``-m gpu`` in
    the pyproject addopts selector; inside the test we still check
    ``torch.cuda.is_available()`` and skip at runtime for clarity — the
    marker gates test *selection*, the guard here gates *execution*.

    Tolerances (``atol=1e-5, rtol=1e-4``) come from plan NFR-1: cuBLAS /
    cuDNN nondeterminism on reductions is real even with
    ``cudnn.deterministic=True``; bit-identity is not achievable.

    Plan clause: T2 §Task T2 named test / NFR-1 GPU half.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host.")

    features, target = _tiny_fixture()
    cfg = _cpu_config(device="cuda")

    model_a = NnMlpModel(cfg)
    model_a.fit(features, target, seed=7)

    model_b = NnMlpModel(cfg)
    model_b.fit(features, target, seed=7)

    pred_a = torch.from_numpy(model_a.predict(features).to_numpy())
    pred_b = torch.from_numpy(model_b.predict(features).to_numpy())
    assert torch.allclose(pred_a, pred_b, atol=1e-5, rtol=1e-4), (
        "Two seeded fits on CUDA must close-match within atol=1e-5 rtol=1e-4 (NFR-1 GPU half)."
    )


# ===========================================================================
# 4. test_nn_mlp_different_seeds_produce_different_state_dicts
# ===========================================================================


def test_nn_mlp_different_seeds_produce_different_state_dicts() -> None:
    """Guards the seed-diversity half of the reproducibility story.

    If two *different* seeds produced identical params, the four-stream
    seed helper would be broken (or the network initialisation wouldn't
    depend on the seed at all).  The assertion is that at least one
    parameter tensor differs — weaker than "every parameter differs"
    because the scaler buffers (``feature_mean`` / ``feature_std`` /
    ``target_mean`` / ``target_std``) are fitted from data and therefore
    are seed-invariant by design.

    Plan clause: T2 §Task T2 named test.
    """
    import torch

    features, target = _tiny_fixture()

    model_a = NnMlpModel(_cpu_config())
    model_a.fit(features, target, seed=1)

    model_b = NnMlpModel(_cpu_config())
    model_b.fit(features, target, seed=2)

    assert model_a._module is not None and model_b._module is not None
    sd_a = model_a._module.state_dict()
    sd_b = model_b._module.state_dict()

    # At least one *parameter* (not buffer) tensor must differ.
    param_names = {name for name, _ in model_a._module.named_parameters()}
    assert param_names, "The MLP module must expose at least one named parameter."
    any_diff = False
    for key in sd_a:
        if key not in param_names:
            continue  # buffers are seed-invariant
        if not torch.equal(sd_a[key], sd_b[key]):
            any_diff = True
            break
    assert any_diff, (
        "At least one parameter tensor must differ between seed=1 and seed=2 fits "
        "(otherwise the seed is not wired into initialisation / training)."
    )


# ===========================================================================
# 5. test_nn_mlp_fit_populates_loss_history_per_epoch  (AC-3)
# ===========================================================================


def test_nn_mlp_fit_populates_loss_history_per_epoch() -> None:
    """Guards AC-3: ``loss_history_`` grows by one dict per epoch.

    Each entry is a dict with keys ``{"epoch", "train_loss", "val_loss"}``
    all of type float.  The ``epoch`` value is 1-indexed and monotonic.

    Plan clause: T2 §Task T2 named test / AC-3 / D6.
    """
    features, target = _tiny_fixture()
    cfg = _cpu_config(max_epochs=4, patience=10)  # patience > max_epochs disables early stop
    model = NnMlpModel(cfg)
    model.fit(features, target, seed=0)

    history = model.loss_history_
    assert isinstance(history, list), f"loss_history_ must be a list; got {type(history).__name__}."
    assert len(history) == cfg.max_epochs, (
        f"loss_history_ length must equal max_epochs={cfg.max_epochs} "
        f"when patience=10 prevents early stop; got {len(history)}."
    )
    for i, entry in enumerate(history, start=1):
        assert set(entry.keys()) == {"epoch", "train_loss", "val_loss"}, (
            f"loss_history_[{i - 1}] keys must be {{'epoch','train_loss','val_loss'}}; "
            f"got {set(entry.keys())!r}."
        )
        assert entry["epoch"] == float(i), (
            f"loss_history_[{i - 1}]['epoch'] must be {float(i)}; got {entry['epoch']!r}."
        )
        for key in ("train_loss", "val_loss"):
            assert isinstance(entry[key], float), (
                f"loss_history_[{i - 1}][{key!r}] must be float; got {type(entry[key]).__name__}."
            )
            assert math.isfinite(entry[key]), (
                f"loss_history_[{i - 1}][{key!r}] must be finite; got {entry[key]!r}."
            )


# ===========================================================================
# 6. test_nn_mlp_fit_invokes_epoch_callback_when_provided  (AC-3)
# ===========================================================================


def test_nn_mlp_fit_invokes_epoch_callback_when_provided() -> None:
    """Guards AC-3: ``epoch_callback`` is invoked once per appended history entry.

    The callback receives the *same* dict that is appended to
    ``loss_history_`` — the live-plot seam in the notebook relies on
    that identity so the plot updates in lock-step with the stored
    history.

    Plan clause: T2 §Task T2 named test / AC-3 / X4 re-scoped.
    """
    features, target = _tiny_fixture()
    cfg = _cpu_config(max_epochs=3, patience=10)

    received: list[dict[str, float]] = []

    def cb(entry: dict[str, float]) -> None:
        received.append(entry)

    model = NnMlpModel(cfg)
    model.fit(features, target, seed=0, epoch_callback=cb)

    assert len(received) == cfg.max_epochs, (
        f"epoch_callback must be invoked once per epoch; "
        f"got {len(received)} calls for max_epochs={cfg.max_epochs}."
    )
    for rx, stored in zip(received, model.loss_history_, strict=True):
        assert rx == stored, (
            "Each dict passed to epoch_callback must equal the matching "
            "loss_history_ entry (the live-plot seam contract)."
        )
        assert set(rx.keys()) == {"epoch", "train_loss", "val_loss"}


# ===========================================================================
# 7. test_nn_mlp_early_stopping_terminates_before_max_epochs_on_plateau
# ===========================================================================


def test_nn_mlp_early_stopping_terminates_before_max_epochs_on_plateau() -> None:
    """Guards D9 early-stopping: patience-based exit before ``max_epochs``.

    Constructs a config with a high ``max_epochs`` but a tiny
    ``patience`` so the run stops early on any loss plateau.  A constant
    target with tiny noise ensures the model reaches its plateau inside
    a few epochs.  The assertion is ``len(loss_history_) < max_epochs``.

    Plan clause: T2 §Task T2 named test / D9 early-stopping.
    """
    rng = np.random.default_rng(0)
    n = 40
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal(size=(n, 2)),
        columns=["f0", "f1"],
        index=index,
    )
    # Near-constant target so the val loss plateaus almost immediately.
    target = pd.Series(
        5.0 + 1e-4 * rng.standard_normal(size=n),
        index=index,
        name="nd_mw",
    )

    cfg = _cpu_config(
        hidden_sizes=[4],
        max_epochs=200,
        patience=2,
        learning_rate=1e-1,
    )
    model = NnMlpModel(cfg)
    model.fit(features, target, seed=0)

    assert len(model.loss_history_) < cfg.max_epochs, (
        f"Early stopping must terminate the fit before max_epochs={cfg.max_epochs}; "
        f"ran for {len(model.loss_history_)} epochs (patience={cfg.patience}). "
        "If this fails, patience tracking is broken or the plateau detector is not "
        "wired to val_loss."
    )
    assert model._best_epoch is not None
    assert 1 <= model._best_epoch <= len(model.loss_history_), (
        f"_best_epoch={model._best_epoch} must be within [1, {len(model.loss_history_)}]."
    )


# ===========================================================================
# 8. test_nn_mlp_fit_uses_cold_start_per_fold_when_called_repeatedly  (D8)
# ===========================================================================


def test_nn_mlp_fit_uses_cold_start_per_fold_when_called_repeatedly() -> None:
    """Guards D8: a second ``fit()`` cold-starts — no carry-over from the first.

    Calls ``fit`` twice on the *same* ``(features, target)`` with the
    *same* seed; the post-second-fit ``loss_history_`` must equal the
    post-first-fit ``loss_history_`` exactly.  If ``fit`` were
    warm-starting from the previous params, the second run's initial
    losses would be much lower than the first run's (the model would
    already be near-optimal).

    Plan clause: T2 §Task T2 named test / D8 cold-start-per-fold.
    """
    features, target = _tiny_fixture()
    cfg = _cpu_config(max_epochs=4, patience=10)

    model = NnMlpModel(cfg)
    model.fit(features, target, seed=11)
    first_history = [dict(e) for e in model.loss_history_]

    model.fit(features, target, seed=11)
    second_history = [dict(e) for e in model.loss_history_]

    assert first_history == second_history, (
        "Two fit() calls with the same seed must produce identical loss_history_ "
        "entries — the second fit must cold-start, not warm-start (D8).\n"
        f"first:  {first_history!r}\n"
        f"second: {second_history!r}"
    )


# ===========================================================================
# 9. test_nn_mlp_select_device_auto_prefers_cuda_then_mps_then_cpu  (D11)
# ===========================================================================


@pytest.mark.parametrize(
    "cuda_available,mps_available,expected_type",
    [
        (True, True, "cuda"),  # CUDA wins over MPS
        (False, True, "mps"),  # MPS wins over CPU
        (False, False, "cpu"),  # CPU fallback
    ],
)
def test_nn_mlp_select_device_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_type: str,
) -> None:
    """Guards D11: ``_select_device("auto")`` precedence CUDA > MPS > CPU.

    Monkeypatches ``torch.cuda.is_available`` and
    ``torch.backends.mps.is_available`` to simulate each hardware
    combination without requiring the hardware to be present.

    Plan clause: T2 §Task T2 named test / D11.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    # ``torch.backends.mps`` may not exist on very old torch; we assume
    # torch >= 2.7 per plan D1.  Monkeypatch always-present attribute.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    device = _select_device("auto")
    assert device.type == expected_type, (
        f"_select_device('auto') with cuda={cuda_available}, mps={mps_available} "
        f"must resolve to device.type={expected_type!r}; got {device.type!r} "
        "(D11 precedence CUDA > MPS > CPU)."
    )


# ===========================================================================
# 10. test_nn_mlp_select_device_respects_explicit_pin  (D11)
# ===========================================================================


def test_nn_mlp_select_device_respects_explicit_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards D11: explicit pins are honoured; unknown values raise ValueError.

    Two sub-scenarios in one test (matches the plan's "+ invalid ValueError"
    bullet):

    1. ``_select_device("cpu")`` on a CUDA-capable host returns a CPU
       device — the auto-selector must not override an explicit pin.
    2. ``_select_device("tpu")`` (or any unknown string) raises
       :class:`ValueError` — a silent fallback would mask config
       mistakes.

    Plan clause: T2 §Task T2 named test / D11.
    """
    import torch

    # Simulate a CUDA-capable host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = _select_device("cpu")
    assert device.type == "cpu", (
        f"_select_device('cpu') on a CUDA-capable host must return device.type='cpu' "
        f"(explicit pin honoured); got {device.type!r}."
    )

    # Unknown value → ValueError.
    with pytest.raises(ValueError, match=r"NnMlpConfig\.device"):
        _select_device("tpu")

    # Sanity: _ALLOWED_DEVICES is the single source of truth.
    assert "tpu" not in _ALLOWED_DEVICES
    for allowed in _ALLOWED_DEVICES:
        assert isinstance(allowed, str), f"_ALLOWED_DEVICES entries must be str; got {allowed!r}."


# ===========================================================================
# Bonus — predict-before-fit RuntimeError (models CLAUDE.md protocol contract)
# Not strictly a T2 named test, but the Stage 4 protocol convention that
# ``predict()`` before ``fit()`` raises ``RuntimeError`` is table stakes.
# ===========================================================================


def test_nn_mlp_predict_before_fit_raises_runtime_error() -> None:
    """Guards the Stage 4 ``predict-before-fit`` protocol convention.

    Plan clause: models/CLAUDE.md "Predict-before-fit" section.
    """
    features, _ = _tiny_fixture()
    model = NnMlpModel(_cpu_config())
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(features)
