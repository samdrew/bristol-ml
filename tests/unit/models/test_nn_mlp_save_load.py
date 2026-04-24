"""Spec-derived tests for the Stage 10 ``NnMlpModel`` save / load path — Task T3.

Every test is derived from:

- ``docs/plans/active/10-simple-nn.md`` §Task T3 (the 3 T3 named tests
  — save/load round-trip, FileNotFound on missing artefact, and the
  single-file structural guard).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-4 (save / load on a
  registry path round-trips the state dict + every scalar envelope field)
  and plan D5 revised (single joblib envelope, no sibling files).
- ``docs/plans/active/10-simple-nn.md`` §R2 / §R3 (Pydantic re-validation
  catches schema drift; ``load_state_dict(strict=True)`` refuses a
  module with missing or extra keys).

No production code is modified here.  If any test below fails the
failure points at a deviation from the spec — do not weaken the test;
surface the failure to the implementer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bristol_ml.models.nn.mlp import NnMlpModel
from conf._schemas import NnMlpConfig

# ---------------------------------------------------------------------------
# Shared tiny fixture + CPU-pinned config — mirrors test_nn_mlp_fit_predict.py
# so T3 remains self-contained but the behavioural contract is identical.
# ---------------------------------------------------------------------------


def _tiny_fixture(
    n: int = 60, n_features: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair.

    Matches the fixture in ``test_nn_mlp_fit_predict.py`` so a fit under
    T2 and a fit under T3 produce the same weights (makes the
    round-trip test's reference ``predict()`` output reproducible).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(size=n)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


def _cpu_config(**overrides: object) -> NnMlpConfig:
    """Return a CPU-pinned, short-budget ``NnMlpConfig``."""
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


def _fit_tiny_model(tmp_path: Path) -> tuple[NnMlpModel, pd.DataFrame, pd.Series]:
    """Fit a reproducible NnMlpModel on the tiny fixture.

    The seed is pinned so ``predict(features)`` is deterministic; callers
    use this to check byte-equality of the round-tripped predictions.
    """
    features, target = _tiny_fixture()
    model = NnMlpModel(_cpu_config())
    model.fit(features, target, seed=17)
    return model, features, target


# ===========================================================================
# 1. test_nn_mlp_save_and_load_round_trips_state_dict_and_hyperparameters
# ===========================================================================


def test_nn_mlp_save_and_load_round_trips_state_dict_and_hyperparameters(
    tmp_path: Path,
) -> None:
    """Guards AC-4 at the unit level.

    After ``save()`` + ``load()``:
    - ``predict(X)`` must match the pre-save predictions within
      ``atol=1e-10`` (weights round-trip bit-exact via ``torch.save`` /
      ``torch.load``; normalisation buffers are part of ``state_dict``
      so the inverse-normalisation step is also exact).
    - Every scalar envelope field round-trips byte-exact:
      ``feature_columns``, ``seed_used``, ``best_epoch``, ``loss_history_``,
      ``fit_utc``, ``_device_resolved``, and the ``NnMlpConfig`` itself.

    Plan clause: T3 §Task T3 / AC-4.
    """
    model_before, features, _target = _fit_tiny_model(tmp_path)
    pred_before = model_before.predict(features)

    artefact_path = tmp_path / "artefact" / "model.joblib"
    model_before.save(artefact_path)

    assert artefact_path.is_file(), f"save() did not create {artefact_path!s}."

    model_after = NnMlpModel.load(artefact_path)

    # Predict round-trip — the strongest behavioural check (AC-4 head).
    pred_after = model_after.predict(features)
    assert torch.allclose(
        torch.from_numpy(pred_before.to_numpy()),
        torch.from_numpy(pred_after.to_numpy()),
        atol=1e-10,
        rtol=0.0,
    ), (
        "NnMlpModel.save/load round-trip: predict() output must match pre-save "
        "within atol=1e-10 (state_dict round-trip including scaler buffers)."
    )
    # Index and dtype must also round-trip (Model protocol AC from Stage 4).
    assert pred_after.index.equals(pred_before.index)
    assert pred_after.name == pred_before.name

    # Scalar envelope fields — byte-exact.
    assert tuple(model_after._feature_columns) == tuple(model_before._feature_columns), (
        "feature_columns must round-trip byte-exact (plan D5 envelope)."
    )
    assert model_after._seed_used == model_before._seed_used, (
        "seed_used must round-trip byte-exact (plan D5 envelope)."
    )
    assert model_after._best_epoch == model_before._best_epoch, (
        "best_epoch must round-trip byte-exact (plan D5 envelope)."
    )
    assert model_after._fit_utc == model_before._fit_utc, (
        "fit_utc must round-trip byte-exact (plan D5 envelope)."
    )
    assert model_after.loss_history_ == model_before.loss_history_, (
        "loss_history_ must round-trip byte-exact (plan D5 envelope / AC-3)."
    )
    assert model_after._device_resolved == model_before._device_resolved, (
        "device_resolved must round-trip byte-exact (plan D5 envelope)."
    )

    # NnMlpConfig must be equal (Pydantic equality on frozen models).
    assert model_after._config == model_before._config, (
        "NnMlpConfig must round-trip via Pydantic re-validation (plan R2)."
    )

    # Metadata (the public provenance record) is a cross-check that
    # hyperparameters survive.
    assert dict(model_after.metadata.hyperparameters) == dict(model_before.metadata.hyperparameters)


# ===========================================================================
# 2. test_nn_mlp_load_raises_file_not_found_for_missing_artefact
# ===========================================================================


def test_nn_mlp_load_raises_file_not_found_for_missing_artefact(tmp_path: Path) -> None:
    """Guards the Stage 4 ``load`` convention.

    ``load()`` on a non-existent path must raise :class:`FileNotFoundError`
    (propagated from :func:`joblib.load`) rather than silently returning
    ``None`` or a half-initialised model.

    Plan clause: T3 §Task T3.
    """
    missing = tmp_path / "does_not_exist" / "model.joblib"

    with pytest.raises(FileNotFoundError):
        NnMlpModel.load(missing)


# ===========================================================================
# 3. test_nn_mlp_save_writes_single_joblib_file_at_given_path
# ===========================================================================


def test_nn_mlp_save_writes_single_joblib_file_at_given_path(tmp_path: Path) -> None:
    """Guards plan D5 revised (single-envelope layout).

    ``save(path)`` must create exactly one file at ``path`` and no
    siblings.  Regressing to the pre-D5-revision two-file layout
    (``model.pt`` + ``hyperparameters.json``) would break the Stage 9
    registry's single-file ``artefact/model.joblib`` contract —
    structurally caught here.

    Plan clause: T3 §Task T3.
    """
    model, _features, _target = _fit_tiny_model(tmp_path)

    artefact_dir = tmp_path / "artefact"
    artefact_path = artefact_dir / "model.joblib"
    model.save(artefact_path)

    # Exactly one file at the configured path.
    assert artefact_path.is_file(), f"save() did not create {artefact_path!s}."

    # No ``.tmp`` sibling left behind — save_joblib is atomic, the tmp
    # is renamed via os.replace, so the post-save directory must be clean.
    siblings = sorted(p.name for p in artefact_dir.iterdir())
    assert siblings == ["model.joblib"], (
        f"NnMlpModel.save created unexpected sibling files: {siblings!r}. "
        f"Plan D5 (revised) mandates the single-envelope layout."
    )

    # Specifically, neither the pre-D5 two-file layout nor a stray
    # ``.tmp`` may exist.
    for banned in ("model.pt", "hyperparameters.json", "model.joblib.tmp"):
        assert not (artefact_dir / banned).exists(), (
            f"NnMlpModel.save created banned sibling {banned!r}. "
            "Plan D5 (revised): single joblib envelope only."
        )


# ===========================================================================
# 4. test_nn_mlp_save_before_fit_raises_runtime_error  (guard derived from
#    Stage 4 protocol: save-before-fit is undefined behaviour)
# ===========================================================================


def test_nn_mlp_save_before_fit_raises_runtime_error(tmp_path: Path) -> None:
    """Guards the Stage 4 :class:`Model` protocol invariant.

    ``save()`` without a prior ``fit()`` has no state_dict to serialise;
    the implementation must raise ``RuntimeError`` rather than writing
    an envelope that would deserialise to a half-initialised model.

    Plan clause: Stage 4 protocol semantics (``models/CLAUDE.md``
    "predict-before-fit" contract — save extends the same invariant).
    """
    model = NnMlpModel(_cpu_config())
    path = tmp_path / "artefact" / "model.joblib"

    with pytest.raises(RuntimeError, match=r"fit\(\) to have been called"):
        model.save(path)


# ===========================================================================
# 5. test_nn_mlp_load_rejects_state_dict_with_missing_buffer  (R3)
# ===========================================================================


def test_nn_mlp_load_rejects_state_dict_with_missing_buffer(tmp_path: Path) -> None:
    """Guards plan R3.

    ``load_state_dict(strict=True)`` must surface a missing scaler
    buffer as a loud error, not silently initialise it to zero and
    skew predictions.  We construct an envelope whose ``state_dict_bytes``
    omits the ``target_std`` buffer, then assert ``load()`` rejects it.

    Plan clause: T3 §R3 / R2 (schema drift must fail loudly).
    """
    import io as _io

    from bristol_ml.models.io import load_joblib, save_joblib

    model, _features, _target = _fit_tiny_model(tmp_path)
    artefact_path = tmp_path / "artefact" / "model.joblib"
    model.save(artefact_path)

    envelope = load_joblib(artefact_path)
    original_state_dict = torch.load(
        _io.BytesIO(envelope["state_dict_bytes"]),
        weights_only=True,
        map_location="cpu",
    )
    assert "target_std" in original_state_dict, (
        "Pre-check: state_dict must carry target_std (plan D4 / R3)."
    )

    # Drop the buffer and re-save the envelope.
    tampered = {k: v for k, v in original_state_dict.items() if k != "target_std"}
    buf = _io.BytesIO()
    torch.save(tampered, buf)
    envelope["state_dict_bytes"] = buf.getvalue()
    save_joblib(envelope, artefact_path)

    with pytest.raises(RuntimeError, match=r"(?i)missing|target_std"):
        NnMlpModel.load(artefact_path)
