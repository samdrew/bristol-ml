"""Spec-derived tests for the Stage 11 ``NnTemporalModel`` save / load path — Task T5.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T5 (lines 515-519): the 4 T5
  named tests (save/load round-trip including seq_len and warmup_features,
  FileNotFound on missing artefact, single-file structural guard, and the
  module-impl pickleability regression pattern inherited from Stage 10).
- ``docs/plans/active/11-complex-nn.md`` §4 AC-4 (save / load on a registry
  path round-trips the state dict + every scalar envelope field + seq_len).
- Plan R7 (explicit seq_len round-trip guard — the envelope carries a top-level
  ``seq_len`` field alongside ``config_dump["seq_len"]`` so corruption is
  detected without Pydantic re-validation alone).
- ``src/bristol_ml/models/nn/CLAUDE.md`` Stage 10 Gotcha 1 (the
  ``sys.modules`` install + ``__module__`` / ``__qualname__`` patch for
  pickleable lazy-built module classes).

No production code is modified here.  If any test below fails the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bristol_ml.models.nn.temporal import (
    NnTemporalModel,
    _build_temporal_module_class,
)
from conf._schemas import NnTemporalConfig

# ---------------------------------------------------------------------------
# Shared tiny fixture + CPU-pinned config — self-contained copies that mirror
# ``test_nn_temporal_fit_predict.py`` so T5 does not import from T4 (each
# test file must be independently runnable).  The same pattern as Stage 10
# ``test_nn_mlp_save_load.py`` defines its own ``_tiny_fixture`` /
# ``_cpu_config`` rather than importing from the sibling T2/T3 file.
# ---------------------------------------------------------------------------


def _tiny_temporal_fixture(
    n: int = 500, n_features: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair for TCN save/load tests.

    Mirrors the fixture in ``test_nn_temporal_fit_predict.py`` so a fit under
    T4 and a fit under T5 produce the same weights (makes the round-trip
    test's reference ``predict()`` output reproducible).

    The index is UTC-aware hourly to match the harness convention.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, n_features)).astype(np.float64)
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + 0.05 * rng.standard_normal(size=n)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


def _cpu_temporal_config(**overrides: object) -> NnTemporalConfig:
    """Return a CPU-pinned, short-budget ``NnTemporalConfig``.

    Matches the config helper in ``test_nn_temporal_fit_predict.py`` so the
    same trained weights appear in both test files when the same seed is used.
    ``weight_norm=False`` keeps state_dict key names simple (no extra
    ``weight_g`` / ``weight_v`` keys) and avoids edge-cases in the strict
    ``load_state_dict`` round-trip check.
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
# 1. test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict  (AC-4)
# ===========================================================================


def test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict(
    tmp_path: Path,
) -> None:
    """Guards AC-4 at the unit level (plan §Task T5 line 516 / plan R7).

    After ``save()`` + ``load()``:

    (a) Every tensor in ``loaded._module.state_dict()`` matches the original
        under ``torch.equal`` (bit-identical round-trip via ``torch.save`` /
        ``torch.load``; scaler buffers ride inside ``state_dict`` so the
        inverse-normalisation step is also exact).
    (b) ``loaded._feature_columns == model._feature_columns``.
    (c) ``loaded._config.seq_len == model._config.seq_len`` (plan R7
        explicit round-trip guard).
    (d) ``loaded.loss_history_ == model.loss_history_``.
    (e) ``loaded.predict(features_test)`` equals ``model.predict(features_test)``
        under ``torch.equal`` (CPU bit-identity, plan NFR-1).
    (f) ``loaded._warmup_features`` DataFrame values equal
        ``model._warmup_features`` values under
        ``pd.testing.assert_frame_equal``.

    Plan clause: T5 §Task T5 / AC-4 / plan R7.
    """
    features, target = _tiny_temporal_fixture()
    model = NnTemporalModel(_cpu_temporal_config())
    model.fit(features, target, seed=17)

    # Out-of-sample test slice.
    features_test = features.iloc[-50:]
    pred_before = model.predict(features_test)

    artefact_path = tmp_path / "artefact" / "model.joblib"
    model.save(artefact_path)

    assert artefact_path.is_file(), f"save() did not create {artefact_path!s}."

    loaded = NnTemporalModel.load(artefact_path)

    # (a) state_dict tensors — bit-identical under torch.equal.
    assert model._module is not None, "Original model._module must not be None after fit()."
    assert loaded._module is not None, "Loaded model._module must not be None after load()."
    sd_orig = model._module.state_dict()
    sd_load = loaded._module.state_dict()
    assert sd_orig.keys() == sd_load.keys(), (
        f"state_dict key sets must match after round-trip; "
        f"original keys: {set(sd_orig)}, loaded keys: {set(sd_load)}."
    )
    for key in sd_orig:
        assert torch.equal(sd_orig[key], sd_load[key]), (
            f"state_dict[{key!r}] must round-trip bit-exactly via "
            "torch.save/torch.load (AC-4 / plan D5)."
        )

    # (b) feature_columns round-trip.
    assert loaded._feature_columns == model._feature_columns, (
        f"_feature_columns must round-trip; "
        f"original {model._feature_columns!r} vs loaded {loaded._feature_columns!r}."
    )

    # (c) seq_len round-trip — the explicit plan R7 guard.
    assert loaded._config.seq_len == model._config.seq_len, (
        f"_config.seq_len must round-trip; "
        f"original {model._config.seq_len} vs loaded {loaded._config.seq_len}."
    )

    # (d) loss_history_ round-trip.
    assert loaded.loss_history_ == model.loss_history_, (
        "loss_history_ must round-trip byte-exact (plan D5 envelope / AC-3)."
    )

    # (e) predict output — CPU bit-identity under torch.equal.
    pred_after = loaded.predict(features_test)
    assert torch.equal(
        torch.from_numpy(pred_before.to_numpy()),
        torch.from_numpy(pred_after.to_numpy()),
    ), (
        "NnTemporalModel.save/load round-trip: predict() output must match "
        "the pre-save output bit-exactly on CPU (state_dict + scaler buffers + "
        "warmup_features must all round-trip cleanly — AC-4)."
    )
    # Index must also round-trip.
    assert pred_after.index.equals(pred_before.index), (
        "Loaded model predict() output must carry the same index as the "
        "pre-save output (Model-protocol AC from Stage 4)."
    )

    # (f) warmup_features DataFrame round-trip.
    assert model._warmup_features is not None, (
        "model._warmup_features must be populated after fit()."
    )
    assert loaded._warmup_features is not None, (
        "loaded._warmup_features must be populated after load()."
    )
    pd.testing.assert_frame_equal(
        loaded._warmup_features,
        model._warmup_features,
        check_exact=True,
        obj="_warmup_features",
    )


# ===========================================================================
# 2. test_nn_temporal_load_raises_file_not_found_for_missing_artefact
# ===========================================================================


def test_nn_temporal_load_raises_file_not_found_for_missing_artefact(
    tmp_path: Path,
) -> None:
    """Guards the Stage 4 ``load`` convention (plan §Task T5 line 517).

    ``load()`` on a non-existent path must raise :class:`FileNotFoundError`
    (propagated from :mod:`joblib`) rather than silently returning ``None``
    or a half-initialised model.

    Plan clause: T5 §Task T5.
    """
    missing = tmp_path / "does_not_exist.joblib"

    with pytest.raises(FileNotFoundError):
        NnTemporalModel.load(missing)


# ===========================================================================
# 3. test_nn_temporal_save_writes_single_joblib_file_at_given_path
# ===========================================================================


def test_nn_temporal_save_writes_single_joblib_file_at_given_path(
    tmp_path: Path,
) -> None:
    """Guards plan D5 revised (single-envelope layout) — Stage 10 T3 structural
    guard inherited (plan §Task T5 line 518).

    ``save(path)`` must create exactly one file at ``path`` and no
    siblings.  Regressing to a two-file layout (e.g. ``model.pt`` +
    ``hyperparameters.json``) would break the Stage 9 registry's single-file
    ``artefact/model.joblib`` contract — structurally caught here.

    Plan clause: T5 §Task T5 / D5 single-joblib envelope.
    """
    features, target = _tiny_temporal_fixture()
    model = NnTemporalModel(_cpu_temporal_config())
    model.fit(features, target, seed=0)

    artefact_dir = tmp_path / "artefact"
    artefact_path = artefact_dir / "model.joblib"
    model.save(artefact_path)

    # Exactly one file at the configured path.
    assert artefact_path.is_file(), f"save() did not create {artefact_path!s}."
    # ``save_joblib`` is atomic (tmp + ``os.replace``), so no ``.tmp`` sibling
    # should survive.  The directory must contain exactly one entry.
    siblings = sorted(p.name for p in artefact_dir.iterdir())
    assert siblings == ["model.joblib"], (
        f"NnTemporalModel.save created unexpected sibling files: {siblings!r}. "
        "Plan D5 (single-joblib envelope) mandates exactly one file at the "
        "given path — no ``model.pt`` / ``hyperparameters.json`` / ``.tmp``."
    )
    # Belt-and-braces: check the three most common pre-D5 regression artefacts
    # individually so the failure message names the culprit.
    for banned in ("model.pt", "hyperparameters.json", "model.joblib.tmp"):
        assert not (artefact_dir / banned).exists(), (
            f"NnTemporalModel.save created banned sibling file {banned!r}. "
            "Plan D5 (single-joblib envelope): single file only."
        )


# ===========================================================================
# 4. test_nn_temporal_module_impl_is_pickleable
# ===========================================================================


def test_nn_temporal_module_impl_is_pickleable() -> None:
    """Guards the ``_NnTemporalModuleImpl`` lazy-class pickleability contract —
    Stage 10 Phase 3 regression pattern inherited (plan §Task T5 line 519).

    ``_build_temporal_module_class`` defines the :class:`torch.nn.Module`
    subclass (``_NnTemporalModuleImpl``) and its inner residual-block class
    (``_TemporalBlockImpl``) inside a function body so ``torch`` stays
    lazily imported.  For :func:`pickle` / :func:`torch.save` to round-trip
    an *instance* of either class, each class must be resolvable via
    ``getattr(sys.modules[cls.__module__], cls.__qualname__)`` — which means
    the factory must (a) patch ``__module__`` / ``__qualname__`` and (b)
    install both classes as attributes on
    :mod:`bristol_ml.models.nn.temporal`.

    Without the ``sys.modules[__name__]._NnTemporalModuleImpl = ...`` install,
    ``pickle.dumps(instance)`` raises ``AttributeError: Can't get attribute
    '_NnTemporalModuleImpl' on <module ...>``.  The current save path avoids
    serialising the module object directly (it writes only ``state_dict_bytes``
    + scalars), but any future ``copy.deepcopy`` / ``joblib.dump`` on a
    fitted ``NnTemporalModel`` instance would trip on it.

    Also exercises ``_TemporalBlockImpl`` in isolation — the first block
    extracted from the built module must also survive a ``pickle`` round-trip.

    Stage 8 precedent: ``test_parametric_fn_is_pickleable``.
    Stage 10 precedent: ``test_nn_mlp_module_impl_is_pickleable``.
    Plan clause: T5 §Task T5 / Stage 10 Phase 3 review (CR-1 / CR-2 /
    Nit-2 inherited).
    """
    # Build the module class lazily (also triggers the sys.modules install
    # for both _NnTemporalModuleImpl and _TemporalBlockImpl).
    module_cls = _build_temporal_module_class()

    # --- (a) _NnTemporalModuleImpl class is resolvable at its patched path ---
    assert module_cls.__module__ == "bristol_ml.models.nn.temporal", (
        f"_NnTemporalModuleImpl.__module__ must be "
        f"'bristol_ml.models.nn.temporal'; got {module_cls.__module__!r}."
    )
    assert module_cls.__qualname__ == "_NnTemporalModuleImpl", (
        f"_NnTemporalModuleImpl.__qualname__ must be '_NnTemporalModuleImpl'; "
        f"got {module_cls.__qualname__!r}."
    )

    # Instantiate a small TCN so the pickle payload contains real tensors
    # (buffers + conv weights) rather than just the default placeholder zeros.
    module = module_cls(
        input_dim=3,
        seq_len=24,
        num_blocks=2,
        channels=8,
        kernel_size=3,
        dropout=0.0,
        weight_norm=False,
    )

    # --- (b) _NnTemporalModuleImpl instance pickle round-trip ---------------
    payload = pickle.dumps(module)
    restored = pickle.loads(payload)

    assert type(restored) is module_cls, (
        f"pickle round-trip must restore the same class; got {type(restored).__name__}."
    )
    # The four scaler buffers must survive the round-trip (they are
    # ``register_buffer`` entries and therefore part of ``state_dict``).
    for buf_name in ("feature_mean", "feature_std", "target_mean", "target_std"):
        assert buf_name in dict(restored.named_buffers()), (
            f"pickle round-trip lost buffer {buf_name!r} — the scaler buffers "
            "must ride inside state_dict (plan D5 / Stage 10 D4 inheritance)."
        )
    # state_dict keys must survive (full round-trip check).
    sd_orig = module.state_dict()
    sd_rest = restored.state_dict()
    assert sd_orig.keys() == sd_rest.keys(), (
        f"pickle round-trip must preserve state_dict keys; "
        f"original: {set(sd_orig)}, restored: {set(sd_rest)}."
    )

    # --- (c) _TemporalBlockImpl instance pickle round-trip ------------------
    # Extract the first block (dilation=1); it is a _TemporalBlockImpl.
    first_block = next(iter(module.blocks))
    import bristol_ml.models.nn.temporal as _temporal_mod

    block_cls = type(first_block)
    assert hasattr(_temporal_mod, "_TemporalBlockImpl"), (
        "_TemporalBlockImpl must be installed on the temporal module namespace "
        "so pickle's getattr lookup resolves (Stage 10 Gotcha 1 inherited)."
    )
    assert block_cls.__module__ == "bristol_ml.models.nn.temporal", (
        f"_TemporalBlockImpl.__module__ must be "
        f"'bristol_ml.models.nn.temporal'; got {block_cls.__module__!r}."
    )
    assert block_cls.__qualname__ == "_TemporalBlockImpl", (
        f"_TemporalBlockImpl.__qualname__ must be '_TemporalBlockImpl'; "
        f"got {block_cls.__qualname__!r}."
    )

    block_payload = pickle.dumps(first_block)
    restored_block = pickle.loads(block_payload)

    # The restored block must expose the same state_dict keys as the original.
    block_sd_orig = first_block.state_dict()
    block_sd_rest = restored_block.state_dict()
    assert block_sd_orig.keys() == block_sd_rest.keys(), (
        f"_TemporalBlockImpl pickle round-trip must preserve state_dict keys; "
        f"original: {set(block_sd_orig)}, restored: {set(block_sd_rest)}."
    )
