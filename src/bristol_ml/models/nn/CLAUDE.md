# `bristol_ml.models.nn` — module guide

This module is the **neural-network sub-layer** of the models layer.
It lands at Stage 10 with one class (`NnMlpModel` — a simple MLP); it
exists to name the conventions that every subsequent NN-family class
(Stage 11's temporal architecture) will inherit.  Everything here
conforms to the Stage 4 `Model` protocol re-exported from
`bristol_ml.models`; nothing here changes that protocol.

Read the sub-layer contract in
[`docs/architecture/layers/models-nn.md`](../../../../docs/architecture/layers/models-nn.md)
before extending this module; the file you are reading documents the
concrete Stage 10 surface.  The parent layer guide at
[`src/bristol_ml/models/CLAUDE.md`](../CLAUDE.md) remains authoritative
for the protocol, `ModelMetadata`, and the joblib IO helpers that every
model family reuses.

## Current surface (Stage 10)

- `bristol_ml.models.nn.NnMlpModel` — small feed-forward MLP (1 hidden
  layer × 128 units, ReLU default).  Conforms to the Stage 4 `Model`
  protocol (`fit`, `predict`, `save`, `load`, `metadata`).  Exposes one
  extra public attribute, `loss_history_: list[dict[str, float]]`,
  populated after `fit()` and consumed by
  `bristol_ml.evaluation.plots.loss_curve`.
- `bristol_ml.models.nn.mlp._select_device(preference) -> torch.device`
  — auto-select helper.  Resolves `"auto"` in order CUDA → MPS → CPU;
  explicit values (`"cpu"` / `"cuda"` / `"mps"`) pinned verbatim;
  unknown values raise `ValueError`.  Module-level so unit tests can
  monkeypatch `torch.cuda.is_available` / `torch.backends.mps.is_available`
  without instantiating `NnMlpModel`.
- `bristol_ml.models.nn.mlp._seed_four_streams(seed, device) -> None`
  — four-stream reproducibility recipe (plan D7').  Seeds `random`,
  `numpy.random`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`;
  on CUDA additionally sets `cudnn.deterministic = True` and
  `cudnn.benchmark = False`.  No-op on the non-CUDA flags off-CUDA.
- `bristol_ml.models.nn.mlp._make_mlp(input_dim, config) -> nn.Module`
  — module factory.  Builds a `_NnMlpModule` (module-level subclass of
  `torch.nn.Module`) with the four z-score scaler buffers registered
  via `register_buffer` — so the scalers ride inside `state_dict()` and
  round-trip cleanly through `torch.save` / `torch.load`
  (plan D4 / D5).
- `python -m bristol_ml.models.nn` / `python -m bristol_ml.models.nn.mlp`
  — standalone CLI.  Prints the resolved `NnMlpConfig` schema + `--help`
  text.  Runs with zero side effects; does **not** fit a model.

## PyTorch specifics (Stage 10)

Every teammate touching this module must know these five things before
they open `mlp.py`.  They are not obvious if you have only worked with
the scipy / statsmodels models.

- **`_NnMlpModuleImpl` is a lazy-built class, *installed* onto the
  module.**  `torch` is imported lazily (next gotcha), so the
  `torch.nn.Module` subclass is defined inside
  `_build_nn_module_class()` rather than at module top-level.  That
  would normally defeat pickleability: `pickle` resolves a class via
  `getattr(sys.modules[cls.__module__], cls.__qualname__)`, which
  fails on a closure-scoped class.  The factory therefore does three
  things before returning the class: (i) patches
  `__module__ = "bristol_ml.models.nn.mlp"`, (ii) patches
  `__qualname__ = "_NnMlpModuleImpl"`, and (iii) *installs* the class
  into `sys.modules[__name__]` so the `getattr` lookup resolves.
  Without step (iii) the name-patch is a lie and
  `pickle.dumps(instance)` raises `AttributeError`.  The current save
  path does not pickle an `nn.Module` instance (the envelope carries
  only `state_dict_bytes` + scalars), but any future use with joblib /
  pickle / `copy.deepcopy` would trip on it — so the guarantee is
  load-bearing.  Regression guard:
  `test_nn_mlp_module_impl_is_pickleable` (Stage 8 precedent:
  `test_parametric_fn_is_pickleable`).
- **Torch is imported lazily, not at module load.**  `mlp.py` imports
  `torch` inside function bodies (`_select_device`,
  `_seed_four_streams`, `_make_mlp`) and guards the class-level type
  hints behind `if TYPE_CHECKING:`.  Rationale: the package-level
  `from bristol_ml.models.nn import NnMlpModel` re-export still works
  during test collection on a host where torch is slow to import, and
  the CLI scaffold path (`python -m bristol_ml.models.nn --help`)
  never pays the ~0.5-1 s torch import cost if the user only wants
  help text.
- **Scaler buffers are registered at module construction, not fit
  time.**  `_NnMlpModule.__init__` unconditionally calls
  `self.register_buffer("feature_mean", ...)` / `"feature_std"` /
  `"target_mean"` / `"target_std"` with placeholder zeros / ones,
  *then* `NnMlpModel.fit` overwrites them with the fitted column
  statistics via in-place assignment.  Rationale (plan R3):
  `load_state_dict(strict=True)` fails loudly if any buffer is
  missing or extra — so the reconstruction path at load time needs
  the freshly-built module to already have the four buffers in its
  `state_dict`, before the loaded bytes overwrite them with the
  fitted values.  Missing this would surface as "model loads fine
  but predict is off by a scale factor because the scaler buffers
  were silently reset to zero" — a class of bug the strict-mode
  round-trip catches by construction.
- **`torch.load(..., weights_only=True, map_location="cpu")` at load
  time.**  PyTorch 2.6+ sets `weights_only=True` by default on
  `torch.load`; we pass it explicitly because being explicit is cheap
  and the upgrade path to `skops.io` at Stage 12 benefits from knowing
  the current safety rail.  `map_location="cpu"` is load-bearing: the
  Stage 12 serving stub is CPU-only and cannot materialise a CUDA
  `state_dict` without it.  `test_nn_mlp_save_and_load_round_trips_
  state_dict_and_hyperparameters` is the regression guard.
- **Single-joblib envelope; `state_dict_bytes` inside.**  The Stage 9
  registry passes a **file** path (`artefact/model.joblib`) to
  `Model.save`, not a directory — this was the mid-T3 plan D5
  revision.  `NnMlpModel.save` therefore writes one joblib file
  containing the envelope dict described in the layer doc (§4).
  Do **not** revert to the research-draft's two-file layout
  (`model.pt` + `hyperparameters.json`); `test_nn_mlp_save_writes_
  single_joblib_file_at_given_path` is the structural guard.

## Reproducibility semantics (plan D7' + NFR-1)

Intent AC-2 says training is reproducible given a seed "within the
constraints of non-deterministic GPU operations".  The sub-layer
honours this with a device-split contract:

- **On CPU** (the default for CI and the notebook's live demo): two
  back-to-back `fit(seed=0)` runs produce `state_dict` tensors that
  compare equal under `torch.equal`, and `predict` outputs that
  compare equal under `torch.equal`.  The test surface is
  `test_nn_mlp_seeded_runs_produce_identical_state_dicts` +
  `test_nn_mlp_seeded_runs_produce_identical_predict_output`.
- **On CUDA / MPS**: the same two runs produce `predict` outputs that
  match under `torch.allclose(atol=1e-5, rtol=1e-4)`.  The test is
  `test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance`,
  `@pytest.mark.gpu`, skipped when `torch.cuda.is_available()` is
  false.

`torch.use_deterministic_algorithms(True)` is **off** — setting it to
`True` would cost real throughput on the Blackwell dev host for a
guarantee the intent does not require.  Plan H-5 / OQ-A closed `NO`.

The per-fold seed derivation is `fold_seed = config.project.seed +
fold_index` when the harness orchestrates (plan D8), or `config.seed`
directly when the caller pins it (e.g. the notebook passes `seed=0`).
`config.seed` on `NnMlpConfig` is `int | None`; `None` falls back to
the `project.seed + fold_index` derivation.

## Extraction seam for Stage 11 (plan D10)

The training-loop body in `NnMlpModel._run_training_loop` is the named
extraction target.  The seam marker is a comment inside that method:

```python
# Stage 11 extraction seam: when the second torch-backed model
# (temporal architecture) arrives with a hand-rolled loop of its own,
# extract the body of this method + ``_make_mlp`` to
# ``src/bristol_ml/models/nn/_training.py`` under a shared helper.
```

Extraction happens when Stage 11's loop diverges from Stage 10's by
more than "add one optimiser kwarg".  Do **not** ship a
`BaseTorchModel` ABC at Stage 10 (scope-diff X7 cut); it would bind
Stage 11's design before Stage 11's requirements are understood.

## Stage 11 additions

Stage 11 extends this sub-layer with the TCN family (`NnTemporalModel`)
and fires the Stage 10 D10 extraction seam — the training-loop body now
lives at `bristol_ml.models.nn._training.run_training_loop` and is
shared by both `NnMlpModel` and `NnTemporalModel`.  The five Stage 10
PyTorch gotchas above all carry forward unchanged (the temporal class
re-applies the `sys.modules` install, the lazy torch import, the
buffer-registration discipline, the `weights_only=True + map_location=
"cpu"` load contract, and the single-joblib envelope).  The four
gotchas that are *new at Stage 11* and that the Stage 12 / 17 reader
cannot guess from `mlp.py` are:

- **Causal padding via `F.pad(x, (left, 0))` *before* a `Conv1d(padding=
  0)`, never `Conv1d(padding=...)`.**  The Bai et al. (2018) recipe.
  `left = (kernel_size - 1) * dilation` is computed once per
  `_TemporalBlockImpl` at `__init__` time.  Inside `forward`,
  `F.pad(x, (self.left_pad, 0))` zero-pads only the left of the time
  axis; the convolution then runs with `padding=0` so it cannot see
  any input position strictly to the right of the current output
  position.  The temptation is to set `padding=self.left_pad` directly
  on `Conv1d` and skip the explicit `F.pad`.  Don't: PyTorch's `Conv1d`
  pads symmetrically (left and right by the same amount), which leaks
  the future into every output and silently breaks day-ahead
  forecasting.  The regression guard is
  `test_nn_temporal_causal_padding_does_not_leak_future` (synthetic
  `inf`-at-future-index fixture; any right-pad leakage produces
  `inf`/`nan` in the output, making the bug loud rather than silent).
  Domain research §6 pitfall 4.
- **Weight-norm via `torch.nn.utils.parametrizations.weight_norm`, not
  `torch.nn.utils.weight_norm`.**  The latter has been deprecated since
  PyTorch 2.1; the former is the parametrizations-API replacement.
  `temporal.py` imports the parametrizations entry point first and
  falls back to the legacy import for `torch < 2.1` (the project pins
  `torch>=2.7,<3` so the fallback is dead code on the supported track,
  but the structure documents the migration cost should the project
  ever extend the floor).  The two APIs have **different `state_dict`
  key shapes**: legacy stores `weight_g` / `weight_v`; parametrizations
  stores `parametrizations.weight.original0` / `original1`.  An
  artefact saved under one API does not load under the other —
  `load_state_dict(strict=True)` catches the mismatch loudly.  If a
  future torch version removes the legacy alias, every Stage-11-era
  artefact still loads because we already use the parametrizations
  path.  Plan R4.
- **`_SequenceDataset` is a lazy-window `torch.utils.data.Dataset` —
  do not eagerly materialise the full `(N - seq_len, seq_len,
  n_features)` tensor.**  At the intent's default feature set (44
  calendar + ~6 weather columns, 43 633 hourly rows, `seq_len=168`)
  the eager tensor would cost ~1.4 GB; the lazy `__getitem__` path
  holds the footprint at `N * n_features * 4` bytes (~10 MB).  The
  class is defined inside `_build_sequence_dataset_class()` for the
  same lazy-torch-import reason as `_NnMlpModuleImpl` (Gotcha 1
  above) and is `sys.modules`-installed under
  `__qualname__ = "_SequenceDatasetImpl"` so any future `pickle` /
  `joblib` round-trip resolves the class by dotted path.  The dataset
  yields **unscaled** `float32` tensors; normalisation happens inside
  `_NnTemporalModuleImpl.forward()` via the four scaler buffers
  (`feature_mean` / `feature_std` / `target_mean` / `target_std`)
  registered the same way Stage 10 registers them — so the scalers
  ride inside `state_dict()` and the strict-mode round-trip catches
  any drift.  Plan D7 / codebase map §4.
- **Single-joblib envelope gains one new field: `seq_len`.**  The
  Stage 10 envelope (`state_dict_bytes` + `config_dump` +
  `feature_columns` + `seed_used` + `best_epoch` + `loss_history` +
  `fit_utc` + `device_resolved`) is preserved verbatim; Stage 11 adds
  one more field, `seq_len: int`, *redundant* with `config_dump
  ["seq_len"]` but explicit so the load path can sanity-check the
  window size before reconstructing the module.  The reconstruction
  path is: read envelope → instantiate `NnTemporalConfig` from
  `config_dump` → assert envelope `seq_len == config.seq_len` →
  rebuild `_NnTemporalModuleImpl(input_dim=len(feature_columns),
  seq_len=config.seq_len, ...)` skeleton → `torch.load(BytesIO(
  state_dict_bytes), weights_only=True, map_location="cpu")` →
  `load_state_dict(strict=True)`.  The strict-mode call catches a
  mismatch in *any* Conv1d kernel shape (so a saved 8-block × 128-channel
  model cannot silently load into a 4-block × 32-channel skeleton —
  the kernel shapes differ).  Regression guards:
  `test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict`
  (asserts the loaded `config.seq_len` matches byte-exact) and
  `test_nn_temporal_save_writes_single_joblib_file_at_given_path`
  (structural; no sibling files).  Plan D5 / R7.

The other Stage 11 item worth surfacing — the harness-factory catch-up
that landed *with* T6's dispatcher wiring rather than as a Stage 10
hotfix — is filed as a one-stop discrepancy fix:

- **D14 — Stage 10 harness-factory gap, closed at Stage 11 T6.**
  Stage 10 added an `NnMlpConfig` branch to `train.py`'s inline
  dispatcher but did not add the matching branch to
  `bristol_ml.evaluation.harness._build_model_from_config`.  The gap
  was latent — the train CLI was the only user of `NnMlpModel` at the
  time — but the Stage-11 codebase-map §1.4 flagged it as a
  one-line discrepancy that would bite anyone driving `model=nn_mlp`
  through the harness CLI directly.  Stage 11 T6 adds *both* the new
  `NnTemporalConfig` branch (plan D13 clause iii) *and* the missing
  `NnMlpConfig` branch in the same commit, titled `Stage 11 T6:
  dispatcher wiring for nn_temporal + Stage 10 NnMlpConfig catch-up`.
  The dispatch-test coverage in
  `tests/unit/evaluation/test_harness.py` now includes a regression
  guard (`test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up`)
  so a Stage-10-style gap cannot recur for a third NN family.

## Running standalone

    python -m bristol_ml.models.nn          --help
    python -m bristol_ml.models.nn.mlp      --help

Both entry points print the resolved `NnMlpConfig` schema.  Neither
path fits a model — the CLI is a schema-help surface, not a training
driver.  To train end-to-end, use the train CLI with the `nn_mlp`
model group:

    python -m bristol_ml.train model=nn_mlp

The `NnMlpConfig` isinstance branch in `train.py` picks up the family;
the Stage 9 registry-save path stores the final-fold fitted model
under `type = "nn_mlp"` automatically (plan D2 clause iii / v).

## Cross-references

- Sub-layer contract — `docs/architecture/layers/models-nn.md`.
- Parent layer contract — `docs/architecture/layers/models.md` (the
  `Model` protocol + `ModelMetadata` + joblib IO shared by every
  family).
- Stage 10 plan — `docs/plans/completed/10-simple-nn.md`.  Key decisions: D3 (architecture defaults),
  D4 (scaler buffers), D5 revised (single-joblib envelope), D6 (loss
  history + `epoch_callback` seam), D7' (four-stream reproducibility),
  D8 (cold start), D9 (internal 10 % val tail + best-epoch restore),
  D10 (training-loop ownership), D11 (device auto-select), D12 (this
  doc).
- Stage 10 retro — `docs/lld/stages/10-simple-nn.md`.
- Notebook — `notebooks/10-simple-nn.ipynb` (AC-3 live-loss-curve
  pedagogical surface).
- PyTorch reproducibility docs —
  <https://pytorch.org/docs/stable/notes/randomness.html>.
- Ash & Adams (NeurIPS 2020) — cold-start justification for rolling
  origin per plan D8.
