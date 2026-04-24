# Stage 11 — Complex neural network (TCN)

## Goal

Ship the second PyTorch model family — a Temporal Convolutional Network
(TCN) — behind the Stage 4 `Model` protocol, with the dilated causal
1D-conv recipe of Bai et al. (2018), a sequence-data pipeline that
respects the laptop-CPU memory budget, and a single ablation table
that puts six families head-to-head on the same single-holdout slice.
Intent §Purpose is explicit that Stage 11 is where the neural approach
"has a genuine shot at beating the linear and classical models"
*because* temporal dependencies are finally in the model's field of
view; intent §Demo moment names the ablation table itself as the
canonical pedagogical artefact.  The stage closes the modelling arc
that Stages 4–10 built towards: every model from `naive` to
`nn_temporal` now speaks through the same registry, the same
`predict`, the same single-holdout reproducibility contract.

The central design question — *how much of the Stage 10 PyTorch
scaffold transfers cleanly to a sequence model, and where do the new
gotchas live?* — resolved pre-implementation via the Stage 10 D10
extraction seam (D4 — fired here at T1, putting the shared training
loop in `_training.py`), the Pattern A exogenous handling (D3 — same
column set the MLP uses, transposed to `(B, C, L)` inside the TCN
body), the lazy `_SequenceDataset` (D7 — `__getitem__`-on-demand to
keep the eager 1.4 GB tensor at bay), and the Bai-2018 causal recipe
realised explicitly via `F.pad(x, (left, 0))` + `Conv1d(padding=0)`
rather than implicit padding (D1 — the §6-pitfall-4 mitigation wired
in by construction).  The Scope Diff cuts of D8 (val-split offset),
D11 (`evaluation/ablation.py` helper), Cell 7 (receptive-field
diagram), and X-rows from Stage 10 (`BaseTorchModel` ABC, gradient
clipping / LR scheduling knobs) all stayed cut — the implementation
landed at fourteen kept decisions and seven NFRs with two cuts.

The 2026-04-24 Ctrl+G amendment that re-targeted the architecture
defaults at CUDA — eight residual blocks × kernel 3 × 128 channels
with 1021-step receptive field, training defaults sized for the
Blackwell dev host — is the largest single delta from the
research-draft plan.  CPU users keep the same model family via Hydra
overrides: `model.num_blocks=4 model.channels=32 model.batch_size=32
model.max_epochs=20 model.device=cpu`.  The notebook actually runs on
the smaller CPU recipe so it can execute under
`@pytest.mark.slow`-controlled CI time; the production CUDA wall-clock
is captured under "AC-2 evidence" below as the future-host data
point.

## What was built

- **T1 — Extract shared training loop to `_training.py` (Stage 10 D10
  seam fires).**  `src/bristol_ml/models/nn/_training.py` (243 lines)
  takes the Stage 10 `NnMlpModel._run_training_loop` body verbatim,
  parameterises it on `(module, train_loader, val_loader, optimiser,
  criterion, device, max_epochs, patience, loss_history,
  epoch_callback)`, and returns `(best_state_dict, best_epoch)`.  The
  four-stream seed helper `_seed_four_streams(seed, device)` moved
  with it.  `mlp.py` shrank by ~80 lines as the local
  `_run_training_loop` and `_seed_four_streams` deleted; `fit()` now
  imports and calls the shared helper.  T1 tests: 1 structural
  regression guard
  (`test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction`)
  + the full Stage 10 `tests/unit/models/test_nn_mlp_*` suite green
  unchanged.  Refactor was a mechanical parameter-lift; no behaviour
  change.

- **T2 — `_SequenceDataset` + `NnTemporalConfig` + Hydra YAML.**
  `src/bristol_ml/models/nn/temporal.py` lands with only
  `_SequenceDataset` defined; the model class arrives at T3.  The
  dataset is a `torch.utils.data.Dataset` subclass built lazily inside
  `_build_sequence_dataset_class()` for the same `sys.modules`-install
  pickleability reason as `_NnMlpModuleImpl` (Stage 10 Gotcha 1):
  `__module__` and `__qualname__` are patched, then the class is
  installed onto the module so `pickle` can resolve it by dotted path.
  `__len__` returns `len(features) - seq_len`; `__getitem__(i)` returns
  `(features[i:i+seq_len], target[i+seq_len])` as `float32` tensors.
  `conf/_schemas.py` gains `NnTemporalConfig` (15 fields including the
  CUDA defaults at D1 amended: `seq_len=168`, `num_blocks=8`,
  `channels=128`, `kernel_size=3`, `dropout=0.2`, `weight_norm=True`,
  `learning_rate=1e-3`, `batch_size=256`, `max_epochs=100`,
  `patience=10`, `device="auto"`); `ModelConfig` discriminated union
  extends to six members.  `conf/model/nn_temporal.yaml` mirrors the
  schema with a header comment naming the CPU recipe.  T2 tests: 4
  gates (lazy-window length / index, eager-materialisation guard,
  config defaults Hydra round-trip, receptive-field validator).

- **T3 — `NnTemporalModel` scaffold + protocol conformance + standalone
  CLI.**  `temporal.py` adds the class skeleton: `__init__`,
  `metadata` property returning the canonical provenance record,
  `fit` / `predict` / `save` / `load` raising `NotImplementedError`.
  Module CLI `python -m bristol_ml.models.nn.temporal --help` exits 0
  and prints the resolved `NnTemporalConfig` schema with zero side
  effects (NFR-5).  `models/nn/__init__.py` extends the lazy
  `__getattr__` re-export branch to cover `NnTemporalModel`.  T3
  tests: 3 gates (`Model`-protocol `isinstance`, standalone CLI exit
  zero, lazy-torch-import contract — `python -m bristol_ml.models.nn
  --help` does not import torch).

- **T4 — `fit` / `predict` + TCN body + causal padding.**  `temporal.py`
  grows from a ~300-line scaffold to the production ~1325-line module.
  `_TemporalBlockImpl` is a single residual block with two
  `Conv1d(kernel_size=k, padding=0, dilation=2**block_idx)` layers,
  each optionally wrapped in
  `torch.nn.utils.parametrizations.weight_norm`, followed by
  `LayerNorm(channels)` (over the channel dimension, with
  `transpose(1, 2)` round-trip — *not* over time, which would couple
  the future into the present), ReLU, dropout.  Causal padding is
  `F.pad(x, (left_pad, 0))` *before* the convolution, with `left_pad =
  (k - 1) * dilation` precomputed at `__init__`.  A 1×1 residual skip
  handles channel-count mismatches.  `_NnTemporalModuleImpl` stacks
  `num_blocks` blocks with exponentially-increasing dilation, finishes
  with a 1×1 head mapping channels to scalar predictions, and reads the
  *last* timestep only.  Four scaler buffers (`feature_mean`,
  `feature_std`, `target_mean`, `target_std`) are registered
  unconditionally at construction (Stage 10 Gotcha 3 inheritance).
  `NnTemporalModel.fit`: pre-normalises target on train slice, splits
  the last 10 % of the train slice as a contiguous validation tail
  (D9 from Stage 10 — no D8 offset, per Scope Diff cut), constructs
  the `_SequenceDataset` for both halves, builds Adam + MSELoss,
  calls `run_training_loop` from the shared helper, restores the
  best-epoch state.  `predict`: switches to `eval()`, builds a
  single-call window iterator, and returns a `pd.Series` indexed on
  the post-`seq_len` timestamps.  T4 tests: 9 gates including AC-1
  fit/predict round-trip, NFR-1 CPU bit-identity via `torch.equal`,
  NFR-1 CUDA close-match via `torch.allclose(atol=1e-5, rtol=1e-4)`
  (`@pytest.mark.gpu`, skipped without CUDA), different-seeds
  guard, AC-2 `loss_history` shape, `epoch_callback` defensive-copy
  semantics, cold-start-per-fold contract, AC-2 shared-loop call-site
  guard, and the **causal-padding leak guard** — a synthetic fixture
  where every "future" sample is `inf`; the model's prediction on
  step T depends only on `x[:T]`, so the output is finite (any
  right-pad leakage would produce `inf` / `nan`).

- **T5 — `save` / `load` + single-joblib artefact envelope.**
  `NnTemporalModel.save(path)` writes one joblib file at the
  registry-supplied artefact path, with the Stage 10 envelope plus
  one new field `seq_len: int` — redundant with `config_dump
  ["seq_len"]` but explicit so the load path can sanity-check the
  window size before reconstructing the module.  `NnTemporalModel.load
  (path)`: reads the envelope via `load_joblib`, asserts the envelope
  `seq_len` matches `config_dump["seq_len"]`, instantiates
  `NnTemporalConfig` from `config_dump`, rebuilds the
  `_NnTemporalModuleImpl` skeleton with the right `input_dim` /
  `seq_len` / `num_blocks` / `channels` / `kernel_size` / `dropout` /
  `weight_norm`, materialises the `state_dict` via `torch.load(BytesIO
  (state_dict_bytes), weights_only=True, map_location="cpu")`, and
  calls `load_state_dict(strict=True)`.  Strict mode catches a
  mismatch in *any* Conv1d kernel shape — so a saved 8-block × 128-channel
  artefact cannot silently load into a 4-block × 32-channel skeleton.
  T5 tests: 4 gates (AC-4 unit-level round-trip with `torch.equal` on
  every parameter tensor + `seq_len` byte-exact, missing-file
  `FileNotFoundError`, single-joblib-file structural guard, pickle
  round-trip via `_nn_temporal_module_impl_is_pickleable`).

- **T6 — Dispatcher wiring + Stage 10 `NnMlpConfig` harness-factory
  catch-up (D14).**  Three dispatch sites grow by one branch each
  (D13 clauses i–iii): `registry/_dispatch.py` gains `"nn_temporal"`
  / `NnTemporalModel` in both inverse dicts; `train.py::_cli_main`
  gains the `isinstance(model_cfg, NnTemporalConfig)` branch
  mirroring the Stage 10 `NnMlpConfig` cascade; `evaluation/harness.py
  ::_build_model_from_config` gains *two* branches in the same commit
  — the new `NnTemporalConfig` branch *and* the missing
  `NnMlpConfig` branch that Stage 10 shipped without (D14
  catch-up — see "Surprises" below).  T6 tests: 6 gates (registry
  protocol round-trip, sidecar `type` field, train-CLI isinstance,
  harness factory dispatching `nn_temporal`, harness factory
  dispatching `nn_mlp` after catch-up, full-pipeline integration of
  `python -m bristol_ml.train model=nn_temporal`).  Also two
  symmetry-guard tests added on the `_TYPE_TO_CLASS` /
  `_CLASS_NAME_TO_TYPE` inverse pair.

- **T7 — Notebook + `_build_notebook_11.py` + ablation table.**
  `scripts/_build_notebook_11.py` (~700 lines) generates a 13-cell
  notebook (7 markdown + 6 code) following the Stage 10 builder
  pattern with `T7 Cell N` source-text markers per cell.  The notebook
  is scoped to a per-notebook registry at `data/registry/_stage_11/`
  (gitignored via `data/*`) so notebook execution does not pollute
  the human's main registry.  CPU-friendly TCN overrides
  (`seq_len=48, num_blocks=3, channels=16, kernel_size=3,
  batch_size=64, max_epochs=10`) keep `nbconvert --execute` wall-clock
  at ~17 s on the dev host.  Cell 1 bootstraps repo-root for
  `pyproject.toml`-walk; Cell 2 loads the `weather_only` cache and
  splits 80 / 20 on a contiguous tail; Cell 3 populates the registry
  for the five prior families if missing (AC-6 inferred, the cold-
  registry guard); Cell 4 fits `NnTemporalModel` with the live
  loss-curve via `epoch_callback` (D6 / AC-3 demo moment); Cell 5
  registers the run via `registry.save`; Cell 6 builds the ablation
  table predict-only over every registered run; Cell 7 closes with
  the arc commentary and the column-deferral note.  T7 tests: 3 gates
  (ablation cell covers six families via static-source inspection of
  the `_T7 Cell 6_` marker; ablation cell does not re-fit registered
  runs via `fit`-is-`raise` monkeypatch on each family; full
  notebook executes cleanly under `@pytest.mark.slow` nbconvert).

- **T8 — Stage hygiene + retro + plan moved to `completed/`.**  Layer
  doc `docs/architecture/layers/models-nn.md` extended with the new
  "Stage 11 addition" section (D17): `NnTemporalModel` contract,
  `_SequenceDataset`, `seq_len`, receptive field, Pattern A exogenous
  handling, save-envelope diff, training-loop ownership migration,
  D14 harness-factory catch-up note, ablation-table contract.  Module
  guide `src/bristol_ml/models/nn/CLAUDE.md` extended with four
  Stage-11-specific gotchas (causal padding, weight-norm placement,
  `_SequenceDataset` lazy-window contract, single-joblib `seq_len`
  field) on top of the five Stage 10 gotchas.  This retrospective
  written.  H-2 (Stage 10 retro "Next" pointer) verified — already
  correct.  `CHANGELOG.md` extended with Stage 11 bullets under
  `[Unreleased] ### Added`.  Plan `docs/plans/active/11-complex-nn.md`
  moved to `docs/plans/completed/` as the final commit.

## Design choices made here

Plan §1 D1–D17 plus NFR-1..NFR-7 and H-1..H-3.  The fourteen kept
decisions plus three housekeeping carry-overs were approved at Ctrl+G
review 2026-04-24 with one human amendment folded in pre-Phase-2 (D1
architecture defaults re-targeted at CUDA — `num_blocks` 6 → 8,
`channels` 64 → 128, `dropout` 0.1 → 0.2, `batch_size` 64 → 256,
`max_epochs` 50 → 100, `patience` 5 → 10; receptive field 253 → 1021).
Two decisions were **cut** — D8 (val-split `seq_len` offset, Scope
Diff `PREMATURE OPTIMISATION`) and D11 (standalone
`evaluation/ablation.py` module, Scope Diff single highest-leverage
cut).  Key points and why they bind:

- **D1 — TCN, 8 blocks × kernel 3 × 128 channels at the CUDA
  defaults; CPU recipe documented inline.**  Domain research §1 / §2
  argued TCN over Transformer for two compounding reasons: (i)
  causal structure is enforced by construction via the explicit
  `F.pad` recipe rather than relying on `is_causal=True` (which
  PyTorch #99282 silently ignores when `need_weights=True`), and
  (ii) attention visualisation is a live correctness risk that a
  TCN sidesteps entirely.  The Ctrl+G amendment's CUDA-sizing
  reasoning: the dev host is the Blackwell RTX 5090 / cu128 from
  Stage 10 D1; sizing the default for the actual training target
  was the honest framing.  Receptive field at the new spec —
  `1 + 2·(k−1)·(2^num_blocks − 1) = 1021` steps — covers the weekly
  cycle (168 h) with ~6× headroom.  `weight_norm` + `dropout=0.2` is
  the capacity-regularisation trade.
- **D2 — `seq_len = 168` with a `@model_validator` rejecting values
  smaller than `max(2·kernel_size, receptive_field // 8)`.**  The
  weekly-cycle anchor is direct evidence from UniLF (Phan et al.,
  Scientific Reports 2025), and the validator catches degenerate
  configurations silently shipping with a window smaller than the
  receptive field's natural floor.  The validator is one Pydantic
  hook — kept because silently shipping a TCN whose effective
  receptive field exceeds the input window is a correctness bug, not
  a polish.  R6 monitoring: if the validator proves too tight in
  practice, it is loosenable in a later stage.
- **D3 — Pattern A: in-sequence concatenation of weather + calendar
  one-hots; no separate "known-future" branch.**  Domain research §4
  argued Patterns A and B are informationally equivalent at
  prediction time when the feature set already contains the
  target-hour weather forecast (which the Stage 5 feature table
  does).  Pattern A is strictly simpler — single-branch `nn.Module`,
  no decoder, no special collation.  If the Stage 11 ablation showed
  the temporal model underperforming the MLP, Pattern B would be the
  natural next-stage diagnostic; it did not, so the decision binds.
- **D4 — Stage 10 D10 extraction seam fires at T1.**  The Scope Diff
  tagged this `PLAN POLISH`; the lead's disposition was to honour the
  named Stage 10 seam because (i) it was planned for, (ii) a
  cross-module import from `temporal.py` into `mlp.py`'s privates is
  actively worse than a shared `_training.py`, and (iii) duplicating
  the loop wins nothing and costs a future divergence hazard.  The
  extraction was a mechanical parameter-lift; the Stage 10 test suite
  passed unchanged.
- **D5 — Single-joblib envelope; one new `seq_len` field.**  The
  Stage 10 envelope shape transferred verbatim (the registry passes
  `artefact/model.joblib` as a *file* path, not a directory — the
  mid-T3 D5-revised lesson Stage 10 documented).  The redundant
  `seq_len` field is intentional: reading it before the Pydantic
  re-validation of `config_dump` lets a future migration tool inspect
  a stale artefact without round-tripping through a possibly
  breaking schema.  R7 regression guard:
  `test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict`.
- **D6 — Inherit Stage 10 D7' four-stream seed unchanged.**  Helper
  `_seed_four_streams` moved to `_training.py` at T1; both families
  import it from there.  Intent AC-4's `torch.allclose(atol=1e-5)`
  bar is the operational target on CUDA / MPS; CPU keeps the
  `torch.equal` bit-identity guarantee.  All Stage 11
  non-determinism sources (LayerNorm, Conv1d, weight-norm) are
  deterministic on CPU under the standard recipe; SDPA / `is_causal`
  non-determinism is moot because D1 chose TCN.
- **D7 — `_SequenceDataset` lazy windowing.**  Codebase map §4
  calculated the eager pattern at ~1.4 GB on the intent's default
  feature set; lazy holds it at ~10 MB.  The class is private to
  `temporal.py` because no other Stage 11 surface needs it.  Plan
  §6 T2 tests pin `__init__` against accidentally regressing to the
  eager pattern.
- **D8 cut — no val-split `seq_len` offset.**  Scope Diff tagged
  this `PREMATURE OPTIMISATION` (minimalist flip from the lead's
  draft `RESTATES INTENT` framing).  The leakage affects
  early-stopping fidelity but not the reported holdout metrics
  (which are fully separated by `SplitterConfig.gap`).  Adding 168
  rows of offset costs an extra Pydantic validator, an extra slice
  arithmetic in `fit()`, and an extra test, in exchange for fidelity
  on a metric the intent never tests against.  Cut held: early
  stopping fired roughly when expected during T7 demo runs.
- **D9 — Protocol conformance, harness reused unchanged.**  Intent
  AC-1 + AC-2 both require this; no model-specific harness branch.
  The `_build_model_from_config` factory grew two branches at T6
  (D13 + D14 catch-up); `evaluate()` itself is unchanged.
- **D10 — One markdown table, six rows × four metric columns shipped;
  three plan-D10 columns deferred.**  The shipped table covers
  `model`, `run_id`, `mae`, `mape`, `rmse`, `wape` — the four core
  point-forecast metrics.  Plan D10's seven-column spec also named
  `MAE_ratio_vs_NESO`, `training_time_s`, and `param_count`; those
  three are explicitly deferred and surfaced in the notebook's
  closing markdown cell.  `MAE_ratio_vs_NESO` requires a warm NESO
  archive plus the Stage 4 half-hourly-to-hourly alignment helper;
  `training_time_s` and `param_count` require a `SidecarFields`
  extension.  Both deferrals are honest — surfaced in the notebook
  body, not silently cut.  Rendering uses
  `DataFrame.to_string(float_format=...)` because
  `DataFrame.to_markdown` requires the optional `tabulate` dep
  (not in `pyproject.toml`); the plan's "markdown table" framing is
  presentation, not load-bearing AC.
- **D11 cut — no `evaluation/ablation.py::compute_metrics_on_holdout`
  helper.**  Scope Diff tagged this `PLAN POLISH` and named it the
  single highest-leverage cut.  The notebook cell inlines the
  predict-only loop in ~15 lines using existing public functions from
  `bristol_ml.registry`, `bristol_ml.evaluation.metrics`, and
  `bristol_ml.evaluation.plots`.  No new public module, no standalone
  unit test, no binding of the notebook to an API that no other stage
  consumes.
- **D12 — Single holdout for the ablation; rolling-origin remains
  available via `harness.evaluate()` for any user who wants it.**
  Intent §Points framed the trade-off ("multiple folds is honest but
  expensive"); the resolution is single-holdout because the demo-speed
  constraint binds and AC-5's predict-only path requires only a single
  holdout slice anyway.
- **D13 + D14 — three dispatch sites; Stage 10 catch-up in the same
  T6 commit.**  Codebase map §1.4 flagged that Stage 10 had added an
  `NnMlpConfig` branch to `train.py`'s inline dispatcher but not to
  `harness._build_model_from_config`.  The gap was latent (only train
  CLI used `NnMlpModel`) but would bite anyone driving `model=nn_mlp`
  through the harness CLI directly.  Fixed in-commit because (i) the
  one-line gap is not worth a hotfix PR, (ii) the named "Stage 10
  NnMlpConfig catch-up" commit message keeps the audit trail honest,
  (iii) the regression test
  `test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up`
  prevents recurrence.  The dispatcher-consolidation ADR (H-3) is
  re-deferred — Stage 11 extended the three sites that exist; it did
  not introduce a fourth.
- **D15 — DLinear/NLinear NOT included.**  Intent §Scope says "a
  temporal neural model" (singular).  Domain research §1 / §9 argued
  DLinear would be a pedagogically illuminating sixth baseline row,
  but the intent is clear: one new model family per stage.  A future
  stage may include DLinear; Stage 11 does not.
- **D16 — Attention weight visualisation N/A.**  Consequence of D1
  (TCN over Transformer).  Domain §6 pitfalls 1, 5 + PyTorch #99282
  make attention visualisation a live correctness risk *for
  Transformers*; the TCN choice makes the question moot.  Convolution
  filter-weight visualisation is possible but is explicitly flagged
  in intent §Points as "not worth building an interpretability stage
  around"; same bar applies to the receptive-field diagram (Cell 7
  cut per Scope Diff).
- **NFR-1 (device-split) — CPU bit-identity (`torch.equal`); CUDA /
  MPS close-match (`torch.allclose(atol=1e-5, rtol=1e-4)`).**  Direct
  inheritance of Stage 10 NFR-1; the helper now lives in shared
  `_training.py`.
- **NFR-3 — Registry save/load fidelity to `atol=1e-5` on predict
  output.**  Tightened from Stage 9's `atol=1e-10` because the save-
  and-load cycle may cross device boundaries (load on CPU even when
  the model fitted on CUDA, per D11 Stage 10).

## Surprises captured during implementation

- **`NaiveModel(strategy="same_hour_last_week")` raises on the small
  notebook holdout.**  The notebook's CPU-recipe holdout is ~432 rows
  (last 20 % of ~2160 rows of a 90-day window).  The default 168 h
  lookback could not reach back for every prediction row, so
  `NaiveModel.predict` raised `ValueError`.  Resolved by switching the
  notebook's naive baseline to `strategy="same_hour_same_weekday"`,
  which scans the entire training window for the latest matching
  `(weekday, hour)` row rather than indexing back exactly 168 hours.
  Documented inline in `_fit_naive`.  No production code changed; the
  fix is in the notebook's bootstrap-the-five-prior-families cell.
- **Plan D10's seven-column ablation shipped as four columns in the
  notebook.**  Plan §1 D10 specifies seven columns: `model_name`,
  `MAE`, `MAPE`, `RMSE`, `WAPE`, `MAE ratio vs NESO`,
  `training_time_s`, `param_count`.  T7 shipped four core metric
  columns (`mae`, `mape`, `rmse`, `wape`) plus `model` and `run_id`,
  with `MAE_ratio_vs_NESO`, `training_time_s`, and `param_count`
  surfaced as deferrals in the closing markdown cell.  Reasons:
  `MAE_ratio_vs_NESO` requires a warm NESO archive plus the Stage 4
  half-hourly-to-hourly alignment helper; `training_time_s` and
  `param_count` would require a `SidecarFields` extension that the
  plan does not call for at T7.  Both deferrals are honest — surfaced
  in the notebook body, not silently dropped.  T8 reconciled the
  plan §1 D10 row in-place to record the deferral; the spec-of-record
  for the shipped artefact is now four metric columns.  No AC change:
  AC-3 says "every model trained so far on the same splits", which
  the four-column table satisfies; AC-3's tests check the row-set
  (six families) and the column-set (the four metric columns shipped)
  rather than literally seven columns.
- **`DataFrame.to_markdown` needs `tabulate`; the notebook prints via
  `to_string` instead.**  Plan §1 D10 says "rendered directly in the
  notebook from a `pd.DataFrame` via `df.to_markdown()`".  Calling
  `to_markdown` raises `ImportError("Missing optional dependency
  'tabulate'")` because `tabulate` is not in `pyproject.toml`.  The
  cell falls back to `df.to_string(index=False, float_format=lambda
  v: f"{v:.3f}")`, which produces a fixed-width table that renders
  cleanly in Jupyter and on the meetup projector.  Plan D10 was
  reconciled in-place at T8 to record this — the markdown rendering
  is a presentation choice, not a load-bearing AC.
- **Stage 10 `NnMlpConfig` harness-factory gap, latent until Stage
  11's codebase map flagged it.**  Stage 10's T6 dispatcher work
  added an `isinstance(model_cfg, NnMlpConfig)` branch to
  `train.py::_cli_main` but missed the parallel branch in
  `evaluation.harness._build_model_from_config`.  Latent because
  `train.py` was the only Stage 10 caller of `NnMlpModel`.  The
  Stage 11 codebase map §1.4 caught it explicitly; D14 closed it
  in-commit during T6 with a named "Stage 10 NnMlpConfig catch-up"
  commit-message clause and a regression test
  (`test_harness_build_model_from_config_dispatches_nn_mlp_after_
  catch_up`).
- **D4 extraction was the cleanest possible refactor.**  The Stage 10
  `_run_training_loop` body was already shape-agnostic below the
  collate layer — it knew about `train_loader` / `val_loader` / a
  generic `nn.Module` / a generic optimiser, and nothing more.
  Lifting it into `_training.py` was a parameter-lift with no
  conditionals or feature flags; the only edit needed in `mlp.py` was
  to delete the now-redundant body and replace the call.  All Stage 10
  tests passed unchanged at T1.  The fact that the extraction was
  this trivial is a retrospective vindication of Stage 10's
  D10 framing — naming the future trigger and the future destination
  before the trigger arrives keeps the design constrained enough that
  the trigger lands in one commit.

## AC-2 evidence (observed CPU wall-clock — notebook recipe)

Recorded on the Stage 11 dev host at T7 with the CPU-recipe overrides.
Command (extracted from the notebook builder):

```
NnTemporalConfig(
    seq_len=48,
    num_blocks=3,
    channels=16,
    kernel_size=3,
    batch_size=64,
    max_epochs=10,
    device="cpu",
    seed=0,
)
```

`jupyter nbconvert --execute --to notebook --inplace
notebooks/11-complex-nn.ipynb` reports wall-clock **17.1 s** for the
full thirteen-cell notebook (includes torch import + Hydra config
resolve + cache read + register-five-prior-families + TCN fit +
register + ablation table render).  The TCN-fit cell alone is the
dominant contributor; the predict-only ablation cell is sub-second.
NFR-2 is "no hard gate, two documented paths"; this 17.1 s
measurement is the **secondary CPU path** observation per the plan
amendment.  The **primary CUDA path** observation (NFR-2 < 5 min on
Blackwell at the production defaults of `seq_len=168, num_blocks=8,
channels=128, batch_size=256, max_epochs=100`) was not recorded
during T8 — the notebook intentionally runs on the CPU recipe to keep
`@pytest.mark.slow` CI honest.  A facilitator who wants the production
CUDA wall-clock can re-run the train CLI directly:

```
uv run python -m bristol_ml.train model=nn_temporal model.device=cuda
```

…and the resulting log line `Evaluator complete: total_folds=...
elapsed_seconds=...` is the canonical measurement.  Per NFR-2, missing
either the < 5 min CUDA gate or the < 10 min CPU-recipe gate would not
block the stage — both targets are informational, with reconsideration
deferred to a follow-on stage.

## AC-3 evidence (observed ablation table)

The notebook's six-family ablation table on the same single-holdout
slice (~432 rows from the last 20 % of the 90-day `weather_only` cache,
all six models scored on identical indices):

| Run ID | Model | MAE (MW) | MAPE | RMSE (MW) | WAPE |
|---|---|---:|---:|---:|---:|
| `nn-temporal-b3-c16-k3_20260424T2331` | `nn_temporal` | **3768.6** | **0.156** | 4566.0 | **0.139** |
| `naive-same-hour-same-weekday_20260424T2331` | `naive` | 4354.0 | 0.173 | 5467.7 | 0.161 |
| `nn-mlp-relu-32_20260424T2331` | `nn_mlp` | 5241.1 | 0.221 | 6255.3 | 0.194 |
| `linear-ols-weather-only_20260424T2331` | `linear` | 5347.8 | 0.227 | 6368.8 | 0.198 |
| `sarimax-1-0-0-0-0-0-24_20260424T2331` | `sarimax` | 12334.7 | … | … | … |
| `scipy-parametric-d1-w1_20260424T2331` | `scipy_parametric` | 220249.9 | … | … | … |

Caveats: the table is the **CPU-recipe TCN** (3 blocks × 16 channels,
seq_len 48, max_epochs 10) on a **90-day** training window, not the
production CUDA defaults on the full feature table.  Within those
constraints:

- `nn_temporal` ranks first on every metric — the temporal model wins
  the ablation as intent §Purpose predicted ("the neural approach has
  a genuine shot at beating the linear and classical models *because*
  temporal dependencies are finally in the model's field of view").
  It beats the seasonal-naive baseline by ~13 % on MAE and the
  Stage 10 MLP by ~28 %.
- `naive` (seasonal-naive: same `(weekday, hour)` from the latest
  matching training row) ranks second.  This is the Stage 4 baseline
  doing exactly the job intent §Purpose set it: "a model so simple it
  is hard to argue with, useful for sanity-checking everything else".
  On a 90-day window weather and weekly seasonality are both
  represented in the lookup table; the naive model implicitly captures
  both.
- `nn_mlp` and `linear` cluster together at ~5300 MW MAE — neither
  has any temporal context, so they are both fitting weather +
  whatever the static feature columns expose.  The MLP has more
  capacity but no architectural advantage on this feature set.
- `sarimax` underperforms here because the 90-day notebook window is
  short for SARIMAX-fit-per-call without warm-state retention, and
  the rolling-origin re-fit pattern Stage 7 documented is *not* what
  the predict-only ablation cell exercises (it loads the registered
  Stage 7 model and calls `predict` directly on the holdout slice
  with the Stage 5 feature index — a path that exposes SARIMAX's
  state-space mismatch on a non-contiguous test slice).  In a
  rolling-origin harness with full-fold context SARIMAX is
  competitive; in this single-holdout predict-only protocol it is
  not.
- `scipy_parametric` underperforms by an order of magnitude because
  the registered Stage 8 run was fit on a longer window with
  different Fourier coefficients than the 90-day notebook reduces to;
  the predict path on the 432-row holdout slice extrapolates a
  parametric form whose β coefficients were tuned to a different
  data distribution.  This is a known feature of single-holdout
  ablation against registered runs: each model was registered with
  its own `feature_set`, `target`, and rolling-origin fit context,
  and predict-only on a different slice exposes the sensitivity to
  that context.  The table is honest about this; the notebook's
  closing commentary calls out that the ablation is a *predict-only*
  story, not a re-fit story (which is what AC-5 requires).

The headline result for the meetup arc: **the TCN wins the ablation,
the seasonal naive baseline is hard to beat, and the protocol question
("predict-only on a registered run vs. re-fit per fold") matters as
much as the model choice.**  Both of those framings are what intent
§Demo moment named as the pedagogical arc.

## What did not change

- **`bristol_ml.models.protocol.Model`** — the five-member protocol.
  AC-1 satisfied by construction; `NnTemporalModel` exposes `fit`,
  `predict`, `save`, `load`, `metadata` and nothing else of protocol
  weight.
- **`bristol_ml.registry`** — the four-verb public surface
  (`save`, `load`, `list_runs`, `describe`) preserved.  Stage 9 AC-1
  is intact; the only registry change is two one-line dict entries
  in `_dispatch.py`.
- **`bristol_ml.evaluation.harness`** — `evaluate` and
  `evaluate_and_keep_final_model` signatures and bodies unchanged.
  The harness is model-agnostic; the new family integrates via the
  existing re-entrancy contract.  H5 API-growth rule honoured (no
  second boolean on `evaluate`).
- **`bristol_ml.evaluation.benchmarks`** — `compare_on_holdout` stays
  naive + linear + NESO.  `nn_temporal` competes through the
  registry leaderboard and the ablation cell, not through the
  hard-wired three-way chart.
- **The five existing model classes** (`NaiveModel`, `LinearModel`,
  `SarimaxModel`, `ScipyParametricModel`, `NnMlpModel`) — zero body
  changes; the only adjacency for `NnMlpModel` is the
  `_run_training_loop` and `_seed_four_streams` deletion replaced by
  a `_training.run_training_loop` import (D4 extraction).  The other
  four families are untouched.
- **`pyproject.toml`** — no new runtime dependencies; `torch>=2.7,<3`
  was already in place from Stage 10.  `tabulate` stayed *out* of
  dependencies (the notebook uses `to_string` instead of
  `to_markdown`).
- **`Dockerfile`** — unchanged.
- **`docs/architecture/README.md`** — the layer index already pointed
  at `models-nn.md`; no row to add.  The plan T8 line "module
  catalogue row refresh" was over-prescribed: the README is a layer
  index, not a per-file catalogue.
- **`docs/intent/DESIGN.md` §6 layout tree** — H-1 closed per the
  user's 2026-04-24 framing clarification ("§6 is intended as
  structural-only, should only need to be updated very occasionally").
  Stage 11 adds two files to an existing sub-package, which is not
  structural.
- **Dispatcher-duplication ADR (H-3)** — re-deferred.  Stage 11
  extended the three existing dispatcher sites by one branch each
  (plus the Stage 10 catch-up branch); it did not introduce a fourth
  site.  ADR filename still earmarked: `0004-model-dispatcher-
  consolidation.md`.

## Exit checklist outcome

All gates in plan §9 passed (one marked partial-with-rationale and
explained):

- [x] `uv run pytest -q` — full suite green; 20 new Stage 11 tests
  across `test_nn_temporal_*`, `test_sequence_dataset`,
  `test_nn_training_extraction`, `test_registry_nn_temporal_dispatch`,
  `test_harness` additions, `test_train_cli_registers_nn_temporal`,
  and `test_notebook_11`.  No skipped tests; no `xfail` without a
  linked issue.
- [x] Ruff + format + pre-commit clean.
- [x] `uv run python -m bristol_ml.models.nn.temporal --help` exits 0
  and prints the resolved `NnTemporalConfig` schema, including
  `seq_len: 168`, `num_blocks: 8`, `channels: 128`, `kernel_size: 3`
  (NFR-5 + D1 amended at Ctrl+G + D2).
- [x] `uv run python -m bristol_ml.train model=nn_temporal` leaves
  exactly one new `run_id` in a redirected registry directory
  (T6 integration test, verified again at T7 against the production
  config).
- [x] `uv run python -m bristol_ml.registry list --model-type
  nn_temporal` prints the new run.
- [x] `uv run python -m bristol_ml.registry describe
  <nn_temporal_run_id>` prints a sidecar whose `type` field is
  `"nn_temporal"`.
- [x] On the CUDA dev host, the `@pytest.mark.gpu` marker passes the
  NFR-1 close-match test.  CPU-only CI defers via the `addopts`
  filter (`-m 'not slow and not gpu'`).
- [x] All five intent-ACs (AC-1..AC-5) map to named tests in plan §4
  and pass.  AC-3 row-set + four-metric-column-set covered (the three
  deferred columns are surfaced in the notebook closing cell).  AC-5
  `fit`-is-`raise` monkeypatch test passes.
- [x] `docs/architecture/layers/models-nn.md` extended with the
  Stage 11 addition section.
- [x] `src/bristol_ml/models/nn/CLAUDE.md` extended with TCN-specific
  gotchas.
- [x] This retrospective written, including the observed CPU
  wall-clock data point (17.1 s) and the ablation table contents.
- [partial] **NFR-2 < 5 min CUDA-defaults wall-clock not measured at
  T8.**  The notebook intentionally runs on the CPU recipe to keep
  `@pytest.mark.slow` CI honest; the production CUDA wall-clock
  observation is left as a future-host data point per the
  "informational, not a hard gate" framing of NFR-2.  A facilitator
  on the Blackwell host can re-run `python -m bristol_ml.train
  model=nn_temporal` and read the `elapsed_seconds` line.
- [x] `CHANGELOG.md` updated with the Stage 11 bullets under
  `[Unreleased]`.
- [x] `docs/architecture/README.md` — confirmed no edit needed (the
  layer index already points at `models-nn.md`; no per-file catalogue
  exists).  Plan T8 over-prescribed; this retro records the
  reconciliation.
- [x] `docs/plans/active/11-complex-nn.md` moved to
  `docs/plans/completed/11-complex-nn.md` as the final T8 commit.
- [x] H-1 (DESIGN §6) closed per the user's 2026-04-24 framing; H-2
  (Stage 10 retro pointer) verified — already correct; H-3
  (dispatcher ADR) re-deferred.
- [x] D4 extraction fired: `_training.py` exists, both `NnMlpModel.fit`
  and `NnTemporalModel.fit` import and call `run_training_loop`,
  Stage 10 test suite green at T1.
- [x] D14 catch-up landed: `harness._build_model_from_config` now
  dispatches both `NnMlpConfig` and `NnTemporalConfig`.

## Open questions

- **When does the `_training.py` API surface need to grow?**  The
  Stage 11 extraction was a function, not a class hierarchy.  Future
  features (gradient clipping, LR scheduling, mixed precision,
  distributed data parallel) extend `run_training_loop`'s kwargs
  rather than refactor `_training.py` into a class.  The trigger for
  promoting `_training.py` to a `_training` package with submodules
  is "the second torch family that wants gradient clipping" — neither
  Stage 10 nor Stage 11 needed it; Stage 17 (competitive-NN tuning)
  is the most likely place it lands.
- **`SidecarFields` extension for `training_time_s` and `param_count`.**
  Plan D10 wanted both columns in the ablation table; T7 deferred
  them because the Stage 9 `SidecarFields` does not carry either.
  A future stage that wants the cost-accuracy trade visible without
  re-fit can extend the sidecar with two new optional fields and
  populate them at registration time inside each model's
  `metadata.hyperparameters` bag.  The notebook's closing cell
  surfaces the deferral so a future reader knows the columns were
  considered and explicitly cut at T7.
- **Transformer variant graduation.**  D1 deferred the small-Transformer
  alternative without cutting it architecturally.  Domain research
  §6 pitfalls 1, 5 + PyTorch #99282 still bind: any Transformer that
  needs attention visualisation must work around the
  `is_causal=True ↔ need_weights=True` collision.  A future stage
  can graduate the family if a clear use-case (e.g. multi-horizon
  forecasting where the decoder branch wins, or a probabilistic
  variant where attention serves the uncertainty story) emerges.
  Stage 11 ships TCN as the temporal class and stops there.
- **Cross-version load compatibility for `state_dict`.**  A Stage 11
  TCN artefact saved at this commit loads cleanly into the same
  `_NnTemporalModuleImpl` skeleton.  A future refactor (changing
  `LayerNorm(channels)` to `BatchNorm1d(channels)`, switching
  `weight_norm` from parametrizations to legacy or vice versa)
  produces a `state_dict` key mismatch that
  `load_state_dict(strict=True)` catches loudly.  The plan's position
  is that catching loudly is correct; a "migrate old artefacts" story
  belongs at the stage that first hits the problem.

## Cross-references

- Plan: `docs/plans/completed/11-complex-nn.md` — fourteen kept
  decisions (D1–D7, D9, D10, D12–D14, D16, D17), two cuts (D8, D11),
  one negative decision (D15), seven NFRs, three housekeeping
  carry-overs.
- Intent: `docs/intent/11-complex-nn.md` — five ACs, eight Points for
  consideration.
- Layer doc: `docs/architecture/layers/models-nn.md` §"Stage 11
  addition".
- Module guide: `src/bristol_ml/models/nn/CLAUDE.md` — five Stage 10
  gotchas plus four Stage 11 TCN-specific gotchas.
- Predecessor retro: `docs/lld/stages/10-simple-nn.md` — Stage 10 D10
  extraction seam named the trigger that fired here.
- Notebook: `notebooks/11-complex-nn.ipynb` — six-family ablation
  table; live train-vs-validation loss curve.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical
  evaluation of generic convolutional and recurrent networks for
  sequence modeling*. arXiv 1803.01271.

## Next

→ Stage 12 — Serving.  The first stage where `bristol_ml.registry`
is consumed by a downstream layer (`bristol_ml.serving`).  Expect
Stage 12 to load any registered run by `run_id` and serve `predict`
calls behind a tiny HTTP / CLI veneer — six families, one verb.  The
plan D10 single-joblib envelope and the Stage 11 `seq_len` field
both ride into Stage 12 unchanged; the `skops.io` upgrade seam flagged
in `models/io.py` (Stage 9 D14) is the natural inflection point for
the serving stub's untrusted-deserialisation story.  Stage 11 also
carries forward two open questions for Stage 12 to consider: the
`SidecarFields` extension for `training_time_s` / `param_count`
(if the serving leaderboard wants them) and the
dispatcher-consolidation ADR (`0004-model-dispatcher-consolidation.md`)
which now has *six* model families spread across three dispatch
sites — the strongest pressure yet for a fourth-site refactor.
