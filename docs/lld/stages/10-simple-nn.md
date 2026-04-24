# Stage 10 — Simple neural network (MLP)

## Goal

Ship the first PyTorch model family — a small MLP — behind the Stage 4
`Model` protocol, with a hand-rolled training loop, a live train-vs-val
loss curve, patience-based early stopping, and a clean Stage 9 registry
round-trip.  Intent §Purpose is explicit that the *analytical* value of
Stage 10 is small ("the model itself is likely not to beat the Stage 5
linear regression by much"); the load-bearing contribution is the
**scaffold** — the training-loop conventions, four-stream
reproducibility discipline, single-joblib envelope, and the
`loss_history_` + `epoch_callback` surface — that Stage 11's temporal
architecture inherits.  The live loss curve is the Demo moment; the
registry round-trip is the structural proof that the five-member
protocol extends to a framework that cannot be pickled as naively as
statsmodels / scipy.

The central design question — *how much PyTorch-specific surface can
Stage 10 contain without binding Stage 11's design?* — resolved
pre-implementation via D5 (single-joblib envelope with
`state_dict_bytes` inside, not two files), D7′ (four-stream seed +
cuDNN deterministic flags; `use_deterministic_algorithms(True)` off),
D10 (hand-rolled loop inside `NnMlpModel._run_training_loop` with an
explicit extraction seam for Stage 11), and the Scope-Diff cuts of
X6 / X7 / NFR-2 / NFR-4 / T8.  That quartet kept AC-1 honest, pinned
the Stage 11 extraction trigger to a single named call-site, and kept
the plan small enough that T1 → T7 took seven sequential commits with
no backtracking.

## What was built

- **T1 — Scaffold + config schema + standalone CLI.**
  `src/bristol_ml/models/nn/{__init__.py, __main__.py, mlp.py}` plus
  `src/bristol_ml/models/nn/CLAUDE.md`.  `NnMlpModel(config)` exposes
  the five protocol members as stubs (`fit` / `predict` raise
  `NotImplementedError`; `save` / `load` round-trip an unfitted model
  only; `metadata` returns the canonical provenance record); module
  CLI prints the resolved `NnMlpConfig` schema + `--help` text with
  zero side effects.  `conf/_schemas.py` gains `NnMlpConfig` (13
  fields including `device: Literal["auto", "cpu", "cuda", "mps"] =
  "auto"` from D11) and `AppConfig.model` union extends to
  `NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig
  | NnMlpConfig`.  `conf/model/nn_mlp.yaml` mirrors the schema.
  `pyproject.toml` gains `torch>=2.7,<3` on `[project].dependencies`
  with `[tool.uv.sources]` + `[[tool.uv.index]]` pinning `torch` to
  the `https://download.pytorch.org/whl/cu128` wheel index on Linux
  (Blackwell / sm_120 support) and to PyPI elsewhere (MPS wheels ship
  on PyPI for macOS).  New `gpu` pytest marker registered; default
  `addopts` extended from `-m 'not slow'` to `-m 'not slow and not
  gpu'` so CI stays CPU-only by default.  T1 tests: 4 gates
  (`test_nn_mlp_scaffold.py`).

- **T2 — `fit` / `predict` + training loop + reproducibility + loss
  history.**  `mlp.py` grows from a 70-line stub to the production
  ~600-line module.  Module-level helpers: `_select_device(preference)`
  resolves `"auto"` in order CUDA → MPS → CPU with unit-testable
  monkeypatched branches; `_seed_four_streams(seed, device)` covers
  `random` + `numpy.random` + `torch.manual_seed` +
  `torch.cuda.manual_seed_all`, and on CUDA additionally sets
  `cudnn.deterministic = True` / `cudnn.benchmark = False` (D7′);
  `_make_mlp(input_dim, config)` instantiates a module-level
  `_NnMlpModule` subclass of `nn.Module` whose `__init__` unconditionally
  calls `register_buffer("feature_mean", …)` / `"feature_std"` /
  `"target_mean"` / `"target_std"` with placeholder zeros and ones so
  `load_state_dict(strict=True)` never drops a buffer (R3 guard).
  `NnMlpModel.fit` Z-score-normalises features and target on the
  training slice per fold (D4), constructs the module, takes the last
  10 % of the train slice as a contiguous validation tail (D9 — never
  a random split, so no time-series leakage), runs the hand-rolled
  epoch loop with patience-based early stopping and best-epoch weight
  restore, appends per-epoch `{"epoch", "train_loss", "val_loss"}`
  dicts to `self.loss_history_`, and calls the optional
  `epoch_callback` once per epoch.  `predict` flips to `eval()`
  mode, forwards through the normalise → MLP → inverse-transform
  pipeline, and returns a `pd.Series` re-indexed on `features.index`
  on CPU.  Cold-start per fold by construction (plan D8 / Ash & Adams
  NeurIPS 2020); the harness's existing re-entrancy contract is
  honoured.  T2 tests: 19 gates (`test_nn_mlp_fit_predict.py`) covering
  AC-1 fit/predict round-trip, AC-2 seeded-run bit-identity on CPU
  (`torch.equal` per parameter tensor and per prediction), AC-2
  `@pytest.mark.gpu` close-match on CUDA (`torch.allclose(atol=1e-5,
  rtol=1e-4)`, skipped when `torch.cuda.is_available()` is false),
  AC-3 `loss_history_` shape + `epoch_callback` invocation,
  early-stopping termination + best-epoch restore, cold-start-per-fold
  contract, three `_select_device` branches via monkeypatch, the
  explicit-pin + invalid-value error paths, and a structural guard
  that the val tail is a contiguous slice from the end of the train
  window.

- **T3 — `save` / `load` + single-joblib artefact envelope.**  Plan D5
  revision: the Stage 9 registry's `_fs.py::_atomic_write_run` hard-codes
  `artefact/model.joblib` as the file path, so `NnMlpModel.save(path)`
  writes **one** joblib file containing the envelope dict
  `{"state_dict_bytes", "config_dump", "feature_columns", "seed_used",
  "best_epoch", "loss_history", "fit_utc", "device_resolved"}`.
  `state_dict_bytes` is `torch.save(state_dict, BytesIO).getvalue()` —
  a plain dict of tensors, not an `nn.Module`, so joblib is not pickling
  a class reference or module topology (domain research §R3 safety
  rail).  `load(path)` inverses the pipeline: `load_joblib(path)` →
  rebuild `NnMlpConfig` from `config_dump` (Pydantic re-validation
  catches schema drift) → instantiate `NnMlpModel(config)` → `torch.load(
  BytesIO(state_dict_bytes), weights_only=True, map_location="cpu")` →
  `load_state_dict(strict=True)`.  `weights_only=True` preserves
  PyTorch 2.6+'s safety rail; `map_location="cpu"` is load-bearing for
  the Stage 12 CPU-only serving stub.  T3 tests: 5 gates
  (`test_nn_mlp_save_load.py`) including the structural guard
  `test_nn_mlp_save_writes_single_joblib_file_at_given_path` (no
  sibling `model.pt` or `hyperparameters.json` — guard against
  regressing to the pre-D5-revision two-file layout) and
  `torch.allclose(atol=1e-10)` on predict output across the
  round-trip.

- **T4 — Registry dispatch + `train.py` wiring.**  One new entry in
  each of `_TYPE_TO_CLASS` (`"nn_mlp": NnMlpModel`) and
  `_CLASS_NAME_TO_TYPE` (`"NnMlpModel": "nn_mlp"`) in
  `registry/_dispatch.py`; one new `isinstance(model_cfg, NnMlpConfig)`
  branch in `src/bristol_ml/train.py::_cli_main` mirroring the Stage 8
  `ScipyParametricConfig` cascade at lines 287–293.  The
  `evaluate_and_keep_final_model` → `registry.save` path from Stage 9
  T5 is reused unchanged — no new harness dispatch, no new registry
  verb (AC-1 of Stage 9 preserved).  T4 tests: 3 gates
  (`test_registry_nn_mlp_dispatch.py`) — `registry.save` +
  `registry.load` round-trip for `NnMlpModel` (fifth model family
  extending the Stage 9 AC-2 suite), sidecar `type == "nn_mlp"` gate,
  and a source-inspection test pinning that `train.py::_cli_main`
  contains the `isinstance(model_cfg, NnMlpConfig)` branch.

- **T5 — `plots.loss_curve` helper + train-CLI integration test.**
  `bristol_ml.evaluation.plots.loss_curve(history, *, title="Training
  loss", ax=None) -> Figure` renders the train + validation curves
  with the Okabe-Ito palette (train `OKABE_ITO[1]` orange, validation
  `OKABE_ITO[2]` sky blue — matches the Stage 6 `forecast_overlay` fix
  that skipped `OKABE_ITO[2]` for the Actual series so the facilitator's
  eyes learn "orange is model, blue is truth").  Accepts the `ax=` seam
  so the notebook can compose it inside a larger matplotlib figure
  (Stage 6 D5 contract).  `evaluation/CLAUDE.md` and
  `evaluation/plots.py::__all__` both extended.  T5 tests: 4 gates
  (`test_plots_loss_curve.py`) + `test_train_cli_registers_nn_mlp.py`
  integration test that runs the full `_cli_main` pipeline with
  `model=nn_mlp model.hidden_sizes=[4] model.max_epochs=3
  model.batch_size=64 model.device=cpu` and asserts exit 0, the
  per-fold banner, the `Registered run_id:` line, exactly one run in
  a tmp registry dir, and `type == "nn_mlp"` on the sidecar.

- **T6 — Notebook + layer docs.**
  `notebooks/10-simple-nn.ipynb` — eleven-cell thin demo:
  bootstrap → data load (last 60 days of the `weather_only` cache) →
  live-demo fit with the `epoch_callback` seam → static
  `plots.loss_curve()` render (AC-3 static half) → three-way
  `harness.evaluate` comparison (naive + linear + nn_mlp) on a narrow
  rolling-origin window → closing Stage 11 forward pointer.  The
  notebook pins `device="cpu"` end-to-end for CI determinism; the
  live-plot cell uses `IPython.display.display(handle)` +
  `clear_output(wait=True)` so the models layer never imports
  IPython.  Generator at `scripts/_build_notebook_10.py` follows the
  Stage 8 / Stage 7 `md` / `code` helper pattern; three-step regen
  is `uv run python scripts/_build_notebook_10.py && jupyter nbconvert
  --execute --to notebook --inplace notebooks/10-simple-nn.ipynb &&
  ruff format notebooks/10-simple-nn.ipynb` (plan D9 of Stage 8
  carried forward).
  `docs/architecture/layers/models-nn.md` (~400 lines) — the new
  sub-layer contract covering module shape, the five-site dispatch
  story, the public interface of `NnMlpModel`, the D7′
  reproducibility contract (device-split NFR-1), the D5 single-joblib
  envelope with full field annotation, D8 cold-start, D9 10 %-tail
  early stopping + best-epoch restore, D11 device auto-select, the
  Stage 11 extraction seam (D10), and an explicit "not shipped" note
  enumerating the X5 / X6 / X7 deferrals.  The parent
  `docs/architecture/layers/models.md` remains authoritative for the
  `Model` protocol + `ModelMetadata` + joblib IO shared by every
  family.  `src/bristol_ml/models/nn/CLAUDE.md` — module-local guide
  naming the five PyTorch-specific gotchas that are not obvious from
  the other model families: (i) `_NnMlpModule` must be module-level
  for joblib pickleability, (ii) `torch` is imported lazily inside
  function bodies so `bristol_ml.models.nn` import during test
  collection does not pay the ~0.5–1 s torch-load cost, (iii) scaler
  buffers are registered at module construction with placeholder
  zeros/ones *then* overwritten by `fit()`'s fitted statistics — so
  `load_state_dict(strict=True)` does not drop them, (iv)
  `torch.load(…, weights_only=True, map_location="cpu")` is spelled
  explicitly even though `weights_only=True` is the 2.6+ default,
  (v) single-joblib envelope — `state_dict_bytes` inside, no sibling
  `.pt` or `.json` files.

- **T7 — Stage hygiene.**  This retro; `CHANGELOG.md` `[Unreleased]`
  Stage 10 bullets under `### Added`; `docs/architecture/README.md`
  module catalogue row for `models/nn/` (H-3 / Scope diff X3); H-2
  verified (Stage 9 retro's "Next" pointer already names Stage 10 with
  accurate scope — the prediction of pressure on the hyperparameter-bag
  shape and final-fold-model semantic under early stopping both came
  true and are documented under §Surprises); H-1 flagged for the
  human at PR review (DESIGN §6 batched edit covering Stages 1–10;
  deny-tier for the lead); H-4 (dispatcher-duplication ADR)
  re-deferred — Stage 10 did not add a third dispatcher site, it
  extended the two existing ones with one new branch each, so the
  Stage 7/8/9 carry-over remains unchanged; H-5 / OQ-A closed `NO` at
  2026-04-24 Ctrl+G.  Plan moved from
  `docs/plans/active/10-simple-nn.md` to
  `docs/plans/completed/10-simple-nn.md` as the final commit action.

## Design choices made here

Plan §1 D1–D12 plus NFR-1 / NFR-3 / NFR-5 / NFR-6 and H-1..H-5.  The
twelve decisions plus the five housekeeping carry-overs were approved
at Ctrl+G review 2026-04-24 with three human amendments folded in
pre-Phase-2 (D1 cu128 wheel index pin; D7′ re-scoped to cover the
CUDA / MPS paths; D11 auto-select across CUDA > MPS > CPU) and one
mid-T2 trim (the Dockerfile pre-warm clause dropped once the
persistent `.venv` mount on `/workspace` made per-session install
cost a non-issue).  Key points and why they bind:

- **D1 — `torch>=2.7,<3` via a cu128 wheel index on Linux, PyPI
  elsewhere; no Dockerfile change.**  The Blackwell / RTX 5090 dev
  host requires the cu128 track (PyTorch's first stable sm_120
  wheels).  `[tool.uv.sources]` + `[[tool.uv.index]] name =
  "pytorch-cu128"` under `marker = "sys_platform == 'linux'"` pins
  torch to the CUDA wheel on Linux and lets the PyPI wheel (which
  ships MPS support) resolve elsewhere.  The persistent `.venv`
  mount means a one-time `uv sync` per container image brings torch
  into the venv; subsequent sessions reuse it.  Reaching outside the
  Docker sandbox to pre-warm the wheel at image-build time was
  dropped mid-T2 as risk without commensurate benefit.
- **D2 — `src/bristol_ml/models/nn/` package with `mlp.py`; five
  dispatch sites touched.**  Codebase-map S3 anticipated this.  Only
  **four** sites actually grow (one on each of `conf/_schemas.py`,
  `conf/model/`, `train.py::_cli_main`, `registry/_dispatch.py`); the
  harness stays model-agnostic (D2 clause iv) and
  `benchmarks.compare_on_holdout` stays unchanged (the registry
  leaderboard is how `nn_mlp` competes against the others at
  Stage 10, not the hard-wired three-way chart).
- **D3 — 1×128 ReLU + Adam lr=1e-3 + batch_size=32 + max_epochs=100 +
  patience=10.**  Smallest defensible MLP on tabular hourly demand.
  The Scope Diff tagged the specific values `PLAN POLISH`; the lead's
  disposition was to lock them and rely on the Pydantic defaults
  round-trip as the single binding test (D3 fields are exposed as
  Hydra knobs, so notebook-demo variation is cheap).
- **D4 — Z-score normalisation on train-set statistics per fold,
  stored inside the `nn.Module` via `register_buffer`.**  The
  alternative (a sibling joblib-pickled sklearn `StandardScaler`
  next to the `.pt`) would add a second file and a second dispatch
  step on `load`; `register_buffer` puts the scalers inside
  `state_dict()` where the strict-mode round-trip catches any drift.
- **D5 (revised mid-T3) — Single joblib envelope.**  The research
  draft and the pre-implementation plan both assumed the two-file
  layout (`model.pt` + `hyperparameters.json`) with the registry
  passing a **directory** path.  The actual Stage 9 registry passes
  `artefact/model.joblib` as a **file** path (`_fs.py::_atomic_write_run`
  hard-codes the filename).  The revision wraps the `state_dict`
  bytes (`torch.save(state_dict, BytesIO).getvalue()`) inside a plain
  dict envelope and writes that envelope through
  `bristol_ml.models.io.save_joblib`.  The `state_dict_bytes` stay
  a plain dict of tensors, so `torch.load(…, weights_only=True)`
  still applies on the inner payload; joblib around the outer
  envelope matches every other model family's serialisation idiom.
  The layer doc names this deviation explicitly; the retro spells
  out that changing the registry to pass a directory path would be
  out-of-scope for Stage 10 (modifies Stage 9's AC-2 surface by
  inference).
- **D6 — `loss_history_` attribute + `plots.loss_curve()` helper +
  `epoch_callback` seam.**  AC-3 requires surfacing, not persistence.
  The three-piece surface makes the live-plot notebook cell a
  callback (not a fork of `fit()`), and keeps the models layer
  ignorant of IPython.  NFR-4 (auto-save loss-curve PNG to registry
  run dir) was the Scope Diff's single highest-leverage cut —
  coupling the plots module to the registry save path adds a module
  dependency and one integration assertion for no AC gain.
- **D7′ — Four-stream seed + `cudnn.deterministic = True` /
  `cudnn.benchmark = False` on CUDA; `use_deterministic_algorithms
  (True)` off.**  Intent AC-2's GPU carve-out ("within the
  constraints of non-deterministic GPU operations") closes OQ-A in
  the negative.  The D7′ recipe gives CPU bit-identity
  (`torch.equal`) and CUDA / MPS close-match (`torch.allclose(atol=
  1e-5, rtol=1e-4)`) without costing throughput on the Blackwell
  host.  NFR-1 split into a CPU path and a `@pytest.mark.gpu` CUDA
  path; the GPU test skips when `torch.cuda.is_available()` is
  false.
- **D8 — Cold-start per fold.**  Ash & Adams NeurIPS 2020 warm-start
  result + the harness's existing re-entrancy contract make this a
  one-line discipline (every `fit()` reconstructs `_NnMlpModule` and
  re-seeds).  For the 1×128 MLP default the per-fold re-fit is
  seconds, well inside AC-5's "reasonable time on a laptop CPU"
  envelope.
- **D9 — Patience-based early stopping with best-epoch weight restore
  on a 10 % contiguous val tail.**  Time-series-safe (the tail is
  taken from the end of the train slice by index, not random-split).
  Best-epoch restore means the registered artefact is the epoch that
  patience caught, not the strictly-worse last epoch.
- **D10 — Hand-rolled loop inside `NnMlpModel._run_training_loop`
  with a named extraction seam for Stage 11.**  DESIGN §1.1
  "walkable at a meetup" says the loop should be readable in one
  scroll.  Shipping a `BaseTorchModel` ABC at Stage 10 (X7 cut)
  would bind Stage 11's design before Stage 11's requirements are
  understood.  The layer doc's "Stage 11 extraction seam" section
  names gradient clipping / LR scheduling as the trigger for the
  extraction, not an in-place addition.
- **D11 — `_select_device(preference)` helper resolves `"auto"` in
  order CUDA > MPS > CPU; pinned values honoured; unknown values
  raise.**  Scope Diff row X5 (originally tagged `PREMATURE
  OPTIMISATION` against a CPU-only D11) was re-opened at the
  2026-04-24 Ctrl+G — the dev host is the CUDA 12.8 / Blackwell
  Dockerfile, Apple Silicon laptops with MPS are another common
  Python ML environment, and hard-coding CPU would leave orders of
  magnitude of free speed on the table for no AC gain.  The helper
  is module-level so unit tests can monkeypatch
  `torch.cuda.is_available` / `torch.backends.mps.is_available`
  without instantiating `NnMlpModel`.
- **D12 — New layer doc at `docs/architecture/layers/models-nn.md`
  (not a section on the existing `models.md`).**  Scope Diff row D12
  tagged `HOUSEKEEPING`; every shipped module layer carries both a
  `CLAUDE.md` and an `architecture/layers/` file, and the
  `evaluation/` layer doc split from models is the precedent.  The
  new file covers the concrete Stage 10 surface; the parent layer
  doc remains authoritative for protocol + metadata + joblib IO.
- **NFR-1 (device-split) — CPU bit-identity (`torch.equal`); CUDA /
  MPS close-match (`torch.allclose(atol=1e-5, rtol=1e-4)`).**  AC-2's
  GPU carve-out operationalised as a test-level split.
- **NFR-3 — Registry save/load fidelity to `atol=1e-10` on predict
  output.**  Matches Stage 9's MLflow adapter round-trip bar
  (`test_registry_run_is_loadable_via_mlflow_pyfunc_adapter`).
- **NFR-5 — Normalisation persistence covered by NFR-3.**  T8 cut as
  redundant with T3's strict-mode round-trip.
- **NFR-6 — Standalone CLI `python -m bristol_ml.models.nn.mlp --help`
  exits 0.**  Every prior model stage shipped this (§2.1.1).  The
  CLI is a schema-help surface, not a training driver — it does not
  fit a model.

## Surprises captured during implementation

- **`_fs.py` hard-codes `artefact/model.joblib` as a file path.**  The
  plan's pre-implementation §5 "On-disk artefact layout" section
  assumed the registry passed a **directory** path and described a
  two-file layout (`model.pt` + `hyperparameters.json`) per the
  domain research draft.  The first T3 test
  (`test_nn_mlp_save_writes_single_joblib_file_at_given_path`) failed
  fast: `registry.save` was calling
  `NnMlpModel.save(run_dir / "artefact" / "model.joblib")` with the
  joblib filename already appended.  Resolved mid-T3 by wrapping the
  `torch.save(state_dict, BytesIO).getvalue()` bytes inside a plain
  dict envelope and writing that through `save_joblib` — the safety
  rail (`torch.load(…, weights_only=True)` on the inner bytes) is
  unchanged.  Plan §5 and D5 rewritten in-place to name the revised
  layout; the layer doc calls out the deviation from the research
  draft so a future reader does not think a silent regression
  dropped the two-file path.  Scope Diff row D5 stays
  `RESTATES INTENT` (the intent is indifferent between one file and
  two) so no tag change was needed.
- **`_NnMlpModule` had to be module-level, not nested in
  `_make_mlp`.**  The pre-T3 implementation defined
  `class _NnMlpModule(nn.Module)` inside `_make_mlp`'s function body
  for encapsulation.  Joblib round-trip failed because the nested
  class is not pickleable by dotted path — the envelope's non-tensor
  payload (the `config_dump`, `loss_history`, etc.) pickled fine,
  but any downstream helper that inspects the module's class
  reference after `load_state_dict` rebuilt the skeleton could not
  resolve it.  Same discipline that `ScipyParametricModel`'s
  `_parametric_fn` required at Stage 8 (plan surprise S2 there).
  Resolved by promoting `_NnMlpModule` to module level.  `models/nn/
  CLAUDE.md` names this as the first of the five PyTorch gotchas so
  future teammates working on Stage 11 do not re-hit it.
- **`register_buffer` ordering inside `load_state_dict(strict=True)`.**
  `_NnMlpModule.__init__` initially registered the four scaler
  buffers *only when `fit()` called a
  `self._install_scaler_buffers(feature_mean, feature_std, …)`
  method*, on the grounds that an unfitted model has no scalers yet.
  `load_state_dict(strict=True)` on the reconstructed module then
  failed because the freshly-built skeleton had no buffers — the
  loaded bytes had all four.  Resolved by unconditionally registering
  the four buffers in `__init__` with placeholder zeros/ones; `fit()`
  overwrites them in-place with the fitted statistics.  The model is
  still "unfitted" before `fit()` (the `metadata.fit_utc` field stays
  `None`), but the buffer slots exist.  This is the third of the five
  gotchas in `models/nn/CLAUDE.md`.
- **The Scope Diff's `@minimalist` tags held up.**  X6 (gradient
  clipping, LR scheduling as configurable knobs), X7 (`BaseTorchModel`
  ABC), NFR-2 (3-min wall-clock ceiling as a regression gate), and
  NFR-4 (auto-save loss-curve PNG) were all still the right call at
  T7 with no new evidence to reopen.  X5 (`device=auto` config field)
  *was* re-opened at the 2026-04-24 Ctrl+G — the human correctly
  flagged that a CPU-only D11 leaves orders of magnitude of free
  speed on the Blackwell dev host; the Scope Diff's original
  `PREMATURE OPTIMISATION` tag rested on a CPU-only D11 which the
  human amended.  Pattern validated: the Scope Diff is a default, not
  a veto.
- **`tests/integration/test_train_cli_registers_nn_mlp.py` needed
  `model.hidden_sizes=[4] model.max_epochs=3` overrides.**  The
  default config (`[128]` × 100 epochs × patience 10) fit in ~20 s on
  the dev host's CPU (observed CPU wall-clock data point below), but
  an integration test running the full six-fold rolling-origin
  pipeline at those defaults would be marginally too slow for the
  fast-test budget.  Resolved by pinning minimal overrides
  (`hidden_sizes=[4], max_epochs=3, batch_size=64`) that still
  exercise every code path (device selection, four-stream seed,
  `_run_training_loop` with early stopping possible, save, registry
  dispatch, sidecar type, …) — the integration test completes in
  ~9 s.  Structural: the test proves the wiring, not the training
  dynamics.  AC-5's "reasonable time on a laptop CPU" is separately
  evidenced by the end-to-end CPU run below.

## AC-5 evidence (observed CPU wall-clock)

Recorded on the Stage 10 dev host at T7 with the full production
pipeline.  Command (see plan §9 exit checklist):

```
uv run python -m bristol_ml.train model=nn_mlp model.device=cpu \
    model.max_epochs=50 \
    evaluation.rolling_origin.min_train_periods=720 \
    evaluation.rolling_origin.step=1344 \
    evaluation.rolling_origin.test_len=168
```

Harness summary at the tail of the log:

> `Evaluator complete: total_folds=6 elapsed_seconds=18.278 summary=
> {'mae': {'mean': 3501.49, 'std': 924.12}, 'mape': {'mean': 0.143,
> 'std': 0.029}, 'rmse': {'mean': 4377.80, 'std': 1079.37}, 'wape':
> {'mean': 0.136, 'std': 0.027}}`

`time` reports wall-clock **22.9 s** (includes torch import + config
resolve + cache read + six folds + registry save + NESO-cache
skip-notice).  The six-fold harness itself completes in **18.3 s**.
AC-5's "reasonable time on a laptop CPU" envelope is therefore
~10× headroom over what a facilitator sees at the meetup.  The
accuracy number (mean MAE 3501 MW) is in the same ballpark as
Stage 5's linear + calendar (~2900 MW) and deliberately does not
beat it — intent §Purpose predicted this.  The stage ships the
*scaffold*, not the peak-accuracy model; Stage 11's temporal
architecture is the first NN that is expected to materially beat
the linear baseline.

NFR-2 (3-min wall-clock ceiling as a regression gate) cut per the
Scope Diff; this measurement is the informational data point the
retro records per plan T7.

## What did not change

- **`bristol_ml.models.protocol.Model`** — the five-member protocol.
  AC-1 "conforms to the Stage 4 interface" is satisfied by construction;
  `NnMlpModel` exposes `fit`, `predict`, `save`, `load`, `metadata`
  and nothing else of protocol weight.
- **`bristol_ml.registry`** — the four-verb public surface
  (`save`, `load`, `list_runs`, `describe`).  Stage 9 AC-1 is
  preserved; the only registry change is two one-line dict entries
  in `_dispatch.py`.
- **`bristol_ml.evaluation.harness`** — `evaluate` and
  `evaluate_and_keep_final_model` signatures and bodies unchanged.
  The harness is model-agnostic (calls `model.fit / .predict`); the
  new family integrates via the existing re-entrancy contract.  H5
  API-growth rule honoured (no second boolean on `evaluate`).
- **`bristol_ml.evaluation.benchmarks`** — `compare_on_holdout` stays
  naive + linear + NESO.  `nn_mlp` competes against the other
  families through the registry leaderboard (Stage 9 Demo moment),
  not through the hard-wired three-way chart (plan D2 clause iv).
- **The four existing model classes** (`NaiveModel`, `LinearModel`,
  `SarimaxModel`, `ScipyParametricModel`) — zero body changes; zero
  import additions.  The only adjacency is `_CLASS_NAME_TO_TYPE` in
  the registry's `_dispatch.py`, which already held the Stage 4 /
  Stage 7 / Stage 8 class names.
- **`pyproject.toml [project].dependencies` except `torch`** — only
  `torch>=2.7,<3` was added as a new runtime dependency.  The
  cu128 wheel index is new; the Dockerfile is not touched.
- **`Dockerfile`** — mid-T2 D1 amendment dropped the pre-warm clause;
  the persistent `.venv` mount on `/workspace` does the job.
- **`docs/intent/DESIGN.md` §6 layout tree** — H-1 batches this for
  the human at PR review (deny-tier for the lead).  Stage 10 extends
  the Stage 9 H-1 batch (Stages 1–9 already pending).
- **Dispatcher-duplication ADR (H-4)** — re-deferred.  Stage 10
  extended the two existing dispatcher sites by one branch each; it
  did not add a third.  The Stage 7/8/9 ADR-worth question about
  `_build_model_from_config` + `train.py::_cli_main` + `registry/
  _dispatch.py` duplication carries forward unchanged to Stage 11 or
  a dedicated housekeeping stage.  ADR filename still earmarked:
  `0004-model-dispatcher-consolidation.md`.

## Exit checklist outcome

All gates in plan §9 passed (one marked N/A and explained):

- [x] `uv run pytest -q` — 644 passed, 5 deselected (4 slow, 1 gpu);
  0 failures, 0 xfail.
- [x] Ruff + format + pre-commit clean.
- [x] `uv run python -m bristol_ml.models.nn.mlp --help` exits 0 and
  prints the resolved `NnMlpConfig` schema, including
  `device="auto"`.
- [x] `uv run python -m bristol_ml.train model=nn_mlp` leaves exactly
  one new `run_id` in a redirected registry directory (T5 integration
  test, verified again at T7 against the production config).
- [x] On the CUDA dev host, the `@pytest.mark.gpu` marker passes the
  NFR-1 close-match test.  CPU-only CI defers via the `addopts`
  filter (`-m 'not slow and not gpu'`).
- [N/A] `docker build` cu128-wheel caching checkpoint — the D1
  amendment dropped the Dockerfile pre-warm mid-T2, so there is
  nothing to measure at `docker build` time.  The persistent `.venv`
  mount carries the wheel across sessions; the first `uv sync`
  inside a fresh container pulls it once.
- [x] `uv run python -m bristol_ml.registry list --model-type nn_mlp`
  prints the new run.
- [x] `uv run python -m bristol_ml.registry describe <nn_mlp_run_id>`
  prints a sidecar whose `type` field is `"nn_mlp"`.
- [x] All five intent-ACs mapped to named tests in plan §4 pass.
- [x] `docs/architecture/layers/models-nn.md` exists and documents
  the four contract points (D5, D6, D7′, D10).
- [x] `src/bristol_ml/models/nn/CLAUDE.md` exists (D12).
- [x] This retrospective written, including the observed CPU
  wall-clock data point above.
- [x] `CHANGELOG.md` updated with the Stage 10 bullets under
  `[Unreleased]`.
- [x] `docs/architecture/README.md` module catalogue extended with
  the `models/nn/` row (H-3).
- [x] `docs/plans/active/10-simple-nn.md` moved to
  `docs/plans/completed/10-simple-nn.md` as the final T7 commit.
- [x] H-1 (DESIGN §6) deferred per Stage 9 precedent (flagged at PR
  review); H-2 (Stage 9 retro "Next" wording) verified; H-3
  (architecture README) actioned; H-4 (dispatcher ADR) re-deferred;
  H-5 / OQ-A resolved `NO` at Ctrl+G.

## Open questions

- **When to extract `_run_training_loop` into
  `src/bristol_ml/models/nn/_training.py` under a shared helper.**
  D10's named trigger is "the second torch-backed model (temporal
  architecture) arrives with a hand-rolled loop of its own".  The
  layer doc carries the extraction-seam comment inside
  `NnMlpModel._run_training_loop` so the Stage 11 teammate finds it
  grep-able.  Do not ship a `BaseTorchModel` ABC at Stage 10 (X7
  cut); it would bind Stage 11's design before Stage 11's
  requirements are understood.
- **When to hash the feature table.**  Stage 9 D6 deferred.  The
  Stage 9 retro's pattern still applies — Stage 17's price-model
  provenance claim may force the issue; the migration point is one
  new `feature_table_sha256` field in `SidecarFields` plus one test.
  Stage 10 did not pressure-test this.
- **Hyperparameter-bag shape discipline.**  Stage 9's retro
  predicted Stage 10 would "pressure-test the hyperparameter-bag
  shape discipline for a family whose `ModelMetadata.hyperparameters`
  will be richer than `{"fit_intercept": true}`".  Stage 10's
  `ModelMetadata.hyperparameters` carries the `NnMlpConfig.model_dump()`
  verbatim plus `seed_used` / `best_epoch` / `device_resolved`.  No
  structural test pins the convention; the Scope Diff tagged that
  `PLAN POLISH` and it stayed cut.  Stage 11 or a dedicated
  housekeeping stage can add a typed `HyperparametersBag` with a
  minimum-fields contract if notebook filtering on a learning-rate
  sweep becomes a live ask.
- **Dispatcher-duplication ADR (H-4 carry-over from Stages 7–9, now
  10).**  The Stage 7 Phase 3 review filed it as candidate ADR B1;
  Stage 10 extended the two existing dispatcher sites by one branch
  each (not a third site).  Revisit at Stage 11 (second NN family
  arriving) or via a dedicated housekeeping stage.  ADR filename
  earmarked: `0004-model-dispatcher-consolidation.md`.
- **Known drift, surfaced for the human (do not fix from within
  Stage 10):**
  - **H-1.**  `docs/intent/DESIGN.md` §6 layout tree — Stages 1–10
    additions batched.  Stage 10 adds `src/bristol_ml/models/nn/`
    (new sub-package inside `models/`), `docs/architecture/layers/
    models-nn.md`, and the `torch` runtime dependency.  Deny-tier
    for the lead; flag at PR review for a batched human edit
    covering all of Stages 1–10.  Stage 9 H-1 carries forward;
    Stage 10 extends the batch.
  - DESIGN §8 "Open-Meteo UKV 2km" claim remains incorrect (from
    Stage 2); unchanged here.

## Next

→ Stage 11 — Complex / temporal neural-network forecaster.  The
first torch-backed model with a hand-rolled loop of its own —
specifically the named trigger for the D10 extraction seam.  Expect
the Stage 11 plan to either (a) extract `_run_training_loop` + `_make_mlp`
into `src/bristol_ml/models/nn/_training.py` under a shared helper at
T1, or (b) ship the temporal model with its own loop and defer the
extraction one stage further — the decision hinges on whether the
temporal loop diverges from Stage 10's by more than "add one
optimiser kwarg".  Stage 11 is also the likely home for gradient
clipping and LR scheduling as first-class knobs (X6 cut forward), the
dispatcher-consolidation ADR (H-4), and the first honest attempt to
materially beat the Stage 5 linear baseline.  Intent §Purpose was
explicit that Stage 10 was the scaffold; Stage 11 is where the
scaffold earns its keep.
