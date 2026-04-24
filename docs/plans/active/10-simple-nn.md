# Plan — Stage 10: Simple neural network

**Status:** `approved` — Ctrl+G on 2026-04-24 accepted the decision table with three amendments (D1 → CUDA-aware torch install + Dockerfile pre-warm, D7' → re-scope seeding to cover the CUDA / MPS paths, D11 → auto-select device across CUDA / MPS / CPU). H-5 / OQ-A (use_deterministic_algorithms) resolved in the negative per the D7' re-scope. Ready for Phase 2.
**Intent:** [`docs/intent/10-simple-nn.md`](../../intent/10-simple-nn.md)
**Upstream stages shipped:** Stages 0–9 (foundation, ingestion, features, four Stage-4/7/8 models, enhanced evaluation, registry).
**Downstream consumers:** Stage 11 (complex / temporal neural network — inherits this stage's training-loop conventions and loss-logging contract), Stage 12 (serving — loads NN runs by name through the registry), Stage 18 (drift monitoring).
**Baseline SHA:** `6267cc0` (tip of `main` after Stage 9 merge via PR #7).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/10-simple-nn-requirements.md`](../../lld/research/10-simple-nn-requirements.md)
- Codebase map — [`docs/lld/research/10-simple-nn-codebase.md`](../../lld/research/10-simple-nn-codebase.md)
- Domain research — [`docs/lld/research/10-simple-nn-domain.md`](../../lld/research/10-simple-nn-domain.md)
- Scope Diff — [`docs/lld/research/10-simple-nn-scope-diff.md`](../../lld/research/10-simple-nn-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition below)

**Pedagogical weight.** Intent §Demo moment names the live train-vs-validation loss curve as the canonical "watch a neural network learn" moment for the meetup series — the facilitator points at the moment validation loss bottoms out and starts rising. Intent §Purpose is equally explicit that Stage 10's *analytical* value is small: "the model itself is likely not to beat the Stage 5 linear regression by much". The stage's load-bearing contribution is the **scaffold** — the training-loop conventions, reproducibility discipline, and registry round-trip that Stage 11's temporal architecture inherits. Every decision below is either a direct consequence of AC-1 (protocol conformance) / AC-2 (reproducibility) / AC-3 (loss curve) / AC-4 (registry round-trip) / AC-5 (laptop-CPU budget), or it is cut per the `@minimalist` scope diff.

---

## 1. Decisions for the human (resolve before Phase 2)

Twelve decision points plus five housekeeping carry-overs. Defaults below lean on the three research artefacts' recommendations, have been filtered through the `@minimalist` Scope Diff, and honour the simplicity bias in `DESIGN.md §2.2.4`. The Evidence column cites the research that *resolved* each decision. Acceptance criteria are cited where the intent supplies a hard constraint the decision operationalises; the intent's "Points for consideration" are not cited as evidence because they *pose* the decision rather than answering it.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Runtime dependency | **Add `torch>=2.7,<3` to `[project].dependencies`, resolved from the CUDA 12.8 wheel index on Linux and from PyPI elsewhere.** Configured via `[tool.uv.sources]` + a `[[tool.uv.index]] name = "pytorch-cu128"` entry pointing at `https://download.pytorch.org/whl/cu128`; `torch` is pinned to that index under `marker = "sys_platform == 'linux'"` and to PyPI otherwise. The PyPI wheel for `torch` ships MPS support on Apple Silicon out of the box, so no second index is needed for macOS. **Dockerfile also pre-warms the wheel cache**: a new `RUN --mount=type=cache,target=/home/${USER_NAME}/.cache/uv uv pip download --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<3"` step lands the ~2 GB CUDA wheel into the uv cache volume during `docker build`, so the first `uv sync` inside the running container is close to instant. The Dockerfile already targets `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` (Blackwell / RTX 5090, sm_120), and cu128 is the first PyTorch wheel track with stable Blackwell support — the torch lower bound is bumped from the research-draft's `>=2.3` to `>=2.7` accordingly. | Intent §Scope names "a small MLP" and DESIGN §8 names PyTorch as the chosen framework; no MLP = no stage. AC-5 says "no GPU requirement", not "no GPU" — training the meetup model on a laptop still works with the PyPI wheel, and on the RTX 5090 dev host the CUDA wheel is many orders of magnitude faster, which materially widens the range of experiments a facilitator can run live. Pre-warming the wheel in the Docker image means a container rebuild is not a ~5-minute wheel download; the cache mount survives image rebuilds and keeps CI/local installs consistent. | Intent AC-1 / AC-5; DESIGN §8; Dockerfile L12 (cu128 / Blackwell). Domain research §R3 (serialisation under PyTorch 2.6+ `weights_only` default; `state_dict` contract stable across 2.7). |
| **D2** | Module structure + five dispatch-site edits | **New package `src/bristol_ml/models/nn/` with `__init__.py`, `__main__.py`, and `mlp.py` (the `NnMlpModel` class plus its module-level helpers).** Five dispatch sites pick up the new family: (i) `conf/_schemas.py` gains `NnMlpConfig`; (ii) new `conf/model/nn_mlp.yaml`; (iii) `src/bristol_ml/train.py` gains an `isinstance(model_cfg, NnMlpConfig)` branch mirroring `ScipyParametricConfig`; (iv) `src/bristol_ml/evaluation/benchmarks.py` *is not* touched (NN is out of the NESO three-way benchmark at Stage 10 — the benchmark is naive + linear + NESO as shipped); (v) `src/bristol_ml/registry/_dispatch.py` gains `{"nn_mlp": NnMlpModel}` / `{"NnMlpModel": "nn_mlp"}` entries. | Intent AC-1 requires Model-protocol conformance; Stage 7 / Stage 8 precedent requires all dispatch sites touched (codebase-map S3: "forgetting one site ships a stage with the family resolvable only by default-config"). Keeping the benchmark untouched respects the Stage 4 / Stage 6 harness API-growth rule; the nn_mlp competes against the others through the registry leaderboard (Stage 9 Demo moment), not through the hard-wired `compare_on_holdout` path. | Intent AC-1 (protocol conformance); codebase map §4 / §8 (dispatch-site census + S3 warning). |
| **D3** | Default architecture | **1 hidden layer × 128 units, ReLU activation, Adam lr=1e-3, batch_size=32, max_epochs=100, patience=10, weight_decay=0.0.** Architecture knobs (`hidden_sizes: list[int]`, `activation: Literal["relu", "tanh", "gelu"]`, `dropout: float = 0.0`, `batch_size: int`, `learning_rate: float`, `max_epochs: int`, `patience: int`) exposed on `NnMlpConfig`. | Intent §Points: "one or two hidden layers, moderate width, standard activations is enough". The single-layer 128-unit default is the smallest defensible MLP on tabular hourly demand data; Adam at 1e-3 is the published default across tabular-NN baselines; batch 32 keeps a full epoch under a few seconds on a 4-core laptop. Exposing all knobs via Hydra keeps notebook-demo variation cheap. The specific values are `PLAN POLISH` per the scope diff (underconstrained by the intent); the lead's disposition is to lock them and not spend a test per field — the Pydantic schema's default-value round-trip is the only binding check. | Domain research §R6 (tabular-NN baseline defaults; Borisov 2022 + Gorishniy 2021 survey). Scope diff row D3 (PLAN POLISH; defaults must be chosen but not over-tested). |
| **D4** | Input (and target) normalisation | **Z-score on train-set statistics per fold, stored inside the `nn.Module` via `self.register_buffer(...)`.** Features and target both normalised; `predict()` inverse-transforms the target before returning. Mean/std computed once inside `fit()` on the training slice; buffers ride inside `state_dict()` and therefore round-trip cleanly through `torch.save(state_dict)`. | Intent §Points names normalisation and its save/load implication as explicitly load-bearing. `register_buffer` is the canonical PyTorch way to pin non-trainable tensors into the module graph so they persist in `state_dict` — alternatives (a sibling sklearn `StandardScaler` joblib-pickled next to the `.pt` file) add a second file and a second dispatch step on load. | Intent §Points (normalisation is load-bearing); intent AC-4 (registry round-trip). Domain research §R2 (register_buffer idiom — PyTorch docs, Gorishniy rtdl library precedent). |
| **D5** | Serialisation format | **Single joblib artefact at the registry-provided path (`artefact/model.joblib`), containing a plain dict `{"state_dict_bytes": <torch.save-to-BytesIO output>, "config_dump": <NnMlpConfig.model_dump()>, "feature_columns": tuple, "seed_used": int, "best_epoch": int, "loss_history": list[dict], "fit_utc": iso-str, "device_resolved": str}`.** Written via `bristol_ml.models.io.save_joblib` so atomic-write + parent-dir-creation match the rest of the models layer. `NnMlpModel.save(path)` assembles the dict; `NnMlpModel.load(path)` unpacks it, reconstructs the `nn.Module` skeleton from `config_dump`, materialises the `state_dict` via `torch.load(BytesIO, weights_only=True)`, and calls `load_state_dict(strict=True)`. The scaler buffers (`feature_mean` / `feature_std` / `target_mean` / `target_std`) ride inside the `state_dict` as D4 specifies — they round-trip because `register_buffer` puts them there. | **Change from research draft:** the research-draft assumed two files (`model.pt` + `hyperparameters.json`) and the registry passing the artefact directory. **The actual Stage 9 registry passes `artefact/model.joblib` as a file path** (`_fs.py::_atomic_write_run` hard-codes the filename); refactoring it to pass a directory is out of scope for Stage 10. A single joblib artefact preserves D5's functional goals: (1) the `state_dict` is a plain dict of tensors, so pickling it through joblib does **not** carry the `torch.save(nn.Module)` coupling problem domain research §R3 flagged — no class reference or module topology is pickled; (2) reconstruction is deterministic from `config_dump`; (3) `torch.load(weights_only=True)` on the inner `BytesIO` still applies the 2.6+ safety rail because we `torch.save` the `state_dict` (to bytes) and `torch.load(..., weights_only=True)` it back. Security surface is unchanged — Stage 9 already documents the "only load artefacts we wrote ourselves" rule; Stage 12 owns the `skops.io` graduation regardless. The layer doc (D12) names this deviation explicitly as a known trade-off against the research-draft's two-file layout. | Domain research §R3 (state_dict-bytes through `torch.load(weights_only=True)` retains the safety rail); codebase precedent §1 (Stage 9 registry file-path contract, `registry/__init__.py::load`). |
| **D6** | Loss-curve surfacing | **Fitted model carries `self.loss_history_: list[dict[str, float]]` populated per-epoch with keys `{"epoch": int, "train_loss": float, "val_loss": float}`.** A new helper `bristol_ml.evaluation.plots.loss_curve(history, *, ax=None) -> Figure` renders the train + validation curve with the Okabe-Ito palette. `fit()` also accepts an optional `epoch_callback: Callable[[dict[str, float]], None] \| None = None`; the notebook's live-demo moment passes a callback that `clear_output(wait=True)` + redraws the figure in a display handle. No live-plot logic in the models layer — the callback is the seam. | Intent AC-3 ("loss curve is produced by the training loop itself and is available as a plot without additional wiring") requires surfacing, not persistence. The `loss_history_` attribute plus `plots.loss_curve()` together satisfy AC-3 without coupling the models layer to matplotlib at import time. The `epoch_callback` shape means notebook X4 (live plot) is one callback, not a fork of `fit()`. **NFR-4 is cut per the Scope Diff** — no automatic PNG to the registry run dir; the layer doc says "render on demand". | Intent AC-3; intent §Demo moment (live loss curve in notebook). Domain research §R4 (dual-mode loss plot; live-plot idioms in notebooks). Scope diff single highest-leverage cut (NFR-4 drop). |
| **D7'** | Reproducibility discipline (re-scoped for CUDA / MPS after D1 + D11 human amendments) | **Seed four streams at the top of `fit()`, conditional on device: `torch.manual_seed(seed)` (covers CPU + default CUDA + MPS generators), `torch.cuda.manual_seed_all(seed)` (explicit, and a no-op on non-CUDA devices), `numpy.random.seed(seed)`, `random.seed(seed)`. `DataLoader(num_workers=0, generator=torch.Generator().manual_seed(seed))` on every dataloader constructed inside `fit()`. On the CUDA path, additionally set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` — the cheap CUDA-side equivalents of `use_deterministic_algorithms(True)` that catch the dominant sources of cuDNN nondeterminism without forcing every op onto a deterministic kernel. `torch.use_deterministic_algorithms(True)` itself stays OFF** — Intent AC-2 explicitly carves out "within the constraints of non-deterministic GPU operations", so the stricter all-op determinism guarantee is beyond the spec bar and would cost real throughput on the Blackwell host. Per-fold seed stays `fold_seed = config.project.seed + fold_index` (deterministic, explicit). NFR-1 (bit-identical `state_dict`) therefore holds on the **CPU path only**; on CUDA / MPS the weaker "numerically close" bar applies (`torch.allclose(atol=1e-5)` on `predict` output of two seeded back-to-back runs). | Intent AC-2 is explicit about the GPU carve-out ("within the constraints of non-deterministic GPU operations"). Adding `cudnn.deterministic = True` / `cudnn.benchmark = False` is the idiomatic PyTorch recipe for "as reproducible as it reasonably gets on CUDA" — recommended in the PyTorch reproducibility docs and zero-cost on CPU / MPS (the flags are silently ignored). The four-stream seed (`torch` + `torch.cuda` + `numpy` + `random`) guards every RNG the training loop or downstream helpers are likely to touch. OQ-A (promoting to `use_deterministic_algorithms(True)`) is resolved in the negative: it would trade throughput for a guarantee the intent does not require. | Intent AC-2 (GPU carve-out); Dockerfile target (CUDA 12.8 / Blackwell). Domain research §R1 (multi-stream seed helper; cuDNN deterministic + benchmark=False recipe; `use_deterministic_algorithms` flagged as optional — cost > benefit at AC-2's bar). |
| **D8** | Rolling-origin per-fold strategy | **Cold-start per fold: fresh weight init on every `fit()` call, no state carry-over between folds.** Per-fold determinism via `fold_seed = config.project.seed + fold_index` (D7'). The harness's existing re-entrancy contract (calling `model.fit(...)` between folds discards prior state) is honoured by construction. | Intent §Points: "Re-training the network on every fold is expensive; there may be room to reuse computation across folds" — the research resolves this in favour of cold-start (Ash & Adams NeurIPS 2020: warm-starting rolling-origin folds leaks information and hurts generalisation). For a 1-hidden-layer 128-unit MLP on hourly demand data the per-fold re-fit budget is bounded (~10 s × 8 folds = ~80 s on a 4-core laptop) — an order of magnitude inside the "reasonable time on a laptop CPU" AC-5 envelope. | Domain research §R5 (Ash & Adams 2020 cold-start result); intent AC-5. |
| **D9** | Early stopping | **Patience-based on validation loss, with best-epoch weight restore at the end of `fit()`.** Validation slice is an internal 10 % tail of the train set per fold (not the harness's test fold). If validation loss does not improve for `patience` consecutive epochs, stop; restore the state_dict snapshot from the best epoch. `loss_history_` records every epoch actually run. | Intent §Scope names "early-stopping and checkpointing tied to the registry from Stage 9" as in scope. Best-epoch restore is the idiomatic shape — otherwise the registered artefact is the *last* epoch's weights, which is strictly worse than the epoch patience caught. A 10 % internal validation tail is the minimum that does not force a second configurable split. | Intent §Scope (early stopping in scope). Domain research §R6 (patience + best-epoch restore as tabular-NN default). Scope diff row D9 (RESTATES INTENT). |
| **D10** | Training-loop ownership | **Hand-rolled training loop as `NnMlpModel._run_training_loop(self, X_train, y_train, X_val, y_val, *, epoch_callback)` — a private method inside `mlp.py`. Explicit Stage 11 extraction seam flagged in the layer doc: when Stage 11 arrives with a second torch model, extract to `src/bristol_ml/models/nn/_training.py` under a shared helper.** | Intent §Points: "How much of the training loop lives inside the model class... separate loops per model are clearer but repeat themselves." A hand-rolled loop at Stage 10 is walkable at a meetup (DESIGN §1.1), and the Stage 11 refactor trigger is one named call-site. Shipping a `BaseTorchModel` abstraction now is X7 `PREMATURE OPTIMISATION` per the scope diff — it binds Stage 11's design before Stage 11's requirements are understood. | Intent §Points (loop ownership is the central scope question); domain research §R7 (hand-rolled for pedagogy). Scope diff row X7 (cut). |
| **D11** | Device | **Auto-select a single device per `fit()` call with a small helper `_select_device(preference: str) -> torch.device` in `mlp.py`. Resolution order when `preference == "auto"`: (1) `torch.cuda.is_available()` → `cuda`; (2) `torch.backends.mps.is_available()` → `mps`; (3) `cpu`.** The resolved device is logged at INFO. A new Hydra-exposed field `device: Literal["auto", "cpu", "cuda", "mps"] = "auto"` on `NnMlpConfig` lets a facilitator pin the device explicitly (useful for "force CPU in this test to get bit-identity", or "run on CUDA even if MPS is also detected"). Distributed / multi-GPU are still out of scope (intent §Out of scope); this is single-device selection only. The resolved device string is persisted in `hyperparameters.json` (D5) as provenance, and in `ModelMetadata.hyperparameters["device"]` so the registry leaderboard surfaces it. | Intent §Points: "if the project is run on a laptop for meetups, CPU-only is fine" — the auto path makes that true by construction on a laptop without CUDA. But the project's dev host is the CUDA 12.8 / Blackwell Dockerfile (D1), and Apple Silicon laptops with MPS are another common Python ML environment; hard-coding CPU would leave orders-of-magnitude of free speed on the table for no AC gain. The single-device-only restriction keeps intent §Out of scope (distributed / multi-GPU) honoured. Scope Diff row X5 (previously tagged `PREMATURE OPTIMISATION` against a CPU-only D11) is re-opened and un-cut by the human Ctrl+G — the lead updates the Scope Diff cross-reference below. | Intent §Out of scope (distributed only); intent §Points; D1 (CUDA-aware install makes auto-detect load-bearing on the dev host). Domain research §R2 (`torch.device` idioms; MPS availability check). |
| **D12** | Layer documentation | **One new file `docs/architecture/layers/models-nn.md`** (rather than a section on the existing `models.md`) covering: module layout, five-site dispatch story, the `fit` / `predict` / `save` / `load` signatures of `NnMlpModel`, the reproducibility contract (D7'), cold-start per fold (D8), early stopping + best-epoch restore (D9), the serialisation format (D5), the loss-curve surfacing contract (D6) + the `epoch_callback` seam, the Stage 11 extraction seam (D10), and a "not shipped" note listing X5 / X6 / X7 deferrals. One new `src/bristol_ml/models/nn/CLAUDE.md` covering the module's concrete Stage-10 surface. | Every shipped module layer carries both a `CLAUDE.md` and an architecture/layers file; Scope diff row D12 tags this `HOUSEKEEPING`. A peer file (not a models.md section) keeps the models-layer doc small and makes the nn-submodule discoverable in its own right. | Scope diff row D12 (HOUSEKEEPING). Codebase precedent §5 (every shipped module layer has a file under `docs/architecture/layers/`; the `evaluation/` layer doc split from models is the pattern). |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-1** | Seeded-run reproducibility, device-aware. **On CPU:** two back-to-back `fit(seed=0)` runs of `NnMlpModel` on the same data produce identical `state_dict` tensors and identical `predict` output under `torch.equal`. **On CUDA or MPS:** the same two runs produce `predict` output that matches under `torch.allclose(atol=1e-5, rtol=1e-4)` — the intent's AC-2 carve-out for "non-deterministic GPU operations" is honoured, but the cudnn-deterministic + benchmark=False recipe (D7') keeps the drift within tolerance. The CPU-path test pins `device="cpu"` on the config; a second `@pytest.mark.gpu` test (skipped when `torch.cuda.is_available() is False`) exercises the CUDA close-match path so the GPU contract is not untested. | Intent AC-2 (including the GPU non-determinism carve-out). Scope diff row NFR-1 (RESTATES INTENT). |
| **NFR-3** | Registry save/load fidelity. `registry.save(model, ...)` then `registry.load(run_id)` produces a model whose `predict` output matches the original to `torch.allclose(atol=1e-10)`. | Intent AC-4. Matches the Stage 9 MLflow adapter round-trip bar. Scope diff row NFR-3 (RESTATES INTENT). |
| **NFR-5** | Normalisation persistence. Scaler mean/std are in `state_dict()` — covered by NFR-3 at the predict-output level; no dedicated buffer-equality unit test (T8 cut per Scope diff). | Intent AC-4; intent §Points. Scope diff row NFR-5 (RESTATES INTENT) + row T8 (cut as redundant with NFR-3 / T2). |
| **NFR-6** | Standalone CLI. `python -m bristol_ml.models.nn.mlp --help` exits 0 and prints the resolved `NnMlpConfig` schema. | DESIGN §2.1.1. Every prior model stage shipped this (Stage 7, 8 precedent). Scope diff row NFR-6 (RESTATES INTENT). |

**NFR-2 (3-min wall-clock ceiling) is cut as a test / gate.** The requirements-analyst flagged it tentative-unverified; AC-5 says "reasonable time on a laptop CPU" without naming a figure. The stage retro (`docs/lld/stages/10-simple-nn.md`) records the *observed* wall-clock on the author's hardware as a comparability data point — a note, not a regression guard. Scope diff row NFR-2 (PREMATURE OPTIMISATION) cut.

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` layout tree — Stage 10 adds `src/bristol_ml/models/nn/` (new sub-package inside `models/`), `docs/architecture/layers/models-nn.md`, and the `torch` runtime dependency. | **Defer to human-led DESIGN §6 edit per Stage 9 Ctrl+G precedent** ("§6 tree is not meant to be comprehensive, only structural layout. Keeping intent up-to-date with implementation at a file-by-file level seems foolish."). The new sub-package is a nested detail; if the human decides a models-layer entry needs updating to mention the nn/ submodule, that is a separate main-session edit. Lead MUST NOT touch §6 unilaterally (deny-tier). |
| **H-2** | Stage 9 retro "Next" pointer to Stage 10 — confirm wording is current. | **Verify at T7 hygiene.** |
| **H-3** | `docs/architecture/README.md` module catalogue — add `models/nn/` row pointing at the new layer doc. | **Edit at T7 hygiene.** Scope diff row X3 (HOUSEKEEPING). |
| **H-4** | Dispatcher-duplication ADR (Stage 7 / 8 / 9 carry-over). Stage 10 adds one more entry to `registry/_dispatch.py`'s two dicts and one branch to `train.py`'s `isinstance` cascade. The fifth model family. | **Re-defer to Stage 11 or a dedicated housekeeping stage.** Stage 10 does not introduce a third dispatcher site; it extends the two that already exist. An ADR that proposes a single registry-driven dispatcher should land before Stage 11's second neural model multiplies the branching further. Not blocking for Stage 10. |
| **H-5** | ~~Open question — **OQ-A** (D7')~~: whether to include `use_deterministic_algorithms(True)` behind a config flag (defaulted off) for future-proofing. | **Resolved at 2026-04-24 Ctrl+G — NO.** Intent AC-2's explicit GPU-nondeterminism carve-out plus the D7' cudnn recipe close this question without a config flag. May be re-opened if a future stage surfaces a reproducibility failure the current recipe cannot meet. |

### Resolution log

- **Drafted 2026-04-23** — pre-human-markup. All decisions D1–D12 are proposed defaults. Awaiting Ctrl+G review of the twelve decisions, the NFR list, the Scope Diff dispositions, and the H-1 / H-5 carry-overs.
- **Amended 2026-04-24 (Ctrl+G approved)** — three human amendments folded in: (1) **D1** — torch is now installed from the CUDA 12.8 wheel index on Linux and from PyPI elsewhere; the Dockerfile gains a pre-warm step to cache the cu128 wheel at image-build time; lower bound bumped `>=2.3 → >=2.7` for Blackwell support. (2) **D7'** — reproducibility recipe re-scoped to cover CUDA / MPS: four-stream seed + `cudnn.deterministic = True` / `cudnn.benchmark = False` on CUDA; NFR-1 split into CPU bit-identity (`torch.equal`) and GPU / MPS close-match (`torch.allclose(atol=1e-5)`). (3) **D11** — auto-select across CUDA > MPS > CPU via `_select_device(preference)` helper; new `device` field on `NnMlpConfig`. Single-device only — distributed / multi-GPU stay out of scope. **OQ-A (H-5)** closed: `torch.use_deterministic_algorithms(True)` remains off, justified by intent AC-2's GPU carve-out and the D7' cudnn recipe. Plan status: `approved` — ready for Phase 2.

### Decisions and artefacts explicitly **not** in Stage 10 (Scope Diff cuts)

- **NFR-4** (auto-save loss-curve PNG to registry run directory). The Scope Diff single highest-leverage cut. AC-3 is satisfied by `loss_history_` + `plots.loss_curve()`; coupling the plots module to the registry save path adds a module dependency and one integration assertion for no AC gain.
- ~~**X5** (`device=auto` config field)~~ — **re-opened and un-cut by the 2026-04-24 Ctrl+G.** D11 now specifies auto-select across CUDA / MPS / CPU; the Scope Diff's original `PREMATURE OPTIMISATION` tag rested on a CPU-only D11 which the human amended. The field graduates into the config schema (§5).
- **X6** (gradient clipping, LR scheduling as configurable knobs) — `PREMATURE OPTIMISATION`; neither named in any AC nor in intent §Points. Requirements analyst flags both as "additionally inferred out of scope."
- **X7** (`BaseTorchModel` abstract base ready for Stage 11 extraction) — `PREMATURE OPTIMISATION`; D10 already flags the extraction seam, and shipping the base class now binds Stage 11's design before Stage 11's requirements are known.
- **X2** (new ADR "Why torch and not flax/sklearn MLP") — `PLAN POLISH`; DESIGN §8 already records the PyTorch choice; no new decision is being made.
- **X8** (model card richer than registry captures) — `PLAN POLISH`; no AC names additional metadata beyond `ModelMetadata` + `run.json` fields.
- **T8** (normalisation buffer round-trip unit test) — `PLAN POLISH`; covered by T2 (registry round-trip predicts `atol=1e-10`, which implicitly exercises scaler-buffer round-trip).

---

## 2. Scope

### In scope

Transcribed from `docs/intent/10-simple-nn.md §Scope`:

- **A small MLP model conforming to the Stage 4 interface** (`Model` protocol: `fit`, `predict`, `save`, `load`, `metadata`) with architecture parameters (layer sizes, activation, dropout) exposed through Hydra configuration.
- **A training loop with loss logging and validation-set monitoring** — hand-rolled inside `NnMlpModel._run_training_loop`; per-epoch entries appended to `self.loss_history_`.
- **Early-stopping and checkpointing tied to the registry from Stage 9** — patience-based early stopping with best-epoch restore (D9); registration through the Stage 9 four-verb surface with a `nn_mlp` entry in `registry/_dispatch.py`.
- **A notebook that trains the model, plots train-vs-validation loss curves live, and compares predictions against prior models** — `notebooks/10-simple-nn.ipynb`. Live curve via the `epoch_callback` seam (D6 / X4).

### Out of scope (do not accidentally implement)

Transcribed from `docs/intent/10-simple-nn.md §Out of scope`, plus items surfaced by discovery and the Scope Diff:

- **Temporal architectures** (recurrent, convolutional, transformer) — Stage 11.
- **Hyperparameter search** (random / grid / Optuna / Ray Tune).
- **Distributed or multi-GPU training.**
- **Any model larger than a small MLP** — specifically, the default is 1 hidden layer × 128 units; a `hidden_sizes: [128, 64]` override is allowed by the schema, but architectures over ~100k parameters are out of the demo budget.
- **Distributed or multi-GPU training** (intent §Out of scope, reaffirmed). D11's auto-select is **single-device only** — it picks the first available of CUDA / MPS / CPU; it does not shard across GPUs, does not stripe across processes, and does not use DDP / FSDP.
- **Gradient clipping, LR scheduling as configurable knobs** (X6 cut).
- **A `BaseTorchModel` abstract base class** (X7 cut — Stage 11 owns the extraction).
- **Automatic saving of the loss-curve PNG into the registry run directory** (NFR-4 cut).
- **A new ADR on framework choice** (X2 cut — DESIGN §8 suffices).
- **Ensembling, model quantisation, or export for deployment** (intent §Out of scope, explicitly deferred).
- **A `NnMlpModel` contribution to the hard-wired `benchmarks.compare_on_holdout(...)` three-way chart** — the registry leaderboard is how `nn_mlp` competes against the other families at Stage 10; D2 clause (iv).

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/10-simple-nn.md`](../../intent/10-simple-nn.md) — the contract; 5 ACs and 8 Points bullets.
2. [`docs/lld/research/10-simple-nn-requirements.md`](../../lld/research/10-simple-nn-requirements.md) — US-1..US-N, AC evidence table, NFR list, OQs.
3. [`docs/lld/research/10-simple-nn-codebase.md`](../../lld/research/10-simple-nn-codebase.md) — dispatch-site census + hazards.
4. [`docs/lld/research/10-simple-nn-domain.md`](../../lld/research/10-simple-nn-domain.md) — §R1–§R8 (PyTorch reproducibility, register_buffer, state_dict, loss-curve idioms, cold-start per fold, tabular-NN defaults, hand-rolled loop pedagogy, expected accuracy against linear baseline).
5. [`docs/lld/research/10-simple-nn-scope-diff.md`](../../lld/research/10-simple-nn-scope-diff.md) — the `@minimalist` critique; every cut below is listed there.
6. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
7. `src/bristol_ml/models/protocol.py` — the `Model` protocol the new class conforms to (AC-1).
8. `src/bristol_ml/models/scipy_parametric.py` — the closest structural precedent (module-level helpers pickleable; hyperparameters carry metadata; standalone CLI).
9. `src/bristol_ml/evaluation/harness.py` — the harness contract the new model is plugged into; re-entrancy requirement for cold-start per fold.
10. `src/bristol_ml/evaluation/plots.py` — the pattern for the new `loss_curve(history)` helper (Okabe-Ito palette, `ax=` composability contract).
11. `src/bristol_ml/registry/_dispatch.py` — the two dicts that gain one key each.
12. `src/bristol_ml/registry/__init__.py` — the `save` / `load` entry points; `load` dispatches via `_dispatch.class_for_type`.
13. `src/bristol_ml/train.py:196-295` — the `isinstance(model_cfg, ScipyParametricConfig)` branch pattern the new branch mirrors.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five criteria are copied verbatim from `docs/intent/10-simple-nn.md §Acceptance criteria`, then grounded in one or more named tests.

- **AC-1.** "The MLP conforms to the Stage 4 interface, with the training loop hidden behind `fit`."
  - Tests:
    - `test_nn_mlp_is_model_protocol_instance` — `isinstance(NnMlpModel(config), bristol_ml.models.Model)` is `True`.
    - `test_nn_mlp_fit_predict_round_trip_on_tiny_fixture` — `fit(X, y, seed=0)` then `predict(X_new)` returns a `pd.Series` of the right length on a 4-column, 2-week fixture.
- **AC-2.** "Training is reproducible given a seed (within the constraints of non-deterministic GPU operations)."
  - Tests:
    - `test_nn_mlp_seeded_runs_produce_identical_state_dicts` (T1) — two `NnMlpModel(config).fit(X, y, seed=0)` calls back-to-back produce `state_dict`s that compare equal under `torch.equal` for every parameter tensor; predict outputs also identical under `torch.equal`.
    - `test_nn_mlp_different_seeds_produce_different_state_dicts` — regression guard that the seed actually flows through (otherwise T1 would trivially pass).
- **AC-3.** "The loss curve is produced by the training loop itself and is available as a plot without additional wiring."
  - Tests:
    - `test_nn_mlp_fit_populates_loss_history_per_epoch` (T4) — `len(model.loss_history_) == actual_epochs_run`; each entry has keys `{"epoch", "train_loss", "val_loss"}`; all values are finite floats.
    - `test_plots_loss_curve_renders_figure_from_history` (T7) — `loss_curve(history)` returns a matplotlib `Figure` without raising; one axis carries a "train" line and one carries a "val" line.
    - `test_nn_mlp_fit_invokes_epoch_callback_when_provided` — exercises the live-plot seam (D6).
- **AC-4.** "Save/load through the registry round-trips cleanly, including the fitted weights."
  - Tests:
    - `test_registry_save_nn_mlp_model_via_protocol` (T2) — instantiate via Hydra config, fit on a small fixture, call `registry.save(model, metrics_df, feature_set=..., target=...)`, round-trip via `registry.load(run_id)`, assert `torch.allclose(original.predict(X), loaded.predict(X), atol=1e-10)`.
    - `test_registry_list_runs_includes_nn_mlp_type` — one extra assertion that the `type` field is `"nn_mlp"` on the sidecar (covered by the extended `test_registry_list_filter_by_model_type` from Stage 9's test surface).
    - `test_nn_mlp_save_and_load_round_trips_state_dict_and_hyperparameters` — unit-level parallel of T2 that does not go through the registry; exercises `NnMlpModel.save(path)` + `NnMlpModel.load(path)` directly so a registry regression does not hide a model-serialisation bug.
- **AC-5.** "Training on the project's data completes in a reasonable time on a laptop CPU (no GPU requirement)."
  - Evidence: the default config (D3) on the resolved `features/weather_only` feature table through the `evaluation.rolling_origin` splitter completes in under **~3 minutes** on a 4-core laptop at the time of PR (observed, not gated). Recorded in the Stage 10 retro (`docs/lld/stages/10-simple-nn.md`) as an informational data point. NFR-2 cut per Scope Diff; no `@pytest.mark.slow` wall-clock gate.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_nn_mlp_early_stopping_terminates_before_max_epochs_on_plateau` (T3) — construct a synthetic dataset where val loss plateaus early; assert `len(model.loss_history_) < max_epochs` and that the restored weights match the epoch with the lowest val loss.
- `test_nn_mlp_fit_uses_cold_start_per_fold_when_called_repeatedly` — two back-to-back `fit(X, y, seed=42)` calls produce identical `state_dict`s; a `fit(X', y', seed=42)` on a different training slice produces a different `state_dict` (cold-start contract).
- `test_train_cli_registers_nn_mlp_final_fold_model` (T5) — full `python -m bristol_ml.train model=nn_mlp` pipeline end-to-end on the warm feature-table fixture; asserts exactly one new `run_id` in a tmp registry dir and that `registry.list_runs(model_type="nn_mlp")` returns it.
- `test_nn_mlp_is_dispatched_by_train_cli_isinstance_branch` (T6) — structural test that `train.py`'s isinstance-cascade handles `NnMlpConfig` (mirrors the Stage 8 `ScipyParametricConfig` dispatch test).

**Total shipped tests: 11** (two AC-1, two AC-2, three AC-3, three AC-4, three D-derived).

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/models/nn/
├── __init__.py          # exports: NnMlpModel
├── __main__.py          # `python -m bristol_ml.models.nn.mlp --help` entrypoint
├── mlp.py               # NnMlpModel class + module-level helpers + _run_training_loop
└── CLAUDE.md            # module guide (D12)
```

`_training.py` is **not** created at Stage 10; the training loop lives as a private method on `NnMlpModel` per D10. Stage 11 is the named extraction trigger.

### Config schema (addition)

```python
# conf/_schemas.py
class NnMlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str = "nd_mw"
    feature_columns: list[str] | None = None  # harness-fallback idiom, matches SarimaxConfig / ScipyParametricConfig

    # Architecture (D3)
    hidden_sizes: list[int] = Field(default_factory=lambda: [128])
    activation: Literal["relu", "tanh", "gelu"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # Optimisation (D3)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)

    # Reproducibility (D7')
    seed: int | None = None  # falls back to config.project.seed + fold_index per D8

    # Device (D11) — auto-select CUDA > MPS > CPU unless pinned
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
```

And `conf/model/nn_mlp.yaml`:

```yaml
# @package _group_
_target_: conf._schemas.NnMlpConfig
target_column: nd_mw
hidden_sizes: [128]
activation: relu
dropout: 0.0
learning_rate: 1.0e-3
weight_decay: 0.0
batch_size: 32
max_epochs: 100
patience: 10
device: auto
```

### On-disk artefact layout (`NnMlpModel.save(path)`)

The Stage 9 registry passes a **file path** to `save()` — specifically `artefact/model.joblib` (see `_fs.py::_atomic_write_run`). `NnMlpModel.save(path)` therefore writes a **single joblib artefact** at exactly that path (D5), with contents:

```python
{
    "state_dict_bytes": bytes,       # torch.save(state_dict, BytesIO).getvalue()
    "config_dump": dict[str, Any],   # NnMlpConfig.model_dump()
    "feature_columns": tuple[str, ...],
    "seed_used": int,
    "best_epoch": int,
    "loss_history": list[dict[str, float]],
    "fit_utc": str,                  # ISO-8601, UTC
    "device_resolved": str,          # "cpu" | "cuda" | "mps" (the device the fit ran on)
}
```

The `state_dict_bytes` payload carries, as a plain dict of tensors (not an `nn.Module`):

- All `nn.Linear` weights + biases (trainable parameters).
- The `feature_mean` / `feature_std` / `target_mean` / `target_std` buffers registered via `self.register_buffer(...)` in the module's `__init__` (D4).

`config_dump` is the reconstruction seed: `hidden_sizes`, `activation`, `dropout`, `target_column`, `feature_columns`, `device` — everything needed to rebuild the `nn.Module` skeleton before `load_state_dict(strict=True)`.

`NnMlpModel.load(path)` reverses the pipeline: `load_joblib(path)` → rebuild `NnMlpConfig` from `config_dump` → instantiate `NnMlpModel(config)` → `torch.load(BytesIO(state_dict_bytes), weights_only=True, map_location="cpu")` → `load_state_dict(strict=True)`. `weights_only=True` keeps PyTorch 2.6+'s safety rail active on the inner bytes payload; joblib around the outer envelope matches Stage 9's contract and every other model's serialisation idiom (plan D5 rationale).

The registry's `run.json` sidecar (Stage 9) is unchanged — it does not duplicate the joblib envelope's keys; it carries what it has carried since Stage 9 (`feature_set`, `target`, `feature_columns`, `fit_utc`, `git_sha`, `hyperparameters` from `ModelMetadata`, cross-fold metrics).

### Public interface (new)

```python
# src/bristol_ml/models/nn/mlp.py

class NnMlpModel:
    """Small MLP conforming to Stage 4's ``Model`` protocol.

    See docs/architecture/layers/models-nn.md for the full contract.
    """

    def __init__(self, config: NnMlpConfig) -> None: ...

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        seed: int | None = None,
        epoch_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None: ...

    def predict(self, features: pd.DataFrame) -> pd.Series: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> NnMlpModel: ...

    @property
    def metadata(self) -> ModelMetadata: ...

    # Stage-10-specific public attribute (populated after fit)
    loss_history_: list[dict[str, float]]
```

### Dispatch sites touched (five, per D2)

1. **`conf/_schemas.py`** — add `NnMlpConfig` class and `AppConfig.model` type union gains `NnMlpConfig`.
2. **`conf/model/nn_mlp.yaml`** — new Hydra group entry.
3. **`src/bristol_ml/train.py`** — new `isinstance(model_cfg, NnMlpConfig)` branch in the existing cascade (structurally identical to the `ScipyParametricConfig` branch at lines 287-293); sets `primary_kind = "nn_mlp"`.
4. **`src/bristol_ml/registry/_dispatch.py`** — `_TYPE_TO_CLASS["nn_mlp"] = NnMlpModel` and `_CLASS_NAME_TO_TYPE["NnMlpModel"] = "nn_mlp"`.
5. **`src/bristol_ml/evaluation/harness.py`** — NOT touched. The harness is model-agnostic (calls `model.fit / .predict`); the new family integrates via the existing re-entrancy contract.

### Integration with `train.py`

`train.py` already calls `harness.evaluate_and_keep_final_model(...)` then `registry.save(...)` on the non-`--no-register` path (Stage 9 T5 wiring). The Stage 10 change is purely:

1. Add an `isinstance(model_cfg, NnMlpConfig)` branch that instantiates `NnMlpModel(model_cfg)` and sets `primary_kind = "nn_mlp"`.
2. The existing registry-save path handles the rest unchanged — dispatch through `_dispatch.model_type(model)` resolves the `"nn_mlp"` sidecar type automatically once the dict entry is added.

### Loss-curve surfacing (D6 / X4)

```python
# src/bristol_ml/evaluation/plots.py  (addition)

def loss_curve(
    history: list[dict[str, float]],
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot train + validation loss per epoch from a NnMlpModel.loss_history_.

    Follows the plots-module ax= composability contract. Train line uses
    OKABE_ITO[0]; validation line uses OKABE_ITO[1]. British English
    axis labels.
    """
    ...
```

And the notebook's live-demo moment:

```python
# notebooks/10-simple-nn.ipynb (Cell N)

from IPython.display import display, clear_output
from bristol_ml.evaluation.plots import loss_curve

handle = display(None, display_id=True)
history: list[dict[str, float]] = []

def on_epoch(entry: dict[str, float]) -> None:
    history.append(entry)
    fig = loss_curve(history)
    clear_output(wait=True)
    handle.update(fig)

model = NnMlpModel(cfg.model)
model.fit(X, y, seed=cfg.project.seed, epoch_callback=on_epoch)
```

The notebook owns the live-plot logic; the models layer never imports IPython.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — NnMlpModel scaffold + config schema + standalone CLI

**Files:**
- new `src/bristol_ml/models/nn/__init__.py` (re-exports `NnMlpModel`).
- new `src/bristol_ml/models/nn/__main__.py` (delegates to `python -m bristol_ml.models.nn.mlp`).
- new `src/bristol_ml/models/nn/mlp.py` (class skeleton: `__init__`, `metadata` property, `save`/`load` stubs, `fit`/`predict` raising `NotImplementedError`, standalone CLI prints resolved config).
- `conf/_schemas.py` — add `NnMlpConfig` (including the `device` field from D11); update `AppConfig.model` union.
- new `conf/model/nn_mlp.yaml`.
- `pyproject.toml` — add `torch>=2.7,<3` to `[project].dependencies`, plus the `[tool.uv.sources]` / `[[tool.uv.index]]` stanza that binds `torch` to the cu128 wheel index on Linux and to PyPI on other platforms (D1). Register a new `gpu: opt-in CUDA-path tests, skipped when torch.cuda.is_available() is False.` entry in `[tool.pytest.ini_options].markers` (parallels the existing `slow` marker); the default `addopts` `-m 'not slow'` is extended to `-m 'not slow and not gpu'` so CI stays CPU-only by default.
- `Dockerfile` — add a `RUN --mount=type=cache,target=…/.cache/uv uv pip download --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<3"` pre-warm step before the `CMD` line (D1). Leaves everything else untouched.

**Tests (T1):**
- `test_nn_mlp_is_model_protocol_instance` (AC-1 scaffold half).
- `test_nn_mlp_config_schema_defaults_round_trip_through_hydra` — resolved defaults match `NnMlpConfig(...)` exactly, including `device == "auto"` (one test row rather than one per field per Scope Diff D3).
- `test_nn_mlp_standalone_cli_exits_zero` (NFR-6).

**Commits as:** `Stage 10 T1: NnMlpModel scaffold + config schema + standalone CLI`.

### Task T2 — fit / predict + training loop + reproducibility + loss history

**Files:** `src/bristol_ml/models/nn/mlp.py` (implement `fit`, `predict`, `_run_training_loop`, module-level helpers `_make_mlp`, `_seed_three_streams`).

**Content:**
- `_select_device(preference: str) -> torch.device` helper: resolves `"auto"` in order CUDA > MPS > CPU; validates pinned values; logs the chosen device at INFO (D11).
- `fit()` calls `_select_device(config.device)`, then seeds four streams (D7'): `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`. On the CUDA path additionally sets `torch.backends.cudnn.deterministic = True` / `torch.backends.cudnn.benchmark = False`. Moves the module + input tensors to the resolved device. Fits scaler buffers on train slice (D4); constructs the `nn.Module` with hidden_sizes + activation + dropout from config; hand-rolls the epoch loop with an internal 10 % val tail for early stopping (D9); appends to `loss_history_` per epoch; invokes `epoch_callback` if provided; restores best-epoch weights on early stop.
- `predict()` sets `eval()` mode, moves inputs to `self._device`, feeds features through (normalise → forward → inverse-transform target), returns `pd.Series` re-indexed on `features.index` on CPU.

**Tests (T2 — the AC-2 / AC-3 core):**
- `test_nn_mlp_fit_predict_round_trip_on_tiny_fixture` (AC-1) — pins `device="cpu"` so the test is deterministic in CI.
- `test_nn_mlp_seeded_runs_produce_identical_state_dicts` (AC-2 / T1 label) — pins `device="cpu"`; asserts `torch.equal` on every parameter tensor + `predict` output.
- `test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance` — `@pytest.mark.gpu`, skipped unless `torch.cuda.is_available()`; asserts `torch.allclose(atol=1e-5, rtol=1e-4)` on two seeded `predict` outputs (NFR-1 GPU close-match).
- `test_nn_mlp_different_seeds_produce_different_state_dicts` (pins `device="cpu"`).
- `test_nn_mlp_fit_populates_loss_history_per_epoch` (AC-3 / T4 label).
- `test_nn_mlp_fit_invokes_epoch_callback_when_provided` (AC-3).
- `test_nn_mlp_early_stopping_terminates_before_max_epochs_on_plateau` (T3 label).
- `test_nn_mlp_fit_uses_cold_start_per_fold_when_called_repeatedly`.
- `test_nn_mlp_select_device_auto_prefers_cuda_then_mps_then_cpu` — three parametrised cases that monkeypatch `torch.cuda.is_available` / `torch.backends.mps.is_available` to exercise each branch of the helper without requiring the hardware to be present.
- `test_nn_mlp_select_device_respects_explicit_pin` — `device="cpu"` on a CUDA-capable host resolves to CPU; invalid values raise `ValueError`.

**Commits as:** `Stage 10 T2: NnMlpModel fit/predict + training loop + reproducibility`.

### Task T3 — save / load + single-joblib artefact envelope

**Files:** `src/bristol_ml/models/nn/mlp.py` (implement `save` and `load`).

**Content:**
- `save(path)` serialises `self.state_dict()` into a `BytesIO` buffer via `torch.save`, assembles the envelope dict described in §5 (`state_dict_bytes`, `config_dump`, `feature_columns`, `seed_used`, `best_epoch`, `loss_history`, `fit_utc`, `device_resolved`), and writes it to the registry-provided file path via `bristol_ml.models.io.save_joblib`. The path is the file path (`artefact/model.joblib`), not a directory — matching the Stage 9 registry contract (D5).
- `load(path)` reads the joblib envelope via `bristol_ml.models.io.load_joblib`, reconstructs `NnMlpConfig` from `config_dump` (Pydantic re-validation catches schema drift), instantiates `NnMlpModel(config)`, materialises the `state_dict` via `torch.load(BytesIO(state_dict_bytes), weights_only=True, map_location="cpu")`, calls `load_state_dict(strict=True)`, restores the scalar attributes (`loss_history_`, `_best_epoch`, `_seed_used`, `_fit_utc`, `_device_resolved`), and returns the fitted instance.

**Tests (T3):**
- `test_nn_mlp_save_and_load_round_trips_state_dict_and_hyperparameters` (AC-4 unit level) — asserts `torch.allclose(atol=1e-10)` on predict output and that every scalar envelope field round-trips byte-exact.
- `test_nn_mlp_load_raises_file_not_found_for_missing_artefact`.
- `test_nn_mlp_save_writes_single_joblib_file_at_given_path` — structural check that `save()` does not create a sibling `model.pt` or `hyperparameters.json` (guard against regressing to the pre-D5-revision two-file layout).

**Commits as:** `Stage 10 T3: NnMlpModel save/load + single-joblib artefact envelope`.

### Task T4 — registry dispatch + end-to-end registry round-trip

**Files:**
- `src/bristol_ml/registry/_dispatch.py` — add `"nn_mlp"` / `NnMlpModel` entries to both dicts (D2 clause v).
- `src/bristol_ml/train.py` — add `isinstance(model_cfg, NnMlpConfig)` branch (D2 clause iii).

**Tests (T4):**
- `test_registry_save_nn_mlp_model_via_protocol` (AC-4 — extends the Stage 9 AC-2 test suite with a fifth model family; `atol=1e-10` on predict round-trip).
- `test_registry_list_runs_includes_nn_mlp_type` — sidecar `type` is `"nn_mlp"`.
- `test_nn_mlp_is_dispatched_by_train_cli_isinstance_branch` (T6 label — structural parallel of the Stage 8 test).

**Commits as:** `Stage 10 T4: registry dispatch + train.py wiring for nn_mlp`.

### Task T5 — plots.loss_curve helper + train-CLI integration test

**Files:**
- `src/bristol_ml/evaluation/plots.py` — add `loss_curve(history, *, ax=None)` helper (D6).
- `src/bristol_ml/evaluation/CLAUDE.md` — add `loss_curve` to the "Current surface (Plots)" bullet list; Okabe-Ito usage note.

**Tests (T5):**
- `test_plots_loss_curve_renders_figure_from_history` (AC-3 / T7 label).
- `test_plots_loss_curve_respects_ax_composability_contract`.
- `test_train_cli_registers_nn_mlp_final_fold_model` (T5 label — full pipeline integration; parallels the Stage 9 `test_train_cli_registers_final_fold_model`).

**Commits as:** `Stage 10 T5: plots.loss_curve helper + train-CLI end-to-end test`.

### Task T6 — Notebook + layer documentation

**Files:**
- new `notebooks/10-simple-nn.ipynb` (live-loss-curve demo + comparison against prior stages' registered runs via `registry.list_runs(...)`).
- new `docs/architecture/layers/models-nn.md` (D12).
- new `src/bristol_ml/models/nn/CLAUDE.md` (D12).

**Tests (T6):** none new (notebooks and layer docs are exercised by the existing `test_notebooks_execute_cleanly` and `test_layer_doc_exists` patterns; if a new test is needed, fold it into the hygiene task).

**Commits as:** `Stage 10 T6: notebook + models-nn layer doc + module CLAUDE.md`.

### Task T7 — Stage hygiene (H-1..H-5)

**Files:**
- `CHANGELOG.md` — `[Unreleased]` Stage 10 bullets under `### Added`.
- `docs/architecture/README.md` — module catalogue row for `models/nn/` (H-3 / Scope diff X3).
- `docs/lld/stages/10-simple-nn.md` — retro per the template, including the observed-CPU-wall-clock data point (AC-5 evidence note).
- `docs/lld/stages/09-model-registry.md` — confirm "Next" pointer to Stage 10 is current (H-2).
- `docs/plans/active/10-simple-nn.md` → `docs/plans/completed/10-simple-nn.md` at the final commit.

**Tests (T7):** none new.

**Commits as:** `Stage 10 T7: stage hygiene + retro + plan moved to completed/`.

---

## 7. Files expected to change

### New

- `src/bristol_ml/models/nn/__init__.py`
- `src/bristol_ml/models/nn/__main__.py`
- `src/bristol_ml/models/nn/mlp.py`
- `src/bristol_ml/models/nn/CLAUDE.md`
- `conf/model/nn_mlp.yaml`
- `tests/unit/models/test_nn_mlp_scaffold.py`
- `tests/unit/models/test_nn_mlp_fit_predict.py`
- `tests/unit/models/test_nn_mlp_save_load.py`
- `tests/unit/evaluation/test_plots_loss_curve.py`
- `tests/unit/registry/test_registry_nn_mlp_dispatch.py`
- `tests/integration/test_train_cli_registers_nn_mlp.py`
- `notebooks/10-simple-nn.ipynb`
- `docs/architecture/layers/models-nn.md`
- `docs/lld/stages/10-simple-nn.md`

### Modified

- `conf/_schemas.py` — add `NnMlpConfig`; extend `AppConfig.model` union.
- `src/bristol_ml/train.py` — add `isinstance(model_cfg, NnMlpConfig)` branch; no change to the registry-save path or the harness call.
- `src/bristol_ml/evaluation/plots.py` — add `loss_curve` helper.
- `src/bristol_ml/evaluation/CLAUDE.md` — `loss_curve` bullet under "Current surface".
- `src/bristol_ml/registry/_dispatch.py` — one new key in each of `_TYPE_TO_CLASS` and `_CLASS_NAME_TO_TYPE`.
- `src/bristol_ml/registry/CLAUDE.md` — "Current surface" mentions `nn_mlp` as a new registered family.
- `pyproject.toml` — `torch>=2.7,<3` added to `[project].dependencies`; new `[tool.uv.sources]` + `[[tool.uv.index]] name = "pytorch-cu128"` stanza pinning the CUDA wheel on Linux and PyPI elsewhere (D1); new `gpu` pytest marker registered; `addopts` extended to `-m 'not slow and not gpu'`.
- `Dockerfile` — new pre-warm step downloading the CUDA 12.8 torch wheel into the uv cache volume at image-build time (D1).
- `docs/architecture/README.md` — module catalogue row for `models/nn/` (H-3).
- `CHANGELOG.md` — `[Unreleased]` Stage 10 bullets.

### Moved (final commit of T7)

- `docs/plans/active/10-simple-nn.md` → `docs/plans/completed/10-simple-nn.md`.

### Explicitly NOT modified

- `docs/intent/DESIGN.md` §6 — deny-tier; H-1 deferred per Stage 9 precedent.
- `docs/intent/10-simple-nn.md` — immutable spec.
- `src/bristol_ml/models/protocol.py` — protocol signature unchanged (AC-1 contract).
- `src/bristol_ml/models/{naive,linear,sarimax,scipy_parametric}.py` — no changes.
- `src/bristol_ml/evaluation/harness.py` — no new dispatch branch (harness is model-agnostic by design; H5 API-growth rule honoured).
- `src/bristol_ml/evaluation/benchmarks.py` — `nn_mlp` does not join the hard-wired three-way chart at Stage 10; leaderboard is the competitive surface.
- `src/bristol_ml/registry/__init__.py` — no change to the four-verb surface (AC-1 of Stage 9 unchanged).

---

## 8. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | The four-stream seed + cuDNN-deterministic recipe is insufficient for the NFR-1 CPU bit-identity bar, or drifts beyond `atol=1e-5` on CUDA / MPS. | Low | Medium | T1 on CPU fails loudly if CPU bit-identity breaks; the `@pytest.mark.gpu` variant fails loudly if the CUDA close-match bar is breached. Escalation on failure is staged: (1) verify no CPU-only code path has been forgotten in the seeding helper; (2) if the CUDA close-match still drifts, widen the tolerance to `atol=1e-4` in the gpu test only and record the relaxation in the stage retro; (3) as a last resort, graduate D7' to `torch.use_deterministic_algorithms(True)` under a new decision — no schema change required. |
| **R2** | `torch.save(state_dict)` + `hyperparameters.json` reconstruction at load time skews from the fit-time architecture (e.g. activation string mismatch) and produces silently-wrong predictions rather than an error. | Low | High | T2 and T3 both round-trip `predict()` output to `torch.allclose(atol=1e-10)`; T3 reconstruction explicitly verifies architecture-config equality before `load_state_dict`. |
| **R3** | `register_buffer` buffers are dropped on `load_state_dict` if the reconstructed module's `__init__` does not register them in the same order. | Low | High | `NnMlpModel.__init__` unconditionally registers all four buffers with a fixed-init-value of zero; `load_state_dict(strict=True)` fails loudly if any buffer is missing or extra. T3 asserts strict loading. |
| **R4** | The 10 % internal val tail is leaky on a time-series — validation data is temporally *within* the train slice if split randomly. | Low | Medium | The val tail is taken from the **end** of the train slice (last 10 % by index), not a random split; documented in the layer doc (D9) and tested via `test_nn_mlp_val_slice_is_contiguous_tail_of_train`. |
| **R5** | Adding `torch` to `[project].dependencies` bloats the install and slows CI. | Medium | Low | The CPU wheel is ~175 MB. Acceptable for a ML-oriented research repo; DESIGN §8 already names PyTorch. Pin the version range (`>=2.3,<3`) so a surprise upgrade does not break the deterministic-CPU contract mid-stage. |
| **R6** | `epoch_callback` invoked after every epoch becomes a notebook perf bottleneck for `max_epochs=100`. | Low | Low | The callback is optional and the notebook's matplotlib redraw is the slow path, not `fit()`'s reporting. If a future facilitator hits this, the contract supports throttling in the callback (e.g. every 5 epochs) without any model change. |
| **R7** | Cold-start per fold (D8) makes rolling-origin slow on larger architectures. | Low | Medium | Default config is 1 × 128 ReLU, ~40k parameters — fit time is seconds per fold on a laptop. If a facilitator raises `hidden_sizes`, they accept the per-fold cost; Stage 11 owns the warm-start trade-off investigation for temporal architectures. |
| **R8** | The Stage 11 extraction seam (D10) is not respected and the hand-rolled loop accretes features (gradient clipping, LR schedule) over time, making Stage 11's refactor expensive. | Medium | Low | Scope Diff X6 pre-emptively cuts these knobs; the layer doc's "Stage 11 extraction seam" section names gradient clipping / LR scheduling as the trigger for the extraction, not an in-place addition. |

---

## 9. Exit checklist

Verified before T7's final commit.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail` without a linked issue.
- [ ] Ruff + format + pre-commit clean: `uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- [ ] `uv run python -m bristol_ml.models.nn.mlp --help` exits 0 and prints the resolved `NnMlpConfig` schema, including the `device` field default of `"auto"` (NFR-6, D11).
- [ ] `uv run python -m bristol_ml.train model=nn_mlp` leaves exactly one new `run_id` in `data/registry/`.
- [ ] On the CUDA dev host, `uv run pytest -m gpu -q` passes (NFR-1 close-match). On CPU-only CI, the same marker is skipped by the `addopts` filter and does not fail the suite.
- [ ] `docker build` caches the cu128 torch wheel into the uv cache volume; a subsequent `uv sync --frozen` inside the container does not re-download it. Recorded in the stage retro as a build-time measurement.
- [ ] `uv run python -m bristol_ml.registry list --model-type nn_mlp` prints the new run.
- [ ] `uv run python -m bristol_ml.registry describe <nn_mlp_run_id>` prints a sidecar whose `type` field is `"nn_mlp"`.
- [ ] All five intent-ACs mapped to named tests in §4 have a passing test.
- [ ] `docs/architecture/layers/models-nn.md` exists and documents the four contract points (D5 serialisation, D6 loss-curve, D7' reproducibility, D10 extraction seam).
- [ ] `src/bristol_ml/models/nn/CLAUDE.md` exists (D12).
- [ ] `docs/lld/stages/10-simple-nn.md` retro written per template, including the observed CPU wall-clock data point (AC-5 informational).
- [ ] `CHANGELOG.md` updated with the stage bullets.
- [ ] `docs/architecture/README.md` module catalogue updated (H-3).
- [ ] `docs/plans/active/10-simple-nn.md` moved to `docs/plans/completed/`.
- [ ] H-1 (DESIGN §6) deferred per Stage 9 precedent; H-2 (Stage 9 retro wording) verified; H-3 (architecture README) actioned; H-4 (dispatcher ADR) re-deferred.
- [ ] OQ-A (D7' use_deterministic_algorithms) resolved at Ctrl+G; either adopted now or explicitly deferred to a named stage.
- [ ] PR description includes: Stage 10 summary, Scope Diff link, any Phase 3 review findings, H-1 DESIGN §6 deferral note.
