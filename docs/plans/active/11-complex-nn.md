# Plan — Stage 11: Complex neural network (temporal)

**Status:** `approved` — Ctrl+G on 2026-04-24 accepted the decision table with one amendment: **D1 architecture defaults re-targeted at CUDA (the Blackwell dev host per Stage 10 D1) and materially enlarged, with the knobs staying in config so CPU-only laptops remain reachable via Hydra overrides.** Proceeding to Phase 2.
**Intent:** [`docs/intent/11-complex-nn.md`](../../intent/11-complex-nn.md)
**Upstream stages shipped:** Stages 0–10 (foundation → ingestion → features → four Stage-4/7/8 classical models → enhanced evaluation → registry → simple MLP).
**Downstream consumers:** Stage 12 (serving — loads temporal runs by name through the registry, same four-verb surface), Stage 18 (drift monitoring). Subsequent modelling stages use the ablation artefact produced here as their "did we beat the best?" reference.
**Baseline SHA:** `6ad2d7a` (tip of `main` after the DESIGN §6 `docs/plans/` addition — PR #9).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/11-complex-nn-requirements.md`](../../lld/research/11-complex-nn-requirements.md)
- Codebase map — [`docs/lld/research/11-complex-nn-codebase.md`](../../lld/research/11-complex-nn-codebase.md)
- Domain research — [`docs/lld/research/11-complex-nn-domain.md`](../../lld/research/11-complex-nn-domain.md)
- Scope Diff — [`docs/lld/research/11-complex-nn-scope-diff.md`](../../lld/research/11-complex-nn-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §Demo moment names **the ablation table** as Stage 11's canonical meetup artefact — one row per model family, same held-out period, facilitator-legible story arc from naïve through linear through SARIMAX through scipy-parametric through MLP to the new temporal model. Intent §Purpose is equally explicit about the stage's analytical stake: this is where the neural approach "has a genuine shot at beating the linear and classical models" *because* temporal dependencies are finally in the model's field of view. The stage closes the modelling arc. Every decision below is either a direct consequence of AC-1 (protocol conformance) / AC-2 (Stage 10 training harness) / AC-3 (ablation table) / AC-4 (registry round-trip including sequence-preprocessing state) / AC-5 (registry-only reproducibility of the table), or it is cut per the `@minimalist` Scope Diff.

**AC-5 reconciliation (surfaced by the codebase map and load-bearing for plan §5).** AC-5 says the ablation table must be reproducible from registered runs without re-training anything already in the registry. `evaluation.benchmarks.compare_on_holdout` internally calls `harness.evaluate()` which re-fits the model on every fold — *it does re-train*. The plan resolves this by **not** using `compare_on_holdout` for the ablation cell; instead the notebook loads each registered run via `registry.load(run_id)`, calls `model.predict(X_holdout)` on the same single-holdout slice (D12), and composes metrics inline using the existing `bristol_ml.evaluation.metrics` functions. This is a pure predict-only path — it honours AC-5 literally. D11 (a standalone `evaluation/ablation.py` helper) is **cut** per the Scope Diff: the notebook-inline implementation satisfies AC-5 at lower blast radius.

---

## 1. Decisions for the human (resolve before Phase 2)

Sixteen decision points (fourteen kept, two cut per the Scope Diff) plus three housekeeping carry-overs. The decision set is filtered through the `@minimalist` Scope Diff in [`docs/lld/research/11-complex-nn-scope-diff.md`](../../lld/research/11-complex-nn-scope-diff.md); three tags were flipped from the lead's draft framing — **D8 cut** (val-split `seq_len` offset), **D11 cut** (standalone `evaluation/ablation.py` module), and **NFR-7 loss_history test** retained as a one-line assertion despite its `PLAN POLISH` tag. Defaults lean on the three research artefacts and the simplicity bias in `DESIGN.md §2.2.4`. The Evidence column cites the research that *resolved* each decision; intent "Points for consideration" are not cited as evidence because they *pose* the decision rather than answering it.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Architecture family (CUDA defaults; CPU override path documented) | **TCN — dilated causal 1D-conv stack. 8 residual blocks × kernel=3 × 128 channels, dilations `[1, 2, 4, 8, 16, 32, 64, 128]`. Weight normalisation (`torch.nn.utils.weight_norm`) on each `Conv1d`. `nn.LayerNorm` between residual blocks. Dropout 0.2. ReLU activations. Left-only causal padding via `F.pad(x, (pad, 0))` + `Conv1d(padding=0)` — the Bai et al. 2018 recipe with the §6-pitfall-4 mitigation wired in by construction. Training defaults also sized for CUDA: `batch_size=256`, `max_epochs=100`, `patience=10`.** All architecture and training hyperparameters are Pydantic-exposed (NFR-6), so a CPU-only facilitator drops to a tractable recipe via CLI override (e.g. `model.num_blocks=4 model.channels=32 model.batch_size=32 model.max_epochs=20 model.device=cpu`). A CPU-recipe snippet rides in the YAML header comment and the layer doc. Small-Transformer variant is **deferred** (not cut at the architecture-class level — a future stage can graduate the family). | Intent §Scope names "a temporal neural model (TCN, small Transformer, or similar)"; intent §Points flags "architecture choice is the main design decision". The project's dev host is a Blackwell RTX 5090 on the CUDA 12.8 / cu128 wheel track (Stage 10 D1) — sizing the default for the actual training target is the honest framing, and a 128-channel × 8-block TCN is ~800k parameters on ~8760 rows, large enough to meaningfully exceed the MLP's capacity without crossing into over-fit territory (weight-norm + dropout=0.2 as the capacity–regularisation trade). Receptive field closed-form: `1 + 2·(k−1)·Σ dilations` = 1021 steps at the new spec — covers the weekly cycle (168 h) with ~6× headroom. The Transformer variant is still deferred because PyTorch #99282 (is_causal silently ignored when need_weights=True) is unavoidable for any Transformer that needs attention visualisation (domain §6 pitfalls 1, 5), and the TCN's causal structure is enforced by construction. CPU budget per domain §8 (~3–8 min at the *smaller* 6-block × 64-channel spec) is now an opt-in path recorded in the retro as a secondary data point; the primary target is CUDA. | Intent §Scope, §Points; Stage 10 D1 (CUDA 12.8 / Blackwell dev host); domain research §1 (family table), §2 (TCN vs Transformer trade-off), §6 pitfalls 1, 4, 5 (causal padding + is_causal bug), §8 (CPU budget, now secondary). Scope Diff D1 (RESTATES INTENT). 2026-04-24 Ctrl+G: CUDA default amendment. |
| **D2** | Sequence length | **`seq_len = 168` hours (the weekly cycle anchor), exposed as `NnTemporalConfig.seq_len: int = 168` so a facilitator can dial it down to 96 h or 48 h for a faster demo. Config validators reject values `< 2·kernel_size·max_dilation` (= 192 at the D1 spec) to prevent degenerate receptive-field configurations silently shipping.** The receptive-field validator is one Pydantic `@model_validator` — kept because silently shipping a model whose receptive field exceeds the input window is a correctness bug, not a polish. | Intent §Points names the weekly cycle (168 h) as "a natural upper bound" and asks "how many hours of history". Domain research §3 cites UniLF (Phan et al., Scientific Reports 2025) showing best STLF accuracy at `in_len=168` specifically on hourly load data; 336 h is used by PatchTST but roughly halves the number of non-overlapping windows from ~52 to ~26 — genuinely risky on ~7900 training rows. 168 h is the only length with both a physical-period anchor (same hour last week) and a literature-direct result on this task shape. | Domain research §3 (UniLF 2025, PatchTST §context-length discussion); intent §Points. Scope Diff D2 (RESTATES INTENT). |
| **D3** | Exogenous feature handling | **Pattern A — in-sequence concatenation. The `_SequenceDataset` yields windows of shape `(seq_len, n_features)` where `n_features` is the same column set the MLP uses (weather + calendar one-hots). No separate "known-future" branch; no TFT-style side channel.** | Intent §Points asks "feed weather as part of the sequence or as side channels". Domain research §4 notes that for a day-ahead forecaster where the feature set *already contains* the target-hour weather forecast (as Stage 5's feature table does — calendar and weather are both joined on the target index), Patterns A and B are informationally equivalent at prediction time. Pattern A is strictly simpler: single-branch `nn.Module`, no decoder, no special collation. Pattern B's benefit is a cleaner training signal for very long horizons or multi-horizon setups; intent §Out of scope eliminates multi-horizon. If the Stage 11 ablation shows the temporal model underperforming the MLP, Pattern B is the natural next-stage diagnostic (not this stage's scope). | Domain research §4; intent §Out of scope (multi-horizon). Scope Diff D3 (RESTATES INTENT). |
| **D4** | Training-loop ownership — **Stage 10 D10 extraction seam fires** | **Extract the Stage 10 `_run_training_loop` body from `NnMlpModel.fit` into a new module `src/bristol_ml/models/nn/_training.py`. Expose a single helper `run_training_loop(module, train_loader, val_loader, *, optimiser, criterion, device, max_epochs, patience, loss_history, epoch_callback) -> tuple[dict[str, Tensor], int]` returning `(best_state_dict, best_epoch)` and populating `loss_history` in place. Both `NnMlpModel.fit` and `NnTemporalModel.fit` import and call it. No `BaseTorchModel` ABC — the extraction is a function, not a class hierarchy (Stage 10 X7 stays cut).** The Stage 10 test suite runs unchanged against the refactored `mlp.py`; any test failure at T1 is a refactor regression and halts the stage. | Stage 10 `CLAUDE.md` lines 145–152 explicitly flag this file, this function name, and this trigger ("when the second torch-backed model arrives with a hand-rolled loop of its own"). The loop body is shape-agnostic below the collate layer — `_SequenceDataset` differs from `TensorDataset` in *what* is yielded, not *what the loop does with it*. The Scope Diff tags this `PLAN POLISH` because AC-2 could also be satisfied by Stage 11 importing private helpers from `mlp.py`; the lead's disposition is to honour the named Stage 10 seam because (i) it was planned for, (ii) a cross-module import from `temporal.py` into `mlp.py`'s privates is actively worse than a shared `_training.py`, and (iii) duplicating the loop wins nothing and costs a future divergence hazard. | Stage 10 plan D10 (extraction seam); Stage 10 `CLAUDE.md` lines 145–152; codebase map §1.1 (training loop body); intent AC-2. Scope Diff D4 (PLAN POLISH — kept with explicit justification). |
| **D5** | Serialisation format | **Single-joblib artefact at the registry-provided path (`artefact/model.joblib`), containing `{"state_dict_bytes": bytes, "config_dump": dict, "feature_columns": tuple, "seed_used": int, "best_epoch": int, "loss_history": list, "fit_utc": str, "device_resolved": str, "seq_len": int}`** — identical envelope to Stage 10 D5 revised plus one new field (`seq_len`) to guard against silently loading a model built for a different window size. Written via `bristol_ml.models.io.save_joblib` (atomic-write + parent-dir creation). | Intent AC-4 names both "full weights" and "the sequence preprocessing state" as round-trip requirements. The scaler buffers (`feature_mean`, `feature_std`, `target_mean`, `target_std`) already ride inside the `state_dict` via `register_buffer` (Stage 10 D4 recipe inherited). The sequence-preprocessing state specific to Stage 11 is (i) `seq_len` — needed to reshape `predict` input, (ii) the feature column order — already in `feature_columns`. The envelope shape is the only pattern the Stage 9 registry accepts (codebase map §3.1 single-file contract). | Intent AC-4; codebase map §3.1; Stage 10 D5 envelope precedent. Scope Diff D5 (RESTATES INTENT). |
| **D6** | Reproducibility discipline | **Inherit Stage 10 D7' unchanged. Four-stream seeding at the top of `fit()`: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`. On CUDA: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`. `torch.use_deterministic_algorithms(True)` stays OFF per Stage 10's Ctrl+G resolution (AC-2's GPU carve-out closes it).** If `_training.py` is extracted per D4, the seeding helper `_seed_four_streams` moves with it and is imported from there by both model modules. | Intent AC-4 names the `torch.allclose(atol=1e-5)` bar, directly inherited from Stage 10 D7'. All Stage 11 non-determinism sources (LayerNorm, Conv1d, weight-norm) are deterministic on CPU under the standard recipe. `nn.MultiheadAttention` / SDPA / `is_causal` non-determinism is **not in scope** because D1 chose TCN — pitfall 1 and pitfall 2 from domain §6 are moot. | Stage 10 plan D7'; intent AC-4; domain research §6 pitfalls 1, 2, 7. Scope Diff D6, NFR-1 (RESTATES INTENT). |
| **D7** | Sequence-data pipeline | **A private `_SequenceDataset(torch.utils.data.Dataset)` class defined inside `temporal.py`. Takes a `pd.DataFrame` and a `target` series plus `seq_len`; `__len__` returns `len(df) - seq_len` (number of valid windows); `__getitem__(i)` returns `(features[i:i+seq_len], target[i+seq_len])` as pre-converted `float32` tensors. Lazy-window by design — no eager `(N, seq_len, n_features)` materialisation.** Kept private because no other Stage 11 surface and no future stage currently needs it. | Intent §Scope explicitly locates the data-pipeline change "in the model's `fit` / `predict` or in a helper reused from Stage 10"; there is no reused Stage 10 helper so the encapsulation is in-module. Codebase map §4 calculates that the eager path costs ~1.4 GB for weather+calendar features (44 calendar one-hots × float32 × 43,633 windows × 168 steps) — lazy is the only option that stays inside a laptop's RAM budget while also respecting intent §Out of scope (no distributed training). Keeping `_SequenceDataset` private avoids publishing an API whose downstream contract is not yet known. | Codebase map §4 (memory-cost calculation); intent §Scope. Scope Diff D7 (RESTATES INTENT). |
| **~~D8~~** | ~~Val-split offset by `seq_len` to prevent sequence-overlap leakage~~ | **CUT** per Scope Diff. | The Stage 10 internal val-split is the last 10 % of the train slice (contiguous tail, not random). Without D8, the last `seq_len - 1` training windows and the first `seq_len - 1` validation windows share overlapping input regions. This affects *early-stopping fidelity* (the val-loss number is slightly optimistic) but **not** the reported holdout metrics — the harness's test fold is fully separated by the `SplitterConfig` gap. No intent AC names internal val-split correctness, and the ablation contract (AC-3 / AC-5) is unaffected. Adds a 168-row offset to `fit()`'s internal split plus a test assertion, for improved early-stopping-fidelity on a metric the intent never tests against. Cut: `PREMATURE OPTIMISATION`. If early stopping fires spuriously early in the retro observation, this decision is re-opened. | Scope Diff D8 (PREMATURE OPTIMISATION — minimalist flip from the lead's draft tag); domain research §6 (flags as subtle risk, not correctness blocker). |
| **D9** | Harness integration | **`NnTemporalModel` conforms to the Stage 4 `Model` protocol exactly. `harness.evaluate()` is reused unchanged; no model-specific branch. The cold-start-per-fold contract from Stage 10 D8 is inherited — `fit()` resets `_module`, `loss_history_`, `_best_epoch`, `_seed_used` at entry.** | Intent AC-1 (protocol conformance) and AC-2 (Stage 10 harness) both require this. Codebase map §2.1 confirms the harness is model-agnostic — the only harness-adjacent change is the `_build_model_from_config` factory (see D13). | Intent AC-1, AC-2; codebase map §2.1; Stage 10 D8. Scope Diff D9 (RESTATES INTENT). |
| **D10** | Ablation presentation | **One markdown table, six rows × seven columns: `model_name`, `MAE (MW)`, `MAPE (%)`, `RMSE (MW)`, `WAPE (%)`, `MAE ratio vs NESO`, `training_time_s`, `param_count`. Rendered directly in the notebook from a `pd.DataFrame` via `df.to_markdown()`. No bar chart, no predicted-vs-actual scatter, no receptive-field diagram** (the latter cut per Scope Diff — see Cell 7 below). | Intent §Demo moment names the table as *the* pedagogical artefact ("A facilitator can look at the table and tell a coherent story"). Intent §Points lists bar chart and scatter as alternatives; both are optional and the scatter is flagged as "visually busy". The seven columns are: five required for model-quality comparison (the NESO-ratio column is load-bearing for US-3 / the "beat NESO" secondary goal), plus `training_time_s` and `param_count` to quantify the cost-accuracy trade-off that underpins the pedagogical arc (domain research §9 names these as the most illuminating columns for a cross-stage ablation). | Intent §Demo moment, §Points; domain research §9. Scope Diff D10 (RESTATES INTENT). |
| **~~D11~~** | ~~New `evaluation/ablation.py::compute_metrics_on_holdout(run_ids, holdout_range, target)` helper~~ | **CUT** per Scope Diff. The notebook cell that builds the ablation table inlines the predict-only loop (`for run_id in runs: model = registry.load(run_id); preds = model.predict(X_holdout); metrics[run_id] = {name: fn(y_holdout, preds) for name, fn in metric_fns.items()}`) using existing public functions from `bristol_ml.registry`, `bristol_ml.evaluation.metrics`, and `bristol_ml.evaluation.plots`. No new public module, no standalone unit test of the helper, no binding of the notebook to an API that no other stage consumes. | AC-5 is load-bearing but does not name a public module. A notebook-inline implementation (~15 lines) satisfies AC-5 literally, avoids extending the `evaluation/` package's surface, and sidesteps the `compare_on_holdout` re-fit problem documented in the preamble. Scope Diff tags this `PLAN POLISH` and names it the single highest-leverage cut. | Scope Diff D11 (PLAN POLISH — minimalist flip from the lead's draft tag); codebase map §2.4 (the `compare_on_holdout` re-fit gap). |
| **D12** | Evaluation regime for the ablation | **Single holdout — the last contiguous block of the feature table (last 20 % or fixed-date boundary — picked at T8 from the Stage 6 holdout fixture). All six models evaluated on the exact same indices.** No rolling-origin for the ablation table. | Intent §Points: "Whether to run the complex model across multiple rolling-origin folds or a single holdout. Multiple folds is honest but expensive." Intent flags the expense; resolution is single-holdout for the ablation, consistent with the notebook's demo-speed constraint. The rolling-origin harness remains available for any user who wants to re-run with folds — `harness.evaluate()` is unchanged. The Stage 10 MLP's registered run was produced under rolling-origin; the single-holdout predict path in the notebook is a pure `model.predict(X_holdout)` call, which is what AC-5 requires anyway. | Intent §Points; AC-5. Scope Diff D12 (RESTATES INTENT — mandatory resolution). |
| **D13** | Dispatcher extension (three sites) | **Three edits: (i) `src/bristol_ml/registry/_dispatch.py` — add `"nn_temporal"` → `NnTemporalModel` to `_TYPE_TO_CLASS` and `"NnTemporalModel"` → `"nn_temporal"` to `_CLASS_NAME_TO_TYPE`. (ii) `src/bristol_ml/train.py` — add `isinstance(model_cfg, NnTemporalConfig)` branch mirroring the existing `NnMlpConfig` branch (no feature-column promotion, same `pragma: no cover` safety net). (iii) `src/bristol_ml/evaluation/harness.py::_build_model_from_config` — add `isinstance(model_cfg, NnTemporalConfig)` branch and, per D14, the missing `NnMlpConfig` branch.** | Intent AC-1 + AC-2 + AC-4 all require end-to-end wiring. Codebase map §3.1 / §3.2 name all three sites as mandatory for the sixth model family; missing any one produces silent runtime failures. The dispatcher-consolidation ADR (`H-4` Stage 10 carry-over) is re-deferred — Stage 11 does not introduce a fourth dispatcher site, it extends the three that exist. | Codebase map §3.1, §3.2; intent AC-1/2/4. Scope Diff D13 (RESTATES INTENT). |
| **D14** | Stage 10 catch-up: `NnMlpConfig` branch in `harness._build_model_from_config` | **Add the missing `NnMlpConfig` branch in the same T7 commit as the new `NnTemporalConfig` branch (D13 clause iii).** The codebase map flagged that Stage 10 shipped without this branch — a gap that is latent until someone calls the harness CLI's internal factory with a `NnMlpConfig`. Fixing in-commit because T7 already touches the function and a named `Stage 10 catch-up` commit message keeps the audit trail honest. | Codebase map §2.1 and §1.4 (the gap); CLAUDE.md §Stage-hygiene ("surface and justify spec drift rather than silently rewrite"). `HOUSEKEEPING` per Scope Diff; defensible in-stage because no separate Stage-10-hotfix PR is worth the overhead for a one-line `isinstance` branch addition. | Codebase map §2.1; Scope Diff D14 (HOUSEKEEPING). |
| **D15** | DLinear / NLinear as an extra ablation row | **NOT included.** Intent §Scope says "a temporal neural model" (singular). Domain research §1 and §9 argue DLinear would be a pedagogically illuminating sixth baseline row — it is explicitly designed to be the simplest defensible temporal model, often beats small Transformers on short-horizon tasks, and trains in seconds. But the intent is clear: one new model family per stage. A future stage may include DLinear; Stage 11 does not. | Intent §Scope; domain research §1 / §9 (risk flag acknowledged). Scope Diff D15 (RESTATES INTENT). |
| **D16** | Attention weight visualisation | **N/A.** Consequence of D1 (TCN over Transformer). Domain research §6 pitfall 5 plus PyTorch #99282 make attention visualisation a live correctness risk *for Transformers*; the TCN choice makes the question moot. Convolution filter-weight visualisation is possible but is explicitly flagged in intent §Points as "not worth building an interpretability stage around" — the same bar governs the receptive-field diagram (see Cell 7 below). | Intent §Points; domain research §6 pitfalls 1, 5. Scope Diff D16 (RESTATES INTENT). |
| **D17** | Layer documentation | **Extend `docs/architecture/layers/models-nn.md` with a "Stage 11 addition" section (the `NnTemporalModel` contract — `_SequenceDataset`, `seq_len`, receptive field, Pattern A exogenous handling, the D4 extraction seam status from "pending" to "fired"). Extend `src/bristol_ml/models/nn/CLAUDE.md` with the TCN-specific gotchas (causal-padding recipe, weight-norm placement, sequence-dataset lazy-window contract, single-joblib envelope `seq_len` field). No new layer file.** | The `models/nn/` layer doc already exists from Stage 10 D12; the temporal model is a sibling class in the same layer, not a new layer. `CLAUDE.md` §Stage hygiene names module-doc updates as mandatory for any meaningfully-touched module. A new layer file would imply a new layer, which is wrong. | Stage 10 D12; CLAUDE.md §Stage hygiene. Scope Diff (HOUSEKEEPING via T10). |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-1** | Seeded-run reproducibility, device-aware. **On CPU:** two back-to-back `fit(seed=0)` runs produce identical `state_dict` tensors and identical `predict` output under `torch.equal`. **On CUDA or MPS:** the same two runs produce `predict` output matching under `torch.allclose(atol=1e-5, rtol=1e-4)` — direct inheritance of Stage 10 D7'/NFR-1. Tested on CPU in the default suite; a second `@pytest.mark.gpu` test exercises the CUDA path on the dev host. | Intent AC-4; Stage 10 NFR-1 precedent. Scope Diff NFR-1 (RESTATES INTENT). |
| **NFR-2** | Training-time budget — no hard gate, two documented paths. **Primary (CUDA, defaults):** training at `num_blocks=8, channels=128, batch_size=256, max_epochs=100, seq_len=168` on the Blackwell dev host completes in < 5 min; observed wall-clock recorded in the retro. **Secondary (CPU override):** training at the documented CPU recipe (`num_blocks=4, channels=32, batch_size=32, max_epochs=20`) completes in < 10 min on a 4-core laptop; observed wall-clock also recorded. Neither is a hard gate — consistent with Stage 10's Scope Diff NFR-2 disposition. If either observed time misses its target, the retro records the miss and the defaults are re-visited in a follow-on stage. | Intent §Points ("if training takes too long to demo, the stage loses pedagogical value"); intent AC-7 (inferred); domain research §8 (CPU budget at smaller spec); Stage 10 D1 (CUDA dev host). Scope Diff NFR-2 (RESTATES INTENT). 2026-04-24 Ctrl+G: CUDA-primary framing. |
| **NFR-3** | Registry save/load fidelity. `registry.save(model, ...)` then `registry.load(run_id)` produces a model whose `predict` output on the same input matches under `torch.allclose(atol=1e-5)`. | Intent AC-4; Stage 10 NFR-3 bar (tightened from `atol=1e-10` to `atol=1e-5` for consistency with the NFR-1 close-match bar when the save-and-load cycle may cross device boundaries). Scope Diff NFR-3 (RESTATES INTENT). |
| **NFR-4** | Provenance sidecar. `run.json` for every registered temporal run records `git_sha`, `fit_utc`, `feature_set`, `target`, `feature_columns`, `seed_used` — direct inheritance of Stage 10 sidecar fields. The `seq_len` knob rides inside the model joblib envelope (D5), not the sidecar, per the Stage 9 sidecar contract. | DESIGN §2.1.6; Stage 10 NFR precedent. Scope Diff NFR-3 (RESTATES INTENT). |
| **NFR-5** | Standalone CLI. `uv run python -m bristol_ml.models.nn.temporal --help` exits 0 and prints the resolved `NnTemporalConfig` schema. | DESIGN §2.1.1; Stage 10 NFR-6 precedent. Scope Diff NFR-4 (RESTATES INTENT). |
| **NFR-6** | Config outside code. `conf/model/nn_temporal.yaml` + `NnTemporalConfig` Pydantic schema carry every architecture and training hyperparameter; no numeric defaults hard-coded in `temporal.py`. | DESIGN §2.1.4; Stage 10 D3/NFR-5 precedent. Scope Diff NFR-5 (RESTATES INTENT). |
| **NFR-7** | Notebooks thin. `notebooks/11-complex-nn.ipynb` imports only from `src/bristol_ml/`; no reimplemented training logic, no reimplemented metric computation, no private-attribute access on loaded models. | DESIGN §2.1.8; Stage 10 NFR-6 precedent. Scope Diff NFR-6 (RESTATES INTENT). |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` — Stage 11 adds `src/bristol_ml/models/nn/_training.py` and `src/bristol_ml/models/nn/temporal.py`. | **Defer.** User clarified on 2026-04-24 (post-Stage-10 Ctrl+G): "§6 is intended as structural-only, should only need to be updated very occasionally (ie not every new package)." The addition of a second file to an existing sub-package is not structural; no §6 edit. H-1 is closed, not rolled forward. |
| **H-2** | Stage 10 retro "Next" pointer to Stage 11 — confirm current. | **Verify at T9 hygiene.** |
| **H-3** | Dispatcher-consolidation ADR (`0004-model-dispatcher-consolidation.md`) — re-deferred from Stage 7 / 8 / 9 / 10. | **Re-defer.** Stage 11 is the sixth model family and does not introduce a fourth dispatcher site — it extends the three that exist. The ADR should land before a future stage introduces a *fourth* site or a seventh family; a dedicated housekeeping stage is the right vehicle. |

### Resolution log

- **Drafted 2026-04-24** — pre-Ctrl+G. All fourteen kept decisions (D1–D7, D9, D10, D12–D14, D16, D17) are proposed defaults; D8 and D11 are cut; D15 is a negative decision.
- **Amended 2026-04-24 (Ctrl+G approved)** — one amendment folded in: **D1** architecture defaults re-targeted at CUDA (Blackwell dev host per Stage 10 D1). Defaults bumped: `num_blocks` 6 → 8, `channels` 64 → 128, `dropout` 0.1 → 0.2, `batch_size` 64 → 256, `max_epochs` 50 → 100, `patience` 5 → 10. Receptive field rises from 253 to 1021 steps. CPU-override path documented in the YAML header, layer doc, and notebook preamble. NFR-2 restructured to record both CUDA-primary and CPU-secondary wall-clocks in the retro. Plan status: `approved` — Phase 2 starts immediately.

### Decisions and artefacts explicitly **not** in Stage 11 (Scope Diff cuts + intent out-of-scope)

- **D8 cut** — val-split `seq_len` offset. See D8 row.
- **D11 cut** — `evaluation/ablation.py::compute_metrics_on_holdout`. See D11 row. *Single highest-leverage cut per the Scope Diff.*
- **Cell 7 cut** — receptive-field diagram in the notebook. `PLAN POLISH`: a static image asset with no AC coverage, bounded by the same "not worth an interpretability stage" bar intent §Points applies to attention visualisation. If the facilitator wants one live, they can draw it at the whiteboard.
- **DLinear baseline row** — cut per D15.
- **Transformer variant** — deferred (not cut architecturally); D1 names TCN as the Stage 11 shipped architecture.
- **Multi-horizon training** (intent §Out of scope explicit).
- **Foundation models for time series** (intent §Out of scope explicit — TimesFM, Chronos, Lag-Llama).
- **Probabilistic variants** (intent §Out of scope explicit).
- **Training-time hyperparameter search** (intent §Out of scope explicit).
- **Knowledge distillation from large to small** (intent §Out of scope, explicitly deferred).
- **A `BaseTorchModel` ABC** (Stage 10 X7 re-affirmed; D4 extracts a function, not a class hierarchy).

---

## 2. Scope

### In scope

Transcribed from `docs/intent/11-complex-nn.md §Scope`:

- **A temporal neural model (a TCN, small Transformer, or similar) conforming to the Stage 4 interface** — `NnTemporalModel` implementing `Model`: `fit`, `predict`, `save`, `load`, `metadata`. D1 resolves the choice to TCN.
- **The data pipeline changes needed to feed sequences rather than flat feature rows** — `_SequenceDataset` (D7), lazy-window, private to `temporal.py`. Pattern A exogenous concatenation (D3).
- **A notebook that trains the model, compares it against every prior model on the same held-out period, and produces an ablation table as a reference artefact** — `notebooks/11-complex-nn.ipynb`, six-row × seven-column table (D10) built inline from registered-run predictions (D11-cut; AC-5 reconciliation).

Additionally in scope as direct consequences of the above:

- **The shared training-loop extraction to `_training.py`** (D4) — realising the Stage 10 D10 extraction seam.
- **A `NnMlpConfig` catch-up branch in `harness._build_model_from_config`** (D14) — the Stage 10 gap flagged by the codebase map, fixed in-commit.

### Out of scope (do not accidentally implement)

Transcribed from `docs/intent/11-complex-nn.md §Out of scope` + §Out of scope, explicitly deferred + items surfaced by discovery and the Scope Diff:

- **Foundation models for time series** (TimesFM, Chronos, Lag-Llama) — separate stage if pursued.
- **Probabilistic variants of the architecture** — no quantile / distributional outputs.
- **Training-time hyperparameter search** (Optuna, Ray Tune, random / grid).
- **Multi-horizon training** — single day-ahead horizon only.
- **Time-series foundation models** (explicit deferred repeat).
- **Knowledge distillation from large to small** (explicit deferred).
- **A Transformer variant** alongside the TCN — D1 deferred.
- **Attention weight visualisation** (D16 — consequence of D1).
- **Receptive-field diagram** in the notebook (Cell 7 cut per Scope Diff).
- **DLinear / NLinear as a sixth baseline row** (D15).
- **A `BaseTorchModel` abstract base** (Stage 10 X7 re-affirmed).
- **A standalone `evaluation/ablation.py` module** (D11 cut).
- **A val-split `seq_len` offset** (D8 cut).
- **Rolling-origin folds for the ablation table** — single holdout only (D12).
- **A fourth model-dispatcher site** — H-3 re-deferred; Stage 11 extends the three that exist.
- **Any change to `docs/intent/DESIGN.md §6`** — H-1 closed per the user's 2026-04-24 framing clarification.

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/11-complex-nn.md`](../../intent/11-complex-nn.md) — the contract; 5 ACs and 8 "Points for consideration".
2. [`docs/lld/research/11-complex-nn-requirements.md`](../../lld/research/11-complex-nn-requirements.md) — US-1..US-5, AC-1..AC-8 (three inferred), NFR-1..NFR-7, OQ-A..OQ-H. OQ-A through OQ-H resolved by the decisions above (OQ-A=TCN per D1; OQ-B=168 h per D2; OQ-C=N/A per D1; OQ-D=no hard gate per NFR-2; OQ-E=Pattern A per D3; OQ-F=table-only per D10; OQ-G=single-holdout per D12; OQ-H=N/A per D1).
3. [`docs/lld/research/11-complex-nn-codebase.md`](../../lld/research/11-complex-nn-codebase.md) — Stage 10 integration points, dispatch-site census, eager-vs-lazy cost calculation, AC-5 reconciliation gap.
4. [`docs/lld/research/11-complex-nn-domain.md`](../../lld/research/11-complex-nn-domain.md) — §1 (architecture family table), §2 (TCN vs Transformer trade-off), §3 (168 h literature anchor), §4 (Pattern A vs B), §6 (causal-padding recipe, `is_causal` bug, BatchNorm leakage), §8 (CPU budget), §9 (ablation prior art).
5. [`docs/lld/research/11-complex-nn-scope-diff.md`](../../lld/research/11-complex-nn-scope-diff.md) — `@minimalist` critique; every cut and retention above is listed there.
6. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
7. [`docs/plans/completed/10-simple-nn.md`](../completed/10-simple-nn.md) — the direct precedent; D4–D12 of this plan either inherit from or extend Stage 10 decisions.
8. `src/bristol_ml/models/nn/mlp.py` — the Stage 10 MLP, full precedent. Read the training loop (lines 529–628) and the save/load envelope.
9. `src/bristol_ml/models/nn/CLAUDE.md` — the five PyTorch gotchas Stage 11 inherits (`sys.modules` install for pickleable modules, lazy-torch-import, scaler-buffer registration, `weights_only=True` at load, single-joblib envelope).
10. `src/bristol_ml/models/protocol.py` — the `Model` protocol (AC-1).
11. `src/bristol_ml/evaluation/harness.py` — `evaluate()` is protocol-pure; `_build_model_from_config` (lines 536–556) gains two `isinstance` branches (D13 clause iii + D14).
12. `src/bristol_ml/registry/_dispatch.py` — `_TYPE_TO_CLASS` / `_CLASS_NAME_TO_TYPE` both gain one entry.
13. `src/bristol_ml/train.py:296–317` — the `isinstance(model_cfg, NnMlpConfig)` branch pattern the new `NnTemporalConfig` branch mirrors.
14. `scripts/_build_notebook_10.py` — builder pattern for `scripts/_build_notebook_11.py`.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five intent-ACs are copied verbatim from `docs/intent/11-complex-nn.md §Acceptance criteria`, then grounded in one or more named tests. AC-6, AC-7, AC-8 are requirements-inferred (see [requirements-analyst output](../../lld/research/11-complex-nn-requirements.md)); listed after the intent-five.

- **AC-1.** "The model conforms to the Stage 4 interface."
  - Tests:
    - `test_nn_temporal_is_model_protocol_instance` — `isinstance(NnTemporalModel(config), bristol_ml.models.Model)` is `True`.
    - `test_nn_temporal_fit_predict_round_trip_on_tiny_fixture` — `fit(X, y, seed=0)` on a 300-row fixture with `seq_len=24` then `predict(X_new)` returns a `pd.Series` of length `len(X_new) - seq_len + 1` (or equivalent — the predict contract is documented in §5 and tested against whatever the implementation ships, with the spec-of-record being that the returned series is indexable from the original feature-table timestamps the prediction covers).
- **AC-2.** "Training uses the harness established in Stage 10."
  - Tests:
    - `test_nn_temporal_fit_uses_shared_training_loop` — structural check that `NnTemporalModel.fit` calls `bristol_ml.models.nn._training.run_training_loop` (via `mock.patch`), *not* a locally-defined loop.
    - `test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction` — regression guard; `NnMlpModel.fit` calls the same helper (protects the D4 extraction from silent re-inlining).
    - `test_nn_temporal_fit_populates_loss_history_per_epoch` — `len(loss_history_) == epochs_run`; each entry has keys `{"epoch", "train_loss", "val_loss"}` with `int` epoch and `float` train/val — one assertion, preserves Stage 10 contract on the new class (Scope Diff `PLAN POLISH` kept at minimal cost).
- **AC-3.** "The ablation table covers every model trained so far on the same splits."
  - Tests:
    - `test_notebook_11_ablation_cell_covers_six_model_families` — notebook-execution smoke test asserts the rendered ablation dataframe has rows for `{"naive", "linear", "sarimax", "scipy_parametric", "nn_mlp", "nn_temporal"}` and columns `{"MAE", "MAPE", "RMSE", "WAPE", "MAE_ratio_vs_NESO", "training_time_s", "param_count"}`.
- **AC-4.** "Save/load through the registry from Stage 9 preserves full weights and the sequence preprocessing state."
  - Tests:
    - `test_registry_save_nn_temporal_model_via_protocol` — instantiate via Hydra config, fit on a small fixture, call `registry.save(model, metrics_df, feature_set=..., target=...)`, round-trip via `registry.load(run_id)`, assert `torch.allclose(original.predict(X), loaded.predict(X), atol=1e-5)` (NFR-3 close-match bar).
    - `test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict` — unit-level parallel. Asserts the loaded model's `seq_len` matches the original and every `state_dict` tensor compares equal under `torch.equal` on CPU.
    - `test_nn_temporal_save_writes_single_joblib_file_at_given_path` — structural guard that `save()` does not emit a sibling file (inherited from Stage 10 T3 pattern).
    - `test_nn_temporal_module_impl_is_pickleable` — regression guard inherited from Stage 10 lessons; pickles a fitted model, unpickles, asserts class identity and buffer preservation.
- **AC-5.** "The notebook's ablation table is reproducible from the registry without re-training anything already registered."
  - Tests:
    - `test_notebook_11_ablation_cell_does_not_refit_registered_runs` — notebook-execution smoke test monkeypatches every registered model's `fit` to `raise RuntimeError("AC-5 violation: registered run was re-fit")`. The ablation cell executes cleanly; only `nn_temporal` is fit as a fresh run before the table is built.
  - Evidence (non-test): the ablation-table notebook cell uses `for run_id in runs: model = registry.load(run_id); preds = model.predict(X_holdout); ...` — a pure predict-only path by construction (D11 rationale).

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_nn_temporal_seeded_runs_produce_identical_state_dicts_on_cpu` (NFR-1 CPU bit-identity, pins `device="cpu"`).
- `test_nn_temporal_seeded_runs_match_on_cuda_within_tolerance` (NFR-1 GPU close-match, `@pytest.mark.gpu`, skipped without CUDA).
- `test_nn_temporal_different_seeds_produce_different_state_dicts` — regression guard that the seed flows through.
- `test_nn_temporal_fit_uses_cold_start_per_fold_when_called_repeatedly` — cold-start contract (D9 inheritance).
- `test_nn_temporal_standalone_cli_exits_zero` (NFR-5).
- `test_nn_temporal_config_schema_defaults_round_trip_through_hydra` — resolved defaults match `NnTemporalConfig(...)` exactly, one test row rather than per-field (Stage 10 pattern).
- `test_nn_temporal_config_rejects_seq_len_smaller_than_receptive_field` — the D2 Pydantic validator.
- `test_sequence_dataset_lazy_window_getitem_matches_pandas_slice` — T2 unit-level check that `_SequenceDataset[i]` returns the tensorised `(features[i:i+seq_len], target[i+seq_len])` pair.
- `test_sequence_dataset_len_is_rows_minus_seq_len` — boundary arithmetic.
- `test_nn_temporal_causal_padding_does_not_leak_future` — a synthetic fixture where every "future" sample is infinity; the model's prediction on step T depends only on `x[:T]`, so the output is finite — any right-pad leakage would produce `inf`/`nan`.
- `test_nn_temporal_is_dispatched_by_train_cli_isinstance_branch` (D13 clause ii — structural parallel of the Stage 10 test).
- `test_harness_build_model_from_config_dispatches_nn_temporal` (D13 clause iii).
- `test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up` (D14).
- `test_train_cli_registers_nn_temporal_final_fold_model` — full pipeline integration test, `python -m bristol_ml.train model=nn_temporal` on the warm feature-table fixture; exactly one new `run_id` in a tmp registry dir; `registry.list_runs(model_type="nn_temporal")` returns it.

**Total shipped tests: 20** (two AC-1, three AC-2, one AC-3, four AC-4, one AC-5, nine D-derived).

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/models/nn/
├── __init__.py          # exports: NnMlpModel, NnTemporalModel
├── __main__.py          # unchanged — delegates to mlp._cli_main
├── mlp.py               # NnMlpModel (refactored to call _training.run_training_loop)
├── temporal.py          # NnTemporalModel (new)
├── _training.py         # shared training-loop helper (new; D4)
└── CLAUDE.md            # extended with Stage 11 additions (D17)
```

`_training.py` holds:

```python
# src/bristol_ml/models/nn/_training.py
def _seed_four_streams(seed: int, device: "torch.device") -> None: ...

def run_training_loop(
    module: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    *,
    optimiser: "torch.optim.Optimizer",
    criterion: "nn.Module",
    device: "torch.device",
    max_epochs: int,
    patience: int,
    loss_history: list[dict[str, float]],
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[dict[str, "Tensor"], int]:
    """Run the shared PyTorch training loop. Returns (best_state_dict, best_epoch).

    Populates ``loss_history`` in place as a list of dicts with keys
    {"epoch": int, "train_loss": float, "val_loss": float}.
    Invokes ``epoch_callback(dict(entry))`` after each epoch (defensive copy).
    """
```

The function is the exact body of Stage 10 `NnMlpModel._run_training_loop` refactored to take its collaborators as arguments (module, dataloaders, optimiser, criterion) rather than closing over `self`. Both `mlp.py` and `temporal.py` build those collaborators and then call `run_training_loop`.

### Config schema (addition)

```python
# conf/_schemas.py
class NnTemporalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["nn_temporal"] = "nn_temporal"

    target_column: str = "nd_mw"
    feature_columns: tuple[str, ...] | None = None

    # Architecture (D1 — CUDA defaults; CPU users override via Hydra)
    seq_len: int = Field(default=168, ge=2)
    num_blocks: int = Field(default=8, ge=1, le=12)
    channels: int = Field(default=128, ge=8, le=512)
    kernel_size: int = Field(default=3, ge=2, le=7)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    weight_norm: bool = True

    # Optimisation (D1 — CUDA defaults; CPU users override)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)

    # Reproducibility (D6)
    seed: int | None = None

    # Device (Stage 10 D11)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    @model_validator(mode="after")
    def _seq_len_covers_receptive_field(self) -> "NnTemporalConfig":
        # receptive field = 1 + 2*(kernel_size-1)*(2^num_blocks - 1)
        receptive = 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_blocks - 1)
        if self.seq_len < max(2 * self.kernel_size, receptive // 8):
            raise ValueError(
                f"seq_len={self.seq_len} is too small for the requested "
                f"receptive field (~{receptive}); either raise seq_len or "
                f"reduce num_blocks/kernel_size."
            )
        return self
```

And `conf/model/nn_temporal.yaml`:

```yaml
# @package model
#
# Defaults target CUDA (Blackwell dev host per Stage 10 D1). For a CPU-only
# laptop demo, override via Hydra CLI — recommended CPU recipe:
#   uv run python -m bristol_ml.train model=nn_temporal \
#     model.num_blocks=4 model.channels=32 model.batch_size=32 \
#     model.max_epochs=20 model.device=cpu
# Observed CUDA wall-clock at defaults: < 5 min on Blackwell. Observed CPU
# wall-clock at the recipe above: < 10 min on a 4-core laptop. Numbers
# recorded in docs/lld/stages/11-complex-nn.md.
type: nn_temporal
target_column: nd_mw
feature_columns: null
seq_len: 168
num_blocks: 8
channels: 128
kernel_size: 3
dropout: 0.2
weight_norm: true
learning_rate: 1.0e-3
weight_decay: 0.0
batch_size: 256
max_epochs: 100
patience: 10
seed: null
device: auto
```

`ModelConfig` discriminated union:

```python
# conf/_schemas.py
ModelConfig = (
    NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig
    | NnMlpConfig | NnTemporalConfig
)
```

### On-disk artefact layout (`NnTemporalModel.save(path)`)

Single joblib artefact at `artefact/model.joblib`:

```python
{
    "state_dict_bytes": bytes,       # torch.save(state_dict, BytesIO).getvalue()
    "config_dump": dict[str, Any],   # NnTemporalConfig.model_dump()
    "feature_columns": tuple[str, ...],
    "seq_len": int,                  # redundant with config_dump but explicit for D2 round-trip test
    "seed_used": int,
    "best_epoch": int,
    "loss_history": list[dict[str, float]],
    "fit_utc": str,                  # ISO-8601, UTC
    "device_resolved": str,          # "cpu" | "cuda" | "mps"
}
```

`state_dict_bytes` carries every trainable parameter (Conv1d kernels + LayerNorm affine), every `register_buffer` (scaler buffers: `feature_mean`, `feature_std`, `target_mean`, `target_std`), and — because `weight_norm` wraps `Conv1d` — both the `weight_g` and `weight_v` tensors per wrapped layer.

### Public interface (new)

```python
# src/bristol_ml/models/nn/temporal.py

class NnTemporalModel:
    """Temporal convolutional network conforming to Stage 4's ``Model`` protocol.

    See ``docs/architecture/layers/models-nn.md`` §"Stage 11 addition" for the
    full contract.
    """

    def __init__(self, config: NnTemporalConfig) -> None: ...

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
    def load(cls, path: Path) -> NnTemporalModel: ...

    @property
    def metadata(self) -> ModelMetadata: ...

    # Stage-11-specific public attribute (populated after fit)
    loss_history_: list[dict[str, float]]
```

### Dispatch sites touched (three — per D13 + D14)

1. **`conf/_schemas.py`** — add `NnTemporalConfig`; extend `ModelConfig` union.
2. **`conf/model/nn_temporal.yaml`** — new Hydra group entry.
3. **`src/bristol_ml/train.py`** — add `isinstance(model_cfg, NnTemporalConfig)` branch mirroring the `NnMlpConfig` branch. Extend `_target_column()` to cover `NnTemporalConfig`.
4. **`src/bristol_ml/registry/_dispatch.py`** — add `"nn_temporal"` / `NnTemporalModel` to both dicts.
5. **`src/bristol_ml/evaluation/harness.py::_build_model_from_config`** — add `NnTemporalConfig` branch (D13 clause iii) *and* the missing `NnMlpConfig` branch (D14).

### Notebook structure (`notebooks/11-complex-nn.ipynb`)

Built by `scripts/_build_notebook_11.py` following the Stage 10 pattern. Six cells (Cell 7 cut per Scope Diff):

1. **Preamble** — Hydra config load, seed, device resolve, split fixture load.
2. **Load feature table + single-holdout split** (D12). Assert or register the five prior-stage runs if missing (AC-6 inferred). The training cells for prior runs are `# skip if already registered` guarded.
3. **Fit `NnTemporalModel` with live loss-curve** via `epoch_callback` seam (Stage 10 D6 pattern, reused unchanged — `plots.loss_curve` already exists). Intent §Demo moment payoff lives here.
4. **`registry.save(model, ...)` with sidecar metadata** — demonstrates AC-4 end to end.
5. **Ablation cell (AC-3 / AC-5 / D10 / D11)** — inline loop over registered run_ids, `registry.load` → `model.predict(X_holdout)` → metric composition → `pd.DataFrame` → `df.to_markdown()`. The seven-column table. No helper module.
6. **Commentary** — markdown cell: "Each row bought us this much accuracy" narrative.

### Training-loop contract (D4)

```python
# src/bristol_ml/models/nn/mlp.py  (refactored NnMlpModel.fit body, abbreviated)
def fit(self, features, target, *, seed=None, epoch_callback=None):
    _seed_four_streams(int(effective_seed), self._device)
    # build TensorDataset / DataLoader as before
    self._module = _NnMlpModule(...)  # unchanged
    optimiser = torch.optim.Adam(...)
    criterion = nn.MSELoss()
    self.loss_history_ = []
    best_state, self._best_epoch = run_training_loop(
        self._module, train_loader, val_loader,
        optimiser=optimiser, criterion=criterion, device=self._device,
        max_epochs=self._config.max_epochs, patience=self._config.patience,
        loss_history=self.loss_history_, epoch_callback=epoch_callback,
    )
    self._module.load_state_dict(best_state, strict=True)
```

```python
# src/bristol_ml/models/nn/temporal.py  (NnTemporalModel.fit body, abbreviated)
def fit(self, features, target, *, seed=None, epoch_callback=None):
    _seed_four_streams(int(effective_seed), self._device)
    dataset = _SequenceDataset(features, target, seq_len=self._config.seq_len)
    # train / val split + DataLoader construction
    self._module = _NnTemporalModule(...)
    optimiser = torch.optim.Adam(...)
    criterion = nn.MSELoss()
    self.loss_history_ = []
    best_state, self._best_epoch = run_training_loop(
        self._module, train_loader, val_loader,
        optimiser=optimiser, criterion=criterion, device=self._device,
        max_epochs=self._config.max_epochs, patience=self._config.patience,
        loss_history=self.loss_history_, epoch_callback=epoch_callback,
    )
    self._module.load_state_dict(best_state, strict=True)
```

The loops are structurally identical below the collate layer. The extraction is a mechanical parameter-lift.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Extract shared training loop to `_training.py`; refactor Stage 10 MLP to use it

**Files:**
- new `src/bristol_ml/models/nn/_training.py` — `_seed_four_streams(seed, device)` (moved from `mlp.py`) + `run_training_loop(...)` (lifted from Stage 10 `NnMlpModel._run_training_loop`).
- modified `src/bristol_ml/models/nn/mlp.py` — `fit()` calls `run_training_loop`; in-module `_run_training_loop` and `_seed_four_streams` removed. Module docstring line 9–11 updated: "D10 extraction seam; Stage 11 has fired the trigger — see `_training.py`."
- modified `src/bristol_ml/models/nn/CLAUDE.md` — Gotcha section updated: "As of Stage 11 the shared training loop lives in `_training.py`; both `NnMlpModel` and `NnTemporalModel` import from there."

**Tests (T1):**
- `test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction` — structural regression guard; `NnMlpModel.fit` invokes `bristol_ml.models.nn._training.run_training_loop` exactly once per call (via `mock.patch`).
- **All existing Stage 10 tests must pass unchanged.** Any failure in `tests/unit/models/test_nn_mlp_*` is a refactor regression and halts the stage.

**Commits as:** `Stage 11 T1: extract shared training loop to _training.py (Stage 10 D10 seam)`.

### Task T2 — `_SequenceDataset` + lazy-window unit tests

**Files:**
- new `src/bristol_ml/models/nn/temporal.py` with only `_SequenceDataset` defined (not yet the model class).
- extended `conf/_schemas.py` — add `NnTemporalConfig` (skeleton with all fields + the `@model_validator`); extend `ModelConfig` union.
- new `conf/model/nn_temporal.yaml`.

**Tests (T2):**
- `test_sequence_dataset_len_is_rows_minus_seq_len`.
- `test_sequence_dataset_lazy_window_getitem_matches_pandas_slice` — `_SequenceDataset[i]` returns the correct `(features[i:i+seq_len], target[i+seq_len])` pair as tensors on the expected dtype.
- `test_sequence_dataset_does_not_eagerly_materialise_full_tensor` — creates a 50,000-row fixture and asserts `_SequenceDataset.__init__` does not allocate a tensor larger than the input frame (structural guard against accidentally regressing to the eager pattern).
- `test_nn_temporal_config_schema_defaults_round_trip_through_hydra`.
- `test_nn_temporal_config_rejects_seq_len_smaller_than_receptive_field`.

**Commits as:** `Stage 11 T2: _SequenceDataset + NnTemporalConfig + nn_temporal.yaml`.

### Task T3 — `NnTemporalModel` scaffold + protocol conformance + standalone CLI

**Files:**
- modified `src/bristol_ml/models/nn/temporal.py` — add the class (with `_build_temporal_module_class` + `sys.modules` install per Stage 10 Gotcha 1); `__init__`, `metadata` property, `save`/`load` as `NotImplementedError` stubs, `fit`/`predict` also as stubs. Standalone CLI at module level prints resolved `NnTemporalConfig`.
- modified `src/bristol_ml/models/nn/__init__.py` — add `NnTemporalModel` to `__all__` and the `__getattr__` lazy-export branch.

**Tests (T3):**
- `test_nn_temporal_is_model_protocol_instance` (AC-1 scaffold half).
- `test_nn_temporal_standalone_cli_exits_zero` (NFR-5).
- `test_nn_temporal_lazy_torch_import_contract` — `python -m bristol_ml.models.nn --help` does not import `torch` (structural check on `sys.modules`).

**Commits as:** `Stage 11 T3: NnTemporalModel scaffold + protocol conformance + standalone CLI`.

### Task T4 — `fit` + `predict` + training loop integration + loss history

**Files:** `src/bristol_ml/models/nn/temporal.py` (implement `fit`, `predict`, `_build_temporal_module_class` body with the 6-block dilated causal TCN + LayerNorm + weight-norm + dropout; causal padding via `F.pad(x, (pad, 0))`).

**Content:**
- `fit()` instantiates the temporal module on the resolved device (Stage 10 D11 `_select_device`), fits scaler buffers on the train slice, builds the `_SequenceDataset` + `DataLoader` (val tail = last 10 % of train contiguous by index, per Stage 10 D9; **no D8 offset** per Scope Diff cut), constructs `Adam` + `MSELoss`, calls `run_training_loop`, restores best-epoch weights.
- `predict()` switches to `eval()`, wraps input as a single-batch `_SequenceDataset` or equivalent, produces one prediction per valid window, returns a `pd.Series` indexed on the target timestamps.

**Tests (T4):**
- `test_nn_temporal_fit_predict_round_trip_on_tiny_fixture` (AC-1, pins `device="cpu"`).
- `test_nn_temporal_seeded_runs_produce_identical_state_dicts_on_cpu` (NFR-1 CPU bit-identity, pins `device="cpu"`; `torch.equal` on every parameter tensor + `predict` output).
- `test_nn_temporal_seeded_runs_match_on_cuda_within_tolerance` — `@pytest.mark.gpu`, skipped unless `torch.cuda.is_available()`; `atol=1e-5, rtol=1e-4`.
- `test_nn_temporal_different_seeds_produce_different_state_dicts`.
- `test_nn_temporal_fit_populates_loss_history_per_epoch` (AC-2; keys + `int`-typed epoch; one assertion per key).
- `test_nn_temporal_fit_invokes_epoch_callback_when_provided` — defensive-copy semantics (inherited from Stage 10 Phase 3 lesson).
- `test_nn_temporal_fit_uses_cold_start_per_fold_when_called_repeatedly`.
- `test_nn_temporal_fit_uses_shared_training_loop` (AC-2 — `run_training_loop` is called exactly once).
- `test_nn_temporal_causal_padding_does_not_leak_future` — `inf` in a strictly-future position yields finite output; any right-pad leakage produces `inf`/`nan`.

**Commits as:** `Stage 11 T4: NnTemporalModel fit/predict + TCN body + causal padding`.

### Task T5 — `save` / `load` + single-joblib artefact envelope

**Files:** `src/bristol_ml/models/nn/temporal.py` (implement `save`, `load`).

**Content:**
- `save(path)` serialises `self._module.state_dict()` into `BytesIO` via `torch.save`, assembles the envelope dict (§5), writes it via `bristol_ml.models.io.save_joblib(path, envelope)`.
- `load(path)` reads the joblib envelope, reconstructs `NnTemporalConfig` from `config_dump`, instantiates `NnTemporalModel(config)`, materialises the `state_dict` via `torch.load(BytesIO(state_dict_bytes), weights_only=True, map_location="cpu")`, calls `load_state_dict(strict=True)`, restores scalar attributes.

**Tests (T5):**
- `test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict` (AC-4 unit level; `torch.equal` on all tensors; `seq_len` field round-trip).
- `test_nn_temporal_load_raises_file_not_found_for_missing_artefact`.
- `test_nn_temporal_save_writes_single_joblib_file_at_given_path` (Stage 10 T3 structural guard inherited).
- `test_nn_temporal_module_impl_is_pickleable` (Stage 10 Phase 3 regression pattern inherited).

**Commits as:** `Stage 11 T5: NnTemporalModel save/load + single-joblib artefact envelope`.

### Task T6 — Dispatcher wiring + Stage 10 harness-factory catch-up

**Files:**
- modified `src/bristol_ml/registry/_dispatch.py` — add `"nn_temporal"` / `NnTemporalModel` entries to both dicts (D13 clause i).
- modified `src/bristol_ml/train.py` — add `isinstance(model_cfg, NnTemporalConfig)` branch mirroring `NnMlpConfig` (D13 clause ii); extend `_target_column()` tuple.
- modified `src/bristol_ml/evaluation/harness.py` — add **two** branches to `_build_model_from_config`: the `NnTemporalConfig` branch (D13 clause iii) **and** the missing `NnMlpConfig` branch (D14 catch-up).
- modified `src/bristol_ml/models/nn/CLAUDE.md` — note the harness-factory catch-up in the "Stage 11 additions" section.

**Tests (T6):**
- `test_registry_save_nn_temporal_model_via_protocol` (AC-4 end-to-end).
- `test_registry_list_runs_includes_nn_temporal_type` — sidecar `type` field.
- `test_nn_temporal_is_dispatched_by_train_cli_isinstance_branch`.
- `test_harness_build_model_from_config_dispatches_nn_temporal` (D13 clause iii).
- `test_harness_build_model_from_config_dispatches_nn_mlp_after_catch_up` (D14 — regression against the Stage 10 gap).
- `test_train_cli_registers_nn_temporal_final_fold_model` — full pipeline integration.

**Commits as:** `Stage 11 T6: dispatcher wiring for nn_temporal + Stage 10 NnMlpConfig catch-up`.

### Task T7 — Notebook + notebook builder script

**Files:**
- new `scripts/_build_notebook_11.py` — mirrors `scripts/_build_notebook_10.py`. Six cells per §5 structure.
- new `notebooks/11-complex-nn.ipynb` — produced by running the builder + `jupyter nbconvert --execute --inplace` + `ruff format`.

**Tests (T7):**
- `test_notebook_11_ablation_cell_covers_six_model_families` (AC-3).
- `test_notebook_11_ablation_cell_does_not_refit_registered_runs` (AC-5).
- Notebook executes cleanly in `test_notebooks_execute_cleanly` (inherited pattern).

**Commits as:** `Stage 11 T7: notebook + _build_notebook_11.py + ablation table`.

### Task T8 — Stage hygiene (H-2, H-3) + layer doc + retro

**Files:**
- modified `docs/architecture/layers/models-nn.md` — new "Stage 11 addition" section (D17): `NnTemporalModel` contract, `_SequenceDataset`, `seq_len`, receptive field, Pattern A exogenous handling, D4 seam status ("fired as of Stage 11"), D14 harness-factory catch-up note.
- modified `src/bristol_ml/models/nn/CLAUDE.md` — Stage 11 TCN-specific gotchas (causal padding, weight-norm placement, `_SequenceDataset` lazy-window contract, single-joblib `seq_len` field).
- new `docs/lld/stages/11-complex-nn.md` — retro per the template (observed CPU wall-clock, NESO-ratio result, any AC-5 gotchas surfaced, D4-extraction-experience note).
- modified `docs/lld/stages/10-simple-nn.md` — confirm "Next" pointer to Stage 11 (H-2).
- modified `CHANGELOG.md` — `[Unreleased]` Stage 11 bullets under `### Added`.
- modified `docs/architecture/README.md` — module catalogue row refresh (`models/nn/` gains `temporal.py` + `_training.py`).
- moved `docs/plans/active/11-complex-nn.md` → `docs/plans/completed/11-complex-nn.md` (final commit of T8).

**Tests (T8):** none new (hygiene-only; existing layer-doc / retro presence tests cover).

**Commits as:** `Stage 11 T8: stage hygiene + retro + plan moved to completed/`.

---

## 7. Files expected to change

### New

- `src/bristol_ml/models/nn/_training.py`
- `src/bristol_ml/models/nn/temporal.py`
- `conf/model/nn_temporal.yaml`
- `tests/unit/models/test_nn_temporal_scaffold.py`
- `tests/unit/models/test_nn_temporal_fit_predict.py`
- `tests/unit/models/test_nn_temporal_save_load.py`
- `tests/unit/models/test_sequence_dataset.py`
- `tests/unit/models/test_nn_training_shared.py` (the D4 shared-loop structural regression guards — both MLP and temporal call `run_training_loop`).
- `tests/unit/registry/test_registry_nn_temporal_dispatch.py`
- `tests/unit/evaluation/test_harness_build_model_from_config_dispatches_nn_models.py` (D13 clause iii + D14)
- `tests/integration/test_train_cli_registers_nn_temporal.py`
- `scripts/_build_notebook_11.py`
- `notebooks/11-complex-nn.ipynb`
- `docs/lld/stages/11-complex-nn.md`

### Modified

- `conf/_schemas.py` — add `NnTemporalConfig`; extend `ModelConfig` union.
- `src/bristol_ml/models/nn/__init__.py` — export `NnTemporalModel` via lazy `__getattr__`.
- `src/bristol_ml/models/nn/mlp.py` — `_run_training_loop` and `_seed_four_streams` moved to `_training.py`; `fit()` refactored to call `run_training_loop` (D4 extraction).
- `src/bristol_ml/models/nn/CLAUDE.md` — Stage 11 TCN-specific additions (D17).
- `src/bristol_ml/train.py` — add `NnTemporalConfig` branch + `_target_column` extension (D13 clause ii).
- `src/bristol_ml/evaluation/harness.py` — two new `isinstance` branches in `_build_model_from_config` (D13 clause iii + D14).
- `src/bristol_ml/registry/_dispatch.py` — one new key in each of `_TYPE_TO_CLASS` and `_CLASS_NAME_TO_TYPE`.
- `docs/architecture/layers/models-nn.md` — Stage 11 addition section (D17).
- `docs/architecture/README.md` — module catalogue refresh.
- `docs/lld/stages/10-simple-nn.md` — "Next" pointer verified (H-2).
- `CHANGELOG.md` — `[Unreleased]` Stage 11 bullets.

### Moved (final commit of T8)

- `docs/plans/active/11-complex-nn.md` → `docs/plans/completed/11-complex-nn.md`.

### Explicitly NOT modified

- `docs/intent/DESIGN.md` — §6 unchanged (H-1 closed per user framing: "§6 is intended as structural-only, should only need to be updated very occasionally"). `docs/intent/11-complex-nn.md` immutable.
- `src/bristol_ml/models/protocol.py` — protocol signature unchanged (AC-1 contract).
- `src/bristol_ml/models/{naive,linear,sarimax,scipy_parametric}.py` — no changes.
- `src/bristol_ml/evaluation/benchmarks.py` — `compare_on_holdout` unchanged; the ablation cell uses `registry.load` + `model.predict` directly (D11 cut; AC-5 reconciliation).
- `src/bristol_ml/evaluation/plots.py` — `loss_curve` unchanged and reused.
- `src/bristol_ml/evaluation/metrics.py` — no new metrics; the ablation cell composes existing metric functions.
- `src/bristol_ml/registry/__init__.py` — no change to the four-verb surface.
- `pyproject.toml`, `Dockerfile`, `uv.lock` — no changes.

---

## 8. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | The D4 extraction regresses Stage 10 behaviour — `NnMlpModel` tests go red after T1. | Medium | High | T1 runs the full `tests/unit/models/test_nn_mlp_*` suite as its exit gate; any regression halts the stage before temporal-model code is written. The extraction is a mechanical parameter-lift (no logic change); if the tests fail, the diff is small and the revert is cheap. |
| **R2** | Causal-padding implementation leaks future — right-pad bug per domain §6 pitfall 4. Training metrics look great; holdout predictions are secretly wrong. | Medium | High | `test_nn_temporal_causal_padding_does_not_leak_future` uses a synthetic `inf`-at-future-index fixture; any right-side leak produces `inf`/`nan` in the output, making the bug loud rather than silent. Plus the explicit `F.pad(x, (pad, 0))` + `Conv1d(padding=0)` recipe (Bai et al. 2018) is simpler to audit than any implicit padding mode. |
| **R3** | A CPU-only user accidentally runs the CUDA-sized defaults and waits 30+ min, or the CUDA defaults miss the < 5 min target on Blackwell. | Medium | Low | Three mitigations compound: (i) the CPU-recipe snippet is prominent in the `nn_temporal.yaml` header comment block, the `models-nn.md` layer doc §"Stage 11 addition", and the notebook's preamble cell; (ii) `device=auto` resolves to CPU when no GPU is present so the facilitator gets a clear device-log line at fit-start naming what they are running on; (iii) the retro records both observed wall-clocks (CUDA default + CPU recipe). If CUDA misses < 5 min or CPU override misses < 10 min, the defaults are re-visited in a follow-on stage — not a blocker for this one. |
| **R4** | `weight_norm` deprecation in a future PyTorch version. `torch.nn.utils.weight_norm` has been "deprecated in favour of `torch.nn.utils.parametrizations.weight_norm`" since PyTorch 2.1. | Low | Low | Pin `torch>=2.7,<3` is unchanged from Stage 10. The deprecated API still works in 2.7 with a `FutureWarning`. If the warning fails a future pre-commit hook, switch to `parametrizations.weight_norm`; the state_dict shape differs (`weight_g`/`weight_v` become `parametrizations.weight.original0`/`original1`), so load-path compatibility is a concrete migration question — documented in the retro. |
| **R5** | AC-5 violation: a registered run is silently re-fit by the ablation cell. | Low | Medium | `test_notebook_11_ablation_cell_does_not_refit_registered_runs` monkeypatches every registered model's `fit` to raise `RuntimeError`; the test passes only if the cell takes the predict-only path. Notebook execution in CI catches regressions. |
| **R6** | The receptive-field validator (D2) rejects a legitimate facilitator config that deliberately trims `seq_len` for demo speed. | Low | Low | The validator's rule is heuristic (`seq_len >= max(2*kernel_size, receptive_field // 8)`) — intentionally loose. A facilitator who wants `seq_len=24` can simultaneously drop `num_blocks` to 3; both are exposed. If the validator proves too tight in practice, it can be loosened in a later stage. |
| **R7** | `_SequenceDataset` fails to round-trip through `NnTemporalModel.load` because `seq_len` is read from `config_dump` rather than the envelope's dedicated `seq_len` field. | Low | Medium | `test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict` asserts the loaded model's `config.seq_len` matches the saved `seq_len` field byte-exact. The envelope's redundant `seq_len` field is the load-path authority; `config_dump` is the validator. |
| **R8** | AC-6 (requirements-inferred — all prior model families must be registered before the ablation runs) is latent: if the Stage 10 run registry is empty at demo time the notebook fails with a confusing error. | Medium | Low | Cell 2 of the notebook explicitly checks `registry.list_runs(model_type=...)` for each prior family; if any is missing, it runs the minimal fit-then-save cell for that family inline before proceeding. Guards the demo against a cold registry without requiring a separate "populate registry" script. |

---

## 9. Exit checklist

Verified before T8's final commit.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail` without a linked issue.
- [ ] Ruff + format + pre-commit clean: `uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- [ ] `uv run python -m bristol_ml.models.nn.temporal --help` exits 0 and prints the resolved `NnTemporalConfig` schema, including `seq_len: 168`, `num_blocks: 8`, `channels: 128`, `kernel_size: 3` (NFR-5 + D1 amended at Ctrl+G + D2).
- [ ] `uv run python -m bristol_ml.train model=nn_temporal` leaves exactly one new `run_id` in `data/registry/`.
- [ ] `uv run python -m bristol_ml.registry list --model-type nn_temporal` prints the new run.
- [ ] `uv run python -m bristol_ml.registry describe <nn_temporal_run_id>` prints a sidecar whose `type` field is `"nn_temporal"`.
- [ ] On the CUDA dev host, `uv run pytest -m gpu -q` passes (NFR-1 CUDA close-match). On CPU-only CI, the `gpu` marker is skipped by the existing `addopts` filter.
- [ ] The five intent-ACs (AC-1..AC-5) map to the named tests in §4 and all pass.
- [ ] AC-5 — the ablation cell's `fit`-is-`raise` monkeypatch test passes.
- [ ] `docs/architecture/layers/models-nn.md` extended with the Stage 11 addition section; `src/bristol_ml/models/nn/CLAUDE.md` extended with TCN-specific gotchas.
- [ ] `docs/lld/stages/11-complex-nn.md` retro written per template, including the observed CPU wall-clock (NFR-2 informational), the ablation table's final contents, and the MAE-ratio-vs-NESO numeric result.
- [ ] `CHANGELOG.md` updated with the Stage 11 bullets under `[Unreleased]`.
- [ ] `docs/architecture/README.md` module catalogue refresh.
- [ ] `docs/plans/active/11-complex-nn.md` moved to `docs/plans/completed/`.
- [ ] H-1 (DESIGN §6) closed per user framing; H-2 (Stage 10 retro pointer) verified; H-3 (dispatcher-consolidation ADR) re-deferred.
- [ ] The D4 extraction fired: `_training.py` exists, both `NnMlpModel.fit` and `NnTemporalModel.fit` import and call `run_training_loop`, Stage 10 test suite is green.
- [ ] D14 catch-up landed: `harness._build_model_from_config` now dispatches both `NnMlpConfig` and `NnTemporalConfig`.
- [ ] PR description includes: Stage 11 summary, observed wall-clock, final ablation-table contents (six rows × seven columns) reproduced inline, Scope Diff link, any Phase 3 review findings, H-1 closure note.
