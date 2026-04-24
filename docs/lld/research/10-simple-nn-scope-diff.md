# Stage 10 — Simple NN — Scope Diff

*Pre-synthesis scope critique by `@minimalist` on the three Phase 1
research artefacts plus the lead's draft decision set. Read-only output;
the lead reconsiders every `PLAN POLISH` and `PREMATURE OPTIMISATION`
row before binding it into `docs/plans/active/10-simple-nn.md`.*

## Inputs reviewed

- `docs/intent/10-simple-nn.md` (authoritative scope)
- `docs/lld/research/10-simple-nn-requirements.md`
- `docs/lld/research/10-simple-nn-codebase.md`
- `docs/lld/research/10-simple-nn-domain.md`

## Scope Diff

| ID | Item | Tag | Rationale |
|----|------|-----|-----------|
| D1 | Add `torch` (CPU build) as runtime dependency | `RESTATES INTENT` | AC-1 / AC-5 require a trainable MLP; PyTorch is named in DESIGN §8. No torch = no stage. |
| D2 | `NnMlpModel` in `models/nn/mlp.py`, dispatched through all five sites (schemas, Hydra YAML, harness, train, registry dispatch) | `RESTATES INTENT` | AC-1 requires protocol conformance and AC-4 requires registry round-trip; all five dispatch sites are mandatory per codebase precedent (Stage 7/8 surprise 3). |
| D3 | Default architecture `128×1, ReLU, Adam lr=1e-3, max_epochs=100, patience=10, batch_size=32` | `PLAN POLISH` | Intent says "one or two hidden layers, moderate width, standard activations" — any conforming default ships; the specific numerical choices are underconstrained by the intent. Fixing these numbers binds one config-schema test row per field (roughly 6 fields × 1 assertion = 6 test conditions). |
| D4 | Z-score normalisation via `register_buffer` inside `nn.Module` | `RESTATES INTENT` | Intent §Points ("How normalisation is done, and where the statistics live, matters for save/load and for serving later") directly names this as load-bearing; AC-4 requires clean registry round-trip of weights including scaler stats. |
| D5 | Persist with `torch.save(state_dict, path)` not joblib; custom loader in `registry/_dispatch.py` | `RESTATES INTENT` | AC-4 requires clean save/load round-trip; domain research R3 documents that joblib of `nn.Module` breaks under PyTorch 2.6+ `weights_only=True` default. Technically load-bearing. |
| D6 | `loss_history_` attribute + new `evaluation.plots.loss_curve(history)` helper shipped to `plots.py` | `RESTATES INTENT` | AC-3 requires loss curve available as a plot without additional wiring; the `plots.py` helper is the only design that respects `evaluation/CLAUDE.md`'s model-agnosticism rule. |
| D7 | Three-stream seeding helper (`torch.manual_seed`, NumPy, Python `random`) + `torch.use_deterministic_algorithms(True)` + `num_workers=0` | `PLAN POLISH` | AC-2 / NFR-1 require reproducibility given a seed; `torch.manual_seed` alone satisfies intent's "reasonable reproducibility given the same random seed." The `use_deterministic_algorithms(True)` and three-stream helper exceed the intent's stated bar and add ~1 test (`T1`). Author should justify or trim to `torch.manual_seed` + `num_workers=0`. |
| D8 | Cold-start per fold, per-fold seed derived as `seed + fold_index` | `RESTATES INTENT` | Intent §Points ("How to handle the rolling-origin folds") and domain research R5 (cold-start generalises better) both name this; AC-2 reproducibility requires deterministic per-fold seeding. |
| D9 | Patience-based early stopping with best-epoch weight restore | `RESTATES INTENT` | Intent §Scope explicitly names "early-stopping and checkpointing tied to the registry from Stage 9." |
| D10 | Hand-rolled training loop in `_run_training_loop`, refactor seam flagged for Stage 11 | `RESTATES INTENT` | Intent §Points names loop ownership as the central scope question; domain research R7 recommends hand-rolled for meetup walkability (DESIGN §1.1). |
| D11 | CPU-only, `torch.device("cpu")` hardcoded | `RESTATES INTENT` | Intent §Out of scope: "Distributed or multi-GPU training." Intent §Points says "CPU-only is fine" for laptop demos. |
| D12 | `src/bristol_ml/models/nn/CLAUDE.md` + `docs/architecture/layers/models-nn.md` | `HOUSEKEEPING` | Every shipped module layer carries a `CLAUDE.md` and a layer doc per project convention (`CLAUDE.md` stage hygiene). |
| T1 | Unit: bit-identical weights and predictions across two `fit(seed=0)` runs | `RESTATES INTENT` | Directly tests AC-2 (reproducibility given a seed). |
| T2 | Unit: registry round-trip `atol=1e-10` | `RESTATES INTENT` | Directly tests AC-4 (registry round-trip). |
| T3 | Unit: early stopping terminates before `max_epochs` on plateau | `RESTATES INTENT` | Directly tests AC-2 / D9 (early stopping is named in intent §Scope). |
| T4 | Unit: `loss_history_` populated, no NaN | `RESTATES INTENT` | Directly tests AC-3 (loss curve produced by the training loop). |
| T5 | Integration: full `train.py --model nn_mlp` end-to-end | `RESTATES INTENT` | AC-4 and AC-1 together require end-to-end Hydra wiring and registry creation. |
| T6 | Integration: harness dispatches `nn_mlp` to `NnMlpModel` | `RESTATES INTENT` | AC-1 requires protocol conformance via the harness; Stage 7/8 precedent requires both dispatch sites tested. |
| T7 | Unit: `plots.loss_curve(history)` produces a valid Figure | `RESTATES INTENT` | Directly tests AC-3 (loss curve available as a plot). |
| T8 | Unit: normalisation buffers round-trip through `state_dict()` | `PLAN POLISH` | Already covered by T2 (registry round-trip `atol=1e-10` implicitly requires scaler stats to survive); T8 is a redundant slice of T2. Forces one extra test file touch with no new behavioural coverage. |
| NFR-1 | Seeded runs bit-identical | `RESTATES INTENT` | AC-2. |
| NFR-2 | CPU training ≤3 min on 4-core laptop | `PREMATURE OPTIMISATION` | AC-5 says "reasonable time on a laptop CPU" — the intent deliberately leaves the ceiling vague. The requirements analyst flags NFR-2 as "tentative — unverified." A wall-clock gate that cannot be verified pre-implementation adds a flaky or arbitrarily-relaxed test; no AC names the 3-minute figure. Forces either a `@pytest.mark.slow` test or a notebook-comment commitment with no enforcement. |
| NFR-3 | Save/load `atol=1e-10` on predictions | `RESTATES INTENT` | AC-4; matches Stage 9 MLflow adapter bar already in the registry layer doc. |
| NFR-4 | Loss-curve PNG saved to registry run directory per run | `PLAN POLISH` | AC-3 says "available as a plot without additional wiring" — it does not mandate automatic PNG artefact persistence to the registry. Saving a PNG per run requires modifying the registry save path or the training loop's post-fit hook, binds the plots module to the registry, and adds at least one integration test assertion. Nothing in the intent mandates this; X1 notebook already demonstrates the curve. |
| NFR-5 | Normalisation persistence in `state_dict` | `RESTATES INTENT` | AC-4 and intent §Points both name this as load-bearing. |
| NFR-6 | `python -m bristol_ml.models.nn.mlp --help` runs | `RESTATES INTENT` | DESIGN §2.1.1; every prior model stage shipped this (Stage 7, 8 precedent). |
| X1 | Notebook `notebooks/10-simple-nn.ipynb` with live loss curve + week-ahead forecast | `RESTATES INTENT` | Intent §Scope explicitly names "A notebook that trains the model, plots train-vs-validation loss curves live, and compares predictions against prior models." Intent §Demo moment makes the live loss curve the central demo artefact. |
| X2 | ADR: "Why torch and not flax/sklearn MLP" | `PLAN POLISH` | DESIGN §8 already records PyTorch as the framework choice. A new ADR adds a file and a maintenance obligation but covers a decision that was made before Stage 10; no AC or §Points bullet calls for it. |
| X3 | Update `docs/architecture/README.md` module catalogue for `models/nn/` | `HOUSEKEEPING` | Standard per-stage architecture doc update; same as every prior model stage. |
| X4 | Live-plot loss curve via IPython display-handle during `fit` | `RESTATES INTENT` | Intent §Demo moment: "Train and validation loss plotted over epochs while the model trains in the notebook." This is the stated demo moment — deferred to post-fit static plot would contradict the intent directly. |
| X5 | `device=auto` config field falling back to CPU | `PREMATURE OPTIMISATION` | Intent §Out of scope: "Distributed or multi-GPU training." Intent §Points notes GPU abstraction "is cheap if done early" but D11 (CPU-only) is accepted. No AC requires GPU fallback. Adds a config field, schema test, and conditional branch to the training loop. |
| X6 | Gradient clipping / LR scheduling as configurable knobs | `PREMATURE OPTIMISATION` | Neither named in any AC nor in the intent §Points. Requirements analyst explicitly lists both as "additionally inferred out of scope." Adds config fields, schema tests, and training-loop branches. |
| X7 | `BaseTorchModel` abstract base ready for Stage 11 extraction | `PREMATURE OPTIMISATION` | Intent §Points acknowledges the shared-harness question but concludes "separate loops per model are clearer." D10 already flags the extraction seam for Stage 11. Shipping the base class now is Stage 11 work disguised as Stage 10 YAGNI — it binds Stage 11's design before Stage 11's requirements are understood. |
| X8 | Model card / metadata richer than registry already captures | `PLAN POLISH` | No AC names additional metadata beyond what `ModelMetadata` and `run.json` already carry. Adds fields with no named consumer and no AC justification. |

## Single highest-leverage cut

If you cut one item to halve this plan's scope, cut **NFR-4**
(auto-save loss-curve PNG to registry run directory) because it
couples the plots module to the registry save path, forces a new
integration test asserting file presence in the run directory, and
the intent's AC-3 is fully satisfied by the `loss_history_` attribute
plus the `plots.loss_curve()` helper already required by D6 / T7.
