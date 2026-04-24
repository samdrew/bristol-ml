# Stage 11 — Complex NN — Scope Diff

*Pre-synthesis scope critique by `@minimalist`. Read-only output; the lead reconsiders every `PLAN POLISH` and `PREMATURE OPTIMISATION` row before binding it into `docs/plans/active/11-complex-nn.md`.*

## Inputs reviewed

- `docs/intent/11-complex-nn.md` (authoritative scope)
- `docs/lld/research/11-complex-nn-requirements.md`
- `docs/lld/research/11-complex-nn-codebase.md`
- `docs/lld/research/11-complex-nn-domain.md`
- `docs/plans/completed/10-simple-nn.md` (precedent)
- `docs/lld/research/10-simple-nn-scope-diff.md` (prior scope diff)

## Scope Diff

| ID | Item | Tag | Rationale |
|----|------|-----|-----------|
| D1 | TCN (6 blocks × 64ch × kernel=3 × dropout=0.1), weight norm, LayerNorm, causal padding; Transformer deferred | `RESTATES INTENT` | Intent §Scope "a temporal neural model (TCN… or similar)" and §Points names CPU budget as the forcing function; choosing one architecture is mandatory to ship the stage. |
| D2 | seq_len=168 h, exposed as config knob | `RESTATES INTENT` | Intent §Points names the weekly cycle as the natural upper bound and asks "how many hours of history"; specifying and exposing the choice is required. |
| D3 | Exogenous features: Pattern A (in-sequence concatenation) | `RESTATES INTENT` | Intent §Points asks "whether to feed weather as part of the sequence or as side channels"; the plan must resolve this to make the data pipeline concrete. Cannot ship without a decision. |
| D4 | Shared training loop extraction into `_training.py`; both `NnMlpModel` and `NnTemporalModel` call it | `PLAN POLISH` | Intent AC-2 says "without re-implementing those mechanisms from scratch" — satisfied by a shared file, but also satisfiable by importing helpers from `mlp.py` directly. The Stage 10 CLAUDE.md extraction seam is conditional ("when Stage 11's loop diverges by more than one optimiser kwarg"). A full module extraction forces Stage 10's existing tests to be re-verified against the new call path, adding 1–2 regression test touches with no new behavioural coverage for Stage 11. Author must justify or confirm the Stage 10 test suite passes unchanged at T1. |
| D5 | Single-joblib artefact envelope, same structure as Stage 10 | `RESTATES INTENT` | Intent AC-4 requires registry round-trip of weights and sequence-preprocessing state; the envelope shape is the only pattern the Stage 9 registry accepts (codebase §3 single-file contract). |
| D6 | Four-stream seeding + cuDNN deterministic flags; `torch.use_deterministic_algorithms(True)` stays OFF | `RESTATES INTENT` | Intent AC-4 requires `torch.allclose(atol=1e-5)` round-trip bar; Stage 10 D7' is the stated inheritance contract. |
| D7 | Private `_SequenceDataset` inside `temporal.py`, lazy-window | `RESTATES INTENT` | Intent §Scope: "data pipeline changes needed to feed sequences… encapsulated in the model's fit/predict"; codebase §4 calculates eager cost at ~1.4 GB, making lazy the only viable choice. Keeping it private is the simplest encapsulation. |
| D8 | Val-split offset by seq_len to prevent sequence-overlap leakage | `PREMATURE OPTIMISATION` | No intent AC names train/val boundary leakage as a correctness requirement for this stage. The test set is managed by the harness and is fully separated; the internal val tail overlap affects early-stopping fidelity, not the reported holdout metrics. Domain §6 point 6 flags this as a risk but neither the intent nor any AC names it as a contract requirement. Adds a fixed 168-row offset to `fit()`'s internal split logic and a corresponding test assertion, for a correctness benefit on a code path (internal val) that the intent does not test against. Cut unless the lead judges early-stopping fidelity load-bearing for AC-4. |
| D9 | Harness `evaluate()` reused unchanged; Stage 11 model conforms to protocol | `RESTATES INTENT` | Intent AC-1 (Stage 4 interface) and AC-2 (harness from Stage 10) both require this. |
| D10 | Ablation table: six rows × (MAE, MAPE, RMSE, WAPE, MAE-ratio-vs-NESO, training-time-s, param-count); no bar chart; no scatter | `RESTATES INTENT` | Intent AC-3 names the ablation table; §Demo moment and domain §9 both name training-time as the pedagogical payoff column. Choosing table-only is the minimum that satisfies the demo moment; bar chart and scatter are optional (§Points). |
| D11 | New `evaluation/ablation.py::compute_metrics_on_holdout(run_ids, holdout_range, target)` helper | `PLAN POLISH` | Intent AC-5 is load-bearing and requires a predict-only path. However, the AC does not specify a named public module — a dozen notebook cells calling `registry.load(run_id)` then `model.predict(X_holdout)` then metric functions would satisfy AC-5 equally. Extracting to `evaluation/ablation.py` creates a new public module, forces at least one unit test of the helper itself, extends the `evaluation` package's public surface, and binds the notebook to that API. A notebook-inline implementation satisfies AC-5 with zero new module surface. Author must name the justification for a standalone module over inline notebook code. |
| D12 | Single holdout for ablation table (not rolling-origin) | `RESTATES INTENT` | Intent §Points asks "rolling-origin folds or single holdout"; choosing single holdout is required to bind the plan. This is a mandatory resolution of an open question, not added scope. |
| D13 | Dispatcher extension: `_dispatch.py`, `train.py`, `harness._build_model_from_config` for `nn_temporal` | `RESTATES INTENT` | Intent AC-1 / AC-2 require end-to-end wiring; codebase §3 and §6 document all three sites as mandatory per sixth-family precedent. |
| D14 | Stage 10 harness-factory catch-up: add `NnMlpConfig` branch to `harness._build_model_from_config` | `HOUSEKEEPING` | This is a pre-existing Stage 10 gap, not Stage 11 scope. Defensible in this PR because T7 touches the same function, but it belongs honestly in a separate commit or housekeeping PR. |
| D15 | DLinear NOT included | `RESTATES INTENT` | Intent §Scope says "a temporal neural model" (singular). Domain §1 / §9 argue for DLinear as a pedagogical row but intent is explicit: one model family per stage. Correctly excluded. |
| D16 | Attention visualisation: N/A (consequence of D1) | `RESTATES INTENT` | Intent §Points: "Attention weight visualisation (for Transformers)… not worth building an interpretability stage around." TCN choice makes this moot. |
| NFR-1 | CPU bit-identity; CUDA/MPS `atol=1e-5` close-match | `RESTATES INTENT` | Intent AC-4 names `torch.allclose(atol=1e-5)`; inherited from Stage 10 D7' contract. |
| NFR-2 | Training-time: record wall-clock in retro; no hard gate | `RESTATES INTENT` | Intent §Points explicitly flags demo-time risk; AC-7 (requirements-inferred) names the retro recording. No hard gate is consistent with Stage 10 precedent (Scope Diff NFR-2 cut). |
| NFR-3 | Provenance sidecar: git SHA, fit_utc, feature_set, target, feature_columns, seed_used | `RESTATES INTENT` | DESIGN §2.1.6 and Stage 10 D5 envelope shape; AC-4 round-trip requires these fields to survive. |
| NFR-4 | Standalone runnability: `python -m bristol_ml.models.nn.temporal --help` | `RESTATES INTENT` | DESIGN §2.1.1; every prior model stage shipped this. |
| NFR-5 | Config outside code: `conf/model/nn_temporal.yaml` + `NnTemporalConfig` Pydantic schema | `RESTATES INTENT` | DESIGN §2.1.4; intent AC-1 requires protocol conformance which requires a config schema. |
| NFR-6 | Notebooks thin: imports from `src/`; no reimplemented logic | `RESTATES INTENT` | DESIGN §2.1.8. |
| NFR-7 / test: protocol conformance | `RESTATES INTENT` | Directly tests AC-1 (Stage 4 interface). |
| NFR-7 / test: seeded state_dict identity | `RESTATES INTENT` | Directly tests AC-4 (`torch.allclose(atol=1e-5)` on seeded replay); new Stage 11 surface. |
| NFR-7 / test: loss_history populated | `PLAN POLISH` | Stage 10 already tests `loss_history_` population (Stage 10 T4); this test duplicates Stage 10 contract behaviour on a different model class. The marginal coverage is that Stage 11's `fit()` wires up `loss_history_` — one assertion, low blast radius, but belt-and-braces on already-tested Stage 10 behaviour. Forces one new test file touch with no new behavioural AC coverage. |
| NFR-7 / test: registry round-trip `atol=1e-5` | `RESTATES INTENT` | Directly tests AC-4. |
| NFR-7 / test: sequence-preprocessing round-trip | `RESTATES INTENT` | Directly tests the new `_SequenceDataset` contract (AC-4 requires the sequence preprocessing state to survive save/load). New Stage 11 surface with no Stage 10 equivalent. |
| T1 | `_training.py` extraction + Stage 10 refactor | `PLAN POLISH` | Consequence of D4 (see D4 row); Stage 10 regression surface is the downstream cost. |
| T2 | `_SequenceDataset` + lazy-window unit tests | `RESTATES INTENT` | AC-4 and the codebase §4 correctness requirement on lazy windowing; new Stage 11 surface. |
| T3 | `NnTemporalConfig` + `conf/model/nn_temporal.yaml` | `RESTATES INTENT` | DESIGN §2.1.4 / NFR-5; required to wire the model into Hydra. |
| T4 | `NnTemporalModel` scaffold + protocol conformance | `RESTATES INTENT` | AC-1. |
| T5 | `fit` + `predict` + four-stream seeding | `RESTATES INTENT` | AC-1, AC-4, NFR-1. |
| T6 | `save` + `load` single-joblib envelope | `RESTATES INTENT` | AC-4. |
| T7 | Dispatcher extension for `nn_temporal` + `NnMlpConfig` catch-up | `RESTATES INTENT` (D13) + `HOUSEKEEPING` (D14) | D13 is load-bearing for AC-1/AC-2; D14 is the Stage 10 gap (see D14 row). |
| T8 | `evaluation/ablation.py` helper | `PLAN POLISH` | Consequence of D11 (see D11 row); forces a new public module and at least one unit test with no AC gain over notebook-inline code. |
| T9 | `notebooks/11-complex-nn.ipynb` + `scripts/_build_notebook_11.py` | `RESTATES INTENT` | Intent §Scope: "A notebook that trains the model, compares it against every prior model… and produces an ablation table." |
| T10 | Stage hygiene: retro, CHANGELOG, layer-doc update | `HOUSEKEEPING` | Standard per-stage hygiene per CLAUDE.md §Stage hygiene. |
| Cell 1 | Preamble (config load, seed, device) | `RESTATES INTENT` | Required setup for any notebook that trains a model. |
| Cell 2 | Load feature table + holdout split | `RESTATES INTENT` | AC-3 / AC-5 require the same held-out period for all models. |
| Cell 3 | Fit `NnTemporalModel` with live loss-curve | `RESTATES INTENT` | Intent §Demo moment: the live training arc is the pedagogical payoff. AC-2 requires the harness / epoch-callback seam to be exercised. |
| Cell 4 | `registry.save(model, ...)` with sidecar metadata | `RESTATES INTENT` | AC-4 requires the registry round-trip to be demonstrated in the notebook. |
| Cell 5 | `compute_metrics_on_holdout([...])` over all six registered runs → ablation table | `RESTATES INTENT` | AC-3 / AC-5: ablation table reproducible from registry. (Whether the helper lives in `ablation.py` or inline is the D11 question, not this cell's question.) |
| Cell 6 | Ablation commentary (markdown) | `RESTATES INTENT` | Intent §Demo moment: "a facilitator can look at the table and tell a coherent story about which techniques bought which accuracy." The markdown cell is that story. |
| Cell 7 | Receptive-field diagram (static image, optional) | `PLAN POLISH` | Intent §Points: attention visualisation "not worth building an interpretability stage around" — the same bar governs TCN receptive-field diagrams. A static image adds one notebook cell and one image asset with no AC coverage. Author must name the justification (pedagogical illustration of dilated-conv stacking) or cut. |

## Single highest-leverage cut

**If you cut one item to halve this plan's scope, cut D11 (`evaluation/ablation.py::compute_metrics_on_holdout`) because AC-5 is fully satisfied by a dozen notebook-inline lines (`registry.load` → `model.predict` → metric functions), and extracting it to a new public module forces a standalone unit test, extends the `evaluation` package's public surface, and binds the notebook's ablation cell to an API that no other stage currently consumes.**

---

**Tags flipped from the draft's implicit framing:**

- **D8** — draft treats val-split leakage offset as a correctness necessity. Tagged `PREMATURE OPTIMISATION`: no intent AC names internal val-split correctness; the reported holdout metrics (the AC-3 / AC-5 contract) are unaffected. The lead should decide whether early-stopping fidelity is worth the implementation complexity.
- **D11** — draft treats the `evaluation/ablation.py` module as the obvious way to satisfy AC-5. Tagged `PLAN POLISH`: AC-5 is load-bearing but does not require a named public module; notebook-inline code satisfies it at lower blast radius.
- **NFR-7 / loss_history test** — draft lists it as a boundary test; tagged `PLAN POLISH` because it duplicates Stage 10 T4 contract behaviour on the new class without covering new Stage 11 contract territory.
