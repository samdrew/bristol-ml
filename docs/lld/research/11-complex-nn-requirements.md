# Stage 11 — Complex Neural Network — Structured Requirements

**Source intent:** `docs/intent/11-complex-nn.md`
**Artefact role:** Phase 1 research deliverable (requirements analyst).
**Audience:** plan author (lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## 1. Goal

Introduce a temporal neural model (TCN or small Transformer) that exploits sequence structure in hourly demand data, produces an ablation table comparing every model built so far on the same held-out period, and closes the modelling arc as the project's most capable forecaster to date.

---

## 2. User stories

**US-1 (pedagogy / facilitator).** Given a meetup attendee who has seen all prior model stages, when the facilitator opens the Stage 11 notebook, then they can point at the ablation table and narrate a coherent story — "each row bought us this much accuracy" — culminating in whichever technique performed best, so that the full modelling arc is legible in one artefact.

**US-2 (pedagogy / attendee).** Given that I attended Stage 10 and understand the MLP training harness, when I read the Stage 11 model class, then I can see how a temporal architecture replaces the flat-feature approach without the surrounding machinery changing, so that I learn how the Stage 4 interface makes model families substitutable.

**US-3 (analytical / beat NESO).** Given the ablation table, when I compare the temporal model's MAE ratio against the NESO day-ahead benchmark column, then I can read off whether temporal modelling bought measurable improvement over the linear/classical/MLP baselines, so that the project's secondary goal is concretely evaluated.

**US-4 (analytical / reproducibility).** Given a future stage that wants to claim "model X beats the best of these", when it loads all prior registered runs and re-runs only its new model, then the ablation table can be reconstructed without re-training anything already in the registry, so that prior results are stable reference points.

**US-5 (analytical / stage 4 interface).** Given that downstream stages (serving, orchestration) depend on the Stage 4 `Model` protocol, when `NnTemporalModel` is instantiated from config and called via `fit` / `predict`, then it is indistinguishable from any other `Model` at the call site, so that no downstream code needs a temporal-model special case.

---

## 3. Acceptance criteria (AC-1..AC-8)

**AC-1.** The temporal model class conforms to the Stage 4 `Model` protocol (`fit`, `predict`, `save`, `load`, `metadata`); `isinstance(model, bristol_ml.models.Model)` is `True`.

**AC-2.** Training uses the harness established in Stage 10 — specifically, the same loss-logging (`loss_history_`), early-stopping, and epoch-callback seam — without re-implementing those mechanisms from scratch. *(Triggers the D10 extraction seam named in Stage 10 plan §5.)*

**AC-3.** The notebook's ablation table covers every model family trained so far (naive, linear, SARIMAX, SciPy-parametric, MLP, and the new temporal model) on the same held-out period, with columns for at least MAE, MAPE, RMSE, WAPE, and MAE ratio vs NESO benchmark.

**AC-4.** `registry.save(model, ...)` / `registry.load(run_id)` round-trips full model weights and the sequence-preprocessing state; `predict` output on the same input matches within `torch.allclose(atol=1e-5)` (matching Stage 10 D7' GPU close-match bar).

**AC-5.** The ablation table in the notebook is reproducible from registered artefacts alone — no model is re-trained to produce the table if it is already in the registry.

**AC-6** *(inferred — ablation depends on it).* All prior model families have valid runs in the registry before the ablation cell executes; the notebook either asserts their presence or provides a cell that registers them from reproducible training calls on the same splits.

**AC-7** *(inferred — demo moment depends on it).* Training the temporal model on the project's data completes in a time consistent with live-demo use on a laptop (see NFR-2); the observed wall-clock is recorded in the stage retro as an informational data point.

**AC-8** *(inferred — Stage 10 D10 extraction seam).* The shared training-harness code is extracted into a module (`_training.py` or equivalent) that both `NnMlpModel` and `NnTemporalModel` import, so the Stage 10 plan's named extraction trigger is honoured.

---

## 4. Non-functional requirements (NFR-1..NFR-7)

**NFR-1 (Reproducibility — §2.1.5 / §2.1.6 + Stage 10 D7').** Seeded training with the four-stream recipe (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`) produces identical `state_dict`s on CPU (`torch.equal`) and close-match outputs on CUDA/MPS (`torch.allclose(atol=1e-5)`). The `seed` parameter propagates from config through the training call. Exact GPU bit-identity is not promised (§2.1.5 carve-out).

**NFR-2 (Training-time budget — pedagogy first).** The intent explicitly flags: "if training takes too long to demo, the stage loses pedagogical value." Target: training must complete in a time suitable for a live demo on a laptop CPU. The exact budget is an open question (OQ-D); the observed wall-clock must be noted in the retro regardless of whether a hard gate is set.

**NFR-3 (Provenance — §2.1.6).** Every registry entry produced by this stage records git SHA, `fit_utc`, `feature_set`, `target`, `feature_columns`, and `seed_used` in the run sidecar, consistent with Stage 10 precedent.

**NFR-4 (Standalone runnability — §2.1.1).** `python -m bristol_ml.models.nn.temporal --help` (or equivalent module path) must not raise; the model module must be importable and its CLI entrypoint must print usage.

**NFR-5 (Config outside code — §2.1.4).** Architecture hyperparameters (sequence length, number of layers/heads, hidden dimension, dropout) and training hyperparameters (learning rate, batch size, max epochs, patience) live in `conf/model/nn_temporal.yaml` and a corresponding Pydantic schema. No numeric defaults are hard-coded in Python.

**NFR-6 (Notebooks are thin — §2.1.8).** The Stage 11 notebook imports from `src/bristol_ml/`; it does not reimplement training logic, metric computation, or registry access.

**NFR-7 (Tests at boundaries — §2.1.7).** At minimum: protocol conformance, seeded-run state-dict identity, loss-history population, registry round-trip (predict `atol=1e-5`), and sequence-preprocessing round-trip. Coverage is not a goal; behavioural clarity is.

---

## 5. Open questions (OQ-A..OQ-H)

**OQ-A — Architecture choice.** `needs-human`. TCN vs small Transformer vs other sequence model (S4, Mamba, etc.). Intent says both are "reasonable." TCNs train faster and are simpler to reason about; Transformers are more fashionable and the positional-encoding sub-question (OQ-C) only arises for Transformers. *Best guess: TCN for pedagogical clarity unless the facilitator has a specific reason to showcase attention.*

**OQ-B — Sequence length.** `needs-domain`. How many hours of history to condition on. Intent flags 168 h (weekly cycle) as a natural upper bound; shorter sequences miss weekly patterns but train faster. *Best guess: 168 h as the default, with the length exposed as a config parameter so it can be reduced for demo speed.*

**OQ-C — Positional encoding (Transformer only).** `needs-domain`. Learned vs sinusoidal vs omitted. Only relevant if OQ-A resolves to Transformer. *Best guess: sinusoidal for short fixed-length sequences (no learned parameters to overfit); omitted is also defensible if sequences are short.*

**OQ-D — Hard wall-clock gate.** `needs-human`. Stage 10 observed ~3 min on CPU for MLP; a temporal model with 168-h sequences may be slower. Should the stage impose a hard failing test gate (e.g. `pytest.mark.slow` with a ceiling), or record the observed time as an informational note only (Stage 10 precedent)? *Best guess: informational note only, consistent with Stage 10 Scope Diff row NFR-2 (PREMATURE OPTIMISATION cut).*

**OQ-E — Weather as sequence input vs side channel.** `needs-domain`. Intent flags both options. Side-channel matches day-ahead reality (weather forecasts are known at prediction time); as-sequence is simpler. *Best guess: side-channel, because it matches the feature-table structure already established and avoids a special data-pipeline branch.*

**OQ-F — Ablation presentation format.** `needs-human`. Intent lists three options: single table (cleanest), bar chart (eye-catching), per-model predicted-vs-actual scatter (illuminating but visually busy). These are not mutually exclusive but the demo moment is described as "the ablation table." *Best guess: table primary, optional bar chart in a follow-on cell; scatter omitted as too busy for the demo moment.*

**OQ-G — Rolling-origin folds vs single holdout for the temporal model.** `needs-human`. Intent flags: "multiple folds is honest but expensive." Stage 10 ran the MLP through rolling-origin folds; consistency argues for the same. *Best guess: single holdout for the temporal model in the notebook ablation (for demo speed), with a note that the rolling-origin harness is available and was used for prior models.*

**OQ-H — Attention weight visualisation (Transformer only).** `needs-human`. Intent says "worth a quick look; not worth building an interpretability stage around." Only relevant if OQ-A resolves to Transformer. *Best guess: one optional notebook cell, not a tested artefact.*

---

## 6. In-scope / out-of-scope checklist

### In scope (mirrored 1:1 from intent §Scope)

- [ ] A temporal neural model (TCN, small Transformer, or similar) conforming to the Stage 4 interface.
- [ ] Data pipeline changes to feed sequences rather than flat feature rows, encapsulated in `fit` / `predict` or in a helper reused from Stage 10.
- [ ] A notebook that trains the model, compares it against every prior model on the same held-out period, and produces an ablation table as a reference artefact.

### Out of scope (mirrored 1:1 from intent §Out of scope and §Out of scope, explicitly deferred)

- [ ] Foundation models for time series (TimesFM, Chronos, Lag-Llama). Separate stage if ever pursued.
- [ ] Probabilistic variants of the architecture.
- [ ] Training-time hyperparameter search.
- [ ] Multi-horizon training (single day-ahead horizon only).
- [ ] Time-series foundation models (explicit deferred repeat).
- [ ] Knowledge distillation from large to small.
