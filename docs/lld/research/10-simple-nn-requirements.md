# Stage 10 — Simple Neural Network — structured requirements

**Source intent:** [`docs/intent/10-simple-nn.md`](../../intent/10-simple-nn.md)
**Artefact role:** Phase 1 research deliverable (requirements-analyst).
**Audience:** plan author (this lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## Goal

Introduce a small, fully-connected MLP that conforms to the Stage 4
`Model` protocol, establishes the PyTorch training-loop, loss-logging,
and early-stopping conventions that Stage 11 will inherit, and produces
a live loss-curve demo moment suitable for meetup facilitation.

---

## User stories

**As a meetup facilitator**, given that I open the Stage 10 notebook on
a laptop with no GPU, when I run all cells, then I see train and
validation loss plotted live over epochs and can narrate the
"watch it learn" moment without any extra setup.

**As a notebook attendee**, given that training completes, when I
inspect the leaderboard with `python -m bristol_ml.registry list`, then
the MLP run appears alongside all prior model entries with comparable
metric columns.

**As a Stage 11 implementer**, given that Stage 10 is shipped, when I
look at `models/nn_mlp.py` and whatever shared harness it uses, then I
have a clear training-loop scaffold (loss logging, early-stopping,
device handling) I can extend for temporal architectures without
re-inventing it.

**As any user**, given a fitted MLP registered to the Stage 9
filesystem registry, when I call `registry.load(run_id)` and invoke
`.predict(features)`, then I receive a Series that matches the pre-save
predictions within numerical round-trip tolerance.

**As a data scientist**, given the same random seed and the same
training data, when I re-train the MLP, then the reported metrics are
reproducible to a degree consistent with DESIGN §2.1.5 (exact
bit-for-bit GPU reproducibility is not promised).

---

## Acceptance criteria (AC-N)

**AC-1.** The MLP conforms to the Stage 4 interface, with the training
loop hidden behind `fit`.

Concrete measurable condition: `isinstance(model, Model)` returns
`True` (runtime check per `src/bristol_ml/models/protocol.py`);
`model.fit(features, target)` returns `None`; `model.predict(features)`
returns a `pd.Series` indexed to `features.index`; `model.save(path)` /
`model.load(path)` round-trip; `model.metadata` returns a
`ModelMetadata` with non-`None` `fit_utc` after fit. A test may prove
conformance against the same fixture used for `NaiveModel` and
`LinearModel`.

**Question (AC-1 ambiguity).** The intent says "training loop hidden
behind `fit`", but does not specify whether `fit` may accept additional
keyword-only arguments (e.g. `validation_fraction`, `verbose`) beyond
the `(features, target)` protocol signature. Protocol conformance
requires exactly `(self, features, target)`. Does Stage 10 extend the
signature with keyword-only arguments, or does all loop configuration
come from the constructor config?

**AC-2.** Training is reproducible given a seed (within the
constraints of non-deterministic GPU operations).

Concrete measurable condition: two sequential calls to `fit` with the
same data and the same seed produce per-metric values that are
identical (CPU path) or differ by less than a declared tolerance (GPU
path if exercised). The seed is supplied via the `NnMlpConfig` Pydantic
schema (or constructor argument). A test must call `fit` twice with a
fixed seed and assert that `predict` output is identical on the CPU
path; it must not exercise GPU.

**Question (AC-2 ambiguity).** The intent says "within the constraints
of non-deterministic GPU operations." Does Stage 10 commit to the
stronger `torch.use_deterministic_algorithms(True)` on CPU (which is
free of non-determinism on CPU) while leaving GPU determinism
unspecified, or is `torch.manual_seed` alone the reproducibility bar?
See OQ-7 below.

**AC-3.** The loss curve is produced by the training loop itself and
is available as a plot without additional wiring.

Concrete measurable condition: after `model.fit(features, target)`,
calling a single method or accessing a single attribute on the model
object returns a `matplotlib.figure.Figure` (or the raw loss history as
a `dict[str, list[float]]` from which the plot can be rendered) without
the caller having to touch `evaluation/plots.py` or write any `plt.*`
code. A test asserts that post-fit loss history is non-empty and has at
least one epoch's worth of entries for both training and validation
loss.

**Question (AC-3 ambiguity).** The phrase "available as a plot without
additional wiring" is ambiguous about whether the model returns a
pre-rendered `Figure` or a data structure that a one-liner can plot.
An existing Stage 6 helper (`forecast_overlay`) takes `pd.Series`
inputs; a loss-curve helper does not yet exist in
`src/bristol_ml/evaluation/plots.py`. Should `plots.py` gain a
`loss_curve(history)` helper (model-agnostic, takes
`dict[str, list[float]]`), or should the model carry its own
`plot_loss()` method? See OQ-1 below.

**AC-4.** Save/load through the registry round-trips cleanly,
including the fitted weights.

Concrete measurable condition: `registry.save(model, metrics_df,
feature_set=..., target=...)` succeeds and returns a `run_id`;
`registry.load(run_id)` returns a `Model` instance; predictions from
the loaded instance match predictions from the pre-save instance with
`numpy.allclose(..., atol=1e-6)` on the CPU path.

Note: the Stage 9 MLflow graduation test uses `atol=1e-10`
(`docs/architecture/layers/registry.md`, MLflow adapter contract).
PyTorch `state_dict` round-tripped through joblib should reproduce
floats exactly on CPU; `atol=1e-10` is achievable and is the preferred
bar, but a concrete ceiling must be stated in the plan. If `atol=1e-10`
is unachievable for any discovered reason, that is a plan-phase
question, not a silent relaxation.

**Question (AC-4 scope).** The `registry._dispatch` table in
`src/bristol_ml/registry/_dispatch.py` must gain an `"nn_mlp"` →
`NnMlpModel` entry. Is this purely a Stage 10 task, or does it require
a plan-level note on the two-site dispatcher duplication risk flagged
in `docs/architecture/layers/models.md` (Open questions —
"Dispatcher duplication")?

**AC-5.** Training on the project's data completes in a reasonable
time on a laptop CPU (no GPU requirement).

Concrete measurable condition: a single `fit` call on the project's
feature table (the Stage 5 assembler output — approximately three to
four years of hourly data, on the order of 30 000 rows) completes in
under a plan-author-declared ceiling on a commodity laptop CPU. See
NFR-2 for the tentative proposed ceiling. The test must not assert
wall-clock time (too flaky); the ceiling is a notebook-comment
commitment and a smoke-test guard at best.

**Question (AC-5 ambiguity).** "A reasonable time" is undefined. The
intent's "Points for consideration" lists this as a concern for
rolling-origin evaluation (re-training per fold is expensive for NNs).
Is AC-5 measuring a single `fit` call or the full rolling-origin
harness across all folds? The two differ by one to two orders of
magnitude. See OQ-5 below.

---

## Non-functional requirements (NFR-N)

**NFR-1 — Reproducibility given seed** (`DESIGN §2.1.5`;
`docs/intent/10-simple-nn.md` Point 4)
`torch.manual_seed(seed)` must be called inside `fit` before any weight
initialisation or data-loading shuffle, using the seed value from the
model's config. On CPU, this guarantees bit-for-bit identical outputs
across runs. The seed value must appear in
`ModelMetadata.hyperparameters` so the registry sidecar captures it.
Trace: `DESIGN §2.1.5` ("Re-running training produces comparable
artefacts; exact reproducibility not promised for GPU-trained models");
`docs/intent/10-simple-nn.md` Point 4.

**NFR-2 — CPU training time budget (tentative)** (`docs/intent/10-simple-nn.md`
Point 6; AC-5)
A single `fit` call on the full project feature table (one train split,
not rolling-origin) must complete in under **3 minutes** on a
commodity laptop CPU (indicative: 2023-era Intel Core i7 or M-series).
This ceiling is **tentative** — the plan author must validate against
actual data and architecture defaults before committing. The ceiling
applies to a single fold, not the full rolling-origin harness; see
OQ-5 for fold-multiplier implications.
Trace: `docs/intent/10-simple-nn.md` §Scope ("Any model larger than a
small MLP can fit on a laptop"), §Points ("Initial architecture choice
… one or two hidden layers, moderate width").

**NFR-3 — Save/load weight-fidelity** (`docs/architecture/layers/registry.md`;
AC-4)
Prediction outputs from a loaded `NnMlpModel` must satisfy
`numpy.allclose(pred_before_save, pred_after_load, atol=1e-10)` on the
CPU path (matching the Stage 9 MLflow adapter bar at
`docs/architecture/layers/registry.md`, MLflow adapter section). If the
joblib serialisation of `torch.nn.Module.state_dict()` cannot achieve
this tolerance, the plan must document the discovered tolerance and
justify any relaxation before shipping.
Trace: `src/bristol_ml/registry/CLAUDE.md` (MLflow adapter
`atol=1e-10`); `docs/architecture/layers/registry.md` (serialisation
notes).

**NFR-4 — Loss curve accessible without extra wiring** (AC-3)
After `fit`, the model must expose loss history in a form that a
single notebook cell can render without importing any module outside
`bristol_ml`. Whether this is a `plot_loss()` method or a
`loss_history_` attribute consumed by a new
`evaluation.plots.loss_curve()` helper is an OQ-1 decision, but
whichever form is chosen must not require the caller to reconstruct
the history from log output.
Trace: `docs/intent/10-simple-nn.md` AC-3 and §Demo moment;
`src/bristol_ml/evaluation/CLAUDE.md` (plots surface,
model-agnosticism rule — "helpers take `pd.Series` / `pd.DataFrame`
inputs — never a `Model` object").

**NFR-5 — Normalisation-statistic persistence** (`docs/intent/10-simple-nn.md`
Point 1)
Any input normalisation statistics (mean, standard deviation) computed
at `fit` time must be serialised alongside the model weights so that
`load` + `predict` produces correctly scaled outputs without access to
the training data. These statistics must appear in
`ModelMetadata.hyperparameters` or be captured inside the serialised
object.
Trace: `docs/intent/10-simple-nn.md` §Points ("How normalisation is
done, and where the statistics live, matters for save/load and for
serving later"); `docs/architecture/layers/registry.md` (sidecar
`hyperparameters` field).

**NFR-6 — Protocol hygiene and standalone CLI** (`DESIGN §2.1.1`;
`src/bristol_ml/models/CLAUDE.md`)
`python -m bristol_ml.models.nn_mlp` must run without error and print
at minimum the `NnMlpConfig` defaults and a pointer to the
training-loop conventions (consistent with the pattern established for
`sarimax.py` and `scipy_parametric.py`).
Trace: `DESIGN §2.1.1`; `src/bristol_ml/models/CLAUDE.md`
("Running standalone").

---

## Explicit open questions (OQ-N)

**OQ-1 — Loss-curve surfacing mechanism**
Should `NnMlpModel` carry a `plot_loss()` method that returns a
`Figure`, expose a `loss_history_: dict[str, list[float]]` attribute
for callers to plot themselves, or trigger the addition of a
`loss_curve(history)` helper to `src/bristol_ml/evaluation/plots.py`?
Tension: `evaluation/CLAUDE.md` mandates that plots helpers take
`pd.Series`/`pd.DataFrame` inputs and are never model-aware; a
`loss_history_` attribute satisfies this constraint if a new
`plots.loss_curve(history)` helper consumes it. A model-attached
`plot_loss()` is simpler but mixes presentation into the model. AC-3
says "without additional wiring" — the plan must resolve what
"wiring" means.
Decision owner: plan author. Latest-responsible moment: pre-plan
synthesis (the answer shapes the models/evaluation module boundary).

**OQ-2 — Normalisation-statistic placement**
Should input normalisation live inside `NnMlpModel` (computed in `fit`,
stored in `self`, serialised with the model), or should it be a
feature-layer preprocessor that is run before `fit` is called?
Implications: an inside-model normaliser simplifies the
`fit`/`predict`/`save`/`load` contract and is more natural for serving
(Stage 12); an external preprocessor is more composable but requires
the caller to persist statistics separately. The intent (Point 1)
flags this as load-bearing for save/load and serving.
Decision owner: plan author with human approval on the serving
implication. Latest-responsible moment: pre-plan synthesis.

**OQ-3 — Training-loop ownership**
Should the training loop live entirely inside `NnMlpModel.fit`, or
should a shared module (e.g. `models/_torch_common.py`) extract the
epoch loop, loss accumulation, early-stopping, and validation logic
so that Stage 11's `NnTemporalModel` can inherit it?
The intent explicitly flags this as a scope question. A shared harness
is more DRY but adds a module boundary and potential over-engineering
for Stage 10 alone.
Decision owner: plan author. Latest-responsible moment: pre-plan
synthesis (determines whether Stage 10 creates a shared module or
defers extraction to Stage 11).

**OQ-4 — CUDA path**
Should Stage 10 declare CPU-only and defer `device="auto"` to
Stage 11, or carry the device-abstraction from day one?
The intent (Point 7) notes that the abstraction is cheap if done
early and meaningful for GPU-equipped contributors. However, the demo
moment is on a laptop; Stage 9 plan D14 and the models-layer doc both
flag GPU as a deferred concern.
Decision owner: plan author. Latest-responsible moment: pre-plan
synthesis.

**OQ-5 — Rolling-origin efficiency**
Should Stage 10 re-train the MLP on every rolling-origin fold (status
quo for all prior models) or explore warm-starts / partial-fit across
folds?
The intent (Point 8) explicitly flags this as expensive. Re-training
per fold is honest and consistent with the harness contract;
warm-starts reduce evaluation time but complicate the `Model` protocol
and break the independence assumption the rolling-origin splitter
guarantees.
Decision owner: plan author. Latest-responsible moment: Phase 2
(before implementation begins); a tentative answer of "re-train per
fold, document the cost" is reasonable for Stage 10 and can be
revisited at Stage 11.

**OQ-6 — Early-stopping criterion and checkpoint semantics**
What constitutes "early stopping": validation-loss patience alone,
best-epoch restore, or both? Should the registry receive the model at
the best validation epoch or only at the end of training?
The intent's `Scope` says "early-stopping and checkpointing tied to
the registry from Stage 9." The registry's save protocol
(`registry.save`) expects a fully-fitted model, not a mid-epoch
checkpoint. The best-epoch restore implies in-memory weight management;
saving every-best to disk conflicts with the registry's last-write-wins
semantics.
Decision owner: plan author. Latest-responsible moment: Phase 2.

**OQ-7 — Reproducibility discipline**
Is `torch.manual_seed(seed)` alone sufficient, or should Stage 10
also set `torch.use_deterministic_algorithms(True)` and document the
`CUBLAS_WORKSPACE_CONFIG` requirement for GPU use?
`torch.use_deterministic_algorithms(True)` on CPU adds no overhead and
eliminates the few non-deterministic CPU ops (e.g. certain scatter
operations); on GPU it forces cublas into a mode requiring a workspace
environment variable. The intent (Point 4) says "reasonable
reproducibility given the same random seed" is achievable.
Decision owner: plan author. Latest-responsible moment: pre-plan
synthesis.

---

## Out of scope (explicitly)

Lifted verbatim from `docs/intent/10-simple-nn.md`:

- Temporal architectures (convolutional, recurrent, attention) —
  Stage 11.
- Hyperparameter search / optimisation.
- Distributed or multi-GPU training.
- Any model architecture larger than a small MLP that fits on a
  laptop.
- Ensembling.
- Model quantisation or export for deployment.

Additionally inferred:

- Attention mechanisms or transformer blocks of any kind.
- Learning-rate schedules beyond a single fixed default (e.g. no
  cosine annealing, no warm restarts, no cyclical LR).
- Automatic mixed precision (AMP / `torch.cuda.amp`).
- Gradient clipping (unless forced by a training instability
  discovered in Phase 2).
- `skops.io` adoption — this is explicitly deferred to Stage 12
  serving (`src/bristol_ml/models/CLAUDE.md`,
  `docs/architecture/layers/registry.md`).
- MLflow as a runtime dependency — test-only adapter only, per
  Stage 9 precedent.
- Cross-version load compatibility — not a goal until Stage 9's
  registry plan explicitly addresses it.
- Probabilistic / quantile predictions — DESIGN §3.3 defers all
  probabilistic forecasting.
- Dispatcher duplication refactor — flagged as a housekeeping task in
  `docs/architecture/layers/models.md`; Stage 10 adds the `"nn_mlp"`
  entry to the existing two-site pattern without consolidating.

---

## Dependencies and deferred consumers

**Upstream (Stage 10 consumes):**

| Contract | File | What Stage 10 relies on |
|---|---|---|
| `Model` protocol (five members) | `src/bristol_ml/models/protocol.py` | `NnMlpModel` must satisfy all five members |
| `ModelMetadata` schema | `conf/_schemas.py` (re-exported via `protocol.py`) | `hyperparameters` bag carries seed, normalisation stats, architecture |
| `evaluate` / `evaluate_and_keep_final_model` | `src/bristol_ml/evaluation/harness.py` | Rolling-origin loop; Stage 10 model is fed through without harness changes |
| API-growth rule | `src/bristol_ml/evaluation/CLAUDE.md` (H5) | No second boolean flag on `evaluate()`; loss history must not require a new harness flag |
| Registry four-verb surface | `src/bristol_ml/registry/__init__.py` | `registry.save` / `registry.load`; `_dispatch.py` gains `"nn_mlp"` entry |
| Plots model-agnosticism rule | `src/bristol_ml/evaluation/plots.py` (AC-3 note) | Loss-curve data must be a plain Python structure, not a `Model` object |
| Okabe-Ito palette and rcParams | `src/bristol_ml/evaluation/plots.py` (OKABE_ITO constant) | Notebook loss-curve plot should use the Stage 6 palette for visual consistency |

**Downstream (consumes Stage 10 output):**

| Consumer | Stage | What it inherits |
|---|---|---|
| Complex neural network | Stage 11 | Training-loop scaffold, loss-logging convention, device-handling pattern, shared harness (if created) |
| Serving endpoint | Stage 12 | `registry.load` → `model.predict`; `skops.io` upgrade seam in `models/io.py` (plan D14) |
| Drift monitoring | Stage 18 | Registered MLP runs appear in leaderboard; `list_runs` filters by `model_type="nn_mlp"` |

---

## Cross-references

- `docs/intent/10-simple-nn.md` — primary intent document.
- `docs/intent/DESIGN.md` — §2.1.5 reproducibility, §7.3 `Model`
  protocol, §8 PyTorch choice.
- `src/bristol_ml/models/protocol.py` — five-member `Model` protocol
  and `ModelMetadata`.
- `src/bristol_ml/models/CLAUDE.md` — re-entrancy, serialisation,
  open questions on dispatcher duplication and CLI parity.
- `docs/architecture/layers/models.md` — protocol semantics,
  serialisation upgrade path, open questions.
- `src/bristol_ml/evaluation/CLAUDE.md` — H5 API-growth rule,
  `return_predictions` note, plots model-agnosticism rule.
- `src/bristol_ml/evaluation/plots.py` — existing seven helpers; no
  loss-curve helper present (gap Stage 10 must decide whether to
  fill).
- `src/bristol_ml/registry/CLAUDE.md` — four-verb cap, `_dispatch.py`
  type table, MLflow `atol=1e-10` bar.
- `docs/architecture/layers/registry.md` — sidecar schema,
  serialisation note, skops seam at Stage 12.

---

**Summary.** Five ACs with measurable conditions and three explicit
ambiguity flags; six NFRs; seven open questions sourced from the
intent's "Points for consideration". The loss-curve surfacing
mechanism (OQ-1) and normalisation-statistic placement (OQ-2) are the
highest-priority decisions for plan synthesis because they determine
the models/evaluation module boundary before implementation begins.
