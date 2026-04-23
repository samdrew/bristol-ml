# Stage 9 — Model registry: requirements

**Source.** `/workspace/docs/intent/09-model-registry.md` (immutable spec).  Where this document and the spec disagree, the spec wins.

**Purpose.** Structured translation of the intent into user stories, acceptance criteria, non-functional requirements, and open questions.  Input to `docs/plans/active/09-model-registry.md`; not itself a plan.

**Baseline SHA:** `575ac9c`.

---

## §1 Goal

Introduce a filesystem-backed registry layer that wraps model artefact saving and loading with a consistent metadata record, retrofits every model shipped in Stages 4, 7, and 8 to save through it, and surfaces a CLI leaderboard so that "which model is best" becomes a single command rather than bespoke code.

---

## §2 User stories (Given/When/Then)

**US-1 — Meetup facilitator (demo moment).**

Given models from Stages 4, 7, and 8 have been trained and saved through the registry, when the facilitator runs `python -m bristol_ml registry list` (or equivalent CLI path per intent §Demo moment), then a leaderboard prints every registered model with its held-out metrics, feature set, and training timestamp, such that the facilitator can say "here is every technique we have built this evening, ranked by MAE" without writing any code on the spot.

**US-2 — Model developer (adding a new model).**

Given a new model class that conforms to the Stage 4 `Model` protocol, when the developer calls the registry's `save` path with that model and the required metadata, then the artefact and metadata record appear in the registry without any changes to the model class itself (intent §Scope line 15: "the interface is what the registry consumes").

**US-3 — Analyst (comparing runs after the fact).**

Given several registered entries produced by different training runs, when the analyst calls the registry's `list` path with a filter (by target, feature set, or model type per intent §Points for consideration), then the returned entries include the training-data version, feature-set name, git SHA, wall-clock timestamp, and held-out metrics for each matched run, with no bespoke querying code required.

**US-4 — Future Stage 12 serving author (loading by name).**

Given a registered model entry identified by name, when the Stage 12 serving layer calls the registry's `load` path, then the fitted model artefact is returned in a state ready for `predict()`, without the serving layer needing to know the on-disk layout or metadata format.

**US-5 — Contributor (hand-inspection).**

Given the registry directory on a local filesystem, when a contributor browses it with standard shell tools (no CLI required), then the on-disk layout and metadata schema are documented clearly enough that the artefact files and associated metadata are unambiguous (intent §AC-5).

---

## §3 Acceptance criteria — transcribed

The following five criteria are the complete set, copied verbatim from `/workspace/docs/intent/09-model-registry.md` §Acceptance criteria, then annotated with candidate evidence.  No additional ACs are introduced here.

**AC-1.  The registry's public interface is small — save, load, list, maybe describe.  If it grows past that, the design is drifting.**

Verbatim: "The registry's public interface is small — save, load, list, maybe describe.  If it grows past that, the design is drifting."

Candidate evidence:
- The public surface of `bristol_ml.registry` exports no more than four callables at the module's `__all__`.  A structural test counts the exported names and fails if the count exceeds four.
- Test name pattern: `test_registry_public_surface_does_not_exceed_four_callables`.

**AC-2.  Every model shipped before this stage can save through the registry without code changes to the model itself (the interface is what the registry consumes).**

Verbatim: "Every model shipped before this stage can save through the registry without code changes to the model itself (the interface is what the registry consumes)."

Candidate evidence:
- Integration tests call the registry `save` path for each of `NaiveModel`, `LinearModel`, `SarimaxModel`, and `ScipyParametricModel` using their existing `save`/`metadata` protocol surface only — no model-internal attributes accessed.
- Test name pattern: `test_registry_save_naive_model_via_protocol`, `test_registry_save_linear_model_via_protocol`, `test_registry_save_sarimax_model_via_protocol`, `test_registry_save_scipy_parametric_model_via_protocol`.

**AC-3.  Metadata is captured automatically where it can be (git SHA, timestamps, feature-set name); the rest is passed in explicitly at save time.**

Verbatim: "Metadata is captured automatically where it can be (git SHA, timestamps, feature-set name); the rest is passed in explicitly at save time."

Candidate evidence:
- A unit test calls the registry `save` path without supplying `git_sha` or `timestamp` and asserts that the stored metadata record contains non-`None` values for both fields.
- A second assertion confirms that omitting an explicitly-required field (e.g. held-out metrics) raises a `TypeError` or `ValidationError` rather than silently storing an incomplete record.
- Test name pattern: `test_registry_save_captures_git_sha_automatically`, `test_registry_save_raises_on_missing_required_explicit_field`.

**AC-4.  The leaderboard query is fast — listing a hundred runs should be instantaneous.**

Verbatim: "The leaderboard query is fast — listing a hundred runs should be instantaneous."

Candidate evidence:
- A unit test populates the registry with one hundred synthetic metadata records (no heavy model artefacts required) and asserts that `list` completes in under 1 second (wall-clock) on a laptop-class CPU.
- Test name pattern: `test_registry_list_hundred_entries_is_fast`.

**AC-5.  The on-disk layout is documented well enough that a contributor could inspect it by hand without the CLI.**

Verbatim: "The on-disk layout is documented well enough that a contributor could inspect it by hand without the CLI."

Candidate evidence:
- A documentation file (location TBD — see OQ-layout) describes the directory tree and metadata schema with field-by-field annotations.
- A test asserts that the documentation file exists at the expected path and is non-empty.
- Test name pattern: `test_registry_layout_documentation_exists`.

---

## §4 Non-functional requirements

Only requirements explicitly named in the intent are listed here (intent §AC-4 and §AC-5).  No additional NFRs are introduced.

**NFR-speed.**  Listing a hundred registered entries must be instantaneous on a laptop-class CPU (intent §AC-4).  The AC-4 test converts "instantaneous" to an under-1-second gate; this threshold must be confirmed by the human before the plan is finalised, as the intent does not quantify it.

**NFR-transparency.**  The on-disk layout must be human-readable without the CLI (intent §AC-5).  This implies that metadata records are stored in a text-inspectable format (e.g. JSON or YAML) rather than a binary format.  The exact serialisation format is unresolved; see OQ-layout.

---

## §5 Open questions

Each "Points for consideration" bullet in the intent (§Points for consideration) produces exactly one OQ.  They are not merged or expanded; the choice is surfaced, not resolved.

**OQ-layout — Flat directory vs. nested hierarchy.**  Intent §Points line 1: "A flat directory with metadata files is simplest; a nested hierarchy by target/model/date is easier to navigate by hand but harder to query."  What is the on-disk layout?  This is the intent's primary design decision for the stage.

**OQ-unit-of-registration — Model vs. run; when are two trainings the same entry?**  Intent §Points line 2: "Two trainings of the same model on different feature sets are two entries; two trainings of the same model on the same feature set on different days is a judgement call."  What counts as one registry entry vs. two?

**OQ-feature-hash — Whether to include a hash of the feature table in the metadata.**  Intent §Points line 3: "It makes reproducibility claims tighter but adds a step to every save."  Should the metadata record include a feature-table hash?

**OQ-query-filters — Whether a small query interface beyond "list everything" is worth building in.**  Intent §Points line 4: "Filtering by target, feature set, model type, or date range — if a meetup is going to involve 'show me the best model for X.'"  Should `list` support filter arguments at Stage 9?

**OQ-leaderboard-sort — The leaderboard's default sort order.**  Intent §Points line 5: "Lowest MAE is a defensible default; the interface should make it easy to sort by any available metric."  What is the shipped default sort key, and how is multi-metric sort exposed?

**OQ-registry-model-save — How the registry interacts with `save`/`load` in the Stage 4 protocol.**  Intent §Points line 6: "The simplest design: the registry calls the model's `save` to produce an artefact and stores metadata alongside it.  The model doesn't need to know the registry exists."  Is this the adopted interaction pattern, and does it require any change to the `Model` protocol at `/workspace/src/bristol_ml/models/protocol.py`?

**OQ-mlflow-path — Graduation path to MLflow.**  Intent §Points line 7: "If the registry interface is kept narrow, the migration is mechanical: a new implementation of the same small interface, backed by MLflow."  What interface contract ensures the graduation is mechanical rather than a rewrite?

**OQ-vcs-policy — Whether to commit the registry directory to version control.**  Intent §Points line 8: "Probably not — runs accumulate fast — but a portable export is useful."  Is the registry directory gitignored, and if so, what does the portable export look like?

---

## §6 Dependencies

**Upstream (all shipped at baseline SHA `575ac9c`):**

- Stage 5 (calendar features and feature assembler) — provides the feature-table schema and `feature_columns` provenance that registry metadata records reference.
- Stage 6 (enhanced evaluation) — provides the held-out metric values (`MAE`, `MAPE`, `RMSE`, `WAPE`) that the leaderboard displays.  The registry does not recompute metrics; it stores what `evaluate()` returns.
- Stage 7 (SARIMAX) — one of the three pre-registry models that AC-2 requires to save through the registry without code changes.
- Stage 8 (SciPy parametric) — the second pre-registry model subject to AC-2; its `ModelMetadata.hyperparameters` carries covariance data that the registry must preserve faithfully through its metadata store.

The Stage 4 naive and linear models are also covered by AC-2 (intent §Scope line 15: "Retrofit of the models shipped in Stages 4, 7, and 8").

**Downstream (per intent §Dependencies):**

- Stage 10 (simple neural network) — neural models save through the registry from the start; they must find a stable `save` interface to call.
- Stage 11 (complex neural network) — same requirement as Stage 10.
- Stage 12 (serving) — loads a named artefact from the registry; needs a stable `load`-by-name interface and a documented entry-point path.
- Stage 17 (price models) — saves price-target models alongside demand-target models; the registry interface must accommodate a `target` field in metadata without schema breakage.
- Stage 18 (drift monitoring) — reads prediction outputs stored alongside model artefacts; needs a documented convention for how prediction logs (if any) are co-located with registered entries.

---

## §7 Out of scope

The following items are out of scope for Stage 9, transcribed verbatim from the intent.

**From intent §Out of scope:**

> - A hosted registry (MLflow, W&B).  DESIGN §10 defers that; this stage is filesystem-based.
> - A UI beyond the CLI listing.
> - Model promotion / staging semantics (dev → staging → prod).
> - Artefact versioning beyond "last write wins" for a named experiment.

**From intent §Out of scope, explicitly deferred:**

> - Hosted registry services (MLflow, W&B).
> - Model promotion / staging.
> - Automated retraining (Stage 19 touches this).
> - Multi-user concurrent access.

---

**Absolute file paths referenced:**

- `/workspace/docs/intent/09-model-registry.md` — the immutable spec this document is derived from.
- `/workspace/src/bristol_ml/models/protocol.py` — the `Model` protocol (`fit`, `predict`, `save`, `load`, `metadata`) that the registry consumes without modification per AC-2.
- `/workspace/conf/_schemas.py` — `ModelMetadata`, which the registry stores and must extend or wrap; the `hyperparameters` escape hatch is the current forward-compatibility mechanism.
- `/workspace/src/bristol_ml/models/CLAUDE.md` — protocol semantics, serialisation rationale, and the `skops.io` upgrade note ("Stage 9 is the inflection point").
- `/workspace/docs/intent/DESIGN.md` — the §6 entry for Stage 9 naming `registry/filesystem.py` and the demo-moment CLI command.
