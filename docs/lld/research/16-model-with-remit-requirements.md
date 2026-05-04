# Stage 16 — Model with REMIT features: requirements

**Source intent:** `docs/intent/16-model-with-remit.md`
**Artefact role:** Phase 1 research deliverable (requirements analyst).
**Audience:** plan author (lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## 1. Goal

Join the structured REMIT features produced by Stage 14's extractor into the hourly feature table, retrain the best-performing prior model against the enriched set, and produce a notebook that presents a three-row ablation — best model without REMIT features, best model with REMIT features, NESO benchmark — so that a meetup audience can see concretely, positive or negative, what a textual regulatory disclosure stream contributes to a day-ahead demand forecast.

---

## 2. User stories

**US-1 — Meetup facilitator (demo path).**
Given the REMIT cache has been ingested (Stage 13), the LLM extractor has been run (Stage 14), and the REMIT-enriched feature table has been built,
when the facilitator opens the Stage 16 notebook and runs it top-to-bottom,
then a three-row metric table is displayed: the best prior model without REMIT features, the same model class retrained with REMIT features, and the NESO day-ahead benchmark — with honest commentary that reads correctly whether the REMIT contribution is positive, negligible, or negative.
(Intent §Demo moment; AC-3, AC-5.)

**US-2 — Self-paced learner (bi-temporal lesson).**
Given the learner has worked through Stage 13's bi-temporal ingestion,
when they read the Stage 16 feature-derivation code,
then the `as_of` primitive from Stage 13 is visibly in the join path and the code makes clear why each training-row's REMIT features must only reflect information published by that row's timestamp — so the learner can explain what "no data leakage" means in a real event-driven dataset.
(Intent §Bi-temporal correctness; AC-1.)

**US-3 — Researcher trying to beat NESO (analytical path).**
Given the same rolling-origin splits and evaluation metrics that all prior model stages have used,
when the researcher retrains the best prior model on the REMIT-enriched feature set,
then they receive MAE, MAPE, RMSE, and WAPE figures that are directly comparable to the prior-stage leaderboard, and can determine whether REMIT features narrow the gap to NESO's own forecast.
(Intent §Purpose; AC-3, AC-4.)

**US-4 — Stage 16 implementer (assembler extension).**
Given the existing assembler already has a `weather_only` and a `weather_calendar` feature set,
when the implementer adds the REMIT-enriched variant,
then downstream model code, the train CLI, and the registry select it via a configuration switch (e.g. `features=weather_calendar_remit`) without any change to the model classes, the harness, or the registry verbs.
(DESIGN.md §2.1.4; AC-2.)

**US-5 — CI runner / offline clone (stub path).**
Given the repository is checked out without an OpenAI API key and `BRISTOL_ML_LLM_STUB=1` is set,
when CI runs the tests for Stage 16,
then the REMIT feature derivation runs against the stub extractor's output, the assembler produces a valid REMIT-enriched feature table, and the model training path exercises the full code path — without any network I/O and without requiring a pre-populated real-extraction cache.
(DESIGN.md §2.1.3; implied from Stage 14's stub discipline.)

---

## 3. Acceptance criteria

The following five are restated verbatim from the intent (lines 31–34), followed by implied criteria that the intent leaves implicit:

**AC-1 — Bi-temporal correctness.**
The REMIT-derived features are computed correctly for historical periods so that each training row's REMIT features reflect only information that was published by or before that row's timestamp. The `as_of(df, t)` primitive from Stage 13 must be visibly used in the derivation path; callers who additionally want "events active at t" must chain a valid-time filter on `effective_from` / `effective_to`. No row in the training set may contain REMIT information published after that row's `timestamp_utc`.

- Testable form: a unit test constructs a synthetic REMIT frame with a revision published after time T and asserts that the derived features for rows at T do not reflect that revision.

**AC-2 — Assembler configuration switch.**
The assembler produces the REMIT-enriched feature set via a configuration switch (a new Hydra feature-set group file). Selecting the enriched set must not break the `weather_only` or `weather_calendar` paths. The enriched schema must be documented as a constant (e.g. `REMIT_OUTPUT_SCHEMA`) analogous to `CALENDAR_OUTPUT_SCHEMA`.

- Testable form: `python -m bristol_ml.features.assembler features=weather_calendar_remit` (or the chosen group name) completes on a machine with stub caches present; `load` with the REMIT schema rejects a `weather_calendar` parquet (wrong column set), and vice versa.

**AC-3 — Same splits and metrics as prior stages.**
The retraining run uses the same rolling-origin split configuration, the same four point-forecast metrics (MAE, MAPE, RMSE, WAPE), and the same NESO benchmark comparison that the prior model stages used. Cross-stage metric comparability requires identical `test_len`, `step`, and `min_train_periods` values to those used when the chosen model was originally evaluated.

- Testable form: the notebook's metric table has columns `mae`, `mape`, `rmse`, `wape` (matching the Stage 9 sidecar keys), and the split config is either loaded from the registered prior run's sidecar or explicitly documented as matching it.

**AC-4 — Ablation is reproducible from the registry.**
Both runs that appear in the ablation — "best model without REMIT" and "best model with REMIT" — must be loadable via `registry.load(run_id)`. The prior-stage run uses its registered `run_id`; the Stage 16 retrained run is registered fresh. The notebook must not re-fit either model at demo time; it loads registered artefacts and calls `predict`.

- Testable form: the notebook's ablation cell does not call `model.fit()`; it calls `registry.load()` then `model.predict()`. A test can assert this via source inspection of the notebook builder script (matching the Stage 11 pattern).

**AC-5 — Honest notebook commentary.**
The notebook contains a clearly labelled prose cell that interprets the ablation result, including the case where REMIT features do not help. This cell must not be empty and must acknowledge the domain reason why REMIT is more informative for price than for demand (unplanned nuclear outages move price more than they move national demand). The commentary must be present and non-trivial regardless of the numerical result.

- Testable form: the notebook builder script contains a markdown cell at the ablation position whose source string length exceeds a minimum threshold (e.g. 200 characters) and contains at least one of the words "price", "demand", "effect", or "null".

**Implicit AC-6 — Forward-looking REMIT feature.**
The feature derivation includes at least one "known unavailability over the next 24 hours" column in addition to the "currently active" signal, because the day-ahead forecasting context makes this defensible and the intent explicitly names it (intent §"Forward-looking features"). Both the current-state and the forward-looking variants must be documented in `features/CLAUDE.md`.

- Testable form: `REMIT_OUTPUT_SCHEMA` (or equivalent) contains at least one column whose name includes a forward-window indicator (e.g. `remit_unavail_mw_next_24h`), and a unit test asserts its value is zero when no future events are scheduled.

**Implicit AC-7 — Zero-event hour handling.**
The REMIT columns are dominated by zeros (most hours have no active events). The feature derivation must not drop zero-valued hours and must not replace zero with NaN. The assembled REMIT-enriched feature table must have no NaN values in the REMIT columns for any hour covered by the demand/weather data, consistent with the assembler's general no-NaN invariant.

- Testable form: a test assembles the REMIT feature table over a period with no active events and asserts all REMIT columns are zero (not NaN) for that period.

**Implicit AC-8 — Module runs standalone.**
`python -m bristol_ml.features.remit` (or the chosen entry point) executes without error, prints the resolved config, and emits a sample row of derived features for a known timestamp.
(DESIGN.md §2.1.1.)

**Implicit AC-9 — At least one test on the public interface.**
`bristol_ml.features.remit` has at minimum one smoke test that derives features from a fixture REMIT frame and asserts the output conforms to `REMIT_OUTPUT_SCHEMA`.
(DESIGN.md §2.1.7.)

**Implicit AC-10 — Notebook runs top-to-bottom under stub mode.**
The Stage 16 notebook executes end-to-end under `BRISTOL_ML_LLM_STUB=1` without error, using stub-extractor output for the REMIT features and producing a complete ablation table (values will reflect stub data, not real extraction quality).

---

## 4. Non-functional requirements

**NFR-1 — Bi-temporal correctness is structurally enforced (AC-1; Stage 13 discipline).**
The feature derivation module must not accept a raw REMIT DataFrame directly. It must call `as_of(df, t)` from `bristol_ml.ingestion.remit` as the first step for each timestamp, ensuring transaction-time filtering is applied before any valid-time filtering. The function signature must require a timezone-aware UTC timestamp. Any caller that bypasses `as_of` will expose themselves to data leakage; the module docstring must state this explicitly.

**NFR-2 — CI uses stub extractor; real-extraction artefact is a separate registered run (DESIGN.md §2.1.3; intent §"Whether to run the stub extractor…").**
The Stage 14 stub extractor output is the default for CI and for the assembler's offline path. A real-extractor-based run, if produced, is registered separately in the registry and referenced from the notebook as an alternative. The notebook must not require a real-extractor run to produce the ablation table; it must produce a valid (if less meaningful) table from stub-extractor features.

**NFR-3 — Idempotent assembly (DESIGN.md §2.1.5).**
Re-running the REMIT feature assembly overwrites or skips the cached parquet; it never corrupts a partially-written file. The atomic-write idiom (`_atomic_write` from `ingestion._common`) must be used. Re-running with identical inputs must produce a byte-identical output parquet.

**NFR-4 — Schema is typed and documented (DESIGN.md §2.1.2).**
A `REMIT_OUTPUT_SCHEMA` constant (pyarrow schema) must enumerate every REMIT-derived column with its arrow type and tz metadata. Column names must follow the project's `snake_case` convention and must not collide with any column in `CALENDAR_OUTPUT_SCHEMA`. The schema must be importable from `bristol_ml.features.remit` without triggering ingestion or extraction.

**NFR-5 — Configuration in YAML (DESIGN.md §2.1.4).**
The aggregation level (e.g. total vs per-fuel-type breakdown), the forward-looking window width (e.g. 24 hours), and the chosen feature subset must all be Hydra YAML fields, not hard-coded values. The Pydantic schema for the REMIT-enriched feature set must live in `conf/_schemas.py` alongside `FeatureSetConfig`.

**NFR-6 — Reproducibility of the registered retraining run (DESIGN.md §2.1.6; AC-4).**
The Stage 16 registry entry must record: the REMIT-enriched feature set name, the feature schema version (column names), whether stub or real extraction was used, the git SHA, and the split configuration. These fields are sufficient to reproduce the run from the same data inputs. The sidecar's `feature_set` field must distinguish `weather_calendar_remit` (or the chosen name) from `weather_calendar`.

**NFR-7 — Notebooks are thin (DESIGN.md §2.1.8).**
The Stage 16 notebook imports from `bristol_ml.features.remit` and from `bristol_ml.registry`; it does not reimplement aggregation, bi-temporal filtering, or metric computation. All substantive computation lives in the module. The notebook executes top-to-bottom without errors given stub caches (no real API calls required).

**NFR-8 — Observability (loguru convention from Stages 13–15).**
The feature derivation module emits a structured `loguru` INFO line per assembly call covering: the number of active-event hours, the number of zero-event hours, the number of forward-window hours with non-zero scheduled capacity, and the final row count. It emits WARNING if the REMIT cache covers fewer hours than the demand/weather cache (potential leakage gap, not necessarily an error).

---

## 5. Out of scope

Restated verbatim from intent (§Out of scope and §Out of scope, explicitly deferred):

- Fine-tuning Stage 14's prompt engineering based on Stage 16 results (conflates extraction quality with model quality).
- Adding every possible REMIT-derived feature; a small, well-motivated set is the target.
- Any restructuring of the rolling-origin split.
- Iterating on Stage 14's extractor based on Stage 16's results.
- REMIT features for the price target (Stage 17 may pick this up).
- Feature-importance or attribution analysis.

---

## 6. Open questions

**OQ-1 — Which prior model is "the best-performing model"?**
The intent says "the best-performing model from prior stages" but does not name it. The Stage 11 ablation table on a 90-day CPU-recipe window shows `nn_temporal` first (MAE 3768 MW), but this was measured on a constrained training window with CPU-recipe architecture. On the full feature table with the production `seq_len=168, num_blocks=8, channels=128` config, the ranking may differ. SARIMAX and SciPy parametric performed poorly in the single-holdout predict-only protocol but better in proper rolling-origin evaluation.
*Best guess:* the TCN (`nn_temporal`) is the most principled choice as the single best-performing temporal model, with the caveat that it requires the CUDA host or significant CPU patience. If the stage is to be demoable on a laptop CPU, retraining `sarimax` or `linear` on the enriched feature set may be more practical for the default demo path. The human should nominate the specific model (or models) before the plan is written.

**OQ-2 — Which aggregation level for the REMIT features?**
The intent names at minimum "total unavailable capacity at this hour by fuel type" and "number of active unplanned outages" but flags collinearity risk in per-fuel-type breakdowns.
*Best guess:* start with three aggregate columns: `remit_unavail_mw_total` (all fuel types summed), `remit_active_unplanned_count` (count of active unplanned events), and `remit_unavail_mw_next_24h` (total scheduled unavailability in the next 24 hours, for the forward-looking signal named in the intent). Per-fuel-type breakdown (e.g. `remit_unavail_mw_nuclear`, `remit_unavail_mw_gas`) is a config-selectable extension, off by default. This is a plan decision the human should confirm before implementation; the number of columns affects the REMIT-enriched schema contract.

**OQ-3 — Should the registered ablation artefact use the stub or the real extractor?**
The intent says "for the registered artefact, the real extractor is probably right; for CI, the stub is." This leaves ambiguous whether the stage ships a real-extraction-based registered run as part of the stage definition of done, or treats real extraction as optional.
*Best guess:* the stage should ship one registered run using stub-extractor features (reproducible in CI, used by default), and provide documented instructions for producing a second registered run with real-extractor features. The notebook must be able to reference either `run_id` by config switch. Whether the real-extraction run is required before the PR merges is a gate the human must set.

**OQ-4 — How many models should be retrained on the enriched set?**
The intent acknowledges "the argument for retraining several is that 'REMIT helps model A but hurts model B' is itself informative. Budget-dependent."
*Best guess:* one model (whichever answers OQ-1) is the minimum for the stage definition of done. Retraining a second model (e.g. linear alongside the best-performing complex model) doubles the demo value for modest implementation cost and makes the "REMIT helps/hurts model X" story concrete. The human should decide whether single-model is sufficient or whether two is worth the extra retraining time.

**OQ-5 — Where does `features/remit.py` live relative to `llm/extractor.py`?**
The intent names a `features/remit.py` module. The feature layer (pure transformations, no I/O against external APIs per DESIGN §3.2) is the correct home for the aggregation logic. But the module must consume Stage 14's extractor output, which lives in the `llm/` layer. The import direction must not go from `features/` into `llm/` at module-import time (that would couple the feature-table schema to the LLM layer on every import). The correct pattern — loading the Stage 14 parquet output by path and processing it as a plain DataFrame — needs to be stated explicitly in the module contract.
*Best guess:* `features/remit.py` accepts the path to Stage 14's extraction parquet (or a DataFrame loaded from it) as a function argument; it does not import from `bristol_ml.llm` at module level. The module also accepts the path to Stage 13's REMIT parquet for the bi-temporal join. This keeps the feature layer independent of the LLM layer.

**OQ-6 — What is the handling of the "no extraction output available" case?**
If Stage 14 has not been run with real extraction (only stub), the extracted features for most historical hours will be stub-quality (fixed sentinel values). Should the assembler warn and continue, or should it gate on a minimum extraction coverage threshold?
*Best guess:* the assembler warns via loguru if more than a configurable fraction of REMIT events have stub-sentinel `confidence` values, but it continues — the notebook commentary (AC-5) is where the "stub extraction quality" caveat lives, not a hard gate. A config field `min_extraction_confidence` (default 0.0, meaning no gate) makes the threshold adjustable.

**OQ-7 — What is the REMIT-enriched feature table's time coverage requirement?**
The demand model trains on 2018-01-01 onwards (DESIGN §5.1). Stage 13 notes that REMIT goes back to 2014. If the REMIT cache covers fewer hours than the demand/weather cache (e.g. because only a recent window was ingested), the outer join will produce zero-filled REMIT features for uncovered hours — which is semantically correct (no known events) but may be misleading if the cache is simply incomplete.
*Best guess:* the assembler inner-joins on the demand/weather timeline (existing convention from Stage 3); REMIT columns default to zero for hours outside the REMIT cache window; a WARNING fires if the REMIT window is shorter than the demand window (NFR-8). The human should confirm whether a minimum REMIT history requirement should be enforced as a hard gate.

---

*This artefact is one of the Phase-1 research inputs for Stage 16. It covers requirements only. Codebase patterns, domain research, and scope-diff analysis are in companion artefacts.*
