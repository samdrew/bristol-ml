# Stage 8 — SciPy Parametric Load Model: Requirements

**Source.** `/workspace/docs/intent/08-scipy-parametric.md` (immutable spec). Where this document and the spec disagree, the spec wins.

**Purpose.** Structured translation of the intent into user stories, acceptance criteria, non-functional requirements, and open questions. Input to `docs/plans/active/08-scipy-parametric.md`; not itself a plan.

---

## §1 Goal

Add a `ScipyParametricModel` that encodes electricity demand's known functional form — a temperature-response curve plus Fourier harmonics for diurnal and weekly seasonality — and fits it with `scipy.optimize.curve_fit`, yielding interpretable named coefficients ("this many megawatts per degree Celsius") and approximate confidence intervals derived from the optimiser's covariance matrix output. The pedagogical payoff is a single picture: the fitted temperature-response curve plotted against the raw demand scatter, with a parameter table printed alongside, making explicit in one notebook cell why the Stage 4 linear regression struggles at temperature extremes and why parameter uncertainty is a natural stepping-stone toward the probabilistic forecasting framing deferred to later stages.

---

## §2 User stories (Given/When/Then)

**US-1 — Meetup facilitator (demo moment)**

Given a warm feature-table cache and `notebooks/08_scipy_parametric.ipynb` open in a running Jupyter session, when the facilitator runs all cells, then a fitted temperature-response curve is rendered against the raw demand scatter and a parameter table with uncertainty bounds is printed, such that the facilitator can say "this coefficient means this many megawatts per degree Celsius, with this much uncertainty" without writing any code on the spot.

**US-2 — Self-paced learner**

Given the shipped Stage 8 notebook and completed Stages 0–7, when the learner reads the notebook top to bottom, then they understand from the parameter table alone what each fitted term represents physically (base load, temperature sensitivity, diurnal amplitude, weekly amplitude), and the notebook prose explains the assumptions behind the confidence intervals (Gaussian approximation near the optimum) honestly enough that the learner could apply the same approach to a different dataset.

**US-3 — Analyst comparing models**

Given fitted instances of `NaiveModel`, `LinearModel`, `SarimaxModel`, and `ScipyParametricModel`, when the analyst calls `evaluate()` on each and collects per-fold metrics, then the parametric model's MAE, MAPE, RMSE, and WAPE figures appear in the same long-form metrics DataFrame as every other model, with no model-specific branches required in the harness or benchmarks code.

**US-4 — Config-driven experimenter**

Given `conf/model/scipy_parametric.yaml` with functional-form parameters (base temperature, harmonic counts, loss function) exposed as top-level fields, when the experimenter overrides `model.loss` or `model.diurnal_harmonics` at the CLI without touching source code, then a different parametric variant trains and evaluates under the same `evaluate()` call, honouring DESIGN §2.1.4.

**US-5 — Future probabilistic-forecasting implementer (Stage 10+)**

Given a saved `ScipyParametricModel` artefact, when the implementer loads it, then `model.metadata.hyperparameters` exposes the fitted parameter vector and the associated covariance matrix, providing a documented entry point for propagating parameter uncertainty into a prediction interval without re-fitting.

---

## §3 Acceptance criteria — transcribed

The following five criteria are copied verbatim from `/workspace/docs/intent/08-scipy-parametric.md` §Acceptance criteria, then annotated with candidate evidence. Where the intent's wording is ambiguous, this is noted.

**AC-1. The parametric model conforms to the Stage 4 interface.**

Verbatim: "The parametric model conforms to the Stage 4 interface."

Candidate evidence:
- A protocol-conformance test passes `isinstance(model, Model)` after fit, and a static type checker produces no errors on the five members (`fit`, `predict`, `save`, `load`, `metadata`).
- File: `tests/unit/models/test_scipy_parametric.py`. Test name pattern: `test_scipy_parametric_conforms_to_model_protocol`.
- *Ambiguity:* "Stage 4 interface" refers to `bristol_ml.models.protocol.Model` (`src/bristol_ml/models/protocol.py:42–89`). The protocol is `@runtime_checkable`; the conformance test must exercise all five members, not only `isinstance`.

**AC-2. Fit and predict round-trip through save/load.**

Verbatim: "Fit and predict round-trip through save/load."

Candidate evidence:
- A unit test fits `ScipyParametricModel` on fixture data, saves to a temp path, loads, calls `predict()` on the same feature slice, and asserts numerical closeness (`np.testing.assert_allclose`).
- The test also verifies that a second `fit()` call on the same instance discards prior state (re-entrancy invariant from `models/CLAUDE.md §Protocol semantics`).
- File: same test module as the protocol conformance test, or a dedicated `test_scipy_parametric_io.py`.

**AC-3. The notebook demonstrates the fitted form visually and prints parameter estimates with confidence intervals.**

Verbatim: "The notebook demonstrates the fitted form visually and prints parameter estimates with confidence intervals."

Candidate evidence:
- A notebook at a path to be resolved by OQ-8. The notebook executes end-to-end under `jupyter nbconvert --execute` without error.
- Cell outputs include: (a) a scatter plot of national demand vs. temperature with the fitted temperature-response curve overlaid; (b) a printed parameter table with point estimates and ± uncertainty bounds; (c) a forecast comparison placing `ScipyParametricModel` alongside at least `LinearModel` and one prior model using the Stage 6 `forecast_overlay` helper.
- *Ambiguity:* the intent does not specify whether confidence intervals are 1-σ (≈68%) or 95% (2-σ) Gaussian bounds. The derivation strategy (covariance diagonal vs. bootstrap) is unresolved; see OQ-3.

**AC-4. The model fits in a reasonable time on the project's data.**

Verbatim: "The model fits in a reasonable time on the project's data."

Candidate evidence:
- A `%%time` cell in the notebook captures single-fold fit time. The intent's "reasonable time" language is vague; NFR-1 below converts it to a concrete gate (under 30 s for a single fold on a laptop CPU).
- *Ambiguity:* "reasonable time" is not quantified in the intent. NFR-1 proposes 30 s as a candidate gate, tighter than SARIMAX's 60 s because `curve_fit` is CPU-cheap. The human must confirm or revise this threshold before the plan is finalised.

**AC-5. Save/load preserves both the parameter values and the confidence-interval information.**

Verbatim: "Save/load preserves both the parameter values and the confidence-interval information."

Candidate evidence:
- The save/load round-trip test (AC-2) asserts not only that `predict()` output is numerically identical but also that `model.metadata.hyperparameters` contains a key for the covariance matrix (or equivalent CI representation) and that its values are bit-exactly equal after the round-trip.
- The save format (how the covariance matrix is embedded) is unresolved; see OQ-6.

---

## §4 Non-functional requirements

**NFR-1 Fit-time budget.**

A single-fold fit of `ScipyParametricModel` on a training window of up to 8 760 rows (one year of hourly data) must complete in under 30 seconds on a laptop CPU (circa 2022–2024, no GPU). `scipy.optimize.curve_fit` is substantially cheaper than a Kalman-filter SARIMAX; this budget is proposed as tighter than the Stage 7 gate (60 s) and must be confirmed by the human before shipping. A full rolling-origin evaluation must complete in under 5 minutes total.

**NFR-2 Save/load determinism.**

Given identical training data and `scipy`/`numpy` versions, `fit()` followed by `save()` then `load()` must return: (a) numerically bit-exact parameter values in `metadata.hyperparameters`; (b) bit-exact covariance-matrix values. `metadata.fit_utc` will differ across runs by design; all other fields must be identical.

**NFR-3 Reproducibility.**

Fixed initial parameter guesses (pinned in `ScipyParametricConfig`) plus identical data must yield identical fitted parameters across runs. If data-driven initial guesses are adopted (see OQ-2), the derivation must itself be deterministic given the training data.

**NFR-4 Numerical stability guard.**

`curve_fit` can converge to a solution with `pcov = inf` when the Jacobian is rank-deficient or parameters are unidentifiable. `ScipyParametricModel.fit()` must detect the `inf` covariance case, log a structured WARNING, and either raise a descriptive `RuntimeError` or store a sentinel (e.g. `None`) in `hyperparameters` rather than silently propagating `inf` values into `metadata`.

**NFR-5 Dependency footprint.**

`scipy` is not a transitive runtime dependency at Stage 7 (confirmed by the existing intelligence). Stage 8 must add `scipy` as a runtime dependency in `pyproject.toml`. The minimum acceptable version range must be documented (candidate: `scipy>=1.11` for stable `curve_fit` covariance semantics and `least_squares` loss support). No other new runtime dependencies are expected.

**NFR-6 Typed public interface.**

All public methods on `ScipyParametricModel` carry full type signatures. `# type: ignore` is not permitted without an in-line justification explaining why the type checker cannot resolve the annotation.

**NFR-7 British English.**

Docstrings, notebook prose, axis labels, config comments, and log messages use British English throughout ("behaviour", "colour", "modelled"), per `CLAUDE.md`.

**NFR-8 Standalone entrypoint.**

`python -m bristol_ml.models.scipy_parametric --help` must work per DESIGN §2.1.1.

---

## §5 Out of scope

The following items are explicitly out of scope for Stage 8, transcribed from the intent's §Out of scope, explicitly deferred:

- Bayesian estimation of the same functional form (MCMC, variational inference).
- Automatic functional-form search.
- Non-linear programming beyond `scipy.optimize.curve_fit` / `least_squares`.
- Probabilistic forecast scoring (pinball loss, CRPS).
- Prediction intervals that fully propagate parameter uncertainty into a calibrated interval (that is Stage 10).
- Model registry integration (that is Stage 9).
- Serving-layer integration (that is Stage 12).
- Any new evaluation-layer metrics or plot helpers beyond what Stage 6 already provides, unless AC-3 cannot be met without them (surface as an open question before implementing).

---

## §6 Open questions

| ID | Question | Proposed default | Cost of alternative |
|----|----------|-----------------|---------------------|
| OQ-1 | **Functional form for the temperature-response term.** Piecewise-linear hinge (two slopes, one breakpoint), smooth quadratic, or an asymmetric quadratic (separate coefficients above/below base temperature)? | Piecewise-linear hinge with a single `base_temp` breakpoint parameter. Interpretable ("heating ramp slope", "cooling ramp slope"), directly motivated by the physical heat-pump and air-conditioning dual drivers, and common in the load-modelling literature. | Smooth quadratic is simpler to code and avoids the non-differentiability at the hinge, but conflates heating and cooling sensitivity. Asymmetric quadratic adds two more parameters; identifiability risk increases. |
| OQ-2 | **Initial parameter guesses: fixed vs. data-driven.** Fixed guesses pinned in config (reproducible, documented, may miss badly scaled datasets) or data-driven (median demand as base load, OLS slope as temperature coefficient)? | Data-driven derivation from the training data as defaults, with config overrides available. This reduces convergence failures on unusual training windows (e.g. short rolling-origin folds) while remaining deterministic given the data. | Fixed guesses are simpler to document but increase convergence failure risk on edge-case folds. The intent explicitly calls out parameter identifiability as a concern, which argues for sensible starting values. |
| OQ-3 | **Confidence-interval derivation strategy.** Diagonal of the `pcov` covariance matrix (Gaussian approximation, assumes approximate linearity near optimum) vs. bootstrap resampling (distribution-free, slower)? | Covariance-diagonal Gaussian approximation (`1-σ` and `2-σ` bounds). Matches what `curve_fit` provides natively; the notebook must be honest about the assumptions (normality of residuals, linearity near optimum). | Bootstrap CIs are more valid under non-Gaussian residuals but add significant fit-time cost and complexity. Bootstrap is a better pedagogical bridge to Stage 10 but contradicts the fast-fit NFR-1 target. |
| OQ-4 | **Robust loss function.** `scipy.optimize.least_squares` supports `loss='linear'` (standard OLS), `'soft_l1'`, `'huber'`, `'cauchy'`, `'arctan'`. Which default? | `'soft_l1'` as the default, exposed as a config field. Less sensitive to demand outliers (e.g. cold-snap spikes) than plain `'linear'`; more stable than `'cauchy'`. The intent explicitly flags least-squares sensitivity to outliers. | `'linear'` is the `curve_fit` default and keeps CI derivation straightforward; `'huber'` is a well-known alternative but the huber delta scale needs an additional parameter. If covariance-based CIs are adopted (OQ-3), note that robust losses invalidate the standard `pcov` interpretation — the notebook must caveat this. |
| OQ-5 | **Diurnal harmonic count.** The Fourier helper `append_weekly_fourier` is parameterised by `harmonics` and `period_hours`. How many diurnal (24 h) harmonics should the default config include? | 4 diurnal harmonics. Captures the morning ramp, midday shoulder, evening peak, and overnight trough typical of GB demand with four sin/cos pairs (8 columns). Consistent with the energy-forecasting literature recommendation of 3–5 harmonics for daily profiles. | 2 harmonics is faster to fit but may underfit the evening peak; 6+ harmonics risk overfitting on short training windows. |
| OQ-6 | **Save format for the covariance matrix.** Options: (a) embed as a nested list under `hyperparameters["pcov"]`; (b) store as a separate numpy array attribute alongside the joblib payload; (c) store only the diagonal (variances) under `hyperparameters["param_variances"]`. | Embed as a nested list under `hyperparameters["pcov"]` and `hyperparameters["popt"]`. This keeps the covariance alongside the parameters in one `metadata.hyperparameters` dict, is visible to US-5's probabilistic use-case, and requires no protocol changes. Nested lists are JSON-serialisable if the Stage 9 registry needs them. | Separate numpy attribute is cleaner for matrix operations but requires callers to know the attribute name outside the protocol surface. Diagonal-only saves space but loses off-diagonal correlation structure needed for correlated-parameter CI propagation. |
| OQ-7 | **UTC-tz index guard stance.** SARIMAX requires `freq="h"` and accepts only tz-aware input. The Fourier helper requires tz-aware input. Should `ScipyParametricModel.fit()` replicate SARIMAX's strict UTC guard (raises on tz-naive) or use the `LinearModel` precedent (accepts tz-naive)? | Match the SARIMAX precedent: raise `ValueError` on tz-naive input with a message naming the `freq="h"` UTC contract. The Stage 3 assembler guarantees UTC-aware output; any tz-naive frame reaching `fit()` indicates a caller bug that should be surfaced, not silently tolerated. | LinearModel's more permissive stance is easier to test but risks silent phase misalignment in the Fourier terms, since `append_weekly_fourier` itself raises on tz-naive. Consistency with SARIMAX reduces the number of guard policies a learner must internalise. |
| OQ-8 | **Notebook target: new `notebooks/08_scipy_parametric.ipynb` vs. append to a prior notebook.** | New `notebooks/08_scipy_parametric.ipynb`. Stage 8 introduces a distinct model with its own narrative (functional form motivation → fit → parameter table → temperature-response plot → cross-model comparison). Every prior modelling stage (4, 7) has its own notebook. | Appending to `07_sarimax.ipynb` is tempting because the cross-model comparison cell already exists there, but it conflates two distinct model narratives in one artefact and makes Stage 8 harder to demo in isolation at a meetup. |
| OQ-9 | **Per-horizon behaviour under rolling-origin: refit every fold or share parameters?** `curve_fit` is cheap (< 30 s per NFR-1), so refitting every fold is feasible. However, the intent notes fold-to-fold parameter stability as a diagnostic; should the notebook print parameter values per fold to demonstrate stability? | Refit every fold (standard rolling-origin semantics, consistent with all other models). Print a parameter-stability diagnostic (e.g. a table of `popt` per fold, or a simple coefficient plot) in the notebook as a pedagogical bonus, not as a protocol requirement. | Sharing parameters across folds would break rolling-origin semantics and introduce data leakage. Not an option. |
| OQ-10 | **Weekly harmonic count.** How many weekly (168 h) Fourier harmonics in the default config? `append_weekly_fourier` with `period_hours=168` can be reused directly (the existing intelligence confirms this). | 3 weekly harmonics (matching the SARIMAX `weekly_fourier_harmonics=3` default from Stage 7). Provides consistency across models and keeps the feature space manageable. | Stage 7 chose 3 based on practical fit-time and interpretability grounds; the same reasoning applies here. Increasing to 5 would improve weekly pattern capture but adds 4 columns and increases the risk of identifiability issues with the temperature-response parameters. |
| OQ-11 | **Should `ScipyParametricConfig` be wired into the Hydra discriminated union?** The `ModelConfig` union (`conf/_schemas.py:542–545`) currently covers `NaiveConfig | LinearConfig | SarimaxConfig`. Two dispatcher sites must be extended (`harness.py:475–501` and `train.py:220–251`). | Yes, wire it in fully: `type: scipy_parametric` in `conf/model/scipy_parametric.yaml`; `ScipyParametricConfig` added to the `ModelConfig` union; a new branch in both dispatcher sites. This is the DESIGN §2.1.4 requirement and enables `python -m bristol_ml.evaluation.harness model=scipy_parametric` without code changes. Both dispatcher sites must be updated together in the same PR. | Notebook-only approach (no Hydra wiring) is faster but is a silent spec deviation from §2.1.4 and must not be shipped silently. |
| OQ-12 | **Minimum `scipy` version bound.** What is the minimum `scipy` version that should be pinned in `pyproject.toml`? | `scipy>=1.11,<2`. `curve_fit` covariance semantics and `least_squares` loss-function support have been stable since `1.7`; `1.11` is the oldest release likely to be available on modern Python 3.12 environments and aligns with the project's likely CI matrix. The human should confirm whether a tighter or looser bound is preferred. | A lower bound (e.g. `>=1.7`) maximises compatibility but risks behaviour differences in `pcov` edge cases. An upper bound cap is not recommended at this stage. |
| OQ-13 | **Dispatcher duplication (Stage 7 B1 carry-over).** Stage 7 Phase 3 code review raised the two-dispatcher duplication (`harness.py:475` + `train.py:220`) as a candidate for an ADR but did not file one. Should Stage 8 file the ADR and unify before extending, or extend both sites once more and revisit later? | Extend both sites once more (fourth branch in each), record the ADR as a new H-number deferral. The cost of a refactor mid-stream outweighs the marginal cost of one more dual edit; the ADR is the right response when a fifth model family appears. | Filing the ADR and unifying now is cleaner but expands Stage 8 scope and risks delaying the pedagogical milestone. |

---

**Absolute file paths referenced:**

- `/workspace/docs/intent/08-scipy-parametric.md` — the immutable spec this document is derived from
- `/workspace/src/bristol_ml/models/protocol.py` — the `Model` protocol (`fit`, `predict`, `save`, `load`, `metadata`) and `ModelMetadata` that `ScipyParametricModel` must conform to
- `/workspace/conf/_schemas.py` — `ModelMetadata` (lines 548–589), `SarimaxConfig` (lines 494–539), and the `ModelConfig` discriminated union (lines 542–545)
- `/workspace/src/bristol_ml/features/fourier.py` — `append_weekly_fourier` (lines 56–152), reusable for the diurnal term with `period_hours=24`
- `/workspace/src/bristol_ml/evaluation/harness.py` — first dispatcher site (lines 475–501)
- `/workspace/src/bristol_ml/train.py` — second dispatcher site (lines 220–251)
- `/workspace/src/bristol_ml/models/CLAUDE.md` — protocol semantics (re-entrancy, predict-before-fit, cross-version compatibility)
- `/workspace/src/bristol_ml/evaluation/CLAUDE.md` — API growth trigger rule (do not add a second boolean flag to `evaluate()`)
