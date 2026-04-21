# Stage 7 — SARIMAX: Requirements

**Source.** `/workspace/docs/intent/07-sarimax.md` (immutable spec). Where this document and the spec disagree, the spec wins.

**Purpose.** Structured translation of the intent into user stories, acceptance criteria, non-functional requirements, and open questions. Input to `docs/plans/active/07-sarimax.md`; not itself a plan.

---

## 1. Goal

Add a statsmodels SARIMAX model that conforms to the Stage 4 `Model` protocol, demonstrates seasonal decomposition as the pedagogical unlock for understanding GB electricity demand structure, and enables the existing Stage 6 diagnostic plot helpers to produce SARIMAX diagnostics without any new plumbing.

---

## 2. User stories (Given/When/Then)

**US-1 — Pedagogical narrator (meetup facilitator)**

Given a warm feature-table cache and a Stage 7 notebook open in a running Jupyter session, when the facilitator runs all cells, then a seasonal decomposition of the demand series appears on screen (trend, seasonal, residual components), followed by a SARIMAX forecast overlay against the linear baseline on the same held-out period, such that the facilitator can narrate "the ACF spike at lag 168 we saw in Stage 6 is exactly what the seasonal component captures here" without writing any code on the spot.

**US-2 — Solo self-learner**

Given the shipped Stage 7 notebook and completed Stages 0–6, when the learner reads the notebook top to bottom, then they understand — from the seasonal decomposition cell alone — why a SARIMAX seasonal order parameter maps to a 24-hour or 168-hour cycle, and the order-selection prose (information-criterion comparison or decomposition-guided rationale) explains the chosen order clearly enough that they could reproduce the decision for a different dataset.

**US-3 — Future-stage implementer (Stages 8, 10, 11)**

Given a fitted `SARIMAXModel` instance and the Stage 6 helper library (`bristol_ml.evaluation.plots`), when the implementer calls `evaluate(..., return_predictions=True)` and passes the resulting `predictions_df` to `forecast_overlay_with_band` or `acf_residuals`, then the helpers produce correct outputs without any model-specific branches — SARIMAX residuals and predictions flow through the same `pd.Series` / `pd.DataFrame` interface as `NaiveModel` and `LinearModel` output.

**US-4 — Config-driven experimenter**

Given a `conf/model/sarimax.yaml` with order parameters and exogenous column list exposed as top-level fields, when the user changes `model.order` or `model.seasonal_order` at the CLI without touching source code, then a different SARIMAX variant trains and evaluates under the same `evaluate()` call, demonstrating DESIGN §2.1.4 (configuration lives outside code).

---

## 3. Acceptance criteria — transcribed

The following five criteria are copied verbatim from `/workspace/docs/intent/07-sarimax.md` §Acceptance criteria, then annotated with required evidence.

**AC-1. The SARIMAX model conforms to the Stage 4 interface.**

Verbatim from intent: "The SARIMAX model conforms to the Stage 4 interface."

Evidence required:
- A protocol-conformance test (see AC-5) passes `isinstance(model, Model)` at runtime and a static type checker produces no errors on the `fit / predict / save / load / metadata` signatures.
- File: `tests/unit/models/test_sarimax_protocol.py` (or the conformance test referenced in AC-5). Test name pattern: `test_sarimax_conforms_to_model_protocol`.

**AC-2. Fit and predict round-trip through save/load.**

Verbatim: "Fit and predict round-trip through save/load."

Evidence required:
- A unit test fits `SARIMAXModel` on synthetic or fixture data, saves it to a temp path, loads it, calls `predict()` on the same test features, and asserts numerical closeness (`pd.testing.assert_series_equal` or `np.testing.assert_allclose`).
- File: same test module as AC-5. Test name pattern: `test_sarimax_save_load_round_trip`.

**AC-3. The model trains in a reasonable time on the project's data (SARIMAX can be slow; the order needs to be sane).**

Verbatim: "The model trains in a reasonable time on the project's data (SARIMAX can be slow; the order needs to be sane)."

Evidence required:
- The notebook runs end-to-end within the time budget defined in NFR-1 (see section 4 below). A `%%time` cell in the notebook captures single-fold fit time; a comment in `conf/model/sarimax.yaml` documents the observed fit time for the chosen order.
- The notebook cell output is the primary evidence. Optionally: a pytest marker `@pytest.mark.slow` wrapping a single-fold timing assertion.

**AC-4. The notebook renders a seasonal decomposition, a fit diagnostic, and a forecast comparison.**

Verbatim: "The notebook renders a seasonal decomposition, a fit diagnostic, and a forecast comparison."

Evidence required:
- A Stage 7 notebook at a path to be resolved by OQ-10 (see section 6). The notebook executes end-to-end under `jupyter nbconvert --execute` without error.
- Cell outputs include: (a) a `statsmodels.tsa.seasonal.seasonal_decompose` or `STL` decomposition plot; (b) at minimum one fit diagnostic (residual plot and/or ACF of SARIMAX residuals via `acf_residuals`); (c) a `forecast_overlay` call placing SARIMAX alongside at least `LinearModel`.

**AC-5. A protocol-conformance test covering fit/predict/save/load exists.**

Verbatim: "A protocol-conformance test covering fit/predict/save/load exists."

Evidence required:
- File: `tests/unit/models/test_sarimax_protocol.py` (or appended to the existing `test_protocol.py` if that pattern was used for Stage 4 models — check the Stage 4 convention).
- Test must exercise all five protocol members (`fit`, `predict`, `save`, `load`, `metadata`) and assert that `isinstance(model, Model)` is `True` after fit.

---

## 4. Non-functional requirements

**NFR-1 Fit-time budget.**

Assumption (explicit): the rolling-origin evaluator uses an expanding training window with `min_train_periods = 8760` rows (one year of hourly data, as per `conf/evaluation/rolling_origin.yaml`). For a demo notebook running a _single_ evaluation fold (not the full rolling-origin sweep), the target is fit completion in under 60 seconds on a laptop CPU (circa 2022–2024 hardware, no GPU). For a full rolling-origin run of N folds, acceptable wall time is under 5 minutes total with the chosen order and training window — if the default configuration exceeds this, the order or training window must be reduced before shipping. These targets are provisional; OQ-3 surfaces the training-window design choice and the human must confirm an acceptable budget. The intent's "reasonable time" language is deliberately vague; this document converts it to a concrete gate.

**NFR-2 Predict-time budget.**

`predict()` on a 24-row test window (the standard day-ahead horizon) must complete in under 2 seconds per fold. SARIMAX prediction is substantially cheaper than fitting; this is expected to be trivially met but must be verified.

**NFR-3 Memory.**

A single SARIMAX fit on a training window of up to 8760 rows with a small exogenous feature set (temperature plus up to four calendar columns) must not exceed 2 GB resident memory. Larger training windows or higher-order models may require this budget to be revisited; if the default configuration approaches the limit, document it in the notebook.

**NFR-4 Determinism.**

Given identical training data, configuration, and statsmodels version, `fit()` followed by `predict()` must return numerically identical results on every run. SARIMAX (via statsmodels) uses deterministic numerical optimisation by default; verify that no random state is introduced by the exogenous feature construction path. `metadata.fit_utc` will differ across runs by design; all other fields and all prediction values must be identical.

**NFR-5 Protocol re-entrancy.**

A second call to `fit()` on the same `SARIMAXModel` instance must discard prior state and not layer state on top. This is a load-bearing invariant of the `Model` protocol (`src/bristol_ml/models/CLAUDE.md` §Protocol semantics). The save/load round-trip test (AC-2) must include a second `fit()` call to verify this.

**NFR-6 Typed public interface.**

All public methods on `SARIMAXModel` carry full type signatures. `# type: ignore` is not permitted without an in-line justification explaining why the type checker cannot resolve the annotation (the typical case is statsmodels' partial typing stubs).

**NFR-7 British English.**

Docstrings, notebook prose, axis labels, and config comments use British English throughout ("behaviour", "colour", "modelled"), per `CLAUDE.md`.

**NFR-8 Standalone entrypoint.**

`python -m bristol_ml.models.sarimax --help` must work per DESIGN §2.1.1.

---

## 5. Out of scope

The following items are explicitly out of scope for Stage 7, transcribed from the intent's §Out of scope, explicitly deferred:

- Automatic order selection (auto-ARIMA or equivalent) as an architectural feature. Order selection in the notebook as a manual pedagogical exercise is permitted and encouraged.
- Multivariate SARIMAX (VARMAX).
- State-space models beyond SARIMAX (Kalman filters, dynamic linear models).
- Bayesian time-series models.
- Vector autoregression.
- Probabilistic forecast evaluation (pinball loss, CRPS).
- Serving-layer integration (that is Stage 12).
- Model registry integration (that is Stage 9).
- Any new evaluation-layer metrics or plot helpers beyond what Stage 6 already provides, unless AC-4 cannot be met without them (in which case, surface as an open question before implementing).

---

## 6. Open questions (numbered, each with proposed default + rationale + cost of alternative)

**OQ-1 — Which ARIMA order to pin?**

The intent flags that hand-picked order is pedagogically clearer than automatic search. The decision is _which_ order to choose.

Proposed default: `SARIMAX(p=1, d=1, q=1)(P=1, D=1, Q=1, s=24)` with a Fourier-term exogenous workaround for the weekly period (see OQ-2), selected via AIC comparison over a small candidate grid `{p,q} ∈ {0,1}`, `{P,Q} ∈ {0,1}` documented in the notebook. This is the canonical "start simple" order that most SARIMAX tutorials use for hourly energy data; the AIC grid is small enough to run in a notebook cell in a few minutes and demonstrates the order-selection methodology without black-boxing it.

Cost of alternative (e.g. fixed `(1,1,1)(2,1,1,24)` with no grid): saves one notebook cell and a few minutes of runtime, but the learner has no evidence that the chosen order is principled. A hand-waved order undercuts the pedagogical goal.

**OQ-2 — How to handle dual seasonal periods (24 h daily, 168 h weekly)?**

Statsmodels SARIMAX with `s=168` is technically correct for a weekly period but is known to be very slow and may fail to converge for a seasonal order above `P=1, Q=1`. Three options: (a) single `s=168` with low seasonal order; (b) stacked seasonal terms (statsmodels does not natively support two `seasonal_order` tuples — requires a workaround); (c) treat daily seasonality via the ARIMA seasonal order (`s=24`) and encode weekly pattern as Fourier exogenous regressors (sin/cos terms at period 168).

Proposed default: option (c) — `s=24` seasonal order for the ARIMA component, plus `sin(2πt/168)` and `cos(2πt/168)` as exogenous columns. This is the standard practical solution for electricity demand data with dual seasonality, it keeps fit time manageable, and the Fourier terms are interpretable. The notebook prose should explain why `s=168` was rejected: it makes the computational cost explicit, which is itself a teaching moment.

Cost of the simplest alternative (single `s=24`, ignore weekly pattern): faster fit, but the weekly residual pattern will be visible in the ACF of SARIMAX residuals — a regression from the Stage 6 story, which pointed at lag 168 as the unsolved problem. Shipping a SARIMAX that ignores the weekly period undermines the stage's narrative hook.

**OQ-3 — Training-window size for demo speed?**

SARIMAX fit cost scales roughly as O(n²) in the number of training rows (Kalman filter iterations). The full expanding rolling-origin window reaches ~8760 rows in the first fold and grows. On a laptop CPU, a full rolling-origin evaluation with ~365 folds at expanding training windows may take an hour or more, depending on the chosen order and `s` value.

Proposed default: cap the training window at 8760 rows (one year of hourly data) for all folds — effectively switching to a fixed sliding window (`fixed_window=true`, `min_train_periods=8760`) for the SARIMAX evaluation specifically, overriding the project-wide expanding-window default. The notebook prose must explain this as a deliberate speed/coverage trade-off, not a mistake. The human must confirm whether this is acceptable or whether a smaller cap (e.g. 2 × 24 × 7 = 336 rows for a two-week demo window) is preferred.

Cost of using the full expanding window: correct methodology, but potentially a 30-minute notebook fit — incompatible with a live-demo setting and AC-3. If the chosen order fits in under 5 minutes for the full window, the cap is unnecessary; OQ-1's order choice and this question are coupled.

**OQ-4 — Which exogenous regressors to include?**

Options range from (a) temperature only; (b) temperature plus Fourier weekly terms (per OQ-2's proposed default); (c) all weather variables from the Stage 3 feature table; (d) temperature plus the Stage 5 calendar one-hots (44 columns).

Proposed default: temperature (`temperature_2m`) plus the two Fourier weekly terms from OQ-2 — nothing else. Temperature is the dominant exogenous driver of electricity demand; adding all 44 calendar one-hots inflates the design matrix and slows fitting without a clearly motivating narrative at this stage. The notebook can note that calendar one-hots are an extension, motivating Stage 8's explicit functional-form approach.

Cost of including all weather plus calendar features: slower fit, risk of near-multicollinearity between calendar dummies and Fourier terms, and a more complex notebook narrative. If the human wants to use the full Stage 5 feature set as a comparison experiment, that is better positioned as a notebook cell than as the default configuration.

**OQ-5 — Should SARIMAX confidence intervals be plotted in the notebook?**

SARIMAX produces analytical confidence intervals as a by-product of the Kalman filter (`get_forecast().conf_int()`). The Stage 6 `forecast_overlay_with_band` helper exists for empirical quantile bands, not analytical ones. DESIGN's global out-of-scope list explicitly defers probabilistic forecasting; the Stage 6 empirical band is the current uncertainty surface.

Proposed default: do not use or plot SARIMAX's analytical confidence intervals in the notebook. Plotting them alongside the Stage 6 empirical band would introduce two competing uncertainty surfaces with different semantics, which is confusing for learners. Reserve analytical CIs for a brief notebook comment explaining that they exist and pointing to Stage 10 (quantile regression) as the canonical probabilistic stage. The `get_forecast().summary_frame()` output may still be printed as a table cell for pedagogical completeness, without being rendered as a plot band.

Cost of plotting the CIs: moderate — mostly pedagogical cost (confusion about two uncertainty conventions) and a minor implementation cost (a new plot helper or overloading `forecast_overlay_with_band`). If the human's view is that free CIs are too good to skip, the least-bad option is a standalone `forecast_overlay_with_sarimax_ci` helper in the notebook itself (not promoted to `plots.py`), keeping the production helper library clean.

**OQ-6 — Which residual diagnostics belong in the notebook vs. the Stage 6 plot helpers?**

The Stage 7 intent recommends Ljung-Box, Jarque-Bera, and heteroscedasticity tests as residual diagnostics. Statsmodels exposes these via `SARIMAXResultsWrapper.test_serial_correlation`, `.test_normality`, and `.test_heteroskedasticity`. The Stage 6 `acf_residuals` helper is already available for the ACF view.

Proposed default: run Ljung-Box, Jarque-Bera, and White/Breusch-Pagan via `statsmodels.stats.stattools` in the notebook directly (two to three lines per test), printed as a table. Do not promote these to new `plots.py` helpers. The Stage 6 ACF helper covers the visual side; the statistical tests are printed numbers, not plots. Promoting them to `plots.py` would be premature generalisation — they are not obviously reusable across non-ARIMA models.

Cost of adding new helpers: an additional `plots.py` function adds surface area that must be maintained for every subsequent stage. If a future stage (Stage 10, Stage 11) also needs ARIMA-style residual tests, that is the point to abstract — not Stage 7.

**OQ-7 — Cross-model comparison: in-notebook or deferred to a future comparison notebook?**

The intent explicitly says: "Whether to compare against the linear baseline within the notebook or wait for a cross-model notebook later. The former is cheaper and feels complete in itself."

Proposed default: include the comparison in the Stage 7 notebook. Use `forecast_overlay` (Stage 6 helper) to overlay SARIMAX against `LinearModel` and `NaiveModel` on the same held-out window. A summary metric table (MAE, MAPE, RMSE, WAPE per model) can be printed using `evaluate()` for each model. This is consistent with every prior modelling stage having produced a self-contained comparison cell, and avoids an open-ended dependency on a future cross-model notebook that has no scheduled stage.

Cost of deferring: the Stage 7 notebook has no quantitative comparison, reducing its standalone value as a learning artefact. The Stage 6 benchmark machinery already supports multi-model comparisons; not using it here wastes the investment.

**OQ-8 — Order selection methodology: seasonal decomposition + manual picks, or AIC grid?**

The intent flags seasonal decomposition as the pedagogical approach. AIC-grid comparison is more rigorous but risks being mistaken for auto-order search (which is out of scope).

Proposed default: a two-step notebook cell sequence — (1) a seasonal decomposition plot that motivates the order qualitatively (this is the "demo moment" from the intent); (2) a small AIC/BIC table comparing four to six candidate orders (`p, q ∈ {0,1}`, `P, Q ∈ {0,1}` with `D=1`) to justify the final pick quantitatively. Both steps together are the order-selection section; the decomposition is the visual hook, the AIC table is the evidence. This approach is well-established in Hyndman & Athanasopoulos `fpp3` Chapter 9 and is directly reusable by learners.

Cost of decomposition-only (no AIC table): faster notebook, but order selection looks arbitrary. Cost of AIC grid only (no decomposition): misses the intent's explicit "demo moment" of making seasonal structure visible.

**OQ-9 — Does the evaluation harness need changes to accommodate SARIMAX's native multi-step forecasting?**

The intent states: "SARIMAX forecasts a horizon natively, which matches the day-ahead framing well. The rolling-origin evaluator needs to be happy with that." The harness currently calls `model.predict(X_test)` and expects a `pd.Series` indexed to `X_test.index`. SARIMAX's `get_forecast(steps=n)` produces a `PredictionResults` object, not a `pd.Series` directly; the `SARIMAXModel.predict()` implementation must convert it. The question is whether this conversion is entirely internal to `predict()` or whether the harness needs a new code path.

Proposed default: hide all SARIMAX multi-step logic inside `SARIMAXModel.predict()`, which accepts a `features: pd.DataFrame` (for the exogenous values over the forecast horizon) and returns a `pd.Series` indexed to `features.index`. From the harness's perspective, `predict()` is identical in shape to `LinearModel.predict()`. The `EvaluationResult` dataclass promotion is NOT triggered by Stage 7 — that trigger fires at the second extension request to `evaluate()`, and Stage 7 makes no such request.

Cost of harness modification: if `predict()` truly cannot be made to return a correctly-indexed `pd.Series` without harness knowledge of SARIMAX internals (e.g. because the forecast horizon must be inferred from steps rather than from the shape of `X_test`), then the harness would need a small adaptation. This is a risk to surface during implementation discovery, not to pre-solve here. If the risk materialises, the resolution options are: (a) a `predict(features, steps=None)` optional parameter on the protocol (potentially a protocol change — escalate); (b) a parallel `predict_horizon(n_steps, exog)` method on `SARIMAXModel` used only in the notebook, not through the harness.

**OQ-10 — Notebook target: new `07_sarimax.ipynb` or append to an existing notebook?**

Stage 6 (D11) appended new cells to `04_linear_baseline.ipynb`. Stage 5 got its own `05_calendar_features.ipynb`. The pattern for modelling stages (Stage 4 baseline, Stage 5 calendar) has been one notebook per stage; Stage 6 was a special case because it was explicitly a "diagnostic surface" stage that enhanced Stage 4's artefact rather than introducing a new model.

Proposed default: a new `notebooks/07_sarimax.ipynb`. SARIMAX is a distinct model with its own narrative arc (seasonal decomposition → order selection → ARIMA fit → cross-model comparison); appending it to the Stage 4 or Stage 5 notebook would make the pedagogical path opaque. Every prior modelling stage (Stage 4, Stage 5) has its own notebook. Stage 7 should follow that pattern.

Cost of appending to an existing notebook: the Stage 6 precedent exists and the comparison cells could live alongside the existing Stage 4 linear-baseline comparison. But it would break the one-notebook-per-model convention and make the Stage 7 artefact harder to demo in isolation.

**OQ-11 — Should `SARIMAXConfig` be wired into the Hydra discriminated union alongside `LinearConfig` and `NaiveConfig`?**

The project config uses a `model` Hydra group with a `type` discriminator field. Adding `SARIMAXConfig` requires: a new `conf/model/sarimax.yaml`, an extension to `conf/_schemas.py`'s discriminated union, and a new factory branch in the harness CLI's `_build_model_from_config`. This is the established pattern from Stages 4–5.

Proposed default: yes, wire it in fully. `type: sarimax` in `conf/model/sarimax.yaml`; `SARIMAXConfig` in `conf/_schemas.py`; factory branch in `harness._build_model_from_config`. This is the DESIGN §2.1.4 requirement and enables `python -m bristol_ml.evaluation.harness model=sarimax` without code changes — the live-demo pattern that every prior stage has honoured.

Cost of notebook-only approach (no Hydra wiring): faster to implement, but `SARIMAXModel` can only be instantiated by hand in a notebook, not via the CLI. This would be a silent spec deviation from §2.1.4 and must not be shipped silently.

**OQ-12 — Should `statsmodels` remain the only new dependency, or is `pmdarima` acceptable for order-selection utilities?**

`pmdarima` provides `auto_arima` and several ARIMA diagnostic helpers. Auto-order search is out of scope, but `pmdarima` also has convenience wrappers for AIC grids and KPSS stationarity tests that are otherwise a few lines of statsmodels boilerplate.

Proposed default: statsmodels only. `statsmodels` is already a runtime dependency; `pmdarima` would be a new runtime dependency for convenience that adds ~40 MB to the install and creates a coupling risk if its API diverges from statsmodels. The AIC grid and stationarity tests are short enough to implement in the notebook or in `SARIMAXModel` directly using `statsmodels.tsa` utilities. The project's simplicity bias (DESIGN §2.2.4) argues against adding a dependency for code that fits in ten lines.

Cost of using `pmdarima`: slightly cleaner notebook AIC table cell; minor dependency risk. If the human disagrees with the blanket exclusion, the acceptable compromise is a dev-only dependency (notebook-only import, not in production code).

---

## 7. Known tensions

**(a) D9 single-flag concession on `evaluate()` vs. what SARIMAX's native multi-horizon output wants.**

The Stage 6 D9 architectural-debt note (codified in `src/bristol_ml/evaluation/CLAUDE.md` §"Harness output — API growth trigger") is explicit: if a future stage needs to extend harness output beyond the single `return_predictions` flag, the correct response is a first-class `EvaluationResult` dataclass, not a second boolean. SARIMAX produces its entire forecast horizon from a single `get_forecast(steps=n)` call rather than iterating point-by-point; if the `predict()` method cannot cleanly return a `pd.Series` indexed to `X_test.index` without the harness knowing about the number of steps, then Stage 7 may be the trigger for the `EvaluationResult` promotion. This is not a certainty — hiding the complexity inside `predict()` is likely viable — but it is a risk that must be checked during implementation discovery. If it fires, the implementer MUST NOT add a second flag; they must propose the dataclass promotion.

**(b) DESIGN §10 probabilistic deferral vs. SARIMAX-free analytical confidence intervals.**

DESIGN's global out-of-scope list defers all probabilistic/quantile forecasting. The Stage 6 empirical-quantile band is the current approved uncertainty surface. SARIMAX's analytical CIs are a natural by-product of the fit that require zero additional code to produce. The tension is: showing them is pedagogically appealing and technically free; not showing them is architecturally clean and avoids introducing a second uncertainty convention before Stage 10 resolves the probabilistic framing. OQ-5 above proposes not plotting them; the human must decide whether this is too conservative given the cost differential.

**(c) Fit-time budget vs. weekly seasonal period 168.**

Using `s=168` directly in the SARIMAX `seasonal_order` makes the model specification match the physical reality (weekly period) but is slow and may be numerically unstable for `P > 1`. The OQ-2 proposed default (Fourier exogenous for the weekly component) resolves this in practice but means the SARIMAX model does not model the weekly period through its SARIMA structure — it does so via external regressors. This is a legitimate and common approach in the literature but must be documented clearly in the notebook, because a learner naively reading the `seasonal_order=(1,1,1,24)` specification would not know the weekly period is being handled elsewhere. The alternative (`s=168`) is the clean specification but may require reducing the notebook's training window to a few hundred rows to stay within the fit-time budget, which has its own pedagogical cost (small training windows produce unstable parameter estimates).

---

**Absolute file paths referenced:**

- `/workspace/docs/intent/07-sarimax.md` — the immutable spec this document is derived from
- `/workspace/src/bristol_ml/models/protocol.py` — the `Model` protocol and `ModelMetadata` that `SARIMAXModel` must conform to
- `/workspace/src/bristol_ml/evaluation/harness.py` — the rolling-origin evaluator; the D9 single-flag debt note is in its docstring
- `/workspace/src/bristol_ml/evaluation/CLAUDE.md` — the "API growth trigger" rule at §Harness output
- `/workspace/conf/evaluation/rolling_origin.yaml` — current splitter defaults (`min_train_periods=8760`, `test_len=24`)
- `/workspace/conf/model/linear.yaml` — pattern for the new `conf/model/sarimax.yaml`
- `/workspace/docs/plans/completed/06-enhanced-evaluation.md` — D9 and D11 decisions that constrain Stage 7
