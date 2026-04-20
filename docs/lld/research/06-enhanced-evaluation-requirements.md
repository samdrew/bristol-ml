# Stage 6 — Enhanced Evaluation & Visualisation: Requirements

**Source.** `docs/intent/06-enhanced-evaluation.md` (the spec). Where this document and the spec disagree, the spec wins.

**Purpose.** Structured translation of the intent into user stories, numbered acceptance criteria, non-functional requirements, and the open questions that the plan must resolve. This document is input to the plan; it is not the plan.

---

## 1. User stories

**US-1 — Meetup facilitator.** As the person running a live demo, I open `notebooks/04_linear_baseline.ipynb` (or a Stage 6 follow-up notebook) and the diagnostics are large enough to read from the back of the room, colourblind-safe, and annotated so I can narrate "this is where the model handles worst" without clicking anywhere.

**US-2 — Self-paced learner.** As someone working through the repo alone, I look at the diagnostics and understand — without reading the source — where the linear baseline leaves residual structure (weekly ripple, autocorrelation at lag 168, per-hour bias), which prepares me for why Stage 7 (SARIMAX) or Stage 10 (NN) might matter.

**US-3 — Future-stage implementer.** As the author of Stage 7, 8, 10, 11, or 16, I add a fitted model + the held-out predictions to the helper library and every diagnostic for my model drops into a notebook cell with about as much code as the Stage 4 `harness.evaluate` line. The helper library does not know the difference between a `LinearModel`, a `SARIMAXModel`, or a neural net.

---

## 2. Acceptance criteria

AC-1 through AC-4 are quoted from intent §Acceptance criteria. AC-5 onwards are derived to make the intent operational.

**From the intent (AC-1..AC-4 — must pass).**

1. **AC-1.** A model that conforms to the Stage 4 `Model` protocol can produce every diagnostic with a small, consistent amount of code. The diagnostic surface is model-agnostic — no `isinstance(model, LinearModel)` branches in the helper module.
2. **AC-2.** The diagnostics are visually legible at meetup-audience distances: large fonts (>=12pt axis labels, >=14pt titles), clear axes, limited clutter.
3. **AC-3.** The helper library has no dependencies on any specific model implementation. Importing it must not import `LinearModel`, `NaiveModel`, or any future model.
4. **AC-4.** The updated notebook (Stage 4's, plus any Stage 6 follow-up) runs top-to-bottom quickly on a laptop — under 60 s for the plots portion, under 120 s end-to-end.

**Derived from intent prose (AC-5..AC-13).**

5. **AC-5.** The Stage 4 notebook's existing residual block is upgraded in place, not duplicated in a new notebook. Stage 6 is a diagnostic-surface stage, not a narrative stage.
6. **AC-6.** Colourblind accessibility: default palette is formally accessibility-certified (Okabe-Ito or equivalent), not `tab10`. One-line opt-out for facilitators who want a bespoke palette.
7. **AC-7.** ACF (and/or PACF) plot is exercised on the linear baseline's residuals with at least 168 lags, so the weekly-seasonality spike motivating Stage 7 is visible. The default lag count from `statsmodels.graphics.tsaplots.plot_acf` is too short for hourly data and must be overridden.
8. **AC-8.** Per-hour and per-weekday error breakdowns are produced. Per-horizon breakdowns are explicitly deferred to the first multi-horizon model stage (intent §Points for consideration).
9. **AC-9.** Forecast overlay — actual vs predicted — is legible on the last N days of a fold. An uncertainty band (derived from rolling-origin errors, not from the model) is a proposed default; resolve in the plan.
10. **AC-10.** Helpers are runnable standalone under `python -m bristol_ml.evaluation.plots --help` per DESIGN §2.1.1.
11. **AC-11.** CI green: `uv run pytest` passes; `uv run ruff check .` clean; `uv run pre-commit run --all-files` clean.
12. **AC-12.** Evaluation-layer CLAUDE.md updated with the new surface; layer architecture doc (`docs/architecture/layers/evaluation.md`) records the new module and a paragraph on the diagnostic contract.
13. **AC-13.** `CHANGELOG.md` `[Unreleased]` gains an `### Added` bullet for Stage 6; `docs/lld/stages/06-enhanced-evaluation.md` retrospective filed; `docs/stages/README.md` row flipped to `shipped` with links.

---

## 3. Non-functional requirements

**NFR-1 Performance.** Each helper function returns within ~2 s on the Stage 4 rolling-origin output (~30-fold × 7-day tests on hourly data). The Stage 4 notebook must still run under its ~120 s ceiling after adding the Stage 6 calls.

**NFR-2 Dependency minimalism.** Prefer libraries already in the environment. `matplotlib` is dev-only at baseline; `statsmodels` is runtime. The plan must resolve whether to promote matplotlib to runtime or lazy-import inside the helper module (both are viable).

**NFR-3 Accessibility.** Default palette must be Okabe-Ito (Wong 2011) or IBM's accessibility palette — formally colourblind-safe for deuteranopia, protanopia, and tritanopia. `tab10` is explicitly NOT acceptable as the default.

**NFR-4 GitHub rendering.** Notebook outputs must render on github.com without Chrome (= no plotly without static kaleido export; no altair without mime-renderer setup). Static matplotlib PNGs are the safe path.

**NFR-5 Standalone entrypoint.** Per DESIGN §2.1.1, the helper module must be runnable as `python -m bristol_ml.evaluation.plots --help` with a minimal self-check.

**NFR-6 Typed public interface.** Every public helper exposes a full type signature; no `# type: ignore` without an in-line justification.

**NFR-7 British English.** Doc strings, axis labels, notebook prose — all British English ("colour", "behaviour"), per CLAUDE.md.

**NFR-8 Thin notebook.** Per §2.1.8, the notebook imports from `src/bristol_ml/` and does not reimplement plotting logic.

---

## 4. Open questions (OQ) — for plan Design Decisions

OQ-1 through OQ-9 are what the plan must resolve with a proposed default + evidence, per the D-numbered pattern from Stage 5.

**OQ-1 — Viz library pick.** Intent §Points for consideration names matplotlib, seaborn, plotly, altair. A single pick reduces cognitive load for downstream stages. Which, and why?

**OQ-2 — Default palette.** Okabe-Ito vs IBM accessibility vs seaborn `"colorblind"` (which is not strictly Okabe-Ito). Which, and how is it injected — rcParams global vs per-plot kwarg?

**OQ-3 — Opinionatedness.** Very opinionated ⇒ every model's notebook looks the same (good for comparison); very unopinionated ⇒ facilitators improvise (good for live demos). Where does Stage 6 draw the line?

**OQ-4 — Breakdown dimensions.** Intent names hour, weekday, month cheap; weather regime / holiday proximity richer but domain-specific. Which to ship at Stage 6; which to defer?

**OQ-5 — ACF/PACF placement.** Intent §Points for consideration says ACF plots "live naturally in this stage, setting up Stage 7". Confirm in the plan; ensure lag count is overridden (see AC-7 + domain research).

**OQ-6 — Per-horizon diagnostics.** Intent §Points for consideration explicitly considers deferring. Confirm "defer until first multi-horizon model stage" in the plan.

**OQ-7 — Uncertainty band.** Intent §Points for consideration opens the door to visualising an empirical-quantile band from rolling-origin per-day errors. This requires per-fold predictions, which the harness currently discards — it only keeps metrics. Either (a) extend the harness to emit a predictions frame, (b) let the plot helper re-run the rolling loop, or (c) require the caller to pass predictions in. Which?

**OQ-8 — `NesoBenchmarkConfig.holdout_start/_end` consumer.** These fields were added at Stage 4 but no consumer exists (Stage 4 retro "Deferred", Stage 5 plan H-1 deferred to Stage 6). Is Stage 6 where they get wired up, via a fixed-window retrospective plot against the NESO benchmark?

**OQ-9 — Module name.** `src/bristol_ml/evaluation/plots.py` vs `viz.py` vs `diagnostics.py`. Naming convention binds on downstream imports.

---

## 5. Inconsistencies flagged for human

**I-1 — Stage 5 retro "Next" section.** `docs/lld/stages/05-calendar-features.md` ends with "**Next** → Stage 6 tree-based models" (or similar wording). Intent `docs/intent/06-enhanced-evaluation.md` is unambiguously "Enhanced evaluation & visualisation" — no new models. Per the spec-drift rule, intent wins. The Stage 5 retro should be corrected as a housekeeping edit during Stage 6 implementation. Flagged here, not silently edited.

**I-2 — matplotlib dev-only.** `pyproject.toml` declares `matplotlib>=3.8,<4` in `dependency-groups.dev`, not `[project].dependencies`. The Stage 6 plots module needs matplotlib at runtime to satisfy AC-10 (`python -m bristol_ml.evaluation.plots --help`). Plan must choose: (a) promote matplotlib to runtime dependency; (b) lazy-import inside functions so `python -m bristol_ml` on a headless install does not break; or (c) document the dev-install requirement and scope CLI to a help-only surface. Options have different blast radii for downstream stages.

---

## 6. Tension

**T-1 — Uncertainty band vs model-agnosticism.** An empirical-quantile band needs per-fold predictions. If the helper recomputes by calling `model.fit` + `model.predict` inside a rolling loop, it knows about `Model`'s methods and is tightly coupled to that interface — satisfies AC-1 but arguably violates AC-3 ("no dependencies on any specific model implementation"). If instead the caller hands in a pre-computed predictions frame, the helper is fully model-agnostic but the notebook needs an extra rolling loop. The plan must resolve: is AC-3 meant as "no dependency on a *particular* model class" (LinearModel, NaiveModel) or "no dependency on the `Model` protocol at all"? The former is the natural reading.

---

## 7. Summary of inputs the plan needs

- **Library pick** (OQ-1) — resolved by the domain-researcher doc.
- **Palette** (OQ-2) — resolved by the domain-researcher doc.
- **Opinionatedness** (OQ-3) — plan's judgement call, informed by user stories.
- **Breakdown dimensions** (OQ-4) — plan's judgement call, informed by intent + research.
- **ACF lags** (OQ-5) — resolved by the domain-researcher doc (lags=168 mandatory).
- **Per-horizon defer** (OQ-6) — intent already defers; plan just confirms.
- **Uncertainty band contract** (OQ-7) — plan decides between three paths.
- **Holdout consumer** (OQ-8) — plan decides whether to wire now or defer.
- **Module name** (OQ-9) — plan's judgement call; `plots.py` is the obvious default.
- **matplotlib runtime** (I-2) — plan decides between promote / lazy-import / scope-down.
