# Plan — Stage 6: Enhanced evaluation & visualisation

**Status:** `approved` — human markup 2026-04-20: D1, D2, D3, D4, D7, D8, D10 accepted verbatim; **D5 amended** (default figsize 12×8, Hydra-configurable); **D6 amended with verification gate** (keep `display_tz="Europe/London"` iff DST transitions render correctly; fall back to UTC-only otherwise); **D7 reinforced** (daily + weekly reference markers made explicit in the spec); **D9 accepted one-off** with an architectural-debt note on re-engineering triggers; **D11 overridden** (addendum cells appended to Stage 4 notebook, not in-place refactor). Phase 2 ready.
**Intent:** [`docs/intent/06-enhanced-evaluation.md`](../../intent/06-enhanced-evaluation.md)
**Upstream stages shipped:** Stages 0–5 (foundation, NESO demand, weather, feature assembler + splitter, linear baseline + evaluation harness + three-way NESO benchmark, calendar features).
**Downstream consumers:** every future modelling stage (7, 8, 10, 11, 16) inherits the Stage 6 diagnostic surface; Stage 9 (model registry) and Stage 18 (drift monitoring) share the same evaluation-layer primitives.
**Baseline SHA:** `a2071da` (tip of `reclaude` at plan time — Stage 5 T7 hygiene commit).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md)
- Codebase map — [`docs/lld/research/06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md)
- External research — [`docs/lld/research/06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md)

**Pedagogical weight.** The intent calls this a "diagnostic surface" stage — no new models, no new metrics. Its value is that every future modelling stage demos with richer diagnostics, so the weekly-ripple story (Stage 5's headline) generalises to an ACF spike at lag 168 (motivating Stage 7's SARIMAX), a per-hour-per-weekday heatmap (motivating interaction terms at Stage 8), and a forecast overlay with an empirical uncertainty band (motivating probabilistic forecasting downstream). Over-opinionated helpers lock every future notebook into the same shape; under-opinionated helpers leave every future stage reinventing matplotlib. The line is drawn at moderate opinionatedness — sensible defaults, all parameters overridable, `ax=` passthrough for composability.

---

## 1. Decisions for the human (resolve before Phase 2)

Eleven decision points plus four housekeeping carry-overs from Stage 5 and one new housekeeping item. For each, I propose a default that honours the simplicity bias + the research evidence, and cite the supporting source. Mark each `ACCEPT` / `OVERRIDE: <alt>` in your reply; I'll update the plan before Phase 2.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Viz library pick | **matplotlib + seaborn.** matplotlib for the core plot primitives and ACF; seaborn for the `colorblind` palette helper and the 24×7 heatmap. No plotly, no altair. | matplotlib renders on github.com without any extra runtime; seaborn is a thin ~295 kB wrapper around matplotlib; plotly/altair need Chrome/kaleido or a mime renderer setup that github.com does not provide. Every existing notebook already uses matplotlib exclusively — zero migration cost. | External research §R1 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md); codebase map §2a + §3 in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md). |
| **D2** | Default palette | **Okabe-Ito qualitative, `cividis` sequential, `RdBu_r` diverging.** Injected at `plots.py` import time via `plt.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)`. Opt-out is a one-liner for facilitators. | Okabe-Ito (Wong 2011, *Nature Methods*) is the de-facto accessibility-certified qualitative palette for mixed-audience presentations. `cividis` is the CVD-safe perceptually-uniform sequential colormap (Nunez et al. 2018). `RdBu_r` is the diverging default that preserves red/blue affect for signed residuals. `tab10` (matplotlib default) is rejected — not colourblind-safe. | External research §R2 + clear-adopt A1 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md). |
| **D3** | matplotlib dependency tier | **Promote `matplotlib>=3.8,<4` from `[dependency-groups].dev` to `[project].dependencies`.** `statsmodels` is already runtime; matplotlib joins it. | AC-10 requires `python -m bristol_ml.evaluation.plots --help` to work on a non-dev install. DESIGN §2.1.1 requires every module run standalone. Lazy-import-inside-function workarounds exist but make the module harder to reason about for ~1.4 MB saved. The project is pedagogical; the audience runs `uv sync` once. Promoting is the simple path. | Codebase map §3 (matplotlib is dev-only today; statsmodels is runtime) in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); requirements doc §5 I-2 in [`06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md). Also promote `seaborn>=0.13,<1` to runtime (per D1). |
| **D4** | Module name | **`src/bristol_ml/evaluation/plots.py`.** Tests in `tests/unit/evaluation/test_plots.py`. Config (if any) in `conf/evaluation/plots.yaml`. | The layer already has `splitter.py`, `metrics.py`, `harness.py`, `benchmarks.py` — all simple noun filenames. `plots.py` is the conventional peer name, matches the `evaluation/` CLAUDE.md wording "Stage 6 will add visualisation and richer-diagnostics primitives". `diagnostics.py` is more specific but longer; `viz.py` is fine but less discoverable. | Codebase map §5 (layer doc anticipates Stage 6 visualisation module) in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); external research §J3 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md). |
| **D5** | Opinionatedness + default figsize | **Moderate opinionatedness; default figsize `(12.0, 8.0)` (human mandate 2026-04-20); Hydra-configurable via `PlotsConfig.figsize`.** Helpers have sensible defaults (Okabe-Ito palette, `Europe/London` display tz per D6, 12×8 figsize, British-English axis labels). Every parameter overridable at the call site. Every helper accepts an optional `ax: matplotlib.axes.Axes` so facilitators can compose figures. The `PlotsConfig.figsize` field in `conf/evaluation/plots.yaml` sets the default used when the helper is called without explicit `figsize=`. | Intent §Points for consideration frames the trade-off explicitly. US-1 (facilitator) wants live-demo improvisation; US-3 (future-stage implementer) wants drop-in usage. The `ax=` passthrough satisfies both: cheap-by-default, composable-when-needed. 12×8 is the projector-friendly default — wider than the matplotlib 6.4×4.8 baseline for meetup legibility (AC-2); taller than the 10×6 default so a 2×2 grid in the notebook does not squash vertically. | Requirements doc §4 OQ-3 + §1 user stories in [`06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md). |
| **D6** | Diagnostic surface (what to ship) + DST handling | **Six helpers, default `display_tz="Europe/London"` gated by a DST-verification test.** (i) `residuals_vs_time(residuals, *, display_tz="Europe/London", ax=None) -> Figure`; (ii) `predicted_vs_actual(y_true, y_pred, *, ax=None) -> Figure`; (iii) `acf_residuals(residuals, *, lags=168, ax=None) -> Figure`; (iv) `error_heatmap_hour_weekday(residuals, *, display_tz="Europe/London", ax=None) -> Figure`; (v) `forecast_overlay(actual, predictions_by_name, *, window=None, ax=None) -> Figure`; (vi) `forecast_overlay_with_band(actual, point_prediction, per_fold_errors, *, quantiles=(0.1, 0.9), ax=None) -> Figure`. **DST-handling verification (human mandate 2026-04-20):** T3 includes a test that exercises each time-axis helper across one spring-forward Sunday (last Sunday in March) and one fall-back Sunday (last Sunday in October). Expected behaviour: pandas `tz_convert("Europe/London")` produces correctly-localised timestamps — spring-forward leaves a 1-hour x-axis gap (no 01:00–02:00 local); fall-back presents two datapoints at the same local wall-clock hour (both UTC-distinct instants). If either case mis-renders (e.g. duplicate timestamps silently overwrite, mdates formatter throws, or heatmap `.hour`-groupby double-counts incorrectly), **fall back to default `display_tz="UTC"`** across all six helpers and record the regression in the Stage 6 retro. Expectation: pandas+matplotlib handle this correctly today, so `Europe/London` is expected to ship. | Hero four (i–iv) map directly to Hyndman fpp3 §5.3's canonical residual-diagnostic layout. (v) is already exercised by notebooks 04 and 05 — promoting it to a helper discharges §2.1.8 (thin notebooks). (vi) is the intent's uncertainty visualisation, kept separate from (v) because it has a different data requirement (per-fold errors). Local-time display makes the hour-of-day heatmap pedagogically readable ("demand at 18:00" is a local concept); UTC-only fallback is acceptable but costs the intuition in meetups. | External research §R3, §R4, §R5, §R6 + clear-adopts A4, A6 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md); codebase map §2c helper-candidate table + §4 Cell 11 target in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); [Stage 5 plan D7 DST rule](../completed/05-calendar-features.md) sets the precedent for documenting DST handling verbatim. |
| **D7** | ACF lag count + reference markers | **`lags=168` with explicit reference markers at lag 24 (daily) AND lag 168 (weekly)** *(human mandate 2026-04-20: daily seasonality is the dominant cycle in electricity demand and the ACF must visually reflect that, not just the weekly spike)*. Default the helper's `lags` parameter to 168 so the full 0–168 range is rendered (which includes every multiple-of-24 daily spike 24/48/72/96/120/144/168 plus the weekly resonance). Annotate with two labelled `axvline` markers: "daily (24)" at lag 24 with alpha 0.35, "weekly (168)" at lag 168 with alpha 0.35. Bartlett 95% confidence band (`alpha=0.05`) via statsmodels default. | `statsmodels.graphics.tsaplots.plot_acf`'s default `lags = min(int(10*log10(len(x))), len(x)//2 - 1)` gives ~39 lags for an 8760-row residual series — it **misses the weekly spike at lag 168 entirely** and truncates the daily cascade after 24–48. Showing both markers makes the daily-then-weekly periodicity story legible for meetup audiences: "the biggest repeat is daily (24 h), and weekly regression ripples on top of that (168 h)." That matches the order of physical dominance in GB demand and sets up the teaching arc into Stage 7 (SARIMAX daily + weekly seasonal orders). | External research §R4 + clear-adopt A2 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md); intent §Points for consideration; Hyndman fpp3 §5.3 ACF interpretation. |
| **D8** | Uncertainty band derivation | **Empirical quantile (q10–q90) of signed errors across folds, per forecast-horizon position.** Non-parametric, model-agnostic. The plot helper accepts a pre-computed `per_fold_errors: pd.DataFrame` indexed by fold and horizon; derivation is the caller's responsibility. | Method (a) from research §R6. Method (b) (`mean ± 1.96σ`) rejected as the simpler approximation because the quantile method is not much more code and is the principled choice. Method (c) (statsmodels prediction interval) rejected — couples helper to `LinearModel.results`, violating AC-3. | External research §R6 + clear-adopt A4 in [`06-enhanced-evaluation-domain.md`](../../lld/research/06-enhanced-evaluation-domain.md); requirements doc §6 T-1 (model-agnosticism tension) in [`06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md). |
| **D9** | Harness predictions emission (**accepted one-off**) | **Extend `evaluation.harness.evaluate` with a backward-compatible `return_predictions: bool = False` parameter.** When `True`, returns `(metrics_df, predictions_df)` instead of `metrics_df`. `predictions_df` has one row per (fold, forecast-horizon-index) with columns `[fold_index, test_start, test_end, horizon_h, y_true, y_pred, error]`. **Architectural-debt note (human mandate 2026-04-20):** this is accepted as a one-off tolerance for a single return-shape divergence. If a future stage needs to extend the harness output further — e.g. Stage 9 adds a run-id column, Stage 18 adds drift scores, a multi-horizon model needs per-horizon predictions with a different shape, or a probabilistic forecaster needs quantiles — the implementer MUST NOT add a second boolean flag. At the second ask, propose a more fully-engineered solution: a first-class `EvaluationResult` dataclass (metrics + optional predictions + optional extras) returned unconditionally, or a typed `evaluate_v2` alongside the deprecated boolean-flag form. The tolerance here trades one bit of API surface for zero new machinery at Stage 6; the re-engineering trigger is explicit and tracked. This note is repeated verbatim in `src/bristol_ml/evaluation/CLAUDE.md` at T2 so that future implementers find it without needing to re-read this plan. | D8 needs per-fold errors; the harness is the natural producer. Extending in-place preserves all Stage 4 call sites (default stays metrics-only). The alternative — a parallel `evaluate_with_predictions` — proliferates API surface without gain *at Stage 6*. The third option (plot helper re-runs the rolling loop) duplicates Stage 3 logic inside Stage 6, which is worse. The debt note captures the human's correct observation that two flag-style extensions would be one too many. | Codebase map §1 harness contract in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); requirements doc §4 OQ-7 in [`06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md); Stage 4 plan D-6 (discriminated-union rationale) as the precedent for promoting a growing surface to a typed structure when the trigger fires. |
| **D10** | `NesoBenchmarkConfig.holdout_start/_end` consumer | **Wire up at Stage 6 via a new helper `benchmark_holdout_bar(candidates, neso, metrics, *, holdout_start, holdout_end) -> Figure`.** Uses the fixed-window `SplitterConfig(fixed_window=True, min_train_periods=..., test_len=<holdout window length>)` derived from the two `holdout_*` config fields. This is the fixed-window retrospective plot anticipated by the Stage 4 retro. | Stage 4 retro deferred this; Stage 5 plan H-1 deferred it to Stage 6; the evaluation layer doc explicitly names Stage 6 as the natural consumer. Closing the loop here prevents the field from staying latent indefinitely. The helper is a thin orchestration over `compare_on_holdout` + a bar plot — low new surface area. | Codebase map §5 open-questions + §7 integration-points in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); [Stage 4 retro §Deferred](../../lld/stages/04-linear-baseline.md); [Stage 5 plan H-1](../completed/05-calendar-features.md). |
| **D11** | Notebook target (**amended: addendum not refactor**) | **Append new "Enhanced diagnostics (Stage 6)" section to `notebooks/04_linear_baseline.ipynb`. Leave Cells 0–12 exactly as Stage 4 shipped them — the inline matplotlib in Cell 11 stays.** *(Human mandate 2026-04-20: AC-5 reads as "a small amount of code per analysis" — each helper call is one-to-three lines — not "the total notebook should shrink". The Stage 4 notebook's existing Cell 11 is part of the Stage 4 pedagogical artefact; the Stage 6 value is the *new* diagnostic capability appended alongside, not the removal of the old.)* Append a new markdown cell with a "Stage 6 — Enhanced diagnostics" section header, followed by: (a) a code cell with the four hero plots via helper calls (`residuals_vs_time`, `predicted_vs_actual`, `acf_residuals`, `error_heatmap_hour_weekday`) in a 2×2 grid; (b) a code cell with the `forecast_overlay` helper call (so the same 48-h window is shown via the thin-notebook pattern — illustrative rather than load-bearing, since Cell 11 already shows the same view via inline matplotlib); (c) a code cell with the uncertainty-band demonstration using `evaluate(..., return_predictions=True)` + `forecast_overlay_with_band`; (d) a code cell with `benchmark_holdout_bar` if the NESO cache is warm (skip-markdown path otherwise); (e) a closing markdown cell on what each plot adds and Stage 7/8 hooks. Notebook 05 is not modified. | Intent §Scope item 2 is "An update to the Stage 4 notebook to use the richer diagnostics" — "update" admits addendum. AC-5 reads naturally as per-analysis concision ("each analysis demonstrates with small code") rather than total-file concision. Keeping Cell 11 preserves the Stage 4 artefact verbatim (important for the documentary / retrospective-reading value) and avoids destructive-edit risk on a notebook with committed cell outputs. The appendix cells prove the helpers' ergonomic claim by showing the same kinds of plots produced from three-line function calls. | Intent §Scope; requirements doc AC-5 re-interpretation in [`06-enhanced-evaluation-requirements.md`](../../lld/research/06-enhanced-evaluation-requirements.md); codebase map §4 Stage 4 cell structure + §8 convention on committed cell outputs in [`06-enhanced-evaluation-codebase.md`](../../lld/research/06-enhanced-evaluation-codebase.md); human markup. |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `NesoBenchmarkConfig.holdout_start/_end` (from Stage 4 retro, deferred via Stage 5 plan H-1). | **Resolved by D10 above** — Stage 6 adds the consumer. |
| **H-2** | `docs/intent/DESIGN.md §6` layout tree — Stages 1–5 additions missing (deny-tier for the lead). Stage 6 will add `evaluation/plots.py`, `tests/unit/evaluation/test_plots.py`, optionally `conf/evaluation/plots.yaml`. | **Flag for human-led batched §6 edit** covering Stages 1–6 at Stage 6 PR review. Lead MUST NOT touch §6 unilaterally. |
| **H-3** | Stage 5 retrospective `docs/lld/stages/05-calendar-features.md` "Next" section wrongly says "Tree-based models" (copy-paste residue). Intent `docs/intent/06-enhanced-evaluation.md` is unambiguously "Enhanced evaluation & visualisation". | **Correct during Stage 6 T7 hygiene** — one-line edit to Stage 5 retro's "Next" cell to read "Enhanced evaluation & visualisation (diagnostic-plot helper library appended to Stage 4 notebook)". Per spec-drift rule, intent wins; fix the retro. |
| **H-4** | `docs/architecture/layers/evaluation.md` module inventory needs `plots.py` row appended when shipped. | **Close at Stage 6 T7.** |
| **H-5** | PACF is a cheap optional extension over ACF (external research §R4). | **Defer** — ACF with `lags=168` covers the weekly-ripple teaching moment; PACF's differential value is SARIMAX order-selection, which is Stage 7's concern. Document in Stage 6 retro's "Deferred" section as a Stage 7 nice-to-have. |

### Resolution log (2026-04-20)

- **D1, D2, D3, D4, D8, D10** — accepted verbatim.
- **D5 amended** — default figsize `(12.0, 8.0)`, not `(10.0, 6.0)`; overridable via `PlotsConfig.figsize` in `conf/evaluation/plots.yaml` (human mandate: projector-friendly default, Hydra-configurable).
- **D6 amended with verification gate** — keep `display_tz="Europe/London"` default conditional on a DST-rendering test (spring-forward + fall-back Sundays); fall back to UTC-only across all six helpers if the test fails. T3 wires the test.
- **D7 reinforced** — explicit daily (lag 24) AND weekly (lag 168) reference markers on the ACF plot. `lags=168` default (covers both). Human note: "most periodicity is daily, and this should be reflected in a regression" — the ACF marker annotation makes that legibility explicit rather than leaving it to the reader.
- **D9 accepted one-off with architectural-debt note** — the `return_predictions: bool` flag is a single concession. At the second ask for a harness-output extension, propose a first-class `EvaluationResult` dataclass or `evaluate_v2`. Note repeated in `evaluation/CLAUDE.md` at T2.
- **D11 overridden** — *appendix* not refactor. Stage 4 Cells 0–12 (including the inline-matplotlib Cell 11) are preserved verbatim; new Stage-6-diagnostics cells are appended. AC-5 reads "a small amount of code per analysis", not "total notebook shrinks". Preserves the Stage 4 pedagogical artefact.
- **All housekeeping items (H-1 through H-5)** accepted as tabled.

Load-bearing propagation notes:
- D5 change propagates to `PlotsConfig.figsize` default and the module-level `plt.rcParams["figure.figsize"]` assignment in T2; `test_plots_config_figsize_default_is_twelve_by_eight` pins it.
- D6 gate propagates to T3 test list: `test_residuals_vs_time_dst_spring_forward_produces_gap`, `test_residuals_vs_time_dst_fall_back_produces_duplicate_hour`, `test_error_heatmap_dst_hour_groupby_behaviour`. If any of these fails at implementation time, T3 includes the UTC-fallback branch.
- D7 reinforcement propagates to T3: the `_annotate_acf_markers(ax)` helper is a named internal, tested to produce exactly two `axvline` artifacts at x=24 and x=168, each with a text label.
- D9 one-off tolerance propagates to T2's `evaluation/CLAUDE.md` update (the "API growth trigger" note) and to T5's `evaluate()` docstring.
- D11 override propagates to T6: Cell 11 untouched; new cells 13–17 appended; `jupyter nbconvert --execute` verifies the appendix runs alongside the existing content without disturbing it.

---

## 2. Scope

### In scope

- A new `src/bristol_ml/evaluation/plots.py` module with six public helper functions (D6), Okabe-Ito palette injection at import time (D2), `_cli_main()` entry-point for `python -m bristol_ml.evaluation.plots --help`, and module docstring + CLAUDE.md update.
- Extension to `src/bristol_ml/evaluation/harness.py`: add `return_predictions: bool = False` parameter (D9) so plot helpers have a principled source for per-fold errors. Default behaviour unchanged — all Stage 4 call sites continue to work.
- Promotion of `matplotlib>=3.8,<4` and `seaborn>=0.13,<1` to `[project].dependencies` in `pyproject.toml` (D3). `uv.lock` regenerated.
- Optional Pydantic config `PlotsConfig` in `conf/_schemas.py` and Hydra group `conf/evaluation/plots.yaml` for user-tweakable knobs (figsize, DPI, display timezone, ACF default lags). `EvaluationGroup` extended with `plots: PlotsConfig` (non-optional; `# @package evaluation.plots`). The helpers have hardcoded defaults that match the config defaults so construction without Hydra still works.
- In-place refactor of `notebooks/04_linear_baseline.ipynb` Cell 11 (D11) — replace inline matplotlib with helper calls; verify notebook still runs end-to-end under 120 s.
- Extension to `src/bristol_ml/evaluation/__init__.py` `__all__` and lazy `__getattr__` surface with the new public symbols.
- Unit tests: `tests/unit/evaluation/test_plots.py` covering determinism, figure-return-type, no-I/O guard, palette injection, ACF lag override, model-agnosticism (helper accepts `pd.Series` residuals, never a `LinearModel` object directly), heatmap shape, uncertainty-band quantile correctness.
- Extension to `tests/unit/evaluation/test_harness.py` with the predictions-emission path.
- Stage-hygiene updates: `CHANGELOG.md`, `docs/lld/stages/06-enhanced-evaluation.md` retrospective (new file), `docs/stages/README.md` row flip to `shipped`, `docs/architecture/layers/evaluation.md` module inventory, `src/bristol_ml/evaluation/CLAUDE.md` surface documentation, correction to `docs/lld/stages/05-calendar-features.md` "Next" cell (H-3).

### Out of scope (do not accidentally implement)

From intent §Out of scope, explicit:
- New models.
- Changes to metric definitions (MAE/MAPE/RMSE/WAPE definitions are fixed at Stage 4 and remain the comparison basis).
- Interactive dashboards, web UIs, anything beyond inline-notebook rendering.
- Probabilistic forecast visualisation of parameterised distributions (the empirical-quantile band D8 is NOT a probabilistic forecast in the proper sense — it's an empirical characterisation of observed errors).
- Model-explainability tooling (SHAP, partial dependence).

Also out of scope for this plan:
- Per-horizon diagnostics (intent §Points for consideration explicitly considers deferring; research A3 confirms defer). Will be revisited when Stage 10 or 11 ships a multi-horizon model.
- Per-weather-regime error breakdown (external research §R5 — domain-specific, defers cleanly; cheap Stage 16+ follow-up).
- Holiday-proximity error breakdown (§R5 — five-minute addition but adds narrative complexity not load-bearing for Stage 6).
- PACF (H-5 above).
- Residual-histogram with Normal overlay — mentioned in research §R3 as Hyndman's fourth panel but not load-bearing for our linear baseline (residuals aren't expected to be Gaussian) and would crowd the hero layout. Can be added trivially later.
- A new notebook (AC-5, D11) — Stage 4 notebook is extended by appendix cells; a standalone Stage 6 notebook is out of scope.
- In-place editing of Stage 4 notebook Cells 0–12 (D11 amendment) — the existing inline-matplotlib Cell 11 is preserved verbatim. Only the appended Stage 6 diagnostic cells are new.
- Changes to `cli.py`, `__main__.py`, or `load_config()` signature.
- Changes to splitter, metrics, or benchmarks modules beyond what D10's `benchmark_holdout_bar` composes (i.e., no internal refactor of `compare_on_holdout`).
- Any registry integration (Stage 9 concern).
- Any drift-monitoring primitive (Stage 18 concern).

---

## 3. Reading order for the implementer

Read top-to-bottom before opening code:

1. `docs/intent/06-enhanced-evaluation.md` — the spec. Where this plan disagrees, the spec wins.
2. `docs/lld/research/06-enhanced-evaluation-requirements.md` — acceptance criteria, open questions, tension analysis.
3. `docs/lld/research/06-enhanced-evaluation-codebase.md` — §1 (evaluation-layer surface; harness return-type contract), §2 (existing notebook plotting idioms), §3 (dependency graph — matplotlib dev-only), §4 (Stage 4 notebook Cell 11 target), §5 (evaluation layer contract + open questions), §6 (Model protocol — LinearModel.results vs NaiveModel), §7 (integration points), §9 (test conventions).
4. `docs/lld/research/06-enhanced-evaluation-domain.md` — §R1 (library choice), §R2 (palettes), §R3 (residual diagnostic idioms), §R4 (ACF conventions, lags=168 mandate), §R5 (hour×weekday heatmap), §R6 (empirical-quantile uncertainty), §7 clear-adopts.
5. `docs/intent/DESIGN.md` §2.1 (principles), §3.2 (layer responsibilities — evaluation paragraph), §5.1 (rolling-origin evaluator), §9 Stage 6 row.
6. `docs/architecture/layers/evaluation.md` — full contract; note Stage 6 is explicitly named as the `holdout_start/_end` consumer.
7. `docs/plans/completed/05-calendar-features.md` — for the D-numbered decision idiom, the task-commit-per-task discipline, the stage-hygiene checklist shape.
8. `src/bristol_ml/evaluation/harness.py` — `evaluate()` signature; where `return_predictions` is added.
9. `src/bristol_ml/evaluation/benchmarks.py` — `compare_on_holdout()` — the orchestration seam for D10's `benchmark_holdout_bar`.
10. `src/bristol_ml/models/protocol.py` + `linear.py` + `naive.py` — understand why helpers take residual `pd.Series`, not `Model` objects directly.
11. `notebooks/04_linear_baseline.ipynb` Cells 0–12 — the Stage 4 artefact; Cell 11's inline matplotlib is the existing data-shape reference and is preserved (per D11 amendment). New Stage 6 cells are appended after Cell 12.
12. `conf/_schemas.py` — the `ConfigDict(extra="forbid", frozen=True)` pattern; how `EvaluationGroup` composes sub-configs.
13. `tests/unit/evaluation/test_harness.py` — fixture patterns (`_make_df`, `loguru_caplog`, direct `SplitterConfig` construction) for the new test module.

CLAUDE.md + `.claude/playbook/` are read once for process, not per-stage.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

Mapped from `docs/lld/research/06-enhanced-evaluation-requirements.md`:

**Intent-quoted (AC-1..AC-4):**

1. **A model that conforms to the Stage 4 interface can produce every diagnostic with a small, consistent amount of code.** (AC-1; satisfied by D6 helper signatures all accepting simple pandas types, not `Model` objects; task T3/T4/T5.)
2. **The diagnostics are visually legible at meetup-audience distances.** (AC-2; satisfied by `plots.py` module rcParams defaults — >=12 pt axis labels, >=14 pt titles, **`figsize=(12, 8)` default per D5 amendment**, `linewidth>=1.2`; task T2.)
3. **The helper library has no dependencies on any specific model implementation.** (AC-3; satisfied by D8 signature choice — helpers accept residuals/predictions as `pd.Series`, not `LinearModel`/`NaiveModel`; task T3.)
4. **The updated notebook runs top-to-bottom quickly on a laptop.** (AC-4; target under 120 s end-to-end; task T6 + CI smoke.)

**Derived (AC-5..AC-13):**

5. Stage 4 notebook extended by appendix cells demonstrating the helper surface; Cells 0–12 (including inline-matplotlib Cell 11) preserved verbatim; no new notebook (D11 amendment). Task T6.
6. Default palette is Okabe-Ito + cividis + RdBu_r, formally CVD-safe (D2). Task T2 (injection); T3 test.
7. ACF plot uses `lags=168` minimum so weekly spike is visible (D7). Task T3.
8. Per-hour and per-weekday breakdowns ship; per-horizon explicitly deferred (D6). Task T3.
9. Forecast overlay with optional empirical-quantile uncertainty band (D6, D8). Task T4.
10. `python -m bristol_ml.evaluation.plots --help` works on a non-dev install (D3). Tasks T1, T2.
11. CI green: `uv run pytest`, `uv run ruff check .`, `uv run pre-commit run --all-files`.
12. `src/bristol_ml/evaluation/CLAUDE.md` updated; `docs/architecture/layers/evaluation.md` module inventory updated. Task T7.
13. `CHANGELOG.md` `[Unreleased]` `### Added` bullet(s); `docs/lld/stages/06-enhanced-evaluation.md` retro filed; `docs/stages/README.md` status cell flipped. Task T7.

§6 repo-layout tree update is deny-tier for the lead; **H-2** captures the flag-for-human posture.

---

## 5. Architecture summary (no surprises)

Data flow — end-to-end for the refactored Stage 4 notebook Cell 11:

```
features: pd.DataFrame (from assembler.load_calendar, UTC index)
linear: LinearModel (fitted on full feature table)
naive: NaiveModel (fitted on full feature table)
linear_per_fold, naive_per_fold: pd.DataFrame (from evaluate())

# Optional: predictions for uncertainty band
linear_per_fold, linear_preds = evaluate(linear, features, splitter_cfg, metrics, return_predictions=True)

# Diagnostic plots
from bristol_ml.evaluation import plots

residuals = features["nd_mw"] - linear.predict(features)
fig1 = plots.residuals_vs_time(residuals)
fig2 = plots.predicted_vs_actual(features["nd_mw"], linear.predict(features))
fig3 = plots.acf_residuals(residuals, lags=168)
fig4 = plots.error_heatmap_hour_weekday(residuals)

# Forecast overlay (existing 48-h window idiom, now a helper)
fig5 = plots.forecast_overlay(
    actual=window["nd_mw"],
    predictions_by_name={"naive": naive.predict(window), "linear": linear.predict(window)},
)

# Uncertainty band (new)
fig6 = plots.forecast_overlay_with_band(
    actual=window["nd_mw"],
    point_prediction=linear.predict(window),
    per_fold_errors=linear_preds,
)
```

Public API surface (Stage 6 adds only these):

```python
# evaluation/plots.py
def residuals_vs_time(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Residuals over time",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

def predicted_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    title: str = "Predicted vs actual",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

def acf_residuals(
    residuals: pd.Series,
    *,
    lags: int = 168,
    alpha: float = 0.05,
    reference_lags: tuple[int, ...] = (24, 168),   # D7: daily + weekly markers
    title: str = "Residual autocorrelation (lag in hours)",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...
# Renders ACF over lags 0..168 by default. Annotates two labelled axvline
# markers at x=24 ("daily") and x=168 ("weekly") per D7, so the daily-then-weekly
# periodicity story is legible without squinting at the spikes.

def error_heatmap_hour_weekday(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Mean signed residual by hour × weekday",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

def forecast_overlay(
    actual: pd.Series,
    predictions_by_name: dict[str, pd.Series],
    *,
    display_tz: str = "Europe/London",
    title: str = "Forecast overlay",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

def forecast_overlay_with_band(
    actual: pd.Series,
    point_prediction: pd.Series,
    per_fold_errors: pd.DataFrame,
    *,
    quantiles: tuple[float, float] = (0.1, 0.9),
    display_tz: str = "Europe/London",
    title: str = "Forecast with empirical uncertainty band",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

def benchmark_holdout_bar(
    candidates: dict[str, Model],
    neso_forecast: pd.DataFrame,
    features: pd.DataFrame,
    metrics: Sequence[MetricFn],
    *,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...   # D10

# Palette constants (public; downstream notebooks can import)
OKABE_ITO: tuple[str, ...]
SEQUENTIAL_CMAP: str  # "cividis"
DIVERGING_CMAP: str  # "RdBu_r"

# Module-level side effect at import (idempotent; run once per process):
# plt.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)
# plt.rcParams["figure.figsize"] = (12.0, 8.0)   # D5 amendment (human mandate 2026-04-20)
# plt.rcParams["figure.dpi"] = 110
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["axes.titlesize"] = 14
# plt.rcParams["legend.fontsize"] = 11

# evaluation/harness.py (extension)
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
    return_predictions: bool = False,   # NEW
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: ...
```

No change to `cli.py`, `__main__.py`, `load_config()`, splitter, metrics, benchmarks internals, feature assembler, or the existing `LinearModel` / `NaiveModel`.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Runtime dependency promotion + optional Pydantic config
*(Unblocks T2–T7.)*

- [ ] Edit `pyproject.toml`: move `matplotlib>=3.8,<4` from `[dependency-groups].dev` to `[project].dependencies`. Add `seaborn>=0.13,<1` to `[project].dependencies`. `cycler` is a matplotlib sub-dependency; no explicit pin needed.
- [ ] Run `uv sync --group dev` to regenerate `uv.lock`.
- [ ] Add `PlotsConfig` to `conf/_schemas.py`:
  ```python
  class PlotsConfig(BaseModel):
      model_config = ConfigDict(extra="forbid", frozen=True)
      figsize: tuple[float, float] = (12.0, 8.0)   # D5 amendment (human mandate 2026-04-20)
      dpi: int = 110
      display_tz: str = "Europe/London"             # D6: gated by DST-rendering tests in T3
      acf_default_lags: int = 168                   # D7: renders daily (24) + weekly (168) markers
      # Palette kept as constants in plots.py, not config, because changing it
      # breaks colourblind-safety guarantees.
  ```
- [ ] Extend `EvaluationGroup` with `plots: PlotsConfig = Field(default_factory=PlotsConfig)` — non-optional; defaults populated by Pydantic if the Hydra file is absent.
- [ ] Create `conf/evaluation/plots.yaml` with `# @package evaluation.plots` and all `PlotsConfig` fields populated with defaults.
- [ ] Update `conf/config.yaml` `defaults:` list: add `- evaluation/plots@evaluation.plots`.
- **Acceptance:** contributes to AC-10, AC-11.
- **Tests (spec-derived):**
  - `test_plots_config_rejects_extra_keys` — `extra="forbid"` verification.
  - `test_plots_config_figsize_default_is_twelve_by_eight` — pins the D5 human-mandated default.
  - `test_plots_config_acf_default_lags_is_168` — pins the D7 default.
  - `test_plots_config_display_tz_default_is_europe_london` — pins the D6 default (conditional on D6 T3 gate passing).
  - `test_plots_config_defaults_match_plots_module` — matches the module-level constants in `plots.py` (pinned at T2).
  - `test_config_loads_plots_group` — `load_config()` yields populated `cfg.evaluation.plots`.
  - `test_config_figsize_overridable_via_hydra` — `load_config(overrides=["evaluation.plots.figsize=[16,10]"])` propagates to `cfg.evaluation.plots.figsize`.
- **Command:** `uv run pytest tests/unit/test_config.py -q && uv run python -c "import matplotlib, seaborn; print(matplotlib.__version__, seaborn.__version__)"`.

### Task T2 — `evaluation/plots.py` module scaffold
*(Depends on T1.)*

- [ ] Create `src/bristol_ml/evaluation/plots.py`:
  - Module docstring describing the six helpers, the palette policy, and the `_cli_main` entry point.
  - Module-level constants: `OKABE_ITO`, `SEQUENTIAL_CMAP = "cividis"`, `DIVERGING_CMAP = "RdBu_r"`.
  - Module-level side effects guarded by `def _apply_style()` called exactly once at import:
    - `plt.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)`.
    - `plt.rcParams["figure.figsize"] = (12.0, 8.0)` (D5 human mandate; matches `PlotsConfig.figsize` default).
    - `plt.rcParams["figure.dpi"] = 110` (matches `PlotsConfig.dpi` default).
    - `plt.rcParams["axes.labelsize"] = 12`.
    - `plt.rcParams["axes.titlesize"] = 14`.
    - `plt.rcParams["legend.fontsize"] = 11`.
  - Type aliases at the top (`FigureOrAxes = matplotlib.figure.Figure`, `OptionalAxes = matplotlib.axes.Axes | None`).
  - Internal helper `_ensure_axes(ax: OptionalAxes, **figkw) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]` — returns (new Figure, Axes) when `ax is None`, otherwise `(ax.figure, ax)`.
  - `_cli_main(argv: list[str] | None = None) -> int`: prints the helper signatures and a one-line description of each, plus the active palette; returns 0.
  - Note: the six helper bodies are implemented in T3/T4; T2 lays only the module scaffold + constants + `_cli_main` skeleton.
- [ ] Add `plots` to `src/bristol_ml/evaluation/__init__.py` lazy re-export (follow the existing `splitter`, `metrics`, `harness`, `benchmarks` pattern). Add `residuals_vs_time`, `predicted_vs_actual`, `acf_residuals`, `error_heatmap_hour_weekday`, `forecast_overlay`, `forecast_overlay_with_band`, `benchmark_holdout_bar`, `OKABE_ITO`, `SEQUENTIAL_CMAP`, `DIVERGING_CMAP` to `__all__`.
- [ ] Extend `src/bristol_ml/evaluation/CLAUDE.md` with a new "Plotting conventions" section: palette policy, `ax=` composability contract, CVD-safety opt-out idiom, British-English labels in docstrings.
- [ ] Extend `src/bristol_ml/evaluation/CLAUDE.md` with a new "Harness output — API growth trigger" section **per D9 architectural-debt note**. Verbatim text:
  > `evaluate(..., return_predictions: bool = False)` is a single-flag concession at Stage 6 for the uncertainty-band helper. **Do not add a second boolean flag to `evaluate()` for any future output extension.** If Stage 9 (registry) needs a run-id column, Stage 18 (drift) needs drift scores, a multi-horizon model needs per-horizon predictions with a different shape, or a probabilistic model needs quantiles — propose a first-class `EvaluationResult` dataclass (metrics + optional predictions + optional extras) returned unconditionally, or a typed `evaluate_v2` alongside a deprecation of the boolean form. The re-engineering trigger is the **second** ask; Stage 6 is the first.
- **Acceptance:** contributes to AC-1, AC-2, AC-6, AC-10, AC-11, AC-12.
- **Tests (spec-derived):**
  - `test_plots_cli_main_returns_zero` — `_cli_main([])` returns 0 and prints helper names.
  - `test_plots_module_has_okabe_ito_constant` — 8 hex colours, each `#RRGGBB`, matches Wong 2011 values.
  - `test_plots_rcparams_prop_cycle_is_okabe_ito` — after import, `plt.rcParams["axes.prop_cycle"]` cycles through `OKABE_ITO`.
  - `test_plots_rcparams_figsize_is_twelve_by_eight` — after import, `plt.rcParams["figure.figsize"] == [12.0, 8.0]` (D5).
  - `test_plots_all_exported_symbols_importable` — `from bristol_ml.evaluation.plots import *` produces the documented surface.
  - `test_plots_module_docstring_british_english` — docstring contains "colour" or "visualisation" (sanity check against drift).
  - `test_evaluation_claude_md_has_api_growth_trigger_note` — parses `src/bristol_ml/evaluation/CLAUDE.md` and asserts the "Harness output — API growth trigger" heading is present (pins D9 debt note).
- **Command:** `uv run pytest tests/unit/evaluation/test_plots.py -q && uv run python -m bristol_ml.evaluation.plots --help`.

### Task T3 — Four hero helpers (residuals_vs_time, predicted_vs_actual, acf_residuals, error_heatmap_hour_weekday)
*(Depends on T2.)*

- [ ] Implement `residuals_vs_time(residuals, *, display_tz="Europe/London", title=..., ax=None) -> Figure`:
  - Converts `residuals.index` to `display_tz` via `tz_convert` (raises if tz-naive).
  - Line plot, `linewidth=1.2`, Okabe-Ito first colour.
  - `axhline(0.0, color="black", linewidth=0.6, alpha=0.5)`.
  - `mdates` formatter: `MonthLocator(interval=2)` + `DateFormatter("%b %Y")` for series >=6 months; `DateFormatter("%d %b")` for shorter.
  - Axis labels: `f"Time ({display_tz})"` (i.e. "Time (Europe/London)" on default), "Residual (MW)". Grid `alpha=0.3`.
  - **D6 DST gate:** run `test_residuals_vs_time_dst_spring_forward_produces_gap` and `test_residuals_vs_time_dst_fall_back_produces_duplicate_hour` as part of this task's pytest run. If either fails, change the `display_tz` default across all six helpers and `PlotsConfig` to `"UTC"` and record the decision in the Stage 6 retro.
- [ ] Implement `predicted_vs_actual(y_true, y_pred, *, title=..., ax=None) -> Figure`:
  - Scatter with `alpha=0.15`, `s=4` (sample-cap at 20 000 rows via `np.random.default_rng(42)` for reproducibility; log the cap if hit).
  - 45° line from `min(y_true.min(), y_pred.min())` to `max(y_true.max(), y_pred.max())`.
  - Axis convention: x = predicted, y = actual (Gelman 2025, research §R3).
  - Axis labels: "Predicted demand (MW)", "Actual demand (MW)".
- [ ] Implement `acf_residuals(residuals, *, lags=168, alpha=0.05, reference_lags=(24, 168), title=..., ax=None) -> Figure`:
  - Wraps `statsmodels.graphics.tsaplots.plot_acf(residuals, lags=lags, alpha=alpha, ax=ax)`.
  - X-axis label: "Lag (hours)".
  - **D7 reinforcement:** annotate two labelled `axvline` markers — one at lag 24 labelled "daily (24)" and one at lag 168 labelled "weekly (168)", both with `linewidth=1.0, alpha=0.35, color=OKABE_ITO[5]` (blue). Label placement: upper-inside each line via `ax.text(x=lag, y=0.9*ax.get_ylim()[1], s=..., rotation=90, fontsize=10)`. The `reference_lags` parameter is overridable so a future caller can drop in e.g. `(24, 168, 336)` for a two-week view; `()` disables annotation.
  - The helper's internal `_annotate_acf_markers(ax, reference_lags)` function draws the markers and is unit-tested directly.
- [ ] Implement `error_heatmap_hour_weekday(residuals, *, display_tz="Europe/London", title=..., ax=None) -> Figure`:
  - Derive `local_ts = residuals.index.tz_convert(display_tz)`; `hour = local_ts.hour`; `weekday = local_ts.dayofweek` (Monday=0).
  - Pivot table: index=weekday (0..6), columns=hour (0..23), values=residual mean.
  - `sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Mean signed residual (MW)"})`.
  - Y-axis tick labels: `["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]` (British English weekday abbreviations).
  - **D6 DST gate:** `test_error_heatmap_dst_hour_groupby_behaviour` exercises one DST transition in the synthetic residual frame — fall-back day has two observations for a given local hour (correct behaviour: mean of two); spring-forward day has zero observations for 01:00 local (correct behaviour: NaN cell rendered blank or with `annot="—"`). If groupby silently aggregates incorrectly, switch the helper's `display_tz` default to `"UTC"`.
- **Acceptance:** AC-1, AC-2, AC-3, AC-6, AC-7, AC-8, AC-11.
- **Tests (spec-derived):**
  - `test_residuals_vs_time_returns_figure` — type check; non-empty axes.
  - `test_residuals_vs_time_rejects_tz_naive` — raises `ValueError` on a tz-naive residual series.
  - `test_residuals_vs_time_ax_passthrough` — passing `ax=` returns the same figure, does not create a new one.
  - `test_predicted_vs_actual_axis_convention` — x-axis label contains "Predicted"; y-axis contains "Actual" (Gelman convention pinned).
  - `test_predicted_vs_actual_45_degree_line_present` — axes contains a line at y=x.
  - `test_acf_residuals_lags_168_default` — default `lags=168`; resulting x-axis extends to 168.
  - `test_acf_residuals_annotates_daily_and_weekly_markers` — axes contain two `axvline` artifacts at x=24 and x=168 (D7 reinforcement).
  - `test_acf_residuals_daily_marker_labelled` — text label "daily (24)" present near x=24.
  - `test_acf_residuals_weekly_marker_labelled` — text label "weekly (168)" present near x=168.
  - `test_acf_residuals_reference_lags_empty_disables_annotation` — `reference_lags=()` produces zero `axvline` artifacts (override path).
  - `test_acf_residuals_reference_lags_custom_respected` — `reference_lags=(24, 168, 336)` produces three markers.
  - `test_acf_residuals_override_lags_respected` — `lags=336` produces an x-axis to 336.
  - `test_error_heatmap_shape_24_by_7` — underlying pivot is 7 rows × 24 columns.
  - `test_error_heatmap_uses_diverging_cmap` — heatmap's cmap attribute is `"RdBu_r"`.
  - `test_error_heatmap_centered_at_zero` — heatmap's `center=0` (via mappable.norm properties).
  - `test_error_heatmap_weekday_labels_british` — y-tick labels are `["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]`.
  - `test_helpers_accept_residuals_series_not_model` — smoke test that helpers do not call any `Model` method; use `np.random` synthetic residuals.
  - `test_helpers_deterministic_on_fixed_seed` — two calls with same input produce byte-identical figure data (via `fig.canvas.tostring_rgb()`).
  - `test_helpers_no_io` — `unittest.mock` over `builtins.open`, `requests.get`, `Path.write_bytes` — none called (AC-3 via no external-resource dependencies).
  - **D6 DST gate tests (gating the Europe/London default):**
    - `test_residuals_vs_time_dst_spring_forward_produces_gap` — synthetic residual frame spanning last Sunday in March 2024 (31 March 2024 — UK spring-forward). Assert the local-time x-axis has no data point at 01:00–02:00 Europe/London, and mdates renders the gap without error.
    - `test_residuals_vs_time_dst_fall_back_produces_duplicate_hour` — synthetic residual frame spanning last Sunday in October 2024 (27 October 2024 — UK fall-back). Assert two UTC-distinct points map to the same Europe/London wall-clock hour, the line plot renders them as two separate points without matplotlib raising, and mdates labels them consistently.
    - `test_error_heatmap_dst_hour_groupby_behaviour` — synthetic residual frame covering a full October 2024. Assert `groupby(["dayofweek","hour"]).mean()` on the local-time index produces a 7×24 pivot without losing or duplicating data; the fall-back Sunday's 01:00 local cell reflects the mean of two UTC-distinct observations (not the sum, not a silent drop).
    - If any of the above three fails at implementation time, swap `display_tz="Europe/London"` → `display_tz="UTC"` as the default across all six helpers + `PlotsConfig.display_tz` in T1 + Stage 6 retro documentation entry.
- **Command:** `uv run pytest tests/unit/evaluation/test_plots.py -q`.

### Task T4 — Forecast overlay + uncertainty band (+ benchmark_holdout_bar)
*(Depends on T2, T3.)*

- [ ] Implement `forecast_overlay(actual, predictions_by_name, *, display_tz, title, ax) -> Figure`:
  - Line plot of `actual` + one line per entry in `predictions_by_name`, each taking successive Okabe-Ito colours.
  - Legend lower-right (matches existing notebook 04 Cell 11 convention).
  - `mdates.DateFormatter("%d %b\n%H:%M")` (matches notebook 04 Cell 11).
- [ ] Implement `forecast_overlay_with_band(actual, point_prediction, per_fold_errors, *, quantiles=(0.1, 0.9), ax) -> Figure`:
  - Requires `per_fold_errors` with column `horizon_h` (per D9).
  - Compute `q_lo, q_hi = per_fold_errors.groupby("horizon_h")["error"].quantile([0.1, 0.9]).unstack()`.
  - Map to timestamps in `actual.index` via positional join; shade with `ax.fill_between(x, point - q_hi, point - q_lo, alpha=0.25, color=OKABE_ITO[1])`.
  - Error-handling contract: if `per_fold_errors` lacks the `horizon_h` column, raise `ValueError("per_fold_errors must have 'horizon_h' column — run evaluate(..., return_predictions=True)")`.
- [ ] Implement `benchmark_holdout_bar(candidates, neso_forecast, features, metrics, *, holdout_start, holdout_end, ax) -> Figure` (D10):
  - Orchestration: derive a fixed-window `SplitterConfig(fixed_window=True, min_train_periods=<derived from features>, test_len=<(holdout_end - holdout_start) in hours>, step=<test_len>)`. Or: slice `features` to `features.loc[holdout_start:holdout_end]` and call `compare_on_holdout` with a pre-sliced frame; simpler.
  - Bar chart: x-axis = model name (including "neso"), y-axis = metric value, bar per metric.
  - Axis labels: "Model", metric name + " (MW)" for MAE/RMSE or "(fraction)" for MAPE/WAPE.
- **Acceptance:** AC-9, AC-11.
- **Tests (spec-derived):**
  - `test_forecast_overlay_plots_actual_plus_each_prediction` — axes contain N+1 line artifacts for N predictions.
  - `test_forecast_overlay_legend_lower_right` — legend anchor matches the notebook convention.
  - `test_forecast_overlay_with_band_requires_horizon_column` — `ValueError` when `per_fold_errors` lacks `horizon_h`.
  - `test_forecast_overlay_with_band_q10_q90_default` — `fill_between` bounds match 10th/90th quantile of synthetic per-fold errors.
  - `test_forecast_overlay_with_band_custom_quantiles_respected` — `quantiles=(0.25, 0.75)` narrows the band.
  - `test_benchmark_holdout_bar_returns_figure_with_metric_bars` — bar count == metric count × (candidate count + 1).
  - `test_benchmark_holdout_bar_uses_holdout_window_from_config` — slicing matches `cfg.evaluation.benchmark.holdout_start / .holdout_end`.
  - `test_benchmark_holdout_bar_model_agnostic` — accepts a dict whose values are any `Model` protocol implementers (unit test with a stub `Model`).
- **Command:** `uv run pytest tests/unit/evaluation/test_plots.py -q`.

### Task T5 — Harness predictions emission
*(Depends on T1; independent of T2–T4 but needed before T6's notebook refactor uses `return_predictions=True`.)*

- [ ] Extend `src/bristol_ml/evaluation/harness.py::evaluate`:
  - Add keyword-only `return_predictions: bool = False`.
  - When `True`, accumulate a list of per-fold prediction records:
    ```python
    for fold_index, (train_idx, test_idx) in enumerate(...):
        ...
        y_pred = model.predict(df.iloc[test_idx][feature_columns])
        if return_predictions:
            fold_preds = pd.DataFrame({
                "fold_index": fold_index,
                "test_start": df.iloc[test_idx[0]]["timestamp_utc"],  # or index
                "test_end": df.iloc[test_idx[-1]]["timestamp_utc"],
                "horizon_h": np.arange(len(test_idx)),
                "y_true": y_true.values,
                "y_pred": y_pred.values,
                "error": (y_true - y_pred).values,
            })
            predictions_frames.append(fold_preds)
    ```
  - Return type: `pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]` depending on the flag.
  - Column-order contract for the predictions frame: `["fold_index", "test_start", "test_end", "horizon_h", "y_true", "y_pred", "error"]`.
  - dtypes: `fold_index: int64`, `test_start/test_end: datetime64[ns, UTC]` (inherits input tz), `horizon_h: int64`, `y_true/y_pred/error: float64`.
- [ ] Extend `docstring` on `evaluate()` to describe the new parameter and return-shape.
- **Acceptance:** AC-9 (uncertainty band via empirical errors), AC-11.
- **Tests (spec + regression):**
  - `test_harness_evaluate_default_returns_metrics_only` — regression; Stage 4 call sites unaffected.
  - `test_harness_evaluate_return_predictions_returns_tuple` — `return_predictions=True` returns a 2-tuple.
  - `test_harness_predictions_column_order_and_dtypes` — exact order + dtypes pinned.
  - `test_harness_predictions_one_row_per_forecast_hour` — total row count == sum of `test_len` across folds.
  - `test_harness_predictions_horizon_h_zero_based_per_fold` — horizon_h resets per fold, starts at 0.
  - `test_harness_predictions_error_equals_y_true_minus_y_pred` — sanity.
- **Command:** `uv run pytest tests/unit/evaluation/test_harness.py -q`.

### Task T6 — Stage 4 notebook addendum (append-only; D11 amendment)
*(Depends on T1–T5.)*

**Per D11 human mandate 2026-04-20: do NOT modify Stage 4 Cells 0–12. The inline-matplotlib Cell 11 stays exactly as shipped at Stage 4 T9 (commit `30c94df`). The Stage 6 value is demonstrated by appending new cells after Cell 12.**

- [ ] Append to `notebooks/04_linear_baseline.ipynb` (new cells numbered starting at index 13):
  - **Cell 13 (md, new):** section header markdown. Title: "Stage 6 — Enhanced diagnostics". Prose: explain that Stage 4's Cell 11 already shows residual-vs-time and a 48-h overlay as inline matplotlib; the cells below demonstrate the same views plus three additional diagnostics via the `bristol_ml.evaluation.plots` helper library, so each analysis collapses to a one-to-three-line call (intent AC-5 reading). Explicit Stage 7 hook: "look for the spike at lag 168 — that's a full week, and it's what Stage 7's SARIMAX is built for".
  - **Cell 14 (code, new):** four hero residual diagnostics in a 2×2 grid:
    ```python
    from bristol_ml.evaluation import plots

    residuals = features["nd_mw"] - linear.predict(features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots.residuals_vs_time(residuals, ax=axes[0, 0])
    plots.predicted_vs_actual(features["nd_mw"], linear.predict(features), ax=axes[0, 1])
    plots.acf_residuals(residuals, lags=168, ax=axes[1, 0])
    plots.error_heatmap_hour_weekday(residuals, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()
    ```
  - **Cell 15 (code, new):** forecast-overlay via helper (illustrative: Cell 11 already shows this inline; the purpose here is proving the helper produces the same view from three lines):
    ```python
    # WINDOW_START / WINDOW_END already defined in Cell 11 — reuse them
    window = features.loc[WINDOW_START:WINDOW_END]
    fig = plots.forecast_overlay(
        actual=window["nd_mw"],
        predictions_by_name={
            "naive": naive.predict(window),
            "linear": linear.predict(window),
            **({"NESO": neso_slice["demand_forecast_mw"]} if forecast_cache_warm else {}),
        },
    )
    plt.show()
    ```
  - **Cell 16 (code, new):** uncertainty-band demonstration using `return_predictions=True`:
    ```python
    _, linear_preds = evaluate(
        LinearModel(linear_cfg), features, splitter_cfg, metrics, return_predictions=True
    )
    fig = plots.forecast_overlay_with_band(
        actual=window["nd_mw"],
        point_prediction=linear.predict(window),
        per_fold_errors=linear_preds,
    )
    plt.show()
    ```
  - **Cell 17 (code, new):** `benchmark_holdout_bar` if NESO cache is warm (skip-markdown path otherwise):
    ```python
    if forecast_cache_warm:
        fig = plots.benchmark_holdout_bar(
            candidates={"linear": linear, "naive": naive},
            neso_forecast=neso_df_local,
            features=features,
            metrics=[METRIC_REGISTRY[m] for m in cfg.evaluation.metrics.names],
            holdout_start=cfg.evaluation.benchmark.holdout_start,
            holdout_end=cfg.evaluation.benchmark.holdout_end,
        )
        plt.show()
    else:
        print("NESO forecast cache cold — skipping benchmark_holdout_bar demo.")
    ```
  - **Cell 18 (md, new closing):** short narrative — what each new plot adds on top of Cell 11's existing view; explicit Stage 7 / Stage 8 hooks. Close with a one-sentence explanation that the empirical-quantile band is NOT a probabilistic prediction interval but an empirical characterisation of rolling-origin errors.
  - Reuse Stage 4's `min_train_periods=720, step=168` config override already set in Cell 1 — do not widen the fold count (notebook still needs to run under 120 s end-to-end, appendix included).
- [ ] **Cells 0–12 are untouched.** The diff on this task should show *additions* only; no modifications to Stage 4's committed cells. Verify with `git diff --stat notebooks/04_linear_baseline.ipynb` before commit — additions only.
- [ ] Run the notebook end-to-end; verify all cells (old and new) render; commit with outputs present (matches Stage 4/5 convention).
- [ ] Do NOT modify `notebooks/05_calendar_features.ipynb`. Its weekly-ripple comparison is Stage 5's narrative moment and is out of scope for Stage 6.
- **Acceptance:** AC-1, AC-2, AC-4, AC-5 (per the D11 amendment reading: small code *per analysis*, not total), AC-11.
- **Tests:** no notebook-execution test in CI (codebase map §9). Manual verification via `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04_linear_baseline.ipynb` before commit.
- **Command:** `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04_linear_baseline.ipynb && uv run ruff check notebooks/ && git diff --stat notebooks/04_linear_baseline.ipynb`.

### Task T7 — Stage hygiene
*(Depends on T1–T6.)*

- [ ] `CHANGELOG.md` — under `[Unreleased]`, add:
  - **Added**
    - `bristol_ml.evaluation.plots` — four residual-diagnostic helpers, forecast overlay, empirical-quantile uncertainty band, and `benchmark_holdout_bar` for fixed-window NESO retrospective comparison.
    - `matplotlib` and `seaborn` promoted to runtime dependencies (D3).
    - `PlotsConfig` and `conf/evaluation/plots.yaml` Hydra group.
    - `evaluate(..., return_predictions=True)` option on the rolling-origin harness (D9).
    - Okabe-Ito CVD-safe palette as the Stage 6 default; `cividis` sequential; `RdBu_r` diverging.
  - **Changed**
    - `notebooks/04_linear_baseline.ipynb` — appended Stage 6 diagnostic-plot appendix (cells 13–18); Stage 4 Cells 0–12 preserved verbatim.
    - Default matplotlib figsize for `bristol_ml.evaluation.plots` helpers is 12×8 (meetup-projector legibility); overridable via `PlotsConfig.figsize`.
- [ ] Create `docs/lld/stages/06-enhanced-evaluation.md` — retrospective following the template in `docs/lld/stages/00-foundation.md`:
  - **What was built** — one paragraph per T1..T7 artefact.
  - **Design choices made here** — recap each `D1..D11` decision with its final resolution (post-human review), a one-line rationale, and a link back to the plan and research docs.
  - **Demo moment** — paste the six-helper command sequence.
  - **Deferred** — PACF (H-5), per-horizon diagnostics, weather-regime breakdown, holiday-proximity breakdown, residual-histogram + Normal overlay, §6 DESIGN.md batched edit (H-2).
  - **Next** → Stage 7 SARIMAX (correctly named this time — use the intent's stage title verbatim for whichever stage comes next in the pipeline).
- [ ] `docs/stages/README.md` — flip Stage 6 row to `shipped` with links:
  ```
  | 6 | Enhanced evaluation & viz | `shipped` | [intent](../intent/06-enhanced-evaluation.md) | [plan](../plans/completed/06-enhanced-evaluation.md) | [evaluation](../architecture/layers/evaluation.md) | — | [retro](../lld/stages/06-enhanced-evaluation.md) |
  ```
- [ ] `docs/architecture/layers/evaluation.md` — module inventory row: add `evaluation/plots.py::residuals_vs_time`, `::predicted_vs_actual`, `::acf_residuals`, `::error_heatmap_hour_weekday`, `::forecast_overlay`, `::forecast_overlay_with_band`, `::benchmark_holdout_bar`. Mark the two Stage 4 open questions resolved:
  - "`NesoBenchmarkConfig.holdout_start/_end` consumer" → resolved by `plots.benchmark_holdout_bar` (Stage 6 D10).
  - "Stage 6 will add visualisation and richer-diagnostics primitives" → shipped.
- [ ] **H-3 fix**: edit `docs/lld/stages/05-calendar-features.md` "Next" cell from "Stage 6 tree-based models" (or similar) to "Stage 6 enhanced evaluation & visualisation (diagnostic-plot helper library; Stage 4 notebook appendix)". One-line edit; ALLOW-tier file.
- [ ] **H-2 flag**: raise the `docs/intent/DESIGN.md §6` batched-edit request to the human at PR review (Stages 1–6 additions: `ingestion/holidays.py`, `features/calendar.py`, `conf/features/weather_calendar.yaml`, `conf/ingestion/holidays.yaml`, `notebooks/05_calendar_features.ipynb`, `evaluation/plots.py`, `conf/evaluation/plots.yaml`, Stage 4 notebook appendix). Lead MUST NOT edit §6.
- [ ] `docs/architecture/ROADMAP.md` — if the Evaluation section has any remaining open questions closed by Stage 6, mark them with back-references; otherwise no-op.
- [ ] Move `docs/plans/active/06-enhanced-evaluation.md` → `docs/plans/completed/06-enhanced-evaluation.md` via `git mv` as the final commit action.
- **Acceptance:** AC-11, AC-12, AC-13.
- **Tests:** `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- **Command:** `uv run python -m bristol_ml --help && uv run python -m bristol_ml.evaluation.plots --help && uv run pytest -q`.

---

## 7. Files expected to change

### New
- `src/bristol_ml/evaluation/plots.py`
- `conf/evaluation/plots.yaml`
- `tests/unit/evaluation/test_plots.py`
- `docs/lld/stages/06-enhanced-evaluation.md`

### Modified
- `pyproject.toml` (matplotlib + seaborn to runtime deps)
- `uv.lock` (regenerated)
- `conf/_schemas.py` (`PlotsConfig` + `EvaluationGroup.plots`)
- `conf/config.yaml` (defaults list adds `- evaluation/plots@evaluation.plots`)
- `src/bristol_ml/evaluation/harness.py` (`return_predictions` parameter)
- `src/bristol_ml/evaluation/__init__.py` (re-exports)
- `src/bristol_ml/evaluation/CLAUDE.md` (plotting conventions section)
- `tests/unit/evaluation/test_harness.py` (predictions-path coverage)
- `notebooks/04_linear_baseline.ipynb` (**append Cells 13–18; do not modify Cells 0–12** per D11 amendment)
- `CHANGELOG.md`
- `README.md` (Stage 6 entry point paragraph; `python -m bristol_ml.evaluation.plots --help` mention)
- `docs/stages/README.md` (Stage 6 row flip + links)
- `docs/architecture/layers/evaluation.md` (module inventory + open-question close-outs)
- `docs/lld/stages/05-calendar-features.md` (H-3 "Next" fix — one line)
- `docs/architecture/ROADMAP.md` (back-references if any remain)

### Moved (final commit)
- `docs/plans/active/06-enhanced-evaluation.md` → `docs/plans/completed/06-enhanced-evaluation.md`

### Explicitly NOT modified
- `docs/intent/DESIGN.md §6` (deny-tier; flag for human — H-2)
- `docs/intent/06-enhanced-evaluation.md` (immutable once shipped)
- `notebooks/05_calendar_features.ipynb` (Stage 5's narrative; leave untouched)
- `src/bristol_ml/models/*` (no model changes — intent out of scope)
- `src/bristol_ml/evaluation/metrics.py` (metrics fixed at Stage 4)
- `src/bristol_ml/evaluation/splitter.py` (no splitter changes)
- `src/bristol_ml/evaluation/benchmarks.py` internals (D10 composes, doesn't modify)
- `src/bristol_ml/features/*`, `src/bristol_ml/ingestion/*` (features/ingestion unchanged)
- `src/bristol_ml/cli.py`, `src/bristol_ml/__main__.py`, `src/bristol_ml/config.py` (no CLI boundary changes)

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Promoting matplotlib to runtime breaks someone's minimal headless install | Low | Medium | matplotlib is a mature, widely-installed package with no GUI dependencies by default (Agg backend). Pins match dev-group. Note in CHANGELOG + README. |
| Okabe-Ito injection conflicts with a user's existing matplotlib style | Low | Low | Side effect runs once at module import; opt-out is one `plt.rcParams.update(...)` call. Documented in `CLAUDE.md` plotting conventions section. |
| `return_predictions=True` doubles evaluation memory footprint | Low | Medium | Optional, default False — Stage 4 call sites unaffected. The notebook's single usage is on the `step=168` narrow-fold-count config, memory cost is ~30 folds × 168 hours × 7 float64 columns ≈ 280 KB. |
| `plot_acf(lags=168)` slow on long residual series | Low | Low | 8760-point series: <100 ms wall-time measured in statsmodels benchmarks. Under the 2 s per-helper NFR. |
| Notebook appendix accidentally modifies Stage 4 Cells 0–12 | Low | Medium | D11 amendment; `git diff --stat` check before commit confirms additions-only. `uv run jupyter nbconvert --execute --to notebook --inplace` catches runtime breakage. |
| D6 DST rendering fails at implementation time, forcing UTC fallback | Low | Low | T3 includes three gating tests (spring-forward gap, fall-back duplicate hour, heatmap groupby behaviour). If any fail, swap default to UTC; costs meetup intuition but keeps semantics correct. Documented fallback path in D6 + Stage 6 retro "Design choices" section. |
| Empirical-quantile band semantics confuse audience ("is this a prediction interval?") | Medium | Low | Cell 14 markdown explicitly labels the band as "empirical error quantiles from rolling-origin evaluation, not a probabilistic prediction interval". Stage 6 retro flags the distinction. |
| `benchmark_holdout_bar` surfaces a hidden bug in the latent `holdout_start/_end` config fields (never exercised since Stage 4) | Low | Medium | First exercise of the fields. Unit test constructs a synthetic holdout window and asserts the produced chart covers it correctly; integration test with the real NESO benchmark cache optional (skip cleanly if absent, like notebooks 04/05 do). |
| `seaborn.heatmap` version drift changes default annotation behaviour | Low | Low | Pin `seaborn>=0.13,<1`. Test asserts heatmap cmap and centre explicitly rather than relying on defaults. |

---

## 9. Exit checklist

Maps to DESIGN §9 definition-of-done.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail`.
- [ ] Ruff clean: `uv run ruff check . && uv run ruff format --check .`.
- [ ] Pre-commit clean: `uv run pre-commit run --all-files`.
- [ ] `python -m bristol_ml --help` works.
- [ ] `python -m bristol_ml.evaluation.plots --help` works (AC-10).
- [ ] `notebooks/04_linear_baseline.ipynb` runs end-to-end under 120 s (AC-4).
- [ ] `CHANGELOG.md` `[Unreleased]` has the Stage 6 Added/Changed bullets.
- [ ] `docs/lld/stages/06-enhanced-evaluation.md` retrospective exists and is complete.
- [ ] `docs/stages/README.md` Stage 6 row is `shipped`.
- [ ] `docs/architecture/layers/evaluation.md` module inventory updated.
- [ ] `src/bristol_ml/evaluation/CLAUDE.md` has the new plotting conventions section.
- [ ] `docs/lld/stages/05-calendar-features.md` "Next" cell corrected (H-3).
- [ ] `docs/intent/DESIGN.md §6` batched-edit flagged to the human (H-2).
- [ ] `docs/plans/active/06-enhanced-evaluation.md` → `docs/plans/completed/06-enhanced-evaluation.md` moved via `git mv`.
- [ ] README Stage 6 entry-point paragraph present.
- [ ] No new `# type: ignore` without an inline justification.
- [ ] British English in all new docstrings, axis labels, and prose.
- [ ] No silent spec deviations. If any surface, surface-then-document.

---

## 10. Open items for human review — resolved 2026-04-20

Both items tabled at the unreviewed draft have been resolved by the human markup:

**OH-1 (was I-1 in requirements) — RESOLVED.** Stage 5 retro's "Next → Stage 6 tree-based models" line will be fixed during Stage 6 T7 hygiene (H-3). One-line edit; ALLOW-tier file.

**OH-2 (was I-2 in requirements; resolved as D3) — RESOLVED.** `matplotlib` + `seaborn` promoted to `[project].dependencies`. Lazy-import alternative rejected.

All D1–D11 decisions are resolved per §1 Resolution log (2026-04-20). Phase 2 is ready to begin at human's signal.
