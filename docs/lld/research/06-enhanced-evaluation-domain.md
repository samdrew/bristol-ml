# Stage 6 — Enhanced Evaluation & Visualisation: External Research

**Purpose.** Ground the plan's design decisions in current practice — viz-library trade-offs, colourblind-safe palettes, residual-diagnostic idioms for time-series, ACF/PACF conventions, and empirical-quantile uncertainty. Input to the plan. Not the plan itself.

**Scope.** Six research questions, tracked R1..R6 below, mapped back to the plan's open questions OQ-1 through OQ-9.

---

## R1 — Viz library: matplotlib, seaborn, plotly, or altair?

**Criteria.** (a) renders statically on github.com (a `*.ipynb` hosted on github.com must show the plots without a running kernel); (b) integrates with `statsmodels.graphics.tsaplots.plot_acf`; (c) near-zero added dependencies; (d) large-text / large-marker configuration is a one-liner; (e) typed API.

**matplotlib (>=3.8).** Already declared in `pyproject.toml` dependency-groups.dev. Native statsmodels target — `plot_acf` returns a matplotlib Figure. Static PNG output renders on github.com unconditionally. Typed stubs incomplete, but pandas + numpy API surface is universally known. Verdict: baseline. **Selected.**

**seaborn (0.13+).** Thin wrapper over matplotlib; same output pipeline. Adds ~295 kB wheel. Provides `heatmap`, `relplot`, palette helpers (`sns.set_palette("colorblind")`), and better default theming than bare matplotlib. Integrates seamlessly with `plot_acf` because it's still matplotlib underneath. Verdict: adopt alongside matplotlib for palette and heatmap support.

**plotly (5+).** Interactive. Static output on github.com requires `kaleido` + Chrome on the render machine — fragile and notably absent from Python-only CI. Static export of plotly figures works in JupyterHub but **does not render on github.com without exported PNG/SVG**. Verdict: reject.

**altair (5+).** Vega-Lite front-end. Notebook rendering requires a configured mime renderer; github.com does not display altair charts without an exported PNG. Verdict: reject.

**Decision.** **matplotlib + seaborn.** seaborn as a thin convenience layer; matplotlib for the ACF/PACF + fine-grained control. No plotly, no altair. Maps to OQ-1.

---

## R2 — Colourblind-safe palettes

**Constraints.** Mixed-audience meetup viewing, projector display, printed handouts. Must be safe under deuteranopia (1 in 12 men), protanopia, tritanopia. Must be perceptually uniform for sequential data (heatmaps, densities) and diverging for signed data (residuals).

**Okabe-Ito 8-colour qualitative palette (Wong 2011, *Nature Methods*).** The de-facto reference: black `#000000`, orange `#E69F00`, sky blue `#56B4E9`, bluish green `#009E73`, yellow `#F0E442`, blue `#0072B2`, vermillion `#D55E00`, reddish purple `#CC79A7`. Formally certified for all three dichromacy types. R's `viridisLite` package bundles it. Matplotlib has it as `mpl.colormaps["Okabe_Ito"]` only in 3.9+; for 3.8 the palette must be declared inline as an RGB list. Verdict: adopt as default qualitative palette.

**Seaborn `"colorblind"`.** Close to Okabe-Ito but not identical — uses `#0173b2, #de8f05, #029e73, #d55e00, #cc78bc, #ca9161, #fbafe4, #949494, #ece133, #56b4e9`. Safe but not formally Okabe-Ito. Acceptable if the palette is fetched from seaborn directly; less useful if committing the colour list.

**IBM accessibility palette (IBM design language).** Five colours `#648fff, #785ef0, #dc267f, #fe6100, #ffb000`. Tighter than Okabe-Ito, handy when only 3-4 series needed. Safe but less universally named.

**Sequential (heatmaps, densities).** `viridis` (default) or `cividis` (formally revised viridis for full CVD accessibility, Nunez et al. 2018). Both perceptually uniform and colourblind-safe. Verdict: `cividis` as the Stage 6 default because it's the more conservative choice.

**Diverging (signed residuals).** `RdBu_r` (reverse of RdBu) is colourblind-safe enough for red/blue dichromats because the hue axis is also value-mapped. Alternatives: `PuOr`, `BrBG`. Verdict: `RdBu_r`.

**`tab10` — explicitly reject.** Not certified colourblind-safe; the red/green pair is confusable under deuteranopia.

**Injection mechanism.** One-liner at plots.py import time:
```python
from cycler import cycler
OKABE_ITO = ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]
plt.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)
```
Opt-out: facilitator calls `plt.rcParams.update(...)` after the import. Verdict: **Okabe-Ito qualitative, cividis sequential, RdBu_r diverging; inject via rcParams at module import**. Maps to OQ-2.

---

## R3 — Residual-diagnostic idioms for point forecasters

Hyndman & Athanasopoulos *Forecasting: Principles and Practice (fpp3)* §5.3 is the canonical reference; Gelman's *Regression and Other Stories* (2021, 2025 repr.) is the regression-diagnostic complement.

**Canonical four-panel for time-series residuals (Hyndman fpp3 §5.3).** (1) residuals vs time; (2) histogram / density of residuals with a Normal overlay; (3) ACF of residuals; (4) predicted-vs-actual scatter (Gelman §11.3). These four are non-redundant and together answer the "is this model's error structure well-behaved?" question at a glance.

**Residual-vs-time.** X-axis = timestamp; y-axis = `y_true - y_pred`. Horizontal zero line. Reveals drift, periodicity, level shifts. For the linear baseline this is where the weekly ripple is visible — the motivation for Stage 5 (now shipped) and Stage 7.

**Residual-vs-fitted (Anscombe plot, homoscedasticity check).** X-axis = `y_pred`; y-axis = residuals. Reveals heteroscedasticity (fan shape), non-linearity (curvature). Less useful than residual-vs-time for seasonal series but diagnostically complete.

**Predicted-vs-actual scatter (Gelman 2025 axis convention).** X-axis = predicted, y-axis = actual. 45° reference line. Key check: are predictions centred on the line or systematically above/below? Good for catching scale bias. Two rival conventions exist — actual on X vs actual on Y — Gelman (2025) argues for actual-on-Y because "you're checking how well predictions explain actuals, not the reverse".

**Residual histogram.** With a Normal(0, sigma) overlay. Not load-bearing for a GEFCom-style linear baseline because residuals needn't be Gaussian; still a useful meetup talking point. Low priority for Stage 6.

**Qualitative recommendation.** Ship four functions: `residuals_vs_time`, `predicted_vs_actual`, `error_heatmap_hour_weekday` (see R5), and `acf_residuals` (see R4). The first three map to the intent's "residual plots", "calibration or reliability plots" (predicted-vs-actual is a calibration plot in Gelman's sense), and "error breakdowns by hour and weekday".

---

## R4 — ACF/PACF conventions

**statsmodels API.** `statsmodels.graphics.tsaplots.plot_acf(x, ax=None, lags=None, alpha=0.05, ...)` and `plot_pacf(x, ...)` are the standard. Return a matplotlib Figure (or plot onto an Axes if `ax=` provided). Bartlett 95% confidence bands are the default and are widely expected by readers.

**Default lag count trap.** `plot_acf`'s default `lags` is `min(int(10*log10(len(x))), len(x) // 2 - 1)`. For the Stage 4 residual series (hourly, ~24 * 365 * N_folds rows), that default gives ~39-50 lags — not enough to see the weekly spike at 168. **Mandatory override: `lags=168`** (one full week of hourly data). For the two-week view, `lags=336` is the secondary choice; `lags=168` is the demo-friendly default.

**PACF need.** The PACF disambiguates direct autocorrelation from chain-reaction autocorrelation and is the natural input to SARIMAX (Stage 7) order selection. Nice-to-have for Stage 6; the load-bearing plot is ACF. Verdict: ship ACF at Stage 6; PACF is a cheap extension.

**Sample-size warning.** `plot_acf` issues a warning when the sample is short; for our residual length (~8000-20000 points) the warning does not fire.

Maps to OQ-5, AC-7.

---

## R5 — Error breakdowns by hour / weekday / regime

**Standard idiom for electricity demand.** A 24 × 7 heatmap of mean signed error (y-axis: weekday; x-axis: hour-of-day; colour: mean residual, diverging palette RdBu_r). Richest single visualisation because it exposes the load-profile vs weekday interaction that a main-effects-only linear model cannot capture. Hong (GEFCom 2014 vanilla benchmark) uses a weaker 1-D hour plot; Amperon (2024) uses the 2-D heatmap.

**sktime / Nixtla / R `feasts`.** Surveyed for ready-made decomposition helpers. None provide a "residuals by (hour, weekday)" primitive out of the box. sktime's `performance_metrics.forecasting` is metric-only; Nixtla's `utilsforecast` is aggregation helpers; R `feasts` has `gg_season` for seasonal subseries plots but nothing bound to a residuals-by-two-calendar-features chart. Verdict: **implement the heatmap directly in Stage 6** — it is cheap and carries the pedagogical load.

**Per-hour 1-D fallback.** Box plot or violin plot of residuals per hour-of-day is the meetup-safe simplification — colleagues less familiar with heatmaps will read it faster. Ship both; the heatmap is the hero plot, the box-per-hour is the legibility safety-net.

**Weather-regime breakdown.** Bin temperature into cold / mild / hot (e.g. Q25, Q75) and plot residuals per bin. Domain-specific; cheap to build; consider for Stage 6. Verdict: defer — temperature-regime breakdown adds narrative complexity without changing the headline Stage 6 story (calendar + weather baseline handles most variance).

**Holiday-proximity breakdown.** Binary axis (holiday-adjacent vs not). Given Stage 5 ships the holiday flags, this is a five-minute addition and lands well in meetups. Verdict: optional Stage 6 stretch goal; document as cheap follow-up.

Maps to OQ-4, AC-8.

---

## R6 — Empirical uncertainty bands for point forecasters

**Rationale.** The Stage 4 `LinearModel` is a point forecaster (OLS predictions). Rolling-origin evaluation produces a *distribution* of per-day errors across folds. That distribution is a legitimate visualisation of forecast uncertainty — not a probabilistic forecast in the proper statistical sense, but a fair empirical characterisation.

**Three candidate methods.**

(a) **Empirical quantile band (q10-q90).** From the rolling-origin output, compute the 10th and 90th percentile of signed errors `(y_true - y_pred)` across folds for each forecast horizon h (h = 0..168, hours ahead). Shade the band around the forecast: `forecast - q10(err@h)` to `forecast - q90(err@h)`. Defensible, non-parametric, matches how Hyndman (fpp3 §5.5) describes empirical prediction intervals for point forecasters.

(b) **Mean ± 1.96 σ band.** Assume residuals are approximately Normal; take the per-horizon standard deviation; shade ±1.96 σ. Simpler than (a), approximately equivalent when errors are roughly symmetric. Gelman calls this the "Normal approximation"; his warning is that when errors are skewed the coverage is wrong. For electricity demand, residuals are broadly symmetric; either works.

(c) **Fitted-conditional variance (statsmodels WLS / prediction interval).** `statsmodels` provides `get_prediction(exog).conf_int(alpha=0.05)` for OLS. This is a *model-based* prediction interval (accounts for parameter-estimation uncertainty AND residual variance under the Normal assumption) — the cleanest theoretically, but tightly couples the plots module to the linear-baseline implementation. Breaks AC-3 (no model-specific dependencies).

**Decision.** **Method (a) empirical quantile band from per-fold rolling-origin errors** — model-agnostic, non-parametric, directly motivated by the rolling-origin design. Method (b) is an acceptable simplification if the quantile path is deemed too much machinery for Stage 6; method (c) is rejected because it violates AC-3.

**Harness implication (linked to OQ-7).** The current Stage 4 harness returns metrics only, not predictions. Three routes to supply predictions to the plot helper: (i) extend `harness.evaluate` with an optional `return_predictions=True` parameter; (ii) add a parallel `harness.evaluate_with_predictions` function; (iii) require the plot helper itself to re-run the rolling loop. Route (i) is minimal and backward-compatible; Route (ii) is cleaner but proliferates API surface; Route (iii) duplicates logic.

Maps to OQ-7, AC-9.

---

## 7. Clear community adopts (high-confidence recommendations)

These are low-judgement-call calls — strong external consensus.

**A1.** **Okabe-Ito qualitative palette** as default, injected via `rcParams["axes.prop_cycle"]` at `plots.py` import. Opt-out is a one-liner. Wong 2011 + R community consensus + Colorbrewer acceptance.

**A2.** **`plot_acf(resid, lags=168, alpha=0.05)`** for Stage 6. Do not accept the statsmodels default lags — they miss the weekly spike that motivates Stage 7.

**A3.** **Defer per-horizon breakdowns** to the first multi-horizon model stage. Intent §Points for consideration already says to consider deferring; the research confirms no community idiom that makes per-horizon diagnostics cheap for a single-horizon model.

**A4.** **Empirical-quantile uncertainty band from rolling-origin errors**, not model-based prediction intervals, to preserve AC-3 model-agnosticism.

**A5.** **No plotly, no altair** — github.com renderability is load-bearing.

**A6.** **Four-panel residual diagnostic layout** (residuals-vs-time, histogram + Normal overlay, ACF, predicted-vs-actual) is standard in Hyndman fpp3 §5.3 and should be the hero layout for Stage 6.

---

## 8. Judgement calls (for the plan)

**J1.** Opinionatedness (OQ-3) — research does not dictate this. Default posture: moderately opinionated helpers (sensible defaults, all parameters overridable), not templating. Facilitators can still pass `ax=` to drop into bespoke figures.

**J2.** Holdout wiring (OQ-8) — `NesoBenchmarkConfig.holdout_start/_end` is ready for a fixed-window retrospective plot consumer. Stage 6 is the natural home. Whether it's load-bearing for the stage or deferred to the first multi-horizon stage is a plan-level call.

**J3.** Module name (OQ-9) — research is indifferent. `plots.py` is conventional; `viz.py` is fine; `diagnostics.py` is more specific but longer. Plan picks one.

**J4.** matplotlib runtime vs dev-only (I-2) — not a research question. Three viable options; plan picks with regard to downstream-stage impact.

---

## 9. Citations

- Wong, B. (2011). "Points of view: Color blindness." *Nature Methods* 8(6), 441. — Okabe-Ito palette.
- Nunez, Anderton & Renslow (2018). "Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data." *PLoS ONE* 13(7). — cividis.
- Hyndman & Athanasopoulos. *Forecasting: Principles and Practice (3rd ed., fpp3)*. — §5.3 residual diagnostics; §5.5 empirical prediction intervals.
- Gelman, Hill & Vehtari. *Regression and Other Stories* (2021, 2025 repr.). — §11.3 predicted-vs-actual axis convention; Normal-approximation caveats.
- Hong, T. (2016). "Probabilistic electric load forecasting: A tutorial review." *Int J Forecasting* 32(3). — GEFCom 2014 residual idioms.
- Ziel, F. & Weron, R. (2018). "Day-ahead electricity price forecasting with high-dimensional structures." *Energy Economics* 70. — LEAR residual treatment.
- Amperon engineering blog (2024). "Residual heatmaps for demand forecasts." — 24×7 heatmap idiom.
- statsmodels >=0.14 documentation — `graphics.tsaplots.plot_acf` defaults, Bartlett bands.
- seaborn >=0.13 palette docs — `"colorblind"` vs Okabe-Ito differences.
- matplotlib >=3.9 colormap additions — `Okabe_Ito` registration.
