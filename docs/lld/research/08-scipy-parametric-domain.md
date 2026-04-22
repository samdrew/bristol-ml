# Stage 8 — SciPy parametric load model: domain research

**Author:** `@researcher`
**Date:** 2026-04-21
**Target plan:** `docs/plans/active/08-scipy-parametric.md` (not yet created)
**Intent source:** `docs/intent/08-scipy-parametric.md`
**Scope:** external literature + scipy primary docs to inform Stage 8 plan decisions. Numbered subsections (§R1–§R10) so the plan can cite by reference.

---

## §R1 — Temperature-response functional forms for electricity demand

Four main functional forms appear in the literature for GB and wider European electricity demand modelling.

**Piecewise-linear hinge (HDD/CDD style).** The most widely cited pattern in UK energy analysis is `max(0, T_base - T)` for heating load and `max(0, T - T_cool)` for cooling load, with the two terms summed. UK industry practice historically uses 15.5 °C as the HDD base temperature, a convention inherited from the Building Research Establishment and Elexon weather-correction methodology [1][2]. The EnergyLens degree-day reference warns that base temperature is building-specific and must be fitted rather than assumed [2], but for a whole-system GB model 15–16 °C is the canonical starting point. The NESO peak demand forecasting literature review [3] identifies temperature and calendar (weekday/holiday) as the dominant drivers in baseline regression models, consistent with this hinge structure. This form produces directly interpretable HDD/CDD coefficients: "X MW per degree of heating demand".

**Smooth quadratic.** `a + b·T + c·T²` captures the U-shaped demand-versus-temperature relationship without a non-differentiable kink. A UK-specific study of cooling demand published as an R Markdown analysis used exactly `poly(outturn, 2, raw=TRUE)` in a regression that also included annual and weekly Fourier terms [4]. The main advantage is that there is no base-temperature parameter to estimate, which reduces identifiability risk (see §R5). The disadvantage is that the coefficients do not map onto an HDD/CDD narrative that operators recognise.

**Piecewise quadratic / spline.** Hyndman and Athanasopoulos use a spline basis for weather covariates in dynamic-harmonic-regression examples [5]; Ceccherini et al. (Frontiers in Energy Research, 2021) use B-spline basis expansion for the temperature-demand relationship in short-term Japanese demand forecasting, reporting competitive accuracy against SVM and random forest [6]. This form is the most flexible but requires knot placement decisions, which are opaque to meetup audiences.

**Consensus for GB.** There is no single mandated form across GB utilities. NESO internal FES modelling [7] weather-corrects using Elexon's HDD/CDD convention (base 15 °C for heating, 22 °C for cooling), and the NESO peak demand literature review [3] describes regression-based baselines using temperature and degree-days as the standard entry. The ESO/NESO FES 2022 modelling methods document [7] splits electricity demand into components calibrated against weather-corrected historic data using the Elexon Balancing and Settlement Code definition. Academic literature on UK electricity demand typically uses either the simple piecewise-linear HDD/CDD form or a quadratic; both appear in peer-reviewed work. No authority mandates a specific form — every major utility defines its own.

**Recommended form for Stage 8.** The piecewise-linear double hinge (HDD + CDD with separate base temperatures, or a combined V-shaped/U-shaped form with a single break) gives the HDD/CDD interpretability the Stage 8 demo moment depends on while remaining tractable. A smooth quadratic is a valid alternative with fewer parameters.

---

## §R2 — `scipy.optimize.curve_fit` vs `scipy.optimize.least_squares`

`scipy.optimize.curve_fit` is a thin wrapper around `scipy.optimize.least_squares` (the low-level function) and `leastsq` (the legacy Fortran wrapper) [8][9]. The API contract is:

- With `method='lm'` (the default when no bounds are provided): `curve_fit` delegates to the legacy `leastsq` (MINPACK's Levenberg-Marquardt). This path does **not** support box constraints (`bounds` parameter is ignored with a warning if provided) and does **not** support robust loss functions. The returned `pcov` is computed from the Jacobian returned by `leastsq` [8].
- With `method='trf'` or `method='dogbox'` (the default when `bounds` is specified): `curve_fit` delegates to `least_squares`. Box constraints and robust loss functions are available via `**kwargs` passed through to `least_squares`. The `pcov` is constructed from the Moore-Penrose pseudoinverse of the Jacobian; when the Jacobian is rank-deficient, `pcov` is filled with `np.inf` [8][9].
- The `loss` and `f_scale` parameters are **not** formal named arguments of `curve_fit` — they are passed via `**kwargs` and forwarded to `least_squares`. This means using robust loss requires explicitly setting `method='trf'` or `method='dogbox'`, otherwise the kwargs are silently ignored or passed to `leastsq` which will raise a `TypeError` [8][9].

**What is the smallest Stage 8 can get away with?** For a first implementation without bounds and without robust loss, plain `curve_fit` with default `method='lm'` is the simplest path and produces standard Gaussian CIs from `pcov`. If the fit is sensitive to outliers (holiday demand spikes, weather station failures), switching to `method='trf'` with `loss='soft_l1'` requires only adding two kwargs; the calling signature does not change. The Stage 8 intent explicitly mentions both approaches [Intent doc §"Points for consideration"], so documenting the switch clearly in the notebook is more important than making a binary choice in the implementation.

**`pcov` semantics.** `curve_fit` computes `pcov` as the inverse of the Jacobian `J^T J` (the Gauss-Newton Hessian approximation) scaled by the estimated residual variance. The diagonal of `pcov` is the variance of each parameter estimate; `sqrt(diag(pcov))` is the standard error. This calculation assumes a linear approximation to the model near the optimum, which is valid when the curvature of the model with respect to the parameters is small relative to the parameter uncertainty [10][11]. When `method='trf'/'dogbox'` and a robust loss is used, `pcov` reflects the Jacobian at the solution to the re-weighted least-squares problem; the standard Gaussian CI is not strictly valid in this case (see §R3 and §R4).

---

## §R3 — Covariance matrix → confidence intervals

**Standard formula.** The 95 % confidence interval for a scalar parameter `θ_i` is `θ_i ± 1.96 · sqrt(pcov[i, i])` under the Gaussian approximation [10][11]. `scipy.optimize.curve_fit` returns the covariance matrix in parameter space (not residual space), so this formula applies directly to `popt` and `pcov` from the function.

**`absolute_sigma=True` vs `absolute_sigma=False`.** When `absolute_sigma=False` (the default), `curve_fit` scales `pcov` so that the reduced chi-squared at the optimum equals 1: `pcov_scaled = pcov_raw × (RSS / (M - N))`, where `M` is the number of observations and `N` the number of parameters [8]. This makes `pcov` an estimate of parameter variance that accounts for the empirical noise level — appropriate when `sigma` is specified only up to a proportionality constant. When `absolute_sigma=True`, `sigma` is taken literally and `pcov` is computed without rescaling — appropriate when `sigma` contains externally calibrated measurement uncertainties. For electricity demand residuals, where there is no external noise calibration and the noise level is estimated from the data itself, `absolute_sigma=False` is the correct default. The practical difference: with `absolute_sigma=False`, the CIs widen appropriately when residuals are larger than expected, whereas `absolute_sigma=True` with a mis-specified `sigma` will produce over- or under-confident intervals [8][10].

**When the Gaussian CI is valid.** The approximation is valid when: (a) residuals are approximately Gaussian and homoscedastic; (b) the model is approximately linear in the parameters near the optimum; (c) the parameter estimates are well away from any bounds [10][11]. For electricity demand residuals with a correctly specified functional form, assumption (a) is approximately satisfied in the bulk but violated at extreme weather events (outliers) and holidays. Assumption (b) is satisfied for the linear harmonic terms and approximately satisfied for the temperature-response terms away from the hinge.

**When it is NOT valid.** Confidence intervals from `pcov` are unreliable when: any parameter estimate lands on a bound; `pcov` contains `np.inf` entries (rank-deficient Jacobian); residuals are heavy-tailed (holiday outliers in demand data); the hinge base temperature is near the edge of the temperature support in the training data; or the number of parameters is close to the number of observations [8][11]. In these cases `np.inf` in `pcov` is the signal to fall back to a bootstrap.

**Parametric bootstrap fallback.** The standard residual-bootstrap procedure for nonlinear regression is: (1) fit the model on the training data; (2) compute residuals `e_i = y_i - f(x_i; popt)`; (3) resample the residuals with replacement to form `e*`; (4) construct `y* = f(x; popt) + e*` and re-fit; (5) repeat B ≥ 500 times and take the 2.5th/97.5th percentile of each parameter's empirical distribution as the 95 % CI [12]. This is preferred when `pcov` is singular or when residuals are visibly non-Gaussian. The caveat noted by University of Virginia Library is that residual resampling assumes homoscedastic errors; for heteroscedastic electricity demand residuals (higher variance at peak hours) a wild bootstrap (multiply resampled residuals by ±1 randomly) is more appropriate [12].

---

## §R4 — Robust loss functions in `curve_fit`

`scipy.optimize.least_squares` (and hence `curve_fit` with `method='trf'` or `method='dogbox'`) supports five loss functions [9]:

| Loss | Formula `rho(z)` | Use case |
|------|-----------------|----------|
| `'linear'` (default) | `z` | Standard least squares; sensitive to outliers |
| `'soft_l1'` | `2·((1+z)^0.5 - 1)` | "Usually a good choice for robust least squares" [9]; smooth approximation of absolute-value loss |
| `'huber'` | `z if z≤1 else 2·z^0.5 - 1` | Similar to soft_l1; flat beyond scale |
| `'cauchy'` | `ln(1+z)` | Severely downweights outliers; can cause optimiser difficulties [9] |
| `'arctan'` | `arctan(z)` | Hard cap on single-residual loss |

`f_scale` (default 1.0) defines the "soft margin" between inlier and outlier residuals: `rho_(f²) = C²·rho(f²/C²)` where `C = f_scale` [9]. Setting `f_scale` to a value in the same units as the residuals (MW) is important: for GB national demand, typical hourly residuals in a well-specified model are O(500 MW), so `f_scale=500.0` is a reasonable starting point that instructs the loss to treat residuals larger than 500 MW as outliers.

**For GB electricity demand:** `soft_l1` is the documentation-recommended first choice [9]. Holiday demand spikes and warm-summer heatwave events introduce outliers of O(1000–2000 MW) against a diurnal range of O(10000 MW); soft_l1 downweights these without the optimiser instability risk of `cauchy`. The NESO FES literature does not specify a loss function (its regression models are ordinary OLS on weather-corrected data [7]), but the academic demand-forecasting literature on robust regression in energy systems generally recommends Huber or soft_l1 over OLS when holiday outliers are present [3].

**`pcov` interpretation under robust loss.** When `loss != 'linear'`, the returned `pcov` from `least_squares` is the inverse of the weighted Gauss-Newton Hessian at the solution — it reflects the uncertainty of the parameters under the re-weighted problem, not under the original Gaussian model. The standard Gaussian CI formula `popt ± 1.96·sqrt(diag(pcov))` is a heuristic approximation in this case, not a rigorous interval. For Stage 8's pedagogical purpose — showing "±Y MW per degree" — this heuristic is acceptable and should be labelled as approximate in the notebook. For rigorous intervals under robust loss, fall back to the bootstrap (§R3) [10][11].

---

## §R5 — Parameter identifiability and initial guesses

The proposed functional form combines a temperature-response term with Fourier harmonic terms. Several pathologies are known:

**Hinge base temperature drift.** If `T_base` is included as a free parameter, it can drift to the edge of the temperature support if the training data contains few observations near the true hinge point (e.g., summer-only data has few sub-15 °C observations). The Levenberg-Marquardt algorithm has no concept of a plausible range for `T_base`; it will report a numerically valid fit with `T_base` outside the physically reasonable 12–20 °C range. **Mitigation:** use `bounds=([10, ...], [22, ...])` with `method='trf'` to constrain `T_base` to the plausible range [9]. Alternatively, use the heating-degree-day formulation with `T_base` fixed at 15.5 °C (the Elexon convention [1][2]), which eliminates the parameter entirely at the cost of a less flexible fit.

**Amplitude/phase aliasing in Fourier terms.** The representation `A·sin(ωt) + B·cos(ωt)` has a unique solution for given amplitude `R = sqrt(A² + B²)` and phase `φ = atan2(A, B)`. The fit is well-conditioned because `sin` and `cos` are orthogonal. However, if the same harmonic is represented as `R·sin(ωt + φ)` (amplitude-phase form), the `R` and `φ` parameters are non-identifiable near `R ≈ 0` (the phase is undefined when the amplitude vanishes). The standard implementation should use the `A, B` pair (sin + cos) form, not the amplitude-phase form, to avoid this singularity [13].

**Scale coefficients swallowing the intercept.** A global intercept `a₀` and a Fourier cosine term at frequency zero are collinear; the optimiser will not converge to a unique partition. Do not include a DC (zero-frequency) Fourier term alongside a free intercept.

**Data-driven initialisation strategies.** Seber and Wild [13] recommend starting from OLS on a linearised version of the model when possible. For Stage 8:
1. Fix `T_base=15.5 °C` for a first fit (eliminating the hinge parameter).
2. Initialise the linear HDD/CDD coefficients from an OLS regression of demand on HDD, CDD, and Fourier terms.
3. Use the OLS coefficients as `p0` for `curve_fit`.
4. Only then relax `T_base` as a free parameter, with tight bounds, if desired.
This sequential strategy avoids the optimiser starting far from a sensible solution.

**Moment-match on segments.** An alternative for the temperature response: segment the data by temperature decile, compute mean demand per decile, and fit a line to each segment slope to initialise the HDD/CDD coefficients. This is robust to Fourier confounding because the temperature segments average out the diurnal cycle.

---

## §R6 — Fourier harmonics for diurnal and weekly cycles

**Diurnal cycle (period = 24 h).** The literature on Fourier-based electricity demand forecasting recommends 3–6 harmonic pairs for the diurnal cycle [14][15]. Cassettari et al. [16] and the Springer Journal of Control paper [14] both find that 3–4 pairs capture the dominant load shape (morning ramp, afternoon peak, evening decay) with diminishing returns beyond 4 pairs. Hyndman and Athanasopoulos [5] recommend using the minimum K for which AIC stops decreasing; for hourly GB demand, empirical fits typically stabilise around K=3 for the diurnal cycle in a model that already has calendar one-hots.

**Weekly cycle (period = 168 h).** 2–3 harmonic pairs for the 168-hour cycle are sufficient in most analyses [5][17]. The Stage 5 feature table already contains day-of-week one-hots, which encode the first Fourier mode of the weekly cycle. Adding 1–2 additional Fourier pairs for the 168-hour period then captures the smooth intra-week shape *beyond* the one-hot step function. If day-of-week one-hots are included as exogenous regressors alongside Fourier terms for the same period, there is partial collinearity; the degrees of freedom are not additive. The safest approach is: either use Fourier-only for the weekly cycle (dropping day-of-week one-hots from the parametric model's exogenous set) or use only one or two Fourier pairs for the weekly cycle to capture the smooth harmonic shape not captured by the one-hots.

**Phase-lock risk.** There is no mathematical phase-lock between the 24-hour and 168-hour harmonics because 24 does not divide 168/7 = 24 evenly in the sense that the daily cycle resets every day; the cross-term `sin(2πt/24)·sin(2πt/168)` is not in the span of the individual harmonics. However, the two Fourier bases are not orthogonal on a finite data set when the window length is not an integer multiple of both 24 and 168. For a window of K weeks of hourly data (K·168 observations), both bases are orthogonal on the sample, and cross-term risk is negligible. For irregular window lengths, mild conditioning issues can arise; `np.linalg.cond(J)` at the optimum is the diagnostic [9].

---

## §R7 — Rolling-origin fit stability

**Expected parameter variance.** Under a rolling-origin protocol, the parametric model is re-fitted on each fold. The temperature-response coefficients (HDD slope, CDD slope) are expected to vary seasonally: a fold whose training window covers mainly winter will have a higher HDD coefficient estimate than a fold covering mainly summer. This is a feature of the rolling-origin design (it captures temporal non-stationarity) rather than a pathology — the Stage 8 notebook should plot parameter trajectories across folds alongside the forecast error trajectory to make this visible.

**Diagnostic threshold.** No published industry standard exists for "acceptable parameter jump between consecutive folds". A working heuristic, consistent with change-detection practice in regression monitoring, is to flag a fold as anomalous if the HDD coefficient changes by more than two standard deviations of the cross-fold parameter distribution [3][17]. This is a heuristic for the notebook demo, not a formal specification.

**Fit-time scaling.** For `curve_fit` on a model with O(10) parameters and O(10000) observations per fold, each fold fit completes in O(milliseconds) on modern hardware. Unlike SARIMAX with `s=168` (Stage 7 §R3 found multi-minute fits per fold), the `curve_fit` computation is dominated by Jacobian evaluation and is O(N·P) where N = observations and P = parameters. Practical experience with similar parametric demand fits confirms sub-second per-fold fit times at single-year training windows with K=3 diurnal + K=2 weekly Fourier pairs (12 parameters total plus 2–3 temperature terms = ~15 parameters) [14][6].

---

## §R8 — Dependency version pin

**Robust loss first appeared in `scipy.optimize.least_squares`, new in scipy 0.17.0 (released November 2016)** [9]. The `curve_fit` `**kwargs` passthrough to `least_squares` was present from the same release. This sets the effective minimum for robust-loss usage.

**Project constraints.** The `pyproject.toml` currently pins `statsmodels>=0.14,<1` and `pandas>=2.2,<3`; no explicit `scipy` pin is present. SciPy is pulled in as a transitive dependency of statsmodels. The current SciPy stable release is **1.17.0 (released 2026-01-10)**, which supports Python 3.11–3.14 and numpy ≥ 1.26.4 [18][19]. The project targets Python 3.12 (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`), which falls within the 1.17 support window.

**SciPy 1.13.0 was the first stable release to formally support NumPy 2.0** [18]. SciPy 1.16 requires numpy ≥ 1.25.2; SciPy 1.17 requires numpy ≥ 1.26.4. Both are compatible with numpy 2.x at runtime.

**Proposed pin:** `scipy>=1.7.0,<2`. Justification: 1.7.0 (released July 2021) is the oldest release that added Pythran as a build dependency, passes CI on Python 3.9+, and has no known `curve_fit`/`least_squares` correctness regressions at Stage 8's usage pattern. The effective minimum for robust loss is 0.17.0, but 1.7.0 is a more conservative lower bound that aligns with the project's SPEC 0 policy (support minor Python versions no older than 3.5 years) [18][20]. The upper bound `<2` guards against a hypothetical scipy 2.0 ABI break. A tighter pin `scipy>=1.11,<2` is acceptable if the project chooses to require `nan_policy` support in `curve_fit` (added in 1.9, documented from 1.11) [8].

**No deprecations.** No `curve_fit` or `least_squares` parameters used by Stage 8 are deprecated as of scipy 1.17.0 [8][9].

---

## §R9 — Plotting the fitted curve

The canonical Stage 8 demo moment — "temperature-response curve plotted against raw scatter, with parameter table alongside" — maps onto two existing `evaluation/plots.py` helpers:

- **`predicted_vs_actual`** uses `OKABE_ITO[1]` (orange) for the scatter. The Stage 8 temperature-response plot reuses the same colour convention: raw scatter in `OKABE_ITO[1]` (alpha 0.15, s=4, matching the existing scatter helper), fitted curve in `OKABE_ITO[6]` (vermillion, a distinct warm colour) as the overlaid line.
- **`_ensure_axes`** composability contract: all helpers accept `ax=None`; the temperature-response scatter + curve can be composed into a `plt.subplots(1, 2)` grid alongside the parameter table (rendered via `ax.table` or a `matplotlib.patches.FancyBboxPatch` text block) without either helper owning figure lifetime.

**New helper vs notebook-inline.** The intent does not mandate a new `plots.py` helper. The appropriate rule from `evaluation/CLAUDE.md` is that plots helpers are "model-agnostic" and "take `pd.Series`/`pd.DataFrame` inputs, never a `Model` object." A `temperature_response_scatter(T, demand, popt, model_fn, *, ax=None)` helper satisfies this: it takes temperature values, demand values, fitted parameters, and the callable model function — no Model object. Whether this belongs in `plots.py` or notebook-inline is a decision for the plan, not the research. If the same plot is shown in multiple contexts (notebook + a future stage comparison notebook) a `plots.py` helper avoids duplication; for a one-off demo, notebook-inline is simpler.

**seaborn.** The existing `plots.py` uses seaborn only for `sns.heatmap` in `error_heatmap_hour_weekday`. For the temperature scatter the direct `ax.scatter` + `ax.plot` approach (as used in `predicted_vs_actual`) is consistent with the existing style and avoids the seaborn-scatter overhead.

---

## §R10 — Where does this sit in the model zoo?

**In GB electricity forecasting specifically**, a hand-crafted parametric model occupies a distinct pedagogical position that also correlates with genuine accuracy characteristics:

**vs OLS / ridge (Stage 4).** OLS with calendar one-hots and temperature as a raw regressor is a restricted special case of the parametric model — it implicitly assumes a linear temperature response with no hinge, and its "harmonic" structure comes from one-hot encoding rather than explicit Fourier terms. The parametric model is strictly more expressive on temperature and more compact on the temporal structure. In accuracy, both sit in the 3–6 % MAPE range on national GB demand [3][16]; the parametric model's advantage is interpretability, not accuracy.

**vs SARIMAX with calendar exog (Stage 7).** SARIMAX explicitly models serial autocorrelation in residuals, which a non-sequential parametric model ignores. On hourly demand, Stage 7's SARIMAX will typically have lower residual ACF. The parametric model has the advantage of explicit parameter interpretability and faster fit (milliseconds vs seconds/minutes for SARIMAX with s=168). At the meetup demo, the two models are complementary: SARIMAX captures "what the parametric model left in the residuals"; the parametric model explains "why the linear regression struggles at temperature extremes" [Stage 8 intent].

**vs gradient boosting / deep learning (Stages 10+).** On national GB demand with rich engineered features, gradient boosting typically achieves 1–3 % MAPE vs 3–6 % for simple parametric models [3][6]. The gap widens in extreme-weather periods where the non-parametric learner extrapolates better (or worse — it depends on training-set coverage of extremes). The parametric model's pedagogical value is exactly this contrast: it is a "genuine contender" that can be beat, with the explicit causal story of *why* it is beat. Ceccherini et al. [6] show that an interpretable nonlinear regression with basis expansion achieves accuracy within 0.1 % MAPE of random forest on Tokyo power data, which is a useful counter-narrative to the "black-box always wins" position.

**Is it a contender or a foil?** Both. It is a contender because for well-behaved periods (typical winter weekday demand, moderate temperatures) it matches OLS and approaches SARIMAX. It is a foil because for extreme weather and unseen calendar effects, machine learning generalises better — and the parameter table showing "±Y MW per degree" lets the facilitator make that argument concretely.

---

## Distilled recommendations (A1–A10)

**A1** — Use a piecewise-linear double hinge with separate heating and cooling base temperatures (`T_heat ≈ 15.5 °C`, `T_cool ≈ 22 °C`), consistent with the Elexon HDD/CDD convention; provides the HDD/CDD interpretability the Stage 8 demo moment depends on.

**A2** — Fix `T_heat = 15.5 °C` and `T_cool = 22 °C` for the first-pass fit (eliminating hinge parameters from the optimisation); relax as bounded free parameters only after a stable baseline fit is confirmed, with bounds `[10, 22]` for heating and `[18, 28]` for cooling.

**A3** — Use `scipy.optimize.curve_fit` with `method='trf'` and `loss='soft_l1'` plus `f_scale=500.0` as the default; simpler than dropping to raw `least_squares` while exposing outlier robustness appropriate for holiday demand spikes.

**A4** — Set `absolute_sigma=False` (the default) so `pcov` automatically scales to the empirical noise level in MW; label CIs as "Gaussian approximation, approximate under robust loss".

**A5** — Use the `A·sin(ωt) + B·cos(ωt)` (quadrature) form for all Fourier terms, not amplitude-phase form, to avoid phase-singularity identifiability failure near zero amplitude.

**A6** — Start with K=3 diurnal pairs (period 24 h) and K=2 weekly pairs (period 168 h); prefer Fourier-only for the weekly cycle in the parametric model and exclude day-of-week one-hots from the parametric model's exogenous set to avoid partial collinearity.

**A7** — Initialise via OLS on a linearised version: fix the hinge temperatures, construct HDD/CDD and Fourier columns, run `numpy.linalg.lstsq`, and use those coefficients as `p0`.

**A8** — When any `pcov` diagonal entry is `np.inf`, fall back to a residual bootstrap (B=500 resamples) for confidence intervals; document this fallback in the notebook as the principled alternative.

**A9** — Pin `scipy>=1.7.0,<2` in `pyproject.toml`; the effective minimum for robust loss is scipy 0.17.0, but 1.7.0 provides a conservative lower bound aligned with the project's SPEC 0 policy and has no known regressions at Stage 8's usage pattern.

**A10** — The temperature-response scatter + fitted curve plot should use the existing `evaluation/plots.py` palette and `_ensure_axes` composability contract; a `temperature_response_scatter(T, demand, popt, model_fn, *, ax=None)` helper in `plots.py` is preferred if the plot appears in more than one notebook, otherwise notebook-inline is acceptable.

---

## Sources

All URLs accessed 2026-04-21.

1. Elexon / Balancing and Settlement Code; degree-day weather correction convention. Referenced via NESO FES 2022 Modelling Methods: https://www.neso.energy/document/263871/download
2. EnergyLens, "Degree Days — Handle with Care!" https://www.energylens.com/articles/degree-days
3. NESO, "Peak demand forecasting literature review (WP1)". https://www.neso.energy/document/354451/download
4. RStudio Pubs, "Cooling Demand for Power in the UK" (R Markdown analysis using quadratic temperature + Fourier regression on GB demand data). https://rstudio-pubs-static.s3.amazonaws.com/515621_f1c5832eaa72427fbe6f2e19e61a10a5.html
5. R. J. Hyndman & G. Athanasopoulos, *Forecasting: Principles and Practice (3rd ed)*, §12.1 "Complex seasonality". https://otexts.com/fpp3/complexseasonality.html
6. H. Ceccherini et al., "Interpretable Modeling for Short- and Medium-Term Electricity Demand Forecasting", *Frontiers in Energy Research* (2021). https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.724780/full
7. NESO, *FES Modelling Methods 2022*. https://www.neso.energy/document/263871/download
8. SciPy (stable), `scipy.optimize.curve_fit`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
9. SciPy (stable), `scipy.optimize.least_squares` (new in version 0.17.0). https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
10. P. R. Bevington & D. K. Robinson, *Data Reduction and Error Analysis for the Physical Sciences*, 3rd ed. McGraw-Hill, 2003. (Amazon listing): https://www.amazon.com/Reduction-Error-Analysis-Physical-Sciences/dp/0072472278
11. SciPy Cookbook, "Robust nonlinear regression in scipy". https://scipy-cookbook.readthedocs.io/items/robust_regression.html
12. University of Virginia Library, "Bootstrapping Residuals for Linear Models with Heteroskedastic Errors Invites Trouble". https://library.virginia.edu/data/articles/bootstrapping-residuals-linear-models-heteroskedastic-errors-invites-trouble
13. G. A. F. Seber & C. J. Wild, *Nonlinear Regression*. Wiley, 1989. (Internet Archive): https://archive.org/details/nonlinearregress0000sebe
14. "Electric Load Movement Evaluation and Forecasting Based on the Fourier-Series Model", *Journal of Control, Automation and Electrical Systems* (Springer). https://link.springer.com/article/10.1007/s40313-015-0186-2
15. "Hourly electricity demand forecasting using Fourier analysis with feedback", *Energy Strategy Reviews* (ScienceDirect). https://www.sciencedirect.com/science/article/pii/S2211467X20300778
16. L. Cassettari et al., "Modeling and forecasting hourly electricity demand by SARIMAX with interactions", *Energy* 165 (2018). https://www.sciencedirect.com/science/article/abs/pii/S0360544218319297
17. R. J. Hyndman, "Cross-validation for time series", *hyndsight* blog. https://robjhyndman.com/hyndsight/tscv/
18. SciPy 1.13.0 Release Notes (first NumPy 2.0 support). https://docs.scipy.org/doc/scipy/release/1.13.0-notes.html
19. SciPy Toolchain Roadmap (Python/NumPy version support table). https://docs.scipy.org/doc/scipy/dev/toolchain.html
20. Scientific Python SPEC 0 — Minimum Supported Dependencies. https://scientific-python.org/specs/spec-0000/

---
