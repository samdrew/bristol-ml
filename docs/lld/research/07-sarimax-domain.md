# Stage 7 — SARIMAX domain research

**Author:** `@researcher`
**Date:** 2026-04-21
**Target plan:** `docs/plans/active/07-sarimax.md` (not yet created)
**Intent source:** `docs/intent/07-sarimax.md`
**Scope:** external literature + statsmodels primary docs to inform Stage 7 plan decisions. Numbered subsections (§R1–§R10) so the plan can cite by reference.

---

## §R1 — statsmodels SARIMAX: canonical API and gotchas

The canonical constructor is

```
SARIMAX(endog, exog=None, order=(p,d,q), seasonal_order=(P,D,Q,s),
        trend=None, measurement_error=False,
        time_varying_regression=False, mle_regression=True,
        simple_differencing=False, enforce_stationarity=True,
        enforce_invertibility=True, hamilton_representation=False,
        concentrate_scale=False, trend_offset=1, use_exact_diffuse=False,
        ...)
```
(Source: statsmodels 0.14 stable docs [1].)

**Default values that matter for this stage (confirmed from the docs page [1]):**

| Param | Default | Effect |
|---|---|---|
| `enforce_stationarity` | `True` | Transforms AR parameters so roots lie outside unit circle. Rejects non-stationary ML optima. |
| `enforce_invertibility` | `True` | Same for MA parameters. |
| `concentrate_scale` | `False` | Keep sigma^2 in the parameter vector. Setting `True` shrinks the parameter vector by 1 and usually speeds optimisation when there are many initial states to estimate [2]. |
| `simple_differencing` | `False` | Uses Harvey representation (differencing inside the state vector, no lost observations). Setting `True` differences the data up front, loses `d + D*s` observations, but shrinks the state-space dimension. |
| `hamilton_representation` | `False` | Keep the default Harvey rep unless reproducing Stata/R output [1]. |
| `low_memory` (fit kwarg) | `False` | On `fit(... low_memory=True)`, the filter does not store smoothed results — diagnostics, in-sample prediction and forecasting still work [2]. |

**Warnings that routinely fire.** The maintainers explicitly changed SARIMAX (PR #4739 [3]) to only *warn* — not fail — when the least-squares starting parameters land outside the stationarity/invertibility region; it then zeroes those starting params. Expect `UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.` on demand data, and plausibly `ConvergenceWarning: Maximum Likelihood optimization failed to converge.` from scipy's optimiser when the seasonal period is large. Both are informational in the SARIMAX code path; SARIMAX returning a fitted result is not blocked by them [3][4]. The burn-in guidance in the FAQ [5] is that residuals for the first `max((P+D)*s + p + d, Q*s + q)` observations are not reliable — for `s=168, D=1` that is ≥ 168 hours of residuals that must not be scored.

## §R2 — Dual seasonality (s=24 and s=168): the core modelling decision

The GB demand series has strong seasonality at both s=24 and s=168 (and a weaker annual term). statsmodels `SARIMAX` takes **exactly one `seasonal_order`**; there is no supported "double seasonal" constructor [6] (GitHub issue #5391 is a feature request with no maintainer commitment). Three routes exist:

**(a) Single SARIMAX with s=168.** Faithful, one line of code, but state dimension is dominated by the seasonal block. Multiple user reports [6][7][8] describe memory usage >20 GB and fit times measured in tens of minutes to hours. Feasible only with `simple_differencing=True` and aggressive order caps (see §R3).

**(b) Stacked seasonal differencing (s=24 captured by SARIMAX, s=168 by a pre-difference).** Not supported by a single `SARIMAX` call; the workaround is to apply a weekly difference (`y_t - y_{t-168}`) outside the model and then fit `SARIMAX(..., seasonal_order=(P,D,Q,24))` on the differenced residual. This is mentioned in community threads [7][8] but has no blessed implementation in statsmodels. It also complicates inverse-transform at predict time.

**(c) Dynamic harmonic regression (Fourier exogenous regressors, non-seasonal ARIMA error).** Hyndman's explicitly recommended route for m > ~200 [9][10][11]. The recipe is: add K_d pairs of `sin(2π k t / 24), cos(2π k t / 24)` and K_w pairs of `sin(2π k t / 168), cos(2π k t / 168)` to `exog`, set `seasonal_order=(0,0,0,0)`, and let a small non-seasonal `(p,d,q)` ARMA model the residual. Hyndman states plainly in *fpp3* §12.1 [11] that "seasonal versions of ARIMA and ETS models are designed for shorter periods … for large m, the estimation becomes almost impossible" and that DHR "allows any length seasonality". The hyndsight post [9] sets the practical cutoff: "`arima()` will allow a seasonal period up to m=350 but in practice will usually run out of memory whenever the seasonal period is more than about 200." *fpp3* §12.1 uses K=10 daily + K=5 weekly + K=3 annual for the Turkish electricity demand example [11][12]; the trade-off is that DHR assumes the seasonal shape is constant over the fitted window.

**What the community recommends when s=168.** The consistent answer across Hyndman's writing [9][10][11], the statsmodels user group, and skforecast docs is: **do not set `s=168` naively**. Either use DHR (preferred for long seasonal periods), stick with `s=24` and absorb the weekly pattern through calendar exogenous regressors (Stage 5 already gives us day-of-week one-hots), or pre-difference. The project has a pedagogical axis — route (a) is worth *showing* with a deliberately small training window so the learner sees the fit time explode. Route (c) is the pedagogical pay-off.

## §R3 — Fit-time budgets and training-window choice

No authoritative peer-reviewed benchmark exists. The anecdata, triangulated:

- `seasonal_order=(P,D,Q,168)` with even P+Q+D >= 1 on a year of hourly data (~8760 obs) has been reported as "kills memory" at 20 GB [6][8]. Issue #5727 reports a (0,1,0,365) model at ~7 GB saved file size [13].
- `seasonal_order=(1,0,0,24)` plus rich exogenous regressors on a year of hourly data is reported to fit in seconds to a couple of minutes depending on optimiser iterations [14].
- `concentrate_scale=True` + `simple_differencing=True` are the two settings explicitly called out by the statsmodels state-space docs [2] as speeding up fits when the seasonal period is large. `low_memory=True` on `fit()` reduces memory during the smoothing pass.
- DHR with K≈10+5 Fourier pairs fits in the same order of time as a plain ARIMA(p,d,q) on the same sample size, since the state space has no seasonal block [9][11].

**Concrete implication for the plan.** A "reasonable time" for the pedagogical notebook (acceptance criterion 3 in the intent) means *either* a month-scale training window with `s=168`, or a year-scale window with `s=24` + DHR/calendar regressors for the weekly term. Route (a) with full rolling-origin across multiple years is almost certainly not feasible in a notebook budget without `simple_differencing=True`, and even then is likely minutes-per-fold.

## §R4 — Order selection for electricity demand

The intent explicitly rejects auto-order search as an architectural feature. The literature does not converge on a single canonical order — electricity series differ by country, aggregation level, and sample length — but the orders that recur in the hourly-demand SARIMAX literature are small:

- Godt (2023) [14] auto-selected SARIMAX(2,0,0)(2,0,0,24) on German hourly consumption.
- Cassettari et al. [15] (ScienceDirect; "Modeling and forecasting hourly electricity demand by SARIMAX with interactions", *Energy* 2018) use a similar low-order structure with rich exogenous interactions.
- Hyndman fpp3 §9 [16] teaches (p,d,q)(P,D,Q,s) selection through ACF/PACF inspection and AICc; for strongly differenced daily-seasonal hourly data, `d=0, D=1` with `p, q ≤ 2, P, Q ≤ 1` is the typical textbook prescription once the series has been differenced once at the seasonal lag.
- Decomposition guidance: STL on GB demand shows additive daily and weekly components with trend on the scale of annual weather; additive (not multiplicative) is the right assumption on the level series after a seasonal difference [11][16].

**Recommendation for the plan:** a hand-picked default such as `order=(1,0,1), seasonal_order=(1,1,1,24)` with weekly structure absorbed by calendar exog + (optionally) Fourier(168, K=3-5). This is conservative, cheap to fit, and the decomposition in the notebook makes the *why* visible. Auto-order search stays a notebook exercise at most.

## §R5 — Exogenous regressor choice

Dominant finding from the hourly-demand SARIMAX literature [15][17][18]: **temperature is the dominant weather driver**; humidity adds a small marginal contribution in hot climates (via cooling load) but is weak at GB latitudes; wind speed, precipitation, and cloud cover are near-zero correlation with peak load after temperature is included [17]. Calendar variables (hour-of-day, day-of-week, month, holiday) carry most of the deterministic seasonality once present [15][18].

Given Stage 5 ships hour-of-day, day-of-week, month and holiday one-hots, adding weather beyond temperature is marginal at Stage 7. Cassettari et al. [15] show the biggest accuracy gain comes from *interactions* (temperature × hour-of-day) rather than adding raw weather channels — but interaction engineering is a Stage-5-ish concern and should not expand here.

## §R6 — Confidence intervals: parametric vs empirical

`SARIMAXResults.get_forecast(h).conf_int(alpha=0.05)` returns intervals derived from the state-space one-step-ahead prediction variance, propagated through the Kalman filter; these are **Gaussian approximation intervals** conditional on the fitted parameters, and use the standard normal (or Student's t if `use_t=True`) [19][20]. They are *trustworthy* when residuals are approximately Gaussian and homoscedastic; they systematically **undercover** when residuals are heavy-tailed (which electricity demand residuals typically are, because of extreme weather and holiday outliers) or heteroscedastic (diurnal variance pattern) [19][21]. They also ignore parameter uncertainty — only innovation uncertainty is propagated.

**Alignment with DESIGN.** DESIGN §10 defers probabilistic forecasting to Stage 10. The SARIMAX CI is a *free by-product*, not a probabilistic forecasting apparatus. The pedagogically honest move is: plot the parametric CI in the notebook **as a teaching moment**, compare it side-by-side with the Stage 6 empirical-quantile band, and call out that Stage 10 replaces it. Do not wire it into the Model interface's return type or the evaluator — that is Stage 10's contract to design. This avoids a Chesterton's Fence violation of the DESIGN §10 deferral.

## §R7 — Residual diagnostics

Chad Fulton (statsmodels maintainer) documents the recommended minimum set [22]:

- **Ljung-Box** — autocorrelation of residuals (surfaced in `results.summary()` as Q and Prob(Q)).
- **Jarque-Bera** — normality.
- **Heteroscedasticity (break-variance)** — `results.test_heteroskedasticity('breakvar')`.
- **Graphical**: `results.plot_diagnostics(figsize=(12,8))` — standardised residuals time plot, histogram+KDE vs N(0,1), Q-Q plot, and a correlogram.

These are built-in one-liners. For the pedagogical notebook the call `results.plot_diagnostics()` plus the numeric tests from `results.summary()` **is** the minimum diagnostic surface [22][14]. Stage 6 plots are demand-series diagnostics (ACF of the target); these are residual diagnostics. They are complementary, not duplicative — the ACF-of-residuals from `plot_diagnostics` tests whether SARIMAX removed the pattern that Stage 6 plots *showed*. The notebook should reuse Stage 6's lag-24/168 reference markers on the residual correlogram to drive that narrative.

## §R8 — State-space pickling and save/load

`SARIMAXResults.save(fname, remove_data=False)` pickles via the standard `pickle` protocol [23]. joblib works on the same object because joblib delegates to pickle for non-numpy-heavy internals. Known gotchas:

1. **File size.** Default `remove_data=False` saves all nobs-length arrays. Reports of 600 MB for a year of hourly data appear in issue threads [24]. Use `remove_data=True` for deployment artefacts.
2. **Bug with exogenous variables (issue #6542 [25]).** Filed Feb 2020 against a specific out-of-sample predict path when `k_exog > 0`. Fixed in the 0.12 series per the assigned milestone; project is on a current 0.14+ statsmodels so this should not hit, but the Stage 7 protocol-conformance test should include a round-trip `fit -> save -> load -> predict` with exogenous variables present to catch any regression.
3. **`SARIMAXResults.apply(new_data, refit=False)` [26]** creates a new results object that evaluates the saved parameters against fresh endog/exog. This is the idiomatic path for predicting on held-out folds without a re-fit. For the rolling-origin evaluator, `apply()` is strictly cheaper than a full `.fit()` and avoids the optimiser entirely.
4. **`remove_data=True` is not fully thorough** — issue #7494 [27] notes some length-nobs arrays are not cleared. Not a functional blocker, just a file-size note.

## §R9 — Clear-adopt list

Decisions supported by unambiguous primary-source consensus. The plan should adopt these without flagging as open questions.

- **A1.** Set `enforce_stationarity=False, enforce_invertibility=False` in the SARIMAX constructor. The default `True` frequently causes ML optimisation failures on non-stationary hourly demand; the statsmodels maintainers' PR #4739 [3] softened the SARIMAX starting-param check precisely because users running real-world seasonal data hit it. (Note: A1 is the *maximiser* setting; the *starting-param* behaviour is already warn-not-fail in SARIMAX.)
- **A2.** Use the Harvey representation (leave `hamilton_representation=False`). Only switch for Stata reproducibility [1].
- **A3.** Call `results.plot_diagnostics()` in the notebook and render the numeric Ljung-Box / JB / Breakvar stats from `results.summary()` — this is the statsmodels-blessed minimum surface [22].
- **A4.** For save/load in the Model interface, use `SARIMAXResults.save(..., remove_data=True)` in the serialisation path, and use `SARIMAXResults.apply(new_endog, new_exog, refit=False)` (not re-fit) at rolling-origin predict time. Covered by a round-trip test including exogenous data to guard against issue #6542 regressions [25][26].
- **A5.** Start exogenous regressors with temperature only plus the Stage 5 calendar one-hots. Humidity/wind/cloud are marginal at GB latitudes once calendar features are present [15][17][18].

## §R10 — Genuine open questions

These are project-specific or the literature is split. They belong in plan §1 as decision rows, not in the findings.

- **Q1. Dual-seasonality strategy: route (a), (b), or (c) in §R2.** Picking one has pedagogical, fit-time, and accuracy trade-offs. Route (a) with small training window is the most pedagogically vivid (because it shows the cost). Route (c) / DHR is what a production SARIMAX practitioner would actually do per Hyndman. Route (b) is the ugly middle. Decide based on the notebook's target run time budget.
- **Q2. Training-window length.** The rolling-origin evaluator (Stage 4) assumes refits per fold. For SARIMAX with `s=168` even the most aggressive settings likely break the notebook budget if the window is multi-year. Either (i) cap the window at e.g. 8–12 weeks for the SARIMAX refit path, (ii) fit once on the full history and `apply()` for every fold (breaks the rolling-origin semantic — honesty demands calling this out), or (iii) use DHR which amortises cheaply across folds. The right answer depends on how literally Stage 4's rolling-origin contract was specified.
- **Q3. Whether to plot the parametric CI.** §R6 lays out the argument for showing it as a teaching comparator against Stage 6's empirical band. But this risks pre-committing to a probabilistic framing that DESIGN §10 deferred. The call is: is the notebook allowed to *show* probabilistic artefacts while keeping them out of the Model interface? The lead needs human sign-off on this narrow question.
- **Q4. Order default.** §R4 recommends `(1,0,1)(1,1,1,24)` but provides no GB-specific justification beyond textbook fpp3 guidance and adjacent papers on German/Turkish grids. A quick AICc sweep in the notebook over a small grid is cheap and pedagogically aligned — but whether that sweep is "auto-order search" (out of scope) or "notebook exercise" (in scope, per intent line 18) is a semantic call for the lead.

---

## Summary table — the decisions at a glance

| Question | Clear-adopt? | Ref |
|---|---|---|
| `enforce_stationarity` / `enforce_invertibility` | Yes — both `False` | A1, [1][3] |
| `hamilton_representation` | Yes — leave `False` | A2, [1] |
| Diagnostic surface | Yes — `plot_diagnostics()` + summary stats | A3, [22] |
| Save/load contract | Yes — `save(remove_data=True)` + `apply(refit=False)` | A4, [23][26] |
| Exogenous choice | Yes — temperature + Stage 5 calendar only | A5, [15][17] |
| Dual-seasonality strategy (s=24 vs s=168 vs DHR) | **Open** | Q1, [6][9][11] |
| Training-window length under rolling-origin | **Open** | Q2, [2][6] |
| Parametric CI in notebook | **Open** (DESIGN §10 interaction) | Q3, [19] |
| Order default | Mostly clear but worth a notebook sweep | Q4, [16][14] |

---

## Citations

All URLs accessed 2026-04-21. Where pages are maintainer-authored they are treated as primary; aggregator/tutorial content is cited only where it is the best available signal on observed fit-time behaviour.

1. statsmodels (stable), *SARIMAX API reference*. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
2. statsmodels (stable), *Time Series Analysis by State Space Methods*. https://www.statsmodels.org/stable/statespace.html
3. ChadFulton (statsmodels maintainer), PR #4739 "REF: SARIMAX: only warn for non stationary starting params". https://github.com/statsmodels/statsmodels/pull/4739
4. statsmodels issue #6225, "why is it that in sarimax we only warn … and in arima we fail?". https://github.com/statsmodels/statsmodels/issues/6225
5. statsmodels (dev), *SARIMAX and ARIMA: FAQ*. https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_faq.html
6. statsmodels issue #5391, "double seasonal SARIMAX". https://github.com/statsmodels/statsmodels/issues/5391
7. statsmodels issue #6034, "SARIMAX memory consumption". https://github.com/statsmodels/statsmodels/issues/6034
8. statsmodels issue #6476, "SARIMAX memory issue". https://github.com/statsmodels/statsmodels/issues/6476
9. R. J. Hyndman, "Forecasting with long seasonal periods", *hyndsight* blog. https://robjhyndman.com/hyndsight/longseasonality/
10. R. J. Hyndman & G. Athanasopoulos, *Forecasting: Principles and Practice (3rd ed)*, §10.5 "Dynamic harmonic regression". https://otexts.com/fpp3/dhr.html
11. Hyndman & Athanasopoulos, *fpp3* §12.1 "Complex seasonality". https://otexts.com/fpp3/complexseasonality.html
12. `forecast::fourier()` reference. https://pkg.robjhyndman.com/forecast/reference/fourier.html
13. statsmodels issue #5727, "SARIMAX model too large". https://github.com/statsmodels/statsmodels/issues/5727
14. T. F. Godt, "Forecasting Hourly Electricity Consumption with ARIMAX, SARIMAX, and LSTM". https://medium.com/@timonfloriangodt/forecasting-hourly-electricity-consumption-with-arimax-sarimax-and-lstm-part-i-cc652cdd905a (tutorial; cited for observed pmdarima-selected order and broad performance numbers, not as primary)
15. L. Cassettari et al., "Modeling and forecasting hourly electricity demand by SARIMAX with interactions", *Energy* 165 (2018), 257–268. https://www.sciencedirect.com/science/article/abs/pii/S0360544218319297
16. Hyndman & Athanasopoulos, *fpp3* §9 "ARIMA models". https://otexts.com/fpp3/arima.html
17. Iatsyshyn et al., "Electricity Demand Prediction Using SARIMA: A Framework", *CEUR-WS* Vol. 4133. https://ceur-ws.org/Vol-4133/S_13_Iatsyshyn.pdf
18. MDPI, "Hybrid Forecasting for Sustainable Electricity Demand in The Netherlands Using SARIMAX, SARIMAX-LSTM, and Sequence-to-Sequence Deep Learning Models", *Sustainability* 2025. https://www.mdpi.com/2071-1050/17/16/7192
19. statsmodels (stable), *SARIMAXResults.conf_int*. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.conf_int.html
20. statsmodels (dev), *SARIMAXResults.get_forecast*. https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html
21. statsmodels issue #8230, "DESIGN: Prediction intervals in tsa". https://github.com/statsmodels/statsmodels/issues/8230
22. C. Fulton, "State space diagnostics". http://www.chadfulton.com/topics/state_space_diagnostics.html
23. statsmodels (stable), *SARIMAXResults.save*. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.save.html
24. Machine Learning Mastery, "How to Save an ARIMA Time Series Forecasting Model in Python" (noting save/load regression history and remediation in statsmodels ≥ 0.12.1). https://machinelearningmastery.com/save-arima-time-series-forecasting-model-python/
25. statsmodels issue #6542, "Bug saving and loading a SARIMAX model incl. solution". https://github.com/statsmodels/statsmodels/issues/6542
26. statsmodels (stable), *SARIMAXResults.apply*. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.apply.html
27. statsmodels issue #7494, "remove_data= option does not appear to remove data fully". https://github.com/statsmodels/statsmodels/issues/7494
