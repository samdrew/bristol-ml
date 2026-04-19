# Stage 4 — external research

> Produced by `domain-researcher` during Phase 1 discovery.  Cites external sources to back each decision in `docs/plans/active/04-linear-baseline.md`.

## R1. Model interface: `typing.Protocol` vs `abc.ABC`

**Canonical sources**

| Source | Summary |
|--------|---------|
| [PEP 544 — Protocols: Structural subtyping](https://peps.python.org/pep-0544/) | Defines `Protocol` as the mechanism for structural subtyping ("static duck typing") in Python |
| [mypy Protocol docs](https://mypy.readthedocs.io/en/stable/protocols.html) | Authoritative guidance on Protocol usage, `@runtime_checkable`, and when to use ABC instead |
| [Real Python — Python Protocols](https://realpython.com/python-protocol/) | Explains structural vs nominal subtyping with practical examples |
| [Justin Ellis — ABC vs Protocol](https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/) | Pedagogical comparison; notes ABCs give instantiation-time errors useful for teaching |
| [Python 3.12 typing docs](https://docs.python.org/3/library/typing.html) | `@runtime_checkable` caveat: only checks attribute presence, not signature |
| [scikit-learn contributing guide](https://scikit-learn.org/stable/developers/contributing.html) | sklearn uses nominal inheritance from `BaseEstimator`; structural duck-typing only internal |

**Answer.** `typing.Protocol` requires no inheritance — any class with the right methods satisfies it at type-check time. `abc.ABC` is explicitly inherited; `TypeError` at instantiation if abstract methods are missing. sklearn uses nominal inheritance from `BaseEstimator`; sktime uses mixins over an ABC.  Neither uses `Protocol` primary.  DESIGN §7.3 already sketches with `class Model(Protocol)`; this is fine if `@runtime_checkable` is added for test conformance.

**Recommendation.** Use `typing.Protocol` with `@runtime_checkable`; write the protocol-conformance test as `assert isinstance(model, Model)`. Flag in the module docstring that signature correctness requires a type checker, not just `isinstance`.

## R2. Point-forecast metrics — rigorous definitions and edge cases

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Hyndman — WAPE definition](https://robjhyndman.com/hyndsight/wape.html) | Establishes WAPE = Σ|y − ŷ| / Σ|y|; traces the name to Kolassa & Schütz (2007) |
| [Wikipedia — WMAPE](https://en.wikipedia.org/wiki/WMAPE) | WAPE and WMAPE are the same formula; denominator is Σ|y|, not Σy |
| [Kim & Kim 2016 — MAAPE](https://www.sciencedirect.com/article/pii/S0169207016000121) | arctan(|ε/y|) variant bounds MAPE near zero; not needed for GB demand |
| [Amperon — NMAE vs MAPE for load](https://www.amperon.co/blog/understanding-the-benefits-of-nmae-over-mape-for-estimating-load-forecast-accuracy) | GB/US grid demand never approaches zero; MAPE is well-defined for national demand |
| [Amazon Forecast metric docs](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html) | Industry-standard operational MAE, RMSE, WAPE/MAPE definitions |

**Definitions (DESIGN §5.3).**

- **MAE:** `mean(|y − ŷ|)`.  No edge cases.
- **MAPE:** `mean(|y − ŷ| / |y|) × 100`; undefined at y = 0, inflates near zero.  GB ND minimum ~15 GW; well-defined here.
- **RMSE:** `sqrt(mean((y − ŷ)²))`.  No edge cases.
- **WAPE:** `Σ|y − ŷ| / Σ|y|` (Kolassa/Hyndman).  Denominator is *sum* of absolute actuals.

NESO itself reports mean APE in the "Day Ahead Half Hourly Demand Forecast Performance" dataset (resource `08e41551-80f8-4e28-a416-ea473a695db9`) — equivalent to MAPE.  NESO does not publish WAPE or RMSE there.

**Recommendation.** Implement all four metrics as above.  Use `Σ|y − ŷ| / Σ|y|` for WAPE.  Docstring note: MAPE is valid here because GB demand never approaches zero.

## R3. Seasonal-naive forecasts for hourly electricity demand

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Hong & Fan 2016, IJF 32(3)](http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf) | GEFCom2012/2014 paper; Tao's Vanilla benchmark is temperature-only linear regression |
| [Ziel & Weron 2018, Energy Economics 70](https://www.sciencedirect.com/article/pii/S014098831730436X) | Price forecasting; naive baseline uses `y_{t−168}` weekends and `y_{t−24}` weekdays |
| [Dancker 2023 — day-ahead load profiles](https://medium.com/@jodancker/comparing-statistical-methods-for-day-ahead-forecasting-of-electricity-load-profiles-d700ca926c2f) | Lag-24 easy to beat; lag-168 harder |
| [Hyndman — Time series cross-validation](https://robjhyndman.com/hyndsight/tscv/) | Establishes seasonal-naive as floor benchmark for seasonal series |

**Answer.**  (a) `y_{t−24}` is easy to beat, especially on Mondays.  (b) `y_{t−168}` captures day-of-week automatically; simple OLS still beats it, margin smaller.  (c) "same weekday most recent" is marginally more complex without material gain over (b).

**Recommendation.** Implement `y_{t−168}` (same hour last week).  Credible floor, beatable by a temperature-only OLS, produces the "temperature helps" narrative that motivates Stage 5 calendar features.  Document the choice in `models/naive.py`'s docstring with Ziel & Weron 2018 citation.

## R4. OLS library choice: statsmodels vs scikit-learn

**Canonical sources**

| Source | Summary |
|--------|---------|
| [statsmodels OLS](https://www.statsmodels.org/stable/regression.html) | `.summary()`: coefficients, SEs, t-stats, F-test, R², AIC, BIC, residual diagnostics |
| [statsmodels get_prediction](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html) | Native `.get_prediction(X).summary_frame()` yields prediction intervals |
| [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html) | `.coef_`, `.intercept_` only; no SEs, no prediction intervals |
| [Saturn Cloud — OLS comparison](https://saturncloud.io/blog/ols-regression-scikit-vs-statsmodels/) | Coefficients numerically identical; differ on statistical output |
| [MachineLearningMastery — integrating](https://machinelearningmastery.com/integrating-scikit-learn-and-statsmodels-for-regression/) | Wrapper to make statsmodels sklearn-compatible: ~20 lines |

**Answer.**  Coefficients are numerically identical; statsmodels adds SEs, t-stats, p-values, R², AIC, BIC, F-stat, `.get_prediction()` with intervals.  For a teaching project, statsmodels wins on output richness.  DESIGN §8 already mandates it.

**Recommendation.** Use `statsmodels.regression.linear_model.OLS`.  Wrap in `Model` protocol; sklearn-Pipeline compatibility deferred.  `.summary()` output is a ready-made notebook demo artefact.

## R5. Rolling-origin evaluation conventions

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Tashman 2000, IJF 16(4)](https://www.sciencedirect.com/article/abs/pii/S0169207000000650) | Foundational rolling-origin paper; defines expanding vs fixed windows |
| [Cerqueira et al. 2022, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/) | Reviews Tashman + subsequent literature; recommends multiple origins |
| [Hyndman — TS cross-validation](https://robjhyndman.com/hyndsight/tscv/) | Expanding-window default; mean across folds recommended; pool when fold size constant |
| [GEFCom2014 (Hong et al. 2016)](http://blog.drhongtao.com/2016/04/probabilistic-energy-forecasting-gefcom2014-papers-and-more.html) | Pinball loss summed over test weeks (pooled) |

**Answer.** Two reporting conventions: per-fold mean (Hyndman's default, with std as spread) and pooled residuals (GEFCom form).  They differ when fold sizes vary.  For Stage 4 with 335 fixed-size daily folds both are defensible; per-fold mean with spread is more pedagogically informative because it reveals seasonal instability.

**Recommendation.** Report both per-fold mean ±1σ (primary) and pooled residual metric (secondary), in the same table.  Label clearly.  Methodology itself becomes a teaching point.

## R6. NESO day-ahead forecast archive

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Historic Day Ahead Demand Forecasts](https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/historic_day_ahead_demand_forecasts) | Cardinal-point forecasts from 2018; resource `9847e7bb-986e-49be-8138-717b25933fbb` |
| [Day Ahead HH Demand Forecast Performance](https://www.neso.energy/data-portal/day-ahead-half-hourly-demand-forecast-performance/day_ahead_half_hourly_demand_forecast_performance) | HH forecast + outturn + APE from Apr 2021; resource `08e41551-80f8-4e28-a416-ea473a695db9` |
| [NESO Open Data Licence v1.0](https://www.neso.energy/data-portal/neso-open-licence) | OGL-based; attribution required |

**Answer.**  Two candidate resources:

1. `9847e7bb...` — cardinal-point only (4 points/day), 2018+.  Converting to hourly requires interpolation via the "Demand Profile Dates" dataset.  Sparse.
2. `08e41551...` — full 48-period half-hourly forecast plus outturn + APE, but only from April 2021.

Coverage gap: 2018–2021 Q1 missing from resource 2.

**Recommendation.** Use `08e41551-80f8-4e28-a416-ea473a695db9` (half-hourly + APE, Apr 2021+).  Directly comparable to the hourly model after mean-aggregation.  Restrict the three-way comparison to the 2021-onwards test window.  Snapshot a 2023-Q4 sample into `data/raw/` as test fixture.  Flag the 2018–2021 gap in the notebook.

## R7. Half-hourly to hourly alignment (MW)

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Calculating GB's half-hourly electrical demand, ScienceDirect 2021](https://www.sciencedirect.com/article/pii/S2211467X21001280) | Each NESO settlement period is an average MW rate over 30 min; hourly aggregation is mean |
| [Stage 3 plan D1](docs/plans/completed/03-feature-assembler.md) | Demand aggregation decided as `mean` — canonical in-project convention |
| [EIA hourly demand notes](https://www.eia.gov/todayinenergy/detail.php?id=42915) | US EIA reports demand in MW (average rate); standard hourly aggregation is mean |

**Answer.**  NESO's ND is a power rate (MW), the average over the settlement period.  **Mean** of the two half-hours is correct (preserves MW scale).  **Sum** would produce MWh (energy) — unit-wrong.  **Take-one** introduces aliasing.

**Recommendation.** Average the two HH NESO forecast values per UTC hour before metric computation.  Consistent with Stage 3 assembler treatment of ND outturn.

## R8. Model serialisation in 2025

**Canonical sources**

| Source | Summary |
|--------|---------|
| [scikit-learn model persistence (v1.8)](https://scikit-learn.org/stable/model_persistence.html) | Recommends skops for security; warns joblib/pickle are arbitrary-code-execution-on-load |
| [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html) | Whitelist inspection before load |

**Answer.** sklearn explicitly flags joblib/pickle as insecure by design; recommends `skops.io` when untrusted artefacts are involved.  For a teaching project with self-produced local artefacts, risk is low; joblib remains pragmatic default.  skops adds audit-step friction.  Intent already allows "joblib/pickle" as sufficient.  Stage 9 (registry) is the natural inflection.

**Recommendation.** Use `joblib.dump` / `joblib.load` for sklearn-compatible objects; use statsmodels' own `.save()` / `.load()` for the `OLSResults` object if it is serialised directly.  Code comment in `models/base.py` naming skops as Stage 9+ upgrade path.  Do NOT introduce skops as a dependency now.

## R9. Notebook runtime ceiling

**Canonical sources**

| Source | Summary |
|--------|---------|
| [DESIGN §11 OQ-1](docs/intent/DESIGN.md) | Open question: "Should notebooks run top-to-bottom in under ~2 min on a laptop?" |
| [Stage 3 D7](docs/plans/completed/03-feature-assembler.md) | 120-second budget already established |
| [sklearn computational perf](https://scikit-learn.org/stable/modules/computational_performance.html) | OLS cost O(n·p²); negligible at n=8760, p=4 |

**Answer.**  OLS at n=8760, p=4 takes <1 ms per fit on any modern laptop.  335 folds × ~1 ms = under 1 s fit time.  statsmodels adds ~5–10 ms per fold for summary stats.  Notebook overhead (parquet load, plotting) dominates.  120 s is safe.

**Recommendation.** 120-second notebook ceiling (consistent with Stage 3 D7).  Offer a `fast_mode` Hydra override (step=168 → 52 weekly folds) for demo pacing, not for performance.  CLI retains 335 daily folds.

## Key recommendations summary

| # | Question | Recommendation in ≤15 words |
|---|----------|------------------------------|
| R1 | Protocol vs ABC | `Protocol` + `@runtime_checkable`; signature check requires type checker |
| R2 | Metric definitions | WAPE = `Σ|y−ŷ|/Σ|y|`; MAPE valid for GB demand |
| R3 | Seasonal-naive | `y_{t-168}` (same hour last week) — credible, beatable |
| R4 | OLS library | statsmodels OLS for `.summary()` richness; DESIGN §8 mandate |
| R5 | Rolling-origin reporting | Per-fold mean ±σ (primary); pooled residual (secondary) |
| R6 | NESO archive | Resource `08e41551` (HH, Apr 2021+); 2023-Q4 fixture offline |
| R7 | HH → hourly | Mean over two half-hours; preserves MW units |
| R8 | Serialisation | joblib now; skops is Stage 9 upgrade path (code comment) |
| R9 | Notebook runtime | 120 s ceiling; `fast_mode` override for pacing not performance |

## Source index

All URLs cited above are listed in-line with each research question.
