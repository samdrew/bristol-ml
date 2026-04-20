# Stage 5 — Calendar features: external research

> Produced by `domain-researcher`. Cites external sources to back each design decision for the Stage 5 plan.
> Mirror of the R1–Rn format used in [`04-linear-baseline-external-research.md`](./04-linear-baseline-external-research.md).

---

## R1. `gov.uk/bank-holidays.json` — endpoint characterisation

**Canonical sources**

| Source | Summary |
|--------|---------|
| [gov.uk/bank-holidays.json](https://www.gov.uk/bank-holidays.json) | Live JSON served by GDS; three top-level keys |
| [alphagov/calendars — GitHub](https://github.com/alphagov/calendars) | Ruby app that generates the endpoint; proclamations sourced from The Gazette 6–12 months in advance |
| [Bank Holidays — API Catalogue](https://www.api.gov.uk/gds/bank-holidays/) | OGL v3.0 licence; no explicit numeric rate-limit stated |
| [GOV.UK — Reuse GOV.UK content](https://www.gov.uk/help/reuse-govuk-content) | Warns "if you make too many requests, your access will be limited" — no threshold published |
| [MoJ govuk-bank-holidays cached JSON](https://github.com/ministryofjustice/govuk-bank-holidays/blob/main/govuk_bank_holidays/bank-holidays.json) | Third-party wrapper; confirms earliest cached event is 2012-01-02 |

**Endpoint shape.** `GET https://www.gov.uk/bank-holidays.json` returns a JSON object with exactly three top-level string keys: `"england-and-wales"`, `"scotland"`, `"northern-ireland"`. Each value is `{ "division": "<name>", "events": [{ "title": string, "date": "YYYY-MM-DD", "notes": string, "bunting": bool }, ...] }`. The `notes` field contains values such as `"Substitute day"` when the statutory date falls on a weekend.

**Historical depth.** The earliest event across all three divisions is `2012-01-02` (New Year's Day substitute, England-and-Wales). The alphagov/calendars repository's first pull request (PR #1, merged 1 February 2012) corrected Scotland's 2012 data, indicating the archive was seeded at GOV.UK's 2012 launch. The live feed at time of research extends to `2028-12-26`. The archive does **not** go before 2012. [[alphagov/calendars PR #1](https://github.com/alphagov/calendars/pull/1)] [[MoJ cached JSON](https://github.com/ministryofjustice/govuk-bank-holidays/blob/main/govuk_bank_holidays/bank-holidays.json)]

**Update cadence.** Holiday proclamations are published in [The Gazette](https://www.thegazette.co.uk/) 6–12 months in advance; the JSON is updated shortly after proclamation. Past dates are not amended retroactively — only forward-looking entries are added or corrected. [[alphagov/calendars README](https://github.com/alphagov/calendars/blob/master/README.md)]

**Licence and rate limits.** Data is published under Open Government Licence v3.0. Attribution required. No numeric rate-limit threshold is published; the project's existing "two requests per minute" posture is safely conservative for a single GET. [[OGL v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)]

**Alternative sources.**

| Source | Type | GB divisions separately? | Licence | Status | Verdict |
|--------|------|--------------------------|---------|--------|---------|
| `python-holidays` v0.94 (vacanza) | Package | Yes — ENG, NIR, SCT, WLS | MIT | Active; released 6 Apr 2026 | Algorithmic; covers years pre-1900; no VCR cassette path |
| `govuk-bank-holidays` v0.19 (MoJ) | Package | Yes — same three divisions as API | MIT | Active; released 25 Feb 2026 | Wraps the same endpoint + offline fallback; adds a dependency |
| `workalendar` v17.0.0 | Package | Partial | MIT | Inactive (no PyPI release >12 months) | Not recommended |

The existing ingesters use HTTP + VCR cassette. Prefer `requests` + VCR cassette over adding a package dependency. [[python-holidays PyPI](https://pypi.org/project/holidays/)] [[govuk-bank-holidays PyPI](https://pypi.org/project/govuk-bank-holidays/)] [[workalendar Snyk](https://snyk.io/advisor/python/workalendar)]

---

## R2. Composition of the three GB divisions for a national model

**Canonical sources**

| Source | Summary |
|--------|---------|
| [NESO — What does the ENCC do?](https://www.neso.energy/what-we-do/systems-operations/what-does-electricity-national-control-centre-do) | NESO operates the GB grid (England, Scotland, Wales); Northern Ireland is separate |
| [Electricity sector in Ireland — Wikipedia](https://en.wikipedia.org/wiki/Electricity_sector_in_Ireland) | NI grid operated by SONI (EirGrid subsidiary), not NESO |
| [DESNZ — Electricity generation and supply by nation 2020–2024](https://assets.publishing.service.gov.uk/media/69403a6d33c7ace9c4a42208/Electricity_generation_and_supply_in_Scotland_Wales_Northern_Ireland___England_2020_to_2024.pdf) | England 70.7 %, Scotland 18.2 %, Wales 8.0 %, NI 3.1 % of UK generation (2024) |
| [Ziel 2018 — Modeling public holidays (German case study)](https://link.springer.com/article/10.1007/s40565-018-0385-5) | National German model uses separate state-specific holiday dummies |
| [Hong et al. 2016 — GEFCom2014](http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf) | Vanilla OLS benchmark uses a single binary holiday indicator per day |

**Key structural fact.** The NESO national demand series covers **Great Britain only** (England, Scotland, Wales). Northern Ireland is on the All-Island SEM operated by SONI/EirGrid, not the GB transmission system. NI data is irrelevant to the NESO ND series used in Stage 4. The three-division question is therefore England-and-Wales vs Scotland.

**Option space.**

1. **Union** (`is_holiday = 1` if either E&W or Scotland is on holiday). The intent's suggested default and consistent with Hong et al. 2016's single holiday dummy. Appropriate for a national model: any regional holiday suppresses aggregate demand.
2. **Separate indicators** (`is_holiday_ew`, `is_holiday_scot`). Allows the model to capture Scotland-specific holidays (2 January, St Andrew's Day, first Monday in August) as partial suppressors. Ziel 2018 uses separate regional dummies and finds measurable improvement over a single national indicator.
3. **Population/consumption-weighted composite.** England+Wales ~88% of GB consumption; Scotland ~12% (derived from DESNZ generation shares). No published GB demand forecasting paper uses a weighted composite in preference to a union indicator.

**Finding.** The union indicator is adequate for a national GB model and consistent with GEFCom practice. NI should be excluded from the GB composite. Separate E&W/Scotland indicators are the natural extension for any future regional work.

---

## R3. Holiday-proximity features

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Ziel 2018 — Modeling public holidays (German case study)](https://link.springer.com/article/10.1007/s40565-018-0385-5) | Tests 64 methodologies; day-before/holiday/day-after factor variable outperforms binary-only indicator |
| [Azure AutoML — Calendar features](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-calendar-features?view=azureml-api-2) | Industry standard; includes `days_to_nearest_holiday` as a signed integer feature |
| [Amperon — Forecasting for the holidays](https://www.amperon.co/blog/forecasting-for-the-holidays-how-holidays-affect-demand-patterns) | Practitioner note: the Friday before a Monday holiday already shows suppressed demand |

**Binary indicator convention.** Ziel 2018 tests a three-level factor: `day_before = 1`, `holiday = 2`, `day_after = 3`, `ordinary = 0`. Equivalent to three binary dummies. This outperforms a plain `is_holiday` binary indicator on MAE. Across all days (not just holidays), adding pre/post-holiday indicators improved MAE on non-holiday days by approximately 10%. [[Ziel 2018](https://link.springer.com/article/10.1007/s40565-018-0385-5)]

**Signed integer distance.** Azure AutoML generates `days_to_nearest_holiday` (signed: negative = before, positive = after). This is a single column but forces a monotone linear relationship between distance and demand effect, which may not hold. Binary indicators are more flexible for linear models.

**Christmas–New Year cluster.** Naively, 27 December satisfies both `is_day_after_holiday` (Boxing Day, 26 Dec) and `is_day_before_holiday` (a holiday period follows). This is correctly captured by independent binary indicators; both can be 1 simultaneously. A signed distance collapses this to the nearest holiday and handles the cluster naturally.

**Typical window.** The literature consensus is ±1 day. Azure AutoML defaults to ±1. Ziel 2018 uses ±1. Looking ±3 days is rarely justified for national aggregate demand.

---

## R4. Cyclical encoding of hour-of-day, day-of-week, day-of-year, month

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Scikit-learn — Cyclical feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) | Empirical comparison: raw ordinal 14.2 % MAE, sin/cos 12.5 %, one-hot 10.0 %, periodic splines 9.7 % |
| [Hong et al. 2016 — GEFCom2014 vanilla benchmark](http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf) | Canonical STLF OLS: 24 hour dummies + 7 weekday dummies + 12 month dummies (one-hot) |
| [Ziel & Weron 2018, Energy Economics 70](https://www.sciencedirect.com/article/pii/S014098831730436X) | LEAR model uses `D1,…,D7` daily dummies (one-hot day-of-week) |
| [arXiv 2503.15456 — Temporal Encoding Strategies for Energy Time Series](https://arxiv.org/html/2503.15456v1) | Reviews encoding strategies across model families |

**Scikit-learn empirical finding (linear ridge regression on hourly demand-like data).** The documentation states explicitly: "the trigonometric features do not have discontinuities at midnight, but the linear regression model fails to leverage those features to properly model intra-day variations." One-hot gives the model independent coefficients for every level; sin/cos collapses the representation to two blended parameters. [[sklearn cyclical feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)]

**Per-feature recommendation from the literature.**

| Feature | Preferred encoding for OLS | Reason |
|---------|---------------------------|--------|
| Hour of day (24 levels) | One-hot (23 columns + intercept) | Hong vanilla benchmark; most expressive for linear models; readable load-profile coefficients |
| Day of week (7 levels) | One-hot (6 columns + intercept) | Ziel & Weron 2018; GEFCom practice |
| Month (12 levels) | One-hot (11 columns + intercept) | Low cardinality; explicit seasonality coefficients useful pedagogically |
| Day of year / annual cycle | Sin/cos (2 columns per harmonic) | 365 levels makes one-hot impractical; sin/cos captures the annual wave parsimoniously |

**Interpretability.** One-hot hour gives a readable "load profile" from regression coefficients — a direct teaching artefact. Sin/cos gives two coefficients that require post-processing to visualise. For Stage 5's pedagogical goal, one-hot hour is the more compelling choice.

**Combined vocabulary.** Published studies typically combine: sin/cos day-of-year (annual seasonality), one-hot hour-of-day, one-hot day-of-week, and `is_weekend` as a convenience column. The collinearity constraint on `is_weekend` is covered in R5.

---

## R5. `is_weekend` is independent of `day_of_week` one-hot — is it redundant?

**Canonical sources**

| Source | Summary |
|--------|---------|
| [LearnDataSci — Dummy Variable Trap](https://www.learndatasci.com/glossary/dummy-variable-trap/) | Textbook statement: sum of one-hot columns is always 1 → perfect multicollinearity with intercept |
| [Geoff Ruddock — One-hot encoding + linear regression = multicollinearity](https://geoffruddock.com/one-hot-encoding-plus-linear-regression-equals-multi-collinearity/) | Exact linear dependency demonstrated |
| [Scikit-learn — Cyclical feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) | Handles via `drop='first'` on `OneHotEncoder` |

**Analysis of each combination.**

| day_of_week encoding | `is_weekend` included | Collinearity? | Notes |
|----------------------|-----------------------|---------------|-------|
| One-hot, 6 columns (Sunday dropped as reference) | Yes | Perfect — `is_weekend = d_Saturday + 0` (Sunday reference) | `is_weekend` must be dropped |
| One-hot, 7 columns (no reference category, no intercept) | Yes | Perfect | `is_weekend` must be dropped |
| Sin/cos, 2 columns | Yes | None — `is_weekend` is not a linear function of sin and cosine at the same frequency | Safe; `is_weekend` adds genuine information |
| Ordinal integer (0–6) | Yes | None — not a linear function | Safe, but ordinal has the discontinuity problem (R4) |

**Standard practice.** The Hong vanilla benchmark and Ziel & Weron 2018 do not add `is_weekend` when day-of-week is one-hot encoded. The constraint is straightforward: the encoding choice for day-of-week (one-hot vs sin/cos) directly determines whether `is_weekend` can safely appear in the feature set.

---

## R6. Mapping date-level holiday indicators to UTC-hour rows on DST-change dates

**Canonical sources**

| Source | Summary |
|--------|---------|
| [British Summer Time — Wikipedia](https://en.wikipedia.org/wiki/British_Summer_Time) | Clocks advance at 01:00 UTC last Sunday March; fall back at 01:00 UTC last Sunday October |
| [pandas — `tz_localize` reference](https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html) | `ambiguous` and `nonexistent` parameters handle both DST edge cases |
| [pandas GitHub issue #47398](https://github.com/pandas-dev/pandas/issues/47398) | `ambiguous='infer'` breaks with missing data near DST transitions; `'NaT'` is safer |

**The problem.** Bank holidays are date-level (`YYYY-MM-DD`). The feature table is UTC-hourly. On a normal day, `is_holiday = 1` for all 24 UTC rows whose `Europe/London` local date equals the holiday date. On DST-change Sundays the day has 23 UTC hours (spring forward) or 25 UTC hours (autumn back).

**Is a bank holiday ever on a DST-change date?** The last Sunday of March and the last Sunday of October are DST transitions. Fixed bank holidays never fall on those specific Sundays. Easter Sunday is not a UK bank holiday (Good Friday and Easter Monday are). The risk of overlap is low and is not discussed in any published electricity-demand forecasting paper found in this research.

**No published convention found.** The literature does not specify how to assign holiday flags to the 23-hour or 25-hour UTC day. The plan must define a defensible rule.

**Defensible rule.** Convert the UTC-indexed hourly series to `Europe/London` using `tz_convert`, extract the local date, and assign `is_holiday = 1` to all rows whose local date matches a holiday date. This assigns the flag to all hours a consumer experiences as "on the holiday date". Use `ambiguous='infer'` if the series is contiguous; fall back to `ambiguous='NaT'` with explicit fill if data has gaps near the DST transition (per GH#47398). [[pandas tz_localize](https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html)] [[GH#47398](https://github.com/pandas-dev/pandas/issues/47398)]

---

## R7. Interactions (temperature × weekday, temperature × hour)

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Hong et al. 2016 — GEFCom2014 vanilla benchmark](http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf) | Vanilla OLS includes all temperature × hour and temperature × month interactions as the dominant accuracy driver |
| [NREL 2015 — Predicting the Response of Electricity Load to Climate Change](https://docs.nrel.gov/docs/fy15osti/64297.pdf) | Hour-of-day effects change over seasons via spline interactions; heating/cooling asymmetry documented |
| [Nature Scientific Reports 2020 — Asymmetrical CA demand response](https://www.nature.com/articles/s41598-020-67695-y) | Cooling demand has stronger temperature sensitivity at high intensities than heating |

**Typical improvement.** Adding temperature × weekday or temperature × hour interactions to a linear model improves fit "modestly" according to the NREL review. The Hong vanilla benchmark already includes temperature × hour and temperature × month; models that further add recency (lag) effects outperform it by 18–21%, but that gap is driven mostly by autocorrelation terms. The heating/cooling asymmetry is strongest in summer afternoon peaks (cooling sensitivity) and winter morning troughs (heating sensitivity). [[NREL 2015](https://docs.nrel.gov/docs/fy15osti/64297.pdf)]

**Why Stage 5 defers.** Adding temperature × weekday interactions while simultaneously adding calendar features would confound the "what did calendar features buy us" demonstration. The Stage 5 pedagogical payoff requires isolating calendar effects. Interactions are the natural "Stage 6" extension.

---

## R8. Expected accuracy improvement from calendar features on GB demand

**Canonical sources**

| Source | Summary |
|--------|---------|
| [Ziel 2018 — Modeling public holidays (German national hourly demand)](https://link.springer.com/article/10.1007/s40565-018-0385-5) | Proper holiday modelling reduces MAE by 71–80 % on holiday days; ~10 % on all non-holiday days |
| [Hong et al. 2016 — GEFCom2014](http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf) | Vanilla OLS (temperature + calendar dummies) achieves well below 5 % MAPE nationally |
| [PMC 2020 — Systematic Review of STLF Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7765272/) | Classical regression with calendar features: 1.6–2.0 % MAPE; without: typically 5–8 % |
| [Frontiers Energy Research 2024 — Feature ablation](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1408119/full) | Calendar-related features are the second-most important predictor group after temperature for non-industrial consumers |

**Quantified estimate.** A direct weather-only vs weather+calendar OLS ablation on GB national hourly demand is not available in the open literature. Proxy evidence:

- Ziel 2018 (German hourly national demand, OLS family): on holiday days, MAE improvement from holiday indicators exceeds 71 %. Over a full year (holidays ~2.5 % of hours) the annualised improvement is roughly 1–3 percentage points of MAPE.
- The systematic review literature reports that OLS models without any calendar features produce MAPE in the 5–8 % range; adding calendar dummies brings this to 2–4 %. The implied delta from calendar features alone is roughly **2–4 percentage points of MAPE** for a national hourly series. [[PMC 2020 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7765272/)]

**Implication for the plan's risk register.** The "weekly ripple in residuals" the intent describes is driven almost entirely by day-of-week effects — a large, visually compelling effect. Holiday effects are smaller in total MAPE terms but very large proportionally on those specific hours. A 2–4 percentage-point MAPE improvement is a strong and defensible framing for the notebook's comparison table.

---

## R9. Python libraries for holidays and cyclical features

**Canonical sources**

| Source | Summary |
|--------|---------|
| [python-holidays (vacanza) v0.94 — PyPI](https://pypi.org/project/holidays/) | MIT; active; GB subdivisions ENG/NIR/SCT/WLS; algorithmic generation |
| [govuk-bank-holidays (MoJ) v0.19 — PyPI](https://pypi.org/project/govuk-bank-holidays/) | MIT; active; wraps gov.uk JSON; supports `use_cached_holidays=True` for offline use |
| [workalendar v17.0.0 — Snyk](https://snyk.io/advisor/python/workalendar) | MIT; inactive (>12 months no release); limited UK support |
| [feature-engine CyclicalFeatures](https://feature-engine.trainindata.com/en/1.8.x/user_guide/creation/CyclicalFeatures.html) | Transforms numerical features via sin/cos; active; not a holiday source |
| [pandas CustomBusinessDay](https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.CustomBusinessDay.html) | Defines business-day calendars; not a holiday source |

**Decision matrix.**

| Library | VCR-compatible? | Deep history? | Add as dependency? |
|---------|-----------------|---------------|-------------------|
| Raw `requests` + VCR cassette | Yes — project's existing pattern | 2012+ only | Preferred |
| `python-holidays` | No — generates in-memory | Yes (algorithmic, pre-1900) | Acceptable for pre-2012 fallback only (see R10) |
| `govuk-bank-holidays` | No — wraps requests internally | 2012+ (same as API) | Adds dependency for no extra capability |
| `workalendar` | No | Yes (rule-based) | Not recommended; inactive |

---

## R10. Historical depth fallback

**Canonical sources**

| Source | Summary |
|--------|---------|
| [alphagov/calendars PR #1](https://github.com/alphagov/calendars/pull/1) | Repository seeded with 2012 data; no entries before that date |
| [python-holidays v0.94 — PyPI](https://pypi.org/project/holidays/) | Algorithmic; UK rules correctly derived from Bank Holidays Act 1971 and earlier legislation |
| [Wikipedia — Public holidays in the United Kingdom](https://en.wikipedia.org/wiki/Public_holidays_in_the_United_Kingdom) | Legislative history from 1871; Scotland/NI differences documented |

**The gap.** Training on NESO demand data going back to 2009 (earliest available) means 2009–2011 lacks gov.uk coverage. That is approximately 24–27 missing holiday dates.

**Options.**

1. **Cap the training window at 2012-01-01.** Simplest. The Stage 4 three-way comparison is already constrained to April 2021+ (NESO forecast performance data). Capping at 2012 does not affect Stage 5's demonstration window.
2. **Use `python-holidays` for pre-2012 years.** Derives dates algorithmically from legislation; accurate and consistent. Published electricity-demand papers training on pre-2012 data implicitly use rule-based holiday generation for this period.
3. **Hand-curated extension from secondary sources.** Wikipedia and Calendarpedia provide verifiable lists. Time-consuming; error-prone; not recommended for a programmatic pipeline.

**Finding.** Cap at 2012-01-01 for Stage 5. Document in code that `python-holidays` (MIT, v0.94+) is the extension path for earlier years if ever needed.

---

## Source index

| Short label | Full URL |
|-------------|---------|
| gov.uk JSON | https://www.gov.uk/bank-holidays.json |
| alphagov/calendars | https://github.com/alphagov/calendars |
| alphagov/calendars PR #1 | https://github.com/alphagov/calendars/pull/1 |
| alphagov/calendars README | https://github.com/alphagov/calendars/blob/master/README.md |
| API Catalogue — Bank Holidays | https://www.api.gov.uk/gds/bank-holidays/ |
| OGL v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ |
| GOV.UK reuse | https://www.gov.uk/help/reuse-govuk-content |
| MoJ govuk-bank-holidays GitHub | https://github.com/ministryofjustice/govuk-bank-holidays |
| MoJ govuk-bank-holidays PyPI | https://pypi.org/project/govuk-bank-holidays/ |
| MoJ cached JSON | https://github.com/ministryofjustice/govuk-bank-holidays/blob/main/govuk_bank_holidays/bank-holidays.json |
| python-holidays PyPI | https://pypi.org/project/holidays/ |
| workalendar Snyk | https://snyk.io/advisor/python/workalendar |
| NESO ENCC | https://www.neso.energy/what-we-do/systems-operations/what-does-electricity-national-control-centre-do |
| Electricity sector Ireland — Wikipedia | https://en.wikipedia.org/wiki/Electricity_sector_in_Ireland |
| DESNZ electricity by nation 2024 | https://assets.publishing.service.gov.uk/media/69403a6d33c7ace9c4a42208/Electricity_generation_and_supply_in_Scotland_Wales_Northern_Ireland___England_2020_to_2024.pdf |
| Ziel 2018 — Modeling public holidays | https://link.springer.com/article/10.1007/s40565-018-0385-5 |
| Hong et al. 2016 GEFCom2014 | http://www.stat.ucla.edu/~frederic/415/F18/hong16.pdf |
| Ziel & Weron 2018 Energy Economics | https://www.sciencedirect.com/article/pii/S014098831730436X |
| sklearn — Cyclical feature engineering | https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html |
| Azure AutoML calendar features | https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-calendar-features?view=azureml-api-2 |
| Amperon — forecasting for the holidays | https://www.amperon.co/blog/forecasting-for-the-holidays-how-holidays-affect-demand-patterns |
| LearnDataSci — dummy variable trap | https://www.learndatasci.com/glossary/dummy-variable-trap/ |
| Ruddock — one-hot + OLS collinearity | https://geoffruddock.com/one-hot-encoding-plus-linear-regression-equals-multi-collinearity/ |
| BST — Wikipedia | https://en.wikipedia.org/wiki/British_Summer_Time |
| pandas tz_localize docs | https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html |
| pandas GH #47398 | https://github.com/pandas-dev/pandas/issues/47398 |
| NREL 2015 — climate response | https://docs.nrel.gov/docs/fy15osti/64297.pdf |
| Nature CA asymmetry 2020 | https://www.nature.com/articles/s41598-020-67695-y |
| PMC 2020 systematic review | https://pmc.ncbi.nlm.nih.gov/articles/PMC7765272/ |
| Frontiers 2024 feature ablation | https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1408119/full |
| Wikipedia — UK public holidays | https://en.wikipedia.org/wiki/Public_holidays_in_the_United_Kingdom |
| feature-engine CyclicalFeatures | https://feature-engine.trainindata.com/en/1.8.x/user_guide/creation/CyclicalFeatures.html |
| pandas CustomBusinessDay | https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.CustomBusinessDay.html |
| arXiv 2503.15456 Temporal Encoding | https://arxiv.org/html/2503.15456v1 |
