# Stage 16 — Model with REMIT features: domain research

**Source intent:** `docs/intent/16-model-with-remit.md`
**Artefact role:** Phase 1 research deliverable (domain researcher).

---

## 1. Bi-temporal point-in-time joins for ML features

The standard discipline for training on event-stream data is known as **point-in-time correctness** (also called an "as-of join" or "temporal join"). The rule is: for every training row with event timestamp T, only feature records whose own timestamp is at or before T may be joined. Both Feast and Tecton formalise this identically. Feast documentation states that it "scans backward in time from the entity dataframe timestamp" and explicitly prevents joining any future feature values to avoid data leakage into training sets ([Feast point-in-time joins docs](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)). Tecton's training-data API expresses the same constraint: "the record in the second table that has both the closest timestamp to r and is less than or equal to r's timestamp is returned," and explicitly warns that "data from after that moment should not be included" ([Tecton constructing training data](https://docs.tecton.ai/docs/reading-feature-data/reading-feature-data-for-training/constructing-training-data)). A practitioner-level summary in [Towards Data Science](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) confirms this is the universally expected behaviour across feature store implementations.

The concrete application to Stage 16: when aggregating REMIT events into an hourly bin for a row at time T, the rule is `published_utc <= T`. This is precisely the bi-temporal `published_utc` field that Stage 13 records; using the event's `event_start` or `event_end` alone would be a leakage error.

---

## 2. Forward-looking features in time-series forecasting

Using "known unavailability over the next 24 hours" as a feature for day-ahead forecasting is a recognised and well-studied pattern. The standard term in the time-series literature is **"known future inputs"** (also rendered as "future covariates" or "known covariates"). Bryan Lim et al.'s Temporal Fusion Transformer paper defines this category explicitly: "Multi-horizon forecasting problems often contain a complex mix of inputs — including static covariates, **known future inputs**, and other exogenous time series that are only observed historically" ([TFT paper, arXiv:1912.09363](https://arxiv.org/abs/1912.09363)). The TFT architecture has a dedicated encoder pathway for this input type, which demonstrates how mainstream the pattern is. AutoGluon's time-series documentation uses the parallel term "known covariates" — features "known for the entire forecast horizon" — and requires them to be passed explicitly to `predict()` ([AutoGluon time-series in-depth guide](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-indepth.html)). For REMIT specifically: because REMIT events are published before their start time and carry explicit start/end windows, the "total unavailable capacity in T+1 to T+24" feature is genuinely known at decision time. It satisfies the known-future-input contract without requiring any imputation or forecast.

---

## 3. REMIT and demand forecasting — prior art

No credible published study was found that directly tests REMIT (or an equivalent structured outage feed) as a feature for GB or European **demand** forecasting. The academic literature on electricity load forecasting (reviewed in [Springer Sustainable Energy Research, 2025](https://link.springer.com/article/10.1186/s40807-025-00149-z) and [MDPI Energies review, 2025](https://www.mdpi.com/1996-1073/18/15/4032)) consistently uses weather, calendar, and lagged-demand features; supply-side outage data does not appear as a standard input.

By contrast, the price forecasting literature does reference outage/unavailability data. A review of electricity price forecasting (EPF) notes that "reserve margin (available generation minus predicted demand) and information about scheduled maintenance and forced outages" are fundamental price drivers ([arXiv:2204.00883](https://arxiv.org/pdf/2204.00883)), and the NBEATSx paper on electricity price forecasting with exogenous variables uses day-ahead load and renewable generation forecasts as future covariates ([arXiv:2104.05522](https://arxiv.org/abs/2104.05522)). The Elexon BMRS API exposes REMIT data as a distinct endpoint ([bmrs.elexon.co.uk/remit](https://bmrs.elexon.co.uk/remit)), and NESO's own demand forecasting methodology documentation does not list generation outage status among its inputs ([NESO Peak Demand Forecasting project](https://www.neso.energy/about/innovation/our-innovation-projects/peak-demand-forecasting)). This absence is itself evidence: the systems operator running the benchmark does not use supply-side features for demand forecasting.

The intent's framing — that REMIT is much more informative for price than demand — is consistent with everything found, but no study was located that directly tests and reports the null result. **Stage 16's ablation would be a small piece of original work.**

---

## 4. Sparse-zero features in linear vs tree-based vs NN models

The core issue with near-always-zero features in linear models is that a single global coefficient must account for both the zero-region (where the feature is uninformative) and the non-zero region (where it may carry a real signal). The model is penalised or diluted in either direction. Tree-based methods sidestep this naturally: a split on an outage-capacity feature fires only in the non-zero region and has no effect on zero-region rows, so the information is extracted without global interference. XGBoost implements an explicit sparsity-aware algorithm that routes missing-or-zero values to a learned "default direction" rather than treating them symmetrically with non-zero values ([XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)). For neural models, the literature on zero-inflated gradient boosting (e.g., [arXiv:2307.07771](https://arxiv.org/abs/2307.07771)) confirms that tree ensembles outperform GLMs on zero-heavy data by modelling the zero-generating and value-generating processes jointly through flexible splits. The intent's claim — "linear may struggle; trees/NN handle this more gracefully" — is well-supported and standard in practice.

---

## Canonical sources

- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) — Feast OSS docs defining the backward-scan AS-OF join for training data.
- [Tecton constructing training data](https://docs.tecton.ai/docs/reading-feature-data/reading-feature-data-for-training/constructing-training-data) — Tecton's formal statement of the "less than or equal to" timestamp rule.
- [Point-in-time correctness in real-time ML (Towards Data Science)](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) — practitioner-level summary corroborating both feature stores.
- [TFT paper arXiv:1912.09363](https://arxiv.org/abs/1912.09363) — defines "known future inputs" as a first-class input type in multi-horizon forecasting.
- [AutoGluon time-series in-depth guide](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-indepth.html) — defines "known covariates" and their contract with `predict()`.
- [NBEATSx arXiv:2104.05522](https://arxiv.org/abs/2104.05522) — demonstrates future-known exogenous variables for electricity price forecasting; prior art on supply-side features for price (not demand).
- [arXiv:2204.00883 — electricity price forecasting review](https://arxiv.org/pdf/2204.00883) — confirms outage/reserve-margin features as price drivers.
- [Elexon BMRS REMIT endpoint](https://bmrs.elexon.co.uk/remit) — canonical source for UK REMIT data publication.
- [NESO Peak Demand Forecasting project](https://www.neso.energy/about/innovation/our-innovation-projects/peak-demand-forecasting) — NESO's own methodology; does not list supply-side outage data.
- [XGBoost sparsity-aware algorithm](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) — documents how trees handle zero/missing values without a global coefficient.
- [Enhanced Gradient Boosting for Zero-Inflated Data arXiv:2307.07771](https://arxiv.org/abs/2307.07771) — empirical comparison confirming tree ensembles outperform GLMs on zero-heavy distributions.
- [Springer Sustainable Energy Research 2025 — load forecasting review](https://link.springer.com/article/10.1186/s40807-025-00149-z)
- [MDPI Energies 2025 — load forecasting review](https://www.mdpi.com/1996-1073/18/15/4032)
