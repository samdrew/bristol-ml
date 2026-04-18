# Stage 18 — Drift monitoring

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 9 (registry), Stage 12 (serving, optional but useful)
**Enables:** a concrete answer to "how would we know if our model went bad in production?"

## Purpose

Add a drift-monitoring surface that covers the two monitoring axes identified in CD4ML: data distribution drift (are the inputs changing?) and prediction quality drift (is the model getting worse against reality?). This stage demonstrates the monitoring layer without building a full observability system — the pedagogical point is that monitoring is a first-class architectural concern, not an afterthought.

## Scope

In scope:
- A module that computes distribution-drift metrics over the feature table, partitioned by time window. Standard metrics (population stability index, Kolmogorov-Smirnov, histogram overlap) applied per feature.
- A module that computes prediction-quality drift: rolling metrics over the held-out period, showing how a model's accuracy evolves over time.
- A notebook that visualises both, on the project's actual data, with known real-world events annotated (COVID, large demand shifts, weather anomalies).
- Integration with the registry so a monitored model's lineage is explicit.

Out of scope:
- Alerting, paging, or any incident-response tooling.
- A live monitoring daemon. This stage is batch / notebook-oriented.
- Concept drift detection methods that go beyond distribution and prediction-quality metrics.
- Anomaly detection on individual predictions.

## Demo moment

A drift plot over the full test period, feature-by-feature, with a vertical line annotated "COVID lockdown starts here" and a visible distribution shift on several features. A second plot shows the model's rolling MAE over the same period, with the same vertical line producing a visible bump. A facilitator can point at this and talk about why the model would have needed retraining, and how a monitoring system would have detected it.

## Acceptance criteria

1. The drift module accepts any feature table and any time-windowing scheme and returns per-feature drift metrics without knowing anything about the model.
2. The prediction-quality module accepts model predictions and actuals and returns rolling metrics at a configurable resolution.
3. The notebook renders both analyses on the project's data, annotated with known events.
4. The code is cheap enough to run that a facilitator can compute drift for a newly trained model as part of a meetup demo.
5. Smoke tests cover the metric functions against fixtures.

## Points for consideration

- Drift metric choice. PSI is widely known and easy to interpret; KS is more theoretically grounded. Either is defensible. The notebook should probably show one with a clear interpretation rather than present a menu.
- Window size. Too short and the drift metric is noisy; too long and it lags real changes. A rolling window of 28 or 90 days is typical for daily-resolution data; the project's data is hourly, so the choice interacts with aggregation.
- Thresholds. PSI > 0.2 is the folklore "real drift" threshold. Folklore is folklore; the notebook should say so.
- Prediction-quality drift is cheaper to compute but lags — it only shows after ground truth arrives. Data drift is forward-looking. Both have their place; pointing out the distinction is part of the lesson.
- Interaction with serving. If Stage 12 is logging requests and predictions, drift monitoring can run against real production traffic. If not, it runs against the held-out period, which is less authentic but still useful.
- Feature-importance-weighted drift. Not every feature matters equally; drift in an unimportant feature matters less than drift in an important one. Weighting is a refinement.
- Visual presentation. A heatmap of features × time coloured by drift severity is compact; small multiples of per-feature distributions over time are more informative. Both have their place.
- Real-event annotations. COVID is the obvious one; warm December 2022, the 2022 energy crisis, and others are also legitimate. A small annotated events file keeps this maintainable.

## Dependencies

Upstream: Stage 9 (registry supplies the models whose outputs are monitored), Stage 12 (serving, optional, supplies production logs).

Downstream: any future retraining-trigger stage (not in the current plan) would consume drift signals.

## Out of scope, explicitly deferred

- Alerting and incident response.
- Live monitoring services.
- Anomaly detection on individual predictions.
- Feature-importance weighting.
