# Stage 4 — Linear regression baseline + evaluation harness

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 0, Stage 3
**Enables:** Stages 5, 6, 7, 8, 9 (fan-out); establishes the model interface every modelling stage conforms to

## Purpose

Land the first working model and the evaluation machinery that every subsequent model will use. This stage has outsized architectural weight: the model interface introduced here is the interface every future model implements, and the metric and benchmark definitions introduced here are what every future model is judged by. The content of this stage is mostly conventions; its value is that all subsequent modelling stages become cleanly additive rather than relitigating basics.

## Scope

In scope:
- A small, shared model interface (fit, predict, save, load, metadata) that both the baseline and subsequent models conform to.
- A seasonal-naive baseline model whose job is to establish a credible floor and prove the interface works without needing a real training loop.
- A linear regression model using the Stage 3 feature table.
- A metrics module covering the point-forecast metrics named in DESIGN §5.3.
- A benchmark comparison against the NESO day-ahead forecast. The archive of those forecasts needs to be fetched and cached, so a small extension to the Stage 1 ingestion module lands here.
- A CLI path that trains a model named in configuration, evaluates it on the rolling-origin splits from Stage 3, and prints a metric table.
- A notebook that fits both models, shows residuals and a forecast overlay, and prints the three-way comparison against the NESO benchmark.

Out of scope:
- Calendar features (Stage 5).
- Any model beyond naive and linear.
- The model registry (Stage 9). Models serialise to disk via the interface's save/load methods, without a registry abstraction.
- Rich diagnostic visualisation (Stage 6).
- Hyperparameter search.

## Demo moment

A single CLI invocation produces a metric table comparing the seasonal-naive baseline, the linear regression, and the NESO day-ahead forecast on the same held-out period. The facilitator can then change one word in the configuration to swap models and re-run. The comparison against the NESO benchmark is the pedagogical payoff — an honest measure of how far "temperature and nothing else" can get.

## Acceptance criteria

1. Both models train, evaluate, and print a metric table from the CLI.
2. The model interface is implementable in very few lines of code — the naive model proves this.
3. Saving a fitted model and reloading it produces identical predictions.
4. Metric functions produce mathematically correct values on hand-computed fixtures.
5. The benchmark comparison produces a three-way metric table on the held-out period.
6. The notebook runs top-to-bottom in a reasonable time on a laptop.
7. A protocol-conformance test exists for both models; metric functions have their own unit tests.

## Points for consideration

- Whether beating the NESO benchmark should be a goal of this stage at all. There is an argument that it should not be — that the linear baseline is meant to lose cleanly, so that Stage 5's without/with comparison against calendar features lands with force. That shape has pedagogical value.
- Seasonal-naive has several plausible definitions ("same hour yesterday", "same hour last week", "same hour same weekday most recent"). They differ in difficulty-to-beat. The harder the baseline, the more work the other models have to do.
- Whether a Protocol (structural typing) or an abstract base class is the better fit for the model interface. Protocol is less ceremonious and more Pythonic for small interfaces; ABCs are more explicit about intent. Both work.
- Statsmodels and scikit-learn both fit OLS, but they expose different things. One gives coefficients with standard errors and residual diagnostics out of the box; the other hides them. For a teaching project, the trade-off leans toward the richer output.
- Model serialisation at this stage can be as plain as `joblib` / `pickle`. The registry in Stage 9 will take over, so over-engineering here is waste.
- How to align the NESO forecast (half-hourly) with the model's hourly output. Aggregation choice (mean, sum, take-one) is a small but real decision.
- The number of rolling-origin folds affects evaluation runtime sharply. A reduced-fold mode for the notebook and a full-fold mode for the CLI is a plausible split.
- The feature set from Stage 3 is weather-only at this point. This stage deliberately inherits that constraint rather than quietly expanding it; Stage 5 does the expansion explicitly.

## Dependencies

Upstream: Stage 0 (configuration and CLI skeleton), Stage 3 (feature table, rolling-origin splitter).

Downstream: every modelling stage (5, 7, 8, 10, 11) implements the interface established here. Stage 9 retrofits these models into the registry. Stage 12 loads models saved by the interface defined here.

## Out of scope, explicitly deferred

- Model registry (Stage 9).
- Calendar features (Stage 5).
- Richer evaluation visualisations (Stage 6).
- Lag features (later features stage).
- Hyperparameter search (not a design goal).
