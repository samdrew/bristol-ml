# Stage 7 — SARIMAX

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 5
**Enables:** contributes a classical-stats entry to the model comparison

## Purpose

Add a Seasonal ARIMA with exogenous regressors to the model roster. The demand series has strong seasonality at daily and weekly scales that SARIMAX captures natively and the linear regression does not. This stage contributes a classical-stats entry to the comparison, exercises the `Model` interface on a model with genuine internal state, and tests whether the evaluation harness handles a model that produces multi-step forecasts natively rather than one-shot predictions.

## Scope

In scope:
- A SARIMAX model conforming to the interface from Stage 4, with order and seasonal-order parameters exposed through configuration.
- A notebook demonstrating order selection via seasonal decomposition and information-criterion comparison, and comparing the SARIMAX forecast against prior models on the same held-out period.

Out of scope:
- Auto-order search as an architectural feature. Order selection can be a notebook exercise, but an automatic search is out of scope for this stage.
- Multi-variate SARIMAX (VARMAX). Univariate with exogenous regressors is sufficient.
- State-space models beyond SARIMAX (Kalman filters, DLM, etc.).

## Demo moment

Seasonal decomposition of the demand series (trend, seasonal, residual) on screen, followed by the SARIMAX forecast overlaid against the linear models. The decomposition is the pedagogical unlock — it makes the seasonal structure visible in a way the linear model's coefficients don't.

## Acceptance criteria

1. The SARIMAX model conforms to the Stage 4 interface.
2. Fit and predict round-trip through save/load.
3. The model trains in a reasonable time on the project's data (SARIMAX can be slow; the order needs to be sane).
4. The notebook renders a seasonal decomposition, a fit diagnostic, and a forecast comparison.
5. A protocol-conformance test covering fit/predict/save/load exists.

## Points for consideration

- Order selection is the main design decision. A hand-picked order based on the seasonal decomposition is pedagogically clearer than an automatic search; an automatic search is faster to get running but obscures the why.
- SARIMAX fits can be expensive for long training windows. A reduced training window or a lower seasonal order may be the difference between a 30-second notebook and a 30-minute one.
- The weekly seasonality is 168 hours, which is a large seasonal period for SARIMAX. Statsmodels handles it, but expect slowness. An alternative is to stack daily and weekly seasonal terms.
- Exogenous regressors (the weather inputs) should slot in; the question is which subset. Starting with temperature alone keeps fit time bounded.
- SARIMAX forecasts a horizon natively, which matches the day-ahead framing well. The rolling-origin evaluator needs to be happy with that; any refactors to support it benefit later state-space models too.
- Confidence intervals on the forecast are a natural by-product of SARIMAX. Whether to show them depends on whether the project is committing to any probabilistic framing — a decision DESIGN §10 deferred.
- Diagnostics on the residuals (Ljung-Box, etc.) are a natural part of a SARIMAX notebook and set up the narrative for neural models that might capture patterns SARIMAX misses.
- Whether to compare against the linear baseline within the notebook or wait for a cross-model notebook later. The former is cheaper and feels complete in itself.

## Dependencies

Upstream: Stage 5 (enriched feature set), Stage 4 (model interface), Stage 6 if shipped (for richer diagnostics).

Downstream: contributes to the cross-model comparison any future ablation-style stage would produce.

## Out of scope, explicitly deferred

- Automatic order selection.
- Bayesian time-series models.
- Vector autoregression.
- Probabilistic forecast evaluation.
