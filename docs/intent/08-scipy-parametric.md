# Stage 8 — SciPy parametric load model

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 5
**Enables:** contributes a domain-motivated parametric entry to the model comparison

## Purpose

Add a model whose structure is chosen by hand from domain knowledge rather than learned implicitly. Electricity demand has a well-understood functional form — a temperature-response curve plus harmonic terms for diurnal and weekly cycles — and fitting that form directly using numerical optimisation produces a model with interpretable parameters and uncertainty estimates. This stage demonstrates that "machine learning" is only one way to solve a forecasting problem, and that a carefully specified parametric model can be a legitimate contender.

## Scope

In scope:
- A parametric model conforming to the Stage 4 interface, built on top of a numerical optimiser. The specification includes a temperature-response term (a piecewise or smooth function of temperature) and harmonic terms for daily and weekly cycles.
- Extraction of parameter estimates and their confidence intervals from the optimiser's covariance output.
- A notebook that fits the model, plots the fitted temperature-response curve, prints the parameter table with uncertainties, and compares forecasts against prior models.

Out of scope:
- Bayesian estimation of the same form.
- Automatic functional-form search.
- Non-linear programming beyond the built-in SciPy optimisers.

## Demo moment

The fitted temperature-response curve plotted against the raw scatter, with the parameter table printed alongside. The curve explains in one picture why the linear regression struggles at temperature extremes, and the parameter table lets a facilitator say "this coefficient means this many megawatts per degree, with this much uncertainty."

## Acceptance criteria

1. The parametric model conforms to the Stage 4 interface.
2. Fit and predict round-trip through save/load.
3. The notebook demonstrates the fitted form visually and prints parameter estimates with confidence intervals.
4. The model fits in a reasonable time on the project's data.
5. Save/load preserves both the parameter values and the confidence-interval information.

## Points for consideration

- Choice of functional form. A common pattern is a piecewise-linear or quadratic temperature response with a hinge around a base temperature, plus Fourier terms for diurnal and weekly seasonality. Many equivalent forms exist; each carries different interpretations.
- Parameter identifiability. Some forms have parameters that are not identifiable from the data without strong initial guesses. Documenting sensible starting values is important.
- Uncertainty extraction. `scipy.optimize.curve_fit` returns a covariance matrix from which confidence intervals can be derived. The validity of those intervals depends on assumptions (normality of residuals, approximate linearity near the optimum) that may not hold. Worth being honest about.
- Robustness to outliers. Least-squares fits are sensitive to outliers in the demand data; robust loss functions (Huber, soft L1) change the answer. Either is a valid choice.
- Fit stability across the rolling-origin folds. If the fit produces wildly different parameters on different folds, something is wrong with the specification. This is itself a good diagnostic.
- A curve-fitting model is a good bridge to probabilistic thinking — parameter uncertainty, prediction intervals — without committing to a full probabilistic-forecasting stage.
- Whether this is really "a different kind of model" or just "OLS with feature engineering done differently" is a valid philosophical debate. At a meetup, the debate itself has value.

## Dependencies

Upstream: Stage 5 (enriched feature set), Stage 4 (model interface).

Downstream: contributes to the model comparison. If a future stage introduces probabilistic forecasting, this model's uncertainty output is a natural starting point.

## Out of scope, explicitly deferred

- Bayesian parametric fits (MCMC, variational inference).
- Probabilistic forecast scoring (pinball loss, CRPS).
- Automatic functional-form search.
