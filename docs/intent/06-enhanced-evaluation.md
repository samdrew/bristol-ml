# Stage 6 — Enhanced evaluation & visualisation

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 5
**Enables:** every subsequent modelling stage inherits richer diagnostics

## Purpose

Upgrade the evaluation surface so subsequent model stages ship with diagnostics that actually show where a model succeeds and fails, rather than collapsing quality to a single number. This stage adds no new models and changes no existing ones; its value is that every future model presents better.

## Scope

In scope:
- A small helper library for evaluation plots — residual plots, calibration or reliability plots, error breakdowns by hour and weekday, forecast overlay with uncertainty bands where a model supports them.
- An update to the Stage 4 notebook to use the richer diagnostics, so the existing models become the first demonstration of the richer surface.
- Documentation showing how a future model stage can pull in the diagnostics with a small amount of boilerplate.

Out of scope:
- New models.
- Changes to the metric definitions themselves (those were fixed in Stage 4 and remain the comparison basis).
- Interactive dashboards, web UIs, or anything beyond what a notebook can display inline.

## Demo moment

The same linear model from Stage 4, the same metrics, but now with diagnostics that a facilitator can point at to motivate the next modelling stage. The weekly-pattern in residuals becomes a talking point; per-hour errors reveal which times of day the model handles worst; the forecast overlay is legible at meetup-audience distances.

## Acceptance criteria

1. A model that conforms to the Stage 4 interface can produce every diagnostic with a small, consistent amount of code.
2. The diagnostics are visually legible at meetup-audience distances (large fonts, clear axes, limited clutter).
3. The helper library has no dependencies on any specific model implementation.
4. The updated notebook runs top-to-bottom quickly on a laptop.

## Points for consideration

- Whether to include any uncertainty visualisation. The Stage 4 models are point forecasters, but a rolling-origin evaluation naturally produces a distribution of per-day errors, which can be shown as a band around the forecast line. This is a visualisation choice, not a probabilistic-forecasting one.
- Matplotlib, seaborn, plotly, altair — each has trade-offs for legibility, interactivity, and static rendering in a notebook. A single pick reduces the cognitive load on later stages.
- How opinionated the helpers should be. Very opinionated means every model's notebook looks the same (good for comparison); very unopinionated means facilitators can improvise (good for live demos).
- Colourblind-safe palettes. The project will be demoed to mixed audiences; accessibility is cheap to get right here and expensive to retrofit.
- Error breakdowns by hour, weekday, and month are cheap; breakdowns by weather regime (cold, mild, hot) or by holiday proximity are richer but domain-specific. Where to draw the line is judgement.
- Residual autocorrelation plots (ACF, PACF) are the natural motivator for Stage 7's SARIMAX; they live naturally in this stage, setting up the next.
- Whether to add per-horizon diagnostics now or defer them until a multi-horizon modelling stage arrives.

## Dependencies

Upstream: Stage 5 (and through it, Stages 3 and 4).

Downstream: every future modelling stage (7, 8, 10, 11, 16). The richer diagnostics also benefit Stage 18's drift monitoring, which shares evaluation primitives.

## Out of scope, explicitly deferred

- Probabilistic forecast visualisation (all current models are point forecasters).
- Model-explainability tooling (SHAP, partial dependence). A separate stage if ever pursued.
- Interactive dashboards.
