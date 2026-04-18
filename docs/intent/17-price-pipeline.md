# Stage 17 — Price pipeline (secondary target)

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 9 (registry), Stage 5 (feature base)
**Enables:** proof that the architecture generalises to a second target

## Purpose

Train a model against GB day-ahead wholesale price using the same pipeline that has been built for demand. The goal is not a competitive price forecast — price is genuinely harder than demand, and a serious price model is out of scope. The goal is architectural: to show that a second target, with its own ingestion and its own natural inputs, slots into the same assembler, registry, and evaluation harness that demand uses.

## Scope

In scope:
- Ingestion of day-ahead market-index prices from Elexon Insights.
- Ingestion of generation-by-fuel-type from Elexon Insights, because renewable share is a primary price driver that demand-side features alone will not capture.
- An extension of the assembler to produce a price-target feature table, reusing the weather and calendar features and adding the generation-mix inputs.
- A training run of one of the existing model implementations against the price target.
- Entries in the registry for price-target models alongside demand-target models.
- A notebook showing the same CLI leaderboard query filtered to the price target.

Out of scope:
- Any effort to close the gap to serious price forecasters.
- Quantile or probabilistic price forecasting.
- Imbalance or system prices (different stream, different motivation).
- Gas price, carbon price, or other exogenous market drivers.

## Demo moment

The registry CLI listing models for two targets side by side. Same interface, same commands, same metrics — only the target differs. The pedagogical point is that none of the architecture needed changes; the second target simply plugged in.

## Acceptance criteria

1. Price and generation-mix ingestion follow the established pattern (local cache, offline-first).
2. The assembler produces a price-target feature table whose schema differs from the demand table only in what's appropriate.
3. At least one existing model trains against the price target without changes to the model itself.
4. Price-target models register alongside demand-target models; the registry's list query supports filtering by target.
5. The notebook shows the leaderboard split by target.

## Points for consideration

- Target choice. The N2EX Day-Ahead Hourly index and the EPEX 60-minute index are the two obvious candidates. They differ slightly; the choice needs a one-line rationale.
- Price is spikier than demand and occasionally negative. Metrics that work well for demand (MAPE, especially) misbehave on prices near zero. A price-appropriate metric set is slightly different.
- Generation mix features. Wind generation in particular is a strong price driver in GB. Whether to use historical generation or forecasted generation depends on whether the target is a true day-ahead forecast; for a training-time exercise, historical is fine.
- Gas price matters for price forecasts but not for demand. Sourcing gas-price history is out of scope for this stage; flag as a likely reason the price model underperforms.
- Rolling-origin semantics need no change; the same splitter works for both targets.
- Whether REMIT features should be used for price. REMIT moves price more reliably than it moves demand — nuclear outages are a clearer price signal than demand signal. If Stage 16 has shipped, wiring its features into the price model is cheap.
- Interaction with Stage 6's diagnostics. Price has different error characteristics (fat tails, heteroscedasticity) that make the standard diagnostics less informative. A note in the notebook is probably enough.
- Whether to position the price result as "see, we can do price too" or "here is how our architecture shows, honestly, that price is harder." The second framing is more honest and more pedagogically valuable.

## Dependencies

Upstream: Stage 9 (registry), Stage 5 (base feature set), and at least one model from Stages 4, 7, 8, 10, or 11.

Downstream: establishes the two-target pattern the rest of the project can exploit.

## Out of scope, explicitly deferred

- Imbalance prices.
- Gas and carbon price inputs.
- Probabilistic price forecasting.
- Cross-border interconnector flows as features.
