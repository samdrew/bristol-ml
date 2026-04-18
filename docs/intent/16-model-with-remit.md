# Stage 16 — Model with REMIT features

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 14 (and indirectly Stage 11 or whichever is the best-performing model)
**Enables:** closes the REMIT chain with a measured accuracy contribution

## Purpose

Join the extracted REMIT features into the feature table and retrain the best-performing model against the enriched inputs. This stage delivers the REMIT chain's payoff: a measurable change in forecast accuracy attributable to a textual data stream that no prior model had access to. Whether the change is positive, negligible, or negative is itself the lesson.

## Scope

In scope:
- A feature-derivation module that aggregates per-event extracted features into hourly columns aligned to the feature table. At minimum, "total unavailable capacity at this hour, by fuel type" and "number of active unplanned outages."
- An extension to the assembler so these features are available as an alternative feature set.
- A retraining run of the best model from the prior stages on the enriched set.
- A notebook that presents the ablation: best model without REMIT features, best model with REMIT features, plus the NESO benchmark.

Out of scope:
- Fine-tuning prompt engineering in Stage 14 based on this stage's results (which would make the extraction stage a moving target).
- Adding every possible REMIT-derived feature. A small set of well-motivated features is the target.
- Any restructuring of the rolling-origin split.

## Demo moment

A three-row metric table: the best model so far (without REMIT), the same model with REMIT features, the NESO benchmark. Whatever the result, the notebook tells a clear story about it.

## Acceptance criteria

1. The REMIT-derived features are computed correctly for historical periods, using the bi-temporal discipline from Stage 13 so that each hour's feature reflects only what was known at that hour.
2. The assembler produces the REMIT-enriched feature set via configuration switch.
3. The retraining uses the same splits and metrics as prior stages.
4. The ablation is reproducible from the registry.
5. The notebook commentary honestly describes the result, including the case where REMIT features do not help.

## Points for consideration

- Bi-temporal correctness. At training time, each row's REMIT features must reflect only information that was published by that row's timestamp. The forecasting-target time and the forecast-made-at time may differ; the discipline is strict. This is where the bi-temporal work in Stage 13 pays off.
- What level of aggregation to use. "Total unavailable capacity in MW" is the obvious feature. Per-fuel-type breakdowns are richer but invite collinearity. Count of active events is a different signal from total capacity.
- Negative results are a real possibility. REMIT events are much more informative for price than demand — unplanned nuclear outages primarily move the price, not the national demand. An honest notebook treats a small or null effect as a finding, not a failure.
- Forward-looking features. REMIT events have start and end times that may be in the future at publication. Using "known unavailability over the next 24 hours" as a feature is defensible for day-ahead forecasting.
- Missingness. Most hours have no active REMIT events; the features will be dominated by zeros with occasional spikes. Linear models may struggle; tree-based or neural models will handle this more gracefully.
- Interaction with the seasonal-naive baseline. The baseline does not use any features at all, so it gives the same answer regardless. The comparison is between the best learned model with and without REMIT.
- Which model to retrain. The argument for the best-performing one is obvious; the argument for retraining several is that "REMIT helps model A but hurts model B" is itself informative. Budget-dependent.
- Whether to run the stub extractor or the real LLM extractor for the training features. The stub is cheaper and reproducible but limited; the real extractor is more faithful. For the registered artefact, the real extractor is probably right; for CI, the stub is.

## Dependencies

Upstream: Stage 14 (extractor), which depends on Stage 13.

Downstream: contributes a referenceable ablation for any future discussion of REMIT's forecasting value.

## Out of scope, explicitly deferred

- Iterating on Stage 14's extractor based on Stage 16's results (that would conflate ingestion quality with model quality).
- REMIT features for the price target (Stage 17 may pick this up).
- Feature-importance or attribution analysis.
