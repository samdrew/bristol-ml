# Stage 5 — Calendar features (without/with comparison)

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 0, Stage 3, Stage 4
**Enables:** Stages 6, 7, 8, 9 (fan-out), and the REMIT chain via the enriched feature table

## Purpose

Add calendar features to the feature table and demonstrate the measurable accuracy gain they produce, running the Stage 4 linear regression unchanged against the enriched inputs. This is the most pedagogically important stage in the project: it shows, concretely, what feature engineering is worth. A facilitator can put two metric tables side by side and say "that's what domain knowledge bought us."

## Scope

In scope:
- A small ingestion module that retrieves and locally caches bank-holiday dates for the three GB divisions (england-and-wales, scotland, northern-ireland).
- A feature derivation module that generates calendar features — hour of day, day of week, month, weekend indicator, bank-holiday indicator, proximity to the nearest holiday.
- An extension to the Stage 3 assembler so the calendar features are available as an alternative feature set, selectable via configuration.
- A notebook that runs the Stage 4 linear regression twice, once on the weather-only feature set and once on the calendar-enriched feature set, and prints both metric tables side by side with the NESO benchmark.

Out of scope:
- Any new model.
- School holiday term dates.
- Sporting events or other one-off demand-affecting events (these belong with the REMIT chain).
- Changes to the rolling-origin split.

## Demo moment

A side-by-side metric table: weather-only linear regression, weather+calendar linear regression, and the NESO benchmark. The improvement is expected to be visible and substantial — enough to justify the feature engineering work on the spot. A second visualisation shows residuals from both models over the same week: the weather-only residuals carry a weekly ripple, the enriched residuals largely do not.

## Acceptance criteria

1. Switching between the weather-only feature set and the calendar-enriched feature set is a configuration change, not a code change.
2. The calendar feature derivation is a pure function: same inputs, same outputs, no side effects.
3. The enriched feature table conforms to the schema once it has been extended.
4. The notebook produces the comparison table end-to-end.
5. The weekly-residual pattern present in the weather-only model and absent in the enriched model is visible in the notebook.
6. The bank-holiday ingestion is idempotent and offline-first against its cache.

## Points for consideration

- How holidays compose across GB divisions. For a national model, "any division is on holiday" is a reasonable composite; a per-division breakdown may pay off for any future regional work.
- Holiday-proximity features (day before a bank holiday, day after) tend to differ from the holiday itself and from ordinary days. Whether to include them is a decision that will also show up at meetups as a worthwhile discussion point.
- Encoding of cyclical features. Hour of day as a plain integer implies hour 23 is "far" from hour 0, which isn't right. Sin/cos encoding or separate indicator columns are the usual fixes. Interpretability and fit can pull in different directions.
- Day of week and month are low-cardinality categorical features. Whether to encode as one-hot, ordinal, or something else is a defensible design decision with consequences for how the linear model's coefficients read.
- Interaction terms (temperature × weekday, for example) are the first thing a "make the linear model more competitive" follow-up would reach for. Not including them in Stage 5 keeps the without/with comparison honest.
- Bank-holiday historical depth is shallow — a decade or so from `gov.uk`. Any long training window will need a fallback for earlier years.
- Bank holidays are date-level, but the feature table is hourly. Resolving date indicators to UTC hours is fiddly on DST dates.
- The stage's position — deliberately after the linear baseline — is part of the pedagogy. Attendees see the mediocre result before the better one, which is a stronger argument than leading with the better one.

## Dependencies

Upstream: Stage 0, Stage 3, Stage 4.

Downstream: every subsequent modelling stage inherits the enriched feature set by default. The notebook becomes a reference point for "how much does feature engineering matter" at future meetups.

## Out of scope, explicitly deferred

- School holiday term dates (sourcing them across four UK nations is more work than this stage justifies).
- Major events that affect demand one-off (addressed, at scale, by the REMIT chain).
- Regional modelling.
- Lag features.
