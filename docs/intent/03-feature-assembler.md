# Stage 3 — Feature assembler + train/test split

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 0, Stage 1, Stage 2
**Enables:** Stage 4 (linear regression) and all subsequent modelling stages

## Purpose

Establish the canonical feature table every model trains on, and the canonical train/test split every evaluator uses. After this stage, modelling work becomes cleanly additive: a new model reads the same feature table, is evaluated on the same splits, and slots into the same scoreboard. Without this canonical layer, every model stage relitigates joins and alignments, and the architecture rots.

## Scope

In scope:
- A module that joins the locally cached outputs of Stages 1 and 2 into a single hourly feature table, aligned on a single timestamp convention.
- A schema definition for the feature table, enforced at the module's output boundary.
- A rolling-origin split utility that yields train/test index pairs given a time-indexed table and a split configuration.
- Configuration describing which feature set to assemble and how the split is parameterised.
- A notebook that runs the assembler, prints the schema, and visualises the rolling-origin folds overlaid on a demand series so the discipline is legible rather than abstract.

Out of scope:
- Calendar features (Stage 5).
- Lag features.
- Any modelling whatsoever.
- Metric functions (Stage 4).
- The NESO benchmark comparison (Stage 4).

## Demo moment

One command produces the feature table as a local file. The notebook prints the schema and plots the rolling-origin folds on top of a real demand series, so an attendee can see where training ends and testing begins for each fold.

## Acceptance criteria

1. The assembler is deterministic: identical inputs produce identical output.
2. The output conforms to the declared schema.
3. The splitter produces no train/test overlap and respects chronological order within each fold.
4. The splitter returns index arrays, so downstream code can slice cheaply.
5. The notebook runs top-to-bottom quickly on a laptop.
6. Smoke test on the assembler against a small fixture; a test on the splitter for no-overlap and chronological discipline.

## Points for consideration

- Row cadence. The demand feed is half-hourly; weather is hourly. Joining at hourly resolution is the natural choice, but the aggregation from half-hourly to hourly (mean, sum, peak) is a decision that affects all downstream modelling. Different conventions make different pedagogical points.
- Missing data policy. A short-gap forward-fill for weather is defensible; dropping rows with missing demand is defensible. Documenting the policy once, at the assembler's docstring level, matters more than the specific choice, because every model inherits it.
- Settlement-period-to-hour alignment on clock-change days. Aggregating within local-time hours and relabelling to UTC is one approach; other approaches exist. The test-set behaviour on those two days a year is a useful check.
- Rolling-origin with a 1-day step gives many folds for a year-long test period, which supports variance estimation. A larger step is cheaper to evaluate but noisier. The splitter can make the step configurable.
- Gate-closure semantics for day-ahead forecasting. In training, everything is in the past so this rarely bites, but encoding the discipline now saves retrofit work later.
- Schema enforcement. `pandera` is idiomatic for DataFrame schemas; Pydantic is already present from Stage 0. Either works. The choice probably warrants a small ADR.
- Where the derived feature table lives on disk. Since it's regenerable, a location outside the repo is reasonable. Pre-committing it is probably premature.
- Feature-set naming. Stage 3's set is weather-only; Stage 5 will extend it with calendar features. Naming them distinctly now (rather than calling both "default") keeps the without/with comparison clean later.

## Dependencies

Upstream: Stages 0, 1, 2.

Downstream: Stage 4 (first model consumes assembler output), Stage 5 (extends the assembler), every subsequent modelling stage.

## Out of scope, explicitly deferred

- Calendar features including bank holidays (Stage 5).
- Lag features (later, if they become necessary).
- REMIT features (Stages 13-16).
- Multi-horizon splits.
- A feature store as a separate service (DESIGN §10).
