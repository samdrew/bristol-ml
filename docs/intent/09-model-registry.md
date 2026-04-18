# Stage 9 — Model registry

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 5 (and whichever of 6-8 are shipped)
**Enables:** Stages 10, 11, 12, 17, 18

## Purpose

Introduce a registry layer that records every trained model's artefacts and metadata and makes the collection inspectable from one place. Until this stage, each model is a loose file somewhere. With it, "which model is best" becomes a leaderboard query, and every subsequent model or training run can be retrieved, compared, and served without bespoke code.

## Scope

In scope:
- A registry module that wraps the saving and loading of model artefacts with a consistent metadata record (training data version, feature set, git SHA, wall-clock timestamp, held-out metrics).
- Retrofit of the models shipped in Stages 4, 7, and 8 so they save through the registry rather than directly to disk.
- A small CLI path that lists registered models with their metrics, filterable by target, feature set, or model type.
- Documentation of the on-disk layout and metadata schema.

Out of scope:
- A hosted registry (MLflow, W&B). DESIGN §10 defers that; this stage is filesystem-based.
- A UI beyond the CLI listing.
- Model promotion / staging semantics (dev → staging → prod).
- Artefact versioning beyond "last write wins" for a named experiment.

## Demo moment

A single CLI command prints a leaderboard: every model trained so far, its metrics on the rolling-origin held-out period, the feature set it used, when it was trained. The facilitator can then load any of them by name and re-run predictions from a notebook.

## Acceptance criteria

1. The registry's public interface is small — save, load, list, maybe describe. If it grows past that, the design is drifting.
2. Every model shipped before this stage can save through the registry without code changes to the model itself (the interface is what the registry consumes).
3. Metadata is captured automatically where it can be (git SHA, timestamps, feature-set name); the rest is passed in explicitly at save time.
4. The leaderboard query is fast — listing a hundred runs should be instantaneous.
5. The on-disk layout is documented well enough that a contributor could inspect it by hand without the CLI.

## Points for consideration

- The filesystem layout is the main design decision. A flat directory with metadata files is simplest; a nested hierarchy by target/model/date is easier to navigate by hand but harder to query.
- What counts as "a model" vs "a run". Two trainings of the same model on different feature sets are two entries; two trainings of the same model on the same feature set on different days is a judgement call.
- Whether to include a hash of the feature table in the metadata. It makes reproducibility claims tighter but adds a step to every save.
- A small query interface beyond "list everything" may be worth building in — filtering by target, feature set, model type, or date range — if a meetup is going to involve "show me the best model for X."
- The leaderboard's sort order is opinionated by necessity. Lowest MAE is a defensible default; the interface should make it easy to sort by any available metric.
- How the registry interacts with save/load in the interface from Stage 4. The simplest design: the registry calls the model's `save` to produce an artefact and stores metadata alongside it. The model doesn't need to know the registry exists.
- Graduation path to MLflow. If the registry interface is kept narrow, the migration is mechanical: a new implementation of the same small interface, backed by MLflow.
- Whether to commit the registry directory to version control. Probably not — runs accumulate fast — but a portable export is useful.

## Dependencies

Upstream: Stage 5 and whichever of 6, 7, 8 are shipped. The registry only has value once several models exist.

Downstream: Stages 10 and 11 (neural models save through it), Stage 12 (serving loads from it), Stage 17 (price models share the registry), Stage 18 (drift monitoring reads prediction outputs stored alongside models).

## Out of scope, explicitly deferred

- Hosted registry services (MLflow, W&B).
- Model promotion / staging.
- Automated retraining (Stage 19 touches this).
- Multi-user concurrent access.
