# Stage 19 — Orchestration

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 12 (and, implicitly, most earlier stages)
**Enables:** a single-command end-to-end pipeline run

## Purpose

Add an orchestration layer that chains the pipeline from raw ingestion through feature assembly, training, evaluation, and registry write, expressible as a single directed graph and runnable from a single command. This stage's value is pedagogical more than operational — it demonstrates what an MLOps-level-1 pipeline feels like end-to-end, and it gives a facilitator a way to say "and here's the whole thing, top to bottom, in one button."

DESIGN §11 flagged this as the stage with the lowest pedagogical value per hour; the reasoning below treats it as optional but worth including once the other stages are mature enough for the orchestration to be interesting rather than premature.

## Scope

In scope:
- A pipeline definition that chains ingestion → feature assembly → training → evaluation → registry-write as a graph rather than a script.
- The ability to run the whole pipeline, or a sub-section of it, from a single command.
- Caching or skip-if-unchanged semantics, so re-running a pipeline that is mostly up-to-date is fast.
- A notebook or small script demonstrating a full end-to-end run.

Out of scope:
- Distributed execution.
- Scheduling against a real cron or production scheduler.
- Error handling beyond what the orchestration framework gives for free.
- Continuous training triggered by drift or new data.
- Workflow UIs.

## Demo moment

A single command runs the whole project. Ingestion fetches or reads cache, features assemble, the configured model trains, evaluation runs, the registry is updated. A facilitator can point at the graph visualisation and say "this is the pipeline; this is what gets automated when you take one of these from prototype to production."

## Acceptance criteria

1. A single command runs the full pipeline end-to-end from a clean state.
2. Re-running with no changes is fast — earlier stages are skipped when their outputs are already present and fresh.
3. Individual stages can be re-run by name for debugging.
4. The pipeline graph is visualisable, ideally by the framework itself.
5. A failure in one stage produces a clear error and does not leave the registry or cache in an inconsistent state.

## Points for consideration

- Framework choice. Prefect was named in DESIGN §8. The alternatives (Airflow, Dagster, Luigi, plain Make) each have trade-offs. For local demonstration, Prefect's lightness is a plus; for "this looks like real production infra," Airflow is more recognisable.
- Whether to use the framework's task caching or build custom skip-if-unchanged logic. Framework caching is cheaper to get started with; custom logic fits the project's idioms better.
- The granularity of tasks. One task per ingestion source, per feature set, per model, per evaluation is probably right. Coarser tasks hide detail; finer tasks are noisy.
- The pipeline graph is a useful artefact in itself. A visualisation that a facilitator can project on a screen is more valuable than the raw DAG code.
- Interaction with the registry. Every pipeline run that produces a model should register it with full lineage. This is largely already true from Stage 9; orchestration just formalises it.
- Dry-run mode. The ability to ask "what would this pipeline do" without running it is useful for demos and for CI.
- Secrets handling. Orchestration frameworks vary in how they handle credentials. The project's convention (environment variables for the LLM API key) should survive any framework choice.
- Whether to make the orchestration idempotent over a run window. Running a pipeline twice for the same date range should not double-append to caches or produce two registry entries with the same lineage.

## Dependencies

Upstream: most earlier stages, but particularly Stage 12 (serving), because the end-to-end story includes "and here is how a model gets from training to being served."

Downstream: nothing in the current plan depends on this stage. A hypothetical continuous-training stage would consume orchestration; a hypothetical scheduled drift-analysis stage would too.

## Out of scope, explicitly deferred

- Distributed or remote execution.
- Continuous training on a schedule or drift trigger.
- A production scheduler integration.
- Workflow UI beyond what the chosen framework provides by default.
