# ADR 0002 — Filesystem registry before MLflow / W&B

- **Status:** Accepted (provisional) — 2026-04-18. Revisit at Stage 9 when the registry is actually built.
- **Deciders:** Project author
- **Related:** `DESIGN.md` §8 (technology table), §10 ("Registry graduation"), §3.2 (registry paragraph)

## Context

Stage 9's demo moment is `python -m bristol_ml registry list` showing a leaderboard across techniques. Two of the project's architectural principles constrain how the registry is built:

- **Principle 2.1.1** (every component runs standalone) — no required running service.
- **Principle 2.2.4** (complexity is earned) — advanced patterns appear only when their purpose is concrete.

A leaderboard over a handful of models is a solved problem at the filesystem level. Introducing a tracking server at Stage 9 adds operational complexity that the pedagogical payoff does not yet demand.

## Decision

The registry is a directory of artefacts with sidecar JSON metadata. The interface (`save`, `load`, `list`, `describe`) is what every downstream stage depends on — not the storage mechanism. Graduation to MLflow or W&B is an adapter behind the same interface, not a rewrite.

## Consequences

- Zero services to run for a live demo — aligned with principle 2.1.1.
- Artefact paths and metadata are inspectable with `ls`, `cat`, `jq` — pedagogically transparent.
- Migration path to MLflow preserves the interface; the work is a new implementation, not a refactor of callers.
- No built-in multi-user support. Acceptable at this maturity level (§3.3: "multi-user serving … out of scope").

## Alternatives considered

- **MLflow local tracking server.** Requires a running process; contradicts principle 2.1.1 for the base demo. Sensible graduation target once metadata volume or query requirements justify it.
- **Weights & Biases.** Requires an account and network — contradicts principle 2.1.3 (stub-first for flaky dependencies) at the base layer.
- **SQLite index.** Adds a schema-migration burden and a second storage format with no concrete pedagogical payoff at Stage 9.

## References

- `DESIGN.md` §3.2, §8, §10.
