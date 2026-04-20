# Stages — per-stage entry point

This should now be redundant, as it has been superseded by plans/ directory.

To work on a stage, open the per-stage brief below. It names every other document to read, in order, with one line on what each contributes. That brief is the canonical entry point — not this README.

This tree is navigational only. Each brief points at four other trees without duplicating their content:

- **`docs/intent/`** — the immutable spec for what a stage does and why.
- **`docs/architecture/layers/`** — the contract a stage's module layer inherits.
- **`docs/lld/`** — first-pass design, research notes, retrospectives-at-ship.
- **`CLAUDE.md`** + **`.claude/playbook/`** — conventions and process.

## Status vocabulary

| Status | Meaning |
|---|---|
| `planning` | Intent exists; no brief; stage not yet next-up. |
| `ready` | Brief written; implementation has not started. |
| `in-progress` | Implementation underway on a `task/` branch. |
| `shipped` | Code on `main`, retrospective filed under `lld/stages/`. |

Stages 2-19 are intentionally `planning` — briefs land when each stage becomes next-up. The intent doc is enough to orient a user jumping ahead; a brief adds value only when implementation is imminent.

## Stage status

| # | Title | Status | Intent | Brief | Layer | LLD | Retro |
|---|---|---|---|---|---|---|---|
| 0 | Project foundation | `shipped` | [intent](../intent/00-foundation.md) | [brief](./00-foundation.md) | — | — | [retro](../lld/stages/00-foundation.md) |
| 1 | NESO demand ingestion | `shipped` | [intent](../intent/01-neso-demand-ingestion.md) | [brief](./01-neso-demand-ingestion.md) | [ingestion](../architecture/layers/ingestion.md) | [lld](../lld/ingestion/neso.md) | [retro](../lld/stages/01-neso-demand-ingestion.md) |
| 2 | Weather ingestion | `shipped` | [intent](../intent/02-weather-ingestion.md) | [brief](./02-weather-ingestion.md) | [ingestion](../architecture/layers/ingestion.md) | [lld](../lld/ingestion/weather.md) | [retro](../lld/stages/02-weather-ingestion.md) |
| 3 | Feature assembler + split | `shipped` | [intent](../intent/03-feature-assembler.md) | [plan](../plans/completed/03-feature-assembler.md) | [features](../architecture/layers/features.md) · [evaluation](../architecture/layers/evaluation.md) | — | [retro](../lld/stages/03-feature-assembler.md) |
| 4 | Linear baseline + eval harness | `shipped` | [intent](../intent/04-linear-baseline.md) | [plan](../plans/completed/04-linear-baseline.md) | [models](../architecture/layers/models.md) · [evaluation](../architecture/layers/evaluation.md) | — | [retro](../lld/stages/04-linear-baseline.md) |
| 5 | Calendar features | `shipped` | [intent](../intent/05-calendar-features.md) | [plan](../plans/completed/05-calendar-features.md) | [features](../architecture/layers/features.md) · [ingestion](../architecture/layers/ingestion.md) | — | [retro](../lld/stages/05-calendar-features.md) |
| 6 | Enhanced evaluation & viz | `planning` | [intent](../intent/06-enhanced-evaluation.md) | — | — | — | — |
| 7 | SARIMAX | `planning` | [intent](../intent/07-sarimax.md) | — | — | — | — |
| 8 | SciPy parametric | `planning` | [intent](../intent/08-scipy-parametric.md) | — | — | — | — |
| 9 | Model registry | `planning` | [intent](../intent/09-model-registry.md) | — | — | — | — |
| 10 | Simple NN | `planning` | [intent](../intent/10-simple-nn.md) | — | — | — | — |
| 11 | Complex NN | `planning` | [intent](../intent/11-complex-nn.md) | — | — | — | — |
| 12 | Serving endpoint | `planning` | [intent](../intent/12-serving.md) | — | — | — | — |
| 13 | REMIT ingestion | `planning` | [intent](../intent/13-remit-ingestion.md) | — | [ingestion](../architecture/layers/ingestion.md) | — | — |
| 14 | LLM extractor | `planning` | [intent](../intent/14-llm-extractor.md) | — | — | — | — |
| 15 | Embedding index | `planning` | [intent](../intent/15-embedding-index.md) | — | — | — | — |
| 16 | Model with REMIT features | `planning` | [intent](../intent/16-model-with-remit.md) | — | — | — | — |
| 17 | Price pipeline | `planning` | [intent](../intent/17-price-pipeline.md) | — | [ingestion](../architecture/layers/ingestion.md) | — | — |
| 18 | Drift monitoring | `planning` | [intent](../intent/18-drift-monitoring.md) | — | — | — | — |
| 19 | Orchestration | `planning` | [intent](../intent/19-orchestration.md) | — | — | — | — |

## Authoring a new brief

When a stage becomes next-up, copy the shape of [`01-neso-demand-ingestion.md`](./01-neso-demand-ingestion.md):

- **Header:** status, intent link, dependency stages (upstream / downstream).
- **Reading order:** numbered list of docs, each with one sentence on what it contributes. Intent first, then DESIGN sections, then layer architecture, then research, then LLD, then CLAUDE.md, then playbook.
- **Acceptance criteria:** quoted from the intent (with a "intent wins on drift" caveat).
- **Files expected to change:** split into *New* and *Modified*. The "Modified" list usually includes `conf/config.yaml`, `pyproject.toml`, `CHANGELOG.md`, `README.md`, `docs/intent/DESIGN.md` §6, and this table's status cell.
- **Exit criteria:** PR checklist mapping to stage hygiene in `CLAUDE.md` + `DESIGN.md` §9.
- **Team-shape recommendation:** default team vs single session; when to spawn researcher / tester in parallel.

Briefs are `ALLOW` tier — freely mutable while the stage is in flight. This README index is `WARN` tier — edit it when a stage transitions between status values, otherwise leave it alone.
