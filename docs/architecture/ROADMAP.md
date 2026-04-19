# Architecture roadmap

Planned layers and cross-cutting concerns that DESIGN.md §3.2 promises
but no stage has yet realised. The architecture index in
[`README.md`](./README.md) lists only what exists; this file lists what
is coming and the architectural questions each will raise.

Stage sequencing and status live in
[`DESIGN.md` §9](../intent/DESIGN.md#9-stage-plan). This file is about
*architecture*, not schedule — a layer appears here because it needs
design work before it can land, regardless of when its stage is
scheduled.

## Planned layers

Each entry names the layer, the stage that first realises it, and the
architectural questions that must be resolved when the layer doc is
written. The questions are prompts for the Phase 1 plan that
introduces the layer — not commitments, not open questions against
current architecture.

### Registry
*First realised by [Stage 9](../intent/DESIGN.md#9-stage-plan).*

Filesystem-first with an MLflow graduation path
([ADR 0002](./decisions/0002-filesystem-registry-first.md)). Questions:

- Artefact layout on disk — directory shape, metadata sidecar format,
  naming of runs vs models.
- Retrofit path — how the four models from Stages 4, 7, 8 save through
  the registry without forcing edits across all four.
- CLI surface — the `python -m bristol_ml registry list` command is
  promised in DESIGN.md §9 Stage 9 but its shape is undesigned.

### Serving
*First realised by [Stage 12](../intent/DESIGN.md#9-stage-plan).*

FastAPI wrapping registry artefacts. Questions:

- Model selection at request time — path param, config, or registry
  alias.
- Feature computation at serve time — does serving call into the
  features layer, or does it expect pre-computed inputs.
- Error taxonomy for unknown models, stale artefacts, malformed
  input.

### LLM
*First realised by [Stage 14](../intent/DESIGN.md#9-stage-plan), preceded
by REMIT ingestion in Stage 13.*

Extractor and embedding index for REMIT. Questions:

- Extractor interface shape — DESIGN.md §7.3's `Model` protocol is for
  forecasters; the extractor's contract is different and needs its own
  protocol.
- Stub vs real parity — how the hand-labelled stub and Claude-backed
  implementation are kept interface-compatible and jointly tested.
- Embedding index graduation — numpy-first, FAISS when scale demands
  it (§8). The trigger condition needs to be written down, not
  felt-out.
- Cost control — where rate-limiting, caching, and the "default to
  stub" rule are enforced.

### Monitoring
*First realised by [Stage 18](../intent/DESIGN.md#9-stage-plan).*

Drift and prediction-quality tracking. Questions:

- What the monitoring layer is a library of vs what it *owns* — PSI
  computation is a utility; the decision of "which features to
  monitor" is a contract with the features layer.
- Output surface — notebook dashboard, registry annotations, or both.
- Relationship with evaluation — rolling-origin metrics and drift
  metrics share machinery; the split between retrospective evaluation
  and ongoing monitoring is not drawn.

## Planned cross-cutting concerns

### Orchestration
*First realised by [Stage 19](../intent/DESIGN.md#9-stage-plan).*

Prefect flow chaining ingestion → features → training → evaluation →
registry write. Questions:

- Where the flow definition lives — `orchestration/` as a peer to the
  module layers, or embedded in the stage that introduces it.
- How a facilitator runs a single component standalone (DESIGN.md
  §2.1.1) while the orchestrated flow exists — the two invocation
  paths must not duplicate logic.
- Whether Stage 19 builds at all. DESIGN.md §11 flags this as an open
  question.

## Deferred indefinitely

Tracked here so the roadmap is complete. Each becomes an ADR when a
stage forces the decision; until then, none has a planned layer doc.

Full list in [`DESIGN.md` §10](../intent/DESIGN.md#10-deferred-decisions).
Headline items: MLflow/W&B graduation, feature store as a service,
CI-driven retraining, real-time serving, probabilistic forecasting.

## Maintenance

- When a layer lands, move its entry out of this file and into the
  layer index in [`README.md`](./README.md).
- When a question raised here is answered by a plan or ADR, strike it
  from the list and cite the resolution.
- This file should shrink over the project's life. If it's growing,
  the architecture is sprawling faster than it's being realised.
