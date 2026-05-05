# LLD — low-level design

Lead and teammates write here freely.  Contents are either
append-only retrospective or discardable scratch — nothing in this
tree is load-bearing for users or for downstream stages.

**Per-stage entry points live in
[`../plans/`](../plans/).**  To work on a stage, read its plan
under `docs/plans/active/` — that document names everything here
(research, retrospective, reframings) in reading order.

```
docs/lld/
├── README.md       # this file
├── stages/         # per-stage retrospectives — one file per completed stage, named NN-slug.md
├── research/       # @researcher / @domain-researcher output (on demand)
└── reframings/     # @reframer output (on demand; written when stuck on the same framing)
```

- [`stages/`](./stages/) — per-stage retrospectives.  Template at
  `stages/00-foundation.md` (the Stage 0 foundation retro shipped
  with the template).
- `research/` — written before implementation when the right
  approach is unclear.  Findings become input to the
  implementation team's spawn prompts.
- `reframings/` — alternative problem framings (no solutions),
  written by the reframer agent when an implementation task has
  failed three or more times with the same framing.

See [`../architecture/`](../architecture/) for structural
decisions and [`../intent/DESIGN.md`](../intent/DESIGN.md) for the
project spec.
