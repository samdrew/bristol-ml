# Architecture

The canonical architecture is **[`DESIGN.md` §3](../intent/DESIGN.md#3-system-architecture)**.
This tree adds depth below that overview; it does not replace it. If
anything here disagrees with `DESIGN.md`, the spec wins — surface the
drift rather than reconcile silently.

## How this tree is organised

```
docs/architecture/
├── README.md       # this file — frame + index
├── layers/         # one file per module layer
└── decisions/      # MADR ADRs — one-off overturnable choices
```

- **`layers/<name>.md`** — a layer's public contract and internal
  design, plus the concrete modules that realise it. Written when the
  layer first lands; updated when a later stage stresses its
  conventions. A layer doc typically carries a **Contract** section
  (load-bearing for other layers; changes warrant an ADR) and an
  **Internals** section (evolves freely; audited at PR review).
- **`decisions/NNNN-*.md`** — ADRs for one-off choices. Overturn by
  supersession, not edit.

Cross-layer conventions belong in layer docs, not ADRs. A would-be ADR
that reads "how we do X everywhere" is really a layer-doc addition or
a principle in DESIGN.md §2.

## Layers

Only layers with a written doc. Unwritten layers are listed in
[`ROADMAP.md`](../../ROADMAP.md) — the architecture tree reflects what
exists, not what's planned.

| Layer | Doc |
|-------|-----|
| Ingestion | [`layers/ingestion.md`](./layers/ingestion.md) |
| Features | [`layers/features.md`](./layers/features.md) |
| Evaluation | [`layers/evaluation.md`](./layers/evaluation.md) |
| Models | [`layers/models.md`](./layers/models.md) |
| Models — NN sub-layer | [`layers/models-nn.md`](./layers/models-nn.md) |
| Registry | [`layers/registry.md`](./layers/registry.md) |

## Cross-cutting concerns

- **Configuration.** Fully specified by
  [`DESIGN.md` §7](../intent/DESIGN.md#7-configuration-and-extensibility)
  (Hydra + Pydantic, override precedence, the `Model` protocol, config
  groups). No layer doc here unless implementation ever diverges from
  §7 — the spec-drift rule says surface the divergence first, then
  decide whether to record or close.
- **Provenance.** Required of every derived artefact
  ([`DESIGN.md` §2.1.6](../intent/DESIGN.md#21-architectural)). The
  concrete form — `retrieved_at_utc` columns in raw parquet, metadata
  sidecars alongside registry entries — is documented in the layer
  that implements it.
- **Orchestration.** Manual (`python -m bristol_ml.<module>`) until
  Stage 19. Prefect flow documentation arrives with that stage.

## Decisions

Append-only by convention — supersede older ADRs rather than editing.

- [`0001-use-hydra-plus-pydantic.md`](./decisions/0001-use-hydra-plus-pydantic.md) — config framework.
- [`0002-filesystem-registry-first.md`](./decisions/0002-filesystem-registry-first.md) — registry storage before Stage 9 builds it.
- [`0003-protocol-for-model-interface.md`](./decisions/0003-protocol-for-model-interface.md) — `typing.Protocol` over `abc.ABC` for the `Model` interface.

## Changing the architecture

1. Update the relevant `layers/<name>.md` or add an ADR in
   `decisions/`.
2. Record the change in `CHANGELOG.md`.
3. If DESIGN.md §3 or §6 misrepresents the new state, update it in
   the same PR. The spec always reflects `main`.
