# Architecture

The canonical, high-level architecture description is **[`DESIGN.md` §3](../intent/DESIGN.md#3-system-architecture)**. This tree adds depth below that overview without duplicating it. If anything here disagrees with `DESIGN.md`, the spec wins — surface the drift, do not rewrite it silently.

## How this tree is organised

```
docs/architecture/
├── README.md              # this file — overarching frame + index
├── layers/                # one file per module layer
└── decisions/             # MADR ADRs — one-off, justifiable choices
```

- **`layers/<name>.md`** — the architecture of one module layer. Contains its public-interface contract, cross-module conventions, swappable vs load-bearing parts, and the list of concrete modules that implement it. Written when the layer first lands (one module in it); updated when a new module in the same layer stresses the conventions.
- **`decisions/NNNN-*.md`** — ADRs, in MADR format, for one-off choices whose justification does not belong inside a layer doc. A good ADR is _overturnable_ (supersede with a later ADR) and _one-off_ (not "how we do X across N layers"). Cross-layer conventions live in the layer docs, not here.

If a would-be ADR is really "the contract for layer Y", it is a layer doc. If a would-be layer doc is really one weighing-up of alternatives that could have gone either way, it is an ADR.

## Layer-doc index

One row per module layer from DESIGN §3.2. Layers without a doc have not yet had a stage that exercises them — the tree is deliberately honest (empty until earned) rather than suggestive (stubs hinting at eventual content).

| Layer | Doc | Status | First stage |
|-------|-----|--------|-------------|
| Ingestion | [`layers/ingestion.md`](./layers/ingestion.md) | Provisional — Stage 1 only | 1 |
| Features | — | Arrives with Stage 3 | 3 |
| Models | — | Arrives with Stage 4 (introduces the `Model` protocol) | 4 |
| Evaluation | — | Arrives with Stage 4 | 4 |
| Registry | — | Arrives with Stage 9 | 9 |
| Serving | — | Arrives with Stage 12 | 12 |
| LLM | — | Arrives with Stage 14 | 14 |
| Monitoring | — | Arrives with Stage 18 | 18 |

## Cross-cutting concerns

- **Configuration.** Covered by [`DESIGN.md` §7](../intent/DESIGN.md#7-configuration-and-extensibility) (Hydra + Pydantic pattern, override precedence, the `Model` interface, config groups, adding a new model). No separate layer doc exists here because the §7 treatment is complete. A `layers/configuration.md` lands only if implementation ever diverges from §7 — the spec-drift rule says surface the divergence first, then decide whether to record or to close.
- **Orchestration.** Prefect-based flow arrives at Stage 19. Until then, orchestration is `python -m bristol_ml.<module>` run by hand; there is nothing to document.
- **Provenance.** Every derived artefact records its inputs, git SHA, and wall-clock timestamp (§2.1.6). The concrete form — `retrieved_at_utc` columns in raw parquet, metadata sidecars in the registry — is documented per-layer where it lands.

## Decisions

- [`decisions/`](./decisions/) — MADR ADRs. Append-only by convention: supersede older ADRs with new ones rather than editing.
  - [`0001-use-hydra-plus-pydantic.md`](./decisions/0001-use-hydra-plus-pydantic.md) — config framework choice.
  - [`0002-filesystem-registry-first.md`](./decisions/0002-filesystem-registry-first.md) — registry storage choice before Stage 9 builds it.

When the architecture changes, update the relevant layer doc or add an ADR, record the change in `CHANGELOG.md`, and if the change alters the spec surface of `DESIGN.md` §3 or §6 update that too.
