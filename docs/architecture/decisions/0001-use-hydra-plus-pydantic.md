# ADR 0001 — Use Hydra for composition and Pydantic for validation

- **Status:** Accepted — 2026-04-18
- **Deciders:** Project author
- **Related:** `DESIGN.md` §7

## Context

The project is a reference ML architecture whose whole point is pluggable components: swappable models, ingestion sources, evaluation configs. This forces three simultaneous requirements on the configuration layer:

1. Config groups with CLI override (`model=sarimax`, `evaluation.folds=12`).
2. Strong type validation at the code boundary, so a YAML typo fails loudly rather than silently coercing.
3. A clean code/config boundary so that notebooks and tests can load configs without reaching into framework internals.

## Decision

Use a two-stage pipeline:

1. **Hydra** composes YAML files from `conf/`, applying CLI overrides and interpolations.
2. `OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)` converts the resolved `DictConfig` into a plain `dict`.
3. **Pydantic** (`AppConfig.model_validate(data)`) validates types, ranges, and cross-field invariants.
4. Application code only ever sees the Pydantic model — never `DictConfig`.

The schema is in `conf/_schemas.py`. The glue is `bristol_ml.config`.

## Consequences

- Downstream code is typed end-to-end; editor tooling works.
- YAML is the only persisted config form. Schema edits are a code change — reviewable.
- `extra="forbid"` makes typos loud; the Stage 0 test suite exercises this path.
- Two frameworks to learn instead of one. Mitigated by the fact that most users only touch YAML.

## Alternatives considered

- **Pydantic Settings alone.** No config groups. Models-as-plugins (§7.3 / §7.5) would require hand-rolled dispatch logic.
- **Hydra alone.** Type validation is too loose — an `int` field silently accepts a string; cross-field invariants are not expressible without custom resolvers.
- **hydra-zen.** Python-native configs obscure the config/code separation the project is teaching.
- **OmegaConf alone.** No CLI override story worth the name.

## References

- `DESIGN.md` §7
- [Hydra docs](https://hydra.cc/)
- [Pydantic v2 docs](https://docs.pydantic.dev/)
