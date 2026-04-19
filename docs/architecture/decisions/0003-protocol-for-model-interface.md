# ADR 0003 — `typing.Protocol` for the `Model` interface

- **Status:** Accepted — 2026-04-19.
- **Deciders:** Project author; Stage 4 lead agent (plan [`04-linear-baseline.md`](../../plans/completed/04-linear-baseline.md) D3).
- **Related:** `DESIGN.md` §7.3 (the `Model` sketch), [`layers/models.md`](../layers/models.md), [`src/bristol_ml/models/protocol.py`](../../../src/bristol_ml/models/protocol.py), PEP 544.

## Context

Stage 4 introduces the `Model` interface that every subsequent modelling stage (5, 7, 8, 10, 11) must implement and that the Stage 9 registry must be able to load. The interface needs exactly five members: `fit`, `predict`, `save`, `load`, and a `metadata` property. Two mechanisms are idiomatic Python:

- `typing.Protocol` + `@runtime_checkable` — structural subtyping per PEP 544. A class implements the protocol simply by exposing the right attributes; inheritance is not required.
- `abc.ABC` with `@abstractmethod` — nominal subtyping. Classes opt in by inheritance and the ABC machinery raises `TypeError` if a subclass fails to override.

Both satisfy the Stage 4 acceptance criterion AC-2 ("interface implementable in very few lines"). The decision is load-bearing because it propagates to every future model class and to the Stage 9 registry's loader.

Three project constraints narrow the choice:

1. **Principle 2.1.2** (typed narrow interfaces) — the contract must be inspectable at development time by a type checker.
2. **Principle 2.2.4** (complexity is earned) — avoid machinery whose benefit is not concrete.
3. **DESIGN §7.3** sketches a `Protocol`-shaped definition verbatim. Divergence from the spec would need its own justification.

## Decision

Use `typing.Protocol` + `@runtime_checkable`. The protocol lives in `src/bristol_ml/models/protocol.py` alongside the `ModelMetadata` re-export. Every concrete model class (Stage 4's `NaiveModel` and `LinearModel`; Stages 5, 7, 8, 10, 11 onwards) implements the five members directly; no base class is inherited.

```python
@runtime_checkable
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

A runtime `isinstance(m, Model)` check verifies the presence of the five attributes but **not** their signatures. This caveat is documented on the protocol's docstring and in [`layers/models.md`](../layers/models.md) — it is the PEP 544 price of admission, not a bug.

## Consequences

- **Structural subtyping.** Any class exposing the five members satisfies `isinstance(m, Model)`. A wrapper that adapts a third-party library (future scikit-learn stage, if it ever lands) does not need to inherit from anything.
- **Static type checking catches signature mismatches.** Mypy / pyright check the full signature at development time; runtime `isinstance` does not. The combination is the standard PEP 544 pattern.
- **Matches DESIGN §7.3 verbatim.** The spec's sketch shipped unmodified, so §7.3 and the implementation do not drift.
- **Cheap evolution of the contract.** Adding a sixth member (e.g. a Stage 9 `checksum` property) is a single edit to `protocol.py`; `runtime_checkable` picks up the new attribute automatically on every implementor.
- **No base-class constructor to route around.** `NaiveModel.__init__(config)` and `LinearModel.__init__(config)` take their family-specific configs directly; they are not constrained by a shared `Model.__init__` signature.

## Alternatives considered

- **`abc.ABC` + `@abstractmethod`.** Rejected. Forces every family to `class NaiveModel(Model):`; noisy when the concrete class already has family-specific base-class needs (Stage 10's NN family will want a `torch.nn.Module` base). `@runtime_checkable` gives the `isinstance` ergonomics without the inheritance constraint.
- **Free-functions dispatched by a registry (`REGISTRY: dict[str, ModelFactory]`).** Rejected. Would duplicate what Hydra's `_target_` pattern already provides, introduce a second place for "which models exist", and push state management outside the class that owns it.
- **No formal interface; rely on duck typing + documentation.** Rejected. DESIGN §7.3 demands a named protocol so the evaluation harness's typed signature is meaningful. Without it, `evaluate(model: Model, ...)` becomes `evaluate(model: Any, ...)` and the Stage 9 registry has no loader contract.

## Supersession

If a future stage introduces model variants that fundamentally cannot implement `fit(features: pd.DataFrame, target: pd.Series) -> None` (e.g. generative models with no supervised target), this ADR should be superseded by a new one that widens or splits the protocol — not edited in place. The `@runtime_checkable` cost of adding a sibling protocol (`ForecastModel`, `GenerativeModel`) is low.

## References

- PEP 544 — Protocols: Structural subtyping (static duck typing). <https://peps.python.org/pep-0544/>
- `DESIGN.md` §7.3 — the original `Model` sketch.
- [`layers/models.md`](../layers/models.md) — the Stage 4 layer architecture that consumes this decision.
- [`docs/plans/completed/04-linear-baseline.md`](../../plans/completed/04-linear-baseline.md) §1 D3 — the human-approved decision record and alternatives rationale.
