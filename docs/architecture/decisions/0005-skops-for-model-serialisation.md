# ADR 0005 — `skops.io` for model serialisation

- **Status:** Accepted — 2026-04-25.
- **Deciders:** Project author (Ctrl+G review of Stage 12 plan); Stage 12 lead agent (plan [`12-serving.md`](../../plans/completed/12-serving.md) D10 — Ctrl+G reversal).
- **Related:** [`layers/models.md`](../layers/models.md), [`layers/serving.md`](../layers/serving.md), [`layers/registry.md`](../layers/registry.md), [`src/bristol_ml/models/io.py`](../../../src/bristol_ml/models/io.py), ADR [`0003`](0003-protocol-for-model-interface.md), Stage 4 plan §6 T2 (joblib decision), Stage 12 plan §1 D10.

## Context

Stage 4 introduced [`bristol_ml.models.io`](../../../src/bristol_ml/models/io.py) with `joblib` as the canonical save / load primitive for every model family. The Stage 4 plan documented a forward look — *"reconsider at Stage 12 when the serving layer lands"* — because `joblib.load` deserialises arbitrary pickle data, and a network-facing serving layer is a network-facing deserialiser.

By Stage 12 the registry held six model families (`naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) all writing `.joblib` artefacts. The Stage 12 plan's pre-Ctrl+G draft (D10) deferred the migration on the grounds that the demo runs on localhost and the threat surface was abstract. At Ctrl+G review the human reversed the deferral:

> "Include skops. This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC."

The decision is load-bearing because it propagates to every existing model family's `save` / `load` pair, to the registry's `load(run_id)` call site, and to the contract every future model family inherits — and because the migration is destructive to existing `data/registry/*.joblib` artefacts, every operator with a populated registry must retrain.

Three project constraints narrow the choice:

1. **Threat model — the serving layer is the project's first network-facing deserialiser.** `joblib.load` on an attacker-controlled artefact is trivially RCE; even on a localhost-only deployment the threat surface is non-zero (the serving process loads any artefact in the registry directory, and the registry directory is a path the operator may share). The Stage 12 plan §1 D10 records the user's Ctrl+G framing of this constraint verbatim.
2. **DESIGN §2.1.5** (idempotent operations) — re-loading an artefact must produce a deterministic, inspectable result. Whatever serialiser ships, the on-disk format must be auditable so an operator can self-diagnose a load failure without reading server logs.
3. **DESIGN §2.2.4** (complexity is earned) — the serialiser must not require model-family-specific code paths in the registry or the serving layer. A single `Model.save(path)` / `Model.load(path)` pair is the contract.

## Decision

Adopt [`skops.io`](https://skops.readthedocs.io/) as the canonical save / load primitive for every model family from Stage 12 onwards. Keep `joblib`-based helpers (`save_joblib`, `load_joblib`) in `bristol_ml.models.io` for one stage with a `DeprecationWarning`; remove them at Stage 13.

Concretely, [`bristol_ml.models.io`](../../../src/bristol_ml/models/io.py) exposes three primitives:

- `save_skops(obj, path)` — atomic skops write (tmp + `os.replace`); mirrors the ingestion-layer `_atomic_write` idiom.
- `load_skops(path)` — skops load that **enforces a project-level trust list**. `skops.io.get_untrusted_types` is invoked first; any reported type that is not registered via `register_safe_types` raises `UntrustedTypeError` naming both the artefact path and the unexpected types. The model layer's concrete classes register themselves on import.
- `register_safe_types(*qualified_names)` — adds fully-qualified class names to the project trust-list (`_PROJECT_SAFE_TYPES`).

Every model family's `save` writes a `.skops` file via `save_skops` and `load` reads it via `load_skops`. The serialised payload is an **envelope of skops-safe primitives** (bytes, str, int, float, list, dict, numpy arrays) — never a custom class instance. For the four pure-Python model families (`naive`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) the envelope is a `dict` of state and config. For `linear` and `sarimax` (whose statsmodels `Results` objects do not round-trip cleanly through skops's restricted unpickler) the envelope wraps `results.save(BytesIO)` native statsmodels bytes inside a skops-safe dict — same envelope-of-bytes pattern Stage 10/11 NN families pioneered for `torch.save(state_dict, BytesIO)`.

Because every artefact today is an envelope of primitives, `skops.io.get_untrusted_types` returns `[]` for every artefact the project produces and the trust-list gate at `load_skops` passes trivially. The trust-list mechanism is in place — but it does not *fire* until a future model family deviates from the envelope-of-primitives pattern. When that happens, the new family must call `register_safe_types("module.path.ClassName")` at import time for every custom class whose `__module__.__qualname__` would appear in `skops.io.get_untrusted_types`; otherwise `load_skops` rejects the artefact at load time with `UntrustedTypeError`, which is the correct fail-safe outcome.

## Consequences

- **Network-facing deserialiser is hardened.** The serving layer is a network-facing deserialiser; `joblib.load` on an attacker-controlled artefact is RCE. `skops.io.load(trusted=...)` refuses to materialise an unregistered custom class, so an attacker who gains write access to the registry directory cannot pivot to RCE through a crafted `.skops` artefact alone.
- **Registry boundary unchanged.** `registry.load(run_id)` still calls `Model.load(path)`; the registry layer does not need to know the serialisation format. The Stage 9 surface is preserved.
- **Breaking change for existing artefacts.** Any `data/registry/*.joblib` artefact written before Stage 12 is rejected by `registry.load` with a clear `RuntimeError`; the operator must retrain. The user explicitly accepted this trade-off at Ctrl+G review.
- **Trust-list contract for downstream stages.** Every future model family that introduces a custom class into its saved artefact must call `register_safe_types(...)` at import time. Failing to do so causes `load_skops` to raise `UntrustedTypeError` at the next service start. The contract is documented in [`models/CLAUDE.md`](../../../src/bristol_ml/models/CLAUDE.md) and cross-referenced from [`layers/serving.md`](../layers/serving.md) and [`layers/registry.md`](../layers/registry.md).
- **One-stage deprecation window.** `save_joblib` / `load_joblib` remain in `bristol_ml.models.io` with a `DeprecationWarning` so any external scripts can complete a one-off migration before Stage 13. They are removed at Stage 13 (no exceptions — joblib at the registry boundary is a security regression).
- **Statsmodels envelope-of-bytes is a generalisable pattern.** `linear` and `sarimax` use the same envelope-of-bytes pattern Stage 10/11 NN families pioneered: `inner.save(BytesIO)` to native bytes, wrap in a skops-safe `dict`. Any future model family whose internal state object cannot round-trip through skops's restricted unpickler should follow the same pattern rather than expanding the trust-list.

## Alternatives considered

- **Continue with `joblib` and rely on operational hardening (registry directory ACLs, signed artefacts).** Rejected. The project is pedagogical — operators will populate `data/registry/` with whatever they train, and the demo facilitator routinely shares `data/registry/` paths with meetup attendees. Operational hardening is the correct answer in production but the wrong answer for this codebase's threat model. Migrating to a serialiser that refuses unsafe loads at the artefact boundary is structurally simpler than locking down a directory.
- **Use `pickle` directly with `hmac` signing.** Rejected. Reproduces every `joblib` failure mode (the registry would still load arbitrary pickle data; the only protection is the signature check) and adds a key-management problem the project does not have. `skops.io` is the native Python ML answer; signing is what one does on top of an already-safe format, not in place of one.
- **ONNX as a portable format.** Rejected for Stage 12 — would require per-family converters (statsmodels has no first-class ONNX export; the SARIMAX state-space form does not map cleanly), and the demo moment does not need cross-runtime portability. Revisit if a Stage-N requirement names cross-runtime serving.
- **Defer to Stage 18 with the drift-monitoring shipping.** Rejected at Ctrl+G review. The serving layer landing without skops would mean every Stage 12 artefact is RCE-exposed for one stage; the marginal cost of migrating now (one stage's model-family churn) is lower than the marginal cost of migrating later (every operator with a populated registry retrains anyway, plus a stage-of-exposure window).

## Supersession

If a future model family's saved state cannot be expressed as an envelope of skops-safe primitives *or* an envelope-of-bytes wrapping a native serialiser, this ADR should be amended (not superseded) to add the third pattern. The trust-list mechanism is already capable of handling individual custom classes — the amendment would be a single new bullet under §Decision rather than a new ADR.

If a future stage introduces a network-facing endpoint that loads model artefacts from sources outside the project's control (e.g. an upload form), this ADR's threat model widens and a follow-up ADR should record the new boundary's harder requirements (signed artefacts, content-addressed storage, etc.).

## References

- [`skops.io` documentation](https://skops.readthedocs.io/en/stable/persistence.html) — persistence security model and trust-set semantics.
- [`docs/plans/completed/12-serving.md`](../../plans/completed/12-serving.md) §1 D10 — the human-approved decision record and Ctrl+G reversal note.
- [`docs/lld/research/12-serving-domain.md`](../../lld/research/12-serving-domain.md) §R8 — Phase-1 research that surfaced the joblib RCE vector.
- [`src/bristol_ml/models/io.py`](../../../src/bristol_ml/models/io.py) — implementation, with the load-bearing trust-list comment block.
- [`docs/architecture/layers/registry.md`](../layers/registry.md) — the consumer side of the contract.
- [`docs/architecture/layers/serving.md`](../layers/serving.md) — the security boundary documentation.
