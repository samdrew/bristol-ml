# ADR 0007 — The ingestion-layer public contract bifurcates by data shape

- **Status:** Accepted — 2026-04-26.
- **Deciders:** Project author; Stage 13 lead agent; Phase-3 `@arch-reviewer` (review finding B-1).
- **Related:** [`layers/ingestion.md`](../layers/ingestion.md), [`src/bristol_ml/ingestion/remit.py`](../../../src/bristol_ml/ingestion/remit.py), [`docs/plans/completed/13-remit-ingestion.md`](../../plans/completed/13-remit-ingestion.md) §1 D13 / D17, [`docs/lld/research/13-remit-ingestion-scope-diff.md`](../../lld/research/13-remit-ingestion-scope-diff.md).

## Context

The ingestion layer carried, at Stages 1–5 / 12, an exclusive contract clause in [`layers/ingestion.md`](../layers/ingestion.md):

> *"Every module exposes **exactly two** public callables: `fetch(config, *, cache=CachePolicy.AUTO) -> Path` and `load(path) -> pd.DataFrame`."*

Five ingesters shipped under this clause without strain — `neso.py` (demand), `weather.py`, `holidays.py`, `neso_forecast.py`, plus the Stage 9 surface that loaded these caches. Each carries *level* data: one row per asset per timestamp, where the value at that timestamp is the entire payload. The single-timestamp convention (`timestamp_utc` column at `timestamp[us, tz=UTC]`) reduces every temporal query to a one-line pandas predicate the caller writes inline.

Stage 13 introduced REMIT, the project's first *event-log* ingester. REMIT messages carry three logically distinct times per row (`published_at` for transaction-time; `effective_from` / `effective_to` for valid-time) plus the standard `retrieved_at_utc` provenance scalar. The Stage 13 plan (D8 / D13) added a third public callable to `bristol_ml.ingestion.remit`:

```python
def as_of(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Active state as known to the market at time t (transaction-time only)."""
```

`as_of` is three lines of pandas — filter `published_at <= t`, `groupby(mrid).idxmax(revision_number)`, drop `Withdrawn` — but it answers the bi-temporal question correctly *by construction* and prevents the silent leakage failure mode a naive "latest revision wins" approach exhibits. The arch-reviewer's Phase-3 finding (B-1) flagged that adding `as_of` to REMIT's public surface violated the layer's "exactly two" clause as written. Stages 14 (LLM extraction) and 16 (REMIT × feature table join) will consume `as_of` directly, making it part of a real cross-stage contract — not a REMIT internal.

Three project constraints narrow the choice of how to reconcile:

1. **DESIGN §2.1.2** (typed narrow interfaces) — adding a verb to the *shared* surface across all five ingesters means each gets a function whose only meaningful definition is the trivial pandas predicate, *and* the layer pays the cost of cross-module type pinning, tests, and documentation for that triviality.
2. **DESIGN §2.1.7** (tests at boundaries, not everywhere) — uniformly adding `as_of` to NESO / weather / holidays would generate four test files of degenerate "yes, the predicate is `<=`" tests that buy nothing.
3. **The layer doc is consumed by future stage authors** — a future event-log source (rare, but possible: market sentiment feeds, certificate trades) needs a clear rule for what it must expose. Leaving the contract as "exactly two" with REMIT silently breaking it produces the worst of both worlds: a documented rule that nobody can rely on.

The decision shapes downstream consumer code (Stage 14 / Stage 16 import paths and call patterns), the layer doc's contract section, and the rule a future event-log author follows when deciding whether their new query primitive belongs on the shared surface or stays module-local.

## Decision

The ingestion-layer public contract **bifurcates by the data shape an ingester targets**:

| Data shape | Public surface | Examples |
|------------|---------------|----------|
| **Level data** — one row per asset per timestamp; single canonical `timestamp_utc` column. | `fetch` + `load`. Temporal queries are one-line pandas predicates the caller writes inline. | NESO demand, NESO forecast, weather, holidays — every Stage 1–5 ingester. |
| **Event log** — append-only multi-temporal records (`(entity_id, revision_number)` grain plus three or more publish-axis timestamps). | `fetch` + `load` + at least one *temporal-query primitive* (e.g. `as_of(df, t)`). The query primitive is mandatory, not optional, because the temporal correctness of the data shape depends on it. | REMIT — the only Stage 0–13 example. |

The trigger that makes a new ingester an *event log* (and therefore mandates a query primitive) is the presence of **multiple temporal axes per row** — minimum: a transaction-time axis (when the message was disclosed) plus a valid-time axis (when the event covered). A single-timestamp source remains level data even if its rows are append-only at the storage level.

Concretely:

- **`fetch` and `load` remain the load-bearing core** — every ingester exposes them. Their signatures are pinned in [`layers/ingestion.md`](../layers/ingestion.md) §"Public interface" exactly as before.
- **`as_of(df, t)` is the canonical temporal-query primitive for event-log ingesters.** REMIT exposes it. A future event-log ingester that ships without an equivalent query primitive is in violation of the contract.
- **Level-data ingesters are *not* required to expose `as_of`.** Their callers compose the equivalent predicate inline; promoting it to a shared verb adds boilerplate without eliminating any failure mode.
- **The layer doc names both rows of the table.** A future author scanning the layer doc sees the bifurcation immediately and either picks "level" (the common case) or "event-log" (the rare case) explicitly.
- **Module-internal helpers stay module-internal.** This ADR governs cross-stage contract verbs only — any `_private_helper` an ingester defines for its own use does not need to mirror across ingesters.

## Consequences

- **Plan D17 (no ADR for bi-temporal storage shape) is unaffected.** That deferral was about the storage shape — four-timestamp columns + `(mrid, revision_number)` grain — and the design lives in [`layers/ingestion.md`](../layers/ingestion.md) §"Bi-temporal storage shape". This ADR is about a *different* question: extending the cross-stage public-surface contract. Different load-bearing concern, different artefact.
- **Layer doc edits.** [`layers/ingestion.md`](../layers/ingestion.md) §"Public interface" replaces the "exactly two" clause with the bifurcated table above and a back-reference to this ADR. The Upgrade-seams table (load-bearing column) remains `fetch` / `load` / `CachePolicy` for every ingester; the temporal-query primitive is added as an event-log-specific load-bearing surface.
- **Stage 14 and Stage 16 read this ADR before importing `as_of`.** The plan's Reading-order section already names `layers/ingestion.md`; that doc now points here for the contract rationale. Consuming code can rely on `as_of` having stable semantics and a cross-stage contract behind it, not just a REMIT-author convention.
- **Future event-log ingesters inherit the contract.** A hypothetical "balancing-mechanism trade events" ingester (Stage 17 territory, but the prices/generation feeds are level data — true event-log feeds are rare in this corpus) would be required to expose `as_of` or document why the data shape is level despite append-only storage.
- **No retroactive change to NESO / weather / holidays / neso\_forecast.** They keep their two-callable surface. No code change, no test change, no doc change beyond the layer doc's contract clause.
- **Plan D13's "module-level function, no class, no index" guidance carries forward** as a sub-decision under this ADR. Future event-log query primitives should follow the same pattern (pure pandas function, three lines, no cached state) unless the data volume forces an indexed structure — in which case the ADR is superseded.

## Alternatives considered

- **Keep "exactly two" by demoting `as_of` to module-internal.** Rejected. `as_of` is the new public primitive Stage 13 introduces and the pedagogical surface for the bi-temporal teaching point. Hiding it behind module-private dotpath access (`bristol_ml.ingestion.remit.as_of`) is technically defensible but undersells the structural feature: the bifurcation between level and event-log ingesters is real, exists in the code, and pretending otherwise produces a layer doc that disagrees with the implementation.
- **Widen "exactly two" to "exactly three" and add `as_of` to every ingester.** Rejected per DESIGN §2.1.7 — the four level-data ingesters would each get a function whose body is `return df[df.timestamp_utc <= t]`, with a test file pinning that triviality. Five files of boilerplate, no failure mode eliminated. The Scope Diff's `PREMATURE OPTIMISATION` framing applies to layer doc rules as much as to code.
- **Move `as_of` and the bi-temporal storage shape into a new `bristol_ml.events` layer.** Rejected. REMIT is structurally an *ingester* — it owns HTTP, parsing, parquet writes, the cache discipline. Splitting the bi-temporal query primitive into a sibling layer would mean either (a) `bristol_ml.events.as_of` reaches into `bristol_ml.ingestion.remit` to read its parquet, breaking the layer-isolation rule, or (b) REMIT writes parquet under `bristol_ml.events`, which makes the layer name a lie about what's inside. The layered architecture's cost outweighs its benefit when there is exactly one event-log source.
- **Defer the contract decision to Stage 14.** Rejected. The Stage-14 author should not have to revisit "is `as_of` part of the cross-stage contract or a REMIT quirk?" before they can write code that consumes it. Resolving the question now, with full Stage 13 context fresh, is cheaper than re-litigating it from inside Stage 14's plan.

## Supersession

If a future stage introduces a second temporal-query primitive (e.g. `valid_at(df, t)` for a strict valid-time filter, distinct from `as_of`'s transaction-time semantics) or an event-log source whose volume forces an indexed query structure rather than the three-line pandas pattern, this ADR should be superseded by a new one recording the expanded primitive list and the indexing requirement. The current decision is correct for the single-event-log-source state of the project and should not be assumed to scale beyond it.

## References

- [`docs/plans/completed/13-remit-ingestion.md`](../../plans/completed/13-remit-ingestion.md) §1 D13 (`as_of` is a module-level pandas function), §1 D17 (no ADR for bi-temporal storage shape — different concern from this ADR).
- [`docs/lld/research/13-remit-ingestion-scope-diff.md`](../../lld/research/13-remit-ingestion-scope-diff.md) — the `@minimalist` Phase-1 critique. Did not flag the layer-contract clause as needing an update; the gap surfaced in Phase 3 review (B-1).
- [`docs/lld/research/13-remit-ingestion-domain.md`](../../lld/research/13-remit-ingestion-domain.md) §R4 — bi-temporal as-of mechanics; the standard two-step "transaction-time then valid-time" decomposition pattern.
- [`docs/architecture/layers/ingestion.md`](../layers/ingestion.md) §"Public interface" + §"Bi-temporal storage shape" — the layer doc that consumes this decision.
- [`src/bristol_ml/ingestion/remit.py`](../../../src/bristol_ml/ingestion/remit.py) — the implementation, including the `as_of` function body and the `__all__` export list.
- DESIGN §2.1.2 (typed narrow interfaces), §2.1.7 (tests at boundaries) — the principles this decision honours.
