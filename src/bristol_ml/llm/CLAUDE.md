# `bristol_ml.llm` — module guide

> **Status (T2 of Stage 14):** *skeleton.* The Pydantic boundary types
> and `Extractor` Protocol are in place; the concrete implementations
> (`StubExtractor`, `LlmExtractor`), the gold-set fixture, the
> evaluation harness, and the cassette-refresh ritual all land in
> later tasks. This file will be filled out at T7.

This module is the **LLM feature-extractor layer**: a typed boundary
(`RemitEvent` → `ExtractionResult`) plus two interchangeable backends
selected by config + env var. Stage 14 introduces the layer; Stages
15 (embedding index) and 16 (feature-table join) consume it.

Read the layer contract in
[`docs/architecture/layers/llm.md`](../../../docs/architecture/layers/llm.md)
once T7 lands; the file you are reading documents the concrete
Stage 14 surface.

## Current surface (Stage 14 — T2)

The package `__init__.py` exposes the **public boundary** only — the
two concrete extractors live in a sibling module so callers
(Stage 15, Stage 16) can depend on the boundary without importing
either backend.

- `bristol_ml.llm.RemitEvent` — typed mirror of the extraction-relevant
  subset of Stage 13's `OUTPUT_SCHEMA` row. UTC-aware datetimes,
  `extra="forbid"`, `frozen=True`.
- `bristol_ml.llm.ExtractionResult` — structured features extracted
  from a single `RemitEvent`. Carries provenance (`prompt_hash`,
  `model_id`) so each result is traceable to the prompt + model
  that produced it.
- `bristol_ml.llm.Extractor` — `runtime_checkable` `Protocol` with
  exactly two methods (`extract`, `extract_batch`). AC-1 caps the
  surface at this size — *"the interface is small enough that
  writing a third implementation in the future is plausible"*.

## Cross-references

- Layer contract — `docs/architecture/layers/llm.md` (lands at T7).
- Stage 14 plan — [`docs/plans/active/14-llm-extractor.md`](../../../docs/plans/active/14-llm-extractor.md).
- Intent — [`docs/intent/14-llm-extractor.md`](../../../docs/intent/14-llm-extractor.md).
- Scope Diff — [`docs/lld/research/14-llm-extractor-scope-diff.md`](../../../docs/lld/research/14-llm-extractor-scope-diff.md).
