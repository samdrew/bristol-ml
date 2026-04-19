# `bristol_ml.features` — module guide

This module is the **features layer**: functions that compose cleaned
per-source data (ingestion-layer output) into model-ready inputs. Stage 2
introduces the layer one stage earlier than DESIGN §9 implies, because
Stage 2 needs a weighted-mean function and the notebook cannot reimplement
it (§2.1.8 — notebooks are thin).

## Current surface (Stage 2)

- `bristol_ml.features.weather.national_aggregate(df, weights)` — collapse
  long-form per-station hourly weather into a wide-form national signal
  using caller-supplied weights. Honours acceptance criterion 3 (subset of
  stations) via the Mapping argument. Honours acceptance criterion 6
  (equal weights on identical inputs yield the identity) — the
  renormalised weighted mean of a constant is the constant.

## Expected additions (Stage 3)

- `bristol_ml.features.assembler.build(...)` — join demand + national
  weather + calendar features onto a canonical hourly frame. The
  assembler will consume `national_aggregate` (or import it directly).
  When Stage 3 lands, the architecture layer doc for this module is
  written in full.

## Extensibility

Each feature-producing function is pure: frame(s) in, frame out. No I/O
inside the layer — ingestion reads from the network and writes parquet;
features read from already-persisted parquet via the ingester's `load`.
Notebooks and the CLI do the wiring.

## Cross-references

- Layer contract sketch → `docs/architecture/layers/` (Stage 3 lands the
  full `features.md` when there is more than one function to describe).
- Stage 2 LLD → `docs/lld/ingestion/weather.md` §6 (the aggregator).
- Design principle §3.2 → ingestion-then-features split.
