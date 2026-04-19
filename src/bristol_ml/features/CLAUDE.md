# `bristol_ml.features` — module guide

This module is the **features layer**: functions that compose cleaned
per-source data (ingestion-layer output) into model-ready inputs. Stage 2
introduces the layer one stage earlier than DESIGN §9 implies, because
Stage 2 needs a weighted-mean function and the notebook cannot reimplement
it (§2.1.8 — notebooks are thin).

## Current surface (Stage 3)

### `weather.national_aggregate(df, weights)` (Stage 2)

Collapse long-form per-station hourly weather into a wide-form national
signal using caller-supplied weights. Honours acceptance criterion 3
(subset of stations) via the Mapping argument. Honours acceptance
criterion 6 (equal weights on identical inputs yield the identity) —
the renormalised weighted mean of a constant is the constant.

### `assembler` (Stage 3)

Public surface:

- `OUTPUT_SCHEMA: pa.Schema` — the declared parquet schema. Column order,
  arrow dtypes and timezone metadata are contractual; downstream models
  may rely on all three.
- `build(demand_hourly, weather_national, config, *, neso_retrieved_at_utc=None,
  weather_retrieved_at_utc=None) -> pd.DataFrame` — inner-join demand with
  national weather, forward-fill weather up to
  `config.forward_fill_hours`, drop remaining NaN rows, project to
  `OUTPUT_SCHEMA` column order. Emits a single structured INFO log line
  per call (D5) with counts of rows dropped/filled at each step.
- `load(path: Path) -> pd.DataFrame` — schema-validated read; refuses
  missing or extra columns. Mirrors `neso.load` / `weather.load`.
- `assemble(cfg: AppConfig, cache="offline") -> Path` — one-shot
  orchestrator that ties `neso.fetch/load → _resample_demand_hourly →
  weather.fetch/load → national_aggregate → build → _atomic_write`. Used
  by the CLI.
- `_resample_demand_hourly(df, agg: Literal["mean", "max"] = "mean") ->
  pd.DataFrame` — floor `timestamp_utc` to the hour and aggregate
  `nd_mw` / `tsd_mw`. Module-private; exposed for testing and for the
  `assemble()` orchestrator. On clock-change days the output is
  23 rows (spring) or 25 rows (autumn) — the UTC timeline is regular;
  the NESO ingester has already unwound DST algebra.

## Invariants (load-bearing for Stage 4 onwards)

The assembler **guarantees** that every `build()` output:

- Has columns exactly equal to `OUTPUT_SCHEMA.names`, in the same order.
- Has a tz-aware UTC `timestamp_utc` column, strictly monotonically
  ascending, unique.
- Has `int32` demand columns (`nd_mw`, `tsd_mw`) and `float32` weather
  columns (cloud_cover widens from `int8` on the source schema — see
  `WEATHER_VARIABLE_COLUMNS` docstring).
- Carries two scalar provenance columns (`neso_retrieved_at_utc`,
  `weather_retrieved_at_utc`) — constant across rows within a single
  `build()` call, per DESIGN §2.1.6.
- Contains no NaN values anywhere: demand-NaN rows are dropped; weather
  gaps shorter than `forward_fill_hours` are filled; longer gaps drop
  the row.

If a change breaks any of these, fix the test only if the invariant itself
is wrong — do not weaken the test to make it pass.

## Expected additions (Stage 5)

- A second feature set `weather_calendar` alongside `weather_only` so the
  with/without comparison is a config swap. The assembler grows a calendar
  join step; the join itself will live beside `build()`.

## Running standalone

    python -m bristol_ml.features.weather [--head N]
    python -m bristol_ml.features.assembler [--cache {auto,refresh,offline}]

Both CLIs honour Hydra overrides in the trailing positional slot.

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
