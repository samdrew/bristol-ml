# `bristol_ml.features` — module guide

This module is the **features layer**: functions that compose cleaned
per-source data (ingestion-layer output) into model-ready inputs. Stage 2
introduces the layer one stage earlier than DESIGN §9 implies, because
Stage 2 needs a weighted-mean function and the notebook cannot reimplement
it (§2.1.8 — notebooks are thin).

## Current surface (Stages 5–7)

### `weather.national_aggregate(df, weights)` (Stage 2)

Collapse long-form per-station hourly weather into a wide-form national
signal using caller-supplied weights. Honours acceptance criterion 3
(subset of stations) via the Mapping argument. Honours acceptance
criterion 6 (equal weights on identical inputs yield the identity) —
the renormalised weighted mean of a constant is the constant.

### `assembler` (Stage 3 + Stage 5)

Public surface (matches `assembler.__all__`):

- `DEMAND_COLUMNS: tuple[str, ...]` — `("nd_mw", "tsd_mw")`. The demand
  columns carried through the hourly resample; both are `int32` MW.
- `WEATHER_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` — the five
  weather aggregate columns and their arrow types. `cloud_cover` is widened
  from `int8` (long-form weather schema) to `float32` here.
- `CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` — the 44
  calendar columns (23 hour-of-day + 6 day-of-week + 11 month + 4 holiday
  flags; all `int8`). Owned by `features.calendar`; re-exported here so the
  assembler's public surface covers both schemas. See §"Stage 5 notes".
- `OUTPUT_SCHEMA: pa.Schema` — the declared parquet schema for the
  `weather_only` feature set (10 columns). Column order, arrow dtypes and
  timezone metadata are contractual; downstream models may rely on all
  three.
- `CALENDAR_OUTPUT_SCHEMA: pa.Schema` — the declared parquet schema for
  the `weather_calendar` feature set (55 columns). Structured as
  `OUTPUT_SCHEMA.names` (10) + `CALENDAR_VARIABLE_COLUMNS` (44) +
  `holidays_retrieved_at_utc` (1). `OUTPUT_SCHEMA.names` is an exact
  prefix of `CALENDAR_OUTPUT_SCHEMA.names[:10]` — downstream code that
  reads only the weather columns continues to work on the calendar frame
  by column-name selection.
- `build(demand_hourly, weather_national, config, *, neso_retrieved_at_utc=None,
  weather_retrieved_at_utc=None) -> pd.DataFrame` — inner-join demand with
  national weather, forward-fill weather up to
  `config.forward_fill_hours`, drop remaining NaN rows, project to
  `OUTPUT_SCHEMA` column order. Emits a single structured INFO log line
  per call (D5) with counts of rows dropped/filled at each step. Feature-
  set-agnostic: reads only `config.forward_fill_hours` and `config.name`,
  so both `weather_only` and `weather_calendar` configs compose.
- `load(path: Path) -> pd.DataFrame` — schema-validated read for
  `OUTPUT_SCHEMA`; refuses missing or extra columns. Mirrors `neso.load` /
  `weather.load`.
- `load_calendar(path: Path) -> pd.DataFrame` — schema-validated read for
  `CALENDAR_OUTPUT_SCHEMA`. Accepts 55 columns exactly; rejects both
  missing and extra columns. A `weather_only` parquet is rejected here
  because its calendar columns are absent, and vice versa.
- `assemble(cfg: AppConfig, cache="offline") -> Path` — one-shot
  orchestrator that ties `neso.fetch/load → _resample_demand_hourly →
  weather.fetch/load → national_aggregate → build → _atomic_write`. Used
  by the CLI. Requires `cfg.features.weather_only` to be populated.
- `assemble_calendar(cfg: AppConfig, *, cache="offline") -> Path` — Stage 5
  orchestrator. Composes the weather-only join (duplicating `assemble()`'s
  NESO/weather/resample/`build` sequence inline — see §"Stage 5 notes"),
  then calls `holidays.fetch/load → derive_calendar`, appends the
  `holidays_retrieved_at_utc` scalar, casts to `CALENDAR_OUTPUT_SCHEMA`,
  and persists via `_atomic_write`. Requires `cfg.features.weather_calendar`,
  `cfg.ingestion.neso`, `cfg.ingestion.weather`, and `cfg.ingestion.holidays`
  all to be populated. Accepts either a `CachePolicy` value or one of the
  three policy strings.
- `_resample_demand_hourly(df, agg: Literal["mean", "max"] = "mean") ->
  pd.DataFrame` — floor `timestamp_utc` to the hour and aggregate
  `nd_mw` / `tsd_mw`. Module-private; exposed for testing and for the
  `assemble()` / `assemble_calendar()` orchestrators. On clock-change days
  the output is 23 rows (spring) or 25 rows (autumn) — the UTC timeline
  is regular; the NESO ingester has already unwound DST algebra.

### `fourier` (Stage 7)

Pure weekly Fourier-harmonic feature helper, added by Stage 7 to supply the weekly exogenous regressors for `SarimaxModel`. Public surface:

- `append_weekly_fourier(df, *, period_hours=168, harmonics=3, column_prefix="week") -> pd.DataFrame` — appends `2 * harmonics` columns (`week_sin_k1..kN`, `week_cos_k1..kN`) to `df` and returns a new frame (no input mutation). Requires a tz-aware `DatetimeIndex`; converts to floating-point hours since the UTC epoch via `(idx - 1970-01-01 UTC) / 1h` so DST transitions do not introduce phase drift **and** the conversion is precision-independent (works on `ns` / `us` / `ms` / `s` indices). The pre-2026-05-04 implementation used `idx.view("int64") // _NANOSECONDS_PER_HOUR` which silently produced collapsed sin/cos columns when given the microsecond-precision indices the assembler emits — see the function's `Notes` section for the failure mode. `harmonics=0` is a no-op fast path. Tz-naive input raises `ValueError`. Module CLI `python -m bristol_ml.features.fourier --help`.

### `calendar` (Stage 5)

Pure derivation of the 44 calendar columns for the `weather_calendar`
feature set. Public surface:

- `derive_calendar(df, holidays_df) -> pd.DataFrame` — appends the 44
  `CALENDAR_VARIABLE_COLUMNS` (`int8`) to an hourly UTC frame. Reads
  `Europe/London` local components for day-of-week / month / holiday
  lookup (plan D-4 / D-7) and the **UTC hour** for hour-of-day dummies
  (human mandate 2026-04-20 — every calendar day has exactly 24 UTC rows,
  including DST-change Sundays). Emits a single structured INFO log line
  per call, plus a single WARNING when pre-window rows are zero-filled
  (plan D-6). No I/O, no global state.
- `CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` —
  ordered constant naming all 44 calendar columns and their arrow types.
  Assembler's `CALENDAR_OUTPUT_SCHEMA` and downstream
  `LinearConfig.feature_columns` read from this single source of truth.
- `is_weekend` is **deliberately not emitted** (external research §R5 —
  perfect collinearity with the day-of-week one-hot). A module-level
  assertion plus a runtime guard in `derive_calendar` pin the invariant.

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

## Stage 5 notes

**Two schemas, one module.** The assembler exposes both `OUTPUT_SCHEMA`
(weather-only, 10 cols) and `CALENDAR_OUTPUT_SCHEMA` (weather + calendar,
55 cols). The weather-only schema is preserved as an exact prefix of the
calendar schema (positions 0..9), so downstream code that reads only the
weather columns composes with either frame by column-name selection.

**Hydra group-swap mutual exclusivity.** The Stage 5 T1 refactor flipped
`- features/weather_only@features.weather_only` (package-override form) to
`- features: weather_only` (group-swap form) in `conf/config.yaml` defaults,
so `features=weather_calendar` at the CLI swaps which file loads. At
runtime exactly one of `cfg.features.weather_only` /
`cfg.features.weather_calendar` is populated — the other is `None`. This
drives two behavioural contracts the lead/agents should keep in mind:

- `assemble()` requires `cfg.features.weather_only`; `assemble_calendar()`
  requires `cfg.features.weather_calendar`. Calling either on the other's
  config raises with a message naming the missing field and the Hydra
  override to use.
- `assemble_calendar()` does **not** delegate to `assemble()` — it
  duplicates the NESO/weather/resample/`build` composition inline. The
  plan (§6 Task T4) originally described delegation, but mutual
  exclusivity makes that impossible. The duplication is acknowledged;
  factoring a shared `_compose_weather_only_frame(cfg, fset, *, cache)`
  helper is a candidate future refactor and is noted in the Stage 5
  retrospective.

**Calendar-column ownership.** `CALENDAR_VARIABLE_COLUMNS` is owned by
`features.calendar` (the derivation module); `assembler.py` imports and
re-exports it so the 55-column `CALENDAR_OUTPUT_SCHEMA` has a single
source of truth for the 44 calendar column names / dtypes.

## Running standalone

    python -m bristol_ml.features.weather   [--head N]
    python -m bristol_ml.features.calendar  [--rows N]
    python -m bristol_ml.features.assembler [--cache {auto,refresh,offline}] [overrides...]
    python -m bristol_ml.features.fourier   [--help]

All four CLIs honour Hydra overrides in the trailing positional slot.

## Regenerating after a feature change

When you add or remove a column in `derive_calendar` (or
`weather.national_aggregate`, or any pure derivation), the relevant
`*_OUTPUT_SCHEMA` constant changes and any existing cached parquet
under `data/features/` is now schema-stale. `load` / `load_calendar`
will refuse it with a `ValueError` naming both the missing column and
the regeneration command — copy-paste from the error itself, or use
the recipes below.

The `assembler` CLI dispatches on the active `features=` group:

    # Stage 3 weather-only frame
    uv run python -m bristol_ml.features.assembler features=weather_only --cache offline

    # Stage 5 weather + calendar frame
    uv run python -m bristol_ml.features.assembler features=weather_calendar --cache offline

Use `--cache offline` (the default) when only the derivation has
changed; the upstream ingester caches stay warm and the assembler
only re-runs `build` / `derive_calendar` against them. `--cache auto`
populates missing ingester caches from the network on first run.
`--cache refresh` re-fetches them. The assembler's own output parquet
is always written via atomic replace, so a re-run never corrupts a
partially-written file (NFR-3).

The dispatch helper (`assembler._resolve_orchestrator`) mirrors
`bristol_ml.train._resolve_feature_set` so the two CLIs agree on
which feature-set group is active. Adding a third sibling (e.g.
Stage 16's `with_remit`) is a one-line extension to both helpers.

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
