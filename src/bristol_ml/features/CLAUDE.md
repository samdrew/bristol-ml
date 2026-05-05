# `bristol_ml.features` — module guide

This module is the **features layer**: functions that compose cleaned
per-source data (ingestion-layer output) into model-ready inputs. Stage 2
introduces the layer one stage earlier than DESIGN §9 implies, because
Stage 2 needs a weighted-mean function and the notebook cannot reimplement
it (§2.1.8 — notebooks are thin).

## Current surface (Stages 5–7, 16)

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
  `assemble()` / `assemble_calendar()` / `assemble_with_remit()` orchestrators.
  On clock-change days the output is 23 rows (spring) or 25 rows (autumn) —
  the UTC timeline is regular; the NESO ingester has already unwound DST algebra.

**Stage 16 additions (also in `assembler.__all__`):**

- `WITH_REMIT_OUTPUT_SCHEMA: pa.Schema` — the declared parquet schema for
  the `with_remit` feature set (59 columns). Composition:
  - positions 0..54 → `CALENDAR_OUTPUT_SCHEMA.names` (the 55-column Stage 5
    weather+calendar schema, unchanged — a strict prefix so downstream code
    reading only calendar columns continues to work by column-name selection).
  - positions 55..57 → `REMIT_VARIABLE_COLUMNS` (3 columns:
    `remit_unavail_mw_total` float32, `remit_active_unplanned_count` int32,
    `remit_unavail_mw_next_24h` float32).
  - position 58 → `remit_retrieved_at_utc` (provenance scalar,
    `timestamp[us, tz=UTC]`), the fourth provenance scalar following the
    Stage 3/5 one-per-ingester convention.
  Column order is contractual; `REMIT_VARIABLE_COLUMNS` is imported from
  `features.remit` and re-exported here as the single source of truth for
  column names and dtypes (plan D2 / NFR-4).

- `assemble_with_remit(cfg: AppConfig, *, cache="offline") -> Path` — Stage 16
  orchestrator. Composes the Stage 5 weather+calendar pipeline (NESO +
  weather + holidays + `derive_calendar`) with the Stage 16 REMIT derivation
  (`derive_remit_features`), appends `remit_retrieved_at_utc`, casts to
  `WITH_REMIT_OUTPUT_SCHEMA`, and persists via `_atomic_write`. Requires
  `cfg.features.with_remit`, `cfg.ingestion.neso`, `cfg.ingestion.weather`,
  `cfg.ingestion.holidays`, and `cfg.ingestion.remit` all to be populated.

  **Auto-run extractor on missing parquet.** If the extracted-features parquet
  at `data/processed/remit_extracted.parquet` (or the override path in
  `WithRemitFeatureConfig.extracted_parquet_filename`) is absent, the function
  logs a `WARNING` naming the missing path and the explicit CLI to use, then
  runs `extract_and_persist(build_extractor(cfg.llm), remit_df, ...)` inline.
  Under `BRISTOL_ML_LLM_STUB=1` (the CI default) this produces a stub-quality
  parquet at zero cost and keeps CI green. For the real-extractor path (plan A3),
  run `python -m bristol_ml.llm.persistence` once beforehand to populate the
  parquet, then rerun the assembler against the warm cache. The WARNING log line
  reads:
  ```
  with_remit assembler: extracted-features parquet missing at <path>;
  running extractor inline (stub-mode default).  Run
  `python -m bristol_ml.llm.persistence` first to use a different extractor.
  ```

  **CLI note.** The `python -m bristol_ml.features.assembler` CLI dispatches
  only on `assemble()` (weather-only path). To run the `with_remit` orchestrator
  from Python code, call `assemble_with_remit(cfg, cache="auto")` directly
  after resolving the config with `load_config(overrides=["features=with_remit"])`.

- `load_with_remit(path: Path) -> pd.DataFrame` — schema-validated read for
  `WITH_REMIT_OUTPUT_SCHEMA`. Rejects both missing and extra columns (exact
  schema contract, same discipline as `load` and `load_calendar`). A
  weather-only or weather+calendar parquet passed here will be rejected because
  its REMIT columns are absent.

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

### `remit` (Stage 16)

New module `bristol_ml.features.remit`. Pure derivation; no I/O. Conforms to
the `calendar.py` shape: one derivation function + one typed column constant +
module docstring naming the bi-temporal contract.

**Public surface** (matches `remit.__all__`):

- `REMIT_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` — the three
  Stage 16 REMIT-derived columns and their pyarrow types:
  - `("remit_unavail_mw_total", pa.float32())` — sum of `affected_mw` for
    events that are both *active* (`effective_from <= t < effective_to`) and
    *known* (the latest revision of its `mrid` published at or before `t`,
    `message_status != "Withdrawn"`).
  - `("remit_active_unplanned_count", pa.int32())` — count of revisions meeting
    the same active-and-known condition whose `cause` matches `"Unplanned"`
    (case-insensitive; "Forced" is a distinct category and is not counted).
  - `("remit_unavail_mw_next_24h", pa.float32())` — sum of `affected_mw` for
    revisions known at `t` whose `effective_from` lies in `[t, t + 24h)` (the
    forward-looking "known future input" signal; plan A2). Column name is fixed
    as `_next_24h` regardless of the configured `forward_lookahead_hours` so
    schema-driven downstream code is stable if the horizon is tuned.
  Column order is contractual; `WITH_REMIT_OUTPUT_SCHEMA` reads this constant
  verbatim.

- `derive_remit_features(remit_df, hourly_index, *, forward_lookahead_hours=24)
  -> pd.DataFrame` — computes the three REMIT columns for `hourly_index`.
  Returns a frame with `timestamp_utc` + the three columns in `REMIT_VARIABLE_COLUMNS`
  order. Zero-event hours produce zero values (no NaN — NFR: AC-7). Raises
  `ValueError` on malformed inputs (tz-naive index, missing columns, etc.).

**Bi-temporal correctness contract (NFR-1).** `derive_remit_features` is
*structurally* correct: the algorithm restricts each revision's contribution to
the open half-interval during which it is both the latest visible revision of
its `mrid` AND active per its event window. No row of the result reflects
information published after that row's `timestamp_utc`. **Bypassing this
function — e.g. by joining the raw REMIT log directly onto the hourly grid —
exposes the caller to leakage.** The module docstring states this explicitly;
callers must not circumvent the function.

**Algorithm overview** (vectorised; O((n_revisions + n_hours) log)):

1. Per-`mrid` validity: sort by `(published_at, revision_number)` ascending;
   tag each revision with its transaction-time validity interval
   `[published_at(r), published_at(r+1))`.
2. Per-revision contribution windows (active, forward, unplanned-count).
3. Delta-event aggregation: emit `(window_from, +d)` and `(window_to, -d)` per
   window; cumsum; look up each hourly `t` via `merge_asof(direction="backward")`.

**Withdrawn-truncates-prior subtlety (codebase hazard H2; `_per_mrid_validity`
implementation).** A `Withdrawn` revision does not contribute to any signal, but
it *does* truncate the prior revision's transaction-time validity interval. The
`tx_valid_to` shift is therefore applied across the **full** sorted log including
`Withdrawn` rows; `Withdrawn` rows are dropped only at the end of
`_per_mrid_validity`. Without this, a sequence `rev0:Active, rev1:Withdrawn`
would leave rev0 appearing valid indefinitely — the inverse of the `as_of` rule.
Any future modification to `_per_mrid_validity` must preserve this ordering.

**Forward-looking column behaviour and the `include_forward_lookahead` config
knob (plan A2 / A4).** The `remit_unavail_mw_next_24h` column is **always
present in the schema** for contract stability — `WITH_REMIT_OUTPUT_SCHEMA`
always has 59 columns. The `WithRemitFeatureConfig.include_forward_lookahead`
flag lives in `conf/features/with_remit.yaml` and governs whether the model's
`feature_columns` field *reads* the column. This is how the two registered Stage
16 runs are differentiated: both use the same parquet on disk; the run with
`include_forward_lookahead=false` omits `remit_unavail_mw_next_24h` from its
`feature_columns` list.

**Dtype-precision note.** The REMIT parquet round-trips at microsecond UTC
precision; the assembler's hourly grid is nanosecond UTC. `_running_total` and
`_lookup_at_grid` both call `.as_unit("ns")` on the timestamp series before the
`merge_asof` so the dtype equality check does not fail silently.

**Running standalone:**

    python -m bristol_ml.features.remit [--rows N] [hydra overrides...]

Loads the cached REMIT parquet and the weather+calendar feature table when warm;
synthesises a 48-hour grid when they are absent. Always prints `REMIT_VARIABLE_COLUMNS`.

---

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

## Stage 16 notes

**Mutual-exclusivity extended to `with_remit`.** The Hydra group-swap pattern
from Stage 5 extends to the third feature set: exactly one of
`cfg.features.weather_only` / `cfg.features.weather_calendar` /
`cfg.features.with_remit` is populated per run; `train._resolve_feature_set`
raises with a message naming `features=with_remit` on any other case. The same
mutual-exclusivity invariant is enforced in `assemble_with_remit`.

**Extracted features parquet flows from `llm/persistence` into the assembler.**
This is the project's first cross-layer data dependency at training time: the
`llm` layer (Stage 14) produces `ExtractionResult` objects in memory; Stage 16's
`llm.persistence` module adds the on-disk persistence step, writing
`data/processed/remit_extracted.parquet` against `EXTRACTED_OUTPUT_SCHEMA` (11
columns, keyed on `(mrid, revision_number)`). The features layer then reads this
parquet in `assemble_with_remit` and joins it onto the REMIT log to override
`affected_mw` with the LLM-extracted `affected_capacity_mw` where available.
The features layer imports `llm.persistence` for `load_extracted` and
`extract_and_persist` only; it does not import `Extractor` or `LlmExtractor`
directly, keeping the features layer free of the OpenAI SDK's import graph
(plan OQ-5).

**`assemble_with_remit` does not delegate to `assemble_calendar`.** The same
mutual-exclusivity constraint that prevents `assemble_calendar` from delegating
to `assemble()` applies here: when `features=with_remit` is active,
`cfg.features.weather_calendar` is `None`, so calling `assemble_calendar`
would raise before the REMIT layer could compose. `assemble_with_remit`
duplicates the Stage 5 calendar composition inline and extends it with the
REMIT step. A future refactor could extract a shared
`_compose_calendar_frame(cfg, fset, *, cache)` helper.

## Running standalone

    python -m bristol_ml.features.weather   [--head N]
    python -m bristol_ml.features.calendar  [--rows N]
    python -m bristol_ml.features.assembler [--cache {auto,refresh,offline}] [overrides...]
    python -m bristol_ml.features.fourier   [--help]
    python -m bristol_ml.features.remit     [--rows N] [hydra overrides...]

All five CLIs honour Hydra overrides in the trailing positional slot.

**Note on `assembler` CLI.** The `python -m bristol_ml.features.assembler` CLI
dispatches only on `assemble()` (the weather-only orchestrator). The
`assemble_calendar` and `assemble_with_remit` orchestrators are invoked
in-process, not via this CLI. Pass `features=with_remit` as a Hydra override
when calling `load_config` before invoking `assemble_with_remit`.

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
