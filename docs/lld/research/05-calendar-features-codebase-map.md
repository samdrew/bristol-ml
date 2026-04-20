# Stage 5 — Calendar features codebase map

**Purpose:** Name every surface the Stage 5 implementer will touch or depend on.
**Date:** 2026-04-19
**Baseline SHA:** see `git log --oneline -1`

---

## §A — Consumable Stage 3/4 surfaces (reused verbatim)

### `bristol_ml.features.assembler.OUTPUT_SCHEMA`
- Import: `from bristol_ml.features.assembler import OUTPUT_SCHEMA`
- Source: `/workspace/src/bristol_ml/features/assembler.py` lines 87–101
- Schema (10 columns, order is contractual):
  `timestamp_utc`, `nd_mw`, `tsd_mw`, `temperature_2m`, `dew_point_2m`,
  `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`,
  `neso_retrieved_at_utc`, `weather_retrieved_at_utc`
- Invariants downstream callers rely on: column order, exact names,
  `pa.timestamp('us', tz='UTC')` for timestamps, `int32` for MW,
  `float32` for weather, no NaN, strictly monotonic ascending unique
  `timestamp_utc`. Adding columns is additive; reordering is breaking.
- Stage 5 must define a parallel `CALENDAR_OUTPUT_SCHEMA` (or extend the
  `weather_only` schema) for `weather_calendar` — see §B.

### `bristol_ml.features.assembler.WEATHER_VARIABLE_COLUMNS`
- Source: assembler.py lines 70–84
- Type: `tuple[tuple[str, pa.DataType], ...]`; five entries
- **Load-bearing coupling:** `harness.evaluate` (line 58 of harness.py)
  and `LinearModel._resolve_feature_columns` (linear.py line 232) both
  import this constant and use it as the default feature set when
  `feature_columns=None`. If Stage 5 adds calendar columns, they are
  *not* in `WEATHER_VARIABLE_COLUMNS` — callers must pass an explicit
  `feature_columns` list for the enriched set, or Stage 5 must introduce
  a parallel constant (e.g. `CALENDAR_VARIABLE_COLUMNS`).

### `bristol_ml.features.assembler.build()`
- Signature (assembler.py lines 188–195):
  `build(demand_hourly, weather_national, config: FeatureSetConfig, *,
  neso_retrieved_at_utc=None, weather_retrieved_at_utc=None) -> pd.DataFrame`
- Invariants: inner-joins on tz-aware UTC `timestamp_utc`; forward-fills
  weather; drops NaN rows; asserts `OUTPUT_SCHEMA`; emits one structured
  loguru INFO line. Pure — no I/O.
- Stage 5 implication: if Stage 5 extends `build()`, its extra calendar
  columns must be appended after the existing columns (additive) and the
  schema assert at line 508 must be updated. See §B for options.

### `bristol_ml.features.assembler.load(path)`
- Source: assembler.py lines 410–440
- Validates every field in `OUTPUT_SCHEMA`; raises on missing or extra
  columns. Stage 5 must either (a) produce a separate loader for the
  calendar feature set, (b) make `load()` accept a schema argument, or
  (c) keep the two feature sets behind separate module-level `load`
  functions. Extra columns cause a hard `ValueError` — load is exact,
  not permissive.

### `bristol_ml.features.assembler.assemble()`
- Source: assembler.py lines 448–516
- Reads `cfg.features.weather_only` at line 464; the check at line 464
  is hard-coded to `weather_only`. Stage 5's assembler for
  `weather_calendar` must use `cfg.features.weather_calendar` — either a
  second `assemble()` or a dispatcher.

### Cache path idiom and `_atomic_write`
- `_cache_path(config: CachePathConfig) -> Path` from
  `bristol_ml.ingestion._common` (lines 118–122): resolves
  `config.cache_dir / config.cache_filename` and `mkdir`s the parent.
- `_atomic_write(table, path)` (lines 201–210): writes via tmp +
  `os.replace`. Both functions accept any object satisfying
  `CachePathConfig` (duck-typed `Protocol`), so a new `FeatureSetConfig`
  with `cache_dir` and `cache_filename` works without changes.
- Path: `${BRISTOL_ML_CACHE_DIR:-data/features}/weather_calendar.parquet`
  by convention (features.md line 86).

---

### `bristol_ml.evaluation.harness.evaluate`
- Source: `/workspace/src/bristol_ml/evaluation/harness.py` lines 72–201
- Signature: `evaluate(model, df, splitter_cfg, metrics, *, target_column="nd_mw", feature_columns=None) -> pd.DataFrame`
- **`feature_columns=None` default:** resolves to
  `[name for name, _ in WEATHER_VARIABLE_COLUMNS]` (harness.py lines
  254–255). For the `weather_calendar` run the caller must pass an
  explicit `feature_columns` list that includes the calendar columns, or
  the harness will silently train on weather features only.
- **H-1 UTC guard** (harness.py lines 225–233): tz-aware non-UTC index
  raises `ValueError` before any fold. Calendar columns added as `int8`
  integers carry no timezone; the guard is on `df.index`, not on column
  dtypes — no regression risk there.
- Stage 5 constraint: the `train.py` without/with demo must pass an
  explicit `feature_columns` list (or rely on `LinearConfig.feature_columns`
  being populated) when running the calendar-enriched set.

### `bristol_ml.evaluation.metrics.METRIC_REGISTRY`
- Source: `/workspace/src/bristol_ml/evaluation/metrics.py`
- `dict[str, MetricFn]`; keys `"mae"`, `"mape"`, `"rmse"`, `"wape"`.
- Stage 5 reuses unchanged. The Stage 5 notebook calls
  `compare_on_holdout` with the same registry entries for both runs
  (weather-only and weather-calendar).

### `bristol_ml.evaluation.benchmarks.compare_on_holdout`
- Source: `/workspace/src/bristol_ml/evaluation/benchmarks.py`
- Signature (from evaluation layer doc): `compare_on_holdout(models, df, neso_forecast, splitter_cfg, metrics, *, aggregation="mean", target_column="nd_mw", feature_columns=None) -> pd.DataFrame`
- Returns DataFrame indexed `[*sorted(models), "neso"]`.
- The Stage 5 notebook's without/with demo calls this **twice** (or once
  with both models in the `models` mapping): once with the
  `weather_only` df and once with the `weather_calendar` df. Each call
  is independent — `compare_on_holdout` is stateless. The `feature_columns`
  kwarg must be passed explicitly for the calendar run.

### `bristol_ml.models.linear.LinearModel` — `feature_columns` resolution
- Source: linear.py lines 221–232
- `_resolve_feature_columns(features)`: if `config.feature_columns is not None`,
  return `tuple(config.feature_columns)`; else return
  `tuple(name for name, _ in WEATHER_VARIABLE_COLUMNS)`.
- Stage 5 can drive `LinearModel` purely via `LinearConfig.feature_columns`
  (an explicit tuple in YAML or a CLI override) — no code path change
  needed. The model will use whatever columns the caller supplies at
  `fit()` time. The `metadata.name` field is hardcoded to
  `"linear-ols-weather-only"` (linear.py line 195); Stage 5 should
  override this for the calendar run, which requires passing an explicit
  name or constructing a second `LinearConfig` — this is the only
  material issue.

### `bristol_ml.train._cli_main` — feature-set selection
- Source: `/workspace/src/bristol_ml/train.py` lines 106–132
- Currently hardcoded to `cfg.features.weather_only` at line 106.
  The variable `fset` is used to derive `feature_cache` (line 123) and
  to pass `cache_filename` to `assembler.load` (line 132).
- Stage 5 must either:
  (a) Extend `train.py` to read from a feature-set discriminator
      (e.g. `cfg.features.active_set` or a new Hydra override
      `features=weather_calendar`), or
  (b) Keep `train.py` fixed and let the notebook drive the two runs
      directly.
- The without/with demo intent (AC-1: "switching is a configuration
  change, not a code change") suggests option (a) — a `features=`
  group override that selects which feature-set config to load.

### `bristol_ml.ingestion._common` protocol-typed helpers
- Source: `/workspace/src/bristol_ml/ingestion/_common.py`
- `CachePolicy` (line 60–72): `"auto" | "refresh" | "offline"`.
- `CacheMissingError` (line 74): raised on `OFFLINE` with absent cache.
- `_retrying_get(client, url, params, config)` (lines 157–193): accepts
  any `RetryConfig` (duck-typed: `max_attempts`, `backoff_*`,
  `request_timeout_seconds`).
- `_respect_rate_limit(last, gap)` (lines 130–145): sleep helper.
- `_atomic_write(table, path)` (lines 201–210): tmp + `os.replace`.
- `_cache_path(config)` (lines 118–122): mkdir + resolve.
- `RetryConfig`, `RateLimitConfig`, `CachePathConfig` are structural
  `Protocol` types (lines 83–111): any Pydantic model with the right
  attribute names satisfies them without a shared base.
- The `holidays.py` ingester will import these six helpers verbatim,
  matching the `neso_forecast.py` copy-and-adapt pattern.

---

## §B — Assembler extension points (where Stage 5 edits)

The central Stage 5 structural question is: does `build()` generalise,
does Stage 5 introduce a second builder, or is there a composable
derivation function? Four options:

### Option 1 — Second builder + discriminated-union `FeaturesConfig`
- Add `build_calendar(demand_hourly, weather_national, holidays_df, config) -> pd.DataFrame`
  alongside `build()` in `assembler.py` (or a new `features/calendar.py`).
- Define a new `CALENDAR_OUTPUT_SCHEMA` constant extending
  `OUTPUT_SCHEMA.names` with calendar columns appended after
  `shortwave_radiation` and before `neso_retrieved_at_utc`.
- Extend `FeaturesGroup` in `conf/_schemas.py` with
  `weather_calendar: FeatureSetConfig | None = None` (already
  anticipated in the `FeatureSetConfig` docstring, _schemas.py line 188).
- Add `conf/features/weather_calendar.yaml` with
  `# @package features.weather_calendar`.
- This option mirrors the Stage 4 `NaiveConfig | LinearConfig`
  discriminated union (ModelConfig) but applied at the feature-set level.
  Functions touched: `assembler.py` lines 87–101 (add schema constant),
  lines 448–516 (`assemble()` needs to dispatch by config field), and
  `conf/_schemas.py` line 211 (add `weather_calendar` field).

### Option 2 — Parameterised single `build()` with a feature-set discriminator
- Add a `feature_set: Literal["weather_only", "weather_calendar"] = "weather_only"`
  parameter to `build()`.
- Internally, after the weather join, conditionally call
  `derive_calendar(joined, holidays_df)` and append calendar columns.
- Single `OUTPUT_SCHEMA` becomes conditional — either the 10-column or
  the N-column version depending on the argument. The schema-assert at
  line 508 must be parameterised accordingly.
- Functions touched: `build()` signature (line 188), the schema-assert
  loop at line 508, `assemble()` at line 448.
- Risk: a single `build()` with branching logic is harder to test in
  isolation than two pure functions.

### Option 3 — Composable `derive_calendar(df) -> pd.DataFrame` (recommended for pedagogy)
- `features/calendar.py` exposes a pure function:
  `derive_calendar(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame`
  that takes any hourly frame with a UTC `timestamp_utc` column and
  returns the same frame with calendar columns appended.
- `assembler.py` adds an `assemble_calendar()` orchestrator that calls
  `assemble()` (weather-only) then joins the calendar derivation.
- `CALENDAR_OUTPUT_SCHEMA` extends `OUTPUT_SCHEMA`.
- A separate `load_calendar(path)` validates the extended schema.
- This is the cleanest seam: `derive_calendar` is a pure function
  (satisfies intent AC-2 — "pure function: same inputs, same outputs"),
  testable without any ingestion fixtures, and the assembler remains
  purely compositional. The `features.md` layer doc (lines 55–78)
  explicitly lists "Calendar feature derivation" as a pure derivation
  shape alongside `national_aggregate`.
- Functions touched: new `features/calendar.py`, new
  `CALENDAR_OUTPUT_SCHEMA` in assembler.py, new `load_calendar()`, new
  `assemble_calendar()` orchestrator, `conf/_schemas.py` `FeaturesGroup`
  gains `weather_calendar` field.

### Option 4 — `weather_only` YAML + `weather_calendar` YAML, same builder, post-join hook
- Keep `build()` unchanged; add a `calendar_join_fn: Callable | None = None`
  kwarg that, when provided, is called on the joined frame before schema
  assertion.
- The YAML selects which callable to inject.
- Complicates the Hydra/Pydantic boundary (callables in configs are
  anti-pattern; DESIGN §7.1 "Configuration lives outside code").
  Not recommended.

**`OUTPUT_SCHEMA` tuple positions affected by any option:**
Current columns 0–9 (0-indexed): `timestamp_utc, nd_mw, tsd_mw,
temperature_2m, dew_point_2m, wind_speed_10m, cloud_cover,
shortwave_radiation, neso_retrieved_at_utc, weather_retrieved_at_utc`.
Calendar columns must be inserted before provenance columns (8, 9) to
keep the schema readable; or appended after position 9 if provenance
must remain last. The features layer doc (line 89) specifies `int8` for
`hour_of_day` / `day_of_week`; `month` could be `int8`; boolean
indicators (`is_weekend`, `is_bank_holiday`) as `bool` or `int8`;
proximity-to-holiday as `int8` (days). Float sin/cos encodings would be
`float32` matching the weather columns.

---

## §C — Patterns to follow

### `# @package` header on Hydra YAML group files
- Every `conf/features/*.yaml`, `conf/model/*.yaml`, `conf/evaluation/*.yaml`
  opens with `# @package <group>.<name>`.
- Example: `conf/features/weather_only.yaml` line 1:
  `# @package features.weather_only`
- Stage 5 must add `conf/features/weather_calendar.yaml` with
  `# @package features.weather_calendar` and
  `conf/ingestion/holidays.yaml` with `# @package ingestion.holidays`.

### `ConfigDict(extra="forbid", frozen=True)` on every Pydantic model
- Source: `conf/_schemas.py` lines 19, 27, 38, 65, 83, 121, etc. — every
  class in the file carries this.
- `HolidaysIngestionConfig` and any new schema for Stage 5 must carry the
  same decorator.

### Lazy re-export idiom in layer `__init__.py`
- Source: `src/bristol_ml/evaluation/__init__.py` lines 26–83
- Pattern: `if TYPE_CHECKING:` imports for type stubs; `__getattr__`
  dispatches by name set to the submodule. `__all__` lists public names.
- Stage 5's `features/__init__.py` does not currently use this pattern
  (see features/CLAUDE.md "Accessed by submodule import"); Stage 5 may
  add `calendar` to the submodule list without changing the re-export idiom.

### `_atomic_write` + `os.replace` pattern
- Source: `_common.py` lines 201–210
- Tmp file `path.with_suffix(path.suffix + ".tmp")`, `pq.write_table`,
  then `os.replace(tmp, path)`.
- Reused verbatim by `assembler.py` line 510 and by `models/io.py`.
  Stage 5 must not invent a parallel write helper.

### VCR cassette recording pattern
- Existing cassettes:
  - `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml`
  - `tests/fixtures/weather/cassettes/weather_2023_01.yaml`
  - `tests/fixtures/neso_forecast/cassettes/neso_forecast_refresh.yaml`
- Recorder scripts:
  - `scripts/record_neso_cassette.py`
  - `scripts/record_weather_cassette.py`
  (no recorder script for `neso_forecast` — it used `--record-mode=once`)
- Stage 5 must add `tests/fixtures/holidays/cassettes/` and optionally
  `scripts/record_holidays_cassette.py`. The cassette records one real
  GET of `https://www.gov.uk/bank-holidays.json`.

### Loguru INFO-line-per-call style
- `features/assembler.py` lines 313–323: one `logger.info(...)` call
  in `build()` with structured named fields in the message string.
- `evaluation/harness.py` lines 160–166: one `logger.info(...)` per fold;
  lines 194–199: one summary `logger.info(...)` on completion.
- Stage 5 `calendar.derive_calendar()` should emit one structured INFO
  line naming the count of bank-holiday dates resolved and the count of
  calendar rows written.

### Module CLI shape
- Pattern: `_cli_main(argv: Iterable[str] | None = None) -> int`
  accepting an explicit argv for in-process testing.
- Sources: `harness.py` line 293, `train.py` line 80,
  `assembler.py` line 548.
- Exit codes: `0` success, `2` missing config/cache, `3` unknown variant.
- Stage 5's `ingestion/holidays.py` and `features/calendar.py` must each
  expose a `_cli_main(argv=None) -> int` in the same shape, and
  `if __name__ == "__main__": raise SystemExit(_cli_main())`.

---

## §D — Stage 4 Phase 3 review deferrals: carry-overs for Stage 5?

From `docs/lld/stages/04-linear-baseline.md` "Deferred" and "Known drift":

| Deferred item | Stage 5 relevant? | Verdict |
|---|---|---|
| Persisting per-fold predictions (`predictions_path` kwarg on `evaluate`) | No | Stage 6 (enhanced diagnostics). |
| `skops.io` adoption for secure artefacts | No | Stage 9 (registry). |
| `ingestion._neso_dst` extraction (DST algebra duplicated in `neso.py` and `neso_forecast.py`) | **Borderline.** Stage 5 adds `holidays.py` which is date-level only — no settlement-period → UTC algebra needed. No third duplication; no trigger. | No — defer to a future refactor. |
| `NesoBenchmarkConfig.holdout_start/_end` consumers | No | Stage 6. |
| `DESIGN.md §6` layout tree accumulation (deny-tier for the lead) | **Yes, in principle.** Stages 1–5 additions should be batched into §6. This is deny-tier for the lead and must be a human-approved main-session edit. Flag explicitly. | Human edit required; Stage 5 PR should prompt the human to update §6. |
| DESIGN §8 "Open-Meteo UKV 2km" inaccuracy | No | Unchanged; Stage 2 surfaced it. |
| `docs/architecture/ROADMAP.md` "Features" open questions answered by Stage 3 | **Stage 5 is the named trigger** in the Stage 3 retro: "Deferred to a ROADMAP pass at Stage 5/6". Stage 5 should update ROADMAP.md to close the "feature-table schema contract" and "population-weighting home" questions. | Close at Stage 5 (minor — warn-tier for the lead). |

**Carry-overs for Stage 5:**
1. Prompt the human to batch-update `DESIGN.md §6` at PR merge.
2. Update `docs/architecture/ROADMAP.md` Features section to mark resolved.

No code-level carry-overs; nothing in the Stage 4 deferred list adds new
work to Stage 5's implementation.

---

## §E — External resource: `gov.uk/bank-holidays.json`

**URL:** `https://www.gov.uk/bank-holidays.json`
**Auth:** None — fully public, no API key.
**Method:** GET; returns JSON directly; `Content-Type: application/json`.

**Response shape** (stable since at least 2012):
```json
{
  "england-and-wales": {
    "division": "england-and-wales",
    "events": [
      {"title": "New Year's Day", "date": "2012-01-02", "notes": "", "bunting": true},
      ...
    ]
  },
  "scotland": { "division": "scotland", "events": [...] },
  "northern-ireland": { "division": "northern-ireland", "events": [...] }
}
```
Top-level keys: exactly `"england-and-wales"`, `"scotland"`,
`"northern-ireland"`. Each event has `title` (str), `date` (ISO 8601
`YYYY-MM-DD`), `notes` (str, often empty), `bunting` (bool).

**Historical depth:** Events go back to 2012-01-02 (first event in the
england-and-wales division). As of 2026, this is 14 years. Training
windows from 2018 (DESIGN §5.1) are fully covered. The Stage 5 intent
notes "about a decade back" — confirmed to be 14 years in practice.

**Update cadence:** The gov.uk team updates the file annually (typically
autumn / winter ahead of the following calendar year) and when
proclamations are issued for new holidays (e.g. extra bank holidays).
The endpoint is idempotent — a repeat GET of the same URL returns the
same payload until an update is published.

**Rate-limit / ToS:** No rate limit documented; no terms-of-service
beyond standard OGL v3 (Open Government Licence). A single GET per
`REFRESH` cache cycle is well within any reasonable budget; the
`min_inter_request_seconds` knob in `HolidaysIngestionConfig` can be
set to `0.0` (unlike the NESO CKAN endpoint which needs 30 s).

**Coverage note:** GB grid spans all three divisions. DESIGN §4.3
confirms "all three used because the GB grid spans all." A composite
`is_bank_holiday` column (any division is on holiday) is documented in
the intent §"Points for consideration" as the reasonable national-model
default.

**Schema for `ingestion/holidays.py` OUTPUT_SCHEMA (proposed):**

| Column | Arrow type | Notes |
|--------|-----------|-------|
| `date` | `pa.date32()` | ISO calendar date |
| `division` | `pa.string()` | `"england-and-wales"` / `"scotland"` / `"northern-ireland"` |
| `title` | `pa.string()` | Holiday name |
| `notes` | `pa.string()` | Supplementary text |
| `bunting` | `pa.bool_()` | Gov.uk flag |
| `retrieved_at_utc` | `pa.timestamp('us', tz='UTC')` | Provenance |

Primary key: `(date, division)` unique; sorted ascending.

---

## §F — Integration points with `train.py` and the notebook

### `src/bristol_ml/train.py`
- **Current binding** (lines 106–132): `fset = cfg.features.weather_only`.
  The cache path and `assembler.load()` call are derived from `fset`.
- **Minimal Stage 5 edit:** Introduce a feature-set selector, e.g.:
  ```python
  fset = cfg.features.weather_calendar or cfg.features.weather_only
  ```
  or a dedicated resolver function `_resolve_feature_set(cfg)` that
  returns a `(FeatureSetConfig, load_fn, feature_column_names)` triple.
  The `load_fn` distinguishes `assembler.load` (weather-only) from
  `assembler.load_calendar` (weather-calendar) because the two schemas
  are different and `load()` rejects extra columns.
- **`feature_columns` plumbing:** `train.py` currently calls `evaluate()`
  without an explicit `feature_columns` — it falls back to the weather
  default. For the calendar run, `train.py` must pass the calendar column
  list (or defer to `LinearConfig.feature_columns` being populated in
  the YAML).
- **Benchmark models in the three-way table** (train.py lines 196–199):
  Currently instantiates `NaiveModel(NaiveConfig())` and
  `LinearModel(LinearConfig())` unconditionally. For the calendar run,
  the linear model must use the calendar feature columns; the naive model
  is unaffected (it does not use feature columns).

### Notebook decision: extend `04_linear_baseline.ipynb` or add `05_calendar_features.ipynb`
- The intent says "a notebook" and names the without/with comparison as
  the demo moment (two metric tables side by side, two residual plots).
- Option A — **Add `notebooks/05_calendar_features.ipynb`**: cleaner
  pedagogical narrative; does not disturb the Stage 4 demo artefact;
  aligns with one-notebook-per-stage precedent (01, 02, 03, 04).
- Option B — **Extend `04_linear_baseline.ipynb`**: the without-result
  is already there; adding a second block avoids duplicating data-loading
  setup.
- **Flag for the plan:** the intent says "a notebook" without specifying
  which; Option A is the lower-surprise choice given the existing pattern.
  The plan must decide explicitly.

---

## §G — Files-list seed

### New files
| Path | Purpose |
|------|---------|
| `src/bristol_ml/ingestion/holidays.py` | Gov.uk bank-holidays ingester; `fetch`/`load` contract + VCR cassette backing |
| `src/bristol_ml/features/calendar.py` | Pure `derive_calendar(df, holidays_df) -> pd.DataFrame`; calendar feature derivation |
| `conf/ingestion/holidays.yaml` | `# @package ingestion.holidays`; endpoint URL, cache path |
| `conf/features/weather_calendar.yaml` | `# @package features.weather_calendar`; name, cache_filename, forward_fill_hours |
| `notebooks/05_calendar_features.ipynb` | Without/with demo notebook (plan decision — see §F) |
| `tests/unit/ingestion/test_holidays.py` | Unit + VCR-backed cassette tests for `holidays.py` |
| `tests/unit/features/test_calendar.py` | Unit tests for `derive_calendar` (pure function; no fixtures needed) |
| `tests/unit/features/test_assembler_calendar.py` | Tests for the calendar-enriched assembler / `load_calendar` |
| `tests/fixtures/holidays/cassettes/holidays_refresh.yaml` | VCR cassette for `gov.uk/bank-holidays.json` |
| `scripts/record_holidays_cassette.py` | Cassette-recorder script (mirrors `record_neso_cassette.py`) |
| `docs/lld/stages/05-calendar-features.md` | Stage 5 retrospective |
| `docs/plans/active/05-calendar-features.md` | Stage 5 plan (created at plan-writing time) |

### Modified files
| Path | Change |
|------|--------|
| `src/bristol_ml/features/assembler.py` | Add `CALENDAR_OUTPUT_SCHEMA`, `load_calendar()`, `assemble_calendar()` (or dispatcher); extend `__all__` |
| `conf/_schemas.py` | Add `HolidaysIngestionConfig`; extend `IngestionGroup` with `holidays: HolidaysIngestionConfig | None = None`; extend `FeaturesGroup` with `weather_calendar: FeatureSetConfig | None = None` |
| `conf/config.yaml` | Add `ingestion/holidays@ingestion.holidays` and `features/weather_calendar@features.weather_calendar` to `defaults:` list |
| `src/bristol_ml/train.py` | Extend `_cli_main` to resolve the active feature set via a Hydra override (e.g. `features=weather_calendar`) |
| `src/bristol_ml/features/__init__.py` | Update module docstring; optionally add `calendar` to submodule list |
| `src/bristol_ml/features/CLAUDE.md` | Document `calendar.py` public surface and `weather_calendar` schema |
| `src/bristol_ml/ingestion/CLAUDE.md` | Document `holidays.py` schema and cassette location |
| `CHANGELOG.md` | `### Added` bullets for the two new modules, YAML groups, notebook, tests |
| `docs/architecture/layers/features.md` | Update module inventory: mark `weather_calendar` as shipped |
| `docs/architecture/ROADMAP.md` | Close "Features" open questions per §D carry-over 2 |
| `docs/stages/README.md` | Flip Stage 5 status cell to `shipped` |
| `docs/plans/completed/05-calendar-features.md` | Move from `active/` at PR merge |
| `README.md` | Add `notebooks/05_calendar_features.ipynb` entry point |

**Deny-tier for the lead (human edit required):**
- `docs/intent/DESIGN.md §6` layout tree — accumulate Stages 1–5 additions.

