# Stage 5 — Calendar features: requirements

Structured intent → engineering-actionable requirements. Synthesised by the requirements-analyst from [`docs/intent/05-calendar-features.md`](../../intent/05-calendar-features.md). Paired with [`05-calendar-features-codebase-map.md`](./05-calendar-features-codebase-map.md) and [`05-calendar-features-external-research.md`](./05-calendar-features-external-research.md); the three together are the Phase 1 inputs to [`docs/plans/active/05-calendar-features.md`](../../plans/active/05-calendar-features.md).

## Goal

Extend the feature layer with bank-holiday ingestion and calendar feature derivation, integrate them into the assembler as a selectable `weather_calendar` feature set, and deliver a notebook that demonstrates the measurable accuracy gain of feature engineering by running the Stage 4 `LinearModel` unchanged against both feature sets side by side.

---

## User stories

**Facilitator running the meetup demo**
- As a facilitator, I want to run `python -m bristol_ml.train features=weather_calendar` and see a better metric table appear beside the weather-only one, so that I can show the room exactly what domain knowledge is worth in one CLI invocation.
- As a facilitator, I want the notebook to display a residual-ripple plot over a shared week, so that I can point at the weekly pattern in the weather-only residuals and show it has vanished in the enriched residuals.

**Attendee self-learning**
- As an attendee, I want switching between `weather_only` and `weather_calendar` to be a configuration change with no code edit, so that I can understand how pluggable components work without touching Python.
- As an attendee, I want the calendar feature derivation to be a pure function I can call in isolation in a notebook cell, so that I can inspect and modify it without running the full pipeline.

**Project author as future maintainer**
- As the project author, I want bank-holiday dates cached offline-first to a provenance-recorded parquet, so that the demo works on a train with no network and every feature row is auditable.
- As the project author, I want the `OUTPUT_SCHEMA` extension for `weather_calendar` to follow the same declaration pattern as `weather_only`, so that Stage 6+ modelling stages can consume the enriched table without learning a new contract.

**Stage 6+ downstream consumer**
- As a downstream modelling stage, I want `features=weather_calendar` to yield a feature table that conforms to a declared `OUTPUT_SCHEMA`, so that I can select calendar columns without schema drift surprises.
- As a downstream modelling stage, I want the calendar derivation to produce only pure date-arithmetic features with no interaction terms, so that the without/with comparison in Stage 5 isolates the calendar contribution honestly and I can add interactions in a later stage without retroactively invalidating the Stage 5 result.

---

## Acceptance criteria

### From the intent (intent §Acceptance criteria, verbatim where quoted)

**AC-1** Switching between the weather-only feature set and the calendar-enriched feature set is a configuration change, not a code change.
- Given a warm feature cache and a fully installed environment,
- When the user runs `python -m bristol_ml.train features=weather_calendar` versus `features=weather_only`,
- Then both invocations complete successfully and print a metric table; no Python source file is edited between the two runs.

**AC-2** The calendar feature derivation is a pure function: same inputs, same outputs, no side effects.
- Given any pandas DataFrame with a tz-aware UTC `timestamp_utc` column and a bank-holiday mapping,
- When `features.calendar.derive(df, holidays)` is called twice with identical inputs,
- Then both calls return byte-identical DataFrames, no files are written, no HTTP requests are made, and no global state is mutated.

**AC-3** The enriched feature table conforms to the schema once it has been extended.
- Given a `weather_calendar` assembler call with valid demand, weather, and holiday inputs,
- When `assembler.build(...)` returns,
- Then the output DataFrame passes pyarrow `OUTPUT_SCHEMA` validation (column names, order, dtypes, timezone metadata) with no NaN values in any row.

**AC-4** The notebook produces the comparison table end-to-end.
- Given warm ingestion caches for demand, weather, and bank holidays,
- When the notebook `notebooks/05_calendar_features.ipynb` is executed top-to-bottom,
- Then it prints two metric tables (weather-only and weather-calendar) and exits with no exceptions.

**AC-5** The weekly-residual pattern present in the weather-only model and absent in the enriched model is visible in the notebook.
- Given the same `LinearModel` and the same evaluation harness applied to both feature sets,
- When the notebook renders the residual-ripple visualisation over a shared test week,
- Then the weather-only residuals show a recognisable weekly oscillation and the weather-calendar residuals do not.

**AC-6** The bank-holiday ingestion is idempotent and offline-first against its cache.
- Given a primed `data/raw/holidays.parquet` on disk,
- When `ingestion.holidays.fetch(config, cache="offline")` is called a second time,
- Then no HTTP request is made, the returned path is identical, and the parquet is bit-for-bit unchanged.

### From DESIGN §9 (implicit definition of done)

**AC-7** CI green: all tests pass, no skipped, no `xfail` without a linked issue.
- Given the full test suite after Stage 5 changes land,
- When `uv run pytest -q` is executed,
- Then the exit code is 0 and the output contains no `SKIPPED` or `XFAIL` entries.

**AC-8** Module `CLAUDE.md` updated for every module created or meaningfully touched.
- Given the new `ingestion/holidays.py` and `features/calendar.py` modules plus the extended assembler,
- When an agent reads the relevant module `CLAUDE.md` files,
- Then the public surface, invariants, and Stage 5 changes are documented accurately.

**AC-9** `README.md` entry-point updated.
- Given the new `python -m bristol_ml.ingestion.holidays` and `python -m bristol_ml.features.calendar` CLIs,
- When a user reads `README.md`,
- Then both entry points are listed or the `train features=weather_calendar` invocation is mentioned.

**AC-10** Retrospective filed at `docs/lld/stages/05-calendar-features.md`.
- Given the stage is complete,
- When the LLD stages directory is inspected,
- Then a retrospective file exists following the template in `docs/lld/stages/00-foundation.md`.

**AC-11** `CHANGELOG.md` entry under `[Unreleased]`.
- Given the stage is merged,
- When `CHANGELOG.md` is read,
- Then `### Added` bullets cover the holidays ingester, calendar feature module, assembler extension, config groups, notebook, and tests.

**AC-12** Notebook demonstrates the output under the D7 time budget.
- Given warm caches (bank holidays, demand, weather),
- When `uv run jupyter nbconvert --to notebook --execute notebooks/05_calendar_features.ipynb --output /tmp/05_test_run.ipynb` is executed,
- Then the command exits 0 in under 120 seconds (inheriting the D7 ceiling from Stage 3 D7 and Stage 4 D7).

---

## Functional requirements

### Bank-holiday ingestion (`ingestion/holidays.py`)

- **F-1** Fetch from `https://www.gov.uk/bank-holidays.json` and write to `data/raw/holidays.parquet` (or the configured cache path) using `ingestion._common._atomic_write`. (DESIGN §4.3)
- **F-2** Expose `fetch(config, cache=CachePolicy.OFFLINE) -> Path` and `load(path) -> pd.DataFrame` matching the ingestion layer contract established by `neso.py` and `weather.py`. (`docs/architecture/layers/features.md` §2)
- **F-3** On-disk `OUTPUT_SCHEMA` declared as a module-level `pa.Schema` constant with at minimum: `division` (string), `date` (date32), `name` (string), `retrieved_at_utc` (timestamp `us` UTC, scalar per fetch). (`docs/architecture/layers/features.md` §3 storage conventions)
- **F-4** Idempotent: a second `fetch` call with `cache="offline"` on a warm cache makes no HTTP request and returns the same path. (intent §AC-6, DESIGN §2.1.5)
- **F-5** Standalone CLI `python -m bristol_ml.ingestion.holidays` prints the output path and a schema summary. (DESIGN §2.1.1)
- **F-6** Covers all three GB divisions: `england-and-wales`, `scotland`, `northern-ireland`, as enumerated at `https://www.gov.uk/bank-holidays.json`. (DESIGN §4.3; intent §Scope)

### Calendar feature derivation (`features/calendar.py`)

- **F-7** Pure function `derive(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame` — frame in, frame out, no I/O. (`docs/architecture/layers/features.md` §2 "Pure derivations"; intent §AC-2)
- **F-8** Produces at minimum the following columns from the UTC `timestamp_utc` index: `hour_of_day` (int8), `day_of_week` (int8, Monday=0), `month` (int8), `is_weekend` (bool or int8), `is_holiday` (int8, 0/1), `holiday_proximity` (see D-5 for shape). (intent §Scope, §Points for consideration)
- **F-9** `is_holiday` derived by checking whether the UTC timestamp's local-date component falls on any bank-holiday date in the composited GB holiday set. The DST mapping rule (UTC hour → local date) stated explicitly in the module docstring. (intent §Points for consideration "Bank holidays are date-level, but the feature table is hourly")
- **F-10** Encoding strategy for `hour_of_day` and `day_of_week` resolved at plan time (see D-3, D-4); the derivation function must not hard-code an encoding that cannot be changed without touching production code.
- **F-11** Standalone CLI `python -m bristol_ml.features.calendar` loads the weather-calendar feature cache and prints a schema summary. (DESIGN §2.1.1)

### Assembler extension (`features/assembler.py`)

- **F-12** A second feature-set configuration `conf/features/weather_calendar.yaml` added alongside `conf/features/weather_only.yaml`; switching is a Hydra override `features=weather_calendar`. (intent §AC-1; `docs/plans/completed/03-feature-assembler.md` D3)
- **F-13** A new `build_calendar` function (or an extended `build` with a config discriminator) calls `features.calendar.derive` and joins the result to the demand+weather table on `timestamp_utc`. Join is inner; any row where the holiday lookup cannot be resolved is dropped and counted in the INFO log. (`docs/architecture/layers/features.md` §3; Stage 3 D5 log convention)
- **F-14** A new `CALENDAR_VARIABLE_COLUMNS` constant (or equivalent mechanism resolved under D-9) declares the calendar column names and their Arrow dtypes. (`src/bristol_ml/features/CLAUDE.md`)
- **F-15** `weather_calendar` assembler emits one structured loguru INFO line per `build` call covering `row_count`, calendar-column summary, and any rows dropped due to unresolvable holiday lookups. (`docs/lld/stages/03-feature-assembler.md` design choices — "Loguru for the D5 structured log")

### Notebook (`notebooks/05_calendar_features.ipynb`)

- **F-16** Calls the `train` harness twice — once with `features=weather_only`, once with `features=weather_calendar` — using the same `LinearModel` class and the same `SplitterConfig`, and prints both metric tables side by side with the NESO benchmark. (intent §Demo moment)
- **F-17** Renders the residual-ripple visualisation: residuals from both models overlaid over the same test week on shared axes with a legend distinguishing `weather_only` and `weather_calendar`. (intent §AC-5)
- **F-18** Thin per DESIGN §2.1.8: all derivation logic lives in `src/bristol_ml/`; notebook cells contain only imports, `load_config`, function calls, and display instructions.
- **F-19** Applies the `step=168` weekly override for notebook pacing, consistent with the Stage 4 D7 precedent. (see D-8)

---

## Non-functional requirements

- **NFR-1** Performance — feature derivation. `features.calendar.derive` on one year of hourly data (8 760 rows) completes in under 1 s on a developer laptop. (DESIGN §11 OQ-1)
- **NFR-2** Performance — notebook end-to-end. Notebook executes top-to-bottom in under 120 s on warm caches. (Stage 3 D7, Stage 4 D7)
- **NFR-3** Logging. Loguru INFO, one structured line per `build()` / `assemble()` call. (`docs/lld/stages/03-feature-assembler.md` design choices)
- **NFR-4** Idempotence. Re-running `ingestion.holidays.fetch` or `features.assembler.assemble(features=weather_calendar)` either overwrites or skips, never corrupts. Atomic writes via `ingestion._common._atomic_write` reused. (DESIGN §2.1.5)
- **NFR-5** British English in docstrings, user-facing log messages, and notebook prose. (CLAUDE.md)
- **NFR-6** Typed public API on all new public function signatures. No `# type: ignore` without an explanatory comment. (DESIGN §2.1.2)
- **NFR-7** Provenance. `weather_calendar` feature table carries the same `neso_retrieved_at_utc` / `weather_retrieved_at_utc` scalar provenance columns as `weather_only`, plus a new `holidays_retrieved_at_utc` scalar column. (DESIGN §2.1.6)
- **NFR-8** Stub-first safety. Bank-holiday ingester honours `CachePolicy.OFFLINE / AUTO / REFRESH` so tests never touch the network. (DESIGN §2.1.3)

---

## Open questions (decision points)

**D-1** Bank-holiday source. Proposed default: **`https://www.gov.uk/bank-holidays.json`** — no auth, DESIGN §4.3 identifies it. Alternatives: manually curated CSV (offline-first by construction; stale by definition); `workalendar` / `holidays` Python libraries (adds a dependency; better historical depth). Recommended: gov.uk JSON as the primary fetch, with offline-first caching. _ACCEPT / OVERRIDE._

**D-2** GB division composition. Proposed default: **single composite `is_holiday` = any-division-on-holiday**. Intent names this as "reasonable composite" for a national model; per-division columns can be added without schema breakage if the human wants them. _ACCEPT / OVERRIDE._

**D-3** Cyclical encoding of `hour_of_day`. Proposed default: **sin/cos pair** (`hour_sin = sin(2π·h/24)`, `hour_cos = cos(2π·h/24)`) as the only encoding. Rationale: interpretable continuous coefficients in OLS; pedagogically motivated at meetups; avoids 24 one-hot columns confounding the without/with story. Risk: less directly readable than a raw integer. _ACCEPT / OVERRIDE._

**D-4** Encoding of `day_of_week` and `month`. Proposed default: **one-hot `day_of_week`, ordinal `int8` `month`**. Rationale: weekday effects are large and non-linear, one-hot coefficients read cleanly; month effects are smoother and ordinal avoids January ≠ December discontinuity with lower column count. _ACCEPT / OVERRIDE._

**D-5** Holiday-proximity feature shape. Proposed default: **binary pair** `is_day_before_holiday` (int8) + `is_day_after_holiday` (int8). Rationale: readable OLS coefficients; minimal addition that captures the pedagogical point; `days_to_nearest_holiday` adds interpretability complexity for small marginal gain. _ACCEPT / OVERRIDE._

**D-6** Historical-depth fallback. `gov.uk/bank-holidays.json` covers approximately 2013 onwards; the Stage 4 training window starts 2018-01-01, so no mismatch today. Proposed default: **log a WARNING if any row falls before the earliest holiday date in the cache; fill `is_holiday=0` for pre-cache dates and continue**. _ACCEPT / OVERRIDE: raise `ValueError`._

**D-7** DST / UTC → local-date mapping rule. Proposed default: **`is_holiday=1` for any UTC hour whose `Europe/London` local-date component is a bank holiday, regardless of spring-forward / autumn-fallback**. Rule stated verbatim in the module docstring; unit test covers a holiday on a clock-change Sunday. _ACCEPT / OVERRIDE._

**D-8** Notebook step override. Proposed default: **inherit `evaluation.rolling_origin.step=168` in-cell override from Stage 4 D7**, applied to both the `weather_only` and `weather_calendar` runs for fair side-by-side. CLI default remains `step=24`. _ACCEPT / OVERRIDE._

**D-9** `OUTPUT_SCHEMA` extension mechanism. Proposed default: **separate `CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` constant + `WEATHER_CALENDAR_OUTPUT_SCHEMA: pa.Schema`** declared in `features/assembler.py`, superset of `weather_only`'s columns (same columns, calendar columns appended). Alternatives: dict keyed by feature-set name; separate modules `assembler_weather.py` / `assembler_calendar.py`. _ACCEPT / OVERRIDE._

---

## Out of scope

| Item | Stage that owns it |
|---|---|
| School holiday term dates (sourcing across four UK nations) | Deferred indefinitely (intent §Out of scope explicitly deferred) |
| Sporting events / one-off demand-affecting events | REMIT chain (Stages 13–16; intent §Out of scope) |
| Regional modelling (per-division demand) | Deferred indefinitely (intent §Out of scope) |
| Lag features | Stage 7+ (intent §Out of scope explicitly deferred) |
| Any new model class | Stage 5 uses the Stage 4 `LinearModel` unchanged (intent §Out of scope) |
| Changes to the rolling-origin split | Stage 6 (intent §Out of scope) |
| Interaction terms (e.g. temperature × weekday) | Future linear model enhancement — intent explicitly excludes them to keep the Stage 5 comparison honest |

---

## Housekeeping carry-overs

**H-1** `NesoBenchmarkConfig.holdout_start/_end` soft deviation (from Stage 4 retro "Deferred"). Schema fields exist but no consumer; Stage 6 is the stated natural consumer. Flag for human: Stage 6, not Stage 5.

**H-2** `ingestion._neso_dst` extraction (from Stage 4 retro "Deferred"). Settlement-period → UTC algebra duplicated in `neso.py` and `neso_forecast.py`. Stage 5 does not add another settlement-period ingester (holidays are date-level). Carry forward to Stage 13 (REMIT ingestion). No Stage 5 action.

**H-3** `DESIGN.md §6` layout tree accumulation. Stages 1–4 additions missing; Stage 5 will add further items (`ingestion/holidays.py`, `features/calendar.py`, `conf/features/weather_calendar.yaml`, `notebooks/05_calendar_features.ipynb`). Flag for human to perform a batched §6 edit covering Stages 1–5 as part of PR review. Lead MUST NOT touch §6 unilaterally (deny-tier).

**H-4** `docs/architecture/ROADMAP.md` Features section. Stage 3 retro deferred a ROADMAP pass to "Stage 5/6". Stage 5 resolves the `weather_calendar` entry in `features.md §Module inventory`. Lead updates ROADMAP.md at the close of Stage 5 (warn-tier write); not blocking for Phase 2.

---

## Tensions detected between intent and shipped Stage 3/4 architecture

**Tension 1 — `compare_on_holdout` signature drift.** The Stage 4 plan §5 advertised a `compare_on_holdout(models, df, neso_forecast, splitter_cfg, metrics, *, aggregation, target_column, feature_columns)` signature, but the shipped version derives the holdout window internally from fold test-periods rather than from `NesoBenchmarkConfig.holdout_start/_end` (Stage 4 retro "Design choices"). The Stage 5 notebook must call `compare_on_holdout` — the implementer must code to the *shipped* signature, not the plan's sketch. Tracked as H-1.

**Tension 2 — `harness.feature_columns` default follows `WEATHER_VARIABLE_COLUMNS`.** The Stage 4 harness default `feature_columns=None` resolves to "all weather float32 columns from `assembler.WEATHER_VARIABLE_COLUMNS`" (Stage 4 retro "Design choices"). When `weather_calendar` adds calendar columns, the harness default will NOT include them unless either (a) `WEATHER_VARIABLE_COLUMNS` is extended, or (b) the caller passes an explicit `feature_columns` override, or (c) the harness learns about a feature-set discriminator. Correct resolution is the crux of D-9 and must land before implementation to avoid a silent column-omission bug in the without/with comparison.
