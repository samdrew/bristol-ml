# Requirements — Stage 3: Feature Assembler + Train/Test Split

**Produced by:** requirements-analyst agent, 2026-04-19
**Sources consulted:**
- Intent: `docs/intent/03-feature-assembler.md`
- Spec: `docs/intent/DESIGN.md` §2, §3, §5, §7, §9
- Stage 1 retro: `docs/lld/stages/01-neso-demand-ingestion.md`
- Stage 2 retro: `docs/lld/stages/02-weather-ingestion.md`
- Ingestion layer contract: `docs/architecture/layers/ingestion.md`
- Ingestion module guide (shipped schemas): `src/bristol_ml/ingestion/CLAUDE.md`
- Features module guide: `src/bristol_ml/features/CLAUDE.md`
- LLD Stage 1: `docs/lld/ingestion/neso.md`
- LLD Stage 2: `docs/lld/ingestion/weather.md`

---

## 1. Goal

Produce a canonical, schema-enforced hourly feature table by joining the Stage 1 half-hourly demand parquet and the Stage 2 long-form weather parquet, and provide a rolling-origin split utility that yields chronologically ordered, non-overlapping train/test index pairs, so that every subsequent modelling stage trains and evaluates on identical inputs without relitigating alignment or split logic.

---

## 2. User Stories

### US-1 — Demoer (meetup facilitator)

**Given** Stages 1 and 2 caches are warm on a laptop,
**When** the facilitator runs `python -m bristol_ml.features.assembler` (or equivalent single command),
**Then** a parquet file appears at a documented, configurable path, and the terminal prints its schema and row count — so an attendee can see "the feature table" as a concrete artefact in under a minute, without needing to understand the upstream join logic.

### US-2 — Demoer (notebook walk-through)

**Given** the feature table parquet file exists,
**When** the facilitator runs the Stage 3 notebook top-to-bottom,
**Then** the schema is printed and a plot of rolling-origin folds overlaid on a demand time series is displayed — so attendees of mixed ability can see where training ends and testing begins for each fold (Intent §Demo moment).

### US-3 — Downstream modeller (Stage 4 and later)

**Given** a fully-assembled feature table parquet at the documented path,
**When** a model stage calls `load(path)` and passes the resulting DataFrame and a split configuration to the rolling-origin splitter,
**Then** it receives integer index arrays for each fold with no data-leakage from test into train — so the model stage can begin fitting without writing any join, resampling, or split code of its own (Intent §Purpose).

### US-4 — Maintainer extending the feature set (Stage 5+)

**Given** the assembler is built with a named feature-set identifier (e.g. `weather_only`),
**When** Stage 5 adds calendar features and names its set distinctly (e.g. `weather_calendar`),
**Then** both sets can co-exist in the config and be selected at runtime via a Hydra override — so the without/with comparison at Stage 5 is a config swap, not a code change (Intent §Points for consideration item 8; DESIGN §7.4).

### US-5 — Maintainer auditing a derived artefact

**Given** a feature table parquet file on disk,
**When** the maintainer reads its provenance metadata,
**Then** they can identify the git SHA, wall-clock timestamp, and source parquet paths that produced it — so they can reproduce or invalidate the file (DESIGN §2.1.6).

---

## 3. Acceptance Criteria

### AC-1 — Determinism

**Given** the two upstream parquets are byte-identical across two runs,
**When** the assembler is invoked twice,
**Then** the output parquet files are byte-identical modulo any per-run provenance timestamp column.

*Corresponds to Intent AC-1.*

### AC-2 — Schema conformance

**Given** the assembler has produced its output file,
**When** a schema assertion is run against that file,
**Then** it passes: all required columns are present, types match the declared schema, and no undeclared columns are present.

*Corresponds to Intent AC-2. The mechanism by which schema is enforced (pandera vs Pydantic vs pyarrow schema assertion) is an open question; see OQ-6.*

### AC-3 — Splitter non-overlap and chronological discipline

**Given** a time-indexed DataFrame and a split configuration (test window length, optional step size),
**When** the rolling-origin splitter yields `(train_idx, test_idx)` pairs,
**Then** for every fold:
- the maximum index in `train_idx` is strictly less than the minimum index in `test_idx`,
- the indices within each of `train_idx` and `test_idx` are in ascending order,
- no index appears in both `train_idx` and `test_idx` of the same fold,
- no index appears in the test set of an earlier fold and the training set of a later fold unless the later fold's training window has explicitly grown to include it.

*Corresponds to Intent AC-3. The last sub-criterion addresses the expanding-window vs sliding-window ambiguity; see OQ-5.*

### AC-4 — Splitter returns index arrays

**Given** a call to the rolling-origin splitter,
**When** the return value is inspected,
**Then** it yields pairs of array-like integer position indices (compatible with `iloc`), not boolean masks or label-based index objects (Intent AC-4).

### AC-5 — Notebook runs quickly on a laptop

**Given** the feature table parquet and the upstream caches are warm,
**When** the Stage 3 notebook is executed top-to-bottom,
**Then** it completes in under **120 seconds** on a laptop with no GPU.

*Corresponds to Intent AC-5. The 120-second threshold is [PROPOSED]; it is consistent with DESIGN §11 OQ-1 ("under ~2 minutes") and the Stage 1 retro's "under 30 seconds when cache is warm" target for a simpler notebook. Lead should confirm.*

### AC-6 — Tests: assembler smoke test and splitter behavioural tests

**Given** a small fixture derived from the actual parquet schemas,
**When** the test suite is run,
**Then**:
- a smoke test on the assembler's public interface passes against the fixture and asserts the output schema,
- at least one test asserts the splitter's no-overlap property on a minimal synthetic time-indexed frame,
- at least one test asserts the splitter's chronological-order property,
- no test is skipped or marked `xfail` without a linked issue (DESIGN §9).

*Corresponds to Intent AC-6.*

---

## 4. Non-Functional Requirements

### NFR-1 — Determinism (strong)

The assembler is a pure transformation (parquet in, parquet out). Identical inputs → row-set-identical and schema-identical output across runs, machines, and session orders. The only permitted per-run variation is a provenance timestamp column (analogous to `retrieved_at_utc` in the ingestion layer, per DESIGN §2.1.6).

### NFR-2 — Laptop performance envelope

- Assembler cold run (join + write), 2018–2025 range (~8 years × 8760 hours): under **30 seconds** on a laptop with warm upstream caches. [PROPOSED]
- Rolling-origin splitter: index generation for a 12-month test window with 1-day step (≈365 folds): under **1 second**. [PROPOSED]
- These thresholds support the notebook-under-120-seconds budget (AC-5).

### NFR-3 — Disk footprint

The derived feature table (weather-only, 2018–2025, hourly) should be under **50 MB** compressed Parquet. Not committed to the repo (regenerable). Location configurable and documented (Intent §Points for consideration item 7).

### NFR-4 — British English

All module docstrings, public function docstrings, and user-facing terminal strings use British English spelling (CLAUDE.md conventions).

### NFR-5 — Type-hint coverage

All public function signatures in `features/assembler.py` and `evaluation/rolling_origin.py` carry complete type hints. `# type: ignore` comments require an explanatory note (DESIGN §2.1.2).

### NFR-6 — Module-standalone

Both `features/assembler.py` and `evaluation/rolling_origin.py` are invocable as `python -m bristol_ml.features.assembler` and `python -m bristol_ml.evaluation.rolling_origin`, producing visible output without an orchestrator (DESIGN §2.1.1).

### NFR-7 — Notebook thinness

The Stage 3 notebook imports from `src/bristol_ml/` for all non-trivial logic. No inline reimplementation of join, aggregation, resampling, or split logic. Display and plot code is permitted (DESIGN §2.1.8).

### NFR-8 — Idempotence

Re-running the assembler overwrites the output file atomically (same `os.replace` pattern as the ingestion layer) and never corrupts the existing file on partial failure (DESIGN §2.1.5; ingestion layer convention).

### NFR-9 — Configuration externalised

Feature-set selection (variables, date range, output path) and splitter parameters (test window length, step, minimum training window) live in YAML under `conf/`, validated by Pydantic schemas in `conf/_schemas.py` (DESIGN §2.1.4, §7).

### NFR-10 — Provenance

The output parquet carries at minimum: git SHA, wall-clock timestamp, paths or content hashes of upstream parquets consumed. Exact schema is a design decision; data must be present (DESIGN §2.1.6).

---

## 5. Out of Scope

Explicitly deferred — must not be included in Stage 3 even partially:

- Calendar features including bank holidays (Stage 5).
- Lag features of any kind (deferred indefinitely).
- REMIT features (Stages 13–16).
- Multi-horizon splits beyond the day-ahead horizon.
- A feature store as a separate service (DESIGN §10).
- Metric functions (`MAE`, `MAPE`, `RMSE`, `WAPE`) — Stage 4.
- The NESO benchmark comparison — Stage 4.
- Any model training whatsoever.
- Forecast weather for serving/inference — only historical reanalysis is in scope.
- Sub-hourly or multi-resolution output.

---

## 6. Open Questions

Each carries evidence that would close it. None should be silently resolved during implementation.

**OQ-1 — Half-hourly to hourly aggregation convention.**
Intent §Points for consideration flags "aggregating within local-time hours and relabelling to UTC" as one option; the aggregation function (mean / sum / peak) is also unspecified. The choice affects all downstream modelling (a mean `nd_mw` has different units and interpretation than a sum). Evidence to close: project-author statement on the canonical convention for GB demand modelling; consistency with NESO day-ahead benchmark comparison at Stage 4.

**OQ-2 — Clock-change day handling at the join boundary.**
The NESO ingester resolves half-hourly timestamps to UTC before writing parquet (Stage 1 retro). The weather feed is natively UTC (Stage 2 retro). If both sides are already UTC-indexed the join is clean, but the aggregation from half-hourly to hourly (OQ-1) must still happen in UTC, not local time. Evidence to close: a worked example on 31 March and 27 October against the Stage 1 parquet, plus explicit tests.

**OQ-3 — Missing data policy.**
Intent flags short-gap forward-fill for weather vs drop-row for demand as "defensible" options. Decision needs documenting in the assembler docstring. Evidence to close: project-author preference, informed by measured gap frequency in 2018–2025 data.

**OQ-4 — Feature-set naming and Hydra config group design.**
Stage 3 is weather-only; Stage 5 extends with calendar features. The names chosen here become config-group keys Stage 5 adds alongside. Options: `weather_only` / `weather_calendar`; `v1` / `v2`; `base` / `full`. Evidence to close: project-author preference; existing naming convention in `conf/`.

**OQ-5 — Expanding vs sliding window rolling-origin split.**
DESIGN §5.1 says "rolling-origin evaluation with one-day step" but does not specify whether the training window expands or stays fixed-size. These differ in variance-bias trade-offs. Evidence to close: DESIGN §5.3 favours expanding window as a benchmark discipline; author confirmation requested.

**OQ-6 — Schema enforcement mechanism.**
Intent flags `pandera` vs Pydantic as "probably warrants a small ADR". `pandera` is not yet a project dependency. Requirements require that (a) schema is declared explicitly, (b) enforced at output boundary (AC-2), (c) the choice is recorded in an ADR if it introduces a new dependency. Evidence to close: author preference; see research doc for options comparison.

**OQ-7 — Output file location and "raw" vs "derived" treatment.**
The ingestion layer uses `data/raw/<source>/`. The feature table is a derived artefact and should not live under `raw/`. Options: `data/features/`, `data/processed/`. The ingestion layer's `cache_dir` convention (`BRISTOL_ML_CACHE_DIR` env var) may or may not extend naturally. Evidence to close: author preference.

**OQ-8 — Station list discrepancy between LLD and shipped code.**
Stage 2 retro documents that the shipped station list differs from the LLD. `docs/lld/ingestion/weather.md` §3.2 lists Belfast and Cardiff; the shipped list (in `ingestion/CLAUDE.md`) has Sheffield and Liverpool instead. Stage 3 `national_aggregate` callers must use the shipped list. Not a blocker, but a trap.

**OQ-9 — `national_aggregate` signature: shipped vs LLD.**
Shipped: `national_aggregate(df, weights: Mapping[str, float])`. LLD sketch: `national_aggregate(df, config, *, stations=None, variables=None)`. Callers must extract `{s.name: s.weight for s in cfg.ingestion.weather.stations}` at the boundary. Implementer must code to the **shipped** signature — verify in `src/bristol_ml/features/weather.py`.

**OQ-10 — Notebook runtime budget (DESIGN §11 OQ-1).**
DESIGN §11 asks "should notebooks run top-to-bottom in under ~2 minutes on a laptop?" — unresolved at the spec level. AC-5 proposes 120 seconds, consistent with that question. Author confirmation needed before the implementer codes to a target.

---

## Summary for the lead — thorniest open questions

Four questions carry the most risk:

- **OQ-1 (aggregation convention)** — most consequential: a wrong default requires retraining all downstream models.
- **OQ-5 (expanding vs sliding)** — second most consequential: reshapes what "rolling origin" means in every subsequent chapter.
- **OQ-6 (schema enforcement)** — decides whether `pandera` becomes a new dependency; needs lead sign-off before `pyproject.toml` edits.
- **OQ-9 (shipped `national_aggregate` signature)** — implementer trap: both the Stage 2 LLD and `features/CLAUDE.md` document a stale signature; only the shipped source is authoritative.
