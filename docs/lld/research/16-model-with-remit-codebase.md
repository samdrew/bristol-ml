# Stage 16 — Model with REMIT features: codebase map

**Source intent:** `docs/intent/16-model-with-remit.md`
**Artefact role:** Phase 1 research deliverable (codebase explorer).

---

## Relevant files

**Upstream data sources (read-only for Stage 16)**

- `/workspace/src/bristol_ml/ingestion/remit.py` — `fetch`, `load`, `as_of`, `OUTPUT_SCHEMA`; the 16-column bi-temporal event log; stub via `BRISTOL_ML_REMIT_STUB=1`
- `/workspace/src/bristol_ml/llm/__init__.py` — `RemitEvent`, `ExtractionResult` (Pydantic, frozen, extra="forbid"); `Extractor` Protocol
- `/workspace/src/bristol_ml/llm/extractor.py` — `StubExtractor`, `LlmExtractor`, `build_extractor(config, *, gold_set_path) -> Extractor`; **NO on-disk persistence of results**
- `/workspace/tests/fixtures/llm/hand_labelled.json` — 76-record gold set consumed by `StubExtractor`
- `/workspace/tests/fixtures/remit/cassettes/remit_2024_01_01.yaml` — ~20 kB cassette for cassette-based integration tests

**Features layer (primary write target)**

- `/workspace/src/bristol_ml/features/assembler.py` — `OUTPUT_SCHEMA` (10 cols), `CALENDAR_OUTPUT_SCHEMA` (55 cols), `build`, `load`, `load_calendar`, `assemble`, `assemble_calendar`; `load()` does EXACT column-count checking
- `/workspace/src/bristol_ml/features/calendar.py` — `derive_calendar`, `CALENDAR_VARIABLE_COLUMNS`
- `/workspace/src/bristol_ml/features/weather.py` — `national_aggregate`
- `/workspace/src/bristol_ml/features/fourier.py` — `append_weekly_fourier` (pure, no I/O)
- NEW: `/workspace/src/bristol_ml/features/remit.py` — to be created; REMIT feature derivation module

**Config (must extend)**

- `/workspace/conf/_schemas.py` — `FeaturesGroup` (`extra="forbid"`, frozen=True) currently has only `weather_only` and `weather_calendar` fields; `AppConfig` aggregates everything; `FeatureSetConfig` is the leaf type for each feature set
- `/workspace/conf/config.yaml` — defaults list drives Hydra group-swap; currently `- features: weather_only`
- `/workspace/conf/features/weather_only.yaml` and `/workspace/conf/features/weather_calendar.yaml` — templates for new `with_remit.yaml`
- `/workspace/conf/ingestion/remit.yaml` — already in `config.yaml` defaults (`- ingestion/remit@ingestion.remit`)

**Training entry point (must extend)**

- `/workspace/src/bristol_ml/train.py` — `_resolve_feature_set(cfg)` currently dispatches only on `weather_only` / `weather_calendar`; must gain a `with_remit` arm; `_cli_main` handles model instantiation (no changes needed there for Stage 16 since no new model type is introduced)

**Evaluation (read-only)**

- `/workspace/src/bristol_ml/evaluation/harness.py` — rolling-origin `evaluate`, `evaluate_and_keep_final_model`; returns metrics DataFrame with one row per fold
- `/workspace/src/bristol_ml/evaluation/benchmarks.py` — NESO three-way comparison
- `/workspace/src/bristol_ml/evaluation/metrics.py` — mae, mape, rmse, wape

**Registry (read-only)**

- `/workspace/src/bristol_ml/registry/__init__.py` — `save(model, metrics_df, *, feature_set, target)`, `load`, `list_runs`, `describe`; `list_runs` accepts a `feature_set=` filter so `"with_remit"` will naturally slot in
- `/workspace/src/bristol_ml/registry/_dispatch.py` — 6-family `_TYPE_TO_CLASS` dict; no change needed for Stage 16

**Models (retrain target)**

- `/workspace/src/bristol_ml/models/temporal.py` — `NnTemporalModel` (TCN); best-performing per Stage 11
- `/workspace/conf/model/nn_temporal.yaml` — CUDA defaults; CPU recipe in file header comment

**Notebooks**

- `/workspace/notebooks/03_feature_assembler.ipynb` — currently modified (per git status); the ablation notebook for Stage 16 is new, following the style of this notebook

**Tests to examine**

- `/workspace/tests/unit/features/test_assembler.py` — programmatic fixture generation pattern; exact-schema assertions
- `/workspace/tests/unit/features/test_assembler_calendar.py` — calendar extension pattern to mirror for `with_remit`
- `/workspace/tests/unit/ingestion/test_remit.py` — `as_of` contract tests
- `/workspace/tests/unit/llm/` — extractor unit tests (4 files)

---

## Data flow / call graph for Stage 16 new work

```
remit.fetch / remit.load
        |
        v
remit.as_of(df, t)            -- transaction-time filter
        |
        + valid-time filter: effective_from <= t, effective_to.isna() | effective_to > t
        |
        v
features/remit.py::derive_remit_features(remit_df, hourly_index)
   -- aggregate per hour: total affected_mw by fuel_type bucket,
      optionally LLM-extracted affected_capacity_mw from ExtractionResult
   -- optionally build forward-looking columns: sum of affected_mw over [t, t+24h)
   -- returns wide frame keyed on timestamp_utc
        |
        v
features/remit.py::REMIT_VARIABLE_COLUMNS   -- defines new columns + pa dtypes
features/assembler.py::WITH_REMIT_OUTPUT_SCHEMA  -- extends CALENDAR_OUTPUT_SCHEMA prefix
        |
        v
features/assembler.py::assemble_with_remit(cfg, *, cache) -> Path
   -- calls assemble_calendar inline (cannot delegate — mutual exclusivity)
   -- appends derive_remit_features output
   -- writes parquet via _atomic_write
        |
        v
features/assembler.py::load_with_remit(path) -> pd.DataFrame
   -- EXACT schema check against WITH_REMIT_OUTPUT_SCHEMA
        |
        v
train.py::_resolve_feature_set(cfg)         -- new with_remit arm
        |
        v
evaluation/harness.py::evaluate_and_keep_final_model
        |
        v
registry.save(model, metrics_df, feature_set="with_remit", target="nd_mw")
```

The LLM extraction sub-path is optional within `derive_remit_features`:

```
llm.extractor.build_extractor(cfg.llm, *, gold_set_path)
        |
        v
extractor.extract_batch(events) -> list[ExtractionResult]
        |   (no on-disk persistence exists — Stage 16 must persist or inline)
        v
merge ExtractionResult.affected_capacity_mw onto remit_df by (mrid, revision_number)
```

---

## Patterns Stage 16 must conform to

**1. Feature-module shape** (cite: `features/CLAUDE.md`, `docs/architecture/layers/features.md`)

Every feature module has exactly four public things: pure derivation function(s), `OUTPUT_SCHEMA`-equivalent constant, `load()`, and `assemble*()` orchestrator. No I/O inside derivation functions. The new `features/remit.py` must follow this shape.

**2. Exact-schema load()** (cite: `assembler.py` lines 60–62; `test_assembler.py`)

`assembler.load()` rejects frames with missing OR extra columns. The new `load_with_remit()` must do the same against `WITH_REMIT_OUTPUT_SCHEMA`. Tests use programmatic fixtures generated from `OUTPUT_SCHEMA` — Stage 16 tests must generate from `WITH_REMIT_OUTPUT_SCHEMA`.

**3. FeaturesGroup extension requires two simultaneous file edits** (cite: `conf/_schemas.py`)

`FeaturesGroup` has `extra="forbid"` and `frozen=True`. Adding `with_remit: FeatureSetConfig | None = None` requires a code change to `conf/_schemas.py` AND a new YAML file `conf/features/with_remit.yaml`. These must land together; a YAML without the schema field will fail validation; a schema field without the YAML will fail Hydra composition.

**4. _resolve_feature_set mutual-exclusivity invariant** (cite: `train.py` lines 87–132)

The resolver enforces that exactly one field in `FeaturesGroup` is non-None at runtime. The `with_remit` arm must follow the same `if X is not None and Y is None and Z is None` pattern. The error message must name the Hydra override (`features=with_remit`).

**5. Atomic writes** (cite: `ingestion/_common.py::_atomic_write`, `assembler.py`, `registry/CLAUDE.md`)

All `assemble*()` orchestrators use `_atomic_write` from `bristol_ml.ingestion._common`. The new `assemble_with_remit` must import and use the same helper.

**6. Structured logging** (cite: `assembler.py::build`, `features/CLAUDE.md`)

Every `build`-equivalent call emits a single structured INFO log line per call with row counts (dropped / filled at each step). `derive_remit_features` must emit a single structured INFO line; one WARNING when no REMIT events are found for an interval.

**7. Stub-first for external dependencies** (cite: `CLAUDE.md`, `ingestion/remit.py`, `llm/extractor.py`)

The LLM path inside `derive_remit_features` must be gated behind `BRISTOL_ML_LLM_STUB=1` / `build_extractor(None)` returning a `StubExtractor`. The REMIT fetch path is already gated behind `BRISTOL_ML_REMIT_STUB=1`. CI must pass with both stubs active.

**8. Notebooks are thin** (cite: `CLAUDE.md §2.1.8`)

The Stage 16 ablation notebook imports from `bristol_ml`; it does not reimplement aggregation logic inline.

**9. Registry artefacts are skops** (cite: `registry/CLAUDE.md`, Stage 12 D10)

`NnTemporalModel.save` already writes skops. No change needed at the registry boundary; note the hazard below.

**10. `FeatureSetConfig.name` becomes the `feature_set` string in the registry** (cite: `registry/__init__.py`)

The sidecar stores `feature_set` as the string name from `FeatureSetConfig.name`. The YAML field `name: with_remit` must match what downstream `list_runs(feature_set="with_remit")` uses.

---

## Empirical evidence for "best model"

**Stage 11 retro** (`/workspace/docs/lld/stages/11-complex-nn.md`), CPU recipe, 90-day window, single-holdout:

| Model | MAE (MW) |
|---|---|
| `nn_temporal` (TCN, 3 blocks × 16 ch) | **3768.6** — best |
| `naive` | 4354.0 |
| `nn_mlp` | 5241.1 |
| `linear` | 5347.8 |
| `sarimax` | 12334.7 |
| `scipy_parametric` | 220249.9 |

**Stage 7 retro** (`/workspace/docs/lld/stages/07-sarimax.md`), rolling-origin harness, larger window:

| Model | MAE (MW) |
|---|---|
| `sarimax` | **1730** — best under rolling-origin |
| `linear` | 1955 |
| `naive` | 2080 |

**The ambiguity the plan author must resolve:** `nn_temporal` is best under Stage 11's predict-only single-holdout protocol. `sarimax` is best under Stage 7's rolling-origin harness. The two protocols are not directly comparable. The intent says "best-performing model" without specifying which protocol is authoritative. If rolling-origin is the canonical protocol (the evaluation harness default), SARIMAX is the empirical winner; if single-holdout predict-only is the reference, TCN wins. **This must be declared in the Stage 16 plan before implementation begins.**

---

## Hazards

**H1 — Missing persistence layer for ExtractionResult.**
`llm/extractor.py::extract_batch` returns `list[ExtractionResult]` in memory; there is no `extract_and_persist()` function and no parquet schema for results. Stage 16 must decide: run extraction on every `assemble_with_remit()` call (slow, non-idempotent unless cached), or add a one-time persistence step that saves results to e.g. `data/processed/remit_extracted.parquet` before the assembler reads them. The latter is safer and more consistent with the ingestion-layer pattern but requires a new schema and a new step in the pipeline. The `features/remit.py` module should own only the aggregation; the extraction + persistence probably belongs in a new `ingestion/remit_extracted.py` or as a CLI step before `assemble_with_remit`.

**H2 — Per-hour bi-temporal aggregation is O(n_hours × n_events) naively.**
`as_of(df, t)` is called once per query timestamp. For a 7-year training window (~61 000 hours) calling it per hour is ~61 000 pandas groupby passes over the full REMIT frame. Stage 16 must vectorise: sort once by `published_at`, then process the hourly grid in a single pass using a "latest revision as-of" trick (e.g., merge_asof or a backward rolling window on the sorted log).

**H3 — `message_description` is frequently NULL on live Elexon API.**
Documented in the Stage 14 LLM layer doc and confirmed by the `hand_labelled.json` fixture. The LLM extraction path degrades gracefully (`StubExtractor` returns `confidence=0.0` on a miss), but if `affected_mw` is also NULL (which it can be per the schema — nullable `float64`), Stage 16 may produce mostly-zero REMIT columns. The intent acknowledges this and requires the ablation notebook to report the fraction of non-zero rows honestly; it explicitly allows a negative result.

**H4 — `FeaturesGroup.extra="forbid"` means adding `with_remit` silently breaks pre-Stage-16 config files if they try to set `features.with_remit`.**
This is the desired behaviour, but it means the with_remit YAML must not be in the defaults list of `config.yaml` until the schema field lands. The two edits (`conf/_schemas.py` field + `conf/features/with_remit.yaml`) must be a single atomic commit; a partial state (schema without YAML or YAML without schema) will fail either Pydantic or Hydra, making CI red.

**H5 — `assemble_with_remit` cannot delegate to `assemble_calendar`.**
The `assemble_calendar` function requires `cfg.features.weather_calendar` to be non-None; the `with_remit` config uses `cfg.features.with_remit`. Mutual exclusivity makes delegation impossible (same constraint that prevented `assemble_calendar` from delegating to `assemble`). The code duplication is acknowledged in the CLAUDE.md under "Stage 5 notes"; the Stage 16 plan should note the same pattern and the candidate future refactor of `_compose_weather_only_frame(cfg, fset, *, cache)`.

**H6 — `NnTemporalConfig` has a `@model_validator` that rejects small `seq_len` vs architecture receptive field.**
The CPU recipe (3 blocks × 16 channels) has a much smaller receptive field than the CUDA defaults (8 blocks). If the Stage 16 plan intends to add REMIT columns to the feature set, `feature_columns` must list them explicitly or the model will silently use all columns including provenance scalars (`neso_retrieved_at_utc`, etc.). Pattern: set `feature_columns` in the `nn_temporal.yaml` or override it to exclude provenance columns.

**H7 — `load_with_remit` must be added to `assembler.py`'s `__all__`, and `assemble_with_remit` likewise.**
The existing `__all__` in `assembler.py` is explicit. Missing these will cause `from bristol_ml.features.assembler import load_with_remit` to fail if the caller relies on `__all__` for discovery. The registry surface test (`test_registry_public_surface_does_not_exceed_four_callables`) applies only to the registry module, not the assembler, so the assembler `__all__` can grow freely — but it must grow explicitly.

**H8 — Forward-looking REMIT features require a different aggregation signature.**
The intent says Stage 16 "may" expose "known unavailability over the next 24 hours". This requires `sum_affected_mw_next_24h(t)` = sum of `affected_mw` for events where `effective_from` is in `[t, t+24h)` as known at `t` (i.e. `as_of(df, t)` first, then future valid-time filter). This is a different function from the point-in-time active-events aggregation. If both backward-looking and forward-looking features are built, `derive_remit_features` must accept a `horizon_hours` parameter; leaving it at 0 produces only backward-looking features. The plan author should decide whether to include forward-looking features in the initial implementation or defer.
