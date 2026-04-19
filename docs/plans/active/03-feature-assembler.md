# Plan — Stage 3: Feature assembler + train/test split

**Status:** `approved` — all eight decisions resolved 2026-04-19; Phase 2 (implementation) in progress.
**Intent:** [`docs/intent/03-feature-assembler.md`](../../intent/03-feature-assembler.md)
**Upstream stages shipped:** Stage 0 (foundation), Stage 1 (NESO demand), Stage 2 (weather + national aggregate).
**Downstream consumers:** Stage 4 (linear baseline), Stage 5 (calendar features extends this assembler), every subsequent modelling stage.
**Baseline SHA:** `8e6b902` (tip of `reclaude` at plan time).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/requirements/03-feature-assembler.md`](../../lld/requirements/03-feature-assembler.md)
- Exploration — [`docs/lld/exploration/03-feature-assembler.md`](../../lld/exploration/03-feature-assembler.md)
- Research — [`docs/lld/research/03-feature-assembler.md`](../../lld/research/03-feature-assembler.md)

---

## 1. Decisions for the human (resolve before Phase 2)

Eight decision points surfaced in discovery. For each I propose a default that honours the simplicity bias and cite the evidence. Mark each `ACCEPT` / `OVERRIDE: <alt>` in your reply; I'll update the plan before starting implementation.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Half-hourly → hourly demand aggregation function | **`mean`** of the two half-hours per UTC hour, config-driven; pluggable with `max` as an alternative. `Literal["mean", "max"]` on the Pydantic model — any override is a config change, not a code change. | Preserves MW scale under the default, allows a peak-demand framing at Stage 4/6 without code churn, keeps the Literal tight so typos fail fast. | **DECIDED 2026-04-19.** Requirements OQ-1; Exploration Gotcha 1. |
| **D2** | Feature-table schema enforcement mechanism | **pyarrow `OUTPUT_SCHEMA`** (mirrors Stage 1 + Stage 2) — no new dependency, no ADR required | Stays within existing idiom; zero onboarding cost; one less framework to teach. | **DECIDED 2026-04-19.** Research §1 (recommends pandera but acknowledges simplicity trade-off); Exploration Gotcha 3; `src/bristol_ml/ingestion/neso.py:82`. |
| **D3** | Feature-set naming for Hydra group | **`weather_only`** (the config group key Stage 5 will sit beside as `weather_calendar`) | Intent explicitly warns against `default` naming; two distinct names keep Stage 5's with/without comparison a config swap. | **DECIDED 2026-04-19.** Intent §Points for consideration item 8; Exploration Gotcha 5. |
| **D4** | Rolling-origin window type | **Expanding window** default, with `fixed_window: bool` config knob for a sliding variant | Matches DESIGN §5.3 ("reports mean and spread across folds" — natural with expanding window); aligns with `sklearn.TimeSeriesSplit` semantics familiar to newcomers. | **DECIDED 2026-04-19.** Requirements OQ-5; Research §2. |
| **D5** | Missing-data policy | Drop rows where **demand is NaN**; **forward-fill weather up to 3 hours**, else propagate NaN and drop the row. **Additionally, log a structured summary** on every `build()` call: rows dropped (demand NaN), rows forward-filled per weather variable, rows dropped (weather NaN after fill cap). Log at `INFO` so it is audible in CLI and notebook runs without needing debug toggles. | Demand NaN on spring-forward day is not recoverable; short weather gaps are a data-quality artefact not a real signal. Logging makes the policy visible to a meetup audience and auditable in stage retros. The 3-hour cap is configurable. | **DECIDED 2026-04-19.** Requirements OQ-3; Exploration §5 Gotcha 1. |
| **D6** | Output file location | **`${oc.env:BRISTOL_ML_CACHE_DIR,data/features}/<feature_set_name>.parquet`** — mirrors ingestion `cache_dir` pattern | Consistent with Stage 1/2 env-var convention; keeps derived file outside the repo (regenerable); one URL pattern to teach. | **DECIDED 2026-04-19.** Requirements OQ-7; Exploration Gotcha 4. |
| **D7** | AC-5 notebook runtime budget | **120 seconds** (i.e. DESIGN §11 OQ-1's "under ~2 minutes") | Consistent with the spec's own open question; assembler cold-run budget 30 s + splitter 1 s + plot + narrative leaves ~60 s slack. | **DECIDED 2026-04-19.** Requirements AC-5, NFR-2; DESIGN §11 OQ-1. |
| **D8** | Provenance on the joined output | Two columns: **`neso_retrieved_at_utc`**, **`weather_retrieved_at_utc`** — scalar per run, not per row (DESIGN §2.1.6) | Straightforward propagation; both source fetches are auditable; cheaper in parquet than a metadata sidecar. | **DECIDED 2026-04-19.** DESIGN §2.1.6; Exploration §1 "Provenance columns". |

**Blocking note:** D1 and D2 are load-bearing; wrong defaults will require rework of every downstream model (D1) or a pyproject.toml + ADR churn mid-stage (D2). Please resolve both before Phase 2 begins.

---

## 2. Scope

### In scope

- A new `src/bristol_ml/features/assembler.py` that joins Stage 1 demand (resampled to hourly per D1) with Stage 2's national weather aggregate into a single hourly feature table, with a declared `OUTPUT_SCHEMA` enforced at the module boundary.
- A new `src/bristol_ml/evaluation/` module with a `splitter.py` that yields `(train_idx, test_idx)` integer-array pairs under rolling-origin rules (D4).
- New Hydra config groups `conf/features/weather_only.yaml` and `conf/evaluation/rolling_origin.yaml` wired into `conf/config.yaml` defaults.
- New Pydantic schemas `FeatureSetConfig`, `FeaturesGroup`, `SplitterConfig`, `EvaluationGroup` added to `conf/_schemas.py`, plus an `AppConfig.features` / `AppConfig.evaluation` field.
- A new `notebooks/03_feature_assembler.ipynb` that runs the assembler and visualises rolling-origin folds overlaid on a demand series.
- `python -m bristol_ml.features.assembler` and `python -m bristol_ml.evaluation.splitter` CLIs, each standalone.
- Unit tests for the splitter (non-overlap, chronological order, index-array shape) and a smoke test for the assembler against a small parquet fixture.
- Stage-hygiene updates: module `CLAUDE.md` files, retrospective, CHANGELOG entry, stages index status cell.

### Out of scope (do not accidentally implement)

From Intent §Out of scope: calendar features (Stage 5), lag features, REMIT features (13–16), multi-horizon splits, any model training, metric functions (Stage 4), the NESO benchmark comparison, feature store as a service.

Also out of scope for this plan:
- The `columns` dead-weight field on `NesoIngestionConfig` (noted in Exploration Gotcha 6) — defer to a separate housekeeping PR.
- Pandera introduction — unless D2 is overridden.
- An ADR — only required if D2 flips to pandera.

---

## 3. Reading order for the implementer

Read top-to-bottom before opening code:

1. `docs/intent/03-feature-assembler.md` — the spec. Where this plan disagrees, the spec wins.
2. `docs/lld/requirements/03-feature-assembler.md` — Given/When/Then acceptance criteria with citations back to the spec.
3. `docs/lld/exploration/03-feature-assembler.md` §1 + §5 — shipped schemas (Stage 1 + Stage 2) and the six gotchas. The gotchas exist for a reason; each has bitten someone in planning.
4. `docs/lld/research/03-feature-assembler.md` — only the §2 "rolling-origin vocabulary" box and §3 "clock-change aggregation approach" are critical. Skim §1 unless D2 is overridden to pandera.
5. `src/bristol_ml/ingestion/CLAUDE.md` — shipped schemas for `neso.load` and `weather.load`. Ground truth over the Stage 2 LLD, which has drift (see Requirements OQ-8, OQ-9).
6. `src/bristol_ml/features/weather.py` — the `national_aggregate(df, weights: Mapping[str, float])` signature the assembler will call. Shipped signature differs from LLD sketch; code to what is in the file, not to the sketch.
7. `docs/architecture/layers/ingestion.md` §3 "Atomic writes" + §5 "Conventions" — patterns Stage 3 inherits.
8. `conf/_schemas.py` — naming, nesting, `ConfigDict(extra="forbid", frozen=True)` convention, the `IngestionGroup | None = None` pattern to mirror for `FeaturesGroup` and `EvaluationGroup`.

CLAUDE.md + `.claude/playbook/` are read once for process, not per-stage.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

1. The assembler is deterministic: identical inputs produce identical output.
2. The output conforms to the declared schema.
3. The splitter produces no train/test overlap and respects chronological order within each fold.
4. The splitter returns index arrays, so downstream code can slice cheaply.
5. The notebook runs top-to-bottom quickly on a laptop. *(Concrete threshold per D7: 120 s.)*
6. Smoke test on the assembler against a small fixture; a test on the splitter for no-overlap and chronological discipline.

Mapping of ACs to tasks is in §6.

---

## 5. Architecture summary (no surprises)

Data flow — faithful to Exploration §6:

```
load_config() → AppConfig
├── .ingestion.neso     → neso.fetch() → neso_demand.parquet
│                         neso.load()   → pd.DataFrame (half-hourly UTC)
│                                         │
│                                         ▼
│                     _resample_demand_hourly(df, agg=D1)
│                                         │
├── .ingestion.weather  → weather.fetch() → weather.parquet
│                         weather.load()   → long-form pd.DataFrame (hourly UTC)
│                                            │
│                                            ▼
│                     features.weather.national_aggregate(df, weights) → wide hourly
│
├── .features.weather_only
│                     assembler.build(demand_hourly, weather_national, cfg)
│                                            │
│                                            ▼
│                     feature_table.parquet  (enforced via OUTPUT_SCHEMA)
│
└── .evaluation.rolling_origin
                      splitter.rolling_origin_split(df, cfg)
                                            │
                                            ▼
                              list[(train_idx, test_idx)] — numpy int arrays
```

Public API surface (Stage 3 adds only these):

```python
# features/assembler.py
OUTPUT_SCHEMA: pa.Schema

def build(demand_hourly: pd.DataFrame, weather_national: pd.DataFrame,
          config: FeatureSetConfig) -> pd.DataFrame: ...
def load(path: Path) -> pd.DataFrame: ...              # validates OUTPUT_SCHEMA
def assemble(cfg: AppConfig, cache: CachePolicy = OFFLINE) -> Path: ...
def _resample_demand_hourly(df: pd.DataFrame, agg: Literal["mean", "max"]) -> pd.DataFrame: ...

# evaluation/splitter.py
def rolling_origin_split(n_rows: int, *, min_train: int, test_len: int,
                         step: int, gap: int = 0,
                         fixed_window: bool = False
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
```

No change to `src/bristol_ml/cli.py`, `__main__.py`, or the `load_config` signature.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Config schemas and Hydra groups
*(Unblocks T2–T5; no downstream data dependency.)*

- [ ] Add to `conf/_schemas.py`:
  - `SplitterConfig` (`min_train_periods`, `test_len`, `step`, `gap`, `fixed_window: bool = False`).
  - `FeatureSetConfig` (`name`, `demand_aggregation: Literal["mean", "max"] = "mean"` — per **D1**, `cache_dir: Path`, `cache_filename: str`, `forward_fill_hours: int = 3`).
  - `FeaturesGroup` with `weather_only: FeatureSetConfig | None = None`.
  - `EvaluationGroup` with `rolling_origin: SplitterConfig | None = None`.
  - `AppConfig.features: FeaturesGroup = Field(default_factory=FeaturesGroup)`.
  - `AppConfig.evaluation: EvaluationGroup = Field(default_factory=EvaluationGroup)`.
- [ ] Create `conf/features/weather_only.yaml` with `# @package features.weather_only` header and defaults per D1, D5, D6.
- [ ] Create `conf/evaluation/rolling_origin.yaml` with `# @package evaluation.rolling_origin` header, `min_train_periods: 8760` (one year), `test_len: 24`, `step: 24`, `gap: 0`, `fixed_window: false`.
- [ ] Add the two entries to `conf/config.yaml` `defaults:` list.
- **Acceptance:** contributes to AC-2, AC-6 (test below).
- **Tests (written by `@test-author`):**
  - `test_features_group_validates` — `AppConfig` instantiates from `conf/config.yaml` with features + evaluation groups populated.
  - `test_feature_set_config_rejects_unknown_aggregation` — `Literal` validator catches garbage.
  - `test_splitter_config_rejects_non_positive_test_len`.
- **Command:** `uv run pytest tests/unit/test_config.py -q`.

### Task T2 — Rolling-origin splitter (evaluation module)
*(No dependency on T1 code paths beyond the `SplitterConfig` Pydantic model; can run in parallel with T3 if two implementers, else serial.)*

- [ ] Create `src/bristol_ml/evaluation/__init__.py` (lazy re-exports mirroring `features/__init__.py`).
- [ ] Create `src/bristol_ml/evaluation/splitter.py` with `rolling_origin_split` matching the signature in §5.
- [ ] Add `_cli_main(argv=None) -> int` that takes `--overrides`, loads config, generates splits over a synthetic range, prints `fold_count, first_fold=(train[0..3], test[0..3])`, and exits 0.
- [ ] Write `src/bristol_ml/evaluation/CLAUDE.md` documenting public surface + known invariants.
- **Acceptance:** AC-3, AC-4, AC-6.
- **Tests (spec-derived, from `@test-author`):**
  - `test_splitter_no_overlap` — every fold: `max(train) < min(test)`; `set(train) ∩ set(test) == ∅`.
  - `test_splitter_chronological_within_fold` — `train` and `test` are monotonically ascending.
  - `test_splitter_returns_integer_arrays` — `isinstance(train, np.ndarray); train.dtype.kind == "i"`.
  - `test_splitter_fold_count_matches_step` — given `n=1000, min_train=100, test_len=10, step=1`, expect 891 folds.
  - `test_splitter_gap_respects_gap_hours` — given `gap=5`, `max(train) + 5 < min(test)`.
  - `test_splitter_fixed_window_keeps_train_size` — every fold has `len(train) == min_train` under `fixed_window=True`.
- **Command:** `uv run pytest tests/unit/evaluation/ -q`.

### Task T3 — Feature assembler (features module)
*(Depends on T1 for `FeatureSetConfig`; independent of T2.)*

- [ ] Create `src/bristol_ml/features/assembler.py`:
  - `OUTPUT_SCHEMA: pa.Schema` — columns per D8 and the national-aggregate variable list.
  - `_resample_demand_hourly(df, agg: Literal["mean", "max"])` — group by `timestamp_utc.dt.floor("h")`, apply `agg`. Test on the fixture covering spring-forward (expect NaN demand rows) and autumn-fallback (expect two rows collapsing normally since everything is UTC). Add a second parametrised test that asserts `agg="max"` returns the half-hour-peak value per hour, not the mean (per **D1**).
  - `build(demand_hourly, weather_national, config) -> pd.DataFrame` — inner join on `timestamp_utc`, forward-fill weather up to `config.forward_fill_hours`, drop remaining NaN rows, assert `OUTPUT_SCHEMA`. Log an `INFO` summary via `logging.getLogger(__name__)` covering: `demand_nan_rows_dropped`, `weather_forward_filled_rows` (per variable), `weather_nan_rows_dropped_after_fill`, and final `row_count` — one log line with a structured message so it is greppable in retros (per **D5**).
  - `load(path) -> pd.DataFrame` — validate `OUTPUT_SCHEMA` field-by-field (mirrors `neso.load` pattern).
- [ ] Create fixture `tests/fixtures/features/toy_feature_inputs/` containing a small demand parquet (half-hourly) and reusing `toy_stations.csv` for weather.
- **Acceptance:** AC-1, AC-2, AC-6.
- **Tests (spec-derived):**
  - `test_assembler_smoke` — fixture in → parquet out; schema passes; row count matches.
  - `test_assembler_deterministic` — two invocations produce byte-identical output **modulo provenance columns** (D8).
  - `test_assembler_clock_change_autumn` — fixture of 50-period autumn day produces 25 UTC-hour rows (per research §3).
  - `test_assembler_clock_change_spring` — fixture of 46-period spring day produces 23 UTC-hour rows with explicit NaN behaviour per D5.
  - `test_assembler_rejects_local_time_join` — constructed negative test: attempt join on `timestamp_local` raises (defensive against Gotcha 2).
  - `test_assembler_output_schema_forbids_extra_columns`.
  - `test_assembler_logs_missing_data_summary` — using `caplog` at `INFO`, asserts a single structured line is emitted on every `build()` call containing `demand_nan_rows_dropped=N`, `weather_forward_filled_rows=…`, `weather_nan_rows_dropped_after_fill=M`, `row_count=R` (per **D5**).
- **Command:** `uv run pytest tests/unit/features/test_assembler.py -q`.

### Task T4 — Assembler CLI + atomic persistence
*(Depends on T3.)*

- [ ] `_cli_main` that ties together `neso.fetch/load → _resample_demand_hourly → weather.fetch/load → national_aggregate → build → _atomic_write` and prints the output path + schema.
- [ ] Reuse `bristol_ml.ingestion._common._atomic_write` (no new helper — Exploration §2 confirms it exists).
- [ ] Update `src/bristol_ml/features/CLAUDE.md` with the assembler surface.
- **Acceptance:** AC-1 (via `python -m …` end-to-end), NFR-6, NFR-8.
- **Tests (implementation-derived, from implementer/me):**
  - `test_assembler_cli_writes_parquet_atomically` — uses `CachePolicy.OFFLINE` with `tmp_path` configs, asserts the output parquet exists and has no leftover `.tmp` sibling.
- **Command:** `uv run pytest tests/unit/features/test_assembler.py -q && uv run python -m bristol_ml.features.assembler --help`.

### Task T5 — Notebook and demo legibility
*(Depends on T3, T4.)*

- [ ] Create `notebooks/03_feature_assembler.ipynb`:
  - Import from `bristol_ml.features.assembler` + `bristol_ml.evaluation.splitter`.
  - Call `assembler.load(path)` assuming warm cache.
  - Print schema via `df.dtypes` and head.
  - Call splitter with `rolling_origin` config; overlay fold boundaries on a demand time series with `.tz_convert('Europe/London')` applied for display only.
  - Cell-level comments walk the concept for a meetup audience (notebook thinness still applies: no assembly logic inline).
- [ ] Add a `nbstripout` pass if configured in pre-commit (it is; see `.pre-commit-config.yaml`).
- **Acceptance:** AC-5, NFR-7.
- **Smoke check:** `uv run jupyter nbconvert --to notebook --execute notebooks/03_feature_assembler.ipynb --output /tmp/03_test_run.ipynb` finishes under 120 s with warm caches.

### Task T6 — Stage hygiene
*(Depends on T1–T5.)*

- [ ] `CHANGELOG.md` under `[Unreleased]`: `### Added` bullets for assembler module, evaluation module, config groups, notebook, fixtures, tests.
- [ ] `docs/lld/stages/03-feature-assembler.md` — retrospective following `docs/lld/stages/00-foundation.md` template.
- [ ] `docs/stages/README.md` — flip Stage 3 status cell to `shipped`, link brief = plan, link retro.
- [ ] `docs/architecture/layers/features.md` — new layer doc (Contract + Internals) following `layers/ingestion.md` shape. **Warn-tier** — I'll draft; human confirms before merge.
- [ ] `docs/architecture/layers/evaluation.md` — same pattern; minimal until Stage 4/6 extend it.
- [ ] `docs/architecture/README.md` layers table gains `features` + `evaluation` rows. **Warn-tier** — same caveat.
- [ ] Move this plan from `docs/plans/active/` to `docs/plans/completed/` **as part of the final commit only**.
- [ ] If and only if D2 is overridden to pandera: add `pandera[pandas]` to `pyproject.toml`, run `uv lock`, and add `docs/architecture/decisions/0003-pandera-for-feature-schemas.md` MADR ADR.
- [ ] **Not** touching `docs/intent/DESIGN.md` §6 unless structural change requires it. If it does, surface to human — deny-tier for me.

---

## 7. Files expected to change

### New
- `src/bristol_ml/features/assembler.py`
- `src/bristol_ml/evaluation/__init__.py`
- `src/bristol_ml/evaluation/splitter.py`
- `src/bristol_ml/evaluation/CLAUDE.md`
- `conf/features/weather_only.yaml`
- `conf/evaluation/rolling_origin.yaml`
- `tests/unit/evaluation/__init__.py`
- `tests/unit/evaluation/test_splitter.py`
- `tests/unit/features/test_assembler.py`
- `tests/fixtures/features/toy_feature_inputs/demand.parquet` (hand-crafted; small)
- `notebooks/03_feature_assembler.ipynb`
- `docs/lld/stages/03-feature-assembler.md`
- `docs/architecture/layers/features.md`
- `docs/architecture/layers/evaluation.md`

### Modified
- `conf/_schemas.py` — four new Pydantic models + two `AppConfig` fields.
- `conf/config.yaml` — two new `defaults:` entries.
- `src/bristol_ml/features/__init__.py` — lazy re-export for `assembler` if needed.
- `src/bristol_ml/features/CLAUDE.md` — assembler surface documented.
- `CHANGELOG.md` — `[Unreleased]` bullets.
- `docs/stages/README.md` — Stage 3 status cell to `shipped`, links.
- `docs/architecture/README.md` — layers table gains `features` + `evaluation`.

### Intentionally not modified
- `docs/intent/**` — deny-tier.
- `src/bristol_ml/ingestion/**` — no Stage 3 cause to touch; `columns` dead-weight cleanup deferred.
- `pyproject.toml` / `uv.lock` — untouched unless D2 flips to pandera.

---

## 8. Exit criteria (definition of done per DESIGN §9)

- All tests pass: `uv run pytest -q` green; no `xfail`, no skipped.
- Lint/format clean: `uv run ruff check .`, `uv run ruff format --check .`.
- Pre-commit clean: `uv run pre-commit run --all-files`.
- `python -m bristol_ml.features.assembler --help` and `python -m bristol_ml.evaluation.splitter --help` both exit 0.
- Notebook runs top-to-bottom under 120 s with warm caches (D7).
- Every new public symbol has a British-English docstring.
- CHANGELOG bullet present.
- Retrospective at `docs/lld/stages/03-feature-assembler.md`.
- Stages index status cell updated.
- This plan moved from `docs/plans/active/` to `docs/plans/completed/` in the final commit.

---

## 9. Team-shape recommendation

**Sequential single-session** work by the lead (me), following the orchestrator playbook — Phase 2 task-by-task, spawning `@test-author` after each code task to write spec-derived tests before I declare the task complete.

Rationale:
- Tasks T1–T4 have clean sequential data dependencies; parallelising buys nothing.
- The only plausible parallelism is T2 ↔ T3 (splitter is independent of assembler), but the cost of spinning up a worktree + two contexts exceeds the 2–3 hours of single-threaded work saved.
- No new model family or unusual data semantics — `@domain-researcher` has already landed the three findings the stage needs. No further research is contemplated.
- No contested specification — the eight decisions in §1 are the only ambiguities, and they resolve in chat before Phase 2 starts.

Escalate to `@reframer` only if a task fails three times with the same framing (per CLAUDE.md §Escalation ladder).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| D1 default overridden after T3 starts | Low | High (invalidates assembler tests) | Resolve D1 at plan approval. |
| D2 override to pandera surfaces late | Low | Medium (triggers ADR + pyproject churn) | Resolve D2 at plan approval. |
| Spring-forward NaN row breaks assembler join | Medium | Low | D5 policy (drop NaN demand rows) covers this; explicit test at T3. |
| Implementer codes to `national_aggregate` LLD signature, not shipped | Medium | Medium | Reading-order item 6 pins the shipped signature; Requirements OQ-9 flagged. |
| Notebook exceeds 120 s on a slow laptop | Low | Low | `.tz_convert` + plot is cheap; assembler load is warm-cache read; budget is forgiving. |
| Fixture parquet drifts from `neso.OUTPUT_SCHEMA` over time | Medium | Low | Generate fixture from `neso.OUTPUT_SCHEMA` programmatically in a `conftest.py` fixture, not hand-rolled. |

---

## Human sign-off (2026-04-19)

- D1 (aggregation): **OVERRIDE** — `mean` default, pluggable `Literal["mean", "max"]`; no `sum`/`peak` in scope.
- D2 (schema enforcement): **ACCEPT** — pyarrow `OUTPUT_SCHEMA`.
- D3 (feature-set name): **ACCEPT** — `weather_only`.
- D4 (window type): **ACCEPT** — expanding default, `fixed_window` knob.
- D5 (missing-data policy): **OVERRIDE** — drop demand NaN + forward-fill ≤3h weather, **and log a structured INFO summary on every `build()` call**.
- D6 (output location): **ACCEPT** — `${BRISTOL_ML_CACHE_DIR:-data/features}/<name>.parquet`.
- D7 (notebook budget): **ACCEPT** — 120 s.
- D8 (provenance columns): **ACCEPT** — `neso_retrieved_at_utc`, `weather_retrieved_at_utc`.
