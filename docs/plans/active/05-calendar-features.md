# Plan — Stage 5: Calendar features (without/with comparison)

**Status:** `unreviewed` — awaiting human approval of D1–D11 and H-1–H-4 before Phase 2 begins.
**Intent:** [`docs/intent/05-calendar-features.md`](../../intent/05-calendar-features.md)
**Upstream stages shipped:** Stage 0 (foundation), Stage 1 (NESO demand), Stage 2 (weather + national aggregate), Stage 3 (feature assembler + splitter), Stage 4 (linear baseline + evaluation harness).
**Downstream consumers:** every subsequent modelling stage (6, 7, 8, 9, 10, 11) inherits the enriched `weather_calendar` feature set by default. Stage 6 (enhanced evaluation) composes the without/with residual-pattern story into a richer visualisation surface.
**Baseline SHA:** `35cd1ec` (tip of `reclaude` at plan time).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/05-calendar-features-requirements.md`](../../lld/research/05-calendar-features-requirements.md)
- Codebase map — [`docs/lld/research/05-calendar-features-codebase-map.md`](../../lld/research/05-calendar-features-codebase-map.md)
- External research — [`docs/lld/research/05-calendar-features-external-research.md`](../../lld/research/05-calendar-features-external-research.md)

**Pedagogical weight.** The intent calls this "the most pedagogically important stage in the project". The without/with comparison is the moment a facilitator can put two metric tables side by side at a meetup and say "that's what domain knowledge bought us." Decisions D-3, D-4, D-5 govern how readable the enriched model's coefficients are; decision D-11 governs whether the demo has a dedicated narrative notebook. Over-engineering the feature shapes would blur the comparison; under-engineering them would understate what feature engineering is worth.

---

## 1. Decisions for the human (resolve before Phase 2)

Eleven decision points plus four housekeeping carry-overs. For each I propose a default that honours the simplicity bias + the research evidence, and cite the supporting source. Mark each `ACCEPT` / `OVERRIDE: <alt>` in your reply; I'll update the plan before Phase 2.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Bank-holiday source | **`https://www.gov.uk/bank-holidays.json`** via `requests` + VCR cassette, mirroring the existing NESO / Open-Meteo / NESO-forecast ingesters. No new package dependency. | OGL v3.0, no auth, stable shape since 2012, same `_common.py` helpers apply verbatim. Adding `python-holidays` or `govuk-bank-holidays` introduces a dependency for zero extra capability against our training window. | External R1, R9; codebase map §E. |
| **D2** | GB division composition | **Union of `england-and-wales` and `scotland`** only. NI excluded from the composite `is_bank_holiday` column because NESO ND is the GB grid (NI runs on SEM via SONI). The gov.uk JSON's `northern-ireland` key is still persisted to the cache for completeness (so the ingester stays policy-agnostic), but the feature derivation ignores it. | Intent: "any division is on holiday" is a reasonable composite for a national model. R2 tightens this by excluding NI on geographic / grid grounds. Separate per-division indicators are the natural extension for future regional work but add 3× column count for no current-model gain. | Intent §Points for consideration; R2 (NESO grid geography, Hong 2016, Ziel 2018). |
| **D3** | Encoding of `hour_of_day` | **One-hot** (23 dummies after dropping hour 0 as the reference category). A readable "load profile" falls out of the OLS coefficients — a direct teaching artefact for meetups. | R4's sklearn empirical benchmark shows one-hot outperforms sin/cos for linear models on hourly demand-like data (10.0% MAE vs 12.5%). Hong 2016 GEFCom vanilla benchmark + Ziel & Weron 2018 both use one-hot hour-of-day. Sin/cos was the initial proposal in the requirements doc but the evidence runs the other way. | External R4 (sklearn cyclical-feature-engineering example; Hong et al. 2016; Ziel & Weron 2018). |
| **D4** | Encoding of `day_of_week` and `month` | **One-hot for both.** Day-of-week: 6 dummies (drop Sunday = reference). Month: 11 dummies (drop January). | Same empirical argument as D3. Weekday effects in GB demand are large and non-linear; month effects are seasonally coarse but one-hot gives readable coefficients at meetups. 17 categorical dummies + 23 hour dummies + 5 weather columns + 3 holiday columns = ~48 coefficients — still a trivial OLS fit, still interpretable. | External R4 (Hong 2016 vanilla benchmark; Ziel & Weron 2018 LEAR model D1..D7 dummies). |
| **D5** | Holiday-proximity feature shape | **Two binary columns** — `is_day_before_holiday` and `is_day_after_holiday` — matching Ziel 2018's factor-variable shape. No signed integer `days_to_nearest_holiday` column. | R3 confirms binary is the industry convention; Azure AutoML's signed-integer alternative forces a monotone linear relationship the data does not support. Binary handles Christmas–New Year clusters correctly (both bits may simultaneously be 1). | External R3 (Ziel 2018; Amperon 2024; Azure AutoML docs). |
| **D6** | Historical-depth fallback | **WARNING-level log + fill `is_bank_holiday=0` + continue** for any row pre-2012-01-02. No pre-cache data is raised as an error. Document in the module docstring that `python-holidays` is the sanctioned extension path for any future stage training pre-2012. | Stage 4 training window starts 2018-01-01 (DESIGN §5.1); the gov.uk archive starts 2012-01-02. Zero rows actually fall into the gap today, so the path is defensive rather than functional. Raising would make the pipeline fragile to an unrelated training-window change. | External R10 (gov.uk historical depth; python-holidays algorithmic-generation path). |
| **D7** | DST / UTC-hour → local-date mapping rule | **"Any UTC hour whose `Europe/London` local-date component matches a bank-holiday date gets `is_bank_holiday=1`,"** regardless of spring-forward / autumn-back. Documented verbatim in the `derive_calendar` module docstring. Unit-tested on a synthetic frame spanning one spring-forward Sunday. | No published convention found (R6); any defensible rule needs to be documented. `Europe/London` local date is what a consumer experiences as "on the holiday"; it also lines up with how `day_of_week` and `hour_of_day` are derived. DST-change Sundays are never statutory bank holidays, so the rule fires only on the edge case of a moving holiday (Easter Monday / Good Friday never fall on clock-change days; Mothering Sunday is not statutory; Easter Sunday is not statutory). | External R6 (pandas `tz_localize` docs; BST Wikipedia; absence of published convention). |
| **D8** | Notebook step override | **Inherit Stage 4's precedent verbatim**: in-cell `evaluation.rolling_origin.min_train_periods=720` + `step=168` override for the demo notebook. CLI `python -m bristol_ml.train features=weather_calendar` uses the full daily-stride defaults. | Stage 4 D7 established this exact pattern for pedagogical pacing — narrower train window + weekly stride fits comfortably under the 120 s notebook ceiling. No reason to diverge. | Stage 4 plan §D7; `docs/lld/stages/04-linear-baseline.md` design choices. |
| **D9** | `OUTPUT_SCHEMA` extension mechanism | **Option 3 — composable `derive_calendar(df, holidays_df)` pure function + parallel `CALENDAR_OUTPUT_SCHEMA` + `assemble_calendar` / `load_calendar` orchestrators** in `features/assembler.py`, with `features/calendar.py` owning the pure derivation. `CALENDAR_VARIABLE_COLUMNS` joins `WEATHER_VARIABLE_COLUMNS` as a sibling constant. | Matches the intent's AC-2 "pure function" requirement verbatim; matches the existing `features.weather.national_aggregate` shape; avoids a branching `build()` (Option 2) and an awkward callable-in-config pattern (Option 4); avoids the discriminator sprawl of Option 1 (schema constants proliferate, but that mirrors the Stage 4 `ModelConfig` discriminated-union shape and is explicit). The codebase-explorer strongly recommends this option. | Codebase map §B Option 3; intent §AC-2; `docs/architecture/layers/features.md` §2 pure-derivation pattern. |
| **D10** | `train.py` feature-set selection | **Hydra `features=` group override.** Add a `features=weather_only \| weather_calendar` override key that selects which `cfg.features.<set>` block `train.py` reads. Internal resolver `_resolve_feature_set(cfg) -> (FeatureSetConfig, load_fn, feature_column_names)` picks the populated set and returns the correct loader + column list; raises if both or neither are populated after Hydra resolution. | Intent AC-1 is explicit: "switching is a configuration change, not a code change." A Hydra `features=` override satisfies this exactly, symmetric with how Stage 4 ships `model=linear` / `model=naive`. The resolver keeps `LinearConfig.feature_columns` as the single source of truth for *which* columns feed the model; the feature-set discriminator picks *which schema* gets loaded. | Codebase map §F (current `train.py` hardcodes `cfg.features.weather_only`); intent §AC-1. |
| **D11** | Notebook shape | **New `notebooks/05_calendar_features.ipynb`** dedicated to the without/with demo. `04_linear_baseline.ipynb` remains the linear-baseline surface and is not modified. | One-notebook-per-stage precedent (Stages 1, 2, 3, 4 all have their own notebook); the Stage 4 demo was narratively complete on its own. Merging would confuse the pedagogical progression "Stage 4 shows the linear baseline; Stage 5 shows what feature engineering adds". Duplicate setup (two `load_config` cells, two `assembler.load` cells) is cheap. | Intent §Demo moment; one-notebook-per-stage precedent across Stages 1-4. |

**Blocking note.** D-2, D-3, D-4, D-9, and D-10 are load-bearing — each governs a decision that propagates downstream:
- D-2 (NI exclusion) determines the `holidays_df` filter in `derive_calendar` and is the one decision that would most surprise a facilitator if wrong.
- D-3 and D-4 (one-hot encoding) determine the final feature column count (~40 calendar columns) and the readability of OLS coefficients at the meetup.
- D-9 (composable `derive_calendar` shape) is the assembler's architectural seam; reversing it would mean reorganising `features/` and `assembler.py` after other work has landed on top.
- D-10 (`features=` Hydra override) is the one change to `train.py` and is the only AC-1 implementation path.

Please resolve all five before Phase 2. D-1, D-5, D-6, D-7, D-8, D-11 are lower-risk single-line reversals.

### Housekeeping carry-overs

From Stage 4 Phase 3 review and the codebase map §D:

| # | Item | Resolution |
|---|---|---|
| **H-1** | `NesoBenchmarkConfig.holdout_start/_end` schema fields exist but have no consumer (Stage 4 retro "Deferred"). | **Defer to Stage 6** (enhanced evaluation; the stated natural consumer). No Stage 5 action. |
| **H-2** | `ingestion._neso_dst` extraction — settlement-period → UTC algebra duplicated in `neso.py` and `neso_forecast.py` (Stage 4 retro "Deferred"). | **Defer to Stage 13** (REMIT ingestion; next natural settlement-period consumer). Stage 5's `holidays.py` is date-level, not settlement-period, so the trigger does not fire here. No Stage 5 action. |
| **H-3** | `docs/intent/DESIGN.md §6` layout tree accumulation — Stages 1–4 additions missing (deny-tier for the lead). Stage 5 will add `ingestion/holidays.py`, `features/calendar.py`, `conf/features/weather_calendar.yaml`, `conf/ingestion/holidays.yaml`, `notebooks/05_calendar_features.ipynb`. | **Flag for human-led batched §6 edit** covering Stages 1–5 at Stage 5 PR review. Lead MUST NOT touch §6 unilaterally. |
| **H-4** | `docs/architecture/ROADMAP.md` Features section — Stage 3 retro deferred a ROADMAP pass to "Stage 5/6"; Stage 5 resolves the `weather_calendar` entry in `features.md §Module inventory`. | **Close at Stage 5 T7** (stage hygiene; warn-tier). Update ROADMAP to mark the Features open questions resolved, with back-reference to Stage 5 retro. |

---

## 2. Scope

### In scope

- A new `src/bristol_ml/ingestion/holidays.py` module that retrieves and caches GB bank-holiday dates from `https://www.gov.uk/bank-holidays.json` with the same `fetch(config, cache=...) -> Path` + `load(path) -> pd.DataFrame` contract as every other ingester, a VCR cassette under `tests/fixtures/holidays/cassettes/` so tests run offline, and a `scripts/record_holidays_cassette.py` one-off recorder.
- A new `src/bristol_ml/features/calendar.py` module exposing a pure function `derive_calendar(df, holidays_df) -> pd.DataFrame` that appends the one-hot / binary calendar columns to any hourly UTC-indexed frame. No I/O, no global state, no network.
- Extensions to `src/bristol_ml/features/assembler.py`: a new `CALENDAR_VARIABLE_COLUMNS` constant, a `CALENDAR_OUTPUT_SCHEMA`, and `assemble_calendar(cfg, cache=...)` / `load_calendar(path)` orchestrators that compose `assemble()` + `derive_calendar()`.
- New Pydantic schemas in `conf/_schemas.py`: `HolidaysIngestionConfig` (mirrors `NesoIngestionConfig` / `WeatherIngestionConfig` structurally), with `IngestionGroup` extended to carry `holidays: HolidaysIngestionConfig | None = None` and `FeaturesGroup` extended to carry `weather_calendar: FeatureSetConfig | None = None`.
- New Hydra config groups: `conf/ingestion/holidays.yaml` and `conf/features/weather_calendar.yaml`, both with the `# @package ...` header convention. `conf/config.yaml` `defaults:` list gains the two new entries.
- Extension to `src/bristol_ml/train.py`: a Hydra `features=weather_only | weather_calendar` group override selects which `cfg.features.<set>` block drives the run; a new `_resolve_feature_set(cfg)` function returns the populated `(config, load_fn, feature_columns)` triple and raises on the degenerate case.
- A new `notebooks/05_calendar_features.ipynb` running the Stage 4 `LinearModel` twice — once on `weather_only`, once on `weather_calendar` — printing both metric tables side by side with the NESO benchmark, and rendering the residual-ripple comparison over a shared test week.
- Unit tests: `test_holidays.py` (VCR cassette replay, schema enforcement, provenance scalar); `test_calendar.py` (pure-function determinism, DST rule, one-hot encoding, holiday-proximity binary pair, collinearity-guard assertions); `test_assembler_calendar.py` (schema conformance, composability with the weather-only pipeline).
- Stage-hygiene updates: `CHANGELOG.md` bullets, `docs/lld/stages/05-calendar-features.md` retrospective, `docs/stages/README.md` status cell, `docs/architecture/layers/features.md` module inventory update, `docs/architecture/ROADMAP.md` Features section close (H-4).

### Out of scope (do not accidentally implement)

From intent §Out of scope, explicit:
- School holiday term dates (sourcing across four UK nations is disproportionate to Stage 5's demo weight).
- Sporting events and other one-off demand-affecting events — REMIT chain (Stages 13–16).
- Regional modelling (per-division demand).
- Lag features — Stage 7+ (SARIMAX).
- Any new model class — Stage 5 reuses the Stage 4 `LinearModel` unchanged.
- Changes to the rolling-origin split — Stage 6 (enhanced evaluation).

Also out of scope for this plan:
- Interaction terms (temperature × weekday, temperature × hour) — intent explicitly excludes them "to keep the without/with comparison honest". Natural Stage 6 candidate.
- Sin/cos day-of-year for annual seasonality — deferred as a Stage 6+ follow-up (D-3/D-4 stay with one-hot for all intra-year cyclical features at Stage 5; the annual cycle is weakly resolved by the 11-dummy month encoding and not strongly enough to justify a second encoding mechanism).
- `python-holidays` package dependency — documented as the sanctioned extension path for pre-2012 training windows (D-6), but not added to `pyproject.toml` because no current stage needs it.
- Pre-2012 training windows — D-6 fallback is defensive; the current pipeline never exercises it.
- Changes to the `Model` protocol, the evaluation harness, or the benchmark helper — Stage 5 is a features-layer stage.
- Changes to `src/bristol_ml/cli.py` or `src/bristol_ml/__main__.py` — feature-set selection is a Hydra group override, not a subcommand.
- Changes to `pyproject.toml` — no new runtime or dev dependencies. `requests` + `pyarrow` + `pandas` + `vcrpy` are all already declared. Confirm at Phase 2 kickoff.

---

## 3. Reading order for the implementer

Read top-to-bottom before opening code:

1. `docs/intent/05-calendar-features.md` — the spec. Where this plan disagrees, the spec wins.
2. `docs/lld/research/05-calendar-features-requirements.md` — acceptance criteria, F-numbers, and the structured translation of the intent.
3. `docs/lld/research/05-calendar-features-codebase-map.md` — §A (consumable Stage 3/4 surfaces), §B (assembler extension points — Option 3 is the chosen seam), §C (patterns to follow), §E (gov.uk endpoint shape), §F (integration with `train.py` and notebook), §G (files list).
4. `docs/lld/research/05-calendar-features-external-research.md` — R1 (endpoint), R2 (NI exclusion), R3 (proximity), R4 (one-hot wins for OLS), R5 (`is_weekend` collinearity trap), R6 (DST rule).
5. `docs/intent/DESIGN.md` §2.1 (principles), §3.2 (layer responsibilities — features and ingestion paragraphs), §4 (data flow), §7 (config), §9 Stage 5 row.
6. `docs/architecture/layers/features.md` — contract + extension plan for `weather_calendar`.
7. `docs/plans/completed/04-linear-baseline.md` — for the discriminator pattern on `AppConfig.model` (symmetric with D-10's `features=` override), the CLI idiom, the test-authoring rhythm, and the stage-hygiene checklist.
8. `src/bristol_ml/features/assembler.py` — `build()`, `load()`, `assemble()`, `OUTPUT_SCHEMA`, and `WEATHER_VARIABLE_COLUMNS`. The plan's D-9 extends these with parallel calendar constants.
9. `src/bristol_ml/ingestion/neso_forecast.py` — the most recent copy-and-adapt template for a new ingester (Stage 4 T7).
10. `src/bristol_ml/ingestion/_common.py` — helpers to reuse verbatim; the protocol-typed `Config` shape that `HolidaysIngestionConfig` must satisfy.
11. `src/bristol_ml/evaluation/harness.py` — the `feature_columns=None` default that Stage 5 must NOT rely on for the calendar run. Callers supply explicit column lists for the calendar-enriched set.
12. `src/bristol_ml/train.py` lines 106–132 — the current hard-coded `cfg.features.weather_only` read that D-10 replaces.
13. `conf/_schemas.py` — the `ConfigDict(extra="forbid", frozen=True)` pattern and the `*Group | None = None` idiom.
14. `tests/conftest.py` — the loguru-to-caplog adapter already available for INFO-line assertions.

CLAUDE.md + `.claude/playbook/` are read once for process, not per-stage.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

Mapped from `docs/lld/research/05-calendar-features-requirements.md`:

1. **Switching between `weather_only` and `weather_calendar` is a configuration change, not a code change.** *(AC-1; satisfied by F-12 + D-10; task T5.)*
2. **The calendar feature derivation is a pure function: same inputs, same outputs, no side effects.** *(AC-2; satisfied by F-7; task T3.)*
3. **The enriched feature table conforms to its schema once extended.** *(AC-3; satisfied by F-13, F-14; task T4.)*
4. **The notebook produces the comparison table end-to-end.** *(AC-4; satisfied by F-16; task T6.)*
5. **The weekly-residual pattern is present in the weather-only residuals and absent in the enriched residuals in the notebook visualisation.** *(AC-5; satisfied by F-17; task T6.)*
6. **Bank-holiday ingestion is idempotent and offline-first against its cache.** *(AC-6; satisfied by F-4; task T2.)*

Implicit DoD ACs (DESIGN §9): CI green (AC-7), module CLAUDE.md updated (AC-8), README entry-point listed (AC-9), retrospective filed (AC-10), CHANGELOG entry (AC-11), notebook demonstrates output under 120 s (AC-12). §6 repo-layout tree update is deny-tier for the lead; H-3 captures the flag-for-human posture.

---

## 5. Architecture summary (no surprises)

Data flow — end-to-end for `python -m bristol_ml.train features=weather_calendar`:

```
load_config(overrides=["features=weather_calendar"]) → AppConfig
├── .features.weather_calendar      (populated; .weather_only is None)
├── .evaluation.rolling_origin      (passed to harness)
├── .evaluation.metrics             → list[MetricFn]
├── .model.linear                   (default linear model)
└── .ingestion.neso_forecast        (optional; benchmark only)

_resolve_feature_set(cfg) → (FeatureSetConfig, load_fn=assembler.load_calendar, feature_column_names=(*weather, *calendar))

assembler.load_calendar(path) → pd.DataFrame (CALENDAR_OUTPUT_SCHEMA; 10 + ~40 columns)

harness.evaluate(model, df, splitter_cfg, metrics, feature_columns=<resolved>) →
    pd.DataFrame (per-fold metrics)

benchmarks.compare_on_holdout({"linear": LinearModel(cfg)}, df, neso_forecast, splitter_cfg, metrics) →
    three-way table

train._cli_main() → print metric table + benchmark table
```

Public API surface (Stage 5 adds only these):

```python
# ingestion/holidays.py
OUTPUT_SCHEMA: pa.Schema   # (date, division, title, notes, bunting, retrieved_at_utc)
def fetch(config: HolidaysIngestionConfig, *, cache: CachePolicy = CachePolicy.AUTO) -> Path: ...
def load(path: Path) -> pd.DataFrame: ...

# features/calendar.py
CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]   # ~40 (name, dtype) pairs
def derive_calendar(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Append one-hot hour-of-day, one-hot day-of-week, one-hot month,
    is_bank_holiday, is_day_before_holiday, is_day_after_holiday columns
    to a UTC-hourly frame. Pure — no I/O. Deterministic."""

# features/assembler.py (additions)
CALENDAR_OUTPUT_SCHEMA: pa.Schema   # 10 weather-only columns + ~40 calendar columns, in order
def assemble_calendar(cfg: AppConfig, cache: CachePolicy = CachePolicy.OFFLINE) -> Path:
    """Orchestrator: calls assemble() then derive_calendar() then writes parquet."""
def load_calendar(path: Path) -> pd.DataFrame:
    """Schema-validating reader for the weather_calendar set."""

# train.py (internal)
def _resolve_feature_set(cfg: AppConfig) -> tuple[FeatureSetConfig, LoadFn, tuple[str, ...]]:
    """Pick the populated feature-set config and return the right loader + column list."""
```

No change to `src/bristol_ml/cli.py`, `__main__.py`, `load_config()` signature, `evaluation/`, `models/`, or the existing `assembler.build()` / `assembler.assemble()` / `assembler.load()` functions (additions only).

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Config schemas and Hydra groups
*(Unblocks T2–T7; no downstream data dependency.)*

- [ ] Add to `conf/_schemas.py`:
  - `HolidaysIngestionConfig` — mirrors `NesoIngestionConfig` structurally: `url: HttpUrl`, `cache_dir: Path`, `cache_filename: str`, `rate_limit`, `retry`, `request_timeout_seconds`, `min_inter_request_seconds`. Add a `divisions: tuple[Literal["england-and-wales", "scotland", "northern-ireland"], ...] = ("england-and-wales", "scotland", "northern-ireland")` field so the cache-on-disk is division-complete even though the feature derivation only unions two of them (per **D2**).
  - Extend `IngestionGroup` with `holidays: HolidaysIngestionConfig | None = None`.
  - Extend `FeaturesGroup` with `weather_calendar: FeatureSetConfig | None = None`. `FeatureSetConfig` is reused verbatim from Stage 3 — the two feature-set configs share the same shape; a new subclass is not needed.
- [ ] Create `conf/ingestion/holidays.yaml` with `# @package ingestion.holidays`:
  - `url: https://www.gov.uk/bank-holidays.json`
  - `cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/holidays}`
  - `cache_filename: holidays.parquet`
  - `min_inter_request_seconds: 0.0` (gov.uk has no documented rate limit; D-1).
  - `rate_limit`, `retry`, `request_timeout_seconds` — copied from `conf/ingestion/neso.yaml`.
  - `divisions` list as above.
- [ ] Create `conf/features/weather_calendar.yaml` with `# @package features.weather_calendar`:
  - `name: weather_calendar`
  - `cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,data/features}`
  - `cache_filename: weather_calendar.parquet`
  - `demand_aggregation: mean` (same as weather_only default).
  - `forward_fill_weather_hours: 3` (same as weather_only default).
- [ ] Add to `conf/config.yaml` `defaults:` list:
  - `- ingestion/holidays@ingestion.holidays`
  - (Do NOT add `- features/weather_calendar@features.weather_calendar` — the whole point of D-10 is that exactly one of `features.weather_only` / `features.weather_calendar` is populated at a time. Treat `features` as a Hydra group, keeping `- features/weather_only@features.weather_only` as the default group entry and relying on a CLI override `features=weather_calendar` to swap.)

  Clarification on Hydra mechanics: if the existing `features/weather_only` entry is written with the `@features.weather_only` package specifier, the equivalent `weather_calendar` yaml must use `@features.weather_calendar`. A CLI `features=weather_calendar` override replaces the default entry and leaves the other unset. `_resolve_feature_set(cfg)` enforces mutual-exclusivity at runtime.
- **Acceptance:** contributes to AC-1, AC-7.
- **Tests (spec-derived, written by `@test-author`):**
  - `test_app_config_default_selects_weather_only` — `load_config()` with no overrides yields `cfg.features.weather_only` populated, `cfg.features.weather_calendar is None`.
  - `test_features_override_swaps_to_weather_calendar` — `load_config(overrides=["features=weather_calendar"])` yields `cfg.features.weather_calendar` populated, `cfg.features.weather_only is None`.
  - `test_holidays_ingestion_config_rejects_extra_keys` — `extra="forbid"` verification.
  - `test_holidays_ingestion_config_divisions_literal` — Literal narrowing on the divisions list.
- **Command:** `uv run pytest tests/unit/test_config.py -q`.

### Task T2 — Bank-holidays ingester
*(Depends on T1 for `HolidaysIngestionConfig`; independent of T3/T4.)*

- [ ] Create `src/bristol_ml/ingestion/holidays.py`:
  - Copy-and-adapt of `neso_forecast.py` / `weather.py`. Target URL is a single GET — no CKAN pagination, no settlement-period algebra.
  - `OUTPUT_SCHEMA: pa.Schema` with six columns: `date` (`date32`), `division` (`string`), `title` (`string`), `notes` (`string`), `bunting` (`bool`), `retrieved_at_utc` (`timestamp("us", tz="UTC")`). Primary key `(date, division)`; sorted ascending.
  - `fetch(config, *, cache=CachePolicy.AUTO) -> Path` — standard `CachePolicy` semantics; calls `_common._retrying_get`; writes all three divisions' events (not only the two needed for the GB composite) so the cache remains policy-agnostic.
  - `load(path) -> pd.DataFrame` — schema-validating reader.
  - `_cli_main(argv=None) -> int` — `python -m bristol_ml.ingestion.holidays --cache auto` prints the cache path and schema summary.
  - Module docstring names the OGL v3 licence, the gov.uk source, and the 2012-01-02 historical lower bound (per **D6**).
- [ ] Add VCR cassette at `tests/fixtures/holidays/cassettes/holidays_refresh.yaml` recording one real GET of `https://www.gov.uk/bank-holidays.json`. Scrub no headers (public endpoint, no auth).
- [ ] Add `scripts/record_holidays_cassette.py` mirroring `scripts/record_neso_cassette.py`.
- [ ] Extend `src/bristol_ml/ingestion/CLAUDE.md` with the new `holidays.py` public surface, the 2012+ coverage note, and the cassette location.
- **Acceptance:** AC-6, AC-7, AC-9 (CLI entry point).
- **Tests (spec-derived + cassette-backed):**
  - `test_holidays_fetch_writes_parquet` — VCR cassette playback; asserts parquet exists, schema passes.
  - `test_holidays_load_schema_enforced` — reading a file with a missing column raises.
  - `test_holidays_provenance_column_populated` — `retrieved_at_utc` is a single scalar across all rows of one fetch.
  - `test_holidays_fetch_idempotent_offline` — second `cache=OFFLINE` call on a warm cache makes no request and returns the same path (AC-6).
  - `test_holidays_fetch_offline_raises_on_missing_cache` — `CacheMissingError` path.
  - `test_holidays_all_three_divisions_present` — parquet contains all three `division` values per **D1** (cache is policy-agnostic even though the feature derivation filters to two).
- **Command:** `uv run pytest tests/unit/ingestion/test_holidays.py -q && uv run python -m bristol_ml.ingestion.holidays --help`.

### Task T3 — Calendar feature derivation
*(Depends on T1 for config types; independent of T2 because `derive_calendar` takes a `holidays_df` argument rather than reading it from disk.)*

- [ ] Create `src/bristol_ml/features/calendar.py`:
  - Pure function `derive_calendar(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame`. Input: hourly frame with tz-aware UTC `timestamp_utc` column or index (document which). Output: same frame with calendar columns appended. No I/O, no global state, no logging of side-effecting content — one `logger.info` line per call summarising the row count, bank-holiday date count, and dropped-row count (if any), matching the Stage 3 D5 convention.
  - `CALENDAR_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...]` — the explicit ordered list, derivable by code but pinned as a constant so downstream (harness / `LinearConfig.feature_columns`) has one source of truth. Dtypes per D-3 / D-4 / D-5:
    - `hour_of_day_{01..23}` — 23 `int8` one-hot columns (hour 0 is reference; drop it).
    - `day_of_week_{1..6}` — 6 `int8` one-hot columns (Sunday = 0 is reference; drop it).
    - `month_{2..12}` — 11 `int8` one-hot columns (January = 1 is reference; drop it).
    - `is_bank_holiday` — 1 `int8` column; union of england-and-wales + scotland per **D2**.
    - `is_day_before_holiday` — 1 `int8` column per **D5**.
    - `is_day_after_holiday` — 1 `int8` column per **D5**.
    - Total: 42 calendar columns (23 + 6 + 11 + 3).
  - DST rule per **D7**: convert the UTC index to `Europe/London`, take the local-date component, compare to the holiday-date set. Document the rule verbatim in the module docstring; cite the plan decision.
  - Collinearity guard: assert (in code, not just docstring) that `is_weekend` is NOT in the output column set — the one-hot `day_of_week` already encodes weekend information and adding `is_weekend` produces perfect multicollinearity (per R5). The guard is a one-line `assert "is_weekend" not in derived.columns` that catches future mistaken extensions.
  - Historical-depth fallback per **D6**: if any row in the input has a local-date before the earliest holiday-date, log `WARNING` and fill `is_bank_holiday` / `is_day_before_holiday` / `is_day_after_holiday` with `0` for those rows. Do not raise.
  - `_cli_main(argv=None) -> int` — `python -m bristol_ml.features.calendar --help` prints the expected output schema. If `cfg.features.weather_calendar` + `cfg.ingestion.holidays` caches are warm, also loads them and prints the first few rows of `derive_calendar`'s output.
- **Acceptance:** AC-2, AC-3, AC-5 (residual-ripple suppression depends on the calendar columns being correct), AC-7.
- **Tests (spec-derived; pure function — no fixtures needed beyond synthetic frames):**
  - `test_derive_calendar_deterministic` — same input twice → byte-identical output (AC-2).
  - `test_derive_calendar_one_hot_hour_sum_leq_one` — for every row, the 23 hour dummies sum to 0 (hour 0) or 1 (any other hour).
  - `test_derive_calendar_one_hot_weekday_sum_leq_one` — same property for day-of-week.
  - `test_derive_calendar_one_hot_month_sum_leq_one` — same property for month.
  - `test_derive_calendar_is_bank_holiday_matches_holidays_df` — synthetic frame with a known Boxing Day row → `is_bank_holiday=1`.
  - `test_derive_calendar_proximity_christmas_cluster` — 27 December (Boxing Day substitute + day-before-New-Year is not a holiday, but day-after Boxing Day is) → `is_day_after_holiday=1`, `is_day_before_holiday=0` if New Year's Day is a Thursday; variant covers both cases.
  - `test_derive_calendar_dst_spring_forward` — synthetic frame spanning the last Sunday of March; bank holiday on that Sunday (hypothetical; or use the Monday after) → `is_bank_holiday=1` for all UTC hours whose `Europe/London` date matches (per **D7**).
  - `test_derive_calendar_pre_2012_warns_and_fills_zero` — synthetic 2011 frame → all three holiday columns == 0, single `loguru` WARNING logged (per **D6**). Uses the `loguru_caplog` fixture in `tests/conftest.py`.
  - `test_derive_calendar_ni_excluded_from_composite` — synthetic `holidays_df` with an NI-only holiday (e.g. 12 July) → `is_bank_holiday=0` for a frame containing that date (per **D2**).
  - `test_derive_calendar_no_is_weekend_column` — output schema must not contain `is_weekend` (per R5 collinearity guard).
  - `test_derive_calendar_column_order_matches_constant` — `derive_calendar`'s output columns end with exactly the `CALENDAR_VARIABLE_COLUMNS` ordered list.
  - `test_derive_calendar_no_io` — `unittest.mock` over `builtins.open`, `requests.get`, `Path.write_bytes` — none called during derivation (AC-2).
- **Command:** `uv run pytest tests/unit/features/test_calendar.py -q`.

### Task T4 — Assembler extension
*(Depends on T1, T3.)*

- [ ] Extend `src/bristol_ml/features/assembler.py`:
  - Add module-level constant `CALENDAR_OUTPUT_SCHEMA: pa.Schema` = `OUTPUT_SCHEMA` (10 weather columns) + `CALENDAR_VARIABLE_COLUMNS` (42 calendar columns) + `holidays_retrieved_at_utc` scalar column. Total: 53 columns. Column order: weather-only columns first (preserves `OUTPUT_SCHEMA.names` as a prefix), calendar columns next, provenance scalars last.
  - Add `assemble_calendar(cfg: AppConfig, *, cache: CachePolicy = CachePolicy.OFFLINE) -> Path`: orchestrator that (a) calls `assemble(cfg, cache=...)` with the weather-only config, (b) loads the weather-only parquet, (c) loads the holidays parquet via `ingestion.holidays.load`, (d) calls `features.calendar.derive_calendar`, (e) appends `holidays_retrieved_at_utc` as a scalar column, (f) schema-asserts against `CALENDAR_OUTPUT_SCHEMA`, (g) writes to `cfg.features.weather_calendar.cache_dir / cache_filename` via `_atomic_write`.
  - Add `load_calendar(path: Path) -> pd.DataFrame`: schema-validating reader for `CALENDAR_OUTPUT_SCHEMA`. Reuses the existing `load` schema-validation idiom but on the extended schema.
  - Do NOT modify the existing `build()`, `assemble()`, or `load()` — those continue to serve the `weather_only` feature set unchanged.
- [ ] Extend `src/bristol_ml/features/__init__.py` — add `assemble_calendar`, `load_calendar`, `CALENDAR_OUTPUT_SCHEMA`, `CALENDAR_VARIABLE_COLUMNS` to the module's lazy re-export set (if the features package uses the pattern; if not, import directly from `assembler`).
- [ ] Extend `src/bristol_ml/features/CLAUDE.md` with the new surface + the two-schemas-in-one-module convention.
- **Acceptance:** AC-3, AC-7.
- **Tests (spec-derived):**
  - `test_calendar_output_schema_is_weather_schema_plus_calendar_plus_provenance` — structural: `CALENDAR_OUTPUT_SCHEMA.names[:10]` matches `OUTPUT_SCHEMA.names`; columns 10–51 match `CALENDAR_VARIABLE_COLUMNS`; last column is `holidays_retrieved_at_utc`.
  - `test_assemble_calendar_writes_parquet` — with mock caches for demand, weather, holidays, `assemble_calendar(cfg)` returns a `Path` that exists and passes `load_calendar`.
  - `test_load_calendar_rejects_weather_only_schema` — loading a `weather_only` parquet via `load_calendar` raises because calendar columns are missing.
  - `test_load_rejects_weather_calendar_schema` — the reverse (existing `load` rejects extra columns — already tested; re-verify the invariant is preserved in this plan).
  - `test_assemble_calendar_is_idempotent` — second call with `cache=AUTO` on a warm cache returns the same path (compose with the upstream `assemble()` idempotence test).
  - `test_assemble_calendar_provenance_scalars_preserved` — the three `*_retrieved_at_utc` columns are all single scalars across the parquet.
- **Command:** `uv run pytest tests/unit/features/test_assembler_calendar.py -q`.

### Task T5 — Train CLI feature-set selector
*(Depends on T1, T4.)*

- [ ] Extend `src/bristol_ml/train.py`:
  - Add `_resolve_feature_set(cfg: AppConfig) -> tuple[FeatureSetConfig, LoadFn, tuple[str, ...]]`:
    - If `cfg.features.weather_only is not None and cfg.features.weather_calendar is None` → returns `(cfg.features.weather_only, assembler.load, tuple(name for name, _ in WEATHER_VARIABLE_COLUMNS))`.
    - If `cfg.features.weather_calendar is not None and cfg.features.weather_only is None` → returns `(cfg.features.weather_calendar, assembler.load_calendar, tuple(weather + calendar names))`.
    - Else → raise `ValueError("Exactly one of features.weather_only or features.weather_calendar must be set; use 'features=<name>' CLI override")`.
    - `LoadFn` is a structural `Protocol` — `Callable[[Path], pd.DataFrame]`.
  - Replace the hard-coded `fset = cfg.features.weather_only` at line 106 with the `_resolve_feature_set` call.
  - Pass the resolved `feature_column_names` to `harness.evaluate(..., feature_columns=...)` so the harness does not silently fall back to the weather-only default.
  - Pass the resolved `feature_column_names` to `LinearModel`'s config at instantiation time (or, if `LinearConfig.feature_columns` is explicitly set via Hydra, respect that and log the override). The metadata `name` on the fitted model should reflect the feature set — `"linear-ols-weather-calendar"` vs `"linear-ols-weather-only"`. Two paths: (a) make `LinearModel` derive its metadata name from `config.feature_columns` content, (b) let `train.py` override `LinearConfig.feature_columns` + name at instantiation. Pick (b) because it keeps `LinearModel` unchanged.
  - Exit-code contract unchanged: `0` success, `2` missing config/cache, `3` unknown variant.
- [ ] Extend `README.md` with the `python -m bristol_ml.train features=weather_calendar` invocation and link the Stage 5 notebook.
- **Acceptance:** AC-1, AC-9.
- **Tests (spec + implementation-derived):**
  - `test_resolve_feature_set_weather_only` — default-config path.
  - `test_resolve_feature_set_weather_calendar` — `features=weather_calendar` override path.
  - `test_resolve_feature_set_both_populated_raises` — the degenerate case.
  - `test_resolve_feature_set_neither_populated_raises` — the other degenerate case.
  - `test_train_cli_features_override_swaps_feature_set` — run `_cli_main(["features=weather_calendar"])` in-process; capture stdout; assert the metric table is printed and `metadata.name` includes `"weather-calendar"`.
  - `test_train_cli_weather_only_still_works` — regression: `_cli_main([])` still prints the Stage-4-era metric table (the existing Stage 4 `test_train_cli.py` tests are not disturbed).
- **Command:** `uv run pytest tests/unit/test_train_cli.py -q && uv run python -m bristol_ml.train features=weather_calendar --help`.

### Task T6 — Demo notebook
*(Depends on T1–T5.)*

- [ ] Create `notebooks/05_calendar_features.ipynb`:
  - **Cell 1 (md):** stage goal; note that the same `LinearModel` class + same `SplitterConfig` are used for both runs; the only difference is the feature set.
  - **Cell 2 (code):** `load_config()` with the notebook D7 / D8 override — `evaluation.rolling_origin.min_train_periods=720`, `evaluation.rolling_origin.step=168`; `assembler.load()` and `assembler.load_calendar()` on warm caches; print the two schemas' column counts.
  - **Cell 3 (code):** `weather_only` run — `LinearModel(LinearConfig(feature_columns=<weather_names>))`; `harness.evaluate(...)`; display the per-fold metric table. Print `results.summary()` as the baseline coefficient print-out.
  - **Cell 4 (code):** `weather_calendar` run — `LinearModel(LinearConfig(feature_columns=<weather_names + calendar_names>))`; `harness.evaluate(...)`; display the per-fold metric table. Print `results.summary()` showing the calendar coefficients — **this is the teaching-moment table**; facilitators point at the hour-of-day dummies to read the load profile.
  - **Cell 5 (code):** side-by-side — concatenate the two metric DataFrames on `.mean()` per metric; display as a 2-row table; markdown framing: "That's what domain knowledge bought us."
  - **Cell 6 (code):** three-way benchmark — `compare_on_holdout({"linear_weather_only": ..., "linear_weather_calendar": ...}, ...)` (if the NESO forecast cache is warm); otherwise markdown skip.
  - **Cell 7 (code):** residual-ripple visualisation (AC-5). Select a shared test week (the last week of the final fold); compute residuals for both models on that week; plot both residual series on shared x-axis, one colour per model; legend distinguishes the two. The weekly oscillation should be visible in `weather_only` and largely absent in `weather_calendar`.
  - **Cell 8 (md):** closing narrative — what calendar features added (MAPE improvement quantified); what they did not (Stage 6 interactions teasing; REMIT's role for one-off events; lag features at Stage 7).
- [ ] Smoke check: `uv run jupyter nbconvert --to notebook --execute notebooks/05_calendar_features.ipynb --output /tmp/05_test_run.ipynb` finishes **under 120 s** with warm caches (per **D8**; AC-12).
- **Acceptance:** AC-4, AC-5, AC-12.
- **No new tests.** nbconvert smoke is the gate (as in Stage 3 T5, Stage 4 T9).

### Task T7 — Stage hygiene
*(Depends on T1–T6.)*

- [ ] `CHANGELOG.md` under `[Unreleased]`: `### Added` bullets for `bristol_ml.ingestion.holidays`, `bristol_ml.features.calendar`, `assembler.assemble_calendar` / `load_calendar` / `CALENDAR_OUTPUT_SCHEMA` / `CALENDAR_VARIABLE_COLUMNS`, `train._resolve_feature_set`, new Hydra groups, notebook, tests. `### Changed` bullet noting the `train` CLI now supports `features=weather_calendar`.
- [ ] `docs/lld/stages/05-calendar-features.md` — retrospective following `docs/lld/stages/00-foundation.md` / `04-linear-baseline.md` template. Document any deviations from this plan; cite the R4 empirical evidence driving D-3; cite the R5 `is_weekend` collinearity guard implemented in `derive_calendar`.
- [ ] `docs/stages/README.md` — flip Stage 5 status cell to `shipped`; link brief → plan, layer → features, retro → `05-calendar-features.md`.
- [ ] `docs/architecture/layers/features.md` — extend module inventory with `calendar.py` (Shipped) and `assembler.py` gaining the `weather_calendar` schema (Shipped); update the `Open questions` section to mark the "feature-table schema contract for Stage 5" and "population-weighting home" questions resolved with a back-reference (per **H-4**).
- [ ] `docs/architecture/ROADMAP.md` — drop the "Features" entry since the layer is now fully realised (per **H-4**).
- [ ] Move this plan from `docs/plans/active/` to `docs/plans/completed/` **as part of the final commit only**.
- [ ] **Not** touching `docs/intent/DESIGN.md §6`; deny-tier for the lead. At PR merge, surface **H-3** to the human so §6 can be batched-updated covering Stages 1–5 in one main-session edit.
- **Acceptance:** AC-8, AC-9, AC-10, AC-11.

---

## 7. Files expected to change

### New
- `src/bristol_ml/ingestion/holidays.py`
- `src/bristol_ml/features/calendar.py`
- `conf/ingestion/holidays.yaml`
- `conf/features/weather_calendar.yaml`
- `tests/unit/ingestion/test_holidays.py`
- `tests/unit/features/test_calendar.py`
- `tests/unit/features/test_assembler_calendar.py`
- `tests/fixtures/holidays/cassettes/holidays_refresh.yaml` (VCR)
- `scripts/record_holidays_cassette.py`
- `notebooks/05_calendar_features.ipynb`
- `docs/lld/stages/05-calendar-features.md`

### Modified
- `conf/_schemas.py` — `HolidaysIngestionConfig`; `IngestionGroup.holidays`; `FeaturesGroup.weather_calendar`.
- `conf/config.yaml` — `ingestion/holidays@ingestion.holidays` in `defaults:`.
- `src/bristol_ml/features/assembler.py` — add `CALENDAR_VARIABLE_COLUMNS` (re-exported from `calendar.py`), `CALENDAR_OUTPUT_SCHEMA`, `assemble_calendar`, `load_calendar`.
- `src/bristol_ml/features/__init__.py` — re-export new symbols.
- `src/bristol_ml/features/CLAUDE.md` — document `calendar.py` + `weather_calendar` schema.
- `src/bristol_ml/ingestion/CLAUDE.md` — document `holidays.py`.
- `src/bristol_ml/train.py` — `_resolve_feature_set`; replace hard-coded `cfg.features.weather_only` read.
- `tests/unit/test_config.py` — features-override and holidays-config tests.
- `tests/unit/test_train_cli.py` — features-override integration tests.
- `CHANGELOG.md` — `[Unreleased]` bullets.
- `README.md` — add the `python -m bristol_ml.train features=weather_calendar` invocation + Stage 5 worked-example section.
- `docs/stages/README.md` — Stage 5 status cell to `shipped`.
- `docs/architecture/layers/features.md` — module inventory + resolved Open questions.
- `docs/architecture/ROADMAP.md` — drop "Features" section.

### Intentionally not modified
- `docs/intent/**` — deny-tier. Intent is immutable. `DESIGN.md §6` update is deferred to a human-led batched edit (H-3).
- `src/bristol_ml/cli.py`, `__main__.py` — feature-set selection is a Hydra override, not a subcommand.
- `src/bristol_ml/config.py` — `load_config` signature unchanged.
- `src/bristol_ml/evaluation/**` — Stage 5 does not touch the harness, metrics, or benchmarks.
- `src/bristol_ml/models/**` — Stage 5 reuses Stage 4's `LinearModel` / `NaiveModel` unchanged. `LinearConfig.feature_columns` is the only surface that gets a non-default population, and that is a config edit not a code edit.
- `pyproject.toml` / `uv.lock` — no new runtime deps. Confirm `vcrpy`, `requests`, `pandas`, `pyarrow`, `loguru` are all already declared at Phase 2 kickoff.
- `docs/architecture/layers/evaluation.md`, `docs/architecture/layers/models.md`, `docs/architecture/layers/ingestion.md` — no contract changes. (The `ingestion.md` module inventory gets a one-row addition for `holidays.py` but the Contract section is unchanged; this is an internals-section edit.)

---

## 8. Exit criteria (definition of done per DESIGN §9)

- All tests pass: `uv run pytest -q` green; no `xfail`, no skipped.
- Lint/format clean: `uv run ruff check .`, `uv run ruff format --check .`.
- Pre-commit clean: `uv run pre-commit run --all-files`.
- Standalone CLIs exit 0: `python -m bristol_ml.ingestion.holidays --help`, `python -m bristol_ml.features.calendar --help`, `python -m bristol_ml.train --help`, `python -m bristol_ml.train features=weather_calendar --help`.
- Hydra feature-set swap works end-to-end: `python -m bristol_ml.train features=weather_only` and `python -m bristol_ml.train features=weather_calendar` both exit 0 and print metric tables.
- Notebook runs top-to-bottom under **120 s** with warm caches (per **D8**; AC-12).
- Every new public symbol has a British-English docstring.
- CHANGELOG bullets present under `[Unreleased]`.
- Retrospective at `docs/lld/stages/05-calendar-features.md`.
- Stages index status cell updated to `shipped`.
- `docs/architecture/layers/features.md` module inventory updated; `docs/architecture/ROADMAP.md` Features section dropped (**H-4**).
- This plan moved from `docs/plans/active/` to `docs/plans/completed/` in the final commit.
- **H-3** surfaced to the human at PR review (§6 layout tree accumulation, deny-tier).

---

## 9. Team-shape recommendation

**Sequential single-session** work by the lead (me), following the orchestrator playbook — Phase 2 task-by-task, spawning `@test-author` after each code task to write spec-derived tests before declaring the task complete.

Rationale:
- T1, T2, T3 have minimal data dependencies on each other and could theoretically run in parallel. In practice, T2's VCR cassette recording is the one task that may need a manual network touch (even via `record_holidays_cassette.py`); serial execution keeps the debugging loop tight.
- T4 strictly depends on T1 + T3; T5 strictly depends on T4; T6 (notebook) strictly depends on all of T1–T5.
- No new research needed beyond R1–R10. No novel algorithms; statsmodels OLS was the hardest library surface in Stage 4 and is reused unchanged.

Escalate to `@reframer` only if a task fails three times with the same framing (per CLAUDE.md §Escalation ladder). Early candidates for fragility:
- T3 (`derive_calendar` DST rule) — pandas `tz_convert` / `tz_localize` edge cases are a known source of subtle bugs; R6 describes the pandas GH#47398 trap. Expect one revision if the synthetic test fixture spans a real spring-forward Sunday.
- T5 (`_resolve_feature_set` Hydra semantics) — Hydra's `features=<name>` group override behaviour for a group that already has a `defaults:` entry may not match the naïve expectation; plan for one revision.

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| gov.uk changes the JSON shape of the endpoint mid-stage | Low | High | VCR cassette pins the response shape; the `OUTPUT_SCHEMA` declaration derives from the cassette. A shape change surfaces as a schema-assert test failure, not a silent data corruption. |
| `python-holidays` (R9) is the wrong fallback path and the human prefers a committed CSV (D-6 override) | Low | Low | D-6 is a single-line reversal; test `test_derive_calendar_pre_2012_warns_and_fills_zero` would be replaced by one asserting the raise. |
| NI composition is actually wanted for a future regional story (D-2 override) | Medium | Low | The ingester persists all three divisions to the cache; `derive_calendar` filters to two. Adding `is_bank_holiday_ni` as a separate column requires one line in `derive_calendar` + one line in `CALENDAR_VARIABLE_COLUMNS` — no architectural change. |
| R4's one-hot vs sin/cos evidence does not translate to GB demand data; the one-hot OLS overfits on the small test folds | Low | Medium | The sklearn evidence is on an analogous hourly demand series with similar data volume. OLS at n=8760, p≈48 is still well-conditioned. If the notebook shows this, revisit D-3/D-4 in Stage 6. No blocker for Stage 5's without/with comparison. |
| `is_weekend` collinearity guard (R5) fires inadvertently when a future stage extends `CALENDAR_VARIABLE_COLUMNS` | Low | Low | The assert has a clear error message citing R5. A future extension must consciously either drop the one-hot day-of-week encoding (change D-4) or omit `is_weekend`. |
| `derive_calendar` fails on a real DST-change day with data-gap edge case (pandas GH#47398) | Medium | Medium | Synthetic test fixture covers a real spring-forward Sunday. If the test passes and production fails, the plan's DST rule is wrong; escalate to `@reframer`. |
| `train.py` Hydra `features=` override semantics don't mutually-exclude the two config blocks as the plan assumes | Medium | Medium | `_resolve_feature_set` raises on the "both populated" / "neither populated" cases. Tests `test_resolve_feature_set_both_populated_raises` / `_neither_populated_raises` pin the expectation. If Hydra populates both, the resolver surfaces that and the plan's YAML wiring must change. |
| Notebook exceeds 120 s budget with ~40 extra feature columns | Low | Low | OLS fit at n=8760, p≈48 is ~10 ms per fold; 52 weekly folds = ~0.5 s for fits alone. The full notebook should comfortably stay under 15 s on a warm cache. If breached, drop `min_train_periods` further. |
| NESO forecast three-way comparison breaks because the two `LinearModel` instances share hidden state across `compare_on_holdout` calls | Low | Medium | `LinearModel` fit state is per-instance; `compare_on_holdout` re-fits per fold. The test `test_benchmarks_three_way_table_shape` from Stage 4 is a regression guard. |
| `CALENDAR_OUTPUT_SCHEMA` column order drifts between `derive_calendar` output and the schema constant | Medium | High | `test_calendar_output_schema_is_weather_schema_plus_calendar_plus_provenance` pins the ordering; `test_derive_calendar_column_order_matches_constant` pins `derive_calendar`'s output. Both must pass before T4 reports complete. |

---

## Human sign-off

*Pending.*

- D1 (gov.uk/bank-holidays.json + VCR cassette): **ACCEPT / OVERRIDE**
- D2 (GB composite = E&W ∪ Scotland; NI excluded): **ACCEPT / OVERRIDE**
- D3 (one-hot `hour_of_day`): **ACCEPT / OVERRIDE**
- D4 (one-hot `day_of_week` + one-hot `month`): **ACCEPT / OVERRIDE**
- D5 (binary pair `is_day_before_holiday` + `is_day_after_holiday`): **ACCEPT / OVERRIDE**
- D6 (WARNING + fill 0 for pre-2012; `python-holidays` documented as extension path): **ACCEPT / OVERRIDE**
- D7 (`Europe/London` local-date DST rule): **ACCEPT / OVERRIDE**
- D8 (notebook `step=168` + `min_train_periods=720` override): **ACCEPT / OVERRIDE**
- D9 (Option 3 — composable `derive_calendar` + parallel `CALENDAR_OUTPUT_SCHEMA`): **ACCEPT / OVERRIDE**
- D10 (Hydra `features=weather_only | weather_calendar` group override; `_resolve_feature_set` in `train.py`): **ACCEPT / OVERRIDE**
- D11 (new `notebooks/05_calendar_features.ipynb`): **ACCEPT / OVERRIDE**

Housekeeping:
- H-1 (`NesoBenchmarkConfig.holdout_start/_end` → Stage 6): **ACCEPT / OVERRIDE**
- H-2 (`ingestion._neso_dst` extraction → Stage 13): **ACCEPT / OVERRIDE**
- H-3 (human-led batched `DESIGN.md §6` edit covering Stages 1–5 at PR review): **ACCEPT / OVERRIDE**
- H-4 (`docs/architecture/ROADMAP.md` Features section closed at Stage 5 T7): **ACCEPT / OVERRIDE**
