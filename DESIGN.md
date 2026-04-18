# Bristol ML Reference — Design Document

**Status:** Draft v0.2
**Audience:** Project author, future contributors, meetup facilitators, and Claude Code
**Scope:** Design intent and implementation plan. Reflects current + near-term reality. Not a user guide.

---

## 1. Project purpose

This project has two goals that inform every design decision. They are listed in order of authority: where they conflict, the pedagogical purpose wins.

### 1.1 Pedagogical purpose (primary)

A reference implementation of a generic ML / data-science architecture, used as a live-demo surface at meetups and as a self-paced learning artefact that matures over months.

**Shape of use:**
- A facilitator at a hands-on meetup picks a component on the fly based on the room's interests and walks through it end-to-end.
- Attendees of mixed ability engage at their own level: some clone and run, some code along with their own API keys, some watch the plots.
- The repo is the artefact; sessions come and go. Quality, completeness and legibility matter more than polish or workshop choreography.
- Claude Code is a first-class tool. The repo is structured so Claude Code can navigate modules with minimal context bloat.

**Quality bar:**
- Every component runs in isolation.
- Every component produces a visible output (plot, printed table, API response).
- `tree -L 2` tells a coherent story.
- Maturation path is one focused stage per session, roughly 2-4 hours each.

### 1.2 Analytical purpose (secondary)

A GB day-ahead electricity demand forecaster, with price forecasting as a secondary target retained to prove architectural generalisation. The project is named `bristol_ml` rather than anything energy-specific so that a future case study (different domain, same architecture) can reuse the scaffolding without a rename.

**Primary target:** GB national electricity demand (`ND`), hourly, day-ahead horizon.
**Secondary target:** GB day-ahead wholesale price (N2EX/EPEX Market Index Data).
**Benchmark:** NESO's own day-ahead demand forecast.

The energy domain is well-chosen because it exercises every architectural slot: numerical time series, weather as exogenous driver, a rich textual disclosure stream (REMIT), and a published benchmark to beat.

---

## 2. Guiding principles

### 2.1 Architectural

1. **Every component runs standalone.** No component depends on a running service, an orchestrator, or environment state beyond the local filesystem and (optionally) network. `python -m bristol_ml.<module>` is the baseline invocation pattern.
2. **Interfaces are typed and narrow.** Parquet with documented schemas at storage boundaries; Pydantic models for configs; typed function signatures at module boundaries. Internal code can be looser.
3. **Stub-first for expensive or flaky dependencies.** The LLM feature extractor is an interface with a hand-labelled stub and a real implementation. The stub is the default for CI and for attendees without API keys.
4. **Configuration lives outside code.** API endpoints, station lists, feature specs, model hyperparameters are all in YAML under `conf/`. Code reads configs; configs do not contain Python logic.
5. **Idempotent operations.** Re-running ingestion overwrites or skips, never corrupts. Re-running training produces comparable artefacts (exact reproducibility not promised for GPU-trained models).
6. **Provenance by default.** Every derived artefact records its inputs, git SHA, and wall-clock timestamp. Registry entries link back to the feature table and raw source versions.
7. **Tests at boundaries, not everywhere.** Each module has at least a smoke test on its public interface. Coverage is not a goal; behavioural clarity is.
8. **Notebooks are thin.** Notebooks call into `src/bristol_ml/`; they do not reimplement logic. They are the demo surface, not the implementation.

### 2.2 Pedagogical

1. **Every stage produces a demoable artefact.** Stages land with a plot, a table, a served endpoint, or a printed comparison.
2. **Stages are small enough for one Claude Code session.** Two to four hours of focused work, one clear goal.
3. **Stages are independently interesting.** An attendee picking a mid-project stage gets something coherent without having internalised every preceding stage.
4. **Complexity is earned.** Advanced patterns (bi-temporal storage, proper LLM extraction) land in stages where their purpose is concretely motivated — not front-loaded "because that's how you do it properly."
5. **The repo explains itself.** `README.md` pitches; `DESIGN.md` (this document) scopes; `docs/stages/` records what each stage did and why; `docs/adrs/` records decisions that seem obvious in retrospect but weren't.

---

## 3. System architecture

The organising model is the three-layer synthesis from our earlier framework discussion: a **process layer** (what and why), a **delivery layer** (how it reaches users reliably), and **cross-cutting concerns** (what stops it rotting). The repo structure collapses this into module directories under `src/bristol_ml/` rather than a literal layer hierarchy — nested paths like `delivery/ingestion/` add noise without adding clarity.

### 3.1 Data flow

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  NESO CKAN  │  │ Open-Meteo  │  │  gov.uk     │  │   Elexon    │
│  (demand,   │  │  (weather)  │  │ (holidays)  │  │  Insights   │
│  forecasts) │  │             │  │             │  │  (REMIT,    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │ price, gen) │
       │                │                │          └──────┬──────┘
       └────────────────┴────────────────┴─────────────────┘
                                │
                         ┌──────▼──────┐
                         │  ingestion  │  fetchers, raw parquet
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  features   │  joins, lags, calendar
                         └──────┬──────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
        ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │   models    │  │     llm     │  │ evaluation  │
        │ (linear,    │  │ (extractor, │  │ (metrics,   │
        │  sarimax,   │  │  embeddings)│  │  rolling-   │
        │  scipy, NN) │  │             │  │  origin)    │
        └──────┬──────┘  └──────┬──────┘  └─────────────┘
               │                │
               └────────┬───────┘
                        │
                 ┌──────▼──────┐
                 │  registry   │  artefacts + metadata
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   serving   │  FastAPI, batch scoring
                 └─────────────┘

Cross-cutting: conf/, monitoring/, orchestration (late stage).
```

### 3.2 Layer responsibilities

**Ingestion** owns the edge: HTTP calls, retries, schema at rest. It knows API shapes. It writes parquet to `data/raw/` and does not transform.

**Features** owns derivations from raw data: calendar features, lags, joins, weather-demand alignment. Pure transformation over parquet. No I/O against external APIs.

**Models** owns training loops, hyperparameter choices, and inference. Each model implements a common interface (see §7.3) and produces a registry artefact.

**Evaluation** owns metrics and the rolling-origin evaluator. It is a library consumed by models and notebooks; it does not own training.

**LLM** owns the REMIT extractor (stub and real) and the embedding index. Exposes typed features that the features layer consumes. Separate from features because its cost and quality properties are different.

**Registry** owns artefact persistence and metadata. Filesystem-based initially; designed so graduation to MLflow is a swap, not a rewrite.

**Serving** owns the FastAPI app that wraps registry artefacts.

**Monitoring** owns drift detection and prediction-quality tracking. Late-stage addition.

### 3.3 What is deliberately not in scope (yet)

- Full CI/CD/CT per Google MLOps level 2. This project sits at level 1 (automated training pipeline) at best.
- Feature store as a separate service. The "feature store" is parquet with a schema.
- GPU-scheduled training. Models are small enough for a laptop.
- Multi-user serving, auth, secrets rotation, deployment targets.
- Probabilistic / quantile forecasting. All models are point forecasters.

---

## 4. Data sources

All sources are free and documented.

### 4.1 NESO Data Portal — demand and forecast benchmark

- **Endpoint pattern:** `https://api.neso.energy/api/3/action/datastore_search?resource_id=<UUID>`
- **Datasets used:** Historic Demand Data (half-hourly `ND` and `TSD` back to 2001, one resource per year); Historic Day-Ahead Demand Forecasts (2018 onwards, benchmark).
- **Licence:** NESO Open Data / OGL.

### 4.2 Open-Meteo — weather

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive` (historical); `https://api.open-meteo.com/v1/forecast`.
- **Resolution:** hourly, ~10 km via UKMO UKV 2 km model.
- **Stations:** configured list of 5-10 UK population centres (London, Manchester, Birmingham, Glasgow, Leeds, Bristol, Cardiff, Belfast, Edinburgh, Newcastle). Weighted aggregation to a national weather signal.
- **Licence:** CC-BY-4.0.
- **Why not Met Office DataHub:** the free tier only returns the last 48 hours of history, insufficient for training.

### 4.3 GOV.UK Bank Holidays

- **Endpoint:** `https://www.gov.uk/bank-holidays.json`
- **Structure:** three divisions (england-and-wales, scotland, northern-ireland). All three used because the GB grid spans all.
- **Auth:** none.

### 4.4 Elexon Insights Solution — price, REMIT, generation mix

- **Base URL:** `https://data.elexon.co.uk/bmrs/api/v1/`
- **Auth:** none (public, no key required).
- **Datasets used:** Market Index Data (`/balancing/pricing/market-index`); REMIT messages (`/remit`); Generation by fuel type; System prices (imbalance, secondary signal).
- **Licence:** BMRS Data Licence Terms (permissive for non-commercial/academic).
- **REMIT specifics:** events carry published time, effective-from, effective-to, and free-text description. The bi-temporal nature motivates REMIT having its own ingestion stage.

### 4.5 Summary

| Feed | Role | Source | Auth | Cadence |
|------|------|--------|------|---------|
| National demand (ND) | Primary target | NESO CKAN | None | Half-hourly |
| Day-ahead demand forecast | Benchmark | NESO CKAN | None | Daily |
| Day-ahead price (MID) | Secondary target | Elexon Insights | None | Half-hourly |
| Generation by fuel | Price model input | Elexon Insights | None | Half-hourly |
| REMIT messages | LLM tier input | Elexon Insights | None | Event-driven |
| Weather (multi-station) | Primary exogenous | Open-Meteo | None | Hourly |
| Bank holidays | Calendar features | gov.uk | None | Annual refresh |

---

## 5. Target definitions

### 5.1 Primary: day-ahead GB demand

- **Variable:** `ND` (National Demand), aggregated from half-hourly to hourly.
- **Horizon:** 24 hourly predictions for day D, given information up to end of day D−1.
- **Training window:** 2018-01-01 onwards.
- **Test split:** last 12 months, rolling-origin evaluation with one-day step.

### 5.2 Secondary: day-ahead GB price

- **Variable:** N2EX Day-Ahead Hourly MID, or EPEX 60-minute MID.
- **Horizon:** same as demand.
- **Role:** design target — it stresses the architecture's ability to handle a second target through the same scaffolding. Price accuracy is not a performance goal.

### 5.3 Evaluation metrics

- Primary point metrics: MAE, MAPE, RMSE, WAPE.
- Comparison against NESO day-ahead forecast (MAE ratio).
- Rolling-origin evaluator reports mean and spread across folds.

---

## 6. Repository layout (Stage 0 only)

This section describes **what Stage 0 creates**. Subsequent stages extend this layout and update this section in the same PR — the document always reflects what exists on `main`.

```
bristol-ml-reference/
├── CLAUDE.md                    # Top-level Claude Code guidance
├── README.md                    # Pitch, quick start, entry points
├── CHANGELOG.md
├── DESIGN.md                    # This document
├── pyproject.toml               # uv-managed
├── uv.lock
├── .pre-commit-config.yaml
├── .gitignore
├── .github/workflows/
│   └── ci.yml                   # Lint + pytest on push
│
├── conf/                        # Hydra config tree
│   ├── config.yaml              # Top-level defaults list
│   └── _schemas.py              # Pydantic schemas for resolved configs
│
├── data/                        # gitignored, local cache
│   └── .gitkeep
│
├── src/bristol_ml/
│   ├── __init__.py
│   ├── cli.py                   # Hydra entry point
│   └── config.py                # Config resolution + validation helpers
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── docs/
    ├── architecture.md          # Pointer into §3 of this doc
    ├── stages/                  # One markdown per completed stage
    │   └── 00-foundation.md
    └── adrs/                    # Architecture decision records
        ├── 0001-use-hydra-plus-pydantic.md
        └── 0002-filesystem-registry-first.md
```

Stage 0 does **not** create the module directories for ingestion, features, models, evaluation, llm, registry, serving, or monitoring. Those arrive in their respective stages along with a `CLAUDE.md`, at least one test, and an entry in `docs/stages/`.

---

## 7. Configuration and extensibility

Configuration is the single most important piece of architectural plumbing in this project, because the whole point is pluggable components. The choices here deserve a section rather than a line in a table.

### 7.1 Framework: Hydra + Pydantic

**Hydra** owns composition, CLI overrides, config groups, and multirun. **Pydantic** validates the resolved config before it reaches application code. YAML is the persisted form. ADR 0001 records the reasoning and rejected alternatives.

The pattern is:

1. Hydra loads and composes YAML from `conf/`, applying any CLI overrides.
2. The resolved `DictConfig` is converted to `dict` via `OmegaConf.to_container(..., resolve=True)`.
3. The `dict` is passed to a Pydantic model that validates types, ranges, and cross-field invariants.
4. Application code receives the Pydantic model. Downstream code never sees raw `DictConfig`.

Rejected alternatives: Pydantic Settings alone (no config groups, which pluggable models require); Hydra alone (type validation too loose — an `int` field silently accepts a string); hydra-zen (Python-native configs obscure the config/code separation the project is teaching); OmegaConf alone (no CLI override story worth the name).

### 7.2 Override precedence

Resolved once, stated once, enforced by the config loader. Higher numbers override lower:

1. Defaults declared in Pydantic schema field definitions.
2. Values from `conf/config.yaml` and composed group files.
3. Environment variables (for secrets only: `BRISTOL_ML_<SCREAMING_SNAKE>`).
4. Hydra CLI overrides (e.g. `model=sarimax training.epochs=50`).
5. Explicit arguments passed programmatically (tests, notebooks).

Secrets never live in YAML. API keys (where required) flow through environment variables only.

### 7.3 The `Model` interface

Every model inherits from `bristol_ml.models.base.Model` and implements:

```python
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

Models are constructed from config via Hydra's `_target_` pattern. `model=linear` instantiates `LinearModel`; `model=sarimax` instantiates `SarimaxModel`. No factory code to edit, no dispatcher to update.

The interface itself is introduced in the stage that adds the first model. Stage 0 commits to the pattern in principle; the concrete `base.py` lands with Stage 4.

### 7.4 Config groups

`conf/` is organised by pluggable concern. Each subdirectory is a Hydra group and each file in it is a selectable option. The structure is sketched here for reference, but only `conf/config.yaml` and `conf/_schemas.py` exist after Stage 0 — groups are added by their respective stages.

```
conf/
├── config.yaml              # Defaults list: which option from each group
├── _schemas.py              # Pydantic schemas
├── ingestion/               # Added by ingestion stages
├── features/                # Added by feature stages
├── model/                   # Added by Stage 4 onward, one file per model
├── evaluation/              # Added by Stage 4
└── registry/                # Added by Stage 9
```

CLI swaps at runtime once groups exist: `python -m bristol_ml train model=sarimax evaluation.folds=12`.

### 7.5 Adding a new model: worked pattern

A new model lands as one Python file plus one YAML file, nothing else:

1. `src/bristol_ml/models/new_model.py` — implements the `Model` protocol.
2. `conf/model/new_model.yaml` — sets `_target_: bristol_ml.models.new_model.NewModel` and default hyperparameters.
3. Optional: a Pydantic schema for the model's specific config block, referenced from `conf/_schemas.py`.

No registry of model names, no factory to edit, no dispatcher to update.

---

## 8. Technology choices

Each choice is swappable. ADRs record reasoning where non-obvious.

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Language | Python 3.12 | Default; widest familiarity |
| Packaging | uv | Fast, modern, lockfile-native |
| Lint/format | ruff | One tool, fast |
| Tests | pytest | Default |
| DataFrames | pandas | Universally known; pedagogy over performance |
| Storage | Parquet (pyarrow) | Typed, columnar, portable |
| Config | Hydra + Pydantic | See §7 |
| Classical stats | statsmodels | OLS, SARIMAX, diagnostics in one place |
| Optimisation models | scipy.optimize | Idiomatic for parametric fits |
| Neural nets | PyTorch | Clearest training loops |
| Serving | FastAPI | Minimal, typed, auto-docs |
| LLM | Anthropic SDK | In keeping with Claude Code usage |
| Embeddings | sentence-transformers | Runs offline; no API cost |
| Vector index | numpy for small, FAISS if scale demands | Keep simple unless forced otherwise |
| Registry | Filesystem + JSON | Swap for MLflow later; see ADR 0002 |
| Orchestration | Prefect | Easier local demo than Airflow |
| Logging | loguru | Readable output for demos |
| Docs | MkDocs + Material | Renders markdown; fine for a reference repo |

---

## 9. Stage plan

Each stage is a unit of Claude Code work: one focused session, one demoable output, one PR, one entry in `docs/stages/`.

Stages are ordered to maximise "something visible throughout" rather than strictly by layer. After the first model arrives (Stage 4), stages fan out so a facilitator or author can pick any of several in-progress threads.

### Stage definition of done

Every stage ships with:
- Code on `main`, CI green.
- At least one test on the public interface.
- `CLAUDE.md` added or updated for any touched module.
- `README.md` updated with any new entry point.
- `docs/stages/NN-name.md` capturing what was built and why.
- A notebook demonstrating the output (where the stage produces user-facing output).
- `CHANGELOG.md` entry.
- Section 6 of this document updated to reflect the new layout.

### Stage list

**Stage 0 — Project foundation**
Repo scaffold: `pyproject.toml`, CI, pre-commit, root `CLAUDE.md`, `README.md`, this `DESIGN.md`, initial ADRs. Hydra + Pydantic config plumbing (`conf/`, `src/bristol_ml/config.py`, `cli.py`). Empty `tests/` and `docs/` trees.
*Demo moment:* `uv run pytest` passes on an empty repo; `python -m bristol_ml --help` shows the Hydra CLI; `tree -L 2` tells the whole story.

**Stage 1 — NESO demand ingestion + first plot**
`ingestion/neso.py` fetches historic demand via CKAN; writes parquet; notebook plots a week of hourly demand.
*Demo moment:* real GB demand on screen in five minutes from a clean clone.

**Stage 2 — Weather ingestion + joined plot**
`ingestion/weather.py` for Open-Meteo, multi-station, population-weighted. Notebook overlays temperature vs demand scatter.
*Demo moment:* the non-linear temperature-demand relationship is visible.

**Stage 3 — Feature assembler + train/test split**
`features/assembler.py` produces the canonical feature table (demand + weather only at this point). `evaluation/rolling_origin.py` provides the split. Clear extension point for future feature additions.
*Demo moment:* one command, one parquet file, one schema printed.

**Stage 4 — Linear regression baseline + evaluation harness**
`models/base.py` (the `Model` protocol), `models/naive.py` (seasonal-naive), `models/linear.py` (statsmodels OLS), `evaluation/metrics.py`, `evaluation/benchmarks.py` (NESO forecast comparison). Notebook: residuals, forecast plot, metric table vs NESO.
*Demo moment:* "we beat/lose to the NESO day-ahead forecast by X%."

**Stage 5 — Calendar features (without/with comparison)**
`ingestion/holidays.py` + `features/calendar.py`. Hour-of-day, day-of-week, month, holiday indicators, extended to the feature assembler. Linear regression re-run; notebook shows the metric table from Stage 4 alongside the new one.
*Demo moment:* the measurable jump in accuracy from adding calendar features — the pedagogical "without/with" point.

### Stage branches: the fan-out after Stage 5

Stages 6 onwards depend on Stage 5 but can be approached in any order. Grouped by theme rather than strict sequence.

**Stage 6 — Enhanced evaluation & visualisation**
Richer diagnostics on the existing pipeline: residual plots, calibration/reliability plots, per-hour and per-weekday error breakdowns, forecast interval visualisation, a small notebook helper library under `evaluation/plots.py`. Does not add new models; improves how existing ones are seen. All subsequent model stages inherit the richer output.
*Demo moment:* the same linear model's predictions, but now with diagnostics that actually show where it fails.

**Stage 7 — SARIMAX**
`models/sarimax.py`. Plugs into the same evaluator. Notebook: seasonal decomposition diagnostic + forecast comparison.
*Demo moment:* classical stats on the same canvas as linear regression.

**Stage 8 — SciPy parametric load model**
`models/scipy_parametric.py`: `curve_fit` on a hand-specified parametric form (temperature response + diurnal + weekly harmonics). Parameter confidence intervals.
*Demo moment:* a model with interpretable parameters plotted against a complex ML model's predictions.

**Stage 9 — Model registry**
`registry/filesystem.py`. Every model produced so far is retrofitted to save through the registry. CLI command lists models with metrics.
*Demo moment:* `python -m bristol_ml registry list` shows a leaderboard across techniques.

**Stage 10 — Simple neural network**
`models/nn_mlp.py`. Small MLP, training loop with loss logging, saves to registry.
*Demo moment:* the loss curve, training vs validation, plotted live.

**Stage 11 — Complex neural network**
`models/nn_temporal.py`. Temporal CNN or small Transformer. Same training harness.
*Demo moment:* ablation table across all techniques built so far.

**Stage 12 — Minimal serving endpoint**
`serving/api.py`. One FastAPI app, one `POST /predict` endpoint, loads from registry.
*Demo moment:* `curl` against localhost returns a forecast.

**Stage 13 — REMIT ingestion**
`ingestion/remit.py` with bi-temporal storage (published_at, effective_from, effective_to). Notebook: REMIT event density over time, colour-coded by fuel type.
*Demo moment:* visual evidence of when nuclear outages actually hit the market.

**Stage 14 — LLM feature extractor (stub + real)**
`llm/extractor.py` interface, `llm/extractor_stub.py` with hand-labelled events, `llm/extractor_claude.py` with Anthropic SDK. Extraction evaluation harness comparing real vs hand-labelled.
*Demo moment:* the same interface, two implementations, and an accuracy report between them.

**Stage 15 — Embedding index for REMIT**
`llm/embeddings.py` using sentence-transformers locally. Simple vector index (numpy or FAISS). Notebook: nearest-neighbour lookup for a sample event.
*Demo moment:* "show me all outages semantically similar to this one."

**Stage 16 — Model with REMIT features**
`features/remit.py` consumes extracted REMIT features; best-performing prior model retrained with them.
*Demo moment:* ablation showing REMIT contribution to forecast accuracy.

**Stage 17 — Price pipeline (secondary target)**
`ingestion/elexon_prices.py`, `ingestion/elexon_generation.py`. Re-use assembler and one model to train against price.
*Demo moment:* same registry CLI listing models for both demand and price targets.

**Stage 18 — Drift monitoring**
`monitoring/drift.py`: PSI on feature distributions, prediction-quality tracking. Notebook dashboard.
*Demo moment:* a drift plot over the test period with an annotated "here's when COVID happened."

**Stage 19 — Orchestration**
Prefect flow chaining ingestion → features → training → evaluation → registry write. One schedulable DAG.
*Demo moment:* one command re-runs the whole pipeline end-to-end.

### Stage dependency graph

```
0 ─► 1 ─┐
        ├─► 3 ─► 4 ─► 5 ─┬─► 6   (enhanced viz)
0 ─► 2 ─┘                ├─► 7   (SARIMAX)
                         ├─► 8   (SciPy)
                         └─► 9   (registry) ─┬─► 10 ─► 11
                                             ├─► 12  (serving) ─► 19 (orchestration)
                                             ├─► 17  (price)
                                             └─► 18  (drift)

                         5 ─► 13 ─► 14 ─► 15 ─► 16   (REMIT chain)
```

Stages 6, 7, 8, 9 are independent leaves off 5. 13 opens the REMIT chain independently. Anything past the fan-outs can be interleaved freely based on author or facilitator interest.

---

## 10. Deferred decisions

Each becomes an ADR when a stage forces the decision.

- **Registry graduation.** Filesystem-based first; MLflow / W&B when metadata volume justifies it.
- **Feature store as a service.** Out of scope unless lag-feature computation becomes a bottleneck.
- **CI-driven retraining.** Out of scope at level 1 MLOps.
- **Real-time serving.** Out of scope; batch day-ahead scoring is the realistic mode.
- **Multi-zone modelling.** GB is a single zone.
- **Probabilistic forecasting.** Stretch topic.
- **Price-model depth.** The price pipeline proves the architecture generalises; performance is not a goal.

---

## 11. Open questions

1. **Notebook runtime budget.** Should notebooks run top-to-bottom in under ~2 minutes on a laptop? Constrains data window sizes and model sizes at demo time.
2. **How aggressively to cache.** Pre-cached test fixtures are uncontroversial. Pre-cached parquet in `data/raw/` committed to git is more aggressive but guarantees a live demo works offline.
3. **LLM costs in the default path.** The stub extractor keeps CI free. Should the default notebook for Stage 14 use the stub, or the real extractor with a small sample?
4. **Orchestration in scope at all.** Stage 19 is the least pedagogically valuable per hour. Worth confirming before it's built.

---

## Appendix A — Source URLs

- NESO Data Portal: https://www.neso.energy/data-portal
- NESO CKAN API base: https://api.neso.energy/api/3/
- Open-Meteo historical: https://archive-api.open-meteo.com/v1/archive
- Open-Meteo docs: https://open-meteo.com/en/docs
- GOV.UK Bank Holidays: https://www.gov.uk/bank-holidays.json
- Elexon Insights API docs: https://developer.data.elexon.co.uk/
- Elexon BMRS portal: https://bmrs.elexon.co.uk/
- REMIT messages: https://bmrs.elexon.co.uk/remit
- Market Index Data: https://bmrs.elexon.co.uk/market-index-prices

## Appendix B — Framework references

- CRISP-DM: the six-phase data-mining lifecycle used as the process backbone.
- Microsoft TDSP: informs the roles/artefacts convention in `docs/`.
- ThoughtWorks CD4ML (Sato, Wider, Windheuser, 2019): informs the delivery-layer component list.
- Google MLOps Practitioners Guide: informs the maturity framing (level 0/1/2). This project targets level 1.
