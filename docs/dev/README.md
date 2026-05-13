# bristol_ml

Reference ML/DS architecture with a GB day-ahead electricity demand forecaster as the worked example. **The project spec lives at [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md)** — start there for the full scope.

## Quick start

```
git clone <repo>
cd bristol-ml-reference
uv sync --group dev
uv run python -m bristol_ml --help
uv run pytest
```

## API keys and offline-by-default

`bristol_ml` is **offline-by-default** (DESIGN §2.1.3). Every CI run, every notebook, and every test in this repository runs without a network connection or any API credentials. External-API integrations are stub-first: each module that calls a network API ships with a hand-labelled stub backed by an environment-variable discriminator, and the stub path is the runtime default. You only need an API key when you opt into the live path.

The only stage with an authenticated outbound dependency is **Stage 14 (LLM feature extractor)**, which calls the OpenAI API to extract structured features from REMIT free-text messages.

### Configuring an OpenAI API key

1. Create or sign in to a key at [platform.openai.com](https://platform.openai.com/).
2. Export the key into your shell — never commit it:

   ```bash
   export BRISTOL_ML_LLM_API_KEY=sk-...
   ```

3. Switch the LLM extractor to the live path via Hydra override:

   ```bash
   uv run python -m bristol_ml.llm.extractor llm.type=openai
   uv run python -m bristol_ml.llm.evaluate  llm.type=openai
   ```

   Without the override, both entry points run against the in-repo hand-labelled stub.

To force the stub regardless of config (CI default; useful when iterating without spending tokens):

```bash
export BRISTOL_ML_LLM_STUB=1
```

The stub path makes zero network calls and requires no API key. CI sets `BRISTOL_ML_LLM_STUB=1` automatically.

### Environment variables

| Variable | Purpose | CI default |
|----------|---------|------------|
| `BRISTOL_ML_LLM_API_KEY` | OpenAI API key for Stage 14 live extraction | unset |
| `BRISTOL_ML_LLM_STUB` | Force the LLM stub regardless of config | `1` |
| `BRISTOL_ML_REMIT_STUB` | Force the REMIT-ingestion stub regardless of config | `1` |
| `BRISTOL_ML_EMBEDDING_STUB` | Force the embedding stub regardless of config (Stage 15) | `1` |
| `BRISTOL_ML_EMBEDDING_MODEL_PATH` | Override the live embedder model path or HF id (Stage 15) | unset |
| `BRISTOL_ML_CACHE_DIR` | Override the default cache root (`~/.cache/bristol_ml`) | unset (uses default) |

Pre-commit hooks and `.gitignore` keep secrets out of the repository; VCR cassette fixtures filter `authorization` and `x-api-key` headers automatically when re-recorded. Never commit a key.

### Before the meetup

The Stage 15 live embedder downloads `Alibaba-NLP/gte-modernbert-base` (~298 MB safetensors at fp32; ~149 MB resident in RAM at fp16) from the Hugging Face Hub on first use. Pre-warm the cache once so the meetup demo does not stall on a cold network:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"
```

The download lands in `~/.cache/huggingface/`. Subsequent runs (and CI) honour `HF_HUB_OFFLINE=1` and never touch the network. To force the offline stub regardless of cache state, export `BRISTOL_ML_EMBEDDING_STUB=1`.

## What is this?

- A pedagogical reference repo for a generic ML / data-science architecture.
- The worked example is GB national electricity demand, day-ahead.
- [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md) has the full scope. Each stage lands with a demoable artefact.

See [`CLAUDE.md`](CLAUDE.md) for Claude Code guidance, [`docs/architecture/`](docs/architecture/) for architectural decisions, and [`docs/lld/stages/`](docs/lld/stages/) for stage retrospectives.

## Regenerating feature caches after a code change

The assembler writes one of two feature-table parquets depending on the active `features=` Hydra group:

| Group | Output parquet | Stage |
|-------|----------------|-------|
| `weather_only` (default) | `data/features/weather_only.parquet` | 3 |
| `weather_calendar` | `data/features/weather_calendar.parquet` | 5 |

When you change a feature derivation (e.g. add a column to `derive_calendar`), the on-disk parquet's schema no longer matches what `load`/`load_calendar` expect, and the train CLI fails with `Cached parquet at <path> is missing required column '<name>'`. The error message names the regeneration command; for reference:

```bash
# Stage 3 weather-only feature table
uv run python -m bristol_ml.features.assembler features=weather_only --cache offline

# Stage 5 weather + calendar feature table
uv run python -m bristol_ml.features.assembler features=weather_calendar --cache offline
```

`--cache offline` (the default) reuses the warm ingester caches (NESO, weather, holidays) and only re-runs the feature derivation — the right policy after a code-only change. `--cache auto` populates missing ingester caches from the network on first run; `--cache refresh` re-fetches them. The assembler's own output parquet is always written via atomic replace, so a re-run never corrupts a partially-written file.

## Worked example: NESO demand (Stage 1)

`notebooks/01_neso_demand.ipynb` loads cached GB half-hourly demand (sourced from the NESO Data Portal under OGL v3), resamples to hourly, and plots a representative week plus daily peaks across the cached window. The first run populates the local cache from the NESO CKAN API; subsequent runs work offline. See [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md) for the on-disk parquet schema, and the [Stage 1 retrospective](docs/lld/stages/01-neso-demand-ingestion.md) for the design choices.

## Worked example: weather + demand (Stage 2)

`notebooks/02_weather_demand.ipynb` primes a second ingestion cache from the [Open-Meteo](https://open-meteo.com) historical archive (CC BY 4.0), composes ten UK population centres into a population-weighted national temperature signal via `bristol_ml.features.weather.national_aggregate`, joins onto hourly demand, and plots the non-linear temperature-vs-demand curve with a LOWESS overlay. The curve is a **hockey-stick** on historical GB data — strong cold arm, flat warm arm — not a symmetric V; the hot-side response is only beginning to emerge post-2020 as residential cooling grows.

## Worked example: feature assembler + rolling-origin folds (Stage 3)

`notebooks/03_feature_assembler.ipynb` assembles the first end-to-end hourly feature table — demand joined with the population-weighted national weather aggregate — and enumerates rolling-origin train/test folds against it. The notebook calls `assembler.assemble(cfg, cache="auto")` to write `data/features/weather_only.parquet`, round-trips it through `assembler.load`, then plots fold boundaries overlaid on the GB demand series. Runs end-to-end in approximately 6 seconds on warm caches (D7). Entry points: `python -m bristol_ml.features.assembler --help` and `python -m bristol_ml.evaluation.splitter --help`. See [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md), [`src/bristol_ml/evaluation/CLAUDE.md`](src/bristol_ml/evaluation/CLAUDE.md), and the [Stage 3 retrospective](docs/lld/stages/03-feature-assembler.md).

## Worked example: linear baseline + evaluation harness (Stage 4)

`notebooks/04_linear_baseline.ipynb` fits a seasonal-naive floor and a statsmodels OLS weather-only regression behind the shared five-member `Model` protocol, runs both through the fold-level `evaluate` harness, and scores the pair against the NESO day-ahead forecast on the period they overlap. The `python -m bristol_ml.train` CLI is the single-invocation equivalent: `uv run python -m bristol_ml.train` runs the default linear model; `uv run python -m bristol_ml.train model=naive` swaps via Hydra; `uv run python -m bristol_ml.train --help` lists every override. Linear loses cleanly to NESO — that's the pedagogical setup for Stage 5 calendar features. Notebook runs end-to-end in ~8 seconds on warm caches (D7). Entry points: `python -m bristol_ml.train --help`, `python -m bristol_ml.models.linear --help`, `python -m bristol_ml.models.naive --help`, `python -m bristol_ml.evaluation.metrics --help`, `python -m bristol_ml.evaluation.harness --help`, `python -m bristol_ml.evaluation.benchmarks --help`, `python -m bristol_ml.ingestion.neso_forecast --help`. See [`src/bristol_ml/models/CLAUDE.md`](src/bristol_ml/models/CLAUDE.md), [`docs/architecture/layers/models.md`](docs/architecture/layers/models.md), and the [Stage 4 retrospective](docs/lld/stages/04-linear-baseline.md).

## Worked example: enhanced evaluation & visualisation (Stage 6)

`notebooks/06_enhanced_evaluation.ipynb` is the standalone demo for the `bristol_ml.evaluation.plots` helper library — applied **side-by-side to both** the Stage 4 weather-only OLS and the Stage 5 weather + calendar OLS so the calendar uplift is visible on every diagnostic surface, not just the metric table. Seven cells exercise all seven public helpers: a 2×2 hero grid per OLS variant (`residuals_vs_time`, `predicted_vs_actual`, `acf_residuals` with daily/weekly markers, `error_heatmap_hour_weekday`), a four-line `forecast_overlay` (naive + both OLS + NESO) on a 48-hour window, a stacked-panel `forecast_overlay_with_band` rendering the empirical q10–q90 band for each OLS variant, and a stacked-panel `benchmark_holdout_bar` comparing each variant + the naive floor against the NESO day-ahead on a rolling weekly-fold holdout window. The bar-chart helper rolls within the holdout in 168-hour folds by default (`fold_len_hours=168`) so seasonal-naive `same_hour_last_week` works without manual splitter twiddling; pass `fold_len_hours=None` for the legacy single-fold behaviour. All plots use a colourblind-safe Okabe-Ito palette (12×8 default figsize for projector legibility) injected into `plt.rcParams` at import time; `ax=` passthrough lets facilitators compose helpers into arbitrary subplot grids. Both feature-table caches must be warm; regenerate via `uv run python -m bristol_ml.features.assembler features={weather_only,weather_calendar} --cache offline`. Entry point: `python -m bristol_ml.evaluation.plots --help` (prints the seven helper names, active palette, and rcParams overrides). See [`src/bristol_ml/evaluation/CLAUDE.md`](src/bristol_ml/evaluation/CLAUDE.md), [`docs/architecture/layers/evaluation.md`](docs/architecture/layers/evaluation.md), and the [Stage 6 retrospective](docs/lld/stages/06-enhanced-evaluation.md).

## Worked example: SARIMAX — dual-seasonality (Stage 7)

`notebooks/07_sarimax.ipynb` adds a `statsmodels` SARIMAX model behind the Stage 4 `Model` protocol. The central design choice is Dynamic Harmonic Regression (Hyndman fpp3 §12.1): `seasonal_order=(1,1,1,24)` absorbs the daily cycle inside SARIMAX; the weekly period (168 h) is handled by three Fourier exogenous regressor pairs appended by `bristol_ml.features.fourier.append_weekly_fourier`. The notebook's narrative payoff is Cell 9: the lag-168 ACF spike visible on the linear-baseline residuals is materially flatter on the SARIMAX residuals, demonstrating that weekly seasonality is now absorbed by the model. SARIMAX wins on all four metrics at the notebook's six-fold demo window (MAE 1730 MW vs 1955 linear vs 2080 naive). Entry points: `python -m bristol_ml.models.sarimax --help` (prints `SarimaxConfig` schema), `python -m bristol_ml.features.fourier --help` (weekly Fourier helper), and `uv run python -m bristol_ml.train model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168` for the full rolling-origin CLI run. See [`src/bristol_ml/models/CLAUDE.md`](src/bristol_ml/models/CLAUDE.md), [`docs/architecture/layers/models.md`](docs/architecture/layers/models.md), and the [Stage 7 retrospective](docs/lld/stages/07-sarimax.md).

## Worked example: SciPy parametric load model (Stage 8)

`notebooks/08_scipy_parametric.ipynb` fits a hand-specified load model behind the Stage 4 `Model` protocol using `scipy.optimize.curve_fit`. The functional form is a piecewise-linear temperature response (`α + β_heat · HDD + β_cool · CDD` with Elexon-convention hinges `T_heat = 15.5 °C`, `T_cool = 22.0 °C`) combined with three diurnal and two weekly Fourier harmonic pairs. The pedagogical payoff is Cell 7's parameter table — "`β_heat = X ± Y MW/°C`" — alongside Cell 8's fitted temperature-response curve overlaid on the raw scatter. Cell 12 is a dedicated assumptions appendix explaining what the Gaussian confidence intervals require (homoscedasticity, near-linearity around the optimum, no parameter at a bound) and naming Stage 10 as the owner of bootstrap / quantile alternatives. Entry points: `python -m bristol_ml.models.scipy_parametric --help` (prints `ScipyParametricConfig` defaults and a `curve_fit` pointer), and `uv run python -m bristol_ml.train model=scipy_parametric features=weather_calendar` for the full rolling-origin CLI run. See [`src/bristol_ml/models/CLAUDE.md`](src/bristol_ml/models/CLAUDE.md), [`docs/architecture/layers/models.md`](docs/architecture/layers/models.md), and the [Stage 8 retrospective](docs/lld/stages/08-scipy-parametric.md).

## Worked example: model registry — leaderboard (Stage 9)

`python -m bristol_ml.registry list` is the Stage 9 Demo moment: a single command prints every registered model with its MAE, RMSE, feature set, and training timestamp, turning "which model is best" into one terminal invocation rather than bespoke notebook code. Every model family from Stages 4, 7, and 8 saves through the registry via the shared `Model` protocol — no change to any model class body (intent AC-2).

```bash
uv run python -m bristol_ml.train model=sarimax      # trains + registers automatically
uv run python -m bristol_ml.registry list            # leaderboard sorted by MAE
uv run python -m bristol_ml.registry list --target nd_mw --model-type linear
uv run python -m bristol_ml.registry describe <run_id>
uv run python -m bristol_ml.train model=naive --no-register   # skip registration
```

The registry is filesystem-backed (`data/registry/`, gitignored); run IDs are human-typeable (`{model_name}_{YYYYMMDDTHHMM}`). The public surface is capped at four verbs — `save`, `load`, `list_runs`, `describe` — and enforced by a structural test. `mlflow` is a `dev`-group dependency only; a test-only PyFunc adapter (`tests/integration/mlflow_adapter.py`) proves the "mechanical migration to a hosted registry" claim is falsifiable. See [`src/bristol_ml/registry/CLAUDE.md`](src/bristol_ml/registry/CLAUDE.md), [`docs/architecture/layers/registry.md`](docs/architecture/layers/registry.md), and the [Stage 9 retrospective](docs/lld/stages/09-model-registry.md).

## Worked example: simple neural network (Stage 10)

`notebooks/10-simple-nn.ipynb` fits a small PyTorch MLP (`NnMlpModel` — 1 hidden layer × 128 units, ReLU, Adam) behind the Stage 4 `Model` protocol. The notebook's demo moments are: a live loss-curve cell that redraws per epoch as `fit()` runs (using the `epoch_callback` seam and `IPython.display`); a static `plots.loss_curve(model.loss_history_)` render after training completes; and a three-way `harness.evaluate` comparison (naive + linear + nn_mlp) on a narrow rolling-origin window so the NN can be placed on the leaderboard from the first session. Training is reproducible on CPU (`seed=0` gives bit-identical `state_dict` tensors across runs). The artefact is saved as a skops file (Stage 12 D10 migrated all families from joblib to `skops.io`) containing the `state_dict` bytes alongside the config and scaler buffers, registered through the Stage 9 registry. Entry points: `python -m bristol_ml.models.nn --help`, `python -m bristol_ml.models.nn.mlp --help`, and `uv run python -m bristol_ml.train model=nn_mlp`. See [`src/bristol_ml/models/nn/CLAUDE.md`](src/bristol_ml/models/nn/CLAUDE.md), [`docs/architecture/layers/models-nn.md`](docs/architecture/layers/models-nn.md), and the [Stage 10 retrospective](docs/lld/stages/10-simple-nn.md).

## Worked example: complex neural network — TCN (Stage 11)

`notebooks/11-complex-nn.ipynb` fits a Bai-et-al.-2018 Temporal Convolutional Network (`NnTemporalModel` — dilated causal 1D convolutions, residual blocks, weight-norm, `seq_len=168` weekly anchor) behind the Stage 4 `Model` protocol and registers it through the Stage 9 registry. The demo moment is a **six-row ablation table** (Cell 6) placing every shipped model family — naive, linear, SARIMAX, scipy parametric, MLP, TCN — head-to-head on the same single-holdout slice, all loaded predict-only via `registry.load`. Training uses the shared `_training.run_training_loop` extracted from Stage 10; the live loss-curve callback is the same Stage 10 `epoch_callback` seam. Entry points: `python -m bristol_ml.models.nn.temporal --help` (prints `NnTemporalConfig` schema), `python -m bristol_ml.models.nn --help`, and `uv run python -m bristol_ml.train model=nn_temporal`. See [`src/bristol_ml/models/nn/CLAUDE.md`](src/bristol_ml/models/nn/CLAUDE.md), [`docs/architecture/layers/models-nn.md`](docs/architecture/layers/models-nn.md), and the [Stage 11 retrospective](docs/lld/stages/11-complex-nn.md).

## Worked example: serving (Stage 12)

`python -m bristol_ml.serving` is the Stage 12 demo moment: a single command starts a FastAPI prediction service that loads the lowest-MAE registered model and answers forecast requests over HTTP.

```bash
# Start the service (requires at least one trained model in data/registry/)
uv run python -m bristol_ml.serving --registry-dir data/registry

# Check which model is being served
curl http://localhost:8000/

# Issue a prediction (replace the feature values with the keys in GET / feature_columns)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"target_dt": "2025-06-15T13:00:00Z", "features": {"temp_c": 12.5, "cloud_cover": 0.4}}'

# Inspect the auto-generated OpenAPI schema
curl http://localhost:8000/openapi.json
```

All six model families (`naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) are served through the same endpoint — no model-family special-casing. The default model is the lowest-MAE registered run across all families. Non-default runs are named via the optional `run_id` field in the request body and lazy-loaded on first use.

**Breaking change — skops migration.** Stage 12 migrated all six model families' save / load paths from `joblib` to `skops.io` for security (the serving layer is a network-facing deserialiser; `joblib.load` on an attacker-controlled artefact is RCE). **Any existing `data/registry/` directory containing `model.joblib` artefacts must be rebuilt by retraining.** `registry.load` rejects pre-Stage-12 artefacts with a clear error naming the path and the required migration action.

See [`src/bristol_ml/serving/CLAUDE.md`](src/bristol_ml/serving/CLAUDE.md), [`docs/architecture/layers/serving.md`](docs/architecture/layers/serving.md), and the [Stage 12 retrospective](docs/lld/stages/12-serving.md).

## Worked example: LLM feature extractor (Stage 14)

`notebooks/14_llm_extractor.ipynb` is the demo surface for the project's first authenticated outbound dependency: an LLM that reads REMIT free-text messages and returns a structured feature row. The architectural lesson is **stub-first discipline** — every CI run, every notebook bootstrap, and every test runs an offline `StubExtractor` backed by a 76-record hand-labelled gold set; no API key, no network call. The pedagogical lesson is the **side-by-side accuracy report**: `python -m bristol_ml.llm.evaluate` runs both implementations over the gold set and prints a per-field exact-match column next to a tolerance column (±5 MW capacity, ±1 h timestamps), showing where the LLM gets it wrong and *how the metric choice itself changes the answer* (intent line 41 — "the metric choice is a lesson itself"). Switching to the live OpenAI path is one Hydra override (`llm.type=openai`) plus an `BRISTOL_ML_LLM_API_KEY` export (see [API keys](#api-keys-and-offline-by-default) above). The integration test against the live path replays a recorded VCR cassette so CI never spends tokens. Public surface: a two-method `Extractor` Protocol (`extract` + `extract_batch`), a Pydantic `ExtractionResult` carrying provenance (`prompt_hash` + `model_id` — first 12 hex chars of SHA-256 of the prompt-file bytes), and a `build_extractor(config)` factory that triple-gates the live path (config discriminator + env-var override + API-key gate). Entry points: `python -m bristol_ml.llm.extractor --help`, `python -m bristol_ml.llm.evaluate --help`. See [`src/bristol_ml/llm/CLAUDE.md`](src/bristol_ml/llm/CLAUDE.md), [`docs/architecture/layers/llm.md`](docs/architecture/layers/llm.md), and the [Stage 14 retrospective](docs/lld/stages/14-llm-extractor.md).

## Worked example: embedding index over REMIT (Stage 15)

`notebooks/15_embedding_index.ipynb` is the demo surface for the project's first vector boundary. The eight-cell notebook synthesises an embeddable text per REMIT message (concatenating fuel type, capacity, valid-from / valid-to, and the optional free-text body — see [the layer doc](docs/architecture/layers/embeddings.md) §"NULL message_description synthesis"), embeds the corpus through `bristol_ml.embeddings.embed_corpus`, persists the result in a content-addressed Parquet cache, runs a top-k nearest-neighbour query, and projects the corpus into 2D via UMAP for a coloured scatter that lets a facilitator point at a cluster and say "that's planned nuclear maintenance." The architectural lesson is **two swappable Protocols** — `Embedder` and `VectorIndex` are independent `runtime_checkable` `typing.Protocol` types, so a future RAG stage can replace `NumpyIndex` with FAISS / Qdrant / a remote vector DB without touching the embedder, and a quantised-model experiment can swap the embedder without touching the index (see [ADR-0008](docs/architecture/decisions/0008-embedding-index-protocol.md)).

The triple-gated stub-first dispatch is the same shape as Stage 14: config discriminator (`embedding.type=stub` is the YAML default), env-var override (`BRISTOL_ML_EMBEDDING_STUB=1`), and a model-availability check (`HF_HUB_OFFLINE=1` plus a missing local snapshot routes back to the stub with a `loguru` WARNING). Cache invalidation is content-addressed: a SHA-256 of the corpus bytes plus the embedder's `model_id` is stored in the cache Parquet's `custom_metadata`; mismatch on either field rebuilds.

```bash
# Standalone CLI — five-row synthetic corpus, top-3 query against "planned nuclear outage"
uv run python -m bristol_ml.embeddings

# Switch to the live ModernBERT path (requires the pre-warm above)
uv run python -m bristol_ml.embeddings embedding.type=sentence_transformers
```

The notebook runs end-to-end in approximately 9 seconds under `BRISTOL_ML_EMBEDDING_STUB=1` and ~45 seconds the first time on the live path (CPU). Entry points: `python -m bristol_ml.embeddings --help`. See [`src/bristol_ml/embeddings/CLAUDE.md`](src/bristol_ml/embeddings/CLAUDE.md), [`docs/architecture/layers/embeddings.md`](docs/architecture/layers/embeddings.md), and the [Stage 15 retrospective](docs/lld/stages/15-embedding-index.md).

## Worked example: REMIT bi-temporal ingestion (Stage 13)

`notebooks/13_remit_ingestion.ipynb` introduces the project's first **bi-temporal** ingester: REMIT messages from the Elexon Insights API (no auth, public). REMIT is an event stream — every message carries three logically distinct times (`published_at` for transaction-time, `effective_from` / `effective_to` for valid-time) plus the standard `retrieved_at_utc` provenance scalar. Storage is append-only at `(mrid, revision_number)` grain; the "active state as known to the market at time `t`" view is a query, not a storage shape. The new public primitive is `bristol_ml.ingestion.remit.as_of(df, t)` — three lines that filter `published_at <= t`, take the max-revision row per mRID, and drop withdrawn messages. The notebook's demo moment is a stacked-area chart of declared unavailable capacity by fuel type, month-over-month: a facilitator can point at a spike and say "that was a nuclear unit going offline on this date." Entry points: `python -m bristol_ml.ingestion.remit --help` (prints the resolved `RemitIngestionConfig` schema; `BRISTOL_ML_REMIT_STUB=1` routes fetch through an in-memory fixture for offline runs). See [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md), [`docs/architecture/layers/ingestion.md`](docs/architecture/layers/ingestion.md) (§"Bi-temporal storage shape"), and the [Stage 13 retrospective](docs/lld/stages/13-remit-ingestion.md).

## Worked example: calendar features (Stage 5)

`notebooks/05_calendar_features.ipynb` adds a 44-column calendar feature block — hour-of-day, day-of-week, and month-of-year one-hots plus `is_bank_holiday_ew` / `is_bank_holiday_sco` / `is_day_before_holiday` / `is_day_after_holiday` flags — on top of the Stage 3 weather feature table. The same `LinearModel` class runs twice: once with the 5-column `weather_only` feature set and once with the 49-column `weather_calendar` feature set. The calendar extension closes much of the gap to the NESO benchmark on MAPE (quantified in the notebook) and visibly flattens the weekly residual ripple. Swap feature sets on the CLI via Hydra group-override: `uv run python -m bristol_ml.train` (weather only; default) or `uv run python -m bristol_ml.train features=weather_calendar`. Bank-holiday data comes from [gov.uk](https://www.gov.uk/bank-holidays.json) (OGL v3); first run warms the local cache, subsequent runs work offline. Entry points: `python -m bristol_ml.ingestion.holidays --help`, `python -m bristol_ml.features.calendar --help`, and the extended `python -m bristol_ml.features.assembler --help`. See [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md) and the [Stage 5 retrospective](docs/lld/stages/05-calendar-features.md).
