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

## What is this?

- A pedagogical reference repo for a generic ML / data-science architecture.
- The worked example is GB national electricity demand, day-ahead.
- [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md) has the full scope. Each stage lands with a demoable artefact.

See [`CLAUDE.md`](CLAUDE.md) for Claude Code guidance, [`docs/architecture/`](docs/architecture/) for architectural decisions, and [`docs/lld/stages/`](docs/lld/stages/) for stage retrospectives.

## Worked example: NESO demand (Stage 1)

`notebooks/01_neso_demand.ipynb` loads cached GB half-hourly demand (sourced from the NESO Data Portal under OGL v3), resamples to hourly, and plots a representative week plus daily peaks across the cached window. The first run populates the local cache from the NESO CKAN API; subsequent runs work offline. See [`src/bristol_ml/ingestion/CLAUDE.md`](src/bristol_ml/ingestion/CLAUDE.md) for the on-disk parquet schema, and the [Stage 1 retrospective](docs/lld/stages/01-neso-demand-ingestion.md) for the design choices.

## Worked example: weather + demand (Stage 2)

`notebooks/02_weather_demand.ipynb` primes a second ingestion cache from the [Open-Meteo](https://open-meteo.com) historical archive (CC BY 4.0), composes ten UK population centres into a population-weighted national temperature signal via `bristol_ml.features.weather.national_aggregate`, joins onto hourly demand, and plots the non-linear temperature-vs-demand curve with a LOWESS overlay. The curve is a **hockey-stick** on historical GB data — strong cold arm, flat warm arm — not a symmetric V; the hot-side response is only beginning to emerge post-2020 as residential cooling grows.

## Worked example: feature assembler + rolling-origin folds (Stage 3)

`notebooks/03_feature_assembler.ipynb` assembles the first end-to-end hourly feature table — demand joined with the population-weighted national weather aggregate — and enumerates rolling-origin train/test folds against it. The notebook calls `assembler.assemble(cfg, cache="auto")` to write `data/features/weather_only.parquet`, round-trips it through `assembler.load`, then plots fold boundaries overlaid on the GB demand series. Runs end-to-end in approximately 6 seconds on warm caches (D7). Entry points: `python -m bristol_ml.features.assembler --help` and `python -m bristol_ml.evaluation.splitter --help`. See [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md), [`src/bristol_ml/evaluation/CLAUDE.md`](src/bristol_ml/evaluation/CLAUDE.md), and the [Stage 3 retrospective](docs/lld/stages/03-feature-assembler.md).

## Worked example: linear baseline + evaluation harness (Stage 4)

`notebooks/04_linear_baseline.ipynb` fits a seasonal-naive floor and a statsmodels OLS weather-only regression behind the shared five-member `Model` protocol, runs both through the fold-level `evaluate` harness, and scores the pair against the NESO day-ahead forecast on the period they overlap. The `python -m bristol_ml.train` CLI is the single-invocation equivalent: `uv run python -m bristol_ml.train` runs the default linear model; `uv run python -m bristol_ml.train model=naive` swaps via Hydra; `uv run python -m bristol_ml.train --help` lists every override. Linear loses cleanly to NESO — that's the pedagogical setup for Stage 5 calendar features. Notebook runs end-to-end in ~8 seconds on warm caches (D7). Entry points: `python -m bristol_ml.train --help`, `python -m bristol_ml.models.linear --help`, `python -m bristol_ml.models.naive --help`, `python -m bristol_ml.evaluation.metrics --help`, `python -m bristol_ml.evaluation.harness --help`, `python -m bristol_ml.evaluation.benchmarks --help`, `python -m bristol_ml.ingestion.neso_forecast --help`. See [`src/bristol_ml/models/CLAUDE.md`](src/bristol_ml/models/CLAUDE.md), [`docs/architecture/layers/models.md`](docs/architecture/layers/models.md), and the [Stage 4 retrospective](docs/lld/stages/04-linear-baseline.md).

## Worked example: enhanced evaluation & visualisation (Stage 6)

`notebooks/04_linear_baseline.ipynb` gains a Stage 6 appendix (Cells 13–18) that demonstrates the new `bristol_ml.evaluation.plots` helper library. Six cells exercise the seven public helpers: a 2×2 hero grid (`residuals_vs_time`, `predicted_vs_actual`, `acf_residuals` with daily/weekly markers, `error_heatmap_hour_weekday`); a three-line `forecast_overlay` call; an empirical q10–q90 uncertainty-band chart via `forecast_overlay_with_band` (driven by `evaluate(..., return_predictions=True)`); and a `benchmark_holdout_bar` grouped bar chart comparing models against the NESO day-ahead benchmark on a fixed holdout window. Stage 4 Cells 0–12 are byte-identical to the Stage 4 shipping view — the appendix is additive only. All plots use a colourblind-safe Okabe-Ito palette (12×8 default figsize for projector legibility) injected into `plt.rcParams` at import time; `ax=` passthrough lets facilitators compose helpers into arbitrary subplot grids. Entry point: `python -m bristol_ml.evaluation.plots --help` (prints the seven helper names, active palette, and rcParams overrides). See [`src/bristol_ml/evaluation/CLAUDE.md`](src/bristol_ml/evaluation/CLAUDE.md), [`docs/architecture/layers/evaluation.md`](docs/architecture/layers/evaluation.md), and the [Stage 6 retrospective](docs/lld/stages/06-enhanced-evaluation.md).

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

`notebooks/10-simple-nn.ipynb` fits a small PyTorch MLP (`NnMlpModel` — 1 hidden layer × 128 units, ReLU, Adam) behind the Stage 4 `Model` protocol. The notebook's demo moments are: a live loss-curve cell that redraws per epoch as `fit()` runs (using the `epoch_callback` seam and `IPython.display`); a static `plots.loss_curve(model.loss_history_)` render after training completes; and a three-way `harness.evaluate` comparison (naive + linear + nn_mlp) on a narrow rolling-origin window so the NN can be placed on the leaderboard from the first session. Training is reproducible on CPU (`seed=0` gives bit-identical `state_dict` tensors across runs). The artefact is saved as a single joblib file containing the `state_dict` bytes alongside the config and scaler buffers — compatible with the Stage 9 registry unchanged (no registry changes required). Entry points: `python -m bristol_ml.models.nn --help`, `python -m bristol_ml.models.nn.mlp --help`, and `uv run python -m bristol_ml.train model=nn_mlp`. See [`src/bristol_ml/models/nn/CLAUDE.md`](src/bristol_ml/models/nn/CLAUDE.md), [`docs/architecture/layers/models-nn.md`](docs/architecture/layers/models-nn.md), and the [Stage 10 retrospective](docs/lld/stages/10-simple-nn.md).

## Worked example: calendar features (Stage 5)

`notebooks/05_calendar_features.ipynb` adds a 44-column calendar feature block — hour-of-day, day-of-week, and month-of-year one-hots plus `is_bank_holiday_ew` / `is_bank_holiday_sco` / `is_day_before_holiday` / `is_day_after_holiday` flags — on top of the Stage 3 weather feature table. The same `LinearModel` class runs twice: once with the 5-column `weather_only` feature set and once with the 49-column `weather_calendar` feature set. The calendar extension closes much of the gap to the NESO benchmark on MAPE (quantified in the notebook) and visibly flattens the weekly residual ripple. Swap feature sets on the CLI via Hydra group-override: `uv run python -m bristol_ml.train` (weather only; default) or `uv run python -m bristol_ml.train features=weather_calendar`. Bank-holiday data comes from [gov.uk](https://www.gov.uk/bank-holidays.json) (OGL v3); first run warms the local cache, subsequent runs work offline. Entry points: `python -m bristol_ml.ingestion.holidays --help`, `python -m bristol_ml.features.calendar --help`, and the extended `python -m bristol_ml.features.assembler --help`. See [`src/bristol_ml/features/CLAUDE.md`](src/bristol_ml/features/CLAUDE.md) and the [Stage 5 retrospective](docs/lld/stages/05-calendar-features.md).
