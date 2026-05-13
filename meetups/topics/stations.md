# Station pick-list

The 14 notebooks under [`/notebooks/`](../../notebooks/) are the demo
surface. Each one is a thin wrapper (it calls into `src/bristol_ml/`,
it doesn't reimplement) producing a visible artefact: a plot, a metric
table, a leaderboard row, an API response, a search hit.

Pull 2–5 into a session, grouped by theme. Every station should run
offline-by-default. Live paths (OpenAI, ModernBERT) are explicit
opt-ins; the offline stub is the default for both CI and meetups
without API keys.

For the full design rationale behind a station, follow the
"Retrospective" link to its `docs/lld/stages/` write-up.

## Theme: data, joined and visible

Good entry-point stations. Low-ceremony; the demo moment is "real GB
data on a plot in five minutes."

| # | Notebook | What you'll see | Time | Retrospective |
|---|----------|-----------------|------|---------------|
| 1 | [`01_neso_demand.ipynb`](../../notebooks/01_neso_demand.ipynb) | Half-hourly GB demand resampled hourly; a representative week plus daily peaks. First run populates the local cache from the NESO CKAN API; later runs are offline. | ~5 min | [Stage 1](../../docs/lld/stages/01-neso-demand-ingestion.md) |
| 2 | [`02_weather_demand.ipynb`](../../notebooks/02_weather_demand.ipynb) | Open-Meteo historical archive composed into a population-weighted national temperature, joined on demand; the non-linear "hockey stick" temperature-vs-demand curve with a LOWESS overlay. | ~5 min | [Stage 2](../../docs/lld/stages/02-weather-ingestion.md) |
| 3 | [`03_feature_assembler.ipynb`](../../notebooks/03_feature_assembler.ipynb) | First end-to-end feature table; rolling-origin train/test folds plotted over the demand series. | ~6 s on warm caches | [Stage 3](../../docs/lld/stages/03-feature-assembler.md) |

## Theme: classical forecasting baselines

Good middle-of-session stations. Pair any of these with the calendar
features and the evaluation stations for a self-contained "build a
forecaster" arc.

| # | Notebook | What you'll see | Time | Retrospective |
|---|----------|-----------------|------|---------------|
| 4 | [`04_linear_baseline.ipynb`](../../notebooks/04_linear_baseline.ipynb) | Seasonal-naive floor + statsmodels OLS fit through the shared `Model` protocol; both scored against the NESO day-ahead. Linear loses cleanly — that's the pedagogical setup for the calendar station. | ~8 s on warm caches | [Stage 4](../../docs/lld/stages/04-linear-baseline.md) |
| 5 | [`05_calendar_features.ipynb`](../../notebooks/05_calendar_features.ipynb) | Adds 44 calendar columns (hour-of-day, day-of-week, month, bank-holiday flags) on top of weather; the same `LinearModel` class runs twice so the uplift is measurable. | ~10 s | [Stage 5](../../docs/lld/stages/05-calendar-features.md) |
| 6 | [`06_enhanced_evaluation.ipynb`](../../notebooks/06_enhanced_evaluation.ipynb) | The full diagnostic toolkit (residuals vs time, predicted vs actual, ACF, hour-by-weekday heatmap, forecast overlay with band, holdout-bar against NESO) applied side-by-side to weather-only and weather+calendar. | ~30 s | [Stage 6](../../docs/lld/stages/06-enhanced-evaluation.md) |
| 7 | [`07_sarimax.ipynb`](../../notebooks/07_sarimax.ipynb) | SARIMAX with dual seasonality (daily inside the model; weekly via Fourier exogenous regressors). The demo moment: the lag-168 ACF spike on the linear residuals disappears. | ~minutes (CPU) | [Stage 7](../../docs/lld/stages/07-sarimax.md) |
| 8 | [`08_scipy_parametric.ipynb`](../../notebooks/08_scipy_parametric.ipynb) | A hand-specified piecewise-linear temperature response fit with `scipy.optimize.curve_fit`; the parameter table ("β_heat = X ± Y MW/°C") is the moment. Includes an assumptions appendix. | ~seconds | [Stage 8](../../docs/lld/stages/08-scipy-parametric.md) |

## Theme: neural networks behind the same protocol

Two stations that drop deep models behind exactly the same `Model`
interface as Stages 4/7/8 — the leaderboard is now six families wide.

| # | Notebook | What you'll see | Time | Retrospective |
|---|----------|-----------------|------|---------------|
| 9 | [`10-simple-nn.ipynb`](../../notebooks/10-simple-nn.ipynb) | A 1×128 PyTorch MLP; loss curve redraws live per epoch via the `epoch_callback` seam; bit-identical CPU runs at `seed=0`. | ~minutes (CPU) | [Stage 10](../../docs/lld/stages/10-simple-nn.md) |
| 10 | [`11-complex-nn.ipynb`](../../notebooks/11-complex-nn.ipynb) | A Bai-2018 Temporal Convolutional Network (dilated causal conv, residual blocks, weight-norm, `seq_len=168`). Demo moment: a six-row ablation table across all model families, all loaded predict-only via the registry. | ~minutes (CPU) | [Stage 11](../../docs/lld/stages/11-complex-nn.md) |

## Theme: serving, registries, and operating models

Where the project starts to look like a production stack.

| # | Notebook / CLI | What you'll see | Time | Retrospective |
|---|----------------|-----------------|------|---------------|
| 11 | `python -m bristol_ml.registry list` | One command prints every registered model with MAE, RMSE, feature set, training timestamp. Four-verb public surface (`save`, `load`, `list_runs`, `describe`). | ~seconds | [Stage 9](../../docs/lld/stages/09-model-registry.md) |
| 12 | `python -m bristol_ml.serving` | A FastAPI service that loads the lowest-MAE model and answers forecast requests over HTTP. `curl /predict` from another terminal is the demo. **All six model families served behind one endpoint.** | ~seconds to start | [Stage 12](../../docs/lld/stages/12-serving.md) |

## Theme: events, LLMs, and embeddings

The last block in the existing arc — bi-temporal event storage, an
authenticated LLM dependency (with a stub-first discipline), and a
vector index over the same events.

| # | Notebook | What you'll see | Time | Retrospective |
|---|----------|-----------------|------|---------------|
| 13 | [`13_remit_ingestion.ipynb`](../../notebooks/13_remit_ingestion.ipynb) | The project's first **bi-temporal** ingester: every REMIT message carries `published_at`, `effective_from`, `effective_to`. The `as_of(df, t)` helper is three lines. Demo moment: stacked-area chart of declared unavailable capacity by fuel type. | ~10 s | [Stage 13](../../docs/lld/stages/13-remit-ingestion.md) |
| 14 | [`14_llm_extractor.ipynb`](../../notebooks/14_llm_extractor.ipynb) | An LLM that reads REMIT free-text and returns a structured row. Side-by-side accuracy report (stub vs OpenAI) on a 76-record hand-labelled gold set; the *metric choice itself* changes the answer. | ~seconds (stub) | [Stage 14](../../docs/lld/stages/14-llm-extractor.md) |
| 15 | [`15_embedding_index.ipynb`](../../notebooks/15_embedding_index.ipynb) | Embed REMIT messages, persist as content-addressed Parquet, run top-k nearest neighbour, project to 2D with UMAP. Point at a cluster and say "that's planned nuclear maintenance." | ~9 s (stub), ~45 s (live first run) | [Stage 15](../../docs/lld/stages/15-embedding-index.md) |
| 16 | [`04_remit_ablation.ipynb`](../../notebooks/04_remit_ablation.ipynb) | A late-arriving ablation: does adding REMIT-derived features to the linear baseline measurably help? Honest answer on a held-out window. | ~seconds | [Stage 16](../../docs/lld/stages/16-model-with-remit.md) |

## Theme: how the project itself was built

Always available as a side-station; see
[`claude-code-as-a-team.md`](claude-code-as-a-team.md) for the full
walkthrough. Ten-minute slot; pairs well with any of the above.

## Picking a session

A complete community session typically combines:

- **One on-ramp station** from the first theme (everyone has something
  on screen in five minutes).
- **One or two middle stations** from the classical or neural themes
  (mixed-ability attendees pair up at their preferred depth).
- **One stretch station** from serving / events / LLMs (a place to
  point at "this is what the production stack starts to look like").
- **A ten-minute Claude Code corner** (always available; never
  consumes a main station slot).

Switch the order, drop a theme, invent a new station. The repo is the
sandbox.
