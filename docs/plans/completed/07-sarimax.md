# Plan — Stage 7: SARIMAX

**Status:** `approved` — human markup 2026-04-21: D1–D12 all ACCEPTED verbatim; H-1, H-2, H-3 all accepted as tabled. Phase 2 ready.
**Intent:** [`docs/intent/07-sarimax.md`](../../intent/07-sarimax.md)
**Upstream stages shipped:** Stages 0–6 (foundation, NESO demand, weather, feature assembler + splitter, linear baseline + evaluation harness + three-way NESO benchmark, calendar features, enhanced-evaluation diagnostic-plot library).
**Downstream consumers:** Stage 8 (explicit functional-form extensions), Stage 10 (quantile/probabilistic framing — where SARIMAX's analytical CI becomes relevant), Stage 11 (further model families). Stage 9 (registry) inherits the SARIMAX save/load artefact for cross-version-compatibility work.
**Baseline SHA:** `8434be3` (tip of `main` at plan time — Stage 6 merged via PR #3).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/07-sarimax-requirements.md`](../../lld/research/07-sarimax-requirements.md)
- Codebase map — [`docs/lld/research/07-sarimax-codebase.md`](../../lld/research/07-sarimax-codebase.md)
- External research — [`docs/lld/research/07-sarimax-domain.md`](../../lld/research/07-sarimax-domain.md)

**Pedagogical weight.** Intent §Demo moment frames Stage 7 as "seasonal decomposition makes seasonal structure visible in a way the linear model's coefficients don't". The stage's narrative inheritance is load-bearing: Stage 6 shipped an ACF plot with a reference marker at lag 168 precisely so Stage 7 could say "that weekly spike is what SARIMAX is designed to eat". Shipping a SARIMAX that ignores the weekly period would contradict that setup. Equally, shipping a SARIMAX that blows the notebook budget (as a naive `s=168` configuration would — external research §R2/§R3) contradicts AC-3 ("reasonable time") and kills the live-demo surface. The central design question of this stage is **how to encode the weekly seasonality without setting `s=168` in the `seasonal_order` tuple**; decisions D1–D4 all turn on that answer.

---

## 1. Decisions for the human (resolve before Phase 2)

Twelve decision points plus three housekeeping carry-overs. For each I propose a default that honours the simplicity bias + the research evidence, and cite the supporting source. Mark each `ACCEPT` / `OVERRIDE: <alt>` in your reply; I'll update the plan before Phase 2.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Dual-seasonality strategy (s=24 vs s=168 vs DHR) | **Dynamic harmonic regression (DHR): `seasonal_order=(P,D,Q,24)` for daily seasonality inside SARIMAX, weekly period 168 h absorbed via Fourier exogenous regressors `sin(2πkt/168), cos(2πkt/168)` for `k=1..3` (six extra columns).** Annual period left to weather + calendar exog. The notebook prose explains why `s=168` was rejected (the computational cost demonstration is itself the teaching moment). | Hyndman's *fpp3* §12.1 is explicit: "seasonal versions of ARIMA … are designed for shorter periods … for large m, the estimation becomes almost impossible"; DHR is his recommended route for m > 200 (practically, electricity hourly data with m=168). Naive `s=168` has community reports of 20 GB memory + hour-scale fits (§R2). Fourier is cheap, interpretable, and keeps AC-3 feasible. K=3 pairs gives six harmonics — enough for GB weekly shape after calendar exog has already absorbed the day-of-week contribution. | Domain §R2 + §R3 + A5; requirements OQ-2; Hyndman hyndsight blog. |
| **D2** | Order default (`order`, `seasonal_order`) | **`order=(1,0,1)` non-seasonal, `seasonal_order=(1,1,1,24)` daily-seasonal.** The notebook exhibits a small AIC sweep (`p,q ∈ {0,1}`, `P,Q ∈ {0,1}`) that justifies the pick, but the shipped config is this fixed choice. | Conservative textbook order (fpp3 §9, Godt 2023, Cassettari et al. 2018) that routinely fits well on hourly electricity. `D=1` takes a seasonal difference at lag 24 which removes the dominant diurnal level changes. `d=0` because the non-seasonal series is stationary after seasonal differencing (to be verified in the notebook via ADF). Sweep is a *pedagogical exercise*, not an *architectural search* — complies with intent out-of-scope (OQ-1). | Domain §R4; requirements OQ-1, OQ-8. |
| **D3** | Exogenous regressor set | **`temperature_2m` + Stage 5 calendar one-hots (44 columns) + three Fourier weekly pairs (6 columns) = 51 exog columns.** Dew point, wind, cloud, shortwave omitted at Stage 7. | Literature consensus (§R5; Cassettari 2018, Iatsyshyn, MDPI 2025) is temperature dominates; other weather channels add <1% after calendar features. Calendar one-hots are already in the Stage 5 feature table — no new ingestion. Six Fourier columns for weekly period keep design-matrix size modest while ensuring the SARIMAX residual ACF no longer has a lag-168 spike (the Stage 6 → Stage 7 narrative payoff). | Domain §R5 + A5; codebase map §4 (calendar features schema); requirements OQ-4. |
| **D4** | Training-window size & rolling-origin mechanics | **Fixed sliding window, `min_train_periods = 8760` (one year — the project default, no change), `test_len = 24` (project default, no change), `step = 168`, `fixed_window = true`.** Overrides applied as per-field Hydra CLI arguments (e.g. `evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168`); **no new `conf/evaluation/*.yaml` file is created**. Every fold re-fits (no `apply(refit=False)` shortcut — see D5). | Full-expanding-window SARIMAX over multi-year history blows AC-3's "reasonable time" budget (research §R3: fit time is roughly O(n²) in training length). Fixing the window at 8760 keeps per-fold fit under ~60s on laptop CPUs per the `s=24` + DHR + small exog estimates in domain §R3. `step=168` gives weekly folds, trimming the demo loop to ~52 folds/year (manageable live-demo length). Per-field CLI overrides are idiomatic Hydra — a new group file is unnecessary ceremony when only two fields differ from project defaults. | Requirements NFR-1, OQ-3; domain §R3, Q2; codebase map §3 (harness contract). |
| **D5** | `fit` vs `apply(refit=False)` at rolling-origin predict time | **Re-fit per fold (status quo).** Do NOT use `SARIMAXResults.apply(refit=False)` inside the harness. The `apply` mechanism lives inside `SarimaxModel.predict()` only as the no-op identity path (fitted results → prediction); fold-to-fold re-fits go through `fit()`. | `apply(refit=False)` would break the rolling-origin semantic that every stage has honoured since Stage 4 (parameters evolve with new data). Changing that semantic for SARIMAX alone turns cross-model comparison into apples vs oranges. The speed gain is real but comes at the cost of honest evaluation. Research flags the alternative (Q2) as a genuine open question; the default here is the conservative one. If fit-time budgets force a change, escalate — don't silently pick `apply()`. | Domain §R8, Q2; requirements NFR-1. |
| **D6** | SARIMAX constructor settings | **`enforce_stationarity=False, enforce_invertibility=False, concentrate_scale=True, simple_differencing=False, hamilton_representation=False, freq="h"`.** Passed to `statsmodels.tsa.statespace.SARIMAX` at construction time; stored in `SarimaxConfig` as a nested `sarimax_kwargs` block (frozen Pydantic model). | `enforce_*=False` is the statsmodels-maintainer-blessed setting for real-world seasonal demand data (domain A1 + §R1; PR #4739 relaxed the same check at the start-param path because users routinely hit it). `concentrate_scale=True` shrinks parameter vector by one and speeds optimisation (§R1, §R3). `simple_differencing=False` keeps Harvey representation (no lost observations, preserves full residual series for Stage 6 ACF helper — important for AC-4). `freq="h"` is required because the assembler does not set `df.index.freq` (codebase surprise 2). | Domain §R1, §R3, A1, A2; codebase map §4 (freq warning trap); requirements NFR-4. |
| **D7** | Confidence-interval surfacing | **Do NOT plot SARIMAX parametric CIs in the notebook. Display `results.plot_diagnostics()` output and the numeric summary stats (Ljung-Box Q, Jarque-Bera, heteroscedasticity) from `results.summary()` as the residual-diagnostic surface. Reserve parametric CIs for a one-sentence notebook comment pointing at Stage 10.** No new plot helper. | DESIGN §10 defers all probabilistic forecasting. The Stage 6 empirical-quantile band is the approved uncertainty surface for the project. Plotting SARIMAX's Gaussian-approximation CI alongside the Stage 6 empirical band would introduce two uncertainty conventions with different semantics before Stage 10 resolves the framing — confusing for learners. The teaching-moment argument (domain §R6) is weaker than the Chesterton's-fence argument for respecting DESIGN §10. Research genuine-open-question Q3 asks whether this is too conservative; default says "yes it's conservative, and that's correct". | Requirements OQ-5; domain §R6, Q3; intent §Points for consideration; DESIGN §10. |
| **D8** | Residual-diagnostic surface | **Reuse `bristol_ml.evaluation.plots.acf_residuals` (Stage 6) for the visual surface. In the notebook, also call `results.plot_diagnostics(figsize=(14, 10))` and print `results.summary()` for the numeric tests (Ljung-Box, Jarque-Bera, breakvar). Do NOT promote any new helper to `plots.py`.** | Stage 6's `acf_residuals` already renders the daily-and-weekly reference markers. The narrative pay-off is that SARIMAX residuals show a *flat* ACF around lag 168 where the linear baseline's residuals showed a spike — this is the "Stage 7 closes Stage 6's open question" moment, achievable with zero new helper code. `plot_diagnostics` is statsmodels-blessed (domain A3) and a one-liner in the notebook. Promoting ARIMA-specific helpers to `plots.py` is premature generalisation — not clear they reuse for Stage 10/11. | Domain §R7, A3; requirements OQ-6; codebase map §10. |
| **D9** | Notebook target | **New `notebooks/07_sarimax.ipynb`.** Follows the Stage 5 one-notebook-per-modelling-stage pattern. No modification to Stage 4 / Stage 5 / Stage 6 notebooks. | SARIMAX has its own narrative arc (decomposition → order selection → fit → diagnostics → cross-model comparison) that warrants a dedicated artefact. Stage 6's append-only addendum was a special case (diagnostic surface enhancing Stage 4's artefact). For a new model, Stage 4 and Stage 5 precedents apply. | Codebase map §8; requirements OQ-10; intent AC-4. |
| **D10** | Config wiring (Hydra discriminated union) | **Full wiring: `SarimaxConfig` added to `conf/_schemas.py` discriminated union alongside `NaiveConfig` / `LinearConfig`; new `conf/model/sarimax.yaml` with `# @package model` header and `type: sarimax`; both `_build_model_from_config` dispatchers (harness CLI at `harness.py:475–487` AND `train.py:220–242`) gain a `SarimaxConfig` branch. `conf/evaluation/rolling_origin_sarimax.yaml` provides the D4 window override; enabled per-run via `python -m bristol_ml.train model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168`.** | DESIGN §2.1.4: configuration lives outside code. Live-demo pattern `model=sarimax` at the CLI is the minimum bar for every modelling stage. Missing either `_build_model_from_config` branch is a silent exit-code-3 bug (codebase map surprise 3). No `cli.py` / `__main__.py` changes. | Codebase map §6 + surprise 3; requirements OQ-11, US-4. |
| **D11** | `SarimaxModel.predict()` contract vs the harness | **Hide all multi-step complexity inside `SarimaxModel.predict(features: pd.DataFrame) -> pd.Series`.** `predict` infers horizon length from `len(features)`, calls `self._results.get_forecast(steps=len(features), exog=features[feature_columns].to_numpy())`, re-indexes the returned series to `features.index`, and casts to `pd.Series(name=config.target_column)`. No harness change. The `predict()` signature matches `LinearModel` / `NaiveModel` exactly. Protocol is unchanged. | Codebase surprise 1: SARIMAX's native output does not carry the caller's `DatetimeIndex`; without re-indexing the harness's metric computation silently produces mismatched series. The fix is inside `predict()`, not in the harness. This honours the Stage 6 D9 debt note ("do not add a second flag to `evaluate()`"). Should the fix prove infeasible during implementation, the correct escalation path is not a harness extension but a protocol-level discussion — which must surface to the human, not be resolved in-branch. | Codebase map §3 + surprise 1; requirements OQ-9; Stage 6 D9 debt note. |
| **D12** | Dependency additions | **None.** `statsmodels>=0.14,<1` is already pinned; `SARIMAX` is in the same package. No `pmdarima`; no new top-level dependency. | DESIGN §2.2.4 simplicity bias. `pmdarima` adds ~40 MB for convenience wrappers that fit in ten lines of statsmodels. Auto-order search is out of scope anyway. The AIC sweep cell (D2) is a small `statsmodels.tsa.statespace.SARIMAX` loop. | Requirements OQ-12; domain §R4; codebase map §5. |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` layout tree — Stages 1–6 additions batched. Stage 7 adds `src/bristol_ml/models/sarimax.py`, `conf/model/sarimax.yaml`, `notebooks/07_sarimax.ipynb`. | **Flag for human-led batched §6 edit** covering Stages 1–7 at Stage 7 PR review. Lead MUST NOT touch §6 unilaterally. Stage 6 H-2 carried this forward; Stage 7 extends the batch. |
| **H-2** | Stage 6 retro "Next" pointer to Stage 7 — confirm wording is current. | **Verify at T8 hygiene.** Edit if it drifted from intent. |
| **H-3** | `docs/architecture/layers/models.md` open questions "Per-model CLI parity" (deferred to Stage 7 per models-layer doc:118) — decide or re-defer. | **Re-defer.** Stage 7 ships a minimal `python -m bristol_ml.models.sarimax --help` (prints config + decomposition-cell placeholder) matching the `naive.py`/`linear.py` convention. A cross-model CLI harmonisation pass is better owned by Stage 11 or a dedicated housekeeping stage when >3 model families co-exist. Record the defer in Stage 7 retro Deferred section. |

### Resolution log

- **Drafted 2026-04-21** — pre-human-markup. All decisions D1–D12 are proposed defaults.
- **Human markup 2026-04-21** — D1–D12 all accepted verbatim as proposed; H-1, H-2, H-3 accepted as tabled. No overrides. Phase 2 may proceed.

Load-bearing propagation notes (for post-markup updating):
- If **D1** is overridden to single-SARIMAX `s=168`: D2's `seasonal_order` changes to `(P,D,Q,168)`; D3 drops the Fourier columns; D4 must shrink the training window materially (likely to 4–8 weeks) to stay within AC-3; D5's re-fit-per-fold stance likely becomes infeasible and `apply(refit=False)` debate reopens.
- If **D4** is overridden (e.g. use full expanding window or shorter fixed window): NFR-1 budgets rescale; D5's fit-vs-apply question may reopen.
- If **D7** is overridden to plot parametric CIs: a new notebook-local helper is needed (not promoted to `plots.py`); T5 adds a cell and T7 adds a test for CI-rendering shape; a retro note on the DESIGN §10 interaction is required.
- If **D11** fails at implementation time (predict cannot re-index cleanly): escalate — do NOT extend the harness. The debt-trigger escape hatch is a protocol-level dataclass (`EvaluationResult`), not a second `evaluate()` flag (Stage 6 D9).

---

## 2. Scope

### In scope

- New `src/bristol_ml/models/sarimax.py` exporting `SarimaxModel` conforming to the Stage 4 `Model` protocol (`fit`, `predict`, `save`, `load`, `metadata`).
- New Pydantic `SarimaxConfig` in `conf/_schemas.py` with fields `type: Literal["sarimax"]`, `target_column`, `feature_columns` (None → weather-only + Fourier weekly by convention), `order: tuple[int,int,int]`, `seasonal_order: tuple[int,int,int,int]`, `trend: str | None`, `sarimax_kwargs: SarimaxKwargs` (nested frozen model holding `enforce_stationarity=False`, `enforce_invertibility=False`, `concentrate_scale=True`, `simple_differencing=False`, `hamilton_representation=False`). `SarimaxConfig` added to the discriminated `ModelConfig` union.
- New `conf/model/sarimax.yaml` Hydra group file with `# @package model` header and defaults aligned to `SarimaxConfig`.
- D4 splitter override applied via CLI/per-field Hydra arguments at runtime; no new `conf/evaluation/*.yaml` file.
- Extensions to `_build_model_from_config` in both `src/bristol_ml/evaluation/harness.py` and `src/bristol_ml/train.py` — each gains a `SarimaxConfig` branch. `SarimaxModel` is also added to the `src/bristol_ml/models/__init__.py` lazy re-export surface.
- Fourier weekly-harmonic helper in `src/bristol_ml/features/calendar.py` (or a new small module `features/fourier.py` — decide in T2) that appends `sin_week_k1, cos_week_k1, sin_week_k2, cos_week_k2, sin_week_k3, cos_week_k3` columns to the feature frame given a UTC datetime index. Deterministic, pure, no I/O.
- New `notebooks/07_sarimax.ipynb` covering: load features → decomposition → ADF stationarity → small AIC sweep (order selection narrative) → single-fold `SarimaxModel.fit` + `results.plot_diagnostics()` + `results.summary()` → rolling-origin evaluation via `evaluate(..., return_predictions=True)` → Stage 6 `acf_residuals` on SARIMAX residuals showing the lag-168 spike has been absorbed → `forecast_overlay` with `NaiveModel`, `LinearModel`, `SarimaxModel` on the same held-out window → markdown closing cell with Stage 8/10 hooks.
- Unit tests in `tests/unit/models/test_sarimax.py` covering: protocol conformance (AC-1, AC-5), fit → predict → save → load round-trip including exog (AC-2, guards issue #6542), re-entrant fit discarding prior state (NFR-5), `RuntimeError` on predict-before-fit, `predict()` returns `pd.Series` indexed to `features.index` (the codebase surprise 1 regression guard), exogenous column order honoured, metadata population (fit_utc tz-aware UTC, feature_columns tuple-order preserved, hyperparameters round-trip).
- Fourier-helper tests in `tests/unit/features/test_fourier.py` (or appended to `test_calendar.py` if co-located): column-name contract, deterministic values at a known timestamp, no clock-dependent behaviour, 168-period sanity (`t=0` and `t=168` yield identical values for all columns).
- Unit tests in `tests/unit/test_config.py` extensions for `SarimaxConfig`, `SarimaxKwargs`, discriminated-union dispatch on `type: "sarimax"`, Hydra override round-trip.
- Stage-hygiene updates in T8: `CHANGELOG.md`, `docs/lld/stages/07-sarimax.md` retrospective (new file), `docs/stages/README.md` Stage 7 row flip to `shipped`, `docs/architecture/layers/models.md` module inventory row + open-question close-outs, `src/bristol_ml/models/CLAUDE.md` surface documentation.

### Out of scope (do not accidentally implement)

From intent §Out of scope, explicit:
- Automatic order selection as an architectural feature (an `auto_order=True` flag on `SarimaxConfig`, an `auto_arima` wrapper, a grid-search CLI).
- Multivariate SARIMAX (VARMAX).
- State-space models beyond SARIMAX (Kalman filters, DLMs, structural time-series).

From intent §Out of scope, explicitly deferred:
- Bayesian time-series models.
- Vector autoregression.
- Probabilistic forecast evaluation (pinball loss, CRPS).

Also out of scope for this plan:
- Serving integration (Stage 12's concern).
- Registry integration (Stage 9's concern).
- Per-horizon evaluation metrics and multi-horizon splitter variants (evaluation-layer open question, deferred to the stage that actually needs it — Stage 10 or 11; intent explicitly says "needs to be happy with [SARIMAX's multi-step output]", not "needs to refactor the evaluator").
- New `plots.py` helpers for ARIMA-specific diagnostics (D8 — statsmodels `plot_diagnostics()` in the notebook is sufficient).
- Parametric SARIMAX CIs in plot form (D7).
- `pmdarima` or any other new dependency (D12).
- Per-model CLI harmonisation (H-3 defers to a future housekeeping stage).
- Any notebook outside `notebooks/07_sarimax.ipynb`. In particular: do NOT append to notebooks 04, 05, or 06-derived cells in 04.
- Any change to Stage 6's `plots.py` surface (the whole point of Stage 6 was that Stage 7 inherits it unchanged).
- Any change to the splitter, metrics, or benchmarks internals. D4 introduces a new Hydra override file only; the splitter code is unchanged.
- Any change to `src/bristol_ml/cli.py`, `src/bristol_ml/__main__.py`, or `src/bristol_ml/config.py`.

---

## 3. Reading order for the implementer

Read top-to-bottom before opening code:

1. `docs/intent/07-sarimax.md` — the spec. Where this plan disagrees, the spec wins.
2. `docs/lld/research/07-sarimax-requirements.md` — full acceptance-criteria evidence map, NFRs, open-question taxonomy.
3. `docs/lld/research/07-sarimax-codebase.md` — §1 (Model protocol contract), §2 (LinearModel template), §3 (harness predict call site — surprise 1), §4 (feature schema + freq trap — surprise 2), §5 (statsmodels import surface), §6 (config schema + `_build_model_from_config` dual sites — surprise 3), §7 (save/load), §8 (notebook conventions), §10 (Stage 6 integration).
4. `docs/lld/research/07-sarimax-domain.md` — §R1 (SARIMAX API + defaults), §R2 (dual-seasonality trap + DHR rationale), §R3 (fit-time budgets), §R4 (order defaults for electricity), §R5 (exogenous choice), §R6 (CI framing + DESIGN §10 interaction), §R7 (residual diagnostics), §R8 (pickling + `apply(refit=False)`), §R9 clear-adopts, §R10 open questions.
5. `docs/intent/DESIGN.md` §2.1 (principles), §2.2.4 (simplicity bias), §3.2 models/evaluation layer responsibilities, §5.1 rolling-origin evaluator, §9 Stage 7 row, §10 global out-of-scope (probabilistic deferral — interacts with D7).
6. `docs/architecture/layers/models.md` — full contract; note open questions for Stage 7 (per-model CLI parity).
7. `docs/architecture/layers/evaluation.md` — harness contract; multi-horizon open question (NOT to be resolved by Stage 7).
8. `docs/plans/completed/06-enhanced-evaluation.md` — for the D-numbered decision idiom, the task-commit-per-task discipline, the stage-hygiene checklist shape, and in particular the **D9 architectural-debt note** (single-flag concession; next extension must go via `EvaluationResult` dataclass — this constrains D11).
9. `src/bristol_ml/models/protocol.py` — the `Model` protocol + `ModelMetadata` re-export.
10. `src/bristol_ml/models/linear.py` — the template implementation (constructor → fit → predict → save → load → metadata property) that `SarimaxModel` follows structurally.
11. `src/bristol_ml/models/io.py` — `save_joblib` / `load_joblib` atomic-write pattern.
12. `src/bristol_ml/evaluation/harness.py:206–259` — the `predict(X_test)` call site that drives D11.
13. `src/bristol_ml/evaluation/harness.py:475–487` and `src/bristol_ml/train.py:220–242` — the two `_build_model_from_config` dispatchers (D10).
14. `src/bristol_ml/features/assembler.py` — feature-frame shape; UTC index guarantees.
15. `conf/_schemas.py` — `NaiveConfig`, `LinearConfig` patterns for `SarimaxConfig`; `ConfigDict(extra="forbid", frozen=True)`; how `AppConfig.model` composes via discriminator.
16. `conf/model/linear.yaml` — template Hydra group for `conf/model/sarimax.yaml`.
17. `conf/evaluation/rolling_origin.yaml` — reference for SARIMAX-specific per-field overrides (D4 applies these at CLI, not via a new file).
18. `tests/unit/models/test_protocol.py` — protocol conformance test pattern.
19. `tests/unit/models/test_linear.py` — model-test convention, fixture patterns.

`CLAUDE.md` + `.claude/playbook/` are read once for process, not per-stage.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

Mapped from `docs/lld/research/07-sarimax-requirements.md`:

**Intent-quoted (AC-1..AC-5):**

1. **The SARIMAX model conforms to the Stage 4 interface.** (AC-1; satisfied by `SarimaxModel` passing `isinstance(model, Model)` and the five-method signature check; tasks T3, T7.)
2. **Fit and predict round-trip through save/load.** (AC-2; satisfied by T7's `test_sarimax_save_load_round_trip_with_exog` including exog to guard issue #6542; task T7.)
3. **The model trains in a reasonable time on the project's data.** (AC-3; satisfied by D1 DHR + D4 fixed 8760-row window + D6 `concentrate_scale=True`; target single-fold fit under 60 s and full notebook evaluation loop under 10 min on laptop CPUs; tasks T3–T6.)
4. **The notebook renders a seasonal decomposition, a fit diagnostic, and a forecast comparison.** (AC-4; decomposition in T5 cell 3, `results.plot_diagnostics()` in T5 cell 5, `forecast_overlay` cross-model comparison in T5 cell 8; task T5.)
5. **A protocol-conformance test covering fit/predict/save/load exists.** (AC-5; `tests/unit/models/test_sarimax.py::test_sarimax_conforms_to_model_protocol`; task T7.)

**Derived (AC-6..AC-11):**

6. `SarimaxConfig` wired into the Hydra discriminated union; `python -m bristol_ml.train model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168` runs end-to-end without code changes (D10). Task T2.
7. `SarimaxModel.predict()` returns `pd.Series` indexed to `features.index` (the codebase surprise 1 regression guard). Task T3, tested in T7.
8. `SARIMAX(..., freq="h", ...)` — no `FutureWarning` / `ValueWarning` about frequency inference during fit (surprise 2 regression guard). Task T3, tested in T7.
9. Stage 6 `acf_residuals` helper fed SARIMAX residuals produces an ACF plot where the lag-168 spike that was present in the linear-baseline residuals is materially reduced — the narrative payoff of Stages 6 → 7 (AC-4's "fit diagnostic"). Task T5 notebook cell.
10. `python -m bristol_ml.models.sarimax --help` works (DESIGN §2.1.1). Task T3.
11. CI green: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pre-commit run --all-files`. Task T8.
12. `src/bristol_ml/models/CLAUDE.md` updated; `docs/architecture/layers/models.md` module inventory updated. Task T8.
13. `CHANGELOG.md` `[Unreleased]` `### Added` bullets; `docs/lld/stages/07-sarimax.md` retro filed; `docs/stages/README.md` Stage 7 row flipped. Task T8.

DESIGN §6 repo-layout tree update is deny-tier for the lead; **H-1** captures the flag-for-human posture.

---

## 5. Architecture summary (no surprises)

Data flow — end-to-end for the Stage 7 notebook:

```
features (10-weather + 44-calendar, UTC index) -- Stage 5 build_calendar_features
   |
   +-- append_weekly_fourier(K=3) -- NEW Stage 7 helper (or inside SarimaxModel.fit)
   |
   v
features_sarimax (60 columns: 10 weather + 44 calendar + 6 weekly Fourier)
   |
   v
SarimaxConfig(order=(1,0,1), seasonal_order=(1,1,1,24), feature_columns=...)
   |
   v
SarimaxModel.fit(X_train, y_train)
  -> SARIMAX(endog=y_train, exog=X_train[feature_columns],
             order=..., seasonal_order=..., freq="h",
             enforce_stationarity=False, enforce_invertibility=False,
             concentrate_scale=True, hamilton_representation=False).fit()
  -> stores self._results: SARIMAXResultsWrapper
   |
   v
SarimaxModel.predict(X_test)
  -> self._results.get_forecast(steps=len(X_test),
                                 exog=X_test[feature_columns].to_numpy())
        .predicted_mean
        .set_axis(X_test.index)   # <-- surprise 1 fix
  -> pd.Series(name=config.target_column)
   |
   v
evaluate(..., return_predictions=True, splitter_cfg=rolling_origin_sarimax)
  -> metrics_df + predictions_df (Stage 6 D9 surface, inherited unchanged)
   |
   v
plots.acf_residuals(predictions_df["error"])      # Stage 6 helper, unchanged
plots.forecast_overlay({"sarimax": ..., "linear": ..., "naive": ...})
```

Public API surface (Stage 7 adds only these):

```python
# models/sarimax.py
class SarimaxModel:
    def __init__(self, config: SarimaxConfig) -> None: ...
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> SarimaxModel: ...
    @property
    def metadata(self) -> ModelMetadata: ...
    @property
    def results(self) -> statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper: ...
    # .results is the statsmodels raw-results passthrough for notebook use
    # (parallels LinearModel.results). Accessing before fit raises RuntimeError.

def _cli_main(argv: list[str] | None = None) -> int: ...
    # Prints SarimaxConfig schema summary + SARIMAX constructor docstring link + palette notice.

# features/fourier.py  (or appended to features/calendar.py — decide in T2)
def append_weekly_fourier(
    df: pd.DataFrame,
    *,
    period_hours: int = 168,
    harmonics: int = 3,
    column_prefix: str = "week",
) -> pd.DataFrame: ...
    # Pure function: takes a UTC-indexed DataFrame, appends 2*harmonics columns
    # (sin/cos for k=1..harmonics at period_hours). No I/O. Deterministic.

# conf/_schemas.py
class SarimaxKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    concentrate_scale: bool = True
    simple_differencing: bool = False
    hamilton_representation: bool = False

class SarimaxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["sarimax"] = "sarimax"
    target_column: str = "nd_mw"
    feature_columns: tuple[str, ...] | None = None  # None => weather + calendar + Fourier weekly
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)
    trend: str | None = None                         # statsmodels trend argument
    weekly_fourier_harmonics: int = 3                # D1+D3: 0 disables Fourier
    sarimax_kwargs: SarimaxKwargs = Field(default_factory=SarimaxKwargs)

# Extended union
ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig

# models/__init__.py lazy re-export adds:
# "SarimaxModel", "SarimaxConfig" (the latter re-exported from conf._schemas for convenience)
```

No change to `cli.py`, `__main__.py`, `load_config()`, splitter code, metrics, benchmarks, feature assembler schema (weather + calendar unchanged; Fourier is append-only inside SarimaxModel), or the existing `LinearModel` / `NaiveModel`.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Config schema: `SarimaxConfig`, `SarimaxKwargs`, Hydra group files
*(Unblocks T2–T8.)*

- [ ] Add `SarimaxKwargs` and `SarimaxConfig` to `conf/_schemas.py` per §5 Architecture summary. Extend `ModelConfig` union to `NaiveConfig | LinearConfig | SarimaxConfig`. Keep `discriminator="type"` on `AppConfig.model`.
- [ ] Create `conf/model/sarimax.yaml` with `# @package model` header, `type: sarimax`, and defaults mirroring `SarimaxConfig` field defaults (including a nested `sarimax_kwargs` block and `weekly_fourier_harmonics: 3`).
- [ ] No change to `conf/config.yaml` defaults list. D4 splitter override is applied via per-field CLI Hydra arguments at runtime (`evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168`); `model=sarimax` switches the model group. No new `conf/evaluation/*.yaml` file is created.
- **Acceptance:** contributes to AC-6, AC-11, AC-12.
- **Tests (spec-derived):**
  - `test_sarimax_config_rejects_extra_keys` — `extra="forbid"` verification.
  - `test_sarimax_kwargs_defaults_match_design_D6` — `enforce_stationarity=False`, `enforce_invertibility=False`, `concentrate_scale=True`, `simple_differencing=False`, `hamilton_representation=False`.
  - `test_sarimax_config_order_default_matches_plan_D2` — `order=(1,0,1)`, `seasonal_order=(1,1,1,24)`.
  - `test_sarimax_config_weekly_fourier_default_is_three_harmonics` — D1+D3 pin.
  - `test_model_config_union_dispatches_on_type_sarimax` — `AppConfig(model={"type": "sarimax", ...})` resolves to a `SarimaxConfig`.
  - `test_config_loads_model_sarimax_via_hydra` — `load_config(overrides=["model=sarimax"])` yields populated `cfg.model` of type `SarimaxConfig`.
  - `test_config_loads_splitter_sarimax_overrides` — `load_config(overrides=["evaluation.rolling_origin.fixed_window=true", "evaluation.rolling_origin.step=168"])` yields `fixed_window=True, min_train_periods=8760, step=168`.
- **Command:** `uv run pytest tests/unit/test_config.py -q && uv run python -m bristol_ml model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168 --cfg job`.

### Task T2 — Weekly-Fourier feature helper
*(Depends on T1; independent of T3.)*

- [ ] Decide file location (one commit message explains the call): `src/bristol_ml/features/fourier.py` (new module, favoured for decoupling) or appended to `src/bristol_ml/features/calendar.py`. Proposal: new module. The function is pure and general (period is parameterised); not specifically "calendar" semantics. Default this decision unless the implementer sees a clear case for co-location.
- [ ] Implement `append_weekly_fourier(df, *, period_hours=168, harmonics=3, column_prefix="week") -> pd.DataFrame`:
  - Requires tz-aware `DatetimeIndex`; raises `ValueError` on tz-naive input.
  - Converts index to integer hours since a fixed epoch (`df.index.view("int64") // 3_600_000_000_000` — nanoseconds to hours) to avoid DST-transition Fourier phase drift; UTC is the correct reference.
  - For `k` in `1..harmonics`, appends two columns: `f"{column_prefix}_sin_k{k}"`, `f"{column_prefix}_cos_k{k}"`.
  - Values: `sin(2πk·t/period_hours)`, `cos(2πk·t/period_hours)`.
  - Returns a *new* DataFrame; does not mutate input (pandas copy-on-write conforming).
  - `harmonics=0` returns `df` unchanged (no-op path).
- [ ] Module docstring + type hints on all public signatures; no `# type: ignore`.
- [ ] Add a `_cli_main` entry point printing one-line descriptions of the public functions (DESIGN §2.1.1).
- **Acceptance:** contributes to AC-6, AC-11 (supports D1/D3).
- **Tests (spec-derived):**
  - `test_append_weekly_fourier_column_names_contract` — six columns for default `harmonics=3`: `week_sin_k1`, `week_cos_k1`, …, `week_cos_k3`.
  - `test_append_weekly_fourier_deterministic_at_fixed_timestamp` — known value assertion at `pd.Timestamp("2024-01-01 00:00:00+00:00")`; verifies `t=0` mapping.
  - `test_append_weekly_fourier_168_period_identity` — values at `t=0` and `t=168h` are equal within float tolerance for all columns (period identity).
  - `test_append_weekly_fourier_harmonics_zero_noop` — `harmonics=0` returns the input frame unchanged.
  - `test_append_weekly_fourier_rejects_tz_naive_index` — `ValueError`.
  - `test_append_weekly_fourier_does_not_mutate_input` — caller's frame is unmodified after call.
  - `test_append_weekly_fourier_output_is_not_dst_sensitive` — on the two DST transition Sundays (March and October 2024), all six columns are continuous (no jump) when evaluated against UTC index.
- **Command:** `uv run pytest tests/unit/features/test_fourier.py -q`.

### Task T3 — `SarimaxModel` scaffold: constructor, metadata, `_cli_main`
*(Depends on T1.)*

- [ ] Create `src/bristol_ml/models/sarimax.py`:
  - Module docstring describing: the model's role in the layer, which protocol methods are implemented, that `SARIMAXResultsWrapper` is stored in `self._results`, and the `append_weekly_fourier` call path.
  - `class SarimaxModel:` with constructor accepting `SarimaxConfig`. Stores `self._config`, `self._results: SARIMAXResultsWrapper | None = None`, `self._feature_columns: tuple[str, ...] = ()`, `self._fit_utc: datetime | None = None`, `self._endog_name: str = ""`.
  - `metadata` property constructs `ModelMetadata` fresh on each call: `name=f"sarimax-{order}-{seasonal_order}"` formatted per `^[a-z][a-z0-9_.-]*$`; `feature_columns=self._feature_columns`; `fit_utc=self._fit_utc`; `git_sha=_git_sha_or_none()` (borrow helper from `linear.py` or extract to `models/io.py`); `hyperparameters` includes `order`, `seasonal_order`, `trend`, `weekly_fourier_harmonics`, and — after fit — `aic`, `bic`, `nobs`, `converged` (bool from `self._results.mle_retvals["converged"]`).
  - `results` property: guard raises `RuntimeError("SarimaxModel must be fit before accessing .results")` when `self._results is None`; otherwise returns `self._results`.
  - Placeholder `fit`, `predict`, `save`, `load` raising `NotImplementedError` — implementation lands in T4.
  - `_cli_main(argv=None) -> int`: prints the `SarimaxConfig` schema (via `SarimaxConfig.model_json_schema()`) + a one-line help banner. Returns 0.
- [ ] Add `SarimaxModel` to the `src/bristol_ml/models/__init__.py` lazy re-export (follow the existing `NaiveModel` / `LinearModel` pattern); add `"SarimaxModel"` to `__all__`.
- [ ] Extend `src/bristol_ml/models/CLAUDE.md` with a new "SARIMAX specifics" subsection (below the existing protocol-invariants list): the `freq="h"` construction requirement (codebase surprise 2), the `predict` re-indexing contract (surprise 1), and a pointer to D5 ("rolling-origin re-fits per fold; `apply(refit=False)` is NOT used inside the harness").
- **Acceptance:** contributes to AC-1, AC-6, AC-10, AC-11.
- **Tests (spec-derived):**
  - `test_sarimax_model_conforms_to_model_protocol` — `isinstance(SarimaxModel(config), Model) is True` (AC-1, AC-5).
  - `test_sarimax_metadata_name_matches_regex` — metadata `name` matches `^[a-z][a-z0-9_.-]*$`.
  - `test_sarimax_metadata_fit_utc_none_before_fit` — unfitted → `fit_utc is None`.
  - `test_sarimax_metadata_feature_columns_empty_before_fit` — unfitted → empty tuple.
  - `test_sarimax_results_property_raises_before_fit` — `RuntimeError` guard.
  - `test_sarimax_cli_main_returns_zero` — `_cli_main([])` returns 0, prints config schema.
- **Command:** `uv run pytest tests/unit/models/test_sarimax.py -q && uv run python -m bristol_ml.models.sarimax --help`.

### Task T4 — `SarimaxModel.fit` and `.predict`
*(Depends on T2, T3.)*

- [ ] Implement `fit(self, features: pd.DataFrame, target: pd.Series) -> None`:
  - Length parity check; raises `ValueError` on mismatch.
  - Resolves `feature_columns`: if `config.feature_columns is not None`, use the configured tuple; else default to every column in `features` (excluding any `timestamp_utc` residual column, which should not be present given harness input).
  - If `config.weekly_fourier_harmonics > 0`, appends weekly-Fourier columns via `append_weekly_fourier(features, harmonics=config.weekly_fourier_harmonics)` and extends `feature_columns` accordingly. The resolved columns (including the Fourier appendages) are stored in `self._feature_columns`.
  - Casts `endog = target.astype("float64")` and `exog = features_with_fourier[self._feature_columns].to_numpy(dtype=np.float64)`.
  - Index regularity guard: if `target.index.freq is None` and timezone is UTC (the Stage 3 assembler's output contract), constructs SARIMAX with `freq="h"`. If timezone is not UTC, raises `ValueError` — the assembler's Stage 3 contract guarantees UTC. (Surprise 2 fix.)
  - Constructs `SARIMAX(endog=endog, exog=exog, order=config.order, seasonal_order=config.seasonal_order, trend=config.trend, freq="h", **config.sarimax_kwargs.model_dump())`.
  - Calls `.fit(disp=False)`. Catches `statsmodels.tools.sm_exceptions.ConvergenceWarning` via `warnings.catch_warnings()` and re-emits at `loguru` WARN level with the fold context (no exception — convergence warnings are informational per domain §R1).
  - Stores `self._results`, `self._feature_columns` (tuple), `self._fit_utc = datetime.now(UTC)`, `self._endog_name = target.name or "target"`.
  - **Re-entrancy:** discards prior `_results` (NFR-5).
- [ ] Implement `predict(self, features: pd.DataFrame) -> pd.Series`:
  - Guard: raises `RuntimeError("SarimaxModel must be fit before predict")` when `self._results is None`.
  - Appends weekly-Fourier columns if `config.weekly_fourier_harmonics > 0` (deterministic given UTC index; produces the same columns as fit did).
  - Builds `exog_test = features_with_fourier[self._feature_columns].to_numpy(dtype=np.float64)`. Raises `KeyError` if a fit-time column is missing from the input.
  - Calls `forecast_result = self._results.get_forecast(steps=len(features), exog=exog_test)`.
  - Extracts `predicted = forecast_result.predicted_mean` (a `pd.Series` with statsmodels' internal index).
  - **Re-indexes** to `features.index` via `.set_axis(features.index)` (surprise 1 fix).
  - Returns `pd.Series(predicted.to_numpy(), index=features.index, name=self._config.target_column)`.
- **Acceptance:** AC-1, AC-2, AC-3, AC-6, AC-7, AC-8, AC-11.
- **Tests (spec + regression):**
  - `test_sarimax_fit_stores_results` — post-fit `self._results` is a `SARIMAXResultsWrapper`.
  - `test_sarimax_fit_rejects_length_mismatch` — `ValueError`.
  - `test_sarimax_fit_rejects_non_utc_index` — `ValueError`.
  - `test_sarimax_fit_emits_freq_hourly_to_sarimax_constructor` — mock/capture the SARIMAX constructor call; assert `freq="h"` present (surprise 2 regression guard).
  - `test_sarimax_fit_is_reentrant` — second `fit()` with different data discards prior state; `len(self._feature_columns)` reflects the latest call only.
  - `test_sarimax_fit_appends_weekly_fourier_when_harmonics_gt_zero` — feature_columns tuple includes `week_sin_k1`…`week_cos_k3`.
  - `test_sarimax_fit_harmonics_zero_skips_fourier` — feature_columns does not contain any `week_` prefix entry.
  - `test_sarimax_predict_returns_series_indexed_to_features_index` — surprise 1 regression guard (the load-bearing test).
  - `test_sarimax_predict_returns_series_with_target_column_name` — `.name == config.target_column`.
  - `test_sarimax_predict_before_fit_raises_runtime_error` — unfitted guard.
  - `test_sarimax_predict_length_matches_features` — `len(pred) == len(features)`.
  - `test_sarimax_predict_raises_on_missing_feature_column` — `KeyError`.
  - `test_sarimax_fit_single_fold_completes_under_60_seconds` — benchmark guard on a 8760-row synthetic endog with default `(1,0,1)(1,1,1,24)` order + 6 Fourier columns. Marked `@pytest.mark.slow`; skipped in default `uv run pytest` via a fast marker filter; run explicitly with `uv run pytest -m slow`. NFR-1 evidence.
- **Command:** `uv run pytest tests/unit/models/test_sarimax.py -q`.

### Task T5 — `SarimaxModel.save` and `.load`; notebook `07_sarimax.ipynb`
*(Depends on T4.)*

- [ ] Implement `save(self, path: Path) -> None`:
  - If `self._results is None`, raises `RuntimeError("Cannot save unfitted SarimaxModel")`.
  - Delegates to `save_joblib(self, path)` (matches `LinearModel.save`). The `SARIMAXResultsWrapper` is pickle-compatible via `MLEResults.__getstate__`; joblib handles it.
  - **Alternative considered and rejected:** calling `self._results.save(path, remove_data=True)` separately — this saves only the results, not the wrapping `SarimaxModel`, and loses the config + feature-column metadata needed for reconstruction. Stick with `save_joblib(self, path)`.
- [ ] Implement `load(cls, path: Path) -> SarimaxModel`:
  - Delegates to `load_joblib(path)`; `isinstance(obj, cls)` guard raises `TypeError` on mismatch (matches `LinearModel.load`).
- [ ] Create `notebooks/07_sarimax.ipynb`:
  - **Cell 0 (md):** title "Stage 7 — SARIMAX", short abstract, links to intent + plan.
  - **Cell 1 (code):** imports + Hydra `load_config(overrides=["model=sarimax", "evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168"])`; load the calendar-feature table via the Stage 5 assembler.
  - **Cell 2 (md):** narrative on GB demand seasonal structure; references Stage 5's weekly-ripple observation and Stage 6's ACF lag-168 spike.
  - **Cell 3 (code):** `statsmodels.tsa.seasonal.STL(y, period=24).fit().plot()` + a separate STL at `period=168` (weekly), both rendered. This is the **AC-4 "seasonal decomposition"** evidence.
  - **Cell 4 (code):** ADF stationarity test on the level series and on the once-seasonally-differenced series; prints p-values. Justifies `d=0, D=1`.
  - **Cell 5 (code):** AIC sweep over a small grid (`p,q ∈ {0,1}`, `P,Q ∈ {0,1}` with `d=0, D=1`) — 16 candidates — printed as a sorted table. Comment explains this is a *notebook exercise*, not an architectural auto-search.
  - **Cell 6 (code):** single-fold `SarimaxModel(config).fit(X_train, y_train)` with `%%time` cell magic (evidence for AC-3 fit-time budget). Emits the fit time as a comment in the YAML later.
  - **Cell 7 (code):** `model.results.plot_diagnostics(figsize=(14,10))` + print `model.results.summary()` — the **AC-4 "fit diagnostic"** evidence + domain §R7 residual-test surface (Ljung-Box, Jarque-Bera, breakvar exposed in summary).
  - **Cell 8 (code):** `evaluate(model, features, splitter_cfg, metrics, return_predictions=True)` over the full SARIMAX-override splitter. Stores `metrics_df, predictions_df`.
  - **Cell 9 (code):** `plots.acf_residuals(predictions_df["error"], lags=168)` — the **Stage 6 → Stage 7 narrative payoff** (AC-9): the lag-168 spike should be materially flattened compared with the linear-baseline ACF from the Stage 4 notebook appendix.
  - **Cell 10 (code):** cross-model comparison. Re-evaluate `NaiveModel(strategy="same_hour_last_week")` and `LinearModel` on the same rolling-origin splits. `plots.forecast_overlay({"naive": ..., "linear": ..., "sarimax": ...})` on a 48-hour window. The **AC-4 "forecast comparison"** evidence.
  - **Cell 11 (code):** summary metric table — `pd.concat([naive_metrics, linear_metrics, sarimax_metrics]).pivot("model","metric","value")`.
  - **Cell 12 (md):** closing narrative — what SARIMAX buys over the linear baseline; hooks into Stage 8 (interactions, functional-form extensions) and Stage 10 (quantile / probabilistic framing, where SARIMAX's analytical CI becomes relevant).
  - Run the notebook end-to-end before commit (`uv run jupyter nbconvert --execute --to notebook --inplace notebooks/07_sarimax.ipynb`). Budget: under 10 minutes end-to-end with the SARIMAX-override splitter; if it exceeds this, reduce `step` to `step=672` (4-weekly folds) and note in the retro.
  - **Budget check:** if single-fold fit blows 60 s in Cell 6, halve the exog set (e.g. drop two of the three Fourier harmonic pairs) and document the trade-off in Cell 6's markdown; fit-time budget is load-bearing for AC-3.
- **Acceptance:** AC-2 (save/load in tests), AC-3, AC-4, AC-9, AC-11.
- **Tests (spec + regression):**
  - `test_sarimax_save_load_round_trip_with_exog` — fit on synthetic 2000-row series with 4 exog columns, save to tmp path, load, predict on 24-row test window, assert `np.testing.assert_allclose(original_pred, reloaded_pred, rtol=1e-10)`. This is the AC-2 test and the issue #6542 regression guard.
  - `test_sarimax_save_unfitted_raises_runtime_error` — pre-fit save raises.
  - `test_sarimax_load_rejects_wrong_type` — loading a pickled `LinearModel` as `SarimaxModel` raises `TypeError`.
  - `test_sarimax_load_preserves_metadata` — reloaded `metadata.feature_columns`, `metadata.fit_utc`, `metadata.hyperparameters["order"]` equal pre-save values.
  - `test_sarimax_reentrant_fit_after_load` — load then `fit()` on fresh data succeeds and discards loaded state (NFR-5 across save/load boundary).
- **Command:** `uv run pytest tests/unit/models/test_sarimax.py -q && uv run jupyter nbconvert --execute --to notebook --inplace notebooks/07_sarimax.ipynb`.

### Task T6 — Harness + train CLI dispatchers
*(Depends on T1, T4.)*

- [ ] Extend `src/bristol_ml/evaluation/harness.py::_build_model_from_config` (lines 475–487 at baseline):
  - Add `isinstance(model_cfg, SarimaxConfig) → return SarimaxModel(model_cfg)` branch before the existing fall-through. Import `SarimaxModel` locally inside the function to avoid module-level circularity.
- [ ] Extend `src/bristol_ml/train.py::_build_model_from_config` (lines 220–242 at baseline) identically. Both dispatchers must be updated — missing either is the surprise-3 silent-exit-code bug.
- [ ] No change to the `evaluate()` signature, the splitter, or any other harness internals. D11's `predict()` contract means the harness treats SARIMAX exactly like the other models.
- **Acceptance:** AC-6, AC-11.
- **Tests (spec + regression):**
  - `test_harness_build_model_dispatches_sarimax_config` — direct test of `_build_model_from_config(SarimaxConfig(...))` returns a `SarimaxModel` instance.
  - `test_train_build_model_dispatches_sarimax_config` — same for `train.py`'s dispatcher.
  - `test_harness_cli_runs_with_model_sarimax` — integration smoke: `uv run python -m bristol_ml.evaluation.harness model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168` exits with code 0 on the fixture feature table. Marked `@pytest.mark.slow` if the run exceeds 30 s in CI; otherwise a fast integration test.
  - `test_train_cli_runs_with_model_sarimax` — same for `train.py`.
- **Command:** `uv run pytest tests/unit/evaluation/test_harness.py tests/unit/test_train.py -q`.

### Task T7 — Protocol-conformance test + residual-ACF regression fixture
*(Depends on T3–T5.)*

- [ ] Add `tests/unit/models/test_sarimax.py` with (consolidated list, drawing from T3/T4/T5/T6 test listings above — this task is for the consolidated conformance coverage and for the residual-ACF narrative-payoff regression test):
  - `test_sarimax_protocol_conformance_all_five_members` — directly invokes `fit`, `predict`, `save`, `load`, and `metadata` on a `SarimaxModel` instance and asserts each behaves per protocol (AC-5, single test that is the ultimate AC-1 + AC-5 pin).
  - `test_sarimax_residual_acf_at_lag_168_materially_lower_than_linear` — fits both `LinearModel` and `SarimaxModel` on the same Stage 5 fixture (~2000-row calendar-feature table), computes ACF at lag 168 on each model's residuals (via `statsmodels.tsa.stattools.acf`), asserts the SARIMAX value is below 50% of the linear-baseline value. This is the AC-9 regression test protecting the Stage 6 → Stage 7 narrative; a failure means the weekly period is not being absorbed (likely D1 or D3 configuration drift).
  - `test_sarimax_fit_emits_no_frequency_userwarning` — asserts no `UserWarning` with message matching `freq` fires during a standard fit (surprise 2 regression guard).
- **Acceptance:** AC-1, AC-5, AC-9.
- **Tests:** the three tests above are the deliverable.
- **Command:** `uv run pytest tests/unit/models/test_sarimax.py -q`.

### Task T8 — Stage hygiene
*(Depends on T1–T7.)*

- [ ] `CHANGELOG.md` — under `[Unreleased]`, add:
  - **Added**
    - `bristol_ml.models.sarimax.SarimaxModel` — SARIMAX model conforming to Stage 4 `Model` protocol, with daily seasonal order + weekly Fourier exogenous regressors (DHR approach per Hyndman fpp3 §12.1).
    - `SarimaxConfig`, `SarimaxKwargs` Pydantic schemas; `conf/model/sarimax.yaml` Hydra group.
    - `bristol_ml.features.fourier.append_weekly_fourier` helper for weekly (period 168 h) Fourier exogenous regressors.
    - `notebooks/07_sarimax.ipynb` — seasonal decomposition, AIC order-selection sweep, fit diagnostics, cross-model comparison.
  - **Changed**
    - `_build_model_from_config` dispatchers in `harness.py` and `train.py` now dispatch `SarimaxConfig` → `SarimaxModel`.
    - `src/bristol_ml/models/__init__.py` re-export surface extended with `SarimaxModel`, `SarimaxConfig`.
- [ ] Create `docs/lld/stages/07-sarimax.md` — retrospective following `docs/lld/stages/00-foundation.md` template:
  - **What was built** — one paragraph per T1..T8 artefact.
  - **Design choices made here** — recap each `D1..D12` decision with its post-markup resolution and a link back to the plan. Flag any DHR-vs-`s=168` debate outcome explicitly.
  - **Demo moment** — paste the command sequence: `uv run python -m bristol_ml.train model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168` + link to the Stage 6 → Stage 7 ACF narrative payoff cell in the notebook.
  - **Deferred** — per-horizon diagnostics (evaluation-layer open question left for Stage 10/11); per-model CLI harmonisation (H-3 Stage 11+); parametric CI plot (D7, Stage 10); `apply(refit=False)` evaluation shortcut (D5, Stage 9 or when fit-time pressure forces the question); auto-order search (intent out-of-scope, not deferred).
  - **Next** → Stage 8 (use the intent's next-stage title verbatim).
- [ ] `docs/stages/README.md` — flip Stage 7 row to `shipped` with links:
  ```
  | 7 | SARIMAX | `shipped` | [intent](../intent/07-sarimax.md) | [plan](../plans/completed/07-sarimax.md) | [models](../architecture/layers/models.md) | — | [retro](../lld/stages/07-sarimax.md) |
  ```
- [ ] `docs/architecture/layers/models.md` — module inventory: add `models/sarimax.py::SarimaxModel`. Mark the "Per-model CLI parity" open question explicitly re-deferred per H-3, with a back-link to this plan.
- [ ] `src/bristol_ml/models/CLAUDE.md` — finalise the "SARIMAX specifics" subsection (scaffolded at T3): the `freq="h"` requirement, the `predict` re-indexing contract, and a pointer to D5.
- [ ] **H-1 flag**: raise the `docs/intent/DESIGN.md §6` batched-edit request to the human at PR review. Batch covers Stages 1–7 additions: `ingestion/holidays.py`, `features/calendar.py`, `features/fourier.py`, `conf/features/weather_calendar.yaml`, `conf/ingestion/holidays.yaml`, `notebooks/05_calendar_features.ipynb`, `evaluation/plots.py`, `conf/evaluation/plots.yaml`, Stage 4 notebook appendix, `models/sarimax.py`, `conf/model/sarimax.yaml`, `notebooks/07_sarimax.ipynb`. Lead MUST NOT edit §6.
- [ ] **H-2 check**: re-read Stage 6 retro "Next" section; if the wording drifted from the current intent, correct it in a one-line edit.
- [ ] `docs/architecture/ROADMAP.md` — if the Models section has any remaining open questions closed by Stage 7, mark them with back-references; otherwise no-op.
- [ ] Move `docs/plans/active/07-sarimax.md` → `docs/plans/completed/07-sarimax.md` via `git mv` as the final commit action.
- **Acceptance:** AC-11, AC-12, AC-13.
- **Tests:** `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- **Command:** `uv run python -m bristol_ml --help && uv run python -m bristol_ml.models.sarimax --help && uv run pytest -q`.

---

## 7. Files expected to change

### New
- `src/bristol_ml/models/sarimax.py`
- `src/bristol_ml/features/fourier.py` (if T2 picks separate-module route)
- `conf/model/sarimax.yaml`
- `tests/unit/models/test_sarimax.py`
- `tests/unit/features/test_fourier.py` (if T2 picks separate-module route; otherwise extend `test_calendar.py`)
- `notebooks/07_sarimax.ipynb`
- `docs/lld/stages/07-sarimax.md`

### Modified
- `conf/_schemas.py` (`SarimaxKwargs`, `SarimaxConfig`; extend `ModelConfig` union)
- `src/bristol_ml/evaluation/harness.py` (add `SarimaxConfig` branch in `_build_model_from_config`)
- `src/bristol_ml/train.py` (add `SarimaxConfig` branch in `_build_model_from_config`)
- `src/bristol_ml/models/__init__.py` (re-exports)
- `src/bristol_ml/models/CLAUDE.md` (SARIMAX specifics subsection)
- `tests/unit/test_config.py` (SarimaxConfig/SarimaxKwargs tests)
- `tests/unit/evaluation/test_harness.py` (dispatch test)
- `tests/unit/test_train.py` (dispatch test)
- `CHANGELOG.md`
- `README.md` (Stage 7 entry-point paragraph; `python -m bristol_ml.models.sarimax --help` mention)
- `docs/stages/README.md` (Stage 7 row flip + links)
- `docs/architecture/layers/models.md` (module inventory + open-question re-defer)
- `docs/architecture/ROADMAP.md` (back-references if any remain)
- `docs/lld/stages/06-enhanced-evaluation.md` (H-2 "Next" check; one-line edit only if drift found)

### Moved (final commit)
- `docs/plans/active/07-sarimax.md` → `docs/plans/completed/07-sarimax.md`

### Explicitly NOT modified
- `docs/intent/DESIGN.md §6` (deny-tier; flag for human — H-1)
- `docs/intent/07-sarimax.md` (immutable once shipped)
- `notebooks/04_linear_baseline.ipynb` (Stage 4 + Stage 6 appendix — do not touch)
- `notebooks/05_calendar_features.ipynb` (Stage 5's narrative — do not touch)
- `src/bristol_ml/evaluation/plots.py` (Stage 6 surface reused as-is — no new helpers)
- `src/bristol_ml/evaluation/harness.py::evaluate` signature (D9 one-off from Stage 6 stands)
- `src/bristol_ml/evaluation/splitter.py`, `metrics.py`, `benchmarks.py` (no internal changes)
- `src/bristol_ml/models/protocol.py` (no protocol change — D11)
- `src/bristol_ml/models/linear.py`, `naive.py` (no touch)
- `src/bristol_ml/features/assembler.py` (schema unchanged — Fourier is appended at model-fit time, not in the assembler)
- `src/bristol_ml/cli.py`, `src/bristol_ml/__main__.py`, `src/bristol_ml/config.py`
- `pyproject.toml` (no new dependency — D12)

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Chosen default order `(1,0,1)(1,1,1,24)` does not fit in the AC-3 time budget on CI | Medium | High | D4 fixed 8760-row window + D6 `concentrate_scale=True` keeps single-fold fit under ~60 s per domain §R3 estimates. `test_sarimax_fit_single_fold_completes_under_60_seconds` is the guard (marked `@pytest.mark.slow`). If it fails, the notebook falls back to `step=672` (4-weekly folds) or reduces Fourier harmonics from 3 to 1. Escalate to human if neither path holds. |
| Lag-168 residual spike persists despite DHR (`test_sarimax_residual_acf_at_lag_168_materially_lower_than_linear` fails) | Low | High | If it fails, D1 / D3 configuration is wrong. Investigate: is Fourier append present in `feature_columns`? Are calendar one-hots colinear with the Fourier terms and being dropped? Diagnose via `model.metadata.feature_columns` inspection before changing D1. Fallback: single SARIMAX with `s=168` on a very small window (D1 override route), at cost of fit-time budget. |
| statsmodels `SARIMAX` emits `FutureWarning` on `freq` handling in a future version | Medium | Low | D6 pins `freq="h"` explicitly. `test_sarimax_fit_emits_no_frequency_userwarning` catches any regression. If statsmodels changes the API, pin minor version or add an adapter. |
| `predict()` re-indexing corrupts the prediction series (e.g. statsmodels returns a shifted-index series and `.set_axis` silently mis-aligns) | Medium | High | `test_sarimax_predict_returns_series_indexed_to_features_index` asserts exact index equality and value correctness on a small synthetic series where ground truth is known. The test must fail-closed: compare `y_pred.values` against an independently-computed forecast, not just the index. |
| Issue #6542 (save/load with exog) regresses in a future statsmodels version | Low | Medium | `test_sarimax_save_load_round_trip_with_exog` covers this directly. Pin guard is already `statsmodels>=0.14,<1`; any regression within that range is caught by the test. |
| `SARIMAXResults` pickle produces large files (>100 MB) and trips the git LFS / repo-hygiene boundary | Medium | Low | The `save` path is for user artefacts, not for the repo. Tests save to `tmp_path` (pytest fixture); no persistence into the repo. Notebook does not call `save()` to a repo-relative path. Document in `models/CLAUDE.md` that users saving SARIMAX artefacts should use `data/models/` (gitignored). |
| The AIC sweep cell in the notebook is misread as auto-order search and breaks the out-of-scope boundary | Medium | Medium | Cell 5 markdown explicitly labels the sweep as "a manual diagnostic exercise, not an architectural feature". Retro notes the distinction. Cell 5 is not invoked from `SarimaxConfig` or the CLI; it is notebook-scoped Python only. |
| `apply(refit=False)` question re-surfaces during implementation because fit-time budget fails | Medium | Medium | D5 decision: stick with re-fit. If the budget fails, escalate to human via plan update — do NOT silently switch to `apply()`. The rolling-origin semantic honesty is load-bearing for cross-model comparisons across the whole project. |
| `append_weekly_fourier` DST-transition handling breaks Fourier column continuity | Low | Medium | T2 test `test_append_weekly_fourier_output_is_not_dst_sensitive` covers this. Using UTC-integer-hour conversion (not `df.index.hour`) sidesteps DST entirely; UTC is the project's canonical timezone (Stage 2 & Stage 3 contracts). |
| The cross-model comparison cell makes SARIMAX look *worse* than the linear baseline on MAE/RMSE | Low | Low (pedagogical risk) | A legitimate research outcome — the Stage 7 notebook should surface it honestly in Cell 11. The teaching story is that classical models are not automatically better; the diagnostic surface (residual ACF flattening at lag 168) is the real payoff. Cell 12 markdown frames this explicitly. |

---

## 9. Exit checklist

Maps to DESIGN §9 definition-of-done.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail` without a linked issue. `@pytest.mark.slow` tests run explicitly via `uv run pytest -m slow` and pass.
- [ ] Ruff clean: `uv run ruff check . && uv run ruff format --check .`.
- [ ] Pre-commit clean: `uv run pre-commit run --all-files`.
- [ ] `python -m bristol_ml --help` works.
- [ ] `python -m bristol_ml.models.sarimax --help` works (AC-10).
- [ ] `python -m bristol_ml.train model=sarimax evaluation.rolling_origin.fixed_window=true evaluation.rolling_origin.step=168` runs end-to-end without code changes (AC-6).
- [ ] `notebooks/07_sarimax.ipynb` runs end-to-end under 10 minutes and renders all three AC-4 views (decomposition, fit diagnostic, forecast comparison).
- [ ] `CHANGELOG.md` `[Unreleased]` has the Stage 7 Added/Changed bullets.
- [ ] `docs/lld/stages/07-sarimax.md` retrospective exists and is complete, including all D1–D12 decisions with final resolutions.
- [ ] `docs/stages/README.md` Stage 7 row is `shipped`.
- [ ] `docs/architecture/layers/models.md` module inventory updated.
- [ ] `src/bristol_ml/models/CLAUDE.md` has the SARIMAX specifics subsection.
- [ ] H-1 DESIGN §6 batched-edit request surfaced to human at PR review (lead MUST NOT edit §6).
- [ ] Plan moved `docs/plans/active/07-sarimax.md` → `docs/plans/completed/07-sarimax.md`.
