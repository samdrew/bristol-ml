# Plan — Stage 8: SciPy parametric load model

**Status:** `approved` — all D1–D13 accepted by the human 2026-04-22 with two minor clarifications folded into D2 and D5 (see Resolution log). Phase 2 may proceed.
**Intent:** [`docs/intent/08-scipy-parametric.md`](../../intent/08-scipy-parametric.md)
**Upstream stages shipped:** Stages 0–7 (foundation, NESO demand, weather, feature assembler + splitter, linear baseline + evaluation harness + three-way benchmark, calendar features, enhanced-evaluation diagnostic plots, SARIMAX).
**Downstream consumers:** Stage 9 (registry — inherits the parametric save/load artefact for cross-version-compatibility work), Stage 10 (quantile / probabilistic framing — where the parametric covariance becomes the starting point for rigorous intervals), Stage 11 (further model families — second successive model extension of the discriminated union).
**Baseline SHA:** `50b1970` (tip of `main` after Stage 7 merge via PR #4).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/08-scipy-parametric-requirements.md`](../../lld/research/08-scipy-parametric-requirements.md)
- Codebase map — [`docs/lld/research/08-scipy-parametric-codebase.md`](../../lld/research/08-scipy-parametric-codebase.md)
- Domain research — [`docs/lld/research/08-scipy-parametric-domain.md`](../../lld/research/08-scipy-parametric-domain.md)

**Pedagogical weight.** Intent §Demo moment frames Stage 8 as "the fitted temperature-response curve plotted against the raw scatter, with the parameter table printed alongside … the curve explains in one picture why the linear regression struggles at temperature extremes, and the parameter table lets a facilitator say 'this coefficient means this many megawatts per degree, with this much uncertainty.'" The central design question of this stage is **how much structure to impose by hand** versus how much to let `curve_fit` discover. Decisions D1–D4 all turn on that answer: functional form (D1), harmonic counts (D2), robust loss (D3), and initial-parameter strategy (D4). Shipping a parametric model whose fitted parameters are not interpretable (D1) or whose CIs are silently invalid (D5) contradicts the demo moment.

---

## 1. Decisions for the human (resolve before Phase 2)

Thirteen decision points plus four housekeeping carry-overs. Defaults below lean on the research agents' A-number recommendations and honour the simplicity bias in `DESIGN.md §2.2.4`. Evidence column cites the three research artefacts (§R-n = domain; §C-n = codebase; OQ-n = requirements).

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Functional form for the temperature-response term | **Piecewise-linear double hinge: `HDD = max(0, T_heat − T)` + `CDD = max(0, T − T_cool)`, with `T_heat = 15.5 °C` and `T_cool = 22 °C` fixed (Elexon convention).** Two free slope coefficients (`β_heat`, `β_cool`) plus a base load `α` per row. Notebook sweeps `T_heat ∈ {14, 15.5, 17}` to demonstrate sensitivity but ships the fixed convention. | Elexon HDD/CDD is the industry-standard framing; interpretable ("X MW per degree of heating demand") in the way the demo moment mandates. Fixing the hinge temperatures eliminates the single largest identifiability risk (§R5 — base temperature drifting to edge of support). Relaxing them later is a one-line config change. | Domain §R1 + A1 + A2; requirements OQ-1; intent §Points for consideration ("piecewise-linear or quadratic response"). |
| **D2** | Diurnal + weekly Fourier harmonic counts + exogenous feature scoping | **`diurnal_harmonics = 3` (period = 24 h, 6 columns); `weekly_harmonics = 2` (period = 168 h, 4 columns).** Call `append_weekly_fourier(df, period_hours=24, harmonics=3, column_prefix="diurnal")` then again with `period_hours=168, harmonics=2, column_prefix="weekly"`. The Fourier helper is reused verbatim — no new module-level code. **Clarification (2026-04-22):** the parametric model's design matrix includes **only** the temperature column + the generated Fourier columns. Stage 5 calendar one-hots (`dow_*`, `is_*`) are explicitly **excluded** from the design matrix to avoid partial collinearity with the Fourier weekly terms (domain §R6 + A6). T2/T4 must implement this exclusion even when `feature_columns=None` at config level. | Literature consensus is K=3 diurnal, K=2 weekly for GB electricity demand. Matches Stage 7 SARIMAX's weekly count (3) closely enough that cross-model comparison is apples-to-apples; the 2-vs-3 weekly difference here is deliberate because the parametric model has no autoregressive component to soak up residual weekly structure. 10 Fourier columns + 3 temperature-response coefficients (α, β_heat, β_cool) = 13 parameters — well below the identifiability danger zone. Day-of-week one-hots duplicate the first Fourier mode of the 168-h cycle — including both inflates CIs and blurs the parameter interpretation. | Domain §R6 + A6; requirements OQ-5, OQ-10; codebase §4. |
| **D3** | Robust loss function | **`loss = "linear"` as the default (plain OLS-equivalent).** The config field is `loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear"`. The notebook exhibits a side-by-side fit with `loss="soft_l1"` as a pedagogical comparison but the shipped config is `"linear"` so the parametric-CI narrative is rigorously valid. | The intent calls out robust-loss as a legitimate alternative ("Robustness to outliers … either is a valid choice"), but `pcov` interpretation under non-linear loss is a heuristic, not a rigorous interval (§R4). For the "±Y MW per degree" demo moment to be honest, the default fit must be the one where CIs are standard. `soft_l1` is available via a CLI override if the analyst wants to explore holiday-spike robustness — but it's a secondary surface, not the default. | Domain §R4; requirements OQ-4; intent §Points. |
| **D4** | Initial-parameter strategy | **Data-driven initial guesses derived deterministically from the training data inside `fit()`.** For each call: `α₀ = target.mean()`; `β_heat₀ = −(mean demand below 10 °C − mean demand above 20 °C) / 10`; `β_cool₀ = +(mean demand above 22 °C − mean demand at 17 °C) / 5`; all Fourier coefficients initialised to zero. Config field `p0: tuple[float, ...] \| None = None` accepts an explicit override. | Fixed numeric defaults in the config are fragile — a rolling-origin fold with only-winter training data starts `β_cool` far from any reasonable value. Deterministic derivation from training data satisfies NFR-3 (reproducibility: same data → same `p0`) while avoiding convergence failures on edge-case folds (§R5 — Seber-Wild recommendation). | Domain §R5 + A7; requirements OQ-2; codebase §6 + S6. |
| **D5** | Confidence-interval derivation | **Gaussian approximation from `pcov` diagonal: `param_std_errors = np.sqrt(np.diag(pcov))`; notebook renders 95 % CIs as `param ± 1.96 × std_error`.** `absolute_sigma=False` (default) so `pcov` scales to the empirical noise level. No bootstrap at Stage 8. When `pcov` contains `inf`, store `float('inf')` in the metadata (JSON-valid) and log a WARNING. **Clarification (2026-04-22):** the notebook must include a **dedicated appendix markdown cell** spelling out the three assumptions underlying the Gaussian approximation — (i) homoscedastic residuals (GB demand residuals are visibly heteroscedastic; peak-hour variance >> off-peak), (ii) near-linear model behaviour near the optimum (weak at the hinge transitions), (iii) no parameter estimate at a bound. The appendix names Stage 10 as the owner of bootstrap / quantile-based alternatives. One paragraph of hedging is replaced by a labelled appendix the facilitator can point at during the demo. | Matches `curve_fit`'s native output; keeps the notebook narrative honest ("Gaussian approximation near the optimum"). Bootstrap CIs are more valid under non-Gaussian residuals but cost a full re-fit loop (`B ≥ 500`) per fold and contradict NFR-1. Bootstrap is Stage 10's job. The appendix raises the bar on "being honest about" from one-sentence-hedge to named-assumption list that the pedagogical moment deserves. | Domain §R3 + A4, A8; requirements OQ-3; intent §Points ("validity of those intervals depends on assumptions … worth being honest about"). |
| **D6** | `curve_fit` method | **`method="lm"` (default, Levenberg-Marquardt, no bounds).** `p0` comes from D4; no bounds are necessary because D1 fixes the hinge temperatures. If the implementer discovers a need for bounds during T4, escalate to the lead — do not silently switch to `"trf"`. `maxfev = 5000`. | Simpler and faster than `trf`; exactly what the default `loss="linear"` needs. `trf`/`dogbox` are needed only for bounded or robust-loss fits; D1 + D3 remove both triggers. | Domain §R2; codebase §6. |
| **D7** | Covariance-matrix save format | **Nested `list[list[float]]` embedded in `metadata.hyperparameters["covariance_matrix"]`, alongside `"param_names"`, `"param_values"`, `"param_std_errors"`.** Use `self._pcov.tolist()` at metadata-construction time. `float('inf')` is preserved. No separate ndarray attribute; the whole `ScipyParametricModel` instance is pickled via `save_joblib` and the metadata dict round-trips naturally. | JSON-serialisable (Stage 9 registry friendliness), inspectable in notebooks without unpacking, no protocol changes. Keeps `metadata.hyperparameters` as the single source of CI information (satisfies AC-5 directly). Alternative (separate attribute) forces callers to know an attribute name outside the protocol. | Requirements OQ-6; codebase §2. |
| **D8** | UTC-tz DatetimeIndex guard | **Match the SARIMAX precedent: `ScipyParametricModel.fit()` and `.predict()` both call a `_require_utc_datetimeindex(features, method=...)` static method that raises `ValueError` on tz-naive input.** Factor out the implementation once on `ScipyParametricModel` (a copy of SARIMAX's — do not unify across model classes in this stage). | `append_weekly_fourier` already requires tz-aware input. Silently tolerating tz-naive would push the error inside the helper with a less useful stack trace. Consistency with SARIMAX keeps the learner's mental model simple. Cross-model unification of the guard is a separate refactor (owner: models-layer housekeeping stage). | Requirements OQ-7; codebase §2. |
| **D9** | Notebook target | **New `notebooks/08_scipy_parametric.ipynb` + regenerator script `scripts/_build_notebook_08.py`.** Directly mirror the Stage 7 generator pattern: 3-step regen (`_build_notebook_08.py` → `nbconvert --execute` → `ruff format`). Notebook cells: Hydra config, feature-cache load, raw temperature scatter, fit, `results.summary()`-style parameter table with ±σ bars, fitted temperature-response curve overlaid on scatter, rolling-origin evaluate across Naive / Linear / SARIMAX / Parametric, forecast-overlay three-way comparison, closing forward pointer to Stage 10. | Precedent: every modelling stage since Stage 4 owns a dedicated notebook. The generator pattern is 100% reusable (codebase §7) — 617-line Stage 7 template drops in with cell-body edits only. Append to `07_sarimax.ipynb` is tempting but conflates two narratives. | Requirements OQ-8; codebase §7. |
| **D10** | Full Hydra wiring | **Extend the `ModelConfig` discriminated union at `conf/_schemas.py:542–545` to `NaiveConfig \| LinearConfig \| SarimaxConfig \| ScipyParametricConfig`; create `conf/model/scipy_parametric.yaml` with `# @package model` header and `type: scipy_parametric`; extend both `_build_model_from_config` sites (`harness.py:475–491` and `train.py:218–262`) with a fourth `isinstance` branch.** `python -m bristol_ml.train model=scipy_parametric` runs end-to-end. No `cli.py` / `__main__.py` changes. | DESIGN §2.1.4: configuration lives outside code. Live-demo pattern `model=scipy_parametric` is the minimum bar for every modelling stage. Missing either dispatcher branch is a silent exit-code-3 bug (Stage 7 surprise 3). No new `conf/evaluation/*.yaml` file is created; any fold-window overrides ride per-field Hydra CLI arguments. | Codebase §1 + §5; requirements OQ-11; US-4. |
| **D11** | Dispatcher-duplication response (Stage 7 Phase 3 B1 carry-over) | **Re-defer: extend both dispatcher sites one more time as an `H-4` housekeeping item; DO NOT refactor in Stage 8.** File the ADR (candidate: `docs/architecture/decisions/0004-model-dispatcher-consolidation.md`) in Stage 8's retro §Deferred section, owned by Stage 11 ("when the fifth model family arrives") or a dedicated housekeeping stage. | Refactoring mid-stream expands Stage 8 scope by a third task and delays the pedagogical milestone. The marginal cost of one more dual edit (~15 lines across two files) is lower than the cost of designing + implementing + testing a shared helper under time pressure. Stage 11 is the natural trigger. | Codebase §5 (decision point flagged); Stage 7 Phase 3 review B1. |
| **D12** | Dependency addition | **Add `scipy>=1.13,<2` to `pyproject.toml [project].dependencies`.** Regenerate `uv.lock` via `uv sync --group dev`. No other new runtime deps. | scipy is transitively available (statsmodels pulls it) but **not declared** at the current baseline — relying on transitive availability is fragile and breaks cleanly-resolved CI environments (codebase §10 surprise 1). `1.13` is the first scipy release with formal NumPy 2.0 support; `<2` guards a hypothetical ABI break. | Domain §R8 + A9; codebase §10 + S1; requirements NFR-5, OQ-12. |
| **D13** | Fit-time NFR + benchmark guard | **`NFR-1` at **≤ 10 seconds** for a single-fold `curve_fit` on an 8760-row training window with the shipped defaults (13 parameters).** Guard: a new `@pytest.mark.slow` test `test_scipy_parametric_fit_single_fold_completes_under_10_seconds` in `tests/unit/models/test_scipy_parametric.py`, default-deselected via the Stage 7 pyproject marker. | `curve_fit` on `n_obs ≈ 8760` × 13 parameters is milliseconds-to-seconds on modern hardware (§R7). A 10 s budget is an order of magnitude above the expected fit-time, giving CI headroom without tolerating a pathological regression. SARIMAX's 60 s budget is the wrong reference class. | Domain §R7; requirements NFR-1; codebase §8. |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` layout tree — Stages 1–8 additions batched. Stage 8 adds `src/bristol_ml/models/scipy_parametric.py`, `conf/model/scipy_parametric.yaml`, `notebooks/08_scipy_parametric.ipynb`, `scripts/_build_notebook_08.py`. | **Flag for human-led batched §6 edit at Stage 8 PR review.** Lead MUST NOT touch §6 unilaterally (deny-tier). Stage 7 H-1 carries forward; Stage 8 extends the batch covering Stages 1–8. |
| **H-2** | Stage 7 retro "Next" pointer to Stage 8 — confirm wording is current. | **Verify at T8 hygiene.** Edit if it drifted from the intent's framing. |
| **H-3** | `docs/architecture/layers/models.md` inventory — Stage 8 row added as `Shipped`; Stage 7's "Per-model CLI parity" open question re-deferred (still `models.md:118`, no new context to resolve it). | **Extend inventory; re-defer CLI parity one more stage.** |
| **H-4** | Dispatcher-duplication ADR (B1 from Stage 7 Phase 3) — owner and timing. | **Record as a new deferred item in Stage 8 retro §Deferred; earmark for Stage 11 or dedicated housekeeping stage.** File ADR candidate filename `docs/architecture/decisions/0004-model-dispatcher-consolidation.md` so Stage 11 inherits a concrete target. |

### Resolution log

- **Drafted 2026-04-22** — pre-human-markup. All decisions D1–D13 are proposed defaults.
- **Approved 2026-04-22** — human ACCEPT on D1, D3, D4, D6–D13 as drafted. Two clarifications folded into the table (no direction change):
  - **D2 clarification:** Stage 5 calendar one-hots (`dow_*`, `is_*`) excluded from the parametric model's design matrix; only `temperature_2m` + the generated Fourier columns feed `curve_fit`. Prevents weekly-harmonic / day-of-week collinearity flagged in domain §R6 + A6.
  - **D5 clarification:** notebook gains a dedicated "Assumptions behind these confidence intervals" appendix cell (homoscedasticity, near-linearity, bound-avoidance), naming Stage 10 as the owner of bootstrap / quantile alternatives. Replaces a single-sentence hedge with an explicit labelled list.
- H-1 through H-4 accepted as tabled. Phase 2 proceeds.

---

## 2. Scope

### In scope

Transcribed from `docs/intent/08-scipy-parametric.md §Scope`:

- **A parametric model conforming to the Stage 4 `Model` protocol**, built on top of `scipy.optimize.curve_fit`. Specification includes:
  - a temperature-response term (D1: piecewise-linear double hinge, HDD/CDD fixed),
  - diurnal Fourier harmonics (D2: K=3),
  - weekly Fourier harmonics (D2: K=2).
- **Extraction of parameter estimates and their confidence intervals** from `curve_fit`'s `pcov` (D5: Gaussian approximation, `1.96 × std_error` for 95 % CIs).
- **A notebook** that:
  - fits the model on the Stage 5 calendar-feature table,
  - plots the fitted temperature-response curve against the raw scatter,
  - prints the parameter table with uncertainties,
  - compares forecasts against Stages 4, 6, and 7 priors (three-way overlay extended to four-way).
- **Full Hydra wiring** (D10): `conf/model/scipy_parametric.yaml`, `ScipyParametricConfig` in the discriminated union, both dispatcher branches.
- **New runtime dependency** (D12): `scipy>=1.13,<2` in `pyproject.toml`, regenerated `uv.lock`.

### Out of scope (do not accidentally implement)

Transcribed from `docs/intent/08-scipy-parametric.md §Out of scope, explicitly deferred`, plus items surfaced by the discovery agents:

- Bayesian estimation of the same functional form (MCMC, variational inference). Stage 10+ territory.
- Automatic functional-form search (no `auto_form=True` flag).
- Non-linear programming beyond `scipy.optimize.curve_fit` / `least_squares` (no `minimize`, no `differential_evolution`).
- Probabilistic forecast scoring (pinball loss, CRPS). Stage 10.
- Prediction intervals that fully propagate parameter uncertainty into a calibrated interval. Stage 10.
- Model registry integration. Stage 9.
- Serving-layer integration. Stage 12.
- New `evaluation/` layer helpers — reuse `forecast_overlay`, `metrics_table`, and (optionally, see T5) a temperature-response scatter helper. If the latter is required, ship it inside `notebooks/08_*.ipynb` as a cell-level function rather than promoting to `plots.py`; promote only if a second consumer emerges.
- Bootstrap CIs. Default CI is Gaussian from `pcov` (D5).
- Cross-model dispatcher consolidation. H-4 defers to Stage 11.
- Modifications to `docs/intent/DESIGN.md §6` (deny-tier; H-1).

---

## 3. Reading order for the implementer

A Phase 2 teammate reading only this plan, the intent, and the three research documents should have enough to execute T1–T8 without re-reading the lead's pre-Phase-1 exploration. Reading order:

1. **`docs/intent/08-scipy-parametric.md`** — the spec. Acceptance criteria are authoritative.
2. **This plan §1 (Decisions)** — every default you need to respect. If a default is OVERRIDE'd in the resolution log, read the new value here.
3. **`docs/lld/research/08-scipy-parametric-requirements.md`** — US-1..US-5 and NFR-1..NFR-8. Internalise NFR-4 (pcov-inf guard) before writing `fit()`.
4. **`docs/lld/research/08-scipy-parametric-codebase.md`** — §1 inventory + §2 contract + §6 `curve_fit` + §10 surprises. If you skip one section, do not skip §10.
5. **`docs/lld/research/08-scipy-parametric-domain.md`** — §R3 (CI semantics) + §R5 (identifiability) + A1..A10 recommendations. Skim R1, R2, R6, R8; read §R3 + §R5 carefully.
6. **`src/bristol_ml/models/sarimax.py`** — implementation template. Not a copy; a mirror. Where SARIMAX binds to `SARIMAXResultsWrapper`, Stage 8 binds to a `(popt, pcov)` pair.
7. **`src/bristol_ml/models/linear.py`** — secondary template for the `metadata.hyperparameters["coefficients"]` pattern that Stage 8 generalises.
8. **`src/bristol_ml/features/fourier.py:56–152`** — called twice from `fit()`. Understand why the integer-hour derivation works for both `period_hours=24` and `period_hours=168`.
9. **`tests/unit/models/test_sarimax.py`** — test-naming convention, `_synthetic_utc_frame` helper pattern, `@pytest.mark.slow` usage.
10. **`scripts/_build_notebook_07.py`** — drop-in template for `scripts/_build_notebook_08.py`.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

Transcribed verbatim from `docs/intent/08-scipy-parametric.md §Acceptance criteria`, each mapped to a named test and an evidence path.

| AC | Intent wording | Named test(s) | Evidence |
|----|---------------|---------------|----------|
| **AC-1** | The parametric model conforms to the Stage 4 interface. | `test_scipy_parametric_conforms_to_model_protocol` | `isinstance(model, Model)` passes after fit; all five members exercised. |
| **AC-2** | Fit and predict round-trip through save/load. | `test_scipy_parametric_save_load_roundtrip_predict_equal` | `np.testing.assert_allclose(predict_before, predict_after)`; second `fit()` discards prior state. |
| **AC-3** | The notebook demonstrates the fitted form visually and prints parameter estimates with confidence intervals. | `test_notebook_08_executes_cleanly` (integration) | `jupyter nbconvert --execute` exits 0; cells 5, 7, 9 produce non-empty outputs (scatter+curve, parameter table, 4-way forecast overlay). |
| **AC-4** | The model fits in a reasonable time on the project's data. | `test_scipy_parametric_fit_single_fold_completes_under_10_seconds` (`@pytest.mark.slow`) | D13: 10 s budget for 8760-row single-fold fit, 13 parameters. |
| **AC-5** | Save/load preserves both the parameter values and the confidence-interval information. | `test_scipy_parametric_save_load_preserves_covariance_matrix` | `metadata.hyperparameters["covariance_matrix"]` bit-exact after round-trip; `param_std_errors` derivable unchanged. |

Additional plan-surfaced criteria (not in intent, but required by NFRs + discovery findings):

| AC | Source | Named test(s) |
|----|--------|---------------|
| **AC-6** | NFR-4 (pcov-inf guard) | `test_scipy_parametric_fit_logs_warning_on_singular_covariance` — deliberately under-determined fit triggers `OptimizeWarning`; loguru WARNING captured. |
| **AC-7** | Requirements US-3 (harness consumption) | `test_harness_dispatches_scipy_parametric_model` + `test_train_cli_runs_with_model_scipy_parametric` — both dispatcher sites exercised; CLI smoke passes. |
| **AC-8** | D8 (UTC-tz guard) | `test_scipy_parametric_fit_raises_on_tz_naive_index` — `ValueError` with message naming the UTC contract. |
| **AC-9** | D4 (deterministic `p0` derivation) | `test_scipy_parametric_fit_same_data_same_params` — two calls on identical data produce bit-equal `popt`. |
| **AC-10** | DESIGN §2.1.1 (standalone entrypoint) | `test_scipy_parametric_module_has_cli_main` — `python -m bristol_ml.models.scipy_parametric --help` exits 0. |

---

## 5. Architecture summary (no surprises)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ conf/model/scipy_parametric.yaml  (# @package model, type: scipy_…)      │
│          │ Hydra compose                                                │
│          ▼                                                              │
│ AppConfig.model : ScipyParametricConfig   (Pydantic, discriminator=type)│
│          │                                                              │
│          ▼                                                              │
│ bristol_ml.train._cli_main                                              │
│   ├─ dispatcher #1 (train.py:218–262)                                   │
│   │     elif isinstance(cfg, ScipyParametricConfig):                    │
│   │         primary = ScipyParametricModel(cfg); primary_kind = "…"     │
│   │                                                                     │
│   ▼                                                                     │
│ bristol_ml.evaluation.harness._build_model_from_config                  │
│   └─ dispatcher #2 (harness.py:475–491)                                 │
│        if isinstance(cfg, ScipyParametricConfig):                       │
│            return ScipyParametricModel(cfg)                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ bristol_ml.models.scipy_parametric.ScipyParametricModel                 │
│   state: _config, _popt, _pcov, _feature_columns, _fit_utc,             │
│          _param_names                                                   │
│                                                                         │
│   fit(features, target):                                                │
│     _require_utc_datetimeindex(features)                                │
│     df = append_weekly_fourier(features, period_hours=24, …)            │
│     df = append_weekly_fourier(df, period_hours=168, …)                 │
│     X = _build_design_matrix(df, target_col=cfg.target_column,          │
│                              feature_cols=cfg.feature_columns)          │
│     p0 = _derive_p0(X, target) if cfg.p0 is None else cfg.p0            │
│     popt, pcov = curve_fit(_parametric_fn, X.to_numpy().T,              │
│                            target.to_numpy(), p0=p0, maxfev=5000)       │
│     _store(popt, pcov, param_names, feature_columns, fit_utc)           │
│                                                                         │
│   predict(features):                                                    │
│     _require_utc_datetimeindex(features)                                │
│     df = append_weekly_fourier(...)  # same two calls                   │
│     X = _build_design_matrix(df, …)                                     │
│     y = _parametric_fn(X.to_numpy().T, *_popt)                          │
│     return pd.Series(y, index=features.index,                           │
│                      name=_config.target_column)                        │
│                                                                         │
│   save/load: save_joblib / load_joblib (same pattern as SARIMAX)        │
│   metadata: hyperparameters = {                                         │
│     "param_names": [...], "param_values": popt.tolist(),                │
│     "param_std_errors": sqrt(diag(pcov)).tolist(),                      │
│     "covariance_matrix": pcov.tolist(),                                 │
│     "target_column", "diurnal_harmonics", "weekly_harmonics", "loss",   │
│   }                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ bristol_ml.evaluation.harness.evaluate(model, X_train, X_test, y, …)    │
│   — unchanged; model is a black-box Model-protocol consumer             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ notebooks/08_scipy_parametric.ipynb                                     │
│   cells 0–12: config load, scatter, fit, parameter table, fitted-curve  │
│   overlay, 4-way forecast overlay, closing pointer to Stage 10.         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow, six steps:**

1. Hydra composes `conf/model/scipy_parametric.yaml` into `AppConfig.model` → typed as `ScipyParametricConfig` by the Pydantic discriminated union.
2. `train.py::_cli_main` dispatches on `isinstance(cfg, ScipyParametricConfig)` → instantiates `ScipyParametricModel(cfg)`.
3. The model's `fit()` requires tz-aware UTC input, appends diurnal (K=3) and weekly (K=2) Fourier columns via the existing helper, selects features, derives `p0` from the training data, calls `scipy.optimize.curve_fit` with `method="lm"`, stores `(popt, pcov)` + provenance.
4. `evaluate()` calls `fit()` per fold and `predict()` per fold; the model is a protocol consumer; no harness changes.
5. `save()` pickles the whole instance via `save_joblib`; `load()` de-pickles and type-checks. Covariance survives as an ndarray in the `_pcov` attribute.
6. The notebook renders the fitted temperature-response curve + parameter table + four-way forecast overlay against Naive / Linear / SARIMAX.

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Config schema: `ScipyParametricConfig`, Hydra group file, pyproject dep
*(Depends on: nothing; baseline SHA `50b1970`.)*

- [ ] Add `ScipyParametricConfig` to `conf/_schemas.py` above the `ModelConfig` union (insert before line 542):
  - `model_config = ConfigDict(extra="forbid", frozen=True)`.
  - `type: Literal["scipy_parametric"] = "scipy_parametric"`.
  - `target_column: str = "nd_mw"`.
  - `feature_columns: tuple[str, ...] | None = None`.
  - `diurnal_harmonics: int = Field(default=3, ge=0, le=10)` (D2).
  - `weekly_harmonics: int = Field(default=2, ge=0, le=10)` (D2).
  - `t_heat_celsius: float = 15.5` (D1).
  - `t_cool_celsius: float = 22.0` (D1).
  - `loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear"` (D3).
  - `max_iter: int = Field(default=5000, ge=1)` (D6).
  - `p0: tuple[float, ...] | None = None` (D4).
- [ ] Extend the discriminated union at `conf/_schemas.py:542–545`:
  ```python
  ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig
  ```
- [ ] Create `conf/model/scipy_parametric.yaml`:
  ```yaml
  # @package model
  type: scipy_parametric
  target_column: nd_mw
  feature_columns: null
  diurnal_harmonics: 3
  weekly_harmonics: 2
  t_heat_celsius: 15.5
  t_cool_celsius: 22.0
  loss: linear
  max_iter: 5000
  p0: null
  ```
- [ ] Add `scipy>=1.13,<2` to `pyproject.toml [project].dependencies` (D12). Run `uv sync --group dev` to regenerate `uv.lock`.
- [ ] **Named tests in `tests/unit/test_config.py`:**
  - `test_scipy_parametric_config_defaults_match_yaml` — load `conf/model/scipy_parametric.yaml` via Hydra, assert field-by-field.
  - `test_scipy_parametric_config_rejects_extra_fields` — `ScipyParametricConfig(type="scipy_parametric", spoof="bad")` → `ValidationError`.
  - `test_scipy_parametric_config_rejects_invalid_loss` — `loss="nonsense"` → `ValidationError`.
  - `test_scipy_parametric_config_rejects_negative_harmonics` — `diurnal_harmonics=-1` → `ValidationError`.
  - `test_model_config_discriminator_parses_scipy_parametric` — `AppConfig.model_validate({"model": {"type": "scipy_parametric", ...}})` resolves to `ScipyParametricConfig`.
- [ ] **Command:** `uv run pytest tests/unit/test_config.py -q && uv run ruff check . && uv run ruff format --check .`

### Task T2 — Parametric-model function + helper module
*(Depends on: T1 for the config fields.)*

- [ ] Create `src/bristol_ml/models/scipy_parametric.py` stub containing at minimum:
  - Module-level pure function `_parametric_fn(X: np.ndarray, *params: float) -> np.ndarray` implementing `α + β_heat · HDD + β_cool · CDD + Σ_k (A_k sin(ωt_d·k) + B_k cos(ωt_d·k)) + Σ_j (C_j sin(ωt_w·j) + D_j cos(ωt_w·j))`. `X` is a 2D ndarray of shape `(n_features, n_obs)` in the column order `[temperature, diurnal_sin_k1, diurnal_cos_k1, …, weekly_sin_j1, weekly_cos_j1, …]`. Parameter order: `[α, β_heat, β_cool, A_1, B_1, …, A_K, B_K, C_1, D_1, …, C_J, D_J]`.
  - Module-level pure function `_derive_p0(design_matrix: pd.DataFrame, target: pd.Series, t_heat: float, t_cool: float) -> np.ndarray` implementing D4's data-driven initialisation.
  - Module-level pure function `_build_param_names(diurnal_harmonics: int, weekly_harmonics: int) -> tuple[str, ...]` returning ordered names `("alpha", "beta_heat", "beta_cool", "diurnal_sin_k1", "diurnal_cos_k1", …, "weekly_sin_j1", …)`.
  - `_cli_main()` that prints `ScipyParametricConfig` defaults + a one-line curve_fit pointer + a Stage 6 palette notice (mirrors `sarimax.py::_cli_main`).
- [ ] **Crucial:** `_parametric_fn` must be module-level (codebase §6 + S2). Do not define it inside `fit()`, do not use a lambda, do not bind as a class method.
- [ ] **Named tests in `tests/unit/models/test_scipy_parametric.py`:**
  - `test_parametric_fn_reproduces_known_sinusoid` — ground-truth `α=10000, β_heat=100, β_cool=50` with three diurnal pairs; feed a known X, check output matches hand calculation within `atol=1e-9`.
  - `test_parametric_fn_is_pickleable` — `pickle.dumps(_parametric_fn)` succeeds; `pickle.loads(...)` recovers a callable that returns the same values.
  - `test_derive_p0_returns_finite_values_on_empty_cooling_segment` — training data with only sub-15 °C temperatures; `p0[2]` (beta_cool) is a finite number (e.g. 0), not NaN or inf.
  - `test_build_param_names_count_matches_fn_arity` — `len(names) == 3 + 2*diurnal + 2*weekly` for `(diurnal, weekly) ∈ {(0,0), (3,2), (4,4)}`.
- [ ] **Command:** `uv run pytest tests/unit/models/test_scipy_parametric.py -q`.

### Task T3 — `ScipyParametricModel` scaffold: constructor, metadata, `results`
*(Depends on: T1 (config) + T2 (module-level functions).)*

- [ ] In `src/bristol_ml/models/scipy_parametric.py`, implement `class ScipyParametricModel`:
  - Constructor `__init__(config: ScipyParametricConfig) -> None` (mirrors `SarimaxModel.__init__` at `sarimax.py:91–107`). Fields: `_config`, `_popt: np.ndarray | None = None`, `_pcov: np.ndarray | None = None`, `_feature_columns: tuple[str, ...] = ()`, `_fit_utc: datetime | None = None`, `_param_names: tuple[str, ...] = ()`.
  - `metadata` property — constructs `ModelMetadata` per call with `name=f"scipy-parametric-d{diurnal}-w{weekly}"` (regex-compliant), `feature_columns=self._feature_columns`, `fit_utc=self._fit_utc`, `git_sha=_git_sha_or_none()`, and `hyperparameters` dict: before fit → `{target_column, diurnal_harmonics, weekly_harmonics, loss}`; after fit → above plus `{param_names, param_values, param_std_errors, covariance_matrix}` (D7). Handle `np.inf` in `pcov` by converting to `float("inf")` in the nested list.
  - `_require_utc_datetimeindex(features, method: str)` static method — copy from SARIMAX (see D8).
- [ ] Extend `src/bristol_ml/models/__init__.py` to lazy-re-export `ScipyParametricModel` and `ScipyParametricConfig`.
- [ ] **Named tests:**
  - `test_scipy_parametric_unfitted_metadata_name_regex` — regex `^[a-z][a-z0-9_.-]*$` matches pre-fit metadata.name.
  - `test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit` — keys `{target_column, diurnal_harmonics, weekly_harmonics, loss}`; NOT `{param_values, covariance_matrix}` before fit.
  - `test_scipy_parametric_module_has_cli_main` (AC-10) — `python -m bristol_ml.models.scipy_parametric --help` exits 0.
  - `test_scipy_parametric_require_utc_raises_on_tz_naive_index` (AC-8) — `ValueError` with "UTC" in message.
- [ ] **Command:** `uv run pytest tests/unit/models/test_scipy_parametric.py -q`.

### Task T4 — `ScipyParametricModel.fit` and `.predict`
*(Depends on: T3.)*

- [ ] Implement `fit(self, features: pd.DataFrame, target: pd.Series) -> None`:
  - `_require_utc_datetimeindex(features, method="fit")`.
  - Length-parity assertion (`len(features) == len(target)`; `RuntimeError` with named model otherwise).
  - Two Fourier-helper calls (D2): `period_hours=24, column_prefix="diurnal"`, then `period_hours=168, column_prefix="weekly"`.
  - Feature-column resolution: `None → all columns`, same helper as SARIMAX's `_resolve_feature_columns`.
  - Derive `p0` via `_derive_p0` (D4) if `config.p0 is None`; else use `config.p0` (length check against param count).
  - Build the design matrix (selected features, ordered as `_parametric_fn` expects — temperature first, then diurnal pairs, then weekly pairs).
  - Call `curve_fit(_parametric_fn, design_matrix.T, target.to_numpy(), p0=p0, maxfev=config.max_iter)` inside `warnings.catch_warnings(record=True)` (capture `OptimizeWarning`, re-emit at loguru WARNING — AC-6).
  - On `pcov` containing `np.inf`, log a structured WARNING naming the model + suggesting tighter data range / better `p0` (NFR-4).
  - Store `_popt`, `_pcov`, `_feature_columns`, `_fit_utc = datetime.now(UTC)`, `_param_names`.
  - Re-entrant: any second `fit()` overwrites all fields.
- [ ] Implement `predict(self, features: pd.DataFrame) -> pd.Series`:
  - `_require_utc_datetimeindex(features, method="predict")`.
  - Guard `self._popt is None` → `RuntimeError("Cannot predict with unfitted ScipyParametricModel")`.
  - Same two Fourier appends as `fit()`.
  - Build the same design matrix shape.
  - Return `pd.Series(_parametric_fn(design_matrix.T, *self._popt), index=features.index, name=self._config.target_column)`.
- [ ] **Named tests:**
  - `test_scipy_parametric_fit_populates_state` — `popt.shape == (n_params,)`, `pcov.shape == (n_params, n_params)`, `fit_utc` is tz-aware UTC.
  - `test_scipy_parametric_fit_is_reentrant_and_discards_prior_state` — two fits with different data produce different `popt`.
  - `test_scipy_parametric_fit_same_data_same_params` (AC-9) — two fits on identical synthetic data produce bit-equal `popt` and `pcov`.
  - `test_scipy_parametric_predict_returns_series_with_target_column_name` — `.name == config.target_column`.
  - `test_scipy_parametric_predict_before_fit_raises_runtime_error` — `RuntimeError` with "fit" in message.
  - `test_scipy_parametric_predict_length_matches_features` — `len(pred) == len(features)`.
  - `test_scipy_parametric_fit_raises_on_tz_naive_index` (AC-8) — `ValueError` with UTC contract message.
  - `test_scipy_parametric_fit_logs_warning_on_singular_covariance` (AC-6) — pathological under-determined fit (e.g. K=10 diurnal pairs on 20 rows) → `OptimizeWarning` captured; loguru WARNING fired.
  - `test_scipy_parametric_fit_recovers_known_parameters_within_tolerance` — synthesise data from known params with light noise, fit, assert `np.allclose(popt, true_params, atol=relative_tolerance)`.
  - `test_scipy_parametric_fit_single_fold_completes_under_10_seconds` (`@pytest.mark.slow`, AC-4 + NFR-1) — 8760-row fit under the D13 budget.
- [ ] **Command:** `uv run pytest tests/unit/models/test_scipy_parametric.py -q`.

### Task T5 — `ScipyParametricModel.save` and `.load`; notebook `08_scipy_parametric.ipynb`
*(Depends on: T4.)*

- [ ] Implement `save(self, path: Path) -> None`:
  - Guard `self._popt is None` → `RuntimeError("Cannot save unfitted ScipyParametricModel")`.
  - Delegate to `save_joblib(self, path)` (identical pattern to `SarimaxModel.save`).
- [ ] Implement `load(cls, path: Path) -> ScipyParametricModel`:
  - Delegate to `load_joblib(path)`; type-check → `TypeError` on mismatch.
- [ ] Create `notebooks/08_scipy_parametric.ipynb` via `scripts/_build_notebook_08.py` (D9 generator pattern):
  - **Cell 0 (md)**: title, abstract, intent + plan links.
  - **Cell 1 (code)**: imports + `load_config(overrides=["model=scipy_parametric"])` + feature-cache load.
  - **Cell 2 (md)**: narrative on the temperature-response physics (heating / cooling / base load).
  - **Cell 3 (code)**: raw scatter `temperature_2m` vs `nd_mw`, `OKABE_ITO[1]` alpha 0.15.
  - **Cell 4 (md)**: functional-form narrative (D1 HDD/CDD framing, why fixed hinge temperatures).
  - **Cell 5 (code)**: single-fold fit, `%%time` cell (AC-4 evidence), print `popt`, `param_std_errors`, and 95 % CIs.
  - **Cell 6 (md)**: parameter-table interpretation prose (what each row means physically).
  - **Cell 7 (code)**: render a pandas DataFrame parameter-table with `value ± 1.96*std` columns (AC-3 evidence).
  - **Cell 8 (code)**: overlay the fitted temperature-response curve on the raw scatter (fitted line computed by evaluating `_parametric_fn` with all Fourier terms zeroed).
  - **Cell 9 (code)**: rolling-origin `evaluate()` across Naive / Linear / SARIMAX / ScipyParametric on 4-6 folds; 4-way metrics DataFrame.
  - **Cell 10 (code)**: `forecast_overlay` 48-hour window, four-way.
  - **Cell 11 (md)**: parameter-stability-across-folds diagnostic (OQ-9 pedagogical bonus): small multiples of `popt` per fold.
  - **Cell 12 (md)**: **Appendix — "Assumptions behind these confidence intervals"** (D5 clarification). Explicitly lists (i) homoscedasticity (with note that GB demand residuals are visibly peak-hour-heteroscedastic), (ii) near-linearity of the model around the optimum (weak at hinge transitions), (iii) no parameter estimate sitting at a bound. Names Stage 10 as the owner of bootstrap / quantile-based alternatives.
  - **Cell 13 (md)**: closing markdown — Stage 9 (registry) and Stage 10 (quantile regression for calibrated intervals) forward pointers.
- [ ] `scripts/_build_notebook_08.py` is a copy of `scripts/_build_notebook_07.py` with the `OUT` path + cell bodies replaced.
- [ ] Run `jupyter nbconvert --execute --to notebook --inplace notebooks/08_scipy_parametric.ipynb` then `uv run ruff format notebooks/08_scipy_parametric.ipynb`.
- [ ] **Named tests:**
  - `test_scipy_parametric_save_unfitted_raises_runtime_error` — matches Stage 7 precedent.
  - `test_scipy_parametric_save_load_roundtrip_predict_equal` (AC-2) — `allclose(predict_before, predict_after)`.
  - `test_scipy_parametric_save_load_preserves_covariance_matrix` (AC-5) — bit-exact `_pcov` after round-trip; `metadata.hyperparameters["covariance_matrix"]` nested-list equal.
  - `test_scipy_parametric_load_wrong_type_raises_type_error` — saving a `LinearModel` and loading as `ScipyParametricModel` raises.
  - Integration: `test_notebook_08_executes_cleanly` (AC-3) — `nbconvert --execute` exits 0.
- [ ] **Command:** `uv run pytest tests/unit/models/test_scipy_parametric.py -q`.

### Task T6 — Harness + train CLI dispatchers
*(Depends on: T1 (config) + T4 (fit/predict) + T5 (save/load).)*

- [ ] Extend `src/bristol_ml/evaluation/harness.py:475–491`'s `_build_model_from_config` with a fourth branch (D10):
  ```python
  if isinstance(model_cfg, ScipyParametricConfig):
      from bristol_ml.models.scipy_parametric import ScipyParametricModel
      return ScipyParametricModel(model_cfg)
  ```
  Import `ScipyParametricConfig` at the top of the function (local import alongside the other three).
- [ ] Extend `src/bristol_ml/train.py:218–262`'s inline `_cli_main` dispatcher with a fourth `elif`:
  ```python
  elif isinstance(model_cfg, ScipyParametricConfig):
      if model_cfg.feature_columns is None:
          scipy_cfg = model_cfg.model_copy(update={"feature_columns": feature_column_names})
      else:
          logger.info(...)  # mirrors SarimaxConfig branch log
          scipy_cfg = model_cfg
      primary = ScipyParametricModel(scipy_cfg)
      primary_kind = "scipy_parametric"
  ```
- [ ] Extend `_target_column` helper in both files to accept `ScipyParametricConfig`.
- [ ] **Named tests:**
  - `test_harness_dispatches_scipy_parametric_model` (AC-7) — `_build_model_from_config(ScipyParametricConfig(...))` returns a `ScipyParametricModel`.
  - `test_train_cli_runs_with_model_scipy_parametric` (AC-7, integration) — `python -m bristol_ml.train model=scipy_parametric` on fixture feature cache exits 0.
  - `test_train_cli_scipy_parametric_feature_columns_promotion` — `feature_columns=None` at the CLI resolves to the feature-set tuple (mirrors R3 fix in Stage 7).
- [ ] **Command:** `uv run pytest -q`.

### Task T7 — Behavioural guards + parametric-recovery regression
*(Depends on: T4.)*

- [ ] Write a parametric-recovery regression test:
  - `test_scipy_parametric_fit_recovers_temperature_coefficient_within_5pct` — synthesise demand with `β_heat = 100 MW/°C`, noise scale 200 MW, fit, assert `abs(popt[1] - 100) / 100 < 0.05`.
- [ ] Write a four-way comparison acceptance test (not run by default; `@pytest.mark.slow`):
  - `test_scipy_parametric_fits_competitive_on_synthetic_data` — on the `_synthetic_utc_frame_with_temperature` fixture, ScipyParametric MAE within 20 % of the best model (likely SARIMAX). Not a strict ordering; just a sanity upper bound.
- [ ] Full-protocol-conformance test:
  - `test_scipy_parametric_conforms_to_model_protocol` (AC-1) — after fit, `isinstance(model, Model)` plus exercise all five members.
- [ ] **Command:** `uv run pytest tests/unit/models/test_scipy_parametric.py -q`.

### Task T8 — Stage hygiene
*(Depends on: all of T1–T7; this task is the final commit before PR.)*

- [ ] `CHANGELOG.md` — add Stage 8 bullets under `[Unreleased]`:
  - `### Added`: ScipyParametricModel, ScipyParametricConfig, conf/model/scipy_parametric.yaml, notebooks/08_scipy_parametric.ipynb, scripts/_build_notebook_08.py, scipy>=1.13,<2 dependency.
  - `### Changed`: `ModelConfig` union extended; both dispatchers gained a fourth branch.
- [ ] Write `docs/lld/stages/08-scipy-parametric.md` following the Stage 7 retro template: Goal / What was built (T1–T8) / Design choices (D1–D13) / Demo moment / Deferred (H-1..H-4 + B1 ADR status) / Next → Stage 9.
- [ ] `docs/stages/README.md` — Stage 8 row flipped to `shipped` with five links (intent, plan, layer, LLD, retro).
- [ ] `docs/architecture/layers/models.md` — inventory row for `scipy_parametric.py` flipped to `Shipped`; H-3 (Per-model CLI parity) re-deferred with a back-reference.
- [ ] `src/bristol_ml/models/CLAUDE.md` — add a `ScipyParametricModel` subsection with the pickleability note (S2) and the `_pcov`-inf guard (NFR-4).
- [ ] Verify H-2: Stage 7's retro "Next" wording is still current; edit if drifted.
- [ ] Flag H-1 and H-4 for the human at PR review as deferred items.
- [ ] Move `docs/plans/active/08-scipy-parametric.md` → `docs/plans/completed/08-scipy-parametric.md` as the final commit action.
- [ ] **Command:** `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.

---

## 7. Files expected to change

### New

- `src/bristol_ml/models/scipy_parametric.py`
- `conf/model/scipy_parametric.yaml`
- `tests/unit/models/test_scipy_parametric.py`
- `notebooks/08_scipy_parametric.ipynb`
- `scripts/_build_notebook_08.py`
- `docs/lld/stages/08-scipy-parametric.md`

### Modified

- `conf/_schemas.py` — `ScipyParametricConfig` added (above union); `ModelConfig` union extended.
- `src/bristol_ml/models/__init__.py` — lazy re-export added.
- `src/bristol_ml/evaluation/harness.py` — fourth dispatcher branch.
- `src/bristol_ml/train.py` — fourth dispatcher branch + `_target_column` extension.
- `src/bristol_ml/models/CLAUDE.md` — `ScipyParametricModel` subsection.
- `tests/unit/test_config.py` — `ScipyParametricConfig` tests.
- `tests/unit/evaluation/test_harness.py` — dispatcher-branch test.
- `tests/unit/test_train_cli.py` — CLI-branch test.
- `pyproject.toml` — `scipy>=1.13,<2` dependency + any slow-marker updates.
- `uv.lock` — regenerated after adding scipy.
- `CHANGELOG.md` — `[Unreleased]` Stage 8 bullets.
- `docs/stages/README.md` — Stage 8 row.
- `docs/architecture/layers/models.md` — inventory row + H-3 re-deferral.
- `docs/lld/stages/07-sarimax.md` — only if H-2 requires a Next-wording tweak.

### Moved (final commit of T8)

- `docs/plans/active/08-scipy-parametric.md` → `docs/plans/completed/08-scipy-parametric.md`.

### Explicitly NOT modified

- `docs/intent/DESIGN.md` §6 — deny-tier; H-1 batches Stages 1–8 for a human-led edit.
- `docs/intent/08-scipy-parametric.md` — immutable spec.
- `src/bristol_ml/evaluation/harness.py::evaluate` signature — no new flags (Stage 6 D9 debt note).
- `src/bristol_ml/evaluation/plots.py` — no new helper; temperature-response scatter is notebook-inline (see §2 Out of scope).
- `src/bristol_ml/models/linear.py`, `naive.py`, `sarimax.py` — untouched except via the `__init__.py` re-export.
- `src/bristol_ml/features/fourier.py` — reused verbatim with `period_hours=24`. No change.
- `src/bristol_ml/cli.py`, `__main__.py`, `config.py` — no changes.
- `docs/architecture/decisions/` — dispatcher-consolidation ADR (B1 / H-4) is deferred to Stage 11+.

---

## 8. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | `curve_fit` converges to a `pcov` full of `inf` on realistic folds, silently producing `nan` std errors and vacuous CIs. | Medium | High | D4 data-driven `p0` + D1 fixed hinge temperatures sharply reduce the risk. NFR-4 makes the WARNING explicit; AC-6 guards it with a test. If a rolling-origin fold hits this in real evaluation, escalate — do not weaken the test. |
| **R2** | The fitted temperature-response coefficients drift wildly across folds (D1 weak under seasonal training windows), undermining the demo moment's "±Y MW per degree" claim. | Medium | Medium | OQ-9 parameter-stability diagnostic in the notebook (Cell 11) makes drift visible. If coefficients jump by >2σ between consecutive folds on real data, flag in the retro and consider a fold-size NFR. |
| **R3** | Chosen 13-parameter form overfits on short training windows (e.g. 30-day initial fold), producing near-singular `pcov`. | Medium | Medium | D4 initialisation + NFR-4 guard. Notebook's six-fold protocol matches Stage 7 (min_train_periods=720 rows, step=1344, test_len=168) — short-window foot-gun is only if an experimenter overrides these. |
| **R4** | `scipy` transitive availability masks a missing declared dep; a fresh Python 3.12 venv without statsmodels resolution fails at `import scipy.optimize`. | Low | High | D12 explicit `scipy>=1.13,<2` + regenerated `uv.lock` in T1. CI's `uv sync --group dev --frozen` catches the gap immediately. |
| **R5** | `_parametric_fn` accidentally defined as a local function or lambda — `save()` fails with `PicklingError`. | Medium | Low | T2 explicitly constrains the function to module level. T2 test `test_parametric_fn_is_pickleable` traps this at CI time. |
| **R6** | Dispatcher duplication (two sites) accidentally skewed — Stage 8 updates one and forgets the other; silent exit-code-3 on one CLI path. | Medium | Medium | T6 has two separate tests (harness vs train CLI). H-4 records the tech debt for Stage 11 consolidation. |
| **R7** | Notebook cell ordering drifts between the generator and the executed notebook (ruff format reflow + nbconvert re-serialisation). | Low | Low | Stage 7's 3-step regen (build → execute → format) is the documented pattern. Run `scripts/_build_notebook_08.py` before every commit that touches the notebook. |
| **R8** | T5's integration test `test_notebook_08_executes_cleanly` blows the CI time budget because `jupyter nbconvert --execute` is slow. | Low | Medium | Mark the integration test `@pytest.mark.slow` if the wall-clock exceeds 60 s; keep unit tests fast-only. |
| **R9** | `loss != "linear"` path (D3 CLI override) produces a pcov that the notebook's CI code treats as Gaussian-valid, giving mis-calibrated intervals. | Low | Medium | D5 narrative is explicit that pcov under robust loss is a heuristic. The notebook's side-by-side cell prints a warning string when `loss != "linear"`. |

---

## 9. Exit checklist

Verified before T8's final commit.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail` without a linked issue. `@pytest.mark.slow` tests run explicitly via `uv run pytest -m slow` and pass on CI-class hardware.
- [ ] Ruff + format + pre-commit clean: `uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- [ ] `uv run python -m bristol_ml.models.scipy_parametric --help` exits 0 and prints the config + pointer (AC-10, DESIGN §2.1.1).
- [ ] `uv run python -m bristol_ml.train model=scipy_parametric` exits 0 end-to-end on the fixture feature cache (AC-7).
- [ ] `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/08_scipy_parametric.ipynb` executes cleanly; all ten code cells produce non-empty outputs (AC-3).
- [ ] All five intent-ACs mapped to named tests in §4 have a passing test.
- [ ] All five additional plan-surfaced ACs (AC-6..AC-10) have a passing test.
- [ ] `docs/lld/stages/08-scipy-parametric.md` retro written per template.
- [ ] `CHANGELOG.md`, `docs/stages/README.md`, `docs/architecture/layers/models.md`, `src/bristol_ml/models/CLAUDE.md` all updated.
- [ ] `docs/plans/active/08-scipy-parametric.md` moved to `docs/plans/completed/`.
- [ ] H-1 (DESIGN §6), H-2 (Stage 7 retro wording), H-3 (CLI parity re-defer), H-4 (dispatcher ADR) all actioned per §1 Housekeeping.
- [ ] PR description includes: Stage 8 summary, any Phase 3 review findings, H-1 DESIGN §6 batched edit request, H-4 ADR follow-up ask.
