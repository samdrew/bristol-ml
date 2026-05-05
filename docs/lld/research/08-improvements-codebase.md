# Stage 8 follow-up — bounded parametric fit (2b): codebase map

## 1. The ScipyParametric module

**File:** `/workspace/src/bristol_ml/models/scipy_parametric.py`

**Public surface.**
- `ScipyParametricModel` — exported via `__all__`.  `ScipyParametricConfig`
  is imported from `conf/_schemas.py` (lines 772–836); not re-exported
  from this module.

**`fit()` and `predict()`.**
- `fit(self, features: pd.DataFrame, target: pd.Series) -> None` (line 297).
- `predict(self, features: pd.DataFrame) -> pd.Series` (line 442).

**Where `curve_fit` is called** (lines 386–406, inside
`with warnings.catch_warnings(record=True) as caught:`).  Two branches:
- `cfg.loss == "linear"` → lines 389–396:
  `curve_fit(_parametric_fn, X, y, p0=p0, method="lm", maxfev=cfg.max_iter)`.
- else → lines 398–406:
  `curve_fit(_parametric_fn, X, y, p0=p0, method="trf", loss=cfg.loss, max_nfev=cfg.max_iter)`.

**popt / pcov plumbing.**
- Assigned `popt, pcov = curve_fit(...)` on line 389/398.
- Published to instance state at lines 428–429:
  `self._popt = np.asarray(popt, ...)` and `self._pcov = pcov_arr`.

**`pcov`-inf WARN log.**
- Lines 416–425 — fires when `not np.all(np.isfinite(np.diag(pcov_arr)))`.
- Secondary WARN for `OptimizeWarning` at lines 408–414.

**`loss=` wiring.**
- `cfg.loss` reaches `curve_fit` only on the else branch (lines
  398–406).  The `linear` branch passes no `loss=` argument (LM does not
  accept one).

## 2. Parameter naming and order

**`_build_param_names`** at line 87 returns the load-bearing ordered
tuple.  For defaults `diurnal_harmonics=3, weekly_harmonics=2`:

```
("alpha", "beta_heat", "beta_cool",
 "diurnal_sin_k1", "diurnal_cos_k1",
 "diurnal_sin_k2", "diurnal_cos_k2",
 "diurnal_sin_k3", "diurnal_cos_k3",
 "weekly_sin_k1",  "weekly_cos_k1",
 "weekly_sin_k2",  "weekly_cos_k2")     # 13 params
```

Total parameter count: `3 + 2*diurnal_harmonics + 2*weekly_harmonics`.
With defaults that is 13.

**Design matrix layout** (`_build_design_matrix`, line 653).  The matrix
is `(n_features, n_obs)` — rows are features.  Row 0 = HDD, row 1 = CDD;
rows 2+ = Fourier pairs in the order
`diurnal_sin_k1, diurnal_cos_k1, ..., weekly_sin_k1, weekly_cos_k1, ...`.

**Fourier coefs are scalar named params, not an array.**  `_parametric_fn`
receives `*params`; `params[3:]` are the Fourier scalars multiplied
one-for-one against Fourier rows (lines 163–174).  They are never
grouped into a sub-array in the solver call.

**Implication for `bounds=`.** The bounds tuple must have exactly
`3 + 2*diurnal_harmonics + 2*weekly_harmonics` entries, in the order
above.  Harmonic counts come from `cfg.diurnal_harmonics` and
`cfg.weekly_harmonics` at fit time, so bounds must be derived at fit
time rather than hard-coded.

**Physical sign conventions** (docstring lines 98–100):
`beta_heat > 0` = colder weather raises demand;
`beta_cool > 0` = hotter weather raises demand.  Both are bounded ≥ 0
under intent option (A).

## 3. Existing tests

**File:** `/workspace/tests/unit/models/test_scipy_parametric.py`

| Test | What it pins |
|---|---|
| `test_parametric_fn_reproduces_known_sinusoid` | hand-calc, `atol=1e-9` |
| `test_parametric_fn_is_pickleable` | pickle round-trip, bit-equal |
| `test_derive_p0_returns_finite_values_on_empty_cooling_segment` | `p0[2] == 0.0` when no above-`t_cool` data |
| `test_build_param_names_count_matches_fn_arity` | `len(names) == 3+2d+2w`, parametrised `(0,0),(3,2),(4,4)` |
| `test_scipy_parametric_unfitted_metadata_name_regex` | `metadata.name` regex |
| `test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit` | pre/post-fit keys |
| `test_scipy_parametric_module_has_cli_main` | `--help` exits 0 |
| `test_scipy_parametric_require_utc_raises_on_tz_naive_index` | ValueError tz-naive |
| `test_scipy_parametric_fit_populates_state` | `_popt.shape==(13,)`, `_pcov.shape==(13,13)`, `_fit_utc` set |
| `test_scipy_parametric_fit_is_reentrant_and_discards_prior_state` | two different-data fits → different popt |
| **`test_scipy_parametric_fit_same_data_same_params`** | **bit-exact** `popt` and `pcov` on two identical-data fits (`np.testing.assert_array_equal`) |
| `test_scipy_parametric_predict_returns_series_with_target_column_name` | `pred.name == config.target_column` |
| `test_scipy_parametric_predict_before_fit_raises_runtime_error` | RuntimeError before fit |
| `test_scipy_parametric_predict_length_matches_features` | `len(pred) == len(features)` |
| `test_scipy_parametric_fit_raises_on_tz_naive_index` | ValueError on fit path |
| `test_scipy_parametric_fit_logs_warning_on_singular_covariance` | WARN contains "pcov"/"non-finite"/"identifiability" |
| `test_scipy_parametric_fit_recovers_known_parameters_within_tolerance` | `alpha_rel_err < 0.05`, `beta_heat_rel_err < 0.10`, `beta_cool_rel_err < 0.20` |
| `test_scipy_parametric_fit_single_fold_completes_under_10_seconds` | `@slow`, wall-clock ≤ 10 s |
| `test_scipy_parametric_save_unfitted_raises_runtime_error` | RuntimeError on save before fit |
| `test_scipy_parametric_save_load_roundtrip_predict_equal` | predictions `atol=1e-12`, `_popt` bit-equal post round-trip |
| `test_scipy_parametric_save_load_preserves_covariance_matrix` | `_pcov` bit-equal post round-trip |
| `test_scipy_parametric_load_wrong_type_raises_type_error` | TypeError on wrong artefact |
| `test_scipy_parametric_fit_recovers_temperature_coefficient_within_5pct` | `beta_heat_rel_err < 0.05` on 720-row clean frame |
| `test_scipy_parametric_fits_competitive_on_synthetic_data` | `@slow`, MAE ≤ 1.5x best of four |
| `test_scipy_parametric_conforms_to_model_protocol` | full Model protocol conformance |
| `test_scipy_parametric_fit_loss_override_changes_fit` | `loss="soft_l1"` popt differs from `linear` by `atol=1.0` |
| `test_scipy_parametric_save_load_skops_roundtrip` | skops round-trip, predictions `atol=1e-12` |

**Hazard test (highest-leverage):**
`test_scipy_parametric_fit_same_data_same_params` (lines 535–574).  Uses
`np.testing.assert_array_equal` — bit-exact.  TRF is deterministic given
same `p0` and data, but two TRF runs produce different bit patterns from
two LM runs.  This test guards *cross-call* determinism on the current
solver — it should still pass under TRF, but if any saved fixture pins
the LM popt vector to bit-exactness elsewhere, that fixture needs
re-recording.

**Recovery tests** (`_within_tolerance`, `_within_5pct`) use relative-
error bounds, not hard popt magnitudes — TRF with correct bounds should
tighten recovery, not loosen it.

## 4. Stage 16 dependency

ScipyParametric appears in
`docs/lld/research/16-model-with-remit-codebase.md` line 176 only as an
informational MAE row (`scipy_parametric: 220249.9 MW`).  No retraining
target; no popt baseline; no assertion that fixes ScipyParametric popt
magnitude in Stage 16 code or tests.

Stage 16 production code lists only `NnTemporalModel` and `SarimaxModel`
as retraining targets.  `notebooks/04_remit_ablation.ipynb` and
`tests/integration/test_notebook_04.py` load from the registry; they do
not call `ScipyParametricModel.fit()`.

**Conclusion.**  No Stage 16 artefact will shift if TRF changes popt
magnitudes.  OQ-3 from the requirements doc can be discharged.

## 5. Notebook 08 plumbing

**Builder:** `/workspace/scripts/_build_notebook_08.py`

The parametric model is invoked in three code cells:
- **Cell 5** (`cell_5`, line 325): `scipy_model.fit(...)` then prints
  param table with CIs.
- **Cell 9** (`cell_9`, line 509): rolling-origin `evaluate(...)`.
- **Cell 11** (`cell_11`, line 614): manual per-fold fits for stability
  diagnostic.

**Cell 12** (`cell_12`, line 681) is **markdown** — the assumptions
appendix.  Critical text:

> No parameter estimate sitting at a bound. `curve_fit` with `method="lm"`
> is unconstrained, so this assumption holds automatically; it is listed
> here because the Gaussian derivation would break down if, say,
> `method="trf"` with explicit bounds were used (not the case in
> Stage 8; a future `loss != "linear"` override would need the same
> audit).

The text directly names the planned change.  **The builder must update
this cell** so the assumptions list reflects the new solver and
acknowledges that bound-saturated parameters violate the Gaussian-CI
assumption.

**Integration test:** `/workspace/tests/integration/test_notebook_08.py`
— `@pytest.mark.slow`, runs `nbconvert --execute` and asserts that
cells `T5 Cell 5`, `T5 Cell 7`, and `T5 Cell 10` produce non-empty
outputs.  No cell-output value inspection; the change does not break
this test mechanically.

## 6. Architecture-doc references

**Layer doc:** `/workspace/docs/architecture/layers/models.md`.  The
`scipy_parametric.py` inventory row (line ~117) explicitly states:

> Default `loss="linear"` + `method="lm"` so `pcov`-derived 95 % CIs
> stay Gaussian (plan D3/D6).

This text must be updated.

**ADRs.**  No ADR specifically covers the parametric solver choice.
ADR 0003 is Model protocol; ADR 0005 is skops serialisation.  The change
is solver mechanics — no new ADR needed.

**Stage 8 retro:** `/workspace/docs/lld/stages/08-scipy-parametric.md`
decision D6 reads:

> `method="lm"` (Levenberg-Marquardt). ACCEPTED. Simpler and faster
> than `trf`; exactly what the default `loss="linear"` needs. No
> bounds are necessary because D1 fixes the hinge temperatures;
> `trf`/`dogbox` would be needed only for bounded or robust-loss fits.

D6 needs a "revisited" addendum noting the empirical reality that
*on rank-deficient training windows*, the unbounded fit diverges, so
bounds are required regardless of fixed hinge temperatures.

## 7. Files to touch

| File | Reason |
|---|---|
| `src/bristol_ml/models/scipy_parametric.py:386–406` | Core change: unify the two `curve_fit` branches under `method="trf"` with bounds; add bound-saturation detection for std-err override |
| `src/bristol_ml/models/scipy_parametric.py:416–425` | Extend WARN to name parameters (rank-deficient *or* bound-saturated) |
| `tests/unit/models/test_scipy_parametric.py` | Add winter-only / summer-only fixtures; possibly re-record the bit-exact determinism vector; add bound-saturation std-err test; update WARN-message regex if granularity tightens |
| `scripts/_build_notebook_08.py` (Cell 12 markdown) | Revise assumptions appendix to acknowledge bounded TRF and bound-saturation case |
| `notebooks/08_scipy_parametric.ipynb` | Regenerated from builder |
| `docs/architecture/layers/models.md` | Inventory row references `method="lm"` |
| `docs/lld/stages/08-scipy-parametric.md` | Append "D6 (revisited)" |
| `src/bristol_ml/models/CLAUDE.md` | "Default `loss="linear"` + `method="lm"`" phrasing |

## 8. Hazards

1. **`max_iter` keyword divergence.**  Linear branch passes
   `maxfev=cfg.max_iter`; TRF branch passes `max_nfev=cfg.max_iter`.
   When unifying, copy the TRF spelling (`max_nfev`).
2. **`pcov` semantics under bounded TRF.**  Per domain research §4: TRF
   uses Moore-Penrose pseudo-inverse, returning a *finite* `pcov` even
   for rank-deficient problems.  The existing finiteness check at
   line 418 will not fire on bound-saturated parameters — the
   implementer must add an explicit `active_mask`-aware override
   that sets `std_err = inf` for parameters at a bound.
3. **`x0` on a bound — historical scipy regressions.**  Per domain
   research §4: scipy 1.11.2 had two regressions on `x0[i] == lb[i]`
   (issues #18793, #19103); both fixed in 1.11.3.  The project pins
   `scipy>=1.13` so the pre-conditions are safe — but defensively, when
   the existing `_derive_p0` produces `0.0` for a parameter whose lower
   bound is also `0.0`, nudge it to `lb + 1e-6`.
4. **D6 was an explicit architectural decision.**  Per the spec-drift
   rule in CLAUDE.md, the divergence must be surfaced.  Append a
   "D6 (revisited)" addendum; do not edit D6 in place.
5. **Cell 12 of nb08 names the planned change.**  Builder text must be
   revised before the notebook is regenerated.
