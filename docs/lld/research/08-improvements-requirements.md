# Stage 8 follow-up — bounded parametric fit (2b): requirements

## 1. Goal

Switch `ScipyParametricModel.fit` from the unconstrained Levenberg-Marquardt
solver to scipy's bounded Trust-Region Reflective solver with
physically-motivated parameter bounds, so that rank-deficient training
folds produce bounded, interpretable output instead of catastrophically
diverged `popt`.

## 2. User stories

**S1 — winter-only fold.** Given a 30-day winter window where every
temperature is below `t_heat` (CDD ≡ 0), when `fit()` completes, then
`popt[beta_cool] == 0.0` (clamped at lower bound), `popt[alpha]` is in
[20 000, 50 000] MW, and the `pcov`-inf WARN names `beta_cool` only.

**S2 — summer-only fold.** Given a 30-day summer window where every
temperature is above `t_cool` (HDD ≡ 0), when `fit()` completes, then
`popt[beta_heat] == 0.0` and the WARN names `beta_heat` only.

**S3 — healthy full-year path.** Given the project-default 1-year+
training window, when `fit()` completes, then fitted `popt` matches
pre-change LM values within rtol < 1e-4 for every parameter.

**S4 — cross-fold mean MAE recovery.** Given the Stage 8 notebook's
existing 30-day rolling splitter (unchanged), when 50-fold CV runs, then
cross-fold mean MAE drops from ~167 600 to <6 000 (within ±20 % of the
~4 850 median).

**S5 — std errors at bound.** Given any fold where a parameter hits a
bound, when `metadata.hyperparameters["param_std_errors"]` is read, then
bound-hitting parameters report `inf` and interior parameters match
pre-change values within rtol < 1e-4.

**S6 — robust losses unchanged.** Given `loss ∈ {soft_l1, huber, cauchy}`,
when `fit()` is called, then it completes without exception and `popt` is
finite (TRF already services these; the change unifies the linear-loss
path onto the same solver).

## 3. Acceptance criteria

**AC-1 (winter-only window).**
- Fixture: synthetic UTC `DatetimeIndex` of 720 hours, `temperature_2m`
  uniform in [-5.0, 10.0] °C, target ~30 000 MW with light Gaussian noise.
- `popt[alpha]` ∈ [20 000, 50 000] MW.
- `popt[beta_cool] == 0.0` (parameter at lower bound).
- A loguru WARNING fires for `beta_cool`'s std-err being `inf`; no such
  WARNING fires for `beta_heat`.

**AC-2 (summer-only window).**
- Fixture: temperatures uniform in [25.0, 35.0] °C, same length and
  target profile as AC-1.
- `popt[beta_heat] == 0.0` (parameter at lower bound).
- WARNING names `beta_heat` only.

**AC-3 (no regression on healthy full-year window).**
- Pre-change `popt` recorded as a golden baseline vector.
- After the switch: every entry of `popt` satisfies
  `|popt_new - popt_old| / |popt_old| < 1e-4`.
- `cross_fold_mean_MAE` on the default CLI config stays within rtol <
  1e-4 of the pre-change value.
- Note: AC-3 must be empirically verified — TRF and LM use different
  termination criteria and may not produce bit-identical popt even on
  healthy data (see domain research §5).  If rtol > 1e-4 is observed on
  healthy fixtures, widen the test, do not loosen the bound.

**AC-4 (cross-fold mean MAE recovery).**
- Stage 8 notebook's `min_train_periods=720, fixed_window=True` rolling
  splitter (unchanged from option-1 pre-state).
- Cross-fold mean MAE < 6 000 MW.

**AC-5 (standard errors at boundary).**
- On AC-1 / AC-2 fixtures: `std_err[bound_param] == inf`; all other
  entries finite and match pre-change to within rtol < 1e-4.

**AC-6 (robust losses unchanged).**
- For `loss ∈ {linear, soft_l1, huber, cauchy}` on a healthy training
  window: `fit()` completes without exception and `popt` is finite for
  all parameters.

## 4. Non-functional requirements

**Performance.**  No explicit wall-time budget.  Expectation: TRF needs
slightly more Jacobian evaluations per iteration on healthy folds, but
saves substantial time on previously-divergent folds where LM exhausts
`maxfev`.  The Stage 8 notebook's 50-fold CV must complete in a
reasonable interactive session time (indicatively < 5 min on the dev
host).  No benchmark regression test is required.

**Backwards compatibility.**
- `popt` on healthy paths matches pre-change LM values within rtol < 1e-4.
- `ScipyParametricConfig` schema shape does not change.  Bounds are
  private to `fit()`.  `extra="forbid"` on the frozen Pydantic model is
  preserved.
- Saved artefacts (skops envelope `scipy-parametric-state-v1`) remain
  load-compatible — only `popt` values change, not the envelope.

**Determinism.**
- Two `fit()` calls on the same data with the new solver must produce
  bit-equal `popt`.  TRF is deterministic given identical `x0`, `bounds`,
  data, and tolerance kwargs (see domain research §5).
- The existing `test_scipy_parametric_fit_same_data_same_params` uses
  `np.testing.assert_array_equal` (atol=0); this test guards
  *cross-call* determinism on a single solver, not cross-solver
  agreement, so it should remain bit-exact under TRF.

**Observability.**
- Existing `pcov`-inf WARN path continues to fire for rank-deficient
  cases.  **New code path required** for "parameter at active bound →
  std_err = inf" because TRF returns a finite (Moore-Penrose pseudo-inverse)
  `pcov` for bound-saturated parameters; without an explicit override the
  existing finiteness check will not fire on AC-1/AC-2 fixtures.
- Update `fit()` docstring: `pcov` unreliability now has two causes —
  singular Jacobian and parameter at a bound.

## 5. Out of scope

Taken from the intent verbatim:

- A second bounded-fit solver (`dogbox`).  TRF is the documented scipy
  default for bounded problems.
- Changes to `ScipyParametricConfig`'s public shape.  Bounds are private
  to `fit()`.
- Bootstrap or quantile-based CI alternatives (deferred to Stage 10).

## 6. Open questions

**OQ-1 — determinism test tolerance (intent line 48–49).**
`test_scipy_parametric_fit_same_data_same_params` asserts bit-exact
equality.  TRF should reproduce TRF on identical inputs, but the *first*
TRF run will not match the LM golden vector bit-exactly.  Decide:
1. Re-record the golden vector under TRF (preferred — the test guards
   cross-call determinism on the *current* solver, not cross-solver
   parity);
2. or widen to `np.testing.assert_allclose(rtol=1e-4)`.
Best guess: re-record the golden vector; the bit-exact test continues to
guard cross-call determinism on the new solver.

**OQ-2 — beta_cool bound (A) vs (B) (intent lines 24–29).**
Intent ships (A) `beta_cool ≥ 0` as default with (B) `beta_cool ≥ -1000`
"documented as an override configuration."  Adding a config field
contradicts the explicit "no changes to `ScipyParametricConfig` shape"
out-of-scope statement.
Best guess: (B) is a notebook/markdown comment only — no new config
field.  Confirm with the human if a config field is wanted.

**OQ-3 — Stage 16 retro re-baseline (intent line 50).**
The Stage 16 plan does not retrain ScipyParametric — it appears only as
an informational baseline-MAE row.  The intent's "worth checking on the
Phase 3 review" can therefore be discharged with: "Stage 16 has no
ScipyParametric retraining target; baseline-MAE rows are advisory."
Best guess: defer entirely; no Stage 16 work in this PR.

**OQ-4 — pcov WARN granularity per parameter.**
AC-1/AC-2 say the WARN fires "for `beta_cool` only" / "for `beta_heat`
only".  The current WARN at `scipy_parametric.py:418` fires once at the
"any diagonal entry non-finite" level — it does not name parameters.
Combined with the domain-research finding that TRF's pseudo-inverse
`pcov` is not non-finite for bound-saturated parameters, **the
implementer must both (a) detect bound saturation and override std_err
to inf, and (b) name the affected parameters in the WARN message**.
Best guess: extend the WARN to enumerate parameters with
non-finite std-err (whether from rank deficiency or bound saturation).

**OQ-5 — D6 reversal in the Stage 8 retro.**
The Stage 8 retro records `method="lm"` as a deliberate decision (D6).
Per the CLAUDE.md spec-drift rule, this divergence must be surfaced.
Best guess: append a "D6 (revisited)" entry to
`docs/lld/stages/08-scipy-parametric.md` documenting the bounded-TRF
follow-up; no new ADR needed (the change is solver mechanics, not
public-interface).
