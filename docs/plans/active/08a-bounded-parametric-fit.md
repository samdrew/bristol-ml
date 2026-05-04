# 08a — Bounded parametric fit (`method="trf"` with bounds)

**Status.** Phase 1 plan, ready for review.

**Branch.** `fix/scipy-parametric-bounded-fit` (off `main`).

**Type.** Stage 8 follow-up (not a new stage).  Reverses Stage 8
decision D6 (`method="lm"`).

## Preamble

### Intent
- [`docs/intent/08-improvements.md`](../../intent/08-improvements.md)

### Phase 1 research artefacts
- [Requirements](../../lld/research/08-improvements-requirements.md)
- [Codebase map](../../lld/research/08-improvements-codebase.md)
- [Domain research](../../lld/research/08-improvements-domain.md) — TRF
  vs LM, `pcov` semantics, scipy version pitfalls
- [Scope diff](../../lld/research/08-improvements-scope-diff.md) —
  minimalist scope-guard pre-synthesis

### Problem in one paragraph
`ScipyParametricModel.fit` calls `curve_fit(..., method="lm")` —
unconstrained Levenberg-Marquardt.  When a training window contains no
observations on one side of a hinge (HDD ≡ 0 in summer-only folds, CDD ≡
0 in winter-only folds), the corresponding slope parameter is
unidentifiable, the design matrix is rank-deficient, and the
unconstrained optimiser wanders into popt with no physical meaning
(alpha at -7×10⁶ MW; Fourier coefficients in the tens of millions).  The
existing `pcov`-inf WARN fires correctly but does not prevent the model
from returning the diverged popt and the harness from scoring it.  On
the Stage 8 notebook's 30-day-window splitter this drags cross-fold
mean MAE from the median's ~4 850 MW to ~167 600 MW.

### Fix in one paragraph
Switch the solver to `method="trf"` (trust-region-reflective, scipy's
bounded least-squares algorithm) with physically-motivated bounds on
every free parameter.  Bounded TRF settles unidentifiable parameters at
the relevant bound rather than diverging.  The fit's behaviour on
well-conditioned data is unchanged within rtol < 1e-4; the change is
observable only on rank-deficient folds, where it converts catastrophic
divergence into the documented "parameter at a bound" failure mode.

## Decisions (binding)

D1.  Switch `curve_fit(..., method="lm")` → `method="trf"` with bounds.
     Unify both branches (currently separate for `loss="linear"` vs
     robust losses) under one `method="trf"` call.

D2.  Bounds (intent option A):

     | parameter | lower | upper |
     |---|---|---|
     | alpha | 0 | 100 000 |
     | beta_heat | 0 | 5 000 |
     | beta_cool | 0 | 5 000 |
     | each Fourier coefficient | -50 000 | 50 000 |

D3.  Bounds derived at fit time inside `fit()`; not exposed as
     `ScipyParametricConfig` fields (intent's "out of scope: no schema
     change").

D4.  Add `active_mask`-aware std-err override.  When TRF settles a
     parameter at a bound, set its std-err to `inf` (per domain research
     §4 — scipy's pseudo-inverse `pcov` is finite at a bound, so the
     existing finiteness WARN will not fire without the override).

D5.  Extend the existing `pcov`-inf WARN to **name affected parameters**
     (whether rank-deficient or bound-saturated).  Required by
     AC-1/AC-2's "WARN fires for `beta_cool` only" wording.

D6.  **CUT** per minimalist scope-guard.  No defensive `p0` nudge —
     `scipy>=1.13` includes the 1.11.3 fix for the `x0`-on-boundary
     regressions (#18793, #19103); the guard is insurance against a bug
     no permitted scipy version exhibits.

D7.  Re-record the popt golden vector in
     `test_scipy_parametric_fit_same_data_same_params`.  The test
     guards *cross-call determinism on the current solver*, not
     LM-vs-TRF parity; re-recording is correct, and bit-exact equality
     stays the determinism contract.

D8.  Revise the Cell 12 markdown in `scripts/_build_notebook_08.py`
     (the assumptions appendix that explicitly names `method="lm"`)
     and regenerate `notebooks/08_scipy_parametric.ipynb`.

D9.  Append "D6 (revisited)" addendum to
     `docs/lld/stages/08-scipy-parametric.md` documenting the reversal.

D10. Update `docs/architecture/layers/models.md` inventory row to drop
     the `method="lm"` claim.

D11. Update `src/bristol_ml/models/CLAUDE.md` to drop the `method="lm"`
     phrasing.

## Acceptance criteria

**AC-1 — winter-only window.**  On a synthetic 30-day training fixture
with all temperatures < `t_heat` (CDD ≡ 0):
- `popt[alpha]` ∈ [20 000, 50 000] MW.
- `popt[beta_cool] == 0.0` (at lower bound).
- A loguru WARNING fires naming `beta_cool`; no WARNING names
  `beta_heat`.

**AC-2 — summer-only window.**  Symmetric: `popt[beta_heat] == 0.0`,
WARN names `beta_heat` only.

**AC-3 — no regression on healthy full-year window.**  On the
project-default 1-year+ window: every entry of `popt` and the
cross-fold-mean MAE on the train CLI default match pre-change LM values
within rtol < 1e-4.  If empirically rtol > 1e-4 is observed (TRF and LM
use different termination criteria — see domain research §5), widen the
test tolerance, do not loosen the bound.

**AC-4 — cross-fold mean MAE recovery.**  On the Stage 8 notebook's
existing 30-day rolling splitter (unchanged), cross-fold mean MAE
< 6 000 MW (currently ~167 600).

**AC-5 — std errors at boundary.**  On AC-1/AC-2 fixtures, std-err for
the bound-saturated parameter is `inf`; all interior parameters' std-err
match pre-change values within rtol < 1e-4.

**AC-6 — robust losses unchanged.**  For `loss ∈ {linear, soft_l1,
huber, cauchy}` on a healthy window: `fit()` completes without
exception and `popt` is finite.

## Tasks

**T1 — `fit()` core change** (`src/bristol_ml/models/scipy_parametric.py`).
- Build a `bounds` 2-tuple at fit time using `_build_param_names()` and
  `cfg.diurnal_harmonics` / `cfg.weekly_harmonics`.
- Unify both branches under `method="trf"`; pass `loss=cfg.loss` and
  `max_nfev=cfg.max_iter` (drop the `maxfev=` LM spelling).
- Use `full_output=True` to retrieve `infodict["active_mask"]` (or
  `popt`-vs-bound comparison) for std-err override.
- Override `std_err[i] = inf` whenever `active_mask[i] != 0` (parameter
  at a bound).
- Update the WARN at lines 416–425 to name the parameters whose std-err
  ends up non-finite.
- Update the `fit()` docstring: `pcov` unreliability now has two causes
  — singular Jacobian and parameter at a bound.
- Acceptance: AC-1 / AC-2 / AC-5 / AC-6.

**T2 — re-record determinism golden vector**
(`tests/unit/models/test_scipy_parametric.py`).
- Run `test_scipy_parametric_fit_same_data_same_params` twice under the
  new solver.  The `assert_array_equal` guard should remain bit-exact
  cross-call.  If a hard-coded `expected_popt` array exists elsewhere in
  the test file, re-record it.
- No tolerance widening unless empirically required.
- Acceptance: existing determinism test stays green.

**T3 — new tests for AC-1, AC-2, AC-5, AC-6**
(`tests/unit/models/test_scipy_parametric.py`).
- Winter-only fixture: 720h, T ∈ [-5, 10] °C.  Assert popt and WARN per
  AC-1; assert std-err per AC-5.
- Summer-only fixture: 720h, T ∈ [25, 35] °C.  Assert per AC-2 / AC-5.
- AC-6 already partially exercised by
  `test_scipy_parametric_fit_loss_override_changes_fit`; add a
  `pytest.mark.parametrize` over all four loss values asserting
  `np.all(np.isfinite(popt))` on the healthy fixture.
- Acceptance: tests pass; new fixtures are deterministic.

**T4 — empirical AC-3 / AC-4 verification**.
- Run the Stage 8 notebook's `evaluate(...)` on the existing 30-day
  rolling splitter; assert cross-fold mean MAE < 6 000 (AC-4).
- Run `python -m bristol_ml.train model=scipy_parametric
  features=weather_calendar` on default config; capture popt and
  compare to a pre-recorded golden vector at rtol < 1e-4 (AC-3).  If
  rtol > 1e-4 is observed, document the empirical tolerance and widen
  the test.
- Add a test under `tests/integration/` asserting AC-3 (golden popt
  vector under default config).
- Acceptance: AC-3, AC-4 verified; integration test green.

**T5 — Stage 8 notebook update**
(`scripts/_build_notebook_08.py`, `notebooks/08_scipy_parametric.ipynb`).
- Revise the Cell 12 markdown to reflect `method="trf"` and acknowledge
  bound-saturation as the second cause of std-err = `inf`.
- Regenerate the notebook (`python scripts/_build_notebook_08.py`).
- Acceptance: builder text up-to-date; nb08 integration test stays
  green.

**T6 — documentation hygiene**.
- `docs/lld/stages/08-scipy-parametric.md`: append "D6 (revisited)"
  documenting the reversal — empirical evidence that the unbounded fit
  diverges on rank-deficient windows.
- `docs/architecture/layers/models.md`: update the `scipy_parametric.py`
  inventory row.
- `src/bristol_ml/models/CLAUDE.md`: drop the `method="lm"` phrasing.
- `CHANGELOG.md`: an entry under `[Unreleased] / Fixed` referring to
  the option-2(b) follow-up.
- Acceptance: spec-drift rule satisfied; no stale `method="lm"` claims.

## Out of scope (explicit)

- A second bounded-fit solver (`dogbox`).
- Changes to `ScipyParametricConfig`'s public shape (no new fields,
  including no `allow_negative_beta_cool` knob — intent option (B) is a
  notebook-text comment only, see OQ-2 in requirements §6).
- Bootstrap or quantile-based CI alternatives (deferred to Stage 10).
- Stage 16 re-baseline (Stage 16 has no ScipyParametric retraining
  target; baseline-MAE rows are advisory — see codebase map §4).

## Risks / open questions

- **R1.** TRF and LM use different termination criteria (domain
  research §5).  Healthy-fold popt may differ from LM at rtol > 1e-4.
  Mitigation: empirical verification in T4; widen test tolerance if
  needed.
- **R2.** The `_derive_p0` initial guess sometimes lands at zero (e.g.
  `beta_cool` p0 when no above-`t_cool` data).  Bound is also zero.
  Per domain research §4 and minimalist scope cut: the scipy version
  pin (`>=1.13`) covers this; if T1 testing surfaces the regression,
  reinstate the defensive nudge.
- **R3.** `infodict.nfev` semantics differ between LM and TRF; no test
  pins this so no impact, but the PR description should note it for
  reviewers.

## Exit checklist

- [ ] `uv run pytest tests/unit/models/test_scipy_parametric.py` green.
- [ ] `uv run pytest tests/integration/test_notebook_08.py` green.
- [ ] `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] `notebooks/08_scipy_parametric.ipynb` regenerated; cross-fold
      mean MAE < 6 000 MW visible in the executed cell output.
- [ ] AC-3 integration test green (popt rtol vs golden vector).
- [ ] Spec-drift hygiene: D6-revisited addendum, layer doc, module
      CLAUDE.md, CHANGELOG entry, all in.
- [ ] Phase 3 reviewer agents (`arch-reviewer`, `code-reviewer`,
      `docs-writer`) Blocking-tier findings addressed or deferred with
      explicit rationale.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/`.
