Stage 8 follow-up — bounded parametric fit (2b)

ScipyParametricModel.fit calls scipy.optimize.curve_fit(..., method="lm") — Levenberg-Marquardt, unconstrained. When the training window contains no observations on one side of a hinge (HDD ≡ 0 in summer-only folds, CDD ≡ 0 in winter-only folds), the corresponding slope parameter is unidentifiable, the design matrix is rank-deficient, and the unconstrained optimiser wanders into popt values with no physical meaning (alpha at -7×10⁶ MW or +4×10⁸ MW, Fourier coefficients in the tens of millions). The existing pcov=inf WARN log fires correctly but does not prevent the model from returning the diverged popt and the harness from scoring against it. On the Stage 8 notebook's 30-day-window splitter this gives ten divergent folds (out of fifty) with per-fold MAE in the hundreds of thousands to millions of MW, dragging the cross-fold mean from ~4 850 (the median, matching the CLI's full-year fit) up to ~167 600.

Proposed fix (2b). Switch the solver from method="lm" to method="trf" (trust-region-reflective, scipy's bounded least-squares algorithm) with physically-motivated bounds on every free parameter. With bounds in place an unidentifiable column can no longer push its parameter to ±∞; the
optimiser instead settles at the relevant bound and the existing pcov-inf diagnostic accurately flags "this parameter wasn't identifiable from this training window." The fit's behaviour on well-conditioned data is unchanged because TRF and LM agree on smooth interior optima; the change is
observable only on rank-deficient folds, where it converts catastrophic numerical divergence into the documented "parameter at a bound" failure mode.

Bounds :

┌──────────────────────┬─────────┬─────────────┬─────────────────────────────────────────────────────────────────┐
│      parameter       │  lower  │    upper    │                            rationale                            │
├──────────────────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────────┤
│ alpha                │       0 │  100 000 MW │ non-negative; GB peak demand has never exceeded ~62 GW          │
├──────────────────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────────┤
│ beta_heat            │       0 │ 5 000 MW/°C │ demand rises with HDD; "negative heating slope" is non-physical │
├──────────────────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────────┤
│ beta_cool            │       0 │ 5 000 MW/°C │ symmetric to beta_heat                                          │
├──────────────────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────────┤
│ Fourier coefficients │ −50 000 │  +50 000 MW │ the periodic signal cannot exceed national demand magnitudes    │
└──────────────────────┴─────────┴─────────────┴─────────────────────────────────────────────────────────────────┘

These bounds are loose enough that every well-conditioned fit on a project-default training window stays interior (typical fitted values: alpha ≈ 26 000, beta_heat ≈ 760, beta_cool ≈ -595). The bound on beta_cool is the only point of real architectural debate — historic GB demand in
extreme heatwaves might fit a slightly negative beta_cool (a noise artefact rather than a physical effect), and pinning ≥ 0 changes that. Two reasonable resolutions:

- (A) keep the symmetric ≥ 0 floor and treat slight negative cooling slopes as "rounded to zero — physically unsupported anyway"; the notebook's Cell 12 assumptions appendix already flags this; or
- (B) allow beta_cool ∈ [-1 000, 5 000] so the fit can exhibit the noise-fit behaviour for pedagogical honesty.

Use (A) for the shipped default with (B) documented as an override configuration for should it be necessary to demo the artefact.

Acceptance criteria sketch.

1. On a winter-only 30-day training window (CDD ≡ 0), fit() returns popt with alpha in [20 000, 50 000] MW and beta_cool == 0 (at the bound). The pcov=inf WARN fires for beta_cool only.
2. On a summer-only 30-day training window (HDD ≡ 0), symmetric: beta_heat == 0, WARN fires for beta_heat only.
3. On the project default 1-year+ training window, fitted popt matches the pre-change values within rtol < 1e-4 for every parameter — no regression on healthy paths. Specifically alpha, beta_heat, beta_cool and the cross-fold-mean MAE on the train-CLI default config stay at their current
values to within rounding.
4. The Stage 8 notebook's cross-fold mean MAE on the existing 30-day-window splitter is within ±20 % of the median (currently 35× the median). Concretely: mean MAE drops from ~167 600 to <6 000 with the splitter unchanged.
5. Standard errors sqrt(diag(pcov)) for parameters that hit a bound are reported as inf (the Gaussian-CI assumption is invalid at a bound — Cell 12's existing language). For interior parameters, std_err is finite and matches the pre-change values to within rtol < 1e-4.
6. The loss="linear" / loss="soft_l1" / loss="huber" / loss="cauchy" config knob continues to work — TRF supports all four; the relationship to robust losses is unchanged.

Out of scope, explicitly deferred.

- A new bounded-fit solver alternative (e.g. dogbox). TRF is the documented default in modern scipy; no second algorithm is needed.
- Updating the parameter names or the public ScipyParametricConfig shape. Bounds are private to fit().

Risks / open questions.

- The pre-change tests pin specific popt values for at least the test_scipy_parametric_fit_same_data_same_params determinism guard. Bound-respecting TRF should give the same popt on healthy fits but the test tolerance may need widening from atol=0 to atol=1e-4 * value — verify before
landing.
- Stage 16 (model-with-REMIT) plan Phase 1 mentioned ScipyParametric as a candidate retraining target. If 2b changes the popt magnitudes on any fold, downstream Stage 16 retros may need a re-baseline. Worth checking on the Phase 3 review.
- The pcov returned by TRF when a parameter is at a bound has documented "may be unreliable" semantics in scipy — already covered by the existing WARN; no new code needed but the docstring should be updated to note the second cause (boundary, not just singular Jacobian).

Dependencies. No new runtime dependencies — TRF ships with scipy, already pinned. Existing pcov=inf plumbing stays unchanged.