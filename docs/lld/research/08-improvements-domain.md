# Domain research — `scipy.optimize.curve_fit` LM → TRF migration for hinge regression

**Scope.** `scipy>=1.13,<2` (project pin from `pyproject.toml`).  All claims
below cite the scipy 1.13 docs (or 1.17 stable where unchanged).

## 1. Canonical sources

- [scipy.optimize.curve_fit — v1.13.1](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.curve_fit.html)
- [scipy.optimize.least_squares — v1.13.0](https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.optimize.least_squares.html)
- [curve_fit — v1.17.0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- [Issue #18793 — TRF wrong result when `x0` close to lower bound](https://github.com/scipy/scipy/issues/18793)
- [Issue #19103 — TRF regression in 1.11.2 when `x0` on boundary](https://github.com/scipy/scipy/issues/19103)

## 2. Recommended approach

Switch `method="lm"` to `method="trf"` and supply explicit
`bounds=([lb_0, ..., lb_N], [ub_0, ..., ub_N])`.  Mechanical justification:
`curve_fit(method="lm")` routes to `leastsq` (MINPACK), which "can only
deal with unconstrained problems"; `method="trf"` routes to
`least_squares`, which enforces box constraints throughout the iteration.
On rank-deficient training windows, TRF settles unidentifiable parameters
at the relevant bound rather than diverging to ±10⁷.

The `loss` switch is also mandatory for completeness: `method="lm"`
supports only `'linear'` loss; `method="trf"` supports `'linear'`,
`'soft_l1'`, `'huber'`, `'cauchy'`, and `'arctan'` — covering all four
values the existing `ScipyParametricConfig.loss` enum exposes.  `curve_fit`
passes `**kwargs` through to `least_squares` when method ≠ `'lm'`, so the
existing `loss=` kwarg in the call site needs no change in spelling, only
the method switch to become legal alongside non-linear losses.

## 3. Alternatives considered and rejected

- **`method="dogbox"`.** Also supports bounds and all loss functions, but
  the scipy docs describe it as "particularly useful with sparse Jacobians"
  and warn it "may cause spurious oscillations in rank-deficient problems."
  The hinge model is *deliberately* rank-deficient on degenerate windows;
  TRF's reflected search is better suited.
- **Clipping `popt` post-fit.** Not equivalent — LM still receives a
  rank-deficient Jacobian and may fail to converge (`ier ∉ {1,2,3,4}`)
  before producing any `popt`.
- **Adding an L2 penalty.** Changes the estimator semantics on healthy
  windows and complicates `pcov` interpretation without addressing the
  root cause of unidentifiability.

## 4. Pitfalls and edge cases

**`pcov` when parameters are at a bound.**  scipy docs:

> If the Jacobian matrix at the solution doesn't have a full rank, then
> 'lm' method returns a matrix filled with `np.inf`; 'trf' and 'dogbox'
> methods use Moore-Penrose pseudoinverse to compute the covariance
> matrix.

The pseudo-inverse path returns a *finite* `pcov` even for rank-deficient
problems, but the matrix has a large condition number and is statistically
unreliable.  The docs further note: "covariance matrices with large
condition numbers … may indicate that results are unreliable."

Implication for AC-1 / AC-2 ("`pcov`-inf WARN fires for `beta_cool` only"):
the existing WARN in `ScipyParametricModel.fit()` triggers on
`np.isnan(pcov_diag) | np.isinf(pcov_diag)`.  Under TRF, that condition
will *not* fire for a parameter at a bound, because the pseudo-inverse
returns a finite (but inflated) variance.  **The implementer must add an
explicit "parameter at active bound → std_err = ∞" code path** that
inspects either `result.active_mask` (from `least_squares` full output) or
the `infodict` returned by `curve_fit(..., full_output=True)`, and overrides
the std-err entries to `inf` for those parameters.  Without this, AC-1 and
AC-5 cannot be met.

**`infodict.nfev` is not comparable across solvers.**  For `method="trf"`,
`curve_fit` docs note: "trf and dogbox do not count function calls for
numerical Jacobian approximation, as opposed to lm method." Any test
that pins `nfev` will fail on the switch — none currently exists in
`tests/unit/models/test_scipy_parametric.py` (verify in codebase pass).

**`mesg` and `ier`.**  Still valid under TRF.  `ier` ∈ {1,2,3,4} indicates
success; TRF returns `ier=1` (typically) when a parameter settles at a
bound, with `mesg` describing the termination condition.  No new failure
modes from the existing convergence-check code path.

**`make_strictly_feasible` pitfall (#18793, #19103).**  If `x0[i]` is set
*exactly* equal to `lb[i]` or `ub[i]`, scipy 1.11.2 had two regressions
that caused TRF to mis-converge.  Both are fixed in 1.11.3+; the project
pins `scipy>=1.13`, so the pre-conditions are safe.  Defensive recipe
nonetheless: when the existing initial-guess construction in `fit()`
produces a value at zero (the natural `p0` for `beta_cool` etc.), nudge it
strictly interior — e.g. `p0[i] = max(p0[i], lb[i] + 1e-6)`.

## 5. Version, compatibility, and determinism

**Determinism.**  TRF is deterministic — `least_squares` consumes no
random seed.  Given identical `x0`, `bounds`, data, and tolerance kwargs,
repeated calls produce bit-identical results.  The "rtol < 1e-4 on healthy
paths" criterion is safe with respect to internal stochasticity.

**Tolerance defaults.**  Both LM and TRF share `ftol=1e-8`, `xtol=1e-8`,
`gtol=1e-8` numerically — but the termination conditions are *defined
differently*.  For TRF, `xtol` triggers when `‖dx‖ < xtol·(xtol + ‖x‖)`
(relative-absolute norm); for LM it triggers when `Δ < xtol·‖xs‖` (trust
radius vs scaled variables).  In practice, a healthy fit may converge at
a slightly different `popt` under TRF than under LM even with identical
`x0`, because the two algorithms take different step trajectories and
apply different stopping criteria.

**Implication for AC-3 ("rtol < 1e-4 vs pre-change values").**  This must
be verified empirically per-fixture, not assumed analytically.  Realistic
tolerance for healthy fits: rtol 1e-5 to 1e-3 on individual `popt` entries
(small-magnitude Fourier coefficients near zero may show looser relative
agreement; large-magnitude `alpha` should be tight).  If a healthy
fixture shows rtol > 1e-4, the *test* tolerance widens, not the bound.

**`bounds` argument shape.**  `curve_fit` accepts `bounds` as a 2-tuple of
array-like: `([lb_0, ..., lb_N-1], [ub_0, ..., ub_N-1])`.  Each entry is
either a scalar (broadcasts to all parameters) or an array of length
equal to the total free parameter count.  No structural distinction
between "scalar" parameters (alpha, beta_heat, beta_cool) and "vector"
parameters (the K Fourier coefficients) — all N appear as positional
entries at their respective indices.  Implementers must therefore know
the *exact* parameter order produced by the existing `_build_param_names`
helper to construct the bounds vectors correctly.

**scipy compatibility.**  All behaviour above is stable across
`scipy>=1.13,<2`.  The `loss` kwarg support for TRF has been present since
`least_squares` was introduced in scipy 0.17.  The Moore-Penrose `pcov`
path is long-standing.  No relevant deprecations announced in 1.13–1.17
changelogs.
