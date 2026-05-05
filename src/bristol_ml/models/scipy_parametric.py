"""SciPy parametric load model — Stage 8.

Follows the Stage 4 :class:`bristol_ml.models.Model` protocol.  Fits a
**piecewise-linear temperature response plus diurnal and weekly Fourier
harmonic** decomposition of demand, using
:func:`scipy.optimize.curve_fit` for the nonlinear least-squares solve.
The output is a small set of interpretable coefficients (X MW per degree
of heating, Y MW per degree of cooling, Fourier amplitudes for the daily
and weekly cycles) each with a Gaussian confidence interval derived from
the covariance matrix ``pcov`` returned by ``curve_fit``.

The model is deliberately simpler than Stages 4, 7 and 10+ — its
pedagogical role is to demonstrate *interpretability*, not peak accuracy.
See ``docs/lld/research/08-scipy-parametric-domain.md`` §R10 for a
comparison against OLS / SARIMAX / gradient boosting.

**Design headline** (plan §1 D1..D6):

- ``f(T, t; theta)`` = ``alpha + b_heat * max(0, T_heat - T) + b_cool * max(0, T - T_cool)``
  ``+ sum_k [A_k sin(w_d * k * t) + B_k cos(w_d * k * t)]``
  ``+ sum_j [C_j sin(w_w * j * t) + D_j cos(w_w * j * t)]``
- ``T_heat = 15.5 °C``, ``T_cool = 22.0 °C`` are **fixed** (Elexon
  convention; plan D1 — eliminates the largest identifiability risk).
- Diurnal harmonics ``k = 1..K_d`` at period 24 h; weekly harmonics
  ``j = 1..K_w`` at period 168 h (plan D2 defaults ``K_d=3, K_w=2``).
- Design matrix is **temperature plus Fourier only** (plan D2
  clarification — Stage 5 calendar one-hots are excluded to avoid
  partial collinearity with the weekly Fourier terms).
- ``method="trf"`` Trust-Region Reflective with **physically-motivated
  bounds** on every free parameter (plan 08a D1/D2 — the bounded fit
  follow-up).  TRF settles unidentifiable parameters at the relevant
  bound rather than diverging into non-physical popt on rank-deficient
  training windows (e.g. ``CDD ≡ 0`` on a winter-only fold).  Bounds
  are derived at fit time from the parameter count; they are not
  exposed on :class:`~conf._schemas.ScipyParametricConfig`.
- Bounds (intent option A): ``alpha ∈ [0, 100 000]``;
  ``beta_heat, beta_cool ∈ [0, 5 000]``; each Fourier coefficient
  ``∈ [-50 000, +50 000]``.  Wide enough that every healthy fit on a
  project-default training window stays interior; the change is
  observable only on rank-deficient folds where the unbounded LM
  predecessor diverged.
- ``loss="linear"`` keeps ``pcov → CI`` Gaussian and rigorous (plan
  Stage 8 D3/D5).  Non-linear losses are available via config override
  but turn ``pcov`` into a heuristic (the notebook's appendix cell
  spells this out).  All four loss values run under the same TRF call;
  the legacy LM path no longer exists.

**Pickleability** (codebase surprise S2).  ``curve_fit`` internally
holds a reference to the target function; joblib / pickle must
therefore be able to pickle that reference.  :func:`_parametric_fn` is
defined at module level (*not* as a closure inside :meth:`fit`, *not* as
a lambda, *not* as a bound method) so the whole ``ScipyParametricModel``
round-trips cleanly.

Running standalone::

    python -m bristol_ml.models.scipy_parametric --help
    python -m bristol_ml.models.scipy_parametric           # prints config schema
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import OptimizeWarning, curve_fit

from bristol_ml.features.fourier import append_weekly_fourier
from bristol_ml.models.io import load_skops, save_skops
from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import ScipyParametricConfig

__all__ = ["ScipyParametricModel"]

# Format tag for the dict-envelope-of-primitives written by
# :meth:`ScipyParametricModel.save`.  Bumped if the envelope schema
# changes; load rejects any tag it does not recognise so a future
# schema migration is never silent.
_SCIPY_PARAMETRIC_ENVELOPE_FORMAT = "scipy-parametric-state-v1"


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------
#
# Every callable below is intentionally module-level so that ``pickle`` (and
# therefore ``joblib``) can serialise references to it.  A closure inside
# :meth:`ScipyParametricModel.fit`, a lambda, or a bound method would not
# round-trip through :func:`save_joblib` (codebase research §6 + S2).
# Keep them pure (no I/O, no globals that mutate) so the unit tests can
# exercise them directly and the demo-time behaviour matches.


def _build_param_names(*, diurnal_harmonics: int, weekly_harmonics: int) -> tuple[str, ...]:
    """Return the ordered parameter names for :func:`_parametric_fn`.

    The order is **load-bearing**: :func:`_parametric_fn` interprets its
    ``*params`` positionally in the same order as this tuple and
    :meth:`ScipyParametricModel.fit` constructs the design matrix in the
    same column order.  Three naming conventions are baked in here:

    - ``alpha`` — constant base-load offset (MW).
    - ``beta_heat`` / ``beta_cool`` — piecewise-linear slopes (MW/°C).
      Domain sign convention: ``beta_heat > 0`` means colder weather
      raises demand (standard for GB heating load); ``beta_cool > 0``
      means hotter weather raises demand (cooling).
    - Fourier pairs are named ``{prefix}_sin_k{k}`` / ``{prefix}_cos_k{k}``
      with ``prefix ∈ {diurnal, weekly}`` — matches the column naming
      convention of :func:`~bristol_ml.features.fourier.append_weekly_fourier`.
    """
    names: list[str] = ["alpha", "beta_heat", "beta_cool"]
    for k in range(1, diurnal_harmonics + 1):
        names.append(f"diurnal_sin_k{k}")
        names.append(f"diurnal_cos_k{k}")
    for j in range(1, weekly_harmonics + 1):
        names.append(f"weekly_sin_k{j}")
        names.append(f"weekly_cos_k{j}")
    return tuple(names)


def _parametric_fn(X: np.ndarray, *params: float) -> np.ndarray:
    """Evaluate the Stage 8 parametric load model on a design matrix.

    Parameters
    ----------
    X
        Float ndarray of shape ``(n_features, n_obs)`` (i.e. transposed;
        this matches ``scipy.optimize.curve_fit``'s calling convention
        where the independent variable is passed column-by-column).
        Row layout (load-bearing):

        - row 0: HDD, i.e. ``max(0, T_heat - T)``.  Pre-computed by
          :meth:`ScipyParametricModel._build_design_matrix` from the
          raw temperature column and the fixed ``T_heat`` from config.
        - row 1: CDD, i.e. ``max(0, T - T_cool)``.  Pre-computed
          symmetrically.
        - rows 2..: diurnal sin/cos pairs then weekly sin/cos pairs,
          in the order emitted by
          :func:`~bristol_ml.features.fourier.append_weekly_fourier`
          (``diurnal_sin_k1, diurnal_cos_k1, ..., weekly_sin_k1,
          weekly_cos_k1, ...``).

        Keeping the hinge transformation in the caller (not here) makes
        this function pure — no config dependency — which is essential
        for the pickle round-trip requirement (codebase surprise S2).

    *params
        Parameter vector in the order returned by
        :func:`_build_param_names`: ``(alpha, beta_heat, beta_cool,
        diurnal_sin_k1, diurnal_cos_k1, ..., weekly_sin_k1,
        weekly_cos_k1, ...)``.

    Returns
    -------
    ndarray
        Predicted demand, shape ``(n_obs,)``.
    """
    # params[0] = alpha; params[1] = beta_heat; params[2] = beta_cool; the
    # remaining entries are sin/cos coefficients, in the order laid out
    # by ``_build_param_names``.
    alpha = params[0]
    beta_heat = params[1]
    beta_cool = params[2]

    # Row 0 = hdd (pre-computed); row 1 = cdd (pre-computed); rows 2.. =
    # diurnal sin/cos pairs then weekly sin/cos pairs.  The caller
    # (:meth:`_build_design_matrix`) guarantees this layout — see below.
    hdd = X[0]
    cdd = X[1]
    fourier_rows = X[2:]

    y = alpha + beta_heat * hdd + beta_cool * cdd
    # Fourier coefficients live at params[3:]; they multiply the
    # remaining rows one-for-one.
    fourier_coeffs = params[3:]
    if fourier_coeffs:
        # numpy broadcasting: ``fourier_rows`` has shape
        # ``(n_fourier, n_obs)`` and the coefficient vector has shape
        # ``(n_fourier,)``.  ``np.tensordot`` / ``@`` returns ``(n_obs,)``.
        coeff_arr = np.asarray(fourier_coeffs, dtype=np.float64)
        y = y + coeff_arr @ fourier_rows
    return np.asarray(y, dtype=np.float64)


# Parameter bounds for the bounded TRF fit (plan 08a D2).  Order matches
# :func:`_build_param_names`: ``alpha, beta_heat, beta_cool`` then the
# Fourier sin/cos pairs.  Bounds are physically motivated:
#
# - ``alpha`` non-negative; GB peak demand has never exceeded ~62 GW so
#   100 000 MW is a generous upper bound.
# - ``beta_heat, beta_cool`` non-negative ("colder/hotter raises demand"
#   sign convention); 5 000 MW/°C is well above any plausibly fitted slope.
# - Fourier coefficients within ±50 000 MW — the periodic signal cannot
#   plausibly exceed national demand magnitudes.
#
# These bounds are *deliberately wide* — every well-conditioned fit on
# a project-default 1-year+ training window stays interior (typical
# fitted values: alpha ≈ 26 000, beta_heat ≈ 760, beta_cool ≈ small).
# The change is observable only on rank-deficient folds, where it
# converts catastrophic divergence into the documented "parameter at a
# bound" failure mode.
_BOUND_ALPHA = (0.0, 100_000.0)
_BOUND_BETA_HEAT = (0.0, 5_000.0)
_BOUND_BETA_COOL = (0.0, 5_000.0)
_BOUND_FOURIER = (-50_000.0, 50_000.0)


def _build_bounds(
    *,
    diurnal_harmonics: int,
    weekly_harmonics: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` bound arrays in :func:`_build_param_names` order.

    Total length is ``3 + 2*diurnal_harmonics + 2*weekly_harmonics``.
    Each entry is a finite ``float64`` — TRF's bound-handling does not
    permit ``np.inf`` here without losing the rank-deficiency safety net.
    """
    n_fourier = 2 * diurnal_harmonics + 2 * weekly_harmonics
    lower = np.empty(3 + n_fourier, dtype=np.float64)
    upper = np.empty(3 + n_fourier, dtype=np.float64)
    lower[0], upper[0] = _BOUND_ALPHA
    lower[1], upper[1] = _BOUND_BETA_HEAT
    lower[2], upper[2] = _BOUND_BETA_COOL
    if n_fourier:
        lower[3:] = _BOUND_FOURIER[0]
        upper[3:] = _BOUND_FOURIER[1]
    return lower, upper


def _detect_at_bound(
    popt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    rtol: float = 1e-6,
) -> np.ndarray:
    """Return a boolean mask flagging parameters TRF settled at a bound.

    Test: ``|popt - bound| <= rtol * (upper - lower)`` against either
    side.  The bound-width-relative tolerance is robust when the bound
    is exactly zero (where any rtol-on-bound check collapses to an
    absolute zero tolerance).  For the project bounds the resulting
    absolute tolerances are: alpha 0.1 MW; beta_heat / beta_cool 0.005
    MW/°C; Fourier coefficients 0.1 MW — tight enough that a genuinely
    interior optimum is not false-flagged (a real fitted slope of
    0.5 MW/°C is correctly classified as interior, well above the
    0.005 threshold), loose enough that ``popt = 1e-7`` next to
    ``lb = 0`` from TRF's ``make_strictly_feasible`` perturbation is
    detected.

    Note: ``rtol = 1e-6`` is *for the at-bound detection itself*, not
    for any external tolerance on a healthy fit.  A parameter that the
    optimiser genuinely settled at a bound (because the data pulled it
    against the bound) sits within this tolerance; a parameter the
    optimiser left interior because the data permitted any value (zero-
    information case) is caught separately by
    :func:`_detect_zero_information_columns`.
    """
    width = upper - lower
    tol = rtol * np.maximum(width, np.abs(lower) + np.abs(upper) + 1.0)
    at_lower = np.abs(popt - lower) <= tol
    at_upper = np.abs(popt - upper) <= tol
    return at_lower | at_upper


def _detect_implausible_variance(
    pcov_diag: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Flag parameters whose ``pcov`` diagonal exceeds the bound-width sanity check.

    Under bounded TRF, ``curve_fit`` uses the Moore-Penrose pseudo-inverse
    to compute ``pcov`` on rank-deficient problems (scipy docs:
    "covariance matrices with large condition numbers ... may indicate
    that results are unreliable").  The pseudo-inverse can return a
    *finite* but enormous diagonal — far larger than the parameter's
    permissible range — even when no parameter is clamped at a bound.

    Threshold: implausible if ``pcov_diag[i] > ((upper[i] - lower[i]) / 2)**2``,
    i.e. the implied 1-sigma exceeds half the bound width.  At that
    point any Gaussian CI is wider than the bound itself and the
    metadata's "value +/- 1.96 * std" rendering would be physically
    nonsense; better to surface the rank deficiency via the existing
    ``inf`` semantics.
    """
    half_width = (upper - lower) / 2.0
    return pcov_diag > half_width * half_width


def _detect_zero_information_columns(
    design_matrix: np.ndarray,
    n_params: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Flag parameters whose Jacobian column carries no information.

    For the Stage 8 parametric model the Jacobian wrt parameter ``i``
    is, for ``i >= 1``, the design-matrix row ``i - 1`` (parameter 0 is
    ``alpha``, whose Jacobian is the constant ``1`` and is always
    full-rank).  When that row is identically constant — most notably
    when ``CDD ≡ 0`` on a winter-only window or ``HDD ≡ 0`` on a
    summer-only one — the corresponding parameter is unidentifiable
    in the strict mathematical sense: the gradient w.r.t. it is zero,
    every value of the parameter produces the same residuals, and TRF
    cannot move the iterate from its initial guess regardless of
    direction.  In that case any standard error reported by the
    pseudo-inverse pcov is meaningless (Moore-Penrose collapses the
    zero-gradient direction to an artificially tight variance).

    Tested via ``np.std(row) < eps`` — captures both the all-zero case
    and the all-constant-nonzero case, both of which are
    rank-deficient with respect to the corresponding parameter.

    Returns a length-``n_params`` boolean mask; entry 0 (alpha) is
    always ``False`` because ``alpha``'s Jacobian column is the
    constant ``1`` and is implicit, not stored as a design-matrix row.
    """
    flags = np.zeros(n_params, dtype=bool)
    n_rows = design_matrix.shape[0]
    for i in range(1, n_params):
        row_idx = i - 1
        if row_idx >= n_rows:
            continue
        if np.std(design_matrix[row_idx]) < eps:
            flags[i] = True
    return flags


def _derive_p0(
    *,
    target: pd.Series,
    temperature: pd.Series,
    t_heat: float,
    t_cool: float,
    diurnal_harmonics: int,
    weekly_harmonics: int,
) -> np.ndarray:
    """Derive a deterministic data-driven starting point (plan D4).

    Initialisation strategy:

    - ``alpha_0`` = mean demand across the training window.
    - ``beta_heat_0`` derived from a coarse HDD regression: the slope
      needed for demand at a representative cold point (observed
      minimum temperature) to match the observed mean demand at that
      point.  Reduces to a sensible non-zero value provided at least
      a handful of sub-``t_heat`` observations exist.
    - ``beta_cool_0`` derived symmetrically from observed cooling-side
      data.  When the training window has no observations above
      ``t_cool`` the slope is initialised to ``0.0`` (domain §R5:
      avoids putting the optimiser far from a sensible value on
      winter-only folds).
    - All Fourier coefficients are initialised to ``0.0`` — the
      optimiser shape of the problem (orthogonal Fourier basis) makes
      this the standard choice and keeps the initial prediction equal
      to ``alpha_0`` (a flat line) so the fit converges towards the
      dominant temperature response first.

    Returns a contiguous ``float64`` ndarray of length
    ``3 + 2*diurnal_harmonics + 2*weekly_harmonics``, matching the
    :func:`_parametric_fn` parameter count.
    """
    n_fourier = 2 * diurnal_harmonics + 2 * weekly_harmonics
    p0 = np.zeros(3 + n_fourier, dtype=np.float64)

    target_arr = np.asarray(target, dtype=np.float64)
    temp_arr = np.asarray(temperature, dtype=np.float64)
    alpha_0 = float(np.mean(target_arr))
    p0[0] = alpha_0

    # Heating-side slope: pick the coolest 10 % of rows to anchor the
    # slope estimate.  A small guard for training windows that contain
    # no sub-``t_heat`` data (unusual — GB winters always drop below
    # 15.5 °C — but possible for a three-month summer-only fold).
    cold_mask = temp_arr < t_heat
    if cold_mask.any():
        cold_temps = temp_arr[cold_mask]
        cold_demand = target_arr[cold_mask]
        hdd_cold = t_heat - cold_temps
        demand_anomaly = cold_demand - alpha_0
        # Least-squares slope of (hdd, demand_anomaly).  Denominator
        # guard: ``hdd_cold.sum()`` is > 0 because every element is > 0.
        denom = float(np.dot(hdd_cold, hdd_cold))
        p0[1] = float(np.dot(hdd_cold, demand_anomaly)) / denom if denom > 0.0 else 0.0
    else:
        p0[1] = 0.0

    # Cooling-side slope.  GB demand has a weak cooling signal compared
    # to heating, so the slope is often near zero, but the data-driven
    # derivation at least reflects the sign correctly.
    hot_mask = temp_arr > t_cool
    if hot_mask.any():
        hot_temps = temp_arr[hot_mask]
        hot_demand = target_arr[hot_mask]
        cdd_hot = hot_temps - t_cool
        demand_anomaly = hot_demand - alpha_0
        denom = float(np.dot(cdd_hot, cdd_hot))
        p0[2] = float(np.dot(cdd_hot, demand_anomaly)) / denom if denom > 0.0 else 0.0
    else:
        p0[2] = 0.0

    # Fourier coefficients left at 0.0.
    return p0


# ---------------------------------------------------------------------------
# ``ScipyParametricModel`` — the public class
# ---------------------------------------------------------------------------


class ScipyParametricModel:
    """SciPy parametric load model with interpretable coefficients + CIs.

    Implements :class:`bristol_ml.models.Model`.  Stores the fitted
    parameter vector ``_popt`` and covariance matrix ``_pcov`` so the
    Stage 8 notebook can render a "parameter +/- 1.96 * std" table directly
    (plan §Task T5 cell 7 / AC-3 / AC-5).

    Implementation landed across Tasks T2 (module-level helpers +
    ``_parametric_fn``), T3 (scaffold + ``metadata`` + ``_cli_main``),
    T4 (``fit`` / ``predict``) and T5 (``save`` / ``load`` + notebook).
    The Stage 8 plan's codebase surprises are covered by regression
    tests in ``tests/unit/models/test_scipy_parametric.py``.
    """

    def __init__(self, config: ScipyParametricConfig) -> None:
        """Store ``config`` and initialise empty fit-state.

        Parameters
        ----------
        config:
            Validated :class:`~conf._schemas.ScipyParametricConfig` (the
            Pydantic model is ``frozen=True`` so the reference is shared
            safely).
        """
        self._config: ScipyParametricConfig = config
        # Populated on fit().
        self._popt: np.ndarray | None = None
        self._pcov: np.ndarray | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None
        self._param_names: tuple[str, ...] = ()

    # ---------------------------------------------------------------------
    # Protocol members
    # ---------------------------------------------------------------------

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit the parametric model via :func:`scipy.optimize.curve_fit`.

        The fit path composes, in order:

        1. Length-parity and index-timezone guards (plan D8 —
           :meth:`_require_utc_datetimeindex` mirrors SARIMAX).
        2. Fourier-column append: diurnal at period 24 h with
           ``column_prefix="diurnal"``, then weekly at period 168 h with
           ``column_prefix="weekly"`` (plan D2).  Uses
           :func:`~bristol_ml.features.fourier.append_weekly_fourier`
           verbatim — no new helper needed.
        3. Design-matrix construction (temperature + Fourier only; plan
           D2 clarification).  Stage 5 calendar one-hots are
           deliberately excluded even when ``config.feature_columns is
           None``.
        4. ``p0`` derivation (plan D4): data-driven inside
           :func:`_derive_p0` unless an explicit ``config.p0`` is set.
        5. :func:`scipy.optimize.curve_fit` with ``method="trf"``,
           bounded by :func:`_build_bounds` (plan 08a D1/D2), with
           ``loss=cfg.loss`` and ``max_nfev=cfg.max_iter``.
           ``OptimizeWarning`` (e.g.  rank-deficient Jacobian → ``pcov``
           full of ``inf``) is captured and re-emitted at ``loguru``
           WARN level.
        6. Bound-saturation override (plan 08a D4).  TRF returns a
           Moore-Penrose pseudo-inverse ``pcov`` even at a bound, so the
           existing ``np.isfinite`` finiteness check would otherwise
           miss "parameter unidentifiable, settled at a bound".
           :func:`_detect_at_bound` flags every popt entry that TRF
           clamped; the matching ``pcov`` diagonal entries are
           overwritten with ``np.inf`` so downstream std-err and CI
           rendering treats them as unreliable.
        7. ``pcov``-inf post-check (NFR-4): a dedicated WARNING is
           emitted when any diagonal entry is non-finite, *naming the
           affected parameters* — whether the cause is rank deficiency
           or bound saturation.

        Re-calling :meth:`fit` discards the previous parameter vector,
        covariance matrix, and param-name tuple entirely (NFR-5).

        Raises
        ------
        ValueError
            If ``len(features) != len(target)``; the index is not a
            tz-aware UTC :class:`~pandas.DatetimeIndex`; the configured
            temperature column is missing; or the configured ``p0``
            length does not match ``3 + 2*diurnal + 2*weekly``.
        """
        # --- 1. Guards --------------------------------------------------
        if len(features) != len(target):
            raise ValueError(
                "ScipyParametricModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        self._require_utc_datetimeindex(features, method="fit")

        cfg = self._config
        if cfg.temperature_column not in features.columns:
            raise ValueError(
                f"ScipyParametricModel.fit: temperature_column "
                f"{cfg.temperature_column!r} not present in features frame "
                f"(columns: {list(features.columns)})."
            )

        # --- 2. Fourier append (D2) -------------------------------------
        features_with_fourier = self._append_fourier_columns(features)

        # --- 3. Design matrix (D2 clarification: temperature + Fourier) -
        design_matrix, fourier_cols = self._build_design_matrix(features_with_fourier)

        # --- 4. Initial parameter vector (D4) ---------------------------
        expected_n_params = 3 + 2 * cfg.diurnal_harmonics + 2 * cfg.weekly_harmonics
        if cfg.p0 is not None:
            if len(cfg.p0) != expected_n_params:
                raise ValueError(
                    f"ScipyParametricModel.fit: config.p0 has length {len(cfg.p0)} "
                    f"but the model has {expected_n_params} free parameters "
                    f"(3 + 2*diurnal[{cfg.diurnal_harmonics}] + "
                    f"2*weekly[{cfg.weekly_harmonics}])."
                )
            p0 = np.asarray(cfg.p0, dtype=np.float64)
        else:
            p0 = _derive_p0(
                target=target,
                temperature=features[cfg.temperature_column],
                t_heat=cfg.t_heat_celsius,
                t_cool=cfg.t_cool_celsius,
                diurnal_harmonics=cfg.diurnal_harmonics,
                weekly_harmonics=cfg.weekly_harmonics,
            )

        # --- 5. curve_fit ----------------------------------------------
        # Plan 08a D1: bounded TRF.  ``method="trf"`` accepts ``loss=`` for
        # all four supported values (linear, soft_l1, huber, cauchy) and
        # uses ``max_nfev`` as the iteration budget.  The bounded fit
        # converts the rank-deficient-fold divergence (alpha at -7e6 MW
        # under unbounded LM) into the documented "parameter at a bound"
        # failure mode, surfaced via the bound-saturation override below.
        # The Gaussian-CI reasoning in Cell 12 of the notebook only
        # holds for ``loss="linear"`` and only when no parameter is at
        # a bound; both caveats are now explicit in the appendix.
        target_arr = np.asarray(target, dtype=np.float64)
        param_names = _build_param_names(
            diurnal_harmonics=cfg.diurnal_harmonics,
            weekly_harmonics=cfg.weekly_harmonics,
        )
        bounds_lower, bounds_upper = _build_bounds(
            diurnal_harmonics=cfg.diurnal_harmonics,
            weekly_harmonics=cfg.weekly_harmonics,
        )
        # Guard against ``p0`` lying outside the bounds — possible only
        # for an explicit ``config.p0`` override; the data-driven
        # ``_derive_p0`` always produces non-negative slope estimates
        # (its slope formula divides positive-by-positive sums) and a
        # mean-of-target alpha well inside [0, 100_000].  Clipping
        # rather than raising keeps the override ergonomic; an out-of-
        # bound config value is treated as a request for "as close to
        # the bound as possible" rather than an error.
        p0 = np.clip(p0, bounds_lower, bounds_upper)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            popt, pcov = curve_fit(
                _parametric_fn,
                design_matrix,
                target_arr,
                p0=p0,
                method="trf",
                bounds=(bounds_lower, bounds_upper),
                loss=cfg.loss,
                max_nfev=cfg.max_iter,
            )

        for w in caught:
            if issubclass(w.category, OptimizeWarning):
                logger.warning(
                    "ScipyParametricModel.fit: scipy OptimizeWarning "
                    "(covariance may be unreliable — plan NFR-4): {}",
                    str(w.message),
                )

        # --- 6. Pcov unreliability overrides (plan 08a D4) --------------
        # Three cases that the raw ``np.isfinite`` check would otherwise
        # miss under bounded TRF:
        #
        # (a) Bound saturation.  TRF uses a Moore-Penrose pseudo-inverse
        #     for ``pcov`` on rank-deficient problems, so a parameter
        #     clamped at a bound would read as having a *finite*
        #     (inflated) covariance — the existing finiteness WARN
        #     would not fire.  Detect via :func:`_detect_at_bound`.
        # (b) Implausibly large pseudo-inverse variance.  A near-
        #     singular Jacobian (e.g. 50-row 43-parameter near-
        #     degenerate-temperature fixture) produces a finite-but-
        #     enormous diagonal (~1e14).  At that point the implied
        #     Gaussian CI is wider than the bound itself; surfacing
        #     as ``inf`` is more honest than publishing a meaningless
        #     number.  Detect via :func:`_detect_implausible_variance`.
        # (c) Zero-information Jacobian column.  When CDD ≡ 0 on a
        #     winter-only fold (or HDD ≡ 0 on a summer-only fold), the
        #     corresponding parameter has gradient identically zero;
        #     TRF cannot move the iterate from its starting point and
        #     ``popt`` is determined by initialisation rather than the
        #     data.  Moore-Penrose collapses the zero-gradient
        #     direction to an artificially tight variance (e.g.
        #     ~1e-15) — the *opposite* of the (b) case but equally
        #     meaningless.  Detect by inspecting the design-matrix
        #     row variance: see :func:`_detect_zero_information_columns`.
        popt_arr = np.asarray(popt, dtype=np.float64)
        pcov_arr = np.asarray(pcov, dtype=np.float64).copy()
        # ``np.diag`` on a 2-D array returns a *view*; ``.copy()`` here
        # is defensive — a future refactor that moves the implausible-
        # variance computation below the diagonal-override loop would
        # otherwise see infinities and silently mis-classify.
        diag_raw = np.diag(pcov_arr).copy()
        at_bound = _detect_at_bound(popt_arr, bounds_lower, bounds_upper)
        implausible = _detect_implausible_variance(diag_raw, bounds_lower, bounds_upper)
        zero_info = _detect_zero_information_columns(design_matrix, n_params=popt_arr.shape[0])
        unreliable = at_bound | implausible | zero_info
        if unreliable.any():
            # Set the entire row and column for every unreliable
            # parameter to ``np.inf`` — leaving the off-diagonal entries
            # at their TRF pseudo-inverse values would publish a
            # numerically-incoherent covariance matrix (a row/column of
            # finite off-diagonal values next to an infinite diagonal
            # is not positive-semidefinite).  Downstream consumers of
            # ``metadata.hyperparameters["covariance_matrix"]`` then
            # see a self-consistent "this parameter's covariance is
            # unreliable in every direction" signal rather than a
            # silently-broken matrix.
            for idx in np.flatnonzero(unreliable):
                pcov_arr[idx, :] = np.inf
                pcov_arr[:, idx] = np.inf
        # Zero-information parameters have a gradient identically zero,
        # so TRF's final ``popt`` for them is determined by
        # ``make_strictly_feasible``'s tiny perturbation off the initial
        # guess, not by the data.  Clamp to ``0`` (the maximally
        # physically-sensible "no effect" default; within the bounds
        # for every parameter — slopes' lower bound is ``0``, Fourier
        # bounds straddle zero) so downstream ``predict()`` does not
        # propagate an arbitrary perturbation as a nominal contribution.
        if zero_info.any():
            for idx in np.flatnonzero(zero_info):
                popt_arr[idx] = float(np.clip(0.0, bounds_lower[idx], bounds_upper[idx]))

        # --- 7. pcov-inf post-check (NFR-4) -----------------------------
        diag = np.diag(pcov_arr)
        non_finite_mask = ~np.isfinite(diag)
        if non_finite_mask.any():
            affected = [param_names[i] for i in np.flatnonzero(non_finite_mask)]
            logger.warning(
                "ScipyParametricModel.fit: pcov diagonal contains "
                "non-finite entries for parameters {} — parameter "
                "identifiability is degraded on this fold (rank-deficient "
                "Jacobian or parameter at a bound; see plan 08a D4). "
                "CIs for affected parameters will be reported as +/- inf.",
                affected,
            )

        # --- 8. Publish state -------------------------------------------
        self._popt = popt_arr
        self._pcov = pcov_arr
        # Deliberately store the design-matrix column order (temperature
        # placeholder + Fourier columns) as feature_columns so the
        # metadata is a faithful record of what curve_fit saw.  The
        # first slot is named ``hdd_cdd`` to document that the raw
        # temperature column was pre-transformed into HDD/CDD rows.
        self._feature_columns = ("hdd", "cdd", *fourier_cols)
        self._fit_utc = datetime.now(UTC)
        self._param_names = param_names

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return parametric-model predictions indexed to ``features.index``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If the features index is not a tz-aware UTC ``DatetimeIndex``
            or the configured temperature column is absent.
        """
        if self._popt is None:
            raise RuntimeError("ScipyParametricModel must be fit() before predict().")
        self._require_utc_datetimeindex(features, method="predict")

        cfg = self._config
        if cfg.temperature_column not in features.columns:
            raise ValueError(
                f"ScipyParametricModel.predict: temperature_column "
                f"{cfg.temperature_column!r} not present in features frame "
                f"(columns: {list(features.columns)})."
            )

        features_with_fourier = self._append_fourier_columns(features)
        design_matrix, _ = self._build_design_matrix(features_with_fourier)
        y_hat = _parametric_fn(design_matrix, *self._popt)
        return pd.Series(
            y_hat,
            index=features.index,
            name=cfg.target_column,
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model via :func:`~bristol_ml.models.io.save_skops`.

        Stage 12 D10 (Ctrl+G reversal): the project moved off
        ``joblib`` and onto :mod:`skops.io` for security — the serving
        layer is a network-facing deserialiser and ``joblib.load`` on an
        attacker-controlled artefact is RCE.  The fitted state of
        :class:`ScipyParametricModel` is already a small bag of numpy
        arrays plus Python primitives, so the artefact is a dict
        envelope of those primitives — no custom-class registration in
        the project trust-list is required, and skops's restricted
        unpickler accepts the load without further configuration.

        The module-level :func:`_parametric_fn` reference is *not*
        stored in the artefact (skops would refuse to dump it); on load
        the function is re-bound from the module namespace at
        :meth:`predict` time.  This eliminates the codebase surprise S2
        entirely — the round-trip no longer depends on
        :func:`_parametric_fn` being pickleable, only on it being
        importable.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._popt is None:
            raise RuntimeError("Cannot save unfitted ScipyParametricModel")
        if self._pcov is None:
            raise RuntimeError("Cannot save unfitted ScipyParametricModel")
        envelope: dict[str, object] = {
            "format": _SCIPY_PARAMETRIC_ENVELOPE_FORMAT,
            "config_dump": self._config.model_dump(),
            "popt": np.asarray(self._popt, dtype=np.float64),
            "pcov": np.asarray(self._pcov, dtype=np.float64),
            "feature_columns": list(self._feature_columns),
            "param_names": list(self._param_names),
            "fit_utc_isoformat": (self._fit_utc.isoformat() if self._fit_utc is not None else None),
        }
        save_skops(envelope, path)

    @classmethod
    def load(cls, path: Path) -> ScipyParametricModel:
        """Load a previously-saved :class:`ScipyParametricModel` from ``path``.

        Reads the dict-envelope-of-primitives written by :meth:`save`
        through :func:`bristol_ml.models.io.load_skops` (skops's
        restricted unpickler enforces the project trust-list).  The
        envelope is then unpacked back into a
        :class:`ScipyParametricModel` instance — the trained parameter
        vector and covariance matrix are reconstructed bit-exact (numpy
        arrays round-trip through skops without lossy conversion).

        Raises
        ------
        TypeError
            If the artefact at ``path`` is not a recognised
            :class:`ScipyParametricModel` envelope (e.g. a
            :class:`~bristol_ml.models.linear.LinearModel` artefact, or
            a wrong ``format`` tag).  Mirrors the Stage 4 ``LinearModel.load``
            guard with the new envelope-tag discriminator.
        """
        envelope = load_skops(path)
        if (
            not isinstance(envelope, dict)
            or envelope.get("format") != _SCIPY_PARAMETRIC_ENVELOPE_FORMAT
        ):
            raise TypeError(
                f"Expected a ScipyParametricModel skops envelope at {path} (format="
                f"{_SCIPY_PARAMETRIC_ENVELOPE_FORMAT!r}); got "
                f"{type(envelope).__name__} with format="
                f"{envelope.get('format') if isinstance(envelope, dict) else None!r}."
            )
        cfg = ScipyParametricConfig(**envelope["config_dump"])
        model = cls(cfg)
        model._popt = np.asarray(envelope["popt"], dtype=np.float64)
        model._pcov = np.asarray(envelope["pcov"], dtype=np.float64)
        model._feature_columns = tuple(envelope["feature_columns"])
        model._param_names = tuple(envelope["param_names"])
        fit_utc_iso = envelope["fit_utc_isoformat"]
        model._fit_utc = datetime.fromisoformat(fit_utc_iso) if fit_utc_iso is not None else None
        return model

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit.

        Before :meth:`fit` has been called ``fit_utc`` is ``None`` and
        ``feature_columns`` is empty — matching the Stage 4 protocol
        convention.  Once fitted, ``hyperparameters`` additionally
        carries ``param_names`` / ``param_values`` /
        ``param_std_errors`` / ``covariance_matrix`` (plan D7) so the
        notebook's parameter-table cell can render "value +/- 1.96 * std"
        directly from ``metadata.hyperparameters`` without reaching into
        private attributes.
        """
        cfg = self._config
        hyperparameters: dict[str, object] = {
            "target_column": cfg.target_column,
            "temperature_column": cfg.temperature_column,
            "diurnal_harmonics": cfg.diurnal_harmonics,
            "weekly_harmonics": cfg.weekly_harmonics,
            "t_heat_celsius": cfg.t_heat_celsius,
            "t_cool_celsius": cfg.t_cool_celsius,
            "loss": cfg.loss,
        }
        if self._popt is not None and self._pcov is not None:
            # ``param_std_errors`` = sqrt(diag(pcov)); non-finite
            # diagonal entries (plan NFR-4) become ``+inf`` here which
            # the ``float("inf")`` round-trip through JSON handles
            # losslessly via :meth:`ndarray.tolist`.
            diag = np.diag(self._pcov)
            std_err = np.where(
                np.isfinite(diag) & (diag >= 0.0),
                np.sqrt(np.where(diag >= 0.0, diag, 0.0)),
                float("inf"),
            )
            hyperparameters["param_names"] = list(self._param_names)
            hyperparameters["param_values"] = self._popt.tolist()
            hyperparameters["param_std_errors"] = std_err.tolist()
            hyperparameters["covariance_matrix"] = self._pcov.tolist()
        return ModelMetadata(
            name=_build_metadata_name(
                diurnal=cfg.diurnal_harmonics,
                weekly=cfg.weekly_harmonics,
            ),
            feature_columns=self._feature_columns,
            fit_utc=self._fit_utc,
            hyperparameters=hyperparameters,
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _require_utc_datetimeindex(features: pd.DataFrame, *, method: str) -> None:
        """Guard ``features.index`` as a UTC-tz ``DatetimeIndex`` (plan D8)."""
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError(
                f"ScipyParametricModel.{method} requires a DatetimeIndex on features; "
                f"got {type(features.index).__name__}."
            )
        if features.index.tz is None:
            raise ValueError(
                f"ScipyParametricModel.{method} requires a tz-aware DatetimeIndex on features "
                "(the Stage 3 assembler contract guarantees UTC). Got tz-naive."
            )
        if str(features.index.tz) != "UTC":
            raise ValueError(
                f"ScipyParametricModel.{method} requires a UTC-tz DatetimeIndex on features "
                "(the Stage 3 assembler contract); got "
                f"tz={features.index.tz!r}. "
                "Convert upstream via df.index = df.index.tz_convert('UTC')."
            )

    def _append_fourier_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        """Append diurnal (24 h) + weekly (168 h) Fourier pairs (plan D2)."""
        cfg = self._config
        out = features
        if cfg.diurnal_harmonics > 0:
            out = append_weekly_fourier(
                out,
                period_hours=24,
                harmonics=cfg.diurnal_harmonics,
                column_prefix="diurnal",
            )
        if cfg.weekly_harmonics > 0:
            out = append_weekly_fourier(
                out,
                period_hours=168,
                harmonics=cfg.weekly_harmonics,
                column_prefix="weekly",
            )
        if cfg.diurnal_harmonics == 0 and cfg.weekly_harmonics == 0:
            # No Fourier append = caller expects an unchanged copy.
            out = out.copy()
        return out

    def _build_design_matrix(
        self, features_with_fourier: pd.DataFrame
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Assemble the ``(n_features, n_obs)`` design matrix for ``curve_fit``.

        Row layout returned:

        - row 0: HDD, i.e. ``max(0, T_heat - T)``.
        - row 1: CDD, i.e. ``max(0, T - T_cool)``.
        - rows 2..: diurnal sin/cos pairs then weekly sin/cos pairs, in
          the order emitted by
          :func:`~bristol_ml.features.fourier.append_weekly_fourier`.

        Returns the matrix plus the tuple of Fourier column names (in
        design-matrix row order) so :meth:`fit` can record them under
        ``metadata.feature_columns``.

        D2 clarification: any column on ``features_with_fourier`` that
        is *not* the temperature column or one of the Fourier columns is
        deliberately dropped here, even if the user left
        ``config.feature_columns`` at the permissive ``None`` default.
        This prevents Stage 5 calendar one-hots from sneaking into the
        parametric model's feature set and colliding with the Fourier
        weekly terms.
        """
        cfg = self._config
        temp = np.asarray(features_with_fourier[cfg.temperature_column], dtype=np.float64)
        hdd = np.maximum(0.0, cfg.t_heat_celsius - temp)
        cdd = np.maximum(0.0, temp - cfg.t_cool_celsius)

        fourier_col_order: list[str] = []
        for k in range(1, cfg.diurnal_harmonics + 1):
            fourier_col_order.append(f"diurnal_sin_k{k}")
            fourier_col_order.append(f"diurnal_cos_k{k}")
        for j in range(1, cfg.weekly_harmonics + 1):
            fourier_col_order.append(f"weekly_sin_k{j}")
            fourier_col_order.append(f"weekly_cos_k{j}")

        # D2 clarification: if ``config.feature_columns`` is explicitly
        # set, it must be a *subset* of the Fourier column names above
        # (no calendar or weather columns).  An empty / None config
        # falls through to the full Fourier tuple.
        if cfg.feature_columns is not None:
            configured = tuple(cfg.feature_columns)
            unknown = [c for c in configured if c not in fourier_col_order]
            if unknown:
                raise ValueError(
                    "ScipyParametricModel.fit: config.feature_columns may "
                    "only name Fourier columns generated by this model "
                    f"(diurnal_*/weekly_*); unknown names: {unknown}."
                )
            fourier_col_order = list(configured)

        rows: list[np.ndarray] = [hdd, cdd]
        for name in fourier_col_order:
            rows.append(np.asarray(features_with_fourier[name], dtype=np.float64))
        return np.vstack(rows), tuple(fourier_col_order)


# ---------------------------------------------------------------------------
# Private helpers (module-level so they stay easy to unit-test)
# ---------------------------------------------------------------------------


def _build_metadata_name(*, diurnal: int, weekly: int) -> str:
    """Build a metadata ``name`` that matches ``ModelMetadata.name``'s regex.

    :class:`~conf._schemas.ModelMetadata` constrains ``name`` to
    ``^[a-z][a-z0-9_.-]*$``.  The format is
    ``scipy-parametric-d{K_d}-w{K_w}``.
    """
    return f"scipy-parametric-d{diurnal}-w{weekly}"


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.scipy_parametric``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.scipy_parametric",
        description=(
            "Print the ScipyParametricConfig JSON schema and a one-line "
            "help banner. Training and prediction are exercised via the "
            "evaluation harness (see `python -m bristol_ml.train`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model.diurnal_harmonics=2",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Prints the :class:`~conf._schemas.ScipyParametricConfig` JSON schema
    to stdout and a one-line help banner pointing at the training CLI.
    Returns ``0`` on success, ``2`` if a non-parametric model resolves
    from config (e.g. if the caller forgot ``model=scipy_parametric``).
    """
    import json

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    print("ScipyParametricConfig JSON schema:")
    print(json.dumps(ScipyParametricConfig.model_json_schema(), indent=2))
    print()
    print(
        "To fit: python -m bristol_ml.train model=scipy_parametric "
        "evaluation.rolling_origin.fixed_window=true "
        "evaluation.rolling_origin.step=168"
    )

    cfg = load_config(overrides=["model=scipy_parametric", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, ScipyParametricConfig):
        print(
            "No ScipyParametricConfig resolved. Ensure "
            "`model=scipy_parametric` or a matching override is present; "
            f"got {type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    scipy_cfg = cfg.model
    logger.info(
        "ScipyParametricModel config: diurnal_harmonics={} weekly_harmonics={} "
        "t_heat_celsius={} t_cool_celsius={} loss={} target_column={}",
        scipy_cfg.diurnal_harmonics,
        scipy_cfg.weekly_harmonics,
        scipy_cfg.t_heat_celsius,
        scipy_cfg.t_cool_celsius,
        scipy_cfg.loss,
        scipy_cfg.target_column,
    )
    print()
    print(f"diurnal_harmonics={scipy_cfg.diurnal_harmonics}")
    print(f"weekly_harmonics={scipy_cfg.weekly_harmonics}")
    print(f"t_heat_celsius={scipy_cfg.t_heat_celsius}")
    print(f"t_cool_celsius={scipy_cfg.t_cool_celsius}")
    print(f"loss={scipy_cfg.loss}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
