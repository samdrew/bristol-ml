"""Spec-derived tests for ``bristol_ml.models.scipy_parametric`` — Tasks T2 and T3.

Every test is derived from:

- ``docs/plans/active/08-scipy-parametric.md`` §Task T2 (lines 269-274): module-level
  helper tests.
- ``docs/plans/active/08-scipy-parametric.md`` §Task T3 (lines 284-289): scaffold,
  metadata, and CLI tests.
- Acceptance criteria AC-8 (``_require_utc_datetimeindex``) and AC-10 (CLI entrypoint),
  referenced in the plan §4.

No production code is modified here.  If a test below fails the failure indicates
a deviation from the spec — do not weaken the test; surface the failure to the
implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- ``ScipyParametricConfig()`` default construction throughout (no extra kwargs needed
  unless the test is specifically exercising non-default parameters).
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bristol_ml.models.scipy_parametric import (
    ScipyParametricModel,
    _build_param_names,
    _derive_p0,
    _parametric_fn,
)
from conf._schemas import ScipyParametricConfig

# ===========================================================================
# Task T2 — Module-level pure helpers
# (plan §Task T2, lines 269-274)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. test_parametric_fn_reproduces_known_sinusoid
# ---------------------------------------------------------------------------


def test_parametric_fn_reproduces_known_sinusoid() -> None:
    """``_parametric_fn`` returns the hand-calculated value within ``atol=1e-9``.

    Ground truth: ``alpha=10000, beta_heat=100, beta_cool=50`` with three
    diurnal pairs (6 Fourier coefficients all set to zero for this test) and
    a two-observation design matrix.

    The plan (T2 note) specifies that the **first two rows** of ``X`` are
    **HDD** and **CDD** (pre-transformed), not raw temperature.  Rows 2+
    are diurnal sin/cos pairs.

    For observation 0:
      hdd = 5.0, cdd = 0.0 → temp contribution = 10000 + 100*5 + 50*0 = 10500.
      Fourier rows all zero → y[0] = 10500.

    For observation 1:
      hdd = 0.0, cdd = 3.0 → temp contribution = 10000 + 100*0 + 50*3 = 10150.
      Fourier rows all zero → y[1] = 10150.

    Plan clause: T2 plan §Task T2 named test ``test_parametric_fn_reproduces_known_sinusoid``.
    """
    # Design matrix shape: (n_features, n_obs) = (2 + 6, 2).
    # Rows: [hdd, cdd, diurnal_sin_k1, diurnal_cos_k1, diurnal_sin_k2,
    #        diurnal_cos_k2, diurnal_sin_k3, diurnal_cos_k3]
    n_obs = 2
    hdd = np.array([5.0, 0.0])
    cdd = np.array([0.0, 3.0])
    # Three diurnal pairs — all zeroed to make the hand calc clean.
    fourier_zero = np.zeros((6, n_obs))
    X = np.vstack([hdd, cdd, fourier_zero])  # shape (8, 2)

    alpha = 10000.0
    beta_heat = 100.0
    beta_cool = 50.0
    # Six Fourier coefficients, all zero.
    params = (alpha, beta_heat, beta_cool) + (0.0,) * 6

    result = _parametric_fn(X, *params)

    expected = np.array([10500.0, 10150.0])
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-9,
        err_msg=(
            "_parametric_fn did not reproduce the hand-calculated values. "
            "Expected y[0]=10500 (hdd=5, cdd=0) and y[1]=10150 (hdd=0, cdd=3). "
            "Plan T2 / ``test_parametric_fn_reproduces_known_sinusoid``."
        ),
    )


# ---------------------------------------------------------------------------
# 2. test_parametric_fn_is_pickleable
# ---------------------------------------------------------------------------


def test_parametric_fn_is_pickleable() -> None:
    """``_parametric_fn`` survives a pickle round-trip and returns identical values.

    Guards codebase surprise S2 (plan §5 risk R5): ``curve_fit`` holds a
    reference to the target function; joblib/pickle must be able to
    serialise that reference.  The function must be module-level (not a
    closure, not a lambda, not a bound method) for this to work.

    Plan clause: T2 plan §Task T2 named test ``test_parametric_fn_is_pickleable``
    / codebase research §6 + S2.
    """
    # Build a minimal design matrix (2 rows: hdd, cdd; no Fourier).
    hdd = np.array([2.0, 0.0, 5.0])
    cdd = np.array([0.0, 1.0, 0.0])
    X = np.vstack([hdd, cdd])  # shape (2, 3)
    params = (9000.0, 80.0, 30.0)

    original_result = _parametric_fn(X, *params)

    # Pickle round-trip.
    blob = pickle.dumps(_parametric_fn)
    recovered_fn = pickle.loads(blob)

    assert callable(recovered_fn), (
        "pickle.loads of _parametric_fn must return a callable; "
        "got a non-callable (T2 / S2 pickleability guard)."
    )
    recovered_result = recovered_fn(X, *params)

    np.testing.assert_array_equal(
        original_result,
        recovered_result,
        err_msg=(
            "Recovered function must return identical values to the original. "
            "Plan T2 / ``test_parametric_fn_is_pickleable`` / S2."
        ),
    )


# ---------------------------------------------------------------------------
# 3. test_derive_p0_returns_finite_values_on_empty_cooling_segment
# ---------------------------------------------------------------------------


def test_derive_p0_returns_finite_values_on_empty_cooling_segment() -> None:
    """``p0[2]`` (beta_cool) is a finite number when no observations exceed t_cool.

    Training data with only sub-15 °C temperatures means the cooling segment
    (above ``t_cool = 22 °C``) is empty.  ``_derive_p0`` must return a finite
    ``0.0`` for ``p0[2]`` rather than ``NaN`` or ``inf``.

    Plan clause: T2 plan §Task T2 named test
    ``test_derive_p0_returns_finite_values_on_empty_cooling_segment``
    / domain §R5 (Seber-Wild recommendation — avoid inf in starting point).
    """
    rng = np.random.default_rng(42)
    n = 200
    # All temperatures well below t_heat=15.5 → no cooling observations.
    temperature = pd.Series(rng.uniform(-5.0, 10.0, n))
    target = pd.Series(10_000.0 + rng.normal(0, 200, n))

    p0 = _derive_p0(
        target=target,
        temperature=temperature,
        t_heat=15.5,
        t_cool=22.0,
        diurnal_harmonics=3,
        weekly_harmonics=2,
    )

    assert np.isfinite(p0[2]), (
        f"p0[2] (beta_cool) must be a finite number when the cooling segment is "
        f"empty; got {p0[2]!r}. Plan T2 / "
        "``test_derive_p0_returns_finite_values_on_empty_cooling_segment``."
    )
    assert p0[2] == 0.0, (
        f"p0[2] (beta_cool) must be 0.0 when no above-t_cool observations exist; "
        f"got {p0[2]!r}. Plan T2 / domain §R5."
    )


# ---------------------------------------------------------------------------
# 4. test_build_param_names_count_matches_fn_arity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "diurnal, weekly",
    [
        (0, 0),
        (3, 2),
        (4, 4),
    ],
)
def test_build_param_names_count_matches_fn_arity(diurnal: int, weekly: int) -> None:
    """``len(names) == 3 + 2*diurnal + 2*weekly`` for all parameterised combos.

    Verifies that ``_build_param_names`` returns a tuple whose length equals
    the expected parameter arity of ``_parametric_fn`` for the same harmonic
    counts.

    Plan clause: T2 plan §Task T2 named test
    ``test_build_param_names_count_matches_fn_arity``.
    """
    names = _build_param_names(diurnal_harmonics=diurnal, weekly_harmonics=weekly)
    expected_len = 3 + 2 * diurnal + 2 * weekly

    assert len(names) == expected_len, (
        f"For (diurnal={diurnal}, weekly={weekly}), len(names) must equal "
        f"{expected_len}; got {len(names)} with names={names!r}. "
        "Plan T2 / ``test_build_param_names_count_matches_fn_arity``."
    )


# ===========================================================================
# Task T3 — ScipyParametricModel scaffold + metadata + CLI
# (plan §Task T3, lines 284-289)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. test_scipy_parametric_unfitted_metadata_name_regex
# ---------------------------------------------------------------------------


def test_scipy_parametric_unfitted_metadata_name_regex() -> None:
    """Pre-fit ``metadata.name`` matches regex ``^[a-z][a-z0-9_.-]*$``.

    The ``ModelMetadata.name`` field constraint requires lower-case
    alphanumeric names with optional ``_``, ``.``, ``-`` separators.
    This test verifies the name produced by the default config before
    any call to ``fit()``.

    Plan clause: T3 plan §Task T3 named test
    ``test_scipy_parametric_unfitted_metadata_name_regex``.
    """
    config = ScipyParametricConfig()
    model = ScipyParametricModel(config)
    name = model.metadata.name

    assert re.match(r"^[a-z][a-z0-9_.-]*$", name), (
        f"Pre-fit metadata.name must match ^[a-z][a-z0-9_.-]*$; "
        f"got {name!r}. Plan T3 / ``test_scipy_parametric_unfitted_metadata_name_regex``."
    )


# ---------------------------------------------------------------------------
# 2. test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit
# ---------------------------------------------------------------------------


def test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit() -> None:
    """Pre-fit ``metadata.hyperparameters`` has config keys but not fitted-state keys.

    Before ``fit()`` is called the ``hyperparameters`` dict must contain all of:
    ``{target_column, temperature_column, diurnal_harmonics, weekly_harmonics,
    t_heat_celsius, t_cool_celsius, loss}``

    And must NOT yet contain any of the fit-time provenance keys:
    ``{param_values, covariance_matrix, param_names, param_std_errors}``.

    Plan clause: T3 plan §Task T3 named test
    ``test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit``
    / plan §5 architecture summary (``metadata`` property pre-/post-fit distinction).
    """
    config = ScipyParametricConfig()
    model = ScipyParametricModel(config)
    hp = model.metadata.hyperparameters

    required_before_fit = {
        "target_column",
        "temperature_column",
        "diurnal_harmonics",
        "weekly_harmonics",
        "t_heat_celsius",
        "t_cool_celsius",
        "loss",
    }
    for key in required_before_fit:
        assert key in hp, (
            f"metadata.hyperparameters must contain {key!r} before fit(); "
            f"got keys {set(hp.keys())!r}. Plan T3 / "
            "``test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit``."
        )

    forbidden_before_fit = {
        "param_values",
        "covariance_matrix",
        "param_names",
        "param_std_errors",
    }
    for key in forbidden_before_fit:
        assert key not in hp, (
            f"metadata.hyperparameters must NOT contain {key!r} before fit(); "
            f"it should only appear after a successful fit(). "
            f"Got keys {set(hp.keys())!r}. Plan T3 / "
            "``test_scipy_parametric_metadata_hyperparameters_contains_expected_keys_before_fit``."
        )


# ---------------------------------------------------------------------------
# 3. test_scipy_parametric_module_has_cli_main  (AC-10)
# ---------------------------------------------------------------------------


def test_scipy_parametric_module_has_cli_main() -> None:
    """``python -m bristol_ml.models.scipy_parametric --help`` exits 0 (AC-10).

    Verifies DESIGN §2.1.1: every module runs standalone via
    ``python -m bristol_ml.<module>``.  The ``--help`` flag must exit 0
    so the entrypoint is reachable from the CLI.

    Plan clause: T3 plan §Task T3 named test ``test_scipy_parametric_module_has_cli_main``
    / AC-10 / DESIGN §2.1.1.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.scipy_parametric", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"``python -m bristol_ml.models.scipy_parametric --help`` must exit 0; "
        f"got returncode={result.returncode}.\n"
        f"stdout: {result.stdout[:500]!r}\n"
        f"stderr: {result.stderr[:500]!r}\n"
        "Plan T3 / AC-10 / DESIGN §2.1.1."
    )


# ---------------------------------------------------------------------------
# 4. test_scipy_parametric_require_utc_raises_on_tz_naive_index  (AC-8)
# ---------------------------------------------------------------------------


def test_scipy_parametric_require_utc_raises_on_tz_naive_index() -> None:
    """``_require_utc_datetimeindex`` raises ``ValueError`` with "UTC" for tz-naive input.

    Constructs a model with the default config and calls
    ``_require_utc_datetimeindex`` directly with a tz-naive DataFrame.
    The error message must contain "UTC" so the user understands the contract.

    Plan clause: T3 plan §Task T3 named test
    ``test_scipy_parametric_require_utc_raises_on_tz_naive_index``
    / AC-8 / plan D8.
    """
    config = ScipyParametricConfig()
    model = ScipyParametricModel(config)

    # Build a tz-naive DataFrame (no timezone on the DatetimeIndex).
    naive_index = pd.date_range("2024-01-01", periods=48, freq="h")  # no tz
    naive_df = pd.DataFrame(
        {"temperature_2m": np.linspace(5.0, 15.0, 48)},
        index=naive_index,
    )
    assert naive_df.index.tz is None, "Precondition: index must be tz-naive."

    with pytest.raises(ValueError) as exc_info:
        model._require_utc_datetimeindex(naive_df, method="fit")

    assert "UTC" in str(exc_info.value), (
        f"ValueError message must contain 'UTC'; got {str(exc_info.value)!r}. "
        "Plan T3 / AC-8 / plan D8."
    )
