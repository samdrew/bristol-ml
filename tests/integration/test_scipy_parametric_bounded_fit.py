"""Integration regression test for the bounded-TRF parametric fit.

Stage 8 follow-up plan ``08a`` (bounded parametric fit) AC-3 and AC-4:

- AC-3: on the project-default 1-year+ training window the fit must
  not regress against the LM predecessor.  The pre-change LM produced
  a cross-fold mean MAE around 5 000 MW on the warm
  ``weather_calendar.parquet`` cache; the new TRF fit must stay within
  the same magnitude.
- AC-4: on the catastrophic 30-day sliding window (the
  rank-deficient seasonal-mono fold mix that exhibited the original
  bug at MAE ~167 600 MW) the cross-fold mean MAE must drop below
  6 000 MW — the empirical "no fold diverges" signal.

The test SKIPs when the warm feature-table cache is not present,
mirroring the cassette-skip pattern used by the LLM and embeddings
notebooks.  CI runners that lack the cache do not exercise this test.

Wall-time budget: ~5 minutes for AC-3 (365 folds), ~1 second for AC-4
(52 folds); marked ``slow`` so the default ``uv run pytest`` skips it
unless the runner opts in via ``-m slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bristol_ml.evaluation.harness import evaluate
from bristol_ml.evaluation.metrics import mae, rmse
from bristol_ml.models.scipy_parametric import ScipyParametricModel
from conf._schemas import ScipyParametricConfig, SplitterConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_CALENDAR_CACHE = REPO_ROOT / "data" / "features" / "weather_calendar.parquet"


def _cache_warm() -> bool:
    return WEATHER_CALENDAR_CACHE.exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not _cache_warm(),
    reason=(
        "weather_calendar.parquet not warm; AC-3/AC-4 regression tests need "
        "the Stage 5 feature-table cache.  Populate via `uv run python -m "
        "bristol_ml.features.assembler features=weather_calendar`."
    ),
)
def test_scipy_parametric_ac3_one_year_window_no_regression() -> None:
    """AC-3: cross-fold mean MAE on the 1-year default window stays under 7 000.

    Pre-change LM baseline (recorded 2026-05-04 against the warm
    ``weather_calendar.parquet`` cache and the project-default
    ``ScipyParametricConfig`` + ``SplitterConfig(min_train_periods=8760,
    test_len=168, step=168)``): mean MAE ~4 900, median ~4 800, max
    ~9 300.  This test guards against a TRF regression that would push
    any of these substantially higher.

    Threshold (mean < 7 000) chosen as the recorded baseline (~4 900)
    plus a 40 % margin — generous enough to absorb scipy minor-version
    drift without false positives, tight enough that a real regression
    (e.g. a fold diverging to 100 000+) immediately fails the test.

    Plan clause: 08a AC-3 / T4 /
    ``test_scipy_parametric_ac3_one_year_window_no_regression``.
    """
    df = pd.read_parquet(WEATHER_CALENDAR_CACHE).set_index("timestamp_utc")
    cfg = ScipyParametricConfig()
    splitter = SplitterConfig(min_train_periods=8760, test_len=168, step=168, fixed_window=False)

    results = evaluate(
        ScipyParametricModel(cfg),
        df,
        splitter_cfg=splitter,
        metrics=[mae, rmse],
        target_column=cfg.target_column,
    )

    mean_mae = float(results["mae"].mean())
    median_mae = float(results["mae"].median())
    max_mae = float(results["mae"].max())

    assert mean_mae < 7_000.0, (
        f"AC-3: cross-fold mean MAE {mean_mae:.1f} MW exceeds the 7 000 "
        f"threshold (median {median_mae:.1f}, max {max_mae:.1f}); "
        f"a TRF regression has pushed the healthy-path fit off baseline. "
        "Plan 08a AC-3."
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _cache_warm(),
    reason=(
        "weather_calendar.parquet not warm; AC-3/AC-4 regression tests need "
        "the Stage 5 feature-table cache.  Populate via `uv run python -m "
        "bristol_ml.features.assembler features=weather_calendar`."
    ),
)
def test_scipy_parametric_ac4_thirty_day_window_no_divergence() -> None:
    """AC-4: cross-fold mean MAE on the catastrophic 30-day window stays under 6 000.

    The 30-day sliding window (``min_train_periods=720, fixed_window=
    True, step=1344``) was the splitter that exhibited the original
    bug — under unbounded LM, ten-of-fifty seasonal-mono folds produced
    per-fold MAE in the hundreds of thousands to millions of MW,
    dragging the cross-fold mean to ~167 600.  Under bounded TRF the
    rank-deficient parameter saturates at its bound rather than
    diverging; this test pins that empirical fact.

    Recorded post-fix baseline (2026-05-04): mean MAE 4 563, median
    4 447, max 7 187.  Threshold (mean < 6 000) per the plan AC-4
    spec.

    Plan clause: 08a AC-4 / T4 /
    ``test_scipy_parametric_ac4_thirty_day_window_no_divergence``.
    """
    df = pd.read_parquet(WEATHER_CALENDAR_CACHE).set_index("timestamp_utc")
    cfg = ScipyParametricConfig()
    splitter = SplitterConfig(min_train_periods=720, test_len=168, step=1344, fixed_window=True)

    results = evaluate(
        ScipyParametricModel(cfg),
        df,
        splitter_cfg=splitter,
        metrics=[mae, rmse],
        target_column=cfg.target_column,
    )

    mean_mae = float(results["mae"].mean())
    max_mae = float(results["mae"].max())

    assert mean_mae < 6_000.0, (
        f"AC-4: cross-fold mean MAE {mean_mae:.1f} MW on the 30-day "
        f"sliding window exceeds 6 000 (max fold MAE {max_mae:.1f}); "
        "a fold has diverged, which means the bounded-TRF + "
        "zero-information-column override is no longer catching the "
        "rank-deficient seasonal-mono case.  Plan 08a AC-4."
    )
    # Belt-and-braces: no individual fold should diverge to the
    # catastrophic pre-fix scale (the unbounded LM predecessor produced
    # per-fold MAE in the millions on the divergent folds).
    assert max_mae < 50_000.0, (
        f"AC-4: a single fold's MAE reached {max_mae:.1f} MW, two orders "
        "of magnitude above the healthy fit — a numerical divergence "
        "the bounded fit is supposed to prevent.  Plan 08a AC-4."
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _cache_warm(),
    reason=(
        "weather_calendar.parquet not warm; AC-3 golden-popt regression "
        "test needs the Stage 5 feature-table cache.  Populate via "
        "`uv run python -m bristol_ml.features.assembler "
        "features=weather_calendar`."
    ),
)
def test_scipy_parametric_ac3_no_fourier_golden_popt() -> None:
    """AC-3 (popt-conformance flavour): no-Fourier fit pins a stable golden vector.

    The plan AC-3 originally asked for ``rtol < 1e-4 vs the LM
    predecessor`` on the project default config.  Empirical
    verification (recorded 2026-05-04) showed that on the project
    default — 13-parameter form with ``diurnal_harmonics=3``,
    ``weekly_harmonics=2`` — even the *unbounded* LM predecessor
    produced popt with ``alpha ~ 4e8 MW`` (predictions correct via
    numerical cancellation; popt physically nonsense).  The bounded
    TRF predecessor settles many Fourier coefficients at the bounds
    (±50 000 MW) for the same reason — the temperature regressor is
    correlated with the weekly-Fourier annual harmonic, so the
    optimum is multi-modal and any solver finds *some* corner.  An
    rtol-vs-LM popt pin is therefore unimplementable on the default
    config; the cross-fold-mean-MAE test above is the operational
    AC-3 substance.

    For *popt* conformance we fit a deliberately well-conditioned
    config — Fourier disabled — so ``alpha``, ``beta_heat``,
    ``beta_cool`` are uniquely determined by the data.  This pins
    a stable golden vector and catches any future implementation
    drift that would perturb the fitted slopes.

    Recorded baseline (2026-05-04, ``slice = df.iloc[:8760]``,
    bounded TRF) — well-determined HDD/CDD model has a unique
    optimum that agrees with unbounded LM to 4 decimal places:
        alpha = 27072.446
        beta_heat = 548.300
        beta_cool = 470.578

    Plan clause: 08a AC-3 / T4 /
    ``test_scipy_parametric_ac3_no_fourier_golden_popt``.
    """
    df = pd.read_parquet(WEATHER_CALENDAR_CACHE).set_index("timestamp_utc")
    slice_df = df.iloc[:8760]

    cfg = ScipyParametricConfig(diurnal_harmonics=0, weekly_harmonics=0)
    model = ScipyParametricModel(cfg)
    model.fit(slice_df, slice_df["nd_mw"])

    expected = np.array([27072.446, 548.300, 470.578], dtype=np.float64)
    assert model._popt is not None
    np.testing.assert_allclose(
        model._popt,
        expected,
        rtol=1e-4,
        err_msg=(
            "AC-3 popt-conformance: well-determined no-Fourier fit on "
            "the first-year slice of weather_calendar.parquet drifted "
            "from the recorded 2026-05-04 baseline by more than rtol "
            "1e-4.  This is the regression-detection signal — investigate "
            "scipy version drift or implementation changes that would "
            "perturb the well-conditioned fit.  Plan 08a AC-3."
        ),
    )
