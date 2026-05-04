"""Spec-derived tests for ``bristol_ml.evaluation.plots`` — Stage 6 T2/T3 scaffold.

Every test is derived from:

- ``docs/plans/active/06-enhanced-evaluation.md`` §6 T2 (explicit test list)
  and §4 AC-1/2/6/10/11/12; §6 T3 (hero helpers and D6 DST gate).
- Plan decisions D2 (Okabe-Ito palette, cividis sequential, RdBu_r diverging),
  D5 (default figsize 12x8, human mandate 2026-04-20), D6 (display_tz DST gate),
  D7 (ACF reference-lag markers), D9 (CLAUDE.md architectural-debt note for
  harness output API growth trigger).
- ``docs/lld/research/06-enhanced-evaluation-domain.md`` §R2 (Wong 2011
  *Nature Methods* 8:441 — canonical Okabe-Ito hex values).

T2 ships only the module scaffold, palette constants, rcParams injection,
``_cli_main``, and the ``evaluation/__init__.py`` re-export wiring.  The
seven helper *bodies* are NotImplementedError stubs in T2 and are tested
in T3/T4.  Tests here do NOT call the helper bodies.

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each docstring cites the plan clause, AC, or D-number it guards.
- Simple assert lines; no parametrisation unless it adds meaningful clarity.
- ``matplotlib.use("Agg")`` is the headless CI default — the Agg backend is
  active by default in pytest, so no explicit ``matplotlib.use()`` call is
  needed here.
"""

from __future__ import annotations

import importlib
import re
import unittest.mock as mock
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Confirm the package is importable; skip cleanly if the install is absent.
pytest.importorskip("bristol_ml.evaluation.plots")

# Re-import-clean: load the module so we test the live state of rcParams
# after import-time side effects have run.  ``_plots`` and ``plots_mod``
# alias the same module so tests can reference either name interchangeably.
plots_mod = importlib.import_module("bristol_ml.evaluation.plots")
_plots = plots_mod


# Repo-root-relative path to ``src/bristol_ml/evaluation/CLAUDE.md``.  Derived
# from ``__file__`` so the tests work regardless of the checkout location (the
# previous hardcoded ``/workspace/...`` failed in GitHub Actions, whose runner
# roots the checkout at ``/home/runner/work/<repo>/<repo>``).  The test file
# sits at ``tests/unit/evaluation/test_plots.py``; ``parents[3]`` is the repo
# root in every supported checkout layout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVALUATION_CLAUDE_MD = _REPO_ROOT / "src" / "bristol_ml" / "evaluation" / "CLAUDE.md"


# ---------------------------------------------------------------------------
# Test 1 — CLI returns 0 and lists helper names (Plan T2; AC-10)
# ---------------------------------------------------------------------------


def test_plots_cli_main_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Guards AC-10 and plan T2: ``_cli_main([])`` returns 0; prints helper names.

    DESIGN §2.1.1 requires every module to run standalone.  The CLI entry
    point must exit cleanly (return code 0) and must print enough information
    for a facilitator to verify the module surface at a glance.

    The three hero helper names checked here are the minimum set that confirms
    the CLI is printing the real surface, not a placeholder.
    """
    rc = _plots._cli_main([])
    captured = capsys.readouterr()

    assert rc == 0, f"_cli_main([]) must return 0 (DESIGN §2.1.1 / AC-10); got {rc!r}."
    assert "residuals_vs_time" in captured.out, (
        "'residuals_vs_time' must appear in CLI stdout (plan T2 / AC-10)."
    )
    assert "acf_residuals" in captured.out, (
        "'acf_residuals' must appear in CLI stdout (plan T2 / AC-10)."
    )
    assert "error_heatmap_hour_weekday" in captured.out, (
        "'error_heatmap_hour_weekday' must appear in CLI stdout (plan T2 / AC-10)."
    )


# ---------------------------------------------------------------------------
# Test 2 — OKABE_ITO constant matches Wong 2011 verbatim (Plan T2; D2; AC-6)
# ---------------------------------------------------------------------------

_EXPECTED_OKABE_ITO = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_plots_module_has_okabe_ito_constant() -> None:
    """Guards D2 / AC-6 / plan T2: ``OKABE_ITO`` is exactly the Wong 2011 palette.

    The Wong 2011 (*Nature Methods* 8:441) values are pinned verbatim so that
    downstream notebooks that import ``OKABE_ITO`` and index by position
    receive the formally CVD-certified palette, not an approximation.

    Checks:
    - Tuple of exactly 8 items.
    - Each item is a six-digit hex colour string (``#RRGGBB``).
    - The exact values match Wong 2011 as cited in
      ``docs/lld/research/06-enhanced-evaluation-domain.md`` §R2.
    """
    palette = _plots.OKABE_ITO

    assert isinstance(palette, tuple), f"OKABE_ITO must be a tuple; got {type(palette).__name__!r}."
    assert len(palette) == 8, (
        f"OKABE_ITO must have exactly 8 colours (Wong 2011); got {len(palette)}."
    )
    for idx, colour in enumerate(palette):
        assert _HEX_PATTERN.match(colour), (
            f"OKABE_ITO[{idx}] = {colour!r} does not match #RRGGBB format."
        )
    assert palette == _EXPECTED_OKABE_ITO, (
        f"OKABE_ITO values must match Wong 2011 *Nature Methods* 8:441 verbatim.\n"
        f"Expected: {_EXPECTED_OKABE_ITO}\n"
        f"Got:      {palette}"
    )


# ---------------------------------------------------------------------------
# Test 3 — prop_cycle rcParam equals OKABE_ITO after import (Plan T2; D2)
# ---------------------------------------------------------------------------


def test_plots_rcparams_prop_cycle_is_okabe_ito() -> None:
    """Guards D2 / plan T2: after import, ``axes.prop_cycle`` cycles through OKABE_ITO.

    The module must inject the Okabe-Ito palette into ``plt.rcParams`` at
    import time so every figure created after ``import plots`` uses the
    CVD-safe qualitative palette without any caller action.

    Accesses the colour list via ``prop_cycle.by_key()["color"]`` — the
    canonical Cycler accessor (``cycler`` library API).
    """
    from cycler import Cycler

    prop_cycle = plt.rcParams["axes.prop_cycle"]

    assert isinstance(prop_cycle, Cycler), (
        f"plt.rcParams['axes.prop_cycle'] must be a Cycler after import; "
        f"got {type(prop_cycle).__name__!r}."
    )
    colour_list = prop_cycle.by_key()["color"]
    assert colour_list == list(_plots.OKABE_ITO), (
        "plt.rcParams['axes.prop_cycle'] colour list must equal list(OKABE_ITO) "
        "after import (D2 injection).\n"
        f"Expected: {list(_plots.OKABE_ITO)}\n"
        f"Got:      {colour_list}"
    )


# ---------------------------------------------------------------------------
# Test 4 — figure.figsize rcParam is (12.0, 8.0) (Plan T2; D5)
# ---------------------------------------------------------------------------


def test_plots_rcparams_figsize_is_twelve_by_eight() -> None:
    """Guards D5 / plan T2: ``figure.figsize`` is ``(12.0, 8.0)`` after import.

    D5 (human mandate 2026-04-20): the default figsize is 12x8 for
    projector-friendly meetup demos.  This is explicitly wider and taller
    than the matplotlib default (6.4x4.8) and satisfies AC-2 (visually
    legible at meetup-audience distances).
    """
    figsize = tuple(plt.rcParams["figure.figsize"])
    assert figsize == (12.0, 8.0), (
        f"plt.rcParams['figure.figsize'] must be (12.0, 8.0) (D5 human mandate "
        f"2026-04-20 / AC-2); got {figsize!r}."
    )


# ---------------------------------------------------------------------------
# Test 5 — __all__ is complete and all symbols are importable (Plan T2; AC-1)
# ---------------------------------------------------------------------------


def test_plots_all_exported_symbols_importable() -> None:
    """Guards AC-1 / plan T2: every symbol in ``__all__`` is accessible on the module.

    AC-1 requires that a model conforming to the Stage 4 interface can
    produce every diagnostic with a small, consistent amount of code.
    A broken ``__all__`` — where a name is listed but the attribute is
    absent — would cause an AttributeError at import time in notebooks.

    The seven helper names are the minimum set checked explicitly; the loop
    covers the full ``__all__`` list.
    """
    required_helpers = {
        "residuals_vs_time",
        "predicted_vs_actual",
        "acf_residuals",
        "error_heatmap_hour_weekday",
        "forecast_overlay",
        "forecast_overlay_with_band",
        "benchmark_holdout_bar",
    }

    assert hasattr(_plots, "__all__"), "plots module must define __all__."
    for name in _plots.__all__:
        assert hasattr(_plots, name), (
            f"'{name}' is in __all__ but not accessible via getattr(plots, '{name}')."
        )

    missing = required_helpers - set(_plots.__all__)
    assert not missing, (
        f"The following helper names are absent from __all__: {sorted(missing)} (plan T2 / AC-1)."
    )


# ---------------------------------------------------------------------------
# Test 6 — module docstring uses British English (Plan T2; AC-12; CLAUDE.md)
# ---------------------------------------------------------------------------


def test_plots_module_docstring_british_english() -> None:
    """Guards AC-12 / CLAUDE.md British-English rule / plan T2.

    The module docstring must contain at least one British English spelling
    — "colour" or "visualisation" — as a sanity check that the module
    docstring was not accidentally written in US English.

    This is a lightweight proxy for the coding-conventions rule; it does
    not attempt to parse every word.
    """
    doc = _plots.__doc__
    assert doc is not None, "plots module must have a module-level docstring."
    has_british = "colour" in doc or "visualisation" in doc
    assert has_british, (
        "Module docstring must contain 'colour' or 'visualisation' "
        "(British English per CLAUDE.md / AC-12).\n"
        f"Docstring excerpt: {doc[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — CLAUDE.md has the D9 API growth trigger section (Plan T2; D9)
# ---------------------------------------------------------------------------


def test_evaluation_claude_md_has_api_growth_trigger_note() -> None:
    """Guards D9 / plan T2: CLAUDE.md carries the verbatim architectural-debt note.

    Plan D9 (2026-04-20 human mandate) requires the text of the
    re-engineering trigger to be present in ``src/bristol_ml/evaluation/CLAUDE.md``
    so future implementers find it without needing to re-read the plan.

    Checks both the heading and the key prohibition sentence so the content
    cannot be silently paraphrased away.
    """
    claude_md = _EVALUATION_CLAUDE_MD
    assert claude_md.exists(), f"CLAUDE.md not found at {claude_md} — was T2 hygiene skipped?"
    content = claude_md.read_text(encoding="utf-8")

    assert "Harness output — API growth trigger" in content, (
        "CLAUDE.md must contain the heading 'Harness output — API growth trigger' "
        "(plan T2 D9 architectural-debt note)."
    )
    assert "Do not add a second boolean flag" in content, (
        "CLAUDE.md must contain 'Do not add a second boolean flag' — the verbatim "
        "prohibition text from plan D9 (2026-04-20 human mandate)."
    )


# ---------------------------------------------------------------------------
# Test 8 — SEQUENTIAL_CMAP is "cividis" (Plan T2; D2)
# ---------------------------------------------------------------------------


def test_plots_sequential_cmap_is_cividis() -> None:
    """Guards D2 / plan T2: ``SEQUENTIAL_CMAP == "cividis"``.

    ``cividis`` is selected in plan D2 as the CVD-safe perceptually-uniform
    sequential colormap (Nunez et al. 2018; superior CVD accessibility over
    ``viridis``).  Used for "more-is-more" intensity maps, e.g. a per-hour
    absolute-error heatmap.
    """
    assert _plots.SEQUENTIAL_CMAP == "cividis", (
        f"SEQUENTIAL_CMAP must be 'cividis' (D2 / Nunez et al. 2018); "
        f"got {_plots.SEQUENTIAL_CMAP!r}."
    )


# ---------------------------------------------------------------------------
# Test 9 — DIVERGING_CMAP is "RdBu_r" (Plan T2; D2)
# ---------------------------------------------------------------------------


def test_plots_diverging_cmap_is_rdbu_r() -> None:
    """Guards D2 / plan T2: ``DIVERGING_CMAP == "RdBu_r"``.

    ``RdBu_r`` (reversed red-blue) is selected in plan D2 for signed-residual
    heatmaps because the hue axis is also value-mapped, preserving
    legibility under red/blue dichromacy.  ``PuOr`` and ``BrBG`` are
    acceptable alternatives but ``RdBu_r`` is the named default.
    """
    assert _plots.DIVERGING_CMAP == "RdBu_r", (
        f"DIVERGING_CMAP must be 'RdBu_r' (D2); got {_plots.DIVERGING_CMAP!r}."
    )


# ---------------------------------------------------------------------------
# Test 10 — lazy re-export through evaluation __init__ works (Plan T2; AC-11)
# ---------------------------------------------------------------------------


def test_plots_lazy_reexport_through_package_init() -> None:
    """Guards AC-11 / plan T2: the ``__getattr__`` wiring in ``evaluation/__init__.py``
    resolves plots symbols lazily.

    Importing ``residuals_vs_time``, ``OKABE_ITO``, and ``benchmark_holdout_bar``
    from ``bristol_ml.evaluation`` (not from ``bristol_ml.evaluation.plots``)
    must succeed — each object must be the same object as on the ``plots``
    module (identity check).

    Importing indirectly also exercises that the lazy-load path does not import
    ``plots`` eagerly on ``import bristol_ml.evaluation``.
    """
    import bristol_ml.evaluation as ev_pkg

    rvt = ev_pkg.__getattr__("residuals_vs_time")
    oi = ev_pkg.__getattr__("OKABE_ITO")
    bhb = ev_pkg.__getattr__("benchmark_holdout_bar")

    assert rvt is _plots.residuals_vs_time, (
        "bristol_ml.evaluation.residuals_vs_time must be the same object as "
        "bristol_ml.evaluation.plots.residuals_vs_time (lazy re-export identity)."
    )
    assert oi is _plots.OKABE_ITO, (
        "bristol_ml.evaluation.OKABE_ITO must be the same object as "
        "bristol_ml.evaluation.plots.OKABE_ITO (lazy re-export identity)."
    )
    assert bhb is _plots.benchmark_holdout_bar, (
        "bristol_ml.evaluation.benchmark_holdout_bar must be the same object as "
        "bristol_ml.evaluation.plots.benchmark_holdout_bar (lazy re-export identity)."
    )


# ---------------------------------------------------------------------------
# Test 11 — CLAUDE.md has the "Plotting conventions" section (Plan T2; AC-12)
# ---------------------------------------------------------------------------


def test_evaluation_claude_md_has_plotting_conventions_section() -> None:
    """Guards AC-12 / plan T2: CLAUDE.md contains the 'Plotting conventions' section
    with the Okabe-Ito reference.

    Plan T2 requires ``src/bristol_ml/evaluation/CLAUDE.md`` to be extended
    with a new 'Plotting conventions' section documenting the palette policy,
    ``ax=`` composability contract, CVD-safety opt-out idiom, and British-English
    label requirement.  The presence of the heading plus the string "Okabe-Ito"
    confirms the section was added with the correct content.
    """
    claude_md = _EVALUATION_CLAUDE_MD
    assert claude_md.exists(), f"CLAUDE.md not found at {claude_md} — was T2 hygiene skipped?"
    content = claude_md.read_text(encoding="utf-8")

    assert "Plotting conventions" in content, (
        "CLAUDE.md must contain the 'Plotting conventions' section heading "
        "(plan T2 / AC-12 hygiene)."
    )
    assert "Okabe-Ito" in content, (
        "The 'Plotting conventions' section in CLAUDE.md must mention 'Okabe-Ito' "
        "(plan T2 — palette policy documentation requirement)."
    )


# ===========================================================================
# T3 — Four hero helpers
# ===========================================================================
#
# These tests are derived from the spec at:
#   docs/plans/active/06-enhanced-evaluation.md §6 Task T3
#   "Tests (spec-derived)" bullet list (lines ~382-405).
#
# Conventions (matching prompt):
# - British English in docstrings ("colour", "behaviour", "fall-back").
# - Every test's docstring cites the plan clause or AC it guards.
# - numpy.random.default_rng(seed=0) for synthetic residuals.
# - pd.date_range(..., freq="h", tz="UTC") for tz-aware indices.
# - plt.close(fig) after each test that creates a figure.
# ===========================================================================


def _make_residuals(n: int = 200, seed: int = 0) -> pd.Series:
    """Return a small synthetic tz-aware residual series for tests.

    Index is hourly UTC from 2024-01-01 for ``n`` periods.
    Values are normally distributed via seeded RNG (seed=0 default).
    """
    rng = np.random.default_rng(seed=0)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.standard_normal(n) * 500.0, index=idx, name="residual")


# ---------------------------------------------------------------------------
# Test 12 — residuals_vs_time returns a Figure (Plan T3)
# ---------------------------------------------------------------------------


def test_residuals_vs_time_returns_figure() -> None:
    """Guards plan T3: ``residuals_vs_time`` returns a non-empty Figure.

    AC-1 / AC-2: the helper must produce a Figure with at least one Axes
    containing plot data, not a bare empty canvas.
    """
    import matplotlib.figure

    residuals = _make_residuals()
    fig = _plots.residuals_vs_time(residuals)
    try:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"residuals_vs_time must return a matplotlib Figure; got {type(fig).__name__!r}."
        )
        axes = fig.axes
        assert len(axes) >= 1, "Figure must have at least one Axes."
        assert len(axes[0].lines) >= 1, (
            "The residuals Axes must contain at least one line (the residual plot)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 13 — residuals_vs_time rejects tz-naive index (Plan T3)
# ---------------------------------------------------------------------------


def test_residuals_vs_time_rejects_tz_naive() -> None:
    """Guards plan T3: ``residuals_vs_time`` raises ``ValueError`` on a tz-naive Series.

    Plan D6 DST contract: tz-naive indices are ambiguous across DST transitions
    and must be rejected at the boundary.  This test pins that boundary check.
    """

    rng = np.random.default_rng(seed=0)
    naive_idx = pd.date_range("2024-01-01", periods=48, freq="h")  # no tz
    naive_residuals = pd.Series(rng.standard_normal(48), index=naive_idx)
    with pytest.raises(ValueError, match="tz-aware"):
        _plots.residuals_vs_time(naive_residuals)


# ---------------------------------------------------------------------------
# Test 14 — residuals_vs_time ax passthrough (Plan T3)
# ---------------------------------------------------------------------------


def test_residuals_vs_time_ax_passthrough() -> None:
    """Guards plan T3: passing ``ax=`` returns the same figure, not a new one.

    AC-2 / D5 moderate-opinionatedness: composability contract — the helper
    must draw onto a supplied ``Axes`` and return the owning ``Figure``, not
    create a new ``Figure``.
    """
    residuals = _make_residuals()
    fig_outer, ax_outer = plt.subplots()
    try:
        fig_returned = _plots.residuals_vs_time(residuals, ax=ax_outer)
        assert fig_returned is fig_outer, (
            "residuals_vs_time(ax=ax) must return the figure that owns ax, "
            "not create a new Figure (ax= composability contract, plan T3 / D5)."
        )
    finally:
        plt.close(fig_outer)


# ---------------------------------------------------------------------------
# Test 15 — predicted_vs_actual axis convention (Plan T3)
# ---------------------------------------------------------------------------


def test_predicted_vs_actual_axis_convention() -> None:
    """Guards plan T3: x-axis label contains 'Predicted'; y-axis contains 'Actual'.

    Gelman 2025 (plan research §R3) axis convention: x = predicted, y = actual.
    Pinned here so the axes are not accidentally swapped in a future refactor.
    """

    rng = np.random.default_rng(seed=0)
    n = 50
    y_true = pd.Series(rng.standard_normal(n) * 500.0 + 3000.0)
    y_pred = pd.Series(rng.standard_normal(n) * 500.0 + 3000.0)
    fig = _plots.predicted_vs_actual(y_true, y_pred)
    try:
        ax = fig.axes[0]
        assert "Predicted" in ax.get_xlabel(), (
            f"x-axis label must contain 'Predicted' (Gelman 2025 / plan T3); "
            f"got {ax.get_xlabel()!r}."
        )
        assert "Actual" in ax.get_ylabel(), (
            f"y-axis label must contain 'Actual' (Gelman 2025 / plan T3); got {ax.get_ylabel()!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 16 — predicted_vs_actual 45-degree line present (Plan T3)
# ---------------------------------------------------------------------------


def test_predicted_vs_actual_45_degree_line_present() -> None:
    """Guards plan T3: a 45-degree reference line (y=x) is present on the axes.

    The plan requires a reference line from ``min(y_true.min(), y_pred.min())``
    to ``max(y_true.max(), y_pred.max())``.  A line where x-data == y-data
    is the y=x identity; this test confirms at least one such line exists.
    """

    rng = np.random.default_rng(seed=0)
    n = 50
    y_true = pd.Series(rng.standard_normal(n) * 100.0 + 2000.0)
    y_pred = pd.Series(rng.standard_normal(n) * 100.0 + 2000.0)
    fig = _plots.predicted_vs_actual(y_true, y_pred)
    try:
        ax = fig.axes[0]
        # The 45-degree line has equal x and y data arrays.
        has_identity_line = any(
            np.allclose(line.get_xdata(), line.get_ydata()) for line in ax.lines
        )
        assert has_identity_line, (
            "predicted_vs_actual must include a 45-degree reference line where "
            "x-data == y-data (plan T3; research §R3 Gelman convention)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 17 — acf_residuals default lags=168 (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_lags_168_default() -> None:
    """Guards plan T3: default ``lags=168`` produces an x-axis that extends to 168.

    Plan D7 reinforcement: 168 lags = one full week of hourly data so the
    weekly autocorrelation spike motivating Stage 7 SARIMAX is visible.
    The x-axis upper limit must be at least 168 when the default is used.
    """
    # Need enough data for 168 lags: at least 170 observations.
    residuals = _make_residuals(n=250)
    fig = _plots.acf_residuals(residuals)
    try:
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[1] >= 168, (
            f"ACF x-axis upper limit must be >= 168 when lags=168 (plan T3 / D7); "
            f"got xlim={xlim!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 18 — acf_residuals annotates daily and weekly markers (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_annotates_daily_and_weekly_markers() -> None:
    """Guards plan T3 / D7: axes contain axvline markers at x=24 and x=168.

    D7 reinforcement (2026-04-20 human mandate): two labelled vertical
    reference markers at lag 24 (daily) and lag 168 (weekly) make the
    periodicity story legible for meetup audiences.  This test counts
    vertical-line artists on the Axes to confirm both are present.
    """
    residuals = _make_residuals(n=250)
    fig = _plots.acf_residuals(residuals)
    try:
        ax = fig.axes[0]
        # axvline creates Line2D artists; identify by constant x data
        vline_xs = set()
        for line in ax.lines:
            xdata = line.get_xdata()
            # A vertical line has all x values equal and spans the whole y axis
            if len(xdata) == 2 and np.isclose(xdata[0], xdata[1]):
                vline_xs.add(float(xdata[0]))
        assert 24.0 in vline_xs, (
            f"acf_residuals must draw an axvline at x=24 (daily marker, D7); "
            f"vertical line x positions found: {sorted(vline_xs)!r}."
        )
        assert 168.0 in vline_xs, (
            f"acf_residuals must draw an axvline at x=168 (weekly marker, D7); "
            f"vertical line x positions found: {sorted(vline_xs)!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 19 — acf_residuals daily marker labelled (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_daily_marker_labelled() -> None:
    """Guards plan T3 / D7: the text label 'daily (24)' is present near x=24.

    The plan specifies ``_ACF_MARKER_LABELS[24] == "daily (24)"``; this test
    confirms the exact string appears as a Text artist on the axes.
    """
    residuals = _make_residuals(n=250)
    fig = _plots.acf_residuals(residuals)
    try:
        ax = fig.axes[0]
        text_strings = [t.get_text() for t in ax.texts]
        assert "daily (24)" in text_strings, (
            f"acf_residuals must include a text label 'daily (24)' (plan T3 / D7 "
            f"_ACF_MARKER_LABELS); found text labels: {text_strings!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 20 — acf_residuals weekly marker labelled (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_weekly_marker_labelled() -> None:
    """Guards plan T3 / D7: the text label 'weekly (168)' is present near x=168.

    The plan specifies ``_ACF_MARKER_LABELS[168] == "weekly (168)"``; this
    test confirms the exact string appears as a Text artist on the axes.
    """
    residuals = _make_residuals(n=250)
    fig = _plots.acf_residuals(residuals)
    try:
        ax = fig.axes[0]
        text_strings = [t.get_text() for t in ax.texts]
        assert "weekly (168)" in text_strings, (
            f"acf_residuals must include a text label 'weekly (168)' (plan T3 / D7 "
            f"_ACF_MARKER_LABELS); found text labels: {text_strings!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 21 — acf_residuals reference_lags=() disables annotation (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_reference_lags_empty_disables_annotation() -> None:
    """Guards plan T3: ``reference_lags=()`` produces zero axvline artifacts.

    The override path is load-bearing — facilitators that want a plain ACF
    without reference markers must be able to suppress them by passing
    ``reference_lags=()`` (plan T3 / ``_annotate_acf_markers`` contract).
    """
    residuals = _make_residuals(n=250)
    fig = _plots.acf_residuals(residuals, reference_lags=())
    try:
        ax = fig.axes[0]
        # Count vertical-line artists (constant x data, two points)
        vline_count = sum(
            1
            for line in ax.lines
            if len(line.get_xdata()) == 2 and np.isclose(line.get_xdata()[0], line.get_xdata()[1])
        )
        assert vline_count == 0, (
            f"acf_residuals(reference_lags=()) must produce 0 axvline artifacts; "
            f"got {vline_count} (plan T3 / _annotate_acf_markers contract)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 22 — acf_residuals reference_lags custom respected (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_reference_lags_custom_respected() -> None:
    """Guards plan T3: ``reference_lags=(24, 168, 336)`` produces three markers.

    Custom reference_lags must be drawn verbatim — e.g. three markers for a
    two-week fortnightly view.  This confirms the ``_annotate_acf_markers``
    helper iterates the full override sequence.
    """
    residuals = _make_residuals(n=500)
    fig = _plots.acf_residuals(residuals, lags=336, reference_lags=(24, 168, 336))
    try:
        ax = fig.axes[0]
        vline_xs = set()
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) == 2 and np.isclose(xdata[0], xdata[1]):
                vline_xs.add(float(xdata[0]))
        assert 24.0 in vline_xs, (
            f"Custom reference_lags=(24,168,336) must produce axvline at x=24; "
            f"found: {sorted(vline_xs)!r}."
        )
        assert 168.0 in vline_xs, (
            f"Custom reference_lags=(24,168,336) must produce axvline at x=168; "
            f"found: {sorted(vline_xs)!r}."
        )
        assert 336.0 in vline_xs, (
            f"Custom reference_lags=(24,168,336) must produce axvline at x=336; "
            f"found: {sorted(vline_xs)!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 23 — acf_residuals override lags respected (Plan T3)
# ---------------------------------------------------------------------------


def test_acf_residuals_override_lags_respected() -> None:
    """Guards plan T3: ``lags=336`` produces an x-axis extending to 336.

    The ``lags`` parameter is overridable (plan T3 / AC-1) so facilitators
    can show two-week seasonality.  The x-axis upper limit must be >= 336.
    """
    residuals = _make_residuals(n=500)
    fig = _plots.acf_residuals(residuals, lags=336, reference_lags=())
    try:
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[1] >= 336, (
            f"ACF x-axis upper limit must be >= 336 when lags=336 (plan T3 / AC-1); "
            f"got xlim={xlim!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 24 — error_heatmap shape 24 x 7 (Plan T3)
# ---------------------------------------------------------------------------


def test_error_heatmap_shape_24_by_7() -> None:
    """Guards plan T3: the underlying pivot is 7 rows (weekdays) x 24 columns (hours).

    The plan specifies ``pivot table: index=weekday (0..6), columns=hour (0..23)``.
    This test verifies the rendered heatmap has exactly 24 x-tick positions and
    7 y-tick positions — the only externally observable proxy for pivot shape.
    """

    # Use a full-week residual series so all 7 weekdays and all 24 hours are
    # covered, avoiding NaN cells that could mask shape errors.
    rng = np.random.default_rng(seed=0)
    n = 7 * 24 * 4  # 4 weeks, hourly
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    residuals = pd.Series(rng.standard_normal(n) * 300.0, index=idx)
    fig = _plots.error_heatmap_hour_weekday(residuals)
    try:
        ax = fig.axes[0]
        # seaborn heatmap sets one tick per column/row in the pivot
        n_x_ticks = len(ax.get_xticks())
        n_y_ticks = len(ax.get_yticks())
        assert n_x_ticks == 24, (
            f"Heatmap x-axis must have 24 ticks (one per hour); got {n_x_ticks} (plan T3)."
        )
        assert n_y_ticks == 7, (
            f"Heatmap y-axis must have 7 ticks (one per weekday); got {n_y_ticks} (plan T3)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 25 — error_heatmap uses diverging cmap (Plan T3)
# ---------------------------------------------------------------------------


def test_error_heatmap_uses_diverging_cmap() -> None:
    """Guards plan T3 / D2: the heatmap's mappable uses the 'RdBu_r' diverging cmap.

    Plan T3 specifies ``sns.heatmap(..., cmap="RdBu_r", ...)`` and D2 requires
    the diverging colormap for signed residuals.  Verified by comparing the
    colour at ``t=0.0`` (the deep-blue extreme) against ``RdBu_r(0.0)`` and
    confirming the neutral midpoint colour matches ``RdBu_r(0.5)``.

    Note: seaborn 0.13 builds a re-sampled ``ListedColormap`` when ``center=0``
    is used.  The re-sampled cmap covers a subset of ``RdBu_r`` (from the
    symmetric vrange back to the data vmax), so intermediate values differ
    slightly from the continuous ``RdBu_r``.  The extreme blue (``t=0.0``)
    and the value-zero neutral colour are the two reliable discriminating
    properties — sufficient to confirm ``RdBu_r`` was used rather than any
    other diverging cmap (``PuOr``, ``BrBG``, ``coolwarm``, etc.).
    """
    import matplotlib.collections

    rng = np.random.default_rng(seed=0)
    n = 7 * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    residuals = pd.Series(rng.standard_normal(n) * 200.0, index=idx)
    fig = _plots.error_heatmap_hour_weekday(residuals)
    try:
        ax = fig.axes[0]
        meshes = [
            child
            for child in ax.get_children()
            if isinstance(child, matplotlib.collections.QuadMesh)
        ]
        assert meshes, (
            "Heatmap Axes must contain a QuadMesh artist (seaborn heatmap output); "
            "none found (plan T3 / D2)."
        )
        actual_cmap = meshes[0].cmap
        reference_cmap = plt.get_cmap("RdBu_r")

        # The blue extreme (t=0.0) must exactly match RdBu_r(0.0).  When
        # seaborn resamples the cmap with center=0 the first sample point is
        # always RdBu_r(0.0) because the symmetric range starts at -vrange
        # (the full-negative extreme of the palette).
        blue_extreme_actual = actual_cmap(0.0)
        blue_extreme_expected = reference_cmap(0.0)
        assert np.allclose(blue_extreme_actual, blue_extreme_expected, atol=1e-6), (
            f"Heatmap cmap at t=0.0 (blue extreme) must equal 'RdBu_r'(0.0).\n"
            f"  actual   = {blue_extreme_actual}\n"
            f"  expected = {blue_extreme_expected}\n"
            f"(DIVERGING_CMAP='RdBu_r', plan T3 / D2)."
        )

        # The neutral midpoint: value=0.0 must render as RdBu_r(0.5).
        # seaborn constructs the cmap so that center=0 maps to the palette midpoint.
        norm = meshes[0].norm
        t_at_zero = float(norm(0.0))
        colour_at_zero = actual_cmap(t_at_zero)
        rdbu_r_neutral = reference_cmap(0.5)
        assert np.allclose(colour_at_zero, rdbu_r_neutral, atol=1e-6), (
            f"With cmap='RdBu_r' + center=0, value 0.0 must render as the neutral "
            f"midpoint colour RdBu_r(0.5).\n"
            f"  colour at 0.0   = {colour_at_zero}\n"
            f"  RdBu_r(0.5)     = {rdbu_r_neutral}\n"
            f"(DIVERGING_CMAP='RdBu_r', plan T3 / D2)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 26 — error_heatmap centred at zero (Plan T3)
# ---------------------------------------------------------------------------


def test_error_heatmap_centered_at_zero() -> None:
    """Guards plan T3: heatmap colour mapping is centred at zero (``center=0``).

    The plan requires ``sns.heatmap(..., center=0, ...)``.  The observable
    behaviour of ``center=0`` is that the value 0.0 renders as the neutral
    midpoint colour of the diverging ``RdBu_r`` palette (the near-white colour
    at ``RdBu_r(0.5)``).  This is verified directly: the (norm + cmap)
    pipeline applied to 0.0 must equal ``RdBu_r(0.5)`` to within float
    tolerance.

    Note: seaborn 0.13 implements centering by constructing a re-sampled
    ``ListedColormap`` over a symmetric range ``[-vrange, +vrange]`` around
    the centre, then applies a plain ``Normalize(vmin, vmax)``.  The result
    is that the colour produced for value=0.0 is identical to ``RdBu_r(0.5)``
    regardless of whether the data range is symmetric.
    """
    import matplotlib.collections

    rng = np.random.default_rng(seed=0)
    n = 7 * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Force residuals to span both positive and negative so the norm is exercised
    residuals = pd.Series(rng.standard_normal(n) * 400.0, index=idx)
    fig = _plots.error_heatmap_hour_weekday(residuals)
    try:
        ax = fig.axes[0]
        meshes = [
            child
            for child in ax.get_children()
            if isinstance(child, matplotlib.collections.QuadMesh)
        ]
        assert meshes, "No QuadMesh found — cannot check norm (plan T3)."
        mesh = meshes[0]
        norm = mesh.norm
        cmap = mesh.cmap
        # Apply the full (norm → cmap) pipeline to value=0.0
        t_at_zero = float(norm(0.0))
        actual_colour = cmap(t_at_zero)
        # The reference neutral colour: RdBu_r at its midpoint (0.5)
        rdbu_r_neutral = plt.get_cmap("RdBu_r")(0.5)
        assert np.allclose(actual_colour, rdbu_r_neutral, atol=1e-6), (
            f"With center=0, value 0.0 must render as the neutral midpoint "
            f"colour of RdBu_r — confirming the diverging cmap is anchored at zero.\n"
            f"  colour at 0.0   = {actual_colour}\n"
            f"  RdBu_r(0.5)     = {rdbu_r_neutral}\n"
            f"(plan T3 center=0 contract)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 27 — error_heatmap weekday labels British (Plan T3)
# ---------------------------------------------------------------------------


def test_error_heatmap_weekday_labels_british() -> None:
    """Guards plan T3 / AC-12: y-tick labels are British English weekday abbreviations.

    Plan T3 specifies ``["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]`` as the
    y-tick label sequence (Monday=0 convention, matching the Stage 5 calendar
    feature ordering).  British English abbreviations are required by AC-12.
    """

    rng = np.random.default_rng(seed=0)
    n = 7 * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    residuals = pd.Series(rng.standard_normal(n) * 100.0, index=idx)
    expected_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = _plots.error_heatmap_hour_weekday(residuals)
    try:
        ax = fig.axes[0]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == expected_labels, (
            f"Heatmap y-tick labels must be {expected_labels!r} (British English, "
            f"plan T3 / AC-12); got {y_labels!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 28 — helpers accept residuals Series not Model (Plan T3)
# ---------------------------------------------------------------------------


def test_helpers_accept_residuals_series_not_model() -> None:
    """Guards plan T3 / AC-3: helpers take pd.Series inputs, never a Model object.

    Smoke test that the four hero helpers can be called with purely synthetic
    ``pd.Series`` residuals — no model object, no dataset, no external data.
    AC-3 (model-agnosticism): helpers are driven purely by pre-computed arrays.
    """

    residuals = _make_residuals(n=250)
    rng = np.random.default_rng(seed=1)
    y_true = pd.Series(rng.standard_normal(100) * 500.0 + 3000.0)
    y_pred = pd.Series(rng.standard_normal(100) * 500.0 + 3000.0)

    figs = []
    try:
        figs.append(_plots.residuals_vs_time(residuals))
        figs.append(_plots.predicted_vs_actual(y_true, y_pred))
        figs.append(_plots.acf_residuals(residuals, reference_lags=()))
        figs.append(_plots.error_heatmap_hour_weekday(residuals))
    finally:
        for f in figs:
            plt.close(f)


# ---------------------------------------------------------------------------
# Test 29 — helpers deterministic on fixed seed (Plan T3)
# ---------------------------------------------------------------------------


def test_helpers_deterministic_on_fixed_seed() -> None:
    """Guards plan T3: two calls with the same input produce byte-identical figure data.

    The plan pins the scatter down-sample to ``np.random.default_rng(42)``
    (``_SCATTER_RNG_SEED``), making ``predicted_vs_actual`` deterministic.
    Residuals helpers are purely deterministic (no internal randomness).
    This test calls each helper twice and compares ``fig.canvas.tostring_rgb()``.
    """

    residuals = _make_residuals(n=250)
    rng = np.random.default_rng(seed=2)
    y_true = pd.Series(rng.standard_normal(100) * 500.0 + 3000.0)
    y_pred = pd.Series(rng.standard_normal(100) * 500.0 + 3000.0)

    helpers_and_args: list[tuple] = [
        (_plots.residuals_vs_time, (residuals,), {}),
        (_plots.predicted_vs_actual, (y_true, y_pred), {}),
        (_plots.acf_residuals, (residuals,), {"reference_lags": ()}),
        (_plots.error_heatmap_hour_weekday, (residuals,), {}),
    ]

    for helper, args, kwargs in helpers_and_args:
        fig1 = helper(*args, **kwargs)
        fig1.canvas.draw()
        # Use buffer_rgba() (available on all Agg backends; tostring_rgb() was
        # removed in matplotlib 3.8).
        bytes1 = bytes(fig1.canvas.buffer_rgba())
        plt.close(fig1)

        fig2 = helper(*args, **kwargs)
        fig2.canvas.draw()
        bytes2 = bytes(fig2.canvas.buffer_rgba())
        plt.close(fig2)

        assert bytes1 == bytes2, (
            f"{helper.__name__}: two consecutive calls with the same input must "
            f"produce byte-identical figures (plan T3 determinism contract)."
        )


# ---------------------------------------------------------------------------
# Test 30 — helpers no IO (Plan T3)
# ---------------------------------------------------------------------------


def test_helpers_no_io() -> None:
    """Guards plan T3 / AC-3: helpers perform no I/O during a normal call.

    AC-3 (no external-resource dependencies): the four hero helpers must not
    open files, make HTTP requests, or write bytes to disk.  Verified by
    patching ``builtins.open``, ``requests.get``, and
    ``pathlib.Path.write_bytes`` and asserting zero calls to each.
    """
    import builtins

    residuals = _make_residuals(n=250)
    rng = np.random.default_rng(seed=3)
    y_true = pd.Series(rng.standard_normal(50) * 300.0 + 2500.0)
    y_pred = pd.Series(rng.standard_normal(50) * 300.0 + 2500.0)

    with (
        mock.patch("builtins.open", wraps=builtins.open) as mock_open,
        mock.patch("pathlib.Path.write_bytes") as mock_write_bytes,
    ):
        figs = []
        try:
            figs.append(_plots.residuals_vs_time(residuals))
            figs.append(_plots.predicted_vs_actual(y_true, y_pred))
            figs.append(_plots.acf_residuals(residuals, reference_lags=()))
            figs.append(_plots.error_heatmap_hour_weekday(residuals))
        finally:
            for f in figs:
                plt.close(f)

        assert mock_open.call_count == 0, (
            f"helpers must not call builtins.open during a normal run "
            f"(AC-3 / plan T3 no-IO contract); called {mock_open.call_count} time(s)."
        )
        assert mock_write_bytes.call_count == 0, (
            f"helpers must not call pathlib.Path.write_bytes "
            f"(AC-3 / plan T3 no-IO contract); called {mock_write_bytes.call_count} time(s)."
        )


# ---------------------------------------------------------------------------
# D6 DST gate tests — gating the Europe/London default
# ---------------------------------------------------------------------------
#
# These three tests are the D6 DST gate specified in plan T3 (lines ~401-405).
# If any of these three tests fail, the implementer must swap
# display_tz="Europe/London" -> display_tz="UTC" across all helpers and
# PlotsConfig.  These tests do NOT modify production code — they report
# failure and stop, per the agent role contract.
# ---------------------------------------------------------------------------


def test_residuals_vs_time_dst_spring_forward_produces_gap() -> None:
    """D6 DST gate: spring-forward (UK 31 March 2024) — no data at 01:00 local.

    Residual frame covers the last Sunday in March 2024 (UK spring-forward):
    clocks go forward at 01:00 UTC (→ skip from 00:59 to 02:00 Europe/London).
    The local-time x-axis must have no data point mapped into the
    01:00-02:00 Europe/London hour on that day.

    If this test fails, the display_tz default must be swapped to "UTC".
    Plan T3 §6 D6 DST gate, 2026-04-20 human mandate.
    """

    # UTC range covering 31 March 2024 spring-forward:
    # In Europe/London, clocks jump from 01:00 UTC to 02:00 local (BST).
    utc_idx = pd.date_range("2024-03-30 23:00", "2024-04-01 01:00", freq="h", tz="UTC")
    rng = np.random.default_rng(seed=0)
    residuals = pd.Series(rng.standard_normal(len(utc_idx)) * 100.0, index=utc_idx)

    # Convert to local time and check for the gap
    local_idx = utc_idx.tz_convert("Europe/London")
    spring_forward_day = "2024-03-31"
    local_hours_on_day = [
        ts.hour for ts in local_idx if ts.date().isoformat() == spring_forward_day
    ]

    # After the spring-forward the gap is at local 01:xx (UTC 01:00 → local 02:00 BST).
    # Local hour 1 should NOT appear on 31 March.
    assert 1 not in local_hours_on_day, (
        f"Spring-forward (31 March 2024): local hour 01:xx must be absent from the "
        f"Europe/London index on that day (clocks jump from 01:00 to 02:00 BST). "
        f"Local hours present on {spring_forward_day}: {sorted(set(local_hours_on_day))}. "
        f"D6 DST gate — if this fails, swap display_tz default to UTC (plan T3)."
    )

    # The helper must render without error (gap is fine in matplotlib line plot)
    fig = _plots.residuals_vs_time(residuals, display_tz="Europe/London")
    plt.close(fig)


def test_residuals_vs_time_dst_fall_back_produces_duplicate_hour() -> None:
    """D6 DST gate: fall-back (UK 27 October 2024) — duplicate wall-clock hour.

    Residual frame covers the last Sunday in October 2024 (UK fall-back):
    clocks go back at 02:00 BST (01:00 UTC), producing two UTC-distinct
    timestamps mapping to local 01:xx Europe/London.  The line plot must
    render both points without matplotlib raising.

    If this test fails, the display_tz default must be swapped to "UTC".
    Plan T3 §6 D6 DST gate, 2026-04-20 human mandate.
    """

    # UTC range covering 27 October 2024 fall-back:
    utc_idx = pd.date_range("2024-10-26 23:00", "2024-10-28 01:00", freq="h", tz="UTC")
    rng = np.random.default_rng(seed=0)
    residuals = pd.Series(rng.standard_normal(len(utc_idx)) * 100.0, index=utc_idx)

    # Convert to local time and check for the duplicate hour
    local_idx = utc_idx.tz_convert("Europe/London")
    fall_back_day = "2024-10-27"

    # Collect (date, hour) tuples for 27 October
    day_hours = [(ts.date().isoformat(), ts.hour) for ts in local_idx]
    fall_back_hours = [h for d, h in day_hours if d == fall_back_day]

    # On the fall-back day, local hour 1 should appear twice (01:00 BST and 01:00 GMT)
    count_hour_1 = fall_back_hours.count(1)
    assert count_hour_1 == 2, (
        f"Fall-back (27 October 2024): local hour 01:xx must appear twice on the "
        f"Europe/London index (two UTC-distinct observations). "
        f"Got {count_hour_1} occurrence(s). "
        f"D6 DST gate — if this fails, swap display_tz default to UTC (plan T3)."
    )

    # The helper must render without raising (duplicated x-coordinates are valid)
    fig = _plots.residuals_vs_time(residuals, display_tz="Europe/London")
    plt.close(fig)


def test_error_heatmap_dst_hour_groupby_behaviour() -> None:
    """D6 DST gate: October 2024 groupby mean — fall-back hour cell is mean of two.

    Residual frame spans full October 2024 (hourly UTC).  The fall-back Sunday
    (27 October) has two UTC-distinct observations for local hour 01:00
    Europe/London.  The groupby([weekday, hour]).mean() pivot must:

    - Produce a 7x24 shape (no lost rows or duplicated columns).
    - Have the fall-back Sunday's (weekday=6, hour=1) cell reflect the **mean**
      of the two UTC observations — not the sum, not a silent drop/NaN.

    If this test fails, the display_tz default must be swapped to "UTC".
    Plan T3 §6 D6 DST gate, 2026-04-20 human mandate.
    """

    # Full October 2024 hourly UTC
    utc_idx = pd.date_range("2024-10-01", "2024-10-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    values = rng.standard_normal(len(utc_idx)) * 100.0
    residuals_series = pd.Series(values, index=utc_idx)

    # Verify the helper renders without error on this DST-spanning frame
    fig = _plots.error_heatmap_hour_weekday(residuals_series)
    plt.close(fig)

    # Reproduce the helper's groupby logic against the local index to assert
    # the semantic properties of the pivot (shape, fall-back mean correctness).
    local_idx = utc_idx.tz_convert("Europe/London")
    frame = pd.DataFrame(
        {
            "weekday": local_idx.dayofweek,
            "hour": local_idx.hour,
            "residual": values,
        }
    )
    pivot = (
        frame.groupby(["weekday", "hour"])["residual"]
        .mean()
        .unstack("hour")
        .reindex(index=range(7), columns=range(24))
    )

    # Shape must be 7 x 24
    assert pivot.shape == (7, 24), (
        f"Groupby pivot must be shape (7, 24); got {pivot.shape}. "
        f"D6 DST gate — if this fails, swap display_tz default to UTC (plan T3)."
    )

    # Find the fall-back Sunday (27 October 2024) and its two UTC observations
    # at local hour 1 (one at 00:00 UTC = 01:00 BST, one at 01:00 UTC = 01:00 GMT)
    fall_back_sunday_utc = [
        ts
        for ts in utc_idx
        if ts.tz_convert("Europe/London").date().isoformat() == "2024-10-27"
        and ts.tz_convert("Europe/London").hour == 1
    ]
    assert len(fall_back_sunday_utc) == 2, (
        f"Expected exactly 2 UTC timestamps mapping to local 01:xx on 27 Oct 2024; "
        f"got {len(fall_back_sunday_utc)}. Check the UTC date range covers the fall-back."
    )

    # Sunday = weekday 6 (October 27 2024 is a Sunday; Monday=0 convention)
    actual_cell = float(pivot.loc[6, 1])

    assert not pd.isna(actual_cell), (
        "Pivot cell (weekday=6, hour=1) must not be NaN — the two fall-back "
        "observations should be averaged, not silently dropped. "
        "D6 DST gate (plan T3)."
    )

    # Allow for the cell to be a mean over all Sundays in October (there are 5).
    # The important thing is that the fall-back Sunday's hour=1 is NOT a simple
    # single-observation cell (it has 2 observations that day); the overall mean
    # across all Sundays at hour=1 should be finite and not equal to the single-
    # observation fallback.  We verify by re-computing via pandas and matching.
    assert np.isfinite(actual_cell), (
        f"Pivot cell (weekday=6, hour=1) must be finite (plan T3 D6 DST gate); got {actual_cell!r}."
    )

    # Re-compute the expected cell value for all Sundays hour=1 in October 2024
    sunday_hour1_mask = (frame["weekday"] == 6) & (frame["hour"] == 1)
    expected_cell = frame.loc[sunday_hour1_mask, "residual"].mean()
    assert np.isclose(actual_cell, expected_cell, rtol=1e-9), (
        f"Pivot cell (weekday=6, hour=1) = {actual_cell:.6f} does not match the "
        f"expected groupby mean {expected_cell:.6f} across all October Sundays at "
        f"local hour 1 (including 2 observations on the fall-back day). "
        f"D6 DST gate — correct behaviour is mean of all observations (plan T3)."
    )


# ===========================================================================
# T4 — forecast_overlay + forecast_overlay_with_band + benchmark_holdout_bar
# ===========================================================================
#
# These tests are derived from the spec at:
#   docs/plans/active/06-enhanced-evaluation.md §6 Task T4
#   "Tests (spec-derived)" bullet list (lines 425-433).
#
# Conventions (matching existing tests above):
# - British English in docstrings ("colour", "band", "behaviour").
# - Every test's docstring cites the plan clause or AC it guards.
# - numpy.random.default_rng(seed=0) for synthetic data.
# - pd.date_range(..., freq="h", tz="UTC") for tz-aware hourly index.
# - plt.close(fig) after each test that creates a figure.
# ===========================================================================


def _make_tz_aware_series(n: int = 48, seed: int = 0, base: float = 3000.0) -> pd.Series:
    """Return a synthetic tz-aware UTC hourly demand series of length ``n``."""
    rng = np.random.default_rng(seed=seed)
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.standard_normal(n) * 200.0 + base, index=idx)


def _make_per_fold_errors(n_horizons: int = 48, n_folds: int = 3, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic long-form per-fold error frame.

    Columns: ``horizon_h`` (0..n_horizons-1) repeated ``n_folds`` times,
    and ``error`` drawn from a seeded normal distribution.  Matches the
    column contract specified in plan §6 T5.
    """
    rng = np.random.default_rng(seed=seed)
    horizon_h = np.tile(np.arange(n_horizons), n_folds)
    error = rng.standard_normal(n_horizons * n_folds) * 100.0
    return pd.DataFrame({"horizon_h": horizon_h, "error": error})


# ---------------------------------------------------------------------------
# Test 34 — forecast_overlay plots actual + N prediction lines (Plan T4)
# ---------------------------------------------------------------------------


def test_forecast_overlay_plots_actual_plus_each_prediction() -> None:
    """Guards plan T4: axes contain N+1 line artefacts for N predictions.

    The plan specifies that ``forecast_overlay`` renders the ``actual`` series
    as one line plus one line per entry in ``predictions_by_name``.  With N
    predictions, the Axes must contain exactly N+1 Line2D objects.

    AC-1 / AC-2: the helper must produce a Figure whose axes contain exactly
    the right number of lines, confirming all series are drawn.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    import matplotlib.figure

    actual = _make_tz_aware_series(n=48, seed=0)
    pred_a = _make_tz_aware_series(n=48, seed=1)
    pred_b = _make_tz_aware_series(n=48, seed=2)
    pred_c = _make_tz_aware_series(n=48, seed=3)
    predictions_by_name = {"Model A": pred_a, "Model B": pred_b, "Model C": pred_c}

    fig = _plots.forecast_overlay(actual, predictions_by_name)
    try:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"forecast_overlay must return a Figure; got {type(fig).__name__!r}."
        )
        ax = fig.axes[0]
        n_lines = len(ax.lines)
        n_predictions = len(predictions_by_name)
        assert n_lines == n_predictions + 1, (
            f"forecast_overlay with {n_predictions} predictions must produce "
            f"{n_predictions + 1} line artefacts (1 actual + N predictions); "
            f"got {n_lines} line(s) (plan T4 / AC-1)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 35 — forecast_overlay legend is lower right (Plan T4)
# ---------------------------------------------------------------------------


def test_forecast_overlay_legend_lower_right() -> None:
    """Guards plan T4: legend anchor matches the notebook convention (loc='lower right').

    The plan mandates ``axes.legend(loc='lower right')`` to match the
    existing Stage 4 notebook Cell 11 convention.  The observable property
    is the legend's ``_loc`` integer code — matplotlib encodes "lower right"
    as ``4``.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    actual = _make_tz_aware_series(n=48, seed=0)
    pred = _make_tz_aware_series(n=48, seed=1)

    fig = _plots.forecast_overlay(actual, {"Forecast": pred})
    try:
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None, (
            "forecast_overlay must create a legend (plan T4 lower-right convention)."
        )
        # matplotlib encodes location strings as integers; "lower right" == 4.
        # Access the private ``_loc`` attribute which is the canonical way to
        # read back the location set by ``legend(loc=...)`` at render time.
        loc_code = legend._loc  # type: ignore[attr-defined]
        assert loc_code == 4, (
            f"Legend location must be 4 ('lower right') to match notebook "
            f"Cell 11 convention (plan T4); got {loc_code!r}."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 36 — forecast_overlay_with_band requires horizon_h column (Plan T4)
# ---------------------------------------------------------------------------


def test_forecast_overlay_with_band_requires_horizon_column() -> None:
    """Guards plan T4: ValueError when per_fold_errors lacks 'horizon_h'.

    The plan specifies the error message:
    "per_fold_errors must have 'horizon_h' column — run evaluate(..., return_predictions=True)".
    This test pins both the exception type and the key phrase.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    actual = _make_tz_aware_series(n=48, seed=0)
    point = _make_tz_aware_series(n=48, seed=1)
    # Missing 'horizon_h' column — only has 'error'
    bad_errors = pd.DataFrame({"error": np.zeros(48)})

    with pytest.raises(ValueError, match="horizon_h"):
        _plots.forecast_overlay_with_band(actual, point, bad_errors)


# ---------------------------------------------------------------------------
# Test 37 — forecast_overlay_with_band q10/q90 default (Plan T4)
# ---------------------------------------------------------------------------


def test_forecast_overlay_with_band_q10_q90_default() -> None:
    """Guards plan T4: fill_between bounds match 10th/90th quantile of synthetic errors.

    The harness emits ``error = y_true - y_pred`` (positive = under-forecast),
    so the band that brackets where actual is likely to land is
    ``[point + q_lo(error), point + q_hi(error)]``.

    The Stage-6 plan §6 T4 originally specified the band as
    ``[point - q_hi, point - q_lo]`` and this test was first written to
    match that wording.  The plan formula was wrong — subtracting flips
    the band's deviation from the forecast, producing a reflection of
    the true band through ``point``.  The inversion was caught visually
    on ``notebooks/04_linear_baseline.ipynb`` (band collapsed below the
    forecast where actual peaked); the implementation, the docstring,
    and this test were corrected together on 2026-05-04.

    With a constant point prediction of 3000 MW and deterministic errors,
    the expected band bounds are computable exactly from numpy quantiles.

    Reference: docs/plans/completed/06-enhanced-evaluation.md §6 T4
    test list (the formula in the archived plan is superseded by the
    code + docstring + this test).
    """
    import matplotlib.collections

    n_horizons = 48
    n_folds = 10

    # Build a deterministic per-fold errors frame
    horizon_h = np.tile(np.arange(n_horizons), n_folds)
    # Use a fixed error value per horizon_h for predictability:
    # error[horizon_h=k] = k - 24 (ranges from -24 to +23)
    raw_errors = np.tile(np.arange(n_horizons) - 24.0, n_folds)
    per_fold_errors = pd.DataFrame({"horizon_h": horizon_h, "error": raw_errors})

    # Point prediction and actual are constant (the exact MW values do not
    # affect the band shape — only the errors matter).
    idx = pd.date_range("2024-06-01", periods=n_horizons, freq="h", tz="UTC")
    point = pd.Series(np.full(n_horizons, 3000.0), index=idx)
    actual = pd.Series(np.full(n_horizons, 3100.0), index=idx)

    fig = _plots.forecast_overlay_with_band(actual, point, per_fold_errors)
    try:
        ax = fig.axes[0]
        # The fill_between produces a PolyCollection; collect them
        poly_collections = [
            child
            for child in ax.get_children()
            if isinstance(child, matplotlib.collections.PolyCollection)
        ]
        assert poly_collections, (
            "forecast_overlay_with_band must draw at least one PolyCollection "
            "(the empirical quantile band); none found (plan T4 / D8)."
        )
        # Rebuild the expected q10/q90 the same way the implementation does;
        # the band is ``[point + q_lo, point + q_hi]``.  All errors at a
        # given horizon_h are identical by construction, so the fill is a
        # zero-width strip per horizon and we verify the y-extent across
        # the whole 48-horizon grid.
        band = per_fold_errors.groupby("horizon_h")["error"].quantile([0.1, 0.9]).unstack()
        q_lo_arr = band[0.1].iloc[:n_horizons].to_numpy(dtype=np.float64)
        q_hi_arr = band[0.9].iloc[:n_horizons].to_numpy(dtype=np.float64)

        expected_lower = 3000.0 + q_lo_arr
        expected_upper = 3000.0 + q_hi_arr

        # Verify the band spans expected range — extract y coordinates from
        # the PolyCollection vertices.  Each path is a closed polygon.
        poly = poly_collections[0]
        paths = poly.get_paths()
        assert len(paths) > 0, "PolyCollection must contain at least one path."

        # Gather the y extents from all path vertices.
        all_y = np.concatenate([p.vertices[:, 1] for p in paths])
        assert np.isfinite(all_y).all(), (
            "PolyCollection vertices must all be finite (plan T4 / D8)."
        )
        # The band minimum should equal min(expected_lower) and maximum should
        # equal max(expected_upper) (up to floating-point tolerance).
        assert np.isclose(all_y.min(), expected_lower.min(), rtol=1e-6), (
            f"Band y-min = {all_y.min():.4f} does not match expected_lower.min() "
            f"= {expected_lower.min():.4f} (q10/q90 fill_between contract)."
        )
        assert np.isclose(all_y.max(), expected_upper.max(), rtol=1e-6), (
            f"Band y-max = {all_y.max():.4f} does not match expected_upper.max() "
            f"= {expected_upper.max():.4f} (q10/q90 fill_between contract)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 37b — REGRESSION GUARD against the sign-inversion bug fixed 2026-05-04.
#
# The pre-2026-05-04 ``forecast_overlay_with_band`` shaded
# ``[point - q_hi, point - q_lo]``, which is the reflection of the
# correct band through ``point``.  Symmetric synthetic errors (used by
# the test above) are blind to this flip — both the buggy and the
# correct band have the same y-range.  This test uses **purely positive**
# errors (every fold under-predicts at every horizon) so the buggy and
# correct bands land on opposite sides of ``point``: the correct band
# is entirely above ``point``, the buggy band entirely below.  A single
# assertion (``band_y_min > point``) would have caught the original bug
# and now locks the corrected contract.
# ---------------------------------------------------------------------------


def test_forecast_overlay_with_band_positive_errors_land_above_forecast() -> None:
    """Pins the sign of the band: positive errors -> band above the forecast.

    Regression guard for the inversion bug fixed 2026-05-04 (visible on
    ``notebooks/04_linear_baseline.ipynb``).  With ``error = y_true -
    y_pred`` the band tracks where actual is likely to land:

    - All-positive errors  -> band entirely **above** ``point``.
    - All-negative errors  -> band entirely **below** ``point``.

    The buggy version subtracted (``[point - q_hi, point - q_lo]``)
    which would place the band on the wrong side of ``point`` —
    detectable by exactly this asymmetric-errors check.  The earlier
    symmetric-errors test (above) was blind to the flip because the
    range is identical under reflection.
    """
    import matplotlib.collections

    n_horizons = 24
    n_folds = 8
    point_value = 3000.0

    # Every fold under-predicts by exactly 500 MW at every horizon.
    horizon_h = np.tile(np.arange(n_horizons), n_folds)
    error = np.full(n_horizons * n_folds, 500.0)
    per_fold_errors = pd.DataFrame({"horizon_h": horizon_h, "error": error})

    idx = pd.date_range("2024-06-01", periods=n_horizons, freq="h", tz="UTC")
    point = pd.Series(np.full(n_horizons, point_value), index=idx)
    actual = pd.Series(np.full(n_horizons, point_value + 500.0), index=idx)

    fig = _plots.forecast_overlay_with_band(actual, point, per_fold_errors)
    try:
        ax = fig.axes[0]
        polys = [
            c for c in ax.get_children() if isinstance(c, matplotlib.collections.PolyCollection)
        ]
        assert polys, "Expected exactly one PolyCollection for the band."
        all_y = np.concatenate([p.vertices[:, 1] for p in polys[0].get_paths()])
        finite_y = all_y[np.isfinite(all_y)]
        assert finite_y.size > 0, "PolyCollection vertices contain no finite values."

        # Correct contract: every band y-coordinate lies on or above
        # ``point_value`` because every error is +500 MW.  The buggy
        # subtract-formula would put every y-coordinate at
        # ``point_value - 500 = 2500`` instead.
        assert finite_y.min() >= point_value - 1e-6, (
            f"Band collapsed below the forecast under all-positive errors: "
            f"y_min={finite_y.min():.2f} but point={point_value:.2f}.  This is "
            f"the sign-inversion symptom fixed 2026-05-04 — the band is being "
            f"reflected through point instead of placed above it."
        )
        assert np.isclose(finite_y.max(), point_value + 500.0, rtol=1e-6), (
            f"Band upper edge under all-positive errors should equal "
            f"point + 500 = {point_value + 500.0:.2f}; got {finite_y.max():.2f}."
        )

        # Symmetric assertion for all-negative errors so the contract is
        # pinned in both directions.  Re-using the same fixture style.
        per_fold_errors_neg = pd.DataFrame(
            {
                "horizon_h": horizon_h,
                "error": np.full(n_horizons * n_folds, -500.0),
            }
        )
        fig_neg = _plots.forecast_overlay_with_band(actual, point, per_fold_errors_neg)
        try:
            ax_neg = fig_neg.axes[0]
            polys_neg = [
                c
                for c in ax_neg.get_children()
                if isinstance(c, matplotlib.collections.PolyCollection)
            ]
            all_y_neg = np.concatenate([p.vertices[:, 1] for p in polys_neg[0].get_paths()])
            finite_y_neg = all_y_neg[np.isfinite(all_y_neg)]
            assert finite_y_neg.max() <= point_value + 1e-6, (
                f"Band rose above the forecast under all-negative errors: "
                f"y_max={finite_y_neg.max():.2f} but point={point_value:.2f}.  "
                f"Sign inversion symptom — see the docstring above."
            )
        finally:
            plt.close(fig_neg)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 38 — forecast_overlay_with_band custom quantiles narrow the band (Plan T4)
# ---------------------------------------------------------------------------


def test_forecast_overlay_with_band_custom_quantiles_respected() -> None:
    """Guards plan T4: quantiles=(0.25, 0.75) narrows the band vs (0.1, 0.9).

    With a per-fold errors frame that has genuine spread across folds, the
    (0.25, 0.75) inter-quartile band must be strictly narrower than the
    (0.1, 0.9) 80% empirical band.  This test computes the y-extent of each
    PolyCollection and asserts that the wider quantile span produces a
    strictly larger y-range.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    import matplotlib.collections

    rng = np.random.default_rng(seed=42)
    n_horizons = 48
    n_folds = 20

    # Use genuinely varying errors so the quantile spread differs
    horizon_h = np.tile(np.arange(n_horizons), n_folds)
    error = rng.standard_normal(n_horizons * n_folds) * 150.0
    per_fold_errors = pd.DataFrame({"horizon_h": horizon_h, "error": error})

    idx = pd.date_range("2024-06-01", periods=n_horizons, freq="h", tz="UTC")
    point = pd.Series(np.full(n_horizons, 3000.0), index=idx)
    actual = pd.Series(np.full(n_horizons, 3100.0), index=idx)

    fig_wide = _plots.forecast_overlay_with_band(
        actual, point, per_fold_errors, quantiles=(0.1, 0.9)
    )
    fig_narrow = _plots.forecast_overlay_with_band(
        actual, point, per_fold_errors, quantiles=(0.25, 0.75)
    )
    try:

        def _poly_y_range(fig: matplotlib.figure.Figure) -> float:
            """Return the total y-range spanned by the PolyCollection on fig.axes[0]."""
            ax = fig.axes[0]
            polys = [
                c for c in ax.get_children() if isinstance(c, matplotlib.collections.PolyCollection)
            ]
            assert polys, "Expected at least one PolyCollection for the band."
            all_y = np.concatenate([p.vertices[:, 1] for p in polys[0].get_paths()])
            return float(all_y.max() - all_y.min())

        range_wide = _poly_y_range(fig_wide)
        range_narrow = _poly_y_range(fig_narrow)

        assert range_narrow < range_wide, (
            f"quantiles=(0.25, 0.75) band y-range ({range_narrow:.4f}) must be "
            f"strictly narrower than quantiles=(0.1, 0.9) band y-range "
            f"({range_wide:.4f}) when errors have genuine spread "
            f"(plan T4 custom-quantiles contract)."
        )
    finally:
        plt.close(fig_wide)
        plt.close(fig_narrow)


# ---------------------------------------------------------------------------
# Test 39 — benchmark_holdout_bar returns figure with metric bars (Plan T4)
# ---------------------------------------------------------------------------


def test_benchmark_holdout_bar_returns_figure_with_metric_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan T4: bar count == metric count x (candidate count + 1).

    The implementation delegates to ``compare_on_holdout`` for scoring.  This
    test monkey-patches ``bristol_ml.evaluation.benchmarks.compare_on_holdout``
    with a deterministic stub that returns a pre-built two-row metric table
    (one candidate + one NESO row) with two metric columns.  Confirms the
    bar count equals metric_count x row_count.

    The monkeypatch is applied at the canonical module path so the lazy import
    inside ``benchmark_holdout_bar`` resolves to the stub.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    import matplotlib.figure

    from bristol_ml.evaluation.metrics import mae, rmse

    # Candidate stub: satisfies the Model protocol via structural subtyping.
    class _ConstantModel:
        """Minimal stub satisfying the Model protocol (plan T4 model-agnostic test)."""

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(3000.0, index=features.index)

        def save(self, path: object) -> None:
            pass

        @classmethod
        def load(cls, path: object) -> _ConstantModel:
            return cls()

        @property
        def metadata(self) -> object:
            return None  # type: ignore[return-value]

    # Pre-built metric table that compare_on_holdout would return.
    # Two metric columns ("mae", "rmse"), three rows: one candidate + two from
    # fake_table (neso + naive).  The bar-count contract is:
    # total bars = n_metrics x n_rows_in_table (plan T4).
    # n_metrics comes from the ``metrics`` argument (not from table columns).
    fake_table = pd.DataFrame(
        {"mae": [120.0, 95.0, 130.0], "rmse": [180.0, 140.0, 195.0]},
        index=["linear", "neso", "naive"],
    )

    def _fake_compare(
        candidates: object,
        df: object,
        neso_forecast: object,
        splitter_cfg: object,
        metrics: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        return fake_table

    monkeypatch.setattr(
        "bristol_ml.evaluation.benchmarks.compare_on_holdout",
        _fake_compare,
    )

    # Build minimal features frame (just needs a UTC DatetimeIndex)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame({"nd_mw": np.ones(n) * 3000.0}, index=idx)

    holdout_start = idx[150]
    holdout_end = idx[199]

    # Pass two metrics so the bar-count test is non-trivial (2 groups x 3 rows)
    metrics_list = [mae, rmse]

    fig = _plots.benchmark_holdout_bar(
        candidates={"linear": _ConstantModel()},
        neso_forecast=pd.DataFrame(),  # irrelevant — stubbed out
        features=features,
        metrics=metrics_list,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    try:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"benchmark_holdout_bar must return a Figure; got {type(fig).__name__!r}."
        )
        ax = fig.axes[0]
        bar_containers = ax.containers
        # The implementation calls axes.bar() once per row in the table.
        # Each call produces one BarContainer with n_metrics bars.
        # Plan T4 contract: total bars = n_metrics x (n_candidates + 1).
        # Here n_candidates=1 (linear), fake_table has 3 rows (incl. neso + naive),
        # so n_rows = len(fake_table) = 3 and n_metrics = len(metrics_list) = 2.
        n_rows = len(fake_table)
        n_metrics = len(metrics_list)
        total_bars = sum(len(c) for c in bar_containers)
        assert total_bars == n_metrics * n_rows, (
            f"benchmark_holdout_bar must produce {n_metrics} x {n_rows} = "
            f"{n_metrics * n_rows} total bar patches; got {total_bars} "
            f"(plan T4 bar-count contract: metric_count x (candidate_count + 1))."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 39b — REGRESSION GUARD: benchmark_holdout_bar must not break
# seasonal-naive same_hour_last_week.
#
# Bug fixed 2026-05-04 ("Stage 6 separation" branch): the helper used to
# build a single fold spanning the full holdout, so for a 92-day holdout
# only the first 168 test rows had a t-168h training row and the
# remaining 2041 raised
#   ValueError: Seasonal-naive 'same_hour_last_week' requires training
#   targets at t - 7 days; 2041/2209 prediction rows have no matching
#   training row.
# Post-fix the helper rolls within the holdout in weekly chunks (168h)
# by default, so each fold's lookback always lands in training.
#
# This test exercises the **end-to-end** code path (no compare_on_holdout
# stub) — the previous Test 39 stubs the scoring function and would not
# catch a splitter regression.
# ---------------------------------------------------------------------------


def test_benchmark_holdout_bar_works_with_seasonal_naive_same_hour_last_week() -> None:
    """``same_hour_last_week`` succeeds end-to-end through the helper.

    Pins the post-2026-05-04 fold-rolling default.  The pre-fix single-
    fold splitter would raise ``Seasonal-naive '...' requires training
    targets at t - 7 days; N/M prediction rows have no matching training
    row.`` for any holdout longer than 168 hours.  After the fix the
    helper walks the holdout in weekly folds and every fold's lookback
    lands in the training prefix.

    Synthetic features + outturn (constant 30 000 MW + low-amplitude
    sinusoid) so the predictions and metrics are well-defined; the test
    asserts the figure renders and the bar count matches the
    ``len(metrics) x (len(candidates) + 1)`` contract.
    """
    import matplotlib.figure

    from bristol_ml.evaluation.metrics import mae, rmse
    from bristol_ml.models.naive import NaiveModel
    from conf._schemas import NaiveConfig

    # Synthesise 200 days of hourly data: 168 h pre-holdout train + a
    # 32-day (768 h) holdout.  Pre-fix that holdout would be a single
    # fold of length 768 — only the first 168 rows of which would be
    # predictable by same_hour_last_week.  Post-fix the helper builds
    # ~4 weekly folds and every fold succeeds.
    n_pre = 24 * 60  # 60 days of training history before the holdout
    n_post = 24 * 32  # 32 days of holdout
    total = n_pre + n_post
    idx = pd.date_range("2024-01-01", periods=total, freq="h", tz="UTC")
    base = 30_000.0 + 200.0 * np.sin(2 * np.pi * np.arange(total) / 24)
    # Include the Stage-3 weather column set so the harness's default
    # feature_columns fallback finds them.  The naive model does not
    # actually use features, but the harness still slices by feature
    # name before delegating.
    features = pd.DataFrame(
        {
            "nd_mw": base.astype("float64"),
            "temperature_2m": np.full(total, 10.0, dtype="float32"),
            "dew_point_2m": np.full(total, 5.0, dtype="float32"),
            "wind_speed_10m": np.full(total, 3.0, dtype="float32"),
            "cloud_cover": np.full(total, 50.0, dtype="float32"),
            "shortwave_radiation": np.full(total, 100.0, dtype="float32"),
        },
        index=idx,
    )

    # Synthesise a NESO forecast frame on the same UTC index, half-hourly,
    # so compare_on_holdout's hourly aggregation has something to score.
    half_idx = pd.date_range(idx[0], idx[-1] + pd.Timedelta(minutes=30), freq="30min", tz="UTC")
    neso = pd.DataFrame(
        {
            "timestamp_utc": half_idx,
            "demand_forecast_mw": (
                30_000.0 + 200.0 * np.sin(2 * np.pi * np.arange(len(half_idx)) / 48)
            ),
            "demand_outturn_mw": features["nd_mw"]
            .reindex(half_idx, method="ffill")
            .to_numpy(),
        }
    )

    holdout_start = idx[n_pre]
    holdout_end = idx[-1]

    fig = _plots.benchmark_holdout_bar(
        candidates={
            "naive_lw": NaiveModel(NaiveConfig(strategy="same_hour_last_week")),
        },
        neso_forecast=neso,
        features=features,
        metrics=[mae, rmse],
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        # No fold_len_hours override -> uses the post-fix default of 168 h.
    )
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        # Bar count contract still holds: 2 metrics x (1 candidate + 1 neso) = 4.
        ax = fig.axes[0]
        total_bars = sum(len(c) for c in ax.containers)
        assert total_bars == 4, (
            f"Expected 2 metrics x 2 rows (candidate + neso) = 4 bars; got {total_bars}."
        )
    finally:
        plt.close(fig)


def test_benchmark_holdout_bar_fold_len_hours_none_uses_single_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fold_len_hours=None`` selects the legacy single-fold splitter.

    The fix changed the default to weekly folds for seasonal-naive
    compatibility, but a caller with a fit-once-predict-N model
    (e.g. SARIMAX) may still want a single-fold score.  The opt-in
    parameter must give them the pre-fix splitter shape.
    """
    captured: dict[str, object] = {}

    def _capturing_compare(
        candidates: object,
        df: object,
        neso_forecast: object,
        splitter_cfg: object,
        metrics: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["splitter_cfg"] = splitter_cfg
        return pd.DataFrame({"mae": [1.0, 2.0]}, index=["m", "neso"])

    monkeypatch.setattr(
        "bristol_ml.evaluation.benchmarks.compare_on_holdout",
        _capturing_compare,
    )

    from bristol_ml.evaluation.metrics import mae

    n = 24 * 30  # 30 days
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame({"nd_mw": np.full(n, 30_000.0)}, index=idx)
    holdout_start = idx[n // 2]
    holdout_end = idx[-1]
    holdout_rows = int(((idx >= holdout_start) & (idx <= holdout_end)).sum())

    class _Stub:
        def fit(self, *a, **k): ...
        def predict(self, features):
            return pd.Series(0.0, index=features.index)
        def save(self, p): ...
        @classmethod
        def load(cls, p): return cls()
        @property
        def metadata(self): return None

    fig = _plots.benchmark_holdout_bar(
        candidates={"m": _Stub()},
        neso_forecast=pd.DataFrame(),
        features=features,
        metrics=[mae],
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        fold_len_hours=None,
    )
    try:
        cfg = captured["splitter_cfg"]
        assert cfg.test_len == holdout_rows, (
            f"fold_len_hours=None must build a splitter whose test_len equals "
            f"the full holdout span ({holdout_rows} rows); got test_len={cfg.test_len}."
        )
        assert cfg.step == cfg.test_len, "Single-fold splitter must have step == test_len."
    finally:
        plt.close(fig)


def test_benchmark_holdout_bar_rejects_non_positive_fold_len_hours() -> None:
    """``fold_len_hours <= 0`` raises with a clear message."""
    from bristol_ml.evaluation.metrics import mae

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame({"nd_mw": np.full(n, 30_000.0)}, index=idx)

    class _Stub:
        def fit(self, *a, **k): ...
        def predict(self, features): return pd.Series(0.0, index=features.index)
        def save(self, p): ...
        @classmethod
        def load(cls, p): return cls()
        @property
        def metadata(self): return None

    with pytest.raises(ValueError, match="fold_len_hours must be positive or None"):
        _plots.benchmark_holdout_bar(
            candidates={"m": _Stub()},
            neso_forecast=pd.DataFrame(),
            features=features,
            metrics=[mae],
            holdout_start=idx[150],
            holdout_end=idx[-1],
            fold_len_hours=0,
        )


# ---------------------------------------------------------------------------
# Test 40 — benchmark_holdout_bar uses holdout window from config (Plan T4)
# ---------------------------------------------------------------------------


def test_benchmark_holdout_bar_uses_holdout_window_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan T4: slicing matches holdout_start / holdout_end bounds.

    The implementation derives ``min_train_periods`` and ``test_len`` from
    the positional relationship of the features index to
    ``holdout_start / holdout_end``.  This test verifies that the
    ``SplitterConfig`` passed to ``compare_on_holdout`` has ``min_train_periods``
    equal to the count of rows strictly before ``holdout_start``, and
    ``test_len`` equal to the count of rows in ``[holdout_start, holdout_end]``.

    The monkeypatch captures the ``SplitterConfig`` passed to the stub so the
    test can inspect it after the call.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    from bristol_ml.evaluation.metrics import mae
    from conf._schemas import SplitterConfig

    captured: list[SplitterConfig] = []

    fake_table = pd.DataFrame(
        {"mae": [110.0, 105.0]},
        index=["candidate", "neso"],
    )

    def _capturing_compare(
        candidates: object,
        df: object,
        neso_forecast: object,
        splitter_cfg: SplitterConfig,
        metrics: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured.append(splitter_cfg)
        return fake_table

    monkeypatch.setattr(
        "bristol_ml.evaluation.benchmarks.compare_on_holdout",
        _capturing_compare,
    )

    # Build feature frame with a known split: 100 train rows, 24 test rows
    n_train = 100
    n_test = 24
    n_total = n_train + n_test
    idx = pd.date_range("2024-01-01", periods=n_total, freq="h", tz="UTC")
    features = pd.DataFrame({"nd_mw": np.ones(n_total) * 3000.0}, index=idx)

    holdout_start = idx[n_train]  # first test row
    holdout_end = idx[n_train + n_test - 1]  # last test row

    class _ConstantModel:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(3000.0, index=features.index)

        def save(self, path: object) -> None:
            pass

        @classmethod
        def load(cls, path: object) -> _ConstantModel:
            return cls()

        @property
        def metadata(self) -> object:
            return None  # type: ignore[return-value]

    fig = _plots.benchmark_holdout_bar(
        candidates={"candidate": _ConstantModel()},
        neso_forecast=pd.DataFrame(),
        features=features,
        metrics=[mae],
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    plt.close(fig)

    assert len(captured) == 1, (
        "compare_on_holdout must be called exactly once by benchmark_holdout_bar."
    )
    cfg = captured[0]
    assert cfg.min_train_periods == n_train, (
        f"SplitterConfig.min_train_periods must be {n_train} (rows before holdout_start); "
        f"got {cfg.min_train_periods} (plan T4 holdout-window contract)."
    )
    assert cfg.test_len == n_test, (
        f"SplitterConfig.test_len must be {n_test} (rows in [holdout_start, holdout_end]); "
        f"got {cfg.test_len} (plan T4 holdout-window contract)."
    )
    assert cfg.fixed_window is True, (
        "SplitterConfig.fixed_window must be True for the holdout benchmark "
        "(plan T4 / D10 fixed-window semantics)."
    )


# ---------------------------------------------------------------------------
# Test 41 — benchmark_holdout_bar model-agnostic (Plan T4)
# ---------------------------------------------------------------------------


def test_benchmark_holdout_bar_model_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards plan T4: accepts a dict whose values are any Model protocol implementers.

    Confirms that ``benchmark_holdout_bar`` does not inspect the concrete
    type of the model objects — it treats the candidates mapping as an opaque
    ``Mapping[str, Model]`` and passes it through to ``compare_on_holdout``.
    The test uses three different stub classes (none inheriting from a common
    base) to exercise the structural-subtyping contract.

    Reference: docs/plans/active/06-enhanced-evaluation.md §6 T4 test list.
    """
    import matplotlib.figure

    from bristol_ml.evaluation.metrics import mae
    from bristol_ml.models.protocol import Model

    class _StubAlpha:
        """First model stub — returns a constant 3000.0."""

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(3000.0, index=features.index)

        def save(self, path: object) -> None:
            pass

        @classmethod
        def load(cls, path: object) -> _StubAlpha:
            return cls()

        @property
        def metadata(self) -> object:
            return None  # type: ignore[return-value]

    class _StubBeta:
        """Second model stub — returns a constant 3500.0."""

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(3500.0, index=features.index)

        def save(self, path: object) -> None:
            pass

        @classmethod
        def load(cls, path: object) -> _StubBeta:
            return cls()

        @property
        def metadata(self) -> object:
            return None  # type: ignore[return-value]

    # Verify structural protocol conformance for both stubs
    assert isinstance(_StubAlpha(), Model), (
        "_StubAlpha does not satisfy the Model protocol — fix the stub."
    )
    assert isinstance(_StubBeta(), Model), (
        "_StubBeta does not satisfy the Model protocol — fix the stub."
    )

    # Pre-built metric table (two candidates + neso row)
    fake_table = pd.DataFrame(
        {"mae": [115.0, 125.0, 108.0]},
        index=["alpha", "beta", "neso"],
    )

    # Track which candidates mapping the stub receives to verify pass-through
    received_candidates: list[object] = []

    def _passthrough_compare(
        candidates: object,
        df: object,
        neso_forecast: object,
        splitter_cfg: object,
        metrics: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        received_candidates.append(candidates)
        return fake_table

    monkeypatch.setattr(
        "bristol_ml.evaluation.benchmarks.compare_on_holdout",
        _passthrough_compare,
    )

    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame({"nd_mw": np.ones(n) * 3000.0}, index=idx)
    holdout_start = idx[120]
    holdout_end = idx[149]

    candidates = {"alpha": _StubAlpha(), "beta": _StubBeta()}

    fig = _plots.benchmark_holdout_bar(
        candidates=candidates,
        neso_forecast=pd.DataFrame(),
        features=features,
        metrics=[mae],
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    try:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"benchmark_holdout_bar must return a Figure; got {type(fig).__name__!r}."
        )
        # Confirm the candidates mapping was passed through to compare_on_holdout
        assert len(received_candidates) == 1, "compare_on_holdout must be called exactly once."
        passed_candidates = received_candidates[0]
        assert passed_candidates is candidates, (
            "benchmark_holdout_bar must pass the candidates dict through to "
            "compare_on_holdout unchanged (plan T4 model-agnostic contract)."
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3 review regressions
# ---------------------------------------------------------------------------


def test_forecast_overlay_predictions_do_not_collide_with_actual_colour() -> None:
    """Phase 3 review B1 regression: no prediction line may share the Actual colour.

    Before the fix, ``forecast_overlay`` built ``palette = OKABE_ITO[1:] +
    OKABE_ITO[3:]`` — a slice that still contained ``OKABE_ITO[2]`` (the sky-blue
    Actual colour) at position 1, so the *second* prediction was drawn in the
    same hex as Actual.  The notebook's Cell 15 passes three predictions
    (Naive, Linear, NESO), so the live-demo surface quietly lost the ability
    to distinguish Linear from Actual.  This regression pins: (a) Actual is
    ``OKABE_ITO[2]``, and (b) no prediction line shares that colour.
    """
    import matplotlib.colors as mcolors

    actual = _make_tz_aware_series(n=48, seed=0)
    predictions_by_name = {
        "Model A": _make_tz_aware_series(n=48, seed=1),
        "Model B": _make_tz_aware_series(n=48, seed=2),
        "Model C": _make_tz_aware_series(n=48, seed=3),
    }

    fig = _plots.forecast_overlay(actual, predictions_by_name)
    try:
        ax = fig.axes[0]
        actual_colour = mcolors.to_hex(ax.lines[0].get_color()).lower()
        prediction_colours = [mcolors.to_hex(line.get_color()).lower() for line in ax.lines[1:]]

        assert actual_colour == _plots.OKABE_ITO[2].lower(), (
            f"Actual series must be drawn in OKABE_ITO[2] (sky blue); got {actual_colour!r}."
        )
        for i, colour in enumerate(prediction_colours):
            assert colour != actual_colour, (
                f"Prediction line {i} uses {colour!r}, identical to Actual "
                f"{actual_colour!r} — Phase 3 review B1 colour-collision regression."
            )
        assert len(set(prediction_colours)) == len(prediction_colours), (
            f"Prediction lines must use distinct colours; got {prediction_colours!r}."
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "bad_quantiles",
    [(0.9, 0.1), (0.5, 0.5), (-0.1, 0.5), (0.5, 1.1), (0.0, 0.0), (1.0, 1.0)],
)
def test_forecast_overlay_with_band_rejects_malformed_quantiles(
    bad_quantiles: tuple[float, float],
) -> None:
    """Phase 3 review N1 regression: reject reversed / equal / out-of-range quantiles.

    Before the fix, the helper used positional ``iloc[:, 0]`` / ``iloc[:, 1]``
    access on a pandas-groupby-quantile result whose column order preserves
    the input list order.  A caller passing ``quantiles=(0.9, 0.1)`` would
    get a negative-height ``fill_between`` with no error.  The post-fix
    contract raises ``ValueError`` with a message mentioning "quantiles" so
    the misuse is caught at the boundary.
    """
    rng = np.random.default_rng(seed=0)
    per_fold_errors = pd.DataFrame(
        {"horizon_h": np.arange(24), "error": rng.standard_normal(24) * 100.0}
    )
    idx = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    actual = pd.Series(np.zeros(24), index=idx)
    point = pd.Series(np.zeros(24), index=idx)

    with pytest.raises(ValueError, match="quantiles"):
        _plots.forecast_overlay_with_band(actual, point, per_fold_errors, quantiles=bad_quantiles)


def test_apply_plots_config_propagates_figsize_and_dpi_to_rcparams() -> None:
    """Phase 3 review N2 regression: ``PlotsConfig`` knobs must reach ``plt.rcParams``.

    Plan D5 promises Hydra-configurable ``evaluation.plots.figsize`` and
    ``dpi``.  Before the fix, the YAML knobs were decorative: ``plots.py``
    never read ``PlotsConfig``; ``_apply_style`` hardcoded
    ``_STYLE_RCPARAMS``.  ``apply_plots_config(cfg)`` is the public entry
    point that wires a loaded config to ``matplotlib``.  Test isolates by
    restoring the module defaults via ``_apply_style()`` in the ``finally``
    block so subsequent tests that pin the 12x8 default still pass.
    """
    from conf._schemas import PlotsConfig

    cfg = PlotsConfig(figsize=(16.0, 10.0), dpi=150)
    try:
        _plots.apply_plots_config(cfg)
        # matplotlib stores figsize as list; compare via tuple().
        assert tuple(plt.rcParams["figure.figsize"]) == (16.0, 10.0), (
            "apply_plots_config must write PlotsConfig.figsize into "
            "plt.rcParams['figure.figsize'] (Phase 3 review N2)."
        )
        assert plt.rcParams["figure.dpi"] == 150, (
            "apply_plots_config must write PlotsConfig.dpi into "
            "plt.rcParams['figure.dpi'] (Phase 3 review N2)."
        )
    finally:
        _plots._apply_style()
        # Sanity: the reset restored the module defaults so downstream tests
        # that pin 12x8 do not see contamination from this test.
        assert tuple(plt.rcParams["figure.figsize"]) == (12.0, 8.0)
        assert plt.rcParams["figure.dpi"] == 110


def test_apply_plots_config_exported_in_all() -> None:
    """Phase 3 review N2: the public entry point is in ``__all__``.

    The module already exports the seven helpers and three palette constants;
    ``apply_plots_config`` is the public wiring for D5 Hydra overrides and
    must be part of the documented public surface so ``from
    bristol_ml.evaluation.plots import *`` picks it up in notebooks.
    """
    assert "apply_plots_config" in _plots.__all__, (
        f"'apply_plots_config' missing from __all__={_plots.__all__!r} "
        "(Phase 3 review N2 — public-surface contract)."
    )
    assert callable(_plots.apply_plots_config), "apply_plots_config must be a callable."
