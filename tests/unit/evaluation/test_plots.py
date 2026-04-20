"""Spec-derived tests for ``bristol_ml.evaluation.plots`` — Stage 6 T2 scaffold.

Every test is derived from:

- ``docs/plans/active/06-enhanced-evaluation.md`` §6 T2 (explicit test list)
  and §4 AC-1/2/6/10/11/12.
- Plan decisions D2 (Okabe-Ito palette, cividis sequential, RdBu_r diverging),
  D5 (default figsize 12x8, human mandate 2026-04-20), D9 (CLAUDE.md
  architectural-debt note for harness output API growth trigger).
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
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

# Confirm the package is importable; skip cleanly if the install is absent.
pytest.importorskip("bristol_ml.evaluation.plots")

# Re-import-clean: load the module so we test the live state of rcParams
# after import-time side effects have run.  ``_plots`` and ``plots_mod``
# alias the same module so tests can reference either name interchangeably.
plots_mod = importlib.import_module("bristol_ml.evaluation.plots")
_plots = plots_mod


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
    claude_md = Path("/workspace/src/bristol_ml/evaluation/CLAUDE.md")
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
    claude_md = Path("/workspace/src/bristol_ml/evaluation/CLAUDE.md")
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
