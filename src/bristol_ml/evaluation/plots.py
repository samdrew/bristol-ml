"""Stage 6 diagnostic-plot helper library for the evaluation layer.

This module is the **Stage 6 diagnostic surface**: a small, opinionated set
of helper functions that render residual diagnostics, forecast overlays,
and a fixed-window NESO benchmark bar chart.  Every future modelling stage
(Stage 7 SARIMAX, Stage 8 tree-based, Stage 10 NN, Stage 11 ensembling,
Stage 16 weather-regime) drops its residuals into these helpers without
needing to know the caller; the helpers are model-agnostic (AC-3) — they
take ``pd.Series`` / ``pd.DataFrame`` inputs, never a ``Model`` object.

The public surface is seven functions plus three palette constants:

- :func:`residuals_vs_time` — line plot of residuals over the test index.
- :func:`predicted_vs_actual` — scatter with 45-degree reference line.
- :func:`acf_residuals` — statsmodels ACF with daily (lag 24) + weekly
  (lag 168) reference markers (D7 reinforcement).
- :func:`error_heatmap_hour_weekday` — 24x7 mean-signed-residual heatmap.
- :func:`forecast_overlay` — actual plus N prediction lines on a common
  time axis.
- :func:`forecast_overlay_with_band` — forecast_overlay plus an empirical
  quantile (q10-q90 default) uncertainty band derived from rolling-origin
  per-fold errors.
- :func:`benchmark_holdout_bar` — fixed-window bar chart for the NESO
  three-way benchmark comparison (D10 — wires up the latent
  ``NesoBenchmarkConfig.holdout_start/_end`` consumer).

Plus the palette constants (Stage 6 D2, pinned by Wong 2011 *Nature
Methods* and cividis / RdBu_r matplotlib built-ins):

- :data:`OKABE_ITO` — eight colourblind-safe qualitative hex values.
- :data:`SEQUENTIAL_CMAP` — ``"cividis"`` (perceptually uniform, CVD-safe).
- :data:`DIVERGING_CMAP` — ``"RdBu_r"`` (signed residuals).

**Palette injection.**  At import time the module calls :func:`_apply_style`
which writes six entries into ``plt.rcParams`` (prop-cycle, figsize, DPI,
axis/title/legend font sizes).  This is a module-level side effect — the
pedagogical cost (every notebook that imports this module inherits the
Okabe-Ito palette and 12x8 default figsize) is the pedagogical win
(every Stage 6+ plot in the repo looks the same for live demos).  To opt
out, call ``plt.rcdefaults()`` or a bespoke ``plt.rcParams.update(...)``
after the import.

**British English.**  All docstrings, axis labels, and legend entries use
British spellings ("colour", "visualisation") per ``CLAUDE.md``.

**Accessibility contract (AC-6).**  The default qualitative palette is
formally certified colourblind-safe for deuteranopia, protanopia, and
tritanopia (Wong 2011, *Nature Methods* 8:441).  ``tab10`` (matplotlib's
default) is explicitly rejected.

Run standalone::

    python -m bristol_ml.evaluation.plots [--help]

The CLI prints the exported helper names, the active palette, and the
rcParams the module has overridden — useful for a live-demo sanity check
and for satisfying DESIGN §2.1.1 (every module runs standalone).

The six helper bodies live in :func:`residuals_vs_time` through
:func:`benchmark_holdout_bar`; this module also exports the palette
constants as part of its documented public surface so downstream
notebooks can reuse them without a circular import.

Cross-references:

- Stage 6 plan — ``docs/plans/active/06-enhanced-evaluation.md`` (decisions
  D1-D11; the four hero helpers' ACF/heatmap/residuals/scatter layout
  maps to Hyndman fpp3 §5.3).
- Evaluation layer contract — ``docs/architecture/layers/evaluation.md``.
- Palette research — ``docs/lld/research/06-enhanced-evaluation-domain.md``
  §R2 (Okabe-Ito evidence base; cividis provenance).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.metrics import MetricFn
    from bristol_ml.models.protocol import Model


__all__ = [
    "DIVERGING_CMAP",
    "OKABE_ITO",
    "SEQUENTIAL_CMAP",
    "acf_residuals",
    "benchmark_holdout_bar",
    "error_heatmap_hour_weekday",
    "forecast_overlay",
    "forecast_overlay_with_band",
    "predicted_vs_actual",
    "residuals_vs_time",
]


# ---------------------------------------------------------------------------
# Palette constants (Stage 6 D2 — colourblind-safe)
# ---------------------------------------------------------------------------

#: Okabe-Ito qualitative palette (Wong 2011, *Nature Methods* 8:441).
#:
#: Eight colours chosen to be distinguishable under deuteranopia,
#: protanopia, and tritanopia.  Ordering matches the original paper so
#: downstream code can index by position (e.g. ``OKABE_ITO[5]`` is the
#: canonical "blue" used for reference-line annotations in
#: :func:`acf_residuals`).
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # 0: black          — axes / reference lines
    "#E69F00",  # 1: orange         — typically "prediction" series
    "#56B4E9",  # 2: sky blue       — typically "actual" series
    "#009E73",  # 3: bluish green   — secondary comparison
    "#F0E442",  # 4: yellow         — tertiary (low-contrast; use sparingly)
    "#0072B2",  # 5: blue           — reference markers (e.g. ACF lag lines)
    "#D55E00",  # 6: vermillion     — highlight / emphasis
    "#CC79A7",  # 7: reddish purple — quaternary
)

#: Perceptually uniform sequential colormap (CVD-safe).  Used for any
#: "more-is-more" intensity map (e.g. a per-hour absolute-error heatmap).
SEQUENTIAL_CMAP: str = "cividis"

#: Diverging colormap centred at zero — used for the signed-residual
#: hour x weekday heatmap (:func:`error_heatmap_hour_weekday`).  Reversed
#: so "red = under-forecast" (actual > predicted) and "blue =
#: over-forecast" (predicted > actual), matching the convention in the
#: Stage 4 notebook.
DIVERGING_CMAP: str = "RdBu_r"


# ---------------------------------------------------------------------------
# rcParams injection — runs once at import
# ---------------------------------------------------------------------------

#: rcParams values written by :func:`_apply_style`.  Kept as a module-level
#: constant so the values are discoverable and testable without running
#: import-time side effects.  Figsize (12.0, 8.0) is the Stage 6 plan D5
#: human-mandated default (2026-04-20); all other values mirror the
#: DESIGN §2.1.2 typography convention (12 pt axis labels, 14 pt titles).
_STYLE_RCPARAMS: dict[str, Any] = {
    "axes.prop_cycle": cycler(color=list(OKABE_ITO)),
    "figure.figsize": (12.0, 8.0),  # D5: projector-friendly default
    "figure.dpi": 110,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def _apply_style() -> None:
    """Write Stage 6 defaults into ``plt.rcParams`` once.

    Called exactly once at module import.  Idempotent — repeated calls
    write the same values.  Downstream code wanting to opt out should call
    :func:`matplotlib.pyplot.rcdefaults` after importing this module, or
    override individual keys via ``plt.rcParams.update(...)``.
    """
    plt.rcParams.update(_STYLE_RCPARAMS)


_apply_style()


# ---------------------------------------------------------------------------
# Shared internal utilities — _ensure_axes, etc.
# ---------------------------------------------------------------------------


def _ensure_axes(
    ax: matplotlib.axes.Axes | None,
    **figkw: Any,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return ``(fig, ax)``, creating a new figure when ``ax is None``.

    The ``ax=`` passthrough pattern (D5 moderate-opinionatedness) lets
    facilitators compose Stage 6 helpers into arbitrary subplot grids
    without the helpers owning figure lifetime.  When ``ax is None`` the
    helper mints a new ``Figure`` sized from ``plt.rcParams["figure.figsize"]``
    (which :func:`_apply_style` has set to the D5 default).

    Parameters
    ----------
    ax:
        Optional existing ``Axes`` to draw on.
    **figkw:
        Forwarded to :func:`matplotlib.pyplot.subplots` when ``ax is None``.
    """
    if ax is None:
        fig, new_ax = plt.subplots(**figkw)
        return fig, new_ax
    return ax.figure, ax  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public helpers (T2: scaffolds only; bodies in T3 and T4)
# ---------------------------------------------------------------------------


def residuals_vs_time(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Residuals over time",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Line plot of forecast residuals over time.

    Renders ``residuals`` as a single-line time-series plot on a local-time
    x-axis (``display_tz`` default ``"Europe/London"``).  A thin horizontal
    zero line highlights where the model is unbiased.

    The body of this helper is implemented in T3; T2 ships only the
    signature and docstring scaffold.

    Parameters
    ----------
    residuals:
        Signed residuals (``actual - predicted``) indexed by a tz-aware
        ``DatetimeIndex``.
    display_tz:
        IANA timezone for the x-axis (default: ``"Europe/London"``).
    title:
        Figure title (British English).
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the plot.
    """
    raise NotImplementedError("residuals_vs_time body is implemented in Stage 6 T3")


def predicted_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    title: str = "Predicted vs actual",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of predicted against actual, with a 45-degree reference.

    Implementation scaffolded in T2; body in T3.
    """
    raise NotImplementedError("predicted_vs_actual body is implemented in Stage 6 T3")


def acf_residuals(
    residuals: pd.Series,
    *,
    lags: int = 168,
    alpha: float = 0.05,
    reference_lags: tuple[int, ...] = (24, 168),
    title: str = "Residual autocorrelation (lag in hours)",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """ACF of residuals with daily (lag 24) and weekly (lag 168) markers.

    Renders ``statsmodels.graphics.tsaplots.plot_acf`` with ``lags=168``
    (Stage 6 D7) — enough to show the weekly spike that motivates
    Stage 7's SARIMAX.  Two labelled vertical reference markers at
    ``reference_lags`` (default ``(24, 168)``, i.e. daily + weekly) make
    the periodicity story legible for meetup audiences (D7 reinforcement,
    2026-04-20 human mandate).

    Implementation scaffolded in T2; body in T3.
    """
    raise NotImplementedError("acf_residuals body is implemented in Stage 6 T3")


def error_heatmap_hour_weekday(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Mean signed residual by hour x weekday",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """24 x 7 heatmap of mean signed residual by hour-of-day and weekday.

    Implementation scaffolded in T2; body in T3.
    """
    raise NotImplementedError("error_heatmap_hour_weekday body is implemented in Stage 6 T3")


def forecast_overlay(
    actual: pd.Series,
    predictions_by_name: dict[str, pd.Series],
    *,
    display_tz: str = "Europe/London",
    title: str = "Forecast overlay",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Actual-vs-prediction line plot with one line per named forecast.

    Implementation scaffolded in T2; body in T4.
    """
    raise NotImplementedError("forecast_overlay body is implemented in Stage 6 T4")


def forecast_overlay_with_band(
    actual: pd.Series,
    point_prediction: pd.Series,
    per_fold_errors: pd.DataFrame,
    *,
    quantiles: tuple[float, float] = (0.1, 0.9),
    display_tz: str = "Europe/London",
    title: str = "Forecast with empirical uncertainty band",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Forecast overlay with an empirical-quantile uncertainty band.

    The band derives from ``per_fold_errors`` via the quantile method
    (Stage 6 D8, plan §1); ``per_fold_errors`` is the frame emitted by
    ``evaluate(..., return_predictions=True)`` (D9).

    Implementation scaffolded in T2; body in T4.
    """
    raise NotImplementedError("forecast_overlay_with_band body is implemented in Stage 6 T4")


def benchmark_holdout_bar(
    candidates: dict[str, Model],
    neso_forecast: pd.DataFrame,
    features: pd.DataFrame,
    metrics: Sequence[MetricFn],
    *,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Fixed-window NESO three-way benchmark bar chart (Stage 6 D10).

    Wires up the latent ``NesoBenchmarkConfig.holdout_start/_end`` fields
    added at Stage 4 by slicing ``features`` to the configured window and
    computing one bar per (model, metric) plus a "neso" row.

    Implementation scaffolded in T2; body in T4.
    """
    raise NotImplementedError("benchmark_holdout_bar body is implemented in Stage 6 T4")


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.evaluation.plots``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.evaluation.plots",
        description=(
            "Print the diagnostic-plot helper surface and the active "
            "palette / rcParams defaults.  Every helper in this module "
            "renders colourblind-safe (Okabe-Ito) diagnostics at a "
            "projector-friendly 12x8 figsize by default (Stage 6 D5)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides, e.g. evaluation.plots.figsize=[16,10] "
            "(reserved for parity with peer modules; this CLI currently "
            "prints the surface and does not apply overrides)."
        ),
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance."""
    parser = _build_cli_parser()
    parser.parse_args(list(argv) if argv is not None else None)

    logger.info(
        "Stage 6 plots surface: helpers={} palette={} sequential={} diverging={}",
        tuple(
            name
            for name in __all__
            if name not in {"OKABE_ITO", "SEQUENTIAL_CMAP", "DIVERGING_CMAP"}
        ),
        OKABE_ITO,
        SEQUENTIAL_CMAP,
        DIVERGING_CMAP,
    )
    print("bristol_ml.evaluation.plots — Stage 6 diagnostic-plot helpers")
    print()
    print("Public helpers:")
    for name in (
        "residuals_vs_time",
        "predicted_vs_actual",
        "acf_residuals",
        "error_heatmap_hour_weekday",
        "forecast_overlay",
        "forecast_overlay_with_band",
        "benchmark_holdout_bar",
    ):
        print(f"  - {name}")
    print()
    print("Palette (Okabe-Ito, Wong 2011 Nature Methods 8:441):")
    for idx, colour in enumerate(OKABE_ITO):
        print(f"  [{idx}] {colour}")
    print()
    print(f"Sequential cmap: {SEQUENTIAL_CMAP}")
    print(f"Diverging cmap:  {DIVERGING_CMAP}")
    print()
    print("Active rcParams (applied at import):")
    for key, value in _STYLE_RCPARAMS.items():
        if key == "axes.prop_cycle":
            # Cycler repr is noisy; print the colour list directly.
            print(f"  {key} = cycler(color={list(OKABE_ITO)!r})")
        else:
            print(f"  {key} = {value!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
