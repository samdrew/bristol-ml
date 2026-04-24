"""Stage 6 diagnostic-plot helper library for the evaluation layer.

This module is the **Stage 6 diagnostic surface**: a small, opinionated set
of helper functions that render residual diagnostics, forecast overlays,
and a fixed-window NESO benchmark bar chart.  Every future modelling stage
(Stage 7 SARIMAX, Stage 8 tree-based, Stage 10 NN, Stage 11 ensembling,
Stage 16 weather-regime) drops its residuals into these helpers without
needing to know the caller; the helpers are model-agnostic (AC-3) — they
take ``pd.Series`` / ``pd.DataFrame`` inputs, never a ``Model`` object.

The public surface is eight functions plus three palette constants:

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
- :func:`loss_curve` — Stage 10 D6 train + validation loss curves from a
  neural-network ``loss_history_`` sequence; the demo-moment plot for
  the simple NN notebook.

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
after the import.  To instead propagate ``evaluation.plots.figsize`` /
``dpi`` from a loaded Hydra config, call :func:`apply_plots_config` with
the ``PlotsConfig`` instance (D5 Hydra-configurable knob).

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
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from loguru import logger
from statsmodels.graphics.tsaplots import plot_acf

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from bristol_ml.evaluation.metrics import MetricFn
    from bristol_ml.models.protocol import Model
    from conf._schemas import PlotsConfig


__all__ = [
    "DIVERGING_CMAP",
    "OKABE_ITO",
    "SEQUENTIAL_CMAP",
    "acf_residuals",
    "apply_plots_config",
    "benchmark_holdout_bar",
    "error_heatmap_hour_weekday",
    "forecast_overlay",
    "forecast_overlay_with_band",
    "loss_curve",
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


def _apply_style(config: PlotsConfig | None = None) -> None:
    """Write Stage 6 defaults into ``plt.rcParams``.

    Called once at module import with ``config=None`` so the hard-coded
    module defaults (``_STYLE_RCPARAMS``) land in ``plt.rcParams`` without
    introducing an import-time dependency on ``conf._schemas``.

    When ``config`` is provided, ``figure.figsize`` and ``figure.dpi`` are
    overridden from ``config.figsize`` / ``config.dpi``; this is how the
    Stage 6 D5 Hydra knobs (``evaluation.plots.figsize``,
    ``evaluation.plots.dpi``) actually reach matplotlib at runtime.  See
    :func:`apply_plots_config` for the public entry point.

    Idempotent — repeated calls write the same values for the same input.
    Downstream code wanting to opt out entirely should call
    :func:`matplotlib.pyplot.rcdefaults` after importing this module, or
    override individual keys via ``plt.rcParams.update(...)``.
    """
    params = dict(_STYLE_RCPARAMS)
    if config is not None:
        params["figure.figsize"] = tuple(config.figsize)
        params["figure.dpi"] = int(config.dpi)
    plt.rcParams.update(params)


_apply_style()


def apply_plots_config(config: PlotsConfig) -> None:
    """Re-apply the Stage 6 rcParams overlay with values from a ``PlotsConfig``.

    Call this from a notebook or CLI after loading a Hydra config if you
    want overrides of ``evaluation.plots.figsize`` or
    ``evaluation.plots.dpi`` to take effect.  Without this call, the
    module-default values written at import time remain in force and the
    YAML knobs are decorative (Stage 6 D5 spec-drift caught in Phase 3
    review N2).

    Example
    -------
    >>> from bristol_ml.config import load_config
    >>> from bristol_ml.evaluation.plots import apply_plots_config
    >>> cfg = load_config()
    >>> apply_plots_config(cfg.evaluation.plots)  # doctest: +SKIP

    The palette (``OKABE_ITO`` qualitative, ``SEQUENTIAL_CMAP`` sequential,
    ``DIVERGING_CMAP`` diverging) is *not* overridable here — Stage 6 D2
    pins it to preserve the colourblind-safety guarantee.
    """
    _apply_style(config)


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


#: Minimum series span (days) above which the x-axis switches from "day + month"
#: to "month + year" tick-format.  Chosen so a 24-hour overlay (Stage 4 notebook
#: Cell 11 convention) uses "%d %b" and a 3-year test horizon uses "%b %Y".
_LONG_SPAN_THRESHOLD_DAYS: int = 180


def _require_tz_aware_datetime_index(
    series_or_index: pd.Series | pd.DatetimeIndex,
    *,
    name: str,
) -> pd.DatetimeIndex:
    """Return the DatetimeIndex of ``series_or_index`` or raise ``ValueError``.

    Load-bearing for the D6 ``display_tz`` contract: every helper that maps a
    timestamp to local wall-clock time (``residuals_vs_time``,
    ``error_heatmap_hour_weekday``, ``forecast_overlay``) requires a tz-aware
    index.  A tz-naive index is ambiguous across DST transitions and is
    rejected at the boundary to keep downstream code honest.
    """
    idx = series_or_index.index if isinstance(series_or_index, pd.Series) else series_or_index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{name}.index must be a DatetimeIndex; got {type(idx).__name__!r}.")
    if idx.tz is None:
        raise ValueError(
            f"{name}.index must be tz-aware so display_tz conversion is unambiguous; "
            f"got a tz-naive DatetimeIndex.  Attach tz=UTC via "
            f'``series.tz_localize("UTC")`` or pass UTC-aware timestamps upstream.'
        )
    return idx


def _apply_time_axis_formatter(ax: matplotlib.axes.Axes, local_index: pd.DatetimeIndex) -> None:
    """Set an mdates locator + formatter appropriate to the series span.

    Series >= 180 days: ``MonthLocator(interval=2)`` + ``%b %Y``.  Shorter:
    ``%d %b`` (with a line break + ``%H:%M`` for the sub-day overlay; plan T4).
    """
    if len(local_index) == 0:
        return
    span_days = (local_index.max() - local_index.min()).total_seconds() / 86400.0
    if span_days >= _LONG_SPAN_THRESHOLD_DAYS:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def residuals_vs_time(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Residuals over time",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Line plot of forecast residuals over time.

    Renders ``residuals`` as a single-line time-series plot on a local-time
    x-axis (``display_tz`` default ``"Europe/London"``, Stage 6 D6 with DST
    verification gate).  A thin horizontal zero line highlights where the
    model is unbiased.

    Parameters
    ----------
    residuals:
        Signed residuals (``actual - predicted``) indexed by a tz-aware
        ``DatetimeIndex``.  A tz-naive index raises ``ValueError``.
    display_tz:
        IANA timezone for the x-axis (default: ``"Europe/London"``).
    title:
        Figure title (British English).
    ax:
        Optional existing axes to draw on.  When ``None`` a new figure is
        created at the ``plt.rcParams["figure.figsize"]`` default.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the plot.

    Raises
    ------
    ValueError
        If ``residuals.index`` is not a tz-aware ``DatetimeIndex``.
    """
    idx = _require_tz_aware_datetime_index(residuals, name="residuals")
    local_idx = idx.tz_convert(display_tz)
    fig, axes = _ensure_axes(ax)

    axes.plot(local_idx, residuals.values, linewidth=1.2, color=OKABE_ITO[0])
    axes.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    _apply_time_axis_formatter(axes, local_idx)
    axes.set_xlabel(f"Time ({display_tz})")
    axes.set_ylabel("Residual (MW)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


#: Maximum scatter point count before the helper down-samples.  20 000 keeps
#: rendering responsive on a laptop Jupyter kernel; above this the visual
#: effect of overplotting saturates anyway.
_SCATTER_SAMPLE_CAP: int = 20_000

#: Deterministic RNG seed for the scatter down-sample so two consecutive
#: calls produce byte-identical figures (T3 ``test_helpers_deterministic_on_fixed_seed``).
_SCATTER_RNG_SEED: int = 42


def predicted_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    title: str = "Predicted vs actual",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of predicted against actual, with a 45-degree reference line.

    Axis convention: x = predicted, y = actual (Gelman 2025, research §R3).
    A 45-degree reference line spans the combined min/max so points on the
    line are perfect predictions.

    Parameters
    ----------
    y_true:
        Ground-truth values (plotted on the y-axis).
    y_pred:
        Model predictions (plotted on the x-axis).
    title:
        Figure title (British English).
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the plot.

    Notes
    -----
    When the series exceed ``_SCATTER_SAMPLE_CAP`` (20 000) rows the scatter
    is down-sampled via a seeded ``numpy.random.default_rng(42)`` so two
    consecutive calls produce byte-identical figures.  A ``logger.info``
    records the cap hit.
    """
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must share the same shape; "
            f"got {true_arr.shape!r} vs {pred_arr.shape!r}."
        )

    n = true_arr.size
    if n > _SCATTER_SAMPLE_CAP:
        rng = np.random.default_rng(_SCATTER_RNG_SEED)
        keep = rng.choice(n, size=_SCATTER_SAMPLE_CAP, replace=False)
        keep.sort()
        true_arr = true_arr[keep]
        pred_arr = pred_arr[keep]
        logger.info(
            "predicted_vs_actual: down-sampled {} -> {} rows (cap={}, seed={})",
            n,
            _SCATTER_SAMPLE_CAP,
            _SCATTER_SAMPLE_CAP,
            _SCATTER_RNG_SEED,
        )

    fig, axes = _ensure_axes(ax)
    axes.scatter(pred_arr, true_arr, alpha=0.15, s=4, color=OKABE_ITO[1])

    lo = float(min(true_arr.min(), pred_arr.min()))
    hi = float(max(true_arr.max(), pred_arr.max()))
    axes.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.6)

    axes.set_xlabel("Predicted demand (MW)")
    axes.set_ylabel("Actual demand (MW)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    return fig


#: Label text for canonical ACF reference-lag markers.  Keyed by lag in hours
#: so downstream tests can assert the exact label strings without touching the
#: helper internals.  Unknown lags fall back to ``"lag (N)"``.
_ACF_MARKER_LABELS: dict[int, str] = {
    24: "daily (24)",
    168: "weekly (168)",
    336: "fortnightly (336)",
}


def _annotate_acf_markers(
    ax: matplotlib.axes.Axes,
    reference_lags: Sequence[int],
) -> None:
    """Draw labelled vertical reference markers on an ACF axes.

    Each entry in ``reference_lags`` becomes one :meth:`matplotlib.axes.Axes.axvline`
    at ``x=lag`` plus one :meth:`matplotlib.axes.Axes.text` label placed
    upper-inside the line at ``y=0.9 * ax.get_ylim()[1]``.  Markers use
    ``OKABE_ITO[5]`` (blue) so they sit clearly against the stem-plot black.

    Plan D7 reinforcement (2026-04-20 human mandate): default two markers at
    lag 24 (daily) and 168 (weekly).  ``reference_lags=()`` disables.
    """
    if not reference_lags:
        return
    y_top = ax.get_ylim()[1]
    label_y = 0.9 * y_top
    for lag in reference_lags:
        ax.axvline(x=lag, linewidth=1.0, alpha=0.35, color=OKABE_ITO[5])
        label = _ACF_MARKER_LABELS.get(int(lag), f"lag ({int(lag)})")
        ax.text(
            x=lag,
            y=label_y,
            s=label,
            rotation=90,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            color=OKABE_ITO[5],
        )


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

    Wraps :func:`statsmodels.graphics.tsaplots.plot_acf` with ``lags=168``
    (Stage 6 D7) — enough to show the weekly spike that motivates Stage 7's
    SARIMAX.  Two labelled vertical reference markers at ``reference_lags``
    (default ``(24, 168)`` — daily + weekly) make the periodicity story
    legible for meetup audiences (D7 reinforcement, 2026-04-20 human
    mandate).

    Parameters
    ----------
    residuals:
        Signed residual series (real-valued; no NaN).
    lags:
        Number of lags to compute; default ``168`` so one full weekly cycle
        is visible.
    alpha:
        Significance level for the confidence band; default ``0.05`` (95%).
    reference_lags:
        Lags (in hours) at which to draw labelled vertical markers.  Default
        ``(24, 168)``.  Pass ``()`` to disable annotation.
    title:
        Figure title.
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the ACF plot.
    """
    fig, axes = _ensure_axes(ax)
    plot_acf(residuals, lags=lags, alpha=alpha, ax=axes)
    axes.set_xlabel("Lag (hours)")
    axes.set_ylabel("Autocorrelation")
    axes.set_title(title)
    _annotate_acf_markers(axes, reference_lags)
    return fig


#: British English weekday abbreviations (Monday=0 convention; matches the
#: Stage 5 calendar-feature ordering).  Load-bearing for the y-axis of
#: :func:`error_heatmap_hour_weekday`.
_WEEKDAY_ABBREV_EN_GB: tuple[str, ...] = (
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
)


def error_heatmap_hour_weekday(
    residuals: pd.Series,
    *,
    display_tz: str = "Europe/London",
    title: str = "Mean signed residual by hour x weekday",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """24 x 7 heatmap of mean signed residual by hour-of-day and weekday.

    Groups ``residuals`` by local-time hour-of-day (columns 0-23) and
    weekday (rows Mon-Sun) and renders the mean signed residual via
    ``seaborn.heatmap`` with the diverging ``RdBu_r`` colormap centred at
    zero.  Red cells are under-forecast (actual > predicted), blue cells
    are over-forecast.

    Parameters
    ----------
    residuals:
        Signed residual series indexed by a tz-aware ``DatetimeIndex``.
    display_tz:
        IANA timezone used for grouping (default: ``"Europe/London"`` —
        Stage 6 D6).  A tz-naive index raises ``ValueError``.
    title:
        Figure title.
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the heatmap.

    Raises
    ------
    ValueError
        If ``residuals.index`` is not a tz-aware ``DatetimeIndex``.
    """
    idx = _require_tz_aware_datetime_index(residuals, name="residuals")
    local_idx = idx.tz_convert(display_tz)

    frame = pd.DataFrame(
        {
            "weekday": local_idx.dayofweek,
            "hour": local_idx.hour,
            "residual": np.asarray(residuals.values, dtype=np.float64),
        }
    )
    pivot = (
        frame.groupby(["weekday", "hour"])["residual"]
        .mean()
        .unstack("hour")
        .reindex(index=range(7), columns=range(24))
    )

    fig, axes = _ensure_axes(ax)
    sns.heatmap(
        pivot,
        cmap=DIVERGING_CMAP,
        center=0,
        ax=axes,
        cbar_kws={"label": "Mean signed residual (MW)"},
    )
    axes.set_yticks(np.arange(7) + 0.5)
    axes.set_yticklabels(_WEEKDAY_ABBREV_EN_GB, rotation=0)
    axes.set_xlabel("Hour of day (local)")
    axes.set_ylabel("Weekday")
    axes.set_title(title)
    return fig


def forecast_overlay(
    actual: pd.Series,
    predictions_by_name: dict[str, pd.Series],
    *,
    display_tz: str = "Europe/London",
    title: str = "Forecast overlay",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Actual-vs-prediction line plot with one line per named forecast.

    Renders the ``actual`` series as a solid line in ``OKABE_ITO[2]`` (sky
    blue) plus one line per entry in ``predictions_by_name``, each taking
    successive Okabe-Ito colours skipping both ``OKABE_ITO[0]`` (black —
    reserved for axes / reference lines) and ``OKABE_ITO[2]`` (sky blue —
    used for the Actual series; reused would silently collide).  Legend is
    placed lower-right to match the Stage 4 notebook Cell 11 convention.

    Parameters
    ----------
    actual:
        Observed series indexed by a tz-aware ``DatetimeIndex``.
    predictions_by_name:
        Ordered mapping of ``label -> predicted series``.  Each prediction
        must share ``actual``'s index (or a subset — unaligned rows are
        dropped via positional intersection).
    display_tz:
        IANA timezone for the x-axis.
    title:
        Figure title.
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the overlay plot.
    """
    idx = _require_tz_aware_datetime_index(actual, name="actual")
    local_idx = idx.tz_convert(display_tz)
    fig, axes = _ensure_axes(ax)

    axes.plot(local_idx, actual.values, linewidth=1.6, color=OKABE_ITO[2], label="Actual")
    # Skip OKABE_ITO[0] (black, reserved for axes/reference lines) AND OKABE_ITO[2]
    # (sky blue — the Actual colour above; reused would silently produce two
    # indistinguishable lines).  Slice carefully: OKABE_ITO[1:] would include
    # index 2 at position 1, re-introducing the collision for the 2nd prediction.
    palette = OKABE_ITO[1:2] + OKABE_ITO[3:]
    for offset, (label, pred) in enumerate(predictions_by_name.items()):
        colour = palette[offset % len(palette)]
        # Align the prediction to the actual's index positionally — the helper
        # does not own alignment semantics, it just plots what it is given.
        pred_idx = _require_tz_aware_datetime_index(pred, name=f"predictions_by_name[{label!r}]")
        axes.plot(
            pred_idx.tz_convert(display_tz),
            pred.values,
            linewidth=1.4,
            color=colour,
            label=label,
        )

    axes.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axes.set_xlabel(f"Time ({display_tz})")
    axes.set_ylabel("Demand (MW)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(loc="lower right")
    return fig


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

    Computes per-horizon quantiles of ``per_fold_errors["error"]`` (the
    signed ``y_true - y_pred`` residuals emitted by
    ``evaluate(..., return_predictions=True)`` — Stage 6 D9) and shades the
    band ``[point - q_hi, point - q_lo]`` around ``point_prediction``.  The
    band is non-parametric and model-agnostic (plan D8).

    Parameters
    ----------
    actual:
        Observed series indexed by a tz-aware ``DatetimeIndex``.
    point_prediction:
        Single-model forecast series (same index as ``actual``).
    per_fold_errors:
        Long-form error frame with columns ``horizon_h`` (int) and
        ``error`` (float).  Raises ``ValueError`` if ``horizon_h`` is
        absent (plan T4 error-handling contract).
    quantiles:
        ``(lo, hi)`` quantile pair used to build the band; default
        ``(0.1, 0.9)`` → 80% empirical interval.
    display_tz:
        IANA timezone for the x-axis.
    title:
        Figure title.
    ax:
        Optional existing axes to draw on.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``per_fold_errors`` lacks a ``horizon_h`` column, or if
        ``quantiles`` is not a strictly ordered ``(lo, hi)`` pair within
        ``[0.0, 1.0]``.
    """
    if "horizon_h" not in per_fold_errors.columns:
        raise ValueError(
            "per_fold_errors must have 'horizon_h' column — "
            "run evaluate(..., return_predictions=True)"
        )
    q_lo_val, q_hi_val = float(quantiles[0]), float(quantiles[1])
    # Reject reversed / equal / out-of-range quantiles at the boundary.  Pandas
    # ``unstack`` preserves the input list order, so positional ``iloc`` access
    # to the resulting DataFrame silently inverts the band when the caller
    # passes e.g. ``quantiles=(0.9, 0.1)``.  Fail fast rather than render a
    # misleading negative-height band.  (Phase 3 review N1.)
    if not (0.0 <= q_lo_val < q_hi_val <= 1.0):
        raise ValueError(
            f"quantiles must be a (lo, hi) pair with 0 <= lo < hi <= 1; got {quantiles!r}."
        )
    band = per_fold_errors.groupby("horizon_h")["error"].quantile([q_lo_val, q_hi_val]).unstack()
    # Label-based column access (not ``iloc[:, 0/1]``) so the band stays
    # correctly oriented regardless of the column order that ``unstack``
    # happens to produce.  Defence-in-depth for the validation above.
    n = min(len(point_prediction), band.shape[0])
    q_lo_series = band[q_lo_val].iloc[:n].to_numpy(dtype=np.float64)
    q_hi_series = band[q_hi_val].iloc[:n].to_numpy(dtype=np.float64)

    idx = _require_tz_aware_datetime_index(actual, name="actual")
    local_idx = idx.tz_convert(display_tz)
    point_arr = np.asarray(point_prediction.values, dtype=np.float64)[:n]
    actual_arr = np.asarray(actual.values, dtype=np.float64)[:n]
    local_idx = local_idx[:n]

    fig, axes = _ensure_axes(ax)
    axes.fill_between(
        local_idx,
        point_arr - q_hi_series,
        point_arr - q_lo_series,
        alpha=0.25,
        color=OKABE_ITO[1],
        label=f"q{int(q_lo_val * 100)}-q{int(q_hi_val * 100)} empirical band",
    )
    axes.plot(local_idx, actual_arr, linewidth=1.6, color=OKABE_ITO[2], label="Actual")
    axes.plot(local_idx, point_arr, linewidth=1.4, color=OKABE_ITO[1], label="Forecast")

    axes.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axes.set_xlabel(f"Time ({display_tz})")
    axes.set_ylabel("Demand (MW)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(loc="lower right")
    return fig


#: Unit suffix per metric name for axis labels on the benchmark bar chart.
#: MAE/RMSE are in MW; MAPE/WAPE are fractions.  Unknown metric names get
#: an empty suffix.
_METRIC_UNIT_LABEL: dict[str, str] = {
    "mae": "(MW)",
    "rmse": "(MW)",
    "mape": "(fraction)",
    "wape": "(fraction)",
}


def benchmark_holdout_bar(
    candidates: Mapping[str, Model],
    neso_forecast: pd.DataFrame,
    features: pd.DataFrame,
    metrics: Sequence[MetricFn],
    *,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Holdout-window benchmark (NESO three-way comparison)",
) -> matplotlib.figure.Figure:
    """Fixed-window NESO three-way benchmark bar chart (Stage 6 D10).

    Wires up the latent ``NesoBenchmarkConfig.holdout_start/_end`` fields
    added at Stage 4 by building a single-fold
    :class:`~conf._schemas.SplitterConfig` that covers ``[holdout_start,
    holdout_end]`` and delegating to
    :func:`bristol_ml.evaluation.benchmarks.compare_on_holdout` for the
    scoring.  The helper then renders one grouped bar per metric, with
    ``len(candidates) + 1`` bars per group (the ``+1`` is the NESO row).

    Parameters
    ----------
    candidates:
        Name → :class:`~bristol_ml.models.protocol.Model` mapping; passed
        straight through to ``compare_on_holdout``.
    neso_forecast:
        Half-hourly NESO forecast archive
        (:func:`bristol_ml.ingestion.neso_forecast.load` output).
    features:
        Hourly feature table (Stage 3 assembler shape; must carry either
        a UTC-aware ``DatetimeIndex`` or a ``timestamp_utc`` column).
    metrics:
        Metric callables (see :mod:`bristol_ml.evaluation.metrics`).
    holdout_start, holdout_end:
        Inclusive bounds of the holdout window (UTC-aware timestamps).
    ax:
        Optional existing axes.
    title:
        Figure title.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the grouped bar chart.

    Raises
    ------
    ValueError
        If ``candidates`` is empty; if the holdout window is empty or
        out of bounds; if ``metrics`` is empty.
    """
    if not candidates:
        raise ValueError("benchmark_holdout_bar: 'candidates' is empty.")
    if not metrics:
        raise ValueError("benchmark_holdout_bar: 'metrics' is empty.")

    # Import lazily to avoid a circular import between plots <-> benchmarks.
    from bristol_ml.evaluation.benchmarks import compare_on_holdout
    from conf._schemas import SplitterConfig

    # Promote the feature frame to a UTC-aware DatetimeIndex if it carries
    # the conventional ``timestamp_utc`` column, so positional arithmetic
    # against holdout_start/_end works uniformly.
    df = features
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp_utc" not in df.columns:
            raise ValueError(
                "benchmark_holdout_bar: 'features' must carry a UTC-aware "
                "DatetimeIndex or a 'timestamp_utc' column."
            )
        df = df.set_index("timestamp_utc")

    start_ts = pd.Timestamp(holdout_start)
    end_ts = pd.Timestamp(holdout_end)
    train_mask = df.index < start_ts
    test_mask = (df.index >= start_ts) & (df.index <= end_ts)
    min_train = int(train_mask.sum())
    test_len = int(test_mask.sum())
    if test_len == 0:
        raise ValueError(
            f"benchmark_holdout_bar: holdout window [{start_ts}, {end_ts}] "
            f"selects zero rows from features.index."
        )
    if min_train == 0:
        raise ValueError(
            f"benchmark_holdout_bar: no rows precede holdout_start={start_ts}; "
            f"a fixed-window splitter needs min_train_periods >= 1."
        )
    splitter_cfg = SplitterConfig(
        min_train_periods=min_train,
        test_len=test_len,
        step=test_len,
        gap=0,
        fixed_window=True,
    )

    table = compare_on_holdout(
        candidates,
        df,
        neso_forecast,
        splitter_cfg,
        metrics,
    )

    # Grouped bar chart: one bar group per metric, one bar per model
    # (including the NESO benchmark row).
    fig, axes = _ensure_axes(ax)
    metric_names = [m.__name__ for m in metrics]
    row_labels = list(table.index)
    n_groups = len(metric_names)
    n_rows = len(row_labels)
    bar_width = 0.8 / max(n_rows, 1)
    x_positions = np.arange(n_groups, dtype=np.float64)

    for offset, row_label in enumerate(row_labels):
        values = table.loc[row_label, metric_names].to_numpy(dtype=np.float64)
        colour = OKABE_ITO[(offset + 1) % len(OKABE_ITO)]
        axes.bar(
            x_positions + offset * bar_width,
            values,
            width=bar_width,
            color=colour,
            label=row_label,
            edgecolor="black",
            linewidth=0.5,
        )

    axes.set_xticks(x_positions + bar_width * (n_rows - 1) / 2.0)
    axes.set_xticklabels(
        [f"{name} {_METRIC_UNIT_LABEL.get(name, '')}".strip() for name in metric_names]
    )
    axes.set_xlabel("Metric")
    axes.set_ylabel("Score")
    axes.set_title(title)
    axes.grid(True, axis="y", alpha=0.3)
    axes.legend(title="Model", loc="best")
    return fig


# ---------------------------------------------------------------------------
# Stage 10 — loss_curve (train + validation loss vs epoch)
# ---------------------------------------------------------------------------


def loss_curve(
    history: Sequence[Mapping[str, float]],
    *,
    title: str = "Training vs validation loss",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Render the train + validation loss curves from a neural-network history.

    Stage 10 D6 — the demo-moment plot for the simple NN.  The plan's
    AC-3 says the loss curve must be "available as a plot without
    additional wiring"; this helper is the one-liner surface that
    ``notebooks/10-simple-nn.ipynb`` and any future Stage 11 / Stage 17
    NN notebook call into.

    ``history`` is the shape :attr:`bristol_ml.models.nn.mlp.NnMlpModel.loss_history_`
    exposes — a list of per-epoch dicts with keys ``{"epoch",
    "train_loss", "val_loss"}``.  The helper is kept strictly
    model-agnostic (AC-3 of Stage 6): no ``NnMlpModel`` import, no
    ``isinstance(model, ...)`` branches.  Pass any sequence of dicts
    that carry the three keys.

    Axis convention:

    - x-axis: epoch index (integer; starts at 1 by convention).
    - y-axis: loss (units are the model's — MSE on normalised target
      for the Stage 10 MLP, so "loss" is the generic label).
    - Train in ``OKABE_ITO[1]`` (orange, the "prediction" colour across
      the Stage 6 helpers so the NN-training palette matches).
    - Validation in ``OKABE_ITO[2]`` (sky blue, the "actual" colour).
    - Legend in upper-right (loss curves shrink toward zero; upper-right
      is the quietest quadrant at the plateau the facilitator points
      to).

    Parameters
    ----------
    history:
        Non-empty sequence of epoch dicts with numeric ``"epoch"``,
        ``"train_loss"``, and ``"val_loss"`` entries.  Empty histories
        are rejected — an unfitted model has no curve to plot.
    title:
        Figure title (British English).  Override for notebook context
        ("Stage 10 NN training — Mean-squared error" etc.).
    ax:
        Optional existing axes to draw on.  When ``None`` a new figure
        is created at the ``plt.rcParams["figure.figsize"]`` default
        (Stage 6 D5).  Mirrors the D5 composability contract shared by
        every other helper in this module.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The matplotlib figure containing the loss-curve plot.

    Raises
    ------
    ValueError
        If ``history`` is empty, or if any dict is missing one of the
        three required keys.  The failure is loud rather than silent —
        a half-populated history typically points at a bug in the
        training loop, not a legitimate plot request.

    Examples
    --------
    >>> from bristol_ml.evaluation.plots import loss_curve
    >>> history = [
    ...     {"epoch": 1, "train_loss": 0.9, "val_loss": 1.0},
    ...     {"epoch": 2, "train_loss": 0.6, "val_loss": 0.8},
    ... ]
    >>> fig = loss_curve(history)  # doctest: +SKIP
    """
    if len(history) == 0:
        raise ValueError(
            "loss_curve: 'history' is empty — no epochs to plot.  An unfitted "
            "model or a model that stopped before completing a single epoch "
            "carries an empty loss_history_; call fit() first."
        )
    required_keys = {"epoch", "train_loss", "val_loss"}
    for i, entry in enumerate(history):
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(
                f"loss_curve: history[{i}] is missing required keys {sorted(missing)!r}; "
                f"expected every epoch dict to carry {sorted(required_keys)!r}."
            )

    epochs = np.asarray([float(e["epoch"]) for e in history], dtype=np.float64)
    train_loss = np.asarray([float(e["train_loss"]) for e in history], dtype=np.float64)
    val_loss = np.asarray([float(e["val_loss"]) for e in history], dtype=np.float64)

    fig, axes = _ensure_axes(ax)
    axes.plot(epochs, train_loss, linewidth=1.6, color=OKABE_ITO[1], label="Train")
    axes.plot(epochs, val_loss, linewidth=1.6, color=OKABE_ITO[2], label="Validation")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper right")
    return fig


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
        "loss_curve",
        "apply_plots_config",
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
