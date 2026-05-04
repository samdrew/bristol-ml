"""Build notebooks/06_enhanced_evaluation.ipynb programmatically.

Stage 6 — enhanced evaluation & visualisation.  The Stage 6 diagnostic
content originally landed as an appendix on the end of
``notebooks/04_linear_baseline.ipynb``, which coupled it to the Stage 4
weather-only OLS and made it awkward to compare the Stage 5 weather +
calendar OLS on the same surface.  This builder produces a standalone
Stage 6 notebook that:

- regenerates the per-fold predictions for **both** OLS models (weather-
  only from Stage 4, weather + calendar from Stage 5);
- shows a 2x2 diagnostic grid for **each** model so the calendar uplift
  is visible in residuals, ACF, and the hour-of-day x weekday error
  heatmap;
- overlays **both** OLS forecasts (plus the naive floor and the NESO
  day-ahead) on the same 48-hour window;
- renders the q10-q90 empirical uncertainty band for **each** OLS model
  as a stacked pair of axes;
- runs ``benchmark_holdout_bar`` with both OLS variants + the naive
  baseline against NESO on the configured holdout window.

The three-step regeneration flow mirrors Stage 14 / 15::

    uv run python scripts/_build_notebook_06.py
    uv run jupyter nbconvert --execute --to notebook --inplace \\
        notebooks/06_enhanced_evaluation.ipynb
    uv run ruff format notebooks/06_enhanced_evaluation.ipynb

The notebook expects both feature-table caches (``weather_only.parquet``
and ``weather_calendar.parquet``) to be warm.  When either is missing
the notebook prints an actionable banner with the assembler CLI command
that regenerates it (see ``docs/architecture/layers/features.md`` and
``README.md`` §"Regenerating feature caches after a code change") and
exits early on that block; the rest of the notebook still executes for
whichever feature set is available.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "06_enhanced_evaluation.ipynb"


_CELL_COUNTER = 0


def _next_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:02d}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id("md"),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell 0 — title + abstract (markdown)
# ---------------------------------------------------------------------------

cell_0 = md(
    """# Stage 6 — Enhanced evaluation & visualisation

The Stage 4 demo proved a metric table can rank models, but a metric
table alone does not show *how* one model loses to another.  Stage 6
adds the diagnostic library
([`bristol_ml.evaluation.plots`](../src/bristol_ml/evaluation/plots.py))
that this notebook drives — five colourblind-safe figures (Okabe-Ito
qualitative palette; `cividis` sequential; `RdBu_r` diverging) plus the
holdout-window benchmark bar chart, applied side-by-side to **both**
OLS variants:

- **Linear OLS — weather only** (Stage 4 / `features=weather_only`).
- **Linear OLS — weather + calendar** (Stage 5 /
  `features=weather_calendar`).

Reading the figures together is the demo moment: the calendar OLS
shrinks the morning-peak residual cluster, sharpens the q10-q90
uncertainty band on weekday afternoons, and narrows the gap to the
NESO day-ahead forecast on the holdout-window benchmark bar.  The
weekly ACF spike at lag-168 survives in *both* — that's the lead-in
for Stage 7's SARIMAX dual-seasonality treatment.

This notebook is a thin renderer — the substantive computation
(rolling-origin folds, residuals, per-fold errors, NESO comparison)
all delegates to `bristol_ml.evaluation.{harness,benchmarks,plots}`.
The plumbing for both feature sets lives behind a single Hydra group
swap (`features=weather_calendar`); see Stage 5's retrospective for
how the swap works.
"""
)


# ---------------------------------------------------------------------------
# Cell 1 — bootstrap (T6 Cell 1)
# ---------------------------------------------------------------------------

cell_1 = code(
    """# T6 Cell 1 — Bootstrap, locate the repo root, load both feature configs.
import os
import sys
from pathlib import Path

NOTEBOOK_DIR = Path.cwd().resolve()
REPO_ROOT = NOTEBOOK_DIR
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    sp = str(_path)
    if sp not in sys.path:
        sys.path.insert(0, sp)
os.chdir(REPO_ROOT)  # so cache_dir interpolations (data/...) resolve

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bristol_ml import load_config  # noqa: E402

# Both feature sets come from the same default config tree; only the
# `features=` group swaps.  Narrow the rolling-origin window so this
# notebook completes in well under a minute on a warm cache; the
# train CLI keeps the wider production defaults.
_HYDRA_OVERRIDES = [
    "evaluation.rolling_origin.min_train_periods=720",
    "evaluation.rolling_origin.test_len=168",
    "evaluation.rolling_origin.step=168",
]
cfg_wonly = load_config(overrides=list(_HYDRA_OVERRIDES))
cfg_wcal = load_config(overrides=["features=weather_calendar", *_HYDRA_OVERRIDES])

assert cfg_wonly.features.weather_only is not None
assert cfg_wcal.features.weather_calendar is not None
print("Stage 6 — enhanced evaluation")
print(f"  rolling_origin: min_train={cfg_wonly.evaluation.rolling_origin.min_train_periods}, "
      f"test_len={cfg_wonly.evaluation.rolling_origin.test_len}, "
      f"step={cfg_wonly.evaluation.rolling_origin.step}")
print(f"  feature sets:   weather_only -> {cfg_wonly.features.weather_only.cache_filename}")
print(f"                  weather_calendar -> {cfg_wcal.features.weather_calendar.cache_filename}")
"""
)


# ---------------------------------------------------------------------------
# Cell 2 — load both feature tables
# ---------------------------------------------------------------------------

cell_2 = code(
    '''# T6 Cell 2 — Load both warm feature-table parquets.
#
# Both caches are pre-populated by running the assembler CLI once per
# feature set; the README's "Regenerating feature caches after a code
# change" section names the exact commands.  When a cache is missing
# the notebook prints an actionable banner and skips that side of the
# comparison rather than crashing — so a half-warm clone still
# executes top-to-bottom.
from bristol_ml.features import assembler


def _load_or_warn(cache_path, loader, *, regen_cmd):
    """Return the loaded frame or None with a documented banner."""
    if cache_path.exists():
        df = loader(cache_path).set_index("timestamp_utc")
        return df
    print(
        f"WARNING: feature-table cache missing at {cache_path}.\\n"
        f"  Regenerate with: {regen_cmd}\\n"
        f"  This notebook will skip the comparison side that needs it."
    )
    return None


wonly_path = (
    cfg_wonly.features.weather_only.cache_dir
    / cfg_wonly.features.weather_only.cache_filename
)
wcal_path = (
    cfg_wcal.features.weather_calendar.cache_dir
    / cfg_wcal.features.weather_calendar.cache_filename
)

_BASE_REGEN = "uv run python -m bristol_ml.features.assembler"
features_wonly = _load_or_warn(
    wonly_path,
    assembler.load,
    regen_cmd=f"{_BASE_REGEN} features=weather_only --cache offline",
)
features_wcal = _load_or_warn(
    wcal_path,
    assembler.load_calendar,
    regen_cmd=f"{_BASE_REGEN} features=weather_calendar --cache offline",
)

if features_wonly is not None:
    print(f"weather_only:     rows={len(features_wonly):,}  cols={features_wonly.shape[1]}")
if features_wcal is not None:
    print(f"weather_calendar: rows={len(features_wcal):,}  cols={features_wcal.shape[1]}")
'''
)


# ---------------------------------------------------------------------------
# Cell 3 — fit both linear OLS models + run the harness
# ---------------------------------------------------------------------------

cell_3 = code(
    """# T6 Cell 3 — Fit both OLS models on the full window AND run rolling-
# origin evaluation against each.  ``return_predictions=True`` emits
# the long-form per-fold-errors frame that ``forecast_overlay_with_band``
# consumes (Stage 6 D9 single-flag concession).
from bristol_ml.evaluation.harness import evaluate
from bristol_ml.evaluation.metrics import METRIC_REGISTRY
from bristol_ml.features.calendar import CALENDAR_VARIABLE_COLUMNS
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from conf._schemas import LinearConfig, NaiveConfig

WEATHER_COLS = tuple(name for name, _ in assembler.WEATHER_VARIABLE_COLUMNS)
CALENDAR_COLS = tuple(name for name, _ in CALENDAR_VARIABLE_COLUMNS)
WEATHER_CALENDAR_COLS = WEATHER_COLS + CALENDAR_COLS

split_cfg = cfg_wonly.evaluation.rolling_origin  # identical across both configs
metric_fns = [METRIC_REGISTRY[name] for name in ("mae", "mape", "rmse", "wape")]

results = {}  # name -> {"per_fold_errors": ..., "linear": ..., "metrics_df": ...}

if features_wonly is not None:
    linear_wonly = LinearModel(LinearConfig(feature_columns=WEATHER_COLS))
    linear_wonly.fit(features_wonly, features_wonly["nd_mw"])
    metrics_wonly, errors_wonly = evaluate(
        LinearModel(LinearConfig(feature_columns=WEATHER_COLS)),
        features_wonly,
        split_cfg,
        metric_fns,
        feature_columns=WEATHER_COLS,
        return_predictions=True,
    )
    results["weather_only"] = {
        "linear": linear_wonly,
        "metrics_df": metrics_wonly,
        "per_fold_errors": errors_wonly,
        "features": features_wonly,
    }
    print(f"weather_only      rolling-origin folds: {len(metrics_wonly)}")

if features_wcal is not None:
    linear_wcal = LinearModel(LinearConfig(feature_columns=WEATHER_CALENDAR_COLS))
    linear_wcal.fit(features_wcal, features_wcal["nd_mw"])
    metrics_wcal, errors_wcal = evaluate(
        LinearModel(LinearConfig(feature_columns=WEATHER_CALENDAR_COLS)),
        features_wcal,
        split_cfg,
        metric_fns,
        feature_columns=WEATHER_CALENDAR_COLS,
        return_predictions=True,
    )
    results["weather_calendar"] = {
        "linear": linear_wcal,
        "metrics_df": metrics_wcal,
        "per_fold_errors": errors_wcal,
        "features": features_wcal,
    }
    print(f"weather_calendar  rolling-origin folds: {len(metrics_wcal)}")

# Side-by-side cross-fold means
if results:
    summary = pd.DataFrame(
        {
            name: r["metrics_df"][["mae", "mape", "rmse", "wape"]].mean()
            for name, r in results.items()
        }
    ).T
    summary.index.name = "feature_set"
    print()
    print("Cross-fold mean metrics (lower is better):")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
"""
)


# ---------------------------------------------------------------------------
# Cell 4 — 2x2 diagnostic grids, one per OLS model
# ---------------------------------------------------------------------------

cell_4 = code(
    """# T6 Cell 4 — 2x2 diagnostic grid for each linear OLS model.
#
# Top-left:    residuals_vs_time          — visible serial structure
#              that the linear fit failed to capture.
# Top-right:   predicted_vs_actual         — calibration (45-degree
#              reference) and tail behaviour.
# Bottom-left: acf_residuals (lags=168)    — the 24h / 168h reference
#              spikes; daily residual periodicity is the lead-in for
#              SARIMAX in Stage 7.
# Bottom-right: error_heatmap_hour_weekday — when the model is biased,
#               by hour of day and day of week.
from bristol_ml.evaluation import plots


def _diagnostic_grid(label, bundle):
    feats = bundle["features"]
    linear = bundle["linear"]
    fitted = linear.results.fittedvalues
    residuals = (feats["nd_mw"] - fitted).rename("residual_mw")
    predictions = linear.predict(feats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots.residuals_vs_time(residuals, ax=axes[0, 0])
    plots.predicted_vs_actual(feats["nd_mw"], predictions, ax=axes[0, 1])
    plots.acf_residuals(residuals, lags=168, ax=axes[1, 0])
    plots.error_heatmap_hour_weekday(residuals, ax=axes[1, 1])
    fig.suptitle(f"Linear OLS diagnostics — {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


for name in ("weather_only", "weather_calendar"):
    if name in results:
        _diagnostic_grid(name, results[name])
"""
)


# ---------------------------------------------------------------------------
# Cell 5 — 48-hour forecast overlay with both OLS models
# ---------------------------------------------------------------------------

cell_5 = code(
    """# T6 Cell 5 — 48-hour forecast overlay: actual + naive + both OLS
# variants + (optionally) NESO day-ahead.  Picks the last 48 hourly
# rows of the smaller of the two feature tables so both forecasts can
# be plotted on a shared axis.
import matplotlib.dates as mdates  # noqa: E402

from bristol_ml.ingestion import neso_forecast as neso_forecast_mod  # noqa: E402

# Pick a 48-hour window common to both feature sets.  Use the calendar
# frame's last 48 rows when present (richer feature set is the
# canonical demo); fall back to the weather-only frame.
def _pick_window(bundle, hours=48):
    return bundle["features"].iloc[-hours:].copy()


canonical = results.get("weather_calendar") or results.get("weather_only")
window = _pick_window(canonical) if canonical is not None else None
naive = NaiveModel(NaiveConfig(strategy="same_hour_last_week"))
if canonical is not None:
    naive.fit(canonical["features"], canonical["features"]["nd_mw"])

predictions_by_name = {}
if window is not None:
    predictions_by_name["Naive (same hour last week)"] = naive.predict(window)
    if "weather_only" in results:
        # The full-window-fit linear is what produces the headline 48h
        # overlay; ``predict`` slices by feature name so the calendar
        # columns are simply ignored when a weather-only model sees a
        # calendar frame.
        wonly_for_window = results["weather_only"]["linear"].predict(
            window[[*WEATHER_COLS, "nd_mw"]]
            if set(WEATHER_COLS).issubset(window.columns)
            else window
        )
        predictions_by_name["Linear (weather only)"] = wonly_for_window
    if "weather_calendar" in results:
        wcal_for_window = results["weather_calendar"]["linear"].predict(window)
        predictions_by_name["Linear (weather + calendar)"] = wcal_for_window

# NESO three-way: only when the forecast cache is warm.
neso_df_local = None
if cfg_wonly.ingestion.neso_forecast is not None:
    forecast_cache = (
        cfg_wonly.ingestion.neso_forecast.cache_dir
        / cfg_wonly.ingestion.neso_forecast.cache_filename
    )
    if forecast_cache.exists():
        neso_df_local = neso_forecast_mod.load(forecast_cache)
        from bristol_ml.evaluation.benchmarks import align_half_hourly_to_hourly

        aligned = align_half_hourly_to_hourly(neso_df_local, aggregation="mean")
        if window is not None:
            window_start, window_end = window.index[0], window.index[-1]
            neso_slice = aligned.loc[window_start:window_end]
            if not neso_slice.empty:
                predictions_by_name["NESO day-ahead"] = neso_slice["demand_forecast_mw"]

if window is not None and predictions_by_name:
    fig = plots.forecast_overlay(
        actual=window["nd_mw"],
        predictions_by_name=predictions_by_name,
        title="48-hour forecast overlay — naive + both OLS variants + NESO",
    )
    plt.show()
else:
    print("No feature-table cache warm — skipping 48-hour overlay.")
"""
)


# ---------------------------------------------------------------------------
# Cell 6 — empirical q10-q90 uncertainty band, one panel per OLS model
# ---------------------------------------------------------------------------

cell_6 = code(
    """# T6 Cell 6 — q10-q90 empirical uncertainty band, one stacked panel
# per OLS model so the calendar uplift on the band width is visible
# at a glance.  The band is non-parametric: per-horizon quantiles of
# the rolling-origin per-fold errors (Stage 6 D8).  After the
# 2026-05-04 sign-inversion fix the band tracks where actual is
# likely to land — when the band is narrow the model is confident at
# that hour-of-day; when wide it is hedging.
present = [name for name in ("weather_only", "weather_calendar") if name in results]
if window is not None and present:
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 5 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, name in zip(axes, present, strict=True):
        bundle = results[name]
        feats = bundle["features"]
        win = _pick_window(bundle)
        plots.forecast_overlay_with_band(
            actual=win["nd_mw"],
            point_prediction=bundle["linear"].predict(win),
            per_fold_errors=bundle["per_fold_errors"],
            title=f"q10-q90 empirical uncertainty band — linear ({name})",
            ax=ax,
        )
    plt.tight_layout()
    plt.show()
else:
    print("No feature-table cache warm — skipping uncertainty band.")
"""
)


# ---------------------------------------------------------------------------
# Cell 7 — benchmark_holdout_bar with both OLS variants vs NESO
# ---------------------------------------------------------------------------

cell_7 = code(
    """# T6 Cell 7 — Holdout-window benchmark bar charts, one per feature set.
#
# ``benchmark_holdout_bar`` slices the feature frame by a single shared
# ``feature_columns`` (the harness's column-set is per-call, not
# per-candidate), so we call it twice — once on the weather-only
# frame, once on the weather + calendar frame — and stack the two
# axes vertically.  The naive baseline appears in both charts so the
# eye reads the calendar uplift directly across the bar groups.
#
# After the 2026-05-04 ``fold_len_hours=168`` default fix, each call
# rolls the holdout in weekly folds — seasonal-naive
# ``same_hour_last_week`` works without manual splitter twiddling.
benchmark_ready = (
    cfg_wonly.evaluation.benchmark is not None
    and neso_df_local is not None
    and present  # at least one feature set warm
)
if benchmark_ready:
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 4.5 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, name in zip(axes, present, strict=True):
        candidates = {
            "naive": NaiveModel(NaiveConfig(strategy="same_hour_last_week")),
            f"linear_{name}": LinearModel(LinearConfig()),
        }
        plots.benchmark_holdout_bar(
            candidates=candidates,
            neso_forecast=neso_df_local,
            features=results[name]["features"],
            metrics=metric_fns,
            holdout_start=cfg_wonly.evaluation.benchmark.holdout_start,
            holdout_end=cfg_wonly.evaluation.benchmark.holdout_end,
            ax=ax,
            title=f"Holdout benchmark — {name}",
        )
    plt.tight_layout()
    plt.show()
else:
    missing = []
    if cfg_wonly.evaluation.benchmark is None:
        missing.append("evaluation.benchmark config")
    if neso_df_local is None:
        missing.append("NESO forecast cache")
    if not present:
        missing.append("any feature-table cache")
    print(
        f"Skipping benchmark_holdout_bar — missing: {', '.join(missing)}.\\n"
        "Populate via `python -m bristol_ml.ingestion.neso_forecast` and "
        "`python -m bristol_ml.features.assembler features=<set>`."
    )
"""
)


# ---------------------------------------------------------------------------
# Cell 8 — closing markdown
# ---------------------------------------------------------------------------

cell_8 = md(
    """## What this lets you say to a meetup audience

The metric table at Stage 4 said "calendar features help"; the figures
above let you point at *where* and *how much*:

- **Residual time series** — the calendar OLS removes the regular morning
  ramp-up bias the weather-only OLS leaves behind.
- **Predicted-vs-actual** — both models track the body of the
  distribution, but the calendar OLS pulls the tails closer to the 45°
  line on weekday peaks (when bank-holiday flags fire).
- **Residual ACF** — the daily lag-24 spike shrinks once hour-of-day is
  encoded; the lag-168 (weekly) spike survives in *both* models, which
  is the lead-in for Stage 7's SARIMAX dual-seasonality treatment.
- **Hour-of-day x weekday error heatmap** — the weather-only OLS shows
  systematic Sunday-evening over-prediction; the calendar OLS flattens it.
- **q10-q90 empirical uncertainty band** — narrower for the calendar
  OLS at weekday afternoons (the regime calendar features inform); the
  band widens at the same rate for both models on shoulder hours where
  neither has more information than the other.
- **Holdout-window benchmark bar** — the calendar OLS row sits between
  the naive floor and the NESO day-ahead, closing the gap that Stage 4
  left open.

The structural lesson: a small, faithful library of diagnostic helpers
(`plots.residuals_vs_time`, `acf_residuals`, `forecast_overlay_with_band`,
`benchmark_holdout_bar`) is enough to turn a metric table into a
narrative.  A future stage extends the same surface — Stage 7 reads
`acf_residuals` to motivate seasonal differencing; Stage 10 reads the
hour-of-day heatmap to motivate the NN's input partitioning.
"""
)


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------

notebook = {
    "cells": [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7, cell_8],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {OUT} ({len(notebook['cells'])} cells)")
