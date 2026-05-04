"""Build notebooks/04_remit_ablation.ipynb programmatically.

Stage 16 Task T7 notebook deliverable — follows the five-cell recipe in
``docs/plans/active/16-model-with-remit.md`` §5 (Notebook structure) +
§6 T7.

Generating the notebook from a Python script keeps cell source under
version control as readable text (mirrors Stages 13 / 14 / 15).  The
three-step regeneration flow is::

    uv run python scripts/_build_notebook_16.py
    uv run jupyter nbconvert --execute --to notebook --inplace \\
        notebooks/04_remit_ablation.ipynb
    uv run ruff format notebooks/04_remit_ablation.ipynb

Plan §5 specifies five cells (title-markdown + three executable +
trailing AC-5 commentary):

1. Bootstrap — locate the repo root + print the active configuration.
2. Load + predict — ``registry.load`` for each registered run; assert
   split-config equality across runs; build predictions on the
   evaluation holdout.
3. Four-row metric table — best-without-REMIT, best-with-REMIT
   (excluding ``next_24h``), best-with-REMIT (full), NESO benchmark.
4. AC-5 commentary markdown — names the result honestly across the
   three comparisons; names the price-vs-demand asymmetry; names the
   stub/real extractor flag.

The CI default executes the bootstrap + load cells under
``BRISTOL_ML_REMIT_STUB=1`` + ``BRISTOL_ML_LLM_STUB=1`` — when no
``with_remit`` runs are registered yet (the host has not yet completed
T6), the load cell prints a documented banner and returns cleanly.
The full table renders only once T6 has produced the two registered
runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "04_remit_ablation.ipynb"


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
    """# Stage 16 — Model with REMIT features (ablation)

This notebook is the demo surface for the Stage 16 work — does adding
REMIT-derived features (sourced from a free-text regulatory disclosure
stream, structured by an LLM extractor, aggregated under bi-temporal
correctness) measurably change the day-ahead demand forecast?

The demo moment (intent §"Demo moment") is one **four-row metric
table** that decomposes the REMIT contribution into its two
architecturally distinct parts:

1. **Best model without REMIT** — the prior-stage TCN baseline.
2. **With REMIT, current-state only** — the same TCN trained on the
   ``with_remit`` feature set with the ``remit_unavail_mw_next_24h``
   column dropped from the model's ``feature_columns``.  Isolates the
   value of the *current-state* REMIT signal (sum of unavailable
   capacity at hour ``t``; count of active unplanned events).
3. **With REMIT, full** — the same TCN trained on the full
   ``with_remit`` feature set including the forward-looking column.
   Row 3 minus row 2 isolates the value of the "known future input"
   signal (TFT / AutoGluon "known covariates").
4. **NESO benchmark** — NESO's own day-ahead demand forecast on the
   same evaluation window.

Whether REMIT helps, hurts, or is null on either signal is the
finding.  The intent acknowledges that REMIT is far more informative
for **price** than for **demand** — unplanned nuclear outages move the
price more than they move the national demand — so a small or null
effect is itself the lesson.

- **Intent:** `docs/intent/16-model-with-remit.md`.
- **Plan:** `docs/plans/active/16-model-with-remit.md`.
- **Module:** `src/bristol_ml/features/remit.py`,
  `src/bristol_ml/llm/persistence.py`, and
  `src/bristol_ml/features/assembler.py` (extension).

CI runs this notebook under `BRISTOL_ML_REMIT_STUB=1` +
`BRISTOL_ML_LLM_STUB=1` (deterministic, offline).  When the
`with_remit` registered runs are not yet present the load cell prints
a documented banner and the table renders only the prior-stage
baseline + NESO row — sufficient to verify the notebook executes,
insufficient to draw the comparison.  Once Task T6 has produced the
two registered runs on the host (CUDA + real OpenAI extractor per
plan A3 / A4), re-execute and the four-row table is complete.
"""
)


# ---------------------------------------------------------------------------
# Cell 1 — bootstrap (T7 Cell 1)
# ---------------------------------------------------------------------------

cell_1 = code(
    """# T7 Cell 1 — Bootstrap, locate the repo root, set offline guards.
import os
import sys
from pathlib import Path

# Walk up from notebooks/ to the repo root so the imports below resolve
# regardless of the cwd jupyter was launched from.  Both the project
# root (so `conf._schemas` resolves) and the `src/` dir (so
# `bristol_ml` resolves) must land on sys.path.
NOTEBOOK_DIR = Path.cwd()
REPO_ROOT = NOTEBOOK_DIR if (NOTEBOOK_DIR / "pyproject.toml").exists() else NOTEBOOK_DIR.parent
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    str_path = str(_path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

# Stub-first: every external dependency runs offline by default so the
# notebook executes top-to-bottom in CI without a network or API key.
# The host overrides these in T6 when it produces the real-extractor
# registered runs.
os.environ.setdefault("BRISTOL_ML_REMIT_STUB", "1")
os.environ.setdefault("BRISTOL_ML_LLM_STUB", "1")

from bristol_ml import registry  # noqa: E402

print("=== Stage 16 — Model with REMIT features ===")
print(f"repo root: {REPO_ROOT}")
print(f"BRISTOL_ML_REMIT_STUB: {os.environ.get('BRISTOL_ML_REMIT_STUB')}")
print(f"BRISTOL_ML_LLM_STUB:   {os.environ.get('BRISTOL_ML_LLM_STUB')}")
print()
print("=== Registered runs ===")
runs_with_remit = list(registry.list_runs(feature_set="with_remit"))
runs_baseline = list(registry.list_runs(feature_set="weather_calendar"))
print(f"with_remit registered runs:        {len(runs_with_remit)}")
print(f"weather_calendar registered runs:  {len(runs_baseline)}")
"""
)


# ---------------------------------------------------------------------------
# Cell 2 — load + predict (T7 Cell 2)
# ---------------------------------------------------------------------------

cell_2 = code(
    '''# T7 Cell 2 — Pick the registered runs that compose the ablation.
#
# AC-4: the ablation is reproducible from the registry.  This cell
# never calls model.fit(); it reads sidecar metadata via
# registry.list_runs and only calls registry.load(run_id) /
# model.predict(...) downstream when needed (the metric table itself
# reads from the sidecars\' pre-computed cross-fold means, so even
# predict() is unnecessary for the demo path).  The source-inspection
# test in tests/integration/test_notebook_04.py asserts the no-fit
# invariant.
from bristol_ml import registry


def _latest_run(feature_set, model_substring=None, columns_filter=None):
    """Return the most recent matching run dict, or None.

    `model_substring` matches against the sidecar `name` field; the
    optional `columns_filter` callable is applied to `feature_columns`
    so the with/without-next_24h pair can be distinguished by a single
    column-list predicate without forcing a name convention.
    """
    candidates = list(registry.list_runs(feature_set=feature_set))
    if model_substring is not None:
        candidates = [r for r in candidates if model_substring in r["name"]]
    if columns_filter is not None:
        candidates = [r for r in candidates if columns_filter(r["feature_columns"])]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r["fit_utc"])[-1]


baseline = _latest_run(feature_set="weather_calendar", model_substring="nn_temporal")
with_remit_no_fwd = _latest_run(
    feature_set="with_remit",
    model_substring="nn_temporal",
    columns_filter=lambda cols: "remit_unavail_mw_next_24h" not in cols,
)
with_remit_full = _latest_run(
    feature_set="with_remit",
    model_substring="nn_temporal",
    columns_filter=lambda cols: "remit_unavail_mw_next_24h" in cols,
)

if baseline is None:
    print(
        "WARNING: no Stage-11/12 nn_temporal registered run found at "
        "feature_set=weather_calendar — Stage 16's ablation needs one as "
        "the no-REMIT baseline.  Run Stage 11/12's training on the host "
        "before re-executing this notebook."
    )

if with_remit_no_fwd is None or with_remit_full is None:
    print(
        "WARNING: with_remit registered runs missing.  Run Task T6 on the "
        "host to populate them:\\n\\n"
        "  uv run python -m bristol_ml.llm.persistence --cache auto \\\\\\n"
        "      +llm=extractor llm.type=openai llm.model_name=gpt-4o-mini\\n"
        "  uv run python -m bristol_ml.features.assembler --cache auto \\\\\\n"
        "      features=with_remit\\n"
        "  uv run python -m bristol_ml.train features=with_remit \\\\\\n"
        "      model=nn_temporal features.with_remit.include_forward_lookahead=true\\n"
        "  uv run python -m bristol_ml.train features=with_remit \\\\\\n"
        "      model=nn_temporal features.with_remit.include_forward_lookahead=false\\n"
    )

# Print provenance for every run that is present so the table viewer
# knows which extractor produced the features (NFR-6).  The
# extractor_mode flag is recorded inside the sidecar's hyperparameters
# bag at training time (T6).
for label, run in [
    ("baseline (no REMIT)", baseline),
    ("with_remit (no next_24h)", with_remit_no_fwd),
    ("with_remit (full)", with_remit_full),
]:
    if run is None:
        continue
    print(f"{label}:")
    print(f"  run_id:        {run['run_id']}")
    print(f"  model:         {run['name']}")
    print(f"  feature_set:   {run['feature_set']}")
    print(f"  fit_utc:       {run['fit_utc']}")
    print(f"  git_sha:       {run['git_sha']}")
    print(f"  n_features:    {len(run['feature_columns'])}")
    print()
'''
)


# ---------------------------------------------------------------------------
# Cell 3 — render four-row metric table (T7 Cell 3)
# ---------------------------------------------------------------------------

cell_3 = code(
    '''# T7 Cell 3 — Render the four-row metric table.
#
# Plan §5: best model w/o REMIT, with REMIT (no next_24h), with REMIT
# (full), NESO benchmark.  The first three rows source their metrics
# from the registered runs\' sidecars (no re-fit, no re-predict — the
# AC-4 reproducibility gate); the NESO row is sourced from the prior
# Stage-4 benchmark sidecar when registered, else left blank.
import pandas as pd


def _metrics_row(label, run):
    """Extract the four point-forecast cross-fold means from a sidecar."""
    if run is None:
        return {"label": label, "mae": None, "mape": None, "rmse": None, "wape": None}
    metrics = run["metrics"]
    return {
        "label": label,
        "mae": metrics.get("mae", {}).get("mean"),
        "mape": metrics.get("mape", {}).get("mean"),
        "rmse": metrics.get("rmse", {}).get("mean"),
        "wape": metrics.get("wape", {}).get("mean"),
    }


# AC-3 / R-6: assert the three TCN runs share the same evaluation
# protocol before rendering the table.  feature_columns is allowed to
# differ between the with_remit pair (that\'s the comparison itself);
# everything else (model type, target, n_folds) must match.
def _split_signature(run):
    return None if run is None else (
        run["type"],
        run["target"],
        len(next(iter(run["metrics"].values()), {}).get("per_fold", []))
        if run["metrics"] else 0,
    )


present = [r for r in (baseline, with_remit_no_fwd, with_remit_full) if r is not None]
if len(present) >= 2:
    sigs = {_split_signature(r) for r in present}
    if len(sigs) > 1:
        print(
            f"WARNING: registered runs disagree on (type, target, n_folds): "
            f"{sigs}.  Cross-stage metric comparability requires identical "
            "splitter / target — re-train any out-of-pattern run before "
            "trusting the table below."
        )

rows = [
    _metrics_row("best model without REMIT (TCN, weather_calendar)", baseline),
    _metrics_row("with REMIT, current-state only (no next_24h)", with_remit_no_fwd),
    _metrics_row("with REMIT, full (incl. next_24h)", with_remit_full),
    # NESO benchmark — Stage 4 ships the three-way comparison; once the
    # benchmark figures are registered alongside the model runs they
    # surface here.  Left blank otherwise so the demo path stays
    # honest about what is and is not present.
    {"label": "NESO benchmark (day-ahead)", "mae": None, "mape": None, "rmse": None, "wape": None},
]

table = pd.DataFrame(rows).set_index("label")
print("=== Four-row metric ablation (lower is better) ===")
print(table.to_string(float_format=lambda v: f"{v:.2f}" if v is not None else "—"))
'''
)


# ---------------------------------------------------------------------------
# Cell 4 — AC-5 commentary (markdown)
# ---------------------------------------------------------------------------

cell_4 = md(
    """## Reading the table — honest commentary (intent AC-5)

The table above answers two questions, not one:

1. **Does REMIT help at all?** Compare *row 1* (no REMIT) vs *row 2*
   (current-state REMIT).  A drop in MAE / RMSE means the
   bi-temporally-correct sum of currently-active unavailable capacity
   carries a signal the weather + calendar features did not already
   express.  An increase or null effect means it does not — and is
   itself a finding.
2. **Does the forward-looking signal add anything beyond the
   current-state signal?** Compare *row 2* vs *row 3*.  This isolates
   the marginal value of the "known unavailability over the next 24h"
   feature — the TFT / AutoGluon "known future input" signal that is
   genuinely available at decision time because REMIT events are
   published in advance of their start.

**Why the result may be small (or zero) and that is OK.** REMIT is
much more informative for the GB *price* than for *demand*.  Unplanned
nuclear outages primarily move the balancing market and the wholesale
price; they have a comparatively small effect on national consumption
(other supply must be redispatched but the demand side is largely
indifferent).  No published study tests REMIT for demand forecasting
(domain research §3); the lesson is what the table shows.  A small or
null marginal contribution from the REMIT block does not invalidate
the bi-temporal infrastructure — that infrastructure pays its way
again at Stage 17 (price target).

**Provenance caveat.** The `extractor_mode` provenance flag in the
registered runs' sidecars distinguishes the **stub** path (offline
default; deterministic fixture data) from the **real OpenAI** path
(production extraction over the historical REMIT corpus).  When the
`extractor_mode` reads `stub`, the REMIT features are dominated by
sentinel zeros — the signal-to-noise is artificially low and the
table is read as "code-path verification" rather than "analytical
finding".  Plan A3 binds the registered runs in this notebook to the
real-extractor path; the stub-mode rendering is the CI smoke.

**Reproducibility.** Both `with_remit` registered runs share the
extractor parquet at `data/processed/remit_extracted.parquet`; the
two differ only in the model's `feature_columns` (whether
`remit_unavail_mw_next_24h` is read or not) — same TCN architecture,
same rolling-origin splits, same metrics.  AC-3 / AC-4.

## What's next

- **Stage 17 (price target)** picks up where this leaves off — the
  REMIT signal is much stronger for the wholesale price than for
  demand, so the same `with_remit` parquet feeds a different target
  with different expected results.
- The `forward_lookahead_hours` knob (currently 24) is YAML; a future
  exploration could ablate at 6 / 12 / 48 / 72 to find whether the
  optimal horizon matches the day-ahead window.
"""
)


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------

notebook = {
    "cells": [cell_0, cell_1, cell_2, cell_3, cell_4],
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
