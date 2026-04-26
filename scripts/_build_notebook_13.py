"""Build notebooks/13_remit_ingestion.ipynb programmatically.

Stage 13 Task T5 notebook deliverable — follows the three-cell recipe in
``docs/plans/active/13-remit-ingestion.md`` §5 (Notebook structure) +
§6 T5.

Generating the notebook from a Python script keeps cell source under
version control as readable text and avoids the format-diff noise that
Jupyter's editor cache produces.  The three-step regeneration flow is::

    uv run python scripts/_build_notebook_13.py
    uv run jupyter nbconvert --execute --to notebook --inplace \\
        notebooks/13_remit_ingestion.ipynb
    uv run ruff format notebooks/13_remit_ingestion.ipynb

The generator's cell-source strings are *not* pre-formatted to ruff's
line-wrapping conventions; the final ``ruff format`` step is mandatory
so the committed notebook passes the repo-wide format check.  The
script itself is idempotent.

Budget (plan AC-9): end-to-end under 30 seconds under
``BRISTOL_ML_REMIT_STUB=1`` (the CI default).  The stub fixture has 10
records spanning 2024-01-01 … 2024-07-01; the monthly-aggregate
calculation iterates ~7 ``as_of`` calls and a single matplotlib
stacked-area plot — comfortably inside the budget.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "13_remit_ingestion.ipynb"


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
# Cell 0 — title + abstract + plan links
# ---------------------------------------------------------------------------

cell_0 = md(
    """# Stage 13 — REMIT ingestion

This notebook is the demo surface for the Stage 13 ingester
``bristol_ml.ingestion.remit``.  It answers one question with one
chart: **how much GB generation capacity has been declared
unavailable each month, broken down by fuel type, as known to the
market at the start of that month?**

The demo moment (intent §Demo moment) is the chart at the bottom: a
facilitator can point at a spike and say "that was a nuclear unit
going offline on this date".

The mechanic that makes the answer correct in the face of revisions
and withdrawals is the new public primitive Stage 13 introduces:

```python
remit.as_of(df, t)  # what did the market know at time t?
```

This is a transaction-time filter — it asks "which messages had been
disclosed by ``t``?", not "which events were active at ``t``?".  The
two questions decompose cleanly: the cell below first calls ``as_of``
(transaction-time), then chains a valid-time filter on
``effective_from`` / ``effective_to`` (the event window), then
aggregates ``affected_mw`` by ``fuel_type``.

- **Intent:** `docs/intent/13-remit-ingestion.md`.
- **Plan:** `docs/plans/active/13-remit-ingestion.md`.
- **Module:** `src/bristol_ml/ingestion/remit.py`.

CI runs this notebook against the stub fixture
(``BRISTOL_ML_REMIT_STUB=1``) — 10 hand-crafted records spanning
seven mRIDs across the first half of 2024, deterministic and offline.
A live run against the warm cassette / archive cache produces a far
denser chart but the same code path.
"""
)


# ---------------------------------------------------------------------------
# Cell 1 — bootstrap + load (T5 Cell 1)
# ---------------------------------------------------------------------------

cell_1 = code(
    """# T5 Cell 1 — Bootstrap, fetch (or cache-hit), load.
import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # cache_dir in conf/ingestion/remit.yaml resolves against cwd

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bristol_ml import CachePolicy, load_config  # noqa: E402
from bristol_ml.ingestion import remit  # noqa: E402

cfg = load_config(config_path=REPO_ROOT / "conf")
assert cfg.ingestion.remit is not None, "REMIT ingestion config not resolved"
path = remit.fetch(cfg.ingestion.remit, cache=CachePolicy.AUTO)
df = remit.load(Path(path))

print(f"Cache path: {path}")
print(f"Rows: {len(df):,}  ·  mRIDs: {df['mrid'].nunique():,}")
print(
    f"Publish window: {df['published_at'].min()}  ..  {df['published_at'].max()}"
)
"""
)


# ---------------------------------------------------------------------------
# Cell 2 — Monthly-aggregate computation header
# ---------------------------------------------------------------------------

cell_2 = md(
    """## Monthly aggregate unavailable capacity

For each month-start ``t`` covered by the loaded data:

1. ``as_of(df, t)`` — transaction-time filter: "what did the market
   know at ``t``?".  Drops superseded revisions and withdrawn messages.
2. Valid-time filter: keep rows whose event window covers ``t``
   (``effective_from <= t < effective_to``, treating
   ``effective_to is NaT`` as "still active").
3. Group by ``fuel_type``; sum ``affected_mw``.

The ``as_of`` step is the pedagogical crux — without it, a brittle
"latest revision wins" approach over-counts withdrawn messages and
mis-counts events whose end-time was extended in a later revision.
"""
)


# ---------------------------------------------------------------------------
# Cell 3 — compute monthly_long
# ---------------------------------------------------------------------------

cell_3 = code(
    """# T5 Cell 3 — Compute (month, fuel_type, total_mw) long-form table.
publish_min = df["published_at"].min()
publish_max = df["published_at"].max()
month_starts = pd.date_range(
    publish_min.normalize().replace(day=1),
    (publish_max + pd.Timedelta(days=31)).normalize().replace(day=1),
    freq="MS",
    tz="UTC",
)
print(f"Month-starts to evaluate: {len(month_starts)}  ({month_starts[0]} → {month_starts[-1]})")

records = []
for t in month_starts:
    known = remit.as_of(df, t)
    if known.empty:
        continue
    active_mask = (known["effective_from"] <= t) & (
        known["effective_to"].isna() | (known["effective_to"] > t)
    )
    active = known[active_mask]
    if active.empty:
        continue
    by_fuel = active.groupby("fuel_type", dropna=False)["affected_mw"].sum()
    for fuel, mw in by_fuel.items():
        records.append({"month": t, "fuel_type": fuel or "Unknown", "total_mw": float(mw)})

monthly_long = pd.DataFrame.from_records(records)
print(f"Aggregated rows: {len(monthly_long)}")
monthly_long.head(10)
"""
)


# ---------------------------------------------------------------------------
# Cell 4 — plot header
# ---------------------------------------------------------------------------

cell_4 = md(
    """## Stacked-area chart — unavailable MW by fuel type

The chart is the demo moment.  Stacking by ``fuel_type`` makes the
mix legible at a glance; spikes correspond to genuine market events.
"""
)


# ---------------------------------------------------------------------------
# Cell 5 — render the plot
# ---------------------------------------------------------------------------

cell_5 = code(
    """# T5 Cell 5 — Stacked-area chart of monthly unavailable MW by fuel type.
if monthly_long.empty:
    print("No active events in any sampled month — nothing to plot.")
else:
    pivot = monthly_long.pivot_table(
        index="month",
        columns="fuel_type",
        values="total_mw",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(pivot.index.to_pydatetime(), pivot.T.to_numpy(), labels=pivot.columns)
    ax.set_title("REMIT — declared unavailable capacity by fuel type, month-start as-of")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unavailable capacity (MW)")
    ax.legend(loc="upper left", fontsize="small", ncol=2)
    fig.autofmt_xdate()
    plt.show()
"""
)


# ---------------------------------------------------------------------------
# Cell 6 — close-out / next-stage pointer
# ---------------------------------------------------------------------------

cell_6 = md(
    """## What's next

Stage 14 reads the ``message_description`` free-text field on this
parquet and extracts structured fields with an LLM.  Stage 16 joins
the bi-temporal frame into the modelling feature table — the
``as_of`` mechanic above is exactly what guarantees the join uses
only information available at training time, no leakage.
"""
)


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------

notebook = {
    "cells": [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6],
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
