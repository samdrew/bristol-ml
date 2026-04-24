"""Build notebooks/10-simple-nn.ipynb programmatically.

Stage 10 Task T6 notebook deliverable — the pedagogical surface for
AC-3 (live train-vs-validation loss curve) and for the registry
leaderboard comparison promised in plan §6 Task T6.

Generating the notebook from a Python script keeps cell source under
version control as readable text and avoids the format-diff noise that
Jupyter's editor cache produces.  The three-step regeneration flow is::

    uv run python scripts/_build_notebook_10.py
    uv run jupyter nbconvert --execute --to notebook --inplace \\
        notebooks/10-simple-nn.ipynb
    uv run ruff format notebooks/10-simple-nn.ipynb

The generator's cell-source strings are *not* pre-formatted to ruff's
line-wrapping conventions; the final ``ruff format`` step is mandatory.
The script itself is idempotent.

Scope discipline: the scope diff cut NFR-4 (auto-save PNG) and
X1-adjacent over-growth.  The notebook is deliberately short — the
live-loss-curve demo moment (Cell 5) is the load-bearing AC-3
evidence; the rolling-origin evaluation against a registered linear
baseline (Cell 8) is the comparison half of intent §Scope.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "10-simple-nn.ipynb"


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
# Cell 0 — Title + framing + plan links
# ---------------------------------------------------------------------------

cell_0 = md(
    """# Stage 10 — Simple neural network

A small MLP (one hidden layer, 128 units, ReLU) fitted with a hand-rolled
PyTorch training loop, wrapped in the Stage 4 `Model` protocol and
registered through the Stage 9 registry.

The pedagogical payoff is the **live train-vs-validation loss curve**
(Cell 5): the facilitator points at the epoch where validation loss
bottoms out and starts rising, and the audience sees overfitting happen
in real time.  Intent §Purpose is clear that Stage 10's *analytical*
contribution is modest — the linear baseline (Stage 4) + SARIMAX
(Stage 7) already beat a single MLP on the weather-only feature set;
the stage's load-bearing contribution is the **scaffold** (training-loop
conventions, reproducibility discipline, registry round-trip) that
Stage 11's temporal architecture inherits.

- **Intent:** `docs/intent/10-simple-nn.md`.
- **Plan:** `docs/plans/active/10-simple-nn.md` (moved to `completed/`
  at T7).
- **Previous stage:** Stage 9 model registry — the four-verb `save` /
  `load` / `list_runs` / `describe` surface that this notebook queries
  at Cell 8 for the cross-model comparison.

Plan decisions applied here: D3 (default architecture —
`hidden_sizes=[128]`, `activation=relu`, `lr=1e-3`, `max_epochs=100`,
`patience=10`), D4 (z-score normalisation stored in `register_buffer`
so it round-trips through `state_dict`), D6 (the `loss_history_`
attribute + `plots.loss_curve` helper + the `epoch_callback` seam for
the live-plot moment), D7' (four-stream seeding for CPU bit-identity),
D8 (cold-start per fold), D9 (internal 10 % val tail + best-epoch
restore), D11 (auto-device selector — this notebook pins `device="cpu"`
so the run is deterministic and laptop-friendly).
"""
)


# ---------------------------------------------------------------------------
# Cell 1 — Bootstrap + imports (follows the Stage 7 / 8 pattern)
# ---------------------------------------------------------------------------

cell_1 = code(
    """# T6 Cell 1 — Bootstrap: walk up to the repo root so
# ``from bristol_ml import ...`` and Hydra's ``conf/`` both resolve,
# then import the Stage 4 / Stage 6 / Stage 9 surfaces we need.

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from IPython.display import clear_output, display  # noqa: E402

from bristol_ml import load_config  # noqa: E402
from bristol_ml.evaluation import plots  # noqa: E402
from bristol_ml.evaluation.harness import evaluate  # noqa: E402
from bristol_ml.evaluation.metrics import METRIC_REGISTRY  # noqa: E402
from bristol_ml.features import assembler  # noqa: E402
from bristol_ml.models.linear import LinearModel  # noqa: E402
from bristol_ml.models.naive import NaiveModel  # noqa: E402
from bristol_ml.models.nn.mlp import NnMlpModel  # noqa: E402
from conf._schemas import LinearConfig, NaiveConfig, NnMlpConfig  # noqa: E402

# Apply the Okabe-Ito palette + figsize defaults (Stage 6 D2 / D5).
plots.apply_plots_config(
    load_config(
        config_path=REPO_ROOT / "conf",
        overrides=["model=nn_mlp"],
    ).evaluation.plots
)

# Notebook-scope torch device: pinned to CPU so the live demo is
# deterministic and laptop-friendly (plan AC-5).  The CUDA / MPS paths
# are exercised by the ``@pytest.mark.gpu`` tests and by the
# ``_select_device`` helper used from a fresh Python session — see
# ``docs/architecture/layers/models-nn.md``.
print(f"torch version: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print("Notebook pins device='cpu' so two runs of this notebook yield identical weights.")
"""
)


# ---------------------------------------------------------------------------
# Cell 2 — Data
# ---------------------------------------------------------------------------

cell_2 = md(
    """## Data — 60 days of hourly GB demand + weather

We load the Stage 3 `weather_only` feature cache and keep the last
60 days (1440 rows).  That window is small enough that a 1-hidden-layer
MLP fits in a handful of seconds on CI CPU — the point of the cell is
to surface the **shape** of the training signal (hourly, weather-only,
target `nd_mw`), not to push for a leaderboard win.

The `features.assembler.load` helper validates the parquet schema on
read (it rejects missing or extra columns), so if the cache is missing
or stale the cell fails loudly with a schema-mismatch error.  Rebuild
with `python -m bristol_ml.features.assembler` if that happens.
"""
)

cell_3 = code(
    """# T6 Cell 3 — Load the Stage 3 weather_only feature cache and keep
# the last 60 days.  A short window keeps the live-demo fit snappy
# (~5-10 s on a 4-core laptop) and makes the "validation loss bottoms
# out" moment land early enough that the audience is still watching.

cfg = load_config(
    config_path=REPO_ROOT / "conf",
    overrides=["model=nn_mlp"],
)
assert cfg.features.weather_only is not None

features_path = (
    cfg.features.weather_only.cache_dir / cfg.features.weather_only.cache_filename
)
df = assembler.load(features_path).set_index("timestamp_utc")

# Last ~60 days (1440 rows) — see the cell docstring in Cell 2.
df = df.iloc[-24 * 60 :].copy()
target = df["nd_mw"].astype("float64")
feature_cols = [c for c, _ in assembler.WEATHER_VARIABLE_COLUMNS]
features = df[feature_cols].astype("float64")
print(f"Feature table: {features.shape} ({features.index.min()} -> {features.index.max()})")
print(f"Target mean: {target.mean():,.0f} MW; std: {target.std():,.0f} MW")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
"""
)


# ---------------------------------------------------------------------------
# Cell 4 — Live-demo section header
# ---------------------------------------------------------------------------

cell_4 = md(
    """## The demo moment — live train-vs-validation loss curve

`NnMlpModel.fit(..., epoch_callback=on_epoch)` invokes `on_epoch` after
every epoch with the per-epoch `{"epoch", "train_loss", "val_loss"}`
dict (plan D6).  The notebook owns the live-plot logic — the models
layer never imports `IPython` — so the seam is a one-function callback
that appends to a list and redraws the figure.

What to watch for:

- **Train loss** falls monotonically (orange) — the optimiser is doing
  its job on the training slice.
- **Validation loss** (sky blue) falls, bottoms out, and starts rising.
  The epoch of the minimum is the point where the model stops learning
  generalisable structure and starts memorising training noise.
- **Early stopping** (plan D9) waits `patience=10` epochs past the
  best-seen validation loss, then stops and restores the best-epoch
  weights.  The saved artefact is therefore the *best* epoch's weights,
  not the last epoch's.

With the defaults below (`max_epochs=40`, `patience=5`) the U-shape is
typically visible by epoch 10-15 on the 60-day window.  A laptop-only
audience can crank `hidden_sizes=[256, 128]` to see the U-shape shift
right (more capacity → later overfit).
"""
)


# ---------------------------------------------------------------------------
# Cell 5 — Live training with epoch_callback
# ---------------------------------------------------------------------------

cell_5 = code(
    """# T6 Cell 5 — Live train-vs-validation loss curve.  AC-3 evidence.
#
# Notebook-tuned architecture: 40 epochs / patience=5 / device=cpu so
# the run is deterministic and finishes in ~5-10 s on a 4-core laptop.
# The live plot redraws every epoch via ``clear_output(wait=True)`` +
# ``display_id`` — no coupling between the models layer and matplotlib.

nn_cfg = NnMlpConfig(
    target_column="nd_mw",
    hidden_sizes=[128],
    activation="relu",
    dropout=0.0,
    learning_rate=1e-3,
    weight_decay=0.0,
    batch_size=32,
    max_epochs=40,
    patience=5,
    device="cpu",
)
model = NnMlpModel(nn_cfg)

handle = display(plt.figure(), display_id=True)
history: list[dict[str, float]] = []


def on_epoch(entry: dict[str, float]) -> None:
    history.append(entry)
    fig = plots.loss_curve(
        history,
        title=f"Training progress — epoch {int(entry['epoch'])}",
    )
    clear_output(wait=True)
    handle.update(fig)
    plt.close(fig)


t0 = time.time()
model.fit(features, target, seed=0, epoch_callback=on_epoch)
elapsed = time.time() - t0
clear_output(wait=True)
print(f"NnMlpModel.fit: {elapsed:.2f}s over {len(model.loss_history_)} epoch(s).")
print(f"Best epoch: {model.metadata.hyperparameters['best_epoch']} (early-stop restored).")
print(f"Device resolved: {model.metadata.hyperparameters['device_resolved']}.")
"""
)


# ---------------------------------------------------------------------------
# Cell 6 — Static loss curve + commentary
# ---------------------------------------------------------------------------

cell_6 = md(
    """## Reading the final curve

The live curve above is discarded when the cell re-renders.  The cell
below reproduces the same curve with `plots.loss_curve(history)` for
the record — exactly the same helper, just called on the settled
history.  Two things to note in the static plot:

- The **gap** between train and validation loss at the best epoch is
  the model's irreducible generalisation gap on this window.  A big
  gap on a narrow feature set (5 weather columns) is expected — there
  is not enough signal for the network to close the gap without
  overfitting.
- The **right-hand rise** in validation loss past the best epoch is
  the reason Stage 9's registry records the *best-epoch* weights, not
  the last epoch's.  `plan D9` / `NnMlpModel._restore_best_state_dict`
  is the load-bearing piece.
"""
)

cell_7 = code(
    """# T6 Cell 7 — Final (static) loss curve.  AC-3 "available as a
# plot without additional wiring" evidence — one function call on the
# fitted model's ``loss_history_`` attribute.

fig = plots.loss_curve(
    model.loss_history_,
    title="Final training curve — 1 hidden layer (128 units), ReLU, 60-day window",
)
best_epoch = model.metadata.hyperparameters["best_epoch"]
best_val = next(
    entry["val_loss"] for entry in model.loss_history_ if entry["epoch"] == best_epoch
)
fig.axes[0].axvline(
    best_epoch,
    color=plots.OKABE_ITO[0],
    linestyle="--",
    linewidth=1.0,
    alpha=0.6,
    label=f"best epoch (val_loss={best_val:.3f})",
)
fig.axes[0].legend(loc="upper right")
fig.tight_layout()
plt.show()
"""
)


# ---------------------------------------------------------------------------
# Cell 8 — Cross-model comparison header
# ---------------------------------------------------------------------------

cell_8 = md(
    """## Comparing against the linear baseline + seasonal-naive

Intent §Scope closes with "*compares predictions against prior models*".
The honest comparison is through the Stage 6 rolling-origin harness:
fit each model on the same fold boundaries, collect MAE / MAPE / RMSE /
WAPE, and print the mean across folds.

The `NnMlpModel` here competes with the same `hidden_sizes=[128]`
defaults as the live-demo cell.  Expect the linear baseline to edge
the MLP on MAE and MAPE — the weather-only feature set is too narrow
for a 128-unit MLP to exploit a non-linear advantage (Stage 11's
temporal architecture is where the non-linearity starts paying
rent).  The seasonal-naive model is the "how hard could it be?" lower
bound: if the MLP cannot beat *that*, the stage has a bug.
"""
)

cell_9 = code(
    """# T6 Cell 9 — Three-way harness comparison over small rolling-origin
# folds.  Budget-friendly config: 3 folds of 168 h each, trained on
# ~720 h = 30 days.  A full-year rolling-origin pass belongs to the
# CLI (``python -m bristol_ml.train model=nn_mlp``), not to the
# notebook.

splitter_cfg = cfg.evaluation.rolling_origin.model_copy(
    update={
        "min_train_periods": 720,
        "test_len": 168,
        "step": 168,
        "fixed_window": True,
    }
)
metric_fns = [METRIC_REGISTRY[name] for name in ("mae", "mape", "rmse", "wape")]

# Instantiate fresh models so state from Cell 5 does not leak in.
cmp_nn_cfg = NnMlpConfig(
    target_column="nd_mw",
    hidden_sizes=[128],
    activation="relu",
    max_epochs=40,
    patience=5,
    batch_size=32,
    device="cpu",
)
naive_cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
linear_cfg = LinearConfig(feature_columns=tuple(feature_cols), target_column="nd_mw")

results: dict[str, pd.DataFrame] = {}
for name, candidate in [
    ("naive", NaiveModel(naive_cfg)),
    ("linear", LinearModel(linear_cfg)),
    ("nn_mlp", NnMlpModel(cmp_nn_cfg)),
]:
    t0 = time.time()
    metrics_df = evaluate(
        candidate,
        df,
        splitter_cfg,
        metric_fns,
        target_column="nd_mw",
        feature_columns=tuple(feature_cols),
    )
    print(f"{name:>8s}  evaluate: {time.time() - t0:5.1f}s  ({len(metrics_df)} folds)")
    results[name] = metrics_df

metric_names = [fn.__name__ for fn in metric_fns]
summary_df = pd.concat(
    [results[m][metric_names].mean().rename(m) for m in results],
    axis=1,
).T
summary_df.index.name = "model"
print()
print("Mean metric across folds (lower is better):")
print(summary_df.to_string(float_format=lambda v: f"{v:.3f}"))
"""
)


# ---------------------------------------------------------------------------
# Cell 10 — Closing
# ---------------------------------------------------------------------------

cell_10 = md(
    """## Closing — what the scaffold buys us, what's next

**What Stage 10 added over Stages 4 / 7.**  A PyTorch-backed `Model`
protocol conformer with a hand-rolled training loop (plan D10), a
four-stream reproducibility recipe (plan D7'), cold-start-per-fold
semantics (plan D8), and a `state_dict`-bytes-inside-joblib artefact
envelope (plan D5 revised) that plugs into the Stage 9 registry
without any registry change.  The analytical payoff (Cell 9's three-way
leaderboard) is typically *not* an MLP win on the weather-only feature
set — and the intent says so up front.  The training-loop conventions
and reproducibility discipline are the load-bearing contribution.

**Stage 11 hook.**  The `_run_training_loop` method carries an explicit
extraction-seam marker in the source — `# Stage 11 extraction seam:
move the body of this method and ``_make_mlp`` to
``src/bristol_ml/models/nn/_training.py`` when Stage 11's temporal
model arrives`.  Gradient clipping / LR scheduling are deliberately
**not** added at Stage 10 (scope-diff X6 cut); they belong behind the
shared helper, not in a growing `NnMlpModel`.

**Registry hook.**  `python -m bristol_ml.train model=nn_mlp` runs the
full rolling-origin pipeline end-to-end and registers the final-fold
model through the Stage 9 surface — the sidecar carries
`type = "nn_mlp"`, and `registry list --model-type nn_mlp` picks it up
alongside the other four families.  The leaderboard test covers this
wiring (see `tests/unit/registry/test_registry_nn_mlp_dispatch.py`).
"""
)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

notebook = {
    "cells": [
        cell_0,
        cell_1,
        cell_2,
        cell_3,
        cell_4,
        cell_5,
        cell_6,
        cell_7,
        cell_8,
        cell_9,
        cell_10,
    ],
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
