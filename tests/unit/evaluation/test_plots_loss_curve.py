"""Spec-derived tests for ``bristol_ml.evaluation.plots.loss_curve`` — Stage 10 T5.

Every test is derived from:

- ``docs/plans/active/10-simple-nn.md`` §6 Task T5 (two loss-curve
  tests: renders-figure and ax-composability).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-3 (loss curve is produced
  by the training loop itself and is available as a plot without
  additional wiring).
- ``docs/plans/active/10-simple-nn.md`` §1 D6 (``loss_history_`` shape is
  ``list[dict]`` with keys ``{"epoch", "train_loss", "val_loss"}``;
  ``loss_curve(history, *, ax=None)`` is the plot surface).
- Stage 6 `ax=` composability contract (``docs/architecture/layers/evaluation.md``).

The helper is **model-agnostic** (Stage 6 AC-3): it takes a sequence of
dicts, never a ``Model`` object.  The tests below feed it purely
synthetic lists so the NN training loop is not exercised — T2/T3's
end-to-end tests already cover the integration side.

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

from bristol_ml.evaluation.plots import OKABE_ITO, loss_curve


def _fake_history(n_epochs: int = 12) -> list[dict[str, float]]:
    """Return a monotonically-decreasing train-loss, U-shaped val-loss history.

    The U-shape matches the pedagogical moment that AC-3 describes (the
    facilitator points at the epoch where validation loss bottoms out
    and starts rising).  The exact numbers do not matter for the
    render-smoke tests — the key contract is the three-key shape.
    """
    history: list[dict[str, float]] = []
    for i in range(n_epochs):
        # Train decreases monotonically; val U-shapes and starts rising
        # around epoch 6 — the classic "this is overfitting" moment.
        train_loss = 1.0 - 0.08 * i + 0.001 * i * i
        val_loss = 1.0 - 0.06 * i + 0.008 * i * i
        history.append(
            {
                "epoch": float(i + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
    return history


# ===========================================================================
# 1. test_plots_loss_curve_renders_figure_from_history  (AC-3 / T5)
# ===========================================================================


def test_plots_loss_curve_renders_figure_from_history() -> None:
    """Guards AC-3 — ``loss_curve(history)`` returns a populated Figure.

    AC-3 requires the loss curve to be "available as a plot without
    additional wiring"; this means the helper, fed a realistic
    ``loss_history_`` payload, must produce a figure with two curves
    (train + validation) and sensible axis labels.  A bare empty
    canvas would not satisfy AC-3.

    Plan clause: T5 §Task T5 / AC-3.
    """
    history = _fake_history()
    fig = loss_curve(history)
    try:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"loss_curve must return a matplotlib Figure; got {type(fig).__name__!r}."
        )
        axes = fig.axes
        assert len(axes) >= 1, "Figure must have at least one Axes."
        lines = axes[0].lines
        assert len(lines) == 2, (
            f"loss_curve must draw exactly two lines (train + validation); got {len(lines)}."
        )
        # Axis labels — British English, matching the rest of the Stage 6 surface.
        assert axes[0].get_xlabel() == "Epoch", (
            f"loss_curve x-axis label must be 'Epoch'; got {axes[0].get_xlabel()!r}."
        )
        assert axes[0].get_ylabel() == "Loss", (
            f"loss_curve y-axis label must be 'Loss'; got {axes[0].get_ylabel()!r}."
        )
        # Legend must carry both series names so the facilitator knows which
        # line is which at a glance.
        legend = axes[0].get_legend()
        assert legend is not None, "loss_curve must draw a legend (train + validation)."
        legend_texts = {t.get_text() for t in legend.get_texts()}
        assert legend_texts == {"Train", "Validation"}, (
            f"loss_curve legend must be {{'Train', 'Validation'}}; got {legend_texts!r}."
        )
        # Colours must come from the Okabe-Ito palette — AC-6 / Stage 6 D2.
        # Train in OKABE_ITO[1] (orange), validation in OKABE_ITO[2] (sky blue).
        colours = {line.get_label(): line.get_color() for line in lines}
        assert colours["Train"].lower() == OKABE_ITO[1].lower(), (
            f"Train curve must use OKABE_ITO[1] ({OKABE_ITO[1]!r}); got {colours['Train']!r}."
        )
        assert colours["Validation"].lower() == OKABE_ITO[2].lower(), (
            f"Validation curve must use OKABE_ITO[2] ({OKABE_ITO[2]!r}); "
            f"got {colours['Validation']!r}."
        )
    finally:
        plt.close(fig)


# ===========================================================================
# 2. test_plots_loss_curve_respects_ax_composability_contract  (D5 / T5)
# ===========================================================================


def test_plots_loss_curve_respects_ax_composability_contract() -> None:
    """Guards plan T5 / Stage 6 D5 — ``ax=`` passthrough contract.

    Every Stage 6 helper accepts an optional ``ax: Axes | None``;
    passing a non-``None`` ``ax`` must return the owning figure rather
    than minting a new one.  This is the composability contract that
    lets facilitators put the loss curve alongside (say) a
    residuals-vs-time plot in a ``plt.subplots(1, 2)`` grid.

    Plan clause: T5 §Task T5 / Stage 6 D5.
    """
    history = _fake_history()
    fig_outer, ax_outer = plt.subplots()
    try:
        fig_returned = loss_curve(history, ax=ax_outer)
        assert fig_returned is fig_outer, (
            "loss_curve(ax=ax) must return the figure that owns ax, "
            "not create a new Figure (Stage 6 D5 composability contract)."
        )
        # Lines were drawn onto the supplied axes, not some new Axes object.
        assert len(ax_outer.lines) == 2, (
            f"loss_curve(ax=ax) must draw its 2 curves on the supplied Axes; "
            f"got {len(ax_outer.lines)} lines on the outer Axes."
        )
    finally:
        plt.close(fig_outer)


# ===========================================================================
# 3. test_plots_loss_curve_rejects_empty_history  (defensive contract)
# ===========================================================================


def test_plots_loss_curve_rejects_empty_history() -> None:
    """Guards D6 — an empty history is a programming bug, not a valid plot request.

    A fresh ``NnMlpModel`` carries ``loss_history_ = []``; calling the
    helper on it means the caller has forgotten to ``fit()``.  The
    helper fails loudly rather than rendering an empty canvas.

    Plan clause: T5 §Task T5 / D6 (history-shape contract).
    """
    with pytest.raises(ValueError, match=r"(?i)empty"):
        loss_curve([])


# ===========================================================================
# 4. test_plots_loss_curve_rejects_missing_required_key  (schema guard)
# ===========================================================================


def test_plots_loss_curve_rejects_missing_required_key() -> None:
    """Guards D6 — a history dict missing one of the three keys is rejected loudly.

    The helper's contract is a three-key dict.  A history entry that
    drops ``val_loss`` (for example, a mistakenly disabled validation
    split) must fail loudly rather than rendering a mysteriously
    single-line figure.

    Plan clause: T5 §Task T5 / D6.
    """
    bad_history = [
        {"epoch": 1.0, "train_loss": 0.5, "val_loss": 0.6},
        {"epoch": 2.0, "train_loss": 0.4},  # val_loss dropped
    ]
    with pytest.raises(ValueError, match=r"val_loss"):
        loss_curve(bad_history)
