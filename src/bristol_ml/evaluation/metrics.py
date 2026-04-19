"""Point-forecast metrics for the Stage 4 evaluation harness.

Four functions — :func:`mae`, :func:`mape`, :func:`rmse`, :func:`wape` —
implemented as pure ``(y_true, y_pred) -> float`` kernels with rigorous
guards against common footguns (length mismatch, ``NaN`` inputs,
zero-denominators).  Each is the formula every future modelling stage is
judged against; see ``DESIGN`` §5.3 and plan D8-D9 for the standards.

Formulae (cited verbatim in the per-function docstrings):

- MAE   = ``mean(|y_true - y_pred|)``
- RMSE  = ``sqrt(mean((y_true - y_pred)**2))``
- MAPE  = ``mean(|(y_true - y_pred) / y_true|)`` — raises if any ``y_true == 0``
- WAPE  = ``sum(|y_true - y_pred|) / sum(|y_true|)`` (Kolassa & Schütz 2007;
  Hyndman form) — raises if ``sum(|y_true|) == 0``

Every function accepts ``np.ndarray | pd.Series | list[float]`` through a
:func:`numpy.asarray` coercion at entry, mirroring sklearn/statsmodels
conventions and avoiding type-dispatch in the caller.

:data:`METRIC_REGISTRY` maps lowercase metric names to the underlying
callables.  The harness in Task T6 consumes this registry via
``MetricsConfig.names``; downstream users should prefer the registry over
hard-coding function references so adding a new metric (Stage 5+) is a
one-line addition here.

Run standalone::

    python -m bristol_ml.evaluation.metrics [--help]

The CLI prints the names + formulae of every metric in
:data:`METRIC_REGISTRY` — useful for a live-demo sanity check that the
resolved config's ``evaluation.metrics.names`` all map to real functions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "METRIC_REGISTRY",
    "MetricFn",
    "mae",
    "mape",
    "rmse",
    "wape",
]


# ``numpy`` does not expose ``ArrayLike`` at runtime in a stable form; we
# pin to the three concrete types our callers actually pass.  Returning
# ``float`` (not ``np.floating``) keeps downstream logging / DataFrame
# assembly free of numpy dtype surprises.
type ArrayLike = np.ndarray | pd.Series | list[float]
type MetricFn = Callable[[ArrayLike, ArrayLike], float]


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _coerce_and_validate(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D ``float64`` arrays and reject malformed pairs.

    The guards here are shared by every metric so each function's kernel
    can stay a single expression.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` differ in length, either array
        contains ``NaN``, or either array is zero-length.
    """
    y_t = np.asarray(y_true, dtype="float64").ravel()
    y_p = np.asarray(y_pred, dtype="float64").ravel()
    if y_t.shape != y_p.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; got {y_t.shape[0]} vs {y_p.shape[0]}."
        )
    if y_t.size == 0:
        raise ValueError("y_true and y_pred must be non-empty; got zero-length arrays.")
    if np.isnan(y_t).any() or np.isnan(y_p).any():
        raise ValueError(
            "Metric inputs must not contain NaN; got "
            f"y_true NaN count={int(np.isnan(y_t).sum())}, "
            f"y_pred NaN count={int(np.isnan(y_p).sum())}."
        )
    return y_t, y_p


# ---------------------------------------------------------------------------
# Metric kernels
# ---------------------------------------------------------------------------


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error: ``mean(|y_true - y_pred|)``.

    DESIGN §5.3 names MAE as one of four point-forecast metrics for the
    Stage 4 evaluator.  Scale-dependent: the unit is the unit of
    ``y_true`` (MW for GB demand).
    """
    y_t, y_p = _coerce_and_validate(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error: ``sqrt(mean((y_true - y_pred)**2))``.

    DESIGN §5.3.  Emphasises large errors more than MAE does; still
    scale-dependent (MW for GB demand).
    """
    y_t, y_p = _coerce_and_validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error: ``mean(|(y_true - y_pred) / y_true|)``.

    Per plan D8 the zero-denominator policy is to **raise** — GB national
    demand never approaches zero in practice, so this guard should never
    fire for in-scope data; it protects callers from silent ``inf``
    poisoning.

    The returned value is a **fraction**, not a percentage
    (``0.02`` = 2 %).  DESIGN §5.3 expresses MAPE this way so downstream
    formatting owns the "*100" decision.

    Raises
    ------
    ValueError
        If ``y_true`` contains any zero (per plan D8).
    """
    y_t, y_p = _coerce_and_validate(y_true, y_pred)
    if np.any(y_t == 0.0):
        raise ValueError(
            "MAPE is undefined when y_true contains zeros "
            f"(found {int((y_t == 0.0).sum())} zero(s); plan D8)."
        )
    return float(np.mean(np.abs((y_t - y_p) / y_t)))


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted absolute percentage error: ``sum(|y_true - y_pred|) / sum(|y_true|)``.

    Kolassa & Schütz (2007); Hyndman form; equivalent to Amazon Forecast's
    WAPE and to Wikipedia's WMAPE.  Per plan D9 we use this definition
    rather than ``mean(|err|/|y|)`` (which is actually the mean form of
    MAPE, not WAPE).  Like MAPE, the returned value is a fraction.

    Raises
    ------
    ValueError
        If ``sum(|y_true|) == 0`` (per plan D9); WAPE is undefined when
        the aggregate magnitude of the actuals is zero.
    """
    y_t, y_p = _coerce_and_validate(y_true, y_pred)
    denom = float(np.sum(np.abs(y_t)))
    if denom == 0.0:
        raise ValueError(
            "WAPE is undefined when sum(|y_true|) == 0 "
            "(plan D9: requires a non-zero aggregate magnitude of actuals)."
        )
    return float(np.sum(np.abs(y_t - y_p)) / denom)


# ---------------------------------------------------------------------------
# Registry — consumed by the Task T6 harness via ``MetricsConfig.names``
# ---------------------------------------------------------------------------


METRIC_REGISTRY: dict[str, MetricFn] = {
    "mae": mae,
    "mape": mape,
    "rmse": rmse,
    "wape": wape,
}
"""Name → metric-function lookup, keyed by the lowercase
``Literal`` values on :class:`conf._schemas.MetricsConfig.names`.

Adding a new metric is a two-step:

1. Add a new pure function here following the :data:`MetricFn` signature.
2. Extend the ``MetricsConfig.names`` ``Literal`` (and this registry) with
   the new name so Hydra configs can reference it.
"""


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.evaluation.metrics``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.evaluation.metrics",
        description=(
            "Print the named point-forecast metrics registered in "
            "METRIC_REGISTRY alongside the resolved ``evaluation.metrics`` "
            "config's selection.  Useful for a live-demo sanity check."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluation.metrics.names=[mae,rmse]",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance."""
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    selected: tuple[str, ...]
    if cfg.evaluation is not None and cfg.evaluation.metrics is not None:
        selected = tuple(cfg.evaluation.metrics.names)
    else:
        selected = ()

    logger.info(
        "Metrics registry: registered={} selected={}",
        tuple(METRIC_REGISTRY),
        selected,
    )
    print("registered:", ", ".join(sorted(METRIC_REGISTRY)))
    if selected:
        print("selected:  ", ", ".join(selected))
    else:
        print("selected:   (no evaluation.metrics config resolved)")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
