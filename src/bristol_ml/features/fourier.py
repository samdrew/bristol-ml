"""Fourier exogenous features for periodic signals (Stage 7).

Pure-function derivation of ``sin``/``cos`` exogenous regressors at a
fixed period.  Built as a general helper (period is a parameter) rather
than a weekly-only utility, but the Stage 7 SARIMAX model only uses the
weekly (``period_hours=168``) default — see plan §5 and decisions D1/D3.

The helper sits in the features layer because it consumes an hourly
UTC-indexed frame and returns a frame with the same index plus two new
``float64`` columns per harmonic.  Unlike ``features.calendar`` it does
not read ``holidays_df``, it does not log, and it does not touch the
``Europe/London`` local date — the integer hour index is computed from
**UTC** nanoseconds directly, so the output is DST-insensitive by
construction (plan §5 rationale; Stage 3 contract: every calendar day
has exactly 24 UTC rows including on DST-change Sundays).

## Why a separate module

The function is pure and general.  Co-locating it with ``calendar.py``
would conflate two distinct derivations: calendar.py emits one-hot
columns derived from local-time components (hour-of-day, day-of-week,
month, holiday flags), while ``fourier.py`` emits continuous
trigonometric columns derived from an integer clock.  Keeping them
separate makes the call sites in SARIMAX (``SarimaxModel.fit`` /
``.predict``) easier to reason about: fit composes
``derive_calendar`` + ``append_weekly_fourier`` + SARIMAX in that
order, with each step testable on its own.

## Run standalone (principle §2.1.1)

::

    python -m bristol_ml.features.fourier [--help]

The CLI prints a one-line description of each public function and
exits.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = ["append_weekly_fourier"]


# Nanoseconds-per-hour constant; spelled out so the ``int64`` division is
# self-documenting.  ``pd.DatetimeIndex.view("int64")`` yields UTC
# nanoseconds since the Unix epoch for tz-aware indices.
_NANOSECONDS_PER_HOUR: int = 3_600_000_000_000


def append_weekly_fourier(
    df: pd.DataFrame,
    *,
    period_hours: int = 168,
    harmonics: int = 3,
    column_prefix: str = "week",
) -> pd.DataFrame:
    """Append Fourier sin/cos exogenous regressors at ``period_hours``.

    Parameters
    ----------
    df
        Hourly-indexed DataFrame with a **tz-aware** ``DatetimeIndex``.
        The Stage 3 assembler contract guarantees UTC, but any tz-aware
        index is accepted — the column values depend only on the
        integer-hour distance between rows, which is DST-insensitive
        because the conversion goes through UTC nanoseconds.
    period_hours
        Period of the Fourier basis, in hours.  Default ``168``
        (one week).  Plan D1/D3 pin the weekly default.
    harmonics
        Number of harmonic pairs to emit.  Default ``3`` — empirical
        research §R2 suggests 3-5 captures the weekly shape of GB
        demand without overfitting.  Must be non-negative.
        ``harmonics=0`` is a no-op: the input frame is returned
        unchanged (still a *new* frame, not an alias — see below).
    column_prefix
        Prefix for the emitted column names.  Default ``"week"``.
        Column names are ``f"{column_prefix}_sin_k{k}"`` and
        ``f"{column_prefix}_cos_k{k}"`` for ``k`` in ``1..harmonics``.

    Returns
    -------
    pandas.DataFrame
        A *new* DataFrame containing every column of ``df`` followed
        by ``2 * harmonics`` ``float64`` columns.  The input frame is
        never mutated (copy-on-write conforming).

    Raises
    ------
    ValueError
        If ``df.index`` is not a tz-aware :class:`DatetimeIndex`, or
        if ``harmonics`` is negative.

    Notes
    -----
    Values are ``sin(2πk · t / period_hours)`` and
    ``cos(2πk · t / period_hours)`` where ``t`` is the integer number
    of hours since the Unix epoch (UTC-anchored).  DST is irrelevant
    because the conversion goes through UTC nanoseconds — a UTC
    timestamp maps to the same integer hour regardless of the local
    clock.  On the two DST-transition Sundays the output is continuous
    across the transition (no jump), which is the property the
    SARIMAX model relies on.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"append_weekly_fourier requires a DatetimeIndex; got {type(df.index).__name__}."
        )
    if df.index.tz is None:
        raise ValueError(
            "append_weekly_fourier requires a tz-aware DatetimeIndex "
            "(the Stage 3 assembler contract guarantees UTC). "
            "Got a tz-naive index."
        )
    if harmonics < 0:
        raise ValueError(f"append_weekly_fourier requires harmonics >= 0; got {harmonics}.")

    # harmonics=0 fast path — still return a shallow copy so callers who
    # mutate the result do not accidentally touch the input frame.
    if harmonics == 0:
        return df.copy()

    # Integer hours since the Unix epoch, UTC-anchored.  ``view("int64")``
    # on a tz-aware DatetimeIndex yields UTC nanoseconds directly; floor
    # division by nanoseconds-per-hour yields the integer-hour clock.
    # This is DST-insensitive by construction: the same UTC timestamp
    # always maps to the same integer hour.
    hours_since_epoch = df.index.view("int64") // _NANOSECONDS_PER_HOUR
    t = np.asarray(hours_since_epoch, dtype=np.float64)

    # Build the new columns in a small dict so the output dtype is
    # unambiguously ``float64`` and the column-insertion order is
    # deterministic.
    new_columns: dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        angular = 2.0 * np.pi * float(k) * t / float(period_hours)
        new_columns[f"{column_prefix}_sin_k{k}"] = np.sin(angular)
        new_columns[f"{column_prefix}_cos_k{k}"] = np.cos(angular)

    # Assemble: copy the input, then attach each new column.  Avoids
    # the ``pd.concat`` overhead on a short column list and keeps the
    # output index identical to ``df.index`` (same object reference).
    out = df.copy()
    for name, values in new_columns.items():
        out[name] = values
    return out


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.features.fourier``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.features.fourier",
        description=(
            "Print a summary of the Fourier feature helpers in "
            "bristol_ml.features.fourier. Pure-function module; no I/O."
        ),
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    parser.parse_args(list(argv) if argv is not None else None)

    print("bristol_ml.features.fourier — Fourier exogenous feature helpers")
    print()
    print("Public functions:")
    print("  append_weekly_fourier(df, *, period_hours=168, harmonics=3, column_prefix='week')")
    print(
        "      Append 2*harmonics sin/cos columns at the given period. "
        "Requires a tz-aware DatetimeIndex. Pure; no I/O. DST-insensitive."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
