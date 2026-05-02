"""REMIT-derived hourly features for the Stage 16 model (point-in-time correct).

Stage 16 (intent ``docs/intent/16-model-with-remit.md``).  Produces the
three REMIT columns appended to the ``weather_calendar`` prefix in
``WITH_REMIT_OUTPUT_SCHEMA``:

- ``remit_unavail_mw_total`` (``float32``) — sum of ``affected_mw`` for
  REMIT revisions that are *active and known* at hour ``t``.  "Active":
  ``effective_from <= t < effective_to`` (or open-ended).  "Known": the
  revision is the latest one of its ``mrid`` whose ``published_at <= t``
  and whose ``message_status != "Withdrawn"`` (the bi-temporal as-of
  rule from Stage 13 ``ingestion.remit.as_of``).
- ``remit_active_unplanned_count`` (``int32``) — the count of revisions
  meeting the same active-and-known condition whose ``cause`` field
  matches ``"Unplanned"`` (case-insensitive).
- ``remit_unavail_mw_next_24h`` (``float32``) — sum of ``affected_mw``
  for revisions known at ``t`` whose ``effective_from`` lies in
  ``[t, t + lookahead_hours)`` — the "known future input" /
  "future covariate" signal (domain research §2 — TFT, AutoGluon).

Bi-temporal correctness (intent §"Bi-temporal correctness"; AC-1) is
**structurally enforced** by the algorithm, not by an opt-in flag: each
revision contributes only over the open half-interval during which it
is the latest visible revision of its ``mrid`` AND its event window
overlaps the relevant temporal predicate.  Bypassing this function
(e.g. by joining the raw REMIT log directly onto the hourly grid) would
expose the caller to leakage; callers must not.

Algorithm (vectorised; single pass over the REMIT log per signal):

1. **Per-mrid revision intervals.**  Within each ``mrid``, sort
   revisions by ``(published_at, revision_number)`` ascending.  The
   transaction-time validity of revision ``r`` is the half-open
   interval ``[published_at(r), published_at(r+1))`` — ``r`` is the
   latest visible revision from its publication until the next
   revision of the same ``mrid`` is published.  The terminal revision
   per ``mrid`` is valid until ``+inf`` (modelled as
   ``pd.Timestamp.max`` clamped to the hourly grid's right edge).
2. **Per-revision contribution windows.**  For each revision (skipping
   Withdrawn rows):

   - *active:*   ``[max(tx_valid_from, effective_from),
                    min(tx_valid_to, effective_to_or_inf))``
   - *forward:*  ``[max(tx_valid_from, effective_from - lookahead_h),
                    min(tx_valid_to, effective_from))``
   - *unplanned-count:* same as *active* but the delta is ``+1``
     instead of ``+affected_mw`` and only fires when ``cause`` matches
     ``"Unplanned"``.
3. **Delta-event aggregation.**  For each window ``[a, b)`` with
   contribution ``d``, emit two delta events: ``(a, +d)`` and
   ``(b, -d)``.  Sort events by timestamp, cumsum, then look up each
   hour ``t`` against the cumsum via :func:`pandas.merge_asof`
   (``direction="backward"``).  The result is the value of the running
   total at every hourly boundary in O((n_revisions + n_hours) log).

This module is pure (no I/O), conforms to the Stage 5 ``calendar.py``
shape — a single derivation function plus a typed column constant —
and emits one structured ``loguru`` INFO line per call (NFR-8).

Run standalone (DESIGN §2.1.1)::

    python -m bristol_ml.features.remit [--help]

The CLI loads the cached REMIT parquet (Stage 13) and the persisted
extractor parquet (Stage 14 + Stage 16 T3) when both are warm and
prints a small sample of the derived feature frame.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

__all__ = [
    "REMIT_VARIABLE_COLUMNS",
    "derive_remit_features",
]


# ---------------------------------------------------------------------------
# Schema — the three Stage 16 REMIT columns appended to the weather+calendar
# prefix.  The forward-looking column is always present in the schema for
# contract stability; ``WithRemitFeatureConfig.include_forward_lookahead``
# governs whether the model's ``feature_columns`` reads it (plan A2 / A4).
# ---------------------------------------------------------------------------


REMIT_VARIABLE_COLUMNS: Final[tuple[tuple[str, pa.DataType], ...]] = (
    ("remit_unavail_mw_total", pa.float32()),
    ("remit_active_unplanned_count", pa.int32()),
    ("remit_unavail_mw_next_24h", pa.float32()),
)
"""The three REMIT-derived columns and their pyarrow types.

Column order is contractual; the assembler's
:data:`bristol_ml.features.assembler.WITH_REMIT_OUTPUT_SCHEMA` reads
this constant verbatim.  The forward-looking column is named
``_next_24h`` regardless of the configured ``forward_lookahead_hours`` —
24 h is the day-ahead horizon and matches every realistic
configuration; renaming the column when the config changes would break
schema-driven downstream code (the model's ``feature_columns`` field,
the registry sidecar, the notebook's commentary).  If a future stage
needs a different lookahead, add a new column rather than renaming
this one.
"""


# Sentinel used as the right edge of the terminal revision's transaction-time
# validity interval and of any open-ended ``effective_to``.  Clamped per call
# to the hourly grid's right edge before delta-event emission so the cumsum
# domain stays compact.
_FAR_FUTURE: Final[pd.Timestamp] = pd.Timestamp("2262-04-11", tz="UTC")
"""A pandas-representable far-future timestamp (just inside the int64-ns range).

Used as the upper bound for "open-ended" intervals (terminal revisions
per mrid; ``effective_to is NaT``).  Clamped per call to the hourly
grid's right edge — see :func:`_clamp_far_future`.
"""


_UNPLANNED_TAG: Final[str] = "unplanned"
"""Case-folded ``cause`` value that fires the ``unplanned_count`` signal.

Live REMIT cause vocabulary uses Title Case (``"Planned"``, ``"Unplanned"``,
``"Forced"``, ...).  Comparison is case-insensitive so a stub fixture
that emits lowercase still fires.  The sentinel "Forced" is **not**
counted — it is upstream-defined as a different category from Unplanned.
"""


# ---------------------------------------------------------------------------
# Public derivation
# ---------------------------------------------------------------------------


def derive_remit_features(
    remit_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    *,
    forward_lookahead_hours: int = 24,
) -> pd.DataFrame:
    """Compute the three REMIT-derived hourly features for ``hourly_index``.

    Bi-temporal correctness is structural: the algorithm restricts each
    revision's contribution to the open half-interval during which it is
    both the latest known revision of its ``mrid`` and active per its
    event window.  No row of the result reflects information published
    after that row's timestamp (intent AC-1; module docstring §"Algorithm").

    Parameters
    ----------
    remit_df:
        A REMIT event log conforming to
        :data:`bristol_ml.ingestion.remit.OUTPUT_SCHEMA`.  Must carry the
        canonical 16-column shape with UTC-aware ``published_at``,
        ``effective_from``, and (nullable) ``effective_to``.  Withdrawn
        revisions are honoured per the bi-temporal as-of rule: a row
        whose ``message_status == "Withdrawn"`` does not contribute
        during its transaction-time validity interval.
    hourly_index:
        The target hourly grid — a tz-aware ``pd.DatetimeIndex`` in UTC,
        strictly monotonically increasing, gap-free, hourly cadence.
        Typically constructed from the ``weather_calendar`` feature
        table's ``timestamp_utc`` column.
    forward_lookahead_hours:
        Width of the forward-looking window for ``remit_unavail_mw_next_24h``
        (in hours).  Default 24 (the day-ahead horizon).  ``0`` is
        permitted but degenerate; the column is then always zero.

    Returns
    -------
    pandas.DataFrame
        A frame with column ``timestamp_utc`` (matching ``hourly_index``)
        and the three REMIT columns from :data:`REMIT_VARIABLE_COLUMNS`,
        in declared order.  Dtypes match :data:`REMIT_VARIABLE_COLUMNS`.
        Hours with no active or forward-windowed events have zero in
        every REMIT column (NFR — invariant: no NaN in REMIT columns;
        AC-7).

    Raises
    ------
    ValueError
        If ``hourly_index`` is tz-naive, not in UTC, not strictly
        monotonically increasing, or zero-length; or if ``remit_df``
        is missing any required column.
    """
    _validate_inputs(remit_df, hourly_index, forward_lookahead_hours)

    # Empty-corpus fast path: no REMIT events at all → all REMIT columns
    # are zero across the hourly grid (AC-7).  This also short-circuits
    # the downstream merge_asof which would otherwise be invoked on an
    # empty events frame.
    if len(remit_df) == 0:
        result = _zero_frame(hourly_index)
        logger.info(
            "REMIT feature derivation: empty REMIT log; emitting all-zero "
            "feature frame for {} hour(s).",
            len(hourly_index),
        )
        return result

    # The far-future sentinel used as the right edge of "open-ended"
    # transaction-time validity / valid-time intervals.  Clamped per call
    # to keep the cumsum domain compact and the merge_asof predictable.
    grid_right_edge = hourly_index[-1] + pd.Timedelta(hours=1)
    far_future = _clamp_far_future(grid_right_edge)

    # --- Step 1: per-mrid transaction-time validity intervals ----------------
    revisions = _per_mrid_validity(remit_df, far_future=far_future)

    # --- Step 2: per-revision contribution windows ---------------------------
    active_windows = _active_windows(revisions, far_future=far_future)
    forward_windows = _forward_windows(
        revisions,
        far_future=far_future,
        lookahead_hours=forward_lookahead_hours,
    )
    # Unplanned count uses the *active* windows but with a +1 / -1 delta
    # restricted to revisions whose cause matches "Unplanned".
    unplanned_mask = revisions["cause_tag"] == _UNPLANNED_TAG
    unplanned_windows = active_windows.loc[unplanned_mask].copy()

    # --- Step 3: delta-event aggregation per signal --------------------------
    total_mw_series = _running_total(active_windows, value_column="affected_mw_safe")
    forward_mw_series = _running_total(forward_windows, value_column="affected_mw_safe")
    unplanned_count_series = _running_total(unplanned_windows, value_column="one")

    # --- Look up each hourly t against the running totals --------------------
    total_mw = _lookup_at_grid(total_mw_series, hourly_index)
    forward_mw = _lookup_at_grid(forward_mw_series, hourly_index)
    unplanned_count = _lookup_at_grid(unplanned_count_series, hourly_index)

    # --- Assemble final frame in declared column order with declared dtypes -
    result = pd.DataFrame(
        {
            "timestamp_utc": hourly_index,
            "remit_unavail_mw_total": total_mw.astype("float32"),
            "remit_active_unplanned_count": unplanned_count.astype("int32"),
            "remit_unavail_mw_next_24h": forward_mw.astype("float32"),
        },
        index=pd.RangeIndex(len(hourly_index)),
    )

    # NFR-8: single structured INFO log per call.
    active_event_hours = int((total_mw > 0).sum())
    zero_event_hours = int((total_mw == 0).sum())
    forward_window_hits = int((forward_mw > 0).sum())
    logger.info(
        "REMIT feature derivation: revisions_in={} active_event_hours={} "
        "zero_event_hours={} forward_window_hits={} row_count={} "
        "lookahead_hours={}",
        len(remit_df),
        active_event_hours,
        zero_event_hours,
        forward_window_hits,
        len(result),
        forward_lookahead_hours,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "mrid",
    "revision_number",
    "message_status",
    "published_at",
    "effective_from",
    "effective_to",
    "cause",
    "affected_mw",
)


def _validate_inputs(
    remit_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    forward_lookahead_hours: int,
) -> None:
    """Reject malformed inputs early so downstream errors are obvious."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in remit_df.columns]
    if missing:
        raise ValueError(
            f"derive_remit_features: REMIT frame is missing required column(s) "
            f"{missing}; got {list(remit_df.columns)!r}.  Pass the output of "
            "bristol_ml.ingestion.remit.load() unmodified."
        )
    if not isinstance(hourly_index, pd.DatetimeIndex):
        raise ValueError(
            "derive_remit_features: hourly_index must be a pandas DatetimeIndex; "
            f"got {type(hourly_index).__name__}."
        )
    if len(hourly_index) == 0:
        raise ValueError(
            "derive_remit_features: hourly_index is empty; the function would "
            "emit a zero-row frame which is never useful — pass the assembler "
            "output's timestamp_utc column directly."
        )
    if hourly_index.tz is None:
        raise ValueError(
            "derive_remit_features: hourly_index is tz-naive.  REMIT bi-temporal "
            "queries are nonsense without a timezone reference — pass the "
            "assembler's tz-aware UTC index unchanged."
        )
    if str(hourly_index.tz) != "UTC":
        raise ValueError(
            f"derive_remit_features: hourly_index must be tz-aware UTC; got "
            f"tz={hourly_index.tz!r}.  Convert via .tz_convert('UTC') first."
        )
    if not hourly_index.is_monotonic_increasing:
        raise ValueError(
            "derive_remit_features: hourly_index is not strictly monotonically "
            "increasing.  The assembler's timestamp_utc is sorted ascending — "
            "upstream layer has regressed."
        )
    if forward_lookahead_hours < 0:
        raise ValueError(
            f"derive_remit_features: forward_lookahead_hours must be non-negative; "
            f"got {forward_lookahead_hours}."
        )


def _clamp_far_future(grid_right_edge: pd.Timestamp) -> pd.Timestamp:
    """Cap the open-ended sentinel at the hourly grid's right edge.

    Revisions that are valid "until the next revision is published" use
    ``pd.Timestamp.max``-shaped sentinels internally; clamping them to
    one hour past the last grid timestamp keeps the cumsum domain
    bounded and keeps the merge_asof's right-edge behaviour predictable
    (the closing ``-d`` delta lands strictly past the last grid hour, so
    the running total stays at its correct value across the entire grid
    rather than dropping to zero at the sentinel).
    """
    return min(_FAR_FUTURE, grid_right_edge + pd.Timedelta(hours=1))


def _per_mrid_validity(
    remit_df: pd.DataFrame,
    *,
    far_future: pd.Timestamp,
) -> pd.DataFrame:
    """Tag each revision with its transaction-time validity interval.

    For each ``mrid`` the revisions are sorted by ``(published_at,
    revision_number)`` ascending.  Within that order, revision ``r`` is
    the latest visible revision from ``published_at(r)`` until
    ``published_at(r+1)`` (or ``far_future`` for the terminal
    revision).

    A subtle point: ``Withdrawn`` revisions still **truncate** the prior
    revision's transaction-time validity, even though they themselves
    do not contribute to any signal.  So the ``tx_valid_to`` shift is
    taken over the **full** sorted log including Withdrawn rows; the
    Withdrawn rows are dropped only at the end.  Without this, a
    sequence ``rev0:Active, rev1:Withdrawn`` would leave rev0 looking
    valid for all eternity — the inverse of what the as_of rule says.

    The function also tags each revision with:

    - ``affected_mw_safe`` — ``affected_mw`` with NaN replaced by 0.0
      (NaN ``affected_mw`` rows do not contribute to any sum but still
      occupy a transaction-time interval; the count column ignores them
      via ``one``-column zeros).
    - ``cause_tag`` — case-folded ``cause`` (``""`` for NULL).
    - ``effective_to_filled`` — ``effective_to`` with NaT replaced by
      ``far_future`` so the active-window math is total.
    - ``one`` — constant 1.0, used as the count delta for the unplanned
      signal.
    """
    df = remit_df.copy()
    # Sort once — used by both the per-mrid groupby and the contribution
    # windowing below.  ``sort=False`` on the groupby preserves the per-
    # group ordering established here.
    df = df.sort_values(["mrid", "published_at", "revision_number"], kind="stable")

    # tx_valid_from is published_at; tx_valid_to is the next revision's
    # published_at within the same mrid (REGARDLESS of message_status,
    # so Withdrawn revisions correctly truncate prior validity), or
    # far_future for the terminal revision per mrid.
    df["tx_valid_from"] = df["published_at"]
    df["tx_valid_to"] = df.groupby("mrid", sort=False)["published_at"].shift(-1)
    df["tx_valid_to"] = df["tx_valid_to"].fillna(far_future)

    df["affected_mw_safe"] = df["affected_mw"].fillna(0.0).astype(float)
    df["effective_to_filled"] = df["effective_to"].fillna(far_future)
    df["cause_tag"] = df["cause"].fillna("").str.casefold()
    df["one"] = 1.0

    # Drop Withdrawn revisions LAST so they shape the prior revision's
    # tx_valid_to but do not themselves contribute.
    return df.loc[df["message_status"] != "Withdrawn"].copy()


def _active_windows(
    revisions: pd.DataFrame,
    *,
    far_future: pd.Timestamp,
) -> pd.DataFrame:
    """Compute the *active* contribution interval per revision.

    A revision contributes to ``remit_unavail_mw_total`` over the
    intersection of its transaction-time validity interval and its
    valid-time event window:

    ``[max(tx_valid_from, effective_from),
       min(tx_valid_to, effective_to_filled))``

    The function returns one row per revision with the window edges
    plus the contribution columns; rows where the interval is empty
    (``window_from >= window_to``) are dropped — they would emit a
    cancelling delta-event pair contributing nothing.
    """
    df = revisions.copy()
    df["window_from"] = df[["tx_valid_from", "effective_from"]].max(axis=1)
    df["window_to"] = df[["tx_valid_to", "effective_to_filled"]].min(axis=1)
    return df.loc[df["window_from"] < df["window_to"]].copy()


def _forward_windows(
    revisions: pd.DataFrame,
    *,
    far_future: pd.Timestamp,
    lookahead_hours: int,
) -> pd.DataFrame:
    """Compute the *forward-looking* contribution interval per revision.

    A revision contributes to ``remit_unavail_mw_next_24h`` over the
    intersection of its transaction-time validity interval and the
    interval ``[effective_from - lookahead, effective_from)`` —
    i.e. the lookahead-wide pre-window during which the event is
    "scheduled to start within the next ``lookahead`` hours" yet has
    not yet started:

    ``[max(tx_valid_from, effective_from - lookahead_h),
       min(tx_valid_to, effective_from))``

    A revision whose ``effective_from`` lies before its
    ``tx_valid_from`` (i.e. is "already started" at publication time)
    contributes nothing to the forward signal — the window collapses
    to empty and is dropped.
    """
    df = revisions.copy()
    lookahead = pd.Timedelta(hours=lookahead_hours)
    pre_start = df["effective_from"] - lookahead
    df["window_from"] = df[["tx_valid_from"]].max(axis=1).combine(pre_start, max)
    df["window_to"] = df[["tx_valid_to", "effective_from"]].min(axis=1)
    return df.loc[df["window_from"] < df["window_to"]].copy()


def _running_total(
    windows: pd.DataFrame,
    *,
    value_column: str,
) -> pd.Series:
    """Convert per-revision contribution windows to a running-total series.

    Emits two delta events per window — ``(window_from, +value)`` and
    ``(window_to, -value)`` — concatenates them, sorts by timestamp,
    and cumsums the deltas.  The returned series is indexed on the
    sorted, unique delta timestamps (tz-aware UTC, ns precision) and
    carries the running total at each one.

    An empty-windows input yields an empty series — handled at the
    grid-lookup step by treating "no events visible" as zero.
    """
    if windows.empty:
        return pd.Series(
            dtype="float64",
            index=pd.DatetimeIndex([], tz="UTC", name="t"),
        )

    # Coerce the timestamp axis to tz-aware UTC nanosecond precision up
    # front: the REMIT parquet round-trips at us precision and the
    # downstream merge_asof against the assembler's hourly index (ns
    # precision) is strict on dtype equality.
    window_from = (
        pd.DatetimeIndex(windows["window_from"]).tz_convert("UTC").as_unit("ns")
    )
    window_to = (
        pd.DatetimeIndex(windows["window_to"]).tz_convert("UTC").as_unit("ns")
    )
    deltas = pd.concat(
        [
            pd.DataFrame(
                {"t": window_from, "delta": windows[value_column].astype(float).values}
            ),
            pd.DataFrame(
                {"t": window_to, "delta": -windows[value_column].astype(float).values}
            ),
        ],
        ignore_index=True,
        copy=False,
    )
    # Sum coincident deltas first so the cumsum's value at any timestamp
    # is the post-event running total (the open half-interval's left
    # edge is +d, right edge is -d, so coincident edges cancel correctly).
    grouped = deltas.groupby("t", sort=True)["delta"].sum()
    return grouped.cumsum()


def _lookup_at_grid(
    running_total: pd.Series,
    hourly_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Look up each hourly timestamp against the running-total series.

    Uses :func:`pandas.merge_asof` with ``direction="backward"`` so
    each hourly ``t`` reads the cumsum value at the most recent delta
    at or before ``t``.  Hours preceding any delta receive zero (the
    pre-corpus baseline).
    """
    if running_total.empty:
        return np.zeros(len(hourly_index), dtype=float)

    rt = running_total.reset_index()
    rt.columns = ["t", "value"]
    # ``_running_total`` already coerces the index to tz-aware UTC ns;
    # the assembler's hourly grid is also ns-precision tz-aware UTC.
    # Coerce the grid for symmetry so a caller passing a pyarrow-derived
    # us-precision index does not surprise the merge_asof dtype check.
    grid_ts = hourly_index.tz_convert("UTC").as_unit("ns")
    grid = pd.DataFrame({"t": grid_ts})
    merged = pd.merge_asof(grid, rt, on="t", direction="backward")
    return merged["value"].fillna(0.0).to_numpy()


def _zero_frame(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return the all-zero REMIT feature frame for an empty REMIT log."""
    return pd.DataFrame(
        {
            "timestamp_utc": hourly_index,
            "remit_unavail_mw_total": np.zeros(len(hourly_index), dtype="float32"),
            "remit_active_unplanned_count": np.zeros(len(hourly_index), dtype="int32"),
            "remit_unavail_mw_next_24h": np.zeros(len(hourly_index), dtype="float32"),
        },
        index=pd.RangeIndex(len(hourly_index)),
    )


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.features.remit`  (DESIGN §2.1.1)
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.features.remit",
        description=(
            "Derive the three Stage 16 REMIT features from the cached REMIT "
            "log and the assembled weather+calendar feature table.  Prints "
            "the resolved schema + a small sample of the derived frame."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of derived sample rows to print (default 5).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides applied on top of the default config.",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local imports keep ``--help`` lightweight.
    from bristol_ml.config import load_config
    from bristol_ml.features.assembler import load_calendar
    from bristol_ml.ingestion import remit as remit_ingest

    cfg = load_config(overrides=["features=with_remit", *args.overrides])

    # Print the schema regardless of whether caches are warm — the
    # standalone-module entry must always do *something* useful.
    print("REMIT_VARIABLE_COLUMNS (Stage 16):")
    for name, dtype in REMIT_VARIABLE_COLUMNS:
        print(f"  {name:<32s} {dtype}")

    if cfg.features.with_remit is None:
        print("Stage 16 with_remit feature config not resolved; skipping sample.")
        return 0
    if cfg.ingestion.remit is None:
        print("REMIT ingestion config not resolved; skipping sample.")
        return 0

    remit_cache = cfg.ingestion.remit.cache_dir / cfg.ingestion.remit.cache_filename
    if not remit_cache.exists():
        print(f"REMIT cache missing at {remit_cache}; skipping sample.")
        return 0

    remit_df = remit_ingest.load(remit_cache)
    print(f"\nREMIT log loaded from {remit_cache}: {len(remit_df)} row(s)")

    # Find any populated weather+calendar cache to derive against; if
    # absent, build a synthetic 48-hour grid spanning the REMIT corpus.
    calendar_cfg = cfg.features.weather_calendar
    if calendar_cfg is None:
        # Synthesise a tiny grid so the standalone smoke prints something.
        if remit_df.empty:
            print("REMIT log empty; cannot synthesise sample grid.")
            return 0
        start = remit_df["effective_from"].min().floor("h")
        index = pd.date_range(start, periods=48, freq="h", tz="UTC")
    else:
        feature_cache = calendar_cfg.cache_dir / calendar_cfg.cache_filename
        if not feature_cache.exists():
            print(f"weather_calendar cache missing at {feature_cache}; using synthetic grid.")
            start = remit_df["effective_from"].min().floor("h")
            index = pd.date_range(start, periods=48, freq="h", tz="UTC")
        else:
            df = load_calendar(feature_cache)
            index = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert("UTC")[: max(args.rows, 48)]

    derived = derive_remit_features(
        remit_df,
        index,
        forward_lookahead_hours=cfg.features.with_remit.forward_lookahead_hours,
    )
    print("\nDerived REMIT features (head):")
    print(derived.head(args.rows).to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
