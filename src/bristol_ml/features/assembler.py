"""Feature assembler — joins hourly demand with the national weather aggregate.

Stage 3 (§DESIGN 5.1; `docs/intent/03-feature-assembler.md`). The assembler
is a pure function: frame(s) in, frame out. It consumes the shipped
ingestion-layer outputs (``bristol_ml.ingestion.neso.load`` and
``bristol_ml.ingestion.weather.load`` → national-aggregate wide frame) and
produces a single hourly feature table keyed on ``timestamp_utc``. Declared
schema is enforced at the module boundary via the ``OUTPUT_SCHEMA``
constant; downstream stages treat the schema as a contract and may rely on
column order, dtype, and timezone metadata.

Decisions (see ``docs/plans/completed/03-feature-assembler.md`` §1):

* **D1** — half-hourly NESO rows are aggregated to hourly via
  ``mean`` (default) or ``max``; the choice is a config field
  (``FeatureSetConfig.demand_aggregation``), not a code change.
* **D5** — demand NaN rows are dropped; weather NaN is forward-filled up to
  ``config.forward_fill_hours`` (default 3), else the row is dropped. Every
  ``build()`` call emits a structured INFO log line naming the counts of
  rows affected by each policy.
* **D8** — two scalar provenance columns, ``neso_retrieved_at_utc`` and
  ``weather_retrieved_at_utc``, propagated from the inputs. Downstream code
  that wants a per-run audit trail reads these columns; they are repeated
  on every row (per DESIGN §2.1.6 — cheap, greppable).

Run standalone::

    python -m bristol_ml.features.assembler [--help]

The CLI ties ``neso.fetch → neso.load → _resample_demand_hourly →
weather.fetch → weather.load → national_aggregate → build → _atomic_write``
and prints the output path. See Task T4 in the plan.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing-only imports
    from conf._schemas import AppConfig, FeatureSetConfig

__all__ = [
    "CALENDAR_OUTPUT_SCHEMA",
    "CALENDAR_VARIABLE_COLUMNS",
    "DEMAND_COLUMNS",
    "OUTPUT_SCHEMA",
    "WEATHER_VARIABLE_COLUMNS",
    "assemble",
    "assemble_calendar",
    "build",
    "load",
    "load_calendar",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


DEMAND_COLUMNS: tuple[str, ...] = ("nd_mw", "tsd_mw")
"""Demand columns carried through the hourly resample. Both are int32 MW."""


WEATHER_VARIABLE_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("temperature_2m", pa.float32()),
    ("dew_point_2m", pa.float32()),
    ("wind_speed_10m", pa.float32()),
    ("cloud_cover", pa.float32()),
    ("shortwave_radiation", pa.float32()),
)
"""Weather aggregate columns and their arrow types in the feature table.

Note that ``cloud_cover`` is widened from ``int8`` (long-form weather
schema) to ``float32`` here: the population-weighted national aggregate is
a weighted mean of integer station observations and is only integer-valued
by coincidence. Keeping it ``float32`` mirrors the other weather variables
and avoids a lossy round at the schema boundary.
"""


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("nd_mw", pa.int32()),
        ("tsd_mw", pa.int32()),
        *WEATHER_VARIABLE_COLUMNS,
        ("neso_retrieved_at_utc", pa.timestamp("us", tz="UTC")),
        ("weather_retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk parquet schema for a Stage 3 feature table.

Column order is contractual — downstream code may rely on it. Adding a new
column is an additive change; renaming or reordering is a breaking one.
"""


# ---------------------------------------------------------------------------
# Stage 5 — weather + calendar schema
# ---------------------------------------------------------------------------
#
# ``CALENDAR_VARIABLE_COLUMNS`` is owned by ``features.calendar`` (the
# derivation module); it is re-exported here so the assembler's public
# surface carries both schemas without forcing downstream callers to import
# two submodules for a single feature-table read.
from bristol_ml.features.calendar import CALENDAR_VARIABLE_COLUMNS  # noqa: E402

CALENDAR_OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        *OUTPUT_SCHEMA,
        *CALENDAR_VARIABLE_COLUMNS,
        ("holidays_retrieved_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)
"""The on-disk parquet schema for a Stage 5 ``weather_calendar`` feature table.

Composition (55 columns total):

- positions 0..9 → ``OUTPUT_SCHEMA.names`` (the 10-column weather-only
  schema, unchanged — the weather-only frame is a prefix of the calendar
  frame so downstream code that reads only the weather columns continues
  to work by column-name selection).
- positions 10..53 → :data:`CALENDAR_VARIABLE_COLUMNS` (44 ``int8``
  columns: 23 one-hot hour, 6 one-hot day-of-week, 11 one-hot month, 4
  holiday flags under plan D-2 / D-5).
- position 54 → ``holidays_retrieved_at_utc`` (``timestamp[us, tz=UTC]``),
  the third provenance scalar (plan D-8 continuation — one per upstream
  ingester; ``neso_retrieved_at_utc`` / ``weather_retrieved_at_utc`` live
  inside the weather-only prefix at positions 8 / 9).

Column order is contractual; additions are additive (append), renames or
reorders are breaking.  See plan T4 for the structural invariant pinned by
``test_calendar_output_schema_is_weather_schema_plus_calendar_plus_provenance``.
"""


# ---------------------------------------------------------------------------
# Demand resampling (§6 Task T3, D1)
# ---------------------------------------------------------------------------


def _resample_demand_hourly(
    df: pd.DataFrame,
    agg: Literal["mean", "max"] = "mean",
) -> pd.DataFrame:
    """Aggregate half-hourly NESO demand to hourly resolution.

    The NESO ingester emits tz-aware UTC timestamps at half-hourly cadence
    (``timestamp_utc`` spaced 30 minutes apart). This function floors each
    timestamp to the containing UTC hour and applies ``agg`` to the two
    settlement periods landing in that hour. On clock-change Sundays the
    UTC timeline is still regular — the NESO layer has already unwound the
    DST algebra — so a spring-forward day collapses to 23 hourly rows and
    an autumn-fallback day to 25.

    Parameters
    ----------
    df
        DataFrame with a tz-aware ``timestamp_utc`` column and the two
        demand columns ``nd_mw`` and ``tsd_mw``. Extra columns are carried
        only if they aggregate meaningfully under ``agg`` — this function
        keeps only the three named columns to avoid silently meaning-shifting
        e.g. ``settlement_period``.
    agg
        Aggregation function. ``"mean"`` is the default and preserves MW
        scale; ``"max"`` returns the half-hourly peak per hour for a peak-
        demand framing (D1). Anything else raises ``ValueError`` — keeping
        the set tight so a config typo fails fast.

    Returns
    -------
    pandas.DataFrame
        One row per UTC hour, sorted ascending by ``timestamp_utc``.
        Columns: ``timestamp_utc``, ``nd_mw``, ``tsd_mw``. Dtypes are
        preserved from the input (int32 MW).

    Raises
    ------
    ValueError
        If ``timestamp_utc`` is missing, if ``timestamp_utc`` is not
        tz-aware, if a demand column is missing, or if ``agg`` is not in
        ``{"mean", "max"}``.
    """
    if agg not in ("mean", "max"):
        raise ValueError(
            f"demand_aggregation must be 'mean' or 'max'; got {agg!r} (Plan D1 Literal contract)."
        )
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"_resample_demand_hourly expects a 'timestamp_utc' column; got {list(df.columns)!r}."
        )
    missing = [c for c in DEMAND_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"_resample_demand_hourly is missing demand column(s) {missing}; "
            f"got {list(df.columns)!r}."
        )
    if df["timestamp_utc"].dt.tz is None:
        raise ValueError(
            "_resample_demand_hourly requires a tz-aware 'timestamp_utc'; got tz-naive. "
            "The NESO ingester emits UTC tz-aware timestamps — upstream layer has regressed."
        )

    hourly = (
        df[["timestamp_utc", *DEMAND_COLUMNS]]
        .assign(timestamp_utc=df["timestamp_utc"].dt.floor("h"))
        .groupby("timestamp_utc", as_index=False, sort=True)
        .agg({col: agg for col in DEMAND_COLUMNS})
    )
    # Cast back to int32 — the pandas aggregation path promotes to float64.
    for col in DEMAND_COLUMNS:
        hourly[col] = hourly[col].round().astype("int32")
    return hourly


# ---------------------------------------------------------------------------
# build — join + fill + drop + enforce schema (§6 Task T3, D5)
# ---------------------------------------------------------------------------


def build(
    demand_hourly: pd.DataFrame,
    weather_national: pd.DataFrame,
    config: FeatureSetConfig,
    *,
    neso_retrieved_at_utc: pd.Timestamp | None = None,
    weather_retrieved_at_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assemble the hourly feature table from demand + national weather.

    Join strategy:

    1. Inner-join demand with the wide-form national weather on
       ``timestamp_utc`` — the intersection of both timelines.
    2. Forward-fill each weather variable up to
       ``config.forward_fill_hours`` consecutive missing hours. This closes
       short ingestion gaps without masking a real upstream outage.
    3. Drop any row whose demand columns are NaN (D5 — spring-forward
       dropped hours, upstream corruption).
    4. Drop any row whose weather columns are still NaN after step 2.
    5. Assert the result conforms to ``OUTPUT_SCHEMA``.

    A single structured INFO log line is emitted on every call, covering
    ``demand_nan_rows_dropped``, ``weather_forward_filled_rows`` (per
    variable), ``weather_nan_rows_dropped_after_fill``, and the final
    ``row_count``. This is audible in notebooks and CLI runs without
    debug toggles (D5) and is greppable in stage retros.

    Parameters
    ----------
    demand_hourly
        Output of :func:`_resample_demand_hourly`. Must have
        ``timestamp_utc``, ``nd_mw``, ``tsd_mw``.
    weather_national
        Output of ``bristol_ml.features.weather.national_aggregate``. A
        wide-form frame indexed by ``timestamp_utc`` (name-preserving),
        one column per weather variable. Extra columns are ignored.
    config
        The resolved :class:`FeatureSetConfig` for this feature set.
        ``forward_fill_hours`` drives the fill cap.
    neso_retrieved_at_utc
        Scalar provenance for the NESO fetch (D8). Optional; if ``None``,
        the ``retrieved_at_utc`` column of ``demand_hourly`` (if present)
        is used, else :func:`pandas.Timestamp.now` with ``tz="UTC"`` at call time.
    weather_retrieved_at_utc
        Scalar provenance for the weather fetch (D8). Optional; same
        fallback semantics as ``neso_retrieved_at_utc``.

    Returns
    -------
    pandas.DataFrame
        The joined feature frame, ordered by ``timestamp_utc`` ascending,
        conforming to ``OUTPUT_SCHEMA`` when cast to arrow. A tz-aware
        UTC ``timestamp_utc`` column is guaranteed.

    Raises
    ------
    ValueError
        If either input is missing ``timestamp_utc``, if the join index
        is not tz-aware UTC, or if any expected weather variable is
        absent from ``weather_national``.
    """
    _validate_build_inputs(demand_hourly, weather_national)

    # --- Weather: ensure the index is a column named 'timestamp_utc' -----
    weather = weather_national.copy()
    if weather.index.name == "timestamp_utc" and "timestamp_utc" not in weather.columns:
        weather = weather.reset_index()
    elif "timestamp_utc" not in weather.columns:
        raise ValueError(
            "weather_national must have 'timestamp_utc' as a column or as the index; "
            f"got columns={list(weather.columns)}, index.name={weather.index.name!r}."
        )

    expected_weather_names = [name for name, _dtype in WEATHER_VARIABLE_COLUMNS]
    missing_weather = [c for c in expected_weather_names if c not in weather.columns]
    if missing_weather:
        raise ValueError(
            f"weather_national is missing expected variable(s): {missing_weather}. "
            "Call national_aggregate with the default Stage 2 variable set."
        )

    weather = weather[["timestamp_utc", *expected_weather_names]].sort_values(
        "timestamp_utc", kind="stable"
    )

    # --- Demand: drop NaN demand rows up-front and remember the count ----
    demand_nan_mask = demand_hourly[list(DEMAND_COLUMNS)].isna().any(axis=1)
    demand_nan_rows_dropped = int(demand_nan_mask.sum())
    demand_clean = demand_hourly.loc[~demand_nan_mask].copy()

    # --- Inner join + weather forward-fill then drop -----------------------
    joined = demand_clean.merge(weather, on="timestamp_utc", how="inner", validate="one_to_one")
    joined = joined.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)

    pre_fill_nan = joined[expected_weather_names].isna().sum()
    if config.forward_fill_hours > 0:
        joined[expected_weather_names] = joined[expected_weather_names].ffill(
            limit=config.forward_fill_hours
        )
    post_fill_nan = joined[expected_weather_names].isna().sum()
    weather_forward_filled_rows = (pre_fill_nan - post_fill_nan).to_dict()

    weather_nan_mask = joined[expected_weather_names].isna().any(axis=1)
    weather_nan_rows_dropped_after_fill = int(weather_nan_mask.sum())
    joined = joined.loc[~weather_nan_mask].reset_index(drop=True)

    # --- Attach provenance -------------------------------------------------
    neso_stamp = _resolve_provenance(
        neso_retrieved_at_utc, demand_hourly, column="retrieved_at_utc"
    )
    weather_stamp = _resolve_provenance(
        weather_retrieved_at_utc, weather_national, column="retrieved_at_utc"
    )
    joined["neso_retrieved_at_utc"] = neso_stamp
    joined["weather_retrieved_at_utc"] = weather_stamp

    # --- Project to OUTPUT_SCHEMA column order -----------------------------
    column_order = [field.name for field in OUTPUT_SCHEMA]
    result = joined[column_order].copy()
    for col in DEMAND_COLUMNS:
        result[col] = result[col].astype("int32")
    for name, _dtype in WEATHER_VARIABLE_COLUMNS:
        result[name] = result[name].astype("float32")

    # --- Single structured INFO log (D5) ----------------------------------
    logger.info(
        "Feature-assembler build: feature_set={} row_count={} "
        "demand_nan_rows_dropped={} weather_forward_filled_rows={} "
        "weather_nan_rows_dropped_after_fill={}",
        config.name,
        len(result),
        demand_nan_rows_dropped,
        dict(weather_forward_filled_rows),
        weather_nan_rows_dropped_after_fill,
    )

    return result


def _validate_build_inputs(
    demand_hourly: pd.DataFrame,
    weather_national: pd.DataFrame,
) -> None:
    """Sanity-check both inputs before the join.

    Defence-in-depth against Gotcha 2 (joining on ``timestamp_local``):
    refuses any frame whose ``timestamp_utc`` is tz-naive or in a non-UTC
    timezone. The downstream ``merge`` would join on value regardless,
    but that join is only safe when both series are in the same UTC
    timeline.
    """
    if "timestamp_utc" not in demand_hourly.columns:
        raise ValueError(
            "demand_hourly must have a 'timestamp_utc' column; "
            f"got {list(demand_hourly.columns)!r}."
        )
    if demand_hourly["timestamp_utc"].dt.tz is None:
        raise ValueError(
            "demand_hourly['timestamp_utc'] is tz-naive. The assembler joins strictly on "
            "tz-aware UTC timestamps — use the canonical 'timestamp_utc' column, never "
            "'timestamp_local' (Plan Gotcha 2)."
        )
    # If a caller hands us a weather frame where the timestamp lives on the
    # index, build() will reset_index it below. We still verify that
    # whichever source holds the timestamp carries UTC tz info.
    if "timestamp_utc" in weather_national.columns:
        tz_series = weather_national["timestamp_utc"]
    elif weather_national.index.name == "timestamp_utc":
        tz_series = weather_national.index.to_series()
    else:
        raise ValueError(
            "weather_national must expose 'timestamp_utc' as a column or index; "
            f"got columns={list(weather_national.columns)}, "
            f"index.name={weather_national.index.name!r}."
        )
    if tz_series.dt.tz is None:
        raise ValueError(
            "weather_national['timestamp_utc'] is tz-naive. national_aggregate preserves the "
            "tz-aware UTC index from the weather ingester — upstream layer has regressed."
        )


def _resolve_provenance(
    override: pd.Timestamp | None,
    source_frame: pd.DataFrame,
    *,
    column: str,
) -> pd.Timestamp:
    """Decide the scalar retrieved_at_utc to write on every row.

    Preference order: explicit argument → column on source frame (first
    non-null; warn if not unique) → ``pd.Timestamp.now("UTC")``. The fallback
    guards against the no-provenance case (e.g. a hand-crafted fixture with
    no ``retrieved_at_utc`` column at all); the assembler does not fail
    hard in that case because a downstream unit test should not need to
    simulate full ingestion provenance.
    """
    if override is not None:
        return pd.Timestamp(override).tz_convert("UTC")

    if column in source_frame.columns:
        non_null = source_frame[column].dropna()
        if len(non_null) == 0:
            return pd.Timestamp.now("UTC").floor("us")
        if non_null.nunique() > 1:
            logger.warning(
                "Provenance column {!r} carries {} distinct values on {} frame; using the first.",
                column,
                non_null.nunique(),
                source_frame.attrs.get("source", "<unknown>"),
            )
        return pd.Timestamp(non_null.iloc[0]).tz_convert("UTC")

    return pd.Timestamp.now("UTC").floor("us")


# ---------------------------------------------------------------------------
# load — schema-validated read (§6 Task T3)
# ---------------------------------------------------------------------------


_REGEN_HINT_WEATHER_ONLY = (
    "Regenerate via: "
    "uv run python -m bristol_ml.features.assembler features=weather_only --cache offline"
)


def load(path: Path) -> pd.DataFrame:
    """Read a feature-table parquet; assert ``OUTPUT_SCHEMA``; return a dataframe.

    Mirrors ``neso.load`` and ``weather.load``: every column in
    ``OUTPUT_SCHEMA`` must be present with the declared arrow type.
    Extra columns trigger a ``ValueError`` — the feature-table contract
    is exact, not permissive, because downstream models may select
    columns positionally for speed.

    Schema-mismatch ``ValueError`` messages name the missing or
    mistyped column **and** the regeneration command, because the most
    common cause of this failure mode in the wild is a schema bump
    (e.g. a new column added to ``derive_calendar``) where the on-disk
    parquet pre-dates the code change.  Naming the recovery command
    in the error itself saves the operator a docs lookup.
    """
    table = pq.read_table(path)
    actual = table.schema

    for field in OUTPUT_SCHEMA:
        if field.name not in actual.names:
            raise ValueError(
                f"Cached parquet at {path} is missing required column "
                f"{field.name!r}.  This usually means the schema changed since "
                f"the cache was written.  {_REGEN_HINT_WEATHER_ONLY}"
            )
        actual_field = actual.field(field.name)
        if actual_field.type != field.type:
            raise ValueError(
                f"Column {field.name!r} in {path} has type {actual_field.type}; "
                f"expected {field.type}.  {_REGEN_HINT_WEATHER_ONLY}"
            )

    expected_names = {field.name for field in OUTPUT_SCHEMA}
    extra = [name for name in actual.names if name not in expected_names]
    if extra:
        raise ValueError(
            f"Cached parquet at {path} has unexpected column(s) {sorted(extra)}; "
            f"the feature-table schema is exact (Plan AC-2).  "
            f"{_REGEN_HINT_WEATHER_ONLY}"
        )

    return table.to_pandas()


# ---------------------------------------------------------------------------
# load_calendar — schema-validated read for the Stage 5 weather_calendar set
# ---------------------------------------------------------------------------


_REGEN_HINT_WEATHER_CALENDAR = (
    "Regenerate via: "
    "uv run python -m bristol_ml.features.assembler features=weather_calendar --cache offline"
)


def load_calendar(path: Path) -> pd.DataFrame:
    """Read a ``weather_calendar`` feature-table parquet; assert ``CALENDAR_OUTPUT_SCHEMA``.

    Mirrors :func:`load` but on the Stage 5 schema. Rejects both
    missing and extra columns: the two feature-table contracts are exact,
    not permissive — a parquet written under :data:`OUTPUT_SCHEMA`
    (weather-only) will be rejected here because its calendar columns are
    absent, and vice versa for a calendar parquet passed to :func:`load`.

    The schema-validated returns carry calendar columns as ``int8`` and
    the ``holidays_retrieved_at_utc`` column as tz-aware UTC.

    Schema-mismatch ``ValueError`` messages include the regeneration
    command so an operator who has just added or removed a calendar
    column gets a copy-pasteable recovery hint without consulting docs.
    """
    table = pq.read_table(path)
    actual = table.schema

    for field in CALENDAR_OUTPUT_SCHEMA:
        if field.name not in actual.names:
            raise ValueError(
                f"Cached parquet at {path} is missing required column "
                f"{field.name!r}.  This usually means the schema changed "
                f"since the cache was written.  {_REGEN_HINT_WEATHER_CALENDAR}"
            )
        actual_field = actual.field(field.name)
        if actual_field.type != field.type:
            raise ValueError(
                f"Column {field.name!r} in {path} has type {actual_field.type}; "
                f"expected {field.type}.  {_REGEN_HINT_WEATHER_CALENDAR}"
            )

    expected_names = {field.name for field in CALENDAR_OUTPUT_SCHEMA}
    extra = [name for name in actual.names if name not in expected_names]
    if extra:
        raise ValueError(
            f"Cached parquet at {path} has unexpected column(s) {sorted(extra)}; "
            f"the weather_calendar feature-table schema is exact (Plan AC-3 / AC-7).  "
            f"{_REGEN_HINT_WEATHER_CALENDAR}"
        )

    return table.to_pandas()


# ---------------------------------------------------------------------------
# assemble — one-shot orchestrator used by the CLI (§6 Task T4)
# ---------------------------------------------------------------------------


def assemble(cfg: AppConfig, cache: str = "offline") -> Path:
    """End-to-end: fetch demand + weather, aggregate, build, atomically write.

    The ``cache`` argument accepts the same three strings as
    ``CachePolicy`` (``"auto"``, ``"refresh"``, ``"offline"``). Default is
    ``"offline"`` — the CI-safe choice; the CLI passes ``"auto"`` for an
    interactive demo.

    Returns the absolute path the feature table was written to. Raises if
    the ingestion-layer configs are not resolved or if either side of the
    join returns no rows after schema assertion.
    """
    from bristol_ml.features.weather import national_aggregate
    from bristol_ml.ingestion import neso, weather
    from bristol_ml.ingestion._common import CachePolicy, _atomic_write, _cache_path

    if cfg.features.weather_only is None:
        raise ValueError(
            "No weather-only feature-set config resolved. Ensure `features: weather_only` "
            "is in `conf/config.yaml` defaults (or use the `features=weather_only` CLI override)."
        )
    if cfg.ingestion.neso is None or cfg.ingestion.weather is None:
        raise ValueError(
            "Both `ingestion.neso` and `ingestion.weather` must be resolved before "
            "the assembler runs."
        )

    policy = CachePolicy(cache)

    neso_path = neso.fetch(cfg.ingestion.neso, cache=policy)
    neso_df = neso.load(neso_path)
    demand_hourly = _resample_demand_hourly(
        neso_df,
        agg=cfg.features.weather_only.demand_aggregation,
    )

    weather_path = weather.fetch(cfg.ingestion.weather, cache=policy)
    weather_df = weather.load(weather_path)
    weights = {s.name: s.weight for s in cfg.ingestion.weather.stations}
    weather_national = national_aggregate(weather_df, weights)

    neso_stamp = (
        pd.Timestamp(neso_df["retrieved_at_utc"].iloc[0]).tz_convert("UTC")
        if "retrieved_at_utc" in neso_df.columns and len(neso_df)
        else None
    )
    weather_stamp = (
        pd.Timestamp(weather_df["retrieved_at_utc"].iloc[0]).tz_convert("UTC")
        if "retrieved_at_utc" in weather_df.columns and len(weather_df)
        else None
    )

    feature_frame = build(
        demand_hourly,
        weather_national,
        cfg.features.weather_only,
        neso_retrieved_at_utc=neso_stamp,
        weather_retrieved_at_utc=weather_stamp,
    )

    table = pa.Table.from_pandas(feature_frame, preserve_index=False).cast(OUTPUT_SCHEMA, safe=True)
    out_path = _cache_path(cfg.features.weather_only)
    _atomic_write(table, out_path)
    logger.info(
        "Feature-assembler cache written: {} rows -> {}",
        len(feature_frame),
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# assemble_calendar — orchestrator for the Stage 5 weather_calendar set
# ---------------------------------------------------------------------------


def assemble_calendar(cfg: AppConfig, *, cache: str | object = "offline") -> Path:
    """End-to-end: build weather-only frame, derive calendar, persist.

    Stage 5 T4 orchestrator — composes Stage 3's weather-only join (demand
    + weather + provenance scalars) with Stage 5's calendar derivation
    (``features.calendar.derive_calendar``) and the bank-holidays ingester
    (``ingestion.holidays``), then writes the 55-column
    :data:`CALENDAR_OUTPUT_SCHEMA` parquet to
    ``cfg.features.weather_calendar.cache_dir / cache_filename``.

    Parameters
    ----------
    cfg
        Resolved :class:`AppConfig`. ``cfg.features.weather_calendar`` must be
        populated (the Hydra group-swap arranges this when the user invokes
        ``features=weather_calendar``); ``cfg.ingestion.neso``,
        ``cfg.ingestion.weather`` and ``cfg.ingestion.holidays`` must all
        be populated.
    cache
        Cache policy passed through to the three ingesters. Accepts either a
        :class:`CachePolicy` value or one of the strings ``"auto"``,
        ``"refresh"``, ``"offline"``. Default is ``"offline"`` — the CI-safe
        choice.

    Returns
    -------
    pathlib.Path
        Absolute path the Stage 5 feature table was written to.

    Notes
    -----
    **Divergence from the plan's literal wording** (logged as a Phase 2
    finding in ``docs/plans/active/05-calendar-features.md`` §"Implementation
    findings"): the plan said this function should "call ``assemble()`` with
    the weather-only config". Under the Stage 5 T1 Hydra group-swap refactor
    only one of ``cfg.features.weather_only`` / ``cfg.features.weather_calendar``
    is populated per run — so when the user invokes ``features=weather_calendar``
    the ``weather_only`` attribute is ``None`` and ``assemble()``'s own guard
    would raise before the calendar layer could compose on top. This
    function instead duplicates the NESO → resample → weather → national-
    aggregate → :func:`build` composition inline, passing
    ``cfg.features.weather_calendar`` as the ``FeatureSetConfig`` to
    :func:`build`. :func:`build` itself is feature-set-agnostic (reads only
    ``config.forward_fill_hours`` and ``config.name``) so the Stage 3
    contract is preserved. A future refactor could factor a private
    ``_compose_weather_only_frame`` helper and share it between both
    orchestrators.
    """
    from bristol_ml.features.calendar import derive_calendar
    from bristol_ml.features.weather import national_aggregate
    from bristol_ml.ingestion import holidays as _holidays_ing
    from bristol_ml.ingestion import neso, weather
    from bristol_ml.ingestion._common import CachePolicy, _atomic_write, _cache_path

    if cfg.features.weather_calendar is None:
        raise ValueError(
            "No weather_calendar feature-set config resolved. Invoke with "
            "`features=weather_calendar` (Hydra group override) or ensure "
            "`conf/features/weather_calendar.yaml` is selected in the defaults list."
        )
    if cfg.ingestion.neso is None or cfg.ingestion.weather is None:
        raise ValueError(
            "Both `ingestion.neso` and `ingestion.weather` must be resolved before "
            "the calendar assembler runs."
        )
    if cfg.ingestion.holidays is None:
        raise ValueError(
            "`ingestion.holidays` must be resolved before the calendar assembler "
            "runs. Ensure `- ingestion/holidays@ingestion.holidays` is in the "
            "`conf/config.yaml` defaults list."
        )

    fset = cfg.features.weather_calendar
    policy = cache if isinstance(cache, CachePolicy) else CachePolicy(cache)

    # --- Weather-only composition (duplicates assemble() — see Notes above) ---
    neso_path = neso.fetch(cfg.ingestion.neso, cache=policy)
    neso_df = neso.load(neso_path)
    demand_hourly = _resample_demand_hourly(
        neso_df,
        agg=fset.demand_aggregation,
    )

    weather_path = weather.fetch(cfg.ingestion.weather, cache=policy)
    weather_df = weather.load(weather_path)
    weights = {s.name: s.weight for s in cfg.ingestion.weather.stations}
    weather_national = national_aggregate(weather_df, weights)

    neso_stamp = (
        pd.Timestamp(neso_df["retrieved_at_utc"].iloc[0]).tz_convert("UTC")
        if "retrieved_at_utc" in neso_df.columns and len(neso_df)
        else None
    )
    weather_stamp = (
        pd.Timestamp(weather_df["retrieved_at_utc"].iloc[0]).tz_convert("UTC")
        if "retrieved_at_utc" in weather_df.columns and len(weather_df)
        else None
    )

    weather_frame = build(
        demand_hourly,
        weather_national,
        fset,
        neso_retrieved_at_utc=neso_stamp,
        weather_retrieved_at_utc=weather_stamp,
    )

    # --- Calendar derivation + holidays provenance scalar ---
    holidays_path = _holidays_ing.fetch(cfg.ingestion.holidays, cache=policy)
    holidays_df = _holidays_ing.load(holidays_path)
    calendar_frame = derive_calendar(weather_frame, holidays_df)

    holidays_stamp = (
        pd.Timestamp(holidays_df["retrieved_at_utc"].iloc[0]).tz_convert("UTC")
        if "retrieved_at_utc" in holidays_df.columns and len(holidays_df)
        else pd.Timestamp.now("UTC").floor("us")
    )
    calendar_frame["holidays_retrieved_at_utc"] = holidays_stamp

    # --- Project to CALENDAR_OUTPUT_SCHEMA column order + write ---
    column_order = [field.name for field in CALENDAR_OUTPUT_SCHEMA]
    missing = [c for c in column_order if c not in calendar_frame.columns]
    if missing:
        raise ValueError(
            f"assemble_calendar: derived frame missing expected columns {missing}. "
            "derive_calendar / build contract has regressed."
        )
    result = calendar_frame[column_order].copy()

    table = pa.Table.from_pandas(result, preserve_index=False).cast(
        CALENDAR_OUTPUT_SCHEMA, safe=True
    )
    out_path = _cache_path(fset)
    _atomic_write(table, out_path)
    logger.info(
        "Feature-assembler (calendar) cache written: {} rows -> {}",
        len(result),
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.features.assembler`
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.features.assembler",
        description=(
            "Assemble an hourly feature-table parquet from the cached upstream "
            "ingestion artefacts.  Dispatches on the active ``features=`` Hydra "
            "group:\n"
            "  features=weather_only     -> data/features/weather_only.parquet      (Stage 3)\n"
            "  features=weather_calendar -> data/features/weather_calendar.parquet  (Stage 5)\n"
            "\n"
            "The default Hydra composition selects ``features=weather_only``, so "
            "running the bare CLI regenerates the Stage 3 cache.  Pass "
            "``features=weather_calendar`` to regenerate the Stage 5 cache.\n"
            "\n"
            "``--cache offline`` (the default) reuses the warm ingester caches "
            "(NESO, weather, holidays) and only re-runs the feature derivation "
            "— the right policy after a code-only change such as adding a "
            "calendar column.  ``--cache auto`` fills missing ingester caches "
            "from the network; ``--cache refresh`` re-fetches them.  The "
            "assembler's own output parquet is always written via atomic "
            "replace, so a re-run never corrupts a partially-written file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        choices=["auto", "refresh", "offline"],
        default="offline",
        help="Cache policy passed through to the ingesters (default: offline).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides, e.g. ``features=weather_calendar`` to swap the "
            "feature-set group, or ``features.weather_only.demand_aggregation=max``."
        ),
    )
    return parser


def _resolve_orchestrator(
    cfg: AppConfig,
) -> tuple[str, object]:
    """Dispatch on the active ``features=`` group.

    Returns the (group-name, orchestrator-callable) pair matching the
    populated field on ``cfg.features``.  Mirrors
    ``bristol_ml.train._resolve_feature_set``'s mutual-exclusivity
    invariant: under the Stage 5 Hydra group-swap refactor exactly one
    of ``cfg.features.weather_only`` / ``cfg.features.weather_calendar``
    is non-``None`` per run, and adding a third sibling (e.g. Stage 16's
    ``with_remit``) is a one-line extension below.

    Returning the orchestrator as ``object`` rather than a Callable type
    keeps this helper free of typing-import gymnastics; the caller
    invokes it positionally with ``(cfg, cache=...)`` per the
    ``assemble`` / ``assemble_calendar`` shared signature.
    """
    populated = [
        name
        for name, value in (
            ("weather_only", cfg.features.weather_only),
            ("weather_calendar", cfg.features.weather_calendar),
        )
        if value is not None
    ]
    if len(populated) != 1:
        raise ValueError(
            "Exactly one of features.weather_only or features.weather_calendar "
            "must be set; pass `features=<name>` at the CLI (e.g. "
            "`features=weather_calendar`).  "
            f"Got populated set(s): {populated or 'none'}."
        )
    name = populated[0]
    if name == "weather_only":
        return ("weather_only", assemble)
    return ("weather_calendar", assemble_calendar)


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local import so `--help` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    try:
        group_name, orchestrator = _resolve_orchestrator(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    logger.info("Feature-assembler CLI: dispatching to features={} orchestrator", group_name)
    try:
        out_path = orchestrator(cfg, cache=args.cache)  # type: ignore[operator]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
