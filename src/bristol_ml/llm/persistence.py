"""Stage 16 T3 — persist LLM extractor output as a parquet sidecar.

The Stage 14 :class:`bristol_ml.llm.Extractor` Protocol returns
:class:`bristol_ml.llm.ExtractionResult` instances **in memory**; there
is no on-disk persistence layer (codebase-explorer hazard H1).  Stage 16
needs the extractor's output as a stable artefact that the feature
assembler reads cheaply on every retraining run, so this module adds the
missing persistence step.

Architectural shape (Stage 16 plan A5 — separate ingestion-style step):

- One module under :mod:`bristol_ml.llm` that owns the
  ``ExtractionResult`` parquet schema and the write/read helpers.  The
  feature assembler depends on this module's public surface only; it
  does not import :class:`Extractor` directly.
- The persistence parquet is keyed on ``(mrid, revision_number)`` —
  the same primary key as Stage 13's REMIT parquet — so the assembler
  joins by that key without needing extra row identifiers.
- Atomic writes via :func:`bristol_ml.ingestion._common._atomic_write`;
  every parquet is either fully present or fully absent (NFR-3).
- Provenance: an ``extracted_at_utc`` scalar provenance column is
  written on every row, mirroring the ingestion layer's
  ``retrieved_at_utc`` convention.

Public surface::

    EXTRACTED_OUTPUT_SCHEMA  — pyarrow schema (11 columns)
    extract_and_persist(extractor, remit_df, *, output_path) -> Path
    load_extracted(path) -> pd.DataFrame

Run standalone::

    python -m bristol_ml.llm.persistence [--cache {auto,refresh,offline}] [--limit N]

The CLI ties ``remit.fetch + remit.load -> build_extractor(cfg.llm) ->
extract_and_persist`` and prints the output path.  Stub-default; live
path activated when ``llm.type=openai`` and the API-key env var is
populated (per :mod:`bristol_ml.llm.extractor`'s triple-gating rules).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from bristol_ml.ingestion._common import _atomic_write
from bristol_ml.llm import Extractor, RemitEvent

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "EXTRACTED_OUTPUT_SCHEMA",
    "extract_and_persist",
    "load_extracted",
]


# ---------------------------------------------------------------------------
# Schema — one row per (mrid, revision_number); mirrors ExtractionResult plus
# the (mrid, revision_number) join key and an extracted_at_utc provenance
# scalar.
# ---------------------------------------------------------------------------


EXTRACTED_OUTPUT_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("mrid", pa.string(), nullable=False),
        pa.field("revision_number", pa.int32(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("fuel_type", pa.string(), nullable=False),
        pa.field("affected_capacity_mw", pa.float64(), nullable=True),
        pa.field("effective_from", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("effective_to", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("prompt_hash", pa.string(), nullable=True),
        pa.field("model_id", pa.string(), nullable=True),
        pa.field("extracted_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
"""pyarrow schema for the persisted ExtractionResult parquet (11 columns).

Column order is contractual; the assembler joins on
``(mrid, revision_number)`` and reads ``affected_capacity_mw``,
``event_type``, ``fuel_type``, ``confidence`` from the named columns.
The ``effective_from`` / ``effective_to`` columns are duplicated from
:class:`bristol_ml.llm.ExtractionResult` so a downstream consumer can
verify the LLM's interpretation against the upstream REMIT row's
valid-time fields.
"""


# Default location under the project's ``data/processed/`` tree —
# parallel to the ``data/raw/`` (ingestion-layer) and ``data/features/``
# (assembler-layer) directories.  The file name omits the
# ``stub_or_real`` distinction because the run's provenance lives in
# the ``model_id`` and ``prompt_hash`` columns.
DEFAULT_OUTPUT_PATH: Final[Path] = Path("data/processed/remit_extracted.parquet")
"""Repo-relative default path for the persisted extractor parquet.

Resolved against the configured ``BRISTOL_ML_CACHE_DIR`` (or the project
root when unset) at the call site of :func:`extract_and_persist`.
"""


# ---------------------------------------------------------------------------
# Extract + persist
# ---------------------------------------------------------------------------


def extract_and_persist(
    extractor: Extractor,
    remit_df: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    """Run ``extractor.extract_batch`` over ``remit_df`` and persist as parquet.

    For each row of ``remit_df`` the function constructs a
    :class:`bristol_ml.llm.RemitEvent`, calls
    ``extractor.extract_batch(...)`` once for the whole frame, joins the
    results back onto ``(mrid, revision_number)``, stamps an
    ``extracted_at_utc`` provenance scalar (constant across the run),
    casts to :data:`EXTRACTED_OUTPUT_SCHEMA`, and writes the result via
    :func:`bristol_ml.ingestion._common._atomic_write` (NFR-3 — partial
    writes leave the previous file intact).

    Parameters
    ----------
    extractor:
        Any :class:`bristol_ml.llm.Extractor` implementation.  In stub
        mode this is a :class:`StubExtractor`; in live mode an
        :class:`LlmExtractor` (the factory's triple-gating decides).
    remit_df:
        A REMIT event log conforming to
        :data:`bristol_ml.ingestion.remit.OUTPUT_SCHEMA`.  All revisions
        are extracted (the bi-temporal as-of restriction lives at
        feature-derivation time, not extraction time).
    output_path:
        Absolute or repo-relative target path for the parquet.  Parent
        directory is created if absent.

    Returns
    -------
    pathlib.Path
        The absolute path the parquet was written to.

    Raises
    ------
    ValueError
        If ``remit_df`` is missing any column required to construct a
        :class:`RemitEvent`.
    """
    required = (
        "mrid",
        "revision_number",
        "message_status",
        "published_at",
        "effective_from",
        "effective_to",
        "fuel_type",
        "affected_mw",
        "event_type",
        "cause",
        "message_description",
    )
    missing = [c for c in required if c not in remit_df.columns]
    if missing:
        raise ValueError(
            f"extract_and_persist: REMIT frame missing required column(s) "
            f"{missing}; pass the unmodified output of "
            "bristol_ml.ingestion.remit.load()."
        )

    extracted_at = datetime.now(UTC).replace(microsecond=0)

    if remit_df.empty:
        # Empty corpus -> empty parquet with the correct schema.  Useful
        # for CI dry-runs and the first build where no REMIT data exists.
        empty = pa.Table.from_pylist([], schema=EXTRACTED_OUTPUT_SCHEMA)
        _atomic_write(empty, output_path)
        logger.info(
            "Extractor persistence: empty REMIT corpus; wrote zero-row parquet to {}",
            output_path,
        )
        return output_path

    events: list[RemitEvent] = [_row_to_event(row) for _, row in remit_df.iterrows()]
    results = extractor.extract_batch(events)
    if len(results) != len(events):
        # Defensive: extract_batch protocol mandates "returned in input
        # order" — a length mismatch is a contract violation.
        raise RuntimeError(
            f"Extractor.extract_batch returned {len(results)} result(s) "
            f"for {len(events)} input(s); the Protocol mandates input-order "
            "preservation.  See bristol_ml.llm.Extractor."
        )

    # Build the wide frame keyed on (mrid, revision_number).  Use
    # parallel arrays sourced from the input frame so the join key is
    # exact (no float-string round-trip).
    frame = pd.DataFrame(
        {
            "mrid": remit_df["mrid"].astype("string").to_numpy(),
            "revision_number": remit_df["revision_number"].astype("int32").to_numpy(),
            "event_type": pd.Series([r.event_type for r in results], dtype="string"),
            "fuel_type": pd.Series([r.fuel_type for r in results], dtype="string"),
            "affected_capacity_mw": pd.Series(
                [r.affected_capacity_mw for r in results], dtype="float64"
            ),
            "effective_from": pd.to_datetime([r.effective_from for r in results], utc=True),
            "effective_to": pd.to_datetime([r.effective_to for r in results], utc=True),
            "confidence": pd.Series([r.confidence for r in results], dtype="float32"),
            "prompt_hash": pd.Series([r.prompt_hash for r in results], dtype="string"),
            "model_id": pd.Series([r.model_id for r in results], dtype="string"),
            "extracted_at_utc": pd.to_datetime([extracted_at] * len(results), utc=True),
        }
    )

    table = pa.Table.from_pandas(frame, preserve_index=False).cast(
        EXTRACTED_OUTPUT_SCHEMA, safe=True
    )
    _atomic_write(table, output_path)
    logger.info(
        "Extractor persistence: {} extraction(s) written to {} "
        "(extractor={}, mean_confidence={:.3f})",
        len(results),
        output_path,
        type(extractor).__name__,
        float(frame["confidence"].mean()) if len(frame) else 0.0,
    )
    return output_path


def load_extracted(path: Path) -> pd.DataFrame:
    """Read the persisted extractor parquet; assert :data:`EXTRACTED_OUTPUT_SCHEMA`.

    Mirrors :func:`bristol_ml.features.assembler.load`: every column in
    :data:`EXTRACTED_OUTPUT_SCHEMA` must be present with the declared
    arrow type; extra columns trigger ``ValueError`` (the schema is
    exact, not permissive — downstream code joins on
    ``(mrid, revision_number)`` and reads named columns).
    """
    table = pq.read_table(path)
    actual = table.schema
    for field in EXTRACTED_OUTPUT_SCHEMA:
        if field.name not in actual.names:
            raise ValueError(
                f"Extracted-features parquet at {path} is missing required column {field.name!r}"
            )
        actual_field = actual.field(field.name)
        if actual_field.type != field.type:
            raise ValueError(
                f"Column {field.name!r} in {path} has type {actual_field.type}; "
                f"expected {field.type}"
            )
    expected_names = {field.name for field in EXTRACTED_OUTPUT_SCHEMA}
    extra = [name for name in actual.names if name not in expected_names]
    if extra:
        raise ValueError(
            f"Extracted-features parquet at {path} has unexpected column(s) "
            f"{sorted(extra)}; the schema is exact (Stage 16 T3)."
        )
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_event(row: pd.Series) -> RemitEvent:
    """Construct a :class:`RemitEvent` from one REMIT-log row.

    Tz-naive timestamps would fail the :class:`RemitEvent` validator —
    the REMIT parquet emits tz-aware UTC by contract, so this is a
    pure boundary-shape helper rather than an arithmetic guard.
    """
    return RemitEvent(
        mrid=str(row["mrid"]),
        revision_number=int(row["revision_number"]),
        message_status=str(row["message_status"]),
        published_at=row["published_at"],
        effective_from=row["effective_from"],
        effective_to=(row["effective_to"] if pd.notna(row["effective_to"]) else None),
        fuel_type=(str(row["fuel_type"]) if pd.notna(row["fuel_type"]) else None),
        affected_mw=(float(row["affected_mw"]) if pd.notna(row["affected_mw"]) else None),
        event_type=(str(row["event_type"]) if pd.notna(row["event_type"]) else None),
        cause=str(row["cause"]) if pd.notna(row["cause"]) else None,
        message_description=(
            str(row["message_description"]) if pd.notna(row["message_description"]) else None
        ),
    )


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.llm.persistence`  (DESIGN §2.1.1)
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.llm.persistence",
        description=(
            "Run the configured LLM extractor over the cached REMIT log "
            "and persist the results as a parquet sidecar at "
            "data/processed/remit_extracted.parquet (or the configured "
            "override).  Stub-default; activate the live OpenAI path "
            "via llm.type=openai and a populated BRISTOL_ML_LLM_API_KEY."
        ),
    )
    parser.add_argument(
        "--cache",
        choices=["auto", "refresh", "offline"],
        default="offline",
        help="Cache policy passed through to the REMIT ingester (default: offline).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Cap the number of REMIT rows fed to the extractor (default 0 "
            "= no cap).  Use a small value to dry-run the live path against "
            "a handful of events before paying for a full pass."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Override output path (default {DEFAULT_OUTPUT_PATH} relative "
            "to BRISTOL_ML_CACHE_DIR or repo root)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides applied on top of conf/config.yaml + +llm=extractor "
            "(e.g. llm.type=openai or llm.model_name=<provider-model-id>)."
        ),
    )
    return parser


def _resolve_output_path(override: Path | None) -> Path:
    """Resolve the output parquet path against the configured cache root.

    Priority: explicit ``--output`` > project default
    (``data/processed/remit_extracted.parquet``).  The path is returned
    as absolute so the atomic-write helper has an unambiguous target.
    """
    if override is not None:
        return override.resolve() if not override.is_absolute() else override
    # The project's other layers honour BRISTOL_ML_CACHE_DIR for ingestion
    # caches, but the ``processed/`` tree is a sibling of ``raw/`` /
    # ``features/`` rather than a member of either, so we use a single
    # repo-relative default and let the human pass --output to redirect.
    return DEFAULT_OUTPUT_PATH.resolve()


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local imports keep ``--help`` lightweight (Hydra + the OpenAI SDK
    # are comparatively heavy).
    from bristol_ml.config import load_config
    from bristol_ml.ingestion import remit as remit_ingest
    from bristol_ml.ingestion._common import CachePolicy
    from bristol_ml.llm.extractor import build_extractor

    overrides = ["+llm=extractor", *args.overrides]
    cfg = load_config(overrides=overrides)
    if cfg.ingestion.remit is None:
        print("ingestion.remit must be resolved before extraction.", file=sys.stderr)
        return 2

    policy = CachePolicy(args.cache)
    try:
        remit_path = remit_ingest.fetch(cfg.ingestion.remit, cache=policy)
    except FileNotFoundError as exc:
        print(f"REMIT cache missing: {exc}", file=sys.stderr)
        return 2
    remit_df = remit_ingest.load(remit_path)
    if args.limit > 0:
        remit_df = remit_df.head(args.limit).copy()

    try:
        extractor = build_extractor(cfg.llm)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Extractor build error: {exc}", file=sys.stderr)
        return 2

    output_path = _resolve_output_path(args.output)
    written = extract_and_persist(extractor, remit_df, output_path=output_path)
    print(written)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
