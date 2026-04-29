"""Stage 15 — standalone-module entry point.

Plan §1 A4 + §6 T8: ``python -m bristol_ml.embeddings`` resolves the
active :class:`~conf._schemas.EmbeddingConfig` via Hydra (composing
``+embedding=default`` automatically), prints a deterministic summary,
and exits 0 in stub mode. No subcommands — the demo flow lives in the
notebook (T9).

Behaviour, in order:

1. Load the active :class:`~conf._schemas.EmbeddingConfig`.
2. Build the embedder via :func:`bristol_ml.embeddings.build_embedder`
   (honouring the ``BRISTOL_ML_EMBEDDING_STUB`` triple-gate).
3. Print active config (model_id, dim, vector_backend, projection_type).
4. Run a one-shot embed of a fixed sample text + a synthetic 5-row
   index, then print the top-3 nearest neighbours.

Mirrors :func:`bristol_ml.llm.extractor._cli_main`'s shape and exit
codes (0 success, 2 config / build error). The output is byte-
deterministic in stub mode so the smoke test
(``tests/unit/embeddings/test_module_runs_standalone.py``) can pin
expected lines.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

import numpy as np
from pydantic import ValidationError

from bristol_ml.config import load_config
from bristol_ml.embeddings._factory import (
    MODEL_PATH_ENV_VAR,
    STUB_ENV_VAR,
    build_embedder,
    build_index,
)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the ``python -m bristol_ml.embeddings`` parser."""
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.embeddings",
        description=(
            "Print the active embedding config, build the embedder + index, "
            "run a sample query, and exit. Hydra-style overrides accepted; "
            "by default the offline stub path is selected. Composes the "
            "+embedding=default config group automatically."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides applied on top of conf/config.yaml + "
            "+embedding=default (e.g. embedding.type=sentence_transformers)."
        ),
    )
    return parser


# Five short, distinct REMIT-flavoured texts. Hand-picked rather than
# loaded from a fixture so the standalone CLI does not touch the file
# system beyond the YAML config — keeps the smoke test trivially
# offline.
_SAMPLE_DOC_TEXTS: list[str] = [
    "Outage Planned Nuclear T_HARTLEPOOL-1",
    "Restriction Forced Gas T_PEMBROKE-1",
    "Outage Unplanned Coal RATCLIFFE-1",
    "Outage Planned Wind GORDONBUSH-1",
    "Outage Planned Solar SOLARFARM-2",
]
_SAMPLE_DOC_IDS: list[str] = [
    "M-DEMO-NUC::0",
    "M-DEMO-GAS::0",
    "M-DEMO-COAL::0",
    "M-DEMO-WIND::0",
    "M-DEMO-SOLAR::0",
]
_SAMPLE_QUERY_TEXT: str = "planned nuclear outage"


def _format_config_block(
    *,
    implementation: str,
    model_id: str,
    dim: int,
    vector_backend: str,
    projection_type: str,
    stub_env_set: bool,
) -> list[str]:
    """Lines describing the resolved config + dispatch state."""
    return [
        "=== Stage 15 embedding index ===",
        f"implementation:     {implementation}",
        f"model_id:           {model_id}",
        f"dim:                {dim}",
        f"vector_backend:     {vector_backend}",
        f"projection_type:    {projection_type}",
        f"{STUB_ENV_VAR}: {'1' if stub_env_set else '0'}",
    ]


def _format_neighbour_block(
    neighbours: list[tuple[str, float]],
    *,
    query_text: str,
) -> list[str]:
    """Lines for the sample top-k block; deterministic in stub mode."""
    lines = [
        "sample query:",
        f"  query_text:       {query_text}",
        "  top neighbours:",
    ]
    for rank, (nid, score) in enumerate(neighbours, start=1):
        lines.append(f"    {rank}. id={nid:<24} score={score:+.4f}")
    lines.append("=== end ===")
    return lines


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry — DESIGN §2.1.1, plan §1 A4 / §6 T8.

    Returns 0 on success, 2 on a config / build error.
    """
    import os

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = ["+embedding=default", *args.overrides]
    try:
        cfg = load_config(overrides=overrides)
    except (ValidationError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if cfg.embedding is None:  # pragma: no cover — overrides force this group
        print(
            "Embedding config is None after composition; "
            "this should not happen given +embedding=default.",
            file=sys.stderr,
        )
        return 2

    try:
        embedder = build_embedder(cfg.embedding)
    except (RuntimeError, ValueError) as exc:
        print(f"Embedder build error: {exc}", file=sys.stderr)
        return 2

    implementation = type(embedder).__name__
    stub_env_set = os.environ.get(STUB_ENV_VAR) == "1"

    config_lines = _format_config_block(
        implementation=implementation,
        model_id=embedder.model_id,
        dim=embedder.dim,
        vector_backend=cfg.embedding.vector_backend,
        projection_type=cfg.embedding.projection_type,
        stub_env_set=stub_env_set,
    )

    # Build a tiny in-memory index from the synthetic docs and query.
    try:
        doc_vectors = embedder.embed_batch(_SAMPLE_DOC_TEXTS)
        query_vector = embedder.embed(_SAMPLE_QUERY_TEXT)
    except Exception as exc:  # pragma: no cover — exercised by the unit test
        print(f"Sample-embed error: {exc}", file=sys.stderr)
        return 2

    index = build_index(cfg.embedding, dim=embedder.dim)
    index.add(_SAMPLE_DOC_IDS, np.asarray(doc_vectors, dtype=np.float32))
    neighbours = index.query(np.asarray(query_vector, dtype=np.float32), k=3)

    lines = config_lines + _format_neighbour_block(
        [(nn.id, nn.score) for nn in neighbours],
        query_text=_SAMPLE_QUERY_TEXT,
    )
    print("\n".join(lines))
    return 0


# Silence unused-import lint warning when type-checkers don't pick up
# MODEL_PATH_ENV_VAR's role as part of the public env-var contract;
# the standalone CLI honours it via :func:`build_embedder`.
_ = MODEL_PATH_ENV_VAR


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(_cli_main())
