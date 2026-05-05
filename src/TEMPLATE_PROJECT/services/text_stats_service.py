"""Text statistics service — the worked-example services module.

Demonstrates the template's ``services/`` layer pattern: read a
config-pointed file, call a pure :mod:`TEMPLATE_PROJECT.core`
function, render the result to stdout.

Run via the unified CLI::

    uv run python -m TEMPLATE_PROJECT services.text_stats.input_path=path/to/file.txt

The service module also exposes a ``run`` function for tests and
notebooks that want to drive it programmatically without re-routing
through Hydra.
"""

from __future__ import annotations

import sys

from loguru import logger

from conf._schemas import TextStatsConfig
from TEMPLATE_PROJECT.config import load_config
from TEMPLATE_PROJECT.core.text_stats import compute_text_statistics

__all__ = ["render", "run"]


def run(config: TextStatsConfig) -> str:
    """Compute text statistics for ``config.input_path`` and return them.

    Returns the rendered string (JSON or human-readable per
    ``config.output_format``); the caller decides what to do with it
    (stdout, file write, GUI, etc.).
    """
    text = config.input_path.read_text(encoding="utf-8")
    stats = compute_text_statistics(text)
    logger.info(
        "text_stats: input_path={} character_count={} word_count={} line_count={}",
        config.input_path,
        stats.character_count,
        stats.word_count,
        stats.line_count,
    )
    return render(stats, output_format=config.output_format)


def render(
    stats: object,
    *,
    output_format: str = "json",
) -> str:
    """Format :class:`TextStatistics` for display."""
    if output_format == "json":
        # ``model_dump_json`` is on the BaseModel; ``mypy`` infers
        # via the parameter type at the call site.
        return stats.model_dump_json(indent=2)  # type: ignore[attr-defined]
    if output_format == "human":
        # Tiny aligned table; one row per metric.
        rows = stats.model_dump()  # type: ignore[attr-defined]
        width = max(len(k) for k in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows.items())
    raise ValueError(f"Unknown output_format {output_format!r}; expected 'json' or 'human'.")


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Resolves the Hydra config (with any ``argv`` overrides), validates,
    and runs the service.  Returns 0 on success, 2 on configuration
    errors.
    """
    overrides = list(argv) if argv is not None else []
    cfg = load_config(overrides=overrides)
    if cfg.services.text_stats is None:
        print(
            "No TextStatsConfig resolved; ensure `services: text_stats` is in "
            "the defaults list of conf/config.yaml.",
            file=sys.stderr,
        )
        return 2
    print(run(cfg.services.text_stats))
    return 0


if __name__ == "__main__":  # pragma: no cover — module-level CLI wrapper
    raise SystemExit(main(sys.argv[1:]))
