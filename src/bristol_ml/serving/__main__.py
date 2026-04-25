"""Standalone CLI launcher for the serving layer (Stage 12 D13 / NFR-2).

Mirrors the project's ``python -m bristol_ml.train`` argparse pattern:
a thin :func:`_cli_main` that parses CLI flags, resolves the
:class:`~conf._schemas.ServingConfig` via Hydra (so ``conf/serving/default.yaml``
keeps the project's external-config invariant), applies any
explicit CLI flags on top, and finally hands off to
``uvicorn.run`` with a closure-bound app factory.

Two design notes worth pinning here:

1.  **``--help`` must stay lightweight.**  The project's
    ``test_serving_module_imports_without_torch`` guard asserts that
    ``import bristol_ml.serving`` does not pull torch into
    ``sys.modules``; the CLI ``--help`` path goes through this module,
    so heavy imports (``uvicorn``, Hydra) are deferred to the body of
    :func:`_cli_main` after argparse has decided whether to show help
    and exit.  The argparse parser itself only depends on
    :class:`ServingConfig` — a Pydantic model with no torch / hydra /
    uvicorn at import time.

2.  **The argparse defaults restate the schema.**  Plan §1 NFR-2
    requires that ``--help`` "prints the resolved ``ServingConfig``
    schema".  We honour this by binding each flag's ``default=`` to the
    corresponding ``ServingConfig()`` field and using
    :class:`argparse.ArgumentDefaultsHelpFormatter` so the help text
    surfaces the resolved values inline (e.g.
    ``--registry-dir DIR (default: data/registry)``).  A Hydra-style
    ``overrides`` positional supports tweaks like
    ``+serving=default serving.port=8080`` without baking them into a
    flag surface that would have to track future schema growth.

DESIGN §2.1.1 binding: every module runs standalone via
``python -m bristol_ml.<module>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from conf._schemas import ServingConfig

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from collections.abc import Iterable


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``python -m bristol_ml.serving``.

    Each ``--flag``'s ``default=`` is the corresponding
    :class:`ServingConfig` field's default, so ``--help`` surfaces the
    resolved schema (plan §1 NFR-2).  The parser depends on
    :class:`ServingConfig` only — no torch / hydra / uvicorn at parse
    time, which keeps the import-graph guard
    (``test_serving_module_imports_without_torch``) honest.
    """
    defaults = ServingConfig()
    parser = argparse.ArgumentParser(
        prog="bristol_ml.serving",
        description=(
            "Start the bristol_ml prediction service. Loads the lowest-MAE "
            "registered run as the default model at startup; lazy-loads "
            "additional run_ids on demand. Localhost-only; intent §Out of "
            "scope: deployment anywhere other than localhost."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry-dir",
        dest="registry_dir",
        type=Path,
        default=defaults.registry_dir,
        help="Path to the Stage 9 registry root.",
    )
    parser.add_argument(
        "--host",
        default=defaults.host,
        help="Host interface uvicorn binds to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=defaults.port,
        help="TCP port uvicorn listens on.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help=(
            "Hydra-style config overrides (forwarded to load_config), e.g. "
            "+serving=default serving.port=8080."
        ),
    )
    return parser


def _resolve_serving_config(args: argparse.Namespace) -> ServingConfig:
    """Resolve the runtime :class:`ServingConfig` from CLI args + Hydra.

    Order of precedence (lowest → highest):

    1.  :class:`ServingConfig` field defaults.
    2.  ``conf/serving/default.yaml`` if the user includes
        ``+serving=default`` in ``args.overrides`` (Hydra group convention).
    3.  Any ``serving.*`` override the user passed in
        ``args.overrides`` (Hydra dotted-path syntax).
    4.  Explicit CLI flags ``--registry-dir`` / ``--host`` / ``--port``.
        These always win — they are the ergonomic surface the demo
        facilitator reaches for.

    Hydra is imported lazily so ``--help`` never pays the resolve cost.
    """
    # Lazy import: keeps `python -m bristol_ml.serving --help` cheap.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    base = cfg.serving if cfg.serving is not None else ServingConfig()

    # Step 4: explicit flags override.  We always copy with the parsed
    # values; if the user did not pass the flag, argparse populated it
    # from the ServingConfig default, so the round-trip is a no-op
    # (and the resulting model is still frozen, since model_copy
    # returns a fresh frozen instance).
    return base.model_copy(
        update={
            "registry_dir": args.registry_dir,
            "host": args.host,
            "port": args.port,
        }
    )


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the serving CLI.

    Parameters
    ----------
    argv:
        Optional override for ``sys.argv[1:]``.  Passing an explicit
        list lets tests drive the CLI in-process or via subprocess; the
        production launcher (``if __name__ == "__main__"``) passes
        ``None`` so argparse reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code.  ``0`` on a clean uvicorn shutdown.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    serving_cfg = _resolve_serving_config(args)

    # Lazy uvicorn import: keeps ``--help`` lightweight (plan §1 NFR-2)
    # and keeps the import-graph guard honest.
    import uvicorn

    from bristol_ml.serving.app import build_app

    # The factory closes over the resolved registry_dir so uvicorn can
    # invoke it without positional args at the ASGI boundary (plan §5).
    # ``factory=True`` ensures the lifespan re-runs per worker (uvicorn
    # constructs a fresh app per process); the project doesn't enable
    # multi-worker today, but the factory shape costs nothing and
    # keeps the door open.
    def _factory() -> object:
        return build_app(serving_cfg.registry_dir)

    uvicorn.run(
        _factory,
        factory=True,
        host=serving_cfg.host,
        port=serving_cfg.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    sys.exit(_cli_main())
