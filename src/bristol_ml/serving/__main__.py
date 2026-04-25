"""Standalone CLI launcher for the serving layer (Stage 12 D13 / NFR-2).

Mirrors the project's ``python -m bristol_ml.train`` argparse pattern.
The full CLI wiring (resolve ``ServingConfig`` via ``load_config``,
launch ``uvicorn.run`` with the app factory) lands at Stage 12 T8.
At T1 (this file) only the scaffold is in place: ``--help`` exits
zero so ``python -m bristol_ml.serving --help`` does not error during
the partial-implementation window.

DESIGN §2.1.1 binding: every module runs standalone via
``python -m bristol_ml.<module>``.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``python -m bristol_ml.serving``.

    The flag surface (``--registry-dir``, ``--host``, ``--port``,
    Hydra-style overrides) is documented at Stage 12 plan §5; the
    full implementation lands at T8.
    """
    parser = argparse.ArgumentParser(
        prog="bristol_ml.serving",
        description=(
            "Start the bristol_ml prediction service. Loads the lowest-MAE "
            "registered run as the default model at startup; lazy-loads "
            "additional run_ids on demand. Localhost-only; intent §Out of "
            "scope: deployment anywhere other than localhost."
        ),
    )
    parser.add_argument(
        "--registry-dir",
        dest="registry_dir",
        default=None,
        help="Override conf/serving.yaml registry_dir (default: data/registry).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override conf/serving.yaml host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override conf/serving.yaml port (default: 8000).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides (forwarded to load_config).",
    )
    return parser


def _cli_main(argv: list[str] | None = None) -> int:
    """Entry point for the serving CLI (Stage 12 T1 scaffold; T8 fills in)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Stage 12 T8 will wire up uvicorn here.  The T1 scaffold simply
    # confirms the CLI surface parses without error so AC NFR-2 has a
    # working ``--help`` path during the partial-implementation window.
    print(
        "bristol_ml.serving CLI parsed args:",
        {
            "registry_dir": args.registry_dir,
            "host": args.host,
            "port": args.port,
            "overrides": args.overrides,
        },
    )
    print("(Stage 12 T1 scaffold: full uvicorn launch lands at T8.)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
