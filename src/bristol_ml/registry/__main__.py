"""CLI entry for the registry — ``python -m bristol_ml.registry``.

Two subcommands satisfy the Stage 9 Demo moment (intent):

- ``list`` prints a leaderboard of every registered run, filterable by
  ``--target``, ``--model-type``, and ``--feature-set`` (D7).  Default
  sort is MAE-ascending (D8).
- ``describe`` pretty-prints a single run's sidecar JSON (AC-1
  "maybe describe").

Both subcommands accept a ``--registry-dir`` override so demos and tests
can point at a throwaway directory; omitting it uses
:data:`bristol_ml.registry.DEFAULT_REGISTRY_DIR`.

Implementation idiom mirrors ``bristol_ml.train`` — ``argparse`` with
subparsers, no Hydra config group (plan D17 cut — the registry has no
``RegistryConfig``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from bristol_ml import registry

__all__ = ["_cli_main"]


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the two-subcommand parser (``list`` / ``describe``)."""
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.registry",
        description=(
            "Inspect the filesystem-backed model registry.  `list` prints "
            "a leaderboard; `describe` prints one sidecar.  Neither "
            "subcommand writes to disk — registration happens via "
            "`python -m bristol_ml.train`."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # list
    list_parser = subparsers.add_parser(
        "list",
        help="Print a leaderboard of every registered run.",
        description=(
            "Print every registered run under --registry-dir as a table "
            "sorted by --sort-by (default: mae, ascending).  Optional "
            "exact-match filters: --target, --model-type, --feature-set."
        ),
    )
    list_parser.add_argument("--registry-dir", type=Path, default=None)
    list_parser.add_argument("--target", default=None)
    list_parser.add_argument("--model-type", default=None)
    list_parser.add_argument("--feature-set", default=None)
    list_parser.add_argument(
        "--sort-by",
        default="mae",
        help="Metric name to sort by (default: mae).",
    )
    list_parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort largest-to-smallest rather than the default smallest-to-largest.",
    )

    # describe
    describe_parser = subparsers.add_parser(
        "describe",
        help="Pretty-print one registered run's sidecar JSON.",
    )
    describe_parser.add_argument("run_id")
    describe_parser.add_argument("--registry-dir", type=Path, default=None)

    return parser


def _format_leaderboard(runs: list[dict]) -> str:
    """Format a list of sidecar dicts as a plain-text leaderboard table.

    Columns: run_id, type, feature_set, target, mae (mean), rmse (mean).
    Missing metrics render as ``—``.  The table is plain ``str.format``
    padding — no tabulate dependency.
    """
    if not runs:
        return "(no registered runs)"
    headers = ("run_id", "type", "feature_set", "target", "mae", "rmse")

    def _cell(run: dict, key: str) -> str:
        if key in ("mae", "rmse"):
            summary = run.get("metrics", {}).get(key)
            if summary is None or "mean" not in summary:
                return "—"
            mean = summary["mean"]
            return f"{mean:.3f}" if mean == mean else "—"  # NaN check
        return str(run.get(key, ""))

    rows = [[_cell(r, h) for h in headers] for r in runs]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1."""
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.subcommand == "list":
        runs = registry.list_runs(
            target=args.target,
            model_type=args.model_type,
            feature_set=args.feature_set,
            sort_by=args.sort_by,
            ascending=not args.descending,
            registry_dir=args.registry_dir,
        )
        print(_format_leaderboard(runs))
        return 0

    if args.subcommand == "describe":
        try:
            sidecar = registry.describe(args.run_id, registry_dir=args.registry_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(sidecar, indent=2, allow_nan=True, ensure_ascii=False))
        return 0

    return 2  # pragma: no cover — argparse enforces required subcommand


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(_cli_main())
