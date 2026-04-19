"""Spec-derived CLI tests for ``bristol_ml.evaluation.metrics``.

Covers plan T5 tests 17-19: the standalone CLI entry point required by
DESIGN §2.1.1 (every module runs standalone via ``python -m <module>``).

Tests use ``subprocess.run`` to exercise the real process boundary, which
guards against import-time errors that would be invisible in the same
process.

Conventions
-----------
- British English in docstrings.
- Each docstring cites the plan clause it guards.
- ``subprocess.run`` is used with ``check=False`` to allow explicit
  return-code assertions rather than relying on exception propagation.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Confirm the module is importable before exercising the CLI; skip the
# entire file if the module is absent rather than producing misleading
# failures about subprocess invocations.
pytest.importorskip("bristol_ml.evaluation.metrics")


# ---------------------------------------------------------------------------
# Test 17 — --help exits zero and prints usage (Plan T5 / DESIGN §2.1.1)
# ---------------------------------------------------------------------------


def test_metrics_cli_help_exits_zero() -> None:
    """Guards plan T5 test 17 / DESIGN §2.1.1: standalone CLI --help.

    ``python -m bristol_ml.evaluation.metrics --help`` must:
    - exit with code 0 (help is not an error).
    - write at least the word "usage:" to stdout (argparse convention).
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.evaluation.metrics", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI --help must exit 0; returncode={result.returncode!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "usage:" in result.stdout.lower(), (
        f"CLI --help stdout must contain 'usage:'; got stdout={result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# Test 18 — Default invocation prints registered and selected metrics
# ---------------------------------------------------------------------------


def test_metrics_cli_prints_registered_and_selected() -> None:
    """Guards plan T5 test 18: default invocation lists registered and selected metrics.

    With no override arguments the CLI reads ``conf/evaluation/metrics.yaml``
    which selects all four metrics.  stdout must contain:
    - ``registered: mae, mape, rmse, wape`` (sorted alphabetically).
    - ``selected:   mae, mape, rmse, wape`` (config-driven; all four by default).

    This test validates that the CLI integrates the resolved config, not
    just the hard-coded registry.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.evaluation.metrics"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Default CLI invocation must exit 0; returncode={result.returncode!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "registered:" in result.stdout, (
        f"stdout must contain 'registered:' line; got {result.stdout!r}"
    )
    # Registered line must list all four names (sorted).
    registered_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("registered:")),
        None,
    )
    assert registered_line is not None, "stdout must include a line beginning with 'registered:'."
    for name in ("mae", "mape", "rmse", "wape"):
        assert name in registered_line, (
            f"'registered:' line must mention '{name}'; got {registered_line!r}"
        )

    # Selected line must include all four when the default config is active.
    selected_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("selected:")),
        None,
    )
    assert selected_line is not None, "stdout must include a line beginning with 'selected:'."
    for name in ("mae", "mape", "rmse", "wape"):
        assert name in selected_line, (
            f"'selected:' line must mention '{name}' under default config; got {selected_line!r}"
        )


# ---------------------------------------------------------------------------
# Test 19 — Override restricts selected metrics (Plan T5)
# ---------------------------------------------------------------------------


def test_metrics_cli_respects_override() -> None:
    """Guards plan T5 test 19: ``evaluation.metrics.names=[mae,rmse]`` override.

    Passing a Hydra override via positional args to the CLI must restrict the
    ``selected:`` output to the named subset.  The ``registered:`` line must
    still list all four (the registry is fixed; only the selection changes).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bristol_ml.evaluation.metrics",
            "evaluation.metrics.names=[mae,rmse]",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI with override must exit 0; returncode={result.returncode!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    selected_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("selected:")),
        None,
    )
    assert selected_line is not None, (
        "stdout must include a line beginning with 'selected:' when override is active."
    )
    assert "mae" in selected_line, f"'selected:' line must contain 'mae'; got {selected_line!r}"
    assert "rmse" in selected_line, f"'selected:' line must contain 'rmse'; got {selected_line!r}"
    # mape and wape must NOT appear in the selected line.
    assert "mape" not in selected_line, (
        f"'selected:' line must NOT contain 'mape' after override; got {selected_line!r}"
    )
    assert "wape" not in selected_line, (
        f"'selected:' line must NOT contain 'wape' after override; got {selected_line!r}"
    )
