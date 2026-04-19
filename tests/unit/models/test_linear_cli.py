"""CLI acceptance tests for ``python -m bristol_ml.models.linear``.

Tests are spec-derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T4 (acceptance criteria).
- ``docs/intent/DESIGN.md`` §2.1.1: every module runs standalone via
  ``python -m bristol_ml.<module>``; ``--help`` must exit 0.
- ``src/bristol_ml/models/linear.py`` ``_cli_main`` docstring: prints resolved
  ``target_column``, ``fit_intercept``, and ``feature_columns``; injects
  ``model=linear`` override before any user-supplied overrides.

Style follows ``tests/unit/models/test_naive_cli.py`` — ``subprocess.run``
with ``capture_output=True, text=True, check=False``.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan clause it guards.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import subprocess
import sys

# ---------------------------------------------------------------------------
# 22. test_linear_cli_help_exits_zero
# ---------------------------------------------------------------------------


def test_linear_cli_help_exits_zero() -> None:
    """Guards DESIGN §2.1.1: ``python -m bristol_ml.models.linear --help`` exits 0.

    The ``--help`` flag must exit with code 0 and write to stdout.  The output
    must contain the word "usage" (case-insensitive) confirming that argparse
    has produced a usage line.

    Plan clause: T4 / DESIGN §2.1.1 standalone-module principle.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.linear", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.linear --help' must exit 0; "
        f"returncode={result.returncode}\nstderr={result.stderr!r}"
    )
    assert "usage" in result.stdout.lower(), (
        f"'--help' output must contain 'usage'; stdout={result.stdout!r} (T4 / DESIGN §2.1.1)."
    )


# ---------------------------------------------------------------------------
# 23. test_linear_cli_prints_target_column
# ---------------------------------------------------------------------------


def test_linear_cli_prints_target_column() -> None:
    """Guards T4: linear CLI with no overrides prints ``target_column=nd_mw``.

    The default ``LinearConfig.target_column`` is ``"nd_mw"`` (the assembler
    OUTPUT_SCHEMA name for national demand).  Running the module with no extra
    overrides must:
    - exit 0.
    - print ``target_column=nd_mw`` on stdout.

    Plan clause: T4 / linear.py ``_cli_main`` docstring.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.linear"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.linear' must exit 0; "
        f"returncode={result.returncode}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "target_column=nd_mw" in result.stdout, (
        f"stdout must contain 'target_column=nd_mw'; "
        f"got stdout={result.stdout!r} (T4 / linear.py _cli_main)."
    )


# ---------------------------------------------------------------------------
# 24. test_linear_cli_respects_fit_intercept_override
# ---------------------------------------------------------------------------


def test_linear_cli_respects_fit_intercept_override() -> None:
    """Guards T4: linear CLI with ``model.fit_intercept=false`` override prints the setting.

    The linear YAML is composed at ``@package model`` so its fields sit directly
    under the ``model`` key.  Passing ``model.fit_intercept=false`` as a Hydra
    override must:
    - exit 0.
    - print ``fit_intercept=False`` on stdout.

    The ``_cli_main`` injects ``model=linear`` before any user overrides so the
    caller need not add it explicitly.

    Plan clause: T4 / linear.py ``_cli_main`` docstring / ``conf/model/linear.yaml``
    override example.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bristol_ml.models.linear",
            "model.fit_intercept=false",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.linear model.fit_intercept=false' must exit 0; "
        f"returncode={result.returncode}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "fit_intercept=False" in result.stdout, (
        f"stdout must contain 'fit_intercept=False'; "
        f"got stdout={result.stdout!r} (T4 / linear.py _cli_main fit_intercept override)."
    )
