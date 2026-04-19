"""CLI acceptance tests for ``python -m bristol_ml.models.naive``.

Tests are spec-derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T3 (acceptance criteria).
- ``docs/intent/DESIGN.md`` §2.1.1: every module runs standalone via
  ``python -m bristol_ml.<module>``; ``--help`` must exit 0.
- ``src/bristol_ml/models/naive.py`` ``_cli_main`` docstring: prints resolved
  strategy and target_column; injects ``model=naive`` so the default
  ``model=linear`` does not collide.

Style follows ``tests/unit/test_config.py::test_python_dash_m_help_exits_zero``
— ``subprocess.run`` with ``capture_output=True`` and ``check=False``.

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
# 20. test_naive_cli_help_exits_zero
# ---------------------------------------------------------------------------


def test_naive_cli_help_exits_zero() -> None:
    """Guards DESIGN §2.1.1: ``python -m bristol_ml.models.naive --help`` exits 0.

    The ``--help`` flag must exit with code 0 and write to stdout.  The output
    must contain the word "usage" (case-insensitive) confirming that argparse
    has produced a usage line.

    Plan clause: T3 / DESIGN §2.1.1 standalone-module principle.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.naive", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.naive --help' must exit 0; "
        f"returncode={result.returncode}\nstderr={result.stderr!r}"
    )
    assert "usage" in result.stdout.lower(), (
        f"'--help' output must contain 'usage'; stdout={result.stdout!r} (DESIGN §2.1.1)."
    )


# ---------------------------------------------------------------------------
# 21. test_naive_cli_prints_strategy_default
# ---------------------------------------------------------------------------


def test_naive_cli_prints_strategy_default() -> None:
    """Guards T3: naive CLI with no overrides prints ``strategy=same_hour_last_week``.

    The default ``NaiveConfig.strategy`` is ``"same_hour_last_week"`` (plan D1).
    Running the module with no extra overrides must:
    - exit 0.
    - print ``strategy=same_hour_last_week`` on stdout.

    Plan clause: T3 / D1 (default strategy) / naive.py ``_cli_main`` docstring.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.naive"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.naive' must exit 0; "
        f"returncode={result.returncode}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "strategy=same_hour_last_week" in result.stdout, (
        f"stdout must contain 'strategy=same_hour_last_week'; "
        f"got stdout={result.stdout!r} (T3 / D1 default strategy)."
    )


# ---------------------------------------------------------------------------
# 22. test_naive_cli_respects_strategy_override
# ---------------------------------------------------------------------------


def test_naive_cli_respects_strategy_override() -> None:
    """Guards T3: naive CLI with ``model.strategy=same_hour_yesterday`` override.

    The naive YAML is composed at ``@package model`` so its fields are directly
    under the ``model`` key (per ``conf/model/naive.yaml`` comment:
    "``model.strategy=same_hour_yesterday``").  Passing this Hydra override must:
    - exit 0.
    - print ``strategy=same_hour_yesterday`` on stdout.

    The ``_cli_main`` injects ``model=naive`` before the user-supplied overrides
    so the user need not specify it; the test passes an additional strategy
    override on top of that.

    Plan clause: T3 / D1 (strategy override) / naive.py ``_cli_main`` docstring /
    ``conf/model/naive.yaml`` override example.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bristol_ml.models.naive",
            "model.strategy=same_hour_yesterday",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"'python -m bristol_ml.models.naive model.strategy=same_hour_yesterday' "
        f"must exit 0; returncode={result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "strategy=same_hour_yesterday" in result.stdout, (
        f"stdout must contain 'strategy=same_hour_yesterday'; "
        f"got stdout={result.stdout!r} (T3 / D1 strategy override)."
    )
