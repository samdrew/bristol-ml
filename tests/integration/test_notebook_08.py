"""Integration test: ``notebooks/08_scipy_parametric.ipynb`` executes cleanly.

Stage 8 AC-3 maps to this test — the notebook is the pedagogical surface
and the acceptance criterion says it must:

- Execute end-to-end without error under ``jupyter nbconvert --execute``.
- Produce non-empty outputs on the single-fold fit cell (Cell 5, AC-4
  evidence), the parameter-table cell (Cell 7, AC-3 "parameter
  estimates + CIs"), and the four-way forecast-overlay cell (Cell 10,
  AC-3 "forecast comparison against Naive / Linear / SARIMAX").

The test is marked ``@pytest.mark.slow`` because ``nbconvert --execute``
on the full 14-cell notebook runs for well over a minute (the
four-way rolling-origin evaluation in Cell 9 is the dominant cost).
The default ``-m 'not slow'`` opt-out registered at
``pyproject.toml [tool.pytest.ini_options]`` keeps the fast-suite
wall time bounded; opt in with ``uv run pytest -m slow``.

Plan reference: ``docs/plans/active/08-scipy-parametric.md`` §Task T5
named test ``test_notebook_08_executes_cleanly`` (AC-3).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "08_scipy_parametric.ipynb"


@pytest.mark.slow
def test_notebook_08_executes_cleanly() -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    The test executes the committed notebook against a sibling copy
    placed under ``notebooks/`` so the test run does not mutate the
    source-controlled artefact; it then re-reads the executed copy,
    walks the cell list, and asserts that the three pedagogically
    load-bearing code cells (Cell 5 / Cell 7 / Cell 10 in the plan's
    1-indexed enumeration — located via the ``T5 Cell N`` source-text
    marker baked into each cell by
    ``scripts/_build_notebook_08.py``) produced at least one non-empty
    output payload.

    The copy lives under ``notebooks/`` rather than ``tmp_path``
    because the notebook's Cell 1 bootstrap walks up from
    ``Path.cwd()`` looking for ``pyproject.toml`` to establish
    ``REPO_ROOT``; a copy under ``/tmp`` defeats that walk and leaves
    Hydra pointing at ``/conf`` instead of ``REPO_ROOT / "conf"``.

    Plan clause: T5 / AC-3.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected source notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_08.py`."
    )

    # Copy source into ``notebooks/`` with a unique sibling name.  The
    # ``try / finally`` below guarantees cleanup even if nbconvert fails.
    executed_path = NOTEBOOK_PATH.with_suffix(".pytest-exec.ipynb")
    executed_path.write_bytes(NOTEBOOK_PATH.read_bytes())
    try:
        _run_and_assert(executed_path)
    finally:
        executed_path.unlink(missing_ok=True)


def _run_and_assert(executed_path: Path) -> None:
    """Run nbconvert --execute on ``executed_path`` and assert key cells produced output."""

    # --execute exits non-zero on any cell error; the assertion below
    # therefore doubles as the "notebook runs end-to-end" check.  Inject
    # ``PYTHONPATH`` explicitly because the notebook's Cell 1 bootstrap
    # walks up from ``Path.cwd()`` looking for ``pyproject.toml``; when
    # the copy lives under ``tmp_path`` that walk never finds the repo
    # and ``from conf._schemas import ...`` fails.  Mirroring
    # ``pyproject.toml [tool.pytest.ini_options] pythonpath = ["src", "."]``
    # restores the import surface that every other test relies on.
    env = dict(os.environ)
    repo_str = str(REPO_ROOT)
    src_str = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join([repo_str, src_str, env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--inplace",
            str(executed_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"nbconvert --execute exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    notebook = json.loads(executed_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    # The plan's Cells 5 / 7 / 10 are (1-indexed against the 14-cell
    # recipe in §Task T5); among code cells only they land at
    # positions 3, 4, 6 (Cell 1 imports + Cell 3 scatter + Cell 5 fit +
    # Cell 7 table + Cell 8 overlay + Cell 9 evaluation + Cell 10
    # four-way overlay + Cell 11 stability diagnostic).  We key off the
    # cell's source-text marker (`T5 Cell N`) so a future reordering of
    # the notebook stays robust.
    expected_markers = ("T5 Cell 5", "T5 Cell 7", "T5 Cell 10")
    for marker in expected_markers:
        matching = [cell for cell in code_cells if marker in "".join(cell.get("source", []))]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the "
            f"executed notebook; found none.  Did the cell labelling "
            f"change in scripts/_build_notebook_08.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after "
            f"execute — AC-3 requires a non-empty payload (parameter "
            f"estimates, table, or overlay figure)."
        )
