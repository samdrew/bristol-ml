"""Integration test: ``notebooks/14_llm_extractor.ipynb`` executes cleanly.

Stage 14 plan §6 T6 maps to this test — the notebook is the
pedagogical surface and the acceptance criterion (intent line 28,
"demo moment") requires it executes top-to-bottom without error.

The test runs against the stub gold-set fixture
(``BRISTOL_ML_LLM_STUB=1``) so it has no network dependency and lives
in the fast suite — the stub's 76 hand-labelled records exercise the
same harness code path as a live run, just with an offline payload.

Mirrors :mod:`tests.integration.test_notebook_13` line-for-line: the
executed copy is written next to the source notebook (nbconvert sets
the kernel cwd to the source notebook's parent and silently ignores
``--ExecutePreprocessor.cwd``), and the executed copy is unlinked in
the ``finally`` block. A session-scoped ``autouse`` fixture sweeps
orphans from prior runs.

Plan reference: ``docs/plans/active/14-llm-extractor.md`` §6 T6 named
test ``test_notebook_14_llm_extractor_executes_top_to_bottom``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "14_llm_extractor.ipynb"


def test_notebook_14_llm_extractor_executes_top_to_bottom(tmp_path: Path) -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    Asserts after a clean execute:

    - Cell 1 (``T5 Cell 1``) — bootstrap + config print produced output.
    - Cell 3 (``T5 Cell 3``) — stub-extraction output table produced output.
    - Cell 5 (``T5 Cell 5``) — harness summary table produced output.

    The three cells span the load-bearing path: config → stub run →
    harness run. Cell 4 (live LlmExtractor banner) intentionally
    prints regardless of cassette presence and is checked indirectly
    via the nbconvert exit code.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_14.py`."
    )

    suffix = tmp_path.name
    executed_path = NOTEBOOK_PATH.with_name(f"{NOTEBOOK_PATH.stem}.pytest-exec-{suffix}.ipynb")
    executed_path.write_bytes(NOTEBOOK_PATH.read_bytes())
    try:
        _run_and_assert(executed_path, tmp_path)
    finally:
        executed_path.unlink(missing_ok=True)


def _run_and_assert(executed_path: Path, tmp_path: Path) -> None:
    """Run ``nbconvert --execute`` and assert key cells produced output."""
    env = dict(os.environ)
    repo_str = str(REPO_ROOT)
    src_str = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join([repo_str, src_str, env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    # Stub-mode forces the offline path (Plan §6 T6 / NFR-1 / AC-2).
    env["BRISTOL_ML_LLM_STUB"] = "1"

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
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"nbconvert --execute exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    notebook = json.loads(executed_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    expected_markers = ("T5 Cell 1", "T5 Cell 3", "T5 Cell 5")
    for marker in expected_markers:
        matching = [cell for cell in code_cells if marker in "".join(cell.get("source", []))]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the executed "
            f"notebook; found none.  Did the cell labelling change in "
            f"scripts/_build_notebook_14.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after execute — "
            f"plan §6 T6 requires the bootstrap / stub-run / harness-summary "
            f"cells to render in the demo notebook."
        )


@pytest.fixture(autouse=True, scope="session")
def _sweep_stale_pytest_exec_notebooks_14() -> None:
    """Belt-and-braces cleanup of orphaned ``.pytest-exec-*.ipynb`` files."""
    notebooks_dir = NOTEBOOK_PATH.parent
    for stale in notebooks_dir.glob(f"{NOTEBOOK_PATH.stem}.pytest-exec-*.ipynb"):
        stale.unlink(missing_ok=True)


@pytest.mark.parametrize("expected_count", [7])
def test_notebook_14_has_expected_cell_count(expected_count: int) -> None:
    """Sanity check on the source notebook's cell shape.

    Plan §5 specifies six cells (bootstrap, gold-set load, run stub,
    run real / banner, side-by-side comparison, discussion); the
    builder wraps those in a leading title-markdown cell, bringing
    the total to seven. This test catches accidental cell drift away
    from the documented structure.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert len(notebook["cells"]) == expected_count, (
        f"Notebook has {len(notebook['cells'])} cells; expected {expected_count}."
    )
