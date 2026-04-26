"""Integration test: ``notebooks/13_remit_ingestion.ipynb`` executes cleanly.

Stage 13 AC-9 / D18i map to this test — the notebook is the
pedagogical surface and the acceptance criterion says it must execute
end-to-end without error.

The test runs against the stub fixture (``BRISTOL_ML_REMIT_STUB=1``)
so it has no network dependency and can live in the fast suite — the
stub's 10 hand-crafted records exercise the same `load` /
`as_of` / aggregate / plot code path as a live run, just with a
deterministic offline payload.

Plan reference: ``docs/plans/active/13-remit-ingestion.md`` §6 T5
named test ``test_notebook_13_remit_executes_top_to_bottom`` (AC-9).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "13_remit_ingestion.ipynb"


def test_notebook_13_remit_executes_top_to_bottom(tmp_path: Path) -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    The test executes the committed notebook against a sibling copy
    placed under ``notebooks/`` so the test run does not mutate the
    source-controlled artefact (Cell 1's bootstrap walks up from
    ``Path.cwd()`` looking for ``pyproject.toml``; a copy under
    ``/tmp`` defeats the walk).  After ``nbconvert`` exits 0 the test
    re-reads the executed copy and asserts the three load-bearing code
    cells produced non-empty output:

    - Cell 1 (``T5 Cell 1``) — fetch + load summary lines.
    - Cell 3 (``T5 Cell 3``) — monthly aggregate table.
    - Cell 5 (``T5 Cell 5``) — stacked-area chart figure.

    The ``BRISTOL_ML_REMIT_STUB=1`` environment variable routes the
    fetch through the in-memory stub fixture; ``BRISTOL_ML_CACHE_DIR``
    points at ``tmp_path`` so the run does not pollute the developer
    cache.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_13.py`."
    )

    executed_path = NOTEBOOK_PATH.with_suffix(".pytest-exec.ipynb")
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
    # Stub-mode + tmp-path cache: deterministic, offline, no pollution.
    env["BRISTOL_ML_REMIT_STUB"] = "1"
    env["BRISTOL_ML_CACHE_DIR"] = str(tmp_path)

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
            f"Expected a code cell containing marker {marker!r} in the executed notebook; "
            f"found none.  Did the cell labelling change in scripts/_build_notebook_13.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after execute — "
            f"AC-9 requires the load summary / aggregate table / stacked-area chart "
            f"to render in the demo notebook."
        )


@pytest.mark.parametrize("expected_count", [7])
def test_notebook_13_has_expected_cell_count(expected_count: int) -> None:
    """Sanity check on the source notebook's cell shape.

    The plan §5 specifies a three-cell minimum (load / compute / plot);
    the build script wraps those in markdown for narrative coherence,
    bringing the total to seven cells.  This test catches accidental
    cell drift away from the documented structure.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert len(notebook["cells"]) == expected_count, (
        f"Notebook has {len(notebook['cells'])} cells; expected {expected_count}."
    )
