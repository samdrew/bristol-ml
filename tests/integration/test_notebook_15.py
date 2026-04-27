"""Integration test: ``notebooks/15_embedding_index.ipynb`` executes cleanly.

Stage 15 plan §6 T9 maps to this test — the notebook is the
pedagogical surface and the acceptance criterion (intent §"Demo
moment") requires it executes top-to-bottom without error.

The test runs against the tiny REMIT corpus fixture under
``BRISTOL_ML_EMBEDDING_STUB=1`` so it has no network dependency and
lives in the fast suite — the stub's deterministic SHA-256-derived
vectors exercise the same Embedder + VectorIndex contract as the
live ``Alibaba-NLP/gte-modernbert-base`` path, just with offline
payload.

Mirrors :mod:`tests.integration.test_notebook_14` line-for-line: the
executed copy is written next to the source notebook (nbconvert sets
the kernel cwd to the source notebook's parent and silently ignores
``--ExecutePreprocessor.cwd``), and the executed copy is unlinked in
the ``finally`` block. A session-scoped ``autouse`` fixture sweeps
orphans from prior runs.

Plan reference: ``docs/plans/active/15-embedding-index.md`` §6 T9 named
test ``test_notebook_15_executes_top_to_bottom``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "15_embedding_index.ipynb"


def test_notebook_15_executes_top_to_bottom(tmp_path: Path) -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    Asserts after a clean execute:

    - Cell 1 (``T5 Cell 1``) — bootstrap + config print produced output.
    - Cell 3 (``T5 Cell 3``) — index-build + provenance print produced output.
    - Cell 4 (``T5 Cell 4``) — top-k neighbours table produced output.
    - Cell 5 (``T5 Cell 5``) — UMAP scatter renders an image output.

    The four cells span the load-bearing path: config → index build
    → query → projection. Cell 6 (cross-stage join) intentionally
    prints regardless of Stage 14 output presence and is checked
    indirectly via the nbconvert exit code.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_15.py`."
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
    # Stub-mode forces the offline path (Plan §6 T9 / NFR-1 / AC-4).
    env["BRISTOL_ML_EMBEDDING_STUB"] = "1"
    # HF_HUB_OFFLINE is set globally in conftest.py but a subprocess
    # copies a snapshot of os.environ — set it explicitly so a
    # regression that re-introduces a network call surfaces here as
    # an OSError rather than a silent download.
    env["HF_HUB_OFFLINE"] = "1"

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
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"nbconvert --execute exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    notebook = json.loads(executed_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    expected_markers = ("T5 Cell 1", "T5 Cell 3", "T5 Cell 4", "T5 Cell 5")
    for marker in expected_markers:
        matching = [cell for cell in code_cells if marker in "".join(cell.get("source", []))]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the executed "
            f"notebook; found none.  Did the cell labelling change in "
            f"scripts/_build_notebook_15.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after execute — "
            f"plan §6 T9 requires the bootstrap / build / query / projection "
            f"cells to render in the demo notebook."
        )


@pytest.fixture(autouse=True, scope="session")
def _sweep_stale_pytest_exec_notebooks_15() -> None:
    """Belt-and-braces cleanup of orphaned ``.pytest-exec-*.ipynb`` files."""
    notebooks_dir = NOTEBOOK_PATH.parent
    for stale in notebooks_dir.glob(f"{NOTEBOOK_PATH.stem}.pytest-exec-*.ipynb"):
        stale.unlink(missing_ok=True)


@pytest.mark.parametrize("expected_count", [8])
def test_notebook_15_has_expected_cell_count(expected_count: int) -> None:
    """Sanity check on the source notebook's cell shape.

    Plan §5 specifies six executable cells (bootstrap, gold-set load,
    build/load index, top-k query, UMAP projection, optional cross-
    stage join); the builder wraps those in a leading title-markdown
    cell *and* a trailing discussion-markdown cell, bringing the
    total to eight. This test catches accidental cell drift away
    from the documented structure.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert len(notebook["cells"]) == expected_count, (
        f"Notebook has {len(notebook['cells'])} cells; expected {expected_count}."
    )
