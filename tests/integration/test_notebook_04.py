"""Integration test: ``notebooks/04_remit_ablation.ipynb`` executes cleanly.

Stage 16 plan §6 T7 maps to this test — the notebook is the demo
surface and the acceptance criterion (intent §"Demo moment") requires
it executes top-to-bottom without error.

The test runs against the stub corpus under ``BRISTOL_ML_REMIT_STUB=1``
and ``BRISTOL_ML_LLM_STUB=1`` so it has no network dependency.  When
no ``with_remit`` registered runs are present (the host has not yet
completed Task T6), the notebook prints the documented runbook banner
and renders an all-dash table — sufficient to verify the notebook
executes, by design.

Mirrors :mod:`tests.integration.test_notebook_15`: the executed copy is
written next to the source notebook and unlinked in the ``finally``
block; a session-scoped ``autouse`` fixture sweeps orphans from prior
runs.

The companion source-inspection test
:func:`test_notebook_04_builder_never_calls_fit` enforces AC-4
(the ablation must not re-fit; it loads from the registry only) by
walking the builder script's source for any reference to ``model.fit``
or ``.fit(`` outside of comments / WARNING strings.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "04_remit_ablation.ipynb"
BUILDER_PATH = REPO_ROOT / "scripts" / "_build_notebook_16.py"


def test_notebook_04_executes_top_to_bottom(tmp_path: Path) -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    Asserts after a clean execute:

    - Cell 1 (``T7 Cell 1``) — bootstrap + registered-runs probe.
    - Cell 2 (``T7 Cell 2``) — registry lookup with stub-mode warning.
    - Cell 3 (``T7 Cell 3``) — four-row metric table.

    The three executable cells span the load-bearing path: bootstrap
    → registry probe → table.  The trailing markdown cell (AC-5
    commentary) is checked structurally below.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_16.py`."
    )

    suffix = tmp_path.name
    executed_path = NOTEBOOK_PATH.with_name(
        f"{NOTEBOOK_PATH.stem}.pytest-exec-{suffix}.ipynb"
    )
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
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_str, src_str, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Plan AC-10: stub-mode forces the offline path so CI has no
    # network or API-key dependency.
    env["BRISTOL_ML_REMIT_STUB"] = "1"
    env["BRISTOL_ML_LLM_STUB"] = "1"
    # Sweep any registry runs that earlier tests may have left behind so
    # the notebook's "no registered runs yet" banner path is exercised
    # deterministically.  The notebook itself does not write to the
    # registry; the host's T6 step does.
    env["BRISTOL_ML_REGISTRY_DIR"] = str(tmp_path / "registry_empty")

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

    expected_markers = ("T7 Cell 1", "T7 Cell 2", "T7 Cell 3")
    for marker in expected_markers:
        matching = [
            cell for cell in code_cells if marker in "".join(cell.get("source", []))
        ]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the executed "
            f"notebook; found none.  Did the cell labelling change in "
            f"scripts/_build_notebook_16.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after execute — "
            f"plan §6 T7 requires the bootstrap / load / table cells to render."
        )


@pytest.fixture(autouse=True, scope="session")
def _sweep_stale_pytest_exec_notebooks_04() -> None:
    """Belt-and-braces cleanup of orphaned ``.pytest-exec-*.ipynb`` files."""
    notebooks_dir = NOTEBOOK_PATH.parent
    for stale in notebooks_dir.glob(f"{NOTEBOOK_PATH.stem}.pytest-exec-*.ipynb"):
        stale.unlink(missing_ok=True)


@pytest.mark.parametrize("expected_count", [5])
def test_notebook_04_has_expected_cell_count(expected_count: int) -> None:
    """Sanity check on the source notebook's cell shape.

    Plan §5 specifies three executable cells (bootstrap, load, table)
    plus a leading title-markdown and a trailing AC-5 commentary
    markdown — five cells total.  Catches accidental drift in the
    builder script.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert len(notebook["cells"]) == expected_count, (
        f"Notebook has {len(notebook['cells'])} cells; expected {expected_count}."
    )


def test_notebook_04_builder_never_calls_fit() -> None:
    """Plan AC-4 / D13 — the ablation notebook must not call ``model.fit()``.

    Source-inspection of the builder script (rather than
    `nbconvert`-running and grepping outputs) keeps the test fast and
    independent of registry state.  The check looks for any
    ``.fit(`` call site outside of comments / docstrings; the
    documented banner string in the runbook ("``model=nn_temporal``"
    etc.) is fine because it does not match the ``.fit(`` regex.
    """
    source = BUILDER_PATH.read_text()
    # Strip Python comments + triple-quoted strings so the runbook text
    # in the documented banner does not produce a false positive.
    no_comments = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
    no_docstrings = re.sub(r'"""[\s\S]*?"""', "", no_comments)
    no_runbook = re.sub(r"'''[\s\S]*?'''", "", no_docstrings)
    fit_calls = re.findall(r"\.fit\s*\(", no_runbook)
    assert not fit_calls, (
        "scripts/_build_notebook_16.py emits one or more `.fit(` call(s) "
        f"outside comments/docstrings: {fit_calls!r}.  The Stage 16 ablation "
        "notebook must load registered runs (registry.load + sidecar metrics) "
        "without re-fitting — plan AC-4 / D13."
    )


def test_notebook_04_commentary_cell_meets_ac5_minimum() -> None:
    """Plan AC-5 — the commentary markdown cell is non-trivial and on-topic.

    Acceptance: the trailing markdown cell exists, exceeds 200
    characters, and mentions at least one of the topic keywords
    (``price``, ``demand``, ``effect``, ``null``, ``stub``).  The
    requirements artefact's testable form for AC-5 spelled this out
    verbatim; this test locks it.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    markdown_cells = [c for c in notebook["cells"] if c["cell_type"] == "markdown"]
    assert markdown_cells, "Notebook has no markdown cells."
    trailing = markdown_cells[-1]
    source = "".join(trailing["source"])
    assert len(source) > 200, (
        f"Trailing AC-5 commentary cell is only {len(source)} chars; the "
        "plan requires a non-trivial honest interpretation of the result."
    )
    keywords = ("price", "demand", "effect", "null", "stub")
    matched = [k for k in keywords if k.lower() in source.lower()]
    assert matched, (
        "Trailing commentary cell mentions none of the AC-5 keywords "
        f"({keywords!r}); the cell must address the price-vs-demand "
        "asymmetry, the null-result possibility, or the stub/real "
        "extractor caveat."
    )
