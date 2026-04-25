"""Integration tests: ``notebooks/11-complex-nn.ipynb`` — the Stage 11 surface.

Stage 11 Task T7 maps to three tests:

- ``test_notebook_11_ablation_cell_covers_six_model_families`` — AC-3
  static-source check that every model family named in the plan §1
  D10 ablation contract is referenced in the ablation cell's narrative
  markdown.  All six families show up because the ablation cell loops
  over the registry's full ``list_runs()`` output and the registry is
  populated by an earlier cell that names each family explicitly.
- ``test_notebook_11_ablation_cell_does_not_refit_registered_runs`` —
  AC-5 static-source check that the ablation cell does not call
  ``.fit(`` / ``harness.evaluate`` / ``compare_on_holdout``.  The
  predict-only path is the literal AC-5 reconciliation in plan
  preamble lines 17 + 73-74.
- ``test_notebook_11_executes_cleanly`` — ``@pytest.mark.slow``
  nbconvert --execute round-trip mirroring
  ``tests/integration/test_notebook_08.py``.  Confirms the live-fit
  cell (Cell 4) and the ablation cell (Cell 6) produce non-empty
  outputs.

Plan reference: ``docs/plans/active/11-complex-nn.md`` §6 Task T7
(lines 541-552); §1 D10/D11 (ablation contract); §4 AC-3 / AC-5.

The fast-suite default (``-m 'not slow'`` registered at
``pyproject.toml [tool.pytest.ini_options]``) skips the nbconvert
check; opt in with ``uv run pytest -m slow``.  The two static-source
tests are fast and run by default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "11-complex-nn.ipynb"

# The six model families named in plan §1 D10 — keys match the registry
# sidecar ``type`` strings registered through
# ``bristol_ml.registry._dispatch._CLASS_NAME_TO_TYPE``.
EXPECTED_MODEL_FAMILIES: tuple[str, ...] = (
    "naive",
    "linear",
    "sarimax",
    "scipy_parametric",
    "nn_mlp",
    "nn_temporal",
)

# AC-5 forbidden tokens: any of these in the ablation cell source means
# a registered run is being re-fit.  ``compare_on_holdout`` re-fits per
# fold internally (codebase map §2.4); ``harness.evaluate`` does the
# same; ``.fit(`` is the bare ``Model.fit`` call.
ABLATION_FORBIDDEN_TOKENS: tuple[str, ...] = (
    ".fit(",
    "harness.evaluate(",
    "compare_on_holdout(",
)


def _load_notebook_cells() -> list[dict]:
    """Return the notebook's cells, raising if the file is missing.

    The Stage 11 builder script
    (``scripts/_build_notebook_11.py``) writes the notebook on demand;
    a missing file means the regen flow has not been run.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_11.py` followed by "
        f"`uv run jupyter nbconvert --execute --inplace {NOTEBOOK_PATH}`."
    )
    nb = json.loads(NOTEBOOK_PATH.read_text())
    return nb["cells"]


def _find_code_cell_by_marker(cells: list[dict], marker: str) -> dict:
    """Return the unique code cell whose source contains ``marker``.

    The builder bakes a ``T7 Cell N`` comment into the first source
    line of every code cell so tests can locate cells robustly even
    after a reorder.  Mirrors the Stage 8 / Stage 10 pattern.
    """
    matches = [
        cell
        for cell in cells
        if cell["cell_type"] == "code" and marker in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, (
        f"Expected exactly one code cell containing marker {marker!r}; "
        f"found {len(matches)}.  Did the labelling change in "
        f"scripts/_build_notebook_11.py?"
    )
    return matches[0]


# ===========================================================================
# AC-3 — six model families covered in the ablation cell narrative
# ===========================================================================


def test_notebook_11_ablation_cell_covers_six_model_families() -> None:
    """AC-3 static-source check: ablation narrative names every family.

    Static-source inspection rather than a runtime check on the
    rendered table.  The ablation code cell itself iterates over
    ``registry.list_runs()`` whose output depends on a populated
    registry; the *contract* the test enforces is that the cell's
    narrative markdown — the closing commentary cell — names every
    family the plan's D10 contract requires.

    The cell looked at is the closing commentary cell (the last
    markdown cell of the notebook), where the closing arc explicitly
    walks through every family the table covers.  This is the
    facilitator's safety-net: if a family is renamed in
    ``conf/_schemas.py`` or removed from the registry's dispatch
    table, the closing narrative drifts and this test fails.

    Plan clause: §6 T7 (line 548); §4 AC-3.
    """
    cells = _load_notebook_cells()
    markdown_sources = [
        "".join(cell.get("source", [])) for cell in cells if cell["cell_type"] == "markdown"
    ]
    closing_md = markdown_sources[-1]
    missing = [name for name in EXPECTED_MODEL_FAMILIES if name not in closing_md]
    assert not missing, (
        f"Closing commentary must name every Stage-11 model family per "
        f"plan §1 D10; missing {missing!r}.  Closing narrative was:\n"
        f"{closing_md}"
    )


# ===========================================================================
# AC-5 — ablation cell is predict-only (no .fit, no harness, no benchmark)
# ===========================================================================


def test_notebook_11_ablation_cell_does_not_refit_registered_runs() -> None:
    """AC-5 static-source check: ablation cell is predict-only.

    Plan preamble lines 17 + 73-74 are explicit: the ablation cell
    must not call ``harness.evaluate`` or ``compare_on_holdout``
    because both internally re-fit the model on every fold; the cell
    must not call ``model.fit`` either.  The literal AC-5 phrasing —
    "the notebook's ablation table is reproducible from the registry
    without re-training anything already registered" — collapses to a
    structural guard on the ablation cell's source.

    Static-source inspection because the runtime guard
    (``test_notebook_11_executes_cleanly``) only proves the cell ran;
    it does not prove the cell took the predict-only path.  A future
    edit that introduces ``.fit(`` would still exit zero but break
    AC-5 silently — this test is the loud-failure surface.

    Plan clause: §1 D11 cut rationale; §4 AC-5; preamble lines 17 / 73-74.
    """
    cells = _load_notebook_cells()
    ablation = _find_code_cell_by_marker(cells, "T7 Cell 6")
    src = "".join(ablation.get("source", []))
    forbidden_present = [tok for tok in ABLATION_FORBIDDEN_TOKENS if tok in src]
    assert not forbidden_present, (
        f"Ablation cell (T7 Cell 6) must take the predict-only path "
        f"per plan §4 AC-5 + §1 D11; found forbidden tokens "
        f"{forbidden_present!r}.  The ablation cell's source was:\n{src}"
    )
    # Defensive positive check — the cell must use the registry
    # load + predict path that AC-5 mandates.
    assert "registry.load(" in src, (
        "Ablation cell must call registry.load() to materialise each "
        "registered run; AC-5's reconciliation requires the predict-only "
        "path."
    )
    assert ".predict(" in src, (
        "Ablation cell must call model.predict() on each loaded run; "
        "AC-5's reconciliation requires the predict-only path."
    )


# ===========================================================================
# AC-3 / AC-5 runtime smoke — nbconvert --execute round-trip
# ===========================================================================


@pytest.mark.slow
def test_notebook_11_executes_cleanly() -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output.

    Mirrors ``tests/integration/test_notebook_08.py`` exactly.  Two
    pedagogically load-bearing code cells must produce non-empty
    outputs:

    - **T7 Cell 4** — the live TCN training cell.  AC-2 evidence (the
      shared ``run_training_loop`` is invoked end-to-end through the
      ``epoch_callback`` seam).
    - **T7 Cell 6** — the ablation cell.  AC-3 / AC-5 evidence (the
      table renders).

    A copy of the notebook is executed under ``notebooks/`` rather
    than ``tmp_path`` because Cell 1's bootstrap walks up from
    ``Path.cwd()`` looking for ``pyproject.toml`` to establish
    ``REPO_ROOT``; a copy under ``/tmp`` defeats that walk.

    Plan clause: §6 T7 (line 550); §4 AC-3 / AC-5.
    """
    assert NOTEBOOK_PATH.exists(), (
        f"Expected source notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_11.py`."
    )

    executed_path = NOTEBOOK_PATH.with_suffix(".pytest-exec.ipynb")
    executed_path.write_bytes(NOTEBOOK_PATH.read_bytes())
    try:
        _run_and_assert(executed_path)
    finally:
        executed_path.unlink(missing_ok=True)


def _run_and_assert(executed_path: Path) -> None:
    """Run nbconvert --execute on ``executed_path``; assert key cells emitted output."""
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
    expected_markers = ("T7 Cell 4", "T7 Cell 6")
    for marker in expected_markers:
        matching = [cell for cell in code_cells if marker in "".join(cell.get("source", []))]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the "
            f"executed notebook; found none.  Did the cell labelling "
            f"change in scripts/_build_notebook_11.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after "
            f"execute — AC-3 / AC-5 require the live-fit and ablation "
            f"cells to emit at least one non-empty payload."
        )
