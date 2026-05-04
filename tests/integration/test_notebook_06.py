"""Integration test: ``notebooks/06_enhanced_evaluation.ipynb`` executes cleanly.

Stage 6 — enhanced evaluation & visualisation.  This notebook is the
demo surface for the diagnostic library
(``bristol_ml.evaluation.plots``) applied side-by-side to the Stage 4
weather-only OLS and the Stage 5 weather + calendar OLS.  It expects
both feature-table caches (``weather_only.parquet`` and
``weather_calendar.parquet``) to be warm; when the host CI runner
does not have either cache primed the test SKIPS rather than fails —
mirroring the cassette-skip pattern in the LLM and embeddings
notebooks.

The smoke checks all four diagnostic surfaces produced output:

- T6 Cell 1 — bootstrap + config print.
- T6 Cell 3 — both rolling-origin evaluations completed (per-fold log
  output + cross-fold mean summary text).
- T6 Cell 4 — 2x2 diagnostic grid for at least one OLS model rendered.
- T6 Cell 6 — q10-q90 uncertainty band rendered.

The structural assertion guards the cell labelling so a future
builder change reorganising the cells fails the test deliberately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "06_enhanced_evaluation.ipynb"
WEATHER_ONLY_CACHE = REPO_ROOT / "data" / "features" / "weather_only.parquet"
WEATHER_CALENDAR_CACHE = REPO_ROOT / "data" / "features" / "weather_calendar.parquet"


def _caches_warm() -> bool:
    """Both feature-table caches present on disk?"""
    return WEATHER_ONLY_CACHE.exists() and WEATHER_CALENDAR_CACHE.exists()


@pytest.mark.skipif(
    not _caches_warm(),
    reason=(
        "Feature-table caches not warm; the Stage 6 notebook needs both "
        "weather_only.parquet and weather_calendar.parquet under data/features/. "
        "Populate via `uv run python -m bristol_ml.features.assembler "
        "features={weather_only,weather_calendar} --cache offline` before "
        "running this integration test."
    ),
)
def test_notebook_06_executes_top_to_bottom(tmp_path: Path) -> None:
    """``jupyter nbconvert --execute`` returns 0 and key cells emit output."""
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH}; regenerate with "
        f"`uv run python scripts/_build_notebook_06.py`."
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
    env.setdefault("BRISTOL_ML_REMIT_STUB", "1")
    env.setdefault("BRISTOL_ML_LLM_STUB", "1")

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

    expected_markers = ("T6 Cell 1", "T6 Cell 3", "T6 Cell 4", "T6 Cell 6")
    for marker in expected_markers:
        matching = [cell for cell in code_cells if marker in "".join(cell.get("source", []))]
        assert matching, (
            f"Expected a code cell containing marker {marker!r} in the executed "
            f"notebook; found none. Did the cell labelling change in "
            f"scripts/_build_notebook_06.py?"
        )
        outputs = matching[0].get("outputs", [])
        assert outputs, (
            f"Code cell tagged {marker!r} produced no outputs after execute — "
            f"the Stage 6 demo requires bootstrap + harness + diagnostic + "
            f"uncertainty-band cells to render."
        )


@pytest.fixture(autouse=True, scope="session")
def _sweep_stale_pytest_exec_notebooks_06() -> None:
    """Belt-and-braces cleanup of orphaned ``.pytest-exec-*.ipynb`` files."""
    notebooks_dir = NOTEBOOK_PATH.parent
    for stale in notebooks_dir.glob(f"{NOTEBOOK_PATH.stem}.pytest-exec-*.ipynb"):
        stale.unlink(missing_ok=True)


@pytest.mark.parametrize("expected_count", [9])
def test_notebook_06_has_expected_cell_count(expected_count: int) -> None:
    """Sanity check on the source notebook's cell shape.

    Builder produces nine cells: title-markdown, seven executable
    (bootstrap, load, harness, diagnostics, overlay, band, benchmark),
    and a closing discussion-markdown.  This test catches accidental
    cell drift away from the documented structure (the user-facing
    requirement is "include both OLS models in 48-hour forecast" /
    "include both models in uncertainty forecast" / "fix
    benchmark_holdout_bar" — a structural change should be deliberate).
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert len(notebook["cells"]) == expected_count, (
        f"Notebook has {len(notebook['cells'])} cells; expected {expected_count}."
    )


def test_notebook_04_has_no_stage_6_appendix() -> None:
    """nb04 must no longer carry the Stage 6 enhanced-evaluation appendix.

    Pins the Stage-4-vs-Stage-6 separation: nb04 is the linear-baseline
    + harness + three-way-benchmark notebook; nb06 owns the diagnostic
    library demos.  A regression that re-merges them (e.g. a future
    edit pasting plot helpers back into nb04) fails this test.
    """
    nb04 = json.loads((REPO_ROOT / "notebooks" / "04_linear_baseline.ipynb").read_text())
    sentinels = (
        "from bristol_ml.evaluation import plots",
        "forecast_overlay_with_band",
        "benchmark_holdout_bar",
    )
    for cell in nb04["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for sentinel in sentinels:
            assert sentinel not in src, (
                f"nb04 (linear baseline) carries a Stage 6 sentinel {sentinel!r}; "
                f"this content must live in nb06 (enhanced evaluation) only.  "
                f"See `scripts/_build_notebook_06.py`."
            )
