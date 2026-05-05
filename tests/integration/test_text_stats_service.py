"""Integration tests for the worked-example text-stats service.

Exercises the service end-to-end via:

1. The programmatic ``run()`` API (Hydra in-process).
2. The standalone ``main()`` entry point (subprocess).

Together these prove the Hydra+Pydantic+CLI wiring works on the
shipped fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conf._schemas import TextStatsConfig
from TEMPLATE_PROJECT.services.text_stats_service import run

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"


def test_run_with_json_output() -> None:
    """``run()`` returns valid JSON with the expected shape."""
    cfg = TextStatsConfig(input_path=FIXTURE_PATH, output_format="json")
    output = run(cfg)
    parsed = json.loads(output)
    assert set(parsed.keys()) == {
        "character_count",
        "non_whitespace_character_count",
        "word_count",
        "line_count",
    }
    # The shipped fixture has 3 lines (one trailing newline).
    assert parsed["line_count"] == 3
    # Word count: 9 + 7 + 8 = 24 words across the three pangrams.
    assert parsed["word_count"] == 24


def test_run_with_human_output() -> None:
    """``run()`` returns a small aligned text table when output_format='human'."""
    cfg = TextStatsConfig(input_path=FIXTURE_PATH, output_format="human")
    output = run(cfg)
    assert "character_count" in output
    assert "word_count" in output
    # Should have one line per metric (4 metrics).
    assert len(output.splitlines()) == 4


def test_run_raises_on_unknown_output_format() -> None:
    """Pydantic should reject an unknown ``output_format`` at config time."""
    with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
        TextStatsConfig(input_path=FIXTURE_PATH, output_format="xml")  # type: ignore[arg-type]


def test_main_cli_prints_json_to_stdout() -> None:
    """``python -m TEMPLATE_PROJECT.services.text_stats_service`` exits 0 and prints JSON."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "TEMPLATE_PROJECT.services.text_stats_service",
            f"services.text_stats.input_path={FIXTURE_PATH}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    parsed = json.loads(result.stdout)
    assert parsed["line_count"] == 3
