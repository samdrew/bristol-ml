"""Spec-derived test for the Stage 15 T8 standalone-module entry.

Plan AC-9 (requirements §3): the module runs standalone via
``python -m bristol_ml.embeddings`` and exits 0 under stub mode.
Plan §6 T8: the CLI prints active config + a sample query and exits
0 (no subcommands; A4-narrowed surface).
"""

from __future__ import annotations

import subprocess
import sys


def test_module_runs_under_stub() -> None:
    """Plan §6 T8: ``python -m bristol_ml.embeddings`` exits 0 under stub mode.

    The smoke test runs the module as a real subprocess so the
    Hydra compose path, the Pydantic validation, the env-var triple
    gate, and the standalone-CLI :func:`_cli_main` are all exercised
    end-to-end. A ``returncode != 0`` here points at any of the above
    being broken.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.embeddings"],
        capture_output=True,
        text=True,
        env={
            "PATH": _PATH_FOR_SUBPROCESS,
            "BRISTOL_ML_EMBEDDING_STUB": "1",
            "HF_HUB_OFFLINE": "1",  # AC-4 belt-and-braces.
        },
        check=False,
    )
    assert result.returncode == 0, (
        f"python -m bristol_ml.embeddings failed under stub mode "
        f"(returncode={result.returncode}). stderr:\n{result.stderr}"
    )
    # Sanity: the deterministic header must be in stdout.
    assert "=== Stage 15 embedding index ===" in result.stdout
    assert "implementation:     StubEmbedder" in result.stdout
    assert "vector_backend:     numpy" in result.stdout


def test_module_overrides_apply() -> None:
    """The standalone CLI accepts Hydra-style overrides (e.g. switch to stub backend)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bristol_ml.embeddings",
            "embedding.vector_backend=stub",
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": _PATH_FOR_SUBPROCESS,
            "BRISTOL_ML_EMBEDDING_STUB": "1",
            "HF_HUB_OFFLINE": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "vector_backend:     stub" in result.stdout


# Resolve the parent's PATH once at import time — the subprocess
# environment is a fresh dict so `python` and the project's tooling
# only resolve correctly when PATH is forwarded.
import os as _os  # noqa: E402

_PATH_FOR_SUBPROCESS: str = _os.environ.get("PATH", "")
