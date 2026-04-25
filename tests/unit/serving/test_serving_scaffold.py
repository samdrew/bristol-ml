"""Spec-derived tests for the Stage 12 T1 serving package scaffold.

Every test here is derived from:

- ``docs/plans/active/12-serving.md`` §6 T1 named test:
  ``test_serving_module_imports_without_torch`` — import-graph guard.
- ``docs/plans/active/12-serving.md`` §5 module structure (the five
  files: ``__init__.py``, ``__main__.py``, ``app.py``, ``schemas.py``,
  ``CLAUDE.md``).
- ``docs/plans/active/12-serving.md`` §1 NFR-2: ``python -m
  bristol_ml.serving --help`` exits 0.

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test;
surface the failure to the implementer.
"""

from __future__ import annotations

import subprocess
import sys


def test_serving_module_imports_without_torch() -> None:
    """Guards Stage 12 T1 named test: ``bristol_ml.serving`` does not pull torch.

    The plan §6 T1 names this test as the import-graph guard.  Stage
    11 established the lazy-torch-import discipline for the NN
    sub-layer; Stage 12 has no torch dep at all (it only consumes
    `Model.predict` through the registry, which goes through each
    family's own torch-aware load path on demand).  This test
    asserts ``import bristol_ml.serving`` does not import ``torch``
    — a guard-by-construction since the only torch consumers are
    the NN families, which the serving layer reaches through the
    registry only when a request actually resolves to an NN run.

    The test runs in a fresh subprocess so the import graph is not
    polluted by anything the rest of the test session has already
    pulled in.
    """
    code = (
        "import sys\n"
        "import bristol_ml.serving  # noqa: F401\n"
        "assert 'torch' not in sys.modules, (\n"
        "    'bristol_ml.serving must not import torch; '\n"
        "    'found torch in sys.modules after import.'\n"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_serving_cli_help_exits_zero() -> None:
    """Guards Stage 12 NFR-2: ``python -m bristol_ml.serving --help`` exits 0
    and surfaces the resolved ``ServingConfig`` schema.

    The plan §6 T8 names this test for the CLI surface.  Two
    behaviours are load-bearing:

    1.  ``--help`` exits zero — no Hydra resolve, no uvicorn import,
        no registry I/O happens during ``--help`` (the imports are
        deferred inside :func:`bristol_ml.serving.__main__._cli_main`),
        so a stale registry or a missing config file does not turn
        ``--help`` into an error path.
    2.  The output surfaces the resolved ``ServingConfig`` defaults
        (``data/registry``, ``127.0.0.1``, ``8000``) inline thanks to
        :class:`argparse.ArgumentDefaultsHelpFormatter`, so the demo
        facilitator can read the schema without opening
        ``conf/serving/default.yaml``.

    Subprocess invocation (rather than in-process call) reproduces the
    way the user actually triggers the CLI.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.serving", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`python -m bristol_ml.serving --help` exited {result.returncode!r}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "bristol_ml" in result.stdout, (
        f"--help output should mention bristol_ml; got {result.stdout!r}"
    )
    # NFR-2: --help must surface the resolved ServingConfig schema.
    # ArgumentDefaultsHelpFormatter prints the defaults inline; we
    # assert each expected default appears so a future schema rename
    # without a flag-default refresh fails this test.
    for expected in ("data/registry", "127.0.0.1", "8000"):
        assert expected in result.stdout, (
            f"--help output should surface ServingConfig default {expected!r}; "
            f"got {result.stdout!r}"
        )


def test_serving_package_exposes_build_app() -> None:
    """Guards Stage 12 §5 surface: ``build_app`` is the named entry point.

    The plan §5 module structure lists ``__init__.py`` exporting
    ``build_app``.  At T1 the stub is a placeholder that raises
    :class:`NotImplementedError`; T7 fills in the real implementation.
    The presence of the symbol is what this test guards.
    """
    import bristol_ml.serving as serving

    assert hasattr(serving, "build_app"), (
        "bristol_ml.serving must expose a `build_app` symbol (Stage 12 §5)."
    )
    assert callable(serving.build_app), "bristol_ml.serving.build_app must be callable."
