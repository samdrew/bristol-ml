"""Spec-derived tests for the Stage 11 ``NnTemporalModel`` scaffold — Task T3.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T3 (lines 473-484): the three
  T3 named tests (``test_nn_temporal_is_model_protocol_instance``,
  ``test_nn_temporal_standalone_cli_exits_zero``,
  ``test_nn_temporal_lazy_torch_import_contract``).
- ``docs/plans/active/11-complex-nn.md`` §4 AC-1 (protocol conformance —
  scaffold half), NFR-5 (``python -m bristol_ml.models.nn.temporal``
  exits 0).
- ``src/bristol_ml/models/nn/temporal.py`` module docstring
  (``NnTemporalModel`` surface: ``__init__``, ``metadata``, stubs for
  ``fit`` / ``predict`` / ``save`` / ``load``, standalone CLI).
- ``src/bristol_ml/models/nn/CLAUDE.md`` "PyTorch specifics — lazy import"
  clause: ``import bristol_ml.models.nn`` and
  ``import bristol_ml.models.nn.temporal`` must not pull ``torch`` into
  ``sys.modules``.

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- No ``xfail``, no ``skip``.
- CLI exit-code tests run via subprocess so the ``__main__`` delegation is
  exercised end-to-end; the current process may already have ``torch``
  loaded so subprocess isolation is the only reliable technique for the
  lazy-import invariant.
"""

from __future__ import annotations

import subprocess
import sys

from bristol_ml.models import Model
from bristol_ml.models.nn.temporal import NnTemporalModel
from conf._schemas import NnTemporalConfig

# ---------------------------------------------------------------------------
# 1. test_nn_temporal_is_model_protocol_instance  (plan §Task T3 / AC-1)
# ---------------------------------------------------------------------------


def test_nn_temporal_is_model_protocol_instance() -> None:
    """Guards T3 named test and AC-1 (scaffold half): ``isinstance(NnTemporalModel(cfg), Model)``.

    The ``@runtime_checkable`` structural-subtype check confirms that all
    five required :class:`~bristol_ml.models.protocol.Model` members
    (``fit``, ``predict``, ``save``, ``load``, ``metadata``) are present on
    :class:`~bristol_ml.models.nn.temporal.NnTemporalModel`.  The PEP 544
    caveat — signatures are not verified at runtime — is already covered by
    ``tests/unit/models/test_protocol.py``; this test only verifies the
    presence bar, matching the Stage 10 ``NnMlpModel`` / Stage 4
    ``SarimaxModel`` / ``NaiveModel`` pattern.

    Plan clause: T3 §Task T3 named test / AC-1 scaffold half.
    """
    model = NnTemporalModel(NnTemporalConfig())
    assert isinstance(model, Model), (
        "NnTemporalModel(NnTemporalConfig()) must pass isinstance(model, Model); "
        "the @runtime_checkable protocol check requires all five members: "
        "fit, predict, save, load, metadata (T3 plan / AC-1 scaffold half)."
    )


# ---------------------------------------------------------------------------
# 2. test_nn_temporal_standalone_cli_exits_zero  (plan §Task T3 / NFR-5)
# ---------------------------------------------------------------------------


def test_nn_temporal_standalone_cli_exits_zero() -> None:
    """Guards T3 named test and NFR-5: ``python -m bristol_ml.models.nn.temporal`` exits 0.

    Verifies DESIGN §2.1.1: every module must run standalone via
    ``python -m bristol_ml.<module>``.  The default invocation (no flags)
    must exit with code 0 — the standalone CLI prints the resolved
    ``NnTemporalConfig`` schema without fitting a model.

    Only the exit code is asserted; stdout/stderr content is left for
    smoke-in-dev (T3 plan instruction: "the structural guard is exit code").

    Plan clause: T3 §Task T3 named test / NFR-5 / DESIGN §2.1.1.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.nn.temporal"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        "``python -m bristol_ml.models.nn.temporal`` must exit 0; "
        f"got returncode={result.returncode}.\n"
        f"stdout: {result.stdout[:500]!r}\n"
        f"stderr: {result.stderr[:500]!r}\n"
        "Plan T3 / NFR-5 / DESIGN §2.1.1."
    )


# ---------------------------------------------------------------------------
# 3. test_nn_temporal_lazy_torch_import_contract  (plan §Task T3 / nn CLAUDE.md)
# ---------------------------------------------------------------------------


def test_nn_temporal_lazy_torch_import_contract() -> None:
    """Guards T3 named test: importing the nn package and temporal module must not pull torch.

    Two invariants are covered in a single subprocess probe:

    a. ``import bristol_ml.models.nn`` — the sub-package ``__init__.py``
       uses ``__getattr__`` to lazy-export ``NnTemporalModel``; importing
       the package itself must not materialise ``torch`` in ``sys.modules``.
    b. ``import bristol_ml.models.nn.temporal`` — the module file imports
       ``torch`` inside function bodies only and guards class-level hints
       behind ``if TYPE_CHECKING:``.  Importing the module at the top level
       must also leave ``torch`` out of ``sys.modules``.

    Subprocess isolation is essential here: the calling pytest process has
    already imported ``torch`` (via ``test_sequence_dataset.py`` collection
    or earlier tests), so checking ``sys.modules`` in-process would be a
    vacuous assertion.  A fresh interpreter starts with an empty module
    cache.

    Plan clause: T3 §Task T3 named test / nn CLAUDE.md "Lazy import"
    invariant / Stage 11 NFR-5 lazy-import guard.
    """
    probe = (
        "import sys; "
        "import bristol_ml.models.nn; "
        "import bristol_ml.models.nn.temporal; "
        "assert 'torch' not in sys.modules, ("
        "    'torch must not be in sys.modules after importing bristol_ml.models.nn '"
        "    'and bristol_ml.models.nn.temporal; '"
        "    f'found modules: {[k for k in sys.modules if \"torch\" in k]!r}'"
        ")"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        "Importing ``bristol_ml.models.nn`` and ``bristol_ml.models.nn.temporal`` "
        "must not pull ``torch`` into ``sys.modules``; the subprocess probe failed.\n"
        f"stdout: {result.stdout[:800]!r}\n"
        f"stderr: {result.stderr[:800]!r}\n"
        "Plan T3 / nn CLAUDE.md lazy-import invariant."
    )
