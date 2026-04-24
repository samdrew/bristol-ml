"""Spec-derived tests for the Stage 10 ``NnMlpModel`` scaffold — Task T1.

Every test is derived from:

- ``docs/plans/active/10-simple-nn.md`` §Task T1 (lines 346-362): the three
  T1 named tests (``test_nn_mlp_is_model_protocol_instance``,
  ``test_nn_mlp_config_schema_defaults_round_trip_through_hydra``,
  ``test_nn_mlp_standalone_cli_exits_zero``).
- ``docs/plans/active/10-simple-nn.md`` §4 AC-1 (protocol conformance — scaffold
  half), NFR-6 (``python -m bristol_ml.models.nn.mlp --help`` exits 0).
- ``src/bristol_ml/models/nn/mlp.py`` module docstring (``NnMlpModel`` surface:
  ``__init__``, ``metadata``, stubs for ``fit`` / ``predict`` / ``save`` /
  ``load``, standalone CLI).
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section
  ("metadata before fit": ``fit_utc=None``, ``feature_columns=()`` are
  permissible pre-fit states).
- ``conf/_schemas.py`` ``NnMlpConfig`` defaults (single source of truth:
  ``hidden_sizes=[128]``, ``activation="relu"``, ``dropout=0.0``,
  ``learning_rate=1e-3``, ``weight_decay=0.0``, ``batch_size=32``,
  ``max_epochs=100``, ``patience=10``, ``seed=None``, ``device="auto"``,
  ``target_column="nd_mw"``, ``feature_columns=None``).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- ``NnMlpConfig()`` default construction throughout (T1 scaffold tests do
  not exercise fit-time behaviour — Tasks T2+ own that coverage).
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import subprocess
import sys

from bristol_ml.config import load_config
from bristol_ml.models import Model
from bristol_ml.models.nn.mlp import NnMlpModel
from conf._schemas import NnMlpConfig

# ---------------------------------------------------------------------------
# 1. test_nn_mlp_is_model_protocol_instance  (plan §Task T1 named test / AC-1)
# ---------------------------------------------------------------------------


def test_nn_mlp_is_model_protocol_instance() -> None:
    """Guards T1 named test and AC-1 (scaffold half): ``isinstance(NnMlpModel(cfg), Model)``.

    The ``@runtime_checkable`` structural-subtype check confirms that all
    five required :class:`~bristol_ml.models.protocol.Model` members
    (``fit``, ``predict``, ``save``, ``load``, ``metadata``) are present on
    :class:`~bristol_ml.models.nn.mlp.NnMlpModel`.  The PEP 544 caveat —
    that signatures are not verified at runtime — is already covered by
    ``tests/unit/models/test_protocol.py``; this test only verifies the
    presence bar, matching the Stage 4 ``SarimaxModel`` / ``NaiveModel``
    pattern.

    Plan clause: T1 §Task T1 named test / AC-1 scaffold half.
    """
    model = NnMlpModel(NnMlpConfig())
    assert isinstance(model, Model), (
        "NnMlpModel(NnMlpConfig()) must pass isinstance(model, Model); "
        "the @runtime_checkable protocol check requires all five members: "
        "fit, predict, save, load, metadata (T1 plan / AC-1 scaffold half)."
    )


# ---------------------------------------------------------------------------
# 2. test_nn_mlp_config_schema_defaults_round_trip_through_hydra
#    (plan §Task T1 named test / Scope Diff D3 one-row-not-per-field)
# ---------------------------------------------------------------------------


def test_nn_mlp_config_schema_defaults_round_trip_through_hydra() -> None:
    """Guards T1 named test: ``load_config(model=nn_mlp)`` matches ``NnMlpConfig()``.

    Resolves the Hydra config with ``model=nn_mlp`` and asserts the resolved
    ``AppConfig.model`` is an :class:`NnMlpConfig` whose every field equals
    the corresponding default on a fresh ``NnMlpConfig()`` — including
    ``device == "auto"``.  The Scope Diff D3 clause specifies a single
    round-trip row rather than one assertion per field; this test honours
    that instruction by comparing the two instances via
    ``model_dump()`` equality, which is tuple/list-aware.

    Plan clause: T1 §Task T1 named test / Scope Diff D3.
    """
    cfg = load_config(overrides=["model=nn_mlp"])

    assert isinstance(cfg.model, NnMlpConfig), (
        f"load_config(overrides=['model=nn_mlp']).model must be an NnMlpConfig "
        f"instance; got {type(cfg.model).__name__!r} (T1 / Hydra discriminated "
        "union dispatch)."
    )

    expected = NnMlpConfig()
    resolved_dump = cfg.model.model_dump()
    expected_dump = expected.model_dump()

    assert resolved_dump == expected_dump, (
        "Resolved NnMlpConfig must match NnMlpConfig() defaults field-by-field "
        "(T1 / Scope Diff D3 single-row round-trip).\n"
        f"resolved: {resolved_dump!r}\n"
        f"expected: {expected_dump!r}"
    )

    # Spot-check the device default because the round-trip row explicitly
    # names it, and because ``device`` is the D11 re-opened decision whose
    # default "auto" is the load-bearing NFR-1 contract (auto-resolves to
    # CUDA > MPS > CPU at fit time).
    assert cfg.model.device == "auto", (
        f"NnMlpConfig.device default must be 'auto' (plan D11); got {cfg.model.device!r}."
    )


# ---------------------------------------------------------------------------
# 3. test_nn_mlp_standalone_cli_exits_zero  (plan §Task T1 named test / NFR-6)
# ---------------------------------------------------------------------------


def test_nn_mlp_standalone_cli_exits_zero() -> None:
    """Guards T1 named test and NFR-6: ``python -m bristol_ml.models.nn.mlp --help`` exits 0.

    Verifies DESIGN §2.1.1: every module must run standalone via
    ``python -m bristol_ml.<module>``.  The ``--help`` flag must exit with
    code 0 and argparse must write a usage line to stdout.

    NFR-6 also requires the ``python -m bristol_ml.models.nn`` package alias
    to work (delegated via ``nn/__main__.py``); the alias is covered by
    ``test_nn_mlp_standalone_cli_package_alias_exits_zero`` below so a
    regression in one entry point does not silently break the other.

    Plan clause: T1 §Task T1 named test / NFR-6 / DESIGN §2.1.1.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.nn.mlp", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "``python -m bristol_ml.models.nn.mlp --help`` must exit 0; "
        f"got returncode={result.returncode}.\n"
        f"stdout: {result.stdout[:500]!r}\n"
        f"stderr: {result.stderr[:500]!r}\n"
        "Plan T1 / NFR-6 / DESIGN §2.1.1."
    )
    assert "usage" in result.stdout.lower(), (
        "``--help`` output must contain 'usage' (case-insensitive); "
        f"got stdout={result.stdout!r}.  Plan T1 / NFR-6 / DESIGN §2.1.1."
    )


# ---------------------------------------------------------------------------
# 4. test_nn_mlp_standalone_cli_package_alias_exits_zero  (plan NFR-6 second half)
# ---------------------------------------------------------------------------


def test_nn_mlp_standalone_cli_package_alias_exits_zero() -> None:
    """Guards NFR-6: ``python -m bristol_ml.models.nn --help`` (package alias) exits 0.

    The package-level alias delegates to :func:`bristol_ml.models.nn.mlp._cli_main`
    via ``nn/__main__.py`` so notebooks and meetup demos have a shorter
    entry point.  This test runs the alias in a subprocess (not an in-process
    ``_cli_main([])`` call) so the ``__main__.py`` delegation is actually
    exercised — an in-process call would bypass the file entirely.

    Plan clause: T1 §Task T1 Files / NFR-6 / ``nn/__main__.py`` delegation.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.models.nn", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "``python -m bristol_ml.models.nn --help`` must exit 0; "
        f"got returncode={result.returncode}.\n"
        f"stdout: {result.stdout[:500]!r}\n"
        f"stderr: {result.stderr[:500]!r}\n"
        "Plan T1 / NFR-6 (package-alias delegation)."
    )
    assert "usage" in result.stdout.lower(), (
        "``--help`` output must contain 'usage' (case-insensitive); "
        f"got stdout={result.stdout!r}.  Plan T1 / NFR-6."
    )


# ---------------------------------------------------------------------------
# 5. test_nn_mlp_unfitted_metadata_matches_pre_fit_contract
#    (plan §5 metadata property; models CLAUDE.md "metadata before fit")
# ---------------------------------------------------------------------------


def test_nn_mlp_unfitted_metadata_matches_pre_fit_contract() -> None:
    """Guards the "metadata before fit" contract from models CLAUDE.md.

    Before :meth:`NnMlpModel.fit` is called:

    - ``metadata.fit_utc`` must be ``None``.
    - ``metadata.feature_columns`` must be the empty tuple.
    - ``metadata.name`` must match the ``ModelMetadata.name`` regex
      (``^[a-z][a-z0-9_.-]*$``) and embed the default architecture
      (``nn-mlp-relu-128``).
    - ``metadata.hyperparameters`` must carry the config-derived keys
      (``learning_rate``, ``device``, etc.) but MUST NOT carry the
      fit-time keys (``device_resolved``, ``seed_used``, ``best_epoch``).

    This test does not appear in the plan's T1 named list but is
    scaffold-protocol table stakes — it pins the Stage 4 protocol
    semantic that downstream stages rely on (the registry reads
    ``fit_utc`` as the "was this fitted?" signal).

    Plan clause: T1 scaffold contract / models CLAUDE.md "metadata before fit".
    """
    model = NnMlpModel(NnMlpConfig())
    md = model.metadata

    assert md.fit_utc is None, f"metadata.fit_utc must be None before fit(); got {md.fit_utc!r}."
    assert md.feature_columns == (), (
        f"metadata.feature_columns must be () before fit(); got {md.feature_columns!r}."
    )

    # Name regex + format (architecture encoding):
    import re

    assert re.match(r"^[a-z][a-z0-9_.-]*$", md.name), (
        f"metadata.name must match ^[a-z][a-z0-9_.-]*$; got {md.name!r}."
    )
    assert md.name == "nn-mlp-relu-128", (
        f"metadata.name must be 'nn-mlp-relu-128' for default config; got {md.name!r}."
    )

    # Hyperparameter bag — config-derived keys present, fit-time keys absent.
    hp = md.hyperparameters
    required_before_fit = {
        "target_column",
        "hidden_sizes",
        "activation",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "device",
    }
    for key in required_before_fit:
        assert key in hp, (
            f"metadata.hyperparameters must carry {key!r} before fit(); "
            f"got keys {set(hp.keys())!r}."
        )

    forbidden_before_fit = {"device_resolved", "seed_used", "best_epoch"}
    for key in forbidden_before_fit:
        assert key not in hp, (
            f"metadata.hyperparameters must NOT carry {key!r} before fit(); "
            f"it should only appear after a successful fit().  "
            f"Got keys {set(hp.keys())!r}."
        )
