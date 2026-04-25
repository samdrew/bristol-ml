"""Spec-derived tests for :class:`conf._schemas.NnTemporalConfig` — Stage 11 T2.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T2 (lines 468-469): the two
  T2 config named tests (Hydra round-trip, seq_len validator).
- ``conf/_schemas.py`` ``NnTemporalConfig`` (lines 720-864): all field
  defaults, the ``@model_validator`` receptive-field heuristic, and the
  docstring's closed-form formula
  ``receptive_field = 1 + 2*(kernel_size-1)*(2**num_blocks - 1)``.
- ``conf/model/nn_temporal.yaml`` — Hydra group file whose values must
  match the schema defaults exactly.
- The Stage 10 pattern established in
  ``tests/unit/models/test_nn_mlp_scaffold.py``
  (``test_nn_mlp_config_schema_defaults_round_trip_through_hydra``):
  single ``model_dump()`` equality row rather than per-field assertions
  (plan Scope Diff D3).

No production code is modified here.  If any test below fails the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the orchestrator.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import pydantic
import pytest

from bristol_ml.config import load_config
from conf._schemas import NnTemporalConfig

# ===========================================================================
# 4. test_nn_temporal_config_schema_defaults_round_trip_through_hydra
#    Plan §Task T2 named test — guards YAML vs schema drift (Stage 10 pattern).
# ===========================================================================


def test_nn_temporal_config_schema_defaults_round_trip_through_hydra() -> None:
    """Guards plan T2: ``load_config(model=nn_temporal)`` matches ``NnTemporalConfig()``.

    Resolves the Hydra config with the ``model=nn_temporal`` override and
    asserts the resolved ``AppConfig.model`` is an :class:`NnTemporalConfig`
    whose ``model_dump()`` equals that of a fresh ``NnTemporalConfig()``.
    A single ``model_dump()`` equality row covers all fields simultaneously
    (plan Scope Diff D3 — "one round-trip row, not one assertion per field").

    If this test fails, a YAML field in ``conf/model/nn_temporal.yaml`` does
    not match the corresponding Pydantic default — the canonical failure mode
    is a YAML default silently diverging from the schema after a refactor.

    Plan clause: §Task T2 named test
    ``test_nn_temporal_config_schema_defaults_round_trip_through_hydra``.
    Stage 10 pattern: ``test_nn_mlp_config_schema_defaults_round_trip_through_hydra``
    in ``tests/unit/models/test_nn_mlp_scaffold.py``.
    """
    cfg = load_config(overrides=["model=nn_temporal"])

    assert isinstance(cfg.model, NnTemporalConfig), (
        f"load_config(overrides=['model=nn_temporal']).model must be an "
        f"NnTemporalConfig instance; got {type(cfg.model).__name__!r}.  "
        "Check that 'nn_temporal' is registered in the ModelConfig discriminated "
        "union and that conf/model/nn_temporal.yaml sets ``type: nn_temporal`` "
        "(plan §Task T2 Hydra round-trip)."
    )

    expected = NnTemporalConfig()
    resolved_dump = cfg.model.model_dump()
    expected_dump = expected.model_dump()

    assert resolved_dump == expected_dump, (
        "Resolved NnTemporalConfig from Hydra must match NnTemporalConfig() "
        "defaults field-by-field (plan §Task T2 / Scope Diff D3 single-row "
        "round-trip — one assertion covers YAML-vs-schema drift for all fields).\n"
        f"resolved: {resolved_dump!r}\n"
        f"expected: {expected_dump!r}"
    )


# ===========================================================================
# 5. test_nn_temporal_config_rejects_seq_len_smaller_than_receptive_field
#    Plan §Task T2 named test — guards the @model_validator heuristic.
# ===========================================================================


def test_nn_temporal_config_rejects_seq_len_smaller_than_receptive_field() -> None:
    """Guards plan T2: the ``@model_validator`` rejects architecturally mismatched configs.

    **Failure scenario (must raise):**
    ``NnTemporalConfig(seq_len=24)`` keeps the default architecture
    (``num_blocks=8``, ``kernel_size=3``).  The receptive field is::

        1 + 2*(3-1)*(2**8 - 1) = 1 + 2*2*255 = 1021

    The minimum ``seq_len`` under the heuristic
    ``max(2*kernel_size, receptive_field // 8)`` is::

        max(2*3, 1021 // 8) = max(6, 127) = 127

    Since ``24 < 127`` the validator must raise
    :class:`pydantic.ValidationError`.  The error message must name:

    - ``seq_len`` — so the user knows which field to fix.
    - The receptive field number (``1021``) — so the user sees the
      architectural constraint.
    - The minimum threshold (``127``) — so the user knows the corrected
      lower bound.

    **Success scenario (must not raise):**
    ``NnTemporalConfig(seq_len=24, num_blocks=3, kernel_size=3)`` shrinks the
    architecture so the receptive field collapses to::

        1 + 2*(3-1)*(2**3 - 1) = 1 + 2*2*7 = 29

    The minimum becomes ``max(2*3, 29 // 8) = max(6, 3) = 6``, and ``24 >= 6``
    passes without error.

    Formula reference: ``conf/_schemas.py`` ``NnTemporalConfig`` docstring §
    ``num_blocks`` field; plan §1 D2 + R6.

    Plan clause: §Task T2 named test
    ``test_nn_temporal_config_rejects_seq_len_smaller_than_receptive_field``.
    """
    # --- Failure scenario: default 8-block arch + tiny seq_len ---------------
    with pytest.raises(pydantic.ValidationError) as exc_info:
        NnTemporalConfig(seq_len=24)

    error_str = str(exc_info.value)

    # The validator message must mention the offending field.
    assert "seq_len" in error_str, (
        f"ValidationError must name 'seq_len' in the error message; "
        f"got:\n{error_str}\n"
        "(plan §Task T2 validator message contract)."
    )

    # The receptive field at defaults (num_blocks=8, kernel_size=3) is 1021.
    assert "1021" in error_str, (
        f"ValidationError must include the receptive field '1021' so the user "
        f"sees the architectural constraint; got:\n{error_str}\n"
        "(plan §Task T2 validator message — receptive-field number)."
    )

    # The minimum threshold is max(2*3, 1021//8) = max(6, 127) = 127.
    assert "127" in error_str, (
        f"ValidationError must include the minimum threshold '127' so the user "
        f"knows the corrected lower bound; got:\n{error_str}\n"
        "(plan §Task T2 validator message — minimum threshold)."
    )

    # --- Success scenario: smaller arch that fits within seq_len=24 ----------
    # Receptive field: 1 + 2*(3-1)*(2**3 - 1) = 29 → minimum = max(6, 3) = 6.
    # seq_len=24 >= 6 → must not raise.
    cfg_small = NnTemporalConfig(seq_len=24, num_blocks=3, kernel_size=3)
    assert cfg_small.seq_len == 24, (
        f"NnTemporalConfig(seq_len=24, num_blocks=3, kernel_size=3) must succeed "
        f"and preserve seq_len=24; got {cfg_small.seq_len} "
        "(plan §Task T2 validator pass-case)."
    )
    assert cfg_small.num_blocks == 3
    assert cfg_small.kernel_size == 3
