"""Spec-derived tests for ``bristol_ml.models.sarimax.SarimaxModel`` scaffold.

Every test is derived from:

- ``docs/plans/active/07-sarimax.md`` §Task T3 (lines 302-322): named test list,
  acceptance criteria AC-1, AC-6, AC-10, AC-11.
- ``src/bristol_ml/models/sarimax.py`` inline contracts (constructor, ``metadata``,
  ``results``, ``_cli_main``).
- ``src/bristol_ml/models/CLAUDE.md`` protocol-semantics section ("metadata before
  fit", "Predict-before-fit" guard convention).
- ``conf/_schemas.py`` ``SarimaxConfig`` defaults: ``order=(1,0,1)``,
  ``seasonal_order=(1,1,1,24)``.

No production code is modified here.  If a test below fails the failure indicates
a deviation from the spec — do not weaken the test; surface the failure to the
implementer.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- ``SarimaxConfig()`` default construction throughout (no extra kwargs needed).
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import re

import pytest

from bristol_ml.models import Model, SarimaxModel
from bristol_ml.models.sarimax import _cli_main
from conf._schemas import SarimaxConfig

# ---------------------------------------------------------------------------
# 1. test_sarimax_model_conforms_to_model_protocol (T3 plan §Task T3, AC-1, AC-5)
# ---------------------------------------------------------------------------


def test_sarimax_model_conforms_to_model_protocol() -> None:
    """Guards T3 named test and AC-1/AC-5: isinstance check against Model protocol.

    ``@runtime_checkable`` structural-subtype check confirms that all five
    required protocol members (``fit``, ``predict``, ``save``, ``load``,
    ``metadata``) are present on ``SarimaxModel``.

    Plan clause: T3 plan §Task T3 named test / AC-1 / AC-5.
    """
    config = SarimaxConfig()
    model = SarimaxModel(config)
    assert isinstance(model, Model), (
        "SarimaxModel(SarimaxConfig()) must pass isinstance(model, Model); "
        "the @runtime_checkable protocol check requires all five members: "
        "fit, predict, save, load, metadata (T3 plan / AC-1 / AC-5)."
    )


# ---------------------------------------------------------------------------
# 2. test_sarimax_metadata_name_matches_regex (T3 plan §Task T3, AC-6, AC-10)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_name_matches_regex() -> None:
    """Guards T3 named test: metadata name matches regex and equals the expected value.

    Two assertions are required by the plan:
    1. The ``name`` field matches the ``ModelMetadata`` constraint regex
       ``^[a-z][a-z0-9_.-]*$``.
    2. With default ``order=(1,0,1)`` and ``seasonal_order=(1,1,1,24)`` the
       name must equal ``"sarimax-1-0-1-1-1-1-24"`` (the format produced by
       ``_build_metadata_name``).

    Plan clause: T3 plan §Task T3 named test / AC-6 / AC-10.
    """
    config = SarimaxConfig()  # defaults: order=(1,0,1), seasonal_order=(1,1,1,24)
    model = SarimaxModel(config)
    name = model.metadata.name

    assert re.match(r"^[a-z][a-z0-9_.-]*$", name), (
        f"metadata.name must match ^[a-z][a-z0-9_.-]*$; got {name!r} (T3 plan §Task T3 / AC-6)."
    )
    expected = "sarimax-1-0-1-1-1-1-24"
    assert name == expected, (
        f"metadata.name must equal {expected!r} for default config "
        f"order=(1,0,1) seasonal_order=(1,1,1,24); got {name!r} "
        "(T3 plan §Task T3 / _build_metadata_name contract)."
    )


# ---------------------------------------------------------------------------
# 3. test_sarimax_metadata_fit_utc_none_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_fit_utc_none_before_fit() -> None:
    """Guards T3 named test: unfitted model's metadata.fit_utc is None.

    Before any call to ``fit()`` the ``fit_utc`` field must be ``None``,
    matching the Stage 4 protocol convention documented in models CLAUDE.md
    ("metadata before fit").

    Plan clause: T3 plan §Task T3 named test / models CLAUDE.md
    "metadata before fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    assert model.metadata.fit_utc is None, (
        f"metadata.fit_utc must be None before fit(); "
        f"got {model.metadata.fit_utc!r} (T3 plan §Task T3)."
    )


# ---------------------------------------------------------------------------
# 4. test_sarimax_metadata_feature_columns_empty_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_metadata_feature_columns_empty_before_fit() -> None:
    """Guards T3 named test: unfitted model's metadata.feature_columns is empty tuple.

    Before any call to ``fit()`` the ``feature_columns`` field must be an
    empty tuple ``()``.  Matching the Stage 4 protocol convention documented
    in models CLAUDE.md ("metadata before fit").

    Plan clause: T3 plan §Task T3 named test / models CLAUDE.md
    "metadata before fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    assert model.metadata.feature_columns == (), (
        f"metadata.feature_columns must be () before fit(); "
        f"got {model.metadata.feature_columns!r} (T3 plan §Task T3)."
    )


# ---------------------------------------------------------------------------
# 5. test_sarimax_results_property_raises_before_fit (T3 plan §Task T3)
# ---------------------------------------------------------------------------


def test_sarimax_results_property_raises_before_fit() -> None:
    """Guards T3 named test: accessing .results before fit raises RuntimeError.

    The ``results`` property must raise ``RuntimeError`` (not return ``None``
    or raise ``AttributeError``) when the model has not yet been fit.  The
    error message must mention "fit" so the user understands the pre-condition.

    Plan clause: T3 plan §Task T3 named test / sarimax.py ``results`` property
    docstring / models CLAUDE.md "Predict-before-fit" protocol semantic.
    """
    model = SarimaxModel(SarimaxConfig())
    with pytest.raises(RuntimeError) as exc_info:
        _ = model.results
    assert "fit" in str(exc_info.value).lower(), (
        f"RuntimeError message must mention 'fit'; "
        f"got {str(exc_info.value)!r} (T3 plan §Task T3 / sarimax.py results guard)."
    )


# ---------------------------------------------------------------------------
# 6. test_sarimax_cli_main_returns_zero (T3 plan §Task T3, AC-11)
# ---------------------------------------------------------------------------


def test_sarimax_cli_main_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Guards T3 named test and AC-11: _cli_main([]) returns 0 and prints config schema.

    ``_cli_main([])`` must:
    - Return the integer 0 (DESIGN §2.1.1 standalone module contract).
    - Print text to stdout that contains "SarimaxConfig" (the JSON schema
      header written by the implementation).

    The call resolves the real Hydra config (``model=sarimax`` override) so
    this test validates the full config-resolution path, not just the return value.

    Plan clause: T3 plan §Task T3 named test / DESIGN §2.1.1 / AC-11.
    """
    result = _cli_main([])
    captured = capsys.readouterr()
    assert result == 0, (
        f"_cli_main([]) must return 0; got {result!r} (T3 plan §Task T3 / DESIGN §2.1.1)."
    )
    assert "SarimaxConfig" in captured.out, (
        f"_cli_main([]) stdout must contain 'SarimaxConfig'; "
        f"got {captured.out[:200]!r} (T3 plan §Task T3 / AC-11)."
    )
