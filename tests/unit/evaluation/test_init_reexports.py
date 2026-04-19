"""Spec-derived tests for the ``bristol_ml.evaluation`` lazy re-export namespace.

Covers plan T5 tests 20-22 (metric symbols and ``__all__`` contract).
Complements the splitter re-export tests already in
``tests/unit/evaluation/test_splitter.py::TestNamespaceReExports``.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T5 tests 20-22.
- ``src/bristol_ml/evaluation/__init__.py`` lazy ``__getattr__`` contract.
- DESIGN §2.1.1 (modules must expose a discoverable public API).

No production code is modified here.

Conventions
-----------
- British English in docstrings.
- Each docstring cites the plan T5 test number and clause it guards.
"""

from __future__ import annotations

import pytest

# Confirm the namespace is importable; skip if the package is absent.
pytest.importorskip("bristol_ml.evaluation")


# ---------------------------------------------------------------------------
# Test 20 — Metric symbols resolve from the namespace (Plan T5 test 20)
# ---------------------------------------------------------------------------


def test_metrics_importable_from_namespace() -> None:
    """Guards plan T5 test 20: metric symbols re-exported from evaluation namespace.

    ``from bristol_ml.evaluation import mae, mape, rmse, wape,
    METRIC_REGISTRY, MetricFn`` must all resolve and must be the same
    objects as those defined in ``bristol_ml.evaluation.metrics``.

    The lazy ``__getattr__`` loader routes metric names via
    ``_METRIC_NAMES``; this test confirms each entry resolves correctly
    and is an identity match with the submodule originals.
    """
    from bristol_ml.evaluation import METRIC_REGISTRY, MetricFn, mae, mape, rmse, wape
    from bristol_ml.evaluation import metrics as _metrics

    assert mae is _metrics.mae, "Namespace-level 'mae' must be the same object as metrics.mae."
    assert mape is _metrics.mape, "Namespace-level 'mape' must be the same object as metrics.mape."
    assert rmse is _metrics.rmse, "Namespace-level 'rmse' must be the same object as metrics.rmse."
    assert wape is _metrics.wape, "Namespace-level 'wape' must be the same object as metrics.wape."
    assert METRIC_REGISTRY is _metrics.METRIC_REGISTRY, (
        "Namespace-level 'METRIC_REGISTRY' must be the same object as metrics.METRIC_REGISTRY."
    )
    assert MetricFn is _metrics.MetricFn, (
        "Namespace-level 'MetricFn' must be the same object as metrics.MetricFn."
    )


# ---------------------------------------------------------------------------
# Test 21 — __all__ includes metric entries and pre-existing splitter entries
# ---------------------------------------------------------------------------


def test_all_exposes_metrics() -> None:
    """Guards plan T5 test 21: evaluation namespace ``__all__`` includes metric symbols.

    ``__all__`` is the declared public surface that tooling (Sphinx autodoc,
    star-imports, IDEs) relies on.  After Stage 4 T5 it must list all four
    metric function names, ``METRIC_REGISTRY``, ``MetricFn``, and also
    retain the Stage 3 splitter symbols.
    """
    import bristol_ml.evaluation as _evaluation

    expected_metric_symbols = {"mae", "mape", "rmse", "wape", "METRIC_REGISTRY", "MetricFn"}
    expected_splitter_symbols = {"rolling_origin_split", "rolling_origin_split_from_config"}

    for sym in expected_metric_symbols | expected_splitter_symbols:
        assert sym in _evaluation.__all__, (
            f"'{sym}' must be in bristol_ml.evaluation.__all__; "
            f"got __all__={sorted(_evaluation.__all__)!r}."
        )


# ---------------------------------------------------------------------------
# Test 22 — Unknown attribute raises AttributeError (Plan T5 test 22)
# ---------------------------------------------------------------------------


def test_getattr_raises_attribute_error_for_unknown_name() -> None:
    """Guards plan T5 test 22: lazy loader falls through to AttributeError.

    The ``__getattr__`` hook must only resolve symbols declared in
    ``_SPLITTER_NAMES`` and ``_METRIC_NAMES``; any other name must raise
    ``AttributeError`` with the standard Python message form so callers
    get a clear diagnostic rather than a silent ``None``.

    Re-asserted here (even though the splitter tests cover the same hook)
    because Stage 4 restructured the lazy loader to handle metric names
    as a second dispatch branch; both branches must still fall through
    to the error path for unknown names.
    """
    import bristol_ml.evaluation as _evaluation

    with pytest.raises(AttributeError, match="has no attribute 'definitely_not_a_symbol'"):
        getattr(_evaluation, "definitely_not_a_symbol")  # noqa: B009
