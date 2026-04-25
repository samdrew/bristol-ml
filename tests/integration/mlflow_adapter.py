"""Test-only MLflow PyFunc adapter for ``bristol_ml.registry`` (Stage 9 D10).

The Stage 9 plan's D10 decision upgrades the MLflow graduation story
from "document the adapter" to "document **and** test the adapter".
This module is the test-support code that satisfies the second half —
it is deliberately under ``tests/integration/`` rather than ``src/``
so that:

- The public surface of :mod:`bristol_ml.registry` stays at exactly
  four verbs (plan AC-1 / D12).
- MLflow is a *test-only* dependency under the ``dev`` dependency
  group; a fresh production install does not pull it in.

The adapter's job is proof-by-construction: a registered run can be
re-packaged as an ``mlflow.pyfunc`` flavour artefact with a small
:class:`mlflow.pyfunc.PythonModel` subclass that delegates to
:func:`bristol_ml.registry.load`.  If a future MLflow release changes
the PyFunc save/load contract, the round-trip test in
``tests/integration/test_registry_mlflow_adapter.py`` fails and the
drift is surfaced at the dep-bump boundary rather than silently in
production.

Nothing under :mod:`bristol_ml` imports this module.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow.pyfunc

if TYPE_CHECKING:  # pragma: no cover — typing only
    import pandas as pd

    from bristol_ml.models.protocol import Model

__all__ = ("RegistryPyfuncAdapter", "package_run_as_pyfunc")


class RegistryPyfuncAdapter(mlflow.pyfunc.PythonModel):
    """Wrap a registered :mod:`bristol_ml.registry` run as an MLflow PyFunc.

    The adapter reads the registry run directory from
    ``context.artifacts["registry_run"]`` inside :meth:`load_context` and
    delegates to the registry's type-dispatched loader
    (:func:`bristol_ml.registry.load`) so every registered model family
    can graduate without a per-family PyFunc subclass.

    :meth:`predict` forwards the caller's input to the wrapped model's
    ``predict`` method unchanged — the PyFunc envelope adds no feature
    engineering, no column renaming, and no type coercion.  Reading the
    MLflow-PyFunc output therefore requires the same feature columns the
    original model was fitted on.
    """

    def __init__(self) -> None:
        # Initialised to ``None`` so that if MLflow ever calls ``predict``
        # before a successful ``load_context`` (e.g. a deserialisation
        # failure path) the error surfaces as an explicit ``RuntimeError``
        # naming the lifecycle, not a bare ``AttributeError``.
        self._model: Model | None = None

    def load_context(self, context: Any) -> None:
        """Load the wrapped model from the registry run directory.

        MLflow passes the resolved artefact path via
        ``context.artifacts["registry_run"]`` — this is the registry
        ``run_dir`` (containing ``run.json`` + ``artefact/model.skops``,
        post-Stage-12 D10 — the canonical artefact filename was
        ``model.joblib`` in Stages 9-11) that
        :func:`package_run_as_pyfunc` staged at packaging time.
        """
        # Local import: keep the registry off the import graph of the
        # adapter module itself so plain ``import mlflow.pyfunc`` does
        # not pull in bristol_ml at MLflow model-registry scan time.
        from bristol_ml import registry

        run_dir = Path(context.artifacts["registry_run"])
        registry_dir = run_dir.parent
        run_id = run_dir.name
        self._model = registry.load(run_id, registry_dir=registry_dir)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Forward ``model_input`` to the wrapped model's ``predict``.

        MLflow's PyFunc contract passes the `context` argument to every
        call; we ignore it (it is only populated during loading) and
        delegate to the wrapped ``Model.predict`` unchanged.
        """
        del context, params  # unused — MLflow-API shape
        if self._model is None:
            raise RuntimeError(
                "RegistryPyfuncAdapter.predict called before load_context — "
                "MLflow did not populate context.artifacts['registry_run']."
            )
        return self._model.predict(model_input)


def package_run_as_pyfunc(run_id: str, dst: Path, *, registry_dir: Path) -> None:
    """Package a registered run as an ``mlflow.pyfunc`` flavour artefact.

    Parameters
    ----------
    run_id:
        The registry run to package; must exist under ``registry_dir``.
    dst:
        Destination path for the MLflow PyFunc artefact.  MLflow
        requires the destination to not already exist; we clear it first
        so callers can reuse a ``tmp_path`` across re-runs.
    registry_dir:
        Registry root containing the ``run_id`` directory.

    The produced MLflow artefact bundles the full registry run directory
    (both the sidecar and the skops artefact) into its ``artifacts/``
    tree so :func:`mlflow.pyfunc.load_model` reconstructs the original
    via :func:`bristol_ml.registry.load` without needing the original
    registry directory to still exist on disk.  Stage 12 D10 (Ctrl+G
    reversal) flipped the canonical artefact filename from
    ``model.joblib`` to ``model.skops``; the bundling logic is
    artefact-agnostic so this comment is the only place the filename
    is named.
    """
    if dst.exists():
        # mlflow.pyfunc.save_model refuses to overwrite an existing path.
        shutil.rmtree(dst)
    run_dir = registry_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot package run {run_id!r}: no directory at {run_dir!s}. "
            "Call bristol_ml.registry.save(...) first."
        )

    mlflow.pyfunc.save_model(
        path=str(dst),
        python_model=RegistryPyfuncAdapter(),
        artifacts={"registry_run": str(run_dir)},
    )
