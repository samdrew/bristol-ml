"""Filesystem-backed model registry (Stage 9).

Four-verb public surface — :func:`save`, :func:`load`, :func:`list_runs`,
:func:`describe` — storing one run per call under
``{DEFAULT_REGISTRY_DIR}/{run_id}/`` with ``artefact/model.joblib`` and a
``run.json`` sidecar.  The Stage 9 pedagogical moment is a single CLI
command that prints a leaderboard of every registered model ranked by a
chosen metric (intent §Demo moment):

.. code-block:: bash

    python -m bristol_ml.registry list
    python -m bristol_ml.registry describe linear-ols-weather-only_20260423T1430

AC-1 caps the public surface at four verbs — *"if it grows past that, the
design is drifting"*.  The structural test
``test_registry_public_surface_does_not_exceed_four_callables`` enforces
the cap.  ``list_runs`` is the exported symbol rather than ``list`` to
avoid shadowing the Python builtin.

The registry is filesystem-only at Stage 9; a hosted MLflow / W&B
implementation is deferred (intent §Out of scope) but the
``mlflow.pyfunc`` graduation adapter is documented in
``docs/architecture/layers/registry.md`` and exercised by a test-only
round-trip test (plan D10).

Layer doc: ``docs/architecture/layers/registry.md``.
Plan: ``docs/plans/active/09-model-registry.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bristol_ml.models.protocol import Model

#: AC-1: four verbs, exported deliberately in noqa-suppressed order so the
#: public surface reads in verb-chronological order (save → load → list →
#: describe) rather than isort-alphabetical.  The structural test
#: ``test_registry_public_surface_does_not_exceed_four_callables`` asserts
#: the *set* matches; readers scan the tuple.
__all__ = ("save", "load", "list_runs", "describe")  # noqa: RUF022

#: Default on-disk root for the registry.  Gitignored via the repo-wide
#: ``data/*`` rule; override by passing ``registry_dir=`` to any of the
#: four verbs, or via the ``--registry-dir`` flag on the
#: ``python -m bristol_ml.registry`` CLI.
DEFAULT_REGISTRY_DIR = Path("data/registry")


def save(
    model: Model,
    metrics_df: pd.DataFrame,
    *,
    feature_set: str,
    target: str,
    registry_dir: Path | None = None,
) -> str:
    """Register a fitted model.

    Stub — Stage 9 T2 implements the body.
    """
    raise NotImplementedError("Stage 9 T2 implements registry.save.")


def load(run_id: str, *, registry_dir: Path | None = None) -> Model:
    """Load a registered run by ``run_id``.

    Stub — Stage 9 T3 implements the body.
    """
    raise NotImplementedError("Stage 9 T3 implements registry.load.")


def list_runs(
    *,
    target: str | None = None,
    model_type: str | None = None,
    feature_set: str | None = None,
    sort_by: str | None = "mae",
    ascending: bool = True,
    registry_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """List registered runs with optional filters and sort.

    Stub — Stage 9 T4 implements the body.
    """
    raise NotImplementedError("Stage 9 T4 implements registry.list_runs.")


def describe(run_id: str, *, registry_dir: Path | None = None) -> dict[str, Any]:
    """Return the full sidecar dict for one registered run.

    Stub — Stage 9 T4 implements the body.
    """
    raise NotImplementedError("Stage 9 T4 implements registry.describe.")
