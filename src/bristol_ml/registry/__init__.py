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

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bristol_ml.models.protocol import Model
from bristol_ml.registry._dispatch import model_type as _model_type
from bristol_ml.registry._fs import _atomic_write_run, _build_run_id
from bristol_ml.registry._git import _git_sha_or_none
from bristol_ml.registry._schema import MetricSummary, SidecarFields

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

#: Columns in the per-fold metrics DataFrame that are *not* metrics — the
#: Stage 6 harness pins this layout (``fold_index``, ``train_end``,
#: ``test_start``, ``test_end``) and every other column is a float metric
#: keyed by the metric callable's ``__name__``.
_NON_METRIC_COLUMNS: frozenset[str] = frozenset(
    {"fold_index", "train_end", "test_start", "test_end"}
)


def save(
    model: Model,
    metrics_df: pd.DataFrame,
    *,
    feature_set: str,
    target: str,
    registry_dir: Path | None = None,
) -> str:
    """Register a fitted model under ``{metadata.name}_{YYYYMMDDTHHMM}``.

    Writes the model's artefact (via the protocol's :meth:`Model.save`) and
    a ``run.json`` sidecar beneath ``registry_dir/{run_id}/``.  The write
    is atomic (temp-dir-then-``os.replace``) so a crash mid-save cannot
    corrupt an existing run.  Returns the ``run_id`` so the caller can
    feed it to :func:`describe` or surface it in a training-CLI log line.

    Parameters
    ----------
    model:
        Any fitted :class:`~bristol_ml.models.Model` implementor.  Must
        have ``metadata.fit_utc is not None``; an unfitted model is
        rejected.
    metrics_df:
        The per-fold metrics DataFrame returned by
        :func:`bristol_ml.evaluation.harness.evaluate` — one row per
        fold with columns ``fold_index``, ``train_end``, ``test_start``,
        ``test_end``, plus one float column per metric keyed by metric
        name.
    feature_set:
        Human-readable label for the feature selection used at fit
        (e.g. ``"weather_only"``, ``"weather_calendar"``).  AC-3: the
        registry does not infer this — it must be passed explicitly.
    target:
        Target-column name (e.g. ``"nd_mw"``).  AC-3: passed explicitly.
    registry_dir:
        Override the default on-disk root.  ``None`` (default) uses
        :data:`DEFAULT_REGISTRY_DIR`.

    Returns
    -------
    str
        The ``run_id`` under which the run was registered.

    Raises
    ------
    RuntimeError
        If ``model.metadata.fit_utc is None`` — unfitted models cannot
        be registered.
    TypeError
        If ``model``'s class is not one of the four registered types;
        see :mod:`bristol_ml.registry._dispatch`.  Also raised (by
        Python) when ``feature_set`` or ``target`` is omitted — AC-3
        second half.
    """
    metadata = model.metadata
    if metadata.fit_utc is None:
        raise RuntimeError(
            "registry.save requires a fitted model; got metadata.fit_utc=None. "
            "Call model.fit(...) before registering."
        )

    registry_root = registry_dir if registry_dir is not None else DEFAULT_REGISTRY_DIR
    run_id = _build_run_id(metadata.name, metadata.fit_utc)
    type_str = _model_type(model)
    git_sha = _git_sha_or_none()
    registered_at = datetime.now(UTC)

    metrics = _summarise_metrics(metrics_df)
    sidecar: SidecarFields = {
        "run_id": run_id,
        "name": metadata.name,
        "type": type_str,
        "feature_set": feature_set,
        "target": target,
        "feature_columns": list(metadata.feature_columns),
        "fit_utc": metadata.fit_utc.isoformat(),
        "git_sha": git_sha,
        "hyperparameters": dict(metadata.hyperparameters),
        "metrics": metrics,
        "registered_at_utc": registered_at.isoformat(),
    }

    _atomic_write_run(
        registry_root,
        run_id,
        artefact_writer=model.save,
        sidecar=sidecar,
    )
    return run_id


def _summarise_metrics(metrics_df: pd.DataFrame) -> dict[str, MetricSummary]:
    """Return ``{metric_name: MetricSummary}`` from a per-fold metrics frame.

    Non-metric columns (``fold_index``, the three timestamp columns) are
    skipped.  Each surviving column is summarised as mean + std + the
    raw per-fold list.  ``NaN`` per-fold values survive round-trip via
    ``json.dumps(..., allow_nan=True)`` (plan D4).
    """
    out: dict[str, MetricSummary] = {}
    for col in metrics_df.columns:
        if col in _NON_METRIC_COLUMNS:
            continue
        series = metrics_df[col]
        per_fold = [float(v) for v in series.to_numpy()]
        out[col] = MetricSummary(
            mean=float(series.mean()),
            std=float(series.std()),
            per_fold=per_fold,
        )
    return out


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
