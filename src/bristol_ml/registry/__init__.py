"""Filesystem-backed model registry (Stage 9; Stage 12 migrated to skops).

Four-verb public surface — :func:`save`, :func:`load`, :func:`list_runs`,
:func:`describe` — storing one run per call under
``{DEFAULT_REGISTRY_DIR}/{run_id}/`` with ``artefact/model.skops`` and a
``run.json`` sidecar.  Stage 12 D10 (Ctrl+G reversal) flipped the
canonical artefact filename from ``model.joblib`` to ``model.skops``;
:func:`load` rejects pre-Stage-12 joblib artefacts with a clear error
pointing the operator at the retraining migration path.

The Stage 9 pedagogical moment is a single CLI command that prints a
leaderboard of every registered model ranked by a chosen metric
(intent §Demo moment):

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
Plan: ``docs/plans/completed/09-model-registry.md``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bristol_ml.models.protocol import Model
from bristol_ml.registry._dispatch import class_for_type as _class_for_type
from bristol_ml.registry._dispatch import model_type as _model_type
from bristol_ml.registry._fs import _atomic_write_run, _build_run_id, _run_dir
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


def _validate_run_id(run_id: str) -> None:
    """Reject ``run_id`` values that would escape ``registry_dir`` on join.

    A well-formed Stage 9 run_id is ``{model_name}_{YYYYMMDDTHHMM}`` — a
    single filename fragment with no separators.  Any ``run_id`` that
    contains ``/``, ``\\``, parent-directory markers, or an absolute path
    would make ``registry_dir / run_id`` resolve outside ``registry_dir``;
    the registry accepts local, author-written run_ids at Stage 9, but
    the ``load`` / ``describe`` read paths are public API, so a one-line
    structural guard keeps a malformed caller from triggering surprising
    filesystem reads.
    """
    if run_id != Path(run_id).name or run_id in {"", ".", ".."}:
        raise ValueError(
            f"run_id must be a single path fragment, got {run_id!r}. "
            "The registry expects the `{model_name}_{YYYYMMDDTHHMM}` "
            "form returned by registry.save()."
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

    Reads the ``run.json`` sidecar, dispatches on the ``type`` field to the
    concrete model class, and delegates to that class's ``load`` classmethod
    (:class:`~bristol_ml.models.Model` protocol).  Returns the fitted model.

    ``_NamedLinearModel`` case: per plan D16 (cut) the registry returns a
    base :class:`~bristol_ml.models.linear.LinearModel`.  The dynamic name
    is preserved on the sidecar's ``name`` field for reading but is not
    re-applied to the loaded instance — sidecar name lookup stays the
    source of truth for human-readable identifiers.

    Parameters
    ----------
    run_id:
        The identifier returned by :func:`save` for the run of interest.
    registry_dir:
        Override the default on-disk root.  ``None`` (default) uses
        :data:`DEFAULT_REGISTRY_DIR`.

    Returns
    -------
    Model
        A fitted model instance; same class as the one originally
        registered (except in the ``_NamedLinearModel`` case above).

    Raises
    ------
    FileNotFoundError
        If ``run_id`` is not present under ``registry_dir`` or its
        ``run.json`` / ``artefact/model.skops`` is missing.
    RuntimeError
        If the run directory contains a pre-Stage-12 ``model.joblib``
        artefact (joblib loads are disabled at the registry boundary
        for security under Stage 12 D10 — the operator must retrain
        to migrate to skops).
    ValueError
        If the sidecar's ``type`` field is not one of the four
        registered model types.
    """
    _validate_run_id(run_id)
    registry_root = registry_dir if registry_dir is not None else DEFAULT_REGISTRY_DIR
    run_dir = _run_dir(registry_root, run_id)
    sidecar_path = run_dir / "run.json"
    artefact_dir = run_dir / "artefact"
    artefact_path = artefact_dir / "model.skops"
    legacy_joblib_path = artefact_dir / "model.joblib"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"No registered run at {run_dir!s}; expected a run.json sidecar. "
            "Check the run_id with `python -m bristol_ml.registry list` or "
            "pass registry_dir=..."
        )
    if not artefact_path.is_file():
        # Stage 12 D10: a pre-existing model.joblib artefact is a hard
        # security failure rather than a missing-file case — joblib is
        # an unrestricted unpickler and the registry is now skops-only.
        if legacy_joblib_path.is_file():
            raise RuntimeError(
                f"Registry artefact at {legacy_joblib_path!s} is in the "
                "pre-Stage-12 joblib format; retrain to migrate to skops "
                "(Stage 12 D10 — joblib loads are disabled at the registry "
                "boundary for security)."
            )
        raise FileNotFoundError(
            f"Registered run {run_id!r} is missing its artefact at {artefact_path!s}. "
            "The run directory is in a partial state; try re-saving."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    model_cls = _class_for_type(sidecar["type"])
    return model_cls.load(artefact_path)


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

    Iterates ``registry_dir`` once (via :meth:`pathlib.Path.iterdir`), reads
    every ``run.json`` sidecar in a non-``.tmp_*`` subdirectory, applies the
    exact-match filters (D7), and sorts by the named metric (D8).  Runs
    missing the ``sort_by`` metric are placed last regardless of
    ``ascending``.

    Parameters
    ----------
    target:
        Exact-match filter on the sidecar's ``target`` field (D7).
    model_type:
        Exact-match filter on the sidecar's ``type`` field (D7).
    feature_set:
        Exact-match filter on the sidecar's ``feature_set`` field (D7).
    sort_by:
        Metric name to sort by (D8).  ``"mae"`` is the default — the
        Demo-moment leaderboard is MAE-ascending by default.  ``None``
        returns runs in the underlying filesystem order (undefined).
    ascending:
        When ``True`` (default), best metrics first.
    registry_dir:
        Override the default on-disk root.  ``None`` (default) uses
        :data:`DEFAULT_REGISTRY_DIR`.

    Returns
    -------
    list[dict[str, Any]]
        One sidecar dict per run.  Empty list if the directory does not
        exist or contains no registered runs.
    """
    registry_root = registry_dir if registry_dir is not None else DEFAULT_REGISTRY_DIR
    if not registry_root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for child in registry_root.iterdir():
        # Skip staging directories (D5) and any stray files.
        if not child.is_dir() or child.name.startswith(".tmp_"):
            continue
        sidecar_path = child / "run.json"
        if not sidecar_path.is_file():
            # Partial / corrupt run — skip rather than crash the leaderboard.
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        # Exact-match filters (D7).
        if target is not None and sidecar.get("target") != target:
            continue
        if model_type is not None and sidecar.get("type") != model_type:
            continue
        if feature_set is not None and sidecar.get("feature_set") != feature_set:
            continue
        runs.append(sidecar)

    if sort_by is None:
        return runs
    # Sort by the requested metric; missing keys go last regardless of direction.
    return sorted(runs, key=lambda r: _sort_key(r, sort_by, ascending=ascending))


def _sort_key(sidecar: dict[str, Any], metric_name: str, *, ascending: bool) -> tuple[int, float]:
    """Return a ``(missing_flag, value)`` key: missing metrics always sort last.

    The first element is ``0`` when the metric is present, ``1`` when it
    is missing — so the missing-last rule holds in both ascending and
    descending sorts.  The second element is the value itself (negated
    for descending to match Python's sort-in-one-direction constraint).
    """
    metrics = sidecar.get("metrics", {})
    summary = metrics.get(metric_name)
    if summary is None or "mean" not in summary:
        return (1, 0.0)
    mean = float(summary["mean"])
    if math.isnan(mean):
        return (1, 0.0)
    return (0, mean if ascending else -mean)


def describe(run_id: str, *, registry_dir: Path | None = None) -> dict[str, Any]:
    """Return the full sidecar dict for one registered run.

    Parameters
    ----------
    run_id:
        The identifier returned by :func:`save`.
    registry_dir:
        Override the default on-disk root.  ``None`` (default) uses
        :data:`DEFAULT_REGISTRY_DIR`.

    Returns
    -------
    dict[str, Any]
        The parsed ``run.json`` sidecar.

    Raises
    ------
    FileNotFoundError
        If no run with this ``run_id`` exists under ``registry_dir``.
    """
    _validate_run_id(run_id)
    registry_root = registry_dir if registry_dir is not None else DEFAULT_REGISTRY_DIR
    run_dir = _run_dir(registry_root, run_id)
    sidecar_path = run_dir / "run.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"No registered run at {run_dir!s}; expected a run.json sidecar.")
    return json.loads(sidecar_path.read_text(encoding="utf-8"))
