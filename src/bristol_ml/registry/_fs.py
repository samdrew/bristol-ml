"""On-disk layout helpers for the registry (Stage 9 D1 / D3 / D5).

Three private building blocks:

- :func:`_build_run_id` — minute-precision ``{model_name}_{YYYYMMDDTHHMM}``
  identifier per plan D3.  Sortable lexicographically so the default
  ``list_runs`` order comes for free.
- :func:`_run_dir` — resolves ``(registry_dir, run_id) → Path``.
- :func:`_atomic_write_run` — stages the artefact + sidecar under a
  ``.tmp_{uuid}/`` sibling of the final run directory and then calls
  :func:`os.replace` to rename atomically (plan D5).  Mirrors the
  ingestion-layer idiom at
  ``src/bristol_ml/ingestion/_common.py::_atomic_write`` and the
  joblib variant at ``src/bristol_ml/models/io.py::save_joblib``.

The same-minute collision path (D2 last-write-wins) first removes the
existing run directory before the rename; this is a deliberate, documented
non-atomic window that matches D2 semantics ("two trainings in the same
minute overwrite"). See §8 R7 of the plan.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from bristol_ml.registry._schema import SidecarFields


def _build_run_id(model_name: str, fit_utc: datetime) -> str:
    """Return ``{model_name}_{YYYYMMDDTHHMM}`` (UTC, minute precision).

    Parameters
    ----------
    model_name:
        The ``ModelMetadata.name`` of the fitted model.  Must already conform
        to the ``^[a-z][a-z0-9_.-]*$`` pattern Pydantic enforces on
        ``ModelMetadata.name``; this helper does not re-validate.
    fit_utc:
        The tz-aware UTC timestamp of fit.  Naive datetimes are rejected —
        the registry never stores ambiguous wall-clock times.  Non-UTC
        tz-aware datetimes are silently converted to UTC before
        formatting.

    Raises
    ------
    ValueError
        If ``fit_utc`` is naive (``tzinfo is None``).
    """
    if fit_utc.tzinfo is None:
        raise ValueError(f"fit_utc must be tz-aware (UTC); got naive datetime {fit_utc!r}.")
    utc = fit_utc.astimezone(UTC)
    stamp = utc.strftime("%Y%m%dT%H%M")
    return f"{model_name}_{stamp}"


def _run_dir(registry_dir: Path, run_id: str) -> Path:
    """Return the on-disk directory for one registered run."""
    return registry_dir / run_id


def _atomic_write_run(
    registry_dir: Path,
    run_id: str,
    *,
    artefact_writer: Callable[[Path], None],
    sidecar: SidecarFields,
) -> Path:
    """Write one registered run atomically.

    Steps:

    1. Create ``registry_dir`` if needed.
    2. Create a staging directory ``.tmp_{short_uuid}/`` inside
       ``registry_dir``.
    3. Create the ``artefact/`` subdirectory and delegate to
       ``artefact_writer`` for the model payload — the registry is
       artefact-agnostic (plan D9: the model's own ``save`` protocol
       writes the file).
    4. Write ``run.json`` with
       ``json.dumps(..., indent=2, allow_nan=True, ensure_ascii=False)``
       — ``allow_nan=True`` accommodates
       ``ScipyParametricModel.metadata.hyperparameters["covariance_matrix"]``
       entries that may be ``float("inf")`` (plan §8 R3).
    5. If a run with the same ``run_id`` already exists (the D2
       last-write-wins same-minute-collision case) remove it first.
    6. ``os.replace(staging, final)`` — atomic on POSIX and NTFS when the
       target does not already exist.

    Returns the path of the final run directory.  On any exception the
    staging directory is cleaned up so ``registry_dir`` does not
    accumulate abandoned ``.tmp_*`` scraps.
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    staging = registry_dir / f".tmp_{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        artefact_dir = staging / "artefact"
        artefact_dir.mkdir()
        artefact_writer(artefact_dir / "model.joblib")
        sidecar_path = staging / "run.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, allow_nan=True, ensure_ascii=False),
            encoding="utf-8",
        )
        final = _run_dir(registry_dir, run_id)
        if final.exists():
            # D2 last-write-wins — documented non-atomic window (plan R7).
            shutil.rmtree(final)
        os.replace(staging, final)
        return final
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
