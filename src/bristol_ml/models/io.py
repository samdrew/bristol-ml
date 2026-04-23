"""Joblib-backed save / load helpers for model artefacts.

Stage 4 standardises on :mod:`joblib` for persisting fitted models, per plan
D6.  joblib is the sklearn-ecosystem default, handles numpy/pandas-heavy
objects efficiently, and round-trips statsmodels ``RegressionResultsWrapper``
instances without extra work.  Everything a :class:`bristol_ml.models.Model`
implementor needs to persist itself flows through the two functions here.

Writes are **atomic** — mirrors the ingestion layer's
``bristol_ml.ingestion._common._atomic_write`` idiom: write to a sibling
``<name>.tmp`` file then rename via :func:`os.replace` (the portable atomic
rename primitive on POSIX + NTFS).  A crash mid-write therefore leaves the
previous artefact intact rather than producing a zero-byte file.

Upgrade path: :mod:`skops.io` for secure artefacts at Stage 12 (serving),
per Stage 9 plan D14 — Stage 12 is the first stage that loads artefacts
from a path not controlled by the training author, which is the correct
inflection point for the security upgrade.  joblib (like pickle) is *not*
a safe deserialiser for untrusted inputs; Stages 4-11 only ever load
artefacts we wrote ourselves (the Stage 9 registry explicitly documents
this), so the audit burden of skops is disproportionate to those stages'
demo focus.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

__all__ = ["load_joblib", "save_joblib"]


def save_joblib(obj: Any, path: Path) -> None:
    """Serialise ``obj`` to ``path`` atomically.

    Writes to a sibling ``<path>.tmp`` first and then renames with
    :func:`os.replace`, so a crash mid-write cannot corrupt an existing
    artefact.  The parent directory is created if missing — callers do not
    need to pre-create it.

    Parameters
    ----------
    obj:
        Any Python object joblib can serialise.  For Stage 4 this is either
        a concrete :class:`bristol_ml.models.Model` instance or a statsmodels
        ``RegressionResultsWrapper``.
    path:
        Destination artefact path.  Convention is ``.joblib`` suffix but
        not enforced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def load_joblib(path: Path) -> Any:
    """Deserialise a joblib artefact previously written by :func:`save_joblib`.

    This is a thin wrapper around :func:`joblib.load` — it exists so the
    Stage 9 registry can swap in :mod:`skops.io` by changing one call site.
    No validation beyond "joblib can parse it" is performed at Stage 4.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    return joblib.load(path)
