"""FastAPI app factory for the bristol_ml serving layer (Stage 12).

The full implementation lands at Stage 12 T7 (app factory + lifespan +
``GET /`` + ``POST /predict``) and T8 (structured logging + standalone
CLI wiring).  At T1 (this file) only the scaffold is in place — calling
:func:`build_app` raises :class:`NotImplementedError` so any caller
reaching it before T7 gets a clear error instead of a silent partial
app.

The full design surface is in ``docs/plans/active/12-serving.md`` §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from pathlib import Path

    from fastapi import FastAPI


def build_app(registry_dir: Path) -> FastAPI:  # type: ignore[empty-body] # T1 scaffold
    """Construct the FastAPI app (placeholder; full implementation at T7).

    Parameters
    ----------
    registry_dir:
        On-disk root of the registry (Stage 9 surface).  At T7 the
        lifespan resolves the lowest-MAE run via
        :func:`bristol_ml.registry.list_runs` and stashes the loaded
        model in ``app.state``.

    Raises
    ------
    NotImplementedError
        Always (T1 scaffold).  Stage 12 T7 fills this in.
    """
    raise NotImplementedError(
        "bristol_ml.serving.app.build_app is the Stage 12 T1 scaffold; the "
        "full implementation lands at T7 (app factory + lifespan + GET / + "
        "POST /predict). See docs/plans/active/12-serving.md §5."
    )
