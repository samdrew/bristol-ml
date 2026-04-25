"""Serving layer — minimal FastAPI prediction endpoint loading from the registry.

Stage 12 introduces this module: a single ``POST /predict`` endpoint
that loads a registered :class:`bristol_ml.models.Model` by ``run_id``
(or the lowest-MAE default at startup) and returns a forecast for a
features-in payload.  All six model families
(``naive``, ``linear``, ``sarimax``, ``scipy_parametric``,
``nn_mlp``, ``nn_temporal``) are served — Stage 11 D5+ baked the
``warmup_features`` window into the ``nn_temporal`` artefact, so
single-row predict works through the boundary uniformly across
families.

The module re-exports :func:`build_app` lazily — importing
``bristol_ml.serving`` does **not** pull :mod:`fastapi` or
:mod:`uvicorn` into the import graph until the app factory is
actually called, so ``python -m bristol_ml --help`` (scaffold
invocation) stays cheap.

Cross-references:

- Layer doc — ``docs/architecture/layers/serving.md`` (lands at
  Stage 12 T9).
- Module guide — ``src/bristol_ml/serving/CLAUDE.md`` (Stage 12 T9).
- Plan — ``docs/plans/active/12-serving.md`` (active during Phase 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from fastapi import FastAPI

__all__ = ["build_app"]


def build_app(*args: object, **kwargs: object) -> FastAPI:
    """Construct the FastAPI serving application (placeholder until T7).

    The real implementation lands at Stage 12 T7.  Calling this stub
    raises :class:`NotImplementedError` so any caller that reaches it
    before T7 sees a clear error rather than a silent partial app.
    """
    from bristol_ml.serving import app as _app

    return _app.build_app(*args, **kwargs)
