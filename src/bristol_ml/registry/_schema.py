"""Sidecar JSON schema for the filesystem-backed registry (Stage 9).

The registry stores one ``run.json`` per registered run.  The shape is the
:class:`SidecarFields` ``TypedDict`` below — a structural type rather than a
Pydantic model because the sidecar is written and read as plain JSON and
only ever consumed programmatically through the four-verb public surface.
Strict validation on read would couple the registry to a Pydantic
dependency we do not currently need at Stage 9 (plan §1 D4).

If Stage 17 (price-target models) or Stage 18 (drift monitoring) needs a
validated read path, promote this to a Pydantic model under
``conf/_schemas.py`` — but do so as a deliberate surface-widening
decision, not an accretion.
"""

from __future__ import annotations

from typing import Any, TypedDict


class MetricSummary(TypedDict):
    """Per-metric roll-up stored in :data:`SidecarFields.metrics`.

    ``mean`` and ``std`` are rolling-origin cross-fold aggregates computed by
    the Stage 6 evaluation harness; ``per_fold`` is the raw list the harness
    emitted, preserved so a reader can re-derive the summary without losing
    information (plan §1 D15).
    """

    mean: float
    std: float
    per_fold: list[float]


class SidecarFields(TypedDict):
    """Structural schema for ``run.json`` (plan §5 sidecar schema).

    Field ordering is cosmetic — readers key in by name.  ``git_sha`` is
    typed ``str | None`` because :func:`bristol_ml.registry._git._git_sha_or_none`
    returns ``None`` when the save happens outside a git working tree.
    """

    run_id: str
    name: str
    type: str  # "naive" | "linear" | "sarimax" | "scipy_parametric"
    feature_set: str
    target: str
    feature_columns: list[str]
    fit_utc: str  # ISO-8601 tz-aware UTC, second precision
    git_sha: str | None
    hyperparameters: dict[str, Any]
    metrics: dict[str, MetricSummary]
    registered_at_utc: str  # ISO-8601 tz-aware UTC, second precision
