"""Pydantic v2 request / response schemas for the serving layer (Stage 12).

The serving layer's HTTP boundary contract.  Both classes live in this
module rather than next to the FastAPI app factory so callers (the
client side of the curl example, downstream Stage 18 log consumers,
notebook tutorials) can import the schemas without pulling
:mod:`fastapi` or :mod:`uvicorn` into the import graph.

Per plan §5 / D4 the request carries an *already-assembled* feature
row — the keys must match :attr:`bristol_ml.config.ModelMetadata.feature_columns`
exactly — and the response echoes the resolved ``run_id`` + the
prediction.  Per D8, ``target_dt`` is :class:`AwareDatetime`; the
serving layer normalises to UTC at the handler boundary before
constructing the feature-frame index.

The ``run_id: str | None = None`` field on :class:`PredictRequest` is
the lazy-load knob (D7): ``None`` resolves to the lowest-MAE default
selected by the lifespan; an explicit value names a specific
registered run, lazy-loaded into ``app.state.loaded`` on first use.
The pre-Ctrl+G plan draft excluded ``nn_temporal`` from the eligible
set; the Ctrl+G D9 reversal makes every family eligible (Stage 11
D5+ baked the warmup window into the artefact), so any registered
run — including ``nn_temporal`` — is a valid value.
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Single-row prediction request body.

    The body shape is "features-in" (D4): the caller has already
    assembled the feature row that matches the registered model's
    ``feature_columns``.  The serving layer does not run any feature
    engineering — that decision is documented in the layer doc as the
    canonical training-serving-skew teaching point.
    """

    # ``extra="forbid"`` makes typos in the request body a 422 rather
    # than a silently dropped field; ``frozen=True`` mirrors the
    # project-wide convention for resolved config models so handler
    # code cannot mutate validated input by mistake.
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_dt: AwareDatetime = Field(
        description=(
            "UTC-aware target timestamp the forecast is for. Naive "
            "(tz-less) datetimes are rejected with HTTP 422 at the "
            "validation boundary (Stage 12 D8)."
        ),
    )
    features: dict[str, float] = Field(
        description=(
            "Assembled feature row matching the registered model's "
            "``feature_columns``. Keys must match exactly; values are "
            "the assembled feature values (Stage 12 D4 — features-in)."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Registry run id naming a specific registered model. "
            "``None`` resolves to the lowest-MAE default model "
            "selected at startup (Stage 12 D6 / D7). Every model "
            "family is eligible — Stage 11 D5+ baked the "
            "``warmup_features`` window into the ``nn_temporal`` "
            "artefact, so single-row predict works through the "
            "boundary uniformly across families (Stage 12 D9 "
            "Ctrl+G reversal)."
        ),
    )


class PredictResponse(BaseModel):
    """Single-row prediction response body.

    Echoes the resolved ``run_id`` and the human-readable model name
    so the caller can confirm which registered model served the
    request — useful when ``run_id`` was left ``None`` and the lifespan
    chose the default for them (D6).  ``target_dt`` is echoed back
    normalised to UTC (D8) to make the round-trip self-documenting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: float = Field(
        description="Forecast value in MW (the unit is project-wide; see DESIGN §3).",
    )
    run_id: str = Field(
        description="The resolved registry run id used to serve this request.",
    )
    model_name: str = Field(
        description=(
            "Human-readable model name, sourced from the registry "
            "sidecar's ``name`` field (e.g. ``'naive-same-hour-last-week'``)."
        ),
    )
    target_dt: AwareDatetime = Field(
        description="Echo of the request ``target_dt``, normalised to UTC.",
    )
