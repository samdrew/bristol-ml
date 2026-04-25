"""FastAPI app factory for the bristol_ml serving layer (Stage 12).

Implements Stage 12 T7 + T8: a single ``POST /predict`` endpoint
backed by the Stage 9 registry, plus a ``GET /`` root that returns the
resolved default model so a facilitator can introspect the service
before the demo curl.  Per-request seven-field structured logging
(plan D11) lands at T8 alongside the standalone CLI in
``bristol_ml.serving.__main__``.

Public surface — see ``docs/plans/active/12-serving.md`` §5:

- :func:`build_app` — construct a :class:`fastapi.FastAPI` instance
  bound to a registry directory.  The lifespan resolves the lowest-MAE
  registered run via :func:`bristol_ml.registry.list_runs` and stashes
  the loaded :class:`bristol_ml.models.Model` in
  ``app.state.loaded`` keyed by ``run_id``.  Subsequent non-default
  ``run_id`` values are lazy-loaded on first use and cached in the
  same dict (D7 — single highest-leverage cut).
- ``GET /`` — returns ``{default_run_id, model_name, feature_columns}``
  so the operator can confirm what is being served.
- ``POST /predict`` — features-in body (D4), ``AwareDatetime`` +
  UTC-normalisation (D8), default-or-supplied ``run_id`` (D5).  Every
  registered model family is served through the same path — Stage 11
  D5+ baked the ``warmup_features`` window into the ``nn_temporal``
  artefact, so single-row predict works uniformly across all six
  families (D9 Ctrl+G reversal).

The per-request log (D11) emits a single ``loguru`` info line per
served prediction, with seven structured fields bound through
``logger.bind(...).info("served prediction")`` so Stage 18's drift
monitoring can consume the log without a retrofit:

- ``request_id``    — UUID4 generated per request.
- ``model_name``    — the resolved model's human-readable name.
- ``model_run_id``  — the registry run id that served the request.
- ``target_dt``     — UTC ISO 8601 (post-``astimezone(dt.UTC)``).
- ``prediction``    — float, MW.
- ``latency_ms``    — wall-clock predict time, ``time.perf_counter``.
- ``feature_hash``  — first 16 hex chars of
  ``sha256(canonical_json(features_in))``.

Logs route to ``stdout`` per the project default; no file rotation
or shipper is configured (intent §Out of scope).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from bristol_ml import registry
from bristol_ml.serving.schemas import PredictRequest, PredictResponse

if TYPE_CHECKING:  # pragma: no cover — typing-only re-exports
    from collections.abc import AsyncIterator

    from bristol_ml.models.protocol import Model

__all__ = ("build_app",)


def _select_default_run(registry_dir: Path) -> dict[str, Any]:
    """Return the lowest-MAE registered-run sidecar (D6).

    Wraps :func:`bristol_ml.registry.list_runs` with the project-wide
    ``sort_by="mae", ascending=True`` invariant so the demo moment is
    deterministic — the service serves the best-scoring run regardless
    of family (D9 Ctrl+G reversal: every family is eligible).

    Raises
    ------
    RuntimeError
        If the registry directory is missing or contains no registered
        runs.  The lifespan converts the raise into a clear startup
        failure that names the registry directory.
    """
    runs = registry.list_runs(
        registry_dir=registry_dir,
        sort_by="mae",
        ascending=True,
    )
    if not runs:
        raise RuntimeError(
            f"Registry at {registry_dir!s} contains no runs; cannot select a "
            "default model. Train at least one model first "
            "(`uv run python -m bristol_ml.train`)."
        )
    return runs[0]


def build_app(registry_dir: Path) -> FastAPI:
    """Construct the FastAPI serving application.

    Parameters
    ----------
    registry_dir:
        On-disk root of the Stage 9 registry.  The lifespan reads this
        directory at startup, picks the lowest-MAE run as the default
        model, and stashes the loaded model in ``app.state.loaded``.

    Returns
    -------
    FastAPI
        A configured :class:`FastAPI` instance.  The lifespan only runs
        when the app is mounted under an ASGI server (uvicorn, or the
        :class:`fastapi.testclient.TestClient`'s ``with`` block).

    The lifespan raises :class:`RuntimeError` if ``registry_dir`` is
    empty — clean fail-fast behaviour for AC-1 ("starts on a clean
    machine without configuration beyond pointing at a registry
    location").
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Resolve the default model and seed the lazy-load cache."""
        sidecar = _select_default_run(registry_dir)
        run_id = sidecar["run_id"]
        name = sidecar["name"]
        feature_columns = list(sidecar["feature_columns"])
        model: Model = registry.load(run_id, registry_dir=registry_dir)
        # Structured info line so the operator sees what is being
        # served before the first request hits the endpoint.  The
        # seven-field per-request log is added at T8; here we both
        # bind the values as loguru extras (for downstream-structured
        # consumers) *and* include them in the message text so the
        # test fixture's message-only formatter can grep them out
        # without a custom sink.
        logger.bind(
            default_run_id=run_id,
            model_name=name,
        ).info(
            f"serving lifespan: default model resolved "
            f"(default_run_id={run_id!r}, model_name={name!r})"
        )
        app.state.loaded = {run_id: model}
        app.state.default_run_id = run_id
        app.state.default_name = name
        app.state.default_feature_columns = feature_columns
        app.state.registry_dir = registry_dir
        try:
            yield
        finally:
            # Drop strong references at shutdown so torch / large
            # model state do not leak between successive TestClient
            # contexts in the test suite.
            app.state.loaded = {}

    app = FastAPI(title="bristol_ml serving", lifespan=lifespan)

    @app.get("/")
    def root() -> dict[str, Any]:
        """Return the default-model summary (also serves as a liveness probe).

        AC-1 is asserted by ``TestClient`` reaching the lifespan; this
        endpoint exposes the resolved default so the demo facilitator
        can introspect the service via ``curl http://localhost:8000/``
        before issuing a predict.
        """
        return {
            "default_run_id": app.state.default_run_id,
            "model_name": app.state.default_name,
            "feature_columns": app.state.default_feature_columns,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Serve a single-row prediction request (D4 features-in)."""
        run_id = req.run_id or app.state.default_run_id
        # D7: lazy-load on first request for a non-default run_id;
        # cache the loaded model so subsequent requests do not hit
        # disk again (the lazy-load cache test asserts this).
        if run_id not in app.state.loaded:
            try:
                app.state.loaded[run_id] = registry.load(
                    run_id, registry_dir=app.state.registry_dir
                )
            except FileNotFoundError as exc:
                # AC-2: clear error on invalid input.  The 404 detail
                # names both the missing run_id and the registry dir
                # so the operator can self-diagnose without reading
                # the server logs.
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"run_id={run_id!r} not found under "
                        f"registry_dir={app.state.registry_dir!s}."
                    ),
                ) from exc
        model = app.state.loaded[run_id]
        sidecar = registry.describe(run_id, registry_dir=app.state.registry_dir)

        # D8: tz-normalise to UTC at the handler boundary so downstream
        # feature-frame indexing is deterministic.  AwareDatetime has
        # already rejected naive inputs at the validation boundary.
        target_utc = req.target_dt.astimezone(dt.UTC)
        # D4: features-in.  Build a single-row DataFrame indexed at the
        # target timestamp; the model's predict contract is
        # ``DataFrame(superset of feature_columns) -> Series``.
        feature_frame = pd.DataFrame(
            [req.features],
            index=pd.DatetimeIndex([target_utc], tz="UTC"),
        )
        # D11 latency window — wraps ``model.predict`` only.  Excludes
        # the lazy-load cost (which would dominate first-call latency
        # for non-default run_ids and confuse the Stage-18 drift
        # consumer) and excludes the response-model construction (a
        # constant cost the operator cannot influence).
        t0 = time.perf_counter()
        prediction_series = model.predict(feature_frame)
        prediction = float(prediction_series.iloc[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # D11 feature_hash: canonical JSON ensures the hash is stable
        # under dict-iteration-order differences (the Pydantic model
        # already gives us the dict, but ``sort_keys=True`` removes any
        # remaining ambiguity from feature-name ordering downstream).
        # Truncated to 16 hex chars per the plan §1 D11 / R8 minimum.
        feature_hash = hashlib.sha256(
            json.dumps(req.features, sort_keys=True).encode()
        ).hexdigest()[:16]

        # D11 seven-field structured log — bound as loguru extras so
        # Stage 18's drift consumer can read each field by name.  Field
        # order matches the plan §1 D11 spec verbatim.
        logger.bind(
            request_id=str(uuid.uuid4()),
            model_name=sidecar["name"],
            model_run_id=run_id,
            target_dt=target_utc.isoformat(),
            prediction=prediction,
            latency_ms=latency_ms,
            feature_hash=feature_hash,
        ).info("served prediction")

        return PredictResponse(
            prediction=prediction,
            run_id=run_id,
            model_name=sidecar["name"],
            target_dt=target_utc,
        )

    return app
