# `bristol_ml.serving` — module guide

This module is the **serving layer**: a minimal FastAPI HTTP application
that loads a fitted model from the Stage 9 registry and answers
forecast requests.  Stage 12 introduces the layer; Stage 18 (drift
monitoring) will consume its structured prediction log; Stage 19
(orchestration) will schedule batch scoring through the same artefacts.

Read the layer contract in
[`docs/architecture/layers/serving.md`](../../../docs/architecture/layers/serving.md)
before extending this module; the file you are reading documents the
concrete Stage 12 surface.

## Public surface

- **`bristol_ml.serving.build_app(registry_dir: Path) -> FastAPI`** —
  the public entry point.  Resolves the lowest-MAE registered run as
  the default model at startup; lazy-loads further `run_id`s on first
  request and caches them in `app.state.loaded`.  All six model
  families (`naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`,
  `nn_temporal`) are served through the same `model.predict(feature_frame)`
  path — see the `nn_temporal` note below.
- **`bristol_ml.serving.schemas.PredictRequest`** — Pydantic v2 body
  schema: `target_dt: AwareDatetime`, `features: dict[str, float]`,
  `run_id: str | None = None`.
- **`bristol_ml.serving.schemas.PredictResponse`** — `prediction: float`,
  `run_id: str`, `model_name: str`, `target_dt: AwareDatetime`.

## Standalone CLI

```bash
uv run python -m bristol_ml.serving --help
uv run python -m bristol_ml.serving --registry-dir data/registry --port 8000
```

`--help` prints the resolved `ServingConfig` schema (plan NFR-2 /
DESIGN §2.1.1).  Defaults: `registry_dir=data/registry`, `host=127.0.0.1`,
`port=8000`.  Hydra-style overrides are also accepted as positional
arguments (`+serving=default serving.port=8080`).

## Quick start — curl one-liner

Start the service (registry must contain at least one trained model):

```bash
uv run python -m bristol_ml.serving --registry-dir data/registry
```

Check what is being served:

```bash
curl http://localhost:8000/
```

Issue a prediction (replace the feature dict with the keys in the
`feature_columns` array from `GET /`):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"target_dt": "2025-06-15T13:00:00Z", "features": {"temp_c": 12.5, "cloud_cover": 0.4}}'
```

The response is a JSON object with `prediction`, `run_id`,
`model_name`, and `target_dt`.

## Key files

| File | Responsibility |
|------|----------------|
| `__init__.py` | Lazy `build_app` trampoline — keeps `import bristol_ml.serving` cheap |
| `__main__.py` | argparse + uvicorn.run; `python -m bristol_ml.serving` |
| `app.py` | `build_app` factory + async lifespan + `GET /` + `POST /predict` |
| `schemas.py` | `PredictRequest` / `PredictResponse` Pydantic models |

`app.py` is the only file that imports FastAPI and loguru at module
level; `__init__.py` and `__main__.py` defer those imports inside
function bodies so the import-graph guard
(`test_serving_module_imports_without_torch`) stays clean.

## `nn_temporal` — warmup envelope teaching point (D9 Ctrl+G reversal)

Stage 11 D5+ **baked the `warmup_features` window inside the
`NnTemporalModel` artefact** so that a loaded `NnTemporalModel.predict(
single_row)` works without the caller knowing about warmup.  The
serving layer inherits this for free: `nn_temporal` looks identical to
the five stateless families at the `POST /predict` boundary.  There is
no `isinstance(model, NnTemporalModel)` branch in `app.py`, no warmup
semantics in `PredictRequest`, no 501 path.

The pre-Ctrl+G plan had deferred `nn_temporal` via a 501 response.
The human reversed this at Ctrl+G: *"Keeping code in the codebase that
isn't supported in the primary implementation is bad-practice."*  Stage 11
invested deliberately in the warmup-envelope design precisely so the
serving boundary would not need model-family awareness.

The `test_serving_prediction_parity_vs_direct_load` parametrised test
(over all six families) is the load-bearing assertion for this
contract.

## Security boundary — skops adoption (D10 Ctrl+G reversal)

The serving layer is a network-facing deserialiser.  At Stage 12
Ctrl+G, the human directed: *"Include skops. This includes a network
facing interface so security should be paramount, as I don't want an
RCE exploit on my PC."*

All six model families' `save` / `load` paths were therefore migrated
from `joblib` to `skops.io` as part of Stage 12 (T2–T5).  The serving
layer never touches the format directly: it calls `registry.load(run_id)`,
which calls the family's `Model.load(path)`, which calls
`bristol_ml.models.io.load_skops`.  `load_skops` enforces the project
trust-list (`_PROJECT_SAFE_TYPES`) and raises `UntrustedTypeError` for
any artefact containing an unregistered type.

**Breaking change for existing users:** any `data/registry/*.joblib`
artefact written before Stage 12 is rejected by `registry.load` with a
clear `RuntimeError`; the operator must retrain.

**Trust-list contract for future model families:** any new model class
added after Stage 12 must call
`bristol_ml.models.io.register_safe_types("module.path.ClassName")` at
import time for every custom class that appears in its saved artefact.
Failing to do so causes `load_skops` to raise `UntrustedTypeError`
when the serving layer tries to load the run.

## Seven-field prediction log (D11 — Stage 18 contract)

Every `POST /predict` response is preceded by a single structured log
line:

```python
logger.bind(
    request_id=...,       # UUID4
    model_name=...,       # from registry sidecar
    model_run_id=...,     # the resolved run_id
    target_dt=...,        # UTC ISO 8601
    prediction=...,       # float, MW
    latency_ms=...,       # time.perf_counter over model.predict only
    feature_hash=...,     # first 16 hex chars of sha256(canonical_json(features))
).info("served prediction")
```

These seven fields are the **load-bearing contract Stage 18 (drift
monitoring) will consume**.  Do not rename or remove them; a Stage 18
implementer who adds a structured log sink can rely on all seven being
present and typed as above.

## Gotchas

1. **`AwareDatetime` rejects naive timestamps at the Pydantic
   boundary.** POST a naive `target_dt` (no `Z` or offset) and FastAPI
   returns HTTP 422 before the handler runs.  The handler then
   normalises to UTC via `.astimezone(datetime.UTC)` — use the
   `.astimezone` path rather than direct equality on `tzinfo` objects
   to avoid `pydantic#8683` / `#6592` / `#9571`.

2. **`features-in` means the caller owns feature assembly.** The server
   does no feature engineering.  The `feature_columns` array in `GET /`
   names exactly the keys the `features` dict must carry.  A mismatch
   produces a natural `model.predict` error (D12 cut — no extra guard);
   the HTTP framework surfaces it as a 422 or 500 depending on whether
   it surfaces before or inside the handler.

3. **Lazy-load (D7) excludes the default model.** The lifespan loads
   only the default model.  `app.state.loaded` starts with one entry.
   Non-default `run_id`s are loaded on first request; the cache is
   cleared at lifespan teardown so successive `TestClient` contexts do
   not share state.

4. **`build_app` is a factory, not a singleton.** Each call returns an
   independent `FastAPI` instance with its own lifespan.  Tests that
   call `build_app(tmp_path)` inside a `with TestClient(app)` block run
   the full lifespan per test — no shared state.

5. **Hydra group convention: `conf/serving/default.yaml`, not
   `conf/serving.yaml`.** Compose via `+serving=default`.  The T6
   implementation discovered that the flat `conf/serving.yaml` path is
   not a valid Hydra group member; only the group-directory layout
   (`conf/serving/default.yaml`) composes correctly via
   `+serving=default`.

## Running standalone tests

```bash
uv run pytest tests/integration/serving/ -v
uv run pytest tests/unit/serving/ -v
```

## Cross-references

- Layer contract — `docs/architecture/layers/serving.md`.
- Stage 12 plan — `docs/plans/active/12-serving.md` (moves to
  `completed/` at T10).
- Stage 12 retro — `docs/lld/stages/12-serving.md`.
- Registry layer — `docs/architecture/layers/registry.md` and
  `src/bristol_ml/registry/CLAUDE.md`.
- Models IO — `src/bristol_ml/models/io.py` (`load_skops`,
  `register_safe_types`, `UntrustedTypeError`).
- NN sub-layer — `src/bristol_ml/models/nn/CLAUDE.md` (warmup-envelope
  contract that makes `nn_temporal` servable without warmup in the
  request body).
