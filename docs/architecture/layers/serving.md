# Serving — layer architecture

- **Status:** Provisional — first realised by Stage 12 (FastAPI prediction endpoint + skops migration).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (serving paragraph); [`DESIGN.md` §8](../../intent/DESIGN.md#8-tooling-and-libraries) (FastAPI / uvicorn selection).
- **Concrete instances:** [Stage 12 retro](../../lld/stages/12-serving.md) — app factory, skops migration, seven-field log contract.
- **Related principles:** §2.1.1 (standalone module), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.7 (tests at boundaries).
- **Upstream layer:** [Registry](registry.md) — `build_app(registry_dir)` consumes the four-verb registry surface; the serving layer has no other runtime upstream dependency.
- **Downstream consumer:** Stage 18 (drift monitoring) consumes the Stage 12 seven-field structured prediction log.

---

## Why this layer exists

The serving layer is the **HTTP boundary** through which a fitted model answers forecast requests. Every stage before Stage 12 accessed registered models programmatically — CLI scripts, notebooks — in the same Python process that trained them. Stage 12 draws a process boundary: a model trained in a notebook answers HTTP requests in a separate process, loading its artefact from the Stage 9 registry.

The pedagogical payoff is the boundary itself. The facilitator starts the service, issues one `curl` command, and gets a forecast back. The model that was a `pd.Series`-returning Python object is now an HTTP resource. Every subsequent modelling stage (Stage 17, Stage 19) crosses this boundary unchanged — the serving layer is one endpoint, not a model-management framework.

The load-bearing design constraint is Stage 12 intent AC-1: *"The service starts on a clean machine without configuration beyond pointing at a registry location."* Every decision flows from that.

---

## Security boundary (D10 — Ctrl+G reversal)

**The serving layer is a network-facing deserialiser.** A `POST /predict` endpoint receives bytes from the network, resolves a model artefact from disk, and materialises a Python object from it. `joblib.load` on an attacker-controlled artefact is trivially a remote code-execution vector.

At Stage 12 Ctrl+G review the human directed: *"Include skops. This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC."* The project therefore migrated **all six** model families' save / load paths off `joblib` and onto `skops.io` before the serving layer was written. This migration is destructive: any `data/registry/*.joblib` artefact written before Stage 12 is rejected by `registry.load` with a clear error; the operator must retrain.

The `skops.io` trust-list primitive (`bristol_ml.models.io.load_skops`) enforces a project-level allow-list of custom classes. Any new model family added in a future stage must register its custom classes in `bristol_ml.models.io._PROJECT_SAFE_TYPES` via `register_safe_types(...)` at import time before its artefacts can be loaded by the serving layer. `load_skops` will raise `UntrustedTypeError` naming the unexpected type rather than silently materialising it.

The serving layer itself never handles the skops envelope directly — it calls `registry.load(run_id)`, which calls the family's `Model.load(path)`, which calls `load_skops`. The security invariant is maintained at the `io.py` boundary, not at the serving endpoint.

---

## Public surface

```python
# src/bristol_ml/serving/__init__.py
from bristol_ml.serving import build_app          # lazy trampoline into app.py

# src/bristol_ml/serving/schemas.py
from bristol_ml.serving.schemas import PredictRequest, PredictResponse

# src/bristol_ml/serving/__main__.py
# python -m bristol_ml.serving [--registry-dir DIR] [--host HOST] [--port PORT] [overrides...]
```

### `build_app(registry_dir: Path) -> FastAPI`

The sole public factory. Constructs a FastAPI application whose lifespan:

1. Calls `registry.list_runs(registry_dir=registry_dir, sort_by="mae", ascending=True)` and takes the first result as the default model (D6 — lowest-MAE run across **all six families**; no family filter).
2. Loads the default model via `registry.load(run_id, registry_dir=registry_dir)` and stashes it in `app.state.loaded[run_id]`.
3. Emits a structured loguru info line naming `default_run_id` and `model_name` so the operator sees what is being served before the first request.
4. Raises `RuntimeError` (with a message naming the registry directory) if the registry contains no runs.

Non-default `run_id` values supplied in `POST /predict` requests are lazy-loaded on first use and cached in the same `app.state.loaded` dict (D7 — single highest-leverage cut; see "Internals").

The factory is a function, not a class. `build_app` is imported lazily in `__init__.py` (a thin trampoline) so `import bristol_ml.serving` does not pull FastAPI or uvicorn into the import graph until the factory is actually called. This keeps `python -m bristol_ml.serving --help` fast and preserves the `test_serving_module_imports_without_torch` import-graph guard.

### `PredictRequest`

```python
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_dt: AwareDatetime          # UTC-aware; naive inputs → HTTP 422
    features: dict[str, float]        # assembled feature row (D4 — features-in)
    run_id: str | None = None         # None → lowest-MAE default (D6 / D7)
```

**`AwareDatetime` for `target_dt` (D8).** Pydantic rejects naive (timezone-less) datetimes at the validation boundary, returning HTTP 422 before the handler runs. The handler normalises the accepted value to UTC via `.astimezone(datetime.UTC)` before constructing the feature-frame index. This guards the three known Pydantic tzinfo edge cases (`pydantic#8683`, `#6592`, `#9571`).

**Features-in (D4).** The request body carries an already-assembled feature row whose keys match the registered model's `feature_columns`. The server does not run any feature engineering — it passes the dict as-is into a single-row `pd.DataFrame` indexed on `target_dt`. This is the canonical **training-serving skew teaching point**: the caller must assemble the same features the model was trained on. The layer doc is honest about the cost — any skew between the caller's feature assembly and the training pipeline's assembly produces a silent accuracy gap — and records it as a deliberate trade-off for Stage 12's minimal scope rather than an oversight.

`run_id: str | None = None` is the lazy-load knob (D7): `None` resolves to the lowest-MAE default; an explicit value names a specific registered run.

### `PredictResponse`

```python
class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: float                 # forecast value in MW
    run_id: str                       # the resolved run id
    model_name: str                   # human-readable name from the registry sidecar
    target_dt: AwareDatetime          # echo of request target_dt, normalised to UTC
```

---

## Endpoints

### `GET /`

Returns the default-model summary so the facilitator can introspect the service before issuing a predict:

```json
{
  "default_run_id": "nn-temporal-b8-c128-k3_20260425T1200",
  "model_name": "nn-temporal-b8-c128-k3",
  "feature_columns": ["temperature_2m", "dewpoint_2m", "week_sin_k1", ...]
}
```

Also serves as a lightweight liveness probe; AC-1 is asserted by `TestClient` reaching this endpoint.

### `POST /predict`

Accepts a `PredictRequest` body, resolves and (if necessary) lazy-loads the model, calls `model.predict(feature_frame)`, and returns a `PredictResponse`. Per-request structured log emitted after `model.predict` returns (D11).

Unknown `run_id` → HTTP 404 with a `detail` naming both the missing `run_id` and the registry directory.

Naive `target_dt` → HTTP 422 from Pydantic validation (before the handler runs).

### `GET /openapi.json`

FastAPI auto-emits the OpenAPI 3.0 document. `PredictRequest` and `PredictResponse` appear as named components under `components/schemas`. AC-4 — schema discoverability — is satisfied by this auto-emission.

---

## The seven-field prediction log (D11)

Every served prediction emits a single `loguru.logger.bind(...).info("served prediction")` record. The seven fields are the **load-bearing contract Stage 18 (drift monitoring) will consume** without a retrofit:

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` (UUID4, 36 chars) | Generated per request. |
| `model_name` | `str` | From the registry sidecar's `name` field. |
| `model_run_id` | `str` | The resolved registry run id. |
| `target_dt` | `str` (UTC ISO 8601) | Post-`astimezone(UTC)` normalisation. |
| `prediction` | `float` | Forecast value in MW. |
| `latency_ms` | `float` | Wall-clock `model.predict` time via `time.perf_counter`. Excludes lazy-load cost and response-model construction. |
| `feature_hash` | `str` (16 hex chars) | First 16 chars of `sha256(canonical_json(features, sort_keys=True))`. |

Fields are bound via `logger.bind(...)` so a downstream sink can read each field by name rather than reparsing free-form text. The default sink is stdout; no file rotation or log shipper is configured (intent §Out of scope).

A Stage 18 implementer who adds a structured log sink must be able to rely on all seven fields being present and typed exactly as above. The `test_request_log_record_carries_seven_fields` test pins each field name and type invariant.

---

## `nn_temporal` serving without warmup semantics (D9 — Ctrl+G reversal)

The pre-Ctrl+G plan deferred `nn_temporal` from the serving endpoint (returning HTTP 501) on the grounds that the TCN needs a sequence window at predict time, not a single row. The human reversed this at Ctrl+G: *"Keeping code in the codebase that isn't supported in the primary implementation is bad-practice."*

The reversal was cost-free because Stage 11 D5+ already saved the `warmup_features` window inside the `NnTemporalModel` artefact. A loaded `NnTemporalModel` exposes a working single-row `predict(X_one_row)` — the warmup buffer is part of the model state, not a per-request input. The serving layer treats `nn_temporal` as just another `Model`. The `POST /predict` handler has no `isinstance(model, NnTemporalModel)` branch; all six families go through `model.predict(feature_frame).iloc[0]` uniformly.

The `test_serving_prediction_parity_vs_direct_load` test is parametrised over all six families. The `nn_temporal` parameter is the load-bearing assertion: it confirms that a single-row predict through the HTTP boundary matches a direct `registry.load(run_id).predict(one_row)` call under `atol=1e-5`.

---

## Training-serving skew teaching point

The `features-in` design (D4) is the canonical example of training-serving skew in this project. The registered model's `feature_columns` records the columns the model was trained on. The caller must supply exactly those columns in the `POST /predict` body. If the caller's feature-assembly pipeline diverges from the training pipeline — a column renamed, a Fourier harmonic count changed — the model will either raise on an unexpected column set or silently produce a biased forecast.

Stage 12 makes this explicit rather than engineering around it. The layer doc is the pedagogical surface; the facilitator walks through the skew risk in the demo. A raw-inputs / assembler-in-the-loop schema (where the server runs feature engineering) is recorded in the plan as a defensible alternative and deferred to a later stage.

---

## Internals

### Lazy-load cache (D7 — single highest-leverage cut)

The lifespan loads only the single default model. When a request names a non-default `run_id`, the handler lazy-loads via `registry.load` and caches the result in `app.state.loaded[run_id]`. A second request with the same `run_id` hits the cache without touching the registry.

This was the `@minimalist` scope critic's single highest-leverage cut from the pre-plan draft, which had proposed loading all registered runs at startup. Loading all runs at startup turns one error path (default model load fails) into N error paths with N tests, and scales startup latency with the registry size. The cut still applies after the D9 reversal — all six families are loadable through the same `skops`-safe path; lazy-on-demand is uniform.

### `build_app` factory semantics

`build_app` is called once per uvicorn process. The lifespan is an async context manager nested inside the factory; it runs on ASGI mount (i.e. when `TestClient` enters its `with` block, or when uvicorn starts the server). The `factory=True` flag in `__main__.py::_cli_main` ensures the lifespan re-runs per worker — the project does not enable multi-worker today, but the factory shape costs nothing and keeps the door open.

At lifespan teardown, `app.state.loaded` is cleared so torch / large model state does not leak between successive `TestClient` contexts in the test suite.

### `_select_default_run` helper

`_select_default_run(registry_dir)` wraps `registry.list_runs(sort_by="mae", ascending=True)` and returns the first sidecar dict. The sort is deterministic within a registry directory: no new runs slip in during the demo unless someone trains in parallel. The function is a private helper in `app.py`; it is not exported.

---

## Standalone CLI

```
uv run python -m bristol_ml.serving --help
uv run python -m bristol_ml.serving --registry-dir data/registry --port 8000
uv run python -m bristol_ml.serving +serving=default serving.port=8080
```

`__main__.py` is a thin `argparse + uvicorn.run(...)` launcher mirroring the project's `python -m bristol_ml.train` pattern (DESIGN §2.1.1). Hydra resolves `ServingConfig` via `load_config`; explicit CLI flags override the resolved config. `--help` prints the resolved `ServingConfig` schema via `ArgumentDefaultsHelpFormatter` (NFR-2).

Uvicorn and Hydra imports are deferred inside `_cli_main` so the `--help` path stays lightweight and the import-graph guard passes.

---

## Curl example

Start the service (requires at least one trained model in the registry):

```bash
uv run python -m bristol_ml.serving --registry-dir data/registry
```

Introspect the default model:

```bash
curl http://localhost:8000/
```

Issue a prediction (adjust the feature dict to match the model's `feature_columns`):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"target_dt": "2025-06-15T13:00:00Z", "features": {"temp_c": 12.5, "cloud_cover": 0.4}}'
```

The `GET /` response's `feature_columns` array names every key the `features` dict must carry for the default model.

---

## Config

`ServingConfig` in `conf/_schemas.py`:

```python
class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_dir: Path = Path("data/registry")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
```

Composed into `AppConfig.serving: ServingConfig | None = None` — `None` when not serving (the train CLI and config-smoke tests are unaffected). Loaded via `+serving=default` using `conf/serving/default.yaml`.

---

## Module structure

```
src/bristol_ml/serving/
├── __init__.py      # lazy build_app trampoline; keeps import cheap
├── __main__.py      # argparse + uvicorn.run; python -m bristol_ml.serving
├── app.py           # build_app factory + lifespan + GET / + POST /predict
├── schemas.py       # PredictRequest + PredictResponse Pydantic models
└── CLAUDE.md        # module guide; curl one-liner; skops + nn_temporal notes
```

---

## Running tests

```bash
uv run pytest tests/integration/serving/ -v
```

17 tests in `tests/integration/serving/test_api.py` cover AC-1..AC-4 (lifespan startup, empty-registry error, D6 default selection, AC-2 validation errors, AC-3 parity over all six families, D7 lazy-load cache, D11 seven-field log, AC-4 OpenAPI schema). The per-family parity test is the load-bearing assertion for the D9 Ctrl+G reversal.

---

## Non-functional requirements

- **NFR-1 (schema discoverability)** — `GET /openapi.json` returns the OpenAPI 3.0 document with `PredictRequest` and `PredictResponse` as named components. Satisfied by FastAPI auto-emission.
- **NFR-2 (standalone CLI)** — `python -m bristol_ml.serving --help` exits 0 with the resolved `ServingConfig` schema; `python -m bristol_ml.serving --registry-dir ...` starts the service. Enforced by `test_serving_cli_help_exits_zero`.
- **NFR-3 (notebooks thin)** — No notebook added. A curl example in this layer doc and in `src/bristol_ml/serving/CLAUDE.md` satisfies the intent's "or curl example" clause.
- **NFR-4 (latency)** — No CI assertion. Observed latencies pending — NFR-4 cut from CI; measure in the Stage 12 retro before merging and record per model family.

---

## Cross-references

- `src/bristol_ml/serving/CLAUDE.md` — concrete module guide; curl one-liner; `nn_temporal` warmup-envelope teaching point; skops adoption note.
- `docs/lld/stages/12-serving.md` — Stage 12 retro including the D9 + D10 Ctrl+G reversal log and observed latencies.
- `docs/architecture/layers/registry.md` — the upstream layer; `registry.load`, `registry.list_runs`, `registry.describe` are the only registry verbs the serving layer calls.
- `src/bristol_ml/registry/CLAUDE.md` — skops migration outcome; joblib rejection error message.
- `src/bristol_ml/models/io.py` — `load_skops`, `register_safe_types`, `UntrustedTypeError`; the trust-list contract every future model family must satisfy.
- `docs/architecture/layers/models-nn.md` — Stage 11 warmup-envelope contract that makes `nn_temporal` servable without warmup semantics in the request body.
- Stage 12 plan — `docs/plans/active/12-serving.md` (active during Phase 2; moves to `completed/` at T10).
