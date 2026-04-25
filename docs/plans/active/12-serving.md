# Plan — Stage 12: Minimal serving endpoint

**Status:** `draft` — awaiting Ctrl+G review.
**Intent:** [`docs/intent/12-serving.md`](../../intent/12-serving.md)
**Upstream stages shipped:** Stages 0–11 (foundation → ingestion → features → six model families → enhanced evaluation → registry → MLP → TCN). Stage 11 is in flight on a sibling branch; Stage 12 branches from `main` and consumes only the Stage 9 registry surface, so the dependency is `Stage 9` not `Stage 11`.
**Downstream consumers:** Stage 18 (drift monitoring — consumes the Stage 12 prediction log), Stage 19 (orchestration — schedules batch scoring through the same artefacts).
**Baseline SHA:** `6ad2d7a` (tip of `main` after the DESIGN §6 `docs/plans/` addition — PR #9).

**Discovery artefacts produced in Phase 1:**

- Requirements — [`docs/lld/research/12-serving-requirements.md`](../../lld/research/12-serving-requirements.md)
- Codebase map — [`docs/lld/research/12-serving-codebase.md`](../../lld/research/12-serving-codebase.md)
- Domain research — [`docs/lld/research/12-serving-domain.md`](../../lld/research/12-serving-domain.md)
- Scope Diff — [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §Demo moment names a one-minute facilitator loop: start the service, curl it with a sample payload, get a forecast back. The pedagogical point is the boundary itself — a model that was trained in a notebook now answers HTTP requests in a separate process — not the framework, not the schema, not the latency. Every decision below is filtered through "does this make the boundary more legible to the meetup audience?"; anything that does not is cut by the Scope Diff or deferred.

**Stage 11 reach.** `NnTemporalModel` (Stage 11) is the only model in the roster that requires sequence state at predict time (a `seq_len`-window plus the `warmup_features` contract added in Stage 11 D5+). The serving endpoint cannot give it a meaningful single-row predict path without exposing the warmup semantics through the request body, which is a distinct design problem. Per **D9** (Scope Diff `RESTATES INTENT`), Stage 12 explicitly returns HTTP 501 for a `nn_temporal` request and documents the deferral as a teaching point. The five other model families — `naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp` — are all stateless single-row predictors and are in scope.

---

## 1. Decisions for the human (resolve before Phase 2)

Eleven decision points (nine kept, two cut per the Scope Diff) plus one housekeeping carry-over. The decision set is filtered through the `@minimalist` Scope Diff in [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md); three tags were flipped from the lead's draft framing — **D2** (`fastapi[standard]` extra → drop the extra), **D7** (load all registered runs → load only the default model — *single highest-leverage cut*), **D12 cut** (column-set boundary assertion). Defaults lean on the three research artefacts and the simplicity bias in `DESIGN.md §2.2.4`. The Evidence column cites the research that *resolved* each decision.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | HTTP framework | **FastAPI.** No alternative considered — `DESIGN.md §8` names FastAPI explicitly. | The decision is already recorded in the spec; the Phase-1 research confirms it (R1). | DESIGN.md §8; domain research §R1. Scope Diff D1 (RESTATES INTENT). |
| **D2** | Dependency footprint | **`fastapi` + `uvicorn` (no `[standard]` extra).** `httpx` is already a runtime dep (Stage 1 ingestion), so `fastapi.testclient.TestClient` works without `[standard]`. The extras `[standard]` would pull (`jinja2`, `python-multipart`, `fastapi-cli`) are not required by any AC. | Scope Diff `PLAN POLISH` flag flipped: minimum viable framework surface. Three transitive packages saved with no AC coverage cost. | Domain research §R1; codebase map §8 (httpx already present); Scope Diff D2 (PLAN POLISH → cut the extras). |
| **D3** | Predict handler shape | **Synchronous `def predict(...)`** — not `async def`. CPU inference has no I/O to await; FastAPI runs sync handlers in a threadpool transparently. | Async without genuine awaitable work would mislead readers about why async is used (§2.2.4 simplicity bias). All handlers in the project are sync; the serving layer follows the same convention. | Domain research §R1; Scope Diff D3 (RESTATES INTENT). |
| **D4** | Request schema | **"Features-in" — the request body carries an already-assembled feature row whose keys must match `model.metadata.feature_columns` exactly. The server calls `model.predict(pd.DataFrame([row], index=[target_dt]))` and returns the scalar.** Raw-inputs / assembler-in-the-loop is deferred (intent §Points names both as defensible; Stage 12 takes the simpler one). | AC-3 (prediction parity) is trivially testable with features-in: the same row through the registry's direct `model.predict` and through the endpoint must match under `numpy.allclose(atol=1e-5)`. Raw-inputs introduces an assembler dependency that the intent does not require and that the codebase map flags as the canonical training-serving-skew surface. The "leakage of the training assumption" cost the requirements analyst notes is honestly acknowledged as a teaching point in the layer doc rather than engineered around. | Requirements OQ-2; domain research §R5; Scope Diff D4 (RESTATES INTENT). |
| **D5** | Endpoint shape | **Single `POST /predict` endpoint.** The request body carries an optional `run_id: str \| None` field — `None` means "use the default model selected at startup". Per-model endpoints (`/predict/linear`, etc.) are not added: a new model family must not require a serving-layer code change. | Intent §Scope: "a single prediction endpoint." Intent §Points: "a model-name parameter is more flexible." Codebase map §3: `registry.load(run_id)` is the natural call site; mirrors the registry's already-shipped surface. | Intent §Scope; requirements OQ-3; Scope Diff D5 (RESTATES INTENT). |
| **D6** | Default-model selection | **Lowest-MAE run from `registry.list_runs(sort_by="mae", ascending=True)[0]` at startup, with `nn_temporal` runs excluded from the candidate set (consequence of D9).** If no eligible run exists, the lifespan raises `RuntimeError` with a clear message naming the registry directory and the empty filter. The selected `run_id` is exposed via `GET /` as `{"default_run_id": "...", "model_name": "...", "feature_columns": [...]}` so the facilitator can curl-introspect it before issuing a predict. | AC-1: "no configuration beyond pointing at a registry location." Picking by MAE makes the demo honest — the service serves the best-scoring eligible model. Determinism is preserved within a registry directory (no new runs slip in during the demo unless someone trains in parallel). | Requirements OQ-1; intent §Points; Scope Diff D6 (RESTATES INTENT). |
| **D7** | Model loading at startup | **Load only the single default model selected by D6 into a module-level `_LOADED: dict[str, Model]` keyed by `run_id`.** When a request supplies a `run_id` other than the default, the server lazy-loads it on first use and caches it in `_LOADED`. A request for a `nn_temporal` `run_id` raises 501 (D9) without loading. | **Single highest-leverage cut from the Scope Diff.** The lead's draft loaded all registered runs at startup; the minimalist flagged this as `PLAN POLISH` because (i) AC-5 / AC-3 only require one model loaded, (ii) eager loading forces all registry-resident model families to be loadable at startup including `NnTemporalModel`, and (iii) eager loading turns one error path (default model load fails) into N error paths with N tests. Lazy-on-demand for non-default runs cuts the startup error surface to one model. | Scope Diff §5 (single highest-leverage cut). Domain research §R7 (`lifespan` pattern). |
| **D8** | Datetime handling | **`AwareDatetime` Pydantic field for the request's `target_dt`. Normalise via `.astimezone(datetime.UTC)` at the handler boundary before constructing the feature-frame index.** Naive datetimes raise 422 (Pydantic-default behaviour); identity comparisons of tzinfo objects are explicitly avoided per pydantic#8683. | AC-2b (clear error on invalid input) and AC-3 (parity) both require timezone correctness. `AwareDatetime` rejects naive inputs at the validation boundary; the explicit `.astimezone(UTC)` neutralises the three known Pydantic-tzinfo pitfalls (#8683, #6592, #9571). | Domain research §R2; Scope Diff D8 (RESTATES INTENT). |
| **D9** | `NnTemporalModel` is out of scope | **A request whose resolved `run_id` points to a `nn_temporal` run returns HTTP 501 with body `{"detail": "Stage 12 does not serve nn_temporal runs; the seq_len-window + warmup_features contract requires a stateful predict path deferred to a future stage."}`. The default-model selector D6 also excludes `nn_temporal` from the candidate set so a fresh registry-only of `nn_temporal` runs does not produce a service that 501s on every request.** | Intent does not name `NnTemporalModel`; OQ-4 in the requirements identifies its warmup semantics as a distinct design problem. 501 is the HTTP-correct verb ("not implemented"); a structured detail string makes the boundary lesson visible (the demo facilitator can curl a `nn_temporal` run and *see* the deferral as a teaching point). | Requirements OQ-4; Stage 11 D5 (warmup_features contract); Scope Diff D9 (RESTATES INTENT). |
| **D10** | Serialisation security | **Defer `skops.io` adoption with a documented note in the layer doc and the registry's own forward-look CLAUDE comment. Stage 12 ships joblib-based artefact loading unchanged from Stage 9.** | Intent §Out of scope: "Deployment anywhere other than localhost." The security concern only matters when receiving artefacts from untrusted sources; localhost-only single-author artefacts are not that. The registry layer doc's "Stage 12 inflection point" call-out is honoured by *making the deferral visible* rather than by adopting skops at this stage. The decision is explicit per the registry's flag. | Requirements NFR-8 / OQ-6; codebase map §8 (`skops` absent); Scope Diff D10 (RESTATES INTENT). |
| **D11** | Request and prediction logging | **Structured per-request logging via `loguru.logger.bind(...).info(...)` to stdout. Seven fields per record, matching the R8 minimum so Stage 18 can consume the log without a retrofit:** `request_id` (UUID4 generated per request), `model_name`, `model_run_id`, `target_dt` (UTC ISO 8601), `prediction` (float, MW), `latency_ms` (float), `feature_hash` (first 16 hex chars of `sha256(canonical_json(features_in))`). Default sink is stdout; no file rotation, no log shipper. The structured record is emitted via `logger.bind(...)` so the project's existing `loguru_caplog` test fixture can assert on the emitted fields. | Intent §Points: "Minimal logging is easy to add and useful for Stage 18's drift monitoring." The minimalist flagged the lead's five-field draft as `PLAN POLISH` and noted it was "neither complete enough for Stage 18 nor required by any AC." Disposition: **commit fully to the seven-field schema.** Cost difference between five and seven fields is two more `bind` keys; benefit is Stage 18 has a working consumer surface from day one and does not need to retrofit the log schema in the serving layer once it ships. The schema is documented in the layer doc; Stage 18 inherits a stable contract. | Requirements OQ-5; domain research §R8; Scope Diff D11 (PLAN POLISH → committed in full per minimalist's "either commit or cut"). |
| **~~D12~~** | ~~Column-set assertion at the serving boundary~~ | **CUT** per Scope Diff. `model.predict(pd.DataFrame([row]))` raises naturally on missing or wrong-named columns (every model family in the roster validates its input frame internally), and FastAPI returns a structured error to the caller. AC-2b (clear error on invalid input) is satisfied by the natural error path; an additional `set(df.columns) == set(model.metadata.feature_columns)` guard adds ~10 lines of inline validation and one test for a failure mode the framework already surfaces. | The `model.predict` natural error path is structurally adequate. Adding a guard above it is `PLAN POLISH` per the Scope Diff. If a Phase-3 reviewer surfaces a case where the framework error is opaque, the guard can be added — at that point it has acceptance-criterion grounding rather than speculative coverage. | Scope Diff D12 (PLAN POLISH); requirements AC-2b; domain research §R4. |
| **D13** | Standalone CLI | **`python -m bristol_ml.serving` launches `uvicorn` with `bristol_ml.serving.app:build_app` as the ASGI factory. `--registry-dir`, `--host`, `--port` are CLI flags (defaults: `data/registry`, `127.0.0.1`, `8000`). `--help` prints the resolved `ServingConfig` schema (NFR-7 / DESIGN §2.1.1).** | DESIGN §2.1.1 binding: every module runs standalone via `python -m bristol_ml.<module>`. The launcher is `argparse + uvicorn.run(...)` — no Hydra wrapper, mirroring the project's existing `python -m bristol_ml.train` pattern (codebase map §1). | DESIGN §2.1.1; codebase map §1; requirements NFR-7; Scope Diff D13 (RESTATES INTENT). |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-1** | Schema discoverability — `GET /openapi.json` returns the OpenAPI 3.0 document with the `/predict` path's `requestBody` and `responses` schemas as JSON Schema objects. | Intent AC-4; domain research §R2; Scope Diff (RESTATES INTENT). |
| **NFR-2** | Standalone CLI — `uv run python -m bristol_ml.serving --help` exits 0 and prints the resolved `ServingConfig` schema; `uv run python -m bristol_ml.serving --registry-dir ...` (with a populated registry) starts the service and serves requests. | DESIGN §2.1.1; requirements NFR-7. |
| **NFR-3** | Notebooks thin — no notebook is added (notebook is `PLAN POLISH` per the Scope Diff §4; a curl example in the layer doc satisfies the intent's "or" clause). | DESIGN §2.1.8; Scope Diff §4. |
| **NFR-4** | Single-request latency — **no CI assertion.** Observed latency is recorded in the retro for each registered model family. The intent says latency "is not a goal here"; the Scope Diff cuts the lead's draft latency NFR as `PREMATURE OPTIMISATION`. | Intent §Points; Scope Diff NFR-1 (cut). |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` — Stage 12 adds the `src/bristol_ml/serving/` package. | **Defer.** User clarified on 2026-04-24 (post-Stage-10 Ctrl+G): "§6 is intended as structural-only, should only need to be updated very occasionally (ie not every new package)." The serving directory is already named in §6 from the Stage-0 scaffold (`serving/ # (deferred — Stage 12)`); the CLAUDE.md status-flip from "deferred" to "shipped" is the only edit needed. **DENY-tier on the lead — flag in the PR description for the human to land.** |

### Resolution log

- **Drafted 2026-04-25** — pre-Ctrl+G. All nine kept decisions (D1, D3–D11, D13) are proposed defaults; D2 retained but with the `[standard]` extra dropped per the minimalist; D7 retained but rewritten to load only the default model per the minimalist's single-highest-leverage cut; D12 cut.

### Decisions and artefacts explicitly **not** in Stage 12 (Scope Diff cuts + intent out-of-scope)

- **D12 cut** — column-set assertion at boundary. See D12 row.
- **`[standard]` extra cut** — `fastapi` + `uvicorn` only.
- **Eager-load all runs cut** — `_LOADED` starts with the single default; non-default runs lazy-load on first request. *Single highest-leverage cut.*
- **Latency CI assertion** — cut per Scope Diff `PREMATURE OPTIMISATION`; observed latency recorded in retro only.
- **`/health` endpoint** — cut per Scope Diff (`PLAN POLISH`); AC-1 (zero-config startup) is asserted by `TestClient` reaching the lifespan, not by a dedicated health URL. `GET /` returns the default-model summary (D6) and is sufficient as a liveness probe for the demo.
- **Notebook** — cut per Scope Diff `PLAN POLISH`. Curl example in `src/bristol_ml/serving/CLAUDE.md` and the README satisfies intent §Scope's "or curl example" clause.
- **`NnTemporalModel` serving** — deferred per D9.
- **`skops.io` adoption** — deferred per D10 with documented note.
- **Authentication / authorisation / rate limiting** (intent §Out of scope explicit).
- **Multi-user concurrent request handling at scale** (intent §Out of scope explicit).
- **Deployment anywhere other than localhost** (intent §Out of scope explicit).
- **A UI beyond the framework's auto-docs** (intent §Out of scope explicit).
- **Batch prediction endpoints** (intent §Out of scope explicit).
- **Model hot-reload** (intent §Out of scope explicit).
- **HTTPS** (intent §Out of scope, explicitly deferred).
- **Model versioning semantics beyond "ask for a model by name"** (intent §Out of scope, explicitly deferred).
- **A SARIMAX-roundtrip regression guard for statsmodels#6542** — cut per Scope Diff `PREMATURE OPTIMISATION` (resolved upstream in statsmodels 0.13+; the project is on 0.14.x and AC-3 already exercises a load → predict path on the registered model).

---

## 2. Scope

### In scope

Transcribed from `docs/intent/12-serving.md §Scope`:

- **A minimal HTTP application with a single prediction endpoint that loads a named model from the Stage 9 registry and returns a forecast for a given set of inputs** — `POST /predict` (D5), default model resolved at startup (D6), lazy-load on demand (D7), features-in request body (D4).
- **Input and output schemas for the prediction endpoint, defined typed and validated at the boundary** — Pydantic v2 `PredictRequest` / `PredictResponse` models; FastAPI auto-emits JSON Schema at `/openapi.json` (NFR-1). `AwareDatetime` for `target_dt` (D8).
- **A small notebook or curl example showing the service answering a request end-to-end** — curl example, *not* a notebook (Scope Diff §4 disposition). The example lives in `src/bristol_ml/serving/CLAUDE.md` and the project README.
- **Documentation of how to start the service locally and how to ask it questions** — layer doc at `docs/architecture/layers/serving.md` (new); module guide at `src/bristol_ml/serving/CLAUDE.md` (new); README section linking to both.

Additionally in scope as direct consequences of the above:

- **`bristol_ml.serving` package** — `__init__.py`, `__main__.py`, `app.py`, `schemas.py`, `CLAUDE.md`.
- **Hydra config schema** — `ServingConfig` in `conf/_schemas.py`; `conf/serving.yaml`. Carries `registry_dir: Path`, `host: str`, `port: int`. Surfaced as `AppConfig.serving: ServingConfig | None = None` so the existing train CLI is unaffected.
- **Layer doc** — `docs/architecture/layers/serving.md` (new layer).
- **Standalone CLI launcher** — `python -m bristol_ml.serving` (D13).
- **Per-request structured log** — seven-field `loguru.bind` record (D11).

### Out of scope (do not accidentally implement)

Restated from `docs/intent/12-serving.md §Out of scope` and §Out of scope, explicitly deferred + items surfaced by discovery and the Scope Diff:

- **Authentication, authorisation, rate limiting** (intent §Out of scope explicit).
- **Multi-user concurrent request handling at scale** (intent §Out of scope explicit).
- **Deployment anywhere other than localhost** (intent §Out of scope explicit).
- **A UI beyond the HTTP framework's auto-docs** (intent §Out of scope explicit).
- **Batch prediction endpoints** (intent §Out of scope explicit).
- **Model hot-reload** (intent §Out of scope explicit).
- **HTTPS** (intent §Out of scope, explicitly deferred).
- **Model versioning semantics beyond "ask for a model by name"** (intent §Out of scope, explicitly deferred).
- **`NnTemporalModel` serving** — deferred per D9 (returns 501 with structured teaching error).
- **`skops.io` adoption at this stage** — deferred per D10.
- **Raw-inputs / assembler-in-the-loop request schema** — deferred per D4; features-in only.
- **`/health` endpoint** — cut.
- **A notebook** — cut; curl example in CLAUDE.md only.
- **Column-set boundary assertion** — D12 cut.
- **Latency assertion in CI** — NFR-4 cut.
- **A SARIMAX save/load regression test** — cut.
- **Eager loading of all registry runs at startup** — D7 single-highest-leverage cut.
- **The `fastapi[standard]` extras (`jinja2`, `python-multipart`, `fastapi-cli`)** — cut.

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/12-serving.md`](../../intent/12-serving.md) — the contract; 5 ACs and 7 "Points for consideration".
2. [`docs/lld/research/12-serving-requirements.md`](../../lld/research/12-serving-requirements.md) — US-1..US-6, AC-1..AC-5, NFR-1..NFR-8, OQ-1..OQ-6. OQ-1 through OQ-6 resolved by the decisions above (OQ-1=lowest-MAE per D6; OQ-2=features-in per D4; OQ-3=single endpoint with `run_id` per D5; OQ-4=501 per D9; OQ-5=seven-field log per D11; OQ-6=defer per D10).
3. [`docs/lld/research/12-serving-codebase.md`](../../lld/research/12-serving-codebase.md) — registry API surface, `Model.predict` contract, hazards (UTC tz guard on SARIMAX, parametric; `NnTemporalModel` warmup; `loguru` adapter fixture; httpx already present).
4. [`docs/lld/research/12-serving-domain.md`](../../lld/research/12-serving-domain.md) — §R1 (FastAPI vs alternatives), §R2 (Pydantic v2 + JSON pitfalls), §R4 (training-serving skew literature), §R5 (features-in vs inputs-in), §R7 (lifespan model-loading pattern), §R8 (Stage-18 log fields).
5. [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md) — `@minimalist` critique; every cut and retention above is listed there.
6. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
7. `docs/architecture/layers/registry.md` — the upstream contract (`save`, `load`, `list_runs`, `describe`).
8. `src/bristol_ml/registry/CLAUDE.md` — concrete registry surface; `DEFAULT_REGISTRY_DIR`, sidecar contract, `feature_columns` field projection.
9. `src/bristol_ml/models/protocol.py` — `Model.predict(features: pd.DataFrame) -> pd.Series`; what the endpoint must marshal.
10. `src/bristol_ml/train.py` — argparse + `_cli_main` pattern that `bristol_ml.serving.__main__` mirrors.
11. `tests/conftest.py` — `loguru_caplog` fixture for asserting on bound log fields.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five intent-ACs are copied verbatim from `docs/intent/12-serving.md §Acceptance criteria`, then grounded in one or more named tests.

- **AC-1.** "The service starts on a clean machine without configuration beyond pointing at a registry location."
  - Tests:
    - `test_serving_lifespan_starts_with_only_registry_dir` — instantiate `build_app(registry_dir=tmp_path)` against a registry containing a single fixture run, exercise `with TestClient(app) as client: client.get("/")` returns HTTP 200 with `default_run_id` populated.
    - `test_serving_lifespan_raises_on_empty_registry` — registry directory is empty; lifespan startup raises `RuntimeError` whose message names the registry directory and identifies the empty filter.

- **AC-2.** "A request with valid inputs returns a prediction; a request with invalid inputs returns a clear error."
  - Tests:
    - `test_predict_valid_request_returns_200_with_prediction` — `TestClient` POSTs a fixture body satisfying `PredictRequest`; asserts HTTP 200 and `response.json()["prediction"]` is a finite float.
    - `test_predict_missing_required_field_returns_422` — POST without `target_dt`; asserts HTTP 422 and at least one `loc=["body", "target_dt"]` entry in the response body's `detail` array.
    - `test_predict_naive_datetime_returns_422` — POST with `target_dt="2025-06-15T13:00:00"` (no `Z` or offset); asserts 422 with a tz-related `msg`.
    - `test_predict_run_id_for_nn_temporal_returns_501` — POST with `run_id` resolving to a registered `nn_temporal` run; asserts HTTP 501 and the `detail` string from D9.

- **AC-3.** "The same model served by the service produces predictions identical to those produced by loading and running the model directly."
  - Tests:
    - `test_serving_prediction_parity_vs_direct_load` — load the fixture model directly via `registry.load(run_id)`, call `model.predict(X_one_row)`. POST the same row through `TestClient`. Assert `numpy.allclose(direct, served, atol=1e-5)` (the registry MLflow-adapter precedent bar). Covers `nn_mlp` as the model with the most non-trivial normalisation surface; one parametrised variant covers `linear` as a sanity baseline.

- **AC-4.** "Input/output schemas are machine-readable (the HTTP framework's schema export is sufficient)."
  - Tests:
    - `test_openapi_json_contains_predict_request_and_response` — `client.get("/openapi.json")` returns 200; assert `paths["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]` is present and `paths["/predict"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]` is present.

- **AC-5.** "A smoke test exercises the endpoint with a small fixture."
  - Tests:
    - The integration test file `tests/integration/serving/test_api.py` itself satisfies AC-5 — every test above runs against a `TestClient` with a tmp-registry fixture; no live registry, no network, no special pytest marker.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_default_model_is_lowest_mae_among_eligible_runs` (D6 — registry contains three runs of mixed model types, including one `nn_temporal`; `default_run_id` resolves to the non-`nn_temporal` lowest-MAE run).
- `test_request_log_record_carries_seven_fields` (D11 — uses `loguru_caplog`; asserts `request_id`, `model_name`, `model_run_id`, `target_dt`, `prediction`, `latency_ms`, `feature_hash` all present and well-typed).
- `test_serving_cli_help_exits_zero` (NFR-2 — `python -m bristol_ml.serving --help` exits 0; the resolved `ServingConfig` schema is in stdout).
- `test_serving_config_round_trips_through_hydra` — `conf/serving.yaml` defaults match `ServingConfig(...)` exactly.
- `test_lazy_load_caches_run_id_after_first_request` (D7 — second request with the same non-default `run_id` does not re-call `registry.load`; verified via a `mock.patch.object(registry, "load", wraps=...)` call-count assertion).
- `test_nn_temporal_run_id_does_not_trigger_load` (D7 + D9 — `mock.patch.object(registry, "load")` confirms `load` is never called on a `nn_temporal` request; the 501 returns before `_LOADED` is touched).
- `test_serving_logs_default_run_id_at_startup` — lifespan emits a structured `loguru` info line containing `default_run_id` and the resolved `model_name` so the operator sees what is being served.

**Total shipped tests: 13** (one AC-1 with two sub-tests, AC-2 four sub-tests, AC-3 one sub-test parametrised over two model families, AC-4 one sub-test, AC-5 satisfied by the file itself, seven D-derived).

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/serving/
├── __init__.py          # exports: build_app, __version__
├── __main__.py          # python -m bristol_ml.serving (argparse + uvicorn.run)
├── app.py               # FastAPI factory + lifespan + /predict + /
├── schemas.py           # Pydantic PredictRequest / PredictResponse
└── CLAUDE.md            # module guide; carries the curl example
```

`build_app(registry_dir: Path) -> FastAPI` is the public surface. The lifespan loads the default model via D6 (lowest-MAE, `nn_temporal`-excluded) and stashes it in `app.state.loaded: dict[str, Model]` keyed by `run_id`. Subsequent non-default `run_id`s lazy-load on first use and cache.

### Public API

```python
# src/bristol_ml/serving/schemas.py
from __future__ import annotations
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dt: AwareDatetime = Field(
        description="UTC-aware target timestamp the forecast is for.",
    )
    features: dict[str, float] = Field(
        description=(
            "Assembled feature row matching model.metadata.feature_columns. "
            "Keys must match exactly; values are the assembled feature values."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Registry run id. None resolves to the default model selected at "
            "startup (lowest-MAE eligible run; nn_temporal excluded)."
        ),
    )

class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: float = Field(description="Forecast value in MW.")
    run_id: str = Field(description="The resolved run id used to serve this request.")
    model_name: str = Field(description="Human-readable model name (e.g. 'NaiveModel').")
    target_dt: AwareDatetime = Field(description="Echo of the request target_dt, normalised to UTC.")
```

### Endpoint surface

```python
# src/bristol_ml/serving/app.py (essentials only)
from __future__ import annotations
import datetime as dt, hashlib, json, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from bristol_ml import registry
from bristol_ml.models import Model
from bristol_ml.serving.schemas import PredictRequest, PredictResponse


def _select_default_run_id(registry_dir: Path) -> tuple[str, str, list[str]]:
    runs = registry.list_runs(registry_dir=registry_dir, sort_by="mae", ascending=True)
    eligible = [r for r in runs if r.type != "nn_temporal"]
    if not eligible:
        raise RuntimeError(
            f"Registry at {registry_dir} contains no eligible runs "
            f"(found {len(runs)} runs; all are nn_temporal which Stage 12 defers)."
        )
    chosen = eligible[0]
    return chosen.run_id, chosen.name, list(chosen.feature_columns)


def build_app(registry_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        run_id, name, fcols = _select_default_run_id(registry_dir)
        model: Model = registry.load(run_id, registry_dir=registry_dir)
        logger.info(
            "serving lifespan: default model resolved",
            default_run_id=run_id, model_name=name,
        )
        app.state.loaded = {run_id: model}
        app.state.default_run_id = run_id
        app.state.default_name = name
        app.state.default_feature_columns = fcols
        app.state.registry_dir = registry_dir
        yield

    app = FastAPI(title="bristol_ml serving", lifespan=lifespan)

    @app.get("/")
    def root() -> dict:
        return {
            "default_run_id": app.state.default_run_id,
            "model_name": app.state.default_name,
            "feature_columns": app.state.default_feature_columns,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        run_id = req.run_id or app.state.default_run_id
        # D9: nn_temporal runs return 501 before any load
        described = registry.describe(run_id, registry_dir=app.state.registry_dir)
        if described["type"] == "nn_temporal":
            raise HTTPException(
                status_code=501,
                detail=(
                    "Stage 12 does not serve nn_temporal runs; the seq_len-window "
                    "+ warmup_features contract requires a stateful predict path "
                    "deferred to a future stage."
                ),
            )
        # D7: lazy-load
        if run_id not in app.state.loaded:
            app.state.loaded[run_id] = registry.load(run_id, registry_dir=app.state.registry_dir)
        model = app.state.loaded[run_id]

        # D8: tz-normalise
        target_utc = req.target_dt.astimezone(dt.UTC)
        # D4: features-in
        df = pd.DataFrame([req.features], index=pd.DatetimeIndex([target_utc], tz="UTC"))
        t0 = time.perf_counter()
        prediction = float(model.predict(df).iloc[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # D11: structured log
        feature_hash = hashlib.sha256(
            json.dumps(req.features, sort_keys=True).encode()
        ).hexdigest()[:16]
        logger.bind(
            request_id=str(uuid.uuid4()),
            model_name=described["name"],
            model_run_id=run_id,
            target_dt=target_utc.isoformat(),
            prediction=prediction,
            latency_ms=latency_ms,
            feature_hash=feature_hash,
        ).info("served prediction")

        return PredictResponse(
            prediction=prediction,
            run_id=run_id,
            model_name=described["name"],
            target_dt=target_utc,
        )

    return app
```

### Config schema (addition)

```python
# conf/_schemas.py
class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_dir: Path = Path("data/registry")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
```

`AppConfig` gains:

```python
serving: ServingConfig | None = None
```

`None` keeps the train CLI / config-smoke surface unchanged for callers that don't need serving. `conf/serving.yaml`:

```yaml
# @package serving
registry_dir: data/registry
host: 127.0.0.1
port: 8000
```

### Standalone CLI

```
$ uv run python -m bristol_ml.serving --help
usage: serving [-h] [--registry-dir REGISTRY_DIR] [--host HOST] [--port PORT] [overrides ...]

Start the bristol_ml prediction service.

positional arguments:
  overrides             Hydra-style config overrides (forwarded to load_config).

options:
  -h, --help            show this help message and exit
  --registry-dir DIR    Override conf/serving.yaml registry_dir.
  --host HOST           Override conf/serving.yaml host.
  --port PORT           Override conf/serving.yaml port.
```

`__main__.py` resolves `ServingConfig` via `load_config`, then calls `uvicorn.run("bristol_ml.serving:build_app", factory=True, host=..., port=...)`. `factory=True` is necessary so the lifespan is recreated per worker; the `build_app` call must not require positional args at the uvicorn boundary, so a thin wrapper `_factory(registry_dir)` is closed over the resolved `ServingConfig` and passed in.

---

## 6. Tasks (sequential — see CLAUDE.md §Phase 2 for sequencing rules)

Each task ends with one or more pytest invocations and a single git commit citing this plan task number. The tester is spawned alongside or before each task per CLAUDE.md §Tester timing.

- **T1 — Dependencies + package scaffold.**
  1. Add `fastapi` and `uvicorn` to `pyproject.toml [project.dependencies]` (no `[standard]` extra per D2).
  2. `uv lock` to refresh `uv.lock`.
  3. Create `src/bristol_ml/serving/{__init__.py, __main__.py, app.py, schemas.py, CLAUDE.md}` skeletons. `__init__.py` exports `build_app`. `__main__.py` is the argparse launcher (skeleton only — uvicorn wiring lands at T3).
  4. `tests/{unit,integration}/serving/__init__.py` empty.
  - **Tests:** `test_serving_module_imports_without_torch` — `import bristol_ml.serving` does not pull `torch` (mirrors the Stage 11 lazy-import discipline; Stage 12 has no torch dep at all so this is a guard-by-construction test).
  - **Commit:** `Stage 12 T1: bristol_ml.serving scaffold + fastapi/uvicorn deps`.

- **T2 — Pydantic schemas + Hydra config.**
  1. `src/bristol_ml/serving/schemas.py` with `PredictRequest` / `PredictResponse` per §5.
  2. `conf/_schemas.py` gains `ServingConfig`; `AppConfig.serving: ServingConfig | None = None`.
  3. `conf/serving.yaml` per §5.
  4. `bristol_ml.config.validate` already handles new top-level fields without code change (verify via the existing test).
  - **Tests:** `test_predict_request_rejects_naive_datetime`, `test_predict_request_round_trips_features_dict`, `test_serving_config_round_trips_through_hydra`, `test_app_config_serving_default_is_none_so_train_cli_unaffected`.
  - **Commit:** `Stage 12 T2: ServingConfig + PredictRequest/Response schemas`.

- **T3 — App factory + lifespan + `/`.**
  1. `build_app(registry_dir)` per §5; lifespan resolves D6 default and stashes in `app.state`.
  2. `_select_default_run_id` excludes `nn_temporal` from candidates per D9.
  3. `GET /` returns the default-model summary.
  4. Empty-registry → `RuntimeError` from the lifespan with a clear message.
  - **Tests:** `test_serving_lifespan_starts_with_only_registry_dir`, `test_serving_lifespan_raises_on_empty_registry`, `test_default_model_is_lowest_mae_among_eligible_runs` (mixed registry: linear, nn_mlp, nn_temporal — the nn_temporal one has the lowest MAE; service must pick the second-lowest non-temporal run), `test_serving_logs_default_run_id_at_startup`.
  - **Commit:** `Stage 12 T3: app factory + lifespan + GET /`.

- **T4 — `/predict` happy path (D4 + D5 + D8).**
  1. `POST /predict` per §5 — features-in, `AwareDatetime` + UTC normalise, default-or-supplied `run_id`, `model.predict(...).iloc[0]`.
  2. `PredictResponse` populated; `target_dt` echoed normalised.
  3. Lazy-load on first non-default `run_id` per D7; cache in `app.state.loaded`.
  - **Tests:** `test_predict_valid_request_returns_200_with_prediction`, `test_predict_naive_datetime_returns_422`, `test_predict_missing_required_field_returns_422`, `test_serving_prediction_parity_vs_direct_load` (parametrised over `linear` and `nn_mlp`), `test_lazy_load_caches_run_id_after_first_request`.
  - **Commit:** `Stage 12 T4: POST /predict happy + invalid-input paths`.

- **T5 — `nn_temporal` 501 + structured logging (D9 + D11).**
  1. `registry.describe(run_id)` short-circuit before load; raise `HTTPException(501, detail=...)` with the D9 message.
  2. `logger.bind(...).info("served prediction")` with seven fields per D11.
  3. UUID4 request_id; sha256-truncated feature_hash; `time.perf_counter()` latency.
  - **Tests:** `test_predict_run_id_for_nn_temporal_returns_501`, `test_nn_temporal_run_id_does_not_trigger_load`, `test_request_log_record_carries_seven_fields`.
  - **Commit:** `Stage 12 T5: nn_temporal 501 + seven-field structured log`.

- **T6 — Standalone CLI + OpenAPI smoke test (D13 + NFR-1 + NFR-2).**
  1. `__main__.py` argparse + uvicorn wiring per §5.
  2. `--help` prints resolved `ServingConfig` schema.
  3. OpenAPI JSON test asserts `/predict` schemas are present.
  - **Tests:** `test_serving_cli_help_exits_zero`, `test_openapi_json_contains_predict_request_and_response`.
  - **Commit:** `Stage 12 T6: standalone CLI + OpenAPI smoke test`.

- **T7 — Layer doc + module guide + README + retro skeleton.**
  1. `docs/architecture/layers/serving.md` (new layer) — surface (build_app, PredictRequest/Response, run_id semantics), training-serving-skew teaching point, the seven-field log contract for Stage 18, the curl example.
  2. `src/bristol_ml/serving/CLAUDE.md` — concrete module guide; the curl one-liner; the `nn_temporal` 501 deferral note; the `skops.io` deferral note.
  3. README — add a "Worked example: serving (Stage 12)" section with the curl one-liner.
  4. `docs/lld/stages/12-serving.md` — retro skeleton; observed latency per model family recorded here per NFR-4.
  5. `CHANGELOG.md` — `### Added` bullets.
  - **Tests:** none (doc edits).
  - **Commit:** `Stage 12 T7: layer doc + module guide + README + retro skeleton`.

- **T8 — Stage hygiene + plan move.**
  1. `git mv docs/plans/active/12-serving.md docs/plans/completed/12-serving.md`.
  2. Final retro updates: actual observed latencies; any decisions deviated from in-stage.
  3. `uv run pytest -q` clean; `uv run ruff check .` clean; `uv run ruff format --check .` clean; `uv run pre-commit run --all-files` clean.
  - **Commit:** `Stage 12 T8: stage hygiene + retro + plan moved to completed/`.

Phase 3 (review):

- **T9 — Phase 3 review.** Spawn `arch-reviewer` (conformance to plan + intent), `code-reviewer` (code quality + security), `docs-writer` (user + developer docs sweep) in parallel. Synthesise findings, address Blocking items in-branch, surface Major+Minor in the PR description.

---

## 7. Exit checklist

Before opening the PR:

- [ ] All thirteen named tests in §4 pass; full `uv run pytest -q` is clean.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `uv run python -m bristol_ml.serving --help` exits 0 with the `ServingConfig` schema printed.
- [ ] Layer doc `docs/architecture/layers/serving.md` exists; module guide `src/bristol_ml/serving/CLAUDE.md` exists with the curl one-liner.
- [ ] README has the worked-example section.
- [ ] `CHANGELOG.md` updated under `[Unreleased] ### Added`.
- [ ] Retro at `docs/lld/stages/12-serving.md` carries observed latencies per registered model family.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/`.
- [ ] PR description surfaces:
  - The `D9` deferral (`nn_temporal` 501) — load-bearing teaching point.
  - The `D10` deferral (`skops.io`) — load-bearing security note.
  - The `H-1` DENY-tier ask — DESIGN.md §6 status flip from "deferred — Stage 12" to "shipped" needs human approval (lead is DENY-tier on intent).
  - The `D7` single-highest-leverage-cut framing — for the next stage's plan author.
