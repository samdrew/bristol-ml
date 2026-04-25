# Plan — Stage 12: Minimal serving endpoint

**Status:** `approved` — Ctrl+G amendments (D9 + D10) folded 2026-04-25; ready for Phase 2.
**Intent:** [`docs/intent/12-serving.md`](../../intent/12-serving.md)
**Upstream stages shipped:** Stages 0–11 (foundation → ingestion → features → six model families → enhanced evaluation → registry → MLP → TCN). Stage 11 is in flight on a sibling branch; Stage 12 branches from `main` and consumes only the Stage 9 registry surface, so the dependency is `Stage 9` not `Stage 11`.
**Downstream consumers:** Stage 18 (drift monitoring — consumes the Stage 12 prediction log), Stage 19 (orchestration — schedules batch scoring through the same artefacts).
**Baseline SHA:** `6ad2d7a` (tip of `main` after the DESIGN §6 `docs/plans/` addition — PR #9).

**Discovery artefacts produced in Phase 1:**

- Requirements — [`docs/lld/research/12-serving-requirements.md`](../../lld/research/12-serving-requirements.md)
- Codebase map — [`docs/lld/research/12-serving-codebase.md`](../../lld/research/12-serving-codebase.md)
- Domain research — [`docs/lld/research/12-serving-domain.md`](../../lld/research/12-serving-domain.md)
- Scope Diff — [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below; the Ctrl+G amendment note at the head of that file records D9 + D10 reversals)

**Pedagogical weight.** Intent §Demo moment names a one-minute facilitator loop: start the service, curl it with a sample payload, get a forecast back. The pedagogical point is the boundary itself — a model that was trained in a notebook now answers HTTP requests in a separate process — not the framework, not the schema, not the latency. Every decision below is filtered through "does this make the boundary more legible to the meetup audience?"; anything that does not is cut by the Scope Diff or deferred.

**Stage 11 reach.** `NnTemporalModel` (Stage 11) was the only model in the roster that needed sequence state at predict time, but Stage 11 D5+ already saved the `warmup_features` window inside the joblib envelope as part of the model artefact, so a loaded `NnTemporalModel.predict(single_row)` works without the caller knowing about warmup. Stage 12 inherits that contract for free: at the serving boundary `nn_temporal` looks identical to the five stateless families and is **served, not deferred** (Ctrl+G reversal of the original D9 — see §1). All six model families — `naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal` — are in scope.

**Security posture (skops).** A network-facing predict endpoint is a network-facing deserialiser. `joblib.load` on an artefact authored by an untrusted party is trivially RCE; even on a localhost-only deployment the threat surface is non-zero (the serving process loads any artefact in the registry directory, and the registry directory is a path the operator may share). Per the Ctrl+G amendment (D10 reversal), Stage 12 migrates all six model families' save/load paths off `joblib` and onto `skops.io` as the canonical primitive, with the envelope-of-bytes pattern for the two model families (`linear`, `sarimax`) whose statsmodels results objects do not round-trip cleanly through skops's restricted unpickler. The migration is destructive to existing `data/registry/` artefacts; this is acceptable for a pre-prod codebase per the user's explicit prioritisation of security over backward compat.

---

## 1. Decisions for the human (resolved at Ctrl+G — recorded for Phase 2)

Thirteen decision points: eleven retained from the pre-Ctrl+G draft, two reversed at Ctrl+G review (D9 + D10), and D12 cut per the Scope Diff. The decision set was filtered through the `@minimalist` Scope Diff in [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md), and three tags were flipped from the lead's draft framing pre-Ctrl+G — **D2** (`fastapi[standard]` extra → drop the extra), **D7** (load all registered runs → load only the default model — *single highest-leverage cut*), **D12 cut** (column-set boundary assertion). At Ctrl+G the human reversed the minimalist's `RESTATES INTENT` framing on **D9** ("nn_temporal should be in scope, if the interface is manageable. Keeping code in the codebase that isn't supported in the primary implementation is bad-practice.") and **D10** ("Include skops. This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC."). The minimalist's critique snapshot is preserved at the head of the Scope Diff file. Defaults lean on the three research artefacts and the simplicity bias in `DESIGN.md §2.2.4`. The Evidence column cites the research that *resolved* each decision.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | HTTP framework | **FastAPI.** No alternative considered — `DESIGN.md §8` names FastAPI explicitly. | The decision is already recorded in the spec; the Phase-1 research confirms it (R1). | DESIGN.md §8; domain research §R1. Scope Diff D1 (RESTATES INTENT). |
| **D2** | Dependency footprint | **`fastapi` + `uvicorn` (no `[standard]` extra).** `httpx` is already a runtime dep (Stage 1 ingestion), so `fastapi.testclient.TestClient` works without `[standard]`. The extras `[standard]` would pull (`jinja2`, `python-multipart`, `fastapi-cli`) are not required by any AC. | Scope Diff `PLAN POLISH` flag flipped: minimum viable framework surface. Three transitive packages saved with no AC coverage cost. | Domain research §R1; codebase map §8 (httpx already present); Scope Diff D2 (PLAN POLISH → cut the extras). |
| **D3** | Predict handler shape | **Synchronous `def predict(...)`** — not `async def`. CPU inference has no I/O to await; FastAPI runs sync handlers in a threadpool transparently. | Async without genuine awaitable work would mislead readers about why async is used (§2.2.4 simplicity bias). All handlers in the project are sync; the serving layer follows the same convention. | Domain research §R1; Scope Diff D3 (RESTATES INTENT). |
| **D4** | Request schema | **"Features-in" — the request body carries an already-assembled feature row whose keys must match `model.metadata.feature_columns` exactly. The server calls `model.predict(pd.DataFrame([row], index=[target_dt]))` and returns the scalar.** Raw-inputs / assembler-in-the-loop is deferred (intent §Points names both as defensible; Stage 12 takes the simpler one). | AC-3 (prediction parity) is trivially testable with features-in: the same row through the registry's direct `model.predict` and through the endpoint must match under `numpy.allclose(atol=1e-5)`. Raw-inputs introduces an assembler dependency that the intent does not require and that the codebase map flags as the canonical training-serving-skew surface. The "leakage of the training assumption" cost the requirements analyst notes is honestly acknowledged as a teaching point in the layer doc rather than engineered around. | Requirements OQ-2; domain research §R5; Scope Diff D4 (RESTATES INTENT). |
| **D5** | Endpoint shape | **Single `POST /predict` endpoint.** The request body carries an optional `run_id: str \| None` field — `None` means "use the default model selected at startup". Per-model endpoints (`/predict/linear`, etc.) are not added: a new model family must not require a serving-layer code change. | Intent §Scope: "a single prediction endpoint." Intent §Points: "a model-name parameter is more flexible." Codebase map §3: `registry.load(run_id)` is the natural call site; mirrors the registry's already-shipped surface. | Intent §Scope; requirements OQ-3; Scope Diff D5 (RESTATES INTENT). |
| **D6** | Default-model selection | **Lowest-MAE run across the entire registry — `registry.list_runs(sort_by="mae", ascending=True)[0]` — at startup. No model-family filter (Ctrl+G reversal of D9 means every family including `nn_temporal` is an eligible candidate).** If the registry is empty, the lifespan raises `RuntimeError` with a clear message naming the registry directory. The selected `run_id` is exposed via `GET /` as `{"default_run_id": "...", "model_name": "...", "feature_columns": [...]}` so the facilitator can curl-introspect it before issuing a predict. | AC-1: "no configuration beyond pointing at a registry location." Picking by MAE makes the demo honest — the service serves the best-scoring run, regardless of family, because `nn_temporal` is now first-class. Determinism is preserved within a registry directory (no new runs slip in during the demo unless someone trains in parallel). | Requirements OQ-1; intent §Points; Ctrl+G amendment to original D6 framing (no longer filters). |
| **D7** | Model loading at startup | **Load only the single default model selected by D6 into a module-level `_LOADED: dict[str, Model]` keyed by `run_id`.** When a request supplies a `run_id` other than the default, the server lazy-loads it on first use and caches it in `_LOADED`. With D9 reversed, all six model families are loadable through the same skops-safe path (D10), so lazy-load is uniform. | **Single highest-leverage cut from the Scope Diff.** The lead's draft loaded all registered runs at startup; the minimalist flagged this as `PLAN POLISH` because (i) AC-5 / AC-3 only require one model loaded, (ii) eager loading turns one error path (default model load fails) into N error paths with N tests, and (iii) startup latency scales with registry size. Lazy-on-demand for non-default runs cuts the startup error surface to one model. The cut still applies under Ctrl+G — it remains independent of which families are served. | Scope Diff §5 (single highest-leverage cut). Domain research §R7 (`lifespan` pattern). |
| **D8** | Datetime handling | **`AwareDatetime` Pydantic field for the request's `target_dt`. Normalise via `.astimezone(datetime.UTC)` at the handler boundary before constructing the feature-frame index.** Naive datetimes raise 422 (Pydantic-default behaviour); identity comparisons of tzinfo objects are explicitly avoided per pydantic#8683. | AC-2b (clear error on invalid input) and AC-3 (parity) both require timezone correctness. `AwareDatetime` rejects naive inputs at the validation boundary; the explicit `.astimezone(UTC)` neutralises the three known Pydantic-tzinfo pitfalls (#8683, #6592, #9571). | Domain research §R2; Scope Diff D8 (RESTATES INTENT). |
| **D9** | `NnTemporalModel` serving | **`nn_temporal` is in scope and served exactly like every other model family. No 501 path; no D6 candidate-set filter; no model-family special-cases at the request handler.** This works because Stage 11 D5+ already saved the `warmup_features` window inside the model artefact (the joblib envelope today, the skops envelope after T2–T4) so a loaded `NnTemporalModel` already exposes a working single-row `predict(X_one_row)` — the warmup buffer is part of the model state, not a per-request input. The serving layer treats `nn_temporal` as just another `Model`. | **Ctrl+G reversal of the pre-Ctrl+G D9.** Human rationale at review: "Keeping code in the codebase that isn't supported in the primary implementation is bad-practice." Stage 11 invested deliberately in baking warmup into the artefact specifically so the serving boundary would not need to know. Honouring that means the boundary cannot grow a `if isinstance(model, NnTemporalModel)` branch — every family goes through `Model.predict` uniformly. AC-3 (parity) is parametrised over all six families to enforce this. | Stage 11 D5+ (warmup_features envelope); requirements OQ-4 (now resolved by upstream design); Scope Diff D9 row reversed (see file's post-Ctrl+G note). |
| **D10** | Serialisation security | **Adopt `skops.io` as the canonical save/load primitive across all six model families. Joblib-based artefacts are no longer written; existing `data/registry/*.joblib` artefacts are invalidated by the migration.** Each family's `save` writes a `.skops` file and `load` reads it. For `naive`, `scipy_parametric`, `nn_mlp`, `nn_temporal`: native skops via the project's `get_untrusted_types` introspection (custom classes registered via `skops.io.add_to_safe`). For `linear` and `sarimax`: envelope-of-bytes pattern — `results.save(BytesIO)` to native statsmodels bytes, wrap in a dict envelope, `skops.io.dumps` the envelope. The serving layer's lazy-load path goes through `registry.load` which now resolves `.skops` files; no serving-layer code needs to know about the format. The registry layer doc's "Stage 12 inflection point" call-out is honoured by *adopting* skops, not by deferring it. | **Ctrl+G reversal of the pre-Ctrl+G D10.** Human rationale at review: "This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC." A network-facing predict endpoint is a network-facing deserialiser; `joblib.load` on an attacker-controlled artefact is RCE. Migrating now (one stage) is cheaper than migrating after an exploit (any stage). The envelope-of-bytes pattern for statsmodels models is the same pattern Stage 10/11 NN models pioneered, so the pattern itself is established. Backward compatibility with existing `data/registry/` artefacts is sacrificed; the user explicitly accepted this trade-off. | Requirements NFR-8 / OQ-6; codebase map §8 (`skops` absent — added at T1); Stage 10/11 envelope-of-bytes precedent; Scope Diff D10 row reversed (see file's post-Ctrl+G note). |
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

- **Drafted 2026-04-25 morning** — pre-Ctrl+G. Decisions D1, D3–D11, D13 proposed as defaults; D2 retained but `[standard]` extra dropped per minimalist; D7 retained but rewritten to load only the default model (single-highest-leverage cut); D12 cut. Original D9 deferred `nn_temporal` via 501; original D10 deferred `skops.io`.
- **Ctrl+G review 2026-04-25** — human reversed two rows. **D9 reversed** ("Keeping code in the codebase that isn't supported in the primary implementation is bad-practice"): `nn_temporal` is now served first-class via the Stage 11 D5+ warmup envelope; the 501 path is removed; the D6 candidate-set filter is removed. **D10 reversed** ("This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC"): `skops.io` is now the canonical save/load primitive across all six families, with envelope-of-bytes for `linear` + `sarimax`; existing `data/registry/*.joblib` artefacts are invalidated. The remaining decisions D1–D8, D11, D13 plus D12-cut were retained verbatim from the pre-Ctrl+G draft. Plan status: `draft` → `approved` and ready for Phase 2.

### Decisions and artefacts explicitly **not** in Stage 12 (Scope Diff cuts + intent out-of-scope)

- **D12 cut** — column-set assertion at boundary. See D12 row.
- **`[standard]` extra cut** — `fastapi` + `uvicorn` only.
- **Eager-load all runs cut** — `_LOADED` starts with the single default; non-default runs lazy-load on first request. *Single highest-leverage cut.*
- **Latency CI assertion** — cut per Scope Diff `PREMATURE OPTIMISATION`; observed latency recorded in retro only.
- **`/health` endpoint** — cut per Scope Diff (`PLAN POLISH`); AC-1 (zero-config startup) is asserted by `TestClient` reaching the lifespan, not by a dedicated health URL. `GET /` returns the default-model summary (D6) and is sufficient as a liveness probe for the demo.
- **Notebook** — cut per Scope Diff `PLAN POLISH`. Curl example in `src/bristol_ml/serving/CLAUDE.md` and the README satisfies intent §Scope's "or curl example" clause.
- **Joblib in the registry** — *removed*, not deferred. All six families flip to `skops.io` as part of Stage 12 (D10 reversal).
- **Authentication / authorisation / rate limiting** (intent §Out of scope explicit).
- **Multi-user concurrent request handling at scale** (intent §Out of scope explicit).
- **Deployment anywhere other than localhost** (intent §Out of scope explicit).
- **A UI beyond the framework's auto-docs** (intent §Out of scope explicit).
- **Batch prediction endpoints** (intent §Out of scope explicit).
- **Model hot-reload** (intent §Out of scope explicit).
- **HTTPS** (intent §Out of scope, explicitly deferred).
- **Model versioning semantics beyond "ask for a model by name"** (intent §Out of scope, explicitly deferred).
- **Backward compatibility with existing `data/registry/*.joblib` artefacts** — explicitly sacrificed at Ctrl+G; users with an active registry must retrain (the project is pre-prod, the registry is a developer artefact, and the security trade-off was made deliberately).
- **A SARIMAX-roundtrip regression guard for statsmodels#6542** — cut per Scope Diff `PREMATURE OPTIMISATION` (resolved upstream in statsmodels 0.13+; the project is on 0.14.x and AC-3 already exercises a load → predict path on the registered model).

---

## 2. Scope

### In scope

Transcribed from `docs/intent/12-serving.md §Scope`:

- **A minimal HTTP application with a single prediction endpoint that loads a named model from the Stage 9 registry and returns a forecast for a given set of inputs** — `POST /predict` (D5), default model resolved at startup (D6), lazy-load on demand (D7), features-in request body (D4). All six model families (`naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) are served — Ctrl+G amendment to D9.
- **Input and output schemas for the prediction endpoint, defined typed and validated at the boundary** — Pydantic v2 `PredictRequest` / `PredictResponse` models; FastAPI auto-emits JSON Schema at `/openapi.json` (NFR-1). `AwareDatetime` for `target_dt` (D8).
- **A small notebook or curl example showing the service answering a request end-to-end** — curl example, *not* a notebook (Scope Diff §4 disposition). The example lives in `src/bristol_ml/serving/CLAUDE.md` and the project README.
- **Documentation of how to start the service locally and how to ask it questions** — layer doc at `docs/architecture/layers/serving.md` (new); module guide at `src/bristol_ml/serving/CLAUDE.md` (new); README section linking to both.

Additionally in scope as direct consequences of the above:

- **`bristol_ml.serving` package** — `__init__.py`, `__main__.py`, `app.py`, `schemas.py`, `CLAUDE.md`.
- **Hydra config schema** — `ServingConfig` in `conf/_schemas.py`; `conf/serving.yaml`. Carries `registry_dir: Path`, `host: str`, `port: int`. Surfaced as `AppConfig.serving: ServingConfig | None = None` so the existing train CLI is unaffected.
- **Layer doc** — `docs/architecture/layers/serving.md` (new layer).
- **Standalone CLI launcher** — `python -m bristol_ml.serving` (D13).
- **Per-request structured log** — seven-field `loguru.bind` record (D11).
- **`skops.io` migration of all six model families' save/load** — Ctrl+G amendment to D10. Concretely:
    - `bristol_ml.models.io` gains `save_skops` / `load_skops` helpers wrapping `skops.io.dump` / `skops.io.load` with a project-level trust-list (custom classes from each family registered via `add_to_safe`); the joblib helpers are kept for one stage with a `DeprecationWarning` so any external scripts can complete a one-off migration before Stage 13.
    - Each family's `save(path)` writes `<path>.skops` (or accepts an explicit `.skops` path) instead of joblib; each family's `load(cls, path)` reads `.skops`.
    - `naive`, `scipy_parametric`, `nn_mlp`, `nn_temporal`: native skops via class-list registration. NN families' existing bytes-envelope (state_dict + scalers) is wrapped in skops-safe types (dicts of numpy / bytes / ints), not pickled.
    - `linear`, `sarimax`: envelope-of-bytes. `results.save(BytesIO())` → bytes; wrap `{"format": "statsmodels-bytes-v1", "kind": "OLS" | "SARIMAX", "blob": bytes, "feature_columns": [...]}` ; `skops.io.dumps(envelope)` → file. Load reverses: `skops.io.loads(file_bytes)` → dict; `BytesIO(dict["blob"])` → `RegressionResults.load(buf)` / `SARIMAXResults.load(buf)`. The envelope itself contains only skops-safe types.
    - Registry: filesystem layout flips from `<run_id>/model.joblib` to `<run_id>/model.skops`; `registry.load` / `registry.save` resolve the new extension. Sidecar JSON unchanged.
    - Existing `data/registry/*.joblib` artefacts are invalidated; the layer doc and `registry/CLAUDE.md` document the migration. Users must retrain.
    - Per-family round-trip tests added (save → load → predict) confirming numerical equality `numpy.allclose(pred_before, pred_after, atol=1e-6)` for each of the six families.

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
- **Backward compatibility with existing `data/registry/*.joblib` artefacts** — explicitly sacrificed at Ctrl+G (D10 reversal).
- **Raw-inputs / assembler-in-the-loop request schema** — deferred per D4; features-in only.
- **`/health` endpoint** — cut.
- **A notebook** — cut; curl example in CLAUDE.md only.
- **Column-set boundary assertion** — D12 cut.
- **Latency assertion in CI** — NFR-4 cut.
- **A SARIMAX save/load regression test for statsmodels#6542 specifically** — cut (per-family skops round-trip tests cover the regression naturally).
- **Eager loading of all registry runs at startup** — D7 single-highest-leverage cut.
- **The `fastapi[standard]` extras (`jinja2`, `python-multipart`, `fastapi-cli`)** — cut.
- **A migration script that rewrites existing joblib artefacts in-place** — explicitly out of scope. Users retrain.

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
    - `test_predict_unknown_run_id_returns_404` — POST with `run_id="does-not-exist"`; asserts 404 with a clear `detail` naming the registry directory and the missing run_id (replaces the deleted `nn_temporal` 501 test; provides equivalent "clear error" coverage on the run_id resolution path).

- **AC-3.** "The same model served by the service produces predictions identical to those produced by loading and running the model directly."
  - Tests:
    - `test_serving_prediction_parity_vs_direct_load` — load the fixture model directly via `registry.load(run_id)`, call `model.predict(X_one_row)`. POST the same row through `TestClient`. Assert `numpy.allclose(direct, served, atol=1e-5)`. **Parametrised over all six model families** (`naive`, `linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) per the D9 reversal. The `nn_temporal` parameter is load-bearing: it asserts that the Stage 11 D5+ warmup envelope makes single-row predict work through the serving boundary identically to direct load.

- **AC-4.** "Input/output schemas are machine-readable (the HTTP framework's schema export is sufficient)."
  - Tests:
    - `test_openapi_json_contains_predict_request_and_response` — `client.get("/openapi.json")` returns 200; assert `paths["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]` is present and `paths["/predict"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]` is present.

- **AC-5.** "A smoke test exercises the endpoint with a small fixture."
  - Tests:
    - The integration test file `tests/integration/serving/test_api.py` itself satisfies AC-5 — every test above runs against a `TestClient` with a tmp-registry fixture; no live registry, no network, no special pytest marker.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_default_model_is_lowest_mae_overall` (D6 — registry contains six runs spanning all six model families; `default_run_id` resolves to whichever has the lowest MAE, *including* a `nn_temporal` run if it is the lowest-scoring; replaces the pre-Ctrl+G `*_among_eligible_runs` test).
- `test_request_log_record_carries_seven_fields` (D11 — uses `loguru_caplog`; asserts `request_id`, `model_name`, `model_run_id`, `target_dt`, `prediction`, `latency_ms`, `feature_hash` all present and well-typed).
- `test_serving_cli_help_exits_zero` (NFR-2 — `python -m bristol_ml.serving --help` exits 0; the resolved `ServingConfig` schema is in stdout).
- `test_serving_config_round_trips_through_hydra` — `conf/serving.yaml` defaults match `ServingConfig(...)` exactly.
- `test_lazy_load_caches_run_id_after_first_request` (D7 — second request with the same non-default `run_id` does not re-call `registry.load`; verified via a `mock.patch.object(registry, "load", wraps=...)` call-count assertion).
- `test_serving_logs_default_run_id_at_startup` — lifespan emits a structured `loguru` info line containing `default_run_id` and the resolved `model_name` so the operator sees what is being served.

Per-family skops round-trip tests (D10 reversal — added to `tests/unit/models/`):

- `test_naive_save_load_skops_roundtrip` — fit, save to `.skops`, load, predict; numerical equality with pre-save predictions.
- `test_linear_save_load_skops_roundtrip` — same shape; exercises the envelope-of-bytes path for `RegressionResultsWrapper`.
- `test_sarimax_save_load_skops_roundtrip` — same shape; exercises the envelope-of-bytes path for `SARIMAXResults`.
- `test_scipy_parametric_save_load_skops_roundtrip` — exercises trust-list registration of the parametric model's class hierarchy.
- `test_nn_mlp_save_load_skops_roundtrip` — exercises skops with the existing NN bytes envelope (state_dict + scalers).
- `test_nn_temporal_save_load_skops_roundtrip` — exercises skops with the warmup_features envelope; a single-row predict after load must equal a single-row predict pre-save (load-bearing for the D9 reversal).
- `test_load_rejects_unknown_artefact_extension` — passing a `.joblib` path to `registry.load` post-migration raises a clear error naming the new format.

**Total shipped tests: 19** — five intent-ACs (AC-1 two, AC-2 four, AC-3 parametrised six-way, AC-4 one, AC-5 by file existence), six D-derived (default-model, log fields, CLI help, hydra round-trip, lazy-load cache, startup log), and seven skops round-trip / migration tests.

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

`build_app(registry_dir: Path) -> FastAPI` is the public surface. The lifespan loads the default model via D6 (lowest-MAE — every family eligible after the D9 reversal) and stashes it in `app.state.loaded: dict[str, Model]` keyed by `run_id`. Subsequent non-default `run_id`s lazy-load on first use and cache.

### Cross-cutting change: skops migration

This is a Stage-12 commitment that lives outside `src/bristol_ml/serving/` but is load-bearing for the serving layer's security posture (D10 reversal). The work touches:

```
src/bristol_ml/models/
├── io.py             # gains save_skops / load_skops + project trust-list registration
├── naive.py          # save/load flip from joblib to skops (native types)
├── parametric/       # ScipyParametricModel save/load flip (native types)
│   └── ...
├── linear.py         # save/load flip via envelope-of-bytes (statsmodels.save → BytesIO → bytes → skops dict)
├── sarimax.py        # save/load flip via envelope-of-bytes (same pattern)
└── nn/
    ├── mlp.py        # save/load flip; existing bytes envelope rewrapped in skops-safe types
    └── temporal.py   # save/load flip; warmup_features envelope rewrapped in skops-safe types
src/bristol_ml/registry/
├── _io.py            # filesystem layout flips .joblib → .skops; resolution helpers updated
└── CLAUDE.md         # documents the migration; old joblib artefacts invalidated
```

The serving layer never touches the format directly: it goes through `registry.load(run_id)`, which goes through each family's `Model.load(path)`, which now reads `.skops`. The single-source-of-truth for the envelope schema lives in each family's `save` / `load`; `bristol_ml.models.io` provides only the skops primitives and the trust-list.

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
    if not runs:
        raise RuntimeError(
            f"Registry at {registry_dir} contains no runs; cannot select a default model. "
            f"Train at least one model first (`uv run python -m bristol_ml.train`)."
        )
    chosen = runs[0]
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
        # D7: lazy-load (every family goes through the same path after Ctrl+G D9 reversal)
        if run_id not in app.state.loaded:
            try:
                app.state.loaded[run_id] = registry.load(run_id, registry_dir=app.state.registry_dir)
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"run_id={run_id!r} not found under registry_dir={app.state.registry_dir}",
                ) from e
        model = app.state.loaded[run_id]
        described = registry.describe(run_id, registry_dir=app.state.registry_dir)

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

The eleven tasks split into two layers:

- **T1–T5: skops migration of all six families + registry.** Lands first because the serving layer's lazy-load path (T7+) must resolve `.skops` artefacts. Each model-family task is independent at file granularity and can run in parallel within an Agent Team if needed; the lead default is sequential to avoid registry-format churn during review.
- **T6–T10: serving layer.** Lands on top of a green skops migration.
- **T11: Phase 3 review.**

### T1 — Dependencies + skops trust-list helper.
1. Add `fastapi`, `uvicorn`, and `skops` to `pyproject.toml [project.dependencies]` (no `[standard]` extra per D2).
2. `uv lock` to refresh `uv.lock`.
3. `src/bristol_ml/models/io.py` gains `save_skops(obj, path) -> None` and `load_skops(path) -> Any` wrapping `skops.io.dump` / `skops.io.load`. The trust-list (project-level allow-list of custom classes) lives in `io.py` and is built lazily at first call from a `_PROJECT_SAFE_CLASSES` tuple. `get_untrusted_types` is invoked on every load and any unexpected type raises a clear `UntrustedTypeError` that names the type and the artefact path.
4. Existing `save_joblib` / `load_joblib` helpers remain in `io.py` with a `DeprecationWarning("Stage 12 migrated to skops; this helper will be removed in Stage 13.")` — kept for one stage so external scripts can complete a one-off migration.
5. Create `src/bristol_ml/serving/{__init__.py, __main__.py, app.py, schemas.py, CLAUDE.md}` skeletons (T6–T9 will fill them); `__init__.py` exports `build_app` (NotImplementedError stub).
6. `tests/{unit,integration}/serving/__init__.py` empty.
- **Tests:** `test_save_skops_then_load_skops_roundtrips_pure_dict` (round-trip a numpy + bytes dict envelope); `test_load_skops_raises_on_untrusted_type` (write a pickle with an out-of-list class, assert `UntrustedTypeError`); `test_save_joblib_emits_deprecation_warning`; `test_serving_module_imports_without_torch` (guard-by-construction).
- **Commit:** `Stage 12 T1: bristol_ml.models.io skops helpers + serving scaffold + deps`.

### T2 — Migrate `naive` + `scipy_parametric` to skops.
1. `NaiveModel.save` / `.load` flip to `save_skops` / `load_skops`. State is a small dict of pandas/numpy primitives — native skops, no envelope.
2. `ScipyParametricModel.save` / `.load` flip to skops; register the parametric class hierarchy in `_PROJECT_SAFE_CLASSES`.
3. File extension flips from `.joblib` to `.skops` in each `save` (default extension; explicit paths honoured as given).
- **Tests:** `test_naive_save_load_skops_roundtrip`, `test_scipy_parametric_save_load_skops_roundtrip`. Existing joblib-based tests in `tests/unit/models/` that referenced `.joblib` paths are updated to `.skops`.
- **Commit:** `Stage 12 T2: NaiveModel + ScipyParametricModel skops migration`.

### T3 — Migrate `nn_mlp` + `nn_temporal` to skops (NN families).
1. `NnMlpModel.save` / `.load` flip to skops. The existing bytes-envelope (state_dict + scalers + metadata) is repackaged as a skops-safe `dict[str, bytes | int | str | list]` — `state_dict` itself is serialised through `torch.save(BytesIO)` to bytes (already the case in Stage 10) and stored as `bytes` in the envelope; scalers are stored as their numpy arrays + class name, *not* pickled scaler instances.
2. `NnTemporalModel.save` / `.load` flip to skops with the same pattern; the `warmup_features` array is stored as a numpy array inside the envelope (already the case from Stage 11 D5+).
3. Register the small set of NN custom classes (the `Model` subclasses themselves) in `_PROJECT_SAFE_CLASSES`. The NN modules (`torch.nn.Module` subclasses) are *not* pickled — they are reconstructed at load time from the saved hyperparameters, with state_dict applied after.
- **Tests:** `test_nn_mlp_save_load_skops_roundtrip`, `test_nn_temporal_save_load_skops_roundtrip` (load-bearing for D9 reversal: single-row predict after load equals single-row predict pre-save).
- **Commit:** `Stage 12 T3: NnMlpModel + NnTemporalModel skops migration`.

### T4 — Migrate `linear` + `sarimax` to skops (envelope-of-bytes).
1. `LinearModel.save`: `results.save(BytesIO())` → bytes; envelope `{"format": "statsmodels-bytes-v1", "kind": "OLS", "blob": bytes, "feature_columns": [...], "metadata": {...}}`; `skops.io.dump(envelope, path)`.
2. `LinearModel.load`: `skops.io.load(path, trusted=...)`; assert envelope `format` and `kind`; `RegressionResults.load(BytesIO(envelope["blob"]))`; reconstruct the `LinearModel` instance.
3. `SarimaxModel.save` / `.load`: same pattern with `kind="SARIMAX"` and `SARIMAXResults.load`.
4. The envelope itself contains only skops-safe types — `dict`, `str`, `bytes`, `list[str]`, `dict[str, Any]` for metadata. The statsmodels objects never go through skops directly.
- **Tests:** `test_linear_save_load_skops_roundtrip`, `test_sarimax_save_load_skops_roundtrip`, `test_linear_envelope_format_field_validates` (hand-build a bad envelope; load raises a clear error naming the format mismatch).
- **Commit:** `Stage 12 T4: LinearModel + SarimaxModel skops migration via envelope-of-bytes`.

### T5 — Registry filesystem layout migration.
1. `bristol_ml/registry/_io.py` (or wherever the layout helper lives): the canonical artefact filename flips from `model.joblib` to `model.skops`. `registry.save` writes `.skops`; `registry.load` reads `.skops`. Sidecar JSON (run metadata) unchanged.
2. `registry.load` rejects a registry directory containing `model.joblib` with a clear error: `"Registry artefact at <path> is in the pre-Stage-12 joblib format; retrain to migrate to skops (Stage 12 D10 — joblib loads are disabled at the registry boundary for security)."`
3. `registry/CLAUDE.md` updated: skops format documented; joblib invalidation noted; the security rationale linked to the layer doc (T9).
- **Tests:** `test_load_rejects_joblib_artefact_in_registry`, `test_registry_save_writes_skops_artefact_only` (after `save`, the run directory contains `model.skops` and not `model.joblib`), `test_registry_roundtrip_per_family` (parametrised over all six families end-to-end).
- **Commit:** `Stage 12 T5: registry filesystem layout migrated to .skops`.

### T6 — `ServingConfig` + Pydantic schemas + Hydra config.
1. `src/bristol_ml/serving/schemas.py` with `PredictRequest` / `PredictResponse` per §5.
2. `conf/_schemas.py` gains `ServingConfig`; `AppConfig.serving: ServingConfig | None = None`.
3. `conf/serving.yaml` per §5.
4. `bristol_ml.config.validate` already handles new top-level fields without code change (verify via the existing test).
- **Tests:** `test_predict_request_rejects_naive_datetime`, `test_predict_request_round_trips_features_dict`, `test_serving_config_round_trips_through_hydra`, `test_app_config_serving_default_is_none_so_train_cli_unaffected`.
- **Commit:** `Stage 12 T6: ServingConfig + PredictRequest/Response schemas`.

### T7 — App factory + lifespan + `GET /` + `POST /predict`.
1. `build_app(registry_dir)` per §5; lifespan resolves D6 default and stashes in `app.state`.
2. `_select_default_run_id` picks the lowest-MAE run with no model-family filter (Ctrl+G D9 reversal).
3. `GET /` returns the default-model summary.
4. `POST /predict` per §5 — features-in, `AwareDatetime` + UTC normalise, default-or-supplied `run_id`, `model.predict(...).iloc[0]`. Lazy-load on first non-default `run_id` per D7; cache in `app.state.loaded`. Unknown `run_id` → 404 with clear `detail`.
5. Empty-registry → `RuntimeError` from the lifespan with the message in §5.
- **Tests:** `test_serving_lifespan_starts_with_only_registry_dir`, `test_serving_lifespan_raises_on_empty_registry`, `test_default_model_is_lowest_mae_overall`, `test_serving_logs_default_run_id_at_startup`, `test_predict_valid_request_returns_200_with_prediction`, `test_predict_naive_datetime_returns_422`, `test_predict_missing_required_field_returns_422`, `test_predict_unknown_run_id_returns_404`, `test_serving_prediction_parity_vs_direct_load` (parametrised over **all six families** per D9 reversal — load-bearing for `nn_temporal`), `test_lazy_load_caches_run_id_after_first_request`.
- **Commit:** `Stage 12 T7: app factory + lifespan + GET / + POST /predict (all six families)`.

### T8 — Structured logging + standalone CLI + OpenAPI smoke test (D11 + D13 + NFR-1 + NFR-2).
1. `logger.bind(...).info("served prediction")` with seven fields per D11. UUID4 request_id; sha256-truncated feature_hash; `time.perf_counter()` latency.
2. `__main__.py` argparse + uvicorn wiring per §5.
3. `--help` prints resolved `ServingConfig` schema.
- **Tests:** `test_request_log_record_carries_seven_fields`, `test_serving_cli_help_exits_zero`, `test_openapi_json_contains_predict_request_and_response`.
- **Commit:** `Stage 12 T8: seven-field structured log + standalone CLI + OpenAPI test`.

### T9 — Layer doc + module guide + README + retro skeleton.
1. `docs/architecture/layers/serving.md` (new layer) — surface (build_app, PredictRequest/Response, run_id semantics), training-serving-skew teaching point, the seven-field log contract for Stage 18, the curl example. *Includes the skops boundary note: "the serving layer is a network-facing deserialiser; the project switched the registry off joblib onto skops at this stage so an attacker cannot RCE through a malicious artefact."*
2. `src/bristol_ml/serving/CLAUDE.md` — concrete module guide; the curl one-liner; the nn_temporal warmup-envelope teaching point ("Stage 11 baked warmup_features into the artefact so single-row predict works through the boundary; this is why Stage 12 serves nn_temporal first-class without exposing warmup semantics in the request body"); the skops adoption note.
3. `docs/architecture/layers/registry.md` — update the "Stage 12 inflection point" call-out to reflect the actual decision (skops adopted, not deferred); link to the new serving layer doc.
4. `src/bristol_ml/registry/CLAUDE.md` — update the joblib note to record the migration outcome.
5. README — add a "Worked example: serving (Stage 12)" section with the curl one-liner; flag that existing `data/registry/` users must retrain for the skops migration.
6. `docs/lld/stages/12-serving.md` — retro skeleton; observed latency per model family recorded here per NFR-4; record the Ctrl+G reversal of D9 + D10 and the rationale.
7. `CHANGELOG.md` — `### Added` bullets (skops migration as a breaking change under `### Changed`).
- **Tests:** none (doc edits).
- **Commit:** `Stage 12 T9: layer doc + module guide + registry doc updates + README + retro skeleton`.

### T10 — Stage hygiene + plan move.
1. `git mv docs/plans/active/12-serving.md docs/plans/completed/12-serving.md`.
2. Final retro updates: actual observed latencies; any decisions deviated from in-stage.
3. `uv run pytest -q` clean; `uv run ruff check .` clean; `uv run ruff format --check .` clean; `uv run pre-commit run --all-files` clean.
- **Commit:** `Stage 12 T10: stage hygiene + retro + plan moved to completed/`.

### T11 — Phase 3 review.
Spawn `arch-reviewer` (conformance to plan + intent), `code-reviewer` (code quality + security — particular focus on the skops trust-list and envelope schemas), `docs-writer` (user + developer docs sweep) in parallel. Synthesise findings, address Blocking items in-branch, surface Major+Minor in the PR description.

---

## 7. Exit checklist

Before opening the PR:

- [ ] All nineteen named tests in §4 pass; full `uv run pytest -q` is clean.
- [ ] All six per-family skops round-trip tests pass (T2–T4).
- [ ] `data/registry/` contains no `model.joblib` artefacts after T5; `registry.load` rejects any pre-existing joblib artefact with the documented error.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `uv run python -m bristol_ml.serving --help` exits 0 with the `ServingConfig` schema printed.
- [ ] Layer doc `docs/architecture/layers/serving.md` exists; module guide `src/bristol_ml/serving/CLAUDE.md` exists with the curl one-liner.
- [ ] `docs/architecture/layers/registry.md` and `src/bristol_ml/registry/CLAUDE.md` updated to record the skops migration outcome.
- [ ] README has the worked-example section + the breaking-change note for existing registry users.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`: skops migration listed under `### Changed` (breaking); serving layer listed under `### Added`.
- [ ] Retro at `docs/lld/stages/12-serving.md` carries observed latencies per registered model family + the D9/D10 Ctrl+G reversal log.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/`.
- [ ] PR description surfaces:
  - The `D9` Ctrl+G reversal — `nn_temporal` is now served first-class via Stage 11's warmup-envelope; the parity test is parametrised over all six families.
  - The `D10` Ctrl+G reversal — skops adopted as canonical save/load primitive; **breaking change**: existing `data/registry/*.joblib` artefacts are invalidated, users must retrain.
  - The `H-1` DENY-tier ask — DESIGN.md §6 status flip from "deferred — Stage 12" to "shipped" needs human approval (lead is DENY-tier on intent).
  - The `D7` single-highest-leverage-cut framing — for the next stage's plan author.
  - The skops trust-list contract for downstream stages — any new model family added in a future stage must register its custom classes in `_PROJECT_SAFE_CLASSES` before its artefacts can be loaded by the serving layer.
