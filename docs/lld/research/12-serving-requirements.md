# Stage 12 — Minimal Serving Endpoint — Structured Requirements

**Source intent:** `docs/intent/12-serving.md`
**Artefact role:** Phase 1 research deliverable (requirements analyst).
**Audience:** plan author (lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## 1. Goal

Expose a single FastAPI prediction endpoint that loads a fitted model from
the Stage 9 registry and returns a forecast for a given input payload,
demonstrating concretely that training and serving are separate pipelines
with a typed boundary between them.

---

## 2. User stories

**US-1 (startup / operator — maps to AC-1).** Given a clean machine with a
populated registry directory, when the operator starts the service by
pointing at that directory, then the service comes up without requiring
any additional configuration, so that the demo moment — "start, curl,
done" — can be achieved within a minute.

**US-2 (happy-path prediction / facilitator — maps to AC-2a).** Given the
service is running and a model is registered, when the facilitator sends
a POST request with a valid input payload, then the service returns a
well-formed JSON response containing a forecast, so that attendees can
see training artefacts cross a process boundary and serve a result.

**US-3 (error handling / facilitator — maps to AC-2b).** Given the service
is running, when a request arrives with an invalid or incomplete input
payload, then the service returns a clear, structured error response
(not an unhandled server traceback), so that the typed boundary lesson
is made concrete: bad inputs are rejected at the boundary, not silently
propagated.

**US-4 (prediction parity / developer — maps to AC-3).** Given a registered
model and its training feature rows, when the developer calls `predict`
on the loaded model directly and then sends the same rows through the
serving endpoint, then the two sets of predictions are numerically
identical (or within floating-point tolerance), so that the serving
layer is demonstrably not introducing a hidden transformation step.

**US-5 (schema discoverability / facilitator — maps to AC-4).** Given the
service is running, when the facilitator navigates to the framework's
built-in schema or documentation URL, then the input and output schemas
are machine-readable and browsable without reading source code, so that
a meetup attendee can reconstruct the request format from the
documentation alone.

**US-6 (smoke test / CI — maps to AC-5).** Given the test suite runs in CI
with a small fixture payload, when the endpoint smoke test executes,
then it exercises the full request-response path (schema validation →
model load → predict → response serialisation) without requiring a live
registry or real feature data, so that the serving layer is
regression-guarded from day one.

---

## 3. Acceptance criteria

Each criterion is restated in concrete, testable form, with the test type
indicated.

**AC-1 — Zero-configuration startup (intent AC-1).**
The service must start given only a `--registry-dir` path (or the
module-level `DEFAULT_REGISTRY_DIR` constant from `bristol_ml.registry`)
and nothing else. No model-type flag, no explicit host/port in code beyond
a configurable default, no database URI. Asserted by: integration smoke
test that starts the FastAPI application with a `TestClient`, passing
only a registry directory populated with one fixture run, and confirms
`GET /` or `GET /health` returns HTTP 200.

**AC-2a — Valid request returns a prediction (intent AC-2, first clause).**
A POST to the prediction endpoint with a payload that satisfies the
declared input schema returns HTTP 200 and a response body that validates
against the declared output schema, containing at least one numeric
forecast value. Asserted by: integration test using `TestClient` with a
fixture payload constructed from the registry smoke-fixture.

**AC-2b — Invalid request returns a structured error (intent AC-2, second
clause).** A POST with a missing required field, a field of the wrong
type, or an empty body returns HTTP 422 (FastAPI's default validation
error status) with a JSON response body that identifies the offending
field(s). Asserted by: unit test sending a deliberately malformed payload
via `TestClient` and asserting status code + at least one `loc`/`msg`
pair in the response body (FastAPI's `RequestValidationError` shape).

**AC-3 — Prediction parity (intent AC-3).**
For any registered model `M` and any feature row `X` within that model's
`feature_columns` list, `serving_endpoint(X)` must equal `M.predict(X)`
within floating-point tolerance (`atol=1e-5`, consistent with the
registry's existing MLflow adapter test). Asserted by: integration test
that (a) loads a fixture model directly via `registry.load`, (b) sends
the same feature row through `TestClient`, and (c) asserts
`numpy.allclose(direct, served, atol=1e-5)`. The test must cover at
least one model family; covering `NnMlpModel` is sufficient given the
complexity of its normalisation path.

**Ambiguity flag on AC-3.** The intent states "identical predictions." If
the serving layer assembles features from raw inputs (see OQ-2 below),
identity is impossible to guarantee without re-running the assembler — a
non-trivial dependency. The intent's own "points for consideration"
section identifies this as the training-serving skew problem. The plan
author must decide whether AC-3 applies to the fully assembled-feature
path or only to the already-assembled-feature path. This requirements
document cannot resolve that choice; it surfaces it as OQ-2.

**AC-4 — Machine-readable schemas (intent AC-4).**
The HTTP framework's schema export (FastAPI's `/openapi.json`) must be
reachable at a documented URL when the service is running. The exported
schema must contain the prediction endpoint's request body schema and
response schema as JSON Schema objects. Asserted by: integration test
that GETs `/openapi.json` and asserts the prediction endpoint's path is
present with `requestBody` and `responses` components.

**AC-5 — Smoke test in CI (intent AC-5).**
A pytest test exercises the endpoint end-to-end with a small fixture
payload without a live registry on disk. The fixture must be
self-contained (no network, no file system beyond a `tmp_path`) and fast
enough to run in CI without a special marker. Asserted by: existence of
`tests/integration/serving/test_api.py` (or equivalent) with at least
one test that imports `TestClient`, starts the app, sends a POST, and
asserts HTTP 200 — the test itself is the criterion.

---

## 4. Non-functional requirements

The following are extracted from the "Points for consideration" section of
the intent. Each is labelled either **Binding NFR** (the intent implies
it must be met for the stage to ship) or **Discuss in plan** (the intent
raises it as a concern to resolve during planning, not a hard gate).

**NFR-1 — Single-request latency.**
Intent: "A single-request round trip of under a second on a laptop is
plausible for all models in the roster; if any model is much slower,
that's a talking point."
Classification: **Discuss in plan.** The intent explicitly says latency
is "not a goal here" and frames outliers as pedagogical talking points
rather than failures. The plan author should decide whether to assert a
latency bound in CI or simply document observed latency in the retro.
Note: `NnTemporalModel` with `seq_len=168` requires a 168-hour input
window — a cold-start serving request cannot be assembled without state.
This may make sub-second latency dependent on whether raw-input or
assembled-feature serving is chosen (see OQ-2).

**NFR-2 — Schema discoverability.**
Intent: "Input and output schemas are machine-readable (the HTTP
framework's schema export is sufficient)."
Classification: **Binding NFR.** This is the intent's own words
describing AC-4. FastAPI's automatic `/openapi.json` is the nominated
mechanism; no additional work is specified. The plan author should
confirm whether the Pydantic request/response models alone are
sufficient or whether additional field-level descriptions are required
for the demo.

**NFR-3 — Training-serving skew prevention.**
Intent: "If the service and the training pipeline compute features
differently, predictions will drift silently. This is the canonical
MLOps failure; pointing at it at a meetup is itself educational."
Classification: **Discuss in plan.** The intent frames skew prevention
as both a technical concern and a pedagogical talking point. It does
not mandate a specific mechanism (e.g. shared assembler code path,
feature hash in sidecar). The plan author must decide whether the
serving layer re-uses `bristol_ml.features.assembler` (safe, but
introduces a features dependency and raises OQ-2), or accepts
already-assembled feature rows (simple, but makes skew a caller
concern). The choice must be made explicit in the plan; the stage's
educational value partly depends on naming the trade-off out loud.

**NFR-4 — Request and prediction logging.**
Intent: "Minimal logging is easy to add and useful for Stage 18's drift
monitoring, which needs production predictions to analyse."
Classification: **Discuss in plan.** The intent does not mandate a log
format, log destination, or log schema for Stage 12. However, it
explicitly names Stage 18 (drift monitoring) as a downstream consumer.
The plan author should decide whether Stage 12 logs predictions to a
structured file (e.g. JSONL in `data/predictions/`) or only to stdout
via `loguru`. If Stage 18's design requires a specific log schema,
deferring that decision to Stage 18 risks a retrofit; surfacing it now
is preferable. This is escalated as OQ-5 below.

**NFR-5 — Model selection strategy.**
Intent: "Which model to serve by default. The best model from the
registry is one answer; a named configuration is another."
Classification: **Discuss in plan.** The intent offers two equally
defensible answers without nominating one. The serving strategy
(best-by-metric vs named run ID vs named model type) has direct
implications for AC-1 (zero-configuration startup). See OQ-1.

**NFR-6 — Deployment neutrality.**
Intent: "The structure of the service — pure Python, registry-driven,
typed boundaries — should not preclude a future 'actually deploy this'
stage."
Classification: **Discuss in plan.** Not a Stage 12 requirement, but
the plan author should ensure no in-scope decision (e.g. hardcoding
`localhost` URLs, injecting mutable global state) accidentally
forecloses Stage 19's orchestration integration.

**NFR-7 — Standalone module operation (§2.1.1).**
The DESIGN.md §2.1.1 principle states every module must run standalone
via `python -m bristol_ml.<module>`. The serving layer must therefore
support `python -m bristol_ml.serving` as a startup path, not only
programmatic instantiation inside a notebook.
Classification: **Binding NFR** (project-wide architectural principle,
not negotiable at this stage).

**NFR-8 — Serialisation security.**
The registry layer doc (`docs/architecture/layers/registry.md`)
explicitly defers `skops.io` adoption to Stage 12, noting that joblib
is not a safe deserialiser for untrusted inputs and that "the
untrusted-input path begins at Stage 12 (serving)." The plan author
must decide whether to adopt `skops.io` at this stage or to document
the known limitation explicitly and defer it.
Classification: **Discuss in plan** — the registry flags it as a
Stage 12 inflection point, but the intent document does not mandate
it. The plan author must make the decision visible.

---

## 5. Out of scope

The following are restated verbatim from the intent's "Out of scope" and
"Out of scope, explicitly deferred" sections, with brief reasoning for
the plan author's scope-diff checklist.

| Item | Source | Reasoning |
|------|--------|-----------|
| Authentication, authorisation, rate limiting | Intent §Scope (out of scope) + §Out of scope, explicitly deferred | Adds surface complexity; the demo is localhost-to-localhost; no user identity model exists in this architecture. |
| Multi-user concurrent request handling at scale | Intent §Scope (out of scope) | Single-author pace; FastAPI's async internals are sufficient for one facilitator's curl loop. |
| Deployment anywhere other than localhost | Intent §Scope (out of scope) + §Out of scope, explicitly deferred | Out of scope for the stage; the structure should not preclude it (NFR-6), but no deployment artefact (Dockerfile target, cloud config) is required. |
| A UI beyond the HTTP framework's auto-docs | Intent §Scope (out of scope) | FastAPI's `/docs` (Swagger UI) is the specified boundary; no custom front end. |
| Batch prediction endpoints | Intent §Scope (out of scope) + §Out of scope, explicitly deferred | The registry already supports batch scoring; serving is for demonstrating the process boundary, not replicating training-time batch inference. |
| Model hot-reload | Intent §Scope (out of scope) | Serving loads on startup; in-process model swapping without restart is not required. |
| HTTPS | Intent §Out of scope, explicitly deferred | Requires certificate management; irrelevant at localhost. |
| Model versioning semantics beyond "ask for a model by name" | Intent §Out of scope, explicitly deferred | Stage 9's `list_runs` + `describe` are sufficient; promotion/staging semantics are not in scope. |

---

## 6. Open questions

The following decisions are explicitly left to the plan author. This
section enumerates the trade-offs; it does not resolve them.

**OQ-1 — Default model selection strategy.**
The intent offers two options without choosing: serve the "best" model
from the registry (best by some metric, presumably MAE mean from
`list_runs`) or serve a named model (by `run_id` or by model type). The
trade-offs are:
- "Best by metric" requires the service to interpret the registry's
  leaderboard at startup and may resolve to different models across
  runs, making the demo non-deterministic if new runs were registered
  since the service last started.
- "Named run ID" is deterministic and explicit, but requires the
  operator to know a valid `run_id` before starting the service —
  which contradicts AC-1's "no configuration beyond registry location."
- "Named model type" (e.g. `model_type=linear`) is a middle ground:
  deterministic within a model family, but requires a default type to
  be chosen or configured.
- The choice also affects whether a single endpoint serves all model
  types or whether different model types are routed differently (see
  OQ-4).

**OQ-2 — Raw inputs vs assembled features at the request boundary.**
The intent explicitly flags this as a "real question" without answering
it. The trade-offs are:
- Accepting raw inputs (timestamps, temperature values) and assembling
  features inside the service is closer to real-world use and avoids
  exposing the training feature table schema to callers. However, it
  introduces a dependency on `bristol_ml.features.assembler` inside the
  serving layer, which in turn requires the calendar feature logic and
  weather inputs to be available at request time. It also makes AC-3
  (prediction parity) contingent on the assembler producing identical
  results to the training path — the canonical skew surface.
- Accepting already-assembled feature rows (the full feature vector in
  the format the model's `feature_columns` field specifies) is simpler
  and allows strict AC-3 verification, but the caller must know the
  feature schema. For a pedagogical service this may be acceptable;
  for a real deployment it would not be.
- The choice also interacts with `NnTemporalModel`'s `seq_len=168`
  window requirement: if raw inputs are accepted, the service must
  hold 168 hours of state (or require the caller to supply them),
  which is non-trivial. If assembled features are accepted, the caller
  must supply a `(168, n_features)` array.

**OQ-3 — Single endpoint with a model-name parameter vs multiple
endpoints.** The intent names both options: a single `/predict` endpoint
that accepts a `run_id` or `model_type` query parameter, vs separate
endpoints per model family (e.g. `/predict/linear`, `/predict/nn_mlp`).
The trade-offs are:
- A single endpoint with a parameter is more flexible and matches how
  `registry.load(run_id)` works. It requires the request schema to be
  general enough for any model's feature columns, which may produce a
  less legible schema in `/openapi.json`.
- Separate endpoints allow per-model typed request schemas, making the
  auto-docs more useful for the demo. However, adding a new model
  family requires a code change in the serving layer, which violates
  the spirit of the registry-driven design.
- The plan author must also decide whether model selection at request
  time (caller passes `run_id` in the request) is the expected
  pattern, or whether model selection is baked into startup
  configuration.

**OQ-4 — NnTemporalModel serving semantics.**
Stage 11's `NnTemporalModel` requires a `seq_len`-length input window
(default 168 hours) at prediction time. The intent does not acknowledge
this model family directly. The trade-offs are:
- If the endpoint accepts assembled feature rows, callers must supply
  a `(seq_len, n_features)` shaped input for temporal models, but a
  `(1, n_features)` shaped input for non-temporal models. This creates
  two distinct request schemas for the same endpoint.
- If the endpoint only supports flat-feature models (NaiveModel,
  LinearModel, SarimaxModel, ScipyParametricModel, NnMlpModel) and
  explicitly excludes `NnTemporalModel`, the serving layer is simpler
  but less complete.
- The plan author should decide whether Stage 12's scope includes
  serving temporal models at all, or whether that is deferred to a
  subsequent stage when the input-state management question can be
  properly addressed.

**OQ-5 — Prediction log schema for Stage 18 compatibility.**
The intent says minimal logging is "useful for Stage 18's drift
monitoring, which needs production predictions to analyse." Stage 18
does not yet exist, so the log schema cannot be validated against a
real consumer. The trade-offs are:
- Logging request inputs and predictions as JSONL now, with a
  provisional schema, means Stage 18 has real data to work with and
  the log format is established. The risk is that Stage 18 needs a
  different schema and a migration is required.
- Logging only to stdout now and deferring the structured log to
  Stage 18 means Stage 18 must add its own logging at the serving
  layer — a retrofit of a shipped module.
- The plan author should at minimum decide whether the Stage 12 log
  is a flat `{timestamp, run_id, inputs, prediction, latency_ms}`
  JSONL record or something richer, and whether it lands in a file
  (durable) or stdout (ephemeral).

**OQ-6 — skops.io adoption.**
The registry layer doc (`docs/architecture/layers/registry.md`)
explicitly flags Stage 12 as "the correct inflection point for
`skops.io` adoption." The intent document does not mention this. The
trade-offs are:
- Adopting `skops.io` at Stage 12 adds a new dependency and requires
  each model family's `save`/`load` methods to be updated. It is the
  architecturally correct moment because the serving layer is the
  first consumer that would receive model artefacts from an untrusted
  source in a real deployment.
- Not adopting `skops.io` keeps Stage 12 in scope and defers the
  security concern with a documented note. The service runs on
  localhost against artefacts the author trained themselves, so the
  practical risk is low for this stage.
- The plan author must make the decision explicit rather than silent;
  the registry layer doc's explicit flag means this cannot be treated
  as already resolved.

---

**Relevant source files:**
- `/workspace/docs/intent/12-serving.md`
- `/workspace/docs/intent/DESIGN.md`
- `/workspace/docs/architecture/layers/registry.md`
- `/workspace/src/bristol_ml/registry/CLAUDE.md`
- `/workspace/src/bristol_ml/models/CLAUDE.md`
- `/workspace/src/bristol_ml/models/nn/CLAUDE.md`
