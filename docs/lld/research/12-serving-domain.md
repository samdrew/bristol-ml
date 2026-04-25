# Stage 12 — Minimal serving endpoint: domain research

**Date:** 2026-04-25
**Target plan:** `docs/plans/active/12-serving.md` (not yet created)
**Intent source:** `docs/intent/12-serving.md`
**Baseline SHA:** main @ `6ad2d7a` (post Stage 10 + DESIGN.md plan refs)

**Scope:** External literature and primary tool documentation the plan
author needs before making framework, schema, and architecture decisions
for Stage 12. Numbered sections (R1–R8) so the plan can cite by
reference. British English throughout.

---

## R1 — HTTP framework choice for minimal Python ML serving in 2026

### Canonical sources

| Source | Summary |
|--------|---------|
| [FastAPI — fastapi.tiangolo.com](https://fastapi.tiangolo.com/) | Official docs; v0.136.1 as of April 2026; covers minimal app setup, lifespan events, OpenAPI generation |
| [FastAPI PyPI — pypi.org/project/fastapi](https://pypi.org/project/fastapi/) | Version history, dependency list; core required deps are Starlette + Pydantic only |
| [Litestar vs FastAPI — betterstack.com](https://betterstack.com/community/guides/scaling-python/litestar-vs-fastapi/) | Side-by-side on performance, ergonomics, serialisation, ecosystem size |
| [Best Python API Framework 2026 — uvik.net](https://uvik.net/blog/python-api-framework/) | Market survey covering FastAPI, Flask, Litestar, DRF for ML and general use |
| [FastAPI vs Flask for production AI APIs — clickittech.com](https://www.clickittech.com/ai/fastapi-vs-flask-for-production-ai-apis/) | ML-specific comparison; async handling, validation, performance notes |
| [Python API frameworks for model serving — dev.to](https://dev.to/ahmedrauhan/python-api-frameworks-compared-whats-best-for-your-model-serving-or-backend-1kpo) | Practical comparison of framework choices specifically for ML backends |
| [Litestar docs — docs.litestar.dev](https://docs.litestar.dev/main/) | Official Litestar documentation; OpenAPI 3.1, Pydantic v2 plugin, msgspec integration |
| [BentoML vs FastAPI — towardsai.net](https://towardsai.net/p/l/bentoml-vs-fastapi-the-best-ml-model-deployment-framework-and-why-its-bentoml-2) | Argues BentoML's batching advantage; concedes FastAPI is correct for prototyping and simple cases |

### Framework-by-framework summary

**FastAPI (v0.136.1, released April 2026)**

`DESIGN.md` §8 already names FastAPI as the serving technology. Its two
_required_ dependencies are Starlette (ASGI routing) and Pydantic
(validation and schema). Since the project already requires Pydantic v2
for its configuration schemas, FastAPI adds essentially zero marginal
schema-layer cost to the dependency tree.

The `fastapi[standard]` extra pulls in `uvicorn[standard]` (ASGI
server), `httpx` (required for `TestClient`), `jinja2`,
`python-multipart`, and `fastapi-cli`. The minimal install — `fastapi`
plus `uvicorn` without the standard extras — is a handful of packages.
For the dev environment managed by `uv`, `fastapi[standard]` is the
natural choice; it makes `TestClient` available without a separate
`httpx` entry in `pyproject.toml`.

A minimum viable `POST /predict` endpoint is eight to twelve lines: one
`BaseModel` subclass for the request, one for the response, one
`@app.post` function returning the response model. OpenAPI 3.0 JSON is
served automatically at `GET /openapi.json` — generated lazily from the
Pydantic models on first access. This directly satisfies AC-4
(machine-readable schemas) with no extra implementation.
`fastapi.testclient.TestClient` provides an in-process HTTP client that
satisfies AC-5 (smoke test) in a single import.

FastAPI is async-first but accepts synchronous (`def`) handlers
transparently — they run in a thread-pool executor. For a single-model-
serving endpoint with CPU inference there is no I/O to parallelise; a
synchronous handler is the correct choice and is more legible for a
pedagogical codebase. Opting into `async def` here without genuine
awaitable work would mislead readers about why async is used.

**Flask (3.1.x)**

Flask 3.0 (September 2023) added async view support, but it is not
first-class. More importantly, Flask has no built-in request body
validation or schema export; achieving parity with FastAPI requires
adding `flask-pydantic` or `marshmallow`, which fragments the dependency
story against a project already on Pydantic v2. Flask's install size
(156 MB in one comparison) is larger than FastAPI's (~127 MB) because
the Flask ecosystem requires more extensions to reach the same feature
set. Flask is a legitimate fallback for teams that want the absolute
minimum framework surface and are willing to hand-roll validation, but
for this project it would add complexity rather than remove it.

**Litestar (2.x)**

Litestar generates OpenAPI 3.1 (one version ahead of FastAPI's 3.0),
supports Pydantic v1 and v2 simultaneously (including mixed within a
single application), and uses `msgspec` for serialisation internally —
claimed to be approximately 12x faster than Pydantic v2 for the
serialisation step alone. For a production high-throughput service this
gap is meaningful; for a single-request pedagogical demo the difference
is unmeasurable. The ecosystem is smaller than FastAPI's, the team
would need to learn a second framework, and Stack Overflow / tutorial
coverage is substantially thinner. For a learning-artefact codebase,
ecosystem familiarity and searchability matter more than serialisation
throughput.

**BentoML / Ray Serve / TorchServe / MLflow Models / Seldon**

All these frameworks target the production end of the spectrum: request
batching, multi-worker replica management, model versioning services,
Kubernetes deployment. `pip install bentoml` pulls in a large transitive
closure. MLflow serving is noted in the literature as "not really doing
anything extra beyond a basic setup" while carrying the full MLflow
metadata-store weight. TorchServe requires a JVM-based model archive
(`.mar`) incompatible with the project's joblib/pickle registry format.
These are the wrong tools for a one-stage laptop demo and should not be
introduced at Stage 12.

### Recommendation

FastAPI is the correct default. The decision is already recorded in
`DESIGN.md` §8; the research confirms it. Viable alternative: Flask, if
minimising transitive dependencies is a hard constraint and the team
accepts hand-rolling validation. Not recommended: Litestar (marginal
gain vs ecosystem risk for this scope), BentoML / Ray Serve / TorchServe
/ MLflow serving (all vastly over-scoped).

---

## R2 — Pydantic v2 and the HTTP boundary

### Canonical sources

| Source | Summary |
|--------|---------|
| [FastAPI handling errors — fastapi.tiangolo.com](https://fastapi.tiangolo.com/tutorial/handling-errors/) | How 422 responses are generated from `RequestValidationError`; custom exception handlers |
| [Pydantic validation errors — docs.pydantic.dev](https://docs.pydantic.dev/latest/errors/validation_errors/) | Structure of `ValidationError.errors()`: `loc`, `msg`, `type` fields |
| [FastAPI extending OpenAPI — fastapi.tiangolo.com](https://fastapi.tiangolo.com/advanced/extending-openapi/) | Exporting or customising the `/openapi.json` endpoint |
| [Pydantic serialization — docs.pydantic.dev](https://docs.pydantic.dev/latest/concepts/serialization/) | `model_dump_json`, `model_dump`, datetime serialisation format details |
| [Pydantic datetime types — docs.pydantic.dev (v2.0)](https://docs.pydantic.dev/2.0/usage/types/datetime/) | `AwareDatetime` enforces timezone-aware input; `NaiveDatetime` enforces naive |
| [Pydantic timezone silent-drop issue #9571 — github.com/pydantic](https://github.com/pydantic/pydantic/issues/9571) | Known: `datetime.time` objects with DST-dependent tzinfo can silently lose tz on serialisation |
| [Pydantic TzInfo vs datetime.UTC issue #8683 — github.com/pydantic](https://github.com/pydantic/pydantic/issues/8683) | Pydantic's internal `TzInfo(UTC)` does not compare equal to `datetime.UTC` in some code paths |
| [Pydantic v2 pandas tzinfo mismatch #6592 — github.com/pydantic](https://github.com/pydantic/pydantic/issues/6592) | Timezone handling differs from v1; tzinfo type mismatch between Pydantic and pandas can cause dtype loss on DataFrame concat |
| [FastAPI extra data types — fastapi.tiangolo.com](https://fastapi.tiangolo.com/tutorial/extra-data-types/) | How FastAPI serialises `datetime` to/from JSON in request/response bodies |
| [Python json module — docs.python.org](https://docs.python.org/3/library/json.html) | Standard library `json` preserves full IEEE 754 float precision by default |
| [ujson float truncation — craigrosie.github.io](https://craigrosie.github.io/posts/beware-ujson-dumps-precision/) | `ujson` silently truncates floats beyond a certain precision; a known pitfall |

### Validation errors → 422

FastAPI catches `RequestValidationError` (raised by Pydantic on request
body failure) and returns HTTP 422 Unprocessable Entity automatically
with a structured JSON body. Each error entry includes `loc` (path into
the body, e.g. `["body", "target_dt"]`), `msg`, and `type`. No handler
code is required for the default behaviour. A custom `exception_handler`
can reshape the error body if the demo audience benefits from a simpler
format, but the default is already human-readable and machine-parseable.

### Generating `openapi.json`

FastAPI builds OpenAPI 3.0 JSON lazily from the Pydantic models attached
to each route. The schema is served at `GET /openapi.json` with no extra
configuration. All standard Pydantic field types — `datetime`, `float`,
`list[float]`, `str`, `int` — map to JSON Schema types automatically.
The spec can be exported offline via `app.openapi()` (returns a dict)
or CLI tools such as `fastapi-export-openapi`. This satisfies AC-4 with
zero additional implementation.

Known issue: `PydanticInvalidForJsonSchema` errors are raised when a
Pydantic model uses `PlainValidatorFunction` validators that cannot be
introspected for JSON Schema (pydantic/pydantic#8328). The mitigation
is to use standard Pydantic field types (no plain-validator lambdas) in
the request and response models. The internal `Model` protocol objects
must not be used directly as FastAPI response models.

### Round-tripping `pd.Timestamp` (UTC) through JSON

The model feature index is UTC-aware. The serving boundary must
represent a point in time in JSON without losing timezone information.

Safe path: type the field as `AwareDatetime` from `pydantic`.
`AwareDatetime` rejects naive datetimes at validation time, producing a
clear 422. When serialised to JSON by Pydantic v2, UTC datetimes are
emitted as ISO 8601 strings — the specific format (whether `Z` or
`+00:00` suffix) changed between Pydantic v2 minor versions and both
are valid RFC 3339. Both are correctly parsed by `pd.Timestamp(value)`.

Important pitfall: Pydantic v2's internal `TzInfo(UTC)` does not
compare equal to `datetime.timezone.utc` or `datetime.UTC` in all code
paths (pydantic#8683). If any code in the feature assembler checks
`isinstance(ts.tzinfo, datetime.timezone)` or
`ts.tzinfo == datetime.UTC`, it may misfire on a timestamp that passed
through Pydantic. The safe practice is to normalise explicitly —
`value.astimezone(datetime.timezone.utc)` — at the serving boundary
before handing the datetime to the feature pipeline, rather than
relying on identity comparisons of tzinfo objects.

A second pitfall: when a Pydantic-validated `datetime` is inserted into
a pandas `DataFrame`, the tzinfo type carried by the object may differ
from what pandas expects (`pytz.UTC`, `ZoneInfo("UTC")`, or
`datetime.timezone.utc`), causing dtype inconsistencies on `pd.concat`
(pydantic#6592). Normalise with `pd.Timestamp(value, tz="UTC")` when
constructing the feature row.

Naive datetime in the request body: if a client sends
`"2025-06-15T13:00:00"` (no `Z` or offset), `AwareDatetime` will reject
it with a 422. This is correct boundary behaviour and should be noted
in the endpoint description.

### Numeric precision

Python's standard `json` module serialises IEEE 754 doubles without
truncation. `json.dumps(1.3959197885670265)` produces
`'1.3959197885670265'`. Pydantic v2's `model_dump_json()` uses the
same `rust`-based serialiser and preserves full precision. For GB
electricity demand (expected values in the range 20 000–55 000 MW),
`float` is more than adequate and there is no practical
trailing-decimal noise problem. `ujson`, sometimes used as a FastAPI
serialisation accelerator, silently truncates floats beyond a precision
threshold and must not be used for this application.

---

## R3 — The ML serving spectrum: where Stage 12 sits

### Canonical sources

| Source | Summary |
|--------|---------|
| [Sculley et al. 2015 — NeurIPS](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) | "Hidden Technical Debt in Machine Learning Systems"; pipeline jungles, undeclared consumers, entanglement |
| [Polyzotis et al. 2018 — ACM SIGMOD Record](https://dl.acm.org/doi/10.1145/3299887.3299891) | "Data Lifecycle Challenges in Production Machine Learning"; training-serving skew as a first-class concern |
| [Sato, Wider, Windheuser 2019 — martinfowler.com](https://martinfowler.com/articles/cd4ml.html) | "Continuous Delivery for Machine Learning" (CD4ML); training and serving as separate pipelines |
| [BentoML blog — bentoml.com](https://www.bentoml.com/blog/breaking-up-with-flask-amp-fastapi-why-ml-serving-requires-a-specialized-framework) | Argues for specialised runtimes when batching and concurrency matter; concedes FastAPI for prototyping |
| [KServe data plane — kserve.github.io](https://kserve.github.io/website/docs/concepts/architecture/data-plane) | V2 open inference protocol; transformer-predictor conceptual split |

### The spectrum

ML serving frameworks span a well-understood spectrum:

1. **Raw model + Flask/FastAPI** — one Python process, model in memory,
   synchronous predict, JSON in / JSON out. Zero infrastructure.
   Sculley et al. 2015 warn about "pipeline jungles" where serving
   logic diverges from training logic; the risk at this level is
   training-serving skew (§R4), not infrastructure complexity. This is
   where Stage 12 sits.

2. **BentoML / MLflow Models serving** — adds model packaging,
   multi-worker, micro-batching, and opinionated serialisation.
   BentoML demonstrates roughly 10x–30x throughput improvement over
   naive FastAPI for CPU batch workloads through request batching. At
   Stage 12 scope (single request, demo), this gain is irrelevant; the
   installation weight is unjustified.

3. **Ray Serve** — distributed serving on the Ray distributed runtime;
   appropriate for GPU-bound parallel inference. Out of scope.

4. **TorchServe** — requires a JVM-based `.mar` model archive,
   incompatible with the project's joblib-based registry. Out of
   scope.

5. **KServe / Seldon** — Kubernetes-native. Out of scope.

### What Stage 12 can borrow from KServe's transformer-predictor pattern

KServe separates each inference service into a "transformer" (raw
request → features) and a "predictor" (features → model output).
Stage 12 does not need to be two separate services, but naming the
conceptual boundary in code — a `prepare_features()` function distinct
from the `model.predict()` call, even in the same file — keeps the
architecture honest, makes the training-serving skew risk visible to a
meetup audience, and makes the code easier to trace.

### What is honestly out of scope at Stage 12

Request batching, multi-worker Uvicorn processes, GPU inference, model
hot-reload (explicitly excluded in the intent), health-check or
readiness endpoints, authentication (explicitly excluded in the
intent).

---

## R4 — Training-serving skew: the canonical MLOps failure

### Canonical sources

| Source | Summary |
|--------|---------|
| [Sculley et al. 2015 — NeurIPS](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) | §3 "Entanglement" and "Undeclared Consumers": structural causes of divergence between training and serving |
| [Polyzotis et al. 2018 — SIGMOD Record PDF](https://sigmodrecord.org/publications/sigmodRecord/1806/pdfs/04_Surveys_Polyzotis.pdf) | Formalises skew detection as a data-validation concern; draws on Google's TFX/TFDV experience |
| [Google Cloud training-serving skew blog — cloud.google.com](https://cloud.google.com/blog/topics/developers-practitioners/monitor-models-training-serving-skew-vertex-ai) | Concrete example: Google Play discovered always-missing features in logs vs training data using TFDV |
| [Ploomber training-serving skew — ploomber.io](https://ploomber.io/blog/train-serve-skew/) | Practitioner summary; common causes (null handling, timezone conversions, encoding mismatches) |
| [System Overflow training-serving skew — systemoverflow.com](https://www.systemoverflow.com/learn/ml-infrastructure-mlops/ci-cd-ml/training-serving-skew-and-environment-parity) | Classifies causes; emphasises "different implementations" as the primary root cause |

### What skew means in this context

Training-serving skew is distinct from statistical drift (where the
input distribution genuinely changes over time). Skew is a _bug_: the
model receives features at serving time that are systematically
different from what it saw at training time, even when the underlying
reality is unchanged. It is silent — the model produces predictions,
just wrong ones. Polyzotis et al. 2018 identify it as one of the most
costly failure modes in production ML.

In the bristol_ml context, the canonical failure path is: the serving
endpoint accepts a timestamp and weather values, assembles features
using a local reimplementation of the calendar/lag logic, and that
reimplementation drifts from `features/assembler.py` as either module
evolves independently. The model then silently produces degraded
predictions with no error signal.

### Three mitigations and their costs

**Mitigation 1: Single feature-pipeline source of truth**

Call `features/assembler.py` (or the relevant sub-function) from the
serving endpoint, rather than duplicating feature logic. The serving
handler imports and calls the same function the training code calls.
This is the standard recommendation in the literature and the primary
mitigation.

Cost: zero new lines. It is a matter of import structure. The serving
module now depends on `features/`, which in turn depends on `pandas`,
`numpy`, etc. These are already in the project's dependency tree.
This mitigation does not add dependencies.

**Mitigation 2: Feature-frame schema assertion at the serving
boundary**

Validate the assembled feature `DataFrame` against the column set the
model was trained on (column names, dtypes, shape) before calling
`model.predict`. This is a runtime check that catches schema
mismatches introduced by pipeline changes after a model was serialised
to the registry.

Cost: 10–20 lines of inline validation, or a `pandera` schema (adds a
dependency not currently in the tree). An inline
`assert set(df.columns) == expected_columns` is sufficient for a demo
and adds no dependencies.

Latency cost: negligible (single-row frame, set comparison).

**Mitigation 3: Log request features alongside the prediction**

Write the feature vector (or a hash of it), model name, and prediction
to the log on every request. This enables Stage 18 drift analysis
without any online monitoring infrastructure at Stage 12 time.

Cost: 5–10 lines using `loguru.logger.bind(...)`. Performance cost is
a single log write per request: unmeasurable at demo scale.

### Which mitigations are realistic for a single-stage demo

Mitigation 1 is not optional for a codebase whose explicit pedagogical
purpose includes demonstrating the training-serving skew failure. The
intent §"Points for consideration" says "pointing at it at a meetup is
itself educational." Reimplementing feature logic in the serving
endpoint would undercut that lesson.

Mitigation 2 adds a meaningful invariant with negligible cost.
Recommended as a lightweight column-set assertion rather than a full
schema library.

Mitigation 3 is low-cost and directly enables Stage 18. Omitting it
would require reopening Stage 12's code later. Recommended.

---

## R5 — Request/response schema design for a forecaster

### Canonical sources

| Source | Summary |
|--------|---------|
| [KServe data plane — kserve.github.io](https://kserve.github.io/website/docs/concepts/architecture/data-plane) | V2 open inference protocol; input tensors vs pre/post-processing |
| [KServe transformer pattern — medium.com](https://medium.com/@nsalexamy/decoupling-ml-inference-from-client-applications-with-kserve-transformers-faa68dd0d04c) | `preprocess()` (raw inputs → features) + predictor (`features → output`); hooks in transformer component |
| [Feast feature store / model inference — docs.feast.dev](https://docs.feast.dev/getting-started/architecture/model-inference) | "Features-in" pattern where the serving layer pulls pre-computed features from a store |
| [DESIGN.md §7.3 — this repo](docs/intent/DESIGN.md) | `Model` protocol: `predict(features: pd.DataFrame) -> pd.Series` |

### Design A — "Features-in"

The request body carries an already-assembled feature row matching the
training DataFrame's column schema. The server calls
`model.predict(pd.DataFrame([request.features]))` and returns the
scalar result.

Advantages: server code is minimal; AC-3 ("predictions identical to
those produced by loading and running the model directly") is
trivially testable; the schema directly mirrors what the model
expects.

Disadvantages: the client must know the exact feature schema — every
column name, encoding decision, and lag window. This leaks the
training abstraction across the process boundary. If the assembler
changes its column naming or adds a feature, all clients must change.
In a real-world setting, callers have raw inputs (timestamps, meter
readings, weather forecasts), not pre-assembled feature matrices.

Where it sits in literature: this is equivalent to KServe's predictor
component after the transformer has already run; it is "half" of the
standard pattern.

### Design B — "Inputs-in"

The request body carries raw inputs (target datetime, weather
variables such as temperature and wind speed). The server runs the
feature assembler, then calls `model.predict`.

Advantages: the API is natural for a caller that has weather forecast
data; training-serving skew is architecturally mitigated because the
assembler is the single source of truth; the serving code is
structurally closer to how real forecasting APIs work.

Disadvantages: the server is now responsible for feature assembly; it
couples the serving process to the `features` module. The request
schema is weather-domain-specific. For a model roster with different
feature sets (linear regression vs TCN vs SARIMAX), the inputs-in
schema either must be a superset across all models (some fields
unused depending on model) or must be model-specific (different
schemas per endpoint).

Where it sits in literature: this is equivalent to KServe's
transformer component, which sits upstream of the predictor.

### What is defensible for a pedagogical demo

The intent §"Points for consideration" explicitly surfaces both as
"defensible". Considerations the plan author should weigh:

The demo moment is `curl localhost:8000/predict` returning a forecast.
A "features-in" request body with 20+ feature columns is difficult to
demonstrate live. An "inputs-in" request body with timestamp + a few
weather scalars is more legible to a meetup audience and tells a
cleaner story.

However, assembling features at serving time with Design B requires
weather data available at the serving boundary. If the service fetches
live weather forecasts, it introduces a network dependency. If it
accepts weather values in the request body, the demo script must
supply them. If it loads them from a cached parquet file, the demo
requires that file to exist.

A lightweight hybrid is possible: the request carries the target
datetime plus optional weather override values; if weather values are
omitted, the service loads the nearest row from the most recent
assembled feature table in `data/`. This is pragmatic for a laptop
demo but adds complexity not clearly justified by Stage 12's
acceptance criteria.

The research surfaces no consensus; both designs appear in production
systems. For Stage 12's explicit learning objective of making
training-serving skew visible, Design B is more architecturally honest
but requires more implementation. The plan author must decide.

---

## R6 — State-bearing models and the warmup problem

### Canonical sources

| Source | Summary |
|--------|---------|
| [NVIDIA Triton stateful sequence batching — docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) | Correlation ID routing; all requests in a sequence routed to the same model instance; CONTROL tensors for sequence start/end |
| [NVIDIA Triton models and schedulers — docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1150/user-guide/docs/models_and_schedulers.html) | Stateful vs stateless definition: "a model is stateful if its internal state carries across requests" |
| [statsmodels SARIMAXResults.predict — statsmodels.org](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.predict.html) | `predict(start, end, exog)` signature; requires exogenous variable values for the forecast horizon |
| [statsmodels SARIMAX save/load bug #6542 — github.com/statsmodels](https://github.com/statsmodels/statsmodels/issues/6542) | Known: out-of-sample prediction after save/load with exogenous variables can fail; reported resolved in later releases |
| [PyTorch saving and loading models — docs.pytorch.org](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | `state_dict` pattern; `map_location` for CPU loading; two-step deserialise then `load_state_dict` |

### The warmup problem for NnTemporalModel

`NnTemporalModel` (Stage 11, a TCN) processes a window of `seq_len`
recent hourly feature rows to produce a prediction. To predict hour
T+1, it needs the feature vectors for hours T, T-1, ...,
T-(seq_len-1). These must come from somewhere at serving time.

Three patterns in the literature:

**Pattern 1 — Request includes the full window.** The caller supplies
`seq_len` feature rows as part of the request. Architecturally
cleanest (the server is stateless per-request). Used by Triton's
sequence batcher for RNN-type models where the client maintains the
correlation ID and history. For a single-shot day-ahead forecast this
means a large request body (e.g. 168 rows × 20 features for a weekly
window). Requires the client to hold historical feature data.

**Pattern 2 — Service loads warmup from the registry.** At startup,
the server loads the `warmup_features` artefact saved alongside the
model in the Stage 11 registry. When a prediction is requested, the
server prepends the warmup data to the input before calling
`model.forward`. The caller supplies only the target timestamp (and
optionally new weather values for the forecast horizon). The warmup
data is fixed at training time — a reasonable approximation for
day-ahead forecasting but a known limitation.

**Pattern 3 — Service refuses state-bearing models.** The serving
endpoint only handles stateless models. `NnTemporalModel` returns a
501 or a descriptive error. Simple; limits the demo value.

For Stage 12, Pattern 2 is the most pragmatic: the warmup artefact
exists in the registry by Stage 11's design, the server loads it once
at startup alongside the model weights, and callers are not burdened
with providing historical feature windows. The fixed-warmup limitation
is worth stating explicitly in the documentation as a teaching point
about stateful serving in production vs demo contexts.

### SARIMAX state

`SarimaxModel` carries a fitted `SARIMAXResults` object. After loading
from the registry (via joblib),
`.predict(start=N, end=N+23, exog=X)` requires the exogenous variable
values for the 24-step forecast horizon. The Kalman filter state from
the training sample is embedded in the results object and does not
need to be re-supplied. This is structurally similar to "features-in"
for the exogenous variables. Known historical issue: statsmodels#6542
reports that out-of-sample prediction after a save/load cycle can
fail when `k_exog > 0`. Reported as resolved in statsmodels 0.13+,
but a regression test (round-trip save/load then predict) is strongly
recommended.

### Summary: where Stage 12 should sit

Triton's sequence batcher and BentoML Runners are the production
solutions for stateful inference at scale. For Stage 12, holding
warmup features in the process state at startup achieves the same
result without infrastructure overhead. The complexity difference
between "warmup from registry" and "warmup from caller" is worth
surfacing to a meetup audience as the transition point between a demo
and a production system.

---

## R7 — Single-prediction latency on a laptop

### Canonical sources

| Source | Summary |
|--------|---------|
| [scikit-learn computational performance — scikit-learn.org](https://scikit-learn.org/stable/computing/computational_performance.html) | Linear model prediction: single matrix multiply; memory layout matters for repeated single-row predictions |
| [PyTorch performance tuning guide — docs.pytorch.org](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | `torch.inference_mode()` preferred over `no_grad()`; channels-last, quantisation for CPU |
| [PyTorch model loading discussion — discuss.pytorch.org](https://discuss.pytorch.org/t/model-loading-times/61959) | Community thread; first-load anomalous slowness due to JIT / C++ extension initialisation |
| [Optimising PyTorch CPU inference — towardsdatascience.com](https://towardsdatascience.com/optimizing-pytorch-model-inference-on-cpu/) | `inference_mode` vs `no_grad`, batch-size effects, quantisation benchmarks on CPU |
| [FastAPI lifespan events — fastapi.tiangolo.com](https://fastapi.tiangolo.com/advanced/events/) | The `lifespan` context manager as the idiomatic model-load-once pattern |
| [FastAPI model loading at startup — apxml.com](https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi) | Lifespan vs lazy loading discussion for ML serving |

### Order-of-magnitude estimates

Note: specific millisecond-level published benchmarks for the exact
model sizes in this project were not found in the literature search.
The figures below are informed estimates [speculation].

**Model load from disk (joblib / torch.load)**

- Sklearn linear model (~1 MB joblib file): 10–50 ms cold load on a
  modern SSD. Dominated by Python import overhead on the first call
  in a process.
- SARIMAX results object (~1–5 MB joblib): similar order; Kalman
  filter matrices are numpy arrays, serialised efficiently by joblib.
- `NnMlpModel` (small MLP, ~5–20 MB PyTorch state_dict + the python
  object envelope): `torch.load()` deserialises the state dict, then
  `load_state_dict()` populates the `nn.Module`. For 2–3 hidden
  layers (~100k parameters) on a modern laptop CPU, estimated
  100–500 ms total — dominated by PyTorch's first-time C++ extension
  initialisation rather than I/O. [speculation]
- `NnTemporalModel` (TCN, larger, ~10–50 MB state_dict): I/O-bound
  for larger models plus the same C++ init overhead. Estimated
  200 ms–1 s. [speculation]

A known PyTorch behaviour: the _first_ `torch.load()` call in a
Python process is anomalously slow due to lazy initialisation of C++
extensions. Subsequent calls are faster. This means the first request
after startup is always the slowest.

**Inference per single row**

- Linear model `.predict(1 row)`: single matrix multiply, <1 ms.
- SARIMAX `.predict(24 steps ahead)`: Kalman filter forward pass.
  Estimated 50–500 ms for 24-step-ahead on a simple seasonal
  structure on a laptop. [speculation]
- `NnMlpModel.forward(1 row)` with `torch.inference_mode()`: one
  forward pass through a small MLP. Estimated <5 ms on CPU.
  [speculation]
- `NnTemporalModel.forward(1 window)` with `torch.inference_mode()`:
  TCN forward pass over dilated convolutions. For `seq_len=168` and
  moderate depth, estimated 10–50 ms on CPU. [speculation]

### Load-once-at-startup is required

One-load-per-request is not viable for PyTorch models: even the MLP
incurs 100–500 ms cold load plus C++ extension init on every request,
breaking the sub-second NFR at the model boundary alone. The correct
pattern is to load once at application startup and hold the model in
process memory.

FastAPI's `lifespan` context manager (the replacement for the
deprecated `@app.on_event("startup")`) is the idiomatic approach. The
model is loaded before the first `yield` in the lifespan function,
stored in a module-level dict or dependency-injected state, and held
for the duration of the process. The FastAPI documentation provides a
first-class example of this exact pattern for ML model loading.

For a demo serving one named model: a module-level global loaded in
the lifespan function is sufficient and explicit. For a demo serving
multiple models (model-name as a request parameter): a
`dict[str, Model]` loaded at startup is correct.

---

## R8 — Logging requests and predictions (Stage 18 forward look)

### Canonical sources

| Source | Summary |
|--------|---------|
| [loguru — github.com/Delgan/loguru](https://github.com/Delgan/loguru) | Project's chosen logger; `bind()` for per-record extra fields; `serialize=True` for JSON log output |
| [loguru structured logging — betterstack.com](https://betterstack.com/community/guides/logging/loguru/) | `contextualize()` for async context; `bind()` for synchronous per-record enrichment |
| [Feature logging at model serving — medium.com/better-ml](https://medium.com/better-ml/feature-logging-at-model-serving-de7f9b26e7d6) | Why logging input features alongside predictions enables post-hoc drift analysis; design guidance |
| [Structured logging for ML — mlops-coding-course.fmind.dev](https://mlops-coding-course.fmind.dev/4.%20Validating/4.3.%20Logging.html) | Practical advice: log inputs or hashes, predictions, metadata; use structured fields for parseability |
| [Vertex AI prediction logging — cloud.google.com](https://cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring) | Production pattern: every prediction logged to BigQuery for drift analysis; schema inferred from first 1000 requests |

### Minimum viable log record

For Stage 18 to analyse drift without reopening Stage 12, each request
log record needs at minimum:

- `request_id`: UUID generated per request, for correlation across
  log lines.
- `logged_at`: wall-clock time of the request (UTC ISO 8601).
- `model_name`: the name under which the model was loaded from the
  registry.
- `model_run_id`: the registry run identifier (links back to training
  metadata and metrics).
- `target_dt`: the datetime for which a forecast was requested (the
  primary grouping key for drift analysis).
- `prediction`: the numeric forecast value (float, MW).
- `feature_hash`: a short hash of the feature row (e.g. first 8 hex
  chars of `sha256(feature_row.to_json())`). This enables Stage 18 to
  identify identical inputs without storing the full feature vector,
  and to check for repeated requests at the same timestamp.

Seven fields. With `loguru`'s `logger.bind(...)`, these are emitted as
additional JSON fields alongside the default timestamp and level.
Stage 18 can parse the log file (or a structured log sink) to
reconstruct the prediction series and compute PSI or KS statistics
against the training feature distribution.

### What is gold-plating at Stage 12

- Full feature vector logged verbatim (storage cost and privacy risk;
  the hash is sufficient for Stage 18 drift detection).
- Per-component latency breakdown (useful for performance tuning, not
  needed for drift detection).
- Log rotation configuration (loguru supports it; the default
  single-file sink is sufficient for a demo).
- Log shipper (Fluentd, Loki, Prometheus, etc.) — completely out of
  scope.
- Sampling (Vertex AI recommends every-Nth at high throughput; at
  demo throughput, log every request).

### loguru integration note

`loguru` uses a single global `logger` object. The project already
uses it (listed in `DESIGN.md` §8). Per-request context can be
attached with `logger.bind(request_id=..., model_name=..., ...)` which
creates a per-call enriched logger without mutating the global. Since
the endpoint handler is a synchronous `def` (correct for CPU
inference — see §R1), `logger.bind(...)` is simpler than the async
`logger.contextualize()` pattern, which requires an `async with`
block.

---

## Known pitfalls / CVEs / deprecations

- **FastAPI `@app.on_event("startup"/"shutdown")` deprecated**: use
  the `lifespan` context manager instead. The old handlers still work
  but emit deprecation warnings and may be removed in a future
  FastAPI release.
  [fastapi.tiangolo.com/advanced/events](https://fastapi.tiangolo.com/advanced/events/)
- **Pydantic v2 timezone silent drop** (pydantic#9571):
  `datetime.time` objects with DST-dependent tzinfo can silently lose
  their timezone during serialisation. Use `datetime.datetime` with
  `AwareDatetime`, not bare `datetime.time`.
- **Pydantic v2 TzInfo identity** (pydantic#8683): Pydantic's
  internal `TzInfo(UTC)` does not compare equal to
  `datetime.timezone.utc` using `==` in all code paths. Normalise to
  `.astimezone(datetime.timezone.utc)` explicitly at the boundary; do
  not rely on identity checks on tzinfo objects.
- **Pydantic v2 and pandas tzinfo mismatch** (pydantic#6592):
  inserting a Pydantic-validated `datetime` into a pandas
  `DataFrame` may produce a different tzinfo type from what pandas
  or pyarrow expects, causing dtype inconsistencies on concat.
  Normalise with `pd.Timestamp(value, tz="UTC")`.
- **statsmodels SARIMAX save/load with exogenous variables**
  (statsmodels#6542): out-of-sample prediction after a save/load
  cycle can fail when `k_exog > 0`. Reported as resolved in
  statsmodels 0.13+, but a round-trip regression test is strongly
  recommended for Stage 12.
- **PyTorch first-load anomalous latency**: the first `torch.load()`
  call in a process triggers JIT / C++ extension initialisation and
  is anomalously slow. Do not use the first post-startup prediction
  as the latency benchmark; warm up the model in the lifespan hook
  with a dummy forward pass if a consistent startup-time measurement
  is needed.
- **ujson float truncation**: `ujson` silently truncates floats
  beyond a precision threshold. Since the project does not currently
  use `ujson`, this is not an active risk, but it must not be added
  as a "performance improvement" without understanding the precision
  implications for MW-scale demand forecasts.
- **FastAPI `PydanticInvalidForJsonSchema`** (pydantic/pydantic#8328):
  raised when a Pydantic model uses `PlainValidatorFunction`
  validators that cannot be reflected into JSON Schema. Do not use
  plain-validator lambdas in request/response models; use standard
  field types.

---

## Version / compatibility notes

| Package | Version in play | Relevant note |
|---------|----------------|---------------|
| FastAPI | 0.136.1 (April 2026) | `lifespan` context manager is the current idiom; `on_event` deprecated |
| Starlette | Transitive dep of FastAPI | `TestClient` available via `fastapi.testclient`; do not import from `starlette.testclient` directly |
| Pydantic | v2 (project-wide) | Use `AwareDatetime` for UTC timestamps; `model_dump_json()` preserves full float precision |
| uvicorn | Via `fastapi[standard]` | `uvicorn[standard]` includes `uvloop` for higher throughput; `uvicorn` (no extras) is pure-Python and lighter |
| httpx | Required for `TestClient` | Pulled in by `fastapi[standard]`; no separate `pyproject.toml` entry needed |
| statsmodels | 0.14.x (project-wide) | SARIMAX save/load exog bug fixed in 0.13+; confirm with regression test |
| PyTorch | CPU, project-wide | Use `torch.inference_mode()` (not `no_grad()`) for serving; produces immutable tensors and lower overhead |
| loguru | Project-wide | `logger.bind()` for per-record structured fields; `serialize=True` sink for JSON log output |
| Python | 3.12 (project-wide) | `datetime.UTC` constant available (Python 3.11+); prefer over `datetime.timezone.utc` for readability |
