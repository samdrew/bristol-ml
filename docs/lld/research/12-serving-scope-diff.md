# Stage 12 — Scope Diff (pre-synthesis critique)

**Date:** 2026-04-25
**Author:** `@minimalist` (pre-synthesis scope critic)
**Inputs:**
- `docs/intent/12-serving.md`
- `docs/lld/research/12-serving-requirements.md`
- `docs/lld/research/12-serving-codebase.md`
- `docs/lld/research/12-serving-domain.md`
- The lead's draft decision set D1–D13 (synthesised from the three artefacts above prior to plan write).

**Audience:** plan author (lead) + Ctrl+G reviewer. Linked from
`docs/plans/active/12-serving.md` preamble.

---

## 1. Scope Diff table

| ID | Item | Tag | Justification |
|----|------|-----|---------------|
| D1 | FastAPI as HTTP framework | `RESTATES INTENT` | DESIGN.md §8 names FastAPI explicitly: "Serving — FastAPI — Minimal, typed, auto-docs." Cannot be cut without contradicting the design spec. |
| D2 | `fastapi[standard]` extra (pulls uvicorn + httpx) | `PLAN POLISH` | The intent names no specific extras. `fastapi` + `uvicorn` (without `[standard]`) suffice; `httpx` is already a runtime dep so `TestClient` needs no extra. Using `[standard]` also pulls `jinja2`, `python-multipart`, `fastapi-cli` — none required by any AC. Adds ~3 transitive packages with no AC coverage. |
| D3 | Synchronous `def` predict handler | `RESTATES INTENT` | AC-2a, AC-3, AC-5 require a functioning predict path; R1 and R7 confirm sync is the correct choice for CPU inference on this project. |
| D4 | "Features-in" request schema; defer assembler coupling | `RESTATES INTENT` | AC-3 (prediction parity) is trivially impossible with an assembler-in-the-loop unless the assembler is the single source of truth. Intent §Points explicitly calls both schemas "defensible"; D4 resolves that open question. Deferring assembler coupling is the minimal-scope choice the intent supports. |
| D5 | Single `/predict` endpoint, `run_id` in request body | `RESTATES INTENT` | Intent §Scope: "a single prediction endpoint." Intent §Points: "a model-name parameter" is the explicit preference over per-model routing. |
| D6 | Default model = lowest-MAE run at startup; `run_id` override at request time | `RESTATES INTENT` | AC-1 requires zero-config startup beyond registry location. Resolves NFR-5 / OQ-1 with the intent's own suggested answer ("best model from the registry"). |
| D7 | Load **all registered runs** into `dict[str, Model]` at startup via `lifespan` | `PLAN POLISH` (single highest-leverage cut — see §5) | AC-5 / AC-3 need a model loaded; loading the single default model satisfies every AC. Loading *all* registered runs at startup is a generalisation the intent does not require — it adds startup latency proportional to registry size, forces all model families (including `NnTemporalModel`) to be loadable at startup, and adds surface for the lifespan error path. |
| D8 | `AwareDatetime` Pydantic field + `.astimezone(timezone.utc)` normalisation | `RESTATES INTENT` | AC-2b (clear error on invalid input) and AC-3 (parity) require timezone correctness. R2 documents the pydantic#8683 pitfall directly. |
| D9 | Exclude `NnTemporalModel`; return HTTP 501 | `RESTATES INTENT` | Intent does not mention `NnTemporalModel` by name; OQ-4 in requirements identifies its warmup semantics as a distinct problem. 501 is the correct minimal-scope choice. |
| D10 | Defer `skops.io`; document known limitation; localhost-only | `RESTATES INTENT` | Intent §Scope: "Deployment anywhere other than localhost" is out of scope. The security concern only matters when receiving artefacts from untrusted sources; localhost-only single-author artefacts are not that. Explicit documentation satisfies the registry layer doc's "must be made visible" requirement. |
| D11 | Minimal logging via `loguru.bind` (request_id, run_id, target_dt, prediction, latency_ms — five fields) | `PLAN POLISH` | Intent §Points: "minimal logging is easy to add and useful for Stage 18." However the intent does not make it a gate — Stage 18 could retrofit. The five-field schema also omits the Stage 18 minimum identified in R8 (model_name + feature_hash). As drafted, D11 is neither complete enough for Stage 18 nor required by any AC. **Either commit to the R8 seven-field schema for Stage 18 utility, or cut to stdout debug logging and defer the structured schema to Stage 18.** Adds ≥1 test. |
| D12 | Column-set assertion at boundary (no pandera dep) | `PLAN POLISH` | No AC requires a column-set assertion independently of `model.predict` raising naturally on bad input. R4 recommends it as a skew mitigation, but "pointing at training-serving skew" is a pedagogical goal, not an AC. Intent AC-2b only requires a "clear error" — FastAPI's 422 on schema mismatch already satisfies that before the assertion fires. Adds ~10 lines and 1 test case. **Cut unless AC-3 coverage demands it.** |
| D13 | `python -m bristol_ml.serving` standalone CLI | `RESTATES INTENT` | DESIGN.md §2.1.1, NFR-7 in requirements: binding project-wide architectural principle. |
| NFR-1 | Single-request latency < 1 s asserted in CI | `PREMATURE OPTIMISATION` | Intent explicitly says "latency is not a goal here." R7 frames outliers as "talking points." No AC asserts a latency bound. Adding a CI latency assertion guards a failure mode the intent does not name and is machine-dependent. **Cut.** |

---

## 2. Anticipated test surface diff

| Test | Driven by | Tag |
|------|-----------|-----|
| `test_predict_valid_request_returns_200` | AC-2a, AC-5 | `RESTATES INTENT` |
| `test_predict_invalid_request_returns_422` | AC-2b | `RESTATES INTENT` |
| `test_prediction_parity_vs_direct_load` | AC-3 | `RESTATES INTENT` |
| `test_openapi_json_contains_predict_schema` | AC-4 | `RESTATES INTENT` |
| `test_startup_no_config_beyond_registry_dir` | AC-1 | `RESTATES INTENT` |
| `test_logging_fields_emitted_per_request` | D11 only | `PLAN POLISH` — requires `loguru_caplog` fixture; adds test for a decision that is itself `PLAN POLISH` |
| `test_column_set_assertion_raises_on_wrong_schema` | D12 only | `PLAN POLISH` — tests a guard that no AC requires independently of model.predict behaviour |
| `test_get_health_returns_200` | no AC | `PLAN POLISH` — health endpoint is not in any AC |
| `test_sarimax_roundtrip_save_load_predict` | R6 hazard only | `PREMATURE OPTIMISATION` — guards a resolved upstream bug in statsmodels 0.13+ |

---

## 3. Dependency footprint diff

| Dep | Added by | Tag | Notes |
|-----|----------|-----|-------|
| `fastapi` | D1 | `RESTATES INTENT` | Named in DESIGN.md §8. |
| `uvicorn` | D2 (via `[standard]`) | `RESTATES INTENT` | Required to run the server; `uvicorn` (no extras) is sufficient — the `[standard]` wrapper is `PLAN POLISH`. |
| `fastapi[standard]` extras: `jinja2`, `python-multipart`, `fastapi-cli` | D2 | `PLAN POLISH` | None required by any AC; `httpx` already present. Replacing `fastapi[standard]` with `fastapi` + `uvicorn` in `pyproject.toml` removes three transitive packages. |
| `httpx` (already present) | — | `HOUSEKEEPING` | Already a runtime dep for ingestion; no new entry needed. |
| `skops` | D10 defers it | n/a — not added | Correct. |

No unjustified dependencies in the lead's draft provided the
`[standard]` extra is scrutinised.

---

## 4. Notebook diff

The intent says "A small notebook **or** curl example." D4 (features-in
schema) makes a curl example perfectly viable: the request body is a
flat JSON object of named floats plus a `target_dt` string. A curl
one-liner satisfies the intent's demo-moment description ("A
facilitator starts the service, curls it with a sample payload"). A
notebook adds a new file, a new cell test surface, and an additional
maintenance burden for a requirement the intent explicitly marks as
optional via "or".

**A curl example in the documentation satisfies the intent; a notebook
is `PLAN POLISH`.**

---

## 5. The single highest-leverage cut

**If you cut one item to halve this plan's scope, cut D7 (load all
registered runs at startup) and replace it with loading only the single
default model, because this eliminates the need to handle load failures
for every model family at startup, removes the surface area for
`NnTemporalModel`'s lifespan edge case (which D9 already defers at
request time), and reduces the startup error-path tests from
N-models-wide to one — the single largest source of implementation and
test blast radius in the draft.**
