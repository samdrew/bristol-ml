# Stage 12 — Minimal serving endpoint

## Goal

Ship a minimal HTTP prediction endpoint — `POST /predict` backed by
the Stage 9 registry — together with the skops migration that makes
the network-facing deserialiser safe.  The pedagogical payoff is the
boundary itself: a model trained in a notebook answers HTTP requests in
a separate process.  The security payoff is the Ctrl+G-mandated
replacement of `joblib` with `skops.io` across all six model families,
blocking the RCE vector that a network-facing deserialiser would
otherwise expose.

## What was built

- **T1 — Dependencies + skops trust-list helper + scaffold.**
  `fastapi`, `uvicorn`, and `skops` added to `pyproject.toml
  [project.dependencies]`.  `src/bristol_ml/models/io.py` gained
  `save_skops` / `load_skops` wrapping `skops.io.dump` / `skops.io.load`
  with the project trust-list (`_PROJECT_SAFE_TYPES`, populated at
  import time by each family via `register_safe_types`).  Any artefact
  containing an unregistered type raises `UntrustedTypeError` naming
  the path and the unexpected types.  The joblib helpers
  (`save_joblib`, `load_joblib`) are retained for one stage with a
  `DeprecationWarning`; they will be removed at Stage 13.  Serving
  package skeleton created: `src/bristol_ml/serving/{__init__.py,
  __main__.py, app.py, schemas.py, CLAUDE.md}`.

- **T2 — `NaiveModel` + `ScipyParametricModel` skops migration.**
  Both families flip to `save_skops` / `load_skops`; state is a small
  dict of numpy / pandas primitives that skops handles natively.
  `ScipyParametricModel` registers its class hierarchy in
  `_PROJECT_SAFE_TYPES`.  File extension changes from `.joblib` to
  `.skops`.  Round-trip tests (`test_naive_save_load_skops_roundtrip`,
  `test_scipy_parametric_save_load_skops_roundtrip`) confirm
  numerical equality under `atol=1e-6`.

- **T3 — `NnMlpModel` + `NnTemporalModel` skops migration (NN
  families).**  The existing bytes-envelope (`state_dict_bytes` +
  scalers + metadata) is repackaged as a skops-safe dict whose
  primitive values are `bytes`, `int`, `str`, `list`, and raw numpy
  arrays — no pickled `nn.Module` instances.  `NnTemporalModel`'s
  envelope includes the `warmup_features` numpy array carried forward
  from Stage 11 D5+.  Both families register their custom classes in
  `_PROJECT_SAFE_TYPES`.  T3 tests confirm `torch.equal` round-trip
  fidelity for `NnMlpModel` and single-row predict equality for
  `NnTemporalModel` (the load-bearing assertion for the D9 Ctrl+G
  reversal).

- **T4 — `LinearModel` + `SarimaxModel` skops migration
  (envelope-of-bytes).**  `results.save(BytesIO())` → raw bytes; wrap
  in `{"format": "statsmodels-bytes-v1", "kind": "OLS"|"SARIMAX",
  "blob": bytes, "feature_columns": [...], "metadata": {...}}`;
  `skops.io.dump(envelope)`.  The envelope contains only skops-safe
  primitive types; the statsmodels objects never go through skops
  directly.  Load reverses the envelope.  T4 tests confirm round-trip
  predict equality and envelope-format validation.

- **T5 — Registry filesystem layout migration.**  `registry._io` flips
  the canonical artefact filename from `model.joblib` to `model.skops`.
  `registry.load` rejects any run directory still carrying
  `model.joblib` with a `RuntimeError` whose message names the path,
  references both formats, and points the operator at the retraining
  migration path.  T5 tests confirm the rejection, that `registry.save`
  writes only `.skops`, and a per-family end-to-end registry round-trip
  parametrised over all six families.

- **T6 — `ServingConfig` + Pydantic schemas + Hydra config.**
  `conf/_schemas.py` gains `ServingConfig` (`registry_dir`, `host`,
  `port`) and `AppConfig.serving: ServingConfig | None = None` (the
  `None` default keeps the train CLI and config-smoke surface
  unchanged).  `conf/serving/default.yaml` (note: group-directory
  layout, not `conf/serving.yaml` — see Surprises) mirrors the schema
  defaults verbatim.  `schemas.py` defines `PredictRequest` and
  `PredictResponse` with `AwareDatetime`-typed `target_dt` and
  `features: dict[str, float]`.

- **T7 — App factory + lifespan + `GET /` + `POST /predict`.**
  `build_app(registry_dir)` constructs the FastAPI application.  The
  lifespan picks the lowest-MAE run across all six families (D6 / D9
  Ctrl+G reversal) and stashes the loaded model in `app.state.loaded`.
  `GET /` returns the default-model summary (run_id, model_name,
  feature_columns).  `POST /predict` is features-in (D4), UTC-
  normalised (D8), default-or-supplied `run_id` (D5), lazy-loaded on
  first non-default request and cached (D7); unknown `run_id` → 404
  with a `detail` naming both the missing id and the registry
  directory.  10 plan-named integration tests green on first run.

- **T8 — Structured logging + standalone CLI + OpenAPI test.**  Seven
  fields bound via `logger.bind(...).info("served prediction")` per
  request (D11 — the load-bearing contract Stage 18 will consume).
  `__main__.py` replaced with the full uvicorn launcher: argparse with
  `ArgumentDefaultsHelpFormatter` surfacing `ServingConfig` defaults,
  Hydra `overrides` positional, lazy imports keeping `--help`
  lightweight.  Three new tests: seven-field log contract (per-field
  type invariants), CLI help exits 0, OpenAPI components named.

## Decisions made here

Plan §1 D1–D13 plus NFR-1..NFR-4 and H-1.  The eleven kept decisions
(after D12 cut and D9 / D10 Ctrl+G reversals) were approved at Ctrl+G
review 2026-04-25.  Key points:

- **D9 Ctrl+G reversal — `nn_temporal` served first-class.**  The
  pre-Ctrl+G plan deferred `nn_temporal` via HTTP 501.  Human rationale
  at Ctrl+G: *"Keeping code in the codebase that isn't supported in the
  primary implementation is bad-practice."*  Stage 11 D5+ already baked
  the `warmup_features` window inside the artefact, so a loaded
  `NnTemporalModel.predict(single_row)` works without any warmup
  semantics in the request body.  The 501 path was removed; the D6
  candidate-set filter was removed; all six families go through
  `model.predict(feature_frame).iloc[0]` uniformly.  The
  `test_serving_prediction_parity_vs_direct_load` parametrised test
  (six-way) is the load-bearing assertion.

- **D10 Ctrl+G reversal — skops adopted as canonical primitive.**
  Pre-Ctrl+G, skops adoption was "deferred per Stage 9 plan D14".
  Human rationale at Ctrl+G: *"Include skops. This includes a network
  facing interface so security should be paramount, as I don't want an
  RCE exploit on my PC."*  Migration is destructive: existing
  `data/registry/*.joblib` artefacts are invalidated.  User must
  retrain.  Backward compatibility was explicitly sacrificed.

- **D7 single highest-leverage cut — startup loads only the default
  model.**  The `@minimalist` scope critic's cut from the pre-plan
  draft, which had proposed loading all registered runs at startup.
  Lazy-on-demand for non-default `run_id`s cuts the startup error
  surface to one model and scales with registry size at zero cost.  Cut
  retained after D9 reversal — all six families are loadable through
  the same skops-safe path; lazy-on-demand is uniform.

- **D4 features-in.**  The caller assembles the feature row; the
  server does no feature engineering.  This is the training-serving-skew
  teaching point and is documented explicitly in the layer doc.

- **D11 — seven-field log schema committed in full.**  The minimalist
  flagged the lead's five-field draft as `PLAN POLISH` and noted it was
  "neither complete enough for Stage 18 nor required by any AC."
  Disposition: commit fully to the seven-field schema.  The cost
  difference is two more `bind` keys; the benefit is Stage 18 has a
  working consumer surface from day one.

- **D12 cut — no column-set boundary assertion.**  `model.predict`
  raises naturally on missing or wrong-named columns; AC-2b (clear
  error on invalid input) is satisfied by the natural error path.
  The cut remains: if a Phase-3 reviewer surfaces a case where the
  framework error is opaque, the guard can be added.

## Surprises captured during implementation

- **AC-3 parametrised test initially failed for `sarimax` and
  `scipy_parametric` due to `metadata.feature_columns` vs raw input
  columns.**  The test built the `features` payload from
  `model.metadata.feature_columns`, which for those two families
  records the *derived* column set (SARIMAX Fourier exogenous regressors,
  ScipyParametric HDD/CDD columns) rather than the raw input columns.
  The serving boundary takes raw input columns and lets the model derive
  the rest — so `metadata.feature_columns` names an unservable payload
  for those families.  Fixed by sourcing the payload keys from
  `predict_features.columns` (the factory's raw-input frame), not from
  `model.metadata.feature_columns`.  The `_features_payload` helper in
  `tests/integration/serving/test_api.py` documents this distinction
  explicitly for the next reader.

- **Hydra group convention: `conf/serving/default.yaml`, not
  `conf/serving.yaml`.**  The T6 implementation initially placed the
  serving config at `conf/serving.yaml` (flat, non-group layout).  Hydra
  does not recognise flat-file paths as group members; the `+serving=
  default` override is only valid when the file lives at
  `conf/serving/default.yaml` (group-directory layout).  Fixed by moving
  the file into a `conf/serving/` subdirectory.  Documented in
  `src/bristol_ml/serving/CLAUDE.md` under Gotchas.

## Tests as documentation

- `test_serving_prediction_parity_vs_direct_load` parametrised over
  all six families — the load-bearing assertion that D9 is upheld: every
  family including `nn_temporal` is served identically to a direct
  `registry.load(run_id).predict(one_row)` call.
- `test_default_model_is_lowest_mae_overall` — pins the D6 + D9
  contract: a registry with six runs where `nn_temporal` scores
  `MAE=10` must resolve `nn_temporal` as the default.
- `test_request_log_record_carries_seven_fields` — pins the D11
  schema; per-field type invariants (UUID4 length, ISO 8601 `T`,
  `len(hex)==16`, `int(hex, 16)` round-trip).
- `test_load_rejects_joblib_artefact_in_registry` (unit) — pins the
  D10 migration outcome: `registry.load` on a `.joblib` run raises
  `RuntimeError` with a message naming the offending path.
- `test_lazy_load_caches_run_id_after_first_request` — pins D7: exactly
  one `registry.load` call for a non-default `run_id` across two
  consecutive predicts; verified via `mock.patch.object`.

## Observed latencies per model family

Observed latencies pending — NFR-4 cut from CI (`PREMATURE OPTIMISATION`
per Scope Diff); measure against a local registry and record here before
the Stage 12 PR merges.  One measurement per family (fit on the
`weather_only` feature table, single-row predict through the serving
boundary, `time.perf_counter` wrapped around `model.predict` only):

| Family | `latency_ms` (observed) |
|--------|------------------------|
| `naive` | — |
| `linear` | — |
| `sarimax` | — |
| `scipy_parametric` | — |
| `nn_mlp` | — |
| `nn_temporal` | — |

## Next-stage handoff

→ Stage 13 — Ingestion stage 2 (real-time NESO demand).  The serving
layer is stable; Stage 13 can consume registered runs without any
serving-layer changes.

Carry-forwards from Stage 12:

- **`save_joblib` / `load_joblib` removal.**  Both deprecated helpers in
  `bristol_ml.models.io` are scheduled for removal at Stage 13.  Any
  external script that still calls them has until Stage 13 to migrate.
- **`H-1` DENY-tier edit (not landed here).**  `docs/intent/DESIGN.md §6`
  should flip the Stage 12 `serving/` entry from `(deferred — Stage 12)`
  to `(shipped)`.  This edit is DENY-tier for the `@docs` agent; it
  needs human approval and is surfaced in the PR description rather than
  landed in this stage.
- **Dispatcher consolidation ADR (H-3 re-deferred).**  Stage 12 did not
  introduce a fourth dispatcher site, but six families now spread across
  three sites.  The `0004-model-dispatcher-consolidation.md` ADR
  earmark carries forward to Stage 13 or Stage 17 (whichever first
  ships a seventh family).
- **Observed latencies.**  Fill in the table above before merging.

## Cross-references

- Plan: `docs/plans/active/12-serving.md` (moves to `completed/` at
  T10).
- Layer doc: `docs/architecture/layers/serving.md`.
- Module guide: `src/bristol_ml/serving/CLAUDE.md`.
- Registry layer: `docs/architecture/layers/registry.md` — skops
  inflection-point section updated.
- Registry module guide: `src/bristol_ml/registry/CLAUDE.md` — skops
  migration outcome.
- Models IO: `src/bristol_ml/models/io.py` — `save_skops`, `load_skops`,
  `register_safe_types`, `UntrustedTypeError`, trust-list contract.
- Predecessor retro: `docs/lld/stages/11-complex-nn.md` — Stage 11 D5+
  warmup-envelope contract that makes `nn_temporal` servable.
