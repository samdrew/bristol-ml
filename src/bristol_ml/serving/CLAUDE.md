# `bristol_ml.serving` — module guide

This module is the **serving layer**: a minimal FastAPI HTTP application
that loads a fitted model from the Stage 9 registry and answers
forecast requests.  Stage 12 introduces the layer; it has no
downstream consumers yet (Stage 18 will consume the structured
prediction log; Stage 19 will schedule batch scoring through the same
artefacts).

Read the layer contract in
[`docs/architecture/layers/serving.md`](../../../docs/architecture/layers/serving.md)
before extending this module; the file you are reading documents the
concrete Stage 12 surface.

## Status

This file is the **T1 scaffold** for Stage 12.  The full module guide
lands at T9 once T2-T8 have shipped the skops migration, the schemas,
the app factory, the predict endpoint, the structured logging, and the
standalone CLI.  The shape of the eventual surface is recorded below
so the scaffold's intent is clear.

## Eventual surface (after T9)

- `bristol_ml.serving.build_app(registry_dir) -> FastAPI` — the public
  entry point.  Loads the lowest-MAE registered run as the default
  model at startup and lazy-loads further `run_id`s on demand.  All
  six model families (`naive`, `linear`, `sarimax`,
  `scipy_parametric`, `nn_mlp`, `nn_temporal`) are served — Stage 11
  D5+ baked the `warmup_features` window into the `nn_temporal`
  artefact, so the boundary treats it identically to the five
  stateless families.
- `bristol_ml.serving.schemas.PredictRequest` / `PredictResponse` —
  Pydantic v2 models with `AwareDatetime`-typed `target_dt` and a
  `features: dict[str, float]` body.  The seven-field structured
  log (`request_id`, `model_name`, `model_run_id`, `target_dt`,
  `prediction`, `latency_ms`, `feature_hash`) is emitted on every
  request via `loguru.logger.bind(...).info("served prediction")` —
  the schema is the load-bearing contract Stage 18 consumes.
- `python -m bristol_ml.serving --help` — standalone CLI launcher.
  Resolves `ServingConfig` via Hydra and starts `uvicorn`.

## Security boundary (D10 — Ctrl+G reversal)

The serving layer is a network-facing deserialiser.  At Ctrl+G the
human directed: *"Include skops. This includes a network facing
interface so security should be paramount, as I don't want an RCE
exploit on my PC."*  Stage 12 therefore migrates **all six** model
families' `save` / `load` paths off `joblib` and onto `skops.io`,
with the envelope-of-bytes pattern for the two model families
(`linear`, `sarimax`) whose statsmodels results objects do not
round-trip cleanly through skops's restricted unpickler.

The serving layer never touches the format directly: it goes through
`registry.load(run_id)`, which goes through each family's
`Model.load(path)`, which now reads `.skops`.

## Cross-references

- Layer contract — `docs/architecture/layers/serving.md` (lands at T9).
- Stage 12 plan — `docs/plans/active/12-serving.md`.
- Registry layer — `docs/architecture/layers/registry.md` and
  `src/bristol_ml/registry/CLAUDE.md` (the single upstream
  consumer).
- Model layer — `src/bristol_ml/models/CLAUDE.md` (every six
  families flow through the same `Model.predict(features)` contract).
