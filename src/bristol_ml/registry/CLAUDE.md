# `bristol_ml.registry` — module guide

This module is the **registry layer**: a filesystem-backed store for
fitted-model artefacts with a structured JSON sidecar per run.  It
exposes exactly four verbs — `save`, `load`, `list_runs`, `describe` —
and nothing else (intent AC-1, plan D12).  Stage 9 introduces the
layer; Stages 10, 11, 12, 17, and 18 consume it.

Read the layer contract in
[`docs/architecture/layers/registry.md`](../../../docs/architecture/layers/registry.md)
before extending this module; the file you are reading documents the
concrete Stage 9 surface.

## Current surface (Stage 9)

- `bristol_ml.registry.save(model, metrics_df, *, feature_set, target,
  registry_dir=None)` — atomically writes the fitted model's artefact
  and a `run.json` sidecar; returns the `run_id`.
- `bristol_ml.registry.load(run_id, *, registry_dir=None)` — loads a
  registered run by name, dispatching on the sidecar's `type` field.
- `bristol_ml.registry.list_runs(*, target=None, model_type=None,
  feature_set=None, sort_by="mae", ascending=True, registry_dir=None)` —
  reads every `run.json` under the registry root and returns a filtered,
  sorted list of dicts.  Named `list_runs` rather than `list` to avoid
  shadowing the builtin; the CLI and layer doc both use the verb "list".
- `bristol_ml.registry.describe(run_id, *, registry_dir=None)` — returns
  the full sidecar for a single run.

The module also exposes `DEFAULT_REGISTRY_DIR = Path("data/registry")`.
It is a module-level constant, not a Hydra config (plan D17 — cut for
scope).  Override by passing `registry_dir=` to any verb or via the
`--registry-dir` flag on the CLI.

## Protocol semantics (load-bearing for every downstream stage)

- **Four-verb cap.** `__all__` has exactly four members.  AC-1 makes
  growing the surface a design smell rather than a feature; the
  structural test
  `test_registry_public_surface_does_not_exceed_four_callables` enforces
  it.  Retire a verb before adding one.
- **`list_runs` vs `list`.** The exported symbol is `list_runs`; the
  CLI subcommand and the layer-doc verb are both "list".  The
  rename is a Python-builtin-shadowing concession only.
- **Last-write-wins.** `save()` with a run_id that already exists
  overwrites the previous run (plan D2 + R7).  Minute-precision
  timestamps (D3) make this rare at single-author pace; collisions in
  the same minute are documented as an explicit trade-off.
- **Atomicity.** Writes go into a `.tmp_<uuid>/` staging directory
  inside `registry_dir` and rename via `os.replace` to the final
  `{run_id}/` (plan D5).  Mirrors
  `ingestion/_common.py::_atomic_write` and `models/io.py::save_joblib`.
  A crash mid-write leaves the previous run intact (or absent) —
  never partial.
- **Git SHA auto-capture.** `save()` calls
  `_git.py::_git_sha_or_none()` at save time and stores the result in
  the sidecar's `git_sha` field (plan D13 / AC-3).  A `None` SHA is a
  legitimate registry state (outside a git tree, shallow clone, git
  binary missing) and round-trips as JSON `null`.
- **Metrics are caller-supplied.** `save()` takes a metrics DataFrame
  with one row per fold (the Stage 6 harness's return shape) and
  computes per-metric mean / std / per_fold at save time.  The
  evaluation harness is unchanged (H5 API-growth rule in
  `evaluation/CLAUDE.md` — no second boolean on `evaluate()`).

## On-disk layout

```
data/registry/
├── .tmp_<uuid>/                              # staging dir during save
├── linear-ols-weather-only_20260423T1430/
│   ├── artefact/
│   │   └── model.joblib
│   └── run.json
└── ...
```

The sidecar `run.json` structure is defined by
`bristol_ml.registry._schema.SidecarFields`.  See the layer doc for
field-by-field annotations.

## Serialisation

Artefacts are joblib files written through each model's existing
`Model.save` protocol method (plan D9 — the registry does not duplicate
the serialisation logic).  The sidecar is JSON with
`json.dumps(..., indent=2, allow_nan=True, ensure_ascii=False)` —
`allow_nan=True` lets `ScipyParametricModel`'s covariance matrix round-trip
its `float("inf")` entries (plan R3).

**Security note.** joblib is not a safe deserialiser for untrusted inputs.
Stage 9 only ever loads artefacts we wrote ourselves; the `skops.io`
upgrade seam is flagged in `models/io.py` and moves to Stage 12 (plan
D14) when the serving layer lands.

## MLflow graduation (plan D10)

The public surface is intentionally shaped to make migration to a
hosted registry mechanical.  A test-only `mlflow.pyfunc.PythonModel`
adapter lives under `tests/integration/mlflow_adapter.py` and
`test_registry_run_is_loadable_via_mlflow_pyfunc_adapter` exercises a
round-trip from a registered run through MLflow's loader.  MLflow is
**not** a runtime dependency — it appears only in the `dev`
dependency group.  See the layer doc for the graduation-path
contract.

## Running standalone

    python -m bristol_ml.registry list
    python -m bristol_ml.registry list --target demand_mw --model-type sarimax
    python -m bristol_ml.registry describe <run_id>

The `--help` output documents the `--registry-dir` override.

## Cross-references

- Layer contract — `docs/architecture/layers/registry.md` (Stage 9 T6).
- Stage 9 plan — `docs/plans/completed/09-model-registry.md`.
- Stage 9 retro — `docs/lld/stages/09-model-registry.md`.
- Intent — `docs/intent/09-model-registry.md` (5 ACs + 8 Points).
