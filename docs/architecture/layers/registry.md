# Registry — layer architecture

- **Status:** Provisional — first realised by Stage 9 (filesystem-backed registry with a four-verb public surface). Stage 12 inflection point landed: `skops.io` adopted as canonical serialiser (D10 Ctrl+G reversal; `model.joblib` → `model.skops`); `nn_temporal` added to the dispatcher (Stage 11). Revisit at Stage 17 (price-target models — first cross-target promotion use-case that may require feature-table content hashing per plan D6).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (registry paragraph); [`DESIGN.md` §10](../../intent/DESIGN.md#10-deferred-concerns) (deferred hosted-registry adoption).
- **Concrete instances:** [Stage 9 retro](../../lld/stages/09-model-registry.md) (filesystem layout, four-verb surface, MLflow PyFunc adapter test).
- **Related principles:** §2.1.1 (standalone module), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.5 (idempotent writes), §2.1.6 (provenance), §2.1.7 (tests at boundaries).
- **Key ADR:** [`decisions/0002-filesystem-registry-first.md`](../decisions/0002-filesystem-registry-first.md) — why joblib + sidecar is enough before a hosted registry.

---

## Why this layer exists

The registry layer is the **named slot** for fitted models. Every modelling stage before Stage 9 saved artefacts directly to ad-hoc `data/` paths chosen by the caller; as the stage plan accumulated four model families (naive, linear, SARIMAX, SciPy parametric) the "which model beats which" question moved from a notebook curiosity to a concrete facilitator ask. The registry is the smallest mechanism that answers that question without reintroducing `if family == "sarimax": ...` in every downstream notebook.

The layer is deliberately thin: a directory of runs, a JSON sidecar per run, and a four-verb Python API. It owns naming, provenance capture (git SHA, fit timestamp, feature-set name), and the leaderboard query. It does **not** own training, evaluation, serving, promotion, or model identity beyond "one entry per `registry.save()` invocation".

The load-bearing design constraint is intent AC-1 — *"save, load, list, maybe describe. If it grows past that, the design is drifting."* The public surface is capped at four callables by a structural test; adding a fifth verb requires retiring one first.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| Filesystem-backed run storage under `data/registry/{run_id}/` | ✓ | — |
| JSON sidecar schema (`run.json`) with caller-supplied + auto-captured provenance | ✓ | — |
| Atomic save (temp-dir-then-rename) | ✓ | — |
| Git-SHA capture helper | ✓ | — |
| Four-verb Python API: `save`, `load`, `list_runs`, `describe` | ✓ | — |
| Leaderboard CLI (`python -m bristol_ml.registry list`) | ✓ | — |
| Type-dispatched `load` (naive / linear / sarimax / scipy_parametric / nn_mlp) | ✓ (registry-local dispatcher in `_dispatch.py`) | — |
| Hosted registry (MLflow / W&B) as a runtime dependency | — | deferred (DESIGN §10; test-only PyFunc adapter proves the graduation path, plan D10) |
| Model promotion / staging (dev → staging → prod) | — | out of scope (intent §Out of scope) |
| Versioning beyond last-write-wins | — | out of scope (intent §Out of scope; plan D2) |
| Multi-user concurrent access / locking | — | out of scope (single-process tool) |
| Feature-table content hash | — | deferred (plan D6; promote if Stage 17 needs it) |
| SQLite / Parquet leaderboard index | — | out of scope (flat scan satisfies AC-4 at 100-run scale) |
| Hydra config group (`RegistryConfig`) | — | out of scope (plan D17 cut; constants live at module level) |
| A `delete` / `update` / `promote` verb | — | out of scope (AC-1 four-verb cap) |
| Portable registry export tool | — | out of scope (plan D11) |
| Training a model | — | models layer (`fit`) / evaluation layer (`harness.evaluate_and_keep_final_model`) |
| Serving predictions | — | serving layer (Stage 12) |

The split is enforced by the public surface: anything that is not one of the four verbs should not exist in `bristol_ml.registry`. Private helpers (`_fs.py`, `_git.py`, `_schema.py`, `_dispatch.py`) are structural support that the verbs call into, not extension points.

## On-disk layout

```
data/registry/                                              # DEFAULT_REGISTRY_DIR, gitignored
├── .tmp_{short_uuid}/                                      # staging dir during save (removed on success)
├── linear-ols-weather-only_20260423T1430/
│   ├── artefact/
│   │   └── model.skops                                     # Stage 12 D10: was model.joblib
│   └── run.json
├── sarimax-d1-d1-s168_20260423T1431/
│   ├── artefact/
│   │   └── model.skops
│   └── run.json
└── naive-same-hour-last-week_20260423T1432/
    ├── artefact/
    │   └── model.skops
    └── run.json
```

The directory is gitignored via the repo-wide `data/*` rule; no new `.gitignore` entry is required. A `.gitkeep` sentinel is **not** used — the registry directory is created lazily on first `save()`.

### Naming — the `run_id`

`run_id = f"{metadata.name}_{fit_utc:%Y%m%dT%H%M}"` (minute-precision UTC; plan D3).

Example: `linear-ols-weather-only_20260423T1430`.

| Property | Why |
|----------|-----|
| Human-typeable | The Demo moment tabs through run IDs at the CLI; minute precision is short enough to speak aloud during a live talk. |
| Lexicographically sortable | Sorting IDs alphabetically gives chronological order for free; the `list` CLI leans on this for the default fallback sort. |
| ISO-8601 compact form | `YYYYMMDDTHHMM` is unambiguous across locales; the extended form with hyphens would be illegal on some filesystems. |
| Last-write-wins on same-minute collisions | Two saves in the same minute under the same model name collide (plan D2, R7). Single-author pace makes this rare; multi-worker training would graduate D3 to second precision. |

### Atomic save

Every `save()` writes into `registry_dir/.tmp_{uuid4.hex}/` and renames the completed staging directory to `registry_dir/{run_id}/` via `os.replace`. Mirrors the two existing atomic-write sites in the codebase (`ingestion/_common.py::_atomic_write` and `models/io.py::save_skops`). A crash mid-write leaves no partial run — the previous run (if any) is intact, or the registry directory shows the run as absent.

## Sidecar JSON schema

One `run.json` per run, UTF-8, `json.dumps(..., indent=2, allow_nan=True, ensure_ascii=False)`. The TypedDict at `bristol_ml.registry._schema.SidecarFields` is the structural type; JSON is the wire format.

```jsonc
{
  "run_id": "linear-ols-weather-only_20260423T1430",
  "name": "linear-ols-weather-only",                        // ModelMetadata.name (possibly _NamedLinearModel dynamic name)
  "type": "linear",                                         // "naive" | "linear" | "sarimax" | "scipy_parametric" | "nn_mlp"
  "feature_set": "weather_only",                            // caller-supplied kwarg (required)
  "target": "nd_mw",                                        // caller-supplied kwarg (required)
  "feature_columns": ["temperature_2m", "dewpoint_2m"],     // from ModelMetadata, ordered
  "fit_utc": "2026-04-23T14:30:17+00:00",                   // tz-aware ISO-8601 (second precision)
  "git_sha": "575ac9c",                                     // auto-captured by _git_sha_or_none(); may be null
  "hyperparameters": { "fit_intercept": true },             // verbatim from ModelMetadata.hyperparameters
  "metrics": {
    "mae":  {"mean": 1234.5, "std": 123.4, "per_fold": [1200.0, 1250.0, 1240.0]},
    "rmse": {"mean": 1567.8, "std": 145.6, "per_fold": [1540.0, 1600.0, 1570.0]},
    "mape": {"mean": 0.043,  "std": 0.004, "per_fold": [0.041, 0.046, 0.042]},
    "wape": {"mean": 0.041,  "std": 0.003, "per_fold": [0.040, 0.043, 0.041]}
  },
  "registered_at_utc": "2026-04-23T14:30:18+00:00"          // wall clock at save()
}
```

### Field-by-field

| Field | Source | Type | Notes |
|-------|--------|------|-------|
| `run_id` | Computed | `str` | `{metadata.name}_{YYYYMMDDTHHMM}`. Same value as the parent directory name. |
| `name` | `model.metadata.name` | `str` | Includes the `_NamedLinearModel` dynamic name (e.g. `linear-ols-weather-only`) verbatim. Sidecar is the source of truth for human-readable names; `load()` does **not** re-apply a dynamic wrapper. |
| `type` | `_dispatch.model_type(model)` | `str` | One of `"naive"`, `"linear"`, `"sarimax"`, `"scipy_parametric"`, `"nn_mlp"`, `"nn_temporal"`. Read at `load()` to pick the correct concrete class. |
| `feature_set` | Caller kwarg | `str` | AC-3 explicit: the registry cannot infer which of the Stage 5 feature sets was in play. |
| `target` | Caller kwarg | `str` | AC-3 explicit: same rationale. Enables D7 filtering. |
| `feature_columns` | `model.metadata.feature_columns` | `list[str]` | Ordered list of training-column names. |
| `fit_utc` | `model.metadata.fit_utc` | `str` (ISO-8601, tz-aware) | Second precision; the run_id is minute-precision. |
| `git_sha` | `_git_sha_or_none()` | `str \| None` | AC-3 auto-capture. `None` is a legitimate state (outside a git tree, shallow clone, git binary absent) and round-trips as JSON `null`. |
| `hyperparameters` | `dict(model.metadata.hyperparameters)` | `dict[str, Any]` | Free-form per-family; `ScipyParametricModel` nests `covariance_matrix` as `list[list[float]]` with possible `float("inf")` entries (plan R3). |
| `metrics` | Caller DataFrame | `dict[str, MetricSummary]` | Each `MetricSummary` is `{"mean": float, "std": float, "per_fold": list[float]}`. Non-metric harness columns (`fold_index`, `train_end`, `test_start`, `test_end`) are dropped. `NaN` survives via `allow_nan=True`. |
| `registered_at_utc` | `datetime.now(UTC)` at `save()` | `str` (ISO-8601, tz-aware) | Distinct from `fit_utc` when models are re-registered (e.g. a notebook retrofit of an already-fitted model). |

Strict validation on read would couple the layer to Pydantic; Stage 9 deliberately does not. If Stage 17 or 18 needs a validated read path, promote `SidecarFields` to a Pydantic model under `conf/_schemas.py` as a deliberate surface-widening decision rather than an accretion.

**`allow_nan=True` caveat — CPython consumers only.** `json.dumps(..., allow_nan=True)` emits the tokens `Infinity`, `-Infinity`, and `NaN` for non-finite floats. These tokens are **not** valid per RFC 8259 / ECMA-404; CPython's `json.loads` accepts them (so `registry.load` and `registry.describe` round-trip cleanly), but strict parsers treat them differently: `jq` silently coerces `Infinity` to `1.797…e+308`, and Node's `JSON.parse` / Go's `encoding/json` reject the file outright. Stage 9 is pitched at CPython tooling — the `describe` CLI prints via `json.dumps` and downstream stages read via `registry.describe` — so this is acceptable for the Demo moment. Non-finite values only land in the sidecar when `ScipyParametricModel` fits a singular covariance (research R3), a rare diagnostic state. If a future stage needs tool-agnostic sidecars, encode `float("inf")` as the string `"Infinity"` and `NaN` as `null` inside `_summarise_metrics` and migrate Stage 9 runs on read. Flagged here so a Stage 12 serving-layer implementer who feeds the sidecar into a non-Python parser is not surprised.

## Public interface

Four callables, exactly. `__all__` is tuple-pinned and structurally enforced by `test_registry_public_surface_does_not_exceed_four_callables`.

```python
# src/bristol_ml/registry/__init__.py
__all__ = ("save", "load", "list_runs", "describe")  # AC-1 four-verb cap

DEFAULT_REGISTRY_DIR = Path("data/registry")

def save(
    model: Model,
    metrics_df: pd.DataFrame,
    *,
    feature_set: str,
    target: str,
    registry_dir: Path | None = None,
) -> str: ...                                           # returns run_id

def load(
    run_id: str,
    *,
    registry_dir: Path | None = None,
) -> Model: ...                                          # dispatch on sidecar.type

def list_runs(
    *,
    target: str | None = None,
    model_type: str | None = None,
    feature_set: str | None = None,
    sort_by: str | None = "mae",
    ascending: bool = True,
    registry_dir: Path | None = None,
) -> list[dict[str, Any]]: ...                           # filtered + sorted sidecars

def describe(
    run_id: str,
    *,
    registry_dir: Path | None = None,
) -> dict[str, Any]: ...                                 # one sidecar
```

### Why `list_runs` instead of `list`

Python's `list` builtin would be shadowed inside the module if the exported symbol were `list`. The rename is a single-character concession; the CLI subcommand and every user-facing mention of the verb are plain "list". The layer doc follows suit — when this document says *the list verb* it means `list_runs` the Python callable and `python -m bristol_ml.registry list` the CLI subcommand. They are the same verb on two sides of the CLI boundary.

### Type dispatch on `load()`

`load()` reads the sidecar's `type` field and looks up the concrete class in `bristol_ml.registry._dispatch._TYPE_TO_CLASS`:

| `type` | Class |
|--------|-------|
| `"naive"` | `bristol_ml.models.naive.NaiveModel` |
| `"linear"` | `bristol_ml.models.linear.LinearModel` |
| `"sarimax"` | `bristol_ml.models.sarimax.SarimaxModel` |
| `"scipy_parametric"` | `bristol_ml.models.scipy_parametric.ScipyParametricModel` |
| `"nn_mlp"` | `bristol_ml.models.nn.mlp.NnMlpModel` |
| `"nn_temporal"` | `bristol_ml.models.nn.temporal.NnTemporalModel` (Stage 11) |

The `_NamedLinearModel` wrapper that `train.py` uses for named linear variants dispatches to `"linear"` (via `_CLASS_NAME_TO_TYPE`); `load()` returns a base `LinearModel`, not a re-wrapped `_NamedLinearModel`. The sidecar's `name` field preserves the dynamic name for readers — sidecar-name lookup is the source of truth for the human-readable identifier, not the loaded instance's type.

This is a **registry-local** dispatcher (plan H4) — it lives under `bristol_ml.registry._dispatch` rather than at a shared site because promoting it to a third dispatcher location would violate the Stage 7/8 ADR-avoidance precedent. A registry verb knowing about every model family is intrinsic to its job; the evaluation harness does not need the same knowledge.

## Integration with `bristol_ml.train`

The training CLI's `evaluate` call-site (`train.py`) switched at Stage 9 (plan D17):

1. `bristol_ml.evaluation.harness.evaluate_and_keep_final_model(...)` is called instead of `evaluate(...)`. The new function returns `(metrics_df, final_fitted_model)` — the final-fold model already in harness memory, not re-fit on full data.
2. `train.py` passes both the metrics DataFrame and the final-fold model to `registry.save(...)`, along with caller-supplied `feature_set` and `target` from the resolved Hydra config.
3. The CLI prints `Registered run_id: {run_id}` on success.

Two `train.py` flags govern the wiring:

- `--registry-dir PATH` — override the default `data/registry/` root (used by tests and by facilitators who want throwaway demos).
- `--no-register` — skip the `registry.save()` step entirely (notebook workflows that care about metrics but not leaderboard persistence).

The `evaluation/CLAUDE.md` H5 API-growth rule forbids adding a second boolean flag to `evaluate()` — the new public function `evaluate_and_keep_final_model` exists to respect that rule.

## Why the final-fold model, not a re-fit

The registered artefact is the model as-fit on the final fold of the rolling-origin evaluation, **not** a re-fit on full training data. Rationale:

- The leaderboard metrics are rolling-origin cross-fold summaries; a re-fit-on-full-data artefact would *not* be the model those metrics describe.
- Re-fitting adds a second training pass per CLI invocation for no named requirement.
- The final-fold model is an honest representative of the measurement.

Facilitators interpreting the leaderboard should read `metrics.<metric>.mean` as the rolling-origin cross-fold aggregate (D15) and understand the `artefact/model.skops` as the final-fold representative. The `save()` docstring and the Stage 9 retro both make this trade-off explicit (plan R2).

## Graduation to MLflow

The four-verb interface is the **contract**. Migration to a hosted MLflow registry is mechanical rather than a rewrite — but "mechanical" is a claim that wants a falsifier, so plan D10 ships a test-only PyFunc adapter:

- `tests/integration/mlflow_adapter.py` defines `RegistryPyfuncAdapter(mlflow.pyfunc.PythonModel)` plus a `package_run_as_pyfunc(run_id, dst, *, registry_dir)` helper.
- `tests/integration/test_registry_mlflow_adapter.py` round-trips a registered `NaiveModel`: `registry.save` → `package_run_as_pyfunc` → `mlflow.pyfunc.load_model` → `predict`, asserting `numpy.allclose(..., atol=1e-10)`.
- MLflow is added to the `dev` dependency group only; `pytest.importorskip("mlflow")` guards the test so a lean install without `--group dev` still passes the rest of the suite.

**Contract for future migration:**

1. The adapter wraps `registry.load()` — the four-verb interface does not need to know about MLflow.
2. Run IDs are **not** preserved across `mlflow-export-import`; hosted MLflow's run IDs are MLflow-generated UUIDs.
3. `mlflow.sklearn.load_model` will not load a bare joblib artefact; `mlflow.pyfunc` is the flavour that wraps arbitrary `predict()` surfaces via a small `PythonModel` subclass.
4. The adapter lives under `tests/` (not `src/`) so the public surface stays at four verbs (AC-1) and the runtime dependency footprint stays at zero.

When a future stage promotes MLflow to a runtime dependency, this subsection is the contract to port — rewrite the adapter against the hosted MLflow API, move it to `src/bristol_ml/registry/mlflow.py`, add a fifth verb only if the four-verb API genuinely cannot express the new use-case.

## Serialisation — skops.io (Stage 12 inflection point — adopted)

Artefacts are written through each model's `Model.save` protocol method. The registry does not duplicate serialisation logic (plan D9). `bristol_ml.models.io.save_skops` and `load_skops` are the canonical helpers from Stage 12 onwards; atomic writes and parent-directory creation are handled there.

**Stage 12 D10 — skops adopted (Ctrl+G reversal).** The Stage 9 plan's "Stage 12 inflection point" for `skops.io` adoption has landed. At Stage 12 Ctrl+G the human directed: *"Include skops. This includes a network facing interface so security should be paramount, as I don't want an RCE exploit on my PC."* All six model families' `save` / `load` paths were migrated from `joblib` to `skops.io` as part of Stage 12 (T2–T5). The registry boundary (`registry.__init__.load` via `registry._fs._atomic_write_run`) now writes `model.skops` and rejects any run directory carrying a `model.joblib` artefact with a clear `RuntimeError`.

**Breaking change for existing users.** Any `data/registry/*.joblib` artefact written before Stage 12 is invalidated; `registry.load` rejects it with a migration message. The operator must retrain. This is deliberate — backward compatibility was explicitly sacrificed in favour of security at the Ctrl+G review.

**Trust-list contract for future stages.** `bristol_ml.models.io.load_skops` enforces a project trust-list (`_PROJECT_SAFE_TYPES`). Any new model family added after Stage 12 must call `register_safe_types("module.path.ClassName")` at import time for every custom class that appears in its saved artefact. See `src/bristol_ml/models/io.py` and the serving layer doc at [`layers/serving.md`](serving.md).

**Envelope-of-bytes for statsmodels families.** `LinearModel` and `SarimaxModel` use the envelope-of-bytes pattern: `results.save(BytesIO())` → raw bytes wrapped in a `{"format": "statsmodels-bytes-v1", "kind": ..., "blob": bytes, ...}` dict → `skops.io.dump(envelope)`. The envelope contains only skops-safe primitive types; the statsmodels objects never go through skops directly.

The `save_joblib` / `load_joblib` helpers in `bristol_ml.models.io` are retained for one stage (Stage 12 → Stage 13) with a `DeprecationWarning` to give any external scripts time to migrate. They will be removed at Stage 13 — no exceptions; joblib at the registry boundary is a security regression.

## Module structure

```
src/bristol_ml/registry/
├── __init__.py          # public API: save, load, list_runs, describe
├── __main__.py          # argparse CLI — `python -m bristol_ml.registry {list,describe}`
├── _dispatch.py         # private: type string ↔ concrete class (registry-local, plan H4)
├── _fs.py               # private: run_id / run_dir / _atomic_write_run
├── _git.py              # private: _git_sha_or_none()
├── _schema.py           # private: SidecarFields / MetricSummary TypedDicts
└── CLAUDE.md            # module guide
```

No `RegistryConfig` Pydantic schema; no Hydra config group; no change to `conf/config.yaml` or `AppConfig`. The registry is configured by the `DEFAULT_REGISTRY_DIR` module constant and the `registry_dir=` keyword argument on every verb (plan D17 cut).

## Running standalone

```bash
# Demo moment — a single command for the leaderboard.
uv run python -m bristol_ml.registry list

# D7 filters; combine freely.
uv run python -m bristol_ml.registry list --target nd_mw --model-type sarimax
uv run python -m bristol_ml.registry list --feature-set weather_calendar --sort-by rmse

# AC-1 "maybe describe" — pretty-printed sidecar JSON.
uv run python -m bristol_ml.registry describe linear-ols-weather-only_20260423T1430

# Throwaway demo directory — no effect on data/registry/.
uv run python -m bristol_ml.registry list --registry-dir /tmp/demo-registry
```

The CLI uses `argparse` (same pattern as `train.py`; plan D16) with two subparsers (`list`, `describe`). The leaderboard table is rendered with `str.format` padding — no `tabulate` dependency.

## Non-functional requirements

- **Leaderboard speed.** `list_runs()` over 100 registered runs returns in under 1 s on laptop-class CPU (NFR-speed; AC-4). Typical local-filesystem walk + `json.load` for 100 small sidecars is tens of milliseconds; the 1 s gate exists to catch pathological regressions rather than to describe normal behaviour. Enforced by `test_registry_list_hundred_entries_is_fast`.
- **Hand-inspectability.** `cat data/registry/{run_id}/run.json` prints parseable JSON with `indent=2`, no binary escapes, no YAML dependency (NFR-transparency; AC-5). Enforced by `test_registry_run_json_is_hand_parseable`.

## Upgrade seams

Each row is swappable without touching downstream code. The four-verb API is what is load-bearing.

| Swappable | Load-bearing |
|-----------|--------------|
| On-disk backend (flat filesystem → SQLite index → hosted MLflow) | `save(...) -> run_id` / `load(run_id) -> Model` / `list_runs(...) -> list[dict]` / `describe(run_id) -> dict` |
| Serialisation backend (joblib → `skops.io` at Stage 12) | Each model's `Model.save(path)` / `Model.load(path)` protocol methods |
| Run-ID format (minute-precise → second-precise → UUID) | `run_id` as a unique `str` under `registry_dir/` |
| Sidecar format (JSON → Pydantic-validated JSON → Parquet-indexed JSON) | `SidecarFields` TypedDict structure |
| Leaderboard rendering (str.format table → tabulate → rich) | `list_runs()` returns `list[dict]` |

## Module inventory

| Module | Responsibility | Stage | Status | Notes |
|--------|-----------|-------|--------|-------|
| `registry/__init__.py` | Public API (four verbs, `DEFAULT_REGISTRY_DIR`) | 9 | Shipped | `__all__ = ("save", "load", "list_runs", "describe")` pinned; AC-1 enforced by structural test. |
| `registry/__main__.py` | `argparse` CLI — `list`, `describe` | 9 | Shipped | Same pattern as `train.py`; no Hydra config group (plan D17). |
| `registry/_dispatch.py` | Type string ↔ class lookup | 9 | Shipped | Registry-local (plan H4); not a third shared dispatcher. Includes `_NamedLinearModel` → `"linear"` mapping by class name to avoid a circular import from `train.py`. |
| `registry/_fs.py` | `_build_run_id`, `_run_dir`, `_atomic_write_run` | 9 | Shipped | Mirrors the two existing atomic-write sites in the codebase. |
| `registry/_git.py` | `_git_sha_or_none()` | 9 | Shipped | 2 s subprocess timeout; `None` on any failure (outside git tree, shallow clone, git binary absent). |
| `registry/_schema.py` | `SidecarFields`, `MetricSummary` TypedDicts | 9 | Shipped | Structural type only; promote to Pydantic if Stage 17/18 needs validated read. |

## Open questions

- **When to hash the feature table.** Plan D6 defers content hashing of the Parquet training table. At single-author pace the git SHA + feature-set name + fit timestamp triple is sufficient provenance. Stage 17 (price-target models) may want to tie a registered run to a specific feature-engineering snapshot; that is the inflection point for adding a `feature_table_sha256` field to `SidecarFields`.
- **When to add a SQLite index.** The flat scan at Stage 9 satisfies NFR-speed at 100 runs with an order-of-magnitude safety margin. If a future stage registers thousands of runs (grid search, repeated notebook re-runs) the scan crosses into noticeable latency. The migration point is a SQLite index alongside `run.json` — not a wholesale database rewrite.
- **When to expose a fifth verb.** Retire one first. AC-1 is load-bearing; the structural test is the gate. A hypothetical `promote(run_id, stage)` for dev/staging/prod semantics is the most likely Stage-18 ask; the adjacent layer doc edit would accompany that decision.

## Cross-references

- [`decisions/0002-filesystem-registry-first.md`](../decisions/0002-filesystem-registry-first.md) — ADR establishing joblib + sidecar as sufficient before a hosted registry; superseded at the serialisation boundary by ADR 0005 (Stage 12 skops migration).
- [`decisions/0003-protocol-for-model-interface.md`](../decisions/0003-protocol-for-model-interface.md) — the five-member `Model` protocol the registry consumes without modification (AC-2).
- [`decisions/0005-skops-for-model-serialisation.md`](../decisions/0005-skops-for-model-serialisation.md) — Stage 12 joblib → skops migration; the registry layer is the consumer side of this ADR (`registry.load(run_id)` calls each family's `Model.load(path)`, which calls `bristol_ml.models.io.load_skops`).
- [`layers/models.md`](models.md) — the `Model` protocol contract; `skops.io` migration rationale.
- [`layers/serving.md`](serving.md) — the first non-author consumer of registry artefacts; security rationale for the skops migration; trust-list contract for future model families.
- [`layers/evaluation.md`](evaluation.md) — harness return shape that `registry.save` consumes; `evaluate_and_keep_final_model` extension (plan D17).
- [Stage 9 retro](../../lld/stages/09-model-registry.md) — implementation notes, NFR measurements, AC mapping.
- [Stage 12 retro](../../lld/stages/12-serving.md) — skops migration outcome; D9 + D10 Ctrl+G reversal log.
