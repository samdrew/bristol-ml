# Stage 9 — Model registry — codebase map

**Audience:** Stage 9 implementer and plan author, before opening any file.
**Status:** Research note — immutable after Stage 9 ships.
**Date:** 2026-04-22
**Baseline SHA:** `575ac9c`

---

## §1. The Model protocol and what metadata exists today

### Protocol signatures

File: `/workspace/src/bristol_ml/models/protocol.py`

```python
# protocol.py:77
def save(self, path: Path) -> None:
    """Serialise the fitted model to ``path`` atomically."""

# protocol.py:81-83
@classmethod
def load(cls, path: Path) -> Model:
    """Load a previously-saved model instance from ``path``."""

# protocol.py:86-88
@property
def metadata(self) -> ModelMetadata:
    """Immutable provenance record for the most recent fit."""
```

`save` and `load` are both `Path`-in / `Path`-out (or instance-out for `load`).
Neither returns metadata; the caller must read `.metadata` separately after `load`.

### `ModelMetadata` fields

File: `/workspace/conf/_schemas.py:615–656`

| Field | Type | How populated |
|-------|------|---------------|
| `name` | `str` matching `^[a-z][a-z0-9_.-]*$` | Hard-coded in each model's `metadata` property; `_NamedLinearModel` overrides it dynamically (`train.py:413`) |
| `feature_columns` | `tuple[str, ...]` | Stored by `fit()` from the resolved column list |
| `fit_utc` | `datetime \| None` (tz-aware UTC) | Set by `fit()` from `datetime.now(tz=UTC)` |
| `git_sha` | `str \| None` | **Never populated today** — every model passes the schema default of `None`. No `_git_sha_or_none()` helper exists anywhere in `src/`. The registry will own the first implementation of this capture. |
| `hyperparameters` | `dict[str, Any]` | Free-form; populated by each model |

`ModelMetadata` is `frozen=True`, `extra="forbid"`. It is not a Hydra config group; it lives in `conf/_schemas.py` purely for Pydantic tooling.

### Per-model `hyperparameters` population

All four models populate `hyperparameters`; none populate `git_sha`.

| Model | `hyperparameters` keys (always present) | Keys present only after `fit` | Extra model-specific state | File:line |
|-------|----------------------------------------|-------------------------------|---------------------------|-----------|
| `NaiveModel` | `strategy`, `target_column` | — | — | `naive.py:208` |
| `LinearModel` | `target_column`, `fit_intercept` | `coefficients` (dict), `rsquared`, `nobs` | — | `linear.py:181` |
| `SarimaxModel` | `target_column`, `order`, `seasonal_order`, `trend`, `weekly_fourier_harmonics` | `aic`, `bic`, `nobs`, `converged` | — | `sarimax.py:319` |
| `ScipyParametricModel` | `target_column`, `temperature_column`, `diurnal_harmonics`, `weekly_harmonics`, `t_heat_celsius`, `t_cool_celsius`, `loss` | `param_names`, `param_values`, `param_std_errors`, `covariance_matrix` | `covariance_matrix` as `list[list[float]]` — a 13×13 nested list (~1 KB per record) | `scipy_parametric.py:519` |

**Asymmetry:** `ScipyParametricModel` stores a covariance matrix — a large, model-specific artefact — inside `hyperparameters`. No other model has comparable model-specific dense state beyond scalars. This is load-bearing: the registry's sidecar JSON (if it serialises `ModelMetadata`) must handle a `covariance_matrix` key whose value is a `list[list[float]]`. Python's `json.dumps` handles this natively; `float("inf")` entries survive as `Infinity` (non-standard JSON) — the registry should use `json.dumps(..., allow_nan=True)` or normalise infinities to a sentinel.

---

## §2. Current save/load call sites

### `Model.save` call sites

Grepping `\.save(` across `src/bristol_ml/`:

```
src/bristol_ml/train.py:398   — _NamedLinearModel.save delegates to self._inner.save(path)
```

**`Model.save` is called from nowhere in production code today.** `_NamedLinearModel.save` at `train.py:398` is a protocol-compliance forwarding stub, not a call site. The harness (`evaluate`) never calls `.save`; `train.py:_cli_main` never calls `.save`. Models are trained then thrown away; no path is written during a normal train run.

This is the primary integration gap Stage 9 must close: it must insert a `registry.save(model, metrics)` call into the train-CLI flow (or replace the harness loop's post-fold behaviour).

### `Model.load` call sites

Grepping `\.load(` across `src/bristol_ml/`:

```
src/bristol_ml/models/io.py:68   — joblib.load(path), the internal implementation
```

No production caller of `Model.load` exists. The only real `load` calls are in `tests/` (round-trip tests per model family). The registry's `load(name)` will be the first production consumer.

### Paths used

No helper resolves a "save-to" path today. The `data/` directory layout is fully under Hydra for run outputs (`data/_runs/<date>/<time>/cli.log` — but no artefact files). There is no convention for where a fitted model lives on disk.

---

## §3. Where metrics live today

### Harness return shape

`harness.evaluate(...)` at `harness.py:114` returns:

- Default (`return_predictions=False`): `pd.DataFrame` with columns `fold_index`, `train_end`, `test_start`, `test_end`, and one float column per metric (e.g. `mae`, `rmse`).
- With `return_predictions=True`: `tuple[pd.DataFrame, pd.DataFrame]` — (metrics_df, predictions_df).

The metrics DataFrame is **returned to the caller** (`train.py:299`), printed to stdout via `_print_metric_table` (`train.py:316`), and then **discarded**. There is no persistence; metrics are not written to a file.

A summary `INFO` log line is emitted at `harness.py:287` (loguru; total folds, elapsed seconds, per-metric mean ± std). Hydra writes a `cli.log` under `data/_runs/<date>/<time>/` capturing all stdout + loguru output — so metrics are retrievable by reading a log file, but there is no structured artefact.

**Consequence for AC-4 (leaderboard):** the registry must either capture the metrics at save time (passed in by the caller) or re-derive them from the predictions artefact. The intent document §"How the registry interacts with save/load" (intent:44) proposes: "the registry calls the model's `save` to produce an artefact and stores metadata alongside it." The per-fold metric table must be explicitly passed to the registry save call; the harness does not persist it.

---

## §4. The train CLI shape

File: `/workspace/src/bristol_ml/train.py`

### Hydra entry shape

`train.py` does **not** use Hydra `@hydra.main`. It uses `argparse` at `train.py:140` to accept raw override strings, then calls `bristol_ml.config.load_config(overrides=...)` at `train.py:184`. This is the same pattern as `harness.py`'s CLI; it avoids the Hydra working-directory side-effect (no `os.chdir` to `data/_runs/`).

### Artefact path

No model artefact is written today. After the `evaluate` call (`train.py:299`) the per-fold metric table is printed and the function returns. The only file output is Hydra's `cli.log`.

### Metrics destination

Printed to stdout via `_print_metric_table` at `train.py:316`. Not written to a structured file.

### Dispatcher duplication (flagged in Stage 8 retro)

The models layer doc at `docs/architecture/layers/models.md:122` confirmed the open question: two separate `isinstance` ladders exist —

- `harness.py:475` — `_build_model_from_config` (standalone function)
- `train.py:223` — inline `elif` chain inside `_cli_main`

Every new model family requires both to be updated. The Stage 8 retro earmarked ADR `0004-model-dispatcher-consolidation.md` (deferred to Stage 11 or a housekeeping stage). Stage 9 does not add a new model family so it does not trigger the two-site update, but **if Stage 9 adds a registry-driven load path it should not create a third dispatcher site**.

---

## §5. Models layer contract

File: `/workspace/docs/architecture/layers/models.md`

**`name` semantics:** `ModelMetadata.name` (`conf/_schemas.py:634`) is described as "human-readable identifier; unique within a stage". The pattern `^[a-z][a-z0-9_.-]*$` enforces lowercase-slug form. Examples: `naive-same-hour-last-week`, `linear-ols-weather-only`, `sarimax-d1-d1-s168`. The layer doc table (`models.md:74`) calls it "Identifier unique within a stage". It is both the human-readable string and the slug — there is no separate slug vs display name distinction. The registry's "list by name" semantics must work against this field directly.

**save/load semantics (layer doc `models.md:84–90`):** atomic joblib writes via `save_joblib` / `load_joblib`; tmp + `os.replace`. The layer doc explicitly names this as the Stage 9 upgrade seam: "Serialisation backend (joblib → `skops.io` at Stage 9)" (`models.md:100`).

**Upgrade seams relevant to Stage 9 (layer doc `models.md:94–102`):**

| Swappable | Load-bearing |
|-----------|--------------|
| Serialisation backend (joblib → `skops.io`) | `save(path)` / `load(path)` return contract unchanged |
| Model-selection mechanism | `python -m bristol_ml.train model=<family>` CLI surface |

The layer doc says `ModelMetadata` "will likely [be read] verbatim as sidecar JSON" by Stage 9 (`models.md:80`). This is the documented design expectation.

---

## §6. Candidate home for the registry module

Confirmed:

- `/workspace/src/bristol_ml/registry/` — **does not exist**
- `/workspace/src/bristol_ml/registry.py` — **does not exist**
- `/workspace/conf/registry/` — **does not exist**
- `RegistryConfig` in `conf/_schemas.py` — **does not exist**
- `registry` key in `conf/config.yaml` — **does not exist**

`AppConfig` at `conf/_schemas.py:659` has top-level fields: `project`, `ingestion`, `features`, `evaluation`, `model`. There is no `registry` field. Stage 9 must add one if registry configuration (artefact store path) is to be Hydra-controlled.

---

## §7. Data directory conventions

### Layout today

```
data/
├── .gitkeep
├── features/
│   ├── weather_calendar.parquet
│   └── weather_only.parquet
├── raw/
│   ├── holidays/
│   ├── neso/
│   └── weather/
└── _runs/
    ├── 2026-04-18/<time>/cli.log
    └── 2026-04-19/<time>/cli.log
```

`.gitignore` lines `20–21`:
```
data/*
!data/.gitkeep
```
Everything under `data/` is gitignored except the `.gitkeep` sentinel. A registry artefact store under `data/registry/` is idiomatic and safe.

### Atomic-write idiom

File: `/workspace/src/bristol_ml/ingestion/_common.py:201–210`

```python
def _atomic_write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
```

The models layer already mirrors this for joblib writes (`io.py:50–53`):

```python
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
joblib.dump(obj, tmp)
os.replace(tmp, path)
```

The registry's metadata sidecar write must follow the same pattern: write to `.json.tmp` then `os.replace`. This is an existing convention the registry must conform to, not design from scratch.

---

## §8. Downstream consumers — minimal interface required

| Stage | Dependency | What it needs from the registry |
|-------|-----------|----------------------------------|
| **Stage 10** (simple NN) | `registry.save`, `registry.load` | Save a fitted model with its metrics; load by name for checkpointing during training. AC-4: "Save/load through the registry round-trips cleanly, including the fitted weights." |
| **Stage 12** (serving) | `registry.load` | Load a named model by string identifier; must work without a training context. AC-1: "starts on a clean machine without configuration beyond pointing at a registry location." |
| **Stage 17** (price pipeline) | `registry.list`, `registry.save` | List models filterable by `target` so price-target and demand-target entries coexist. AC-4: "registry's list query supports filtering by target." |
| **Stage 18** (drift monitoring) | `registry.load`, access to prediction outputs | Load a model by name to re-run predictions; optionally access prediction logs stored alongside the artefact. |

**Minimal common interface:** `save(model, metrics, *, run_id)`, `load(name)` (or `load(run_id)`), `list(*, target=None, model_type=None)`. `describe(run_id)` is a convenience wrapper over the sidecar JSON. Everything beyond these four is candidate scope creep.

---

## §9. Existing Hydra-group / `ModelConfig` precedent

File: `/workspace/conf/_schemas.py:609–612`

```python
# ``AppConfig.model`` is a Pydantic discriminated union: exactly one of the
# model variants is active per run (matching Hydra group-override semantics).
# The ``type`` discriminator is written into the YAML by each Hydra group file.
ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig
```

`AppConfig.model` (`conf/_schemas.py:670`) is a **discriminated union at top level** — Hydra selects one variant via the `model=<family>` group override. This is the established pattern for swappable components.

For `RegistryConfig`, the question is: top-level on `AppConfig`, nested under `model`, or separate Hydra group? The existing structure has:

- `ingestion` — top-level group, one sub-config per ingester
- `features` — top-level group, one variant active per run
- `evaluation` — top-level group
- `model` — top-level discriminated union

The registry is cross-cutting (not per-model, not per-feature-set) and is shared by train, serve, and monitor. The natural placement is **a top-level `registry` field on `AppConfig`**, similar to `evaluation`. A `RegistryConfig` Pydantic model with fields like `artefact_dir: Path` and `metadata_format: Literal["json"]` would sit in `conf/_schemas.py` alongside the existing groups and be wired via a new `conf/registry/default.yaml` Hydra group file. `conf/config.yaml` would gain:

```yaml
defaults:
  - registry@registry: default    # new line
```

There is no prior `registry` or `artefact_store` key in `conf/config.yaml`.

---

## §10. Surprises and hazards

### H1 — `joblib`-vs-`skops.io` migration: Stage 9 is the named inflection point

`src/bristol_ml/models/io.py:15–18` explicitly names Stage 9 as the upgrade point:

> "Upgrade path: `skops.io` for secure artefacts once the Stage 9 registry lands. joblib (like pickle) is *not* a safe deserialiser for untrusted inputs; at Stage 4 we only ever load artefacts we wrote ourselves, so the audit burden of skops is disproportionate to the stage's demo focus."

`docs/architecture/layers/models.md:100` repeats this as a named upgrade seam. The decision on whether Stage 9 actually performs the migration (vs. deferring again) must be explicit in the plan. `load_joblib` at `io.py:68` is the single call site to replace; the registry is the only new load consumer, so the migration is cheap. If deferred again, the plan must say so and name the new trigger.

### H2 — `git_sha` is never populated: the registry cannot claim "auto-captured"

`ModelMetadata.git_sha` is defined at `conf/_schemas.py:643` as `str | None = None`. The intent document §AC-3 says: "Metadata is captured automatically where it can be (git SHA, timestamps, feature-set name)". However, **zero current models populate `git_sha`**. No `_git_sha_or_none()` helper exists in any model file. The Stage 8 retro's T3 note (`docs/lld/stages/08-scipy-parametric.md`) references `_git_sha_or_none()` as if it exists, but the grep confirms it does not.

Stage 9 must implement git SHA capture itself — either as a shared helper in `registry/` called at save time (preferred; the model is not responsible for provenance it cannot know), or as a helper in `models/io.py` called from each model's `save`. The registry-side capture is cleaner: `subprocess.run(["git", "rev-parse", "--short", "HEAD"])` at registry save time, gracefully returning `None` outside a git tree.

### H3 — SARIMAX `RegressionResultsWrapper` round-trip through a registry layer

Stage 7 retro and `models/CLAUDE.md` both confirm that joblib is "sufficient for the `SARIMAXResultsWrapper` round-trip". A registry wrapping the existing `model.save(path)` call does not change the serialisation; the model still writes itself. The hazard only emerges if Stage 9 attempts to intercept the serialisation (e.g. for `skops.io`), in which case the `SARIMAXResultsWrapper` must also be migrated. At the simpler registry design (call `model.save(path)`, write a sidecar JSON), there is no new risk.

### H4 — `_NamedLinearModel.load` is unimplemented

`train.py:400–408`: `_NamedLinearModel.load` raises `NotImplementedError`. A registry `load(name)` that reconstructs a `_NamedLinearModel` cannot use `_NamedLinearModel.load`. The registry must load via `LinearModel.load(path)` and the name override is lost — or the registry must store the dynamic name in its sidecar JSON and re-apply it on load. This is the only model whose load semantics are broken; all four concrete model classes have working `load` classmethods.

### H5 — Harness output API growth trigger

`evaluation/CLAUDE.md` ("Harness output — API growth trigger") explicitly warns: "Do not add a second boolean flag to `evaluate()` for any future output extension." If Stage 9 needs a run-id column in the per-fold metrics, this rule mandates a `EvaluationResult` dataclass or a `evaluate_v2` rather than `evaluate(..., run_id=...)`. Stage 9 is the second potential ask on the harness API; the trigger is noted.

### H6 — `ScipyParametricModel.hyperparameters["covariance_matrix"]` is large

A 13×13 `list[list[float]]` is ~1.6 KB as JSON. For a small registry (100 entries) this is inconsequential. For a large registry or a model with more parameters it grows quadratically. The registry's sidecar JSON includes the full `ModelMetadata` including `hyperparameters`; this is by design per `models.md:80`, but it is worth noting in the plan.

### H7 — Over-build pressure: Hydra already writes structured run output

`conf/config.yaml:29–34` redirects every Hydra run to `data/_runs/<date>/<time>/`. The `cli.log` in each run directory contains the full stdout (metrics table, benchmark table) from every `python -m bristol_ml.train` invocation. A "registry" implemented as a thin index over these existing run directories (reading the log for metrics + reading the model file from the same dir) requires no new persistence mechanism. This is an under-build option worth considering before adding a dedicated metadata store.

---

## Relevant file index

| Path | Purpose |
|------|---------|
| `/workspace/src/bristol_ml/models/protocol.py` | `Model` protocol; `save`/`load`/`metadata` signatures (lines 77–89) |
| `/workspace/src/bristol_ml/models/io.py` | `save_joblib` / `load_joblib`; skops.io upgrade note (line 15) |
| `/workspace/src/bristol_ml/models/naive.py:208` | `hyperparameters` population; no `git_sha` |
| `/workspace/src/bristol_ml/models/linear.py:181` | `hyperparameters` population; no `git_sha` |
| `/workspace/src/bristol_ml/models/sarimax.py:319` | `hyperparameters` population; no `git_sha` |
| `/workspace/src/bristol_ml/models/scipy_parametric.py:519` | `hyperparameters` population including `covariance_matrix`; no `git_sha` |
| `/workspace/src/bristol_ml/train.py:299–316` | Harness call + metric print (no persistence); dispatcher at lines 223–281 |
| `/workspace/src/bristol_ml/train.py:397–408` | `_NamedLinearModel.save` (working) and `.load` (raises `NotImplementedError`) |
| `/workspace/src/bristol_ml/evaluation/harness.py:114–313` | `evaluate` return shape; per-fold metrics DataFrame |
| `/workspace/src/bristol_ml/evaluation/harness.py:475–495` | `_build_model_from_config` — second dispatcher site |
| `/workspace/src/bristol_ml/ingestion/_common.py:201–210` | `_atomic_write` — the canonical atomic-write idiom |
| `/workspace/conf/_schemas.py:615–656` | `ModelMetadata` — all fields, validators |
| `/workspace/conf/_schemas.py:659–670` | `AppConfig` — where `registry` field must be added |
| `/workspace/conf/config.yaml` | Top-level Hydra defaults list; no `registry` group today |
| `/workspace/docs/architecture/layers/models.md` | `name` semantics, save/load contract, upgrade seams (lines 74–102) |
| `/workspace/docs/intent/09-model-registry.md` | Acceptance criteria (authoritative) |
