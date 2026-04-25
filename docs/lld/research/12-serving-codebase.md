# Stage 12 — Serving: codebase map

Artefact type: researcher output. Audience: implementing team and plan author.
Baseline SHA: 6267cc0 (Stage 9 merged). Date: 2026-04-25.

---

## 1. Entry points

All existing standalone module surfaces follow `python -m bristol_ml.<module>` via a module-level `__main__.py` that calls an internal `_cli_main()` function and raises `SystemExit` with its return code.

| Entry point | Location | Pattern |
|---|---|---|
| `python -m bristol_ml` | `src/bristol_ml/__main__.py:1` | Imports `cli.main`, no own `_cli_main` |
| `python -m bristol_ml.train` | `src/bristol_ml/train.py` | argparse + `_cli_main()` returning int exit code |
| `python -m bristol_ml.registry` | `src/bristol_ml/registry/__main__.py` | argparse with `list`/`describe` subcommands |
| `python -m bristol_ml.models.nn` | `src/bristol_ml/models/nn/__main__.py` | prints schema; no model fit |
| `python -m bristol_ml.features.assembler` | `src/bristol_ml/features/assembler.py` | orchestration + CLI |

`cli.py` is the Hydra entry point (`@hydra.main`), activated by `python -m bristol_ml`. It calls `config.validate(cfg)` and prints the resolved Pydantic model as JSON — purely a config-smoke surface. It does not train, register, or serve.

`train.py` is the closest analogue for Stage 12: it has its own `argparse` CLI that accepts Hydra overrides as positional strings, calls `load_config(overrides=...)`, instantiates the selected model class, runs evaluation, and saves to the registry. The serving module will follow this same pattern — argparse CLI, `load_config(overrides=...)` call or a `registry_dir` flag, no Hydra decorator on the main entry point (train.py uses the same no-Hydra-decorator pattern for its CLI).

The new `python -m bristol_ml.serving` will sit at:
- `src/bristol_ml/serving/__main__.py` (calls `_cli_main()`)
- `src/bristol_ml/serving/app.py` (HTTP application body + `_cli_main`)

---

## 2. Registry API surface

**Public module:** `bristol_ml.registry` (`src/bristol_ml/registry/__init__.py`)

Four verbs — `save`, `load`, `list_runs`, `describe`. A structural test (`test_registry_public_surface_does_not_exceed_four_callables`) hard-enforces `len(__all__) == 4`.

### `load` — the serving verb

```python
def load(run_id: str, *, registry_dir: Path | None = None) -> Model:
```

- `run_id` must be a single filesystem fragment (validated by `_validate_run_id`; path-traversal guard). Pattern: `{model_name}_{YYYYMMDDTHHMM}`.
- Reads `{registry_dir}/{run_id}/run.json` for the sidecar, dispatches on `sidecar["type"]` via `_dispatch.class_for_type`, then calls `model_cls.load(artefact_path)` where `artefact_path = {registry_dir}/{run_id}/artefact/model.joblib`.
- Returns a fitted `Model` instance ready for `predict()`.
- Raises `FileNotFoundError` if the run directory, sidecar, or artefact is absent.
- Raises `ValueError` if `sidecar["type"]` is not in the dispatch table.

### `describe` — sidecar projection

```python
def describe(run_id: str, *, registry_dir: Path | None = None) -> dict[str, Any]:
```

Returns the full parsed `run.json`. The serving layer needs these fields for a response payload:

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | identity |
| `name` | `str` | human-readable model name |
| `type` | `str` | `"naive"` / `"linear"` / `"sarimax"` / `"scipy_parametric"` / `"nn_mlp"` / `"nn_temporal"` |
| `feature_set` | `str` | which feature set was used at training |
| `target` | `str` | target column name (e.g. `"nd_mw"`) |
| `feature_columns` | `list[str]` | **ordered list the model expects at predict time** |
| `fit_utc` | `str` | ISO-8601 UTC timestamp |
| `metrics` | `dict[str, MetricSummary]` | `{"mae": {"mean": …, "std": …, "per_fold": […]}}` |

### `list_runs` — model selection

```python
def list_runs(
    *,
    target: str | None = None,
    model_type: str | None = None,
    feature_set: str | None = None,
    sort_by: str | None = "mae",
    ascending: bool = True,
    registry_dir: Path | None = None,
) -> list[dict[str, Any]]:
```

Returns sidecar dicts. `sort_by="mae", ascending=True` gives lowest-MAE run first — the natural default for "serve the best available model".

### Default registry location

`bristol_ml.registry.DEFAULT_REGISTRY_DIR = Path("data/registry")` (`src/bristol_ml/registry/__init__.py:59`). This is a module-level constant, not a Hydra/Pydantic config field (plan D17 cut). Override at runtime by passing `registry_dir=` to any verb, or via a CLI flag.

### On-disk layout

```
data/registry/
├── linear-ols-weather-only_20260423T1430/
│   ├── artefact/
│   │   └── model.joblib
│   └── run.json
```

No nested subdirectories inside a run directory. The artefact path is always `{run_dir}/artefact/model.joblib`.

### Security note on joblib

`src/bristol_ml/models/io.py:15-22` explicitly flags this inflection point: joblib is not a safe deserialiser for untrusted inputs. `io.py` carries a docstring comment naming Stage 12 serving as the adoption point for `skops.io`. The registry `CLAUDE.md` echoes it. Whether Stage 12 ships `skops.io` or accepts the single-author risk is a plan decision, not a codebase constraint — the upgrade seam is `load_joblib(path)` in `src/bristol_ml/models/io.py:60`.

---

## 3. Model protocol

**File:** `src/bristol_ml/models/protocol.py`

```python
class Model(Protocol):
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

`predict` contract: accepts any DataFrame whose columns are a **superset** of the training columns; returns a `pd.Series` indexed to `features.index`. Extra columns are silently ignored by all concrete implementations.

### Class-specific quirks load-bearing for serving

**`NaiveModel`** — no index requirement. Lookups into training-time actuals. Fast.

**`LinearModel`** — no index requirement. Statsmodels OLS predict.

**`SarimaxModel`** — requires a UTC-aware `DatetimeIndex`. `sarimax.py:371` raises `ValueError` if the index is not a tz-aware `DatetimeIndex`. Also internally calls `append_weekly_fourier` on the predict input (the model appends Fourier columns itself). The serving layer must supply an hourly UTC-indexed DataFrame. `freq` need not be set on the index — `SarimaxModel` constructs its SARIMAX at fit time with `freq="h"` and the predict path does not re-construct the model.

**`ScipyParametricModel`** — same UTC `DatetimeIndex` guard as SARIMAX (`_require_utc_datetimeindex` at `sarimax.py:371` is duplicated at `scipy_parametric.py` rather than shared, per `CLAUDE.md §SciPy parametric specifics`). The design matrix is built internally from the temperature column + generated Fourier columns; the caller needs to supply at minimum the `temperature_column` named in `ScipyParametricConfig` (default: `"temperature_2m"`).

**`NnMlpModel`** — no index requirement. Feature frame normalised internally via registered scaler buffers in the `torch.nn.Module`. `map_location="cpu"` is set at load time (`mlp.py`) so the serving host need not have CUDA. Any extra columns in the input frame are silently discarded.

**`NnTemporalModel`** — the most complex predict path:
- Requires `len(features) > 0`.
- At load time, the envelope contains `warmup_features`: a DataFrame of the last `seq_len` rows of the training feature frame, stored inside the joblib envelope (`temporal.py:1062`). `load()` restores it at `temporal.py:1159`.
- `predict()` prepends `self._warmup_features` to the input features (`temporal.py:964`), giving `combined = pd.concat([warmup, X_df])` of length `seq_len + len(features)`. The TCN then produces one prediction per row of `features` — the warmup rows are consumed as context only.
- The returned `pd.Series` is indexed to `features.index` (not the combined index).
- `map_location="cpu"` is set at load time; the serving host need not have CUDA.
- A single-row predict call (common in a per-request serving scenario) is valid — the warmup prefix supplies the required context window automatically.

`_warmup_features` is serialised inside the single joblib envelope (`temporal.py:1037-1062`), so `registry.load(run_id)` returns a fully self-contained model ready for single-row predict calls without any external context.

---

## 4. Feature pipeline

**Module:** `bristol_ml.features.assembler`
**File:** `src/bristol_ml/features/assembler.py`

Two feature sets are possible at training time:

| Feature set | Schema constant | Columns |
|---|---|---|
| `weather_only` | `OUTPUT_SCHEMA` | 10 cols: `timestamp_utc`, `nd_mw`, `tsd_mw`, 5 weather floats, 2 provenance timestamps |
| `weather_calendar` | `CALENDAR_OUTPUT_SCHEMA` | 55 cols: above 10 + 44 calendar `int8` one-hots + `holidays_retrieved_at_utc` |

The five **weather variable columns** (from `assembler.WEATHER_VARIABLE_COLUMNS`): `temperature_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation` — all `float32`.

The 44 **calendar columns** (from `features.calendar.CALENDAR_VARIABLE_COLUMNS`): 23 hour-of-day one-hots, 6 day-of-week one-hots, 11 month one-hots, 4 holiday flags — all `int8`. The UTC hour (not local) drives the hour-of-day dummies.

The assembler pipeline is: `neso.fetch/load → _resample_demand_hourly → weather.fetch/load → national_aggregate → build`. `build()` does an inner join on `timestamp_utc`, forward-fills weather gaps up to `config.forward_fill_hours`, drops remaining NaN rows, projects to schema column order. **No NaN values in the output.**

The `sidecar["feature_columns"]` list contains the model's **actual training column set** — a subset of the feature table columns (e.g. the 5 weather columns for a weather-only LinearModel, or all 49 weather+calendar columns for a SarimaxModel on the calendar feature set). The serving layer reads `feature_columns` from the sidecar to know what columns to expect.

### Open question for the plan: raw vs assembled inputs

The intent §Points for consideration flags this explicitly. The raw pipeline requires: NESO demand (ingestible, but demand is the *target*, not an input for serving), weather data from Open-Meteo, optional holiday data. For a day-ahead forecast request the serving input naturally is: a DatetimeIndex of target hours + weather forecast values for those hours. Passing assembled feature rows (i.e. the caller pre-populates all `feature_columns` values) is technically simpler but leaks the training column names as an API contract. The codebase map cannot resolve this — it is the plan's core design question.

---

## 5. Config architecture

**Files:**
- `src/bristol_ml/config.py` — `load_config`, `validate`
- `conf/_schemas.py` — Pydantic models
- `conf/config.yaml` — Hydra defaults list

The resolution pipeline is:

```
Hydra compose (YAML + overrides)
  → DictConfig
  → config.validate(cfg) → AppConfig  (Pydantic, extra="forbid", frozen=True)
  → passed to CLI body as plain typed object
```

`bristol_ml.config.load_config(overrides, config_name, config_path)` is the programmatic entry point (`config.py:41`); tests and notebooks use it directly. `validate(cfg)` accepts either a `DictConfig` or a plain mapping.

`AppConfig` has `model: ModelConfig | None` with a discriminated union keyed on `"type"`. `ModelConfig` is the union of all six config types (`NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig | NnMlpConfig | NnTemporalConfig`).

The serving config will add a `ServingConfig` sub-model to `AppConfig` (following the same `extra="forbid", frozen=True` pattern) and a `conf/serving.yaml` group file. Addition to `AppConfig` requires adding one optional field plus a new `defaults` entry in `conf/config.yaml`.

Existing models have no `ServingConfig` in `AppConfig` today — the `serving` field is entirely absent. The pattern to follow: `evaluation: EvaluationGroup = Field(default_factory=EvaluationGroup)` — a top-level group with a default factory so pre-serving configs validate unchanged.

---

## 6. Logging

**Library:** `loguru` — `from loguru import logger` in every production module that emits structured logs (`train.py:58`, `models/linear.py`, `models/sarimax.py`, `models/nn/mlp.py`, `models/nn/temporal.py`, `features/assembler.py`, `evaluation/harness.py`, etc.).

**Pattern:** structured keyword interpolation via `logger.info("text {}", value)`. No `%`-style formatting. INFO for normal operations; WARNING for covariance singularity, zero-variance features, SARIMAX convergence.

**Test adapter:** `tests/conftest.py:16` provides a `loguru_caplog` fixture that routes loguru into pytest's `caplog` via `logger.add(caplog.handler, ...)`. Any serving tests that assert on logged request lines need this fixture.

No `logging.basicConfig` or stdlib `logging` anywhere in production code. The serving layer should import and use `loguru` directly; do not introduce stdlib `logging`.

---

## 7. Testing patterns

**Layout:** `tests/{unit,integration,fixtures}/`, mirroring `src/bristol_ml/`. Unit tests are in `tests/unit/<module>/`; integration tests are in `tests/integration/`.

**Unit test style:**
- `pytest` + `pytest.MonkeyPatch` for env overrides.
- Fixtures as module-level helper functions (e.g. `_write_feature_cache`) or pytest fixtures in `conftest.py`.
- Synthetic data via `np.random.default_rng(seed=42)`.
- No `xfail`, no skipped tests in the default suite.

**Integration test style:**
- Exercises `_cli_main()` in-process (fast path) or via `subprocess.run` for subprocess smoke (e.g. `test_train_cli_registers_nn_mlp.py:1`).
- Writes to `tmp_path` to avoid polluting `data/registry`.
- Asserts on exit code, on registry content via `registry.describe(run_id)`, and on prediction agreement.

**HTTP fixture patterns:** There are none in the current test suite. `httpx` is already a runtime dependency (for ingestion), but the `TestClient`-style pattern used by FastAPI/Starlette test suites (passing the ASGI app directly to `httpx.Client`) does not appear yet. Stage 12 will introduce it. The project does not use `pytest-anyio` or `anyio` — if an async framework is chosen, async test infrastructure will need to be added to the dev group.

**VCR cassettes:** `tests/integration/ingestion/` uses `pytest-recording` + `vcrpy` cassettes for HTTP replay. Ingestion tests are marked `@pytest.mark.vcr`. Serving tests will likely not need VCR cassettes (the serving layer talks to the registry, not external HTTP APIs), but the tooling is present if needed.

**`@pytest.mark.slow` / `@pytest.mark.gpu`:** declared in `pyproject.toml:103`. The default suite runs with `-m 'not slow and not gpu'`. Serving smoke tests should be fast enough to run unmarked.

---

## 8. Dependency surface

**Runtime dependencies** (`pyproject.toml [project.dependencies]`):

| Package | Version | Role |
|---|---|---|
| `httpx>=0.27,<1` | present | ingestion HTTP client; **also usable as a test client for ASGI apps** |
| `pydantic>=2.7,<3` | present | Pydantic v2 |
| `hydra-core>=1.3,<2` | present | config composition |
| `loguru>=0.7,<1` | present | structured logging |
| `joblib>=1.4,<2` | present | model serialisation |
| `pandas>=2.2,<3` | present | DataFrame |
| `torch>=2.7,<3` | present | NN models (lazy-imported) |
| `statsmodels>=0.14,<1` | present | SARIMAX / Linear |
| `scipy>=1.13,<2` | present | ScipyParametric |

**HTTP framework:** none present. Stage 12 will add one (FastAPI + uvicorn is the natural fit given the project already uses Pydantic v2 schemas, but the intent does not mandate it). `httpx` is already a runtime dep with the correct version for use as an `AsyncClient` or via `httpx.Client` wrapping an ASGI transport — this means the dev-group does not need a separate `httpx` test-client addition if an ASGI framework is chosen.

**`skops`:** not present. `io.py:15` flags it as a future upgrade path for Stage 12 (`skops.io` for secure deserialisation). If adopted, it would be a new runtime dependency.

**Dev dependencies** (`[dependency-groups.dev]`):
- `mlflow>=2.14,<3` — test-only PyFunc adapter (Stage 9 D10). Not relevant to serving.
- `pytest-recording`, `vcrpy` — cassette-based HTTP replay for ingestion tests.
- No async test runner (`anyio`, `pytest-asyncio`) — would need adding if an async framework is chosen.

---

## 9. Integration points

### Files to create

| Path | Purpose |
|---|---|
| `src/bristol_ml/serving/__init__.py` | module stub + docstring |
| `src/bristol_ml/serving/app.py` | HTTP application + `_cli_main()` |
| `src/bristol_ml/serving/schemas.py` | Pydantic request/response models |
| `src/bristol_ml/serving/CLAUDE.md` | module guide |
| `src/bristol_ml/serving/__main__.py` | `raise SystemExit(_cli_main())` |
| `conf/serving.yaml` | Hydra config group file |
| `tests/unit/serving/__init__.py` | test package stub |
| `tests/integration/test_serving.py` | acceptance test (endpoint smoke) |

### Files to modify

| Path | Change |
|---|---|
| `conf/_schemas.py` | add `ServingConfig` + optional `serving: ServingConfig \| None` field on `AppConfig` |
| `conf/config.yaml` | add `- serving: default@serving` (or similar) to defaults list |
| `src/bristol_ml/serving/CLAUDE.md` | n/a — new file |
| `CHANGELOG.md` | `### Added` bullet |
| `docs/lld/stages/12-serving.md` | retrospective |

### Call graph for a single request

```
http POST /predict
  → app.py: parse + validate PredictRequest (schemas.py)
  → registry.load(run_id, registry_dir=cfg.serving.registry_dir)
      → _validate_run_id(run_id)
      → json.loads(sidecar_path.read_text())
      → _dispatch.class_for_type(sidecar["type"])
      → model_cls.load(artefact_path)  # joblib.load inside
  → model.predict(features_df)        # features_df built from request body
  → schemas.PredictResponse(predictions=..., run_id=..., feature_columns=...)
  → HTTP 200 JSON
```

On startup (alternative pattern — lazy vs eager load):
- **Eager**: load model at app startup, cache in application state, call `predict()` per request.
- **Lazy**: load model on first request or per-request.
The plan must choose; the registry's `load()` is synchronous and joblib is not async-safe.

---

## Hazards

**H1 — `NnTemporalModel._warmup_features` is inside the joblib envelope, not a separate file.** The warmup DataFrame (last `seq_len` rows of training features) serialises inside `artefact/model.joblib` at `temporal.py:1062`. A serving request for a temporal model therefore requires no extra state beyond the registry run — but if the joblib file is large (large `seq_len`, many feature columns), the load call will materialise the full warmup DataFrame on every new process start. No workaround needed at Stage 12; flag for if a lazy-load pattern is adopted.

**H2 — `SarimaxModel.predict` requires a UTC `DatetimeIndex`.** If the serving request supplies timestamps in any other format (ISO strings, integers, naive datetimes), the `_require_utc_datetimeindex` guard at `sarimax.py:239` will raise `ValueError`. The request schema must construct a UTC-aware `DatetimeIndex` before calling `predict`. This guard is duplicated in `ScipyParametricModel` at the same path — both models must be handled.

**H3 — joblib security seam.** `models/io.py:15` explicitly defers `skops.io` adoption to Stage 12 as "the correct inflection point". The plan must decide: accept the joblib risk (serving is localhost-only, single-author artefacts) or adopt `skops.io` now. If `skops.io` is adopted, the single call site is `load_joblib()` in `io.py:60`; all four model families' `load()` classmethods delegate to it.

**H4 — dispatcher table at `_dispatch.py`.** The registry dispatcher (`src/bristol_ml/registry/_dispatch.py:43`) maps `type` strings to classes. If Stage 12 needs to add a new model type to the dispatch (unlikely — all six are already wired), both `_TYPE_TO_CLASS` and `_CLASS_NAME_TO_TYPE` must be updated. The CLAUDE.md for registry warns against a third dispatcher site — the serving layer should not introduce its own type↔class mapping; rely exclusively on `registry.load`.

**H5 — `AppConfig.model` is the Hydra-resolved *training* model config, not the serving model identity.** The serving endpoint identifies models by `run_id` (a registry concept), not by the Hydra `model=` group. The serving config will carry `run_id: str` (or `"best"` / logic to select best) separately from `AppConfig.model`. At serving time `AppConfig.model` may be `None` or irrelevant; the serving code should not read from it to decide which model to serve.

**H6 — `_NamedLinearModel` round-trip.** `registry.load()` for a `"linear"` sidecar returns a base `LinearModel`, not `_NamedLinearModel` (plan D16 cut, documented in `_dispatch.py:18`). The loaded model's `metadata.name` reflects the base class default, not the dynamic `linear-ols-weather-only` name. For serving purposes the human-readable name should be read from `sidecar["name"]` via `registry.describe()`, not from `model.metadata.name`.

**H7 — four-verb cap on registry.** `registry.__all__` is capped at four members by a structural test. Stage 12 must consume registry functionality through the existing four verbs only. Any "select best model" logic (e.g. `list_runs(sort_by="mae")[0]`) lives in the serving layer, not as a fifth registry verb.

**H8 — `NnMlpModel` and `NnTemporalModel` use `map_location="cpu"` at load time** (documented in `models/nn/CLAUDE.md §PyTorch specifics`, gotcha 4). This is correct for serving on a CPU-only host. If the serving host has CUDA and wants GPU inference, the load path would need modification — out of scope for Stage 12, but the constant is load-bearing.

**H9 — `SarimaxModel.predict` calls `get_forecast(steps=len(features))`.** The statsmodels SARIMAX out-of-sample forecast requires the model's internal time axis to have ended exactly where the in-sample data ended at fit time. For a serving request with a target future DatetimeIndex that is not immediately adjacent to the training end, this will produce a long out-of-sample forecast silently (statsmodels extrapolates). This is a training-serving skew hazard the intent §Points flags directly.

**H10 — no `RegistryConfig` in Hydra.** `DEFAULT_REGISTRY_DIR = Path("data/registry")` is a module-level constant (`registry/__init__.py:59`), not a Pydantic/Hydra config field (plan D17 cut). The serving config must carry `registry_dir: Path` explicitly so the demo facilitator can point at a non-default location via the Hydra override or CLI flag — the same pattern as `--registry-dir` on `train.py` and `registry __main__.py`.
