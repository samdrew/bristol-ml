# Stage 4 — Linear regression baseline + evaluation harness: codebase map

**Purpose.** Pre-implementation orientation for the Stage 4 implementing team. Covers consumable surfaces from Stages 0–3, filesystem layout to create, patterns to follow, deferred housekeeping items, NESO forecast data source status, and integration hazards. Stage 4 intent lives at `docs/intent/04-linear-baseline.md`.

---

## A. Consumable Stage 3 surfaces

### `bristol_ml.features.assembler.load`

**File:** `/workspace/src/bristol_ml/features/assembler.py`

```python
def load(path: Path) -> pd.DataFrame: ...
```

Reads the feature-table parquet, validates every column in `OUTPUT_SCHEMA` field-by-field (name, arrow type), then rejects extra columns. Returns a `pd.DataFrame`. Raises `ValueError` on missing or extra columns. Wall-clock cost: a warm-cache read of a ~8 760-row single-year parquet is roughly 6 ms; a full multi-year file is still a sub-second I/O call.

`OUTPUT_SCHEMA` is `pa.schema` with these columns, in this order (order is contractual):

| Column | Arrow type |
|--------|-----------|
| `timestamp_utc` | `timestamp[us, tz=UTC]` |
| `nd_mw` | `int32` |
| `tsd_mw` | `int32` |
| `temperature_2m` | `float32` |
| `dew_point_2m` | `float32` |
| `wind_speed_10m` | `float32` |
| `cloud_cover` | `float32` |
| `shortwave_radiation` | `float32` |
| `neso_retrieved_at_utc` | `timestamp[us, tz=UTC]` |
| `weather_retrieved_at_utc` | `timestamp[us, tz=UTC]` |

Guarantees (load-bearing for Stage 4): tz-aware UTC `timestamp_utc`, strictly monotonically ascending and unique; no NaN anywhere; `int32` demand columns; `float32` weather columns; scalar-per-run provenance columns.

### `bristol_ml.features.assembler.assemble`

```python
def assemble(cfg: AppConfig, cache: str = "offline") -> Path: ...
```

Used when the baseline CLI wants cache-auto semantics rather than pre-baked parquet. Orchestrates `neso.fetch/load → _resample_demand_hourly → weather.fetch/load → national_aggregate → build → _atomic_write`. Returns the absolute path written. Stage 4 only needs this if it provides a `train` subcommand that can populate its own caches; if the notebook and CLI assume a warm cache, calling `load()` directly on the known path is sufficient.

### `bristol_ml.evaluation.splitter.rolling_origin_split_from_config`

**File:** `/workspace/src/bristol_ml/evaluation/splitter.py`

```python
def rolling_origin_split_from_config(
    n_rows: int,
    config: SplitterConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
```

Unpacks `SplitterConfig` fields into `rolling_origin_split`. Returns an `Iterator[tuple[np.ndarray, np.ndarray]]`. Both arrays are `np.int64`; `max(train_idx) < min(test_idx)` guaranteed. Expanding window by default; sliding with `config.fixed_window=True`. The underlying `rolling_origin_split` takes primitive kwargs and is data-structure-agnostic — Stage 4 code that slices the feature DataFrame uses `df.iloc[train_idx]`.

### Pydantic models — `conf/_schemas.py`

**File:** `/workspace/conf/_schemas.py`

All models use `ConfigDict(extra="forbid", frozen=True)`.

- `AppConfig` — top-level; fields relevant to Stage 4: `.project.seed`, `.features.weather_only`, `.evaluation.rolling_origin`. Stage 4 extends it with a `models: ModelsGroup` field and optionally a richer `evaluation: EvaluationGroup`.
- `FeatureSetConfig` — `name`, `demand_aggregation: Literal["mean", "max"]`, `cache_dir: Path`, `cache_filename: str`, `forward_fill_hours: int`.
- `SplitterConfig` — `min_train_periods: int`, `test_len: int`, `step: int`, `gap: int = 0`, `fixed_window: bool = False`. All three of `min_train_periods`, `test_len`, `step` have `ge=1` validators but no explanatory comments (flagged in §D below).
- `FeaturesGroup` — `weather_only: FeatureSetConfig | None = None`.
- `EvaluationGroup` — `rolling_origin: SplitterConfig | None = None`. Stage 4 should add `metrics: MetricsConfig | None = None` here if metric selection is config-driven.

### `conf/config.yaml` — defaults list

**File:** `/workspace/conf/config.yaml`

```yaml
defaults:
  - _self_
  - ingestion/neso@ingestion.neso
  - ingestion/weather@ingestion.weather
  - features/weather_only@features.weather_only
  - evaluation/rolling_origin@evaluation.rolling_origin
```

A new `- model/linear@model.linear` entry slots in here. The `@model.linear` syntax places the resolved YAML under `AppConfig.model.linear` (or whatever nesting the Stage 4 schema uses). No existing default must be removed or reordered.

### Hydra CLI wiring

**Files:** `/workspace/src/bristol_ml/cli.py`, `/workspace/src/bristol_ml/__main__.py`, `/workspace/src/bristol_ml/config.py`

`__main__.py` is a one-liner delegating to `cli.main`. `cli.main` is a `@hydra.main` decorator on a single function that resolves config, validates, and prints. Stage 4 does NOT need to add a new entry point or a subcommand to `cli.py` — DESIGN §7.3 says model selection is via a `model=linear` config group override, not a subcommand. The `train` action is best wired as a new top-level Python module (`bristol_ml.models._cli` or `python -m bristol_ml.models.linear`) following the `_cli_main(argv=None) -> int` pattern from `splitter.py` and `assembler.py`. `load_config(overrides=...)` is the programmatic entry point for tests and notebooks — signature unchanged, safe to call from any Stage 4 module.

### Ingestion template for `neso_forecast.py`

**File:** `/workspace/src/bristol_ml/ingestion/neso.py` (template)  
**Helper:** `/workspace/src/bristol_ml/ingestion/_common.py`

The NESO forecast ingester is a copy-and-adapt of `neso.py`. Key shared helpers from `_common.py`: `CachePolicy`, `CacheMissingError`, `_atomic_write(table, path)`, `_cache_path(config)`, `_retrying_get(client, url, params, config)`, `_respect_rate_limit(last, gap)`. All accept structural `Protocol` types (`RetryConfig`, `RateLimitConfig`, `CachePathConfig`) so any new Pydantic config model that exposes the same attribute names is accepted without inheriting a base class. The retry policy is tenacity 3-attempt exponential backoff (base 1 s, cap 10 s), retrying only `ConnectError | ReadTimeout | HTTP 5xx/429`. Provenance column: `retrieved_at_utc` (`timestamp[us, tz=UTC]`) written on every row, scalar per fetch call.

---

## B. Where Stage 4 artefacts land

All directories under `src/bristol_ml/models/` are **absent today** — no files exist. Similarly `conf/model/` does not exist. Below is what Stage 4 must create versus extend.

### New module: `src/bristol_ml/models/`

Per DESIGN §7.3 and §9, Stage 4 creates:

| File | Role |
|------|------|
| `src/bristol_ml/models/__init__.py` | Lazy re-exports, mirrors `evaluation/__init__.py` shape |
| `src/bristol_ml/models/CLAUDE.md` | Module guide; required by stage-hygiene rules |
| `src/bristol_ml/models/protocol.py` | `Model` Protocol (or ABC); defines `fit`, `predict`, `save`, `load`, `metadata` |
| `src/bristol_ml/models/naive.py` | Seasonal-naive implementation of the `Model` protocol |
| `src/bristol_ml/models/linear.py` | statsmodels OLS implementation |
| `src/bristol_ml/models/io.py` | `save`/`load` helpers (joblib or pickle until Stage 9 registry) |

DESIGN §7.3 gives the exact protocol signature:

```python
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

### Extend: `src/bristol_ml/evaluation/`

Add to the existing module:

| File | Role |
|------|------|
| `src/bristol_ml/evaluation/metrics.py` | Pure `(y_true, y_pred) -> float`: `mae`, `mape`, `rmse`, `wape` |
| `src/bristol_ml/evaluation/harness.py` | `evaluate(model, df, splitter_cfg, metrics) -> pd.DataFrame` fold-level loop |

`evaluation/__init__.py` currently exports only `rolling_origin_split` — extend `__all__` as new symbols land.

### New module: `src/bristol_ml/ingestion/neso_forecast.py`

Does not exist. Template is `neso.py`. Target column is `FORECASTDEMAND` (National Demand, half-hourly). Resource URL: `https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/day_ahead_national_demand_forecast` — **the CKAN resource UUID for the forecast archive is not yet documented anywhere in the repo** (see §E).

### New config files

| File | Notes |
|------|-------|
| `conf/model/linear.yaml` | `# @package model.linear`, `_target_: bristol_ml.models.linear.LinearModel`, hyperparameters |
| `conf/model/naive.yaml` | `# @package model.naive`, `_target_: bristol_ml.models.naive.NaiveModel`, naive-definition choice |
| `conf/ingestion/neso_forecast.yaml` | `# @package ingestion.neso_forecast`, mirrors `neso.yaml` shape |
| `conf/evaluation/metrics.yaml` | Only needed if metric selection is runtime-configurable; evaluation layer doc treats it as a weak signal |

### Test directories (all absent today)

| Path | Role |
|------|------|
| `tests/unit/models/__init__.py` | Required for pytest discovery |
| `tests/unit/models/test_protocol.py` | Protocol-conformance tests for naive and linear |
| `tests/unit/models/test_naive.py` | Unit tests for seasonal-naive logic |
| `tests/unit/models/test_linear.py` | Unit tests for OLS, save/load round-trip |
| `tests/unit/evaluation/test_metrics.py` | Hand-computed metric fixtures (acceptance criterion 4) |
| `tests/unit/evaluation/test_harness.py` | Fold-level harness tests |
| `tests/integration/ingestion/test_neso_forecast_cassettes.py` | VCR cassette tests for forecast ingester |

### Notebook

`notebooks/04_linear_baseline.ipynb` — does not exist. Must import from `bristol_ml.models` and `bristol_ml.evaluation`; must not reimplement logic inline (DESIGN §2.1.8). Template: `notebooks/03_feature_assembler.ipynb`.

---

## C. Patterns to follow

### Pydantic models

Every new Pydantic model in `conf/_schemas.py` uses `ConfigDict(extra="forbid", frozen=True)` — see `SplitterConfig`, `FeatureSetConfig`, `AppConfig`, etc. all at lines 18–205 of `/workspace/conf/_schemas.py`.

### Hydra config group files

`# @package <group>.<name>` as the first non-blank line; no bare `name:` field at top-level (the `@package` header is the namespace). See `/workspace/conf/features/weather_only.yaml` line 1 and `/workspace/conf/evaluation/rolling_origin.yaml`.

### Parquet write idiom

`_atomic_write(table: pa.Table, path: Path)` in `/workspace/src/bristol_ml/ingestion/_common.py` lines 201–210: write to `path.with_suffix(path.suffix + ".tmp")` then `os.replace(tmp, path)`. Stage 4's forecast ingester and any model-serialisation that uses parquet must follow this.

### Loguru + caplog fixture

Production code uses `from loguru import logger`. Tests that assert on log lines use the `loguru_caplog` fixture at `/workspace/tests/conftest.py` lines 16–34 — add `loguru_caplog` as a parameter and then call `caplog.handler` propagation is wired automatically. The evaluation layer doc (`docs/architecture/layers/evaluation.md`) explicitly names the harness as the second planned caller.

### `OUTPUT_SCHEMA` module-level constant

Each module that owns a parquet schema declares `OUTPUT_SCHEMA: pa.Schema = pa.schema([...])` at module level. See `/workspace/src/bristol_ml/features/assembler.py` lines 87–101 and `neso.py`. Stage 4 model serialisation may not need a parquet schema (joblib/pickle is simpler), but the NESO forecast ingester needs its own `OUTPUT_SCHEMA`.

### Provenance column

Every ingested parquet carries `retrieved_at_utc: pa.timestamp("us", tz="UTC")` — scalar per fetch, repeated on every row (DESIGN §2.1.6). See `neso.py` and `weather.py`.

### Hydra override testing idiom

`tests/unit/test_config.py` uses `load_config(overrides=["key=value"])` directly — no subprocess, no `@hydra.main` decorator in tests. Example at lines 44–47 (`test_load_config_rejects_unknown_key`). Use the same pattern for any Stage 4 config-group tests.

### Notebook thinness

`notebooks/03_feature_assembler.ipynb` is the template: import from `bristol_ml.*`, call published API functions, never reimplement logic inline. Cell comments explain concepts pedagogically.

### Module CLAUDE.md shape

See `/workspace/src/bristol_ml/features/CLAUDE.md` and `/workspace/src/bristol_ml/evaluation/CLAUDE.md` — sections: current surface (with full signatures), invariants (load-bearing for downstream), expected additions, running standalone, cross-references.

### `_cli_main(argv=None) -> int` pattern

Every standalone module exposes `_cli_main` plus `if __name__ == "__main__": raise SystemExit(_cli_main())`. See `assembler.py` lines 548–567 and `splitter.py` lines 179–227. The CLI parser takes `overrides` as positional nargs and calls `load_config(overrides=...)`.

### `statsmodels` for OLS

`pyproject.toml` already has `statsmodels>=0.14,<1` in `[project.dependencies]`. No new dependency needed for linear regression. `scikit-learn` is not a declared dependency; do not introduce it for OLS.

---

## D. Stage 3 deferred items Stage 4 must touch

### `src/bristol_ml/evaluation/splitter.py` — UTC check

Index validation lives in `rolling_origin_split` lines 104–118 (numeric parameter checks only). There is no UTC check on the caller's DataFrame — the splitter is data-structure-agnostic and receives only `n_rows: int`. If Stage 4 wants UTC validation it belongs in the harness (`harness.py`) where the actual DataFrame is held, not in the splitter. Adding a UTC check to the splitter would break its data-structure-agnosticism invariant.

### `src/bristol_ml/evaluation/__init__.py` — current `__all__`

`__all__ = ["rolling_origin_split"]` at line 24. Stage 4 must extend this as `metrics` and `harness` symbols land. The lazy `__getattr__` pattern must be extended in parallel.

### `conf/_schemas.py` — `SplitterConfig` comment gap

`min_train_periods`, `test_len`, `step` have `ge=1` validators (lines 181–183) but no explanatory comments. The Stage 3 plan (§6 Task T1) flagged this as R5 (explanatory comments per requirements). Stage 4's housekeeping sub-task should add one-line docstring-style inline comments to each field. Note: models on `frozen=True` Pydantic models cannot add `Field(description=...)` after the fact without also touching `AppConfig` field forward declarations — adding Python comments above each field is the lower-friction path.

---

## E. NESO forecast data source

**Is the CKAN resource UUID documented in the repo?** No. The ingestion layer architecture doc (`docs/architecture/layers/ingestion.md` line 114) names `neso_forecast.py` as Stage 4 scope with target column `FORECASTDEMAND`, but gives no UUID or CKAN dataset URL. The research note (`docs/lld/research/01-neso-ingestion.md` §7) cites the data portal landing page URL (`https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/day_ahead_national_demand_forecast`) but does not enumerate the per-year resource UUIDs that `conf/ingestion/neso_forecast.yaml` will need.

**Action required:** the Stage 4 plan must specify at least one known resource UUID (or a range) before implementation begins, following the same year → UUID mapping pattern as `conf/ingestion/neso.yaml`. DESIGN §4.1 says the dataset is "2018 onwards" — the plan author needs to enumerate those UUIDs from the data portal.

**Retry + timeout pattern to reuse:** `_retrying_get` + `_respect_rate_limit` from `_common.py`. The NESO rate-limit guidance (2 req/min) is documented in the demand ingester CLAUDE.md and applies equally to the forecast resource. The `NesoDayAheadForecastConfig` Pydantic model should expose the same structural fields as `NesoIngestionConfig` (`max_attempts`, `backoff_base_seconds`, `backoff_cap_seconds`, `request_timeout_seconds`, `min_inter_request_seconds`, `cache_dir`, `cache_filename`) so `_common.py` helpers accept it without change.

**Alignment decision deferred:** DESIGN §4, Stage 4 intent §"Points for consideration" flags that the NESO forecast is half-hourly and must be aligned to the model's hourly output. The aggregation choice (mean, sum, take-one) is unresolved in the repo and must be recorded as a plan decision before Stage 4 implementation begins.

---

## F. Integration points

### CLI routing — no entry-point change needed

`python -m bristol_ml` calls `cli.main` which is `@hydra.main` — it resolves config, validates, and prints. Adding a `model=linear` config group does NOT require adding a subcommand or changing `cli.py`. Stage 4's train/evaluate action is best a standalone `python -m bristol_ml.models.linear` (or `bristol_ml.models._cli`) following the module-standalone pattern (DESIGN §2.1.1). The `cli.py` print-config behaviour is the Stage 0 demo moment and must continue to work unchanged after Stage 4 adds its config groups.

### `load_config()` — unchanged, safe

`bristol_ml.config.load_config(overrides, config_name, config_path)` at `/workspace/src/bristol_ml/config.py` lines 41–54 is the single programmatic entry point for tests and notebooks. Stage 4 calls it exactly as Stage 3 does; no signature change is needed.

### `project.seed` propagation

`AppConfig.project.seed` is `20260418` in `conf/config.yaml`. It is read by `load_config()` and available on the `AppConfig.project.seed` field but is currently consumed nowhere in the codebase (no production code reads it beyond printing in the Stage 0 demo). Stage 4's linear model is deterministic (OLS has a closed-form solution) so there is no stochastic component to seed. The naive model may have a stochastic element if random tie-breaking is chosen — if so, `cfg.project.seed` is the correct source; pass it explicitly as a constructor argument rather than setting a global random state.

---

## G. Test fixture inventory

### Existing fixtures

| Path | Contents |
|------|---------|
| `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml` | VCR cassette for the NESO demand ingester |
| `tests/fixtures/neso/clock_change_rows.csv` | Hand-crafted BST-transition edge cases |
| `tests/fixtures/weather/cassettes/weather_2023_01.yaml` | VCR cassette for the weather ingester |
| `tests/fixtures/weather/toy_stations.csv` | Small station list for weather tests |

There is **no `tests/fixtures/features/` directory** — Stage 3 generates its fixtures programmatically from `neso.OUTPUT_SCHEMA` and `weather.OUTPUT_SCHEMA` in conftest fixtures (Stage 3 retro §"Design choices"). Stage 4 must do the same: generate synthetic feature-table DataFrames from `assembler.OUTPUT_SCHEMA` in `conftest.py` fixtures or test-local helpers; do not commit binary parquet fixture files.

### `loguru_caplog` fixture

`tests/conftest.py` lines 16–34. Wires loguru into pytest `caplog` at INFO level. Any Stage 4 test that asserts on harness or metric log lines must list `loguru_caplog` as a fixture parameter. The conftest comment explicitly names the harness as the second planned caller.

### Programmatic fixture pattern

See `tests/unit/features/test_assembler.py` and `test_assembler_cli.py` for the established style: build DataFrames that conform to the relevant `OUTPUT_SCHEMA`, write to `tmp_path`, pass paths to the function under test. No network; no pre-committed binary blobs. Stage 4 unit tests for the harness should build a toy feature DataFrame (a few dozen rows, all five weather columns, `nd_mw` as target) inline in the test module.

---

*Map generated 2026-04-19. Read `docs/intent/04-linear-baseline.md` and `docs/plans/` before opening code.*
