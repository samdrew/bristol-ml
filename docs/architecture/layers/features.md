# Features — layer architecture

- **Status:** Provisional — realised by Stage 2 (national weather aggregate, shipped) and Stage 3 (feature assembler, shipped). Revisit at Stage 5 (calendar features) and Stage 16 (REMIT features), each of which extends the layer in a way that will stress these conventions.
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (features paragraph).
- **Concrete instances:** Stage 2 retro [`lld/stages/02-weather-ingestion.md`](../../lld/stages/02-weather-ingestion.md) (the `national_aggregate` function); Stage 3 retro [`lld/stages/03-feature-assembler.md`](../../lld/stages/03-feature-assembler.md) (the assembler).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.6 (provenance), §2.1.7 (tests at boundaries), §2.1.8 (notebook thinness).

---

## Why this layer exists

The features layer is the transformation seam between ingestion's per-source parquet and the modelling stages' single feature table. It is the only place that:

- Reads multiple source parquets (demand + weather + calendar + ...).
- Joins across sources on a canonical time key.
- Derives aggregates, lags, and calendar features.
- Writes a parquet the modelling stages consume directly.

It is deliberately stateless — every function in the layer is pure, takes pandas frames in, and returns a pandas frame out. The orchestration that wires ingestion through to an on-disk feature table (`assemble()` in Stage 3) is the only I/O point, and it is a thin wrapper around the pure functions plus `ingestion._common._atomic_write`.

Three concrete feature-layer artefacts land across the stage plan: the Stage 2 `weather.national_aggregate`, the Stage 3 `assembler` (feature-set `weather_only`), and the Stage 5 extension to a second feature-set `weather_calendar`. Stage 16 extends further when REMIT features land. All four share the conventions below.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| Population-weighted / NaN-safe aggregation across per-source rows | ✓ | — |
| Joining per-source parquets on `timestamp_utc` | ✓ | — |
| Half-hourly → hourly demand resample | ✓ | — |
| Forward-filling short weather gaps | ✓ | — |
| Calendar feature derivation (bank holidays, day-of-week, hour-of-day) | ✓ (Stage 5) | — |
| Lag and rolling-window features | ✓ (Stage 7+) | — |
| Declaring an on-disk `OUTPUT_SCHEMA` for each feature set | ✓ | — |
| HTTP calls, API parsing, cassette recording | — | ingestion layer |
| Train/test splitting, metric functions | — | evaluation layer |
| Model training, fitting, prediction | — | models layer |
| Serving or benchmarking | — | serving / evaluation layers |

The split is enforced by the layer's public surface: every feature-producing function takes pandas frame(s) in and returns a pandas frame out. A function that opens an HTTP connection or writes a file directly is leaking into the ingestion layer; a function that takes a fitted model object is leaking into models.

## Cross-module conventions

Every features module follows the same four-part shape. `features.weather` is the single-function template; `features.assembler` is the multi-function template with on-disk persistence.

### 1. Module shape

- `src/bristol_ml/features/<set>.py` — the feature-producing function(s). Either a single named helper (`weather.national_aggregate`) or a multi-function module with an `assemble()` orchestrator when the output is persisted on disk.
- `conf/features/<feature_set>.yaml` — Hydra group file when the feature set has runtime-configurable behaviour (aggregation mode, forward-fill cap, cache location). Not required for pure derivations without tunable knobs.
- No dedicated fixtures directory — test inputs are generated programmatically from the ingestion layer's declared `OUTPUT_SCHEMA`s, which makes fixture drift impossible.

### 2. Public interface

Features-layer functions split into two shapes:

**Pure derivations** — frame in, frame out, no I/O:

```python
def national_aggregate(df: pd.DataFrame,
                       weights: Mapping[str, float]) -> pd.DataFrame: ...
```

**Feature-set assemblers** — orchestrators that tie ingestion fetch/load through pure derivations to an on-disk parquet:

```python
OUTPUT_SCHEMA: pa.Schema
def build(demand_hourly: pd.DataFrame, weather_national: pd.DataFrame,
          config: FeatureSetConfig, *,
          neso_retrieved_at_utc: pd.Timestamp | None = None,
          weather_retrieved_at_utc: pd.Timestamp | None = None
          ) -> pd.DataFrame: ...
def load(path: Path) -> pd.DataFrame: ...      # validates OUTPUT_SCHEMA
def assemble(cfg: AppConfig, cache: str = "offline") -> Path: ...
```

- `build` is the pure derivation — the core of the module — and must be callable from a unit test without touching the filesystem or the network. Provenance timestamps are keyword arguments so tests can pin them for byte-equality assertions.
- `load` mirrors the ingestion-layer `load(path) -> pd.DataFrame` shape: validates the on-disk schema field-by-field and returns a typed dataframe. Missing or extra columns are hard errors.
- `assemble` is the side-effectful orchestrator: it calls `ingestion.<source>.fetch/load`, runs the pure derivations, and writes atomically via `ingestion._common._atomic_write`. It consumes an `AppConfig` (not a `DictConfig`) — the CLI wrapper is the only Hydra composition point.
- `python -m bristol_ml.features.<module>` is every module's CLI (§2.1.1); minimally prints the output path plus a schema summary.

### 3. Storage conventions

Assembled feature sets write parquet; pure derivations do not persist at this layer.

- **Root:** `data/features/`, configurable via `config.cache_dir`. Defaults to `${BRISTOL_ML_CACHE_DIR:-data/features}`. Gitignored.
- **Format:** Parquet via `pyarrow`, one file per feature set.
- **Filename:** `<feature_set_name>.parquet`, with the feature-set name matching the `conf/features/<name>.yaml` group key. The file lives on the branch `data/features/weather_only.parquet` today; Stage 5 adds `data/features/weather_calendar.parquet` beside it.
- **Schema:** declared as a module-level `OUTPUT_SCHEMA: pa.Schema` constant. Column order, arrow dtypes, and timezone metadata are **contractual** — downstream modelling stages may rely on all three, and may slice columns positionally.
- **Timestamps:** `pa.timestamp('us', tz='UTC')` for the canonical `timestamp_utc`, strictly monotonically ascending, unique. Local-time columns are not retained in the feature table — the ingestion layer's `timestamp_local` is a demo affordance, not a join key.
- **Integer types:** sized to the data — `int32` for MW values, `int8` for hour-of-day / day-of-week when they arrive at Stage 5. No `int64` by default.
- **Atomic writes:** reuse `ingestion._common._atomic_write`. No features-layer copy; the ingestion helper is already battle-tested.
- **Provenance:** two scalar columns per run (`neso_retrieved_at_utc`, `weather_retrieved_at_utc`) — constant across rows within a single `build()` call. Repeats per row on disk; cheap in columnar parquet; makes every feature row auditable back to its source-fetch timestamps.

### 4. Schema assertion at the boundary

Each feature set's module:

- **Declares** `OUTPUT_SCHEMA: pa.Schema` as a module-level constant.
- **Asserts** the schema in `build()` before returning: column names equal `OUTPUT_SCHEMA.names`, in the same order; dtypes are castable to the declared Arrow types; timezone metadata matches.
- **Validates** on `load(path)` — every column the schema names is present; no extra columns; dtypes match. Missing-column and extra-column errors are distinguished, named in the exception message, and halt loading.
- **Forbids** NaN in any persisted row. Demand-NaN rows are dropped upstream; weather gaps shorter than `config.forward_fill_hours` are filled; longer gaps drop the row. The D5 missing-data policy is logged at INFO on every `build()` call so a facilitator can audit what the assembler decided to drop or fill.

Silent schema drift is the specific failure mode this rule prevents. A modelling stage that positionally selects the first five weather columns will break quietly if a new column is inserted upstream; the `OUTPUT_SCHEMA` assert catches the upstream change at the assembler boundary, not in a silently-incorrect metric two stages later.

## Upgrade seams

Each of these is swappable without touching downstream code. The `OUTPUT_SCHEMA` + `build` / `load` / `assemble` interface is what's load-bearing.

| Swappable | Load-bearing |
|-----------|--------------|
| Half-hourly → hourly aggregation (`mean` ↔ `max` ↔ future modes) — `FeatureSetConfig.demand_aggregation: Literal[...]` | `OUTPUT_SCHEMA` column order, names, and dtypes |
| Forward-fill cap (`config.forward_fill_hours`) | `timestamp_utc` tz-aware UTC, monotonic, unique |
| Parquet partitioning (flat → year-partitioned) when Stage 16 REMIT features grow the table | `build()` callable without I/O for testability |
| Schema-enforcement mechanism (pyarrow → pandera, if ever warranted) | `load(path) -> pd.DataFrame` returning a typed frame |
| Calendar-feature source (gov.uk → alternative) at Stage 5 | Scalar-per-run provenance columns |

## Module inventory

| Module | Feature set | Stage | Status | Notes |
|--------|-------------|-------|--------|-----|
| `features/weather.py::national_aggregate` | — (helper) | 2 | Shipped | Pure derivation; no on-disk output. |
| `features/assembler.py` | `weather_only` | 3 | Shipped | First assembled feature set; ten-column `OUTPUT_SCHEMA`; notebook 03 is the demo surface. |
| `features/assembler.py` | `weather_calendar` | 5 | Planning | Extends the same module with a calendar join step; with-without comparison becomes a config swap. |
| `features/lags.py` (tentative name) | — (helper) | 7 | Planning | Lag-feature derivation for SARIMAX and beyond. |
| `features/remit.py` (tentative name) | `with_remit` | 16 | Planning | REMIT bi-temporal features collapsed to the hourly grid. |

## Open questions

- **Feature-set naming convention.** `weather_only` / `weather_calendar` / `with_remit` follow no explicit rule beyond "describe the columns". A prefix / suffix convention would prevent drift when Stages 16+ add more sets. Decide if / when the third set lands.
- **Lag-feature placement.** The line between "features/lags.py computes a lag" and "a model's own preprocessing step adds the lag" is not yet drawn — SARIMAX wants integrated lags, linear wants them as columns, neural may want either. Revisit at Stage 7 when the first lag-hungry model lands.
- **Multi-horizon feature tables.** Every feature row keys on a single `timestamp_utc`. Day-ahead-only forecasting is fine; week-ahead would either need a horizon column or a separate feature set per horizon. Deferred until a week-ahead model is actually in scope.
- **Feature-table partitioning.** Flat single-file parquet will cross GB when REMIT arrives at Stage 16. `pyarrow.dataset.write_dataset` with year partitioning is the natural move; retrofittable without changing the public interface.
- **Pandera introduction.** Considered and explicitly rejected at Stage 3 (decision D2) — pyarrow `OUTPUT_SCHEMA` is idiomatic, dependency-free, and sufficient for the invariants. Revisit only if a feature set needs cross-column invariants (e.g. `timestamp_utc.is_monotonic_increasing` declared as data) that pyarrow cannot express.
- **Lead/lag asymmetry in the forward-fill cap.** Weather gaps are forward-filled only — backward-fill is not applied. On the first few hours after a cold start this may drop rows that a backward-fill would keep. Acceptable at Stage 3's assumption of years-deep history; revisit if a freshly-primed cache becomes a common demo scenario.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (layer responsibilities), [§5](../../intent/DESIGN.md#5-features) (feature set definitions), [§7](../../intent/DESIGN.md#7-configuration-and-extensibility) (configuration).
- [`docs/intent/03-feature-assembler.md`](../../intent/03-feature-assembler.md) — the Stage 3 intent.
- [`docs/lld/stages/02-weather-ingestion.md`](../../lld/stages/02-weather-ingestion.md), [`docs/lld/stages/03-feature-assembler.md`](../../lld/stages/03-feature-assembler.md) — retrospectives applying this architecture.
- [`docs/lld/research/03-feature-assembler.md`](../../lld/research/03-feature-assembler.md) — rolling-origin vocabulary and clock-change aggregation notes.
- [`src/bristol_ml/features/CLAUDE.md`](../../../src/bristol_ml/features/CLAUDE.md) — module-local guide, invariants, Stage 5 expected additions.
- [PyArrow Schema docs](https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html).
