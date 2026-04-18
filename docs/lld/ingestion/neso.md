# LLD — Stage 1, NESO demand ingestion

- **Status:** First-pass design, pre-implementation — 2026-04-18.
- **Scope:** `src/bristol_ml/ingestion/neso.py`, `conf/ingestion/neso.yaml`, `tests/unit/ingestion/test_neso.py`, `tests/integration/ingestion/test_neso_cassettes.py`, `tests/fixtures/neso/`.
- **Authoritative inputs:** `docs/intent/01-neso-demand-ingestion.md` (intent); `docs/architecture/layers/ingestion.md` (layer architecture); `docs/lld/research/01-neso-ingestion.md` (facts).
- **Spec-drift rule:** where this LLD and the stage intent disagree, the intent wins; surface the drift, do not silently rewrite.

## 1. Module layout

```
src/bristol_ml/ingestion/
├── __init__.py
├── CLAUDE.md                   # module-local guide; schema is documented here
├── neso.py                     # fetcher + loader (this stage)
└── _common.py                  # NOT created in Stage 1; arrives when Stage 2 has a second caller

conf/ingestion/
└── neso.yaml                   # Hydra group file

tests/
├── unit/ingestion/
│   └── test_neso.py            # pure-Python logic: settlement-period conversion, schema assertion
├── integration/ingestion/
│   └── test_neso_cassettes.py  # end-to-end via pytest-recording cassettes
└── fixtures/neso/
    ├── cassettes/              # vcrpy YAML; committed; narrow slice of 2023
    └── clock_change_rows.csv   # hand-crafted spring/autumn rows for unit tests
```

## 2. Public interface

```python
# src/bristol_ml/ingestion/neso.py

from enum import Enum
from pathlib import Path
import pandas as pd

from conf._schemas import NesoIngestionConfig  # added to AppConfig in this stage


class CachePolicy(str, Enum):
    AUTO = "auto"          # default: use cache if present, fetch if not
    REFRESH = "refresh"    # always fetch; overwrite cache
    OFFLINE = "offline"    # never touch network; fail if cache missing


def fetch(
    config: NesoIngestionConfig,
    *,
    cache: CachePolicy = CachePolicy.AUTO,
) -> Path:
    """Fetch (or reuse) NESO historic demand. Returns path to the cached parquet."""

def load(path: Path) -> pd.DataFrame:
    """Read the cached parquet, assert schema, return a tz-aware dataframe."""
```

Module CLI — `python -m bristol_ml.ingestion.neso` — calls `fetch(config, cache=CachePolicy.AUTO)` with the composed Hydra config and prints the path. Satisfies principle §2.1.1.

Both callables are re-exported from `bristol_ml.ingestion` so `from bristol_ml.ingestion import neso` works as expected.

## 3. Config shape

### 3.1 Pydantic schema (`conf/_schemas.py`)

```python
class NesoYearResource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    year: int = Field(ge=2001, le=2100)
    resource_id: UUID

class NesoIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: HttpUrl = HttpUrl("https://api.neso.energy/api/3/action/")
    resources: list[NesoYearResource]          # year → UUID catalogue
    cache_dir: Path                            # resolved, absolute
    cache_filename: str = "neso_demand.parquet"
    page_size: int = Field(default=32_000, ge=1, le=32_000)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=10.0, gt=0)
    columns: list[str] = ["ND", "TSD"]         # demand columns to retain
```

`AppConfig` gains an `ingestion: IngestionGroup` field; `IngestionGroup` has an optional `neso: NesoIngestionConfig`. Adding a second feed in Stage 2 adds `weather: WeatherIngestionConfig` as a sibling.

### 3.2 YAML (`conf/ingestion/neso.yaml`)

```yaml
# @package ingestion.neso
base_url: https://api.neso.energy/api/3/action/
cache_dir: ${oc.env:BRISTOL_ML_CACHE_DIR,${hydra:runtime.cwd}/data/raw/neso}
resources:
  - { year: 2018, resource_id: fcb52c4e-... }
  - { year: 2019, resource_id: ... }
  - { year: 2020, resource_id: ... }
  - { year: 2021, resource_id: ... }
  - { year: 2022, resource_id: ... }
  - { year: 2023, resource_id: bf5ab335-9b40-4ea4-b93a-ab4af7bce003 }
  - { year: 2024, resource_id: ... }
  - { year: 2025, resource_id: ... }
columns: [ND, TSD]
page_size: 32000
```

Top `conf/config.yaml` gains:

```yaml
defaults:
  - _self_
  - ingestion/neso@ingestion.neso
```

Rationale: year→UUID is a config decision (research §1); the IDs are discovered from the NESO portal and are not derivable from the year. Adding a year is one YAML line.

## 4. Output parquet schema

Stored at `<cache_dir>/<cache_filename>` (default `data/raw/neso/neso_demand.parquet`).

| Column             | Parquet type                  | Unit / notes                                              |
|--------------------|-------------------------------|-----------------------------------------------------------|
| `timestamp_utc`    | `timestamp[us, tz=UTC]`       | Canonical. Start of the half-hour period in UTC.          |
| `timestamp_local`  | `timestamp[us, tz=Europe/London]` | Pedagogical legibility. Not for arithmetic.           |
| `settlement_date`  | `date32`                      | As reported by NESO (local date of the settlement day).   |
| `settlement_period`| `int8`                        | 1–50. 46 on spring-forward, 50 on autumn-fallback.        |
| `nd_mw`            | `int32`                       | National Demand in MW. GB peak ~60 000, int32 is ample.   |
| `tsd_mw`           | `int32`                       | Transmission System Demand in MW.                         |
| `source_year`      | `int16`                       | Which NESO resource (year) supplied the row.              |
| `retrieved_at_utc` | `timestamp[us, tz=UTC]`       | Provenance (principle §2.1.6).                            |

Primary key: `(timestamp_utc)` — unique. Sorted by `timestamp_utc` ascending.

Embedded wind/solar and interconnector columns are **not** persisted at Stage 1 (intent §"Out of scope"). They are tolerated on ingest (warning-and-drop per the layer architecture, "Schema assertion at ingest") but not written.

Schema is documented in `src/bristol_ml/ingestion/CLAUDE.md` — the module-local guide — so downstream stages can cite it without opening this LLD.

## 5. Data flow

```
  ┌──────────────────────────────────────────────────────────────┐
  │ fetch(config, cache)                                         │
  │                                                              │
  │   1. resolve cache_path from config                          │
  │   2. if cache == OFFLINE:                                    │
  │        require cache_path exists, else raise                 │
  │   3. if cache == AUTO and cache_path exists: return          │
  │   4. for year, resource_id in config.resources:              │
  │        raw_df = _fetch_year(year, resource_id, config)       │
  │        assert_schema(raw_df, year)                           │
  │   5. combined = concat(raw_dfs).sort_values("timestamp_utc") │
  │   6. table = to_arrow(combined, SCHEMA)                      │
  │   7. atomic_write(table, cache_path)                         │
  │   8. return cache_path                                       │
  └──────────────────────────────────────────────────────────────┘
```

Internal helpers (private, lowercase `_prefix`):

- `_fetch_year(year, resource_id, config) -> pd.DataFrame` — paginates CKAN via `offset`, stops when `cumulative >= total`.
- `_assert_schema(df, year) -> None` — required columns present; unknown columns warned and dropped.
- `_to_utc(df) -> pd.DataFrame` — builds `timestamp_utc` from `(SETTLEMENT_DATE, SETTLEMENT_PERIOD)` (§6).
- `_atomic_write(table, path) -> None` — `tmp = path.with_suffix(path.suffix + ".tmp"); pq.write_table(table, tmp); os.replace(tmp, path)`.
- `_retrying_get(url, params, config) -> httpx.Response` — tenacity-wrapped `httpx.get`.

## 6. Settlement period → UTC conversion

> **Updated 2026-04-18 to match shipped behaviour.** The pre-implementation draft of this section reflected a narrower Elexon numbering model (gap-skipping on clock-change days, with periods 3–4 as the ambiguous pair on autumn-fallback and no period 3–4 on spring-forward) that does not match real NESO data. Real Elexon numbering is contiguous: 1..46 on spring-forward, 1..50 on autumn-fallback. The shipped `_to_utc` reconstructs local wall-clock time with a period-dependent shift and enforces the per-day range explicitly.

Research §3 captured the shape of the half-hourly period scheme correctly. The shipped conversion is:

```python
def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    # Elexon numbering is contiguous 1..46 on spring-forward and 1..50 on
    # autumn-fallback. A naive (period-1)*30-min mapping mis-places periods
    # >= 3 on spring-forward (because 01:00-02:00 does not exist) and
    # periods >= 5 on autumn-fallback (because 01:00-02:00 is lived twice
    # without a gap in numbering). Apply a period-dependent shift to
    # reconstruct naive local wall-clock time:
    #
    #   Spring-forward (46 periods): periods >= 3 -> +60 min.
    #   Autumn-fallback (50 periods): periods >= 5 -> -60 min.
    #
    # tz_localize then resolves the one remaining local-time ambiguity
    # (01:00-02:00 wall-clock on autumn-fallback) via a deterministic
    # mask: periods 3-4 are first-occurrence (BST); periods 5-6 are
    # second-occurrence (GMT).
    settlement_date = pd.to_datetime(df["SETTLEMENT_DATE"])
    period = df["SETTLEMENT_PERIOD"].astype("int64")

    # Explicit per-day range check; anything beyond the day's valid count
    # is corrupt upstream data and raises before any shift is applied.
    #   spring-forward day: 1..46
    #   autumn-fallback day: 1..50
    #   normal day: 1..48
    # A ValueError is raised citing the offending (date, period) rows.
    ...

    minutes = (period - 1) * 30
    minutes = minutes.where(~(is_autumn & (period >= 5)), minutes - 60)
    minutes = minutes.where(~(is_spring & (period >= 3)), minutes + 60)
    naive_local = settlement_date + pd.to_timedelta(minutes, unit="m")

    ambiguous = is_autumn & period.isin([3, 4])
    local = naive_local.dt.tz_localize(
        "Europe/London",
        ambiguous=ambiguous.to_numpy(),   # True for autumn periods 3-4 (first, BST);
                                          # False for 5-6 (second, GMT)
        nonexistent="raise",              # retained as a defence; under contiguous
                                          # numbering this should never fire
    )
    df["timestamp_local"] = local
    df["timestamp_utc"] = local.dt.tz_convert("UTC")
    return df
```

`is_spring` / `is_autumn` are computed deterministically from the calendar — the last Sunday of March / October for each year seen in the frame — not inferred from data. The autumn-fallback `ambiguous` mask is therefore `True` for periods 3–4 (first occurrence, BST) and `False` for periods ≥ 5 (second occurrence, GMT). `nonexistent="raise"` is retained as a structural defence against a future upstream change; under the contiguous-numbering reality the shift formula keeps periods off the vanished 01:00–02:00 local hour, so it is expected never to fire.

The per-day range check raises `ValueError` on any row whose period exceeds the day's valid count (`> 46` on spring-forward days, `> 50` on autumn-fallback days, `> 48` on normal days), naming the offending `(SETTLEMENT_DATE, SETTLEMENT_PERIOD)` pairs. That replaces the pre-implementation assumption that spring-forward days simply "have no period 3 or 4 in valid NESO data".

Unit test fixture `tests/fixtures/neso/clock_change_rows.csv` contains:
- Spring-forward 2024 (31 March), periods 1, 2, 3, 46, 47 — the trailing period 47 exercises the per-day range check (spring-forward days are capped at 46) and must raise.
- Autumn-fallback 2024 (27 October), periods 1–7 — periods 3 and 4 resolve to the first (BST) 01:00 / 01:30 occurrences; periods 5 and 6 resolve to the second (GMT) 01:00 / 01:30 occurrences, producing distinct UTC timestamps one hour apart.

## 7. Cache semantics

- **Path:** `<cache_dir>/neso_demand.parquet`. `<cache_dir>` resolves to `${BRISTOL_ML_CACHE_DIR:-./data/raw/neso}` via Hydra interpolation.
- **`CachePolicy.AUTO`** (default): if file exists, return its path. Never re-hash, never check staleness. Staleness is a Stage 19 concern, not Stage 1's.
- **`CachePolicy.REFRESH`**: fetch all configured years, overwrite atomically.
- **`CachePolicy.OFFLINE`**: if file exists, return its path; else raise `CacheMissingError` naming the expected path and the override to re-fetch.
- **Idempotence:** two back-to-back `REFRESH` calls produce byte-identical parquet if upstream is unchanged (modulo `retrieved_at_utc`, which is monotonically increasing by design — it is provenance, not payload).

The acceptance criterion "running the ingestion twice in a row produces the same on-disk result" is satisfied with a narrower reading: two `AUTO` calls are no-ops; two `REFRESH` calls produce the same row-set and schema, even if `retrieved_at_utc` differs.

## 8. Retry policy

Exactly one tenacity wrapper, applied to `_retrying_get`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(config.max_attempts),
    wait=wait_exponential(multiplier=config.backoff_base_seconds,
                          max=config.backoff_cap_seconds),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout, _RetryableStatusError)),
    reraise=True,
)
def _retrying_get(url, params): ...
```

`_RetryableStatusError` is raised by a response-check helper for 5xx and 429 only. On final failure, the exception message contains the URL, attempt count, and the last-seen status; this is what a facilitator sees on a flaky demo.

Client-side throttling: NESO advertises 2 req/min on the datastore. The loader inserts a minimum inter-request delay of `max(0, 30 - elapsed)` seconds between paginated calls. Configurable; default conservative.

## 9. Fixture strategy

- **Library:** `pytest-recording` (vcrpy under the hood).
- **Cassette location:** `tests/fixtures/neso/cassettes/`.
- **Recording mode:** `--record-mode=none` in CI (enforced in `pyproject.toml` pytest config); developers re-record with `--record-mode=once` against a fresh cassette directory.
- **Scope:** one year (2023), paginated into two pages at a small `page_size` so the paginator is exercised. Full years would inflate the repo; vcrpy response-body truncation is used to cap cassette size at ~200 kB.
- **Filtering:** no auth headers today; the filter list is nevertheless set up now to prevent a future secret leaking through a re-record. Query-string filters strip cache-busting params if any appear.
- **Hand-crafted fixtures:** `clock_change_rows.csv` for DST unit tests (§6) — easier to reason about than a cassette for this case.

## 10. Tests

| Test                                              | Level       | Asserts                                                                    |
|---------------------------------------------------|-------------|----------------------------------------------------------------------------|
| `test_to_utc_spring_forward`                      | Unit        | `nonexistent="raise"` fires if period 3 or 4 appears on 31 Mar 2024.       |
| `test_to_utc_autumn_fallback`                     | Unit        | Periods 3 and 4 on 27 Oct 2024 produce distinct UTC timestamps one hr apart.|
| `test_assert_schema_missing_required_raises`      | Unit        | Missing `ND` or `SETTLEMENT_PERIOD` raises a named error.                  |
| `test_assert_schema_unknown_column_warns_and_drops` | Unit      | Unknown column produces a warning; column is not persisted.                |
| `test_fetch_offline_raises_when_cache_missing`    | Unit        | `OFFLINE` + no cache → `CacheMissingError` names the expected path.        |
| `test_fetch_auto_returns_cached_path_no_network`  | Unit        | `AUTO` + cache present performs zero HTTP calls (assert via cassette off). |
| `test_fetch_refresh_end_to_end`                   | Integration | `REFRESH` + cassette → parquet written; schema matches §4; rows sorted.   |
| `test_fetch_idempotent`                           | Integration | Two consecutive `REFRESH` runs produce identical rows (modulo provenance). |
| `test_cli_help`                                   | Smoke       | `python -m bristol_ml.ingestion.neso --help` exits 0.                      |

Acceptance-criteria trace from the stage intent:

| Intent criterion                                                           | Covered by                                        |
|----------------------------------------------------------------------------|---------------------------------------------------|
| 1. Cache present → completes offline                                       | `test_fetch_auto_returns_cached_path_no_network`  |
| 2. No cache → fetches and persists                                         | `test_fetch_refresh_end_to_end`                   |
| 3. Two runs produce the same result                                        | `test_fetch_idempotent`                           |
| 4. Schema documented in module's `CLAUDE.md`                               | Reviewed on PR; not a runtime assertion.          |
| 5. Notebook runs top-to-bottom quickly                                     | Manual; see §11.                                  |
| 6. Tests exercise the public interface using recorded fixtures             | Integration suite above.                          |

## 11. Notebook

`notebooks/01_neso_demand.ipynb` is thin (principle §2.1.8):

1. `from bristol_ml import load_config; from bristol_ml.ingestion import neso`
2. `cfg = load_config(overrides=["ingestion=neso"])`
3. `path = neso.fetch(cfg.ingestion.neso, cache=CachePolicy.AUTO)`
4. `df = neso.load(path)`
5. Plot a representative week of hourly `nd_mw` (resample from half-hourly with `.resample("H").mean()`).
6. Plot a year of daily peaks (`.groupby(pd.Grouper(freq="D"))["nd_mw"].max()`).
7. Prose note on embedded solar/wind (intent §"Points for consideration") for a facilitator to riff on.

Target runtime: under 30 seconds when the cache is warm.

## 12. Risks, open questions, deferred items

- **Hardcoded year→UUID list goes stale.** When NESO adds 2026, someone must update `conf/ingestion/neso.yaml`. A discovery mode (call `package_show`, auto-fill) is defensible but out of scope — the research note flags it; adding it now is complexity not yet earned (principle §2.2.4).
- **Schema drift across years.** Earlier resources lack the post-2023 interconnector columns. The warn-and-drop rule (layer architecture, "Schema assertion at ingest") handles it, but a NESO-side renaming of `ND` would be a hard failure. Acceptable; a loud break is the correct behaviour.
- **Half-hourly vs hourly at the storage boundary.** Stored at half-hourly (matches source); hourly is a features-layer derivation (Stage 3). The notebook resamples for display.
- **Population of `retrieved_at_utc`.** Per-fetch, not per-row, to keep byte-equal idempotence achievable within a single run.
- **Atomic-write on Windows.** `os.replace` is atomic on POSIX and on NTFS; not atomic on FAT32. Out of scope; facilitators will not be demoing on FAT32.
- **Licence metadata.** OGL v3 is not carried into the parquet file. A `LICENSE` note in the module's `CLAUDE.md` is sufficient at this stage.
- **Cassette size budget.** Target ≤200 kB. If the vcrpy filter settings do not achieve this, fall back to a hand-written JSON fixture and a simpler `responses`-style mock for the paginator tests.
- **Deferred to Stage 4.** The day-ahead forecast archive lives at a different resource UUID but reuses this module's scaffolding verbatim. The design here is shaped to make that a copy-and-adapt, not a refactor.

## 13. Traceability

- Intent → **Stage 1 intent** (`docs/intent/01-neso-demand-ingestion.md`).
- Layer architecture → **Ingestion layer** (`docs/architecture/layers/ingestion.md`).
- Facts → **Research note** (`docs/lld/research/01-neso-ingestion.md`).
- Principles → **DESIGN.md** §2.1.1, §2.1.2, §2.1.3, §2.1.5, §2.1.6, §2.1.7, §2.1.8, §7.1–§7.5.
