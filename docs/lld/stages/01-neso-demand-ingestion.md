# Stage 1 — NESO demand ingestion

## Goal

Bring real GB electricity demand data into the project and produce the first plot. Establish the ingestion-layer template every subsequent feed will copy (seven modules across the stage plan — see [`docs/architecture/layers/ingestion.md`](../../architecture/layers/ingestion.md)).

## What was built

- `src/bristol_ml/ingestion/neso.py` — the module. Public surface: `fetch(config, *, cache=CachePolicy.AUTO) -> Path`, `load(path) -> pd.DataFrame`, the `CachePolicy` StrEnum (`AUTO | REFRESH | OFFLINE`), and `CacheMissingError`. Re-exported lazily from `bristol_ml.ingestion` and `bristol_ml` (see "Design choices" below).
- `conf/_schemas.py` — new Pydantic types `NesoYearResource`, `NesoIngestionConfig`, `IngestionGroup`, plus an `AppConfig.ingestion: IngestionGroup` field (defaults to an empty group so Stage 0 configs still validate).
- `conf/ingestion/neso.yaml` — full 2018–2025 year → CKAN-UUID catalogue, resolved one-off from `action/package_show?id=historic-demand-data` on 2026-04-18. `cache_dir` defaults to `${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/neso}`.
- Module CLI `python -m bristol_ml.ingestion.neso` with `--cache {auto,refresh,offline}` and Hydra overrides; prints the resulting cache path.
- `notebooks/01_neso_demand.ipynb` — thin notebook per §2.1.8; loads the cached parquet, plots a week of hourly `nd_mw`, plots daily peaks across the cached window, and carries a short prose caveat about embedded solar/wind.
- `src/bristol_ml/ingestion/CLAUDE.md` — module-local guide with the output parquet schema table (acceptance criterion 4).
- Test suite (35 tests, all green):
  - 15 implementer-derived unit tests in `tests/unit/ingestion/test_neso.py` covering helpers, dtype handling, rate-limit sleep, and CLI smoke.
  - 16 spec-derived unit tests (same file, distinct classes) covering the public interface, DST boundaries, schema-assertion contract, and CLI `--help`.
  - 5 integration tests in `tests/integration/ingestion/test_neso_cassettes.py` exercising `REFRESH` / `AUTO` / `OFFLINE` via cassette replay.
  - 3 pre-existing Stage 0 tests in `tests/unit/test_config.py`, untouched.
- Recorded cassette `tests/fixtures/neso/cassettes/neso_2023_refresh.yaml` (~94.5 kB, 2 paginator pages of 500 rows each), plus the one-off recorder at `scripts/record_neso_cassette.py` (outside the `bristol_ml` package so it is excluded from the wheel build).

## Design choices made here

- **DST numbering — accepted spec-drift from the LLD.** The pre-implementation LLD §6 and the research note described gap-skipping period numbering on autumn-fallback days (e.g. "period 3 = first 01:00 BST, period 4 = second 01:00 GMT"). Real Elexon / NESO numbering is contiguous 1..46 on spring-forward Sundays and 1..50 on autumn-fallback Sundays. `_to_utc` applies a period-dependent shift — `+60 min` on spring-forward for periods ≥ 3, `−60 min` on autumn-fallback for periods ≥ 5 — to map contiguous numbering back to naive local time, and raises explicitly on any row whose period exceeds its day's valid range (`> 46` on spring-forward, `> 50` on autumn-fallback, `> 48` otherwise). The autumn-fallback `ambiguous` mask is now `True` for periods 3-4 (first occurrence, BST) and `False` for periods ≥ 5 (second occurrence, GMT). Tests were rewritten against the real numbering. **Follow-up:** the next revision of [`docs/lld/ingestion/neso.md`](../ingestion/neso.md) §6 should be updated (it's ALLOW-tier, mutable); the research note reflects the state at the time of research and should be left as historical record.
- **Column-case contract.** `_to_utc` preserves source-case `SETTLEMENT_DATE` / `SETTLEMENT_PERIOD` / `ND` / `TSD` alongside the new `timestamp_utc` / `timestamp_local`. The rename to the lowercase final schema (`settlement_date`, `settlement_period`, `nd_mw`, `tsd_mw`) happens once in `_to_arrow`, the storage boundary. Keeps the conversion layer referring to the NESO-native names, so exceptions cite what the upstream response actually called the column.
- **`_fetch_year` signature.** `_fetch_year(year, resource_id, config, *, client=None)` — the three positional arguments match the LLD §5 contract; `client` is keyword-only. When omitted a short-lived `httpx.Client` is created and torn down inside the call; the outer `fetch` loop passes a shared pooled client to avoid per-year connection setup.
- **Lazy re-exports.** `bristol_ml.ingestion.__init__` and `bristol_ml.__init__` re-export `neso`, `CachePolicy`, and `CacheMissingError` via module-level `__getattr__`. Avoids pulling pandas / pyarrow into every Hydra CLI invocation (which only needs the config path) and suppresses the `RuntimeWarning` `runpy` otherwise emits on `python -m bristol_ml`.
- **`cache_dir` interpolation.** The YAML uses `${oc.env:BRISTOL_ML_CACHE_DIR,data/raw/neso}` rather than `${hydra:runtime.cwd}/data/raw/neso`. The `hydra:` resolver only resolves inside `@hydra.main` — programmatic `load_config` (used by the notebook and tests) raises when asked to resolve it. The notebook performs `os.chdir(REPO_ROOT)` before calling `load_config` so the relative path resolves consistently with the module CLI.
- **Cassette filtering.** A vcrpy `before_record_response` hook in `scripts/record_neso_cassette.py` strips response columns down to `{_id, SETTLEMENT_DATE, SETTLEMENT_PERIOD, ND, TSD}` and rewrites `result.total` to 1000 so replay halts cleanly after two paginator pages of 500 rows each. Keeps the cassette under the 200 kB budget the LLD §9 sets, without losing paginator coverage.
- **No `ingestion/_common.py` extracted at this stage.** The layer architecture's "Open questions" section already calls this out — a shared helper earns its place when Stage 2 arrives with a second concrete caller, not before. Extracting now would fit the first feed and fight the second.
- **`scripts/record_neso_cassette.py` sits outside the package.** Deliberately not under `src/bristol_ml/` — re-recording is a developer action, not production behaviour, and shipping a recorder in the wheel would blur the line. Hatchling's default `src/` layout excludes it automatically.
- **`tenacity` + explicit rate-limit sleep.** Three attempts, exponential backoff base 1 s / cap 10 s, retries only on `httpx.ConnectError`, `httpx.ReadTimeout`, and 5xx / 429. A separate `_respect_rate_limit` helper honours NESO's advertised 2 requests/min by sleeping between paginated calls (default 30 s, configurable). Not a tenacity concern — it is about cadence, not failure.

## Demo moment

From a clean clone (Stage 0 already built):

```
uv sync --group dev
uv run pytest                                                           # 35 passed
uv run python -m bristol_ml.ingestion.neso --help                       # CLI help
uv run python -m bristol_ml.ingestion.neso --cache refresh              # first fetch (long)
uv run python -m bristol_ml.ingestion.neso --cache offline              # subsequent runs, no network
uv run jupyter nbconvert --to notebook --execute notebooks/01_neso_demand.ipynb  # ~8 s on a warm cache
```

The notebook renders a canonical twin-peak weekly profile and a multi-year daily-peak series.

## Deferred

- **Shared `ingestion/_common.py`.** The atomic-write primitive, the tenacity retry wrapper, and the cassette-fixture harness are the obvious candidates. Extraction lands when Stage 2 (weather) introduces the second caller.
- **Discovery mode (LLD §12).** Runtime enumeration of year → UUID via `action/package_show`. Captured in the layer architecture's Open Questions; not yet earned.
- **2026 NESO resource.** The resource UUID is already queryable from `package_show`, but is excluded from `conf/ingestion/neso.yaml` per the 2018–2025 scope in the stage brief. Adding it is a one-line YAML edit when it becomes next-up.
- **Per-year required-column table.** Earlier NESO years lack the post-2023 interconnector columns; today the warn-and-drop rule from the layer architecture handles this. A tighter per-year schema would be worth reconsidering if a stale year ever regresses.
- **Cache portability.** Flagged in the Stage 1 intent as a plausible follow-up (a portable cache archive so one machine can seed another). If it lands, the `CachePolicy` enum is the natural place to extend.
- **`docs/lld/ingestion/neso.md` revision.** LLD §6 still describes the gap-skipping numbering that the implementation diverged from; a clean-up pass should rewrite it against the shipped behaviour. ALLOW-tier so the next stage-1-adjacent session can handle it.

## Next

→ Stage 2 — Weather ingestion (the second concrete ingestion module; drives the extraction of `ingestion/_common.py` and revisits the layer architecture after two callers rather than one).
