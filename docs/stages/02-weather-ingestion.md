# Stage 2 — Weather ingestion + joined plot — work brief

- **Status:** `ready`
- **Intent (authoritative):** [`docs/intent/02-weather-ingestion.md`](../intent/02-weather-ingestion.md) — immutable once stage is shipped.
- **Depends on:** Stage 0 (shipped), Stage 1 (shipped — Stage 2 extends the ingestion pattern and extracts `_common.py` from it).
- **Enables:** Stage 3 (feature assembler joins weather to demand), and every modelling stage that uses weather as an exogenous driver.

## One-sentence framing

Fetch hourly Open-Meteo archive weather for ten UK population centres, compose per-station data into a population-weighted national signal, and plot national temperature against national demand — proving the ingestion pattern generalises and landing the first analytical insight.

## Pre-implementation: three spec-drifts to resolve

The LLD §0 surfaces three divergences between the intent/spec and the research. Resolve before implementation begins:

1. **DESIGN §4.2 — data model.** Spec says "~10 km via UKMO UKV 2 km model". Open-Meteo's archive is ERA5 / ERA5-Land / CERRA at ~9–11 km; UKV 2 km is only available via the separate `historical-forecast-api` and only from 2022-03-01 (incompatible with the 2018 training window). Corrective edit belongs in DESIGN §4.2 (main session, human approval).
2. **Demo "V-shape"** is historically a hockey-stick on GB data — flat-to-noisy warm arm until post-2020. Notebook commentary frames accordingly; intent text stays immutable.
3. **Population-weighted national temperature** is a defensible pedagogical default, not an industry-verified standard. GB's documented national weather variable (National Gas CWV) is gas-demand weighted; Thornton et al. (2016) use CET unweighted. Notebook acknowledges this.

None of these blocks the implementation; they need surfacing so the implementer does not over-claim.

## Reading order

Read these before opening any code. Each entry is one sentence on what the document contributes.

1. [`docs/intent/02-weather-ingestion.md`](../intent/02-weather-ingestion.md) — **what and why**: in / out of scope, demo moment, six acceptance criteria, points for consideration (station selection, weighting rationale, variables, DST, rate limits, caching stance).
2. [`docs/intent/DESIGN.md`](../intent/DESIGN.md) §2.1 (principles), §3.2 (ingestion + features layer split), **§4.2 flagged for correction** (see "Pre-implementation" above), §7 (Hydra + Pydantic config). Skim §9 only for ordering context.
3. [`docs/architecture/layers/ingestion.md`](../architecture/layers/ingestion.md) — **the contract this stage instantiates**: `fetch`/`load`/`CachePolicy`, parquet + UTC + atomic-write, schema assertion, retries, fixtures. Note the revised "multi-endpoint sources" storage convention and the `_common.py` extraction trigger (both updated in response to Stage 2's shape).
4. [`docs/lld/research/02-weather-ingestion.md`](../lld/research/02-weather-ingestion.md) — **empirical facts**: Open-Meteo endpoint/params, rate-limit accountant, variable spellings, underlying data models (ERA5/ERA5-Land/CERRA — **not** UKV for the archive), UK station coordinates and 2011 populations, timezone handling, caching granularity trade-offs, `openmeteo-requests` vs direct httpx, pandas weighted-mean idioms, hockey-stick literature.
5. [`docs/lld/ingestion/weather.md`](../lld/ingestion/weather.md) — **first-pass design**: public interface, Pydantic schemas, output parquet schema (long-form: `timestamp_utc`, `station`, 5 variables, provenance), data flow, `features/weather.py` aggregator, `_common.py` extraction plan, fixture scope, full test list with acceptance-criteria trace, notebook outline, risks.
6. [`docs/lld/ingestion/neso.md`](../lld/ingestion/neso.md) — **reference for the existing template**. Stage 2 copies the public shape and extracts shared helpers (atomic write, retry, rate-limit). Read to understand what's being generalised.
7. [`src/bristol_ml/ingestion/neso.py`](../../src/bristol_ml/ingestion/neso.py) and [`src/bristol_ml/ingestion/CLAUDE.md`](../../src/bristol_ml/ingestion/CLAUDE.md) — **shipped Stage 1 code** as the concrete template. The `_common.py` extraction operates on named symbols here.
8. [`CLAUDE.md`](../../CLAUDE.md) — module boundaries, coding conventions, quality gates, team conventions, stage hygiene.
9. [`.claude/playbook/git-protocol.md`](../../.claude/playbook/git-protocol.md) — branch + attempt + commit conventions.

## Acceptance criteria

Quoted verbatim from the intent. Intent wins on drift.

1. Running the ingestion with a cache present completes offline.
2. Running the ingestion without a cache fetches all configured stations.
3. The national aggregation accepts any subset of the configured station list, so a demo can run with fewer stations to show the effect.
4. The notebook runs top-to-bottom quickly on a laptop.
5. The notebook's commentary motivates the choice of Open-Meteo over Met Office DataHub briefly.
6. Smoke test for the fetcher; a test for the aggregation that asserts equal weights on identical inputs yield the identity.

Per-criterion test mapping: LLD §10 acceptance-criteria trace.

## Files expected to change

**New:**
- `src/bristol_ml/ingestion/_common.py` — shared helpers extracted from `neso.py` (`CachePolicy`, `CacheMissingError`, `_atomic_write`, `_retrying_get`, `_RetryableStatusError`, `_respect_rate_limit`).
- `src/bristol_ml/ingestion/weather.py` — `fetch`, `load`, CLI; imports from `_common`.
- `src/bristol_ml/features/__init__.py` — package marker; first introduction of `features/`.
- `src/bristol_ml/features/weather.py` — `national_aggregate` (weighted mean with NaN-safe handling and station-subset support).
- `conf/ingestion/weather.yaml` — Hydra group file with ten stations, variables, date range, retry knobs.
- `tests/unit/ingestion/test_weather.py`, `tests/unit/features/__init__.py`, `tests/unit/features/test_weather_aggregate.py`.
- `tests/integration/ingestion/test_weather_cassettes.py`.
- `tests/fixtures/weather/cassettes/*.yaml`, `tests/fixtures/weather/station_subset.csv`.
- `notebooks/02_weather_joined_plot.ipynb`.
- `docs/lld/stages/02-weather-ingestion.md` — retrospective at ship.

**Modified:**
- `src/bristol_ml/ingestion/neso.py` — imports `CachePolicy`, `CacheMissingError`, `_atomic_write`, `_retrying_get`, `_RetryableStatusError`, `_respect_rate_limit` from `_common`. Pure refactor; existing tests continue to pass unchanged.
- `src/bristol_ml/ingestion/CLAUDE.md` — adds `weather.py` output-schema table alongside `neso.py`'s.
- `conf/_schemas.py` — adds `WeatherStation`, `WeatherIngestionConfig`; extends `IngestionGroup` with optional `weather`.
- `conf/config.yaml` — adds `ingestion/weather@ingestion.weather` to the defaults list.
- `pyproject.toml` — runtime dep additions: `statsmodels` (notebook LOWESS), possibly nothing else — `httpx`, `tenacity`, `pyarrow`, `pandas`, `loguru` are already in from Stage 1.
- `README.md` — entry point for the new notebook.
- `CHANGELOG.md` — `### Added` bullet under `[Unreleased]` naming `weather.py`, `features/weather.py`, and `_common.py` extraction.
- `docs/intent/DESIGN.md` §4.2 — **corrective edit** on the "UKV 2 km" claim. Mechanical factual correction; main-session, human approval.
- `docs/intent/DESIGN.md` §6 — add `features/` and `ingestion/weather.py` and `ingestion/_common.py` to the layout tree.
- `docs/stages/README.md` — Stage 2 `ready` → `in-progress` then → `shipped`.

## Exit criteria

PR checklist — derived from CLAUDE.md "Stage hygiene" and DESIGN.md §9.

- [ ] All tests pass locally (`uv run pytest`) and CI green on the PR.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `src/bristol_ml/ingestion/CLAUDE.md` documents the `weather.py` output parquet schema.
- [ ] NESO tests continue to pass without modification after the `_common.py` extraction (pure refactor).
- [ ] Retrospective filed at `docs/lld/stages/02-weather-ingestion.md`.
- [ ] `CHANGELOG.md` has an `### Added` bullet under `[Unreleased]`.
- [ ] `README.md` references `notebooks/02_weather_joined_plot.ipynb` as a new entry point.
- [ ] `docs/intent/DESIGN.md` §4.2 corrective edit merged (data model: ERA5/ERA5-Land/CERRA at ~11 km, not UKV 2 km).
- [ ] `docs/intent/DESIGN.md` §6 layout tree updated.
- [ ] `docs/stages/README.md` row updated to `shipped` with retrospective link.
- [ ] No `xfail` tests; no skipped tests without a linked issue.

## Team-shape recommendation

Default team (lead + implementer + tester + docs). Specifics:

- **Researcher** — already complete. [`docs/lld/research/02-weather-ingestion.md`](../lld/research/02-weather-ingestion.md) covers the ground. Fresh researcher only if Open-Meteo's schema or model availability changes unexpectedly.
- **Implementer** — two logical commits on the task branch: (1) pure-refactor extraction of `_common.py` from `neso.py`; (2) `weather.py`, `features/weather.py`, config, tests, notebook. Tester confirms the refactor is pure before the implementer proceeds.
- **Tester** — spawn **in parallel** with the implementer, not after. Tester writes spec-derived tests for acceptance criteria and aggregator invariants (identity under equal weights, subset behaviour, NaN handling). Tester cannot modify production code.
- **Docs** — spawned after implementation stabilises to update `src/bristol_ml/ingestion/CLAUDE.md`, the notebook prose, `README.md`, and the DESIGN §4.2 correction. DESIGN edits are deny-tier for the lead — route through the main session.
- **Escalation** — lead manages retries directly; escalate to human after two failed attempts. The three pre-flagged spec drifts (see top of this brief) are points where the lead may ask the human to confirm framing before proceeding.

## Notes

- **This stage hits the `_common.py` trigger.** The extraction is not optional extra work — it is the right thing to do now that there are two concrete callers. The layer architecture's "Open questions" has been updated to reflect this.
- **Concurrency deferred.** Ten serial station fetches with httpx is ~5-15 seconds total. `httpx.AsyncClient` is the obvious upgrade if a live demo starts to feel the latency; deferred to when measured, not speculated.
- **`features/` arrives in Stage 2**, one stage earlier than DESIGN §9 implies. This is explicit: the stage needs a weighted-mean function and the notebook cannot reimplement it (§2.1.8). The features layer's proper architecture doc lands at Stage 3 when the assembler earns it.
- **Stage 4 inherits.** `neso_forecast.py` at Stage 4 should be shorter than `neso.py` was thanks to `_common.py` — that's the test of the extraction being the right shape.
