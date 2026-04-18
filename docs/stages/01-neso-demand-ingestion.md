# Stage 1 — NESO demand ingestion — work brief

- **Status:** `ready`
- **Intent (authoritative):** [`docs/intent/01-neso-demand-ingestion.md`](../intent/01-neso-demand-ingestion.md) — immutable once stage is shipped.
- **Depends on:** Stage 0 (shipped).
- **Enables:** Stage 3 (feature assembler), Stage 4 (day-ahead forecast archive reuses this module's scaffolding).

## One-sentence framing

Bring real GB electricity demand from the NESO Data Portal into local parquet so a notebook can plot it — the first concrete instance of the ingestion-layer pattern that six later feeds will inherit.

## Reading order

Read these before opening any code. Each entry is one sentence on what the document contributes.

1. [`docs/intent/01-neso-demand-ingestion.md`](../intent/01-neso-demand-ingestion.md) — **what and why**: scope in / out, demo moment, six acceptance criteria, explicit points for consideration (year → UUID mapping, DST periods, ND vs TSD, timestamp zone, caching stance, retry stance).
2. [`docs/intent/DESIGN.md`](../intent/DESIGN.md) sections §2.1 (architectural principles), §3.2 (layer responsibilities — ingestion paragraph), §4.1 (NESO source endpoint and licence), §5.1 (primary target = ND, day-ahead, hourly), §7 (Hydra + Pydantic config pattern). Skim §9 only if the stage-ordering context helps.
3. [`docs/architecture/layers/ingestion.md`](../architecture/layers/ingestion.md) — **the contract this stage instantiates**: `fetch`/`load`/`CachePolicy` public interface, parquet + UTC + atomic-write storage conventions, required-loud / unknown-warn schema assertion rule, tenacity retry defaults, `pytest-recording` fixture library.
4. [`docs/lld/research/01-neso-ingestion.md`](../lld/research/01-neso-ingestion.md) — **empirical facts**: CKAN response shape and pagination, 46/50-period clock-change numbering, pandas `tz_localize` idioms, parquet timestamp types, HTTP-fixture library trade-offs, retry-library trade-offs, ND-vs-TSD benchmark resolution.
5. [`docs/lld/ingestion/neso.md`](../lld/ingestion/neso.md) — **first-pass design**: concrete public interface, Pydantic `NesoIngestionConfig` + YAML shape, output parquet schema table, data-flow sketch, settlement-period → UTC conversion, retry policy, fixture strategy, test list with acceptance-criteria trace, notebook outline, risks and deferred items.
6. [`CLAUDE.md`](../../CLAUDE.md) — project-wide module boundaries, coding conventions, quality gates, team conventions, stage hygiene. Especially the "Stage hygiene" and "Quality gates" sections before the first commit.
7. [`.claude/playbook/git-protocol.md`](../../.claude/playbook/git-protocol.md) — branch naming (`task/<id>-<slug>`), baseline-SHA recording, commit-message template, per-attempt protocol.

Optional at this point (consult when a question arises, not upfront):
- [`docs/intent/DESIGN.md` §9](../intent/DESIGN.md#stage-definition-of-done) — the definition-of-done checklist.
- [`docs/architecture/decisions/0001-use-hydra-plus-pydantic.md`](../architecture/decisions/0001-use-hydra-plus-pydantic.md) — if a config design question becomes load-bearing.

## Acceptance criteria

Quoted verbatim from the intent. Intent wins on any drift — surface the divergence, do not rewrite the intent to match code.

1. Running the ingestion with a cache present completes offline.
2. Running the ingestion without a cache fetches from the NESO CKAN API and writes a local copy for subsequent runs.
3. Running the ingestion twice in a row produces the same on-disk result.
4. The output schema is documented in the module's Claude Code guide.
5. The notebook runs top-to-bottom quickly on a laptop.
6. Tests exercise the public interface of the module using recorded fixtures.

Traceability from each criterion to the test asserting it: see the LLD §10 table.

## Files expected to change

**New:**
- `src/bristol_ml/ingestion/__init__.py` — package marker, re-exports `neso`.
- `src/bristol_ml/ingestion/neso.py` — `fetch`, `load`, `CachePolicy`, private helpers.
- `src/bristol_ml/ingestion/CLAUDE.md` — module-local guide; **output parquet schema documented here** (satisfies acceptance criterion 4).
- `conf/ingestion/neso.yaml` — Hydra group file with year → resource-id catalogue and cache settings.
- `tests/unit/ingestion/__init__.py`, `tests/unit/ingestion/test_neso.py` — pure-Python unit tests (period→UTC, schema assertion, cache policy branches).
- `tests/integration/ingestion/__init__.py`, `tests/integration/ingestion/test_neso_cassettes.py` — end-to-end via `pytest-recording`.
- `tests/fixtures/neso/cassettes/*.yaml` — recorded CKAN responses (narrow slice of 2023).
- `tests/fixtures/neso/clock_change_rows.csv` — hand-crafted spring/autumn rows.
- `notebooks/01_neso_demand.ipynb` — thin notebook: load from cache, aggregate to hourly, plot.
- `docs/lld/stages/01-neso-demand-ingestion.md` — retrospective, filed at ship.

**Modified:**
- `conf/config.yaml` — add `ingestion/neso@ingestion.neso` to the defaults list.
- `conf/_schemas.py` — add `NesoIngestionConfig`, `NesoYearResource`, and an `IngestionGroup` field on `AppConfig`.
- `src/bristol_ml/__init__.py` — re-export as needed for the notebook's import-from-root pattern.
- `pyproject.toml` — runtime deps: `httpx`, `tenacity`, `pyarrow`, `pandas`, `loguru`; dev deps: `pytest-recording`.
- `README.md` — entry point for the new notebook.
- `CHANGELOG.md` — one bullet under `[Unreleased]` / `### Added`.
- `docs/intent/DESIGN.md` §6 — add `ingestion/` under `src/bristol_ml/` and `conf/ingestion/` under `conf/` in the layout tree. Mechanical edit; main-session / human approval.
- `docs/stages/README.md` — flip Stage 1 status cell from `ready` → `in-progress` at start, `in-progress` → `shipped` at ship.

## Exit criteria

PR checklist — derived from `CLAUDE.md` "Stage hygiene" and `DESIGN.md` §9.

- [ ] All tests pass locally (`uv run pytest`) and CI green on the PR.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `src/bristol_ml/ingestion/CLAUDE.md` documents the output parquet schema table.
- [ ] Acceptance-criterion traceability (LLD §10) updated if any test renames occur.
- [ ] Retrospective filed at `docs/lld/stages/01-neso-demand-ingestion.md` following the shape of `00-foundation.md`.
- [ ] `CHANGELOG.md` has an `### Added` bullet under `[Unreleased]` naming the new module and the notebook.
- [ ] `README.md` references the notebook as a new entry point.
- [ ] `docs/intent/DESIGN.md` §6 layout tree updated (mechanical; main-session authority).
- [ ] `docs/stages/README.md` row updated to `shipped` with a link to the retrospective.
- [ ] No `xfail` tests; no skipped tests without a linked issue.
- [ ] Hypothesis enumeration (CLAUDE.md "Quality gates") performed on any debugging work longer than ~15 minutes.

## Team-shape recommendation

Default team (lead + implementer + tester + docs). Specifics:

- **Researcher** — not needed in this round. [`docs/lld/research/01-neso-ingestion.md`](../lld/research/01-neso-ingestion.md) already covers the ground. Spawn a fresh researcher only if the NESO schema drifts in a way the existing note does not cover (e.g. a new required column on a newer year resource).
- **Implementer** — first worktree-isolated attempt against the LLD. The LLD's public interface is the contract; deviations must surface to the lead before the implementer commits to them.
- **Tester** — spawn **in parallel** with the implementer, not after. The tester writes spec-derived tests against the intent's six acceptance criteria and against the LLD's test-list table. The tester cannot modify production code; if a test fails, the tester argues the intent back to the implementer.
- **Docs** — spawned only after implementation stabilises. In this stage, `docs/` updates are scoped to `src/bristol_ml/ingestion/CLAUDE.md`, `README.md`, and the Stage 1 retrospective — `docs/intent/` and `docs/architecture/layers/ingestion.md` are authoritative and not rewritten for Stage 1.
- **Escalation** — lead manages retries directly (default team); escalate to human after two failed attempts. Defensive team's escalation ladder (`.claude/playbook/escalation-ladder.md`) only applies if the defensive shape is in effect.

## Notes

- **The Stage 1 code is the template for Stage 2 (weather) and Stage 4 (forecast archive).** Favour clarity over density; a concise module that reads left-to-right is worth more than one that pre-generalises for seven callers.
- **Do not extract shared helpers (`ingestion/_common.py`) in this stage** — ADR to that effect is embedded in the layer doc under "Open questions". The second caller in Stage 2 is when shared helpers earn their keep.
- **Live-demo posture.** The failure modes a facilitator cares about are: stale cache shown as fresh, silent network fetch at demo time, a cryptic retry message. The `CachePolicy` enum, the "loud retry failure" rule, and the offline-CI default all exist to prevent these.
