# Stage 0 — Foundation: implementation plan

**Status.**  Completed; archived from `docs/plans/active/` after
shipping.

## Preamble

### Intent

- [`docs/intent/00-foundation.md`](../../intent/00-foundation.md)

### Sources

The template was extracted from `bristol_ml`
(commit `9f04e10` at branch-out time) — a reference architecture
for a GB electricity-demand forecaster.  The bristol_ml domain
content was stripped; the infrastructure (Hydra+Pydantic wiring,
.claude/ agent roster, four-tier doc methodology, test layout,
pyproject.toml, lint/format/test config, pre-commit, Dockerfile)
was carried forward.

## Decisions

D1.  **Branch `template` on the bristol_ml repo**, not a sibling
     repo or a subdir.  Lets `git diff main..template` show the
     extraction explicitly.

D2.  **Worked example first.**  Ship the toy text-statistics
     service alongside the scaffold so a reader sees the layer
     pattern, the config wiring, and the test layout in action.
     The empty-scaffold variant lives on its own branch (a small
     diff off `template`).

D3.  **Full pipeline methodology.**  Keep the four-tier doc system
     (intent / architecture / plans / lld), the 13-agent roster,
     the path-tier write hooks.  Concrete projects can lighten as
     needed; the template ships with the strong opinion.

D4.  **Renameable layer stubs (`core/` + `services/`).**  Rather
     than ship empty `ingestion/features/models/evaluation/` (too
     ML-pipeline-specific) or no layer dirs at all (no hint at the
     layer pattern), ship two generically-named stubs that
     demonstrate the pure-logic-vs-IO-wrapper split.

D5.  **Project-name placeholder is the literal string
     `TEMPLATE_PROJECT`** (not `{{PROJECT_NAME}}`-style mustache
     brackets).  The literal is importable and executable as-is;
     mustache brackets break Python imports.  Users `git mv` and
     `sed` on instantiation per `TEMPLATE_USAGE.md`.

## Tasks

T1.  **Strip bristol_ml content.**  Delete the eight bristol_ml
     module subtrees, all notebooks, all data fixtures, all 16
     completed plans + 16 stage retros + 40+ research docs, all
     bristol_ml-specific tests, all bristol_ml-specific Hydra YAML
     groups, ADRs 0002-0008, the per-stage intent docs.  Keep the
     boilerplate (`__init__/__main__/cli/config/py.typed`),
     `tests/conftest.py`, `.claude/` (almost entirely generic),
     ADR 0001, and the empty `data/` + `docs/{plans,lld}/`
     directory structure.

T2.  **Rename + slim boilerplate.**
     - `git mv src/bristol_ml -> src/TEMPLATE_PROJECT`.
     - Strip `__init__.py`, slim `_schemas.py` to 6 generic
       schemas + the toy `TextStatsConfig`, write a minimal
       `conf/config.yaml`, slim `pyproject.toml` (drop heavy
       deps, drop the PyTorch CUDA index), reduce `Dockerfile`
       to `python:3.12-slim`, drop the `HF_HUB_OFFLINE` block from
       `tests/conftest.py`.

T3.  **Toy worked example.**
     - `core/text_stats.py` (pure function) + `core/CLAUDE.md`.
     - `services/text_stats_service.py` (Hydra wrapper) +
       `services/CLAUDE.md`.
     - `conf/services/text_stats.yaml`.
     - 7 unit tests on `compute_text_statistics`, 4 integration
       tests on the service, 3 smoke tests on the config wiring.
     - `tests/fixtures/sample_text.txt`.

T4.  **Carry `.claude/` with substitutions.**  Phase-1 explore
     showed only one bristol-specific reference (a "Stage 4"
     example in the minimalist agent's prompt).  Generalised that
     line; everything else carries verbatim.

T5.  **Rewrite top-level docs.**  New `CLAUDE.md` (project-
     agnostic routing), new `README.md` (template-instantiation
     pitch), new `CHANGELOG.md` (skeleton with the Stage 0 entry),
     new `TEMPLATE_USAGE.md` (one-page how-to-instantiate guide).

T6.  **Doc tier for the worked example.**  This file
     (`docs/plans/completed/00-foundation.md`),
     `docs/intent/00-foundation.md`,
     `docs/lld/stages/00-foundation.md`,
     `docs/architecture/{README.md, layers/core.md, layers/services.md}`,
     `docs/intent/DESIGN.md` template.

## Acceptance criteria

Mirrors the intent's AC list:

- AC-1: `uv sync --group dev` succeeds on a clean checkout.
- AC-2: `uv run python -m TEMPLATE_PROJECT --help` works; bare
  invocation prints AppConfig as JSON.
- AC-3: text-stats service end-to-end reads the fixture, prints
  valid JSON, exits 0.
- AC-4: 14 tests pass.
- AC-5: ruff lint + format clean.

## Out of scope (per intent)

- Empty-scaffold variant.
- Richer worked example.
- Domain-specific code.

## Exit checklist

- [x] `uv sync --group dev` clean.
- [x] `uv run pytest` — 14/14 passing.
- [x] `uv run python -m TEMPLATE_PROJECT --help` works.
- [x] `uv run ruff check . && uv run ruff format --check .` clean.
- [x] All 6 Cs (C1 strip → C6 doc tier) committed on `template`.
- [x] Plan archived to `docs/plans/completed/00-foundation.md`.
