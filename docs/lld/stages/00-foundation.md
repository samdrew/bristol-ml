# Stage 0 — Foundation: retrospective

## Goal

Land the template scaffold and a worked-example service that
demonstrates the layer pattern, the config wiring, and the test
layout end-to-end.  Sources at
[`docs/intent/00-foundation.md`](../../intent/00-foundation.md);
plan at
[`docs/plans/completed/00-foundation.md`](../../plans/completed/00-foundation.md).

## What was built

The template was extracted from `bristol_ml` (a GB
electricity-demand forecaster reference architecture) by stripping
the domain content and carrying the infrastructure forward across
six commits:

- **C1 (strip)** — deleted 8 bristol_ml module subtrees, all
  notebooks + data fixtures, all 16 completed plans + 16 stage
  retros + 40+ research docs, all bristol_ml-specific tests, all
  bristol_ml-specific Hydra YAML groups, ADRs 0002-0008, the
  per-stage intent docs.  Removed `uv.lock`.
- **C2 (rename)** — `git mv src/bristol_ml -> src/TEMPLATE_PROJECT`,
  swapped imports, slimmed `pyproject.toml` (drop ~13 heavy domain
  deps and the PyTorch CUDA index), slimmed `_schemas.py` from
  1267 LOC to ~200 LOC (6 generic schemas + 1 toy schema), reduced
  `Dockerfile` to `python:3.12-slim`, dropped the
  `HF_HUB_OFFLINE` block from `tests/conftest.py`.
- **C3 (worked example)** — added `core/text_stats.py` (pure),
  `services/text_stats_service.py` (Hydra wrapper), the
  `services/text_stats.yaml` Hydra group, 14 tests (7 unit + 4
  integration + 3 config-smoke), the `sample_text.txt` fixture.
- **C4 (.claude)** — turned out to be a one-line edit:
  `.claude/agents/minimalist.md` had a "since Stage 4" reference;
  generalised.  Everything else under `.claude/` carries verbatim.
- **C5 (top-level docs)** — rewrote `CLAUDE.md`, `README.md`,
  `CHANGELOG.md`, added `TEMPLATE_USAGE.md`.
- **C6 (doc tier)** — wrote `docs/intent/{DESIGN.md, 00-foundation.md}`,
  `docs/plans/completed/00-foundation.md`, this file,
  `docs/architecture/README.md`, `docs/architecture/layers/{core,services}.md`,
  updated ADR 0001's one bristol_ml reference.

Total scope: 14 unit + integration tests pass, lint clean, CLI
boots, default config validates.  Down from bristol_ml's ~900-LOC
`uv.lock` package count to a slim runtime-deps-only lock.

## Design choices

**D1 (template branch).**  Used the existing `template` branch on
the bristol_ml repo rather than a sibling repo or a subdirectory.
`git diff main..template` shows the extraction explicitly — useful
for sanity checks during template maintenance.

**D2 (worked example first).**  Shipped the toy text-stats service
alongside the scaffold rather than an empty-scaffold variant.  The
worked example exercises every load-bearing piece of the template
(layer separation, config wiring, CLI, unit + integration tests,
all four doc tiers) without dragging in domain-specific dependencies.
The empty-scaffold variant becomes a small follow-up branch off
`template`.

**D3 (full pipeline methodology).**  Kept the bristol_ml
methodology (four-tier docs, 13-agent roster, path-tier write
hooks) intact.  Concrete projects can lighten as needed; the
template ships the strong opinion.

**D4 (renameable layer stubs).**  Shipped `core/` + `services/`
rather than empty ML-pipeline directories (`ingestion/`,
`features/`, `models/`, `evaluation/`) or no layer dirs at all.
`core/` + `services/` reads as a generic
"pure-logic + IO-wrapper" split that fits a wide range of project
domains.  Concrete projects rename or replace.

**D5 (literal `TEMPLATE_PROJECT` placeholder).**  Used the literal
string instead of mustache `{{PROJECT_NAME}}` brackets.  Trade-off:
mustache makes the placeholder unambiguous but breaks Python
imports / pipeline tooling that processes the template before
substitution; the literal is executable as-is and the user
`sed`-substitutes on instantiation per `TEMPLATE_USAGE.md`.

## Demo moment

```bash
$ uv run python -m TEMPLATE_PROJECT.services.text_stats_service
{
  "character_count": 124,
  "non_whitespace_character_count": 100,
  "word_count": 24,
  "line_count": 3
}
```

The output is reproducible from the shipped fixture
`tests/fixtures/sample_text.txt`.  Three pangrams; the figure
above is what the service prints with the default config.

## Deferred

- **Empty-scaffold variant.**  Still pending; a follow-up branch
  off `template` that strips the worked example and replaces with
  `.gitkeep` stubs.  Plan: do it after a few real instantiations
  of the worked-example template surface any ergonomic gaps.
- **GitHub Actions CI workflow.**  The template doesn't ship a
  `.github/workflows/ci.yml`.  bristol_ml has one but it's
  bristol_ml-specific.  A generic CI yaml would test:
  `uv sync --group dev`, `uv run pytest`, `uv run ruff check`,
  `uv run ruff format --check`.  Worth adding before the next
  real instantiation.
- **Cookiecutter / copier wrapper.**  The current template is
  `git mv` + `sed`.  A Cookiecutter wrapper would let users
  parameterise more (project name, author, license, layer names)
  without manual substitution.  Defer until enough real
  instantiations to know which parameters are worth lifting.

## Scope-diff observations

The Phase 1 explore agents over-classified the bristol_ml content.
Three notable cases:

1. **`.claude/` was claimed to need 8 agents updated**; in practice
   only 1 line in 1 agent (`minimalist.md`) had a bristol-specific
   reference.  Lesson: ask explore agents to cite grep evidence,
   not summary classifications.
2. **The Phase-1 inventory agent claimed `_parametric_fn`-style
   bristol-specific files in `tests/conftest.py`**; in practice
   the conftest had only an `HF_HUB_OFFLINE` block.  Same lesson.
3. **`Dockerfile` was in `.git/info/exclude`** (locally untracked);
   needed `git add -f` to commit on the template branch.

## Next

The template is ready for instantiation.  Two follow-up items:

- **Empty-scaffold variant** (separate branch off `template`).
- **First real instantiation** — pick a small Python project the
  user wants to start, run `TEMPLATE_USAGE.md` end-to-end, and
  feed back any rough edges into the template.

The methodology has been exercised end-to-end on this stage; the
agent roster, the doc tiers, the path hooks, the lint + test +
format gates all work.  Stage 0's job is done.
