# TEMPLATE_PROJECT — Design Document

**Status:** Draft v0.1 — template scaffold.
**Audience:** Project author, future contributors, and Claude Code.
**Scope:** Design intent and implementation plan.  Reflects current
+ near-term reality.  Not a user guide.

> **Template note.**  This file is a template carried over from
> `bristol_ml`.  Replace the content of every section with your
> project's specifics; keep the section headings so the four-tier
> doc methodology and the lead orchestrator's expectations still
> apply.

---

## 1. Project purpose

> _Replace this section with one or two paragraphs naming the
> project's purpose.  If the project has both a pedagogical and an
> analytical purpose (as `bristol_ml` does), separate them and
> declare which wins when they conflict._

The shipped template's worked example is a **text-statistics
service** — a trivial demonstration of the layer pattern, the
config wiring, and the test layout.  Replace this section once your
own project's purpose is settled.

---

## 2. Guiding principles

The numbered principles below are referenced from `CLAUDE.md` (and
from agent prompts) as `§2.1.X`.  Adapt freely; the numbering is
load-bearing only insofar as the agent prompts cite specific
principles.

### 2.1 Architectural

1. **Every component runs standalone.**
   `python -m TEMPLATE_PROJECT.<module>` is the baseline invocation
   pattern.  No component depends on a running service or
   environment state beyond the local filesystem.
2. **Interfaces are typed and narrow.**  Pydantic models for
   configs; typed signatures at module boundaries; documented
   schemas at storage boundaries.
3. **Stub-first for expensive or flaky external dependencies.**
   When you add a network-calling component, ship a stub backed by
   an in-repo fixture and gate the live path on an env-var.
4. **Configuration lives outside code.**  YAML under `conf/`,
   Pydantic schemas in `conf/_schemas.py`.  Code reads configs;
   configs do not contain logic.
5. **Idempotent operations.**  Re-running a job overwrites or
   skips, never corrupts.
6. **Provenance by default.**  Every derived artefact records its
   inputs, git SHA, and wall-clock timestamp.
7. **Tests at boundaries, not everywhere.**  Each module has at
   least a smoke test on its public interface.  Coverage is not a
   goal; behavioural clarity is.
8. **Thin presentation layers.**  If your project ships notebooks,
   GUIs, or HTTP services, they import from `src/TEMPLATE_PROJECT/`
   and do not reimplement logic.

### 2.2 Process

1. **Every stage produces a demonstrable artefact.**
2. **Stages are small.**  Two to four hours of focused work.
3. **Stages are independently interesting.**  Someone picking a
   mid-project stage gets something coherent without internalising
   every preceding stage.
4. **Complexity is earned.**  Advanced patterns land when their
   purpose is concretely motivated, not front-loaded.
5. **The repo explains itself.**  `README.md` pitches;
   `docs/intent/DESIGN.md` (this document) scopes;
   `docs/lld/stages/` records what each stage did and why;
   `docs/architecture/decisions/` records architectural choices.

---

## 3. System architecture

> _Replace this section with a brief description of your project's
> data flow / control flow / module structure.  An ASCII diagram
> works well; the bristol_ml original used one._

### 3.1 Layer responsibilities

The shipped scaffold has two renameable layer stubs:

- **`core/`** — pure, dependency-free domain logic.  No IO, no
  Hydra dependency, trivially unit-testable.
- **`services/`** — Hydra-driven wrappers around `core/`.  Each
  service reads inputs per its config, calls one or more `core/`
  functions, and renders the output.

This split is a starting point, not a constraint.  Rename / replace
the layers as your domain demands.  Common alternatives:

- **ML pipeline:** `ingestion/` → `features/` → `models/` →
  `evaluation/` → `registry/` → `serving/`.
- **HTTP service:** `api/` → `services/` → `repositories/` →
  `db/`.
- **Data analysis:** `loaders/` → `transforms/` → `analyses/` →
  `reports/`.

### 3.2 What is deliberately not in scope (yet)

> _Use this section to surface deferrals — things you've considered
> and decided to defer, with rationale.  Helps future contributors
> avoid relitigating settled questions._

---

## 4. Data sources

> _Replace this section with your project's data sources.  Drop the
> section entirely if your project does not consume external data._

---

## 5. Targets and metrics

> _Replace this section with your project's prediction targets,
> evaluation metrics, and benchmarks (if any).  Drop the section
> entirely if your project is not measurement-driven._

---

## 6. Repository layout

This section reflects the **shipped template scaffold**.  Update it
as you add modules, layers, or notebooks.  The lead orchestrator's
docs-writer agent looks for `### Stage hygiene` / `## §6` updates
when structural changes land.

```
TEMPLATE_PROJECT/
├── src/TEMPLATE_PROJECT/
│   ├── __init__.py          # exposes __version__ and load_config
│   ├── __main__.py          # enables `python -m TEMPLATE_PROJECT`
│   ├── cli.py               # Hydra entry point
│   ├── config.py            # resolve-then-validate glue
│   ├── py.typed
│   ├── core/                # pure domain logic
│   │   └── text_stats.py    # worked example
│   └── services/            # Hydra-driven wrappers
│       └── text_stats_service.py    # worked example
│
├── conf/                    # Hydra YAML + Pydantic schemas
│   ├── __init__.py
│   ├── _schemas.py          # 6 generic schemas + TextStatsConfig
│   ├── config.yaml          # root defaults
│   └── services/text_stats.yaml
│
├── tests/
│   ├── conftest.py          # loguru-bridge fixture
│   ├── unit/
│   │   ├── test_config.py
│   │   └── core/test_text_stats.py
│   ├── integration/
│   │   └── test_text_stats_service.py
│   └── fixtures/sample_text.txt
│
├── docs/
│   ├── intent/DESIGN.md     # this file
│   ├── architecture/
│   │   ├── README.md
│   │   ├── decisions/0001-use-hydra-plus-pydantic.md
│   │   └── layers/{core,services}.md
│   ├── plans/{active,completed}/
│   └── lld/{stages,research,reframings}/
│
├── .claude/                 # agents + hooks + playbook
│
├── CLAUDE.md                # routing for Claude Code
├── README.md                # project pitch
├── TEMPLATE_USAGE.md        # template-instantiation guide
├── CHANGELOG.md
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
└── uv.lock
```

---

## 7. Configuration and extensibility

### 7.1 Framework: Hydra + Pydantic

Hydra composes YAML; Pydantic validates the resolved tree at the
CLI boundary.  Downstream code only ever sees the Pydantic model —
never raw `DictConfig`.  See
[ADR 0001](../architecture/decisions/0001-use-hydra-plus-pydantic.md)
for the rationale.

### 7.2 Adding a new config group

1. Drop a YAML file under `conf/<group>/<name>.yaml` with a
   `# @package <group>.<name>` header.
2. Reference the variant in `conf/config.yaml` under `defaults:` if
   it should activate by default; otherwise compose at the entry
   point with `+<group>=<name>`.
3. Add a Pydantic schema for it in `conf/_schemas.py` (frozen +
   `extra="forbid"`).
4. Add a field to `AppConfig` (or to the matching group container,
   e.g. `EvaluationGroup`, `ServicesGroup`) typed by the new
   schema.

### 7.3 Override precedence

Hydra's standard precedence applies: CLI overrides win over config
defaults, programmatic `load_config(overrides=[...])` argument
overrides win over both.  Test fixtures use the programmatic form;
the CLI uses Hydra's `key=value` syntax.

---

## 8. Technology choices

> _Document your project's load-bearing dependency choices and the
> reasons behind them.  This section is a thin pointer to the ADRs
> under `docs/architecture/decisions/`._

The template ships with:

- **Python 3.12** — current stable; matches `requires-python` in
  `pyproject.toml`.
- **uv** — package + venv manager.
- **Hydra + Pydantic** — see ADR 0001.
- **loguru** — structured logging without `logging.Logger`
  ceremony.
- **ruff** — lint + format.
- **pytest** — tests.
- **pre-commit** — local hook scaffold mirroring CI.

Add your own as the project demands; document load-bearing choices
as ADRs.

---

## 9. Stage plan

The methodology ships one stage by default:

### Stage 0 — Foundation

Land the template scaffold and the worked-example text-statistics
service.  Acceptance criteria:

- `uv sync --group dev` succeeds.
- `uv run pytest` passes (14 tests).
- `uv run python -m TEMPLATE_PROJECT --help` works.
- `uv run ruff check . && uv run ruff format --check .` clean.

Retrospective at
[`docs/lld/stages/00-foundation.md`](../lld/stages/00-foundation.md).

> _Add your project's stages below.  Each stage has an intent doc
> at `docs/intent/NN-<slug>.md`, a plan at
> `docs/plans/active/NN-<slug>.md`, and a retrospective at
> `docs/lld/stages/NN-<slug>.md`.  See `CLAUDE.md` for the full
> stage hygiene rules._

### Stage definition of done

A stage is "done" when:

- All tests pass on the stage's acceptance criteria.
- The stage's retrospective is written under `docs/lld/stages/`.
- `CHANGELOG.md` has an entry under `[Unreleased]`.
- The stage's plan moves from `docs/plans/active/` to
  `docs/plans/completed/`.
- Any module CLAUDE.md / layer doc / DESIGN.md §6 changes are
  committed.

---

## 10. Deferred decisions

> _Use this section to record decisions you considered and chose
> not to make yet, with the trigger that would resurface them.
> Examples: "monitoring layer — defer until we have a deployed
> service that needs metrics"; "FastAPI vs Flask — defer until
> Stage N when serving lands"._

---

## Appendix A — Source URLs

> _Drop a list of canonical references for the data sources / APIs
> / standards your project consumes._

---

## Appendix B — Framework references

- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Loguru documentation](https://loguru.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [pytest documentation](https://docs.pytest.org/)
