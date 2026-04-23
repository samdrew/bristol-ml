# CLAUDE.md

## Project overview

`bristol_ml` is a reference ML / data-science architecture in Python 3.12, built around a GB day-ahead electricity demand forecaster as the worked example. Two goals: pedagogical (live-demo surface at meetups, self-paced learning artefact) first; analytical (beat the NESO day-ahead forecast) second.

**The project spec lives at [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md)** — read it before any non-trivial change. Where this file and the spec disagree, the spec wins. Principles referenced below as "§2.1.X" are the spec's numbered principles.

This project runs inside a Docker container with no GitHub credentials. Git operations stay local; the human pushes from the host. Never attempt to push, configure remotes, or handle authentication.


## Commands

```
uv sync --group dev --frozen            # setup; installs from uv.lock
uv run pytest                           # tests
uv run ruff check .                     # lint
uv run ruff format --check .            # format check
uv run python -m bristol_ml --help      # Hydra CLI help
uv run python -m bristol_ml             # resolve + validate + print config
uv run pre-commit run --all-files       # mirrors CI hooks locally
```

All commands run through `uv`; never invoke `pip` or a system Python directly. CI runs the same pytest / ruff commands as above.


## Module boundaries

At Stage 0 only the scaffold exists. Directories marked *(deferred)* arrive in their respective stage, each with its own sub-`CLAUDE.md`, at least one test, and an entry in `docs/lld/stages/`.

```
src/bristol_ml/
├── __init__.py          # exposes __version__ and load_config
├── __main__.py          # enables `python -m bristol_ml`
├── cli.py               # Hydra entry point
├── config.py            # resolve-then-validate glue
├── py.typed
├── ingestion/           # (deferred — Stages 1, 2, 5, 13, 17)
├── features/            # (deferred — Stages 3, 5, 16)
├── models/              # (deferred — Stages 4, 7, 8, 10, 11)
├── evaluation/          # (deferred — Stages 3, 4, 6)
├── llm/                 # (deferred — Stages 14, 15)
├── registry/            # Stage 9 — filesystem-backed model registry (save, load, list_runs, describe)
├── serving/             # (deferred — Stage 12)
└── monitoring/          # (deferred — Stage 18)

conf/                    # Hydra YAML + Pydantic schemas
tests/{unit,integration,fixtures}/   # tests/<module>/ mirrors src/bristol_ml/<module>/
data/                    # gitignored local cache; .gitkeep pinned
```

Doc layout is a tiered write surface — see next section.


## Doc tiers and write surface

The `docs/` tree has four tiers enforced by the lead agent's write hook:

```
docs/
├── intent/              # DENY  — project spec
│   └── DESIGN.md
├── architecture/        # WARN  — structural design
│   ├── README.md        # overarching frame + layer-doc index
│   ├── layers/          # one file per module layer, written as each layer lands
│   └── decisions/       # MADR ADRs (append-only by convention)
├── plans/               # ALLOW — per-stage plans
│   ├── active/          # in-flight plans; one NN-<slug>.md per stage
│   └── completed/       # archived plans, moved here at PR merge
└── lld/                 # ALLOW — low-level design
    ├── README.md
    ├── stages/          # one markdown per completed stage
    ├── research/        # @researcher output (on demand)
    └── reframings/      # @reframer output (on demand)
```

| Tier | Path | Policy | Notes |
|------|------|--------|-------|
| Intent | `docs/intent/` | Deny | Substantive edits require human approval; mechanical edits (e.g. DESIGN.md §6 at stage boundaries) happen in the main Claude Code session, not in `--agent lead`. |
| Architecture | `docs/architecture/` | Warn | Layer docs under `layers/` and new ADRs under `decisions/` are legitimate lead activities; ADRs are append-only (supersede rather than edit). |
| Active plans | `docs/plans/active/NN-*.md` | Allow | Living work briefs for the current stage; the lead updates status, links produced LLD, and tweaks reading order in flight. Moves to `completed/` at PR merge. |
| Completed plans | `docs/plans/completed/NN-*.md` | Allow | Archive of shipped plans; kept for reference, not edited in place. |
| LLD | `docs/lld/` | Allow | Stage retros, research, reframings, ad-hoc notes — freely mutable. |

Root-level files: `CHANGELOG.md` = allow; `CLAUDE.md` = warn; `README.md` = warn. Everything else in the repo (`src/`, `tests/`, `conf/`, `data/`, `.github/`, `.claude/`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `.gitignore`, `.pre-commit-config.yaml`) is hard-denied to the lead by the hook's fail-closed default.

Other agents have narrower hooks: `@docs` writes under `docs/`, `@tester` under `tests/`, `@researcher` under `docs/lld/research/`. `@implementer` uses worktree isolation instead of a path hook.

Hook mechanics (allow/warn/ask/deny outcomes, longest-prefix match, fail-closed default, agent frontmatter wiring) are in [`.claude/playbook/path-restrictions.md`](.claude/playbook/path-restrictions.md).

**Spec-drift rule.** When the implementation diverges from the spec, the implementation is what users will experience and the spec must describe reality — but the divergence itself must be surfaced and justified before any spec edit. No silent spec rewrites to match code drift.


## Coding conventions

- British English in documentation and user-facing strings.
- Type hints on all public function signatures; never `# type: ignore` without a comment explaining why (§2.1.2).
- Docstrings on all public classes and functions; module docstrings state intent in one or two sentences.
- Line length 100, ruff config in `pyproject.toml` (`known-first-party = ["bristol_ml", "conf"]`).
- Every module runs standalone via `python -m bristol_ml.<module>` (§2.1.1).
- Configuration lives outside code: YAML in `conf/`, Pydantic schemas in `conf/_schemas.py` (§2.1.4). Never hard-code values that belong in a config.
- Downstream code never sees raw `DictConfig` — convert via `bristol_ml.config.validate` at the CLI boundary and pass the Pydantic model onward.
- Parquet (pyarrow) with documented schemas at storage boundaries.
- Notebooks are thin (§2.1.8) — they import from `src/bristol_ml/`, they do not reimplement logic.
- Stub-first for expensive or flaky external dependencies (§2.1.3); real and stub behind the same interface.
- Idempotent ingestion (§2.1.5) — re-running overwrites or skips, never corrupts.
- Tests at boundaries, not everywhere (§2.1.7). Coverage is not a goal; behavioural clarity is.


## Stage hygiene

Every stage PR updates:

- The module's own `CLAUDE.md` (for any module created or meaningfully touched).
- `docs/lld/stages/NN-name.md` — retrospective following the template in `docs/lld/stages/00-foundation.md`.
- `CHANGELOG.md` — an `### Added` / `### Changed` / `### Fixed` bullet under `[Unreleased]`.

Structural changes also update `docs/intent/DESIGN.md` §6 and `README.md` (user-facing entry points). §6 edits are deny-tier for the lead — make them in the main session with human approval. The authoritative definition-of-done is in `docs/intent/DESIGN.md` §9.


## Quality gates and debugging

- All tests pass before a task reports complete. No skipped tests; no `xfail` without a linked issue.
- Lead never accepts "done" from an implementer without the tester confirming independently.
- No silent spec deviations. If the implementer needs to deviate from the spec, surface to the lead; the lead escalates to the human.
- Testers never weaken a test to make it pass. If a test is wrong (does not match the spec), fix it for the right reason and document the change.
- **Hypothesis enumeration.** Before any debugging task, enumerate ≥4 distinct hypotheses ranked by likelihood, with evidence for and against each. Attempt in ranked order. Skip a hypothesis only by writing why.


## Team conventions

**Default team shape.** Most multi-component tasks use:

- **Lead** — coordinates, owns the contract between teammates, enforces quality gates. Does not implement code itself.
- **`@implementer`** — production code and implementation-derived tests (unit, regression). Runs in an isolated worktree.
- **`@tester`** — spec-derived tests (acceptance, integration, behavioural). Cannot modify production code. Argues with the implementer when results do not match the spec.
- **`@docs`** — externally-visible documentation, after implementation is stable. Spawned only when public interfaces change.

**Tester timing.** Spawn the tester first or in parallel with the implementer, never after. The tester's job is to write tests against the spec, not against the implementation.

**When to use a team.** Team for: well-specified multi-component features with a shared contract; refactors with real interdependencies; tasks where spec is clear and execution is the bottleneck. Single session for: bug fixes, single-file changes, anything where requirements are still being figured out, anything where human judgement is the bottleneck. Not a team task: research (use chat-based deep research); messy codebases (fix the codebase first).

**Spawn prompts must embed:** the spec path, the acceptance criteria, the baseline SHA, and any context the lead gathered that the teammate would otherwise rediscover. Teammates do not inherit the lead's conversation history.

**When the right approach is unclear** (new model family, unusual data semantics), spawn `@researcher` before the implementing team. Findings land in `docs/lld/research/` and become input to the implementers' spawn prompts.

**Scope-guard at Phase 1.** After the three Phase-1 research agents return and before the lead writes `docs/plans/active/NN-<slug>.md`, spawn `@minimalist` as a pre-synthesis critic. It reads the intent, the three research artefacts, and the draft decision set; emits a Scope Diff tagging every decision / NFR / test / dep / notebook cell as `RESTATES INTENT`, `PLAN POLISH`, `PREMATURE OPTIMISATION`, or `HOUSEKEEPING`; closes with one "single highest-leverage cut" sentence. Output is persisted to `docs/lld/research/NN-<slug>-scope-diff.md` as the fourth Phase-1 research artefact and linked from the plan's preamble, so the Ctrl+G reviewer sees both. The `@arch-reviewer` at Phase 3 applies the same four-tag taxonomy to the implementation diff to catch bloat added post-plan.


## Git protocol

All teammate work runs on `task/<id>-<slug>` branches from a recorded baseline SHA. Hard rules: never push; never modify main directly; never carry partial state from a failed attempt (reset is unconditional); always clean up worktrees after consolidating or discarding. Full per-attempt protocol, commit-message template, and success/failure report formats in [`.claude/playbook/git-protocol.md`](.claude/playbook/git-protocol.md).


## Escalation ladder

Applies only under the defensive team shape. For the default team, the lead manages retries directly and escalates to the human after two failed attempts.

| Tier | Trigger | Action | Meta-role |
|------|---------|--------|-----------|
| 0 | Every 1st debugging attempt | Four-hypothesis enumeration (above) | — |
| 1 | 1 failed attempt | Role swap: critique then retry | `@sceptic` |
| 2 | 2 failed attempts | Cross-domain explanation then retry | `@translator` |
| 3 | 3 failed attempts | Adversarial reframing; reset to baseline | `@reframer` |
| 4 | 4 failed attempts | Stop; structured failure report to human | — |

Do not skip tiers. Do not allow more than four attempts without human escalation. Full procedure, meta-role sourcing, and attempt-tracking rules in [`.claude/playbook/escalation-ladder.md`](.claude/playbook/escalation-ladder.md).


## Session conventions

- For any "implement Stage N" request, the canonical entry point is `docs/plans/active/NN-<slug>.md` — it names every other document to read, in order, with acceptance criteria and exit checklist. Read the plan before opening code; spawn teammates with the plan (not this CLAUDE.md) as the self-contained context. (The legacy `docs/stages/` tree is deprecated and retained for historical briefs only.)
- Publish interface contracts to `docs/lld/` (subdir per module as needed) before implementation begins, so teammates work against an agreed schema.
- When the team finishes, clean it up explicitly. Do not leave teammates idle.
