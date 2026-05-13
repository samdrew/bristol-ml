# Claude Code as a team

The [`.claude/`](../../.claude/) directory at the root of this repo is part
of the demo material, not just project tooling. It carries a multi-agent
collaboration setup that's been the engine room of the `bristol_ml`
project — and it's cross-cutting across every session theme. Any meetup
can spend ten minutes poking at it as a side-station.

This page is the standing reference. Every session-page template
includes a one-line "Claude Code corner" pointer here so it surfaces
consistently without consuming a main station slot.

## What's in `.claude/`

```
.claude/
├── agents/          # 13 narrow-role agents
├── playbook/        # how the team operates
├── hooks/           # write-path tier enforcement
└── settings*.json   # session config, hook wiring
```

### Agents

Thirteen narrow-role agents under [`.claude/agents/`](../../.claude/agents/),
each with its own prompt template:

| Agent | Role |
|-------|------|
| `lead` | Coordinates teammates, owns the contract between them, enforces quality gates. **Does not implement code itself.** |
| `implementer` | Production code and implementation-derived tests. Runs in an isolated git worktree. |
| `test-author` | Spec-derived tests (acceptance, integration, behavioural). **Cannot modify production code.** Argues with the implementer when results don't match the spec. |
| `code-review`, `arch-reviewer` | Two reviewers with different briefs — code-quality/correctness vs plan-conformance. |
| `docs-writer` | Externally-visible documentation, after implementation is stable. |
| `researcher`, `domain-researcher` | Web research and library/RFC/standards research; outputs land in `docs/lld/research/`. |
| `requirements-analyst`, `codebase-explorer` | Phase-1 sub-agents the lead spawns to scope a stage before planning. |
| `minimalist` | A pre-synthesis critic that tags every proposed decision as `RESTATES INTENT` / `PLAN POLISH` / `PREMATURE OPTIMISATION` / `HOUSEKEEPING` to catch scope creep before it lands. |
| `sceptic`, `translator`, `reframer` | Escalation-ladder meta-roles (see playbook below). |

The lead never accepts "done" from an implementer without the
test-author confirming independently. Test-authors never weaken a test
to make it pass.

### Playbook

Three documents under [`.claude/playbook/`](../../.claude/playbook/):

- [`escalation-ladder.md`](../../.claude/playbook/escalation-ladder.md) —
  a four-tier debug ladder. Each failed attempt triggers a different
  meta-role (sceptic → translator → reframer), and after four attempts
  the lead stops and escalates to the human with a structured failure
  report. **Do not skip tiers.** Do not allow more than four attempts.
- [`git-protocol.md`](../../.claude/playbook/git-protocol.md) —
  worktree isolation, baseline-SHA tracking, unconditional reset on
  failed attempts. Teammates never push; never modify main directly;
  never carry partial state.
- [`path-restrictions.md`](../../.claude/playbook/path-restrictions.md) —
  the hook mechanics behind the doc-tier write surface (allow / warn /
  ask / deny, longest-prefix match, fail-closed default).

### Hooks

The `tiered-paths.sh` hook enforces a four-tier write surface across
`docs/`:

| Tier | Path | Policy |
|------|------|--------|
| Intent | `docs/intent/` | **Deny** — the spec is human-guarded |
| Architecture | `docs/architecture/` | **Warn** — layer docs and ADRs are legitimate but flagged |
| Plans | `docs/plans/` | **Allow** — living per-stage work briefs |
| LLD | `docs/lld/` | **Allow** — retrospectives, research, ad-hoc notes |

Different agents have different hooks: `@docs-writer` writes under
`docs/`, `@test-author` under `tests/`, `@implementer` is sandboxed via
worktree isolation rather than a path hook. The full mechanics are in
the playbook.

## How to demo it during a session

A ten-minute Claude Code corner can take any of these shapes:

**The agent roster tour.** Open
[`.claude/agents/lead.md`](../../.claude/agents/lead.md) and one or
two others (implementer, test-author). Point at the explicit non-overlap
of roles ("test-author cannot modify production code"). Ten minutes.

**The escalation ladder.** Walk through
[`escalation-ladder.md`](../../.claude/playbook/escalation-ladder.md).
The premise is that *every* failed debug attempt is followed by a
meta-role swap, not another attempt at the same framing. Point at how
this maps onto an actual previously-stuck stage from `docs/lld/stages/`
if you have time.

**A live stage retrospective.** Pull any file from
[`docs/lld/stages/`](../../docs/lld/stages/) — each is a Goal / What
was built / Design choices / Demo moment / Deferred / Next document,
written *after* the implementing team finished. Pair with the matching
plan in `docs/plans/completed/` for the before-and-after.

**The plan-then-build-then-review pipeline.** Open
[`CLAUDE.md`](../../CLAUDE.md) and walk through the
"Default team shape" section — the lead spawns the implementer and
test-author *in parallel* (never the implementer first, then tests
after), and the docs-writer only after the public surface is stable.

## Why this is meetup-worthy

The setup answers a question most ML teams haven't yet had to:
**when an AI teammate is doing the work, how do you actually run the
team?** The answer in this repo is concrete — narrow roles, explicit
quality gates, worktree isolation, an escalation ladder with no shortcut,
human-guarded specs, append-only retrospectives.

It's also reproducible. The whole setup is twelve markdown agent files,
three playbook documents, one shell hook, and one settings JSON. You
can clone it into another repo in an afternoon.
