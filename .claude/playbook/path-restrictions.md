# Path restriction enforcement

Teammate write permissions are enforced by two hooks in
`~/.claude/hooks/`:

- `tiered-write-paths.sh` — four-tier decision: allow, warn, ask, deny.
- `limit-write-paths.sh` — simpler single-directory limiter used by
  some agents (e.g. `@tester`, `@docs`).

The top-level `CLAUDE.md` documents the project-specific tier map.
This file covers the hook mechanics themselves.

## Decision outcomes (`tiered-write-paths.sh`)

- **allow** — write proceeds silently.
- **warn** — write proceeds but a warning is injected into the
  agent's context, encouraging the agent to reconsider or surface the
  change to the lead.
- **ask** — write requires explicit user confirmation before
  proceeding. Rarely used; reserved for paths where human judgement
  is genuinely wanted on every edit.
- **deny** — write is hard-blocked. The agent sees an error message
  and must surface the request to the lead if the change is genuinely
  needed.

## Matching rules

Each rule is specified as a directory or single-file path prefix.
When a write is attempted, the hook:

1. Resolves the target path to an absolute canonical form (via
   `realpath -ms`, which handles `..` traversal safely).
2. Checks every rule across all four tiers.
3. Picks the rule whose prefix is the longest match against the
   target. **Longest-prefix wins** — so `--warn docs` combined with
   `--deny docs/intent` denies writes under `docs/intent/` while
   warning on writes elsewhere under `docs/`.
4. If no rule matches, the write is **hard-denied** (fail-closed
   default). Agents must have every writable path explicitly
   enumerated.

## Scope and limits

The hook intercepts `Edit` and `Write` tool calls only. It does not
restrict `Bash`, so an agent with Bash access can theoretically write
outside its allowed paths via shell commands. The hook catches
accidental boundary violations; deliberate evasion is not the threat
model. The Docker container provides the hard boundary.

## Wiring in agent frontmatter

Example for the `lead` agent:

```yaml
hooks:
  PreToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "~/.claude/hooks/tiered-write-paths.sh --deny docs/intent --warn docs/architecture --warn CLAUDE.md --warn README.md --allow docs/lld --allow CHANGELOG.md"
```

Every legitimate write surface must be enumerated. Anything not
listed is hard-denied by the fail-closed default.
