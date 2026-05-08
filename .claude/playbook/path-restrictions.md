# Path restriction enforcement

Teammate file-tool permissions are enforced by hook scripts in
`.claude/hooks/`:

- **`tiered-paths.sh`** — three-tier decision (deny / warn / allow)
  with a configurable default; works for both write tools (Edit /
  Write / NotebookEdit / MultiEdit) and the Read tool.  Recommended
  for new agents.
- `tiered-write-paths.sh` — earlier four-tier variant (allow / warn
  / ask / deny) using path-length-wins precedence.  Kept for
  backward compatibility; new wiring should prefer
  `tiered-paths.sh`.
- `limit-write-paths.sh` — simpler single-directory limiter; used
  by `@researcher` to scope writes to `docs/lld/research/`.

The top-level `CLAUDE.md` documents the project-specific tier map.
This file covers the hook mechanics themselves.

## Referencing hooks portably (`$CLAUDE_PROJECT_DIR`)

Hook commands run through a shell, so environment variables in the
`command:` field are expanded at hook-invocation time.  Claude Code
sets **`CLAUDE_PROJECT_DIR`** to the project root (the directory
that contains `.claude/`) for every hook invocation, regardless of
whether the call originates from the session-level agent, a
subagent, or a worktree.  Always reference project-local hook
scripts via `$CLAUDE_PROJECT_DIR`:

```yaml
command: "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/tiered-paths.sh --deny secrets"
```

The double quotes around the variable matter — they protect the
path against any whitespace in the parent directory chain (a
project cloned to `~/Code/My Project/` would otherwise break).

`~/.claude/hooks/...` paths are home-relative and assume each user
installs the scripts manually; do not use them in committed agent
files.  Bare relative paths (`.claude/hooks/...`) sometimes work
because the hook is invoked with cwd at the project root, but this
is not guaranteed across every subagent / worktree configuration —
`$CLAUDE_PROJECT_DIR` is the documented contract.

## `tiered-paths.sh` — recommended

### Decision outcomes

- **allow** — silent allow.  Tool call proceeds; no JSON emitted.
- **warn** — allow with a context note injected into the agent's
  next-turn context, encouraging the agent to reconsider or surface
  the change to the lead.
- **deny** — hard block.  The agent sees a structured stderr error
  and must surface the request rather than work around it.

### Tier precedence

**`DENY > WARN > ALLOW > DEFAULT`**, regardless of path depth.
A target that matches rules across multiple tiers resolves to the
most-severe tier.  Examples:

- `--allow docs --deny docs/intent` allows everywhere under
  `docs/`, except `docs/intent/` (deny on the subtree wins).
- `--allow source/foo --deny source` denies everywhere under
  `source/`, including `source/foo/` (deny on the parent wins
  over allow on the child).

When no rule matches, the `--default` decides.  Default behaviour
is **`allow`** (think of the hook as opt-in restriction); set
`--default deny` for opt-in permission.

### CLI shape

```
tiered-paths.sh [--default allow|deny]
                [--allow DIR ...]
                [--warn DIR ...]
                [--deny DIR ...]
```

Each tier flag consumes positional args until the next flag, so
multiple paths after one flag are equivalent to repeating the flag.
The two forms below are identical:

```
--allow docs tests src --deny secrets keys
--allow docs --allow tests --allow src --deny secrets --deny keys
```

### Wiring examples

Edit / Write hook with a default-deny posture:

```yaml
hooks:
  PreToolUse:
    - matcher: "Edit|Write|NotebookEdit|MultiEdit"
      hooks:
        - type: command
          command: "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/tiered-paths.sh --default deny --allow docs/lld CHANGELOG.md --warn docs/architecture CLAUDE.md README.md --deny docs/intent"
```

Read hook (independent of the write hook; both can coexist on the
same agent):

```yaml
    - matcher: "Read"
      hooks:
        - type: command
          command: "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/tiered-paths.sh --deny secrets keys"
```

The same script services both matchers — only the tier rules and
the default change.

## `tiered-write-paths.sh` — legacy (for reference)

Predates `tiered-paths.sh`.  Differs in three ways:

1. Four tiers: **allow / warn / ask / deny**.  The `ask` tier
   surfaces a Y/N prompt to the user before proceeding.
2. **Longest-prefix wins** rather than tier-precedence.  A deeper
   `allow` beats a shallower `deny`, and vice versa — only path
   depth matters.  This makes `--allow docs --deny docs/intent`
   work identically to the new script (deeper deny wins because
   it's longer), but `--allow source/foo --deny source` *allows*
   `source/foo/` (deeper allow wins on path length).
3. **Fail-closed default.**  Anything not matched is denied; there
   is no `--default allow` option.  Every writable path must be
   explicitly enumerated.

Use only on agents that are already wired to it; new agents should
use `tiered-paths.sh`.

## `limit-write-paths.sh` — simpler limiter

A single-directory whitelist (positional args, no flags).  Allows
writes only under the listed directories; denies everything else.
Equivalent to `tiered-paths.sh --default deny --allow DIR1 DIR2 ...`
but the flag-less form is friendlier when the policy is genuinely
"writes here only."  Used by `@researcher` to confine writes to
`docs/lld/research/`.

```yaml
command: "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/limit-write-paths.sh docs/lld/research"
```

## Path resolution semantics

Behaviour shared by all three scripts:

- Missing components are tolerated (every script uses `realpath`
  with the `-m` flag), so `Write` can name a not-yet-existent file
  without the hook erroring out on stat.
- Rule directories are resolved against the **repository root**
  (via `git rev-parse --show-toplevel`).  In subagent worktrees
  that root is the worktree, not the main checkout — which is the
  correct behaviour, because the agent's tool calls land inside
  the worktree.  Rule paths may be absolute or relative; leading
  and trailing slashes are optional.
- Every rule is anchored with a trailing slash, so the rule `src`
  matches `src/foo.py` but **not** `src-tools/foo.py`.
- Targets resolving outside the repo root match no repo-rooted
  rule and fall through to the `--default`.

### `tiered-paths.sh` — symlink handling

`tiered-paths.sh` is asymmetric on symlinks, by design.  The
target is canonicalised two ways:

- **literal** (`realpath -ms`) — the path the agent typed,
  symlinks not followed;
- **resolved** (`realpath -m`) — the actual filesystem
  destination, symlinks followed.

| Tier | Target form matched against rule |
|------|----------------------------------|
| `--deny`  | literal **or** resolved (either match → deny) |
| `--warn`  | resolved only |
| `--allow` | resolved only |

Why the asymmetry: a symlink at `allowed/decoy → denied/secret`
has a literal path that looks fine but a resolved path that lands
in the denied tree.  Checking deny against both forms catches the
bypass.  Checking allow / warn against the resolved form only
keeps grants conservative — an allow rule never grants permission
on the basis of symbolic intent, only on actual filesystem
effect.

The hook trusts the filesystem state at `PreToolUse` time.  A
symlink swapped between the hook firing and the actual write is
out of scope; many higher-priority bypass routes would need to be
closed first before that becomes relevant.

### Legacy scripts

`tiered-write-paths.sh` and `limit-write-paths.sh` use a single
`realpath -ms` canonicalisation and judge by the literal path
only.  This is **vulnerable to the symlink-bypass attack** above
— an agent in an allowed tree can write into a denied tree via
a symlink the hook never resolves.  Migrate any agent still using
these scripts to `tiered-paths.sh`.

## Scope and limits

The hooks intercept the Read / Edit / Write / NotebookEdit /
MultiEdit tools only — whatever the agent's `matcher:` regex
selects.  The hooks do not restrict `Bash`, so an agent with Bash
access can theoretically write outside its allowed paths via shell
commands.  **The hooks catch accidental boundary violations;
deliberate evasion is not the threat model.**  The Docker
container (or whatever sandbox the project deploys into) provides
the hard boundary.

## Wiring contract — summary

- Always reference hook scripts via
  `"$CLAUDE_PROJECT_DIR"/.claude/hooks/<script>.sh`.
- Pick `tiered-paths.sh` for new wiring.
- Document the agent's tier policy in the agent's frontmatter
  (the `command:` line is the source of truth).
- If multiple tools need different policies (e.g. tighter Read
  rules than Write rules), use multiple `matcher:` entries in the
  same `PreToolUse:` block — each invokes its own hook command.
- The top-level `CLAUDE.md` documents the project-specific tier
  map; keep it in sync with the agent frontmatter when changing
  policy.
