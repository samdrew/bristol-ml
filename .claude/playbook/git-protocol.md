# Git protocol

This document defines the branching, commit, and worktree discipline
for Claude Code teammate work. It applies to every task run under the
default or defensive team shape. The top-level `CLAUDE.md` keeps the
hard rules and a pointer here; the full procedure lives in this file.

## Branch model

When the lead starts a task, it creates a branch:

```
task/<task-id>-<short-slug>
```

from the current head of the development branch. The lead records the
baseline SHA in the task description. This SHA is the fixed point that
all reset operations refer to — it does not move even if other work
lands on the development branch while the task is in progress.

## Per-attempt protocol

1. Before each attempt, verify the task branch matches the baseline
   SHA. If the previous attempt left changes on the branch, hard reset
   to the baseline first. If using worktree isolation, discard the
   previous worktree and create a fresh one.

2. Spawn the implementer with the task description and baseline SHA.

3. If the implementer was spawned with `isolation: worktree`, after it
   reports completion the lead must explicitly verify where the
   changes are: run `git worktree list`, `git branch`, and diff the
   main checkout against the worktree. Consolidate the changes onto
   the task branch in the main checkout. Do not assume reintegration
   happened automatically.

4. Run the test suite from the main checkout (not the worktree) to
   confirm the consolidated state works.

5. Commit the result on the task branch with a structured message:

   ```
   attempt(<task-id>): <one-line summary>

   Outcome: <success | failed-tests | failed-build | failed-review>
   Hypothesis: <which hypothesis was tested, if debugging>
   Next: <what the next attempt should try, or "escalate">
   ```

6. If the attempt failed, reset the task branch to the baseline before
   the next attempt. Do not carry partial work forward. The failed
   attempt's commit remains in the reflog if needed for post-mortem.

## Task completion (success)

- Squash the task branch's attempt commits into a single clean commit
  on the development branch.
- Delete the task branch.
- Report to the human: "the change is on [branch] in the working
  directory; push when ready."

## Task completion (failure / escalation)

- Do not delete the task branch. Leave it intact with all attempt
  commits visible.
- Report the branch name and structured failure context to the human.
- The human can checkout the branch from the host to inspect.

## Hard rules

- Never push. There are no credentials and nothing should be on a
  remote until the human chooses.
- Never carry partial state from a failed attempt into the next one.
  Reset is unconditional.
- Never modify the development branch directly. All work happens on
  task branches and is consolidated only on success.
- Clean up worktrees after consolidating or discarding their changes.
  Run `git worktree list` to confirm no stale worktrees remain.
