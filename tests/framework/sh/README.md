# Shell-framework tests

Spec-derived tests for the shell scripts that ship with this
template's agent infrastructure (the `.claude/hooks/` family).
These scripts are a generic capability, not part of the Python
project, so their tests live in this separate tree rather than
under `tests/{unit,integration}/` (which mirror
`src/TEMPLATE_PROJECT/`).

## Running

```sh
tests/framework/sh/run.sh                       # all tests
tests/framework/sh/run.sh test_tiered_paths.sh  # one file
```

The runner exits 0 only if every test script exits 0.  No
dependencies beyond `bash`, `jq`, `git`, and `realpath` —
exactly what the hooks themselves require.

## Conventions

- Each test file is `test_*.sh` and is executable.
- Each test inside a file is a `test_<short_name>` shell function;
  a single `run_test <fn>` block at the bottom drives them.
- Tests are **spec-derived**: they cite a clause from the relevant
  playbook page (`.claude/playbook/path-restrictions.md`) or
  script header docstring, and exercise observable behaviour
  (exit code, stdout, stderr).  They do not inspect script
  internals.
- Each test sets up a fresh fixture (e.g. via `mk_repo` from
  `helpers.sh`) and tears it down before exiting, to prevent
  state leakage.

## Adding a test

1. Identify the spec clause you're pinning down.  Quote it in
   a comment at the top of the test group.
2. Write a `test_<name>` function that exercises the smallest
   behaviour-visible path through the script.  Use the helpers
   in `helpers.sh` (`mk_repo`, `mk_input_filepath`, `run_hook`,
   `assert_rc`, …).
3. Add a `run_test <name>` line at the bottom of the file.
4. Run the file directly to confirm pass/fail.

## Symlink semantics (`tiered-paths.sh`)

The hook resolves the target path two ways:

- **literal** (`realpath -ms`): the path the agent typed, with
  symlinks not followed;
- **resolved** (`realpath -m`): the actual filesystem destination,
  with symlinks followed.

Tier matching is asymmetric:

| Tier | Matches against |
|------|-----------------|
| `--deny`  | literal **or** resolved (either match → deny) |
| `--warn`  | resolved only |
| `--allow` | resolved only |

Rationale: deny is strict (block on any sign of trouble); allow is
conservative (grant only on actual filesystem effect, never on
symbolic intent).  This closes the symlink-bypass attack — a
symlink at `allowed/decoy → denied/secret` looks fine on the
literal path but the resolved deny check fires — without
over-rejecting innocuous symlinks that genuinely land in allowed
territory.

The covering tests are the four `test_symlink_*` and
`test_new_file_via_symlinked_directory_*` cases in
`test_tiered_paths.sh`; if you change the resolution flags or
matching strategy, those are the regression suite.

The hook still trusts the filesystem state at hook-invocation
time.  An agent capable of swapping a symlink between
`PreToolUse` and the actual write is out of scope — there are
many higher-priority bypass routes that would have to be closed
first.
