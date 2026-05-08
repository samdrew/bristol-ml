#!/bin/bash
#
# limit-write-paths.sh — Claude Code PreToolUse hook
#
# Restricts Edit/Write tool calls to one or more allowed directories
# under the repository root. Exits 0 to allow, 2 to block.
#
# Usage (in agent frontmatter):
#   hooks:
#     PreToolUse:
#       - matcher: "Edit|Write"
#         hooks:
#           - type: command
#             command: "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/limit-write-paths.sh docs tests"
#
# Each argument is a directory path relative to the repository root.
# Trailing slashes are optional. The tool call is allowed iff the
# target file path resolves to a location inside at least one of the
# allowed directories.

set -euo pipefail

# --- Argument validation -----------------------------------------------------

if [ "$#" -eq 0 ]; then
  echo "limit-write-paths.sh: no allowed directories specified" >&2
  exit 2
fi

# --- Read hook input ---------------------------------------------------------

# Claude Code passes a JSON object on stdin. We need the file path from
# tool_input. Different tools use different field names; check the common
# ones in order.

INPUT=$(cat)

if ! command -v jq >/dev/null 2>&1; then
  echo "limit-write-paths.sh: jq is required but not installed" >&2
  exit 2
fi

TARGET=$(echo "$INPUT" | jq -r '
  .tool_input.file_path
  // .tool_input.path
  // .tool_input.notebook_path
  // empty
')

# If we cannot find a path field, fail closed: refuse the call rather
# than allow it on the assumption that there is nothing to check. A
# tool call we do not recognise is one we cannot validate.
if [ -z "$TARGET" ]; then
  TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
  echo "limit-write-paths.sh: could not extract target path from $TOOL_NAME tool input" >&2
  exit 2
fi

# --- Locate the repository root ----------------------------------------------

# We resolve allowed directories relative to the repo root, not the
# current working directory, because subagents may have a different cwd
# than the project root and we want path restrictions to be stable.

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)

if [ -z "$REPO_ROOT" ]; then
  echo "limit-write-paths.sh: not inside a git repository, cannot resolve repo root" >&2
  exit 2
fi

# --- Resolve the target to an absolute, canonical path ----------------------

# The target may be relative (to cwd) or absolute. We need its canonical
# absolute form so we can check whether it lies under an allowed dir.
#
# realpath -m resolves the path without requiring it to exist (Edit/Write
# may be creating a new file). The -m flag means "missing components are
# OK". We do NOT want to follow symlinks on existing components either,
# because a symlink under tests/ pointing to src/ would otherwise let an
# agent escape the sandbox. Using realpath -ms gives us "canonicalise
# without following symlinks, allow missing components".

if ! command -v realpath >/dev/null 2>&1; then
  echo "limit-write-paths.sh: realpath is required but not installed" >&2
  exit 2
fi

# If the target is relative, resolve it against cwd first. realpath -ms
# handles both relative and absolute inputs, so we can pass it directly.
ABS_TARGET=$(realpath -ms -- "$TARGET")

# --- Build the list of allowed absolute prefixes -----------------------------

# Each argument is a directory path relative to the repo root. We strip
# any trailing slashes the user typed, prefix with the repo root, and
# canonicalise. We then re-append a trailing slash so that prefix matching
# is unambiguous: "$REPO_ROOT/tests/" matches "$REPO_ROOT/tests/foo.py"
# but not "$REPO_ROOT/tests-helper/foo.py", which a slash-less prefix
# match would incorrectly allow.

ALLOWED_PREFIXES=()

for dir in "$@"; do
  # Strip leading and trailing slashes from the user input. Leading
  # because the directory should be relative to the repo root, and a
  # leading slash on the input would otherwise make realpath treat it
  # as absolute. Trailing because we are about to add our own.
  cleaned=${dir#/}
  cleaned=${cleaned%/}

  if [ -z "$cleaned" ]; then
    echo "limit-write-paths.sh: empty directory argument" >&2
    exit 2
  fi

  abs_dir=$(realpath -ms -- "$REPO_ROOT/$cleaned")
  ALLOWED_PREFIXES+=("$abs_dir/")
done

# --- Check the target against the allowed prefixes ---------------------------

# We compare ABS_TARGET against each allowed prefix with a slash appended
# to ABS_TARGET as well, but only for the comparison — this lets us match
# the case where someone tries to write to the directory itself rather
# than a file inside it. In practice Edit/Write always target files, but
# being defensive costs nothing.

TARGET_WITH_SLASH="${ABS_TARGET}/"

for prefix in "${ALLOWED_PREFIXES[@]}"; do
  case "$TARGET_WITH_SLASH" in
    "$prefix"*)
      exit 0
      ;;
  esac
done

# --- Refused: build a clear error message ------------------------------------

# Show the target relative to the repo root for readability, and list
# the directories that would have been acceptable.

REL_TARGET=${ABS_TARGET#"$REPO_ROOT/"}

{
  echo "Blocked: write to '$REL_TARGET' is outside the allowed directories for this agent."
  echo "Allowed directories (relative to repo root):"
  for dir in "$@"; do
    cleaned=${dir#/}
    cleaned=${cleaned%/}
    echo "  - $cleaned/"
  done
  echo "If this write is genuinely necessary, surface the request to the lead rather than working around the restriction."
} >&2

exit 2
