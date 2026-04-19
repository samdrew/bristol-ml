#!/bin/bash
#
# tiered-write-paths.sh — Claude Code PreToolUse hook
#
# Restricts Edit/Write tool calls based on path-based tiers:
#   --allow <dir>     allow silently
#   --warn <dir>      allow with a warning injected into agent context
#   --ask <dir>       require explicit user confirmation
#   --deny <dir>      hard block
#
# Anything not matched by any rule is hard-denied (fail closed).
# Multiple flags of each type are allowed. Trailing slashes optional.
#
# Usage in agent frontmatter:
#   hooks:
#     PreToolUse:
#       - matcher: "Edit|Write"
#         hooks:
#           - type: command
#             command: "~/.claude/hooks/tiered-write-paths.sh --allow tests --ask docs/architecture --warn docs --deny src/generated"
#
# Tier resolution: more specific paths win. If a target matches several
# tiers (e.g. "docs" is warn, "docs/architecture" is ask), the deepest
# match takes precedence.

set -euo pipefail

# --- Argument parsing --------------------------------------------------------

ALLOW_DIRS=()
WARN_DIRS=()
ASK_DIRS=()
DENY_DIRS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --allow)
      [ -n "${2:-}" ] || { echo "tiered-write-paths.sh: --allow requires a directory" >&2; exit 2; }
      ALLOW_DIRS+=("$2")
      shift 2
      ;;
    --warn)
      [ -n "${2:-}" ] || { echo "tiered-write-paths.sh: --warn requires a directory" >&2; exit 2; }
      WARN_DIRS+=("$2")
      shift 2
      ;;
    --ask)
      [ -n "${2:-}" ] || { echo "tiered-write-paths.sh: --ask requires a directory" >&2; exit 2; }
      ASK_DIRS+=("$2")
      shift 2
      ;;
    --deny)
      [ -n "${2:-}" ] || { echo "tiered-write-paths.sh: --deny requires a directory" >&2; exit 2; }
      DENY_DIRS+=("$2")
      shift 2
      ;;
    *)
      echo "tiered-write-paths.sh: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

if [ "${#ALLOW_DIRS[@]}" -eq 0 ] && \
   [ "${#WARN_DIRS[@]}" -eq 0 ] && \
   [ "${#ASK_DIRS[@]}" -eq 0 ] && \
   [ "${#DENY_DIRS[@]}" -eq 0 ]; then
  echo "tiered-write-paths.sh: no path rules specified" >&2
  exit 2
fi

# --- Dependency checks -------------------------------------------------------

for cmd in jq realpath git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "tiered-write-paths.sh: $cmd is required but not installed" >&2
    exit 2
  fi
done

# --- Read hook input ---------------------------------------------------------

INPUT=$(cat)

TARGET=$(echo "$INPUT" | jq -r '
  .tool_input.file_path
  // .tool_input.path
  // .tool_input.notebook_path
  // empty
')

if [ -z "$TARGET" ]; then
  TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
  echo "tiered-write-paths.sh: could not extract target path from $TOOL_NAME tool input" >&2
  exit 2
fi

# --- Locate the repository root ----------------------------------------------

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)

if [ -z "$REPO_ROOT" ]; then
  echo "tiered-write-paths.sh: not inside a git repository" >&2
  exit 2
fi

# --- Resolve target to absolute canonical form ------------------------------

ABS_TARGET=$(realpath -ms -- "$TARGET")
TARGET_WITH_SLASH="${ABS_TARGET}/"

# --- Helper: resolve a relative dir argument to its absolute prefix ---------

resolve_prefix() {
  local raw="$1"
  local cleaned="${raw#/}"
  cleaned="${cleaned%/}"

  if [ -z "$cleaned" ]; then
    echo "tiered-write-paths.sh: empty directory in rule" >&2
    exit 2
  fi

  realpath -ms -- "$REPO_ROOT/$cleaned"
}

# --- Find the most specific matching tier -----------------------------------
#
# We check every rule from every tier, and pick the one whose prefix is the
# longest match against the target. Length-based "most specific wins" lets
# rules like "--warn docs --ask docs/architecture" do the obvious thing.

BEST_MATCH_TIER=""
BEST_MATCH_LENGTH=0
BEST_MATCH_PREFIX=""

check_tier() {
  local tier="$1"
  shift
  for dir in "$@"; do
    local abs_prefix
    abs_prefix=$(resolve_prefix "$dir")
    local prefix_with_slash="${abs_prefix}/"

    case "$TARGET_WITH_SLASH" in
      "$prefix_with_slash"*)
        local len=${#prefix_with_slash}
        if [ "$len" -gt "$BEST_MATCH_LENGTH" ]; then
          BEST_MATCH_TIER="$tier"
          BEST_MATCH_LENGTH="$len"
          BEST_MATCH_PREFIX="$abs_prefix"
        fi
        ;;
    esac
  done
}

check_tier "allow" "${ALLOW_DIRS[@]:-}"
check_tier "warn"  "${WARN_DIRS[@]:-}"
check_tier "ask"   "${ASK_DIRS[@]:-}"
check_tier "deny"  "${DENY_DIRS[@]:-}"

# --- Apply the decision ------------------------------------------------------

REL_TARGET=${ABS_TARGET#"$REPO_ROOT/"}
REL_PREFIX=${BEST_MATCH_PREFIX#"$REPO_ROOT/"}

case "$BEST_MATCH_TIER" in
  allow)
    # Silent allow. Exit 0 with no JSON.
    exit 0
    ;;

  warn)
    # Allow but inject a warning into agent context.
    jq -n --arg path "$REL_TARGET" --arg dir "$REL_PREFIX" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "allow",
        additionalContext: ("Note: writing to '\''" + $path + "'\'' which is under the discouraged area '\''" + $dir + "/'\''. Proceed only if this change is necessary; prefer routing through the lead if the change is structural.")
      }
    }'
    exit 0
    ;;

  ask)
    # Require explicit user confirmation.
    jq -n --arg path "$REL_TARGET" --arg dir "$REL_PREFIX" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "ask",
        permissionDecisionReason: ("Write to '\''" + $path + "'\'' is under the protected area '\''" + $dir + "/'\''. Confirm whether this change should proceed.")
      }
    }'
    exit 0
    ;;

  deny)
    {
      echo "Blocked: write to '$REL_TARGET' is under the hard-deny area '$REL_PREFIX/'."
      echo "This directory is protected from agent writes. Surface the change request to the human."
    } >&2
    exit 2
    ;;

  "")
    # No rule matched at all. Fail closed.
    {
      echo "Blocked: write to '$REL_TARGET' is not covered by any path rule for this agent."
      echo "Allowed paths (relative to repo root):"
      [ "${#ALLOW_DIRS[@]}" -gt 0 ] && for d in "${ALLOW_DIRS[@]}"; do echo "  allow: ${d%/}/"; done
      [ "${#WARN_DIRS[@]}"  -gt 0 ] && for d in "${WARN_DIRS[@]}";  do echo "  warn:  ${d%/}/"; done
      [ "${#ASK_DIRS[@]}"   -gt 0 ] && for d in "${ASK_DIRS[@]}";   do echo "  ask:   ${d%/}/"; done
    } >&2
    exit 2
    ;;
esac
