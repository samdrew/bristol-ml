#!/bin/bash
#
# tiered-paths.sh — Claude Code PreToolUse hook
#
# Restricts file-touching tool calls (Read, Edit, Write, NotebookEdit,
# MultiEdit) by path, with three tiers and a configurable default.
#
# Flags:
#   --allow DIR [DIR ...]    silent allow
#   --warn  DIR [DIR ...]    allow with a warning injected into agent context
#   --deny  DIR [DIR ...]    hard block
#   --default allow|deny     fallback when no rule matches; default: allow
#
# Each tier flag consumes positional args until the next flag, so you
# can pass multiple paths after one flag.  The two forms are
# equivalent:
#
#   --allow docs tests src --deny secrets keys
#   --allow docs --allow tests --allow src --deny secrets --deny keys
#
# Tier precedence:  DENY  >  WARN  >  ALLOW  >  DEFAULT
#
# A target that matches rules across multiple tiers resolves to the
# most-severe tier; path depth is irrelevant.  This means:
#
#   `--allow docs --deny docs/intent`
#       allows writes anywhere under docs/, except docs/intent/
#       (deny on the subtree wins over allow on the parent).
#
#   `--allow source/foo --deny source`
#       denies writes anywhere under source/, including source/foo/
#       (deny on the parent wins over allow on the child).
#
# Use as a Read or Edit/Write hook by setting the matcher in the
# agent's frontmatter; the script behaves identically either way:
#
#   hooks:
#     PreToolUse:
#       - matcher: "Edit|Write|NotebookEdit"
#         hooks:
#           - type: command
#             command: "~/.claude/hooks/tiered-paths.sh --default deny --allow docs/lld --warn docs/architecture --deny docs/intent"
#       - matcher: "Read"
#         hooks:
#           - type: command
#             command: "~/.claude/hooks/tiered-paths.sh --deny secrets/"

set -euo pipefail

# --- Argument parsing --------------------------------------------------------
#
# A "positional consumer" loop: --allow / --warn / --deny each set a
# state variable; subsequent non-flag args accumulate into that tier's
# list until the next flag (or end of args).  --default takes exactly
# one argument.

ALLOW_DIRS=()
WARN_DIRS=()
DENY_DIRS=()
DEFAULT_TIER="allow"

current_flag=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --allow|--warn|--deny)
      current_flag="${1#--}"
      shift
      ;;
    --default)
      [ -n "${2:-}" ] || {
        echo "tiered-paths.sh: --default requires 'allow' or 'deny'" >&2
        exit 2
      }
      case "$2" in
        allow|deny) DEFAULT_TIER="$2" ;;
        *)
          echo "tiered-paths.sh: --default must be 'allow' or 'deny', got '$2'" >&2
          exit 2
          ;;
      esac
      current_flag=""
      shift 2
      ;;
    --*)
      echo "tiered-paths.sh: unknown flag '$1'" >&2
      exit 2
      ;;
    *)
      if [ -z "$current_flag" ]; then
        echo "tiered-paths.sh: positional '$1' before any --allow/--warn/--deny" >&2
        exit 2
      fi
      case "$current_flag" in
        allow) ALLOW_DIRS+=("$1") ;;
        warn)  WARN_DIRS+=("$1") ;;
        deny)  DENY_DIRS+=("$1") ;;
      esac
      shift
      ;;
  esac
done

# --- Dependency checks -------------------------------------------------------

for cmd in jq realpath git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "tiered-paths.sh: $cmd is required but not installed" >&2
    exit 2
  fi
done

# --- Read hook input ---------------------------------------------------------
#
# Claude Code sends a JSON envelope on stdin.  Edit/Write/Read use
# .tool_input.file_path; NotebookEdit uses .tool_input.notebook_path;
# some legacy variants use .tool_input.path.  Bash / WebFetch /
# Glob / Grep have no path field — if the matcher in the agent
# frontmatter accidentally captures one of those tools, this hook
# exits 2 (configuration error) rather than silently allowing.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
TARGET=$(echo "$INPUT" | jq -r '
  .tool_input.file_path
  // .tool_input.path
  // .tool_input.notebook_path
  // empty
')

if [ -z "$TARGET" ]; then
  echo "tiered-paths.sh: could not extract target path from $TOOL_NAME tool input" >&2
  exit 2
fi

# --- Locate the repository root ----------------------------------------------

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
  echo "tiered-paths.sh: not inside a git repository" >&2
  exit 2
fi

# --- Resolve target ----------------------------------------------------------
#
# Compute the target two ways:
#
#   ABS_TARGET_LITERAL  (realpath -ms): the literal path the agent
#                                       typed; symlinks are NOT
#                                       followed.
#
#   ABS_TARGET_RESOLVED (realpath -m):  the actual filesystem
#                                       destination; symlinks ARE
#                                       followed.  This is where the
#                                       bytes will land.
#
# Both flags include -m, so missing components are tolerated (Write
# may be creating a new file).
#
# DENY rules are matched against BOTH forms — if either the agent's
# literal intent OR its actual filesystem effect lands in a denied
# tree, the write is blocked.  This closes the symlink-bypass attack:
# a symlink at  allowed/decoy -> denied/secret  has literal form
# "allowed/decoy" (looks fine) but resolved form "denied/secret"
# (matches deny rule), so the deny rule fires.
#
# ALLOW and WARN rules are matched only against the RESOLVED form —
# permission is granted on the basis of the actual filesystem
# effect, not the symbolic intent.  This is the conservative
# direction: if a symlink obscures the destination, the rule does
# not grant permission.

ABS_TARGET_LITERAL=$(realpath -ms -- "$TARGET")
ABS_TARGET_RESOLVED=$(realpath -m  -- "$TARGET")
TARGET_LITERAL_SLASH="${ABS_TARGET_LITERAL}/"
TARGET_RESOLVED_SLASH="${ABS_TARGET_RESOLVED}/"

# --- Helper: check whether the target matches any rule in a list ------------
#
# Returns the matched absolute prefix on stdout (empty if no match).
# Each rule is repo-rooted, canonicalised, and trailing-slash-anchored
# so that the rule "src" matches "src/foo.py" but NOT "src-tools/foo.py"
# (a string-prefix match without the trailing slash would incorrectly
# allow "src-tools/").

resolve_prefix() {
  local raw="$1"
  local cleaned="${raw#/}"
  cleaned="${cleaned%/}"
  if [ -z "$cleaned" ]; then
    echo "tiered-paths.sh: empty directory in rule" >&2
    exit 2
  fi
  realpath -ms -- "$REPO_ROOT/$cleaned"
}

matches_any() {
  # First arg: target with trailing slash.
  # Remaining args: raw rule directories (possibly zero).
  local target="$1"
  shift
  for raw in "$@"; do
    [ -z "$raw" ] && continue
    local abs_prefix
    abs_prefix=$(resolve_prefix "$raw")
    local prefix_with_slash="${abs_prefix}/"
    case "$target" in
      "$prefix_with_slash"*)
        printf '%s' "$abs_prefix"
        return 0
        ;;
    esac
  done
  printf ''
}

# --- Decide tier (DENY > WARN > ALLOW > DEFAULT) -----------------------------
#
# Deny checks both target canonicalisations (literal first, then
# resolved); warn and allow check only the resolved form.  See the
# "Resolve target" block above for the rationale.
#
# Note: ${ARRAY[@]:-} is the bash idiom for "expand the array, but
# tolerate it being empty under set -u".  It introduces one empty
# string element; matches_any() guards against that with the
# "[ -z "$raw" ] && continue" line.

DECISION=""
MATCHED_PREFIX=""

deny_match=$(matches_any "$TARGET_LITERAL_SLASH" "${DENY_DIRS[@]:-}")
if [ -z "$deny_match" ]; then
  deny_match=$(matches_any "$TARGET_RESOLVED_SLASH" "${DENY_DIRS[@]:-}")
fi

if [ -n "$deny_match" ]; then
  DECISION="deny"
  MATCHED_PREFIX="$deny_match"
else
  warn_match=$(matches_any "$TARGET_RESOLVED_SLASH" "${WARN_DIRS[@]:-}")
  if [ -n "$warn_match" ]; then
    DECISION="warn"
    MATCHED_PREFIX="$warn_match"
  else
    allow_match=$(matches_any "$TARGET_RESOLVED_SLASH" "${ALLOW_DIRS[@]:-}")
    if [ -n "$allow_match" ]; then
      DECISION="allow"
      MATCHED_PREFIX="$allow_match"
    else
      DECISION="$DEFAULT_TIER"
      MATCHED_PREFIX=""
    fi
  fi
fi

# --- Apply the decision ------------------------------------------------------

# Report the literal path in user-facing messages — that's what the
# agent typed and will recognise.
REL_TARGET=${ABS_TARGET_LITERAL#"$REPO_ROOT/"}
REL_PREFIX=${MATCHED_PREFIX#"$REPO_ROOT/"}

case "$DECISION" in
  allow)
    # Silent allow.  Exit 0 with no JSON.
    exit 0
    ;;

  warn)
    # Allow but inject a context note.  The apostrophes in the
    # rendered message are written as the six-character JSON Unicode
    # escape \u + 0027 in the source below — jq emits each as a
    # literal apostrophe in the output, but the source can sit
    # inside the single-quoted shell string without breaking
    # the shell quoting.
    jq -n \
      --arg path "$REL_TARGET" \
      --arg dir "$REL_PREFIX" \
      --arg tool "$TOOL_NAME" \
      '{
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision: "allow",
          additionalContext: ($tool + " on \u0027" + $path + "\u0027 is permitted but discouraged (matches warn rule \u0027" + $dir + "/\u0027). Proceed only if necessary; prefer routing through the lead if the change is structural.")
        }
      }'
    exit 0
    ;;

  deny)
    {
      if [ -n "$MATCHED_PREFIX" ]; then
        echo "Blocked: $TOOL_NAME on '$REL_TARGET' is denied (matches deny rule '$REL_PREFIX/')."
      else
        echo "Blocked: $TOOL_NAME on '$REL_TARGET' is denied (no rule matched and --default deny is set)."
        if [ "${#ALLOW_DIRS[@]}" -gt 0 ] || [ "${#WARN_DIRS[@]}" -gt 0 ]; then
          echo "Permitted paths (relative to repo root):"
          [ "${#ALLOW_DIRS[@]}" -gt 0 ] && for d in "${ALLOW_DIRS[@]}"; do echo "  allow: ${d%/}/"; done
          [ "${#WARN_DIRS[@]}"  -gt 0 ] && for d in "${WARN_DIRS[@]}";  do echo "  warn:  ${d%/}/"; done
        fi
      fi
      echo "If this $TOOL_NAME is genuinely needed, surface the request to the lead rather than working around the restriction."
    } >&2
    exit 2
    ;;
esac
