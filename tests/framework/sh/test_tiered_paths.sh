#!/usr/bin/env bash
#
# tests/framework/sh/test_tiered_paths.sh
#
# Spec-derived tests for .claude/hooks/tiered-paths.sh.
#
# Spec sources (in order of authority):
#   1. .claude/playbook/path-restrictions.md  — the recommended
#      `tiered-paths.sh` section, plus "Path resolution semantics"
#      and "Wiring contract — summary".
#   2. The tiered-paths.sh header docstring (lines 1–46) — restates
#      the same contract from the script author's view.
#
# Tests exercise observable behaviour: exit code, stdout JSON, stderr
# message.  They do not inspect script internals (function names,
# variable names, control flow).

set -uo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/../../.." && pwd)
HOOK="$REPO/.claude/hooks/tiered-paths.sh"

# shellcheck source=helpers.sh
source "$DIR/helpers.sh"

if [ ! -x "$HOOK" ]; then
  echo "Hook script not executable: $HOOK" >&2
  echo "(chmod +x .claude/hooks/tiered-paths.sh — see README)" >&2
  exit 1
fi

# =============================================================================
# Group A — Decision outcomes
#
# Spec: "allow → silent allow.  Tool call proceeds; no JSON emitted."
#       "warn → allow with a context note injected into the agent's
#        next-turn context."
#       "deny → hard block.  The agent sees a structured stderr error."
# =============================================================================

test_allow_decision_is_silent_exit_zero() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/note.md")
  run_hook "$repo" "$input" -- --allow docs
  assert_rc 0
  assert_stdout_empty
  rm -rf "$repo"
}

test_warn_decision_emits_permission_decision_allow_with_context() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Edit" "$repo/CLAUDE.md")
  run_hook "$repo" "$input" -- --warn CLAUDE.md
  assert_rc 0
  assert_stdout_is_json_with '.hookSpecificOutput.hookEventName == "PreToolUse"'
  assert_stdout_is_json_with '.hookSpecificOutput.permissionDecision == "allow"'
  assert_stdout_is_json_with '.hookSpecificOutput.additionalContext | length > 0'
  rm -rf "$repo"
}

test_deny_decision_blocks_with_nonzero_exit_and_stderr() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/intent/DESIGN.md")
  run_hook "$repo" "$input" -- --allow docs --deny docs/intent
  # Spec is silent on the exact non-zero code; assert "non-zero".
  if [ "$RC" -eq 0 ]; then
    mark_fail "expected non-zero exit for deny, got 0"
  fi
  assert_stderr_contains "Blocked"
  rm -rf "$repo"
}

# =============================================================================
# Group B — Tier precedence (DENY > WARN > ALLOW > DEFAULT)
#
# Spec: "A target that matches rules across multiple tiers resolves to
#        the most-severe tier; path depth is irrelevant."
#       "--allow docs --deny docs/intent → allows everywhere under
#        docs/, except docs/intent/ (deny on the subtree wins)."
#       "--allow source/foo --deny source → denies everywhere under
#        source/, including source/foo/ (deny on the parent wins)."
# =============================================================================

test_deny_on_subtree_beats_allow_on_parent() {
  # Spec example #1: --allow docs --deny docs/intent.
  local repo; repo=$(mk_repo)
  # Path under docs/ but outside docs/intent/ → allowed.
  local input1; input1=$(mk_input_filepath "Write" "$repo/docs/note.md")
  run_hook "$repo" "$input1" -- --allow docs --deny docs/intent
  assert_rc 0

  # Path under docs/intent/ → denied (deny on subtree wins).
  local input2; input2=$(mk_input_filepath "Write" "$repo/docs/intent/DESIGN.md")
  run_hook "$repo" "$input2" -- --allow docs --deny docs/intent
  if [ "$RC" -eq 0 ]; then
    mark_fail "deny on docs/intent should beat allow on docs"
  fi
  rm -rf "$repo"
}

test_deny_on_parent_beats_allow_on_child() {
  # Spec example #2: --allow source/foo --deny source.
  local repo; repo=$(mk_repo)
  # Path under source/foo/ → denied (deny on parent wins despite the
  # allow rule sitting on a deeper path).
  local input; input=$(mk_input_filepath "Write" "$repo/source/foo/x.txt")
  run_hook "$repo" "$input" -- --allow source/foo --deny source
  if [ "$RC" -eq 0 ]; then
    mark_fail "deny on source/ should beat allow on source/foo (depth-irrelevant precedence)"
  fi
  rm -rf "$repo"
}

test_warn_beats_allow_when_both_match() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/architecture/x.md")
  run_hook "$repo" "$input" -- --allow docs --warn docs/architecture
  # Warn should win → JSON output with permissionDecision=allow,
  # rather than silent allow.
  assert_rc 0
  assert_stdout_is_json_with '.hookSpecificOutput.permissionDecision == "allow"'
  rm -rf "$repo"
}

test_deny_beats_warn_when_both_match() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/secrets/api.key")
  run_hook "$repo" "$input" -- --warn secrets --deny secrets/api.key
  if [ "$RC" -eq 0 ]; then
    mark_fail "deny should beat warn when both rules match"
  fi
  rm -rf "$repo"
}

# =============================================================================
# Group C — Default behaviour
#
# Spec: "When no rule matches, the --default decides.  Default
#        behaviour is allow (think of the hook as opt-in restriction);
#        set --default deny for opt-in permission."
# =============================================================================

test_default_is_allow_when_no_rules_match() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/random/file.txt")
  # No --default flag; only an unrelated --deny.  random/ matches nothing.
  run_hook "$repo" "$input" -- --deny secrets
  assert_rc 0
  assert_stdout_empty
  rm -rf "$repo"
}

test_default_deny_blocks_when_no_rules_match() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/random/file.txt")
  run_hook "$repo" "$input" -- --default deny --allow docs
  if [ "$RC" -eq 0 ]; then
    mark_fail "expected --default deny to block unmatched paths"
  fi
  rm -rf "$repo"
}

test_default_deny_still_permits_explicit_allow() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/note.md")
  run_hook "$repo" "$input" -- --default deny --allow docs
  assert_rc 0
  rm -rf "$repo"
}

# =============================================================================
# Group D — CLI shape
#
# Spec: "Each tier flag consumes positional args until the next flag,
#        so multiple paths after one flag are equivalent to repeating
#        the flag."
#       "--default takes exactly one argument (allow or deny)."
# =============================================================================

test_multiple_paths_after_one_flag_equal_repeated_flags() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/tests/x.py")

  # Form 1: one flag, multiple paths.
  run_hook "$repo" "$input" -- --allow docs tests src
  local rc1="$RC"
  # Form 2: repeated flags.
  run_hook "$repo" "$input" -- --allow docs --allow tests --allow src
  local rc2="$RC"

  if [ "$rc1" != "$rc2" ]; then
    mark_fail "two equivalent CLI forms produced different exit codes ($rc1 vs $rc2)"
  fi
  if [ "$rc1" -ne 0 ]; then
    mark_fail "tests/x.py should be allowed by --allow tests, got $rc1"
  fi
  rm -rf "$repo"
}

test_default_with_invalid_value_fails() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/x.md")
  run_hook "$repo" "$input" -- --default banana --allow docs
  if [ "$RC" -eq 0 ]; then
    mark_fail "expected non-zero exit for --default with invalid value"
  fi
  assert_stderr_contains "default"
  rm -rf "$repo"
}

# =============================================================================
# Group E — Path resolution semantics
#
# Spec: "Targets are canonicalised via realpath -ms: missing
#        components are tolerated (so Write can create new files), and
#        symlinks are not followed (a symlink under an allowed tree
#        pointing into a denied tree must not bypass the deny rule)."
#       "Rule directories are resolved against the repository root."
#       "Rule paths may be absolute or relative; leading and trailing
#        slashes are optional."
#       "Every rule is anchored with a trailing slash, so the rule
#        'src' matches 'src/foo.py' but NOT 'src-tools/foo.py'."
#       "Targets resolving outside the repo root match no rule and
#        fall through to the default."
# =============================================================================

test_missing_target_components_are_tolerated() {
  # Write may create a new file in a not-yet-existent directory.  The
  # hook must still reach a decision rather than failing on stat.
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/new/nested/dir/file.md")
  run_hook "$repo" "$input" -- --allow new
  assert_rc 0
  rm -rf "$repo"
}

test_rule_dir_resolves_against_repo_root_not_cwd() {
  # The hook is invoked with cwd=$repo (typical for Claude Code), but
  # the spec pins resolution to `git rev-parse --show-toplevel` —
  # which yields the same path.  We confirm the rule "docs" maps to
  # $repo/docs/ even when cwd is a subdirectory of $repo.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/sub"
  local input; input=$(mk_input_filepath "Write" "$repo/docs/x.md")
  local out_file err_file
  out_file=$(mktemp); err_file=$(mktemp)
  set +e
  printf '%s' "$input" | (cd "$repo/sub" && "$HOOK" --allow docs) \
    >"$out_file" 2>"$err_file"
  RC=$?
  set -e
  STDOUT=$(cat "$out_file"); STDERR=$(cat "$err_file")
  rm -f "$out_file" "$err_file"
  assert_rc 0
  rm -rf "$repo"
}

test_absolute_and_relative_rule_paths_are_equivalent() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/x.md")

  run_hook "$repo" "$input" -- --allow docs
  local rc_rel="$RC"
  run_hook "$repo" "$input" -- --allow "$repo/docs"
  local rc_abs="$RC"

  if [ "$rc_rel" != "$rc_abs" ]; then
    mark_fail "absolute and relative rule paths produced different decisions ($rc_rel vs $rc_abs)"
  fi
  rm -rf "$repo"
}

test_leading_and_trailing_slashes_in_rule_are_optional() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "$repo/docs/x.md")

  for rule in "docs" "docs/" "/docs" "/docs/"; do
    run_hook "$repo" "$input" -- --allow "$rule"
    if [ "$RC" -ne 0 ]; then
      mark_fail "rule form '$rule' produced non-zero exit ($RC)"
    fi
  done
  rm -rf "$repo"
}

test_trailing_slash_anchoring_distinguishes_sibling_directories() {
  # The spec example: rule "src" matches "src/foo.py" but NOT
  # "src-tools/foo.py".  A naive string-prefix would fail this.
  local repo; repo=$(mk_repo)

  local input_inside; input_inside=$(mk_input_filepath "Write" "$repo/src/foo.py")
  run_hook "$repo" "$input_inside" -- --default deny --allow src
  assert_rc 0  # src/foo.py is inside src/ → allowed.

  local input_sibling; input_sibling=$(mk_input_filepath "Write" "$repo/src-tools/foo.py")
  run_hook "$repo" "$input_sibling" -- --default deny --allow src
  if [ "$RC" -eq 0 ]; then
    mark_fail "rule 'src' must not match sibling directory 'src-tools/'"
  fi
  rm -rf "$repo"
}

test_target_outside_repo_root_falls_through_to_default() {
  # An /etc/... target is outside the repo root.  Spec: "Targets
  # resolving outside the repo root match no rule and fall through
  # to the default."  With default=allow, this is silently allowed.
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "/etc/hosts")
  run_hook "$repo" "$input" -- --allow docs --deny etc
  # The deny rule "etc" resolves to "$repo/etc/", which does NOT
  # match "/etc/hosts".  No rule matches → default (allow) → exit 0.
  assert_rc 0
  rm -rf "$repo"
}

test_target_outside_repo_root_blocked_by_default_deny() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Write" "/etc/hosts")
  run_hook "$repo" "$input" -- --default deny --allow docs
  if [ "$RC" -eq 0 ]; then
    mark_fail "expected --default deny to block /etc/hosts"
  fi
  rm -rf "$repo"
}

# Symlink tests.  The hook judges deny on EITHER the literal path
# (what the agent typed) OR the resolved path (where bytes actually
# land).  Allow / warn judge only on the resolved path.  This makes
# the security boundary strict in the deny direction and conservative
# in the allow direction.

test_symlink_in_denied_tree_pointing_to_allowed_target_is_denied() {
  # Direction: symlink at denied/decoy → allowed/innocent.  Literal
  # path is in denied/, so the deny rule fires on the literal check.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/denied" "$repo/allowed"
  : >"$repo/allowed/innocent"
  ln -s "$repo/allowed/innocent" "$repo/denied/decoy"

  local input; input=$(mk_input_filepath "Write" "$repo/denied/decoy")
  run_hook "$repo" "$input" -- --allow allowed --deny denied
  if [ "$RC" -eq 0 ]; then
    mark_fail "symlink in denied tree must be denied via its literal path"
  fi
  rm -rf "$repo"
}

test_symlink_in_allowed_tree_pointing_to_denied_target_is_denied() {
  # Direction: symlink at allowed/decoy → denied/secret.  Literal
  # path is in allowed/, so the literal check passes — but the
  # resolved path lands in denied/, so the resolved deny check
  # fires.  This is the bypass attack: an agent in an allowed tree
  # using a symlink to write into a denied tree.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/allowed" "$repo/denied"
  : >"$repo/denied/secret"
  ln -s "$repo/denied/secret" "$repo/allowed/decoy"

  local input; input=$(mk_input_filepath "Write" "$repo/allowed/decoy")
  run_hook "$repo" "$input" -- --allow allowed --deny denied
  if [ "$RC" -eq 0 ]; then
    mark_fail "symlink in allowed tree pointing into denied tree must not bypass deny"
  fi
  rm -rf "$repo"
}

test_new_file_via_symlinked_directory_into_denied_is_denied() {
  # Variant of the bypass: a symlinked DIRECTORY in the allowed tree
  # pointing at the denied tree, with the agent attempting to create
  # a new (not-yet-existent) file beneath it.  The literal path
  # (allowed/dir_link/new_file) looks fine; the resolved path
  # (denied/new_file) lands in the denied tree.  Resolved deny check
  # must fire.  This is the case `realpath -m` (without -s) handles
  # by following the directory symlink even though the leaf file
  # does not exist.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/allowed" "$repo/denied"
  ln -s "$repo/denied" "$repo/allowed/dir_link"

  local input; input=$(mk_input_filepath "Write" "$repo/allowed/dir_link/new_file")
  run_hook "$repo" "$input" -- --allow allowed --deny denied
  if [ "$RC" -eq 0 ]; then
    mark_fail "creating a new file through a symlinked directory must not bypass deny"
  fi
  rm -rf "$repo"
}

test_symlink_escape_to_outside_repo_blocked_by_default_deny() {
  # A symlink in an allowed tree pointing outside the repo entirely
  # (e.g. to /etc/passwd).  Resolved path is outside the repo, so no
  # repo-rooted rule matches.  Under --default deny, this must fall
  # through to deny.  Under --default allow (the script's default),
  # it falls through to allow — same posture as a direct write to a
  # path outside the repo, which is consistent.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/allowed"
  ln -s "/etc/passwd" "$repo/allowed/escape"

  local input; input=$(mk_input_filepath "Write" "$repo/allowed/escape")
  run_hook "$repo" "$input" -- --default deny --allow allowed
  if [ "$RC" -eq 0 ]; then
    mark_fail "symlink escaping the repo with --default deny must be blocked"
  fi
  rm -rf "$repo"
}

test_allow_does_not_grant_via_obscuring_symlink() {
  # The allow direction is conservative: matching uses only the
  # resolved path, so a symlink pointing OUT of an allowed tree
  # does not have its destination granted by the allow rule.
  # Concretely: --allow safe, with safe/decoy → outside_safe/foo.
  # The literal path is in safe/ (would match allow under --ms
  # semantics), but the resolved path lands outside, so the allow
  # rule does NOT grant.  Under --default deny, the write is
  # blocked.
  local repo; repo=$(mk_repo)
  mkdir -p "$repo/safe" "$repo/outside_safe"
  : >"$repo/outside_safe/foo"
  ln -s "$repo/outside_safe/foo" "$repo/safe/decoy"

  local input; input=$(mk_input_filepath "Write" "$repo/safe/decoy")
  run_hook "$repo" "$input" -- --default deny --allow safe
  if [ "$RC" -eq 0 ]; then
    mark_fail "allow rule must not grant via a symlink whose target is outside the allowed tree"
  fi
  rm -rf "$repo"
}

# =============================================================================
# Group F — Tool integration
#
# Spec: "Edit/Write/Read use .tool_input.file_path; NotebookEdit uses
#        .tool_input.notebook_path; some legacy variants use
#        .tool_input.path."
#       "The same script services both matchers — only the tier rules
#        and the default change."
# =============================================================================

test_edit_tool_uses_file_path() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Edit" "$repo/docs/x.md")
  run_hook "$repo" "$input" -- --allow docs
  assert_rc 0
  rm -rf "$repo"
}

test_read_tool_uses_file_path_same_script() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_filepath "Read" "$repo/.env")
  run_hook "$repo" "$input" -- --deny .env
  if [ "$RC" -eq 0 ]; then
    mark_fail "Read on .env with --deny .env should be blocked"
  fi
  rm -rf "$repo"
}

test_notebookedit_tool_uses_notebook_path() {
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_notebookpath "NotebookEdit" "$repo/notebooks/x.ipynb")
  run_hook "$repo" "$input" -- --allow notebooks
  assert_rc 0
  rm -rf "$repo"
}

test_legacy_path_field_is_supported() {
  # Spec: "some legacy variants use .tool_input.path."
  local repo; repo=$(mk_repo)
  local input; input=$(mk_input_path "LegacyTool" "$repo/docs/x.md")
  run_hook "$repo" "$input" -- --allow docs
  assert_rc 0
  rm -rf "$repo"
}

test_missing_path_field_is_a_configuration_error() {
  # Spec (header docstring): "Bash / WebFetch / Glob / Grep have no
  # path field — if the matcher in the agent frontmatter accidentally
  # captures one of those tools, this hook exits 2 (configuration
  # error) rather than silently allowing."
  local repo; repo=$(mk_repo)
  local input; input=$(jq -cn '{tool_name: "Bash", tool_input: {command: "ls"}}')
  run_hook "$repo" "$input" -- --allow docs
  if [ "$RC" -eq 0 ]; then
    mark_fail "missing path field should not silently allow"
  fi
  rm -rf "$repo"
}

# =============================================================================
# Run all tests
# =============================================================================

run_test test_allow_decision_is_silent_exit_zero
run_test test_warn_decision_emits_permission_decision_allow_with_context
run_test test_deny_decision_blocks_with_nonzero_exit_and_stderr

run_test test_deny_on_subtree_beats_allow_on_parent
run_test test_deny_on_parent_beats_allow_on_child
run_test test_warn_beats_allow_when_both_match
run_test test_deny_beats_warn_when_both_match

run_test test_default_is_allow_when_no_rules_match
run_test test_default_deny_blocks_when_no_rules_match
run_test test_default_deny_still_permits_explicit_allow

run_test test_multiple_paths_after_one_flag_equal_repeated_flags
run_test test_default_with_invalid_value_fails

run_test test_missing_target_components_are_tolerated
run_test test_rule_dir_resolves_against_repo_root_not_cwd
run_test test_absolute_and_relative_rule_paths_are_equivalent
run_test test_leading_and_trailing_slashes_in_rule_are_optional
run_test test_trailing_slash_anchoring_distinguishes_sibling_directories
run_test test_target_outside_repo_root_falls_through_to_default
run_test test_target_outside_repo_root_blocked_by_default_deny
run_test test_symlink_in_denied_tree_pointing_to_allowed_target_is_denied
run_test test_symlink_in_allowed_tree_pointing_to_denied_target_is_denied
run_test test_new_file_via_symlinked_directory_into_denied_is_denied
run_test test_symlink_escape_to_outside_repo_blocked_by_default_deny
run_test test_allow_does_not_grant_via_obscuring_symlink

run_test test_edit_tool_uses_file_path
run_test test_read_tool_uses_file_path_same_script
run_test test_notebookedit_tool_uses_notebook_path
run_test test_legacy_path_field_is_supported
run_test test_missing_path_field_is_a_configuration_error

summarise
