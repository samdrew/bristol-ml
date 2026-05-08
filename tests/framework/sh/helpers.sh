#!/usr/bin/env bash
#
# tests/framework/sh/helpers.sh — shared fixtures and assertions
# for shell-framework tests.  Sourced by individual test_*.sh files.
#
# Conventions:
#   - Each test is a function named `test_<short_description>`.
#   - Tests call `pass` / `fail` / `assert_*` helpers; failures are
#     accumulated rather than aborting the script, so a single test
#     run reports every failure.
#   - Each test sets up a fresh fixture via `mk_repo` to prevent
#     state leakage between cases.
#   - Tests that document a known spec/implementation drift use the
#     `expect_fail` wrapper to flip pass/fail semantics, so a green
#     run means "the drift is still present" — a red run means the
#     drift has been resolved (in either direction) and the test
#     needs revisiting.

set -uo pipefail

# --- Global state ------------------------------------------------------------

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_NAMES=()
EXPECTING_FAILURE=0   # set by `expect_fail` to suppress inline FAIL output

if [ -t 1 ]; then
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_DIM=$'\033[2m'
  C_OFF=$'\033[0m'
else
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_DIM=""; C_OFF=""
fi

# --- Pass / fail accounting --------------------------------------------------

pass() {
  TESTS_PASSED=$((TESTS_PASSED + 1))
  echo "  ${C_GREEN}PASS${C_OFF}  $CURRENT_TEST"
}

fail() {
  TESTS_FAILED=$((TESTS_FAILED + 1))
  FAILED_NAMES+=("$CURRENT_TEST")
  echo "  ${C_RED}FAIL${C_OFF}  $CURRENT_TEST"
  if [ "$#" -gt 0 ]; then
    echo "        $1"
  fi
}

run_test() {
  CURRENT_TEST="$1"
  TESTS_RUN=$((TESTS_RUN + 1))
  TEST_FAILED=0
  "$1"
  if [ "$TEST_FAILED" -eq 0 ]; then
    pass
  fi
}

# Used by assertions: mark the current test as failed without aborting.
# When wrapped in `expect_fail`, the inline FAIL line is suppressed —
# the wrapper renders the final verdict (XFAIL or UPASS) instead.
mark_fail() {
  if [ "$TEST_FAILED" -eq 0 ]; then
    TEST_FAILED=1
    if [ "$EXPECTING_FAILURE" -eq 0 ]; then
      fail "$1"
    fi
  else
    if [ "$EXPECTING_FAILURE" -eq 0 ]; then
      echo "        $1"
    fi
  fi
}

# Wrap a test that documents a known spec/implementation drift.
# Usage: expect_fail "reason" test_function_name
# - If the inner test passes, this reports FAIL (drift resolved → revisit).
# - If the inner test fails, this reports PASS (drift still present).
expect_fail() {
  local reason="$1"
  local fn="$2"
  CURRENT_TEST="$fn (xfail: $reason)"
  TESTS_RUN=$((TESTS_RUN + 1))
  TEST_FAILED=0
  EXPECTING_FAILURE=1
  "$fn"
  EXPECTING_FAILURE=0
  if [ "$TEST_FAILED" -eq 0 ]; then
    TESTS_FAILED=$((TESTS_FAILED + 1))
    FAILED_NAMES+=("$CURRENT_TEST")
    echo "  ${C_RED}UPASS${C_OFF} $CURRENT_TEST"
    echo "        Expected this test to fail, but it passed.  The"
    echo "        documented drift may have been resolved; revisit."
  else
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo "  ${C_YELLOW}XFAIL${C_OFF} $CURRENT_TEST"
  fi
}

summarise() {
  echo
  echo "Ran $TESTS_RUN test(s): ${C_GREEN}$TESTS_PASSED passed${C_OFF}, ${C_RED}$TESTS_FAILED failed${C_OFF}."
  if [ "$TESTS_FAILED" -gt 0 ]; then
    for n in "${FAILED_NAMES[@]}"; do
      echo "  - $n"
    done
    return 1
  fi
  return 0
}

# --- Fixture: isolated git repository ----------------------------------------

# Create a fresh git repo in a tempdir, echo its absolute path.
# Caller is responsible for cleanup via `rm -rf`.
mk_repo() {
  local dir
  dir=$(mktemp -d)
  (
    cd "$dir"
    git init --quiet
    git config user.email "test@example.com"
    git config user.name "Test"
  )
  echo "$dir"
}

# --- Fixture: hook input envelopes -------------------------------------------

# Build a Claude Code PreToolUse JSON envelope for an Edit/Write/Read
# tool call against a file_path.
mk_input_filepath() {
  local tool="$1"
  local path="$2"
  jq -cn --arg t "$tool" --arg p "$path" \
    '{tool_name: $t, tool_input: {file_path: $p}}'
}

# Build an envelope for NotebookEdit, which uses notebook_path.
mk_input_notebookpath() {
  local tool="$1"
  local path="$2"
  jq -cn --arg t "$tool" --arg p "$path" \
    '{tool_name: $t, tool_input: {notebook_path: $p}}'
}

# Build an envelope using the legacy `path` field.
mk_input_path() {
  local tool="$1"
  local path="$2"
  jq -cn --arg t "$tool" --arg p "$path" \
    '{tool_name: $t, tool_input: {path: $p}}'
}

# --- Hook invocation ---------------------------------------------------------

# Run the hook with given args, inside a given repo, with the given
# JSON envelope on stdin.  Sets RC, STDOUT, STDERR.
#
# Usage:
#   run_hook <repo_dir> <stdin_json> -- <hook_args...>
run_hook() {
  local repo="$1"
  local stdin="$2"
  shift 2
  if [ "$1" = "--" ]; then shift; fi

  local out_file err_file
  out_file=$(mktemp)
  err_file=$(mktemp)

  set +e
  printf '%s' "$stdin" | (cd "$repo" && "$HOOK" "$@") \
    >"$out_file" 2>"$err_file"
  RC=$?
  set -e

  STDOUT=$(cat "$out_file")
  STDERR=$(cat "$err_file")
  rm -f "$out_file" "$err_file"
}

# --- Assertions --------------------------------------------------------------

assert_rc() {
  local expected="$1"
  if [ "$RC" != "$expected" ]; then
    mark_fail "expected exit code $expected, got $RC.  STDERR: ${STDERR:-<empty>}"
    return 1
  fi
  return 0
}

assert_stdout_empty() {
  if [ -n "$STDOUT" ]; then
    mark_fail "expected empty stdout, got: $STDOUT"
    return 1
  fi
  return 0
}

assert_stdout_contains() {
  local needle="$1"
  case "$STDOUT" in
    *"$needle"*) return 0 ;;
    *)
      mark_fail "expected stdout to contain '$needle', got: $STDOUT"
      return 1
      ;;
  esac
}

assert_stderr_contains() {
  local needle="$1"
  case "$STDERR" in
    *"$needle"*) return 0 ;;
    *)
      mark_fail "expected stderr to contain '$needle', got: $STDERR"
      return 1
      ;;
  esac
}

assert_stdout_is_json_with() {
  # Argument: a jq filter that should evaluate to true on the stdout.
  local filter="$1"
  if ! echo "$STDOUT" | jq -e "$filter" >/dev/null 2>&1; then
    mark_fail "stdout did not satisfy jq filter '$filter'.  STDOUT: $STDOUT"
    return 1
  fi
  return 0
}
