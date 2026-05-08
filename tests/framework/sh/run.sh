#!/usr/bin/env bash
#
# tests/framework/sh/run.sh — entry point for shell-framework tests.
#
# Discovers and runs every executable test_*.sh file in this
# directory.  Each test script is responsible for its own setup,
# teardown, and reporting; this runner only aggregates exit codes.
#
# Usage:
#   tests/framework/sh/run.sh              # run all
#   tests/framework/sh/run.sh test_foo.sh  # run one
#
# Exit code: 0 if every test script exited 0, 1 otherwise.

set -uo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

if [ "$#" -gt 0 ]; then
  TESTS=("$@")
else
  mapfile -t TESTS < <(find . -maxdepth 1 -name 'test_*.sh' -type f | sort)
fi

if [ "${#TESTS[@]}" -eq 0 ]; then
  echo "No test_*.sh files found in $DIR" >&2
  exit 1
fi

FAILED=0
for t in "${TESTS[@]}"; do
  echo
  echo "=== ${t#./} ==="
  if bash "$t"; then
    :
  else
    FAILED=$((FAILED + 1))
  fi
done

echo
if [ "$FAILED" -eq 0 ]; then
  echo "All test scripts passed."
  exit 0
else
  echo "$FAILED test script(s) failed."
  exit 1
fi
