#!/bin/bash
INPUT=$(cat)
PATH_ARG=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

if [ -z "$PATH_ARG" ]; then
  exit 0
fi

# Resolve to absolute, then check it's under tests/
ABS_PATH=$(realpath "$PATH_ARG" 2>/dev/null || echo "$PATH_ARG")
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

case "$ABS_PATH" in
  "$REPO_ROOT/tests/"*)
    exit 0
    ;;
  *)
    echo "Blocked: tester can only write to tests/. Refused: $PATH_ARG" >&2
    exit 2
    ;;
esac
