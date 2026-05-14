#!/bin/bash
# Build the bristol_ml dev image and drop into a tmux session inside it.
#
# Host mounts default to the standard Claude Code locations; override via
# environment variables if yours live elsewhere:
#
#   CLAUDE_HOST_DIR     host source for ~/.claude        (default: $HOME/.claude)
#   CLAUDE_HOST_CONFIG  host source for ~/.claude.json   (default: $HOME/.claude.json)
#   GITCONFIG_HOST      host source for ~/.gitconfig     (default: $HOME/.gitconfig)
#   GPU_FLAGS           docker run GPU flags             (default: --gpus all;
#                                                         set empty to disable)

set -euo pipefail

SESSION_NAME="claude-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLAUDE_HOST_DIR="${CLAUDE_HOST_DIR:-$HOME/.claude}"
CLAUDE_HOST_CONFIG="${CLAUDE_HOST_CONFIG:-$HOME/.claude.json}"
GITCONFIG_HOST="${GITCONFIG_HOST:-$HOME/.gitconfig}"
GPU_FLAGS="${GPU_FLAGS---gpus all}"

# If not already in tmux, create/attach the session and re-exec ourselves
# inside it. `exec` replaces the current process — no separate exit path.
if [ -z "${TMUX:-}" ]; then
    exec tmux new-session -A -s "$SESSION_NAME" "$0" "$@"
fi

USER_NAME=$(id -un)
USER_UID=$(id -u)
USER_GID=$(id -g)

# Layer caching keeps this near-instant when nothing has changed; explicit
# rebuilds pick up Dockerfile edits without a separate command.
docker build \
    --build-arg USER_NAME="${USER_NAME}" \
    --build-arg USER_UID="${USER_UID}" \
    --build-arg USER_GID="${USER_GID}" \
    -t claude-dev:latest \
    "$SCRIPT_DIR"

docker run -it --rm \
    ${GPU_FLAGS} \
    -v "${CLAUDE_HOST_DIR}:/home/${USER_NAME}/.claude" \
    -v "${CLAUDE_HOST_CONFIG}:/home/${USER_NAME}/.claude.json" \
    -v "${GITCONFIG_HOST}:/home/${USER_NAME}/.gitconfig" \
    -v "$(pwd):/workspace" \
    -w /workspace \
    claude-dev:latest
