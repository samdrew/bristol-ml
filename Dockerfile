# syntax=docker/dockerfile:1.7
#
# Dev container for TEMPLATE_PROJECT — a Python project scaffold
# (Hydra+Pydantic config, Claude Code agent infrastructure, four-tier
# doc methodology) driven primarily through Claude Code.
#
# CPU-first: the base is `python:3.12-slim`.  If your project needs
# GPU support (PyTorch CUDA, etc.), swap the base to
# `nvidia/cuda:<version>-cudnn-runtime-<ubuntu-tag>` and add the
# matching system packages and `uv` index — see the bristol_ml repo
# for the worked GPU example.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------
# System packages
# ---------------------------------------------------------------
# Minimal set for Claude Code plus a uv-managed Python project.  Heavy
# runtime deps come from wheels via `uv sync` at project time, not
# from apt.
#
#   build-essential + pkg-config : fallback path for any sdist builds.
#   ca-certificates, curl, git    : Claude Code + uv require these.
#   ripgrep                       : Claude Code's in-repo search.
#   less, openssh-client          : quality-of-life for git.
#   tmux, jq                      : interactive workflow + JSON munging.
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        locales \
        openssh-client \
        pkg-config \
        ripgrep \
        sudo \
        tmux \
        tree \
        unzip \
        vim \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Locale — en_GB.UTF-8
# ---------------------------------------------------------------
RUN sed -i '/en_GB.UTF-8/s/^# //' /etc/locale.gen && locale-gen
ENV LANG=en_GB.UTF-8 LC_ALL=en_GB.UTF-8

# ---------------------------------------------------------------
# uv — project / venv / Python manager
# ---------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ---------------------------------------------------------------
# User setup
# ---------------------------------------------------------------
# USER_NAME / USER_UID / USER_GID are overridable at build time so
# bind-mounted volumes have matching ownership inside and out.
ARG USER_NAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN if getent passwd ${USER_UID} >/dev/null; then \
        userdel -r $(getent passwd ${USER_UID} | cut -d: -f1); \
    fi && \
    if getent group ${USER_GID} >/dev/null; then \
        groupmod -n ${USER_NAME} $(getent group ${USER_GID} | cut -d: -f1); \
    else \
        groupadd -g ${USER_GID} ${USER_NAME}; \
    fi && \
    useradd -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    mkdir -p /workspace && chown ${USER_UID}:${USER_GID} /workspace

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# ---------------------------------------------------------------
# Claude Code — native installer, no Node.js required
# ---------------------------------------------------------------
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# ---------------------------------------------------------------
# uv hints
# ---------------------------------------------------------------
ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/${USER_NAME}/.cache/uv

# ---------------------------------------------------------------
# tmux config — true-colour, mouse, snappy vim
# ---------------------------------------------------------------
RUN printf '%s\n' \
    'set -g default-terminal "tmux-256color"' \
    'set -as terminal-features ",*:RGB"' \
    'set -g mouse on' \
    'set -sg escape-time 0' \
    > "$HOME/.tmux.conf"

WORKDIR /workspace
CMD ["tmux", "new-session", "-s", "claude", "/bin/bash"]
