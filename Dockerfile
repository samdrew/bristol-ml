# syntax=docker/dockerfile:1.7
#
# Dev container for bristol_ml. Build via tmux-claude.sh so the image
# picks up your host UID/GID and the standard Claude Code host mounts.
#
# CUDA 12.8 is the first cuDNN-runtime tag with stable Blackwell (sm_120)
# support; it still works on Ada/Ampere/Hopper, so leave it as the default
# unless you have a reason to pin lower.

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages — minimal set for Claude Code plus a uv-managed Python
# project. Heavy runtime deps come from wheels via `uv sync`, not apt.
# ripgrep is required by Claude Code; libgomp1 is cheap insurance for the
# numpy/scipy/sklearn OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        htop \
        jq \
        less \
        libgomp1 \
        locales \
        nvtop \
        openssh-client \
        pkg-config \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        ripgrep \
        sudo \
        tmux \
        tree \
        unzip \
        vim \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Locale — en_GB.UTF-8.
RUN sed -i '/en_GB.UTF-8/s/^# //' /etc/locale.gen && locale-gen
ENV LANG=en_GB.UTF-8 LC_ALL=en_GB.UTF-8

# uv — copy the static binary from astral's official image. Pin the tag
# (e.g. `:0.5.x`) for bit-identical rebuilds; `:latest` is fine for dev.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# User setup. USER_NAME / USER_UID / USER_GID default to dev / 1000 / 1000;
# tmux-claude.sh overrides them with the host user's identity so bind-mounted
# file ownership matches inside and out. Ubuntu 24.04 ships an `ubuntu` user
# at 1000:1000 that collides with the common host UID, so we rename or
# replace any existing user in that slot rather than add alongside.
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

# Claude Code — native installer, no Node.js required.
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# uv: copy instead of hardlink (avoids warnings across bind-mount
# filesystems); pin the cache dir for named-volume mounts.
ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/${USER_NAME}/.cache/uv

# tmux — true-colour, mouse, snappy escape.
RUN printf '%s\n' \
    'set -g default-terminal "tmux-256color"' \
    'set -as terminal-features ",*:RGB"' \
    'set -g mouse on' \
    'set -sg escape-time 0' \
    > "$HOME/.tmux.conf"

WORKDIR /workspace
CMD ["tmux", "new-session", "-s", "claude", "/bin/bash"]
