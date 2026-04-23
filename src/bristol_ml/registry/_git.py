"""Git-SHA capture helper for the registry sidecar (Stage 9 D13).

``_git_sha_or_none`` runs ``git rev-parse HEAD`` via :mod:`subprocess` at
save time.  The helper lives inside the registry module because git
provenance is the registry's concern — not the model's (plan §1 D13;
AC-3 "metadata captured automatically").  A 2 s timeout guards against a
pathologically slow git hook; a missing ``git`` executable or a
working directory outside any git tree returns ``None`` rather than
raising, because a ``None`` SHA is a documented registry state
(``ModelMetadata.git_sha`` already allows it).
"""

from __future__ import annotations

import subprocess


def _git_sha_or_none() -> str | None:
    """Return the current git HEAD SHA (full 40-char hex) or ``None``.

    Returns ``None`` when ``git`` is not on ``$PATH``, when the call times
    out, or when the working directory is not inside a git tree (including
    the CI-shallow-clone case where HEAD exists but ``rev-parse`` somehow
    fails — defensive rather than expected).

    The returned string is the raw ``git rev-parse HEAD`` output stripped of
    trailing whitespace; no length assertion is performed so a future
    ``--short`` convention can be absorbed by changing this call alone.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            timeout=2.0,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None
