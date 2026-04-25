"""Save / load helpers for model artefacts.

Stage 4 introduced this module with :mod:`joblib` as the default
serialiser, with a documented forward-look to :mod:`skops.io` "at
Stage 12 when the serving layer lands".  Stage 12 is here, and the
human's Ctrl+G review reversed the original D10 disposition: every
model family now saves and loads through :mod:`skops.io`, because the
serving layer is a network-facing deserialiser and ``joblib.load`` on
an attacker-controlled artefact is an RCE vector.

The two skops helpers below are the canonical primitives every model
family flips to in T2-T4 of the Stage 12 plan:

- :func:`save_skops` — atomic skops write (tmp + ``os.replace``);
  mirrors the ingestion-layer ``_atomic_write`` idiom.
- :func:`load_skops` — skops load that **enforces a project-level
  trust list**.  :func:`skops.io.get_untrusted_types` is invoked first;
  any reported type that is not registered via
  :func:`register_safe_types` raises :class:`UntrustedTypeError`
  naming both the artefact path and the unexpected types.  The model
  layer's concrete classes register themselves on import.

The joblib helpers (:func:`save_joblib`, :func:`load_joblib`) are
preserved for one stage with a :class:`DeprecationWarning`, so any
external scripts can complete a one-off migration before Stage 13.
They will be removed in Stage 13 (no exceptions — joblib at the
registry boundary is a security regression).

Writes are **atomic** in both APIs — write to a sibling ``<path>.tmp``
file, then rename via :func:`os.replace` (the portable atomic-rename
primitive on POSIX + NTFS).  A crash mid-write leaves the previous
artefact intact rather than producing a zero-byte file.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import skops.io as sio

__all__ = [
    "UntrustedTypeError",
    "load_joblib",
    "load_skops",
    "register_safe_types",
    "save_joblib",
    "save_skops",
]


class UntrustedTypeError(RuntimeError):
    """Raised when a skops artefact contains a type outside the project trust-list.

    The error message names both the artefact path and the unexpected
    types so the operator can either (i) register the new type via
    :func:`register_safe_types` if it is legitimate, or (ii) treat the
    artefact as malicious and refuse to load it.
    """


# Project-level trust list: fully-qualified names of custom classes the
# project's ``Model`` implementations need.  Each model family
# registers its own classes here at import time via
# :func:`register_safe_types`.  The skops library trusts a generous set
# of primitive types out of the box (numpy, pandas, builtins,
# statsmodels' standard lib usage); the trust-list below only needs to
# cover the project's *custom* classes.
_PROJECT_SAFE_TYPES: set[str] = set()


def register_safe_types(*qualified_names: str) -> None:
    """Add fully-qualified type names to the project trust-list.

    Each model family calls this at import time for any class whose
    ``__module__.__qualname__`` appears in the saved artefact.  The
    trust-list is used both to gate :func:`load_skops` (artefacts
    containing an unregistered type are rejected) and to satisfy
    :func:`skops.io.load`'s ``trusted=`` argument.

    Parameters
    ----------
    *qualified_names:
        Fully-qualified class names in ``module.path.ClassName`` form.
        Order does not matter; duplicates are deduplicated by the
        underlying set.
    """
    _PROJECT_SAFE_TYPES.update(qualified_names)


def save_skops(obj: Any, path: Path) -> None:
    """Serialise ``obj`` to ``path`` atomically using :mod:`skops.io`.

    Writes to a sibling ``<path>.tmp`` first and then renames with
    :func:`os.replace`, so a crash mid-write cannot corrupt an existing
    artefact.  The parent directory is created if missing — callers do
    not need to pre-create it.

    Parameters
    ----------
    obj:
        Any object skops can serialise.  For Stage 12 onwards this is
        either a concrete :class:`bristol_ml.models.Model` instance
        (where the family's custom classes are registered via
        :func:`register_safe_types`) or a dict envelope of skops-safe
        primitives (the pattern Stage 10/11 NN families use, extended
        to ``linear`` / ``sarimax`` at Stage 12 T4).
    path:
        Destination artefact path.  Convention is ``.skops`` suffix but
        not enforced here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sio.dump(obj, tmp)
    os.replace(tmp, path)


def load_skops(path: Path) -> Any:
    """Deserialise a skops artefact, enforcing the project trust-list.

    The load proceeds in three steps:

    1. :func:`skops.io.get_untrusted_types` is invoked on the artefact.
       Skops's default trust-set covers the standard primitives
       (numpy, pandas, builtins, etc.); anything beyond that is
       reported here.
    2. Any reported type *not* in the project trust-list (built up via
       :func:`register_safe_types`) raises :class:`UntrustedTypeError`,
       naming both the artefact path and the unexpected types so the
       operator can decide whether to register the new type or refuse
       the load entirely.
    3. Otherwise, :func:`skops.io.load` is called with the project
       trust-list passed as ``trusted=``.  This is the load-bearing
       hop: skops will refuse to materialise an untrusted type even
       after :func:`get_untrusted_types` reports it, unless the type
       is explicitly listed in ``trusted=``.

    Parameters
    ----------
    path:
        Artefact path previously written by :func:`save_skops`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    UntrustedTypeError
        If the artefact contains a type that is not in the project
        trust-list.
    """
    if not path.is_file():
        raise FileNotFoundError(f"skops artefact not found at {path!s}.")
    untrusted = sio.get_untrusted_types(file=path)
    unexpected = sorted(t for t in untrusted if t not in _PROJECT_SAFE_TYPES)
    if unexpected:
        raise UntrustedTypeError(
            f"skops artefact at {path!s} contains untrusted types not in the "
            f"project trust-list: {unexpected!r}. Either register the new type "
            "via bristol_ml.models.io.register_safe_types (if it is a known "
            "project class added since this artefact was written) or treat "
            "the artefact as malicious and refuse to load it."
        )
    return sio.load(path, trusted=sorted(_PROJECT_SAFE_TYPES))


def save_joblib(obj: Any, path: Path) -> None:
    """[DEPRECATED] Atomic joblib write — use :func:`save_skops` instead.

    Stage 12 (D10 — Ctrl+G reversal) migrated all model families to
    :mod:`skops.io` for security: the serving layer is a network-facing
    deserialiser, and ``joblib.load`` on an attacker-controlled
    artefact is RCE.  This helper is retained for one stage so any
    external scripts can complete a one-off migration before Stage 13;
    it will be removed at the next release.

    The atomic-write semantics (tmp file + :func:`os.replace`) are
    preserved so existing callers behave identically apart from the
    deprecation warning.
    """
    warnings.warn(
        "bristol_ml.models.io.save_joblib is deprecated; "
        "Stage 12 D10 migrated all model families to skops.io for security. "
        "Use save_skops instead. This helper will be removed in Stage 13.",
        DeprecationWarning,
        stacklevel=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def load_joblib(path: Path) -> Any:
    """[DEPRECATED] Joblib load — use :func:`load_skops` instead.

    Stage 12 (D10 — Ctrl+G reversal) migrated all model families to
    :mod:`skops.io` for security: the serving layer is a network-facing
    deserialiser, and ``joblib.load`` on an attacker-controlled
    artefact is RCE.  This helper is retained for one stage so any
    external scripts can complete a one-off migration before Stage 13;
    it will be removed at the next release.

    Note that :func:`load_joblib` does **not** enforce a trust-list —
    joblib has no equivalent of skops's untrusted-type inspection — so
    callers loading artefacts from any source must migrate to
    :func:`load_skops` before Stage 13.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    warnings.warn(
        "bristol_ml.models.io.load_joblib is deprecated; "
        "Stage 12 D10 migrated all model families to skops.io for security. "
        "Use load_skops instead. This helper will be removed in Stage 13.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joblib.load(path)
