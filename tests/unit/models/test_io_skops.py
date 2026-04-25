"""Spec-derived tests for the Stage 12 T1 skops helpers in ``bristol_ml.models.io``.

Every test here is derived from:

- ``docs/plans/active/12-serving.md`` §1 D10 (Ctrl+G reversal — adopt
  skops.io as the canonical save/load primitive).
- ``docs/plans/active/12-serving.md`` §6 T1 (named tests:
  ``test_save_skops_then_load_skops_roundtrips_pure_dict``,
  ``test_load_skops_raises_on_untrusted_type``,
  ``test_save_joblib_emits_deprecation_warning``).
- ``src/bristol_ml/models/io.py`` docstring (atomic write, parent-dir
  creation, ``UntrustedTypeError`` semantics, project trust-list).

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test;
surface the failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan AC / D-number / task number it
  guards.
- ``tmp_path`` (pytest built-in) is used for all filesystem operations.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from bristol_ml.models.io import (
    UntrustedTypeError,
    load_joblib,
    load_skops,
    register_safe_types,
    save_joblib,
    save_skops,
)

# ---------------------------------------------------------------------------
# T1 named test — save_skops + load_skops round-trip a pure dict envelope
# ---------------------------------------------------------------------------


def test_save_skops_then_load_skops_roundtrips_pure_dict(tmp_path: Path) -> None:
    """Guards Stage 12 T1 named test: skops round-trip preserves a dict envelope.

    The Stage 12 plan §6 T1 names this test as the foundational
    round-trip guard for the skops helpers — every model family's
    save/load path will eventually reduce to either a native skops dump
    of a custom class or a dict envelope of skops-safe primitives
    (numpy + bytes + ints; the pattern Stage 10/11 NN families
    pioneered, extended to ``linear`` / ``sarimax`` at T4).

    A plain dict ``{"a": [1, 2, 3], "b": "x"}`` round-trips through
    :func:`save_skops` + :func:`load_skops` with equality preserved.
    """
    payload = {"a": [1, 2, 3], "b": "x"}
    dest = tmp_path / "envelope.skops"

    save_skops(payload, dest)
    loaded = load_skops(dest)

    assert loaded == payload, (
        f"skops round-trip must return an equal object; got {loaded!r} "
        f"instead of {payload!r} (Stage 12 T1 named test)."
    )


# ---------------------------------------------------------------------------
# T1 — atomic write (no .tmp sibling after success)
# ---------------------------------------------------------------------------


def test_save_skops_writes_atomically(tmp_path: Path) -> None:
    """Guards Stage 12 T1: ``save_skops`` is atomic via tmp + ``os.replace``.

    The :mod:`bristol_ml.models.io` docstring says writes are atomic
    "in both APIs"; the joblib helper has had this discipline since
    Stage 4 and the skops helper inherits it.  After a successful
    write the target path exists and no ``.tmp`` sibling remains.
    """
    payload = {"key": "value"}
    dest = tmp_path / "artefact.skops"

    save_skops(payload, dest)

    files_in_dir = list(tmp_path.iterdir())
    tmp_files = [f for f in files_in_dir if f.suffix == ".tmp" or f.name.endswith(".tmp")]
    assert tmp_files == [], (
        f"No .tmp sibling must remain after a successful save_skops; "
        f"found {tmp_files!r} in {tmp_path} (Stage 12 T1 atomic-write contract)."
    )
    assert dest.exists(), f"Target path {dest} must exist after save_skops."


# ---------------------------------------------------------------------------
# T1 — parent directory is created automatically
# ---------------------------------------------------------------------------


def test_save_skops_creates_parent_directory(tmp_path: Path) -> None:
    """Guards :func:`save_skops` parent-dir creation.

    The ``save_skops`` docstring says: "The parent directory is created
    if missing — callers do not need to pre-create it."  A deeply
    nested path must succeed.
    """
    nested_dest = tmp_path / "nested" / "dir" / "file.skops"
    payload = {"deep": True}

    save_skops(payload, nested_dest)

    assert nested_dest.exists(), (
        f"save_skops must create the parent directory; "
        f"{nested_dest} does not exist (Stage 12 T1 parent-dir contract)."
    )
    loaded = load_skops(nested_dest)
    assert loaded == payload


# ---------------------------------------------------------------------------
# T1 — load raises FileNotFoundError on missing path
# ---------------------------------------------------------------------------


def test_load_skops_raises_on_missing_path(tmp_path: Path) -> None:
    """Guards :func:`load_skops` error contract on missing artefact.

    ``load_skops`` raises :class:`FileNotFoundError` (not a generic
    error) so callers can distinguish "registry layout has shifted"
    from "artefact contents are bad".
    """
    missing = tmp_path / "does-not-exist.skops"
    with pytest.raises(FileNotFoundError):
        load_skops(missing)


# ---------------------------------------------------------------------------
# T1 named test — load_skops raises on an untrusted type
# ---------------------------------------------------------------------------


class _UnregisteredTrustListClass:
    """A class deliberately *not* registered via :func:`register_safe_types`.

    Used to construct a skops artefact whose load must fail under the
    project trust-list because no model family has registered this
    class.  The class lives at module level so its qualified name is
    stable across test runs (skops resolves classes by dotted path).
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnregisteredTrustListClass) and self.value == other.value


def test_load_skops_raises_on_untrusted_type(tmp_path: Path) -> None:
    """Guards Stage 12 T1 named test: an unregistered custom class fails to load.

    The :func:`load_skops` contract: any type reported by
    :func:`skops.io.get_untrusted_types` that is **not** in the
    project trust-list raises :class:`UntrustedTypeError` naming both
    the artefact path and the unexpected types.

    This is the load-bearing security guarantee of the D10 reversal —
    a malicious actor cannot smuggle an arbitrary class through a
    skops artefact at the registry boundary, even if the artefact is
    syntactically valid skops.
    """
    obj = _UnregisteredTrustListClass(value=42)
    dest = tmp_path / "untrusted.skops"
    save_skops(obj, dest)

    # Sanity: the class is not in the project trust-list.  If it ever
    # is registered (e.g. someone moves this test class into the
    # project trust-list), this test should fail loudly so the security
    # contract is renegotiated explicitly rather than silently weakened.
    with pytest.raises(UntrustedTypeError) as excinfo:
        load_skops(dest)

    msg = str(excinfo.value)
    assert "trust-list" in msg, (
        f"UntrustedTypeError message must mention the trust-list; got {msg!r}"
    )
    assert str(dest) in msg, f"UntrustedTypeError message must name the artefact path; got {msg!r}"


def test_load_skops_succeeds_after_register_safe_types(tmp_path: Path) -> None:
    """Guards :func:`register_safe_types`: registering a type unblocks the load.

    Round-trip for the model layer is: import a family → the family
    calls :func:`register_safe_types` for its custom classes → the
    artefact then loads.  This test confirms the second hop of that
    chain works in isolation.
    """
    # Make a fresh local class so the registration does not leak
    # globally across other tests.  The class must live at module
    # level for skops to resolve it; we use a distinct name from
    # ``_UnregisteredTrustListClass`` so the registration is scoped
    # and the negative-path test above cannot be polluted.
    qualified = f"{_RegisteredViaTrustList.__module__}.{_RegisteredViaTrustList.__qualname__}"
    register_safe_types(qualified)

    obj = _RegisteredViaTrustList(value=7)
    dest = tmp_path / "trusted.skops"
    save_skops(obj, dest)

    loaded = load_skops(dest)

    assert loaded == obj, (
        f"After register_safe_types, load_skops must return an equal object; "
        f"got {loaded!r} instead of {obj!r}"
    )


class _RegisteredViaTrustList:
    """Companion class for the positive trust-list test.

    Lives at module level so its qualified name is stable across runs.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RegisteredViaTrustList) and self.value == other.value


# ---------------------------------------------------------------------------
# T1 named test — save_joblib + load_joblib emit DeprecationWarning
# ---------------------------------------------------------------------------


def test_save_joblib_emits_deprecation_warning(tmp_path: Path) -> None:
    """Guards Stage 12 T1 named test: ``save_joblib`` warns deprecated.

    The plan §6 T1 retains the joblib helpers for one stage so
    external scripts can complete a one-off migration, but emits a
    :class:`DeprecationWarning` on every call so any latent caller
    surfaces during CI.
    """
    payload = {"k": 1}
    dest = tmp_path / "x.joblib"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_joblib(payload, dest)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings, (
        "save_joblib must emit a DeprecationWarning per Stage 12 T1; "
        f"got {[type(w.category).__name__ for w in caught]!r}"
    )
    msg = str(deprecation_warnings[0].message)
    assert "save_skops" in msg, (
        f"DeprecationWarning message must point callers at save_skops; got {msg!r}"
    )


def test_load_joblib_emits_deprecation_warning(tmp_path: Path) -> None:
    """Guards Stage 12 T1: ``load_joblib`` warns deprecated.

    The Stage 12 T1 plan list pairs the save and load helpers; the
    load-side warning is implied by the same migration discipline so
    a separate test pins it down explicitly.
    """
    # Need a real artefact to load.  ``save_joblib`` itself emits the
    # warning so suppress it during fixture setup; the test under
    # examination is on the load side.
    payload = {"k": 1}
    dest = tmp_path / "x.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        save_joblib(payload, dest)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_joblib(dest)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings, "load_joblib must emit a DeprecationWarning per Stage 12 T1."
    msg = str(deprecation_warnings[0].message)
    assert "load_skops" in msg, (
        f"DeprecationWarning message must point callers at load_skops; got {msg!r}"
    )
    assert loaded == payload, (
        "load_joblib must still load the artefact (the deprecation does not "
        "break behaviour, only signals migration urgency)."
    )
