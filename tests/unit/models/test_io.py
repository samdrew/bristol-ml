"""Spec-derived tests for ``bristol_ml.models.io`` — joblib save/load helpers.

Every test is derived from:

- ``docs/plans/active/04-linear-baseline.md`` §6 Task T2 (acceptance criteria).
- ``docs/plans/active/04-linear-baseline.md`` §1 D6: atomic write via
  ``<name>.tmp`` + ``os.replace``; ``skops.io`` noted as the Stage 9 upgrade path.
- ``src/bristol_ml/models/io.py`` docstring (atomicity contract, parent-dir
  creation, ``FileNotFoundError`` on missing path).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan AC or D-number it guards.
- ``tmp_path`` (pytest built-in) is used for all filesystem operations.
- No ``xfail``, no ``skip``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bristol_ml.models.io import load_joblib, save_joblib

# ---------------------------------------------------------------------------
# T2 AC-3 foundation — basic round-trip
# ---------------------------------------------------------------------------


def test_joblib_round_trip(tmp_path: Path) -> None:
    """Guards T2 AC-3 foundation: save then load returns an equal object.

    Writes a plain dict ``{"a": [1, 2, 3], "b": "x"}`` to disk and
    reloads it, asserting the returned value equals the original.  This is
    the fundamental contract of ``save_joblib`` / ``load_joblib``: any
    Python object joblib can serialise is preserved through the round-trip.

    Guards T2 AC-3 (saving a fitted model and reloading it produces
    identical predictions; the io helpers are the foundation of that
    guarantee).
    """
    payload = {"a": [1, 2, 3], "b": "x"}
    dest = tmp_path / "artefact.joblib"

    save_joblib(payload, dest)
    loaded = load_joblib(dest)

    assert loaded == payload, (
        f"Round-trip must return an equal object; got {loaded!r} instead of {payload!r} "
        "(T2 AC-3 foundation)."
    )


# ---------------------------------------------------------------------------
# Plan D6 — atomic write (no .tmp sibling after success)
# ---------------------------------------------------------------------------


def test_joblib_round_trip_atomic(tmp_path: Path) -> None:
    """Guards Plan D6: after a successful write, no ``.tmp`` sibling remains.

    ``save_joblib`` writes atomically via a sibling ``<path>.tmp`` file then
    renames with ``os.replace``.  Once the call returns normally, the ``.tmp``
    file must have been removed and only the target path should exist in the
    directory.

    Guards Plan D6 (atomic write contract — T2 named test).
    """
    payload = {"key": "value"}
    dest = tmp_path / "artefact.joblib"

    save_joblib(payload, dest)

    files_in_dir = list(tmp_path.iterdir())
    tmp_files = [f for f in files_in_dir if f.suffix == ".tmp" or f.name.endswith(".tmp")]
    assert tmp_files == [], (
        f"No .tmp sibling must remain after a successful save_joblib; "
        f"found {tmp_files!r} in {tmp_path} (Plan D6 atomic write)."
    )
    assert dest.exists(), f"Target path {dest} must exist after save_joblib (Plan D6 atomic write)."
    loaded = load_joblib(dest)
    assert loaded == payload, (
        f"Round-trip after atomic write must return original payload; "
        f"got {loaded!r} (Plan D6 atomic write)."
    )


# ---------------------------------------------------------------------------
# io.py docstring contract — parent directory creation
# ---------------------------------------------------------------------------


def test_save_joblib_creates_parent_directory(tmp_path: Path) -> None:
    """Guards ``save_joblib`` parent-dir creation: nested path is created automatically.

    The ``save_joblib`` docstring states: "The parent directory is created if
    missing — callers do not need to pre-create it."  A deeply nested path
    (``tmp_path / "nested" / "dir" / "file.joblib"``) must succeed and the
    loaded value must equal the original.

    Guards io.py docstring contract (parent-directory creation).
    """
    nested_dest = tmp_path / "nested" / "dir" / "file.joblib"
    payload = {"deep": True}

    save_joblib(payload, nested_dest)

    assert nested_dest.exists(), (
        f"save_joblib must create the parent directory; "
        f"{nested_dest} does not exist (parent-dir creation contract)."
    )
    loaded = load_joblib(nested_dest)
    assert loaded == payload, (
        f"Round-trip through nested path must return original payload; "
        f"got {loaded!r} (parent-dir creation contract)."
    )


# ---------------------------------------------------------------------------
# Plan D6 — overwrite is atomic
# ---------------------------------------------------------------------------


def test_save_joblib_overwrites_existing_artefact_atomically(tmp_path: Path) -> None:
    """Guards Plan D6: writing v2 over v1 leaves v2 and no ``.tmp`` sibling.

    A second ``save_joblib`` call to the same path must atomically replace
    the first artefact.  After completion: the loaded value equals v2, and
    no ``.tmp`` file remains in the directory.

    Guards Plan D6 (atomic overwrite — idempotence, DESIGN §2.1.5).
    """
    dest = tmp_path / "model.joblib"
    v1 = {"version": 1}
    v2 = {"version": 2}

    save_joblib(v1, dest)
    save_joblib(v2, dest)

    loaded = load_joblib(dest)
    assert loaded == v2, (
        f"After overwrite, load must return v2={v2!r}; got {loaded!r} (Plan D6 atomic overwrite)."
    )

    tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert tmp_files == [], (
        f"No .tmp sibling must remain after atomic overwrite; "
        f"found {tmp_files!r} (Plan D6 atomic overwrite)."
    )


# ---------------------------------------------------------------------------
# io.py docstring contract — FileNotFoundError on missing path
# ---------------------------------------------------------------------------


def test_load_joblib_raises_on_missing_path(tmp_path: Path) -> None:
    """Guards ``load_joblib`` error contract: missing path raises ``FileNotFoundError``.

    The ``load_joblib`` docstring states: "Raises FileNotFoundError if
    ``path`` does not exist."  Attempting to load from a non-existent path
    must raise ``FileNotFoundError``, not silently return ``None`` or raise
    a different exception class.

    Guards io.py docstring contract (error path).
    """
    missing = tmp_path / "does-not-exist.joblib"

    with pytest.raises(FileNotFoundError):
        load_joblib(missing)
