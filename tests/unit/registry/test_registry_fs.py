"""Unit tests for the Stage 9 registry's on-disk helpers and public surface.

Covers the Stage 9 T1 test list:

- ``test_registry_public_surface_does_not_exceed_four_callables`` (AC-1)
- ``test_registry_build_run_id_format`` (D3 minute precision)
- ``test_registry_git_sha_helper_returns_str_in_git_tree`` (D13 / AC-3)

Plus a handful of defensive guards (naive-datetime rejection, git-helper
outside a git tree, atomic-write staging cleanup) that would be
painful to learn about in T2.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from bristol_ml import registry
from bristol_ml.registry._fs import _atomic_write_run, _build_run_id, _run_dir
from bristol_ml.registry._git import _git_sha_or_none
from bristol_ml.registry._schema import SidecarFields

# ---------------------------------------------------------------------------
# AC-1 — public surface cap
# ---------------------------------------------------------------------------


def test_registry_public_surface_does_not_exceed_four_callables() -> None:
    """The registry's ``__all__`` has exactly four members (AC-1 / D12)."""
    assert len(registry.__all__) == 4
    assert set(registry.__all__) == {"save", "load", "list_runs", "describe"}
    for name in registry.__all__:
        obj = getattr(registry, name)
        assert callable(obj), f"registry.{name} is not callable"


# ---------------------------------------------------------------------------
# D3 — run-ID format (minute precision)
# ---------------------------------------------------------------------------


def test_registry_build_run_id_format() -> None:
    """Run IDs use minute precision: ``{model_name}_{YYYYMMDDTHHMM}`` (D3)."""
    fit_utc = datetime(2026, 4, 23, 14, 30, 17, tzinfo=UTC)
    rid = _build_run_id("linear-ols-weather-only", fit_utc)
    assert rid == "linear-ols-weather-only_20260423T1430"
    assert re.match(r"^[a-z][a-z0-9_.-]*_\d{8}T\d{4}$", rid) is not None


def test_registry_build_run_id_drops_seconds() -> None:
    """Two fits in the same minute collapse to the same run_id (D2 last-write-wins)."""
    a = _build_run_id("m", datetime(2026, 4, 23, 14, 30, 0, tzinfo=UTC))
    b = _build_run_id("m", datetime(2026, 4, 23, 14, 30, 59, tzinfo=UTC))
    assert a == b == "m_20260423T1430"


def test_registry_build_run_id_normalises_non_utc_tz() -> None:
    """A fixed-offset (+01:00) fit_utc converts to UTC before formatting."""
    plus_one = timezone(timedelta(hours=1))
    rid = _build_run_id("m", datetime(2026, 4, 23, 15, 30, 0, tzinfo=plus_one))
    assert rid == "m_20260423T1430"


def test_registry_build_run_id_rejects_naive_datetime() -> None:
    """Naive datetimes are rejected; the registry never stores ambiguous wall-clock times."""
    with pytest.raises(ValueError, match="tz-aware"):
        _build_run_id("m", datetime(2026, 4, 23, 14, 30))


# ---------------------------------------------------------------------------
# D13 / AC-3 — git SHA helper
# ---------------------------------------------------------------------------


def test_registry_git_sha_helper_returns_str_in_git_tree() -> None:
    """Inside the project's git tree the helper returns a non-empty hex string (AC-3)."""
    sha = _git_sha_or_none()
    assert sha is not None, "expected a git SHA inside the project working tree"
    assert isinstance(sha, str)
    assert len(sha) >= 7, f"git SHA should be at least 7 hex chars, got {sha!r}"
    assert all(c in "0123456789abcdef" for c in sha.lower()), (
        f"git SHA should be lowercase hex, got {sha!r}"
    )


def test_registry_git_sha_helper_returns_none_outside_git_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a git tree the helper returns ``None`` rather than raising."""
    # pytest's tmp_path is under /tmp which has no .git ancestor.
    monkeypatch.chdir(tmp_path)
    assert _git_sha_or_none() is None


# ---------------------------------------------------------------------------
# D1 / D5 — on-disk layout + atomic write
# ---------------------------------------------------------------------------


def _sidecar(run_id: str) -> SidecarFields:
    """Minimal well-typed sidecar for atomic-write tests."""
    return SidecarFields(
        run_id=run_id,
        name="m",
        type="naive",
        feature_set="minimal",
        target="demand_mw",
        feature_columns=[],
        fit_utc="2026-04-23T14:30:17+00:00",
        git_sha=None,
        hyperparameters={},
        metrics={},
        registered_at_utc="2026-04-23T14:30:18+00:00",
    )


def test_atomic_write_run_produces_expected_layout(tmp_path: Path) -> None:
    """``_atomic_write_run`` creates ``{run_id}/artefact/model.joblib`` + ``run.json``."""
    run_id = "m_20260423T1430"
    final = _atomic_write_run(
        tmp_path,
        run_id,
        artefact_writer=lambda p: p.write_bytes(b"artefact-placeholder"),
        sidecar=_sidecar(run_id),
    )
    assert final == _run_dir(tmp_path, run_id)
    assert (final / "artefact" / "model.joblib").exists()
    assert (final / "run.json").exists()
    # No staging directories left behind.
    assert not list(tmp_path.glob(".tmp_*"))


def test_atomic_write_run_last_write_wins_on_collision(tmp_path: Path) -> None:
    """A second write to the same run_id overwrites the first (plan D2 / R7)."""
    run_id = "m_20260423T1430"
    _atomic_write_run(
        tmp_path,
        run_id,
        artefact_writer=lambda p: p.write_bytes(b"first"),
        sidecar=_sidecar(run_id),
    )
    _atomic_write_run(
        tmp_path,
        run_id,
        artefact_writer=lambda p: p.write_bytes(b"second"),
        sidecar=_sidecar(run_id),
    )
    final = _run_dir(tmp_path, run_id)
    assert (final / "artefact" / "model.joblib").read_bytes() == b"second"


def test_atomic_write_run_cleans_up_staging_on_error(tmp_path: Path) -> None:
    """If the artefact writer raises, the ``.tmp_*`` staging directory is removed."""

    def explode(_path: Path) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _atomic_write_run(
            tmp_path,
            "m_20260423T1430",
            artefact_writer=explode,
            sidecar=_sidecar("m_20260423T1430"),
        )
    assert not list(tmp_path.glob(".tmp_*"))
    assert not (tmp_path / "m_20260423T1430").exists()


# ---------------------------------------------------------------------------
# Stubs are wired (verbs import clean and raise NotImplementedError until later Ts)
# ---------------------------------------------------------------------------


def test_registry_verbs_load_list_describe_are_stubs_until_later_tasks() -> None:
    """T2 implemented ``save``; ``load`` / ``list_runs`` / ``describe`` fill in at T3/T4."""
    with pytest.raises(NotImplementedError):
        registry.load("any")
    with pytest.raises(NotImplementedError):
        registry.list_runs()
    with pytest.raises(NotImplementedError):
        registry.describe("any")


def test_registry_exposes_default_registry_dir() -> None:
    """``DEFAULT_REGISTRY_DIR`` is the documented module-level constant."""
    assert Path("data/registry") == registry.DEFAULT_REGISTRY_DIR
