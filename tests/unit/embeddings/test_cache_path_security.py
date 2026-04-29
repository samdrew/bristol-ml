"""Adversarial tests for the cache-path sanitiser + containment check.

Phase-3 code review B4 surfaced that the original
``_FILESYSTEM_SAFE`` regex permitted ``.`` (and therefore ``..``)
through, with no follow-up ``Path.resolve`` containment check on the
resulting cache path. The risk surface is small (the regex strips
``/`` already so a single ``..`` cannot escape on its own), but a
future regex change could open a path-traversal bug. The defence is
two layers:

1. The sanitiser strips a leading ``.`` so adversarial inputs cannot
   pass through as a dotted prefix that ``Path`` would interpret as
   parent-directory traversal once joined with the cache root.
2. ``_default_cache_path`` resolves the candidate path and asserts
   the cache root is one of its parents — a future regex regression
   surfaces here as a ``ValueError`` before any I/O happens.

These tests pin both layers so a future change cannot accidentally
weaken either one.
"""

from __future__ import annotations

from pathlib import Path

from bristol_ml.embeddings._factory import (
    _default_cache_path,
    _sanitised_model_id,
)


class TestSanitiser:
    """Plan §1 D14: the sanitiser maps non-filesystem-safe runs to ``_``."""

    def test_path_separators_become_underscores(self) -> None:
        assert _sanitised_model_id("Alibaba-NLP/gte-modernbert-base") == (
            "Alibaba-NLP_gte-modernbert-base"
        )

    def test_dotfile_prefix_is_stripped(self) -> None:
        # Phase-3 code review B4: a leading ``.`` would otherwise
        # become a valid sanitised name (``.`` survives the regex)
        # that, when joined with the cache root, parses as the
        # current-directory marker.
        assert not _sanitised_model_id(".secret-model").startswith(".")

    def test_double_dot_does_not_escape(self) -> None:
        # ``..`` survives the regex because both characters are in
        # the kept set. The ``lstrip('.')`` then removes the leading
        # dots so the result cannot be interpreted as a parent-
        # directory traversal once joined.
        sanitised = _sanitised_model_id("..")
        assert sanitised == "unnamed_model"

    def test_traversal_attempt_is_neutralised(self) -> None:
        # ``/`` becomes ``_`` so even a deliberate ``../../etc/passwd``
        # cannot survive as a traversal sequence — the ``/`` separators
        # are gone before ``Path`` ever sees the string.  Literal ``..``
        # *within* a single path segment is harmless (a filename like
        # ``_.._etc_passwd`` is just a filename); the load-bearing
        # invariants are "no path separators" and "no leading dot".
        sanitised = _sanitised_model_id("../../etc/passwd")
        assert "/" not in sanitised
        assert not sanitised.startswith(".")
        assert not sanitised.startswith("..")

    def test_empty_string_falls_back(self) -> None:
        assert _sanitised_model_id("") == "unnamed_model"

    def test_only_punctuation_falls_back(self) -> None:
        assert _sanitised_model_id("///") == "unnamed_model"


class TestDefaultCachePathContainment:
    """Phase-3 code review B4: containment check is load-bearing."""

    def test_normal_model_id_resolves_under_cache_root(self) -> None:
        path = _default_cache_path("Alibaba-NLP/gte-modernbert-base")
        # Walk up the resolved path until we find ``embeddings`` —
        # confirm it sits directly under ``data/embeddings/``.
        assert path.parent.name == "embeddings"
        assert path.parent.parent.name == "data"
        assert path.suffix == ".parquet"

    def test_resolved_path_lives_under_cache_root(self) -> None:
        # The candidate path's resolved form must be a descendant of
        # the resolved cache root — not merely a string-prefix match.
        path = _default_cache_path("benign-model")
        cache_root = (Path(__file__).resolve().parents[3] / "data" / "embeddings").resolve()
        assert cache_root in path.parents

    def test_dot_prefix_does_not_escape(self) -> None:
        # ``.escape`` strips to ``escape`` per the sanitiser; the
        # result lands under cache_root as expected.
        path = _default_cache_path(".escape")
        assert path.name == "escape.parquet"

    def test_pure_dots_fall_back_to_unnamed(self) -> None:
        # ``..`` -> ``unnamed_model`` -> ``data/embeddings/unnamed_model.parquet``.
        path = _default_cache_path("..")
        assert path.name == "unnamed_model.parquet"
