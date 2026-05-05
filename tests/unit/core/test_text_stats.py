"""Unit tests for :func:`TEMPLATE_PROJECT.core.text_stats.compute_text_statistics`.

These tests exercise the pure-function contract directly — no Hydra,
no IO, no service plumbing.  They are the load-bearing regression
guard for the worked example.
"""

from __future__ import annotations

import pytest

from TEMPLATE_PROJECT.core.text_stats import TextStatistics, compute_text_statistics


def test_empty_string_yields_all_zeroes() -> None:
    stats = compute_text_statistics("")
    assert stats == TextStatistics(
        character_count=0,
        non_whitespace_character_count=0,
        word_count=0,
        line_count=0,
    )


def test_single_line_no_trailing_newline() -> None:
    stats = compute_text_statistics("hello world")
    assert stats.character_count == 11
    assert stats.non_whitespace_character_count == 10  # "helloworld"
    assert stats.word_count == 2
    assert stats.line_count == 1


def test_single_line_with_trailing_newline_still_counts_as_one_line() -> None:
    stats = compute_text_statistics("hello world\n")
    assert stats.line_count == 1


def test_multi_line() -> None:
    stats = compute_text_statistics("line one\nline two\nline three\n")
    assert stats.line_count == 3
    assert stats.word_count == 6  # 2 words per line x 3 lines


def test_unicode_characters_count_as_one_each() -> None:
    # Non-ASCII characters are counted as single characters
    stats = compute_text_statistics("café résumé")
    assert stats.character_count == 11  # spaces + accented chars
    assert stats.word_count == 2


def test_non_whitespace_excludes_tabs_and_newlines() -> None:
    stats = compute_text_statistics("a\tb\nc")
    # 5 chars total: 'a', '\t', 'b', '\n', 'c'
    # 3 non-whitespace: 'a', 'b', 'c'
    assert stats.character_count == 5
    assert stats.non_whitespace_character_count == 3


def test_returned_model_is_frozen() -> None:
    """The TextStatistics model must be immutable so callers can rely on it."""
    stats = compute_text_statistics("anything")
    with pytest.raises(Exception):  # noqa: B017 — Pydantic raises ValidationError on frozen mutation
        stats.word_count = 999  # type: ignore[misc]
