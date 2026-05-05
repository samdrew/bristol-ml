"""Text statistics — the worked-example core module.

Demonstrates the template's ``core/`` layer pattern: a single pure
function that takes a primitive input (``str``) and returns an
immutable Pydantic model with summary statistics.  No IO; no Hydra
dependency; trivially unit-testable.

A real project replaces this module with its actual domain logic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TextStatistics", "compute_text_statistics"]


class TextStatistics(BaseModel):
    """Immutable summary of a text input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    character_count: int = Field(ge=0)
    non_whitespace_character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    line_count: int = Field(ge=0)


def compute_text_statistics(text: str) -> TextStatistics:
    """Return :class:`TextStatistics` for ``text``.

    Definitions:

    - ``character_count`` — total characters including whitespace and
      newlines (i.e. ``len(text)``).
    - ``non_whitespace_character_count`` — characters excluding any
      character for which :py:meth:`str.isspace` returns ``True``.
    - ``word_count`` — number of whitespace-separated tokens (i.e.
      ``len(text.split())``).
    - ``line_count`` — number of newline-separated lines.  An empty
      string yields ``0`` lines; every non-empty input yields at
      least ``1`` line, regardless of trailing newline presence.
    """
    # ``splitlines()`` does not produce a trailing empty entry for a
    # final newline ("a\n" -> ["a"]), matching the conventional
    # ``wc -l + 1 if no trailing newline`` behaviour without a special
    # case.  An empty input yields 0 lines; any non-empty input yields
    # at least 1.
    line_count = 0 if text == "" else len(text.splitlines()) or 1
    return TextStatistics(
        character_count=len(text),
        non_whitespace_character_count=sum(1 for c in text if not c.isspace()),
        word_count=len(text.split()),
        line_count=line_count,
    )
