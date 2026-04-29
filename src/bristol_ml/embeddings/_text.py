"""Stage 15 T6 — embeddable-text synthesis from a Stage 13 REMIT row.

Plan §1 D9: ``message_description`` is **frequently NULL** on the live
Elexon stream endpoint (``ingestion/remit.py:431-436``); Stage 14's
:class:`~bristol_ml.llm.RemitEvent` already documents this in its
field comment. Stage 15 inherits the constraint — embedding ``None``
text is meaningless, so when ``message_description`` is NULL the
embedder synthesises a short descriptive string from the structured
fields available on the same row (``event_type`` + ``cause`` +
``fuel_type`` + ``affected_unit``).

Mirroring Stage 14's OQ-B resolution (``StubExtractor`` /
``LlmExtractor`` synthesise structured-field defaults in the same
shape) keeps the two LLM-adjacent layers consistent: a row that the
extractor handled by structural-field synthesis will be embedded
from the same fields here, so a downstream join (Stage 16) sees a
self-consistent provenance story.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

__all__ = [
    "synthesise_embeddable_text",
]


def _coerce_to_text(value: Any) -> str | None:
    """Render a Stage 13 cell value to text, or return ``None`` if absent.

    The corpus comes through ``pandas.read_parquet`` so a NULL is
    typically ``pd.NA`` / ``np.nan`` / ``None`` depending on dtype;
    each must collapse to ``None`` here. Everything else (string,
    numeric, datetime) is rendered via ``str()`` and stripped.

    ``pandas.isna`` recognises every NULL flavour pandas / numpy can
    produce — :class:`pandas.NA`, :class:`pandas.NaT`,
    :func:`numpy.nan`, :class:`None` — and avoids the
    ``pd.NA != pd.NA`` ambiguity (which raises rather than returning
    ``True``). It returns array-of-bool for sequences; we guard the
    scalar path explicitly because a list-typed cell is passed
    through to ``str()`` instead.
    """
    if value is None:
        return None
    try:
        # ``pd.isna`` returns ``True`` for the scalar NULL flavours
        # and ``False`` for genuine values; on a sequence-typed input
        # it returns an array, which we treat as not-NULL (the
        # ``bool(array)`` branch raises and we fall through to the
        # ``str()`` path).
        is_na = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover — defensive
        is_na = False
    if isinstance(is_na, bool) and is_na:
        return None
    text = str(value).strip()
    return text if text else None


def synthesise_embeddable_text(row: Mapping[str, Any]) -> str:
    """Return embeddable text for one REMIT row.

    Plan §1 D9: prefer ``message_description`` when non-NULL; else
    join the available structured fields with single spaces.

    The synthesised fallback is intentionally short and unpunctuated
    — the embedder tokeniser handles the rest. Empty fields are
    skipped silently (a row with all structural fields NULL still
    produces a stable, distinct output: the empty string, which the
    embedder turns into a deterministic vector).

    Parameters
    ----------
    row
        Any mapping with the Stage 13 ``OUTPUT_SCHEMA`` columns.
        Pandas Series accepted; row-of-DataFrame.iloc[i] works
        directly.

    Returns
    -------
    str
        Either ``message_description`` verbatim (stripped) when
        present, or a space-joined synthesis from
        ``(event_type, cause, fuel_type, affected_unit)`` when not.

    Notes
    -----
    The Stage 14 :class:`~bristol_ml.llm.RemitEvent` boundary type
    does not carry ``affected_unit`` (Stage 14 dropped it from the
    extraction-relevant subset). Stage 15 reads the raw Stage 13
    parquet, which *does* carry it; the synthesis includes it when
    available because the BMU id is a strong semantic signal in the
    REMIT corpus (e.g. ``"T_HARTLEPOOL-1"``) that the extractor's
    free-text field would have mentioned otherwise.
    """
    description = _coerce_to_text(row.get("message_description"))
    if description is not None:
        return description

    parts: list[str] = []
    for field in ("event_type", "cause", "fuel_type", "affected_unit"):
        coerced = _coerce_to_text(row.get(field))
        if coerced is not None:
            parts.append(coerced)
    return " ".join(parts)
