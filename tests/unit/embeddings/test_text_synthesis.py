"""Spec-derived tests for the Stage 15 T6 :func:`synthesise_embeddable_text`.

Every test here is derived from:

- ``docs/plans/active/15-embedding-index.md`` §6 T6 named test
  ``test_null_message_uses_structured_fallback``.
- Plan §1 D9: ``message_description`` is frequently NULL on the live
  Elexon stream endpoint; the embedder must accept NULL and synthesise
  embeddable text from the structured fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bristol_ml.embeddings import synthesise_embeddable_text

# ---------------------------------------------------------------------
# Plan §6 T6 — NULL message → structural fallback
# ---------------------------------------------------------------------


def test_null_message_uses_structured_fallback() -> None:
    """Plan §6 T6: NULL ``message_description`` → space-joined structural fields.

    Mirrors Stage 14's OQ-B resolution — a row with a NULL free-text
    field is rendered from ``(event_type, cause, fuel_type,
    affected_unit)`` so a downstream Stage 16 join sees a self-
    consistent provenance story.
    """
    row = {
        "message_description": None,
        "event_type": "Outage",
        "cause": "Planned",
        "fuel_type": "Nuclear",
        "affected_unit": "T_HARTLEPOOL-1",
    }
    text = synthesise_embeddable_text(row)
    assert text == "Outage Planned Nuclear T_HARTLEPOOL-1"


def test_non_null_message_returned_verbatim() -> None:
    """Plan §1 D9: non-NULL ``message_description`` is the embedded text."""
    row = {
        "message_description": "Planned outage for refuelling at Hartlepool 1.",
        "event_type": "Outage",
        "cause": "Planned",
        "fuel_type": "Nuclear",
        "affected_unit": "T_HARTLEPOOL-1",
    }
    text = synthesise_embeddable_text(row)
    assert text == "Planned outage for refuelling at Hartlepool 1."


def test_message_description_is_stripped() -> None:
    """Plan §1 D9: leading / trailing whitespace removed from the verbatim path."""
    row = {
        "message_description": "  Planned outage  \n",
        "event_type": "Outage",
        "cause": "Planned",
        "fuel_type": "Nuclear",
        "affected_unit": "X",
    }
    assert synthesise_embeddable_text(row) == "Planned outage"


@pytest.mark.parametrize(
    "null_value",
    [
        None,
        np.nan,
        pd.NA,
        pd.NaT,
        "",
        "   ",
    ],
)
def test_null_variants_all_trigger_fallback(null_value) -> None:
    """Plan §1 D9: pandas / numpy NULL representations all collapse to fallback.

    A pandas read of a parquet with nullable string columns can
    surface NULLs as :class:`pandas.NA`, :class:`numpy.nan`, or
    :class:`None` depending on dtype. All three must trigger the
    structural fallback. Empty / whitespace-only strings are also
    treated as missing because the embedder can't extract signal
    from them.
    """
    row = {
        "message_description": null_value,
        "event_type": "Outage",
        "cause": "Planned",
        "fuel_type": "Nuclear",
        "affected_unit": "T_HARTLEPOOL-1",
    }
    text = synthesise_embeddable_text(row)
    assert text == "Outage Planned Nuclear T_HARTLEPOOL-1"


def test_structural_fallback_skips_null_subfields() -> None:
    """Plan §1 D9: NULL structural fields are skipped silently in the join."""
    row = {
        "message_description": None,
        "event_type": "Outage",
        "cause": None,  # missing
        "fuel_type": "Wind",
        "affected_unit": None,  # missing
    }
    text = synthesise_embeddable_text(row)
    assert text == "Outage Wind"


def test_fully_null_row_returns_empty_string() -> None:
    """Plan §1 D9: an all-NULL row produces empty text (a stable, distinct output).

    The embedder turns "" into a deterministic vector (the QUERY_PREFIX
    on the query path makes it distinguishable from a different
    all-NULL row's document path). Empty is the stable output, not
    a raise.
    """
    row = {
        "message_description": None,
        "event_type": None,
        "cause": None,
        "fuel_type": None,
        "affected_unit": None,
    }
    assert synthesise_embeddable_text(row) == ""


def test_pandas_series_row_accepted() -> None:
    """The synthesiser accepts a Pandas Series — the iteration site uses ``df.iterrows``."""
    series = pd.Series(
        {
            "message_description": None,
            "event_type": "Restriction",
            "cause": "Forced",
            "fuel_type": "Gas",
            "affected_unit": "T_PEMBROKE-1",
        }
    )
    text = synthesise_embeddable_text(series)
    assert text == "Restriction Forced Gas T_PEMBROKE-1"


def test_extra_row_columns_ignored() -> None:
    """Robustness: a row with additional columns (e.g. mrid) does not affect output.

    The synthesiser reads only the five named fields; extra columns
    on the Stage 13 ``OUTPUT_SCHEMA`` row (``mrid``, ``revision_number``,
    timestamps, etc.) are silently ignored because their values are
    not embeddable as natural-language text.
    """
    row = {
        "mrid": "M-1",
        "revision_number": 0,
        "published_at": pd.Timestamp("2024-01-01", tz="UTC"),
        "message_description": "verbatim text",
        "event_type": "Outage",
        "cause": "Planned",
        "fuel_type": "Nuclear",
        "affected_unit": "X",
    }
    assert synthesise_embeddable_text(row) == "verbatim text"
