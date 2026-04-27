"""Build notebooks/14_llm_extractor.ipynb programmatically.

Stage 14 Task T6 notebook deliverable — follows the six-cell recipe in
``docs/plans/completed/14-llm-extractor.md`` §5 (Notebook structure) +
§6 T6.

Generating the notebook from a Python script keeps cell source under
version control as readable text (matches Stage 13's pattern). The
three-step regeneration flow is::

    uv run python scripts/_build_notebook_14.py
    uv run jupyter nbconvert --execute --to notebook --inplace \\
        notebooks/14_llm_extractor.ipynb
    uv run ruff format notebooks/14_llm_extractor.ipynb

Plan §6 T6 says "CI default is BRISTOL_ML_LLM_STUB=1; cassette absence
printed banner on Cell 4." The notebook does both: the stub path runs
the gold set end-to-end (cells 1, 2, 3, 5, 6); the live path (cell 4)
catches missing API key / cassette and prints a banner explaining how
to record one. Either way the notebook executes top-to-bottom and the
side-by-side comparison (cell 5) renders.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "14_llm_extractor.ipynb"


_CELL_COUNTER = 0


def _next_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:02d}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id("md"),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell 0 — title + abstract
# ---------------------------------------------------------------------------

cell_0 = md(
    """# Stage 14 — LLM feature extractor

This notebook is the demo surface for the Stage 14 extractor
``bristol_ml.llm`` and its evaluation harness
``bristol_ml.llm.evaluate``.  It answers one question with one
side-by-side table: **how often does the LLM agree with the
hand-labelled set, and where does it disagree?**

The demo moment (intent §"Demo moment") is the per-field summary
plus the disagreement listing. The exact-vs-tolerance split is the
pedagogical pause: the same data produces different numbers under
different metric choices, which is the lesson.

The mechanic that makes the demo runnable in any environment — with
or without an API key — is the stub-first discipline plus the env-var
discriminator:

- ``BRISTOL_ML_LLM_STUB=1`` forces the offline stub path
  (default in CI / notebooks).
- ``llm.type=openai`` + a populated ``BRISTOL_ML_LLM_API_KEY`` opt
  into the live path; the integration test replays a recorded VCR
  cassette so CI never touches the network.

- **Intent:** `docs/intent/14-llm-extractor.md`.
- **Plan:** `docs/plans/completed/14-llm-extractor.md`.
- **Module:** `src/bristol_ml/llm/`.

CI runs this notebook against the gold-set fixture
(``BRISTOL_ML_LLM_STUB=1``, 76 hand-labelled records) — deterministic,
offline, no API call. A live run hits OpenAI's ``chat.completions``
endpoint via the recorded cassette and produces the same shape of
output with real LLM disagreements visible.
"""
)


# ---------------------------------------------------------------------------
# Cell 1 — bootstrap + load config (T5 Cell 1 — naming convention from plan)
# ---------------------------------------------------------------------------

cell_1 = code(
    """# T5 Cell 1 — Bootstrap, locate the repo root, load config.
import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import json  # noqa: E402

import pandas as pd  # noqa: E402

from bristol_ml import load_config  # noqa: E402
from bristol_ml.llm import RemitEvent  # noqa: E402
from bristol_ml.llm.evaluate import evaluate, format_report  # noqa: E402
from bristol_ml.llm.extractor import (  # noqa: E402
    DEFAULT_GOLD_SET_PATH,
    StubExtractor,
    build_extractor,
)

# Compose the +llm=extractor group; conf/config.yaml does not include
# it in defaults (parallel to ServingConfig).  The notebook always
# wants the LLM config slot populated.
cfg = load_config(config_path=REPO_ROOT / "conf", overrides=["+llm=extractor"])
assert cfg.llm is not None, "LLM config not resolved"
print(f"llm.type:        {cfg.llm.type}")
print(f"llm.model_name:  {cfg.llm.model_name}")
print(f"llm.prompt_file: {cfg.llm.prompt_file}")
"""
)


# ---------------------------------------------------------------------------
# Cell 2 — load the hand-labelled gold set (T5 Cell 2)
# ---------------------------------------------------------------------------

cell_2 = code(
    """# T5 Cell 2 — Load the hand-labelled gold set; print size + per-fuel breakdown.
gold_set_payload = json.loads(DEFAULT_GOLD_SET_PATH.read_text(encoding="utf-8"))
records = gold_set_payload["records"]
print(f"Gold set:        {DEFAULT_GOLD_SET_PATH}")
print(f"schema_version:  {gold_set_payload['schema_version']}")
print(f"records:         {len(records)}")

gold_df = pd.DataFrame(
    [
        {
            "mrid": r["event"]["mrid"],
            "fuel_type": r["expected"]["fuel_type"],
            "event_type": r["expected"]["event_type"],
            "affected_capacity_mw": r["expected"]["affected_capacity_mw"],
        }
        for r in records
    ]
)
print()
print("Per-fuel-type breakdown:")
print(gold_df["fuel_type"].value_counts().to_string())
"""
)


# ---------------------------------------------------------------------------
# Cell 3 — run the stub on the gold set (T5 Cell 3)
# ---------------------------------------------------------------------------

cell_3 = code(
    """# T5 Cell 3 — Run the stub extractor; show first 3 results inline.
# Force-stub independent of the YAML default — keeps the cell deterministic
# even if a future YAML edit flips the default.
os.environ["BRISTOL_ML_LLM_STUB"] = "1"
stub = build_extractor(cfg.llm)
print(f"Active extractor: {type(stub).__name__}")
print(f"Gold-set size:    {stub.gold_set_size if isinstance(stub, StubExtractor) else 'n/a'}")
print()

sample_events = [RemitEvent(**r["event"]) for r in records[:3]]
sample_results = stub.extract_batch(sample_events)
for ev, res in zip(sample_events, sample_results, strict=True):
    print(
        f"{ev.mrid}: event_type={res.event_type:<11s}  "
        f"fuel_type={res.fuel_type:<10s}  "
        f"capacity={res.affected_capacity_mw}  "
        f"confidence={res.confidence}"
    )
"""
)


# ---------------------------------------------------------------------------
# Cell 4 — try the live LLM via VCR cassette (T5 Cell 4)
# ---------------------------------------------------------------------------

cell_4 = code(
    """# T5 Cell 4 — Try the live LlmExtractor; print a banner if the cassette is absent.
# The cassette lives at tests/fixtures/llm/cassettes/ and is replayed by
# pytest-recording in the integration test.  In a notebook context we don't
# have VCR running, so this cell tries to construct LlmExtractor and reports
# whether the API path *would* run if invoked.  The banner explains how to
# record a cassette.
cassette_path = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "llm"
    / "cassettes"
    / "test_llm_extractor_against_cassette.yaml"
)
api_key_present = bool(os.environ.get(cfg.llm.api_key_env_var, "").strip())
stub_forced = os.environ.get("BRISTOL_ML_LLM_STUB") == "1"

if cassette_path.exists():
    print(f"VCR cassette: {cassette_path}  ({cassette_path.stat().st_size:,} bytes)")
else:
    print(
        "VCR cassette absent.  Record one once with:\\n"
        "  BRISTOL_ML_LLM_API_KEY=sk-... uv run pytest \\\\\\n"
        "    tests/integration/llm/test_llm_extractor_cassette.py \\\\\\n"
        "    --record-mode=once\\n"
        "After recording, CI replays the cassette with --record-mode=none."
    )

if stub_forced:
    print(
        "BRISTOL_ML_LLM_STUB=1 is set in this notebook (Cell 3) — the live "
        "path will not be exercised here.  Unset the env var and provide an "
        "API key to switch to the live path."
    )
elif not api_key_present:
    print(
        f"Live path requires {cfg.llm.api_key_env_var!r} populated; not set in "
        "this kernel.  See README.md (LLM section) for setup."
    )
else:
    print(
        f"Live path ready: {cfg.llm.api_key_env_var!r} populated and stub override "
        "not set.  An LlmExtractor would call OpenAI on each extract()."
    )
"""
)


# ---------------------------------------------------------------------------
# Cell 5 — side-by-side comparison via the harness (T5 Cell 5)
# ---------------------------------------------------------------------------

cell_5 = code(
    """# T5 Cell 5 — Run the evaluation harness; print the per-field table inline.
report = evaluate(cfg.llm)
print(format_report(report))
print()

# A pandas frame of disagreements for easy filtering / sorting in the demo.
if report.disagreements:
    disagreements_df = pd.DataFrame(
        [
            {"mrid": d.mrid, "field": d.field, "expected": d.expected, "actual": d.actual}
            for d in report.disagreements
        ]
    )
    print("Disagreements (DataFrame view):")
    # ``.head(len(disagreements_df))`` keeps the standard Jupyter
    # last-expression rendering of the frame while sidestepping
    # ruff B018 ("bare name; useless expression") on the otherwise
    # idiomatic notebook-trailing ``disagreements_df``.
    disagreements_df.head(len(disagreements_df))
else:
    print("No disagreements — stub on its own gold set always agrees.")
"""
)


# ---------------------------------------------------------------------------
# Cell 6 — markdown discussion of metric choice (T5 Cell 6)
# ---------------------------------------------------------------------------

cell_6 = md(
    """## Why two columns?

Intent line 41: *"different choices produce different numbers, which
makes the metric choice a lesson itself."*

The harness above prints two columns side-by-side because the same
extraction can be **right** or **wrong** depending on what we mean
by "right":

- **Exact match** is unforgiving. ``affected_capacity_mw=600.0`` vs
  ``601.5`` is a miss. For a model evaluation report this is what
  you want: it pins down hallucination to the megawatt.
- **Tolerance match** says "close enough is good enough" — ±5 MW for
  capacity, ±1 hour for timestamps. A model that reads the right
  start-of-month from a verbose REMIT message but writes
  ``2024-01-01T00:30:00Z`` because it copied the publish time is
  practically right for downstream forecasting.

Stage 16 will join extracted features into the modelling feature
table; the choice of metric *here* is the choice of "what counts as
a usable feature" *there*. The disagreement listing is the demo
punch line: it shows where the LLM gets it wrong, which is the
conversation worth having about *what* it's getting wrong.

## What's next

Stage 15 builds an embedding index over the same REMIT corpus —
parallel thread on the same data, no dependency on the extractor
output. Stage 16 joins the bi-temporal frame *and* the extracted
features into the modelling feature table. The ``as_of`` mechanic
from Stage 13 + the typed ``ExtractionResult`` boundary from
Stage 14 are what guarantee the join uses only information available
at training time, no leakage.
"""
)


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------

notebook = {
    "cells": [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {OUT} ({len(notebook['cells'])} cells)")
