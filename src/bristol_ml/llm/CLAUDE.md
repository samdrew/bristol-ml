# `bristol_ml.llm` — module guide

This module is the **LLM feature-extractor layer**: a typed boundary
(`RemitEvent` → `ExtractionResult`) plus two interchangeable backends
selected by config + env var. Stage 14 introduced the layer; Stages 15
(embedding index) and 16 (feature-table join) consume it.

The architectural narrative — why the layer exists, the stub-first
discipline, the Protocol contract, the demo-moment harness — lives in
[`docs/architecture/layers/llm.md`](../../../docs/architecture/layers/llm.md).
The file you are reading is the **module guide**: schema reference,
the cassette-refresh ritual, gold-set curation guidance, and the
downstream-consumer warnings that don't belong in the layer doc.

## Public surface

```python
from bristol_ml.llm import RemitEvent, ExtractionResult, Extractor
from bristol_ml.llm.extractor import (
    StubExtractor,
    LlmExtractor,
    build_extractor,
    DEFAULT_GOLD_SET_PATH,
    STUB_ENV_VAR,        # "BRISTOL_ML_LLM_STUB"
)
from bristol_ml.llm.evaluate import (
    evaluate,            # LlmExtractorConfig | None -> EvaluationReport
    format_report,       # EvaluationReport -> str (deterministic stdout block)
    EvaluationReport,
    FieldAgreement,
    TOLERANCE_CAPACITY_MW,   # 5.0
    TOLERANCE_TIME,          # timedelta(hours=1)
)
```

The `__init__.py` exposes only the boundary types — `Extractor` is the
Protocol, `RemitEvent` and `ExtractionResult` are the Pydantic models
that cross every call. Concrete implementations live in `extractor.py`
so callers (Stage 15, Stage 16) can import the schema without dragging
the OpenAI SDK into their import graph.

## `RemitEvent` — input schema

`RemitEvent` is the typed mirror of the extraction-relevant subset of
Stage 13's `OUTPUT_SCHEMA` row.

| Field | Type | Notes |
|-------|------|-------|
| `mrid` | `str` | The Elexon message id; primary key half. |
| `revision_number` | `int >= 0` | The revision; primary key half. |
| `message_status` | `str` | `"Active"`, `"Withdrawn"`, etc. |
| `published_at` | `datetime` (UTC-aware) | Transaction-time. Naive datetimes raise `ValidationError`. |
| `effective_from` | `datetime` (UTC-aware) | Valid-time start. |
| `effective_to` | `datetime \| None` | Valid-time end; `None` for open-ended events. |
| `fuel_type` | `str \| None` | Elexon vocabulary (`"Coal"`, `"Gas"`, …); `None` allowed. |
| `affected_mw` | `float \| None` | Unavailable capacity in MW; `None` allowed. |
| `event_type` | `str \| None` | `"Outage"`, `"Restriction"`, …; `None` allowed. |
| `cause` | `str \| None` | `"Planned"` / `"Unplanned"`; `None` allowed. |
| `message_description` | `str \| None` | **Frequently NULL** on the live API; the extractor synthesises a prompt input from the structured fields when NULL (plan §1 D6 / OQ-B). |

`extra="forbid"`, `frozen=True` — typos in caller dicts surface as
`ValidationError` at construction rather than silently dropping fields.

## `ExtractionResult` — output schema

| Field | Type | Notes |
|-------|------|-------|
| `event_type` | `str` | LLM-canonicalised; one of `"Outage"`, `"Restriction"`, `"Withdrawn"`, `"Other"`. |
| `fuel_type` | `str` | LLM-canonicalised; Elexon vocabulary (see prompt for the full list). |
| `affected_capacity_mw` | `float \| None` | Capacity in MW or `None` if unrecoverable. |
| `effective_from` | `datetime` (UTC-aware) | Mirrored from input or LLM-extracted; UTC-aware always. |
| `effective_to` | `datetime \| None` | `None` for open-ended events. |
| `confidence` | `float ∈ [0.0, 1.0]` | **Documented sentinel, not calibrated probability.** See below. |
| `prompt_hash` | `str \| None` | First 12 hex chars of SHA-256(prompt-bytes); `None` for stub. |
| `model_id` | `str \| None` | E.g. `"gpt-4o-mini"`; `None` for stub. |

**Downstream-consumer warning (Stage 16 join planner).** `confidence`
is a sentinel:

| Path | `confidence` | Meaning |
|------|------|---------|
| Stub hand-labelled hit | `1.0` | `(mrid, revision_number)` matches the gold set; expected value lifted from the fixture. |
| Stub default fallback | `0.0` | Unknown event; structural fields synthesised from input where non-NULL. |
| Live OpenAI hit | LLM-emitted | Strict-mode response; typically `1.0` when fully grounded, lower when the LLM hedges. |
| Live fallback | `0.0` | Network / parse / validation failure; documented default returned (NFR-6). |

OpenAI strict mode does not return per-token logprobs by default and
the schema does not require calibrated probabilities (plan §1 D14).
**Treat `confidence` as a sentinel for "did anything go wrong"** (a
`0.0` is a fallback marker), not a calibrated reliability signal.

The provenance fields (`prompt_hash`, `model_id`) are stamped on
**fallback rows too** — a downstream consumer can still see "this row
came from prompt X on model Y" even when the LLM call failed.

## `Extractor` Protocol

`runtime_checkable`, two methods, no other public attributes:

```python
@runtime_checkable
class Extractor(Protocol):
    def extract(self, event: RemitEvent) -> ExtractionResult: ...
    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]: ...
```

Adding a method is a Stage 15/16 contract change — discuss before
doing it. The two-method shape is the AC-1 cap (intent line 32).

## Persistence (Stage 16)

New module `bristol_ml.llm.persistence`. Adds the on-disk persistence step that
Stage 14's extractor was missing (codebase hazard H1 — `Extractor.extract_batch`
returns `list[ExtractionResult]` in memory with no on-disk persistence layer).
Stage 16 needs the extractor output as a stable artefact that the feature
assembler reads cheaply on every retraining run.

### `EXTRACTED_OUTPUT_SCHEMA` (11 columns)

`pyarrow.Schema` constant. Column order is contractual; the assembler joins on
`(mrid, revision_number)` and reads `affected_capacity_mw`, `event_type`,
`fuel_type`, and `confidence` by name.

| Column | Type | Notes |
|--------|------|-------|
| `mrid` | `string` | Primary key half (mirrors Stage 13 REMIT log). |
| `revision_number` | `int32` | Primary key half. |
| `event_type` | `string` | LLM-canonicalised; non-nullable. |
| `fuel_type` | `string` | LLM-canonicalised; non-nullable. |
| `affected_capacity_mw` | `float64` | Nullable — `None` when unrecoverable. |
| `effective_from` | `timestamp[us, tz=UTC]` | Mirrored from input or LLM-extracted. |
| `effective_to` | `timestamp[us, tz=UTC]` | Nullable for open-ended events. |
| `confidence` | `float32` | Documented sentinel (see `ExtractionResult.confidence`). |
| `prompt_hash` | `string` | Nullable — 12-char SHA-256 prefix; `None` for stub. |
| `model_id` | `string` | Nullable — `None` for stub. |
| `extracted_at_utc` | `timestamp[us, tz=UTC]` | Provenance scalar (constant per run). |

### `extract_and_persist(extractor, remit_df, *, output_path) -> Path`

Runs `extractor.extract_batch(events)` over the entire `remit_df`, joins the
results back onto `(mrid, revision_number)`, stamps `extracted_at_utc`, casts
to `EXTRACTED_OUTPUT_SCHEMA`, and writes via
`ingestion._common._atomic_write` (idempotent — partial writes leave the
previous file intact, NFR-3). Returns the absolute output path. Raises
`ValueError` if `remit_df` is missing required columns; raises `RuntimeError` if
`extract_batch` returns a different number of results than inputs (Protocol
contract violation).

### `load_extracted(path) -> pd.DataFrame`

Schema-validated read for `EXTRACTED_OUTPUT_SCHEMA`. Rejects both missing and
extra columns (exact schema contract). The function's discipline mirrors
`features.assembler.load`.

### CLI: `python -m bristol_ml.llm.persistence`

Ties `remit.fetch + remit.load -> build_extractor(cfg.llm) ->
extract_and_persist` and prints the output path.

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cache {auto,refresh,offline}` | `offline` | Cache policy for the REMIT ingester. |
| `--limit N` | `0` (no cap) | Cap the number of REMIT rows fed to the extractor. Use a small value to dry-run the live path. |
| `--output PATH` | `data/processed/remit_extracted.parquet` | Override the output path. |

Hydra overrides follow as trailing positional arguments (e.g.
`llm.type=openai llm.model_name=gpt-4o-mini`). The group `+llm=extractor` is
composed automatically by the CLI so callers do not need to add it manually.

**Stub-default.** `BRISTOL_ML_LLM_STUB=1` (the CI default) routes all
extraction through `StubExtractor`; the output is a valid parquet with
stub-quality values (gold-set hits at `confidence=1.0`; misses at
`confidence=0.0`). This keeps CI green without incurring OpenAI costs.

### Architectural reason (plan A5)

Stage 14's extractor returns in-memory results only, which was intentional for
the Stage 14 scope. Stage 16 required a stable on-disk artefact that the feature
assembler reads without re-running extraction on every training call. Rather than
adding persistence directly to `extractor.py`, the separate module follows the
project's ingestion-style pattern (codebase §1: fetch-then-load; `_atomic_write`
for idempotency) and keeps the features layer free of LLM-layer imports beyond
`load_extracted` (plan OQ-5).

The assembler's auto-run fallback (when the parquet is absent) calls
`extract_and_persist` inline under stub mode to keep CI green; the human running
the real-extractor path executes `python -m bristol_ml.llm.persistence`
beforehand, then lets the assembler read the warm cache.

## Quick recipes

### Run the stub against a single event (offline)

```python
from datetime import UTC, datetime
from bristol_ml.llm import RemitEvent
from bristol_ml.llm.extractor import StubExtractor

stub = StubExtractor()
print(f"loaded {stub.gold_set_size} hand-labelled records")
event = RemitEvent(
    mrid="M-A",
    revision_number=0,
    message_status="Active",
    published_at=datetime(2024, 1, 1, 9, tzinfo=UTC),
    effective_from=datetime(2024, 1, 15, tzinfo=UTC),
    effective_to=datetime(2024, 1, 20, tzinfo=UTC),
    fuel_type="Nuclear",
    affected_mw=600.0,
    event_type="Outage",
    cause="Planned",
    message_description="Stub: planned nuclear outage for refuelling.",
)
print(stub.extract(event))
```

### Run the live extractor (live OpenAI; spends tokens)

```bash
export BRISTOL_ML_LLM_API_KEY=sk-...
uv run python -m bristol_ml.llm.extractor llm.type=openai
```

The CLI prints a one-extraction summary against M-A's structured fields
so you can see the live model's output without writing a script. Add
`llm.model_name=gpt-4o` etc. to override the model.

### Run the evaluation harness

```bash
uv run python -m bristol_ml.llm.evaluate                       # stub (default)
uv run python -m bristol_ml.llm.evaluate llm.type=openai       # live harness
uv run python -m bristol_ml.llm.evaluate --max-disagreements 5
```

The harness prints the deterministic stdout block (header → per-field
table → disagreement listing) and emits a structured `loguru` INFO
record with the same summary so a CLI run leaves a captured trace.

## Cassette-refresh ritual

Replay-only by default. The integration test at
`tests/integration/llm/test_llm_extractor_cassette.py` runs under
`--record-mode=none` and **skips when the cassette is absent** — same
pattern as the REMIT cassette test.

When does it need a refresh?

- The active prompt at `conf/llm/prompts/extract_v1.txt` changed (the
  request body is part of the cassette match keys; a prompt edit
  produces an unrecorded request → "no match found" on replay).
- The configured `llm.model_name` changed (model name appears in the
  request URL path / body for OpenAI Chat Completions).
- The `_sample_events()` fixture in the integration test changed.

### Recording

```bash
export BRISTOL_ML_LLM_API_KEY=sk-...
uv run pytest \
    tests/integration/llm/test_llm_extractor_cassette.py \
    --record-mode=once
```

`pytest-recording` writes the cassette to
`tests/fixtures/llm/cassettes/test_llm_extractor_against_cassette.yaml`.

The cassette covers ~5 strata-spanning gold-set events (M-A Nuclear,
M-B Gas, M-C Coal, M-H synthesise-on-NULL probe, M-I Solar). Five is
deliberately small — a larger cassette balloons the diff and adds
maintenance pressure with diminishing return.

### Verifying no key leaked

VCR is configured to filter `authorization`, `cookie`, `set-cookie`,
`x-api-key` headers (see `vcr_config` fixture in the test). Verify
manually after recording:

```bash
grep -inE 'sk-|Bearer ' tests/fixtures/llm/cassettes/*.yaml || echo "no key tokens found"
```

A non-empty match is a critical issue — **do not commit**, delete the
cassette, and report. The known-good shape after filtering shows
`Authorization: DUMMY` (or absent) in every recorded request.

The request body still contains the prompt + event JSON, which is
fine because the prompt is open-source in this repo. The event JSON
is synthetic (gold-set fixture) and contains no secrets.

### Replay (CI default)

```bash
uv run pytest tests/integration/llm/test_llm_extractor_cassette.py -v
```

The test honours `--record-mode=none` (the project default in
`pyproject.toml`) and never touches the network. When recording is
needed, run with an explicit `--record-mode=once`.

## Gold-set curation

The hand-labelled fixture at `tests/fixtures/llm/hand_labelled.json`
is the project's eval ground truth. Stage 14 shipped 76 records;
intent line 40 names "dozens to low hundreds" as the working range
(plan §1 D8 / Ctrl+G OQ-C → 80 was the target).

### File shape

```json
{
  "schema_version": 1,
  "records": [
    {
      "event": { /* RemitEvent fields */ },
      "expected": { /* ExtractionResult fields minus prompt_hash/model_id */ }
    }
  ]
}
```

`schema_version: 1` is asserted by `_load_gold_set` — bumping the
version is a code change (`bristol_ml.llm.extractor`) plus a fixture
edit, not a fixture-only edit, so a structural drift cannot ship
silently.

### Adding a record

1. Pick a real REMIT message (or synthesise a plausible one, marking
   it with a `Stub:` prefix in `message_description` like the
   existing M-A..M-K block).
2. Hand-label the expected extraction. The `expected` payload omits
   `prompt_hash` and `model_id` (the stub fills them with `None`).
3. Choose `(mrid, revision_number)` so it does not collide with an
   existing key. The convention is alphabetic prefixes: M-A..M-K
   (early stage records), M-L..M-Y (Gas batch), M-AA..M-AI (Nuclear
   batch), M-BA..M-BJ (Wind batch), M-CA..M-CD (Coal), M-DA..M-DF
   (Solar), M-EA..M-ED (Hydro), M-FA..M-FD (Pumped Storage),
   M-GA..M-GC (Biomass), M-HA..M-HB (Oil), M-IA..M-IB
   (Interconnector), M-JA..M-JB (Battery), M-NA..M-ND
   (synthesise-on-NULL probes), M-NM (missing-capacity probe).
4. `expected.confidence` should be `1.0` for a hand-labelled record
   (the stub returns this verbatim). The default-fallback `0.0`
   sentinel is reserved for the miss path; never put it in the gold
   set.
5. Run the harness: `uv run python -m bristol_ml.llm.evaluate`
   — it should report the new record's row in the per-field
   summary count.

### When to bump the size

The gold set is calibration ground truth, not training data. Add
records when:

- A new fuel-type or event-type appears in the live REMIT stream and
  is not represented (the harness will report the disagreement at
  Stage 16 join time, but the gold-set fix is upstream).
- A new failure mode is observed during a live cassette run
  — encode it as a probe (M-NA..M-NM convention).
- The cassette refresh surfaces a regression that the existing gold
  set did not catch.

Stage 14 ships at 76 records — within the intent's range. Pushing
past ~150 starts costing curation time; the per-record marginal value
falls off as the categorical surface saturates.

## Notebook

`notebooks/14_llm_extractor.ipynb` is the demo surface — six cells
(plus title markdown):

1. **Bootstrap** — repo-root walk + `+llm=extractor` Hydra compose.
2. **Gold set** — load the JSON, print per-fuel-type breakdown.
3. **Stub run** — `os.environ["BRISTOL_ML_LLM_STUB"]="1"` then
   `build_extractor`; first 3 results inline.
4. **Live banner** — checks cassette + API key + stub-override; prints
   one of three banners explaining the live-path readiness.
5. **Side-by-side** — `evaluate(cfg.llm)` + `format_report(...)`;
   pandas-frame disagreements view.
6. **Discussion** — markdown on metric choice (intent line 41).

Regenerate via the three-step ritual:

```bash
uv run python scripts/_build_notebook_14.py
uv run jupyter nbconvert --execute --to notebook --inplace \
    notebooks/14_llm_extractor.ipynb
uv run ruff format notebooks/14_llm_extractor.ipynb
```

## Tests

Located alongside the production code:

- `tests/unit/llm/test_extractor.py` — Protocol shape, `RemitEvent` /
  `ExtractionResult` validation (UTC-aware datetimes, confidence in
  `[0, 1]`), stub gold-set hits + miss-path defaults, factory dispatch
  (env-var override, `None` config), live-path init guards.
- `tests/unit/llm/test_evaluate.py` — harness against the stub,
  byte-deterministic stdout, provenance header, tolerance counters,
  loguru summary record.
- `tests/integration/llm/test_llm_extractor_cassette.py` — live path
  against the recorded cassette (skipped without the cassette in CI).
- `tests/integration/test_notebook_14.py` — `nbconvert --execute` on
  the demo notebook under `BRISTOL_ML_LLM_STUB=1`.

Run the LLM-only suite:

```bash
uv run pytest tests/unit/llm/ tests/integration/llm/ \
              tests/integration/test_notebook_14.py -v
```

## Cross-references

- Layer doc — [`docs/architecture/layers/llm.md`](../../../docs/architecture/layers/llm.md).
- Stage 14 retro — [`docs/lld/stages/14-llm-extractor.md`](../../../docs/lld/stages/14-llm-extractor.md).
- Stage 16 retro — [`docs/lld/stages/16-model-with-remit.md`](../../../docs/lld/stages/16-model-with-remit.md).
- Intent — [`docs/intent/14-llm-extractor.md`](../../../docs/intent/14-llm-extractor.md).
- Plan — [`docs/plans/completed/14-llm-extractor.md`](../../../docs/plans/completed/14-llm-extractor.md).
- ADR-0003 — Protocol-over-ABC for swappable interfaces.
- Sibling boundary — `src/bristol_ml/ingestion/CLAUDE.md` (REMIT row;
  `OUTPUT_SCHEMA` is the upstream `RemitEvent` mirrors).
- Downstream consumer — `src/bristol_ml/features/CLAUDE.md` §"Stage 16 notes"
  (how the persistence parquet flows into the assembler).
- README §"Configuring an OpenAI API key" — operator-facing setup.
