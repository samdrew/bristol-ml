# LLM — layer architecture

- **Status:** Provisional — first realised by Stage 14 (LLM feature
  extractor: `Extractor` Protocol + `StubExtractor` + `LlmExtractor` +
  evaluation harness). Stage 15 (embedding index) and Stage 16
  (feature-table join) consume the boundary types defined here without
  importing either backend.
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities)
  (LLM paragraph); [`docs/intent/14-llm-extractor.md`](../../intent/14-llm-extractor.md)
  (Stage 14 intent — the demo moment + offline-default discipline).
- **Concrete instances:** [Stage 14 retro](../../lld/stages/14-llm-extractor.md) — gold-set composition, cassette outcome, evaluation harness output.
- **Related principles:** §2.1.1 (standalone module), §2.1.2 (typed narrow interfaces), §2.1.3 (stub-first for expensive / flaky external dependencies — the load-bearing principle for this layer), §2.1.4 (config outside code), §2.1.6 (provenance — every result carries `prompt_hash` + `model_id`), §2.1.7 (tests at boundaries).
- **Upstream layer:** [Ingestion](ingestion.md) — Stage 13's REMIT bi-temporal store. The extractor consumes a single row of `OUTPUT_SCHEMA` lifted into a typed `RemitEvent`; it does not write back to the parquet.
- **Downstream consumers:** Stage 15 (embedding index — parallel thread on the same data), Stage 16 (feature-table join — extracted features become model inputs).

---

## Why this layer exists

The LLM layer is the **expensive-and-flaky-dependency boundary** for the project. REMIT messages carry free-text fields whose structured-feature equivalents (`event_type`, `fuel_type`, `affected_capacity_mw`, two timestamps) are what downstream stages actually want. An LLM is the cheapest mechanism that produces those features at archive scale — but every property of an LLM is the opposite of what an ML pipeline normally accepts: the call costs money, the network can fail, the response can drift between model versions, and a regression is silent unless the project tests against a fixed gold set.

Stage 14 draws a typed boundary that hides every one of those properties from downstream callers. The two-method `Extractor` Protocol takes a `RemitEvent` and returns an `ExtractionResult`; whether that result came from a hand-labelled stub, a live OpenAI call, or a future locally-served model is invisible to the caller. The pedagogical payoff is the **stub-first discipline**: every CI run, every notebook bootstrap, and every unit test runs the stub path with no API key and no network call (intent AC-2). The architectural payoff is the **evaluation harness**: a side-by-side accuracy report against the gold set turns "the LLM is right" into a reproducible measurement (intent AC-4, plan §1 D11).

The load-bearing design constraint is intent line 9: *"this is as much architectural as analytical — any expensive or flaky external dependency in an ML system benefits from a stub-first design with an evaluation harness that holds the real implementation to account."* Every decision flows from that.

---

## Stub-first discipline (offline-by-default — AC-2)

The Stage 14 extractor is **triple-gated** so that no path through the project can fire a live OpenAI call by accident:

1. **Config discriminator (plan §1 D3 / D4).** `LlmExtractorConfig.type: Literal["stub", "openai"]` defaults to `stub` in `conf/llm/extractor.yaml`. A YAML override (`llm.type=openai`) is required to opt in.
2. **Env-var override (plan §1 D4).** `BRISTOL_ML_LLM_STUB=1` forces `StubExtractor` regardless of `config.type`. The notebook sets it explicitly in Cell 3; CI sets it in `tests/integration/test_notebook_14.py` and `pyproject.toml`. The override is the load-bearing fallback when the YAML is misconfigured — a misconfigured live `type` plus a missing API key would otherwise raise at init; the env-var short-circuits before that.
3. **API-key gate (plan §1 D5).** `LlmExtractor.__init__` reads `BRISTOL_ML_LLM_API_KEY` (or whatever `config.api_key_env_var` names) and raises `RuntimeError` at init time if it is empty. The error message names *both* env-vars — the missing key *and* the `BRISTOL_ML_LLM_STUB=1` escape hatch — so an operator hitting the error sees the offline path immediately.

The dispatch happens in a single function (`bristol_ml.llm.extractor.build_extractor`); calling it with `config=None` is also supported and returns a `StubExtractor`. Stage 15 / Stage 16 unit tests rely on this `None`-tolerance.

This is the env-var pattern the next authenticated-API stage (when there is one) will copy. The `BRISTOL_ML_REMIT_STUB` precedent at the ingestion layer set the shape; Stage 14 confirms it as the project convention.

---

## Public surface

```python
# src/bristol_ml/llm/__init__.py
from bristol_ml.llm import RemitEvent, ExtractionResult, Extractor

# src/bristol_ml/llm/extractor.py
from bristol_ml.llm.extractor import (
    StubExtractor,
    LlmExtractor,
    build_extractor,        # factory: LlmExtractorConfig | None -> Extractor
    DEFAULT_GOLD_SET_PATH,  # tests/fixtures/llm/hand_labelled.json
    STUB_ENV_VAR,           # "BRISTOL_ML_LLM_STUB"
)

# src/bristol_ml/llm/evaluate.py
from bristol_ml.llm.evaluate import evaluate, format_report, EvaluationReport

# src/bristol_ml/llm/__main__-style entry points (DESIGN §2.1.1)
# python -m bristol_ml.llm.extractor [overrides...]
# python -m bristol_ml.llm.evaluate  [overrides...] [--redact-timestamps] [--max-disagreements N]
```

### `Extractor` Protocol (in `bristol_ml/llm/__init__.py`)

```python
@runtime_checkable
class Extractor(Protocol):
    def extract(self, event: RemitEvent) -> ExtractionResult: ...
    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]: ...
```

Two methods is the AC-1 cap. Adding a third method to the Protocol is the project's signal that the layer is drifting — intent line 32: *"the interface is small enough that writing a third implementation in the future is plausible."* A future open-weights / locally-served extractor slots in by adding a literal value to `LlmExtractorConfig.type` and a dispatch branch in `build_extractor`; the Protocol is unchanged.

`runtime_checkable` lets unit tests assert structural conformance with `isinstance(_, Extractor)` (ADR-0003 precedent for Protocol-over-ABC for swappable interfaces). The Protocol does not import `StubExtractor` or `LlmExtractor`; downstream callers depend on the Protocol alone.

### `RemitEvent` (in `bristol_ml/llm/__init__.py`)

```python
class RemitEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mrid: str
    revision_number: int = Field(ge=0)
    message_status: str
    published_at: datetime          # UTC-aware
    effective_from: datetime        # UTC-aware
    effective_to: datetime | None
    fuel_type: str | None
    affected_mw: float | None
    event_type: str | None
    cause: str | None
    message_description: str | None # frequently NULL on live API
```

The typed mirror of the extraction-relevant subset of Stage 13's `OUTPUT_SCHEMA` row. UTC-aware datetime validators reject naive timestamps at construction — passing a naive `datetime` through the Stage 14 boundary would silently break the Stage 13 timezone contract; the validator catches it before it propagates.

### `ExtractionResult` (in `bristol_ml/llm/__init__.py`)

```python
class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str
    fuel_type: str
    affected_capacity_mw: float | None
    effective_from: datetime              # UTC-aware
    effective_to: datetime | None
    confidence: float                     # in [0.0, 1.0]
    prompt_hash: str | None               # 12-char SHA-256 prefix; None for stub
    model_id: str | None                  # e.g. "gpt-4o-mini"; None for stub
```

Every value crossing the extractor boundary is an `ExtractionResult` instance, never a raw dict (intent AC-5). The provenance pair (`prompt_hash`, `model_id`) is `None` on the stub path and populated on the live path — **stamped on fallback rows too** so a downstream consumer can still tell "this came from prompt X on model Y" even when the LLM call failed and the documented default was returned.

`confidence` is a documented sentinel, not a calibrated probability:

| Path | `confidence` | When |
|------|------|------|
| Stub hand-labelled hit | `1.0` | `(mrid, revision_number)` matches the gold set. |
| Stub default fallback | `0.0` | Unknown event; structural fields synthesised from input where non-NULL (plan §1 D16). |
| Live OpenAI hit | LLM-emitted (typically `1.0` when fully grounded) | Strict-mode response passed Pydantic validation. |
| Live fallback | `0.0` | Network / parse / validation failure; documented default returned (NFR-6). |

Stage 16 callers must treat `confidence` as a sentinel, not a probability — see [`src/bristol_ml/llm/CLAUDE.md`](../../../src/bristol_ml/llm/CLAUDE.md) for the downstream-consumer warning.

---

## Provenance via prompt hash (NFR-5)

The active prompt is a plain-text file under `conf/llm/prompts/extract_v1.txt` (no Jinja). The first 12 hex chars of the SHA-256 digest of the file's bytes is recorded in every `ExtractionResult.prompt_hash` so a swap of the prompt produces a different hash. *"We swapped the prompt and everything changed"* is then diagnosable from the output.

Why a hash, not a filename or an embedded version field?

- The filename is human-readable but mutable; `extract_v1.txt` can silently drift if edited in place.
- An embedded version field would solve that, but adds a parsing contract on top of "the file is plain text".
- A bytes hash is the cheapest content-derived identity that survives edits *and* renames; collisions over the project lifetime are vanishing (12 hex chars = 48 bits = 1 in 281 trillion).

The truncation is a readability choice — the hash appears inline in stdout output and parquet preview frames. The full digest is available via `bristol_ml.llm._prompts.prompt_sha256_full` if a future stage needs to join onto an external prompt registry.

The hash is computed once at `LlmExtractor.__init__` so a deploy-time stale prompt fails loudly at construction (`FileNotFoundError`) rather than at the first call, and every call shares the same prompt-bytes-derived identity.

---

## Live path: OpenAI Chat Completions strict mode (D6)

`LlmExtractor` calls `openai.chat.completions.create` with a strict
`response_format`:

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "extraction_result",
        "strict": True,
        "schema": _OPENAI_RESPONSE_SCHEMA,
    },
}
```

OpenAI strict mode (GA since August 2024) constrains the model at decode time via CFG token masking — the model cannot emit JSON that would fail the schema. This is the most field-tested constrained-generation route across providers (researcher R1.2).

**Schema-shape gotcha.** OpenAI strict mode requires `additionalProperties: false` on every object *and* every property in `required` (no implicit-optional via `default`). Pydantic's `ExtractionResult.model_json_schema()` emits `additionalProperties: false` for `extra="forbid"` models but it also emits `default: null` on `Optional[...]` fields and omits them from `required` — both would be rejected by the OpenAI API at first call. The extractor module therefore hand-authors `_OPENAI_RESPONSE_SCHEMA` for the LLM-populated subset (`event_type`, `fuel_type`, `affected_capacity_mw`, two timestamps, `confidence`). The provenance fields are stamped by the extractor *after* the LLM returns and so are not part of what the model populates.

**`message_description` strategy (D6 / Ctrl+G OQ-B).** The Stage 13 stream endpoint frequently returns NULL for `message_description` on live rows (`remit.py:431`). Stage 14 accepts NULL and synthesises the prompt input from the structured fields (`event_type`, `cause`, `fuel_type`, `affected_mw`, `effective_from`, `effective_to`). No follow-up `GET /remit/{messageId}` call. The structured fields are themselves the extraction signal in production REMIT data — the LLM is being asked to *interpret + standardise* them. Hydration (~45,000 extra calls per archive run) is deferred to Stage 16 if its feature-join needs richer text.

**Graceful degradation (NFR-6 / D16).** On any network / parse / Pydantic-validation failure, `extract` logs WARNING with the event id and the failure reason, then returns an `ExtractionResult` populated with the documented default (`event_type` / `fuel_type` from the structured fields when non-NULL, capacity `None`, times mirrored from input, `confidence=0.0`, `prompt_hash` and `model_id` recorded). **`extract` never raises an unhandled exception.** Constrained decoding makes parse failures rare but not impossible (network truncation, Pydantic stricter than the LLM contract on UTC-aware datetimes).

---

## Evaluation harness — the demo moment (AC-4)

`bristol_ml.llm.evaluate.evaluate(config)` runs the active extractor over the gold set and returns an `EvaluationReport`; `format_report(report)` renders it as the deterministic stdout table:

```
=== Stage 14 LLM extractor evaluation ===
implementation: stub
prompt_hash:    none
model_id:       none
gold_set:       tests/fixtures/llm/hand_labelled.json (sha256: a1b2c3d4e5f6, n=76)
generated_at:   2026-04-27T14:23:11Z

per-field agreement:
  field                   exact   tolerance
  event_type              76/76    76/76
  fuel_type               76/76    76/76
  affected_capacity_mw    76/76    76/76    (tolerance: ±5 MW)
  effective_from          76/76    76/76    (tolerance: ±1 h)
  effective_to            76/76    76/76    (tolerance: ±1 h)

disagreements (first 10):
  (none — stub on its own gold set always agrees)

=== end ===
```

**Two-metric design (D11).** Exact-match-per-field *and* tolerance-match on the same fields, side-by-side. The same data produces different numbers under different metric choices, which is the lesson — intent line 41: *"different choices produce different numbers, which makes the metric choice a lesson itself."* Categorical fields (`event_type`, `fuel_type`) have exact == tolerance (no notion of "close enough" for strings); the tolerance column is kept for layout uniformity.

**Specific tolerances (D12).** ±5 MW for `affected_capacity_mw`; ±1 h for the two timestamps. Documented choices, not magic numbers — see the `evaluate.py` module docstring for the rationale and override path.

**Reproducibility (NFR-3).** Two runs with the same inputs produce byte-identical stdout *modulo* the single `generated_at` header line. `--redact-timestamps` replaces that line with a fixed sentinel; `tests/unit/llm/test_evaluate.py::test_evaluate_harness_output_is_deterministic_modulo_timestamp` is the byte-equality assertion.

**Provenance (AC-4).** The header carries every dimension the operator might swap that would change the result: prompt hash, model id, gold-set sha256, gold-set size, implementation name. `tests/unit/llm/test_evaluate.py::test_evaluate_harness_records_provenance` pins the header shape.

**Captured trace (D15 / OQ-D).** The same content is also emitted as a single structured `loguru` INFO record (`logger.bind(summary={...}).info("Stage 14 evaluation: implementation=... gold_set_size=... disagreements=...")`) so a non-notebook CLI run leaves a captured trace in the standard log stream rather than disappearing into stdout.

---

## Internals

### Gold-set fixture (`tests/fixtures/llm/hand_labelled.json`)

JSON file at `tests/fixtures/llm/hand_labelled.json`; top-level `schema_version: 1` plus a list of `{event: ..., expected: ...}` records. Indexed by `(mrid, revision_number)` — the same primary key Stage 13's REMIT parquet uses, so every distinct revision can carry its own labelled extraction. Gold-set composition shipped at Stage 14: **76 records** stratified across event type (Outage 51, Restriction 25) and the project's 11 fuel-type vocabulary (Gas 18, Nuclear 13, Wind 13, Solar 8, Coal 6, Hydro 5, Pumped Storage 4, Biomass 3, Oil 2, Interconnector 2, Battery 2). Just below the researcher-R3 reference of 100 records (±10 % margin at 95 % CI / 80 % accuracy) — adequate for the demo, light enough to curate (Ctrl+G OQ-C).

A few records (M-NA..M-ND) carry `message_description: None` to exercise the synthesise-on-NULL path (D6 / OQ-B); M-NM carries `affected_mw: None` to exercise the missing-capacity path. These probes are the load-bearing test of the live extractor's NULL-tolerance.

The fixture is loaded once at `StubExtractor.__init__` and indexed; per-call `extract` is an O(1) dict lookup. Construction is cheap (one JSON read + Pydantic validation per record) so re-instantiating per call site is fine. The stub is thread-safe after construction — the lookup map is built once and read-only thereafter.

### Cassette refresh ritual

The cassette at `tests/fixtures/llm/cassettes/test_llm_extractor_against_cassette.yaml` is replayed by pytest-recording under `--record-mode=none` (the CI default). The integration test at `tests/integration/llm/test_llm_extractor_cassette.py` *skips* when the cassette is absent, mirroring the REMIT cassette pattern.

Recording requires a real OpenAI API key and is performed once locally:

```bash
BRISTOL_ML_LLM_API_KEY=sk-... uv run pytest \
    tests/integration/llm/test_llm_extractor_cassette.py \
    --record-mode=once
```

VCR is configured to filter `authorization`, `cookie`, `set-cookie`, `x-api-key` headers (the project's standard fixture, mirrored from `tests/integration/ingestion/test_remit_cassettes.py`). The request body still contains the prompt + event JSON, which is fine because the prompt is open-source in this repo. After recording, CI replays under `--record-mode=none` and never touches the network.

The full cassette-refresh ritual — including how to verify no key leaked into the recorded YAML — is documented in [`src/bristol_ml/llm/CLAUDE.md`](../../../src/bristol_ml/llm/CLAUDE.md).

### `build_extractor` factory

```python
def build_extractor(
    config: LlmExtractorConfig | None,
    *,
    gold_set_path: Path | None = None,
) -> Extractor:
```

Single dispatch point. Returns `StubExtractor` when `config is None`, when `config.type == "stub"`, or when `BRISTOL_ML_LLM_STUB=1` is set in the environment. Returns `LlmExtractor(config)` when `config.type == "openai"` and the env-var is not set. A future `Literal["stub", "openai", "<future>"]` extension that forgets to update this branch raises a defensive `ValueError` — silent fall-through to the stub would mask the misconfiguration.

When the env-var override fires against an `openai` config, a loguru INFO record names the override so the operator sees what happened.

### Module structure

```
src/bristol_ml/llm/
├── __init__.py      # boundary types: Extractor Protocol, ExtractionResult, RemitEvent
├── extractor.py     # StubExtractor + LlmExtractor + build_extractor + standalone CLI
├── evaluate.py      # EvaluationReport + evaluate() + format_report() + standalone CLI
├── _prompts.py      # SHA-256 hashing + load_prompt(path) -> (text, prompt_hash)
└── CLAUDE.md        # module guide; cassette-refresh ritual; gold-set curation
```

The split between `__init__.py` (boundary types) and `extractor.py` (implementations + factory) is the load-bearing mechanism for AC-5: *"the schema is importable from `bristol_ml.llm` without importing either concrete implementation."* Stage 15 / Stage 16 unit tests rely on this — they need `ExtractionResult` shape without dragging the OpenAI SDK into their import graph.

---

## Standalone CLI

```bash
uv run python -m bristol_ml.llm.extractor --help
uv run python -m bristol_ml.llm.extractor                       # stub by default
uv run python -m bristol_ml.llm.extractor llm.type=openai       # live path

uv run python -m bristol_ml.llm.evaluate --help
uv run python -m bristol_ml.llm.evaluate                        # stub harness run
uv run python -m bristol_ml.llm.evaluate llm.type=openai        # live harness run
uv run python -m bristol_ml.llm.evaluate --redact-timestamps    # for byte-equality tests
```

Both CLIs compose `+llm=extractor` automatically (the `llm` group is *not* in `conf/config.yaml`'s defaults — parallel to `serving`); explicit Hydra overrides flow through. NFR-8 (DESIGN §2.1.1) is enforced by `tests/unit/llm/test_extractor.py::test_extractor_module_runs_standalone` and `tests/unit/llm/test_evaluate.py::test_evaluate_module_runs_standalone_help`.

---

## Config

```python
# conf/_schemas.py
class LlmExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["stub", "openai"]            # discriminator (D3)
    model_name: str | None = None              # e.g. "gpt-4o-mini"
    api_key_env_var: str = "BRISTOL_ML_LLM_API_KEY"
    prompt_file: Path | None = None            # e.g. conf/llm/prompts/extract_v1.txt
    request_timeout_seconds: float = 30.0

# AppConfig
llm: LlmExtractorConfig | None = None
```

```yaml
# conf/llm/extractor.yaml
# @package llm
type: stub
model_name: gpt-4o-mini
api_key_env_var: BRISTOL_ML_LLM_API_KEY
prompt_file: conf/llm/prompts/extract_v1.txt
request_timeout_seconds: 30.0
```

The `llm` group is *not* listed in `conf/config.yaml`'s defaults — entry points (the extractor CLI, the evaluation harness, the notebook) compose it explicitly via `+llm=extractor`. The `None` default on `AppConfig.llm` keeps every prior stage's CLI / config-smoke test unaffected.

---

## Cross-references

- `src/bristol_ml/llm/CLAUDE.md` — concrete module guide; gold-set curation; cassette-refresh ritual + leak-check.
- [Stage 14 retro](../../lld/stages/14-llm-extractor.md) — observed cassette outcome, gold-set distribution, deviations.
- [`docs/intent/14-llm-extractor.md`](../../intent/14-llm-extractor.md) — the contract (5 ACs + 7 points for consideration).
- [`docs/plans/completed/14-llm-extractor.md`](../../plans/completed/14-llm-extractor.md) — Stage 14 plan including the 20-decision table + Ctrl+G resolution log.
- [`docs/lld/research/14-llm-extractor-scope-diff.md`](../../lld/research/14-llm-extractor-scope-diff.md) — `@minimalist` Phase-1 critique. The `@arch-reviewer` at Phase 3 applies the same four-tag taxonomy to the implementation diff.
- `docs/architecture/layers/ingestion.md` — Stage 13's bi-temporal store; `OUTPUT_SCHEMA` is the upstream contract that `RemitEvent` mirrors.
- ADR-0003 (`decisions/0003-protocol-for-model-interface.md`) — `typing.Protocol` over `abc.ABC` for swappable interfaces; the `Extractor` Protocol applies the same pattern.
- README.md §"Configuring an OpenAI API key" — operator-facing setup walkthrough.
