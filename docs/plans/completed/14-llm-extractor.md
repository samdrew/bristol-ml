# Plan — Stage 14: LLM feature extractor (stub + real)

**Status:** `approved` — Ctrl+G review on 2026-04-27 confirmed the four open questions (OQ-A: OpenAI, OQ-B: synthesise on NULL, OQ-C: target 80, OQ-D: stdout / loguru). Decision table updated to bind those answers. Ready for Phase 2.
**Intent:** [`docs/intent/14-llm-extractor.md`](../../intent/14-llm-extractor.md)
**Upstream stages shipped:** Stages 0–13 (foundation → ingestion → features → six model families → enhanced evaluation → registry → MLP → TCN → serving → REMIT bi-temporal store).
**Downstream consumers:** Stage 15 (embedding index — parallel thread on the same data), Stage 16 (feature-table join — extracted features become model inputs).
**Baseline SHA:** `da544db` (tip of `main` after Stage 13 + ADR-index follow-up merged).

**Discovery artefacts produced in Phase 1:**

- Requirements — [`docs/lld/research/14-llm-extractor-requirements.md`](../../lld/research/14-llm-extractor-requirements.md)
- Codebase map — [`docs/lld/research/14-llm-extractor-codebase.md`](../../lld/research/14-llm-extractor-codebase.md)
- Domain research — [`docs/lld/research/14-llm-extractor-domain.md`](../../lld/research/14-llm-extractor-domain.md)
- Scope Diff — [`docs/lld/research/14-llm-extractor-scope-diff.md`](../../lld/research/14-llm-extractor-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §"Demo moment" names a single moment: a facilitator runs the evaluation harness and sees a side-by-side comparison — gold, stub, and (when an API key is present) the real LLM — with disagreements visible so the conversation can be about *where the LLM gets it wrong*. Everything else — the env-var discriminator, the Pydantic schema, the prompt-hash provenance — is plumbing in service of that one moment, plus the contract Stages 15 and 16 will consume.

**Architectural weight.** The intent's load-bearing point (intent line 9) is *as much architectural as analytical*: any expensive or flaky external dependency in an ML system benefits from a stub-first design with an evaluation harness that holds the real implementation to account. Stage 14 is the first authenticated outbound call in the project; it sets the env-var pattern, the cassette pattern for an API-keyed endpoint, and the eval-harness pattern that future stub-first stages will copy.

**Upstream-data sharp edge.** The Stage 13 stream endpoint does not populate `message_description` in live responses (per the comment at `remit.py:431` and codebase map §"Hazards"). The free-text payload Stage 14 is meant to extract from is currently NULL on every live row and short synthetic strings on every stub row. **Stage 14 must explicitly choose a strategy** for the live path — accept NULLs and skip, or hydrate via a follow-up `GET /remit/{messageId}` call. See D6 below; this is the highest-priority external constraint.

---

## 1. Decisions for the human (drafted by lead, filtered through `@minimalist`, awaiting Ctrl+G)

Twenty decision points drawn from the three research artefacts and filtered through the [`Scope Diff`](../../lld/research/14-llm-extractor-scope-diff.md) `@minimalist` critique. The table below records the lead's draft framing, the minimalist's tag, and the binding disposition. Five rows were flipped or softened from the lead's draft framing:

- **D8 — softened** from "exactly 100 records" to "target ~80, range 50–120" per intent line 40.
- **D13 — CUT** in full. The `max_events_per_run` config field and `--sample N` CLI flag operationalise a problem the project does not yet have; researcher R7 confirms a full-archive run at Haiku 4.5 batch rates costs ~£16. **This is the single highest-leverage cut from the Scope Diff.** The stub-by-default discipline (D4) is the structural cost guard.
- **D15 — softened** from "stdout + timestamped markdown file under `docs/lld/llm/`" to "stdout-only with deterministic formatting". Simpler reproducibility test; AC-4 is satisfied by deterministic stdout.
- **D17, D20 — CUT.** Both restated decisions already in the intent's §"Out of scope" and the researcher's own recommendations; defending decisions never under threat is plan polish.
- **NFR-4 — CUT** as it binds D13.

The Evidence column cites the artefact that *resolved* each decision.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | Module location | **`src/bristol_ml/llm/extractor.py`** with `__init__.py` and module-local `CLAUDE.md`. | Mirrors the `CLAUDE.md` §"Module boundaries" entry: *`llm/` (deferred — Stages 14, 15)*. | Codebase map §1, §8; Scope Diff D1 (RESTATES INTENT). |
| **D2** | Public interface | **`Extractor` `Protocol`** in `src/bristol_ml/llm/__init__.py` with two methods: `extract(event: RemitEvent) -> ExtractionResult` and `extract_batch(events: list[RemitEvent]) -> list[ExtractionResult]`. | Implements AC-1 (small, implementation-agnostic). ADR-0003 sets the precedent for `Protocol` over `ABC` for swappable interfaces. The two-method shape is the smallest surface that still allows batch optimisations later. | Requirements AC-1; ADR-0003; codebase map §2. |
| **D3** | Two implementations + discriminated union | **`StubExtractor` and `LlmExtractor` in `extractor.py`**, selected by `LlmExtractorConfig` whose `type: Literal["stub", "openai"]` field is the Pydantic discriminator. The literal is intentionally narrow at Stage 14; a future open-weights / Anthropic / local-server slot can be added by extending the union without breaking callers. | Implements intent §Scope. Mirrors `ModelConfig` pattern at `conf/_schemas.py:931–941`. | Codebase map §2; Scope Diff D3 (RESTATES INTENT). |
| **D4** | Stub default + env-var discriminator | **`BRISTOL_ML_LLM_STUB=1`** activates the stub regardless of config. Default in CI, notebooks, and any path without an API key. The Hydra default `conf/llm/extractor.yaml` selects `type: stub`. | Implements AC-2 (offline by default). Mirrors `BRISTOL_ML_REMIT_STUB` precedent. The double-gating (env-var *or* config-type) means even a misconfigured live run cannot fire when the env-var is set — load-bearing for CI safety. | Codebase map §2; intent AC-2; Scope Diff D4 (RESTATES INTENT). |
| **D5** | API-key env var | **`BRISTOL_ML_LLM_API_KEY`** read once at `LlmExtractor.__init__`. If absent and `type != "stub"`, raise `RuntimeError` at init time with a message naming both the env-var and the `BRISTOL_ML_LLM_STUB=1` escape hatch. | Implements AC-3 sub-criterion (clear init-time error). Codebase map §3 confirms there is no existing env-var precedent — Stage 14 sets the pattern. | Codebase map §3; intent AC-3. |
| **D6** | LLM provider default + `message_description` strategy | **Provider:** OpenAI **GPT-4o-mini** via the `openai` Python SDK using `response_format={"type": "json_schema", "json_schema": {"name": "extraction_result", "strict": true, "schema": ...}}` (CFG token masking; GA since August 2024 — the most field-tested constrained-generation route across providers). Bound at Ctrl+G 2026-04-27 (OQ-A) — the human has an OpenAI account already, no provider rotation needed. **Strategy for missing `message_description`:** the extractor **accepts NULL `message_description`** as a valid input (bound at OQ-B). When NULL, it constructs a synthetic prompt input from the structured fields (`event_type`, `cause`, `fuel_type`, `affected_mw`, `effective_from`, `effective_to`) and proceeds. No follow-up `GET /remit/{messageId}` call at Stage 14. | Researcher R1.2 confirms OpenAI strict mode is GA with mature CFG token masking; R7 confirms GPT-4o-mini full-archive cost ≈ £12.50 — well below the cost-guardrail threshold (and one reason D13 is `PREMATURE OPTIMISATION`). **The NULL strategy is the lowest-friction path that ships a demo-able stage** — the structured fields are themselves the extraction signal in production REMIT data, so the LLM is being asked to *interpret + standardise* them. The hydration approach (~45,000 extra calls per archive run) is deferred to Stage 16 if needed. | Researcher R1.2, R7; codebase map §1, §"Hazards"; Scope Diff D6 (RESTATES INTENT); Ctrl+G 2026-04-27 (OQ-A, OQ-B). |
| **D7** | Prompt file location | **`conf/llm/prompts/extract_v1.txt`** as a plain-text template (no Jinja). Filename encodes the version. Active prompt path is a config field on `LlmExtractorConfig`. | Implements NFR-5 (intent line 45). Codebase has no prior prompt-file precedent; `conf/llm/` is consistent with `conf/ingestion/`, `conf/features/`, etc. | Researcher R5; intent line 45; Scope Diff D7 (PLAN POLISH → kept; low cost). |
| **D8** | Hand-labelled set size | **Target 80 records.** Stratified across event type (planned / unplanned / withdrawn) and the four most common fuel types in the Stage 13 cassette (Gas, Nuclear, Wind, Coal at minimum). Bound at Ctrl+G 2026-04-27 (OQ-C). | Intent line 40 ("dozens to low hundreds"). Researcher R3 §"Practical sizing" gives 100 records as a ±10 % margin reference at 80 % accuracy; 80 records is just below that threshold — adequate for the demo, light enough to curate. | Intent line 40; researcher R3; Ctrl+G 2026-04-27 (OQ-C). |
| **D9** | Hand-labelled set location | **`tests/fixtures/llm/hand_labelled.json`** — a JSON list of `{event: {...}, expected: {...}}` records. Versioned with the code; carries a `schema_version: int` field at the top level so additions are detectable. | OQ-2 default disposition. Intent line 46 ("in-repo for a small set"). At ~80 records × ~6 fields the file is small enough to commit; the `schema_version` field guards future additions. | Requirements OQ-2; intent line 46. |
| **D10** | Provenance on extraction output | **Extraction outputs carry prompt-version provenance**: every `ExtractionResult` includes `prompt_hash: str | None` (SHA-256 hex digest, first 12 chars; `None` for the stub) and `model_id: str | None` (`None` for the stub). When extraction results are persisted (e.g. by Stage 16's join), these become parquet columns. The exact column names and storage format are the implementer's call. | Implements NFR-5 + intent line 45. Researcher R5 §"Minimal-viable shape". The Pydantic model carries the fields; how they're persisted is downstream's problem. | Researcher R5; intent line 45. |
| **D11** | Two-metric agreement design | **The harness reports both** an exact-match-per-field metric and a tolerance/F1 metric on the same fields. The two numbers are printed side-by-side in the harness output so the gap is the demo lesson. | Implements intent line 41 ("different choices produce different numbers, which makes the metric choice a lesson itself"). Researcher R4 endorses this as the demo design. | Intent line 41; researcher R4. |
| **D12** | Specific tolerances | Representative thresholds named in the harness for the demo: **±5 MW for `affected_capacity_mw`; ±1 hour for `effective_from`/`effective_to`**. These are documented choices in the harness module docstring, not magic numbers in code. The implementer may revise based on the gold-set distribution observed during curation. | Researcher R4 suggests ±5 MW or ±10 % for capacity, ±1 h for time. We name representative numbers without binding the implementer to them. | Researcher R4; Scope Diff D12 (PLAN POLISH → softened). |
| ~~**D13**~~ | ~~Cost guardrail (`max_events_per_run` + `--sample N`)~~ | **CUT** per Scope Diff. **Single highest-leverage cut.** Researcher R7 confirms the full one-year archive at Haiku 4.5 batch rates costs ~£16. The stub-default discipline (D4) is the only structural cost guard the project needs at this stage. A developer running a deliberate full-archive backfill can self-guard with awareness; adding `max_events_per_run` to the YAML schema and `--sample N` to the CLI ahead of any evidence we need them is `PREMATURE OPTIMISATION`. | Researcher R7; Scope Diff D13. |
| **D14** | Pydantic extraction schema | **`ExtractionResult`** with fields: `event_type: str`, `fuel_type: str`, `affected_capacity_mw: float \| None`, `effective_from: datetime`, `effective_to: datetime \| None`, `confidence: float`, `prompt_hash: str \| None`, `model_id: str \| None`. All `datetime` fields are timezone-aware UTC consistent with Stage 13. `confidence` is a documented sentinel (`1.0` for stub hand-labelled hit, `0.0` for stub default fallback, fixed `1.0` for live — OpenAI's strict mode does not return per-token logprobs by default and the schema does not require calibrated probabilities). | Implements AC-5. The fields are the intent §Scope line 14 enumeration plus the D10 provenance pair. | Intent §Scope line 14; AC-5; D10. |
| **D15** | Eval harness output | **Stdout primary**, with a deterministic format (sorted records, redacted timestamps in test mode). The same content is also emitted as a single structured `loguru` INFO record (`extra={"summary": {...}}`) so the non-notebook (CLI / orchestration) run path captures the report into the standard log stream — bound at Ctrl+G 2026-04-27 (OQ-D, "stdout, or captured in logging in the non-notebook case"). No file write. The harness prints: header (implementation name, prompt hash, model id, gold-set hash + size), per-field summary table (exact match, tolerance match), then a side-by-side disagreement listing limited to the first 10 disagreements. | Implements AC-4. Stdout-primary is simpler to test for reproducibility (capture stdout, compare with timestamps redacted) than file-byte equality. The notebook displays the same table inline; the loguru capture means a CLI run leaves a structured trace without a markdown destination. | Intent line 28; researcher R5 (no registry dep needed); Scope Diff D15 (PLAN POLISH → softened); Ctrl+G 2026-04-27 (OQ-D). |
| **D16** | Graceful degradation | **On any LLM parse / validation failure:** log the raw response at DEBUG, log a WARNING with the event id and failure reason, return an `ExtractionResult` populated with the documented default (event_type/fuel_type from the structured fields if non-NULL, capacity = `None`, times mirrored from input, `confidence = 0.0`, `prompt_hash` and `model_id` recorded). **Never raise an unhandled exception from `extract`.** | Implements NFR-6 + intent line 47. Constrained decoding (D6) makes parse failures rare but not impossible (network truncation, schema-validator edge cases). | Intent line 47; NFR-6. |
| ~~**D17**~~ | ~~Restate "out of scope" list (prompt engineering, fine-tuning, ensembles, streaming)~~ | **CUT** per Scope Diff. The intent §"Out of scope, explicitly deferred" already names these. Plan restating the list defends a decision not under threat. | Scope Diff D17 (PLAN POLISH → cut). |
| **D18** | Real-LLM integration test via VCR | **One VCR cassette** recorded against OpenAI's Chat Completions endpoint at `api.openai.com/v1/chat/completions` (per OQ-A binding D6 to OpenAI). The cassette covers ~5 representative gold-set events. `vcr_config` filters `authorization`, `cookie`, `set-cookie`, `x-api-key` headers (already in the project's standard fixture). Cassette refresh is documented in the module's `CLAUDE.md` as a manual ritual. | Implements R-1 mitigation. Without this, the real path is never CI-tested and accumulates silent regressions. The cassette is the cheapest mechanism that runs in CI. | Researcher R1.1; codebase map §7; risk R-1. |
| **D19** | Notebook | **`notebooks/14_llm_extractor.ipynb`** — six cells: (i) bootstrap + load config; (ii) load gold set; (iii) run stub on the gold set; (iv) run the real LLM via VCR cassette (or skip with a printed banner if the cassette is absent); (v) print the side-by-side comparison; (vi) markdown discussion of metric choice. Reuses module logic; reimplements nothing (§2.1.8). | Implements intent §"Demo moment" line 28. Six cells is the minimum to surface the side-by-side comparison; the metric-choice cell is the pedagogical pause. | Intent line 28; codebase map §6; Scope Diff cell tags. |
| ~~**D20**~~ | ~~Plan negative-list ("we will NOT add `instructor`/`langchain`")~~ | **CUT** per Scope Diff. Researcher R2 already endorses this. Restating defends a decision not under threat. | Researcher R2; Scope Diff D20 (PLAN POLISH → cut). |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-1** | Offline by default (§2.1.3) | Stub path makes zero HTTP calls. Test asserts via `monkeypatch` of `httpx.Client` to raise on any request. | Requirements NFR-1; AC-2; DESIGN §2.1.3. |
| **NFR-2** | Typed boundary (§2.1.2) | Every value crossing the public interface is an `ExtractionResult` instance, never a raw dict. Pydantic validates at the boundary; type errors become `ValidationError`. | Requirements NFR-2; AC-5; DESIGN §2.1.2. |
| **NFR-3** | Eval harness reproducibility | Harness pins prompt hash, model id, gold-set hash + size, implementation name. Two runs with the same inputs produce byte-identical stdout (modulo a single redacted wall-clock timestamp on the header line). | Requirements NFR-3; AC-4. |
| ~~**NFR-4**~~ | ~~Cost control (`max_events_per_run`, `--sample N`)~~ | **CUT** per Scope Diff (binds D13). | Scope Diff NFR-4. |
| **NFR-5** | Prompt versioning | Prompt is a file under `conf/llm/prompts/`; SHA-256 hex digest (first 12 chars) is recorded in every `ExtractionResult`. Any change to the file produces a different hash. | Requirements NFR-5; intent line 45; researcher R5. |
| **NFR-6** | Graceful degradation | Parse / validation failures log + return default; never raise from `extract`. Test fixture with deliberately malformed mock LLM output exercises this. | Requirements NFR-6; intent line 47. |
| **NFR-7** | YAML config (§2.1.4) | Provider, model name, prompt-file path, API-key env-var name, implementation type all in `conf/llm/extractor.yaml`. No values hard-coded in the module. | Requirements NFR-7; DESIGN §2.1.4. |
| **NFR-8** | Module standalone (§2.1.1) | `python -m bristol_ml.llm.extractor --help` prints the active implementation name + gold-set size + sample extraction; exits 0. | Requirements NFR-8; DESIGN §2.1.1. |
| **NFR-9** | Observability | Use `loguru` consistent with the ingestion modules (INFO for non-trivial work, WARNING for parse failures, ERROR for init failures, DEBUG for cache / fallback paths). One test asserts on the WARNING + event id on parse failure (covers the load-bearing case). The exact level for every other call is the implementer's call. | Requirements NFR-9; Scope Diff NFR-9 (PLAN POLISH → softened). |

### Decisions and artefacts explicitly **not** in Stage 14

- **D13 / NFR-4** — Cost guardrail. Cut as the single highest-leverage cut.
- **D17, D20** — Plan-level restatements of the intent's own out-of-scope list and the researcher's framework recommendations. Cut as plan polish.
- **`message_description` hydration** via `GET /remit/{messageId}` — D6 chooses NULL-tolerant strategy. Hydration deferred to Stage 16 if its feature-join needs richer text.
- **An open-weights / local-server backend.** D3 leaves the `type` discriminator extensible but Stage 14 ships only `stub` and `openai`.
- **Embedding or semantic search** — Stage 15.
- **Use of extracted features in a model** — Stage 16.
- **Prompt engineering as an ongoing activity** — intent §"Out of scope".
- **Fine-tuning, distillation, ensembles, streaming** — intent §"Out of scope, explicitly deferred".
- **An ADR** for the LLM extractor interface or the env-var pattern. The extractor interface is captured in this plan + the Stage 14 layer doc; the env-var pattern is `BRISTOL_ML_LLM_API_KEY`, parallel to the existing stub-env-vars. Promote either to an ADR only if a future stage finds the choice contested.

### Open questions for Ctrl+G review (resolved)

All four blocking open questions were resolved at Ctrl+G review on 2026-04-27.

- **OQ-A — Provider choice (D6).** **Resolved → OpenAI GPT-4o-mini.** Drafted default was Anthropic Claude Haiku 4.5; the human has an OpenAI API account + credit already, so OpenAI is the lower-friction path and removes the need for a provider rotation in the demo. Researcher R1.2 confirms strict mode is GA since August 2024 and is the most field-tested constrained-generation route. Pricing ($0.15/$0.60 per MTok; ~£12.50 full archive per R7) is the lowest of the three majors.
- **OQ-B — `message_description` strategy (D6).** **Resolved at default.** Accept NULL and synthesise prompt input from the structured fields. Hydration via `GET /remit/{messageId}` deferred to Stage 16 if its feature-join needs richer text.
- **OQ-C — Hand-labelled set size (D8).** **Resolved → target 80.** Inside intent's "dozens to low hundreds" range; just below the 100-record reference for ±10 % margin at 95 % CI / 80 % accuracy (R3) — adequate for the demo, light enough to curate.
- **OQ-D — Eval-harness output (D15).** **Resolved at default with addendum.** Stdout-primary; same content also emitted as a single structured `loguru` INFO record so a non-notebook CLI run leaves a captured trace.

### Resolution log

- **Drafted 2026-04-26** — pre-Ctrl+G. All 20 decisions proposed; D13/NFR-4 cut, D17/D20 cut, D8/D12/D15/NFR-9 softened per `@minimalist` Scope Diff. Four open questions surfaced for human review.
- **Ctrl+G review 2026-04-27** — human accepted the cuts and softens; resolved OQ-A → OpenAI, OQ-B at default (synthesise on NULL), OQ-C → 80, OQ-D at default + loguru addendum. Status flipped `draft → approved` and ready for Phase 2.

---

## 2. Scope

### In scope

Transcribed from `docs/intent/14-llm-extractor.md §Scope`:

- **An interface that takes a REMIT event (or a batch) and returns structured features.** `Extractor` Protocol with `extract` / `extract_batch` (D2) returning `ExtractionResult` (D14).
- **A stub implementation backed by a small hand-labelled set.** `StubExtractor` reading `tests/fixtures/llm/hand_labelled.json` (D9), returning the labelled features for known events and a documented default (D16) for unknown events.
- **A real implementation that calls an LLM.** `LlmExtractor` calling OpenAI GPT-4o-mini via Chat Completions strict-mode `response_format={"type": "json_schema", ...}` (D6 — bound to OpenAI at OQ-A); guarded by env var (D5) and config switch (D3/D4).
- **An evaluation harness** that runs both implementations over the gold set and reports agreement: `python -m bristol_ml.llm.evaluate` printing the side-by-side comparison to stdout (D11/D15).
- **Configuration selecting the active implementation:** discriminated-union `LlmExtractorConfig` (D3) in `conf/llm/extractor.yaml`.

Additionally in scope as direct consequences of the above:

- **Hydra config + Pydantic schema** — `conf/llm/extractor.yaml`, `conf/llm/prompts/extract_v1.txt`, `LlmExtractorConfig` in `conf/_schemas.py`, `AppConfig.llm: LlmExtractorConfig | None = None` slot.
- **Layer doc** — new `docs/architecture/layers/llm.md` capturing the extractor's contract (Protocol surface, env-var pattern, stub-vs-live discriminator) since this is a new layer.
- **Module guide** — `src/bristol_ml/llm/CLAUDE.md` carrying the schema + the cassette-refresh ritual.
- **Stage retro + CHANGELOG.**

### Explicit out-of-scope

(See §1 "Decisions explicitly not in Stage 14".)

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/14-llm-extractor.md`](../../intent/14-llm-extractor.md) — the contract; 5 ACs and 7 "Points for consideration".
2. [`docs/lld/research/14-llm-extractor-requirements.md`](../../lld/research/14-llm-extractor-requirements.md) — US-1..US-5, AC-1..AC-5, NFR-1..NFR-9, OQ-1..OQ-8. The OQ-1..OQ-8 defaults are bound by §1's decisions (OQ-1 = D8; OQ-2 = D9; OQ-3 = D11; OQ-4 = D6 structured-output mode; OQ-5 = D6 OpenAI GPT-4o-mini per OQ-A binding; OQ-6 = D2 batch is `list[RemitEvent]`; OQ-7 = D15 stdout-only; OQ-8 = D7 prompt file).
3. [`docs/lld/research/14-llm-extractor-codebase.md`](../../lld/research/14-llm-extractor-codebase.md) — Stage 13 `OUTPUT_SCHEMA` (§1), env-var discriminator pattern (§2), discriminated-union pattern (§2), no existing API-key precedent (§3), `LlmExtractorConfig` placement (§4), notebook bootstrap (§6), VCR cassette fixture (§7), hazards (§9 — particularly the `message_description` NULL constraint).
4. [`docs/lld/research/14-llm-extractor-domain.md`](../../lld/research/14-llm-extractor-domain.md) — §R1.1 (Anthropic `output_config` mechanics, supported keywords, Haiku 4.5 pricing), §R2 (constrained decoding vs `instructor` — and why we don't add `instructor`), §R3 (gold-set sizing rationale), §R4 (agreement metrics), §R5 (prompt versioning — SHA-hash-in-output is minimum-viable), §R7 (cost — confirms D13 is `PREMATURE OPTIMISATION`).
5. [`docs/lld/research/14-llm-extractor-scope-diff.md`](../../lld/research/14-llm-extractor-scope-diff.md) — `@minimalist` critique; every cut and retention is listed there.
6. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
7. `docs/architecture/layers/ingestion.md` — Stage 13's bi-temporal storage shape; Stage 14 reads `OUTPUT_SCHEMA` rows, does not write back to the REMIT parquet.
8. `src/bristol_ml/ingestion/remit.py` — concrete upstream source. Read `OUTPUT_SCHEMA`, `MESSAGE_STATUSES`, `FUEL_TYPES`, `_stub_records`, `_parse_message`, `as_of`.
9. `conf/_schemas.py` — sibling schema patterns: `ServingConfig` (lines 988–1022) for "self-contained block, `None` in default config" precedent; `ModelConfig` (lines 931–941) for the discriminated-union pattern.
10. `tests/integration/ingestion/test_remit_cassettes.py` — the VCR cassette fixture pattern Stage 14 will replicate against the OpenAI Chat Completions endpoint (per OQ-A).
11. `tests/conftest.py` — the `loguru_caplog` fixture for asserting on log records (NFR-6, NFR-9).
12. `notebooks/13_remit_ingestion.ipynb` — the bootstrap-cell pattern + the `# T5 Cell N —` comment convention for notebooks.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five intent-ACs are copied verbatim from `docs/intent/14-llm-extractor.md §Acceptance criteria`, then grounded in one or more named tests.

- **AC-1 (intent).** *"The interface is small enough that writing a third implementation in the future is plausible."*
  - Tests:
    - `test_extractor_protocol_has_two_methods` — `Extractor` Protocol has exactly `extract` + `extract_batch`; no other public attributes.
    - `test_stub_and_llm_extractors_satisfy_protocol_structurally` — both classes pass an `isinstance(_, Extractor)` runtime-checkable test.
    - `test_llm_extractor_config_discriminator_supports_third_literal_slot` — extending the discriminator with a third literal value (e.g. `Literal["stub", "openai", "future"]`) parses cleanly; the config object resolves even if the implementation is not present.

- **AC-2 (intent).** *"The stub is the default; running anything that consumes the extractor works offline with no API key."*
  - Tests:
    - `test_stub_active_when_env_var_set` — `BRISTOL_ML_LLM_STUB=1` selects stub regardless of config `type`.
    - `test_stub_default_in_yaml_config` — `conf/llm/extractor.yaml` resolves to `type: stub` by default.
    - `test_stub_makes_no_network_call` — monkeypatch `httpx.Client.send` to raise; stub `extract` succeeds on a known-event input.
    - `test_stub_returns_default_for_unknown_event` — unknown event → `ExtractionResult` with `confidence=0.0` and structural-field-derived defaults.

- **AC-3 (intent).** *"The real implementation is guarded by a configuration switch and uses an environment variable for the API key."*
  - Tests:
    - `test_llm_extractor_init_raises_when_api_key_missing` — `type: openai`, `BRISTOL_ML_LLM_API_KEY` unset, `BRISTOL_ML_LLM_STUB` unset → `RuntimeError` at init naming both env-vars.
    - `test_llm_extractor_init_succeeds_when_api_key_present` — env-var set; init does not raise.
    - `test_llm_extractor_against_cassette` (integration) — VCR cassette covers ~5 events against `api.openai.com/v1/chat/completions`; assert structural correctness of the returned `ExtractionResult`s.
    - `test_model_name_and_endpoint_are_in_yaml_not_code` — grep-style test asserting no string starting `gpt-` or `claude-` appears literally in `src/bristol_ml/llm/`.

- **AC-4 (intent).** *"The evaluation harness produces a reproducible accuracy report against the hand-labelled set."*
  - Tests:
    - `test_evaluate_harness_runs_against_stub` — `python -m bristol_ml.llm.evaluate` exits 0 with stub config; stdout contains the per-field summary table.
    - `test_evaluate_harness_output_is_deterministic_modulo_timestamp` — capture stdout twice with stub config; redact the single header timestamp; assert byte-identical.
    - `test_evaluate_harness_records_provenance` — stdout header contains the prompt hash, model id, gold-set hash, gold-set size, implementation name.

- **AC-5 (intent).** *"The extracted feature schema is typed and validated at the interface boundary."*
  - Tests:
    - `test_extraction_result_pydantic_validation_on_bad_types` — constructing `ExtractionResult(affected_capacity_mw="not a number")` raises `ValidationError`.
    - `test_extraction_result_datetimes_are_utc_aware` — naive `datetime` inputs to `ExtractionResult.effective_from` raise.
    - `test_extraction_result_confidence_in_unit_interval` — values outside `[0.0, 1.0]` raise.
    - `test_schema_importable_without_concrete_implementations` — `from bristol_ml.llm import ExtractionResult, Extractor` succeeds without importing `extractor.py`.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_llm_extractor_config_round_trips_through_hydra` — `conf/llm/extractor.yaml` resolves to a valid `LlmExtractorConfig` and back.
- `test_app_config_llm_default_is_none_so_existing_callers_unaffected` — every prior stage's CLI continues to validate.
- `test_extractor_module_runs_standalone` — `python -m bristol_ml.llm.extractor --help` exits 0 (NFR-8).
- `test_malformed_llm_response_logs_warning_and_returns_default` — fixture mocks an `output_config` response that fails Pydantic validation; `extract` returns the default result and emits a WARNING with the event id (NFR-6).
- `test_prompt_hash_changes_when_prompt_file_changes` — write two prompt files with one-character difference; `prompt_hash` differs (NFR-5).
- `test_notebook_14_llm_extractor_executes_top_to_bottom` (integration) — `nbclient` against the cassette + stub fixture.

**Total shipped tests: ~21** — three AC-1, four AC-2, four AC-3, three AC-4, four AC-5, plus six D-derived (Hydra round-trip, AppConfig default, standalone CLI, malformed-LLM, prompt-hash, notebook). Refined during implementation.

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/llm/                 # NEW
├── __init__.py                     # Extractor Protocol; ExtractionResult; RemitEvent
├── extractor.py                    # StubExtractor + LlmExtractor + _cli_main
├── evaluate.py                     # eval harness; __main__ wires python -m bristol_ml.llm.evaluate
├── _prompts.py                     # prompt loading + SHA-256 hashing
└── CLAUDE.md                       # module guide + cassette-refresh ritual
```

`src/bristol_ml/llm/__init__.py` exports:

```python
__all__ = [
    "Extractor",          # Protocol
    "ExtractionResult",   # Pydantic model
    "RemitEvent",         # Pydantic model populated from Stage 13 row
]
```

`src/bristol_ml/llm/extractor.py` exports:

```python
__all__ = [
    "StubExtractor",
    "LlmExtractor",
    "build_extractor",   # factory: LlmExtractorConfig -> Extractor
]
```

### Pydantic models (in `__init__.py` so the schema is importable without the concrete classes)

```python
class RemitEvent(BaseModel):
    """One row of Stage 13's REMIT parquet, lifted into a typed boundary."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    mrid: str
    revision_number: int
    message_status: str
    published_at: datetime  # UTC-aware
    effective_from: datetime
    effective_to: datetime | None
    fuel_type: str | None
    affected_mw: float | None
    event_type: str | None
    cause: str | None
    message_description: str | None  # may be NULL — D6 strategy applies

class ExtractionResult(BaseModel):
    """Structured features extracted from a single RemitEvent."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str
    fuel_type: str
    affected_capacity_mw: float | None
    effective_from: datetime  # UTC-aware
    effective_to: datetime | None
    confidence: float          # in [0.0, 1.0]
    prompt_hash: str | None    # 12-char SHA-256 hex prefix; None for stub
    model_id: str | None       # e.g. "gpt-4o-mini"; None for stub
```

### Protocol (in `__init__.py`)

```python
@runtime_checkable
class Extractor(Protocol):
    """The Stage 14 extraction contract.

    Implementations: StubExtractor (default), LlmExtractor (real).
    Adding a third implementation requires no changes to this Protocol.
    """

    def extract(self, event: RemitEvent) -> ExtractionResult:
        ...

    def extract_batch(self, events: list[RemitEvent]) -> list[ExtractionResult]:
        ...
```

### `LlmExtractorConfig` (in `conf/_schemas.py`)

```python
class LlmExtractorConfig(BaseModel):
    """Stage 14 — LLM feature extractor."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["stub", "openai"]  # discriminator (D3)

    # Real-implementation fields (ignored when type == "stub")
    model_name: str | None = None       # e.g. "gpt-4o-mini"
    api_key_env_var: str = "BRISTOL_ML_LLM_API_KEY"
    prompt_file: Path | None = None     # e.g. conf/llm/prompts/extract_v1.txt
    request_timeout_seconds: float = 30.0

# AppConfig — add field:
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

`conf/config.yaml` does **not** add `llm` to the defaults list (parallel to `serving`). Consuming entry points (the LLM CLI, the evaluation harness, the notebook) compose `llm` in explicitly via Hydra override.

### Eval harness output format (stdout, deterministic)

```
=== Stage 14 LLM extractor evaluation ===
implementation: stub
prompt_hash:    none
model_id:       none
gold_set:       tests/fixtures/llm/hand_labelled.json (sha256: a1b2c3d4e5f6, n=80)
generated_at:   2026-04-26T14:23:11Z   (← redacted in tests)

per-field agreement:
  field                   exact   tolerance
  event_type              80/80    80/80
  fuel_type               80/80    80/80
  affected_capacity_mw    80/80    80/80    (tolerance: ±5 MW)
  effective_from          80/80    80/80    (tolerance: ±1 h)
  effective_to            80/80    80/80    (tolerance: ±1 h)

disagreements (first 10):
  (none — stub on its own gold set always agrees)

=== end ===
```

When `implementation: openai` is run from the cassette, the disagreement listing is the demo punch line — it shows where the LLM gets it wrong against the gold set. The same content is also emitted as a single structured `loguru` INFO record (`extra={"summary": {...}}`) so a non-notebook CLI run leaves a captured trace.

### Standalone CLI

```
$ uv run python -m bristol_ml.llm.extractor --help
usage: extractor [-h] [overrides ...]

Run a single extraction against the active configuration; print result.

$ uv run python -m bristol_ml.llm.evaluate --help
usage: evaluate [-h] [overrides ...]

Run the extractor over the hand-labelled set; print side-by-side report.
```

Both CLIs use the `bristol_ml.config.load_config()` boundary; Hydra-style overrides supported (e.g. `llm.type=openai`).

### Notebook structure (`notebooks/14_llm_extractor.ipynb`)

Six cells (per D19):

1. **Bootstrap** (`# T5 Cell 1 —`) — `chdir REPO_ROOT` + `sys.path` insert, identical to Stage 13's pattern.
2. **Load config + gold set** (`# T5 Cell 2 —`) — `load_config()`; load `tests/fixtures/llm/hand_labelled.json`; print size.
3. **Run stub** (`# T5 Cell 3 —`) — `build_extractor(cfg.llm)` with `BRISTOL_ML_LLM_STUB=1`; extract over the gold set; show first 3 results inline.
4. **Run real** (`# T5 Cell 4 —`) — try-except around `LlmExtractor` against VCR cassette; if cassette absent, print a banner explaining how to record one.
5. **Side-by-side comparison** (`# T5 Cell 5 —`) — print the same per-field summary table the harness prints; show disagreements inline as a pandas frame.
6. **Discussion** (`# T5 Cell 6 —`) — markdown cell on metric choice (intent line 41 — "the metric choice is a lesson itself").

---

## 6. Tasks (sequential — see CLAUDE.md §Phase 2 for sequencing rules)

Each task ends with one or more pytest invocations and a single git commit citing this plan task number. The `@tester` is spawned alongside or before each task per CLAUDE.md §"Tester timing".

### T1 — `LlmExtractorConfig` + Hydra config + `AppConfig.llm` slot.
1. Add `LlmExtractorConfig` to `conf/_schemas.py` per §5.
2. Add `llm: LlmExtractorConfig | None = None` to `AppConfig`.
3. Create `conf/llm/extractor.yaml` per §5.
4. Create `conf/llm/prompts/extract_v1.txt` placeholder (a working prompt is the implementer's call during T3/T4).
5. Do **not** add `llm` to `conf/config.yaml` defaults — parallel to `serving`.
- **Tests:** `test_llm_extractor_config_round_trips_through_hydra`, `test_app_config_llm_default_is_none_so_existing_callers_unaffected`.
- **Commit:** `Stage 14 T1: LlmExtractorConfig + Hydra config + AppConfig slot`.

### T2 — Pydantic models + `Extractor` Protocol + module skeleton.
1. Create `src/bristol_ml/llm/__init__.py` with `RemitEvent`, `ExtractionResult`, `Extractor` Protocol per §5.
2. Create `src/bristol_ml/llm/CLAUDE.md` skeleton (will be filled out in T6).
3. The Protocol + Pydantic models are testable without any extractor implementation — they ship first to lock the boundary.
- **Tests (AC-1 partial, AC-5):** `test_extractor_protocol_has_two_methods`, `test_extraction_result_pydantic_validation_on_bad_types`, `test_extraction_result_datetimes_are_utc_aware`, `test_extraction_result_confidence_in_unit_interval`, `test_schema_importable_without_concrete_implementations`.
- **Commit:** `Stage 14 T2: ExtractionResult + RemitEvent + Extractor Protocol`.

### T3 — `StubExtractor` + gold-set fixture + standalone CLI + factory.
1. Create `tests/fixtures/llm/hand_labelled.json` with ~10 records bootstrapped from Stage 13's `_stub_records()` synthetic descriptions, plus 5–10 hand-written entries covering the four fuel types named in D8. (Full ~80-record curation is T5 work; T3 just needs enough records to make the stub testable.)
2. Implement `StubExtractor` in `src/bristol_ml/llm/extractor.py`. Behaviour: load JSON at init; `extract(event)` returns the labelled record if `(mrid, revision_number)` matches; otherwise returns the documented default per D16.
3. Implement `build_extractor(config)` factory dispatching on `config.type` and the `BRISTOL_ML_LLM_STUB` env var per D4.
4. Implement `_cli_main(argv)` for `python -m bristol_ml.llm.extractor`.
- **Tests (AC-1 ctd, AC-2, AC-5 ctd, NFR-1, NFR-8):** `test_stub_and_llm_extractors_satisfy_protocol_structurally` (stub half), `test_stub_active_when_env_var_set`, `test_stub_default_in_yaml_config`, `test_stub_makes_no_network_call`, `test_stub_returns_default_for_unknown_event`, `test_extractor_module_runs_standalone`.
- **Commit:** `Stage 14 T3: StubExtractor + gold-set fixture + factory + standalone CLI`.

### T4 — `LlmExtractor` against OpenAI + prompt loading + VCR cassette.
1. Add `openai` to `pyproject.toml` dependencies.
2. Implement `_prompts.py` — load prompt file, compute SHA-256 hex prefix.
3. Implement `LlmExtractor` in `extractor.py`. `__init__` reads the API key from the configured env var; raises with both env-vars named on absence (D5). `extract` calls `openai.chat.completions.create` with `response_format={"type": "json_schema", "json_schema": {"name": "extraction_result", "strict": True, "schema": ExtractionResult.model_json_schema()}}` per researcher R1.2; constructs `ExtractionResult` from the validated response; on any validation failure logs + returns the default per D16. **Schema-shape note:** OpenAI strict mode requires `additionalProperties: false` on every object and every field listed in `required` (no implicit optional). `ExtractionResult.model_json_schema()` may need a small post-processor to add these — the implementer should verify the first cassette recording exercises this path.
4. Record one VCR cassette under `tests/fixtures/llm/cassettes/test_llm_extractor_against_cassette.yaml` covering ~5 representative gold-set events against `api.openai.com/v1/chat/completions`. `vcr_config` fixture filters `authorization`, `cookie`, `set-cookie`, `x-api-key` per the existing project convention (covers the `Authorization: Bearer sk-...` header automatically).
5. The implementer chooses whether to fill out `conf/llm/prompts/extract_v1.txt` here or as part of T5 — either is fine.
- **Tests (AC-1 ctd, AC-3, NFR-5, NFR-6):** `test_stub_and_llm_extractors_satisfy_protocol_structurally` (LLM half), `test_llm_extractor_config_discriminator_supports_third_literal_slot`, `test_llm_extractor_init_raises_when_api_key_missing`, `test_llm_extractor_init_succeeds_when_api_key_present`, `test_llm_extractor_against_cassette` (integration), `test_model_name_and_endpoint_are_in_yaml_not_code`, `test_malformed_llm_response_logs_warning_and_returns_default`, `test_prompt_hash_changes_when_prompt_file_changes`.
- **Commit:** `Stage 14 T4: LlmExtractor against OpenAI + prompt hashing + VCR cassette`.

### T5 — Evaluation harness + gold-set curation + harness CLI.
1. Curate the gold set up to D8's range (target ~80 records). Stratify across event type and the four most-common fuel types per D8. Source records from the Stage 13 cassette + the existing stub records.
2. Implement `evaluate.py` with the per-field exact + tolerance metrics (D11/D12); print the deterministic stdout format from §5. Wire `python -m bristol_ml.llm.evaluate`.
3. Implementation note: the harness's "redacted-timestamp test mode" is the simplest production toggle — a `--redact-timestamps` flag, default off, that the test sets.
- **Tests (AC-4):** `test_evaluate_harness_runs_against_stub`, `test_evaluate_harness_output_is_deterministic_modulo_timestamp`, `test_evaluate_harness_records_provenance`.
- **Commit:** `Stage 14 T5: evaluation harness + gold-set curation + harness CLI`.

### T6 — Notebook + notebook smoke test.
1. Create `notebooks/14_llm_extractor.ipynb` per §5 (six cells).
2. Notebook smoke test follows Stage 13's pattern: `nbclient` against the cassette + stub fixture; CI default is `BRISTOL_ML_LLM_STUB=1`; cassette absence printed banner on Cell 4.
- **Tests:** `test_notebook_14_llm_extractor_executes_top_to_bottom`.
- **Commit:** `Stage 14 T6: extractor notebook + smoke test`.

### T7 — Documentation.
1. Create `docs/architecture/layers/llm.md` capturing: the Protocol contract, the env-var pattern, the stub-vs-live discriminator, the prompt-hash provenance convention. Add the layer row to `docs/architecture/README.md`'s layer index.
2. `src/bristol_ml/llm/CLAUDE.md` — module guide: schema table for `ExtractionResult`, the cassette-refresh ritual (manual; how to re-record + how to redact secrets), the gold-set curation guidance.
3. README — short bullet under a new "LLM" section linking to the new layer doc and the notebook.
4. `docs/lld/stages/14-llm-extractor.md` — retro skeleton; observed cassette size + final gold-set size recorded here.
5. `CHANGELOG.md` — `### Added` bullet under `[Unreleased]`.
- **Tests:** none (doc edits).
- **Commit:** `Stage 14 T7: layer doc + module guide + README + retro skeleton + CHANGELOG`.

### T8 — Stage hygiene + plan move.
1. `git mv docs/plans/active/14-llm-extractor.md docs/plans/completed/14-llm-extractor.md`.
2. Final retro updates: actual cassette size, actual gold-set count, any decisions deviated from in-stage.
3. `uv run pytest -q` clean; `uv run ruff check .` clean; `uv run ruff format --check .` clean; `uv run pre-commit run --all-files` clean.
- **Commit:** `Stage 14 T8: stage hygiene + retro + plan moved to completed/`.

### T9 — Phase 3 review.
Spawn `arch-reviewer` (conformance to plan + intent — particular focus on the AC-1 "small interface" promise and the AC-2 offline-by-default invariant), `code-reviewer` (code quality + security — particular focus on the API-key handling and VCR cassette filtering), `docs-writer` (user + developer docs sweep) in parallel. Synthesise findings, address Blocking items in-branch, surface Major+Minor in the PR description.

---

## 7. Exit checklist

Before opening the PR:

- [ ] All ~21 named tests in §4 pass; full `uv run pytest -q` is clean.
- [ ] All four AC-2 stub-default tests pass — the offline-default invariant is non-negotiable.
- [ ] VCR cassette ≤ 100 kB; integration test against cassette is deterministic across two runs.
- [ ] `BRISTOL_ML_LLM_API_KEY` never appears in any committed file (cassette filter verified manually).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `uv run python -m bristol_ml.llm.extractor --help` and `uv run python -m bristol_ml.llm.evaluate --help` exit 0.
- [ ] Layer doc `docs/architecture/layers/llm.md` exists; layer index in `docs/architecture/README.md` lists it.
- [ ] Module guide `src/bristol_ml/llm/CLAUDE.md` carries the schema + cassette-refresh ritual.
- [ ] README has a brief reference to the new module + the notebook.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`: extractor module + evaluation harness listed under `### Added`.
- [ ] Retro at `docs/lld/stages/14-llm-extractor.md` carries the observed cassette size + gold-set size + any deviations.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/`.
- [ ] PR description surfaces:
  - The Protocol contract for Stage 15 / Stage 16 (the two methods + `ExtractionResult` schema).
  - The `message_description` NULL strategy chosen at D6 — for Stage 16's join planner.
  - The `BRISTOL_ML_LLM_API_KEY` env-var pattern + cassette-refresh ritual — for the next authenticated-API stage's plan author.
  - The Scope-Diff cuts (D13/NFR-4 = single highest-leverage cut, D17/D20 cut, D8/D12/D15/NFR-9 softened) — for the stage retro narrative.
