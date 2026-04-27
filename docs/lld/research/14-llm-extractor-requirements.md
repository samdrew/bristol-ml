# Stage 14 — LLM feature extractor: requirements

**Source intent:** `docs/intent/14-llm-extractor.md`
**Artefact role:** Phase 1 research deliverable (requirements analyst).
**Audience:** plan author (lead), `@minimalist` pre-synthesis critic, Ctrl+G reviewer.

---

## 1. Goal

Define a narrow, typed interface that turns free-text REMIT event descriptions into structured features, backed by two interchangeable implementations — a hand-labelled stub (the default) and an LLM-calling real implementation — so that CI, notebooks, and attendees without API keys all work offline, while a live evaluation harness demonstrates concretely where the LLM succeeds and fails against ground truth.

---

## 2. User stories

**US-1 — Meetup facilitator (evaluation demo).**
Given the hand-labelled set is present in the repository and no LLM API key is configured,
when the facilitator runs the evaluation harness,
then a side-by-side report is printed: hand-labelled truth, stub output (trivially correct on known events), and a column for the real LLM output if an API key is available — so the conversation can be about "here is where the LLM got it wrong, and here is what that tells us about extraction reliability" (intent line 28).

**US-2 — CI runner / attendee (offline, stub path).**
Given the extractor is configured to use the stub implementation (the default),
when any code that consumes extracted features runs in CI or on an attendee's offline clone,
then the extractor returns typed, schema-valid features for known events and a documented default for unknown events, with zero network calls and no API key required (intent line 15; AC-2).

**US-3 — Stage 16 model builder (downstream consumer).**
Given Stage 13 has populated the REMIT cache and Stage 14 has run against it,
when Stage 16 joins extracted features onto the hourly feature table,
then each event carries at minimum: event type, affected fuel type, affected capacity in MW, normalised start/end times, and a confidence indicator — all as named fields on a Pydantic-validated model, so Stage 16 can reference fields by name rather than column index (intent lines 14–15; AC-5).

**US-4 — Stage 15 embedding index (parallel downstream consumer).**
Given Stage 14 exposes a batch extraction call,
when Stage 15 passes a list of REMIT events through the extractor interface,
then it receives a list of structured feature objects without needing to know whether the stub or real implementation is active — the interface is implementation-agnostic (intent line 53; AC-1).

**US-5 — Developer with LLM API key (real implementation).**
Given the configuration switch for the real implementation is set and a valid API key is present as an environment variable,
when the developer calls the extractor on a REMIT event,
then the LLM is invoked, its response is parsed and validated against the extraction schema, and any malformed response is caught and logged rather than raised — so the failure mode is a recorded default, not a crash (intent line 47; AC-3).

---

## 3. Acceptance criteria

**AC-1 — Interface is small and implementation-agnostic.** (Intent line 32)
The public interface accepts a single REMIT event object (or a batch) and returns a typed extraction result. The interface must not reference the LLM, the prompt, or the stub data store. Writing a third implementation must require no changes to the interface module.
- Sub-criterion: the interface is expressible as a Python `Protocol` or abstract base class with at most two methods (single-event and batch variants).
- Sub-criterion: Stage 15 and Stage 16 depend only on the interface type, not on either concrete class.

**AC-2 — Stub is the default; offline operation is unconditional.** (Intent line 34)
The Hydra configuration default selects the stub implementation. Running any downstream consumer (notebook, Stage 15, Stage 16, the evaluation harness) without an API key and without network access must complete successfully.
- Sub-criterion: there is no import-time or initialisation-time network call in the stub path.
- Sub-criterion: `CachePolicy.OFFLINE` equivalence — if the stub cannot find a hand-labelled entry for an event, it returns a documented default feature object, not an error.
- Sub-criterion: at least one CI test exercises the full stub path end-to-end.

**AC-3 — Real implementation is configuration-gated and key-guarded.** (Intent line 35)
The real LLM implementation is activated only when a named configuration switch is set. The API key is read exclusively from a named environment variable; it is never hard-coded or committed.
- Sub-criterion: if the configuration selects the real implementation but the environment variable is absent, the extractor raises a clear, user-readable error at initialisation time, not at inference time.
- Sub-criterion: the model name and endpoint are in YAML configuration (§2.1.4), not in code.

**AC-4 — Evaluation harness produces a reproducible report.** (Intent line 36)
Running the evaluation harness twice over the same hand-labelled set and the same implementation produces byte-identical output (modulo wall-clock timestamps).
- Sub-criterion: the report records which implementation was evaluated, the prompt version (for the real implementation), and the hand-labelled set's file path or content hash.
- Sub-criterion: the harness is runnable as `python -m bristol_ml.llm.evaluate` with no required arguments beyond configuration (§2.1.1).
- Sub-criterion: agreement metrics are computed per-field, not only as a single aggregate, so per-field accuracy differences are visible in the demo moment.

**AC-5 — Extraction schema is typed and validated at the interface boundary.** (Intent line 37)
Every value that crosses the interface boundary is an instance of a Pydantic model, not a raw dict. The schema includes at minimum: `event_type`, `fuel_type`, `affected_capacity_mw`, `effective_from`, `effective_to`, `confidence`. Type violations raise `ValidationError` at the boundary.
- Sub-criterion: the schema is importable from `bristol_ml.llm` without importing either concrete implementation.
- Sub-criterion: `effective_from` and `effective_to` are timezone-aware UTC `datetime` fields, consistent with the Stage 13 temporal convention.
- Sub-criterion: `confidence` has a documented range and meaning, and the stub always returns a fixed sentinel value (e.g. `1.0` for hand-labelled entries, `0.0` for the default fallback).

---

## 4. Non-functional requirements

**NFR-1 — Offline by default (§2.1.3; AC-2).**
The stub implementation must complete without any network I/O. This is enforced by test: at least one test must assert that no HTTP call is made when the stub is active (e.g. by monkeypatching `httpx` or `requests` to raise on any call).

**NFR-2 — Typed and validated interface boundary (§2.1.2; AC-5).**
Downstream code (Stage 15, Stage 16) must never receive a raw `dict` from the extractor. The Pydantic model is the contract. If the LLM returns a structurally valid JSON blob that fails Pydantic validation (wrong types, missing fields), the real implementation logs the failure at WARNING level and returns the documented default, consistent with the graceful-degradation pattern (intent line 47).

**NFR-3 — Evaluation harness reproducibility (AC-4).**
The harness must pin: the hand-labelled set version (file path + content hash in the report header), the implementation name, and — for the real implementation — the prompt version identifier and model name. Reports produced without these pins are non-compliant.

**NFR-4 — Cost control (intent §"Points for consideration", line 44).**
The real implementation must not be invoked during CI, notebook startup, or any path where the stub would suffice. Configuration default is stub. A `--dry-run` or `--sample N` flag on the evaluation harness CLI limits real LLM calls to at most N events. The YAML config must include a `max_events_per_run` guard that defaults to a small number (suggested: 20) to prevent accidental full-archive extraction.

**NFR-5 — Prompt versioning and artefact provenance (intent line 45; §2.1.6).**
The prompt used for any real extraction is versioned: it lives in a file under `conf/llm/` (not in source code), carries a version identifier, and is recorded in every extraction result's metadata. Any registered feature-extraction batch must be traceable to the prompt version that produced it.

**NFR-6 — Graceful degradation on malformed LLM output (intent line 47).**
If the LLM returns output that cannot be parsed or validated, the real implementation must: (a) log the raw response at DEBUG level, (b) log a WARNING with the event identifier and failure reason, (c) return the documented default feature object. It must not raise an unhandled exception. This behaviour must be covered by at least one test using a fixture with deliberately malformed LLM output.

**NFR-7 — Configuration in YAML (§2.1.4).**
LLM provider, model name, endpoint URL, API key environment-variable name, `max_events_per_run`, prompt file path, and implementation selector all live in `conf/llm/extractor.yaml`. No values hard-coded in the module.

**NFR-8 — Module standalone (§2.1.1).**
`python -m bristol_ml.llm.extractor` (or an equivalent entry point) runs without error and prints a meaningful summary — e.g. the active implementation name, the hand-labelled set size, and a sample extraction from one known event.

**NFR-9 — Observability (§2.1 cross-cutting).**
The real implementation emits structured `loguru` log lines at INFO for each LLM call (event ID, prompt version, response latency), at WARNING for parse failures, and at ERROR for initialisation failures (missing key). The stub emits at DEBUG for cache hits and at DEBUG for default-fallback events.

---

## 5. Open questions

**OQ-1 — What is the right size and composition of the hand-labelled set?** (Intent line 40)
The set must be large enough that the accuracy estimate is meaningful, small enough to curate without becoming a sub-project, and representative enough to cover the main event types and fuel types.
*Default disposition:* 50–80 events, stratified by event type (planned / unplanned) and by the six most common fuel types in the REMIT archive. Evidence needed to bind: a frequency table from Stage 13's parquet (event type × fuel type distribution) run before the plan is finalised.

**OQ-2 — Where does the hand-labelled set live?** (Intent line 46)
Options: (a) a small JSON or CSV committed directly to the repo under `src/bristol_ml/llm/` or `tests/fixtures/llm/`; (b) a separate data file under `data/` with a documented source and a seeding script; (c) a VCR-style cassette alongside tests.
*Default disposition:* a JSON file committed to `tests/fixtures/llm/hand_labelled.json`, versioned with the code. Its small size (50–80 events × 6 fields) makes in-repo storage defensible. The file carries a schema version field so future additions are detectable. Evidence needed to rebind: if the set grows above ~200 events or contains raw free text (copyright concern), move to `data/` with a gitignored convention.

**OQ-3 — What counts as "agreement" between stub and LLM?** (Intent line 41)
Exact-match on every field is harsh (capacity figures may differ by rounding); semantic-match on key fields only (event type, fuel type, approximate capacity within ±10 %) is more forgiving. The choice is itself a pedagogical point.
*Default disposition:* report both. Exact-match per field is the primary metric (clearest); a relaxed "capacity within ±10 % and event type matches" composite is reported as a secondary metric. The harness should make the metric definition explicit in the report header, not buried in code. Evidence needed: none — both metrics cost the same to compute; commit to reporting both.

**OQ-4 — Structured-output API (JSON-schema mode) versus free-form-then-parse?** (Intent line 43)
JSON-schema-constrained mode (supported by OpenAI and Anthropic) reduces parse failures but may reduce extraction quality. Free-form-then-parse with a permissive regex/json parser gives the LLM more latitude but requires robust error handling.
*Default disposition:* use structured-output mode if the chosen provider supports it, because it simplifies the graceful-degradation path and is easier to teach. Record the choice as a config field (`response_mode: structured | freeform`) so it can be switched in a demo. Evidence needed: confirm the chosen provider's structured-output capability before the plan is written.

**OQ-5 — Which LLM provider and model?** (Intent line 43, implied)
The intent does not name a provider. Options: OpenAI GPT-4o-mini (low cost, structured output), Anthropic Claude Haiku (similarly low cost), a local ollama model (zero cost, offline-capable). The choice affects cost, latency, and whether the real implementation can ever run in CI.
*Default disposition:* OpenAI GPT-4o-mini as the default real implementation, because it is the most widely held API key among meetup attendees and has documented structured-output support. The provider is a config value, not a hard-coded import, so switching costs only a YAML change. Evidence needed: confirm GPT-4o-mini's JSON-schema mode is stable before the plan is written.

**OQ-6 — Should the batch interface accept a `pd.DataFrame` or a `list[RemitEvent]`?** (Intent line 14, downstream)
Stage 15 (embedding index) will likely pass a batch; Stage 16 (feature join) may call single-event or batch. A typed list is more explicit and easier to test; a DataFrame matches the existing ingestion boundary convention but loses field names in transit.
*Default disposition:* `list[RemitEvent]` for the batch interface, where `RemitEvent` is a Pydantic model populated from the Stage 13 parquet schema. Callers convert from DataFrame to list at their own boundary; this keeps the LLM interface free of pandas. Evidence needed: confirm Stage 15 and Stage 16 authors are comfortable with the conversion cost (one `df.itertuples()` call each).

**OQ-7 — How should the evaluation harness integrate with the registry?** (Intent line 36, §2.1.6)
Options: (a) the harness writes its report to `docs/lld/` as a markdown file (human-readable, no registry dependency); (b) the harness registers the report as a registry artefact (full provenance, but adds a Stage 9 dependency); (c) the harness prints to stdout only (simplest, ephemeral).
*Default disposition:* option (a) — write a timestamped markdown report to `docs/lld/llm/` and print a summary to stdout. No registry dependency at Stage 14; if Stage 16 needs evaluation artefacts in the registry, that is Stage 16's problem. Evidence needed: none — option (a) is the lowest-dependency path that still produces a demoable artefact.

**OQ-8 — Where does the prompt file live and how is it versioned?** (Intent line 45)
Options: (a) `conf/llm/prompts/v1.txt` with a version field in `extractor.yaml`; (b) inline in `extractor.yaml` as a multi-line string; (c) embedded in the Python module.
*Default disposition:* option (a) — a separate file under `conf/llm/prompts/`, with the filename encoding the version (e.g. `extract_v1.txt`). The active prompt path is a config field in `extractor.yaml`. This makes "we swapped the prompt and everything changed" diagnosable from the config diff alone, without reading Python. Evidence needed: none — this is the lowest-friction versioning path consistent with §2.1.4.

---

## 6. Risks and surprises

**R-1 — Stub becomes the only path that is ever tested.**
Because the stub is offline and fast, CI will naturally run only the stub path. The real implementation may accumulate silent regressions (prompt changes, provider API changes, schema drift) that are never caught until a facilitator runs the evaluation harness live. *Mitigation:* at least one integration test must exercise the real implementation against a recorded VCR cassette of a real LLM response (not a mock). The cassette must be refreshed at least once per stage that touches the prompt.

**R-2 — LLM provider changes pricing, deprecates a model, or alters structured-output behaviour.**
This has happened repeatedly with OpenAI and Anthropic. A model deprecation mid-stage breaks the real implementation without any code change.
*Mitigation:* the model name is a config value. The evaluation harness report records the model name so "it used to score 92 % and now scores 71 %" is diagnosable. Document a model-rotation procedure in the module's `CLAUDE.md`.

**R-3 — Hand-labelled set bias.**
If the 50–80 labelled events are drawn from a single season or a single fuel type, the accuracy estimate will not generalise. The facilitator may quote "92 % agreement" in a demo that is actually measuring "92 % on nuclear planned outages in winter 2022."
*Mitigation:* OQ-1 requires stratified sampling. The harness report must print the per-stratum breakdown, not only the overall figure.

**R-4 — Schema drift between Stage 13's parquet and Stage 14's `RemitEvent` model.**
Stage 13 may rename or retype a field after Stage 14 is written. Because the two stages are in the same repo, this is detectable — but only if there is an integration test that reads real Stage 13 parquet and passes it through the extractor interface.
*Mitigation:* AC-5 and AC-1 together require a typed boundary. Add an integration test at Stage 14 that reads the Stage 13 fixture parquet and asserts that `RemitEvent` construction succeeds without field errors.

**R-5 — Cost explosion during development.**
A developer iterating on the prompt or the evaluation harness may inadvertently call the real LLM hundreds of times. With no guard, this can produce an unexpected API bill before the session ends.
*Mitigation:* NFR-4 mandates a `max_events_per_run` config guard with a small default. The harness CLI must print the cost estimate (token count × price per token) before making any LLM calls and prompt for confirmation when `--yes` is not passed.

**R-6 — "Confidence" field is meaningless in the stub.**
The stub always returns `confidence=1.0` for known events. Stage 16 may consume this field for weighting or filtering. If Stage 16 treats the stub's `1.0` as a real probability, it will silently over-weight all REMIT features in training.
*Mitigation:* document clearly in the schema that the stub's confidence is a sentinel, not a calibrated probability. Stage 16 should be warned in the downstream-consumer note in the module's `CLAUDE.md`.

**R-7 — The evaluation harness is the only place the real implementation is validated, and it requires manual running.**
If the harness is not run before a stage ships, the real implementation can be broken at merge time with no CI signal.
*Mitigation:* require a harness run (with cassette, not live LLM) as part of the stage definition of done. The cassette approach in R-1 is the mechanism.

---

*This artefact is one of four Phase-1 research inputs for Stage 14. It covers requirements only. API surface detail, codebase patterns, and scope boundaries are in the companion artefacts.*
