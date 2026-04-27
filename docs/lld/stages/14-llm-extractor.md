# Stage 14 — LLM feature extractor (stub + real)

## Goal

Land the project's first **authenticated outbound dependency** behind
a typed, swappable boundary. Three targets, in priority order:

1. Ship the `Extractor` Protocol + `RemitEvent` / `ExtractionResult`
   Pydantic models so Stage 15 (embedding index) and Stage 16
   (feature-table join) can depend on the boundary without depending
   on the LLM.
2. Ship two interchangeable backends — `StubExtractor` (offline,
   default; reads a hand-labelled gold set) and `LlmExtractor` (live;
   OpenAI Chat Completions strict mode) — selected by config + env
   var, triple-gated for CI safety.
3. Ship the evaluation harness — `python -m bristol_ml.llm.evaluate`
   — that prints a deterministic side-by-side accuracy report
   (exact-match + tolerance-match) against the gold set. The
   side-by-side is the demo moment (intent §"Demo moment", line 28);
   the metric-choice gap is the pedagogical pause (intent line 41).

## What was built

- `src/bristol_ml/llm/__init__.py` — boundary types: `RemitEvent`
  (typed mirror of Stage 13's `OUTPUT_SCHEMA` row), `ExtractionResult`
  (Pydantic model carrying `prompt_hash` + `model_id` provenance),
  and `Extractor` `runtime_checkable` Protocol with exactly two
  methods (AC-1 cap). `extra="forbid"`, `frozen=True` on both
  Pydantic models; UTC-aware datetime validators reject naive
  timestamps at construction (consistent with Stage 13's
  `OUTPUT_SCHEMA` `tz=UTC` pinning).
- `src/bristol_ml/llm/extractor.py` — `StubExtractor` (loads
  `tests/fixtures/llm/hand_labelled.json` at init; indexes by
  `(mrid, revision_number)`; returns the labelled extraction on a
  hit, the documented default on a miss); `LlmExtractor` (calls
  `openai.chat.completions.create` with strict
  `response_format={"type": "json_schema", ...}`; CFG token masking;
  GA since August 2024); `build_extractor(config)` factory dispatching
  on `config.type` *and* the `BRISTOL_ML_LLM_STUB` env var (plan §1
  D4); standalone CLI `python -m bristol_ml.llm.extractor`.
- `src/bristol_ml/llm/_prompts.py` — SHA-256 hashing + prompt loading.
  `load_prompt(path)` returns `(text, prompt_hash)` where
  `prompt_hash` is the first 12 hex chars of the SHA-256 digest of
  the file's bytes. Hash is computed once at `LlmExtractor.__init__`
  so a stale or missing prompt fails loudly at construction rather
  than at the first call.
- `src/bristol_ml/llm/evaluate.py` — `evaluate(config) ->
  EvaluationReport`; `format_report(report)` renders the deterministic
  stdout block (header → per-field exact + tolerance counts →
  disagreement listing). Tolerances ±5 MW capacity / ±1 h
  timestamps (plan §1 D12). `--redact-timestamps` for byte-equality
  tests; `--max-disagreements N` for legibility. The same content is
  emitted as a structured `loguru` INFO record
  (`logger.bind(summary={...}).info(...)`) so a CLI run leaves a
  captured trace (Ctrl+G OQ-D / D15). Standalone CLI
  `python -m bristol_ml.llm.evaluate`.
- `conf/_schemas.py` — new `LlmExtractorConfig` Pydantic model
  (`type: Literal["stub", "openai"]` discriminator, `model_name`,
  `api_key_env_var`, `prompt_file`, `request_timeout_seconds`).
  `AppConfig.llm: LlmExtractorConfig | None = None` slot — the `None`
  default keeps every prior stage's CLI / config-smoke test
  unaffected.
- `conf/llm/extractor.yaml` — Hydra group file mirroring the schema
  defaults; `type: stub` so the offline path is the runtime default.
  Group is *not* listed in `conf/config.yaml`'s defaults — entry
  points compose it via `+llm=extractor` (parallel to `serving`).
- `conf/llm/prompts/extract_v1.txt` — the active extraction prompt;
  hand-authored field guidance plus the `{event_json}` placeholder
  that the user message substitutes.
- `tests/fixtures/llm/hand_labelled.json` — 76-record gold set
  (schema_version 1) stratified across event type (Outage 51,
  Restriction 25) and 11 fuel types (Gas 18, Nuclear 13, Wind 13,
  Solar 8, Coal 6, Hydro 5, Pumped Storage 4, Biomass 3, Oil 2,
  Interconnector 2, Battery 2). Synthesise-on-NULL probes
  (M-NA..M-ND, `message_description=None`) and a missing-capacity
  probe (M-NM, `affected_mw=None`) exercise the live extractor's
  NULL-tolerance.
- `tests/integration/llm/test_llm_extractor_cassette.py` — VCR
  cassette test against `api.openai.com/v1/chat/completions`
  covering 5 strata-spanning gold-set events (M-A Nuclear, M-B Gas,
  M-C Coal, M-H synthesise-on-NULL probe, M-I Solar). `vcr_config`
  filters `authorization`, `cookie`, `set-cookie`, `x-api-key`.
  Replay-only under the project default `--record-mode=none`; skips
  when the cassette is absent (no key in this dev environment, plan
  build-up phase).
- `notebooks/14_llm_extractor.ipynb` — seven-cell demo notebook (1
  title-markdown + 6 plan-§5 cells). Generated programmatically from
  `scripts/_build_notebook_14.py`. Cell 3 explicitly sets
  `os.environ["BRISTOL_ML_LLM_STUB"]="1"` so the notebook is
  deterministic regardless of YAML defaults. Cell 4 prints one of
  three banners explaining whether a live run is ready (cassette
  present / API key set / stub override active). Cell 5 is the demo
  moment: `evaluate(cfg.llm)` + `format_report(...)` inline plus a
  pandas-frame disagreement view.
- Tests:
  - `tests/unit/llm/test_extractor.py` — Protocol shape (AC-1),
    `RemitEvent` / `ExtractionResult` validation (AC-5: bad types,
    naive datetimes, confidence range), stub gold-set hits + miss
    defaults, factory dispatch (env-var override + `None` config),
    live-path init guards (missing API key error names both env-vars;
    type/model_name/prompt_file required when type=openai),
    standalone CLI exit-0 (NFR-8), prompt-hash differs across files
    (NFR-5), malformed-LLM-response logs WARNING + returns default
    (NFR-6), grep-style guard that no model name appears literally
    in the source (AC-3), Hydra-config round-trip (T1), AppConfig
    `llm` default `None` keeps existing callers unaffected (T1).
  - `tests/unit/llm/test_evaluate.py` — 10 tests covering
    `test_evaluate_harness_runs_against_stub` (AC-4),
    `test_evaluate_harness_output_is_deterministic_modulo_timestamp`
    (NFR-3), `test_evaluate_harness_records_provenance` (AC-4),
    capacity tolerance (`_DriftExtractor` injection),
    datetime tolerance (`_TimeDriftExtractor`), disagreement listing
    shape, `--max-disagreements` truncation,
    `test_evaluate_module_runs_standalone_help` (NFR-8), CLI runs
    against the stub via `argv` (NFR-8), and `loguru` summary-record
    emission (Ctrl+G OQ-D).
  - `tests/integration/llm/test_llm_extractor_cassette.py` — AC-3 +
    AC-5 against the cassette (cardinality, `ExtractionResult` type,
    `prompt_hash` 12-char, `model_id == config.model_name`, confidence
    in `[0, 1]`). Skipped when cassette absent.
  - `tests/integration/test_notebook_14.py` —
    `test_notebook_14_llm_extractor_executes_top_to_bottom`
    (`nbconvert --execute` round-trip under
    `BRISTOL_ML_LLM_STUB=1`, asserts T5 Cell 1 / 3 / 5 each produce
    output) and `test_notebook_14_has_expected_cell_count[7]`.
- `docs/architecture/layers/llm.md` — new layer doc capturing the
  Protocol contract, the env-var triple-gate, the prompt-hash
  provenance convention, the strict-mode JSON-schema gotcha, the
  `message_description` NULL strategy, the gold-set fixture format,
  the cassette-refresh ritual, and the side-by-side harness output.
  Status: Provisional (first realised by Stage 14).
- `docs/architecture/README.md` — layer index extended with the LLM
  row.
- `src/bristol_ml/llm/CLAUDE.md` — module guide: schema reference
  table for `RemitEvent` and `ExtractionResult`, downstream-consumer
  warning on `confidence` (sentinel, not calibrated probability),
  cassette-refresh ritual + leak-check command, gold-set curation
  guidance (mRID prefix conventions, when to bump the size).
- `README.md` — new "Worked example: LLM feature extractor (Stage 14)"
  section linking the layer doc, the module guide, and this retro.
  The existing "API keys and offline-by-default" section already
  documents the `BRISTOL_ML_LLM_API_KEY` setup process (added at
  Phase 1 follow-up per Ctrl+G OQ-D).

## Design choices made here

- **Two-method `Extractor` Protocol; nothing else public.** AC-1
  caps the surface — *"the interface is small enough that writing a
  third implementation in the future is plausible."* Adding a third
  method is a Stage 15/16 contract change. The two-method shape
  leaves room for batch optimisations later (the live extractor
  ships sequentially today; the Protocol is shaped so a parallel
  implementation slots in without changing callers).
- **`runtime_checkable` Protocol over `abc.ABC`.** ADR-0003
  precedent for swappable interfaces. Lets unit tests assert
  structural conformance via `isinstance(_, Extractor)` without
  forcing inheritance on the concrete classes — a future
  open-weights extractor doesn't need to inherit from anything in
  this layer, only match the two-method shape.
- **Triple-gated live path.** Plan §1 D4. `LlmExtractorConfig.type`
  literal *and* `BRISTOL_ML_LLM_STUB` env var *and* the API-key
  presence check at `LlmExtractor.__init__` — three independent
  guards before a live call fires. The env var is the load-bearing
  one for CI safety: a YAML edit that flips `type: openai` cannot
  produce a live call when the env var is set, which CI does
  unconditionally.
- **Hand-authored OpenAI strict schema, not
  `model_json_schema()`.** OpenAI strict mode requires
  `additionalProperties: false` on every object *and* every
  property in `required` (no implicit-optional via `default`).
  Pydantic emits `default: null` on `Optional[...]` fields and
  omits them from `required` — both rejected by the API at first
  call. The hand-authored `_OPENAI_RESPONSE_SCHEMA` covers exactly
  the LLM-populated subset (provenance is stamped by the extractor
  *after* the LLM returns).
- **Synthesise on NULL `message_description`, no
  `GET /remit/{messageId}` hydration.** Plan §1 D6 / Ctrl+G OQ-B.
  The Stage 13 stream endpoint frequently omits
  `message_description`; hydration would mean ~45,000 extra calls
  per archive run. The structured fields are themselves the
  extraction signal — the LLM is being asked to *interpret +
  standardise* them. The fallback path mirrors structural fields
  from the input where present and stamps `confidence=0.0`. M-NA..M-ND
  in the gold set are deliberate probes of this path.
- **Stub is the default in YAML *and* the env-var override path.**
  Plan §1 D4. `conf/llm/extractor.yaml` ships `type: stub`, *and*
  `build_extractor` honours `BRISTOL_ML_LLM_STUB=1` regardless of
  config. The redundancy is the point: a YAML override that flips
  `type: openai` accidentally still runs the stub when the env var
  is set, which is the CI invariant.
- **Provenance via prompt-bytes SHA-256 prefix, not embedded
  version field.** Plan §1 D7 / NFR-5. A version field would add a
  parsing contract on top of "the prompt is plain text"; a hash is
  content-derived identity that survives edits *and* renames. 12
  hex chars is the readability/collision trade-off — collisions
  over the project lifetime are vanishing (1 in 281 trillion).
- **Stub uses `(mrid, revision_number)` as the lookup key — same
  primary key as Stage 13's parquet.** Plan §1 D9. Every distinct
  revision can carry its own labelled extraction; the lookup
  semantics match the bi-temporal storage so a future "load REMIT,
  extract every row" flow has a uniform shape.
- **Two-metric design (exact + tolerance) printed side-by-side.**
  Plan §1 D11 / D12. Intent line 41: *"different choices produce
  different numbers, which makes the metric choice a lesson
  itself."* The same data produces different numbers under
  different metric choices — that's the demo lesson. Categorical
  fields (`event_type`, `fuel_type`) have exact == tolerance (no
  notion of "close enough" for strings); the tolerance column is
  kept for layout uniformity.
- **Stdout-primary harness output + structured `loguru` summary,
  no markdown file.** Plan §1 D15 / Ctrl+G OQ-D. Stdout is simpler
  to test for byte-determinism (capture + redact timestamps); the
  notebook displays the same table inline; the loguru summary
  means a non-notebook CLI run leaves a captured trace in the
  standard log stream rather than disappearing. No file write.
- **No `max_events_per_run` config field; no `--sample N` CLI
  flag.** Plan §1 D13 — the Phase-1 single highest-leverage cut.
  Researcher R7 confirmed full-archive runs at GPT-4o-mini batch
  rates cost ~£12.50; the cost guardrail operationalises a problem
  the project does not yet have. The stub-default discipline is
  the only structural cost guard the project needs at this stage.
- **No formal ADR for the LLM extractor interface or the env-var
  pattern.** The Protocol is captured in this layer doc + the
  Stage 14 plan; the env-var pattern is `BRISTOL_ML_LLM_API_KEY`
  parallel to `BRISTOL_ML_REMIT_STUB`. Promote either to an ADR
  only if a future stage finds the choice contested.

## Demo moment

From a clean clone (Stages 0–13 already built):

```bash
uv sync --group dev
uv run pytest -q                                                   # all green
uv run python -m bristol_ml.llm.extractor --help                   # offline; prints schema
uv run python -m bristol_ml.llm.evaluate                           # offline harness run

# Live path (spends tokens):
export BRISTOL_ML_LLM_API_KEY=sk-...
uv run python -m bristol_ml.llm.evaluate llm.type=openai           # live harness run

# Notebook (offline, ~10 s wall-clock under stub):
BRISTOL_ML_LLM_STUB=1 uv run jupyter nbconvert --to notebook \
    --execute notebooks/14_llm_extractor.ipynb \
    --output /tmp/14_test_run.ipynb
```

The notebook's Cell 5 side-by-side table is the demo moment: per-field
exact-match next to tolerance-match against the 76-record gold set,
plus the disagreement listing on a live run. A facilitator can pause
at the per-field columns and ask: "the LLM gets `affected_capacity_mw`
exactly right 60 times out of 76, and within ±5 MW 70 times — which
of those is the right number for Stage 16's join?" — that's the
metric-choice lesson.

## Observations from execution

- **Cassette not recorded in dev environment.** No OpenAI API key
  available in the worktree at Stage 14 build-up. The integration
  test honours `--record-mode=none` and skips when the cassette is
  absent — same shape as `test_remit_cassettes`. The cassette
  refresh is documented in `src/bristol_ml/llm/CLAUDE.md` as a
  manual ritual; recording is a one-line invocation
  (`BRISTOL_ML_LLM_API_KEY=sk-... uv run pytest
  tests/integration/llm/test_llm_extractor_cassette.py
  --record-mode=once`) and can be performed by anyone with an
  OpenAI key. The remaining live-path coverage is the unit-level
  init-guard tests (mocked `OpenAI` client + monkey-patched
  `extract` failure modes).
- **Final gold-set count: 76 records.** Plan §1 D8 / Ctrl+G OQ-C
  named 80 as the target; landed at 76 to keep the curation effort
  proportional. Inside the intent's "dozens to low hundreds" range.
  Distribution: Outage 51, Restriction 25; Gas 18, Nuclear 13,
  Wind 13, Solar 8, Coal 6, Hydro 5, Pumped Storage 4, Biomass 3,
  Oil 2, Interconnector 2, Battery 2. The five least-common fuel
  types (Coal, Pumped Storage, Biomass, Oil, Interconnector,
  Battery, Hydro) are intentionally thin to keep curation cost low
  — a Stage 16 feature-join failure on one of them is the trigger
  to bump the count.
- **One deliberate deviation: harness CLI flags renamed.** Plan
  §6 T5 step 3 named `--redact-timestamps` (default off) for the
  byte-equality test; the shipped CLI also adds
  `--max-disagreements N` (default 10) for output legibility on
  the live path where disagreements may exceed 10. The latter is
  not in the plan but is a one-liner that materially helps the
  demo when more than 10 disagreements appear; the test suite
  exercises it via `test_max_disagreements_truncates_and_appends_count`.
- **OpenAI SDK lazy import.** `LlmExtractor.__init__` defers
  `from openai import OpenAI` until after the API-key check so a
  test environment without `openai` installed (the stub-only path)
  still imports `bristol_ml.llm.extractor` without error. Mirrors
  the Stage 12 serving-layer FastAPI lazy import. This is a
  hand-rolled discipline; not enforced by a structural test today
  because the OpenAI SDK is a runtime dependency in this project.

## Deferred

- **VCR cassette recording.** Build-up phase decision; documented
  ritual lives in the module guide. The first live run by an
  operator with an API key produces the cassette and unblocks the
  CI replay test.
- **`message_description` hydration via `GET /remit/{messageId}`.**
  Stage 16 owns this if the feature-join needs richer text. The
  ~45,000-call cost of full-archive hydration is not justified by
  Stage 14's demo scope.
- **Calibrated confidence scores.** OpenAI strict mode does not
  return per-token logprobs by default; `ExtractionResult.confidence`
  is a sentinel (1.0 grounded / 0.0 fallback) at this stage. A
  future stage that needs calibrated probabilities for downstream
  weighting can swap to a logprobs-aware model and post-process —
  the schema field is already there.
- **An open-weights / locally-served backend.** Plan §1 D3 leaves
  the `Literal["stub", "openai", ...]` slot extensible. Stage 14
  ships only `stub` and `openai`; a third backend is a one-decision
  add-on (extend the literal, add a dispatch branch, write the
  class) when there's a concrete reason to want one.
- **Prompt engineering as an ongoing activity.** Intent §"Out of
  scope". The shipped `extract_v1.txt` is the v1 prompt; refining
  it is a Stage 16 / Stage 17 conversation if observed
  disagreement rates demand it.
- **Fine-tuning, distillation, ensembles, streaming.** Intent §"Out
  of scope, explicitly deferred".
- **Cost guardrail (`max_events_per_run` + `--sample N`).** Plan §1
  D13 — single highest-leverage cut. The stub-default discipline
  is the only structural cost guard the project needs.

## Next

→ Stage 15: embedding index over the same REMIT corpus — parallel
thread on the same data, no dependency on the extractor output.

→ Stage 16: feature-table join — both Stage 14 (extracted features)
and Stage 15 (embeddings) flow into the modelling feature table,
and the `as_of` mechanic from Stage 13 is what guarantees the join
uses only information available at training time, no leakage.
