# Stage 14 — LLM feature extractor: scope diff

**Artefact role:** Phase 1 research deliverable (`@minimalist` pre-synthesis critic).
**Audience:** plan author (lead), Ctrl+G reviewer at human plan review.
**Method:** read `docs/intent/14-llm-extractor.md`, the three Phase-1 research artefacts, and the lead's draft decision set; tag every decision / NFR / test / dependency / notebook cell as one of:

- `RESTATES INTENT` — directly implements an explicit AC, scope bullet, or "Points for consideration" line.
- `PLAN POLISH` — operational detail the plan needs but with more rigid specificity than warranted.
- `PREMATURE OPTIMISATION` — addresses a problem the project does not yet have.
- `HOUSEKEEPING` — repo hygiene / convention upkeep, not feature work.

Default disposition on `PREMATURE OPTIMISATION` is **cut**. `PLAN POLISH` rows should be reconsidered before binding into the plan.

---

## Decisions (lead's draft set)

| # | Decision | Tag | Reason |
|---|----------|-----|--------|
| D1 | Module location: `src/bristol_ml/llm/extractor.py` | RESTATES INTENT | Mirrors §"Module boundaries" in `CLAUDE.md`. |
| D2 | Public interface: `Extractor` `Protocol` with `extract(event)` and `extract_batch(events)` | RESTATES INTENT | AC-1 (small, implementation-agnostic interface) + ADR-0003 precedent. |
| D3 | Two implementations: `StubExtractor`, `LlmExtractor`; discriminated union on `type` | RESTATES INTENT | Intent §Scope; codebase pattern at `conf/_schemas.py:931–941`. |
| D4 | Stub default via env var `BRISTOL_ML_LLM_STUB=1` | RESTATES INTENT | AC-2; matches `BRISTOL_ML_REMIT_STUB` precedent. |
| D5 | API-key env var `BRISTOL_ML_LLM_API_KEY` | RESTATES INTENT | AC-3 mandates env-var; codebase has no precedent — Stage 14 sets it. |
| D6 | Provider default: Anthropic Claude Haiku 4.5 with `output_config.format` | RESTATES INTENT | Intent §Points line 43; researcher R1.1 confirms GA, no beta header. |
| D7 | Prompt file location: `conf/llm/prompts/extract_v1.txt` | PLAN POLISH | Reasonable default; `conf/prompts/` (researcher R5) is equally defensible — the directory choice is not load-bearing. |
| D8 | Hand-labelled set size: exactly 100 records | PLAN POLISH | Intent says "dozens to low hundreds"; binding to 100 is more rigid than the intent. Express as a range with a target. |
| D9 | Hand-labelled set location: `tests/fixtures/llm/hand_labelled.json` | RESTATES INTENT | OQ-2 default disposition; intent line 46 ("in-repo for a small set"). |
| D10 | Output parquet carries `prompt_hash`, `model_id`, `extracted_at_utc` columns | PLAN POLISH | NFR-5 / intent line 45 require provenance; the specific column names are a plan-stage choice but not load-bearing — leave to implementer. |
| D11 | Two-metric agreement: exact match + tolerance/F1 | RESTATES INTENT | Intent §Points line 41; researcher R4 endorses the dual-metric design as the demo lesson. |
| D12 | Specific tolerances: ±5 MW capacity, ISO-8601 exact time | PLAN POLISH | Researcher R4 suggests ±5 MW or ±10 % capacity, ±1 h time. The plan can name representative thresholds without binding the implementer. |
| D13 | Cost guardrail: `max_events_per_run` config field (default 20) + `--sample N` CLI flag on eval harness | **PREMATURE OPTIMISATION** | Researcher R7: full one-year archive at Haiku 4.5 batch rates is ~£16. The project is pedagogical, not a production data pipeline. The stub-default (D4) already prevents accidental notebook/CI cost. Adding a guardrail config field operationalises a problem we do not have. **Single highest-leverage cut.** |
| D14 | Pydantic schema with `event_type`, `fuel_type`, `affected_capacity_mw`, `effective_from`, `effective_to`, `confidence` | RESTATES INTENT | AC-5 enumerates these fields; intent §Scope line 14. |
| D15 | Eval report destination: timestamped markdown file under `docs/lld/llm/` + stdout summary | PLAN POLISH | OQ-7 default; AC-4 requires reproducibility but does not require a written file. Stdout-only is simpler, and the reproducibility test is easier to write against stdout. |
| D16 | Graceful degradation: log + return documented default on LLM parse failure | RESTATES INTENT | Intent line 47; NFR-6. |
| D17 | Out-of-scope explicit list: prompt engineering, fine-tuning, ensembles, streaming | PLAN POLISH | Intent §"Out of scope, explicitly deferred" already names these. Restating in the plan defends a decision never under threat. |
| D18 | VCR cassette for real LLM integration test | RESTATES INTENT | R-1 mitigation; established pattern at `tests/integration/ingestion/test_remit_cassettes.py`. |
| D19 | Notebook `notebooks/14_llm_extractor.ipynb` running stub end-to-end | RESTATES INTENT | Intent §"Demo moment" line 28; established convention. |
| D20 | Plan negative-list: "we will NOT add `instructor` or `langchain` as dependencies" | PLAN POLISH | Researcher R2 explicitly recommends against these; restating in the plan defends a decision not under threat. |

## Non-functional requirements

| NFR | Tag | Reason |
|-----|-----|--------|
| NFR-1 — Offline by default | RESTATES INTENT | AC-2; §2.1.3. |
| NFR-2 — Typed boundary | RESTATES INTENT | AC-5; §2.1.2. |
| NFR-3 — Eval reproducibility | RESTATES INTENT | AC-4. |
| NFR-4 — Cost control (`max_events_per_run`, `--sample N`) | **PREMATURE OPTIMISATION** | Binds D13. Cut for the same reason. |
| NFR-5 — Prompt versioning via file + hash | RESTATES INTENT | Intent line 45; §2.1.6. |
| NFR-6 — Graceful degradation | RESTATES INTENT | Intent line 47. |
| NFR-7 — YAML config | RESTATES INTENT | §2.1.4. |
| NFR-8 — Module standalone | RESTATES INTENT | §2.1.1. |
| NFR-9 — Loguru observability levels (INFO / WARNING / DEBUG / ERROR) | PLAN POLISH | The convention is established repo-wide; restating the level mapping for one more module is plan polish. Keep the test for "log on parse failure" (NFR-6 covers this); drop the rigid level-mapping spec. |

## Tests (lead's draft list)

| Test | Tag | Reason |
|------|-----|--------|
| Stub returns fixed features for hand-labelled events | RESTATES INTENT | AC-2. |
| No network call when stub active (monkeypatch httpx) | RESTATES INTENT | NFR-1. |
| Real implementation against VCR cassette | RESTATES INTENT | R-1 mitigation; AC-3. |
| Missing API key + live config → clear error at init | RESTATES INTENT | AC-3 sub-criterion. |
| Malformed LLM response → log + return default | RESTATES INTENT | NFR-6. |
| Pydantic ValidationError at boundary on bad input | RESTATES INTENT | AC-5. |
| Eval harness produces reproducible (byte-identical mod timestamps) report | PLAN POLISH | AC-4 requires this; the test design depends on D15. If the report is stdout-only with timestamps redacted, the test is simpler than file-byte-comparison. |
| Eval harness `python -m bristol_ml.llm.evaluate` CLI runs | RESTATES INTENT | AC-4 sub-criterion; §2.1.1. |
| Notebook executes top-to-bottom | RESTATES INTENT | Stage convention. |

## Notebook cells (draft)

| Cell | Tag | Reason |
|------|-----|--------|
| Bootstrap (chdir + sys.path) | RESTATES INTENT | Established convention. |
| Load config | RESTATES INTENT | Established convention. |
| Run stub on hand-labelled events | RESTATES INTENT | Demo moment. |
| Run eval harness | RESTATES INTENT | Demo moment (intent line 28). |
| Show side-by-side report | RESTATES INTENT | Demo moment. |
| Discussion of metric choice | RESTATES INTENT | Intent §Points line 41. |
| Closing wrap-up md cell | PLAN POLISH | Adds prose without information; the side-by-side report is the punch line. |

## Dependencies

| Dep | Tag | Reason |
|-----|-----|--------|
| `anthropic` SDK | RESTATES INTENT | AC-3 + D6 imply an LLM client; researcher R1.1 endorses direct SDK. |
| `vcrpy` (existing dev dep) | RESTATES INTENT | Established cassette convention; no new dep. |

## Single highest-leverage cut

**Cut D13 (the `max_events_per_run` config field and `--sample N` evaluation-harness flag), and the NFR-4 it binds.** The intent's cost concern is addressed structurally by D4 (stub-by-default in CI, notebooks, and any path without an API key). The full one-year REMIT archive costs ~£16 at Haiku 4.5 batch rates per researcher R7 — the only context where cost matters is a deliberate full-archive backfill, which a developer running `python -m bristol_ml.llm.evaluate --full-archive` can self-guard with awareness rather than a config field. Adding `max_events_per_run` to the YAML schema and `--sample N` to the CLI operationalises a problem the project does not yet have, ahead of any evidence that it does.

---

## Reconsiderations (lead's binding decisions before writing the plan)

The lead has reconsidered every `PLAN POLISH` and `PREMATURE OPTIMISATION` row. Final dispositions:

| # | Disposition | Rationale |
|---|-------------|-----------|
| D7 | **Keep** as-is | Low-cost specificity; the directory choice has zero downstream blast radius. |
| D8 | **Soften** to "target ~80 records, range 50–120" | Honours intent line 40; allows implementer to right-size during curation. |
| D10 | **Keep** the requirement, drop the column-name list | The implementer chooses column names; the plan binds only "prompt provenance is recorded in extraction output". |
| D12 | **Soften** to "tolerances are documented choices, not magic numbers" | Plan names representative thresholds (±5 MW; ±1 h) but the implementer may revise with test evidence. |
| D13 | **Cut** | Single highest-leverage cut. |
| D15 | **Cut the file**, keep stdout | Simpler reproducibility test; AC-4 satisfied by deterministic stdout (timestamps redacted in the test fixture). |
| D17 | **Cut** | Restates the intent's own §"Out of scope" verbatim. |
| D20 | **Cut** | Defending a decision (no `instructor` / `langchain`) that is not under threat. |
| NFR-4 | **Cut** | Binds D13. |
| NFR-9 | **Soften** to "use `loguru` consistent with other modules" | Drops the per-level specification; the existing test for parse-failure logging covers the load-bearing case. |
| Eval reproducibility test | **Keep, simplified** | Tests stdout determinism rather than file-byte equality. |
| Closing wrap-up cell | **Cut** | Side-by-side report is the demo punch line. |

These reconsiderations are reflected in the plan at `docs/plans/active/14-llm-extractor.md`.
