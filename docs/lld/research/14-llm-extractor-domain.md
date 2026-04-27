# Stage 14 — LLM feature extractor: domain research

**Date:** 2026-04-26
**Target plan:** `docs/plans/active/14-llm-extractor.md` (not yet created)
**Intent source:** `docs/intent/14-llm-extractor.md`
**Baseline SHA:** main @ `7f6b511` (Stage 13 merged; Stage 14 not yet started)

**Scope:** External technical context for the lead to bind decisions on. Citations-grounded. Not a design document — findings only. British English throughout. Numbered subsections (R1–R7) so the plan can cite by reference.

---

## Upstream context: `message_description` in Stage 13

Stage 13's `OUTPUT_SCHEMA` carries a `message_description` column (pyarrow `string`, nullable) mapped from the Elexon JSON field `messageDescription`. The Stage 13 implementation note in `_parse_message` reads:

> "The stream endpoint does not return a long-form message description today (per the live response observed 2026-04 in domain research §R6); kept on the schema so Stage 14 can populate it from a follow-up `/remit/{mrid}` call without a schema migration."

The stub fixture populates `message_description` with short English sentences. In a live context the field will frequently be `None`. The `relatedInformation` content (the Stage 13 domain research's intended Stage 14 input) may only be available via `GET /remit/{messageId}` from the opinionated endpoint family.

**Sharp edge for Stage 14:** The extractor must either (a) accept null `message_description` values and skip/default them, (b) treat the combination of structured fields (`cause`, `event_type`, `fuel_type`, `affected_mw`) as the extraction context rather than free text, or (c) issue follow-up hydration calls. Option (c) adds ~45,000 additional HTTP calls for a full archive run. This is a scope decision for the plan, not a decision this artefact makes — but it is the highest-priority external constraint because it determines the architecture of the live path.

---

## R1 — Structured-output APIs across major providers

### R1.1 Anthropic Claude

**Mechanism:** Two complementary routes exist as of 2026-04:

1. **`output_config.format`** (`{"type": "json_schema", "schema": {...}}`) — primary extraction method. Passed at the top level of the Messages API request. Uses constrained decoding during generation: output is guaranteed to conform to the schema before the API returns it. No post-processing or retries required for schema violations. No beta header required — this is generally available.

2. **`strict: true` on tool definitions** — enforces `input_schema` during generation via the same mechanism. Older, longer-tested in production than `output_config`.

**Supported JSON Schema keywords (`output_config` route, confirmed from Anthropic docs):**
Basic types (`object`, `array`, `string`, `integer`, `number`, `boolean`, `null`); `enum` (primitives only); `const`; `anyOf`; `allOf`; `$ref`, `$def`, `definitions`; `required`; `additionalProperties: false`; string `format` values (`date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid`); array `minItems` (0 or 1 only); `default`.

**Explicitly NOT supported:** Recursive schemas; numerical constraints (`minimum`, `maximum`, `multipleOf`); string constraints (`minLength`, `maxLength`); array constraints beyond `minItems 0/1`; `additionalProperties` set to anything other than `false`; complex types in enums; external `$ref`.

**Validation timing:** During generation (constrained decoding). Not post-hoc. Output is structurally guaranteed before the API call returns.

**Models (2026-04-26):** Claude Haiku 4.5, Sonnet 4.5, Sonnet 4.6, Opus 4.5, Opus 4.6, Opus 4.7. Available on Claude API and AWS Bedrock. Not available on Google Vertex AI for Opus 4.7/Mythos Preview.

Note: Claude Sonnet 3.7 is deprecated per the Anthropic pricing page (2026-04-26); do not target it in new code.

**Pricing (confirmed from Anthropic pricing page, 2026-04-26):**

| Model | Input $/MTok | Output $/MTok | Batch Input $/MTok | Batch Output $/MTok |
|---|---|---|---|---|
| Claude Haiku 3.5 | $0.80 | $4.00 | $0.40 | $2.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 | $0.50 | $2.50 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | $1.50 | $7.50 |

Prompt caching: cache writes at 1.25× input cost; cache hits at 0.10× input cost (90% saving on re-used context).

**Community reliability note:** The `output_config` feature is newer than OpenAI's structured output (which launched August 2024). Third-party benchmarks specific to the Claude `output_config` route are sparse as of April 2026. The older `tools`-based extraction approach has more community validation.

**Sources:**
- [Anthropic Structured Outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) — `output_config` mechanism, supported keywords, constrained decoding guarantee
- [Anthropic Pricing page](https://platform.claude.com/docs/en/about-claude/pricing) — confirmed pricing table
- [Tessl.io — Anthropic Structured Outputs announcement](https://tessl.io/blog/anthropic-brings-structured-outputs-to-claude-developer-platform-making-api-responses-more-reliable/) — GA launch coverage; confirms `output_config` API shape

---

### R1.2 OpenAI

**Mechanism:** `response_format={"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": {...}}}` on Chat Completions. Uses a Context-Free Grammar (CFG) engine to mask invalid tokens at generation time. The model cannot generate non-conforming output. First request with a new schema incurs added latency (grammar compilation); subsequent requests do not.

**Unsupported keywords in strict mode (from OpenAI community and docs):**
- `oneOf` — explicitly unsupported; community reports errors (thread #966047)
- Numerical constraints (`minimum`, `maximum`, `multipleOf`)
- String constraints (`minLength`, `maxLength`)
- All fields must be in `required`; optional fields expressed as `type: ["string", "null"]`
- Recursive schemas beyond ~5–10 levels
- `allOf` at some nesting levels (community reports API rejections)
- `patternProperties`

**Supported:** `type`, `properties`, `required`, `items`, `enum`, `additionalProperties: false`, `description`, `$ref`/`$defs`, array `minItems`/`maxItems`.

**JSON mode (`type: "json_object"`)** is now considered legacy — it guarantees only valid JSON syntax, not schema conformance. Strict mode is the production default for extraction.

**Models for extraction (2026-04):** GPT-4o-mini at $0.15/$0.60 per MTok (input/output) is the budget choice and supports structured outputs. GPT-4o at $2.50/$10.00.

**Community reliability:** Strict mode is the most field-tested constrained generation approach across providers (first to market, August 2024). Simmering.dev describes it as offering a structural guarantee that instructor/function-calling cannot match. Edge cases with `anyOf`/`oneOf` and complex nested schemas are documented in community forums.

**Sources:**
- [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — strict mode, CFG token masking, schema subset
- [OpenAI Introducing Structured Outputs blog](https://openai.com/index/introducing-structured-outputs-in-the-api/) — August 2024 GA; CFG mechanism
- [Simmering.dev — structured output vs instructor](https://simmering.dev/blog/openai_structured_output/) — constrained generation vs. learned behaviour
- [OpenAI community — oneOf/allOf in strict mode](https://community.openai.com/t/oneof-allof-usage-has-problems-with-strict-mode/966047) — concrete unsupported-feature reports
- [OpenAI Pricing](https://openai.com/api/pricing/) — GPT-4o-mini $0.15/$0.60 per MTok

---

### R1.3 Google Gemini

**Mechanism:** Set `responseMimeType: "application/json"` and provide a `responseSchema` in `GenerationConfig`. Same constrained-decoding principle as Anthropic and OpenAI. The documentation states the API "guarantees syntactically correct JSON" but explicitly notes it "does not guarantee the values are semantically correct."

**Supported keywords:** Basic types (`string`, `number`, `integer`, `boolean`, `object`, `array`, `null`); `properties`; `required`; `additionalProperties`; `enum`; `format` (`date-time`, `date`, `time`); `minimum`, `maximum`; `items`, `prefixItems`, `minItems`, `maxItems`; `title`, `description`.

**Limitations:** Not all JSON Schema features supported; "the model ignores unsupported properties" rather than erroring. The API may reject very large or deeply nested schemas. The silent-ignore behaviour is a reliability risk for schemas that use unsupported keywords — the schema appears to work but the constraint is not enforced.

**Current models (2026-04):** Gemini 2.5 Pro/Flash/Flash-Lite and Gemini 3.1 Pro Preview, Gemini 3 Flash Preview. Gemini 2.0 Flash is deprecated and shuts down 1 June 2026.

**Pricing:** Gemini 2.5 Flash at $0.30/$2.50 per MTok; Gemini 3 Flash Preview at $0.50/$3.00; Gemini 2.5 Flash-Lite at $0.10/$0.40.

**Community reliability:** An independent provider comparison (Glukhov, Medium 2024) found Gemini more likely to silently ignore unsupported schema properties than to raise an error, which is harder to detect than an API rejection. Gemini 3-series models appear to have improved but independent benchmarks are limited as of April 2026.

**Sources:**
- [Gemini Structured Output docs](https://ai.google.dev/gemini-api/docs/structured-output) — `responseSchema`, supported types, semantic caveat
- [Google Blog — Gemini Structured Outputs improvements, April 2026](https://blog.google/technology/developers/gemini-api-structured-outputs/) — April 2026 GA enhancements
- [Medium — Structured output comparison across providers (Glukhov)](https://medium.com/@rosgluk/structured-output-comparison-across-popular-llm-providers-openai-gemini-anthropic-mistral-and-1a5d42fa612a) — Gemini silent-ignore behaviour report

---

### R1.4 Open-weights option

**Current leading candidates (2026-04):**

**Qwen3 (Alibaba, Apache 2.0):** Current generation (mid-2025), dense models from 0.6 B to 32 B plus MoE variants (Qwen3-235B-A22B at 22 B active parameters, Qwen3-30B-A3B at 3 B active). vLLM supports Qwen3 tool calling via `--tool-call-parser qwen3_xml`. Active bug reports on malformed JSON in early Qwen3 tool-call implementations (vLLM issue #21711); a `Qwen35CoderToolParser` was introduced to address streaming extraction issues. Qwen3-8B is the most widely cited sub-10 B model for structured extraction as of early 2026. Apache 2.0 licence allows commercial use and redistribution.

**Llama 3.3-70B (Meta, Llama licence):** Most capable dense open model in the 70 B class for function calling; vLLM supports it with `--tool-call-parser llama3_json`. Commercial use requires Meta's Llama licence (not Apache 2.0).

**Integration with the existing `_common.py` httpx client:** Both vLLM and llama-cpp-python expose an OpenAI-compatible `/v1/chat/completions` endpoint. The `instructor` library supports an `openai`-compatible mode. A self-hosted model can slot into an `httpx`-based client with only the `base_url` changed — no new HTTP client layer is needed.

**Practical constraint for Stage 14:** Self-hosting requires GPU infrastructure and a warm-start server process. For a project prioritising offline-CI and pedagogical simplicity, a cloud API is the lower-friction default; the open-weights path is an attendee option for those who bring a local GPU server.

**Sources:**
- [Qwen3 blog post](https://qwenlm.github.io/blog/qwen3/) — model family, Apache 2.0 licence
- [vLLM Tool Calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/) — qwen3_xml parser; JSON structured output support
- [vLLM Qwen guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) — official Qwen+vLLM integration
- [vLLM issue #21711 — Qwen3 tool call bug](https://github.com/vllm-project/vllm/issues/21711) — active community reports of malformed JSON

---

## R2 — JSON-schema vs. free-form-then-parse

**Two approaches:**

**Constrained decoding (structured output):** Schema given to the API before generation; token choices masked by a grammar engine. Structural guarantee — the output cannot violate the schema. All three major providers now implement this. The guarantee is structural not semantic.

**Free-form-then-parse with `instructor`:** Prompt describes the desired JSON shape; model generates freely; client parses and, on failure, re-prompts with the validation error. The `instructor` library formalises this with Pydantic validation + automatic retry. Informally quantified as "~10% of raw JSON mode calls fail" in the instructor docs. Independent benchmarks do not provide a precise figure, but the figure is consistent with practitioner reports.

**Quality comparison:** The Simmering.dev comparison establishes the mechanistic distinction: constrained generation offers a structural guarantee that retry-based approaches cannot match. However, for simple, flat schemas (4–6 fields, no deep nesting), well-prompted models rarely fail. The StructEval benchmark (ArXiv 2505.20139, 2025) shows GPT-4o at 76% on its complex multi-format benchmark — but Stage 14's extraction schema is far simpler than that benchmark's scope, so real-world accuracy will be substantially higher.

**Provider authoritative disagreement:** The Anthropic docs claim "always valid" for `output_config`; the Gemini docs claim "syntactically correct but not semantically correct." Both are accurate about their respective features but describe the same fundamental limitation: schema conformance does not guarantee that extracted field values are correct — a model can return a structurally valid JSON with a wrong capacity value.

**`instructor` library:** The most widely used Python abstraction for structured LLM output (3 M+ monthly downloads, 11 k GitHub stars, April 2026). Supports Anthropic, OpenAI, Gemini, and OpenAI-compatible local models via a unified interface. Adds retry logic on top of provider structured output where available; falls back to free-form retry where not. For a single-provider project using Anthropic `output_config`, `instructor` adds value primarily as a multi-provider portability layer and as a graceful degradation wrapper for the `message_description = null` case.

**LangChain `.with_structured_output()`:** Abstracts over provider APIs (routes to `output_config`, `response_format`, or `responseSchema` depending on the bound model). Adds significant dependency weight. More useful if Stage 14 needs to target multiple providers.

**Researcher view:** For Stage 14, use Anthropic's `output_config` directly (or the `tools` route) rather than adding `instructor` or LangChain as a dependency. The added value of `instructor` is retry logic on validation failure — which constrained decoding makes unnecessary. If a future stage targets a self-hosted model, `instructor`'s OpenAI-compatible mode is the cleanest addition point at that time.

**Sources:**
- [instructor library](https://python.useinstructor.com/) — primary docs; validation + retry approach
- [instructor — Why use Instructor?](https://python.useinstructor.com/why/) — "~10% failure rate" for raw JSON mode; retry-on-validation approach
- [Simmering.dev — structured output vs instructor](https://simmering.dev/blog/openai_structured_output/) — mechanistic comparison
- [ACM IPM — Are LLMs good at structured outputs?](https://dl.acm.org/doi/10.1016/j.ipm.2024.103809) — benchmark; GPT-4o 76% on complex schemas
- [cleanlab.ai — Structured Output Benchmarks](https://cleanlab.ai/blog/structured-output-benchmark/) — benchmark ground-truth quality issues
- [agenta.ai — Guide to structured outputs](https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms) — practitioner comparison of approaches

---

## R3 — Hand-labelled gold-set sizing for IE evaluation

**Statistical basis:** For an 80% pass rate and a 5% margin of error at 95% confidence, the binomial proportion formula gives approximately 246 samples (Maxim AI, 2025). For a 10% margin of error at the same confidence: ~97 samples. These are per-field estimates; a set that covers the field with the lowest expected accuracy sets the binding constraint.

**IE / NER literature:** Academic NER gold sets are typically hundreds to thousands of annotated instances. However, academic sets serve both training and evaluation; for a pure evaluation set on a narrow, well-defined extraction task, a smaller size is defensible. Invoice extraction practitioners recommend "canary sets of 50–200 examples" for per-field benchmarking and drift detection (ArXiv 2510.15727).

**Ontotext principle:** A small high-quality set is preferable to a large low-quality set. Verification requires at least two independent raters to avoid annotator bias; inter-annotator agreement must be computed to validate that the task is well-defined.

**Practical sizing for Stage 14:** 100 records is a reasonable minimum — it offers ±10% margin of error at 95% confidence at 80% accuracy and is achievable as a curation exercise. The set must cover:
- All 12 Elexon fuel types (Nuclear, Gas, Coal, Wind, Hydro, etc.)
- Both planned and unplanned events
- Records with `message_description = null` (to exercise the default-fallback path)
- The informal schedule-encoding format (`~/E_DERBY-1,...`)
- Single-revision and multi-revision mRID chains

150–200 records would improve statistical robustness without becoming a project in its own right. The intent's "dozens to low hundreds" is well-calibrated.

**Where the set lives:** In-repo under `data/gold/remit_extraction.json` (or `.csv`) is defensible for ≤200 records. The set should carry a provenance comment documenting the mRID/revision range it was drawn from and the date it was labelled.

**Sources:**
- [Maxim AI — Building a Golden Dataset](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/) — 246-sample formula for 5% margin at 80% pass rate
- [Ontotext — Gold Standard in IE](https://www.ontotext.com/blog/gold-standard-key-to-information-extration-data-quality-control/) — two independent raters; quality over quantity
- [ArXiv 2510.15727 — Invoice IE evaluation](https://arxiv.org/pdf/2510.15727) — per-field ground truth; canary sets of 50–200
- [ArXiv 2407.02464 — Confidence Intervals for IR Evaluation](https://arxiv.org/abs/2407.02464) — prediction-powered inference for reliable CIs

---

## R4 — Agreement metrics for extraction

**Metrics in common use:**

| Metric | When to use | Notes |
|---|---|---|
| Exact match per field | Categorical fields (`fuel_type`, `event_type`) drawn from a closed vocabulary | Strictest; penalises any deviation |
| Token-level F1 (span F1) | Free-text or numeric fields where partial matches matter | ACL consensus for NER-type extraction; does not require defining negatives |
| Semantic similarity (BERTScore / cosine) | Free-text fields where paraphrase equivalence matters | More complex; harder to explain in a demo |
| Cohen's κ | Categorical inter-annotator agreement between two raters | Applies to labelling the gold set, not comparing LLM vs. gold |
| Krippendorff's α | IAA with >2 annotators or missing data | Generalises κ; appropriate if multiple people label the gold set |

**For Stage 14's specific fields:**

- `event_type`: exact match (closed vocabulary from `OUTPUT_SCHEMA`)
- `fuel_type`: exact match (already in the structured data; extraction confirms it from free text — agreement should be near-perfect)
- `affected_capacity_mw`: exact match with a numeric tolerance (e.g. ± 5 MW); or token-level F1 on the extracted number string
- `start_time` / `end_time`: exact match on parsed ISO-8601, or delta in hours from gold
- Confidence indicator: treat as a rank; evaluate as correlation with a human-assigned reliability score

**The two-metric design as a teaching point:** Using both exact match (harsh) and a ± tolerance / semantic match (forgiving) on the same field produces two numbers whose gap illustrates the impact of metric choice on reported accuracy. This is the demo moment for the evaluation harness: "the LLM agrees with our gold set on 72% of capacity values exactly, but on 91% within 10 MW — what does that tell us about the task?"

**ACL community position on IAA:** For span-based annotations, "pairwise F1 is preferred over κ because F1 does not require defining negative examples" (ArXiv 2603.06865). For the Stage 14 evaluation this translates to: use F1 for free-text / numeric span fields; use κ (or simple percentage agreement) for categorical fields.

**Sources:**
- [ArXiv 2603.06865 — Counting on Consensus: IAA Metrics](https://arxiv.org/html/2603.06865) — comprehensive review; F1 vs. κ guidance
- [Prodigy — Annotation Metrics](https://prodi.gy/docs/metrics) — pairwise F1 for NER; κ for categorical labels
- [Surge AI — Introduction to Cohen's Kappa](https://surge-ai.medium.com/inter-annotator-agreement-an-introduction-to-cohens-kappa-statistic-dcc15ffa5ac4) — accessible κ explanation
- [ArXiv 2512.20352 — Multi-LLM Thematic Analysis, dual-metric](https://arxiv.org/html/2512.20352) — precedent for combining κ + cosine similarity

---

## R5 — Prompt versioning conventions

**The production spectrum:** At the minimal end, hash the prompt string (SHA-256 of UTF-8 bytes) and record the hex digest as a column in the extraction output parquet. At the maximal end, use a dedicated prompt management service (Langfuse, PromptLayer, Weave) with registry, version labels, and trace linkage.

**Minimal-viable shape for Stage 14 (consistent with the project's registry pattern):**

1. Store the prompt template as a file under `conf/prompts/remit_extractor_v1.txt` (or `.jinja2`).
2. At extraction time, compute `hashlib.sha256(prompt_bytes).hexdigest()[:12]` and write it into the output parquet as `prompt_hash`.
3. Record `model_id` (e.g. `claude-haiku-4-5`) as a separate column.
4. The output parquet carries `(mrid, revision_number, prompt_hash, model_id, extracted_fields...)`.

Any change to the prompt file produces a different hash — "we swapped the prompt and everything changed" is immediately diagnosable from the run record without any external service.

**Langfuse (open-source):** Provides a prompt registry with SHA-hash tracking, tagging, and side-by-side diff. Prompt versions are linked to traces automatically. Self-hostable. Most popular open-source option as of 2025–2026. Adds a service dependency — defensible for a production team, optional for a pedagogical project.

**MLflow Prompt Registry (MLflow 3.x):** Integrates with experiment tracking; links prompt versions to LoggedModel versions via git commit hashes. Natural fit if the project later adopts MLflow as the run tracker for Stage 16 models. Not currently used in the project.

**Prompt-as-code vs. prompt-as-data:** Prompt-as-code (checked into version control as a file) is the recommended pattern for small projects: free diff history, branch-level isolation, no external service. Prompt-as-data (registry service) enables runtime swapping without a code deploy — relevant at scale.

**Researcher view:** The hash-in-parquet pattern is the smallest increment over the current Stage 9 registry convention. A `conf/prompts/` directory with one file per prompt version is the right location; it is parallel to `conf/` YAML files and consistent with §2.1.4 (configuration outside code).

**Sources:**
- [Langfuse Prompt Management](https://langfuse.com/docs/prompt-management/overview) — open-source prompt CMS; SHA-hash tracking; trace linkage
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/) — MLflow 3.x; git-hash versioning; experiment tracking integration
- [Maxim AI — Top 5 Prompt Versioning Tools 2026](https://www.getmaxim.ai/articles/top-5-prompt-versioning-tools-for-enterprise-ai-teams-in-2026/) — comparative overview including Langfuse, PromptLayer, Weave
- [PromptLayer — 5 best tools for prompt versioning](https://blog.promptlayer.com/5-best-tools-for-prompt-versioning/) — Langfuse, Weave, Arize Phoenix; feature comparison

---

## R6 — REMIT-domain prior art on structured extraction

**Finding: no published prior art exists on NLP-based or LLM-based structured extraction from REMIT UMM `relatedInformation` / `messageDescription` free text.**

A search of energy-industry tooling and academic databases found:

- REMIT compliance platforms (Nord Pool UMM, ENTSO-E transparency platform, Energy Quantified REMIT data product, Statkraft UMM UK) focus on structured *submission* and *display* of REMIT messages, not on extraction from the informal free-text field.
- ACER/Ofgem guidance documents describe required structured fields of a UMM; they do not address extraction from informal text.
- General NLP in energy utilities (EPRI 2019, ScienceDirect 2023) addresses maintenance-log classification and outage-report text analysis for operational purposes, but not the specific REMIT message format or the `relatedInformation` field.

**Stage 14 is novel at this level of specificity.** There is no benchmark, no pre-trained extractor, and no domain-specific tokeniser for REMIT `message_description` text. Closest analogies are:

1. **Invoice / document data extraction** — short, semi-structured business documents; ArXiv 2510.15727 is the strongest methodological reference.
2. **Maintenance log NLP in utilities** — similar semi-formal language with dates, capacities, and equipment identifiers.

**The informal encoding format** (`~/E_DERBY-1,2024-01-16 23:30:00,2024-01-17 06:00:00,0`) is documented in Stage 13 domain research as "bespoke to individual Elexon participants and appears undocumented." An LLM extractor must handle this gracefully — log the raw value, flag it as non-standard, and fall back to a default rather than raising.

**Sources:**
- [Nord Pool — REMIT UMM platform](https://www.nordpoolgroup.com/en/services/compliance/umm/) — structured submission; no extraction tooling
- [Elexon Insights — REMIT portal](https://bmrs.elexon.co.uk/remit) — display only
- [Energy Quantified — REMIT data](https://www.energyquantified.com/features/remit) — structured REMIT feed; no free-text extraction
- [Utility Analytics Institute — LLMs and grid analytics](https://utilityanalytics.com/large-language-models-grid-analytics/) — general LLM use in utilities; not REMIT-specific
- [EPRI — NLP in energy](https://www.epri.com/research/products/000000003002017321) — general NLP in utilities; maintenance-log focus

---

## R7 — Cost guardrails

### Per-event cost at Claude Haiku 4.5

Assumptions: input ~200 tokens (system prompt ~50 + `message_description` ~150), output ~150 tokens (5–6 field structured JSON).

| Scenario | Per-event USD | Per-event GBP (£1 = $1.35) |
|---|---|---|
| Standard, Haiku 4.5 ($1.00/$5.00) | ~$0.00095 | ~£0.00070 |
| Batch API, Haiku 4.5 ($0.50/$2.50) | ~$0.000475 | ~£0.00035 |
| Standard, Haiku 3.5 ($0.80/$4.00) | ~$0.00076 | ~£0.00056 |
| Standard, Sonnet 4.6 ($3.00/$15.00) | ~$0.00285 | ~£0.00211 |

With a 1,000-token few-shot system prompt cached at 0.10× the input rate, the cached input cost per call is $0.0001 (vs. $0.001 uncached) — a 90% saving on the system-prompt portion.

### Full-archive cost (~45,000 records, ~1-year history at ~125/day)

| Scenario | USD | GBP |
|---|---|---|
| Standard, Haiku 4.5 | ~$42.75 | ~£31.70 |
| Batch API, Haiku 4.5 | ~$21.38 | ~£15.84 |
| Standard, Sonnet 4.6 | ~$427.50 | ~£316.67 |
| Standard, GPT-4o-mini ($0.15/$0.60) | ~$16.88 | ~£12.50 |

**Assessment:** At Haiku 4.5 batch rates, the full one-year archive costs approximately £16. This is not a binding cost constraint. However, **running the extractor in CI or notebooks without a guard would silently accumulate costs at ~£0.07 per notebook execution** (100-record sample × £0.00070). The intent's requirement that CI default to the stub implementation is therefore load-bearing and must be enforced via an environment variable (e.g. `BRISTOL_ML_LLM_STUB=1`).

A 7-year archive (2018–2025; ~320,000 records per Stage 13 domain research §Archive depth) would cost ~£110 at Haiku 4.5 batch rates — still modest but worth confirming before a full backfill is triggered.

**Note on uncertainty:** These estimates assume ~200 input tokens and ~150 output tokens. The actual token count depends on whether the system prompt includes few-shot examples (adds 500–2,000 tokens) and whether `message_description` is null (many records). Pricing figures are from the Anthropic pricing page accessed 2026-04-26; flag and re-check before committing to a full-archive run.

**Sources:**
- [Anthropic Pricing page](https://platform.claude.com/docs/en/about-claude/pricing) — confirmed Haiku 4.5 $1/$5 standard; $0.50/$2.50 batch; caching 0.10×
- [OpenAI Pricing](https://openai.com/api/pricing/) — GPT-4o-mini $0.15/$0.60 per MTok
- [PoundSterlingLive — GBP/USD 2026](https://www.poundsterlinglive.com/history/GBP-USD-2026) — GBP/USD ~1.35 in April 2026

---

## Deprecations and sharp edges

| Item | Risk |
|---|---|
| **Claude Sonnet 3.7** | Deprecated per Anthropic pricing page (2026-04-26); do not target in new code |
| **Gemini 2.0 Flash** | Shut down 1 June 2026; migrate to Gemini 2.5 Flash or later before that date |
| **`output_config` beta header** | No longer required; structured outputs are GA on the Claude API |
| **`message_description` null rate** | The Elexon `/datasets/REMIT/stream` endpoint does not populate `messageDescription` in live responses (confirmed in Stage 13 implementation); Stage 14 must handle null values explicitly |
| **Informal schedule encoding** | `~/E_DERBY-1,...` format in `relatedInformation` is undocumented and participant-specific; graceful degradation (log + default) is the correct handling |
| **Qwen3 vLLM tool-call edge cases** | Active bug reports on malformed JSON from Qwen3 tool-calling in vLLM (issue #21711); not a risk if using a cloud API |

---

## Version / compatibility notes

| Item | Status as of 2026-04-26 |
|---|---|
| Anthropic Python SDK | `output_config` parameter on `messages.create`; GA, no beta header |
| Claude Haiku 4.5 model ID | `claude-haiku-4-5`; supported on Claude API + AWS Bedrock |
| Claude Sonnet 4.6 model ID | `claude-sonnet-4-6`; supported on Claude API + AWS Bedrock |
| OpenAI `response_format` strict | GA since August 2024; `strict: true` in `json_schema` format |
| `instructor` library | PyPI `instructor`; actively maintained; 3 M+ monthly downloads |
| Langfuse | v3.x; open-source; self-hostable |
| Qwen3 on vLLM | `--tool-call-parser qwen3_xml`; some streaming edge cases still open |
| GBP/USD rate | ~1.35 (April 2026); used for all GBP cost estimates in this document |

---

## Canonical sources

| Source | URL | One-line summary |
|---|---|---|
| Anthropic Structured Outputs docs | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | `output_config` mechanism, supported keywords, constrained-decoding guarantee |
| Anthropic Pricing | https://platform.claude.com/docs/en/about-claude/pricing | Haiku 4.5 $1/$5 MTok; Sonnet 4.6 $3/$15 MTok; Batch API 50% off; cache hits 0.10× |
| OpenAI Structured Outputs guide | https://platform.openai.com/docs/guides/structured-outputs | strict mode, CFG token masking, schema limitations |
| OpenAI Introducing Structured Outputs | https://openai.com/index/introducing-structured-outputs-in-the-api/ | August 2024 GA; CFG mechanism described |
| OpenAI Pricing | https://openai.com/api/pricing/ | GPT-4o-mini $0.15/$0.60 per MTok |
| Gemini Structured Output docs | https://ai.google.dev/gemini-api/docs/structured-output | `responseSchema`, supported types, semantic-only caveat |
| Google Gemini Structured Outputs blog (April 2026) | https://blog.google/technology/developers/gemini-api-structured-outputs/ | April 2026 enhancements; Gemini 3-series model support |
| instructor library | https://python.useinstructor.com/ | Pydantic validation + retry loop over LLM responses |
| instructor — Why use Instructor? | https://python.useinstructor.com/why/ | ~10% raw JSON failure rate; retry-on-validation approach |
| Simmering.dev — structured output vs instructor | https://simmering.dev/blog/openai_structured_output/ | Constrained generation vs. learned behaviour |
| ACM IPM — Are LLMs good at structured outputs? | https://dl.acm.org/doi/10.1016/j.ipm.2024.103809 | Benchmark; GPT-4o 76% on complex schemas |
| cleanlab.ai — Benchmark quality | https://cleanlab.ai/blog/structured-output-benchmark/ | Ground-truth errors in existing benchmarks |
| ArXiv 2603.06865 — IAA Metrics | https://arxiv.org/html/2603.06865 | F1 preferred over κ for span annotations |
| Prodigy Annotation Metrics | https://prodi.gy/docs/metrics | Pairwise F1 for NER; κ for categorical labels |
| Surge AI — Cohen's Kappa | https://surge-ai.medium.com/inter-annotator-agreement-an-introduction-to-cohens-kappa-statistic-dcc15ffa5ac4 | Accessible κ explanation for categorical labels |
| Maxim AI — Golden Dataset sizing | https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/ | 246-sample formula for 5% margin at 80% pass rate |
| ArXiv 2510.15727 — Invoice IE evaluation | https://arxiv.org/pdf/2510.15727 | Per-field ground truth; canary sets 50–200 examples |
| Langfuse Prompt Management | https://langfuse.com/docs/prompt-management/overview | Open-source prompt CMS; SHA-hash tracking |
| MLflow Prompt Registry | https://mlflow.org/docs/latest/genai/prompt-registry/ | MLflow 3.x; git-hash versioning; experiment run linkage |
| Maxim AI — Top 5 Prompt Versioning Tools 2026 | https://www.getmaxim.ai/articles/top-5-prompt-versioning-tools-for-enterprise-ai-teams-in-2026/ | Comparative overview of Langfuse, PromptLayer, Weave |
| Qwen3 blog | https://qwenlm.github.io/blog/qwen3/ | Apache 2.0; dense 0.6 B–32 B; MoE variants |
| vLLM Tool Calling docs | https://docs.vllm.ai/en/stable/features/tool_calling/ | qwen3_xml parser; structured output support |
| vLLM issue #21711 — Qwen3 tool-call bug | https://github.com/vllm-project/vllm/issues/21711 | Active malformed-JSON reports for Qwen3 tool calling |
| Nord Pool — REMIT UMM | https://www.nordpoolgroup.com/en/services/compliance/umm/ | Structured UMM submission; no free-text extraction tooling |
| Energy Quantified — REMIT | https://www.energyquantified.com/features/remit | Structured REMIT data feed; no extraction |
| PoundSterlingLive — GBP/USD 2026 | https://www.poundsterlinglive.com/history/GBP-USD-2026 | GBP/USD ~1.35 in April 2026 |

---

*This artefact is one of four Phase-1 research inputs for Stage 14. It covers external technical context only. Requirements, codebase mapping, and scope boundaries are in the companion artefacts.*
