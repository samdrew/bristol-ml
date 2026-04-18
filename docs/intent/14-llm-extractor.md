# Stage 14 — LLM feature extractor (stub + real)

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 13
**Enables:** Stages 15, 16

## Purpose

Turn free-text REMIT descriptions into structured features, and do so through an interface that has two implementations: a hand-labelled stub, and a real implementation that calls an LLM. The stub is the default, so that CI, tests, and attendees without API keys can run everything. The real implementation is the production path. This stage's architectural point is as much as the extraction itself: any expensive or flaky external dependency in an ML system benefits from a stub-first design with an evaluation harness that holds the real implementation to account.

## Scope

In scope:
- An interface that takes a REMIT event (or a batch) and returns structured features — event type, affected fuel type, affected capacity, normalised start/end times, a confidence or reliability indicator.
- A stub implementation backed by a small hand-labelled set of events, returning fixed features for the known events and a sensible default for unknown ones.
- A real implementation that calls an LLM to extract features from the free text.
- An evaluation harness that runs both implementations over the hand-labelled set and reports agreement and per-field accuracy.
- Configuration selecting which implementation is active.

Out of scope:
- Embedding or semantic search (Stage 15).
- Use of the extracted features in a model (Stage 16).
- Prompt engineering as an ongoing activity.
- Fine-tuning or distillation.

## Demo moment

A facilitator runs the evaluation harness and sees a side-by-side comparison: the hand-labelled truth, the stub's output (trivially correct on the known set), and the real LLM's output with its agreement score. Any disagreements are visible, so the conversation can be about "here's where the LLM got it wrong, and here's what that tells us about extraction reliability."

## Acceptance criteria

1. The interface is small enough that writing a third implementation in the future is plausible.
2. The stub is the default; running anything that consumes the extractor works offline with no API key.
3. The real implementation is guarded by a configuration switch and uses an environment variable for the API key.
4. The evaluation harness produces a reproducible accuracy report against the hand-labelled set.
5. The extracted feature schema is typed and validated at the interface boundary.

## Points for consideration

- The hand-labelled set's size and composition. Too small and the accuracy estimate is meaningless; too large and curating it is a project of its own. Something in the range of dozens to low hundreds is plausible, covering a mix of event types and fuels.
- What counts as "agreement" between the stub and the LLM. Exact-match on every field is harsh; semantic-match on key fields (event type, approximate capacity) is more forgiving. Different choices produce different numbers, which makes the metric choice a lesson itself.
- Prompt design. A single prompt with few-shot examples is the obvious starting point. More elaborate prompting (chain-of-thought, self-critique, structured-output constraints) is a rabbit hole; it has diminishing returns for this task.
- Structured-output APIs vs free-form parsing. Some LLM APIs support a JSON-schema-constrained mode. This is cheaper to parse but may reduce extraction quality compared to free-form-then-parse.
- Cost. Running the real extractor across the full REMIT archive has a meaningful cost. Default behaviour in notebooks and CI should avoid that cost; running against a sample is a reasonable middle ground.
- Versioning the prompt. The extractor's behaviour is a function of the prompt as much as the model. Any registered feature extraction should record which prompt produced it, so "we swapped the prompt and everything changed" is diagnosable.
- Where the hand-labelled set lives. In-repo is defensible for a small set; as a separate data file with a documented source is more honest about its nature.
- Failure modes. The LLM will occasionally return malformed output. Graceful degradation (log, fall back to a default) is better than raising; this is a production pattern worth establishing early.

## Dependencies

Upstream: Stage 13 (REMIT ingestion).

Downstream: Stage 15 (embedding index is a parallel thread on the same data), Stage 16 (the extracted features join the feature table).

## Out of scope, explicitly deferred

- Fine-tuning a model on REMIT data.
- Active-learning loops on the hand-labelled set.
- Multi-model ensemble extraction.
- Streaming-friendly extraction for real-time messages.
