# Stage 0 — Project Foundation

**Status:** Intent (immutable once stage is shipped)
**Depends on:** nothing
**Enables:** every subsequent stage

## Purpose

Establish the repository skeleton and shared conventions that every later stage will attach to. This stage writes almost no application code. Its value is that all nineteen later stages can assume a working build, a green test run, a configuration framework, and a documentation home exist.

## Scope

In scope:
- Build tooling, lint, format, test framework.
- Continuous integration on every push.
- The configuration framework (Hydra + Pydantic per DESIGN §7), producing a validated typed object from composed YAML plus CLI overrides.
- A minimal CLI entry point that resolves the configuration and prints it.
- Top-level project documentation: a pitch-level README, this stage's completion record, the initial architecture decision records, and a top-level guide for Claude Code.
- An empty but present tests tree with one smoke test that fails loudly if the package cannot be imported.

Out of scope:
- Any application module (ingestion, features, models, evaluation, LLM, registry, serving, monitoring).
- Real data of any kind.
- Notebook tooling.
- Full documentation site generation.

## Demo moment

From a clean clone, the project builds, tests pass, and the CLI's `--help` output demonstrates that the configuration framework is live and composable, without any feature code existing yet. The repository explains itself on a `tree` inspection.

## Acceptance criteria

1. A contributor cloning the repo can install and run the test suite in one or two commands, with fast feedback.
2. CI on the first pull request is green.
3. Lint and format checks pass across the whole tree.
4. The configuration framework loads a minimal default configuration and rejects malformed overrides with a clear error.
5. The top-level Claude Code guide fits within the lightweight length target recommended by Claude Code best-practice writing.

## Points for consideration

- How long the top-level Claude Code guide should be before it starts being ignored. Recent advice from Claude Code practitioners points at "short enough to be read in full, every time."
- Whether notebook output stripping should be wired into pre-commit now or retrofitted when notebooks first appear.
- How strict the CI should be on its first pass. Very strict is cheap to soften; very lax is expensive to tighten.
- Whether the lockfile for the package manager belongs in version control. For a reference implementation where reproducibility at a meetup matters, the case for committing it is strong.
- How to name environment variables so that secrets (when they arrive) sit in an obvious namespace distinct from ordinary configuration.
- Whether to scaffold placeholder configuration groups now, or leave the configuration tree empty until the stages that populate it arrive. An empty tree is honest; a placeholder tree is suggestive.
- Hydra's current major version and whether pre-release versions of the next major are worth the churn.

## Dependencies

Upstream: none.

Downstream: every subsequent stage depends on Stage 0. Stages 4, 9, and 13 lean most heavily on the configuration framework.

## Out of scope, explicitly deferred

- Any form of runtime caching of fetched data (arrives with the first ingestion stage).
- Any `retrieved_at` or bi-temporal storage convention (arrives when it first matters, likely Stage 4 or Stage 13).
- Documentation site generation beyond raw markdown.
