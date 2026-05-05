# Stage 0 — Foundation

**Goal.**  Land the template scaffold and a worked example that
demonstrates the four-tier doc methodology, the `core/` + `services/`
layer split, the Hydra+Pydantic config wiring, and the unit +
integration test layout.

**Demo moment.**  Run
`uv run python -m TEMPLATE_PROJECT.services.text_stats_service`
and watch it print character / word / line counts for a fixture
file as JSON.  Run `uv run pytest` and watch 14 tests pass.  Read
`docs/intent/DESIGN.md` and see the project spec template, then
read this file and see how a stage's intent doc looks.

## Acceptance criteria

1. **Boot.**  `uv sync --group dev` succeeds on a clean checkout.
2. **CLI.**  `uv run python -m TEMPLATE_PROJECT --help` prints
   Hydra help.  `uv run python -m TEMPLATE_PROJECT` resolves the
   default config and prints `AppConfig` as JSON.
3. **Service end-to-end.**
   `uv run python -m TEMPLATE_PROJECT.services.text_stats_service`
   reads `tests/fixtures/sample_text.txt`, calls
   `core.text_stats.compute_text_statistics`, prints JSON with
   `character_count`, `non_whitespace_character_count`,
   `word_count`, `line_count`.  Exits 0.
4. **Tests.**  `uv run pytest` runs 14 tests, all pass.  Includes:
   7 unit tests on `compute_text_statistics`, 4 integration tests
   on the service (`run()` + subprocess), 3 smoke tests on the
   config wiring.
5. **Lint.**  `uv run ruff check . && uv run ruff format --check .`
   clean.

## Out of scope

- The empty-scaffold variant of the template (separate branch).
- Replacing the worked example with a richer demo.
- Any project-specific code (this is the template; concrete
  projects build on top).

## Dependencies

None.  This is the foundation stage.

## Points for consideration

- The **renameable layer stubs** (`core/`, `services/`) are a
  starting point, not a constraint.  The first decision a real
  project makes after instantiating the template is whether to
  keep these names, rename them, or replace them with an
  ML-pipeline / HTTP-service / data-analysis layout.
- The **worked example** is pedagogical scaffolding.  Concrete
  projects strip it on first commit (option A in
  `TEMPLATE_USAGE.md` step 4) or keep it as a reference while
  building the actual domain (option B).
- The **methodology weight** can be lightened.  `CLAUDE.md`'s
  three-phase pipeline + four-tier docs + 13-agent roster is
  opinionated; concrete projects can drop the agent roster, drop
  the per-stage retro convention, drop the four-tier doc system
  — `CLAUDE.md` documents which lighter modes are coherent.
