# Adapting `TEMPLATE_PROJECT` to your project

This template carries the **infrastructure** (Hydra+Pydantic config,
Claude Code agent roster, four-tier doc methodology, test layout,
lint/format/test config, pre-commit, Dockerfile) without imposing a
domain.  A trivial worked example (the text-statistics service)
demonstrates the layer pattern; you'll replace it with your own
domain code on first use.

## Step 1 — Rename the package

The placeholder is the literal string `TEMPLATE_PROJECT`.  Pick a
snake-case name for your project (for the rest of this doc:
`my_project`).  Then:

```bash
# Rename the source directory
git mv src/TEMPLATE_PROJECT src/my_project

# Find every reference to the old name
grep -rln TEMPLATE_PROJECT . \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=.pytest_cache \
    --exclude-dir=.ruff_cache \
    --exclude-dir=outputs

# Update them in place (review the diff before committing)
grep -rl TEMPLATE_PROJECT . \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=.pytest_cache \
    --exclude-dir=.ruff_cache \
    --exclude-dir=outputs \
    | xargs sed -i 's/TEMPLATE_PROJECT/my_project/g'
```

Files that will change: `pyproject.toml`, `CLAUDE.md`, `README.md`,
`Dockerfile`, `src/my_project/{__init__,__main__,cli}.py`,
`src/my_project/{core,services}/CLAUDE.md`,
`tests/{unit,integration}/...`, `conf/_schemas.py`.

## Step 2 — Set the project metadata

Edit `pyproject.toml`:

- `[project].name = "my_project"`
- `[project].description = "..."`
- `[project].version = "0.0.0"` (or whatever)

Edit `conf/config.yaml`:

- `project.name: my_project` (lowercase snake_case; matches the
  Pydantic regex on `ProjectConfig.name`).

## Step 3 — Choose your dependency stack

The template ships with the minimum runtime deps: `hydra-core`,
`omegaconf`, `pydantic`, `loguru`.  Add what your project needs:

```bash
uv add pandas pyarrow httpx                    # data work
uv add fastapi uvicorn                         # HTTP service
uv add scikit-learn                            # classical ML
uv add torch                                   # neural networks
# etc.
```

Re-run `uv sync --group dev` after each change.

## Step 4 — Decide what to keep from the worked example

The text-stats service is pedagogical scaffolding, not load-bearing.
You have three options:

**A. Strip it entirely** before your first commit.  Delete:

- `src/my_project/core/text_stats.py`
- `src/my_project/services/text_stats_service.py`
- `conf/services/text_stats.yaml` (and remove the
  `services: text_stats` line from `conf/config.yaml`'s defaults
  list; you'll likely want to remove the `services` field from
  `AppConfig` and `ServicesGroup` from `conf/_schemas.py`)
- `tests/unit/core/test_text_stats.py`
- `tests/integration/test_text_stats_service.py`
- `tests/fixtures/sample_text.txt`
- `docs/architecture/layers/{core,services}.md` (or rename them to
  reflect your actual layer pattern)
- `src/my_project/{core,services}/CLAUDE.md` (likewise)

There's also an **empty-scaffold variant** of this template on its
own branch — pull from there instead if you want a guaranteed-clean
starting point.

**B. Keep it as a reference** while you build.  Add your own modules
alongside; delete the example when it's no longer useful.

**C. Adapt it.**  The text-stats shape (read input file → call pure
function → render output) is generic enough that a lot of CLI
utilities fit it.  Rename `text_stats` to your verb-noun and rewrite
the body; keep the surrounding plumbing.

## Step 5 — Edit the project spec

`docs/intent/DESIGN.md` describes the project: what it is, what
principles it holds to, what stages it ships in.  The template
ships a generic version with the worked example as Stage 0; rewrite
it for your project.

The four-tier doc methodology (`intent` / `architecture` / `plans`
/ `lld`) is described in `CLAUDE.md`.  Read that section before you
start the first stage so the agent roster and the doc-tier write
hooks match your expectations.

## Step 6 — Verify

```bash
uv sync --group dev
uv run python -m my_project --help
uv run python -m my_project
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pre-commit run --all-files
```

If all of these pass on a renamed clean checkout, the template is
healthy.

## Step 7 — Start your first stage

Open Claude Code and let the lead orchestrator drive:

```bash
claude --agent lead
```

The lead reads `CLAUDE.md`, sees the empty `docs/plans/active/`,
and prompts you for the first feature's intent.  From there it
follows the three-phase pipeline (Discovery → Implementation →
Review) described in `CLAUDE.md`.

If you don't want the full pipeline ceremony for early-stage work,
you can:

- Skip the formal intent doc and start a plan directly under
  `docs/plans/active/`.
- Skip the per-stage retro until you have something worth recording.
- Drop the agent roster entirely — `CLAUDE.md`'s structural
  conventions still apply, and you can run a single Claude Code
  session without `--agent lead`.

The methodology is opinionated by default; lighten it as your
project's stakes warrant.
