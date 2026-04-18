# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Reorganised `docs/` into a tiered `intent/` / `architecture/` / `lld/` layout to support `.claude/hooks/tiered-write-paths.sh` from the lead agent. `DESIGN.md` moved to `docs/intent/DESIGN.md`; ADRs to `docs/architecture/decisions/`; stage retrospectives to `docs/lld/stages/`. Lead agent frontmatter wired to the tiered hook.

## [0.0.0] — 2026-04-18

### Added

- Project scaffold: `pyproject.toml` (hatchling, Python 3.12), `.gitignore`, `README.md`, `CLAUDE.md`.
- Hydra + Pydantic config pipeline: `conf/config.yaml`, `conf/_schemas.py`, `src/bristol_ml/config.py`, `src/bristol_ml/cli.py`.
- `python -m bristol_ml` entry point via `src/bristol_ml/__main__.py`.
- `py.typed` marker.
- Test suite (`tests/unit/test_config.py`) covering config load, `extra="forbid"` rejection, and the `--help` smoke.
- CI workflow (`.github/workflows/ci.yml`) — ruff format check, ruff lint, pytest via uv.
- Pre-commit hooks (`.pre-commit-config.yaml`).
- Docs: `docs/architecture.md` (pointer), ADRs 0001 and 0002, `docs/stages/00-foundation.md`.
