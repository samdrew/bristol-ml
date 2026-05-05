# Changelog

All notable changes to this project will be documented in this
file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial template scaffold extracted from `bristol_ml`.  Ships the
  Hydra+Pydantic config wiring (`TEMPLATE_PROJECT.config`), the
  generic schemas (`ProjectConfig`, `SplitterConfig`,
  `MetricsConfig`, `PlotsConfig`, `ServingConfig`, `ModelMetadata`),
  the four-tier documentation methodology
  (`intent/architecture/plans/lld`), the Claude Code agent roster
  (`.claude/agents/`), and the path-tier write hooks
  (`.claude/hooks/`).
- Worked-example text-statistics service demonstrating the
  template's `core/` + `services/` layer pattern end-to-end:
  `src/TEMPLATE_PROJECT/core/text_stats.py` (pure function),
  `src/TEMPLATE_PROJECT/services/text_stats_service.py` (Hydra
  wrapper), `conf/services/text_stats.yaml`, plus 7 unit and 4
  integration tests.
- Stage 0 foundation docs: intent
  (`docs/intent/00-foundation.md`), plan
  (`docs/plans/completed/00-foundation.md`), retro
  (`docs/lld/stages/00-foundation.md`), one ADR
  (`docs/architecture/decisions/0001-use-hydra-plus-pydantic.md`),
  two layer docs (`docs/architecture/layers/{core,services}.md`).
