"""Smoke tests for the Hydra+Pydantic config wiring.

Pins the template's load-bearing contract: ``load_config()`` resolves
the YAML tree, ``validate()`` enforces the Pydantic schemas, and the
default ``conf/config.yaml`` produces a valid :class:`AppConfig`.

When you adapt the template, extend these tests to pin your project's
specific config defaults — but keep the smoke test that proves the
default config loads.
"""

from __future__ import annotations

from pathlib import Path

from conf._schemas import AppConfig
from TEMPLATE_PROJECT.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def test_default_config_loads_and_validates() -> None:
    """``conf/config.yaml`` resolves to a valid ``AppConfig``."""
    cfg = load_config(config_path=CONF_DIR)
    assert isinstance(cfg, AppConfig)
    # Project block is mandatory.
    assert cfg.project.name == "template_project"
    assert cfg.project.seed == 0


def test_default_config_includes_text_stats_service() -> None:
    """The default defaults list activates the worked-example service."""
    cfg = load_config(config_path=CONF_DIR)
    assert cfg.services.text_stats is not None
    assert cfg.services.text_stats.output_format == "json"


def test_overrides_propagate() -> None:
    """Hydra-style overrides reach the validated AppConfig."""
    cfg = load_config(
        config_path=CONF_DIR,
        overrides=["project.seed=42", "services.text_stats.output_format=human"],
    )
    assert cfg.project.seed == 42
    assert cfg.services.text_stats is not None
    assert cfg.services.text_stats.output_format == "human"
