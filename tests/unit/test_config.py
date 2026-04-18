"""Stage 0 smoke tests for the config pipeline and the `python -m bristol_ml` demo moment."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from bristol_ml import load_config
from conf._schemas import AppConfig


def test_load_config_defaults_produce_app_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "bristol_ml"
    assert cfg.project.seed >= 0


def test_load_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides=["+project.bogus=1"])


def test_python_dash_m_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Powered by Hydra" in result.stdout
