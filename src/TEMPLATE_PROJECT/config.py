"""Config resolution and validation.

Hydra composes YAML from `conf/`; this module converts the resolved `DictConfig`
into a validated Pydantic `AppConfig`. Downstream code only ever sees the
Pydantic model — never raw `DictConfig`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from conf._schemas import AppConfig


def to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve interpolations and return a plain dict."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(container, dict)
    return container


def validate(cfg: DictConfig | Mapping[str, Any]) -> AppConfig:
    """Validate a resolved config against the Pydantic schema.

    Accepts either a `DictConfig` (from `@hydra.main`) or a plain mapping
    (from tests constructing configs programmatically). The top-level
    `hydra` block that Hydra injects into the resolved config is ignored
    by the schema's `extra="forbid"` rule only because it is stripped
    by Hydra before the user's callback runs; when it does reach here
    (e.g. via `compose(return_hydra_config=True)`), callers must strip it.
    """
    data = to_plain_dict(cfg) if isinstance(cfg, DictConfig) else dict(cfg)
    return AppConfig.model_validate(data)


def load_config(
    overrides: Sequence[str] = (),
    config_name: str = "config",
    config_path: str | Path = "conf",
) -> AppConfig:
    """Programmatic loader for tests and notebooks.

    Uses `initialize_config_dir` with an absolute path so the loader is
    independent of the caller's working directory.
    """
    abs_path = Path(config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(abs_path)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return validate(cfg)
