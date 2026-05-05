"""Hydra entry point for ``python -m TEMPLATE_PROJECT``.

The template's CLI does nothing but prove the resolve-then-validate
pipeline: compose the Hydra config, validate via Pydantic, print the
result as JSON.  Concrete projects mount their own behaviour by
dispatching on the validated ``AppConfig`` (see the worked example
in ``services.text_stats_service``) or by adding subcommands.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from TEMPLATE_PROJECT.config import validate


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    validated = validate(cfg)
    print(validated.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
