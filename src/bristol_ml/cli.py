"""Hydra entry point for `python -m bristol_ml`.

Stage 0's CLI does nothing but prove the resolve-then-validate pipeline:
compose the config, validate it, print it. Subsequent stages mount their
own subcommands or use config overrides to select behaviour.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bristol_ml.config import validate


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    validated = validate(cfg)
    print(validated.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
