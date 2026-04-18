"""Pydantic schemas for the resolved application config.

Hydra composes YAML; this module validates the resolved tree. Adding a new
config group (ingestion, model, evaluation, ...) means adding a sub-model
here and a field on `AppConfig`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: ProjectConfig
