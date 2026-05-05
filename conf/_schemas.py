"""Pydantic schemas for the resolved application config.

Hydra composes YAML from ``conf/``; this module validates the resolved
tree.  Adding a new config group means adding a sub-model here and a
field on :class:`AppConfig`.

Six schemas ship with the template:

- :class:`ProjectConfig` — the project's logical name + RNG seed.
- :class:`SplitterConfig` — rolling-origin time-series CV recipe
  (kept because most quantitative-analysis projects need it; safe to
  drop if your project is not time-series-shaped).
- :class:`MetricsConfig` — list of named metric functions to compute.
- :class:`PlotsConfig` — diagnostic-plot defaults (figsize, DPI,
  display timezone).
- :class:`ServingConfig` — host / port / artefact-root for projects
  that ship an HTTP service.
- :class:`ModelMetadata` — immutable provenance record carried by
  any artefact-producing component.

Plus one toy schema for the worked example:

- :class:`TextStatsConfig` — the ``services.text_stats`` group (see
  ``src/TEMPLATE_PROJECT/services/text_stats_service.py``).

Replace :class:`TextStatsConfig` and the ``services`` field on
:class:`AppConfig` when you adapt the template to your project.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectConfig(BaseModel):
    """Project-level identifiers consumed by every component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Lowercase snake_case logical name — used in run identifiers and
    # provenance records.  Distinct from the Python package name (which
    # is ``TEMPLATE_PROJECT`` until you rename the package).
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    # Single project-wide seed; components consume it deterministically.
    seed: int = Field(ge=0)


class SplitterConfig(BaseModel):
    """Rolling-origin time-series train/test splitter configuration.

    ``min_train_periods`` fixes the minimum training window before the
    first test origin.  ``test_len`` is the fold's test-window length.
    ``step`` advances the origin by this many rows between folds.
    ``gap`` introduces an embargo between the end of training and the
    start of testing.  ``fixed_window=True`` turns the default
    expanding window into a sliding window of size
    ``min_train_periods``.

    Drop this schema (and its ``EvaluationGroup`` field) if your
    project is not a time-series forecasting workload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_train_periods: int = Field(ge=1)
    test_len: int = Field(ge=1)
    step: int = Field(ge=1)
    gap: int = Field(default=0, ge=0)
    fixed_window: bool = False


class MetricsConfig(BaseModel):
    """Named metric functions an evaluator harness should compute.

    The ``Literal`` enumerates the metric names your project supports.
    The default below is a placeholder; extend it to match your
    project's metric registry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: tuple[Literal["mae", "mse", "rmse"], ...] = ("mae", "rmse")


class PlotsConfig(BaseModel):
    """Diagnostic-plot defaults: figsize, DPI, display timezone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Projector-friendly default; override per-helper for finer control.
    figsize: tuple[float, float] = (12.0, 8.0)
    dpi: int = Field(default=110, ge=50, le=400)
    # IANA timezone for human-readable axis labels; ``UTC`` is the
    # safe default for any DST-sensitive analysis.
    display_tz: str = Field(default="UTC")


class EvaluationGroup(BaseModel):
    """Container for evaluation-side configs (optional)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    splitter: SplitterConfig | None = None
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)


class ServingConfig(BaseModel):
    """HTTP serving configuration (optional).

    Localhost-only by default; deployment beyond localhost is your
    project's call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artefact_dir: Path = Path("data/artefacts")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class ModelMetadata(BaseModel):
    """Immutable provenance record carried by any fitted artefact.

    The shape is deliberately minimal: a name, the feature columns the
    artefact was produced from, when it was produced, the Git SHA at
    that time, and a free-form ``hyperparameters`` bag for
    component-specific state.  ``hyperparameters`` is the extensibility
    escape hatch — concrete projects extend it without churning the
    other fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    feature_columns: tuple[str, ...]
    fit_utc: datetime | None = None
    git_sha: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fit_utc(self) -> ModelMetadata:
        """Reject naive ``fit_utc`` values: only tz-aware UTC times."""
        if self.fit_utc is not None and self.fit_utc.tzinfo is None:
            raise ValueError(
                f"fit_utc must be tz-aware (UTC); got naive datetime {self.fit_utc!r}."
            )
        return self


# ---------------------------------------------------------------------------
# Worked-example schema — replace when adapting the template
# ---------------------------------------------------------------------------


class TextStatsConfig(BaseModel):
    """Configuration for the worked-example text-statistics service.

    Demonstrates the template's pattern for adding a service:

    1. Define a Pydantic schema here (frozen + ``extra="forbid"``).
    2. Add a Hydra group YAML under ``conf/services/<name>.yaml`` with
       ``# @package services.<name>``.
    3. Reference the YAML in ``conf/config.yaml`` under ``defaults``.
    4. Add a field to :class:`AppConfig` typed by this schema.
    5. Implement the service module under
       ``src/TEMPLATE_PROJECT/services/<name>.py`` and dispatch from
       ``cli.py`` (or a service-local ``__main__``) on the validated
       value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Path to the input file the service reads.  Resolved relative to
    # the Hydra working directory; pass an absolute path to override.
    input_path: Path
    # ``json`` (machine-readable) or ``human`` (a small text table).
    output_format: Literal["json", "human"] = "json"


class ServicesGroup(BaseModel):
    """Container for service-layer configs.

    Each shipped service has a field below.  Replace
    :class:`TextStatsConfig` (and the ``text_stats`` field) with your
    own service schemas when adapting the template.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_stats: TextStatsConfig | None = None


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Validated root config — the single object every component reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project: ProjectConfig
    evaluation: EvaluationGroup = Field(default_factory=EvaluationGroup)
    serving: ServingConfig | None = None
    services: ServicesGroup = Field(default_factory=ServicesGroup)
