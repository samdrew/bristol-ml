"""Pydantic schemas for the resolved application config.

Hydra composes YAML; this module validates the resolved tree. Adding a new
config group (ingestion, model, evaluation, ...) means adding a sub-model
here and a field on `AppConfig`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)


class NesoYearResource(BaseModel):
    """One year of the NESO Historic Demand Data dataset.

    The mapping from year to CKAN resource UUID is not derivable — NESO assigns
    a distinct UUID per annual CSV. Adding a new year is one entry here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    year: int = Field(ge=2001, le=2100)
    resource_id: UUID


class NesoIngestionConfig(BaseModel):
    """Configuration for the NESO historic demand ingestion (Stage 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: HttpUrl = Field(default=HttpUrl("https://api.neso.energy/api/3/action/"))
    resources: list[NesoYearResource]
    cache_dir: Path
    cache_filename: str = "neso_demand.parquet"
    page_size: int = Field(default=32_000, ge=1, le=32_000)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=10.0, gt=0)
    columns: list[str] = Field(default_factory=lambda: ["ND", "TSD"])
    min_inter_request_seconds: float = Field(default=30.0, ge=0)


class WeatherStation(BaseModel):
    """One weather station (a single point query to Open-Meteo's archive).

    Coordinates identify the geodesic point passed to the archive API; the
    API snaps to the nearest ERA5-Land / ERA5 / CERRA grid cell via
    ``cell_selection=land`` (default), so ±0.05° of lat/lon jitter is
    analytically irrelevant. ``weight`` is population-based (ONS 2011 BUA) by
    default; the source string records provenance for a later refresh.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    weight: float = Field(gt=0)
    weight_source: str = Field(default="")


class WeatherIngestionConfig(BaseModel):
    """Configuration for the Open-Meteo historical-weather archive ingestion.

    Stage 2 introduces this alongside the NESO config. The retry/rate-limit
    knobs mirror ``NesoIngestionConfig`` structurally so the shared helpers
    in ``bristol_ml.ingestion._common`` accept both configs via ``Protocol``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: HttpUrl = Field(default=HttpUrl("https://archive-api.open-meteo.com/v1/archive"))
    stations: list[WeatherStation]
    variables: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ]
    )
    start_date: date
    end_date: date | None = None
    cache_dir: Path
    cache_filename: str = "weather.parquet"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=10.0, gt=0)
    min_inter_request_seconds: float = Field(default=0.25, ge=0)
    timezone: str = Field(default="UTC")

    @model_validator(mode="after")
    def _validate_dates(self) -> WeatherIngestionConfig:
        """Enforce ``start_date <= end_date`` and unique station names."""
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(f"end_date {self.end_date} precedes start_date {self.start_date}")
        names = [s.name for s in self.stations]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate station name(s): {sorted(dupes)}")
        if not self.stations:
            raise ValueError("At least one station must be configured.")
        return self


class IngestionGroup(BaseModel):
    """Container for per-source ingestion configs.

    Each field is optional so stage-0 configs (no ingestion section) still
    validate. Stages add sibling fields as they land (weather, holidays,
    Elexon, ...).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    neso: NesoIngestionConfig | None = None
    weather: WeatherIngestionConfig | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: ProjectConfig
    ingestion: IngestionGroup = Field(default_factory=IngestionGroup)
