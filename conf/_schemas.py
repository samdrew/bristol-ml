"""Pydantic schemas for the resolved application config.

Hydra composes YAML; this module validates the resolved tree. Adding a new
config group (ingestion, model, evaluation, ...) means adding a sub-model
here and a field on `AppConfig`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal
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


class NesoForecastIngestionConfig(BaseModel):
    """Configuration for the NESO day-ahead demand forecast archive (Stage 4).

    Target CKAN resource: the *Day Ahead Half Hourly Demand Forecast Performance*
    dataset (resource UUID ``08e41551-80f8-4e28-a416-ea473a695db9``), which
    publishes the half-hourly day-ahead forecast alongside outturn and APE
    from April 2021 onwards.  Stage 4's benchmark comparison aggregates this
    half-hourly series to hourly per ``NesoBenchmarkConfig.aggregation``.

    Retry/rate-limit fields mirror ``NesoIngestionConfig`` structurally so the
    shared helpers in ``bristol_ml.ingestion._common`` accept both configs via
    their structural ``Protocol`` types.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: HttpUrl = Field(default=HttpUrl("https://api.neso.energy/api/3/action/"))
    # Single-resource archive (unlike NesoIngestionConfig's per-year list): the
    # Day Ahead HH Demand Forecast Performance dataset is one CKAN resource.
    resource_id: UUID
    cache_dir: Path
    cache_filename: str = "neso_forecast.parquet"
    page_size: int = Field(default=32_000, ge=1, le=32_000)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=10.0, gt=0)
    min_inter_request_seconds: float = Field(default=30.0, ge=0)
    # Subset of the forecast resource's columns we cache locally. Default
    # covers the demand forecast, outturn, and published APE — enough for the
    # three-way benchmark comparison without persisting extraneous columns.
    columns: list[str] = Field(
        default_factory=lambda: [
            "FORECASTDEMAND",
            "OUTTURN",
            "APE",
        ]
    )


class IngestionGroup(BaseModel):
    """Container for per-source ingestion configs.

    Each field is optional so stage-0 configs (no ingestion section) still
    validate. Stages add sibling fields as they land (weather, holidays,
    Elexon, ...).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    neso: NesoIngestionConfig | None = None
    weather: WeatherIngestionConfig | None = None
    neso_forecast: NesoForecastIngestionConfig | None = None


class FeatureSetConfig(BaseModel):
    """Configuration for one named feature set (Stage 3 onwards).

    A feature set is a reproducible recipe: which upstream parquets to join,
    how to resample demand to hourly cadence, how long a weather gap may be
    forward-filled before the row is dropped, and where to persist the
    resulting parquet. Stage 3 ships the ``weather_only`` set; Stage 5 adds
    ``weather_calendar`` alongside it so the with/without comparison is a
    config swap, not a code change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    demand_aggregation: Literal["mean", "max"] = "mean"
    cache_dir: Path
    cache_filename: str
    forward_fill_hours: int = Field(default=3, ge=0)


class FeaturesGroup(BaseModel):
    """Container for named feature-set configs.

    Each field is optional so stage-0/1/2 configs (no features section) still
    validate. Stage 5 will add ``weather_calendar: FeatureSetConfig | None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    weather_only: FeatureSetConfig | None = None


class SplitterConfig(BaseModel):
    """Configuration for the rolling-origin train/test splitter (Stage 3).

    ``min_train_periods`` fixes the minimum training window before the first
    test origin. ``test_len`` is the fold's test-window length (hours for the
    Stage 3 hourly feature table). ``step`` advances the origin by this many
    rows between folds. ``gap`` introduces an embargo between the end of
    training and the start of testing (zero is the day-ahead default; non-zero
    encodes a gate-closure-style discipline). ``fixed_window=True`` turns the
    default expanding window into a sliding window of size ``min_train_periods``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Minimum training rows before the first test origin — e.g. 8760 = one year of hourly data.
    min_train_periods: int = Field(ge=1)
    # Length of each fold's test window, in rows (e.g. 24 = one day of hourly data).
    test_len: int = Field(ge=1)
    # Row count the origin advances between folds (e.g. 24 = non-overlapping daily folds).
    step: int = Field(ge=1)
    gap: int = Field(default=0, ge=0)
    fixed_window: bool = False


class MetricsConfig(BaseModel):
    """Named metric functions the Stage 4 evaluator harness should compute.

    Names refer to the pure functions in ``bristol_ml.evaluation.metrics``
    (``mae``, ``mape``, ``rmse``, ``wape`` per DESIGN §5.3).  The default
    enumerates all four so the stdout metric table is complete out of the
    box; override (e.g. ``evaluation.metrics.names=[mae,rmse]``) to shorten
    the table for notebook fluency.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: tuple[Literal["mae", "mape", "rmse", "wape"], ...] = (
        "mae",
        "mape",
        "rmse",
        "wape",
    )


class NesoBenchmarkConfig(BaseModel):
    """Three-way benchmark comparison configuration (Stage 4).

    The NESO day-ahead forecast is published at half-hourly resolution; the
    Stage 3 feature table and this project's models run at hourly resolution.
    ``aggregation`` selects the hour-align rule; per D4 the default ``mean``
    preserves the MW unit and matches Stage 3 D1 (the assembler aggregates the
    ND outturn identically).  ``holdout_start`` / ``holdout_end`` bracket the
    test period over which the three-way metric table is computed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # D4 (plan §1): mean preserves MW scale; 'first' takes the first settlement
    # period of each hour (loses information but avoids averaging a forecasted
    # value).  Omit 'sum' — unit-wrong for an MW rate series.
    aggregation: Literal["mean", "first"] = "mean"
    holdout_start: datetime
    holdout_end: datetime

    @model_validator(mode="after")
    def _validate_holdout(self) -> NesoBenchmarkConfig:
        """Enforce ``holdout_start < holdout_end`` and tz-awareness (UTC-aware)."""
        if self.holdout_end <= self.holdout_start:
            raise ValueError(
                f"holdout_end {self.holdout_end} must strictly follow "
                f"holdout_start {self.holdout_start}."
            )
        for name, value in (
            ("holdout_start", self.holdout_start),
            ("holdout_end", self.holdout_end),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be tz-aware (UTC); got naive datetime {value!r}.")
        return self


class EvaluationGroup(BaseModel):
    """Container for evaluation-side configs.

    Each field is optional so pre-Stage-3 configs still validate.  Stage 3
    shipped ``rolling_origin``; Stage 4 adds ``metrics`` (named point-forecast
    metrics) and ``benchmark`` (three-way NESO comparison).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rolling_origin: SplitterConfig | None = None
    metrics: MetricsConfig | None = None
    benchmark: NesoBenchmarkConfig | None = None


class NaiveConfig(BaseModel):
    """Configuration for the seasonal-naive baseline model (Stage 4).

    The naive model is a look-up into training-time actuals.  ``strategy``
    chooses the lag: per D1 the default ``same_hour_last_week`` (``y_{t-168}``)
    captures the dominant weekly seasonality of GB electricity demand and is
    the credible-but-beatable floor against which the linear OLS must fight.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union; literal
    # value matches the Hydra group filename (``conf/model/naive.yaml``).
    type: Literal["naive"] = "naive"
    # D1 (plan §1): 'same_hour_last_week' (lag=168) is the default seasonal
    # naive definition; 'same_hour_yesterday' (lag=24) is easier to beat;
    # 'same_hour_same_weekday' picks the most recent matching (hour, weekday)
    # pair.  All three preserve the no-training-loop invariant of intent AC-2.
    strategy: Literal[
        "same_hour_yesterday",
        "same_hour_last_week",
        "same_hour_same_weekday",
    ] = "same_hour_last_week"
    # Target column on the Stage 3 feature table; default matches the assembler's
    # ``OUTPUT_SCHEMA`` name for national demand in MW.
    target_column: str = "nd_mw"


class LinearConfig(BaseModel):
    """Configuration for the linear (OLS) regression model (Stage 4).

    Per D2 the estimator is ``statsmodels.regression.linear_model.OLS``; sklearn
    is not a declared dependency.  ``feature_columns=None`` (the default) means
    "use every weather column from the Stage 3 ``assembler.OUTPUT_SCHEMA``";
    an explicit tuple narrows the regressor set for ablation experiments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union.
    type: Literal["linear"] = "linear"
    # Target column on the Stage 3 feature table.
    target_column: str = "nd_mw"
    # ``None`` means "all float32 weather columns from the assembler schema"
    # (enumerated at fit-time to stay in sync with the feature-table contract);
    # an explicit tuple narrows the regressor set.
    feature_columns: tuple[str, ...] | None = None
    # statsmodels' ``OLS`` does not add an intercept column automatically;
    # ``LinearModel.fit()`` calls ``sm.add_constant`` iff this is True.
    fit_intercept: bool = True


# ``AppConfig.model`` is a Pydantic discriminated union: exactly one of the
# model variants is active per run (matching Hydra group-override semantics).
# The ``type`` discriminator is written into the YAML by each Hydra group file.
ModelConfig = NaiveConfig | LinearConfig


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: ProjectConfig
    ingestion: IngestionGroup = Field(default_factory=IngestionGroup)
    features: FeaturesGroup = Field(default_factory=FeaturesGroup)
    evaluation: EvaluationGroup = Field(default_factory=EvaluationGroup)
    # ``None`` keeps pre-Stage-4 programmatic ``AppConfig(...)`` construction
    # valid (e.g. in ``tests/unit/features/test_assembler_cli.py``).  The
    # defaults list in ``conf/config.yaml`` always selects one variant when
    # composed via ``load_config()``.
    model: ModelConfig | None = Field(default=None, discriminator="type")
