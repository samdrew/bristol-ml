"""Pydantic schemas for the resolved application config.

Hydra composes YAML; this module validates the resolved tree. Adding a new
config group (ingestion, model, evaluation, ...) means adding a sub-model
here and a field on `AppConfig`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
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
    # Names are the CKAN-native casing as returned by
    # ``datastore_search`` on resource ``08e41551-80f8-4e28-a416-ea473a695db9``
    # (verified 2026-04 via the NESO API package-show metadata): the identity
    # columns ``Date`` / ``Settlement_Period`` / ``Datetime`` are always
    # fetched and converted; this list picks which measurement columns are
    # persisted alongside them.
    columns: list[str] = Field(
        default_factory=lambda: [
            "Demand_Forecast",
            "Demand_Outturn",
            "APE",
        ]
    )


class HolidaysIngestionConfig(BaseModel):
    """Configuration for the GB bank-holidays ingester (Stage 5).

    Source: ``https://www.gov.uk/bank-holidays.json``, published under the
    Open Government Licence v3.0.  The endpoint returns all three UK
    divisions (``england-and-wales``, ``scotland``, ``northern-ireland``)
    with complete coverage from 2012-01-02 onwards.  The ingester persists
    every division even though the Stage 5 feature derivation only encodes
    England & Wales and Scotland (see plan D-2): keeping the cache
    policy-agnostic means future regional work does not need to re-ingest.

    Retry / rate-limit fields mirror ``NesoIngestionConfig`` structurally so
    the shared helpers in ``bristol_ml.ingestion._common`` accept this config
    via their ``Protocol`` types.  ``min_inter_request_seconds`` defaults to
    zero because gov.uk publishes no documented rate limit and the endpoint
    is cheap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl = Field(default=HttpUrl("https://www.gov.uk/bank-holidays.json"))
    cache_dir: Path
    cache_filename: str = "holidays.parquet"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=10.0, gt=0)
    # gov.uk has no documented rate limit; set to zero so the retry loop
    # does not artificially throttle recorder runs.
    min_inter_request_seconds: float = Field(default=0.0, ge=0)
    # Divisions persisted to the parquet cache.  Defaults to all three so the
    # cache is policy-agnostic; the feature derivation (per plan D-2) only
    # encodes E&W and Scotland.  Override to a subset if a future stage wants
    # a narrower cache.
    divisions: tuple[
        Literal["england-and-wales", "scotland", "northern-ireland"],
        ...,
    ] = (
        "england-and-wales",
        "scotland",
        "northern-ireland",
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
    holidays: HolidaysIngestionConfig | None = None


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
    validate.  Stage 5 adds ``weather_calendar`` alongside ``weather_only``;
    per plan D-10 exactly one is populated per run — the ``features=`` Hydra
    group override (wired in ``conf/config.yaml`` defaults as ``- features:
    weather_only``) selects which file loads, and ``bristol_ml.train.
    _resolve_feature_set`` enforces the mutual-exclusivity invariant at
    runtime.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    weather_only: FeatureSetConfig | None = None
    weather_calendar: FeatureSetConfig | None = None


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


class PlotsConfig(BaseModel):
    """Configuration for the Stage 6 diagnostic-plot helper library.

    Settings consumed by ``bristol_ml.evaluation.plots``.  Every field matches
    the module-level default applied at import time; overriding a field here
    propagates through ``matplotlib.rcParams`` when ``plots`` is imported with
    a non-default config (the notebook Cell 11+ appendix reads from this group
    so facilitators can tweak figsize/DPI without touching Python code).

    **Field notes.**

    - ``figsize`` defaults to ``(12.0, 8.0)`` — the projector-legible default
      mandated by the Stage 6 plan D5 amendment (2026-04-20 human mandate).
      Wider than matplotlib's 6.4x4.8 baseline for meetup legibility (AC-2);
      taller than 10x6 so a 2x2 grid in the notebook does not squash.
    - ``dpi`` defaults to 110 — a middle ground between the 100-dpi default
      (blurry on HiDPI) and 150-dpi (PNG bloat on github.com).
    - ``display_tz`` defaults to ``"Europe/London"`` per D6.  T3 includes DST
      verification gates (spring-forward gap, fall-back duplicate hour); if
      those fail at implementation time the default swaps to ``"UTC"`` and the
      regression is documented in the Stage 6 retro.
    - ``acf_default_lags`` defaults to 168 per D7 — enough lag to cover both
      the daily (lag 24) and weekly (lag 168) periodicity markers annotated on
      the ACF plot.

    The palette (Okabe-Ito qualitative, ``cividis`` sequential, ``RdBu_r``
    diverging) is *not* configurable here — it lives as module constants in
    ``plots.py`` because changing it silently breaks colourblind-safety
    guarantees (D2).  Facilitators who want a bespoke palette override
    ``plt.rcParams`` directly in a notebook cell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # D5 (human mandate 2026-04-20): projector-friendly default, not 10x6.
    figsize: tuple[float, float] = (12.0, 8.0)
    # D5: matches the ``figure.dpi`` rcParam applied at import time.
    dpi: int = Field(default=110, ge=50, le=400)
    # D6: gated by the three DST verification tests in T3. ``"UTC"`` is the
    # documented fallback if the gates fail.
    display_tz: str = Field(default="Europe/London")
    # D7: 168 hourly lags = one full week; covers daily (24) and weekly (168)
    # reference markers on the ACF plot.
    acf_default_lags: int = Field(default=168, ge=1)


class EvaluationGroup(BaseModel):
    """Container for evaluation-side configs.

    Each field is optional so pre-Stage-3 configs still validate.  Stage 3
    shipped ``rolling_origin``; Stage 4 adds ``metrics`` (named point-forecast
    metrics) and ``benchmark`` (three-way NESO comparison); Stage 6 adds
    ``plots`` (diagnostic-plot defaults).

    ``plots`` is non-optional with a ``default_factory=PlotsConfig`` so that
    pre-Stage-6 configs (which lack an ``evaluation.plots`` section) still
    validate and downstream consumers always see a populated ``PlotsConfig``
    with the documented defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rolling_origin: SplitterConfig | None = None
    metrics: MetricsConfig | None = None
    benchmark: NesoBenchmarkConfig | None = None
    # Stage 6 D5: non-optional with Pydantic-populated defaults so construction
    # without a Hydra file (e.g. programmatic ``EvaluationGroup()``) yields a
    # ready-to-use ``PlotsConfig`` instance rather than ``None``.
    plots: PlotsConfig = Field(default_factory=PlotsConfig)


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


class SarimaxKwargs(BaseModel):
    """statsmodels ``SARIMAX`` constructor kwargs, pinned by Stage 7 D6.

    All five defaults are the *non-project-default* statsmodels values recommended
    for real-world seasonal demand data.  ``enforce_stationarity`` and
    ``enforce_invertibility`` are relaxed because the ML optimiser routinely
    finds non-stationary optima on hourly electricity series; statsmodels PR
    #4739 softened the same check at the starting-parameter stage for the same
    reason.  ``concentrate_scale=True`` removes sigma^2 from the parameter
    vector and materially speeds optimisation.  ``simple_differencing=False``
    keeps the Harvey representation so the full residual series reaches the
    Stage 6 ``acf_residuals`` helper without the first ``d + D*s`` observations
    being dropped.  ``hamilton_representation=False`` stays on the default
    Harvey form; only flip for Stata/R reproducibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    concentrate_scale: bool = True
    simple_differencing: bool = False
    hamilton_representation: bool = False


class SarimaxConfig(BaseModel):
    """Configuration for the SARIMAX model (Stage 7).

    Stage 7 D1 picks the dynamic-harmonic-regression (DHR) approach for the
    dual-seasonality problem: daily seasonality at ``s=24`` is handled inside
    the SARIMAX ``seasonal_order``; the weekly period (168 h) is absorbed by
    Fourier exogenous regressors (``weekly_fourier_harmonics`` sin/cos pairs).
    Setting ``weekly_fourier_harmonics=0`` disables the weekly Fourier path.

    Per D2 the default ``(p,d,q)(P,D,Q,s) = (1,0,1)(1,1,1,24)`` is the
    conservative textbook order from Hyndman *fpp3* §9 for hourly electricity
    demand; the notebook exhibits an AIC grid sweep that justifies the pick as
    a pedagogical exercise (not an architectural auto-search, which is out of
    scope per the intent).

    ``feature_columns=None`` means "use every non-target column from the input
    feature frame at fit time" (including the Stage 5 calendar one-hots when
    the weather-plus-calendar feature table is supplied); ``SarimaxModel.fit``
    then appends the weekly-Fourier columns on top.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union.
    type: Literal["sarimax"] = "sarimax"
    # Target column on the Stage 3 feature table.
    target_column: str = "nd_mw"
    # ``None`` means "every non-target column in the supplied feature frame";
    # an explicit tuple narrows the exogenous-regressor set for ablation runs.
    feature_columns: tuple[str, ...] | None = None
    # D2 (plan §1): (p, d, q) non-seasonal ARIMA order.
    order: tuple[int, int, int] = (1, 0, 1)
    # D2 (plan §1): (P, D, Q, s) seasonal order; ``s=24`` is the daily period.
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)
    # Optional statsmodels trend argument: ``"n"`` (no trend), ``"c"`` (constant),
    # ``"t"`` (linear), ``"ct"`` (both), or ``None`` (statsmodels default).  The
    # four-value ``Literal`` rejects typos like ``"linear"`` at config-load time
    # rather than at ``fit()`` time (Stage 7 Phase 3 review R2).
    trend: Literal["n", "c", "t", "ct"] | None = None
    # D1+D3 (plan §1): number of sin/cos Fourier harmonic pairs at the
    # 168-hour weekly period to append as exogenous regressors.  ``0`` disables
    # the weekly Fourier path (e.g. for ablation experiments or if using
    # ``s=168`` directly in ``seasonal_order``).
    weekly_fourier_harmonics: int = Field(default=3, ge=0, le=10)
    # D6 (plan §1): the SARIMAX constructor kwargs bundle.
    sarimax_kwargs: SarimaxKwargs = Field(default_factory=SarimaxKwargs)


# ``AppConfig.model`` is a Pydantic discriminated union: exactly one of the
# model variants is active per run (matching Hydra group-override semantics).
# The ``type`` discriminator is written into the YAML by each Hydra group file.
ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig


class ModelMetadata(BaseModel):
    """Immutable provenance record attached to a fitted model (Stage 4+).

    ``ModelMetadata`` is *not* a Hydra config group — it is a lightweight record
    carried as the ``.metadata`` property of every ``Model`` protocol implementor
    (see ``bristol_ml.models.protocol``).  It captures the minimum information
    needed to reason about a serialised artefact without re-loading the model's
    heavy state: which estimator produced it, which features it expects, when
    it was fit, and under which commit.

    The shape is deliberately minimal.  Per plan §10 risk register, Stage 9
    (registry) may extend this contract; the ``hyperparameters: dict[str, Any]``
    escape hatch absorbs future additions so the other fields can stay stable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Human-readable identifier; unique within a stage (e.g.
    # ``"naive-same-hour-last-week"``, ``"linear-ols-weather-only"``).
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    # The ordered tuple of feature-column names the model was fit on.  Stored
    # as a tuple (not a list) so the metadata stays hashable and immutable.
    feature_columns: tuple[str, ...]
    # UTC timestamp at which ``fit()`` completed; ``None`` before fitting.  The
    # ``model_validator`` below enforces tz-awareness when present.
    fit_utc: datetime | None = None
    # Short Git SHA recorded at fit time; ``None`` when the fit happens outside
    # a Git working tree (e.g. pip-installed wheels).
    git_sha: str | None = None
    # Free-form bag for estimator-specific state (coefficients, R², strategy).
    # ``dict[str, Any]`` is unavoidable here — the whole point is extensibility
    # without ABI churn on this schema.
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fit_utc(self) -> ModelMetadata:
        """Reject naive ``fit_utc`` values: we only store tz-aware UTC times."""
        if self.fit_utc is not None and self.fit_utc.tzinfo is None:
            raise ValueError(
                f"fit_utc must be tz-aware (UTC); got naive datetime {self.fit_utc!r}."
            )
        return self


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
