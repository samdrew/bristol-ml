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


class RemitIngestionConfig(BaseModel):
    """Configuration for the Elexon REMIT ingester (Stage 13).

    Source: the Elexon Insights API at ``https://data.elexon.co.uk/bmrs/api/v1/``,
    the public unauthenticated successor to the decommissioned BMRS API
    (per Stage 13 domain research §R1).  Stage 13 reads the bulk-streaming
    endpoint ``GET /datasets/REMIT/stream`` (no observed window cap, unlike
    ``GET /datasets/REMIT`` which is capped at 24 hours per call).

    REMIT messages are append-only: every revision is preserved on disk so
    the bi-temporal "what did the market know at time T?" query is correct
    over historical points.  The schema keeps four UTC-aware timestamp axes
    on every row: ``published_at`` (transaction-time), ``effective_from`` /
    ``effective_to`` (valid-time; ``effective_to`` nullable for open-ended
    events), ``retrieved_at_utc`` (project-axis provenance per row).

    Retry / rate-limit / cache fields mirror ``NesoIngestionConfig`` and
    ``HolidaysIngestionConfig`` structurally so the shared helpers in
    ``bristol_ml.ingestion._common`` accept this config via their ``Protocol``
    types — same seven structural fields (``max_attempts``,
    ``backoff_base_seconds``, ``backoff_cap_seconds``,
    ``request_timeout_seconds``, ``min_inter_request_seconds``, ``cache_dir``,
    ``cache_filename``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: HttpUrl = Field(
        default=HttpUrl("https://data.elexon.co.uk/bmrs/api/v1/"),
    )
    endpoint_path: str = Field(default="datasets/REMIT/stream")
    # Default window matches the demand-model training window referenced in
    # the intent §Points; backfill via Hydra CLI override.
    window_start: date = Field(default=date(2018, 1, 1))
    window_end: date | None = Field(default=None)

    cache_dir: Path
    cache_filename: str = Field(default="remit.parquet")
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=5, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_cap_seconds: float = Field(default=30.0, gt=0)
    # Polite default for the public Insights API; can be raised on slow
    # networks via Hydra override.
    min_inter_request_seconds: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def _check_window_order(self) -> RemitIngestionConfig:
        if self.window_end is not None and self.window_end < self.window_start:
            raise ValueError(
                f"window_end ({self.window_end}) must be on or after "
                f"window_start ({self.window_start})"
            )
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
    neso_forecast: NesoForecastIngestionConfig | None = None
    holidays: HolidaysIngestionConfig | None = None
    remit: RemitIngestionConfig | None = None


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


class NnMlpConfig(BaseModel):
    """Configuration for the small-MLP neural network model (Stage 10).

    Stage 10 ships a small multi-layer perceptron that conforms to the Stage 4
    :class:`~bristol_ml.models.protocol.Model` protocol.  The architecture is
    deliberately minimal — 1 hidden layer x 128 units with ReLU activation by
    default (plan D3) — because the stage's load-bearing contribution is the
    training-loop + reproducibility + registry-round-trip **scaffold** that
    Stage 11's temporal architecture inherits, not the analytical value of the
    MLP itself (intent §Purpose).

    A brief tour of the knobs:

    - ``hidden_sizes`` is an ordered list: ``[128]`` = one layer of width 128;
      ``[128, 64]`` = two hidden layers.  The intent caps this at "moderate
      width"; architectures above ~100k parameters are out of the Stage 10
      budget (plan §2 Out of scope).
    - ``dropout`` is applied after every hidden ReLU; ``0.0`` (default)
      disables it.  The validator uses ``lt=1.0`` not ``le`` because
      ``dropout=1.0`` would zero the entire hidden activation and crash the
      loss.
    - ``feature_columns=None`` means "use the harness-supplied feature set" —
      same harness-fallback idiom as :class:`SarimaxConfig` and
      :class:`LinearConfig`; the train-CLI promotes the resolved
      feature-set tuple into a copy of the config so the stored metadata is
      faithful (plan D2 + the SARIMAX / linear precedent).  Unlike
      :class:`ScipyParametricConfig`, ``feature_columns`` here names *raw*
      input columns from the feature table, not a subset of generated
      Fourier columns.
    - ``seed=None`` means "derive per-fold seed from ``config.project.seed +
      fold_index``" (plan D8 cold-start per-fold determinism).  Passing an
      explicit integer pins the seed and makes per-fold runs reproducible in
      isolation.
    - ``device`` selects CUDA / MPS / CPU at ``fit`` time via the
      ``_select_device`` helper in ``mlp.py``; the resolution order when
      ``"auto"`` is CUDA > MPS > CPU (plan D11 — re-opened at the 2026-04-24
      Ctrl+G that added CUDA-aware install).  Pin to ``"cpu"`` in tests that
      need bit-identical CPU reproducibility (NFR-1 CPU path); pin to ``"cuda"``
      to force GPU use when both CUDA and MPS are present on the host.

    Design decisions and their evidence:

    - Default ``[128]`` + ReLU + Adam(1e-3) + max_epochs=100 + patience=10:
      plan D3 (tabular-NN baselines; domain research §R6).
    - ``weight_decay=0.0`` default: gradient regularisation knobs are
      configurable for experiments but Stage 10 default ships off (scope diff
      X6 cuts LR scheduling / gradient clipping; plan keeps `weight_decay`
      available because it is a zero-extra-code field on ``torch.optim.Adam``).
    - ``patience=10`` default: plan D9 (patience-based early stopping with
      best-epoch restore; 10 epochs ≈ 10 % of the default 100-epoch budget).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union; literal
    # value matches the Hydra group filename (``conf/model/nn_mlp.yaml``).
    type: Literal["nn_mlp"] = "nn_mlp"
    # Target column on the Stage 3 / Stage 5 feature table.
    target_column: str = "nd_mw"
    # ``None`` means "use the harness-supplied raw feature set" (same pattern
    # as Linear / SARIMAX).  An explicit tuple narrows the regressor set for
    # ablation experiments.
    feature_columns: tuple[str, ...] | None = None

    # --- Architecture (plan D3) ---------------------------------------------
    # Ordered list of hidden-layer widths.  Default ``[128]`` = one hidden
    # layer of width 128.  Length-2 lists like ``[128, 64]`` produce a
    # two-hidden-layer MLP; intent §Points ("one or two hidden layers,
    # moderate width") bounds the pedagogical envelope.
    hidden_sizes: list[int] = Field(default_factory=lambda: [128])
    # ReLU is the tabular-NN default.  ``tanh`` and ``gelu`` exposed for
    # notebook experimentation without code changes.
    activation: Literal["relu", "tanh", "gelu"] = "relu"
    # Dropout probability applied after every hidden-layer activation.
    # ``0.0`` disables dropout; ``lt=1.0`` avoids the degenerate "zero the
    # entire layer" edge case that would crash the loss.
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # --- Optimisation (plan D3) ---------------------------------------------
    # Adam learning rate; 1e-3 is the published default across tabular-NN
    # baselines (domain research §R6).
    learning_rate: float = Field(default=1e-3, gt=0)
    # L2 regularisation strength on Adam.  Default off because Stage 10
    # ships the simplest defensible MLP; override for experiments.
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    # Patience-based early stopping: halt when val loss has not improved for
    # ``patience`` consecutive epochs and restore best-epoch weights (D9).
    patience: int = Field(default=10, ge=1)

    # --- Reproducibility (plan D7') ----------------------------------------
    # ``None`` defers to ``config.project.seed + fold_index`` inside ``fit``
    # (plan D8).  Pin an explicit integer to make per-fold runs independently
    # reproducible (handy in notebook experimentation).
    seed: int | None = None

    # --- Device (plan D11, re-opened at 2026-04-24 Ctrl+G) ------------------
    # Auto-select CUDA > MPS > CPU unless pinned.  The resolved device is
    # logged at INFO inside ``fit`` and persisted in the sidecar (D5) as
    # provenance.  Distributed / multi-GPU remain out of scope (intent §Out of
    # scope).
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class ScipyParametricConfig(BaseModel):
    """Configuration for the SciPy parametric load model (Stage 8).

    Stage 8 D1 picks a piecewise-linear double-hinge temperature response
    (heating and cooling degree-day style, ``max(0, T_heat - T)`` and
    ``max(0, T - T_cool)``) with Elexon-convention fixed hinge temperatures
    (``T_heat = 15.5``, ``T_cool = 22.0`` °C).  On top of the two hinge
    coefficients sit a base-load intercept ``alpha`` plus diurnal and weekly
    Fourier harmonic pairs (Stage 8 D2, defaults ``diurnal=3``, ``weekly=2``).

    ``feature_columns=None`` is **deliberately narrower here than in
    SarimaxConfig**: the parametric model's design matrix is always just the
    temperature column + the generated Fourier columns (D2 clarification —
    Stage 5 day-of-week one-hots are excluded to avoid partial collinearity
    with the weekly Fourier terms).  The field is retained for override-time
    ablation experiments ("drop the weekly harmonics", "fit on a subset of
    Fourier columns") but the default resolution narrows to the
    temperature-plus-Fourier column set.

    ``loss="linear"`` (plan D3) keeps ``curve_fit``'s Gaussian-pcov → CI
    mapping rigorous; the other loss choices (``soft_l1`` / ``huber`` /
    ``cauchy``) are available as CLI overrides but turn ``pcov`` into a
    heuristic.  ``p0=None`` (plan D4) triggers deterministic data-driven
    initialisation inside ``fit()``; passing an explicit tuple pins the
    starting point (useful for reproducibility experiments).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union.
    type: Literal["scipy_parametric"] = "scipy_parametric"
    # Target column on the Stage 3 feature table.
    target_column: str = "nd_mw"
    # ``None`` + parametric design-matrix rule → only ``temperature_column`` +
    # generated Fourier columns.  An explicit tuple narrows further.
    feature_columns: tuple[str, ...] | None = None
    # The temperature feature used for HDD/CDD computation; override if the
    # feature table uses a different name (e.g. ``temp_c``).
    temperature_column: str = "temperature_2m"
    # D2 (plan §1): number of sin/cos harmonic pairs at the 24-hour diurnal
    # cycle.  ``0`` disables the diurnal component entirely (ablation).
    diurnal_harmonics: int = Field(default=3, ge=0, le=10)
    # D2 (plan §1): number of sin/cos harmonic pairs at the 168-hour weekly
    # cycle.  ``0`` disables the weekly component entirely (ablation).
    weekly_harmonics: int = Field(default=2, ge=0, le=10)
    # D1 (plan §1): heating-degree-day base temperature (°C).  Elexon
    # convention = 15.5.  Fixed by D1, not a free parameter of curve_fit.
    t_heat_celsius: float = 15.5
    # D1 (plan §1): cooling-degree-day base temperature (°C).  Elexon
    # convention = 22.0.  Fixed by D1, not a free parameter of curve_fit.
    t_cool_celsius: float = 22.0
    # D3 (plan §1): robust-loss selector passed through to
    # ``scipy.optimize.least_squares`` when the user opts into a non-default
    # loss.  ``"linear"`` keeps the standard pcov → Gaussian-CI mapping;
    # ``"soft_l1"`` / ``"huber"`` / ``"cauchy"`` downweight outliers but turn
    # the CI into a heuristic (see plan D5 notebook appendix).
    loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear"
    # D6 (plan §1): maximum ``curve_fit`` function evaluations.  5000 is an
    # order of magnitude above the convergence envelope for 13 parameters
    # x 8760 rows; raise only if a pathological fold bounces off the cap.
    max_iter: int = Field(default=5000, ge=1)
    # D4 (plan §1): optional explicit starting point; ``None`` triggers the
    # data-driven derivation inside ``fit()``.  Length must match the
    # parameter count ``3 + 2*diurnal_harmonics + 2*weekly_harmonics``.
    p0: tuple[float, ...] | None = None


class NnTemporalConfig(BaseModel):
    """Configuration for the temporal convolutional network model (Stage 11).

    Stage 11 ships a Bai-et-al.-2018-style Temporal Convolutional Network
    (TCN) — dilated causal 1D convolutions stacked into residual blocks —
    conforming to the Stage 4 :class:`~bristol_ml.models.protocol.Model`
    protocol.  The family is the second torch-backed model after Stage 10's
    MLP; together they fire the D10 extraction seam flagged in Stage 10
    (the shared training loop now lives in
    :mod:`bristol_ml.models.nn._training`).

    Defaults target the Blackwell dev host (CUDA 12.8 / cu128 wheels, per
    Stage 10 D1) — materially larger than the MLP so the neural family
    gets a fair shot at beating the classical / parametric baselines on
    the ablation table.  Every architecture and training knob is exposed
    at the YAML / Hydra layer so a CPU-only facilitator can tract the
    demo via CLI overrides; the recommended CPU recipe rides in the
    ``conf/model/nn_temporal.yaml`` header comment.

    A brief tour of the knobs:

    - ``seq_len`` is the number of historical hours the model sees for
      each prediction.  ``168`` is the weekly cycle anchor (plan D2;
      UniLF 2025 on hourly STLF).  A Pydantic ``@model_validator`` rejects
      configurations where ``seq_len`` is small relative to the
      architecture's receptive field — the heuristic is loose
      (``seq_len >= max(2*kernel_size, receptive_field // 8)``) but catches
      the common footgun of keeping the 8-block CUDA defaults while
      dialling ``seq_len`` down to 24.
    - ``num_blocks`` x ``channels`` x ``kernel_size`` shape the TCN body.
      Dilations double per block (``[1, 2, 4, ..., 2**(num_blocks-1)]``) so
      the receptive field grows exponentially with ``num_blocks``
      (closed-form: ``1 + 2*(kernel_size-1)*(2**num_blocks - 1)``).  At
      defaults this is 1021 steps — ~6x the weekly cycle, covering the
      intent's "same hour last week" pedagogical anchor with ample
      headroom.
    - ``weight_norm`` wraps every ``Conv1d`` via
      :func:`torch.nn.utils.parametrizations.weight_norm` — the modern
      parametrizations API.  ``state_dict`` keys round-trip as
      ``...parametrizations.weight.original0`` /
      ``...parametrizations.weight.original1`` (not the legacy
      ``weight_g`` / ``weight_v``); the load path uses
      ``strict=True`` so a saved artefact and a freshly-built skeleton
      either agree exactly or raise.  ``temporal.py`` keeps a one-shot
      fallback to the legacy ``torch.nn.utils.weight_norm`` for
      ``torch < 2.1`` but the project pin is ``torch>=2.7`` so the
      parametrizations path is always taken in practice.
    - ``dropout`` is applied once per residual block after the second
      conv.  ``0.2`` is the capacity-regularisation trade at the defaults.
    - ``feature_columns=None`` means "use the harness-supplied raw feature
      set" — same idiom as :class:`NnMlpConfig`, :class:`LinearConfig`,
      :class:`SarimaxConfig`.  Pattern A exogenous handling (plan D3): the
      dataset yields sequences of shape ``(seq_len, n_features)`` where
      ``n_features`` is the full column count; there is no separate
      known-future side branch.
    - ``seed=None`` means "derive per-fold seed from
      ``config.project.seed + fold_index``" — inherited from Stage 10 D8
      cold-start-per-fold determinism.
    - ``device=auto`` resolves CUDA > MPS > CPU at fit time through the
      same ``_select_device`` helper ``NnMlpModel`` uses.  Pin to
      ``"cpu"`` in tests that need bit-identical reproducibility
      (NFR-1 CPU path) or for the documented CPU-recipe demo override.

    Design decisions and their evidence:

    - Default ``num_blocks=8``, ``channels=128``, ``kernel_size=3``,
      ``dropout=0.2``: plan D1 (CUDA-targeted; domain research §1-3 TCN
      family table and receptive-field mathematics).
    - Default ``batch_size=256``, ``max_epochs=100``, ``patience=10``,
      ``learning_rate=1e-3``: plan D1 (CUDA-sized training budget) +
      Stage 10 D3 / D9 precedent.
    - ``seq_len=168``: plan D2 (UniLF 2025 on hourly STLF; weekly cycle
      anchor).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Discriminator tag for the ``AppConfig.model`` tagged union; literal
    # value matches the Hydra group filename (``conf/model/nn_temporal.yaml``).
    type: Literal["nn_temporal"] = "nn_temporal"
    # Target column on the Stage 3 / Stage 5 feature table.
    target_column: str = "nd_mw"
    # ``None`` means "use the harness-supplied raw feature set"; same
    # pattern as Linear / SARIMAX / NnMlp.  Pattern A exogenous handling
    # (plan D3) — all listed columns ride inside the sequence window.
    feature_columns: tuple[str, ...] | None = None

    # --- Architecture (plan D1 — CUDA defaults) -----------------------------
    # Historical context window length in hours.  ``168`` anchors the
    # weekly cycle (plan D2).  ``ge=2`` because a 1-step window degenerates
    # to the flat feature-row baseline and there is a separate model
    # family for that.
    seq_len: int = Field(default=168, ge=2)
    # Number of residual TCN blocks.  Dilations double per block;
    # receptive field grows as ``1 + 2*(kernel_size-1)*(2**num_blocks - 1)``.
    # ``le=12`` because beyond that the receptive field exceeds every
    # reasonable ``seq_len`` and the model is wasting capacity on
    # unreachable history.
    num_blocks: int = Field(default=8, ge=1, le=12)
    # Channel width of every Conv1d layer.  ``le=512`` is the pragmatic
    # upper bound for a single-GPU run at ``batch_size=256`` on a 24 GB
    # card; ``ge=8`` is the practical lower bound for the pedagogical
    # CPU recipe.
    channels: int = Field(default=128, ge=8, le=512)
    # Convolution kernel size.  ``3`` is the Bai-et-al.-2018 default and
    # balances receptive-field growth against parameter count.
    kernel_size: int = Field(default=3, ge=2, le=7)
    # Dropout after the second conv in each residual block.  Clamped at
    # ``lt=1.0`` to avoid the degenerate "zero every activation" case.
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    # Apply weight-norm reparametrisation to every Conv1d via
    # :func:`torch.nn.utils.parametrizations.weight_norm` (modern API on
    # the pinned ``torch>=2.7``; ``temporal.py`` keeps a fallback to the
    # legacy ``torch.nn.utils.weight_norm`` for ``torch<2.1`` but it is
    # never exercised under the project's pinned floor).
    weight_norm: bool = True

    # --- Optimisation (plan D1 — CUDA defaults) ----------------------------
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)

    # --- Reproducibility (plan D6 — inherits Stage 10 D7') -----------------
    # ``None`` defers to ``config.project.seed + fold_index`` inside
    # ``fit`` (Stage 10 D8 cold-start).  Pin an explicit integer to make
    # per-fold runs independently reproducible.
    seed: int | None = None

    # --- Device (plan D6 — inherits Stage 10 D11) --------------------------
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    @model_validator(mode="after")
    def _seq_len_covers_receptive_field(self) -> NnTemporalConfig:
        """Reject configurations whose ``seq_len`` is too small for the architecture.

        The receptive-field heuristic is intentionally loose: the rule is
        ``seq_len >= max(2 * kernel_size, receptive_field // 8)``.  A
        facilitator who wants ``seq_len=24`` can simultaneously drop
        ``num_blocks`` to 3 (receptive field collapses accordingly); both
        knobs are exposed.  The validator exists to prevent the silent
        footgun of keeping 8-block CUDA defaults while dialling only
        ``seq_len`` down — the model would be architecturally able to see
        1021 steps of history but receive a 24-step window, wasting its
        capacity on padding.  Plan D2 + R6.
        """
        receptive = 1 + 2 * (self.kernel_size - 1) * (2**self.num_blocks - 1)
        minimum = max(2 * self.kernel_size, receptive // 8)
        if self.seq_len < minimum:
            raise ValueError(
                f"NnTemporalConfig.seq_len={self.seq_len} is too small for the "
                f"requested architecture (num_blocks={self.num_blocks}, "
                f"kernel_size={self.kernel_size} → receptive field ~{receptive}); "
                f"require seq_len >= {minimum}.  Either raise seq_len or reduce "
                f"num_blocks / kernel_size."
            )
        return self


# ``AppConfig.model`` is a Pydantic discriminated union: exactly one of the
# model variants is active per run (matching Hydra group-override semantics).
# The ``type`` discriminator is written into the YAML by each Hydra group file.
ModelConfig = (
    NaiveConfig
    | LinearConfig
    | SarimaxConfig
    | ScipyParametricConfig
    | NnMlpConfig
    | NnTemporalConfig
)


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


class ServingConfig(BaseModel):
    """Configuration for the Stage 12 serving layer.

    Carries the three settings the standalone CLI launcher
    (``python -m bristol_ml.serving``) needs to bring up the FastAPI
    app: the on-disk registry root the lifespan reads at startup, and
    the host/port the uvicorn process binds to.  All three have
    sensible localhost-only defaults so a fresh clone serves out of
    the box once a model is registered.

    The shape is plan §5 verbatim: per Stage 12 D13 (DESIGN §2.1.1) the
    launcher is a thin ``argparse + uvicorn.run(...)`` wrapper, not a
    Hydra-decorated entry point — Hydra resolves this config via
    :func:`bristol_ml.config.load_config` and the launcher hands the
    validated values to uvicorn.

    The host/port defaults are deliberately localhost-only (``127.0.0.1``
    rather than ``0.0.0.0``); intent §Out of scope explicitly defers
    "deployment anywhere other than localhost".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Filesystem root the registry layer reads via
    # :func:`bristol_ml.registry.list_runs` at lifespan startup.  The
    # default mirrors the project-wide convention seen elsewhere in
    # ``conf/`` (Stage 9 ``registry/`` lives at ``data/registry``).
    registry_dir: Path = Path("data/registry")
    # Localhost only by default; deployment beyond localhost is explicit
    # intent §Out of scope for Stage 12.
    host: str = "127.0.0.1"
    # Port range guarded by Pydantic so a bad override fails at config
    # validation rather than at uvicorn bind time.
    port: int = Field(default=8000, ge=1, le=65535)


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
    # Stage 12: ``None`` keeps the train CLI / Stage-0 config-smoke surface
    # unchanged — only the serving CLI ever requires this group, and it
    # composes ``serving`` into the defaults list at its own entry point.
    # The default ``conf/config.yaml`` for the train pipeline does *not*
    # include ``serving``, so ``cfg.serving`` resolves to ``None`` for
    # callers that do not need it.
    serving: ServingConfig | None = None
