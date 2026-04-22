"""Stage 0 + Stage 3 T1 + Stage 6 T1 + Stage 7 T1 smoke/acceptance tests for the config pipeline.

Stage 0 tests cover the fundamental `load_config()` → `AppConfig` round-trip
and the `python -m bristol_ml` demo moment.

Stage 3 T1 tests encode the acceptance criteria for the new Hydra config groups
(`conf/features/weather_only.yaml`, `conf/evaluation/rolling_origin.yaml`) and
the four new Pydantic models (`FeatureSetConfig`, `FeaturesGroup`,
`SplitterConfig`, `EvaluationGroup`).  Each test docstring cites the plan
decision or acceptance criterion it guards so future readers can trace back to
`docs/plans/completed/03-feature-assembler.md`.

Stage 7 T1 tests guard `SarimaxConfig` and `SarimaxKwargs` schema correctness,
the discriminated-union dispatch on ``type: "sarimax"``, and the full Hydra
round-trip including the D4 splitter per-field overrides (plan §6 T1).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bristol_ml import load_config
from conf._schemas import (
    AppConfig,
    EvaluationGroup,
    FeatureSetConfig,
    FeaturesGroup,
    HolidaysIngestionConfig,
    IngestionGroup,
    LinearConfig,
    MetricsConfig,
    ModelMetadata,
    NaiveConfig,
    NesoBenchmarkConfig,
    NesoForecastIngestionConfig,
    PlotsConfig,
    ProjectConfig,
    SarimaxConfig,
    SarimaxKwargs,
    ScipyParametricConfig,
    SplitterConfig,
)

# ---------------------------------------------------------------------------
# Stage 0 — baseline smoke tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 3 T1 — positive (behavioural) cases
# ---------------------------------------------------------------------------


def test_app_config_populates_features_weather_only_from_defaults() -> None:
    """Guards AC-6: AppConfig fully populated from config.yaml defaults (T1).

    The ``conf/features/weather_only.yaml`` group (D3 name, D1 aggregation,
    D5 forward-fill, D6 cache location) must be composed into
    ``AppConfig.features.weather_only`` without any override.
    """
    cfg = load_config()

    assert cfg.features is not None, "AppConfig.features must not be None after Stage 3 T1."
    fset = cfg.features.weather_only
    assert fset is not None, (
        "cfg.features.weather_only must be populated from the defaults list "
        "in conf/config.yaml after Stage 3 T1."
    )
    assert isinstance(fset, FeatureSetConfig)

    # D3: feature-set name is 'weather_only'.
    assert fset.name == "weather_only", (
        f"D3: feature-set name must be 'weather_only'; got {fset.name!r}."
    )
    # D1: default aggregation function is 'mean'.
    assert fset.demand_aggregation == "mean", (
        f"D1: demand_aggregation default must be 'mean'; got {fset.demand_aggregation!r}."
    )
    # D5: forward-fill cap is 3 hours by default.
    assert fset.forward_fill_hours == 3, (
        f"D5: forward_fill_hours default must be 3; got {fset.forward_fill_hours!r}."
    )
    # D6: cache_filename is set.
    assert fset.cache_filename == "weather_only.parquet", (
        f"D6: cache_filename must be 'weather_only.parquet'; got {fset.cache_filename!r}."
    )
    # D6: cache_dir resolves to a Path (env-var interpolation: data/features fallback).
    assert isinstance(fset.cache_dir, Path), (
        f"D6: cache_dir must resolve to a Path; got {type(fset.cache_dir).__name__!r}."
    )


def test_app_config_populates_evaluation_rolling_origin_from_defaults() -> None:
    """Guards AC-6: AppConfig fully populated from config.yaml defaults (T1).

    The ``conf/evaluation/rolling_origin.yaml`` group (D4 expanding window,
    1-year minimum training period, 24-hour day-ahead test window) must be
    composed into ``AppConfig.evaluation.rolling_origin`` without any override.
    """
    cfg = load_config()

    assert cfg.evaluation is not None, "AppConfig.evaluation must not be None after Stage 3 T1."
    splitter = cfg.evaluation.rolling_origin
    assert splitter is not None, (
        "cfg.evaluation.rolling_origin must be populated from the defaults list "
        "in conf/config.yaml after Stage 3 T1."
    )
    assert isinstance(splitter, SplitterConfig)

    # D4: one full seasonal cycle before the first test origin.
    assert splitter.min_train_periods == 8760, (
        f"D4: min_train_periods must be 8760 (one year of hourly data); "
        f"got {splitter.min_train_periods!r}."
    )
    # Day-ahead horizon (DESIGN §5.1).
    assert splitter.test_len == 24, f"test_len must be 24 (1 day); got {splitter.test_len!r}."
    # 24-hour step => non-overlapping daily folds.
    assert splitter.step == 24, (
        f"step must be 24 (non-overlapping daily folds); got {splitter.step!r}."
    )
    # D4: gate-closure gap zero for historical training.
    assert splitter.gap == 0, f"D4: gap must be 0 for historical training; got {splitter.gap!r}."
    # D4: expanding window is the default.
    assert splitter.fixed_window is False, (
        f"D4: fixed_window must default to False (expanding window); got {splitter.fixed_window!r}."
    )


def test_feature_set_config_accepts_override_demand_aggregation_max() -> None:
    """Guards D1: the Literal accepts 'max' as a valid override via Hydra.

    Per D1: 'mean' is the default; 'max' is the only other valid value.
    A CLI override ``features.weather_only.demand_aggregation=max`` must
    produce a validated ``FeatureSetConfig`` with ``demand_aggregation == 'max'``.
    """
    cfg = load_config(overrides=["features.weather_only.demand_aggregation=max"])

    assert cfg.features.weather_only is not None
    assert cfg.features.weather_only.demand_aggregation == "max", (
        "D1: override to 'max' must validate and be reflected in the resolved config."
    )


# ---------------------------------------------------------------------------
# Stage 3 T1 — negative (validator) cases
# ---------------------------------------------------------------------------


def test_feature_set_config_rejects_unknown_aggregation() -> None:
    """Guards D1 Literal narrowing: values outside {'mean', 'max'} must be rejected.

    The ``Literal["mean", "max"]`` annotation on ``FeatureSetConfig.demand_aggregation``
    must cause Pydantic to raise ``ValidationError`` when any other string
    (e.g. 'median', 'sum') is supplied.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["features.weather_only.demand_aggregation=median"])


def test_splitter_config_rejects_non_positive_test_len() -> None:
    """Guards SplitterConfig field constraint: test_len must be >= 1.

    A rolling-origin splitter with a zero-length test window is undefined
    (no rows to evaluate against). ``Field(ge=1)`` must raise.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["evaluation.rolling_origin.test_len=0"])


def test_splitter_config_rejects_non_positive_step() -> None:
    """Guards SplitterConfig field constraint: step must be >= 1.

    A step of zero would produce an infinite loop of identical folds.
    ``Field(ge=1)`` must raise.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["evaluation.rolling_origin.step=0"])


def test_splitter_config_rejects_negative_gap() -> None:
    """Guards SplitterConfig field constraint: gap must be >= 0.

    A negative gap is physically meaningless (test cannot precede training).
    ``Field(ge=0)`` must raise for gap=-1.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["evaluation.rolling_origin.gap=-1"])


def test_feature_set_config_rejects_negative_forward_fill_hours() -> None:
    """Guards FeatureSetConfig field constraint: forward_fill_hours must be >= 0.

    A negative forward-fill cap is meaningless. ``Field(ge=0)`` must raise.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["features.weather_only.forward_fill_hours=-1"])


def test_feature_set_config_rejects_bad_name_pattern() -> None:
    """Guards FeatureSetConfig.name regex: pattern is ``^[a-z][a-z0-9_]*$``.

    A name starting with an uppercase letter (e.g. 'Weather_Only') must be
    rejected by the ``Field(pattern=...)`` validator.  This mirrors the same
    constraint on ``WeatherStation.name`` and ``ProjectConfig.name``, keeping
    the naming convention machine-enforceable across all config layers.
    """
    with pytest.raises(ValidationError):
        load_config(overrides=["features.weather_only.name=Weather_Only"])


def test_features_group_forbids_extra_keys() -> None:
    """Guards ConfigDict(extra='forbid') on FeaturesGroup (and by extension all group models).

    Adding an unknown key to the features group (e.g. via ``+features.nonsense=1``)
    must be rejected.  This prevents silent config drift where a mis-typed key
    is silently ignored rather than raising.
    """
    with pytest.raises((ValidationError, Exception)):
        # ``+`` in Hydra appends a new key; because the schema forbids extras
        # this should fail at the Pydantic validation step.
        load_config(overrides=["+features.nonsense=1"])


# ---------------------------------------------------------------------------
# Stage 3 T1 — direct Pydantic model smoke tests (no Hydra round-trip)
# ---------------------------------------------------------------------------


def test_feature_set_config_direct_construction_smoke() -> None:
    """Smoke test: FeatureSetConfig constructs correctly with required fields.

    Guards AC-2: output conforms to the declared schema.  This exercises
    the model in isolation (without Hydra), verifying defaults and frozen
    immutability.
    """
    fset = FeatureSetConfig(
        name="test_set",
        cache_dir=Path("/tmp/features"),
        cache_filename="test_set.parquet",
    )
    assert fset.name == "test_set"
    assert fset.demand_aggregation == "mean"
    assert fset.forward_fill_hours == 3
    assert fset.cache_dir == Path("/tmp/features")

    # ConfigDict(frozen=True): mutation must raise.
    with pytest.raises((ValidationError, TypeError)):
        fset.name = "mutated"  # type: ignore[misc]  # testing frozen enforcement


def test_splitter_config_direct_construction_smoke() -> None:
    """Smoke test: SplitterConfig constructs correctly with all required fields.

    Guards AC-2: schema is sound.  Exercises the model in isolation and
    asserts the ``fixed_window`` default.
    """
    sc = SplitterConfig(min_train_periods=100, test_len=24, step=24)
    assert sc.min_train_periods == 100
    assert sc.test_len == 24
    assert sc.step == 24
    assert sc.gap == 0
    assert sc.fixed_window is False

    # ConfigDict(frozen=True): mutation must raise.
    with pytest.raises((ValidationError, TypeError)):
        sc.test_len = 48  # type: ignore[misc]  # testing frozen enforcement


def test_features_group_defaults_to_none_fields() -> None:
    """Guards FeaturesGroup optional fields: pre-Stage-3 configs still validate.

    ``FeaturesGroup()`` with no arguments must produce an instance whose
    ``weather_only`` field is ``None``.  This keeps Stage 0/1/2 config files
    (which lack a ``features:`` section) valid.
    """
    fg = FeaturesGroup()
    assert fg.weather_only is None


def test_evaluation_group_defaults_to_none_fields() -> None:
    """Guards EvaluationGroup optional fields: pre-Stage-3 configs still validate.

    ``EvaluationGroup()`` with no arguments must produce an instance whose
    ``rolling_origin`` field is ``None``.  This keeps Stage 0/1/2 config files
    (which lack an ``evaluation:`` section) valid.
    """
    eg = EvaluationGroup()
    assert eg.rolling_origin is None


# ---------------------------------------------------------------------------
# Stage 4 T1 — config group tests
# ---------------------------------------------------------------------------


def test_app_config_populates_model_linear_from_defaults() -> None:
    """Guards AC-6 / F-9: ``conf/config.yaml`` defaults list selects ``model: linear``.

    The ``- model: linear`` entry in ``conf/config.yaml`` must compose
    ``conf/model/linear.yaml`` into ``AppConfig.model`` without any CLI override.
    After defaults-only ``load_config()`` the resolved object must be a
    ``LinearConfig`` with the plan D2-mandated defaults: ``type="linear"``,
    ``target_column="nd_mw"``, ``feature_columns is None`` (meaning "all weather
    columns from the assembler schema"), and ``fit_intercept=True``.
    """
    cfg = load_config()

    assert cfg.model is not None, "AppConfig.model must not be None after defaults-only load."
    assert isinstance(cfg.model, LinearConfig), (
        f"AC-6/F-9: expected LinearConfig from defaults; got {type(cfg.model).__name__!r}."
    )
    assert cfg.model.type == "linear", (
        f"AC-6: discriminator tag must be 'linear'; got {cfg.model.type!r}."
    )
    assert cfg.model.target_column == "nd_mw", (
        f"F-9: target_column default must be 'nd_mw'; got {cfg.model.target_column!r}."
    )
    assert cfg.model.feature_columns is None, (
        "F-9: feature_columns default must be None (all weather columns resolved at fit-time)."
    )
    assert cfg.model.fit_intercept is True, (
        "F-9: fit_intercept default must be True (statsmodels OLS requires explicit constant)."
    )


def test_app_config_populates_evaluation_metrics_from_defaults() -> None:
    """Guards F-4: all four DESIGN §5.3 metrics are present after defaults-only load.

    ``conf/evaluation/metrics.yaml`` must compose into ``AppConfig.evaluation.metrics``
    with the full tuple ``("mae", "mape", "rmse", "wape")`` so that the harness
    computes every point-forecast metric out of the box.
    """
    cfg = load_config()

    assert cfg.evaluation is not None
    assert cfg.evaluation.metrics is not None, (
        "F-4: cfg.evaluation.metrics must be populated from the defaults list."
    )
    assert isinstance(cfg.evaluation.metrics, MetricsConfig)
    assert cfg.evaluation.metrics.names == ("mae", "mape", "rmse", "wape"), (
        f"F-4: default metric names must be ('mae','mape','rmse','wape'); "
        f"got {cfg.evaluation.metrics.names!r}."
    )


def test_app_config_populates_evaluation_benchmark_from_defaults() -> None:
    """Guards F-7: benchmark config defaults compose correctly from ``evaluation/benchmark.yaml``.

    Per D4 (plan §1) ``aggregation`` defaults to ``"mean"`` (preserves the MW
    scale).  Both ``holdout_start`` and ``holdout_end`` must be tz-aware UTC
    datetimes so the ``NesoBenchmarkConfig._validate_holdout`` validator does not
    fire.
    """
    cfg = load_config()

    assert cfg.evaluation is not None
    assert cfg.evaluation.benchmark is not None, (
        "F-7: cfg.evaluation.benchmark must be populated from the defaults list."
    )
    bench = cfg.evaluation.benchmark
    assert bench.aggregation == "mean", (
        f"D4: aggregation default must be 'mean'; got {bench.aggregation!r}."
    )
    assert bench.holdout_start.tzinfo is not None, (
        "F-7: holdout_start must be tz-aware; got naive datetime."
    )
    assert bench.holdout_end.tzinfo is not None, (
        "F-7: holdout_end must be tz-aware; got naive datetime."
    )
    # Confirm UTC specifically (offset == 0).
    assert bench.holdout_start.utcoffset().total_seconds() == 0, (  # type: ignore[union-attr]
        "F-7: holdout_start must be UTC (offset == 0)."
    )
    assert bench.holdout_end.utcoffset().total_seconds() == 0, (  # type: ignore[union-attr]
        "F-7: holdout_end must be UTC (offset == 0)."
    )


def test_app_config_populates_neso_forecast_from_defaults() -> None:
    """Guards F-6: ``ingestion.neso_forecast.resource_id`` is the pinned CKAN UUID.

    Research R6 fixed the resource on ``08e41551-80f8-4e28-a416-ea473a695db9``
    (Day Ahead HH Demand Forecast Performance, Apr 2021+).  The YAML must pin
    this UUID so stage-4 benchmark runs do not accidentally hit the wrong
    dataset.
    """
    from uuid import UUID

    cfg = load_config()

    assert cfg.ingestion is not None
    assert cfg.ingestion.neso_forecast is not None, (
        "F-6: cfg.ingestion.neso_forecast must be populated from the defaults list."
    )
    assert isinstance(cfg.ingestion.neso_forecast, NesoForecastIngestionConfig)
    expected = UUID("08e41551-80f8-4e28-a416-ea473a695db9")
    assert cfg.ingestion.neso_forecast.resource_id == expected, (
        f"F-6/R6: resource_id must be {expected}; got {cfg.ingestion.neso_forecast.resource_id!r}."
    )


# ---------------------------------------------------------------------------
# Stage 4 T1 — model-swap tests (Hydra group override)
# ---------------------------------------------------------------------------


def test_model_swap_to_naive_via_override() -> None:
    """Guards US-3 / F-9: ``model=naive`` override swaps the model variant.

    This is the "demo-moment" swap from the plan: a single CLI word changes
    which model runs without any code change.  After ``load_config(overrides=
    ['model=naive'])``, ``AppConfig.model`` must be a ``NaiveConfig`` with
    ``type="naive"`` and ``strategy="same_hour_last_week"`` (D1 default).
    """
    cfg = load_config(overrides=["model=naive"])

    assert cfg.model is not None, "cfg.model must not be None after model=naive override."
    assert isinstance(cfg.model, NaiveConfig), (
        f"US-3/F-9: expected NaiveConfig after model=naive; got {type(cfg.model).__name__!r}."
    )
    assert cfg.model.type == "naive", (
        f"US-3: discriminator tag must be 'naive'; got {cfg.model.type!r}."
    )
    assert cfg.model.strategy == "same_hour_last_week", (
        f"D1: default strategy must be 'same_hour_last_week'; got {cfg.model.strategy!r}."
    )


def test_model_swap_preserves_discriminator() -> None:
    """Guards D3 invariant: the discriminated union always resolves to the correct concrete type.

    After ``model=naive``, ``isinstance(cfg.model, NaiveConfig)`` must be
    ``True`` and ``cfg.model.type`` must be ``"naive"``.  After ``model=linear``
    (the default), ``isinstance(cfg.model, LinearConfig)`` must be ``True``.
    This invariant is load-bearing: every downstream consumer that branches on
    ``cfg.model.type`` depends on it.
    """
    naive_cfg = load_config(overrides=["model=naive"])
    assert isinstance(naive_cfg.model, NaiveConfig), (
        "D3: model=naive must resolve to NaiveConfig instance."
    )
    assert naive_cfg.model.type == "naive"

    linear_cfg = load_config(overrides=["model=linear"])
    assert isinstance(linear_cfg.model, LinearConfig), (
        "D3: model=linear must resolve to LinearConfig instance."
    )
    assert linear_cfg.model.type == "linear"


# ---------------------------------------------------------------------------
# Stage 4 T1 — field-level override tests
# ---------------------------------------------------------------------------


def test_naive_config_accepts_strategy_override_same_hour_yesterday() -> None:
    """Guards D1 Literal: ``same_hour_yesterday`` is a valid ``NaiveConfig.strategy`` value.

    The Literal type on ``strategy`` must admit all three declared values;
    this test exercises the non-default ``same_hour_yesterday`` path via a
    Hydra override so the full round-trip (YAML → DictConfig → Pydantic) is
    confirmed.
    """
    cfg = load_config(overrides=["model=naive", "model.strategy=same_hour_yesterday"])

    assert cfg.model is not None
    assert isinstance(cfg.model, NaiveConfig)
    assert cfg.model.strategy == "same_hour_yesterday", (
        "D1: 'same_hour_yesterday' must be accepted as a valid strategy override."
    )


def test_linear_config_accepts_feature_columns_override() -> None:
    """Guards ``LinearConfig.feature_columns``: explicit column list replaces the None default.

    A single-column ablation (``model.feature_columns=[temperature_2m]``) must
    produce a ``LinearConfig`` with ``feature_columns == ("temperature_2m",)``
    so the harness can narrow the regressor set without code changes.
    """
    cfg = load_config(overrides=["model=linear", "model.feature_columns=[temperature_2m]"])

    assert cfg.model is not None
    assert isinstance(cfg.model, LinearConfig)
    assert cfg.model.feature_columns == ("temperature_2m",), (
        f"feature_columns override must yield ('temperature_2m',); "
        f"got {cfg.model.feature_columns!r}."
    )


# ---------------------------------------------------------------------------
# Stage 4 T1 — negative / validator tests
# ---------------------------------------------------------------------------


def test_naive_config_rejects_unknown_strategy() -> None:
    """Guards D1 Literal narrowing: values outside the three declared strategies raise.

    ``NaiveConfig.strategy`` is typed as
    ``Literal["same_hour_yesterday", "same_hour_last_week", "same_hour_same_weekday"]``.
    Any other string (e.g. ``"same_hour_next_month"``) must cause a
    ``ValidationError`` so mis-typed strategies fail fast rather than silently
    producing nonsense predictions.
    """
    with pytest.raises((ValidationError, Exception)):
        load_config(overrides=["model=naive", "model.strategy=same_hour_next_month"])


def test_linear_config_rejects_extra_keys() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on ``LinearConfig``.

    An unknown key injected via ``+model.bogus=1`` must be rejected at the
    Pydantic validation step rather than silently accepted.  This prevents
    silent config drift.
    """
    with pytest.raises((ValidationError, Exception)):
        load_config(overrides=["model=linear", "+model.bogus=1"])


def test_metrics_config_rejects_unknown_metric_name() -> None:
    """Guards ``MetricsConfig.names`` Literal narrowing: unknown metric names raise.

    The ``names`` field is typed as
    ``tuple[Literal["mae", "mape", "rmse", "wape"], ...]``; any metric name
    outside this set (e.g. ``"fantasy_score"``) must be rejected so the harness
    never tries to look up a non-existent metric function.
    """
    with pytest.raises((ValidationError, Exception)):
        load_config(overrides=["evaluation.metrics.names=[mae,fantasy_score]"])


def test_benchmark_config_rejects_end_before_start() -> None:
    """Guards ``NesoBenchmarkConfig._validate_holdout``: ``holdout_end <= holdout_start`` raises.

    The holdout window must be a positive-duration interval; an inverted or
    zero-duration window is meaningless and the ``_validate_holdout``
    ``model_validator`` must raise ``ValidationError`` for it.
    """
    start = datetime(2023, 10, 1, tzinfo=UTC)
    end = datetime(2023, 9, 1, tzinfo=UTC)  # before start

    with pytest.raises(ValidationError):
        NesoBenchmarkConfig(aggregation="mean", holdout_start=start, holdout_end=end)


def test_benchmark_config_rejects_naive_datetime() -> None:
    """Guards ``NesoBenchmarkConfig._validate_holdout``: naive (tz-none) datetimes raise.

    The NESO forecast archive uses UTC timestamps; a naive datetime would
    introduce ambiguity in the holdout window alignment and must be rejected
    by the ``_validate_holdout`` validator.
    """
    naive_start = datetime(2023, 10, 1)  # no tzinfo
    naive_end = datetime(2024, 1, 1)  # no tzinfo

    with pytest.raises(ValidationError):
        NesoBenchmarkConfig(aggregation="mean", holdout_start=naive_start, holdout_end=naive_end)


def test_neso_forecast_config_rejects_extra_keys() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on ``NesoForecastIngestionConfig``.

    An unknown key (e.g. ``bogus_field``) must be rejected at construction
    time rather than silently accepted so config drift is caught early.
    """
    from pathlib import Path
    from uuid import UUID

    with pytest.raises(ValidationError):
        NesoForecastIngestionConfig(
            resource_id=UUID("08e41551-80f8-4e28-a416-ea473a695db9"),
            cache_dir=Path("/tmp/neso_forecast"),
            bogus_field="should_fail",  # type: ignore[call-arg]  # testing extra="forbid"
        )


# ---------------------------------------------------------------------------
# Stage 4 T1 — discriminated-union direct construction tests
# ---------------------------------------------------------------------------


def test_model_config_discriminates_on_type_tag() -> None:
    """Guards D3: Pydantic resolves the discriminated union on the ``type`` tag.

    Constructing an ``AppConfig`` with a raw dict payload whose ``type`` is
    ``"naive"`` must yield a ``NaiveConfig`` instance under ``AppConfig.model``
    — not a ``LinearConfig``.  This round-trip confirms the discriminator wiring
    without going through Hydra, so it tests Pydantic schema correctness in
    isolation.
    """
    app = AppConfig.model_validate(
        {
            "project": {"name": "bristol_ml", "seed": 0},
            "model": {
                "type": "naive",
                "strategy": "same_hour_last_week",
                "target_column": "nd_mw",
            },
        }
    )

    assert isinstance(app.model, NaiveConfig), (
        f"D3: type='naive' payload must resolve to NaiveConfig; got {type(app.model).__name__!r}."
    )
    assert app.model.type == "naive"
    assert app.model.strategy == "same_hour_last_week"


# ---------------------------------------------------------------------------
# Stage 4 T1 — backwards-compatibility tests
# ---------------------------------------------------------------------------


def test_app_config_model_field_defaults_to_none() -> None:
    """Guards backwards compatibility: programmatic ``AppConfig(...)`` without ``model`` is valid.

    Pre-Stage-4 call sites (e.g. ``tests/unit/features/test_assembler_cli.py``
    ``_make_app_config()``) construct ``AppConfig`` without passing a ``model``
    argument.  The ``model: ModelConfig | None = None`` default must preserve
    this pattern so those tests do not break when Stage 4 schemas land.
    """
    app = AppConfig(project=ProjectConfig(name="bristol_ml", seed=0))

    assert app.model is None, (
        "Backwards-compat: AppConfig constructed without model= must have model=None."
    )


# ---------------------------------------------------------------------------
# Stage 4 T2 — ModelMetadata schema tests
# ---------------------------------------------------------------------------


def test_model_metadata_defaults() -> None:
    """Guards T2: ``ModelMetadata`` field defaults.

    Per ``conf/_schemas.py`` the schema has:
    - ``fit_utc: datetime | None = None``
    - ``git_sha: str | None = None``
    - ``hyperparameters: dict[str, Any] = {}`` (via ``default_factory=dict``)

    Constructs with only the two required fields (``name``, ``feature_columns``)
    and asserts every optional field carries its documented default.

    Guards T2 (``ModelMetadata`` schema contract).
    """
    meta = ModelMetadata(name="linear-ols", feature_columns=("t2m",))

    assert meta.fit_utc is None, (
        f"fit_utc default must be None; got {meta.fit_utc!r} (T2 ModelMetadata defaults)."
    )
    assert meta.git_sha is None, (
        f"git_sha default must be None; got {meta.git_sha!r} (T2 ModelMetadata defaults)."
    )
    assert meta.hyperparameters == {}, (
        f"hyperparameters default must be {{}}; got {meta.hyperparameters!r} "
        "(T2 ModelMetadata defaults)."
    )


def test_model_metadata_frozen_rejects_mutation() -> None:
    """Guards T2: ``ModelMetadata`` is immutable (``frozen=True``).

    ``ConfigDict(frozen=True)`` on ``ModelMetadata`` must prevent in-place
    attribute assignment.  Attempting ``meta.name = "other"`` must raise
    (either ``ValidationError`` from Pydantic or ``TypeError`` from Python's
    ``__setattr__`` hook — the project tests accept both per the pattern in
    ``test_feature_set_config_direct_construction_smoke``).

    Guards T2 (immutable provenance record — ``conf/_schemas.ModelMetadata``
    docstring "Immutable provenance record").
    """
    meta = ModelMetadata(name="linear-ols", feature_columns=())

    with pytest.raises((ValidationError, TypeError)):
        meta.name = "other"  # type: ignore[misc]  # testing frozen enforcement


def test_model_metadata_forbids_extra_keys() -> None:
    """Guards T2: ``ModelMetadata`` uses ``extra="forbid"``.

    An unknown keyword argument (e.g. ``bogus=1``) must be rejected at
    construction time with a ``ValidationError`` so config drift is caught
    early rather than silently accepted.

    Guards T2 (``ConfigDict(extra='forbid')`` on ``ModelMetadata``).
    """
    with pytest.raises(ValidationError):
        ModelMetadata(  # type: ignore[call-arg]  # testing extra="forbid"
            name="x",
            feature_columns=(),
            bogus=1,
        )


def test_model_metadata_rejects_naive_fit_utc() -> None:
    """Guards T2: naive ``fit_utc`` is rejected by the ``model_validator``.

    The ``_validate_fit_utc`` validator in ``ModelMetadata`` must raise
    ``ValidationError`` when ``fit_utc`` is a naive ``datetime`` (i.e. one
    with no ``tzinfo``).  The error message must name ``fit_utc`` so the
    caller can diagnose the problem.

    Guards T2 (``model_validator`` rejects naive ``fit_utc`` — ``conf/_schemas.py``
    docstring: "Reject naive ``fit_utc`` values: we only store tz-aware UTC times").
    """
    naive_dt = datetime(2024, 1, 1)  # no tzinfo — intentionally naive

    with pytest.raises(ValidationError, match="fit_utc"):
        ModelMetadata(
            name="linear-ols",
            feature_columns=(),
            fit_utc=naive_dt,
        )


def test_model_metadata_accepts_tz_aware_fit_utc() -> None:
    """Guards T2: a UTC-aware ``fit_utc`` constructs without error.

    The counterpart to ``test_model_metadata_rejects_naive_fit_utc``: a
    properly tz-aware datetime (UTC offset == 0) must pass the
    ``_validate_fit_utc`` validator without raising.

    Guards T2 (positive path for tz-aware ``fit_utc``).
    """
    aware_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    meta = ModelMetadata(
        name="linear-ols",
        feature_columns=("t2m", "wind_speed"),
        fit_utc=aware_dt,
    )

    assert meta.fit_utc == aware_dt, (
        f"fit_utc must be stored as supplied; got {meta.fit_utc!r} (T2 tz-aware positive path)."
    )
    assert meta.fit_utc.tzinfo is not None, (
        "fit_utc must retain tzinfo after round-trip through Pydantic (T2 tz-aware positive path)."
    )


def test_model_metadata_name_pattern_rejects_uppercase() -> None:
    """Guards T2: ``ModelMetadata.name`` pattern rejects uppercase letters.

    The field is annotated ``Field(pattern=r'^[a-z][a-z0-9_.-]*$')`` — names
    must start with a lowercase letter and contain only lowercase letters,
    digits, underscores, dots, or hyphens.  A name starting with an uppercase
    letter (e.g. ``"BadName"``) must be rejected with a ``ValidationError``
    so model identifiers stay machine-friendly and consistent with
    ``ProjectConfig.name`` and ``FeatureSetConfig.name`` conventions.

    Guards T2 (``Field(pattern=...)`` on ``ModelMetadata.name``).
    """
    with pytest.raises(ValidationError):
        ModelMetadata(name="BadName", feature_columns=())


# ---------------------------------------------------------------------------
# Stage 5 T1 — calendar features config
# ---------------------------------------------------------------------------


def test_app_config_default_selects_weather_only() -> None:
    """Pins D-10 + defaults-list invariant: no override populates weather_only only.

    After a defaults-only ``load_config()`` the resolved config must have
    ``cfg.features.weather_only`` populated as a ``FeatureSetConfig`` and
    ``cfg.features.weather_calendar`` must be ``None``.  This guards the D-10
    decision that ``- features: weather_only`` is the ``conf/config.yaml``
    default, and that switching to the calendar set is a config change rather
    than a code change.  If either assertion fails the defaults-list wiring
    is broken.
    """
    cfg = load_config()

    assert cfg.features is not None, "AppConfig.features must not be None after defaults-only load."
    assert isinstance(cfg.features.weather_only, FeatureSetConfig), (
        "D-10: cfg.features.weather_only must be a FeatureSetConfig with no override."
    )
    assert cfg.features.weather_calendar is None, (
        "D-10: cfg.features.weather_calendar must be None when no features= override is given."
    )


def test_features_override_swaps_to_weather_calendar() -> None:
    """Pins D-10 group swap: ``features=weather_calendar`` populates weather_calendar only.

    After ``load_config(overrides=['features=weather_calendar'])`` the resolved
    config must have ``cfg.features.weather_calendar`` populated with the values
    from ``conf/features/weather_calendar.yaml`` — specifically ``name='weather_calendar'``,
    ``forward_fill_hours==3``, and ``demand_aggregation=='mean'`` — and
    ``cfg.features.weather_only`` must be ``None``.

    This is the mirror of how ``model=naive`` swaps the model variant; a single
    CLI word selects the feature set without any code change (AC-1).
    """
    cfg = load_config(overrides=["features=weather_calendar"])

    assert cfg.features is not None
    fset = cfg.features.weather_calendar
    assert fset is not None, (
        "D-10: cfg.features.weather_calendar must be populated after "
        "features=weather_calendar override."
    )
    assert isinstance(fset, FeatureSetConfig), (
        "D-10: weather_calendar must be a FeatureSetConfig instance."
    )
    assert fset.name == "weather_calendar", (
        f"D-10: feature-set name must be 'weather_calendar'; got {fset.name!r}."
    )
    assert fset.forward_fill_hours == 3, (
        f"D-10: forward_fill_hours default must be 3; got {fset.forward_fill_hours!r}."
    )
    assert fset.demand_aggregation == "mean", (
        f"D-10: demand_aggregation default must be 'mean'; got {fset.demand_aggregation!r}."
    )
    assert cfg.features.weather_only is None, (
        "D-10: cfg.features.weather_only must be None after features=weather_calendar override."
    )


def test_features_group_both_fields_optional() -> None:
    """Pins backwards-compatibility contract: FeaturesGroup() with no args is valid.

    ``FeaturesGroup()`` must construct successfully with both ``weather_only``
    and ``weather_calendar`` as ``None``.  This ensures that pre-Stage-5 code
    that builds ``FeaturesGroup`` programmatically (without supplying either
    field) is not broken by the Stage 5 addition of the ``weather_calendar``
    field.  Guards the ``| None = None`` idiom applied to both fields.
    """
    fg = FeaturesGroup()

    assert fg.weather_only is None, (
        "Backwards-compat: FeaturesGroup().weather_only must be None by default."
    )
    assert fg.weather_calendar is None, (
        "Backwards-compat: FeaturesGroup().weather_calendar must be None by default."
    )


def test_holidays_ingestion_config_defaults_from_yaml() -> None:
    """Pins D-1: load_config() populates cfg.ingestion.holidays from holidays.yaml.

    After a defaults-only ``load_config()`` the resolved config must have
    ``cfg.ingestion.holidays`` populated as a ``HolidaysIngestionConfig``
    with the values pinned in ``conf/ingestion/holidays.yaml``.  Specifically:

    - ``url`` contains ``www.gov.uk/bank-holidays.json`` (D-1 source).
    - ``cache_filename == 'holidays.parquet'``.
    - ``min_inter_request_seconds == 0.0`` (gov.uk has no documented rate limit; D-1).
    - ``divisions == ("england-and-wales", "scotland", "northern-ireland")``
      (all three divisions cached, policy-agnostic; D-1 / D-2).
    """
    cfg = load_config()

    assert cfg.ingestion is not None
    holidays = cfg.ingestion.holidays
    assert holidays is not None, (
        "D-1: cfg.ingestion.holidays must be populated from the defaults list."
    )
    assert isinstance(holidays, HolidaysIngestionConfig)
    assert "www.gov.uk/bank-holidays.json" in str(holidays.url), (
        f"D-1: url must contain 'www.gov.uk/bank-holidays.json'; got {holidays.url!r}."
    )
    assert holidays.cache_filename == "holidays.parquet", (
        f"D-1: cache_filename must be 'holidays.parquet'; got {holidays.cache_filename!r}."
    )
    assert holidays.min_inter_request_seconds == 0.0, (
        f"D-1: min_inter_request_seconds must be 0.0; got {holidays.min_inter_request_seconds!r}."
    )
    assert holidays.divisions == ("england-and-wales", "scotland", "northern-ireland"), (
        f"D-1/D-2: divisions must be all three UK divisions; got {holidays.divisions!r}."
    )


def test_holidays_ingestion_config_rejects_extra_keys() -> None:
    """Pins ConfigDict(extra='forbid') on HolidaysIngestionConfig.

    Supplying an unknown keyword argument at direct-construction time must raise
    ``ValidationError``.  This prevents silent config drift where a mis-typed
    field name is accepted rather than surfaced immediately.
    """
    with pytest.raises(ValidationError):
        HolidaysIngestionConfig(  # type: ignore[call-arg]  # testing extra="forbid"
            cache_dir=Path("/tmp/x"),
            bogus=1,
        )


def test_holidays_ingestion_config_divisions_literal_narrowing() -> None:
    """Pins Literal narrowing on HolidaysIngestionConfig.divisions.

    The ``divisions`` field uses a ``tuple[Literal["england-and-wales",
    "scotland", "northern-ireland"], ...]`` annotation.  Any string outside
    those three values (e.g. ``"isle-of-man"``) must be rejected with a
    ``ValidationError`` so an invalid division name fails fast rather than
    silently producing a malformed cache.
    """
    with pytest.raises(ValidationError):
        HolidaysIngestionConfig(
            cache_dir=Path("/tmp/x"),
            divisions=("isle-of-man",),  # type: ignore[arg-type]  # testing Literal narrowing
        )


def test_holidays_ingestion_config_direct_construction_smoke() -> None:
    """Smoke test: HolidaysIngestionConfig constructs correctly with only cache_dir supplied.

    Guards AC-7 schema soundness.  With only the required ``cache_dir`` argument
    all optional fields must assume their documented defaults: ``url`` is the
    gov.uk bank-holidays endpoint, ``cache_filename`` is ``'holidays.parquet'``,
    and the model is frozen (mutation raises).
    """
    cfg = HolidaysIngestionConfig(cache_dir=Path("/tmp/holidays"))

    assert "www.gov.uk/bank-holidays.json" in str(cfg.url), (
        f"D-1: default url must point to gov.uk bank-holidays endpoint; got {cfg.url!r}."
    )
    assert cfg.cache_filename == "holidays.parquet", (
        f"D-1: default cache_filename must be 'holidays.parquet'; got {cfg.cache_filename!r}."
    )

    # ConfigDict(frozen=True): mutation must raise.
    with pytest.raises((ValidationError, TypeError)):
        cfg.cache_filename = "changed.parquet"  # type: ignore[misc]  # testing frozen enforcement


def test_ingestion_group_holidays_field_defaults_to_none() -> None:
    """Pins backwards-compatibility: IngestionGroup() with no args produces holidays=None.

    Pre-Stage-5 call sites that construct ``IngestionGroup`` programmatically
    (without a ``holidays`` argument) must not break.  The ``holidays:
    HolidaysIngestionConfig | None = None`` default must hold so Stage 1-4 code
    paths remain valid after the Stage 5 schema extension.
    """
    ig = IngestionGroup()

    assert ig.holidays is None, (
        "Backwards-compat: IngestionGroup().holidays must be None by default."
    )


def test_weather_calendar_override_cache_filename_from_yaml() -> None:
    """Pins D-10: weather_calendar.yaml sets cache_filename to 'weather_calendar.parquet'.

    After a ``features=weather_calendar`` override, ``cfg.features.weather_calendar``
    must carry ``cache_filename == 'weather_calendar.parquet'`` — the value written
    in ``conf/features/weather_calendar.yaml``.  If this assertion fails the
    YAML file diverges from the plan.
    """
    cfg = load_config(overrides=["features=weather_calendar"])

    assert cfg.features is not None
    assert cfg.features.weather_calendar is not None
    assert cfg.features.weather_calendar.cache_filename == "weather_calendar.parquet", (
        f"D-10: cache_filename must be 'weather_calendar.parquet'; "
        f"got {cfg.features.weather_calendar.cache_filename!r}."
    )


def test_weather_calendar_rejects_invalid_demand_aggregation() -> None:
    """Pins Literal narrowing on FeatureSetConfig.demand_aggregation via the new path.

    After selecting ``features=weather_calendar``, overriding
    ``features.weather_calendar.demand_aggregation=median`` must raise
    ``ValidationError`` because ``'median'`` is not in the ``Literal["mean", "max"]``
    set.  This confirms the D-10 feature-set config shares the same validator
    as the weather-only set — the schema is reused verbatim (plan T1 note).
    """
    with pytest.raises(ValidationError):
        load_config(
            overrides=[
                "features=weather_calendar",
                "features.weather_calendar.demand_aggregation=median",
            ]
        )


# ---------------------------------------------------------------------------
# Stage 6 T1 — PlotsConfig + evaluation.plots group
# ---------------------------------------------------------------------------


def test_plots_config_rejects_extra_keys() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on ``PlotsConfig`` (AC-11).

    Supplying an unknown keyword argument at direct-construction time must
    raise ``ValidationError``.  This pins ``extra="forbid"`` so a mis-typed
    config key is caught immediately rather than silently ignored (plan T1).
    """
    with pytest.raises((ValidationError, Exception)):
        PlotsConfig(bogus=1)  # type: ignore[call-arg]  # testing extra="forbid"


def test_plots_config_figsize_default_is_twelve_by_eight() -> None:
    """Guards D5 human-mandated default: ``PlotsConfig().figsize == (12.0, 8.0)``.

    The Stage 6 plan §1 D5 amendment (2026-04-20 human mandate) specifies
    ``figsize=(12.0, 8.0)`` as the projector-legible default — wider than
    matplotlib's 6.4x4.8 baseline for meetup legibility (AC-2).  Pinned here
    so a future edit to ``_schemas.py`` cannot silently regress the value.
    """
    assert PlotsConfig().figsize == (12.0, 8.0), (
        "D5: PlotsConfig().figsize must be (12.0, 8.0) per the 2026-04-20 human mandate."
    )


def test_plots_config_acf_default_lags_is_168() -> None:
    """Guards D7: ``PlotsConfig().acf_default_lags == 168``.

    168 hourly lags = one full week, covering both the daily (lag 24) and
    weekly (lag 168) periodicity markers annotated on the ACF plot.  The
    ``statsmodels`` default is too short for hourly data (misses the weekly
    spike); this pin ensures the helper always renders the full 0-168 range
    unless explicitly overridden (D7, AC-7).
    """
    assert PlotsConfig().acf_default_lags == 168, (
        "D7: PlotsConfig().acf_default_lags must be 168 (covers daily + weekly seasonality)."
    )


def test_plots_config_display_tz_default_is_europe_london() -> None:
    """Guards D6: ``PlotsConfig().display_tz == 'Europe/London'``.

    Per D6 the default display timezone is ``"Europe/London"`` so the
    hour-of-day heatmap renders local wall-clock time ('demand at 18:00' is
    a local concept).  This test is conditional on the D6 T3 DST gate
    passing at implementation time; if it later fails, both the default and
    this assertion are updated together to ``"UTC"`` (plan §1 D6 amendment).
    """
    assert PlotsConfig().display_tz == "Europe/London", (
        "D6: PlotsConfig().display_tz must default to 'Europe/London' "
        "(conditional on the T3 DST-rendering gate passing)."
    )


def test_plots_config_defaults_match_plots_module() -> None:
    """Guards that all four ``PlotsConfig`` fields carry the D5/D6/D7 mandated defaults.

    Tests each field individually rather than deferring to a T2 module import
    so that the config contract is verifiable before ``plots.py`` ships.
    Serves as the explicit-field version of 'test_plots_config_defaults_match_plots_module'
    listed in the T1 test plan (plan §6 T1).
    """
    cfg = PlotsConfig()

    # D5 amendment (2026-04-20 human mandate) — projector-friendly figsize.
    assert cfg.figsize == (12.0, 8.0), (
        f"D5: figsize default must be (12.0, 8.0); got {cfg.figsize!r}."
    )
    # D5: middle ground between blurry 100 dpi and bloated 150 dpi.
    assert cfg.dpi == 110, f"D5: dpi default must be 110; got {cfg.dpi!r}."
    # D6: Europe/London local time for pedagogical readability.
    assert cfg.display_tz == "Europe/London", (
        f"D6: display_tz default must be 'Europe/London'; got {cfg.display_tz!r}."
    )
    # D7: 168 lags renders daily (24) + weekly (168) seasonality markers.
    assert cfg.acf_default_lags == 168, (
        f"D7: acf_default_lags default must be 168; got {cfg.acf_default_lags!r}."
    )


def test_config_loads_plots_group() -> None:
    """Guards AC-10/AC-11: ``load_config()`` yields a populated ``cfg.evaluation.plots``.

    After a defaults-only ``load_config()`` the resolved config must have
    ``cfg.evaluation.plots`` populated as a ``PlotsConfig`` instance with all
    four documented defaults.  This verifies the ``- evaluation/plots@evaluation.plots``
    entry in ``conf/config.yaml`` and the ``# @package evaluation.plots`` header
    in ``conf/evaluation/plots.yaml`` are wired correctly (plan T1).
    """
    cfg = load_config()

    assert cfg.evaluation is not None, "AppConfig.evaluation must not be None."
    plots = cfg.evaluation.plots
    assert isinstance(plots, PlotsConfig), (
        f"cfg.evaluation.plots must be a PlotsConfig instance; got {type(plots).__name__!r}."
    )
    # D5: figsize from the Hydra group file.
    assert plots.figsize == (12.0, 8.0), (
        f"D5: cfg.evaluation.plots.figsize must be (12.0, 8.0); got {plots.figsize!r}."
    )
    # D5: dpi from the Hydra group file.
    assert plots.dpi == 110, f"D5: cfg.evaluation.plots.dpi must be 110; got {plots.dpi!r}."
    # D6: display_tz from the Hydra group file.
    assert plots.display_tz == "Europe/London", (
        f"D6: cfg.evaluation.plots.display_tz must be 'Europe/London'; got {plots.display_tz!r}."
    )
    # D7: acf_default_lags from the Hydra group file.
    assert plots.acf_default_lags == 168, (
        f"D7: cfg.evaluation.plots.acf_default_lags must be 168; got {plots.acf_default_lags!r}."
    )


def test_config_figsize_overridable_via_hydra() -> None:
    """Guards AC-10: ``evaluation.plots.figsize`` is overridable via Hydra CLI (D5).

    ``load_config(overrides=["evaluation.plots.figsize=[16,10]"])`` must
    propagate to ``cfg.evaluation.plots.figsize == (16.0, 10.0)``.  Casts the
    Hydra ``ListConfig`` to a tuple for the equality check.
    """
    cfg = load_config(overrides=["evaluation.plots.figsize=[16,10]"])

    assert cfg.evaluation is not None
    figsize = tuple(cfg.evaluation.plots.figsize)
    assert figsize == (16.0, 10.0), (
        f"D5: override evaluation.plots.figsize=[16,10] must yield (16.0, 10.0); got {figsize!r}."
    )


def test_plots_config_forbids_negative_dpi() -> None:
    """Guards ``Field(ge=50, le=400)`` on ``PlotsConfig.dpi`` (AC-11).

    A ``dpi`` value below the minimum (e.g. 20) must raise ``ValidationError``
    because it is outside the declared ``Field(ge=50, le=400)`` bounds.  A
    positive in-range value (e.g. 150) must construct without error.
    """
    # Positive path: in-range dpi constructs successfully.
    cfg = PlotsConfig(dpi=150)
    assert cfg.dpi == 150, f"dpi=150 must be accepted; got {cfg.dpi!r}."

    # Negative path: below-minimum dpi raises.
    with pytest.raises(ValidationError):
        PlotsConfig(dpi=20)


def test_evaluation_group_plots_field_is_populated_by_default_factory() -> None:
    """Guards backwards-compat invariant: ``EvaluationGroup()`` yields ``plots: PlotsConfig``.

    ``EvaluationGroup`` uses ``plots: PlotsConfig = Field(default_factory=PlotsConfig)``
    (plan T1).  Constructing ``EvaluationGroup()`` with no arguments must produce
    a ``plots`` field that is a ``PlotsConfig`` instance (not ``None``), so that
    pre-Stage-6 programmatic call sites always see a populated config with the
    documented defaults rather than an absent field (AC-11).
    """
    eg = EvaluationGroup()

    assert isinstance(eg.plots, PlotsConfig), (
        f"EvaluationGroup().plots must be a PlotsConfig instance; "
        f"got {type(eg.plots).__name__!r} — "
        "backwards-compat invariant: default_factory=PlotsConfig must populate the field."
    )


# ---------------------------------------------------------------------------
# Stage 7 T1 — SarimaxConfig + SarimaxKwargs schema tests
# ---------------------------------------------------------------------------


def test_sarimax_config_rejects_extra_keys() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on ``SarimaxConfig`` (plan §6 T1).

    An unknown keyword argument supplied at direct-construction time must raise
    ``ValidationError``.  This mirrors the same constraint applied to
    ``LinearConfig``, ``NaiveConfig``, and all other model configs — config
    drift is caught at validation time rather than silently accepted.

    Cited criterion: plan §6 T1, ``test_sarimax_config_rejects_extra_keys``.
    """
    with pytest.raises(ValidationError):
        SarimaxConfig(bogus_kwarg=42)  # type: ignore[call-arg]  # testing extra="forbid"


def test_sarimax_kwargs_defaults_match_plan_d6() -> None:
    """Guards ``SarimaxKwargs`` field defaults match the plan §1 D6 mandate.

    D6 pins five constructor kwargs for the real-world seasonal demand case:
    ``enforce_stationarity=False`` and ``enforce_invertibility=False`` because
    the ML optimiser routinely finds non-stationary optima on hourly electricity
    series; ``concentrate_scale=True`` speeds optimisation; ``simple_differencing=False``
    keeps the Harvey representation so the full residual series reaches the Stage 6
    ``acf_residuals`` helper; ``hamilton_representation=False`` retains the
    statsmodels Harvey default.

    All five must match D6 exactly so downstream ``SARIMAX(...)`` calls inherit
    the correct settings from the config rather than statsmodels' own defaults.

    Cited criterion: plan §6 T1, ``test_sarimax_kwargs_defaults_match_design_D6``.
    """
    kwargs = SarimaxKwargs()

    assert kwargs.enforce_stationarity is False, (
        f"D6: enforce_stationarity default must be False; got {kwargs.enforce_stationarity!r}."
    )
    assert kwargs.enforce_invertibility is False, (
        f"D6: enforce_invertibility default must be False; got {kwargs.enforce_invertibility!r}."
    )
    assert kwargs.concentrate_scale is True, (
        f"D6: concentrate_scale default must be True; got {kwargs.concentrate_scale!r}."
    )
    assert kwargs.simple_differencing is False, (
        f"D6: simple_differencing default must be False; got {kwargs.simple_differencing!r}."
    )
    assert kwargs.hamilton_representation is False, (
        f"D6: hamilton_representation default must be False; got "
        f"{kwargs.hamilton_representation!r}."
    )


def test_sarimax_config_order_default_matches_plan_d2() -> None:
    """Guards ``SarimaxConfig`` ARIMA order defaults from plan §1 D2.

    D2 picks ``order=(1,0,1)`` (non-seasonal ARIMA) and
    ``seasonal_order=(1,1,1,24)`` (daily-seasonal, ``s=24``).  These are the
    conservative textbook defaults from Hyndman *fpp3* §9 for hourly electricity
    demand.  Pinning them here ensures a future edit to ``_schemas.py`` does not
    silently regress the documented defaults.

    Cited criterion: plan §6 T1, ``test_sarimax_config_order_default_matches_plan_D2``.
    """
    cfg = SarimaxConfig()

    assert cfg.order == (1, 0, 1), f"D2: order default must be (1, 0, 1); got {cfg.order!r}."
    assert cfg.seasonal_order == (1, 1, 1, 24), (
        f"D2: seasonal_order default must be (1, 1, 1, 24); got {cfg.seasonal_order!r}."
    )


def test_sarimax_config_weekly_fourier_default_is_three_harmonics() -> None:
    """Guards ``SarimaxConfig.weekly_fourier_harmonics`` default = 3 (plan §1 D1+D3).

    D1 specifies the DHR strategy: daily seasonality at ``s=24`` inside
    ``seasonal_order``; the weekly period (168 h) is absorbed by
    ``weekly_fourier_harmonics`` sin/cos Fourier pair columns.  D3 fixes three
    harmonic pairs (``k=1..3``, six columns) as the default — enough to capture
    the GB weekly demand shape after calendar exogenous regressors have already
    absorbed the day-of-week contribution.

    Cited criterion: plan §6 T1, ``test_sarimax_config_weekly_fourier_default_is_three_harmonics``.
    """
    cfg = SarimaxConfig()

    assert cfg.weekly_fourier_harmonics == 3, (
        f"D1+D3: weekly_fourier_harmonics default must be 3; got {cfg.weekly_fourier_harmonics!r}."
    )


def test_sarimax_config_weekly_fourier_rejects_negative() -> None:
    """Guards ``Field(ge=0)`` on ``SarimaxConfig.weekly_fourier_harmonics`` (plan §6 T1).

    A negative harmonic count is physically meaningless — ``ge=0`` must cause
    Pydantic to raise ``ValidationError`` for ``weekly_fourier_harmonics=-1``.
    ``harmonics=0`` is the documented no-op path (disables the weekly Fourier
    exogenous path) so only strictly negative values are rejected.

    Cited criterion: plan §6 T1, ``test_sarimax_config_weekly_fourier_rejects_negative``.
    """
    with pytest.raises(ValidationError):
        SarimaxConfig(weekly_fourier_harmonics=-1)


def test_model_config_union_dispatches_on_type_sarimax() -> None:
    """Guards D3: the discriminated union resolves ``type='sarimax'`` to ``SarimaxConfig``.

    Constructing an ``AppConfig`` with a raw dict payload whose ``type`` is
    ``"sarimax"`` must yield a ``SarimaxConfig`` instance under ``AppConfig.model``
    — not a ``LinearConfig`` or ``NaiveConfig``.  This round-trip confirms the
    discriminator wiring (``discriminator="type"`` on ``AppConfig.model``) without
    going through Hydra, so it tests Pydantic schema correctness in isolation.

    Uses ``type(app.model).__name__ == "SarimaxConfig"`` to be explicit about
    the resolved class, as required by plan §6 T1.

    Cited criterion: plan §6 T1, ``test_model_config_union_dispatches_on_type_sarimax``.
    """
    app = AppConfig.model_validate(
        {
            "project": {"name": "bristol_ml", "seed": 0},
            "model": {"type": "sarimax"},
        }
    )

    assert type(app.model).__name__ == "SarimaxConfig", (
        f"D3: type='sarimax' payload must resolve to SarimaxConfig; "
        f"got {type(app.model).__name__!r}."
    )
    assert isinstance(app.model, SarimaxConfig), (
        "D3: resolved model must be an instance of SarimaxConfig."
    )
    assert app.model.type == "sarimax", (
        f"D3: discriminator tag must be 'sarimax'; got {app.model.type!r}."
    )


def test_config_loads_model_sarimax_via_hydra() -> None:
    """Guards AC-6: ``load_config(overrides=['model=sarimax'])`` resolves to ``SarimaxConfig``.

    Verifies that the ``conf/model/sarimax.yaml`` Hydra group file is correctly
    wired and populates ``cfg.model`` as a ``SarimaxConfig`` instance with the
    documented D2 and D6 defaults:

    - ``type == 'sarimax'``
    - ``order == (1, 0, 1)`` (D2)
    - ``seasonal_order == (1, 1, 1, 24)`` (D2)
    - ``weekly_fourier_harmonics == 3`` (D1+D3)
    - ``sarimax_kwargs.concentrate_scale is True`` (D6)

    Cited criterion: plan §6 T1, ``test_config_loads_model_sarimax_via_hydra``.
    """
    cfg = load_config(overrides=["model=sarimax"])

    assert cfg.model is not None, "cfg.model must not be None after model=sarimax override."
    assert isinstance(cfg.model, SarimaxConfig), (
        f"AC-6: expected SarimaxConfig after model=sarimax; got {type(cfg.model).__name__!r}."
    )
    assert cfg.model.type == "sarimax", (
        f"AC-6: discriminator tag must be 'sarimax'; got {cfg.model.type!r}."
    )
    # D2: order and seasonal_order defaults from conf/model/sarimax.yaml.
    assert tuple(cfg.model.order) == (1, 0, 1), (
        f"D2: order must be (1, 0, 1) from YAML defaults; got {tuple(cfg.model.order)!r}."
    )
    assert tuple(cfg.model.seasonal_order) == (1, 1, 1, 24), (
        f"D2: seasonal_order must be (1, 1, 1, 24) from YAML defaults; "
        f"got {tuple(cfg.model.seasonal_order)!r}."
    )
    # D1+D3: weekly Fourier harmonics default.
    assert cfg.model.weekly_fourier_harmonics == 3, (
        f"D1+D3: weekly_fourier_harmonics must be 3; got {cfg.model.weekly_fourier_harmonics!r}."
    )
    # D6: concentrate_scale is the most load-bearing of the SarimaxKwargs fields.
    assert cfg.model.sarimax_kwargs.concentrate_scale is True, (
        "D6: sarimax_kwargs.concentrate_scale must be True from YAML defaults."
    )


def test_config_loads_splitter_sarimax_overrides() -> None:
    """Guards plan §1 D4: per-field Hydra overrides wire the SARIMAX splitter settings.

    D4 mandates a fixed sliding window with ``fixed_window=true`` and
    ``step=168`` (weekly folds) to keep per-fold fit time within AC-3's budget
    on laptop CPUs.  These overrides are applied as per-field CLI Hydra arguments
    rather than a new ``conf/evaluation/*.yaml`` file (plan §6 T1 note: no new
    group file created).

    The test asserts:
    - ``cfg.evaluation.rolling_origin.fixed_window is True`` (D4 fixed window).
    - ``cfg.evaluation.rolling_origin.step == 168`` (D4 weekly step).
    - ``cfg.evaluation.rolling_origin.min_train_periods == 8760`` — project
      default unchanged (one year of hourly data — no override applied).

    The ``model=sarimax`` override is also applied so the full D4 invocation
    is tested as a unit.

    Cited criterion: plan §6 T1, ``test_config_loads_splitter_sarimax_overrides``.
    """
    cfg = load_config(
        overrides=[
            "model=sarimax",
            "evaluation.rolling_origin.fixed_window=true",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert cfg.evaluation is not None, "cfg.evaluation must not be None."
    splitter = cfg.evaluation.rolling_origin
    assert splitter is not None, (
        "cfg.evaluation.rolling_origin must be populated when the override is applied."
    )

    # D4: fixed sliding window.
    assert splitter.fixed_window is True, (
        f"D4: fixed_window must be True after override; got {splitter.fixed_window!r}."
    )
    # D4: weekly step (168 h).
    assert splitter.step == 168, (
        f"D4: step must be 168 (weekly folds) after override; got {splitter.step!r}."
    )
    # Project default: min_train_periods stays at 8760 — no override applied.
    assert splitter.min_train_periods == 8760, (
        f"Project default: min_train_periods must remain 8760 (one year of hourly data); "
        f"got {splitter.min_train_periods!r}."
    )


# ---------------------------------------------------------------------------
# Stage 8 T1 — ScipyParametricConfig schema tests
# ---------------------------------------------------------------------------


def test_scipy_parametric_config_defaults_match_yaml() -> None:
    """Guards plan §1 D1+D2+D3: ``ScipyParametricConfig()`` defaults match the YAML group file.

    The YAML group file ``conf/model/scipy_parametric.yaml`` and the Pydantic
    ``ScipyParametricConfig`` model must have the same defaults; a drift between
    them is a silent correctness hazard (the YAML wins at Hydra-load time, the
    Pydantic defaults win when the class is constructed directly).

    Checked defaults:

    - ``type == 'scipy_parametric'`` (discriminator tag).
    - ``target_column == 'nd_mw'``.
    - ``feature_columns is None`` — the parametric design-matrix rule narrows
      this further at fit time (D2 clarification) but the config field default
      stays permissive.
    - ``temperature_column == 'temperature_2m'``.
    - ``diurnal_harmonics == 3`` (D2).
    - ``weekly_harmonics == 2`` (D2).
    - ``t_heat_celsius == 15.5`` (D1 Elexon convention).
    - ``t_cool_celsius == 22.0`` (D1 Elexon convention).
    - ``loss == 'linear'`` (D3 — keeps Gaussian-pcov CI rigorous).
    - ``max_iter == 5000`` (D6).
    - ``p0 is None`` (D4 — triggers data-driven derivation inside fit()).

    Cited criterion: plan §6 T1, ``test_scipy_parametric_config_defaults_match_yaml``.
    """
    cfg = ScipyParametricConfig()

    assert cfg.type == "scipy_parametric", (
        f"D10: type default must be 'scipy_parametric'; got {cfg.type!r}."
    )
    assert cfg.target_column == "nd_mw", (
        f"target_column default must be 'nd_mw'; got {cfg.target_column!r}."
    )
    assert cfg.feature_columns is None, (
        f"feature_columns default must be None; got {cfg.feature_columns!r}."
    )
    assert cfg.temperature_column == "temperature_2m", (
        f"temperature_column default must be 'temperature_2m'; got {cfg.temperature_column!r}."
    )
    assert cfg.diurnal_harmonics == 3, (
        f"D2: diurnal_harmonics default must be 3; got {cfg.diurnal_harmonics!r}."
    )
    assert cfg.weekly_harmonics == 2, (
        f"D2: weekly_harmonics default must be 2; got {cfg.weekly_harmonics!r}."
    )
    assert cfg.t_heat_celsius == 15.5, (
        f"D1: t_heat_celsius default must be 15.5 (Elexon); got {cfg.t_heat_celsius!r}."
    )
    assert cfg.t_cool_celsius == 22.0, (
        f"D1: t_cool_celsius default must be 22.0 (Elexon); got {cfg.t_cool_celsius!r}."
    )
    assert cfg.loss == "linear", f"D3: loss default must be 'linear'; got {cfg.loss!r}."
    assert cfg.max_iter == 5000, f"D6: max_iter default must be 5000; got {cfg.max_iter!r}."
    assert cfg.p0 is None, f"D4: p0 default must be None; got {cfg.p0!r}."


def test_scipy_parametric_config_rejects_extra_fields() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on ``ScipyParametricConfig`` (plan §6 T1)."""
    with pytest.raises(ValidationError):
        ScipyParametricConfig(spoof_field="bad")  # type: ignore[call-arg]


def test_scipy_parametric_config_rejects_invalid_loss() -> None:
    """Guards D3: unknown ``loss`` string must be rejected by the ``Literal[...]`` field.

    The four accepted values are ``linear`` / ``soft_l1`` / ``huber`` /
    ``cauchy``; anything else is a typo or a future addition that must first
    reach the Pydantic schema before the YAML ships.
    """
    with pytest.raises(ValidationError):
        ScipyParametricConfig(loss="nonsense")  # type: ignore[arg-type]


def test_scipy_parametric_config_rejects_negative_harmonics() -> None:
    """Guards ``Field(ge=0)`` on both harmonic counts.

    Negative harmonic counts are physically meaningless; ``harmonics=0`` is the
    documented no-op (ablation) path.
    """
    with pytest.raises(ValidationError):
        ScipyParametricConfig(diurnal_harmonics=-1)
    with pytest.raises(ValidationError):
        ScipyParametricConfig(weekly_harmonics=-1)


def test_model_config_discriminator_parses_scipy_parametric() -> None:
    """Guards plan §1 D10: discriminated union dispatches ``type='scipy_parametric'`` correctly."""
    app = AppConfig.model_validate(
        {
            "project": {"name": "bristol_ml", "seed": 0},
            "model": {"type": "scipy_parametric"},
        }
    )

    assert type(app.model).__name__ == "ScipyParametricConfig", (
        f"D10: type='scipy_parametric' payload must resolve to ScipyParametricConfig; "
        f"got {type(app.model).__name__!r}."
    )
    assert isinstance(app.model, ScipyParametricConfig)
    assert app.model.type == "scipy_parametric"


def test_config_loads_model_scipy_parametric_via_hydra() -> None:
    """Guards AC-7 / D10: ``load_config(overrides=['model=scipy_parametric'])`` resolves.

    Verifies that the ``conf/model/scipy_parametric.yaml`` Hydra group file is
    correctly wired and populates ``cfg.model`` with D1 through D6 defaults.
    """
    cfg = load_config(overrides=["model=scipy_parametric"])

    assert cfg.model is not None
    assert isinstance(cfg.model, ScipyParametricConfig), (
        f"expected ScipyParametricConfig; got {type(cfg.model).__name__!r}."
    )
    assert cfg.model.type == "scipy_parametric"
    assert cfg.model.diurnal_harmonics == 3
    assert cfg.model.weekly_harmonics == 2
    assert cfg.model.t_heat_celsius == 15.5
    assert cfg.model.t_cool_celsius == 22.0
    assert cfg.model.loss == "linear"
