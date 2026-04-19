"""Stage 0 + Stage 3 T1 smoke/acceptance tests for the config pipeline.

Stage 0 tests cover the fundamental `load_config()` → `AppConfig` round-trip
and the `python -m bristol_ml` demo moment.

Stage 3 T1 tests encode the acceptance criteria for the new Hydra config groups
(`conf/features/weather_only.yaml`, `conf/evaluation/rolling_origin.yaml`) and
the four new Pydantic models (`FeatureSetConfig`, `FeaturesGroup`,
`SplitterConfig`, `EvaluationGroup`).  Each test docstring cites the plan
decision or acceptance criterion it guards so future readers can trace back to
`docs/plans/completed/03-feature-assembler.md`.
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
    LinearConfig,
    MetricsConfig,
    NaiveConfig,
    NesoBenchmarkConfig,
    NesoForecastIngestionConfig,
    ProjectConfig,
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
