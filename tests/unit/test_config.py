"""Stage 0 + Stage 3 T1 smoke/acceptance tests for the config pipeline.

Stage 0 tests cover the fundamental `load_config()` → `AppConfig` round-trip
and the `python -m bristol_ml` demo moment.

Stage 3 T1 tests encode the acceptance criteria for the new Hydra config groups
(`conf/features/weather_only.yaml`, `conf/evaluation/rolling_origin.yaml`) and
the four new Pydantic models (`FeatureSetConfig`, `FeaturesGroup`,
`SplitterConfig`, `EvaluationGroup`).  Each test docstring cites the plan
decision or acceptance criterion it guards so future readers can trace back to
`docs/plans/active/03-feature-assembler.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from bristol_ml import load_config
from conf._schemas import (
    AppConfig,
    EvaluationGroup,
    FeatureSetConfig,
    FeaturesGroup,
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
