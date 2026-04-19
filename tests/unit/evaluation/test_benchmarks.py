"""Spec-derived tests for the three-way benchmark helper (Stage 4 T8).

Plan: ``docs/plans/active/04-linear-baseline.md`` §6 Task T8.

Every test is derived from the plan acceptance criteria or from a
contract corner documented in the helper's docstring.  No production
code is modified here; if a test fails, the failure indicates a
deviation from the spec.

Conventions
-----------
- British English in docstrings.
- ``np.random.default_rng(seed=42)`` for reproducible synthetic data.
- Synthetic NESO frames mirror the real ingester's output:
  ``timestamp_utc`` UTC-aware, plus ``demand_forecast_mw`` /
  ``demand_outturn_mw`` numeric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bristol_ml.evaluation.benchmarks import (
    align_half_hourly_to_hourly,
    compare_on_holdout,
)
from bristol_ml.evaluation.metrics import mae, mape, rmse, wape
from bristol_ml.features.assembler import WEATHER_VARIABLE_COLUMNS
from bristol_ml.models.linear import LinearModel
from bristol_ml.models.naive import NaiveModel
from conf._schemas import LinearConfig, NaiveConfig, SplitterConfig

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

_WEATHER_COLS = [name for name, _ in WEATHER_VARIABLE_COLUMNS]
_METRICS = [mae, rmse, mape, wape]


def _make_feature_df(n_hours: int = 24 * 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly feature table with a UTC ``DatetimeIndex`` + ``nd_mw``.

    Mirrors the Stage 3 assembler shape minus the provenance columns —
    those do not matter to the benchmark helper.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-10-01", periods=n_hours, freq="1h", tz="UTC")
    weather = {
        name: rng.normal(loc=10, scale=3, size=n_hours).astype("float32") for name in _WEATHER_COLS
    }
    # Deterministic daily sine with noise — weather-only regression has
    # no hope of fitting this, which is the pedagogical point.
    nd = 30_000 + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 200, n_hours)
    return pd.DataFrame({"nd_mw": nd, **weather}, index=idx)


def _make_neso_df(n_hours: int = 24 * 60, seed: int = 7) -> pd.DataFrame:
    """Synthetic half-hourly NESO forecast frame over the same window as the feature df."""
    rng = np.random.default_rng(seed)
    hh = pd.date_range("2023-10-01", periods=2 * n_hours, freq="30min", tz="UTC")
    base = rng.normal(30_000, 500, 2 * n_hours)
    return pd.DataFrame(
        {
            "timestamp_utc": hh,
            "demand_forecast_mw": base + rng.normal(0, 200, 2 * n_hours),
            "demand_outturn_mw": base + rng.normal(0, 50, 2 * n_hours),
        }
    )


def _splitter_cfg() -> SplitterConfig:
    return SplitterConfig(
        min_train_periods=24 * 21,
        test_len=24 * 7,
        step=24 * 7,
        gap=0,
        fixed_window=False,
    )


# ---------------------------------------------------------------------------
# align_half_hourly_to_hourly — kernel contract
# ---------------------------------------------------------------------------


def test_benchmarks_aligns_half_hourly_to_hourly() -> None:
    """Plan T8 acceptance: synthetic HH mean aggregation matches hand computation."""
    ts = pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "demand_forecast_mw": [10, 20, 30, 40, 50, 60],
            "demand_outturn_mw": [11, 21, 31, 41, 51, 61],
        }
    )

    out_mean = align_half_hourly_to_hourly(df, aggregation="mean")

    # Two settlement periods per UTC hour → pairwise means.
    expected_forecast = [15.0, 35.0, 55.0]
    expected_outturn = [16.0, 36.0, 56.0]
    assert list(out_mean["demand_forecast_mw"]) == expected_forecast
    assert list(out_mean["demand_outturn_mw"]) == expected_outturn
    assert out_mean.index.tz is not None
    assert str(out_mean.index.tz) == "UTC"
    assert len(out_mean) == 3


def test_benchmarks_align_first_keeps_first_settlement_period_per_hour() -> None:
    """'first' rule drops the second settlement period; plan D4 ablation path."""
    ts = pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "demand_forecast_mw": [10, 20, 30, 40, 50, 60],
            "demand_outturn_mw": [11, 21, 31, 41, 51, 61],
        }
    )

    out_first = align_half_hourly_to_hourly(df, aggregation="first")

    assert list(out_first["demand_forecast_mw"]) == [10.0, 30.0, 50.0]
    assert list(out_first["demand_outturn_mw"]) == [11.0, 31.0, 51.0]


def test_benchmarks_align_rejects_unknown_aggregation() -> None:
    """Only 'mean' and 'first' are valid — mirrors the Pydantic Literal."""
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"),
            "demand_forecast_mw": [10, 20],
            "demand_outturn_mw": [11, 21],
        }
    )
    with pytest.raises(ValueError, match="aggregation must be 'mean' or 'first'"):
        align_half_hourly_to_hourly(df, aggregation="sum")  # type: ignore[arg-type]


def test_benchmarks_align_rejects_non_utc_timestamp() -> None:
    """Tz-naive or non-UTC ``timestamp_utc`` is a hard error."""
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="30min"),
            "demand_forecast_mw": [10, 20],
            "demand_outturn_mw": [11, 21],
        }
    )
    with pytest.raises(ValueError, match="UTC-aware"):
        align_half_hourly_to_hourly(df)


def test_benchmarks_align_rejects_missing_value_column() -> None:
    """Missing value columns raise before resampling — fail fast."""
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"),
            "demand_forecast_mw": [10, 20],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        align_half_hourly_to_hourly(df)


# ---------------------------------------------------------------------------
# compare_on_holdout — plan-named acceptance tests
# ---------------------------------------------------------------------------


def test_benchmarks_three_way_table_shape() -> None:
    """Plan T8 acceptance: three-row DataFrame with one column per metric."""
    df = _make_feature_df()
    neso = _make_neso_df()

    models = {
        "naive": NaiveModel(NaiveConfig(strategy="same_hour_last_week")),
        "linear": LinearModel(LinearConfig()),
    }
    table = compare_on_holdout(models, df, neso, _splitter_cfg(), _METRICS)

    assert table.shape == (3, 4)
    assert set(table.index) == {"naive", "linear", "neso"}
    assert list(table.columns) == ["mae", "rmse", "mape", "wape"]
    # All four metric values must be finite floats for every row.
    assert np.isfinite(table.to_numpy()).all()


def test_benchmarks_neso_row_matches_direct_metric_computation() -> None:
    """Hand-check: the NESO row equals ``metric(outturn, forecast)`` on the aligned slice.

    A regression in the alignment or the holdout intersection would
    silently move the NESO scores; pinning the row to an explicit
    hand computation catches it.
    """
    df = _make_feature_df()
    neso = _make_neso_df()
    models = {"naive": NaiveModel(NaiveConfig())}

    table = compare_on_holdout(models, df, neso, _splitter_cfg(), _METRICS)

    # Replicate the internal alignment + holdout intersection.
    hourly = align_half_hourly_to_hourly(neso, aggregation="mean").dropna()
    # Holdout span = [first fold's test_start, last fold's test_end].
    # With _splitter_cfg() over the default 60-day feature frame this
    # resolves to 2023-10-22 00:00Z → 2023-11-29 23:00Z; recompute from
    # the splitter here to stay in sync with any future tweak.
    from bristol_ml.evaluation.splitter import rolling_origin_split_from_config

    folds = list(rolling_origin_split_from_config(len(df), _splitter_cfg()))
    test_start = df.index[folds[0][1][0]]
    test_end = df.index[folds[-1][1][-1]]
    mask = (hourly.index >= test_start) & (hourly.index <= test_end)
    slice_df = hourly.loc[mask]

    assert table.loc["neso", "mae"] == pytest.approx(
        mae(slice_df["demand_outturn_mw"], slice_df["demand_forecast_mw"])
    )
    assert table.loc["neso", "rmse"] == pytest.approx(
        rmse(slice_df["demand_outturn_mw"], slice_df["demand_forecast_mw"])
    )


def test_benchmarks_rejects_empty_models() -> None:
    """At least one model must be passed; no silent empty table."""
    df = _make_feature_df()
    neso = _make_neso_df()
    with pytest.raises(ValueError, match="'models' is empty"):
        compare_on_holdout({}, df, neso, _splitter_cfg(), _METRICS)


def test_benchmarks_rejects_empty_metrics() -> None:
    """At least one metric must be passed."""
    df = _make_feature_df()
    neso = _make_neso_df()
    models = {"naive": NaiveModel(NaiveConfig())}
    with pytest.raises(ValueError, match="'metrics' is empty"):
        compare_on_holdout(models, df, neso, _splitter_cfg(), [])


def test_benchmarks_raises_when_holdout_intersection_is_empty() -> None:
    """If NESO coverage does not overlap the holdout, raise rather than return empty."""
    df = _make_feature_df()
    # NESO frame in a distant year — zero overlap with the feature df.
    hh = pd.date_range("2019-01-01", periods=48, freq="30min", tz="UTC")
    neso = pd.DataFrame(
        {
            "timestamp_utc": hh,
            "demand_forecast_mw": np.full(48, 30_000.0),
            "demand_outturn_mw": np.full(48, 30_000.0),
        }
    )
    models = {"naive": NaiveModel(NaiveConfig())}

    with pytest.raises(ValueError, match="intersection"):
        compare_on_holdout(models, df, neso, _splitter_cfg(), _METRICS)


def test_benchmarks_accepts_dataframe_with_timestamp_column_or_index() -> None:
    """Either a ``timestamp_utc`` column or a UTC DatetimeIndex is acceptable."""
    df_idx = _make_feature_df()
    df_col = df_idx.reset_index().rename(columns={"index": "timestamp_utc"})
    neso = _make_neso_df()
    models = {"naive": NaiveModel(NaiveConfig())}

    table_idx = compare_on_holdout(models, df_idx, neso, _splitter_cfg(), _METRICS)
    table_col = compare_on_holdout(models, df_col, neso, _splitter_cfg(), _METRICS)
    # The harness sees identical inputs post-normalisation, so the
    # metric values must match exactly.
    pd.testing.assert_frame_equal(table_idx, table_col)


def test_benchmarks_rejects_non_utc_feature_index() -> None:
    """H-1 guard propagates from harness — non-UTC feature index raises."""
    df = _make_feature_df()
    df = df.tz_convert("Europe/London")
    neso = _make_neso_df()
    models = {"naive": NaiveModel(NaiveConfig())}
    with pytest.raises(ValueError, match="UTC"):
        compare_on_holdout(models, df, neso, _splitter_cfg(), _METRICS)


def test_benchmarks_importable_from_namespace() -> None:
    """Helpers are lazily re-exported from the evaluation namespace (plan H-3 idiom)."""
    import bristol_ml.evaluation as ev

    assert ev.compare_on_holdout is compare_on_holdout
    assert ev.align_half_hourly_to_hourly is align_half_hourly_to_hourly
