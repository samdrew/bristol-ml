"""Acceptance tests for the Stage 4 ``python -m bristol_ml.train`` CLI.

Plan: ``docs/plans/active/04-linear-baseline.md`` §6 Task T8.

The CLI wires the harness, metrics, benchmark, and model registry into
a single entry point.  The tests here exercise it primarily in-process
via :func:`bristol_ml.train._cli_main` (fast; lets the test capture
stdout with :class:`pytest.capsys`), plus one subprocess smoke that
confirms the module-level ``__main__`` wiring works under
``python -m bristol_ml.train``.

Every test writes a synthetic feature-table parquet to a ``tmp_path``
location and drives the CLI's Hydra overrides so the run is
hermetic — no reliance on ``data/features/`` on the host machine.

Conventions
-----------
- British English in docstrings.
- ``np.random.default_rng(seed=42)`` for reproducibility.
- ``pytest.MonkeyPatch`` sets ``BRISTOL_ML_CACHE_DIR`` because multiple
  config groups resolve cache paths from it.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bristol_ml.features.assembler import (
    CALENDAR_OUTPUT_SCHEMA,
    CALENDAR_VARIABLE_COLUMNS,
    OUTPUT_SCHEMA,
    WEATHER_VARIABLE_COLUMNS,
)
from bristol_ml.train import _cli_main, _resolve_feature_set

# ---------------------------------------------------------------------------
# Shared synthetic feature-table fixture
# ---------------------------------------------------------------------------


def _write_feature_cache(path: Path, n_hours: int = 24 * 90, seed: int = 42) -> Path:
    """Write a minimal feature-table parquet matching ``assembler.OUTPUT_SCHEMA``.

    The table is synthetic but schema-conformant: 90 days of hourly UTC
    timestamps, demand with a daily sine + noise, Gaussian weather.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-10-01", periods=n_hours, freq="1h", tz="UTC")
    nd = (
        30_000 + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 200, n_hours)
    ).astype("int32")
    tsd = (nd + 3_000).astype("int32")
    weather = {
        name: rng.normal(loc=10, scale=3, size=n_hours).astype("float32")
        for name, _ in WEATHER_VARIABLE_COLUMNS
    }
    retrieved_at = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "nd_mw": nd,
            "tsd_mw": tsd,
            **weather,
            "neso_retrieved_at_utc": [retrieved_at] * n_hours,
            "weather_retrieved_at_utc": [retrieved_at] * n_hours,
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(OUTPUT_SCHEMA, safe=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def warm_feature_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate a warm feature-table cache and point Hydra at it.

    ``BRISTOL_ML_CACHE_DIR`` drives the ``${oc.env:...}`` interpolation in
    every cache-owning YAML, so pointing it at ``tmp_path`` gives the CLI
    a writable sandbox.  The feature-table parquet lands at
    ``tmp_path/weather_only.parquet`` (matching the default
    ``cache_filename``).

    Stage 9 D17 hygiene: we also monkeypatch
    :data:`bristol_ml.registry.DEFAULT_REGISTRY_DIR` so every train-CLI
    test run registers into ``tmp_path/registry`` rather than polluting
    the repo-level ``data/registry/`` with synthetic fold artefacts.
    Tests that want to assert on the registry pass ``--registry-dir``
    explicitly.
    """
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    # Stage 9 hygiene: redirect the registry root to the per-test tmpdir.
    from bristol_ml import registry

    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_DIR", tmp_path / "registry")
    _write_feature_cache(tmp_path / "weather_only.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# In-process acceptance tests (plan T8 acceptance set)
# ---------------------------------------------------------------------------


def test_train_cli_prints_metric_table(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T8 acceptance: the CLI prints a metric table on stdout."""
    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    # The per-fold table header comes from _cli_main; metric-column names
    # from METRIC_REGISTRY.
    assert "Per-fold metrics for model=linear" in out
    for metric in ("mae", "mape", "rmse", "wape"):
        assert metric in out
    # At least one fold row present — the per-fold DataFrame always has
    # a ``fold_index`` column.
    assert "fold_index" in out


def test_train_cli_model_swap(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T8 acceptance: ``model=naive`` override produces naive-specific output."""
    exit_code = _cli_main(
        [
            "model=naive",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=naive" in out


def test_train_cli_exits_2_when_feature_cache_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing feature-table cache produces exit code 2 with a named-path error."""
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    # Deliberately do NOT call _write_feature_cache — the cache is absent.

    exit_code = _cli_main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Feature-table cache missing" in err


def test_train_cli_skips_benchmark_when_forecast_cache_missing(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No forecast cache → harness table printed, benchmark table omitted.

    This is the expected behaviour during a fresh-clone live demo: the
    assembler cache is warm but the slow NESO forecast ingest has not
    been run yet.  The CLI must still exit 0 and print the per-fold
    table; the benchmark section is elided.
    """
    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics" in out
    assert "Benchmark comparison" not in out


def test_train_cli_runs_benchmark_when_forecast_cache_warm(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Warm forecast cache → three-way benchmark table printed after the harness output."""
    # Synthesise a minimal NESO forecast parquet at the default cache
    # path — ``${oc.env:BRISTOL_ML_CACHE_DIR}/neso_forecast.parquet``.
    from bristol_ml.ingestion.neso_forecast import OUTPUT_SCHEMA as NESO_SCHEMA

    n_hours = 24 * 90
    hh = pd.date_range("2023-10-01", periods=2 * n_hours, freq="30min", tz="UTC")
    base = np.full(2 * n_hours, 30_000.0)
    rng = np.random.default_rng(7)
    local = hh.tz_convert("Europe/London")
    retrieved = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp_utc": hh,
            "timestamp_local": local,
            "settlement_date": local.normalize().date,
            "settlement_period": np.tile(np.arange(1, 49, dtype="int8"), n_hours // 24),
            "demand_forecast_mw": (base + rng.normal(0, 200, 2 * n_hours)).astype("int32"),
            "demand_outturn_mw": (base + rng.normal(0, 50, 2 * n_hours)).astype("int32"),
            "ape_percent": rng.normal(0.0, 0.5, 2 * n_hours).astype("float32"),
            "retrieved_at_utc": [retrieved] * (2 * n_hours),
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(NESO_SCHEMA, safe=True)
    pq.write_table(table, tmp_path / "neso_forecast.parquet")

    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Per-fold metrics" in out
    assert "Benchmark comparison" in out
    # Three-way row identifiers should appear in the benchmark section.
    for label in ("naive", "linear", "neso"):
        assert label in out


# ---------------------------------------------------------------------------
# Subprocess smoke — confirms ``python -m bristol_ml.train`` wiring works
# ---------------------------------------------------------------------------


def test_train_cli_help_exits_zero_via_subprocess() -> None:
    """``python -m bristol_ml.train --help`` exits 0 with usage text on stdout.

    This covers the ``__main__`` wiring in ``train.py`` — the in-process
    tests above call ``_cli_main`` directly and do not exercise the
    ``if __name__ == "__main__"`` branch.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Train and evaluate" in result.stdout


# ---------------------------------------------------------------------------
# Stage 5 T5 - _resolve_feature_set unit tests
# Plan: docs/plans/active/05-calendar-features.md lines 330-335
# ---------------------------------------------------------------------------

# Minimal AppConfig construction for resolver tests — bypasses Hydra.
# Only cfg.features needs to be populated; the resolver does not inspect
# ingestion, evaluation, or model sections.


def _make_minimal_app_config_weather_only(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Construct an AppConfig with weather_only populated, weather_calendar=None."""

    from conf._schemas import (
        AppConfig,
        EvaluationGroup,
        FeatureSetConfig,
        FeaturesGroup,
        IngestionGroup,
        ProjectConfig,
    )

    return AppConfig(
        project=ProjectConfig(name="test_wo", seed=0),
        ingestion=IngestionGroup(),
        features=FeaturesGroup(
            weather_only=FeatureSetConfig(
                name="weather_only",
                cache_dir=tmp_path,
                cache_filename="weather_only.parquet",
            ),
            weather_calendar=None,
        ),
        evaluation=EvaluationGroup(),
    )


def _make_minimal_app_config_weather_calendar(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Construct an AppConfig with weather_calendar populated, weather_only=None."""
    from conf._schemas import (
        AppConfig,
        EvaluationGroup,
        FeatureSetConfig,
        FeaturesGroup,
        IngestionGroup,
        ProjectConfig,
    )

    return AppConfig(
        project=ProjectConfig(name="test_wc", seed=0),
        ingestion=IngestionGroup(),
        features=FeaturesGroup(
            weather_only=None,
            weather_calendar=FeatureSetConfig(
                name="weather_calendar",
                cache_dir=tmp_path,
                cache_filename="weather_calendar.parquet",
            ),
        ),
        evaluation=EvaluationGroup(),
    )


# --- Resolver tests ---------------------------------------------------------


def test_resolve_feature_set_weather_only(tmp_path: Path) -> None:
    """Plan T5 line 330: default-config path returns weather_only config and loader.

    Asserts:
    - Returned config is the same weather_only FeatureSetConfig object.
    - Returned loader is assembler.load (identity check).
    - Returned column names tuple == the 5 WEATHER_VARIABLE_COLUMNS names, in order.
    """
    from bristol_ml.features import assembler

    cfg = _make_minimal_app_config_weather_only(tmp_path)
    fset_cfg, load_fn, col_names = _resolve_feature_set(cfg)

    assert fset_cfg is cfg.features.weather_only, (
        "_resolve_feature_set must return the same weather_only config object."
    )
    assert load_fn is assembler.load, (
        "_resolve_feature_set must return assembler.load for the weather_only path."
    )
    expected_names = tuple(name for name, _ in WEATHER_VARIABLE_COLUMNS)
    assert col_names == expected_names, (
        f"Weather-only column names must be exactly {expected_names!r}; got {col_names!r}."
    )
    assert len(col_names) == 5, f"WEATHER_VARIABLE_COLUMNS has 5 entries; got {len(col_names)}."


def test_resolve_feature_set_weather_calendar(tmp_path: Path) -> None:
    """Plan T5 line 331: features=weather_calendar override path.

    Asserts:
    - Returned config is the same weather_calendar FeatureSetConfig object.
    - Returned loader is assembler.load_calendar (identity check).
    - Returned column names tuple == (weather_names..., calendar_names...) with length 49.
    """
    from bristol_ml.features import assembler

    cfg = _make_minimal_app_config_weather_calendar(tmp_path)
    fset_cfg, load_fn, col_names = _resolve_feature_set(cfg)

    assert fset_cfg is cfg.features.weather_calendar, (
        "_resolve_feature_set must return the same weather_calendar config object."
    )
    assert load_fn is assembler.load_calendar, (
        "_resolve_feature_set must return assembler.load_calendar for the calendar path."
    )
    weather_names = tuple(name for name, _ in WEATHER_VARIABLE_COLUMNS)
    calendar_names = tuple(name for name, _ in CALENDAR_VARIABLE_COLUMNS)
    expected_names = weather_names + calendar_names
    assert col_names == expected_names, (
        "Calendar column names must be weather_names + calendar_names; mismatch at first "
        "differing position."
    )
    assert len(col_names) == 49, (
        f"weather_calendar feature set must have 49 columns (5 weather + 44 calendar); "
        f"got {len(col_names)}."
    )


def test_resolve_feature_set_both_populated_raises(tmp_path: Path) -> None:
    """Plan T5 line 332: both feature sets populated raises ValueError.

    The error message must contain 'Exactly one of' and 'features=<name>'
    (the Hydra override hint).
    """
    from conf._schemas import (
        AppConfig,
        EvaluationGroup,
        FeatureSetConfig,
        FeaturesGroup,
        IngestionGroup,
        ProjectConfig,
    )

    cfg = AppConfig(
        project=ProjectConfig(name="test_both", seed=0),
        ingestion=IngestionGroup(),
        features=FeaturesGroup(
            weather_only=FeatureSetConfig(
                name="weather_only",
                cache_dir=tmp_path,
                cache_filename="weather_only.parquet",
            ),
            weather_calendar=FeatureSetConfig(
                name="weather_calendar",
                cache_dir=tmp_path,
                cache_filename="weather_calendar.parquet",
            ),
        ),
        evaluation=EvaluationGroup(),
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_feature_set(cfg)

    msg = str(exc_info.value)
    assert "Exactly one of" in msg, (
        f"ValueError message must contain 'Exactly one of'; got: {msg!r}"
    )
    assert "features=" in msg, (
        f"ValueError message must contain the Hydra override hint 'features='; got: {msg!r}"
    )


def test_resolve_feature_set_neither_populated_raises(tmp_path: Path) -> None:
    """Plan T5 line 333: neither feature set populated raises ValueError.

    Same ValueError pattern as the both-populated case.
    """
    from conf._schemas import (
        AppConfig,
        EvaluationGroup,
        FeaturesGroup,
        IngestionGroup,
        ProjectConfig,
    )

    cfg = AppConfig(
        project=ProjectConfig(name="test_neither", seed=0),
        ingestion=IngestionGroup(),
        features=FeaturesGroup(
            weather_only=None,
            weather_calendar=None,
        ),
        evaluation=EvaluationGroup(),
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_feature_set(cfg)

    msg = str(exc_info.value)
    assert "Exactly one of" in msg, (
        f"ValueError message must contain 'Exactly one of'; got: {msg!r}"
    )
    assert "features=" in msg, (
        f"ValueError message must contain the Hydra override hint 'features='; got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Stage 5 T5 — calendar-cache helper and fixtures
# ---------------------------------------------------------------------------


def _write_calendar_feature_cache(path: Path, n_hours: int = 24 * 50, seed: int = 99) -> Path:
    """Write a minimal calendar feature-table parquet matching CALENDAR_OUTPUT_SCHEMA.

    Synthetic but schema-conformant: n_hours of hourly UTC timestamps starting
    at 2023-01-02 (a bank holiday date so is_bank_holiday_ew fires on some rows),
    demand with a daily sine + noise, Gaussian weather, and calendar columns
    derived via derive_calendar from a hand-crafted holidays frame.

    We use a 50-day window (1200 rows) so min_train_periods=720 is satisfiable
    with at least one fold of test_len=168 remaining.
    """
    from bristol_ml.features.calendar import derive_calendar
    from bristol_ml.ingestion.holidays import OUTPUT_SCHEMA as HOL_SCHEMA

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_hours, freq="1h", tz="UTC")
    nd = (
        30_000 + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 200, n_hours)
    ).astype("int32")
    tsd = (nd + 3_000).astype("int32")
    weather = {
        name: rng.normal(loc=10, scale=3, size=n_hours).astype("float32")
        for name, _ in WEATHER_VARIABLE_COLUMNS
    }
    retrieved_at = pd.Timestamp("2024-01-01T00:00:00Z")
    weather_frame = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "nd_mw": nd,
            "tsd_mw": tsd,
            **weather,
            "neso_retrieved_at_utc": [retrieved_at] * n_hours,
            "weather_retrieved_at_utc": [retrieved_at] * n_hours,
        }
    )

    # Build a minimal holidays DataFrame using the ingestion OUTPUT_SCHEMA.
    # Cover the 2023 window with a representative set of entries so that the
    # is_bank_holiday_ew and related columns fire correctly on the data range.
    holidays_retrieved = pd.Timestamp("2023-12-01T00:00:00Z")
    holidays_rows = [
        {
            "date": date(2023, 1, 2),
            "division": "england-and-wales",
            "title": "New Year's Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": holidays_retrieved,
        },
        {
            "date": date(2023, 1, 2),
            "division": "scotland",
            "title": "2nd January",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": holidays_retrieved,
        },
        {
            "date": date(2023, 1, 2),
            "division": "northern-ireland",
            "title": "New Year's Day (substitute day)",
            "notes": "",
            "bunting": True,
            "retrieved_at_utc": holidays_retrieved,
        },
        {
            "date": date(2023, 4, 7),
            "division": "england-and-wales",
            "title": "Good Friday",
            "notes": "",
            "bunting": False,
            "retrieved_at_utc": holidays_retrieved,
        },
        {
            "date": date(2023, 4, 7),
            "division": "scotland",
            "title": "Good Friday",
            "notes": "",
            "bunting": False,
            "retrieved_at_utc": holidays_retrieved,
        },
        {
            "date": date(2023, 4, 7),
            "division": "northern-ireland",
            "title": "Good Friday",
            "notes": "",
            "bunting": False,
            "retrieved_at_utc": holidays_retrieved,
        },
    ]
    holidays_df = pd.DataFrame(holidays_rows)
    hol_table = pa.Table.from_pandas(holidays_df, preserve_index=False).cast(HOL_SCHEMA, safe=True)
    holidays_df = hol_table.to_pandas()

    # derive_calendar appends the 44 calendar columns to the weather frame.
    cal_frame = derive_calendar(weather_frame, holidays_df)

    # Append the holidays_retrieved_at_utc provenance scalar.
    cal_frame["holidays_retrieved_at_utc"] = holidays_retrieved

    # Cast to the canonical CALENDAR_OUTPUT_SCHEMA.
    table = pa.Table.from_pandas(cal_frame, preserve_index=False).cast(
        CALENDAR_OUTPUT_SCHEMA, safe=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def warm_calendar_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate a warm calendar feature-table cache and point Hydra at it.

    Mirrors the ``warm_feature_cache`` fixture for the weather_only schema.
    The parquet lands at ``tmp_path/weather_calendar.parquet`` (matching the
    default ``cache_filename`` for the weather_calendar feature set).
    """
    monkeypatch.setenv("BRISTOL_ML_CACHE_DIR", str(tmp_path))
    _write_calendar_feature_cache(tmp_path / "weather_calendar.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# Stage 5 T5 - CLI integration tests
# Plan: docs/plans/active/05-calendar-features.md lines 334-335
# ---------------------------------------------------------------------------


def test_train_cli_features_override_swaps_feature_set(
    warm_calendar_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T5 line 334: features=weather_calendar swaps the feature set in the CLI.

    Runs _cli_main in-process with the calendar cache warm.  Asserts:
    - Exit code is 0.
    - Stdout contains the per-fold metrics header ("Per-fold metrics for model=linear").
    - Stdout contains "weather-calendar" (the metadata.name substring that signals
      the calendar feature set was selected and _NamedLinearModel applied the
      correct override).
    """
    exit_code = _cli_main(
        [
            "features=weather_calendar",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0, f"CLI must exit 0 on a warm calendar cache; got {exit_code}."
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=linear" in out, (
        "Per-fold header must appear in stdout when features=weather_calendar."
    )
    assert "weather-calendar" in out, (
        "metadata.name must include 'weather-calendar' when features=weather_calendar is set; "
        f"stdout was:\n{out}"
    )


def test_train_cli_weather_only_still_works(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan T5 line 335: regression guard — plain _cli_main([]) still works after T5 changes.

    Uses the existing warm_feature_cache fixture (weather_only schema).  Asserts:
    - Exit code is 0.
    - Stdout contains "Per-fold metrics for model=linear".
    - Stdout contains "weather-only" (the metadata.name substring confirms
      _NamedLinearModel applies "linear-ols-weather-only" for the default path).
    """
    exit_code = _cli_main(
        [
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=168",
            "evaluation.rolling_origin.step=168",
        ]
    )

    assert exit_code == 0, f"CLI must exit 0 on a warm weather-only cache; got {exit_code}."
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=linear" in out, (
        "Per-fold header must appear in stdout for the default weather_only path."
    )
    assert "weather-only" in out, (
        "metadata.name must include 'weather-only' for the default feature-set path; "
        f"stdout was:\n{out}"
    )


# ---------------------------------------------------------------------------
# Stage 7 T6 — SARIMAX dispatcher wiring (train CLI)
# Plan: docs/plans/active/07-sarimax.md §6 Task T6
# ---------------------------------------------------------------------------


def test_train_build_model_dispatches_sarimax_config(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 7 T6 (AC-6): ``train.py``'s inline dispatcher handles ``SarimaxConfig``.

    ``train.py`` does not expose a standalone ``_build_model_from_config``
    function — the model-dispatch logic is inlined inside ``_cli_main``.
    The equivalent unit-level test therefore exercises that inline branch via an
    end-to-end ``_cli_main`` call with ``model=sarimax``, and asserts that the
    output contains the ``"Per-fold metrics for model=sarimax"`` banner.  A
    missing or wrong dispatch would produce exit code 2/3 or a different banner.

    The ``seasonal_order=[0,0,0,24]`` and ``weekly_fourier_harmonics=0`` overrides
    disable the heavy seasonal and Fourier components so the SARIMA fit completes
    quickly on the 90-day synthetic cache.

    Plan clause: docs/plans/active/07-sarimax.md §6 Task T6 —
    ``test_train_build_model_dispatches_sarimax_config``.
    """
    exit_code = _cli_main(
        [
            "model=sarimax",
            "model.order=[1,0,0]",
            "model.seasonal_order=[0,0,0,24]",
            "model.weekly_fourier_harmonics=0",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=24",
            "evaluation.rolling_origin.step=720",
            "evaluation.rolling_origin.fixed_window=true",
        ]
    )

    assert exit_code == 0, (
        f"_cli_main must exit 0 when model=sarimax is selected; got {exit_code} (Stage 7 T6)."
    )
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=sarimax" in out, (
        "stdout must contain 'Per-fold metrics for model=sarimax' confirming the "
        f"inline SarimaxConfig branch was taken; stdout was:\n{out} (Stage 7 T6 AC-6)."
    )


def test_train_cli_runs_with_model_sarimax(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 7 T6 (AC-6, AC-11): ``train`` CLI exits 0 and prints SARIMAX per-fold table.

    Full-path integration smoke for the ``model=sarimax`` CLI path.  Uses the
    same ``warm_feature_cache`` fixture as the other ``test_train_cli_*`` tests
    and the same override recipe as ``test_harness_cli_runs_with_model_sarimax``
    in ``tests/unit/evaluation/test_harness.py``.

    Asserts:
    - Exit code is 0 (the SARIMAX dispatcher ran without error).
    - Stdout contains ``"Per-fold metrics for model=sarimax"`` — the substring
      that ``test_train_cli_model_swap`` uses for naive; applying the same check
      here ensures the banner is correctly parameterised for SARIMAX.

    Plan clause: docs/plans/active/07-sarimax.md §6 Task T6 —
    ``test_train_cli_runs_with_model_sarimax``.
    """
    exit_code = _cli_main(
        [
            "model=sarimax",
            "model.order=[1,0,0]",
            "model.seasonal_order=[0,0,0,24]",
            "model.weekly_fourier_harmonics=0",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=24",
            "evaluation.rolling_origin.step=720",
            "evaluation.rolling_origin.fixed_window=true",
        ]
    )

    assert exit_code == 0, (
        f"_cli_main must exit 0 for model=sarimax on a warm feature cache; "
        f"got exit_code={exit_code} (Stage 7 T6 AC-11)."
    )
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=sarimax" in out, (
        "stdout must contain 'Per-fold metrics for model=sarimax'; "
        f"stdout was:\n{out} (Stage 7 T6 AC-11)."
    )


# ---------------------------------------------------------------------------
# Stage 8 T6 — ScipyParametric dispatcher wiring (train CLI)
# Plan: docs/plans/active/08-scipy-parametric.md §6 Task T6
# ---------------------------------------------------------------------------


def test_train_cli_runs_with_model_scipy_parametric(
    warm_feature_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 8 T6 (AC-7): ``train`` CLI exits 0 and prints ScipyParametric per-fold table.

    Full-path integration smoke for the ``model=scipy_parametric`` CLI path.
    Uses the same ``warm_feature_cache`` fixture as the other
    ``test_train_cli_*`` tests.  The ``diurnal_harmonics=1`` and
    ``weekly_harmonics=1`` overrides reduce the number of Fourier terms so
    ``curve_fit`` completes quickly on the 90-day synthetic cache.
    Tight rolling-origin parameters yield a single test fold.

    Asserts:
    - Exit code is 0 (the ScipyParametric dispatcher ran without error).
    - Stdout contains ``"Per-fold metrics for model=scipy_parametric"`` —
      confirming the inline ``ScipyParametricConfig`` branch was taken and the
      per-fold table header is correctly parameterised.

    Plan clause: docs/plans/active/08-scipy-parametric.md §6 Task T6 —
    ``test_train_cli_runs_with_model_scipy_parametric``.
    """
    exit_code = _cli_main(
        [
            "model=scipy_parametric",
            "model.diurnal_harmonics=1",
            "model.weekly_harmonics=1",
            "evaluation.rolling_origin.min_train_periods=720",
            "evaluation.rolling_origin.test_len=24",
            "evaluation.rolling_origin.step=720",
            "evaluation.rolling_origin.fixed_window=true",
        ]
    )

    assert exit_code == 0, (
        f"_cli_main must exit 0 for model=scipy_parametric on a warm feature cache; "
        f"got exit_code={exit_code} (Stage 8 T6 AC-7)."
    )
    out = capsys.readouterr().out
    assert "Per-fold metrics for model=scipy_parametric" in out, (
        "stdout must contain 'Per-fold metrics for model=scipy_parametric'; "
        f"stdout was:\n{out} (Stage 8 T6 AC-7)."
    )
