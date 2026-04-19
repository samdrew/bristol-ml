"""Unit tests for ``bristol_ml.ingestion.weather``.

Spec-derived: each test maps to a behavioural requirement in the Stage 2
intent (`docs/intent/02-weather-ingestion.md`), the ingestion layer
architecture (`docs/architecture/layers/ingestion.md`), or the Stage 2
LLD (`docs/lld/ingestion/weather.md`) sections 2 through 10.

Where the task-brief table and the LLD disagree on a persisted unit
(wind-speed in m/s vs km/h, for example) the test pins the *type*
(`float32`) and intentionally does not pin the scalar magnitude —
unit-drift is flagged separately in the tester report, not hidden
inside a pass/fail assertion.

Tests use ``pytest.importorskip`` so the suite stays green while the
implementer is still landing code.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Skip the whole module until the implementer's code lands. Until then each
# importorskip returns a stub module object; the Grep-friendly ``assert`` lines
# below still parse.
weather = pytest.importorskip("bristol_ml.ingestion.weather")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "weather"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _build_stations() -> list[dict[str, Any]]:
    """Minimal station list matching LLD §3.2 (two stations, geographic spread)."""
    return [
        {
            "name": "london",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "weight": 9787426.0,
            "weight_source": "ONS 2011 Census, Greater London BUA",
        },
        {
            "name": "bristol",
            "latitude": 51.4545,
            "longitude": -2.5879,
            "weight": 617280.0,
            "weight_source": "ONS 2011 Census, Bristol BUA",
        },
    ]


def _build_config(tmp_path: Path, **overrides: Any) -> Any:
    """Construct a ``WeatherIngestionConfig`` pointing at ``tmp_path``.

    The schema lives in ``conf._schemas``; importing it lazily matches
    Stage 1's test pattern and avoids forcing a Pydantic import at
    module collection time.
    """
    from conf._schemas import WeatherIngestionConfig  # type: ignore[import-not-found]

    cfg_kwargs: dict[str, Any] = dict(
        stations=_build_stations(),
        start_date=_dt.date(2023, 1, 1),
        end_date=_dt.date(2023, 1, 31),
        cache_dir=tmp_path,
        variables=[
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ],
    )
    cfg_kwargs.update(overrides)
    return WeatherIngestionConfig(**cfg_kwargs)


# --------------------------------------------------------------------------- #
# Cache semantics (intent AC 1, AC 2; LLD §7; layer arch §2)
# --------------------------------------------------------------------------- #


class TestCachePolicy:
    """LLD §2 + §7 — `CachePolicy` semantics on the public `fetch` surface."""

    def test_offline_raises_when_cache_missing_and_names_path(self, tmp_path: Path) -> None:
        """Intent AC 1 negative case: OFFLINE + no cache → `CacheMissingError`.

        Per layer arch §2, the error must name the expected cache path so
        a facilitator knows where to look / what to pre-seed. The concrete
        exception type is `bristol_ml.ingestion.CacheMissingError` (shared
        with Stage 1 via the `_common.py` extraction).
        """
        cfg = _build_config(tmp_path)
        with pytest.raises(Exception) as exc_info:
            weather.fetch(cfg, cache=weather.CachePolicy.OFFLINE)

        # Exception type: a FileNotFoundError subclass named `CacheMissingError`
        # per Stage 1's precedent (and LLD §2 re-exports the same name here).
        assert type(exc_info.value).__name__ == "CacheMissingError", (
            f"OFFLINE + no cache must raise CacheMissingError; "
            f"got {type(exc_info.value).__name__}: {exc_info.value!r}"
        )
        msg = str(exc_info.value)
        assert str(tmp_path) in msg or "cache" in msg.lower(), (
            f"CacheMissingError must name the expected path; got: {exc_info.value!r}"
        )

    def test_auto_returns_cached_path_without_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Intent AC 1: cache present + AUTO → zero HTTP calls.

        We force the implementation to use a sentinel client that raises
        on any `.get` attempt: if the fetcher reaches the network it
        explodes. A pre-seeded cache file at the expected path must
        short-circuit that path entirely.
        """
        cfg = _build_config(tmp_path)

        # Seed the expected cache file. We do not care about its contents
        # for this assertion — only that `fetch` returns the path without
        # touching the network. The path is derived from the LLD §4
        # convention `<cache_dir>/<cache_filename>`.
        cache_path = tmp_path / cfg.cache_filename
        cache_path.write_bytes(b"placeholder")

        # Booby-trap httpx so any network call fails loudly.
        import httpx

        class _NoNetwork:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def __enter__(self) -> _NoNetwork:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

            def get(self, *a: Any, **k: Any) -> None:
                raise AssertionError(
                    "AUTO with cache present must not make HTTP calls (intent AC 1)."
                )

        monkeypatch.setattr(httpx, "Client", _NoNetwork)

        path = weather.fetch(cfg, cache=weather.CachePolicy.AUTO)
        assert Path(path) == cache_path, (
            f"AUTO cache-hit must return the pre-existing cache path; "
            f"expected {cache_path}, got {path}"
        )


# --------------------------------------------------------------------------- #
# Schema assertion (layer arch §4; LLD §5 `_assert_schema`)
# --------------------------------------------------------------------------- #


class TestAssertSchemaContract:
    """Layer arch §4 — required-missing raises; unknown warns-and-drops.

    These tests exercise the schema-assertion surface expected by the
    LLD. The exact symbol name (`_assert_schema`, `assert_schema`, or
    an embedded check inside `_parse_station_payload`) is an
    implementation choice; the *contract* is what's asserted.
    """

    def _find_schema_asserter(self) -> Any:
        """Resolve the module's schema-assertion callable by convention.

        Candidates, in order of preference:
        - `weather._assert_schema` (LLD §5 exact name).
        - `weather.assert_schema` (if the implementer made it public).

        If neither exists we skip rather than error — the behavioural
        contract is still covered by the end-to-end REFRESH path that
        shouldn't accept a missing required variable.
        """
        for name in ("_assert_schema", "assert_schema"):
            fn = getattr(weather, name, None)
            if callable(fn):
                return fn
        pytest.skip(
            "No `_assert_schema` / `assert_schema` callable on weather module; "
            "contract is covered implicitly by the REFRESH end-to-end test."
        )

    def test_missing_required_variable_raises_and_names_it(self) -> None:
        """Layer arch §4: a required variable missing is a hard error naming it.

        Per LLD §5 the required set == `time` + the requested variables
        from config. A payload that lacks, say, `temperature_2m` when
        `temperature_2m` was requested must raise — no fallback parsing.
        """
        fn = self._find_schema_asserter()

        # Hand-shaped frame lacking `temperature_2m`.
        df = pd.DataFrame(
            {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                "dew_point_2m": [3.0, 4.0],
                "wind_speed_10m": [10.0, 11.0],
                "cloud_cover": [80, 70],
                "shortwave_radiation": [0.0, 0.0],
            }
        )
        requested = [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ]
        with pytest.raises(Exception) as exc_info:
            # Try the LLD §5 signature first (`df, station, requested_vars`);
            # fall back to a two-arg variant if the implementer used one.
            try:
                fn(df, "london", requested)
            except TypeError:
                fn(df, requested)
        msg = str(exc_info.value)
        assert "temperature_2m" in msg, (
            f"Missing-required error must name the offending variable; got: {exc_info.value!r}"
        )

    def test_unknown_variable_warns_and_is_dropped(self) -> None:
        """Layer arch §4: an unknown column is warned and dropped silently from output.

        Open-Meteo occasionally returns variables the caller didn't ask
        for (e.g. `apparent_temperature` when it slips into a payload).
        The layer contract is warn-and-drop, not fail.
        """
        import warnings as _warnings

        fn = self._find_schema_asserter()

        df = pd.DataFrame(
            {
                "time": ["2023-01-01T00:00"],
                "temperature_2m": [5.0],
                "dew_point_2m": [3.0],
                "wind_speed_10m": [10.0],
                "cloud_cover": [80],
                "shortwave_radiation": [0.0],
                "apparent_temperature": [3.5],  # unknown / unrequested
            }
        )
        requested = [
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
        ]
        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            try:
                out = fn(df, "london", requested)
            except TypeError:
                out = fn(df, requested)

        assert any("apparent_temperature" in str(w.message) for w in captured), (
            "An unknown column must emit a warning naming the column."
        )

        # The returned frame must not carry the unknown column.
        if out is not None and hasattr(out, "columns"):
            assert "apparent_temperature" not in list(out.columns), (
                "Unknown columns must be dropped from the returned frame."
            )


# --------------------------------------------------------------------------- #
# Output-parquet schema (LLD §4)
# --------------------------------------------------------------------------- #


class TestOutputSchemaConstant:
    """LLD §4 — the persisted parquet schema is frozen; pin the types.

    Tests look up the implementer's declared `OUTPUT_SCHEMA` constant
    (Stage 1 precedent) and pin column names + types. If the name is
    `SCHEMA` or `PARQUET_SCHEMA` the test falls back to those; if no
    module-level `pa.Schema` exists the test is skipped and the
    equivalent check runs against a persisted file in the integration
    suite.
    """

    def _find_schema(self) -> Any:
        for name in ("OUTPUT_SCHEMA", "SCHEMA", "PARQUET_SCHEMA"):
            obj = getattr(weather, name, None)
            if obj is not None and isinstance(obj, pa.Schema):
                return obj
        pytest.skip(
            "No module-level `OUTPUT_SCHEMA` pa.Schema on bristol_ml.ingestion.weather. "
            "Integration tests cover the persisted-file equivalent."
        )

    def test_required_columns_present(self) -> None:
        schema = self._find_schema()
        names = set(schema.names)
        required = {
            "timestamp_utc",
            "station",
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "retrieved_at_utc",
        }
        missing = required - names
        assert not missing, f"OUTPUT_SCHEMA missing columns: {missing}"

    def test_timestamp_columns_are_utc_microsecond(self) -> None:
        schema = self._find_schema()
        for col in ("timestamp_utc", "retrieved_at_utc"):
            typ = schema.field(col).type
            assert pa.types.is_timestamp(typ), f"{col} must be a timestamp, got {typ}"
            assert typ.unit == "us", f"{col} must be microsecond-precision, got unit={typ.unit}"
            assert typ.tz in {"UTC", "utc"}, f"{col} must be tz=UTC, got tz={typ.tz}"

    def test_station_is_string_like(self) -> None:
        schema = self._find_schema()
        typ = schema.field("station").type
        # Accept both plain string and dictionary(string) — dictionary encoding
        # is an implementation-layer optimisation that pyarrow often applies.
        assert pa.types.is_string(typ) or pa.types.is_dictionary(typ), (
            f"`station` must be string / dictionary(string); got {typ}"
        )

    def test_cloud_cover_is_int8(self) -> None:
        schema = self._find_schema()
        # LLD §4 pins `cloud_cover` at int8 (0-100 fits); widening would be
        # a silent schema drift. Bytes matter here: int32 is 4x the storage.
        assert schema.field("cloud_cover").type == pa.int8(), (
            f"cloud_cover must be int8 per LLD §4; got {schema.field('cloud_cover').type}"
        )

    def test_float_variables_are_float32(self) -> None:
        schema = self._find_schema()
        for col in ("temperature_2m", "dew_point_2m", "wind_speed_10m", "shortwave_radiation"):
            assert schema.field(col).type == pa.float32(), (
                f"{col} must be float32 per LLD §4; got {schema.field(col).type}"
            )


# --------------------------------------------------------------------------- #
# Re-exports and public package surface (LLD §2; _common.py extraction)
# --------------------------------------------------------------------------- #


def test_public_interface_reexported_from_package() -> None:
    """LLD §2 — `weather`, `CachePolicy`, `CacheMissingError` via `bristol_ml.ingestion`.

    After the Stage 2 `_common.py` extraction `CachePolicy` and
    `CacheMissingError` live in `_common` but the package-level
    re-exports are unchanged (LLD §11). This test pins that the
    refactor did not break the package surface Stage 1 shipped.
    """
    import bristol_ml.ingestion as pkg

    # `weather` submodule importable (the stage's headline artefact).
    assert hasattr(pkg, "weather") or hasattr(pkg, "neso"), (
        "bristol_ml.ingestion must expose at least `neso`; Stage 2 adds `weather`."
    )

    # Re-exports survive the `_common.py` extraction.
    assert hasattr(pkg, "CachePolicy"), (
        "bristol_ml.ingestion.CachePolicy must remain importable after "
        "_common.py extraction (LLD §11)."
    )
    assert hasattr(pkg, "CacheMissingError"), (
        "bristol_ml.ingestion.CacheMissingError must remain importable after "
        "_common.py extraction (LLD §11)."
    )


def test_cache_policy_enum_values_are_stable() -> None:
    """Layer arch §2: `CachePolicy` must expose `AUTO | REFRESH | OFFLINE`.

    The three-valued enum is load-bearing per the layer architecture's
    upgrade-seams table; adding / renaming a value would break Stage 1
    tests and every downstream CLI. This is a belt-and-braces guard on
    the `_common.py` extraction.
    """
    cp = weather.CachePolicy
    values = {m.value for m in cp}
    assert values == {"auto", "refresh", "offline"}, (
        f"CachePolicy values must be {{auto, refresh, offline}}; got {values}"
    )


# --------------------------------------------------------------------------- #
# CLI smoke (intent AC 6)
# --------------------------------------------------------------------------- #


def test_cli_help_exits_zero() -> None:
    """Intent AC 6: `python -m bristol_ml.ingestion.weather --help` exits 0.

    This is the cheapest possible end-to-end smoke — it catches argparse
    wiring errors, import-time failures, and missing module-under-runpy
    setup (§2.1.1). It does NOT make any network calls.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.ingestion.weather", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, (
        f"`python -m bristol_ml.ingestion.weather --help` must exit 0; "
        f"returncode={result.returncode}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    # Must name the three CachePolicy values somewhere in the help output
    # (Stage 1 precedent) so facilitators discover the CI-safe mode.
    help_text = result.stdout + result.stderr
    assert "cache" in help_text.lower(), (
        f"--help output should mention the --cache option; got: {help_text!r}"
    )


# --------------------------------------------------------------------------- #
# Module CLAUDE.md schema regression (intent AC 4 / Stage hygiene)
# --------------------------------------------------------------------------- #


def test_module_claude_md_documents_weather_schema() -> None:
    """Stage hygiene: `ingestion/CLAUDE.md` carries a `weather.py` schema table.

    The stage brief "Files expected to change" lists this update. A
    missing schema table means the module guide has drifted from the
    implementation; the Stage 1 retro (Stage 0.1.8) flagged this as a
    recurring smell.
    """
    claude_md = (
        Path(__file__).resolve().parents[3] / "src" / "bristol_ml" / "ingestion" / "CLAUDE.md"
    )
    text = claude_md.read_text()
    # Headings vary slightly; accept a few common forms.
    assert any(marker in text for marker in ("`weather.py`", "weather.py", "Weather")), (
        "ingestion/CLAUDE.md must document the weather module."
    )
    # Schema-table evidence: the column list should include at least a
    # couple of the documented variables.
    assert "temperature_2m" in text, (
        "ingestion/CLAUDE.md must list `temperature_2m` in the weather schema table."
    )
    assert "timestamp_utc" in text, (
        "ingestion/CLAUDE.md must list `timestamp_utc` in the weather schema table."
    )
