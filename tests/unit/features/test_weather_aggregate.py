"""Unit tests for ``bristol_ml.features.weather.national_aggregate``.

Maps to Stage 2 acceptance criteria:

- **AC 3.** "The national aggregation accepts any subset of the
  configured station list, so a demo can run with fewer stations to
  show the effect."
- **AC 6.** "A test for the aggregation that asserts equal weights on
  identical inputs yield the identity."

Plus invariants drawn from LLD §6 (weighted-mean idiom, NaN-safety,
per-variable behaviour).

Stage 2 spec-drift note (contract)
----------------------------------
LLD §2 sketches a config-based signature
(``national_aggregate(df, config, *, stations=None, variables=None)``);
the shipped implementation takes ``weights: Mapping[str, float]``
instead. The features layer is deliberately decoupled from
``conf._schemas`` and the caller is responsible for translating a
``WeatherIngestionConfig`` into a plain ``dict[str, float]``. This
drift was accepted as a Stage 2 decision — the tests below encode the
shipped ``Mapping[str, float]`` contract and the subset/renormalise/
raise semantics documented in ``features/weather.py``:

- stations in ``weights`` but absent from ``df`` → ``ValueError``;
- stations in ``df`` but absent from ``weights`` → silent exclusion;
- remaining weights are renormalised within the intersection
  (``sum(w_i * v_i) / sum(w_i)``);
- a NaN at one station drops that station from that (hour, variable)
  slot, with the surviving weights renormalised for that slot.

The canonical idiom for translating a config is::

    weights = {s.name: s.weight for s in cfg.stations}
    result = national_aggregate(df, weights)

``_build_config`` is retained because fetch/load tests in the
ingestion suite still need a ``WeatherIngestionConfig`` — it is not
passed to ``national_aggregate`` here.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

# Skip the whole module until the implementer's `features/weather.py` lands.
feat_weather = pytest.importorskip("bristol_ml.features.weather")


FIXTURE_CSV = Path(__file__).resolve().parents[2] / "fixtures" / "weather" / "toy_stations.csv"

VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_toy_frame() -> pd.DataFrame:
    """Load the hand-crafted station x hour x variable fixture.

    Rows look like: `(timestamp_utc, station, latitude, longitude,
    temperature_2m, dew_point_2m, wind_speed_10m, cloud_cover,
    shortwave_radiation)`. Parsed with tz-aware UTC timestamps to match
    the persisted parquet schema (LLD §4).
    """
    df = pd.read_csv(FIXTURE_CSV)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _build_config(stations: list[dict[str, Any]], tmp_path: Path) -> Any:
    """Construct a `WeatherIngestionConfig` carrying the given station weights.

    Retained for fetch/load tests under ``tests/unit/ingestion/`` that
    genuinely need the Pydantic object. ``national_aggregate`` takes a
    ``Mapping[str, float]`` and should not be passed a config — use
    :func:`_weights_from_stations` instead.
    """
    from conf._schemas import WeatherIngestionConfig  # type: ignore[import-not-found]

    return WeatherIngestionConfig(
        stations=stations,
        start_date=_dt.date(2023, 1, 1),
        end_date=_dt.date(2023, 1, 31),
        cache_dir=tmp_path,
    )


def _weights_from_stations(stations: list[dict[str, Any]]) -> dict[str, float]:
    """Project the station-dict list onto the shipped ``Mapping[str, float]``.

    Mirrors the production idiom in ``features/weather.py``'s CLI::

        weights = {s.name: s.weight for s in cfg.stations}
    """
    return {s["name"]: float(s["weight"]) for s in stations}


def _station(
    name: str,
    *,
    weight: float,
    lat: float = 51.5,
    lon: float = -1.0,
    source: str = "test",
) -> dict[str, Any]:
    return {
        "name": name,
        "latitude": lat,
        "longitude": lon,
        "weight": weight,
        "weight_source": source,
    }


# --------------------------------------------------------------------------- #
# AC 6 — equal weights on identical inputs yield the identity
# --------------------------------------------------------------------------- #


class TestEqualWeightsIdentity:
    """Intent AC 6: equal weights on identical inputs → the identity.

    Phrased as an algebraic invariant: if every station carries weight
    `w` and every station reports the same value `v` at some hour, the
    weighted mean for that hour equals `v` (independent of `w`).
    """

    def test_identical_values_collapse_to_that_value(self, tmp_path: Path) -> None:
        """Every station reports 10.0 °C with equal weight → output is 10.0."""
        stations = [
            _station("london", weight=1.0),
            _station("bristol", weight=1.0),
            _station("manchester", weight=1.0),
            _station("glasgow", weight=1.0),
        ]
        weights = _weights_from_stations(stations)

        rows = []
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        for s in stations:
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "station": s["name"],
                    **{v: 10.0 for v in VARIABLES if v != "cloud_cover"},
                    "cloud_cover": 50,
                }
            )
        df = pd.DataFrame(rows)

        out = feat_weather.national_aggregate(df, weights)

        # The aggregate output must carry one row for this hour.
        # Shape is wide (one column per variable) per LLD §6.
        assert timestamp in out.index or any(
            (out["timestamp_utc"] == timestamp)
            if "timestamp_utc" in out.columns
            else pd.Series([False])
        ), "Aggregate output must include the input hour."

        # Resolve the value for `temperature_2m` regardless of
        # index-vs-column layout.
        temp = self._scalar(out, timestamp, "temperature_2m")
        assert math.isclose(temp, 10.0, abs_tol=1e-9), (
            f"Equal-weights identity: equal 10.0 °C everywhere must yield 10.0; got {temp}"
        )

    def test_identity_holds_for_unequal_but_uniform_weights(self, tmp_path: Path) -> None:
        """Equal weights (same non-unit value) and identical inputs → identity.

        Belt-and-braces: the identity should not be accidentally true
        only for weight=1.0. Pick weight=7.5 and confirm it still
        collapses.
        """
        stations = [_station(f"s{i}", weight=7.5) for i in range(4)]
        weights = _weights_from_stations(stations)

        rows = []
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        for s in stations:
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "station": s["name"],
                    "temperature_2m": 3.3,
                    "dew_point_2m": 1.1,
                    "wind_speed_10m": 5.0,
                    "cloud_cover": 10,
                    "shortwave_radiation": 0.0,
                }
            )
        df = pd.DataFrame(rows)

        out = feat_weather.national_aggregate(df, weights)
        temp = self._scalar(out, timestamp, "temperature_2m")
        assert math.isclose(temp, 3.3, abs_tol=1e-9)

    @staticmethod
    def _scalar(agg: pd.DataFrame, timestamp: pd.Timestamp, variable: str) -> float:
        """Extract a scalar from the aggregate regardless of index shape.

        LLD §6 returns a wide frame indexed by `timestamp_utc`. If the
        implementer chose to reset the index, accommodate that variant.
        """
        if (
            isinstance(agg.index, pd.DatetimeIndex)
            or getattr(agg.index, "name", None) == "timestamp_utc"
        ):
            return float(agg.loc[timestamp, variable])
        row = agg[agg["timestamp_utc"] == timestamp]
        assert len(row) == 1, f"Expected exactly one aggregate row for {timestamp}, got {len(row)}"
        return float(row.iloc[0][variable])


# --------------------------------------------------------------------------- #
# AC 3 — subset of configured stations (shipped Mapping[str, float] contract)
# --------------------------------------------------------------------------- #


class TestStationSubset:
    """Intent AC 3: accepts any subset of the configured station list.

    The shipped ``national_aggregate(df, weights)`` encodes subset
    selection through the ``weights`` mapping itself — no separate
    ``stations=`` keyword. The contract (from
    ``features/weather.py``'s docstring):

    - A station in ``weights`` but **absent from the frame** raises
      ``ValueError``: the caller asked for a weight on a signal that
      is not available; silent dropping would produce a subtly
      different weighted mean than the caller intended.
    - A station in the frame but **absent from ``weights``** is
      silently excluded (AC 3's primary case: narrow the mapping to
      narrow the aggregate).
    - The remaining weights are renormalised to sum-to-one on the
      intersection, so ``sum(w_i v_i) / sum(w_i)`` over the subset.
    """

    def test_full_mapping_matches_passing_only_frame_stations(self, tmp_path: Path) -> None:
        """Passing weights for every station in the frame = the canonical case.

        Sanity bridge: when the mapping covers the full frame exactly,
        the renormalised weighted mean equals the straightforward
        ``sum(w_i v_i) / sum(w_i)``.
        """
        stations = [
            _station("london", weight=8.0),
            _station("bristol", weight=2.0),
            _station("manchester", weight=2.0),
            _station("glasgow", weight=1.0),
        ]
        weights = _weights_from_stations(stations)
        df = _load_toy_frame()

        out = feat_weather.national_aggregate(df, weights)
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        temp = _resolve_scalar(out, timestamp, "temperature_2m")
        expected = (8.0 * 5.0 + 2.0 * 6.0 + 2.0 * 4.0 + 1.0 * 2.0) / (8.0 + 2.0 + 2.0 + 1.0)
        assert math.isclose(temp, expected, abs_tol=1e-6)

    def test_two_station_subset_runs_and_reflects_only_subset(self, tmp_path: Path) -> None:
        """Subset dict ``{"london": 8, "bristol": 2}`` aggregates over those two only.

        The toy fixture at t=00:00 gives London 5.0 °C and Bristol
        6.0 °C with weights 8 and 2. Renormalised within the subset
        the weighted mean is ``(8*5 + 2*6) / (8 + 2) = 52/10 = 5.2``.
        The full-config mean (weights 8/2/2/1 and values 5/6/4/2)
        would be ``62/13 ≈ 4.769``; the subset result must track the
        former, not the latter (AC 3 — "accepts any subset … to show
        the effect").
        """
        df = _load_toy_frame()
        weights_subset = {"london": 8.0, "bristol": 2.0}

        out = feat_weather.national_aggregate(df, weights_subset)
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        temp = _resolve_scalar(out, timestamp, "temperature_2m")
        expected = (8.0 * 5.0 + 2.0 * 6.0) / (8.0 + 2.0)
        assert math.isclose(temp, expected, abs_tol=1e-6), (
            f"Two-station subset at t=00:00: expected weighted mean {expected:.4f}, got {temp:.4f}."
        )

    def test_single_station_subset_equals_that_station(self, tmp_path: Path) -> None:
        """A one-station subset collapses to that station's values.

        Smoke check of the subset machinery: a singleton weights dict
        makes the aggregator degenerate into station-identity.
        Exercises AC 3 at its smallest non-trivial case.
        """
        df = _load_toy_frame()
        weights_subset = {"manchester": 2.0}

        out = feat_weather.national_aggregate(df, weights_subset)
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        temp = _resolve_scalar(out, timestamp, "temperature_2m")
        # Manchester reports 4.0 at t=00:00.
        assert math.isclose(temp, 4.0, abs_tol=1e-9), (
            f"Single-station subset must equal the station's own value; got {temp}"
        )

    def test_station_missing_from_frame_raises(self, tmp_path: Path) -> None:
        """A weight named for a station not in the frame must raise ``ValueError``.

        Shipped contract rationale: if the caller asks for a weight on
        ``mars`` but the frame has no such station, silently dropping
        it would produce a weighted mean that differs subtly from
        what the caller asked for. A named ``ValueError`` is the
        right failure shape.
        """
        df = _load_toy_frame()
        weights_with_ghost = {"london": 8.0, "bristol": 2.0, "mars": 1.0}

        with pytest.raises(ValueError) as exc_info:
            feat_weather.national_aggregate(df, weights_with_ghost)
        msg = str(exc_info.value).lower()
        assert "mars" in str(exc_info.value) or "station" in msg, (
            f"Missing-from-frame error must identify the offender; got: {exc_info.value!r}"
        )

    def test_station_missing_from_weights_is_silently_excluded(self, tmp_path: Path) -> None:
        """AC 3 primary case: stations in the frame but not in ``weights`` drop out.

        The frame has london/bristol/manchester/glasgow. A weights
        dict covering only london+bristol must aggregate over the two
        weighted stations; the unweighted stations are neither error
        nor contaminant. The renormalised weighted mean must match
        the two-station subset exactly.
        """
        df = _load_toy_frame()
        weights_subset = {"london": 8.0, "bristol": 2.0}

        out = feat_weather.national_aggregate(df, weights_subset)
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        temp = _resolve_scalar(out, timestamp, "temperature_2m")
        expected = (8.0 * 5.0 + 2.0 * 6.0) / (8.0 + 2.0)
        assert math.isclose(temp, expected, abs_tol=1e-6), (
            f"Silent-exclusion: stations missing from weights must drop; expected {expected}, "
            f"got {temp}"
        )

    def test_equal_subset_weights_on_identical_inputs_identity(self, tmp_path: Path) -> None:
        """Identity under subset + renormalisation: constant → constant.

        Belt-and-braces on AC 6 applied through the subset path: pick
        a two-station subset with equal weights against identical
        values; the renormalised weighted mean must still be the
        constant (the renormalisation does not break the identity).
        """
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        df = pd.DataFrame(
            [
                {"timestamp_utc": timestamp, "station": "london", "temperature_2m": 4.2},
                {"timestamp_utc": timestamp, "station": "bristol", "temperature_2m": 4.2},
                {"timestamp_utc": timestamp, "station": "manchester", "temperature_2m": 4.2},
            ]
        )
        weights_subset = {"london": 3.0, "bristol": 3.0}

        out = feat_weather.national_aggregate(df, weights_subset)
        temp = _resolve_scalar(out, timestamp, "temperature_2m")
        assert math.isclose(temp, 4.2, abs_tol=1e-9), (
            f"Renormalised identity on equal subset weights failed; got {temp}"
        )


# --------------------------------------------------------------------------- #
# Per-variable behaviour (LLD §6)
# --------------------------------------------------------------------------- #


class TestPerVariableAggregation:
    """Every documented variable in LLD §4 must aggregate on the toy fixture."""

    @pytest.fixture
    def setup(self, tmp_path: Path) -> dict[str, Any]:
        stations = [
            _station("london", weight=8.0),
            _station("bristol", weight=2.0),
            _station("manchester", weight=2.0),
            _station("glasgow", weight=1.0),
        ]
        weights = _weights_from_stations(stations)
        df = _load_toy_frame()
        out = feat_weather.national_aggregate(df, weights)
        return {"out": out, "total_weight": 13.0}

    def test_temperature_2m_at_t0(self, setup: dict[str, Any]) -> None:
        """(8*5 + 2*6 + 2*4 + 1*2) / 13 = 62/13."""
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(setup["out"], timestamp, "temperature_2m")
        expected = (8 * 5.0 + 2 * 6.0 + 2 * 4.0 + 1 * 2.0) / setup["total_weight"]
        assert math.isclose(val, expected, abs_tol=1e-5), (
            f"temperature_2m at t=00:00: expected {expected}, got {val}"
        )

    def test_dew_point_2m_at_t0(self, setup: dict[str, Any]) -> None:
        """(8*3 + 2*4 + 2*2 + 1*0) / 13 = 36/13."""
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(setup["out"], timestamp, "dew_point_2m")
        expected = (8 * 3.0 + 2 * 4.0 + 2 * 2.0 + 1 * 0.0) / setup["total_weight"]
        assert math.isclose(val, expected, abs_tol=1e-5)

    def test_wind_speed_10m_at_t0(self, setup: dict[str, Any]) -> None:
        """(8*10 + 2*12 + 2*15 + 1*20) / 13."""
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(setup["out"], timestamp, "wind_speed_10m")
        expected = (8 * 10.0 + 2 * 12.0 + 2 * 15.0 + 1 * 20.0) / setup["total_weight"]
        assert math.isclose(val, expected, abs_tol=1e-5)

    def test_cloud_cover_at_t0(self, setup: dict[str, Any]) -> None:
        """(8*80 + 2*60 + 2*100 + 1*90) / 13 = 1050/13 ≈ 80.769.

        Note: cloud_cover is int8 on disk (LLD §4), but the aggregator
        is expected to yield a float result — weighted means of integers
        don't stay integral, and LLD §6 does not cast the output back.
        """
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(setup["out"], timestamp, "cloud_cover")
        expected = (8 * 80 + 2 * 60 + 2 * 100 + 1 * 90) / setup["total_weight"]
        assert math.isclose(val, expected, abs_tol=1e-3)

    def test_shortwave_radiation_at_t0_all_zero(self, setup: dict[str, Any]) -> None:
        """All stations report 0.0 W/m² at midnight → weighted mean is 0.0."""
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(setup["out"], timestamp, "shortwave_radiation")
        assert math.isclose(val, 0.0, abs_tol=1e-9)

    def test_two_hour_frame_produces_two_output_rows(self, setup: dict[str, Any]) -> None:
        """Fixture has two UTC hours; aggregate must preserve both."""
        out = setup["out"]
        if (
            isinstance(out.index, pd.DatetimeIndex)
            or getattr(out.index, "name", None) == "timestamp_utc"
        ):
            hours = set(out.index.tolist())
        else:
            hours = set(out["timestamp_utc"].tolist())
        t0 = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        t1 = pd.Timestamp("2023-01-01T01:00", tz="UTC")
        assert t0 in hours and t1 in hours, f"Aggregator dropped or merged hours; got {hours}"


# --------------------------------------------------------------------------- #
# NaN handling (LLD §6 — drop NaN values, renormalise within the hour)
# --------------------------------------------------------------------------- #


class TestNanHandling:
    """LLD §6: NaNs at individual stations are dropped before weighted mean.

    The canonical pandas weighted-mean idiom (research §9) masks NaN
    inputs before summing, so a missing value at one station weighted
    = `w` effectively redistributes that weight across the remaining
    reporting stations for that (hour, variable) cell.
    """

    def test_nan_at_one_station_drops_that_station_from_that_hour(self, tmp_path: Path) -> None:
        """NaN at London drops it from that hour; subset is Bristol + Manchester + Glasgow.

        Expected: `(2*6 + 2*4 + 1*2) / (2+2+1) = 22/5 = 4.4`.
        """
        stations = [
            _station("london", weight=8.0),
            _station("bristol", weight=2.0),
            _station("manchester", weight=2.0),
            _station("glasgow", weight=1.0),
        ]
        weights = _weights_from_stations(stations)
        df = _load_toy_frame()
        mask = (df["station"] == "london") & (
            df["timestamp_utc"] == pd.Timestamp("2023-01-01T00:00", tz="UTC")
        )
        df.loc[mask, "temperature_2m"] = float("nan")

        out = feat_weather.national_aggregate(df, weights)
        timestamp = pd.Timestamp("2023-01-01T00:00", tz="UTC")
        val = _resolve_scalar(out, timestamp, "temperature_2m")
        expected = (2 * 6.0 + 2 * 4.0 + 1 * 2.0) / (2.0 + 2.0 + 1.0)
        assert math.isclose(val, expected, abs_tol=1e-5), (
            f"NaN at one station must drop it from that hour's weighted mean; "
            f"expected {expected}, got {val}"
        )


# --------------------------------------------------------------------------- #
# helper: cross-layout scalar resolver
# --------------------------------------------------------------------------- #


def _resolve_scalar(agg: pd.DataFrame, timestamp: pd.Timestamp, variable: str) -> float:
    """Pick a scalar out of the aggregate whether it's indexed or flat.

    LLD §6 returns a frame indexed by `timestamp_utc`; implementations
    sometimes reset the index. Support both without the tests caring.
    """
    if (
        isinstance(agg.index, pd.DatetimeIndex)
        or getattr(agg.index, "name", None) == "timestamp_utc"
    ):
        return float(agg.loc[timestamp, variable])
    row = agg[agg["timestamp_utc"] == timestamp]
    assert len(row) == 1, (
        f"Expected exactly one aggregate row for {timestamp} in variable {variable}; "
        f"got {len(row)} row(s)."
    )
    return float(row.iloc[0][variable])
