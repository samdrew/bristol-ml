# Research: Stage 3 — Feature assembler + train/test split

**Researcher:** domain-researcher agent
**Date:** 2026-04-19
**Intent doc consulted:** `docs/intent/03-feature-assembler.md`
**Design principles:** `docs/intent/DESIGN.md` §2.1 (simplicity bias throughout)

---

## 1. Schema enforcement for the feature DataFrame

### Canonical sources

| Source | Summary |
|--------|---------|
| [pandera PyPI (v0.31.1, Apr 2026)](https://pypi.org/project/pandera/) | Current release; optional pandas extra; core hard deps are `pydantic`, `typeguard`, `typing_extensions`, `typing_inspect`, `packaging` |
| [pandera DataFrame Models docs](https://pandera.readthedocs.io/en/latest/dataframe_models.html) | Class-based `DataFrameModel` API — Pydantic-style subclass syntax with `pa.Field()` for constraints |
| [pandera DataFrameSchema docs](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html) | Dict-based `DataFrameSchema` API — alternative entry point |
| [Pandera 0.20.0: PyArrow data type support (Union.ai, 2026)](https://www.union.ai/blog-post/pandera-0-20-0-pyarrow-data-type-support) | pandera pandas engine accepts pyarrow dtypes; the two are complementary |
| [pandas issue #30189: no schema enforcement in `to_parquet`](https://github.com/pandas-dev/pandas/issues/30189) | pyarrow enforces schema at parquet boundary only; no in-memory idiom |
| [pandera issue #1577: pydantic/typeguard as hard deps](https://github.com/unionai-oss/pandera/issues/1577) | `typeguard` and `typing_inspect` are currently hard requirements |

### Options compared

**Option A — `pandera[pandas]` `DataFrameModel`.** Subclass `pa.DataFrameModel`; annotate columns as `pa.typing.Series[float]`; constrain with `pa.Field(ge=0, nullable=False)`. The syntax is deliberately modelled on Pydantic's `BaseModel`. Catches dtype mismatches, nullability violations, range constraints, categorical membership, regex on strings. Raises `pa.errors.SchemaError` with per-violation report. Dependency cost: `typeguard`, `typing_inspect`, `typing_extensions` (small, no heavy transitive deps). Learning curve: low — same pattern as existing Pydantic config models.

**Option B — pyarrow schema at parquet boundary.** `pa.schema([pa.field("col", pa.float32()), ...])` passed to `write_table`. Catches column presence and dtype on read; does not catch nullability or value ranges. Zero new dependency. Enforcement gap between in-memory assembly and storage is invisible to a newcomer.

**Option C — Pydantic row model.** Iterate rows, validate each. Correct in principle; fights pandas; slow on multi-year hourly DataFrames.

**Option D — Hand-rolled asserts.** `assert df["demand_mw"].notna().all()` etc. Transparent, brittle, no structured error report; accumulates debt as feature set grows through Stages 5, 16, 17.

### Recommendation

**Use `pandera[pandas]` with a `DataFrameModel` subclass.**

Rationale tied to the simplicity bias: the `DataFrameModel` syntax is a net-zero learning cost — it is the same pattern as `BaseModel` already present in `conf/_schemas.py`, applied to DataFrames. Enforcement happens at the in-memory assembly boundary, where dtype drift and unexpected nulls actually appear. The dependency overhead is three small packages. The schema class doubles as inline documentation of the feature table.

Trade-off: pandera is a new runtime dependency. Fallback if the project later adopts a zero-new-dependency rule: Option D, with structured `_validate_features` function.

**Questions for the human:**

1. Willing to add `pandera[pandas]` as a runtime dependency to buy structured schema enforcement, or stay within the zero-new-dependency constraint?
2. Should the `DataFrameModel` class live in `conf/_schemas.py` alongside the Pydantic config schemas, or as a module-local definition in `features/assembler.py`?

---

## 2. Rolling-origin cross-validation for time series

### Canonical sources

| Source | Summary |
|--------|---------|
| [Tashman (2000), Int. J. Forecasting 16(4):437–450](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000650) | Seminal paper establishing "rolling origin" terminology |
| [openforecast.org §2.4 Rolling origin (ADAM book)](https://openforecast.org/adam/rollingOrigin.html) | Modern textbook exposition; expanding vs sliding window; horizon fixed |
| [sklearn TimeSeriesSplit docs (v1.6)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | API reference; `.split()` returns integer index arrays; expanding window only |
| [sklearn issue #22523: rolling window missing](https://github.com/scikit-learn/scikit-learn/issues/22523) | Open since 2021; fixed-size sliding window not supported |
| [sklearn issue #33520: RollingTimeSeriesSplit feature request](https://github.com/scikit-learn/scikit-learn/issues/33520) | Confirms gap; proposes `train_size`, `test_size`, `step_size`, `embargo` |
| [Hyndman et al.: Forecast evaluation pitfalls (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/) | Minimum training window conventions |
| [MachineLearningMastery: Backtesting time series models](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/) | Walk-forward = rolling origin in ML literature |

### Vocabulary (authoritative mapping)

- **Rolling origin** (Tashman 2000) — canonical academic term. Origin advances one step; horizon fixed.
- **Walk-forward validation** — same method in ML practitioner literature.
- **Expanding window** — training set grows at each step (sklearn `TimeSeriesSplit` default).
- **Sliding / rolling window** — fixed-size training set. sklearn does NOT support this natively.
- **TimeSeriesSplit** — sklearn's expanding-window implementation. Returns integer index arrays.

### Fold count and training window

- 365-day test period with 1-day step → 365 folds × 24-sample test set per fold.
- Minimum training window: at least one full seasonal cycle. For GB demand (annual primary cycle) → 12 months floor. DESIGN.md §5.1 starts training 2018-01-01, so first fold has ~5+ years — well above floor.
- Gap/embargo: no universal convention for electricity demand. For day-ahead forecasting the gate-closure discipline is the relevant constraint. Configurable `gap_hours` parameter costs nothing.

### sklearn TimeSeriesSplit assessment

Returns integer index arrays (`train_idx, test_idx = next(tscv.split(X))`) — matches AC-4. Limitations:
- Expanding window only (issue #22523 open since 2021).
- `n_splits` configures fold count indirectly; hard to reason about for "365 daily folds".
- Abstracts the logic away from the pedagogical moment.

### Recommendation

**Hand-roll a simple index generator. Do not use `sklearn.TimeSeriesSplit`.**

The implementation is ~15 lines of pure Python. It makes `min_train_periods`, `test_len`, `step`, `gap` first-class and visible in Hydra config. It teaches the concept directly. Return shape `(np.ndarray, np.ndarray)` matches AC-4 identically.

Sketch (illustrative, not prescriptive):

```python
def rolling_origin_splits(
    n_rows: int,
    *,
    min_train: int,
    test_len: int,
    step: int,
    gap: int = 0,
    fixed_window: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) integer-array pairs in chronological order."""
    for start in range(min_train + gap, n_rows - test_len + 1, step):
        train_start = start - min_train - gap if fixed_window else 0
        train = np.arange(train_start, start - gap)
        test = np.arange(start, start + test_len)
        yield train, test
```

Legible to a meetup attendee in under 30 seconds.

**Questions for the human:**

1. Should the splitter default to an expanding window (training grows from fixed start) or a sliding window (fixed-size training)? Default shapes what gets demoed.
2. Is a `gap_hours` config parameter worth adding in Stage 3 now (one config key, zero logic cost), or should it land with Stage 4 when gate-closure semantics become concrete?

---

## 3. UK clock-change handling for half-hourly GB electricity data

### Canonical sources

| Source | Summary |
|--------|---------|
| [Elexon: Volume Notification on Long Clock Change Day (Oct 2025)](https://www.elexon.co.uk/bsc/event/volume-notification-on-long-clock-change-day/) | Official: autumn day has settlement periods 1–50 |
| [Elexon: Short Clock Change Day (Mar 2025)](https://www.elexon.co.uk/2025/03/28/treatment-of-volume-notifications-on-the-short-clock-change-day/) | Official: spring day has periods 1–46 |
| [stottp.com: Live UK Settlement Period tracker](https://www.stottp.com/resources/uk-settlement-period) | SP1 = 00:00–00:30 local; 48 periods on a normal day |
| [NESO Historic Demand Data 2023 portal](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2023) | Columns: `SETTLEMENT_DATE` (local), `SETTLEMENT_PERIOD` (integer); "follows clock change" |
| [pandas `DatetimeIndex.tz_localize` docs](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.tz_localize.html) | `ambiguous='infer'` (autumn fold-back); `nonexistent='shift_forward'` (spring gap) |
| [pandas timeseries user guide](https://pandas.pydata.org/docs/user_guide/timeseries.html) | Store in UTC internally; convert for display |
| [ADGEfficiency: Elexon API with Python](https://adgefficiency.com/blog/elexon-api-uk-electricity-grid-data-with-python/) | Confirms Elexon/NESO `(SettlementDate, SettlementPeriod)` scheme |

### How NESO encodes clock-change days

- **Normal day:** 1–48 (SP1 = 00:00–00:30 local).
- **Spring day (last Sunday of March, clocks forward):** 1–46. The 01:00–02:00 BST hour does not exist.
- **Autumn day (last Sunday of October, clocks back):** 1–50. The 01:00–02:00 hour occurs twice.

Raw NESO CSV has no UTC timestamps; purely local-time based.

### Cleanest aggregation approach

1. Reconstruct naive local timestamps: `SETTLEMENT_DATE + (SETTLEMENT_PERIOD − 1) × 30min`.
2. Localize: `tz_localize('Europe/London', ambiguous='infer', nonexistent='shift_forward')`.
3. Convert to UTC immediately: `.tz_convert('UTC')`.
4. Resample: `.resample('1h').mean()` (or `.sum()` — policy TBD per OQ-1). UTC resampling is unambiguous.
5. Store with UTC `DatetimeIndex`. Convert to `Europe/London` only in notebooks for plotting.

Avoids the double-counting bug that groupby-on-local-time creates in autumn.

### What a clock-change test should assert

Fixture: one spring day (46 rows) + one autumn day (50 rows) of synthetic half-hourly demand. Assertions:

1. Spring day assembled hourly UTC frame has **23 rows** for the corresponding UTC day.
2. Autumn day assembled hourly UTC frame has **25 rows** for the corresponding UTC day.
3. No `NaT` values in the timestamp index after localization.
4. No duplicate UTC timestamps after conversion.

### Recommendation

**Store all timestamps as UTC throughout. Localize NESO settlement periods to `Europe/London`, convert to UTC immediately, then resample.**

`ambiguous='infer'` + `nonexistent='shift_forward'` handles both anomalous days in one readable call. UTC storage matches Open-Meteo weather (already UTC hourly) and pandas best practice.

Trade-off: notebooks must explicitly `.tz_convert('Europe/London')` before any time-of-day plots, or BST-vs-GMT displays look shifted.

**Questions for the human:**

1. Should the assembler's docstring fix a specific half-hourly-to-hourly aggregation policy (mean / sum / peak)? Choice is consequential for Stage 4.
2. Timestamp column name: `timestamp_utc` (explicit, self-documenting) or `timestamp` with UTC dtype (conventional pandas)? Explicit is more legible at demos.

---

## Overall recommendations summary

| Question | Recommendation | Trade-off accepted |
|----------|---------------|--------------------|
| Schema enforcement | `pandera[pandas]` `DataFrameModel` — same syntax as existing Pydantic models | New runtime dependency (three small packages) |
| Rolling-origin splits | Hand-roll a ~15-line index generator; do not use `sklearn.TimeSeriesSplit` | More code to own; pedagogically direct; more configurable |
| Clock-change handling | Localize NESO periods to `Europe/London`, convert to UTC before any aggregation | Plots need explicit UTC→`Europe/London` conversion for BST display |
