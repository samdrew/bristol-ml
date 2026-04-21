# `bristol_ml.models` — module guide

This module is the **models layer**: the `Model` protocol that every
estimator implements, the `ModelMetadata` provenance record, joblib-backed
IO helpers, and the concrete model classes (`NaiveModel`, `LinearModel`,
`SarimaxModel`).  Stage 4 introduces the layer; every subsequent modelling
stage (5, 7, 8, 10, 11) adds further model classes that conform to the
same protocol.

Read the layer contract in
[`docs/architecture/layers/models.md`](../../../docs/architecture/layers/models.md)
before extending this module; the file you are reading documents the
concrete Stage 4 surface.

## Current surface (Stage 4)

- `bristol_ml.models.Model` — a `@runtime_checkable` `typing.Protocol` with
  five members (`fit`, `predict`, `save`, `load`, `metadata`). See
  `protocol.py`. Every estimator class must expose these; inheriting from a
  base is neither required nor encouraged.
- `bristol_ml.models.ModelMetadata` — a frozen Pydantic model capturing
  name, `feature_columns`, `fit_utc`, `git_sha`, and a free-form
  `hyperparameters` bag. Defined in `conf/_schemas.py`; re-exported from
  `protocol.py` for ergonomic notebook use.
- `bristol_ml.models.save_joblib(obj, path)` — atomic joblib write (tmp +
  `os.replace`). Creates the parent directory if missing.
- `bristol_ml.models.load_joblib(path)` — joblib deserialiser.
- `bristol_ml.models.naive.NaiveModel` — seasonal-naive baseline.
- `bristol_ml.models.linear.LinearModel` — statsmodels OLS.
- `bristol_ml.models.sarimax.SarimaxModel` — statsmodels SARIMAX with
  daily seasonal order and weekly Fourier exogenous regressors (Stage 7).

## Protocol semantics (load-bearing for every downstream stage)

- **`@runtime_checkable` caveat.** `isinstance(m, Model)` only verifies that
  the object exposes the named attributes; it does *not* verify their
  signatures. Static type checkers catch signature mismatches at
  development time; runtime `isinstance` alone does not. This is the price
  of PEP 544 structural subtyping — keep it visible rather than hidden.
- **Re-entrancy.** `fit()` must be re-entrant: a second call discards the
  previous fit, it does not layer state on top. The save/load round-trip
  tests (AC-3 / plan F-10) rely on this.
- **Predict-before-fit.** `predict()` before `fit()` is undefined
  behaviour. Implementors should raise `RuntimeError` rather than returning
  stale or partial output.
- **`metadata` before fit.** Callers may observe `metadata.fit_utc is None`
  before the first fit. `feature_columns` may be empty in the same state.
- **Cross-version compatibility.** Not a Stage 4 goal. A model saved with
  version *X* is only guaranteed to load with version *X*. Stage 9
  (registry) owns the cross-version story.

## Serialisation

joblib is the Stage 4 default (plan D6). The sklearn-ecosystem choice;
handles numpy/pandas-heavy objects efficiently; round-trips statsmodels
`RegressionResultsWrapper` without extra work. Writes are atomic — tmp
file + `os.replace` — mirroring the ingestion layer's
`_atomic_write` idiom.

**Security note.** joblib (like `pickle`) is not a safe deserialiser for
untrusted inputs. Stage 4 only loads artefacts we wrote ourselves, so the
audit burden of `skops.io` is disproportionate to the stage's demo focus.
The Stage 9 registry is the inflection point for `skops.io` adoption;
`io.py` carries a comment pointing the upgrade path.

## SARIMAX specifics (Stage 7)

A few quirks of the statsmodels SARIMAX surface are load-bearing for
`SarimaxModel` and for any future estimator that wraps the same
state-space machinery — capture them here before you touch `sarimax.py`.

- **`freq="h"` is mandatory.**  The Stage 3 assembler emits an hourly
  UTC-indexed frame but does *not* set `df.index.freq`; statsmodels
  raises a `ValueWarning` on every fit without an explicit
  `freq="h"` on the SARIMAX constructor and occasionally mis-aligns
  forecasts.  `SarimaxModel.fit` constructs SARIMAX with `freq="h"`;
  `test_sarimax_fit_emits_no_frequency_userwarning` is the regression
  guard (Stage 7 plan surprise 2).
- **`predict` must re-index to `features.index`.**  The call
  `SARIMAXResults.get_forecast(steps=n, exog=X_test).predicted_mean`
  does *not* preserve `features.index` — it returns a Series indexed on
  the model's internal time axis.  `SarimaxModel.predict` re-indexes
  the prediction to `features.index` before returning it;
  `test_sarimax_predict_returns_series_indexed_to_features_index` is
  the load-bearing regression guard (Stage 7 plan surprise 1).
- **Rolling-origin re-fits per fold (plan D5).**  The statsmodels idiom
  `results.apply(refit=False)` is attractive for in-fold rolling updates
  but it re-uses the fitted parameters on new data and therefore breaks
  the rolling-origin semantics the harness assumes.  Inside the Stage 6
  harness SARIMAX is re-fit per fold just like every other `Model`.  If
  fit-time pressure ever forces the `apply(refit=False)` shortcut it
  belongs inside a dedicated evaluation-layer fast path, not inside
  `SarimaxModel`.
- **Weekly Fourier, not `s=168`.**  Dual seasonality (24 h daily +
  168 h weekly) is handled via Dynamic Harmonic Regression
  (Hyndman fpp3 §12.1): `seasonal_order[3] = 24`; the weekly period is
  absorbed by `features.fourier.append_weekly_fourier` — set
  `weekly_fourier_harmonics=0` to disable.  A `seasonal_order[3] = 168`
  SARIMAX is rejected at the research layer (Stage 7 research §R2:
  numerically unstable, slow to fit).

## Running standalone

    python -m bristol_ml.models.naive   --help
    python -m bristol_ml.models.linear  --help
    python -m bristol_ml.models.sarimax --help

The `io.py` and `protocol.py` submodules are not standalone — they are
consumed by the concrete models. This is intentional: the layer's public
CLIs are the models, not the plumbing.

## Cross-references

- Layer contract — `docs/architecture/layers/models.md` (Stage 4 lands the
  initial version; Stage 7 updates the SARIMAX column).
- Stage 4 plan — `docs/plans/completed/04-linear-baseline.md` §5 (public
  API surface) and §6 Tasks T2–T4.
- Stage 7 plan — `docs/plans/completed/07-sarimax.md` §5 (SARIMAX config
  schema + call path) and §6 Tasks T3–T5 (scaffold, fit/predict,
  save/load + notebook); retro at `docs/lld/stages/07-sarimax.md`.
- Protocol rationale — `docs/architecture/decisions/0003-protocol-for-model-interface.md`
  (ADR filed in Task T10).
- Intent — `docs/intent/04-linear-baseline.md` AC-2 (interface must be
  implementable in very few lines); `docs/intent/07-sarimax.md` AC-1/AC-5
  (SARIMAX must land behind the same interface).
