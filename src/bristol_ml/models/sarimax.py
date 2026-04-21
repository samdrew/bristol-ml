"""SARIMAX (seasonal ARIMA with exogenous regressors) model — Stage 7.

Follows the Stage 4 :class:`bristol_ml.models.Model` protocol.  The heavy
lifting is delegated to :class:`statsmodels.tsa.statespace.sarimax.SARIMAX`
— this module wires the protocol surface, manages config-driven construction
of the exogenous matrix, and owns the two codebase surprises flagged in
the Stage 7 plan:

**Surprise 1 — predict re-indexing (§`predict`).**  The statsmodels call
``SARIMAXResults.get_forecast(steps=n, exog=X_test).predicted_mean`` does
**not** preserve ``features.index``; it returns a Series indexed on the
model's internal time axis.  :meth:`predict` therefore re-indexes the
prediction to ``features.index`` before returning it.  The
load-bearing regression guard is
``test_sarimax_predict_returns_series_indexed_to_features_index`` in
``tests/unit/models/test_sarimax.py``.

**Surprise 2 — ``freq="h"`` trap (§`fit`).**  The Stage 3 assembler emits
an hourly UTC-indexed frame but does *not* set ``df.index.freq``; without
an explicit ``freq="h"`` on the SARIMAX constructor, statsmodels raises a
``ValueWarning`` on every fit and occasionally mis-aligns forecasts.
:meth:`fit` constructs SARIMAX with ``freq="h"`` verbatim; the regression
guard is ``test_sarimax_fit_emits_no_frequency_userwarning``.

**Rolling-origin semantics (plan D5).**  The statsmodels idiom
``results.apply(refit=False)`` is attractive for in-fold rolling updates
but it re-uses the fitted parameters on new data and therefore breaks
the rolling-origin re-fit semantics the harness assumes.  Inside the
Stage 6 harness SARIMAX is therefore **re-fit per fold** just like every
other :class:`bristol_ml.models.Model`.  If fit-time pressure ever forces
the ``apply(refit=False)`` shortcut it belongs inside a dedicated
evaluation-layer fast path, not inside ``SarimaxModel``.

**Weekly Fourier (plan D1, D3).**  The dual seasonality of GB hourly
demand (24 h daily + 168 h weekly) is handled via Dynamic Harmonic
Regression (Hyndman fpp3 §12.1): SARIMAX carries the daily seasonal
order ``s=24``; the weekly period is absorbed by
:func:`bristol_ml.features.fourier.append_weekly_fourier` which
:meth:`fit` appends to the exogenous frame when
``SarimaxConfig.weekly_fourier_harmonics > 0``.  Setting
``weekly_fourier_harmonics=0`` disables the Fourier path; ``s=168``
seasonal SARIMAX is explicitly rejected at the config level (research
§R2: numerically unstable + slow to fit).

Running standalone::

    python -m bristol_ml.models.sarimax --help
    python -m bristol_ml.models.sarimax             # prints config schema
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from bristol_ml.features.fourier import append_weekly_fourier
from bristol_ml.models.io import load_joblib, save_joblib
from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import SarimaxConfig

__all__ = ["SarimaxModel"]


class SarimaxModel:
    """SARIMAX model (daily seasonal order + weekly Fourier exog).

    Implements :class:`bristol_ml.models.Model`.  Stores the fitted
    :class:`~statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper`
    on ``self._results`` so the Stage 7 notebook can print
    ``print(model.results.summary())`` and call
    ``model.results.plot_diagnostics()`` directly (plan §Task T5 cell 7).

    Implementation landed across Tasks T3 (scaffold + metadata +
    ``_cli_main``), T4 (``fit`` / ``predict``) and T5 (``save`` /
    ``load`` + notebook).  The Stage 7 plan's three codebase surprises
    are covered by regression tests in
    ``tests/unit/models/test_sarimax.py``.
    """

    def __init__(self, config: SarimaxConfig) -> None:
        """Store ``config`` and initialise empty fit-state.

        Parameters
        ----------
        config:
            Validated :class:`~conf._schemas.SarimaxConfig` (the Pydantic
            model is ``frozen=True`` so the reference is shared safely).
        """
        self._config: SarimaxConfig = config
        # Populated on fit().
        self._results: SARIMAXResultsWrapper | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None
        # Captured so that metadata can carry the target name.  Empty
        # string is the "not yet fit" sentinel.
        self._endog_name: str = ""

    # ---------------------------------------------------------------------
    # Protocol members
    # ---------------------------------------------------------------------

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit SARIMAX on the aligned ``(features, target)`` pair.

        The fit path composes, in order:

        1. Length-parity and index-timezone guards (surprise 2 requires
           a tz-aware UTC index; raising here surfaces the wrong caller
           configuration before statsmodels' opaque ``ValueWarning``).
        2. Resolution of the regressor set: if
           ``config.feature_columns`` is ``None`` every column in
           ``features`` is taken; otherwise the configured tuple is used.
        3. Weekly-Fourier append (plan D1/D3): when
           ``config.weekly_fourier_harmonics > 0`` the
           :func:`append_weekly_fourier` helper adds ``2*harmonics``
           ``float64`` columns and the resolved feature-column tuple is
           extended accordingly.
        4. SARIMAX construction with **``freq="h"``** (surprise 2 fix)
           and the spread ``config.sarimax_kwargs.model_dump()``.
        5. ``.fit(disp=False)`` with a warnings-catch that re-emits any
           :class:`~statsmodels.tools.sm_exceptions.ConvergenceWarning`
           at ``loguru`` WARN level — convergence warnings are
           informational per domain §R1, not fatal.

        Re-calling ``fit`` discards the previous results wrapper entirely
        (NFR-5).

        Raises
        ------
        ValueError
            If ``len(features) != len(target)``; or if ``features.index``
            is not a tz-aware :class:`~pandas.DatetimeIndex`; or if the
            index timezone is not UTC (the Stage 3 assembler contract
            guarantees UTC — other timezones are out of scope); or if a
            resolved feature column is missing from ``features``.
        """
        # --- 1. Guards ---------------------------------------------------
        if len(features) != len(target):
            raise ValueError(
                "SarimaxModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError(
                "SarimaxModel.fit requires a DatetimeIndex on features; "
                f"got {type(features.index).__name__}."
            )
        if features.index.tz is None:
            raise ValueError(
                "SarimaxModel.fit requires a tz-aware DatetimeIndex on features "
                "(the Stage 3 assembler contract guarantees UTC). Got tz-naive."
            )
        if str(features.index.tz) != "UTC":
            raise ValueError(
                "SarimaxModel.fit requires a UTC-tz DatetimeIndex on features "
                "(the Stage 3 assembler contract); got "
                f"tz={features.index.tz!r}. "
                "Convert upstream via df.index = df.index.tz_convert('UTC')."
            )

        # --- 2. Feature-column resolution --------------------------------
        features_with_fourier = self._append_fourier_if_configured(features)
        resolved_columns = self._resolve_feature_columns(features_with_fourier)
        missing = [c for c in resolved_columns if c not in features_with_fourier.columns]
        if missing:
            raise ValueError(
                "SarimaxModel.fit: features DataFrame is missing configured "
                f"columns {missing}. Supply every column named in "
                "SarimaxConfig.feature_columns (or let the default resolve "
                "them from the input frame)."
            )

        # --- 3. Build the numeric matrices --------------------------------
        # Surprise 2: the Stage 3 assembler emits a UTC hourly index but
        # does not set ``df.index.freq``; even with ``freq="h"`` on the
        # SARIMAX constructor, statsmodels' ``_init_dates`` emits a
        # ``ValueWarning`` when the endog index has no freq set (the
        # warning is an informational heads-up that the kwarg freq is being
        # used).  Set freq on a copy of the index so the warning is
        # suppressed and callers' DataFrames are not mutated.
        endog_index = features.index.copy()
        try:
            endog_index.freq = "h"
        except (ValueError, TypeError):
            # Non-uniform or otherwise incompatible index; fall back to
            # the constructor's ``freq="h"`` kwarg path.  The warning may
            # still fire but the fit still works.
            pass
        endog = pd.Series(
            np.asarray(target, dtype=np.float64),
            index=endog_index,
            name=target.name if target.name is not None else "target",
        )
        exog = features_with_fourier[list(resolved_columns)].to_numpy(dtype=np.float64)

        # --- 4. Construct SARIMAX with freq="h" (surprise 2) --------------
        sm_model = SARIMAX(
            endog=endog,
            exog=exog if exog.size else None,
            order=tuple(self._config.order),
            seasonal_order=tuple(self._config.seasonal_order),
            trend=self._config.trend,
            freq="h",
            **self._config.sarimax_kwargs.model_dump(),
        )

        # --- 5. Fit; capture convergence warnings -------------------------
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = sm_model.fit(disp=False)

        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                logger.warning(
                    "SarimaxModel.fit: convergence warning from statsmodels "
                    "(informational per domain §R1): {}",
                    str(w.message),
                )

        # --- 6. Publish state --------------------------------------------
        self._results = results
        self._feature_columns = tuple(resolved_columns)
        self._fit_utc = datetime.now(UTC)
        self._endog_name = str(target.name) if target.name is not None else "target"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return SARIMAX predictions indexed to ``features.index``.

        Surprise 1 fix: ``SARIMAXResults.get_forecast(...).predicted_mean``
        does *not* carry ``features.index``; it returns a Series on the
        model's internal time axis.  We construct the returned Series
        with the input's index directly so the harness's downstream
        ``predictions_df = pd.concat(...)`` alignment works.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        KeyError
            If ``features`` (after the optional Fourier append) is
            missing a column the model was fit on.
        """
        if self._results is None:
            raise RuntimeError("SarimaxModel must be fit() before predict().")

        features_with_fourier = self._append_fourier_if_configured(features)
        missing = [c for c in self._feature_columns if c not in features_with_fourier.columns]
        if missing:
            raise KeyError(
                f"SarimaxModel.predict: features DataFrame is missing columns "
                f"the model was fit on: {missing}."
            )
        exog_test = features_with_fourier[list(self._feature_columns)].to_numpy(dtype=np.float64)

        forecast_result = self._results.get_forecast(
            steps=len(features),
            exog=exog_test if exog_test.size else None,
        )
        predicted_values = np.asarray(forecast_result.predicted_mean, dtype=np.float64)

        # Surprise 1: re-index to features.index rather than letting the
        # statsmodels internal time axis leak through.
        return pd.Series(
            predicted_values,
            index=features.index,
            name=self._config.target_column,
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted :class:`SarimaxModel` atomically (plan D6).

        Delegates to :func:`bristol_ml.models.io.save_joblib`.  The
        :class:`~statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper`
        is pickle-compatible via
        :meth:`~statsmodels.tsa.statespace.mlemodel.MLEResults.__getstate__`,
        so joblib handles the whole ``SarimaxModel`` instance — including
        its config, feature-column tuple, fit_utc, and the results wrapper —
        in one round trip.

        The alternative considered and rejected was
        ``self._results.save(path, remove_data=True)``: it only serialises
        the statsmodels results and loses the wrapping ``SarimaxModel``
        metadata needed for reconstruction (config, feature-columns,
        fit_utc).  Sticking with :func:`save_joblib` keeps the Stage 4
        save/load contract in sync across every concrete model.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._results is None:
            raise RuntimeError("Cannot save unfitted SarimaxModel")
        save_joblib(self, path)

    @classmethod
    def load(cls, path: Path) -> SarimaxModel:
        """Load a previously-saved :class:`SarimaxModel` from ``path``.

        Raises
        ------
        TypeError
            If the artefact at ``path`` is not a :class:`SarimaxModel`
            (e.g. a :class:`~bristol_ml.models.linear.LinearModel` pickled
            at the same path).  Mirrors the Stage 4 ``LinearModel.load``
            guard.
        """
        obj = load_joblib(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a SarimaxModel artefact at {path}; got {type(obj).__name__}."
            )
        return obj

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit.

        Before :meth:`fit` has been called, ``fit_utc`` is ``None`` and
        ``feature_columns`` is empty — matching the Stage 4 protocol
        convention.  Once fitted, ``hyperparameters`` additionally carries
        ``aic``, ``bic``, ``nobs``, and ``converged`` so the notebook's
        AIC-sweep cell can compare candidates apples-to-apples.
        """
        hyperparameters: dict[str, object] = {
            "target_column": self._config.target_column,
            "order": list(self._config.order),
            "seasonal_order": list(self._config.seasonal_order),
            "trend": self._config.trend,
            "weekly_fourier_harmonics": self._config.weekly_fourier_harmonics,
        }
        if self._results is not None:
            hyperparameters["aic"] = float(self._results.aic)
            hyperparameters["bic"] = float(self._results.bic)
            hyperparameters["nobs"] = int(self._results.nobs)
            # ``mle_retvals`` is the statsmodels optimiser bag.  The
            # ``converged`` key is present when the optimiser ran (which
            # is every real fit); guard with ``.get`` so a hypothetical
            # future statsmodels API change surfaces as ``None`` rather
            # than a KeyError.
            mle_retvals = getattr(self._results, "mle_retvals", None) or {}
            hyperparameters["converged"] = bool(mle_retvals.get("converged", False))
        return ModelMetadata(
            name=_build_metadata_name(self._config.order, self._config.seasonal_order),
            feature_columns=self._feature_columns,
            fit_utc=self._fit_utc,
            hyperparameters=hyperparameters,
        )

    # ---------------------------------------------------------------------
    # Public helper — lets the notebook print ``.results.summary()``
    # ---------------------------------------------------------------------

    @property
    def results(self) -> SARIMAXResultsWrapper:
        """The underlying statsmodels SARIMAX results wrapper (post-fit).

        Exposing the wrapper is deliberate: the Stage 7 demo moment is
        ``print(model.results.summary())`` and
        ``model.results.plot_diagnostics()`` in the notebook.  Before
        :meth:`fit` this raises :class:`RuntimeError` rather than returning
        ``None``.
        """
        if self._results is None:
            raise RuntimeError("SarimaxModel must be fit before accessing .results")
        return self._results

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _append_fourier_if_configured(self, features: pd.DataFrame) -> pd.DataFrame:
        """Append weekly-Fourier columns when ``weekly_fourier_harmonics > 0``.

        The helper is deterministic given a UTC index (plan D1/D3 —
        Dynamic Harmonic Regression).  When harmonics is zero this is a
        no-op and we return a shallow copy so callers can attach columns
        without ever mutating the input frame.
        """
        if self._config.weekly_fourier_harmonics <= 0:
            return features.copy()
        return append_weekly_fourier(
            features,
            period_hours=168,
            harmonics=self._config.weekly_fourier_harmonics,
            column_prefix="week",
        )

    def _resolve_feature_columns(self, features_with_fourier: pd.DataFrame) -> tuple[str, ...]:
        """Return the ordered tuple of regressor columns for this fit.

        - If ``config.feature_columns is None`` (the default) every
          column in ``features_with_fourier`` is taken — this is the
          "weather + calendar + Fourier" path that the Stage 7 notebook
          exercises.
        - If ``config.feature_columns`` is a tuple we append the Fourier
          column names (when harmonics > 0) to it so the caller does not
          have to know about them.
        - An empty result means SARIMAX runs with ``exog=None`` — a
          pure (S)ARIMA fit.  The fit path translates this to
          ``exog=None`` on construction, so statsmodels handles it
          cleanly.
        """
        if self._config.feature_columns is None:
            return tuple(features_with_fourier.columns)
        configured = tuple(self._config.feature_columns)
        if self._config.weekly_fourier_harmonics > 0:
            fourier_names = tuple(
                name
                for k in range(1, self._config.weekly_fourier_harmonics + 1)
                for name in (f"week_sin_k{k}", f"week_cos_k{k}")
            )
            # Preserve configured order; append Fourier names only if
            # not already in the configured tuple (edge case: caller
            # spelled them out manually).
            extra = tuple(n for n in fourier_names if n not in configured)
            return configured + extra
        return configured


# ---------------------------------------------------------------------------
# Private helpers (module-level so they stay easy to unit-test)
# ---------------------------------------------------------------------------


def _build_metadata_name(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> str:
    """Build a metadata ``name`` that matches ``ModelMetadata.name``'s regex.

    :class:`~conf._schemas.ModelMetadata` constrains ``name`` to
    ``^[a-z][a-z0-9_.-]*$`` — tuples formatted via ``repr()`` introduce
    parentheses and commas that the regex rejects.  The format spelled
    out here is ``sarimax-{p}-{d}-{q}-{P}-{D}-{Q}-{s}`` and is stable
    enough to be compared verbatim across notebooks and registry records.
    """
    p, d, q = order
    sp, sd, sq, s = seasonal_order
    return f"sarimax-{p}-{d}-{q}-{sp}-{sd}-{sq}-{s}"


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.sarimax``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.sarimax",
        description=(
            "Print the SarimaxConfig JSON schema and a one-line help banner. "
            "Training and prediction are exercised via the evaluation harness "
            "(see `python -m bristol_ml.train`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model.weekly_fourier_harmonics=0",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Prints the :class:`~conf._schemas.SarimaxConfig` JSON schema to stdout
    and a one-line help banner pointing at the training CLI.  Returns
    ``0`` on success, ``2`` if a non-SARIMAX model resolves from config
    (e.g. if the caller forgot ``model=sarimax``).
    """
    import json

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    # Print the schema unconditionally — this is the useful standalone
    # output even without a warm Hydra config.
    print("SarimaxConfig JSON schema:")
    print(json.dumps(SarimaxConfig.model_json_schema(), indent=2))
    print()
    print(
        "To fit: python -m bristol_ml.train model=sarimax "
        "evaluation.rolling_origin.fixed_window=true "
        "evaluation.rolling_origin.step=168"
    )

    cfg = load_config(overrides=["model=sarimax", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, SarimaxConfig):
        print(
            "No SarimaxConfig resolved. Ensure `model=sarimax` or a matching "
            "override is present; got "
            f"{type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    sarimax_cfg = cfg.model
    logger.info(
        "SarimaxModel config: order={} seasonal_order={} trend={} "
        "weekly_fourier_harmonics={} target_column={}",
        sarimax_cfg.order,
        sarimax_cfg.seasonal_order,
        sarimax_cfg.trend,
        sarimax_cfg.weekly_fourier_harmonics,
        sarimax_cfg.target_column,
    )
    print()
    print(f"order={list(sarimax_cfg.order)}")
    print(f"seasonal_order={list(sarimax_cfg.seasonal_order)}")
    print(f"trend={sarimax_cfg.trend}")
    print(f"weekly_fourier_harmonics={sarimax_cfg.weekly_fourier_harmonics}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
