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
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

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

    The scaffold landed in Task T3: constructor, ``metadata`` and
    ``results`` properties, and ``_cli_main`` are complete.  :meth:`fit`,
    :meth:`predict`, :meth:`save`, and :meth:`load` raise
    :class:`NotImplementedError` and are filled in Tasks T4 / T5.
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
        """Fit SARIMAX on ``(features, target)``.  Filled in Task T4."""
        raise NotImplementedError(
            "SarimaxModel.fit lands in Stage 7 Task T4. "
            "See docs/plans/active/07-sarimax.md §Task T4."
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return SARIMAX predictions indexed to ``features.index``.

        Filled in Task T4.
        """
        raise NotImplementedError(
            "SarimaxModel.predict lands in Stage 7 Task T4. "
            "See docs/plans/active/07-sarimax.md §Task T4."
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model atomically via joblib.

        Filled in Task T5.
        """
        raise NotImplementedError(
            "SarimaxModel.save lands in Stage 7 Task T5. "
            "See docs/plans/active/07-sarimax.md §Task T5."
        )

    @classmethod
    def load(cls, path: Path) -> SarimaxModel:
        """Load a previously-saved :class:`SarimaxModel` from ``path``.

        Filled in Task T5.
        """
        raise NotImplementedError(
            "SarimaxModel.load lands in Stage 7 Task T5. "
            "See docs/plans/active/07-sarimax.md §Task T5."
        )

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
