"""Linear regression (statsmodels OLS) baseline model.

Per plan D2 the estimator is :func:`statsmodels.regression.linear_model.OLS`
— sklearn is not a declared dependency and will not be introduced at
Stage 4.  The choice is pedagogical as well as architectural: the
:meth:`RegressionResultsWrapper.summary` method is a ready-made notebook
artefact (coefficients, standard errors, t-stats, R², AIC/BIC, residual
diagnostics) that sklearn's ``LinearRegression`` simply does not expose.

Feature-column resolution:

- If :attr:`LinearConfig.feature_columns` is ``None`` (the default), fit
  uses every float32 weather column declared in the Stage 3 feature-table
  contract (``bristol_ml.features.assembler.WEATHER_VARIABLE_COLUMNS``).
  Resolving at ``fit`` time — not ``__init__`` — keeps the regressor set
  in sync with the assembler contract as new weather variables land.
- If a tuple is provided it narrows the regressor set for ablation
  experiments (plan D11 commentary).

Intercept handling is explicit: statsmodels' ``OLS`` does *not* add a
constant column automatically.  With ``fit_intercept=True`` (the default)
we call :func:`statsmodels.api.add_constant` with ``has_constant="add"``
so re-fitting after a fit that already carried an intercept is
well-defined.

Running standalone::

    python -m bristol_ml.models.linear --help
    python -m bristol_ml.models.linear              # prints resolved config
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from statsmodels.regression.linear_model import RegressionResultsWrapper

from bristol_ml.features.assembler import WEATHER_VARIABLE_COLUMNS
from bristol_ml.models.io import load_skops, save_skops
from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import LinearConfig

__all__ = ["LinearModel"]


# Column name ``statsmodels.api.add_constant`` writes for the intercept.
_INTERCEPT_COLUMN = "const"


# ---------------------------------------------------------------------------
# Stage 12 D10: skops envelope tags for the statsmodels-bytes-v1 pattern.
# Both LinearModel (this file) and SarimaxModel share the *format* tag and
# discriminate by the *kind* tag — see the layer doc and ``sarimax.py`` for
# the matching constants.  The ``RegressionResultsWrapper`` is pickled into
# an opaque ``bytes`` blob inside a skops-safe envelope so the registry
# boundary stops accepting attacker-controlled artefacts (Stage 12 plan
# §Task T4 / D10 Ctrl+G reversal).
# ---------------------------------------------------------------------------

_STATSMODELS_ENVELOPE_FORMAT = "statsmodels-bytes-v1"
_LINEAR_KIND = "OLS"


class LinearModel:
    """Ordinary least-squares regression model.  Implements ``Model``.

    Stores the fitted :class:`~statsmodels.regression.linear_model.RegressionResultsWrapper`
    so downstream callers (and the demo notebook) can inspect
    ``.results.summary()`` directly.  The wrapper pickles cleanly through
    joblib — no custom serialisation needed.
    """

    def __init__(self, config: LinearConfig) -> None:
        """Store ``config`` and initialise empty fit-state."""
        self._config: LinearConfig = config
        # Fully populated on fit().
        self._results: RegressionResultsWrapper | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None

    # ---------------------------------------------------------------------
    # Protocol members
    # ---------------------------------------------------------------------

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit OLS on the resolved feature columns against ``target``.

        ``features`` must contain every column the resolved regressor set
        names; extra columns are tolerated and ignored.  ``target`` must
        align with ``features`` on length (index alignment is delegated to
        the caller — the harness slices both off the same DataFrame).

        Re-calling ``fit`` discards the previous results wrapper entirely
        (plan §10 risk register: re-entrancy).

        Raises
        ------
        ValueError
            If ``target`` and ``features`` have mismatched lengths, or if
            any resolved feature column is missing from ``features``.
        """
        if len(features) != len(target):
            raise ValueError(
                "LinearModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        columns = self._resolve_feature_columns(features)
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"LinearModel.fit: features DataFrame is missing required columns {missing}. "
                "Supply every column named in LinearConfig.feature_columns (or let the config "
                "default resolve the weather set from the assembler schema)."
            )
        X = features[list(columns)].astype("float64")
        if self._config.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        y = pd.Series(np.asarray(target, dtype="float64"), index=features.index)

        # ``RegressionResultsWrapper`` carries everything we need downstream.
        self._results = sm.OLS(y, X).fit()
        # Record the columns in the order the design matrix carried them
        # (without the intercept — the intercept is a fit-time artefact).
        self._feature_columns = tuple(columns)
        self._fit_utc = datetime.now(UTC)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return OLS predictions as a Series indexed to ``features.index``.

        The returned Series carries ``name=config.target_column`` to match
        :class:`bristol_ml.models.naive.NaiveModel`'s convention.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not yet been called.
        ValueError
            If ``features`` is missing a column the model was fit on.
        """
        if self._results is None:
            raise RuntimeError("LinearModel must be fit() before predict().")
        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"LinearModel.predict: features DataFrame is missing columns the model "
                f"was fit on: {missing}."
            )
        X = features[list(self._feature_columns)].astype("float64")
        if self._config.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        preds = self._results.predict(X)
        return pd.Series(
            np.asarray(preds, dtype="float64"),
            index=features.index,
            name=self._config.target_column,
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model atomically as a skops envelope-of-bytes.

        Stage 12 D10 (Ctrl+G reversal): the project moved off
        :mod:`joblib` and onto :mod:`skops.io` so the serving layer is
        not a network-facing pickle deserialiser.  The
        :class:`~statsmodels.regression.linear_model.RegressionResultsWrapper`
        does not have a skops codec, so this method writes an
        envelope-of-bytes: the wrapper is serialised to a ``bytes`` blob
        via :meth:`statsmodels.base.model.Results.save` and the blob
        rides inside a dict envelope of skops-safe primitives
        (``str``, ``bytes``, ``list[str]``, ``dict[str, Any]``).

        Envelope shape::

            {
                "format": "statsmodels-bytes-v1",
                "kind": "OLS",
                "blob": <bytes>,
                "config_dump": <dict>,           # LinearConfig.model_dump()
                "feature_columns": [str, ...],
                "fit_utc_isoformat": <str | None>,
            }

        The opaque blob is *not* loaded through skops's restricted
        unpickler — the security narrative is layered: skops protects
        the envelope (no arbitrary class instantiation through
        :func:`skops.io.load`); the registry's own write contract
        guarantees that the blob bytes were produced by this project's
        :meth:`save` and were not attacker-supplied.  An attacker who
        can replace ``model.skops`` on disk has already breached the
        registry, which is itself a higher trust boundary.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not yet been called.
        """
        import io as _io

        if self._results is None:
            raise RuntimeError("LinearModel must be fit() before save().")

        buf = _io.BytesIO()
        # ``Results.save`` accepts a file-like object via ``fname``;
        # ``remove_data=False`` keeps the design matrix and residuals so
        # the reloaded wrapper can answer the same ``predict``/``summary``
        # surface the original did.
        self._results.save(buf, remove_data=False)
        blob: bytes = buf.getvalue()

        envelope: dict[str, object] = {
            "format": _STATSMODELS_ENVELOPE_FORMAT,
            "kind": _LINEAR_KIND,
            "blob": blob,
            "config_dump": self._config.model_dump(),
            "feature_columns": list(self._feature_columns),
            "fit_utc_isoformat": (self._fit_utc.isoformat() if self._fit_utc is not None else None),
        }
        save_skops(envelope, path)

    @classmethod
    def load(cls, path: Path) -> LinearModel:
        """Load a previously-saved :class:`LinearModel` from ``path``.

        Reads the skops envelope, validates the format/kind tags, then
        deserialises the inner statsmodels wrapper from the bytes blob
        via :meth:`RegressionResults.load`.  The Pydantic re-validation
        of ``config_dump`` is the schema-drift guard (mirrors the NN
        families' Stage 10/11 R2).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        TypeError
            If the loaded envelope's ``format`` tag is not
            ``"statsmodels-bytes-v1"`` or its ``kind`` tag is not
            ``"OLS"`` — the latter prevents a SARIMAX envelope from
            being silently loaded as if it were a LinearModel artefact.
        """
        import io as _io

        from statsmodels.regression.linear_model import RegressionResults

        envelope = load_skops(path)

        if not isinstance(envelope, dict):
            raise TypeError(
                f"Expected a LinearModel skops envelope (dict) at {path}; "
                f"got {type(envelope).__name__}."
            )
        if envelope.get("format") != _STATSMODELS_ENVELOPE_FORMAT:
            raise TypeError(
                f"LinearModel.load: envelope at {path} does not carry the "
                f"expected format tag {_STATSMODELS_ENVELOPE_FORMAT!r}; got "
                f"{envelope.get('format')!r}."
            )
        if envelope.get("kind") != _LINEAR_KIND:
            raise TypeError(
                f"LinearModel.load: envelope at {path} carries kind "
                f"{envelope.get('kind')!r}; expected {_LINEAR_KIND!r}.  "
                "A SARIMAX-kind envelope cannot be loaded as a LinearModel."
            )

        # Pydantic re-validation — schema-drift guard.
        config = LinearConfig.model_validate(envelope["config_dump"])

        results = RegressionResults.load(_io.BytesIO(envelope["blob"]))

        model = cls(config)
        model._results = results
        model._feature_columns = tuple(envelope.get("feature_columns") or ())
        fit_utc_iso = envelope.get("fit_utc_isoformat")
        model._fit_utc = datetime.fromisoformat(fit_utc_iso) if fit_utc_iso is not None else None
        return model

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record, including fitted coefficients when present."""
        hyperparameters: dict[str, object] = {
            "target_column": self._config.target_column,
            "fit_intercept": self._config.fit_intercept,
        }
        if self._results is not None:
            # Coefficients are a pandas Series indexed by (intercept, *columns).
            # Serialise to a plain dict so the metadata object stays JSON-
            # friendly and does not depend on pandas at load time.
            hyperparameters["coefficients"] = {
                str(k): float(v) for k, v in self._results.params.items()
            }
            hyperparameters["rsquared"] = float(self._results.rsquared)
            hyperparameters["nobs"] = int(self._results.nobs)
        return ModelMetadata(
            name="linear-ols-weather-only",
            feature_columns=self._feature_columns,
            fit_utc=self._fit_utc,
            hyperparameters=hyperparameters,
        )

    # ---------------------------------------------------------------------
    # Public helper — lets the notebook print ``.results.summary()``
    # ---------------------------------------------------------------------

    @property
    def results(self) -> RegressionResultsWrapper:
        """The underlying statsmodels results wrapper (post-fit).

        Exposing the wrapper is deliberate: the Stage 4 demo moment is
        ``print(model.results.summary())`` in the notebook.  Before fit
        this raises ``RuntimeError`` rather than returning ``None``.
        """
        if self._results is None:
            raise RuntimeError("LinearModel must be fit() before accessing .results.")
        return self._results

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _resolve_feature_columns(self, features: pd.DataFrame) -> tuple[str, ...]:
        """Return the tuple of regressor columns for this fit.

        When ``config.feature_columns is None`` we resolve the float32
        weather columns from the Stage 3 assembler contract — the
        feature-table is guaranteed to carry them (D2 of the Stage 3
        plan), so no further validation is required here.  An explicit
        tuple in ``config`` narrows the set for ablation.
        """
        if self._config.feature_columns is not None:
            return tuple(self._config.feature_columns)
        return tuple(name for name, _dtype in WEATHER_VARIABLE_COLUMNS)


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.linear``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.linear",
        description=(
            "Print the resolved linear-model config (target, regressors, "
            "fit_intercept) from the default Hydra composition. Training and "
            "prediction are exercised via the evaluation harness (see "
            "`python -m bristol_ml.train`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model.fit_intercept=false",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Returns ``0`` on success; ``2`` if the resolved config does not select
    a linear variant.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    cfg = load_config(overrides=["model=linear", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, LinearConfig):
        print(
            "No LinearConfig resolved. Ensure `model=linear` or a matching override "
            "is present; got "
            f"{type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    lin_cfg = cfg.model
    resolved = (
        lin_cfg.feature_columns
        if lin_cfg.feature_columns is not None
        else tuple(name for name, _dtype in WEATHER_VARIABLE_COLUMNS)
    )
    logger.info(
        "LinearModel config: target_column={} fit_intercept={} feature_columns={}",
        lin_cfg.target_column,
        lin_cfg.fit_intercept,
        resolved,
    )
    print(f"target_column={lin_cfg.target_column}")
    print(f"fit_intercept={lin_cfg.fit_intercept}")
    print(f"feature_columns={list(resolved)}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
