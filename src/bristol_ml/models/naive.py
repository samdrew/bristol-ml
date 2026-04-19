"""Seasonal-naive baseline model.

The credible-but-beatable floor against which the Stage 4 OLS is measured
(plan D1 / intent AC-2: "the model interface is implementable in very few
lines of code — the naive model proves this").  Three strategies are
supported:

- ``same_hour_yesterday`` — predict :math:`y_{t-24h}`.  Easiest to beat; kept
  for ablation.
- ``same_hour_last_week`` — predict :math:`y_{t-168h}`.  The default.
  Captures the dominant weekly seasonality in GB national demand without
  any training-loop complexity.
- ``same_hour_same_weekday`` — for each prediction at time :math:`t`, look up
  the most recent training row with matching ``(weekday, hour)``.  Behaves
  like ``same_hour_last_week`` on contiguous data but gracefully handles
  gaps between training and test (returns the latest available match).

All three are one-line lookups against the training target; there is no
training loop.  Per plan §10 risk register ``fit()`` is re-entrant — a
second call discards the previous state.

The look-back guard raises :class:`ValueError` when the required historical
row is missing from training (per plan T3 acceptance).  This protects the
naive predictor from returning ``NaN`` silently; the harness in Task T6
configures ``min_train_periods >= 168`` so the first fold already has a
week of history.

Running standalone::

    python -m bristol_ml.models.naive --help
    python -m bristol_ml.models.naive              # prints resolved strategy
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from bristol_ml.models.io import load_joblib, save_joblib
from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import NaiveConfig

__all__ = ["NaiveModel"]


class NaiveModel:
    """Seasonal-naive baseline.  Implements :class:`bristol_ml.models.Model`.

    ``NaiveModel`` does not use the ``features`` argument at fit or predict
    time — the seasonal-naive rule is a pure function of the target history.
    ``features`` is accepted to satisfy the :class:`~bristol_ml.models.Model`
    protocol signature; the column list is recorded in
    :attr:`metadata.feature_columns` for provenance only.

    Predictions are indexed to the ``features`` DataFrame passed to
    :meth:`predict` so the harness can align them with the corresponding
    target slice.  The returned Series carries ``name=config.target_column``.
    """

    def __init__(self, config: NaiveConfig) -> None:
        """Store ``config`` and initialise empty fit-state.

        Parameters
        ----------
        config:
            Validated :class:`~conf._schemas.NaiveConfig` instance (the
            Pydantic model is ``frozen=True`` so the reference is shared
            safely).
        """
        self._config: NaiveConfig = config
        self._target: pd.Series | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None

    # ---------------------------------------------------------------------
    # Protocol members
    # ---------------------------------------------------------------------

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Record the full training target indexed by its ``DatetimeIndex``.

        ``features`` is accepted only to satisfy the ``Model`` protocol
        signature; the naive predictor uses ``target`` alone.  The feature
        column list is stored for :attr:`metadata` provenance.

        Re-calling ``fit`` discards the previous fit (plan §10 risk row).

        Raises
        ------
        TypeError
            If ``target.index`` is not a :class:`pandas.DatetimeIndex`.
        ValueError
            If ``target`` and ``features`` have misaligned lengths, or if
            the target index is not strictly ascending (the look-up relies
            on index equality).
        """
        if not isinstance(target.index, pd.DatetimeIndex):
            raise TypeError(
                "NaiveModel.fit requires a DatetimeIndex on target; "
                f"got {type(target.index).__name__}."
            )
        if len(features) != len(target):
            raise ValueError(
                "NaiveModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        if not target.index.is_monotonic_increasing:
            raise ValueError(
                "NaiveModel.fit requires a strictly ascending DatetimeIndex on target; "
                "sort the training frame by timestamp_utc before fitting."
            )
        # Store a detached copy so downstream mutation of the caller's
        # frame does not bleed into the fitted model's state.
        self._target = target.copy()
        self._target.name = self._config.target_column
        self._feature_columns = tuple(features.columns)
        self._fit_utc = datetime.now(UTC)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return seasonal-naive predictions indexed to ``features.index``.

        The strategy (plan D1) is dispatched at call time so a single fit
        can feed multiple evaluation passes.  The returned series has
        ``name == config.target_column`` to match :meth:`fit`'s storage
        convention.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not yet been called.
        TypeError
            If ``features.index`` is not a :class:`pandas.DatetimeIndex`.
        ValueError
            If any required look-back row is missing from training.  The
            error names the missing-row count and instructs the caller to
            supply more training history.
        """
        if self._target is None:
            raise RuntimeError("NaiveModel must be fit() before predict().")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError(
                "NaiveModel.predict requires a DatetimeIndex on features; "
                f"got {type(features.index).__name__}."
            )
        strategy = self._config.strategy
        if strategy == "same_hour_yesterday":
            return self._predict_fixed_lag(features.index, pd.Timedelta(hours=24))
        if strategy == "same_hour_last_week":
            return self._predict_fixed_lag(features.index, pd.Timedelta(hours=168))
        # same_hour_same_weekday
        return self._predict_same_weekday(features.index)

    def save(self, path: Path) -> None:
        """Serialise the fitted model to ``path`` atomically.

        Delegates to :func:`bristol_ml.models.io.save_joblib` — the tmp-file
        + ``os.replace`` write is identical to the ingestion layer's atomic
        pattern.  A crash mid-write therefore leaves the prior artefact
        intact.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not yet been called; we refuse to persist
            empty state.
        """
        if self._target is None:
            raise RuntimeError("NaiveModel must be fit() before save().")
        save_joblib(self, path)

    @classmethod
    def load(cls, path: Path) -> NaiveModel:
        """Load a previously-saved :class:`NaiveModel` from ``path``.

        Raises
        ------
        TypeError
            If the artefact at ``path`` is not a :class:`NaiveModel`
            instance — we never silently hand back the wrong class.
        """
        obj = load_joblib(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected a NaiveModel artefact at {path}; got {type(obj).__name__}.")
        return obj

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit.

        Before :meth:`fit` has been called, ``fit_utc`` is ``None`` and
        ``feature_columns`` is empty — matching the
        :class:`~bristol_ml.models.Model` protocol's "observable empty
        state before fit" semantics documented in the models CLAUDE.md.
        """
        # ModelMetadata.name matches ``^[a-z][a-z0-9_.-]*$``; convert
        # underscores in the strategy name to hyphens for readability.
        return ModelMetadata(
            name=f"naive-{self._config.strategy.replace('_', '-')}",
            feature_columns=self._feature_columns,
            fit_utc=self._fit_utc,
            hyperparameters={
                "strategy": self._config.strategy,
                "target_column": self._config.target_column,
            },
        )

    # ---------------------------------------------------------------------
    # Internal dispatch
    # ---------------------------------------------------------------------

    def _predict_fixed_lag(self, index: pd.DatetimeIndex, lag: pd.Timedelta) -> pd.Series:
        """Look up ``target[t - lag]`` for every ``t`` in ``index``.

        Uses :meth:`pandas.Series.reindex` which returns ``NaN`` for missing
        timestamps; we then count and raise rather than let the ``NaN``
        propagate into downstream metrics (which would surface as a confusing
        ``NaN`` metric rather than a missing-data error).
        """
        assert self._target is not None  # narrowed by public caller
        lookups = index - lag
        values = self._target.reindex(lookups).to_numpy()
        missing = np.isnan(values)
        if missing.any():
            n_missing = int(missing.sum())
            raise ValueError(
                f"Seasonal-naive '{self._config.strategy}' requires training targets "
                f"at t - {lag}; {n_missing}/{len(index)} prediction rows have no "
                "matching training row. Supply more training history "
                "(e.g. raise SplitterConfig.min_train_periods)."
            )
        return pd.Series(values, index=index, name=self._config.target_column)

    def _predict_same_weekday(self, index: pd.DatetimeIndex) -> pd.Series:
        """For each ``t``, find the latest training row with matching (weekday, hour).

        Not optimised — naive is never the bottleneck (~O(n_test * n_train)
        in the worst case, but with a cheap vectorised mask per ``t``).
        """
        assert self._target is not None  # narrowed by public caller
        tr = self._target
        tr_idx = tr.index
        values = np.empty(len(index), dtype=np.float64)
        for i, t in enumerate(index):
            mask = (tr_idx.weekday == t.weekday()) & (tr_idx.hour == t.hour) & (tr_idx <= t)
            if not mask.any():
                raise ValueError(
                    "Seasonal-naive 'same_hour_same_weekday' found no training "
                    f"row matching weekday={t.weekday()} hour={t.hour} on or "
                    f"before {t.isoformat()}. Supply more training history "
                    "(e.g. raise SplitterConfig.min_train_periods)."
                )
            # max() on the filtered index returns the latest matching timestamp.
            candidate_idx = tr_idx[mask].max()
            values[i] = tr.loc[candidate_idx]
        return pd.Series(values, index=index, name=self._config.target_column)


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.naive``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.naive",
        description=(
            "Print the resolved naive-model config (strategy, target_column) "
            "from the default Hydra composition. Training and prediction are "
            "exercised via the evaluation harness (see "
            "`python -m bristol_ml.train`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model=naive model.naive.strategy=same_hour_yesterday",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 compliance.

    Returns ``0`` on success; ``2`` if the resolved config does not select
    a naive variant (the user forgot ``model=naive`` on the CLI).
    """
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local import so ``--help`` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=["model=naive", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, NaiveConfig):
        print(
            "No NaiveConfig resolved. Ensure `model=naive` or a matching "
            "override is present; got "
            f"{type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    naive_cfg = cfg.model
    logger.info(
        "NaiveModel config: strategy={} target_column={}",
        naive_cfg.strategy,
        naive_cfg.target_column,
    )
    print(f"strategy={naive_cfg.strategy}")
    print(f"target_column={naive_cfg.target_column}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
