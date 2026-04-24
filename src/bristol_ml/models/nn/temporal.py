"""Temporal convolutional network model — Stage 11.

Implements the Stage 4 :class:`bristol_ml.models.Model` protocol via a
PyTorch ``nn.Module``-backed Temporal Convolutional Network (TCN — the
Bai-et-al.-2018 dilated-causal-conv recipe).  Stage 11 is the second
torch-backed model family; Stage 10 (``mlp.py``) fired the D10 extraction
seam with it, so the hand-rolled training loop and four-stream seed
recipe now live in :mod:`bristol_ml.models.nn._training` and are shared
between both families.

T2 landed the sequence-data pipeline (:class:`_SequenceDataset`) plus the
:class:`~conf._schemas.NnTemporalConfig` Pydantic schema and matching Hydra
YAML group.  T3 (this commit) adds the class surface (``__init__``,
``metadata``, and stubbed ``fit`` / ``predict`` / ``save`` / ``load``
raising :class:`NotImplementedError`), plus the standalone CLI entry point
(NFR-5).  T4 fills ``fit`` / ``predict`` with the TCN body + causal
padding; T5 fills ``save`` / ``load`` with the single-joblib envelope.

Design context (plan §1):

- **Lazy windowing (plan D7).**  :class:`_SequenceDataset` stores two
  flat numpy arrays — features of shape ``(N, n_features)`` and target
  of shape ``(N,)`` — and computes ``(features[i:i+seq_len],
  target[i+seq_len])`` per ``__getitem__`` call.  An eagerly
  materialised ``(N-seq_len, seq_len, n_features)`` tensor at the
  intent's default feature set (44 calendar + ~6 weather cols, 43 633
  hourly rows, ``seq_len=168``) would cost ~1.4 GB — comfortably
  outside the Stage 11 "runs on a laptop" envelope (domain research §8,
  codebase map §4).  The lazy pattern holds the footprint at
  ``N * n_features * 4`` bytes (~10 MB at the same fixture).
- **Pattern A exogenous handling (plan D3).**  The dataset yields a
  window of shape ``(seq_len, n_features)`` where ``n_features`` is the
  full harness-supplied column count; there is no separate "known
  future" branch.  The TCN body (T4) consumes that window directly
  after transposing to ``(n_features, seq_len)`` for Conv1d.
- **Raw-scale values; scaling happens inside the module (plan D5 /
  Stage 10 D4 inheritance).**  :class:`_SequenceDataset` yields
  unscaled ``float32`` tensors.  The z-score scaler buffers
  (``feature_mean`` / ``feature_std`` / ``target_mean`` / ``target_std``)
  ride inside the :class:`torch.nn.Module` via ``register_buffer`` —
  same recipe as Stage 10 ``_NnMlpModule`` — so they round-trip through
  ``state_dict`` automatically and the caller does not need to hold
  normalisation state outside the module.  The caller pre-normalises
  the target before constructing this dataset (so MSE loss operates on
  the O(1) normalised scale); features are normalised *inside*
  ``forward()``.

Running standalone (NFR-5)::

    python -m bristol_ml.models.nn.temporal --help
    python -m bristol_ml.models.nn.temporal          # prints config schema

The CLI prints the :class:`~conf._schemas.NnTemporalConfig` JSON schema
plus the resolved defaults; it does **not** fit a model.  To train,
use the train CLI with the ``nn_temporal`` model group::

    python -m bristol_ml.train model=nn_temporal
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import NnTemporalConfig

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from torch import nn
    from torch.utils.data import Dataset

__all__ = ["NnTemporalModel"]


# ``_SequenceDataset`` is a ``torch.utils.data.Dataset`` subclass defined
# inside :func:`_build_sequence_dataset_class` so the top-of-module import
# graph does not force an eager ``import torch`` — the CLI ``--help`` path
# (T3) and the Stage 10 lazy-import regression guard
# (``test_nn_mlp_lazy_torch_import_contract``) both require that
# ``import bristol_ml.models.nn.temporal`` costs zero torch cycles.  The
# lazy-build recipe mirrors the Stage 10 ``_NnMlpModule`` pattern, including
# the ``sys.modules`` install step so ``pickle``'s
# ``getattr(sys.modules[__module__], __qualname__)`` lookup resolves when a
# future refactor increases ``DataLoader(num_workers > 0)``.  At the Stage 11
# shipped default of ``num_workers=0`` the install step is latent guard-rail
# rather than an active requirement — same disposition as Stage 10
# ``_NnMlpModuleImpl``.
_sequence_dataset_cls: type[Dataset[tuple[Any, Any]]] | None = None


def _build_sequence_dataset_class() -> type[Dataset[tuple[Any, Any]]]:
    """Return the :class:`torch.utils.data.Dataset` subclass used by the TCN.

    Cached in :data:`_sequence_dataset_cls` after first construction so
    repeated ``fit`` calls don't re-build the class object.  The class is
    defined inside a function because ``torch`` is imported lazily (to
    keep the CLI / scaffold path cheap).  Pickleability across a future
    ``num_workers > 0`` refactor is achieved by (a) setting ``__module__``
    / ``__qualname__`` to the import path and (b) installing the class
    as a module attribute on :mod:`bristol_ml.models.nn.temporal` before
    returning it — mirrors the Stage 10 ``_NnMlpModuleImpl`` recipe; see
    ``src/bristol_ml/models/nn/CLAUDE.md`` "PyTorch specifics" gotcha 1.
    """
    global _sequence_dataset_cls
    if _sequence_dataset_cls is not None:
        return _sequence_dataset_cls

    import torch
    from torch.utils.data import Dataset

    class _SequenceDatasetImpl(Dataset):  # type: ignore[type-arg]
        """Lazy-window sequence dataset for :class:`NnTemporalModel` (plan D7).

        Stores the input feature frame and target series as flat
        ``float32`` numpy arrays; materialises one ``(seq_len,
        n_features)`` window + scalar target per ``__getitem__`` call.
        No eager ``(N-seq_len, seq_len, n_features)`` tensor is
        allocated — that would cost ~1.4 GB at the default Stage 5
        feature-set shape and blow the laptop RAM budget (domain
        research §8, codebase map §4).

        The dataset does **not** normalise — both raw features and the
        raw target flow through.  The caller (:meth:`NnTemporalModel.fit`)
        is responsible for pre-normalising the target series before
        constructing this dataset; feature normalisation happens inside
        ``forward()`` via the module's scaler buffers (plan D5 /
        Stage 10 D4 inheritance).

        Parameters
        ----------
        features:
            ``pd.DataFrame`` with ``len(features) > seq_len``; the raw
            feature table slice the harness passes to ``fit()``.
            Column order is preserved in the yielded tensor.
        target:
            ``pd.Series`` aligned to ``features`` (same length).
        seq_len:
            Window length (``>= 2``).  The dataset's length is
            ``len(features) - seq_len``; each window covers indices
            ``[i, i+seq_len)`` and the aligned target is
            ``target[i+seq_len]``.

        Raises
        ------
        ValueError
            If ``len(features) != len(target)``, if ``seq_len < 2``,
            or if ``len(features) <= seq_len`` (not enough rows for at
            least one window).
        """

        def __init__(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            seq_len: int,
        ) -> None:
            if len(features) != len(target):
                raise ValueError(
                    "_SequenceDataset requires len(features) == len(target); "
                    f"got {len(features)} vs {len(target)}."
                )
            if seq_len < 2:
                raise ValueError(f"_SequenceDataset requires seq_len >= 2; got {seq_len}.")
            if len(features) <= seq_len:
                raise ValueError(
                    "_SequenceDataset requires len(features) > seq_len so at "
                    f"least one window + aligned target exists; got "
                    f"len(features)={len(features)} vs seq_len={seq_len}."
                )

            # Flat storage.  Cast to float32 once — DataLoader workers
            # (num_workers=0 today; guard-rail for future parallelism)
            # share this buffer via torch.from_numpy's view semantics.
            self._features = np.ascontiguousarray(
                features.to_numpy(dtype=np.float32, copy=False),
                dtype=np.float32,
            )
            self._target = np.ascontiguousarray(
                target.to_numpy(dtype=np.float32, copy=False),
                dtype=np.float32,
            )
            self._seq_len = int(seq_len)

        def __len__(self) -> int:
            """Number of valid ``(window, target)`` pairs.

            Equal to ``len(features) - seq_len`` — the final valid
            window starts at index ``len(features) - seq_len - 1`` and
            ends at ``len(features) - 1``, with aligned target at
            index ``len(features) - 1``.  No half-windows are padded.
            """
            return len(self._features) - self._seq_len

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            """Return ``(features[i:i+seq_len], target[i+seq_len])`` as tensors.

            The feature window is returned in shape ``(seq_len,
            n_features)``; the TCN body in T4 transposes to
            ``(n_features, seq_len)`` at the Conv1d boundary.  The
            aligned target is a **scalar** 0-d tensor — consistent with
            the ``nn.MSELoss`` contract the shared training loop
            consumes.

            The returned tensors own their storage (``.copy()`` on the
            numpy slice), so any subsequent mutation of ``self._features``
            / ``self._target`` does not silently corrupt an already-
            fetched batch.  At the Stage 11 shipped default of
            ``num_workers=0`` the ownership cost is ~1-2 KB per window —
            a single hot-loop copy that keeps the worker-pickling
            contract honest if a future fit raises ``num_workers``.
            """
            start = int(index)
            end = start + self._seq_len
            window = self._features[start:end].copy()
            target_value = float(self._target[end])
            return (
                torch.from_numpy(window),
                torch.tensor(target_value, dtype=torch.float32),
            )

    _SequenceDatasetImpl.__module__ = "bristol_ml.models.nn.temporal"
    _SequenceDatasetImpl.__qualname__ = "_SequenceDatasetImpl"
    # Install into the module's namespace so pickle's
    # ``getattr(sys.modules[__module__], __qualname__)`` lookup resolves
    # — the Stage 10 lesson captured in ``CLAUDE.md`` PyTorch-gotcha 1.
    sys.modules[__name__]._SequenceDatasetImpl = _SequenceDatasetImpl  # type: ignore[attr-defined]
    _sequence_dataset_cls = _SequenceDatasetImpl
    return _SequenceDatasetImpl


def _SequenceDataset(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int,
) -> Dataset[tuple[Any, Any]]:
    """Construct a lazy-window sequence dataset via the lazy factory.

    Behaves identically to instantiating the class directly; routes
    through :func:`_build_sequence_dataset_class` so the
    :class:`torch.utils.data.Dataset` subclass is only defined after
    ``torch`` has been imported.  Keeps the top-of-module import cost
    at zero torch cycles for scaffold / CLI callers.
    """
    cls = _build_sequence_dataset_class()
    return cls(features, target, seq_len)


# ---------------------------------------------------------------------------
# ``NnTemporalModel`` — the public class (T3 scaffold; T4 fills fit/predict;
# T5 fills save/load).
# ---------------------------------------------------------------------------


class NnTemporalModel:
    """TCN conforming to the Stage 4 :class:`Model` protocol.

    Stage 11 T3 (this commit) scaffolds the class surface: ``__init__``
    stores the config and initialises empty fit-state, ``metadata``
    returns a provenance record built from ``self._config`` plus whatever
    fit-state is populated, and ``fit`` / ``predict`` / ``save`` / ``load``
    are :class:`NotImplementedError` stubs pending T4 / T5.

    See ``docs/architecture/layers/models-nn.md`` §"Stage 11 addition"
    for the full contract; the five Stage 10 PyTorch gotchas in
    ``src/bristol_ml/models/nn/CLAUDE.md`` all apply unchanged to this
    class (``sys.modules`` install for pickleable modules, lazy
    ``import torch``, scaler-buffer registration at module construction,
    ``torch.load(..., weights_only=True, map_location="cpu")``,
    single-joblib envelope).
    """

    def __init__(self, config: NnTemporalConfig) -> None:
        """Store ``config`` and initialise empty fit-state.

        Parameters
        ----------
        config:
            Validated :class:`~conf._schemas.NnTemporalConfig`.  Pydantic's
            ``frozen=True`` makes the reference shareable without
            defensive copy; the receptive-field validator (plan D2 / R6)
            already ran as part of validation.
        """
        self._config: NnTemporalConfig = config
        # Populated on ``fit()`` (T4).
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None
        self._device: torch.device | None = None
        self._device_resolved: str | None = None
        self._seed_used: int | None = None
        self._best_epoch: int | None = None
        # The fitted :class:`torch.nn.Module` — populated in ``fit`` (T4).
        self._module: nn.Module | None = None
        #: One dict per epoch after ``fit``; keys
        #: ``{"epoch", "train_loss", "val_loss"}`` (plan D6 / AC-2).  Public
        #: attribute (trailing underscore mirrors sklearn fitted-state
        #: convention) so the notebook's live-plot callback and
        #: :func:`bristol_ml.evaluation.plots.loss_curve` can consume it
        #: without reaching into private state.  Same semantics as
        #: Stage 10 ``NnMlpModel.loss_history_``.
        self.loss_history_: list[dict[str, float]] = []

    # ---------------------------------------------------------------------
    # Protocol members
    # ---------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        seed: int | None = None,
        epoch_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Fit the TCN on the aligned ``(features, target)`` pair.

        T3 ships this as a :class:`NotImplementedError` stub; T4 fills the
        body with the Stage 10 training-pipeline recipe adapted to
        sequences: four-stream seeding via
        :func:`bristol_ml.models.nn._training._seed_four_streams`,
        module construction (8-block dilated causal TCN per plan D1),
        scaler-buffer fitting on the train slice, ``_SequenceDataset``
        wrapping, internal 10 %-tail val split (Stage 10 D9 pattern
        inherited — **no** D8 offset per the Scope Diff cut), and the
        shared training loop via
        :func:`bristol_ml.models.nn._training.run_training_loop`.
        """
        raise NotImplementedError("NnTemporalModel.fit is not implemented yet; lands in plan T4.")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return predictions indexed to the valid-window timestamps.

        T3 ships this as a :class:`NotImplementedError` stub; T4 fills the
        body with a single-batch forward pass over a
        :class:`_SequenceDataset` wrapping ``features``, yielding one
        prediction per valid window (windows that have both their full
        ``seq_len`` of history *and* an aligned target index).

        Raises
        ------
        RuntimeError
            (Once implemented) if :meth:`fit` has not been called.
        """
        raise NotImplementedError(
            "NnTemporalModel.predict is not implemented yet; lands in plan T4."
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model to the registry-supplied file path.

        T3 ships this as a :class:`NotImplementedError` stub; T5 fills the
        body with the Stage 10 D5 envelope pattern inherited — a single
        joblib artefact containing ``state_dict_bytes``, ``config_dump``,
        ``feature_columns``, ``seq_len``, ``seed_used``, ``best_epoch``,
        ``loss_history``, ``fit_utc``, ``device_resolved``.  The ``seq_len``
        field is redundant with ``config_dump`` but rides alongside for
        an explicit round-trip guard (plan R7).
        """
        raise NotImplementedError("NnTemporalModel.save is not implemented yet; lands in plan T5.")

    @classmethod
    def load(cls, path: Path) -> NnTemporalModel:
        """Load a previously-saved :class:`NnTemporalModel` from ``path``.

        T3 ships this as a :class:`NotImplementedError` stub; T5 fills the
        body with the Stage 10 D5 load-path pattern: joblib envelope
        read, Pydantic re-validation of ``config_dump``, module
        reconstruction, ``torch.load(BytesIO(state_dict_bytes),
        weights_only=True, map_location="cpu")``, and a strict
        ``load_state_dict`` that fails loudly on any buffer / parameter
        mismatch (plan R3).
        """
        raise NotImplementedError("NnTemporalModel.load is not implemented yet; lands in plan T5.")

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit.

        Before :meth:`fit` has been called, ``fit_utc`` is ``None`` and
        ``feature_columns`` is empty — matching the Stage 4 protocol
        convention and the Stage 10 ``NnMlpModel.metadata`` precedent.
        Once fitted, ``hyperparameters`` carries the architecture
        summary (``seq_len``, ``num_blocks``, ``channels``,
        ``kernel_size``, ``dropout``, ``weight_norm``), the optimisation
        knobs (``learning_rate``, ``weight_decay``, ``batch_size``,
        ``max_epochs``, ``patience``), the resolved device (plan D6 /
        Stage 10 D11), and the seed actually used (plan D6 / Stage 10
        D7').  A downstream registry sidecar reader can reconstruct the
        fit conditions without reaching into private attributes.
        """
        cfg = self._config
        hyperparameters: dict[str, Any] = {
            "target_column": cfg.target_column,
            "seq_len": cfg.seq_len,
            "num_blocks": cfg.num_blocks,
            "channels": cfg.channels,
            "kernel_size": cfg.kernel_size,
            "dropout": cfg.dropout,
            "weight_norm": cfg.weight_norm,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            # ``device`` (config-requested) vs ``device_resolved`` (the
            # actual device ``fit`` landed on after auto-select) are
            # intentionally both recorded — same provenance split as
            # Stage 10 ``NnMlpModel.metadata``.
            "device": cfg.device,
        }
        if self._device_resolved is not None:
            hyperparameters["device_resolved"] = self._device_resolved
        if self._seed_used is not None:
            hyperparameters["seed_used"] = self._seed_used
        if self._best_epoch is not None:
            hyperparameters["best_epoch"] = self._best_epoch
        return ModelMetadata(
            name=_build_metadata_name(cfg),
            feature_columns=self._feature_columns,
            fit_utc=self._fit_utc,
            hyperparameters=hyperparameters,
        )


# ---------------------------------------------------------------------------
# Private helpers (module-level for testability and pickle safety).
# ---------------------------------------------------------------------------


def _build_metadata_name(config: NnTemporalConfig) -> str:
    """Build a metadata ``name`` matching ``ModelMetadata.name``'s regex.

    ``ModelMetadata.name`` is constrained to ``^[a-z][a-z0-9_.-]*$``.  The
    format is ``nn-temporal-b{num_blocks}-c{channels}-k{kernel_size}`` —
    e.g. ``nn-temporal-b8-c128-k3``, ``nn-temporal-b4-c32-k3`` for the
    CPU override recipe.  Encodes enough of the architecture to be
    greppable in a registry leaderboard without serialising every
    hyperparameter.
    """
    return f"nn-temporal-b{config.num_blocks}-c{config.channels}-k{config.kernel_size}"


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.nn.temporal``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.nn.temporal",
        description=(
            "Print the NnTemporalConfig JSON schema and the resolved config. "
            "Training and prediction are exercised via the evaluation "
            "harness (see `python -m bristol_ml.train model=nn_temporal`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=("Hydra overrides, e.g. model.num_blocks=4 model.channels=32 model.device=cpu"),
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 / plan NFR-5.

    Prints the :class:`~conf._schemas.NnTemporalConfig` JSON schema
    followed by the resolved config summary.  Returns ``0`` on success,
    ``2`` if a non-``nn_temporal`` model resolves from config (e.g. if
    the caller forgot ``model=nn_temporal``).
    """
    import json

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    print("NnTemporalConfig JSON schema:")
    print(json.dumps(NnTemporalConfig.model_json_schema(), indent=2))
    print()
    print(
        "To fit: python -m bristol_ml.train model=nn_temporal "
        "evaluation.rolling_origin.fixed_window=true "
        "evaluation.rolling_origin.step=168"
    )

    cfg = load_config(overrides=["model=nn_temporal", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, NnTemporalConfig):
        print(
            "No NnTemporalConfig resolved. Ensure `model=nn_temporal` or a "
            "matching override is present; got "
            f"{type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    nn_cfg = cfg.model
    logger.info(
        "NnTemporalConfig: seq_len={} num_blocks={} channels={} "
        "kernel_size={} dropout={} weight_norm={} learning_rate={} "
        "batch_size={} max_epochs={} patience={} device={} "
        "target_column={}",
        nn_cfg.seq_len,
        nn_cfg.num_blocks,
        nn_cfg.channels,
        nn_cfg.kernel_size,
        nn_cfg.dropout,
        nn_cfg.weight_norm,
        nn_cfg.learning_rate,
        nn_cfg.batch_size,
        nn_cfg.max_epochs,
        nn_cfg.patience,
        nn_cfg.device,
        nn_cfg.target_column,
    )
    print()
    print(f"seq_len={nn_cfg.seq_len}")
    print(f"num_blocks={nn_cfg.num_blocks}")
    print(f"channels={nn_cfg.channels}")
    print(f"kernel_size={nn_cfg.kernel_size}")
    print(f"dropout={nn_cfg.dropout}")
    print(f"weight_norm={nn_cfg.weight_norm}")
    print(f"learning_rate={nn_cfg.learning_rate}")
    print(f"weight_decay={nn_cfg.weight_decay}")
    print(f"batch_size={nn_cfg.batch_size}")
    print(f"max_epochs={nn_cfg.max_epochs}")
    print(f"patience={nn_cfg.patience}")
    print(f"device={nn_cfg.device}")
    print(f"target_column={nn_cfg.target_column}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
