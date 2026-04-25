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
YAML group.  T3 added the class surface (``__init__``, ``metadata``,
standalone CLI entry point) with ``fit`` / ``predict`` / ``save`` /
``load`` as :class:`NotImplementedError` stubs.  T4 (this commit) fills
``fit`` and ``predict`` with the TCN body, causal padding, and the
shared training-loop integration; T5 fills ``save`` / ``load`` with the
single-joblib envelope.

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
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from bristol_ml.models.nn._training import _seed_four_streams, run_training_loop
from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import NnTemporalConfig

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

__all__ = ["NnTemporalModel"]


# ---------------------------------------------------------------------------
# Format tag for the dict-envelope-of-primitives written by
# :meth:`NnTemporalModel.save`.  Bumped if the envelope schema changes;
# load rejects any tag it does not recognise so a future schema migration
# is never silent.  Stage 12 T3 introduced the tag at the same time as
# the joblib → skops migration (D10 Ctrl+G reversal).
# ---------------------------------------------------------------------------

_NN_TEMPORAL_ENVELOPE_FORMAT = "nn-temporal-state-v1"


# ---------------------------------------------------------------------------
# Valid device strings (plan D6 / Stage 10 D11 inheritance).  Duplicated
# from ``mlp.py`` rather than imported so a future ``_select_device``
# consolidation into ``_training.py`` is a mechanical search-and-replace
# and the log message family ("NnTemporalModel: ...") stays honest.
# ---------------------------------------------------------------------------

_ALLOWED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")


def _select_device(preference: str) -> torch.device:
    """Resolve a ``NnTemporalConfig.device`` string to a concrete :class:`torch.device`.

    Stage 10 D11 inheritance — same resolution order as
    :func:`bristol_ml.models.nn.mlp._select_device`.  Duplicated here so
    the INFO log line names the right model family and so a future
    consolidation into ``_training.py`` stays mechanical.

    Resolution order when ``preference == "auto"``:

    1. :func:`torch.cuda.is_available` → ``cuda``;
    2. :func:`torch.backends.mps.is_available` → ``mps``;
    3. ``cpu``.

    Explicit values (``"cpu"`` / ``"cuda"`` / ``"mps"``) are honoured
    verbatim; unknown values raise :class:`ValueError` rather than
    falling back silently to CPU.
    """
    import torch

    if preference not in _ALLOWED_DEVICES:
        raise ValueError(
            f"NnTemporalConfig.device must be one of {_ALLOWED_DEVICES!r}; got {preference!r}."
        )
    if preference == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(preference)
    logger.info(
        "NnTemporalModel: device preference={!r} resolved to {!r}.",
        preference,
        str(resolved),
    )
    return resolved


# ---------------------------------------------------------------------------
# Helper — DataLoader device wrapper
# ---------------------------------------------------------------------------


class _DeviceDataLoader:
    """Iterable adapter that moves each batch from a ``DataLoader`` to ``device``.

    The Stage 10 MLP's ``fit`` pre-loads the training tensors onto the
    resolved device *before* constructing the :class:`TensorDataset`, so
    its DataLoader already yields device tensors and
    :func:`run_training_loop`'s docstring-level "batches already on
    device" contract holds trivially.  Stage 11's :class:`_SequenceDataset`
    returns CPU tensors (from numpy via :func:`torch.from_numpy`) because
    pre-loading all windows onto the GPU would reconstruct the
    eagerly-materialised memory footprint plan D7 cut (domain research
    §4).  We therefore move each batch at iteration time; on CPU this is
    a no-op, on CUDA / MPS it is a single ``.to(device)`` per batch.

    Iterable semantics (not generator): :meth:`__iter__` returns a fresh
    iterator on each call so the outer epoch loop inside
    :func:`run_training_loop` can re-iterate across epochs.  A bare
    generator returned by ``yield`` at module level would be exhausted
    after one epoch and silently stop training.
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for x_batch, y_batch in self._loader:
            yield x_batch.to(self._device), y_batch.to(self._device)

    def __len__(self) -> int:
        return len(self._loader)


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
# TCN module (plan D1 — dilated causal conv stack, Bai et al. 2018 recipe).
# Lazy-built for the same reason as :class:`_SequenceDataset` — keeps the
# scaffold / CLI import path torch-free.  The installed class follows the
# Stage 10 Gotcha 1 pickleability recipe (``__module__`` / ``__qualname__``
# patched + module-level ``sys.modules`` install).
# ---------------------------------------------------------------------------

_nn_temporal_module_cls: type[nn.Module] | None = None


def _build_temporal_module_class() -> type[nn.Module]:
    """Return the :class:`torch.nn.Module` subclass used by the TCN.

    Cached in :data:`_nn_temporal_module_cls` after first construction so
    repeated :meth:`NnTemporalModel.fit` calls don't re-build the class
    object.  The class is defined inside a function because ``torch`` is
    imported lazily; pickleability across the save/load round-trip is
    achieved by (a) setting ``__module__`` / ``__qualname__`` to the
    import path and (b) installing the class as a module attribute on
    :mod:`bristol_ml.models.nn.temporal` before returning it.  Same
    recipe as Stage 10 ``_NnMlpModuleImpl`` — see
    ``src/bristol_ml/models/nn/CLAUDE.md`` "PyTorch specifics" gotcha 1.

    The ``_TemporalBlockImpl`` inner class (a single residual TCN block)
    is also installed onto ``sys.modules[__name__]`` so a future
    ``torch.save`` on a module instance that contains a block survives
    the full getattr-based pickle protocol.
    """
    global _nn_temporal_module_cls
    if _nn_temporal_module_cls is not None:
        return _nn_temporal_module_cls

    import torch
    import torch.nn.functional as F
    from torch import nn

    try:
        # PyTorch 2.1+ moved ``weight_norm`` to the parametrizations API.
        from torch.nn.utils.parametrizations import weight_norm as _weight_norm
    except ImportError:  # pragma: no cover — torch < 2.1 fallback
        from torch.nn.utils import weight_norm as _weight_norm

    class _TemporalBlockImpl(nn.Module):
        """Single residual TCN block with left-only causal padding.

        Two stacked dilated ``Conv1d`` layers, each followed by
        :class:`~torch.nn.LayerNorm` (over the channel dimension — not
        over time, which would leak BatchNorm-style running statistics
        across timesteps per domain §6 pitfall 4), ReLU, and dropout.
        A 1x1 convolution provides the residual skip when the channel
        count changes between the block's input and output.

        Causal padding is realised via :func:`torch.nn.functional.pad`
        with ``pad = (left, 0)`` — right-side zero, left-side
        ``(kernel_size - 1) * dilation`` — applied *before* a
        ``Conv1d(padding=0)`` call.  This is the Bai et al. 2018 recipe
        with the domain §6 pitfall-4 mitigation wired in by construction:
        the convolution cannot see any input position strictly to the
        right of the current output position, no matter the dilation.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int,
            dilation: int,
            dropout: float,
            weight_norm: bool,
        ) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.dilation = dilation
            self.left_pad = (kernel_size - 1) * dilation

            conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=0, dilation=dilation
            )
            conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, padding=0, dilation=dilation
            )
            if weight_norm:
                conv1 = _weight_norm(conv1)
                conv2 = _weight_norm(conv2)
            self.conv1 = conv1
            self.conv2 = conv2
            # LayerNorm(channels) applied to ``x.transpose(1, 2)`` (shape
            # ``(B, L, C)``) normalises across C for each (B, L) pair —
            # the conventional TCN choice.
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)
            # ReLU is stateless; one module shared across both
            # activations is fine but instantiating two matches
            # weight-norm placement conventions in Bai 2018.
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            if in_channels == out_channels:
                self.downsample: nn.Module | None = None
            else:
                self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """``x: (B, C_in, L)`` → ``(B, C_out, L)`` with causal receptive field."""
            residual = x if self.downsample is None else self.downsample(x)

            y = F.pad(x, (self.left_pad, 0))
            y = self.conv1(y)
            # LayerNorm over channels: transpose (B, C, L) ↔ (B, L, C).
            y = self.norm1(y.transpose(1, 2)).transpose(1, 2)
            y = self.relu1(y)
            y = self.dropout1(y)

            y = F.pad(y, (self.left_pad, 0))
            y = self.conv2(y)
            y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
            y = self.relu2(y)
            y = self.dropout2(y)

            return y + residual

    class _NnTemporalModuleImpl(nn.Module):
        """TCN with z-score input + target buffers (plan D1 + D5 + Stage 10 D4).

        Forward signature: ``(B, seq_len, n_features)`` in → ``(B,)`` out,
        both ``float32``.  The output is in **normalised target space**;
        the caller (:meth:`NnTemporalModel.predict`) applies the inverse
        ``y = y_norm * target_std + target_mean`` transform.  Training
        loss is computed against the **normalised** target yielded by the
        ``_SequenceDataset`` wrapper (MSELoss operates on O(1) scale,
        matching Stage 10's convention).
        """

        def __init__(
            self,
            *,
            input_dim: int,
            seq_len: int,
            num_blocks: int,
            channels: int,
            kernel_size: int,
            dropout: float,
            weight_norm: bool,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.seq_len = seq_len
            self.num_blocks = num_blocks
            self.channels = channels
            self.kernel_size = kernel_size
            self.dropout_p = dropout
            self.weight_norm = weight_norm

            # Scaler buffers — registered deterministically at construction
            # time so ``load_state_dict(strict=True)`` has the keys in
            # place before the loaded bytes overwrite them (Stage 10 D4
            # / Gotcha 3).  Placeholder zeros/ones keep pre-fit inference
            # numerically finite; :meth:`NnTemporalModel.fit` overwrites
            # them from the training slice's column statistics.
            self.register_buffer("feature_mean", torch.zeros(input_dim))
            self.register_buffer("feature_std", torch.ones(input_dim))
            self.register_buffer("target_mean", torch.zeros(1))
            self.register_buffer("target_std", torch.ones(1))

            blocks: list[nn.Module] = []
            in_ch = input_dim
            for block_idx in range(num_blocks):
                dilation = 2**block_idx
                blocks.append(
                    _TemporalBlockImpl(
                        in_ch,
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                        weight_norm=weight_norm,
                    )
                )
                in_ch = channels
            self.blocks: nn.ModuleList = nn.ModuleList(blocks)
            # 1x1 head maps channel axis down to 1 scalar per timestep;
            # the forward pass then reads the last timestep only.
            self.head: nn.Module = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """``x: (B, seq_len, n_features)`` → ``(B,)`` normalised predictions."""
            # Normalise features via the scaler buffers (plan D5).
            z = (x - self.feature_mean) / self.feature_std
            # Transpose to the (B, C, L) layout Conv1d expects.
            y = z.transpose(1, 2)
            for block in self.blocks:
                y = block(y)
            # (B, channels, L) → (B, 1, L) → (B, 1) → (B,).
            y = self.head(y)
            y = y[:, :, -1]
            return y.squeeze(-1)

    _TemporalBlockImpl.__module__ = "bristol_ml.models.nn.temporal"
    _TemporalBlockImpl.__qualname__ = "_TemporalBlockImpl"
    _NnTemporalModuleImpl.__module__ = "bristol_ml.models.nn.temporal"
    _NnTemporalModuleImpl.__qualname__ = "_NnTemporalModuleImpl"
    # Install into module namespace so pickle's
    # ``getattr(sys.modules[__module__], __qualname__)`` lookup resolves
    # — Stage 10 Gotcha 1 inherited.  Without this, any future
    # ``copy.deepcopy`` on a fitted ``NnTemporalModel`` raises
    # ``AttributeError`` despite the ``__module__`` patch.
    sys.modules[__name__]._TemporalBlockImpl = _TemporalBlockImpl  # type: ignore[attr-defined]
    sys.modules[__name__]._NnTemporalModuleImpl = _NnTemporalModuleImpl  # type: ignore[attr-defined]
    _nn_temporal_module_cls = _NnTemporalModuleImpl
    return _NnTemporalModuleImpl


def _make_tcn(input_dim: int, config: NnTemporalConfig) -> nn.Module:
    """Construct a TCN :class:`torch.nn.Module` from ``config``.

    Thin factory that routes through :func:`_build_temporal_module_class`
    so the :class:`torch.nn.Module` subclass is only defined after
    ``torch`` has been imported — keeps the scaffold / CLI import path
    torch-free.  The returned module's ``input_dim`` attribute is
    load-bearing for the save/load round-trip (T5): the loaded
    envelope carries ``feature_columns`` and the reconstruction path
    calls ``_make_tcn(len(feature_columns), config)`` to rebuild a
    key-compatible module before ``load_state_dict(strict=True)``.
    """
    cls = _build_temporal_module_class()
    return cls(
        input_dim=input_dim,
        seq_len=config.seq_len,
        num_blocks=config.num_blocks,
        channels=config.channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        weight_norm=config.weight_norm,
    )


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
        # Last ``seq_len`` rows of the training feature frame — prepended
        # to the predict input so every predict row receives a window of
        # ``seq_len`` history.  Gives the harness the
        # ``len(predict) == len(features)`` length-match it needs
        # (evaluation-layer contract §5).  Populated in ``fit`` (T4).
        self._warmup_features: pd.DataFrame | None = None
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

        Pipeline (mirrors Stage 10 :meth:`NnMlpModel.fit` modulo the
        sequence-window step):

        1. Shape / length guard — ``len(features) == len(target)`` and
           ``len(features) > seq_len`` (otherwise no valid window exists).
        2. Feature-column resolution (``config.feature_columns`` wins
           when set; otherwise the full input-frame column order is used,
           matching the Linear / SARIMAX / MLP convention).
        3. Device selection via :func:`_select_device` and four-stream
           seeding via
           :func:`bristol_ml.models.nn._training._seed_four_streams`
           (plan D6 / Stage 10 D7' inheritance).
        4. Train / validation split — an **internal 10 % tail** of the
           training slice (Stage 10 D9 pattern; **no D8 offset** per the
           Scope Diff cut).  The tail length is clamped to
           ``max(seq_len + 1, n // 10)`` so the val partition always yields
           at least one window.
        5. Scaler buffers (``feature_mean`` / ``feature_std`` /
           ``target_mean`` / ``target_std``) fitted from the train
           partition only.  Zero-variance feature columns are clamped to
           ``std=1`` and logged (a zero-std column would otherwise make
           the z-score produce ``inf`` / ``nan``).
        6. Module construction via :func:`_make_tcn` and movement to the
           resolved device; the scaler buffers are copied in via
           ``register_buffer`` so they ride inside ``state_dict()``.
        7. ``_SequenceDataset`` for train + val (plan D7 lazy windowing),
           wrapped in :class:`torch.utils.data.DataLoader` (seeded via a
           :class:`torch.Generator`, ``num_workers=0``) and then
           :class:`_DeviceDataLoader` so each batch reaches the resolved
           device at iteration time (:func:`run_training_loop`'s
           batches-already-on-device contract).  Training operates on
           the **normalised** target (MSE scale O(1)); features are
           normalised inside ``forward``.
        8. Shared training loop via
           :func:`bristol_ml.models.nn._training.run_training_loop` —
           Adam + MSELoss, patience-based early stopping, best-epoch
           weight restore via a strict ``load_state_dict``.
        9. Re-entrancy: ``loss_history_`` / ``_best_epoch`` / ``_module``
           are discarded at the top of ``fit`` — the cold-start-per-fold
           contract (Stage 10 D8 pattern inherited).  ``_warmup_features``
           is overwritten with the last ``seq_len`` rows of the training
           features so :meth:`predict` can prepend them.

        Parameters
        ----------
        features:
            Feature frame; ``features.index`` is not constrained here
            (the harness passes a UTC-aware ``DatetimeIndex`` but the
            dataset converts to numpy and discards the index).
        target:
            Aligned target series; ``len(target) == len(features)``.
        seed:
            Optional seed override.  ``None`` uses ``config.seed``
            (which may itself be ``None`` — in that case the per-fold
            determinism is the caller's responsibility, or ``0`` is used
            as the deterministic fallback so a bare ``fit(X, y)`` call
            is still reproducible).
        epoch_callback:
            Optional callable invoked after each epoch with the dict
            ``{"epoch", "train_loss", "val_loss"}`` — the live-plot seam
            for the notebook's Demo moment (plan D6 / AC-3).
        """
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader

        # --- 1. Guards --------------------------------------------------
        if len(features) != len(target):
            raise ValueError(
                "NnTemporalModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        if len(features) == 0:
            raise ValueError("NnTemporalModel.fit requires at least one training row.")

        cfg = self._config

        if len(features) <= cfg.seq_len:
            raise ValueError(
                "NnTemporalModel.fit requires len(features) > seq_len so at "
                "least one window + aligned target exists; got "
                f"len(features)={len(features)} vs seq_len={cfg.seq_len}."
            )

        # --- 2. Feature-column resolution ------------------------------
        if cfg.feature_columns is not None:
            feature_cols: tuple[str, ...] = tuple(cfg.feature_columns)
            missing = [c for c in feature_cols if c not in features.columns]
            if missing:
                raise ValueError(
                    "NnTemporalModel.fit: configured feature_columns not "
                    f"present in features: {missing!r}.  Available: "
                    f"{list(features.columns)!r}."
                )
            X_df = features[list(feature_cols)]
        else:
            feature_cols = tuple(features.columns)
            X_df = features

        # --- 3. Device + seed ------------------------------------------
        device = _select_device(cfg.device)
        effective_seed = seed if seed is not None else cfg.seed
        if effective_seed is None:
            # Fall back to a deterministic default so a bare ``fit(X, y)``
            # is still reproducible; the harness path passes the per-fold
            # seed explicitly.
            effective_seed = 0
        _seed_four_streams(int(effective_seed), device)

        # --- 4. Train / val tail split ---------------------------------
        # Stage 10 D9 pattern inherited; val tail clamped to
        # ``seq_len + 1`` so at least one validation window exists.  No
        # D8 offset — the Scope Diff cut (plan §Scope Diff / X-entries).
        n = len(X_df)
        n_val = max(cfg.seq_len + 1, n // 10)
        n_train = n - n_val
        if n_train <= cfg.seq_len:
            raise ValueError(
                "NnTemporalModel.fit: after splitting off a val tail of "
                f"{n_val} rows, the train slice has only {n_train} rows, "
                f"which is not > seq_len={cfg.seq_len}. Need at least "
                f"{2 * cfg.seq_len + 2} total rows."
            )

        train_X_df = X_df.iloc[:n_train]
        train_y = target.iloc[:n_train]
        val_X_df = X_df.iloc[n_train:]
        val_y = target.iloc[n_train:]

        # --- 5. Scaler buffers (plan D5 / Stage 10 D4 inheritance) -----
        train_X_np = np.asarray(train_X_df.to_numpy(), dtype=np.float32)
        feat_mean = train_X_np.mean(axis=0).astype(np.float32)
        feat_std = train_X_np.std(axis=0).astype(np.float32)
        zero_var_mask = feat_std < 1e-12
        if zero_var_mask.any():
            zero_cols = [feature_cols[i] for i in np.where(zero_var_mask)[0]]
            logger.info(
                "NnTemporalModel.fit: zero-variance feature columns clamped to std=1: {}.",
                zero_cols,
            )
            feat_std = np.where(zero_var_mask, 1.0, feat_std).astype(np.float32)

        train_y_np = np.asarray(train_y.to_numpy(), dtype=np.float32)
        tgt_mean = float(train_y_np.mean())
        tgt_std = float(train_y_np.std())
        if tgt_std < 1e-12:
            logger.warning(
                "NnTemporalModel.fit: target std is ~0 on the train slice; clamping to 1."
            )
            tgt_std = 1.0

        # --- 6. Module construction ------------------------------------
        module = _make_tcn(input_dim=len(feature_cols), config=cfg)
        with torch.no_grad():
            module.feature_mean.copy_(torch.from_numpy(feat_mean))  # type: ignore[union-attr]
            module.feature_std.copy_(torch.from_numpy(feat_std))  # type: ignore[union-attr]
            module.target_mean.copy_(  # type: ignore[union-attr]
                torch.tensor([tgt_mean], dtype=torch.float32)
            )
            module.target_std.copy_(  # type: ignore[union-attr]
                torch.tensor([tgt_std], dtype=torch.float32)
            )
        module = module.to(device)

        # --- 7. DataLoaders (plan D7 lazy windowing) -------------------
        # Pre-normalise the target so MSE loss operates on O(1) scale;
        # features are normalised inside ``forward`` via the scaler
        # buffers (plan D5 / Stage 10 D4 inheritance).
        train_y_norm = pd.Series(
            ((train_y_np - tgt_mean) / tgt_std).astype(np.float32),
            index=train_y.index,
        )
        val_y_np = np.asarray(val_y.to_numpy(), dtype=np.float32)
        val_y_norm = pd.Series(
            ((val_y_np - tgt_mean) / tgt_std).astype(np.float32),
            index=val_y.index,
        )

        train_ds = _SequenceDataset(train_X_df, train_y_norm, cfg.seq_len)
        val_ds = _SequenceDataset(val_X_df, val_y_norm, cfg.seq_len)

        dl_generator = torch.Generator(device="cpu")
        dl_generator.manual_seed(int(effective_seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            generator=dl_generator,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        train_loader_dev = _DeviceDataLoader(train_loader, device)
        val_loader_dev = _DeviceDataLoader(val_loader, device)

        optimizer = optim.Adam(
            module.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()

        # Reset re-entrant state before appending per-epoch entries.
        loss_history: list[dict[str, float]] = []

        best_state_dict, best_epoch = run_training_loop(
            module,
            train_loader_dev,
            val_loader_dev,
            optimiser=optimizer,
            criterion=loss_fn,
            device=device,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            loss_history=loss_history,
            epoch_callback=epoch_callback,
        )

        # Restore best-epoch weights on the fit device so predict() stays
        # on ``device``.  The shared helper returns CPU clones.
        module.load_state_dict(
            {k: v.to(device) for k, v in best_state_dict.items()},
            strict=True,
        )

        # --- 8. Publish state -----------------------------------------
        self._feature_columns = feature_cols
        self._fit_utc = datetime.now(UTC)
        self._device = device
        self._device_resolved = str(device)
        self._seed_used = int(effective_seed)
        self._best_epoch = best_epoch
        self._module = module
        self.loss_history_ = loss_history
        # Warmup prefix: last seq_len rows of the training feature frame
        # so :meth:`predict` can prepend them and yield one prediction per
        # input row (evaluation-layer length-match contract §5).
        self._warmup_features = features[list(feature_cols)].tail(cfg.seq_len).copy()

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return predictions indexed to ``features.index``.

        The TCN's natural predict output length is
        ``len(features) - seq_len`` (one prediction per full-history
        window).  The Stage 6 evaluation harness, however, expects
        ``len(y_pred) == len(y_test)`` so its ``metric(y_test, y_pred)``
        call does not raise on a length mismatch.  The reconciliation:
        :meth:`fit` stashed the last ``seq_len`` rows of the training
        feature frame in ``self._warmup_features``; :meth:`predict`
        prepends them, giving ``len(combined) = seq_len + len(features)``
        and therefore ``len(predict) == len(features)`` — one prediction
        per input row, with the first prediction using only warmup
        history and each subsequent prediction consuming one more row of
        ``features``.  The returned :class:`pandas.Series` carries
        ``features.index`` so downstream ``.reindex`` / ``.align`` work
        as expected.

        ``features`` must carry every column the model was fit on; extra
        columns are silently ignored (mirrors the Linear / SARIMAX /
        MLP contract).  Predict-before-fit raises
        :class:`RuntimeError` (Stage 4 protocol convention).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If any of ``self._feature_columns`` is missing from
            ``features``.
        """
        import torch
        from torch.utils.data import DataLoader

        if self._module is None or self._device is None or self._warmup_features is None:
            raise RuntimeError("NnTemporalModel must be fit() before predict().")

        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"NnTemporalModel.predict: features frame is missing fitted columns {missing!r}. "
                f"Available: {list(features.columns)!r}."
            )

        if len(features) == 0:
            return pd.Series(
                np.empty(0, dtype=np.float64),
                index=features.index,
                name=self._config.target_column,
            )

        X_df = features[list(self._feature_columns)]
        # Prepend the warmup prefix so every row of ``features`` gets a
        # prediction — one window per input row (evaluation-layer §5
        # length-match contract).  ``pd.concat`` stacks rows in order;
        # ``_SequenceDataset`` is position-based (``to_numpy()`` discards
        # the index) so overlapping timestamps would not confuse it.
        combined = pd.concat([self._warmup_features, X_df], axis=0)
        # Dummy target — :class:`_SequenceDataset` requires one for its
        # ``(window, target)`` contract, but predict consumes only the
        # window; the target slot is ignored downstream.
        dummy_target = pd.Series(
            np.zeros(len(combined), dtype=np.float32),
            index=combined.index,
        )
        seq_len = self._config.seq_len
        dataset = _SequenceDataset(combined, dummy_target, seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        loader_dev = _DeviceDataLoader(loader, self._device)

        self._module.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch, _ in loader_dev:
                y_norm = self._module(x_batch)
                # Inverse target normalisation; scalers live on the module.
                y = (
                    y_norm * self._module.target_std  # type: ignore[union-attr]
                    + self._module.target_mean  # type: ignore[union-attr]
                )
                chunks.append(y.detach().cpu().numpy().astype(np.float64))
        y_np = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        return pd.Series(
            y_np,
            index=features.index,
            name=self._config.target_column,
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model to the registry-supplied file path.

        Stage 12 D10 (Ctrl+G reversal): the project moved off ``joblib``
        and onto :mod:`skops.io` for security — the serving layer is a
        network-facing deserialiser and ``joblib.load`` on an
        attacker-controlled artefact is RCE.  Plan D5's single-envelope
        layout is preserved; the fields that did **not** round-trip
        natively through skops's restricted unpickler
        (:class:`pandas.DataFrame` for ``warmup_features``, ``datetime``
        for ``fit_utc``, ``tuple`` for ``feature_columns``) are
        flattened to skops-safe primitives — the warmup DataFrame
        becomes an ``(n_warmup, n_features)`` numpy array plus its
        ``int64`` nanos-since-epoch index plus a tz string, ``fit_utc``
        becomes an ISO-8601 string, ``feature_columns`` becomes a list —
        so no project trust-list registration is needed.

        Plan D5: ``path`` is the single artefact file path (the Stage 9
        registry hard-codes ``artefact/model.<ext>``).  The envelope
        written carries the ``state_dict`` bytes (via ``torch.save`` to
        a ``BytesIO`` — scaler buffers ride inside), the
        ``NnTemporalConfig.model_dump()``, the resolved feature column
        list, the ``seq_len`` (redundant with ``config_dump`` but rides
        alongside for the explicit round-trip guard — plan R7), the
        warmup-prefix arrays (so :meth:`predict` can produce one
        prediction per input row after a load — evaluation-layer §5
        length-match contract), and the provenance scalars
        (``seed_used``, ``best_epoch``, ``loss_history``,
        ``fit_utc_isoformat``, ``device_resolved``).

        The write is atomic — :func:`bristol_ml.models.io.save_skops`
        stages to a sibling ``.tmp`` and renames via :func:`os.replace`,
        matching the Stage 4 contract.  No sibling ``model.pt`` or
        ``hyperparameters.json`` file is created — the D5 single-envelope
        disposition is structurally enforced by
        ``test_nn_temporal_save_writes_single_joblib_file_at_given_path``
        (renamed in spirit at T3; the test still pins single-file).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called — save-before-fit is
            undefined per the Stage 4 :class:`Model` protocol.
        """
        import io as _io

        import torch

        from bristol_ml.models.io import save_skops

        if self._module is None or self._warmup_features is None:
            raise RuntimeError(
                "NnTemporalModel.save requires fit() to have been called first; "
                "state_dict and warmup prefix are undefined before fit."
            )

        # state_dict → bytes.  ``.cpu()`` on every tensor so the envelope
        # is portable across devices; ``torch.load(..., map_location="cpu")``
        # in :meth:`load` mirrors this (Stage 10 D5 recipe inherited).
        cpu_state_dict = {k: v.detach().cpu() for k, v in self._module.state_dict().items()}
        buf = _io.BytesIO()
        torch.save(cpu_state_dict, buf)
        state_dict_bytes: bytes = buf.getvalue()

        # Break the warmup DataFrame down to skops-safe primitives.
        # ``DatetimeIndex.asi8`` is the canonical int64 nanos-since-epoch
        # vector; the tz and freq are captured as portable strings so
        # tzinfo subclasses and ``pd.tseries.offsets.BaseOffset`` (each a
        # custom type under skops) stay out of the artefact entirely.
        # The freq string is load-bearing for the warmup-DataFrame
        # round-trip equality check (``pd.testing.assert_frame_equal``
        # is strict on index ``freq``); without it a freshly-fitted
        # hourly-index frame round-trips to a ``freq=None`` index and
        # the AC-4 round-trip test fails loudly.
        warmup_df = self._warmup_features
        warmup_index = warmup_df.index
        warmup_index_tz: str | None = str(warmup_index.tz) if warmup_index.tz is not None else None
        warmup_index_freq: str | None = (
            warmup_index.freqstr if warmup_index.freq is not None else None
        )

        envelope: dict[str, Any] = {
            "format": _NN_TEMPORAL_ENVELOPE_FORMAT,
            "state_dict_bytes": state_dict_bytes,
            "config_dump": self._config.model_dump(),
            # List rather than tuple — both are skops-safe but the list
            # keeps the envelope a pure JSON-shape primitive bag.  Load
            # re-tuples on the way out to preserve the immutability
            # downstream code relies on.
            "feature_columns": list(self._feature_columns),
            # ``seq_len`` is redundant with ``config_dump["seq_len"]`` but
            # rides alongside for the explicit round-trip guard (plan R7
            # / test_nn_temporal_save_and_load_round_trips_seq_len_and_state_dict).
            "seq_len": int(self._config.seq_len),
            # Warmup prefix — broken down to numpy values + int64 nanos +
            # tz string + column list so skops's restricted unpickler
            # accepts the load without trust-list registration.  ``predict``
            # reconstructs the DataFrame on the way out so the contract
            # ``len(predict) == len(features)`` (evaluation-layer §5) is
            # preserved end-to-end.
            "warmup_values": np.ascontiguousarray(warmup_df.to_numpy(dtype=np.float64)),
            "warmup_index_nanos": np.asarray(warmup_index.asi8, dtype=np.int64),
            "warmup_index_tz": warmup_index_tz,
            "warmup_index_freq": warmup_index_freq,
            "warmup_columns": list(warmup_df.columns),
            "seed_used": self._seed_used,
            "best_epoch": self._best_epoch,
            "loss_history": [dict(entry) for entry in self.loss_history_],
            # ISO-8601 string keeps ``datetime`` / ``tzinfo`` out of the
            # artefact (each is a custom type under skops and would each
            # need a trust-list entry).
            "fit_utc_isoformat": (self._fit_utc.isoformat() if self._fit_utc is not None else None),
            "device_resolved": self._device_resolved,
        }
        save_skops(envelope, path)

    @classmethod
    def load(cls, path: Path) -> NnTemporalModel:
        """Load a previously-saved :class:`NnTemporalModel` from ``path``.

        Stage 12 D10 (Ctrl+G reversal): reads via :mod:`skops.io`'s
        restricted unpickler.  The envelope schema is the
        :data:`_NN_TEMPORAL_ENVELOPE_FORMAT`-tagged dict-of-primitives
        written by :meth:`save`; the warmup ``pd.DataFrame`` is
        reconstructed from the ``warmup_values`` numpy array + the
        ``warmup_index_nanos`` int64 vector + the ``warmup_index_tz``
        string + the ``warmup_columns`` list, and the ``fit_utc``
        ``datetime`` is reconstructed from the ``fit_utc_isoformat``
        ISO-8601 string — both reconstructions stay on the
        skops-default-trusted primitives so no project trust-list entry
        is needed.

        Plan D5: reconstructs :class:`NnTemporalConfig` from
        ``config_dump`` (Pydantic re-validation catches schema drift —
        plan R2), instantiates an ``NnTemporalModel``, materialises the
        ``state_dict`` via ``torch.load(BytesIO(state_dict_bytes),
        weights_only=True, map_location="cpu")``, and calls
        ``load_state_dict(strict=True)``.  ``weights_only=True`` keeps
        PyTorch 2.6+'s safety rail active on the inner bytes payload;
        ``strict=True`` is load-bearing for plan R3 (a dropped scaler
        buffer would silently break inverse-normalisation at
        :meth:`predict` otherwise).

        The loaded model lives on CPU regardless of the device it was
        fitted on — :meth:`predict` moves tensors to ``self._device`` at
        call time and the buffers ride along with the module.  The
        warmup-prefix DataFrame is restored so ``predict`` yields
        ``len(predict) == len(features)`` (evaluation-layer §5).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist (propagated from :mod:`skops.io`).
        TypeError
            If the loaded envelope's ``format`` tag does not match
            :data:`_NN_TEMPORAL_ENVELOPE_FORMAT` (a structural signal it
            was not produced by a fitted ``NnTemporalModel``).
        ValueError
            If the envelope is missing ``feature_columns`` or the
            warmup-prefix fields, or if the envelope's ``seq_len``
            disagrees with the one in ``config_dump`` (plan R7 explicit
            round-trip guard).
        """
        import io as _io

        import torch

        from bristol_ml.models.io import load_skops

        envelope = load_skops(path)

        # Format-tag discriminator — TypeError on mismatch so a
        # non-NnTemporalModel artefact (e.g. a NaiveModel envelope or a
        # raw dict) cannot be silently loaded as if it were one.
        if not isinstance(envelope, dict) or envelope.get("format") != _NN_TEMPORAL_ENVELOPE_FORMAT:
            actual_tag: object = (
                envelope.get("format") if isinstance(envelope, dict) else type(envelope).__name__
            )
            raise TypeError(
                "NnTemporalModel.load: artefact at "
                f"{path!s} does not carry the expected format tag "
                f"{_NN_TEMPORAL_ENVELOPE_FORMAT!r}; got {actual_tag!r}."
            )

        # Pydantic re-validation — schema-drift guard (plan R2).
        config = NnTemporalConfig.model_validate(envelope["config_dump"])

        # Explicit round-trip guard (plan R7): the top-level ``seq_len``
        # field and the value inside ``config_dump`` must agree.
        envelope_seq_len = envelope.get("seq_len")
        if envelope_seq_len is not None and int(envelope_seq_len) != int(config.seq_len):
            raise ValueError(
                "NnTemporalModel.load: envelope seq_len field "
                f"({envelope_seq_len}) disagrees with config_dump seq_len "
                f"({config.seq_len}); artefact is corrupted (plan R7)."
            )

        model = cls(config)

        # Rebuild the nn.Module with the same input_dim the fit used.
        feature_cols: tuple[str, ...] = tuple(envelope["feature_columns"])
        if not feature_cols:
            raise ValueError(
                "NnTemporalModel.load: envelope carries empty feature_columns; "
                "this envelope was not produced by a fitted NnTemporalModel."
            )
        module = _make_tcn(input_dim=len(feature_cols), config=config)

        # Materialise the saved state_dict on CPU and load strictly — a
        # missing or extra key (e.g. a dropped scaler buffer) fails
        # loudly here rather than silently skewing predictions later.
        buf = _io.BytesIO(envelope["state_dict_bytes"])
        state_dict = torch.load(buf, weights_only=True, map_location="cpu")
        module.load_state_dict(state_dict, strict=True)

        # Reconstruct the warmup DataFrame from skops-safe primitives.
        # ``warmup_values`` carries the raw float64 values; the index is
        # restored from int64 nanos-since-epoch + the captured tz string.
        warmup_values = envelope.get("warmup_values")
        warmup_index_nanos = envelope.get("warmup_index_nanos")
        warmup_columns = envelope.get("warmup_columns")
        if warmup_values is None or warmup_index_nanos is None or warmup_columns is None:
            raise ValueError(
                "NnTemporalModel.load: envelope is missing one of the "
                "warmup_values / warmup_index_nanos / warmup_columns "
                "fields; this artefact predates the Stage 12 D10 envelope "
                "schema or was produced by code that violated the save "
                "contract."
            )
        warmup_index = pd.DatetimeIndex(
            np.asarray(warmup_index_nanos, dtype="datetime64[ns]"),
            tz=envelope.get("warmup_index_tz"),
            freq=envelope.get("warmup_index_freq"),
        )
        warmup_df = pd.DataFrame(
            np.asarray(warmup_values, dtype=np.float64),
            index=warmup_index,
            columns=list(warmup_columns),
        )

        # Reconstruct ``fit_utc`` from the ISO-8601 string (skops-safe;
        # avoids storing ``datetime`` / ``tzinfo`` directly).
        fit_utc_iso = envelope.get("fit_utc_isoformat")
        fit_utc = datetime.fromisoformat(fit_utc_iso) if fit_utc_iso is not None else None

        # Restore fit-state attributes.  ``loss_history_`` is rebuilt as
        # a fresh list so mutations via the returned instance do not
        # alias back into the envelope dict.
        model._feature_columns = feature_cols
        model._fit_utc = fit_utc
        model._device = torch.device("cpu")
        model._device_resolved = envelope.get("device_resolved")
        model._seed_used = envelope.get("seed_used")
        model._best_epoch = envelope.get("best_epoch")
        model._module = module
        model._warmup_features = warmup_df
        model.loss_history_ = [dict(e) for e in envelope.get("loss_history", [])]

        return model

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
