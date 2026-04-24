"""Temporal convolutional network model — Stage 11 (T2 scaffolding).

This commit (T2) lands the sequence-data pipeline — :class:`_SequenceDataset`
— plus the :class:`~conf._schemas.NnTemporalConfig` Pydantic schema and the
matching Hydra YAML group.  The actual :class:`NnTemporalModel` class
(`fit`, `predict`, `save`, `load`, `metadata`) arrives in T3-T5.

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

Running standalone (NFR-5; `fit` path not yet implemented — T3)::

    python -m bristol_ml.models.nn.temporal --help    # T3
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from torch.utils.data import Dataset


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
