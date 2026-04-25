"""Shared training-loop helpers for the ``bristol_ml.models.nn`` sub-layer.

Extracted at Stage 11 T1 from :mod:`bristol_ml.models.nn.mlp` per Stage 10
D10 — the extraction seam flagged inside ``NnMlpModel._run_training_loop``.
Both :class:`~bristol_ml.models.nn.mlp.NnMlpModel` (Stage 10) and
:class:`~bristol_ml.models.nn.temporal.NnTemporalModel` (Stage 11) import
and call :func:`run_training_loop` so the four-stream seed recipe, the
best-val tracking, the early-stopping ladder, and the per-epoch
``loss_history`` convention live in exactly one place.  Duplicating the
loop across the two families would invite the divergence hazard the
D10 seam was written to prevent.

Design notes
------------
- ``torch`` is imported lazily inside each function.  The
  ``bristol_ml.models.nn`` package surface is scaffolded to cost zero
  torch import cycles for callers that only want ``--help`` text.  That
  rule applies transitively to every module reachable from the package
  ``__init__``, so ``_training`` follows the same convention.
- :func:`_seed_four_streams` is load-bearing for NFR-1 (plan D7' —
  bit-identity on CPU, numerical closeness on CUDA / MPS).  Moved here
  verbatim from ``mlp.py``; the name is preserved so the regression
  guards in ``tests/unit/models/`` continue to identify the right
  function.
- :func:`run_training_loop` takes two DataLoaders.  Callers build the
  val loader with ``batch_size = len(val_dataset)`` + ``shuffle=False``
  so the single-batch forward-pass semantics of the Stage 10 val path
  are preserved bit-for-bit; Stage 11's TCN uses a batched val loader
  and the mean-over-batches is what it wants anyway.
- The helper does **not** own the scaler buffers, the module factory,
  the optimiser instantiation, or the DataLoader construction.  Callers
  own those so family-specific details (MLP vs TCN channel layout,
  MLP vs Temporal normalisation, optimiser kwargs, DataLoader seeding)
  stay out of the shared path.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

__all__ = ["_seed_four_streams", "run_training_loop"]


def _seed_four_streams(seed: int, device: torch.device) -> None:
    """Seed the four RNG streams that matter at fit time (plan D7').

    Python's :mod:`random`, NumPy, and the two PyTorch streams
    (``torch.manual_seed`` — covers CPU + default CUDA + MPS generators
    — plus :func:`torch.cuda.manual_seed_all` as an explicit
    multi-device hedge).  On CUDA additionally sets
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` — the idiomatic PyTorch
    "as reproducible as it reasonably gets on CUDA" recipe (PyTorch
    reproducibility docs).  Zero cost on CPU / MPS where the flags are
    silently ignored, so setting them unconditionally is safe.

    Intent AC-2 explicitly carves out "within the constraints of
    non-deterministic GPU operations"; NFR-1 therefore guarantees
    bit-identity on CPU only and numerical-closeness on CUDA / MPS.

    Parameters
    ----------
    seed:
        Integer seed.  The caller decides whether this is
        ``config.seed`` (explicit) or
        ``config.project.seed + fold_index`` (cold-start derivation per
        plan D8).
    device:
        The :class:`torch.device` returned by the family-specific
        device-selection helper.  Used only to decide whether to apply
        the cuDNN flags; the ``torch.cuda.manual_seed_all`` call is a
        no-op off-CUDA.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training_loop(
    module: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    loss_history: list[dict[str, float]],
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Run the hand-rolled training loop with best-val weight restore.

    Extracted at Stage 11 T1 from ``NnMlpModel.fit`` per Stage 10 D10.
    Shared between :class:`NnMlpModel` and :class:`NnTemporalModel` so
    neither family carries its own copy of the loop, seed recipe, or
    early-stopping contract.

    Loop contract:

    1. For each epoch ``1..max_epochs``: iterate ``train_loader``,
       forward/backward/step, accumulate per-batch loss values.
    2. After the training pass, evaluate on ``val_loader`` in
       ``module.eval()`` + :func:`torch.no_grad` mode; validation loss
       is the arithmetic mean of per-batch losses.  Callers that want
       single-batch semantics (Stage 10) pass a DataLoader with
       ``batch_size == len(val_dataset)``.
    3. Append ``{"epoch", "train_loss", "val_loss"}`` to
       ``loss_history`` and invoke ``epoch_callback`` with a defensive
       dict-copy so an external live-plot callback cannot mutate the
       canonical history payload.
    4. Track the best validation loss; when it improves beyond
       ``1e-12`` tolerance, snapshot a CPU ``state_dict`` clone as the
       new best-epoch weights.  Stale non-improvements increment a
       patience counter; when the counter hits ``patience``, break.
    5. Return ``(best_state_dict, best_epoch)`` — the caller is
       responsible for calling ``module.load_state_dict(strict=True)``
       on the best weights (family-specific detail: the caller chooses
       whether to restore on the GPU device or on CPU).

    Parameters
    ----------
    module:
        The :class:`torch.nn.Module` being trained.  Must already be on
        ``device``; this helper does not move it.
    train_loader:
        Iterable of ``(x_batch, y_batch)`` pairs already on ``device``.
    val_loader:
        Iterable of ``(x_batch, y_batch)`` pairs already on ``device``.
        Stage 10 passes a single-batch loader (``batch_size == len``);
        Stage 11 may pass a multi-batch loader and accepts the
        mean-over-batches semantics.
    optimiser:
        Pre-constructed :class:`torch.optim.Optimizer`.  The helper does
        not choose Adam / AdamW — that's a family-specific decision.
    criterion:
        Pre-constructed loss module (``nn.MSELoss()`` for both Stage 10
        and Stage 11).
    device:
        Resolved :class:`torch.device`.  Used only for logging context;
        tensors are assumed to already be on this device.
    max_epochs:
        Upper bound on the outer epoch loop.
    patience:
        Number of consecutive non-improving epochs after which the loop
        breaks early.
    loss_history:
        Pre-existing ``list`` passed in by the caller; populated in
        place with one dict per epoch.  The caller aliases this to
        ``model.loss_history_`` so the live-plot seam (plan D6) picks
        up the same list without a copy.
    epoch_callback:
        Optional per-epoch callable.  Invoked with a defensive copy of
        the history entry so an external mutation can't corrupt the
        canonical history.

    Returns
    -------
    tuple[dict[str, torch.Tensor], int]
        ``(best_state_dict, best_epoch)``.  The ``state_dict`` tensors
        are CPU clones (matching the Stage 10 save-path contract that
        envelope tensors ride on CPU regardless of the fit device).
        ``best_epoch`` is the 1-based index of the best-val epoch; if
        no epoch improved on the placeholder ``inf``, it is ``0`` and
        the snapshot is the pre-loop ``state_dict``.
    """
    import torch

    best_val_loss: float = float("inf")
    best_state_dict: dict[str, torch.Tensor] = {
        k: v.detach().clone().cpu() for k, v in module.state_dict().items()
    }
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    for epoch in range(1, max_epochs + 1):
        module.train()
        epoch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            y_hat = module(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        module.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                y_val_hat = module(x_val_batch)
                val_losses.append(float(criterion(y_val_hat, y_val_batch).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        entry = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        loss_history.append(entry)
        if epoch_callback is not None:
            # Defensive copy — we do not trust an external callback not
            # to mutate the dict, which would corrupt ``loss_history``
            # and the live-curve payload.
            epoch_callback(dict(entry))

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "run_training_loop: early stopping at epoch {} "
                    "(best_epoch={} best_val_loss={:.6g} device={}).",
                    epoch,
                    best_epoch,
                    best_val_loss,
                    device,
                )
                break

    return best_state_dict, best_epoch
