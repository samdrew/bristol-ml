"""Small-MLP neural network model — Stage 10.

Implements the Stage 4 :class:`bristol_ml.models.Model` protocol via a
PyTorch ``nn.Module``-backed small MLP.  Stage 10's pedagogical role
(intent §Demo moment) is the live train-vs-validation loss curve in
``notebooks/10-simple-nn.ipynb``; the model's *analytical* contribution
is expected to be modest (intent §Purpose).  The load-bearing pieces
for Stage 11 are:

- the hand-rolled training loop (plan D10; extraction seam flagged for
  Stage 11);
- the four-stream reproducibility recipe (plan D7'; bit-identical on
  CPU, close-match on CUDA / MPS);
- the cold-start-per-fold contract (plan D8);
- the single-joblib artefact layout (plan D5 revised), which plugs into
  the Stage 9 registry's ``artefact/model.joblib`` file-path contract
  without any registry change.

Task T1 scaffolded the class surface.  Task T2 filled ``fit`` /
``predict`` with the hand-rolled training loop, the four-stream seed
helper, the auto-device selector, and the ``_make_mlp`` module-level
module factory.  Task T3 (this commit) fills ``save`` / ``load`` with
the single-joblib-envelope layout (plan D5 revised) that plugs into
the Stage 9 registry's ``artefact/model.joblib`` file-path contract.

Running standalone::

    python -m bristol_ml.models.nn.mlp --help
    python -m bristol_ml.models.nn.mlp           # prints config schema
    python -m bristol_ml.models.nn               # same (package alias)
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import NnMlpConfig

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from torch import nn

__all__ = ["NnMlpModel"]


# ---------------------------------------------------------------------------
# Valid device strings (plan D11) — pinned here so the schema and the
# runtime check cannot drift.
# ---------------------------------------------------------------------------

_ALLOWED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# Module-level helpers (public-ish within the module; private via leading
# underscore per convention).  Module-level so they are pickleable for a
# future refactor and trivially unit-testable.
# ---------------------------------------------------------------------------


def _select_device(preference: str) -> torch.device:
    """Resolve a ``NnMlpConfig.device`` string to a concrete :class:`torch.device`.

    Plan D11 (re-opened at 2026-04-24 Ctrl+G).  Resolution order when
    ``preference == "auto"``:

    1. :func:`torch.cuda.is_available` → ``cuda``;
    2. :func:`torch.backends.mps.is_available` → ``mps``;
    3. ``cpu``.

    Explicit values (``"cpu"`` / ``"cuda"`` / ``"mps"``) are honoured
    verbatim; unknown values raise :class:`ValueError` rather than
    falling back to CPU — a silent downgrade would hide configuration
    mistakes.  The resolved device is logged at INFO so a live-demo
    facilitator can see which device the fit actually landed on.

    Parameters
    ----------
    preference:
        One of :data:`_ALLOWED_DEVICES`.

    Returns
    -------
    :class:`torch.device`
        The concrete device the caller should move the module and
        tensors to.

    Raises
    ------
    ValueError
        If ``preference`` is not one of :data:`_ALLOWED_DEVICES`.
    """
    import torch  # local import — keep the module import graph cheap

    if preference not in _ALLOWED_DEVICES:
        raise ValueError(
            f"NnMlpConfig.device must be one of {_ALLOWED_DEVICES!r}; got {preference!r}."
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

    logger.info("NnMlpModel: device preference={!r} resolved to {!r}.", preference, str(resolved))
    return resolved


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
        The :class:`torch.device` returned by :func:`_select_device`.
        Used only to decide whether to apply the cuDNN flags; the
        ``torch.cuda.manual_seed_all`` call is a no-op off-CUDA.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_mlp(
    input_dim: int,
    config: NnMlpConfig,
) -> nn.Module:
    """Build a small MLP :class:`torch.nn.Module` from ``config``.

    The module embeds the z-score scaler buffers (``feature_mean``,
    ``feature_std``, ``target_mean``, ``target_std``) via
    :meth:`torch.nn.Module.register_buffer` so they round-trip cleanly
    through ``state_dict()`` (plan D4).  Buffers are initialised to
    placeholder zeros / ones at module construction and then overwritten
    by :meth:`NnMlpModel.fit` from the training slice's column statistics.

    The forward pass is: ``(x - feature_mean) / feature_std → Linear →
    activation → Dropout → … → Linear(1) → y_norm``; the caller is
    responsible for the inverse target transform
    (``y = y_norm * target_std + target_mean``) because the scaled target
    is what the loss is computed against.

    Parameters
    ----------
    input_dim:
        Number of input features — determines the first ``nn.Linear``
        input dimension.  Load-bearing: a mis-match between the
        training-time ``input_dim`` and a load-time reconstruction
        would silently produce wrong predictions, so the Stage 10 T3
        save/load path stores the fitted ``feature_columns`` tuple and
        reconstructs ``input_dim = len(feature_columns)``.
    config:
        :class:`NnMlpConfig`; ``hidden_sizes``, ``activation``, and
        ``dropout`` shape the module.

    Returns
    -------
    :class:`torch.nn.Module`
        A concrete module (a :class:`_NnMlpModule` instance — a private
        subclass of :class:`torch.nn.Module`).  The class is module-level
        for pickleability.
    """
    return _NnMlpModule(
        input_dim=input_dim,
        hidden_sizes=tuple(config.hidden_sizes),
        activation=config.activation,
        dropout=config.dropout,
    )


# ``_NnMlpModule`` is a module-level :class:`torch.nn.Module` subclass so
# (a) it is pickleable — a closure or local class could not survive the
# save/load round-trip (AC-4) — and (b) unit tests can construct it
# directly without going through the full ``NnMlpModel.fit`` path.  Kept
# private via the leading underscore; ``NnMlpModel`` owns the public
# surface.  The class itself is built lazily by
# :func:`_build_nn_module_class` (below) so that importing this module does
# not force an eager ``import torch`` — the CLI scaffold path and the T1
# protocol test never touch torch.

_nn_mlp_module_cls: type[nn.Module] | None = None


def _build_nn_module_class() -> type[nn.Module]:
    """Return the :class:`torch.nn.Module` subclass used by the MLP.

    Cached in :data:`_nn_mlp_module_cls` after first construction so
    repeated ``fit`` calls don't re-build the class object.  The class
    is defined inside a function because ``torch`` is imported lazily
    (to keep the CLI / scaffold path cheap); the defined class is
    pickleable because its ``__module__`` / ``__qualname__`` are
    explicitly set to the import path ``bristol_ml.models.nn.mlp``.
    """
    global _nn_mlp_module_cls
    if _nn_mlp_module_cls is not None:
        return _nn_mlp_module_cls

    import torch
    from torch import nn

    _ACTIVATION_FNS: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }

    class _NnMlpModuleImpl(nn.Module):
        """Feed-forward MLP with z-score input + target buffers (plan D4)."""

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_sizes: tuple[int, ...],
            activation: str,
            dropout: float,
        ) -> None:
            super().__init__()
            if activation not in _ACTIVATION_FNS:
                raise ValueError(
                    f"Unsupported activation {activation!r}; expected one of "
                    f"{tuple(_ACTIVATION_FNS)}."
                )

            self.input_dim = input_dim
            self.hidden_sizes = hidden_sizes
            self.activation_name = activation
            self.dropout_p = dropout

            # Scaler buffers — registered with deterministic placeholder
            # values (zeros / ones) so the module has the exact same
            # ``state_dict`` keys regardless of whether fit() has been
            # called yet.  ``load_state_dict(strict=True)`` in T3 relies
            # on this.
            self.register_buffer("feature_mean", torch.zeros(input_dim))
            self.register_buffer("feature_std", torch.ones(input_dim))
            self.register_buffer("target_mean", torch.zeros(1))
            self.register_buffer("target_std", torch.ones(1))

            layers: list[nn.Module] = []
            prev = input_dim
            activation_cls = _ACTIVATION_FNS[activation]
            for width in hidden_sizes:
                layers.append(nn.Linear(prev, width))
                layers.append(activation_cls())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                prev = width
            layers.append(nn.Linear(prev, 1))
            self.backbone: nn.Sequential = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Normalise → backbone → squeeze last dim to ``(N,)`` predictions."""
            # ``feature_std`` is fitted > 0 (guarded in ``NnMlpModel.fit``),
            # so division is safe without an epsilon.  The pre-fit
            # placeholder is 1.0 which is also safe.
            z = (x - self.feature_mean) / self.feature_std
            y = self.backbone(z).squeeze(-1)
            return y

    _NnMlpModuleImpl.__module__ = "bristol_ml.models.nn.mlp"
    _NnMlpModuleImpl.__qualname__ = "_NnMlpModuleImpl"
    _nn_mlp_module_cls = _NnMlpModuleImpl
    return _NnMlpModuleImpl


def _NnMlpModule(**kwargs: Any) -> nn.Module:
    """Construct a torch-backed MLP module via the lazy factory.

    Looks identical to instantiating a class directly — callers in
    :func:`_make_mlp` and in tests write ``_NnMlpModule(input_dim=..., ...)``
    — but routes through :func:`_build_nn_module_class` so the
    :class:`torch.nn.Module` subclass is only defined after ``torch`` has
    been imported.  Keeps the top-of-module import cost at zero torch
    cycles for scaffold / CLI callers.
    """
    cls = _build_nn_module_class()
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# ``NnMlpModel`` — the public class
# ---------------------------------------------------------------------------


class NnMlpModel:
    """Small MLP conforming to the Stage 4 :class:`Model` protocol.

    Stage 10 T2 fills ``fit`` / ``predict`` on top of the T1 scaffold.
    T3 will fill ``save`` / ``load``.

    See ``docs/architecture/layers/models-nn.md`` for the full contract.
    """

    def __init__(self, config: NnMlpConfig) -> None:
        """Store ``config`` and initialise empty fit-state.

        Parameters
        ----------
        config:
            Validated :class:`~conf._schemas.NnMlpConfig`.  Pydantic's
            ``frozen=True`` makes the reference shareable without
            defensive copy.
        """
        self._config: NnMlpConfig = config
        # Populated on fit().
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None
        self._device: torch.device | None = None
        self._device_resolved: str | None = None
        self._seed_used: int | None = None
        self._best_epoch: int | None = None
        # The fitted :class:`torch.nn.Module` — populated in ``fit``.
        self._module: nn.Module | None = None
        #: One dict per epoch after ``fit``; keys
        #: ``{"epoch", "train_loss", "val_loss"}`` (plan D6 / AC-3).
        #: Public attribute (trailing underscore mirrors sklearn fitted-state
        #: convention) so the notebook's live-plot callback and
        #: :func:`bristol_ml.evaluation.plots.loss_curve` can consume it
        #: without reaching into private state.
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
        """Fit the MLP on the aligned ``(features, target)`` pair.

        Pipeline:

        1. Shape / length guard.
        2. Feature-column resolution (``config.feature_columns`` wins
           when set; otherwise the full input-frame column order is
           used, matching the Linear / SARIMAX convention).
        3. Device selection via :func:`_select_device` and seeding via
           :func:`_seed_four_streams` (four-stream recipe, plan D7').
        4. Train / validation split — an **internal 10 % tail** of the
           training slice (plan D9).  The tail-split is deterministic
           and time-respecting; no random shuffle.
        5. Scaler buffers (``feature_mean``, ``feature_std``,
           ``target_mean``, ``target_std``) are fitted from the train
           portion **only**.  A zero ``feature_std`` column is clamped
           to ``1.0`` so division is safe; the log records the
           zero-variance columns.
        6. Module construction via :func:`_make_mlp` and movement to
           the resolved device.
        7. Hand-rolled training loop (plan D10) — :class:`torch.optim.Adam`,
           MSE loss on **normalised** target values, ``config.batch_size``
           minibatches, DataLoader seeded via a :class:`torch.Generator`,
           ``num_workers=0`` (plan D7').
        8. Patience-based early stopping with best-epoch weight restore
           (plan D9); ``loss_history_`` appended per epoch; optional
           ``epoch_callback`` invoked per epoch.
        9. Re-entrancy: any pre-existing ``loss_history_`` /
           ``_best_epoch`` / ``_module`` are discarded at the top of
           ``fit`` — the cold-start-per-fold contract (plan D8).

        Parameters
        ----------
        features:
            Feature frame; ``features.index`` is not constrained here
            (the harness passes a UTC-aware ``DatetimeIndex`` but
            nothing in the MLP cares about time alignment).
        target:
            Aligned target series; ``len(target) == len(features)``.
        seed:
            Optional seed override.  ``None`` uses ``config.seed``
            (which may itself be ``None`` — in that case the per-fold
            determinism is the caller's responsibility via
            ``config.project.seed + fold_index``).  An integer here
            pins the seed and overrides the config — used by the
            T1-label reproducibility test
            ``test_nn_mlp_seeded_runs_produce_identical_state_dicts``.
        epoch_callback:
            Optional callable invoked after each epoch with the dict
            ``{"epoch", "train_loss", "val_loss"}`` — the live-plot
            seam for the notebook's Demo moment (plan D6 / X4).
        """
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset

        # --- 1. Guards --------------------------------------------------
        if len(features) != len(target):
            raise ValueError(
                "NnMlpModel.fit requires len(features) == len(target); "
                f"got {len(features)} vs {len(target)}."
            )
        if len(features) == 0:
            raise ValueError("NnMlpModel.fit requires at least one training row.")

        cfg = self._config

        # --- 2. Feature-column resolution -------------------------------
        if cfg.feature_columns is not None:
            feature_cols: tuple[str, ...] = tuple(cfg.feature_columns)
            missing = [c for c in feature_cols if c not in features.columns]
            if missing:
                raise ValueError(
                    f"NnMlpModel.fit: configured feature_columns not present in features: "
                    f"{missing!r}.  Available: {list(features.columns)!r}."
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
            # call is still reproducible.  The harness path passes the
            # per-fold seed explicitly, so this branch matters for
            # notebook experimentation rather than CI.
            effective_seed = 0
        _seed_four_streams(int(effective_seed), device)

        # --- 4. Train / val split (plan D9) ----------------------------
        n = len(X_df)
        n_val = max(1, n // 10)
        n_train = n - n_val
        if n_train < 1:
            raise ValueError(
                f"NnMlpModel.fit needs at least 2 rows to split off a val tail; got {n}."
            )

        X_arr = np.asarray(X_df.to_numpy(), dtype=np.float32)
        y_arr = np.asarray(target.to_numpy(), dtype=np.float32)

        X_train_np = X_arr[:n_train]
        y_train_np = y_arr[:n_train]
        X_val_np = X_arr[n_train:]
        y_val_np = y_arr[n_train:]

        # --- 5. Scaler buffers (plan D4) -------------------------------
        feat_mean = X_train_np.mean(axis=0).astype(np.float32)
        feat_std = X_train_np.std(axis=0).astype(np.float32)
        zero_var_mask = feat_std < 1e-12
        if zero_var_mask.any():
            zero_cols = [feature_cols[i] for i in np.where(zero_var_mask)[0]]
            logger.info(
                "NnMlpModel.fit: zero-variance feature columns clamped to std=1: {}.",
                zero_cols,
            )
            feat_std = np.where(zero_var_mask, 1.0, feat_std).astype(np.float32)
        tgt_mean = float(y_train_np.mean())
        tgt_std = float(y_train_np.std())
        if tgt_std < 1e-12:
            logger.warning("NnMlpModel.fit: target std is ~0 on the train slice; clamping to 1.")
            tgt_std = 1.0

        # --- 6. Module construction ------------------------------------
        module = _make_mlp(input_dim=len(feature_cols), config=cfg)
        # Fit scaler buffers from the train slice.  ``register_buffer``
        # reserved the tensors in ``__init__``; we now copy real values
        # in so they ride inside ``state_dict()``.
        with torch.no_grad():
            module.feature_mean.copy_(torch.from_numpy(feat_mean))  # type: ignore[union-attr]
            module.feature_std.copy_(torch.from_numpy(feat_std))  # type: ignore[union-attr]
            module.target_mean.copy_(torch.tensor([tgt_mean], dtype=torch.float32))  # type: ignore[union-attr]
            module.target_std.copy_(torch.tensor([tgt_std], dtype=torch.float32))  # type: ignore[union-attr]
        module = module.to(device)

        # --- 7. Training loop (plan D10) -------------------------------
        X_train_t = torch.from_numpy(X_train_np).to(device)
        y_train_t = torch.from_numpy(y_train_np).to(device)
        X_val_t = torch.from_numpy(X_val_np).to(device)
        y_val_t = torch.from_numpy(y_val_np).to(device)

        # Work on normalised target values so the loss scale is O(1).
        y_train_norm = (y_train_t - tgt_mean) / tgt_std
        y_val_norm = (y_val_t - tgt_mean) / tgt_std

        # Seeded :class:`torch.Generator` for the DataLoader (plan D7').
        dl_generator = torch.Generator(device="cpu")
        dl_generator.manual_seed(int(effective_seed))
        train_dataset = TensorDataset(X_train_t, y_train_norm)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            generator=dl_generator,
            drop_last=False,
        )

        optimizer = optim.Adam(
            module.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()

        # Best-epoch tracking (plan D9).
        best_val_loss: float = float("inf")
        best_state_dict: dict[str, torch.Tensor] = {
            k: v.detach().clone().cpu() for k, v in module.state_dict().items()
        }
        best_epoch: int = 0
        epochs_without_improvement: int = 0

        # Reset re-entrant state before appending per-epoch entries (D8).
        loss_history: list[dict[str, float]] = []

        for epoch in range(1, cfg.max_epochs + 1):
            module.train()
            epoch_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_hat = module(x_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

            module.eval()
            with torch.no_grad():
                y_val_hat = module(X_val_t)
                val_loss = float(loss_fn(y_val_hat, y_val_norm).detach().cpu().item())

            entry = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            loss_history.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)

            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.detach().clone().cpu() for k, v in module.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    logger.info(
                        "NnMlpModel.fit: early stopping at epoch {} "
                        "(best_epoch={} best_val_loss={:.6g}).",
                        epoch,
                        best_epoch,
                        best_val_loss,
                    )
                    break

        # Restore best-epoch weights (plan D9).
        module.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()}, strict=True)

        # --- 8. Publish state -----------------------------------------
        self._feature_columns = feature_cols
        self._fit_utc = datetime.now(UTC)
        self._device = device
        self._device_resolved = str(device)
        self._seed_used = int(effective_seed)
        self._best_epoch = best_epoch
        self._module = module
        self.loss_history_ = loss_history

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return predictions indexed to ``features.index``.

        ``features`` must carry every column the model was fit on; extra
        columns are silently ignored (mirrors the Linear / SARIMAX
        contract).  Predict-before-fit raises :class:`RuntimeError`
        rather than returning stale or zero output (protocol convention
        — see ``models/protocol.py`` docstring).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If any of ``self._feature_columns`` is missing from
            ``features``.
        """
        import torch

        if self._module is None or self._device is None:
            raise RuntimeError("NnMlpModel must be fit() before predict().")

        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"NnMlpModel.predict: features frame is missing fitted columns {missing!r}. "
                f"Available: {list(features.columns)!r}."
            )

        X_df = features[list(self._feature_columns)]
        X_arr = np.asarray(X_df.to_numpy(), dtype=np.float32)
        X_t = torch.from_numpy(X_arr).to(self._device)

        self._module.eval()
        with torch.no_grad():
            y_norm = self._module(X_t)
            # Inverse target normalisation; scalers live on the module.
            y = y_norm * self._module.target_std + self._module.target_mean  # type: ignore[union-attr]
        y_np = y.detach().cpu().numpy().astype(np.float64)
        return pd.Series(
            y_np,
            index=features.index,
            name=self._config.target_column,
        )

    def save(self, path: Path) -> None:
        """Serialise the fitted model to the registry-supplied file path.

        Plan D5 (revised): ``path`` is the single joblib-artefact file
        path (the Stage 9 registry hard-codes ``artefact/model.joblib``
        — see ``src/bristol_ml/registry/_fs.py::_atomic_write_run``).
        The envelope written is a plain dict carrying the ``state_dict``
        bytes (via ``torch.save`` to a ``BytesIO``), the
        ``NnMlpConfig.model_dump()``, the resolved feature-column tuple,
        and provenance scalars (``seed_used``, ``best_epoch``,
        ``loss_history``, ``fit_utc``, ``device_resolved``).

        The write is atomic — :func:`bristol_ml.models.io.save_joblib`
        stages to a sibling ``.tmp`` and renames via :func:`os.replace`,
        matching the Stage 4 contract.  No sibling ``model.pt`` or
        ``hyperparameters.json`` is created — plan D5 revision (single
        envelope) is structurally enforced by the T3 "single file" test.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called — save-before-fit is
            undefined per the Stage 4 :class:`Model` protocol.
        """
        import io as _io

        import torch

        from bristol_ml.models.io import save_joblib

        if self._module is None:
            raise RuntimeError(
                "NnMlpModel.save requires fit() to have been called first; "
                "state_dict is undefined before fit."
            )

        # state_dict → bytes.  ``.cpu()`` on every tensor so the envelope
        # is portable across devices; ``torch.load(..., map_location="cpu")``
        # in :meth:`load` mirrors this.
        cpu_state_dict = {k: v.detach().cpu() for k, v in self._module.state_dict().items()}
        buf = _io.BytesIO()
        torch.save(cpu_state_dict, buf)
        state_dict_bytes: bytes = buf.getvalue()

        envelope: dict[str, Any] = {
            "state_dict_bytes": state_dict_bytes,
            "config_dump": self._config.model_dump(),
            "feature_columns": tuple(self._feature_columns),
            "seed_used": self._seed_used,
            "best_epoch": self._best_epoch,
            "loss_history": list(self.loss_history_),
            "fit_utc": self._fit_utc,
            "device_resolved": self._device_resolved,
        }
        save_joblib(envelope, path)

    @classmethod
    def load(cls, path: Path) -> NnMlpModel:
        """Load a previously-saved :class:`NnMlpModel` from ``path``.

        Plan D5 (revised): reads the joblib envelope, reconstructs
        :class:`NnMlpConfig` from ``config_dump`` (Pydantic re-validation
        catches schema drift), instantiates an ``NnMlpModel``, materialises
        the ``state_dict`` via
        ``torch.load(BytesIO(state_dict_bytes), weights_only=True,
        map_location="cpu")``, and calls ``load_state_dict(strict=True)``.
        ``weights_only=True`` keeps PyTorch 2.6+'s safety rail active on
        the inner bytes payload; ``strict=True`` is load-bearing for
        plan R3 (a dropped buffer would silently break inverse-normalisation
        at :meth:`predict` otherwise).

        The loaded model lives on CPU regardless of the device it was
        fitted on — :meth:`predict` moves tensors to ``self._device`` at
        call time and the buffers ride along with the module.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist (propagated from :mod:`joblib`).
        """
        import io as _io

        import torch

        from bristol_ml.models.io import load_joblib

        envelope = load_joblib(path)

        # Pydantic re-validation is the schema-drift guard (plan R2).
        config = NnMlpConfig.model_validate(envelope["config_dump"])
        model = cls(config)

        # Rebuild the nn.Module with the same input_dim the fit used.
        feature_cols: tuple[str, ...] = tuple(envelope["feature_columns"])
        if not feature_cols:
            raise ValueError(
                "NnMlpModel.load: envelope carries empty feature_columns; "
                "this envelope was not produced by a fitted NnMlpModel."
            )
        module = _make_mlp(input_dim=len(feature_cols), config=config)

        # Materialise the saved state_dict on CPU and load strictly — a
        # missing or extra key (e.g. a dropped scaler buffer) fails loudly
        # here rather than silently skewing predictions later.
        buf = _io.BytesIO(envelope["state_dict_bytes"])
        state_dict = torch.load(buf, weights_only=True, map_location="cpu")
        module.load_state_dict(state_dict, strict=True)

        # Restore fit-state attributes.  ``loss_history_`` is rebuilt as a
        # fresh list so mutations via the returned instance do not alias
        # back into the envelope dict.
        model._feature_columns = feature_cols
        model._fit_utc = envelope.get("fit_utc")
        model._device = torch.device("cpu")
        model._device_resolved = envelope.get("device_resolved")
        model._seed_used = envelope.get("seed_used")
        model._best_epoch = envelope.get("best_epoch")
        model._module = module
        model.loss_history_ = [dict(e) for e in envelope.get("loss_history", [])]

        return model

    @property
    def metadata(self) -> ModelMetadata:
        """Immutable provenance record for the most recent fit.

        Before :meth:`fit` has been called, ``fit_utc`` is ``None`` and
        ``feature_columns`` is empty — matching the Stage 4 protocol
        convention.  Once fitted, ``hyperparameters`` carries the
        architecture summary plus the resolved device (plan D11) and
        the seed actually used (plan D7'/D8) so a downstream registry
        sidecar reader can reconstruct the fit conditions without
        reaching into private attributes.
        """
        cfg = self._config
        hyperparameters: dict[str, Any] = {
            "target_column": cfg.target_column,
            "hidden_sizes": list(cfg.hidden_sizes),
            "activation": cfg.activation,
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            # ``device`` (config-requested) versus ``device_resolved`` (the
            # actual device ``fit`` landed on after auto-select) are
            # intentionally both recorded — the former is reproducibility
            # provenance, the latter is what the weights were trained on.
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
# Private helpers (module-level for testability and for state_dict pickle
# safety — a closure inside ``fit`` would not survive a future refactor
# that serialises a submodule containing a Lambda layer).
# ---------------------------------------------------------------------------


def _build_metadata_name(config: NnMlpConfig) -> str:
    """Build a metadata ``name`` matching ``ModelMetadata.name``'s regex.

    ``ModelMetadata.name`` is constrained to ``^[a-z][a-z0-9_.-]*$``.  The
    format is ``nn-mlp-{activation}-{hidden_widths_joined_by_dashes}`` — e.g.
    ``nn-mlp-relu-128``, ``nn-mlp-gelu-128-64``.  Encodes enough of the
    architecture to be greppable in a registry leaderboard without
    serialising every hyperparameter.
    """
    widths = "-".join(str(w) for w in config.hidden_sizes)
    return f"nn-mlp-{config.activation}-{widths}"


# ---------------------------------------------------------------------------
# CLI — ``python -m bristol_ml.models.nn.mlp``
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.models.nn.mlp",
        description=(
            "Print the NnMlpConfig JSON schema and the resolved config. "
            "Training and prediction are exercised via the evaluation "
            "harness (see `python -m bristol_ml.train model=nn_mlp`)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model.hidden_sizes=[128,64] model.device=cpu",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    """Standalone CLI entry point — DESIGN §2.1.1 / plan NFR-6.

    Prints the :class:`~conf._schemas.NnMlpConfig` JSON schema followed
    by the resolved config summary.  Returns ``0`` on success, ``2`` if
    a non-nn_mlp model resolves from config (e.g. if the caller forgot
    ``model=nn_mlp``).
    """
    import json

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from bristol_ml.config import load_config

    print("NnMlpConfig JSON schema:")
    print(json.dumps(NnMlpConfig.model_json_schema(), indent=2))
    print()
    print(
        "To fit: python -m bristol_ml.train model=nn_mlp "
        "evaluation.rolling_origin.fixed_window=true "
        "evaluation.rolling_origin.step=168"
    )

    cfg = load_config(overrides=["model=nn_mlp", *list(args.overrides)])
    if cfg.model is None or not isinstance(cfg.model, NnMlpConfig):
        print(
            "No NnMlpConfig resolved. Ensure `model=nn_mlp` or a matching "
            "override is present; got "
            f"{type(cfg.model).__name__ if cfg.model is not None else 'None'}.",
            file=sys.stderr,
        )
        return 2

    nn_cfg = cfg.model
    logger.info(
        "NnMlpConfig: hidden_sizes={} activation={} dropout={} "
        "learning_rate={} batch_size={} max_epochs={} patience={} "
        "device={} target_column={}",
        nn_cfg.hidden_sizes,
        nn_cfg.activation,
        nn_cfg.dropout,
        nn_cfg.learning_rate,
        nn_cfg.batch_size,
        nn_cfg.max_epochs,
        nn_cfg.patience,
        nn_cfg.device,
        nn_cfg.target_column,
    )
    print()
    print(f"hidden_sizes={list(nn_cfg.hidden_sizes)}")
    print(f"activation={nn_cfg.activation}")
    print(f"dropout={nn_cfg.dropout}")
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
