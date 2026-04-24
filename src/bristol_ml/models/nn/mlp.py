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

Task T1 (this commit) scaffolds the class surface: ``__init__``,
``metadata``, a standalone CLI, and ``NotImplementedError``-raising
stubs for ``fit`` / ``predict`` / ``save`` / ``load``.  Tasks T2 and T3
fill the stubs.

Running standalone::

    python -m bristol_ml.models.nn.mlp --help
    python -m bristol_ml.models.nn.mlp           # prints config schema
    python -m bristol_ml.models.nn               # same (package alias)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from bristol_ml.models.protocol import ModelMetadata
from conf._schemas import NnMlpConfig

if TYPE_CHECKING:  # pragma: no cover — typing-only
    # ``torch.device`` is referenced in a type hint; guarded so the import
    # does not fire for callers that only need the CLI or the config schema.
    import torch

__all__ = ["NnMlpModel"]


# ---------------------------------------------------------------------------
# ``NnMlpModel`` — the public class (scaffold; T2/T3 fill the body).
# ---------------------------------------------------------------------------


class NnMlpModel:
    """Small MLP conforming to the Stage 4 :class:`Model` protocol.

    Stage 10 T1 ships the scaffold: ``__init__`` stores the config,
    ``metadata`` returns an empty-state :class:`ModelMetadata`
    (``fit_utc=None``, ``feature_columns=()``), and the four remaining
    protocol members raise :class:`NotImplementedError` until the
    subsequent tasks land:

    - T2 implements ``fit`` / ``predict`` and the hand-rolled training
      loop (plan D10) with the four-stream reproducibility recipe (D7')
      and the cold-start per-fold contract (D8).
    - T3 implements ``save`` / ``load`` via the single-joblib artefact
      envelope (D5 revised — a dict of ``state_dict`` bytes + config
      dump written via :func:`bristol_ml.models.io.save_joblib` at the
      registry-supplied ``artefact/model.joblib`` file path).

    The class is ``@runtime_checkable`` protocol-compliant from T1 — all
    five named attributes are present — so
    ``test_nn_mlp_is_model_protocol_instance`` passes against the
    scaffold; signature mismatches (impossible here given the straight
    mirror of the protocol) would only be caught by a static type
    checker, per the Stage 4 protocol docstring caveat.

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
        # Populated on fit() in T2.
        self._feature_columns: tuple[str, ...] = ()
        self._fit_utc: datetime | None = None
        self._device: torch.device | None = None
        self._device_resolved: str | None = None
        self._seed_used: int | None = None
        self._best_epoch: int | None = None
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
    ) -> None:
        """Fit the MLP on the aligned ``(features, target)`` pair.

        Stub until Task T2 — see ``docs/plans/active/10-simple-nn.md``
        §6 Task T2 for the full contract (four-stream seed, cudnn
        determinism on CUDA, ``_select_device`` helper, hand-rolled
        training loop with internal validation tail + patience-based
        early stopping + best-epoch restore, per-epoch
        ``loss_history_`` entries, optional ``epoch_callback`` seam).

        Raises
        ------
        NotImplementedError
            Always — T2 fills this in.
        """
        raise NotImplementedError(
            "NnMlpModel.fit is implemented in Stage 10 Task T2 — "
            "the T1 scaffold ships the class surface only."
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return predictions indexed to ``features.index``.

        Stub until Task T2.

        Raises
        ------
        NotImplementedError
            Always — T2 fills this in.
        """
        raise NotImplementedError("NnMlpModel.predict is implemented in Stage 10 Task T2.")

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

        Stub until Task T3.

        Raises
        ------
        NotImplementedError
            Always — T3 fills this in.
        """
        raise NotImplementedError("NnMlpModel.save is implemented in Stage 10 Task T3.")

    @classmethod
    def load(cls, path: Path) -> NnMlpModel:
        """Load a previously-saved :class:`NnMlpModel` from ``path``.

        Plan D5 (revised): reads the joblib envelope, reconstructs
        :class:`NnMlpConfig` from ``config_dump``, instantiates an
        ``NnMlpModel``, materialises the ``state_dict`` via
        ``torch.load(BytesIO(state_dict_bytes), weights_only=True,
        map_location="cpu")``, and calls ``load_state_dict(strict=True)``.
        ``weights_only=True`` keeps PyTorch 2.6+'s safety rail active on
        the inner bytes payload.

        Stub until Task T3.

        Raises
        ------
        NotImplementedError
            Always — T3 fills this in.
        """
        raise NotImplementedError("NnMlpModel.load is implemented in Stage 10 Task T3.")

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
