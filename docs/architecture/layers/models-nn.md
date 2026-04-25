# Models — neural-network sub-layer

- **Status:** Stable for two families — first realised by Stage 10
  (`NnMlpModel` — simple feed-forward MLP; shipped) and extended at
  Stage 11 (`NnTemporalModel` — TCN; shipped).  The Stage 10 D10
  extraction seam fired at Stage 11 T1, so the hand-rolled training
  loop and four-stream seed recipe now live at
  `bristol_ml.models.nn._training` and are shared by both classes.
  Revisit when a third torch-backed family arrives (Stage 17 candidate
  for a competitive-NN tuning surface, or earlier if a Transformer
  variant graduates from the Stage 11 D1 deferral).
- **Parent layer doc:** [`docs/architecture/layers/models.md`](models.md)
  (the `Model` protocol + `ModelMetadata` + joblib IO helpers that every
  family conforms to).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities)
  (models paragraph); [`DESIGN.md` §8](../../intent/DESIGN.md#8-technology-choices)
  (PyTorch as the chosen NN framework).
- **Concrete instances:** [Stage 10 retro](../../lld/stages/10-simple-nn.md)
  (`NnMlpModel`, `_select_device`, `_seed_four_streams`, `_make_mlp`,
  `_NnMlpModule`).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow
  interfaces), §2.1.4 (config outside code), §2.1.5 (idempotence — cold
  start per fold), §2.1.6 (provenance — device, seed, best-epoch), §2.1.7
  (tests at boundaries).
- **Key plan:** [`docs/plans/completed/10-simple-nn.md`](../../plans/completed/10-simple-nn.md)
  — decisions D1 (dependency), D3
  (architecture defaults), D4 (normalisation), D5 (artefact envelope),
  D6 (loss-curve surfacing), D7' (reproducibility), D8 (cold start),
  D9 (early stopping), D10 (training-loop ownership), D11 (device),
  D12 (this doc).

---

## Why this sub-layer exists

The models layer is deliberately shaped around one protocol and many
concrete classes (`NaiveModel`, `LinearModel`, `SarimaxModel`,
`ScipyParametricModel`, `NnMlpModel`).  The first four families share a
common lineage — they are numpy / pandas / statsmodels / scipy wrappers
with a joblib-native serialisation path.  `NnMlpModel` is the first
family that brings **two distinct infrastructural concerns** on top of
the protocol:

1. **A hand-rolled PyTorch training loop** with its own epoch
   bookkeeping, early stopping, and loss-history surfacing (plan D10 /
   AC-3).
2. **A two-stage serialisation path** — `state_dict` → bytes → joblib
   envelope — that keeps the Stage 9 registry's single-file artefact
   contract (plan D5 revised / AC-4).

Stage 11 has now added the second class — `NnTemporalModel` (TCN) —
that shares exactly these two concerns.  Rather than rewrite the
training-loop idiom in every subsequent NN stage, this sub-layer
names the conventions once.  The Stage 10 D10 extraction seam fired
at Stage 11 T1: the training-loop body and the four-stream seed
recipe moved from `NnMlpModel._run_training_loop` to
`src/bristol_ml/models/nn/_training.py::run_training_loop`, and both
classes import and call it.  The seam continues to point forward —
the next swap (gradient clipping, LR scheduling, mixed precision)
extends `_training.py` rather than re-forking the loop into each
family.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| PyTorch-backed `Model` protocol conformers | ✓ | — |
| Hand-rolled training loop (per-epoch accounting, val tail, early stopping) | ✓ | evaluation harness (fold-level loop) |
| Four-stream reproducibility recipe (`_seed_four_streams`) | ✓ | — |
| Auto-device selection (`_select_device`: CUDA > MPS > CPU) | ✓ | — |
| Z-score scaler buffers persisted in `state_dict` via `register_buffer` | ✓ | sklearn `StandardScaler` sibling file |
| Single-joblib artefact envelope (`state_dict_bytes` + `config_dump` + scalars) | ✓ | two-file (`model.pt` + `hyperparameters.json`) layout |
| Loss-history surfacing (`loss_history_: list[dict]` + `epoch_callback` seam) | ✓ | live-plot rendering (notebook owns that) |
| Gradient clipping, LR scheduling as configurable knobs | — | Stage 11 deferred to a future stage; `run_training_loop` does not currently expose these knobs (Stage 10 X6 stays cut) |
| `BaseTorchModel` abstract base class | — | Stage 11 D4 extracted a *function*, not a class hierarchy — Stage 10 X7 cut re-affirmed at Stage 11 |
| Sequence-data adapter (`_SequenceDataset` lazy windowing) | ✓ | inside `temporal.py` — private to the temporal model; Stage 10 has no use for it |
| Causal-padding recipe (`F.pad(x, (left, 0))` + `Conv1d(padding=0)`) | ✓ | TCN-specific; documented in §"Stage 11 addition" below |
| Hyperparameter search (random / Optuna / Ray Tune) | — | out of scope at Stage 10; undesigned |
| Distributed / multi-GPU training (DDP / FSDP / model sharding) | — | explicitly out of scope per intent §Out of scope |
| Automatic loss-curve PNG saved to the registry run dir | — | scope-diff cut (NFR-4): AC-3 is satisfied by `loss_history_` + `plots.loss_curve`, no registry coupling |

The split is deliberate: every concern kept in the sub-layer above has a
direct tie to an AC (conformance, reproducibility, loss curve, registry
round-trip) or to the Stage 11 extraction seam.  Concerns below the
line were considered and explicitly deferred — either to Stage 11's
shared helper (the Stage-10 extraction seam) or to a later stage that
owns the primitive (Stage 12 for serving, Stage 18 for drift).

## Cross-module conventions

### 1. Module shape

```
src/bristol_ml/models/nn/
├── __init__.py          # exports: NnMlpModel, NnTemporalModel (lazy __getattr__)
├── __main__.py          # `python -m bristol_ml.models.nn` → delegates to mlp.py
├── mlp.py               # NnMlpModel + _NnMlpModule + module-level helpers
├── temporal.py          # NnTemporalModel + _NnTemporalModuleImpl + _TemporalBlockImpl + _SequenceDataset
├── _training.py         # run_training_loop + _seed_four_streams (shared by both families)
└── CLAUDE.md            # module guide (this file's sibling)
```

`_training.py` was the Stage 10 D10 extraction destination, fired at
Stage 11 T1.  The next swap on the loop body (gradient clipping, LR
scheduling, mixed precision) extends this module rather than
re-forking the loop into each family.

### 2. Public interface

```python
# src/bristol_ml/models/nn/mlp.py

class NnMlpModel:
    """Small MLP conforming to the Stage 4 ``Model`` protocol."""

    def __init__(self, config: NnMlpConfig) -> None: ...

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        seed: int | None = None,
        epoch_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None: ...

    def predict(self, features: pd.DataFrame) -> pd.Series: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> NnMlpModel: ...

    @property
    def metadata(self) -> ModelMetadata: ...

    # Stage-10-specific public attribute (populated after fit)
    loss_history_: list[dict[str, float]]
```

- `fit` shares the Stage 4 signature plus two Stage-10 kwargs: `seed`
  overrides `config.seed` (the cold-start-per-fold hook at plan D8),
  and `epoch_callback` is the live-plot seam (plan D6 / AC-3).
  `epoch_callback(entry)` is invoked after every training epoch with
  `entry = {"epoch": int, "train_loss": float, "val_loss": float}`;
  the models layer never imports `IPython` / `matplotlib` — the
  notebook owns the live-plot rendering.
- `predict` returns a `pd.Series` indexed to `features.index`.  Before
  `fit`, it raises `RuntimeError` (protocol convention).
- `save(path)` writes the single-joblib envelope (see §4 below); the
  path is the **file** path `.../artefact/model.joblib` the Stage 9
  registry passes down, not a directory.  Plan D5 revised.
- `load(path)` is a `classmethod`.  Reconstructs the config from the
  envelope's `config_dump`, rebuilds the `nn.Module` skeleton, and
  applies `load_state_dict(strict=True)` so a missing buffer or an
  extra key fails loudly (plan R3).
- `metadata` is a property.  Before `fit`, `metadata.fit_utc is None`.
  After `fit`, `metadata.hyperparameters` carries `best_epoch`,
  `device_resolved`, `seed_used`, and the full `NnMlpConfig` dump.

### 3. Reproducibility (plan D7')

The four-stream seed recipe lives in `_seed_four_streams(seed, device)`:

1. `random.seed(seed)` — Python's stdlib RNG.
2. `numpy.random.seed(seed)` — legacy NumPy global RNG (the downstream
   code that still reaches for `np.random.*` rather than
   `numpy.random.default_rng` is covered).
3. `torch.manual_seed(seed)` — covers CPU + the default CUDA / MPS
   generators.
4. `torch.cuda.manual_seed_all(seed)` — explicit multi-CUDA-device
   hedge (no-op on non-CUDA hosts).

On CUDA, the helper additionally sets
`torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` — the idiomatic PyTorch "as
reproducible as it reasonably gets on CUDA" recipe from the PyTorch
reproducibility docs.  It does **not** set
`torch.use_deterministic_algorithms(True)`: intent AC-2 explicitly
carves out "within the constraints of non-deterministic GPU
operations", so the stricter guarantee would cost throughput for a
contract the spec does not demand (plan H-5 / OQ-A resolved `NO`).

The device-aware NFR-1 contract is therefore split:

- **CPU:** two `fit(seed=0)` runs on the same data produce identical
  `state_dict` tensors (bit-identity, `torch.equal`).
- **CUDA / MPS:** two `fit(seed=0)` runs produce `predict` outputs
  matching under `torch.allclose(atol=1e-5, rtol=1e-4)`.

The CPU guarantee is covered by `test_nn_mlp_seeded_runs_produce_
identical_state_dicts`; the CUDA close-match guarantee is covered by
`test_nn_mlp_seeded_runs_match_on_cuda_within_tolerance`, which is
`@pytest.mark.gpu` and skipped when `torch.cuda.is_available()` is
false.

### 4. Serialisation (plan D5 revised)

The Stage 9 registry passes the artefact **file path**
`artefact/model.joblib` to `Model.save(path)` — not a directory.  The
path is produced by `bristol_ml.registry._fs._atomic_write_run` and the
filename is hard-coded there.  `NnMlpModel.save(path)` therefore writes
a single joblib artefact at exactly that path, with contents:

```python
{
    "state_dict_bytes": bytes,        # torch.save(state_dict, BytesIO).getvalue()
    "config_dump": dict[str, Any],    # NnMlpConfig.model_dump()
    "feature_columns": tuple[str, ...],
    "seed_used": int,
    "best_epoch": int,
    "loss_history": list[dict[str, float]],
    "fit_utc": str,                   # ISO-8601, UTC
    "device_resolved": str,           # "cpu" | "cuda" | "mps"
}
```

The `state_dict_bytes` payload carries a plain dict of tensors — it is
**not** a pickled `nn.Module`.  This sidesteps the
`torch.save(nn.Module)` coupling problem (domain research §R3: torch's
2.6+ `weights_only=True` default rejects non-tensor pickle frames):
`state_dict` is just tensors plus parameter / buffer names, and
`torch.load(..., weights_only=True, map_location="cpu")` applies the
safety rail on the inner bytes without forcing the class graph into
the serialised blob.

`config_dump` is the reconstruction seed.  Everything needed to rebuild
the `nn.Module` skeleton (`hidden_sizes`, `activation`, `dropout`,
`target_column`, `feature_columns`, `device`) survives a Pydantic
round-trip; schema drift raises on load.  `load_state_dict(strict=True)`
then fails loudly if any buffer (especially the four scaler buffers
from plan D4) is missing or extra.

The scaler buffers (`feature_mean`, `feature_std`, `target_mean`,
`target_std`) ride inside `state_dict` because
`self.register_buffer(...)` puts them there — no sibling joblib scaler
file and no bespoke serialisation branch.  That is the whole reason
plan D4 picks `register_buffer` over a sibling sklearn `StandardScaler`.

joblib wraps the outer envelope so every model family's artefact keeps
the same filename (`model.joblib`) and the Stage 12 `skops.io`
graduation applies uniformly.  Stage 9's "only load artefacts we wrote
ourselves" rule is unchanged.

### 5. Cold start per fold (plan D8)

The Stage 6 harness already calls `model.fit(...)` per fold and expects
re-entrancy (a second call discards the previous fit).  `NnMlpModel`
honours this by construction: `fit` reinitialises `self._module` via
`_make_mlp` on every call, which means a fresh weight init (the
PyTorch default `kaiming_uniform` on `nn.Linear` — driven by the
per-fold seed through `_seed_four_streams`).

The per-fold seed itself is `fold_seed = config.project.seed + fold_index`
when the harness orchestrates (plan D8) or `config.seed` directly when
the caller pins it (e.g. the notebook's live-demo fit at `seed=0`).

The analytical justification is Ash & Adams (NeurIPS 2020): warm-starting
across rolling-origin folds leaks training-fold information into later
folds' "fresh" evaluations and hurts generalisation.  For a 1-hidden-layer
128-unit MLP on hourly demand data, the per-fold fit budget is bounded
to ~10 s × 8 folds = ~80 s on a 4-core laptop — well inside the AC-5
"reasonable time on a laptop CPU" envelope.

### 6. Internal validation tail + early stopping (plan D9)

Early stopping is patience-based on **validation loss**, with best-epoch
weight restore.  The validation slice is an internal 10 % **tail** of
the training slice (last 10 % by index, not a random split — otherwise
the split would leak future information into the validation loss on a
time-series).

If validation loss does not improve for `patience` consecutive epochs,
`fit` restores the best-epoch `state_dict` snapshot and stops.  The
registered artefact is therefore the *best* epoch's weights, not the
last — a user who inspects `metadata.hyperparameters["best_epoch"]`
sees exactly which epoch the saved weights came from.

### 7. Device selection (plan D11)

`_select_device(preference: str) -> torch.device` honours
`NnMlpConfig.device` with the resolution order:

| `preference` | Resolution |
|------|-----------|
| `"auto"` | `cuda` if `torch.cuda.is_available()`, else `mps` if `torch.backends.mps.is_available()`, else `cpu` |
| `"cpu"` / `"cuda"` / `"mps"` | pinned verbatim |
| any other string | `ValueError` |

The resolved device is:

- Logged at INFO at fit time so the facilitator can see which device
  the run landed on.
- Persisted in `metadata.hyperparameters["device_resolved"]` so the
  Stage 9 registry leaderboard can surface it.
- Stored in the save envelope's `device_resolved` field so a load-time
  reader knows which device the weights came from.

`load` always materialises the `state_dict` via
`map_location="cpu"` so a CUDA-trained artefact loads cleanly on a
CPU-only host (the common case for a Stage 12 serving stub).

## Stage 11 addition — `NnTemporalModel` (TCN)

Stage 11 lands the second torch-backed family in this sub-layer, a
Temporal Convolutional Network conforming to the same
`Model` protocol surface as `NnMlpModel`.  Every Stage-10 contract
above (D5 envelope, D7' four-stream seed, D8 cold-start, D9 best-epoch
restore, D11 device selection) is inherited unchanged; the additions
below are temporal-specific.  Shipping a second concrete class is the
event that **fires the Stage 10 D10 extraction seam** — see the
`_training.py` row in the module inventory above.

### A. Architecture (plan D1)

`_NnTemporalModuleImpl` is a stack of `num_blocks` residual TCN blocks.
Each block (`_TemporalBlockImpl`) holds two `Conv1d` layers with
`kernel_size=k`, `padding=0`, `dilation=2**block_idx`, optionally
wrapped in `torch.nn.utils.parametrizations.weight_norm`.  Causal
padding is realised inside the block's `forward` via
`F.pad(x, (left_pad, 0))` where `left_pad = (k - 1) * dilation` — the
Bai et al. (2018) recipe.  Activations are ReLU; LayerNorm is applied
over the channel dimension (transpose to `(B, L, C)` → `LayerNorm(C)`
→ transpose back) — *not* over time, because BatchNorm-style
running statistics across timesteps would silently couple the future
into the present.  A 1×1 head (`Conv1d(channels, 1, kernel_size=1)`)
maps the channel axis down to one scalar per timestep; the forward
pass reads the *last* timestep only, producing one prediction per
window.

The four scaler buffers (`feature_mean`, `feature_std`, `target_mean`,
`target_std`) ride inside `state_dict()` via `register_buffer` — same
recipe as Stage 10, same regression guard
(`load_state_dict(strict=True)` catches a missing buffer loudly).
Features are normalised inside `forward()`; the target is normalised
*before* the `_SequenceDataset` is constructed so the MSE loss in the
shared training loop operates on the O(1) normalised scale.

The CUDA defaults (8 blocks × 128 channels × kernel 3) yield a
receptive field of `1 + 2·(k-1)·(2^num_blocks − 1) = 1021` timesteps —
enough to cover the weekly cycle (168 h) with ~6× headroom.  A
`@model_validator` on `NnTemporalConfig` rejects `seq_len <
max(2·kernel_size, receptive_field // 8)` so a degenerate
configuration (window smaller than a meaningful fraction of the
receptive field) cannot ship silently.

### B. `_SequenceDataset` (plan D7) — lazy windowing

```python
class _SequenceDataset(torch.utils.data.Dataset):
    """Lazy sliding-window dataset for temporal training.

    Stores two flat numpy arrays — features ``(N, n_features)`` and
    target ``(N,)`` — and computes ``(features[i:i+seq_len],
    target[i+seq_len])`` per ``__getitem__`` call.  ``__len__`` is
    ``N - seq_len`` (number of valid windows).  Eager materialisation
    of the full ``(N - seq_len, seq_len, n_features)`` tensor would
    cost ~1.4 GB on the intent's default 44-calendar + ~6-weather
    feature set; the lazy path is ~10 MB.
    """
```

The class is private to `temporal.py` because no other Stage 11 surface
needs it.  Lazy windowing is load-bearing for both the laptop-CPU
override path and the meetup-demo memory budget; the eager pattern is
guarded against by `test_sequence_dataset_does_not_eagerly_materialise_
full_tensor`, which asserts `__init__` does not allocate a tensor
larger than the input frame.

### C. Pattern A exogenous handling (plan D3)

The dataset yields windows of shape `(seq_len, n_features)` where
`n_features` is the same column set the MLP uses (weather + calendar
one-hots).  No separate "known-future" branch; no TFT-style side
channel.  At day-ahead horizon the calendar / weather features are
already aligned to the target index in the Stage 5 feature table, so
Patterns A and B (in-sequence vs side-channel) are informationally
equivalent — Pattern A is strictly simpler (single-branch
`nn.Module`, no decoder, no special collation).

### D. Save-envelope addition (plan D5 / R7)

The Stage 10 single-joblib envelope is preserved verbatim plus *two
new fields*:

```python
{
    "state_dict_bytes": bytes,
    "config_dump": dict[str, Any],
    "feature_columns": tuple[str, ...],
    "seq_len": int,                  # NEW at Stage 11 — redundant with
                                     # config_dump["seq_len"], explicit
                                     # so the load path can sanity-check
                                     # before reconstructing the module
    "warmup_features": pd.DataFrame, # NEW at Stage 11 — the last
                                     # ``seq_len`` rows of the training
                                     # feature frame (see below)
    "seed_used": int,
    "best_epoch": int,
    "loss_history": list[dict[str, float]],
    "fit_utc": str,
    "device_resolved": str,
}
```

The redundant `seq_len` field is intentional.  Reading it before the
Pydantic re-validation of `config_dump` lets a future migration tool
inspect a stale artefact without round-tripping through a possibly
breaking schema.  `test_nn_temporal_save_and_load_round_trips_seq_len_
and_state_dict` is the regression guard.

`warmup_features` carries the last `seq_len` rows of the training
feature frame and is **load-bearing for the harness contract**.  A
sliding-window TCN that consumes `seq_len` historical hours per
prediction naturally outputs `len(features) - seq_len` predictions for
a feature frame of length `len(features)`; the Stage 6 harness
(`harness.evaluate(...)`) and the protocol shape `predict(features)
-> pd.Series` indexed on `features.index` both expect
`len(y_pred) == len(features)`.  `predict()` therefore concatenates
`warmup_features` to the front of the caller's input frame, runs the
windowed forward pass, and slices the prediction back to the caller's
index.  Without `warmup_features` the load path could not reproduce
the predict-time alignment the harness enforces, so every loaded
`NnTemporalModel` would silently drop the first `seq_len` rows of any
prediction request.

This field is **the one Stage 11 contract addition not named in plan
D5**.  The plan implicitly assumed alignment was handled inside
`predict()` via re-indexing; the implementation chose the warmup
prefix because (i) it preserves the protocol's
`len(y_pred) == len(features)` invariant without truncating the
caller's index, (ii) it makes the alignment honest about what the
model actually conditions on (the warmup rows ride the
`feature_mean` / `feature_std` scalers exactly as training data did),
and (iii) it survives joblib round-trip without a custom serialiser.
The cost is a small parquet-shaped tail in the artefact; on the
Stage 11 production defaults at `seq_len=168` × ~50 columns ×
`float64` this is ~70 kB, negligible against the `state_dict_bytes`
payload.  Adversarial regression guards
(`test_nn_temporal_load_rejects_envelope_with_mismatched_seq_len` and
`test_nn_temporal_load_rejects_envelope_missing_warmup_features`)
pin both load-time defences against silent corruption.

### E. Training-loop ownership has moved

Both `NnMlpModel.fit` and `NnTemporalModel.fit` now build their own
collaborators (module + dataloaders + optimiser + criterion) and call
the shared `bristol_ml.models.nn._training.run_training_loop(module,
train_loader, val_loader, *, optimiser, criterion, device, max_epochs,
patience, loss_history, epoch_callback) -> tuple[best_state_dict,
best_epoch]`.  The Stage 10 D7' seed recipe (`_seed_four_streams`)
moved with the loop and is imported from `_training.py` by both
families.  The protocol-level shape of `loss_history_` —
`list[dict[str, float]]` with keys `{"epoch", "train_loss",
"val_loss"}` — is preserved bit-for-bit; the live-loss-curve cell in
both notebooks (`notebooks/10-simple-nn.ipynb`,
`notebooks/11-complex-nn.ipynb`) consumes the same shape via the
`epoch_callback` seam.  Two structural regression guards
(`test_nn_mlp_fit_still_uses_shared_training_loop_after_extraction`
and `test_nn_temporal_fit_uses_shared_training_loop`) pin both
classes to the shared call-site so a future refactor cannot silently
re-inline the loop body into either family.

### F. Harness-factory catch-up (plan D14)

Stage 10 added a `NnMlpConfig` branch to `train.py`'s inline dispatcher
but did not add the matching branch to
`evaluation.harness._build_model_from_config`.  The gap was latent
(the train CLI was the only user of `NnMlpModel` until Stage 11); the
codebase map flagged it as a one-line discrepancy.  Stage 11 T6 closes
the gap *in the same commit* as the new `NnTemporalConfig` branch
(D13 clause iii), so anyone driving `model=nn_mlp` through the harness
CLI's internal factory now reaches the right family.  The regression
guard is `test_harness_build_model_from_config_dispatches_nn_mlp_
after_catch_up`; see the D14 note in `models/nn/CLAUDE.md` for the
audit-trail framing.

### G. Notebook surface — ablation table (plan D10 + AC-3 / AC-5)

`notebooks/11-complex-nn.ipynb` ships the Stage 11 demo moment: a
predict-only ablation table covering all six families (`naive`,
`linear`, `sarimax`, `scipy_parametric`, `nn_mlp`, `nn_temporal`) on
the same single-holdout slice (D12).  The cell loops over
`registry.list_runs(...)` results, calls `registry.load(run_id)` then
`model.predict(X_holdout)`, and composes metrics inline using the
existing `bristol_ml.evaluation.metrics` functions.  No re-fitting of
already-registered runs (AC-5 by construction); the cell is
explicitly tested with a `fit`-is-`raise` monkeypatch
(`test_notebook_11_ablation_cell_does_not_refit_registered_runs`).

The shipped table renders four metric columns (`mae`, `mape`, `rmse`,
`wape`) plus `model` and `run_id`.  Plan D10's seven-column spec also
named `MAE_ratio_vs_NESO`, `training_time_s`, and `param_count`; those
three are deferred and surfaced in the notebook's closing markdown
cell.  The deferral is **not silently cut**: `MAE_ratio_vs_NESO`
requires a warm NESO archive plus the Stage 4 half-hourly-to-hourly
alignment helper, and `training_time_s` / `param_count` would require
a `SidecarFields` extension.  Both are candidate items for a future
housekeeping stage.

## Upgrade seams

Each of these is swappable without touching downstream code.

| Swappable | Load-bearing |
|-----------|--------------|
| Architecture (`hidden_sizes`, `activation`, `dropout` for MLP; `num_blocks`, `channels`, `kernel_size` for TCN) | `NnMlpConfig.model_dump()` / `NnTemporalConfig.model_dump()` round-trip through `config_dump` |
| Optimiser (Adam → AdamW → SGD) | The `run_training_loop` shared contract in `_training.py` — per-epoch `train_loss` / `val_loss` floats; the optimiser is built by the calling family |
| Early-stopping policy (patience → max-minus-N plateau) | The `loss_history_` shape: `list[dict]` with keys `{"epoch", "train_loss", "val_loss"}` |
| Scaler family (z-score → min-max → robust) | `register_buffer` round-trip through `state_dict` |
| Serialisation backend (joblib envelope → `skops.io` envelope at Stage 12) | `state_dict_bytes` + `config_dump` as the two reconstruction primitives (plus `seq_len` for `NnTemporalModel`) |
| Device policy (CPU-only → auto → pinned) | `_select_device(preference)` returning `torch.device` |
| Training-loop ownership (per-model → shared `_training.py`) | Fired at Stage 11; future swap is `_training.py` → `_training_v2.py` behind a feature flag |
| Weight-norm API (legacy → parametrizations) | Already on the parametrizations API at Stage 11; legacy fallback retained for `torch < 2.1` |
| Window-data adapter (Stage 11 `_SequenceDataset` lazy → an eager `torch.compile`-friendly variant) | The two-call surface: `__len__` returns `N - seq_len`; `__getitem__(i)` returns `(features[i:i+seq_len], target[i+seq_len])` as `float32` tensors |

## Module inventory

| Module | Family | Stage | Status | Notes |
|--------|--------|-------|--------|-------|
| `models/nn/__init__.py` | — (re-exports) | 10 | Shipped | Lazy `__getattr__` re-exports `NnMlpModel` (Stage 10) and `NnTemporalModel` (Stage 11); no heavy imports at package load. |
| `models/nn/__main__.py` | — (CLI entry) | 10 | Shipped | `python -m bristol_ml.models.nn` delegates to `mlp.py --help`, printing the resolved `NnMlpConfig` schema + help text. |
| `models/nn/mlp.py` | Simple MLP | 10 | Shipped | `NnMlpModel` + `_NnMlpModule` + `_select_device`, `_make_mlp`. 1 hidden layer × 128 units default; ~40 k parameters. Handles CPU / CUDA / MPS via `_select_device`. State-dict-inside-joblib artefact envelope per plan D5 revised. As of Stage 11 calls `_training.run_training_loop` rather than a local loop. |
| `models/nn/_training.py` | — (shared training loop) | 11 | Shipped | `run_training_loop(...)` + `_seed_four_streams`. Extracted from `mlp.py` at Stage 11 T1 firing the Stage 10 D10 seam. Both `NnMlpModel.fit` and `NnTemporalModel.fit` import and call it; the shared body owns the per-epoch accounting, val-loss tracking, patience-based early stopping, best-epoch weight restore, and `loss_history` population. |
| `models/nn/temporal.py` | TCN (temporal) | 11 | Shipped | `NnTemporalModel` + `_NnTemporalModuleImpl` + `_TemporalBlockImpl` + `_SequenceDataset` + `_select_device`. 8 dilated residual blocks × kernel 3 × 128 channels at the CUDA defaults; ~1 M parameters; receptive field 1021 steps. Single-joblib envelope inherits Stage 10 D5 plus a redundant `seq_len` field for explicit round-trip auditing. |

## Known trade-offs

- **Single-joblib envelope vs two-file layout.** The research-draft
  proposed two files (`model.pt` + `hyperparameters.json`) and the
  registry passing a directory.  Stage 9's registry hard-codes
  `artefact/model.joblib` as a file path (see `registry/_fs.py::_atomic_write_run`);
  refactoring it to pass a directory was out of scope for Stage 10.
  The single-joblib envelope is functionally equivalent — `state_dict`
  is a dict-of-tensors, so pickling it through joblib does not carry
  the `torch.save(nn.Module)` coupling problem, and
  `torch.load(..., weights_only=True)` on the inner bytes still applies
  the 2.6+ safety rail.  Plan D5 revised.
- **No `use_deterministic_algorithms(True)`.** Intent AC-2 carves out
  GPU non-determinism; `cudnn.deterministic = True` + `cudnn.benchmark
  = False` catch the dominant sources of cuDNN nondeterminism without
  the throughput hit of forcing every op onto a deterministic kernel.
  Plan H-5 / OQ-A closed `NO`.
- **No auto-saved loss-curve PNG.** `loss_history_` + `plots.loss_curve`
  satisfy AC-3.  Coupling the plots helper to the registry save path
  adds a module dependency (`models.nn` → `evaluation.plots`) for no
  AC gain.  Scope-diff single highest-leverage cut (NFR-4).
- **Hand-rolled training loop at Stage 10.**  Plan D10 justifies this:
  a walkable loop at Stage 10 is load-bearing for the meetup pedagogy
  (DESIGN §1.1), and the Stage 11 extraction trigger is one named
  call-site.  Shipping a `BaseTorchModel` abstraction now would bind
  Stage 11's design before Stage 11's requirements are understood
  (scope-diff X7 cut).
- **`_select_device` duplicated in `mlp.py` and `temporal.py`.**  The
  Stage 11 extraction at T1 lifted the training loop and
  `_seed_four_streams` into `_training.py` but deliberately left
  `_select_device` defined twice — once per family — so each module's
  log message can name its own family verbatim ("`NnMlpModel resolving
  device 'auto' to 'cuda'`" rather than a generic
  "`bristol_ml.models.nn._training resolving device …`").  The two
  copies are byte-identical at Stage 11; consolidation into
  `_training.py` is the obvious next step at the third NN family.
  The trade-off is one duplicated 30-line helper now versus a
  wrapper-per-family at the consolidation point; the duplication keeps
  the call-site honest about which family is asking, at the cost of a
  search-and-replace when the helper next changes shape.  Tracked as
  a follow-up housekeeping item (no separate hot-fix branch) for the
  next NN-family stage.

## Open questions

- **`_training.py` extraction timing.** The plan names "the arrival of
  Stage 11's second torch-backed model" as the trigger.  If Stage 11
  lands with a materially different training-loop shape (gradient
  accumulation, mixed precision, distributed data parallel), the
  extraction may be a thin shared core + per-model overrides rather
  than a single shared body.  Revisit at Stage 11 T1.
- **Hyperparameter search composition.** Inherited from the parent
  `models.md` — NN families bring a realistic tuning surface (lr,
  weight_decay, batch_size, hidden_sizes) but the harness has no hook
  for nested cross-validation.  Revisit at Stage 11 / Stage 17 when the
  competitive-NN tuning story is next load-bearing.
- **Cross-version load compatibility for `state_dict`.**  A Stage 10
  artefact loads cleanly into Stage 10 `NnMlpModel`.  A future refactor
  (adding a dropout layer, changing activation semantics) would produce
  a `state_dict` key mismatch that `load_state_dict(strict=True)`
  catches loudly.  The plan's position is that catching it loudly is
  correct; a "migrate old artefacts" story belongs at the stage that
  first hits the problem.
- **Device persistence on save.**  The artefact records
  `device_resolved` (the device the fit ran on) but `load` always
  materialises on CPU.  Whether `load` should honour the recorded
  device is undecided — the Stage 12 serving stub is CPU-only by
  DESIGN §5.5, so "always load on CPU" is the right default today.
  Revisit if a facilitator wants to reload a CUDA-trained artefact
  for further training on the same GPU.

## References

- [`docs/intent/10-simple-nn.md`](../../intent/10-simple-nn.md) — the
  five ACs + "Points for consideration".
- [`docs/intent/11-complex-nn.md`](../../intent/11-complex-nn.md) —
  Stage 11 ACs + the ablation-table demo moment.
- [`docs/plans/completed/10-simple-nn.md`](../../plans/completed/10-simple-nn.md)
  — twelve decisions, five NFRs, scope diff.
- [`docs/plans/completed/11-complex-nn.md`](../../plans/completed/11-complex-nn.md)
  — fourteen kept decisions (two cut), seven NFRs, scope diff;
  Stage 10 D10 extraction seam fired here.
- [`docs/lld/stages/10-simple-nn.md`](../../lld/stages/10-simple-nn.md)
  — Stage 10 retrospective.
- [`docs/lld/stages/11-complex-nn.md`](../../lld/stages/11-complex-nn.md)
  — Stage 11 retrospective; observed wall-clocks, ablation results,
  AC-5 reconciliation.
- [`docs/lld/research/10-simple-nn-requirements.md`](../../lld/research/10-simple-nn-requirements.md),
  [`codebase.md`](../../lld/research/10-simple-nn-codebase.md),
  [`domain.md`](../../lld/research/10-simple-nn-domain.md),
  [`scope-diff.md`](../../lld/research/10-simple-nn-scope-diff.md) —
  Stage 10 Phase 1 discovery artefacts.
- [`docs/lld/research/11-complex-nn-requirements.md`](../../lld/research/11-complex-nn-requirements.md),
  [`codebase.md`](../../lld/research/11-complex-nn-codebase.md),
  [`domain.md`](../../lld/research/11-complex-nn-domain.md),
  [`scope-diff.md`](../../lld/research/11-complex-nn-scope-diff.md) —
  Stage 11 Phase 1 discovery artefacts.
- [`src/bristol_ml/models/nn/CLAUDE.md`](../../../src/bristol_ml/models/nn/CLAUDE.md)
  — module-local concrete surface guide.
- [`src/bristol_ml/models/nn/mlp.py`](../../../src/bristol_ml/models/nn/mlp.py),
  [`temporal.py`](../../../src/bristol_ml/models/nn/temporal.py),
  [`_training.py`](../../../src/bristol_ml/models/nn/_training.py)
  — implementations.
- [`docs/architecture/layers/models.md`](models.md) — parent layer doc
  (the `Model` protocol + `ModelMetadata` + joblib IO that this
  sub-layer conforms to).
- [`docs/architecture/layers/registry.md`](registry.md) — the Stage 9
  file-path contract that drove the plan D5 revision.
- Ash, J., & Adams, R. P. (2020). *On warm-starting neural network
  training*. NeurIPS. (Plan D8 cold-start justification.)
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical
  evaluation of generic convolutional and recurrent networks for
  sequence modeling*. arXiv 1803.01271. (Plan D1 TCN recipe.)
- Wong, B. (2011). *Points of view: Color blindness*. Nature Methods
  8:441. (Stage 6 D2 — the Okabe-Ito palette used by `plots.loss_curve`
  at plan D6.)
- PyTorch docs — *Reproducibility*:
  <https://pytorch.org/docs/stable/notes/randomness.html>. (Plan D7'.)
