# Models — neural-network sub-layer

- **Status:** Provisional — first realised by Stage 10 (`NnMlpModel` — simple
  feed-forward MLP; shipped). Revisit at Stage 11 (complex / temporal
  neural network — inherits this sub-layer's training-loop conventions,
  reproducibility recipe, and `state_dict`-inside-joblib artefact envelope).
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
- **Key plan:** [`docs/plans/active/10-simple-nn.md`](../../plans/active/10-simple-nn.md)
  (moved to `completed/` at T7) — decisions D1 (dependency), D3
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

Stage 11 (complex / temporal neural network) will add a second class
that shares exactly these two concerns.  Rather than rewrite the
training-loop idiom in every subsequent NN stage, this sub-layer names
the conventions once.  The `src/bristol_ml/models/nn/` package is the
extraction destination flagged at plan D10 — when Stage 11's model
arrives with a second hand-rolled loop, the training-loop body in
`NnMlpModel._run_training_loop` moves to
`src/bristol_ml/models/nn/_training.py` under a shared helper, and the
two classes call into it.

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
| Gradient clipping, LR scheduling as configurable knobs | — | Stage 11 owns these behind a shared `_training.py` helper |
| `BaseTorchModel` abstract base class | — | Stage 11 extraction trigger; a premature ABC now would bind Stage 11's design before its requirements are known |
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
├── __init__.py          # exports: NnMlpModel
├── __main__.py          # `python -m bristol_ml.models.nn` → delegates to mlp.py
├── mlp.py               # NnMlpModel + _NnMlpModule + module-level helpers
└── CLAUDE.md            # module guide (this file's sibling)
```

`_training.py` is **not** shipped at Stage 10.  It lands at Stage 11 as
the extraction destination when the second torch-backed model arrives.
Plan D10 names the refactor trigger explicitly: the extraction happens
when Stage 11's training loop diverges from Stage 10's by more than
"add one optimiser kwarg", not before.

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

## Upgrade seams

Each of these is swappable without touching downstream code.

| Swappable | Load-bearing |
|-----------|--------------|
| Architecture (`hidden_sizes`, `activation`, `dropout`) | `NnMlpConfig.model_dump()` round-trips through `config_dump` |
| Optimiser (Adam → AdamW → SGD) | The `_run_training_loop` private contract — per-epoch `train_loss` / `val_loss` floats |
| Early-stopping policy (patience → max-minus-N plateau) | The `loss_history_` shape: `list[dict]` with keys `{"epoch", "train_loss", "val_loss"}` |
| Scaler family (z-score → min-max → robust) | `register_buffer` round-trip through `state_dict` |
| Serialisation backend (joblib envelope → `skops.io` envelope at Stage 12) | `state_dict_bytes` + `config_dump` as the two reconstruction primitives |
| Device policy (CPU-only → auto → pinned) | `_select_device(preference)` returning `torch.device` |
| Training-loop ownership (per-model → shared `_training.py`) | `_run_training_loop` signature — the Stage 11 extraction trigger |

## Module inventory

| Module | Family | Stage | Status | Notes |
|--------|--------|-------|--------|-------|
| `models/nn/__init__.py` | — (re-exports) | 10 | Shipped | `from bristol_ml.models.nn.mlp import NnMlpModel`; no heavy imports at package load. |
| `models/nn/__main__.py` | — (CLI entry) | 10 | Shipped | `python -m bristol_ml.models.nn` delegates to `mlp.py --help`, printing the resolved `NnMlpConfig` schema + help text. |
| `models/nn/mlp.py` | Simple MLP | 10 | Shipped | `NnMlpModel` + `_NnMlpModule` + `_select_device`, `_seed_four_streams`, `_make_mlp`. 1 hidden layer × 128 units default; ~40 k parameters. Handles CPU / CUDA / MPS via `_select_device`. State-dict-inside-joblib artefact envelope per plan D5 revised. |
| `models/nn/_training.py` (planned) | — (shared training loop) | 11 | Planning | Extraction destination when Stage 11's second torch-backed model arrives. Named at plan D10; not shipped at Stage 10. |

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
- [`docs/plans/active/10-simple-nn.md`](../../plans/active/10-simple-nn.md)
  (moved to `completed/` at T7) — twelve decisions, five NFRs, scope
  diff.
- [`docs/lld/stages/10-simple-nn.md`](../../lld/stages/10-simple-nn.md)
  — Stage 10 retrospective.
- [`docs/lld/research/10-simple-nn-requirements.md`](../../lld/research/10-simple-nn-requirements.md),
  [`codebase.md`](../../lld/research/10-simple-nn-codebase.md),
  [`domain.md`](../../lld/research/10-simple-nn-domain.md),
  [`scope-diff.md`](../../lld/research/10-simple-nn-scope-diff.md) —
  Phase 1 discovery artefacts.
- [`src/bristol_ml/models/nn/CLAUDE.md`](../../../src/bristol_ml/models/nn/CLAUDE.md)
  — module-local concrete surface guide.
- [`src/bristol_ml/models/nn/mlp.py`](../../../src/bristol_ml/models/nn/mlp.py)
  — implementation.
- [`docs/architecture/layers/models.md`](models.md) — parent layer doc
  (the `Model` protocol + `ModelMetadata` + joblib IO that this
  sub-layer conforms to).
- [`docs/architecture/layers/registry.md`](registry.md) — the Stage 9
  file-path contract that drove the plan D5 revision.
- Ash, J., & Adams, R. P. (2020). *On warm-starting neural network
  training*. NeurIPS. (Plan D8 cold-start justification.)
- Wong, B. (2011). *Points of view: Color blindness*. Nature Methods
  8:441. (Stage 6 D2 — the Okabe-Ito palette used by `plots.loss_curve`
  at plan D6.)
- PyTorch docs — *Reproducibility*:
  <https://pytorch.org/docs/stable/notes/randomness.html>. (Plan D7'.)
