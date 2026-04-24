# Stage 10 — Simple MLP for GB day-ahead electricity demand: domain research

**Date:** 2026-04-23
**Target plan:** `docs/plans/active/10-simple-nn.md` (not yet created)
**Intent source:** `docs/intent/10-simple-nn.md`
**Baseline SHA:** main @ `6267cc0` (Stage 9 merged)

**Scope:** External literature and primary tool documentation to inform
Stage 10 plan decisions.  Numbered subsections (R1–R8) so the plan can
cite by reference.  British English throughout.

---

## R1 — Reproducibility in PyTorch (CPU-only)

### Canonical sources

| Source | Summary |
|--------|---------|
| [PyTorch Reproducibility — docs.pytorch.org](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Official reference; covers seeds, deterministic algorithms, DataLoader workers |
| [torch.use_deterministic_algorithms — docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) | Lists every operation affected and whether it falls back or raises |
| [DataLoader + NumPy seed bug — tanelp.github.io](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/) | Concrete demonstration of the `num_workers > 0` inheritance trap |

### Which seeds are needed for CPU-only?

The PyTorch reproducibility page lists three independent RNG streams
that must each be seeded:

1. `torch.manual_seed(seed)` — seeds PyTorch's CPU RNG (and CUDA RNG
   when a GPU is present, though it is a no-op for CUDA streams on a
   CPU-only run).
2. `random.seed(seed)` — seeds CPython's built-in RNG, required if any
   custom operators or data-augmentation code calls `random.*`.
3. `numpy.random.seed(seed)` — seeds NumPy's legacy global RNG,
   required whenever NumPy is used in a Dataset or feature pipeline.

`torch.cuda.manual_seed_all(seed)` is CUDA-only.  On a CPU-only run it
is harmless but unnecessary; including it costs nothing and
future-proofs the code if the project ever moves to GPU (Stage 11+).

### `torch.use_deterministic_algorithms(True)` — cost and scope

The official documentation states: "Deterministic operations are often
slower than nondeterministic operations, so single-run performance may
decrease for your model."  For CPU tensor operations the affected
operations are primarily scatter/gather and indexed writes:

- `torch.Tensor.index_put()` with `accumulate=True` — falls back to a
  serialised kernel.
- `torch.Tensor.put_()` with `accumulate=True` — falls back.
- `torch.Tensor.put_()` with `accumulate=False` — raises `RuntimeError`
  (no deterministic alternative exists).
- `torch.Tensor.index_copy()` on CPU — falls back.
- `torch.Tensor.__getitem__()` when differentiating with a
  tensor-list index — falls back.

None of these are on the critical path of a standard forward-pass MLP
on tabular data (`nn.Linear` → activation → `nn.Linear`).  The
performance penalty for Stage 10's use case is therefore expected to
be negligible on CPU.

The GPU-facing cost is much higher.  A PyTorch GitHub issue (#109856)
documented a ~3× slowdown (8.91 it/s → 3.35 it/s) under
`use_deterministic_algorithms(True)` on CUDA in PyTorch 2.0, with GPU
utilisation dropping below 50 % under deterministic mode.  This is
irrelevant for Stage 10 (CPU laptop) but flags a real cost for
Stage 11.

### DataLoader workers — the `num_workers > 0` trap

When `num_workers > 0`, each worker subprocess is forked from the
parent, inheriting the parent's NumPy and Python RNG state identically.
Every worker therefore draws identical random sequences, making
shuffling and any stochastic augmentation correlated rather than
independent.  Tanel Pärnamaa's analysis demonstrates this clearly:
datasets that call `numpy.random` inside `__getitem__` return the same
values from all workers when no `worker_init_fn` is set.

The fix is a `worker_init_fn` that re-seeds each worker from a
per-worker offset:

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
```

This must be passed alongside a seeded `torch.Generator` to
`DataLoader`:

```python
g = torch.Generator()
g.manual_seed(seed)
DataLoader(dataset, worker_init_fn=seed_worker, generator=g, num_workers=N)
```

Setting `num_workers=0` (in-process loading) sidesteps the problem
entirely.  For a ~44 k-row tabular dataset with pre-loaded tensors on
CPU, `num_workers=0` is both simpler and adequate — the bottleneck is
not I/O but forward-pass arithmetic.

### CUDA flag note (out of scope for Stage 10)

For CUDA ≥ 10.2, `torch.use_deterministic_algorithms(True)` requires
the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or
`:16:8`) to be set before PyTorch initialises cuBLAS; omitting it
raises a `RuntimeError`.  This is a Stage 11 concern; no action needed
for Stage 10 CPU-only training.

### Recommendation for Stage 10

Define a single `set_seed(seed: int) -> None` helper that calls
`torch.manual_seed`, `numpy.random.seed`, `random.seed`, and
`torch.cuda.manual_seed_all` (for forward-compatibility).  Opt in to
`torch.use_deterministic_algorithms(True)` — for a CPU MLP the
performance cost is zero on the hot path.  Use `num_workers=0` in
`DataLoader` to avoid the worker-seed trap without requiring
`worker_init_fn` boilerplate.  Document the three-stream seed contract
in the module docstring.

---

## R2 — Input normalisation strategies for tabular regression

### Canonical sources

| Source | Summary |
|--------|---------|
| [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) | Authoritative semantics for fit/transform and the leakage risk |
| [PyTorch `register_buffer` docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) | Non-parameter state that survives `state_dict()` round-trips |
| [MachineLearningMastery — Data Scaling for Neural Networks](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/) | Applied guidance including target normalisation for regression |
| [PyTorch Serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html) | Confirms buffers are included in `state_dict` by default |

### StandardScaler vs. MinMaxScaler

**StandardScaler (z-score)** centres each feature to mean 0 and unit
variance.  It equalises gradient scales across features, which is
important for gradient-based optimisers when features have
heterogeneous units (temperatures in Kelvin, hour-of-day 0–23, binary
holiday flags).  It is robust to heavy-tailed distributions unless
outliers are extreme.

**MinMaxScaler** maps each feature to [0, 1].  It is sensitive to
outliers — a single extreme value can collapse 99 % of the data into
a narrow band.  Its main advantage is preserving zero-valued features
(e.g., a zero-precipitation column stays zero), which is secondary for
weather-driven demand data.

**No normalisation** is acceptable for tree-based models but actively
harmful for MLPs: gradient magnitudes become dominated by high-variance
features, and learning slows or diverges.  The choice between Standard
and MinMax is not load-bearing for a simple MLP on this scale;
StandardScaler is the conventional default for neural regression and
the recommendation here.

### Leakage hazard

The scaler must be fit only on training-fold data, never on validation
or test data.  Under the harness's rolling-origin contract, `fit()` is
called fresh per fold; the scaler must therefore be fit inside `fit()`
and the fitted statistics must be carried through to `predict()`.
This is the same discipline as sklearn pipelines — fit once on
training, transform both train and test consistently.

### Placement: inside the model as `register_buffer`

There are two natural placements for the scaler statistics:

**Option A — outside the model (sklearn preprocessor held by the
harness).**  The scaler is fit in `fit()`, stored on the wrapper
object, and applied in `predict()` before the tensor is passed to the
`nn.Module`.  The `nn.Module` receives normalised inputs and its
`state_dict` need not carry scaler parameters.

**Option B — inside the model as `register_buffer` tensors.**  The
scaler's `mean_` and `scale_` are registered as persistent,
non-trainable tensors via `self.register_buffer("input_mean", ...)`.
They appear in `state_dict()` round-trips automatically, move with the
model via `.to(device)`, and accompany the model wherever its
checkpoint goes.  `BatchNorm` uses exactly this pattern for its running
statistics.

Option B is recommended for Stage 10.  Storing scaler statistics as
buffers inside the model ensures a single artefact (the saved
`state_dict`) contains everything needed for inference.  Option A
requires coordinating two separate objects (model + scaler) in the
registry, which contradicts Stage 9's design goal of one file per
model run.  The precedent is well-established in core PyTorch itself.

The target variable (`nd_mw`) should also be normalised for regression
stability — large-scale targets (GW range) produce large gradient
values that can destabilise training.  Store target mean and scale as
named buffers too and inverse-transform in `predict()`.

---

## R3 — PyTorch serialisation semantics vs. the Stage 9 registry

### Canonical sources

| Source | Summary |
|--------|---------|
| [PyTorch Saving and Loading Models tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | `state_dict` vs. full-model comparison; security guidance |
| [PyTorch Serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html) | `register_buffer` in `state_dict`; `weights_only` shift in 2.6 |
| [torch.load docs](https://docs.pytorch.org/docs/stable/generated/torch.load.html) | `weights_only=True` default as of PyTorch 2.11 |
| [PyTorch 2.6 release blog](https://pytorch.org/blog/pytorch2-6/) | Official announcement of `weights_only=True` as new default |
| [joblib GitHub issue #1104](https://github.com/joblib/joblib/issues/1104) | Conflict between joblib multiprocessing and PyTorch DataLoader |

### `state_dict` vs. full-model save

`torch.save(model.state_dict(), path)` saves only parameter tensors and
registered buffers as a Python `dict`.  Loading requires reconstructing
the class first, then calling `load_state_dict()`.  It survives
refactors as long as parameter and buffer names are stable, and the
pickle payload is restricted to tensors.

`torch.save(model, path)` uses Python's `pickle` to serialise the
entire object graph, including a reference to the class definition's
import path.  A class rename, a module restructure, or a missing
import at load time causes a `ModuleNotFoundError` or `AttributeError`.
The PyTorch documentation explicitly recommends against this approach
for anything expected to outlive a single session.

### PyTorch 2.x `weights_only=True` security shift

As of **PyTorch 2.6.0** (released early 2025), `torch.load()` defaults
to `weights_only=True`.  This is a backwards-compatibility-breaking
change — the PyTorch developer mailing list described it as "expected
to be quite a BC-breaking change, especially if any `torch.load` calls
are not loading `state_dict`s of plain tensors."  The restricted
unpickler will only reconstruct plain tensors and primitive Python
types; anything richer raises `_pickle.UnpicklingError`.

Loading a full-model `torch.save(model, path)` under this default
fails unless `weights_only=False` is passed explicitly.  The docs
label this "inherently unsafe" when the source is untrusted.  Saving
and loading a `state_dict` is compatible with `weights_only=True` by
construction.

### Joblib round-trip of a `nn.Module`

Joblib is pickle-backed for Python objects.  It technically can
serialise a `nn.Module`, but:

- It inherits all the same class-path fragility as
  `torch.save(model, path)`.
- Joblib's large-array optimisation (memory-mapped NumPy arrays) is
  irrelevant for PyTorch tensors, which are not NumPy arrays.
- Loading a joblib-serialised `nn.Module` under PyTorch 2.6+ bypasses
  PyTorch's `weights_only` restriction entirely, providing no security
  benefit.
- joblib GitHub issue #1104 documents conflicts between joblib's
  multiprocessing and PyTorch's DataLoader, confirming the two are not
  designed to compose.

Joblib is the right choice for Stage 4's sklearn models.  It is the
wrong choice for PyTorch modules.

### Recommendation for `SimpleMlpModel.save(path)`

Use `torch.save(model.state_dict(), path / "model.pt")`.  Store
`ModelMetadata` as a sidecar `metadata.json` in the same directory
entry (matching the Stage 9 registry layout: one directory per run,
containing the model file and the metadata sidecar).  On load,
reconstruct the class, call `load_state_dict()` with
`weights_only=True` (the 2.6+ default).  Buffers (scaler statistics)
are included in the `state_dict` automatically and require no special
handling.

Do not wrap the `nn.Module` in joblib.  Do not use
`torch.save(model, path)` (full-object pickling).

---

## R4 — Early stopping and live loss-curve plotting

### Canonical sources

| Source | Summary |
|--------|---------|
| [Prechelt 1998 — "Early Stopping — But When?" (Springer)](https://link.springer.com/chapter/10.1007/3-540-49430-8_3) | Systematic empirical treatment; patience tradeoff analysed across 1,296 runs |
| [PyTorch Lightning EarlyStopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html) | Reference implementation of patience-based callback |
| [GitHub: Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) | Minimal standalone EarlyStopping class for raw PyTorch |
| [MachineLearningMastery — Managing checkpoints and early stopping](https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/) | Applied pattern: best-epoch weight restore |
| [livelossplot — GitHub](https://github.com/stared/livelossplot) | Live loss plotting library for Jupyter + PyTorch |
| [Matplotlib Backends — matplotlib.org](https://matplotlib.org/stable/users/explain/figure/backends.html) | Agg / headless CI guidance; `MPLBACKEND` environment variable |

### Conventional training-loop shape

The canonical PyTorch loop structure is:

1. `model.train()` — enables dropout and batch-norm training mode.
2. For each batch: `optimizer.zero_grad()` → forward pass → loss
   computation → `loss.backward()` → `optimizer.step()`.
3. After each epoch: switch to `model.eval()` with `torch.no_grad()`
   for validation pass.
4. Record train loss and validation loss per epoch.
5. Check early-stopping criterion; optionally save best-epoch
   checkpoint.

Calling `model.eval()` before the validation pass is not optional —
dropout and BatchNorm behave differently in training mode, producing
inconsistent validation metrics if omitted.

### Early stopping — Prechelt 1998

Prechelt's 1998 empirical study across 1,296 training runs found that
slower stopping criteria (longer patience) produce small
generalisation gains (~4 % on average) at substantially higher
compute cost (~4× longer training time).  For a CPU-laptop budget the
sweet spot is moderate patience (5–10 epochs).  The paper introduced
the "patience" concept now universally adopted: stop if validation
loss has not improved for `patience` consecutive checks.

The standard implementation saves a copy of `model.state_dict()`
whenever validation loss reaches a new minimum (best-epoch
restoration).  At training completion the best checkpoint is restored
regardless of whether early stopping fired, ensuring the returned
model is not from the final, potentially overfitted epoch.

### Live plotting — Jupyter vs. CLI headless

**Jupyter live plotting** uses `IPython.display.clear_output(wait=True)`
followed by `plt.show()` (or `display(fig)`) at the end of each epoch.
The `wait=True` flag suppresses flicker by deferring the clear until
the next output is ready.  Cost: one matplotlib redraw per epoch —
negligible at the scale of a small MLP where epochs take seconds.  The
`livelossplot` library wraps this pattern with a clean API and
supports PyTorch natively.

**Headless / CI environments** have no GUI backend.  Without setting a
backend, matplotlib may raise or warn about a missing display.  The
safe pattern is the `Agg` backend (pure raster, no display required),
set via:

```python
import matplotlib
matplotlib.use("Agg")   # must precede any pyplot import
```

or the environment variable `MPLBACKEND=Agg` before invocation.  Under
`Agg`, `plt.show()` is a no-op; `plt.savefig(path)` works normally.

### Recommended dual-mode pattern for Stage 10

Accumulate epoch losses into a list throughout training.  After
training completes, always call
`_save_loss_curve(train_losses, val_losses, output_dir)` which saves a
PNG using `plt.savefig()`.  In a Jupyter context, also call
`plt.show()`.  In a headless context the PNG is the output.

For live in-notebook feedback, accept a `live_plot: bool = False`
constructor or config argument that enables the `clear_output` redraw
per epoch.  When `live_plot=False` (the default, matching CI),
plotting happens only once at the end.  This satisfies intent AC-3
(loss curve produced by the loop) without requiring a display.

Do not make live plotting mandatory in the training contract.  CI must
pass without a display server.

---

## R5 — Rolling-origin efficiency for neural networks

### Canonical sources

| Source | Summary |
|--------|---------|
| [Hyndman fpp3 §5.10 — Time series cross-validation](https://otexts.com/fpp3/tscv.html) | Canonical description of rolling-origin evaluation |
| [Ash & Adams NeurIPS 2020 — "On Warm-Starting Neural Network Training"](https://proceedings.neurips.cc/paper/2020/hash/288cd2567953f06e460a33951f55daaf-Abstract.html) | Empirical demonstration that warm-start harms generalisation; shrink-perturb remedy |
| [Warm-Start vs. Cold-Start — ScienceDirect 2026](https://www.sciencedirect.com/science/article/abs/pii/S0893608026001097) | Confirms cold-start generalises better; warm-start introduces more pronounced overfitting |
| [Ogunlao 2020 — Cross-validation and reproducibility in NNs](https://ogunlao.github.io/2020/05/08/cross-validation-and-reproducibility-in-neural-networks.html) | Practitioner warning: do not carry weights across folds |

### Warm-start vs. cold-start per fold

Rolling-origin evaluation for a ~44 k-row time series with, say, 5
folds means re-training a small MLP 5 times.  On CPU, each training
run takes seconds to low minutes.  The cost of re-initialising weights
per fold (cold-start) is modest.

The statistical argument for cold-start is strong.  Ash and Adams
(NeurIPS 2020) demonstrate that warm-starting — initialising fold
*k+1* from the weights found at fold *k* — produces models that
generalise worse than cold-started models, with the generalisation gap
growing as more folds accumulate.  The root cause is that warm-started
weights have "memorised" the previous fold's loss landscape and
resist adapting to the expanded training set.  Ash and Adams propose
the "shrink and perturb" remedy — scale weights toward zero and add
noise — which partially recovers cold-start generalisation at
warm-start speed.

A 2026 study on gradient-based hyperparameter tuning corroborates:
"warm-start strategy yields a faster convergence rate, while it
obtains worse generalisation performance than cold-start strategy,
i.e., more pronounced overfitting to the validation set."

The existing SARIMAX implementation (Stage 7) re-fits per fold
unconditionally.  Consistency with that precedent, combined with the
statistical evidence, points firmly to cold-start for Stage 10.

### Recommendation

Re-initialise model weights and optimiser state at the start of each
fold (cold-start).  Derive a per-fold seed from the global seed and
fold index (e.g., `seed + fold_index`) so cross-fold variation is
deterministic and reproducible.  The compute overhead of 5
cold-starts on a CPU laptop is acceptable for a small MLP.  Do not
warm-start or carry optimiser momentum across folds.  If Stage 11
introduces a larger model where cold-start per fold becomes expensive,
revisit with the shrink-perturb technique as the first option.

---

## R6 — MLP architecture sizing for hourly tabular demand data

### Canonical sources

| Source | Summary |
|--------|---------|
| [Gorishniy et al. NeurIPS 2021 — "Revisiting Deep Learning Models for Tabular Data"](https://arxiv.org/abs/2106.11959) | Benchmarks MLP, ResNet, FT-Transformer; tuned MLP is a strong baseline |
| [Kadra et al. 2021 — "Well-tuned Simple Nets Excel on Tabular Datasets"](https://arxiv.org/abs/2106.11189) | MLP with regularisation wins on 19/40 datasets |
| [Raschka 2022 — "A Short Chronology of Deep Learning for Tabular Data"](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html) | Survey; MLP consistently competitive across tabular benchmarks |
| [PyTorch Adam docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) | Default `lr=1e-3` |
| [PyTorch Tabular — Optimizer defaults](https://pytorch-tabular.readthedocs.io/en/latest/optimizer/) | Adam at `lr=1e-3` as framework default for tabular regression |

### 1 vs. 2 hidden layers

Gorishniy et al. (NeurIPS 2021) found that a well-tuned MLP is
competitive with ResNet-style architectures and, on some datasets,
with their FT-Transformer — the headline finding being that "MLP-like
models are good baselines for tabular deep learning, and prior work
does not outperform them" at the population level.  The MLP baseline
in that paper uses 1 to 8 layers; on small-to-medium tabular
regression tasks (which this project matches in row count and feature
count) 1–2 layers consistently suffice.

A second hidden layer helps when the target relationship is
compound-nonlinear.  For electricity demand forecasting the dominant
drivers — temperature, hour-of-day, weekday, seasonal trend — are
approximately monotone or piecewise-linear.  A single hidden layer
captures this.  Start with 1; expose `n_layers` as a config parameter
so Stage 11's hyperparameter tuner can evaluate 2.

### Width

With 49 input features and 1 output, choose width from
{64, 128, 256}.  For ~44 k rows, 128 units is the appropriate default:
large enough to represent cross-feature interactions (e.g.,
temperature × hour), small enough to train in seconds on CPU.  The
risk of underfitting at width 64 is lower than the risk of slow
overfitting at width 256 on tabular data of this scale.

### Activation

**ReLU** is the default.  **GELU** has shown consistent small
improvements over ReLU and ELU on tabular benchmarks — one analysis
found "a consistent improvement in accuracy when using GELU compared
to ReLU and ELU" across three benchmark datasets — but the margins
are sub-1 % in most reported cases.  For a pedagogical reference
implementation ReLU is the better default: it is simpler, involves no
approximation, and is universally understood.  Expose
`activation: str` (`"relu"` or `"gelu"`) in config for Stage 11
exploration.

### Dropout

Evidence for dropout benefiting small MLPs on tabular data of this
scale is mixed.  Kadra et al. (2021) tested 13 regularisation
techniques including dropout and found its benefit is highly
dataset-dependent; weight decay and batch normalisation were more
consistently helpful across their 40-dataset benchmark.  On ~44 k rows
a small MLP (128 units, 1 layer) is unlikely to overfit enough to
benefit materially from dropout.

Omit dropout from the Stage 10 default.  Add it as a configurable
option for Stage 11's hyperparameter search.

### Learning rate

Adam with `lr=1e-3` is the near-universal starting point for MLP
regression.  The PyTorch Adam documentation lists `lr=1e-3` as its
default.  PyTorch Tabular, a framework built specifically for tabular
deep learning, also defaults to Adam at `1e-3`.  No domain-specific
reason to deviate exists for this problem scale.

### Recommended shipping default

```
input_dim  : int   = 49        # from Stage 5 feature set
hidden_dim : int   = 128
n_layers   : int   = 1
activation : str   = "relu"
dropout    : float = 0.0
optimizer  : Adam(lr=1e-3)
max_epochs : int   = 100       # subject to early stopping
patience   : int   = 10
```

Expose `hidden_dim`, `n_layers`, and `activation` as Hydra config
parameters.  Keep `dropout` out of Stage 10 config to avoid scope
creep.

---

## R7 — Training-loop ownership

### Canonical sources

| Source | Summary |
|--------|---------|
| [PyTorch Lightning EarlyStopping docs](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html) | `LightningModule` / `Trainer` hook pattern |
| [skorch docs — NeuralNetRegressor](https://skorch.readthedocs.io/) | sklearn-compatible wrapper; callback-based customisation |
| [skorch GitHub](https://github.com/skorch-dev/skorch) | Actively maintained; "do not hide PyTorch" philosophy |
| [Neptune.ai — Model Training Libraries in the PyTorch Ecosystem](https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem) | Overview of Lightning, Ignite, skorch, and hand-rolled patterns |

### Options compared

**PyTorch Lightning (`LightningModule`).**  Splits training logic into
`training_step`, `validation_step`, and `configure_optimizers` hooks.
The shared `Trainer` handles the epoch and batch loops, device
management, logging, and checkpointing.  Advantage: Stage 11+ could
share one `Trainer` across model families.  Disadvantage:
`SimpleMlpModel` must inherit from `LightningModule`, binding it to
Lightning's conventions and its version-sensitive API.  This is
significant added dependency for a single-model stage, and Lightning's
training loop is not transparent enough for a meetup walkthrough.

**skorch (`NeuralNetRegressor`).**  Wraps `nn.Module` in an
sklearn-compatible interface.  Direct integration with the Stage 4
evaluation harness's `fit`/`predict` protocol is appealing.
Disadvantage: skorch controls the training loop, so customising early
stopping or loss-curve logging requires using skorch's callback API,
hiding the loop from pedagogical audiences.  Adds a dependency not
currently in the lock file.  Conflicts with the project's meetup-demo
purpose (DESIGN §1.1) which requires the loop to be visible and
walkable.

**Hand-rolled loop inside `SimpleMlpModel`.**  The model owns its
training loop in a private method.  The public interface (`fit`,
`predict`) matches the Stage 4 protocol.  Early stopping, loss
accumulation, and plot saving are implemented directly in ~40–60
lines of plain PyTorch visible to any reader.  The seam for Stage 11
refactor is: extract the loop body into a standalone `Trainer` class
that accepts a duck-typed training interface.

### Recommendation

Hand-rolled loop for Stage 10.  The training loop is the most
instructive part of the stage for meetup audiences — making it
visible and self-contained is a first-order concern given the
project's pedagogical purpose (DESIGN §1.1, §2.2).  Implement the
loop in a private `_run_training_loop(...)` method so Stage 11 can
extract it without touching the public API.  Do not introduce
Lightning or skorch as Stage 10 dependencies.

The refactor seam: `SimpleMlpModel.fit()` delegates to
`self._run_training_loop(train_loader, val_loader, ...)`.  Stage 11
introduces a `Trainer` class that accepts the model's `training_step`
and replaces that delegation.  The `fit`/`predict` public interface
is unchanged across the refactor.

---

## R8 — GB electricity demand — neural-forecasting precedents

### Canonical sources

| Source | Summary |
|--------|---------|
| [Mohan et al. 2019 — "MLP for short-term load forecasting: global to local approach" (Springer NCA)](https://link.springer.com/article/10.1007/s00521-019-04130-y) | MLP applied to hourly short-term load forecasting; competitive without decomposition |
| [GEFCom2014 MLP submission — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207015001442) | MLP without special preprocessing competitive in the GEFCom2014 load track |
| [Frontiers in Energy Research 2024 — ML algorithms for load demand](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1408119/full) | 2024 benchmark: no single model consistently dominates; linear and MLP comparable |
| [arXiv 2601.02856 — Electricity Price Forecasting: linear models vs. NNs](https://arxiv.org/html/2601.02856) | Linear model outperforms DNN benchmark on some European markets |

### Published precedent for plain MLP on demand data

Mohan et al. (2019) applied a plain MLP to hourly short-term load
forecasting across multiple European zones, demonstrating it
competitive with more complex seasonal decomposition-based
approaches.  The GEFCom2014 probabilistic load track included at
least one MLP submission that used the Levenberg–Marquardt algorithm
with Bayesian regularisation and achieved respectable results
"without any special data preprocessing, such as detrending,
deseasonality or decomposition."  This is the closest published
analogue to Stage 10.

The 2024 Frontiers benchmark evaluated linear regression, MLP, SVR,
random forest, and LSTM on electricity load data with heterogeneous
features; results were highly dataset-dependent with no single model
consistently dominant, and linear models were competitive on
well-behaved hourly data.

### Linear vs. neural performance gap

A 2025 arXiv review of electricity price forecasting (closely related
domain) notes explicitly: "even a linear model such as FLin
demonstrates better forecasting performance than the nonlinear
benchmark DNN" on the German market.  This is consistent with the
intent's expectation — the Stage 10 MLP is likely to match or
marginally improve on SARIMAX rather than dominate it.  The practical
benefit of the MLP stage is architectural (introducing PyTorch into
the pipeline) rather than primarily performance-driven.

### GB-specific context

No published paper specifically benchmarks a vanilla MLP against the
NESO day-ahead demand forecast on GB national demand (`ND`).  The
closest analogues are studies on neighbouring markets (Ireland,
France, Spain) using comparable weather and calendar feature sets, all
showing similar conclusions: MLPs capture nonlinear interactions
(temperature saturation, weekday-hour interactions) that linear models
partially miss, but the marginal gain on clean hourly data is modest.

The project's own Stage 6 evaluation harness, with the SARIMAX
baseline from Stage 7, provides the most directly relevant in-domain
comparison.  The Stage 10 MLP result will be compared against that
baseline, not against published literature.

---

## Open questions for the plan author

1. **Scaler placement decision.**  The recommendation (R2) is to store
   scaler statistics as `register_buffer` inside the `nn.Module`.
   This requires `SimpleMlpModel.fit()` to compute `mean_` and `std_`
   and write them into the module before the first forward pass.
   Does the Stage 10 intent support modifying module state inside
   `fit()`?  If the model must be stateless outside of `state_dict`,
   a thin `NormalisingWrapper` class that stores the scaler externally
   and wraps the raw `nn.Module` is the alternative.

2. **Target normalisation.**  Should `nd_mw` be normalised before
   computing MSE loss, with an inverse-transform in `predict()`?  The
   recommendation says yes for training stability (GW-range targets
   produce large gradients).  Does the intent specify whether the
   registry should store raw or normalised targets?  If normalised,
   the inverse-transform parameters must be in `state_dict`.

3. **Live plot coupling with CI.**  The recommended dual-mode pattern
   (`live_plot: bool = False`) requires the training loop to detect
   or receive a flag.  Is this flag driven from the Hydra config or
   from a method argument on `fit()`?  Clarify the boundary before
   the implementer writes the training-loop signature.

4. **Fold count default.**  The Stage 10 intent does not specify a
   default fold count for rolling-origin evaluation.  SARIMAX
   (Stage 7) used a fixed fold count.  Should Stage 10 default to the
   same value, or is fold count an explicit Stage 10 config
   parameter?

5. **Stage 9 registry interface.**  Stage 9's registry expects a
   `save(path)` / `load(path)` interface on the model.  Confirm
   whether `path` is the model's directory entry in the registry tree
   (into which `model.pt` and `metadata.json` are written) or the
   `.pt` file path directly.  The naming convention matters for
   Stage 9 compatibility.

6. **Activation config type.**  If `activation` is a config string
   (`"relu"` / `"gelu"`), the model constructor maps it to
   `nn.ReLU()` / `nn.GELU()`.  Confirm this is the intended
   config-to-code boundary — YAML strings, not Python callables — in
   line with DESIGN §2.1.4.

7. **Layer count search space.**  The recommendation is to ship with
   `n_layers=1` and expose `n_layers` as a config parameter for
   Stage 11 tuning.  Confirm that `n_layers ∈ {1, 2}` is the intended
   search space, not arbitrary depth (which would require a different
   model constructor pattern).

---

## References

[1] PyTorch Reproducibility documentation — https://docs.pytorch.org/docs/stable/notes/randomness.html
[2] PyTorch GitHub issue #109856 — severe performance regression under deterministic algorithms in torch 2.0 — https://github.com/pytorch/pytorch/issues/109856
[3] Tanel Pärnamaa — "A Bug That Plagues Thousands of Open Source ML Projects" — https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
[4] MachineLearningMastery — "How to Use Data Scaling to Improve Deep Learning Model Stability and Performance" — https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
[5] PyTorch Saving and Loading Models tutorial — https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
[6] PyTorch 2.6 release blog — https://pytorch.org/blog/pytorch2-6/; PyTorch developer mailing list announcement — https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
[7] joblib GitHub issue #1104 — https://github.com/joblib/joblib/issues/1104
[8] PyTorch Module docs — `model.eval()` behaviour — https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
[9] Prechelt, L. (1998). "Early Stopping — But When?" In: *Neural Networks: Tricks of the Trade*. Springer LNCS 1524. — https://link.springer.com/chapter/10.1007/3-540-49430-8_3
[10] livelossplot GitHub — https://github.com/stared/livelossplot
[11] Ash, J.T. and Adams, R.P. (NeurIPS 2020). "On Warm-Starting Neural Network Training." — https://proceedings.neurips.cc/paper/2020/hash/288cd2567953f06e460a33951f55daaf-Abstract.html
[12] Warm-Start or Cold-Start? A comparison of generalisability in gradient-based hyperparameter tuning. *Neural Networks*, 2026. — https://www.sciencedirect.com/science/article/abs/pii/S0893608026001097
[13] Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (NeurIPS 2021). "Revisiting Deep Learning Models for Tabular Data." — https://arxiv.org/abs/2106.11959
[14] Kadra, A. et al. (2021). "Well-tuned Simple Nets Excel on Tabular Datasets." — https://arxiv.org/abs/2106.11189
[15] GELU vs. ReLU comparison — https://towardsai.net/p/l/is-gelu-the-relu-successor
[16] PyTorch Adam docs — https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
[17] PyTorch Tabular — Optimizer and Learning Rate Scheduler defaults — https://pytorch-tabular.readthedocs.io/en/latest/optimizer/
[18] Mohan, N. et al. (2019). "Multilayer perceptron for short-term load forecasting: from global to local approach." *Neural Computing and Applications*. — https://link.springer.com/article/10.1007/s00521-019-04130-y
[19] GEFCom2014 MLP submission. *International Journal of Forecasting*, 2016. — https://www.sciencedirect.com/science/article/abs/pii/S0169207015001442
[20] Frontiers in Energy Research — "Evaluation of electrical load demand forecasting using various machine learning algorithms" (2024) — https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1408119/full
[21] Electricity Price Forecasting: Bridging Linear Models, Neural Networks and Online Learning — arXiv 2601.02856 — https://arxiv.org/html/2601.02856
[22] PyTorch Serialization semantics — https://docs.pytorch.org/docs/stable/notes/serialization.html
[23] Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.) §5.10 — https://otexts.com/fpp3/tscv.html
[24] Matplotlib Backends documentation — https://matplotlib.org/stable/users/explain/figure/backends.html
[25] torch.use_deterministic_algorithms docs — https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
[26] Raschka, S. (2022). "A Short Chronology of Deep Learning for Tabular Data." — https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
[27] skorch documentation — https://skorch.readthedocs.io/
[28] PyTorch Tabular — https://pytorch-tabular.readthedocs.io/en/latest/
