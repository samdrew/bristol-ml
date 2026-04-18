# Stage 10 — Simple neural network

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 9
**Enables:** Stage 11 (complex neural network)

## Purpose

Introduce a small multilayer perceptron to the model roster. The model itself is likely not to beat the Stage 5 linear regression by much, but its value is elsewhere: it establishes the training-loop surface, the loss-logging convention, and the GPU/CPU handling that the complex neural model in Stage 11 will inherit. This stage is the scaffold; Stage 11 is where the neural approach earns its place.

## Scope

In scope:
- A small MLP model conforming to the Stage 4 interface, with architecture parameters (layer sizes, activation, dropout) exposed through configuration.
- A training loop with loss logging and validation-set monitoring.
- Early-stopping and checkpointing tied to the registry from Stage 9.
- A notebook that trains the model, plots train-vs-validation loss curves live, and compares predictions against prior models.

Out of scope:
- Temporal architectures — those land in Stage 11.
- Hyperparameter search.
- Distributed or multi-GPU training.
- Any model larger than a small MLP can fit on a laptop.

## Demo moment

The live loss curve. Train and validation loss plotted over epochs while the model trains in the notebook. A facilitator can point at overfitting as it happens — the moment the validation loss bottoms out and starts rising. This is the canonical "watch a neural network learn" moment.

## Acceptance criteria

1. The MLP conforms to the Stage 4 interface, with the training loop hidden behind `fit`.
2. Training is reproducible given a seed (within the constraints of non-deterministic GPU operations).
3. The loss curve is produced by the training loop itself and is available as a plot without additional wiring.
4. Save/load through the registry round-trips cleanly, including the fitted weights.
5. Training on the project's data completes in a reasonable time on a laptop CPU (no GPU requirement).

## Points for consideration

- Input normalisation. Neural networks are sensitive to input scale in a way linear regression is not. How normalisation is done, and where the statistics live, matters for save/load and for serving later.
- How much of the training loop lives inside the model class and how much is shared between neural models. A shared training harness is more DRY but obscures what each model is doing; separate loops per model are clearer but repeat themselves.
- Framework choice. PyTorch was chosen in DESIGN §8; the choice doesn't need revisiting, but the pattern of hiding PyTorch details inside the model class (so it looks like any other Stage 4 model from the outside) does.
- Reproducibility across hardware. Exact bit-for-bit reproducibility is not promised in DESIGN §2.1.5, but "reasonable reproducibility given the same random seed" is achievable if care is taken with RNG placement.
- What counts as overfitting for a small MLP on tabular data with a lot of rows. It may not happen. If it doesn't, the pedagogical value of the live loss curve is reduced; early stopping is still mechanically relevant.
- Initial architecture choice. One or two hidden layers, moderate width, standard activations is enough. Larger architectures rarely pay off on this kind of tabular problem and distract from the scaffolding point.
- Whether to provide a CUDA path at all. If the project is run on a laptop for meetups, CPU-only is fine; if GPU is available the speedup is meaningful, and the abstraction to support both is cheap.
- How to handle the rolling-origin folds efficiently. Re-training the network on every fold is expensive; there may be room to reuse computation across folds.

## Dependencies

Upstream: Stage 9 (registry).

Downstream: Stage 11 (complex neural network) inherits this stage's training harness and loss-logging convention.

## Out of scope, explicitly deferred

- Temporal architectures (Stage 11).
- Ensembling.
- Hyperparameter optimisation.
- Model quantisation or export for deployment.
