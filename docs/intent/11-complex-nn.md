# Stage 11 — Complex neural network

**Status:** Intent (immutable once stage is shipped)
**Depends on:** Stage 10
**Enables:** a cross-model ablation any subsequent stage can draw from

## Purpose

Introduce a model that exploits the temporal structure of the data rather than treating each hour as an independent observation. This is where the neural approach has a genuine shot at beating the linear and classical models, because the temporal dependencies in electricity demand — same-hour-yesterday, same-hour-last-week — are learnable by architectures designed to see sequences. The stage closes the modelling arc the project has been building since Stage 4.

## Scope

In scope:
- A temporal neural model (a temporal convolutional network, a small Transformer, or similar) conforming to the Stage 4 interface.
- The data pipeline changes needed to feed sequences rather than flat feature rows, encapsulated in the model's `fit` / `predict` or in a helper reused from Stage 10.
- A notebook that trains the model, compares it against every prior model on the same held-out period, and produces an ablation table that becomes a reference artefact.

Out of scope:
- Foundation models for time series (TimesFM, Chronos, Lag-Llama). A separate stage if ever pursued.
- Probabilistic variants of the architecture.
- Training-time hyperparameter search.
- Multi-horizon training (single day-ahead horizon only).

## Demo moment

The ablation table. One row per model, columns for the core metrics, same held-out period. A facilitator can look at the table and tell a coherent story about which techniques bought which accuracy, culminating in whatever the best model turns out to be.

## Acceptance criteria

1. The model conforms to the Stage 4 interface.
2. Training uses the harness established in Stage 10.
3. The ablation table covers every model trained so far on the same splits.
4. Save/load through the registry from Stage 9 preserves full weights and the sequence preprocessing state.
5. The notebook's ablation table is reproducible from the registry without re-training anything already registered.

## Points for consideration

- Architecture choice is the main design decision. TCNs are simpler to reason about and train faster than Transformers at this scale; small Transformers are more fashionable and show off a more transferable pattern. Both are reasonable.
- Sequence length. How many hours of history to condition on. Short sequences miss weekly patterns; long sequences slow training. The weekly cycle is 168 hours, which is a natural upper bound.
- Whether to feed weather forecasts as part of the sequence or as side channels. "As side channels" matches day-ahead reality (weather forecasts are known at prediction time); "as part of the sequence" is simpler and may be equivalent for training purposes.
- Training time on a laptop. Small Transformers can still be slow without GPU. If training takes too long to demo, the stage loses pedagogical value.
- Positional encoding for Transformers — learned, sinusoidal, or omitted for short sequences. Each has trade-offs; none is universally right.
- How to present the ablation. A single table is cleanest; a bar chart is more eye-catching; a scatter of predicted vs actual for every model on the same test point is most illuminating but visually busy.
- Whether to run the complex model across multiple rolling-origin folds or a single holdout. Multiple folds is honest but expensive.
- Attention weight visualisation (for Transformers) is sometimes illuminating for temporal data, sometimes not. Worth a quick look; not worth building an interpretability stage around.

## Dependencies

Upstream: Stage 10 (training harness).

Downstream: the ablation artefact produced here becomes a reference point for future stages that want to show "model X now beats the best of these."

## Out of scope, explicitly deferred

- Time-series foundation models.
- Probabilistic architectures.
- Multi-horizon output.
- Knowledge distillation from large to small.
