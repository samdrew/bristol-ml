# Stage 11 — Complex NN: codebase map

Artefact type: researcher output. Audience: plan author (Phase 2).
Baseline SHA: 6ad2d7a. Date: 2026-04-24.

---

## 1. Stage 10 integration points

### 1.1 `src/bristol_ml/models/nn/mlp.py` — the precedent

**Module docstring seam reference** (line 9–11):
```
- the hand-rolled training loop (plan D10; extraction seam flagged for
  Stage 11);
```
No comment exists *inside* `NnMlpModel.fit` in the source — the seam marker lives only in `CLAUDE.md` (lines 145–150) as a quoted code block. Stage 11 therefore triggers the extraction by adding the new file; the marker is a doc contract, not an in-code flag.

**CLAUDE.md seam block verbatim** (`src/bristol_ml/models/nn/CLAUDE.md` lines 145–150):
```python
# Stage 11 extraction seam: when the second torch-backed model
# (temporal architecture) arrives with a hand-rolled loop of its own,
# extract the body of this method + ``_make_mlp`` to
# ``src/bristol_ml/models/nn/_training.py`` under a shared helper.
```
Extraction is conditional: "when Stage 11's loop diverges from Stage 10's by more than 'add one optimiser kwarg'" (CLAUDE.md line 152). Do **not** ship a `BaseTorchModel` ABC (scope-diff X7 cut, CLAUDE.md line 153).

**Training loop body** — inlined inside `NnMlpModel.fit`, lines 529–628 of `mlp.py`. Steps (referenced by line):
- L529–532: tensor creation and device movement.
- L535–536: normalised target for loss (MSE on z-scored target).
- L539–549: seeded `torch.Generator` + `TensorDataset` + `DataLoader` (`num_workers=0`, `shuffle=True`, `drop_last=False`).
- L551–556: `optim.Adam` + `nn.MSELoss`.
- L558–563: best-val-loss tracking with `detach().clone().cpu()` best state dict.
- L569–616: epoch loop — `module.train()` → minibatch forward/backward/step → `module.eval()` → val loss → `loss_history` append → `epoch_callback` → patience counter → early-stop break.
- L618–619: best-epoch weight restore via `load_state_dict(strict=True)`.

**Module-level helpers**:

| Function | Signature | Reuse in Stage 11 |
|---|---|---|
| `_select_device` | `(preference: str) -> torch.device` (L72) | Reused as-is. Reads `_ALLOWED_DEVICES` (L62). |
| `_seed_four_streams` | `(seed: int, device: torch.device) -> None` (L125) | Reused as-is. Seeds `random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`; sets cuDNN flags on CUDA. |
| `_make_mlp` | `(input_dim: int, config: NnMlpConfig) -> nn.Module` (L165) | Adapted — Stage 11 needs an analogous `_make_temporal(input_dim, seq_len, config)` returning the temporal module. The MLP factory itself is not reused for a sequence model. |
| `_build_nn_module_class` | `() -> type[nn.Module]` (L225) | Pattern reused; Stage 11 needs its own `_build_temporal_module_class()` following the same lazy-construction + sys.modules install recipe. |
| `_NnMlpModule` | Constructor shim routing to `_build_nn_module_class()` (L322) | Pattern reused under a different name. |
| `_build_metadata_name` | `(config: NnMlpConfig) -> str` (L848) | Adapted — Stage 11 writes its own `_build_temporal_metadata_name(config)` with its own `^[a-z][a-z0-9_.-]*$`-compliant format. |

### 1.2 `src/bristol_ml/models/nn/__init__.py` and `__main__.py`

`__init__.py` (47 lines): uses `__getattr__` lazy re-export to keep `torch` off cheap import paths. Stage 11 adds a second class (`NnTemporalModel` or equivalent) to `__all__` and a second branch in `__getattr__`. Pattern (line 34–46):
```python
def __getattr__(name: str) -> object:
    if name == "NnMlpModel":
        from bristol_ml.models.nn.mlp import NnMlpModel
        return NnMlpModel
    raise AttributeError(...)
```
Add an `elif name == "NnTemporalModel":` branch importing from the new module.

`__main__.py` (15 lines): delegates to `bristol_ml.models.nn.mlp._cli_main`. Stage 11 either leaves this as-is (pointing at the mlp CLI, noting it is an alias for the whole sub-package) or makes it dispatch to both. The simpler path is a new `python -m bristol_ml.models.nn.temporal` entry point mirroring the mlp pattern, with `__main__.py` unchanged.

### 1.3 `src/bristol_ml/models/nn/CLAUDE.md` — the five PyTorch gotchas

Each gotcha heading and Stage 11 disposition:

1. **"`_NnMlpModuleImpl` is a lazy-built class, *installed* onto the module"** (CLAUDE.md line 52). Stage 11 **inherits unchanged** — must perform the same three-step routine: (i) `__module__` patch, (ii) `__qualname__` patch, (iii) `sys.modules[__name__].ClassName = cls` install. The pickleable-class regression test must have a Stage 11 counterpart.

2. **"Torch is imported lazily, not at module load"** (CLAUDE.md line 72). Stage 11 **inherits unchanged** — all `import torch` calls inside function bodies, `TYPE_CHECKING` guard on class-level hints.

3. **"Scaler buffers are registered at module construction, not fit time"** (CLAUDE.md line 83). Stage 11 **inherits unchanged** for the feature normalisation buffers. Adaptation needed: if a temporal model adds sequence-level statistics (e.g. a learnt embedding offset) as buffers, the same `register_buffer` + placeholder → overwrite-at-fit pattern applies.

4. **"`torch.load(..., weights_only=True, map_location=\"cpu\")` at load time"** (CLAUDE.md line 95). Stage 11 **inherits unchanged** — pass both flags explicitly.

5. **"Single-joblib envelope; `state_dict_bytes` inside"** (CLAUDE.md line 107). Stage 11 **inherits unchanged** — one `model.joblib` at the registry-supplied path; `state_dict` serialised to `BytesIO` bytes inside the envelope dict. The structural guard test must have a Stage 11 counterpart.

### 1.4 `conf/_schemas.py` — `NnMlpConfig` definition

Full definition, lines 542–644:
```python
class NnMlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["nn_mlp"] = "nn_mlp"
    target_column: str = "nd_mw"
    feature_columns: tuple[str, ...] | None = None
    hidden_sizes: list[int] = Field(default_factory=lambda: [128])
    activation: Literal["relu", "tanh", "gelu"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)
    seed: int | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
```

Stage 11's sibling config (`NnTemporalConfig` or similar) must carry:
- `type: Literal["nn_temporal"] = "nn_temporal"` (or the chosen slug).
- All optimisation fields from `NnMlpConfig` (`learning_rate`, `weight_decay`, `batch_size`, `max_epochs`, `patience`, `seed`, `device`, `dropout`) — same validators, same defaults unless Stage 11 research says otherwise.
- A `seq_len: int` field (the lookback window) — architecture-specific addition.
- Architecture-specific fields (e.g. `hidden_size`, `num_layers` for RNN; `num_heads`, `d_model` for transformer) — Phase 2 picks the form.
- Same `extra="forbid", frozen=True` policy.

`ModelConfig` discriminated union (line 717):
```python
ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig | NnMlpConfig
```
Stage 11 appends `| NnTemporalConfig`.

### 1.5 `conf/model/nn_mlp.yaml` — the Hydra group pattern

```yaml
# @package model
type: nn_mlp
target_column: nd_mw
feature_columns: null
hidden_sizes: [128]
activation: relu
dropout: 0.0
learning_rate: 1.0e-3
weight_decay: 0.0
batch_size: 32
max_epochs: 100
patience: 10
seed: null
device: auto
```

Stage 11 creates `conf/model/nn_temporal.yaml` with `# @package model` header and `type: nn_temporal`. All optimisation defaults mirror `nn_mlp.yaml` unless Stage 11 research changes them.

---

## 2. Harness + evaluation path

### 2.1 `src/bristol_ml/evaluation/harness.py` — `evaluate()` and `evaluate_and_keep_final_model()`

```python
def evaluate(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
    return_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: ...

def evaluate_and_keep_final_model(
    model: Model,
    df: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Model]: ...
```

The harness is **pure protocol** — it calls `model.fit(X_train, y_train)` and `model.predict(X_test)` with no model-specific knowledge (line 214–215). Stage 11 plugs in without touching the harness. The only harness code that needs changing for Stage 11 is `_build_model_from_config` (the harness CLI's internal factory, lines 536–556) — add an `isinstance(model_cfg, NnTemporalConfig)` branch there. That function is not on the hot path for the train CLI.

The cold-start-per-fold contract (plan D8) is enforced by the harness calling `fit` on each fold without re-instantiating the object — `NnTemporalModel.fit` must discard all prior state at entry exactly as `NnMlpModel.fit` does (lines 565–567 of `mlp.py`).

### 2.2 `src/bristol_ml/evaluation/plots.py` — `loss_curve`

```python
def loss_curve(
    history: Sequence[Mapping[str, float]],
    *,
    title: str = "Training vs validation loss",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...
```

`history` is `list[dict[str, float]]` with required keys `{"epoch", "train_loss", "val_loss"}` (lines 1020–1027). Stage 11 reuses this unchanged — `NnTemporalModel.loss_history_` must carry the same dict structure. The helper is model-agnostic (AC-3); no `NnMlpModel` import inside `plots.py`.

### 2.3 `src/bristol_ml/evaluation/benchmarks.py` — `compare_on_holdout`

```python
def compare_on_holdout(
    models: Mapping[str, Model],
    df: pd.DataFrame,
    neso_forecast: pd.DataFrame,
    splitter_cfg: SplitterConfig,
    metrics: Sequence[MetricFn],
    *,
    aggregation: Literal["mean", "first"] = "mean",
    target_column: str = "nd_mw",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame: ...
```

Stage 11's ablation table is a pure registry-read + `compare_on_holdout` call — the function is model-agnostic, accepts any `Mapping[str, Model]`. Pass a dict of loaded registry models (`registry.load(run_id)`) keyed by name. No code change needed here.

### 2.4 Ablation over registered runs — AC-5 gap assessment

AC-5 ("reproducible from the registry without re-training anything already registered") is structurally supportable: `registry.load(run_id)` returns a fitted model; calling `model.predict(X_holdout)` computes the holdout prediction without re-fitting. However, `compare_on_holdout` internally calls `evaluate()` which calls `model.fit` per fold — it does **re-train**. To score a previously-registered model without re-training, the caller must bypass `compare_on_holdout` and call `model.predict(holdout_features)` directly, then compute metrics manually. This is a gap between AC-5's intent and the current `compare_on_holdout` API. Surface this to the planner; the Stage 11 plan should either (a) document that "ablation from registry" means re-training on each fold (which is reproducible if the seed is recorded), or (b) add a `compare_predictions_only` helper that skips the fit loop.

---

## 3. Registry dispatch

### 3.1 `src/bristol_ml/registry/_dispatch.py` — current type↔class dicts

```python
_TYPE_TO_CLASS: dict[str, type] = {
    "naive": NaiveModel,
    "linear": LinearModel,
    "sarimax": SarimaxModel,
    "scipy_parametric": ScipyParametricModel,
    "nn_mlp": NnMlpModel,
}

_CLASS_NAME_TO_TYPE: dict[str, str] = {
    "NaiveModel": "naive",
    "LinearModel": "linear",
    "_NamedLinearModel": "linear",
    "SarimaxModel": "sarimax",
    "ScipyParametricModel": "scipy_parametric",
    "NnMlpModel": "nn_mlp",
}
```

Stage 11 adds one entry to each dict:
- `_TYPE_TO_CLASS["nn_temporal"] = NnTemporalModel` (choose slug to match `type` literal).
- `_CLASS_NAME_TO_TYPE["NnTemporalModel"] = "nn_temporal"`.

The module docstring (line 15–17) warns against adding a third dispatcher site alongside `_dispatch.py` and `train.py`; Stage 11 is the sixth model family and the consolidation trigger is named as a "Housekeeping carry-over H-4". Stage 11 should not add a third site.

### 3.2 `src/bristol_ml/train.py` — dispatch ladder for `isinstance`

New branch to insert after the `NnMlpConfig` branch (currently line 301–311):
```python
elif isinstance(model_cfg, NnTemporalConfig):
    primary = NnTemporalModel(model_cfg)
    primary_kind = "nn_temporal"
```
Also extend `_target_column()` (line 500–515) to include `NnTemporalConfig` in the `isinstance` tuple. The `else: # pragma: no cover` branch (line 312) is the safety net — any omission at this site produces exit-code-3.

The `NnMlpConfig` branch at line 301–311 deliberately does **not** do a `model_copy` feature-column promotion (comment at line 302–309 explains why: `NnMlpModel.fit` falls back to `tuple(features.columns)` when `feature_columns is None`). Stage 11 should follow the same no-promotion pattern unless the temporal model has a reason to differ.

---

## 4. Feature table shape

**Emitter:** `src/bristol_ml/features/assembler.py`, function `build()`. Output persisted via `assembler.assemble()` (weather-only, 10 cols) or `assembler.assemble_calendar()` (weather+calendar, 55 cols).

**Index semantics:** `timestamp_utc` column promoted to index via `df.set_index("timestamp_utc")` in `train.py` (line 244). The index is a `pd.DatetimeIndex` with `tz=UTC`, strictly monotonically ascending, **hourly cadence but `freq` attribute is `None`** after parquet round-trip (Stage 7 SARIMAX surprise 2 — do not rely on `df.index.freq`). The feature table carries no NaN values (assembler invariant). Rows with demand or weather gaps are dropped during `build()`.

**Columns (weather_only):** `nd_mw` (int32), `tsd_mw` (int32), `temperature_2m` (float32), `dew_point_2m` (float32), `wind_speed_10m` (float32), `cloud_cover` (float32), `shortwave_radiation` (float32), `neso_retrieved_at_utc`, `weather_retrieved_at_utc`. Active feature columns for a model: `WEATHER_VARIABLE_COLUMNS` (5 columns, float32).

**Columns (weather_calendar):** 55 total — the 10 weather-only columns as an exact prefix, then 44 int8 calendar one-hots, then `holidays_retrieved_at_utc`.

**`target_column` resolution:** `NnMlpConfig.target_column = "nd_mw"` (default). The harness slices `df[target_column]` and `df[feature_columns]` before calling `fit`.

**Sequence window memory cost (rough order of magnitude):**
- 5 years of hourly data ≈ 43,800 rows; lookback `W=168` h → 43,633 windows.
- Materialising a float32 3D tensor `(43633, 168, 5)` = **147 MB** (weather-only).
- Materialising `(43633, 168, 49)` = **1.4 GB** (weather+calendar, 44 calendar cols cast to float32).
- Lazy windowing via a custom `torch.utils.data.Dataset` (yield one window at a time) avoids materialising the full tensor; this is the required pattern for Stage 11.

---

## 5. Notebook pattern

**Builder:** `scripts/_build_notebook_10.py` — a plain Python script that constructs a `dict` matching the `nbformat 4.5` schema (cells as `md()` / `code()` helper calls), serialises with `json.dumps(notebook, indent=1)`, and writes to `notebooks/10-simple-nn.ipynb`. Three-step regeneration flow (script docstring):
```
uv run python scripts/_build_notebook_10.py
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/10-simple-nn.ipynb
uv run ruff format notebooks/10-simple-nn.ipynb
```

The script is idempotent; the notebook `.ipynb` is the shipped artefact; cell source lives in the builder as readable Python strings under version control.

Stage 11 must create `scripts/_build_notebook_11.py` mirroring this pattern exactly. Output to `notebooks/11-complex-nn.ipynb`.

---

## 6. Pre-existing risks — things Stage 11 must not regress

**Cheap-CLI lazy-torch-import contract.** `python -m bristol_ml.models.nn --help` and `python -m bristol_ml.models.nn.mlp --help` run without importing `torch` (enforced by the lazy-import pattern). Stage 11's module must follow the same pattern — all `import torch` inside function bodies, no top-level `import torch` in `temporal.py` or in the `__init__.py` additions.

**Single-joblib envelope structural guard.** `test_nn_mlp_save_writes_single_joblib_file_at_given_path` asserts exactly one file is written at the registry-supplied path. Stage 11's `save` must satisfy the same constraint; a sibling `model.pt` or any second file at the artefact path breaks the registry's `_atomic_write_run` path contract.

**Four-stream seeding semantics.** `_seed_four_streams` is called with `int(effective_seed)` (line 480 of `mlp.py`). If Stage 11 reuses `_seed_four_streams` from `mlp.py`, it must import it explicitly from `bristol_ml.models.nn.mlp`. If it is extracted to `_training.py`, both `NnMlpModel` and `NnTemporalModel` import from there. Do not copy-paste the function body into `temporal.py` — the four-stream recipe has a regression test and duplicating it creates a maintenance hazard.

**`loss_history_` shape consumed by `loss_curve`.** `plots.loss_curve` validates that each dict carries keys `{"epoch", "train_loss", "val_loss"}` (lines 1020–1027 of `plots.py`). Stage 11's `NnTemporalModel.loss_history_` must emit dicts with those exact keys. Any additional keys (e.g. `"grad_norm"`) are silently ignored by `loss_curve`; missing keys raise `ValueError`.

**`weights_only=True` on torch.load.** PyTorch 2.6+ defaults to `weights_only=True`; setting it explicitly is the declared safety rail (CLAUDE.md gotcha 4). Do not omit it in `NnTemporalModel.load` even if it appears redundant.

**Harness cold-start / re-entrancy.** `NnMlpModel.fit` resets `loss_history_`, `_best_epoch`, `_module`, `_feature_columns` at the top of each call (lines 565–576). `NnTemporalModel.fit` must do the same. The harness relies on re-entrancy; a model that accumulates state across folds will corrupt the per-fold metric table silently.

**Registry four-verb cap.** `test_registry_public_surface_does_not_exceed_four_callables` enforces `len(__all__) == 4`. Stage 11 adds nothing to the registry public surface.

**Dispatcher-duplication hazard (H-4 carry-over).** Both `train.py` (`isinstance` ladder, line 248–317) and `evaluation/harness.py` (`_build_model_from_config`, lines 536–556) must be updated. Missing either is silent at import time but produces exit-code-3 or `None`-model returns at runtime. The ADR earmark is `docs/architecture/decisions/0004-model-dispatcher-consolidation.md`; Stage 11 is the sixth family and is the named consolidation trigger.

---

## Relevant files

- `/workspace/src/bristol_ml/models/nn/mlp.py` — Stage 10 MLP, full precedent for Stage 11.
- `/workspace/src/bristol_ml/models/nn/__init__.py` — lazy re-export; Stage 11 adds a branch.
- `/workspace/src/bristol_ml/models/nn/__main__.py` — CLI alias; no change needed or new peer entry point.
- `/workspace/src/bristol_ml/models/nn/CLAUDE.md` — five PyTorch gotchas + extraction seam spec.
- `/workspace/conf/_schemas.py` — `NnMlpConfig` (lines 542–644), `ModelConfig` union (line 717); Stage 11 adds `NnTemporalConfig`.
- `/workspace/conf/model/nn_mlp.yaml` — Hydra group pattern to mirror.
- `/workspace/src/bristol_ml/evaluation/harness.py` — `evaluate()` (line 114), `evaluate_and_keep_final_model()` (line 316), `_build_model_from_config()` (line 536); Stage 11 adds one branch to the last.
- `/workspace/src/bristol_ml/evaluation/plots.py` — `loss_curve()` (line 944); consumed unchanged.
- `/workspace/src/bristol_ml/evaluation/benchmarks.py` — `compare_on_holdout()` (line 167); consumed unchanged; ablation gap documented above.
- `/workspace/src/bristol_ml/registry/_dispatch.py` — `_TYPE_TO_CLASS` / `_CLASS_NAME_TO_TYPE` (lines 42–63); Stage 11 adds one entry each.
- `/workspace/src/bristol_ml/train.py` — `isinstance` dispatch ladder (lines 248–317), `_target_column()` (line 500); Stage 11 adds one branch each.
- `/workspace/src/bristol_ml/features/assembler.py` — `WEATHER_VARIABLE_COLUMNS`, `OUTPUT_SCHEMA`, `build()`.
- `/workspace/scripts/_build_notebook_10.py` — builder pattern to mirror for `_build_notebook_11.py`.
- `/workspace/notebooks/10-simple-nn.ipynb` — Stage 10 notebook; Stage 11 creates `11-complex-nn.ipynb`.
