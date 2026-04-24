# Stage 10 — Simple MLP: codebase map

Artefact type: researcher output. Audience: implementing team.
Baseline SHA: 575ac9c. Date: 2026-04-23.

---

## 1. `Model` protocol surface

**File:** `/workspace/src/bristol_ml/models/protocol.py`
**Layer contract:** `/workspace/docs/architecture/layers/models.md`

```python
@runtime_checkable
class Model(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

Contract annotations (load-bearing):

- `fit` must be **re-entrant** — a second call discards prior state entirely (protocol.py:55–56).
- `predict` must return a `pd.Series` indexed to `features.index` (protocol.py:73). SARIMAX required an explicit re-index fix because `get_forecast().predicted_mean` does not preserve the caller's index (Stage 7 surprise 1).
- `predict` before `fit` must raise `RuntimeError`.
- `save` is atomic (tmp+`os.replace`). Delegates to `save_joblib` in every existing model.
- `load` is a `@classmethod`. Every existing model type-checks the loaded object and raises `TypeError` on mismatch (scipy_parametric.py:499–503, sarimax.py:302–306).
- `metadata` is a `@property`, not a method. Callable before `fit`; `metadata.fit_utc is None` in that state.

How the four existing models implement each member and what generalises:

| Member | NaiveModel / LinearModel | SarimaxModel | ScipyParametricModel | Generalises to MLP? |
|--------|--------------------------|--------------|----------------------|---------------------|
| `fit` | store sklearn/statsmodels result | store `SARIMAXResultsWrapper`, set `_fit_utc` | store `_popt`, `_pcov`, set `_fit_utc` | yes — store `torch.nn.Module` state dict, scaler stats, `_fit_utc` |
| `predict` | pandas arithmetic / `results.predict` | `get_forecast().predicted_mean` + re-index | build design matrix, call `_parametric_fn` | yes — normalise features, run `model.forward`, return `pd.Series` |
| `save` | `save_joblib(self, path)` | `save_joblib(self, path)` | `save_joblib(self, path)` | **does not generalise cleanly** — see §4 |
| `load` | `load_joblib(path)` + type-check | `load_joblib(path)` + type-check | `load_joblib(path)` + type-check | **does not generalise cleanly** — see §4 |
| `metadata` | fresh `ModelMetadata` on each call | fresh `ModelMetadata` with `aic`, `bic` | fresh `ModelMetadata` with `param_values`, `covariance_matrix` | yes — fresh `ModelMetadata` with `train_losses`, `val_losses`, `final_train_loss`, `final_val_loss`, epoch count |

The `metadata` property is **constructed fresh on each call** in every existing model — not cached. Follow the same pattern.

---

## 2. `ModelMetadata` shape

**File:** `/workspace/conf/_schemas.py` lines 615–670

Fields:
- `name: str` — pattern `^[a-z][a-z0-9_.-]*$`. Convention: `"simple-mlp-{hidden_sizes}"` e.g. `"simple-mlp-64-32"`.
- `feature_columns: tuple[str, ...]` — ordered, immutable. Set at fit time.
- `fit_utc: datetime | None` — tz-aware UTC; `None` before fit. Validator rejects naive datetimes (line 649–655).
- `git_sha: str | None` — populated via `_git_sha_or_none()` (see `bristol_ml.registry._git`).
- `hyperparameters: dict[str, Any]` — free-form bag.

Precedents for non-trivial `hyperparameters` content:

- `NaiveModel`: `{"strategy": ..., "target_column": ...}` — simple flat dict.
- `LinearModel`: `{"r2": ..., "n_obs": ..., "feature_columns": [...], "fit_intercept": ...}` — scalars only.
- `SarimaxModel`: `{"aic": ..., "bic": ..., "nobs": ..., "converged": ...}` — scalars from statsmodels fit summary.
- `ScipyParametricModel` (most complex): `{"param_names": [...], "param_values": [...], "param_std_errors": [...], "covariance_matrix": [[...], ...], "target_column": ..., "diurnal_harmonics": ..., "weekly_harmonics": ..., "loss": ...}`. The covariance matrix is stored as `list[list[float]]` via `self._pcov.tolist()` (scipy_parametric.py, plan D7) — JSON-serialisable, registry-friendly.

For Stage 10, the MLP's `hyperparameters` bag should carry:
- `train_losses: list[float]` — per-epoch training loss (length = actual epochs trained).
- `val_losses: list[float]` — per-epoch validation loss.
- `final_train_loss: float`, `final_val_loss: float` — scalars for quick leaderboard inspection without unpacking the full list.
- `architecture: {"hidden_sizes": [...], "activation": str, "dropout": float}` — mirrors the config.
- `normalisation: {"feature_mean": [...], "feature_std": [...]}` — the scaler statistics needed to reproduce predictions (see §4 hazard).
- `n_epochs_trained: int`, `stopped_early: bool`.

The `list[float]` loss-curve precedent is the covariance matrix row from Stage 8; the registry stores `hyperparameters` verbatim in `run.json` via `json.dumps(..., allow_nan=True)` (`registry/__init__.py:167`).

---

## 3. Harness integration

**File:** `/workspace/src/bristol_ml/evaluation/harness.py`

`evaluate(model, df, splitter_cfg, metrics, *, target_column, feature_columns, return_predictions)` call graph per fold (lines 206–260):

```
rolling_origin_split_from_config(len(df), splitter_cfg)
  → (train_idx, test_idx)
model.fit(X_train, y_train)           # ← harness calls this
y_pred = model.predict(X_test)        # ← harness calls this
metric_values = {metric(y_test, y_pred) for metric in metrics}
```

`evaluate_and_keep_final_model` (lines 316–374) delegates to `evaluate(return_predictions=False)` and returns `(metrics_df, model)`. Stage 9 registry calls this to get the final-fold fitted model without re-fitting on the full set. Stage 10 uses the same path.

**No hooks for a loss-curve callback exist in the harness.** The harness calls `model.fit(X_train, y_train)` as a black box. The loss curve accumulation must live **entirely inside `SimpleMlpModel.fit`**. The harness is unchanged.

`return_predictions` flag: single-flag concession from Stage 6 (evaluation/CLAUDE.md, "Harness output — API growth trigger"). **Do not add a second boolean flag.** If Stage 10 needs per-epoch loss curves exposed from the harness, the trigger is a first-class `EvaluationResult` dataclass — that is out of scope for Stage 10; loss curves stay in `metadata.hyperparameters`.

---

## 4. Registry interaction

**File:** `/workspace/src/bristol_ml/registry/__init__.py`

`registry.save(model, metrics_df, *, feature_set, target)` (lines 90–178):

```python
_model_type(model)               # → _dispatch._CLASS_NAME_TO_TYPE[type(model).__name__]
model.save(registry_root / run_id / "artefact" / "model.joblib")
# sidecar written as run.json
```

`registry.load(run_id)` (lines 203–257):

```python
sidecar = json.loads(sidecar_path.read_text())
model_cls = _class_for_type(sidecar["type"])  # → _dispatch._TYPE_TO_CLASS["simple_mlp"]
return model_cls.load(artefact_path)           # → SimpleMlpModel.load(path)
```

Artefact path is always `{run_dir}/artefact/model.joblib`. The registry does not care what `save` writes inside that file — it just passes the `Path` to `model.save`.

**PyTorch serialisation hazard.** `save_joblib(self, path)` pickles the whole instance via `joblib.dump`. A `torch.nn.Module` is pickleable, but the canonical PyTorch pattern is `torch.save(model.state_dict(), path)` rather than pickling the whole module. Both approaches work at Stage 10; the pickle approach is simpler and consistent with every prior model. However, pickled `torch.nn.Module` objects are tied to the PyTorch version and the class definition — the same cross-version caveat every other model carries (models layer doc, "Cross-version load compatibility" open question). Two safe options:

1. **Pickle the whole `SimpleMlpModel` instance** (consistent with SarimaxModel / ScipyParametricModel). Normalisation stats and config round-trip automatically. The risk is that `torch.nn.Module` attribute state must be pickleable — standard `nn.Linear` layers are.
2. **Pickle a dict payload** (consistent with the dict-payload pattern SarimaxModel uses at save: `{"config": ..., "results": ..., ...}`). More explicit; `load` re-hydrates via `__new__` + attribute assignment.

`SarimaxModel.save` pickles `self` directly (line 288: `save_joblib(self, path)`), same as `ScipyParametricModel.save` (line 484). The `SimpleMlpModel` can follow the same pattern if the normalisation stats and the `torch.nn.Module` are both stored as instance attributes.

**Critical:** normalisation statistics (feature mean and std) must be saved and loaded alongside the weights, because `predict` must apply the same normalisation as `fit`. If they are instance attributes (`self._scaler_mean`, `self._scaler_std`), they round-trip with joblib automatically.

---

## 5. Pre-Stage-10 dispatch entries

Every new model family requires edits at **five sites**:

| Site | File | Current state | What changes |
|------|------|---------------|--------------|
| Config discriminated union | `/workspace/conf/_schemas.py` line 612 | `NaiveConfig \| LinearConfig \| SarimaxConfig \| ScipyParametricConfig` | Add `SimpleMlpConfig` to union |
| Hydra group | `/workspace/conf/model/` | 4 YAML files | Add `conf/model/simple_mlp.yaml` with `# @package model` header and `type: simple_mlp` |
| Harness CLI dispatcher | `/workspace/src/bristol_ml/evaluation/harness.py` lines 536–556 | 4 `isinstance` branches | Add `SimpleMlpConfig → SimpleMlpModel` branch |
| Train CLI dispatcher | `/workspace/src/bristol_ml/train.py` lines 241–293 | 4 `elif isinstance` branches | Add `SimpleMlpConfig → SimpleMlpModel` branch; `_target_column` helper at line 484 also needs extending |
| Registry dispatch | `/workspace/src/bristol_ml/registry/_dispatch.py` lines 41–60 | 4 entries in each dict | Add `"simple_mlp": SimpleMlpModel` to `_TYPE_TO_CLASS`; add `"SimpleMlpModel": "simple_mlp"` to `_CLASS_NAME_TO_TYPE` |

The train CLI `else` branch at line 294 is annotated `# pragma: no cover — the discriminated union is exhaustive`. Adding a fifth config type without extending the `isinstance` ladder produces exit-code-3 silently — this is "Stage 7 surprise 3" and "Stage 8 D10" concern repeated. The architecture layer doc flags this as the dispatcher-duplication hazard (models.md lines 122–124); the ADR earmark is `docs/architecture/decisions/0004-model-dispatcher-consolidation.md`. Stage 10 is the trigger stage (fifth model family arrives) but Stage 11 is named as the consolidation owner. The implementer should at minimum extend both dispatchers correctly and flag the consolidation to the lead.

Additionally: `src/bristol_ml/models/__init__.py` lazy re-export at lines 43–94 needs `SimpleMlpModel` and `SimpleMlpConfig` added to `__all__` and the `__getattr__` dispatch dict.

---

## 6. Plots layer

**File:** `/workspace/src/bristol_ml/evaluation/plots.py`

Existing public surface (lines 99–111, `__all__`):
- `residuals_vs_time(residuals, *, display_tz, ax)` — takes `pd.Series` of residuals.
- `predicted_vs_actual(y_true, y_pred, *, ax)` — takes two `pd.Series`.
- `acf_residuals(residuals, *, lags, alpha, reference_lags, ax)` — takes `pd.Series` of residuals.
- `error_heatmap_hour_weekday(residuals, *, display_tz, ax)` — takes `pd.Series` indexed by UTC datetime.
- `forecast_overlay(actual, predictions_by_name, *, display_tz, ax)` — takes `pd.Series` + `Mapping[str, pd.Series]`.
- `forecast_overlay_with_band(actual, point_prediction, per_fold_errors, *, quantiles, ax)` — takes series + errors frame.
- `benchmark_holdout_bar(candidates, neso_forecast, features, metrics, *, holdout_start, holdout_end, ax)` — runs harness internally.
- `apply_plots_config(config: PlotsConfig) -> None`.

All helpers accept `pd.Series` / `pd.DataFrame`, never a `Model` object (AC-3 model-agnosticism contract, evaluation/CLAUDE.md). Every helper accepts `ax: matplotlib.axes.Axes | None` — draws onto supplied axes or mints a new figure sized from `plt.rcParams["figure.figsize"]`.

**No existing loss-curve helper.** Intent AC-3 states "the loss curve is produced by the training loop itself and is available as a plot without additional wiring." The cleanest interpretation: `SimpleMlpModel.fit` accumulates `train_losses` and `val_losses` as instance state; the notebook calls a `loss_curve(train_losses, val_losses)` helper directly. Stage 10 needs to add this helper to `plots.py`.

Shape a new `loss_curve(train_losses, val_losses, *, ax=None) -> Figure` helper:
- Inputs: `Sequence[float]` for each (epoch count implicit as `range(len(train_losses))`).
- Draws two lines (train in `OKABE_ITO[1]`, val in `OKABE_ITO[5]`); x-axis = epoch; y-axis = loss value (log scale optional but useful).
- Adds a vertical line at `argmin(val_losses)` to mark best-epoch (early stopping).
- Returns the figure. Follows the `ax=` composability contract.
- Add to `__all__` and expose it from the module's `_cli_main` help.

This matches the pattern of every other helper: pure function, data in, figure out, no `Model` dependency.

---

## 7. Configuration surface

**File:** `/workspace/conf/_schemas.py`

Convention established by prior models:

```python
class SimpleMlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["simple_mlp"] = "simple_mlp"
    target_column: str = "nd_mw"
    feature_columns: tuple[str, ...] | None = None
    hidden_sizes: tuple[int, ...] = (64, 32)
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)            # early stopping
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    batch_size: int = Field(default=256, ge=1)
    seed: int = Field(default=0, ge=0)
```

Discriminated-union implication: `ModelConfig` at line 612 becomes:
```python
ModelConfig = NaiveConfig | LinearConfig | SarimaxConfig | ScipyParametricConfig | SimpleMlpConfig
```

`AppConfig.model` field stays `ModelConfig | None = Field(default=None, discriminator="type")` — no change to the field itself.

Hydra YAML at `conf/model/simple_mlp.yaml`:
```yaml
# @package model
#
# Stage 10 simple MLP. Select with: python -m bristol_ml.train model=simple_mlp
type: simple_mlp
target_column: nd_mw
feature_columns: null
hidden_sizes: [64, 32]
activation: relu
dropout: 0.0
learning_rate: 0.001
max_epochs: 100
patience: 10
val_fraction: 0.2
batch_size: 256
seed: 0
```

The `# @package model` header is mandatory — every existing model YAML carries it (see `conf/model/scipy_parametric.yaml` line 1).

---

## 8. Existing PyTorch state

**File:** `/workspace/pyproject.toml`

`torch` is **not** a declared dependency. Current runtime dependencies (lines 13–36):
`hydra-core`, `omegaconf`, `pydantic`, `httpx`, `tenacity`, `pyarrow`, `pandas`, `loguru`, `statsmodels`, `scipy`, `joblib`, `matplotlib`, `seaborn`.

`torch` must be added to `[project].dependencies`. Minimum constraint: `torch>=2.2,<3` (PyTorch 2.x is the stable API; 2.2 is the first release with `torch.compile` stable and Python 3.12 support). This is a significant dependency (~800 MB for CPU-only wheel); the CPU-only variant `torch==2.x.x+cpu` is available from the PyTorch index and is better for laptop demos. The implementer should decide whether to use the standard PyPI wheel or the CPU-only index.

Flag for the plan: **torch is a new dep, uv.lock must be regenerated, and `pyproject.toml` must be updated before any import of `torch` in production code**.

---

## 9. Notebook pattern

**Directory:** `/workspace/notebooks/`

Existing notebooks: `01–05`, `07`, `08` (no `06`). Each stage's notebook is a `.ipynb` file generated by a `scripts/_build_notebook_NN.py` generator script and then executed via `jupyter nbconvert --execute`. The notebook is the shipped artefact; the generator makes re-generation reproducible.

Stage 08 notebook structure (14 cells, representative):
1. Markdown — title + math formula + cross-references (plan pointers, dependencies).
2. Code — `REPO_ROOT` bootstrap + all imports + `load_config` + `apply_plots_config`.
3. Markdown — narrative on the model's design rationale.
4. Code — raw scatter / exploratory visualisation.
5. Markdown — functional form explanation.
6. Code — single-fold `%%time` fit + parameter printout (fit-time evidence for AC).
7. Markdown — reading the output.
8. Code — parameter table from `metadata.hyperparameters`.
9. Code — fitted-curve overlay.
10. Code — rolling-origin `evaluate` across all prior-stage models.
11. Code — `forecast_overlay` four-way comparison.
12. Code — parameter stability / diagnostics per fold.
13. Markdown — assumptions appendix.
14. Markdown — closing / forward pointers.

**Stage 10 "live loss curve" moment** — what is new:

The live loss curve requires the notebook to display the curve as it updates during training, not just after training completes. This is new: prior notebooks display static plots on pre-computed results. The pattern is:

- In a Jupyter notebook, `matplotlib` inline rendering does not update live. The canonical pattern is `IPython.display.clear_output(wait=True)` + `plt.show()` inside the training loop, or passing a callback to `fit` that refreshes a cell output.
- The Stage 10 intent says the loss curve is "available as a plot without additional wiring" (AC-3) — not necessarily live-updating during the run. The simplest conforming implementation is: `SimpleMlpModel.fit` accumulates losses internally, and after `fit` returns, the notebook calls `plots.loss_curve(model.metadata.hyperparameters["train_losses"], model.metadata.hyperparameters["val_losses"])`.
- The "live" aspect (updating cell output while `fit` runs) is additive: `SimpleMlpModel.fit` can accept an optional `epoch_callback: Callable[[int, float, float], None] | None = None` parameter (epoch, train_loss, val_loss) that the notebook passes a `clear_output` closure to. This keeps the protocol's `fit(features, target) -> None` signature unchanged (the callback is a `fit`-internal concern, not part of the `Model` protocol).

Closest existing pattern: Stage 08 Cell 6 uses `%%time` and prints intermediate progress to stdout — a degenerate version of the same idea.

---

## 10. Relevant retros

**Stage 7 (`/workspace/docs/lld/stages/07-sarimax.md`) — surprises the MLP inherits:**

- **Surprise 1 (predict re-indexing):** `model.predict(X_test)` must return a `pd.Series` indexed to `features.index`. Any PyTorch output tensor must be wrapped: `pd.Series(tensor.detach().numpy(), index=features.index, name=target_column)`. Add a regression test for this.
- **Surprise 2 (freq on DatetimeIndex):** Not directly relevant to MLP, but a reminder that `features.index.freq` may be `None` after parquet load — the MLP's fit loop should not depend on the index being regular.
- **Surprise 3 (dual dispatcher sites):** Both `harness.py:_build_model_from_config` and `train.py:_cli_main` must be updated. Missing either produces silent exit-code-3.

**Stage 8 (`/workspace/docs/lld/stages/08-scipy-parametric.md`) — surprises the MLP inherits:**

- **S2 (module-level functions for pickle):** The PyTorch `nn.Module` must be pickleable. Standard `nn.Linear` modules are, but any custom forward function or lambda inside the class definition must be at module scope or a named method. Verify `save_joblib(self, path)` round-trips cleanly with a `test_simple_mlp_save_load_round_trip` test before integrating.
- **S1 (explicit dep declaration):** scipy was transitively available but had to be declared. torch will definitely need to be declared in `pyproject.toml` — it is not transitively available from any existing dep.
- **T7 MAE-bound loosening:** Synthetic-data tests may require relaxed bounds if the MLP over-fits on periodic synthetic data. Document any relaxation in the test docstring (Stage 8 precedent).
- **Dual dispatcher update:** Same as Stage 7 surprise 3 — both dispatcher sites must be updated.

---

## Integration points — every file the Stage 10 plan must touch

- `/workspace/conf/_schemas.py` — add `SimpleMlpConfig`; extend `ModelConfig` union.
- `/workspace/conf/model/simple_mlp.yaml` — new Hydra group file.
- `/workspace/src/bristol_ml/models/simple_nn.py` — new module (`SimpleMlpModel`).
- `/workspace/src/bristol_ml/models/__init__.py` — extend `__all__` and `__getattr__`.
- `/workspace/src/bristol_ml/evaluation/harness.py` — extend `_build_model_from_config` and `_target_column`.
- `/workspace/src/bristol_ml/evaluation/plots.py` — add `loss_curve(train_losses, val_losses, *, ax)` helper; extend `__all__`.
- `/workspace/src/bristol_ml/registry/_dispatch.py` — add `"simple_mlp"` to both dicts.
- `/workspace/src/bristol_ml/train.py` — extend `isinstance` ladder and `_target_column` helper.
- `/workspace/pyproject.toml` — add `torch>=2.2,<3` to `[project].dependencies`.
- `/workspace/uv.lock` — regenerate after dep add.
- `/workspace/src/bristol_ml/models/CLAUDE.md` — add `SimpleMlpModel` subsection.
- `/workspace/notebooks/10_simple_nn.ipynb` — new artefact.
- `/workspace/scripts/_build_notebook_10.py` — new generator (Stage 7/8 precedent).
- `/workspace/tests/unit/models/test_simple_nn.py` — new test file (protocol conformance, fit/predict, save/load round-trip, predict-before-fit RuntimeError, re-entrancy, predict re-index).
- `/workspace/tests/unit/evaluation/test_plots.py` — extend for `loss_curve` helper.
- `/workspace/CHANGELOG.md` — `### Added` bullet under `[Unreleased]`.
- `/workspace/docs/lld/stages/10-simple-nn.md` — retrospective (stage hygiene).

---

## Recommended reading order for the implementer

1. `/workspace/docs/intent/10-simple-nn.md` — acceptance criteria and scope.
2. `/workspace/docs/architecture/layers/models.md` — full protocol contract, serialisation notes, open questions (especially "Dispatcher duplication" and "Hyperparameter search composition").
3. `/workspace/src/bristol_ml/models/scipy_parametric.py` — closest structural precedent: model with non-trivial internal state (`_popt`, `_pcov`, normalisation-equivalent stats), module-level helpers for pickle safety, UTC-tz guard, `metadata` property populating `hyperparameters` with a list payload.
4. `/workspace/src/bristol_ml/registry/_dispatch.py` — the two dicts that must grow when the fifth model family lands.
5. `/workspace/docs/lld/stages/08-scipy-parametric.md` — the most recent completed-stage retro; S1 (explicit dep) and S2 (pickle safety) are directly load-bearing for Stage 10.
