# Models — layer architecture

- **Status:** Provisional — first realised by Stage 4 (seasonal-naive + OLS linear baseline, shipped), extended by Stage 7 (SARIMAX — first model with training state richer than a coefficient vector, shipped), extended by Stage 8 (SciPy parametric — first model with covariance-derived confidence intervals stored in `ModelMetadata.hyperparameters`, shipped). Revisit at Stages 10–11 (simple + complex NNs) and Stage 9 (registry integration, which may introduce a cross-version load contract).
- **Canonical overview:** [`DESIGN.md` §3.2](../../intent/DESIGN.md#32-layer-responsibilities) (models paragraph); [`DESIGN.md` §7.3](../../intent/DESIGN.md#73-the-model-protocol) (the `Model` protocol sketch).
- **Concrete instances:** [Stage 4 retro](../../lld/stages/04-linear-baseline.md) (the `Model` protocol, `NaiveModel`, `LinearModel`); [Stage 7 retro](../../lld/stages/07-sarimax.md) (`SarimaxModel` with Dynamic Harmonic Regression for weekly seasonality); [Stage 8 retro](../../lld/stages/08-scipy-parametric.md) (`ScipyParametricModel` — `curve_fit` on a hand-specified HDD/CDD + Fourier form, with Gaussian CIs derived from the covariance matrix).
- **Related principles:** §2.1.1 (standalone), §2.1.2 (typed narrow interfaces), §2.1.4 (config outside code), §2.1.6 (provenance), §2.1.7 (tests at boundaries).
- **Key ADR:** [`decisions/0003-protocol-for-model-interface.md`](../decisions/0003-protocol-for-model-interface.md) — `typing.Protocol` vs `abc.ABC`.

---

## Why this layer exists

The models layer is the **contract** every forecaster in this repo implements. It is deliberately small: five named methods (`fit`, `predict`, `save`, `load`, `metadata`) and one provenance record (`ModelMetadata`). Every subsequent modelling stage (5, 7, 8, 10, 11) adds new classes conforming to the contract; nothing in the evaluation, registry, or serving layers depends on the concrete model type beyond these five members.

The layer is deliberately stateless at the package level — `bristol_ml.models` does not register models, does not dispatch by name, and does not import heavy dependencies eagerly. A user wanting `LinearModel` imports `from bristol_ml.models.linear import LinearModel`; the package's `__getattr__` lazily re-exports only the protocol, the metadata record, and the joblib IO helpers.

Two concrete model classes land at Stage 4: `NaiveModel` (seasonal lookup; no training loop) and `LinearModel` (`statsmodels.OLS` behind the same protocol). The plan's "implementable in very few lines" acceptance criterion (AC-2) is proved by `NaiveModel`: a class with no `numpy` tricks, just dataframe arithmetic, satisfies the full contract in ~80 statement-lines.

## What lives here, what does not

| Concern | In | Out |
|---------|----|-----|
| The `Model` protocol + `ModelMetadata` provenance | ✓ | — |
| Seasonal-naive and OLS baseline implementations | ✓ | — |
| SARIMAX, SciPy parametric, NN (Stages 7–11) | ✓ | — |
| Joblib-backed `save_joblib` / `load_joblib` helpers | ✓ | — |
| Fold-level evaluation loop | — | evaluation layer (`harness.evaluate`) |
| Splitter / metric definitions | — | evaluation layer |
| Persistence to a named registry slot | — | registry layer (Stage 9) |
| Hyperparameter search | — | deferred (Stage 10+ candidate) |
| Feature engineering, lag computation | — | features layer |
| Serving predictions behind an HTTP endpoint | — | serving layer |

The split is enforced by the `Model` protocol: a class with `fit`/`predict`/`save`/`load`/`metadata` and nothing else can be used by the evaluator, notebooks, and (Stage 9+) the registry. A class that also exposes `fit_fold_k()` or `predict_batch()` is overreaching — the evaluation layer's harness is the single orchestration point.

## Cross-module conventions

Every model module follows the same four-part shape. `NaiveModel` is the minimal template; `LinearModel` is the statsmodels-wrapping template.

### 1. Module shape

- `src/bristol_ml/models/<family>.py` — one module per model family (e.g. `naive.py`, `linear.py`, `sarimax.py`). Each module exports one `class <Family>Model`.
- `src/bristol_ml/models/protocol.py` — the `Model` protocol + `ModelMetadata` re-export. Shared by every family; never family-specific.
- `src/bristol_ml/models/io.py` — joblib-backed save/load helpers. Reused by every family's `save`/`load` methods.
- `conf/model/<family>.yaml` — Hydra group file with `# @package model.<family>` header, carrying the `type` discriminator tag plus family-specific knobs (naive `strategy`, linear `fit_intercept`, …). The `conf/config.yaml` `defaults:` list selects one variant; `model=<family>` at the CLI swaps.
- **No dispatcher.** Hydra's `_target_` pattern is the only factory; `train.py` discriminates via `isinstance(model_cfg, <Family>Config)` on the resolved `AppConfig.model`.

### 2. Public interface — the `Model` protocol

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

- `fit(features, target)` — pure side-effecting: mutate `self` with the fitted state. Must be **re-entrant** — a second call discards the previous fit. The save/load round-trip tests rely on this.
- `predict(features)` — returns a `pd.Series` indexed to `features.index`. Predict-before-fit is undefined behaviour; implementors raise `RuntimeError`.
- `save(path)` / `load(path)` — joblib-backed; `load` is a classmethod returning a fully-reconstituted instance.
- `metadata` — a property (not a method) returning a `ModelMetadata` snapshot of the current fit state. Callable before fit; `metadata.fit_utc is None` signals "unfitted".
- `python -m bristol_ml.models.<family>` — every family's CLI (§2.1.1). Prints the `.summary()` / `.metadata` / family-specific help.

### 3. Provenance — `ModelMetadata`

Every fitted model carries a frozen `ModelMetadata` record with five fields:

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str` (lowercase-snake pattern) | Identifier unique within a stage (`naive-same-hour-last-week`, `linear-ols-weather-only`). |
| `feature_columns` | `tuple[str, ...]` | Ordered, immutable; the exact columns `fit` was trained on. |
| `fit_utc` | `datetime \| None` (tz-aware UTC when set) | `None` before fit; tz-naive values rejected by a validator. |
| `git_sha` | `str \| None` | Short SHA at fit time; `None` outside a Git working tree. |
| `hyperparameters` | `dict[str, Any]` | Free-form bag. The escape hatch — stable fields above, extensibility below. |

`ModelMetadata` is **not** a Hydra config group. It lives in `conf/_schemas.py` because it is a Pydantic model with `frozen=True` / `extra="forbid"`, but it is consumed via `from bristol_ml.models import ModelMetadata` (the `protocol.py` re-export) rather than the config loader. Stage 9 (registry) will likely read these records verbatim as sidecar JSON.

### 4. Serialisation — joblib (with a Stage 12 upgrade path)

Every model's `save`/`load` goes through `bristol_ml.models.save_joblib` / `load_joblib`:

- **Atomic writes.** Tmp file + `os.replace`, mirroring `ingestion._common._atomic_write`. A partial write leaves the previous artefact intact.
- **Creates the parent directory.** Call sites do not need to `mkdir` first.
- **Handles statsmodels `RegressionResultsWrapper` natively.** The joblib pickle protocol is sufficient; no special-casing in `LinearModel.save`.

joblib (like `pickle`) is not a safe deserialiser for untrusted input. Per Stage 9 plan D14, the Stage 12 **serving** layer is the inflection point for `skops.io` adoption — the first stage that loads artefacts from a path not controlled by the training author. Stage 9 (the registry) explicitly documents that it only ever loads artefacts the training author wrote themselves, so Stages 4–11 do not shoulder the `skops.io` audit burden; `models/io.py` carries a one-line docstring note pointing at Stage 12.

## Upgrade seams

Each of these is swappable without touching downstream code. The `Model` protocol + `ModelMetadata` + joblib IO interface is what is load-bearing.

| Swappable | Load-bearing |
|-----------|--------------|
| Concrete model family (OLS → SARIMAX → NN) | The five-member `Model` protocol |
| Hyperparameter shape per family | `ModelMetadata.hyperparameters: dict[str, Any]` |
| Serialisation backend (joblib → `skops.io` at Stage 12 per Stage 9 plan D14) | `save(path)` / `load(path)` return `Path` / an instance |
| Model-selection mechanism (Hydra `_target_` → Stage 9 registry name dispatch — shipped) | `python -m bristol_ml.train model=<family>` CLI surface; `python -m bristol_ml.registry {list,describe}` name-indexed retrieval |
| Feature-column resolution (explicit tuple → assembler-schema default) | `feature_columns: tuple[str, ...] \| None = None` contract |

## Module inventory

| Module | Family | Stage | Status | Notes |
|--------|--------|-------|--------|-------|
| `models/protocol.py` | — (layer contract) | 4 | Shipped | `Model` + `ModelMetadata` re-export. |
| `models/io.py` | — (shared helpers) | 4 | Shipped | `save_joblib` / `load_joblib`. |
| `models/naive.py` | Seasonal-naive baseline | 4 | Shipped | Three strategies; `same_hour_last_week` default (D1). Proves AC-2 in ~80 statement-lines. |
| `models/linear.py` | statsmodels OLS | 4 | Shipped | `fit_intercept=True` adds `sm.add_constant` (D2); `results.summary()` is the notebook demo payoff. |
| `models/sarimax.py` | Seasonal ARIMA-X | 7 | Shipped | Dual-seasonality via Dynamic Harmonic Regression (plan D1): `seasonal_order=(1,1,1,24)` inside SARIMAX + three Fourier pairs at period 168 h for the weekly cycle. Re-fit per fold in the rolling-origin harness (plan D5). joblib sufficient for the `SARIMAXResultsWrapper` round-trip (plan D12). |
| `models/scipy_parametric.py` | SciPy `curve_fit` parametric load | 8 | Shipped | Hand-specified `α + β_heat · HDD + β_cool · CDD + diurnal + weekly Fourier` form with Elexon-convention fixed hinges (plan D1: `T_heat = 15.5 °C`, `T_cool = 22.0 °C`); 13 parameters default (plan D2: `diurnal_harmonics=3` + `weekly_harmonics=2`). Default `loss="linear"` + `method="lm"` so `pcov`-derived 95 % CIs stay Gaussian (plan D3/D6). `p0` derived deterministically from training data inside `fit()` (plan D4). Covariance matrix round-trips via `ModelMetadata.hyperparameters["covariance_matrix"]` as `list[list[float]]` (plan D7). `_parametric_fn`, `_derive_p0`, `_build_param_names` are module-level pure functions — required for joblib pickle round-trip (plan S2). `ScipyParametricConfig.feature_columns` constrains *Fourier* columns, not raw inputs (plan D2 clarification). NFR-4 `pcov`-inf WARNING path stores `float('inf')` rather than silent vacuous CIs. |
| `models/nn/mlp.py` | Simple MLP (first NN family) | 10 | Shipped | `NnMlpModel` — PyTorch MLP behind the `Model` protocol. Full sub-layer contract at [`docs/architecture/layers/models-nn.md`](models-nn.md). |
| `models/nn/_training.py` (planned) | Shared NN training-loop helper | 11 | Planning | Extraction destination when Stage 11's second torch-backed model arrives (plan D10). |
| `models/complex_nn.py` (planned) | Complex / temporal NN | 11 | Planning | Will inherit the Stage 10 sub-layer conventions. |

## Open questions

- **Hyperparameter search composition.** Nested cross-validation inside the rolling-origin harness is the honest shape, but the Stage 4 harness has no hook for it. Whether hyperparameter search becomes a harness feature, a registry concern, or a per-model internal is undesigned. Revisit at Stage 10 when the NN family arrives with a realistic tuning surface.
- **Cross-version load compatibility.** A model saved with version *X* is only guaranteed to load with version *X* today. Stage 9's registry captures the `git_sha` at save time (plan D13) so a future reader can know which source tree the artefact was fitted against, but the registry explicitly does *not* attempt cross-version loads. Whether to pin a semantic package version in `ModelMetadata.hyperparameters` or to require a content hash is undecided — revisit when the first `sklearn` / `statsmodels` major-version bump breaks an existing registered run.
- **`ModelMetadata.hyperparameters` shape discipline.** The bag is deliberately free-form, which makes registry leaderboards hard to filter without convention. A light "top-level keys match the Pydantic `<Family>Config` field names" convention is emerging from Stage 4 (`NaiveModel.metadata.hyperparameters == {"strategy": ..., "target_column": ...}`), but it is not enforced. Stage 9's leaderboard stores the bag verbatim in `run.json`; filtering beyond `target` / `model_type` / `feature_set` is out-of-scope per plan D7 and would benefit from a convention the layer does not yet enforce. Revisit when a downstream stage surfaces a concrete "sort leaderboard by hyperparameter" ask.
- **Per-model CLI parity.** `python -m bristol_ml.models.linear` prints `results.summary()`; `python -m bristol_ml.models.naive` prints help + strategy description; `python -m bristol_ml.models.sarimax` prints the `SarimaxConfig` defaults, the `SARIMAX` constructor docstring link, and a Stage 6 palette notice; `python -m bristol_ml.models.scipy_parametric` prints the `ScipyParametricConfig` defaults, the `curve_fit` docstring pointer, and a Stage 6 palette notice. Whether every family's CLI should converge on a common shape (e.g. "fit on the default feature cache, save to `/tmp`, print metadata") or stay family-specific was *re-deferred at Stage 7* (plan H-3) and *re-deferred again at Stage 8* (plan H-3) — four model families still do not motivate a harmonisation pass ahead of the NN families at Stages 10–11. Revisit at Stage 11+ when >4 model families co-exist, or open a dedicated housekeeping stage.
- **Dispatcher duplication.** Both `_build_model_from_config` in `evaluation/harness.py:475` and the inline `isinstance` ladder in `train.py:_cli_main` have now gained four matching `isinstance` branches across Stages 4 / 7 / 8. Every new model family requires two edits in two files; missing either is a silent exit-code-3 bug (Stage 7 surprise 3). The Stage 7 Phase 3 review filed this as candidate ADR B1; Stage 8 re-deferred the refactor (plan D11 / H-4) rather than expand its scope. ADR filename earmarked: `docs/architecture/decisions/0004-model-dispatcher-consolidation.md`. Revisit when the fifth model family arrives (Stage 11) or open a dedicated housekeeping stage.

## References

- [`DESIGN.md` §2.1](../../intent/DESIGN.md#21-architectural) (principles), [§3.2](../../intent/DESIGN.md#32-layer-responsibilities) (models paragraph), [§7.3](../../intent/DESIGN.md#73-the-model-protocol) (the `Model` sketch), [§8](../../intent/DESIGN.md#8-technology-choices) (statsmodels + joblib).
- [`decisions/0003-protocol-for-model-interface.md`](../decisions/0003-protocol-for-model-interface.md) — Protocol vs ABC.
- [`decisions/0002-filesystem-registry-first.md`](../decisions/0002-filesystem-registry-first.md) — why joblib + sidecar is enough until Stage 9.
- [`docs/intent/04-linear-baseline.md`](../../intent/04-linear-baseline.md) — the Stage 4 intent.
- [`docs/lld/stages/04-linear-baseline.md`](../../lld/stages/04-linear-baseline.md) — retrospective applying this architecture.
- [`src/bristol_ml/models/CLAUDE.md`](../../../src/bristol_ml/models/CLAUDE.md) — module-local guide, protocol semantics, serialisation notes.
- PEP 544 — structural subtyping (`typing.Protocol`).
