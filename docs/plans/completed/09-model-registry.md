# Plan — Stage 9: Model registry

**Status:** `approved` — reviewed and approved 2026-04-23 with two amendments (D3 drops seconds; D10 ships a test-only MLflow PyFunc adapter). Phase 2 in progress.
**Intent:** [`docs/intent/09-model-registry.md`](../../intent/09-model-registry.md)
**Upstream stages shipped:** Stages 0–8 (foundation, ingestion, features, four Stage-4/7/8 models, enhanced evaluation).
**Downstream consumers:** Stage 10 (simple NN — saves through registry from day one), Stage 11 (complex NN — same), Stage 12 (serving — loads by name), Stage 17 (price models), Stage 18 (drift monitoring).
**Baseline SHA:** `575ac9c` (tip of `main` after Stage 8 merge via PR #5 and `@minimalist` process change via PR #6).

**Discovery artefacts produced in Phase 1:**
- Requirements — [`docs/lld/research/09-model-registry-requirements.md`](../../lld/research/09-model-registry-requirements.md)
- Codebase map — [`docs/lld/research/09-model-registry-codebase.md`](../../lld/research/09-model-registry-codebase.md)
- Domain research — [`docs/lld/research/09-model-registry-domain.md`](../../lld/research/09-model-registry-domain.md)
- Scope Diff — [`docs/lld/research/09-model-registry-scope-diff.md`](../../lld/research/09-model-registry-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition)

**Pedagogical weight.** Intent §Demo moment frames Stage 9 as the "leaderboard moment" — a single CLI command prints every registered model with its metrics, feature set, and training timestamp, turning "which model is best" into one command rather than bespoke code. The central design question of this stage is **how small the registry's public surface can be while still enabling the leaderboard, the retrofit, and the Stage-12 load-by-name handoff**. AC-1 ("save, load, list, maybe describe") is the load-bearing constraint; every decision below either directly operationalises it, operationalises another named AC, or is cut.

---

## 1. Decisions for the human (resolve before Phase 2)

Nineteen decision points plus four housekeeping carry-overs. Defaults below lean on the three research artefacts' recommendations and honour the simplicity bias in `DESIGN.md §2.2.4`. The Evidence column cites the research that *resolved* each decision — domain research findings, codebase precedents and hazards, requirements user-stories. Acceptance criteria are cited where the intent supplies a hard constraint the decision operationalises; the intent's open questions (`§Points for consideration`) are not cited as evidence because they *pose* the decision rather than answering it.

| # | Decision | Proposed default | Simplicity rationale | Evidence |
|---|---|---|---|---|
| **D1** | On-disk layout | **Flat: `data/registry/{run_id}/` with an `artefact/` subdirectory (holding `model.joblib`) and a `run.json` sidecar next to it.** Cross-run listing = one `os.listdir(registry_root)` + read each sidecar. | Flat is the simplest layout that satisfies both AC-4 (instantaneous listing — one directory scan) and AC-5 (hand-inspectable — one sidecar per run is obvious). Nested (`target/model/date/`) would be easier to browse per-target but forces multi-level traversal for any cross-run query. MLflow's nested file store has the directory-explosion pathology the research called out. A subdirectory per run also prevents joblib's occasional multi-file output from leaking across runs. | Domain research §R1 (canonical registry layouts — MLflow, DVC, hand-rolled); §R4 (query performance; one-file-per-key is the root of MLflow's slowness); §R10 anti-pattern 5 (joblib multi-file output). Codebase map §7 (existing `data/` conventions and `.gitignore` rule). |
| **D2** | Unit of registration | **One registry entry per `registry.save()` invocation.** Last-write-wins: same-second `{model_name}_{timestamp}` collisions overwrite. No de-duplication by feature-set or hyperparameter identity. | Inventing identity semantics adds scope without a requirement. Every industry precedent treats each training invocation as its own entry. Last-write-wins is cheap to describe and cheap to test. | Domain research §R6 (run conventions in MLflow, W&B, DVC — UUID-per-invocation is dominant). Intent §Out of scope explicitly defers "Artefact versioning beyond 'last write wins'". |
| **D3** | Run ID format | **`{model_name}_{YYYYMMDDTHHMM}`** (minute-precision ISO-8601, UTC). Example: `linear-ols-weather-only_20260423T1430`. Two runs in the same minute collide under D2 last-write-wins. | Raw UUIDs fail the demo-moment typeability test. Appending a UUID suffix adds characters without reducing collision risk at single-author scale. Minute precision is shorter still than second precision and is as easy to say aloud during a live demo as it is to tab-complete. At single-author pace (one training run per tens of minutes) minute-collision is vanishingly rare; last-write-wins is explicit intent (D2). Timestamp is sortable lexicographically so default list ordering is free. | Domain research §R6 (run-ID conventions; MLflow UUID, W&B short string, BentoML `{name}:{datetime}`). Intent §Demo moment supplies the hard constraint that the ID must be human-typeable at the CLI — human Ctrl+G amendment 2026-04-23 further simplifies to minute precision for oral-demo typeability. |
| **D4** | Sidecar metadata format | **Single `run.json` file per run, UTF-8, `json.dumps(..., indent=2, allow_nan=True, ensure_ascii=False)`.** No YAML, no TOML, no SQLite index. | JSON is parseable without a new dependency; `allow_nan=True` is the one-line accommodation for `ScipyParametricModel`'s covariance matrix which can contain `float('inf')`. One file per run sidesteps MLflow's one-file-per-key directory explosion. `indent=2` makes `cat run.json` readable. | Domain research §R3 (metadata sidecar formats; JSON dominates machine-readable, YAML would add `pyyaml`). Codebase map §1 (`ScipyParametricModel` stores `covariance_matrix` as `list[list[float]]` with possible `inf` entries). Acceptance criterion AC-5 (hand-inspectable). |
| **D5** | Atomic save | **Temp-dir-then-rename.** Write artefact + sidecar into `data/registry/.tmp_{short_uuid}/` then `os.rename(...)` to the final `{run_id}/` directory. | Mirrors the canonical atomic-write idiom the codebase already uses at two sites — four lines of Python, same pattern the reader has seen twice. Eliminates the MLflow "dead-run" pathology (sidecar without artefact or vice versa) at negligible cost. | Codebase map §7 (existing `_atomic_write` at `ingestion/_common.py:201-210`; matching joblib version at `models/io.py:50-53`). Domain research §R7 (POSIX atomic write pattern); §R10 anti-patterns 2 and 3 (metadata drift and dead runs). |
| **D6** | Feature-table hash | **Defer.** Sidecar records the feature-set *name* (already present in `ModelMetadata.feature_columns`) and the git SHA; no content hash of the Parquet. | Git SHA + feature-set name pins both the feature-engineering code and the selected columns; at single-author scale a content hash adds a step to every save for a reproducibility gain that no AC requires. Promote later if Stage 17's price-model provenance needs it. | Domain research §R2 (minimum reproducible metadata — MLflow autolog precedents, Model Cards, Sugimura & Hartley); recommendation 4 explicitly labels the hash as premature at this scale. |
| **D7** | `list` filter arguments | **Three optional `str` filters, each an exact match against the sidecar field of the same name: `target`, `model_type`, `feature_set`.** No wildcards, no date range at Stage 9. | Exact-match keeps the `list_runs` signature small and the test surface bounded. Date-range filtering requires a parse + comparison step that no AC demands. | Codebase map §8 (downstream consumers' minimal common interface; Stage 17 needs `target` filtering, Stage 12 needs `list` for name discovery, no downstream asks for date range). Requirements user-story US-3 (analyst filtering "by target, feature set, or model type"). |
| **D8** | Leaderboard default sort | **`sort_by="mae"` ascending.** `list_runs` accepts a `sort_by: str \| None` argument matching any metric key present in every returned sidecar; records missing the key are placed last. Descending via `ascending: bool = True` default. | Making the defensible default the built-in default keeps `registry list` a one-word CLI call. Accepting any metric key keeps the interface sortable-by-anything without a metric-registry enum. | Domain research §R9 (interface-surface guidance; MLflow `search_runs`, BentoML, ZenML — all default-sorted on a primary metric). Requirements user-story US-1 (facilitator wants "ranked by MAE" out of the box). |
| **D9** | Registry–model interaction | **Registry calls `model.save(path)` on the artefact path, then writes `run.json` alongside. `Model` protocol is unchanged.** `load()` calls `Model.load(path)` for the appropriate concrete class, dispatched via the sidecar's `type` field. | The `Model` protocol stays at five methods; AC-2 ("without code changes to the model itself") is satisfied by construction. The load dispatcher is a local dict literal — no third site added to the existing two-dispatcher footprint. | Acceptance criterion AC-2 (no model-code changes). Codebase map §1 (existing `save`/`load`/`metadata` protocol signatures); §4 (two existing dispatcher sites; Stage 7/8 retro warning against a third). |
| **D10** | MLflow graduation path | **Document the adapter contract in the layer doc AND ship a test-only `mlflow.pyfunc` adapter with a round-trip integration test. No MLflow runtime dependency; `mlflow` is added to the `dev` dependency group only.** The "Graduation to MLflow" layer-doc subsection names: (i) the four-verb interface is the contract, (ii) mechanical migration requires a thin `mlflow.pyfunc.PythonModel` subclass wrapping `registry.load()`, (iii) run IDs are not preserved across `mlflow-export-import`. The adapter class and packaging helper live under `tests/integration/mlflow_adapter.py` (test support code, not production surface); an integration test asserts `mlflow.pyfunc.save_model` → `mlflow.pyfunc.load_model` → `predict` round-trips a registered run with numerically-identical output. | Making the "mechanical migration" claim falsifiable — a passing test is worth more than a paragraph of prose, and the test catches adapter-contract drift if MLflow changes its PyFunc API. Test-only housing preserves AC-1's four-verb public surface and the zero-runtime-dep stance; `mlflow.pyfunc` is the MLflow flavour for non-sklearn protocols and wraps arbitrary `predict()` surfaces in a small subclass. | Acceptance criterion AC-1 (public surface stays at four verbs — adapter lives in tests/). Domain research §R5 (MLflow migration tooling; `mlflow migrate-filestore` is filesystem→SQLite only; `mlflow-export-import` does not preserve run IDs; bare joblib cannot be loaded through `mlflow.sklearn.load_model` without an envelope; `mlflow.pyfunc` is the graduation-safe wrapper for custom protocols). Human Ctrl+G amendment 2026-04-23 upgrades "document the adapter" to "document and test the adapter". |
| **D11** | Registry directory VCS policy | **`data/registry/` gitignored via the existing `data/*` rule; no portable-export tool at Stage 9.** A future export tool is named in the layer doc as out-of-scope. | `data/*` already excludes `data/`, so there is nothing to add to `.gitignore`. A portable export adds scope no AC asks for. | Codebase map §7 (existing `.gitignore` rule `data/*` with `!data/.gitkeep` sentinel). |
| **D12** | Public surface | **Four verbs: `save(model, metrics_df, *, feature_set, target)`, `load(run_id)`, `list(*, target=None, model_type=None, feature_set=None, sort_by="mae", ascending=True)`, `describe(run_id)`.** Exported as `__all__ = ("save", "load", "list_runs", "describe")` — **`list_runs` is the exported symbol** to avoid shadowing the builtin `list`. A structural test counts `__all__` and fails if > 4. | AC-1 is the load-bearing constraint and `__all__` is the one place the constraint can be structurally pinned. `list_runs` vs `list` is the minimal concession to Python builtin shadowing; the layer doc and CLI both use the verb "list" so the demo story is unchanged. | Acceptance criterion AC-1 (four-verb hard cap — "if it grows past that, the design is drifting"). Domain research §R9 (industry interface breadth — BentoML 5 verbs, ZenML 11, MLflow broader; four is the minimum viable). Codebase map §8 (downstream consumers' minimum common interface). |
| **D13** | Git SHA capture | **Inside the registry module.** `_git_sha_or_none()` in `src/bristol_ml/registry/_git.py` calls `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=False, timeout=2.0)` at save time and returns the stripped stdout on `returncode == 0`, else `None`. | The model is not responsible for provenance it cannot know. Capturing at save time inside the registry means the four model classes stay untouched (AC-2) and the helper has one canonical home. A 2 s subprocess call beats a libgit2 binding. | Acceptance criterion AC-3 (automatic capture of git SHA). Codebase map hazard H2 (no existing `_git_sha_or_none` helper; every model today leaves `ModelMetadata.git_sha = None`). Domain research §R2 (MLflow's system-tag precedent captures git commit without user intervention). |
| **D14** | Serialisation backend | **Keep joblib; defer skops to Stage 12.** `src/bristol_ml/models/io.py` gains a one-line module-docstring addition naming Stage 12 as the `skops` trigger and citing sklearn's own "research / single-user" guidance. No code change. | Stage 12 (serving) is the first consumer that may load from a path not controlled by the training author — the correct inflection point for the security upgrade. A one-line docstring edit is the cheapest way to record the deferral. | Codebase map hazard H1 (the `skops.io` upgrade seam is flagged in `models/io.py:15-18` and `docs/architecture/layers/models.md:100` as a Stage-9 decision). Domain research §R8 (sklearn's decision tree sanctions joblib for single-author research tools; `skops` has its own CVEs and requires `trusted=` annotations). |
| **D15** | Metrics capture path | **Caller passes the metrics DataFrame explicitly to `save()`.** Registry extracts per-metric mean and std from the per-fold table at save time; the harness is unchanged. | Adding a new flag to `harness.evaluate()` would trigger the harness's API-growth rule (second ask on that interface). Explicit caller-passes keeps the harness pure and the registry's save signature self-contained. | Acceptance criterion AC-3 ("the rest is passed in explicitly at save time"). Codebase map §3 (harness return shape already includes the metrics DataFrame); hazard H5 (`evaluation/CLAUDE.md` explicitly prohibits a second boolean flag on `evaluate()`). |
| **D16** | CLI entry | **New `src/bristol_ml/registry/__main__.py` providing `python -m bristol_ml.registry list` (the Demo moment command) and `python -m bristol_ml.registry describe <run_id>` (AC-1 "maybe describe").** Uses `argparse` — same pattern as `train.py`. `list` prints a formatted table; `describe` pretty-prints one sidecar. | argparse is already in the stdlib and already the pattern `train.py` uses, so the CLI adds zero dependencies and zero new patterns. A `--registry-dir` override keeps the CLI-under-test flexible without introducing a Hydra schema. | Intent §Demo moment supplies the CLI command verbatim ("A single CLI command prints a leaderboard"). Codebase map §4 (`train.py` uses `argparse` at line 140 with `load_config(overrides=...)` — same pattern the registry CLI follows). |
| **D17** | `train.py` wiring | **After `harness.evaluate(...)` returns, save the final-fold fitted model (already in harness memory) via `registry.save(...)`, passing the per-fold metrics DataFrame.** No re-fit on full data. Add a new public function `harness.evaluate_and_keep_final_model(...)` returning `(metrics_df, final_fitted_model)`; the existing `evaluate()` signature is unchanged. | Re-fit-on-full-data adds a second training pass per CLI invocation for no named requirement. The final-fold model is an honest representative of what the cross-fold metrics describe. A new named function respects the harness API-growth rule. | Acceptance criterion AC-2 (retrofit — but does not require re-fit). Codebase map §3 (the harness today discards the fitted model; `train.py:299-316` is the insertion point); hazard H5 (no new flag on `evaluate()`). |
| **D18** | Layer documentation | **New `docs/architecture/layers/registry.md`** covering: on-disk layout (directory-tree diagram), sidecar JSON schema (field-by-field annotation), the four-verb contract, the last-write-wins versioning rule, the MLflow graduation subsection (D10), and the `skops` deferral note (D14). Non-empty gate enforced by `test_registry_layout_documentation_exists`. | One file, one gate test, nothing new to discover. | Acceptance criterion AC-5 (documentation requirement). Codebase map §5 (existing layer-doc precedent — every shipped module layer has a file under `docs/architecture/layers/`). |
| **D19** | Stage hygiene (H-1..H-4) | See "Housekeeping carry-overs" below. | Standard per `CLAUDE.md` stage-hygiene section. | `CLAUDE.md` stage-hygiene section. |

### Non-functional requirements

| # | NFR | Default | Evidence |
|---|-----|---------|----------|
| **NFR-speed** | `list_runs()` over 100 registered runs completes in **< 1 s** on laptop-class CPU. Intent says "instantaneous"; the 1 s gate quantifies it. **Please confirm or adjust at Ctrl+G** — 1 s is an order of magnitude above plausible local-filesystem performance for 100 small JSON files, giving CI headroom without tolerating a pathological regression. | Acceptance criterion AC-4 (the "instantaneous" constraint). Domain research §R4 (query-performance benchmarks — a flat directory scan of 100 JSON sidecars is tens of milliseconds on local filesystems; MLflow's one-file-per-key layout is the slow case, which D1 explicitly avoids). |
| **NFR-transparency** | Sidecar is text-inspectable (`cat run.json` prints parseable JSON without tooling). `indent=2`, no binary escape, no YAML dependency. | Acceptance criterion AC-5 (hand-inspection without the CLI). Domain research §R3 (metadata-sidecar formats — JSON is the machine-readable default across MLflow / DVC / hand-rolled registries; YAML would add a dependency for no reader-facing gain). |

### Housekeeping carry-overs

| # | Item | Resolution |
|---|---|---|
| **H-1** | `docs/intent/DESIGN.md §6` layout tree — Stages 1–9 additions. Stage 9 adds `src/bristol_ml/registry/` (new layer), `docs/architecture/layers/registry.md`, `data/registry/` (gitignored). | **Flag for human-led batched §6 edit at Stage 9 PR review.** Lead MUST NOT touch §6 unilaterally (deny-tier). Stage 8 H-1 carries forward. |
| **H-2** | Stage 8 retro "Next" pointer to Stage 9 — confirm wording is current. | **Verify at T7 hygiene.** |
| **H-3** | `docs/architecture/layers/models.md` — `skops.io` upgrade-seam note moved from "Stage 9" to "Stage 12" per D14. | **Edit at T7 hygiene.** One-line change. |
| **H-4** | Dispatcher-duplication ADR (Stage 7/8 carry-over) — still deferred. Stage 9 does not add a model family so no third dispatcher site is created. | **Confirm no regression; re-defer to Stage 11 or a dedicated housekeeping stage.** |

### Resolution log

- **Drafted 2026-04-22** — pre-human-markup. All decisions D1–D19 are proposed defaults. Awaiting Ctrl+G review of the nineteen decisions, the NFR-speed threshold, and the H-1 batched §6 edit.
- **2026-04-23 Ctrl+G approval with two amendments:**
  - **D3** — run-ID format simplified from `{model_name}_{YYYYMMDDTHHMMSS}` to `{model_name}_{YYYYMMDDTHHMM}` (minute precision). Same-minute collisions are handled by D2 last-write-wins; same-minute is vanishingly rare at single-author pace. R7 added to §8 to make the trade-off explicit.
  - **D10** — upgraded from "document the MLflow adapter" to "document AND test the adapter". A test-only `mlflow.pyfunc.PythonModel` adapter lands under `tests/integration/mlflow_adapter.py`; `mlflow>=2.14,<3` is added to the `dev` dependency group; one new integration test (`test_registry_run_is_loadable_via_mlflow_pyfunc_adapter`) proves the mechanical-migration claim is falsifiable. Public surface stays at four verbs (AC-1) because the adapter is test-only. A new task T7 is inserted for this work; the previous T7 (stage hygiene) becomes T8.
  - Phase 2 begins.

---

## 2. Scope

### In scope

Transcribed from `docs/intent/09-model-registry.md §Scope`:

- **A registry module that wraps saving and loading of model artefacts with a consistent metadata record** (training-data reference via `feature_columns`, feature-set name, git SHA, wall-clock timestamp, held-out metrics).
- **Retrofit of the models shipped in Stages 4, 7, and 8** so they save through the registry rather than directly to disk. The four concrete classes: `NaiveModel`, `LinearModel`, `SarimaxModel`, `ScipyParametricModel`. No change to any model class body; the `Model` protocol is unchanged.
- **A small CLI path** (`python -m bristol_ml.registry list`) that lists registered models with their metrics, filterable by target, model-type, or feature-set.
- **Documentation of the on-disk layout and metadata schema** in a new `docs/architecture/layers/registry.md`.

### Out of scope (do not accidentally implement)

Transcribed from `docs/intent/09-model-registry.md §Out of scope`, plus items surfaced by discovery:

- **A hosted registry (MLflow, W&B).** DESIGN §10 defers; Stage 9 is filesystem-based. An adapter contract is documented (D10) and a test-only PyFunc adapter ships to prove the contract holds, but no runtime dependency on MLflow is added; `mlflow` appears only in the `dev` dependency group.
- **A UI beyond the CLI listing.** No web dashboard, no notebook widget beyond the standard leaderboard-table rendering.
- **Model promotion / staging semantics** (dev → staging → prod).
- **Artefact versioning beyond "last write wins" for a named experiment.** Same-second run-ID collisions overwrite.
- **Multi-user concurrent access / file locking.** Single-process tool.
- **A feature-table content hash** (D6 defers).
- **A SQLite or Parquet index file.** An in-memory scan of sidecars satisfies NFR-speed at the 100-run scale.
- **`RegistryConfig` Pydantic schema and Hydra config group.** A module-level `DEFAULT_REGISTRY_DIR` plus a `--registry-dir` CLI flag and keyword argument cover every call site; no `conf/_schemas.py`, `conf/registry/default.yaml`, or `AppConfig` touch.
- **`_NamedLinearModel` dynamic-name round-trip on `load()`.** The dynamic name is stored in the sidecar's `name` field but not re-applied; `load()` returns the base `LinearModel`.
- **Re-fit on full training data before saving.** The final-fold fitted model (already in harness memory) is registered.
- **Migration of `src/bristol_ml/models/io.py` to `skops.io`.** Deferred to Stage 12 per D14; one-line docstring update only.
- **A portable-export tool** for `data/registry/`.
- **A dedicated `delete` or `update` verb** on the public surface.
- **Any fix to `_NamedLinearModel.load`'s `NotImplementedError`** at `train.py:405`. Left for a future `train.py` cleanup.

---

## 3. Reading order for the implementer

Self-contained context for Phase 2 — read top-to-bottom before opening any file.

1. [`docs/intent/09-model-registry.md`](../../intent/09-model-registry.md) — the contract; 5 ACs and 8 Points bullets.
2. [`docs/lld/research/09-model-registry-requirements.md`](../../lld/research/09-model-registry-requirements.md) — US-1..US-5 and AC evidence table.
3. [`docs/lld/research/09-model-registry-codebase.md`](../../lld/research/09-model-registry-codebase.md) — where the registry plugs in; hazards H1–H7.
4. [`docs/lld/research/09-model-registry-domain.md`](../../lld/research/09-model-registry-domain.md) — §R1–§R10, especially §R3 (artefact/metadata separation) and §R7 (atomic write).
5. This plan §1 (decisions), §4 (acceptance criteria), §5 (architecture summary).
6. `src/bristol_ml/models/protocol.py:77-89` — `save`/`load`/`metadata` signatures.
7. `conf/_schemas.py:615-656` — `ModelMetadata` fields.
8. `src/bristol_ml/ingestion/_common.py:201-210` — the atomic-write idiom D5 mirrors.
9. `src/bristol_ml/models/io.py:50-53` — the joblib atomic-write sibling.
10. `src/bristol_ml/train.py:299-316` — harness call site where D17 wiring lands.

---

## 4. Acceptance criteria (quoted from intent; plan wins on mechanics, intent wins on intent)

All five criteria are copied verbatim from `docs/intent/09-model-registry.md §Acceptance criteria`, then grounded in a named test.

- **AC-1.** "The registry's public interface is small — save, load, list, maybe describe. If it grows past that, the design is drifting."
  - Test: `test_registry_public_surface_does_not_exceed_four_callables` — asserts `len(bristol_ml.registry.__all__) <= 4` and `set(bristol_ml.registry.__all__) == {"save", "load", "list_runs", "describe"}`.
- **AC-2.** "Every model shipped before this stage can save through the registry without code changes to the model itself (the interface is what the registry consumes)."
  - Tests (one per model class; all must pass without any import from or modification to model-class internals):
    - `test_registry_save_naive_model_via_protocol`
    - `test_registry_save_linear_model_via_protocol`
    - `test_registry_save_sarimax_model_via_protocol`
    - `test_registry_save_scipy_parametric_model_via_protocol`
  - Each test: instantiate via Hydra config, fit on a small fixture, call `registry.save(...)`, round-trip via `registry.load(...)`, assert `predict()` output matches to `atol=1e-10`.
- **AC-3.** "Metadata is captured automatically where it can be (git SHA, timestamps, feature-set name); the rest is passed in explicitly at save time."
  - Tests:
    - `test_registry_save_captures_git_sha_automatically` — save without supplying `git_sha`; assert `run.json["git_sha"]` is a non-empty `str` in CI (which runs inside a git tree).
    - `test_registry_save_raises_on_missing_required_explicit_field` — calling `save()` without the required `feature_set` kwarg raises `TypeError`.
- **AC-4.** "The leaderboard query is fast — listing a hundred runs should be instantaneous."
  - Test: `test_registry_list_hundred_entries_is_fast` — populate a fresh registry directory with 100 synthetic sidecars, assert `list_runs()` returns in `< 1.0` second.
- **AC-5.** "The on-disk layout is documented well enough that a contributor could inspect it by hand without the CLI."
  - Tests:
    - `test_registry_layout_documentation_exists` — asserts `docs/architecture/layers/registry.md` exists and is non-empty (> 50 lines).
    - `test_registry_run_json_is_hand_parseable` — writes a run via `save()`, `json.loads(path.read_text())` succeeds, every required field is present with the expected type.

Additional plan-surfaced tests (D-derived, not intent-AC):

- `test_registry_list_filter_by_target` (D7).
- `test_registry_list_default_sort_is_mae_ascending` (D8).
- `test_registry_run_is_loadable_via_mlflow_pyfunc_adapter` (D10) — uses the test-only `mlflow.pyfunc.PythonModel` adapter under `tests/integration/mlflow_adapter.py` to assert a registered run can be packaged and loaded through MLflow with numerically-identical `predict()` output (`atol=1e-10`). Skips if `mlflow` is not importable (only runs when the `dev` group is installed).

Atomic-write (D5) is covered by an inline assertion inside the AC-2 save tests rather than a dedicated crash-simulation test.

**Total shipped tests: 11** (four AC-2, two AC-3, one AC-4, two AC-5, one AC-1, two D7/D8, one D10).

---

## 5. Architecture summary (no surprises)

### Module structure

```
src/bristol_ml/registry/
├── __init__.py          # public API: save, load, list_runs, describe, __all__
├── __main__.py          # `python -m bristol_ml.registry <list|describe>`
├── _fs.py               # private: on-disk layout (run_id generation, paths, atomic write)
├── _git.py              # private: _git_sha_or_none() (D13)
└── _schema.py           # private: the sidecar JSON schema (TypedDict; not a Hydra config)
```

No `RegistryConfig` in `conf/_schemas.py`; no `conf/registry/default.yaml`; no change to `AppConfig`.

### On-disk layout

```
data/registry/                               # DEFAULT_REGISTRY_DIR, gitignored
├── .tmp_{short_uuid}/                       # staging dir during save (temp-dir-then-rename)
├── linear-ols-weather-only_20260423T1430/
│   ├── artefact/
│   │   └── model.joblib
│   └── run.json
├── sarimax-d1-d1-s168_20260423T1431/
│   ├── artefact/
│   │   └── model.joblib
│   └── run.json
└── naive-same-hour-last-week_20260423T1432/
    ├── artefact/
    │   └── model.joblib
    └── run.json
```

### Sidecar JSON schema

```jsonc
{
  "run_id": "linear-ols-weather-only_20260423T1430",
  "name": "linear-ols-weather-only",            // ModelMetadata.name (possibly _NamedLinearModel dynamic name)
  "type": "linear",                              // "naive" | "linear" | "sarimax" | "scipy_parametric"
  "feature_set": "weather_only",                 // required explicit kwarg
  "target": "demand_mw",                         // required explicit kwarg
  "feature_columns": ["temperature_2m", "..."],  // from ModelMetadata
  "fit_utc": "2026-04-23T14:30:17Z",             // from ModelMetadata (second-precise)
  "git_sha": "575ac9c",                          // auto-captured by D13 helper
  "hyperparameters": { /* full ModelMetadata.hyperparameters verbatim */ },
  "metrics": {
    "mae":  {"mean": 1234.5, "std": 123.4, "per_fold": [1200.0, 1250.0, 1240.0, ...]},
    "rmse": {"mean": 1567.8, "std": 145.6, "per_fold": [...]},
    "mape": {"mean": 0.043,  "std": 0.004, "per_fold": [...]},
    "wape": {"mean": 0.041,  "std": 0.003, "per_fold": [...]}
  },
  "registered_at_utc": "2026-04-23T14:30:18Z"    // save-time wall clock (second-precise)
}
```

### Public interface

```python
# src/bristol_ml/registry/__init__.py
__all__ = ("save", "load", "list_runs", "describe")

DEFAULT_REGISTRY_DIR = Path("data/registry")

def save(
    model: Model,
    metrics_df: pd.DataFrame,
    *,
    feature_set: str,
    target: str,
    registry_dir: Path | None = None,
) -> str: ...  # returns run_id

def load(run_id: str, *, registry_dir: Path | None = None) -> Model: ...

def list_runs(
    *,
    target: str | None = None,
    model_type: str | None = None,
    feature_set: str | None = None,
    sort_by: str | None = "mae",
    ascending: bool = True,
    registry_dir: Path | None = None,
) -> list[dict[str, Any]]: ...

def describe(run_id: str, *, registry_dir: Path | None = None) -> dict[str, Any]: ...
```

### Integration with `train.py`

`train.py:299-316` currently calls `harness.evaluate(...)`, prints the metric table, and discards the fitted model. The Stage 9 change (D17):

1. Add a new public function `harness.evaluate_and_keep_final_model(...)` returning `(metrics_df, final_fitted_model)`. The existing `evaluate()` signature is unchanged (respects the `evaluation/CLAUDE.md` H5 API-growth rule).
2. `train.py` calls the new function on the registry-save path and then `registry.save(final_fitted_model, metrics_df, feature_set=..., target=...)`.

Caller-supplied `feature_set` and `target` come from the resolved Hydra config (`cfg.features.name` and `cfg.model.target_column` — both already present).

### MLflow PyFunc adapter (test-only, D10)

```
tests/integration/
├── mlflow_adapter.py                # RegistryPyfuncAdapter + package_run_as_pyfunc helper
└── test_registry_mlflow_adapter.py  # D10 round-trip test
```

The adapter subclasses `mlflow.pyfunc.PythonModel`:

```python
# tests/integration/mlflow_adapter.py (test-only, ~25 lines)
class RegistryPyfuncAdapter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        run_dir = Path(context.artifacts["registry_run"])
        # Reuse registry's dispatcher (imported at call time):
        self._model = _load_from_run_dir(run_dir)

    def predict(self, context, model_input):
        return self._model.predict(model_input)


def package_run_as_pyfunc(run_id: str, dst: Path, *, registry_dir: Path) -> None:
    import mlflow.pyfunc
    run_dir = registry_dir / run_id
    mlflow.pyfunc.save_model(
        path=str(dst),
        python_model=RegistryPyfuncAdapter(),
        artifacts={"registry_run": str(run_dir)},
    )
```

Why `tests/` not `src/`: keeps `bristol_ml.registry` at four verbs (AC-1) and the runtime dependency footprint zero. The adapter proves the "mechanical migration" claim in D10 is falsifiable without leaking MLflow into production imports.

### CLI

```bash
# Demo moment (intent §Demo moment)
python -m bristol_ml.registry list

# With filters (D7)
python -m bristol_ml.registry list --target demand_mw --model-type sarimax

# Inspection (AC-1 "maybe describe")
python -m bristol_ml.registry describe linear-ols-weather-only_20260423T1430
```

---

## 6. Tasks (ordered — work strictly top-to-bottom; each commits individually)

### Task T1 — Registry module scaffold + on-disk layout + git helper

**Files:** new `src/bristol_ml/registry/__init__.py`, `_fs.py`, `_git.py`, `_schema.py`.
**Content:**
- `__all__` with the four verbs (stubs raising `NotImplementedError` acceptable for T1).
- `DEFAULT_REGISTRY_DIR = Path("data/registry")`.
- `_fs.py`: `_build_run_id(model_name: str, fit_utc: datetime) -> str`; `_run_dir(registry_dir, run_id) -> Path`; `_atomic_write_run(registry_dir, run_id, artefact_writer, sidecar_dict) -> None` (temp-dir-then-rename).
- `_git.py`: `_git_sha_or_none() -> str | None` (D13).
- `_schema.py`: `SidecarFields` `TypedDict` matching §5 schema.

**Tests (T1):**
- `test_registry_public_surface_does_not_exceed_four_callables` (AC-1).
- `test_registry_build_run_id_format` — asserts minute-precision ISO form `^[a-z][a-z0-9_.-]*_\d{8}T\d{4}$`.
- `test_registry_git_sha_helper_returns_str_in_git_tree`.

**Commits as:** `Stage 9 T1: registry module scaffold + on-disk layout + git-SHA helper`.

### Task T2 — `save()` implementation (all four models)

**Files:** `src/bristol_ml/registry/__init__.py` (implement `save`); no model-code changes.
**Content:**
- Construct the sidecar dict from `model.metadata`, caller-supplied kwargs, auto-captured `git_sha`, and the metrics DataFrame (per-metric mean/std/per_fold).
- Call `model.save(run_dir / "artefact" / "model.joblib")` via the existing protocol.
- Write `run.json` with `json.dumps(..., indent=2, allow_nan=True, ensure_ascii=False)`.
- Atomic via `_atomic_write_run` (D5).
- Return the `run_id` string.

**Tests (T2 — the four AC-2 save tests + AC-3):**
- `test_registry_save_naive_model_via_protocol`.
- `test_registry_save_linear_model_via_protocol`.
- `test_registry_save_sarimax_model_via_protocol`.
- `test_registry_save_scipy_parametric_model_via_protocol`.
- `test_registry_save_captures_git_sha_automatically`.
- `test_registry_save_raises_on_missing_required_explicit_field`.

**Commits as:** `Stage 9 T2: registry.save + AC-2 retrofit tests for all four models`.

### Task T3 — `load()` + round-trip completeness

**Files:** `src/bristol_ml/registry/__init__.py` (implement `load`).
**Content:**
- Parse `run.json`; read the `type` field; dispatch to the appropriate `Model.load(...)` classmethod via a local dict literal mapping `type` → class (no third shared dispatcher site).
- Return the fitted `Model` instance.
- `_NamedLinearModel` case: return base `LinearModel`; the sidecar's `name` field preserves the dynamic name for reading but is not re-applied to the loaded model instance.

**Tests (T3):**
- The four AC-2 save tests extend to round-trip: `save()` + `load()` + `predict()` agreement to `atol=1e-10`.
- `test_registry_load_raises_on_missing_run_id`.
- `test_registry_load_named_linear_returns_base_class` — asserts the loaded instance is `LinearModel`, not `_NamedLinearModel`, and that `run.json["name"]` retains the dynamic name.

**Commits as:** `Stage 9 T3: registry.load + AC-2 round-trip`.

### Task T4 — `list_runs()` + `describe()` + filtering + sort

**Files:** `src/bristol_ml/registry/__init__.py` (implement `list_runs`, `describe`).
**Content:**
- `list_runs`: `os.listdir(registry_dir)`; for each non-`.tmp_*` subdirectory, read `run.json`, apply filters (D7), sort by `sort_by` (D8), return list of dicts.
- `describe`: read and return one sidecar as a dict.

**Tests (T4):**
- `test_registry_list_hundred_entries_is_fast` (AC-4).
- `test_registry_list_filter_by_target` (D7).
- `test_registry_list_default_sort_is_mae_ascending` (D8).
- `test_registry_describe_returns_full_sidecar`.

**Commits as:** `Stage 9 T4: registry.list_runs + describe + filter/sort + AC-4 performance gate`.

### Task T5 — CLI entry + harness final-model extension + train-CLI wiring

**Files:**
- new `src/bristol_ml/registry/__main__.py` (D16).
- `src/bristol_ml/evaluation/harness.py` — add `evaluate_and_keep_final_model(...)` (existing `evaluate` untouched).
- `src/bristol_ml/train.py` — switch to `evaluate_and_keep_final_model` and call `registry.save(...)` after the metric print (D17).

**Tests (T5):**
- `test_registry_cli_list_prints_leaderboard_table` — shells out to `python -m bristol_ml.registry list`.
- `test_registry_cli_describe_prints_json`.
- `test_train_cli_registers_final_fold_model` — end-to-end on a tiny fixture: `python -m bristol_ml.train model=naive` leaves exactly one new `run_id` in a tmp registry dir.
- `test_harness_evaluate_and_keep_final_model_returns_tuple`.

**Commits as:** `Stage 9 T5: registry CLI + harness final-model handle + train-CLI wiring`.

### Task T6 — Layer documentation + AC-5 tests

**Files:** new `docs/architecture/layers/registry.md` (D18).
**Content:**
- Directory-tree diagram (from §5).
- Sidecar JSON schema, field-by-field.
- Four-verb interface contract with signatures.
- On-disk lifecycle (save writes to `.tmp_*`, atomic rename to final `run_id/`).
- VCS policy (D11).
- Graduation to MLflow subsection (D10) — names the PyFunc adapter pattern and references the T7 test as proof of mechanical migration.
- `skops` deferral note (D14) pointing at Stage 12.

**Tests (T6):**
- `test_registry_layout_documentation_exists` (AC-5).
- `test_registry_run_json_is_hand_parseable` (AC-5).

**Commits as:** `Stage 9 T6: registry layer doc + AC-5 tests`.

### Task T7 — MLflow PyFunc adapter (test-only) + round-trip test (D10)

**Files:**
- new `tests/integration/mlflow_adapter.py` — `RegistryPyfuncAdapter` subclass + `package_run_as_pyfunc` helper (test support code; not imported from production).
- new `tests/integration/test_registry_mlflow_adapter.py` — the D10 round-trip test.
- `pyproject.toml` — add `mlflow>=2.14,<3` to the `dev` dependency group only (runtime dependencies unchanged).

**Content:**
- `RegistryPyfuncAdapter(mlflow.pyfunc.PythonModel)` implementing `load_context` (reads `context.artifacts["registry_run"]` and delegates to the registry's type-dispatched loader) and `predict` (forwards to the wrapped model).
- `package_run_as_pyfunc(run_id, dst, *, registry_dir)` helper calling `mlflow.pyfunc.save_model(path=dst, python_model=adapter, artifacts={"registry_run": str(run_dir)})`.

**Tests (T7):**
- `test_registry_run_is_loadable_via_mlflow_pyfunc_adapter` — save a tiny `NaiveModel` via `registry.save`, package it with `package_run_as_pyfunc`, `mlflow.pyfunc.load_model(...)`, call `predict`, assert `numpy.allclose(..., atol=1e-10)` against the original model's output. Skipped via `pytest.importorskip("mlflow")` if mlflow is unavailable (laptop-only lean-install case).

**Commits as:** `Stage 9 T7: MLflow PyFunc adapter + round-trip test (D10)`.

### Task T8 — Stage hygiene (H-1..H-4)

**Files:**
- `CHANGELOG.md` — `[Unreleased]` Stage 9 bullets under `### Added`.
- `docs/architecture/layers/models.md` — `skops.io` upgrade-seam note moved from "Stage 9" to "Stage 12" (H-3).
- `src/bristol_ml/models/io.py` — one-line module-docstring edit naming Stage 12 as `skops` trigger (D14).
- `docs/lld/stages/09-model-registry.md` — retro per the template.
- `docs/lld/stages/08-scipy-parametric.md` — "Next" pointer confirmed current (H-2).
- `docs/plans/active/09-model-registry.md` → `docs/plans/completed/09-model-registry.md` at the final commit.

**Tests (T8):** none new.

**Commits as:** `Stage 9 T8: stage hygiene + retro + plan moved to completed/`.

---

## 7. Files expected to change

### New

- `src/bristol_ml/registry/__init__.py`
- `src/bristol_ml/registry/__main__.py`
- `src/bristol_ml/registry/_fs.py`
- `src/bristol_ml/registry/_git.py`
- `src/bristol_ml/registry/_schema.py`
- `tests/unit/registry/test_registry_fs.py`
- `tests/unit/registry/test_registry_save_load.py`
- `tests/unit/registry/test_registry_list_describe.py`
- `tests/unit/registry/test_registry_cli.py`
- `tests/integration/test_train_cli_registers_model.py`
- `tests/integration/mlflow_adapter.py` (D10, test-only)
- `tests/integration/test_registry_mlflow_adapter.py` (D10, test-only)
- `docs/architecture/layers/registry.md`
- `docs/lld/stages/09-model-registry.md`

### Modified

- `src/bristol_ml/evaluation/harness.py` — new public `evaluate_and_keep_final_model` function.
- `src/bristol_ml/train.py` — switch to `evaluate_and_keep_final_model`; call `registry.save` after metric print.
- `src/bristol_ml/models/io.py` — one-line module-docstring edit (D14).
- `docs/architecture/layers/models.md` — `skops.io` row moved Stage 9 → Stage 12 (H-3).
- `pyproject.toml` — add `mlflow>=2.14,<3` to the `dev` dependency group (D10, test-only; no runtime dependency change).
- `CHANGELOG.md` — `[Unreleased]` Stage 9 bullets.

### Moved (final commit of T7)

- `docs/plans/active/09-model-registry.md` → `docs/plans/completed/09-model-registry.md`.

### Explicitly NOT modified

- `docs/intent/DESIGN.md` §6 — deny-tier; H-1 batches for human-led edit.
- `docs/intent/09-model-registry.md` — immutable spec.
- `conf/_schemas.py`, `conf/registry/`, `conf/config.yaml` — no `RegistryConfig`, no Hydra group, no `AppConfig` field.
- `src/bristol_ml/models/protocol.py` — protocol signature unchanged per D9.
- `src/bristol_ml/models/{naive,linear,sarimax,scipy_parametric}.py` — no body changes (AC-2 contract).
- `src/bristol_ml/train.py:397-408` — `_NamedLinearModel.save`/`.load` untouched.
- `src/bristol_ml/evaluation/harness.py::evaluate` — signature unchanged (H5).
- `pyproject.toml` `[project].dependencies` — unchanged (MLflow goes into `dev`, not runtime).

---

## 8. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | `git_sha` helper returns `None` in CI (detached HEAD or shallow clone), hiding a bug. | Medium | Medium | Test asserts a concrete non-empty `str` in CI. T1 unit test gates the helper. |
| **R2** | Final-fold model is not representative of full training data, so the leaderboard shows metrics from a model a facilitator cannot reproduce. | Medium | Medium | The leaderboard metric row is the cross-fold summary (D15: per-metric mean ± std) — an honest rolling-origin estimate. `save()` docstring + layer doc §"What artefact is saved" make the final-fold choice explicit. |
| **R3** | `ScipyParametricModel`'s covariance matrix with `float('inf')` entries trips JSON serialisation. | Low | Medium | D4 mandates `allow_nan=True`; `test_registry_save_scipy_parametric_model_via_protocol` includes a fold that produces `inf` entries. |
| **R4** | 100-run AC-4 test is flaky on slow CI. | Medium | Low | Threshold is 1 s; expected local runtime ~10 ms. If CI flakes, mark `@pytest.mark.slow`; do NOT weaken the threshold. |
| **R5** | `.tmp_*` staging directories left behind after a crashed save litter `data/registry/`. | Medium | Low | T4 `list_runs` filters directories whose names start with `.tmp_`. Cleanup-on-list is out of scope. |
| **R6** | MLflow version drift breaks the PyFunc adapter test (the `PythonModel` / `save_model` API has moved in past MLflow majors). | Low | Low | `mlflow>=2.14,<3` pinned in the `dev` group; the test lives under `tests/integration/` and `pytest.importorskip("mlflow")` guards the skip path so a fresh-install dev without `--group dev` still passes the rest of the suite. If a future MLflow 3.x release breaks PyFunc semantics, address at the version-bump boundary — the test is the guardrail, not a silent passthrough. |
| **R7** | Two runs started in the same minute collide under D3 minute precision and D2 last-write-wins, losing the earlier artefact. | Low | Low | Single-author pace makes same-minute collisions rare; D2 explicit last-write-wins semantics are documented in the layer doc (D18). If future multi-worker training raises the collision rate, D3 can graduate to second precision under a new decision — no schema change required. |

---

## 9. Exit checklist

Verified before T7's final commit.

- [ ] All tests pass: `uv run pytest -q`. No skipped tests; no `xfail` without a linked issue. `@pytest.mark.slow` tests run explicitly via `uv run pytest -m slow` and pass on CI-class hardware.
- [ ] Ruff + format + pre-commit clean: `uv run ruff check . && uv run ruff format --check . && uv run pre-commit run --all-files`.
- [ ] `uv run python -m bristol_ml.registry --help` exits 0 and prints subcommand usage.
- [ ] `uv run python -m bristol_ml.registry list` exits 0 on an empty registry and on a populated one.
- [ ] `uv run python -m bristol_ml.train model=<family>` for each of `naive`, `linear`, `sarimax`, `scipy_parametric` leaves exactly one new `run_id` in `data/registry/`.
- [ ] All five intent-ACs mapped to named tests in §4 have a passing test.
- [ ] `test_registry_run_is_loadable_via_mlflow_pyfunc_adapter` (D10) passes when `uv sync --group dev` is active; skipped-with-reason when `mlflow` is absent.
- [ ] `docs/architecture/layers/registry.md` exists and is > 50 lines (AC-5).
- [ ] `docs/lld/stages/09-model-registry.md` retro written per template.
- [ ] `CHANGELOG.md`, `docs/architecture/layers/models.md` (H-3) updated.
- [ ] `docs/plans/active/09-model-registry.md` moved to `docs/plans/completed/`.
- [ ] H-1 (DESIGN §6), H-2 (Stage 8 retro wording), H-3 (skops seam Stage 9 → Stage 12), H-4 (dispatcher ADR re-defer) all actioned per §1 Housekeeping.
- [ ] PR description includes: Stage 9 summary, Scope Diff link, any Phase 3 review findings, H-1 DESIGN §6 batched edit request, NFR-speed threshold confirmation.
