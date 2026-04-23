# Stage 9 — Model registry: domain research

**Date:** 2026-04-22
**Target plan:** `docs/plans/active/09-model-registry.md` (not yet created)
**Intent source:** `docs/intent/09-model-registry.md`
**Baseline SHA:** `575ac9c`

**Scope:** external literature and primary tool documentation to inform Stage 9 plan decisions.  Numbered subsections (§R1–§R10) so the plan can cite by reference.  Recommendations are pre-tagged with `AC-aligned` / `PLAN POLISH` / `PREMATURE OPTIMISATION` so the Phase-1 `@minimalist` can use the triage directly.

---

## §R1 — Filesystem-backed model registry: canonical layouts

**MLflow file store layout.**  The MLflow file store writes under `mlruns/` with a two-level hierarchy: `mlruns/{experiment_id}/{run_id}/`.  Inside each run directory there are four subdirectories (`metrics/`, `params/`, `tags/`, `artifacts/`) and one file (`meta.yaml`).  Every individual metric, param, and tag is stored as a **separate file** — the file name is the key, the file content is the value.  As of MLflow 3.7 (December 2025) the file store is deprecated in favour of SQLite as the default backend; it is in "Keep-the-Light-On (KTLO) mode" and will not receive new features.  Migration to SQLite is a single `mlflow migrate-filestore` command.

**DVC cache layout.**  DVC uses a content-addressable cache under `.dvc/cache/files/md5/` with a two-level sharding scheme.  Directories get a `.dir` JSON sidecar listing each child file's hash and relative path.  DVC does not maintain a "run registry" in the MLflow sense; model artefacts are declared in `dvc.yaml` `artifacts:` sections with name, path, type, and labels metadata.  The registry is Git-tracked YAML, not a separate data store.

**Hand-rolled research registries.**  Hand-rolled registries (fast.ai `learn.export()`, scikit-learn tutorial galleries) use a simple pattern: one directory per experiment name, one `.pkl` or `.joblib` file for the artefact, and optionally a sidecar `metadata.json` in the same directory.  Neither follows a formal layout specification.

**Flat vs nested: the hand-inspection trade-off.**  The MLflow file store uses a nested layout (`experiment_id/run_id/`) which is easy to browse per-experiment but makes cross-experiment listing require traversing two directory levels.  A flat layout (`{run_id}/` directly under a registry root, with all metadata in one sidecar JSON per run) makes cross-run listing a single `os.listdir` + parse.  The intent (§Points line 39) correctly identifies this as the primary design decision: flat = easier to query; nested = easier to navigate by hand.

- **Implication for Stage 9:** The MLflow one-file-per-key scheme is the root cause of all its file store performance problems (§R4).  A flat layout with **one JSON sidecar per run** avoids that pathology while remaining hand-inspectable.  `AC-5` (inspectable by hand) and `AC-4` (listing 100 runs is instantaneous) both point to this choice.

---

## §R2 — Metadata schema minimum

**MLflow automatically-captured fields.**  Per the `mlflow/utils/mlflow_tags.py` source, MLflow captures the following system tags without any user call: `mlflow.user`, `mlflow.source.name`, `mlflow.source.type`, `mlflow.source.git.commit`, `mlflow.source.git.branch`, `mlflow.source.git.repoURL`, `mlflow.source.git.dirty`.  The `run_id`, `start_time`, and `lifecycle_stage` are stored in `meta.yaml` (not as tag files).  There is no automatic logging of feature-set name or dataset hash — those are user-supplied.

**Model Card schema (Mitchell et al. 2019).**  The Model Cards paper specifies nine top-level sections: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, and caveats.  Fields available at training time: model details (name, version, type, training framework), training data (dataset name, preprocessing), metrics (what metric was optimised), quantitative analyses (per-slice performance on held-out data).  Fields requiring retrospective completion: intended use (requires deployment context), ethical considerations (requires stakeholder review), caveats (requires post-deployment monitoring).  For a single-author research registry, the trainable-at-training-time fields reduce to: model name, type, training data description, git SHA, timestamp, and held-out metrics.

**Minimum reproducible set.**  Across the reproducibility literature (Sugimura & Hartley 2018 arXiv:1810.04570, and MLflow autolog docs), the consensus minimum to reproduce a run is: random seed, software environment (Python + library versions), training data reference (path or hash), hyperparameters, and git commit.  A feature-table hash adds tighter data-lineage guarantees but is not standard practice in hand-rolled single-author tools — DVC uses it for dataset versioning as an opt-in step, and MLflow's `MetaDataset` type provides it as an optional `log_input` call.

- **Implication for Stage 9:** A feature-table hash is `PREMATURE OPTIMISATION` at this scale; it adds a step to every save for a single-author tool where the git SHA already pins the feature-engineering code.  `Points §line-41` is answerable: defer the hash, keep the git SHA.  `AC-3` (metadata captured automatically where possible) is satisfiable with: git SHA, timestamp, and source file name — all from the MLflow system-tag precedents.

---

## §R3 — Artefact vs metadata separation

**Industry norm: split blob from metadata.**  All mature registries separate the binary artefact from the textual metadata.  MLflow stores artefacts under `artifacts/` and metadata in `meta.yaml` + individual tag/param/metric files.  DVC stores the binary in its MD5 cache and the metadata in `dvc.yaml`.  BentoML stores models in a local "model store" directory with a separate `saved_model.yaml` sidecar.  The consistent pattern is: one directory per run/version, binary blob inside it, metadata alongside it as a human-readable text file.

**Metadata file format.**  JSON dominates for machine-readable sidecar files (MLflow `meta.yaml` is actually YAML; DVC uses YAML for `dvc.yaml` and JSON for `.dir` directory manifests).  TOML is rare in ML tooling.  Parquet is used for tabular data, not for metadata sidecars.  For a sidecar that must be both human-readable and machine-parseable, YAML is the most common choice in ML tooling; JSON is a valid alternative and is strictly simpler to parse in Python without a YAML library.

**Content-hash addressing.**  DVC uses content-addressing (the file's identity is its MD5 hash) as an explicit design choice for deduplication across large datasets.  For a model registry at the scale of 100–1000 runs where models are small (sklearn pipelines, not neural network checkpoints), content-addressing adds complexity without deduplication benefit — models from different runs will nearly always have different weights.  Content-addressing is appropriate for dataset registries, not for model artefact registries at this scale.  MLflow's own file store does not content-address model artefacts; it stores them under the run UUID directory.

**SemVer vs UUID vs timestamp addressing.**  MLflow uses UUID v4 (32 hex chars, auto-generated per training run).  W&B uses a short random string as the run ID with a human-readable display name.  DVC uses git SHAs to version datasets.  BentoML uses `{name}:{version}` where version is a datetime string by default.  For a filesystem registry where the "address" must be human-typeable (demo use case), a `{model_name}_{timestamp}` or `{model_name}_{short_uuid}` scheme beats raw UUID for hand-inspection.

- **Implication for Stage 9:** Store one sidecar JSON (or YAML) per run alongside the artefact binary.  Use JSON over YAML to avoid a runtime dependency on `pyyaml` for the metadata reader.  `AC-5` (hand-inspectable).  `NON-INTENT`: joblib's multi-file output (`.joblib` + `.npy` siblings) means the artefact directory must store the joblib output directory, not a single file — the registry should save into a per-run subdirectory, then write the sidecar alongside it.

---

## §R4 — Query / leaderboard patterns

**MLflow file store performance at scale.**  The file store has no index: `search_runs` scans every run directory, opens `meta.yaml`, then opens each metric/param/tag file individually.  Reported real-world performance: 120 runs × 9 metrics takes ~30 seconds on a shared filesystem (GitHub issue #1902).  A single experiment with 30 runs can take ~1 second with 90% of time in param/metric file reads.  MLflow's own issue tracker describes "50k–100k files" for modest run counts because every param and metric is a separate file.  At 100 runs this is well within the "instantaneous" threshold if metadata is consolidated; at 1000 runs with the one-file-per-key design it is not.

**Consequences for layout.**  The performance problem is entirely caused by the one-file-per-key design.  A single JSON sidecar per run (all metadata in one file) reduces the file count by one to two orders of magnitude and makes listing 100 runs a matter of reading 100 small JSON files — achievable in O(10 ms) on a local filesystem.

**Indexing alternatives.**  Three alternatives to directory-scanning:
1. A SQLite database alongside the artefacts (MLflow's "sql store" approach, now the default from v3.7).  Provides SQL filtering; adds a dependency and a daemon process for remote access.
2. A single `index.parquet` or `index.json` rebuilt on startup from the per-run sidecars.  No extra dependency; no daemon; slightly stale if two processes write concurrently (out of scope per intent §Out of scope line 59).
3. In-memory index rebuilt by scanning all sidecars at list time.  Simplest; adequate for 100–1000 runs if each sidecar is one JSON file (not 50 k files).

**MLflow's silent 1000-run limit.**  `MlflowClient().search_runs()` silently defaults to a maximum of 1000 results; `mlflow.search_runs()` defaults to 100 000 but internally paginates through `MlflowClient`.  This is a usability trap (GitHub issue #15814) relevant only when wrapping MLflow, not when rolling a custom registry.

- **Implication for Stage 9:** A single JSON sidecar per run (in-memory scan at list time) satisfies `AC-4` at the project's expected scale of ~100 runs.  No index file is needed.  `PREMATURE OPTIMISATION` to add a SQLite index at Stage 9.

---

## §R5 — Graduation path to hosted registries

**MLflow migration tooling.**  The official path from filesystem to MLflow is a one-command migration introduced in MLflow ≥ 3.10: `mlflow migrate-filestore --source /path/to/mlruns --target sqlite:///mlflow.db`.  This is atomic (rolls back on error) and targets SQLite only.  For migration to a remote MLflow server, the community `mlflow-export-import` package (PyPI: `mlflow-export-import`) exports experiments/runs to an intermediate directory and re-imports them via the REST API.  Limitation: original run IDs and timestamps are not preserved across import/export.

**Is the "mechanical migration" claim supportable?**  The intent's claim (§Points line 45: "the migration is mechanical: a new implementation of the same small interface, backed by MLflow") is **conditionally supportable**.  It is supported if: (a) the Stage 9 registry interface is a thin Python class with `save`, `load`, `list`, and optionally `describe`; (b) the on-disk artefacts are stored in a layout that MLflow can ingest (i.e., standard joblib files, not a custom binary format).  It is not supported if the filesystem registry's run IDs are embedded in downstream code that must remain stable — the `mlflow-export-import` tool does not preserve run IDs.

**Actual migration experience.**  Thin evidence in the public literature — the only well-documented migration path is the official `mlflow migrate-filestore` command for filesystem-to-SQLite.  The claim that "a new implementation of the same small interface" is sufficient is aspirational for the model-loading case: MLflow's `mlflow.sklearn.load_model` expects artefacts logged via `mlflow.sklearn.log_model`, which wraps the joblib file in a `MLmodel` YAML envelope.  A hand-rolled registry that stores a bare joblib file cannot be directly loaded via MLflow's model-loading API without a thin adapter.

- **Implication for Stage 9:** The graduation path is real but requires an adapter layer, not just a new backend class.  Storing artefacts in MLflow's `MLmodel`-compatible envelope from the start would make the migration drop-in, but adds MLflow as a dependency from Stage 9 — likely not the intent.  Document the adapter requirement explicitly.  `Points §line-45`.

---

## §R6 — "What counts as a run" conventions

**MLflow convention.**  Each call to `mlflow.start_run()` is a run; the run is identified by a UUID v4 auto-generated at creation time.  Runs are grouped into experiments (by experiment ID).  A "model" in the model registry is a named entity with one or more versions; each version points to a run's artefact path.  The run UUID is the primary key; the model name + version number is a secondary human-readable address.

**W&B convention.**  Each call to `wandb.init()` is a run, identified by a short random string (run ID).  W&B uses a random two-word display name by default.  Run IDs must be unique within the project and cannot be reused after deletion.

**DVC convention.**  DVC does not have a "run" concept in the MLflow sense; it versions data and pipeline outputs using git commits.  An "experiment" in DVC is a git commit with a modified `dvc.yaml` or params file.

**De-facto industry convention.**  UUID per training invocation is the most common choice (MLflow, W&B).  Auto-incrementing integers are used in some SQL-backed registries for readability.  A `(model_name, feature_set, timestamp)` composite key is less common but appears in hand-rolled registries.  Content-hash as primary key (like DVC) is used only for dataset registries, not model registries, because two runs trained on the same data with the same hyperparameters are still distinct events.

- **Implication for Stage 9:** A UUID or ISO-8601 timestamp as run ID is both simple and consistent with industry practice.  The intent's judgement call (§Points line 40) is best resolved as: one entry per training invocation, identified by a short UUID (8 hex chars is sufficient for a single-author tool).  `Points §line-40`.

---

## §R7 — Write safety / atomicity on filesystem

**Standard atomic write pattern.**  The canonical safe pattern for multi-file artefacts on a POSIX filesystem is: write all files into a temp directory under the same filesystem partition, then `os.rename()` (or `pathlib.Path.rename()`) the temp directory to the target name.  `os.rename()` is atomic on POSIX for same-filesystem moves; the target either appears fully or not at all.  This is the approach described in the USENIX OSDI and FAST filesystem papers and in the Python ecosystem (`tempfile.mkdtemp` + rename).

**MLflow's approach.**  MLflow's file store does not use an atomic temp-dir-then-rename pattern for runs: it creates the run directory and `meta.yaml` first, then writes params/metrics/tags as individual files.  A crash between the `meta.yaml` write and the artefact write leaves a "dead run" — a run directory with metadata but no artefact.  MLflow addresses this via its `lifecycle_stage` field: a run that was never ended (no `end_time` in `meta.yaml`) is identifiable as incomplete.  The `.trash` folder is used for soft-deleted runs.  The FileStore has a known bug (GitHub issue #8177) where `lifecycle_stage` is not reliably updated to `DELETED` on the file store backend.

**"No locking needed" vs "process-level safety still needed".**  The intent defers multi-user concurrent access (§Out of scope line 59).  For a single-process, single-author tool, a crash leaving a partial write is the only concern.  A temp-dir-then-rename pattern eliminates the "dead run" pathology: either the entire run directory (artefact + sidecar) appears, or nothing does.

- **Implication for Stage 9:** Implement the temp-dir-then-rename pattern for the `save` path.  Cost: trivial (four lines of Python).  Benefit: eliminates the most common filesystem registry pathology (dead runs with metadata but no artefact).  No lock files or WAL needed for a single-process tool.  `NON-INTENT` but directly relevant to the "known pitfalls" the plan must address.

---

## §R8 — Security posture

**CVE-2024-34997 (joblib).**  Snyk assigned this CVE to joblib v1.4.2 for insecure use of `pickle.load()` in `NumpyArrayWrapper.read_array()`.  The CVE was subsequently **revoked**: the joblib maintainers disputed it as "not a valid vulnerability in the context of the library" because `NumpyArrayWrapper` is only used to cache trusted content.  The vendor position is that the risk is only present when loading models from untrusted sources.

**Scikit-learn's official guidance.**  The scikit-learn 1.x documentation (Model Persistence page) provides a decision tree: for a **research / single-user environment with trusted data**, `pickle` with `protocol=5` is the recommended approach.  For **production or security-sensitive environments**, `skops.io` is recommended.  Joblib is recommended only when efficient memory-mapping for multi-process loading is needed.  The guidance explicitly differentiates single-user research tools from multi-user production systems.

**skops CVEs.**  Two skops CVEs exist: CVE-2024-37065 and CVE-2025-54886, both for "Deserialization of Untrusted Data" in skops itself — skops is not unconditionally safe.  The trust model requires the caller to explicitly pass `trusted=` types at load time.

**Stage 9 inflection point.**  Stage 9 is a single-author research tool loading only its own saved models.  The sklearn guidance explicitly categorises this as the "research / single-user" case where pickle/joblib is acceptable.  `skops` adoption is appropriate at Stage 12 (serving), where models may be loaded from a path not controlled by the training author.

- **Implication for Stage 9:** Joblib is acceptable at Stage 9 under sklearn's own guidance.  Add a docstring warning that the registry loads only files written by the same codebase.  Defer `skops` to Stage 12.  `NON-INTENT` for Stage 9; relevant for the Stage 12 plan.  CVE-2024-34997 is **disputed and revoked**.

---

## §R9 — Interface surface guidance

**MLflow core verbs.**  MLflow's public interface decomposes into: `start_run` / `end_run` (lifecycle), `log_param` / `log_metric` / `log_artifact` (write), `search_runs` / `get_run` (read), `register_model` / `load_model` (model registry).  The leaderboard is `search_runs()` with a filter expression and sort key.

**ZenML minimum abstract interface.**  The `BaseModelRegistry` abstract class in ZenML requires implementing 11 methods across two groups: model registration (`register_model`, `delete_model`, `update_model`, `get_model`, `list_models`) and model version management (`register_model_version`, `delete_model_version`, `update_model_version`, `list_model_versions`, `get_model_version`, `load_model_version`).  This is 11 methods minimum.

**BentoML minimum interface.**  BentoML's model store exposes: `save_model(name, model, ...)`, `load_model(tag)`, `get(tag)`, `list()`, `delete(tag)`.  Five methods.  Tags are `{name}:{version}` strings.

**AC-1 alignment.**  The intent says "save, load, list, maybe describe" (AC-1).  This four-verb surface is consistent with BentoML's minimal public interface and is narrower than ZenML's 11-method abstract class.  MLflow's full interface is much broader.  The intent's surface is the minimum viable: `save` (write artefact + metadata), `load` (retrieve artefact), `list` (return records filterable by target/model/feature-set), `describe` (return full metadata for one run).  A fifth verb, `delete`, is not mentioned in the intent but is present in every other registry.

- **Implication for Stage 9:** The four-verb surface (save / load / list / describe) is well-supported by industry precedent as a minimum useful interface.  ZenML's 11-method abstract class and MLflow's broader surface are both over-specified for this stage.  `AC-1 AC-aligned`.

---

## §R10 — Dogfood risks / known anti-patterns

**Anti-pattern 1: directory entry explosion.**  MLflow's one-file-per-key scheme creates 50 000–100 000 files for modest run counts (GitHub issue #1902).  Root cause: every metric name, param name, and tag key becomes a separate inode.  Impact: `ls` is slow, `search_runs` is slow, filesystem backup is slow.  Mitigation: consolidate all metadata into one JSON sidecar per run.

**Anti-pattern 2: metadata drift between artefact and sidecar.**  If the artefact write succeeds but the sidecar write fails (or vice versa), the registry is inconsistent: a loadable model with no queryable metadata, or queryable metadata pointing to a missing artefact.  MLflow's FileStore has this risk because it writes `meta.yaml` and the artefact in separate operations.  Mitigation: temp-dir-then-rename (§R7) ensures both appear atomically.

**Anti-pattern 3: dead runs (save succeeded, metadata write failed).**  Inverse of anti-pattern 2.  A directory exists with a model artefact but no sidecar, or with a sidecar marked as "started" but no `end_time`.  MLflow mitigates this via `lifecycle_stage` in `meta.yaml`; its FileStore has a known bug where `lifecycle_stage` is not reliably updated on deletion (GitHub issue #8177).  Mitigation: write sidecar last after artefact is confirmed; the presence of a complete sidecar is the signal that a run is valid.

**Anti-pattern 4: run ID collision / reuse.**  Auto-incrementing integers collide if runs are deleted and the counter is reset.  UUIDs do not collide.  Timestamp-based IDs collide if two runs start within the same clock tick.  W&B explicitly states run IDs "cannot be reused once deleted."  Mitigation: use UUID4 (or a timestamp + 4-hex-char suffix for readability).

**Anti-pattern 5: joblib multi-file output ignored.**  When `joblib.dump` is called with compression, it may produce a single file.  Without compression, on some model types it produces a `.joblib` file plus `.npy` siblings.  If the registry copies only the `.joblib` file, the model is unloadable.  Mitigation: save each run's artefact into its own subdirectory; copy the entire directory, not a single file.

- **Implication for Stage 9:** Anti-patterns 2, 3, and 5 are directly preventable by the temp-dir-then-rename pattern and subdirectory-per-run layout.  Anti-pattern 1 is prevented by one-JSON-sidecar-per-run.  Anti-pattern 4 is prevented by UUID4 run IDs.  `NON-INTENT` but all five are plan-level concerns.

---

## Recommendations for consideration

Pre-tagged so the `@minimalist` scope critic can triage directly.

1. **Flat layout, one JSON sidecar per run.** Default: `registry/{run_id}/` containing the artefact subdirectory and a `run.json` sidecar.  Listing is `os.listdir(registry_root)` + read each `run.json`.  This satisfies `AC-4` and `AC-5` and avoids the MLflow file store's directory-explosion pathology.  **`AC-aligned`**.

2. **Temp-dir-then-rename for atomic saves.** Default: write artefact and sidecar into a temp directory under the registry root, then `Path.rename()` to the final run directory.  Eliminates dead-run and metadata-drift pathologies at negligible implementation cost.  **`AC-aligned`** (supports AC-2 by making the save path reliable for retrofitted models).

3. **UUID4 run ID (short form).** Default: `uuid.uuid4().hex[:8]` (8 chars) as the run ID, embedded in the directory name as `{model_name}_{timestamp}_{short_uuid}`.  Human-typeable for the demo; collision-resistant at 100–1000 runs.  **`PLAN POLISH`**.

4. **Defer feature-table hash.** Default: do not include a hash of the feature Parquet in the sidecar.  Git SHA + feature-set name + training date range is sufficient to reproduce a run at Stage 9 scale.  Feature hashing is a `PREMATURE OPTIMISATION` until Stage 17 or later when data provenance across multiple datasets becomes meaningful.  `Points §line-41`.  **`PREMATURE OPTIMISATION`**.

5. **Defer skops adoption to Stage 12.** Default: use joblib (consistent with Stage 4) for artefact serialisation at Stage 9.  Sklearn's own guidance sanctions pickle/joblib for single-author research tools.  Add a module-level warning docstring about untrusted-source loading.  Flag Stage 12 (serving) as the adoption point for skops.  **`PLAN POLISH`** (documentation addition only).

---

## Sources

All URLs accessed 2026-04-22.

1. MLflow file store source code: https://github.com/mlflow/mlflow/blob/master/mlflow/store/tracking/file_store.py
2. MLflow system tags source: https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py
3. MLflow backend store documentation (KTLO status, deprecation): https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/
4. MLflow filesystem backend deprecation notice: https://github.com/mlflow/mlflow/issues/18534
5. MLflow migrate-from-file-store documentation: https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store/
6. MLflow search_runs performance issue: https://github.com/mlflow/mlflow/issues/1902
7. MLflow search_runs silent 1000-run limit: https://github.com/mlflow/mlflow/issues/15814
8. MLflow lifecycle_stage FileStore bug: https://github.com/mlflow/mlflow/issues/8177
9. MLflow PR #18497 — switch default to SQLite: https://github.com/mlflow/mlflow/pull/18497
10. DVC internal files / cache structure: https://doc.dvc.org/user-guide/project-structure/internal-files
11. DVC project structure overview: https://doc.dvc.org/user-guide/project-structure
12. DVC artifacts in dvc.yaml: https://dvc.org/doc/user-guide/project-structure/dvcyaml-files
13. Mitchell et al. (2019), "Model Cards for Model Reporting": https://arxiv.org/abs/1810.03993
14. Sugimura & Hartley (2018), "Building a Reproducible Machine Learning Pipeline": https://arxiv.org/pdf/1810.04570
15. MLflow autolog documentation: https://mlflow.org/blog/mlflow-autolog
16. MLflow model registry documentation: https://mlflow.org/docs/latest/model-registry/
17. scikit-learn model persistence (joblib vs skops decision): https://scikit-learn.org/stable/model_persistence.html
18. CVE-2024-34997 Snyk entry (disputed): https://security.snyk.io/vuln/SNYK-PYTHON-JOBLIB-6913425
19. CVE-2024-34997 NVD entry: https://nvd.nist.gov/vuln/detail/CVE-2024-34997
20. skops persistence documentation: https://skops.readthedocs.io/en/stable/persistence.html
21. ZenML custom model registry interface: https://docs.zenml.io/stacks/stack-components/model-registries/custom
22. BentoML sklearn framework reference: https://docs.bentoml.org/en/latest/reference/frameworks/sklearn.html
23. W&B wandb.init documentation: https://docs.wandb.ai/ref/python/init/
24. Tales Marra, "Simple MLOps #2: Model Registry": https://medium.com/@talesmarra/simple-mlops-2-model-registry-39a106bb7daf
25. MLflow export-import tool (PyPI): https://pypi.org/project/mlflow-export-import/
26. MLflow export-import GitHub: https://github.com/mlflow/mlflow-export-import
