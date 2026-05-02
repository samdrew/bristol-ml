# Stage 16 — Model with REMIT features

## Goal

Extend the model training pipeline to consume REMIT-derived features produced
from Stage 13's bi-temporal event log, enriched by Stage 14's LLM extractor,
and compare the result against the Stage 11/12 best-model baseline.  Three
targets in priority order:

1. Ship `features/remit.py` — a vectorised, point-in-time-correct derivation
   function that collapses the REMIT event log to three hourly columns with no
   future leakage.
2. Ship the complete `with_remit` feature set: Pydantic schema, Hydra config,
   assembler extension (`assemble_with_remit` / `load_with_remit` /
   `WITH_REMIT_OUTPUT_SCHEMA`), and the training-entry arm in
   `train._resolve_feature_set`.
3. Ship the ablation notebook `notebooks/04_remit_ablation.ipynb` — a four-row
   metric table (without-REMIT baseline / with-REMIT current-state-only /
   with-REMIT full / NESO benchmark) that is the demo moment and the honest
   record of whether REMIT features help GB demand forecasting.

T6 (the two registered Stage 16 training runs using the real OpenAI extractor
on a CUDA GPU) is **deferred to the human's host**: it requires a CUDA device
and a populated `BRISTOL_ML_LLM_API_KEY`.  See the T6 host runbook below.

## What was built

- `conf/_schemas.py` — new `WithRemitFeatureConfig` (extends `FeatureSetConfig`
  with `forward_lookahead_hours: int = 24`, `include_forward_lookahead: bool =
  True`, and `extracted_parquet_filename: str | None = None`); new
  `FeaturesGroup.with_remit: WithRemitFeatureConfig | None = None` field.
- `conf/features/with_remit.yaml` — Hydra group file (`features` group);
  `name: with_remit`, `demand_aggregation: mean`, `forward_lookahead_hours: 24`,
  `include_forward_lookahead: true`, `extracted_parquet_filename: null`.  Selects
  the `with_remit` arm via `features=with_remit` at the CLI.
- `src/bristol_ml/features/remit.py` — new module.  Public surface:
  `REMIT_VARIABLE_COLUMNS` (typed column constant, 3 entries) and
  `derive_remit_features(remit_df, hourly_index, *, forward_lookahead_hours=24)
  -> pd.DataFrame`.  Pure derivation; no I/O.  Standalone CLI `python -m
  bristol_ml.features.remit`.  Module docstring names the bi-temporal contract
  and the leakage warning explicitly.
- `src/bristol_ml/llm/persistence.py` — new module (plan A5).  Public surface:
  `EXTRACTED_OUTPUT_SCHEMA` (11-column `pa.Schema`), `extract_and_persist(
  extractor, remit_df, *, output_path) -> Path`, `load_extracted(path) ->
  pd.DataFrame`, and `DEFAULT_OUTPUT_PATH`.  Standalone CLI `python -m
  bristol_ml.llm.persistence --cache {auto,refresh,offline} [--limit N]
  [--output PATH] [overrides...]`.
- `src/bristol_ml/features/assembler.py` — extended.  New exports:
  `WITH_REMIT_OUTPUT_SCHEMA` (59-column `pa.Schema`), `assemble_with_remit(cfg,
  *, cache) -> Path`, `load_with_remit(path) -> pd.DataFrame`.  All three added
  to `__all__`.
- `src/bristol_ml/train.py` — new `with_remit` arm in `_resolve_feature_set`;
  mutual-exclusivity invariant extended to the third feature set.
- `notebooks/04_remit_ablation.ipynb` — five-cell demo notebook (1
  title-markdown + 3 executable cells + 1 closing-discussion markdown).
  Generated from `scripts/_build_notebook_16.py`.  Ships with a runbook banner
  that prints when no `with_remit` runs are registered (T6 deferred).
- `scripts/_build_notebook_16.py` — notebook builder (mirrors
  `_build_notebook_15.py` shape).
- Tests — 11 new unit tests at `tests/unit/features/test_remit.py` and 4
  integration tests at `tests/integration/test_notebook_04.py`.  Stub-mode
  parquet round-trip and mutual-exclusivity guards in
  `tests/unit/llm/test_extractor_persistence.py`.

## Design choices made here

- **Delta-event aggregation, not per-hour `as_of()` calls.**  Each revision
  contributes a `+d` event at `window_from` and a `-d` event at `window_to`;
  cumsum + `merge_asof(direction="backward")` resolves all hours in one pass.
  O((n_revisions + n_hours) log) instead of O(n_hours × n_revisions).  Plan D9 /
  NFR-1.
- **`REMIT_VARIABLE_COLUMNS` owned by `features/remit.py`; re-exported from
  `assembler.py`.**  Same single-source-of-truth pattern as
  `CALENDAR_VARIABLE_COLUMNS`.  `WITH_REMIT_OUTPUT_SCHEMA` reads the constant
  verbatim so a column rename surfaces at `import` time, not at runtime.
- **Schema always carries `remit_unavail_mw_next_24h`; the flag lives in model
  config.**  Both registered Stage 16 runs use the same 59-column parquet; the
  `include_forward_lookahead=false` run omits the column from `feature_columns`
  only.  This prevents the on-disk schema from fragmenting across flag values
  (plan A2 / A4).
- **`assemble_with_remit` auto-runs the extractor when the parquet is absent.**
  Logs a WARNING naming the explicit CLI (`python -m bristol_ml.llm.persistence`)
  to use for a different extractor, then runs `extract_and_persist` inline under
  stub mode.  Keeps CI green without a pre-step; the human running the real path
  executes the explicit CLI first and the assembler reads the warm cache.  This
  deviated from plan A5's letter ("the assembler reads this parquet; it does not
  invoke the extractor inline") in favour of a better CI experience; the human
  clarification in the Ctrl+G review endorsed the trade-off.
- **Separate `llm/persistence.py` module (plan A5).**  Not an extension of
  `extractor.py`, preserving the features layer's freedom from OpenAI-SDK imports
  (plan OQ-5).  The assembler imports only `load_extracted` and
  `extract_and_persist` from `llm.persistence`; it never imports `Extractor` or
  `LlmExtractor` directly.

## What landed differently from the plan

- **T6 deferred (CUDA + OpenAI not available in the dev container).**  The two
  registered Stage 16 training runs (plan A3 — real-extractor required as a DoD
  gate) are a host-side step.  The notebook ships with a banner cell that detects
  the absence of `with_remit` runs in the registry and prints the T6 runbook
  commands rather than raising.  CI exercises the T7 nbconvert path under stub
  mode instead.
- **`assemble_with_remit` auto-runs extractor inline (deviation from plan A5
  letter).**  Plan A5 said "the assembler reads this parquet; it does not invoke
  the extractor inline".  The implementation added the auto-run fallback as a
  pragmatic CI affordance.  The warning log line preserves the separation of
  concerns at the operator level.
- **`python -m bristol_ml.features.assembler` does not dispatch on
  `features=with_remit`.**  The CLI calls `assemble()` (weather-only) directly;
  it was not extended to dispatch on the active feature set.  The T6 runbook
  below documents the correct in-process call pattern.
- **YAML comment in `conf/features/with_remit.yaml` cites 58 columns.**  The
  comment pre-dates the final schema and undercounts by one: the actual
  `WITH_REMIT_OUTPUT_SCHEMA` has 59 columns (the 55-column calendar prefix
  includes `holidays_retrieved_at_utc` at position 54).  The authoritative count
  lives in `assembler.py`; the YAML comment is informational only.

## What surprised us during implementation

- **Withdrawn-truncates-prior bug found during T2 stub-mode smoke.**  An early
  draft of `_per_mrid_validity` dropped `Withdrawn` rows before computing
  `tx_valid_to`, leaving prior revisions valid indefinitely after a withdrawal.
  The smoke test revealed the issue: a synthetic sequence `rev0:Active →
  rev1:Withdrawn` should leave rev0 contributing only until `rev1.published_at`,
  but the draft made rev0 contribute for all time.  The fix — apply `shift(-1)`
  over the full sorted log including `Withdrawn`, then drop `Withdrawn` rows at
  the end — is the Withdrawn-truncates-prior rule documented explicitly in the
  production code and the module guide.
- **Dtype-precision mismatch on `merge_asof`.**  The REMIT parquet round-trips
  at microsecond UTC (`timestamp[us, tz=UTC]`); the assembler's hourly grid is
  nanosecond UTC (`timestamp[ns, tz=UTC]`).  `pandas.merge_asof` is strict on
  dtype equality for the join key.  `_running_total` and `_lookup_at_grid` both
  call `.as_unit("ns")` on the timestamp series to normalise before the
  `merge_asof`.
- **Auto-run extractor choice in `assemble_with_remit`.**  The plan said the
  assembler reads a pre-existing parquet; a clean CI run had no pre-existing
  parquet.  Rather than fail loudly and require operators to run a pre-step, the
  assembler was given the fallback (with WARNING log) to auto-populate under stub
  mode.  The real-extractor path still needs the explicit `llm.persistence` CLI
  first.

## Observations from execution

- **11 unit tests + 4 integration tests pass under stub mode.**  The leakage
  test (`test_remit_features_do_not_use_future_revisions`) produces the
  clearest assertion of the bi-temporal correctness guarantee: a revision
  published after `t` does not appear in the features at `t`.
- **Notebook cell count: 5** (1 title-markdown + 3 code + 1 commentary markdown).
  The banner cell detects whether registered `with_remit` runs exist and either
  renders the four-row metric table or prints the T6 runbook.
- **No new runtime dependencies** were added at Stage 16.  `pandas.merge_asof`,
  `pyarrow`, `loguru`, and the existing model stack were already in lock.

## T6 host runbook

This section documents the exact commands to run on a CUDA host
(16 GB+ Ampere GPU recommended; reference: RTX 5090) to produce the two
registered Stage 16 runs that the ablation notebook renders.

Estimated cost for the OpenAI extraction step: a few USD for `gpt-4o-mini`
over a 7-year REMIT corpus (10 k–50 k events at approximately $0.0001 per
event).  Estimated training time: approximately 15 minutes per TCN run.

### Step 1 — Populate the REMIT cache

```bash
uv run python -m bristol_ml.ingestion.remit --cache refresh
```

This fetches the full REMIT history and writes
`data/raw/remit/remit.parquet`.

### Step 2 — Produce the real-extractor parquet

```bash
BRISTOL_ML_LLM_API_KEY=sk-... \
uv run python -m bristol_ml.llm.persistence \
    --cache auto \
    +llm=extractor \
    llm.type=openai \
    llm.model_name=gpt-4o-mini
```

Writes `data/processed/remit_extracted.parquet` (11 columns,
`EXTRACTED_OUTPUT_SCHEMA`).  The `--limit N` flag caps the corpus for a
test run before committing to the full pass.  The `model_id` column in
the output records which model performed the extraction for provenance.

### Step 3 — Build the `with_remit` feature parquet

The `python -m bristol_ml.features.assembler` CLI dispatches only on the
weather-only path.  Invoke `assemble_with_remit` directly:

```python
from bristol_ml.config import load_config
from bristol_ml.features.assembler import assemble_with_remit

cfg = load_config(overrides=["features=with_remit"])
out = assemble_with_remit(cfg, cache="auto")
print(out)  # data/features/with_remit.parquet
```

Or as a one-liner:

```bash
uv run python -c "
from bristol_ml.config import load_config
from bristol_ml.features.assembler import assemble_with_remit
cfg = load_config(overrides=['features=with_remit'])
print(assemble_with_remit(cfg, cache='auto'))
"
```

Because `data/processed/remit_extracted.parquet` is now warm (Step 2),
the assembler reads it directly without triggering the stub fallback.

### Step 4 — Train the two registered runs

Run (a) — with the forward-looking column (`include_forward_lookahead=true`):

```bash
uv run python -m bristol_ml.train \
    features=with_remit \
    model=nn_temporal \
    features.with_remit.include_forward_lookahead=true
```

Run (b) — without the forward-looking column (`include_forward_lookahead=false`):

```bash
uv run python -m bristol_ml.train \
    features=with_remit \
    model=nn_temporal \
    features.with_remit.include_forward_lookahead=false
```

Both runs use the production TCN recipe (`seq_len=168, num_blocks=8,
channels=128`) per plan A1.  After each run completes, verify the registry:

```bash
uv run python -c "
from bristol_ml.registry import list_runs
for r in list_runs(feature_set='with_remit'): print(r)
"
```

Both runs should appear; their `feature_columns` lists should differ by
exactly `remit_unavail_mw_next_24h`.

### Step 5 — Re-execute the ablation notebook

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/04_remit_ablation.ipynb
```

Or in JupyterLab / VS Code: open and run all cells.  The notebook's
second code cell loads the three registered run IDs (prior baseline +
two Stage 16 runs), asserts split-config equality across all three, and
renders the four-row metric table.

## Deferred

- **T6 registered runs** — see T6 host runbook above.  Requires CUDA + OpenAI
  API key; not available in the dev container.  The notebook banner prints the
  runbook commands at demo time until the human has populated the registry.
- **`extractor_mode` provenance scalar in the registered-run sidecar (NFR-6).**
  The notebook's commentary cell promises that the registered runs' sidecars
  carry an ``extractor_mode: "stub" | "real"`` flag so a viewer of the metric
  table can read which extractor produced the features.  The current
  ``train.py`` does not inject this flag — it passes the standard arguments
  to ``registry.save`` and the model's ``ModelMetadata.hyperparameters``
  bag does not yet carry the LLM-mode information.  Closing this gap is a
  prerequisite to running T6: before the host runs the two TCN training
  commands, ``train.py`` should be extended to read ``BRISTOL_ML_LLM_STUB``
  (or inspect ``cfg.llm.type``) and inject ``{"extractor_mode": ...}`` into
  the model's hyperparameters via ``model_copy`` so the sidecar reflects
  the truth.  The notebook already reads from
  ``run["hyperparameters"]["extractor_mode"]`` defensively (returns "—" when
  absent), so the gap is non-fatal for the demo path but flagged here.
- **Multi-horizon REMIT features.**  The current `remit_unavail_mw_next_24h`
  column has a hard-coded 24-hour horizon name (configurable via
  `forward_lookahead_hours`, but the column name does not change).  A week-ahead
  or 48-hour window would need a new column to avoid schema breakage.
- **Hydra-callable assembler CLI for `with_remit`.**  The
  `python -m bristol_ml.features.assembler` CLI does not dispatch on
  `features=with_remit`; this is documented in `features/CLAUDE.md` and in this
  retro.  Wiring the CLI to dispatch based on the active feature group is a
  candidate future refactor.

## References

- Plan — [`docs/plans/completed/16-model-with-remit.md`](../../plans/completed/16-model-with-remit.md)
- Research artefacts:
  - Requirements — [`docs/lld/research/16-model-with-remit-requirements.md`](../research/16-model-with-remit-requirements.md)
  - Codebase map — [`docs/lld/research/16-model-with-remit-codebase.md`](../research/16-model-with-remit-codebase.md)
  - Domain research — [`docs/lld/research/16-model-with-remit-domain.md`](../research/16-model-with-remit-domain.md)
  - Scope Diff — [`docs/lld/research/16-model-with-remit-scope-diff.md`](../research/16-model-with-remit-scope-diff.md)
- Layer doc — [`docs/architecture/layers/features.md`](../../architecture/layers/features.md)
- Module guides — [`src/bristol_ml/features/CLAUDE.md`](../../../src/bristol_ml/features/CLAUDE.md)
  and [`src/bristol_ml/llm/CLAUDE.md`](../../../src/bristol_ml/llm/CLAUDE.md)
- Upstream retros — [`docs/lld/stages/13-remit-ingestion.md`](13-remit-ingestion.md),
  [`docs/lld/stages/14-llm-extractor.md`](14-llm-extractor.md),
  [`docs/lld/stages/15-embedding-index.md`](15-embedding-index.md)
