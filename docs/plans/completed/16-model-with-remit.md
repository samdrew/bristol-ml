# Plan - Stage 16: Model with REMIT features

**Status:** `approved` — Ctrl+G 2026-05-02 bound all five A-rows; Phase 2 implementation unblocked.

**Ctrl+G dispositions (2026-05-02):**
- **A1 (=D10) — bound:** default retraining target is `nn_temporal` (TCN), under the production CUDA recipe (`seq_len=168, num_blocks=8, channels=128`). The feature-derivation path is intentionally model-agnostic — any prior model class can be retrained against `features=with_remit` via the same Hydra entry point; TCN is the *default*, not an exclusive binding.
- **A2 (=D3, ties D17/AC-6) — bound:** include the forward-looking `remit_unavail_mw_next_24h` column. AC-6 binds. Notebook gains a quantitative with/without-`next_24h` comparison so the marginal value of the forward-looking signal is visible (notebook table grows from three rows to four — see §5).
- **A3 (=D11) — bound:** **real-extractor registered run is a stage-DoD gate.** The Stage 16 with-REMIT registered runs (T6) use the real LLM extractor, not the stub. The extractor persistence parquet (T3) is produced once via the real extractor for the historical REMIT corpus and re-used across the two T6 training runs. CI continues to use the stub (NFR-2) — this is the offline-clone path, not the registered-artefact path.
- **A4 — bound:** one model *family* retrained (TCN per A1), but two registered with-REMIT runs (one with `next_24h`, one without) so A2's comparison is reproducible from the registry. Stage assumes a moderate CUDA host (16 GB+ Ampere or later — actual: RTX 5090); CPU recipes are no longer the demo default. R-4 dropped; new R-8 added (CUDA host dependency).
- **A5 (=D5) — bound:** separate ingestion-style step writes `data/processed/remit_extracted.parquet` via a new module/CLI; the assembler reads this parquet and does not invoke the extractor inline. Consistent with the project's strong modular separation.

**Intent:** [`docs/intent/16-model-with-remit.md`](../../intent/16-model-with-remit.md)
**Upstream stages shipped:** Stages 0-15 (foundation -> ingestion -> features -> six model families -> enhanced evaluation -> registry -> MLP -> TCN -> serving -> REMIT bi-temporal store -> LLM feature extractor -> embedding index).
**Downstream consumers:** Stage 17 (price target — may pick up REMIT features for that target). The ablation result is the load-bearing reference for any future "does textual context help" discussion.
**Baseline SHA:** `440670c` (tip of `main` after Stage 15 merge, PR #16).

**Discovery artefacts produced in Phase 1:**

- Requirements - [`docs/lld/research/16-model-with-remit-requirements.md`](../../lld/research/16-model-with-remit-requirements.md)
- Codebase map - [`docs/lld/research/16-model-with-remit-codebase.md`](../../lld/research/16-model-with-remit-codebase.md)
- Domain research - [`docs/lld/research/16-model-with-remit-domain.md`](../../lld/research/16-model-with-remit-domain.md)
- Scope Diff - [`docs/lld/research/16-model-with-remit-scope-diff.md`](../../lld/research/16-model-with-remit-scope-diff.md) (fourth artefact; `@minimalist` critique + lead disposition recorded in §1 below)

**Pedagogical weight.** Intent §"Demo moment" names a single moment: a three-row metric table — best model without REMIT, same model with REMIT, NESO benchmark — with honest commentary that reads correctly whether the contribution is positive, negligible, or negative. The lesson is as much about the rigour of the bi-temporal join (no leakage, even in the presence of revisions) as about the numerical result. The intent explicitly endorses a null finding as a successful outcome, because no published study tests REMIT for demand forecasting (domain research §3); the ablation itself is the contribution.

**Architectural weight.** This is the first stage that consumes Stage 13's `as_of()` primitive at training-time scale. It establishes the project's pattern for **point-in-time correct** feature derivation from an event-log source — the pattern Stage 17 (price) will inherit and any future event-driven feature stage will follow. The pattern: sort-once + vectorised as-of join into the hourly index, never per-row `as_of()` calls.

**Upstream-data sharp edges.**
- The Stage 14 extractor returns `list[ExtractionResult]` in memory; **there is no on-disk persistence** of extraction output today (codebase H1). Stage 16 must decide whether to add a one-time persistence step (parquet sidecar) or to persist via the assembler's own caching. **A5 below.**
- The Stage 13 stream endpoint frequently leaves `message_description` NULL (`src/bristol_ml/ingestion/remit.py:431-436`). Stage 14 already mitigated this for extraction; Stage 16 inherits the consequence: under stub mode and on real data with sparse descriptions, REMIT columns will be dominated by zeros (codebase H3). The notebook commentary (AC-5) is the right place to surface this honestly.
- The two prior empirical "best model" claims disagree across protocols: Stage 11's predict-only single-holdout puts `nn_temporal` first (MAE 3768 MW); Stage 7's rolling-origin harness puts `sarimax` first (MAE 1730 MW). The intent says "best-performing model" without naming the protocol. **A1 below.**

---

## 1. Decisions

The decision set is twenty rows. To respect reviewer time, the lead split them into:

- **§1.A - five rows that genuinely require Ctrl+G disposition** before Phase 2 begins.
- **§1.B - fifteen rows whose defaults bind on the strength of intent + research convergence**.

The full Scope Diff is at [`docs/lld/research/16-model-with-remit-scope-diff.md`](../../lld/research/16-model-with-remit-scope-diff.md). The `@minimalist` flagged five items (D3, D5, D9, D13, D16/NFR-8, D17/AC-6) for cuts or softening. The lead's draft dispositions, before Ctrl+G:

- **Cuts kept:** D9's wall-time acceptance test (the highest-leverage cut) — vectorisation stays, the bound-asserting test goes; D13's separate end-to-end ablation test — replaced with notebook source-inspection per Stage 11 precedent; NFR-8 softened to INFO-only with no exact-text assertions.
- **Awaiting Ctrl+G:** D3 column count, D5 persistence approach, D17/AC-6 forward-looking column status — all bind off A2.

### §1.A - Ctrl+G dispositions

| # | Decision | Final disposition | Resolution rationale |
|---|---|---|---|
| **A1 (=D10)** | **Which model gets retrained on the REMIT-enriched feature set?** | **`nn_temporal` (TCN) is the default, under the production CUDA recipe** (`seq_len=168, num_blocks=8, channels=128`). The feature-derivation path is **model-agnostic** — `_resolve_feature_set(cfg)` returns a feature DataFrame, not a model decision; any prior model class can be retrained against `features=with_remit` via the existing Hydra training entry. The Stage 16 *registered* run is the TCN; other models remain compatible. | Bound 2026-05-02. The user's framing — "default model to retrain, but I don't see that it should be a strict either/or" — matches the existing architecture: the new `with_remit` arm in `train.py::_resolve_feature_set` is keyed on the feature-set config, not on the model. No additional plumbing is needed to keep the path open for SARIMAX, linear, etc.; documenting this explicitly in `features/CLAUDE.md` is sufficient. |
| **A2 (=D3, ties D17/AC-6)** | **Which REMIT columns ship in the initial schema, and how is the value of the forward-looking column made visible?** | **Three columns** in `WITH_REMIT_OUTPUT_SCHEMA`: `remit_unavail_mw_total`, `remit_active_unplanned_count`, `remit_unavail_mw_next_24h`. AC-6 binds. The notebook table grows from three rows to **four** — best model w/o REMIT, best model with REMIT (current-state only, no `next_24h`), best model with REMIT (full), NESO benchmark — so the marginal contribution of the forward-looking signal is read directly from the table. | Bound 2026-05-02. The user's request to "provide comparison between with/without in notebook output, so that its value can be evaluated easily" is most cleanly expressed as a fourth row in the metric table, which means **two with-REMIT registered runs** are produced (one with `next_24h` enabled in the model's `feature_columns`, one without). This is consistent with A4's "single model family, multiple registered runs of it." Affects T6 and §5 notebook structure. |
| **A3 (=D11)** | **Is the registered Stage 16 ablation artefact required to use the real LLM extractor, or is the stub-default sufficient for the stage DoD?** | **Real-extractor required.** Both T6 registered runs consume features derived from a single one-shot real-extractor pass over the historical REMIT corpus, persisted to `data/processed/remit_extracted.parquet` per A5. CI remains on the stub path (NFR-2). The notebook (AC-5) records the `extractor_mode` provenance flag (NFR-6) so a viewer reads the table knowing which extractor produced the features. | Bound 2026-05-02. The user's reading aligns with the strict interpretation of intent §Points ("for the registered artefact, the real extractor is probably right"). Adds a one-shot OpenAI extraction cost to the stage DoD; otherwise no architectural change relative to the stub-only path. |
| **A4** | **How many models get retrained — one or two?** | **One model family (TCN per A1), trained twice on the with-REMIT feature set** — once including `next_24h`, once excluding it (per A2). Both registered runs land in the registry. The Stage assumes a moderate CUDA host (16 GB+ Ampere or later; reference: RTX 5090); CPU-only recipes are dropped from the demo path. | Bound 2026-05-02. The user's CUDA-host assumption removes the budget constraint that originally argued for "single model"; it does not change the intent's "single model family" framing. Two runs of the same family give A2 the quantitative comparison it asks for without expanding scope to a multi-family ablation. |
| **A5 (=D5)** | **Where does the Stage 14 extractor output get persisted, and by which module?** | **Separate ingestion-style step.** New module `src/bristol_ml/llm/persistence.py` (or extension of `extractor.py` with a top-level CLI) writes `data/processed/remit_extracted.parquet` via `_atomic_write` from `ingestion/_common.py`. Schema is keyed on `(mrid, revision_number)` and mirrors `ExtractionResult`. The assembler reads this parquet; it does not invoke the extractor inline. Stub mode auto-produces a stub-extraction parquet (zero cost). | Bound 2026-05-02. The user explicitly cites "consistent with architecture and goals (strong modular separation in architecture)." The separate step replicates the Stage 13 fetch+load pattern (codebase pattern §1) and keeps the feature layer free of LLM-layer imports (codebase OQ-5). |

### §1.B - Fifteen decisions that bind on default

The Evidence column cites the artefact that resolved each decision. Tags from the Scope Diff: most are `RESTATES INTENT`, two are `HOUSEKEEPING`. No engagement needed unless something looks wrong.

| # | Decision | Default | Tag | Evidence |
|---|---|---|---|---|
| **D1** | New module `src/bristol_ml/features/remit.py` — pure derivation, no I/O against external APIs (mirrors `calendar.py` shape: derivation function + `REMIT_VARIABLE_COLUMNS` constant + module docstring naming the bi-temporal contract). | RESTATES INTENT | Intent §Scope; AC-1, AC-2; codebase pattern §1. |
| **D2** | Schema constants: `REMIT_VARIABLE_COLUMNS` in `features/remit.py`; `WITH_REMIT_OUTPUT_SCHEMA` in `features/assembler.py` (extends `CALENDAR_OUTPUT_SCHEMA` prefix with the A2-bound columns). | RESTATES INTENT | AC-2; codebase §Exact-schema load(). |
| **D4** | Assembler extension: `assemble_with_remit(cfg, *, cache) -> Path` and `load_with_remit(path) -> pd.DataFrame` in `assembler.py`. Mirrors `assemble_calendar`; cannot delegate due to mutual-exclusivity (codebase H5). Both added to `__all__`. | RESTATES INTENT | AC-2; codebase H5, H7. |
| **D6** | Pydantic schema: new `WithRemitFeatureConfig` (extends `FeatureSetConfig`) in `conf/_schemas.py`; new `with_remit: WithRemitFeatureConfig | None = None` field on `FeaturesGroup`. Lands in the same commit as D7. | RESTATES INTENT | AC-2; DESIGN.md §2.1.4; codebase pattern §3. |
| **D7** | Hydra group file: `conf/features/with_remit.yaml`. Sets `name: with_remit`, the column set per A2, and the forward-looking window width (24 h) as a YAML field per NFR-5. | RESTATES INTENT | AC-2; codebase pattern §3. |
| **D8** | Training entry-point extension: new `with_remit` arm in `train.py::_resolve_feature_set`. Same mutual-exclusivity invariant as the existing arms (`if X is not None and Y is None and Z is None`). Error message names `features=with_remit`. | RESTATES INTENT | AC-3; codebase §4. |
| **D9** | **Vectorised bi-temporal aggregation** using a single `merge_asof` over the REMIT log sorted by `published_utc`, rather than per-hour `as_of()` calls. **No wall-time acceptance test** (cut by Scope Diff highest-leverage); correctness is asserted by AC-1's leakage test, performance is left as an implementation choice. | RESTATES INTENT (test-bound cut per Scope Diff) | AC-1; codebase H2; Scope Diff D9 (highest-leverage cut). |
| **D12** | Ablation notebook `notebooks/04_remit_ablation.ipynb` (numbered after `03_feature_assembler.ipynb`; the model-stage notebooks reset numbering at each stage per `notebooks/15_embedding_index.ipynb` precedent). Three executable cells: load baseline registry run + load Stage 16 registry run + render three-row metric table; one AC-5 commentary markdown cell that names the price-vs-demand asymmetry. Notebook builder script `scripts/_build_notebook_16.py` mirroring Stage 15's pattern. | RESTATES INTENT | Intent §Demo moment; AC-5; codebase §8 (notebook is thin); D14 below. |
| **D13** | **No standalone end-to-end ablation integration test** (cut by Scope Diff D13). AC-4 ("ablation reproducible from the registry") is verified by source-inspection of `_build_notebook_16.py`: the test asserts the notebook builder calls `registry.load(...)` and `model.predict(...)` and does not call `model.fit(...)`. This matches the Stage 11 precedent and avoids forcing both registered runs into CI. | PLAN POLISH (cut per Scope Diff) | AC-4; Scope Diff D13. |
| **D14** | Stage hygiene: `src/bristol_ml/features/CLAUDE.md` updated; `docs/architecture/layers/features.md` extended with the REMIT-enriched feature set; `docs/lld/stages/16-model-with-remit.md` retrospective; `CHANGELOG.md` `[Unreleased]` entry; plan moved from `active/` to `completed/` in the final commit. | HOUSEKEEPING | CLAUDE.md §Stage hygiene. |
| **D15** | Task ordering: schema/config (D6+D7) → derivation module (D1+D2) → assembler extension (D4) → extractor persistence per A5 → training run (D8 + registered Stage 16 run) → notebook (D12) → docs (D14). | HOUSEKEEPING | Internal lead decision. |
| **D16** | Test fixtures: `tests/fixtures/features/remit_tiny.parquet` — 5–10 hourly rows + a synthetic REMIT log with at least one revision after T (for the leakage test) and at least one event with future `effective_from` (for the next-24h test). Programmatically generated to keep the schema in sync with `OUTPUT_SCHEMA` per the existing `test_assembler_calendar.py` pattern. | RESTATES INTENT | DESIGN.md §2.1.7; codebase §6. |
| **D17** | Tests: `tests/unit/features/test_remit.py` (six tests: leakage, schema conformance, zero-handling, forward-looking, no-NaN invariant, standalone-module exit-zero) and `tests/unit/features/test_assembler_with_remit.py` (mirrors `test_assembler_calendar.py`: load schema rejection, round-trip, mutual-exclusivity in the resolver). | RESTATES INTENT | AC-1, AC-2, AC-7, AC-8, AC-9. |
| **D18** | NFR-8 softened: `loguru` INFO line per assembly call (active-event hours, zero-event hours, forward-window hits, total rows). **No WARNING-on-short-cache assertion**, no exact-message-text test (per Scope Diff softening). The structured-record content is convention only. | PLAN POLISH (softened per Scope Diff) | DESIGN.md §2.1; Scope Diff NFR-8. |
| **D19** | Dependencies: **none new at runtime.** `pandas.merge_asof`, `pyarrow`, `loguru`, the existing model dependency stack — all already in lock. The extractor persistence step (A5) reuses `pyarrow` parquet write + `_atomic_write` from `ingestion/_common.py`. | RESTATES INTENT | Codebase §1, §5. |
| **D20** | Live-path cassette: **none**. The data sources Stage 16 consumes are the Stage 13 REMIT parquet (already produced under stub) and the Stage 14 extractor (stub-default by D11/A3). No HTTP at training time. | RESTATES INTENT | DESIGN.md §2.1.3; A3 (when bound). |

### Non-functional requirements

| # | NFR | Source |
|---|---|---|
| **NFR-1** | Bi-temporal correctness is structurally enforced. The derivation module's first step on the REMIT log is a `published_utc <= t` filter (vectorised via merge_asof over the sorted log). Module docstring states explicitly that bypassing this filter exposes the caller to leakage. | AC-1; intent §Points; domain research §1. |
| **NFR-2** | Stub-first for CI; real-extraction artefact (when produced) is a separate registered run, not gated on. `BRISTOL_ML_LLM_STUB=1` is the CI default. | AC-4; DESIGN.md §2.1.3. |
| **NFR-3** | Idempotent assembly. `_atomic_write` from `ingestion/_common.py` for both the extractor-persistence parquet (A5) and the `assemble_with_remit` output. Re-running with identical inputs produces a byte-identical output. | DESIGN.md §2.1.5; codebase §5. |
| **NFR-4** | Schema typed and documented. `WITH_REMIT_OUTPUT_SCHEMA` is a `pyarrow.schema` constant; `load_with_remit` rejects frames with missing or extra columns; column names are `snake_case` and do not collide with `CALENDAR_OUTPUT_SCHEMA`. The schema constant is importable from `bristol_ml.features.assembler` without triggering ingestion or extraction. | DESIGN.md §2.1.2; AC-2. |
| **NFR-5** | Configuration in YAML. The forward-looking window width (24 h) and the column subset (per A2) live in `conf/features/with_remit.yaml`, not as code constants. | DESIGN.md §2.1.4. |
| **NFR-6** | Reproducibility of the registered retraining run. Sidecar records: `feature_set: with_remit`, the column-name list (schema version), `extractor_mode: stub | real`, git SHA (already produced by registry), and the full split config. | DESIGN.md §2.1.6; AC-4. |
| **NFR-7** | Notebook is thin. `notebooks/04_remit_ablation.ipynb` imports from `bristol_ml.registry` and `bristol_ml.evaluation.benchmarks`; it does not reimplement aggregation, bi-temporal filtering, or metric computation. | DESIGN.md §2.1.8; AC-10. |
| **NFR-8** | Loguru observability (convention only). INFO record per assembly call covering active-event hours, zero-event hours, forward-window hits, row count. **No WARNING-on-short-cache, no exact-text assertions** (softened per Scope Diff). | DESIGN.md §2.1; Scope Diff NFR-8 (PLAN POLISH softened). |

---

## 2. Architecture sketch

```
src/bristol_ml/features/
├── remit.py                  # NEW. Public surface:
│                             #   - REMIT_VARIABLE_COLUMNS (pyarrow fields)
│                             #   - derive_remit_features(remit_df, extracted_df,
│                             #         hourly_index, *, horizon_hours=24) -> pd.DataFrame
│                             # First step: vectorised published_utc <= t filter via
│                             # merge_asof on the sorted log; second step: hourly
│                             # aggregation; third step: forward-window aggregation
│                             # by valid-time filter at each t. No I/O.
└── assembler.py              # EXTENDED. Adds:
                              #   - WITH_REMIT_OUTPUT_SCHEMA (pyarrow schema)
                              #   - assemble_with_remit(cfg, *, cache) -> Path
                              #   - load_with_remit(path) -> pd.DataFrame
                              # Both added to __all__.

src/bristol_ml/llm/
└── extractor.py              # EXTENDED per A5. Either:
                              #   (a) New top-level entry that runs extract_batch over
                              #       the loaded REMIT log and writes
                              #       data/processed/remit_extracted.parquet
                              #       via _atomic_write, OR
                              #   (b) New sibling module bristol_ml/llm/persistence.py
                              # Bound at A5 disposition.

conf/
├── _schemas.py               # EXTENDED. New WithRemitFeatureConfig; new
│                             # with_remit field on FeaturesGroup.
└── features/
    └── with_remit.yaml       # NEW. Column subset per A2; horizon_hours: 24;
                              # name: with_remit.

notebooks/
└── 04_remit_ablation.ipynb   # NEW (built from scripts/_build_notebook_16.py).
                              # Three executable cells (registry-load + table) +
                              # one AC-5 commentary markdown cell.

scripts/
└── _build_notebook_16.py     # NEW. Mirrors _build_notebook_15.py.

tests/
├── unit/features/
│   ├── test_remit.py                   # NEW. Six tests (D17).
│   └── test_assembler_with_remit.py    # NEW. Mirrors test_assembler_calendar.py.
├── unit/llm/
│   └── test_extractor_persistence.py   # NEW per A5. Stub-mode parquet round-trip.
└── fixtures/features/
    └── remit_tiny.parquet              # NEW (programmatically generated).
```

**Bi-temporal join shape** (the load-bearing pattern):

```python
# Inside derive_remit_features, in pseudocode:
remit_sorted = remit_df.sort_values("published_utc")
hourly = pd.DataFrame({"timestamp_utc": hourly_index})
# Step 1 — transaction-time filter via vectorised merge_asof:
joined = pd.merge_asof(
    hourly,
    remit_sorted,
    left_on="timestamp_utc",
    right_on="published_utc",
    direction="backward",            # only revisions known at t are visible
    tolerance=None,                  # no horizon limit on visibility
)
# Step 2 — valid-time filter for "active at t":
active_mask = (joined["effective_from"] <= joined["timestamp_utc"]) & (
    joined["effective_to"].isna() | (joined["effective_to"] > joined["timestamp_utc"])
)
# Step 3 — forward-window aggregation (separate pass per A2's third column):
forward_mask = (joined["effective_from"] >= joined["timestamp_utc"]) & (
    joined["effective_from"] < joined["timestamp_utc"] + pd.Timedelta(hours=horizon_hours)
)
```

The merge_asof at step 1 replaces the per-hour `as_of()` loop and is what NFR-1's "vectorised" claim points at.

---

## 3. Risks

| # | Risk | Mitigation |
|---|---|---|
| **R-1** | The chosen retraining model (per A1) silently picks up provenance scalar columns (e.g. `neso_retrieved_at_utc`) as features and produces an unstable result. | Set `feature_columns` explicitly in the model's YAML override (codebase H6). Add a unit test that asserts the resolved feature column list excludes `*_retrieved_at_utc` columns. |
| **R-2** | Vectorised merge_asof loses semantics relative to per-row `as_of()` for the multi-revision case (the `as_of(df, t)` primitive returns the *latest* revision per `mrid`, not just the most recent published row). | The merge_asof step must be applied **per `mrid`-group** so that the "latest revision known at t" semantics is preserved (groupby + apply, or pre-pivot). Acceptance test: synthetic frame with two revisions of the same `mrid`, one published before and one after T; assert the derived features at T reflect the earlier revision only. |
| **R-3** | The Stage 14 stub extractor returns a fixed sentinel for unmapped events; under stub mode, REMIT columns will be heavily dominated by stub-quality values and the ablation result is misleading. | Notebook commentary cell (AC-5) explicitly names this. The cell prints the stub/real flag from the registry sidecar (NFR-6) so a meetup viewer knows what they are looking at. No further mitigation needed — this is the expected behaviour and is part of the lesson the stage teaches. |
| **R-4** | *(removed by A4 — the CUDA-host assumption replaces the laptop-CPU constraint.)* | — |
| **R-8** | The registered runs depend on a CUDA host to reproduce; a meetup machine without a GPU cannot retrain end-to-end. | The trained model artefacts (skops via Stage 12) are reloadable on CPU for `predict()`; the notebook (NFR-7, AC-10) loads from registry and never calls `fit()`. The CUDA dependency is reproduction-only, not demo-time. Document the CUDA-host requirement in `docs/lld/stages/16-model-with-remit.md` and the registered-run sidecar (NFR-6). |
| **R-5** | The REMIT cache covers a narrower historical window than the demand/weather cache, so the assembler produces zero-filled REMIT columns for the early years and the model effectively trains on weather-only for those years. | Inner-join on the demand timeline (existing convention from Stage 3); REMIT columns default to zero for hours outside the REMIT cache window; INFO log records the coverage gap (NFR-8, no WARNING per softened scope). The notebook prints the coverage fraction so a viewer can see what they are working with. |
| **R-6** | The two registered runs in the ablation (with-REMIT and without-REMIT) used different rolling-origin split configurations and the comparison is invalid. | The Stage 16 run loads its split config from the prior registered run's sidecar (or the test asserts the two configs match). Cross-stage comparability is checked by an explicit assertion in the notebook before the metric table is rendered (one line: `assert prior_split == this_split`). |
| **R-7** | The `with_remit` Hydra YAML is added to `config.yaml`'s defaults list before the `WithRemitFeatureConfig` schema field is added, breaking CI on partial state (codebase H4). | D6 + D7 land in a single commit; the task list (T-tasks below) makes this explicit. |

---

## 4. Acceptance criteria (intent restatement)

- **AC-1.** Bi-temporal correctness: each row's REMIT features reflect only information published by that row's timestamp. (Test: revision-after-T leakage test on the fixture frame.)
- **AC-2.** Assembler produces the REMIT-enriched feature set via configuration switch. (Test: `python -m bristol_ml.features.assembler features=with_remit` exits 0; load schema rejects mismatched parquet.)
- **AC-3.** Retraining uses the same splits and metrics as prior stages. (Test: notebook asserts split config equality before rendering metric table.)
- **AC-4.** Ablation is reproducible from the registry. (Verified by source-inspection of `_build_notebook_16.py` per D13.)
- **AC-5.** Notebook commentary honestly describes the result. (Verified by source-inspection: markdown cell length + presence of "price"/"demand"/"effect"/"null" — see requirements artefact AC-5 testable form.)

Plus implicit ACs from the requirements artefact §3, kept where the Scope Diff judged them `RESTATES INTENT`:

- **AC-7.** No NaN in REMIT columns; zero-event hours produce zero-valued features.
- **AC-8.** `python -m bristol_ml.features.remit` runs standalone and exits 0.
- **AC-9.** At least one smoke test on the public surface of `features.remit`.
- **AC-10.** Notebook executes top-to-bottom under `BRISTOL_ML_LLM_STUB=1`.

**AC-6 (forward-looking column required)** binds off A2 — if the human keeps the third column, AC-6 binds; if they drop it, AC-6 is removed.

---

## 5. Notebook structure

`notebooks/04_remit_ablation.ipynb` — three executable cells plus a leading title-markdown cell and a trailing AC-5 commentary cell:

1. **Bootstrap.** Print the three registered run IDs (without-REMIT baseline, with-REMIT excluding `next_24h`, with-REMIT full), plus the NESO benchmark identifier.
2. **Load and predict.** `registry.load(...)` for all three model runs, NESO benchmark predictions for the same evaluation window. Assert split-config equality across all three runs (R-6).
3. **Four-row metric table** (per A2). Render MAE, MAPE, RMSE, WAPE for:
   1. Best model without REMIT (Stage-11/12 baseline).
   2. Best model with REMIT, **excluding** `remit_unavail_mw_next_24h` (current-state-only signal).
   3. Best model with REMIT, **including** `remit_unavail_mw_next_24h` (full forward-looking signal).
   4. NESO benchmark.

   Row 2-vs-1 isolates the marginal value of the current-state REMIT columns; row 3-vs-2 isolates the marginal value of the forward-looking column. Per A2: "so that its value can be evaluated easily."
4. **(markdown — AC-5)** Commentary cell. Names the result honestly across all three comparisons (does REMIT help at all? does the forward-looking signal add anything beyond current-state?). Names the price-vs-demand asymmetry (domain research §3). Names the stub/real extractor flag and its caveat. Reads correctly whether REMIT helps, hurts, or is null on either signal.

Total: 5 cells (1 title-markdown + 3 code + 1 closing-discussion markdown). The demo moment is small but the table now decomposes the REMIT contribution into its two architecturally distinct parts.

---

## 6. Task list (for Phase 2)

| Task | Description | Acceptance test |
|---|---|---|
| **T0** | *(complete 2026-05-02 — all five A-rows bound; status `approved`.)* | — |
| **T1** | Schema + config: `WithRemitFeatureConfig` in `conf/_schemas.py`; `conf/features/with_remit.yaml`. Single commit. (D6 + D7.) | `tests/unit/test_config.py` extension: `with_remit` group composes; `extra="forbid"` rejects unknown fields. |
| **T2** | `features/remit.py`: `REMIT_VARIABLE_COLUMNS`, `derive_remit_features`, module docstring + standalone `__main__` block. Vectorised merge_asof per-`mrid` (R-2). (D1 + D2 + D9.) | `tests/unit/features/test_remit.py`: leakage test, schema conformance, zero-handling, forward-looking, no-NaN, multi-revision-per-mrid. |
| **T3** | A5-bound extractor persistence: a `data/processed/remit_extracted.parquet` writer (location bound at A5). Stub mode produces a stub-extraction parquet automatically. (D5/A5.) | `tests/unit/llm/test_extractor_persistence.py`: stub-mode parquet round-trip; idempotent re-write. |
| **T4** | Assembler extension: `WITH_REMIT_OUTPUT_SCHEMA`, `assemble_with_remit`, `load_with_remit`. Both in `__all__`. (D4.) | `tests/unit/features/test_assembler_with_remit.py`: round-trip; load schema rejects calendar parquet; mutual-exclusivity at the resolver. |
| **T5** | Training entry: `with_remit` arm in `train.py::_resolve_feature_set`. (D8.) | Existing config tests extended; resolver error message names `features=with_remit`. |
| **T6** | **Two** registered Stage 16 runs (per A4): train TCN under `features=with_remit` (a) with `feature_columns` including `remit_unavail_mw_next_24h` and (b) with `feature_columns` excluding it. Both use the production CUDA recipe (per A1). Record `feature_set`, `extractor_mode`, split config, git SHA, and the `feature_columns` list (NFR-6). | Run sidecars contain the required fields; `registry.list_runs(feature_set="with_remit")` returns both runs; `feature_columns` differs between them by exactly one column. |
| **T7** | Notebook: `notebooks/04_remit_ablation.ipynb` via `scripts/_build_notebook_16.py`. Five cells per §5. (D12.) | `tests/integration/test_notebook_04.py`: nbconvert smoke test under `BRISTOL_ML_LLM_STUB=1`. Source-inspection test (D13): `_build_notebook_16.py` calls `registry.load` + `model.predict`, never `model.fit`. |
| **T8** | Stage hygiene: `features/CLAUDE.md`; `architecture/layers/features.md` extension; `lld/stages/16-model-with-remit.md` retrospective; `CHANGELOG.md`; plan move. (D14.) | docs review at Phase 3. |

---

## 7. Dependencies

**No new runtime dependencies.** `pandas.merge_asof`, `pyarrow`, `loguru`, the existing model stack — all already in lock.

**No new dev dependencies.** `pytest`, `pytest-mock` already present.

---

## 8. Exit checklist

- [x] Ctrl+G dispositions for A1–A5 recorded at the top of this plan (all bound 2026-05-02); status flipped to `approved`.
- [ ] `uv run pytest -q` clean (target: ~8–10 new tests across `tests/unit/features/`, `tests/unit/llm/`, `tests/integration/`).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [ ] `uv run pre-commit run --all-files` clean.
- [ ] `python -m bristol_ml.features.remit` exits 0 under stub mode.
- [ ] `python -m bristol_ml.features.assembler features=with_remit` exits 0 under stub mode.
- [ ] `notebooks/04_remit_ablation.ipynb` executes top-to-bottom via nbconvert.
- [ ] Three registered runs referenced by the notebook: the prior baseline (re-used), Stage 16 with-REMIT excluding `next_24h` (real extractor), Stage 16 with-REMIT including `next_24h` (real extractor).
- [ ] `docs/architecture/layers/features.md` updated with the REMIT-enriched feature set entry.
- [ ] `docs/lld/stages/16-model-with-remit.md` retrospective written.
- [ ] `CHANGELOG.md` entry under `[Unreleased]`.
- [ ] Plan moved from `docs/plans/active/` to `docs/plans/completed/` in the final commit.
- [ ] Three Phase-3 reviewers (`arch-reviewer`, `code-reviewer`, `docs-writer`) run; blocking findings addressed; PR description drafted from synthesis.
