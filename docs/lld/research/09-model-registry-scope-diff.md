# Stage 9 — Model registry: Scope Diff

**Status:** Research artefact (fourth Phase-1 output) — immutable after Stage 9 ships.
**Date:** 2026-04-22
**Baseline SHA:** `575ac9c`
**Author:** `@minimalist` scope critic (spawned pre-synthesis per `/workspace/.claude/agents/lead.md` Phase 1).
**Input draft decision set:** the lead's internal 21-decision draft, captured verbatim in the table below.
**Linked from:** `/workspace/docs/plans/active/09-model-registry.md` (preamble).

---

## Scope Diff (verbatim, @minimalist output)

| ID | Summary | TAG | Justification |
|----|---------|-----|---------------|
| D-1 | Flat `data/registry/{run_id}/` with artefact subdir + `run.json` sidecar | RESTATES INTENT | Directly resolves Points line 1 (flat vs nested) and satisfies AC-5 hand-inspection. |
| D-2 | One entry per `registry.save()` invocation | RESTATES INTENT | Directly resolves Points line 2 (unit of registration). |
| D-3 | Run ID `{model_name}_{iso_ts}_{uuid4[:8]}` | PLAN POLISH | Intent is silent on ID format; composite scheme binds every filename test and Stage 12 load-by-name contract. |
| D-4 | Single JSON sidecar per run, no YAML dep | RESTATES INTENT | Operationalises AC-5 (text-inspectable) and keeps dep surface at zero. |
| D-5 | Atomic temp-dir-then-rename save | PREMATURE OPTIMISATION | Intent names no crash/partial-write hazard; guards a failure mode not in scope and adds the `test_registry_save_round_trip_atomic` test. |
| D-6 | Defer feature-table hash | RESTATES INTENT | Directly resolves Points line 3. |
| D-7 | `list` takes `target=`, `model_type=`, `feature_set=` filters | RESTATES INTENT | Directly operationalises Points line 4 ("show me the best model for X"). |
| D-8 | Leaderboard default sort MAE asc, `sort_by=` arg | RESTATES INTENT | Directly resolves Points line 5. |
| D-9 | Registry calls `model.save(path)`; `Model` protocol unchanged | RESTATES INTENT | Directly operationalises AC-2 + Points line 6. |
| D-10 | MLflow graduation adapter documented, no dep | RESTATES INTENT | Directly resolves Points line 7. |
| D-11 | `data/registry/` gitignored, no export tool | RESTATES INTENT | Directly resolves Points line 8 (defers portable export). |
| D-12 | Public surface = 4 verbs; `__all__` counted | RESTATES INTENT | Directly operationalises AC-1. |
| D-13 | Git SHA captured via `subprocess.run(...)` inside registry | RESTATES INTENT | Directly operationalises AC-3; codebase H2 confirms no existing helper. |
| D-14 | Keep joblib; defer skops; add trust-scope docstring | PLAN POLISH | Intent silent on serialisation; forces a new module docstring review and pins a Stage 12 hand-off but ships either way. |
| D-15 | Caller passes metrics DataFrame to `save()`; harness unchanged | RESTATES INTENT | Required by AC-3 (metrics passed explicitly) and avoids triggering `evaluation/CLAUDE.md` H5 API-growth rule. |
| D-16 | Registry stores dynamic `_NamedLinearModel` name, re-applies on load | PREMATURE OPTIMISATION | Intent does not require round-trip of the dynamic-named wrapper; binds `test_registry_load_reapplies_named_linear_model` and couples registry load semantics to a `train.py` internal. |
| D-17 | New `RegistryConfig` schema + `conf/registry/default.yaml` Hydra group | PLAN POLISH | Intent does not require Hydra-controlled registry path; a module-level default satisfies AC-4/AC-5 and this adds a schema, YAML file, and `AppConfig` field test. |
| D-18 | `python -m bristol_ml.registry list` CLI entry | RESTATES INTENT | Directly operationalises Demo moment + US-1. |
| D-19 | `train.py` re-fits on full data and calls `registry.save(...)` after harness | PLAN POLISH | Intent says retrofit existing models to save through registry; re-fit-on-full-data is extra semantics not named in AC-2, adds a retro bullet and a dispatcher touchpoint (codebase H4 risk). |
| D-20 | New `docs/architecture/layers/registry.md` layer doc with on-disk schema | RESTATES INTENT | Directly operationalises AC-5 (documented layout). |
| D-21 | CHANGELOG, DESIGN.md §6, stage retro | HOUSEKEEPING | Standard stage-hygiene batch per `CLAUDE.md`. |
| NFR-speed | list 100 runs < 1 s | PLAN POLISH | Intent says "instantaneous"; picking 1 s binds `test_registry_list_hundred_entries_is_fast` to a specific threshold the human has not confirmed. |
| NFR-transparency | Metadata as text | RESTATES INTENT | Directly implied by AC-5. |
| test-public-surface | `__all__` ≤ 4 | RESTATES INTENT | AC-1 evidence. |
| test-save-naive/linear/sarimax/scipy (×4) | Each model saves via protocol | RESTATES INTENT | AC-2 evidence — all four retrofit targets named in intent §Scope. |
| test-git-sha-captured | Auto git SHA | RESTATES INTENT | AC-3 evidence. |
| test-raises-on-missing-field | Explicit-field required | RESTATES INTENT | AC-3 second half (rest passed explicitly). |
| test-list-100-fast | 100 runs < 1 s | RESTATES INTENT | AC-4 evidence. |
| test-layout-doc-exists | Layer doc non-empty | RESTATES INTENT | AC-5 evidence. |
| test-save-round-trip-atomic | Temp-dir rename atomicity | PREMATURE OPTIMISATION | Rides on D-5; forces a crash-simulation test for a hazard intent does not name. |
| test-list-filter-by-target | Filter arg works | RESTATES INTENT | Evidence for Points line 4 / D-7. |
| test-list-default-sort-mae | Default sort MAE asc | RESTATES INTENT | Evidence for Points line 5 / D-8. |
| test-load-reapplies-named-linear | Dynamic name preserved | PREMATURE OPTIMISATION | Rides on D-16; couples test suite to a `train.py` wrapper the intent does not name. |
| test-covariance-roundtrip-json | `covariance_matrix` survives JSON | PLAN POLISH | AC-2 for `ScipyParametricModel` already covers round-trip; a dedicated covariance JSON test duplicates that coverage and adds a fixture. |
| deps | None | RESTATES INTENT | Zero-dep stance is load-bearing for the thin-interface AC-1. |

**If you cut one item to halve this plan's scope, cut D-16 because removing the `_NamedLinearModel` dynamic-name round-trip eliminates one test, unblocks deferring the codebase H4 hazard to a later stage (or to a `train.py` cleanup), and prevents the registry load contract from being coupled to a `train.py` internal wrapper — which would otherwise bind every downstream Stage 10/12 load path.**

---

## Lead disposition (post-critique, pre-plan)

Each flagged row below has been reconsidered.  The default on `PREMATURE OPTIMISATION` is cut; the default on `PLAN POLISH` is justify-or-cut.

| ID | Flag | Disposition | Rationale |
|----|------|-------------|-----------|
| **D-3** | PLAN POLISH | **Simplify, retain.** Drop the `uuid4[:8]` suffix.  New format: `{model_name}_{YYYYMMDDTHHMMSS}`.  Human-typeable at the `registry list` CLI prompt; collision-acceptable at single-author scale (two runs in the same second are vanishingly rare and last-write-wins is explicit intent).  Near-RESTATES INTENT after the simplification (Demo moment requires typeable names). |
| **D-5** | PREMATURE OPTIMISATION | **Retain; retag HOUSEKEEPING.** The codebase has this idiom at two existing sites (`ingestion/_common.py:201` and `models/io.py:50-53`).  Matching established convention is housekeeping, not new scope — four lines of Python, no test fixture inflation.  Note the test `test_registry_save_round_trip_atomic` is demoted to an assertion inside the existing save tests rather than a dedicated crash-simulation test. |
| **D-14** | PLAN POLISH | **Retain as docs-only; retag HOUSEKEEPING.** One-line module docstring on existing `src/bristol_ml/models/io.py` naming Stage 12 as the `skops` trigger.  Zero new files; zero new tests.  Documents a deferred upgrade that is otherwise unowned. |
| **D-16** | PREMATURE OPTIMISATION | **CUT.** Registry's `load()` returns base `LinearModel`; the dynamic name from `_NamedLinearModel` is stored in the sidecar's `name` field only.  No load-side re-application.  No `train.py` internal touched.  Test `test_registry_load_reapplies_named_linear_model` is **cut** with this decision. |
| **D-17** | PLAN POLISH | **CUT.** Use a module-level constant `DEFAULT_REGISTRY_DIR = Path("data/registry")` inside `registry/`, overridable via a `--registry-dir` flag on the `registry list` CLI.  No `RegistryConfig` Pydantic schema, no `conf/registry/default.yaml`, no `AppConfig` edit, no Hydra discriminated-union change.  Cuts four expected files from the plan. |
| **D-19** | PLAN POLISH | **Reframe.** `train.py` calls `registry.save(...)` on the **final-fold fitted model** already held in harness memory — no re-fit on full data.  Removes the "extra semantics not named in AC-2" the critic flagged.  The last-fold model is a faithful-enough representative for the leaderboard; full-data re-fit can follow later under a different trigger if needed. |
| **NFR-speed** | PLAN POLISH | **Retain with human-gate.** Keep `list 100 runs < 1 s` as the test threshold; flag it in §1 decision table as "threshold quantifies `instantaneous`; confirm at Ctrl+G".  The test needs a concrete gate and this one is an order of magnitude above plausible laptop-class performance. |
| **test-save-round-trip-atomic** | PREMATURE OPTIMISATION | **Demote.** The atomicity property is exercised by existing save tests hitting the full write path; no dedicated crash-simulation test is added.  The test name is removed from §4. |
| **test-covariance-roundtrip-json** | PLAN POLISH | **CUT.** Redundant with `test_registry_save_scipy_parametric_model_via_protocol` (AC-2), which already exercises the full round-trip including the covariance matrix. |

### Net change after disposition

- **Decisions:** 21 → 19 (cut D-16, D-17).
- **Tests:** 12 → 10 (cut `test_registry_load_reapplies_named_linear_model`, `test_registry_save_covariance_matrix_json_roundtrip`; demote `test_registry_save_round_trip_atomic` to inline assertion).
- **New config files:** 3 → 0 (no `RegistryConfig`, no `conf/registry/default.yaml`, no `AppConfig` edit).
- **Model-code touches:** 1 → 0 (no `_NamedLinearModel.load` work; the H4 hazard is left as-is).
- **Dependencies added:** 0 (unchanged — zero-dep stance preserved).
- **Hydra discriminated-union touches:** 1 → 0.

The highest-leverage cut the minimalist named (D-16) is accepted; D-17 is the larger downstream-surface cut (four files vs one test).  Together they remove five expected-file entries from the plan's §7.

---

## Sources

- `/workspace/docs/intent/09-model-registry.md` (contract)
- `/workspace/docs/lld/research/09-model-registry-requirements.md`
- `/workspace/docs/lld/research/09-model-registry-codebase.md`
- `/workspace/docs/lld/research/09-model-registry-domain.md`
- `/workspace/.claude/agents/minimalist.md` (agent spec driving this artefact)
- `/workspace/.claude/agents/lead.md` (Phase 1 workflow mandating its production)
