# Stage 8 follow-up — bounded parametric fit (2b): scope diff

| Decision | One-line summary | Tag | Justification |
|---|---|---|---|
| D1 | Switch `method="lm"` → `"trf"` with bounds; unify both `curve_fit` branches | RESTATES INTENT | Direct implementation of the fix described in the intent and AC-6 (loss variants must all work under one solver). |
| D2 | Specific bound values per parameter (intent option A) | RESTATES INTENT | Exact values are spelled out in the intent's bounds table; option (A) is the shipped default per intent lines 26–29. |
| D3 | Bounds derived at fit time; no `ScipyParametricConfig` schema change | RESTATES INTENT | Explicitly named "out of scope: no schema change" in the intent and requirements §5. |
| D4 | `active_mask`-aware std-err override: set `inf` for bound-saturated params | RESTATES INTENT | Required by AC-5 and called out in requirements §4 Observability as "new code path required"; without it AC-1/AC-2 cannot be met. |
| D5 | Extend pcov-inf WARN to name affected parameters | RESTATES INTENT | AC-1 and AC-2 each specify "WARN fires for `beta_cool` only" / "for `beta_heat` only" — naming the parameter is mandatory to satisfy the AC. |
| D6 | Defensive `p0` nudge `lb + 1e-6` when initial guess sits on a bound | PREMATURE OPTIMISATION | Intent notes the scipy regressions are fixed in ≥1.11.3; project pins ≥1.13; guard is unnecessary insurance. Adds one code path and a corresponding test assertion, neither required by any AC. |
| D7 | Re-record the `test_scipy_parametric_fit_same_data_same_params` golden popt vector | RESTATES INTENT | Requirements §4 Determinism explicitly resolves OQ-1 as "re-record the golden vector"; test guards cross-call determinism on the current solver per AC-3. |
| D8 | Revise builder Cell 12 markdown and regenerate nb08 | RESTATES INTENT | Codebase map §5 flags Cell 12 directly names the old solver and must be updated; regenerating the notebook is a mechanical consequence. |
| D9 | Append "D6 (revisited)" addendum to `docs/lld/stages/08-scipy-parametric.md` | HOUSEKEEPING | CLAUDE.md spec-drift rule requires surfacing D6's reversal; it is repo hygiene, not new scope. |
| D10 | Update `docs/architecture/layers/models.md` inventory row | HOUSEKEEPING | Stale `method="lm"` claim must track reality per spec-drift rule; standard cross-stage doc upkeep. |
| D11 | Update `src/bristol_ml/models/CLAUDE.md` `method="lm"` phrasing | HOUSEKEEPING | Same spec-drift hygiene; module CLAUDE.md carries stale text. |
| D12 (winter-only AC-1) | New test: winter-only fold popt and WARN | RESTATES INTENT | AC-1 directly specifies this fixture and assertions. |
| D12 (summer-only AC-2) | New test: summer-only fold popt and WARN | RESTATES INTENT | AC-2 directly specifies this fixture and assertions. |
| D12 (bound-saturated std-err) | New test: `std_err == inf` for bound-hitting params on AC-1/AC-2 | RESTATES INTENT | AC-5 directly specifies this assertion. |
| D12 (WARN names params) | New test: WARN message names affected parameter | RESTATES INTENT | AC-1/AC-2 both specify WARN names specific parameters; test operationalises that requirement. |
| D12 (existing tests stay green) | Recovery / determinism tests remain passing | RESTATES INTENT | AC-3 and AC-6 require no regression on healthy paths. |
| D13 | NFR: no new wall-time budget | RESTATES INTENT | Requirements §4 Performance explicitly states "No benchmark regression test is required." |
| D14 | Out-of-scope: dogbox, new config fields, bootstrap CIs, Stage 16 re-baseline | RESTATES INTENT | All four deferments are verbatim from the intent's "Out of scope" section. |

**Single highest-leverage cut.** Cut **D6** (the defensive `p0`-nudge): it
is the only decision not load-bearing for any acceptance criterion,
adds a live code path with an implicit test obligation, and guards a
bug that has been fixed in every scipy version the project permits
(`scipy>=1.13` ≫ the 1.11.3 fix).
