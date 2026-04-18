# Escalation ladder

This ladder applies only when the lead has decided to use the
defensive team shape. For the default team, the lead manages retries
directly and escalates to the human after two failed attempts. The
top-level `CLAUDE.md` keeps the tier table and applicability rule; the
tier-by-tier procedure lives here.

When a teammate reports a failed implementation attempt, the lead
determines which escalation tier applies based on prior attempts on
the same task.

## Tier 0 — Prevention (always applied)

The four-hypothesis enumeration (see `CLAUDE.md` §"Quality gates and
debugging"). This is not an escalation step; it applies on every
first attempt at a debugging task. Its purpose is to prevent
anchoring on the first plausible hypothesis.

## Tier 1 — One failed attempt: role swap

Spawn the `@sceptic` against the failed work. Pass the sceptic's
critique back to the original implementer for a second attempt with
the critique in mind. The implementer keeps their context; only the
challenge is new.

## Tier 2 — Two failed attempts: cross-domain explanation

Spawn the `@translator` with a deliberately mismatched specialism.
Have the stuck teammate explain the problem to the translator. After
the exchange, the stuck teammate attempts the task once more with the
explanation transcript in their context.

## Tier 3 — Three failed attempts: adversarial reframing

Spawn the `@reframer`. Pass it the original task, the failed
attempts, and any prior reframings. When it delivers its three
alternative framings, the lead picks the most credible one and spawns
a fresh implementing teammate against that framing — not the
original. The fresh implementer must not be told which previous
attempts failed, only the new framing and the original intent layer
of the spec. Reset to baseline before this attempt (per git
protocol).

## Tier 4 — Four failed attempts: human escalation

Stop. Produce a structured report: the original task, all four
attempts and their failure modes, the reframings considered, the
lead's best guess at what is actually wrong. Surface to the human and
wait. Do not attempt a fifth implementation.

## Attempt tracking

The lead tracks the attempt count per task in the task list. If the
count becomes uncertain, default to the higher tier. Do not skip
tiers. Do not allow more than four attempts without escalating to the
human.

## Meta-role sourcing

The meta-role agents (`@sceptic`, `@translator`, `@reframer`) are not
part of the default team. They exist as user-scoped agents in
`~/.claude/agents/` and are spawned reactively by the lead only when
the escalation ladder requires them. Do not include them in the
initial team composition for normal tasks.
