# YYYY-MM-DD — &lt;session title&gt;

**Venue:** &lt;where, room&gt; &nbsp;·&nbsp; **Start:** &lt;time&gt; &nbsp;·&nbsp; **Length:** &lt;hours&gt;

> Replace the placeholders above and below. Delete this blockquote when
> you're ready to publish. Save the file under
> `meetups/sessions/YYYY-MM-DD-<slug>.md`.

## Audience and prerequisites

- **Assumed background:** &lt;e.g. comfortable with Python; some pandas;
  no ML experience required&gt;
- **Install before arriving:** `git`, Python 3.12, `uv`. From the repo
  root: `uv sync --group dev --frozen`.
- **Optional pre-warm:** &lt;e.g. ModernBERT embedder cache — see
  [`../topics/facilitator-guide.md`](../topics/facilitator-guide.md#cache-pre-warming)&gt;

## Theme

&lt;One paragraph. What is the evening about? What do we want attendees
to be able to point at on screen when they leave?&gt;

## Stations

Pick 2–5 from [`../topics/stations.md`](../topics/stations.md) (or
invent new ones). Each station gets its own notebook and a clear demo
moment.

### Station 1 — &lt;short name&gt;

- **Notebook:** [`notebooks/NN_name.ipynb`](../../notebooks/)
- **Goal:** &lt;1–2 sentences&gt;
- **Expected duration:** &lt;e.g. 25 min&gt;
- **Stretch:** &lt;optional follow-on, e.g. swap features=weather_calendar&gt;

### Station 2 — &lt;short name&gt;

- **Notebook:**
- **Goal:**
- **Expected duration:**
- **Stretch:**

## Self-serve paths

For attendees working solo (or arriving late), here's how to get to the
session's demo moment without a facilitator:

1. &lt;step-by-step pointer through the stations above&gt;
2. &lt;...&gt;

## Facilitator notes

- **Setup checklist:** &lt;projector, network expectations, any caches
  to warm up, fixtures to copy in&gt;
- **Common gotchas:** &lt;e.g. cold ModernBERT download stalls a demo;
  Hydra override syntax confuses; cache regeneration error message
  names the fix command&gt;
- **Whiteboard prompts:** &lt;questions to seed conversation between
  stations&gt;

## Claude Code corner

See [`../topics/claude-code-as-a-team.md`](../topics/claude-code-as-a-team.md)
for the cross-session walkthrough. Session-specific angle:
&lt;optional — e.g. "this session's stations all came out of one
`@implementer` worktree; we'll show the diff trail"&gt;

## Attendee takeaways

Promise these to attendees before the session starts. Keep it to 3–5
concrete, demonstrable items.

- &lt;e.g. "Run a rolling-origin evaluation against the NESO benchmark
  on your laptop"&gt;
- &lt;...&gt;

## Post-session retrospective

_Fill in after the session._

- **What worked:**
- **What didn't:**
- **Follow-ups:** &lt;issues to file, notebooks to refine, ideas for the
  next session&gt;
- **Notebook deltas:** &lt;PRs / commits prompted by what we noticed on
  the night&gt;
- **Headcount and energy:** &lt;optional — useful for future planning&gt;
