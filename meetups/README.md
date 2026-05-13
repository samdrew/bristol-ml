# Bristol ML meetups

This directory is the meetup hub for the `bristol_ml` repo. Each session
gets a dedicated page under [`sessions/`](sessions/). Evergreen
cross-session material lives under [`topics/`](topics/). Slides, diagrams,
and handouts (if a facilitator brings any) live under
[`assets/`](assets/).

The meetup format is **community, hands-on, multi-station** — not a
speaker-led talk. A facilitator picks a theme, lays out 2–5 stations
grounded in working notebooks, and attendees of mixed ability pair up or
work solo. Quality, completeness and legibility matter more than polish
or workshop choreography (DESIGN.md §1.1).

## Next session

_None scheduled._ When one is, a new page lands under
[`sessions/`](sessions/) named `YYYY-MM-DD-<slug>.md` and gets linked
here.

## Past sessions

_None yet._ Past sessions stay archived under
[`sessions/`](sessions/) with their post-session retrospective filled in.

## For attendees

- **Bring a laptop** with `git`, Python 3.12, and `uv` installed.
- Before the session, run `uv sync --group dev --frozen` from the repo
  root so dependencies are warm.
- A facilitator may also ask you to pre-warm the ModernBERT embedder
  cache — see the [facilitator guide](topics/facilitator-guide.md#cache-pre-warming)
  for the one-liner.
- You don't need an API key for any default-path station. Stations that
  exercise the live OpenAI path will say so explicitly and offer the
  offline stub as the no-key fallback.

## For facilitators

Start with the [facilitator guide](topics/facilitator-guide.md) — it
covers session shape, pre-warm checklist, and post-session retrospective
expectations. Then copy [`_template.md`](_template.md) to
`sessions/YYYY-MM-DD-<slug>.md` and fill it in.

The [stations menu](topics/stations.md) is the pick-list — 14 working
stations, each grounded in a notebook, grouped by theme. Pull 2–5 into a
session.

Every session should include a one-line **Claude Code corner** pointer
at [`topics/claude-code-as-a-team.md`](topics/claude-code-as-a-team.md);
the team setup in `.claude/` is cross-cutting demo material relevant to
any theme.

## Topic guides

- [`topics/stations.md`](topics/stations.md) — the station pick-list,
  grouped by theme, each grounded in a notebook.
- [`topics/claude-code-as-a-team.md`](topics/claude-code-as-a-team.md) —
  what's in `.claude/` and how to demo it during any session.
- [`topics/facilitator-guide.md`](topics/facilitator-guide.md) — how to
  run a community-format session in this repo, including the pre-warm
  checklist.

## Session template

[`_template.md`](_template.md) — copy this to
`sessions/YYYY-MM-DD-<slug>.md` for a new session.
