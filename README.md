# bristol_ml

A shared sandbox for **Bristol's community ML/data meetups** — hands-on,
multi-station, mixed-ability, not speaker-led. Built around a working GB
day-ahead electricity demand forecaster so every session has real code, real
data, and a real demo moment to point at.

The repo is two things at once:

- **A meetup hub** ([`meetups/`](meetups/README.md)) — one page per session,
  evergreen topic guides, a station-pick menu grounded in 14 working
  notebooks.
- **A reference ML/DS architecture in Python 3.12** — full project spec at
  [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md). The pedagogical intent is
  the project's primary goal (DESIGN §1.1), not a sideline.

## Next session

_None scheduled yet._ When one is, it appears at the top of
[`meetups/README.md`](meetups/README.md) and as a dated page under
[`meetups/sessions/`](meetups/sessions/).

## Past sessions

_None yet._ The session archive lives at
[`meetups/README.md`](meetups/README.md#past-sessions).

## For the self-serve attendee

You can clone this repo, run a notebook, and try a station on your own
without waiting for a meetup. The shortest path:

```bash
git clone https://github.com/samdrew/bristol-ml.git
cd bristol-ml
uv sync --group dev --frozen
uv run jupyter lab
```

Then pick a notebook from [`notebooks/`](notebooks/) — every notebook is
thin (it calls into `src/bristol_ml/`, it doesn't reimplement) and produces a
visible artefact: a plot, a leaderboard row, an API response, a search hit.

The full pick-list lives at
[`meetups/topics/stations.md`](meetups/topics/stations.md) — 14 stations
grouped by theme, each with an expected runtime and a deeper-dive pointer.
Everything runs **offline-by-default**: no API keys, no network — the live
paths are explicit opt-ins.

## Claude Code as a team

The [`.claude/`](.claude/) directory is part of the demo material, not just
project tooling. It carries a multi-agent collaboration setup — lead,
implementer, tester, reviewers, researcher, plus a four-tier escalation
ladder and write-path tier hooks — that's been the engine room of this
project. Any session can spend ten minutes poking at it as a side-station.

See [`meetups/topics/claude-code-as-a-team.md`](meetups/topics/claude-code-as-a-team.md)
for what's in there and how to demo it.

## For the project contributor

If you've arrived to read or contribute to the underlying Python project:

- [`docs/dev/README.md`](docs/dev/README.md) — quickstart, environment
  variables, cache regeneration, the full per-stage walkthrough.
- [`docs/intent/DESIGN.md`](docs/intent/DESIGN.md) — the project spec.
- [`CLAUDE.md`](CLAUDE.md) — guidance for Claude Code sessions in this repo.
- [`docs/architecture/`](docs/architecture/) — layer docs and ADRs.
- [`docs/lld/stages/`](docs/lld/stages/) — one retrospective per shipped stage
  (00–16 to date), each with a "demo moment" call-out you can adopt for a
  station.

## Licence and contact

[MIT](LICENSE). Issues, suggestions, and session pitches welcome via
GitHub Issues on [samdrew/bristol-ml](https://github.com/samdrew/bristol-ml).
