# Facilitator guide

How to run a community-format session in this repo. Read once before
your first session; skim before each subsequent one.

## What "community-format" means here

- **Multi-station, parallel work.** 2–5 stations laid out at once.
  Attendees pair up or work solo; nobody watches one person present
  for two hours.
- **Mixed ability.** Engineers comfortable with Python but new to ML
  sit next to people building production forecasters. Every station
  has an on-ramp (notebook runs end-to-end with no edits) and a
  stretch (the "what would you change to..." question).
- **Offline-by-default.** Don't assume the venue has reliable wifi.
  Don't assume attendees have API keys. The live OpenAI and
  ModernBERT paths are explicit opt-ins; the stub is the default for
  every notebook.
- **Demoable moments, not slides.** Every station ends with a visible
  artefact: a plot, a leaderboard row, an API response, a search hit.
  The `docs/lld/stages/NN-*.md` retrospective for that station names
  the moment explicitly.

## Picking a session

Start at [`stations.md`](stations.md). Pull:

- One on-ramp station from "data, joined and visible".
- One or two middle stations from the classical or neural themes.
- One stretch station from serving / events / LLMs.
- Optionally, a ten-minute Claude Code corner — see
  [`claude-code-as-a-team.md`](claude-code-as-a-team.md).

Copy [`../_template.md`](../_template.md) to
`../sessions/YYYY-MM-DD-<slug>.md`. Fill in the stations and the
session-specific bits (theme, audience, takeaways). Commit the page
**before** the session so attendees who pre-clone the repo see what
they're walking into.

## Pre-session checklist

### Before publication (1–2 weeks ahead)

- [ ] Session page exists at `meetups/sessions/YYYY-MM-DD-<slug>.md`
      with theme, stations, takeaways, prerequisites.
- [ ] Linked from `meetups/README.md` as "Next session".
- [ ] Notebooks for the chosen stations all execute end-to-end from a
      fresh `uv sync --group dev --frozen`.

### The night before

- [ ] `git pull origin <main-branch>` on your facilitator laptop.
- [ ] `uv sync --group dev --frozen` then `uv run pytest` — green.
- [ ] Execute every station notebook top-to-bottom on the projector
      laptop. Cold caches; flush any stub gotchas now, not in front of
      the room.
- [ ] **Cache pre-warming** (see below) — only if any station touches
      Stage 15 / the embedding index.

### On the night

- [ ] Projector laptop on a fresh `git pull`.
- [ ] One copy of the session page open on the projector for
      reference.
- [ ] Five-minute kickoff: explain the format, point at the session
      page URL, name the stations and the demo moment for each.
- [ ] Reset to the on-ramp station whenever someone arrives late —
      `notebooks/01_neso_demand.ipynb` is the universal entry point.

## Cache pre-warming

Only relevant if your session includes Stage 15
(`15_embedding_index.ipynb`) or any other station using the live
ModernBERT embedder. The first live-path run downloads
`Alibaba-NLP/gte-modernbert-base` (~298 MB safetensors at fp32; ~149 MB
resident in RAM at fp16) from the Hugging Face Hub. Pre-warm the cache
so the demo doesn't stall on a cold network:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Alibaba-NLP/gte-modernbert-base')"
```

The download lands in `~/.cache/huggingface/`. Subsequent runs (and CI)
honour `HF_HUB_OFFLINE=1` and never touch the network. To force the
offline stub regardless of cache state, export
`BRISTOL_ML_EMBEDDING_STUB=1`.

For attendees who can't pre-warm (no wifi, no time): every embedding
station has a working stub path that requires no download and no key.

## During the session

- **Walk between stations.** Don't park at the projector. Most
  questions come from people who've hit a Hydra-override syntax wall
  or a cache-regeneration error — both have explicit error messages
  that name the fix, but attendees need pointing at them.
- **Encourage pairing.** Mixed-ability pairs work; pure-beginner
  pairs stall.
- **Don't optimise for completeness.** Better to finish two stations
  with a good demo than five with a mediocre one.
- **Use the Claude Code corner as a refuge.** When the room hits
  collective fatigue around the 90-minute mark, ten minutes of
  meta-content (the agent roster, the escalation ladder, a stage
  retrospective walkthrough) resets attention.

## Post-session

- [ ] Within a week, fill in the **Post-session retrospective**
      section of the session page (What worked / What didn't /
      Follow-ups / Notebook deltas).
- [ ] File any follow-up issues. Common shapes: "Station N's notebook
      stalls on a cold cache — pre-warm in the docstring", "Hydra
      override syntax tripped three people — add a one-liner to the
      station card".
- [ ] Move the page from "Next session" to the "Past sessions" list
      in `meetups/README.md`.
- [ ] If the session prompted any notebook deltas, link the PRs from
      the retrospective.

## A note on tone

This is **community** content, not speaker-led. The repo is the
sandbox; the facilitator's job is to lower the activation energy for
attendees to play in it. Every session should leave someone with one
visible artefact they made, even if it's just "I ran a notebook and
saw a plot." That's the bar.
