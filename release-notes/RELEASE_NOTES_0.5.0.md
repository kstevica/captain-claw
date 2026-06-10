# Captain Claw v0.5.0 Release Notes

**Release title:** Flows — applications, grown up
**Release date:** 2026-06-06

---

## TL;DR

**Flows** graduate from a one-shot trigger→steps engine into a full **composition
language with self-authoring programs**. Flows now call flows (`gosub`), run work
in parallel (`spawn` / `join`), return values, handle errors, and can be **written
by agents themselves** into a curated scratch space that earns its way to
permanence. You drive them by voice or text from any channel — and control a
running flow mid-flight (`/flow pause`, `/flow stop hs`, `/flow status`).

If 0.4.x made Flows useful, **0.5.0 makes them a platform.**

---

## What this represents

Web and mobile apps are **nouns** — a place you open, a UI you navigate, where the
screen *is* the product. Flows are **verb-first**: an intent arrives on a channel
(text, voice, a glasses photo), agents do the thinking, and the result comes back
on the right channel. The interface collapses; the artifact is the *flow*, not the
screen.

What keeps this from being "an AI that does random stuff" is the core design
principle: **a deterministic spine with agent judgment at the leaves.** A flow is a
real program — triggers, branches, calls, parallelism, error handlers — and the
open-ended thinking happens only in the steps where you want it (`agent` / `vision`).
Not a black box; not rigid code. And every flow — hand-built or model-written —
is a **legible, shareable artifact** (a small text program), validated by a
deterministic parser. The model is never the source of truth; the parser is.

This is the shape we think applications take next: intent in, judgment in the
middle, result out — composable, inspectable, and increasingly **authored by the
system itself.**

## Who it's for

- **Builders of agent-native products** who want repeatable, auditable automations
  instead of hoping a single prompt behaves.
- **People living in chat / voice / glasses** — anyone who'd rather *say what they
  want* than open an app and tap through screens.
- **Operators** who need to see, pause, stop, and trust what an automation is doing
  in real time.
- **Agents themselves** — they can now turn a repeated task into a reusable flow.

---

## What's new

### Flows compose — `gosub`, args, `return`
A flow can call another flow as a subroutine and use its result
(`{{calls.<id>.output}}`). Arguments pass with `with <k>: <v>`. `return [value]`
exits from anywhere (including inside a branch). Flows are functions now.

### Flows run in parallel — `spawn` / `join` / futures
`spawn` launches a flow as an independent background worker and returns a future;
`join` collects it (with a timeout). Three lookups that took 9 seconds in series
finish in 3. Spawned flows are independently controllable and survive their
parent's stop.

### Flows handle failure — `error` steps + `on error`
Any `gosub`/`join`/`spawn` can carry `on error -> <step>`; an `error` handler step
reports `{{error.message}}` and recovers. Or branch inline on
`{{calls.<id>.status}}`. No more silent half-failures.

### Control a *running* flow from any channel
Send `/flow status`, `/flow pause`, `/flow resume`, `/flow stop` (slash optional)
to the flow you triggered. When several run at once, each has a short **handle**
(`[hs]`) shown on its prompts — target one with `/flow stop hs`, or `/flow stop all`.
Pausing a flow that's waiting on input no longer eats your next message. The same
controls are **Pause / Resume / Stop** buttons in the run log.

### Agents author their own flows — synthesis + a scratch space
The new `synthesize_flow` tool (and a **Synthesize** composer in the UI) turn a
plain-language goal into a validated, **call-only** flow stored in a separate
**scratch space**. Re-synthesizing the same behaviour dedups to one entry. A
synthesized flow **earns promotion** by running well (3 clean runs → ⭐ candidate;
3 failures → quarantined and never re-created), with tiered TTL + GC keeping the
space tidy. **Promote** the good ones into your permanent flows (safely call-only
until you switch their trigger on).

### Write flows as code, or describe them
The **Code** tab is a clean declarative DSL with a live validator and lossless
round-trip to the visual builder. Or **describe** a flow in plain English (or ask
for a change to an existing one) and a model writes the DSL — always run through
the real parser, with one-shot auto-repair.

### Trust, built in
A synthesized flow **may not** call a permanent *world-acting* flow (one that
messages you, runs a tool, or asks for input) without being promoted first — so
agent-written automations can't borrow your vetted authority. Permanent names win
over scratch (no silent shadowing). Provenance is tracked on every flow.

---

## How to use it

1. **Flight Deck → Flows → New Flow.** Build visually, or switch to **Code** and
   write the DSL (or **Describe it** and let a model write it).
2. **Trigger it.** Triggers are case-insensitive substring matches on the channels
   you choose (`any` / `whatsapp` / `web` / `glasses`).
3. **Watch / control it.** `view log` shows a depth-indented run trace; `/flow
   status` lists what's running with handles and a call-stack breadcrumb.
4. **Let agents help.** On the **Synthesized** tab, describe a goal and click
   **Synthesize**, or ask an agent in chat. Run it, watch it earn a ⭐, **Promote**.

Full language reference: **`FLOWS.md`** (also in-app via the **📖 Flow language
docs** button in the Code tab).

---

## Four worked examples

> Paste each into **Flows → New Flow → Code → Validate & apply → Save**.
> Helper flows below use a never-match trigger (`…-internal`) so they only run when
> called — you can also just toggle them **off** in the list.

### 1) Simple — "Gift Idea"
*Trigger → ask one question → think → reply.*

```text
flow "Gift Idea"
description "Ask who it's for, then suggest gift ideas"
trigger any when contains "gift idea" or contains "present idea"

step who:
  input
  prompt: "Who is the gift for, and what's the occasion + budget?"

step ideas:
  agent on origin
  prompt: "Suggest 5 thoughtful, specific gift ideas for: {{steps.who.output}}. Keep it short, with rough prices."

step reply:
  emit "{{steps.ideas.output}}"

output -> same
```

Send `gift idea` → it asks (note the `[gi]` handle on the prompt) → reply *"my dad,
60th, ~$100"*. Try `/flow status`, `/flow pause`, then `/flow resume`.

### 2) Moderate — "Trip Planner" (+ a reusable sub-flow)
*`gosub` with args + `return`, a branch on the call's result, and an `on error`
handler. `/flow status` shows the breadcrumb `Trip Planner › Place Lookup`.*

**Helper — save first:**
```text
flow "Place Lookup"
description "Look up basics about a place — call-only"
trigger any when contains "place-lookup-internal"

step info:
  agent on origin
  prompt: "Give a 3-line snapshot of {{args.place}}: best time to visit, one must-see, typical daily budget. If it is NOT a real place, reply exactly: UNKNOWN"

step done:
  return {{steps.info.output}}

output -> return
```

**The flow you trigger:**
```text
flow "Trip Planner"
description "Plan a short trip; re-ask if the place isn't found"
trigger any when contains "plan a trip" or contains "trip to"

step where:
  input
  prompt: "Where do you want to go, and for how many days?"

step lookup:
  gosub "Place Lookup"
  with place: {{steps.where.output}}
  on error -> oops

step check:
  branch
  if {{calls.lookup.output}} contains "UNKNOWN" -> retry
  else -> tips

step retry:
  emit "I couldn't find that place — give me a real city or country (and days)."

step loop:
  branch
  else -> where

step tips:
  agent on origin
  prompt: "Using this snapshot:\n{{calls.lookup.output}}\n\nWrite a short {{steps.where.output}} plan: 3 day-by-day bullets and one packing tip."

step reply:
  emit "{{steps.tips.output}}"
  return

step oops:
  error "Trip planning hit a snag: {{error.message}}"
  return

output -> same
```

Send `plan a trip` → *"Lisbon, 3 days"* → it looks the place up, then plans. Answer
with gibberish and it **loops back and re-asks** (branches can jump backward).

### 3) Complex — "Morning Briefing" (parallel `spawn`/`join`)
*One input fans out to three background workers that run **in parallel**, then
joins and merges them — with a partial-failure handler. While it runs, `/flow
status` shows the three workers as their own handles; `/flow stop all` stops
everything.*

**Three helpers — save each:**
```text
flow "Weather Brief"
trigger any when contains "weather-brief-internal"
step w:
  agent on origin
  prompt: "In 2 lines, give today's weather for {{args.place}} (assume typical seasonal weather if unsure)."
step done:
  return {{steps.w.output}}
output -> return
```
```text
flow "News Brief"
trigger any when contains "news-brief-internal"
step n:
  agent on origin
  prompt: "Give 3 short bullet headlines about {{args.topic}}."
step done:
  return {{steps.n.output}}
output -> return
```
```text
flow "Focus Tip"
trigger any when contains "focus-tip-internal"
step t:
  agent on origin
  prompt: "One concrete productivity tip for someone whose focus today is: {{args.focus}}. One sentence."
step done:
  return {{steps.t.output}}
output -> return
```

**The flow you trigger:**
```text
flow "Morning Briefing"
description "Parallel briefing: weather + news + a focus tip"
trigger any when contains "morning briefing" or contains "brief me"

step ask:
  input
  prompt: "Your city, a news topic, and your main focus today? (e.g. 'Zagreb, AI, finish the report')"

step w:
  spawn "Weather Brief"
  with place: {{steps.ask.output}}

step n:
  spawn "News Brief"
  with topic: {{steps.ask.output}}

step t:
  spawn "Focus Tip"
  with focus: {{steps.ask.output}}

step jw:
  join w
  timeout: 60
  on error -> partial

step jn:
  join n
  timeout: 60

step jt:
  join t
  timeout: 60

step merge:
  emit "Weather:\n{{joins.w.output}}\n\nNews:\n{{joins.n.output}}\n\nFocus:\n{{joins.t.output}}"
  return

step partial:
  emit "Briefing partly failed ({{error.message}}) - here is what I got:\n{{joins.w.output}}"
  return

output -> same
```

Send `brief me` → *"Zagreb, AI, finish the report"*. The three workers run together,
then you get a merged briefing.

### 4) Synthesis — let the system write the flow
No DSL at all. **Flows → Synthesized tab → describe a goal → Synthesize** (or ask
an agent in chat with the `synthesize_flow` tool):

> *"When I say 'standup', ask what I did yesterday and what's blocking me, then
> write a tight one-line status."*

It compiles to a validated, call-only flow in the **scratch space**. Run it a few
times — watch it pick up ✓ counts and earn a **⭐ ready** badge — then **Edit** or
**Promote** it into your permanent flows (it lands call-only; switch its trigger on
to make `standup` live). Re-describe the same goal and it **reuses** the existing
one instead of making a twin.

---

## New / changed endpoints (Flight Deck)

| Endpoint | Purpose |
|---|---|
| `POST /fd/flows/synthesize` | goal → validated scratch flow (dedup, optional run) |
| `GET  /fd/flows/scratch` | list synthesized flows (self-maintains) |
| `POST /fd/flows/scratch/maintain` | janitor: reclassify + GC (schedulable) |
| `POST /fd/flows/{id}/promote` | promote a scratch flow to permanent |
| `POST /fd/flows/runs/{run_id}/pause\|resume\|stop` | control a live run |
| `POST /fd/flows/compile` | English → validated flow (model-assisted) |

New agent tool: **`synthesize_flow`** (always-on; needs an agent respawn + a
reachable `FD_URL`).

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild only if you build locally:
npm --prefix flight-deck run build
```

- **Restart Flight Deck** — activates the engine, store migrations, and all flow
  endpoints, plus the committed UI.
- **Respawn agents** — only needed for the `synthesize_flow` *tool* (agents
  synthesizing autonomously in chat). The UI **Synthesize** button works without it
  (it borrows a running agent's model, like the Code-view AI compiler).
- Optional: schedule `POST /fd/flows/scratch/maintain` (the scratch list already
  self-maintains on view).

Backward compatible — existing flat flows behave identically; all new step types
and the scratch space are additive (the flows DB migrates in place).

---

## Documentation

- **`FLOWS.md`** — the complete Flow language reference: triggers, every step type
  (`tool` / `agent` / `vision` / `input` / `emit` / `branch` / `gosub` / `spawn` /
  `join` / `error` / `return`), selectors, templating, branch conditions, control
  commands + handles, the synthesis lifecycle, a cookbook, and a grammar
  cheat-sheet. Linked from the README and the in-app docs button.
