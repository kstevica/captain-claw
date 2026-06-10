# Captain Claw v0.5.1 Release Notes

**Release title:** Flows — data, loops & time
**Release date:** 2026-06-06

A focused follow-up to 0.5.0 that fills the gaps in the Flow language: **lists,
variables, iteration, loops, retries, pauses, and scheduling.** Flows go from "a
pipeline" to "a real little program" — and a reactive automation can now become a
**proactive** one on a timer. Fully additive and backward compatible.

---

## What's new

### Data — `set` + lists + a value-expression mini-language
Compute and accumulate without dropping into an agent:

```text
step cities:
  set cities = split({{steps.ask.output}}, ",")   # text → list

step count:
  set n = {{vars.n}} + 1
```

`set <name> = <expr>` stores into `{{vars.<name>}}`. Expressions support `+ - * /`
(`+` also concatenates strings/lists), `{{path}}` operands, list literals
`[a, b]`, and functions `split`, `join`, `len`, `upper`, `lower`, `trim`, `first`,
`last`, `append`, `int`, `str`, `contains`. **Lists are first-class** — they render
newline-joined and re-split cleanly. (Heavy data-crunching still belongs in a
`tool`/`agent` step — this is just enough to orchestrate.)

### Iterate — `foreach` (sequential or parallel map)
Run a flow once per item of a list, instead of repeating near-identical steps:

```text
step lookups:
  foreach city in {{vars.cities}}
  gosub "City Fact"          # sequential; use `spawn` for a parallel map
  with place: {{city}}
# → {{steps.lookups.output}} is the list of each result
```

### Loop — `while` (+ `retry`)
First-class loops, plus per-call retries:

```text
step loop:
  while {{vars.n}} > 0 -> tick     # jumps to `tick`, whose path loops back

step fetch:
  gosub "Flaky Lookup"
  retry: 3                          # re-run on failure before on-error
  on error -> give_up
```

### Time — `sleep` + `wait until`
- `sleep 30s | 5m | 2h | 1d` pauses the run (`/flow stop` still interrupts it).
- `wait until contains "approved"` parks the flow until an **inbound message
  matches** — non-matching messages fall through to the agent. Perfect for
  approval gates.

```text
step gate:
  wait until contains "approved"
```

### Scheduling — run a flow on a timer
The Flight Deck **Scheduler** can now run a **Flow**, not just a text prompt:
in a scheduler job, switch the action to **Flow**, pick it, and set a schedule
(`daily 08:00`, `every 2h`, …) plus a delivery target. A self-contained
`Morning Briefing` can fire at 8am instead of waiting for "brief me".
(Scheduled flows should be self-contained — no `input`/`wait` at fire time.)

### Smarter "Describe it → flow"
The AI compiler now knows the full vocabulary — describing a flow in plain English
(or asking for a change) can produce `set` / `foreach` / `while` / `sleep` /
`wait` / `retry`, with a built-in "when to use what" guide. Output is still run
through the real parser with one-shot auto-repair.

### Builder + docs
New step editors (set / foreach / while / sleep / wait + a retry field), and
**`FLOWS.md`** gains a full reference for the new steps plus five worked examples
(City Compare, Topic Digest, Countdown, Deploy approval, Daily Digest).

---

## Two quick examples

**Parallel map over a user-provided list:**

```text
flow "Topic Digest"
trigger any when contains "digest"
step ask:
  input
  prompt: "Which topics? (comma-separated)"
step topics:
  set topics = split({{steps.ask.output}}, ",")
step heads:
  foreach topic in {{vars.topics}}
  spawn "Topic Headlines"
  with topic: {{topic}}
  timeout: 60
step out:
  emit "{{steps.heads.output}}"
output -> same
```

**An approval gate:**

```text
flow "Deploy"
trigger any when contains "deploy please"
step confirm:
  emit "Reply 'approved' to proceed."
step gate:
  wait until contains "approved"
step go:
  gosub "Run Deploy"
output -> same
```

See **FLOWS.md → Worked examples** for the full set (with helper flows).

---

## New / changed (Flight Deck)

- Step types: `set`, `foreach`, `while`, `sleep`, `wait`; the `retry:` modifier.
- Scheduler jobs gain a `flow_id` (run a Flow); the Scheduler UI gets a Prompt/Flow toggle.
- `/fd/flows/compile` + `/fd/flows/synthesize` prompts updated with the new primitives.

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild only if you build locally:
npm --prefix flight-deck run build
```

- **Restart Flight Deck** — activates the engine, the scheduler `flow_id`
  migration (in place), and the committed UI.
- No agent respawn needed for these (the new steps run in Flight Deck).

Backward compatible with 0.5.0 — existing flows behave identically; everything
here is additive.
