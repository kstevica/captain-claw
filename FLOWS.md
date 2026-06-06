# Captain Claw Flows

**Flows** are declarative automations that run inside Flight Deck and dispatch work to your agent pool. A flow is a **trigger** (what starts it) plus an ordered list of **steps** (what it does) plus an **output** (where the result goes). Flight Deck owns the deterministic plumbing — triggering, routing, sequencing, branching, pausing for input, guardrails — while agents do the judgment work as steps.

You can build a flow two ways, and they are the *same* flow:

- **Builder** — a form-based UI (click to add steps).
- **Code (DSL)** — a small text language, this document’s subject. The two round-trip losslessly: edit as code, switch to Builder, and vice-versa.

There is also an **AI compiler**: describe what you want in plain English and a model writes the DSL for you, which is then checked by the real parser before it’s applied.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Anatomy of a flow](#anatomy-of-a-flow)
3. [Triggers](#triggers)
4. [Steps](#steps)
   - [tool](#tool-step)
   - [agent](#agent-step)
   - [vision](#vision-step)
   - [input](#input-step)
   - [emit](#emit-step)
   - [branch](#branch-step)
5. [Agent selectors (`on`)](#agent-selectors-on)
6. [Templating — `{{ … }}`](#templating)
7. [Branch conditions](#branch-conditions)
8. [Stopping a flow](#stopping-a-flow)
9. [Output](#output)
10. [Execution & delivery model](#execution--delivery-model)
11. [Channels](#channels)
12. [The AI compiler](#the-ai-compiler)
13. [Cookbook](#cookbook)
14. [Common errors](#common-errors)
15. [Grammar cheat-sheet](#grammar-cheat-sheet)

---

## Quick start

```text
flow "Hungry helper"
trigger any when contains "hungry"

step where:
  input
  prompt: "Where are you right now?"

step find:
  agent on origin
  prompt: "Find good ćevapi places near {{steps.where.output}}."

step reply:
  emit "{{steps.find.output}}"

output -> same
```

What this does: when any inbound message contains “hungry”, the flow asks the user for their location and **waits** for their reply, hands that to the triggering agent to search, then sends the answer back on the same channel.

Paste it into the **Code** tab → **Validate & apply** → switch to **Builder** to see the steps, then **Save**.

---

## Anatomy of a flow

```text
flow "<name>"                  # optional but recommended
description "<text>"           # optional
priority 50                    # optional (higher = matched first; default 50)
trigger <channel> [when <rules>]

step <id>:
  <type> [on <selector>]
  <field>: <value>
  ...

step <id>:
  ...

output -> <channel>            # where the result goes; default `same`
```

- Lines beginning with `#` are **comments** (and `#` after code on a line is an inline comment, unless inside quotes). Comments are an authoring aid — they are **not stored** with the flow, so they won’t reappear after Save/round-trip. Use **Load example** in the Code view to re-read the annotated example anytime.
- Indentation groups a step’s fields under its `step <id>:` header. Top-level directives (`flow`, `trigger`, `step`, `output`, …) start at column 0.
- Order matters: steps run top-to-bottom unless a [branch](#branch-step) jumps elsewhere.
- Every flow needs a trigger, at least one step, and an `output` line.

---

## Triggers

```text
trigger <channel> [when <rule> and <rule> …]
trigger <channel> always
```

**Channels:** `any`, `whatsapp`, `web`, `glasses`. Use `any` (the default) unless you want the flow to fire on only one surface.

**Rules** (all listed rules must match — they’re AND-ed; join with `and` or commas):

| Rule | Matches when… |
|---|---|
| `has image` | the message carries an image |
| `has video` | …a video |
| `has audio` | …an audio/voice note |
| `has document` | …a document |
| `has text` | …non-empty text |
| `contains "word"` | the text contains *word* (case-insensitive substring) |
| `from_waid "<id>"` | the sender’s WhatsApp id equals *id* (exact) |
| `mime "image/*"` | the attachment MIME matches the glob |
| `regex "^hi"` | the text matches the regular expression (case-insensitive) |

**Combining rules:** join with **`and`** (every rule must match) or **`or`** (any one matches). Pick one — don’t mix `and` and `or` in a single trigger (split into two flows if you need that). Commas mean `and`.

`trigger <channel> always` fires on **every** message on that channel (use sparingly). A trigger with **no** `when` matches any message on the channel.

Examples:

```text
trigger whatsapp when has image
trigger any when contains "hungry"
trigger any when contains "hungry" or contains "gladan" or contains "gladni"   # ANY word
trigger web when regex "^/order\b" and has text                                # ALL conditions
trigger whatsapp when has image and from_waid "385991234567"
```

In the **Builder**, the same choice is the **Match: ALL (and) / ANY (or)** toggle above the rules.

> If two enabled flows could match the same message, the one with the higher `priority` wins. No match → the message is handled normally (a regular agent turn).

---

## Steps

Every step has an **id** (referenced by branches and templating) and a **type**. The type appears as the first keyword inside the step block, optionally followed by `on <selector>`.

### tool step

A **deterministic, single, named tool** call — no LLM turn. Use it only when you know the exact tool name.

```text
step id:
  tool on fd
  tool: face_identify
  arg image: {{trigger.fd_image_path}}
```

- `tool:` — the tool name (**required**).
- `arg <name>: <value>` — one line per argument.
- `on` defaults to `origin`. On `fd` the available internal tool is **`face_identify`** (recognizes faces in an image; returns JSON — see [templating](#templating)).

> ⚠️ A `tool` step **must** have a `tool:` name. For open-ended work (“search”, “research”, “find”, “answer”), use an [agent step](#agent-step) instead — the agent picks its own tools.

### agent step

A scoped turn on a pooled agent — the agent uses its own tools/knowledge to do open-ended work.

```text
step research:
  agent on origin
  prompt: "Research {{steps.topic.output}} and give 3 bullet points."
  attach: {{trigger.image_path}}   # optional file/image to hand to the agent
  deny: shell, scripts             # optional: tools to forbid this turn
```

- `prompt:` — the instruction (**required**).
- `attach:` — optional path to a file/image to send to the agent first (it’s uploaded to the target and verified; the agent then sees it). When an attachment is present the step is locked to *describe the attachment only* and `shell`/`scripts`/`read` are blocked, so don’t use `attach` for general work.
- `deny:` — comma-separated tools to forbid for this turn.
- `on` defaults to `origin`.

### vision step

A **raw image description** — the vision model is called directly with no agent loop, memory, tools, or history. This is the reliable way to “look at” an image.

```text
step describe:
  vision on capability:vision
  prompt: "Describe what's in this photo in one sentence."
  image: {{trigger.fd_image_path}}
```

- `prompt:` — what to ask about the image.
- `image:` — the image to look at (a path/template).
- `on` defaults to `capability:vision` (an agent that has a vision model).

### input step

**Pauses the flow**, asks the user a question, and resumes with their reply.

```text
step name:
  input
  prompt: "What is your name?"
  timeout: 3600      # optional seconds to wait (default 3600 = 1 hour)
```

- The user is messaged: `⏳ *<FlowName>* needs your input:` followed by your prompt (the flow name is always announced).
- Their next reply becomes `{{steps.<id>.output}}` and the flow continues.
- If no reply arrives within `timeout`, the run fails.
- Works on WhatsApp and on agent-handled channels (web/glasses) — see [channels](#channels).

### emit step

Sends a message to the user mid-flow.

```text
step note:
  emit "Working on it…"
```

Long form (to target a specific channel):

```text
step note:
  emit
  channel: whatsapp
  body: "Working on it…"
```

`channel:` defaults to `same` (the originating channel). Other values: `whatsapp`, `web`, `glasses`, `log`.

### branch step

A **switch**: evaluate conditions top-to-bottom; the first true one jumps to its target. An optional `else` is the fallback.

```text
step route:
  branch
  if {{steps.face.confident}} == true -> greet
  elif {{steps.face.count}} == 0 -> describe
  else -> ask_who
```

- Each `if`/`elif` is `<condition> -> <target>`.
- `else -> <target>` runs when nothing matched.
- A **target** is a step id, or `stop` to end the flow (see [stopping](#stopping-a-flow)).
- If nothing matches and there’s no `else`, execution falls through to the next step.
- See [branch conditions](#branch-conditions) for the full condition language.

### gosub step

**Calls another flow as a subroutine** and waits for it to finish — flows compose
like functions. The called flow’s return value is `{{calls.<step_id>.output}}`
and its outcome is `{{calls.<step_id>.status}}` (`done` / `error`).

```text
step geo:
  gosub "Geocode"
  with place: {{steps.where.output}}
  with units: metric
```

- The target is matched **by flow name** (case-insensitive).
- `with <name>: <value>` lines pass **arguments**; inside the called flow they’re
  available as `{{args.<name>}}` (and as `{{trigger.<name>}}`).
- The call **blocks** until the child returns; the child runs as a nested frame
  of the same run (you’ll see it indented in the run log, and in `/flow status`).
- A child that pauses for `input` pauses the whole chain — your reply continues it.
- Guards: a recursion **depth cap** and a **shared step budget** across the whole
  call tree stop runaways.

### return step

**Ends the flow now and hands a value back** to the caller (or to the output
channel for a top-level flow). Works as its own step *or* as a trailing line on
any step, including inside a branch path.

```text
step done:
  return {{calls.geo.output}}
```

```text
step oops:
  emit "Couldn't do it."
  return            # exit here — supersedes the bare `stop` flag
```

- `return <expr>` returns that value; bare `return` returns the last step’s output.
- A flow meant to be `gosub`’d should end with `output -> return` (its value goes
  to the caller, not to a user channel).

### spawn step + join step

**Run flows in the background and collect them later** — for parallelism. `spawn`
launches a flow as an independent background run and continues immediately; `join`
waits for it.

```text
step w1:
  spawn "Geocode"
  with place: {{trigger.text}}

step w2:
  spawn "Weather"
  with place: {{trigger.text}}

step got_geo:
  join w1
  timeout: 30

step got_weather:
  join w2

step done:
  return "{{joins.w1.output}} — {{joins.w2.output}}"
```

- `join <spawn_step_id>` exposes `{{joins.<id>.output}}` and `{{joins.<id>.status}}`
  (`done` / `error` / `timeout`). Default timeout is 300s; set `timeout: <seconds>`.
- A spawned flow is **independent**: it isn’t killed by your `/flow stop` (use
  `/flow stop all`), and it has its own entry in `/flow status`.
- Stopping *this* flow aborts a pending `join` (the spawned flow keeps running).

### error step + `on error`

**Handle failures.** Any `gosub` / `join` / `spawn` step can carry
`on error -> <step>` — if that call fails (error/timeout), the flow jumps to the
named step instead of continuing. An `error` step is a tidy handler that reports
the problem; the failure is available as `{{error.message}}` and `{{error.status}}`.

```text
step geo:
  gosub "Geocode"
  with place: {{steps.where.output}}
  on error -> failed

step done:
  return {{calls.geo.output}}

step failed:
  error "Couldn't locate you: {{error.message}}"
  return
```

Without `on error`, a failed call just sets its `{{calls|joins.<id>.status}}` — you
can branch on that instead (the inline form).

---

## Agent selectors (`on`)

The `on` clause says **where** a step runs:

| Selector | Runs on |
|---|---|
| `origin` | the agent that received the triggering message (default for `tool`/`agent`) |
| `fd` | Flight Deck itself, in-process (for internal tools like `face_identify`) |
| `any` | any available pooled agent |
| `capability:vision` | an agent that has a vision model (default for `vision`) |
| `name:DeepSeek V4 Flash` | a specific agent by name |

> Cross-agent file steps transfer the file to the target and use the target-local path automatically — you don’t manage paths.

---

## Templating

Anywhere a value is expected, `{{ … }}` pulls from the run context. It’s resolved **at the moment the step runs** (so later steps see earlier results).

**Trigger fields** — about the message that started the flow:

| Token | Value |
|---|---|
| `{{trigger.text}}` | the message text |
| `{{trigger.channel}}` | `whatsapp` / `web` / `glasses` |
| `{{trigger.waid}}` | sender’s WhatsApp id (if any) |
| `{{trigger.mime}}` | attachment MIME type |
| `{{trigger.image_path}}` | image path **on the agent** that received it |
| `{{trigger.fd_image_path}}` | image path **local to Flight Deck** (use this for `on: fd` tools like `face_identify`) |
| `{{trigger.video_path}}` / `{{trigger.audio_path}}` | media paths |
| `{{trigger.origin_name}}` | name of the receiving agent |

**Step outputs:**

| Token | Value |
|---|---|
| `{{steps.<id>.output}}` | that step’s output text |
| `{{steps.<id>.<field>}}` | a field, when the step returned JSON |

> **Important:** JSON fields attach **flat**. `face_identify` returns `{confident, name, person_id, confidence, count, card}`, so you reference `{{steps.identify.name}}` and `{{steps.identify.confident}}` — **not** `{{steps.identify.output.name}}`. `{{steps.identify.output}}` is the raw JSON string.

**System fields:**

| Token | Value |
|---|---|
| `{{system.now}}` | ISO timestamp |
| `{{system.date}}` | `YYYY-MM-DD` |
| `{{system.time}}` | `HH:MM` |
| `{{system.agent}}` | origin agent name |
| `{{system.channel}}` | origin channel |

A token that resolves to nothing becomes an empty string.

---

## Branch conditions

Branch `if`/`elif` conditions use a small, safe boolean language (it is **not** `eval` — a malformed condition just evaluates to false rather than crashing the run).

**Logical operators:** `and`, `or`, `not` (also `&&`, `||`, `!`), grouped with parentheses `( )`.

**Comparisons:**

| Operator | Meaning |
|---|---|
| `==`, `!=` | string equality / inequality |
| `>`, `<`, `>=`, `<=` | numeric (falls back to lexicographic if not numbers) |
| `contains` | left contains right (case-insensitive substring) |
| `matches` (or `~`) | left matches the right regular expression (case-insensitive) |

**Operands** are `{{templates}}`, `"quoted strings"`, or bare words/numbers.

**Truthiness:** a condition that is just an operand is **true** unless it resolves to empty, `false`, `0`, `none`, `no`, or `null`. So `{{steps.face.confident}}` alone is a valid condition.

Examples:

```text
if {{steps.face.confident}} == true -> greet
elif {{steps.face.count}} > 0 and not {{steps.face.confident}} -> ask_who
elif {{trigger.text}} contains "urgent" or {{trigger.text}} matches "^!!" -> escalate
else -> describe
```

> Booleans render lowercase: a JSON `true` compares as `== true`. Numbers compare numerically (`> 0`, `>= 3`).

---

## Stopping a flow

By default a flow ends after its last step. To end **early** (typically inside a branch path):

**1. Stop after a step** — add `stop` as the last line of any non-branch step:

```text
step done:
  emit "All set — nothing more to do."
  stop
```

**2. Stop from a branch** — point a target at `stop`:

```text
step route:
  branch
  if {{steps.x.output}} == "skip" -> stop
  else -> continue
```

When a flow stops (or finishes), its **last executed step’s output** is delivered to the [output](#output) channel.

**3. Stop, pause or resume a *running* flow by message** — from any channel the
flow runs on, send a control command (the leading `/` is optional):

| Command | Effect |
|---|---|
| `/flow status` | List your running flows — each with a **handle** `[hs]`, state, and call-stack breadcrumb |
| `/flow stop` | Stop the most-recent flow. Add a phrase to send it first: `/flow stop ok, cancelled` |
| `/flow stop <handle>` | Stop a specific flow by its handle, e.g. `/flow stop hs` |
| `/flow stop <name>` | Stop by flow-name fragment, e.g. `/flow stop weather` |
| `/flow stop all` | Stop **every** running flow |
| `/flow pause [handle\|name\|all]` | Pause (most-recent by default) |
| `/flow resume [handle\|name\|all]` | Continue a paused flow (`/flow continue` also works) |

**Handles.** When several flows run at once, each gets a short stable tag like
`[hs]` (shown on its input prompts and in `/flow status`) so you can address it:
`/flow pause hs`, `/flow stop weather`. A bare command targets your most-recent
flow; `all` targets every one.

These reach the flows **you** triggered (matched by your WhatsApp number, or by
your web/glasses session). `stop` works even while a flow is paused waiting on an
`input` step. `spawn`’d background flows are independent — a plain `/flow stop`
won’t touch them; use `/flow stop all` (or their handle). The same per-run
controls are also **Pause / Resume / Stop** buttons in the Flight Deck run log.

**Pause + input:** if you pause a flow while it’s waiting for your input, your
next messages go to the agent as normal chat — they are **not** swallowed as the
flow’s answer. The flow stays on that step; `/flow resume` re-shows the question,
and your next reply continues it from there.

---

## Output

```text
output -> <channel>
```

`<channel>` is `same` (reply on the originating channel — the default), `whatsapp`, `web`, `glasses`, `log` (record only, no message sent), or `return` (hand the value back to the caller — for a flow meant to be [`gosub`’d](#gosub-step)).

The output line delivers the **last executed step’s** output — unless that step was already an `emit` to a user channel (then it isn’t re-sent). For a `gosub`’d sub-flow the value goes to the **caller**, not to a user channel, unless the child explicitly `emit`s.

---

## Execution & delivery model

- Steps run **top-to-bottom**; a `branch` can jump (including backwards, forming loops). A `max_steps` guardrail (default 20) bounds total executed steps so a loop can’t run forever.
- Each step’s output is stored under its id and exposed via `{{steps.<id>.output}}` (plus flat JSON fields).
- The flow’s final reply is the **last executed** step’s output, sent to the `output` channel.
- Tool/vision steps are deterministic single calls; agent steps are scoped one-shot turns (no open-ended tool spirals unless the agent decides to use its tools); input steps pause until the user replies.

---

## Channels

Flows behave consistently across channels, but the delivery mechanics differ:

- **WhatsApp** — Flight Deck owns the channel. The flow runs independently of any agent turn, prompts/resumes input over WhatsApp, and replies via WhatsApp. The `origin` agent is free, so `on: origin` steps work without contention.
- **Web / glasses (agent-handled)** — a flow that pauses for input or consults `origin` runs in the **background** so the originating agent isn’t blocked; its messages (input prompts, emits, final output) are pushed into the chat asynchronously. A user reply that a paused flow is waiting for **resumes** the flow instead of starting a new turn.
- Glasses input arrives via WhatsApp in the current setup; HUD output is delivered on the channel bus.

You usually don’t need to think about this — write the flow once and it works on each surface. Test on **WhatsApp** for the fullest experience (input + origin steps).

---

## The AI compiler

In the **Code** tab:

- **Describe it (AI → flow):** type plain English (e.g. *“when someone says they’re hungry, ask for their location then search for ćevapi nearby”*), pick which agent’s model compiles it, and click **Compile with AI**. The model writes canonical DSL, which is then run through the **real parser/validator** — invalid output is rejected (with the error and the raw DSL shown), never silently saved. If the first attempt fails to compile, the error is fed back to the model once to self-correct.
- **Flow code (DSL):** the editor. **Validate & apply** compiles your code and updates the Builder (or shows `line N: <error>`). **Load example** drops in a heavily-commented sample to explore.

The AI is a convenience front-end; the deterministic parser is the source of truth. A stronger model produces better flows.

---

## Synthesized flows (the scratch space)

Agents can **author their own flows on the fly** with the `synthesize_flow` tool:
they describe a repeatable goal, Flight Deck compiles it through the same
validator, and stores it in a separate **scratch space** (`origin: agent`,
call-only — no message trigger). Re-synthesizing the same behaviour **dedups** to
one entry (canonical hash), and each use is counted.

- **Two spaces.** Your hand-built flows are *permanent*; agent-synthesized ones
  live in *scratch*, listed under **“Synthesized (scratch)”** in the Flows panel —
  separate from your real list. **Promote** one to move it into your permanent
  flows; discard the rest (they also expire on a TTL).
- **Reuse them.** A scratch flow is callable — `gosub "Its Name"` (the permanent
  space wins on a name clash, so a synthesized flow can never silently shadow a
  vetted one).
- **Trust guard.** A synthesized flow **may not** `gosub`/`spawn` a *permanent
  world-acting* flow (one that messages the user, runs a tool, or asks for input)
  — that call is blocked until the synthesized flow is **promoted**. Synthesized
  flows can freely call read-only flows and each other.
- **When agents synthesize.** Only for work that’s repeatable, durable, auditable,
  or worth handing off — not for a one-off answer.

Endpoints: `POST /fd/flows/synthesize` (goal → validated scratch flow, optional
run), `GET /fd/flows/scratch` (list), `POST /fd/flows/{id}/promote`.

---

## Cookbook

### Recognize a face, greet or ask

```text
flow "Smart photo triage"
trigger whatsapp when has image

step identify:
  tool on fd
  tool: face_identify
  arg image: {{trigger.fd_image_path}}

step route:
  branch
  if {{steps.identify.confident}} == true -> greet
  elif {{steps.identify.count}} == 0 -> describe
  else -> ask_who

step greet:
  emit "👋 Hey {{steps.identify.name}}!"
  stop

step ask_who:
  input
  prompt: "I don't recognize this person — who is it?"

step ack:
  emit "Thanks! I'll remember {{steps.ask_who.output}}."
  stop

step describe:
  vision on capability:vision
  prompt: "Describe what's in this photo in one sentence."
  image: {{trigger.fd_image_path}}

output -> same
```

### Ask, then act

```text
flow "Hungry helper"
trigger any when contains "hungry"

step where:
  input
  prompt: "Where are you right now?"

step find:
  agent on origin
  prompt: "Find good ćevapi places near {{steps.where.output}}."

step reply:
  emit "{{steps.find.output}}"

output -> same
```

### Describe any inbound photo

```text
flow "Auto-describe photos"
trigger whatsapp when has image

step look:
  vision on capability:vision
  prompt: "Describe this image and read any visible text."
  image: {{trigger.fd_image_path}}

output -> same
```

### Route on keywords

```text
flow "Triage"
trigger any when has text

step route:
  branch
  if {{trigger.text}} contains "refund" -> billing
  elif {{trigger.text}} matches "(broken|crash|error)" -> support
  else -> general

step billing:
  agent on origin
  prompt: "Handle this billing request: {{trigger.text}}"
  stop

step support:
  agent on origin
  prompt: "Triage this technical issue: {{trigger.text}}"
  stop

step general:
  agent on origin
  prompt: "Respond helpfully to: {{trigger.text}}"

output -> same
```

---

## Common errors

| Message | Cause / fix |
|---|---|
| `step '<id>': tool step needs a 'tool' name` | A `tool` step with no `tool:`. Use an **agent** step for open-ended work, or add the real tool name + args. |
| `step '<id>': <type> step needs a prompt` | `agent`/`vision`/`input` need a `prompt:`. |
| `branch goto '<x>' is not a step id` | A branch points to a step that doesn’t exist. Fix the id, or use `stop`. |
| `unknown trigger channel '<x>'` | Channels are `any`/`whatsapp`/`web`/`glasses`. |
| `unknown field '<k>' for <type> step` | Typo’d field name — see the step’s allowed fields above. |
| `input step: no channel available to prompt the user` | An `input` step ran where there’s no reply channel (e.g. a dry test). Trigger it on a real channel. |
| `(no FD-internal tool '<x>')` | `on: fd` only exposes `face_identify`. |
| AI compile: `generated DSL invalid (…)` | The model produced bad DSL even after one self-repair. The raw DSL is in the editor — fix by hand, or rephrase / pick a stronger model. |

---

## Grammar cheat-sheet

```text
flow "Name"
description "..."                 # optional
priority 50                       # optional
trigger <any|whatsapp|web|glasses> [when <rule> and <rule> …]   # or: trigger <ch> always
    rules: has <image|video|audio|text|document>
           contains "x" | from_waid "x" | mime "glob" | regex "re"

step <id>:
  tool   on <selector>            # tool: <name> ; arg <k>: <v>
  agent  on <selector>            # prompt: "..." ; attach: <path> ; deny: a, b
  vision on capability:vision     # prompt: "..." ; image: <path>
  input                           # prompt: "..." ; timeout: <seconds>
  emit "body"                     # or: emit + channel: <ch> + body: "..."
  branch                          # if <cond> -> <target>
                                  # elif <cond> -> <target>
                                  # else -> <target>          (target: <step id> | stop)
  stop                            # (non-branch) end the flow after this step

output -> <same|whatsapp|web|glasses|log>

selectors : origin | fd | any | capability:vision | name:<Agent>
templating: {{trigger.*}} {{steps.<id>.output}} {{steps.<id>.<field>}} {{system.*}}
conditions: and or not ( )  ==  !=  >  <  >=  <=  contains  matches/~  (bare = truthy)
```
