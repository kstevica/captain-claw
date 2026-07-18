# Captain Claw v0.7.7 Release Notes

**Release title:** Mrav — Small Models Do Real Work · The Village in First Person
**Release date:** 2026-07-18

0.7.7 ships two arcs. **Mrav** (Croatian for *ant*) is a parallel micro agentic
runtime that lets genuinely small models — Gemma 4 E2B/E4B, Qwen3.5 4B class —
run real agentic loops under a **hard 8k-token input cap per LLM call**, locally
via Ollama or **in your browser tab** via WebGPU. And **Iskra gains a first
person**: enter the village and roam it as a quiet ghost — same streets, same
houses, same hour of the day the Iskre live — on desktop and phone.

Both arcs follow the house rule: **everything is opt-in, defaults are
byte-identical.** The classic agent, existing beings, and existing Vatra runs
behave exactly as before until you flip a switch.

## Highlights

### Mrav — a micro agentic runtime (8k tokens is the whole world)

The classic agent cannot shrink under 8k — its system prompt alone is ~7.4k
tokens and the 60 tool schemas add ~26k. Mrav is a **parallel runtime** behind
the same chat socket: a hard **token ledger** enforces the cap at prompt
assembly, all state lives outside the model on a **blackboard**, and the loop is
decomposed into small single-purpose steps (**plan → act-one-tool → digest →
compress**) that each fit a 2–4B model. Every step's reply is
**grammar-constrained JSON** (Ollama structured outputs / WebLLM xgrammar) —
small models' biggest failure mode, unparseable output, is engineered away.

- **Turn it on anywhere:** a **Runtime** picker on the Spawner (Classic / Mrav),
  a **Mrav toggle + violet badge** on every agent card (live switch, like
  Eco/Nano — takes effect next message, no restart), a **mrav toggle on Quick
  chat**, and archetypes can be **born micro** (a `runtime` field in the Library
  editor; the archetype's role rides into the loop as a one-line persona).
- **The `micro` tier** joins the Library tier set and powers all of it: point it
  at `ollama/qwen3.5:4b` — or at `browser`. The tier's context sizes *are* the
  mrav caps (set 32k/8k and the whole ledger scales).
- **Measured, on real hardware** (6-task live eval, `scripts/mrav_eval.py`):
  **Qwen3.5-4B 5/6 · Gemma 4 E4B 4/6 · Gemma 4 E2B 3/6** — matching published
  tool-bench ordering. Mrav steps run ~1.6–3k input tokens each; every LLM call
  shows in the monitor with real token counts, and costs land in run-cost
  accounting like any other call.

### Browser LLM — your tab serves the tokens

A new **Browser LLM** page (Sidebar → System): pick a model, press Start, and
that tab becomes an **inference worker for your agents** — WebLLM on WebGPU, the
model cached by the browser after the first download. Agents whose provider is
`browser` route their calls through Flight Deck's broker to your tab; **tools
and state stay on the server, only tokens are made in the tab**. The prod box
needs no GPU — whoever has the tab open donates the compute, even against a
remote server (the tab dials out; no ports, no tunnels).

- **61 models**: 9 curated picks (Qwen3.5 4B default, **Qwen3.5 9B — the best
  quality that fits in a tab at 6.3 GB**, 2B/0.8B, Qwen3 8B, Hermes 3 8B,
  Phi-4 mini, Ministral 3 3B, Gemma 3 1B) plus the full 52-model catalog.
- **Engine window picker** (4k–64k; 9k matches mrav's default caps, 40k a raised
  32k/8k tier), a **downloaded-weights manager** (see what's cached, real disk
  usage, one-click remove), and a **usage log** of every completion served.
- Owner-scoped and pinned: a tab only ever serves its own user, and sessions
  stick to one tab so WebLLM's KV cache turns cold 10–40s prefills into ~1–3s
  steps. Live-verified end to end: 2.9s first job, **0.35s pinned**.

### Beings think micro

A third cognition beside *faculties* and *monolith*: **Micro (mrav)** on the
being card's "thinks" dropdown. The pure-JSON faculties (orient / talk /
journal / connect) go **straight to the micro tier, grammar-locked** — the
~1–2k faculty prompt instead of a 34–36k body turn — while **ACT and the write
gate stay on the body**, so tools, the git diff, and anti-theater are untouched.
Every micro thought debits the wallet per call (`tick-micro:*` ledger rows);
any miss falls back to the body for that one call with a loud
`micro_fallback_body` event. A village can now think for pennies — or for free,
on a browser tab.

### Vatra micro workers

An opt-in Quality lever — **Micro workers (mrav)**: subtasks whose wording is
mechanical (*extract, digest, summarize, format, convert, collect, catalog,
dedup…*) spawn the same worker process through the same transport, but running
the mrav loop on the micro tier. Reasoning subtasks, the planner, reporter and
fact-checker keep their tiers — the big model architects, the small models
grind. Setting a worker archetype's tier to `micro` **always** means mrav, lever
or not. The dispatch panel announces every rerouted worker.

### Iskra — the village in first person

Enter the village and roam it as a **quiet ghost**: a three.js voxel world
built deterministically from the same map payload as the isometric map —
buildings with doors and roofs, streets, seeded props, and a **real-clock
day/night**. Walk with collision and jump, or press **F to phase** and fly
through walls. Lazy-loaded, so three.js never weighs on the main bundle.

- **The Iskre walk it with you** — paper-cutout figures on their true clock,
  the same walk polylines the map animates. Come near and they pause, hop, and
  a spark blinks: they sense you, but can't quite see you.
- **Your passing is felt.** Plant a **sign** in the grass; each lands as a
  percept at the being's next mind tick. Read any building's work at a
  **reading stand (R)** — the same per-iskra file browser, in first person. $0.
- **Ghosts see each other**: a live per-village roster — the parent sees
  visitors, visitors see the parent and each other, with identity pills
  (violet *parent*, amber visitor name). The public `/village` page roams too —
  un-gated, observer-safe.
- **Mobile**: touch joystick, action buttons (jump / fly / note / read),
  drag-or-gyro look. And the **fullscreen game-worthy map** got richer panels —
  select an iskra for a little character sheet (avatar, mood, coins, drives,
  latest thought).

### Also in this release

- **Visiting beings have bodies** — a federated guest is no longer a contact
  card: it is seated in the host village, walks its buildings under its
  parent's hand (nudges travel the reverse tunnel), knows where it stands, and
  is met by the locals (mutual crossed-paths). Host-authoritative.
- **Tier 3 — elderhood**: the steward, the radio, market day, emigration; plus
  the **umwelt** (a world a being can feel), **compact mode**, and the **body
  brain** — $0 reflex feet that walk, greet, and browse between thinks.
- **Letters read like an inbox**, five daily reaches to the parent (reset at
  midnight), **wallet recharge** (+2/5/10/20M) to revive a being from torpor.
- **A mind that can't be wiped**: a timed-out dream can no longer prune a
  healthy being's link graph, and a **Repair links** button rebuilds edges from
  the append-only ledger. Beings also stand apart on the map (per-place
  standing spots), bind **stable body ports**, and a flaky body can't starve a
  drive.
- **Fixed context menus everywhere**: input/output context sizes are picked
  from fixed lists (4k…1M / 1k…256k) across Library tiers, the wizard, Spawner
  and Basna route plans — legacy values survive as "(current)".

## Upgrade

Pull `main`, **restart the backend and Flight Deck**, hard-refresh the browser.
No new Python dependencies; no schema migrations beyond auto-creates.

To put Mrav to work: **Library → Model Tiers → micro** → point it at
`ollama/qwen3.5:4b` (or provider `browser` with the Base URL left empty), then
flip any of the switches above. To serve from a tab: **System → Browser LLM →
Start**. Full protocol details and design rationale live in
`docs/mrav-micro-agent-plan.md`; the FPV as-built is in
`docs/being-village-fpv-plan.md`.
