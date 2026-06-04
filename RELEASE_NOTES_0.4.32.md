# Captain Claw v0.4.32 Release Notes

**Release title:** Flows — the Process Engine

**Release date:** 2026-06-04

## Highlights

Captain Claw 0.4.32 introduces **Flows** — a declarative process/automation engine that runs **inside Flight Deck** and dispatches steps to your **agent pool**. A Flow is a trigger plus an ordered list of steps; deterministic plumbing (triggering, routing, sequencing, guardrails) is owned by Flight Deck, while agents do the judgment work as steps. Build them in a form-based UI with a live run log — no code.

This release also ships a **lean `vision` step** (raw image-describe with no agent baggage) and a large batch of **image / inter-agent reliability** fixes hardened across real WhatsApp + glasses use.

## Flows — the engine

- **Declarative spec, sqlite-backed, UI from day 0.** Trigger → steps → output, with guardrails. Stored in `flows.db`; runs and per-step results are first-class so the UI shows a live, step-by-step **run log**.
- **Step types:**
  - **`tool`** — a deterministic single-tool RPC to a pooled agent (`/api/tool`), no LLM turn.
  - **`agent`** — a scoped judgment turn on a pooled agent (consult), with optional file `attach` and per-step tool guardrails.
  - **`vision`** — *new primitive:* a **raw image-describe** (`/api/vision`) with **no agent loop, memory, tools, or history** — the reliable way to describe an image.
  - **`branch`** — `when` → `goto` conditional.
  - **`emit`** — send to a channel.
- **Triggers:** rule-based matching on inbound messages (cheap, rules-first) — `has_image`/`has_video`/`has_audio`/`has_text`, plus custom rules (`contains:…`, `from_waid:…`, `mime:…`, or a bare word = substring match). Fires on **WhatsApp, glasses, and web**; no match → the normal agent turn (inert until you enable a Flow). Cron/`always` also supported.
- **Templating:** `{{trigger.*}}`, `{{steps.<id>.output}}`, and a real `{{system.*}}` namespace (now/date/time/agent/channel). The builder surfaces every variable as **click-to-insert chips**.
- **Agent selection / affinity:** `origin` (the triggering agent), `capability:vision`, `name:<agent>` (dropdown populated from the live fleet), or `fd` (internal tool). File-bound steps run where the file is; cross-agent steps transfer the file (verified) and use the **target-local path**.
- **Image Flows override the built-in WhatsApp image automation** when a Flow matches (selective by trigger), falling back to the built-in otherwise.

## Reliability hardening (image + inter-agent)

- **No-resend gate:** relaying a delegated result can no longer auto-resend the task (deterministic — `flight_deck`/`consult_peer` denied on relay turns). Fixes the inter-agent message flood.
- **Self-delegation blocked** (an agent can't delegate to itself).
- **Busy peers queue** instead of being rejected; busy-retry on the delegate path.
- **Auth fix:** peer consults resolve the token **by agent** (not by a colliding port), fixing 401s; `origin` resolves to the live fleet entry.
- **Capability-aware image hint:** inline-vision agents are told to look directly (don't call `image_vision`, don't delegate).
- **Rich-session contamination fixed:** image-describe turns suppress memory/insights injection so the model describes the *attached* image, not a remembered one.
- **Attachment correctness:** files are uploaded to the target and verified before use; `shell/scripts/read` denied on attach so the model uses the image, not the path.
- **WhatsApp markdown:** `**bold**` → `*bold*` so replies render cleanly.

## New endpoints (agent web server)

- `POST /api/tool` — deterministic single-tool execution (admin-locked).
- `POST /api/vision` — raw model image-describe (admin-locked).

## How to use

In Flight Deck → **Flows** → New Flow:

```
Trigger:  on=message, channel=any, rule = has image
Step 1 (vision): Run on = <your vision agent>, Prompt = "Describe this image.",
                 Image to look at = {{trigger.image_path}}
Output:   same (reply on origin)
```

Enable it, send a photo (WhatsApp / glasses / web), and watch the run log. Add a `branch`/`agent`/`emit` step to build multi-step automations.

## Backward compatibility

Fully compatible with 0.4.31. Flows are additive and inert until you create + enable one. No migration required.

## Upgrade

```bash
git pull
# rebuild the Flight Deck UI only if you build assets locally (they're committed):
npm --prefix flight-deck run build
# restart Flight Deck; (re)spawn agents to pick up the new /api/tool + /api/vision
# endpoints and the agent-side guardrails.
```
