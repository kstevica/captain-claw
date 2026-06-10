# Captain Claw v0.4.33 Release Notes

**Release title:** Flows, Grown Up — code, conditions, conversations, and faces

**Release date:** 2026-06-04

## Highlights

0.4.33 turns **Flows** from a one-shot trigger→steps engine into a real automation language. You can now:

- **Write flows as code** — a clean declarative DSL with a live syntax checker, *or* describe them in plain English and let a model compile them (validated by the real parser).
- **Pause for input** mid-flow and resume on the user's reply — on **any channel**.
- **Branch richly** — `and`/`or`/`not`, comparisons, `contains`/`matches`, and multi-case switches — and **stop** a flow from any path.
- **Recognize and enroll faces hands-free** over WhatsApp/glasses with a simple `face` mode.

Plus an in-app **Flow language reference** and a heavily-commented **example flow** to learn from.

---

## Flows

### Write flows as code (declarative DSL)
A flow is now also a small text program that mirrors the visual builder 1:1 — switch between **Builder** and **Code** freely; they round-trip losslessly.

```text
flow "Hungry helper"
trigger any when contains "hungry" or contains "gladan" or contains "gladni"

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

- **Deterministic parser/validator** with precise `line N: <error>` messages — bad code is never silently saved.
- **`tool` / `agent` / `vision` / `input` / `emit` / `branch`** step types; selectors `origin` / `fd` / `any` / `capability:vision` / `name:<agent>`.
- `{{trigger.*}}`, `{{steps.<id>.output}}` (+ flat JSON fields like `{{steps.id.name}}`), and `{{system.*}}` templating.

### AI compiler (English → flow)
Type a description, pick which agent's model compiles it, and the model writes canonical DSL that's then **run through the real validator**. Invalid output is rejected (with the error and raw DSL shown), and a **one-shot auto-repair** feeds the error back to the model to fix itself. Smarter guidance keeps it on the rails (open-ended work → `agent` step, not a bare `tool`; default trigger channel `any`).

### Pause for user input — on any channel
A new **`input`** step pauses the run, messages the user (always naming the flow), and resumes with their reply as `{{steps.<id>.output}}`. Works on WhatsApp **and** the web/glasses surfaces: flows that pause for input or consult the `origin` agent now run in the background and deliver asynchronously, so the originating agent is never deadlocked.

### Richer branching + stop
- Conditions support **`and` / `or` / `not`**, parentheses, **`==` `!=` `>` `<` `>=` `<=` `contains` `matches`**, and bare-operand truthiness — evaluated safely (never `eval`).
- A branch is a **switch**: `if / elif / else -> <target>`, first match wins.
- **Stop a flow** from any path: a per-step "Stop after this step" flag, or a branch target of `stop`.

### OR triggers
Trigger rules combine with **`and`** (all match) or **`or`** (any matches) — so "fire on any of these words" finally works. The Builder has a **Match: ALL / ANY** toggle.

### Learn it in-app
- **📖 Flow language docs** button in the Code view renders the full reference (see `FLOWS.md`) without leaving the app.
- **Load example** drops in a guided, heavily-commented flow you can explore and "Validate & apply".

---

## Faces (glasses, over WhatsApp)

Because captionless glasses photos carry no text, a sticky **face mode** drives them:

- `face on` — recognize faces in incoming photos; a face → person card, **no face → the scene is described** automatically.
- `face off` — stop.
- `face enroll <name>` … `face enroll off` — save the next photos as that person.

Commands work with or without a leading slash and accept natural phrasings (`recognition on`, `enroll Ana`), while ordinary chat (`face on the wall…`) passes through untouched.

---

## New / changed endpoints (Flight Deck)

- `POST /fd/flows/dsl/compile` — DSL text → flow (structured errors)
- `POST /fd/flows/dsl/decompile` — flow → DSL text
- `POST /fd/flows/compile` — English → validated flow (model-assisted)
- `GET  /fd/flows/docs` — serves `FLOWS.md` for the in-app viewer
- `POST /api/chat/push` (agent) — async delivery into a chat UI (web/glasses flow output)

---

## Documentation

- **`FLOWS.md`** — a thorough Flow language reference: triggers, step types, templating, the branch condition language, stopping, channels, the AI compiler, a cookbook, a common-errors table, and a grammar cheat-sheet. Linked from the README and the in-app docs button.

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild only if you build locally:
npm --prefix flight-deck run build
# Restart Flight Deck, and (re)spawn agents to pick up /api/chat/push and the
# agent-side flow guardrails.
```

Backward compatible with 0.4.32 — Flows are additive and existing flows keep working (single-condition branches and AND-only triggers are still honored).
