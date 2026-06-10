# Captain Claw v0.5.3 Release Notes

**Release title:** Free agents — one OpenRouter key, a zero-cost fleet
**Release date:** 2026-06-09

The headline is **Quick Free Agent**: paste a single free OpenRouter key and
spawn an agent that runs entirely on **free, tool-capable models** — and switch
between them live, right on the card. Around it, the **desktop standalone app**
got a real overhaul (process-first onboarding, reliable file writes, restart
fix) and **Flight Deck** picked up a round of UI polish. Additive and backward
compatible with 0.5.2.

---

## What's new

### Quick Free Agent (OpenRouter)
A green **Quick Free Agent** button on the Spawn Agent page opens a guided modal:

- Step-by-step instructions to grab a **free** OpenRouter API key.
- One click fetches the **currently-free** models straight from OpenRouter's
  public catalog (no key needed just to list).
- The list is **filtered to tool-capable models** — Captain Claw agents live on
  tools, so models that can't call functions are hidden automatically.
- Pick a default; **every** free model is added to the agent's *allowed* list so
  you can switch between them at runtime.

Free agents wear a **"Freebie"** badge. When one is **stopped**, the card's
Actions menu offers **Refresh free models** — it re-fetches the current free
roster and rewrites all three of the agent's config files in place, so the
fleet stays current as OpenRouter's free tier changes.

### Live model switching on the card
Agent cards now show an **Active model** dropdown listing the agent's allowed
models. Switching is **live over the chat channel — no restart** — and takes
effect on the next message. (Process cards gained the picker that container
cards already had.)

### Desktop standalone, reworked
- Flight Deck opens straight into a clean **"No agents yet → Create an agent"**
  flow that routes to the Spawn Agent page; the old single-agent landing is gone.
- The first supervisor now spawns as a **local process** (not Docker), so its
  `write` tool saves reliably — fixing the Docker-Desktop bind-mount write
  failures on macOS.
- **New agents deploy in eco mode** by default (compact prompts, lazy tools).
- Spawned agents always get a correct **`FD_URL`** callback, derived from Flight
  Deck's actual bound port.

### Flight Deck polish
- **Agent cards:** Actions moved to a gear (⚙) in the top-right next to the size
  toggle; **Chat** and **Open** now sit in the action row with Files/Logs/Data/
  Intentions. The **Open** menu renders above the card instead of being clipped.
- **Director:** per-agent **show/hide** toggle keeps the desktop tidy; the header
  reads **"X of Y"** visible.
- **Chat:** intermediary tool calls + between-step narration are grouped into a
  neutral, **collapsible Activity panel** that shows the whole run and folds away
  once the turn finishes.
- **Sidebar:** menu reordered and trimmed (Workflows / Projects / Code Apps
  removed from the nav).
- **Spawn page:** header is **sticky**, so Quick Free Agent / Save stay reachable
  while you scroll the form, which now prefills an **`FD_URL`** env var.

---

## Fixes
- **Process agents start after an app restart.** Start / restart / reattach now
  resolve the bundled `captain-claw-web` binary (PyInstaller standalone) instead
  of assuming it's on `PATH` — previously they failed silently in the packaged
  app. Failures are now logged.
- Flight Deck records its actual bound port (`FD_PORT`) so the auto-injected
  `FD_URL` is correct even when the default 25080 isn't in use.

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild the frontend only if you build locally:
npm --prefix flight-deck run build
```

- **Desktop app:** rebuild with `./build-desktop.sh` — the Quick Free Agent
  spawn/refresh endpoints, the eco-on-spawn flag, the FD_URL/port handling, and
  the process-start fix all live in the `captain-claw-fd` / `captain-claw-web`
  binaries.
- **Restart Flight Deck** to pick up the backend changes; hard-reload
  (`Cmd+Shift+R`) for the committed UI.
- The live **Active model** dropdown populates once an agent is running and has
  an *allowed* list (so spawn a fresh Freebie on the new backend to see it).

Backward compatible with 0.5.2 — everything here is additive.
