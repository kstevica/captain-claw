# Captain Claw v0.4.28 Release Notes

**Release title:** WhatsApp PA & Intentions

**Release date:** 2026-06-02

## Highlights

Captain Claw 0.4.28 is a big, agent-native step: WhatsApp becomes a genuine **two-way personal-assistant channel**, and a brand-new **Intentions** primitive lets the assistant hold, propose, and act on *future* actions — with your permission. Around those two headliners it also lands a proactive **Flight Deck scheduler**, a **glasses dashboard** with **face recognition**, reliable **fleet collaboration**, and a set of **Eco-mode and thinking-model reliability fixes** that make all of the above actually show up and not crash.

Everything new is **additive and opt-in**. Existing 0.4.27 setups keep working unchanged; the WhatsApp bridge, scheduler, and the proactive Intentions generator stay dormant until you configure them.

## What changed

### 1. WhatsApp bridge — a two-way PA channel (`captain_claw/flight_deck/whatsapp_bridge.py`)

A full Meta WhatsApp Cloud API bridge that routes a WhatsApp chat to a Flight Deck agent and back:

- **Inbound**: text, **voice notes** (transcribed via Soniox STT and forwarded as text), location, and contacts (formatted as FYI context).
- **Outbound**: text replies, optional **voice replies** (Soniox TTS MP3 when `WHATSAPP_AUDIO_REPLY=on`), and now **documents**.
- **Document delivery** — the new `whatsapp_send_file` tool sends a file the agent saved (its `saved/` workspace) to **the current WhatsApp chat** by default, or to any allow-listed number via `to`. Find files by `path`, fuzzy `filename`, or `latest`; `action: "list"` for discovery. An explicit MIME map guarantees Meta-accepted types for `.pptx/.docx/.xlsx/.pdf`, and text formats (`.md/.html/.py/.csv/…`) are sent as `text/plain` so they deliver as named documents instead of being rejected.
- **Allow-list + slash commands** — `WHATSAPP_ALLOWED_WAIDS` gates every send; `/c <channel>` rebinds, `/mute`/`/unmute` control proactive pushes.
- **Proactive push** — `push_to_waid()` / `POST /whatsapp/push` deliver agent-initiated messages (used by the scheduler and the Intentions generator), honouring mute and the allow-list.

### 2. Caption-routed inbound images + face enrollment

Inbound photos are **no longer kidnapped into face recognition**. The image **caption** (or, for a bare photo, your follow-up reply) routes it:

- *"who is this?" / "tko je ovo"* → **face recognition** (Flight Deck `face_index`).
- *"remember this is Alice, colleague from X" / "ovo je Alice"* → **face enrollment** so they're recognised next time.
- anything else (*"summarise this slide", "read this"*) → forwarded to the **agent's vision** pipeline with the caption as the prompt.
- a **bare photo** → the bot asks what to do and routes your next reply.

Face recognition (recognise + enroll) stays **entirely on Flight Deck** — never an agent tool. Multi-face recognition is supported.

### 3. Intentions — proactive, permissioned future actions

A new control-plane primitive (`captain_claw/intentions.py`, the `intentions` tool, a Flight Deck **Intentions** panel) sitting between *noticing* (insights) and *doing* (cron/scheduler):

- **User intentions** — notes-to-self, surfaced back into the agent's context when relevant.
- **Agent intentions** — proactive actions the agent **announces** (low-risk / read-only) or **asks permission** for (anything that sends or changes data). Risk drives the mode automatically.
- **Channel-agnostic decision bus** — a pending decision can be resolved by a **freeform WhatsApp reply** ("yes/no/later", any wording, interpreted by the agent) *or* a **Flight Deck panel button** (Approve / Decline / Later / Stop). Both converge on one resolution path.
- **Follow-through** — approving a **repeatable** intention **materialises a Flight Deck scheduler job**; declining writes a **negative-feedback insight** so the agent won't re-propose it; announce-mode supports an undo window.
- **Phase 3 — proactive generator** (`captain_claw/intentions_generator.py`, opt-in) — a cooldown-gated, quiet-hours-aware background pass that reviews recent messages + insights and proposes new intentions, deduped against active + declined ones. Tunable via `intentions.proactivity` (`conservative` / `balanced` / `eager`), `interval_hours`, `max_per_day`, and quiet hours. Off by default.
- Agent HTTP endpoints (`/api/intentions`, `/api/intentions/decisions`, `…/resolve`) + an FD→agent proxy power the panel.

### 4. Flight Deck scheduler (`captain_claw/flight_deck/fd_scheduler.py`)

Recurring and one-shot jobs that run an agent turn and **deliver the result** to WhatsApp, the glasses channel, or Telegram. Schedule formats like `daily 09:00`, `weekly fri 17:00`, `every 15m`, `in 3d`, `once <ISO>`. Respects quiet hours; REST-managed via `/scheduler/jobs`. This is the engine approved repeatable intentions materialise into.

### 5. Glasses dashboard & face recognition

- Multi-face recognition with enrollment, surfaced through the WhatsApp/glasses pipeline.
- A Flight Deck file-preview dashboard for browsing agent-saved files.

### 6. Fleet collaboration always available

The `flight_deck` tool (list / consult / delegate / spawn peers) is now part of the Eco-mode core set, so an agent can reliably reach other agents on demand (e.g. *"ask deepseek what's new"*). It self-discovers peers via `/fd/fleet`, so it works whenever `FD_URL` is set and a peer is running.

### 7. Reliable tool availability in Eco mode

Eco mode defers most tools and only surfaces them on intent match — which previously hid important capabilities. Google (Gmail/Drive/Calendar), `whatsapp_send_file`, `intentions`, and `flight_deck` are now **always offered** (Eco-core / always-enabled), so the assistant doesn't silently "lose" its PA toolkit.

### 8. Thinking-model reliability fix (`captain_claw/llm/__init__.py`)

Stall-retry logic forces `tool_choice="required"`, but thinking/reasoning models (e.g. DeepSeek thinking mode via an OpenAI-compatible endpoint) reject it with a 400 and crash the turn. The provider now **retries once without `tool_choice`** on that rejection and remembers it for the session — so forced-tool retries never crash on models that don't support `tool_choice`.

## How to use

**WhatsApp file delivery** — over WhatsApp: *"send me that report"*, *"whatsapp me the last document"*. The agent picks the file from its saved workspace and delivers it as a document to the current chat.

**Intentions (manual)** — tell the agent *"set up a weekly Monday portfolio brief"*; it asks, you reply *"yes"*, and it schedules a recurring job. *"remember I want to follow up with X"* records a note that resurfaces. Manage open intentions and pending decisions from the **Intentions** panel on each agent card in Flight Deck.

**Intentions (proactive, opt-in)** — in the agent's `config.yaml`:

```yaml
intentions:
  auto_generate: true        # off by default
  proactivity: balanced      # conservative | balanced | eager
  interval_hours: 6
  max_per_day: 4
  quiet_hours_start: 22
  quiet_hours_end: 8
  push_to_whatsapp: true     # ping the current WhatsApp chat with new proposals
```

**WhatsApp bridge setup** — see [docs/whatsapp-bridge-setup.md](docs/whatsapp-bridge-setup.md) and the env vars (`WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_APP_SECRET`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_ALLOWED_WAIDS`, optional `WHATSAPP_AUDIO_REPLY`).

## Backward compatibility

Fully backward compatible with 0.4.27. New surfaces are opt-in:
- The WhatsApp bridge is dormant unless its env vars are set.
- The Intentions proactive generator is off unless `intentions.auto_generate=true`.
- Eco-core/always-enabled tool changes only add availability; nothing is removed.

## Upgrade

```bash
git pull
# Python deps unchanged; rebuild the Flight Deck UI if serving the bundled assets:
npm --prefix flight-deck run build   # only if you build the frontend locally
# restart the agent(s) and Flight Deck
```
