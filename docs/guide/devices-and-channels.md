# Devices and channels — your agents, wherever you are

[← README](../../README.md) · [All guides](README.md)

The same agents, the same conversation, from wherever you are.

**On this page:** [Voice devices](#voice-devices) · [Four devices, one firmware](#four-devices-one-firmware) · [Smart glasses](#smart-glasses) · [Chat channels](#chat-channels) · [WhatsApp Web](#whatsapp-web--your-own-whatsapp) · [Notifications](#notifications) · [Agents in your inbox](#agents-in-your-inbox)

## Voice devices

<p align="center">
  <img src="../screenshots/devices-0.8.2.webp" width="100%" alt="M5Stack StopWatch speaking an answer and showing its clock face, PaperColor e-paper showing its dashboard and an answer">
</p>

Push-to-talk Captain Claw clients on three M5Stack devices and a Waveshare board:

- the **StopWatch**: a 466×466 round AMOLED with a clock face, touch and vibration;
- the **PaperColor**: a 4" colour e-paper dashboard with two status lights and a room sensor;
- the **StickS3**;
- the **Waveshare ePaper 1.54** (0.8.4): a 200×200 black-and-white e-paper board, with or without touch.

Press, talk, press again, or just stop talking. Flight Deck transcribes it, one of your agents answers, and the device **speaks the answer sentence by sentence while the model is still writing it**, the text following the voice. Since 0.8.2 the voice no longer stops at about 66 seconds: Flight Deck speaks up to about 1,500 characters of an answer, and the device shows the rest.

- Out of Wi-Fi, it keeps **voice notes** and sends them when the deck is back.
- Every question carries the battery, the room's temperature and the local time.
- All of them take **signed over-the-air updates** from your own deck.

One firmware tree: `cd firmware/device && pio run -e stopwatch -t upload` (or `-e sticks3`, `-e papercolor_b`, `-e epaper154`, `-e epaper154_touch`). Details in [firmware/device/README.md](../../firmware/device/README.md).

[![Every StopWatch screen: the clock face with the agent's character, listening, working, an answer whose spoken sentence is lit, a background picture, and the info page](../screenshots/stopwatch-screens.png)](../../firmware/device/README.md)

[![Every PaperColor screen: the e-paper dashboard with the clock, room climate, battery and voice notes, and an answer](../screenshots/papercolor-screens.png)](../../firmware/device/README.md)

### Four devices, one firmware

| | StickS3 | StopWatch | PaperColor | Waveshare ePaper 1.54 |
|---|---|---|---|---|
| screen | 240×135 LCD | 466×466 round AMOLED, touch | 4" E Ink Spectra 6, six inks | 200×200 black-and-white e-paper, optional touch |
| talk | front button | yellow button, or the on-screen record button | A (B build) / C (A build) | the upper button (the gear), or a tap in the middle of the screen on the touch board |
| extras | 18 animated characters | clock face, raise to wake, vibration, background picture, info page | two status lights, room sensor, microSD notes, a dashboard every five minutes | a drawn menu, a sleeping clock on a timer you pick, a green status LED, PWR cancels a recording |
| build | `pio run -e sticks3` | `pio run -e stopwatch` | `pio run -e papercolor_b` (or `papercolor`) | `pio run -e epaper154` (or `epaper154_touch`); V2 boards only |

- **Signed over-the-air updates** from your own deck (`tools/publish.py`), rolled back if a build can't reach the deck within 90 s.
- **Press to talk, press again — or just stop talking** (2.5 s of quiet sends it), with blips and chimes; the PaperColor records straight from sleep. On the Waveshare board only a press ends a recording, and a press from sleep only wakes it.
- **Answers as they're written** — spoken sentence by sentence while the model is still writing; the lit sentence and the pages follow the voice; pause, replay, stop.
- **Offline voice notes** (ADPCM in flash, or on the PaperColor's microSD) sent as one turn once the deck is back, and a **note-taking mode** for a run of notes.
- **Context with every question** — battery and time left, Wi-Fi, motion, room temperature and humidity, local time, as a `[Context: …]` line the model reads.
- **"Volume up", "beep off"** handled by the deck, never sent to the agent; a turn survives a dropped link.
- **Something waiting?** (0.8.3, firmware 0.19.1 / PaperColor B 1.0.21) — a beep, a banner on the StopWatch and StickS3, and a magenta count badge on every screen when a proposal or nudge waits for you in Flight Deck. Quiet hours keep the count and skip the beep. The text stays in Flight Deck.
- **A link that renews itself** (0.8.3) — 30 seconds after its agent goes missing (backing off to every 10 minutes) and after 20 minutes idle, never in the middle of something.
- **Asleep or switched off** (firmware 0.19.2 / PaperColor B 1.0.22) — the agent's dot turns blue while Flight Deck has put it to sleep for being idle (talking wakes it), red while it's stopped; the PaperColor writes the agent's name in the same colour.
- **Waveshare ePaper 1.54** (0.8.4, firmware 0.19.10) — partial refresh in 0.6 s and full in 1.8 s, fast enough for a drawn menu and answers that page with the voice. Double-click PWR (or double-tap the touch board) for home; the sleeping screen keeps the clock, with a sleeping robot; the green LED is steady while recording, breathes while the agent works and blinks twice when something waits. Its onboard sensor isn't used, because the board's own heat throws it off. Flash it once over USB; updates come over the air after that.

## Smart glasses

- **Meta Ray-Ban Display HUD** (0.8.0) — a lean Flight Deck at `/hud/` built for a 600×600 display: chat with your agents (approvals, Stop, read-aloud), read their Markdown files, browse the datastore; about 150 KB on the wire, paired with a code you approve at `/hud/pair`. [docs/glasses-hud.md](../glasses-hud.md)

## Chat channels

- **WhatsApp** — a two-way personal-assistant bridge (Meta Cloud API). Every file you send reaches the agent — documents, albums, stickers, voice notes transcribed; agents send pictures, video, audio and documents back and deliver what a turn made once the reply is out. Messages sent while the agent is busy are retried in order, every message gets a reaction, `/new` answers.
- **Telegram** — a native bot with photos, documents, contacts, voice notes, audio and video, and live narration; `/new` starts a fresh session; a team can share one bot, each member relayed to their own agent after admin approval; the owner's own chat continues the main conversation; agents read a read-only **Telegram inbox** and join **groups** through the bot's add-to-group link (0.8.1).
- **One conversation** (0.8.1) — every message records where it came from; the model reads `[via WhatsApp] …`, the kiosk shows *via M5Stack StickS3 (desk)*.
- **Claude** — Flight Deck is an MCP server at `/fd/mcp-server` (OAuth 2.1 for claude.ai and Claude Desktop, a personal access token for Claude Code): `list_agents`, `send_task`, `get_result`, `cancel_task`, `jev_ask`.
- **Slack, Discord, Twitter/X** — message and thread handling, channels and DMs, and an OAuth 2.0 `twitter` tool.
- **WhatsApp Web** (0.8.3, experimental) — your own WhatsApp as a linked device; see [below](#whatsapp-web--your-own-whatsapp).

A refused WhatsApp Cloud API send logs Meta's error code (0.8.3), and a de-registered number is an error that names the fix (`POST /<phone-number-id>/register`); the Connections test button says so too.

## WhatsApp Web — your own WhatsApp

New in 0.8.3, **experimental and opt-in**.

> WhatsApp Web support uses an unofficial client ([neonize](https://github.com/krypton-byte/neonize)). That is against WhatsApp's terms of service, and WhatsApp can ban an account that uses one. It is off until an admin allows it (*Allow WhatsApp Web*) and each user consents, and it isn't available in the desktop app. Install it with `pip install 'captain-claw[whatsapp-web]'`.

- **Link your own number** in **Connections → WhatsApp Web** by scanning a QR code or typing a pairing code, like any linked device. Flight Deck keeps the history, sealed at rest, for as long as you choose (7, 30, 90, 365 days, or everything).
- **Your agents read it** with the `whatsapp_web` tool (`status`, `chats`, `read`, `recent`, `search`, `contact`, `attachment`, `send`): chats, messages, contacts and attachments, with voice notes transcribed when Soniox is set up. Each turn starts with a line on what came in since the agent's last turn. Only the owner's agents read it; shared-agent members never do.
- **Sending is off by default.** When on, an agent sends only to existing chats you whitelisted (it never starts a conversation), within a daily limit (30, up to 200), 6 a minute per chat, and never the same text to the same chat twice in 24 hours. Every message carries your signature. It never reacts, marks messages read or changes your presence. The fabrication guard catches a claimed WhatsApp send that didn't happen.
- **Auto-replies in chats you choose.** Tick *auto-reply* on a whitelisted chat, pick one of your agents and its trigger words. The agent answers when it's named, called by a trigger word or `@word`, or replied to, and may keep talking for 15 minutes after its last message. It sees only that chat's last 30 messages and may run one web search; never your mail, files, calendar, memory or other chats. At most 10 answers an hour per chat; messages older than 10 minutes and other assistants are ignored; nobody in the chat can change its rules.
- **Auto-reply activity** shows each decision: a one-line reason, the reply or the reason for silence, the model's reasoning, the search, tokens and time.
- **A ban doesn't take your history.** **Disconnect** and **Switch number** ask whether to keep it; if WhatsApp bans the number or the device is removed on the phone, it's kept. Agents go on reading it, nothing is sent from it, and you delete it on the card when you want to.
- Kill switches: `FD_WHATSAPP_WEB=off`, `FD_WHATSAPP_WEB_SEND=off`, `FD_WHATSAPP_WEB_AUTOREPLY=off`.

## Notifications

Since 0.8.3, Autonomous Work's nudges and the proposals waiting for your approval reach you wherever you choose on the **Notifications** card in Connections (also in the kiosk's Settings and the admin console):

- **WhatsApp** (on), **Telegram** (off until you turn it on) and **your devices** (on: a beep, a banner and a count);
- **the bell** in Flight Deck, always, with **Approve / Reject** on each proposal and a ⤢ bigger view. Nudges and proposals render as Markdown, with pictures dropped and links shown as text.

Quiet hours send the count without the beep. A channel that fails never stops the others.

## Agents in your inbox

- **Microsoft 365** — `microsoft_mail`, `microsoft_calendar` and `microsoft_onedrive`, the Outlook twins of the Google tools: per user, owner-only. Outlook mail reads, searches, drafts and files; it **never sends**. Calendar changes that would email attendees need `notify_attendees: true` and a turn where a person asked.
- **Gmail sends — opt-in, audited** — `google_mail` gains `send`, `send_draft`, `update_draft`, `list_drafts`; under Flight Deck every send passes one gate with a per-user policy (off by default), a recipient allowlist, a daily limit, duplicate suppression, an audit table and a bell per send. `FD_GMAIL_SEND=off` is the deck kill switch.
- **Never the same email twice**; **attachments both ways** (`get_attachment` in, `saved/` and `vfs:` files out — 20 files, 25 MB, secrets refused); **filing** (archive, move, labels, categories — never trash); **Google Sheets and Docs edited in place**; an **agentic signature** on every email an agent writes.
- **Topic auto-replies** — off by default: an agent answers inbound Gmail that is *only* about one of your narrow topics, behind a lint, sender authentication, an outbound guard, a second judge, and held cards for anything doubtful. `FD_GMAIL_AUTOREPLY=off` stops it deck-wide.
- **Agents that don't make things up** — mail is human-only in Autonomous Work and automated turns; a **fabrication guard** rejects a claimed send or calendar change with no tool call behind it and asks, at most twice, for a true reply; a garbled reply gets a notice recommending `/new`.
