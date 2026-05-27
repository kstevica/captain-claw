# Captain Claw v0.4.27 Release Notes

**Release title:** Glasses Bridge

**Release date:** 2026-05-27

## Highlights

Captain Claw 0.4.27 ships a brand-new **Glasses Bridge** — a mobile-web → agent → glasses-web pipeline that lets you talk to any Flight Deck agent from your phone and have its reply rendered (and spoken) on Meta Ray-Ban Display smart glasses. Plus **Tavily** as a web-search provider — a huge thank-you to the Tavily team for an excellent search API that's a pleasure to integrate.

The bridge is three small pages glued together by a channel-based pub/sub bus living inside Flight Deck:

- **Mobile bridge** — pick an agent, type a message, optionally attach a photo, hit Send. Installable as a PWA on iOS and Android.
- **Glasses view** — last 3 messages rendered with full markdown (including tables), optional Soniox TTS (real-time streaming, 60+ languages, 28 voices), pulsing "thinking" indicator while the agent works.
- **Settings page** — tap-target button grids for voice and language (dropdowns are unusable with Neural Band gestures, so they're hidden behind a dedicated settings page).

Microphone access is not exposed to web apps on the glasses today, so input comes from the phone (text plus optional photo); the glasses are the output surface.

Everything is additive and opt-in — existing 0.4.26 setups keep working unchanged. The Glasses Bridge is dormant unless you visit `/glasses/*`. Tavily is a per-config-key opt-in (the default web-search provider remains Brave).

## What changed

### 1. Glasses Bridge — `captain_claw/flight_deck/glasses_bridge.py`

A new Flight Deck module with three HTML pages, an in-memory channel bus, and a focused set of routes:

**Routes**
- `GET /glasses` — redirects to `/glasses/mobile?c=<random>` (zero-config entry).
- `GET /glasses/mobile?c=X` — mobile bridge UI (PWA-installable).
- `GET /glasses/view?c=X` — glasses HUD page (PNG icons + manifest so Meta's wearables launcher picks up the brand).
- `GET /glasses/settings?c=X` — tap-target settings page (voice, language).
- `GET /glasses/agents` — JSON list of running Flight Deck **process** agents the mobile bridge can target.
- `POST /glasses/send` — `{channel, host, port, text, image_path?}` — routes the message to the chosen agent via its WS.
- `POST /glasses/upload-image` — multipart proxy that forwards a photo to the agent's `/api/image/upload` and returns the resulting path.
- `POST /glasses/tts` — one-shot Soniox TTS (returns audio bytes for `<audio>`).
- `WS /glasses/tts-stream` — streaming Soniox TTS (PCM s16le @ 24 kHz binary frames + a `{type:"info"}` header).
- `WS /glasses/ws?c=X&role=mobile|glasses` — channel pub/sub.
- `GET /glasses/manifest.webmanifest` — PWA manifest for the mobile bridge.
- `GET /glasses/view-manifest.webmanifest` — separate manifest for the glasses launcher (PNG icons; Meta does not accept SVG).
- `GET /glasses/sw.js` — minimal service worker (no caching — the project hinges on fresh-from-server every load).

**Architecture**
- **Channel bus** — channel id is a URL param (`?c=`), kept as an in-memory dict on Flight Deck. Mobile, glasses, and any other tab on the same channel all subscribe to the same bus and see the same events.
- **Per-channel persistent outbound WS to the agent** — bridge keeps one WebSocket to the bound agent and fans `chat_message` events back out to channel subscribers. Re-binding to a different agent cancels the old upstream and rebinds cleanly.
- **Hidden glasses system context** — the first message of every channel→agent binding gets a prepended `[SYSTEM CONTEXT — do not echo]` block telling the model the reply will render on a tiny HUD with limited viewport. Sent to the agent only; never broadcast to the channel bus, so it never appears in the mobile log or the glasses view.
- **Image attachments** — phone photo → bridge → agent's `/api/image/upload` → path forwarded in the WS `chat` message. Reuses the same `image_path` contract the captain-claw web UI already uses.

### 2. Soniox TTS streaming — `WS /glasses/tts-stream`

Real-time Soniox TTS over WebSocket. First-audio latency ~150–300 ms; the rest of the clip plays while later chunks are still being synthesized.

- Browser receives `{type:"info", format, sample_rate, channels}` first, then a stream of **binary** PCM frames decoded server-side from Soniox's base64 (halves wire payload vs. forwarding base64).
- Glasses-view scheduler — lazy `AudioContext`, per-chunk `AudioBufferSource.start(nextStart)` with a 10 ms safety margin so chunks don't clip.
- Auto-fallback to the one-shot `/glasses/tts` if the streaming WS opens but closes with zero chunks (transient upstream blip).
- **60+ languages**, **28 multilingual voices** (Neutral / British / American-Spanish / Australian / Indian accent groups). Picked via the settings page; persisted in `localStorage`.
- Stream / one-shot toggle in the glasses header (📡 / 📦), persisted across reloads.

### 3. Voice off by default + autoplay gesture handling

- Glasses view starts with TTS **muted**. The user explicitly taps 🔊 to enable.
- A first-load banner reading `🔇 Voice is off — tap 🔊 to enable` explains it; auto-dismisses when audio is enabled.
- Hard short-circuit before the queue — when muted, **zero** `/glasses/tts*` calls are made. Agent replies still render visually, just no audio traffic.
- The mute toggle counts as a user gesture, which satisfies browser autoplay policy so the first audio plays without `NotAllowedError`.

### 4. Glasses settings page — `/glasses/settings`

Tap-target button grid for language (60+) and voice (28, grouped by accent under section headers). Replaces native `<select>` dropdowns which are essentially unusable with Neural Band tap gestures. Auto-saves to `localStorage`; flashes a "Saved" pill on tap. Bridges back to the glasses view via a "← Glasses" pill that preserves the channel id and any auth token.

### 5. Mobile bridge — PWA install + photo attach

- **PWA installable** on iOS and Android — manifest, service worker, PNG icons (96 / 192 / 512 / 180 apple-touch). On iOS: Share → Add to Home Screen; on Android: install prompt or three-dot menu → Install app. Standalone mode, dark theme, scoped to `/glasses/`.
- **Photo attach** — paperclip button opens camera or gallery (`capture="environment"` on the input) → preview thumbnail with remove button → Send. Bridge proxies multipart to the agent's `/api/image/upload`; agent receives the image as the familiar `[Attached image: …]` prefix in the prompt.
- **Editable channel** — channels are URL params, freely editable (Enter / blur to apply, ⟳ for a fresh one). The URL stays in sync via `history.replaceState`, so reloads and shared links preserve the channel.
- **Activity log** collapses; **status pill** animates green (live) / red (offline).

### 6. Glasses view — visual design

- **Floating HUD bar** at the top (blurred gradient that fades into content) with a Captain Claw brand pill (purple gradient + favicon), the SSR freshness token, an animated `live` pill, and a row of TTS controls (🔊 / 📡 / ⚙ / engine label).
- **Message bubbles** redesigned — no harsh left borders; bigger rounded boxes with role-coloured backgrounds, a small role badge with a coloured dot, and a glowing border ring on the newest message.
- **Markdown rendering** via `marked` with GFM enabled — tables, fenced code, lists, blockquotes, autolinks. Tables full-width with rounded corners and an uppercase header row.
- **Thinking state** — when the agent emits a `busy` status, the footer status bar grows dramatically: 22 px text, amber gradient background, animated pulsing dot, double-glow shadow. Returns to compact the moment the answer arrives (a synthetic `ready` status is injected as soon as a `chat_message` lands so the bar doesn't lag the visible reply).
- **Captain Claw icons** (PNG, 96 / 180 / 192 / 512) wired into the head via `<link rel="icon">` + `apple-touch-icon` + a glasses-specific manifest at `/glasses/view-manifest.webmanifest` so Meta's wearables launcher displays the brand icon.

### 7. Tavily web-search provider

`web_search` now supports **Tavily** as an alternative to Brave. Huge thanks to the Tavily team for the excellent API — it's drop-in simple to integrate and the results are great.

```yaml
tools:
  web_search:
    provider: tavily        # or "brave" (default)
    tavily_api_key: ""      # or set TAVILY_API_KEY in env / .env
    max_results: 5
```

The provider switch is transparent to the agent — same tool, same `query` parameter. Tavily ignores some Brave-specific knobs (`offset`, `country`, `freshness`, `safesearch`); a debug log warns when those are passed, but they're otherwise harmless.

## How to use the Glasses Bridge

### Prerequisites

- Flight Deck running locally: `captain-claw-fd` (binds to `http://0.0.0.0:25080`).
- A public HTTPS URL pointing at Flight Deck — Meta Web Apps require HTTPS. Easiest options:
  - `cloudflared tunnel --url http://localhost:25080`
  - `ngrok http 25080`
- At least one captain-claw **process** agent spawned from Flight Deck → Agents (the bridge lists process agents, not Docker containers).
- For TTS (optional): a Soniox API key. Set `SONIOX_API_KEY` in Flight Deck's environment before launching. Without it the bridge still works visually — agent replies just won't speak.

```bash
export SONIOX_API_KEY=sk_…        # optional
# optional overrides
export SONIOX_TTS_VOICE=Adrian
export SONIOX_TTS_LANGUAGE=en
captain-claw-fd
```

### Step 1 — Open the mobile bridge on your phone

Visit `https://<your-tunnel>/glasses/mobile` on your phone. The page generates a random channel id (e.g. `?c=abc12345`) and bounces to it. The mobile bridge is installable:

- **iOS Safari**: Share → Add to Home Screen.
- **Android Chrome**: install prompt, or three-dot menu → Install app.

Once installed it launches in standalone mode (no browser chrome) with the Captain Claw icon.

Pick your agent from the dropdown, type a message, and (optionally) tap 📷 to attach a photo.

### Step 2 — Open the glasses view on the glasses

In the Meta AI mobile app's Web Apps section, add a Web App at:

```
https://<your-tunnel>/glasses/view?c=abc12345
```

— using the same channel id you saw in the mobile bridge URL. Launch it from the Meta Ray-Ban Display app launcher.

On the glasses:
- Tap the **⚙** icon in the top-right corner to open `/glasses/settings` and pick voice + language from the button grids (don't use dropdowns — Neural Band tap doesn't navigate them well).
- Tap the **🔊** icon to enable TTS (it starts off — your tap is what satisfies the browser's autoplay gesture requirement, so the first reply will speak without issue).
- Tap **📡** to switch between streaming and one-shot TTS (streaming is the default and is noticeably lower latency on long replies).

### Step 3 — Talk to the agent

Type on the phone → Send. The message appears on the glasses (newest message highlighted with a coloured ring), the status bar at the bottom pulses amber while the agent is thinking, then the reply renders in full markdown and — if TTS is on — speaks through whatever audio output the phone is routing to (which, if the glasses are paired as a Bluetooth audio sink, will be the glasses' open-ear speakers).

### Optional: attach a photo

In the mobile bridge, tap 📷 → take a photo or pick from gallery → Send. The bridge uploads to the chosen agent, then includes the image path in the chat. The agent receives it as a normal attachment and can analyze it (the glasses show 📷 You in the role badge to confirm a photo was attached).

### Switching agents / channels

- **Switch agents** on the mobile bridge: pick a different one in the dropdown — the bridge cancels the previous outbound WS, rebinds, and re-sends the hidden glasses context.
- **Switch channels**: edit the channel id directly (Enter or tap-out to apply), or tap ⟳ for a fresh random one. Glasses-view URL updates accordingly — copy the new `/glasses/view?c=…` link to the glasses if you want to follow.

## Files

New:
- `captain_claw/flight_deck/glasses_bridge.py` — router, channel bus, TTS proxies, PWA manifest, glasses-view manifest, service worker.
- `captain_claw/flight_deck/static/glasses_mobile.html` — mobile bridge UI (PWA-installable).
- `captain_claw/flight_deck/static/glasses_view.html` — glasses HUD.
- `captain_claw/flight_deck/static/glasses_settings.html` — voice/language picker.
- `captain_claw/flight_deck/static/icon-96.png` / `icon-192.png` / `icon-512.png` / `apple-touch-icon.png` — Captain Claw launcher icons (PNG; required by Meta).
- `meta-glasses-test/server.py` — standalone freshness probe (no deps, pure stdlib) for verifying fresh-from-server behaviour on the glasses webview.

Modified:
- `captain_claw/flight_deck/server.py` — `include_router(glasses_router)`.
- `captain_claw/tools/web_search.py` — Tavily provider added.
- `captain_claw/config.py` — `WebSearchToolConfig.tavily_api_key`, `TAVILY_API_KEY` env-overlay.

## Backward compatibility

Everything is additive. Existing 0.4.26 setups keep working unchanged. The Glasses Bridge is dormant unless you visit `/glasses/*`. Tavily is opt-in (default provider remains Brave) — no migration needed.
