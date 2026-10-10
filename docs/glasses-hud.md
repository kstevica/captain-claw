# Captain Claw HUD: Flight Deck on smart glasses

The HUD is Flight Deck's frontend for smart glasses. It lives at **`/hud/`**
and is built for the Meta Ray-Ban Display, a 600 × 600 display you drive with
swipes and pinches. From the glasses you can:

- **Chat** with your running agents, in the same conversation you see in the
  dashboard. You can dictate or handwrite with Meta's composer and have replies
  read aloud.
- **Read markdown files** your agents wrote: headings, lists, code and tables,
  sized for the display.
- **Browse datastore tables**: the table list, then rows, then one record at a
  time. Everything is read-only.

The HUD is a separate, lean page (`flight-deck/hud.html` → `src/hud/main.tsx`),
so the glasses never download the dashboard bundle. Meta's budget for a first
load is under 300 KB.

Platform facts, sources and confidence levels are in
[glasses-webview-capabilities.md](glasses-webview-capabilities.md).

## Requirements

| | |
|---|---|
| Glasses | Meta Ray-Ban Display. Web apps need firmware **v125+**, text input (the composer) needs **v127+**, and the on-screen keyboard needs **v129+**. |
| Phone | Meta AI app **v272+** |
| Neural Band | Optional; the temple touchpad also works. You need the band for handwriting. |
| Flight Deck | Reachable at a **public HTTPS URL**, with accounts on. See the [deployment checklist](#deployment-checklist). |
| Build | Run `npm run build` in `flight-deck/` to produce `static/hud.html`. Until you do, `/hud/` answers 503 "HUD not built". |

## Install on Meta Ray-Ban Display

1. **Turn on Developer Mode** in the Meta AI app: Settings → App Info → tap the
   **version number 5 times**. The setting stays on.
2. **Add the web app.** In the Meta AI app, go to **Devices** → **Display glasses
   settings** → **App connections** → **Web apps** → **Add a web app**:
   - Name: `Captain Claw`
   - URL: `https://<your-host>/hud/`
3. **Open it on the glasses.** Captain Claw appears at the bottom of the app grid,
   and you can pin it there.

Menu labels differ a little between Meta AI app versions. Meta's own docs show
both "Apps → Web Apps → Connect Web App" and "App connections → Add a Web App".

**QR code instead.** Meta's publish tooling (the toolkit's publish script)
registers web apps with a QR code that encodes this deep link:

```
fb-viewapp://web_app_deep_link?appName=Captain%20Claw&appUrl=https%3A%2F%2F<your-host>%2Fhud%2F
```

URL-encode both values, turn the link into a QR code, and scan it with the
phone that runs the Meta AI app. The format comes from Meta's toolkit repo,
not its public docs. The QR code is also the workaround when **Add a web app**
is missing for your account
([known issue](https://github.com/facebook/meta-wearables-webapp/issues/9)).

**Updates.** Redeploying Flight Deck is enough. The HUD page is fetched
network-first and its hashed assets are cached by a service worker, so the next
online launch picks up the new build. If a launch ever shows an old build, open
the universal menu (middle tap) and choose **Restart**.

## Sign in

You never type your password on the glasses. They sign in with a **pairing
code** that you approve on a device where you're already signed in.

### Pairing code (recommended)

1. **Open Captain Claw on the glasses.** It shows a code such as **`BCDF-GHJK`**,
   the address to open, and a 10-minute countdown.
2. **On your phone or computer, open `https://<your-host>/hud/pair`.** Sign in to
   Flight Deck if asked, with the account the glasses should use.
3. **Type the code.** The page shows **which device is asking**: its label (for
   example "Meta Ray-Ban Display"), browser, IP address and age. Choose
   **Approve** or **Deny**.
4. **Wait a few seconds.** The glasses pick up the approval and open your agent
   list.

Only approve a code that is on **your** glasses right now.

How the codes work:
- A code is 8 letters long, with no vowels and no look-alike characters.
- Each code works once and expires after 10 minutes.
- The glasses check for approval every 3 s, and pause while the display sleeps.
- If a code expires or is denied, pinch **New code**.
- Rate limits: 10 new codes per 10 minutes per IP address, and 30 lookups or
  approvals per minute per account.

### Email and password (fallback)

On the sign-in screen, pinch **Sign in with email instead**. Then pinch each
field to open the composer, and dictate, handwrite or type (the keyboard needs
firmware v129+).

Limits:
- **Some characters can't be typed.** The glasses keyboard has no key for
  `' " _ \ < > [ ] { } | ~ ^` or the backtick. You can't enter a password that
  contains any of them.
- **Dictating a password says it out loud.** Meta's composer never opens on
  `type="password"` fields, so on the glasses the password is an ordinary text
  field, masked on screen. Use the keyboard or handwriting for it.
- **No composer, no typing.** Without the composer (firmware below v127, or a
  build where it is unavailable) you can't type anything. Use pairing.

### Staying signed in

- **What a session is.** Signing in creates a normal Flight Deck session:
  - a 15-minute access token, kept in memory;
  - the httpOnly `fd_refresh` cookie (path `/fd/auth`). It lasts **7 days and
    slides**: every refresh starts a new 7 days.

  While the HUD is open it refreshes every 10 minutes, and again when the
  display wakes up.
- **Open it at least once a week.** The glasses then stay signed in, **as long
  as** their WebView keeps cookies between launches. Meta doesn't document
  whether it does, so check on your device (see the
  [on-device test checklist](#on-device-test-checklist)).
- **Sign out.** In the agent list, pinch **Sign out**, then pinch **Confirm sign
  out** within 5 seconds. To sign back in, you need your phone or computer
  again.

## Using the HUD

### Controls

| Gesture | Neural Band | Does |
|---|---|---|
| Swipe | Thumb along the index finger | Moves focus up, down, left or right. On long text it scrolls a page. |
| Pinch | Index finger to thumb | Selects the focused item |
| Back | Middle finger to thumb | Goes back one screen. On the agent list it opens the glasses' system menu. |

The temple touchpad works as well: swipe on it to move.

- **No Back buttons.** There are no on-screen Back buttons. Back always puts
  focus on the item you came from.
- **What takes focus.** The focused item has a bright ring. Only things you can
  act on take focus, plus blocks of text for reading.
- **Long text reads in place.** For a long reply, file section or field value,
  each swipe down scrolls one page. At the end, the next swipe moves on.

### Agents

- **The agent list.** The home screen lists your running agents: your own
  process and Docker agents, plus agents shared with you.
- **Stopped agents** appear dimmed and can't be opened; start them from Flight
  Deck. **Refresh** reloads the list.
- **Opening an agent.** Pinch an agent to open its **Chat**. The agent you used
  last is marked **last**, and launching the app reopens its chat directly; Back
  then goes to the list.
- **Agent screen header.** It shows:
  - **Chat / Files / Data** tabs. On a shared agent, Files and Data appear only
    if its owner shares them.
  - a status dot: live, thinking or offline;
  - the clock.
- **Tabs don't add Back steps.** Back from any tab returns to the agent list.

### Chat

- **One conversation.** This is the agent's main conversation, the same one you
  see in the dashboard. The connection stays open while you look at Files or
  Data, and a badge on the **Chat** tab counts replies you haven't seen yet.
- **To send a message:**
  1. Focus the field at the bottom ("Pinch to write or speak…") and **pinch**
     it. Meta's composer opens.
  2. Dictate, handwrite, or swipe down for the keyboard (v129+), then confirm.
  3. The text lands in the field and focus moves to **Send**. Pinch it.
- **Reading.** Each message is a reading block. Swipe up through the
  conversation; a long reply scrolls a page per swipe.
- **Below the conversation:**
  - **Next-step chips**: replies the agent suggests. Pinch one to send it; these
    work without the composer.
  - **File chips**: markdown files a reply mentions. Pinch one to open it.
  - **Approve / Deny**, when the agent asks for approval.
  - **Stop** (while the agent is working), **New chat** (pinch twice to
    confirm), **Read aloud: On/Off**, and **Reconnect** if the connection
    drops.
- **Read aloud** speaks new replies through the glasses speaker. There is one
  English voice, and the app can't change its volume. The glasses remember
  this setting.
- **Message limit.** The HUD keeps the latest 40 messages on screen.

### Files

- **The file list.** The **Files** tab lists the agent's markdown files (`.md`,
  `.markdown`), newest first, with age and size. Files that someone other than
  the agent's owner wrote are marked with the author's name.
- **Reading a file.** Pinch a file to read it:
  - Swipe up or down to walk through headings, paragraphs, list items, code and
    tables.
  - Swipe **left or right** to turn a whole page.
- **Tables.** A table with up to 3 columns stays a table. Wider tables become
  **one card per row** (`column: value` lines), so nothing scrolls sideways.
- **Images and links.** Images show as their alt text and are never downloaded.
  Links are shown, but you can't follow them. Embedded HTML is dropped.
- **Size limits.** The glasses don't open files over 1 MB. For very long files
  they show the first 200 KB.

### Data

- **Tables.** The **Data** tab lists the agent's datastore tables, with row and
  column counts and when each table changed. Everything is read-only.
- **Rows.** Pinch a table to see its rows, **8 per page, newest first**. Swipe
  **left or right** (or pinch the **‹ Prev / Next ›** chips) to change page.
- **Records.** Pinch a row to open the record, one field per block. Swipe
  **left or right** to step to the previous or next record. Back returns to the
  rows page that holds the record you were on.

### Voice control (WebMCP)

If Meta enables WebMCP on your glasses, Meta AI can drive the HUD by voice
through three tools. WebMCP is off by default; Meta turns it on per device,
through Developer Mode or its rollout.

| Tool | Does |
|---|---|
| `claw_get_state` | Read-only. Reports the screen, the open agent, whether it is working, its latest reply as plain text, any approval waiting, and the suggested replies. |
| `claw_send_message` | Sends a message (up to 4,000 characters) to the agent open on the glasses. The reply appears on the display. |
| `claw_open_screen` | Opens `chat`, `files` or `data` for the current agent, or `agents` for the agent list. |

Approvals can only be answered on the display. Everything works with swipes and
pinches when WebMCP is off.

## Opening /hud/ automatically

- **The redirect.** When the Display's browser opens the dashboard root
  (`https://<your-host>/`), Flight Deck redirects it to `/hud/`. It recognises
  the Display's WebView by its User-Agent (`Greatwhite` plus `; wv)`), the same
  check Meta's toolkit uses.
- **`/?ui=full`** opens the full dashboard instead and remembers that with a
  cookie (`fd_ui=full`, kept for a year), so those glasses stop being redirected.
- **`/?ui=hud`** clears that cookie and goes to `/hud/`.
- **Register `/hud/` anyway.** Registering `/hud/` directly (above) is still the
  reliable way in. Meta doesn't document the User-Agent, and Android's planned
  User-Agent reduction may remove the `Greatwhite` token.

## Other glasses

The HUD needs a "Display web app" host. That means two things:

- a browser engine **on the glasses** loads a URL;
- the host delivers swipes as arrow keys and pinches as Enter.

Today only the Meta Display does this, plus community hosts that copy it (Rokid
Lumen).

| Glasses | Runs the HUD? | Notes |
|---|---|---|
| **Meta Ray-Ban Display** | Yes, the main target | This guide. |
| **Rokid Glasses + Rokid Lumen** | Should work; we haven't tested it | Lumen is a community host that runs Display web apps unchanged: a 600 × 600 CSS viewport, Neural Band input as arrow keys and Enter, and its own composer for text fields. Register the same `https://<your-host>/hud/` URL in Lumen. Use Lumen's default GeckoView engine; its system-WebView option is Chromium 95, older than the browsers the HUD is built for. ([rokid-lumen](https://github.com/beyondlevi/rokid-lumen), community) |
| **Even Realities G2** | No | G2 apps are web pages that run in the Even app **on the phone**. They send text and image containers over Bluetooth to a 576 × 288 green display, so no HTML runs on the glasses. ([Even Hub SDK](https://www.npmjs.com/package/@evenrealities/even_hub_sdk)) |
| **Even Realities G1** | No | No web runtime; the glasses speak a Bluetooth protocol only. |
| **Brilliant Labs Halo / Frame** | No | No browser. Lua runs on the glasses, driven by a phone or desktop SDK (Python, Flutter, Web Bluetooth). ([halo-firmware](https://github.com/brilliantlabsAR/halo-firmware)) |
| **Vuzix Z100** | No | No browser. A native phone app pushes content with the Ultralite SDK. ([UltraliteSDK](https://github.com/Vuzix/UltraliteSDK-releases-iOS)) |
| **Android XR display glasses** | No | No documented web runtime. Apps are native (Jetpack Compose Glimmer), projected from the phone. Chrome and WebXR on Android XR cover headsets and wired XR glasses only. ([Android XR devices](https://developer.android.com/develop/xr/devices), [Android XR for web](https://developer.android.com/develop/xr/web)) |
| **XREAL (One, One Pro, Aura, Beam Pro), INMO Air 3, RayNeo X3 Pro** | Use the normal dashboard | These run full Android, or act as a tethered monitor for a phone or PC. They have a large viewport and normal pointer and keyboard input, so open the usual Flight Deck URL in their browser. |

## Deployment checklist

Meta's glasses only load web apps from a **public HTTPS URL**: no HTTP, no LAN
address. Usually that means a tunnel or reverse proxy in front of Flight Deck.
Check these before you register the URL:

| Setting | Why |
|---|---|
| A public HTTPS URL for Flight Deck | Meta requires HTTPS and a URL anyone can reach. Don't put HTTP auth or an access gate in front of `/hud/`: the glasses can't get through it, and the HUD handles sign-in itself. |
| `FD_AUTH_ENABLED=true` | With accounts off, anyone with the URL acts as the local admin. The HUD then skips sign-in, shows a "Sign-in off" warning, and pairing is unavailable. |
| `FD_JWT_SECRET` set | Without it, Flight Deck generates a random secret at every start, so all access tokens die on restart. (The HUD recovers through the refresh cookie.) |
| `FD_PUBLIC_URL=https://<your-host>` or `FD_ALLOWED_HOSTS=<your-host>` | The origin guard refuses requests whose host or origin it doesn't know. Without this, every HUD request through the tunnel fails. |
| `FD_COOKIE_SECURE=1` (or `FD_LOCKDOWN=1`) | Marks the refresh cookie and `fd_ui` as Secure behind HTTPS. |
| `FD_GLASSES_BRIDGE_TOKEN` set | The legacy `/glasses/*` bridge has no user accounts. On a public URL, anyone can use it unless this shared secret is set. The HUD doesn't use the bridge. |
| `npm run build` in `flight-deck/` | Produces `static/hud.html`. Until then, `/hud/` returns 503. |
| Client IPs through the proxy (optional) | The approval page shows the requesting device's IP address, and pairing is rate-limited per IP. Behind a reverse proxy, uvicorn has to trust the proxy's `X-Forwarded-For` (`FORWARDED_ALLOW_IPS`). Otherwise every device shows up, and is rate-limited, as the proxy. |

## For developers

### Where the code is

| Path | What |
|---|---|
| `flight-deck/hud.html` | The entry page: `mrbd-web-app-capable`, description, PNG icons, manifest link, first-paint style |
| `flight-deck/src/hud/main.tsx` | Mounts `HudApp` and registers the service worker. No `StrictMode`: its dev double effects would open two chat sockets and two pairings. |
| `HudApp.tsx` | Startup (auth status → refresh → pairing or the app), token keep-alive, routes to screens, attaches chat, registers WebMCP. `/hud/pair` renders the approval page. |
| `router.ts` | History router. Routes live in the query string, e.g. `/hud/?v=agent&a=…&t=chat`. |
| `focus.ts`, `hooks.ts` | D-pad focus engine; `useActivate`, `useAutoFocus` |
| `ui.tsx`, `hud.css` | `ScreenFrame`, `Row`, `Btn`, `Pill`, `StateView`; design tokens |
| `api.ts`, `agentsStore.ts` | Authenticated data layer; the agent list |
| `agents/`, `chat/`, `files/`, `data/`, `auth/` | The screens: agent list; chat, composer and chat store; file list and reader; tables → rows → record; pairing, email login and `/hud/pair` |
| `markdown/` | `react-markdown` + `remark-gfm`, plus a small rehype pass that drops raw HTML, turns wide tables into cards and makes blocks focusable. Large files are split and rendered progressively. |
| `webmcp.ts` | The three WebMCP tools |
| `device.ts` | Host detection (`meta-display`, `rokid-lumen`) and the device label sent with pairing |
| `captain_claw/flight_deck/hud_routes.py` | `/hud`, `/hud/{rest}`, the manifest, the service worker, `/fd/hud/config` and the `/` redirect |
| `captain_claw/flight_deck/auth_routes.py` | The `/fd/auth/pair/*` endpoints |

To develop locally, run Flight Deck on port 25080 and `npm run dev` in
`flight-deck/`, then open `http://localhost:5173/hud/`. Vite rewrites `/hud*`
to `hud.html` and proxies `/fd`, the manifest and the service worker.

### Rules the screens follow

**Focus engine (`focus.ts`)**
- **It owns the arrow keys.** It calls `preventDefault()` and moves focus
  geometrically inside the one mounted `.hud-screen`, as Meta's toolkit does.
  - A component that handles a key itself (Left/Right paging) calls
    `preventDefault()` first. The engine skips events that are already handled.
- **Tall elements are read before focus leaves.** A focused element taller than
  its scroller scrolls page by page first. The first and last stops pin the
  scroller to its ends.
- **Keys.** It ignores `key === 'Unidentified'` and uses `.key`, never `.code`.
- **One pinch, one action.**
  - `useActivate` dedupes Enter, Space and click on the same element within
    500 ms.
  - `quietActivations()` mutes the late pinch after the composer closes.
  - `pinFocus()` undoes the host's focus reset after Back or after the composer
    closes.
- **Escape / Backspace go back** on desktop and in the Simulator only, never on
  the Display, whose shell already calls `history.back()`.
- **What can take focus.**
  - Interactive items are `<div role="button" tabIndex=0>` (`Btn`, `Row`,
    `useActivate`).
  - Long read-only text is a focusable `.hud-block`.
  - Rows without an action are not focus stops.
  - Native `<button>`s appear only on `/hud/pair`.
- **Scrolling.** Each screen has one vertical scroll owner (`.hud-scroll`) and
  no horizontal overflow.

**History (`router.ts`)**
- **Push and replace.** The first entry is seeded with `replaceState`. Each
  drill-down pushes one entry; tab switches and pagers replace.
- **Depth.** The deepest chain is 4 entries (agents → agent → rows → record),
  under Meta's 5-entry cap. A deep link on a fresh launch rebuilds its ancestor
  chain, so Back walks up instead of exiting.
- **Focus restore.** Each pushed entry remembers the focused element's
  `data-fk`, and Back restores focus to it. Give list rows a stable `fk`.
- **Reloads.** The route lives in the URL, so a reload (universal menu →
  Restart) returns to the same screen.

**Data access**
- **Guarded routes only.** The HUD calls only Flight Deck's JWT-guarded `/fd/*`
  routes and `/fd/agent-ws`. These check that you own the agent or that it is
  shared with you.
  - Never the legacy `/glasses/*` bridge.
  - Agent secrets are resolved on the server and never appear in the page.
- **Text only.** Everything renders as React text: no
  `dangerouslySetInnerHTML`, no raw HTML in markdown, no remote images.

### API routes used

| Route | Auth | Used for |
|---|---|---|
| `GET /fd/auth/status`, `POST /fd/auth/refresh`, `POST /fd/auth/login`, `POST /fd/auth/logout` | public / refresh cookie | Startup, the session, the email fallback, sign-out |
| `POST /fd/auth/pair/start`, `POST /fd/auth/pair/poll` | public, rate-limited | Glasses: get a code and wait for approval (the approved poll returns the session) |
| `GET /fd/auth/pair/lookup?code=`, `POST /fd/auth/pair/approve` | signed in | `/hud/pair`: show the requesting device; approve or deny |
| `GET /fd/processes`, `/fd/containers`, `/fd/shared-agents` | Bearer | The agent list (filtered by owner) |
| `GET /fd/agent-files/localhost/{port}`, `/fd/agent-file-view/localhost/{port}?path=` | Bearer, owner-checked | Your own agents' files |
| `GET /fd/shared-agents/files?ref=`, `/fd/shared-agents/files/view?ref=&id=` | Bearer, member | Shared agents' files |
| `GET /fd/agent-datastore/localhost/{port}/tables`, `…/tables/{table}/rows` | Bearer, owner-checked | Your own agents' datastore |
| `GET /fd/shared-agents/datastore/tables?ref=`, `…/tables/{table}/rows?ref=` | Bearer, member | Shared agents' datastore |
| `WS /fd/agent-ws/localhost/{port}?fd_token=`, `WS /fd/agent-ws-shared?ref=` | access token | Chat (the agent's main lane) |
| `GET /fd/hud/config` | Bearer | The glasses rendering rules, sent with the first chat message of each connection |
| `GET /hud/manifest.webmanifest`, `GET /hud/sw.js` | public | The launcher manifest (PNG icons); the service worker (cache-first for `/assets/*`, network-first for `/hud*`, never touches `/fd/*`) |

### Testing on a desktop

- **A 600 × 600 surface.** Install the **Meta Ray-Ban Display Simulator** Chrome
  extension (Chrome Web Store id `jpjlmmodokemlepklkdbimceggpbjcll`), which
  gives a 600 × 600 surface with D-pad and Select. Or use a 600 × 600 window
  (the DevTools device toolbar, DPR 1).
- **Keys.** Arrow keys are swipes, Enter is a pinch, and Escape is Back.
  Backspace is also Back, except inside a text field.
- **The desktop is not the glasses.** Meta: desktop arrow and Enter testing "is
  only an approximation". Test on the device before calling a change done.
  - There is no composer: you type normally, and Enter in a field sends.
  - Escape works.
  - The device's input timing quirks don't happen: double pinches, focus
    resets, late pinches.
- **Size and overflow.** Check that nothing overflows sideways, and that the
  first load stays under 300 KB (DevTools Network with "Disable cache" and
  throttling).

### On-device test checklist

- [ ] On a fresh install, the first launch shows the pairing code, and approving it on `/hud/pair` opens the agent list.
- [ ] After closing and relaunching the app, you're still signed in (cookie persistence). Check again after the display has slept for a few minutes.
- [ ] Swipes move focus, and one pinch triggers exactly one action: no double send, no double navigation.
- [ ] Back from a file or a record lands on the row you opened, with focus on it. Back on the agent list opens the system menu.
- [ ] Chat: pinching the field opens the composer, dictated text arrives, focus moves to Send, and the closing pinch presses nothing else.
- [ ] Keyboard (v129+): swiping down in the composer raises it, and the symbol limits match this guide.
- [ ] A long reply reads page by page as you swipe, and the first and last stops reach the ends.
- [ ] A wide markdown table renders as cards, and nothing scrolls sideways.
- [ ] Rows: Left/Right change page. Record: Left/Right step between records.
- [ ] Read aloud speaks a new reply.
- [ ] A warm relaunch is fast (the service worker cache works).
- [ ] With the network off, screens show error states with Retry, and chat reconnects when the network returns.
- [ ] Sign out needs a second pinch.
- [ ] With WebMCP on, asking Meta AI what Captain Claw is showing calls `claw_get_state`.

## Known limitations

- **No microphone, camera or speech recognition.** The Display gives web apps
  none of these, so there are no voice notes, no photos and no push-to-talk.
  Voice input comes only from the composer's dictation, or from WebMCP.
- **Composer limits.**
  - It opens only on a pinch, never programmatically.
  - It can't type `' " _ \ < > [ ] { } | ~ ^` or the backtick.
  - It never opens on password fields.
  - It may be unavailable on some builds; next-step chips and WebMCP still work
    then.
- **Cookies between launches.** Meta doesn't document whether cookies persist
  between launches, so verify it on your glasses. If the WebView drops them,
  you'll have to pair again at every launch.
- **Use it weekly.** The session slides over 7 days. After a week without use,
  pair again.
- **Read-only.** You can read files and data but can't edit, upload or run SQL.
  Only markdown files are listed.
- **Stopped agents** can't be started from the glasses.
- **The `/` → `/hud/` redirect** depends on a User-Agent token that Meta doesn't
  document and that Android's User-Agent reduction may remove. Register `/hud/`
  directly.
- **WebMCP is off by default.** Until Meta enables it on your device, there is
  no voice control through Meta AI.
- **The legacy `/glasses/view` page remains** for the phone-bridge workflow,
  where a phone is the input and you share channel links. It is separate from
  the HUD and has no user accounts. On a public URL, protect it with
  `FD_GLASSES_BRIDGE_TOKEN`.

## Sources

The full source list, with confidence labels, is in
[glasses-webview-capabilities.md](glasses-webview-capabilities.md#sources-october-2026-update).
The main ones:

- **Meta's web-app docs** (search extracts):
  [Build](https://wearables.developer.meta.com/docs/develop/webapps/build/),
  [Setup](https://wearables.developer.meta.com/docs/develop/webapps/setup/),
  [Test](https://wearables.developer.meta.com/docs/develop/webapps/test/),
  [Agent tools](https://wearables.developer.meta.com/docs/develop/webapps/agent-tools/),
  [FAQ](https://developers.meta.com/wearables/faq/).
- **Meta's repos**, read directly:
  - [facebook/meta-wearables-webapp](https://github.com/facebook/meta-wearables-webapp):
    `AGENTS.md`, covering the publish flow, the QR deep link and the device
    envelope; plus the WebMCP skill.
  - [facebook/meta-ray-ban-display-ui-toolkit-web](https://github.com/facebook/meta-ray-ban-display-ui-toolkit-web).
- **Community:**
  - [beyondlevi/rokid-lumen](https://github.com/beyondlevi/rokid-lumen), for Rokid Lumen.
  - [handzlikchris/Glasscast](https://github.com/handzlikchris/Glasscast), for on-device input quirks.
  - [nickustinov/even-g2-notes](https://github.com/nickustinov/even-g2-notes), for Even G2.
