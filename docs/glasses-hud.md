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
load is under 300 KB transferred. The HUD's first load is about 490 KB raw but
about 150 KB on the wire, in 9 requests (the page plus 8 script and style
files): Flight Deck gzips the HUD page and the built `/assets/*` files itself,
and browsers keep the content-hashed assets for good. Later launches come from
the service worker's cache.

Platform facts, sources and confidence levels are in
[glasses-webview-capabilities.md](glasses-webview-capabilities.md).

## Requirements

| | |
|---|---|
| Glasses | Meta Ray-Ban Display. Web apps need firmware **v125+**, text input (the composer) needs **v127+**, and the on-screen keyboard needs **v129+**. |
| Phone | Meta AI app **v272+** |
| Neural Band | Optional; the temple touchpad also works. You need the band for handwriting. |
| Flight Deck | Reachable at a **public HTTPS URL**, with accounts on. See the [deployment checklist](#deployment-checklist). |
| Build | None needed to deploy: the built HUD (`static/hud.html` plus its files in `static/assets/`) ships with Flight Deck. If `static/hud.html` is missing, `/hud/` answers 503 "HUD not built". |

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

**When Flight Deck is down.** If the page doesn't arrive within 4 s, the network
fails, or the answer is an error page from your tunnel or proxy (a 502 or
Cloudflare's 530, for example), the service worker starts the last version it
has. That version keeps retrying Flight Deck and recovers by itself once it
answers. On the very first launch there is nothing cached yet, so it waits for
the network.

## Sign in

You never type your password on the glasses. They sign in with a **pairing
code** that you approve on a device where you're already signed in.

### Pairing code (recommended)

1. **Open Captain Claw on the glasses.** It shows a code such as **`BCDF-GHJK`**,
   the address to open, and a 10-minute countdown.
2. **On your phone or computer, open `https://<your-host>/hud/pair`.** Sign in to
   Flight Deck if asked, with the account the glasses should use.
3. **Type the code.** Always type it yourself: the page never takes a code from
   its link, and the glasses only ever show the bare address. The page then
   shows **which device is asking**:
   - the **IP address** Flight Deck saw, and how long ago the code was made;
   - under **"Reported by the device (not verified)"**, the name it gave (for
     example "Meta Ray-Ban Display") and its browser. Any device can claim
     these.

   Choose **Approve** or **Deny**.
4. **Wait a few seconds.** The glasses pick up the approval and open your agent
   list.

Only approve a code that is on **your** glasses right now. If someone sends you
a code or a link with one, it is not your glasses asking.

**A full sign-in is needed to approve.** Approving signs another device in, so
an access token alone isn't enough: Flight Deck also checks the browser's own
`fd_refresh` cookie for a live session of the same account. If the page says
"Approving a device needs a full sign-in in this browser", use **Sign out** on
that page, sign in again and retry.

How the codes work:
- A code is 8 letters long, with no vowels and no look-alike characters.
- Each code works once and expires after 10 minutes.
- The glasses check for approval every 3 s, and pause while the display sleeps.
- **Restart keeps the code.** Restart from the universal menu shows the same
  code again while it has at least 10 s left, so the code you are typing on the
  phone stays valid. Closing and relaunching the app may start a new one.
- If a code expires or is denied, pinch **New code**.
- **No answer from Flight Deck.** The glasses say "Can't reach Flight Deck —
  trying again…" and keep trying; **Try now** skips the wait. **Sign in with
  email instead** stays available, also while a code is loading.
- **Rate limits** are per client network (an IPv4 address, or an IPv6 /64):
  - 10 new codes per 10 minutes. When that runs out, the glasses say so and
    give the time by which a new code will certainly work.
  - At most 3 pending codes at a time. A fourth code replaces that network's
    oldest pending one.
  - Lookups and approvals: 30 per minute per account, each.

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

  While the HUD is open it checks the token once a minute and when the display
  wakes up, and refreshes it only when less than 3 minutes are left.
- **Open it at least once a week.** The glasses then stay signed in, **as long
  as** their WebView keeps cookies between launches. Meta doesn't document
  whether it does, so check on your device (see the
  [on-device test checklist](#on-device-test-checklist)).
- **A bad connection doesn't sign you out.** If a refresh fails because the
  network or Flight Deck is down, the session is kept and the screen shows a
  network error with **Retry**. The session ends only when Flight Deck itself
  refuses the refresh cookie.
- **When the session ends mid-use**, the sign-in screen appears and the HUD
  steps back to its first history entry, so Back on the sign-in screen opens
  the system menu. Signing in again resumes the last agent. History from before
  a **Restart** is left alone (going back to it reloads the page), so there each
  Back is a visible reload of the sign-in screen until the system menu opens.
- **Sign out.** In the agent list, pinch **Sign out**, then pinch **Confirm sign
  out** within 5 seconds. To sign back in, you need your phone or computer
  again.
- **Only the glasses can sign themselves out.** Neither the approval page nor
  the dashboard can end the glasses' session, and changing your password
  doesn't either. Otherwise it ends after 7 days without use.

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
- **Up and down stay in the content.** In a conversation, file or list, swipes
  walk the content first, then scroll it, and only at its end move to the tabs
  at the top or the controls at the bottom.
- **Focus never gets lost.** If the focused control goes away (Stop when the
  reply ends, for example), focus moves to the nearest one, so the next pinch
  always does something.

### Agents

- **The agent list.** The home screen lists your running agents: your own
  process and Docker agents, plus agents shared with you.
- **Stopped agents** appear dimmed and can't be opened; start them from Flight
  Deck. **Refresh** reloads the list. If the list can't be loaded, the screen
  says so and offers **Retry**.
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
  3. The text lands in the field and focus moves to **Send**. Pinch it. For
     0.8 s after the composer closes, pinches are ignored, so the one that
     closed it can't press Send by accident.
- **Reading.** Each message is a reading block. Swipe up through the
  conversation; a long reply scrolls a page per swipe.
  - The latest 10 messages are shown. Pinch **Earlier messages (N)** for the
    rest; the HUD keeps the latest 40.
  - A very long reply is cut at about 6 KB, with a note saying how much more
    there is to read on Flight Deck.
  - Turns you didn't write (scheduled jobs, flows, tasks from other agents)
    show as one short system line, not as messages from you.
- **Below the conversation:**
  - **Next-step chips**: replies the agent suggests. Pinch one to send it; these
    work without the composer.
  - **File chips**: markdown files a reply mentions. Pinch one to open it.
  - **Approve / Deny**, when the agent asks for approval. The card counts down:
    if you don't answer in time, the agent **approves on its own** (after about
    15 s for a playbook on your own agent, about 60 s otherwise), and an answer
    that comes too late isn't sent. With **Read aloud** on, the glasses say
    "Approval needed".
  - **Stop** while the agent is working.
  - **New chat** (pinch twice to confirm). It appears only on your own agents
    and only while the agent is idle.
  - **Read aloud: On/Off.**
  - **Reconnect**, when the HUD has stopped trying (see below).
- **Connection.** After a network drop the chat reconnects by itself, and the
  status dot shows offline meanwhile. The **Reconnect** chip appears only when
  the HUD stops trying:
  - Flight Deck ended the connection, for example because your access to a
    shared agent was revoked or the agent went away;
  - the agent wasn't running when you opened the chat;
  - your own agent didn't answer 6 attempts in a row, about 30 s ("Agent
    unreachable").
- **While the display sleeps** the connection stays open for 90 s. After a
  longer sleep it closes, and it reopens when the display wakes: the agent then
  replays the conversation, which takes a moment on the glasses' slow link.
- **Read aloud** speaks new replies through the glasses speaker. There is one
  English voice, and the app can't change its volume. The glasses remember
  this setting.
- **Shared agents** answer as they would on the desktop: the glasses
  formatting rules don't reach them (see [Known limitations](#known-limitations)).

### Files

- **The file list.** The **Files** tab lists the agent's markdown files (`.md`,
  `.markdown`), newest first, with age and size.
- **Who wrote it.** Files someone else wrote are marked with the author's name:
  - on your own agents, files by anyone but you;
  - on a shared agent, files by anyone but you, **the owner's included**.

  If the file list couldn't be loaded, the reader says "Author unknown".
- **Reading a file.** Pinch a file to read it:
  - Swipe up or down to walk through headings, paragraphs, list items, code and
    tables.
  - Swipe **left or right** to turn a whole page.
- **Saved copies.** A file you opened before opens at once from memory. If the
  file list it came from is a few seconds old, the HUD checks for a newer
  version in the background and, if there is one, shows it with "Updated to
  the latest version."
- **Tables.** A table with up to 3 columns stays a table. Wider tables become
  **one card per row** (`column: value` lines), so nothing scrolls sideways.
- **Images and links.** Images show as their alt text and are never downloaded.
  Links are shown, but you can't follow them.
- **Embedded HTML** never renders as HTML: `<br>` becomes a line break, and an
  HTML block shows as its plain text.
- **Size limits.** The glasses don't open files over 1 MB. For very long files
  they download about the first 250 KB and show the first 200 KB. A slow
  download gets 90 s before it times out.

### Data

- **Tables.** The **Data** tab lists the agent's datastore tables, with row and
  column counts and when each table changed. Everything is read-only.
- **Rows.** Pinch a table to see its rows, **8 per page, newest first**. Swipe
  **left or right** (or pinch the **‹ Prev / Next ›** chips) to change page.
  - **Refresh** re-reads the page, since the agent may be writing to the table.
  - Opening a table from the table list always starts at the newest page.
- **Records.** Pinch a row to open the record, one field per block. Swipe
  **left or right** to step to the previous or next record. Back returns to the
  rows page that holds the record you were on.
  - The HUD follows the record itself, not its position. If the agent adds or
    deletes rows meanwhile, you stay on the same record and a note says the
    table changed. A record that was deleted stays on screen, marked as
    deleted.
  - Dates in a `date` column show as the calendar day written. Date-times with
    a time zone show in local time.

### Voice control (WebMCP)

If Meta enables WebMCP on your glasses, Meta AI can drive the HUD by voice
through three tools. WebMCP is off by default; Meta turns it on per device,
through Developer Mode or its rollout.

| Tool | Does |
|---|---|
| `claw_get_state` | Read-only. Reports the screen, the open agent, whether it is connected or working, any approval waiting, the suggested replies and the unread count. The latest reply comes as plain text in `latest_reply_quoted`, marked untrusted: a reply can quote web pages, emails or files, so Meta AI is told to treat it as information for you, never as instructions. |
| `claw_draft_message` | Writes a message (up to 4,000 characters) into the chat box of the agent open on the glasses and shows its Chat tab. It **never sends**: the box says "Written by Meta AI — not sent", focus is on **Send**, and you pinch Send (or edit the text first). Agent commands (text starting with `/`) are refused. |
| `claw_open_screen` | Opens `chat`, `files` or `data` for the current agent, or `agents` for the agent list. It adds no Back steps: tab changes replace the screen, and `agents` goes back to the list. |

Nothing reaches an agent on Meta AI's word alone: you pinch Send yourself, and
approvals can only be answered on the display. Everything works with swipes and
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
| **XREAL (One, One Pro, Aura, Beam Pro)** | Use the normal dashboard | A tethered monitor for a phone or PC, or a full Android host (Beam Pro). The viewport is large and input comes from the host's pointer and keyboard, so open the usual Flight Deck URL in the host's browser. |
| **INMO Air 3** | Normal dashboard; we haven't tested it | Full Android with a 1920 × 1080 viewport, so the dashboard fits. Its own input is a ring, the temple touchpad or voice, so pair a Bluetooth keyboard or mouse for the dashboard, or try `https://<your-host>/hud/`. |
| **RayNeo X3 Pro** | Try the HUD; we haven't tested it | A 640 × 480 display with temple-touchpad or ring input. How that input reaches web pages is unknown (on Android, probably arrow keys and Enter), and no browser has been reviewed for it. If one is available, open `https://<your-host>/hud/` (or `/?ui=hud`): the small viewport and arrow-key input suit the HUD better than the dashboard. |

## Deployment checklist

Meta's glasses only load web apps from a **public HTTPS URL**: no HTTP, no LAN
address. Usually that means a tunnel or reverse proxy in front of Flight Deck.

**A public URL is a team deployment.** First complete the
[team deployment checklist](team-deployment.md): set every variable in its
"Required environment variables" table. Then check these before you register
the URL:

| Setting | Why |
|---|---|
| A public HTTPS URL, at the root of the host | Meta requires HTTPS and a URL anyone can reach. Serve Flight Deck at the root (`https://<your-host>/`): every URL the HUD uses starts at `/`, so a path prefix such as `/deck/` breaks it. |
| No access gate on what the HUD uses | The glasses can't get through HTTP auth or an access gate (Cloudflare Access, basic auth), and the HUD handles sign-in itself. If you gate the dashboard, let through `/hud*`, `/assets/*`, `/fd/*` (including the `/fd/agent-ws…` WebSockets), `/icon-*.png` and `/apple-touch-icon.png`, plus `/` if you rely on the glasses redirect. `/fd/*` is then protected by Flight Deck's own sign-in only, so `FD_AUTH_ENABLED=true` is a must. |
| `FD_AUTH_ENABLED=true` | With accounts off, anyone with the URL acts as the local admin. The HUD then skips sign-in, shows a "Sign-in off" warning, and pairing is unavailable. |
| `FD_JWT_SECRET` set | Without it, Flight Deck generates a random secret at every start, so all access tokens die on restart. (The HUD recovers through the refresh cookie.) |
| `FD_LOCKDOWN=1` | **Required on a public URL.** It turns off routes that are not for the internet: the `/fd/projects/*` router, which has no sign-in at all (without lockdown, anyone can create projects there and send tasks to your agents), `/fd/vfs/browse-fs` and `POST /fd/vfs/links`. It also makes the agent secret mandatory even from loopback, so a tunnel or proxy on the same host can't pass remote callers off as local ones. The HUD uses none of these routes. |
| `FD_AGENT_SHARED_SECRET` set | A long random string. Agents send it as `X-Agent-Secret` to Flight Deck's agent routes (`/fd/basna/agent/*`, `/fd/vatra/agent/*`, the scheduler). Under `FD_LOCKDOWN` those routes no longer trust loopback, so agents need it. |
| `FD_PUBLIC_URL=https://<your-host>` or `FD_ALLOWED_HOSTS=<your-host>` | The origin guard refuses requests whose host or origin it doesn't know. Without this, every HUD request through the tunnel fails. |
| Secure cookies | `FD_LOCKDOWN=1` already marks the refresh cookie and `fd_ui` as Secure. Leave `FD_COOKIE_SECURE` unset (or `1`): `FD_COOKIE_SECURE=0` turns that off, and is for local http development only. |
| `FD_GLASSES_BRIDGE_TOKEN` set | The legacy `/glasses/*` bridge has no user accounts. On a public URL, anyone can use it unless this shared secret is set. The HUD doesn't use the bridge. |
| Compression through the tunnel | Flight Deck gzips the HUD page and `/assets/*` itself, so the first load is about 150 KB on the wire. Check that `/assets/hud-*.js` reaches the glasses with `content-encoding: gzip` (or `br`). A proxy that asks Flight Deck for uncompressed files and doesn't compress them itself puts the first load back at about 490 KB: roughly 8 s on the glasses' ~500 Kbps link, against Meta's 5 s target. |
| Client IPs through the proxy | The approval page shows the requesting device's IP address, and pairing is limited per client network. A tunnel or proxy **on the same host** that connects from 127.0.0.1 (cloudflared, ngrok, a local Caddy, or nginx with `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`) already works: uvicorn trusts `X-Forwarded-For` from 127.0.0.1 by default. If the proxy connects **from another address** (a Docker network, another host, or `::1` when Flight Deck listens on `::`), set `FORWARDED_ALLOW_IPS` to that proxy's address. Otherwise every device shows up as the proxy and they share one pairing budget: 10 codes per 10 minutes and 3 pending codes, so a fourth device's code replaces the first one's. **Never use `*`**: uvicorn then believes the header from anyone, so any client can choose the IP Flight Deck sees and get past the pairing limits. |

The built HUD ships with Flight Deck, so deploying needs no build step. If you
changed `flight-deck/src/hud`, see [Building the HUD](#building-the-hud).

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
| `markdown/` | `react-markdown` + `remark-gfm`, plus a small rehype pass that turns embedded HTML into plain text, turns wide tables into cards and makes blocks focusable. Large files are split into pieces that parse on their own and are rendered progressively. |
| `webmcp.ts` | The three WebMCP tools |
| `device.ts` | Host detection (`meta-display`, `rokid-lumen`) and the device label sent with pairing |
| `captain_claw/flight_deck/hud_routes.py` | `/hud`, `/hud/{rest}`, the manifest, the service worker, `/fd/hud/config`, the `/` redirect, and the middleware that gzips `/hud*` and `/assets/*` and marks hashed assets immutable |
| `captain_claw/flight_deck/auth_routes.py` | The `/fd/auth/pair/*` endpoints |

To develop locally, run Flight Deck on port 25080 and `npm run dev` in
`flight-deck/`, then open `http://localhost:5173/hud/`. Vite rewrites `/hud*`
to `hud.html` and proxies `/fd`, the manifest and the service worker.

### Building the HUD

Flight Deck serves the committed build in `captain_claw/flight_deck/static/`.
After editing `flight-deck/src/hud/**` or `flight-deck/hud.html`:

1. Run `npm run build` in `flight-deck/`. It builds both the dashboard and the
   HUD. Old hashed files are kept (`emptyOutDir` is off).
2. Commit `captain_claw/flight_deck/static/hud.html` and the new files in
   `static/assets/`.

Without this, `/hud/` keeps serving the previous build, with no warning: the
503 "HUD not built" only appears when `static/hud.html` is missing.

### Rules the screens follow

**Focus engine (`focus.ts`)**
- **It owns the arrow keys.** It calls `preventDefault()` and moves focus
  geometrically inside the one mounted `.hud-screen`, as Meta's toolkit does.
  - A component that handles a key itself (Left/Right paging) calls
    `preventDefault()` first. The engine skips events that are already handled.
- **Tall elements are read before focus leaves.** A focused element taller than
  its scroller scrolls page by page first. The first and last stops pin the
  scroller to its ends.
- **Up/Down stay in the scroller.** From inside `.hud-scroll`, Up/Down look for
  a stop in the scroller first (nearest row, then the closest item in it),
  then scroll it a page, and only at its end move to the header or footer.
  From the header or footer, scroller stops count only where they are visible.
- **Keys.** It ignores `key === 'Unidentified'` and uses `.key`, never `.code`.
- **One pinch, one action.**
  - `useActivate` → `claimActivation()` drops any activation within 500 ms of
    the previous one, **on any element**, unless an arrow key came in between.
    The two halves of a pinch (Enter and click) can land on different elements
    when the first half moves focus.
  - The Enter half of such a pinch that lands on a text field is swallowed in
    the capture phase, so no newline is inserted.
  - `quietActivations()` mutes activations for a while: the composer uses it
    for 800 ms after a commit, against the late closing pinch.
  - After Back, `focusInitial()` restores the remembered element even if the
    host's focus reset got there first. `pinFocus()` undoes a reset that comes
    later, and the one after the composer closes.
- **Lost focus is recovered.** When the focused element is removed and focus
  falls to `<body>`, the engine focuses the element with the same `data-fk`,
  else the nearest stop, else the screen's first stop. A plain blur (the host's
  composer opening) is left alone.
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
- **Going up.** "Up" actions (**All agents**, the record screen's **Back to
  list**, `claw_open_screen` from a deeper screen) call `navigateUp(target)`. When
  the target entry is below in history, they go back to it, so its focused row
  comes back and Back from there still leaves. Otherwise they replace the
  current entry; they never push a copy.
- **Depth.** The deepest chain is 4 entries (agents → agent → rows → record),
  under Meta's 5-entry cap. A deep link on a fresh launch rebuilds its ancestor
  chain, so Back walks up instead of exiting.
- **Focus restore.** Each pushed entry remembers the focused element's
  `data-fk`, and Back restores focus to it. Give list rows a stable `fk`.
- **Reloads.** The route lives in the URL, so a reload (universal menu →
  Restart) returns to the same screen.
- **Session end.** `collapseHistory()` goes back to the lowest entry this page
  load created and turns it into a plain `/hud/` launch (see
  [Staying signed in](#staying-signed-in)).

**Data access**
- **Guarded routes only.** The HUD calls only Flight Deck's JWT-guarded `/fd/*`
  routes and `/fd/agent-ws`. These check that you own the agent or that it is
  shared with you.
  - Never the legacy `/glasses/*` bridge.
  - Agent secrets are resolved on the server and never appear in the page.
- **Session renewal (`api.ts`).** Only Flight Deck's own 401s (or an expired
  access token) renew the session, once, single-flight. A 401 an agent relays
  through a proxy comes back as "The agent refused Flight Deck's access." and
  touches nothing. A failed refresh ends the session only if Flight Deck is
  reachable and refuses a second try too; otherwise the request fails as a
  network error and the session is kept.
- **Timeouts.** Response headers get 20 s, then the body 20 s (90 s for a
  file, whose read stops after about 250 KB). Pairing calls get 15 s.
- **Text only.** Everything renders as React text: no
  `dangerouslySetInnerHTML`, no raw HTML in markdown, no remote images.

### API routes used

| Route | Auth | Used for |
|---|---|---|
| `GET /fd/auth/status`, `POST /fd/auth/refresh`, `POST /fd/auth/login`, `POST /fd/auth/logout` | public / refresh cookie | Startup, the session, the email fallback, sign-out |
| `POST /fd/auth/pair/start`, `POST /fd/auth/pair/poll` | public, rate-limited per client network | Glasses: get a code and wait for approval (the approved poll returns the session). The HUD gives each call 15 s. |
| `GET /fd/auth/pair/lookup?code=`, `POST /fd/auth/pair/approve` | `Authorization: Bearer` header **plus** the approver's own `fd_refresh` cookie for a live session of the same user; else 403 (the `?fd_token=` fallback is refused) | `/hud/pair`: show the requesting device; approve or deny |
| `GET /fd/processes`, `/fd/containers`, `/fd/shared-agents` | Bearer | The agent list (filtered by owner) |
| `GET /fd/agent-files/localhost/{port}`, `/fd/agent-file-view/localhost/{port}?path=` | Bearer, owner-checked | Your own agents' files |
| `GET /fd/shared-agents/files?ref=`, `/fd/shared-agents/files/view?ref=&id=` | Bearer, member | Shared agents' files |
| `GET /fd/agent-datastore/localhost/{port}/tables`, `…/tables/{table}/rows` | Bearer, owner-checked | Your own agents' datastore |
| `GET /fd/shared-agents/datastore/tables?ref=`, `…/tables/{table}/rows?ref=` | Bearer, member | Shared agents' datastore |
| `WS /fd/agent-ws/localhost/{port}?fd_token=`, `WS /fd/agent-ws-shared?ref=` | access token | Chat (the agent's main lane) |
| `GET /fd/hud/config` | Bearer | The glasses rendering rules, sent with the first chat message of each connection to your own agents (never to shared agents; see [Known limitations](#known-limitations)) |
| `GET /hud`, `/hud/`, `/hud/{rest}` | public | The page: never cached by the browser (the service worker is the only cache), never framed (`X-Frame-Options: DENY`, `frame-ancestors 'none'`), gzipped |
| `GET /hud/manifest.webmanifest`, `GET /hud/sw.js` | public | The launcher manifest (PNG icons); the service worker. It is cache-first for `/assets/*` and network-first for `/hud*` pages, falling back to the last good page when Flight Deck doesn't answer within 4 s, the network fails, or the answer is an error page (a proxy's 502, a tunnel's 530). It never touches `/fd/*`. |

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
  first load stays under 300 KB transferred (DevTools Network with "Disable
  cache" and throttling). Measure the built page served by Flight Deck, not
  the Vite dev server: it is gzipped and comes to about 150 KB. Through a
  tunnel, also check the `content-encoding` of `/assets/*`.

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
- [ ] With the network off, screens show error states with Retry, you stay signed in, and chat reconnects when the network returns.
- [ ] After the display sleeps for under 90 s, the chat is still connected on wake; after a longer sleep it reconnects.
- [ ] Sign out needs a second pinch.
- [ ] With WebMCP on, asking Meta AI what Captain Claw is showing calls `claw_get_state`, and asking it to message the agent puts the text in the chat box, unsent.

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
- **No remote sign-out.** Only Sign out on the glasses ends their session
  early. The approval page and the dashboard can't, and a password change
  doesn't either.
- **A network drop at launch** can show the sign-in screen although the session
  is still valid: the launch's refresh failed. Restart from the universal menu
  once the network is back.
- **Back after the session ends.** History entries from before a Restart stay
  in place; each Back over one of them reloads the sign-in screen before the
  system menu opens (see [Staying signed in](#staying-signed-in)).
- **Shared agents don't get glasses formatting.** Flight Deck doesn't pass the
  glasses surface along a member's chat, so the agent would ignore the rules;
  the HUD doesn't send them. Replies from a shared agent are written for the
  desktop: long paragraphs and wide tables. The HUD still pages them one swipe
  at a time, turns wide tables into cards, and cuts very long replies at about
  6 KB.
- **Approvals don't badge the Chat tab.** A card that arrives while you are on
  Files or Data is announced only when Read aloud is on, and the agent approves
  on its own if you don't answer in time.
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
