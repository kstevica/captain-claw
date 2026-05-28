# WhatsApp ↔ Captain Claw cookbook

End-to-end recipe for getting a WhatsApp Business Cloud API number to talk
to a Captain Claw agent through Flight Deck, with the reply landing back in
WhatsApp **and** on the glasses HUD on the same channel. Every step was
walked through on a real install (May 2026); each gotcha that bit us is
called out inline so you don't repeat them.

**Time budget:** ~30 minutes if everything goes right on the first try.
Realistically 1–2 hours including the Meta-side click-throughs.

## What you're building

```
WhatsApp user (your phone)
        ▼ message
Meta Cloud API
        ▼ webhook POST (HMAC-signed)
FD ─── /whatsapp/webhook ──▶ whatsapp_bridge
                              ├─ HMAC verify
                              ├─ PSID/WAID allowlist
                              ├─ /c <channel> slash command
                              ├─ photo? → face_index.recognize() + agent upload
                              └─ user event → channel bus
                                              ▲
                              agent reply ────┤
                                ├─▶ Graph API Send ───▶ user's WhatsApp thread
                                └─▶ glasses_view via WS ─▶ glasses HUD
```

A Messenger bridge bound to the same channel (`/c lounge`) fans out to the
Messenger thread as a third surface — same conversation, three displays.

## Prerequisites

- A Facebook account with a Meta Business Account (free; create one at
  business.facebook.com if you don't have one).
- A phone number on which you use WhatsApp — this is the **recipient**
  (the user side). You do **not** need a second SIM.
- A public HTTPS URL for Captain Claw's Flight Deck — typically via
  cloudflared or ngrok. Throughout this doc: `https://glasses.example.com`.
- Flight Deck running with the WhatsApp bridge code (post the changes
  in [`captain_claw/flight_deck/whatsapp_bridge.py`](../captain_claw/flight_deck/whatsapp_bridge.py)).
- A Captain Claw agent already running and visible in the FD UI — note
  its **slug** (the cyan pill in the agent card) and **auth token** (from
  the agent's `config.yaml`, under `web.auth_token`).

## Step 1 — Create a Meta App and enable WhatsApp

1. Go to https://developers.facebook.com/apps/ → **Create App**.
2. App type: **Business**. Display name: anything (e.g. "claw-bridge").
3. Inside the app, **Add Product** → **WhatsApp** → **Set up**.

Meta provisions a **test sender number** for you (looks like `+1 555 …`).
Inside the API setup page you'll see:

- `Phone number ID` — copy as `WHATSAPP_PHONE_NUMBER_ID`
- `WhatsApp Business Account ID` (WABA ID) — copy for later (Step 4)
- `Temporary access token` — copy as `WHATSAPP_ACCESS_TOKEN` (this
  rotates every 24 h; we'll fix that in Step 7)

**Gotcha**: The test sender can only send to phone numbers you explicitly
allow in this same page. Click **Add phone number** under "To" and enter
your own number. Meta sends an SMS OTP — enter it. Repeat for up to 5
recipient numbers.

## Step 2 — Find your App Secret

App dashboard → **App Settings** → **Basic** → **App Secret** → **Show**.

Copy as `WHATSAPP_APP_SECRET`. Used for HMAC verification of every webhook
POST — if this is wrong, every webhook gets 401 in FD's log.

## Step 3 — Configure the webhook

Same WhatsApp section in the dashboard → **Configuration** → **Webhook** →
**Edit**:

- **Callback URL**: `https://glasses.example.com/whatsapp/webhook`
- **Verify token**: any string you like — copy as `WHATSAPP_VERIFY_TOKEN`.
  This is what FD echoes back to prove ownership.

Before clicking **Verify and save**, make sure FD has the env var set and
restart it (see Step 5 — `.env` setup). Then click Verify and save.

**Gotcha #1**: If the response body comes back as HTML instead of the
challenge string, your `whatsapp_router` isn't mounted (FD's SPA catch-all
is serving `index.html`). Run `curl -i "https://glasses.example.com/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=<token>&hub.challenge=test"` to confirm. Expected response:

```
HTTP/2 200
content-type: text/plain; charset=utf-8

test
```

**Gotcha #2**: Meta's dashboard sometimes shows "couldn't be validated" on
first try even when FD returns 200 cleanly. Click Verify and save a second
time.

After verifying, **subscribe to the `messages` field** — there's a list
below the URL with **Subscribe** buttons next to each field. Click it for
`messages`. Without this, no events flow.

## Step 4 — Subscribe your App to the WABA (the easy-to-miss step)

This is the single biggest gotcha. There are **two separate subscriptions**:

1. **App-level webhook config** (Step 3 above) — controls which URL Meta
   calls. Used for the dashboard's "Send Test" button.
2. **WABA → app subscription** — controls whether real user messages get
   forwarded to your app at all.

The dashboard UI does not always create #2 when you do #1. You'll see
"Send Test" work but real messages silently dropped, with a "Sample
Webhooks" preview that taunts you with the payload Meta refuses to
actually forward.

Check the current state:

```bash
curl -s "https://graph.facebook.com/v18.0/<WABA_ID>/subscribed_apps" \
     -H "Authorization: Bearer <WHATSAPP_ACCESS_TOKEN>" \
     | python3 -m json.tool
```

If you see only Meta's default app ("WA DevX Webhook Events 1P App") or
empty `data: []`, your app isn't subscribed. Fix:

```bash
curl -i -X POST \
     "https://graph.facebook.com/v18.0/<WABA_ID>/subscribed_apps" \
     -H "Authorization: Bearer <WHATSAPP_ACCESS_TOKEN>"
```

Expected: `{"success": true}`. Re-run the GET to confirm your app now
appears in `data`.

## Step 5 — Set Captain Claw env vars

Edit `/path/to/captain-claw/.env`:

```
# Required
WHATSAPP_PHONE_NUMBER_ID=<from Step 1>
WHATSAPP_ACCESS_TOKEN=<from Step 1, or Step 7 if you've done it>
WHATSAPP_APP_SECRET=<from Step 2>
WHATSAPP_VERIFY_TOKEN=<from Step 3>
WHATSAPP_ALLOWED_WAIDS=<your phone, no '+', e.g. 385976707736>

# Channel + agent binding
WHATSAPP_DEFAULT_CHANNEL=skchannel
WHATSAPP_DEFAULT_AGENT_SLUG=<from FD UI: the cyan pill on the agent card>
WHATSAPP_DEFAULT_AGENT_AUTH=<from agent's config.yaml web.auth_token>
```

Multiple WAIDs: comma-separate. Leading `+` is tolerated/stripped.

**Gotcha**: Do **not** also set `WHATSAPP_DEFAULT_AGENT_PORT` — slug-based
binding survives port reassignment and wins by priority. A stale port
left in `.env` used to silently override slug and route WS traffic to FD
itself (port 8765 or wherever FD listens), producing 403s; the bridge now
prefers slug, but a stray port still confuses things.

**Gotcha**: `.env` must be in **FD's current working directory**. The
bridge calls `load_dotenv()` on startup which walks up from CWD. If FD is
started from a different directory, the file isn't found and every value
is silently missing.

Restart FD so it picks up the env:

```bash
pkill -f captain-claw-fd
# restart as usual; ensure you start from the dir with .env
```

Verify the env actually loaded:

```bash
tr '\0' '\n' < /proc/$(pgrep -f flight_deck | head -1)/environ | grep WHATSAPP
```

Should print all six variables.

## Step 6 — Switch the app to Live mode

Meta refuses to deliver real user messages while the app is in
"Development" mode — only the dashboard's "Send Test" button works. To
get user messages flowing:

1. App dashboard → top of page, **App Mode** toggle showing
   **Development**. Click **Live**.
2. Meta checks that you have:
   - A **Privacy Policy URL** set in App Settings → Basic. Any HTTPS URL
     serving any HTML works for personal use; even a `/privacy` route on
     your own tunnel is fine.
   - An **App Icon** uploaded
   - A **Category** set
3. With those filled in, the toggle flips.

For self-hosted personal use, no business verification is needed — just
the above three fields.

## Step 7 — Get a System User permanent token (do this once)

The token from Step 1's API setup page **expires every 24 hours**. Every
day, your agent stops replying and the FD log fills with
`401 Unauthorized`. Solve it permanently with a System User token:

1. Meta dashboard top-right → **Business Settings**
   (business.facebook.com).
2. Sidebar → **Users** → **System Users** → **Add**. Name: `claw-bridge`.
   Role: **Admin**.
3. With the system user selected, **Assign Assets**:
   - Add your **App** (full control)
   - Add your **WhatsApp Business Account** (full control)
4. **Generate new token**:
   - App: pick yours
   - Token expiration: **Never**
   - Permissions: `whatsapp_business_messaging`, `whatsapp_business_management`
5. Copy the token (Meta only shows it once). Update
   `WHATSAPP_ACCESS_TOKEN` in `.env`, restart FD.

You can now ignore the daily rotation forever.

## Step 8 — End-to-end test

1. On the prod box, tail the FD log so you can watch events arrive.
2. Open `https://glasses.example.com/glasses/view?c=skchannel` in any
   browser tab (or on the glasses themselves) and keep it visible.
3. On your phone, open WhatsApp and start a chat with the test sender
   number (the `+1 555 …` from Step 1).
4. Send a message: `Hello`.

Expected log sequence within ~3 seconds:

```
POST /whatsapp/webhook 200                              ← Meta delivered
POST graph.facebook.com/.../messages "HTTP/1.1 200 OK"  ← typing indicator
POST graph.facebook.com/.../messages "HTTP/1.1 200 OK"  ← agent reply going out
```

Expected on screens:

- Phone (WhatsApp): blue double-tick on your message + "typing…" briefly +
  the agent's reply appears
- Glasses HUD: a "You" bubble with "Hello" + an "Agent" bubble with the
  reply

If you see both, you're done.

## Slash commands inside WhatsApp

The bridge intercepts these before forwarding to the agent. Send them
as ordinary WhatsApp messages:

| Command | Effect |
|---|---|
| `/c lounge` | Move your channel binding to `lounge`. Subsequent messages route there. |
| `/c` | Reply with your current channel. |

Channel rebind survives until FD restarts (in-memory state).

## Diagnostic table — what each failure mode means

| Symptom | Cause | Fix |
|---|---|---|
| No `POST /whatsapp/webhook` in log when you send | App still in dev mode, OR `messages` field not subscribed, OR WABA not subscribed to your app | Step 6, Step 3, Step 4 respectively |
| `POST /whatsapp/webhook 401` | `WHATSAPP_APP_SECRET` mismatch | Step 2 — copy fresh from dashboard |
| `POST 200` then "ignored: no allowlist" | `WHATSAPP_ALLOWED_WAIDS` empty in FD process | Check env loaded (Step 5 verify command) |
| `POST 200`, then `WebSocket /ws ... 403` | Wrong port (stale `WHATSAPP_DEFAULT_AGENT_PORT`) or wrong auth | Remove port from `.env`; verify `AGENT_AUTH` matches agent's `config.yaml` |
| `POST 200` followed by `Bridge offline: no agent available` reply | Slug doesn't match any running agent | Confirm slug via `curl /glasses/agents` |
| Webhook works but no reply lands in WhatsApp | `WHATSAPP_ACCESS_TOKEN` expired (you'll see `HTTP/1.1 401 Unauthorized` from `graph.facebook.com` in the log) | Refresh token in Step 1, or do Step 7 once |
| Real user message ignored but Meta dashboard shows it in "Sample Webhooks" | WABA not subscribed to your app | Step 4 — `POST /<WABA_ID>/subscribed_apps` |

## Where things live in the code

- **Bridge entry**: [`captain_claw/flight_deck/whatsapp_bridge.py`](../captain_claw/flight_deck/whatsapp_bridge.py) — webhook routes, Send API, attachment download, slash commands.
- **Shared Meta helpers**: [`captain_claw/flight_deck/meta_webhook_bridge.py`](../captain_claw/flight_deck/meta_webhook_bridge.py) — HMAC verify, channel-bus callback registrar, agent slug resolver.
- **Channel bus**: [`captain_claw/flight_deck/glasses_bridge.py`](../captain_claw/flight_deck/glasses_bridge.py) — `_ChannelState`, `_broadcast`, `_ensure_agent_binding`.
- **`.env` loading**: [`captain_claw/flight_deck/server.py`](../captain_claw/flight_deck/server.py) top — `from dotenv import load_dotenv; load_dotenv()`.

## Reference: what the bridge does on every inbound message

1. HMAC-verify the request body against `WHATSAPP_APP_SECRET`. Reject 401 if mismatch.
2. Parse the payload; extract the message and the sender WAID.
3. Drop if WAID isn't in `WHATSAPP_ALLOWED_WAIDS` (silent).
4. Fire `_mark_read_and_typing()` — best-effort: blue tick + "typing…".
5. If text starts with `/c`, handle the slash command and stop.
6. Resolve target channel (default or per-PSID rebound).
7. Resolve target agent via `WHATSAPP_DEFAULT_AGENT_SLUG` → FD's process
   registry → current port + auth (falling back to `_AGENT_AUTH` env var
   override if registry has no `web_auth`).
8. If photo: download via 2-step Cloud API media fetch, run
   `face_index.recognize()` (broadcasts person card to the channel), then
   forward photo to the agent's `/api/image/upload`.
9. Broadcast a `type: "user"` event to the channel bus (glasses HUD sees
   it instantly).
10. Send to the agent over WS as a `chat` message, with the
    glasses-rendering system prompt prepended on the first message of
    this binding.

When the agent replies, it broadcasts `type: "agent"` to the channel.
Subscribers receive it:

- Glasses view (over WebSocket) renders it
- WhatsApp callback strips markdown, calls Send API → user's thread

A Messenger user bound to the same channel gets the same reply
simultaneously via the parallel Messenger callback.

## Future work / known limits

- **Voice notes** are not transcribed yet. Could add via Whisper or a
  similar STT. Deferred because the WhatsApp on Ray-Ban Display transcribes
  inbound voice notes natively for the user already.
- **Templates / proactive sending** outside the 24-hour service window
  isn't implemented. The bridge reacts; it doesn't initiate.
- **Mark-as-read** without typing is not separately exposed. We always
  mark+type as a single call.
- **`/c` rebinds** are in-memory. Reset on FD restart. Fine for hack
  status; persistent storage is ~10 lines if needed.
