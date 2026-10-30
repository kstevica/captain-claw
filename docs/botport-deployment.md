# BotPort — connecting Flight Decks

BotPort is the hub that lets agents on one Flight Deck work with agents on
another. Each Flight Deck dials **out** to the hub and keeps one WebSocket
open; the hub routes requests between decks; the receiving deck hands each
request to its own agent, exactly as it would a local consult, and streams the
answer back.

```
 Agent "Planner"          Flight Deck "vps1"            BotPort hub            Flight Deck "pi"          Agent "Researcher"
 flight_deck tool  ──▶  /fd/consult-peer  ══ wss ══▶  routes Agent@deck  ══ ws ══▶  BotPort link  ──▶  own lane on the agent
   "Researcher@pi"   ◀──  ndjson events   ◀═════════   progress + result ◀═══════   proxy (FD)    ◀──   answer
```

* Only the hub needs to be reachable. Decks behind NAT (a Raspberry Pi at home,
  a laptop) work, because they connect out.
* Agents never talk to the hub and never hold its credentials: an agent asks
  its own Flight Deck, which carries the request.
* Any number of decks. An agent on another deck is addressed as
  **`Agent@deck`** (e.g. `Researcher@pi`).

The examples set up the hub as a plain Python process in **tmux on a Raspberry
Pi**, with Flight Decks on two VPSs (`vps1`, `vps2`) and one on the Pi itself
(`pi`). Nothing depends on that layout: any Linux box can run the hub, deck
names are yours to choose, and any number of decks can join.

**Contents:** [1 Install](#1-install-the-hub) · [2 Keys](#2-keys--one-per-deck) ·
[3 Configure](#3-configure-the-hub) · [4 Run](#4-run-it-in-tmux) ·
[5 Reach it](#5-make-the-hub-reachable) · [6 Connect the decks](#6-connect-each-flight-deck) ·
[7 Share agents](#7-choose-what-each-deck-shares) · [8 Use it](#8-use-it) ·
[9 Watch it](#9-watch-the-network-the-hub-dashboard) · [10 First test](#10-first-test-across-your-decks) ·
[Safeguards](#what-a-remote-request-can-and-cannot-do) · [Reliability](#reliability--what-happens-when-things-drop) ·
[Troubleshooting](#troubleshooting) · [Upgrading](#upgrading)

---

## 1. Install the hub

The hub needs **Python 3.11+** (Raspberry Pi OS Bookworm has it; Bullseye's
Python is too old).

```bash
sudo apt install -y python3-venv tmux git
git clone https://github.com/kstevica/captain-claw.git ~/captain-claw   # or: cd ~/captain-claw && git pull
python3 -m venv ~/botport-venv
~/botport-venv/bin/pip install ~/captain-claw/botport
~/botport-venv/bin/botport --version
```

This is a normal (not editable) install: after every `git pull`, run the
`pip install` line again and restart the hub (`scripts/botport-tmux.sh
restart`).

The hub package is small (aiohttp, aiosqlite, pydantic, pyyaml). LLM-assisted
routing of *unaddressed* requests is optional and pulls in `litellm`:
`~/botport-venv/bin/pip install "$HOME/captain-claw/botport[llm]"`.
Deck-to-deck requests are addressed (`Agent@deck`) and never need it.

## 2. Keys — one per deck

Every Flight Deck connects with a key of its own, and the key is **bound to
the deck's name**: a deck that connects with it *is* that deck and cannot
pretend to be another. Two ways to make keys, side by side:

* **On the hub's dashboard (recommended).** Once the hub runs, its **Decks**
  page adds a Flight Deck in one step ([step 6](#6-connect-each-flight-deck)):
  it makes the key, gives the deck a display name and a description, and hands
  you a *join code* to paste into that deck. The deck can connect at once: no
  file to edit, no restart. Rotating or removing a key there also takes effect
  at once. With this, leave `auth.keys` empty and go on to step 3. (The hub
  makes keys only with auth on: on an open hub any deck joins without one.)
* **In the config file.** `botport gen-key` prints an `auth.keys` entry;
  the hub reads the file at start, so a new or changed key needs a restart.
  Handy when the dashboard is off.

  ```bash
  ~/botport-venv/bin/botport gen-key pi
  ```

  It prints the entry for the hub (`instance`, `key`, `secret`) and the values
  to enter in that Flight Deck. A key without an `instance` cannot be used by a
  Flight Deck at all (only by a legacy agent), and no agent may register under
  a deck's name.

## 3. Configure the hub

```bash
mkdir -p ~/.botport && chmod 700 ~/.botport
cp ~/captain-claw/botport/config.example.yaml ~/.botport/config.yaml
chmod 600 ~/.botport/config.yaml
openssl rand -hex 24          # the admin token for the dashboard and the API
nano ~/.botport/config.yaml
```

Set the admin token and empty the example keys (decks added on the
dashboard need nothing here); the rest can stay:

```yaml
server:
  host: "0.0.0.0"          # see "Make the hub reachable" for 127.0.0.1 behind a proxy / tunnel
  port: 23180
  admin_token: "<the openssl output>"
auth:
  enabled: true
  keys: []                 # or `botport gen-key` entries, if you make keys that way
logging:
  file: "~/.botport/botport.log"   # rotated; survives a closed tmux pane
```

The dashboard's keys live in the hub's database (`botport.db` in its data
directory, `~/.botport/` by default) with their display names and
descriptions. Only a fingerprint of each secret is stored there.

Check it. This prints the effective configuration (secrets masked) and every
warning, and exits non-zero on mistakes; the last line should be `OK`. With no
keys in the file it warns *auth.keys is empty: only Flight Decks added on the
dashboard (Decks) can connect*, which is what you want here:

```bash
~/botport-venv/bin/botport check -c ~/.botport/config.yaml
```

**Always pass `-c`** (the tmux script does). Where settings come from: `-c FILE`
(or `$BOTPORT_CONFIG`), else `./botport.yaml`, else `./config.yaml` *if it is a
BotPort config*, else `~/.botport/config.yaml`. Started by hand from
`~/captain-claw/botport/` without `-c`, the hub would load the repository's
development `botport/config.yaml` (port 33080, auth off).

Environment variables override the file: `BOTPORT_<SECTION>__<KEY>`, e.g.
`BOTPORT_SERVER__ADMIN_TOKEN=…`. To keep secrets out of the file, put such
lines in `~/.botport/env` (`chmod 600`): the tmux script loads it for `check`,
for every (re)start of the hub and at boot. Variables only exported in your
shell would not reach a hub started by cron.

## 4. Run it in tmux

```bash
~/captain-claw/scripts/botport-tmux.sh start    # tmux session "botport"; restarts the hub if it exits
~/captain-claw/scripts/botport-tmux.sh status   # is it up? (/healthz)
~/captain-claw/scripts/botport-tmux.sh logs     # the last 200 lines
~/captain-claw/scripts/botport-tmux.sh attach   # watch it (detach: Ctrl-b d)
~/captain-claw/scripts/botport-tmux.sh restart  # after editing the config or upgrading
~/captain-claw/scripts/botport-tmux.sh stop
```

The script uses `~/botport-venv/bin/botport`, `~/.botport/config.yaml` and (if
present) `~/.botport/env`; override with `BOTPORT_BIN=…`, `BOTPORT_CONFIG=…`,
`BOTPORT_ENV_FILE=…`, `BOTPORT_SESSION=…`. `start` runs `check` first;
`restart` leaves a running hub alone when the new configuration does not pass.
To start the hub at boot, `crontab -e` and add:

```
@reboot /home/<user>/captain-claw/scripts/botport-tmux.sh start
```

What the hub prints when it is up:

```
BotPort v0.8.4 (protocol 2) listening on http://0.0.0.0:23180
  Flight Decks / agents connect to: ws://<this-host>:23180/ws
  Dashboard: http://<this-host>:23180/  (admin token required)
  Health:    http://<this-host>:23180/healthz
  Auth: enabled (0 key(s) in the config file, 3 deck(s) added on the dashboard)
```

It never silently moves to another port: if 23180 is taken it says so and
exits. `SIGTERM` / Ctrl-C closes the decks' sockets with "going away" and they
reconnect on their own when the hub is back.

## 5. Make the hub reachable

The deck's key and secret travel in the first frame, and requests carry your
agents' work: **never use plain `ws://` across the open internet**. Pick one:

| Option | Hub config | Address a remote deck uses | Notes |
|---|---|---|---|
| **Tailscale / WireGuard** (simplest) | `host: "0.0.0.0"` | `ws://<pi-tailnet-ip>:23180/ws` | Encrypted by the VPN, nothing exposed publicly, works behind home NAT. Recommended. |
| **Cloudflare Tunnel** | `host: "127.0.0.1"`, `behind_proxy: true` | `wss://botport.example.com/ws` | `cloudflared tunnel` → `http://localhost:23180`; WebSockets work out of the box. |
| **Port forward + Caddy** | `host: "127.0.0.1"`, `behind_proxy: true` | `wss://botport.example.com/ws` | Router forwards 443 to the Pi; `Caddyfile`: `botport.example.com { reverse_proxy 127.0.0.1:23180 }` (needs a domain / DDNS). |
| **Port forward + built-in TLS** | `tls_cert`, `tls_key` set | `wss://botport.example.com:23180/ws` | Certificates you renew yourself; the hub loads them at start, so restart after renewing. |

With Tailscale: install it on the Pi and every remote deck's machine, put them
on one tailnet, read the Pi's address with `tailscale ip -4`, and check from
each remote machine:

```bash
curl -s http://<pi-tailnet-ip>:23180/healthz      # must show "protocol": 2
```

If that hangs and the Pi runs ufw: `sudo ufw allow in on tailscale0 to any
port 23180 proto tcp`.

With `behind_proxy: true`, bind the hub to `127.0.0.1` (so only the proxy
reaches it): it then takes the client address from the last `X-Forwarded-For`
hop — believed only from `server.trusted_proxies`, loopback by default — and no
longer treats loopback as "local", so the dashboard needs `admin_token`.

**The Pi's own deck** connects locally, `ws://127.0.0.1:23180/ws`, when the hub
listens on `0.0.0.0` or `127.0.0.1` without built-in TLS. Bound to the tailnet
IP, use that IP; with built-in TLS, use the public `wss://` name (the
certificate does not name 127.0.0.1). Always include the port for a `ws://`
address: a bare host means port 80.

## 6. Connect each Flight Deck

Two steps per deck: add it on the hub, then paste its join code into the deck.

### Add it on the hub

Open the hub's dashboard (signing in: [step 9](#9-watch-the-network-the-hub-dashboard)),
go to **Decks** and press **Add Flight Deck**:

* **Display name** — what people see: on the hub's map and Decks page, in
  every Flight Deck's **Network** card, and in the list your agents get
  (*Flight Deck 'pi' (Kitchen Pi)*).
* **Deck name** — the deck's address on the network: other decks reach its
  agents as `Agent@pi`. Letters, digits, `-`, `_`, `.`, up to 63 characters.
  It is suggested from the display name and cannot change later.
* **Description** (optional) — who runs it, what is on it, what it is for.

**Create key** shows what the deck needs, **once**:

* the **join code**: address, deck name, key and secret in a single line;
* the **hub address** the deck should use. It is taken from the page's own
  address: change it when the deck reaches the hub another way (the dashboard
  open on `127.0.0.1` on the Pi, but a VPS dialling the tailnet IP or your
  `wss://` name). The join code follows the change.
* the same, field by field, and as `FD_BOTPORT_*` variables for a headless deck.

The hub keeps only a fingerprint of the secret: copy the join code before you
close the dialog (lost it? **Rotate**, below). The join code contains the
secret, so pass it on the way you would a password. The dialog stays open and
turns green the moment the deck connects.

### Paste it into the deck

Sign in to the Flight Deck as an **admin**, open the Admin console (the shield
button, *Admin console — models, archetypes, connections, users, settings*) and
choose **BotPort** in its left menu — or go straight to
`http://<deck>:25080/console/network`.

In the **Connection** card, paste the join code into **Join code**: the
address, deck name, key and secret fill in and **Connect to BotPort** switches
on. Press **Save**. That's all.

**By hand** (keys from `botport gen-key`, or to check the fields), in the same card:

1. **Hub address** — the address from step 5:

   | Deck | Hub address |
   |---|---|
   | pi (the hub's own machine) | `ws://127.0.0.1:23180/ws` |
   | vps1, vps2 | `ws://<pi-tailnet-ip>:23180/ws` (or your `wss://` name) |

   `host:port`, `http(s)://` and `ws(s)://` forms are accepted; `/ws` is added
   when missing.
2. **Test** — checks the typed address without connecting as the deck.
   Expected: *BotPort v0.8.4 reachable (protocol 2)*.
3. **Deck name** — the key's `instance` (`pi`, `vps1`, …): letters, digits,
   `-`, `_`, `.`, up to 63 characters, no spaces or `@`.
4. **Key** and **Secret** — from the hub's dialog, or that deck's `gen-key`
   output. The secret is write-only and stored encrypted.
5. Switch on **Connect to BotPort**. Leave **Let this deck's agents consult
   other decks** on (the default).
6. **Save**.

The **Status** card then reads **Connected**, *This deck on the network* shows
the name the hub confirmed, and the **Network** card lists every other deck,
by its display name and description, with the agents it shares. On a problem
it shows *Retrying in Ns* or *Error* with the reason (see
[Troubleshooting](#troubleshooting)).

### Manage the decks

The hub's **Decks** page lists every deck it knows: added there, keyed in the
config file, or (on a hub with auth off) seen connecting. Each card shows the
deck's display name and description, whether it is online (and since when),
waiting for its first connection or offline, the agents it shares, its load
and address, and its key (*Dashboard key*, *botport.yaml*, or *open hub*).

* **Edit** — the display name and description. The hub map and every deck's
  Network card pick the change up within seconds. Works for decks keyed in the
  config file too.
* **Rotate** — a new secret for the same key. The old one stops working at
  once and the deck is disconnected; its Status card says *this deck's secret
  was replaced on the hub*. Paste the new join code there and **Save**. Use it
  when a secret may have leaked or was lost.
* **Remove** — the key stops working, the deck is disconnected, requests it was
  handling fail, and its name and description are forgotten. To bring it back,
  add it again.

A key from the config file is changed there (and the hub restarted); its card
offers **Edit** only.

> Two other BotPort settings in Flight Deck do **not** connect the deck: the
> sidebar's *Connection settings → BotPort Address* (only the browser's link to
> a hub dashboard) and the Spawner's *Direct BotPort link (legacy)*.

**A deck with sign-in off** (`FD_AUTH_ENABLED=false`) takes BotPort settings
only from its own machine. From elsewhere, tunnel to it —
`ssh -L 35080:127.0.0.1:25080 <user>@<deck>` and open
`http://127.0.0.1:35080/console/network` — or use the environment variables
below. Internet-facing decks should run with sign-in on (see
[team-deployment.md](team-deployment.md)).

**Headless decks** can be configured by environment instead (Flight Deck reads
them at start; a field set this way shows *Set by FD_BOTPORT_… in the
environment* and cannot be changed in the UI). For Docker, put them under
`environment:` in `docker-compose.flight-deck.yml`.

| Variable | Meaning |
|---|---|
| `FD_BOTPORT_URL` | hub address (setting it also switches the link on) |
| `FD_BOTPORT_DECK_NAME` | this deck's name |
| `FD_BOTPORT_KEY`, `FD_BOTPORT_SECRET` | credentials from the hub's Decks page (or `botport gen-key`) |
| `FD_BOTPORT_EXPOSE` | comma-separated agent names/slugs, or `*` |
| `FD_BOTPORT_ACCEPT_FROM` | comma-separated deck names, or `*` |
| `FD_BOTPORT_ALLOW_OUTBOUND` | `true` / `false` — this deck's agents may consult other decks |
| `FD_BOTPORT_MAX_CONCURRENT` | remote tasks run at once (default 4) |
| `FD_BOTPORT_DENY_TOOLS` | extra tools remote requests may not use |
| `FD_BOTPORT_USER_SHARING` | `false` to stop users sharing their own agents |
| `FD_BOTPORT_ENABLED` | `false` to switch the link off |
| `FD_BOTPORT_NO_PROXY` | `1` to ignore `HTTP(S)_PROXY` for the hub connection |

## 7. Choose what each deck shares

Nothing is shared until someone picks it. Two ways to put an agent on the
network, side by side:

**The admin** — Admin console → BotPort → **Exposed agents**:

| Setting | Recommended start | Notes |
|---|---|---|
| **Expose every agent** | Off; tick the agents you want | With sign-in on, "every agent" covers only agents someone owns. Flight Deck's own workers are never exposed. |
| **Let users share their own agents** | On (the default) | Users put their own agents on the network themselves (below). Off: only the admin's list counts; users' shares are kept but don't count. |
| **Accept requests from** | **Only these decks**, listing the others (on `pi`: `vps1, vps2`) | Deck names are verified by the hub. An empty list lets no deck in. Default: *Any deck on the hub*. |
| **Remote tasks at once** | `4` | 1–64 (the hub caps a deck at 10). Beyond it, senders get "busy — retry". |
| **Longest remote task (seconds)** | `900` | The limit is the smaller of this and what the asker requests (agents ask for 480 s to consult, 600 s to delegate). |
| **Tools remote requests may not use** | `shell, scripts, write, edit` | On top of the tools a remote request never gets (see [Safeguards](#what-a-remote-request-can-and-cannot-do)). |

Each agent shared by its owner shows **shared by owner** in that list, with a
**Stop** button to take it off the network. (An admin offers other people's
agents through this list; only an agent's owner shares it from its card.)

**Users** — on their own agent: in the kiosk, the agent's **⋯** menu →
**Share on network…**; in the Agent Desktop, the card's actions menu (the
small gear) → **Share on network…**. A dialog explains what sharing means
(who can ask it, what stays out of reach, what it can still draw on, and that
it counts toward the deck's usage) and lists the decks on the network now.
Confirm, and the agent shows a
**network** badge while it is on the network; **Stop sharing on network** (same
menu) takes it off. The item appears only when the deck is connected to a hub
and the admin lets users share. A share
belongs to the agent's owner: if the agent changes hands it lapses until the
new owner shares it, and removing the agent (or replacing it with a clone or
a new container of the same name) removes its share.

An agent must be **running** to take requests (other decks see a stopped one
struck through). Every Save in the settings briefly reconnects the link, so
don't save while a remote task is running. Other decks see changes within
about 30 seconds.

## 8. Use it

You reach a remote agent **through one of your local agents**: ask it in chat.

| Goal | Say to a local agent |
|---|---|
| See who is reachable | *List all agents, including the ones on other Flight Decks* |
| A quick question, wait for the answer | *Ask Researcher@pi: … — wait for the answer* (consult) |
| Bigger work, answer later | *Delegate to Researcher@pi: …* (delegate) |

Use the full `Agent@deck` name (case doesn't matter; a bare name matches local
agents first). If the agent picks the wrong action, be explicit: *Use the
flight_deck tool to consult Researcher@pi with: …*. Files do not cross decks
yet: paste the content into the message.

Agents see remote agents through the `flight_deck` tool:

```
flight_deck list_agents
  Fleet agents (2): …
  Agents on other Flight Decks (2) — consult/delegate them by their full Agent@deck name:
  - Researcher@pi [remote · Flight Deck 'pi' via BotPort] — Finds things
  - Coder@vps2 [remote · Flight Deck 'vps2' via BotPort]
```

* **consult** waits for the answer, with the remote agent's progress live in
  the chat; your agent then answers you.
* **delegate** returns at once; the result arrives later in the agent's main
  chat as *[Delegated result from Researcher@pi] …* and the agent passes it on.

Where to watch it:

* **Asking deck** — Admin console → BotPort → **Activity**: *Waiting on other
  decks*, then *Recent* (ok / failed / ended).
* **Receiving deck** — the same page: *Running here for other decks*. The
  request runs in a lane of its own (`BP…`), not in the agent's chat tabs.
* **The hub** — the dashboard (next step) and `scripts/botport-tmux.sh logs`
  (`Concern … routed`, `Result relayed`).

## 9. Watch the network: the hub dashboard

Open `http://<hub>:23180/` (from the Pi: `http://127.0.0.1:23180/`). With an
`admin_token` it asks for the token once and remembers it in that browser; the
lock button in its top bar forgets it again. You can also pass it once in the
address, `http://<hub>:23180/?token=<admin-token>` — it is stored and removed
from the address bar.

* **Network** — the hub in the middle, every Flight Deck around it with its
  shared agents, load and status. Each request is drawn as it travels (asker →
  hub → agent); progress and the answer come back the same way, failures in
  red. A deck that is working glows amber; one that is asking, cyan. Around it:
  decks online, shared agents, requests in flight, the last 24 hours, the
  success rate and the median answer time; the requests in flight with each
  agent's latest step; recent results; and a live ticker.
* **Requests** — everything recent, filterable by state and deck, searchable.
* **Decks** — add, name, rotate and remove Flight Decks ([step 6](#6-connect-each-flight-deck)).
* Click a deck or a request for its details (Esc or the × closes them). A
  deck's details show its description, with a link to edit it.
* **Swarm** — the hub's multi-agent workflows.

**Kiosk mode** turns it into a wall display: full screen, scaled to the
screen, cursor hidden when idle. Toggle it with the ⛶ button or **K**, or open
`http://<hub>:23180/?kiosk=1`. On the Pi with a screen attached, for example
(Raspberry Pi OS may call the browser `chromium-browser`):

```bash
chromium --kiosk --noerrdialogs --disable-infobars "http://127.0.0.1:23180/?kiosk=1&token=<admin-token>"
```

For scripts: `/healthz` (open), `/api/status` (decks, agents, load, counters),
`/api/overview` (what the dashboard draws) and `/api/decks` (the Decks page:
`GET` lists, `POST {"name", "label", "description"}` adds and returns the
credentials, `PATCH /api/decks/<name>` edits, `POST /api/decks/<name>/rotate`,
`DELETE /api/decks/<name>`), all with `Authorization: Bearer <admin-token>`;
changes must be sent as JSON.

## 10. First test across your decks

1. On the hub: `scripts/botport-tmux.sh status` → *up* and `"protocol": 2`.
2. From each remote deck's machine: `curl -s http://<pi-tailnet-ip>:23180/healthz`.
3. Each deck's **Status** reads **Connected**; the hub dashboard shows every
   deck online.
4. Share one running agent per deck. Within ~30 s each deck's **Network** card
   lists the others with their `Agent@deck` names (a deck never lists itself).
5. From an agent on `vps1`, *list all agents, including the ones on other
   Flight Decks*.
6. Consult in every direction: `vps1 → pi`, `pi → vps1`, `vps1 → vps2` (VPS to
   VPS through the hub). Watch the dashboard draw each one.
7. Delegate `vps2 → <Agent>@pi`; the result lands in the main chat of the `vps2`
   agent.
8. Optional: on `pi`, set *Accept requests from* to `vps1` only and consult
   `pi` from `vps2` — expect *Flight Deck 'pi' does not accept requests from
   'vps2'*. Put it back.
9. Restart the hub (`scripts/botport-tmux.sh restart`) while nothing runs: the
   decks show *Retrying…*, then **Connected** again within seconds.
10. Restart one deck: the others show it offline, then it rejoins and its
    agents reappear.

## What a remote request can and cannot do

* It runs in **its own lane** of the target agent (`BP` + 80 bits of a hash of
  the asking deck's hub-verified name): never in the owner's main chat, never
  in another deck's history. The lane sees no owner profile, context packs,
  topic recall or members' data.
* It can **never write or send email**, never triggers the agent's Flows, and
  its reply goes only back to the requester (not to the agent's channels). An
  answer that drew on members' private data is not sent off the deck.
* It never gets the network tools (`flight_deck`, `consult_peer`, `botport` —
  no A → B → A loops, no fan-out on someone else's behalf) or the tools bound
  to the owner's own session and memory (`cron`, `sessions`, `history`,
  `topics`, `personality`, `terminal`, `insights`, `intentions`,
  `shared_agent_usage`) — the agent itself enforces this for every `BP` lane —
  plus whatever the admin denies (for an agent other people's decks use,
  consider `shell, scripts, write, edit`).
* It still answers **with the agent's own tools and data**: its files, its
  datastore, and accounts connected to it (an inbox, for instance — reading
  mail is not blocked, only writing). Share an agent only when its owner is
  comfortable with other decks getting answers drawn from those, or deny the
  tools in question (e.g. `read, datastore, google_mail, microsoft_mail`) for remote requests. The
  "Share on network" dialog tells users exactly this.
* Only Flight Decks are identified by the hub. A legacy agent (or a hub swarm)
  asking is let in only when *Accept requests from* is *Any deck*, and gets a
  lane of its own.
* File attachments do not cross decks yet.

## Reliability — what happens when things drop

| Event | What happens |
|---|---|
| A deck's connection drops | It reconnects by itself: about 1 s after a drop that ended a healthy session; repeated failures back off up to 60 s (with jitter). |
| A request's acknowledgement is lost in a drop | The deck reports it as not-yet-acknowledged and resubmits it; the hub keeps the work running. |
| The **asking** deck drops mid-task | The answer waits on the hub (`result_retention_seconds`, 15 min) and is delivered when it reconnects. |
| The **working** deck drops mid-task | The task keeps running there; the hub waits `reconnect_grace_seconds` (90 s); the result is sent after the reconnect. |
| The working deck **restarts** | The hub sees the task is gone and tells the requester at once. |
| A request is lost in a blip | Submission is idempotent: the deck resubmits; a dispatch never acknowledged is re-sent. |
| The hub restarts | Decks reconnect; requests that were in flight fail with "the hub restarted — retry". |
| No progress for `idle_timeout_seconds` | Both sides are told; the worker stops. |
| The requester gives up | The worker is told to stop and hangs up on its agent. |
| Two Flight Deck processes share one deck identity | The one taken over says so ("took over the link (code 4000)") and waits 5 min instead of taking the link back. |

## Troubleshooting

**Connecting a deck** (its Status card, or `scripts/botport-tmux.sh logs` on the hub):

| Message | Cause | Fix |
|---|---|---|
| `hub refused registration: Authentication failed` | Wrong **Key** or **Secret** | Paste the join code again (or enter them again), **Save**, then **Reconnect** (after this refusal the link otherwise waits up to 10 min). |
| `… Authentication failed: this deck's secret was replaced on the hub …` | Someone pressed **Rotate** for this deck on the hub | Paste the new join code, **Save**. |
| `… Authentication failed: this key was removed on the hub` | The deck was removed on the hub's Decks page | Add it there again and paste the new join code. |
| `… Too many failed registrations from this address; try again later` | 10 failures in a minute for this key from this address: blocked for 5 min | Wait, restart the hub, or **Rotate** the deck's key on the dashboard (that clears the block). |
| ``… This key is not bound to a deck name: set `instance:` …`` | The key has no `instance:` | Add `instance: <deck>` in the hub's `auth.keys`, restart the hub. |
| `… A different Flight Deck is already connected as 'x'` | Two decks use one name or key — or a deck's data directory was reset while its old connection is still open | Use unique names; a dead connection drops within about a minute. |
| `another Flight Deck process with this deck's identity took over the link (code 4000) …` | Two Flight Deck processes share one data directory | Stop the duplicate. |
| `the hub speaks BotPort protocol 1; Flight Deck links need protocol 2 …` | The hub is older than the decks | Upgrade BotPort on the hub, then **Reconnect**. |
| `hub answered 404 — is the URL path right (…/ws)?` | Wrong path | The address ends in `/ws`. |
| `hub answered HTTP <code> instead of a WebSocket upgrade` / TLS errors | A proxy or tunnel does not pass WebSockets, or a certificate problem | Fix the proxy / tunnel. |
| `connection refused — is BotPort running and the port open?` | Hub down, wrong port, firewall | `scripts/botport-tmux.sh status` on the hub. |
| `timed out connecting to the hub` / `network error: …` | VPN down, firewall | `tailscale status`; `curl …/healthz` from the deck's machine. |
| **Test** says `no /healthz there — not BotPort, or a BotPort older than protocol 2` | Old hub, or the wrong port | Upgrade / fix the address. |
| **Test** passes but the link fails (or the reverse) on a host with `HTTP(S)_PROXY` | The link uses the proxy, **Test** does not | Set `FD_BOTPORT_NO_PROXY=1` and restart Flight Deck. |
| `BotPort settings can only be changed from this machine while Flight Deck sign-in is off …` | Sign-in off, browser not on the deck | SSH tunnel (step 6), environment variables, or turn sign-in on. |

**Using it** (what an agent reports):

| Message | Cause | Fix |
|---|---|---|
| `Agent 'X' not found in fleet. Available: …` | Target deck offline; agent not shared or not running; this deck's link down or *consult other decks* off; or the bare name was used | Check this deck's **Network** card; use `X@deck`. |
| `'X' is ambiguous — it runs on several Flight Decks …` | The name exists on several decks | Use `X@deck`. |
| `Flight Deck 'b' does not accept requests from 'a'` | The target deck's *Accept requests from* | Add `a` there. |
| `No exposed agent '…' on this deck` | Not shared | Share it (step 7). |
| `'…' is not running right now` | The agent is stopped | Start it. |
| `this deck is busy (N remote tasks running) — retry later` | *Remote tasks at once* reached | Retry, or raise the limit. |
| `Timed out waiting for … ` / `… did not finish in time` | Too long for a consult | Delegate instead; check *Longest remote task*. |
| `BotPort no longer knows this request (the hub restarted). Please retry.` | The hub restarted | Retry. |
| `'…' restarted and lost this task` / `… went offline and did not reconnect within 90s` | The working deck restarted or dropped | Retry when it is back. |
| `files cannot be attached to a remote (other Flight Deck) agent yet …` | A file was attached | Paste the content instead. |
| `<name>'s answer drew on private data that cannot leave this Flight Deck` | The answer used members' private data | Ask without that data. |
| No *Agents on other Flight Decks* in `list_agents`, though everything is connected | The asking agent runs old code | Restart it (process) or rebuild it (Docker). |

**Sharing**:

| Message | Fix |
|---|---|
| `Sharing by users is turned off on this Flight Deck` | The admin switched off *Let users share their own agents*; the admin's *Exposed agents* list still works. |
| `This Flight Deck is not connected to a BotPort network` | Connect the deck first (step 6). |
| `Only the agent's owner can share it on the network` | Ask its owner — or, as admin, tick it under *Exposed agents*. |
| `Only the agent's owner (or an admin) can stop sharing it` | Ask its owner or an admin. |

**The hub**:

| Symptom | Fix |
|---|---|
| `Cannot listen on 0.0.0.0:23180 — … Is another BotPort already running?` | An old hub still holds the port: find it (`ss -ltnp \| grep 23180`) and stop it. |
| `error: config check failed …` | Run `botport check -c ~/.botport/config.yaml` and fix each `ERROR:`. |
| `WARNING auth.keys is empty: only Flight Decks added on the dashboard (Decks) can connect` | Expected when you add decks on the dashboard. With the dashboard off it says so, and you need `gen-key` entries. |
| **Join code** field says *That isn't a whole join code* | It was cut off when copied: press **Copy join code** again (it starts with `bpj1.`), or enter the fields by hand. A join code that was shown once and lost can't be shown again: **Rotate**. |
| `WARNING: … readable by other users (chmod 600 …)` | `chmod 600 ~/.botport/config.yaml` |
| Banner shows port 33080 or `Auth: DISABLED (open)` | The hub was started without `-c`: use the tmux script. |
| Dashboard / API answer 401 | Enter the admin token (or `Authorization: Bearer …`); without `admin_token` set, the API answers only on the hub's own machine, opened as `http://127.0.0.1:23180/` or `http://localhost:23180/` (not by another name: that is what a DNS-rebinding page would use). |
| Decks page: *Two keys for one deck* | A deck has a key in botport.yaml and one made on the dashboard; both connect as it. Remove one: the dashboard's with **Remove**, the file's by deleting the entry and restarting. |
| **Add Flight Deck** says *Turn on auth first* | The hub runs with auth off, where keys change nothing. Set `auth.enabled: true` and restart once. |

## Upgrading

Upgrade **the hub first**: a newer Flight Deck refuses an older hub (protocol
check) and waits before retrying.

* **Hub** — `cd ~/captain-claw && git pull`, then
  `~/botport-venv/bin/pip install ~/captain-claw/botport` and
  `scripts/botport-tmux.sh restart`. Decks added on the dashboard live in
  `botport.db` in the data directory and survive upgrades; back that file up
  with the config. If an older hub was started some other
  way (e.g. from `~/captain-claw/botport/` on port 33080), stop it first
  (`tmux ls`, `ss -ltnp | grep -E ':(23180|33080)'`).
* **Flight Deck (Python install)** — `git pull`, `pip install -e .` (or
  `pip install .`) in its environment, and restart it **from the same directory
  or with the same `FD_DATA_DIR`** (the data directory defaults to `./fd-data`;
  starting elsewhere gives an empty deck with a new identity). The built UI is
  in the repository; no npm build is needed. Restart its process agents too:
  an agent on old code doesn't see remote agents.
* **Flight Deck (Docker)** — rebuild or pull the `kstevica/captain-claw` image,
  recreate the container (settings stay in `./fd-data`), and **Rebuild** the
  Docker agents: they run the code baked into their image.

### From per-agent BotPort links

Before Flight Deck carried BotPort, each agent could connect to the hub on its
own (the Spawner's *Direct BotPort link (legacy)*). That still works — the hub
accepts those connections and decks can address them by name — but remote work
now goes through the deck's link, where the safeguards above apply. New agents
have it off; switch it off on old ones (with sign-in on, such an agent would
need a key of its own, and none may use a deck's name).
