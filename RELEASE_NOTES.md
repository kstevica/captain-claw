# Captain Claw v0.6.2 Release Notes

**Release title:** The remote terminal — text your own machine and watch it work
**Release date:** 2026-06-25

A focused release that gives the agent a **real interactive terminal on a machine you choose** — your Mac, a laptop, a remote box — drivable from **both Flight Deck web chat and WhatsApp**. Unlike the `shell` tool (one command in, output out, no tty), the new **`terminal`** tool holds long-lived pseudo-terminal sessions: it can run REPLs, interactive CLIs (`claude`, `ssh`, `psql`), and full-screen TUIs by typing keystrokes — including control keys — into a live pty and reading back what appears. The machine that actually runs the terminal can sit **behind NAT**: a small daemon dials *out* to Flight Deck and the agent's calls ride back down that connection. Additive and backward compatible with 0.6.1.

---

## What's new

### The `terminal` tool — a live terminal on your own machine

A new agent tool that drives a real PTY on the user's paired machine — explicitly *not* the agent's host. `shell` runs on the server; `terminal` runs on your computer. The tool's description steers the agent to prefer it whenever the user refers to their own machine ("on my Mac", "my laptop", "at home").

- **One-shot or interactive.** `action="run"` executes a single command and returns its output (open → execute → capture → close in one call — as easy as `shell`, but on your machine). For anything that needs a live tty, the session flow: `open` (returns a `session_id`) → `send` text/keys and read what prints back → `send` again → `close`. Sessions persist across turns; reuse the id.
- **Keystroke emulation.** `send` takes free text, a named `key` / chord (`enter`, `tab`, `esc`, arrows, `ctrl-c`, `ctrl-d`, `ctrl-l`, …), and an `enter` flag — so the agent can drive `claude`'s REPL, answer a prompt, or interrupt a runaway with Ctrl-C.
- **Waiting-for-input detection.** When a program parks at a prompt — a menu, `(y/n)`, a `read -p` line, `claude`'s "Do you want to…?" confirm — the tool flags it (output ends with a ⏳ marker) so the agent answers it with `send` instead of assuming the program finished or stalling. A one-shot `run` that hits a prompt bails with guidance to switch to a session, rather than blocking.
- **Readable output.** ANSI escape codes are stripped by default for text channels (WhatsApp / chat); pass `raw=true` to keep them.
- **Connection status.** `action="list"` doubles as a probe — it reports whether your machine is connected (✓ + active sessions), has no daemon running (✗ worker offline), or is unreachable — so the agent can tell "your Mac is offline" apart from "I don't have this tool".

Because the agent is already reachable from Flight Deck web chat and WhatsApp, the same terminal is drivable from **both channels with no new Flight Deck transport**.

### The PTY daemon — your machine's side

A standalone daemon (`python -m captain_claw.terminal.daemon`) owns the pseudo-terminal sessions. It's deliberately decoupled from the agent, so sessions **survive agent restarts** (which happen on every code change).

- **Default working directory** is the folder the daemon was launched from, so terminals open where you're working.
- **Live mirror.** PTY output (plus a session header with cmd + cwd and an exit-code footer) is echoed to the daemon's own console, so you can **watch sessions happen in real time** in the daemon window. On by default; `CLAW_PTY_MIRROR=0` to silence.
- **Login-shell semantics.** A command string runs as `$SHELL -lc "<cmd>"`, so shell syntax (`cd`, `&&`, pipes, env), your login `PATH`, and interactive programs all work; pass an explicit argv list when you want literal, no-shell execution.

### Two ways to connect — reachable or behind NAT

The tool ↔ daemon link is plain HTTP, so the daemon can live anywhere the agent can reach. Pick by network reality:

- **Reachable network** (same LAN, VPN, Tailscale, or a public box): the daemon binds a port (`CLAW_PTY_HOST` / `CLAW_PTY_PORT`) and the agent points `CLAW_PTY_URL` at it. The daemon **refuses a non-loopback bind without a token** unless `CLAW_PTY_INSECURE=1`.
- **Behind NAT (dial-out)** — the headline path: the daemon sets `CLAW_PTY_RELAY` and opens a persistent **outbound WebSocket** to Flight Deck, registering as a named `CLAW_PTY_WORKER`. A new **Flight Deck relay** (`/fd/pty/connect` for daemons, `/fd/pty/{worker}/{op}` for the agent) tunnels each tool call down the worker's socket. **No inbound ports** on your machine; sessions survive a dropped socket and reconnect with backoff.

The dial-out client uses the `websockets` library rather than aiohttp specifically so it tolerates the duplicate handshake headers reverse proxies (Cloudflare/nginx) add in front of Flight Deck.

---

## Security

A PTY session is arbitrary code execution on the target machine, so the bridge is gated:

- **Shared token.** One `CLAW_PTY_TOKEN` authenticates all three legs — the daemon's register frame, the relay's HTTP, and the tool's `X-Claw-Token` header. Set it on Flight Deck (its environment, inherited by spawned agents) and on the daemon's launch command.
- **Keep it private.** Run the dial-out over `wss://` (TLS) and/or a private network (Tailscale/WireGuard/SSH tunnel). The daemon will not expose a raw RCE port to the network without a token.
- **Known limitation.** The `terminal` tool does **not** yet run through `shell`'s ask/allow/deny approval gate (those checks are keyed to the `shell` tool). For now, the token + private transport are the fence — a sender allowlist / approval gate is a planned follow-up. Keep the daemon paired to a machine you control and a token you trust.

---

## Migration & compatibility

Additive — **backward compatible with 0.6.1**. Nothing changes unless you opt in.

- **Enable the tool.** Add `terminal` to `tools.enabled`. On a deployment with a `~/.captain-claw/config.yaml` overlay (written by the Flight Deck settings UI), add it there too — the home overlay's `tools.enabled` list **replaces** the project `config.yaml` list, it doesn't merge. Then restart the agent.
- **Run the daemon** on the machine you want to drive (`python -m captain_claw.terminal.daemon`), with the env for your chosen mode. New env vars are documented in `.env.example`: `CLAW_PTY_TOKEN`, `CLAW_PTY_URL`, plus daemon-side `CLAW_PTY_HOST` / `CLAW_PTY_PORT` / `CLAW_PTY_RELAY` / `CLAW_PTY_WORKER` / `CLAW_PTY_MIRROR` / `CLAW_PTY_INSECURE`.
- **Flight Deck** gains the relay router automatically (`/fd/pty/*`); restart FD to pick it up. The relay's worker registry is in-process — run Flight Deck single-process, or add sticky/shared state if you run multiple uvicorn workers.
- No databases, no schema changes, no removed endpoints.

See [release-notes/RELEASE_NOTES_0.6.1.md](release-notes/RELEASE_NOTES_0.6.1.md) for the previous release, or the [release-notes/](release-notes/) folder for the full history.
