# Agent OS sandbox — Landlock + seccomp around every process agent

**Status:** plan. The sandbox itself is not implemented. The containment layer it builds
on is PRs #255 and #258 (`captain_claw/deck_confine.py` and the Flight Deck routes),
merged 2026-10-27, not deployed yet.
**Decision (2026-10-27):** for now the glasses deck stays on the #255/#258 containment —
file confinement and no shell for non-admins — and the OS sandbox is parked. The only
part that would need a server change there is closing the signal gap (§6: an HWE-kernel
reboot, or installing bubblewrap). The Landlock + seccomp launcher (§1) needs no install,
no root and no reboot — the glasses trial below ran it as-is — so it can be added later
without touching the server, with the signal gap left open (it matters only once
non-admins have a shell again).
**Origin:** on the glasses deck a Fil Rouge Capital Deal Screener agent globbed other
users' VFS folders and other agents' workspaces, and its shell listed the deck's own
databases. Every process agent runs as the deck's OS user, so the OS never said no.

---

## Where #255 leaves us

#255 is a check inside the agent's Python. It holds every path argument of every tool
to the agent's own roots, and on a sign-in deck it takes code-running tools (`shell`,
`scripts`, `code`, `terminal`, …) away from agents whose owner isn't an admin.

What it can't do:

- **An admin's agent still has a full shell** as the deck's OS user: every file the deck
  user can read, every same-user process and socket, and — on glasses — Docker, which is
  root on the host (the deck user is in the `docker` group).
- **Non-admins have no shell at all.** Code mode, Vatra/Basna workers spawned for them
  and many research tasks are worse for it.
- **It's only as good as the rules.** A tool whose path parameter nobody ruled on, a
  library that opens a file by itself, a bug in the resolver — each is a way out.

The OS sandbox puts a wall the kernel enforces around the whole agent process and
everything it starts. Then a shell is safe to hand back, and #255 becomes the second
layer that gives clear refusal messages.

---

## Fix first: paths around #255 that don't need a shell — DONE in PR #258

The audit for this plan found paths around #255 that go through Flight Deck itself, so no
agent sandbox would stop them. They are fixed in PR #258 (`fix/agent-route-callers`, open,
not deployed): agent routes bind a caller to an owner only from a bearer or the agent's
own `web_auth` on a sign-in deck; hosting `kind='app'` publish/start is admin-only; and a
`user`-tier agent's `browser`/`direct_api`/`apis`/`web_fetch` refuse loopback and private
addresses. The detail below is kept as the record of what was closed.

1. **`hosting` publishes and starts an app.** `kind='app'` makes FD run the agent's
   `start_cmd` with `shell=True`, as the deck user, with FD's environment minus
   `_FD_ONLY_ENV_VARS` (`flight_deck/vfs_hosting.py` `start_app`). `hosting` is not in
   `deck_confine.EXEC_TOOLS`, so a non-admin's agent can still run any command through
   it.
2. **Any agent can make FD's agent routes act as someone else.**
   `code_routes._resolve_agent_caller` (used by `/fd/code/agent/*` and
   `/fd/hosting/agent/*`) falls back to the body's `owner_id` when neither `web_auth` nor
   `source_port` resolves — on sign-in decks too. The transport guard lets in any
   loopback caller (`server._agent_caller_ok`). `basna_routes._resolve_owner` (Basna,
   Vatra, Bat) already refuses a bare `owner_id` on sign-in decks, but all of them still
   accept `source_port` on the caller's word — and every agent's port is visible in
   `/proc/<pid>/cmdline`.
3. **Agents can reach those routes without a shell.** `direct_api` is always enabled
   (`config.ToolsConfig._ALWAYS_ENABLED`) and calls any URL with GET/POST/PUT/PATCH,
   loopback included. That also reaches Caddy's admin API on `127.0.0.1:2019` and the PTY
   daemon.

The fix, one PR:

- FD: on a sign-in deck, an agent route resolves its caller **only** from a bearer or the
  agent's `web_auth` token; `source_port` and `owner_id` become consistency hints, as
  `owner_id` already is in Basna. (Check that every agent-side caller sends `web_auth`.)
- FD: `hosting` `kind='app'` (publish with `auto_start`, `start`) is admin-only on a
  sign-in deck until apps run inside the same sandbox as agents. Static sites stay open.
- Agent: `deck_confine` refuses `hosting` app actions for the `user` tier, with the same
  message as the other code-running tools.
- Agent: for the `user` tier, `direct_api`, `apis`, `web_fetch` and `browser` refuse
  loopback, link-local and private addresses (resolved, not just the literal host).

---

## The glasses trial (2026-10-27)

### The host

| | |
|---|---|
| OS / kernel | Ubuntu 24.04.3, `6.8.0-139-generic`, x86_64 |
| bubblewrap | **not installed** |
| User namespaces | `kernel.apparmor_restrict_unprivileged_userns = 1` — `unshare -Ur` fails writing `uid_map`; bwrap would fail the same way |
| Root | `kstevica` is in `sudo`, but sudo needs a password |
| Landlock | in the LSM list; **ABI 4** (files + TCP ports) |
| Docker | `kstevica` is in the `docker` group |
| HWE kernel | `linux-generic-hwe-24.04` candidate `7.0.0-38` (not installed) |

So bubblewrap needs two root steps (install it, and an AppArmor profile that lets it
create user namespaces). Landlock needs none: a process restricts itself, and the
restriction is inherited by everything it forks and execs.

### What an agent can reach

A probe ran as the deck user three ways: as agents run today, under Landlock, and under
Landlock plus a seccomp filter. It only opened files for 16 bytes, listed names,
connected and closed, and sent signal 0.

| Target | Today | Landlock | + seccomp |
|---|---|---|---|
| Own agent folder, owner's VFS folder | ✅ | ✅ | ✅ |
| `fd-data/flight-deck.db`, `fd-data/agent_secret` | ✅ | ⛔ | ⛔ |
| Another agent's folder, `fd-data/vfs/` (all users) | ✅ | ⛔ | ⛔ |
| Checkout `.env`, home folder, `~/.ssh`, `/`, host `/tmp` | ✅ | ⛔ | ⛔ |
| `/proc/<FD pid>/environ` | ✅ | ⛔ | ⛔ |
| `/proc/<FD pid>/cmdline` | ✅ | ✅ | ✅ |
| Signal 0 to the FD process | ✅ | ✅ | ✅ |
| TCP: FD `:8765`, Ollama `:11434`, internet `:443` | ✅ | ✅ | ✅ |
| TCP: another agent `:8791`, FRC deck `:27000`, Caddy admin API `:2019` | ✅ | ⛔ | ⛔ |
| UNIX sockets: tmux server, systemd user manager, D-Bus user bus, Docker | ✅ | ✅ | ⛔ |
| `docker ps` from the agent's shell | ✅ | ✅ | ⛔ |
| Import `captain_claw.web_server` + tool registry | ✅ | ✅ (1.1 s) | ✅ |

The tmux, systemd and Docker sockets are each a full escape: `tmux send-keys` into the
deck's pane, `systemd-run --user`, `docker run -v /:/host`. Landlock on 6.8 doesn't
cover `connect()` on a UNIX socket, which is why the seccomp filter is part of the
design, not an extra.

`/proc/<pid>/environ` is refused because Landlock scopes ptrace-style access to its own
domain; command lines stay visible (they hold ports, not secrets).

### A real agent inside it

`captain-claw-web --port 28999` started under Landlock + seccomp, answered HTTP 200
after 2 s, logged no permission errors, and shut down cleanly. The ruleset:

```text
read+exec   /usr /lib /lib64 /bin /sbin /etc /proc /sys
            ~/glasses/.venv-glasses
            ~/glasses/captain-claw/captain_claw      ← the package only
            ~/.cache/huggingface
devices     /dev (read/write files, no create)
read+write  <agent dir>  <owner VFS root>  /dev/shm
connect     8765 443 80 53 11434
bind        28999
env         HOME=<agent dir>/home  TMPDIR=<agent dir>/tmp
```

One trap the trial found: glasses is an **editable install**, so the agent's code is read
from the checkout — which also holds `fd-data/` and `.env`. The sandbox must grant the
package directories (`captain_claw/`, `botport/`), never the checkout root.

---

## Design

### 1. The launcher — `captain_claw/agent_sandbox.py`

FD stops exec'ing `captain-claw-web` directly and runs:

```text
python -I -m captain_claw.agent_sandbox -- captain-claw-web --port N
```

with the ruleset in `CLAW_SANDBOX_SPEC` (JSON). The launcher runs unsandboxed just long
enough to:

1. build the Landlock ruleset at the highest ABI the kernel offers (files, plus TCP
   ports from ABI 4, plus signal / abstract-socket scoping from ABI 6);
2. add the path and port rules (a missing path is skipped, not an error);
3. `PR_SET_NO_NEW_PRIVS`, `landlock_restrict_self`;
4. install the seccomp filter: `socket(AF_UNIX, …)` and `io_uring_*` fail with `EACCES`,
   any non-x86_64/aarch64 syscall ABI kills the process; `socketpair()` stays (asyncio's
   self-pipe needs it);
5. set `CLAW_SANDBOX=landlock:<abi>` and `exec` the agent.

It **fails closed**: if the deck wants a sandbox and any step fails, the agent doesn't
start and FD shows why. No library dependency — the trial launcher is ~120 lines of
`ctypes`.

### 2. The spec — FD computes it at spawn and restart

Next to `_pin_confinement` in `server.py`, from the same inputs (owner, tier, agent dir,
workspace).

- **read + exec:** system dirs; `sys.prefix` and `sys.base_prefix`; each installed
  package dir (`captain_claw`, `botport`) resolved from the module, not the checkout;
  model and browser caches that exist (`~/.cache/huggingface` — litert reads the real
  one, `llm/__init__.py` ~4826; `~/.cache/ms-playwright`); `~/.gitconfig` and
  `~/.ssh/known_hosts` for Code mode's git (`code_routes._git_env`).
- **read + write:** the agent dir; the owner's VFS root (Drive/OneDrive mounts, the
  datastore and Vatra's notes live inside it); a Code-mode workspace (it lives under the
  owner's VFS root); `/dev/shm`; `fd-data/models/vision` for the `cv` tool, or better,
  point `CAPTAIN_CLAW_VISION_MODELS` into the agent dir.
- **shared folders:** FD resolves them per call today (`pack_access.py` → POST
  `/fd/context-packs/agent/vfs` → `<fd-data>/vfs/<publisher>/<project>`). FD can compute
  the same list at spawn from the DB, and already hooks pack changes
  (`server.py` ~8827), so a grant, a revoke or a membership change restarts the agent.
- **shared agents:** a member's call switches the VFS root to that member's
  (`vfs.py` ~128, `speaker.py` ~504). The spec grants every current member's VFS root,
  and a member joining restarts the agent; `deck_confine` keeps members apart inside it.
- **admins only:** their VFS link targets (read from `.vfs-links.json` at spawn; a link
  change restarts the agent) and `tools.read.extra_dirs`, read-only or read-write as the
  link says.
- **never:** the checkout root, `fd-data/` itself, `.processes.json` (every agent's
  `web_auth`), `data_key`, `flight-deck.db`, `agent_secret`.
- **connect:** FD's port; 80, 443, 53; the ports of the agent's configured model
  endpoints (`model.base_url`), BotPort and SMTP server; Ollama 11434 and Typesense 8108
  when configured; plus `FD_AGENT_SANDBOX_PORTS` for a deck's own extras. Not the PTY
  daemon (23190), PinchTab (9867) or Caddy's admin API (2019). MCP goes through FD.
- **bind:** the agent's web port **and the next 199** — `web_server.py` walks up to 200
  ports when its own is taken (`_probe_port_free` binds to test), and a refused bind
  there reads as "taken", so a single-port rule would turn a busy port into a failed
  start.
- `HOME` and `TMPDIR` inside the agent dir. FD sets no `TMPDIR` today, and about 15
  places use `tempfile` (`tools/shell.py` included).

Anything not listed doesn't exist as far as the agent is concerned.

What the no-UNIX-socket rule breaks, all acceptable: git over SSH through `ssh-agent`
(use HTTPS remotes or a per-agent deploy key), the Docker CLI, X11/Wayland tools
(clipboard, desktop control — not on a server deck), and nscd/sssd lookups (glibc falls
back to files; to verify). The agent's own Python opens no UNIX sockets.

### 3. Secrets

The sandbox hides files, not the agent's own environment. Once a user has a shell
(Phase 3) they can print everything their agent holds:

- **FD's whole environment** except the five `_FD_ONLY_ENV_VARS` (`server.py` ~561).
  FD loads its `.env` into its environment at startup, so that includes provider keys,
  `GOOGLE_OAUTH_CLIENT_SECRET`, `SMTP_PASSWORD`, mail-service keys, FD's
  `TELEGRAM_BOT_TOKEN`, `CLAW_BROWSER_CREDENTIAL_KEY`, `CLAW_PTY_TOKEN`,
  `FD_GLASSES_BRIDGE_TOKEN`, the `WHATSAPP_*` keys and the deck-wide
  `FD_AGENT_SHARED_SECRET`.
- **Team-tier keys:** FD swaps `@system` for the real key and writes it as
  `<PROVIDER>_API_KEY` into the agent's `.env` and environment (`server.py` ~2348,
  ~8722).
- **`config.yaml`:** the agent's own `web.auth_token` (fine — it's its own identity) and
  any channel tokens.

So for sandboxed agents the base environment becomes an **allowlist**: what an agent
needs to run and find FD (`PATH`, `LANG`, `HOME`, `TMPDIR`, `FD_URL`, `FD_OWNER_ID`,
`FD_AGENT_SLUG`, `CLAW_*` pins), plus the agent's own `.env` and env_vars, plus keys FD
shares on purpose. Nothing from FD's `.env` reaches an agent unless it's on that list.

Team keys are a product decision: either a user may see the team key their agent uses,
or model calls for team tiers go through an FD proxy and the key never leaves FD.

### 4. Tools

- `shell`, `scripts`, Code mode's git/test runs: child processes of the agent, so they
  inherit the sandbox. They come back for non-admins in Phase 3.
- `terminal` (sends commands to the PTY daemon), `app_runner`, `hosting` apps and
  anything else that asks **another** process to run code escape any agent sandbox. They
  stay admin-only (`hosting` isn't yet — see "Fix first"). Later, FD can start hosted
  apps through the same launcher with the owner's spec.
- `deck_confine` stays on: it catches the same paths earlier, with a message the model
  can act on, and it covers the Mac desktop deck where there is no Landlock.

### 5. Spawned workers

Vatra/Basna workers and child agents are spawned by FD, so they get their own launcher
and spec. Anything an agent forks inherits its domain automatically.

### 6. The gap Landlock ABI 4 leaves: signals

On kernel 6.8 a sandboxed process can still `kill` any same-user process — the deck,
other agents, the tmux server. That's a way to take the deck down, not to read data, but
it's enough to keep non-admins' shell off. Three ways to close it:

| | Root step | Disruption | Notes |
|---|---|---|---|
| **HWE kernel** (Landlock ABI 6+: signal + abstract-socket scoping) | `apt install linux-generic-hwe-24.04` | **reboot** of the whole box | One mechanism, nothing else to maintain |
| **bubblewrap** PID namespace in front of the launcher | install bwrap + an AppArmor profile granting it `userns` | none | Also gives a private `/tmp` and `/proc` |
| **seccomp user-notify supervisor** in FD | none | none | FD approves each signal to a descendant; the most code |

### 7. What stays open: the network

Port rules are host-agnostic (allowing 443 allows every host's 443) and FD's own port
must stay reachable. Still to do after the sandbox:

- audit FD routes that trust a caller for being on loopback;
- replace the one deck-wide agent secret with a per-agent credential, so a sandboxed
  agent can't speak for another one.

---

## Phases

**Done — the "Fix first" PR (#258).** Independent of everything below; needs only an FD
restart and agent restarts.

**Phase 0 — launcher, spec, self-test.** `FD_AGENT_SANDBOX=off|on|auto` (default `off`),
Linux only. FD runs the launcher once at startup on a throwaway command; `auto` turns the
sandbox on when that passes. The agent secret travels in the environment, since the
`agent_secret` file in `fd-data/` is out of reach. Tests: spec building (pure), the
seccomp program, and a Linux-only integration test that runs the probe above.

**Phase 1 — on for every agent on glasses**, admins included. Landlock 6.8 has no audit
log, so file and shell tools log `sandbox_denied` with the path when they get `EACCES`
under `CLAW_SANDBOX`. Fix the spec from what that log shows.

**Phase 2 — secrets.** The environment allowlist from section 3, and the team-key
decision. This has to land before Phase 3, because a shell is what makes the environment
readable.

**Phase 3 — shell back for non-admins.** After the signal gap is closed (section 6),
`deck_confine` drops `shell` and `scripts` from the refused set when `CLAW_SANDBOX`
reports signal scoping or a PID namespace.

**Phase 4 — network.** Loopback-trust audit, per-agent credential.

---

## Decisions for you

0. **The "Fix first" PR (#258):** done — review and merge when ready.
1. **Signals:** HWE kernel + reboot, bubblewrap + AppArmor profile, or neither for now
   (non-admins stay without a shell).
2. **Admins' agents sandboxed too?** Recommended yes — today an admin agent's shell is
   root on glasses through Docker.
3. **Port allowlist:** strict from day one, or files-only first and ports in Phase 1.
4. **Team keys:** visible to the users whose agents use them, or proxied through FD.
5. **cohort001:** run the same probe there before Phase 1 (not tested yet).
