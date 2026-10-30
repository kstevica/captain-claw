# Flight Deck — Team Deployment & Security Checklist

When Flight Deck is reachable by more than one person (a team launch, anything
behind a public URL or shared network), the defaults that are fine for a
single-user machine are not enough. Set the environment variables below on the
Flight Deck backend **before** the first teammate logs in.

## Required environment variables

| Variable | Set to | Why it matters with a second user |
| --- | --- | --- |
| `FD_AUTH_ENABLED` | `true` | Enforces JWT login on every route. Without it FD runs as a synthetic local admin and anyone reaching the port is that admin. |
| `FD_JWT_SECRET` | a long random string (e.g. `openssl rand -hex 32`) | If unset, a new secret is generated on every boot, so **every session's tokens die on restart**. A fixed secret keeps people logged in and makes tokens forgeable only with the secret. |
| `FD_LOCKDOWN` | `1` | Disables the host-filesystem surfaces that make no sense off-machine (`/fd/vfs/browse-fs`, `POST /fd/vfs/links`, the auth-less `/fd/projects/*` router) and makes the agent shared-secret mandatory even from loopback (so a same-host TLS proxy can't launder remote callers into "loopback"). Also flips the refresh cookie to `Secure` by default. |
| `FD_PUBLIC_URL` / `FD_ALLOWED_HOSTS` | your deck's public URL (e.g. `https://claw.example.com`) / its domain(s), comma-separated | Since 0.8.0 Flight Deck refuses any request whose `Host` it doesn't recognise (DNS-rebinding guard). Loopback, IP literals and this machine's hostnames are always allowed; **a deck reached through a domain must list it here**, or the SPA, webhooks, OAuth callbacks and inbound MCP are all refused (the refusal log names the setting). `FD_ALLOWED_HOSTS` takes `.example.com` / `*.example.com` for subdomains. |
| `FD_CORS_ORIGINS` | only extra origins, if any (e.g. a separate frontend domain) | Since 0.8.0 the default is the deck's own origins (no longer `*`), and every request with a foreign `Origin` is refused. Add an origin only for a frontend served from elsewhere; `*` is the explicit legacy opt-out. |
| `FD_AGENT_SHARED_SECRET` | a long random string | Agents authenticate to FD's internal `/fd/basna/agent/*`, `/fd/vatra/agent/*`, `/fd/deliver`, and scheduler routes with this. Without it those routes fall back to loopback-only, which a reverse proxy can spoof. |
| `FD_GLASSES_BRIDGE_TOKEN` | a long random string | Shared secret for the glasses/scheduler bridge. Internal automation (`intentions.py` → scheduler) sends it. Setting it closes the old hole where an unset value let the scheduler accept **every** request. |
| `FD_COOKIE_SECURE` | `1` (optional) | Forces the refresh cookie to `Secure` regardless of `FD_LOCKDOWN`. With `FD_LOCKDOWN=1` this is already the default; set it explicitly if you terminate TLS elsewhere. Use `0` only for local http development. |
| `FD_REGISTRATION_OPEN` | leave **unset** for a closed team; `1` to allow public self-signup | Self-registration is **closed by default** once the first (bootstrap admin) account exists. Teammates get accounts created by an admin on the Admin page. `FD_REGISTRATION_DISABLED=1` forces it closed even if `_OPEN` is set. |

Generate secrets with e.g.:

```bash
openssl rand -hex 32
```

## What the launch hardening changed (0.8.x)

These are code changes already in this build; the env vars above turn them on.

- **Agent proxies are owner-scoped.** Every port-addressed agent proxy
  (`/fd/agent-*/{host}/{port}`, `/fd/orchestrator/{host}/{port}`, and the
  `agent-ws` WebSocket) now refuses a caller who is not the agent's owner
  (admins may reach any). Previously any logged-in teammate could iterate ports
  and read another user's agent data (memory, files, tool credentials).
- **Flows and the scheduler require authentication.** Flow CRUD/lifecycle routes
  require a logged-in user; scheduler routes require a logged-in user, the agent
  shared secret, the glasses token, or a loopback caller — never "anyone",
  which is what an unset `FD_GLASSES_BRIDGE_TOKEN` used to allow.
- **Org provider keys never reach the browser.** `GET /fd/settings/provider-keys`
  returns only which providers are configured plus a last-4 hint. The Spawner
  offers "use the system key" by sending a `@system` sentinel that FD resolves
  server-side at spawn time.
- **Registration is closed by default**, `POST /fd/google/config` is admin-only,
  the refresh cookie is `Secure` behind TLS, and `POST /fd/deliver` requires a
  trusted internal caller (loopback / agent secret) or a logged-in user.
- **Agents stay in their own files** (0.8.4). On a deck with sign-in, every
  process agent's file tools are checked by real path against its folder, its
  workspace, its owner's VFS and the call's shared folders. `.env` is out of
  reach, control files can't be written, and `glob` and `grep` results are
  filtered. `FD_AGENT_CONFINE=off` turns this off for a deck whose users all
  trust each other. A role change applies at the agent's next start.
- **Non-admins' agents don't run code or drive the host** (0.8.4). `shell`,
  `terminal`, `scripts`, `app_runner`, `code`, `termux`, `bat`, `desktop_action`,
  `screen_capture` and `pinchtab` are refused and hidden; skills install no
  dependencies; `browser`, `direct_api`, `apis` and `web_fetch` take public
  addresses only. Admins keep those. This is containment, not an OS sandbox: an
  admin's agent still has a shell that reaches the whole host
  ([sandbox plan](agent-os-sandbox-plan.md)).
- **Hosted apps and host folders are admin-only** (0.8.4). Publishing or starting
  a hosted app runs a command on the server; static sites stay open to everyone.
  `POST /fd/vfs/links` and `/fd/vfs/browse-fs` are admin-only with sign-in on.
- **Team keys stay where the team sends them** (0.8.4). `GET /fd/agent-config`
  shows non-admins a team key or any secret-named value as `····` and its last
  four; saving a masked value keeps the key. A provider's team key, and a key in
  Flight Deck's own environment, goes only to the provider's own endpoint or one
  the published team sets route it through. Spawning or changing an agent so that
  it would send one elsewhere is re-keyed or refused.
- **Pinned variables** (0.8.4). The variables that would have a non-admin's agent
  load code from a path or route its calls elsewhere (`PATH`, `PYTHON*`, `LD_*`,
  `HTTP_PROXY`, CA bundles and more) keep Flight Deck's values. Adding or changing
  one in the agent editor answers 400, naming it, and writes nothing.
- **Callers prove who they are** (0.8.4). Agent routes bind an owner only from a
  verified bearer or the agent's own token; a `source_port` or `owner_id` in a
  request counts as identity only with sign-in off.

## First-run checklist

1. Set the environment variables above and start the backend.
2. Register the **first** account — it becomes the admin automatically and opens
   in the **Admin console** (`<deck>/console`), whose Overview carries a setup
   checklist.
3. Add people (**Console → Users**, or the Users page): **Invite by email** —
   sent from your own connected Gmail or Outlook; they choose their password
   from a one-time link — or **Add user** with a temporary password they must
   change at their first sign-in. With sign-up closed, *No account?* leads to a
   waitlist; approving an application can email the applicant their sign-in
   link.
4. Set the org **provider API keys** (Admin page, or Console → Models → team
   API keys). These stay server-side.
5. Publish a **team tier set** (Library → tier sets → *Publish to team*, or
   Console → Models) so teammates run models without pasting keys.
6. Check the **daily plan limits** (Console → Usage & limits, or Admin → Plans):
   since 0.8.1 every non-admin is counted per UTC day — by default **free** 30
   messages and 300,000 tokens, **premium** 300 and 3,000,000, **enterprise**
   unlimited. Put each user on the right plan; grant free resets if someone
   runs out.
7. Decide who gets the **kiosk** (Admin → Settings → Kiosk mode: Off,
   Everyone, Selected users) and set the deck's **branding** (instance name,
   logo, description).
8. Ask each teammate to connect their **own** Google or Microsoft account
   (Connections). For the M5Stack devices and voice notes, share a **Soniox**
   team key (Admin → Integrations) or let each person add their own.

## Collaboration features (0.8.x)

- **Shareable tier sets.** Library → configure a tier set → *Publish to team*
  (admin). Teammates who never set up models automatically run on it. Keys
  never reach the browser — each tier stores a `@system` sentinel resolved
  server-side — but **publishing does share them as team keys**: a key on a
  provider's own endpoint becomes (or replaces) that provider's org key, and a
  custom-endpoint (gateway) key goes into a per-endpoint org store. The publish
  answer lists what was added, updated, left unshared or missing. Process agents
  with no model key get the team key in their `.env` and are restarted
  automatically; a spawn nothing can supply a key for is refused (409). Decks
  that published before 0.8.0 should publish again. Others can *Duplicate to
  customize*.
- **Per-user Google Drive.** Each teammate connects their own Google account
  (Connections → Google) and browses/mounts their own Drive — no shared token.
  For a shared reference corpus, mount a Shared-Drive folder with `clonemd=true`
  and share that VFS folder with the team.
- **Share results, then build on them.** Share a run / folder / archetype from
  its page. A shared folder shows *Copy to my workspace* — that clones it into
  the recipient's own VFS root so their agents can build on it.
- **Export to Drive.** Any VFS file has an *Export to Drive* action that uploads
  it to the exporter's own Google Drive (optionally into a folder id).
- **Duplicate-as-mine archetypes.** Shared/base archetype cards have a copy
  button that forks an editable copy into your own library.
- **Notifications.** A share you receive or a run that finishes while your tab is
  closed lands in the bell (persisted server-side, polled every ~30s).
- **Team cost rollup.** Admin → Usage → *Team spend*: dollars by user and by run
  kind across the whole team.
- **Passwords.** Admin → Users → a user → *Set password* (with "sign them out
  everywhere"); an admin-set password must be changed at the next sign-in.
  Everyone can change their own under Profile → Password.
- **MCP servers are private to whoever added them** (0.8.1). Only the owner's
  agents use a server, and only the owner sees or manages it — so one person's
  OAuth sign-in never reaches everyone. An admin can share a server with the
  whole deck (*deck-wide*) from Connections → MCP Servers or Admin →
  Integrations.
- **Your own AI keys** (0.8.4). A user can run their agents on their own
  Anthropic, OpenAI, Gemini, OpenRouter or xAI keys (Settings → Your own AI keys),
  with no daily limits. On by default; turning it off for the deck moves everyone
  back to the team's keys.
- **Onboarding** (0.8.4). What an admin teaches a user's agent, acting for them, is
  its base: the user adds to it but can't reword or remove it. On a shared agent,
  members suggest lessons and the owner decides. **Pass it on** turns an onboarded
  agent into an archetype others can start from.
- **The Taskboard** (0.8.4). Every user has their own board: a member's tasks go to
  that member, and owners and admins never see them. Personal inboxes are opt-in.
  Next steps and Jev's readings use the board user's own or the team's Jev key.

## Connect Claude to your agents (inbound MCP)

Flight Deck hosts an MCP server so an external MCP client (Claude Code / Claude
Desktop / claude.ai) can list your active agents, send them tasks, and read back
results — all scoped to your own agents. Tools exposed: `list_agents`,
`send_task` (async — returns a `task_id`), `get_result` (poll for live progress
+ the final answer), `cancel_task`. Both entry points are in Flight Deck →
**Connections → Agent access for Claude (MCP)**.

**Claude Desktop / claude.ai (custom connector — OAuth):** add a custom
connector and paste `https://YOUR-FD-HOST/fd/mcp-server`. The connector runs a
standard OAuth 2.1 sign-in (dynamic client registration + PKCE) — the user logs
in with their Flight Deck account; no token to copy. Connected apps are listed
in the same panel and can be disconnected there.

**Claude Code (CLI — static token):** generate a personal access token in the
panel (shown once) and run:

```bash
claude mcp add --transport http captain-fleet https://YOUR-FD-HOST/fd/mcp-server --header "Authorization: Bearer cc_pat_…"
```

Requirements: serve FD over **HTTPS** and set **`FD_PUBLIC_URL=https://YOUR-FD-HOST`**
so the OAuth discovery documents advertise the right URLs (it also honors an
`X-Forwarded-Proto`/`-Host` reverse proxy, but the explicit env is safest).
`FD_JWT_SECRET` must be set (a stable secret) — the OAuth access tokens are
signed with it. Backend-only; the `personal_access_tokens` and `oauth_*` tables
auto-create on restart.

## Gmail sending (opt-in, per user)

Agents can send Gmail on a user's behalf only after that user turns it on in
**Connections → Google → Email sending** — off by default, so deploying this
build changes nothing until someone opts in. Each user sets their own policy:
on/off, an optional allowed-recipients list (addresses or `@domain`), and a
daily limit (default 50 per 24 hours). No admin step and no new Google scope:
the `gmail.compose` scope decks already grant for drafts lets Google accept a
send, so the gate is Flight Deck's policy, not the scope.

Flight Deck is the single enforcement and audit point: agents ask
`POST /fd/google/gmail/send` (same agent gate as `/fd/google/access_token`),
and FD re-checks the owner's policy, sends with the owner's token, records the
send in the `gmail_sends` table (auto-created on restart; its `status` column
is added on restart too) and notifies the owner's bell. A send Gmail never
confirmed (5xx / no answer) is recorded as `status = 'unknown'` — counted, shown
and duplicate-checked like a send. Users see their recent sends in the same
panel. The `gws` Workspace CLI tool is retired (never registered, stripped from
every tools list, refused by the shell tool), so agents' tools can't send around
this gate.

| Variable | Set to | Effect |
| --- | --- | --- |
| `FD_GMAIL_SEND` | `off` (or `0` / `false` / `no`) | Deck-wide kill switch: every agent send is refused regardless of user policies. Unset = users decide. |

The gate stops model mistakes and prompt injection ("send this to …" inside an
email or page). It is not a boundary against an agent with a shell tool, which
can fetch its owner's Google token and call Gmail directly — as with every
shell-capable agent. Deploy = backend restart (plus the frontend bundle for the
Connections panel).

## Known follow-ups (not blockers)

- Flows and scheduler jobs are authenticated but **not yet per-user isolated**
  (all teammates see the same lists). Owner columns on `flows.db` /
  `scheduler_jobs` are the next step.
- A grantee's agents still can't read a shared folder in place — they use *Copy
  to my workspace* first. Owner-qualified VFS addressing is the deeper fix.
- Continuing someone else's Basna/Vatra run is still owner-only.
