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
| `FD_CORS_ORIGINS` | your exact web origin(s), comma-separated (e.g. `https://claw.example.com`) | The default is `*` **with credentials**, which is unsafe for a cookie-bearing app. Pin it to your real origin(s). |
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

## First-run checklist

1. Set the environment variables above and start the backend.
2. Register the **first** account — it becomes the admin automatically.
3. On the **Admin** page: create an account for each teammate (you can set their
   password directly; they can change it later under their profile).
4. On the **Admin** page: set the org **provider API keys** (Anthropic, OpenAI,
   …). These stay server-side.
5. Publish a **team tier set** (Library → tier sets → *Publish to team*) so
   teammates run models without pasting keys.
6. Ask each teammate to connect their **own** Google account (Connections) if
   they will use Drive.

## Collaboration features (0.8.x)

- **Shareable tier sets.** Library → configure a tier set → *Publish to team*
  (admin). Teammates who never set up models automatically run on it, and API
  keys are never shared — each tier stores a `@system` sentinel resolved
  server-side from the org key store. Others can *Duplicate to customize*.
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
- **Admin password reset.** Admin → Users → expand a user → *Reset password*.

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

## Known follow-ups (not blockers)

- Flows and scheduler jobs are authenticated but **not yet per-user isolated**
  (all teammates see the same lists). Owner columns on `flows.db` /
  `scheduler_jobs` are the next step.
- A grantee's agents still can't read a shared folder in place — they use *Copy
  to my workspace* first. Owner-qualified VFS addressing is the deeper fix.
- Continuing someone else's Basna/Vatra run is still owner-only.
