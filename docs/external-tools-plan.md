# External tools — plan

Written 2026-11-06. Status: Phase 0 under way.

### Settled with the user (2026-11-06)

1. People who added a tool follow its new versions automatically. No holds.
2. "Share with everybody" needs an admin's approval, and the admin can turn
   that approval off.
3. Tools are added per user, not per agent.
4. A tool runs with the secrets of the person running it. On a shared agent
   that's the member whose turn it is, not the agent's owner.
5. Python only. The desktop app ships a Python (uv-installed 3.11) and nothing
   else, so tools are Python programs and dependencies come from
   `requirements.txt`.
6. Turning the feature on is an admin setting, not an environment variable.
   User-facing name: "Custom tools" (internal ids stay `external_tools`).

## What it is

An external tool is a small program an agent can call when a job is better done
by code than by a model: parse a file format, reconcile two lists, call an API
the same way every time, compute something exactly. Crew's coding agents write
it; Flight Deck keeps it, runs it and shares it.

A tool comes about in one of two ways:

- **The user asks.** "Make me a tool that turns these bank exports into one
  CSV." The agent starts a Crew task to build it.
- **The agent notices.** It has done the same multi-step job by hand several
  times, or a job keeps failing when done by a model. It proposes a tool; the
  owner approves; then a Crew task builds it.

Rules the user set:

- Building a tool needs Crew (an approved user with coding agents available).
- Tools live in the user's workspace on Flight Deck, each in its own folder,
  because a tool can have dependencies and several files.
- Crew develops a tool in that folder.
- A tool is private or shared with everybody on the deck. People it's shared
  with can use it, never edit it.

## Decisions in this plan

1. **The owner is in the path.** `<fd-data>/external_tools/<owner_id>/<slug>/`,
   where fd-data is Flight Deck's `DATA_DIR` (`FD_DATA_DIR`), next to `crew/` and
   `vfs/`. Two people can each have a tool called `invoice-parse`, and who owns
   a tool is read from the path.
2. **Each tool folder holds a git repository, and Crew works in a worktree of
   it.** Nobody edits the version people are calling. The owner's Approve
   merges, Flight Deck installs and tests the merged commit, and only a commit
   that passes goes live. History and rollback come with git.
3. **Flight Deck runs tools, never the agent.** Non-admin agents on multi-user
   decks have no shell or scripts (`EXEC_TOOLS` in `deck_confine.py`), so an
   agent running the script itself would fail for exactly the people sharing is
   for. The agent asks Flight Deck; Flight Deck starts the program, the way it
   starts code apps (`app_runtime.py`) and proxies MCP calls (`mcp_routes.py`).
4. **One call, one process, JSON in and out.** Flight Deck starts the tool's
   command, writes the input as JSON on stdin and reads one JSON value from
   stdout. Nothing stays running between calls.
5. **Making and publishing need Crew; using doesn't.** A user who isn't approved for Crew
   can still use a tool shared with them. They don't choose the code, and
   Flight Deck runs it within limits.
6. **A tool runs as the person running it.** Their secrets, their files (only
   those passed in), their audit log. Never the owner's. On a shared agent that's
   the member whose turn it is.
7. **Nobody's agent runs someone else's tool until that person added it.** A
   share is an offer. Adding it is consent, shown with what the tool says it
   needs (network, secrets, files).
8. **"Share with everybody" needs an admin's approval**, unless the admin turned
   that off (Admin → Settings → Custom tools). A share with everybody runs one
   person's code on everybody's inputs.
9. **Python only.** The desktop app ships one Python and nothing else.
10. **On/off is an admin setting** (`external_tools_enabled` in the deck's
    system config), off by default. No environment variable.

## Folder layout

```
<fd-data>/external_tools/
  <owner_id>/
    <slug>/
      repo/                  git repository; main = what the owner approved
      versions/
        <sha12>/             clean copy of that commit + its dependencies
          tool.json
          ...source...
          .deps/             the tool's venv, made at publish time
          .xtool-ready       written once the version is built and tested
      live.json              {"version", "sha", "published_at", "by", "previous": [...]}
      publish.json           last publish: state, steps with log tails, error
  .runs/
    <caller_id>/<call_id>/   one per call, removed when the call ends
      in/  out/  home/
  .cache/                    uv / pip download caches shared by publishes
```

- `repo/` is the only place code changes, and only through Crew (or the owner's
  own commits if they have a terminal: `grant.terminal`).
- `versions/<sha12>/` is copied from git with `git cat-file --batch`, as Crew
  copies results (`crew_runner._keep_results`), so no smudge filter or hook
  runs. Made read-only (`chmod -R a-w`) after dependencies are in. That stops
  accidents, not a tool that chmods it back: see "Security".
- Keep the live version plus the last 3 for rollback; older ones are deleted.
- `live.json` is written to a temp file and renamed, so a reader never sees half.

## The tool contract: `tool.json`

Every tool has one at the repo root. Crew's brief for a tool task (see Phase 2)
tells the worker to write it and keep it true.

```json
{
  "name": "invoice-parse",
  "title": "Read an invoice PDF",
  "description": "Takes one invoice PDF and returns supplier, date, totals and lines as JSON.",
  "runtime": "python",
  "entry": "main.py",
  "test": ["-m", "pytest", "-q"],
  "input": { "type": "object", "properties": { "file": { "type": "string", "format": "file" } }, "required": ["file"] },
  "output": { "type": "object" },
  "examples": [
    { "input": { "file": "examples/sample.pdf" }, "expect": { "supplier": "ACME d.o.o." } }
  ],
  "secrets": [],
  "network": false,
  "timeout_s": 60,
  "memory_mb": 1024
}
```

- `runtime`: always `python`. `entry` is the Python file Flight Deck runs with
  the tool's venv Python; `test` is the arguments after that Python (the
  starter uses `unittest`, so a tool with no dependencies needs no install).
  Dependencies come from `requirements.txt`, test-only ones from
  `requirements-test.txt`.
- `input` / `output`: JSON Schema. `format: "file"` marks a string that names a
  file (see "Files").
- `examples`: at least one is required. Publishing runs every example and
  checks that `expect` is a subset of the output. That's the evidence that the
  live version works, the same idea as Crew's evidence before a merge.
- `secrets`: names and a one-line reason each. Values never live in the folder.
- `network`: what the tool says it needs. Shown when someone adds the tool;
  enforced only once the OS sandbox exists (Phase 6).
- `timeout_s` 1–300 (default 60), `memory_mb` 128–4096 (default 1024). 512 MB
  of address space is too little for numpy-style imports.

Validation uses `jsonschema` (now declared in `pyproject.toml`).

## Running a tool

`POST /fd/external-tools/{slug}/run` (signed-in user, for "Try it"; built in
Phase 0) or `POST /fd/external-tools/agent/run` (agent route, Phase 1).

1. **Who.** The person running it: the signed-in user, or for an agent, the
   person whose turn it is on Flight Deck's record (the owner, or on a shared
   agent the member), never a name the agent sends.
2. **Which tool and version.** The caller must own the tool, or have added it.
   The version is always `live.json`'s.
3. **Check the input** against `input`. Refuse with the schema error, so a weak
   model can fix its call.
4. **Make the call folder** `.runs/<caller>/<call_id>/` with `in/`, `out/`,
   `home/`. Copy each `format: "file"` input in (see "Files") and rewrite it to
   the path inside `in/`.
5. **Build the environment from nothing.** `PATH` (the version's `.deps` bin
   first), `LANG`, `TZ`, `HOME=<call>/home`, `TMPDIR=<call>/home`,
   `XT_IN`, `XT_OUT`, and the caller's values for the tool's `secrets`. Nothing
   from Flight Deck's own environment. In particular **don't reuse
   `app_runtime._build_subprocess_env`**: it hands code apps `FD_AGENT_SECRET`,
   which would let a tool call Flight Deck as any agent.
6. **Start it** through a small launcher, run as a script with Flight Deck's
   Python so it needs no import path:
   `python -E -s xtool_launch.py --mem <mb> --cpu <s> -- <venv python> -E -s -B <entry>`,
   which sets `RLIMIT_AS`, `RLIMIT_CPU` and `RLIMIT_CORE=0` (as
   `app_subprocess.py` does; macOS often refuses `RLIMIT_AS`) and then
   `execv`s the entry. Not `RLIMIT_NPROC`: it counts every process of Flight
   Deck's OS user, so a low value would stop the tool from starting at all.
   The process group kill covers what a tool spawns.
   cwd = the version folder, own process group (`start_new_session=True`).
   `XT_CALL=<call_id>` is in its environment.
7. **Wait** up to `timeout_s` for the process to end, then kill its group
   either way (leftovers too). It watches the exit code, not `Process.wait()`:
   on Python 3.11 that returns only once every pipe closes, so a child left
   holding stdout would make a finished call look like a timeout. Read stdout
   up to 1 MB and stderr's last 8 KB.
8. **Check the output**: one JSON value, valid against `output`. A non-zero
   exit, bad JSON or a schema miss is an error with stderr's tail.
9. **Hand back files** written to `out/`: copied to the caller's
   `vfs:custom-tools/<slug>/<YYYY-MM-DD>-<call_id>/` (50 files, 100 MB cap) and
   listed in the result.
10. **Clean up** the call folder. Phase 1 also records the call in the caller's
    audit (`tool_audit_store`): tool, version, duration, exit, sizes.

Limits: 2 calls at once per person per tool, 8 per deck. A call waiting for a
slot waits up to 30 s, then says the tool is busy.

### Files

An input marked `format: "file"` (a top-level string, or an array of them)
takes a path the caller could read: a `vfs:<folder>/<path>` in their own VFS
(Phase 0), later also shares under their permission (`vfs_routes._eff_owner`)
and files in the agent's workspace (Phase 1). Flight Deck copies
it into `in/`. The tool never gets a path outside its call folder. 200 MB per
call in v1.

## What the agent sees

One tool, `custom_tools` (the name the model sees), added to `_ECO_CORE_TOOLS`
(`agent_orchestration_mixin.py`) and shown by the registry only while the deck
has custom tools on, so it isn't dropped by eco mode. Crew's note applies: without that, agents never get
its schema.

| action | who | what |
|---|---|---|
| `list` | anyone | tools this agent can use: name, title, one line, version |
| `describe` | anyone | input and output schema, examples, secrets and whether they're set |
| `run` | anyone | run it; returns the output JSON and files |
| `build` | owner with Crew | start a Crew task that makes a new tool |
| `improve` | tool owner with Crew | start a Crew task on an existing tool, optionally with a failed call attached |
| `propose` | anyone | suggest a tool to the owner (Phase 5) |

The system prompt lists up to 20 usable tools by name and one line, so the agent
knows they exist without calling `list`. One dispatcher tool keeps the prompt
small; giving each tool its own schema (as MCP tools get) is a later option for
tools an agent uses all the time.

`build` and `improve` reply as soon as the Crew task exists (like `crew`), and
count as the turn's deliverable so the stall detector doesn't force polling
(Crew round 2). The agent hears back through Crew's callback when the task
finishes, and again when publishing passes or fails.

## Building with Crew

A new Crew place kind, `tool`: `{ "tool": "<slug>" }`.

- `crew_runner._resolve_repo` resolves it to `external_tools/<uid>/<slug>/repo`
  after checking the caller owns it. A folder on this computer (`repo_path`) is
  trusted-only today; this place isn't, because Flight Deck made the folder and
  the owner check is the path.
- `build` with a new slug first makes the folder and the repository with a
  starter commit (`external_tools.create_tool`, built in Phase 0): `tool.json`,
  an echo `main.py` reading stdin and writing stdout, `tests/test_main.py`
  (unittest), `README.md` with the contract, `.gitignore`. It publishes as is.
- The brief Crew sends the worker gets a tool block: the contract above, "keep
  `tool.json` true", "at least one example with `expect`", "tests must pass with
  `test`", "read JSON on stdin, write one JSON value on stdout, logs to stderr",
  "secrets come from the environment, never from files".
- The worker can run the tool and its tests in its worktree as it does today
  (Crew dev environment: `crew_env.py`).
- The owner approves the merge as for any Crew task (`crew_runner.merge`, only
  from a person's click or Approve). After a merge into `main` of a tool repo,
  `merge` calls `external_tools.publish(owner, slug, sha)`.
- The worker never touches `versions/` or `live.json`. They're outside its
  worktree.

## Publishing

`publish(owner, slug, sha)` runs in the background and writes `publish.json`:

1. Copy the commit to a building folder (git cat-file, as above).
2. Read and check `tool.json`. Refuse a missing entry, example, or schema.
3. Make `.deps/` from Flight Deck's own Python (a frozen build looks for
   `python3` instead), with network, a 10-minute limit, `HOME` in a temp folder:
   `uv venv` + `uv pip install -r requirements.txt` when `uv` is on the PATH,
   else `python -m venv` + pip. No requirements: a venv without pip, in
   milliseconds.
4. Run `test` in the building folder, clear what tests leave (`__pycache__`,
   `.pytest_cache`), mark it ready, make it read-only and move it to
   `versions/<sha12>/`. Then run every example through the real run path as the
   person publishing, with their secrets.
5. All pass → write `live.json`, tell the owner (bell + Crew callback to the
   agent that asked). Any fail → keep the previous live version, tell the owner
   with the failing step and its log tail, so the agent can `improve`.

Rollback: the page's version list has "Make live" on each kept version. It only
rewrites `live.json`.

## Sharing

- Resource type `external_tool` in `share_routes.VALID_TYPES`, resource id
  `<owner_id>/<slug>`, permission always `use` (like archetypes, which are
  use-only).
- "Everybody" needs a grantee that isn't a person. `resource_shares.grantee_id`
  is NOT NULL and references `users(id)`, so offers to everybody live in their
  own table, `external_tool_listings` (built that way in Phase 3).
- A share with a person is an offer: it shows up under "Shared with me" with
  "Add". A share with everybody goes to the admin for approval first; once
  approved it shows in the deck's catalog with "Add".
- **Adding** records the caller's consent for this tool at this version: what it
  said about network and secrets. Only added tools are listed to their agents.
- Taking a share back: the tool disappears from their list at once; calls in
  progress finish.
- The owner deleting a tool removes it for everybody, after a confirm that says
  how many people use it.

### Updates to a shared tool

People who added a tool follow its new versions automatically (settled). When a
version declares more than the one before (network newly on, a new secret, a
larger timeout or memory), they get a bell saying what changed; nothing waits.

### Secrets

Each user enters their own values for a tool's `secrets` on the tool's page,
sealed with `fd_crypto.seal` (as Crew → Logins seals `crew:own_model`). Key
`xtool:<owner_id>/<slug>:<NAME>`. A call with a secret not set is refused with
"<NAME> isn't set: Custom tools → <title> → Secrets", never run without it.
The values are always the runner's: the owner's own values never reach anyone
else's call.

### What the owner sees of other people's calls

A shared tool runs on other people's inputs. Their inputs, outputs, files and
stderr belong to them and stay in their audit. The owner sees counts per day,
failure counts by kind (timeout, bad output, exit code) and which versions are
in use. A user can choose "Send this failure to the owner" on a failed call,
which shares that one call's stderr tail and input.

## The page

"Custom tools" in the full layout, in the Admin console's look like the Crew page;
kiosk reaches it through Settings.

- **Mine**: each tool with status (building, live vN, publish failed), last
  calls, Share, Versions (with Make live), Secrets, Delete. "New tool" opens a
  short form (what should it do, inputs, outputs) that starts a Crew task.
- **Shared with me** and **Catalog** (everybody): Add / Remove, what it declares,
  who made it.
- **Tool page**: description, input and output schema as a form, "Try it" (runs
  through the same path, as the signed-in user), examples, my recent calls.
- **Admin → Approvals**: requests to share with everybody, with tool.json and
  the last publish (only while the admin's approval switch is on).

## Phases

### Phase 0: storage and runner (Flight Deck only) — built 2026-11-07

- `flight_deck/external_tools.py`: paths, `tool.json` checks, `create_tool`
  (starter repo), `publish` / `start_publish`, `run_tool`, `make_live`,
  `delete_tool`, limits.
- `flight_deck/xtool_launch.py`: the launcher (standard library only).
- `flight_deck/external_tools_routes.py`, `/fd/external-tools`: `status`, list,
  create, get, delete, `publish` (POST starts, GET follows), `live`, `run`,
  `secrets` (GET/PUT, sealed per person under `xtool:`, a server-owned settings
  prefix). Own tools only. Create and publish need a Crew grant.
- Admin switches in the system config: `external_tools_enabled` (default off)
  and `external_tools_everybody_approval` (default on), on the console's
  Settings page and the full layout's Admin → Settings.
- Tests: `tests/test_flight_deck/test_external_tools.py` (34: the starter,
  failing example keeps the old version live, failing tests, bad tool.json
  cases, a dependency from requirements, going back, names, delete, empty
  environment, the runner's secrets, input and output schemas, crash / bad JSON
  / too much output, timeout kills what the tool started, a child left holding
  stdout, read-only folder, files in and out, memory (Linux only), and over
  HTTP: off until an admin turns it on, Crew needed, a whole tool with a VFS
  file and secret, someone else's tool out of reach).
- Live check on a throwaway deck: switched on from the Settings page, made a
  CSV tool, changed it in its repo as a worker would, published, ran it on a
  VFS file; the summary landed in `vfs:custom-tools/`.

### Phase 1: the agent tool — built 2026-11-07

- Agent: `captain_claw/custom_tools_client.py` (status refreshed per turn on the
  owner's turns, the owner's tools for the prompt, calls with the member grant
  on a member's turn) and `tools/custom_tools.py` (`list`, `describe`, `run`),
  registered with `requires_custom_tools`, in `_ECO_CORE_TOOLS`, listed in the
  prompt (up to 20, owner's turns), refused in rehearsals, probed by self-check.
- Deck: `external_tools_routes.agent_router` (`/fd/external-tools/agent/status`,
  `tools`, `describe`, `run`), included before the main router; tool names
  `agent`, `status`, … are reserved. The runner is the agent's owner on Flight
  Deck's record, or the member through `speaker_grants.acting_member` (the three
  routes are grant-aware). A member gets `custom_tools` in the member allowlist
  (list / describe / run only) and runs their own tools with their own secrets.
- Audit: every tool call is already recorded by the agent's tool-call register.
- Tests: `tests/test_flight_deck/test_external_tools_agent.py` (14), including
  the real tool class over HTTP into the deck app, and a member's turn running
  the member's tool with the member's secret.

### Phase 2: Crew builds tools — built 2026-11-07

- Crew: a `tool` place (`DispatchBody.tool`, `_place_of`, `_tool_place`):
  `repo_kind = "tool"`, the caller's own tool only (path = ownership), no
  admin's trust needed. A tool task's first prompt adds
  `external_tools.WORKER_CONTRACT` after the environment block (the stored
  brief stays clean). `next_steps` gives only the merge step (no pull request
  for a tool). After a local merge, `_spawn_tool_publish` runs
  `external_tools.publish` in the background as the owner with their secrets,
  keeps the result in `delivery.publish`, and tells the owner (bell) and the
  agent that asked ("is live" / "wasn't published", with the failing step).
- Deck: `POST /fd/external-tools/build` and `/{slug}/improve` for the page;
  `/fd/external-tools/agent/build` and `/agent/improve` for agents, through
  Crew's own agent dispatch (callback, budget, origin). A tool made for a task
  that then can't start is removed again. Members can't build.
- Agent: `custom_tools` gains `build` and `improve` (tool, title, what,
  harness, model).
- Tests: `test_external_tools_crew.py` (4: Crew's fake CLI, a real merge that
  publishes, a merge with a wrong example that keeps the live version, place
  rules, the contract only for tool tasks) and `test_external_tools_build.py`
  (5: person and agent build/improve, Crew needed, members refused, nothing
  left behind when the task can't start, the tool's arguments).
### Phase 3: sharing — built 2026-11-07

- To a person: the existing share system, `resource_type = "external_tool"`,
  id `<owner>/<slug>`, always use-only; the bell says to add it under Custom
  tools → Shared with me.
- To everybody: a table `external_tool_listings` (owner, slug, status
  pending / approved / rejected, who decided), not a nullable grantee in
  `resource_shares`: that table's grantee is NOT NULL with a foreign key, and
  SQLite can't change that without rebuilding it. Pending waits for an admin
  (bell to admins, `GET /fd/external-tools/approvals`, `POST
  /approvals/{owner}/{slug}`); with the admin's approval switched off, pending
  counts as offered. Only a published tool can be offered.
- Consent: `POST /fd/external-tools/added` stores what the live version
  declares in the person's server-owned setting `xtool-added:<owner>/<slug>`;
  `DELETE /added` removes it and their secrets for the tool. `can_use` = own
  tool, or added and still offered (a share or a live listing), checked on
  every call.
- Use, never change: `GET /{slug}?owner=`, `POST /{slug}/run?owner=`,
  `GET/PUT /{slug}/secrets?owner=` for someone else's tool (what it does and
  needs, never its repository, versions or logs); every write route stays
  owner-only. Agents list own + added tools; an added tool's `ref` is its
  name, or `<owner>/<name>` when two would share it.
- `GET /shared`, `GET /catalog`, `GET /{slug}/sharing` (people, the everybody
  offer, how many added it), `PUT /{slug}/everybody`, `GET /{slug}/usage`
  (calls and failures per day, failure kinds, versions, how many people;
  callers kept as a short hash, never inputs or outputs). These fixed paths
  are on `catalog_router`, included before the main router; `shared`,
  `catalog`, `approvals`, `added`, `build` are reserved tool names.
- Updates: `external_tools.AFTER_PUBLISH` tells everyone who added the tool
  when a new version asks for more (internet, a secret, more time or memory).
  They follow it automatically.
- Delete takes the tool's shares, listing, everyone's consent and everyone's
  secrets for it.
- Tests: `test_external_tools_sharing.py` (9).

### Phase 4: the page

- `components/tools/` with the panes above; kiosk Settings entry; phone width.

### Phase 5: agents propose tools

- After a turn in which the agent did the same kind of job by hand that it has
  done before, or a long chain of steps that could be one program, it may draft
  a proposal (owner turns only, the same gates as `lesson_proposals.py`: no
  automation, member, relay, private or exam turns; cooldown; daily cap).
- `propose` from the agent tool goes the same way.
- Proposals land in the owner's inbox with a bell; Approve starts the Crew task.
  Nothing is built without that click: every build spends Crew time and the
  user's coding-agent limits.

### Phase 6: the OS sandbox

- Run tools under the per-agent sandbox from `docs/agent-os-sandbox-plan.md`
  (Landlock + seccomp on Linux): version folder read-only, call folder
  read-write, nothing else; network only when `tool.json` says so.
- Until then, `network` and "only its folder" are declarations, not walls.

## Security

What holds from Phase 0:

- Agents never run tool code; Flight Deck does, after an access check on its own
  record of who's calling.
- A tool gets no Flight Deck environment, no agent secret, no owner's secrets,
  and only the files passed to it.
- Memory, CPU, time and output size are limited; a tool's leftover processes
  are killed with its group.
- Nobody's agent runs a tool its owner didn't add; sharing with everybody needs
  an admin unless the admin turned that off.

What doesn't hold until Phase 6, said plainly: a tool process runs as Flight
Deck's OS user, so deliberate code can read what that user can (other users'
folders, the deck's database) and reach the network. That is the same reach an
approved Crew worker has today, and only Crew-approved users can author tools,
so this adds no new person with that reach. It does mean a tool runs
unattended, at someone else's call. That's why adding is consent and "everybody"
is approved by an admin. Decks that can't accept this leave Custom tools off
(the default) until the sandbox exists.

## Still open

- **Windows desktop**: the launcher passes the exit code on, but no limits apply
  there and the process-group kill is `proc.kill()` only. Untested.

## Existing pieces this reuses

| Need | Already in the code |
|---|---|
| Per-deck folder | `server.DATA_DIR`, as `crew_store` uses |
| Subprocess with limits | `app_subprocess.py` (rlimits), `app_runtime.py` (spawn, process group) |
| Agent identity on agent routes | `agent_identity.require_agent_caller` |
| Coding agents, worktrees, Approve, merge | `crew_runner.py` (`_resolve_repo`, `merge`, `_keep_results`) |
| Clean copy of a commit | `git cat-file --batch` in Crew results |
| Sharing table and modal | `share_routes.py`, `db.resource_shares`, `ShareModal.tsx` |
| Sealed per-user secrets | `fd_crypto.seal`, `crew_logins.py` |
| Audit of calls | `tool_audit_store.py` |
| Proposals with an inbox and Approve | `lesson_proposals.py` (agent and deck sides) |
| Files out to the caller | VFS (`vfs_routes._eff_owner`), as Crew uses `vfs:crew/` |
