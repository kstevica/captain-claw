# External tools — plan

Written 2026-11-06. Status: Phases 0–6 built and merged on 2026-11-07 (#450–#456), the New tool
spec check too (#457). Phase 7, tools that call the agent, planned 2026-11-07.

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
larger timeout or memory, and from Phase 7 a new `uses` entry or host, or `asks`), they
get a bell saying what changed; nothing waits.

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
  large dialog in two steps (added 2026-11-07, after the user's first look):
  1. **Describe it**: name, title, and the spec in a text box that fills the
     dialog (up to 8000 characters).
  2. **Check it and build**: `POST /fd/external-tools/review`
     (`external_tool_review.py`) makes one call to the person's own model
     (balanced, else reason, else fast). The prompt includes the worker contract, how a tool runs
     (one process per call, nothing kept between calls, files as vfs paths,
     limits, sandbox, runner's secrets), what agents already do without a tool,
     and the person's own and added tools. The model answers with a fuller spec,
     a list of changes with a reason for each, and any overlap with something
     that already exists. The person
     switches between the checked spec and their own, edits it, picks the
     coding agent and builds. Nothing is kept. A check that fails (no model,
     an error, no spec in the answer) says why, and the person can still build
     what they wrote. Escape and clicks outside don't close the dialog while it
     holds text.

  In the kiosk dialog the page takes the dialog's full width (two or three
  columns of cards, by width).
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

### Phase 4: the page — built 2026-11-07

- `components/customTools/`: `CustomToolsPage` (tabs Mine / Shared with me /
  Everybody / Approvals for an admin), `ToolSheet` (Try it from the input
  schema or as JSON, your secrets for it, publishing with its steps, versions
  with Make live, Change it through Crew, Offer it to people (ShareModal) and
  to everybody, how it's used, delete), `NewToolDialog` (name, title, what it
  should do, coding agent), `toolParts`. `services/customToolsApi.ts`,
  `stores/customToolsStore.ts` (deck status, kiosk open state, a focus a
  bell can set).
- Full layout: view `custom-tools`, a sidebar entry under Multi-Agent shown
  only while the deck has custom tools on. Kiosk: Settings → Custom tools
  opens it in a wide dialog. Bells with `custom_tool` /
  `custom_tool_approval` open the page (on the Approvals tab for an admin's).
- Checked on a throwaway deck with sign-in: the admin's page and a tool's
  sheet (a run on a VFS file with the owner's secret, the report file, the
  sections), offering to everybody and approving it; a non-admin in kiosk mode
  without Crew reaching it from Settings, Add, the missing-secret refusal,
  their own secret, a run on their own file; phone width (no sideways scroll,
  bottom sheet).

### Phase 5: agents propose tools — built 2026-11-07

- Agent: `captain_claw/tool_proposals.py`, after the owner's turns only (the
  same gate as lesson proposals in `chat_handler`: no automation, public,
  member or relay turns, no workers, no private or exam turns). A cheap gate
  first: a turn of 8+ tool calls (`min_tool_calls`), or words that read like a
  repeated mechanical job plus 3+ tool calls. Then the agent's own model with
  `instructions/tool_proposal_prompt.md` drafts one tool (name, title, what,
  why) or none. Only when the deck has custom tools on and the owner can
  build. Config `tool_proposals` (enabled, cooldown 600 s, 4 a day);
  interaction label `tool_proposal` is background.
- `custom_tools action=propose` sends a draft the same way (source "asked").
- Deck: `flight_deck/external_tool_proposals.py`, stored per owner in
  `external_tools/<owner>/.proposals.json`; drops a draft whose name is an
  existing tool, or like one waiting or dismissed (`retrieval.same_fact`),
  at most 20 waiting; one unread bell per owner, settled when nothing waits.
  `POST /fd/external-tools/agent/propose` (owners only, who can build),
  `GET /fd/external-tools/proposals`, `POST /proposals/{id}/build` (makes the
  tool and starts Crew as the agent that proposed it, so it hears back) and
  `/dismiss`.
- Page: "Your agents suggest" at the top of Mine, with Build it (coding agent
  picker) and Dismiss; the bell opens the page.
- Tests: `tests/test_tool_proposals.py` (10) and
  `test_flight_deck/test_external_tool_proposals.py` (5).

### Phase 6: the OS sandbox — built 2026-11-07

- `xtool_launch.py` puts every run (and a publish's test step) in a Landlock
  sandbox on Linux before it becomes the tool, with ctypes and the standard
  library only, no root:
  - files: read and run only the version folder, the Python it runs on
    (`sys.base_prefix` and its executable's folder), `/usr`, `/lib*`, `/bin`,
    `/sbin`, `/etc` and `/proc/self` (the tool's own /proc, so not Flight
    Deck's environment); read and write only the call's folder (and, for a
    test step, the not-yet-live version) and a few devices;
  - network (ABI 4+): no TCP connect or bind unless tool.json says
    `"network": true`;
  - signals and abstract sockets (ABI 6+): nothing outside the sandbox.
- `--probe` says what a host offers; `external_tools.sandbox_info()` asks once
  and `/fd/external-tools/status` shows it, as does the Custom tools card in
  Settings.
- Admin switch `external_tools_require_sandbox` ("Run tools only in the
  sandbox", off by default): on, a host without the sandbox runs no tools
  (503 from Flight Deck, and the launcher itself refuses with exit 126).
- Not sandboxed: installing dependencies at publish (it needs the network and
  the download cache). That runs the owner's own `requirements.txt`.
- Not covered by Landlock: UDP, so DNS and other UDP traffic still work with
  `"network": false`; and reads of world-readable system files under the read
  paths.
- Checked in a Linux container (kernel 6.12, Landlock ABI 6) from the Mac,
  running as root, so the refusals are Landlock's: a file outside unreadable,
  the tool's own folder unwritable, the call folder writable, another
  process's `/proc/<pid>/environ` unreadable, no signal to Flight Deck, TCP
  refused with `--no-net` and allowed without; and a tool running from its own
  venv under exactly the read-only list Flight Deck passes.
- Tests: `test_xtool_sandbox.py` (needs Linux with Landlock; skips elsewhere;
  runs standalone too) and `test_external_tools_sandbox.py` (the flags each run
  gets, the refusal, the switch and the status).

### Phase 7: tools that call the agent — planned 2026-11-07

A custom tool can use the agent's own tools (read a page, search mail, write a
spreadsheet) and other custom tools while it runs. It never does that work
itself: it asks the agent that ran it, and that agent does it, with its own
connections, keys, folder limits, rules and log.

#### Settled with the user (2026-11-07)

1. A tool may use the agent's **read** and **write** tools. **Send** tools work
   only when the tool's owner runs it. **Run code**, **money** and anything
   unclassified are never available.
2. When a new version of a shared tool uses more, people who added it keep
   following it; the Phase 3 bell says what it now uses.
3. Custom tools can call other custom tools, in this phase.
4. "Try it" on the tool's page asks which of the person's agents runs it.
5. Who a request runs as comes with the tool's request: Flight Deck sends the
   run's identity (the runner, and on a shared agent the member and their
   grant) with every request, and the agent acts as that identity. It doesn't
   rely on what the turn's context happens to hold.
6. Writes that reach other people count as send (below).
7. A tool can ask the person a question, or a set of them, and waits for the
   answer (Asking the person, below).

Rules that follow from these (the first two proposed with the plan; the third
settled as #6):

- **Web reads name their hosts.** `web_fetch` and the other web readers are
  "read", but a URL can carry data out. A tool that uses them lists the hosts it
  reads from (`"hosts": ["api.frankfurter.app"]`); Flight Deck refuses any
  other host, and the hosts are shown before anyone adds the tool.
- **Nothing that changes the agent or sets up later work**, whatever its
  class: `cron`, `personality`, `playbooks`, `insights`, `facts`,
  `intentions`, `project_memory`, `reasoning_effort`, `apis`,
  `synthesize_flow`, `todo`, `topics`. A tool must not leave behind a
  scheduled job or a new habit in someone's agent.
- **Writes that reach other people count as send**: creating, changing or
  deleting a calendar event (invitations and cancellations go out), and
  answering an invitation. Checked against every action of the allowed tools:
  the Drive and OneDrive tools have no sharing action. Counted as send for
  custom tools only (`external_tool_uses`), so the tool-call register's own
  classes don't change.

#### The rule: the agent that ran the tool does the work

Only the agent whose `custom_tools run` started this call (or, from the page,
the agent the person picked) answers its requests. Flight Deck already knows
which agent and whose turn it is (`Runner`: user, agent slug, member), and
sends that identity with every request (settled #5). So:

- the runner's connections are used: on a shared agent, the member's turn
  uses the member's Gmail, as their secrets are used today;
- the agent's guards still apply (`_execute_tool_with_guard`: the script/tool
  guard, blast radius, folder confinement);
- every call is in that agent's tool-call log, as "via custom tool <title>";
- the agent can refuse: a tool it doesn't have, or one its rules block, fails
  as it would in chat.

#### How a call travels

1. The agent's `custom_tools run` makes a callback id (random, 32 bytes),
   remembers its waiting run under it (the session, who the turn is for, its
   abort), and sends the id with the run.
2. Flight Deck starts the tool with one end of a socket pair as file
   descriptor 3 (`XT_CAPTAIN_FD=3`; `pass_fds`, kept through the launcher's
   `execv`) and reads the other end while the tool runs.
3. The tool asks through a small helper module, `captain`, that Flight Deck
   puts on its path (standard library only):

   ```python
   import captain
   page = captain.tool("web_fetch", url="https://api.frankfurter.app/latest")
   # → {"ok": True, "content": "...", "files": []}
   total = captain.custom_tool("csv-total", {"file": captain.file("out/rows.csv"), "column": "amount"})
   # → the other tool's output JSON
   ```

   On the socket that is one JSON line per request,
   `{"id": 1, "op": "tool" | "custom_tool", "name": ..., "input": {...}}`, and
   one line back, `{"id": 1, "ok": ..., "content" | "output": ..., "files": [...], "error": ...}`.
   One request at a time in v1. (`op: "ask"` goes to the person, not the
   agent: Asking the person, below.)
4. Flight Deck checks the request (below) and passes it to the agent:
   `POST http://127.0.0.1:<web_port>/api/custom-tools/callback` with the
   agent's web token, as it calls `/api/llm/complete` today. The body carries
   the callback id and the run's identity: the runner's user id, whether it is
   a member, and the member's speaker grant (the one the run came with).
5. The agent finds the waiting run by its id and refuses any other id, so only
   a call it is waiting on right now can make it act. It also refuses an
   identity that isn't the one that run started with. It sets the speaker from
   the request's identity for this call (as a member's turn sets it), runs the
   tool through `_execute_tool_with_guard`, and answers. From Try it there is
   no turn: the identity is the person on the page.
6. Flight Deck hands files across (below) and writes the answer back.

When the run ends, or the person stops the turn, the agent forgets the id and
later requests are refused, so the tool fails soon after.

#### What a tool declares: `uses`

```json
"uses": [
  {"tool": "web_fetch", "hosts": ["api.frankfurter.app"], "why": "reads today's exchange rates"},
  {"tool": "google_mail", "actions": ["search", "read"], "why": "finds the invoice mails"},
  {"custom_tool": "csv-total", "why": "adds up the rows it extracted"}
]
```

- Each entry is one of the agent's tools (optionally only some of its actions)
  or a custom tool, always with a `why`.
- Publishing refuses an entry that `tool_risk.risk_for` puts in run code,
  money or other, a tool on the never list, a tool name it doesn't know, and a
  web tool without `hosts`.
- At call time Flight Deck refuses anything not declared and classifies the
  real call with `risk_for(tool, input)`: read and write pass; send passes only
  when the runner is the tool's owner; the rest never.
- A tool that needs an interactive approval is refused in v1 ("needs your
  approval: do it from chat").
- At most 50 requests per top-level run, counted across nested tools. Each
  waits only as long as the run has left.

#### Custom tools calling custom tools

- `{"custom_tool": "csv-total"}` names one of the tool owner's own tools.
- It travels through the agent like any request, so the inner run has the same
  runner, secrets and files, and shows in the agent's log.
- It runs only if the runner can use it (their own, or one they added).
  Adding a tool lists what it depends on, with Add for each one offered to
  them, and says plainly when one isn't ("won't work for you: csv-total isn't
  offered to you"). Offering a tool to someone or everybody offers to share
  its dependencies the same way.
- Flight Deck keeps each run's record: call id, owner and name, depth,
  deadline, request count. The agent passes `via: <outer call id>` with the
  inner run. Refused: depth over 3, a tool already in the chain. The inner run
  gets its own timeout or what the outer has left less 5 seconds, whichever is
  smaller.

#### Files across the line

- **To an agent tool:** `captain.file(path)` marks a file the tool wrote in
  its run folder. Flight Deck copies it into
  `vfs:custom-tools/<slug>/<day>-<call>/work/` (beside the call's delivered
  files) and puts that path in the input. Any other path is refused.
- **From an agent tool:** its answer is text (`content`). A file it saved is
  brought in with `captain.vfs_file("vfs:<folder>/<path>")`, which copies it
  into the run's `in/` folder and returns that path; it needs
  `{"tool": "read"}` in `uses`.
- **From an inner custom tool (7b):** the files it made are copied into the
  run's `in/` folder, and the tool gets those paths.
- The 200 MB per call covers all of it. The tool never gets a path outside its
  call folder.

#### Try it on the page

- A tool with `uses` asks "Run as": one of the person's own agents. Asleep
  ones are woken (`agent_idle`).
- Flight Deck asks that agent for a hosting slot, `POST /api/custom-tools/host`
  → a callback id valid for the run's timeout. Its requests run in a session
  of their own (`custom-tools-try`), not in the person's chat.
- A member of a shared agent tries such a tool from chat in v1.
- A tool without `uses` runs as today.

#### Publishing and tests

- There's no agent when a version publishes. Examples carry the replies the
  tool should get, in order:
  `{"input": ..., "expect": ..., "calls": [{"tool": "web_fetch", "reply": {"ok": true, "content": "..."}}]}`.
  Flight Deck answers from them. A request that doesn't match the next one
  fails the example and says which.
- Tests: when `XT_FAKE_CALLS` names a JSON file of the same shape, the helper
  answers from it.
- `WORKER_CONTRACT` and the New tool check prompt explain `uses`, `hosts`, the
  helper, the recorded replies, and that a tool reading the web or a
  model-backed tool doesn't give the same answer every time.

#### What people see

- The declares line and the tool page: "Uses your agent: web_fetch
  (api.frankfurter.app), google_mail (search, read)" and "Uses: csv-total",
  each with its why, and a "Calls your agent" pill.
- The update bell (Phase 3) names new `uses` entries; nothing waits (settled
  #2).
- The agent's tool-call log has every request; the owner's usage counts
  include requests per run (counts only, as today).
- `custom_tools describe` shows `uses`, so the agent knows the tool will call
  back.

#### Asking the person (settled #7)

A tool can stop and ask the person running it for an answer, a decision or a
choice (one or several), and carries on with what they answer. Flight Deck
handles it directly; the agent isn't needed.

- **Declared:** `"asks": true` in tool.json. It is shown before anyone adds the
  tool ("Asks you questions").
- **The helper:** one question gives back its answer; a set gives back a dict
  by id, all shown in one dialog.

  ```python
  account = captain.ask("Which account?", choices=["Business", "Personal"])
  answers = captain.ask([
      {"id": "account", "question": "Which account?", "choices": ["Business", "Personal"]},
      {"id": "months", "question": "Which months?", "choices": ["Jan", "Feb", "Mar"], "multiple": True, "min": 1},
      {"id": "send", "question": "Send the summary to the accountant?", "confirm": True},
      {"id": "note", "question": "Anything to add?", "text": True, "optional": True},
  ], wait_s=900)
  # → {"account": "Business", "months": ["Jan", "Feb"], "send": True, "note": ""}
  ```

  Kinds: **text** (one line, or `"lines": n`), **choice** (one of a list),
  **multiple** (several, with `min` and `max`), **confirm** (yes or no). Each
  question can have `help`, `default` and `optional`. On the socket it is
  `{"op": "ask", "questions": [...], "wait_s": ...}`.
- **How it ends:** the answers; `captain.Declined` when the person closes it
  with "Don't answer"; `captain.NoAnswer` when the wait runs out or there's
  nobody to ask. The tool decides what to do then.
- **Who is asked:** the runner, from the tool's request (settled #5): on a
  shared agent, the member whose turn it is; from Try it, the person on the
  page. Nobody else ever sees the question.
- **Where:** every open Flight Deck tab of that person opens the dialog (full
  layout, kiosk, phone width). It is also a Needs-you item in Line and the
  bell, so someone not looking finds it there. Answered in one tab, it closes
  in the others. The bell checks every 30 seconds, too slowly for a dialog, so
  each signed-in tab waits on `GET /fd/external-tools/asks?wait=25` (only
  while custom tools are on) and gets an open question at once. The answer
  goes to `POST /fd/external-tools/asks/<id>`, as `{answers}` or
  `{"declined": true}`. Flight Deck checks the answers against the questions
  (a choice from the list, how many for multiple, lengths) before the tool
  gets them.
- **The dialog:** "<tool title> asks", with "a tool by <owner>" when it isn't
  the person's own, "csv-total, used by bank-report" for an inner tool, the
  agent's name and the time left. A fixed line: "Never type a password or key
  here: a tool's secrets go in its Secrets." No password fields.
- **Paused:** while a question is open the tool waits on its socket, doing
  nothing, and Flight Deck stops its time limit. The question's own wait
  applies instead: `wait_s`, 10 minutes by default, at most an hour. Limits:
  5 asks per run, 10 questions per ask, 500 characters a question, 50 choices
  of 200 characters each, 4000 characters an answer.
- **The agent waits too.** So a long wait isn't cut by an HTTP timeout,
  `custom_tools run` starts the run and then follows it
  (`GET /fd/external-tools/agent/runs/<id>?wait=30` until it's done) instead
  of holding one long request. While a question is open, the run's state says
  so, and the agent's chat shows "waiting for your answer in Flight Deck".
  Stopping the turn kills the run and closes the question.
- **Nobody to ask:** a run from a scheduled task or autonomous work (the agent
  sends its turn's origin with the run) gets `captain.NoAnswer` at once, with
  "no one to ask: this ran from a schedule", and no dialog opens.
- **Chat channels** (Telegram, WhatsApp, the phone app): in v1 the question
  waits in Flight Deck, and the person finds it there. Asking in the channel
  the turn came from is later.
- **Publishing and tests:** examples record answers in `calls` like the
  agent's replies (`{"ask": ["account"], "reply": {"account": "Business"}}`,
  in order); `XT_FAKE_CALLS` the same.
- **Audit:** the questions and the answers are part of the run's record in the
  runner's audit. The owner of a shared tool sees counts only, as today.

#### Where it runs

- Linux and macOS. On Windows a tool with `uses` or `asks` is refused
  ("needs Linux or macOS"); later, a loopback socket with a per-run token.
- Landlock doesn't rule an inherited, already connected socket (it rules TCP
  connect and bind, and abstract sockets), so `--no-net` stays on. Checked in
  the Linux container as in Phase 6.

#### Built so far

- **7a (2026-11-07):** `external_tool_uses.py` (the rules: declared uses,
  classes, never list, hosts, invitations as send); `xtool_captain.py` (the
  helper, copied into every version as `captain.py`, into new tools' starter
  and into each Crew worktree of a tool); the line in `_run_proc` (a socket
  pair, `pass_fds`, `$XT_CAPTAIN_FD`); `CallContext`, `_answer` and
  `RecordedLink` in `external_tools.py` (examples' `calls`, checked with
  tool.json); `AgentLink` in the routes (the agent's registry entry and token,
  the identity with every request, handed files into the VFS, `vfs_file`);
  on the agent `custom_tool_callbacks.py` and `POST /api/custom-tools/callback`
  (owner-locked like `/api/tool`; mail authority `custom_tool`, allowed only
  when the owner runs the tool). An agent too old to send a callback id gets
  "restart the agent". The page's Try it refuses such a tool until 7c.
  Tests: `test_external_tool_uses.py` (25), `test_external_tools_calls.py`
  (11), and the sandbox check (the line works under Landlock with `--no-net`,
  container, ABI 6).
- **7b (2026-11-07):** `{"custom_tool": "<name>", "why": ...}` in `uses` (one
  of the owner's own tools; not itself). `captain.custom_tool()` goes to the
  agent (`op: "custom_tool"`), which runs it with its own `custom_tools` tool,
  naming the run that asked (`via`) and handing back Flight Deck's result as
  JSON. Flight Deck keeps each running call that can ask for more
  (`external_tools._RUNS`); a run with `via` must be that call's own runner,
  declared in its uses, not already in the chain, at most `MAX_DEPTH` (3)
  deep, and it gets the outer run's time left less 5 s; the 50-request budget
  is shared. The inner tool's files come into the outer run's `in/`
  (`captain.last_files`). People need the tools it runs: `depends_on` on a
  tool's view (yours, added, offered, not offered) with Add on its sheet, a
  line on the card, and for the owner "Offer them too"
  (`POST /fd/external-tools/<slug>/offer-dependencies`: the same people, and
  everybody if it's offered to everybody). Tests:
  `test_external_tools_nested.py` (5).
- **7c (2026-11-07):** Try it on a tool with `uses` asks "Run as" (the
  person's own agents from `GET /fd/external-tools/agents`; an asleep one is
  woken, one that's off is refused). Flight Deck asks that agent for a hosting
  slot (`POST /api/custom-tools/host` on the agent, owner-locked; `{"close":
  id}` ends it when the run does); its requests run as the person in the
  session `custom-tools-try`. The declares line and the tool sheet show what
  it asks of the agent ("Calls your agent: write, web_fetch (example.com)",
  each use with its why) and the tools it runs; the update bell names new
  uses and tools it now runs. The New tool check knows a tool can use its
  agent and should prefer that to network access and new secrets. Tests:
  `test_external_tools_run_as.py` (3).

#### Build order (one PR each)

- **7a: the agent's tools.** The socket and helper, Flight Deck's relay, the
  agent's callback endpoint and its waiting runs, `uses` with the class gate,
  hosts and never list, files, recorded replies at publish and
  `XT_FAKE_CALLS`, the worker contract, the Windows refusal. Tests on both
  sides; the sandbox check.
- **7b: custom tools calling custom tools.** The chain, depth, cycles, time,
  the request budget; dependencies when adding and offering a tool.
- **7c: the page.** Run as (agent picker and hosting slot), the declares line,
  the pill, the bell text, and the New tool check prompt.
- **7d: asking the person.** `asks` and `captain.ask`, the question store and
  its two routes, the dialog in every layout with the waiting check, the
  paused time limit, the agent following its run instead of one long request,
  Needs-you in Line, the nobody-to-ask answer, recorded answers.
- Then a live check on a throwaway deck with a running agent: a tool that
  reads a page, calls another custom tool and asks a question, run from chat
  and from Try it.

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

What holds since Phase 6, on Linux with Landlock: a tool reads only its own
files, its Python and the system's folders, writes only its call's folder, has
no TCP unless it declares it, and can't signal or reach other processes. On a
host without Landlock (macOS, Windows, an old kernel) a tool still runs as
Flight Deck's OS user and can reach what that user can (other users' folders,
the deck's database) — the same reach an approved Crew worker has — unless the
admin turns on "Run tools only in the sandbox", and then that host runs none.
Settings says which it is.

What Phase 7 adds (planned):

- A tool reaches an agent only through its own run's socket, and an agent acts
  only for a run it is waiting on. Flight Deck checks every request against
  what the tool declared and against the risk class of the real call; the
  agent's own guards check it again.
- Never: run code, money, unclassified tools, and tools that change the agent
  or set up later work. Send only when the owner runs the tool. Web reads only
  from the declared hosts.
- A shared tool with `uses` acts as the person's agent within what it
  declared: it can read what those tools read (their mail, if it declares
  `google_mail` read) and write what they write. Adding the tool is the
  consent, with each use and its why shown. Because new versions follow
  automatically (settled), new uses take effect at once and the bell comes
  after.
- Requests cost the runner's model tokens when the agent's tool uses a model;
  the 50-request cap and their daily allowance bound it.
- A question puts the tool author's words in front of the runner. The dialog
  says whose tool asks, has no password fields, and warns against typing keys;
  answers go only to that run.

## Still open

- **Windows desktop**: the launcher passes the exit code on, but no limits apply
  there and the process-group kill is `proc.kill()` only. Untested.
- **Not tried with a real model**: an agent calling `custom_tools` (list,
  run, build, propose) and drafting a proposal after a turn. The routes and
  the tool are tested over HTTP; Crew building a tool with a real CLI was
  checked (Phase 2).
- **The sandbox on a real deck**: checked in a Linux container, not yet on
  the Pi or the glasses deck. `python3 tests/test_flight_deck/test_xtool_sandbox.py`
  there says.
- **UDP** isn't ruled by Landlock: a tool with `"network": false` can still
  send UDP (DNS included). seccomp could close that.
- **Phase 7 before building:** whether an agent's turn has a wall-clock limit
  that a tool waiting for a person's answer (up to an hour) would hit; and
  which actions of the allowed tools reach other people, for
  `tool_risk._SEND_ACTIONS`.

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
