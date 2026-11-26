# Captain on Harbor benchmarks

[Harbor](https://github.com/laude-institute/harbor) runs agents on containerised
tasks. ShellBench and Terminal-Bench are written in its task format. This folder
holds a Harbor agent, `captain_harbor:Captain`, that runs Captain on those tasks.

## What a trial does

1. Harbor builds the task's container. The adapter installs Captain into
   `/opt/captain-claw` with uv and its own Python, without touching the image's Python.
2. It runs `captain-claw-once` in the task's working directory (usually `/app`).
   That command:
   - starts a **private Flight Deck** in the container (loopback, sign-in off,
     own data folder);
   - has the deck spawn a process agent whose workspace is the task folder;
   - sends the task through the deck's inbound MCP (`send_task`, then
     `get_result`), the same way Claude Code hands work to an agent;
   - reads the transcript and the agent's recorded LLM usage back, compares the
     workspace before and after, and writes `captain-result.json`.
3. Harbor's verifier grades the files the agent left, and the adapter reports
   tokens and estimated cost to Harbor.

The agent's files land where it asks for them (`CLAW_WRITE_DIRECT`). In the app,
writes go under `saved/` instead, and a grader would never see them there.

The task arrives as an automated MCP turn ("not a live message from the user"),
so the agent acts on it instead of asking clarifying questions that nobody
would answer.

## Run it

```bash
pip install harbor
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY
PYTHONPATH=evals/harbor harbor run -p path/to/tasks \
  --agent captain_harbor:Captain -m anthropic/claude-haiku-5-5 -k 3 -n 4
```

**By default, a Harbor trial runs Captain at full stretch.** Reasoning effort is
`max` (the highest level the model offers, so `high` on a model without `max`)
and Drive is `ambitious`. Both are **locked** for the whole run:
- every main model call uses them;
- the agent's `reasoning_effort` and `drive_level` tools refuse to change them;
- housekeeping calls keep their quiet defaults.

To change this, pass `--ak reasoning_effort=high`, `--ak drive=persistent`, or
`auto` to leave a setting to Captain. `--ak lock=false` sets the levels as the
agent's defaults without forcing them. `captain-claw-once` reads the lock as
`--lock` and passes it to the agent as `CLAW_LOCK_REASONING_EFFORT` and
`CLAW_LOCK_DRIVE`.

By default the agent is a plain new agent, like **New agent → Advanced** with
nothing chosen. `--ak archetype=<id>` spawns an agent from the deck's **agent
store** instead, the same call the store's Add button makes. It gets that store
agent's tools, mode and instructions, but still runs the model given with `-m`.
If the store agent's tier switches the model, the runner sets it back and
restarts the agent before sending the task. `captain-claw-once --list-store`
prints the store's ids, with the Featured ones first.

Reasoning effort and Drive are set on the agent through the deck, as its own
defaults (Settings → Reasoning effort / Drive). The MCP task counts as an
automated `api` turn, so both defaults apply to it:

- `reasoning_effort`: `off` · `minimal` · `low` · `medium` · `high` · `xhigh` ·
  `max`. A level the model doesn't offer is clamped to the nearest one it does.
  Unset: the model's own default.
- `drive`: `relaxed` · `diligent` · `persistent` · `ambitious`. At `persistent`
  and above, the agent checks its result before delivering it. Ambitious's
  extra unrequested step runs only on a person's own turns, never on this
  automated task. Unset: whatever the agent's cognitive mode suggests, else
  `diligent`.

The model is given as `provider/model`, using Captain's provider names. For
example, `openrouter/deepseek/deepseek-v4-pro` means provider `openrouter` and
model `deepseek/deepseek-v4-pro`. A web-search key (`BRAVE_API_KEY`, …) set on
the host is passed to the agent too.

Until a Captain release includes `captain_claw.bench`, install from a wheel of
this checkout. Pin its dependencies to the locked versions, so every trial runs
the same code you tested; the newest versions of everything can behave
differently:

```bash
uv build --wheel -o dist/
uv export --frozen --no-hashes --no-dev --no-emit-project --format requirements-txt -o dist/constraints.txt
PYTHONPATH=evals/harbor harbor run ... \
  --ak source=$PWD/dist/captain_claw-0.9.3-py3-none-any.whl \
  --ak constraints=$PWD/dist/constraints.txt
```

| `--ak` option | meaning |
|---|---|
| `source` | pip spec: `captain-claw==0.9.4`, a git URL, or a local wheel path (default: the latest release) |
| `constraints` | pip constraints file pinning dependencies (see above) |
| `reasoning_effort` | `off` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max` · `auto` (default `max`) |
| `drive` | `relaxed` · `diligent` · `persistent` · `ambitious` · `auto` (default `ambitious`) |
| `lock` | force effort and Drive for the whole run (default `true`) |
| `max_iterations` | tool-call budget for the turn |
| `timeout_seconds` | stop the task after this long (30 min at most) |
| `workspace` | folder the agent works in (default: the container's working directory) |
| `archetype` | spawn the agent from the deck's agent store, by id |
| `standalone` | `true`: a bare in-process agent, no Flight Deck (for comparison) |

Each trial's `agent/` log folder contains:

- `captain.txt`: progress lines and the answer;
- `captain-result.json`: answer, usage, cost, changed files, events, transcript;
- `captain-run/`: the deck's and the agent's own logs.

## Without Harbor

The same runner works on any folder:

```bash
cd some/task/folder
captain-claw-once -p anthropic -m claude-haiku-5-5 --instruction-file ../instruction.md
```

It prints the answer and writes `result.json` to a temp run folder
(`--run-dir` chooses it). `--deck-url` uses a deck already running on the same
machine instead of a private one. Spawning an agent there needs either sign-in
turned off on that deck or a browser sign-in token (`--deck-token`). A personal
access token (`cc_pat_…`) can't spawn agents.

## Tests

```bash
python -m pytest tests/test_bench_run_once.py tests/test_bench_deck_run.py
PYTHONPATH=evals/harbor python -m pytest evals/harbor/test_captain_harbor.py   # needs harbor
```
