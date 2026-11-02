# Captain Claw v0.8.5 Release Notes

**Release title:** All Hands on Deck
**Release date:** 2026-11-02

0.8.4 taught an agent your work. 0.8.5 gives it hands: a crew that builds things,
and a team that forms around what you ask for.

**Crew** puts coding CLIs to work for you: **Claude Code, Codex, OpenCode and Pi**,
each in a git worktree of its own, several at once. Talk to your agent ("have Codex
fix the login test and Claude add dark mode") or use the Crew page. Flight Deck
watches them without spending tokens and tells you when one finishes or needs a
decision, on the channel you asked from. Nothing lands until a person says so.

**Forge agents** is for people who know what they need but not which agents would do
it. Say it in your own words and attach your files. Captain plans a small team of
standing agents, you approve it, and it recommends what should run on a schedule and
what software is worth building. Crew builds those apps to a fixed contract, and your
agents connect to them.

**A deck updates itself.** **Console → Updates** tries the new version on a spare port,
warns everyone with a two-minute countdown, stops the agents, pulls, restarts in place
and goes back by itself if the new version doesn't come up.

**The Taskboard takes many tasks at once.** Tick them as in Gmail and give one instruction
for all ("thank them and politely decline"). The agent works through them one at a time, each
with its own details, and you can stop the rest at any point.

Also in 0.8.5: real **Excel and PowerPoint** without scripts (`xlsx`, `pptx`); **agent
groups** in the kiosk; a Users page with last login and tokens; and a new look for the devices,
with Captain as the only character.

49 pull requests, three days after 0.8.4.

**Restart Flight Deck, then every agent.** See [Upgrading](#upgrading). **Approving someone
for Crew trusts them with the server:** [read this first](#before-you-approve-anyone-for-crew).

## Highlights

### Crew: how a task flows

Open **Crew** from the sidebar, or in the kiosk from the agent column or **Settings**, or just ask your agent.

**1. Ask.** Tell your agent, in any channel, or press **New task**. Pick a CLI and say what's
wanted. **Change the code** (a `ship` task) works on a branch; **Only look, and report** (a `scout`
task) changes nothing. The same brief, for the same CLI in the same place, while the first is still
active, is that task rather than a second worker.

**2. Say where.** **A project**, **a repository address** or, for admins, **a folder here**.
Crew clones an address itself, in the background: a wrong address is refused at once, the task
shows **Getting the repository** with how far it has got, and a 216 MB clone doesn't hold anything
up. `crew action=clone` gets a repository here with no work and no tokens.

**3. Say what it runs on.**

- **Its own:** the CLI's own login or, for OpenCode, its free models.
- **An agent's model:** the model one of your agents is set to, on that agent's endpoint and key.
- **A deck tier:** a tier from your models in Library, or the team's.
- **Enter one:** an API address, a model name, a context size and a key, for this task only.

With the last three the CLI needs no login, and gets none. The worker gets the deck's address and a key
that works only during its current turn, and the deck puts the real key on the way out. A worker told to
print its environment has nothing to print, so a team's key is safe to use this way. Tokens count toward
your daily limit the way an agent's do: only what missed the model's prompt cache, plus its output. Your
agent can do this too, when you ask: `runs_on` takes `me`, `agent:<name>` or `tier:<name>`. A key is never
typed through a chat.

**4. Give it what it needs.**

- **Files.** Paste a screenshot, drop files or use the paperclip, on New task and in any later message, up
  to 50 MB each and 30 files or 200 MB a task. Send your agent a screenshot and say "have OpenCode fix this",
  and it passes it on.
- **Folders of your files**, as many as you like. **Read** gives the worker its own copy, remade before each
  turn. **Read and write** works in your real folder, at once, with no undo. Google Drive and OneDrive
  folders aren't offered, and a folder shared with you to view stays that way. Why Read is a copy: OpenCode
  wrote straight through a read-only link when we tried it.
- **An environment to run and test in, for any project.** Prepare it once, in the project's main folder: by
  hand, in a **terminal in the browser**, or with **Prepare it with**, where a worker works out what's needed,
  installs it, checks it runs and reports its test and start commands. Every later task gets its own instant
  copy of package folders, the shared Python environment with `PYTHONPATH` on its own code, and its own
  ports (`CREW_PORT`) and temp folder. Captain never runs anything a repository brings with it.

**5. It works, and you can step in.** Flight Deck watches from disk and spends no tokens. A worker outlives a
deck restart.

- **Send now** (Enter) stops what the worker is doing, including any command it had running, and it goes on with
  your message: its files stay. **After this turn** (⌘/Ctrl+Enter) waits for its turn to end.
- **A line while it's quiet.** After a minute with nothing new, the page says what it's waiting on: the model
  writing its reply, with how long and how much has come in, a command running, or a failed call being retried.
- **It rides out a bad connection.** Each CLI asks again by itself, and that shows as a grey note, not a red
  error. If it gives up or goes quiet, Captain starts it again from where it stopped, after 20 seconds, a minute,
  then three. A refused key or a spent limit isn't retried. A worker running a long command is never restarted
  for being quiet.

**6. It finishes, and your agent hears.** The agent is called back on the channel the task came from. A worktree is out
of an agent's reach, so the results come to your VFS: `vfs:crew/<project>/<task>/` has `report.md` and `files/`. The
task's **Files** tab shows them, with no `.git`, and only what was committed.

**7. A person lands it.** Every task works on its own branch and never pushes. **Merge**, or Approve in the bell on a merge
your agent asked for. Captain pushes and opens pull requests itself, with the GitHub token in **Crew → Logins**, and
merges them too: a worker never sees that token. Work that hasn't landed is never removed unless you say so. **Tidy up**
clears finished tasks off a project's page.

**Put it online.** **Live sites** lists everything you've published, whoever did it: **Stop** takes it offline and keeps its
address, **Start** brings it back, and **Publish** puts up what a task made or one of your folders. On a deck with sign-in
only an admin can run an app; a page is anyone's.

**Working on Captain itself.** Point a task at this deck's repository or your fork, land it through a pull request, then
**Console → Updates** installs it, or your agent asks you to Approve that with `crew action=update_deck`.

#### The `crew` tool

| Action | What it does |
|---|---|
| `options` | The CLIs you may use, their logins, the model each last ran on and its limits (like Claude's weekly limit), the models `runs_on` can name, and the projects |
| `dispatch` | Start a task: `harness`, `brief`, one of `project`, `repo_url`, `repo_path`, and optionally `kind`, `deliver`, `model`, `runs_on`, `effort`, `attachments`, `vfs_read`, `vfs_write` |
| `clone` | A repository here and no work yet: no worker, no tokens, and a callback when it's ready |
| `env`, `prepare_env` | What's installed for a project, and a worker that sets it up |
| `list`, `status`, `files` | What's going on, one task, and where its results are |
| `steer` | The next instruction, or an answer to a worker's question; `now=true` stops it to hear this at once |
| `folders` | Which VFS folders a task has, from its next turn |
| `stop` | Stop a task and keep its work |
| `pr` | Captain pushes the branch and opens the pull request |
| `request_merge` | Ask the person to Approve a merge |
| `update_deck` | An admin's agent asks them to Approve updating this deck |

The agent is told to report what git shows, not what a worker claims. A task's id exists only once a `crew` result gives it,
and a made-up one is caught. It can't merge or discard work through this tool: a person's Approve does that.

#### Who can use it

- **An admin approves it, per user.** On **Users**, open someone's row and tick Crew: which CLIs, and how many at once. It's off for
  everyone else. Withdrawing it stops what's running at once. Admins, and a deck with sign-in off, can always use it.
- **The browser terminal has its own tick** for a non-admin. `FD_CREW_TERMINAL=off` takes it from everyone.
- **Each person's own logins**, in **Crew → Logins**: a `claude setup-token` token or an Anthropic key, an OpenAI key or Codex's
  `auth.json`, provider keys for OpenCode and Pi ("Your own AI keys" count), and a GitHub token. A non-admin's worker never falls back
  to the deck owner's login, and its environment holds none of the deck's secrets. OpenCode's free models need no login at all, though
  a task with no key must name its model: your code goes to whichever model runs, and [they differ in what happens to it](https://opencode.ai/docs/zen/).
- **Your own model.** Crew → Logins → *Your own model* runs OpenCode and Pi on your llama.cpp or Ollama. The key is in no file.
- **Models and limits.** The page shows the model each CLI last ran on and its account's limits, as of its last run. Claude Code reports
  both, Pi its model, and Codex and OpenCode neither.

Install the CLIs on the deck's host: Crew → Logins shows which are there and the command for those that aren't. OpenCode 1 and 2 both work.

### Forge agents: say what you need, get a team

In the kiosk and the simple layout, **New agent → Forge agents**. Close the dialog at any point: it carries on, and the bell opens it again.

1. **Describe.** Your request in your own words, and up to 10 files of 25 MB each. **Plan my team** starts it.
2. **Review.** Vatra's planner opens your files, the Lead picks the fewest standing agents that cover the request, and the Long Horizon Planner writes
   each one's job and may ask you questions. Rename an agent, change what it's based on, rewrite its job or leave it out. **Update the plan** starts
   from your edits. Nothing exists until you press **Create N agents**, and a plan never exceeds the room your agent limit leaves.
3. **Create.** Each agent gets its archetype's way of working plus its job and its teammates. The team shares a folder (`vfs:<team>/`, your files under
   `inputs/`) and one set of data tables, and teammates reach each other with `consult_peer`.
4. **Finish.** Captain recommends what should run by itself and what software is worth building, with why and how you'd use each. Only what you keep gets made.
   - **What runs by itself** becomes scheduled tasks, listed under Scheduled tasks like any other.
   - **Software to build** goes to **Crew** as ordinary tasks (**Build it with Crew**), so Crew's approval and limits apply, and you land the work there. Every app is built to a
     [fixed contract](https://github.com/kstevica/captain-claw/blob/main/captain_claw/instructions/forge/app_contract.md): an HTTP API under `/api/agent/` with a bearer
     key and a manifest, a WebSocket at `/agent/ws` that speaks the agent link protocol, or both.
   - **Connect your agents** works once the app is running: give its address and key. Flight Deck reads the manifest, registers the API as `direct_api` calls on each agent
     you choose, and makes an outbound agent link to the socket. The key stays in the sealed link and the agents' stored calls, never in the plan or their instructions.

**Agent Forge** is still there in the full layout, for designing a team around a goal and spawning it in one click. Forge agents goes further, to a standing team with
its schedules and its software, and it's written for someone who has never seen an archetype.

**An agent now knows its job with no one in its chat.** Instructions used to arrive only with a browser chat, so an agent nobody had opened, or one restarted by idle sleep,
ran scheduled and linked turns with no job. Flight Deck now tells it before each scheduled turn and before a linked system's first turn after it starts. A link also follows its
agent when a restart moves it to another port. Both need accounts on the deck: with sign-in off, instructions live in the browser.

**Known gaps.** It has run end to end with a stand-in model, CLI and app; a real model's plan and a real worker building to the contract have yet to be tried together. A team's
tables live in `vfs:<team>/.datastore/store.db`, which the Data library doesn't list. A non-admin can't host a runnable app, so Connect needs an admin or another host for it.
Agents come from the archetype catalog; no new ones are minted. An admin managing a user's agents doesn't get it in that dialog.

### The Taskboard: many tasks, one instruction

0.8.4's board handed over one task at a time. Now it works like Gmail here too.

- **Pick several.** A tick box on every row, `x` on the open task or the cursor's row, Shift+click for a run, the list's own box, and `*` then `a` for all shown or `n` for none.
- **One move for all.** A bar over the list offers **Hand to agent**, **Done**, **Snooze**, **Next move**, **Not a task**, **Reopen** and **Got it**, and the same keys act on what's ticked. Each applies as if made on its own, with Undo for the lot.
- **One instruction, one at a time.** **Hand to agent** takes one set of words for all: "thank them and politely decline". Every task leaves **Needs me** at once and waits in a line. The agent takes them in order, each in a session of its own with only that task's details, read when its turn comes.
- **You see how it stands.** A line over the board says how far it has got and which task it's on. **Stop the rest** sends what still waits back to you, and the run under way finishes.
- **It's your own request**, so it writes email or messages if that's what you ask, and your Email sending switch and its recipient list still apply. When the agent is through, each task is read and [checked like any hand-over](https://github.com/kstevica/captain-claw/blob/main/release-notes/RELEASE_NOTES_0.8.4.md#the-taskboard-where-everything-stands).
- **Things that change along the way.** A task you closed, merged or moved on before its turn is passed over and comes back saying why, as it does if the agent is off or the allowance is spent. A run that fails stops nothing else. A deck restart goes on with what waited, but the run it cut isn't sent again, since it may have sent its email, and its task says so.
- **Not in bulk:** nothing approves or rejects a next step, and there's no bulk merge. Each is read on its own.

Also: a reply that lands on a task already waiting on you is no longer closed over.

### A deck updates itself

**Console → Updates → Check now**, then **Update now**. It follows the branch it's on, so a fork follows the fork, and it only moves forward, to the commit you saw.

1. **Test start.** The new commit runs from a temp folder on a spare port, with an empty data folder and none of the deck's keys. If its page lacks files or its code doesn't import, nothing has been touched.
2. **Warning.** Everyone signed in gets a toast, then one they can't close, counting down two minutes.
3. **Stop the agents.** Mid-turn ones can be waited for, up to 5 more minutes. Crew's workers keep going.
4. **Pull.** New packages went in before the test start, so you don't run `pip` yourself.
5. **Restart in place**: the same process, command line and tmux pane, so it can't lose the environment you started it with.
6. **Check.** If it doesn't come up, the old commit goes back and the deck starts on that, by itself.

Every admin gets an email with the date and time in the subject, from the Gmail or Outlook of the admin who started it or else the first admin who has one. **Automatically** is **Off** by default, or
**Tell me**, or **Install by itself**: with that last one, whatever is merged to the branch goes live once it passes the test start.

A desktop build or a container has no checkout, so the page says to update it the way it was installed. The first update to this needs one pull and restart by hand.

### Two new tools for real office files

| Tool | What it does |
|---|---|
| `xlsx` | Make, read, edit and check Excel workbooks with no scripts. Only what a change touches is rewritten, so charts, pivots, sparklines and macros stay. Formulas stay live and carry results from about 200 functions, compared with Microsoft Excel. Every call lists each cell it changed and each total that moved. Inserting a row rewrites what Excel would. `check` finds errors at their source, broken patterns and a `SUM` that stops one cell short. Functions that spill (`FILTER`, `LET`) aren't written, and a formula it can't calculate is reported rather than left with a stale number. |
| `pptx` | Make, read, edit, check and render PowerPoint decks with no scripts. Slides come from the deck's own layouts and fill their placeholders, so a firm's template keeps doing the design. Text is measured against PowerPoint's own results, and every result says what doesn't fit. Native charts and tables. Pictures in any format or from a link, placed where they work, with captions and alt text. `render` needs LibreOffice. |

`write` and `edit` refuse these files and point to the tools. A user's upload is never changed: the result is a copy. With `crew`, that makes **68 built-in tools** per agent. `crew` shows only for an owner an admin approved.

### The kiosk

- **Agent groups.** Folders in the agent list, with a name and a colour, that fold away and still show an agent at work or a new reply. Drag an agent onto a group, or use its **⋯ → Group**. **Pin** one group to the top of the list, and as many agents as you like to the top of their group. Each person has their own, on all their devices.
- **A page left open stays signed in.** Crew and the libraries renew their sign-in rather than showing "Token expired".

### Running a deck

- **Users page.** Last login with sorting, tokens used this month and all time (no prompt-cache reads, as the daily allowance counts them), and **Ask for feedback**, an editable email from your own Gmail or Outlook with the user's name first in its subject. A shared agent's member turns count against its owner.
- **Hosting from an agent works.** `hosting` was refused with "agent routes require a bearer token" when the session's address went through a proxy or under `FD_LOCKDOWN`, and its `stop` and `start` always answered 401. Agents now call their own deck with their own identity and get a public link back, not a `localhost` one. A static site really stops now: its address answers 404 and comes back on Start.
- **Confinement.** The `knowledge` tool's path and the file `payments` reads now follow the deck's path rules.
- A [manual test checklist](https://github.com/kstevica/captain-claw/blob/main/docs/TESTING_CHECKLIST.md) for the dashboard, for anyone trying it on a clean machine.

### Devices

- **Captain is the only character**, on every board: the 18-character choice and its menu row are gone. The robot is drawn from the app icon's geometry at each size, with the lamp and eyes moving on top, and its face shows the agent's state.
- **StopWatch: a new look** (0.20.0). Black, gold, cyan and coral. Captain above the time on a dial with the seconds going round; the voice as bars while listening; what the agent is doing while it works; an answer's pages as an arc.
- **StopWatch: no automatic stop** (0.20.1). A 2.5 s pause no longer ends a recording. Press or tap when you're done, however long you pause. The 110 s limit stays.
- **PaperColor B** (1.0.24 to 1.0.30). A dashboard in blocks of colour, the answer as a conversation, an icon beside each button instead of a legend, recording that works offline, a white pass only hourly and at power-on, and B held 4 s to switch off, leaving a capsized robot.
- **Waveshare ePaper 1.54** (0.19.12 to 0.20.3). A new home screen with note mode at a glance, a tap on the tile to talk on the touch board, stop and cancel icons while recording, and the same capsized robot when switched off.
- **A weak link no longer clogs the device** (0.20.4 / B 1.0.31). Buttons, screen, mic and speaker share one loop, and a weak Wi-Fi link holds it up. The device now joins the strongest access point by its address, a task keeps the mic recording into a 4 s ring while the loop is behind, and speech that runs out waits for a second of it, so one gap instead of a stutter. Status shows the access point, its channel, the loop's slowest pass and how often the voice ran out. Diagnosed from the code, and not yet reproduced on a device.
- **Firmware 0.20.4** / PaperColor B **1.0.31**.

### What's still open

- **Merging is enforced at the `crew` tool.** A non-admin's agent has no shell, so for them that's the whole way to a merge. An admin's agent has a shell until agents run in an OS sandbox, so for it the rule is one it's told to follow, not one it can't break.
- **Models through the deck** have run with all four CLIs on Ollama, which stands in for all three kinds of API, and with OpenCode on DeepSeek. Direct Anthropic, OpenAI, Gemini, xAI and OpenRouter endpoints haven't been run this way yet.
- **On macOS**, a command a worker detached and left running at the natural end of a turn isn't caught, since macOS doesn't show a process's environment. **Stop** and **Send now** catch it on every system.
- **Old worktrees aren't cleaned up automatically**, and Crew runs on the deck's own host only.
- **Taskboard batches** have been run end to end with stand-in agents, not yet with a real agent, Gmail and Jev together.

## Before you approve anyone for Crew

**Approving someone trusts them with the server.** A worker runs commands as the same system user as Flight Deck, so it can reach what that user can, including other people's files and the deck's database. This is the reach [0.8.4's confinement](https://github.com/kstevica/captain-claw/blob/main/release-notes/RELEASE_NOTES_0.8.4.md#security-a-deck-for-people-who-dont-share-a-machine) took away from non-admins' agents, handed back on purpose to someone you name. No OS sandbox holds a worker to its worktree yet ([the plan](https://github.com/kstevica/captain-claw/blob/main/docs/agent-os-sandbox-plan.md)); Codex runs in its own `workspace-write` sandbox and the others don't. The approval dialog says so and needs a tick.

So Crew is off for every non-admin until an admin ticks it. Admins, and a deck with sign-in off, can always use it. `FD_CREW=off` turns it off for the whole deck.

## Upgrading

1. **Each Flight Deck** — pull, and `pip install .` where Captain Claw is installed non-editable. There are no new Python packages since 0.8.4.
2. **Restart Flight Deck, then every agent.** Agents need the restart for `crew`, `xlsx`, `pptx` and the `hosting` fix. Then hard-refresh every browser. From now on **Console → Updates** does this for a deck on a git checkout.
3. **For Crew** — install the CLIs you want on the deck's host, tick Crew per user on **Users** (see [above](#before-you-approve-anyone-for-crew)), and each person saves their logins in **Crew → Logins**.
4. **Devices** — publish the new firmware from `firmware/device` with `python tools/publish.py <env> --deck https://<deck>`; devices take it from *Menu → Software update* or by themselves.
5. **The desktop app** embeds its own deck and bundle, so it needs a new build to get any of this.

BotPort hubs need nothing new: 0.8.5 keeps protocol 2, and BotPort's code is unchanged since 0.8.2.

**Changed**

- The StopWatch doesn't end a recording on a pause, and devices have no character to pick.
- The Users page shows tokens, not dollars.
- Scheduled and linked turns now carry the agent's instructions, on a deck with accounts. An agent that ran with no job before now runs with its job.
- A stopped static site answers 404 until you Start it again.

**New settings**

| Setting | Default | What it does |
|---|---|---|
| `FD_CREW` | on | `off` turns Crew off for the whole deck |
| `FD_CREW_TERMINAL` | on | `off` takes the browser terminal from everyone, admins too |
| `FD_CREW_CLAUDE_PERMISSION` | `auto` | Claude Code's permission mode for workers: `auto`, `acceptEdits` or `bypassPermissions` |
| `FD_CREW_MAX_RUNNING` | `8` | Workers running at once on this deck |
| `FD_CREW_MAX_TURN_HOURS` | `6` | The longest one turn may run |
| `FD_CREW_STALL_MINUTES` | `15` | Silence before you're told a worker seems stuck |
| `FD_CREW_WORKING_SECONDS` | `60` | How often the still-working line refreshes |
| `FD_CREW_MODEL_SILENCE_SECONDS` | `300` | How long a model server through the deck can say nothing |
| `FD_CREW_ERROR_WAIT_SECONDS` | `300` | How long a CLI gets to show it's past a failed call |
| `FD_CREW_RETRY_WAIT_SECONDS` | `20` | The first wait between tries |
| `FD_CREW_CLAUDE_BIN`, `_CODEX_BIN`, `_OPENCODE_BIN`, `_PI_BIN` | found on `PATH` | Where a CLI is, if Flight Deck can't find it |
| `FD_UPDATE_REPO` | worked out | The checkout, when the deck can't work it out |
| `FD_UPDATE_POLL_S` | `900` | How often **Tell me** and **Install by itself** look |
| `FD_UPDATE_COUNTDOWN_S` | `120` | How long everyone gets before the restart |
| `FD_UPDATE_BOOT_TIMEOUT_S` | `300` | How long the new version gets to come up |
| *Crew* (per user, on Users) | off | Which CLIs, how many at once, and the terminal |
| *Automatically* (Console → Updates) | Off | Off, Tell me or Install by itself |

New stores create themselves on start: `crew.db` and `<data>/crew/` for Crew, `<data>/update/` for updates. Forge plans live with Vatra's runs. No manual migration.
