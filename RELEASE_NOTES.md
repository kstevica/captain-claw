# Captain Claw v0.9.0 Release Notes

**Release title:** Shipyard
**Release date:** 2026-11-09

Your agents can now get tools built for them. Describe a tool, and Crew's coding agents write it, test it
and publish it to Flight Deck, which runs it for your agents in a sandbox on Linux. Agents also fit in one
`.captain` file now, so you can back them up, move them to another deck or hand a trained one to a client
and update it later. They hand each other work, answer email meant for them, read any IMAP inbox, learn
from your corrections once you approve, share one external memory, and know the goals their work is for.

52 pull requests, #436 to #487. **Restart Flight Deck, then every agent, and hard-refresh the browsers.**
Two new things are on by default (lesson proposals, and agents picking up email that names them); see
[Upgrading](#upgrading).

## New

- **Custom tools: Python tools of your own, built by Crew and run by Flight Deck.** Off by default: an
  admin turns them on in **Admin console → Settings → Custom tools**. Then **Custom tools** is under
  **Multi-Agent** in the full layout and in the kiosk's **Settings** menu, with **Mine**, **Shared with me**,
  **Everybody** and, for admins, **Approvals**.
  - **Making one** (for people approved for Crew): **New tool**, then **Name**, **Title** and **What it
    should do**, then **Check it**. Your own model returns a fuller spec, with a reason for each change and
    an **Already there?** warning when agents can already do it. Keep **Checked** or **Yours**, pick a coding
    agent and **Build it**. The worker writes the tool, its tests and its examples in the tool's own git
    repository. When you approve the merge, Flight Deck publishes it in the background, and it goes live
    only if the tests and every example pass; otherwise the old version stays live. The last three versions
    can be made live again.
  - **Agents use them** through the new `custom_tools` tool (`list`, `describe`, `run`, `build`,
    `improve`, `propose`). Flight Deck runs every call, never the agent: one process per call, JSON in and
    out, as the person whose turn it is, with that person's own secrets (**Your secrets for it**) and files,
    and nothing else from the deck. Calls get 60 seconds (300 at most) and 1 GB of memory by default, and
    land in the agent's tool-call register.
  - **Agents suggest tools.** After a long turn of repeated, mechanical work, an agent's own model may draft
    one. It waits in **Your agents suggest** on your page, and nothing is built until you click **Build
    it**.
  - **Offer a tool** to **People…** or to **Everybody on this deck** (which waits for an admin's approval
    unless that is turned off). An offer runs for nobody until they click **Add**, which is their consent
    to what the tool says it needs. Others can run it with their own secrets, but never change it or see
    its code, versions or logs. When a new version asks for more, everyone who added it gets a bell.
  - **A tool can ask for more**: the agent that ran it uses its own tools for it (declared in `tool.json`
    with a reason each; sending mail or messages only when the tool's owner runs it; never code, money or
    changes to the agent), it runs your other custom tools (3 levels deep at most), and it asks the person
    running it a question in Flight Deck. **Try it** has **Run as** for tools that call an agent.
  - **The sandbox.** On Linux 5.13 or later, each run is held in a Landlock sandbox: it reads only its own
    version and the system folders, writes only to the call's folder, has no TCP unless `tool.json` says
    `"network": true`, and can't signal anything outside. **macOS, Windows and the stock Raspberry Pi 5
    kernel have no Landlock**, so tools there run as Flight Deck's own user and reach what it can; the
    admin card says which kind of host you have. **Run tools only in the sandbox** stops every tool on a
    host without one. Installing a tool's own packages at publish time runs outside the sandbox.
  - Details and what's still open: [docs/guide/flight-deck.md#custom-tools](https://github.com/kstevica/captain-claw/blob/main/docs/guide/flight-deck.md#custom-tools).
- **`ask_user`: an agent asks you, in Flight Deck.** One question (text, one choice, several, yes/no) or a
  short form of up to 10, shown as a dialog in every Flight Deck tab of the person the turn is for, owner
  or member. The bell says *‹agent› asks you something*. The turn waits 10 minutes by default (up to an
  hour); **Send** or **Don't answer**, and either way the agent says what it assumed. Automatic turns
  (schedules, flows, other agents, Autonomous Work, the Taskboard, the API, agent links, exams) have nobody
  to ask, and turns from Telegram, WhatsApp, glasses, devices or public links ask in their reply instead. Custom tools ask through the same dialog.
- **Agents in one file: export, import, packages and backups.** Any process agent you own fits in one
  `.captain` file. In the kiosk: the agent's **⋯ → Back up / export**, and **New agent → From a file**.
  - **What goes in**: **Template** (who it is and what it was taught), **Trained** (and what it learned
    since, without conversations and files) or **Full backup** (everything). Before a Template or Trained
    file leaves, you look over what it learned: insights usually about you or your clients start unticked,
    and **People it knows** is a separate tick.
  - **Never in the file**: its `.env`, keys and tokens, connections, logs and your own profile. Connections
    travel by name, and the import lists what to set up again. A Full backup can carry its keys **sealed
    with a passphrase** that Flight Deck doesn't keep.
  - **Bringing one in** shows what the file holds, who made it, and whether it is **Signed by this deck**,
    by a deck you trust, or unsigned, before anything is made. Make a new agent on **Its own model** or
    **One of your models**, or pour the file into an existing agent: **Add to what it has** or **Replace
    it**. A backup is taken first, kept a week, with **Undo**. Scheduled tasks arrive switched off.
  - **Packages and updates**: name a Template or Trained export under **Updates later** and each export
    becomes its next version. Bringing in a newer one offers **Update to version N**, which swaps the old
    version's items and keeps what the agent learned or what was changed there.
  - **Scheduled backups**: kiosk **Settings → Agent backups**, daily or weekly, keeping 1–60 per agent,
    with an optional copy into your files, and **Restore it** on any of them.
  - **Signed files**: every deck signs its bundles; a changed file is refused. Admins list **Decks this one
    trusts** under **Admin console → Settings → Agent files** and can take only signed files.
- **Agents as colleagues.** An agent hands a piece of work to another of your agents with the new
  `colleagues` tool (`list`, `hand_over`, `handed`) and carries on with the answer. Each hand-over is its
  own Taskboard task under **Between your agents**, and the answer goes back to the agent that asked, in
  the same task. Authority never grows along the way: the turn's mail permissions travel with it, read by
  code. At most 4 agents in a chain and 60 hand-overs per owner a day.
- **Agents with their own address.** **⋯ → Identity & email** gives an agent a **Role**, **What it
  does**, **Its own inbox**, **Addresses that land in your inbox** (`you+researcher@gmail.com`), **Names it
  answers to** and **Colleagues it may hand work to**. Every 60 seconds Flight Deck reads your Gmail,
  Outlook and IMAP inboxes, and mail sent to an agent's address, naming it in the subject, or greeting it
  ("Hi Rita,") becomes its task under **Email for an agent**. Automatic, bulk, list and no-reply mail is
  skipped. Those turns write drafts only, unless you allow more.
- **Gmail replies sent, when you say so.** **Replies in your Gmail** per agent: **Drafts** (the default),
  **Send to you** (when the email came from you) or **Send to anyone** (when Gmail verified the sender).
  Only that one reply to that sender, with no cc or bcc, through your Google email-sending switch,
  allowlist and daily limit. Outlook replies stay drafts.
- **Any mailbox, with IMAP.** **Connections → Email (IMAP)** connects up to 20 IMAP/SMTP inboxes per
  person: a company mail server, Fastmail, iCloud, Yahoo, Zoho, GMX, Gmail with an app password, or Proton
  Mail through Bridge. Type the address and the server settings fill in. The new `imap_mail` tool lists,
  searches, reads, drafts, files and sends; it never deletes. The password is sealed and the mail work
  happens inside Flight Deck, so it never reaches an agent. **Sending is off** until you tick *Let my
  agents send email from this inbox*, with **Only to these recipients** and a **Daily limit**. The
  connection test takes seconds, and when a host blocks port 465 it offers **Use port 587 (STARTTLS)**.
  Needs sign-in on, or the desktop app.
- **Lessons from your corrections, approved by you.** When you correct an agent or state a standing rule
  ("too long", "always CC Marko"), it drafts one lesson. **Lessons to approve** (a badge beside the bell,
  and a card in each kiosk agent's ⋯ menu) shows it as a **Rule**, **Ask first** or **Fact**, with your
  words it came from: **Approve**, **Edit** or **Dismiss**. Nothing is taught without your yes, and **Just
  learned** keeps an **Undo**. Only your own turns are read, and only messages that read like a correction
  cost a model call.
- **External memory, one per person.** **Connections → External memory** links **Membase** or
  **Hindsight** (MIT, self-hosted with one Docker command, or Hindsight Cloud) to all of a person's agents,
  through the new `external_memory` tool, with the provider's own actions and rules. **Copied in on its
  own** mirrors new insights, intentions and lessons (reflections and intuitions if you tick them) through
  an outbox that waits out a refused key or a full quota. An admin can connect one Hindsight server for
  the whole deck, with a bank per person. Agents use it through the tool; it isn't recalled into the
  context by itself yet.
- **Course: the goals your agents' work is for.** **Workspace → Course** in the full layout, **Settings →
  You → Course** in the kiosk. Goals for **The deck** (set by an admin), **You** and **Your agents**, each
  **Strategic**, **Long-term**, **Medium-term** or **Short-term**, in a tree. Every agent working for you
  gets the live goals in its prompt and uses them to decide what matters next; a goal is never permission
  to do anything. Agents (with the new `course` tool) and Flight Deck suggest goals, which wait on your Line
  for **Accept**, **Edit on Course** or **Decline**. Work can be marked **Serves a goal**, the Taskboard
  sorts by **Serves your course**, and goals nothing has moved show as **Off course**.
- **`image_manipulation`: pictures changed without scripts.** Resize, crop, rotate, combine, overlay,
  watermark, annotate, animate, split and convert (png, jpg, webp, gif, bmp, tiff, ico, avif, pdf), about
  45 operations, in-process with no model or network. Inputs are never changed, phone photos open upright,
  and text has č, ć, đ, š and ž. Always on.
- **Admin console → Activity.** Who is using the deck, who has gone quiet, who never started, and whether
  you've written to them: a bar per day for what each person typed on any channel (their agents' own work
  never counts), **Write** from your own Gmail or Outlook, and **Mark as written** for a call or a chat in
  person. With Gmail read access it shows your mail with each person too, as dates and subjects kept 10
  minutes in memory.
- That makes **79 built-in tools** per agent (78 in the desktop app): `custom_tools`, `ask_user`,
  `colleagues`, `imap_mail`, `external_memory`, `course` and `image_manipulation` are new.

## Better

- **Learning that doesn't harden failures.** Insights, reflections and playbooks no longer keep "tool X
  doesn't work" (they keep the fix instead), failures caused by setup or cured by a retry, guesses about
  your traits, or a one-off request as a standing preference.
- **Scheduled tasks on an agent that's off.** A task starts an agent its owner switched off about five
  minutes before the run, then switches it off again when nothing else needs it. It used to fail with
  "agent not running".
- **The kiosk's agent menu** is a wide card in four labelled rows: **At work**, **What it knows**, **Who it
  is** and **Its place**.
- **Admin fixes found while planning the admin tutorial**: a deck without sign-in (the desktop app's)
  applies its console Settings; **Make it a company archetype**; Plan limits shows only the limits the
  deck enforces, and limit messages say to ask the admin; an empty deck asks for **Create the admin
  account**; waitlist approvals can go into kiosk mode; **Recent activity** reads as sentences with
  people's names, and **Reset today** is recorded; **Ask for feedback** greets people by first name.
- **Generate a group** streams its model calls, so writing archetype instructions no longer dies after a
  minute on some networks.
- **The kiosk tutorial matches the kiosk** (**New agent**, **Create**, retaken screenshots), message
  actions show on touch screens, files in your own message show as chips, and Escape closes the Line, the
  bell, Custom tools and the Share dialog.

## Fixed

- **A chat message whose connection dies before the agent takes it** is held and sent again when the chat
  reconnects, never twice.
- **An uploaded document is read under its own name**, not a `tmp….docx` one: way-of-working uploads,
  knowledge, profile documents and research, the queue planner, Drive and OneDrive folders, and
  `google_drive`.
- **Auto-save follows only this message's own words** and never overwrites an existing file.
- **A Gmail draft in Trash** no longer blocks a new reply. **Connections → Google → Granted access** names
  **Email address** and **Basic profile** instead of raw scopes.
- **The onboarding exam**: skill cards and exam writing may use 10k output tokens (reasoning models ran
  out at 4k); `1,850` and `1850` are the same figure; the grader's *why* stays in the question's language;
  figures the question itself gives, and a deadline counted from today, are no longer flagged as made up;
  exam situations keep the documents' own rates and limits.
- The desktop app's **About** shows the build year and the project's addresses (new installers only).

## Upgrading

1. **The desktop app**: **Check for Updates…** (it also offers 0.9.0 within a minute of starting). New
   installers are on this release, needed only for a fresh install or the new About panel:
   `Captain-0.9.0-arm64.dmg` (Apple silicon), `Captain-0.9.0-x64.dmg` (Intel) and
   `Captain-Setup-0.9.0-x64.exe` (Windows). They aren't signed, so macOS asks first (System Settings →
   Privacy & Security → **Open Anyway**), and so does Windows (**More info → Run anyway**).
2. **Each Flight Deck**: **Console → Updates**, or pull and reinstall by hand. `jsonschema` is now a
   declared dependency (it used to come only through LiteLLM), so `pip install -e .` (or `uv sync`) in the
   deck's environment. Then **restart Flight Deck, then every agent**: agents need the restart for the
   seven new tools, the Course prompt and the lesson proposals. Hard-refresh every browser.
3. **Look at what's on by default**:
   - **Lesson proposals**: a small model call on turns that read like a correction, 12 a day at most.
     `lesson_proposals.enabled: false` in an agent's `config.yaml` turns them off.
   - **Agents pick up email that names them**, from the first new mail after the upgrade (older mail is
     only a baseline). Switch **Pick up email meant for ‹agent›** off per agent, or set
     `FD_AGENT_MAIL=off` for the deck. Give agents with common-word names a specific name to answer to.
   - **A deck without sign-in now applies its console Settings** (branding, deck name, Docker) that it
     used to ignore, and **deleting a user now stops and removes their agents**, folders and chats.
4. **Custom tools** stay off until an admin turns them on. They need `git` on the host (`uv` is optional).
   On a Linux host, `python3 tests/test_flight_deck/test_xtool_sandbox.py` says whether the kernel has
   Landlock.
5. **Back up the new data**: `external_tools/`, `agent-bundles/` and `course.db` in the deck's data
   folder, and `bundle_signing_key` in its home (losing it gives the deck a new fingerprint).

New environment variables: `FD_IMAP_SEND=off` (no IMAP sending on this deck),
`FD_IMAP_REPEAT_CHECK_DAYS` (14), `FD_AGENT_MAIL=off`, `FD_MEMBASE_BASE_URL`, `FD_HINDSIGHT_BASE_URL`,
`FD_AGENT_BUNDLE_MAX_MB` (2048) and `FD_COURSE_REVIEW_TICK_S` (1800). New agent config sections:
`lesson_proposals` and `tool_proposals`.

Devices need nothing: firmware stays **0.20.4** / PaperColor B **1.0.31**. BotPort hubs need nothing:
0.9.0 keeps protocol 2. Captain for Chrome and Captain for Mac need nothing from this release.

## Known limits

- Custom tools, `ask_user`, lesson proposals, tool proposals and agent email were tested with stand-in
  models and stubbed mailboxes; IMAP was tested live against GreenMail and Dovecot, Hindsight against a
  self-hosted 1.2.0, Membase and Hindsight Cloud not at all.
- Export and import are in the kiosk only; Docker agents can't be exported yet.
- Copies of scheduled backups in your files aren't pruned.

## Pull requests

#436 Desktop About: build year and addresses · #437 Admin console: Activity · #438 Email (IMAP): several
inboxes per user · #439 IMAP: fast test, 587 fallback · #440 Learning prompts keep failures and guesses out
· #441 Agents propose lessons from your corrections · #442 Kiosk: Lessons to approve · #443 Agents as
colleagues: hand-overs, identity, email by name · #444 Send the Gmail reply · #445 External memory
(Membase) · #446 External memory: Hindsight, one server for the deck · #447 A chat message isn't lost when
its socket dies · #448 A task starts its agent when it's off · #449 `image_manipulation` · #450–#456 Custom
tools, phases 0–6 · #457 New tool with a spec check · #458–#461 Custom tools 7a–7d: uses, nested tools, Run
as, questions · #462 `ask_user` · #463 Recorded answers checked like real ones · #464 Questions at chat size
· #465 Questions name the agent; no Later · #466–#470 Agent export 1/5–5/5 · #471 Skill cards and exams may
write 10k tokens · #472 Agent menu in rows · #473 Kiosk tutorial matches the kiosk · #474 Auto-save follows
this message only · #475 Attachments as chips, Escape closes · #476 #477 Documents read under their own name
· #478 #479 Exam figures and language · #480 Small UI fixes from filming · #481 Builder told the publish
limits · #482 Google scope labels, Trash drafts · #483 Course · #484 Eleven admin fixes · #485 Generate a
group streams · #486 Recent activity in plain words · #487 Feedback greets by first name
