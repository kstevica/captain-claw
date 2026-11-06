# Agent colleagues — plan

Two changes that make an owner's standing agents work as a team, not as one
person's separate assistants.

1. **Agents hand tasks to each other on the Taskboard.** An agent gives a piece
   of work to another of its owner's agents (a colleague). It shows on the
   owner's board as a task of its own, the colleague starts on it at once, and
   when it is done the agent that handed it over gets the result back and
   carries on.
2. **Each agent has an identity of its own**: a role, a line on what it does,
   the names it answers to, and its own email: its own mailbox (an IMAP inbox
   marked as the agent's) or an alias that lands in the owner's inbox
   (`you+researcher@gmail.com`). Email that reaches it, or that names it in the
   subject or addresses it in the body in the owner's Gmail, Outlook or IMAP
   inboxes, becomes a task assigned to that agent, which starts on it.

## User decisions (2026-11-06)

| # | Question | Answer |
|---|----------|--------|
| 1 | Agent-to-agent hand-over: at once or after the owner's Approve? | **At once.** Never more permission than the agent that handed it over. The owner sees the chain on the board and can stop it. The first agent gets the result back. |
| 2 | What may an agent do with an email that reaches it? | **Work at once. Replies from its OWN inbox go out within that inbox's send settings** (off by default, allowlist, daily limit). Replies in the owner's inbox stay drafts. |
| 3 | What is an agent's own inbox? | **A mailbox or an alias.** A connected IMAP inbox marked as the agent's (only it uses it), or addresses that land in the owner's inbox. |
| 4 | Which owner inboxes are watched for mail that names an agent? | **Gmail, IMAP and Outlook.** |

## Identity

Kept per agent by Flight Deck (`colleagues.db`, table `identities`, keyed by
owner + slug):

- `role` (≤ 60), `about` (≤ 300): what colleagues and the roster see.
- `names`: what it answers to in email (the agent's name by default).
- `addresses`: aliases that land in the owner's inbox (`you+researcher@…`).
- `inbox`: the id of an IMAP inbox that is its own. Setting it makes that
  inbox's agent allowlist just this agent.
- `pick_up_mail`: pick up the email meant for it — its own inbox, its
  aliases, and mail in the owner's inboxes that names it. **On for every
  agent unless switched off** (user decision 2026-11-06, after the first
  build had it off): an agent with no saved identity answers to its own name.
  A switch in each agent's settings (kiosk Agent settings → Email, the full
  layout's Agent options → General → Email) and in Identity & email. The
  watcher's first look at each source is a baseline, so a deploy never picks
  up older mail.
- `hands_to`: `all` (default) or a list of colleagues' slugs.

## Agent to agent

Tool `colleagues` (owner turns under Flight Deck, like `line`):
`list` (who is on the team, with role, what they do, email),
`hand_over` (to, instructions, title?), `handed` (what this agent handed out
and how far each one is).

Flight Deck (`colleagues.py`, `POST /fd/colleagues/agent/*`, the caller
resolved from its `X-Agent-Auth`, never its word):

- Every hand-over is a **new task** on the owner's board (kind `request`,
  direction `internal`, channel `colleague`), part of the task the first agent
  is on when it was on one (`flags.colleague.parent`), with the chain in
  `flags.colleague` (`from`, `to`, `via`).
- It goes through the Taskboard's line (`handover_line`): one at a time per
  agent, survives a deck restart the same way.
- **Authority never grows.** The tool sends the turn's own mail authority
  (`mail_authority.current()`, read by code, not by the model). A human turn's
  words become the colleague's `job_text` (it may email only if the person
  asked for one); an automated turn's authority passes as it is. An inbox's
  `send_from` never passes on.
- **Guards:** no handing to itself or to an agent already in the chain; a
  chain at most 4 agents long; at most 60 agent hand-overs per owner per day.
- **Coming back:** when the colleague finishes, its task closes with its
  answer (or comes to Needs me when it couldn't), the parent task gets a line,
  and the first agent gets the answer: in the task's own session when it was
  working on a board task, else as a notification in the chat it was asked in.

## Email to an agent

`agent_mail.py`, a watcher (60 s) for owners with at least one agent that has
an identity with mail:

- **Sources:** each IMAP inbox (UID cursor per inbox), Gmail (the owner's
  token, after-date + seen list), Outlook (Graph, receivedDateTime cursor).
  The first look is a baseline: only mail after it counts.
- **Gate (no model):** automated, bulk, list and no-reply mail is left out;
  so is mail an agent or the owner's own sends wrote.
- **Match** (`mail_match.py`, pure): an address (To/Cc/Delivered-To has the
  agent's own inbox address or an alias) wins; else the subject names the
  agent as a word; else the body addresses it ("Hi Researcher", "Researcher,
  please…", "@Researcher", "for Researcher", "this is for Researcher"). Mail
  in an agent's own inbox is that agent's.
- **Dispatch:** a task on the owner's board (the same thread keys the board
  uses, so a passive Gmail inbox task is the same task), assigned to the agent,
  and a turn in its line (automation kind `inbox`), the email fenced as data.
- **Authority:** `mail_write: draft` (drafts only), plus `send_from` = its own
  IMAP inbox when it came there: an `imap_mail` send from that inbox goes to
  Flight Deck, where that inbox's own gate decides.

## Sending replies from the owner's Gmail (2026-11-06, after staging)

Decision #2 kept replies in the owner's inbox as drafts. After the glasses
agent answered a real email with a Gmail draft, the user asked for a way to
send. Per agent, `gmail_replies`:

- `off` (default): a draft, as before.
- `me`: the reply is sent when the email came from the owner — the deck
  account's address, the Gmail mailbox's own addresses (its send-as aliases),
  or a connected IMAP inbox's address.
- `verified`: the reply is sent to whoever wrote.

Either way only when Gmail verified the sender (`gmail_inbound.sender_authenticated`,
so a forged From never counts), and only the one reply: the turn carries
`reply_id` (the Gmail message) and `reply_to` (the sender), and
`mail_authority.check_mail_write` lets a `google_mail` `send` through only with
that `reply_to_message_id`, `to` exactly that address and no cc / bcc
(`send_draft` stays refused). The send still goes through Flight Deck's Gmail
gate (Connections → Google → Email sending: on/off, allowlist, daily limit).
The reply never passes to a colleague; it does come back to the same agent
with a colleague's answer. UI: the "Replies in your Gmail" choice under the
Email switch (agent settings) and in Identity & email, shown when Gmail is
connected.

Agent-made hand-overs (colleague, answer back, email) carry the `handover`
mark, so the board's ledger reader no longer shows them as a proposal and an
approval.

## Agent-side changes

- `mail_authority`: kinds `colleague` and `inbox` (labels, prefixes), mode
  `draft`, field `send_from`.
- `turn_source`, `msg_origin`, `turn_register`, `chat_handler`: the two kinds
  run on the automation lane in a task session like `taskboard`, are not
  mirrored into the main chat.
- Tool `colleagues` (metadata `requires_line`).

## UI

- Agent "⋯" menu (kiosk and the simple layout) → **Identity & email**
  (`KioskAgentIdentityDialog.tsx`, `services/colleaguesApi.ts`): role, what
  it does, its own inbox (picking one makes that inbox only this agent's),
  aliases, the names it answers to, the pick-up switch, and the colleagues it
  may hand work to.
- Taskboard: channels **Colleagues** and **Outlook**; a task an agent handed
  over shows **Between your agents** (the chain, "Open the task it is part
  of"); an email routed to an agent shows **Email for an agent** (which agent,
  and how it knew). Agent-made lines never show on the user's batch banner
  (`ah_` batch ids).

## Not in this version

- Agents of other owners as colleagues (a shared agent, a team roster).
- An agent's own Gmail or Microsoft account (only IMAP mailboxes and aliases).
- Sending from the owner's Gmail or Outlook on an email's turn.
