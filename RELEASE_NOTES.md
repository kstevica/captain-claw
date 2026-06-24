# Captain Claw v0.6.1 Release Notes

**Release title:** The closed loop & the topic memory — Captain Claw acts, remembers, and learns
**Release date:** 2026-06-24

A big behavioural release. **Autonomous Work** finally closes its loop end-to-end — the assistant notices events in the user's world, decides what to do, takes safe action, judges the outcome, and learns from it. A new **conversation topic memory** auto-clusters everything that flows over comms channels into durable, recallable topics — with a full Flight Deck panel for browsing, searching, refining, and chatting inside any topic. The catalog of autonomous actions and the set of event sources are now **user-extensible** from the UI — every connected agent tool can be promoted into a "hand" or a "sense". The agent itself learned a few honesty rules along the way (grounding negative claims, routing self-corrections to the real memory channel). Additive and backward compatible with 0.6.0.

---

## What's new

### Autonomous Work — the closed loop

The four topics first sketched in 0.5.x are all wired and live, plus five "Jarvis" gaps closed on top of them. Everything is per-user, shipped under a `propose` ceiling, and configurable from a dedicated **Autonomous Work** page.

- **Arbiter** decides ONE next action per heartbeat — `nudge`, `run_prompt`, `basna`, `materialize_schedule`, `stop_run`, `tool_action`, or `track` — using the user's reflections, intentions, and current events as candidates.
- **Efferent dispatch** runs the decision via the user's strongest agent (or, for `tool_action`, the deterministic agent-WS rail).
- **Judge & learn** — every outcome (auto or human) updates a Bayesian reliability weight. The arbiter suppresses kinds that keep failing; **trust ladder** auto-promotes a specific reversible tool action to auto-fire after it earns its keep (≥0.85 weight over ≥3 runs).
- **Reflections → intentions** — the agent's own self-reflections become candidate work.
- **Action catalog** (#1) — 4 reversible auto-eligible (note.write, calendar.hold, mail.draft, reminder.schedule) + 4 human-only (mail.send, calendar.invite/delete, message.send, drive.delete). Shell/browser/social/payments are HARD-EXCLUDED.
- **Event spine** (#2) — a dedicated `events.db` plus FD-side **Google Calendar + Gmail pollers**. New emails / upcoming events surface as arbiter candidates; an automated-sender filter drops no-reply/notification noise before it ever reaches the loop.
- **Plans** (#4) — give it a goal; it decomposes into catalog-validated steps. Manual or auto-advance; a failed step now **re-decomposes the remainder** rather than failing the whole plan.
- **Grounded verification** (#5) — after a catalog action runs, the side effect is read back (note → `read(path)`, calendar → `get_event(id)`). Absent → fail; couldn't read → kept but flagged `[unverified]`. Trust never builds on phantom success.

### Soft reminders, follow-ups, and escalation

Beyond *act-now* and *drop*, the arbiter has a third outcome: **`track`** — a soft reminder or "waiting on you" item that goes into a per-user follow-up list with a due date. When due, it's re-fed as a candidate; if the resulting nudge fires, the follow-up **re-arms sooner each time** (3d → 2d → 1d) and retires to `stale` after enough nudges, so it never nags forever.

### Tools & Sources — open the platform

Action catalog and event sources are no longer hardcoded; the user's connected agent tools become a menu they can promote.

- **Promote a tool → action** (a "hand"): set risk, reversibility, optional reverse tool, grant. Defaults to *propose-only*; can only auto-fire after the trust ladder earns it. Shell/browser/social/payment tools are hard-excluded and cannot be added.
- **Promote a read tool → source** (a "sense"): a generic tool-poller calls a list/search tool on a cadence, parses rows (auto or via `items_path`), dedups by your chosen id field, and stamps a **fetch contract** (`fetch_tool` + `id_field`) onto every event so the agent can later open the right item by id.
- **Table-driven grounding.** When the arbiter dispatches an event-derived action, the agent is told: *"this is a REAL item, open it BY ID with `<fetch_tool>`, never search, never claim it doesn't exist."* Built-ins (gmail thread, calendar event) and any custom source share the same grounding line.
- **Tools & Sources tab** in Flight Deck — "How it works" walk-throughs, a library of one-click templates (Granola meetings, Gmail search, Drive recent files, Web watch, save to memory / datastore, label Gmail thread, fetch a web page, summarise files, etc.), and clickable agent-tool chips for one-tap promotion. Hard-excluded tools are dimmed and disabled.
- New routes under `/fd/autonomy/*` (catalog, agent-tools, follow-ups, plans, run-action). Per-user data persists through the existing config save.

### WhatsApp delivery for nudges

Proactive nudges fan out to the user's WhatsApp (configured `WHATSAPP_ALLOWED_WAIDS`) on top of the web chat. Off by default for new users; controlled by `nudge_to_whatsapp` on the Autonomous Work page. A dedup guard at the outbound chokepoint prevents the same reply from being sent twice when both the agent's natural reply channel and the explicit push fire.

### Conversation topic memory

A new memory layer on top of comms traffic. The agent has long had session and semantic memory; **topics** is the layer that turns "the Munich trip" or "the Vesna VC deal" into a single durable object the agent can recall later — without scrolling history.

- **Auto-classifier.** Every ~15 comms messages a background pass (mirroring the dreaming and insight-extraction passes) takes the new user + assistant turns and the turn's narration, shows the LLM the existing topics, and assigns each message to an existing topic or creates a new one. Persistent and cross-session.
- **Always-on `topics` tool** — `list`, `search`, `get`. The agent can pull a thread's full context at once.
- **Flight Deck Topics modal** on every agent card — fullscreen toggle, resizable splitter, markdown rendering (tables included), and **.md export** of any topic.
- **Operations on topics:** **Refresh** (re-pull full text from the live session by msg id), **Combine** (merge selected topics, dedup by msg id, merge keywords), **Reset** (wipe with optional checkbox-preserved subset), **Reclassify** a single topic (free its messages so the next pass redistributes them).
- **Search & navigation:** LIKE substring search across labels/summaries/keywords (instant), sort by **Recent** or **A–Z**, **starred-on-top** with a star toggle per row, and **drag a message between topics** for surgical fixes.
- **Groups** — user-defined groups (Work, Private, anything custom) with full many-to-many: a topic can belong to several groups; group filter chips + per-topic assignment chips. **Combined filters** (text + group + tags) all AND together server-side; clicking a tag pins it as a removable chip.
- **Per-topic chat** — at the bottom of the detail pane, a live chat scoped to the topic. Streaming reply, live narration, tool-usage rows, paste + file attachments, busy-gated send (Stop cancels). Renders only the turns you initiate here (ignores the agent's session replay), and persists each completed turn to the selected topic by its real session msg id.
- **Generate / Reset behind a gear icon** in the modal header to keep the top bar calm; status surfaces next to the title while a backfill runs.

New SQLite store `conversation_topics.db` (topics + topic_messages + FTS + groups + many-to-many table + backfill-progress markers). New endpoints under `/api/topics/*` (list, search, refresh, combine, reset, unclassify, append, groups, star, message-move, backfill) with matching `/fd/agent-topic*` proxies.

### Agent honesty — ground negative claims, real self-correction

Three new rules in the system prompts (full, micro, nano) after live cases where the agent confidently said "I didn't search" (it had) and later promised to "fix my internal logic" (it can't):

- Never assert "I did NOT do X" from memory. Scan the conversation's tool calls first; if you can't see the result, say "let me check" and verify rather than denying. Trust the logs over memory.
- You cannot rewrite your own base prompt or code. But you CAN make a behavioural correction durable: save it via `insights` (action='add') or your `personality` instructions (action='update'), **once**, this turn, then move on. Don't keep restating the resolution at the user.

Companion grounding for surfaced events: arbiter actions about a real event carry the event's id; dispatch tells the agent how to open it by id and instructs it to never claim it doesn't exist.

### Basna polish

- `/basna <task>` slash command on **web, WhatsApp, Telegram** (with help text on no-arg). Croatian verb-stem matching so `pokrenuti basnu` (not just `pokreni`) also triggers the deterministic relay.
- **Recursion forbidden** — a Basna worker can't start another Basna (tool stripped from the spawn + env-marker guard).
- **Stop / hard-stop primitive** for Basna runs and Council deliberations (UI buttons + arbiter `stop_run` proposal). Per-owner run-rate breaker (6/300s) prevents runaway spawns.
- **Deepen** — a follow-up run that resolves a finished run's blind spots. Lineage links (deepened from / into), an in-flight "Investigate blind spots" button, large prior synthesis routes to a workspace file (16k char cap inline; bigger → file approach), live per-agent panels for server-side runs.

---

## Fixes & hardening

Eventful release. The notable ones:

- **Scale-coverage gate.** When the scale_micro_loop completed N items, glasses-mode agents were dropping items to honour brevity. The finalize gate now requires the user-facing reply to fingerprint every processed item; if any is missing it re-iterates with a "list ALL items" hint. Glasses brevity was also softened: "brevity is a TARGET, not a ceiling — a complete short list beats an incomplete shorter one."
- **Topic-message duplication.** Two converging bugs were doubling every chat-driven turn on agent restart. The periodic classifier now dedupes against `classified_msg_ids | seen_msg_ids` (so an in-memory `last_idx=0` after restart can't re-classify the whole session), and chat appends now use the agent's real session msg ids (so the classifier's dedup later catches them).
- **Reasoning-model starvation.** DeepSeek V4 Flash was burning the whole output budget on its `<reasoning>` and emitting zero JSON. Raised classifier `max_tokens` to 8000, dropped batch size to 15, and reinforced "spend output on the JSON, not reasoning" in the prompt.
- **Backfill batching.** Topic backfill used to do many LLM calls inside one HTTP request and trip the 15s proxy timeout. Now one batch per call with `remaining`; the UI loops until drained, with a live "N done · M topic updates · K left" counter and a clean error path.
- **Duplicate WhatsApp nudges.** Both the channel-bus pump and the explicit push were delivering the same reply. Added a 30s same-text dedup guard at the single outbound chokepoint.
- **Replay-capture bug** in `_send_chat_and_collect`: autonomous nudges through a reused agent were grabbing the agent's *previous* session reply as "this turn's reply" because the pre-existing replay wasn't drained before sending. Fixed to drain `replay_done` before the task goes out.
- **Poll-cadence persistence.** `last_poll_at` was using `time.monotonic()`, which resets to ~0 on machine restart while the stored value was minutes-old — polling silently stalled. Switched to `time.time()`.
- **Arbiter follow-up loops.** Soft reminders ("kindly check the offer") used to be silently ignored (scored below `arbiter_min_score`). Now they're tracked as durable follow-ups; the loop only nudges when due, escalates with age, and retires to `stale` after enough nudges.
- **Event reconsideration.** Events were marked `surfaced` immediately at intake, so a single whiffed arbiter pass dropped them forever and a manual "Run arbiter now" couldn't reconsider. Now the event stays `new` until a pass produces an action; bounded retry budget prevents churn.
- A handful of smaller things: arbiter no longer re-proposes reworded variants of completed work; nudges count as success on delivery (don't LLM-judge them); arbiter prompt now produces parseable JSON reliably across model tiers; ranker rates agents by Library tier rather than model string.

---

## Migration & compatibility

Additive — **backward compatible with 0.6.0**.

- New SQLite stores auto-create on first run under `FD_DATA_DIR` (or `~/.captain-claw/`): `conversation_topics.db`, `events.db`, `autonomy.db`. Existing autonomy/events DBs migrate in place (new columns added with `ALTER TABLE`).
- New endpoints live under `/api/topics/*`, `/api/autonomy/*` (action catalog, follow-ups, plans, run-action, agent-tools), and the matching `/fd/agent-topic*` / `/fd/autonomy/*` Flight Deck proxies. No old endpoints removed.
- The Autonomous Work loop **ships off**. Opt in per user from the Autonomous Work page. Shipped ceiling is `propose` (every action waits for human approval) until you grant specific actions or let the trust ladder earn auto-fire on its own.
- Conversation topic memory ships **on** with sensible defaults (15-message classify interval, 120s cooldown, 300 max topics). Turn it off via `conversation_topics.enabled` in config.
- A `topics` tool is added to the agent's always-enabled set. Glasses-mode brevity rules were softened for multi-item replies; if you've customised them, re-check.
- Theme A (custom actions / sources) defaults to safe: a promoted action is `human_only` until you widen it; a promoted source defaults `enabled=false`. Hard-excludes (shell, browser, social, payments) are enforced centrally — a misconfigured action with one of those tools is dropped.

See [release-notes/RELEASE_NOTES_0.6.0.md](release-notes/RELEASE_NOTES_0.6.0.md) for the previous release, or the [release-notes/](release-notes/) folder for the full history.
