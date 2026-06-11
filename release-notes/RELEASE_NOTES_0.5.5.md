# Captain Claw v0.5.5 Release Notes

**Release title:** Self-aware in time — timing context, live token meters, and a calmer agent
**Release date:** 2026-06-11

A self-awareness + reliability release. The agent now reasons about **time**
(when you last spoke, when it last replied, when the next scheduled run fires),
Flight Deck shows **live token usage** and a new **Connections** health panel,
and a batch of guard fixes stops a weak model from talking itself into a corner —
no more spurious *"I got stuck"* on topic switches, no more derailing a finished
task into a generic greeting. Additive and backward compatible with 0.5.4.

---

## What's new

### Activity-timing context — the agent knows *when*
Every turn now carries a compact timing block in the system prompt, alongside the
current clock:

- **Last user message / last reply / last scheduled-or-cron run** — stamped into
  session metadata (persists across reconnects). Real user messages are tracked
  separately from cron/scheduler-driven runs, so automated turns don't masquerade
  as user activity.
- **Next scheduled run ETA** — "in 2h (daily 09:00)" for this session's soonest
  enabled cron job, so the agent can reason about its own upcoming wakes.
- **Part-of-day + weekday hint** and a **conversation cadence** count, for tone
  and recency. The "last user message" line doubles as an idle-gap signal.

Compact one-liner in eco/micro mode, omitted in nano.

### Live token meters in the activity panel
Each LLM call broadcasts running cumulative usage, shown right in the activity
panel header:

```
Activity · 16 tools · 17.0k↑ 340↓ · 5.1k cached
```

Input / output / cache-read tokens, **live** while the turn runs and **frozen**
onto each activity group when it ends — so scrolling back shows each turn's own
numbers, not the latest. Background maintenance (reflection/insight/dream) is
excluded.

### Connections panel (Director)
A new **Connections** tab in the Director panel polls external dependencies and
shows a traffic-light per connector:

- 🟢 connected **and** a read-only test call succeeds
- 🟡 connected but the read-only test fails
- 🔴 not connected / not configured · ⚪ disabled

Covers **Google** (a live read-only probe — refreshes the token and calls
userinfo, not just a token-presence check) and every enabled **MCP server**
(read-only `tools/list`). Polls every 10 minutes with a manual re-check.

### Truthful phase statuses
The status line now reflects the phase the agent is actually **in**, not the last
thing it finished: `Using web_fetch…` while a tool runs, then
`Calling LLM (model) · turn_3 · 17k ctx tokens (11%)…` during the call (the slow
part), with a ⚡ prefix once tokens start streaming.

---

## Reliability fixes

- **No more spurious "I got stuck" on topic switches.** A chatty reply no longer
  pins a stale "clarification" anchor that hijacked your *next* (unrelated)
  message and forced a heavyweight pipeline a one-line request could never
  satisfy. Topic switches are detected and the stale context is dropped; simple
  follow-ups skip the contract pipeline; and a conversational turn that can't show
  "progress" now returns its actual answer instead of an error.
- **A finished task no longer derails into a greeting.** A correct *"✅ Done —
  saved to <path>"* summary after a file write was being killed by the
  false-web-claim guard (it referenced research from an *earlier* turn) and the
  forced retry pushed a weak model into a generic "how can I help?" greeting. A
  turn that produced a real deliverable is now recognized as summarizing completed
  work — stall and false-claim retries are suppressed for it.
- **No blind file rewrites.** A second full `write` to the same file in one turn,
  with no read/shell/browser in between, is redirected to `edit` or finalize —
  killing the regenerate-from-scratch double-write loop.
- **Anthropic Fable models** — `temperature` is now omitted (Fable rejects it with
  a 400); other models are unaffected.
- **Calmer background work** — reflection, insight extraction, and dreaming run
  silently; the agent stays in the idle/waiting state instead of looking busy
  after it already replied. Auto-reflection now needs 20 messages (was 10).

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild the frontend only if you build locally:
npm --prefix flight-deck run build
```

- **Restart the agent** (`captain-claw-web`) for the timing context, token-usage
  emit, truthful statuses, and all the guard fixes.
- **Restart the Flight Deck server** for the Google connection probe behind the
  Connections tab.
- **Hard-reload Flight Deck** for the Connections tab, token meters, and status
  changes.
- **Desktop app:** `./build-desktop.sh` — everything above is in the rebuilt
  bundle.

Backward compatible with 0.5.4 — everything here is additive or a fix; no
configuration or API changes.
