# Keep on course: a task anchor for every agent, and a Jev course check on top

**Status:** being built, one PR per phase. Written 2026-11-27.

## Why

Weak models lose the person's request in the middle of the work. It happens at three moments:

1. **After a fold or compaction.** The opening message has been summarised or now sits far back
   (`_maybe_fold_turn_context`, `agent_context_governor_mixin.py:608`;
   `_turn_start_after_compaction`, `agent_session_mixin.py:1213`).
2. **On "continue" turns.** When the step budget runs out, the reply asks the person to say "continue",
   so the next turn's request is literally that one word.
3. **In long tool chains.** Twenty tool results bury the one line that said what the job was.

Two reminders exist today, and both have gaps:

- **The plan note** (`planning_context`) only appears when a task contract was made.
- **Jev's progress check** (`jev_progress.py`) only runs once the step budget is used up.

Nothing reminds the model what the job is in an ordinary long turn, and nothing notices drift
while it is happening.

## Decisions (2026-11-27)

1. **The anchor reminder always runs, with or without Jev.** It has no UI switch. A config key,
   `context.task_anchor` (default `true`), exists only so the benchmark harness can turn it off
   for a comparison arm.
2. **The Jev layer sits behind a per-agent switch in the UI.** The switch is on by default when
   the owner has Jev connected. Without Jev, the switch is disabled with "Needs a Jev key" and the
   anchor still runs.
3. **This is not a fourth judge.** The course check becomes shared state for the whole turn: the
   progress check and block review read it instead of asking Jev again.
4. **Notes are never saved into the session.** They go into the live block, which is rebuilt on
   every model call. A wrong nudge can't stay in the history and repeat on every later turn, the
   way the false-action-claim accusation did.
5. **No extra model calls.** The anchor is cut from the person's own words. Jev classifies; it
   never writes text, and every line the model sees comes from a fixed template.

## What this builds on

| Piece | Where | Reused for |
|---|---|---|
| Live block, rebuilt each call and appended to the last message (keeps the prompt cache) | `_LIVE_NOTE_ORDER`, `agent_context_mixin.py:71`; suffix at ~5912 | the `task_course` note |
| Jev state builder and readiness check | `jev_progress._step_lines`, `ready()` | the course check's state, owner-only gating |
| Jev at the step budget | `agent_orchestration_mixin.py:~2575` | reads the latest course check first |
| Jev after a refused call | `_jev_review_refusal`, `agent_tool_loop_mixin.py:1081` | gets the anchor and stage in its state |
| Topic-switch heuristic | `_looks_like_topic_switch`, `agent_reasoning_mixin.py:152` | anchor relation without Jev |
| Session metadata (persisted) | `Session.metadata`, `session/__init__.py:48` | where the chat anchor lives |
| Per-agent switch file + FD route + Settings switch | task rephrase: `server.py:5905`, `KioskAgentDialogs.tsx:811` | the UI toggle |
| Worker dispatch with live events | Basna `_dispatch_one` (`basna_routes.py:3001`), Vatra dispatch | carries the anchor to workers, brings checks back |
| Vatra ledger | `vatra_ledger.py` (`focus` per owner) | new `course` kind per subtask |

## 1. The task anchor (always on)

### What it holds

The anchor lives in `session.metadata["task_course"]`, so it outlasts a turn. When several
surfaces share a session, it follows the person from device to device; sessions that are kept
apart each have their own.

```json
{
  "id": "tc_…",
  "anchor": "the person's request, their words, ≤ 600 chars",
  "amendments": ["≤ 3 newest, ≤ 200 chars each"],
  "latest": "the latest message when it differs from the anchor, ≤ 200 chars",
  "source": {"session": "…", "turn_msg": 123},
  "requirements": ["from the task contract when one exists, ≤ 5"],
  "opened_at": "…", "updated_at": "…",
  "status": "open | done | dropped",
  "checks": []
}
```

- **The words:** the person's message with the bridge's surface block stripped
  (`msg_origin.split_surface_block`), whitespace collapsed and cut to 600 chars. A longer request
  ends with "(the full request is earlier in the conversation)".
- **Requirements:** copied from the task contract when the turn made one.

### Keep it or start a new one, without Jev

The rules work in any language, because most of the signal is length and timing:

- **No open anchor:** the turn's message becomes the anchor.
- **"Continue" words:** continue / go on / nastavi / dalje / weiter / sigue and the like keep the
  anchor unchanged.
- **Short message (≤ 12 words), open anchor, previous turn within 2 hours:** keep the anchor; the
  message goes into `latest`, and into `amendments` when it adds to the request rather than
  answering a question.
- **Anything else:** if `_looks_like_topic_switch(message, anchor)` says so, or the message is
  longer than 12 words and shares almost no content words with the anchor, the old anchor is
  marked `dropped` and a new one starts.
- **Gap of more than 2 hours after a turn that ended with a final answer:** a new anchor.

A wrong guess costs little, because the note always shows the latest message beside the anchor
and the model can see which one it is answering. With Jev on, the relation is decided by Jev
instead (section 2).

### When the model sees it

A `task_course` note goes **first** in `_LIVE_NOTE_ORDER`, but only from the turn's second model
call onwards, after the first tool call. A plain chat reply never carries it and costs nothing.
After a fold or compaction it is still there, because the live block is rebuilt on every call.

Without Jev:

```
Task (the person's request): <anchor>
Added since: <amendment>; <amendment>
Their latest message: <latest>
```

The live block's existing wrapper already marks it as reference only, not to be repeated or
quoted in the reply.

### Closing

An anchor is marked `done` when a turn ends with a final answer and the next human message starts
a new anchor; it is marked `dropped` when it is replaced mid-way. Up to 10 closed anchors are kept
in metadata as a short summary: the anchor, final stage, number of checks, and corrections shown
and followed.

## 2. The Jev course check (switch on and Jev usable)

A new module, `captain_claw/jev_course.py`, is modelled on `jev_progress.py` and reuses its
`_step_lines` and `ready()`. Readiness means the switch is on, it is not a shared-agent member's
turn, and the owner has Jev. Any timeout or error means no check this time; the anchor note is
unaffected.

### Questions

**At turn start** (only while an anchor is open and the message isn't a "continue" word), one
`choice` question, `task_relation`:

- `new_task`: a different job. The old anchor is dropped and a new one starts.
- `continues`: the same job, nothing added.
- `amends`: the same job with a change or addition, which goes into `amendments`.
- `not_a_task`: thanks, ok, chit-chat. The anchor stays as it is.

The heuristics above apply only when confidence is below 0.6. This replaces the keyword list,
which only knows English.

**During the turn**, one call with three questions:

| Name | Type | Meaning |
|---|---|---|
| `stage` | choice, 5 options | orienting · gathering · producing · checking · delivered — a choice rather than a score, so the answer reads without guessing the score's level numbering; the stage is the probability-weighted 1–5 |
| `on_track` | noul | Are the recent steps serving the anchor (not a side quest, not a different question, not redoing finished work)? |
| `next` | choice | `continue` · `refocus` · `finish` · `ask_person` |

The state Jev reads: the anchor, the amendments, the latest message, the requirements, "N steps
used" and the recent steps (`_step_lines`, ≤ 9k chars, newest kept).

### When it runs

- Every 6 tool steps once a turn passes 6, the same window as the stagnation detector.
- Right after a fold or compaction.
- After 3 failed tool results in a row.
- At most 8 checks per turn, and never two at once.
- **In the background:** the call starts after a step's tool results come in, and runs while the
  next model call is generating. The answer applies from the following call, so it adds almost no
  waiting. A result older than 6 steps when it arrives is thrown away.

### What the model sees

The anchor note gains a stage line:

```
Stage: producing (3 of 5), checked 2 steps ago
```

At most one corrective line is added, and only when Jev is confident:

| When | Line |
|---|---|
| `on_track` p ≤ 0.25 and `next` = refocus with ≥ 0.6 | Your recent steps don't look like they serve this task. Re-read the task above and take the step it needs next. |
| `stage` ≥ 4.5 and `next` = finish with ≥ 0.7 | The task looks done. If the result is ready, give the person your answer now. |
| `next` = ask_person with ≥ 0.7 | If you need something only the person can give (a login, a choice, missing information), ask them now. |

- At most 2 corrective lines per turn.
- The same line never shows twice in a row.
- A line is shown for one model call and then drops back to the stage line.

### Shared with the other Jev checks

- **The progress check at the step budget:** if the latest course check is at most 3 steps old,
  it decides instead of a new call. `on_track` ≥ 0.6 with `next` = continue extends the budget;
  anything else stops as today. Older than 3 steps: `jev_progress` asks as it does now.
- **Block review:** its state also carries the anchor and the latest stage, so it can pick better.
- **The stagnation detector** is unchanged; a "yes" from Jev can't keep a looping turn alive.

### What the person sees

The stage goes into the existing thinking/status line ("Stage 3 of 5 · producing") through
`_emit_thinking`, on web and on the bridges that already show live narration. It is never a new
message and never reaches a phone on its own.

### Logging

Each check writes a monitor row, `completion_gate step=jev_course`, with `stage`, `p_on_track`,
`next`, its confidence, which line (if any) was shown, and the latency. It is also appended to
`task_course.checks`, which keeps the newest 20.

## 3. Basna and Vatra

Workers run the same agent loop, so the check runs inside each worker. Flight Deck supplies the
anchor and collects the results; it does not run a second implementation.

- **Anchor:** the dispatch frame gets a `course` field,
  `{"goal": <root ask>, "part": <subtask title + brief, ≤ 600>, "requirements": […]}`. The worker
  sets its anchor from it instead of guessing. The note then reads "Run goal: … / Your part: …".
  Basna workers get the theme as the goal and their angle as the part.
- **Switch:** Flight Deck writes `tools.jev.course_check` into each worker's config from the
  launching agent's switch. A run started from the Vatra or Basna page uses the default (on when
  the owner has Jev). The value is stored with the run's other settings, so continuation rounds
  inherit it.
- **Collecting results:** the worker's `jev_course` monitor events already reach Flight Deck
  through `on_action`. They become a ledger row of kind `course`, keyed by subtask, holding
  `stage`, `p_on_track`, `next` and the time.
  - `vatra(action="map")` shows each owner's stage.
  - The run page's Live agents tab shows a stage pill per owner.
  - An owner off track in 2 checks in a row is flagged on the run page.
- **Cost guard:** at most 8 checks per worker turn (same as chat), and at most 150 per run
  (`FD_COURSE_MAX_CHECKS_PER_RUN`). After that, workers keep the anchor note only.
- **Verify first:** ephemeral workers must resolve to the run's owner in `/fd/jev/agent/*`
  (`_agent_caller`, `jev_routes.py`). If they don't, Jev is off for workers and only the anchor
  runs. That still helps, but it should be fixed.
- **Not in v1:** the coordinator acting on a flagged owner (re-brief, re-dispatch).

## 4. The UI switch

**Where:** the agent's Settings dialog in the kiosk (`KioskAgentDialogs.tsx`), as a new section
next to "Rewrite requests" (~line 811), using the same `SwitchTrack` button.

- **Section title:** Keep on course.
- **Switch label:** Check progress with Jev.
- **Help text:** "Every few steps Jev checks whether the work still serves your request and how
  far along it is, and reminds the agent when it drifts. The agent always gets a reminder of your
  request either way. Applies from its next message."
- **Without Jev:** the switch shows off and disabled, with "Needs a Jev key — Connections → Jev".

**Storage and route:** mirror task rephrase.

- A switch file, `course_check.txt` (`on` / `off`), in the agent's home dirs. A missing file
  means on.
- `tools.jev.course_check` in every config copy, so the setting holds after a restart and travels
  with the agent's config and `.captain` export.
- `GET /fd/agent-course-check/{kind}/{id}` returns `{"enabled": bool, "jev_connected": bool}`.
  `PUT` writes the file and the config copies. No restart.
- `services/docker.ts` gets the two calls beside the rephrase ones.

**Agent side:**

- `JevToolConfig.course_check: bool = True` (`config.py:448`).
- `jev_course.enabled()` reads the switch file every turn, falling back to config.
- **Effective state = enabled AND Jev ready.** That is "on by default if Jev exists".

**Frontend:** write dark-first zinc classes, which the light theme remaps; then rebuild and commit
the bundle.

## 5. Tracking whether it helps

**Recorded per check:** stage, p_on_track, next, the line shown, latency.

**Recorded per turn**, in the existing usage record and `process.log`:

- **Outcome:** finished, stuck, budget landing, cancelled.
- **Steps and tokens.**
- **Anchor changes.**
- **Follow rate for each corrective line:** do the 3 steps after it use different tool targets
  (paths, URLs, queries) from the 3 before it? This is computed deterministically from the turn's
  messages.

**Questions those records answer:**

- Do turns with corrections finish more often than turns without?
- Do weak models follow refocus lines?
- How often is "finish" right? A sample of turns gets a manual look.

**Proof:** an A/B on the WildClawBench/Harbor runner with the local Qwen. Three arms on the same
tasks, three seeds each:

| Arm | `context.task_anchor` | `tools.jev.course_check` |
|---|---|---|
| Off | false | false |
| Anchor | true | false |
| Anchor + Jev | true | true |

Measure score, steps, tokens and the stuck rate. This runs only when the owner asks for it.

## 6. Privacy and cost

- **Data:** Jev sees what the existing Jev checks see: the request and the turn's steps, cut to a
  few thousand characters. It is owner-only and never runs on a shared-agent member's turn. There
  is no new kind of data, but there is more of it: up to 8 calls per long turn, times the workers
  in a run.
- **Cost:**
  - Short turns pay nothing.
  - A long chat turn pays a few Jev calls.
  - A Vatra run is capped at 150.

## Phases

| Phase | What | Deploy |
|---|---|---|
| **P1: Anchor** | `task_course` record + relation heuristics + live note (from the 2nd call) + closing; `context.task_anchor` | agent restarts |
| **P2: Jev in chat + switch** | `jev_course.py` (relation + in-turn check, background, caps, lines), monitor rows, reuse by progress check and block review, status line; config key, switch file, FD route, Settings switch, bundle | FD restart + agent restarts + hard refresh |
| **P3: Basna/Vatra** | `course` in dispatch frames, worker config switch, ledger kind + map, run page stage pill and off-track flag, per-run cap; verify worker → owner in `/fd/jev/agent/*` | FD restart (workers spawn fresh) |
| **P4: Measure** | per-turn records + follow rate; A/B when asked | none |

## Tests

The suite hangs when run as a whole; run it per file.

- **Anchor:**
  - relation rules: continue words in several languages, short follow-up, long new request,
    thanks, the 2-hour gap
  - the 600/200-char caps and surface-block stripping
  - no note on the first call
  - the note is never in `session.messages`
  - it survives a fold
- **`jev_course`:**
  - state building
  - thresholds for each line, at most 2 lines per turn, no repeats
  - the 8-check cap, no overlapping checks, stale results dropped
  - not ready (no key, member turn, switch off) means no call
  - the progress check reuses a check ≤ 3 steps old
- **Switch:** missing file → on; route writes the file and the config copies; `jev_connected` in
  the GET.
- **Basna/Vatra:** the `course` frame sets the anchor; the ledger `course` row comes from a monitor
  event; the per-run cap.
