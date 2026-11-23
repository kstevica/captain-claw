# Drive — how far the harness pushes a turn

**Status:** plan / nothing built · **Date:** 2026-11-23 · **Owner decisions locked** (see §9)

A per-chat setting, next to reasoning effort, that says how hard Captain's harness
works to finish what was asked: how many steps a turn may take, how many times a
weak answer is sent back, whether the result is checked before it is delivered, and
(at the top level only) whether the agent may do a little more than was asked.

The name is **Drive** (decided, §9).

---

## 1. Why

Three settings shape a turn today, and none of them covers this:

| Setting | What it changes | Where it acts |
|---|---|---|
| Reasoning effort | how much the model thinks per call | the model |
| Cognitive mode | how the agent approaches work | prompt text |
| **Drive (new)** | how far the harness pushes to finish | harness code |

Today the harness has one fixed level of persistence for everyone. A quick question
and a two-hour research job get the same step budget rules, the same validation
retries and the same stall handling. People who want a fast, cheap answer cannot
ask for less, and people who want the agent to keep going cannot ask for more
without reaching for Bat, which is a separate long-running mode.

It is also the right home for the one harness number that cognitive modes still
carry (`completion_strictness`). Modes should be text only; Drive should be the only
setting that touches gates.

## 2. The levels

Four levels. **Diligent is exactly today's behaviour**, so nothing changes for
anyone who does not touch the setting.

| | Relaxed | Diligent (today) | Persistent | Ambitious |
|---|---|---|---|---|
| For | quick chat, cheap scheduled jobs | everyday work | work that must be finished | work where a bit extra is welcome |
| Step ceiling (× planned budget) | 2×, at most 100 | 4×, at most 300 | 6×, at most 450 | as Persistent |
| Steps earned by progress alone (× planned) | 1× | 2× | 3× | as Persistent |
| Validation retries (general / after a list run) | 0 / 0 | 2 / 1 | 3 / 2 | as Persistent |
| Stall retries (announced but did not act) | 2 | 2 | 3 | 3 |
| Check before delivering | no | no | yes | yes |
| May do more than asked | no | no | no | yes, within §5 |

Rules that hold at every level:

1. **No level lowers a guard about honesty or safety.** False-claim retries, the
   stuck detector, mail authority, tool risk checks and confirmations are the same
   at Relaxed as at Ambitious.
2. **Relaxed still finishes.** It accepts the first complete answer and skips
   validation rounds; it does not allow an unfinished one. The stall retry stays at 2
   for that reason.
3. **Higher levels cost more.** The picker says so in plain words.

## 3. What exists today (the knobs)

Read on 2026-11-23, main at 543009cc.

| Knob | Today | Where |
|---|---|---|
| Base steps per turn | 40 (`max_iterations`, env override) | `captain_claw/agent.py:167` |
| Planned budget for a turn | from task count and requirements, up to 140 | `agent_pipeline_mixin.py:1463` `_compute_turn_iteration_budget` |
| Hard ceiling and free extension | 4× planned (max 300), free up to 2× | `agent_orchestration_mixin.py:87` `_turn_limits` |
| Extension decision | progress earns steps; Jev judges past the free limit | `agent_orchestration_mixin.py:57` `_budget_extends`, `:2537` |
| Floating budget | `CLAW_FLOATING_BUDGET` env, used by Bat workers | `agent_orchestration_mixin.py:75` |
| Stall retries | `MAX_STALL_RETRIES = 2` | `agent_orchestration_mixin.py:478`, used at `:4089` |
| Validation retry caps | general 2, after a list run 1; mode shifts them by one | `agent_completion_mixin.py:806–832` |
| Stuck detector | warm-up from the budget | `agent_orchestration_mixin.py:103` — **not touched by Drive** |
| Reasoning effort per turn | `reasoning_effort` on the chat frame → context variable | `web/ws_handler.py:544`, `llm/reasoning.py:81`, `flight-deck/src/stores/chatStore.ts:837` |
| Mode in logs | on usage rows and turn records | PR #771 (merged, main f33e4e18) |

Not read closely for this plan, to check during the build: how `use_contract_pipeline`
decides that a turn gets a contract (`agent_orchestration_mixin.py:2254`), and what
`flight_deck/tool_risk.py` already classifies.

## 4. Design

### 4.1 One module, levels as data

`captain_claw/drive.py`:

- `DriveLevel`: a frozen dataclass with one field per row of the table in §2.
- `LEVELS`: the four levels. `DEFAULT = "diligent"`.
- `turn_drive(level)`: a context manager that sets the level for the turn, the same
  pattern as `reasoning.turn_effort`. Tasks created inside the turn inherit it.
- `current()`: the level in effect: the turn's pick, else the agent's own
  default, else the default its cognitive mode suggests (§4.2), else Diligent.

A test fails if a `DriveLevel` field is not read anywhere, the same rule the mode
parameters now have.

### 4.2 Where the level comes from

1. **The chat box.** A picker next to reasoning effort. The frame carries `drive`;
   `ws_handler` wraps the turn in `turn_drive(...)`. "Auto" sends nothing.
2. **The agent's default.** A config value (`drive.default`), editable in the
   agent's settings, written the way reasoning effort's default is. Empty means
   "not set".
3. **The mode's suggestion, when the agent's default is not set.** Ionian and
   Phrygian suggest Persistent; every other mode, and no mode, suggests Diligent.
   This replaces the extra validation retry those two modes had. An explicit
   agent default always wins over the suggestion.
4. **Everything without a person in the chat** (scheduled jobs, hand-overs,
   Telegram, WhatsApp, links) uses the agent's default (2, then 3). New scheduled
   tasks are not forced to Relaxed. A level picked in one chat never leaks into a
   scheduled run.
5. **Workers** spawned by Vatra, Basna and Council run at Diligent unless the
   orchestrator passes a level. Bat workers keep their floating budget and ignore
   Drive.

### 4.3 What reads it

- `_turn_limits` takes the level's two multipliers and cap instead of the fixed
  4×/300 and 2×.
- The completion gate takes its two retry caps from the level.
  `completion_strictness` is removed from cognitive modes. Ionian and Phrygian
  agents get Persistent as their suggested default instead (§4.2), so they keep
  the extra retry and also gain Persistent's larger step ceiling and delivery
  check. That is more than they had, and it costs more; the picker in agent
  settings shows the suggested level so it can be turned down.
- The stall retry limit comes from the level.
- Persistent and Ambitious add the check in §4.4.
- Ambitious adds the initiative text in §5.

### 4.4 The check before delivering (Persistent, Ambitious)

One extra model call at the end of a turn that used tools or produced a file, before
the reply goes out: compare the result with what was asked, list anything missing
or unverified, and either continue working or deliver with the gaps stated.

- Skipped for turns with no tool calls (small talk, simple answers).
- At most one round, so it cannot loop.
- Uses the agent's own model.
- This is the part of Bat worth borrowing. Bat itself stays a separate mode: a
  durable loop with an independent judge on another model. Drive never starts Bat.

## 5. Initiative (Ambitious only)

The top level lets the agent do the obvious next step without being asked. This is
the only part of Drive that changes *what* gets done, so it is fenced:

- **Allowed:** more of the same task inside the agent's own workspace, such as
  checking a second source, adding the missing section, or producing the companion
  file the request clearly implies.
- **Never:** anything that leaves the workspace or is hard to undo, such as sending
  mail or messages, posting, paying, deleting, changing settings or creating
  accounts. Those stay governed by the existing guards and are not offered as
  "extra".
- **Always visible:** the reply separates what was asked from what was added, in
  one short section, so the extra can be ignored or undone.
- **Bounded:** extra work spends at most a fixed share of the turn's budget
  (proposal: 25% of planned steps).

Implemented as a short prompt section plus a harness check that the extra steps
used no outward tool. If the tool-risk classification is not good enough to rely
on, Ambitious ships as "Persistent plus suggestions" until it is.

## 6. Recording

The level goes on the same two records the mode now uses: a `drive` column on
`llm_usage` and on the deck's turn records, and the per-mode statistics gain a
per-level split. Without this there is no way to tell what each level costs or
whether it finishes more.

## 7. Phases

Each phase is its own PR and leaves Diligent unchanged.

| Phase | What ships | Visible change |
|---|---|---|
| 1 | `drive.py`, the context variable, config default, the frame field, recording in logs | none: every turn records "diligent" |
|   | *Phase 1 status (2026-11-23): built on branch `feat/drive-phase1`, not yet merged. The mode suggestion (§4.2 point 3) moves to phase 2 with the rest of the behaviour.* | |
| 2 | Budget limits, validation caps and stall retries read the level; `completion_strictness` leaves modes, Ionian and Phrygian suggest Persistent | Relaxed and Persistent work when set through config or the frame |
|   | *Phase 2 status (2026-11-23): built on branch `feat/drive-phase2` (stacked on phase 1). Relaxed keeps the one contract check and only drops the retries; Ambitious has Persistent's limits until phase 5.* | |
| 3 | Picker in the chat box, default in agent settings, cost wording; bundle rebuild | people can choose |
|   | *Phase 3 status (2026-11-23): built on branch `feat/drive-phase3`. The agent's own default lives in `~/.captain-claw/drive.txt` (read every turn, like the task-rephrase switch), with `drive.default` in the config as the fallback. A member of a shared agent gets no picker.* | |
| 4 | The check before delivering | Persistent and Ambitious catch gaps |
|   | *Phase 4 status (2026-11-23): built on branch `feat/drive-phase4`. The check is skipped when the reply was already judged against a contract (no second opinion on the same reply), on a turn's last allowed step, and when the checker fails or answers badly. A reply that is open about what it could not do passes.* | |
| 5 | Initiative, behind its own switch | Ambitious does bounded extra work |
|   | *Phase 5 status (2026-11-23): built on branch `feat/drive-phase5`. The switch is `drive.initiative` in the agent's config (no UI yet). Extra work is its own step after the answer has passed every gate, so the harness knows which tool calls are extra: only read-class calls and file-writing tools run, at most 25% of the base step budget in tool calls (10), never on an automated turn or a shared agent member's turn. The answer is delivered unchanged with what was added under it.* | |
| 6 | Statistics per level, and a comparison on real tasks | evidence for the default values |

Phases 1–3 are the useful core. Phases 4 and 5 are each optional.

## 8. Tests

- Diligent reproduces today's numbers exactly (`_turn_limits`, retry caps, stall limit).
- Each level's numbers reach the code that uses them.
- A level set in one chat turn does not apply to the next turn or to a scheduled run.
- No level changes the stuck detector, false-claim retries or mail authority.
- The delivery check runs at most once and is skipped for turns without tool calls.
- Extra work at Ambitious that touches an outward tool is refused.
- Every `DriveLevel` field is read somewhere.
- An Ionian or Phrygian agent with no default set runs at Persistent; with a
  default set, at that level; a chat pick wins over both.

## 9. Decisions (owner, 2026-11-23)

1. **Name:** Drive.
2. **Levels:** four: Relaxed, Diligent, Persistent, Ambitious. Ambitious is still
   built last (Phase 5), behind its own switch.
3. **Scheduled jobs:** use the agent's default. No special Relaxed default.
4. **Numbers:** the multipliers and caps in §2 are accepted as the starting
   values; Phase 6 may adjust them.
5. **Plan limits:** none. Every level is open to everyone; the daily allowance
   already limits spend.
6. **Mode retries:** Ionian and Phrygian suggest Persistent as the default level
   (§4.2), in place of their extra validation retry.

## 10. Risks

- **Cost surprise** at the top two levels. Mitigation: the wording in the picker,
  the per-level statistics, and the fixed ceilings.
- **Unasked actions** at Ambitious. Mitigation: the fence in §5, and shipping it
  last, behind its own switch.
- **Three overlapping "try harder" switches** (effort, Drive, Bat). Mitigation: the
  table in §1 in the docs and the picker's help text; Drive never launches Bat.
- **Relaxed mistaken for careless.** Mitigation: it never lowers an honesty guard
  and still has to deliver a complete answer.
