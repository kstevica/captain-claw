# Context engine — what each LLM call sees, and why

**Status:** P0, P1 (365b39c4), P4 (4e72f8e2), P2 (5da8d309), P3 (baabbaf2) and P5 implemented on `fix/context-placement-weak-models` (2026-10-08). P6 next.
**Origin:** weak models (Haiku, DeepSeek V4 Flash, GPT-6 Luna, GLM flash) behave well in a fresh
session and degrade once it is crowded: truncated answers, echoed notes, "stupid" replies.
The fixes already on `fix/context-placement-weak-models` (context notes moved into one block on the
turn's question, compaction-turn index, earlier turns' tool steps merged) handled where things sit.
This plan handles what goes in at all.

---

## Where a crowded prompt's tokens actually go

Measured on 116 local session DBs (dev agents), with the new layout. Prod calibration pending.

| Part of the request | Size | Counted in the budget? |
|---|---|---|
| Current turn's own tool chain | median **82%** of turns over 4k tokens (p90 46k, max 484k); `web_fetch` + `datastore` + `web_search` are 92% of it | yes |
| Tool schemas | 19–23k tokens on every call (55 tools, ~93k chars) | **no** |
| Earlier turns, as sent | median ~800 tokens (old tool results are already dropped) | yes |
| Context-notes block | median 2.3k, p90 11k | yes |
| Synthetic messages saved as `user` (nudges, fleet notices, cron prompts, time anchors) | 8.8% of history, present in 66% of turns | yes |
| `reasoning_content` replayed on assistant messages | +17.9% on top | **no** |
| Debug echoes persisted as session messages (`memory_select`, `pipeline_trace`, …) | 14.5% of stored transcript characters, re-indexed by semantic memory | filtered from prompts, counted by compaction |

Two more findings:

- The per-minute clock and host lines sit in the system prompt's dynamic tail, **ahead of all
  history**, so no provider can reuse cached history across turns.
- The 160–200k budget almost never binds (2 of 267 snapshots dropped anything). Crowding is not a
  budget overflow; it is unrelated, synthetic and duplicated content inside a large budget.

## Design rules

1. **Relevance or absence.** Every source either passes a gate for this turn or is not sent.
2. **Cards, never fake turns.** Recalled material is a dated, delimited note ("NOT the current
   request") inside the per-turn block. Never injected as user/assistant turns.
3. **The user sees everything; the model sees what helps.** Filtering applies to the prompt only,
   never to transcripts.
4. **Tiers decide budgets.** No inferred "weak model" profile: the user picks the tier, and model
   windows keep growing (32k six months ago, 256k now). The engine makes the budget it is given
   accurate and clean; it does not shrink it.
5. **Stable prefix.** Anything that changes per turn lives in the per-turn block, never ahead of
   history.
6. **Every decision is visible.** Per-section tokens and "why included" for each recalled item.

## User decisions (2026-10-08)

| Question | Decision |
|---|---|
| May recall cross channels (a WhatsApp turn surfacing an FD-chat thread)? | **Yes.** |
| Automated traffic (cron, autonomy, fleet notices) | **Own sessions, still accessible to the user.** (P6) |
| Smaller working budget for weak models? | **No.** Keep the tier the user selected. |
| Topic classifier model | **Same model as the agent.** |
| Prod data for calibration | Prod `sessions.db` provided (201 sessions, deepseek-v4-flash). The matching `conversation_topics.db` sits next to it under `fd-data/<agent>/data/home-config-parent/.captain-claw/`. |

---

## P0 — measure, and no-regret hygiene

**Done.** Replay of the largest prod session (deepseek-v4-flash) after P0: the first request of
a turn repeats **96–98%** of the previous turn's last request (prod measured 5.6% cached on those
calls before). Remaining weight is in earlier turns' replayed reasoning: ~39k of ~50k prior-history
tokens — the P4 decision below. Compaction no longer counts debug echoes or reasoning, so it stops
firing on tokens the model never sees.

Nothing user-visible changes except cheaper, cleaner prompts.

- **Trace.** `last_context_window` gains tokens per section: system static and dynamic, tool
  schemas, prior history, current chain, notes per kind, `reasoning_content`, synthetic messages
  suppressed. Additive keys; the existing meters keep reading the aggregate fields.
- **Count what is sent.** `reasoning_content` is counted in message token counts; tool-schema
  tokens are counted once per tool-set signature and reported (not yet subtracted from history).
- **Debug echoes become UI-only** where nothing reads them back from the session (see the
  inventory); they stop inflating compaction and semantic memory.
- **Semantic memory quick fixes.** bm25 rank sign (every keyword hit scored 1.0); the passive note
  stops searching the active session, which is already in the prompt; temporal decay applied after
  the min-score floor.
- **Clock out of the system prompt.** Per-minute time and host lines move to an `env_now` note in
  the per-turn block, which is never dropped by fitting. History becomes cacheable across turns.
- **Replay harness** (`scripts/context_replay.py`): rebuild every human turn of a stored session,
  run the real `_build_messages` on a stub agent, print per-section tokens, synthetic messages
  sent, duplicates between history and notes, and predicted prefix stability. Runs on read-only
  copies of prod DBs.

## P1 — message provenance (audit item 5)

**Done.** In the fleet-heavy prod session the model's history leaves out 31 synthetic messages per
turn (fleet notices, earlier correctives with the replies they rejected); the fleet changes since
the previous turn ride as one line in the context block. Bridges now send a `surface` field on
every frame; the surface rules ride only on that surface's turns.

- Every session message carries an **`origin`**: `human`, `assistant`, `tool`, `nudge`
  (correctives and guard nudges), `fleet_notice`, `cron`, `delegated_result`, `time_anchor`,
  `life_tick`, `surface_context` (glasses/WhatsApp system block), `notification`, `automation`.
  Writers pass it explicitly; old messages fall back to a regex classifier over code-literal
  prefixes (`message_origin.py`). A test fails if a `role=user` write passes no origin.
- `_build_messages` does **not replay earlier turns'** nudges, correctives, time anchors and life
  ticks. The current turn's own nudges still go out.
- Runs of fleet notices collapse to one rolling line ("fleet changes since your last message").
- The glasses/WhatsApp system block moves to `session.metadata["surfaces"][channel]` and is
  rendered in the per-turn block only on that channel's turns; today it is a one-time `user`
  message that leaks brevity into FD replies in the shared lane.
- Transcripts are unchanged: users still see every message.

## P2 — topic store repair and ranked search

**Done.** A topic is now made of what people typed and the final reply of each turn they
opened; progress lives in the store (everything after the newest classified or attempted
message is pending), so restarts and compaction neither re-read the session nor stall the pass.
`rank_legs` / `rank_topics` / `search_topics` rank with FTS5 bm25 (long words by stem) plus an
optional embedding leg with a floor (local model2vec: related topics score 0.4–0.8, unrelated
≤ 0.2); typed search lists word matches, then substrings, then meaning-only matches. The
classifier sees every content word of its batch against the topics, the 15 most relevant and
the 10 most recent topics with whole summaries. Topics built mostly from machine text are
hidden once per store (still listed in the panel; starring one, new conversation filed under
it, or `POST /api/topics/{id}/hide` brings it back). Narration is off by default
(`include_narration`), and `interval_messages` (now 6) counts conversation messages. CLI-mode
turns (terminal and remote platforms on the main agent) run the pass too; the web server's
per-user Telegram agents don't, since the store is the owner's.

- Ingest only `origin=human` user messages and final assistant replies (no narration duplicates,
  no fleet/cron/nudge text: 59% of user excerpts today).
- Message-id watermark instead of an in-memory index (it stalls after compaction); page through the
  backlog instead of keeping only the last 15; call `mark_seen`; run from the agent's own turn end
  on every channel (today FD web chat only).
- Classifier stays on the agent's model, but sees the top-K topics relevant to the batch plus the
  10 most recent, with full summaries — not all 300 cut to 160 chars.
- `rank_topics(query)`: FTS5 bm25 over the already-maintained `topics_fts` (label×3, keywords×2,
  summary×1) fused with a model2vec cosine leg. Replaces the whole-string `LIKE`.
- Store `session_id` and `channel` on topic messages; hide junk topics reversibly.

## P3 — topic recall card and the `topics` tool

**Done, in shadow mode.** `topic_recall.decide` picks at most one topic per human turn from
the turn's own words (attachment markers, links, paths and ids left out; no LLM call): the
word and meaning rankings put the same topic first and something besides one shared word sets
it apart, or 2+ matched words lead the next topic by bm25 × 1.3, or the meaning match is ≥ 0.5
and 0.15 ahead of the runner-up; ties never count. A topic whose latest messages are still in
the live session is never recalled, and nothing is recalled on a turn that asked for no memory,
for public visitors, BotPort dispatches or Iskra bodies. `conversation_topics.recall`
= `shadow` (default: the decision is logged and stored as `last_context_window.topic_recall`,
nothing is sent) | `on` (the card rides the per-turn block) | `off`. Pinned topics
(`topics pin`, 5 turns by default, at most 3; one on small tiers) ride the block in every
mode, on people's turns only (cron and autonomy turns neither show nor spend them). Members recall only
topics they spoke in, with their own excerpts; the owner's cards carry the owner's excerpts.
The meaning leg uses only an in-process embedder (model2vec) in the turn path. Next: read the
shadow decisions on real traffic (the prod `conversation_topics.db` and logs), then flip to
`on`.

- Query from the turn's human text: content words, EN+HR stopwords removed, ≤8 terms, prefix
  stems. No LLM keyword call (~3 s on the agent's own model, and it would break the per-turn
  freeze).
- Gate (abstain by default): FTS and embedding agree, or ≥2 matched terms with a clear margin, or a
  strong embedding margin. Skip topics whose latest messages are still in the live history.
- One card (label, date, ≤300-char summary, 2–3 excerpts, pointer to `topics get`) inside the
  per-turn block. Size scales with the tier budget, not a fixed weak-model cap.
- Recall crosses channels (user decision). Shared-agent members see only topics they have
  excerpts in.
- Shadow mode first (logged, not sent), then inject.
- `topics` tool: ranked `search`, one-call `recall`, `pin`/`unpin` (keeps a topic in the block for
  N turns — the real way to "include it in context", since tool results are dropped from later
  turns). One line in the system prompt.

## P4 — accurate accounting within the tier budget (audit item 4, reshaped)

**Done.** History budget = tier window − system prompt − the tool schemas measured on the
previous call. Earlier turns' `reasoning_content` is counted wherever the provider sends it
back (LiteLLM, non-Anthropic — DeepSeek's thinking mode requires it), and nowhere else.
Compaction triggers on what the next turn would send (old tool output and hidden synthetic rows
weigh nothing), plus a storage guard at 4× the window. A text answer stopped by the output
limit or a broken stream is continued up to twice and joined at a clean seam; a reasoning tail
is re-asked, not continued; a cut-off compaction summary is retried once with more room.
`context.tool_result_max_chars` (0 = off) caps tool results.

- Tool schemas and `reasoning_content` are subtracted from the history budget, so the tier's
  window is what is actually sent.
- `reasoning_content` from earlier turns is replayed only where the provider requires it.
- `finish_reason == "length"` and interrupted streams are failures (continue or retry), not
  successful answers.
- Optional per-tier knobs, **off by default**: tool-result size cap, masking older tool results
  within a long turn.

## P5 — one retrieval path (audit item 6)

**Done (scoring shared; stores keep their own indexes).** `captain_claw/retrieval.py` holds the
rules every recall source uses: content words (EN+HR stopwords; attachment markers, links, paths,
ids left out), FTS5 OR-queries with stems for long words and prefixes for 4+ letter words,
FTS-consistent matched words, bm25 normalised to 0..1, reciprocal-rank fusion, and age decay
after the relevance floor. Topics and semantic memory use it (semantic memory's keyword leg had
no stems and no Croatian stopwords; it keeps paths, commits and ids, which topics drop).
Insights are query-relevant (`insights.context_mode: relevant`, the default), refreshed every
turn: the `core_items_in_prompt` (3) most important ones and every behaviour rule (feedback,
preference) of importance 8+ always, then those the turn's words match in content, tags or a
rule's why / how-to-apply (two words, the only word asked, or one rare word); expired insights
never show. The old query path ANDed every word of the message and so never matched. `context.notes_allocator: capped` (off by default) cuts each note to its source's
share of the notes budget before fitting, so one large note can't crowd out the rest. Not done:
merging the stores into one index (semantic memory's chunk index and the topic/insight FTS
tables stay separate).

One primitive (normalised bm25 + cached embeddings, fused, per-source decay after the floor,
dedupe by message id) shared by topics, the session-history archive, workspace and insights.
Insights become query-relevant instead of top-8 by importance. `_fit_context_notes` becomes a
per-source capped allocator behind a flag.

## P6 — sessions that don't grow forever (audit item 7)

- Automated traffic (cron, autonomy, fleet notices) gets its own sessions, still listed for the
  user (user decision).
- A New-session button in FD; rotation on an explicit cue ("new topic", "nova tema").
- Compaction writes a deterministic, topic-grouped digest with message-id handles instead of the
  agent's model summarising its own history.

## Order

P0 → P1 → (P2 ∥ P4) → P3 → P5 → P6

## Verification

- Replay harness on prod copies before and after every phase: 0 synthetic messages replayed after
  P1; per-section tokens; predicted cache-hit share.
- Recall quality (P3): labelled sets for follow-ups, returns to old threads, new subjects, noisy
  turns; precision first, thresholds in config.
- Unit tests per phase; the per-file regression diff against `main` (see the test-baseline notes).
