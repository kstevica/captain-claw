# Captain Claw v0.4.26 Release Notes

**Release title:** Code Apps & Memory Taxonomy

**Release date:** 2026-05-19

## Highlights

Captain Claw 0.4.26 lands two large additions on top of the 0.4.25 plan-mode foundation:

1. **Code Apps** — agent-authored, sandboxed mini-apps with their own backend (`backend.py`) and frontend (`frontend.html`), each running as a managed subprocess behind the Flight Deck proxy. Apps can publish read-only data endpoints via a `data_api` manifest block and consume each other's data through a typed `sibling()` SDK, so a Contacts app and a CRM app can share state without duplication.
2. **Typed memory taxonomy** for the insights layer — two new categories (`feedback`, `reference`), three new fields (`why`, `how_to_apply`, `polarity`), and a rewritten extraction prompt that captures *how the user wants you to work*, not just facts. Insights now distinguish corrections from confirmations, carry their motivation, and tell future-you when the rule applies.

Plus DeepSeek provider support, a four-bucket reversibility/blast-radius taxonomy in the system prompt, end-of-turn discipline, an exploratory-question response shape, a Flight Deck Ctrl+C fix, and assorted task-loop tightening.

Everything is additive and opt-in — 0.4.25 setups keep working unchanged. The insights database migrates additively on first launch (three nullable columns).

## What changed

### 1. Code Apps — `app_runner` tool + Flight Deck subprocess runtime

The biggest new surface in 0.4.26. The agent can now scaffold an interactive mini-app inside the user's Flight Deck workspace — pick a slug, get a `backend.py` (Python) + `frontend.html` (single file, sandboxed iframe), point the user at it. The app runs as a managed subprocess behind a per-slug Unix domain socket; Flight Deck proxies HTTP through to it.

- **`app_runner` tool** with actions: `scaffold` (create new app, full rewrite), `read_source` (load `backend.py` + `frontend.html` for in-place edits), `edit_file` (targeted edits, auto-restarts the subprocess), `restart`, `logs`, `proxy` (smoke-test by hitting the app's HTTP surface), `query_app` (chat-side reads against a sibling app), `list` (enumerate available apps + which ones publish a `data_api`).
- **`AppRuntime`** ([captain_claw/flight_deck/app_runtime.py](captain_claw/flight_deck/app_runtime.py)) — async subprocess pool with an idle reaper, per-app stdout/stderr ring buffers, log files at `code_dir/logs/{stdout,stderr}.log`, and `SIGTERM → 3s grace → SIGKILL` shutdown. Apps live under their own process group so the kill semantics work cleanly.
- **`app_sdk`** ([captain_claw/app_sdk.py](captain_claw/app_sdk.py)) — the `sibling(slug)` helper an app's `backend.py` imports to call another app's published endpoints (`get_json` / `post_json` / `request`). Auth is automatic per-call; misses raise `SiblingError` so the caller can degrade gracefully.
- **`data_api` manifest block** — apps declare which read-only endpoints they want siblings to be able to hit. Without a `data_api` block, sibling calls into the app return 403. Write endpoints are not publishable in v1 — read-only by design.
- **Self-repair loop** — after `scaffold`, the agent is expected to call `app_runner` with `action='proxy'` to smoke-test. On 5xx the agent calls `action='logs'`, reads the traceback, fixes `backend.py`, then `action='restart'`. The system prompt's app-authoring policy walks the agent through this loop.
- **Editing existing apps** — strict rule baked into the prompt: never re-scaffold an existing app (it wipes the source). The agent must `read_source` first, then `edit_file` per targeted change.
- **AppHost in Flight Deck** — `flight-deck/src/app-runtime/AppHost.tsx` renders the agent-authored `frontend.html` inside a sandboxed iframe on the **Code Apps** page. Authoring dialog wraps `scaffold`/`edit` flows with previews.
- **Datastore client + agent secret** — `app_datastore_client.py` gives apps a typed gateway into the shared datastore using per-app credentials (`agent_secret.py`); apps never see other apps' secrets.
- **Cross-app chat queries** — from the chat itself the agent can answer "how many notes do I have?" by calling `app_runner` with `action='query_app'` against an app that publishes a `data_api`, instead of scaffolding yet another notes app.

The system prompt section that governs all of this lives in [captain_claw/instructions/system_prompt.md](captain_claw/instructions/system_prompt.md) under "App authoring policy" — it precedes the visualization policy and is the deciding factor between "build a notes app" (interactive, persistent) and "make me a chart of these numbers" (read-only HTML).

### 2. Typed memory taxonomy in `insights`

The `insights` layer (auto-extracted facts) gained two new categories and three new optional fields so it can capture *how the user wants you to work*, not just what they know.

- **New category: `feedback`** — corrections AND confirmations about how the agent should approach work. Required `polarity` field: `"positive"` (the user confirmed a non-obvious choice, "yes exactly, keep doing that") or `"negative"` (a correction, "stop summarizing every turn"). Required body shape: lead with the rule, then `why` (motivation) and `how_to_apply` (when/where it kicks in).
- **New category: `reference`** — pointers to where information lives in external systems (a Google Doc, a Linear project, the FRiC startups database) without snapshotting the contents. Memory tells the agent where to look; the actual data stays in the system of record.
- **New columns** in the `insights` SQLite table: `why TEXT`, `how_to_apply TEXT`, `polarity TEXT` — all nullable. Migration is additive (`ALTER TABLE … ADD COLUMN`) and happens on the next launch. No backfill required; older rows render fine with the new fields empty.
- **`polarity` is normalized** to null on any non-`feedback` insight, so storage stays clean even if the extractor produces noise.
- **`CONFLICT_STAGED_CATEGORIES`** now includes `feedback` — competing feedback rules (e.g. "always summarize" vs "never summarize") are staged into the pending-review queue instead of being silently de-duped.
- **Extraction prompt** ([insight_extraction_system_prompt.md](captain_claw/instructions/insight_extraction_system_prompt.md)) was rewritten end-to-end with the typed-taxonomy `when_to_save` / `how_to_use` / `body_structure` shape, an explicit "save from success AND failure" rule (confirmations are the quieter half — `"yes exactly"`, accepting an unusual choice without pushback), a "what NOT to save" list (ephemeral task state, derivable code/file/git information, volatile peer-agent rosters, activity summaries), and everyday-agent examples (contacts, meetings, deadlines, FRiC references) rather than coding examples.
- **Surfaced-insights rendering** ([agent_context_mixin.py:1062](captain_claw/agent_context_mixin.py:1062)) — the per-turn injected insights now include polarity tags (`[feedback/pos]`, `[feedback/neg]`) and indented `Why:` / `How to apply:` lines so the agent can see *why* the rule exists and judge edge cases. The system-prompt insights block teaches the agent to read those tags, to save successes as well as corrections, and to verify-before-recommending (file paths, deadlines, references, decisions can all go stale).

### 3. Reversibility and blast-radius taxonomy

The system prompt gained an explicit **"Executing actions with care"** section that replaces the old scattered "confirm before dangerous commands" line with a four-bucket checklist.

1. **Destructive** — deleting files / contacts / calendar events / notes / app records / memory insights; overwriting a user file the user did not just ask to overwrite; `app_runner action='scaffold'` on an existing app; clearing or resetting persistent state; shell `rm`, dropping a datastore table.
2. **Hard-to-reverse** — sending messages (email, Discord, WhatsApp, Telegram), publishing posts; scheduling/cancelling/moving calendar events the user did not propose in-turn; moving money, placing orders, submitting forms; removing/downgrading dependencies; modifying `config.yaml` keys that change persistence locations; restructuring `saved/` folders that other sessions depend on.
3. **Shared-state / visible-to-others** — posting in channels, commenting on PRs/issues, replying to threads; publishing an `app_runner` app's `data_api` or opening it to siblings; creating `cron` jobs or sister sessions that will continue running; changing fleet membership or shared resource permissions.
4. **Third-party upload** — sending user content (documents, images, audio, transcripts, personal data) to external renderers / pastebins / gists / transcription services / AI APIs the user did not authorize for *this* content. Even later-deleted content may be cached or indexed.

Plus an "obstacle ≠ shortcut" rule (don't bypass a safeguard to make an error go away — investigate first; unfamiliar files / cron jobs / lock files may represent in-progress work) and a "match scope to what was asked" rule (one-time authorization stands for the scope named, not beyond).

### 4. End-of-turn discipline

New section in the system prompt, immediately after the existing "Never announce intent without acting" block — they're paired rules for the front and back of a turn. After the last tool call: one or two sentences max, what changed and what's next (if anything). No generic "Let me know if you need anything else" closers. No section headers wrapping a short reply. The deliverable IS the response when the deliverable is prose (a memo, a research summary).

### 5. Exploratory-question response shape

Placed immediately before "Bias toward action" as the read-first exception. When the user asks an open-ended question ("what could we do about X?", "should we use A or B?", "any ideas?"), respond in 2–3 sentences with a recommendation and the main tradeoff, presented as a redirectable proposal — not a decided plan, not a tool-call cascade. This is the one case where the front-of-turn "never announce intent" rule yields: the recommendation IS the deliverable. Bias-to-action still applies to everything else.

### 6. DeepSeek provider support

The DeepSeek family is now a first-class provider through [`captain_claw/llm/__init__.py`](captain_claw/llm/__init__.py). The provider switch covers function-calling, streaming, prompt-cache reporting, and the same guard/orchestration plumbing the OpenAI/Anthropic/Gemini/Ollama/OpenRouter providers go through. Configure via the standard model field in `config.yaml` (`model: deepseek/<model-id>`).

### 7. Flight Deck Ctrl+C exits on first press

Before 0.4.26: hitting Ctrl+C in the Flight Deck terminal triggered uvicorn's graceful shutdown, which waited *indefinitely* for in-flight streaming responses (consult SSE, ndjson event streams, the busy stream) to close. Users had to press Ctrl+C two or three times for the process to actually exit, and the log filled with one `CancelledError` traceback per cancelled stream.

- **`uvicorn.run(..., timeout_graceful_shutdown=3)`** ([flight_deck/server.py](captain_claw/flight_deck/server.py)) — graceful shutdown is now capped at 3 seconds. After that, uvicorn force-closes remaining connections and proceeds to the lifespan shutdown (which already handled managed agents + app subprocesses + the vast.ai poller cleanly).
- **`_SuppressShutdownCancelFilter`** — a small logging filter on `uvicorn.error` that drops the noisy "Exception in ASGI application" records whose root cause is an `asyncio.CancelledError` raised during shutdown. Real ASGI exceptions (different root cause) still pass through. Each open browser tab still cancels its own stream on exit; the user just no longer sees a traceback per tab.

End-to-end behavior: single Ctrl+C, ~3 second wait, clean exit.

### 8. Task-loop tightening

Several smaller improvements rolled in alongside the main features:

- `agent_orchestration_mixin.py` got nano-mode tool-set hardening, eco-mode lazy tool definitions with intent-based preselection (regex against the user message + tools used recently in this session + always-on MCP tools), and a force-script-mode tool restriction for cases where the LLM must shell out instead of driving a long tool chain.
- Stall detection (`_STALL_FIRST_LINE_RE`) catches "Let me look that up", "I'll fetch the file now"-style intent-only replies and silently retries up to twice per turn with `tool_choice="required"`, so weak models no longer leave the user holding an empty turn.
- `agent_tool_loop_mixin.py` and `agent_session_mixin.py` saw matching adjustments to keep the new lazy-tool-definition path consistent with the registry's view of what's available.

### 9. Plan-mode fixes (carried from late 0.4.25)

Two follow-up commits since the 0.4.25 cut hardened plan mode in the cases that exposed during real runs:

- **`plan_mode.py`** stabilized — fixes around plan loading, JSON parsing edge cases, and how `OrchestratorTask` fields round-trip after a revision.
- **`tools/write.py`** picked up small robustness improvements for the planner's preferred `saved/tmp/<slug>.md` write pattern.
- **`orchestrator_worker_prompt.md`** clarification on per-step output expectations when the step is part of a plan-mode workflow.

## File / module additions

| File | Purpose |
|---|---|
| `captain_claw/app_authoring_plan.py` | Authoring-time plan generation for new code-apps (manifest + scaffold structure). |
| `captain_claw/app_sdk.py` | `sibling(slug)` helper an app's `backend.py` uses to call another app's published `data_api`. |
| `captain_claw/tools/app_runner.py` | The `app_runner` tool — `scaffold` / `read_source` / `edit_file` / `restart` / `logs` / `proxy` / `query_app` / `list`. |
| `captain_claw/flight_deck/agent_secret.py` | Per-app credential issuance so apps reach the datastore without seeing other apps' secrets. |
| `captain_claw/flight_deck/app_builtin_routes.py` | Built-in FD routes the code-app runtime exposes. |
| `captain_claw/flight_deck/app_code_routes.py` | HTTP routes for browsing / editing an app's source from Flight Deck. |
| `captain_claw/flight_deck/app_datastore_client.py` | Datastore gateway scoped to a single app's credentials. |
| `captain_claw/flight_deck/app_entities.py` | Entity definitions for the code-apps surface. |
| `captain_claw/flight_deck/app_files.py` / `app_files_routes.py` | File browser surface for an app's code dir + logs. |
| `captain_claw/flight_deck/app_manifest_authoring.py` / `app_manifests.py` | Manifest authoring + registry (includes `data_api` block). |
| `captain_claw/flight_deck/app_routes.py` | Top-level code-apps API. |
| `captain_claw/flight_deck/app_runtime.py` | Subprocess runtime + idle reaper + log pumps + per-app socket. |
| `captain_claw/flight_deck/app_subprocess.py` | One-app process lifecycle wrapper. |
| `captain_claw/flight_deck/examples/notes_demo/` | Worked example of a code-app (backend + frontend + manifest). |
| `captain_claw/flight_deck/examples/_smoke_test*.py` | Smoke tests for the scaffold + self-repair loop. |
| `flight-deck/src/app-runtime/` | Front-end app-host: `AppHost.tsx`, `AuthoringDialog.tsx`, primitives (`ActionButton`, `EntityDetail`, `EntityList`, `InboxList`, `Upload`), surface renderer, store, types, hooks. |
| `flight-deck/src/pages/AppCodePage.tsx` | Code-apps page. |
| `flight-deck/src/stores/appCodeStore.ts` | Code-apps frontend state. |
| `RELEASE_NOTES_0.4.26.md` | This file. |

## Existing files materially changed

- `captain_claw/__init__.py` (version + build date), `pyproject.toml` (version).
- `captain_claw/insights.py` — `feedback` + `reference` categories, `polarity` enum, `why` / `how_to_apply` / `polarity` columns + migration, `add()` / `update()` / `_row_to_dict` / extraction-loop wiring.
- `captain_claw/instructions/insight_extraction_system_prompt.md` — full rewrite with typed taxonomy + everyday-agent examples.
- `captain_claw/instructions/system_prompt.md` — blast-radius taxonomy, end-of-turn discipline, exploratory-question shape, app-authoring policy refinements.
- `captain_claw/agent_context_mixin.py` — surfaced-insights rendering with polarity tags + `Why:` / `How to apply:` lines; insights system-prompt block expanded.
- `captain_claw/agent_orchestration_mixin.py` — nano / eco / force-script tool restriction, stall detection, intent-based tool preselection.
- `captain_claw/flight_deck/server.py` — `timeout_graceful_shutdown=3`, `_SuppressShutdownCancelFilter` logging filter, code-app routes registration.
- `captain_claw/llm/__init__.py` — DeepSeek provider integration.
- `captain_claw/plan_mode.py`, `captain_claw/tools/write.py`, `captain_claw/instructions/orchestrator_worker_prompt.md` — late-0.4.25 plan-mode follow-ups.
- `flight-deck/src/components/agents/ChatPanel.tsx`, `flight-deck/src/components/agents/PlanCard.tsx`, `flight-deck/src/stores/chatStore.ts`, `flight-deck/src/components/layout/Sidebar.tsx` — chat / plan-card / sidebar updates for the new code-apps surface.

## Migrating from 0.4.25

Nothing required.

1. **Insights DB.** First launch on 0.4.26 adds three nullable columns (`why`, `how_to_apply`, `polarity`) to the `insights` table. Older rows render fine; new extractions populate the fields where they apply. No backfill needed.
2. **System prompt.** The new sections (blast-radius, end-of-turn, exploratory questions) are additive. If you've customized your personality / fleet instructions, nothing in them is invalidated.
3. **Code apps.** Pure addition. You can ignore the `app_runner` tool entirely and 0.4.26 behaves like 0.4.25-plus-fixes.
4. **DeepSeek.** Opt-in via `model:` field in `config.yaml`. No action needed if you're on a different provider.
5. **Flight Deck.** Restart the FD process to pick up the new graceful-shutdown timeout and log filter. If you run the packaged binary (`captain_claw.spec`), rebuild via `./build.sh`.

## Known limitations / next steps

- **Code apps are single-process per slug.** The runtime spawns one subprocess per app slug; horizontal scale-out for a single app is not supported. Fine for personal / fleet-of-one workloads.
- **`data_api` is read-only.** Apps can't publish write endpoints to siblings in v1. Cross-app writes still go through the user (a chat turn that explicitly says "write this to the notes app").
- **No insight back-migration helper.** Existing rows stay as-is; if you want old `preference` / `workflow` rows restructured into the new typed-memory shape with `why` / `how_to_apply`, that's a manual `update()` per row today.
- **Stall detection is heuristic.** The first-line regex catches the common cases ("Let me look…", "I'll fetch…") but a model that emits an intent-only reply in a fresh phrasing will slip through and the turn ends empty.
- **End-of-turn discipline is prompt-level.** Models that have a strong baseline tendency to recap (smaller / older releases) will still occasionally recap. The instruction reduces it; it doesn't suppress it entirely.
