# Captain Claw v0.4.22 Release Notes

**Release title:** Nano Mode, Remote GPU, Smart Web Fetch, and Token-Speed Telemetry

**Release date:** 2026-04-28

## Highlights

Captain Claw 0.4.22 broadens the runtime story at both ends of the spectrum. On the small end, **Nano Mode** ships a restricted-tool runtime tuned for tiny local models (Qwen3, Llama 3.2, Phi) so a 3B-parameter model on a laptop can still drive a useful loop. On the big end, **vast.ai remote GPU** support means you can rent a H100/A100 by the hour and have the agent transparently route Ollama traffic to it (auto-wake on first request, auto-sleep when idle). Tooling gets two upgrades: **smart `web_fetch`** auto-falls-back from a plain HTTP fetch to a headless browser when the page is JS-rendered, and the new **prompt builder** lets you compose multi-step prompt templates with variable interpolation. The UI now shows **live token-per-second telemetry** in the agent sidebar so you can see at a glance which model is dragging.

This release also includes a **major reliability fix** for local LLMs that fall into empty-response loops, plus tighter handling of memory-injection content that previously could leak into user-facing replies on smaller models.

## Nano Mode — Tiny-Model Runtime

A new mode for small local models (3B–8B params) that don't have the steering capability to handle the full 44-tool surface. Toggle the **Nano Mode** chip in Flight Deck or set `mode: nano` in the agent config.

**What it changes:**

- **Restricted tool set** — only `shell`, `write`, `read`, `edit`, `glob`, `datastore`, and `insights` are surfaced. The dropped tools (web search, browsing, calendar, mail, image generation, MCP, etc.) confuse small models and waste tokens.
- **Compressed system prompt** — a separate `nano_system_prompt.md` template, ~40 lines instead of 400+. Tells the model to either reply directly for chit-chat OR write a script and run it via `shell` for tasks. Stops the "reason step-by-step over 20 tool calls" failure mode.
- **Aggressive context filtering** — empty assistant messages (a common failure in small-model history) are stripped at message-build time so the model doesn't pattern-match its way into a self-reinforcing empty-output loop.
- **Memory-injection delimiters** — semantic, deep, cross-session, and playbook context notes are now wrapped in `[INTERNAL CONTEXT — reference only, do not repeat in your reply]` markers. Small/cheap models were echoing the raw memory dump back as their reply; the explicit delimiter stops that.

**Recommended models for nano mode:**

- `qwen2.5:3b`, `qwen2.5:7b`
- `llama3.2:3b`
- `phi3:mini`, `phi3:medium`
- Larger Qwen3 quantized to fit on a laptop (`qwen3.6:35b:iq3` runs on a 24 GB Mac at ~6 tok/s)

## Remote GPU via vast.ai

Captain Claw can now drive an Ollama server hosted on vast.ai with auto-wake and idle-sleep. Useful for occasional heavy lifting (a 70B model for an evening's work) without paying for a 24/7 cloud GPU.

- **Auto-wake on first request** — `vastai_wake.maybe_wake_instance(base_url)` runs before every Ollama call. If the instance is stopped, it's started, then the request blocks until SSH is reachable.
- **Idle sleep** — a configurable timer stops the instance after N minutes of no requests. Avoids the "I left it running over the weekend" bill.
- **Per-agent override** — set `provider: ollama` + `base_url: https://<instance>.vast.ai:11434` in the agent's model override. No global config change needed.
- **Lazy Ollama provider construction** — the wake hook is a no-op when the URL points at a local server, so existing local-Ollama setups are unaffected.

## Smart `web_fetch`

The `web_fetch` tool now decides between plain HTTP and a headless-browser fetch based on what comes back.

- **First attempt:** plain HTTP GET via `httpx`.
- **Fallback trigger:** if the response is < 2 KB *or* the body is dominated by a JS bundle reference (no actual rendered text), the tool transparently retries through `playwright` and waits for `networkidle` before extracting text.
- **Result:** SPAs (React/Vue/Angular sites) that used to return `<div id="root"></div>` now return the actual rendered article text, while static sites stay fast (no Playwright cold-start tax).
- **Auto-API capture** still works on both paths — URLs containing `/api/` or `/v[0-9]+/` are auto-registered as APIs in the agent's APIs memory regardless of fetch path.

If Playwright isn't installed, the tool returns the original HTTP body with a one-line note suggesting `playwright install`. No hard failure.

## Prompt Builder

A new in-Flight-Deck tool for composing reusable prompt templates with variable slots.

- Define a template once with `{{variable}}` placeholders (`{{user_name}}`, `{{project_root}}`, `{{deadline}}`, etc.).
- Save it under a name; recall and fill in the variables when you want to fire a structured request.
- Templates support multi-step prompts (e.g., "research X, then summarize, then draft an email") so the model gets a coherent multi-stage instruction in one shot.
- Built-in variables: session ID, workspace path, user email, today's date.

## Live Token-Speed Telemetry

The agent sidebar now shows **real-time tokens-per-second** for the current LLM, updated as each completion streams.

- Display format: `<output_tps> tok/s / <total_llm_tps> llm` — for example `6.9 tok/s / 13.3 llm`.
- The first number is the visible output rate (what the user sees streaming); the second is the underlying LLM rate including reasoning tokens.
- Computed from streaming chunk timestamps — no extra LLM call.
- Especially useful for comparing local-Ollama setups (a 30B model at 2 tok/s on a Mac vs 60 tok/s on vast.ai H100 is now visible at a glance).

## Gaming System

A new "games" subsystem that lets the agent play turn-based text games against itself or the user. Initial games included:

- **Number guesser** — agent picks a number, user narrows it down.
- **Word-association** — agent extends a chain, user judges.
- **Twenty questions** — agent picks a concept, user asks yes/no questions.

Primarily a stress-test for the agent's loop and memory, but also a pleasant cognitive-tempo break for users.

## Reliability Fixes

### Empty-response loop on local Ollama models

Earlier versions could fall into a self-reinforcing failure loop with thinking-style local models (Qwen3, DeepSeek-R1): the model would emit only its `<think>` block and an empty `content` field, the empty response would be persisted to the session, the next turn would see prior `assistant: ""` entries in history, and the model would conclude that empty IS the correct response. All subsequent turns failed identically.

**Fixes:**

- `OllamaProvider` now sends `"think": false` in the request body by default, so Qwen3-style thinking models put their answer in `content` directly instead of dumping plans into the `thinking` field.
- Empty assistant responses are no longer persisted to the session.
- When building the LLM request, empty assistant messages from earlier in the session are filtered out, so existing already-poisoned sessions self-recover.
- `_strip_reasoning_artifacts` has a fallback: when the entire response is `<think>…</think>` with nothing after, the last paragraph of the thinking block is surfaced as the answer instead of returning empty.
- LiteLLM and streaming paths now accumulate `reasoning_content` / `thinking` / `reasoning` deltas across all three field names different providers use, and surface them as fallback content when the main `content` field is empty.

### Memory-injection echoes on small models

Cheap cloud models (deepseek-v4-flash, qwen3-flash) and small local models occasionally treated context-injection notes — added to the prompt as fake `assistant` messages with `tool_name=memory_context` — as something to mimic, dumping the raw memory match block into their user-visible reply.

**Fix:** all context-injection notes (memory_context, semantic_memory_context, deep_memory_context, cross_session_context, playbook_context, planning_context, list_task_memory, scale_progress, todo_context, contacts_context, scripts_context, workspace_manifest) are now wrapped at message-build time with explicit `[INTERNAL CONTEXT — reference only, do not repeat in your reply]` / `[END INTERNAL CONTEXT]` delimiters. Doesn't add tokens to the underlying note, just frames it.

### Memory tool output no longer leaks as user reply

`memory_select`, `memory_semantic_select`, and `memory_deep_select` are now classified as monitor-only tool names. Their debug output stays in the trace timeline and never feeds back into the model's context as a candidate "previous answer".

## Other Changes

- **LLM session logger** writes every LLM call to `logs/<session>/session_log.md` when `logging.llm_session_logging: true` — useful for debugging tool-call loops without re-running.
- **Cancellation** is now plumbed through the main iteration loop via `agent.cancel_event` — Ctrl+C / ESC / WS cancel breaks cleanly between iterations rather than waiting for the in-flight LLM call to finish.
- **`_is_monitor_only_tool_name`** centralizes the list of tool names whose output is monitor-only and shouldn't feed back into the model context, replacing scattered string comparisons.

## Upgrading

```bash
pip install --upgrade captain-claw
```

No config migrations required. Existing sessions that fell into the empty-response loop will self-heal on the next turn (the empty-message filter strips poisoned history at request build time).

To enable Nano Mode on an existing agent: open the agent's Flight Deck card and click the **Nano Mode** chip, or add `mode: nano` to the agent's config override.

To enable vast.ai remote GPU: provision an Ollama instance on vast.ai, then set the agent's model override to `provider: ollama` with `base_url: https://<your-instance>.vast.ai:11434`.
