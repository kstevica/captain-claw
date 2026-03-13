# Summary: chat_handler.py

# chat_handler.py Summary

Handles incoming chat messages from the web UI, routing them through an AI agent while maintaining WebSocket responsiveness through non-blocking async task management. Implements intelligent task naming via LLM micro-calls, continuation detection, and orchestration routing with concurrent background processing.

## Purpose

Solves the problem of processing long-running AI agent operations without blocking the WebSocket read-loop, enabling real-time message cancellation and concurrent task naming. Bridges user input from the web interface to the agent execution layer while maintaining session state, usage tracking, and next-steps extraction.

## Most Important Functions/Classes

1. **`handle_chat(server, ws, content, image_path, file_path)`**
   - Main entry point for processing user messages; validates agent availability, prepends file/image context, marks server as busy, broadcasts user message, spawns concurrent naming and agent tasks, maintains prompt history for continuation detection. Ensures WebSocket loop remains responsive by delegating heavy work to background tasks.

2. **`_run_agent(server, content, naming_task)`**
   - Background coroutine executing the actual agent logic; awaits task naming completion, routes `/orchestrate` commands to orchestrator or calls `agent.complete()`, broadcasts assistant response with model details, extracts and broadcasts next steps, updates usage/session info, triggers auto-reflection, and handles cleanup in finally block.

3. **`_generate_task_name(user_text, recent_prompts, model, api_key, base_url, extra_headers)`**
   - Micro LLM call (via litellm) to generate concise 5-6 word task names; detects continuations and includes recent prompt history for context, implements timeout and error handling, truncates verbose responses, stores result in `server.agent._current_task_name` for usage tracking.

4. **`_is_continuation(text)`**
   - Regex-based pattern matcher identifying user affirmations/continuation signals (e.g., "continue", "yes", "sounds good"); filters by length (<60 chars) to avoid false positives, used for prompt history management and task naming context.

5. **`_MAX_RECENT_PROMPTS` & `_CONTINUATION_RE`**
   - Module-level constants: regex pattern for continuation detection and history window size (3 prompts); enables context-aware naming of follow-up messages without cluttering history with affirmations.

## Architecture & Dependencies

- **Async-first design**: Uses `asyncio.create_task()` for fire-and-forget execution, `asyncio.wait_for()` for timeouts, enabling non-blocking WebSocket operations
- **External dependencies**: `aiohttp` (WebSocket), `litellm` (LLM calls), internal modules (`captain_claw.logging`, `captain_claw.next_steps`, `captain_claw.reflections`, `captain_claw.web_server`)
- **State management**: Maintains `server._busy`, `server._active_task`, `server._recent_prompts` on WebServer instance; uses `server._broadcast()` for multi-client updates
- **Integration points**: Calls `server.agent.complete()` for main inference, `server._orchestrator.orchestrate()` for special commands, `extract_next_steps()` for UI suggestions, `maybe_auto_reflect()` for post-turn analysis
- **Error handling**: Graceful degradation (naming failures logged but non-fatal), exception catching in finally blocks, timeout protection on naming calls