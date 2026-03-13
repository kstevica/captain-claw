# Captain Claw v0.3.5 Release Notes

**Release title:** Self-Reflection System, Desktop Automation, Chat UX Improvements

**Release date:** 2026-03-13

## Highlights

This release introduces a self-reflection system that enables the agent to periodically assess its own performance and inject improvement directives into the system prompt. A new desktop automation tool (`desktop_action`) adds cross-platform GUI control via `pyautogui`, pairing with the existing screen capture tool for coordinate-based interaction. The chat UI gains message feedback (like/dislike) and a copy-to-clipboard button on every message.

## New Features

### Self-Reflection System
- New `reflections.py` module following the personality.py pattern: dataclass → markdown file → mtime caching → prompt block injection
- Gathers context from recent session messages (last 20), memory facts (up to 30), completed tasks/cron since the last reflection, and the previous reflection summary
- LLM generates actionable, generalized self-improvement instructions — never references specific tasks or sessions
- Latest reflection injected into the system prompt via `{reflection_block}` placeholder (both full and micro prompts)
- Auto-trigger fires after agent turns when both cooldown (4 hours) and minimum activity (10 messages) thresholds are met
- Auto-trigger runs as fire-and-forget `asyncio.create_task()` — non-blocking, failures are non-fatal
- Reflections stored as timestamped Markdown files in `~/.captain-claw/reflections/`
- Mtime-based caching ensures the file is only re-read when modified
- Token usage logged to the LLM usage table (`interaction="reflection"`)
- Slash commands: `/reflection` (show latest), `/reflection generate` (trigger new), `/reflection list` (list recent)
- Web UI dashboard at `/reflections` — expandable cards with timestamps, topics, active badge, delete button, and generate button
- REST API: `GET /api/reflections`, `GET /api/reflections/latest`, `POST /api/reflections/generate`, `DELETE /api/reflections/{timestamp}`
- Homepage card added (🪞 Reflections)

### Desktop Automation Tool
- New `desktop_action` tool: cross-platform desktop GUI automation via `pyautogui`
- Actions: `click`, `double_click`, `right_click`, `move`, `type`, `press`, `hotkey`, `scroll`, `drag`, `open`, `mouse_position`, `screenshot_click`
- The `screenshot_click` action combines vision-based element detection with automated clicking — describe what to click and the agent finds and clicks it
- Platform-specific app/URL launchers: macOS `open -a`, Linux `xdg-open`, Windows `os.startfile`
- `pyautogui.FAILSAFE` enabled — move mouse to top-left corner to abort runaway automation
- Pairs naturally with `screen_capture` for a see-then-act workflow
- Requires `pip install pyautogui`

### Chat UI: Message Feedback
- Like/dislike buttons on every assistant message in the chat UI
- Feedback stored per-message in the session via WebSocket (`message_feedback` event)
- Visual toggle: click to set, click again to clear
- Thumbs-up (good) and thumbs-down (bad) with hover states

### Chat UI: Copy Button
- Copy-to-clipboard button on every message bubble (chat and assistant)
- One-click copy of the full message text
- Brief checkmark confirmation after copying

## Bug Fixes

### Reflection Parser: Empty Summary
- Fixed a bug where LLM-generated markdown headers inside the reflection summary were treated as section delimiters, causing the summary to appear empty despite successful token generation
- Parser now only recognizes known section headers (`Timestamp`, `Summary`, `Topics Reviewed`, `Token Usage`) as delimiters — all other `##` headers in the LLM response are preserved as content

### Reflection Quality
- Updated the reflection system prompt with explicit rules: generalize insights into reusable principles, never reference specific tasks/sessions/turns, output flat bullet lists only (no markdown headers)

## Configuration Changes

### New Tool
- `desktop_action` added to the default enabled tools list in config

### New System Prompt Variables
- `{reflection_block}` — injected into both `system_prompt.md` and `micro_system_prompt.md` after `{visualization_style_block}`

## Web UI Changes

- New `/reflections` page with dark-themed card layout, expandable summaries, and active reflection indicator
- 🪞 Reflections card added to the homepage navigation grid
- Like/dislike feedback buttons on assistant messages
- Copy-to-clipboard button on all message bubbles
- Delete button on reflection cards (visible at 50% opacity, full opacity on hover)

## Internal

- New files: `captain_claw/reflections.py`, `captain_claw/web/rest_reflections.py`, `captain_claw/web/static/reflections.html`, `captain_claw/instructions/reflection_system_prompt.md`, `captain_claw/instructions/reflection_user_prompt.md`, `captain_claw/tools/desktop_action.py`
- Modified: `agent_context_mixin.py` (reflection block loading + render), `web_server.py` (routes + commands), `slash_commands.py` (`/reflection` handler), `chat_handler.py` (auto-reflection trigger), `static_pages.py` (serve reflections), `home.html` (reflections card), `system_prompt.md` and `micro_system_prompt.md` (`{reflection_block}`), `app.js` (feedback + copy buttons), `ws_handler.py` (feedback event), `style.css` (feedback + copy styling)
- PyInstaller `captain_claw.spec` updated with new hidden imports
- Version bumped from 0.3.4.1 to 0.3.5
