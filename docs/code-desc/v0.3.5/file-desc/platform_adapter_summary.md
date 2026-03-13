# Summary: platform_adapter.py

# platform_adapter.py Summary

**Summary:** Unified async wrapper providing a single parameterized interface for multi-platform chat operations (Telegram, Slack, Discord). Consolidates previously duplicated per-platform helper functions and bridges into a cohesive `PlatformAdapter` class that handles message sending, media delivery, user pairing workflows, and event monitoring across all supported platforms.

**Purpose:** Eliminates code duplication from tripled per-platform implementations nested in `run_interactive()` by providing platform-agnostic methods that internally dispatch to the appropriate bridge implementation. Manages user authentication via pairing tokens, handles media file extraction from tool outputs, and provides unified logging/monitoring across platforms.

---

## Most Important Functions/Classes

### 1. **PlatformAdapter (class)**
Core abstraction layer providing unified async operations for a single chat platform. Maintains platform state (approved users, pending pairings), bridges to platform-specific implementations, and handles all outgoing communication. Key responsibility: translate platform-agnostic method calls into platform-specific bridge calls with proper error handling and retry logic.

### 2. **send() (async method)**
Sends text messages with platform-specific parameter mapping (chat_id for Telegram, channel_id for Slack/Discord). Includes intelligent retry logic: if message send fails with reply_to reference, automatically retries without the reply to handle stale message targets. Monitors outgoing events and logs to session.

### 3. **run_with_typing() (async method)**
Wraps long-running operations with continuous typing indicator heartbeat. Spawns background task sending periodic chat actions at configurable intervals (default 4s) while work coroutine executes, then cleanly cancels heartbeat on completion. Prevents UI timeout perception during processing.

### 4. **maybe_send_audio_for_turn() (async method)**
Orchestrates automatic text-to-speech delivery. Collects pre-generated audio from tool outputs; if none exist and user requested audio, invokes pocket_tts tool with truncated assistant text, extracts resulting MP3 paths, and sends as audio files. Handles tool execution, session persistence, and error reporting.

### 5. **pair_unknown_user() / approve_pairing_token() (async methods)**
Implements two-phase user authentication workflow. `pair_unknown_user()` generates 8-character alphanumeric tokens with TTL, stores in pending_pairings, sends token to user. `approve_pairing_token()` validates token, moves user to approved_users dict, persists state, and notifies user of approval. Includes expiration cleanup.

### 6. **collect_turn_generated_audio_paths() / collect_turn_generated_image_paths() (functions)**
Extract resolved file paths from session message history. Filter tool messages by role/tool_name (pocket_tts, image_gen, termux, browser), parse "Path:" prefixed lines from content, validate file existence and extension, deduplicate by resolved path. Return ordered list of valid files.

### 7. **extract_audio_paths_from_tool_output() / extract_image_paths_from_tool_output() (functions)**
Parse tool output text to extract file paths. Handle "Path: <path> (requested: ...)" format, expand user paths, validate file existence and correct extension (.mp3 for audio, {.png, .jpg, .jpeg, .webp, .gif} for images), deduplicate. Return list of resolved Path objects.

### 8. **send_audio_file() / send_image_file() (async methods)**
Dispatch media files to platform bridges with platform-specific method signatures. Telegram uses send_audio_file/send_photo with chat_id; Slack/Discord use send_audio_file/send_image_file with channel_id. Include upload chat actions (Telegram), file size tracking, and comprehensive error monitoring with fallback system messages.

### 9. **cleanup_expired_pairings() / generate_pairing_token() (functions)**
Maintenance utilities for pairing state. `cleanup_expired_pairings()` removes tokens with expires_at <= now. `generate_pairing_token()` generates cryptographically random 8-char tokens from alphanumeric alphabet, checking against existing pending_pairings to ensure uniqueness.

### 10. **load_state() / _save_state() (async methods)**
Persist approved_users and pending_pairings dicts to database via session_manager. `load_state()` deserializes JSON from app_state keys on initialization. `_save_state()` serializes current state dicts to JSON and writes back. Enables pairing state survival across restarts.

---

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.runtime_context.RuntimeContext` – provides platform state, agent, session manager, UI
- `captain_claw.config.get_config()` – retrieves tool configuration (pocket_tts max_chars)
- `captain_claw.cron` – UTC time utilities (now_utc, to_utc_iso)
- `captain_claw.logging.log` – structured logging
- Platform bridges (Telegram/Slack/Discord) – accessed via `self.state.bridge`

**Role in System:** Sits between agent/command handlers and platform-specific bridge implementations. Provides single entry point for all outgoing platform communication, handles user authentication/authorization, manages media delivery workflows, and maintains unified event monitoring across heterogeneous chat platforms. Enables agent to operate platform-agnostically while preserving platform-specific capabilities (e.g., Telegram reply_to_message_id vs Slack reply_to_message_ts).

**State Management:** Maintains two user dicts per platform (approved_users, pending_pairings) with TTL-based expiration. Persists to database for durability. Supports multi-platform user isolation (same user_id on different platforms treated separately).