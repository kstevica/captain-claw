# Summary: hotkey_daemon.py

# hotkey_daemon.py Summary

**Summary:** A background daemon that listens for global hotkey events (double/triple-tap of a configurable trigger key) to initiate voice recording and optional screenshot capture. Implements a sophisticated tap-count state machine that distinguishes between voice-only mode (double-tap) and screen+voice mode (triple-tap), with seamless integration into the asyncio event loop via threading bridges.

**Purpose:** Solves the problem of hands-free, context-aware command input by allowing users to trigger voice transcription and screen capture through keyboard gestures without explicit UI interaction. Handles platform-specific constraints (Wayland unsupported), clipboard-based text selection detection, and audio recording with fallback transcription paths (Soniox realtime vs. WAV-based).

---

## Most Important Functions/Classes/Procedures

### 1. **HotkeyState** (Class)
Multi-tap detection state machine that tracks keyboard press/release events and manages activation modes. Implements a timer-based logic: double-tap starts a 300ms window waiting for a third tap; if the key is held and no third tap arrives, activates "voice" mode; a third tap within the window immediately activates "screen" mode. Uses threading locks to synchronize between the pynput listener thread and asyncio callbacks. Core methods: `on_press()`, `on_release_key()`, `_do_activate()`, `_timer_fired()`.

### 2. **start_hotkey_daemon(server)** (Async Function)
Lifecycle entry point that initializes the hotkey listener. Performs dependency checks (pynput, mss for non-macOS), platform validation (rejects Wayland), and instantiates the HotkeyState machine. Starts a daemon thread running pynput's keyboard listener and bridges activation callbacks to the asyncio event loop via `asyncio.run_coroutine_threadsafe()`. Stores listener and state on the server object for later cleanup.

### 3. **_do_activation(server, audio_stop_event, mode)** (Async Function)
Main orchestration handler for hotkey activation. Implements a four-stage pipeline: (1) capture voice via Soniox realtime or WAV recording, (2) grab selected text or screenshot if in "screen" mode, (3) transcribe WAV audio if fallback path used, (4) construct user prompt and submit to agent via WebSocket. Handles mode-specific prompt engineering (voice-only vs. selected-text vs. screenshot analysis) and integrates pocket_tts tool calls for spoken responses.

### 4. **_grab_selected_text()** (Sync Function)
Clipboard-based text selection detector (macOS-only, with TODO for Linux/Windows). Saves current clipboard, simulates Cmd+C via AppleScript to copy selection, detects if clipboard changed, and restores original content. Returns empty string if no selection detected. Designed to run in a thread via `asyncio.to_thread()` to avoid blocking the event loop.

### 5. **_check_platform_support()** (Function)
Platform compatibility validator that returns a tuple `(supported: bool, reason: str)`. Currently rejects Wayland sessions with a user-friendly message directing users to X11 or the `/screenshot` command. Extensible for future platform constraints.

---

## Architecture & Dependencies

**Threading Model:**
- **pynput listener thread** (daemon): Runs global keyboard listener, calls `HotkeyState.on_press/on_release_key()`.
- **Timer thread** (daemon): Executes `HotkeyState._timer_fired()` after triple-tap wait expires.
- **asyncio event loop** (main): Handles async orchestration; receives activation signals via `asyncio.run_coroutine_threadsafe()`.
- **Thread pool** (via `asyncio.to_thread()`): Offloads blocking I/O (audio recording, clipboard ops, screenshot capture).

**Key Dependencies:**
- `pynput.keyboard`: Global hotkey listening (optional, warns if missing).
- `captain_claw.tools.stt`: Audio recording (`_record_audio_sync`), realtime transcription (`realtime_stt_sync`), and batch transcription (`transcribe_audio`).
- `captain_claw.tools.screen_capture`: Screenshot capture (`capture_and_save`), platform detection (`_IS_MACOS`, `_HAS_MSS`).
- `captain_claw.web.chat_handler`: Submits captured context and voice text to the agent.
- `captain_claw.instructions`: Loads default screenshot analysis prompts.
- `structlog` or `logging`: Structured logging with context.

**Configuration Source:**
- `captain_claw.config.get_config()` provides `screen_capture` settings: `hotkey_enabled`, `hotkey_trigger_key`, `hotkey_double_tap_ms`, `hotkey_triple_tap_wait_ms`, `audio_sample_rate`, `max_recording_seconds`, `save_audio`, `default_monitor`.

**Synchronization Primitives:**
- `threading.Lock()` (HotkeyState): Protects tap counter and activation state.
- `threading.Event()` (audio_stop_event): Signals audio recording thread to stop on key release.
- `asyncio.Lock()` (_activation_lock): Prevents overlapping activation handlers.
- `threading.Timer`: Implements triple-tap wait timeout.

**Integration Points:**
- **Server lifecycle:** Called from `web_server._run_server()` startup/shutdown via `start_hotkey_daemon()` / `stop_hotkey_daemon()`.
- **WebSocket broadcast:** Sends status updates ("listening...", "recording...", "capturing...") to connected clients.
- **Agent submission:** Routes captured context (voice + screenshot/text) to agent via `handle_chat()` with optional image attachment.

**Error Handling:**
- Gracefully degrades if dependencies missing (logs warnings, returns early).
- Catches macOS Input Monitoring permission errors with actionable hints.
- Guards against overlapping activations with async lock.
- Fallback transcription path if Soniox unavailable.
- Handles platform-specific failures (e.g., clipboard ops, screenshot capture).