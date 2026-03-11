"""Background global hotkey listener for screen capture + voice commands.

Lifecycle follows the Telegram daemon pattern:
    ``start_hotkey_daemon(server)`` / ``stop_hotkey_daemon(server)``
called from ``web_server._run_server()`` startup / shutdown.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from structlog import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from pynput import keyboard as _keyboard

    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


# ---------------------------------------------------------------------------
# Platform support check
# ---------------------------------------------------------------------------


def _check_platform_support() -> tuple[bool, str]:
    """Return (supported, reason) for the current platform."""
    if sys.platform == "linux":
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            return False, (
                "Global hotkey listening is not supported on Wayland. "
                "Use the /screenshot command or switch to X11."
            )
    return True, ""


# ---------------------------------------------------------------------------
# Clipboard-based selected text grab
# ---------------------------------------------------------------------------


def _grab_selected_text() -> str:
    """Try to capture selected text in the active app using the clipboard.

    Strategy (macOS-centric, but portable):
      1. Save the current clipboard text.
      2. Simulate Cmd+C (macOS) / Ctrl+C (Linux/Windows) to copy the
         selection into the clipboard.
      3. Short sleep to let the paste-board update.
      4. Read the clipboard — if it differs from step 1, that's the
         selected text.
      5. Restore the original clipboard contents.

    Returns the selected text, or "" if nothing was selected.

    This function is synchronous and should be called via
    ``asyncio.to_thread()``.
    """
    import subprocess as _sp

    if sys.platform != "darwin":
        # TODO: xclip / xsel + xdotool for Linux, win32clipboard for Windows
        return ""

    # 1. Save current clipboard.
    original = ""
    try:
        original = _sp.run(
            ["pbpaste"], capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        pass

    # 2. Clear the clipboard so we can detect if Cmd+C actually copied something.
    try:
        _sp.run(
            ["pbcopy"], input="", text=True, timeout=2,
        )
    except Exception:
        pass

    # 3. Simulate Cmd+C via AppleScript (works across all apps).
    try:
        _sp.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke "c" using command down',
            ],
            capture_output=True,
            timeout=3,
        )
    except Exception:
        # Restore clipboard on failure.
        try:
            _sp.run(["pbcopy"], input=original, text=True, timeout=2)
        except Exception:
            pass
        return ""

    # 4. Small delay for the pasteboard to update.
    time.sleep(0.15)

    # 5. Read what Cmd+C put into the clipboard.
    copied = ""
    try:
        copied = _sp.run(
            ["pbpaste"], capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        pass

    # 6. Restore the original clipboard.
    try:
        _sp.run(["pbcopy"], input=original, text=True, timeout=2)
    except Exception:
        pass

    # If the clipboard is still empty (or unchanged), nothing was selected.
    selected = copied.strip()
    if not selected or selected == original.strip():
        return ""

    log.debug("grabbed_selected_text", length=len(selected))
    return selected


# ---------------------------------------------------------------------------
# Tap-count state machine (double-tap / triple-tap + hold)
# ---------------------------------------------------------------------------


class HotkeyState:
    """Tracks multi-tap detection and hold-to-record state.

    * **Double-tap + hold** → ``"voice"`` mode (voice-only, no screenshot).
    * **Triple-tap + hold** → ``"screen"`` mode (screenshot / selected text + voice).

    After the 2nd tap the state machine waits ``triple_tap_wait_ms`` for
    a potential 3rd tap.  If none arrives and the key is still held, it
    activates in ``"voice"`` mode.  A 3rd tap within the window
    immediately activates ``"screen"`` mode.
    """

    def __init__(
        self,
        trigger_key: str = "ctrl",
        double_tap_ms: int = 400,
        triple_tap_wait_ms: int = 300,
    ) -> None:
        self.trigger_key = trigger_key.lower()
        self.double_tap_threshold = double_tap_ms / 1000.0
        self.triple_tap_wait = triple_tap_wait_ms / 1000.0

        self._lock = threading.Lock()
        self._last_press_time: float = 0.0
        self._tap_count: int = 0
        self._activated: bool = False
        self._held: bool = False
        self._key_is_pressed: bool = False
        self._pending_timer: threading.Timer | None = None
        self._activation_mode: str = ""  # "voice" or "screen"

        # Callbacks set by the daemon after construction.
        self.on_activate: Any = None  # (mode: str) -> None
        self.on_release: Any = None   # () -> None

    # -- helpers --

    def _key_matches(self, key: Any) -> bool:
        """Return *True* if *key* matches the configured trigger."""
        key_name = ""
        if hasattr(key, "name"):
            key_name = str(key.name).lower()
        return self.trigger_key in key_name or key_name.startswith(self.trigger_key)

    def _cancel_pending(self) -> None:
        """Cancel any pending triple-tap wait timer.  Caller must hold ``_lock``."""
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None

    def _do_activate(self, mode: str) -> None:
        """Fire activation.  Caller must hold ``_lock``."""
        if self._activated:
            return
        self._activated = True
        self._activation_mode = mode
        self._held = self._key_is_pressed
        self._tap_count = 0
        if self.on_activate:
            self.on_activate(mode)

    def _timer_fired(self) -> None:
        """Called by the background Timer when triple-tap wait expires."""
        with self._lock:
            self._pending_timer = None
            if not self._activated and self._key_is_pressed:
                self._do_activate("voice")

    # -- pynput callbacks (run in the listener thread) --

    def on_press(self, key: Any) -> None:
        with self._lock:
            if not self._key_matches(key):
                # Non-trigger key resets tap counter.
                self._tap_count = 0
                self._cancel_pending()
                return

            # Ignore auto-repeat while the key is physically held.
            if self._key_is_pressed:
                return

            self._key_is_pressed = True
            now = time.monotonic()

            # Already activated and key re-pressed → mark as held.
            if self._activated and not self._held:
                self._held = True
                return

            # Tap counting.
            if now - self._last_press_time <= self.double_tap_threshold:
                self._tap_count += 1
            else:
                self._tap_count = 1
                self._cancel_pending()
            self._last_press_time = now

            if self._tap_count >= 3 and not self._activated:
                # Triple-tap → immediate screen mode activation.
                self._cancel_pending()
                self._do_activate("screen")
            elif self._tap_count == 2 and not self._activated:
                # Double-tap → start timer; voice mode unless 3rd tap arrives.
                self._cancel_pending()
                self._pending_timer = threading.Timer(
                    self.triple_tap_wait, self._timer_fired,
                )
                self._pending_timer.daemon = True
                self._pending_timer.start()

    def on_release_key(self, key: Any) -> None:
        with self._lock:
            if not self._key_matches(key):
                return
            self._key_is_pressed = False
            if self._activated and self._held:
                self._held = False
                self._activated = False
                self._activation_mode = ""
                if self.on_release:
                    self.on_release()


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------


async def start_hotkey_daemon(server: WebServer) -> None:
    """Start the global hotkey listener (if configured and dependencies met)."""
    from captain_claw.config import get_config

    cfg = get_config()
    screen_cfg = cfg.tools.screen_capture

    if not screen_cfg.hotkey_enabled:
        return

    if not _HAS_PYNPUT:
        log.warning(
            "Hotkey listener requires 'pynput'. "
            "Install with: pip install captain-claw[screen]"
        )
        return

    # Verify screenshot capability is available.
    # macOS uses the native `screencapture` CLI (no extra deps), while
    # Linux/Windows require the `mss` package.
    try:
        from captain_claw.tools.screen_capture import _HAS_MSS, _IS_MACOS

        if not _IS_MACOS and not _HAS_MSS:
            log.warning(
                "Screen capture requires 'mss'. "
                "Install with: pip install captain-claw[screen]"
            )
            return
    except ImportError:
        return

    # Platform check (e.g. Wayland).
    supported, reason = _check_platform_support()
    if not supported:
        log.warning("hotkey_daemon_skipped", reason=reason)
        print(f"  Hotkey daemon skipped: {reason}")
        return

    loop = asyncio.get_event_loop()

    state = HotkeyState(
        trigger_key=screen_cfg.hotkey_trigger_key,
        double_tap_ms=screen_cfg.hotkey_double_tap_ms,
        triple_tap_wait_ms=screen_cfg.hotkey_triple_tap_wait_ms,
    )

    # Bridge between pynput thread and asyncio: a threading.Event signals
    # the audio recording thread to stop when the key is released.
    audio_stop_event = threading.Event()

    # -- callbacks (run in pynput's / timer thread) --

    def _on_activate(mode: str) -> None:
        audio_stop_event.clear()
        asyncio.run_coroutine_threadsafe(
            _handle_activation(server, audio_stop_event, mode=mode),
            loop,
        )

    def _on_release() -> None:
        audio_stop_event.set()

    state.on_activate = _on_activate
    state.on_release = _on_release

    # Start pynput listener as a daemon thread.
    try:
        listener = _keyboard.Listener(
            on_press=state.on_press,
            on_release=state.on_release_key,
        )
        listener.daemon = True
        listener.start()
    except Exception as exc:
        # macOS: missing Input Monitoring permission.
        log.warning(
            "hotkey_listener_failed",
            error=str(exc),
            hint=(
                "On macOS, grant Input Monitoring permission for Python "
                "in System Settings > Privacy & Security."
            ),
        )
        print(f"  Hotkey listener failed to start: {exc}")
        return

    server._hotkey_listener = listener
    server._hotkey_state = state

    trigger = screen_cfg.hotkey_trigger_key
    log.info(
        "hotkey_daemon_started",
        trigger=trigger,
        double_tap_ms=screen_cfg.hotkey_double_tap_ms,
        triple_tap_wait_ms=screen_cfg.hotkey_triple_tap_wait_ms,
    )
    print(f"  Hotkey active: 2×{trigger} = voice · 3×{trigger} = screen+voice")


async def stop_hotkey_daemon(server: WebServer) -> None:
    """Stop the hotkey listener gracefully."""
    listener = getattr(server, "_hotkey_listener", None)
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass
        server._hotkey_listener = None
        server._hotkey_state = None


# ---------------------------------------------------------------------------
# Activation handler (runs on the asyncio event loop)
# ---------------------------------------------------------------------------

# Guard against overlapping activations.
_activation_lock = asyncio.Lock()


async def _handle_activation(
    server: WebServer,
    audio_stop_event: threading.Event,
    *,
    mode: str = "screen",
) -> None:
    """Orchestrate hotkey activation.

    Modes:
      * ``"voice"``  — double-tap: record voice → transcribe → submit text only.
      * ``"screen"`` — triple-tap: record voice + capture screenshot / selected text → submit.
    """

    if _activation_lock.locked():
        return  # already processing a capture

    async with _activation_lock:
        await _do_activation(server, audio_stop_event, mode=mode)


async def _do_activation(
    server: WebServer,
    audio_stop_event: threading.Event,
    *,
    mode: str = "screen",
) -> None:
    from captain_claw.config import get_config

    if not server.agent:
        return

    cfg = get_config()
    screen_cfg = cfg.tools.screen_capture

    session = getattr(server.agent, "session", None)
    session_id = session.id if session else "hotkey"
    workspace = cfg.resolved_workspace_path()
    saved_root = workspace / "saved"

    # ── 1. Capture voice input (blocks until key is released) ──
    transcription = ""
    wav_bytes: bytes | None = None

    # Prefer Soniox realtime: streams mic → Soniox WebSocket → text.
    # No audio files created or uploaded.
    use_realtime = False
    soniox_key = ""
    try:
        from captain_claw.tools.stt import _HAS_AUDIO, _HAS_SONIOX, _resolve_stt_api_key

        soniox_key = _resolve_stt_api_key("soniox")
        use_realtime = _HAS_AUDIO and _HAS_SONIOX and bool(soniox_key)
    except ImportError:
        pass

    if use_realtime:
        # ── Soniox realtime path (no audio files) ──
        server._broadcast({"type": "status", "status": "🎙️ listening... (release key to stop)"})
        try:
            from captain_claw.tools.stt import realtime_stt_sync

            transcription = await asyncio.to_thread(
                realtime_stt_sync,
                soniox_key,
                audio_stop_event,
                sample_rate=screen_cfg.audio_sample_rate,
                max_duration=screen_cfg.max_recording_seconds,
            )
        except Exception as exc:
            log.warning("hotkey_realtime_stt_failed", error=str(exc))
    else:
        # ── Fallback: record WAV, transcribe later with OpenAI/Gemini ──
        has_audio = False
        try:
            from captain_claw.tools.stt import _HAS_AUDIO, _record_audio_sync

            has_audio = _HAS_AUDIO
        except ImportError:
            pass

        if has_audio:
            server._broadcast({"type": "status", "status": "🎙️ recording... (release key to stop)"})
            try:
                wav_bytes = await asyncio.to_thread(
                    _record_audio_sync,
                    duration_limit=screen_cfg.max_recording_seconds,
                    sample_rate=screen_cfg.audio_sample_rate,
                    channels=1,
                    stop_event=audio_stop_event,
                )
            except Exception as exc:
                log.warning("hotkey_audio_failed", error=str(exc))
                wav_bytes = None
        else:
            # No audio deps — wait for key release before capturing screenshot.
            log.info("Audio dependencies not available, waiting for key release.")
            server._broadcast({"type": "status", "status": "hold key... (release to capture)"})
            await asyncio.to_thread(audio_stop_event.wait, screen_cfg.max_recording_seconds)

    # ── 2. Key was released — grab context (screen mode only) ──
    selected_text = ""
    screenshot_path: Path | None = None

    if mode == "screen":
        server._broadcast({"type": "status", "status": "capturing..."})

        # Try to grab any selected text from the focused app via a clipboard
        # swap: save current clipboard → Cmd+C → read new clipboard → restore.
        selected_text = await asyncio.to_thread(_grab_selected_text)

        # If we got selected text, skip the screenshot — text is the context.
        if not selected_text:
            from captain_claw.tools.screen_capture import capture_and_save

            try:
                screenshot_path = await capture_and_save(
                    session_id=session_id,
                    saved_root=saved_root,
                    monitor_index=screen_cfg.default_monitor,
                    label="hotkey-capture",
                )
            except Exception as exc:
                log.error("hotkey_screenshot_failed", error=str(exc))
                server._broadcast({"type": "error", "message": f"Screenshot failed: {exc}"})
                return

    # ── 3. Transcribe audio (WAV fallback path only) ──
    if not use_realtime and wav_bytes:
        # Optionally save the WAV.
        if screen_cfg.save_audio:
            try:
                from captain_claw.tools.write import WriteTool

                sid = WriteTool._normalize_session_id(session_id)
                stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                audio_dest = saved_root / "media" / sid / f"voice-{stamp}.wav"
                audio_dest.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(audio_dest.write_bytes, wav_bytes)
            except Exception:
                pass

        server._broadcast({"type": "status", "status": "transcribing..."})
        try:
            from captain_claw.tools.stt import transcribe_audio

            transcription = await transcribe_audio(wav_bytes)
        except Exception as exc:
            log.warning("hotkey_transcription_failed", error=str(exc))

    # ── 4. Submit to agent ──
    voice_text = transcription.strip()
    image_path: str | None = str(screenshot_path) if screenshot_path else None

    if mode == "voice":
        # ── Voice-only mode (double-tap) ──
        if not voice_text:
            server._broadcast({"type": "status", "status": "No voice detected."})
            return
        user_content = (
            f"{voice_text}\n\n"
            f"IMPORTANT: After completing the task, use the pocket_tts tool "
            f"to speak your answer aloud so I can hear it. Keep the spoken "
            f"response brief and conversational."
        )
    elif selected_text:
        # ── Selected text mode (triple-tap, no screenshot) ──
        if voice_text:
            user_content = (
                f"Here is some text I selected on my screen:\n\n"
                f"---\n{selected_text}\n---\n\n"
                f"Here is what I need you to do:\n\n{voice_text}\n\n"
                f"IMPORTANT: After completing the task, use the pocket_tts tool "
                f"to speak your answer aloud so I can hear it. Keep the spoken "
                f"response brief and conversational."
            )
        else:
            user_content = (
                f"Here is some text I selected on my screen:\n\n"
                f"---\n{selected_text}\n---\n\n"
                f"Briefly analyze or summarize this text. "
                f"Suggest 2-3 useful actions I could take with it."
            )
    else:
        # ── Screenshot mode (triple-tap, no selected text) ──
        if voice_text:
            user_content = (
                f"I'm looking at my screen (screenshot attached). "
                f"Here is what I need you to do:\n\n{voice_text}\n\n"
                f"IMPORTANT: After completing the task, use the pocket_tts tool "
                f"to speak your answer aloud so I can hear it. Keep the spoken "
                f"response brief and conversational."
            )
        else:
            from captain_claw.instructions import InstructionLoader

            try:
                loader = InstructionLoader()
                user_content = loader.load("screenshot_analysis_prompt.md")
            except Exception:
                user_content = (
                    "Use image_vision on the attached screenshot. "
                    "Briefly describe what is on screen. "
                    "List 2-3 suggestions for what I could do next. "
                    "Do NOT run any other tools."
                )

    # Find a connected WebSocket to use for the chat handler.
    ws = next(iter(server.clients), None)
    if ws is None or ws.closed:
        log.warning("hotkey_no_ws_client", hint="No WebSocket client connected.")
        return

    from captain_claw.web.chat_handler import handle_chat

    await handle_chat(server, ws, user_content, image_path=image_path)
