"""Unified async wrapper around chat platform bridges (Telegram/Slack/Discord).

Replaces the tripled per-platform helper functions that were nested inside
``run_interactive()`` with a single parameterized implementation.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from captain_claw.config import get_config
from captain_claw.cron import now_utc, to_utc_iso
from captain_claw.logging import log

if True:  # TYPE_CHECKING style import to avoid circular at runtime
    from captain_claw.runtime_context import RuntimeContext

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Text helpers (shared across platforms)
# ---------------------------------------------------------------------------

def truncate_chat_text(text: str, max_chars: int = 220) -> str:
    compact = str(text or "").strip().replace("\n", " ")
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def chat_user_display(record: dict[str, object]) -> str:
    user_id = str(record.get("user_id", "")).strip()
    username = str(record.get("username", "")).strip()
    first_name = str(record.get("first_name", "")).strip()
    if username:
        return f"@{username} ({user_id})"
    if first_name:
        return f"{first_name} ({user_id})"
    return user_id or "unknown"


def remote_user_requested_audio(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    audio_hints = (
        "tts", "text to speech", "text-to-speech", "voice", "audio",
        "voice note", "read aloud", "mp3", "audio overview", "spoken overview",
    )
    return any(hint in lowered for hint in audio_hints)


def extract_audio_paths_from_tool_output(content: str) -> list[Path]:
    """Extract resolved MP3 paths from pocket_tts tool output text."""
    text = str(content or "")
    paths: list[Path] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("path:"):
            continue
        raw_path = stripped.split(":", 1)[1].strip()
        if " (requested:" in raw_path:
            raw_path = raw_path.split(" (requested:", 1)[0].strip()
        if not raw_path:
            continue
        try:
            resolved = Path(raw_path).expanduser().resolve()
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        if resolved.suffix.lower() != ".mp3":
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        paths.append(resolved)
    return paths


def collect_turn_generated_audio_paths(session: Any, turn_start_idx: int) -> list[Path]:
    """Collect pocket_tts generated MP3 files from current turn tool outputs."""
    if not session:
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for msg in session.messages[max(0, int(turn_start_idx)):]:
        if str(msg.get("role", "")).strip().lower() != "tool":
            continue
        if str(msg.get("tool_name", "")).strip().lower() != "pocket_tts":
            continue
        if str(msg.get("content", "")).strip().lower().startswith("error:"):
            continue
        for path in extract_audio_paths_from_tool_output(str(msg.get("content", ""))):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def cleanup_expired_pairings(pending_map: dict[str, dict[str, object]]) -> None:
    if not pending_map:
        return
    now_ts = now_utc().timestamp()
    expired = []
    for token, payload in pending_map.items():
        expires_at_raw = str(payload.get("expires_at", "")).strip()
        if not expires_at_raw:
            continue
        try:
            expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00"))
            if expires_at.timestamp() <= now_ts:
                expired.append(token)
        except Exception:
            expired.append(token)
    for token in expired:
        pending_map.pop(token, None)


def generate_pairing_token(pending_map: dict[str, dict[str, object]]) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        token = "".join(secrets.choice(alphabet) for _ in range(8))
        if token not in pending_map:
            return token


# ---------------------------------------------------------------------------
# PlatformAdapter
# ---------------------------------------------------------------------------

class PlatformAdapter:
    """Unified operations for a single chat platform.

    Usage::

        adapter = PlatformAdapter(ctx, "telegram")
        await adapter.send(chat_id, "Hello!")
        result = await adapter.run_with_typing(chat_id, some_coroutine)
    """

    def __init__(self, ctx: RuntimeContext, platform: str) -> None:
        self.ctx = ctx
        self.platform = platform
        self.state = ctx.get_platform_state(platform)

    @property
    def bridge(self) -> Any:
        return self.state.bridge

    # -- Send text message ------------------------------------------------

    async def send(
        self,
        channel_id: Any,
        text: str,
        *,
        reply_to: Any = None,
    ) -> None:
        if not self.bridge:
            return
        try:
            if self.platform == "telegram":
                await self.bridge.send_message(
                    chat_id=int(channel_id),
                    text=text,
                    reply_to_message_id=reply_to,
                )
            elif self.platform == "slack":
                await self.bridge.send_message(
                    channel_id=str(channel_id),
                    text=text,
                    reply_to_message_ts=str(reply_to or ""),
                )
            elif self.platform == "discord":
                await self.bridge.send_message(
                    channel_id=str(channel_id),
                    text=text,
                    reply_to_message_id=str(reply_to or ""),
                )
            await self.monitor_event(
                "outgoing_message",
                **self._channel_kwargs(channel_id),
                **self._reply_kwargs(reply_to),
                chars=len(str(text or "")),
                text_preview=truncate_chat_text(text),
            )
        except Exception as e:
            self.ctx.ui.append_system_line(f"{self.platform.title()} send failed: {e}")

    # -- Chat action (typing indicator) -----------------------------------

    async def send_chat_action(self, channel_id: Any, action: str = "typing") -> None:
        if not self.bridge:
            return
        try:
            if self.platform == "telegram":
                await self.bridge.send_chat_action(chat_id=int(channel_id), action=action)
            else:
                await self.bridge.send_chat_action(channel_id=str(channel_id), action=action)
        except Exception as e:
            self.ctx.ui.append_system_line(f"{self.platform.title()} chat action failed: {e}")

    # -- Mark read --------------------------------------------------------

    async def mark_read(self, message: Any) -> None:
        if not self.bridge:
            return
        try:
            if self.platform == "telegram":
                connection_id = str(getattr(message, "business_connection_id", "") or "").strip()
                if not connection_id:
                    return
                if int(message.message_id) <= 0:
                    return
                await self.bridge.read_business_message(
                    business_connection_id=connection_id,
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                )
            elif self.platform == "slack":
                await self.bridge.mark_read(
                    channel_id=message.channel_id,
                    message_ts=message.message_ts,
                )
            elif self.platform == "discord":
                await self.bridge.mark_read(
                    channel_id=message.channel_id,
                    message_id=message.id,
                )
        except Exception as e:
            self.ctx.ui.append_system_line(f"{self.platform.title()} mark-read failed: {e}")

    # -- Monitor event (unified logging) ----------------------------------

    async def monitor_event(self, step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        body_lines = [f"step={step}"]
        for key, value in args.items():
            body_lines.append(f"{key}={value}")
        body_text = "\n".join(body_lines)
        self.ctx.ui.append_tool_output(self.platform, payload, body_text)
        if self.ctx.agent.session:
            self.ctx.agent.session.add_message(
                role="tool",
                content=body_text,
                tool_name=self.platform,
                tool_arguments=payload,
                token_count=self.ctx.agent._count_tokens(body_text),
            )
            await self.ctx.agent.session_manager.save_session(self.ctx.agent.session)

    # -- Typing heartbeat -------------------------------------------------

    async def run_with_typing(
        self,
        channel_id: Any,
        work: Awaitable[T],
        *,
        heartbeat_seconds: float = 4.0,
    ) -> T:
        if not self.bridge:
            return await work

        stop = asyncio.Event()

        async def _heartbeat() -> None:
            interval = max(1.0, float(heartbeat_seconds))
            while not stop.is_set():
                await self.send_chat_action(channel_id)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        task = asyncio.create_task(_heartbeat())
        try:
            return await work
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # -- Send audio file --------------------------------------------------

    async def send_audio_file(
        self,
        channel_id: Any,
        path: Path,
        *,
        caption: str = "",
        reply_to: Any = None,
    ) -> bool:
        if not self.bridge:
            return False
        try:
            if self.platform == "telegram":
                await self.send_chat_action(channel_id, action="upload_document")
                await self.bridge.send_audio_file(
                    chat_id=int(channel_id),
                    file_path=path,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            elif self.platform == "slack":
                await self.bridge.send_audio_file(
                    channel_id=str(channel_id),
                    file_path=path,
                    caption=caption,
                )
            elif self.platform == "discord":
                await self.bridge.send_audio_file(
                    channel_id=str(channel_id),
                    file_path=path,
                    caption=caption,
                    reply_to_message_id=str(reply_to or ""),
                )
            size_bytes = 0
            try:
                size_bytes = int(path.stat().st_size)
            except Exception:
                pass
            await self.monitor_event(
                "outgoing_audio",
                **self._channel_kwargs(channel_id),
                **self._reply_kwargs(reply_to),
                path=str(path),
                size_bytes=size_bytes,
            )
            return True
        except Exception as e:
            await self.monitor_event(
                "outgoing_audio_error",
                **self._channel_kwargs(channel_id),
                **self._reply_kwargs(reply_to),
                path=str(path),
                error=str(e),
            )
            self.ctx.ui.append_system_line(f"{self.platform.title()} audio send failed: {e}")
            return False

    # -- Auto TTS for turn ------------------------------------------------

    async def maybe_send_audio_for_turn(
        self,
        channel_id: Any,
        reply_to: Any,
        user_prompt: str,
        assistant_text: str,
        turn_start_idx: int,
    ) -> None:
        generated_paths = collect_turn_generated_audio_paths(
            self.ctx.agent.session, turn_start_idx,
        )
        if generated_paths:
            for path in generated_paths:
                await self.send_audio_file(
                    channel_id, path, caption="Requested audio output", reply_to=reply_to,
                )
            return

        if not remote_user_requested_audio(user_prompt):
            return
        if "pocket_tts" not in self.ctx.agent.tools.list_tools():
            return

        source_text = str(assistant_text or "").strip()
        if not source_text and self.ctx.agent.session:
            for msg in reversed(self.ctx.agent.session.messages):
                if str(msg.get("role", "")).strip().lower() != "assistant":
                    continue
                candidate = str(msg.get("content", "")).strip()
                if candidate:
                    source_text = candidate
                    break
        if not source_text:
            return

        cfg = get_config()
        max_chars = max(1, int(getattr(cfg.tools.pocket_tts, "max_chars", 4000)))
        tts_text = source_text[:max_chars]
        tool_args: dict[str, object] = {"text": tts_text}
        try:
            result = await self.ctx.agent._execute_tool_with_guard(
                name="pocket_tts",
                arguments=tool_args,
                interaction_label=f"{self.platform}_auto_tts",
            )
            tool_output = result.content if result.success else f"Error: {result.error}"
            self.ctx.agent._add_session_message(
                role="tool",
                content=tool_output,
                tool_name="pocket_tts",
                tool_arguments=tool_args,
            )
            self.ctx.agent._emit_tool_output("pocket_tts", tool_args, tool_output)
            if self.ctx.agent.session:
                await self.ctx.agent.session_manager.save_session(self.ctx.agent.session)

            if not result.success:
                await self.send(
                    channel_id,
                    f"Audio generation failed: {result.error or 'unknown error'}",
                    reply_to=reply_to,
                )
                return

            for path in extract_audio_paths_from_tool_output(tool_output):
                await self.send_audio_file(
                    channel_id, path, caption="Audio summary", reply_to=reply_to,
                )
        except Exception as e:
            await self.monitor_event(
                "auto_tts_error",
                **self._channel_kwargs(channel_id),
                **self._reply_kwargs(reply_to),
                error=str(e),
            )
            self.ctx.ui.append_system_line(f"{self.platform.title()} auto TTS failed: {e}")

    # -- User pairing -----------------------------------------------------

    async def pair_unknown_user(self, message: Any) -> None:
        """Issue a pairing token to an unknown user."""
        cleanup_expired_pairings(self.state.pending_pairings)
        user_id_key = str(self._message_user_id(message))
        if user_id_key in self.state.approved_users:
            return

        existing_token = ""
        for token, payload in self.state.pending_pairings.items():
            if str(payload.get("user_id", "")).strip() == user_id_key:
                existing_token = token
                break

        if not existing_token:
            existing_token = generate_pairing_token(self.state.pending_pairings)
            cfg = self.state.config
            ttl_minutes = max(1, int(cfg.pairing_ttl_minutes)) if cfg else 30
            expires = datetime.fromtimestamp(
                now_utc().timestamp() + ttl_minutes * 60,
                tz=now_utc().tzinfo,
            )
            self.state.pending_pairings[existing_token] = {
                "user_id": getattr(message, "user_id", ""),
                "chat_id": self._message_channel_id(message),
                "channel_id": self._message_channel_id(message),
                "username": getattr(message, "username", ""),
                "first_name": getattr(message, "first_name", ""),
                "created_at": to_utc_iso(now_utc()),
                "expires_at": expires.isoformat(),
            }
            await self._save_state()

        channel_id = self._message_channel_id(message)
        reply_to = self._message_reply_id(message)
        await self.send(
            channel_id,
            (
                "Pairing required.\n"
                f"Your pairing token: `{existing_token}`\n\n"
                "Ask the Captain Claw operator to approve you with:\n"
                f"/approve user {self.platform} {existing_token}"
            ),
            reply_to=reply_to,
        )

    async def approve_pairing_token(self, raw_token: str) -> tuple[bool, str]:
        """Approve a pending pairing token. Returns (success, message)."""
        token = str(raw_token or "").strip().upper()
        if not token:
            return False, f"Usage: /approve user {self.platform} <token>"

        cleanup_expired_pairings(self.state.pending_pairings)
        record = self.state.pending_pairings.get(token)
        if not isinstance(record, dict):
            return False, f"{self.platform.title()} pairing token not found or expired: {token}"

        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            self.state.pending_pairings.pop(token, None)
            await self._save_state()
            return False, f"{self.platform.title()} pairing token invalid: {token}"

        self.state.approved_users[user_id] = {
            "user_id": record.get("user_id", ""),
            "chat_id": record.get("chat_id", record.get("channel_id", "")),
            "channel_id": record.get("channel_id", record.get("chat_id", "")),
            "username": str(record.get("username", "")).strip(),
            "first_name": str(record.get("first_name", "")).strip(),
            "approved_at": to_utc_iso(now_utc()),
            "token": token,
        }
        self.state.pending_pairings.pop(token, None)
        await self._save_state()

        channel_id = self.state.approved_users[user_id].get(
            "chat_id", self.state.approved_users[user_id].get("channel_id", ""),
        )
        if channel_id and self.bridge:
            await self.send(
                channel_id,
                (
                    "Pairing approved. You can now use Captain Claw.\n"
                    "All chat and supported slash commands are available."
                ),
            )
        return True, f"Approved {self.platform.title()} user: {chat_user_display(self.state.approved_users[user_id])}"

    # -- State persistence ------------------------------------------------

    async def _save_state(self) -> None:
        import json
        await self.ctx.agent.session_manager.set_app_state(
            self.state.state_key_approved,
            json.dumps(self.state.approved_users, ensure_ascii=True, sort_keys=True),
        )
        await self.ctx.agent.session_manager.set_app_state(
            self.state.state_key_pending,
            json.dumps(self.state.pending_pairings, ensure_ascii=True, sort_keys=True),
        )

    async def load_state(self) -> None:
        """Load persisted approved/pending user state from database."""
        import json
        for key, target_attr in [
            (self.state.state_key_approved, "approved_users"),
            (self.state.state_key_pending, "pending_pairings"),
        ]:
            raw = await self.ctx.agent.session_manager.get_app_state(key)
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, dict):
                setattr(self.state, target_attr, parsed)

    # -- Platform-specific field accessors --------------------------------

    def _message_user_id(self, message: Any) -> Any:
        return getattr(message, "user_id", "")

    def _message_channel_id(self, message: Any) -> Any:
        if self.platform == "telegram":
            return getattr(message, "chat_id", 0)
        return getattr(message, "channel_id", "")

    def _message_reply_id(self, message: Any) -> Any:
        if self.platform == "telegram":
            return getattr(message, "message_id", None)
        if self.platform == "slack":
            return getattr(message, "message_ts", "")
        if self.platform == "discord":
            return getattr(message, "id", "")
        return None

    def _channel_kwargs(self, channel_id: Any) -> dict[str, Any]:
        if self.platform == "telegram":
            return {"chat_id": channel_id}
        return {"channel_id": channel_id}

    def _reply_kwargs(self, reply_to: Any) -> dict[str, Any]:
        if not reply_to:
            return {}
        if self.platform == "telegram":
            return {"reply_to_message_id": reply_to}
        if self.platform == "slack":
            return {"reply_to_message_ts": reply_to}
        return {"reply_to_message_id": reply_to}


async def approve_chat_pairing_token(
    ctx: RuntimeContext, platform: str, raw_token: str,
) -> tuple[bool, str]:
    """Approve a pairing token for any platform."""
    target = str(platform or "").strip().lower()
    if target not in ("telegram", "slack", "discord"):
        return False, "Usage: /approve user <telegram|slack|discord> <token>"
    adapter = PlatformAdapter(ctx, target)
    return await adapter.approve_pairing_token(raw_token)
