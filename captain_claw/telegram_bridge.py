"""Minimal Telegram Bot API bridge (long polling + send message)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class TelegramMessage:
    """Normalized incoming Telegram message payload."""

    update_id: int
    message_id: int
    chat_id: int
    user_id: int
    username: str
    first_name: str
    text: str
    business_connection_id: str = ""


class TelegramBridge:
    """Telegram Bot API helper."""

    def __init__(self, token: str, api_base_url: str = "https://api.telegram.org"):
        self.token = token.strip()
        self.api_base_url = (api_base_url or "https://api.telegram.org").rstrip("/")
        self._client = httpx.AsyncClient(timeout=40.0)

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return f"{self.api_base_url}/bot{self.token}/{method}"

    async def close(self) -> None:
        await self._client.aclose()

    async def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[TelegramMessage]:
        """Poll Telegram updates and normalize text messages."""
        params: dict[str, Any] = {"timeout": max(1, int(timeout))}
        if offset is not None:
            params["offset"] = int(offset)
        response = await self._client.get(self._url("getUpdates"), params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return []
        results = payload.get("result")
        if not isinstance(results, list):
            return []

        messages: list[TelegramMessage] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            update_id = int(item.get("update_id", 0))
            msg = item.get("message")
            if not isinstance(msg, dict):
                msg = item.get("business_message")
            if not isinstance(msg, dict):
                continue
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            from_user = msg.get("from")
            chat = msg.get("chat")
            if not isinstance(from_user, dict) or not isinstance(chat, dict):
                continue
            user_id = int(from_user.get("id", 0))
            chat_id = int(chat.get("id", 0))
            if user_id == 0 or chat_id == 0:
                continue
            messages.append(
                TelegramMessage(
                    update_id=update_id,
                    message_id=int(msg.get("message_id", 0)),
                    chat_id=chat_id,
                    user_id=user_id,
                    username=str(from_user.get("username", "")).strip(),
                    first_name=str(from_user.get("first_name", "")).strip(),
                    text=text,
                    business_connection_id=str(msg.get("business_connection_id", "")).strip(),
                )
            )
        return messages

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send text to Telegram, splitting at 4096 chars when needed."""
        raw = str(text or "").strip()
        if not raw:
            return
        max_len = 3800
        chunks: list[str] = []
        while raw:
            if len(raw) <= max_len:
                chunks.append(raw)
                break
            split_at = raw.rfind("\n", 0, max_len)
            if split_at < 800:
                split_at = max_len
            chunks.append(raw[:split_at].rstrip())
            raw = raw[split_at:].lstrip()

        for idx, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": int(chat_id),
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if reply_to_message_id and idx == 0:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            response = await self._client.post(self._url("sendMessage"), json=payload)
            response.raise_for_status()

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Send transient chat action status (e.g., `typing`)."""
        payload: dict[str, Any] = {
            "chat_id": int(chat_id),
            "action": str(action or "typing").strip() or "typing",
        }
        response = await self._client.post(self._url("sendChatAction"), json=payload)
        response.raise_for_status()

    async def send_audio_file(
        self,
        chat_id: int,
        file_path: str | Path,
        *,
        caption: str = "",
        reply_to_message_id: int | None = None,
    ) -> None:
        """Upload audio file to Telegram using sendAudio."""
        audio_path = Path(file_path).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(f"Telegram audio file not found: {audio_path}")
        payload: dict[str, Any] = {"chat_id": int(chat_id)}
        caption_text = str(caption or "").strip()
        if caption_text:
            payload["caption"] = caption_text
        if reply_to_message_id:
            payload["reply_to_message_id"] = int(reply_to_message_id)
        with audio_path.open("rb") as handle:
            files = {
                "audio": (
                    audio_path.name,
                    handle,
                    "audio/mpeg",
                )
            }
            response = await self._client.post(
                self._url("sendAudio"),
                data=payload,
                files=files,
            )
        response.raise_for_status()

    async def read_business_message(
        self,
        business_connection_id: str,
        chat_id: int,
        message_id: int,
    ) -> None:
        """Mark incoming business message as read (Business API only)."""
        payload: dict[str, Any] = {
            "business_connection_id": str(business_connection_id or "").strip(),
            "chat_id": int(chat_id),
            "message_id": int(message_id),
        }
        if not payload["business_connection_id"]:
            return
        response = await self._client.post(self._url("readBusinessMessage"), json=payload)
        response.raise_for_status()

    async def set_my_commands(self, commands: list[tuple[str, str]]) -> None:
        """Register slash commands shown in Telegram command picker."""
        payload_commands: list[dict[str, str]] = []
        for name, description in commands:
            command = str(name or "").strip().lower().lstrip("/")
            desc = str(description or "").strip()
            if not command or not desc:
                continue
            payload_commands.append({"command": command, "description": desc})
        if not payload_commands:
            return
        payload: dict[str, Any] = {"commands": payload_commands}
        response = await self._client.post(self._url("setMyCommands"), json=payload)
        response.raise_for_status()

    async def set_chat_menu_button_commands(self) -> None:
        """Configure chat menu button to open command list."""
        payload: dict[str, Any] = {"menu_button": {"type": "commands"}}
        response = await self._client.post(self._url("setChatMenuButton"), json=payload)
        response.raise_for_status()
