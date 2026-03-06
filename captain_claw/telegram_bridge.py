"""Minimal Telegram Bot API bridge (long polling + send message)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


def _md_to_telegram_html(text: str) -> str:
    """Convert common Markdown formatting to Telegram-compatible HTML.

    Supports: **bold**, *italic*, `code`, ```code blocks```, [text](url),
    ~~strikethrough~~.  Unsupported or nested patterns are left as-is.
    HTML entities in the source text are escaped first so the output is safe
    to send with ``parse_mode="HTML"``.
    """
    # Escape HTML entities FIRST so user content doesn't break tags.
    text = html.escape(text, quote=False)

    # Fenced code blocks: ```lang\n...\n``` → <pre>...</pre>
    text = re.sub(
        r"```(?:\w*)\n(.*?)```",
        lambda m: f"<pre>{m.group(1).rstrip()}</pre>",
        text,
        flags=re.DOTALL,
    )
    # Inline code: `...` → <code>...</code>
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)

    # Bold: **text** → <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    # Italic: *text* → <i>text</i> (but not inside <b> tags from bold)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    # Strikethrough: ~~text~~ → <s>text</s>
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Links: [text](url) → <a href="url">text</a>
    text = re.sub(
        r"\[([^\]]+?)\]\(([^)]+?)\)",
        r'<a href="\2">\1</a>',
        text,
    )

    # Strip heading markers: ### Heading → <b>Heading</b>
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    return text


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
    photo_file_id: str = ""
    business_connection_id: str = ""


@dataclass
class TelegramCallbackQuery:
    """Normalized incoming Telegram callback query (inline keyboard button press)."""

    update_id: int
    callback_query_id: str
    chat_id: int
    user_id: int
    username: str
    first_name: str
    data: str  # callback_data from the button
    message_id: int = 0  # message the button was attached to


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

    async def get_updates(
        self, offset: int | None = None, timeout: int = 25,
    ) -> list[TelegramMessage | TelegramCallbackQuery]:
        """Poll Telegram updates and normalize messages and callback queries."""
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

        updates: list[TelegramMessage | TelegramCallbackQuery] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            update_id = int(item.get("update_id", 0))

            # Handle callback queries (inline keyboard button presses).
            cbq = item.get("callback_query")
            if isinstance(cbq, dict):
                cb_from = cbq.get("from")
                cb_msg = cbq.get("message")
                if isinstance(cb_from, dict):
                    cb_chat_id = 0
                    cb_message_id = 0
                    if isinstance(cb_msg, dict):
                        cb_chat = cb_msg.get("chat")
                        if isinstance(cb_chat, dict):
                            cb_chat_id = int(cb_chat.get("id", 0))
                        cb_message_id = int(cb_msg.get("message_id", 0))
                    updates.append(
                        TelegramCallbackQuery(
                            update_id=update_id,
                            callback_query_id=str(cbq.get("id", "")),
                            chat_id=cb_chat_id or int(cb_from.get("id", 0)),
                            user_id=int(cb_from.get("id", 0)),
                            username=str(cb_from.get("username", "")).strip(),
                            first_name=str(cb_from.get("first_name", "")).strip(),
                            data=str(cbq.get("data", "")).strip(),
                            message_id=cb_message_id,
                        )
                    )
                continue

            # Handle regular messages.
            msg = item.get("message")
            if not isinstance(msg, dict):
                msg = item.get("business_message")
            if not isinstance(msg, dict):
                continue
            # Extract text, or caption + photo for image messages.
            text = str(msg.get("text", "")).strip()
            photo_file_id = ""
            photo_array = msg.get("photo")
            if isinstance(photo_array, list) and photo_array:
                # Telegram sends multiple sizes; pick the largest (last).
                largest = photo_array[-1]
                if isinstance(largest, dict):
                    photo_file_id = str(largest.get("file_id", "")).strip()
                # Use caption as text for photo messages.
                if not text:
                    text = str(msg.get("caption", "")).strip()
            if not text and not photo_file_id:
                continue
            from_user = msg.get("from")
            chat = msg.get("chat")
            if not isinstance(from_user, dict) or not isinstance(chat, dict):
                continue
            user_id = int(from_user.get("id", 0))
            chat_id = int(chat.get("id", 0))
            if user_id == 0 or chat_id == 0:
                continue
            updates.append(
                TelegramMessage(
                    update_id=update_id,
                    message_id=int(msg.get("message_id", 0)),
                    chat_id=chat_id,
                    user_id=user_id,
                    username=str(from_user.get("username", "")).strip(),
                    first_name=str(from_user.get("first_name", "")).strip(),
                    text=text,
                    photo_file_id=photo_file_id,
                    business_connection_id=str(msg.get("business_connection_id", "")).strip(),
                )
            )
        return updates

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send text to Telegram, splitting at 4096 chars when needed.

        Markdown in *text* is converted to Telegram HTML so formatting
        renders natively in the Telegram app.  If the API rejects the
        HTML (malformed tags etc.), the message is re-sent as plain text.
        """
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
            html_chunk = _md_to_telegram_html(chunk)
            payload: dict[str, Any] = {
                "chat_id": int(chat_id),
                "text": html_chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_to_message_id and idx == 0:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            response = await self._client.post(self._url("sendMessage"), json=payload)
            # Fallback: if Telegram rejects the HTML, resend as plain text.
            if response.status_code == 400:
                payload["text"] = chunk
                del payload["parse_mode"]
                response = await self._client.post(self._url("sendMessage"), json=payload)
            response.raise_for_status()

    async def send_message_with_inline_keyboard(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[dict[str, str]]],
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send a message with inline keyboard buttons.

        *buttons* is a list of rows, each row is a list of button dicts
        with keys ``text`` and ``callback_data``.
        """
        raw = str(text or "").strip() or "\u200b"  # zero-width space fallback
        html_text = _md_to_telegram_html(raw)
        payload: dict[str, Any] = {
            "chat_id": int(chat_id),
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "reply_markup": {"inline_keyboard": buttons},
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = int(reply_to_message_id)
        response = await self._client.post(self._url("sendMessage"), json=payload)
        if response.status_code == 400:
            payload["text"] = raw
            del payload["parse_mode"]
            response = await self._client.post(self._url("sendMessage"), json=payload)
        response.raise_for_status()

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
    ) -> None:
        """Acknowledge a callback query (inline keyboard button press)."""
        payload: dict[str, Any] = {
            "callback_query_id": str(callback_query_id),
        }
        if text:
            payload["text"] = str(text)[:200]
        response = await self._client.post(self._url("answerCallbackQuery"), json=payload)
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

    async def send_photo(
        self,
        chat_id: int,
        file_path: str | Path,
        *,
        caption: str = "",
        reply_to_message_id: int | None = None,
    ) -> None:
        """Upload an image file to Telegram using sendPhoto."""
        photo_path = Path(file_path).expanduser().resolve()
        if not photo_path.exists() or not photo_path.is_file():
            raise FileNotFoundError(f"Telegram photo file not found: {photo_path}")
        payload: dict[str, Any] = {"chat_id": int(chat_id)}
        caption_text = str(caption or "").strip()
        if caption_text:
            payload["caption"] = caption_text
        if reply_to_message_id:
            payload["reply_to_message_id"] = int(reply_to_message_id)
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = mime_types.get(photo_path.suffix.lower(), "image/png")
        with photo_path.open("rb") as handle:
            files = {
                "photo": (
                    photo_path.name,
                    handle,
                    mime,
                )
            }
            response = await self._client.post(
                self._url("sendPhoto"),
                data=payload,
                files=files,
            )
        response.raise_for_status()

    async def download_file(self, file_id: str, dest: Path) -> Path:
        """Download a Telegram file by file_id to a local path."""
        # Step 1: getFile to get file_path on Telegram servers.
        resp = await self._client.post(
            self._url("getFile"), json={"file_id": file_id},
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError(f"getFile failed: {data}")
        file_path = data.get("result", {}).get("file_path", "")
        if not file_path:
            raise RuntimeError("getFile returned no file_path")
        # Step 2: Download the file bytes.
        download_url = f"{self.api_base_url}/file/bot{self.token}/{file_path}"
        dl_resp = await self._client.get(download_url)
        dl_resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(dl_resp.content)
        return dest

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
