"""Minimal Discord Bot API bridge (DM polling + send message)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class DiscordMessage:
    """Normalized incoming Discord message payload."""

    id: str
    channel_id: str
    user_id: str
    username: str
    text: str
    guild_id: str = ""
    mentioned_bot: bool = False


class DiscordBridge:
    """Discord REST helper for DM + guild-channel polling interaction."""

    def __init__(self, token: str, api_base_url: str = "https://discord.com/api/v10"):
        self.token = str(token or "").strip()
        self.api_base_url = (api_base_url or "https://discord.com/api/v10").rstrip("/")
        self._client = httpx.AsyncClient(timeout=40.0)
        self._bot_user_id = ""

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def close(self) -> None:
        await self._client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.api_base_url}/{path.lstrip('/')}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bot {self.token}"}

    async def _api_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await self._client.get(
            self._url(path),
            headers=self._auth_headers(),
            params=params or {},
        )
        response.raise_for_status()
        return response.json()

    async def _api_post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = await self._client.post(
            self._url(path),
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    async def _list_dm_channels(self) -> list[dict[str, Any]]:
        payload = await self._api_get("users/@me/channels")
        if not isinstance(payload, list):
            return []
        channels: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            channel_type = int(entry.get("type", 0) or 0)
            if channel_type in {1, 3}:
                channels.append(entry)
        return channels

    async def _get_bot_user_id(self) -> str:
        cached = str(self._bot_user_id or "").strip()
        if cached:
            return cached
        payload = await self._api_get("users/@me")
        if not isinstance(payload, dict):
            return ""
        bot_id = str(payload.get("id", "")).strip()
        if bot_id:
            self._bot_user_id = bot_id
        return bot_id

    async def _list_guild_text_channels(self) -> list[dict[str, Any]]:
        payload = await self._api_get("users/@me/guilds")
        if not isinstance(payload, list):
            return []
        channels: list[dict[str, Any]] = []
        for guild in payload:
            if not isinstance(guild, dict):
                continue
            guild_id = str(guild.get("id", "")).strip()
            if not guild_id:
                continue
            guild_channels = await self._api_get(f"guilds/{guild_id}/channels")
            if not isinstance(guild_channels, list):
                continue
            for entry in guild_channels:
                if not isinstance(entry, dict):
                    continue
                channel_id = str(entry.get("id", "")).strip()
                try:
                    channel_type = int(entry.get("type", -1))
                except Exception:
                    channel_type = -1
                if not channel_id or channel_type not in {0, 5}:
                    continue
                channels.append(
                    {
                        "id": channel_id,
                        "guild_id": guild_id,
                    }
                )
        return channels

    @staticmethod
    def _id_to_int(raw: str) -> int:
        try:
            return int(str(raw or "0"))
        except Exception:
            return 0

    @staticmethod
    def _format_username(author: dict[str, Any]) -> str:
        global_name = str(author.get("global_name", "")).strip()
        if global_name:
            return global_name
        username = str(author.get("username", "")).strip()
        discriminator = str(author.get("discriminator", "")).strip()
        if username and discriminator and discriminator != "0":
            return f"{username}#{discriminator}"
        return username

    @staticmethod
    def _strip_leading_mentions(content: str) -> str:
        text = str(content or "").strip()
        while text.startswith("<@"):
            end = text.find(">")
            if end <= 0:
                break
            text = text[end + 1 :].lstrip(" ,:\t")
        return text.strip()

    async def get_updates(
        self,
        offsets: dict[str, str] | None = None,
        *,
        limit_per_channel: int = 50,
        include_guild_channels: bool = True,
    ) -> tuple[list[DiscordMessage], dict[str, str]]:
        """Poll Discord channels and return updates with refreshed offsets."""
        current_offsets = dict(offsets or {})
        next_offsets = dict(current_offsets)
        updates: list[DiscordMessage] = []
        bot_user_id = ""
        try:
            bot_user_id = await self._get_bot_user_id()
        except Exception:
            bot_user_id = ""

        channels: list[dict[str, Any]] = []
        seen_channel_ids: set[str] = set()

        try:
            dm_channels = await self._list_dm_channels()
        except Exception:
            dm_channels = []
        for channel in dm_channels:
            channel_id = str(channel.get("id", "")).strip()
            if not channel_id or channel_id in seen_channel_ids:
                continue
            seen_channel_ids.add(channel_id)
            channels.append({"id": channel_id, "guild_id": ""})

        if include_guild_channels:
            try:
                guild_channels = await self._list_guild_text_channels()
            except Exception:
                guild_channels = []
            for channel in guild_channels:
                channel_id = str(channel.get("id", "")).strip()
                if not channel_id or channel_id in seen_channel_ids:
                    continue
                seen_channel_ids.add(channel_id)
                channels.append(
                    {
                        "id": channel_id,
                        "guild_id": str(channel.get("guild_id", "")).strip(),
                    }
                )

        for channel in channels:
            channel_id = str(channel.get("id", "")).strip()
            if not channel_id:
                continue
            guild_id = str(channel.get("guild_id", "")).strip()
            after_id = str(current_offsets.get(channel_id, "")).strip()
            params: dict[str, Any] = {"limit": max(1, min(100, int(limit_per_channel)))}
            if after_id:
                params["after"] = after_id
            payload = await self._api_get(f"channels/{channel_id}/messages", params=params)
            if not isinstance(payload, list):
                continue
            sorted_messages = sorted(
                (entry for entry in payload if isinstance(entry, dict)),
                key=lambda entry: self._id_to_int(str(entry.get("id", ""))),
            )
            last_seen_id = after_id
            for entry in sorted_messages:
                message_id = str(entry.get("id", "")).strip()
                if message_id:
                    if not last_seen_id or self._id_to_int(message_id) > self._id_to_int(last_seen_id):
                        last_seen_id = message_id
                content = str(entry.get("content", "")).strip()
                author = entry.get("author")
                if not isinstance(author, dict):
                    continue
                if bool(author.get("bot")):
                    continue
                user_id = str(author.get("id", "")).strip()
                mentioned_bot = False
                mentions = entry.get("mentions")
                if isinstance(mentions, list) and bot_user_id:
                    for mention in mentions:
                        if not isinstance(mention, dict):
                            continue
                        if str(mention.get("id", "")).strip() == bot_user_id:
                            mentioned_bot = True
                            break
                if guild_id and mentioned_bot:
                    content = self._strip_leading_mentions(content)
                if not user_id or not content:
                    continue
                updates.append(
                    DiscordMessage(
                        id=message_id,
                        channel_id=channel_id,
                        user_id=user_id,
                        username=self._format_username(author),
                        text=content,
                        guild_id=guild_id,
                        mentioned_bot=mentioned_bot,
                    )
                )
            if last_seen_id:
                next_offsets[channel_id] = last_seen_id

        updates.sort(key=lambda item: self._id_to_int(item.id))
        return updates, next_offsets

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_message_id: str = "",
    ) -> None:
        """Send text to Discord channel, splitting large content into chunks."""
        raw = str(text or "").strip()
        if not raw:
            return
        max_len = 1800
        chunks: list[str] = []
        while raw:
            if len(raw) <= max_len:
                chunks.append(raw)
                break
            split_at = raw.rfind("\n", 0, max_len)
            if split_at < 500:
                split_at = max_len
            chunks.append(raw[:split_at].rstrip())
            raw = raw[split_at:].lstrip()

        for idx, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "content": chunk,
                "allowed_mentions": {"parse": []},
            }
            if reply_to_message_id and idx == 0:
                payload["message_reference"] = {"message_id": str(reply_to_message_id)}
            await self._api_post_json(f"channels/{channel_id}/messages", payload)

    async def send_chat_action(self, channel_id: str, action: str = "typing") -> None:
        """Send Discord typing indicator while processing a prompt."""
        if str(action or "").strip().lower() != "typing":
            return
        response = await self._client.post(
            self._url(f"channels/{channel_id}/typing"),
            headers=self._auth_headers(),
        )
        response.raise_for_status()

    async def mark_read(self, channel_id: str, message_id: str) -> None:
        """Bots cannot mark messages read in Discord; kept as no-op hook."""
        _ = (channel_id, message_id)
        return

    async def send_audio_file(
        self,
        channel_id: str,
        file_path: str | Path,
        *,
        caption: str = "",
        reply_to_message_id: str = "",
    ) -> None:
        """Upload MP3 audio file to Discord channel."""
        audio_path = Path(file_path).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(f"Discord audio file not found: {audio_path}")
        payload: dict[str, Any] = {
            "content": str(caption or "").strip(),
            "allowed_mentions": {"parse": []},
        }
        if reply_to_message_id:
            payload["message_reference"] = {"message_id": str(reply_to_message_id)}
        with audio_path.open("rb") as handle:
            files = {
                "files[0]": (
                    audio_path.name,
                    handle,
                    "audio/mpeg",
                )
            }
            response = await self._client.post(
                self._url(f"channels/{channel_id}/messages"),
                headers=self._auth_headers(),
                data={"payload_json": json.dumps(payload, ensure_ascii=True)},
                files=files,
            )
        response.raise_for_status()
