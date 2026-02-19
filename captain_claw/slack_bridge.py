"""Minimal Slack Web API bridge (DM polling + send message)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class SlackMessage:
    """Normalized incoming Slack message payload."""

    channel_id: str
    message_ts: str
    user_id: str
    username: str
    text: str
    thread_ts: str = ""


class SlackBridge:
    """Slack Web API helper for DM-first bot interaction."""

    def __init__(self, token: str, api_base_url: str = "https://slack.com/api"):
        self.token = str(token or "").strip()
        self.api_base_url = (api_base_url or "https://slack.com/api").rstrip("/")
        self._client = httpx.AsyncClient(timeout=40.0)
        self._user_cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def close(self) -> None:
        await self._client.aclose()

    def _url(self, method: str) -> str:
        return f"{self.api_base_url}/{method.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def _api_get(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._client.get(
            self._url(method),
            headers=self._headers(),
            params=params or {},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid Slack response for {method}")
        if payload.get("ok") is not True:
            raise RuntimeError(f"Slack API {method} failed: {payload.get('error', 'unknown_error')}")
        return payload

    async def _api_post(
        self,
        method: str,
        *,
        json_payload: dict[str, Any] | None = None,
        data_payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._client.post(
            self._url(method),
            headers=self._headers(),
            json=json_payload,
            data=data_payload,
            files=files,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid Slack response for {method}")
        if payload.get("ok") is not True:
            raise RuntimeError(f"Slack API {method} failed: {payload.get('error', 'unknown_error')}")
        return payload

    async def _resolve_username(self, user_id: str) -> str:
        user_id_clean = str(user_id or "").strip()
        if not user_id_clean:
            return ""
        cached = self._user_cache.get(user_id_clean)
        if cached is not None:
            return cached
        try:
            payload = await self._api_get("users.info", {"user": user_id_clean})
            user = payload.get("user")
            if isinstance(user, dict):
                profile = user.get("profile")
                if isinstance(profile, dict):
                    display_name = str(profile.get("display_name", "")).strip()
                    if display_name:
                        self._user_cache[user_id_clean] = display_name
                        return display_name
                real_name = str(user.get("real_name", "")).strip()
                if real_name:
                    self._user_cache[user_id_clean] = real_name
                    return real_name
                handle = str(user.get("name", "")).strip()
                if handle:
                    self._user_cache[user_id_clean] = handle
                    return handle
        except Exception:
            pass
        self._user_cache[user_id_clean] = ""
        return ""

    async def _list_dm_channels(self) -> list[dict[str, Any]]:
        channels: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {
                "types": "im",
                "exclude_archived": True,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            payload = await self._api_get("conversations.list", params)
            items = payload.get("channels")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        channels.append(item)
            metadata = payload.get("response_metadata")
            next_cursor = ""
            if isinstance(metadata, dict):
                next_cursor = str(metadata.get("next_cursor", "")).strip()
            if not next_cursor:
                break
            cursor = next_cursor
        return channels

    @staticmethod
    def _ts_to_float(raw: str) -> float:
        try:
            return float(str(raw or "0"))
        except Exception:
            return 0.0

    async def get_updates(
        self,
        offsets: dict[str, str] | None = None,
        *,
        limit_per_channel: int = 20,
    ) -> tuple[list[SlackMessage], dict[str, str]]:
        """Poll Slack DM messages and return updates with refreshed offsets."""
        current_offsets = dict(offsets or {})
        next_offsets = dict(current_offsets)
        updates: list[SlackMessage] = []

        for channel in await self._list_dm_channels():
            channel_id = str(channel.get("id", "")).strip()
            if not channel_id:
                continue
            oldest = str(current_offsets.get(channel_id, "")).strip()
            params: dict[str, Any] = {
                "channel": channel_id,
                "limit": max(1, int(limit_per_channel)),
                "inclusive": False,
            }
            if oldest:
                params["oldest"] = oldest
            payload = await self._api_get("conversations.history", params)
            messages = payload.get("messages")
            if not isinstance(messages, list):
                continue
            sorted_messages = sorted(
                (entry for entry in messages if isinstance(entry, dict)),
                key=lambda entry: self._ts_to_float(str(entry.get("ts", ""))),
            )
            last_seen_ts = oldest
            for entry in sorted_messages:
                ts = str(entry.get("ts", "")).strip()
                if ts:
                    if not last_seen_ts or self._ts_to_float(ts) > self._ts_to_float(last_seen_ts):
                        last_seen_ts = ts
                text = str(entry.get("text", "")).strip()
                user_id = str(entry.get("user", "")).strip()
                subtype = str(entry.get("subtype", "")).strip().lower()
                if not text or not user_id:
                    continue
                if subtype in {"bot_message", "message_changed", "message_deleted"}:
                    continue
                username = await self._resolve_username(user_id)
                updates.append(
                    SlackMessage(
                        channel_id=channel_id,
                        message_ts=ts,
                        user_id=user_id,
                        username=username,
                        text=text,
                        thread_ts=str(entry.get("thread_ts", "")).strip(),
                    )
                )
            if last_seen_ts:
                next_offsets[channel_id] = last_seen_ts

        updates.sort(key=lambda item: self._ts_to_float(item.message_ts))
        return updates, next_offsets

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_message_ts: str = "",
    ) -> None:
        """Send text to Slack DM, splitting into safe-sized chunks."""
        raw = str(text or "").strip()
        if not raw:
            return
        max_len = 3500
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
                "channel": str(channel_id),
                "text": chunk,
                "mrkdwn": True,
            }
            if reply_to_message_ts and idx == 0:
                payload["thread_ts"] = str(reply_to_message_ts)
            await self._api_post("chat.postMessage", json_payload=payload)

    async def send_chat_action(self, channel_id: str, action: str = "typing") -> None:
        """Best-effort typing signal (Slack Web API has limited support)."""
        action_name = str(action or "").strip().lower()
        if action_name != "typing":
            return
        try:
            await self._api_post(
                "conversations.mark",
                json_payload={"channel": str(channel_id)},
            )
        except Exception:
            # Slack Web API does not provide a stable typing endpoint for bots.
            return

    async def mark_read(self, channel_id: str, message_ts: str) -> None:
        """Best-effort mark read for DM channel."""
        if not str(channel_id).strip() or not str(message_ts).strip():
            return
        try:
            await self._api_post(
                "conversations.mark",
                json_payload={"channel": str(channel_id), "ts": str(message_ts)},
            )
        except Exception:
            return

    async def send_audio_file(
        self,
        channel_id: str,
        file_path: str | Path,
        *,
        caption: str = "",
    ) -> None:
        """Upload MP3 file to Slack channel."""
        audio_path = Path(file_path).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(f"Slack audio file not found: {audio_path}")
        data: dict[str, Any] = {
            "channels": str(channel_id),
            "initial_comment": str(caption or "").strip(),
            "filename": audio_path.name,
            "title": audio_path.name,
        }
        with audio_path.open("rb") as handle:
            files = {
                "file": (
                    audio_path.name,
                    handle,
                    "audio/mpeg",
                )
            }
            await self._api_post(
                "files.upload",
                data_payload=data,
                files=files,
            )
