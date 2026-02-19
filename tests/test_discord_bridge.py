from pathlib import Path

import pytest

from captain_claw.discord_bridge import DiscordBridge


@pytest.mark.asyncio
async def test_get_updates_filters_bot_messages_and_updates_offsets(monkeypatch):
    bridge = DiscordBridge(token="discord-test")

    async def fake_api_get(path: str, params=None):
        if path == "users/@me":
            return {"id": "BOT1"}
        if path == "users/@me/channels":
            return [{"id": "C123", "type": 1}]
        if path == "channels/C123/messages":
            return [
                {
                    "id": "200",
                    "content": "hello from discord",
                    "author": {"id": "U1", "username": "alice", "bot": False},
                },
                {
                    "id": "201",
                    "content": "ignore",
                    "author": {"id": "B1", "username": "bot", "bot": True},
                },
            ]
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(bridge, "_api_get", fake_api_get)

    updates, offsets = await bridge.get_updates({"C123": "100"}, include_guild_channels=False)

    assert len(updates) == 1
    assert updates[0].channel_id == "C123"
    assert updates[0].user_id == "U1"
    assert updates[0].text == "hello from discord"
    assert updates[0].guild_id == ""
    assert offsets["C123"] == "201"


@pytest.mark.asyncio
async def test_get_updates_guild_message_requires_mention_flag(monkeypatch):
    bridge = DiscordBridge(token="discord-test")

    async def fake_api_get(path: str, params=None):
        if path == "users/@me":
            return {"id": "BOT1"}
        if path == "users/@me/channels":
            return []
        if path == "users/@me/guilds":
            return [{"id": "G123"}]
        if path == "guilds/G123/channels":
            return [{"id": "T123", "type": 0}]
        if path == "channels/T123/messages":
            return [
                {
                    "id": "300",
                    "content": "<@BOT1> what's up?",
                    "mentions": [{"id": "BOT1"}],
                    "author": {"id": "U2", "username": "bob", "bot": False},
                }
            ]
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(bridge, "_api_get", fake_api_get)

    updates, offsets = await bridge.get_updates({"T123": "250"})

    assert len(updates) == 1
    assert updates[0].guild_id == "G123"
    assert updates[0].mentioned_bot is True
    assert updates[0].text == "what's up?"
    assert offsets["T123"] == "300"


class _FakeResponse:
    def __init__(self, content: bytes = b"{}"):
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {}


class _FakeClient:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    async def post(self, url: str, **kwargs):
        payload = {"url": url}
        payload.update(kwargs)
        files = payload.get("files")
        if isinstance(files, dict) and "files[0]" in files:
            _, handle, _ = files["files[0]"]
            handle.seek(0)
            payload["bytes"] = handle.read()
        self.posts.append(payload)
        return _FakeResponse()

    async def get(self, url: str, **kwargs):
        return _FakeResponse()

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_send_audio_file_posts_discord_multipart(tmp_path: Path):
    bridge = DiscordBridge(token="discord-test")
    fake_client = _FakeClient()
    bridge._client = fake_client

    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"ID3FAKE")

    await bridge.send_audio_file("C123", audio_path, caption="summary", reply_to_message_id="99")

    assert len(fake_client.posts) == 1
    call = fake_client.posts[0]
    assert "channels/C123/messages" in call["url"]
    assert "payload_json" in call["data"]
    assert bytes(call["bytes"]).startswith(b"ID3")
