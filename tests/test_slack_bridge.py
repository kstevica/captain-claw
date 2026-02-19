from pathlib import Path

import pytest

from captain_claw.slack_bridge import SlackBridge


@pytest.mark.asyncio
async def test_get_updates_filters_bot_messages_and_updates_offsets(monkeypatch):
    bridge = SlackBridge(token="xoxb-test")

    async def fake_api_get(method: str, params=None):
        if method == "conversations.list":
            return {"ok": True, "channels": [{"id": "D123"}], "response_metadata": {"next_cursor": ""}}
        if method == "conversations.history":
            return {
                "ok": True,
                "messages": [
                    {"ts": "1700000002.0001", "user": "U1", "text": "hello from slack"},
                    {"ts": "1700000002.0000", "subtype": "bot_message", "text": "ignore me"},
                ],
            }
        if method == "users.info":
            return {"ok": True, "user": {"name": "alice"}}
        raise AssertionError(f"Unexpected method: {method}")

    monkeypatch.setattr(bridge, "_api_get", fake_api_get)

    updates, offsets = await bridge.get_updates({"D123": "1700000000.0000"})

    assert len(updates) == 1
    assert updates[0].channel_id == "D123"
    assert updates[0].user_id == "U1"
    assert updates[0].text == "hello from slack"
    assert offsets["D123"] == "1700000002.0001"


@pytest.mark.asyncio
async def test_send_audio_file_uses_files_upload(monkeypatch, tmp_path: Path):
    bridge = SlackBridge(token="xoxb-test")
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"ID3FAKE")
    captured: dict[str, object] = {}

    async def fake_api_post(method: str, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        files = kwargs.get("files")
        if isinstance(files, dict) and "file" in files:
            _, handle, _ = files["file"]
            handle.seek(0)
            captured["bytes"] = handle.read()
        return {"ok": True}

    monkeypatch.setattr(bridge, "_api_post", fake_api_post)

    await bridge.send_audio_file("D123", audio_path, caption="summary")

    assert captured["method"] == "files.upload"
    kwargs = captured["kwargs"]
    assert kwargs["data_payload"]["channels"] == "D123"
    assert kwargs["data_payload"]["initial_comment"] == "summary"
    assert bytes(captured["bytes"]).startswith(b"ID3")
