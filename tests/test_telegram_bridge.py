from pathlib import Path

import pytest

from captain_claw.telegram_bridge import TelegramBridge


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self):
        self.posts: list[dict] = []

    async def post(self, url: str, **kwargs):
        payload = {"url": url}
        payload.update(kwargs)
        files = payload.get("files")
        if isinstance(files, dict) and "audio" in files:
            name, handle, content_type = files["audio"]
            handle.seek(0)
            payload["audio_meta"] = {
                "name": name,
                "content_type": content_type,
                "bytes": handle.read(),
            }
        self.posts.append(payload)
        return _FakeResponse()

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_send_audio_file_posts_multipart_mp3(tmp_path: Path):
    bridge = TelegramBridge(token="token")
    fake_client = _FakeClient()
    bridge._client = fake_client

    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"ID3FAKE-MP3-DATA")

    await bridge.send_audio_file(
        chat_id=12345,
        file_path=audio_path,
        caption="Audio summary",
        reply_to_message_id=99,
    )

    assert len(fake_client.posts) == 1
    call = fake_client.posts[0]
    assert "sendAudio" in call["url"]
    assert int(call["data"]["chat_id"]) == 12345
    assert str(call["data"]["caption"]) == "Audio summary"
    assert int(call["data"]["reply_to_message_id"]) == 99
    audio_meta = call.get("audio_meta", {})
    assert audio_meta.get("name") == "sample.mp3"
    assert audio_meta.get("content_type") == "audio/mpeg"
    assert bytes(audio_meta.get("bytes", b"")).startswith(b"ID3")


@pytest.mark.asyncio
async def test_send_audio_file_raises_when_missing():
    bridge = TelegramBridge(token="token")
    fake_client = _FakeClient()
    bridge._client = fake_client

    with pytest.raises(FileNotFoundError):
        await bridge.send_audio_file(chat_id=1, file_path="/tmp/does-not-exist.mp3")
