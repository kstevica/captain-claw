import types
from pathlib import Path

import pytest

from captain_claw.config import get_config, set_config
from captain_claw.tools.pocket_tts import PocketTTSTool


class _FakePocketModel:
    default_voice = "af_demo"
    sample_rate = 16000

    @classmethod
    def load_model(cls):
        return cls()

    def get_state_for_audio_prompt(self, voice: str):
        return {"voice": voice}

    def generate_audio(self, *args, **kwargs):
        del args, kwargs
        return [0.0, 0.25, -0.25, 0.5, -0.5]


@pytest.mark.asyncio
async def test_pocket_tts_reports_missing_dependency(monkeypatch, tmp_path: Path):
    import importlib

    original_import_module = importlib.import_module

    def _fake_import(name: str, package=None):
        if name == "pocket_tts":
            raise ModuleNotFoundError("pocket_tts not installed")
        return original_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    tool = PocketTTSTool()
    result = await tool.execute(
        text="hello world",
        _saved_base_path=tmp_path / "saved",
        _session_id="demo",
    )

    assert result.success is False
    assert "pocket-tts" in str(result.error or "").lower()
    assert "install" in str(result.error or "").lower()


@pytest.mark.asyncio
async def test_pocket_tts_generates_mp3_under_saved_media(monkeypatch, tmp_path: Path):
    import importlib

    original_import_module = importlib.import_module

    def _fake_import(name: str, package=None):
        if name == "pocket_tts":
            return types.SimpleNamespace(TTSModel=_FakePocketModel)
        return original_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    monkeypatch.setattr(
        "captain_claw.tools.pocket_tts._encode_mp3_bytes",
        lambda pcm_data, sample_rate, bitrate_kbps: b"ID3FAKE",
    )

    tool = PocketTTSTool()
    saved_root = tmp_path / "saved"
    result = await tool.execute(
        text="Captain Claw speaking",
        voice="af_sample",
        _saved_base_path=saved_root,
        _session_id="session-1",
    )

    assert result.success is True
    assert "Generated speech audio with pocket-tts." in result.content
    assert "Voice: af_sample" in result.content
    assert "Format: mp3" in result.content
    assert "Bitrate: 128 kbps" in result.content

    output_files = list((saved_root / "media" / "session-1").glob("*.mp3"))
    assert len(output_files) == 1
    assert output_files[0].read_bytes().startswith(b"ID3")


@pytest.mark.asyncio
async def test_pocket_tts_enforces_max_text_length():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.pocket_tts.max_chars = 5
    set_config(cfg)
    try:
        tool = PocketTTSTool()
        result = await tool.execute(text="this text is longer than five chars")
        assert result.success is False
        assert "Text too long" in str(result.error or "")
    finally:
        set_config(old_cfg)
