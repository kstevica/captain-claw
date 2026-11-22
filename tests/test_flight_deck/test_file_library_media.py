"""The File library reads pictures, recordings and videos too.

A picture is shown to a model that can see — when its agent's vision is on —
and its description goes into search. A recording or a video is transcribed
with Soniox (the sound taken out with ffmpeg first, when there is one) and
summarised from the transcript, which search goes through. A file set aside
for want of vision, a Soniox key or ffmpeg is taken up again by the first
scan that has it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import types
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from captain_claw import llm
from captain_claw.flight_deck import file_library as fl
from captain_claw.flight_deck import server, soniox_routes
from captain_claw.llm import reasoning, vision
from captain_claw.soniox_api import SonioxError, Transcript
from captain_claw.tools import video_vision

SLUG = "desk-bot"
USER = {"id": "local"}
LIBRARY_TIER = {"provider": "anthropic", "model": "claude-haiku-4-5", "api_key": "sk-lib"}
TEXT_ONLY_TIER = {"provider": "openai", "model": "qwen3.5-35b-text-only", "base_url": "http://127.0.0.1:1234/v1",
                  "api_key": "local"}


@pytest.fixture
def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A deck without sign-in, one stopped agent, no Soniox key and no
    ffmpeg — each test hands out what it needs."""
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(tmp_path / "file_library.db"))
    monkeypatch.setattr(fl, "_JOBS", {})
    # Refused pictures are remembered here, never in this machine's own store.
    monkeypatch.setattr(vision, "_REJECTED_STORE", tmp_path / "vision_rejected.json")
    monkeypatch.setattr(vision, "_VISION_REJECTED", {})
    monkeypatch.setattr(vision, "_rejected_loaded", False)
    server._save_process_registry({SLUG: {"slug": SLUG, "name": "Desk", "web_port": 24101,
                                          "web_auth": "tok", "owner": "", "pid": None}})
    (data / SLUG / "data" / "workspace").mkdir(parents=True)

    state = _State()

    async def tier(_uid):
        return state.tier

    async def soniox_status(_owner):
        return {"key_set": bool(state.soniox_key), "verify_error": state.soniox_verify}

    async def transcribe_for(owner, audio, **kw):
        state.transcribed.append({"owner": owner, "audio": audio, **kw})
        if state.soniox_fail is not None:
            if getattr(state.soniox_fail, "refused", False):
                state.soniox_verify = "refused"      # as soniox_routes marks the key
            raise state.soniox_fail
        return Transcript(text=state.transcript)

    async def call_model(tier, system, user, owner_id, max_tokens=700, request_timeout=None, images=None):
        state.calls.append({"tier": tier, "system": system, "user": user, "images": images,
                            "max_tokens": max_tokens, "effort": reasoning.requested_effort()})
        model = tier.get("model")
        state.in_flight[model] = state.in_flight.get(model, 0) + 1
        state.most_in_flight[model] = max(state.most_in_flight.get(model, 0), state.in_flight[model])
        try:
            if state.delay:
                await asyncio.sleep(state.delay)
        finally:
            state.in_flight[model] -= 1
        if images and tier.get("model") in state.refuse:
            # What the real provider does when its server refuses the picture.
            provider = llm.create_provider(provider=tier["provider"], model=tier["model"],
                                           base_url=tier.get("base_url") or None, api_key="k")
            llm._remember_provider_rejected_images(provider)
            raise fl.PictureRefusedError("no pictures here")
        if tier.get("model") in state.broken:
            raise RuntimeError(f"model '{tier.get('model')}' not found")
        return json.dumps(state.reply), 123

    monkeypatch.setattr(fl, "_resolve_tier", tier)
    monkeypatch.setattr(soniox_routes, "status_for_owner", soniox_status)
    monkeypatch.setattr(soniox_routes, "transcribe_for", transcribe_for)
    monkeypatch.setattr(fl, "call_model", call_model)
    monkeypatch.setattr(fl, "_find_tool", lambda name: state.tools.get(name, ""))
    state.data = data
    return state


class _State:
    def __init__(self) -> None:
        self.tier: dict | None = dict(LIBRARY_TIER)
        self.soniox_key = ""
        self.soniox_verify: str | None = None
        self.soniox_fail: Exception | None = None
        self.transcript = "We agreed to move the launch of Project Halibut to March."
        self.transcribed: list[dict] = []
        self.calls: list[dict] = []
        self.refuse: set[str] = set()
        self.broken: set[str] = set()
        self.delay = 0.0
        self.in_flight: dict[str, int] = {}
        self.most_in_flight: dict[str, int] = {}
        self.tools: dict[str, str] = {}
        self.reply = {"title": "A title", "summary": "What it is.", "category": "notes",
                      "tags": ["one", "two"], "description": "A red bicycle against a brick wall; "
                      "a sign reads OSTERIA LUNA."}
        self.data = Path()

    @property
    def workspace(self) -> Path:
        return self.data / SLUG / "data" / "workspace"

    def agent_model(self, **model) -> None:
        import yaml
        (self.data / SLUG / "config.yaml").write_text(yaml.safe_dump({"model": model}))


async def _scan(**kw) -> fl.ScanJob:
    job = fl.ScanJob(running=True)
    await fl.run_scan(USER, job, **kw)
    return job


def _row(name: str) -> dict:
    [row] = [r for r in fl.get_store().search([SLUG], []) if r["filename"] == name]
    return row


def _found(q: str) -> list[str]:
    return sorted(r["filename"] for r in fl.get_store().search([SLUG], q.casefold().split()))


def _picture(path: Path, size=(3000, 2000), mode="RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, (200, 30, 30, 128) if mode == "RGBA" else (200, 30, 30)).save(path)
    return path


def _sound(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i % 64) * 200) for i in range(int(16000 * seconds))))
    return path


# ── Pictures ──────────────────────────────────────────────────────────


async def test_picture_is_described_when_the_agents_vision_is_on(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    _picture(deck.workspace / "saved" / "bike.png", mode="RGBA")
    await _scan()

    [call] = deck.calls
    assert call["tier"] is not None and call["tier"]["model"] == "claude-haiku-4-5"
    [pic] = call["images"]
    assert pic.mime == "image/jpeg" and max(pic.width, pic.height) == 1280
    assert call["max_tokens"] >= 1500
    assert '"description"' in call["system"]
    row = _row("bike.png")
    assert row["status"] == "done" and row["waiting_for"] == ""
    assert row["title"] == "A title" and "OSTERIA LUNA" in row["excerpt"]
    # Search goes through what is in the picture, and says why it matched.
    assert _found("osteria luna") == ["bike.png"]
    [item] = [fl._item(r, {SLUG: {"name": "Desk"}}, ["osteria"])
              for r in fl.get_store().search([SLUG], ["osteria"])]
    assert "OSTERIA LUNA" in item["snippet"]


async def test_picture_waits_while_vision_is_off_and_is_read_once_it_is_on(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=False)
    _picture(deck.workspace / "chart.jpg")
    await _scan()
    row = _row("chart.jpg")
    assert row["status"] == "skipped" and row["waiting_for"] == "vision"
    assert "vision is off" in row["summary"]
    assert deck.calls == []

    await _scan()                       # still off: nothing happens
    assert deck.calls == [] and _row("chart.jpg")["status"] == "skipped"

    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    await _scan()
    assert len(deck.calls) == 1 and _row("chart.jpg")["status"] == "done"


async def test_agents_own_model_looks_when_the_library_model_cant(deck: _State):
    deck.tier = dict(TEXT_ONLY_TIER)
    deck.agent_model(provider="gemini", model="gemini-2.5-flash")      # auto: Gemini sees
    (deck.data / SLUG / ".env").write_text("GEMINI_API_KEY=gm-agent-key\n")
    _picture(deck.workspace / "scan.webp", size=(400, 300))
    await _scan()
    [call] = deck.calls
    assert call["tier"] == {"provider": "gemini", "model": "gemini-2.5-flash", "base_url": "",
                            "api_key": "gm-agent-key"}
    assert call["images"][0].width == 400        # small pictures keep their size
    assert _row("scan.webp")["model"] == "gemini-2.5-flash"


async def test_a_refused_picture_goes_to_the_next_model_then_waits(deck: _State):
    deck.agent_model(provider="openai", model="qwen2.5-vl-7b", base_url="http://gw.local/v1", vision=True)
    deck.refuse = {"claude-haiku-4-5"}
    _picture(deck.workspace / "a.png", size=(50, 50))
    await _scan()
    assert [c["tier"]["model"] for c in deck.calls] == ["claude-haiku-4-5", "qwen2.5-vl-7b"]
    assert _row("a.png")["status"] == "done"

    deck.calls.clear()
    deck.refuse = {"claude-haiku-4-5", "qwen2.5-vl-7b"}
    _picture(deck.workspace / "b.png", size=(60, 60))
    await _scan()
    row = _row("b.png")
    assert row["status"] == "skipped" and row["waiting_for"] == "vision"
    assert "turned pictures down" in row["summary"]


async def test_a_failing_model_hands_the_picture_to_the_next(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    deck.broken = {"claude-haiku-4-5"}
    _picture(deck.workspace / "a.png", size=(50, 50))
    job = await _scan()
    assert [c["tier"]["model"] for c in deck.calls] == ["claude-haiku-4-5", "claude-sonnet-4-6"]
    assert _row("a.png")["status"] == "done" and job.failed == 0

    deck.calls.clear()
    deck.broken = {"claude-haiku-4-5", "claude-sonnet-4-6"}
    _picture(deck.workspace / "b.png", size=(60, 60))
    job = await _scan()
    row = _row("b.png")
    assert row["status"] == "error" and "claude-sonnet-4-6" in row["error"] and job.failed == 1


async def test_a_model_that_turns_a_picture_down_gets_no_more_that_scan(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    deck.refuse = {"claude-haiku-4-5"}
    for n in range(3):
        _picture(deck.workspace / f"p{n}.png", size=(40, 40))
    await _scan()
    sent = [c["tier"]["model"] for c in deck.calls if c["images"]]
    assert sent.count("claude-haiku-4-5") == 1 and sent.count("claude-sonnet-4-6") == 3
    assert all(_row(f"p{n}.png")["status"] == "done" for n in range(3))


async def test_a_text_only_ollama_model_is_asked_once_then_left_alone(deck: _State):
    """Ollama counts every model as seeing; a text-only one refuses the
    first picture. The refusal is remembered under the endpoint the provider
    really calls (a blank base_url is Ollama's default), so later scans don't
    send the pictures again."""
    deck.tier = {"provider": "ollama", "model": "qwen3:8b", "base_url": "", "api_key": ""}
    deck.agent_model(provider="ollama", model="qwen3:8b")
    deck.refuse = {"qwen3:8b"}
    for n in range(4):
        _picture(deck.workspace / f"p{n}.png", size=(40, 40))
    await _scan()
    assert len([c for c in deck.calls if c["images"]]) == 1
    for n in range(4):
        row = _row(f"p{n}.png")
        assert row["status"] == "skipped" and row["waiting_for"] == "vision"
        assert "turned pictures down" in row["summary"]
    await _scan()
    await _scan()
    assert len([c for c in deck.calls if c["images"]]) == 1
    # The agent's switch says vision is on, but the model turned pictures down:
    # the message says so instead of asking to turn vision on.
    deck.agent_model(provider="ollama", model="qwen3:8b", vision=True)
    _picture(deck.workspace / "new.png", size=(40, 40))
    await _scan()
    assert "turned pictures down" in _row("new.png")["summary"]
    assert len([c for c in deck.calls if c["images"]]) == 1


def test_models_are_named_as_their_provider_names_them():
    assert fl._vision_id("openai", "gpt-6-luna-none", "http://gw/v1/") == ("openai", "openai/gpt-6-luna", "http://gw/v1")
    assert fl._vision_id("xai", "grok-4.5", "") == ("xai", "xai/grok-4.5", "")
    assert fl._vision_id("claude", "claude-sonnet-4-6", "") == ("anthropic", "anthropic/claude-sonnet-4-6", "")
    assert fl._vision_id("ollama", "qwen3:8b-high", "")[1:] == ("qwen3:8b-high", "http://127.0.0.1:11434")


async def test_a_refusal_under_an_effort_suffixed_name_is_remembered(deck: _State):
    deck.tier = {"provider": "openai", "model": "qwen3-30b-high", "base_url": "http://gw.local/v1",
                 "api_key": "k", "vision": True}
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    deck.refuse = {"qwen3-30b-high", "claude-sonnet-4-6"}
    for n in range(2):
        _picture(deck.workspace / f"p{n}.png", size=(40, 40))
    for _ in range(3):
        await _scan()
    sent = [c["tier"]["model"] for c in deck.calls if c["images"]]
    assert sent == ["qwen3-30b-high", "claude-sonnet-4-6"]


async def test_a_failing_first_model_does_not_hold_pictures_in_line(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    deck.broken = {"claude-haiku-4-5"}
    deck.delay = 0.05
    for n in range(4):
        _picture(deck.workspace / f"p{n}.png", size=(40, 40))
    await _scan()
    assert deck.most_in_flight["claude-haiku-4-5"] > 1
    assert all(_row(f"p{n}.png")["status"] == "done" for n in range(4))


async def test_vision_on_with_a_text_only_runtime_says_so(deck: _State):
    deck.tier = dict(TEXT_ONLY_TIER)
    deck.agent_model(provider="claude-cli", model="sonnet", vision=True)
    _picture(deck.workspace / "p.png", size=(40, 40))
    await _scan()
    assert "takes text only" in _row("p.png")["summary"] and deck.calls == []


def test_a_16_bit_grey_picture_keeps_its_greys(tmp_path: Path):
    np = pytest.importorskip("numpy")
    ramp = np.tile(np.linspace(0, 65535, 400, dtype=np.uint16), (300, 1))
    Image.fromarray(ramp).save(tmp_path / "depth.png")
    pic = fl._encode_picture(tmp_path / "depth.png")
    out = Image.open(io.BytesIO(base64.b64decode(pic.data))).convert("L")
    assert 90 < ImageStat.Stat(out).mean[0] < 170          # a ramp, not a white sheet


def test_a_webp_has_a_lower_pixel_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fl, "_MAX_WEBP_PIXELS", 100)
    Image.new("RGB", (20, 20), "red").save(tmp_path / "p.webp")
    Image.new("RGB", (20, 20), "red").save(tmp_path / "p.png")
    with pytest.raises(ValueError, match="too many pixels"):
        fl._encode_picture(tmp_path / "p.webp")
    assert fl._encode_picture(tmp_path / "p.png").width == 20


async def test_a_model_that_doesnt_see_says_so(deck: _State):
    deck.agent_model(provider="openai", model="gpt-4-text-only", base_url="http://x.local/v1")
    _picture(deck.workspace / "p.png", size=(40, 40))
    await _scan()
    row = _row("p.png")
    assert row["waiting_for"] == "vision" and "doesn't see pictures" in row["summary"]


async def test_svg_and_icons_are_never_shown_to_a_model(deck: _State):
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    (deck.workspace / "logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Acme Rockets</text></svg>')
    _picture(deck.workspace / "favicon.ico", size=(32, 32))
    await _scan()
    assert all(not c["images"] for c in deck.calls)
    assert _row("favicon.ico")["status"] == "skipped"
    assert _row("logo.svg")["status"] == "done"          # read as text, as before


# ── Thinking ──────────────────────────────────────────────────────────


async def test_every_description_is_asked_with_thinking_off(deck: _State):
    # A thinking model counts its thinking against max_tokens: on a small
    # budget it can think through all of it and never write the JSON.
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    deck.soniox_key = "sx-key"
    (deck.workspace / "notes.md").write_text("Launch moved to March.\n")
    _picture(deck.workspace / "saved" / "bike.png")
    _sound(deck.workspace / "saved" / "standup.wav")
    await _scan()

    assert len(deck.calls) == 3
    assert [c["effort"] for c in deck.calls] == [reasoning.QUIET] * 3
    # …and only for the call: nothing leaks out of it.
    assert reasoning.requested_effort() is None


# ── Recordings ────────────────────────────────────────────────────────


async def test_recording_waits_for_soniox_then_is_transcribed_and_searchable(deck: _State):
    _sound(deck.workspace / "saved" / "standup.wav")
    await _scan()
    row = _row("standup.wav")
    assert row["status"] == "skipped" and row["waiting_for"] == "soniox"
    assert "connect Soniox" in row["summary"] and deck.transcribed == []

    deck.soniox_key = "sx-key"
    await _scan()
    [t] = deck.transcribed
    assert t["owner"] == "local" and t["mime"] == "audio/wav" and t["filename"] == "standup.wav"
    assert t["poll_max"] >= 600 and t["source"] == "library" and t["agent"] == SLUG
    assert t["audio"] == (deck.workspace / "saved" / "standup.wav").read_bytes()
    [call] = deck.calls
    assert call["images"] is None and "Project Halibut" in call["user"]
    assert "transcript" in call["user"]
    row = _row("standup.wav")
    assert row["status"] == "done" and row["excerpt"].startswith("We agreed")
    # What is said in it finds it, and the snippet shows where.
    assert _found("halibut march") == ["standup.wav"]
    item = fl._item(row, {SLUG: {"name": "Desk"}}, ["halibut"])
    assert "Project Halibut" in item["snippet"]


async def test_a_failed_summary_keeps_the_transcript_and_never_transcribes_again(deck: _State):
    deck.soniox_key = "sx-key"
    deck.broken = {"claude-haiku-4-5"}
    _sound(deck.workspace / "standup.wav")
    await _scan()
    row = _row("standup.wav")
    assert row["status"] == "error" and row["waiting_for"] == "summary"
    assert _found("halibut") == ["standup.wav"]          # searchable meanwhile
    assert len(deck.transcribed) == 1

    deck.broken = set()
    deck.soniox_key = ""                                  # not even needed for this
    await _scan(retry_errors=True)
    assert len(deck.transcribed) == 1
    row = _row("standup.wav")
    assert row["status"] == "done" and row["waiting_for"] == "" and "Project Halibut" in row["excerpt"]
    assert "Project Halibut" in deck.calls[-1]["user"]

    # Changed since: it is a new recording, transcribed afresh.
    deck.soniox_key = "sx-key"
    deck.broken = {"claude-haiku-4-5"}
    await _scan()                                         # nothing to do yet
    _sound(deck.workspace / "standup.wav", seconds=1.0)
    os.utime(deck.workspace / "standup.wav", (1, 1))
    await _scan()
    assert len(deck.transcribed) == 2


async def test_summarising_a_recording_again_reuses_its_transcript(deck: _State):
    deck.soniox_key = "sx-key"
    _sound(deck.workspace / "standup.wav")
    await _scan()
    assert len(deck.transcribed) == 1
    row = _row("standup.wav")
    item = await fl.library_resummarize(row["id"], user=USER)
    assert item["status"] == "done" and len(deck.transcribed) == 1
    assert "Project Halibut" in deck.calls[-1]["user"]
    # Changed on disk since the last walk: transcribed afresh, once, and
    # stored as the version on disk — a second click and the next scan reuse it.
    _sound(deck.workspace / "standup.wav", seconds=1.0)
    os.utime(deck.workspace / "standup.wav", (5, 5))
    await fl.library_resummarize(row["id"], user=USER)
    assert len(deck.transcribed) == 2
    assert _row("standup.wav")["fingerprint"] == fl._fingerprint(deck.workspace / "standup.wav")
    await fl.library_resummarize(row["id"], user=USER)
    await _scan()
    assert len(deck.transcribed) == 2


async def test_a_kept_transcript_of_an_older_version_is_not_reused(deck: _State):
    deck.soniox_key = "sx-key"
    deck.broken = {"claude-haiku-4-5"}
    _sound(deck.workspace / "memo.wav")
    await _scan()
    assert _row("memo.wav")["waiting_for"] == "summary"
    deck.broken = set()
    deck.transcript = "Second version: the venue moved to Split."
    _sound(deck.workspace / "memo.wav", seconds=1.0)
    os.utime(deck.workspace / "memo.wav", (5, 5))
    item = await fl.library_resummarize(_row("memo.wav")["id"], user=USER)
    assert item["status"] == "done" and len(deck.transcribed) == 2
    assert _row("memo.wav")["excerpt"].startswith("Second version")


async def test_a_recording_grown_since_the_walk_is_not_read_whole(deck: _State, monkeypatch):
    monkeypatch.setattr(fl, "_MAX_SOUND_UPLOAD_BYTES", 20_000)
    deck.soniox_key = "sx-key"
    path = _sound(deck.workspace / "memo.wav", seconds=0.1)       # ~3 KB at the walk
    fl.get_store().sync_agent(SLUG, fl.walk_workspace(SLUG))
    _sound(path, seconds=2.0)                                       # ~64 KB now
    row = _row("memo.wav")
    reach = fl.MediaReach(owner="local", tier=None, soniox=True)
    out = await fl._sound_of(row, path, reach)
    assert isinstance(out, dict) and out["waiting_for"] == "ffmpeg" and "too large" in out["summary"]


@pytest.mark.parametrize("filler", ["Straße ", "ﬁnal ", "İstanbul ", "ΣΑΣ ", "e\u0301te\u0301 "])
def test_snippet_points_at_the_match_after_letters_that_fold_longer(filler: str):
    text = filler * 120 + "the Halibut launch moves to March"
    row = {"filename": "a.wav", "rel_path": "a.wav", "title": "", "summary": "", "tags": "[]",
           "category": "", "excerpt": text}
    snippet = fl._snippet(row, ["halibut"])
    assert "Halibut" in snippet and snippet.endswith("March")
    lead = fl._snippet({**row, "excerpt": "Halibut " + text}, ["halibut"])
    assert lead.startswith("Halibut")


async def test_a_long_transcript_is_kept_whole_for_search(deck: _State):
    deck.soniox_key = "sx-key"
    deck.transcript = ("blah " * 3000) + "the secret word is periwinkle"
    _sound(deck.workspace / "long.wav")
    await _scan()
    assert len(deck.calls[0]["user"]) < 9000          # the model reads the start…
    assert _found("periwinkle") == ["long.wav"]       # …search reads it all


async def test_the_agents_own_spoken_replies_are_not_transcribed(deck: _State):
    deck.soniox_key = "sx-key"
    _sound(deck.workspace / "saved" / "media" / "s1" / "speech-20261022-101010-abc123.wav")
    await _scan()
    assert deck.transcribed == []
    assert "spoken reply" in _row("speech-20261022-101010-abc123.wav")["summary"]


async def test_no_speech_is_said_plainly(deck: _State):
    deck.soniox_key = "sx-key"
    deck.transcript = ""
    _sound(deck.workspace / "hum.wav")
    await _scan()
    row = _row("hum.wav")
    assert row["status"] == "skipped" and row["summary"] == "No speech in it." and row["waiting_for"] == ""
    assert deck.calls == []


async def test_soniox_refusing_the_key_stops_it_for_the_scan(deck: _State, monkeypatch):
    monkeypatch.setattr(fl, "_SOUND_CONCURRENCY", 1)
    deck.soniox_key = "sx-bad"
    deck.soniox_fail = SonioxError(401, "Soniox refused the API key")
    for n in range(3):
        _sound(deck.workspace / f"r{n}.wav")
    await _scan()
    assert len(deck.transcribed) == 1
    for n in range(3):
        row = _row(f"r{n}.wav")
        assert row["status"] == "skipped" and row["waiting_for"] == "soniox"
        assert "turned the key down" in row["summary"]
    # The turned-down key is still stored: nothing is sent again with it…
    await _scan()
    await _scan()
    assert len(deck.transcribed) == 1 and _row("r0.wav")["status"] == "skipped"
    # …until it is replaced (or checked again) and works.
    deck.soniox_verify, deck.soniox_fail = None, None
    await _scan()
    assert len(deck.transcribed) == 4 and _row("r0.wav")["status"] == "done"


async def test_soniox_failing_leaves_the_rest_for_the_next_scan(deck: _State, monkeypatch):
    monkeypatch.setattr(fl, "_SOUND_CONCURRENCY", 1)
    deck.soniox_key = "sx-key"
    deck.soniox_fail = SonioxError(0, "Soniox timed out after 900s")
    for n in range(5):
        _sound(deck.workspace / f"r{n}.wav")
    job = await _scan()
    assert len(deck.transcribed) == fl._MAX_SONIOX_FAILURES
    statuses = sorted(_row(f"r{n}.wav")["status"] for n in range(5))
    assert statuses == ["error"] * 3 + ["pending"] * 2
    assert "Soniox isn't answering" in job.message
    assert (job.done, job.failed) == (0, 3)
    # The model never failed, so nothing else stopped.
    assert "model keeps failing" not in job.message


async def test_a_file_soniox_cant_read_is_not_an_outage(deck: _State, monkeypatch):
    monkeypatch.setattr(fl, "_SOUND_CONCURRENCY", 1)
    deck.soniox_key = "sx-key"
    deck.soniox_fail = SonioxError(200, "Soniox couldn't process the audio: bad data")
    for n in range(5):
        _sound(deck.workspace / f"r{n}.wav")
    job = await _scan()
    assert len(deck.transcribed) == 5 and "isn't answering" not in job.message
    for n in range(5):
        row = _row(f"r{n}.wav")
        assert row["status"] == "skipped" and row["waiting_for"] == ""
        assert "couldn't transcribe" in row["summary"]
    await _scan(retry_errors=True)              # left alone until the file changes
    assert len(deck.transcribed) == 5


async def test_an_unreadable_recording_is_skipped_not_blamed_on_the_model(deck: _State):
    deck.soniox_key = "sx-key"
    paths = [_sound(deck.workspace / f"r{n}.wav") for n in range(5)]
    for p in paths:
        p.chmod(0)
    try:
        job = await _scan()
    finally:
        for p in paths:
            p.chmod(0o644)
    assert "model keeps failing" not in job.message and job.failed == 0
    assert all("Could not be read" in _row(f"r{n}.wav")["summary"] for n in range(5))


# ── Videos ────────────────────────────────────────────────────────────


async def test_video_without_ffmpeg_waits_unless_soniox_reads_it(deck: _State):
    deck.soniox_key = "sx-key"
    (deck.workspace / "clip.mov").write_bytes(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64)
    (deck.workspace / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    await _scan()
    mov = _row("clip.mov")
    assert mov["status"] == "skipped" and mov["waiting_for"] == "ffmpeg"
    assert "ffmpeg" in mov["summary"]
    [t] = deck.transcribed                     # the mp4 went whole
    assert t["mime"] == "video/mp4" and t["filename"] == "clip.mp4"
    assert _row("clip.mp4")["status"] == "done"


async def test_video_sound_is_taken_out_with_ffmpeg(deck: _State, monkeypatch):
    deck.soniox_key = "sx-key"
    deck.tools = {"ffmpeg": "/x/ffmpeg", "ffprobe": "/x/ffprobe"}
    taken: list[str] = []

    async def extract(ffmpeg, path, **kw):
        taken.append(path)
        return b"OggS-opus", "audio.ogg", "audio/ogg"

    async def streams(ffprobe, path):
        return 0 if path.name == "silent.mov" else 1

    async def duration(ffprobe, path):
        return 754.0

    monkeypatch.setattr(video_vision, "_extract_audio", extract)
    monkeypatch.setattr(video_vision, "_ffprobe_duration", duration)
    monkeypatch.setattr(fl, "_sound_streams", streams)
    (deck.workspace / "talk.mov").write_bytes(b"\x00" * 128)
    (deck.workspace / "silent.mov").write_bytes(b"\x00" * 128)
    await _scan()
    [t] = deck.transcribed
    assert t["audio"] == b"OggS-opus" and t["mime"] == "audio/ogg"
    assert taken == [str(deck.workspace / "talk.mov")]
    assert "a video, 12:34 long" in deck.calls[0]["user"]
    silent = _row("silent.mov")
    assert silent["status"] == "skipped" and silent["waiting_for"] == "" and "No sound" in silent["summary"]


async def test_a_file_ffmpeg_cant_read_is_not_sent_whole(deck: _State, monkeypatch):
    deck.soniox_key = "sx-key"
    deck.tools = {"ffmpeg": "/x/ffmpeg"}

    async def extract(ffmpeg, path, **kw):
        return b"", "audio.wav", "audio/wav"

    monkeypatch.setattr(video_vision, "_extract_audio", extract)
    (deck.workspace / "broken.mp4").write_bytes(b"\xa6\xf9\x7f(" * 64)
    await _scan()
    assert deck.transcribed == []
    row = _row("broken.mp4")
    assert row["status"] == "skipped" and "damaged" in row["summary"] and row["waiting_for"] == ""


async def test_a_video_waiting_for_ffmpeg_is_read_once_it_is_there(deck: _State, monkeypatch):
    deck.soniox_key = "sx-key"
    (deck.workspace / "clip.mkv").write_bytes(b"\x1aE\xdf\xa3" + b"\x00" * 64)
    await _scan()
    assert _row("clip.mkv")["waiting_for"] == "ffmpeg"

    async def extract(ffmpeg, path, **kw):
        return b"OggS", "audio.ogg", "audio/ogg"

    monkeypatch.setattr(video_vision, "_extract_audio", extract)
    deck.tools = {"ffmpeg": "/x/ffmpeg"}
    await _scan()
    assert _row("clip.mkv")["status"] == "done"


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
async def test_real_ffmpeg_takes_the_sound_out_and_sees_silent_video(deck: _State, tmp_path: Path):
    wav = _sound(tmp_path / "a.wav", seconds=2.0)
    silent = tmp_path / "silent.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-c:v", "mpeg4", str(silent)], check=True)
    reach = fl.MediaReach(owner="local", tier=None, soniox=True,
                          ffmpeg=shutil.which("ffmpeg"), ffprobe=shutil.which("ffprobe"))
    row = {"extension": ".wav", "filename": "a.wav", "rel_path": "a.wav", "size": wav.stat().st_size,
           "kind": "media"}
    audio, filename, mime, seconds = await fl._sound_of(row, wav, reach)
    assert audio and mime in ("audio/ogg", "audio/wav") and 1.5 < seconds < 2.5
    assert await fl._sound_streams(reach.ffprobe, silent) == 0
    out = await fl._sound_of({**row, "extension": ".mp4", "filename": "silent.mp4"}, silent, reach)
    assert isinstance(out, dict) and "No sound" in out["summary"]


# ── The library around them ───────────────────────────────────────────


async def test_text_files_are_summarised_as_before(deck: _State):
    (deck.workspace / "notes.md").write_text("# Plans\n\nShip the thing.\n")
    await _scan()
    [call] = deck.calls
    assert call["images"] is None and "Ship the thing" in call["user"]
    assert _row("notes.md")["status"] == "done"


async def test_search_knows_category_and_kind_names(deck: _State):
    deck.reply = {**deck.reply, "category": "media"}
    deck.agent_model(provider="anthropic", model="claude-sonnet-4-6", vision=True)
    _picture(deck.workspace / "p.png", size=(20, 20))
    deck.soniox_key = "sx-key"
    _sound(deck.workspace / "v.wav")
    await _scan()
    assert _found("images") == ["p.png", "v.wav"]       # "Images & media"
    assert "v.wav" in _found("audio video") and "p.png" not in _found("audio video")


def test_stills_cut_from_a_video_are_not_library_files(deck: _State):
    _picture(deck.workspace / "saved" / "video_frames" / "ab12cd34" / "frame_000.jpg", size=(10, 10))
    _picture(deck.workspace / "saved" / "kept.jpg", size=(10, 10))
    assert [f["filename"] for f in fl.walk_workspace(SLUG)] == ["kept.jpg"]


def test_new_sound_and_video_kinds_are_listed(deck: _State):
    for name in ("memo.m4a", "note.opus", "call.amr", "clip.mov", "film.mkv"):
        (deck.workspace / name).write_bytes(b"\x00" * 8)
    listed = {f["filename"]: f["kind"] for f in fl.walk_workspace(SLUG)}
    assert listed == dict.fromkeys(("memo.m4a", "note.opus", "call.amr", "clip.mov", "film.mkv"), "media")


def test_a_library_from_before_reads_the_pictures_and_recordings_it_set_aside(tmp_path: Path):
    db = tmp_path / "file_library.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE library_files (
            id TEXT PRIMARY KEY, agent_slug TEXT NOT NULL, rel_path TEXT NOT NULL, filename TEXT NOT NULL,
            extension TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'other', size INTEGER NOT NULL DEFAULT 0,
            modified REAL NOT NULL DEFAULT 0, fingerprint TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending', title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '', category TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
            excerpt TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
            tokens INTEGER NOT NULL DEFAULT 0, summarized_at REAL, added_at REAL NOT NULL DEFAULT 0);
        INSERT INTO library_files (id, agent_slug, rel_path, filename, extension, kind, status, summary)
        VALUES ('1', 'a', 'p.png', 'p.png', '.png', 'image', 'skipped', 'Image — nothing to read, so not summarised.'),
               ('2', 'a', 'v.mp4', 'v.mp4', '.mp4', 'media', 'skipped', 'Too large to read here.'),
               ('3', 'a', 'x.zip', 'x.zip', '.zip', 'archive', 'skipped', 'Archive — not summarised.'),
               ('4', 'a', 'n.md', 'n.md', '.md', 'document', 'done', 'Notes.');
    """)
    conn.commit()
    conn.close()
    store = fl.LibraryStore(db)
    got = {r["id"]: (r["status"], r["summary"], r["waiting_for"]) for r in store.search(["a"], [])}
    assert got == {"1": ("pending", "", ""), "2": ("pending", "", ""),
                   "3": ("skipped", "Archive — not summarised.", ""), "4": ("done", "Notes.", "")}
    # Opening it again changes nothing.
    store.save_result("1", "", status="skipped", summary="Not described.", waiting_for="vision")
    again = fl.LibraryStore(db)
    assert again.get("1")["status"] == "skipped"


async def test_call_model_sends_the_picture_and_notices_a_refusal(monkeypatch):
    """A server that won't take pictures is retried without them by the
    provider, which then turns ``sees_images`` off — that is the refusal."""
    import captain_claw.llm as llm
    from captain_claw.flight_deck import basna_routes

    sent: list = []

    class Provider:
        def __init__(self, refuse: bool) -> None:
            self.refuse = refuse

        async def complete(self, messages, **kw):
            sent.append(messages)
            if self.refuse and messages[-1].images:
                self.sees_images = False
            return types.SimpleNamespace(content='{"title": "t"}', usage={"total_tokens": 42})

    refuse = {"on": False}
    monkeypatch.setattr(llm, "create_provider", lambda **kw: Provider(refuse["on"]))
    monkeypatch.setattr(basna_routes, "tier_api_key", lambda tier: "k")
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    pic = vision.EncodedImage(path="p.png", mime="image/jpeg", data="AAAA", width=1, height=1)

    content, tokens = await fl.call_model(LIBRARY_TIER, "sys", "user", "", images=[pic])
    assert (content, tokens) == ('{"title": "t"}', 42)
    assert sent[-1][-1].images == [pic]

    refuse["on"] = True
    with pytest.raises(fl.PictureRefusedError):
        await fl.call_model(LIBRARY_TIER, "sys", "user", "", images=[pic])
    # Text alone is never taken for a refusal.
    assert (await fl.call_model(LIBRARY_TIER, "sys", "user", ""))[1] == 42
    assert sent[-1][-1].images is None
