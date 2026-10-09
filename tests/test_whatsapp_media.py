"""WhatsApp replies carry files and pictures: photos as images, MP4 as video,
audio as audio, the rest as documents — sent with the tool, and what a
WhatsApp turn produced delivered at its end."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from captain_claw.config import get_config
from captain_claw.session import Session
from captain_claw.tools import whatsapp_send_file as wa


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.delenv("WHATSAPP_ALLOWED_WAIDS", raising=False)


@pytest.fixture
def graph(monkeypatch):
    """Record Cloud API calls instead of making them."""
    calls: list[dict] = []

    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, files=None, json=None):
            calls.append({"url": url, "files": files, "json": json})
            return _Resp({"id": f"media-{len(calls)}"})

    monkeypatch.setattr(wa.httpx, "AsyncClient", _Client)
    return calls


def _box(kind: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _mp4(*formats: bytes, mdat: bytes = b"", entries: int = 1) -> bytes:
    """A minimal MP4: ftyp, a moov with one stsd per track, then mdat."""
    tracks = b"".join(_box(b"stsd", b"\x00" * 4 + entries.to_bytes(4, "big") + _box(f, b"\x00" * 8))
                      for f in formats)
    return _box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _box(b"moov", tracks) + _box(b"mdat", mdat or b"x" * 16)


def _png(depth: int = 8, colour: int = 6) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (64).to_bytes(4, "big") * 2
            + bytes([depth, colour]) + b"x" * 32)


def _jpeg(components: int = 3, precision: int = 8) -> bytes:
    """SOI, an APP0 segment, then a baseline SOF0 frame header."""
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = (b"\xff\xc0" + (8 + 3 * components).to_bytes(2, "big") + bytes([precision])
            + (64).to_bytes(2, "big") * 2 + bytes([components]) + b"\x01\x11\x00" * components)
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


JPEG = _jpeg()
PNG = _png()
MP4_H264 = _mp4(b"avc1", b"mp4a")
MP4_HEVC = _mp4(b"hvc1", b"mp4a", mdat=b"..avc1..")         # 'avc1' only in the media data
MP4_OPUS_AUDIO = _mp4(b"avc1", b"Opus")
M4A_AAC = _mp4(b"mp4a")
M4A_ALAC = _mp4(b"alac")
OGG_OPUS = b"OggS" + b"\x00" * 24 + b"OpusHead" + b"x" * 32
OGG_VORBIS = b"OggS" + b"\x00" * 24 + b"\x01vorbis" + b"x" * 32


@pytest.mark.parametrize("name,size,head,kind,mime", [
    ("photo.jpg", 1000, JPEG, "image", "image/jpeg"),
    ("print.jpg", 1000, _jpeg(components=4), "document", "image/jpeg"),     # CMYK
    ("gray.jpg", 1000, _jpeg(components=1), "document", "image/jpeg"),
    ("chart.png", 1000, PNG, "image", "image/png"),
    ("poster.png", 1000, JPEG, "image", "image/jpeg"),             # the bytes decide
    ("broken.png", 1000, b"not an image", "document", "image/png"),
    ("deep.png", 1000, _png(depth=16), "document", "image/png"),  # 8-bit only
    ("huge.png", 5_000_001, PNG, "document", "image/png"),         # over Meta's 5 MB
    ("clip.mp4", 1000, MP4_H264, "video", "video/mp4"),
    ("iphone.mp4", 1000, MP4_HEVC, "document", "video/mp4"),       # only H.264 plays
    ("opus.mp4", 1000, MP4_OPUS_AUDIO, "document", "video/mp4"),   # and only AAC audio
    ("dual.mp4", 1000, _mp4(b"avc1", b"mp4a", b"mp4a"), "document", "video/mp4"),  # one audio stream
    ("multi.mp4", 1000, _mp4(b"avc1", b"mp4a", entries=2), "document", "video/mp4"),
    ("song.m4a", 1000, M4A_AAC, "audio", "audio/mp4"),
    ("lossless.m4a", 1000, M4A_ALAC, "document", "audio/mp4"),
    ("fake.mp3", 1000, b"RIFF....WAVE", "document", "audio/mpeg"),
    ("old.3gp", 1000, b"x", "document", "video/3gpp"),
    ("note.mp3", 1000, b"ID3", "audio", "audio/mpeg"),
    ("voice.ogg", 1000, OGG_OPUS, "audio", "audio/ogg"),
    ("song.ogg", 1000, OGG_VORBIS, "document", "audio/ogg"),       # only Opus in Ogg plays
    ("sticker.webp", 1000, b"RIFF", "document", "image/webp"),
    ("report.pdf", 1000, b"%PDF", "document", "application/pdf"),
])
def test_media_kind(name, size, head, kind, mime):
    assert wa.media_kind(name, size, head) == (kind, mime)


@pytest.mark.parametrize("name,blob,kind", [("p.jpg", JPEG, "image"), ("c.mp4", MP4_H264, "video"),
                                            ("n.mp3", b"ID3" + b"x" * 50, "audio"),
                                            ("r.pdf", b"%PDF" + b"x" * 50, "document")])
async def test_each_kind_goes_out_as_that_message_type(tmp_path, creds, graph, name, blob, kind):
    f = tmp_path / name
    f.write_bytes(blob)
    ok, got, err = await wa.send_whatsapp_media("385911", f, caption="For you")
    assert ok and got == kind and not err
    upload, message = graph
    assert upload["url"].endswith("/123/media")
    body = message["json"]
    assert body["type"] == kind and body["to"] == "385911"
    assert body[kind]["id"] == "media-1"
    assert ("caption" in body[kind]) == (kind != "audio")         # audio carries none
    assert ("filename" in body[kind]) == (kind == "document")


async def test_the_original_file_can_go_as_a_document(tmp_path, creds, graph):
    f = tmp_path / "scan.png"
    f.write_bytes(PNG)
    ok, kind, _err = await wa.send_whatsapp_media("385911", f, as_document=True)
    assert ok and kind == "document" and graph[-1]["json"]["type"] == "document"


async def test_a_refused_photo_goes_again_as_a_document(tmp_path, creds, monkeypatch):
    calls: list = []

    async def fake_upload(token, pid, blob, name, mime):
        calls.append(("upload", mime))
        return "m1", ""

    async def fake_send(token, pid, to, media_id, name, caption, kind="document"):
        calls.append(("send", kind))
        if kind == "image":
            return False, 'send rejected (400): {"error":{"code":131053,"message":"media"}}'
        return True, ""

    monkeypatch.setattr(wa.WhatsAppSendFileTool, "_meta_upload", staticmethod(fake_upload))
    monkeypatch.setattr(wa.WhatsAppSendFileTool, "_meta_send", staticmethod(fake_send))
    f = tmp_path / "chart.png"
    f.write_bytes(PNG)
    ok, kind, _err = await wa.send_whatsapp_media("385911", f)
    assert ok and kind == "document"
    assert calls == [("upload", "image/png"), ("send", "image"), ("upload", "image/png"), ("send", "document")]


async def test_other_refusals_are_not_retried_as_documents(tmp_path, creds, monkeypatch):
    calls: list = []

    async def fake_upload(token, pid, blob, name, mime):
        calls.append("upload")
        return "m1", ""

    async def fake_send(token, pid, to, media_id, name, caption, kind="document"):
        calls.append("send")
        return False, 'send rejected (401): {"error":{"code":190,"message":"token expired"}}'

    monkeypatch.setattr(wa.WhatsAppSendFileTool, "_meta_upload", staticmethod(fake_upload))
    monkeypatch.setattr(wa.WhatsAppSendFileTool, "_meta_send", staticmethod(fake_send))
    f = tmp_path / "chart.png"
    f.write_bytes(PNG)
    ok, kind, err = await wa.send_whatsapp_media("385911", f)
    assert not ok and kind == "image" and "token expired" in err and calls == ["upload", "send"]


async def test_the_allowlist_still_guards_media(tmp_path, creds, graph, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "385900")
    f = tmp_path / "p.jpg"
    f.write_bytes(JPEG)
    ok, _kind, err = await wa.send_whatsapp_media("385911", f)
    assert not ok and "WHATSAPP_ALLOWED_WAIDS" in err and graph == []


async def test_the_tool_sends_a_photo_as_a_photo_and_remembers_it(tmp_path, creds, graph):
    saved = tmp_path / "saved"
    (saved / "media" / "s1").mkdir(parents=True)
    photo = saved / "media" / "s1" / "sunset.jpg"
    photo.write_bytes(JPEG)
    session = Session(id="s1", name="d")
    session.metadata["whatsapp_waid"] = "385911"
    agent = types.SimpleNamespace()
    res = await wa.WhatsAppSendFileTool().execute(
        action="send", filename="sunset", _saved_base_path=str(saved), _session=session, _agent=agent)
    assert res.success and "as a photo" in res.content
    assert graph[-1]["json"]["type"] == "image"
    assert wa.sent_this_turn(agent) == {str(photo.resolve())}
    res = await wa.WhatsAppSendFileTool().execute(
        action="send", filename="sunset", send_as="document", _saved_base_path=str(saved),
        _session=session, _agent=agent)
    assert res.success and "as a document" in res.content


# ── End of a WhatsApp turn ─────────────────────────────────────────────


def _turn(tmp_path):
    """A WhatsApp turn that generated a picture, took a photo, spoke a reply,
    downloaded a Drive file, wrote documents (one with a bare path, which
    lands in tmp/) and looked up a script."""
    import os
    import time

    saved = tmp_path / "saved"
    for sub in ("media/s1", "showcase/s1", "downloads/s1", "scripts/s1", "tmp/s1", "showcase/other"):
        (saved / sub).mkdir(parents=True)
    old = saved / "showcase" / "s1" / "last-week.pdf"
    old.write_bytes(b"%PDF")
    stale = time.time() - 3600
    os.utime(old, (stale, stale))
    started = time.time() - 1
    files = {
        "generated.png": saved / "media" / "s1" / "generated.png",
        "screenshot.png": saved / "media" / "s1" / "screenshot.png",
        "camera.jpg": saved / "media" / "s1" / "camera.jpg",
        "summary.mp3": saved / "media" / "s1" / "summary.mp3",
        "Q4 plan.md": saved / "downloads" / "s1" / "Q4 plan.md",
        "final-report.docx": saved / "showcase" / "s1" / "final-report.docx",
        "report.docx": saved / "showcase" / "s1" / "report.docx",        # a draft, not named
        "notes.txt": saved / "tmp" / "s1" / "notes.txt",                # write(path="notes.txt")
        "draft.csv": saved / "showcase" / "s1" / "draft.csv",
        "notes.md": saved / "scripts" / "s1" / "notes.md",
        "theirs.pdf": saved / "showcase" / "other" / "theirs.pdf",      # another session's file
    }
    for path in files.values():
        path.write_bytes(b"x" * 10)
    s = Session(id="s1", name="d")
    s.add_message("user", "make me a poster and the report", origin="human", channel="whatsapp")
    for tool, name in (("image_gen", "generated.png"), ("browser", "screenshot.png"),
                       ("termux", "camera.jpg"), ("pocket_tts", "summary.mp3"),
                       ("google_drive", "Q4 plan.md"), ("scripts", "notes.md")):
        s.add_message("tool", f"Done.\n  Path: {files[name]}", tool_name=tool, tool_call_id=tool)
    reply = ("Here's the poster and your report: final-report.docx, plus my notes.txt. I read "
             "Q4 plan.md for it. The old last-week.pdf is unchanged; theirs.pdf and the script "
             "notes.md helped.")
    s.add_message("assistant", reply, origin="model")
    agent = types.SimpleNamespace(session=s, last_turn_start_idx=0,
                                  tools=types.SimpleNamespace(get_saved_base_path=lambda create=False: saved),
                                  _current_session_slug=lambda: "s1")
    return agent, files, reply, started


@pytest.fixture
def outbox(monkeypatch):
    sent: list = []
    notes: list = []

    async def fake_send(to, path, caption="", as_document=False):
        sent.append((to, Path(path).name))
        return True, "image", ""

    async def fake_text(to, body):
        notes.append((to, body))
        return True, ""

    monkeypatch.setattr(wa, "send_whatsapp_media", fake_send)
    monkeypatch.setattr(wa, "send_whatsapp_text", fake_text)
    return sent, notes


async def test_a_whatsapp_turn_delivers_what_it_made_for_the_user(tmp_path, creds, outbox):
    agent, files, reply, started = _turn(tmp_path)
    wa.mark_sent(agent, files["camera.jpg"])                     # the agent already sent this one
    names = await wa.deliver_turn_media(agent, "385911", 0, reply=reply, turn_started_at=started)
    # The picture, the spoken summary, and the files this session wrote that
    # the reply names by their whole name (the tmp/ note too) — not the photo
    # already sent, the browser screenshot, the draft whose name sits inside
    # another, the Drive input, the script, the old PDF or another session's.
    assert names[:2] == ["generated.png", "summary.mp3"]
    assert sorted(names[2:]) == ["final-report.docx", "notes.txt"]
    assert outbox[1] == []


async def test_failures_and_the_cap_are_told_in_the_chat(tmp_path, creds, monkeypatch, outbox):
    agent, _files, reply, started = _turn(tmp_path)

    async def failing(to, path, caption="", as_document=False):
        return (False, "image", "boom") if Path(path).name == "generated.png" else (True, "audio", "")

    monkeypatch.setattr(wa, "send_whatsapp_media", failing)
    monkeypatch.setattr(get_config().whatsapp, "max_media_per_turn", 2)
    names = await wa.deliver_turn_media(agent, "385911", 0, reply=reply, turn_started_at=started)
    assert names == ["camera.jpg"]
    assert outbox[1] == [("385911", "Couldn't send generated.png here. 3 more file(s) are in the agent's files.")]


async def test_nothing_named_means_no_scan(tmp_path, creds, outbox, monkeypatch):
    agent, _files, _reply, started = _turn(tmp_path)
    agent.tools = None                                           # a scan would fail loudly
    assert wa._named_new_files(agent, "All done — sent the poster.", started) == []


async def test_delivery_respects_its_switch(tmp_path, creds, monkeypatch, outbox):
    agent, _files, reply, started = _turn(tmp_path)
    monkeypatch.setattr(get_config().whatsapp, "auto_send_media", False)
    assert await wa.deliver_turn_media(agent, "385911", 0, reply=reply, turn_started_at=started) == []
    monkeypatch.setattr(get_config().whatsapp, "auto_send_media", True)
    assert await wa.deliver_turn_media(agent, "", 0, reply=reply, turn_started_at=started) == []


async def test_no_credentials_no_delivery(tmp_path, monkeypatch, outbox):
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    agent, _files, reply, started = _turn(tmp_path)
    assert await wa.deliver_turn_media(agent, "385911", 0, reply=reply, turn_started_at=started) == []


async def test_a_delivery_still_running_never_touches_the_next_turn(tmp_path, creds, monkeypatch):
    agent, _files, reply, started = _turn(tmp_path)
    release = asyncio.Event()

    async def slow(to, path, caption="", as_document=False):
        await release.wait()
        return True, "image", ""

    monkeypatch.setattr(wa, "send_whatsapp_media", slow)
    task = asyncio.create_task(wa.deliver_turn_media(agent, "385911", 0, reply=reply, turn_started_at=started))
    for _ in range(3):
        await asyncio.sleep(0)
    wa.reset_turn(agent)                                          # the next turn starts
    release.set()
    await task
    assert wa.sent_this_turn(agent) == set()


# ── Wiring in the chat handler ─────────────────────────────────────────


def test_the_reply_chat_comes_from_the_waid_or_the_origin():
    from captain_claw.web.chat_handler import _reply_waid

    assert _reply_waid("+385911", None) == "385911"
    assert _reply_waid(None, {"kind": "whatsapp", "address": "385922"}) == "385922"
    assert _reply_waid(None, {"kind": "telegram", "address": "x"}) == ""
    assert _reply_waid(None, None) == ""
    assert _reply_waid(None, None, "385933") == "385933"          # a scheduled job's chat


class _Agent:
    def __init__(self):
        self.session = Session(id="s1", name="d")
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self.provider = object()
        self.tools = types.SimpleNamespace(set_session_policy=lambda *a: None,
                                           clear_session_policy=lambda *a: None)

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    def _current_session_slug(self):
        return "s1"

    async def complete(self, content):
        assert wa.sent_this_turn(self) == set()                  # reset for the turn
        self.reply_to_seen = wa.reply_to(self)
        self.session.add_message("assistant", "Here you go.", origin="model")
        return "Here you go."


class _Server:
    LANE_MAIN = "A"

    def __init__(self, agent):
        self.agent = agent
        self._busy = True
        self._active_task = None
        self._orchestrator = None
        self.sent: list = []

    def _broadcast(self, msg, exclude=None):
        self.sent.append(msg)

    def _session_info(self, agent=None):
        return {}


@pytest.mark.parametrize("waid,expected", [("385911", [("385911", 0)]), ("", [])])
async def test_a_whatsapp_turn_hands_its_files_to_delivery(monkeypatch, waid, expected):
    from captain_claw.web import chat_handler

    async def _noop(*a, **k):
        return None

    for target in ("captain_claw.reflections.maybe_auto_reflect",
                   "captain_claw.insights.maybe_extract_insights",
                   "captain_claw.nervous_system.maybe_dream",
                   "captain_claw.conversation_topics.maybe_classify_topics",
                   "captain_claw.intentions_generator.maybe_auto_propose"):
        monkeypatch.setattr(target, _noop)
    monkeypatch.setattr(get_config().ui, "next_steps", False)
    handed: list = []

    async def fake_deliver(agent, to, start, reply="", turn_started_at=0.0, already_sent=None):
        handed.append((to, start))
        assert reply == "Here you go." and turn_started_at > 0
        return []

    monkeypatch.setattr(wa, "deliver_turn_media", fake_deliver)
    agent = _Agent()
    wa.mark_sent(agent, Path("/tmp/stale"))
    await chat_handler._run_agent(_Server(agent), None, agent, "make a poster", None, lane="A",
                                  no_flow=True, reply_waid=waid)
    for _ in range(5):
        await asyncio.sleep(0)
    assert handed == expected
    assert agent.reply_to_seen == waid                           # the tool's default recipient
    assert wa.reply_to(agent) == ""                              # …for that turn only



def test_a_whatsapp_turn_is_told_how_files_reach_the_chat(monkeypatch, creds):
    from captain_claw.agent import Agent

    agent = Agent.__new__(Agent)
    agent.session = Session(id="s1", name="d")
    agent._turn_origin = ("human", "", "whatsapp")
    # A WhatsApp turn whose delivery isn't set up (no chat to answer): no promise.
    assert agent._whatsapp_delivery_note().endswith("use whatsapp_send_file.")
    wa.reset_turn(agent, "385911")
    assert "names it by its file name" in agent._whatsapp_delivery_note()
    monkeypatch.setattr(get_config().whatsapp, "auto_send_media", False)
    assert agent._whatsapp_delivery_note().endswith("use whatsapp_send_file.")
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN")
    assert "can't send files or pictures" in agent._whatsapp_delivery_note()
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    agent._turn_origin = ("cron", "", None)                        # a scheduled job…
    wa.reset_turn(agent, automated=True)
    assert agent._whatsapp_delivery_note() == ""
    wa.reset_turn(agent, "385911")                                 # …that delivers to WhatsApp
    assert agent._whatsapp_delivery_note().startswith("Your reply goes to a WhatsApp chat")


async def test_a_scheduled_job_names_its_chat_only_when_pushes_may_reach_it(monkeypatch):
    from captain_claw.flight_deck import fd_scheduler, whatsapp_bridge

    seen: dict = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return "brief"

    async def fake_deliver(kind, target, text):
        return True, "ok"

    muted: set = set()
    monkeypatch.setattr(fd_scheduler, "run_prompt_and_capture", fake_run)
    monkeypatch.setattr(fd_scheduler, "_deliver", fake_deliver)
    monkeypatch.setattr(fd_scheduler, "resolve_agent_by_slug", lambda slug, auth: ("127.0.0.1", 24001, ""))
    monkeypatch.setattr(whatsapp_bridge, "_allowed_waids", lambda: {"385911"})
    monkeypatch.setattr(whatsapp_bridge, "is_push_muted", lambda waid: waid in muted)
    job = {"agent_slug": "a", "prompt": "brief", "delivery_kind": "whatsapp",
           "delivery_target": "385911", "ignore_quiet_hours": True}
    await fd_scheduler.execute_job(job, force=True)
    assert seen["whatsapp_media_to"] == "385911"
    muted.add("385911")                                            # /mute
    await fd_scheduler.execute_job(job, force=True)
    assert seen["whatsapp_media_to"] == ""
    await fd_scheduler.execute_job({**job, "delivery_target": "385999"}, force=True)
    assert seen["whatsapp_media_to"] == ""                         # not allowlisted
    await fd_scheduler.execute_job({**job, "delivery_kind": "telegram", "delivery_target": "1"}, force=True)
    assert seen["whatsapp_media_to"] == ""


async def test_the_media_target_rides_only_on_automated_frames(monkeypatch):
    from captain_claw.web import ws_handler

    seen: list = []

    async def fake_chat(server, ws, content, **kwargs):
        seen.append(kwargs.get("whatsapp_media_to"))

    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", fake_chat)
    server = types.SimpleNamespace(agent=types.SimpleNamespace(plan_mode_auto=False))
    await ws_handler.handle_ws_message(server, types.SimpleNamespace(), {
        "type": "chat", "content": "brief", "whatsapp_media_to": "385911",
        "automation": {"kind": "fd_scheduler"}})
    assert seen == ["385911"]


def test_process_agents_get_the_shared_keys(monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "fd-token")
    env = {"WHATSAPP_PHONE_NUMBER_ID": "own-id", "SONIOX_API_KEY": ""}
    server._share_fd_secrets(env)
    assert env["WHATSAPP_ACCESS_TOKEN"] == "fd-token"           # filled from Flight Deck
    assert env["WHATSAPP_PHONE_NUMBER_ID"] == "own-id"          # the agent's own value stays


async def test_a_file_written_with_a_bare_name_is_delivered_when_named(tmp_path, creds, outbox):
    import time

    from captain_claw.tools.write import WriteTool

    saved = tmp_path / "saved"
    saved.mkdir()
    started = time.time() - 1
    target = WriteTool._normalize_under_saved("report.md", saved, "s1")    # -> tmp/s1/report.md
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# Report")
    s = Session(id="s1", name="d")
    agent = types.SimpleNamespace(session=s, last_turn_start_idx=0,
                                  tools=types.SimpleNamespace(get_saved_base_path=lambda create=False: saved),
                                  _current_session_slug=lambda: "s1")
    names = await wa.deliver_turn_media(agent, "385911", 0, reply="Here is report.md.", turn_started_at=started)
    assert names == ["report.md"] and target.parent.name == "s1" and target.parent.parent.name == "tmp"



async def test_an_automated_turn_without_a_chat_names_its_recipient(tmp_path, creds, graph):
    saved = tmp_path / "saved"
    (saved / "media" / "s1").mkdir(parents=True)
    (saved / "media" / "s1" / "chart.jpg").write_bytes(JPEG)
    session = Session(id="s1", name="d")
    session.metadata["whatsapp_waid"] = "385911"                 # the owner's chat, not this job's
    agent = types.SimpleNamespace(session=session)
    wa.reset_turn(agent, "", automated=True)                       # FD withheld the job's chat (muted)
    res = await wa.WhatsAppSendFileTool().execute(
        action="send", filename="chart", _saved_base_path=str(saved), _session=session, _agent=agent)
    assert not res.success and "pass 'to'" in res.error and graph == []
    res = await wa.WhatsAppSendFileTool().execute(
        action="send", filename="chart", to="385922", _saved_base_path=str(saved), _session=session,
        _agent=agent)
    assert res.success and graph[-1]["json"]["to"] == "385922"
    wa.reset_turn(agent, "")                                        # an interactive web turn
    res = await wa.WhatsAppSendFileTool().execute(
        action="send", filename="chart", _saved_base_path=str(saved), _session=session, _agent=agent)
    assert res.success and graph[-1]["json"]["to"] == "385911"   # the session's chat, as before
