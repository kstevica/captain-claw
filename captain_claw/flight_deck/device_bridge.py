"""Device bridge — a handheld voice client (M5Stack StickS3 / StopWatch) talks to an agent.

One WebSocket per device: ``/device/ws?token=<cc_pat_…>&device=<id>``
(optional ``agent=<slug|name|port>``, ``lang=<tts language>``, ``speak=0``,
``voice=male|female|<Soniox voice>``, ``hw=sticks3&fw=0.8.0`` for updates,
``sw``/``sh`` = screen size for pictures, default 240×135,
``round=1`` = a round screen: pictures fit inside the circle,
``img=chunks`` = pictures may span several frames, so they can be bigger,
``paper=1`` = a colour e-paper screen: pictures come at their own shape on
white, for the device to lay out with the text).

The device records while its button is held and streams the audio here;
Flight Deck transcribes it (Soniox), sends the words to one of the owner's
agents over the glasses channel bus (surface ``"device"``), and returns the
reply as text, spoken audio (Soniox TTS) and any pictures/audio the turn
made — so the stick stays a thin client with no audio codecs or HTTP stack.

Device → Flight Deck
  text  {"type":"hello", ...}                    optional, informational
  text  {"type":"rec_start","rate":16000}        starts a recording (stops playback)
  bin   raw PCM s16le mono at that rate          while recording
  text  {"type":"rec_end","ctx":{…}}             → transcribe → agent; ctx = the device's
                                                 readings as it sent (battery, Wi-Fi, motion,
                                                 temperature, clock), optional — see
                                                 device_context_line()
  text  {"type":"rec_cancel"}                    drop the recording
  text  {"type":"text","text":"…","ctx":{…}}     send typed words instead
  text  {"type":"stop"}                          stop playback, clear the queue
  text  {"type":"replay"}                        play the last answer's sounds again (its speech
                                                 is synthesized afresh) → audio, or {"type":"replay_none"}
  text  {"type":"agents"} / {"type":"select","agent":"<slug>"}
  text  {"type":"prefs","voice":"male|female|<name>","speak":bool}   change voice / speaking live
  text  {"type":"ping","t":<device ms>}          echoed back (round-trip time)
  text  {"type":"notes_begin","count":n}         offline voice notes (recorded while the deck was
  text  {"type":"note_start","id","rate","seconds","recorded","epoch"}   out of reach), then bin PCM,
  text  {"type":"note_end","id"} … {"type":"notes_end","ctx":{…}}        → transcribed into ONE agent
                                                 turn marked as offline notes → {"type":"notes_done",
                                                 "ids":[…]} (the device deletes those) or
                                                 {"type":"notes_failed","text":…}; notes_cancel drops
                                                 a batch cut short
  text  {"type":"say","text":"…"}                speak a short line aside (an e-paper's menu row):
                                                 audio_start with "aside":true; never over an
                                                 answer, never kept for a replay
  text  {"type":"update_check"}                  → update offer or {"type":"update_none"}

Flight Deck → device
  text  {"type":"bound","agent":name,"slug":slug,"linked":bool} | {"type":"agents","items":[…],"current":slug}
  text  {"type":"state","state":"connected|agent_offline|transcribing|thinking|ready"}
  text  {"type":"transcript","text":"…"}         what it heard
  text  {"type":"status","text":"…"}             progress narration
  text  {"type":"activity","text":"…","tool":"…"} what the agent is doing (tools, LLM calls)
  text  {"type":"reply","text":"…","proactive":bool}
  text  {"type":"audio_start","rate":24000[,"aside":true]} … {"type":"audio_end"}
  bin   0x01 + PCM s16le mono                    audio chunks between those two
  text  {"type":"image","name":"…","w":240,"h":135,"size":n} then bin 0x02 + baseline JPEG
        (img=chunks: bin 0x03 + part … while more follow, then 0x02 + the last part)
  text  {"type":"error","text":"…"} | {"type":"pong","t":<echoed>}
  text  {"type":"update","version","size","sha256","sig","notes","url"} | {"type":"update_none"}
  text  {"type":"replay_none"}                   nothing to replay (the device plays what it kept)

Auth: a personal access token (Profile → MCP agent access) names the owner;
the device only ever reaches that owner's agents. With Flight Deck auth off
(single-user desktop) no token is needed.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import json
import logging
import math
import re
import secrets
import shutil
import struct
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from captain_claw.flight_deck import glasses_bridge as gb
from captain_claw.flight_deck import usage_quota

log = logging.getLogger(__name__)

router = APIRouter()

# Hidden context for the device's agent binding (first message only — the
# agent keeps it per surface and renders it on device turns).
_DEVICE_SYSTEM_CONTEXT = (
    "[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
    "The user is talking to you through a tiny handheld voice device: they spoke, "
    "and your reply is read aloud and shown on a {screen} screen.\n"
    "Output rules for every reply on this session:\n"
    "  - Lead with the answer. No preamble, no recap.\n"
    "  - Keep it short and speakable: 1–3 short sentences, or a short list of "
    "items when the task produced several (never drop results to stay short).\n"
    "  - Plain text only: no tables, code blocks, headings or URLs unless asked.\n"
    "  - Pictures you generate with image_gen are shown on the device's screen and "
    "audio you make (audio speak, pocket_tts) is played on it, automatically.\n"
    "  - The transcript of their speech may contain recognition errors; read it "
    "charitably.\n"
    "  - A message may open with a [Context: Device readings …] line: the device's "
    "battery, Wi-Fi, sensors and clock as it sent the message. Use them when they "
    "matter (the user asks about them, or the battery is nearly empty); never recite "
    "them otherwise.\n"
    "  - Do not mention the device (unless they ask about it), this context, or these rules.\n"
    "Apply these rules to this and every subsequent message until told otherwise.\n"
    "---\n"
    "USER MESSAGE:\n"
)

# Binary frame kinds, Flight Deck → device (first byte of each frame).
BIN_AUDIO = 0x01
BIN_JPEG = 0x02        # a picture, or the last part of one
BIN_JPEG_PART = 0x03   # a part of a picture, more follow (devices that ask: img=chunks)
BIN_BG_PART = 0x04     # a part of the background picture, more follow
BIN_BG = 0x05          # the background picture's last part

PCM_CHUNK = 4096                 # bytes per audio frame sent to the device
TTS_RATE = gb._STREAM_SAMPLE_RATE
MAX_REC_SECONDS = 120            # a held button can't fill memory
MIN_REC_SECONDS = 0.35
MAX_SPOKEN_CHARS = 1500          # long replies are shown in full, spoken in part
MEDIA_WINDOW_S = 180.0           # turn_media trails the reply; accept it this long
AUDIO_FILE_MAX_S = 300
MAX_JPEG_BYTES = 14000           # the device's WebSocket library takes ≤15 KB frames
MAX_CHUNKED_JPEG_BYTES = 120_000  # a picture sent in parts (img=chunks)
MAX_SAY_CHARS = 120              # a spoken menu row, not a reply
MAX_NOTES_PER_BATCH = 20         # offline voice notes taken in one upload
MAX_NOTE_SECONDS = 120
# A sentence of the answer as it streams: a full stop, ! ? or … followed by
# whitespace (so "3.5" and "e.g." hold), or a line break. The last one comes
# with the complete answer.
_SENTENCE_END = re.compile(r"[.!?…](?=\s)|\n")
# "New topic" said with nothing after it: no answer follows the new session.
NEW_TOPIC_REPLY = "New topic — go ahead."
TURN_KEEP_S = 600.0              # a turn (its answer) waits this long for its device to come back
RESUME_GRACE_S = 1.5             # a reconnect's talk-from-sleep recording starts within this
HEARD_AFTER_S = 2.0              # a device message this long after an answer: it got there

# The device turn per channel, kept across reconnects (weak Wi-Fi drops the
# link mid-turn): open from the send until "ready", and its answer until the
# device is heard from after it. A reconnect carries on with it.
_turns: dict[str, dict] = {}
_conn_serials = itertools.count(1)

# Offline voice notes already handed to the agent, per device channel: a
# batch sent again (its notes_done lost with the link) is acknowledged, not
# delivered twice.
_delivered_notes: dict[str, list[str]] = {}
_notes_locks: dict[str, asyncio.Lock] = {}

OFFLINE_NOTES_CONTEXT = (
    "Offline voice notes: the user recorded these on the device while it had no connection to you, "
    "each at the time shown. They weren't waiting for an answer then. Handle them now, in order — do "
    "what each one asks, or keep it as a note when it is one — and answer once, briefly."
)

# The last answer's sounds per device channel, for a replay — kept here, not
# on the connection, so it survives a reconnect (a dropped link cuts an
# answer short just like a stop does). The device replays a complete answer
# from its own memory; it asks for this one only when part never arrived.
_answers: dict[str, list[tuple[str, Any]]] = {}


# ── Small pure helpers (tested) ───────────────────────────────────────


def wav_from_pcm(pcm: bytes, rate: int, channels: int = 1) -> bytes:
    """Wrap raw little-endian int16 PCM in a WAV header."""
    byte_rate = rate * channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, channels * 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_URL = re.compile(r"https?://\S+")
_MD_MARKS = re.compile(r"(\*\*|__|`+|~~|^#{1,6}\s*|^>\s*)", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_MD_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)


def speakable(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """A reply as it should be read aloud: markdown marks, links and URLs
    out, list items as sentences, cut at a sentence end near *limit*."""
    out = _MD_LINK.sub(r"\1", str(text or ""))
    out = _MD_URL.sub("", out)
    out = _MD_TABLE_RULE.sub("", out)
    out = _MD_MARKS.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = out.replace("|", ", ").replace("*", "")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    out = " ".join(ln if ln[-1] in ".!?:;," else ln + "." for ln in lines)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if len(out) > limit:
        cut = out[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        out = cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0] + "…"
    return out


def screen_text(text: str, has_dj: bool = False) -> str:
    """A reply as the tiny screen should show it: markdown marks out (the
    device renders plain text), structure kept. *has_dj*: the device's font
    has Đ (the StopWatch's does; the StickS3's has only đ)."""
    out = _MD_LINK.sub(r"\1", str(text or ""))
    out = _MD_TABLE_RULE.sub("", out)
    out = re.sub(r"(\*\*|__|`+|~~)", "", out)
    out = re.sub(r"^#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s*[-*]\s+", "• ", out, flags=re.MULTILINE)
    if not has_dj:
        out = out.replace("Đ", "Dj")  # the StickS3's font has đ but not Đ
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# Emoji and symbol marks the device font can't draw (agent activity uses a few).
_NO_GLYPH = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200D]")
# Runtime statuses that are turn bookkeeping, not activity worth showing.
_QUIET_STATUSES = {"", "thinking", "ready", "idle", "done", "completed", "agent_connected",
                   "agent_reconnecting", "subscribed"}


def activity_text(text: str, limit: int = 120) -> str:
    """An activity line as the device shows it: one line, no emoji, "…"."""
    out = _NO_GLYPH.sub("", str(text or "")).replace("\n", " ")
    out = re.sub(r"\.\.\.$", "…", re.sub(r"\s{2,}", " ", out).strip())
    return out[:limit]


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def cue_alone(text: str) -> bool:
    """*text* is only a new-topic cue ("New topic.", "Nova tema"): the agent
    starts a new session and has nothing more to answer."""
    try:
        from captain_claw.web.slash_commands import rotation_cue

        return rotation_cue(text) == ""
    except Exception:
        return False


def round_cover(width: int, height: int, aspect: float) -> tuple[int, int]:
    """A picture of *aspect* (w/h) that fills the circle a round width×height
    screen shows: its shorter side the diameter, the longer one at most
    twice that (a panorama is cropped to that). The device shows the middle,
    or all of it scaled down on a tap."""
    d = min(width, height)
    if aspect >= 1:
        return min(2 * d, max(1, round(d * aspect))), d
    return d, min(2 * d, max(1, round(d / aspect)))


def fit_jpeg(blob: bytes, width: int, height: int, quality: int = 85,
             max_bytes: int = MAX_JPEG_BYTES, round_screen: bool = False,
             paper: bool = False) -> bytes:
    """Any picture as a baseline JPEG of exactly width×height (letterboxed on
    black) — what the device's decoder takes — no larger than *max_bytes*
    (the device's WebSocket frame limit; quality steps down to fit). On a
    round screen it fills the circle instead (round_cover): the device shows
    the middle, or all of it on a tap. For e-paper (*paper*) it keeps its own
    shape within width×height, on white, and the device lays it out with the
    text."""
    from PIL import Image, ImageOps

    bg_rgb = (255, 255, 255) if paper else (0, 0, 0)
    im = Image.open(io.BytesIO(blob))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, bg_rgb + (255,))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if round_screen:
        canvas = ImageOps.fit(im, round_cover(width, height, im.width / im.height), Image.LANCZOS)
    elif paper:
        canvas = ImageOps.contain(im, (width, height), Image.LANCZOS)
    else:
        im = ImageOps.contain(im, (width, height), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), bg_rgb)
        canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
    while True:
        out = io.BytesIO()
        canvas.save(out, "JPEG", quality=quality, progressive=False, optimize=False)
        if out.tell() <= max_bytes or quality <= 20:
            return out.getvalue()
        quality -= 10


def jpeg_frames(jpeg: bytes, part: int = MAX_JPEG_BYTES) -> list[tuple[int, bytes]]:
    """A picture as device frames for img=chunks: (0x03, part) while more
    follow, then (0x02, the last part). One that fits a frame is a single
    (0x02, bytes) — exactly what a device without img=chunks gets."""
    parts = [jpeg[i:i + part] for i in range(0, len(jpeg), part)] or [b""]
    return [(BIN_JPEG_PART, p) for p in parts[:-1]] + [(BIN_JPEG, parts[-1])]


def voice_name(choice: str) -> str:
    """The Soniox voice for a device's choice: "male"/"female" map to
    ``DEVICE_VOICE_MALE`` / ``DEVICE_VOICE_FEMALE`` (Adrian / Emma), anything
    else is taken as a voice name; "" = the deck's ``SONIOX_TTS_VOICE``."""
    import os

    c = str(choice or "").strip()
    if c.lower() in ("male", "m"):
        return os.environ.get("DEVICE_VOICE_MALE", "").strip() or "Adrian"
    if c.lower() in ("female", "f"):
        return os.environ.get("DEVICE_VOICE_FEMALE", "").strip() or "Emma"
    return re.sub(r"[^A-Za-z0-9_-]", "", c)[:32]


_DEVICE_MODELS = {"sticks3": "M5Stack StickS3", "stopwatch": "M5Stack StopWatch",
                  "papercolor": "M5Stack PaperColor"}
_CTX_TEXT = re.compile(r"[^\w .,:+\-/()'’@#&°%]")


def _num(value: Any, lo: float, hi: float) -> float | None:
    """A reading as a number within [lo, hi], else None (missing, junk)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) and lo <= v <= hi else None


def _ctx_text(value: Any, limit: int = 40) -> str:
    return _CTX_TEXT.sub("", str(value or "")).strip()[:limit]


def device_context_line(ctx: Any, hw: str = "") -> str:
    """One line for the agent from the readings a device sent with a turn::

        {"battery": {"percent": 64, "mv": 3920, "charging": false, "usb": false},
         "wifi": {"ssid": "Home", "rssi": -61},
         "motion": {"orientation": "face up", "movement": "still"},
         "environment": {"temperature_c": 23.1, "humidity": 41.2},
         "local_time": "2026-10-19 14:32 (+0200)", "uptime_s": 3600}

    Every part is optional; anything missing or malformed is left out, and
    "" when nothing usable came."""
    if not isinstance(ctx, dict):
        return ""
    parts: list[str] = []
    bat = ctx.get("battery") if isinstance(ctx.get("battery"), dict) else {}
    pct = _num(bat.get("percent"), 0, 100)
    volts = _num(bat.get("mv"), 2000, 5500)
    if pct is not None:
        power = ("charging" if bat.get("charging") is True
                 else "on USB power" if bat.get("usb") is True else "on battery")
        parts.append(f"battery {pct:.0f}%" + (f" ({volts / 1000:.2f} V)" if volts else "") + f", {power}")
    wifi = ctx.get("wifi") if isinstance(ctx.get("wifi"), dict) else {}
    rssi = _num(wifi.get("rssi"), -120, 0)
    if rssi is not None:
        quality = "strong" if rssi > -55 else "good" if rssi > -67 else "fair" if rssi > -75 else "weak"
        ssid = _ctx_text(wifi.get("ssid"), 32)
        parts.append(f"Wi-Fi {ssid + ' ' if ssid else ''}{rssi:.0f} dBm ({quality})")
    motion = ctx.get("motion") if isinstance(ctx.get("motion"), dict) else {}
    pose = ", ".join(t for t in (_ctx_text(motion.get("orientation"), 24),
                                    _ctx_text(motion.get("movement"), 24)) if t)
    if pose:
        parts.append(pose)
    env = ctx.get("environment") if isinstance(ctx.get("environment"), dict) else {}
    temp = _num(env.get("temperature_c"), -40, 85)
    hum = _num(env.get("humidity"), 0, 100)
    if temp is not None or hum is not None:
        reading = " and ".join(t for t in (f"{temp:.1f} °C" if temp is not None else "",
                                           f"{hum:.0f}% humidity" if hum is not None else "") if t)
        parts.append(f"{reading} (sensor inside the device; reads a little warm while it's busy)")
    clock = _ctx_text(ctx.get("local_time"), 40)
    if clock:
        parts.append(f"device clock {clock}")
    if not parts:
        return ""
    model = _DEVICE_MODELS.get(hw, "")
    return f"Device readings{' (' + model + ')' if model else ''} as it sent this: " + "; ".join(parts) + "."


def offline_notes_message(notes: list[dict]) -> str:
    """Transcribed offline voice notes as one message, oldest first: each
    numbered, with when it was recorded and how long it was."""
    lines = []
    for i, note in enumerate(notes, 1):
        when = _ctx_text(note.get("recorded"), 40) or "time unknown"
        secs = int(_num(note.get("seconds"), 0, 3600) or 0)
        lines.append(f"{i}. ({when}, {secs // 60}:{secs % 60:02d}) {str(note.get('text', '')).strip()}")
    head = ("Voice notes I recorded while you were out of reach:" if len(notes) > 1
            else "A voice note I recorded while you were out of reach:")
    return head + "\n" + "\n".join(lines)


def _note_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(value or ""))[:40]


def _int_param(value: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except Exception:
        return default


def _picture_options(q) -> dict:
    """How pictures are made for a device, from its query (sw, sh, round,
    img, paper); with none of them, the StickS3's 240×135 in one frame."""
    opts = {
        "screen": (_int_param(q.get("sw", ""), 240, 32, 800), _int_param(q.get("sh", ""), 135, 32, 800)),
        "round_screen": q.get("round", "") in ("1", "true", "on"),
        "chunked": q.get("img", "") == "chunks",
    }
    if q.get("paper", "") in ("1", "true", "on"):
        opts["paper"] = True
    return opts


# ── Auth + agent choice ───────────────────────────────────────────────


async def _device_user(token: str) -> dict | None:
    """The owner a device token names (a personal access token), or None."""
    from captain_claw.flight_deck import auth as A

    if not A._fd_auth_enabled():
        return dict(A._LOCAL_USER)
    token = (token or "").strip()
    if not token.startswith(A.PAT_PREFIX):
        return None
    db = A.get_db()
    row = await db.get_pat_by_hash(A.hash_token(token))
    if not row:
        return None
    user = await db.get_user_by_id(row["user_id"])
    if not user:
        return None
    try:
        await db.touch_pat(row["id"])
    except Exception:
        pass
    return user


def _running_agents(user_id: str) -> list[dict]:
    from captain_claw.flight_deck.mcp_server_routes import _user_agents

    return [
        a for a in _user_agents(user_id)
        if a.get("port") and str(a.get("status", "")).lower() in ("running", "up")
    ]


def _pick_agent(user_id: str, selector: str) -> dict | None:
    """The agent a selector names (slug, name or port) among the owner's
    running agents, else their first running one."""
    agents = _running_agents(user_id)
    sel = (selector or "").strip()
    if sel:
        for a in agents:
            if str(a.get("port")) == sel or a.get("slug") == sel or a.get("name") == sel:
                return a
        low = sel.lower()
        for a in agents:
            if low in str(a.get("name", "")).lower() or low in str(a.get("slug", "")).lower():
                return a
    return agents[0] if agents else None


# ── One connected device ──────────────────────────────────────────────


class _DeviceConn:
    def __init__(self, ws: WebSocket, user: dict, device_id: str, *, lang: str,
                 speak: bool, screen: tuple[int, int], voice: str = "",
                 round_screen: bool = False, chunked: bool = False,
                 paper: bool = False, hw: str = "", fw: str = "") -> None:
        self.ws = ws
        self.hw = hw  # device kind + firmware version, for over-the-air updates
        self.fw = fw
        self.user = user
        self.user_id = str(user.get("id", ""))
        self.device_id = device_id
        self.lang = lang
        self.speak = speak
        self.voice = voice_name(voice)
        self.screen = screen
        self.round_screen = round_screen  # pictures fit inside the circle
        self.chunked = chunked            # pictures may span several frames
        self.paper = paper                # colour e-paper: pictures at their own shape
        self._image_lock = asyncio.Lock()  # one picture's parts at a time
        self.ch: gb._ChannelState | None = None
        self.agent: dict | None = None
        self.serial = next(_conn_serials)
        self.closed = False  # the link is gone; a turn still open keeps listening
        self.opened_at = self.last_heard_at = time.monotonic()
        self.bg_id: str | None = None  # the background it has ("" none); None: it can't show one
        # The answer as the model writes it (agent_delta): shown as it grows
        # and spoken sentence by sentence as they complete.
        self.partial = ""
        self.spoken_upto = 0     # chars of partial handed to the voice
        self.spoken_total = 0    # chars spoken of this answer so far
        self.streamed = False    # some of this answer went out as it came
        self.last_part = ""      # the text last shown as a part
        self.speech_parts = 0    # sounds sent for this answer: the first gets the cue, the rest carry on
        self._send_lock = asyncio.Lock()
        self.recording = False
        self.rec_rate = 16000
        self.rec = bytearray()
        # A device turn is open from our send until the agent says "ready".
        self.pending = False
        self.cue_alone = False  # the open turn is a bare "new topic"
        self.last_reply_at = 0.0
        self._last_activity = ""
        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._audio_task: asyncio.Task | None = None
        self._audio_worker: asyncio.Task | None = None
        self._say_task: asyncio.Task | None = None
        # Offline voice notes arriving: the one now (note_start … note_end), the batch so far.
        self.note: dict | None = None
        self.notes: list[dict] = []
        self._bg: set[asyncio.Task] = set()

    # ── over-the-air updates ──
    async def offer_update(self, reply_none: bool = False) -> bool:
        """Offer the newest build of this device's kind when it's newer than
        what it runs (device_firmware.py); "update_none" on request."""
        from captain_claw.flight_deck.device_firmware import offer_for

        offer = offer_for(self.hw, self.fw)
        if offer:
            await self.send_json(offer)
        elif reply_none:
            await self.send_json({"type": "update_none"})
        return offer is not None

    # ── wire ──
    async def send_json(self, obj: dict) -> None:
        if self.closed:
            return
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj, default=str))

    async def send_bin(self, kind: int, payload: bytes) -> None:
        if self.closed:
            return
        async with self._send_lock:
            await self.ws.send_bytes(bytes([kind]) + payload)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ── binding ──
    @property
    def channel_id(self) -> str:
        return f"device:{self.user_id}:{self.device_id}"

    async def bind(self, selector: str) -> bool:
        agent = _pick_agent(self.user_id, selector)
        if agent is None:
            await self.send_json({"type": "error", "text": "No running agent. Start one in Flight Deck."})
            return False
        ch = await gb._get_or_create_channel(self.channel_id)
        if self.ch is not ch:
            if self.ch is not None and self.on_bus in self.ch.callback_subscribers:
                self.ch.callback_subscribers.remove(self.on_bus)
            self.ch = ch
        # One device socket per channel listens (a reconnect replaces the old).
        ch.callback_subscribers[:] = [cb for cb in ch.callback_subscribers
                                      if getattr(cb, "__self__", None) is None
                                      or not isinstance(cb.__self__, _DeviceConn)]
        ch.callback_subscribers.append(self.on_bus)
        await gb._ensure_agent_binding(ch, str(agent.get("host") or "localhost"), int(agent["port"]))
        self.agent = agent
        await self.send_json({"type": "bound", "agent": agent.get("name", ""), "slug": agent.get("slug", ""),
                              "linked": ch.agent_ws is not None})
        return True

    def unbind(self) -> None:
        if self.ch is not None and self.on_bus in self.ch.callback_subscribers:
            self.ch.callback_subscribers.remove(self.on_bus)

    # ── agent bus → device ──
    async def on_bus(self, ev: dict) -> None:
        etype = ev.get("type")
        if etype == "agent":
            proactive = bool(ev.get("proactive")) and not ev.get("fd_delivers")
            if not (self.pending or proactive):
                return  # a turn from another surface on the same agent
            turn = _turns.get(self.channel_id) if self.pending else None
            text = str(ev.get("text", "") or "")
            closes = False
            if ev.get("command") == "/new":
                # A spoken "new topic, …": the device hears the answer that
                # follows, not "New session created". Said alone, nothing
                # follows — a short line closes the turn instead.
                if not (self.pending and self.cue_alone):
                    return
                self.cue_alone = False
                self._end_turn()
                text, closes = NEW_TOPIC_REPLY, True
            if not text.strip():
                return
            await self._show_reply(text, proactive, turn)
            if closes:
                await self.send_json({"type": "state", "state": "ready"})
        elif etype == "narration":
            if self.pending and ev.get("text"):
                await self.send_json({"type": "status", "text": str(ev["text"])[:200]})
        elif etype == "agent_delta":
            if self.pending and not self.cue_alone and not self.closed:
                await self._on_delta(str(ev.get("text", "") or ""))
        elif etype == "activity":
            if self.pending and ev.get("phase") != "done":
                tool = str(ev.get("tool", "") or "")
                if tool and tool != "progress":
                    await self._reset_stream()  # what the model wrote before a tool call isn't the answer
                await self._send_activity(str(ev.get("text", "")), tool)
        elif etype == "status":
            status = str(ev.get("status", ""))
            # "Using <tool>..." duplicates the tool's activity line (which says more).
            tool_start = status.startswith("Using ") and status.endswith("...")
            if self.pending and status.strip().lower() not in _QUIET_STATUSES and not tool_start:
                await self._send_activity(status, "")
            if status == "thinking" and self.pending:
                await self.send_json({"type": "state", "state": "thinking"})
            elif status == "ready" and self.pending:
                self._end_turn()
                await self.send_json({"type": "state", "state": "ready"})
            elif status == "agent_connected":
                await self.send_json({"type": "state", "state": "connected"})
            elif status == "agent_reconnecting":
                await self.send_json({"type": "state", "state": "agent_offline"})
                await self.send_json({"type": "status", "text": "Reconnecting to agent…"})
        elif etype == "error":
            if self.pending:
                self._end_turn()
                text = ("The agent is busy — try again in a moment." if ev.get("retryable")
                        else str(ev.get("text", "") or "Agent error"))
                await self.send_json({"type": "error", "text": text[:300]})
        elif etype == "media":
            if self.closed or not (self.pending or time.monotonic() - self.last_reply_at < MEDIA_WINDOW_S):
                return
            for item in ev.get("items") or []:
                kind = str(item.get("kind", ""))
                if kind == "image":
                    self._spawn(self._send_image(item))
                elif kind == "audio":
                    _answers.setdefault(self.channel_id, []).append(("file", item))
                    await self._audio_q.put(("file", item))

    async def _send_activity(self, text: str, tool: str) -> None:
        line = activity_text(text)
        if not line or line == self._last_activity:
            return
        self._last_activity = line
        await self.send_json({"type": "activity", "text": line, "tool": tool})

    async def _show_reply(self, text: str, proactive: bool, turn: dict | None = None) -> None:
        """An answer on the screen and spoken (what of it wasn't already, as it
        streamed); a turn's answer is also kept until the device is heard from
        after it (a dropped link loses it)."""
        if turn is not None:
            turn.update(answer={"text": text, "proactive": proactive}, via=self.serial,
                        sent_at=time.monotonic(), heard=False)
        if self.closed:
            self._forget_stream()
            return  # the reconnect brings it
        self.last_reply_at = time.monotonic()
        if self.streamed:
            # Spoken as it came: only what follows the last sentence read is left.
            read = self.partial[:self.spoken_upto]
            tail = text[len(read):] if text.startswith(read) else text[_common_prefix(text, read):]
            spoken = speakable(tail) if self.speak else ""
            total = self.spoken_total + len(spoken)
        else:
            spoken = speakable(text) if self.speak else ""
            total = len(spoken)
            _answers[self.channel_id] = []  # a new answer
            self.speech_parts = 0
        # Round screens (the StopWatch) and e-paper use Inter, which has Đ.
        # "spoken": how much of it is read aloud, so the watch can follow the
        # voice through the text (0: nothing is).
        await self.send_json({"type": "reply", "text": screen_text(text, has_dj=self.round_screen or self.paper),
                              "proactive": proactive, "spoken": total})
        if spoken:
            _answers.setdefault(self.channel_id, []).append(("tts", spoken))
            await self._audio_q.put(("tts", spoken))
        if total:
            await self._audio_q.put(("end", None))  # speech_end, once the voice has all gone out
        self._forget_stream()

    # ── the answer as it streams ──
    async def _on_delta(self, text: str) -> None:
        self.partial += text
        while True:
            m = _SENTENCE_END.search(self.partial, self.spoken_upto)
            if not m:
                return
            sentence = self.partial[self.spoken_upto:m.end()]
            self.spoken_upto = m.end()
            await self._speak_part(sentence)

    async def _speak_part(self, sentence: str) -> None:
        """A sentence that just completed: on the screen (the answer so far)
        and read aloud, straight after the one before."""
        if not self.streamed:
            self.streamed = True
            self.speech_parts = 0
            self.last_part = ""
            _answers[self.channel_id] = []  # a new answer
            self.last_reply_at = time.monotonic()
        shown = screen_text(self.partial[:self.spoken_upto], has_dj=self.round_screen or self.paper)
        if shown and shown != self.last_part:
            self.last_part = shown
            await self.send_json({"type": "reply_part", "text": shown})
        spoken = speakable(sentence) if self.speak else ""
        if spoken:
            _answers.setdefault(self.channel_id, []).append(("tts", spoken))
            self.spoken_total += len(spoken)
            await self._audio_q.put(("tts", spoken))

    async def _reset_stream(self) -> None:
        """Text that came before a tool call wasn't the answer: forgotten, and
        the device starts the answer's sound afresh (what of it was already
        sent plays out)."""
        if self.streamed:
            while not self._audio_q.empty():  # not yet sent: dropped
                self._audio_q.get_nowait()
            await self.send_json({"type": "speech_reset"})
        self._forget_stream()
        self.speech_parts = 0

    def _forget_stream(self) -> None:
        self.partial = ""
        self.spoken_upto = self.spoken_total = 0
        self.streamed = False
        self.last_part = ""

    async def stop_agent(self) -> None:
        """The device's stop button while the agent works: the agent stops
        after its current step (its "ready" ends the turn); what was queued
        to be spoken is dropped."""
        ch = self.ch
        if not (self.pending and ch is not None and ch.agent_ws is not None):
            return
        await self.stop_audio()
        self._forget_stream()
        try:
            await ch.agent_ws.send(json.dumps({"type": "cancel"}))
        except Exception as exc:
            await self.send_json({"type": "error", "text": f"Couldn't stop: {exc}"[:200]})
            return
        await self.send_json({"type": "status", "text": "Stopping…"})

    # ── a turn across reconnects ──
    def _end_turn(self) -> None:
        self.pending = False
        turn = _turns.get(self.channel_id)
        if turn is not None:
            turn["open"] = False
        if self.closed:
            self.unbind()  # nothing more to keep for the reconnect

    def _kept_turn(self) -> dict | None:
        turn = _turns.get(self.channel_id)
        if turn and time.monotonic() - max(turn["at"], turn["sent_at"]) <= TURN_KEEP_S:
            return turn
        return None

    def take_over(self) -> None:
        """A device back after a dropped link: the turn still running on its
        channel is this connection's now, so its answer comes here."""
        turn = self._kept_turn()
        if turn and turn["open"]:
            self.pending = True

    async def resume(self) -> None:
        """After take_over: the answer the device may have missed comes
        again, or a turn still running shows as thinking again."""
        turn = self._kept_turn()
        if turn is None:
            return
        await asyncio.sleep(RESUME_GRACE_S)  # woken by talking: that recording goes first
        if self.recording or _turns.get(self.channel_id) is not turn:
            return  # they're asking something new
        answer = turn.get("answer")
        if answer and not turn["heard"] and turn["via"] != self.serial:
            await self._show_reply(answer["text"], answer["proactive"], turn)
            if not turn["open"]:
                await self.send_json({"type": "state", "state": "ready"})
        elif turn["open"] and not answer:
            await self.send_json({"type": "state", "state": "thinking"})

    def heard(self) -> None:
        """Something came from the device: the answer sent to it a moment
        before got there."""
        self.last_heard_at = time.monotonic()
        turn = _turns.get(self.channel_id)
        if (turn and turn.get("answer") and turn["via"] == self.serial
                and time.monotonic() - turn["sent_at"] >= HEARD_AFTER_S):
            turn["heard"] = True

    # ── device → agent ──
    def system_context(self) -> str:
        w, h = self.screen
        if self.paper:
            screen = (f"4-inch colour e-paper ({w}×{h} px; it takes about 20 s to redraw, "
                      "so your reply is heard first and stays readable on it after)")
        elif self.round_screen:
            screen = f"small round {min(w, h)} px"
        else:
            screen = f"{w}×{h} pixel"
        return _DEVICE_SYSTEM_CONTEXT.replace("{screen}", screen)

    async def send_to_agent(self, text: str, ctx: Any = None, extra_notes: list[str] | None = None) -> bool:
        """The words to the agent as a device turn, with the device's
        readings (and any *extra_notes*) as context lines; False when it
        didn't go (the device has been told why)."""
        ch = self.ch
        if ch is None:
            await self.send_json({"type": "error", "text": "No agent bound."})
            return False
        for _ in range(50):  # the first send after binding waits for the link
            if ch.agent_ws is not None:
                break
            await asyncio.sleep(0.1)
        if ch.agent_ws is None:
            await self.send_json({"type": "error", "text": "Agent isn't reachable."})
            return False
        allowed, quota = await usage_quota.admit_message(self.user_id, text)
        if not allowed:
            await self.send_json({"type": "error", "text": usage_quota.short_limit_text(quota)})
            await self.send_json({"type": "state", "state": "ready"})
            return False
        async with ch.send_lock:
            content = text if ch.context_sent else self.system_context() + text
            payload = {
                "type": "chat", "content": content, "surface": "device",
                "client_msg_id": "dev_" + secrets.token_hex(6),
            }
            readings = device_context_line(ctx, self.hw)
            notes = ([readings] if readings else []) + list(extra_notes or [])
            if notes:
                payload["context_notes"] = notes
            try:
                await ch.agent_ws.send(json.dumps(payload))
                ch.context_sent = True
            except Exception as exc:
                await self.send_json({"type": "error", "text": f"Send failed: {exc}"})
                return False
        self.pending = True
        self._forget_stream()  # a new answer
        self.speech_parts = 0
        self.cue_alone = cue_alone(text)
        _turns[self.channel_id] = {"at": time.monotonic(), "open": True, "answer": None,
                                   "via": 0, "sent_at": 0.0, "heard": False}
        self._last_activity = ""
        try:
            await self.send_json({"type": "state", "state": "thinking"})
        except Exception:
            pass  # the device went away: the agent has the turn all the same
        return True

    async def _note_text(self, note: dict) -> tuple[str, str]:
        """(text, why it failed): ("", "") is a note with nothing said in it."""
        from captain_claw.flight_deck import soniox_routes
        from captain_claw.flight_deck.whatsapp_bridge import _stt_language_hints

        try:
            out = await soniox_routes.transcribe_for(
                self.user_id or None, wav_from_pcm(bytes(note["pcm"]), note["rate"]), mime="audio/wav",
                filename="note.wav", language_hints=_stt_language_hints(), source="device",
            )
        except (soniox_routes.SonioxNotConnected, soniox_routes.SonioxError) as exc:
            return "", str(exc) or "transcription failed"
        return (out.text or "").strip(), ""

    async def deliver_notes(self, batch: list[dict], ctx: Any = None) -> None:
        """Offline voice notes → one turn for the agent, marked as recorded
        offline. notes_done names the notes the device may delete: the ones
        the agent got, the silent ones, and any delivered before (a batch sent
        again). A note that couldn't be transcribed stays on the device."""
        lock = _notes_locks.setdefault(self.channel_id, asyncio.Lock())
        async with lock:  # one batch at a time per device, even across reconnects
            delivered = _delivered_notes.setdefault(self.channel_id, [])
            fresh = [n for n in batch if n["id"] not in delivered]
            if fresh and await usage_quota.spent(self.user_id) is not None:
                await self.send_json({"type": "notes_failed", "text": "Over today's limit."})
                return  # don't transcribe what can't be sent
            done = [n["id"] for n in batch if n["id"] in delivered]
            heard: list[dict] = []
            failed = ""
            for note in fresh:
                if len(note["pcm"]) < MIN_REC_SECONDS * 2 * note["rate"]:
                    done.append(note["id"])  # nothing in it
                    continue
                text, err = await self._note_text(note)
                if text:
                    heard.append({**note, "text": text})
                elif err:
                    failed = err  # kept on the device for a later try
                else:
                    done.append(note["id"])  # nothing said
            if heard:
                # Shielded: once the agent has the turn, the ids are kept as
                # delivered even if this connection is torn down meanwhile.
                sent = await asyncio.shield(self._send_notes_turn(heard, ctx, delivered))
                if not sent:
                    await self.send_json({"type": "notes_failed", "text": "The agent didn't get them."})
                    return
                done += [n["id"] for n in heard]
            delivered.extend(i for i in done if i not in delivered)
            del delivered[:-200]
            await self.send_json({"type": "notes_done", "ids": done, "heard": len(heard)})
            if failed:
                await self.send_json({"type": "notes_failed", "text": f"Couldn't transcribe some: {failed}"[:200]})

    async def _send_notes_turn(self, heard: list[dict], ctx: Any, delivered: list[str]) -> bool:
        said = " · ".join("“" + n["text"] + "”" for n in heard)
        shown = said if len(heard) == 1 else f"{len(heard)} offline notes: {said}"
        try:
            await self.send_json({"type": "transcript",
                                  "text": screen_text(shown[:300], has_dj=self.round_screen or self.paper)})
        except Exception:
            pass  # the device is gone; the agent can still have them
        if not await self.send_to_agent(offline_notes_message(heard), ctx, extra_notes=[OFFLINE_NOTES_CONTEXT]):
            return False
        delivered.extend(n["id"] for n in heard if n["id"] not in delivered)
        return True

    async def finish_recording(self, ctx: Any = None) -> None:
        pcm = bytes(self.rec)
        self.rec = bytearray()
        seconds = len(pcm) / (2 * self.rec_rate)
        if seconds < MIN_REC_SECONDS:
            await self.send_json({"type": "error", "text": "Too short — hold the button while you talk."})
            await self.send_json({"type": "state", "state": "ready"})
            return
        quota = await usage_quota.spent(self.user_id)
        if quota is not None:  # don't transcribe what can't be sent
            await self.send_json({"type": "error", "text": usage_quota.short_limit_text(quota)})
            await self.send_json({"type": "state", "state": "ready"})
            return
        await self.send_json({"type": "state", "state": "transcribing"})
        from captain_claw.flight_deck import soniox_routes
        from captain_claw.flight_deck.whatsapp_bridge import _stt_language_hints

        # The owner's Soniox key (their own, else the team's, else the deck's).
        text, err = await soniox_routes.transcribe_text(
            self.user_id or None, wav_from_pcm(pcm, self.rec_rate), "audio/wav",
            language_hints=_stt_language_hints(), source="device",
        )
        text = (text or "").strip()
        if not text:
            await self.send_json({"type": "error", "text": err or "Didn't catch that."})
            await self.send_json({"type": "state", "state": "ready"})
            return
        await self.send_json({"type": "transcript", "text": text})
        await self.send_to_agent(text, ctx)

    # ── the background picture (device_backgrounds.py) ──
    async def sync_background(self) -> int:
        """The owner's background when the device has another one, or its
        removal when they cleared it; 1 when something went."""
        if self.bg_id is None or self.closed:
            return 0
        from captain_claw.flight_deck import device_backgrounds

        have = device_backgrounds.current(self.user_id, self.hw)
        want, jpeg = device_backgrounds.for_device(have[1], self.fw) if have else ("", b"")
        if want == self.bg_id:
            return 0
        if jpeg is None:  # firmware that shows it as it comes: darkened here
            jpeg = await asyncio.to_thread(device_backgrounds.darken, have[1])
        async with self._image_lock:  # never between an answer's picture parts
            await self.send_json({"type": "background", "id": want, "size": len(jpeg)})
            for kind, part in (jpeg_frames(jpeg) if jpeg else []):
                await self.send_bin(BIN_BG if kind == BIN_JPEG else BIN_BG_PART, part)
        self.bg_id = want
        return 1

    # ── pictures ──
    async def _fetch_from_agent(self, path: str) -> bytes:
        import httpx

        ch = self.ch
        if ch is None or not ch.bound_port:
            raise RuntimeError("no agent")
        auth = await gb._resolve_channel_agent_auth(ch)
        params = {"path": path}
        if auth:
            params["token"] = auth
        url = f"http://{ch.bound_host or 'localhost'}:{ch.bound_port}/api/media"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"agent media {resp.status_code}: {resp.text[:120]}")
        return resp.content

    async def _send_image(self, item: dict) -> None:
        try:
            blob = await self._fetch_from_agent(str(item["path"]))
            w, h = self.screen
            max_bytes = MAX_CHUNKED_JPEG_BYTES if self.chunked else MAX_JPEG_BYTES
            jpeg = await asyncio.to_thread(fit_jpeg, blob, w, h, max_bytes=max_bytes,
                                           round_screen=self.round_screen, paper=self.paper)
            frames = jpeg_frames(jpeg) if self.chunked else [(BIN_JPEG, jpeg)]
            if self.paper or self.round_screen:  # its own shape: say which
                from PIL import Image

                w, h = Image.open(io.BytesIO(jpeg)).size
            async with self._image_lock:  # two pictures' parts must not interleave
                await self.send_json({"type": "image", "name": str(item.get("name", "")), "w": w, "h": h,
                                      "size": len(jpeg)})
                for kind, part in frames:
                    await self.send_bin(kind, part)
        except Exception as exc:
            log.warning("device image delivery failed: %s", exc)
            await self.send_json({"type": "status", "text": f"Couldn't show {item.get('name', 'the picture')}."})

    # ── audio queue (one sound at a time) ──
    def start_audio_worker(self) -> None:
        self._audio_worker = asyncio.create_task(self._audio_loop())

    async def _audio_loop(self) -> None:
        while True:
            kind, arg = await self._audio_q.get()
            if kind == "end":  # the answer's voice has all gone out
                await self.send_json({"type": "speech_end"})
                continue
            await self._stop_say()  # an answer's sound never shares the wire with an aside
            job = self._play_tts(arg) if kind == "tts" else self._play_file(arg)
            self._audio_task = asyncio.create_task(job)
            try:
                await self._audio_task
                if kind == "file":  # a sound the agent made: the answer's closing chime may follow it
                    await self.send_json({"type": "speech_end"})
            except asyncio.CancelledError:
                me = asyncio.current_task()
                if me is not None and me.cancelling():
                    raise  # the worker itself is stopping (device gone)
                # else: stop_audio() cancelled just this sound — next one
            except Exception as exc:
                log.warning("device audio failed: %s", exc)
            finally:
                self._audio_task = None

    async def replay(self) -> bool:
        """Play the last answer's sounds again, from the start."""
        sounds = list(_answers.get(self.channel_id) or [])
        if not sounds:
            await self.send_json({"type": "replay_none"})
            return False
        await self.stop_audio()
        self.speech_parts = 0
        for sound in sounds:
            await self._audio_q.put(sound)
        await self._audio_q.put(("end", None))
        return True

    async def say(self, text: str) -> bool:
        """Speak a short line aside from the answer (an e-paper's menu row,
        which the paper takes ~20 s to show): never while an answer plays or
        waits, the newest replaces one still speaking, in English (the
        device's own words), and never kept for a replay."""
        line = speakable(text, MAX_SAY_CHARS)
        busy = (self._audio_task is not None and not self._audio_task.done()) or not self._audio_q.empty()
        if not line or busy:
            return False
        await self._stop_say()
        self._say_task = asyncio.create_task(self._play_tts(line, aside=True, language="en"))
        return True

    async def _stop_say(self) -> None:
        task = self._say_task
        self._say_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    async def stop_audio(self) -> None:
        await self._stop_say()
        while not self._audio_q.empty():
            self._audio_q.get_nowait()
        task = self._audio_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    async def _stream_pcm(self, chunks, aside: bool = False) -> None:
        start: dict = {"type": "audio_start", "rate": TTS_RATE}
        if aside:
            start["aside"] = True
        else:
            if self.speech_parts:
                start["cont"] = True  # the answer's next sound: no cue before it
            self.speech_parts += 1
        await self.send_json(start)
        try:
            pending = b""
            async for chunk in chunks:
                pending += chunk
                while len(pending) >= PCM_CHUNK:
                    await self.send_bin(BIN_AUDIO, pending[:PCM_CHUNK])
                    pending = pending[PCM_CHUNK:]
            if len(pending) % 2:
                pending = pending[:-1]
            if pending:
                await self.send_bin(BIN_AUDIO, pending)
        finally:
            try:
                await self.send_json({"type": "audio_end"})
            except Exception:
                pass

    async def _play_tts(self, text: str, aside: bool = False, language: str | None = None) -> None:
        from captain_claw.flight_deck.soniox_routes import key_for

        gen = gb.soniox_tts_pcm(text, language=language or self.lang, voice=self.voice,
                                api_key=await key_for(self.user_id or None))
        try:
            await self._stream_pcm(gen, aside=aside)
        except gb.SonioxTTSError as exc:
            await self.send_json({"type": "status", "text": f"Voice unavailable: {exc}"[:200]})
        finally:
            await gen.aclose()

    async def _play_file(self, item: dict) -> None:
        if not shutil.which("ffmpeg"):
            await self.send_json({"type": "status", "text": "Can't play audio files (no ffmpeg on the server)."})
            return
        blob = await self._fetch_from_agent(str(item["path"]))
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-t", str(AUDIO_FILE_MAX_S),
            "-f", "s16le", "-ac", "1", "-ar", str(TTS_RATE), "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def _feed() -> None:
            try:
                proc.stdin.write(blob)
                await proc.stdin.drain()
            finally:
                proc.stdin.close()

        async def _chunks():
            while True:
                data = await proc.stdout.read(PCM_CHUNK)
                if not data:
                    return
                yield data

        feeder = asyncio.create_task(_feed())
        try:
            await self._stream_pcm(_chunks())
        finally:
            feeder.cancel()
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()

    async def close(self) -> None:
        self.closed = True
        if not self.pending:
            self.unbind()  # (a turn still running keeps listening: its answer waits for the reconnect)
        await self.stop_audio()
        if self._audio_worker is not None:
            self._audio_worker.cancel()
        for task in list(self._bg):
            task.cancel()


# ── Route ─────────────────────────────────────────────────────────────

# Connected devices, so a freshly published build reaches them right away.
_live: set[_DeviceConn] = set()


async def push_background(user_id: str, hw: str) -> int:
    """The owner's background (or its removal) to their connected devices of
    kind ``hw``; how many got it."""
    sent = 0
    for conn in list(_live):
        if conn.user_id == str(user_id) and conn.hw == hw:
            try:
                sent += await conn.sync_background()
            except Exception as exc:
                log.warning("device background send failed: %s", exc)
    return sent


async def offer_to_connected(hw: str) -> int:
    """Offer the newest build to every connected device of kind ``hw``."""
    offered = 0
    for conn in list(_live):
        if conn.hw != hw:
            continue
        try:
            offered += await conn.offer_update()
        except Exception:
            pass
    return offered



@router.websocket("/device/ws")
async def device_ws(ws: WebSocket) -> None:
    q = ws.query_params
    user = await _device_user(q.get("token", ""))
    if user is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    device_id = re.sub(r"[^A-Za-z0-9_-]", "", q.get("device", "") or "")[:32] or "stick"
    lang = re.sub(r"[^A-Za-z-]", "", q.get("lang", "") or "")[:8]
    conn = _DeviceConn(
        ws, user, device_id, lang=lang, speak=q.get("speak", "1") not in ("0", "false", "off"),
        voice=q.get("voice", ""), **_picture_options(q),
        hw=re.sub(r"[^a-z0-9_-]", "", (q.get("hw", "") or "").lower())[:24],
        fw=re.sub(r"[^0-9.]", "", q.get("fw", "") or "")[:16],
    )
    conn.start_audio_worker()
    _live.add(conn)
    code: Any = None
    try:
        conn.take_over()  # before bind: from here on, the running turn's answer is ours
        if await conn.bind(q.get("agent", "")):
            conn._spawn(conn.resume())
        await conn.offer_update()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                code = msg.get("code")
                break
            conn.heard()
            data = msg.get("bytes")
            if data is not None:
                if conn.recording:
                    if len(conn.rec) + len(data) <= MAX_REC_SECONDS * 2 * conn.rec_rate:
                        conn.rec.extend(data)
                elif conn.note is not None:
                    if len(conn.note["pcm"]) + len(data) <= MAX_NOTE_SECONDS * 2 * conn.note["rate"]:
                        conn.note["pcm"].extend(data)
                continue
            try:
                obj = json.loads(msg.get("text") or "")
            except Exception:
                continue
            mtype = obj.get("type")
            if mtype == "rec_start":
                await conn.stop_audio()
                conn.rec_rate = _int_param(str(obj.get("rate", 16000)), 16000, 8000, 48000)
                conn.rec = bytearray()
                conn.recording = True
            elif mtype == "rec_end":
                if conn.recording:
                    conn.recording = False
                    conn._spawn(conn.finish_recording(obj.get("ctx")))
            elif mtype == "rec_cancel":
                conn.recording = False
                conn.rec = bytearray()
            elif mtype == "text":
                text = str(obj.get("text", "")).strip()
                if text:
                    await conn.stop_audio()
                    conn._spawn(conn.send_to_agent(text, obj.get("ctx")))
            elif mtype == "stop":
                await conn.stop_audio()
            elif mtype == "stop_agent":
                await conn.stop_agent()
            elif mtype == "replay":
                await conn.replay()
            elif mtype == "say":
                await conn.say(str(obj.get("text", "")))
            elif mtype in ("notes_begin", "notes_cancel"):
                conn.notes, conn.note = [], None
            elif mtype == "note_start":
                conn.note = None
                if len(conn.notes) < MAX_NOTES_PER_BATCH and _note_id(obj.get("id")):
                    conn.note = {
                        "id": _note_id(obj.get("id")),
                        "rate": _int_param(str(obj.get("rate", 16000)), 16000, 8000, 48000),
                        "seconds": obj.get("seconds"),
                        "recorded": str(obj.get("recorded", "") or "")[:40],
                        "pcm": bytearray(),
                    }
            elif mtype == "note_end":
                if conn.note is not None and conn.note["id"] == _note_id(obj.get("id")):
                    conn.notes.append(conn.note)
                conn.note = None
            elif mtype == "notes_end":
                batch, conn.notes, conn.note = conn.notes, [], None
                if batch:
                    conn._spawn(conn.deliver_notes(batch, obj.get("ctx")))
                else:
                    await conn.send_json({"type": "notes_done", "ids": [], "heard": 0})
            elif mtype == "agents":
                items = [{"slug": a.get("slug"), "name": a.get("name")} for a in _running_agents(conn.user_id)]
                await conn.send_json({"type": "agents", "items": items,
                                      "current": (conn.agent or {}).get("slug", "")})
            elif mtype == "select":
                await conn.bind(str(obj.get("agent", "")))
            elif mtype == "prefs":
                if "voice" in obj:
                    conn.voice = voice_name(str(obj.get("voice", "")))
                if "speak" in obj:
                    conn.speak = bool(obj.get("speak"))
                if "bg" in obj:  # firmware that shows a background names the one it has
                    conn.bg_id = re.sub(r"[^0-9a-f]", "", str(obj.get("bg") or ""))[:32]
                    conn._spawn(conn.sync_background())
            elif mtype == "update_check":
                await conn.offer_update(reply_none=True)
            elif mtype == "ping":
                pong = {"type": "pong"}
                if isinstance(obj.get("t"), (int, float)):
                    pong["t"] = obj["t"]
                await conn.send_json(pong)
    except WebSocketDisconnect as exc:
        code = exc.code
    except Exception as exc:
        log.warning("device ws error: %s", exc)
    finally:
        _live.discard(conn)
        # How it ended and how long the device had been quiet before: tells a
        # link that faded (weak Wi-Fi) from one closed on purpose.
        now = time.monotonic()
        log.info("device %s (%s %s) left after %.0fs: code %s, last heard %.0fs before%s",
                 conn.device_id, conn.hw or "?", conn.fw or "?", now - conn.opened_at, code,
                 now - conn.last_heard_at, ", mid-turn" if conn.pending else "")
        await conn.close()
