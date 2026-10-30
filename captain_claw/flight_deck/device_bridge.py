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
  text  {"type":"audio_hold"} / {"type":"audio_go"}   its answer's sound is in a ring: hold the
                                                 voice while it's full of what's still to play
                                                 (a long answer, a pause), go on once it has room

  A device that connects with ``cmds=sound`` takes spoken sound commands: a
  recording that is only "volume up / down" or "beep on / off" (English or
  Croatian) never reaches the agent — the device gets {"type":"sound","cmd":…}
  and changes it itself (sound_command()).

Flight Deck → device
  text  {"type":"bound","agent":name,"slug":slug,"linked":bool,"power":p} | {"type":"agents","items":[…],"current":slug}
  text  {"type":"state","state":"connected|agent_offline|transcribing|thinking|ready"}
        connected / agent_offline carry "power": running | asleep (put to sleep for being idle:
        the next word wakes it) | stopped (switched off, or none to bind: then before the error);
        while the link is down agent_offline comes again when that changes. No "power": Flight
        Deck can't tell.
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
  text  {"type":"sound","cmd":"volume_up|volume_down|beep_on|beep_off"}  a spoken sound command
                                                 (cmds=sound devices), then state ready
  text  {"type":"waiting","count":n,"text":"…","beep":bool}   how many proposals wait for the
                                                 owner's approval in Flight Deck (0: none); with
                                                 "beep", something new came (a proposal, a nudge:
                                                 "text" says it in a line) — beep, buzz, blink.
                                                 On connect, and whenever the count moves
                                                 (owner_notify.py)

Auth: a personal access token (Profile → MCP agent access) names the owner;
the device only ever reaches that owner's agents. With Flight Deck auth off
(single-user desktop) no token is needed.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import re
import shutil
import struct
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from captain_claw.flight_deck import glasses_bridge as gb
from captain_claw.flight_deck import usage_quota
from captain_claw.flight_deck.bus_client import (  # noqa: F401 — activity_text: the device's activity line
    HEARD_AFTER_S,
    TURN_KEEP_S,
    BusClient,
    activity_text,
    running_agents,
)

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
AUDIO_HOLD_MAX_S = 180.0         # a device that asked to hold and never said go: carry on after this
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
RESUME_GRACE_S = 1.5             # a reconnect's talk-from-sleep recording starts within this

# The device turn per channel, kept across reconnects (weak Wi-Fi drops the
# link mid-turn): open from the send until "ready", and its answer until the
# device is heard from after it (TURN_KEEP_S, HEARD_AFTER_S). A reconnect
# carries on with it.
_turns: dict[str, dict] = {}

# Offline voice notes already handed to the agent, per device channel: a
# batch sent again (its notes_done lost with the link) is acknowledged, not
# delivered twice.
_delivered_notes: dict[str, list[str]] = {}
_notes_locks: dict[str, asyncio.Lock] = {}

OFFLINE_NOTES_CONTEXT = (
    "Voice notes: the user recorded these on the device — while it had no connection to you, or in its "
    "note-taking mode — each at the time shown. They weren't waiting for an answer then. Handle them "
    "now, in order — do what each one asks, or keep it as a note when it is one — and answer once, briefly."
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


def _blen(s: str) -> int:
    """Length in UTF-8 bytes: how the firmware counts its text (č, đ, ° are two)."""
    return len(s.encode("utf-8"))


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# Spoken sound commands: the whole recording is the command (a "please"
# either side allowed), so "turn up the volume on the kitchen speaker" still
# goes to the agent.
_SOUND_COMMANDS = {
    "volume_up": ("volume up", "louder", "turn it up", "turn up the volume", "turn the volume up",
                  "increase the volume", "raise the volume", "glasnije", "pojačaj", "pojačaj zvuk",
                  "pojačaj glasnoću", "glasnoća gore"),
    "volume_down": ("volume down", "quieter", "softer", "turn it down", "turn down the volume",
                    "turn the volume down", "decrease the volume", "lower the volume", "tiše", "stišaj",
                    "utišaj", "smanji zvuk", "smanji glasnoću", "glasnoća dolje"),
    "beep_on": ("beep on", "beeps on", "beeping on", "turn on the beep", "turn on the beeps",
                "turn the beeps on", "bip uključi", "uključi bip", "uključi bipove", "bipovi uključeni"),
    "beep_off": ("beep off", "beeps off", "beeping off", "no beeps", "turn off the beep",
                 "turn off the beeps", "turn the beeps off", "bip isključi", "isključi bip",
                 "isključi bipove", "bipovi isključeni", "bez bipova"),
}
_SOUND_PHRASES = {p: cmd for cmd, phrases in _SOUND_COMMANDS.items() for p in phrases}
_POLITE = re.compile(r"^(?:please|molim(?: te)?)\s+|\s+(?:please|molim(?: te)?)$")


def sound_command(text: str) -> str | None:
    """"volume_up" / "volume_down" / "beep_on" / "beep_off" when *text* is
    only that command ("Volume up.", "Beep, off!", "pojačaj molim te")."""
    said = re.sub(r"[^\w\s]", " ", text.lower().replace("-", " "))
    said = _POLITE.sub("", " ".join(said.split())).strip()
    return _SOUND_PHRASES.get(said)


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
                  "papercolor": "M5Stack PaperColor", "epaper154": "Waveshare ePaper 1.54",
                  "epaper154_touch": "Waveshare Touch ePaper 1.54"}
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
    head = ("Voice notes I recorded on the device:" if len(notes) > 1
            else "A voice note I recorded on the device:")
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
    img, paper, mono); with none of them, the StickS3's 240×135 in one frame."""
    opts = {
        "screen": (_int_param(q.get("sw", ""), 240, 32, 800), _int_param(q.get("sh", ""), 135, 32, 800)),
        "round_screen": q.get("round", "") in ("1", "true", "on"),
        "chunked": q.get("img", "") == "chunks",
    }
    if q.get("paper", "") in ("1", "true", "on"):
        opts["paper"] = True
    if q.get("mono", "") in ("1", "true", "on"):
        opts["mono"] = True
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


def _pick_agent(user_id: str, selector: str) -> dict | None:
    """The agent a selector names (slug, name or port) among the owner's
    running agents, else their first running one."""
    agents = running_agents(user_id)
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


class _DeviceConn(BusClient):
    MSG_ID_PREFIX = "dev_"

    def __init__(self, ws: WebSocket, user: dict, device_id: str, *, lang: str,
                 speak: bool, screen: tuple[int, int], voice: str = "",
                 round_screen: bool = False, chunked: bool = False,
                 paper: bool = False, mono: bool = False, hw: str = "", fw: str = "",
                 sound_cmds: bool = False) -> None:
        super().__init__(ws, user)
        self.hw = hw  # device kind + firmware version, for over-the-air updates
        self.fw = fw
        self.sound_cmds = sound_cmds  # it changes volume / beeps when told to ("cmds=sound")
        self.device_id = device_id
        self.lang = lang
        self.speak = speak
        self.voice = voice_name(voice)
        self.screen = screen
        self.round_screen = round_screen  # pictures fit inside the circle
        self.chunked = chunked            # pictures may span several frames
        self.paper = paper                # colour e-paper: pictures at their own shape
        self.mono = mono                  # small black-and-white e-paper: the same pictures, its own fonts
        self._image_lock = asyncio.Lock()  # one picture's parts at a time
        self.bg_id: str | None = None  # the background it has ("" none); None: it can't show one
        # The answer as the model writes it (agent_delta): shown as it grows
        # and spoken sentence by sentence as they complete.
        self.partial = ""
        self.spoken_upto = 0     # chars of partial handed to the voice
        self.spoken_total = 0    # chars spoken of this answer so far
        self.streamed = False    # some of this answer went out as it came
        self.last_part = ""      # the text last shown as a part
        self.speech_parts = 0    # sounds sent for this answer: the first gets the cue, the rest carry on
        self.recording = False
        self.rec_rate = 16000
        self.rec = bytearray()
        self.cue_alone = False  # the open turn is a bare "new topic"
        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._audio_task: asyncio.Task | None = None
        self._audio_worker: asyncio.Task | None = None
        self._say_task: asyncio.Task | None = None
        # The device's answer audio is a ring: it asks to hold ("audio_hold")
        # while it's full of what's still to play, and says "audio_go" again.
        self.audio_go = asyncio.Event()
        self.audio_go.set()
        # Offline voice notes arriving: the one now (note_start … note_end), the batch so far.
        self.note: dict | None = None
        self.notes: list[dict] = []

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

    # ── binding ──
    @property
    def channel_id(self) -> str:
        return f"device:{self.user_id}:{self.device_id}"

    def turns(self) -> dict[str, dict]:
        return _turns

    def pick_agent(self, selector: str) -> dict | None:
        return _pick_agent(self.user_id, selector)

    async def _no_agent(self) -> None:
        # None running or asleep: the device shows its agent switched off.
        self._power = "stopped"
        await self.send_json({"type": "state", "state": "agent_offline", "power": "stopped"})

    # ── agent bus → device ──
    def _command_answer(self, command: str, text: str) -> tuple[str | None, bool]:
        if command != "/new":
            return text, False
        # A spoken "new topic, …": the device hears the answer that follows,
        # not "New session created". Said alone, nothing follows — a short
        # line closes the turn instead.
        if not (self.pending and self.cue_alone):
            return None, False
        self.cue_alone = False
        self._end_turn()
        return NEW_TOPIC_REPLY, True

    async def _on_media(self, ev: dict) -> None:
        if self.closed or not (self.pending or time.monotonic() - self.last_reply_at < MEDIA_WINDOW_S):
            return
        for item in ev.get("items") or []:
            kind = str(item.get("kind", ""))
            if kind == "image":
                self._spawn(self._send_image(item))
            elif kind == "audio":
                _answers.setdefault(self.channel_id, []).append(("file", item))
                await self._audio_q.put(("file", item))

    async def _show_reply(self, text: str, proactive: bool, turn: dict | None = None) -> None:
        """An answer on the screen and spoken (what of it wasn't already, as it
        streamed); a turn's answer is also kept until the device is heard from
        after it (a dropped link loses it)."""
        self._keep_answer(turn, text, proactive)
        if self.closed:
            self._forget_stream()
            return  # the reconnect brings it
        self.last_reply_at = time.monotonic()
        has_dj = self.round_screen or self.paper
        shown = screen_text(text, has_dj=has_dj)
        if self.streamed:
            # Spoken as it came: only what follows the last sentence read is left.
            read = self.partial[:self.spoken_upto]
            tail = text[len(read):] if text.startswith(read) else text[_common_prefix(text, read):]
            spoken = speakable(tail) if self.speak else ""
            total = self.spoken_total + _blen(spoken)
            span = (_blen(screen_text(read, has_dj=has_dj)), _blen(shown))
        else:
            spoken = speakable(text) if self.speak else ""
            total = _blen(spoken)
            span = (0, _blen(shown))
            _answers[self.channel_id] = []  # a new answer
            self.speech_parts = 0
        # Round screens (the StopWatch) and e-paper use Inter, which has Đ.
        # "spoken": how much of it is read aloud (UTF-8 bytes, as the device
        # counts), so the watch can follow the voice through the text (0: none).
        await self.send_json({"type": "reply", "text": shown, "proactive": proactive, "spoken": total})
        if spoken:
            _answers.setdefault(self.channel_id, []).append(("tts", spoken, span))
            await self._audio_q.put(("tts", spoken, span))
        if total:
            await self._audio_q.put(("end", None))  # speech_end, once the voice has all gone out
        self._forget_stream()

    # ── the answer as it streams ──
    async def _on_delta(self, text: str) -> None:
        if self.cue_alone:
            return  # a bare "new topic": its short line is the answer
        self.partial += text
        while True:
            m = _SENTENCE_END.search(self.partial, self.spoken_upto)
            if not m:
                return
            start, end = self.spoken_upto, m.end()
            self.spoken_upto = end
            await self._speak_part(self.partial[start:end], start, end)

    async def _speak_part(self, sentence: str, start: int, end: int) -> None:
        """A sentence that just completed: on the screen (the answer so far)
        and read aloud, straight after the one before. Its sound says which
        characters of the screen text it covers, so the watch's lit sentence
        follows the voice exactly at each sentence and by its length within."""
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
            has_dj = self.round_screen or self.paper
            span = (_blen(screen_text(self.partial[:start], has_dj=has_dj)),
                    _blen(screen_text(self.partial[:end], has_dj=has_dj)))
            _answers.setdefault(self.channel_id, []).append(("tts", spoken, span))
            self.spoken_total += _blen(spoken)
            await self._audio_q.put(("tts", spoken, span))

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

    async def stop_output(self) -> None:
        await self.stop_audio()  # the device's stop button: what was queued to be spoken is dropped

    # ── a turn across reconnects ──
    async def _resume_ready(self) -> bool:
        await asyncio.sleep(RESUME_GRACE_S)  # woken by talking: that recording goes first
        return not self.recording

    # ── device → agent ──
    def source_fields(self) -> dict[str, str]:
        """The frame's ``source`` (``device:<kind>``) and its label — the
        model's name and this device's id."""
        kind = self.hw if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", self.hw or "") else ""
        model = _DEVICE_MODELS.get(kind, "")
        label = f"{model} ({self.device_id})" if model else f"Device ({self.device_id})"
        return {"source": f"device:{kind}" if kind else "device", "source_label": label}

    def system_context(self) -> str:
        w, h = self.screen
        if self.paper:
            screen = (f"4-inch colour e-paper ({w}×{h} px; it takes about 20 s to redraw, "
                      "so your reply is heard first and stays readable on it after)")
        elif self.mono:
            screen = f"1.54-inch black-and-white e-paper ({w}×{h} px)"
        elif self.round_screen:
            screen = f"small round {min(w, h)} px"
        else:
            screen = f"{w}×{h} pixel"
        return _DEVICE_SYSTEM_CONTEXT.replace("{screen}", screen)

    def chat_fields(self, ch: Any, text: str, ctx: Any, extra_notes: list[str] | None) -> dict:
        """A device turn: the device rules ahead of the binding's first
        message, and the device's readings (and any *extra_notes*) as
        context lines."""
        fields = {
            "content": text if ch.context_sent else self.system_context() + text, "surface": "device",
            # Which device it was (turn_source): recorded on the message
            # and the key of the agent's reasoning effort for devices.
            **self.source_fields(),
            # This turn's own origin (the Taskboard's record of it): a reply the
            # user approves there comes back to this device's channel. Not the
            # session's durable origin, which a cron result follows.
            "turn_origin": {"kind": "channel", "address": self.channel_id},
        }
        readings = device_context_line(ctx, self.hw)
        notes = ([readings] if readings else []) + list(extra_notes or [])
        if notes:
            fields["context_notes"] = notes
        return fields

    def _turn_started(self, text: str, turn: dict) -> None:
        self._forget_stream()  # a new answer
        self.speech_parts = 0
        self.cue_alone = cue_alone(text)

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
        cmd = sound_command(text) if self.sound_cmds else None
        if cmd:  # "volume up", "beep off": the device's own business, not the agent's
            log.info("device %s: sound command %s (%r)", self.device_id, cmd, text[:60])
            await self.send_json({"type": "sound", "cmd": cmd})
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
                                           round_screen=self.round_screen, paper=self.paper or self.mono)
            frames = jpeg_frames(jpeg) if self.chunked else [(BIN_JPEG, jpeg)]
            if self.paper or self.mono or self.round_screen:  # its own shape: say which
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
            item = await self._audio_q.get()
            kind, arg = item[0], item[1]
            span = item[2] if len(item) > 2 else None  # (from, to): the sound's words in the text
            if kind == "end":  # the answer's voice has all gone out
                await self.send_json({"type": "speech_end"})
                continue
            await self._stop_say()  # an answer's sound never shares the wire with an aside
            if kind == "tts":
                job = self._play_tts(arg, span=span) if span is not None else self._play_tts(arg)
            else:
                job = self._play_file(arg)
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
        self.audio_go.set()  # a stopped answer holds nothing back
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

    async def _stream_pcm(self, chunks, aside: bool = False, span: tuple | None = None) -> None:
        start: dict = {"type": "audio_start", "rate": TTS_RATE}
        if aside:
            start["aside"] = True
        else:
            if self.speech_parts:
                start["cont"] = True  # the answer's next sound: no cue before it
            self.speech_parts += 1
            if span is not None:  # where this sound's words are in the shown text
                start["from"], start["to"] = int(span[0]), int(span[1])
        await self.send_json(start)
        # The voice is taken from TTS as fast as it comes (a long hold mustn't
        # time that stream out); only the sending waits for the device's room.
        got: asyncio.Queue = asyncio.Queue()

        async def pull() -> None:
            try:
                async for chunk in chunks:
                    got.put_nowait(chunk)
            except BaseException as exc:  # (cancelled too) handed to the sender
                got.put_nowait(exc)
                return
            got.put_nowait(None)

        puller = asyncio.create_task(pull())
        try:
            pending = b""
            while (item := await got.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                pending += item
                while len(pending) >= PCM_CHUNK:
                    if not aside:
                        await self._audio_room()
                    await self.send_bin(BIN_AUDIO, pending[:PCM_CHUNK])
                    pending = pending[PCM_CHUNK:]
            if len(pending) % 2:
                pending = pending[:-1]
            if pending:
                await self.send_bin(BIN_AUDIO, pending)
        finally:
            if not puller.done():
                puller.cancel()
            try:
                await puller
            except BaseException:
                pass
            try:
                await self.send_json({"type": "audio_end"})
            except Exception:
                pass

    async def _audio_room(self) -> None:
        """Until the device has room for more of the answer's sound (it said
        "audio_go", or never "audio_hold"); at most AUDIO_HOLD_MAX_S."""
        if self.audio_go.is_set():
            return
        try:
            await asyncio.wait_for(self.audio_go.wait(), AUDIO_HOLD_MAX_S)
        except asyncio.TimeoutError:
            self.audio_go.set()

    async def _play_tts(self, text: str, aside: bool = False, language: str | None = None,
                        span: tuple | None = None) -> None:
        from captain_claw.flight_deck.soniox_routes import key_for

        gen = gb.soniox_tts_pcm(text, language=language or self.lang, voice=self.voice,
                                api_key=await key_for(self.user_id or None))
        try:
            await self._stream_pcm(gen, aside=aside, span=span)
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

    async def _close_output(self) -> None:
        await self.stop_audio()
        if self._audio_worker is not None:
            self._audio_worker.cancel()


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


async def push_waiting(user_id: str, count: int, text: str = "", beep: bool = False) -> list[str]:
    """How many of the owner's proposals wait (and with ``beep``, that
    something new came: ``text``) to their connected devices; the ids of
    those that got it (owner_notify.py)."""
    from captain_claw.flight_deck.autonomy import _norm_user

    owner = _norm_user(user_id)
    frame = {"type": "waiting", "count": max(0, int(count)), "text": text, "beep": bool(beep)}
    reached: list[str] = []
    for conn in list(_live):
        if _norm_user(conn.user_id) != owner or conn.closed:
            continue
        try:
            await conn.send_json(frame)
            reached.append(conn.device_id)
        except Exception as exc:
            log.debug("device waiting send failed: %s", exc)
    return reached


def connected_devices(user_id: str) -> int:
    """How many of the owner's devices are connected now."""
    from captain_claw.flight_deck.autonomy import _norm_user

    owner = _norm_user(user_id)
    return sum(1 for conn in list(_live) if _norm_user(conn.user_id) == owner and not conn.closed)


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
        sound_cmds="sound" in (q.get("cmds", "") or "").split(","),
    )
    conn.start_audio_worker()
    _live.add(conn)
    code: Any = None
    try:
        conn.take_over()  # before bind: from here on, the running turn's answer is ours
        if await conn.bind(q.get("agent", "")):
            conn._spawn(conn.resume())
        await conn.offer_update()
        # How many proposals wait in Flight Deck (and a beep for what came
        # while it slept). Firmware that doesn't know the frame ignores it.
        from captain_claw.flight_deck.owner_notify import device_connected
        waiting = device_connected(conn.user_id, conn.device_id)
        if waiting:
            await conn.send_json(waiting)
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
            elif mtype == "audio_hold":
                conn.audio_go.clear()
            elif mtype == "audio_go":
                conn.audio_go.set()
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
                items = [{"slug": a.get("slug"), "name": a.get("name")} for a in running_agents(conn.user_id)]
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
