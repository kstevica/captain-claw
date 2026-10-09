"""Send a file the agent saved to a WhatsApp chat.

Agent-generated files live on the agent's own filesystem under
``<workspace>/saved/<category>/<session>/<filename>`` (the same files
shown in the Flight Deck / glasses file list). This tool runs *inside*
the agent process: it resolves one of those files, reads its bytes, and
sends it to a WhatsApp chat via the Meta Cloud API — a photo (JPEG/PNG up
to 5 MB) as an image, an H.264 MP4 up to 16 MB as a video, MP3/M4A/AAC/AMR
or Opus audio up to 16 MB as audio, anything else (or anything larger, or a
file whose content doesn't match what WhatsApp plays) as a document.

Two actions:
  * ``list`` — list the agent's saved files (newest first) so the agent can
    discover which one the user means ("that last report").
  * ``send`` — deliver a file. Identify it by ``path`` (what the agent knows,
    e.g. ``showcase/<session>/report.docx``), by ``filename`` (fuzzy, newest
    match wins), or ``latest`` for the most recently saved file.

Recipient resolution for ``send``:
  * An explicit ``to`` (phone number, digits only, no ``+``) wins.
  * Otherwise the *current* WhatsApp chat — captured per session when the
    conversation arrived over WhatsApp (``session.metadata['whatsapp_waid']``).

Requirements: ``WHATSAPP_ACCESS_TOKEN`` + ``WHATSAPP_PHONE_NUMBER_ID`` in the
environment (inherited from Flight Deck). WhatsApp only permits free-form
documents within 24h of the recipient's last message; out-of-window sends
are rejected by Meta and surfaced here as a clear error.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import httpx

from captain_claw.tools.registry import Tool, ToolResult

# WhatsApp Cloud API caps documents at 100 MB; stay just under.
_MAX_DOC_BYTES = 95 * 1024 * 1024
_GRAPH_BASE = "https://graph.facebook.com/v18.0"

# Media the Cloud API shows inline: extension -> (message type, MIME, max
# bytes; Meta's MB are decimal). Anything else, anything over the cap, or a
# file whose bytes don't match what WhatsApp plays goes as a document —
# documents have no codec rules, so they always arrive.
_IMAGE_CAP = 5_000_000
_AV_CAP = 16_000_000
_MEDIA_BY_EXT: dict[str, tuple[str, str, int]] = {
    ".jpg": ("image", "image/jpeg", _IMAGE_CAP),
    ".jpeg": ("image", "image/jpeg", _IMAGE_CAP),
    ".png": ("image", "image/png", _IMAGE_CAP),
    ".mp4": ("video", "video/mp4", _AV_CAP),
    ".mp3": ("audio", "audio/mpeg", _AV_CAP),
    ".m4a": ("audio", "audio/mp4", _AV_CAP),
    ".aac": ("audio", "audio/aac", _AV_CAP),
    ".amr": ("audio", "audio/amr", _AV_CAP),
    ".ogg": ("audio", "audio/ogg", _AV_CAP),
    ".opus": ("audio", "audio/ogg", _AV_CAP),
}
# Per session (on the agent), for the turn running now: files sent to
# WhatsApp, so the end-of-turn delivery doesn't send them again, and the
# chat the turn answers plus whether it is automated (the tool's default
# recipient). Cleared when the turn ends.
SENT_THIS_TURN_ATTR = "_whatsapp_sent_paths"
REPLY_TO_ATTR = "_whatsapp_reply_to"
# Meta errors that mean "this file as this media type" (worth one retry as a
# document) — not auth, rate-limit, 24h-window or network trouble.
_FORMAT_ERRORS = {100, 131051, 131052, 131053}
# MP4 sample entries WhatsApp plays: H.264 video, AAC audio, timed text.
_MP4_PLAYABLE = {b"avc1", b"avc3", b"mp4a", b"tx3g"}

# Meta's Cloud API rejects any document MIME outside a fixed allowlist
# (error #100), and the host's mimetypes DB can't be trusted to produce
# the exact strings Meta wants (e.g. .pptx). Map the common cases
# explicitly; this is the canonical Meta-accepted set for documents.
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
}
# Text-based formats Meta doesn't list individually but accepts when sent
# as text/plain documents (the file keeps its name/extension for the user).
_TEXT_EXT = {
    ".md", ".markdown", ".html", ".htm", ".csv", ".tsv", ".json", ".log",
    ".py", ".js", ".ts", ".css", ".xml", ".yaml", ".yml", ".rtf",
    ".svg", ".vcf", ".ics",
}


def _guess_doc_mime(filename: str) -> str:
    """Return a WhatsApp-accepted document MIME for *filename*."""
    ext = Path(filename).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    if ext in _TEXT_EXT:
        return "text/plain"
    if ext in _MEDIA_BY_EXT:              # fixed, not the host's mimetypes guess
        return _MEDIA_BY_EXT[ext][1]
    if ext == ".3gp":
        return "video/3gpp"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _mp4_tracks(blob: bytes) -> list[tuple[bytes, int]]:
    """(first sample-entry format, entry count) of every ``stsd`` box inside
    ``moov`` — one per track: avc1, hvc1, mp4a, alac… Empty when the file
    isn't a readable MP4."""
    pos, moov = 0, b""
    while pos + 8 <= len(blob):
        size = int.from_bytes(blob[pos:pos + 4], "big")
        kind = blob[pos + 4:pos + 8]
        header = 8
        if size == 1 and pos + 16 <= len(blob):
            size, header = int.from_bytes(blob[pos + 8:pos + 16], "big"), 16
        elif size == 0:
            size = len(blob) - pos
        if size < header:
            break
        if kind == b"moov":
            moov = blob[pos + header:pos + size]
            break
        pos += size
    tracks: list[tuple[bytes, int]] = []
    at = moov.find(b"stsd")
    while at != -1 and at + 20 <= len(moov):
        tracks.append((moov[at + 16:at + 20], int.from_bytes(moov[at + 8:at + 12], "big")))
        at = moov.find(b"stsd", at + 4)
    return tracks


def _jpeg_is_8bit_colour(blob: bytes) -> bool:
    """A baseline/progressive JPEG with 8-bit samples and three components
    (YCbCr) — not CMYK, grayscale or 12-bit."""
    pos = 2
    while pos + 9 < len(blob):
        if blob[pos] != 0xFF:
            return False
        marker = blob[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = int.from_bytes(blob[pos + 2:pos + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return blob[pos + 4] == 8 and blob[pos + 9] == 3
        if marker == 0xDA or length < 2:
            return False
        pos += 2 + length
    return False


def _plays_on_whatsapp(kind: str, ext: str, blob: bytes) -> str | None:
    """The upload MIME when *blob* is media WhatsApp shows inline as *kind*,
    else None (send it as a document)."""
    if kind == "image":
        if blob.startswith(b"\xff\xd8\xff"):
            return "image/jpeg" if _jpeg_is_8bit_colour(blob) else None
        # PNG: 8-bit RGB or RGBA only (IHDR bit depth / colour type).
        if blob.startswith(b"\x89PNG") and len(blob) > 25 and blob[24] == 8 and blob[25] in (2, 6):
            return "image/png"
        return None
    if ext in (".ogg", ".opus"):
        return "audio/ogg" if b"OpusHead" in blob[:512] else None   # Opus only
    if ext in (".mp4", ".m4a"):
        # One sample entry per track, every track playable, one audio stream.
        tracks = _mp4_tracks(blob)
        formats = {fmt for fmt, _n in tracks}
        if (not tracks or any(n != 1 for _f, n in tracks) or not formats <= _MP4_PLAYABLE
                or sum(1 for fmt, _n in tracks if fmt == b"mp4a") > 1):
            return None
        if ext == ".mp4":
            return "video/mp4" if formats & {b"avc1", b"avc3"} else None
        return "audio/mp4" if formats == {b"mp4a"} else None
    if ext == ".mp3":
        ok = blob.startswith(b"ID3") or (len(blob) > 1 and blob[0] == 0xFF and blob[1] & 0xE0 == 0xE0)
        return "audio/mpeg" if ok else None
    return _MEDIA_BY_EXT[ext][1]


def media_kind(filename: str, size: int, head: bytes | None = None) -> tuple[str, str]:
    """(message type, upload MIME) WhatsApp gets *filename* as: image, video
    or audio when the format, size and — given the file's bytes as *head* —
    the content allow, else document."""
    ext = Path(filename).suffix.lower()
    kind = _MEDIA_BY_EXT.get(ext)
    if not kind or size > kind[2]:
        return "document", _guess_doc_mime(filename)
    if head is None:
        return kind[0], kind[1]
    mime = _plays_on_whatsapp(kind[0], ext, head)
    return (kind[0], mime) if mime else ("document", _guess_doc_mime(filename))


def _session_key(agent: Any) -> str:
    return str(getattr(getattr(agent, "session", None), "id", "") or "")


def reset_turn(agent: Any, reply_to: str = "", *, automated: bool = False) -> None:
    """Start (or, with no arguments, end) a turn on the agent's current
    session: nothing sent yet, *reply_to* (the WhatsApp chat this turn
    answers, or "") as the tool's default recipient, and whether the turn is
    automated — an automated turn without a chat names its recipient
    explicitly instead of falling back to the session's WhatsApp chat."""
    key = _session_key(agent)
    for attr, value in ((SENT_THIS_TURN_ATTR, set()),
                        (REPLY_TO_ATTR, {"to": reply_to, "automated": automated})):
        table = getattr(agent, attr, None)
        if not isinstance(table, dict):
            table = {}
            try:
                setattr(agent, attr, table)
            except Exception:
                return
        table[key] = value


def _turn_entry(agent: Any) -> dict[str, Any]:
    table = getattr(agent, REPLY_TO_ATTR, None)
    entry = table.get(_session_key(agent)) if isinstance(table, dict) else None
    return entry if isinstance(entry, dict) else {}


def reply_to(agent: Any) -> str:
    """The WhatsApp chat the running turn answers ("" when none)."""
    return str(_turn_entry(agent).get("to") or "")


def turn_is_automated(agent: Any) -> bool:
    return bool(_turn_entry(agent).get("automated"))


def mark_sent(agent: Any, path: Path) -> None:
    """Remember that *path* went to WhatsApp during this session's turn."""
    if agent is None:
        return
    table = getattr(agent, SENT_THIS_TURN_ATTR, None)
    if not isinstance(table, dict):
        table = {}
        try:
            setattr(agent, SENT_THIS_TURN_ATTR, table)
        except Exception:
            return
    table.setdefault(_session_key(agent), set()).add(str(Path(path).resolve()))


def sent_this_turn(agent: Any) -> set[str]:
    table = getattr(agent, SENT_THIS_TURN_ATTR, None)
    if not isinstance(table, dict):
        return set()
    return set(table.get(_session_key(agent), set()))


def _meta_error_code(error: str) -> int | None:
    match = re.search(r'"code"\s*:\s*(\d+)', str(error or ""))
    return int(match.group(1)) if match else None


async def send_whatsapp_media(
    to: str, path: Path, caption: str = "", *, as_document: bool = False,
) -> tuple[bool, str, str]:
    """Upload *path* and send it to WhatsApp *to* as the right message type
    (*as_document*: the original file, never re-encoded). A photo, video or
    audio Meta refuses goes again as a document. Returns ``(ok, kind,
    error)``; never raises."""
    token = _env("WHATSAPP_ACCESS_TOKEN")
    pid = _env("WHATSAPP_PHONE_NUMBER_ID")
    if not token or not pid:
        return False, "", "WhatsApp not configured (WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID)."
    to = str(to or "").lstrip("+").strip()
    if not to:
        return False, "", "no recipient"
    allowed = _allowed_waids()
    if allowed and to not in allowed:
        return False, "", f"Recipient {to} is not in WHATSAPP_ALLOWED_WAIDS."
    try:
        blob = Path(path).read_bytes()
    except Exception as exc:
        return False, "", f"Could not read file: {exc}"
    if not blob:
        return False, "", "File is empty."
    if len(blob) > _MAX_DOC_BYTES:
        return False, "", f"File is {len(blob)} bytes; WhatsApp documents max ~100 MB."
    name = Path(path).name
    if as_document:
        kind, mime = "document", _guess_doc_mime(name)
    else:
        kind, mime = media_kind(name, len(blob), blob)
    ok, err = await _upload_and_send(token, pid, to, blob, name, mime, caption, kind)
    if not ok and kind != "document" and _meta_error_code(err) in _FORMAT_ERRORS:
        # Refused as this media type: a document has no format rules.
        ok, _doc_err = await _upload_and_send(
            token, pid, to, blob, name, _guess_doc_mime(name), caption, "document")
        if ok:
            return True, "document", ""
    return (True, kind, "") if ok else (False, kind, err)


async def _upload_and_send(
    token: str, pid: str, to: str, blob: bytes, name: str, mime: str, caption: str, kind: str,
) -> tuple[bool, str]:
    media_id, err = await WhatsAppSendFileTool._meta_upload(token, pid, blob, name, mime)
    if not media_id:
        return False, f"WhatsApp upload failed: {err}"
    ok, err = await WhatsAppSendFileTool._meta_send(token, pid, to, media_id, name, caption, kind=kind)
    return (True, "") if ok else (False, f"WhatsApp send failed: {err}")


# Made for the user, so always delivered: generated pictures, phone photos,
# spoken audio. (Browser screenshots are the agent's own working views.)
_DELIVERED_IMAGE_TOOLS = ("image_gen", "termux")
_DELIVERED_AUDIO_TOOLS = ("pocket_tts",)
# saved/ folders that never hold deliverables: working code, and inputs the
# agent fetched (a Drive file it summarises isn't sent back unasked).
_WORKING_DIRS = ("scripts", "tools", "skills", "downloads")
# Extensions a reply can name to have the file delivered (all of them go
# through Meta: as media, or as a document Meta accepts).
_NAMEABLE = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".tsv", ".txt", ".md",
    ".html", ".json", ".vcf", ".ics", ".svg", ".rtf", ".png", ".jpg", ".jpeg", ".webp",
    ".mp3", ".m4a", ".aac", ".amr", ".ogg", ".opus", ".mp4",
}
_NAMED_EXT_RE = re.compile(
    r"\.(?:" + "|".join(sorted(e.lstrip(".") for e in _NAMEABLE)) + r")\b", re.IGNORECASE)


def _named_new_files(agent: Any, reply: str, since: float) -> list[Path]:
    """Files this session wrote during the turn (under
    ``saved/<category>/<session>/``, modified at or after *since*) that the
    reply names by their whole file name — "here's report.docx". Not
    scripts, tools, skills or downloads."""
    text = str(reply or "").lower()
    if not text or since <= 0 or not _NAMED_EXT_RE.search(text):
        return []
    try:
        base = Path(agent.tools.get_saved_base_path(create=False))
        slug = agent._current_session_slug()
    except Exception:
        return []
    if not base.is_dir() or not slug:
        return []
    found: list[tuple[float, Path]] = []
    for category in base.iterdir():
        folder = category / slug
        if category.name in _WORKING_DIRS or not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            try:
                if path.suffix.lower() not in _NAMEABLE or not path.is_file():
                    continue
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < since:
                continue
            if re.search(r"(?<![\w.-])" + re.escape(path.name.lower()) + r"(?![\w-])", text):
                found.append((mtime, path))
    found.sort()
    return [p for _m, p in found]


async def deliver_turn_media(
    agent: Any, waid: str, turn_start_idx: int, reply: str = "", turn_started_at: float = 0.0,
    already_sent: set[str] | None = None,
) -> list[str]:
    """After a turn that answers a WhatsApp chat, send what it made for the
    user into that chat: pictures it generated or took with the phone, audio
    it spoke, and any file this session wrote during the turn that the reply
    names ("here's report.docx") — not ones it already sent with
    ``whatsapp_send_file``. If something can't be sent, the chat gets one
    short line saying so. Off with ``whatsapp.auto_send_media``. Returns the
    names sent; never raises."""
    import logging

    from captain_claw.config import get_config
    from captain_claw.platform_adapter import (
        effective_turn_start_idx,
        extract_audio_paths_from_tool_output,
        extract_image_paths_from_tool_output,
    )

    log = logging.getLogger(__name__)
    try:
        cfg = get_config().whatsapp
        session = getattr(agent, "session", None)
        if not cfg.auto_send_media or not str(waid or "").strip() or session is None:
            return []
        if not _env("WHATSAPP_ACCESS_TOKEN") or not _env("WHATSAPP_PHONE_NUMBER_ID"):
            log.info("WhatsApp media delivery skipped: no WhatsApp credentials in this agent")
            return []
        # Everything is read now, before any upload awaits: the next turn may
        # start meanwhile, and this one's state must not leak into it.
        already = set(already_sent) if already_sent is not None else sent_this_turn(agent)
        start = effective_turn_start_idx(agent, turn_start_idx)
        found: list[Path] = []
        for msg in session.messages[start:]:
            if str(msg.get("role", "")) != "tool":
                continue
            tool = str(msg.get("tool_name", "")).strip().lower()
            content = str(msg.get("content", "") or "")
            if content.strip().lower().startswith("error"):
                continue
            if tool in _DELIVERED_IMAGE_TOOLS:
                found.extend(extract_image_paths_from_tool_output(content))
            elif tool in _DELIVERED_AUDIO_TOOLS:
                found.extend(extract_audio_paths_from_tool_output(content))
        found.extend(await asyncio.to_thread(_named_new_files, agent, reply, turn_started_at))
        queue: list[Path] = []
        seen = set(already)
        for path in found:
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                queue.append(Path(path))
        limit = max(0, int(cfg.max_media_per_turn))
        sent: list[str] = []
        failed: list[str] = []
        for path in queue[:limit]:
            ok, _kind, err = await send_whatsapp_media(waid, path)
            if ok:
                sent.append(path.name)
            else:
                failed.append(path.name)
                log.warning("WhatsApp media delivery failed for %s: %s", path.name, err)
        notes = []
        if failed:
            notes.append("Couldn't send " + ", ".join(failed) + " here.")
        if len(queue) > limit:
            notes.append(f"{len(queue) - limit} more file(s) are in the agent's files.")
        if notes:
            await send_whatsapp_text(waid, " ".join(notes))
        return sent
    except Exception as exc:
        log.warning("WhatsApp media delivery failed: %s", exc)
        return []


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _allowed_waids() -> set[str]:
    raw = _env("WHATSAPP_ALLOWED_WAIDS")
    return {p.strip().lstrip("+") for p in raw.split(",") if p.strip()} if raw else set()


async def send_whatsapp_text(to: str, body: str) -> tuple[bool, str]:
    """Send a plain WhatsApp text message via the Meta Cloud API.

    Reuses the same creds (``WHATSAPP_ACCESS_TOKEN`` / ``WHATSAPP_PHONE_NUMBER_ID``)
    and recipient allowlist (``WHATSAPP_ALLOWED_WAIDS``) as file sending. This is
    the agent-side counterpart to the bridge's ``_send_whatsapp_text`` — used for
    interim progress updates (e.g. video_vision). Best-effort: returns
    ``(ok, error)`` and never raises.
    """
    to = str(to or "").lstrip("+").strip()
    body = str(body or "").strip()
    if not to or not body:
        return False, "missing recipient or body"
    token = _env("WHATSAPP_ACCESS_TOKEN")
    pid = _env("WHATSAPP_PHONE_NUMBER_ID")
    if not token or not pid:
        return False, "WhatsApp not configured (WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID)"
    allowed = _allowed_waids()
    if allowed and to not in allowed:
        return False, f"recipient {to} not in WHATSAPP_ALLOWED_WAIDS"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": body[:4096]},  # Meta caps text bodies at 4096 chars
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_GRAPH_BASE}/{pid}/messages",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            )
    except Exception as exc:
        return False, f"send request failed: {exc}"
    if resp.status_code != 200:
        return False, f"send rejected ({resp.status_code}): {resp.text[:300]}"
    return True, ""


class WhatsAppSendFileTool(Tool):
    """List and deliver a saved file (document, photo, video, audio) to a WhatsApp chat."""

    name = "whatsapp_send_file"
    timeout_seconds = 180.0
    description = (
        "Send a file the agent saved to a WhatsApp chat — photos arrive as "
        "images, MP4s as videos, audio as voice/audio, everything else as a "
        "document. Use this whenever the user wants a file or picture "
        "delivered over WhatsApp — e.g. 'send me that report on WhatsApp', "
        "'whatsapp me the photo', 'get me that word document here'. Set "
        "send_as='document' when they want the original, full-quality file. "
        "By default it sends into the current "
        "WhatsApp chat; pass 'to' (phone number, digits only, no '+') to send "
        "to a specific number. Identify the file by 'path' (e.g. "
        "'showcase/<session>/report.docx'), by 'filename' (fuzzy match), or "
        "set 'latest' for the most recently saved file. If unsure which file "
        "the user means, call action='list' first. WhatsApp only allows "
        "sending within 24 hours of the recipient's last message."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "list"],
                "description": (
                    "'send' delivers a file (default); 'list' returns the "
                    "agent's saved files so you can pick which to send."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Path of the saved file to send, as the agent knows it "
                    "(e.g. 'showcase/<session>/report.docx' or an absolute path)."
                ),
            },
            "filename": {
                "type": "string",
                "description": (
                    "Filename (or part of it) to find among saved files. "
                    "Fuzzy, case-insensitive; newest match wins."
                ),
            },
            "latest": {
                "type": "boolean",
                "description": "Send the most recently saved file.",
            },
            "to": {
                "type": "string",
                "description": (
                    "Recipient WhatsApp number, digits only, no '+'. Omit to "
                    "reply into the current WhatsApp chat."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Optional caption shown under the file or picture (not shown for audio).",
            },
            "send_as": {
                "type": "string",
                "enum": ["auto", "document"],
                "description": (
                    "'auto' (default): photos as images, videos as video, audio as audio. "
                    "'document': the original file, never re-compressed by WhatsApp."
                ),
            },
        },
        "required": [],
    }

    # ── entry ─────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action") or "send").strip().lower()
        saved_base = self._saved_base(kwargs)

        if action == "list":
            files = self._scan(saved_base)
            if not files:
                return ToolResult(success=True, content="No saved files found.")
            lines = [
                f"- {logical}  ({p.stat().st_size} bytes)"
                for p, logical in files[:50]
            ]
            return ToolResult(
                success=True,
                content="Saved files (newest first):\n" + "\n".join(lines),
            )

        return await self._send(kwargs, saved_base)

    # ── send ──────────────────────────────────────────────────────────

    async def _send(self, kwargs: dict[str, Any], saved_base: Path | None) -> ToolResult:
        token = _env("WHATSAPP_ACCESS_TOKEN")
        pid = _env("WHATSAPP_PHONE_NUMBER_ID")
        if not token or not pid:
            return ToolResult(
                success=False,
                error="WhatsApp not configured (WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID).",
            )

        path_arg = str(kwargs.get("path") or "").strip()
        filename = str(kwargs.get("filename") or "").strip()
        latest = bool(kwargs.get("latest"))
        to = str(kwargs.get("to") or "").lstrip("+").strip()
        caption = str(kwargs.get("caption") or "").strip()

        if not path_arg and not filename and not latest:
            return ToolResult(
                success=False,
                error=(
                    "Specify which file to send: 'path', 'filename', or "
                    "latest=true. Use action='list' to see available files."
                ),
            )

        # Resolve recipient: explicit 'to' wins, then the chat this turn
        # answers, else the session's WhatsApp chat.
        if not to:
            to = reply_to(kwargs.get("_agent")).lstrip("+").strip()
        if not to and turn_is_automated(kwargs.get("_agent")):
            return ToolResult(
                success=False,
                error=("No recipient: this automated turn has no WhatsApp chat to answer — "
                       "pass 'to' (digits only) if the file should go to a number."),
            )
        if not to:
            session = kwargs.get("_session")
            meta = getattr(session, "metadata", None) if session is not None else None
            if isinstance(meta, dict):
                to = str(meta.get("whatsapp_waid") or "").lstrip("+").strip()
        if not to:
            return ToolResult(
                success=False,
                error=(
                    "No recipient: this conversation isn't a WhatsApp chat and "
                    "no 'to' number was provided."
                ),
            )
        # Resolve the file on the agent's filesystem.
        resolved = self._resolve_file(kwargs, saved_base, path_arg, filename, latest)
        if resolved is None:
            hint = path_arg or filename or "latest"
            return ToolResult(
                success=False,
                error=f"Could not find a saved file for '{hint}'. Try action='list'.",
            )
        ok, kind, err = await send_whatsapp_media(
            to, resolved, caption,
            as_document=str(kwargs.get("send_as") or "").strip().lower() == "document",
        )
        if not ok:
            return ToolResult(success=False, error=err)
        mark_sent(kwargs.get("_agent"), resolved)
        shown = {"image": "photo", "video": "video", "audio": "audio"}.get(kind, "document")
        return ToolResult(success=True, content=f"Sent '{resolved.name}' to WhatsApp {to} as a {shown}.")

    # ── filesystem resolution ──────────────────────────────────────────

    @staticmethod
    def _saved_base(kwargs: dict[str, Any]) -> Path | None:
        base = kwargs.get("_saved_base_path")
        if base:
            return Path(base)
        runtime = kwargs.get("_runtime_base_path")
        return (Path(runtime) / "saved") if runtime else None

    def _resolve_file(
        self,
        kwargs: dict[str, Any],
        saved_base: Path | None,
        path_arg: str,
        filename: str,
        latest: bool,
    ) -> Path | None:
        # 1. Explicit path: try the file registry, then plausible bases.
        if path_arg:
            registry = kwargs.get("_file_registry")
            if registry is not None:
                try:
                    physical = registry.resolve(path_arg)
                except Exception:
                    physical = None
                if physical and Path(physical).is_file():
                    return Path(physical)
            cand = Path(path_arg).expanduser()
            if cand.is_absolute() and cand.is_file():
                return cand
            stripped = path_arg[len("saved/"):] if path_arg.startswith("saved/") else path_arg
            for base in (saved_base, kwargs.get("_runtime_base_path")):
                if base:
                    for rel in (path_arg, stripped):
                        p = (Path(base) / rel).resolve()
                        if p.is_file():
                            return p
            # Fall through to filename match on the basename.
            filename = filename or Path(path_arg).name

        files = self._scan(saved_base)
        if not files:
            return None
        # 2. Fuzzy filename match (newest first).
        if filename:
            fl = filename.lower()
            exact = [p for p, _ in files if p.name.lower() == fl]
            if exact:
                return exact[0]
            sub = [p for p, _ in files if fl in p.name.lower()]
            if sub:
                return sub[0]
            return None
        # 3. Latest overall.
        if latest:
            return files[0][0]
        return None

    @staticmethod
    def _scan(saved_base: Path | None) -> list[tuple[Path, str]]:
        """Return (path, logical_path) for saved files, newest first."""
        if not saved_base or not Path(saved_base).is_dir():
            return []
        base = Path(saved_base)
        out: list[tuple[float, Path, str]] = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            try:
                mtime = p.stat().st_mtime
                logical = str(p.relative_to(base))
            except (OSError, ValueError):
                continue
            out.append((mtime, p, logical))
        out.sort(key=lambda t: t[0], reverse=True)
        return [(p, logical) for _, p, logical in out]

    # ── Meta Cloud API ─────────────────────────────────────────────────

    @staticmethod
    async def _meta_upload(
        token: str, pid: str, blob: bytes, filename: str, mime: str
    ) -> tuple[str, str]:
        url = f"{_GRAPH_BASE}/{pid}/media"
        files = {
            "file": (filename or "file", blob, mime),
            "messaging_product": (None, "whatsapp"),
            "type": (None, mime),
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url, headers={"Authorization": f"Bearer {token}"}, files=files
                )
        except Exception as exc:
            return "", f"upload request failed: {exc}"
        if resp.status_code != 200:
            return "", f"upload rejected ({resp.status_code}): {resp.text[:300]}"
        try:
            return str((resp.json() or {}).get("id") or ""), ""
        except Exception:
            return "", "upload returned no id"

    @staticmethod
    async def _meta_send(
        token: str, pid: str, to: str, media_id: str, filename: str, caption: str,
        kind: str = "document",
    ) -> tuple[bool, str]:
        body: dict[str, Any] = {"id": media_id}
        if kind == "document":
            body["filename"] = filename or "file"
        if caption and kind != "audio":          # audio messages carry no caption
            body["caption"] = caption[:1024]
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": kind,
            kind: body,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{_GRAPH_BASE}/{pid}/messages",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except Exception as exc:
            return False, f"send request failed: {exc}"
        if resp.status_code != 200:
            return False, f"send rejected ({resp.status_code}): {resp.text[:400]}"
        return True, ""
