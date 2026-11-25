"""Pictures on a borrowed call — ``POST /api/llm/complete`` with images.

Flight Deck borrows an agent's model for work of its own (WhatsApp Web
auto-replies, …) and sometimes that work is about a picture: someone posts a
photo in a group and asks the agent what it shows. A message of the call may
carry ``images``: ``[{"data": <base64>, "mime": "image/jpeg", "label":
"picture 1"}, …]``.

* **A model that sees** (:meth:`Agent.sees_images`) gets them inline, as the
  agent's own chat does: each one is written to a scratch folder for the call,
  rides on :attr:`Message.images`, and its ``[Attached image: …]`` marker
  follows its label in the text — so a server that refuses pictures after all
  has them described by the agent's fallback (llm's refusal heal) instead.
* **A model that doesn't** gets what the agent's vision fallback (a vision
  model, else a multimodal peer) says each one shows, as text under its label,
  or a note that nothing here can see it.

The folder is removed when the call ends (:func:`cleanup`). Pictures are never
logged or kept.
"""

from __future__ import annotations

import base64
import binascii
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_PICTURES = 4
MAX_PICTURE_BYTES = 10 * 1024 * 1024
_EXTENSIONS = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
               "image/webp": ".webp", "image/gif": ".gif"}

INLINE = "inline"        # the model got the pictures themselves
DESCRIBED = "described"  # the vision fallback described them
UNSEEN = "unseen"        # nothing here can see a picture

DESCRIBE_PROMPT = (
    "Describe this picture for someone who can't see it: what it shows, and — when it is "
    "something known (a place, a building, a vehicle, ship or aircraft, an artwork, a product, "
    "a well-known person) — what it most likely is and how sure you are. Quote any visible text."
)
_UNSEEN_NOTE = "(you can't see this picture: nothing here can look at pictures)"


class BadPictures(ValueError):
    """The request's pictures can't be taken (too many, too large, not a picture)."""


@dataclass(frozen=True)
class Picture:
    label: str
    data: bytes
    ext: str


def take(raw_messages: list[Any]) -> dict[int, list[Picture]]:
    """The pictures each message carries, by message index; :class:`BadPictures`
    when they can't be taken. Messages without ``images`` aren't listed."""
    found: dict[int, list[Picture]] = {}
    total = 0
    for index, raw in enumerate(raw_messages or []):
        items = raw.get("images") if isinstance(raw, dict) else None
        if not items:
            continue
        if not isinstance(items, list):
            raise BadPictures("images must be a list")
        pictures = []
        for item in items:
            total += 1
            if total > MAX_PICTURES:
                raise BadPictures(f"at most {MAX_PICTURES} pictures a call")
            if not isinstance(item, dict):
                raise BadPictures("a picture is {data, mime, label}")
            mime = str(item.get("mime") or "").strip().lower()
            ext = _EXTENSIONS.get(mime)
            if ext is None:
                raise BadPictures(f"not a picture type this takes: {mime or 'none'}")
            data = str(item.get("data") or "")
            if len(data) > MAX_PICTURE_BYTES * 4 // 3 + 4:
                raise BadPictures("a picture is too large")
            try:
                raw_bytes = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                raise BadPictures("a picture's data isn't base64") from None
            if not raw_bytes or len(raw_bytes) > MAX_PICTURE_BYTES:
                raise BadPictures("a picture is empty or too large")
            label = " ".join(str(item.get("label") or "").split())[:40] or f"picture {total}"
            pictures.append(Picture(label=label, data=raw_bytes, ext=ext))
        found[index] = pictures
    return found


def _sees(agent: Any) -> bool:
    sees = getattr(agent, "sees_images", None)
    if not callable(sees):
        return False
    try:
        return bool(sees())
    except Exception as exc:  # noqa: BLE001 — a broken check sees nothing
        log.warning("Borrowed call: couldn't tell whether the model sees pictures", error=type(exc).__name__)
        return False


async def _describe(agent: Any, paths: list[str]) -> list[str | None]:
    from captain_claw.web.chat_handler import describe_images_without_vision

    try:
        described = await describe_images_without_vision(agent, paths, DESCRIBE_PROMPT)
    except Exception as exc:  # noqa: BLE001 — the call goes on, the pictures unseen
        log.warning("Borrowed call: describing pictures failed", error=type(exc).__name__)
        return [None] * len(paths)
    by_path = dict(described)
    out: list[str | None] = []
    for path in paths:
        text = str(by_path.get(path) or "").strip()
        out.append(None if not text or text.startswith("(automatic analysis failed") else text)
    return out


async def attach(agent: Any, provider: Any, messages: list[Any],
                 pictures: dict[int, list[Picture]]) -> tuple[list[Any], dict[str, Any], str]:
    """*messages* with their pictures on them — ``(messages, report, folder)``.
    *report* is ``{"count", "seen"}`` for the caller (:data:`INLINE`,
    :data:`DESCRIBED` or :data:`UNSEEN`); *folder* holds the call's files until
    :func:`cleanup`. Nothing to do without pictures: ``(messages, {}, "")``."""
    if not pictures or agent is None:
        return messages, {}, ""
    folder = tempfile.mkdtemp(prefix="cc-borrowed-pictures-")
    out = list(messages)
    sees = _sees(agent)
    count = described = 0
    for index, items in sorted(pictures.items()):
        if index >= len(out):
            continue
        paths = []
        for n, pic in enumerate(items, 1):
            path = Path(folder) / f"m{index}-{n}{pic.ext}"
            path.write_bytes(pic.data)
            paths.append(str(path))
        count += len(items)
        msg = out[index]
        if sees:
            lines = [f"{pic.label}: [Attached image: {path}]" for pic, path in zip(items, paths)]
            out[index] = replace(msg, content=f"{msg.content}\n\n" + "\n".join(lines), images=paths)
            continue
        texts = await _describe(agent, paths)
        described += sum(1 for t in texts if t)
        lines = [f"[{pic.label} — described by a vision model, you don't see it yourself]\n{text}" if text
                 else f"[{pic.label}] {_UNSEEN_NOTE}" for pic, text in zip(items, texts)]
        out[index] = replace(msg, content=f"{msg.content}\n\n" + "\n\n".join(lines))
    if sees and provider is not None:
        # The copy picks this call's pictures (no marker scan), and a refusal
        # has them described by the agent's fallback.
        try:
            provider.sees_images = True
            if callable(fallback := getattr(agent, "_describe_refused_images", None)):
                provider.describe_refused_images = fallback
        except Exception:
            pass
    seen = INLINE if sees else DESCRIBED if described else UNSEEN
    return out, {"count": count, "seen": seen}, folder


def cleanup(folder: str) -> None:
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
