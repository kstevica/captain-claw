"""Pictures a reply hands the person — so the web chat shows them and the
channel the message came from (WhatsApp, Telegram, a device) gets them.

Picture-making tools (image_gen, visualize, termux, payments) report their
files on a ``Path:`` line, and every surface already shows those. A reply can
also hand over a picture made some other way: a plotting script the agent ran
wrote ``saved/charts/q3.png`` and the reply says "here's the chart: q3.png",
or the reply says "the chart is ready" and names no file at all. The web chat
showed nothing then, and channels found only files under
``saved/<category>/<session>/`` named by their whole name.

Two steps, cheapest first, over the pictures this turn wrote under ``saved/``
that no picture-making tool reported:

1. **Names in the reply.** Each reference ending in a picture extension —
   ``saved/…/x.png``, an absolute path, ``![](x.png)``, a bare ``x.png`` — is
   matched to one of those pictures: by its path (a relative one against the
   workspace and ``saved/``), else by its file name.
2. **Jev**, only when the reply names none of them: one call asks whether the
   reply hands the person a picture (``noul``) and, when the turn made more
   than one, which file it is (``choice`` over their paths — the name carries
   the type, png or jpg, so Jev picks the file itself). Only when Jev is
   usable (the owner's key in Flight Deck, or a standalone key), never for a
   shared-agent member's turn, and never in the way: no key, a timeout or an
   error means no pick. Off with ``tools.jev.find_images: false``.

Only deliverables (:mod:`turn_deliverables`): inside ``saved/``, not a working
folder, nothing hidden, written during this turn, and never a file the person
attached to this turn. Each surface drops what it already showed or sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from captain_claw import jev_client
from captain_claw import turn_deliverables as _deliverables
from captain_claw.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})

# Tools that report pictures made FOR the person; every surface handles those.
_PICTURE_TOOLS = frozenset({"image_gen", "visualize", "termux", "payments"})

# How many pictures one reply may hand over, and how many new pictures Jev
# chooses among (the newest).
MAX_IMAGES = 6
_MAX_CHOICES = 24
# Bounds the walk of saved/ (files and folders looked at).
_MAX_WALK_ENTRIES = 50_000

# p(yes) at or above this means the reply hands over a picture; the chosen
# file's probability at or above PICK_THRESHOLD names it. A wrong yes posts an
# unasked picture into the chat, so both sit well above a coin toss.
THRESHOLD = 0.7
PICK_THRESHOLD = 0.5
TIMEOUT_SECONDS = 12.0
_REPLY_CHARS = 6000
_REQUEST_CHARS = 1500

SHOWS = "shows_picture"
PICK = "picture_file"
_NONE = "none"

# `a name with spaces.png` in backticks, or a path-like token ending in a
# picture extension (a URL's scheme is captured so it can be skipped).
_REF_RE = re.compile(
    r"`([^`\n]*?\.(?:png|jpe?g|gif|webp))`"
    r"|(?<![\w/.~%@+-])((?:[a-z][a-z0-9+.-]*://)?[\w/.~%@+-]*?\.(?:png|jpe?g|gif|webp))(?![\w-])",
    re.IGNORECASE,
)
_ATTACHED_RE = re.compile(r"\[Attached [^:\]\n]*:\s*([^\]\n]+)\]")


def enabled() -> bool:
    """``tools.jev.find_images`` (default on)."""
    try:
        from captain_claw.config import get_config

        return bool(getattr(get_config().tools.jev, "find_images", True))
    except Exception:
        return True


def references(reply: str) -> list[str]:
    """Picture references in the reply, in order, without repeats: paths and
    bare file names ending in a picture extension; web addresses skipped."""
    out: list[str] = []
    for match in _REF_RE.finditer(str(reply or "")):
        ref = (match.group(1) or match.group(2) or "").strip()
        if ref.lower().startswith("file://"):
            ref = ref[len("file://"):]
        elif "://" in ref:
            continue
        if "%" in ref:
            ref = unquote(ref)
        if ref and ref not in out:
            out.append(ref)
    return out


def _new_pictures(saved_root: Path, since: float, skip: set[str]) -> list[Path]:
    """Pictures written under ``saved_root`` at or after *since*, newest
    first: deliverables only, none of *skip* (real paths)."""
    found: list[tuple[float, Path]] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(saved_root):
        here = Path(dirpath)
        top = here == saved_root
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and not (top and _deliverables._is_working_dir(saved_root, d))
        ]
        seen += len(dirnames) + len(filenames)
        if seen > _MAX_WALK_ENTRIES:
            log.debug("Reply pictures: saved/ walk cut short", root=str(saved_root))
            break
        for name in filenames:
            if name.startswith(".") or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            path = here / name
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < since:
                continue
            real = Path(os.path.realpath(path))
            if str(real) in skip or not _deliverables.is_deliverable(real, saved_root, since=since):
                continue
            found.append((mtime, real))
    found.sort(key=lambda item: item[0], reverse=True)
    return [p for _m, p in found]


def _match(ref: str, pictures: list[Path], saved_root: Path) -> Path | None:
    """The picture *ref* names: by its path (relative ones against the
    workspace and ``saved/``), else by its file name (the newest)."""
    raw = os.path.expanduser(ref)
    by_path = {str(p): p for p in pictures}
    tries = [Path(raw)] if os.path.isabs(raw) else [saved_root.parent / raw, saved_root / raw]
    for candidate in tries:
        try:
            hit = by_path.get(os.path.realpath(candidate))
        except (OSError, ValueError):
            hit = None
        if hit is not None:
            return hit
    name = Path(raw).name.casefold()
    for picture in pictures:  # newest first
        if picture.name.casefold() == name:
            return picture
    return None


def _turn_facts(agent: Any, turn_start_idx: int) -> tuple[str, set[str], bool]:
    """The person's request (surface rules removed), the real paths of
    pictures this turn's picture-making tools reported or the person
    attached, and whether the turn ran any tool at all — read from the
    session before any await."""
    from captain_claw import msg_origin
    from captain_claw.platform_adapter import (
        effective_turn_start_idx,
        extract_image_paths_from_tool_output,
    )

    session = getattr(agent, "session", None)
    if session is None:
        return "", set(), False
    start = effective_turn_start_idx(agent, turn_start_idx)
    request = ""
    skip: set[str] = set()
    ran_tools = False
    for i, msg in enumerate(session.messages[start:]):
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "") or "")
        if role == "user":
            if i == 0:
                request = msg_origin.split_surface_block(content)[1].strip()
            for attached in _ATTACHED_RE.findall(content):
                skip.add(os.path.realpath(os.path.expanduser(attached.strip())))
        elif role == "tool":
            ran_tools = True
            tool = str(msg.get("tool_name", "")).strip().lower()
            if tool in _PICTURE_TOOLS and not content.strip().lower().startswith("error"):
                skip.update(str(p) for p in extract_image_paths_from_tool_output(content))
    return request, skip, ran_tools


async def ready() -> bool:
    """Whether Jev may be asked for this turn: on, not a shared-agent
    member's turn (Jev is owner-only) and a usable key. Never raises."""
    if not enabled():
        return False
    try:
        if jev_client._member_call():
            return False
        return bool(await jev_client.refresh_connected())
    except Exception as exc:
        log.debug("Jev picture check unavailable", error=str(exc) or type(exc).__name__)
        return False


def build_questions(options: list[str]) -> dict[str, dict[str, Any]]:
    """The yes/no question, and the file choice when there is more than one."""
    questions: dict[str, dict[str, Any]] = {
        SHOWS: {
            "type": "noul",
            "instructions": (
                "An AI agent just answered the person. During this turn it saved the "
                "picture files listed in new_pictures. Does the reply hand the person "
                "a picture — something it made or fetched for them to look at?"
            ),
            "yes_means": (
                "The reply presents a picture as part of the answer: it says it made, "
                "drew, rendered, plotted, captured or found an image, chart, diagram, "
                "screenshot or photo for the person, or points them to one."
            ),
            "no_means": (
                "The reply hands over no picture: it only answers in words, describes "
                "a picture the person sent, mentions pictures the agent used for its "
                "own work, or says making the picture failed."
            ),
        },
    }
    if len(options) > 1:
        questions[PICK] = {
            "type": "choice",
            "instructions": (
                "Which of the new picture files is the one the reply hands the person? "
                "Judge by what the reply says the picture shows and by the file names."
            ),
            "criteria": {**{name: None for name in options},
                         _NONE: "The reply hands over none of these files."},
        }
    return questions


def _choice(answer: Any, options: list[str]) -> tuple[str, float] | None:
    if not isinstance(answer, dict):
        return None
    choice = str(answer.get("choice") or "").strip()
    if choice not in options:
        return None
    probs = answer.get("probabilities")
    p = probs.get(choice) if isinstance(probs, dict) else None
    if not isinstance(p, (int, float)):
        p = answer.get("confidence")
    return (choice, float(p)) if isinstance(p, (int, float)) else None


async def _ask(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    if jev_client.under_flight_deck():
        return await jev_client.ask_via_flight_deck(state, questions, timeout=TIMEOUT_SECONDS)
    return await jev_client.standalone_api().ask(state, questions)


async def jev_pick(request: str, reply: str, pictures: list[Path], saved_root: Path) -> Path | None:
    """The picture Jev says the reply hands over, or None (not confident,
    off, no key, a member's turn, a timeout or an error). Never raises."""
    if not pictures or not await ready():
        return None
    shown = pictures[:_MAX_CHOICES]
    names = [str(p.relative_to(saved_root)) for p in shown]
    state = {
        "request": request[:_REQUEST_CHARS],
        "reply": str(reply or "")[-_REPLY_CHARS:],
        "new_pictures": names,
    }
    try:
        resp = await asyncio.wait_for(_ask(state, build_questions(names)), TIMEOUT_SECONDS + 3)
    except Exception as exc:  # TimeoutError, JevError, JevInputError, anything
        log.debug("Jev picture check skipped", error=str(exc) or type(exc).__name__)
        return None
    answers = resp.get("answers") if isinstance(resp, dict) else None
    p_shows = jev_client.answer_values(resp).get(SHOWS)
    if not isinstance(p_shows, (int, float)):
        return None
    pick: Path | None = None
    p_pick = 1.0
    if len(names) == 1:
        pick = shown[0] if p_shows >= THRESHOLD else None
    elif p_shows >= THRESHOLD:
        picked = _choice(answers.get(PICK) if isinstance(answers, dict) else None, [*names, _NONE])
        if picked and picked[0] != _NONE and picked[1] >= PICK_THRESHOLD:
            pick, p_pick = shown[names.index(picked[0])], picked[1]
    log.info("Jev picture check", p_shows_picture=round(float(p_shows), 3), candidates=len(names),
             picked=str(pick.relative_to(saved_root)) if pick else "", p_pick=round(p_pick, 3))
    return pick


async def _find(saved_root: Path, request: str, skip: set[str], reply: str, since: float) -> list[Path]:
    pictures = await asyncio.to_thread(_new_pictures, saved_root, since, skip)
    if not pictures:
        return []
    named: list[Path] = []
    for ref in references(reply):
        hit = _match(ref, pictures, saved_root)
        if hit is not None and hit not in named:
            named.append(hit)
    if named:
        return named[:MAX_IMAGES]
    pick = await jev_pick(request, reply, pictures, saved_root)
    return [pick] if pick is not None else []


# One answer per turn for every surface that asks (the web chat and the
# channel the turn came from), so Jev is asked once.
_RESULTS: dict[tuple[Any, ...], tuple[float, asyncio.Future]] = {}
_RESULT_TTL = 600.0


async def find(agent: Any, reply: str, turn_start_idx: int, since: float) -> list[Path]:
    """The pictures this turn's reply hands the person (real paths, at most
    :data:`MAX_IMAGES`), beyond those picture-making tools reported. Never
    raises."""
    session = getattr(agent, "session", None)
    key = (id(agent), str(getattr(session, "id", "") or id(session)), round(float(since or 0), 3),
           hashlib.sha1(str(reply or "").encode("utf-8", "replace")).hexdigest())
    now = time.monotonic()
    for old in [k for k, (at, _f) in _RESULTS.items() if now - at > _RESULT_TTL]:
        _RESULTS.pop(old, None)
    cached = _RESULTS.get(key)
    if cached is not None:
        future = cached[1]
    else:
        saved_root = _deliverables.saved_root_of(agent)
        if saved_root is None or not since or since <= 0 or not str(reply or "").strip():
            return []
        # Read the session now: the next turn may start while this one waits.
        request, skip, ran_tools = _turn_facts(agent, turn_start_idx)
        if not ran_tools:
            return []  # a turn without tools wrote no picture: saved/ isn't walked
        future = asyncio.ensure_future(_find(saved_root, request, skip, reply, since))
        _RESULTS[key] = (now, future)
    try:
        return list(await asyncio.shield(future))
    except Exception as exc:
        log.warning("Reply picture lookup failed", error=str(exc) or type(exc).__name__)
        return []
