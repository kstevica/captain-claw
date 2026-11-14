"""What each file of a data room is about, so Claude (and the person) can
pick the right files without reading them all.

After a room's run, the trusted model reads each clean file's text — the
anonymized text Claude reads, tokens in it — and writes:

* one line (level 1), a paragraph (level 2) and a summary of 3–5
  paragraphs (level 3);
* up to 10 keywords and up to 10 topics.

In English (one language for a room of Croatian and English files, and the
one Claude searches in), tokens kept as written. A description is checked
for real values like the text itself; one that holds any is held back. A
token the model wrote that isn't in what it read is taken out: it would name
the wrong party.

A description belongs to the text it was made from (``text_sha``): a file
whose text changed is described again on the next pass; a "Same party?"
merge rewrites the tokens in place instead. Descriptions run in the
background, one file at a time, each file taking the rooms' run slot — a
room's own run goes first. Files waiting are kept in ``rooms.db``
(``about.state == "waiting"``), so a deck restart carries on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

from captain_claw.anonymize import engine
from captain_claw.flight_deck import rooms_store as rs
from captain_claw.logging import get_logger

log = get_logger(__name__)

EXCERPT_CHARS = int(os.environ.get("FD_ROOMS_DESCRIBE_CHARS") or 16000)  # what the model reads of one file
MIN_PIECE = 700            # a sheet, page or slide shown by less than this isn't worth showing
MAX_ONE = 300
MAX_PARAGRAPH = 1500
MAX_SUMMARY = 6000
MAX_TERMS = 10
MAX_TERM = 80
TRIES = 2                  # an answer that can't be read is asked for once more
STOP_WAIT = 10.0
CONTENT = ("one", "paragraph", "summary", "keywords", "topics")
EMPTY_ONE = "No text could be read from this file."

SYSTEM = (
    "You describe one file of an anonymized data room, so that people and other models can tell what it is and "
    "whether to read it, without opening it.\n"
    "Names of people and companies, addresses and ID numbers in it are tokens like ⟦ORG_7⟧ or ⟦PERSON_2:3⟧. "
    "When you name a party, copy its token exactly as written, brackets included. Never guess who a token is, "
    "and never invent a name or a token.\n"
    "Write in English, even when the file is in another language; keep the file's own titles, codes and terms "
    "of art as written (in quotes when they aren't English).\n"
    "Be specific and factual: what kind of document it is, who it is between, from or about, the period or "
    "date, the key amounts, counts and figures, and what it says, decides or shows. Say nothing the file "
    "doesn't say. When parts of a long file are left out (marked), describe what you were shown.\n\n"
    "Answer in exactly this form — these five headings, nothing before or after:\n"
    "ONE LINE: <one sentence of at most 25 words: what the file is>\n"
    "PARAGRAPH: <one paragraph of 3 to 6 sentences>\n"
    "SUMMARY:\n"
    "<3 to 5 paragraphs, separated by blank lines: the content in more detail — parties, dates, figures, "
    "terms, conclusions>\n"
    "KEYWORDS: <up to 10 specific words or short phrases someone would search for, separated by semicolons: "
    "the document type, the main parties' tokens, terms, products, places, periods>\n"
    "TOPICS: <up to 10 broader subjects of 1 to 4 words, separated by semicolons, e.g. payroll; lease "
    "agreement; related-party transactions>")


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def has(about: dict[str, Any] | None) -> bool:
    """Whether *about* holds a description to show."""
    return bool((about or {}).get("one"))


def needs(f: dict[str, Any]) -> bool:
    """A clean file with no description yet, one that failed, or one made from
    text that has changed since."""
    about = f.get("about") or {}
    if about.get("state") in ("failed", "held", "unavailable") or not has(about):
        return True
    return bool(f.get("text_sha")) and about.get("text_sha") != f["text_sha"]


# ── what the model reads ────────────────────────────────────────────


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> tuple[str, bool]:
    """(what the model reads of *text*, whether parts were left out). A long
    file is shown by its outline and the start of its sheets, pages or slides
    — spread over the whole file when there are many — else by its start, a
    piece of its middle and its end."""
    from captain_claw.flight_deck import rooms_job as rj

    text = text.strip()
    if len(text) <= limit:
        return text, False
    pieces = rj.pages_of(text)
    labelled = len(pieces) >= 2 and not pieces[0][0].isdigit()
    if labelled:
        fit = max(1, limit // MIN_PIECE)
        if len(pieces) > fit:
            step = (len(pieces) - 1) / max(1, fit - 1)
            keep = sorted({0, *(round(i * step) for i in range(fit))})[:fit]
        else:
            keep = list(range(len(pieces)))
        each = limit // len(keep)
        parts: list[str] = []
        last = -1
        for i in keep:
            if i - last > 1:
                parts.append(f"[… {i - last - 1} more left out …]")
            body = pieces[i][1].strip()
            parts.append(body[:each] + ("\n[… the rest of this part left out …]" if len(body) > each else ""))
            last = i
        if last < len(pieces) - 1:
            parts.append(f"[… {len(pieces) - 1 - last} more left out …]")
        return "\n\n".join(parts), True
    heads = [ln[:160] for ln in text.splitlines() if re.match(r"#{1,4} \S", ln)]
    outline = ""
    if len(heads) >= 3:
        outline = ("The file's headings:\n" + "\n".join(heads[:60]))[: limit // 6] + "\n\n---\n\n"
    budget = limit - len(outline)
    head, middle = int(budget * 0.6), int(budget * 0.2)
    tail = budget - head - middle
    mid_at = max(head, len(text) // 2 - middle // 2)
    return (outline + text[:head] + "\n\n[… left out …]\n\n" + text[mid_at:mid_at + middle]
            + "\n\n[… left out …]\n\n" + text[-tail:]), True


def _ask(path: str, kind: str, info: dict[str, Any] | None, text: str) -> tuple[str, str, bool]:
    """(the model's message, what it reads of the file, whether parts were left out)."""
    body, cut = excerpt(text)
    bits = [kind.upper() or "a file"]
    if (info or {}).get("sheets"):
        bits.append(f"{info['sheets']} sheet(s)")
    if (info or {}).get("rows"):
        bits.append(f"{info['rows']:,} rows")
    head = (f"File: {path}\nKind: {', '.join(bits)}\nLength: {len(text):,} characters"
            + (" — parts are left out below, where marked" if cut else "") + "\n\n---\n\n")
    return head + body, body, cut


# ── what the model says ─────────────────────────────────────────────

_HEAD_RE = re.compile(
    r"(?im)^[ \t>#*_-]*(one[ -]?line(?: description)?|one-liner|oneliner|paragraph|summary|key ?words|topics)"
    r"(?:[ \t*_]*:|[ \t*_]*$)[ \t*_]*")
_KEYS = {"one": "one", "oneline": "one", "oneliner": "one", "one_line": "one", "line": "one", "paragraph": "paragraph",
         "summary": "summary", "keywords": "keywords", "key_words": "keywords", "topics": "topics"}


def _key(name: str) -> str:
    name = re.sub(r"[\s-]+", "_", name.strip().casefold()).removesuffix("_description")
    return _KEYS.get(name, _KEYS.get(name.replace("_", ""), ""))


def _from_sections(answer: str) -> dict[str, Any]:
    marks = list(_HEAD_RE.finditer(answer))
    out: dict[str, Any] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
        key = _key(m.group(1))
        if key and key not in out:
            out[key] = answer[m.end():end].strip()
    return out


def _from_json(answer: str) -> dict[str, Any]:
    i, j = answer.find("{"), answer.rfind("}")
    if i == -1 or j <= i:
        return {}
    try:
        data = json.loads(answer[i:j + 1])
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = _key(str(k))
        if key and key not in out:
            out[key] = "\n\n".join(map(str, v)) if isinstance(v, list) and key in ("one", "paragraph", "summary") else v
    return out


def _prose(value: Any, limit: int, *, one_line: bool = False) -> str:
    text = re.sub(r"\*\*|__", "", str(value or "")).strip().strip('"').strip()
    if one_line:
        text = re.sub(r"\s+", " ", text)
    else:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: max(cut.rfind(". ") + 1, cut.rfind(" "), limit // 2)].rstrip() + "…"
    return text


def _terms(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(v) for v in value]
    else:
        text = str(value or "")
        sep = ";" if ";" in text else "\n" if "\n" in text.strip() else ","
        items = text.split(sep)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item)
        item = re.sub(r"\s+", " ", item.replace("**", "")).strip().strip("\"'.,").strip()
        if not item or len(item) > MAX_TERM or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        out.append(item)
        if len(out) >= MAX_TERMS:
            break
    return out


def parse(answer: str) -> dict[str, Any] | None:
    """The five parts of the model's answer, tidied; None when it has no one line."""
    answer = re.sub(r"(?is)<think>.*?</think>", "", str(answer or "")).strip()
    got = _from_sections(answer)
    if not got.get("one"):
        got = _from_json(answer) or got
    one = _prose(got.get("one"), MAX_ONE, one_line=True)
    if not one:
        return None
    return {"one": one, "paragraph": _prose(got.get("paragraph"), MAX_PARAGRAPH, one_line=True),
            "summary": _prose(got.get("summary"), MAX_SUMMARY), "keywords": _terms(got.get("keywords")),
            "topics": _terms(got.get("topics"))}


_SQUARE_RE = re.compile(r"\[\[?([A-Z][A-Z0-9_]*?_\d{1,6}(?::[a-z0-9]{1,3})?)\]\]?")


def tidy_tokens(about: dict[str, Any], seen: set[str]) -> tuple[dict[str, Any], int]:
    """Tokens as the room writes them (⟦ORG_7⟧, not [ORG_7] or ORG_7), and the
    ones the model didn't read taken out — a keyword or topic with one is
    dropped, prose gets "[?]". *seen* holds the base tokens it read. Returns
    the description and how many it took out."""
    dropped = 0

    def fix(text: str) -> str:
        def square(m: re.Match[str]) -> str:
            base = "⟦" + m.group(1).split(":")[0] + "⟧"
            return f"⟦{m.group(1)}⟧" if base in seen else m.group(0)

        def bare(m: re.Match[str]) -> str:
            tok = engine.make_token(m.group(1), int(m.group(2)))
            return (tok[:-1] + (f":{m.group(3)}" if m.group(3) else "") + "⟧") if tok in seen else m.group(0)

        text = _SQUARE_RE.sub(square, text)
        return engine.BARE_TOKEN_RE.sub(bare, text)

    def strange(text: str) -> list[str]:
        """Tokens (bracketed any way, or bare) of a party it didn't read."""
        out = [m.group(0) for m in engine.TOKEN_RE.finditer(text)
               if engine.make_token(m.group(1) or m.group(3), int(m.group(2) or m.group(4))) not in seen]
        out += [m.group(0) for m in _SQUARE_RE.finditer(text)
                if m.group(0) not in out and engine.BARE_TOKEN_RE.fullmatch(m.group(1))]
        return out + [m.group(0) for m in engine.BARE_TOKEN_RE.finditer(text)]

    out = dict(about)
    for key in ("one", "paragraph", "summary"):
        text = fix(out.get(key) or "")
        bad = set(strange(text))
        if bad:
            for pattern in (engine.TOKEN_RE, _SQUARE_RE, engine.BARE_TOKEN_RE):
                text = pattern.sub(lambda m: "[?]" if m.group(0) in bad else m.group(0), text)
            dropped += len(bad)
        out[key] = text
    for key in ("keywords", "topics"):
        kept = []
        for term in out.get(key) or []:
            term = fix(term)
            if strange(term):
                dropped += 1
                continue
            kept.append(term)
        out[key] = kept
    return out, dropped


def every_text(about: dict[str, Any]) -> str:
    return "\n".join([about.get("one") or "", about.get("paragraph") or "", about.get("summary") or "",
                      *(about.get("keywords") or []), *(about.get("topics") or [])])


Call = Any  # (system, user) -> (answer, tokens): anonymize_trusted.completer


async def describe(path: str, kind: str, info: dict[str, Any] | None, text: str, call: Call,
                   needle_list: list[tuple[str, Any]]) -> tuple[dict[str, Any], int]:
    """A file's description, from its text in tokens: ``(about, tokens spent)``.
    *about* has ``state`` done, failed (with an ``error``) or held (it held
    a real value: nothing kept)."""
    from captain_claw.anonymize.llm_lane import AnswerCut
    from captain_claw.flight_deck import rooms_job as rj

    now = time.time()
    if len(text.strip()) < 20:
        return {"state": "done", "one": EMPTY_ONE, "paragraph": "", "summary": "", "keywords": [], "topics": [],
                "text_sha": text_sha(text), "at": now, "model": ""}, 0
    user, body, cut = _ask(path, kind, info, text)
    seen = {engine.make_token(m.group(1) or m.group(3), int(m.group(2) or m.group(4)))
            for m in engine.TOKEN_RE.finditer(body + "\n" + path)}
    spent = 0
    got = None
    for _ in range(TRIES):
        try:
            answer, n = await call(SYSTEM, user)
        except AnswerCut:
            return {"state": "failed", "error": "The model's answer ran past its limit.", "at": now}, spent
        except Exception as exc:  # noqa: BLE001 — type only: a provider's error can quote the text
            # The trusted model's own failures carry the error's type alone.
            why = str(exc) if isinstance(exc, RuntimeError) and re.fullmatch(r"\w{1,60}", str(exc)) \
                else type(exc).__name__
            return {"state": "failed", "error": f"The model failed ({why}).", "at": now}, spent
        spent += n
        got = parse(answer)
        if got:
            break
    if not got:
        return {"state": "failed", "error": "The model's answer couldn't be read.", "at": now}, spent
    got, dropped = tidy_tokens(got, seen)
    if await asyncio.to_thread(rj.leaks, every_text(got), needle_list):
        return {"state": "held", "error": "The description held a real name, so it wasn't kept. Describe it again.",
                "at": now}, spent
    return {"state": "done", **got, "text_sha": text_sha(text), "at": now, "cut": cut,
            **({"dropped": dropped} if dropped else {})}, spent


def rewrite(about: dict[str, Any], mapping: dict[str, str], rewrite_text: Any) -> dict[str, Any]:
    """*about* with its tokens written as a "Same party?" merge says."""
    out = dict(about)
    for key in ("one", "paragraph", "summary"):
        if out.get(key):
            out[key] = rewrite_text(out[key], mapping)
    for key in ("keywords", "topics"):
        if out.get(key):
            out[key] = list(dict.fromkeys(rewrite_text(t, mapping) for t in out[key]))
    return out


def public(about: dict[str, Any] | None, *, full: bool = True) -> dict[str, Any] | None:
    """A description as Claude gets it (None when there is none to give)."""
    if not has(about):
        return None
    assert about is not None
    if not full:
        return {"one": about["one"]}
    out = {k: about.get(k) for k in CONTENT}
    if about.get("cut"):
        out["note"] = "Made from parts of a long file."
    return out


# ── running in the background ───────────────────────────────────────

_TASKS: dict[str, asyncio.Task] = {}
_STOP: set[str] = set()
_AGAIN: set[str] = set()
_PROGRESS: dict[str, dict[str, Any]] = {}


def running(room_id: str) -> bool:
    task = _TASKS.get(room_id)
    return task is not None and not task.done()


def progress(room_id: str) -> dict[str, Any] | None:
    """Where describing a room stands, for its page (None when it isn't)."""
    if not running(room_id):
        return None
    return dict(_PROGRESS.get(room_id) or {"done": 0, "failed": 0, "of": 0})


def _start(room_id: str) -> None:
    if running(room_id):
        _AGAIN.add(room_id)  # it looks for waiting files again before it ends
        return
    _STOP.discard(room_id)
    _TASKS[room_id] = asyncio.get_running_loop().create_task(_run(room_id))


async def queue(room: dict[str, Any], *, force: bool = False, file_ids: list[str] | None = None) -> int:
    """Files of *room* to describe — the ones that need it, or (*force*) every
    clean one; only *file_ids* when given. Returns how many wait now."""
    store = rs.store()
    want = set(file_ids) if file_ids is not None else None
    n = 0
    for f in await store.files(room["id"]):
        if f["state"] != "done" or (want is not None and f["id"] not in want):
            continue
        about = f.get("about") or {}
        if about.get("state") != "waiting":
            if not force and not needs(f):
                continue
            await store.set_file(room["id"], f["id"], about={**about, "state": "waiting"})
        n += 1
    if n:
        _start(room["id"])
    return n


async def after_run(room_id: str) -> int:
    """A room's run ended: describe what it made or changed, when a trusted
    model is set up."""
    from captain_claw.flight_deck import anonymize_trusted as trusted
    from captain_claw.flight_deck.auth import get_db

    room = await rs.store().get_any(room_id)
    if room is None or (room.get("settings") or {}).get("describe") is False:
        return 0
    if await trusted.usable(get_db()) is None:
        return 0
    return await queue(room)


async def stop(room_id: str) -> bool:
    """Stop describing a room: files done stay done, the rest go back."""
    task = _TASKS.get(room_id)
    if task is None or task.done():
        await _unwait(room_id)
        return False
    _STOP.add(room_id)
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), STOP_WAIT)
    except (asyncio.CancelledError, TimeoutError, Exception):  # noqa: BLE001 — it tidies itself up
        pass
    return True


async def resume_all() -> int:
    """At deck start: rooms with files still waiting for their description."""
    n = 0
    for room_id in await rs.store().describe_waiting():
        _start(room_id)
        n += 1
    return n


async def _unwait(room_id: str, state: str = "") -> None:
    """Files still waiting go back to what they had (*state* says why, when not stopped)."""
    store = rs.store()
    for f in await store.files(room_id):
        about = f.get("about") or {}
        if about.get("state") != "waiting":
            continue
        kept = {k: v for k, v in about.items() if k != "state" and k != "error"}
        if state:
            kept.update({"state": state, "error": "No trusted model is set up."})
        elif has(about):
            kept["state"] = "done"
        await store.set_file(room_id, f["id"], about=kept)


async def _vault(room: dict[str, Any]) -> list[tuple[str, Any]]:
    from captain_claw.flight_deck import anonymize_vault as av
    from captain_claw.flight_deck import rooms_job as rj

    vaults = av.store()
    row = await vaults.get(room["owner_id"], room["vault_id"])
    if row is None:
        raise rs.RoomError("this room's vault is gone")
    async with vaults.lock(row["id"]):
        state = await vaults.load_state(row)
    return rj.room_needles(state, room)


async def _run(room_id: str) -> None:
    from captain_claw.flight_deck import anonymize_trusted as trusted
    from captain_claw.flight_deck import rooms_job as rj
    from captain_claw.flight_deck.auth import get_db

    store = rs.store()
    prog: dict[str, Any] = {"done": 0, "failed": 0, "of": 0, "started_at": time.time()}
    _PROGRESS[room_id] = prog
    try:
        while room_id not in _STOP:
            _AGAIN.discard(room_id)
            room = await store.get_any(room_id)
            if room is None:
                return
            waiting = [f for f in await store.files(room_id) if (f.get("about") or {}).get("state") == "waiting"]
            if not waiting:
                if room_id in _AGAIN:
                    continue
                return
            model = await trusted.usable(get_db())
            if model is None:
                await _unwait(room_id, "unavailable")
                return
            call = trusted.completer(model)
            needle_list = await _vault(room)
            prog.update({"of": prog["done"] + prog["failed"] + len(waiting), "model": model.label})
            for f in waiting:
                if room_id in _STOP:
                    return
                prog.update({"current": f["id"], "waiting_for_slot": True, "file_started_at": time.time()})
                async with rj._RUN_SLOT:  # a room's run goes first; its texts are final after it
                    prog["waiting_for_slot"] = False
                    prog["file_started_at"] = time.time()
                    cur = await store.file(room_id, f["id"])
                    old = (cur or {}).get("about") or {}
                    if cur is None or old.get("state") != "waiting":
                        continue
                    if cur["state"] != "done":
                        await store.set_file(room_id, f["id"], about={k: v for k, v in old.items() if k != "state"})
                        continue
                    path = store.text_file(room_id, f["id"])
                    text = await asyncio.to_thread(path.read_text, "utf-8") if path.exists() else ""
                    sha = text_sha(text)
                    if cur.get("text_sha") != sha:
                        await store.set_file(room_id, f["id"], text_sha=sha)
                    about, spent = await describe(cur["clean_path"], cur["kind"], cur.get("info"), text, call,
                                                  needle_list)
                about["model"] = about.get("model", model.label)
                if about["state"] == "failed" and has(old):
                    about = {**{k: old[k] for k in (*CONTENT, "text_sha", "cut", "model") if k in old}, **about}
                await store.set_file(room_id, f["id"], about=about)
                await store.index_about(room_id, f["id"], cur["clean_path"], about if has(about) else None)
                prog["done" if about["state"] == "done" else "failed"] += 1
                log.info("Data room file described", room=room_id, file=f["id"], state=about["state"],
                         chars=len(text), cut=bool(about.get("cut")), tokens=spent,
                         seconds=round(time.time() - prog["file_started_at"], 1))
                if spent and model.basis != "local":
                    try:
                        from captain_claw.flight_deck import usage_quota

                        await usage_quota.record_tokens(get_db(), room["owner_id"], spent)
                    except Exception:  # noqa: BLE001
                        pass
    except asyncio.CancelledError:
        if room_id not in _STOP:
            raise  # the deck is shutting down: the waiting files carry on at the next start
        task = asyncio.current_task()
        if task is not None and hasattr(task, "uncancel"):
            task.uncancel()
    except Exception as exc:  # noqa: BLE001 — the waiting files go back; the room says nothing new
        log.warning("Data room describing failed", room=room_id, error=type(exc).__name__)
        _STOP.add(room_id)
    finally:
        _PROGRESS.pop(room_id, None)
        if _TASKS.get(room_id) is asyncio.current_task():
            _TASKS.pop(room_id, None)
    if room_id in _STOP:
        _STOP.discard(room_id)
        await _unwait(room_id)
