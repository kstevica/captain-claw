"""Case endings after restoring (``docs/data-room-plan.md`` P7).

A token comes back as the name's stored form, so Claude's "s ⟦PERSON_3⟧"
becomes "s Ivan Horvat" where Croatian wants "s Ivanom Horvatom". For a
Croatian delivery, the trusted model (on the deck: it may see real names)
gives each restored person's name in the case its sentence needs:

* **Input:** one item per restored name, its sentence with the name marked
  «like this» — never more than the name's own surroundings.
* **Output:** the name, in the right case (JSON, id → form).
* **Guard:** a form is taken only when every word is the same word with a
  Croatian case ending — same stem (a fleeting a, Pavao → Pavla and Luka →
  Luci allowed), same capitals, initials untouched, no word added or dropped.
  Anything else keeps the name as stored.

Only people: a company's legal name or a place stays as written. The language
is told from Claude's own text (in tokens: the real names, often Croatian even
in an English memo, don't count) — diacritics and common words.

It runs after the delivery is in, so Claude's call doesn't wait on a local
model: the plain restore is there at once, the fixed one joins it when the
model is done ("fixing case endings…" until then). Both are kept; the owner
picks either. A deck restart picks up what was waiting.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from captain_claw.anonymize import engine
from captain_claw.logging import get_logger

log = get_logger(__name__)

Edit = tuple[int, int, str]
Call = Callable[[str, str], Awaitable[tuple[str, int]]]

BATCH = 40                 # names per call to the trusted model
MAX_NAMES = 600            # different (sentence, name) pairs per delivery; more stay as stored
BEFORE, AFTER = 160, 100   # characters of the name's surroundings the model sees
OFFICE = ("docx", "xlsx", "pptx")
TEXT_FILES = ("md", "markdown", "txt", "csv", "tsv", "json")

# ── the language ────────────────────────────────────────────────────

# Words that are common in one language and not words in the other ("i",
# "to", "on", "a" are both, so they don't count).
_HR = frozenset("""je u na se za da od su sa iz do po kod koji koja koje kojeg kojem što nije nisu kao ili te
    bi biti bio bila bilo može mogu prema nakon prije između još samo ali već sve svi ovaj ova ovo taj ta
    njegov njezin njihov godine godina ugovor ugovora društvo društva tvrtke tvrtka uprave predsjednik
    direktor iznos iznosi kuna eura sastanak sastanku dana mjesec također odnosno ugovoru društvu tvrtki
    smo sam ste će ćemo kojim kojoj kojih njime njoj""".split())
_EN = frozenset("""the and of is in for with that are was were be by this as at from or an it which will has
    have had not but their its they he she his her been would should could into than about after before
    company agreement board meeting director amount""".split())
_DIACRITICS = re.compile(r"[čćžšđČĆŽŠĐ]")
_WORDS = re.compile(r"[^\W\d_]+")


def language(text: str) -> str:
    """"hr", "en" or "" (can't tell) for Claude's text in tokens."""
    text = engine.BARE_TOKEN_RE.sub(" ", engine.TOKEN_RE.sub(" ", str(text or "")))[:400_000]
    words = [w.casefold() for w in _WORDS.findall(text)]
    hr = sum(1 for w in words if w in _HR)
    en = sum(1 for w in words if w in _EN)
    marks = len(_DIACRITICS.findall(text))
    if hr >= 3 and hr > en:
        return "hr"
    if marks >= 3 and hr >= 1 and hr >= en:
        return "hr"
    return "en" if en >= 3 else ""


# ── which restored values are a person's name ───────────────────────

_TOKEN_KIND = re.compile(r"^[⟦\[]*([A-Z][A-Z0-9_]*?)_\d")
_NAME = re.compile(r"[^\W\d_](?:[^\W\d_]|['’.-])*(?: [^\W\d_](?:[^\W\d_]|['’.-])*){0,5}")


def _kind(token_text: str) -> str:
    m = _TOKEN_KIND.match(token_text)
    return m.group(1) if m else ""


def _name_like(value: str) -> bool:
    """A name as people write it in a sentence — not a link's slug ("ivan-horvat")."""
    return bool(value) and value[0].isupper() and bool(_NAME.fullmatch(value))


def _in_link(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start else " "
    after = text[end] if end < len(text) else " "
    return before in "/@=#\\" or after in "@/\\"


def person_count(text: str, state: engine.VaultState) -> int:
    """Restored people's names in *text* (tokens) the pass would look at."""
    n = 0
    edits, _, _ = engine.restore_edits(text, state)
    for s, e, v in edits:
        if _kind(text[s:e]) == "PERSON" and _name_like(v) and not _in_link(text, s, e):
            n += 1
    return n


# ── the guard ───────────────────────────────────────────────────────

_ENDINGS = frozenset({"a", "e", "i", "o", "u", "om", "em", "oj", "ju", "ja", "je", "ji", "jom", "jem"})
_VOWELS = "aeiou"
_LETTERS = re.compile(r"[^\W\d_]+")
_PALATAL = {"k": "c", "g": "z", "h": "s"}


def _stems(word: str) -> list[tuple[str, frozenset[str]]]:
    """The stems a case form of *word* is built on, with the endings each takes."""
    out = [(word, _ENDINGS)]
    last = word[-1].lower()
    if last in _VOWELS and len(word) > 2:
        out.append((word[:-1], _ENDINGS))                       # Marko → Mark-a, Ana → An-om
    if len(word) > 3 and word[-2:].lower() in ("ar", "ac", "aj"):
        out.append((word[:-2] + word[-1], _ENDINGS))            # Petar → Petr-a, Juraj → Jurj-a
    if len(word) > 4 and word[-2:].lower() == "ao":
        out.append((word[:-2] + ("L" if word[-1].isupper() else "l"), _ENDINGS))   # Pavao → Pavl-a
    if len(word) > 3 and last == "a" and word[-2].lower() in _PALATAL:
        c = _PALATAL[word[-2].lower()]
        out.append((word[:-2] + (c.upper() if word[-2].isupper() else c), frozenset({"i"})))  # Luka → Luci
    return out


def _word_ok(old: str, new: str) -> bool:
    if old == new:
        return True
    if len(old) < 2 or not _LETTERS.fullmatch(old) or not _LETTERS.fullmatch(new):
        return False  # initials, numbers, punctuation: never changed
    if old.isupper():
        if not new.isupper():
            return False
        old, new = old.lower(), new.lower()
    for stem, endings in _stems(old):
        if new.startswith(stem) and new[len(stem):] in endings:
            return True
    return False


def guard(old: str, new: Any) -> bool:
    """Whether *new* is *old* with only Croatian case endings changed."""
    if not isinstance(new, str):
        return False
    old_words, new_words = old.split(), new.split()
    if not new_words or len(old_words) != len(new_words) or " ".join(new_words) != new.strip():
        return False
    for o, n in zip(old_words, new_words):
        o_parts, n_parts = o.split("-"), n.split("-")
        if len(o_parts) != len(n_parts):
            return False
        if not all(_word_ok(a, b) for a, b in zip(o_parts, n_parts)):
            return False
    return True


# ── asking the trusted model ────────────────────────────────────────

SYSTEM = (
    "You put Croatian personal names into the grammatical case their sentence needs. Each item has a "
    "sentence with one name marked «like this», and the name as it is stored.\n"
    "Answer with one JSON object and nothing else: {\"<id>\": \"<the name in the right case>\", …}, one "
    "entry for every id.\n"
    "Which case — look at the word right before the name and at the name's role:\n"
    "- the subject: nominative, unchanged (Ivan Horvat je potpisao; Ana Kovač je CFO)\n"
    "- after s, sa: instrumental (s Ivanom Horvatom, s Anom Kovač)\n"
    "- after o, pri, and u, na, po meaning about or at: locative (o Ivanu Horvatu, o Ani Kovač)\n"
    "- after od, do, iz, kod, bez, protiv, umjesto, nakon, prije, tijekom, and 'of' (ured Ivana Horvata): "
    "genitive (od Ivana Horvata, od Ane Kovač)\n"
    "- after za, kroz, and the direct object (pozvali smo Ivana Horvata): accusative (za Ivana Horvata, "
    "za Anu Kovač)\n"
    "- after prema, k, ka, and 'to someone' (poslali smo Ivanu Horvatu): dative (prema Ivanu Horvatu, "
    "prema Ani Kovač)\n"
    "- someone addressed: vocative (Ivane Horvate, Ana Kovač)\n"
    "Change only the endings of the name's words; a woman's surname that ends in a consonant never changes "
    "(Kovač, Horvat). Keep every word, its spelling and its capital letters; never add, drop or reorder a "
    "word; leave initials as they are. If the name already fits its sentence, or the sentence isn't "
    "Croatian, give the name back unchanged.")


def _context(restored: str, a: int, b: int) -> str:
    before = restored[max(0, a - BEFORE):a]
    nl = before.rfind("\n")
    if nl != -1:
        before = before[nl + 1:]
    elif a > BEFORE and " " in before:
        before = before.split(" ", 1)[1]
    after = restored[b:b + AFTER]
    nl = after.find("\n")
    if nl != -1:
        after = after[:nl]
    elif b + AFTER < len(restored) and " " in after:
        after = after.rsplit(" ", 1)[0]
    return f"{before}«{restored[a:b]}»{after}".strip()


def _answer(text: str) -> dict[str, Any]:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return {}
    try:
        data = json.loads(text[i:j + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class Result:
    edits: list[list[Edit]]     # per text: restore edits, names in their case where the guard let them
    names: int = 0              # restored people's names looked at
    asked: int = 0              # different (sentence, name) pairs sent to the model
    changed: int = 0            # names now in another case
    refused: int = 0            # forms the guard turned down
    left: int = 0               # pairs over MAX_NAMES, kept as stored
    tokens: int = 0


async def inflect(texts: list[str], state: engine.VaultState, call: Call) -> Result:
    """The restore edits of each text (in tokens), people's names put in the case
    their sentence needs. A failed call raises; nothing is half-applied."""
    all_edits: list[list[Edit]] = []
    spots: list[tuple[int, int, str, str]] = []   # (text, edit, sentence, name)
    for ti, text in enumerate(texts):
        edits, _, _ = engine.restore_edits(text, state)
        edits = list(edits)
        all_edits.append(edits)
        pieces, pos, length, places = [], 0, 0, []
        for s, e, v in edits:
            pieces.append(text[pos:s])
            length += s - pos
            places.append((length, length + len(v)))
            pieces.append(v)
            length += len(v)
            pos = e
        pieces.append(text[pos:])
        restored = "".join(pieces)
        for ei, (s, e, v) in enumerate(edits):
            if _kind(text[s:e]) == "PERSON" and _name_like(v) and not _in_link(text, s, e):
                a, b = places[ei]
                spots.append((ti, ei, _context(restored, a, b), v))
    result = Result(all_edits, names=len(spots))
    pairs = list(dict.fromkeys((c, v) for _, _, c, v in spots))
    asked, result.left = pairs[:MAX_NAMES], max(0, len(pairs) - MAX_NAMES)
    result.asked = len(asked)
    forms: dict[tuple[str, str], str] = {}
    for i in range(0, len(asked), BATCH):
        batch = asked[i:i + BATCH]
        items = [{"id": n + 1, "sentence": c, "name": v} for n, (c, v) in enumerate(batch)]
        answer, tokens = await call(SYSTEM, json.dumps(items, ensure_ascii=False))
        result.tokens += tokens
        got = _answer(answer)
        for n, (c, v) in enumerate(batch):
            form = got.get(str(n + 1))
            if not isinstance(form, str) or form.strip() == v:
                continue
            form = form.strip()
            if guard(v, form):
                forms[(c, v)] = form
            else:
                result.refused += 1
    for ti, ei, c, v in spots:
        form = forms.get((c, v))
        if form:
            s, e, _ = all_edits[ti][ei]
            all_edits[ti][ei] = (s, e, form)
            result.changed += 1
    return result


def apply(text: str, edits: list[Edit]) -> str:
    out, pos = [], 0
    for s, e, v in edits:
        out.append(text[pos:s])
        out.append(v)
        pos = e
    out.append(text[pos:])
    return "".join(out)


# ── after a delivery: queue, run, resume ────────────────────────────

_TASKS: dict[tuple[str, int], asyncio.Task] = {}
_LOCKS: dict[int, asyncio.Lock] = {}


def _one_at_a_time() -> asyncio.Lock:
    """One delivery at a time on this loop: the trusted model is often one local GPU."""
    loop = asyncio.get_running_loop()
    return _LOCKS.setdefault(id(loop), asyncio.Lock())


async def first_report(text: str, state: engine.VaultState) -> dict[str, Any] | None:
    """What a new delivery's report says about case endings: None when there's
    nothing to do (not Croatian, or no person's name)."""
    if language(text) != "hr" or not person_count(text, state):
        return None
    from captain_claw.flight_deck import anonymize_trusted as trusted
    from captain_claw.flight_deck.auth import get_db

    if await trusted.usable(get_db()) is None:
        return {"state": "unavailable", "language": "hr"}
    return {"state": "waiting", "language": "hr"}


def queue(room_id: str, seq: int) -> None:
    key = (room_id, seq)
    if key in _TASKS and not _TASKS[key].done():
        return
    _TASKS[key] = asyncio.get_running_loop().create_task(_fix(room_id, seq))


def running(room_id: str, seq: int) -> bool:
    task = _TASKS.get((room_id, seq))
    return task is not None and not task.done()


async def resume_all() -> int:
    """At deck start: deliveries still waiting for their case endings."""
    from captain_claw.flight_deck import rooms_store as rs

    n = 0
    for room_id, seq in await rs.store().grammar_waiting():
        queue(room_id, seq)
        n += 1
    return n


async def _texts_of(room: dict[str, Any], seq: int, payload: dict[str, Any], report: dict[str, Any],
                    cipher: Any) -> tuple[str, list[str], Any]:
    """(what it is, the texts in tokens — the title last, what the file rebuild needs)."""
    from captain_claw.anonymize import formats, office
    from captain_claw.flight_deck import rooms_inbox as inbox

    ext = str(report.get("file") or "")
    title = str(payload.get("title_tokens") or "")
    if not report.get("kind"):
        return "text", [str(payload.get("body_tokens") or ""), title], None
    blob = cipher.decrypt(await asyncio.to_thread(inbox.file_path(room["id"], seq, ext, tokens=True).read_bytes))
    if ext in OFFICE:
        prep = await asyncio.to_thread(office.prepare, blob, ext, scrub=False)
        return "office", [s.text for s in prep.segments] + [title], prep
    if ext in TEXT_FILES:
        return "textfile", [formats.decode_text(blob), title], None
    raise ValueError(f"no case endings for .{ext}")


async def _set(room_id: str, seq: int, at: float, grammar: dict[str, Any], payload_patch: dict[str, Any] | None,
               cipher: Any, payload: dict[str, Any]) -> bool:
    from captain_claw.flight_deck import rooms_store as rs

    d = await rs.store().delivery(room_id, seq)
    if d is None or d["at"] != at:
        return False  # deleted meanwhile (or its number taken by a newer one)
    report = dict(d["report"])
    report["grammar"] = grammar
    blob = None
    if payload_patch is not None:
        blob = cipher.encrypt(json.dumps({**payload, **payload_patch}, ensure_ascii=False).encode("utf-8")).decode(
            "ascii")
    return await rs.store().update_delivery(room_id, seq, report=report, payload=blob)


async def _fix(room_id: str, seq: int) -> None:
    from captain_claw.flight_deck import anonymize_trusted as trusted
    from captain_claw.flight_deck import rooms_inbox as inbox
    from captain_claw.flight_deck import rooms_job as rj
    from captain_claw.flight_deck import rooms_store as rs
    from captain_claw.flight_deck.auth import get_db

    async with _one_at_a_time():
        room = await rs.store().get_any(room_id)
        d = await rs.store().delivery(room_id, seq)
        if room is None or d is None:
            return
        at, report = d["at"], d["report"]
        lang = (report.get("grammar") or {}).get("language") or "hr"
        try:
            row, cipher, state = await inbox._vault(room)
            payload = inbox.open_payload(cipher, d)
        except Exception as exc:  # noqa: BLE001 — the vault is gone or locked: nothing to fix with
            log.warning("Data room case endings: delivery can't be opened", room=room_id, error=type(exc).__name__)
            return
        model = await trusted.usable(get_db())
        if model is None:
            await _set(room_id, seq, at, {"state": "unavailable", "language": lang}, None, cipher, payload)
            return
        try:
            what, texts, prep = await _texts_of(room, seq, payload, report, cipher)
            result = await inflect(texts, state, trusted.completer(model))
        except Exception as exc:  # noqa: BLE001 — type only: a provider error can quote the text
            log.warning("Data room case endings failed", room=room_id, seq=seq, error=type(exc).__name__)
            await _set(room_id, seq, at, {"state": "failed", "language": lang, "error": type(exc).__name__,
                                          "model": model.label}, None, cipher, payload)
            return
        if result.tokens and model.basis != "local":
            try:
                from captain_claw.flight_deck import usage_quota

                await usage_quota.record_tokens(get_db(), room["owner_id"], result.tokens)
            except Exception:  # noqa: BLE001
                pass
        grammar = {"state": "done", "language": lang, "names": result.names, "changed": result.changed,
                   "refused": result.refused, "left": result.left, "model": model.label}
        patch: dict[str, Any] | None = None
        fixed_path = None
        if result.changed:
            title_fixed = apply(texts[-1], result.edits[-1])
            patch = {"title_fixed": title_fixed}
            if what == "text":
                patch["body_fixed"] = apply(texts[0], result.edits[0])
            else:
                ext = str(report.get("file") or "")
                if what == "office":
                    blob = await asyncio.to_thread(prep.rebuild, result.edits[:-1])
                else:
                    blob = apply(texts[0], result.edits[0]).encode("utf-8")
                fixed_path = inbox.file_path(room_id, seq, ext, fixed=True)
                await asyncio.to_thread(fixed_path.write_bytes, cipher.encrypt(blob))
                stem = title_fixed.rsplit(".", 1)[0] if title_fixed.lower().endswith(f".{ext}") else title_fixed
                patch["body_fixed"] = await asyncio.to_thread(rj.text_view, inbox.safe_filename(stem, ext), blob)
        if not await _set(room_id, seq, at, grammar, patch, cipher, payload) and fixed_path is not None:
            fixed_path.unlink(missing_ok=True)
        log.info("Data room case endings", room=room_id, seq=seq, names=result.names, changed=result.changed,
                 refused=result.refused)
