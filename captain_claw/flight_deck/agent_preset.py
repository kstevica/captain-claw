"""Agent presets — "your way of working", set up right after a kiosk agent is created.

The New agent dialog has a "set up its way of working next" box; with it
ticked, the agent is created as usual and the setup dialog opens on it. The
person adds instructions and documents (policies, FAQs, procedures, price
lists, templates) — uploaded, or picked from their File library — and may say
what each document is and how the agent should use it.

``POST /fd/agent-preset/documents`` reads each document in full as it is added
(agent_preset_docs), so the dialog shows its pages and what reading it costs
before anything is spent. ``POST /fd/agent-preset/prepare`` then reads the
material in the background: in steps sized to the model's context window
(``step_tokens``), each pulling out what the agent must know and do — rules,
the requests it handles (intents, each with its steps), facts, when to ask
first, output templates, a glossary and a summary per document, every item
with the document and section (page) it came from — then folded without the
model (repeats once) and merged in one call; a longer read keeps more of each
(``limits_for``). Material over the person's
budget (quick, standard, deep) isn't read front to back: its most
instructive sections are, closest to what they asked to focus on
(agent_preset_select). A document marked "knowledge only" (the default for
reference material) isn't read at all. The read is kept on disk beside its
documents, so it goes on with the dialog closed and after Flight Deck
restarts (``resume_reads``); the bell says when it is ready, and the dialog
opens on it (``GET /fd/agent-preset/prepare?agent=``). Nothing reaches the
agent yet: the dialog shows the preset for review, and ``POST
/fd/agent-preset/apply`` gives the agent the items the person kept:

- rules, when-to-ask and templates → a short "Your owner's rules" block after
  its instructions (the full lists in ``knowledge/owner-rules.md``);
- intents → one skill each, in the agent's own skills folder
  (``~/.captain-claw/skills/<name>/SKILL.md`` under its HOME);
- documents → their whole text as Markdown in its workspace's ``knowledge/``
  folder — a long one split into parts small enough for semantic memory to
  index — with an ``INDEX.md`` and the glossary, for this agent only (not
  the owner-wide deep memory);
- spreadsheets (.xlsx, .csv) → also one datastore table per sheet, no model
  involved (agent_preset_tables), so prices, plans and codes are exact
  lookups — through the agent's REST, or at its next start when it is off;
  a spreadsheet given again replaces its tables;
- facts → insights, through the agent's REST once it is up.

A running agent needs no restart for any of it: it re-reads its skills folder,
indexes its workspace on its next message, and the chat sends it the new
instructions. The documents' text waits server-side under the preset's id for
a day, so Apply doesn't upload them again. The agent being set up reads them
with its own model (its ``/api/llm/complete``: its provider, key and context
window), woken first if it is asleep; only when it is off or stops answering
does the owner's model read them (balanced tier first). A model server that
says "not now" (a rate limit, overloaded) is waited out. The tokens count
toward the owner's daily allowance.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from captain_claw.exceptions import is_busy, retry_after
from captain_claw.flight_deck import agent_idle, agent_lessons, file_library
from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.flight_deck import agent_preset_outline as poutline
from captain_claw.flight_deck import agent_preset_select as psel
from captain_claw.flight_deck import agent_preset_tables as ptables
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.flight_deck.file_library import (
    ModelReply,
    _auth_enabled,
    _quota_spent,
    _resolve_tier,
    cut_off,
    model_reply,
    reply_json,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-preset", tags=["agent-preset"])

_PROMPT_FILE = Path(__file__).parent.parent / "instructions" / "agent_preset_system_prompt.md"
_MERGE_PROMPT_FILE = Path(__file__).parent.parent / "instructions" / "agent_preset_merge_prompt.md"

GROUPS = ("rules", "intents", "facts", "escalation", "templates", "glossary", "knowledge")
FACT_CATEGORIES = ("contact", "decision", "preference", "fact", "deadline", "project",
                   "workflow", "feedback", "reference")
# The most of each group a preset keeps from a step's worth of material (and
# what one step, or the merge call, gives); a longer read keeps more
# (``limits_for``), up to CEILINGS. TOTALS: what an agent keeps from all its
# setups together — a book read in passes, several books.
LIMITS = {"rules": 30, "intents": 12, "facts": 40, "escalation": 15, "templates": 8,
          "glossary": 60, "knowledge": 12}
CEILINGS = {"rules": 150, "intents": 30, "facts": 300, "escalation": 30, "templates": 20,
            "glossary": 400, "knowledge": 12}
TOTALS = {g: n * 3 for g, n in CEILINGS.items()}
_BLOCK_NAMES = 30                      # skills or templates the block names; the rest are in their folders

_MAX_FILES = 12
_MAX_INSTRUCTIONS = 20000
_CHAPTERS_CHARS = 1500                 # a selectively read document's chapters, as the model is told them
_STEP_MIN, _STEP_MAX = 4000, 16_000    # tokens of material in one step: with its prompt, a call stays under ~20k
_PROMPT_TOKENS = 3000                  # the prompt around a step's material
_PART_BYTES = 200_000                  # a knowledge file: semantic memory skips files over 256 KB
_MAP_MIN_CHARS = 20_000                # a document this long, with a few headings, gets a section map
_JOB_KEEP = 3600                       # a finished read stays in memory this long (on disk, a day)
_WATCHED_S = 10.0                      # the dialog asked after a read this recently: no bell when it ends
_TIERS = ("balanced", "reason", "fast")  # when the agent can't read: a careful read beats a quick one
_MAX_TOKENS = 8000
_RETRY_TOKENS = 16000
_CALL_TIMEOUT = 300.0
_WAITS = (15, 30, 60, 60, 60, 60)      # seconds between tries while the model's server says "not now" (~5 min)
_WAIT_CAP = 120.0                      # the longest one retry-after is honoured for
_AGENT_READY_S = 90.0                  # an agent just created, or being woken, starting up
_PACE_TPM = 40_000                     # a read's tokens a minute, at most (FD_SETUP_TOKENS_PER_MIN; 0: unpaced)
_PACE_GAP = 3.0                        # seconds between two calls' starts, at least
_RETRY_NUDGE = "Reply now with the JSON object only: no thinking aloud, no prose."
_READ_EFFORT = "off"                   # a thinking model on a step (FD_SETUP_REASONING: a level, or auto)
_LEARN_MIN_CHARS = 8000                # a prompt this long, at least, teaches the read the model's count
_RECUT = (0.87, 1.3)                   # the model's count this far off the steps' estimate: the rest cut again
_TTL = 24 * 3600                       # a prepared preset nobody created waits this long
_BLOCK_CAP = 5000                      # the rules block in the agent's instructions
_MAX_INSTRUCTIONS_TOTAL = 64000        # AgentConfig.fleet_instructions' own cap
_FACTS_READY_S = 90.0
_MAX_DESCRIPTION = 500

# How the person wants a document used: what the model is told, and what the
# agent's knowledge index says. "auto" leaves it to the model.
USAGES = {
    "auto": ("", ""),
    "rules": ("Rules to follow",
              "rules and policies the agent must follow — extract its rules and when to ask first"),
    "procedures": ("Procedures",
                   "procedures — turn what it describes into intents, each with its steps"),
    "reference": ("Reference",
                  "reference material to look things up in — pull out the key facts and summarise it "
                  "well, but do not turn it into rules"),
    "templates": ("Templates",
                  "templates and examples of answers — extract them as templates"),
}


# ── Cleaning what the model (or the dialog) sent ──────────────────────


def _line(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _body(value: Any, limit: int) -> str:
    return str(value or "").replace("\r\n", "\n").strip()[:limit]


def _source(raw: Any) -> dict:
    d = raw if isinstance(raw, dict) else {}
    return {"doc": _line(d.get("doc"), 160) or "instructions", "section": _line(d.get("section"), 160)}


def skill_name(value: Any) -> str:
    """A skill's folder and id: lowercase, hyphens, at most 48 characters."""
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:48].strip("-")


def _strings(raw: Any, limit: int, count: int) -> list[str]:
    out: list[str] = []
    for v in raw if isinstance(raw, list) else []:
        if isinstance(v, dict):  # {"text": …} where a plain string belongs
            v = next((v[k] for k in ("text", "content", "warning", "conflict") if isinstance(v.get(k), str)), "")
        v = _line(v, limit)
        if v and v not in out:
            out.append(v)
    return out[:count]


def limits_for(tokens: int) -> dict[str, int]:
    """What a read of *tokens* tokens of material keeps of each group:
    LIMITS for a step's worth, more as the square root of what was read —
    a long read finds more, though it repeats itself more too — up to
    CEILINGS."""
    grow = max(1.0, math.sqrt(max(0, tokens) / _STEP_MAX))
    return {g: min(CEILINGS[g], round(n * grow)) for g, n in LIMITS.items()}


def normalise(raw: Any, limits: dict[str, int] | None = None) -> dict:
    """A preset with every group present, each item cleaned, clipped and
    deduplicated — the model's reply and the dialog's edits alike — with
    at most *limits* (LIMITS) of each group."""
    data = raw if isinstance(raw, dict) else {}
    caps = limits or LIMITS

    def items(group: str) -> list[dict]:
        got = data.get(group)
        return [i for i in got if isinstance(i, dict)] if isinstance(got, list) else []

    def unique(rows: list[dict], key: str, group: str) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            k = r[key].casefold()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out[:caps[group]]

    intents: list[dict] = []
    names: set[str] = set()
    for i in items("intents"):
        when = _line(i.get("when"), 300)
        steps = _strings(i.get("steps"), 400, 15)
        base = skill_name(i.get("name")) or skill_name(when)
        if not base or not (when or steps):
            continue
        name, n = base, 2
        while name in names:
            name = f"{base[:44]}-{n}"
            n += 1
        names.add(name)
        intents.append({"name": name, "when": when, "steps": steps,
                        "examples": _strings(i.get("examples"), 200, 3), "source": _source(i.get("source"))})

    def taught(r: dict) -> dict:
        # A rule taught by hand or by an exam keeps who taught it, and when.
        return {k: r[k] for k in ("taught_by", "added") if isinstance(r.get(k), (str, int, float)) and r.get(k)}

    facts: list[dict] = []
    for f in items("facts"):
        category = str(f.get("category") or "").strip().lower()
        try:
            importance = max(1, min(10, int(f.get("importance") or 5)))
        except (TypeError, ValueError):
            importance = 5
        facts.append({"content": _line(f.get("content"), 500),
                      "category": category if category in FACT_CATEGORIES else "fact",
                      "importance": importance, "source": _source(f.get("source"))})

    return {
        # ``always``: kept in the agent's instructions (it applies to every conversation); the rest reach
        # it as lessons with the messages they bear on (agent_lessons). Asking first is always, unless unset.
        "rules": unique([{"text": _line(r.get("text"), 400), "always": r.get("always") is True,
                          "source": _source(r.get("source")), **taught(r)}
                         for r in items("rules")], "text", "rules"),
        "intents": intents[:caps["intents"]],
        "facts": unique(facts, "content", "facts"),
        "escalation": unique([{"text": _line(r.get("text"), 400), "always": r.get("always") is not False,
                               "source": _source(r.get("source")), **taught(r)}
                              for r in items("escalation")], "text", "escalation"),
        "templates": unique([{"name": _line(t.get("name"), 80), "body": _body(t.get("body"), 3000),
                              "source": _source(t.get("source"))}
                             for t in items("templates") if str(t.get("body") or "").strip()], "name", "templates"),
        "glossary": unique([{"term": _line(g.get("term"), 80), "meaning": _line(g.get("meaning"), 300),
                             "source": _source(g.get("source"))}
                            for g in items("glossary") if str(g.get("meaning") or "").strip()], "term", "glossary"),
        "knowledge": unique([{"doc": _line(k.get("doc"), 160), "title": _line(k.get("title"), 120),
                              "summary": _line(k.get("summary"), 800)}
                             for k in items("knowledge")], "doc", "knowledge"),
        "conflicts": _strings(data.get("conflicts"), 400, 10),
        "warnings": _strings(data.get("warnings"), 400, 10),
    }


def _salvage(content: str) -> dict:
    """The whole items of each group in a reply that is not valid JSON — one
    cut off at the output limit mid-list, say."""
    text = content or ""
    decoder = json.JSONDecoder()
    found: dict[str, list] = {}
    for group in (*GROUPS, "conflicts", "warnings"):
        m = re.search(r'"%s"\s*:\s*\[' % group, text)
        if not m:
            continue
        pos, got = m.end(), []
        while True:
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text) or text[pos] == "]":
                break
            try:
                value, pos = decoder.raw_decode(text, pos)
            except ValueError:
                break
            got.append(value)
        found[group] = got
    return found


def parse_preset(content: str) -> dict:
    """The preset in a model's reply, cleaned; raises when there is none."""
    try:
        raw = reply_json(content)
    except ValueError:
        raw = _salvage(content)
    preset = normalise(raw)
    if not any(preset[g] for g in GROUPS):
        raise ValueError("the model found nothing to set up")
    return preset


# ── The documents ─────────────────────────────────────────────────────


def _display_name(name: str, taken: set[str]) -> str:
    base = Path(name or "document").name[:120] or "document"
    out, n = base, 2
    while out in taken:
        out = f"{base} ({n})"
        n += 1
    taken.add(out)
    return out


def note(raw: Any) -> dict:
    """The person's note on a document: what it is, and how to use it."""
    d = raw if isinstance(raw, dict) else {}
    usage = str(d.get("usage") or "auto")
    return {"description": _line(d.get("description"), _MAX_DESCRIPTION),
            "usage": usage if usage in USAGES else "auto"}


def _no_text(name: str, err: Any) -> str:
    log.info("agent preset: no text in a document", file=name, error=err)
    return f"{name}: no text could be read from it (a scanned PDF or a picture?), so it was left out."


async def stage_upload(root: Path, uf: UploadFile) -> tuple[dict | None, str]:
    """An upload read in full and staged, or None and what to tell the person."""
    name = Path(uf.filename or "document").name[:120] or "document"
    try:
        raw = await uf.read()
    except Exception:
        raw = b""
    if not raw:
        return None, f"{name}: the file is empty."
    if len(raw) > pdocs.MAX_FILE_BYTES:
        return None, f"{name}: larger than 25 MB, left out."
    text, err = await asyncio.to_thread(pdocs.extract_bytes, raw, Path(name).suffix)
    if not text.strip():
        return None, _no_text(name, err)
    return await asyncio.to_thread(pdocs.stage, root, name, text, "upload", "", raw), ""


async def stage_library(root: Path, file_id: str, user: dict) -> tuple[dict | None, str]:
    """A File library file — only ever one of *user*'s own agents' — read in
    full and staged, or None and what to tell the person."""
    try:
        row, agents = file_library._owned_row(str(file_id or ""), user)
        path = file_library._resolve_path(row)
    except HTTPException:
        row, path = None, None
    if row is None or path is None:
        return None, "A file picked from your library is no longer there, so it was left out."
    name = row["filename"]
    try:
        too_big = path.stat().st_size > pdocs.MAX_FILE_BYTES
    except OSError:
        too_big = False
    if too_big:
        return None, f"{name}: larger than 25 MB, left out."
    text, err = await asyncio.to_thread(pdocs.extract, path)
    if not text.strip():
        return None, _no_text(name, err)
    agent = agents.get(row["agent_slug"], {}).get("name") or row["agent_slug"]
    try:
        raw = await asyncio.to_thread(path.read_bytes) if ptables.is_spreadsheet(name) else None
    except OSError:
        raw = None
    return await asyncio.to_thread(pdocs.stage, root, name, text, "library", agent, raw), ""


# ── Waiting presets (the documents between Prepare and Apply) ─────────


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def _root() -> Path:
    return _data_dir() / ".agent-presets"


def owner_key(owner_id: str) -> str:
    """The folder a person's presets wait in; a deck without sign-in has one."""
    key = re.sub(r"[^A-Za-z0-9_-]", "", str(owner_id or ""))[:64]
    return key if key and _auth_enabled() else "_deck"


def owner_root(owner_id: str) -> Path:
    """A person's folder: their presets, and their staged documents under ``staged/``."""
    return _root() / owner_key(owner_id)


def _preset_dir(owner_id: str, preset_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{24}", str(preset_id or "")):
        return None
    return owner_root(owner_id) / preset_id


def save_docs(owner_id: str, docs: list[dict], archetype: str) -> str:
    """Keep a preset's documents — their whole text, and a spreadsheet's own
    file (``source``) — until it is applied."""
    preset_id = secrets.token_hex(12)
    folder = _preset_dir(owner_id, preset_id)
    assert folder is not None
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    meta = {"created": time.time(), "archetype": archetype, "docs": []}
    for n, d in enumerate(docs):
        file = f"{n}.txt"
        (folder / "docs" / file).write_text(d["text"], encoding="utf-8")
        source = Path(d["source"]) if d.get("source") else None
        kept = ""
        if source is not None and source.suffix.lower() in ptables.TABLE_EXTS and source.is_file():
            kept = f"{n}{source.suffix.lower()}"
            shutil.copyfile(source, folder / "docs" / kept)
        meta["docs"].append({"name": d["name"], "file": file, "chars": len(d["text"]),
                             "pages": d.get("pages") or 0, "read": bool(d.get("read", True)),
                             "origin": str(d.get("origin") or "upload"),
                             **{k: d[k] for k in ("skip", "earlier_summary") if d.get(k)},
                             **({"source": kept} if kept else {}), **note(d)})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return preset_id


def load_docs(owner_id: str, preset_id: str) -> list[dict]:
    """A waiting preset's documents (``name``, ``text``, ``pages``, ``read``,
    the person's ``description`` and ``usage``, a spreadsheet's ``source``
    file); empty when it is gone, expired, or another person's."""
    folder = _preset_dir(owner_id, preset_id)
    if folder is None or not (folder / "meta.json").is_file():
        return []
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if time.time() - float(meta.get("created") or 0) > _TTL:
            return []
        def source(d: dict) -> dict:
            name = str(d.get("source") or "")
            ok = re.fullmatch(r"\d+\.(xlsx|csv)", name) and (folder / "docs" / name).is_file()
            return {"source": folder / "docs" / name} if ok else {}

        return [{"name": d["name"], "text": (folder / "docs" / d["file"]).read_text(encoding="utf-8"),
                 "pages": int(d.get("pages") or 0), "read": bool(d.get("read", True)),
                 "origin": str(d.get("origin") or "upload"), "skip": list(d.get("skip") or []),
                 "earlier_summary": str(d.get("earlier_summary") or ""),
                 **({"coverage": d["coverage"]} if isinstance(d.get("coverage"), dict) else {}),
                 **source(d), **note(d)}
                for d in meta.get("docs") or []
                if re.fullmatch(r"\d+\.txt", str(d.get("file") or ""))
                and (folder / "docs" / d["file"]).is_file()]
    except Exception:
        log.warning("agent preset: could not read a waiting preset", preset=preset_id, exc_info=True)
        return []


def discard(owner_id: str, preset_id: str) -> None:
    folder = _preset_dir(owner_id, preset_id)
    if folder is not None and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def prune(now: float | None = None) -> int:
    """Drop presets nobody applied, and documents nobody read, within a day."""
    now = time.time() if now is None else now
    gone = 0
    root = _root()
    if not root.is_dir():
        return 0
    for owner in root.iterdir():
        if not owner.is_dir():
            continue
        gone += pdocs.prune_staged(owner, now)
        for folder in owner.iterdir():
            if folder.name == "staged":
                continue
            try:
                created = json.loads((folder / "meta.json").read_text(encoding="utf-8")).get("created")
            except Exception:
                created = folder.stat().st_mtime if folder.exists() else now
            if now - float(created or 0) > _TTL:
                shutil.rmtree(folder, ignore_errors=True)
                gone += 1
    return gone


# ── Reading: the material in steps, then the parts merged ─────────────


def step_tokens(tier: dict) -> int:
    """How much material one step gives the model: half its context window,
    less the prompt around it, within 4k–16k tokens — small calls, which a
    provider's per-minute caps take in their stride. A tier that doesn't
    say its window is taken at 32k (Ollama) or 128k."""
    ctx = int(tier.get("input_ctx") or 0) or (32768 if tier.get("provider") == "ollama" else 128000)
    return max(_STEP_MIN, min(_STEP_MAX, ctx // 2 - _PROMPT_TOKENS))


def _retry_tokens(tier: dict) -> int:
    ctx = int(tier.get("input_ctx") or 0) or (32768 if tier.get("provider") == "ollama" else 128000)
    return min(_RETRY_TOKENS, max(_MAX_TOKENS, ctx - step_tokens(tier) - _PROMPT_TOKENS))


def _chapters(text: str) -> str:
    """A document's chapters as one line for the model: its first heading
    level with more than one section (a title above them all isn't one)."""
    sections = poutline.outline(text)
    levels = sorted({s["level"] for s in sections})
    level = next((lv for lv in levels if sum(1 for s in sections if s["level"] == lv) > 1), 1)
    titles = [s["title"] for s in sections if s["level"] == level]
    line = ""
    for n, t in enumerate(titles):
        if len(line) + len(t) > _CHAPTERS_CHARS:
            return f"{line}; and {len(titles) - n} more"
        line = f"{line}; {t}" if line else t
    return line


def material_cpt(docs: list[dict]) -> float:
    """The characters a token of the documents to read, about
    (``pdocs.chars_per_token``, each weighed by its length)."""
    texts = [d["text"] for d in docs if d.get("read", True) and d.get("text")]
    chars = sum(len(t) for t in texts)
    if not chars:
        return float(pdocs.CHARS_PER_TOKEN)
    return chars / sum(len(t) / pdocs.chars_per_token(t) for t in texts)


def plan_steps(instructions: str, docs: list[dict], step_tok: int, budget: str = psel.DEFAULT_BUDGET,
               focus: str = "", cpt: float = pdocs.CHARS_PER_TOKEN) -> tuple[list[dict], list[str]]:
    """The steps the model reads the material in — each at most *step_tok*
    tokens, small documents sharing one, a long one across many, the
    instructions in the first — and what the person should know about what
    wasn't read. Only documents marked ``read`` are; all of them reach the
    agent's knowledge either way. Material over the person's *budget* isn't
    read front to back: its most instructive sections are (closest to
    *focus*, when one is given — agent_preset_select), in document order,
    the first of each document with its chapters, so the model knows what
    the whole of it covers. *cpt*: the material's characters a token
    (``material_cpt``)."""
    limit = int(step_tok * cpt)
    budget_chars = int(psel.BUDGETS.get(budget, psel.BUDGETS[psel.DEFAULT_BUDGET]) * cpt)
    reading = [d for d in docs if d.get("read", True)]
    notes: list[str] = []
    pieces: list[dict] = []
    # A document the agent already keeps, read again: only its sections not read before.
    if sum(len(d["text"]) for d in reading) <= budget_chars and not any(d.get("skip") for d in reading):
        for d in reading:
            at = 0
            parts = pdocs.split_text(d["text"], limit)
            for n, p in enumerate(parts, 1):
                pieces.append({"doc": d, "text": p["text"], "label": p["label"], "part": n, "parts": len(parts),
                               "start": at, "end": at + len(p["text"])})
                at += len(p["text"])
    else:
        # Sections in pieces that fit a step, and the budget several times over.
        picked, summary = psel.select(reading, budget_chars, focus,
                                      max_chars=min(psel.UNIT_MAX_CHARS, limit // 2, budget_chars // 4))
        introduced: set[int] = set()
        for u in picked:
            d = reading[u["doc"]]
            pieces.append({"doc": d, "text": d["text"][u["start"]:u["end"]], "label": u["title"],
                           "path": u["path"], "selected": True, "start": u["start"], "end": u["end"],
                           **({"chapters": _chapters(d["text"])} if u["doc"] not in introduced else {})})
            introduced.add(u["doc"])
        why = (f"closest to “{focus.strip()[:80]}” and with the most rules, steps and definitions" if focus.strip()
               else "the ones with the most rules, steps and definitions")
        for d, s in zip(reading, summary):
            more = " more" if s["before"] else ""
            earlier = f" ({s['before']} read before)" if s["before"] else ""
            if not s["sections"]:
                if s["before"] < s["of"]:
                    notes.append(f"{d['name']}: none of it{' that was left' if s['before'] else ''} fitted in the "
                                 "budget, so it wasn't read. It is still kept whole for the agent to search.")
            elif s["chars"] < s["left"]:
                size = f"about {max(1, int(s['chars'] / cpt) // 1000)}k tokens"
                what = (f"{s['sections']}{more} of its {s['of']} sections{earlier}"
                        if s["before"] + s["sections"] < s["of"]
                        else f"about {max(1, round(100 * s['chars'] / s['left']))}% of what was left of it")
                notes.append(f"{d['name']}: read {what}, {size} — {why}. The rest wasn't read, but stays whole "
                             "and searchable for the agent.")
            elif s["before"]:
                notes.append(f"{d['name']}: read the {s['sections']} sections not read before — now all of it is.")

    return _pack(pieces, limit, len(instructions), bool(instructions)), notes


def _pack(pieces: list[dict], limit: int, lead: int = 0, instructions: bool = False) -> list[dict]:
    """*pieces* in steps of at most *limit* characters — the first with
    *lead* characters of instructions in it — each labelled with what it
    reads."""
    steps: list[dict] = []
    cur: list[dict] = []
    size = lead
    for piece in pieces:
        piece_size = len(piece["text"])
        if cur and size + piece_size > limit:
            steps.append({"pieces": cur})
            cur, size = [], 0
        cur.append(piece)
        size += piece_size
    if cur or (instructions and not steps):
        steps.append({"pieces": cur})
    for st in steps:
        names: list[str] = []
        for _, run in itertools.groupby(st["pieces"], key=lambda p: id(p["doc"])):
            run = list(run)
            first = run[0]
            if first.get("selected"):
                more = len(run) - 1
                names.append(f"{first['doc']['name']}, {first['label']}"
                             + (f" and {more} more section{'s' if more > 1 else ''}" if more else ""))
            else:
                names += [p["doc"]["name"] + (f", {p['label']}" if p["parts"] > 1 and p["label"] else "")
                          for p in run]
        st["label"] = "; ".join(names) or "your instructions"
    return steps


def refit(steps: list[dict], first: int, step_tok: int, cpt: float = pdocs.CHARS_PER_TOKEN) -> list[dict]:
    """The steps from *first* (1-based) on, cut again for a reader that
    takes less in one step — a read that went on after a restart with
    another model — or to the model's own count of the material's tokens
    (*cpt*, characters a token): a piece too big in parts, all of them
    packed anew. The steps before *first* stay as they are."""
    limit = int(step_tok * cpt)
    pieces: list[dict] = []
    for st in steps[first - 1:]:
        for p in st["pieces"]:
            if len(p["text"]) <= limit:
                pieces.append(p)
                continue
            cut = pdocs.split_text(p["text"], limit)
            at = int(p.get("start") or 0)
            for n, q in enumerate(cut, 1):
                label = f"{p['label']} ({n}/{len(cut)})" if p.get("selected") else (q["label"] or p["label"])
                pieces.append({**p, "text": q["text"], "label": label, "start": at, "end": at + len(q["text"]),
                               **({"chapters": ""} if n > 1 else {})})
                at += len(q["text"])
    return steps[:first - 1] + _pack(pieces, limit)


async def _archetype(owner_id: str, archetype_id: str) -> dict | None:
    from captain_claw.flight_deck.archetypes import merged_archetypes
    aid = str(archetype_id or "").partition("@")[0]
    if not aid:
        return None
    try:
        from captain_claw.flight_deck.auth import get_db
        archetypes = await merged_archetypes(get_db(), owner_id or None)
    except Exception:
        return None
    return next((a for a in archetypes if a.get("id") == aid), None)


def _doc_head(d: dict, title: str) -> list[str]:
    head = [title]
    if d.get("description"):
        head.append(f"The owner says: {d['description']}")
    hint = USAGES.get(d.get("usage") or "auto", ("", ""))[1]
    if hint:
        head.append(f"Use it as: {hint}")
    return head


def _doc_part(piece: dict) -> str:
    d = piece["doc"]
    title = f"### {d['name']}"
    if piece["parts"] > 1:
        title += f" — {piece['label'] or 'part'} (part {piece['part']} of {piece['parts']} of this document)"
    return "\n".join(_doc_head(d, title)) + "\n" + piece["text"]


def _selected_part(run: list[dict]) -> str:
    """A document's selected sections in one step, under one heading: its
    chapters the first time it comes up, each section with where it sits."""
    d = run[0]["doc"]
    head = _doc_head(d, f"### {d['name']} — selected sections")
    if run[0].get("chapters"):
        head.append(f"Its chapters (the whole document): {run[0]['chapters']}")
    body = [(f"(In: {' › '.join(p['path'])})\n" if p.get("path") else "") + p["text"].strip() for p in run]
    return "\n".join(head) + "\n\n" + "\n\n".join(body)


def _step_material(pieces: list[dict]) -> str:
    out: list[str] = []
    for _, run in itertools.groupby(pieces, key=lambda p: (id(p["doc"]), bool(p.get("selected")))):
        run = list(run)
        out += [_selected_part(run)] if run[0].get("selected") else [_doc_part(p) for p in run]
    return "\n\n".join(out)


def step_prompt(arch: dict | None, instructions: str, step: dict, n: int, total: int) -> str:
    parts: list[str] = []
    if arch:
        parts.append(
            f"## The archetype\nRole: {arch.get('role') or arch.get('id')}\n"
            f"Description: {arch.get('description') or ''}\n\n"
            "Its general procedure (for reference — extract only what the owner's material adds):\n"
            f"{str(arch.get('fleet_instructions') or '')[:3000]}")
    if total > 1:
        parts.append(
            f"## Reading in parts\nThis is part {n} of {total} of the owner's material. Extract only what "
            "is in this part — the parts are merged afterwards. For `knowledge`, give one entry per document "
            "in this part (`doc` is its file name exactly as given, without the part): its `title` is the whole "
            "document's (its own heading, or a name for all of it — not this part's), and its `summary` says "
            "what this part contains, as content (\"Refund rules for EU orders; the returns form\") — never "
            "about parts, sections still to come, or merging.")
    if any(p.get("selected") for p in step["pieces"]):
        parts.append(
            "## Selected sections\nSome documents are too long to read whole, so you are given only their "
            "most instructive sections — the agent keeps all of each document to search. Extract only from "
            "what is given. For such a document's `knowledge`, say what the whole of it covers (its chapters "
            "are listed), not only these sections.")
    if n == 1 and instructions:
        parts.append(f"## The owner's instructions\n{instructions}")
    if step["pieces"]:
        parts.append("## The owner's documents\n" + _step_material(step["pieces"]))
    return "\n\n".join(parts)


def _parse_part(content: str) -> dict | None:
    """A step's reply, cleaned; None when it isn't JSON. A reply with nothing
    in it is a part with nothing to set up, not a failure."""
    try:
        raw = reply_json(content)
    except ValueError:
        raw = _salvage(content)
        if not raw:
            return None
    return normalise(raw)


def combine(parts: list[dict]) -> dict:
    """The parts merged without the model: each group interleaved, so its
    cap keeps some of every part; one knowledge entry per document."""
    out: dict[str, list] = {}
    for g in (*GROUPS, "conflicts", "warnings"):
        lists = [p.get(g) or [] for p in parts]
        out[g] = [x for row in itertools.zip_longest(*lists) for x in row if x is not None]
    by_doc: dict[str, dict] = {}
    for p in parts:
        for k in p.get("knowledge") or []:
            e = by_doc.setdefault(k["doc"], {"doc": k["doc"], "title": "", "summary": []})
            e["title"] = e["title"] or k.get("title") or ""
            if k.get("summary"):
                e["summary"].append(k["summary"])
    out["knowledge"] = [{**e, "summary": " ".join(e["summary"])} for e in by_doc.values()]
    return normalise(out)


@dataclass
class ReadJob:
    """One read of a person's material, running in the background — and on
    disk beside its documents (``job.json``, the plan, each step's part,
    the result), so it outlives the dialog and a restart of Flight Deck."""
    id: str
    owner: str                      # owner_key: the folder it is kept in
    owner_id: str = ""
    agent: str = ""                 # the agent being set up
    agent_name: str = ""
    budget: str = psel.DEFAULT_BUDGET
    focus: str = ""
    docs: list[dict] = field(default_factory=list)  # each document's name and whether it is read
    state: str = "running"          # running | done | error | cancelled
    phase: str = "reading"          # reading | merging
    step: int = 0
    steps: int = 0
    current: str = ""
    tokens: int = 0
    cpt: float = 0.0                # the material's characters a token, as the steps are cut
    seen_chars: int = 0             # prompts the model counted (_learn) …
    seen_tokens: int = 0            # … and its count of them
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    result: dict | None = None
    model: str = ""                 # who reads: the agent's own model, or the owner's
    via: str = ""                   # agent | tiers
    waiting: dict | None = None     # {"why": waking | busy | pacing, ...} while it waits
    notes: list[str] = field(default_factory=list)  # found while reading, for the review
    spec: dict = field(default_factory=dict)        # what it needs to go on: instructions, archetype, warnings
    watched: float = 0.0            # when the dialog last asked after it
    stopping: bool = False          # the person stopped it (a shutdown cancels it too, and keeps it)
    task: asyncio.Task | None = None

    def view(self) -> dict:
        out = {"id": self.id, "state": self.state, "phase": self.phase, "step": self.step,
               "steps": self.steps, "current": self.current, "tokens": self.tokens,
               "started": self.started, "finished": self.finished, "error": self.error,
               "model": self.model, "via": self.via, "waiting": self.waiting, "agent": self.agent,
               "agent_name": self.agent_name, "budget": self.budget, "focus": self.focus, "docs": self.docs}
        if self.result is not None:
            out["result"] = self.result
        return out


_JOBS: dict[str, ReadJob] = {}
_JOB_FILE, _PLAN_FILE, _RESULT_FILE = "job.json", "plan.json", "result.json"
_KEPT_FIELDS = ("id", "owner", "owner_id", "agent", "agent_name", "budget", "focus", "docs", "state", "phase",
                "step", "steps", "current", "tokens", "started", "finished", "error", "model", "via", "notes", "spec")


def _forget_old_jobs(now: float) -> None:
    """Finished reads leave memory after an hour; one waiting for its review
    is still on disk (``_find``) until its documents expire."""
    for jid in [j.id for j in _JOBS.values() if j.finished and now - j.finished > _JOB_KEEP]:
        _JOBS.pop(jid, None)


def _folder(job: ReadJob) -> Path:
    return _root() / job.owner / job.id


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    """Whole or not at all: a restart mid-write leaves the last copy."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_job(job: ReadJob) -> None:
    """Where the read stands, beside its documents (gone with them)."""
    folder = _folder(job)
    if folder.is_dir():
        try:
            _write_json(folder / _JOB_FILE, {k: getattr(job, k) for k in _KEPT_FIELDS})
        except OSError:
            log.warning("agent preset: could not keep the read's state", preset=job.id, exc_info=True)


def _job_from(data: dict) -> ReadJob:
    return ReadJob(**{k: data[k] for k in _KEPT_FIELDS if k in data})


def _find(key: str, preset_id: str) -> ReadJob | None:
    """A person's read: in memory, else the finished one on disk (its result
    with it) — None when it is gone, expired, or not theirs."""
    job = _JOBS.get(str(preset_id or ""))
    if job is not None:
        return job if job.owner == key else None
    if not re.fullmatch(r"[0-9a-f]{24}", str(preset_id or "")):
        return None
    folder = _root() / key / preset_id
    data = _read_json(folder / _JOB_FILE)
    meta = _read_json(folder / "meta.json")
    if not isinstance(data, dict) or not isinstance(meta, dict) or data.get("owner") != key:
        return None
    if time.time() - float(meta.get("created") or 0) > _TTL or data.get("state") != "done":
        return None
    job = _job_from(data)
    job.result = _read_json(folder / _RESULT_FILE)
    if not isinstance(job.result, dict):
        return None
    _JOBS[job.id] = job
    return job


def _save_plan(job: ReadJob, steps: list[dict], notes: list[str], docs: list[dict], step_tok: int,
               cpt: float) -> None:
    """The steps as planned — their text, each piece's document by its place
    in the preset, the step size they were cut to and the characters a
    token they were cut by — so a read that goes on after a restart reads
    the same."""
    index = {id(d): n for n, d in enumerate(docs)}
    plan = {"notes": notes, "step_tokens": step_tok, "cpt": round(cpt, 3), "steps": [
        {"label": st["label"], "pieces": [{**{k: v for k, v in p.items() if k != "doc"}, "doc": index[id(p["doc"])]}
                                          for p in st["pieces"]]} for st in steps]}
    folder = _folder(job)
    if folder.is_dir():
        (folder / "parts").mkdir(exist_ok=True)
        _write_json(folder / _PLAN_FILE, plan)


def _load_plan(job: ReadJob, docs: list[dict]) -> tuple[list[dict], list[str], int, float] | None:
    plan = _read_json(_folder(job) / _PLAN_FILE)
    if not isinstance(plan, dict):
        return None
    try:
        steps = [{"label": st["label"], "pieces": [{**p, "doc": docs[int(p["doc"])]} for p in st["pieces"]]}
                 for st in plan["steps"]]
        step_tok = int(plan.get("step_tokens") or 0)
        cpt = float(plan.get("cpt") or pdocs.CHARS_PER_TOKEN)   # a plan from before: cut at 4
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return steps, [str(n) for n in plan.get("notes") or []], step_tok, cpt


def _save_part(job: ReadJob, n: int, part: dict | None) -> None:
    """A step's part (None: the model's answer couldn't be read)."""
    folder = _folder(job) / "parts"
    if folder.is_dir():
        _write_json(folder / f"{n}.json", {"part": part})


def _done_parts(job: ReadJob) -> dict[int, dict | None]:
    out: dict[int, dict | None] = {}
    folder = _folder(job) / "parts"
    for path in folder.glob("*.json") if folder.is_dir() else []:
        got = _read_json(path)
        if path.stem.isdigit() and isinstance(got, dict):
            out[int(path.stem)] = normalise(got["part"]) if isinstance(got.get("part"), dict) else None
    return out


def _read_state(cov: Any) -> dict:
    """How much of a document is read, for the dialog: ``state`` (none,
    partial, full — unknown for a setup from before this was kept),
    ``sections`` read and ``of``."""
    if not isinstance(cov, dict):
        return {"state": "unknown", "sections": 0, "of": 0}
    keys, of = len(cov.get("keys") or []), int(cov.get("of") or 0)
    state = "full" if cov.get("all") else "partial" if keys else "none"
    return {"state": state, "sections": min(keys, of) if of else keys, "of": of}


def read_coverage(steps: list[dict], read: set[int], docs: list[dict]) -> dict[str, dict]:
    """Per document, by name: which of its sections are read now — by the
    steps in *read* (1-based), with those read before (``skip``) — as
    ``all``, the ``keys`` read and ``of``: its sections."""
    ranges: dict[int, list[tuple[int, int]]] = {id(d): [] for d in docs}
    for n, st in enumerate(steps, 1):
        if n in read:
            for p in st["pieces"]:
                if "start" in p and id(p["doc"]) in ranges:
                    ranges[id(p["doc"])].append((int(p["start"]), int(p["end"])))
    out: dict[str, dict] = {}
    for d in docs:
        keys, of = psel.coverage(d["text"], ranges[id(d)], set(d.get("skip") or ()))
        out[d["name"]] = {"all": len(keys) >= of if of else bool(ranges[id(d)]), "keys": keys, "of": of}
    return out


def _save_coverage(job: ReadJob, cover: dict[str, dict]) -> None:
    """What the read covered, beside its documents — for Apply to keep."""
    path = _folder(job) / "meta.json"
    meta = _read_json(path)
    if not isinstance(meta, dict):
        return
    for d in meta.get("docs") or []:
        if isinstance(d, dict) and d.get("name") in cover:
            d["coverage"] = cover[d["name"]]
    _write_json(path, meta)


async def _tell(job: ReadJob) -> None:
    """The bell, when a read ends with nobody watching it (the dialog
    closed): ready to review, or why it stopped. Never raises."""
    if job.state not in ("done", "error") or time.time() - job.watched < _WATCHED_S:
        return
    name = job.agent_name or "your agent"
    if job.state == "done":
        kind, title = "agent_setup", f"{name}'s way of working is ready to review"
        body = (f"Your material is read ({job.steps} part{'s' if job.steps != 1 else ''}, "
                f"{job.tokens:,} tokens). Open it to keep what's right and apply it — it waits a day.")
    else:
        kind, title, body = "agent_setup_error", f"Reading your material for {name} stopped", job.error
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.autonomy import _norm_user

        await get_db().add_notification(_norm_user(job.owner_id), kind, title[:200], body[:2000],
                                        ref_type="agent_setup" if job.agent else "", ref_id=job.agent)
    except Exception:
        log.debug("agent preset: bell notification failed", exc_info=True)


class _UnreadableError(Exception):
    pass


class _AgentGoneError(Exception):
    """The agent being set up doesn't answer at all — not its model's error."""


class _ModelError(Exception):
    """The agent's model failed a call, as the agent reports it."""

    def __init__(self, message: str, wait: Any = None):
        super().__init__(message)
        self.wait = wait


class _BusyError(Exception):
    """The model's server still said "not now" after every wait."""


# ── Who reads: the agent's own model, else the owner's ────────────────


def _agent_config(slug: str) -> dict:
    """The agent's config.yaml — the model it runs on — or {}."""
    from captain_claw.flight_deck import server as srv
    path = srv.DATA_DIR / slug / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def agent_reader(slug: str, entry: dict) -> dict | None:
    """The agent's own model as the reader: what it runs on (its config,
    else its registry entry), its context window, and whether that is on
    its owner's own keys. None when its model isn't known."""
    cfg = _agent_config(slug)
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    ctx_cfg = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
    provider = str(model_cfg.get("provider") or entry.get("provider") or "")
    model = str(model_cfg.get("model") or entry.get("model") or "")
    if not model:
        return None
    try:
        ctx = int(ctx_cfg.get("max_tokens") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if provider == "ollama" and not model.lower().endswith((":cloud", "-cloud")):
        ctx = min(ctx or 32768, 32768)  # where the agent's local Ollama window is clamped
    return {"agent": slug, "name": str(entry.get("name") or slug), "provider": provider, "model": model,
            "base_url": str(model_cfg.get("base_url") or ""), "input_ctx": ctx,
            "byok": entry.get("key_source") == "byok"}


def pace_tpm(reader: dict) -> int:
    """The tokens a minute a read with *reader* keeps to — so a provider's
    per-minute and hourly caps stay far off — or 0: unpaced (a model on
    this machine has no provider cap to keep off)."""
    try:
        tpm = int(os.getenv("FD_SETUP_TOKENS_PER_MIN") or _PACE_TPM)
    except ValueError:
        tpm = _PACE_TPM
    model, base = str(reader.get("model") or ""), str(reader.get("base_url") or "")
    # Ollama's cloud models go through the local daemon, to its hosted (and capped) servers.
    hosted = model.endswith((":cloud", "-cloud"))
    local = not hosted and (reader.get("provider") == "ollama" or bool(
        re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])[:/]", base)))
    return 0 if local else max(0, tpm)


_NEXT_CALL: dict[str, float] = {}   # a person's model: when its next call may start (_pace)


async def _pace(reader: dict, job: ReadJob, tokens: int) -> None:
    """Wait for this call's turn: calls start no faster than the read's
    tokens a minute allow (``pace_tpm``), and never closer than 3 s — the
    person's reads together, when two agents are set up at once with one
    model."""
    tpm = pace_tpm(reader)
    if not tpm:
        return
    slot = f"{job.owner}|{reader.get('provider') or ''}|{reader.get('model') or ''}"
    while (pause := _NEXT_CALL.get(slot, 0.0) - time.time()) > 0:
        job.waiting = {"why": "pacing", "until": _NEXT_CALL[slot]}
        try:
            await asyncio.sleep(pause)
        finally:
            job.waiting = None
    _NEXT_CALL[slot] = time.time() + max(_PACE_GAP, tokens * 60.0 / tpm)


async def choose_reader(owner_id: str, slug: str) -> tuple[dict | None, str]:
    """Who reads the material, and what the person should know about it:
    the agent being set up, with its own model, when it is running or
    asleep (woken when the read starts); else the owner's tier."""
    tier = await _resolve_tier(owner_id, prefer=_TIERS)
    if not slug:
        return tier, ""
    entry = _own_process(slug, owner_id)
    own = agent_reader(slug, entry)
    if own is None:
        return tier, ""
    if agent_idle.is_available(slug):
        return {**own, "fallback": tier}, ""
    if tier:
        return tier, f"{own['name']} is switched off, so {tier.get('model')} reads your material instead of its own model."
    return None, f"{own['name']} is switched off, and no other model is set up to read with. Switch it on and try again."


async def _agent_up(reader: dict, job: ReadJob) -> bool:
    """Wake the agent if it is asleep, and wait for it to answer (one just
    created is still starting): True when it does."""
    from captain_claw.flight_deck import server as srv
    slug = reader["agent"]
    if agent_idle.is_asleep(slug):
        job.waiting = {"why": "waking", "name": reader["name"]}
        try:
            await agent_idle.wake(slug)
        finally:
            job.waiting = None
    if not srv._process_is_alive(slug):
        return False
    token = str((srv._load_process_registry().get(slug) or {}).get("web_auth") or "")
    port = await _wait_ready(lambda: (srv._load_process_registry().get(slug) or {}).get("web_port") or 0,
                             token, _AGENT_READY_S)
    return bool(port)


def _fall_back(reader: dict, job: ReadJob, note: str) -> bool:
    """Read on with the owner's model (*note* says why, in the review) —
    False when there is none."""
    tier = reader.get("fallback")
    if not reader.get("agent") or not tier:
        return False
    log.warning("agent preset: reading with the owner's model instead", agent=reader["agent"], why=note)
    job.notes.append(note.format(name=reader["name"], model=tier.get("model") or "your model"))
    reader.clear()
    reader.update(tier)
    job.model, job.via = str(tier.get("model") or ""), "tiers"
    return True


async def _count(owner_id: str, tokens: int, byok: bool) -> None:
    """Tokens the agent's model spent for the setup, toward the owner's day
    (the agent doesn't count calls to its /api/llm/complete)."""
    if not (owner_id and _auth_enabled() and tokens):
        return
    try:
        from captain_claw.flight_deck import usage_quota
        from captain_claw.flight_deck.auth import get_db
        await usage_quota.record_tokens(get_db(), owner_id, tokens, byok=byok)
    except Exception:
        log.debug("agent preset: token report failed", exc_info=True)


def read_effort() -> str | None:
    """How hard a thinking model thinks on a reading step or the merge: off —
    it extracts, it needn't deliberate, and its thoughts count against the
    answer's max_tokens (a model that can't stop thinking takes its lowest
    level) — or FD_SETUP_REASONING's level; None ("auto") leaves the model's
    own."""
    from captain_claw.llm import reasoning
    return reasoning.normalize_level(os.environ.get("FD_SETUP_REASONING", "").strip() or _READ_EFFORT)


async def _agent_complete(reader: dict, system: str, prompt: str, owner_id: str,
                          max_tokens: int) -> ModelReply:
    """One call to the agent's own model, through its REST."""
    import httpx

    from captain_claw.flight_deck import server as srv
    slug = reader["agent"]
    entry = srv._load_process_registry().get(slug) or {}
    port, token = int(entry.get("web_port") or 0), str(entry.get("web_auth") or "")
    if not port or not srv._process_is_alive(slug):
        raise _AgentGoneError("not running")
    agent_idle.touch(slug)
    effort = read_effort()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2 * _CALL_TIMEOUT + 30, connect=5.0),
                                     trust_env=False) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/api/llm/complete", params={"token": token} if token else {},
                json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                      "temperature": 0.2, "max_tokens": max_tokens, "timeout": _CALL_TIMEOUT,
                      **({"reasoning_effort": effort} if effort else {})})
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise _AgentGoneError("not answering") from exc
    finally:
        agent_idle.touch(slug)
    try:
        data = resp.json()
    except ValueError:
        data = None
    # 503: no model at all; 401/403/404: not this agent, or one without the route.
    if resp.status_code in (401, 403, 404, 503) or not isinstance(data, dict):
        raise _AgentGoneError(f"HTTP {resp.status_code}")
    if not data.get("ok"):
        raise _ModelError(str(data.get("error") or f"HTTP {resp.status_code}"), data.get("retry_after"))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tokens = int(usage.get("total_tokens") or 0) or (
        int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
    await _count(owner_id, tokens, bool(reader.get("byok")))
    return ModelReply(str(data.get("content") or ""), tokens, int(usage.get("prompt_tokens") or 0),
                      str(data.get("finish_reason") or ""), str(data.get("reasoning") or ""))


async def _call(reader: dict, system: str, prompt: str, owner_id: str, job: ReadJob,
                max_tokens: int) -> ModelReply:
    """One model call: the agent's model, or the owner's tier. A server that
    says "not now" (a rate limit, overloaded) is waited out — for as long as
    it asks, else 15 s, 30 s, then a minute at a time, some 5 minutes in
    all; an agent that stops answering hands over to the owner's model.
    Either thinks as hard as ``read_effort`` says."""
    waits = iter(_WAITS)
    tokens_in = int((len(system) + len(prompt)) / (job.cpt or pdocs.CHARS_PER_TOKEN))
    while True:
        await _pace(reader, job, tokens_in + min(max_tokens, 2000))  # the reply, as one usually runs
        try:
            if reader.get("agent"):
                reply = await _agent_complete(reader, system, prompt, owner_id, max_tokens)
            else:
                from captain_claw.llm import reasoning
                with reasoning.call_effort(read_effort()):
                    reply = await model_reply(reader, system, prompt, owner_id, max_tokens=max_tokens,
                                              request_timeout=_CALL_TIMEOUT)
            job.tokens += int(reply.tokens or 0)
            return reply
        except asyncio.CancelledError:
            raise
        except _AgentGoneError as exc:
            if not _fall_back(reader, job, "{name} stopped answering, so {model} read the rest."):
                raise _UnreadableError(f"{reader.get('name') or 'The agent'} stopped answering ({exc}).") from exc
        except Exception as exc:
            if not is_busy(exc):
                raise
            pause = next(waits, None)
            if pause is None:
                raise _BusyError(str(exc)) from exc
            try:
                asked = float(getattr(exc, "wait", None) or retry_after(exc) or 0)
            except (TypeError, ValueError):
                asked = 0.0
            if asked > 0:
                pause = min(_WAIT_CAP, max(1.0, asked))
            log.info("agent preset: the model is busy, waiting", model=reader.get("model"), seconds=pause)
            job.waiting = {"why": "busy", "model": str(reader.get("model") or ""), "until": time.time() + pause}
            try:
                await asyncio.sleep(pause)
            finally:
                job.waiting = None


class _CutOffError(Exception):
    """The model's answer stopped at its max_tokens, twice, without the JSON:
    what happened, after the model's name ("used up its … thinking")."""


def _cut_why(reply: ModelReply, max_tokens: int) -> str:
    # Its thoughts, or an answer far shorter than its room would hold (some
    # 3–4 characters a token): the rest went on thinking — which not every
    # provider hands back (Ollama's don't reach the reply).
    if reply.reasoning or len(reply.text) < 2 * max_tokens:
        return f"used up its {max_tokens:,} tokens thinking before it finished answering"
    return f"ran past {max_tokens:,} tokens in its answer, which was cut off"


def _learn(job: ReadJob, chars: int, prompt_tokens: int) -> None:
    """A prompt of *chars* characters the model counted as *prompt_tokens*:
    the read cuts its next steps by what it has learned (``learned_cpt``).
    A count far off the estimate — a server that leaves out what it had
    cached — teaches nothing."""
    expected = chars / (job.cpt or pdocs.CHARS_PER_TOKEN)
    if chars >= _LEARN_MIN_CHARS and prompt_tokens > 0 and 0.5 <= expected / prompt_tokens <= 2.0:
        job.seen_chars += chars
        job.seen_tokens += prompt_tokens


def learned_cpt(job: ReadJob) -> float | None:
    """The characters a token of the read's prompts, in the model's own count."""
    if not job.seen_tokens:
        return None
    return min(6.0, max(1.5, job.seen_chars / job.seen_tokens))


async def _ask(reader: dict, system: str, prompt: str, owner_id: str, job: ReadJob) -> dict | None:
    """One step's model call — again, with room and a nudge, when the reply
    isn't JSON (a reasoning model that thought past the budget, prose around
    it). None when it never is; :class:`_CutOffError` when that's because
    the answer was cut off. The first call's prompt, in the model's count,
    teaches the read its tokens (``_learn``)."""
    for attempt in range(2):
        max_tokens = _MAX_TOKENS if attempt == 0 else _retry_tokens(reader)
        reply = await _call(reader, system, prompt if attempt == 0 else f"{prompt}\n\n{_RETRY_NUDGE}",
                            owner_id, job, max_tokens)
        if attempt == 0:
            _learn(job, len(system) + len(prompt), reply.prompt_tokens)
        got = _parse_part(reply.text)
        if got is not None:
            return got
        if cut_off(reply):
            log.warning("agent preset: the answer was cut off", model=reader.get("model"), max_tokens=max_tokens,
                        answer_chars=len(reply.text), thinking_chars=len(reply.reasoning))
    if cut_off(reply):
        raise _CutOffError(_cut_why(reply, max_tokens))
    return None


# What identifies an item when steps found it twice; rules, when-to-ask and
# facts also match in other words (retrieval.same_fact).
_ITEM_KEY = {"rules": "text", "escalation": "text", "facts": "content",
             "intents": "name", "templates": "name", "glossary": "term"}
_REWORDED = ("rules", "escalation", "facts")
_MERGE_SUMMARY_CHARS = 3000   # a document's part summaries, as the merge sees them


def _richer(group: str, kept: dict, other: dict) -> dict:
    """Of two items saying the same, the more specific — more numbers, then
    more words; a skill with more steps, both one's examples."""
    if group == "intents":
        best = kept if len(kept.get("steps") or []) >= len(other.get("steps") or []) else other
        return {**best, "examples": list(dict.fromkeys([*kept.get("examples", []), *other.get("examples", [])]))[:3]}
    field_ = {"templates": "body", "glossary": "meaning"}.get(group, _ITEM_KEY[group])

    def weight(x: dict) -> tuple[int, int]:
        s = str(x.get(field_) or "")
        return len(re.findall(r"\d", s)), len(s)

    best = other if weight(other) > weight(kept) else kept
    if group in ("rules", "escalation") and (kept.get("always") or other.get("always")):
        best = {**best, "always": True}   # one step saw it apply to everything
    return best


class _Alike:
    """The items of one group so far, by their key's text, for telling
    whether another is one of them: the same text, or — rules, when-to-ask
    and facts — the same thing in other words (``retrieval.same_fact``:
    "14 days" and "30 days" stay two)."""

    def __init__(self, group: str) -> None:
        self.reworded = group in _REWORDED
        self.texts: list[str] = []
        self.exact: dict[str, int] = {}
        self.by_stem: dict[str, list[int]] = {}
        self._last: tuple[str, list[str]] = ("", [])   # the stems find() worked out, for add()

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.casefold().split())

    def _stems(self, text: str) -> list[str]:
        from captain_claw import retrieval
        if not self.reworded:
            return []
        if self._last[0] == text:
            return self._last[1]
        mine = list(dict.fromkeys(retrieval.stem(retrieval.fold(t))
                                  for t in retrieval.query_terms(retrieval.uncontract(text), max_terms=32)))
        self._last = (text, mine)
        return mine

    def find(self, text: str) -> int | None:
        """Which item *text* is (its place), or None for a new one; once
        found, the same text finds it again."""
        from captain_claw import retrieval
        norm = self._norm(text)
        hit = self.exact.get(norm)
        mine = self._stems(text) if hit is None else []
        if mine:
            shared: dict[int, int] = {}
            for s in mine:
                for c in self.by_stem.get(s, ()):
                    shared[c] = shared.get(c, 0) + 1
            hit = next((c for c, n in sorted(shared.items(), key=lambda kv: -kv[1])
                        if n >= len(mine) / 2 and retrieval.same_fact(text, self.texts[c])), None)
        if hit is not None:
            self.exact[norm] = hit
        return hit

    def add(self, text: str) -> int:
        """*text* as a new item; its place."""
        at = len(self.texts)
        self.texts.append(text)
        self.exact[self._norm(text)] = at
        for s in self._stems(text):
            self.by_stem.setdefault(s, []).append(at)
        return at


def fold(parts: list[dict]) -> dict:
    """What every step found, as one preset, without the model: an item
    several steps found — a rule or fact in other words too (``_Alike``), a
    skill, a template or a term by its name — is kept once, in its most
    specific wording, and ranks by how many steps found it; ties keep the
    steps' order, interleaved, so a group's cap keeps some of every step.
    Nothing is capped here; ``normalise`` caps."""
    out: dict[str, Any] = {}
    for group, key in _ITEM_KEY.items():
        clusters: list[dict] = []
        alike = _Alike(group)
        for row in itertools.zip_longest(*[p.get(group) or [] for p in parts]):
            for item in (x for x in row if x is not None):
                text = str(item.get(key) or "")
                hit = alike.find(text)
                if hit is None:
                    hit = alike.add(text)
                    clusters.append({"item": item, "count": 0})
                else:
                    clusters[hit]["item"] = _richer(group, clusters[hit]["item"], item)
                clusters[hit]["count"] += 1
        out[group] = [c["item"] for c in sorted(clusters, key=lambda c: -c["count"])]  # stable: ties keep order
    for group in ("conflicts", "warnings"):
        rows = itertools.zip_longest(*[p.get(group) or [] for p in parts])
        out[group] = list(dict.fromkeys(x for row in rows for x in row if x))
    by_doc: dict[str, dict] = {}
    for p in parts:
        for k in p.get("knowledge") or []:
            e = by_doc.setdefault(k["doc"], {"doc": k["doc"], "title": "", "summaries": []})
            e["title"] = e["title"] or k.get("title") or ""
            if k.get("summary") and k["summary"] not in e["summaries"]:
                e["summaries"].append(k["summary"])
    out["knowledge"] = [{**e, "summary": " ".join(e["summaries"])} for e in by_doc.values()]
    return out


def _sampled(summaries: list[str], limit: int) -> str:
    """A document's part summaries within *limit* characters: evenly from
    all of them, not just the first parts'."""
    joined = " ".join(summaries)
    if len(joined) <= limit:
        return joined
    every = max(1, round(len(joined) / limit + 0.5))
    return " ".join(summaries[::every])[:limit]


_MERGED = ("rules", "intents", "facts", "escalation", "glossary")   # the groups the merge call sees


def _merge_payload(folded: dict, room: int, limits: dict[str, int] | None = None) -> tuple[str, dict[str, int]] | None:
    """The folded preset as the merge call sees it, and how many of each
    group that is — a group up to twice LIMITS for the call to keep the best
    of, or LIMITS (also when the read keeps more of it: the rest follows the
    call, ``_top_up``); templates left out (the fold already has them once
    each) — or None when even that is more than *room* characters."""
    limits = limits or LIMITS
    for times in (2, 1):
        view = {g: folded[g][:LIMITS[g] * (times if limits[g] <= LIMITS[g] else 1)] for g in _MERGED}
        view["knowledge"] = [{"doc": k["doc"], "title": k["title"],
                              "summary": _sampled(k["summaries"], _MERGE_SUMMARY_CHARS)} for k in folded["knowledge"]]
        view["conflicts"], view["warnings"] = folded["conflicts"], folded["warnings"]
        payload = json.dumps(view, ensure_ascii=False)
        if len(payload) <= room:
            return payload, {g: len(view[g]) for g in _MERGED}
    return None


def merge_prompt() -> str:
    """The merge call's prompt, with how many of each group it keeps."""
    text = _MERGE_PROMPT_FILE.read_text(encoding="utf-8")
    for g, n in LIMITS.items():
        text = text.replace(f"{{{g}}}", str(n))
    return text


def _top_up(merged: dict, folded: dict, seen: dict[str, int], limits: dict[str, int]) -> dict:
    """The merge call's preset and, after it in each group, what the fold has
    past what the call saw (found by fewer steps), up to *limits*: a long
    read keeps more than one call can settle. One the call already has, in
    its words or the fold's, isn't added twice."""
    out = dict(merged)
    for group in _MERGED:
        key = _ITEM_KEY[group]
        have = list(merged.get(group) or [])
        alike = _Alike(group)
        for item in have:
            alike.add(str(item.get(key) or ""))
        for item in folded[group][seen.get(group, 0):]:
            if len(have) >= limits[group]:
                break
            text = str(item.get(key) or "")
            if alike.find(text) is None:
                alike.add(text)
                have.append(item)
        out[group] = have
    return normalise(out, limits)


async def merge_parts(reader: dict, parts: list[dict], owner_id: str, job: ReadJob,
                      limits: dict[str, int] | None = None, folded: dict | None = None) -> dict:
    """What every step found, as one preset of at most *limits* (LIMITS) of
    each group: folded without the model (``fold``, unless given), then —
    however many steps there were — one call that settles what the fold
    can't (the same thing said very differently, conflicts between steps)
    among what most steps found, keeps the best of it and writes each
    document's summary anew; the rest of the fold follows it (``_top_up``).
    Without that call — too big for one, or an answer that can't be read —
    the fold stands, capped."""
    if len(parts) == 1:
        return parts[0]
    limits = limits or LIMITS
    folded = folded if folded is not None else fold(parts)
    got_view = _merge_payload(folded, int(step_tokens(reader) * (job.cpt or pdocs.CHARS_PER_TOKEN)), limits)
    if got_view is None:
        log.info("agent preset: too much for one merge call, the fold stands", parts=len(parts))
        return normalise(folded, limits)
    payload, seen = got_view
    try:
        got = await _ask(reader, merge_prompt(), f"## What the parts found\n{payload}", owner_id, job)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("agent preset: merging failed, the fold stands", error=str(exc))
        got = None
    if got is None:
        return normalise(folded, limits)
    got["templates"] = folded["templates"]
    return whole_summaries(_top_up(got, folded, seen, limits), parts)


def whole_summaries(merged: dict, parts: list[dict]) -> dict:
    """A merge that kept one part's summary of a document read in several
    parts — a small model copies rather than rewrites — gets every part's
    summary instead, so the agent's index describes all of it."""
    per_doc: dict[str, list[str]] = {}
    for p in parts:
        for k in p.get("knowledge") or []:
            per_doc.setdefault(k["doc"], []).append(k.get("summary") or "")
    joined = {k["doc"]: k["summary"] for k in combine(parts)["knowledge"]}
    for k in merged.get("knowledge") or []:
        summaries = per_doc.get(k["doc"], [])
        if len(summaries) > 1 and k.get("summary") in summaries:
            k["summary"] = joined.get(k["doc"], k["summary"])
    return merged


def _finish(preset: dict, docs: list[dict], warnings: list[str], model: str, steps: int, tokens: int,
            preset_id: str, cover: dict[str, dict] | None = None, found: dict[str, int] | None = None) -> dict:
    """The read's result: every document with a knowledge entry — summarised
    by the model, else by the person's own words; one the model invented a
    name for doesn't get in. ``found``: how many of each group the steps
    found together, before the read's limits (``limits_for``) kept the most
    found."""
    names = [d["name"] for d in docs]
    known: dict[str, dict] = {}
    for k in preset["knowledge"]:
        name = k["doc"] if k["doc"] in names else next((n for n in names if k["doc"].startswith(n)), "")
        if name and name not in known:
            known[name] = {**k, "doc": name}
    preset["knowledge"] = [dict(known.get(n) or {"doc": n, "title": Path(n).stem[:120], "summary": ""})
                           for n in names]
    for k, d in zip(preset["knowledge"], docs):
        k["summary"] = k["summary"] or d["description"] or d.get("earlier_summary") or ""
    preset["warnings"] = (warnings + preset["warnings"])[:20]
    return {
        "id": preset_id,
        "preset": preset,
        "docs": [{"name": d["name"], "chars": len(d["text"]), "pages": d.get("pages") or 0,
                  "pages_exact": bool(d.get("pages_exact")), "truncated": bool(d.get("truncated")),
                  "origin": d.get("origin") or "upload", "description": d["description"], "usage": d["usage"],
                  "read": bool(d.get("read", True)), **({"agent": d["agent"]} if d.get("agent") else {}),
                  **({"tables": int(d["tables"])} if d.get("tables") else {}),
                  **({"coverage": _read_state((cover or {}).get(d["name"]))} if cover else {})}
                 for d in docs],
        "model": model,
        "steps": steps,
        "tokens": tokens,
        **({"found": found} if found else {}),
    }


def read_tokens(steps: list[dict], read: set[int], cpt: float = pdocs.CHARS_PER_TOKEN) -> int:
    """The tokens of material in the steps in *read* (1-based)."""
    return int(sum(len(p["text"]) for n, st in enumerate(steps, 1) if n in read
                   for p in st["pieces"]) / cpt)


async def run_read(job: ReadJob, reader: dict | None, arch: dict | None, instructions: str,
                   docs: list[dict], warnings: list[str], owner_id: str) -> None:
    """Read the material step by step — with the agent's own model, woken
    first if it is asleep, or the owner's when it doesn't answer — merge
    what each step found, and leave the result on *job* for the dialog to
    fetch, and on disk for the review later; with the dialog closed, the
    bell says it's ready. A step whose answer can't be read is reported and
    skipped; the person's allowance running out, or the model's server
    staying busy, stops the reading but keeps what was found. A read that
    Flight Deck stopped mid-way (``resume_reads``) plans nothing again and
    reads no step twice. Steps are cut to the material's tokens as
    estimated (``material_cpt``), then — once the model has said how it
    counts them, and that is well off — the rest are cut again to its
    count."""
    try:
        if reader and reader.get("agent") and not await _agent_up(reader, job):
            if not _fall_back(reader, job, "{name} didn't answer, so {model} read your material "
                                           "instead of its own model."):
                raise _UnreadableError(f"{reader['name']} isn't answering, and no other model is set up "
                                       "to read with. Switch it on and try again.")
        step_tok = step_tokens(reader) if reader else _STEP_MAX
        done = await asyncio.to_thread(_done_parts, job)
        planned = await asyncio.to_thread(_load_plan, job, docs)
        if planned is not None and planned[2] > step_tok:
            # Going on with a model that takes less at once: nothing read yet, plan again; else cut the rest smaller.
            planned = None if not done else (
                refit(planned[0], max(done) + 1, step_tok, planned[3]), planned[1], step_tok, planned[3])
            if planned is not None:
                await asyncio.to_thread(_save_plan, job, planned[0], planned[1], docs, step_tok, planned[3])
        if planned is None:
            cpt = await asyncio.to_thread(material_cpt, docs)
            steps, notes = await asyncio.to_thread(plan_steps, instructions, docs, step_tok, job.budget, job.focus,
                                                   cpt)
            await asyncio.to_thread(_save_plan, job, steps, notes, docs, step_tok, cpt)
        else:
            steps, notes, _, cpt = planned
        job.cpt = cpt
        warnings = warnings + notes
        job.steps = len(steps)
        save_job(job)
        parts: list[dict] = []
        read: set[int] = set()   # the steps whose part is in
        failed: str = ""
        stopped = False
        system = _PROMPT_FILE.read_text(encoding="utf-8")
        n = 0
        while n < len(steps):
            n += 1
            step = steps[n - 1]
            if n in done:
                if done[n] is None:
                    warnings.append(f"Part {n} of {len(steps)} ({step['label']}) couldn't be read by the model, "
                                    "so nothing was taken from it.")
                else:
                    parts.append(done[n])
                    read.add(n)
                continue
            if n > 1 and await _quota_spent(owner_id):
                warnings.append(f"Today's allowance ran out after part {n - 1} of {len(steps)}, so the "
                                "rest wasn't read. The documents are still kept whole for the agent.")
                stopped = True
                break
            seen = learned_cpt(job)
            if seen and not _RECUT[0] <= seen / cpt <= _RECUT[1]:
                # The model counts this material otherwise than estimated: the rest cut again to its count.
                log.info("agent preset: steps cut again to the model's count", model=(reader or {}).get("model"),
                         estimated=round(cpt, 2), counted=round(seen, 2), step=n)
                cpt = job.cpt = seen
                steps = refit(steps, n, step_tok, cpt)
                step = steps[n - 1]
                job.steps = len(steps)
                await asyncio.to_thread(_save_plan, job, steps, notes, docs, step_tok, cpt)
            job.step, job.current = n, step["label"]
            save_job(job)
            why = ""
            try:
                got = await _ask(reader, system, step_prompt(arch, instructions, step, n, len(steps)),
                                 owner_id, job)
            except (asyncio.CancelledError, _UnreadableError):
                raise
            except _BusyError:
                model = (reader or {}).get("model") or "The model"
                if not parts:
                    raise _UnreadableError(
                        f"{model} is still rate-limited or overloaded after waiting about 5 minutes. Try again "
                        "later, or keep the long documents as knowledge only, which needs no model.") from None
                rest = f"part {n} wasn't" if n == len(steps) else f"parts {n}–{len(steps)} weren't"
                warnings.append(f"{model} was still rate-limited or overloaded after waiting, so {rest} read. "
                                "The documents are still kept whole for the agent.")
                stopped = True
                break
            except Exception as exc:
                log.warning("agent preset: a step failed", step=n, model=(reader or {}).get("model"), error=str(exc))
                if isinstance(exc, _CutOffError):
                    why = f"{(reader or {}).get('model') or 'the model'} {exc}"
                failed = failed or (f"it {exc}." if isinstance(exc, _CutOffError) else str(exc))
                got = None
            _save_part(job, n, got)
            if got is None:
                warnings.append(f"Part {n} of {len(steps)} ({step['label']}) couldn't be read"
                                f"{': ' + why if why else ' by the model'}, so nothing was taken from it.")
                continue
            parts.append(got)
            read.add(n)
        if steps and not parts and not stopped:
            model = (reader or {}).get("model") or "the model"
            raise _UnreadableError(f"Couldn't read the material with {model}: {failed}" if failed
                              else "The model's answers couldn't be read. Try again, or with less material.")
        found: dict[str, int] = {}
        if len(parts) > 1:
            job.phase, job.current = "merging", "Putting the parts together"
            save_job(job)
            folded = await asyncio.to_thread(fold, parts)
            found = {g: len(folded[g]) for g in _ITEM_KEY}
            limits = limits_for(read_tokens(steps, read, cpt))
            preset = (normalise(folded, limits) if stopped
                      else await merge_parts(reader, parts, owner_id, job, limits, folded))
        else:
            preset = parts[0] if parts else normalise({})
        cover = await asyncio.to_thread(read_coverage, steps, read, docs)
        await asyncio.to_thread(_save_coverage, job, cover)
        job.result = _finish(preset, docs, job.notes + warnings, str((reader or {}).get("model") or ""),
                             len(steps), job.tokens, job.id, cover, found)
        job.state, job.finished = "done", time.time()
        if _folder(job).is_dir():
            _write_json(_folder(job) / _RESULT_FILE, job.result)
        save_job(job)
        await _tell(job)
    except asyncio.CancelledError:
        if job.stopping:
            job.state = "cancelled"
            discard(owner_id, job.id)
        # Otherwise Flight Deck is stopping: the read stays on disk as it is, and goes on at its next start.
        raise
    except Exception as exc:
        if not isinstance(exc, _UnreadableError):
            log.warning("agent preset: reading failed", exc_info=True)
        job.state, job.error = "error", str(exc) or "Reading failed."
        discard(owner_id, job.id)
        job.finished = time.time()
        await _tell(job)
    finally:
        job.finished = job.finished or (time.time() if job.state != "running" else None)


async def resume_reads(delay: float = 0.0) -> int:
    """At Flight Deck's start, once its agents are back: reads it stopped
    mid-way go on from the step after the last one they finished — with the
    agent's own model again (the owner's, when the agent is off, the rest cut
    to fit it) — and a finished one waiting for its review is there for the
    dialog. How many went on."""
    await asyncio.sleep(delay)
    root = _root()
    resumed = 0
    for owner in sorted(root.iterdir()) if root.is_dir() else []:
        for folder in sorted(owner.iterdir()) if owner.is_dir() else []:
            data = _read_json(folder / _JOB_FILE)
            meta = _read_json(folder / "meta.json")
            if (not isinstance(data, dict) or not isinstance(meta, dict) or data.get("state") != "running"
                    or folder.name in _JOBS or time.time() - float(meta.get("created") or 0) > _TTL):
                continue
            try:
                job = _job_from(data)
                docs = await asyncio.to_thread(load_docs, job.owner_id, job.id)
                spec = job.spec or {}
                who, why = await choose_reader(job.owner_id, job.agent)
                _JOBS[job.id] = job
                if not who:
                    job.state, job.error = "error", why or "No model is set up to read with any more."
                    discard(job.owner_id, job.id)
                    job.finished = time.time()
                    await _tell(job)
                    continue
                job.model, job.via, job.waiting = str(who.get("model") or ""), "agent" if who.get("agent") else "tiers", None
                job.notes.append("Flight Deck restarted during the read, so it went on from where it was.")
                if why:
                    job.notes.append(why)
                arch = await _archetype(job.owner_id, str(spec.get("archetype") or ""))
                job.task = asyncio.get_running_loop().create_task(run_read(
                    job, who, arch, str(spec.get("instructions") or ""), docs,
                    [str(w) for w in spec.get("warnings") or []], job.owner_id))
                resumed += 1
                log.info("agent preset: a read goes on after the restart", preset=job.id, agent=job.agent,
                         step=job.step, steps=job.steps)
            except Exception:
                log.warning("agent preset: a read couldn't go on after the restart", preset=folder.name,
                            exc_info=True)
    return resumed


# ── Routes: documents, reading ────────────────────────────────────────


def _json_list(raw: str) -> list:
    try:
        got = json.loads(raw or "[]")
    except ValueError:
        raise HTTPException(400, "The list of library files couldn't be read.")
    return got if isinstance(got, list) else []


def _own_process(slug: str, owner_id: str) -> dict:
    """The caller's own process agent's registry entry — 404 for anyone else's."""
    from captain_claw.flight_deck.server import _verify_process_owner
    return _verify_process_owner(str(slug or ""), owner_id)


def _owner(user: dict) -> str:
    return str(user.get("id") or "") if _auth_enabled() else ""


@router.get("/reader")
async def reader(agent: str = "", user: dict = Depends(get_managed_user)) -> dict:
    """The model that would read the material — the agent's own, else the
    owner's — and how much of it one step takes: what the dialog estimates
    steps and tokens from (and, with ``pace_tpm``, how long it takes).
    ``note``: why it isn't the agent's model."""
    who, note = await choose_reader(_owner(user), agent)
    if not who:
        return {"model": "", "step_tokens": 0, "pace_tpm": 0, "via": "", "note": note}
    return {"model": str(who.get("model") or ""), "step_tokens": step_tokens(who), "pace_tpm": pace_tpm(who),
            "via": "agent" if who.get("agent") else "tiers", "note": note}


@router.post("/documents")
async def add_documents(
    files: list[UploadFile] = File(default=[]),
    library: str = Form(""),
    user: dict = Depends(get_managed_user),
) -> dict:
    """Read documents in full — uploads, and ``library`` (JSON list of File
    library ids) — and keep them for the setup a day. Spends no tokens: says
    each one's pages, characters and the tokens reading it would take."""
    root = owner_root(_owner(user))
    try:
        await asyncio.to_thread(prune)
    except Exception:
        log.debug("agent preset: prune failed", exc_info=True)
    out: list[dict] = []
    warnings: list[str] = []
    for uf in [f for f in files or [] if f and f.filename]:
        got, said = await stage_upload(root, uf)
        out += [got] if got else []
        warnings += [said] if said else []
    for pick in _json_list(library):
        file_id = pick.get("id") if isinstance(pick, dict) else pick
        got, said = await stage_library(root, str(file_id or ""), user)
        out += [got] if got else []
        warnings += [said] if said else []
    return {"docs": out, "warnings": warnings}


class DocChoice(BaseModel):
    id: str = Field(max_length=32)
    description: str = Field(default="", max_length=2000)
    usage: str = Field(default="auto", max_length=20)
    # Read it for rules, skills and facts; otherwise it is only kept, whole,
    # for the agent to look things up in (no tokens).
    read: bool = True


class PrepareBody(BaseModel):
    agent: str = Field(default="", max_length=128)
    archetype: str = Field(default="", max_length=200)
    instructions: str = Field(default="", max_length=_MAX_INSTRUCTIONS * 2)
    docs: list[DocChoice] = Field(default_factory=list, max_length=_MAX_FILES)
    # How much of material bigger than this is read: its most instructive
    # sections, closest to ``focus`` when one is given (agent_preset_select).
    budget: Literal["quick", "standard", "deep"] = psel.DEFAULT_BUDGET
    focus: str = Field(default="", max_length=300)


@router.post("/prepare")
async def prepare(body: PrepareBody, user: dict = Depends(get_managed_user)) -> dict:
    """Start reading the person's instructions and staged documents into a
    preset for review — in the background; ``GET /prepare/{id}`` follows it.
    ``agent``: the agent being set up (its archetype frames the read). The
    documents are kept, whole, under the returned id; nothing reaches an
    agent until Apply."""
    owner_id = _owner(user)
    instructions = body.instructions.strip()[:_MAX_INSTRUCTIONS]
    if not instructions and not body.docs:
        raise HTTPException(400, "Add some instructions or at least one document first.")
    archetype = body.archetype
    if body.agent:
        archetype = archetype or str(_own_process(body.agent, owner_id).get("archetype") or "")
    try:
        await asyncio.to_thread(prune)
    except Exception:
        log.debug("agent preset: prune failed", exc_info=True)

    root = owner_root(owner_id)
    docs: list[dict] = []
    warnings: list[str] = []
    taken: set[str] = set()
    for choice in body.docs:
        staged = await asyncio.to_thread(pdocs.load_staged, root, choice.id)
        if staged is None:
            warnings.append("A document added more than a day ago is gone — add it again.")
            continue
        name = _display_name(staged["name"], taken)
        if staged.get("truncated"):
            warnings.append(f"{name}: only its first {pdocs.MAX_CHARS:,} characters are kept.")
        docs.append({**{k: staged.get(k) for k in ("text", "pages", "pages_exact", "truncated", "origin", "agent",
                                                   "source", "tables")},
                     # One the agent already keeps (stage_knowledge): what was read of it before, and its summary.
                     "skip": list(staged.get("read_keys") or []), "earlier_summary": str(staged.get("summary") or ""),
                     "name": name, "read": choice.read, **note(choice.model_dump())})
    if not instructions and not docs:
        raise HTTPException(400, " ".join(warnings) or "None of the documents are there any more.")

    who = None
    if instructions or any(d["read"] for d in docs):
        if await _quota_spent(owner_id):
            raise HTTPException(429, "Today's allowance is used up, so the documents can't be read now. "
                                     "Keep them as knowledge only, or try again tomorrow.")
        who, why = await choose_reader(owner_id, body.agent)
        if not who:
            raise HTTPException(409, why or "No model is set up to read the documents with. Ask your admin.")
        if why:
            warnings.append(why)
    preset_id = await asyncio.to_thread(save_docs, owner_id, docs, archetype)
    arch = await _archetype(owner_id, archetype)

    now = time.time()
    _forget_old_jobs(now)
    key = owner_key(owner_id)
    # One read per agent: a new one replaces one still running, and an earlier one waiting for review.
    for other in [j for j in _JOBS.values() if j.owner == key and j.agent == body.agent]:
        _stop(other)
    for other in await asyncio.to_thread(_on_disk, key, body.agent):
        if other != preset_id and other not in _JOBS:
            discard(owner_id, other)
    entry = _own_process(body.agent, owner_id) if body.agent else {}
    job = ReadJob(id=preset_id, owner=key, owner_id=owner_id, agent=body.agent,
                  agent_name=str(entry.get("name") or body.agent), budget=body.budget, focus=body.focus.strip(),
                  docs=[{"name": d["name"], "read": bool(d["read"])} for d in docs],
                  model=str((who or {}).get("model") or ""),
                  via=("agent" if who.get("agent") else "tiers") if who else "",
                  spec={"instructions": instructions, "archetype": archetype, "warnings": warnings},
                  watched=now)
    _JOBS[preset_id] = job
    save_job(job)
    job.task = asyncio.get_running_loop().create_task(
        run_read(job, who, arch, instructions, docs, warnings, owner_id))
    return job.view()


def _stop(job: ReadJob) -> None:
    """The person's own stop (a shutdown's cancel keeps the read): a running
    read is cancelled and its documents dropped; a finished one is put
    away unreviewed."""
    job.stopping = True
    if job.state == "running" and job.task is not None and not job.task.done():
        job.task.cancel()
    elif job.state == "done":
        job.state = "cancelled"
        discard(job.owner_id, job.id)
        _JOBS.pop(job.id, None)


def _on_disk(key: str, agent: str) -> list[str]:
    """The reads kept for *agent* in a person's folder — under way, or
    waiting for review — newest first."""
    found: list[tuple[float, str]] = []
    folder = _root() / key
    for sub in folder.iterdir() if folder.is_dir() else []:
        data = _read_json(sub / _JOB_FILE)
        if isinstance(data, dict) and data.get("agent") == agent and data.get("state") in ("running", "done"):
            found.append((float(data.get("started") or 0), sub.name))
    return [name for _, name in sorted(found, reverse=True)]


def _job(preset_id: str, user: dict) -> ReadJob:
    job = _find(owner_key(_owner(user)), preset_id)
    if job is None:
        raise HTTPException(404, "That reading isn't here any more. Read the material again.")
    return job


@router.get("/prepare")
async def latest_read(agent: str = "", user: dict = Depends(get_managed_user)) -> dict:
    """The read the setup dialog opens on for *agent*: the newest one still
    under way, or done and waiting for its review — ``job`` None when there
    is none, and the dialog starts from the material."""
    key = owner_key(_owner(user))
    _forget_old_jobs(time.time())
    for preset_id in await asyncio.to_thread(_on_disk, key, str(agent or "")):
        job = _find(key, preset_id)
        if job is not None and job.state in ("running", "done"):
            job.watched = time.time()
            return {"job": job.view()}
    return {"job": None}


@router.get("/prepare/{preset_id}")
async def read_status(preset_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Where a read stands: its step, the tokens spent, and — once done — the
    preset for review."""
    _forget_old_jobs(time.time())
    job = _job(preset_id, user)
    job.watched = time.time()
    return job.view()


@router.delete("/prepare/{preset_id}")
async def cancel_read(preset_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Stop a read — the step under way is abandoned — or put away one
    waiting for its review. Either way its documents are dropped."""
    job = _job(preset_id, user)
    task = job.task
    _stop(job)
    if task is not None and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except BaseException:
            pass
    return job.view()


# ── Writing the setup into an agent ───────────────────────────────────


def _file_stem(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", Path(name).stem.lower()).strip("-.")[:60] or "document"
    out, n = base, 2
    while out in taken or out in {"index", "glossary", "owner-rules"}:
        out = f"{base}-{n}"
        n += 1
    taken.add(out)
    return out


def _cite(source: dict) -> str:
    doc, section = source.get("doc") or "", source.get("section") or ""
    return f"{doc} — {section}" if doc and section else doc or section


_FROM = re.compile(r"^(?P<text>.+?) \(from (?P<doc>[^()]+?)(?: — (?P<section>[^()]+))?\)$")


def _from(source: dict) -> str:
    """" (from policy.pdf — Refunds)" after an item the owner's document gave;
    nothing for one from their own instructions."""
    doc = source.get("doc") or ""
    return f" (from {_cite(source)})" if doc and doc != "instructions" else ""


def _unfrom(line: str) -> tuple[str, dict]:
    """An item as ``_from`` wrote it: its text, and where it came from."""
    m = _FROM.match(line)
    if not m:
        return line, {"doc": "instructions", "section": ""}
    return m["text"], {"doc": m["doc"], "section": m["section"] or ""}


def _skill_md(intent: dict) -> str:
    title = intent["name"].replace("-", " ").capitalize()
    lines = ["---", f"name: {intent['name']}",
             f"description: {json.dumps(intent['when'] or title, ensure_ascii=False)}", "---", "",
             f"# {title}", ""]
    if intent["when"]:
        lines += [f"When: {intent['when']}", ""]
    if intent["examples"]:
        lines += ["How it's usually asked:", *[f"- {e}" for e in intent["examples"]], ""]
    if intent["steps"]:
        lines += ["## Steps", *[f"{n}. {s}" for n, s in enumerate(intent["steps"], 1)], ""]
    cite = _cite(intent["source"])
    if cite:
        lines += [f"Source: {cite} (from your owner, when they set you up)", ""]
        if (intent["source"].get("doc") or "instructions") != "instructions":
            lines += ["These steps are a short version: for the details, read that section with the knowledge "
                      "tool.", ""]
    return "\n".join(lines)


def write_skills(skills_dir: Path, intents: list[dict]) -> list[str]:
    """One skill per intent; the names written."""
    written: list[str] = []
    for intent in intents:
        folder = skills_dir / intent["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(_skill_md(intent), encoding="utf-8")
        written.append(intent["name"])
    return written


def _owner_rules_md(preset: dict) -> str:
    """Every rule, when-to-ask and template in full: the prompt block may
    have to leave some out."""
    lines = ["# Your owner's rules", "",
             "Set up by your owner from their instructions and documents.", ""]
    if preset["rules"]:
        lines += ["## Rules", *[f"- {r['text']}{_from(r.get('source') or {})}" for r in preset["rules"]], ""]
    if preset["escalation"]:
        lines += ["## Ask your owner first", *[f"- {r['text']}{_from(r.get('source') or {})}" for r in preset["escalation"]], ""]
    if preset.get("facts"):
        lines += ["## Facts", *[f"- {f['content']}{_from(f.get('source') or {})}" for f in preset["facts"]], ""]
    for t in preset["templates"]:
        lines += [f"## Template: {t['name']}", "", t["body"], ""]
    return "\n".join(lines)


_TABLES_NOTE = ("Your owner's spreadsheets, also as tables in your datastore — query them (the datastore tool) "
                "for exact values such as prices, plans and codes rather than searching the text.")


def _table_line(t: dict) -> str:
    cols = [c for c in t.get("columns") or [] if isinstance(c, str)]
    sheet = f", sheet “{t['sheet']}”" if t.get("sheet") else ""
    line = f"- `{t['name']}` — {t.get('doc') or 'a spreadsheet'}{sheet}: {int(t.get('rows') or 0):,} rows"
    if cols:
        line += "; columns " + ", ".join(cols[:12]) + (", …" if len(cols) > 12 else "")
    return line + (f" (only its first {ptables.MAX_ROWS:,} rows)" if t.get("cut") else "")


def _own_tables_line(own: list[dict]) -> str:
    """What a spreadsheet's knowledge file says about its tables."""
    if len(own) == 1:
        return f"It is also a table in your datastore: `{own[0]['name']}` — query it for exact values."
    return ("It is also in your datastore, a table per sheet: "
            + ", ".join(f"`{t['name']}` ({t.get('sheet') or 'sheet'})" for t in own)
            + " — query them for exact values.")


def write_knowledge(folder: Path, preset: dict, docs: list[dict], kept: list[dict] | None = None,
                    tables: list[dict] | None = None) -> list[dict]:
    """The kept documents — their whole text — as Markdown, with an index,
    the glossary and the full rules; the documents written (``doc``,
    ``file``, ``title``, ``parts``, ``files``). A document too long for one
    file the agent's semantic memory indexes is an overview plus parts
    (``<name>-part-01.md`` …), each cut at a page or heading. ``kept``: the
    documents an earlier setup left, which stay in the index. ``tables``:
    the tables the agent has from setups (``name``, ``doc``, ``sheet``,
    ``rows``, ``columns``), listed in the index and named in their
    spreadsheet's own file."""
    folder.mkdir(parents=True, exist_ok=True)
    doc_of = {d["name"]: d for d in docs}
    kept = kept or []
    tables = tables or []
    # The documents an earlier setup left keep their files and their place in the index.
    taken: set[str] = {Path(f).stem for e in kept for f in e.get("files") or []}
    written: list[dict] = []
    for k in preset["knowledge"]:
        doc = doc_of.get(k["doc"])
        if doc is None:
            continue
        stem = _file_stem(k["doc"], taken)
        title = k["title"] or Path(k["doc"]).stem
        use = USAGES.get(doc.get("usage") or "auto", ("", ""))[0]
        source = f"Source: {k['doc']} — given to you by your owner."
        head = [f"# {title}", "", source, ""]
        if doc.get("description"):
            head += [f"Your owner says: {doc['description']}", ""]
        if use:
            head += [f"Use it as: {use.lower()}.", ""]
        if k["summary"] and k["summary"] != doc.get("description"):
            head += [k["summary"], ""]
        own = [t for t in tables if t.get("doc") == k["doc"]]
        if own:
            head += [_own_tables_line(own), ""]
        text = doc["text"].strip()
        parts = pdocs.split_text(text, _PART_BYTES)
        # A long document gets a map of its sections: which file and lines hold each (agent_preset_outline).
        sections = poutline.outline(text) if len(text) >= _MAP_MIN_CHARS else []
        map_name = f"{stem}.map.md" if len(sections) >= 3 else ""
        pointer = [f"Its section map, {map_name}, says which lines hold each section.", ""] if map_name else []
        files: list[dict] = []  # each file's text and where it starts, for the map's line numbers
        if len(parts) <= 1:
            prefix = "\n".join(head + pointer) + "\n---\n\n"
            (folder / f"{stem}.md").write_text(prefix + text + "\n", encoding="utf-8")
            files.append({"name": f"{stem}.md", "label": "", "start": 0, "raw": text, "body": text, "prefix": prefix})
        else:
            listing: list[str] = []
            at = 0
            for n, part in enumerate(parts, 1):
                name = f"{stem}-part-{n:02d}"
                taken.add(name)
                where = f" ({part['label']})" if part["label"] else ""
                prefix = "\n".join([f"# {title} — part {n} of {len(parts)}{where}", "", source,
                                    f"The whole document: {stem}.md.", "", "---", ""]) + "\n"
                body = part["text"].strip()
                (folder / f"{name}.md").write_text(prefix + body + "\n", encoding="utf-8")
                files.append({"name": f"{name}.md", "label": part["label"], "start": at, "raw": part["text"],
                              "body": body, "prefix": prefix})
                at += len(part["text"])
                listing.append(f"- [Part {n}{where}]({name}.md)")
            (folder / f"{stem}.md").write_text("\n".join([
                *head, *pointer, f"It is long, so it is kept in {len(parts)} parts:", "", *listing, ""]), encoding="utf-8")
        if map_name:
            taken.add(f"{stem}.map")
            (folder / map_name).write_text(
                poutline.map_md(title, k["doc"], files, poutline.add_gists(text, sections)), encoding="utf-8")
        cov = doc.get("coverage") if isinstance(doc.get("coverage"), dict) else {}
        read = {"all": bool(cov.get("all")), "keys": list(cov.get("keys") or []),
                "of": int(cov["of"]) if "of" in cov else psel.unread(text, set())[2]}
        written.append({"doc": k["doc"], "file": f"{stem}.md", "title": title, "summary": k["summary"], "use": use,
                        "read": read,
                        "parts": len(parts), "sections": len(sections) if map_name else 0, "map": map_name,
                        "files": [f"{stem}.md", *(f"{stem}-part-{n:02d}.md" for n in range(1, len(parts) + 1)
                                                  if len(parts) > 1), *([map_name] if map_name else [])]})
    listed = kept + written
    _write_or_drop(folder / "INDEX.md", (listed or tables) and "\n".join([
        "# Knowledge", "",
        "Documents your owner gave you. Read the one that fits before you answer from memory.", "",
        *[f"- [{w['title']}]({w['file']})" + (f" — {w['summary']}" if w.get("summary") else "")
          + (f" ({w['use'].lower()})" if w.get("use") else "")
          + (f" — {w['parts']} parts" if (w.get("parts") or 1) > 1 else "")
          + (f" — section map: {w['map']}" if w.get("map") else "")
          for w in listed],
        *(["", "## Tables", "", _TABLES_NOTE, "", *map(_table_line, tables)] if tables else []), ""]))
    _write_or_drop(folder / "glossary.md", preset["glossary"] and "\n".join([
        "# Glossary", "", *[f"- **{g['term']}** — {g['meaning']}{_from(g.get('source') or {})}" for g in preset["glossary"]],
        ""]))
    _write_or_drop(folder / "owner-rules.md", (preset["rules"] or preset["escalation"] or preset["templates"]
                                                or preset.get("facts")) and _owner_rules_md(preset))
    return written


def _write_or_drop(path: Path, text: Any) -> None:
    """Write *text* to *path*; nothing to say takes away what it said before."""
    if text:
        path.write_text(text, encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def rules_block(preset: dict, knowledge: list[dict], skills: list[str], cap: int = _BLOCK_CAP,
                tables: list[dict] | None = None, facts: int | None = None) -> str:
    """The short block the new agent's instructions end with — at most
    ``cap`` characters. With lessons (*facts*: how many facts it was taught
    as lessons): only the rules and when-to-ask marked ``always`` — the rest
    reach it with the messages they bear on (agent_lessons), which it's told.
    Without (an agent running from before lessons): every rule that fits,
    the rest pointed at in knowledge/owner-rules.md."""
    head = ["## Your owner's rules", "",
            "Set up by your owner. Where they differ from the general procedure above, "
            "follow these."]
    def named(names: list[str], more: str) -> str:
        rest = len(names) - _BLOCK_NAMES
        return ", ".join(names[:_BLOCK_NAMES]) + (f", and {rest} more ({more})" if rest > 0 else "")

    tail: list[str] = []
    if preset["templates"]:
        tail += ["", "### Templates",
                 "Use these formats when they fit (full text in knowledge/owner-rules.md): "
                 + named([t["name"] for t in preset["templates"]], "in knowledge/owner-rules.md") + "."]
    if skills:
        tail += ["", "### What you handle",
                 "You have a skill for each of these — read it before you start: "
                 + named(skills, "in your skills list") + "."]
    if knowledge or preset["glossary"]:
        line = "Your workspace's knowledge/ folder holds what your owner gave you"
        if knowledge:
            line += (f" ({len(knowledge)} document{'s' if len(knowledge) != 1 else ''}: "
                     + "; ".join(k["title"] for k in knowledge)[:600] + "; index: knowledge/INDEX.md)")
        if preset["glossary"]:
            line += ", and their terms in knowledge/glossary.md"
        line += ". Look there before you answer from memory"
        if knowledge:
            line += (": the knowledge tool searches all of it (search — each hit says its document, section and "
                     "lines), reads a whole section by its title (section) and outlines a document (outline)")
        if any(k.get("map") for k in knowledge):
            line += ". A long document also has a section map (<name>.map.md) saying which lines hold each section"
        line += "."
        if knowledge:
            line += (" Your rules, skills, facts and terms are a short version: those drawn from these documents "
                     "say which document and section they came from (for the rules above, see "
                     "knowledge/owner-rules.md). When one decides an answer, or you need its exact wording, figures "
                     "or the steps around it, read that section with the knowledge tool and go by the document.")
        tail += ["", "### Your knowledge", line]
    if tables:
        tail += ["", "### Your tables",
                 "Your owner's spreadsheets are also tables in your datastore — query them (the datastore tool) "
                 "for exact values such as prices, plans and codes, rather than searching the text: "
                 + ", ".join(t["name"] for t in tables)[:600] + " (what is in each: knowledge/INDEX.md)."]

    lessons = facts is not None
    if lessons:
        later = sum(1 for g in ("rules", "escalation") for r in preset[g] if not r.get("always"))
        if later or facts:
            what = " and ".join(x for x in (later and f"{later} more rule{'s' if later != 1 else ''}",
                                             facts and f"{facts} fact{'s' if facts != 1 else ''}") if x)
            tail = ["", "### When it matters",
                    f"Your owner also taught you {what}. Each reaches you with the messages it bears on, under "
                    "\"Your owner's lessons\" — follow them as you follow these; all of them are in "
                    "knowledge/owner-rules.md.", *tail]
    budget = cap - len("\n".join(head + tail)) - 200  # room for headings and the pointer
    kept: dict[str, list[str]] = {"escalation": [], "rules": []}
    dropped = 0
    for group in ("escalation", "rules"):
        for r in preset[group]:
            if lessons and not r.get("always"):
                continue
            line = f"- {r['text']}"
            if len(line) + 1 <= budget:
                kept[group].append(line)
                budget -= len(line) + 1
            else:
                dropped += 1
    body: list[str] = []
    if kept["rules"]:
        body += ["", "### Rules", *kept["rules"]]
    if kept["escalation"]:
        body += ["", "### Ask your owner first", *kept["escalation"]]
    if dropped and not lessons:
        body += ["", f"({dropped} more rule{'s' if dropped != 1 else ''} in knowledge/owner-rules.md — read it.)"]
    if not body and not tail:
        return ""
    return "\n".join(head + body + tail)[:cap]


_BLOCK_OPEN, _BLOCK_CLOSE = "<!-- owner-setup -->", "<!-- /owner-setup -->"
_MARKED_BLOCK = re.compile(r"\s*<!-- owner-setup -->.*?<!-- /owner-setup -->", re.S)
# A block written before it was marked: its heading to the next ``## `` or the end.
_BARE_BLOCK = re.compile(r"(?ms)\s*^## Your owner's rules\n\nSet up by your owner.*?(?=^## |\Z)")


def strip_block(instructions: str) -> str:
    """The instructions without the block an earlier setup put in them."""
    text = _MARKED_BLOCK.sub("", instructions or "")
    # What followed an unmarked block keeps its blank line.
    return _BARE_BLOCK.sub(lambda m: "" if m.end() == len(m.string) else "\n\n", text).rstrip()


def with_block(instructions: str, block: str) -> str:
    """The agent's instructions ending with the owner's block — in place of
    one an earlier setup left — within the instructions' own cap (the block
    wins: the rest is clipped). The block is marked, so the next setup finds
    it again."""
    base = strip_block(instructions)
    if not block:
        return base
    marked = f"{_BLOCK_OPEN}\n{block}\n{_BLOCK_CLOSE}"
    base = base[:max(0, _MAX_INSTRUCTIONS_TOTAL - len(marked) - 2)]
    return f"{base}\n\n{marked}" if base else marked


# ── What a setup gave an agent (its record, outside its workspace) ────

_RECORD = ".owner-setup.json"
_KEPT_GROUPS = ("rules", "escalation", "templates", "glossary")


def read_record(agent_dir: Path) -> dict:
    """What earlier setups gave the agent: the groups it goes by, its skills
    and documents (with their files) and how many facts — {} for none."""
    try:
        got = json.loads((agent_dir / _RECORD).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


_INDEX_LINE = re.compile(r"^- \[(?P<title>.+?)\]\((?P<file>[a-z0-9._-]+\.md)\)(?: — (?P<rest>.*))?$")
_SETUP_SKILL = re.compile(r"\(from your owner, when (?:they set you up|you were created)\)")


def _record_from_files(skills_dir: Path, folder: Path) -> dict:
    """The record of a setup written before setups kept one, read back from
    the files it wrote: the documents in INDEX.md, the rules in
    owner-rules.md, the glossary, and the skills that say they came from
    the owner — {} when there are none."""
    def bullets(text: str, heading: str) -> list[dict]:
        m = re.search(rf"(?ms)^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text)
        return [dict(zip(("text", "source"), _unfrom(ln[2:].strip())))
                for ln in (m.group(1) if m else "").splitlines() if ln.startswith("- ")]

    record: dict[str, Any] = {}
    try:
        index = (folder / "INDEX.md").read_text(encoding="utf-8") if (folder / "INDEX.md").is_file() else ""
        knowledge = []
        for line in index.splitlines():
            m = _INDEX_LINE.match(line.strip())
            if not m or not (folder / m["file"]).is_file():
                continue
            stem = m["file"][:-3]
            parts = sorted(p.name for p in folder.glob(f"{stem}-part-*.md"))
            maps = [f"{stem}.map.md"] if (folder / f"{stem}.map.md").is_file() else []
            summary = re.sub(r" — section map: [a-z0-9._-]+\.map\.md$", "", m["rest"] or "")
            summary = re.sub(r"( — \d+ parts)?$", "", summary)
            summary = re.sub(r" \((?:rules to follow|procedures|reference|templates)\)$", "", summary)
            knowledge.append({"doc": m["file"], "file": m["file"], "title": m["title"], "summary": summary,
                              "use": "", "parts": max(1, len(parts)), "map": maps[0] if maps else "",
                              "files": [m["file"], *parts, *maps]})
        rules = (folder / "owner-rules.md").read_text(encoding="utf-8") if (folder / "owner-rules.md").is_file() else ""
        glossary = (folder / "glossary.md").read_text(encoding="utf-8") if (folder / "glossary.md").is_file() else ""
        templates = [{"name": m.group(1).strip(), "body": m.group(2).strip()}
                     for m in re.finditer(r"(?ms)^## Template: (.+?)\n\n(.*?)(?=^## |\Z)", rules)]
        terms = [{"term": m.group(1), **dict(zip(("meaning", "source"), _unfrom(m.group(2).strip())))}
                 for m in re.finditer(r"(?m)^- \*\*(.+?)\*\* — (.+)$", glossary)]
        skills = [d.name for d in sorted(skills_dir.iterdir()) if (d / "SKILL.md").is_file()
                  and _SETUP_SKILL.search((d / "SKILL.md").read_text(encoding="utf-8"))] if skills_dir.is_dir() else []
        record = {"rules": bullets(rules, "Rules"), "escalation": bullets(rules, "Ask your owner first"),
                  "templates": templates, "glossary": terms, "skills": skills, "knowledge": knowledge, "facts": 0}
    except OSError:
        log.warning("agent preset: could not read back an earlier setup", folder=str(folder), exc_info=True)
        return {}
    return record if any(record[k] for k in ("rules", "escalation", "templates", "glossary", "skills", "knowledge")) else {}


def earlier_setup(agent_dir: Path, workspace: Path) -> dict:
    """What earlier setups gave the agent: its record, else what the files
    of a setup from before records say."""
    return read_record(agent_dir) or _record_from_files(
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "skills", workspace / "knowledge")


def _unlink(folder: Path, names: list) -> None:
    for name in names or []:
        if isinstance(name, str) and re.fullmatch(r"[a-z0-9._-]+\.md", name):
            (folder / name).unlink(missing_ok=True)


def _undo(record: dict, skills_dir: Path, knowledge: Path) -> None:
    """Take out what a setup wrote — its skills, its documents and the files
    made from its rules — leaving whatever else the agent has."""
    for name in record.get("skills") or []:
        if isinstance(name, str) and skill_name(name) == name and name:
            shutil.rmtree(skills_dir / name, ignore_errors=True)
    for entry in record.get("knowledge") or []:
        _unlink(knowledge, entry.get("files") or [])
    _unlink(knowledge, ["INDEX.md", "glossary.md", "owner-rules.md"])


def _table_summary(t: dict) -> dict:
    """What the record, the index and the dialog keep of a table."""
    return {"name": t["name"], "doc": t["doc"], "sheet": t.get("sheet"), "rows": len(t["rows"]),
            "columns": [c["name"] for c in t["columns"]], "cut": bool(t.get("cut"))}


def new_tables(agent_dir: Path, preset: dict, docs: list[dict], drop: list[str],
               kept: list[dict]) -> tuple[list[dict], list[str]]:
    """The tables the kept spreadsheets give — one per sheet (``name``,
    ``doc``, ``sheet``, ``columns``, ``rows``, ``cut``), each named so it
    takes no table the agent keeps for itself, nor one an earlier setup gave
    that stays — within the room its datastore has; and what was left out,
    said for the person."""
    doc_of = {d["name"]: d for d in docs}
    sources = [(k["doc"], doc_of[k["doc"]]["source"]) for k in preset["knowledge"]
               if k["doc"] in doc_of and doc_of[k["doc"]].get("source")]
    if not sources:
        return [], []
    taken = (ptables.existing_tables(agent_dir) - set(drop)) | {t["name"] for t in kept}
    most = ptables.max_tables(agent_dir)
    room = most - len(taken)
    out: list[dict] = []
    warnings: list[str] = []
    for doc, source in sources:
        got, notes = ptables.read_tables(Path(source))
        warnings += [f"{doc}: {n}." for n in notes]
        left_out = 0
        for t in got:
            if room <= 0:
                left_out += 1
                continue
            room -= 1
            name = ptables.unique(ptables.table_name(doc, t["sheet"] if len(got) > 1 else None), taken)
            if t["cut"]:
                where = f", sheet “{t['sheet']}”" if t["sheet"] and len(got) > 1 else ""
                warnings.append(f"{doc}{where}: only its first {ptables.MAX_ROWS:,} rows are in the table.")
            out.append({**t, "name": name, "doc": doc})
        if left_out:
            warnings.append(f"{doc}: {left_out} sheet{'s' if left_out != 1 else ''} didn't become "
                            f"table{'s' if left_out != 1 else ''} — the agent's datastore holds at most {most}.")
    return out, warnings


def _more_read(before: Any, now: Any) -> dict:
    """What is read of a document: what was before, and what was now."""
    a = before if isinstance(before, dict) else {}
    b = now if isinstance(now, dict) else {}
    keys = list(dict.fromkeys([*(a.get("keys") or []), *(b.get("keys") or [])]))
    of = int(b.get("of") or a.get("of") or 0)
    return {"all": bool(a.get("all") or b.get("all") or (of and len(keys) >= of)), "keys": keys, "of": of}


def seed(agent_dir: Path, workspace: Path, owner_id: str, body: Any, mode: str = "add") -> dict:
    """Write the reviewed preset into an agent's folder — added to what an
    earlier setup gave it (the new items first; a document given again
    replaces its earlier copy, its tables too), or in its place
    (``mode="replace"``: the earlier skills, documents, tables and rules go
    first; facts it learned, and lessons an exam taught it, stay). Its rules,
    when-to-ask and facts become its lessons (agent_lessons, written for it
    here). Returns the ``block`` its instructions end with (``block_all``:
    the one for an agent running from before lessons), the ``tables`` to
    make in its datastore and those to ``drop`` first, what this setup wrote
    (``skills``, ``knowledge``, ``counts``), what the agent now has from
    setups (``totals``) and ``warnings`` for the person. The waiting
    documents are dropped afterwards."""
    data = body if isinstance(body, dict) else {}
    preset = normalise(data, CEILINGS)
    preset_id = str(data.get("id") or "")
    docs = load_docs(owner_id, preset_id) if preset_id else []
    if preset["knowledge"] and not docs and preset_id:
        log.warning("agent preset: its documents are gone (expired?)", preset=preset_id)
    skills_dir = agent_dir / "data" / "home-config-parent" / ".captain-claw" / "skills"
    folder = workspace / "knowledge"
    earlier = earlier_setup(agent_dir, workspace)
    before_lessons = list(earlier.get("lessons") or [])
    earlier_tables = [t for t in earlier.get("tables") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)]
    before_knowledge = list(earlier.get("knowledge") or [])
    if mode == "replace":
        _undo(earlier, skills_dir, folder)
        # What its owner or an exam taught by hand isn't a setup's: it stays.
        earlier = {g: [i for i in earlier.get(g) or [] if isinstance(i, dict)
                       and i.get("taught_by") not in (None, "", "setup")] for g in ("rules", "escalation")}

    merged = normalise({g: preset[g] + [i for i in earlier.get(g) or [] if isinstance(i, dict)]
                        for g in _KEPT_GROUPS}, TOTALS)
    new_skills = write_skills(skills_dir, preset["intents"])
    skills = new_skills + [n for n in earlier.get("skills") or []
                           if isinstance(n, str) and n not in new_skills and (skills_dir / n / "SKILL.md").is_file()]
    # Read again from the agent's own knowledge (stage_knowledge): adding, it keeps its files, tables and
    # summary, and only what is read of it grows; replacing, it is written again like any other.
    again = {d["name"]: d for d in docs if d.get("origin") == "knowledge"}
    for name, d in again.items():
        before = next((e for e in before_knowledge if isinstance(e, dict) and e.get("doc") == name), {})
        d["coverage"] = _more_read(before.get("read"), d.get("coverage"))
    if mode != "replace":
        given_again = set(again)
    else:
        given_again = set()
    given = ({k["doc"] for k in preset["knowledge"]} & {d["name"] for d in docs}) - given_again
    kept: list[dict] = []
    for entry in earlier.get("knowledge") or []:
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        if entry.get("doc") in given:
            _unlink(folder, entry.get("files") or [])
        elif (folder / entry["file"]).is_file():
            if entry.get("doc") in given_again:
                summary = next((k["summary"] for k in preset["knowledge"] if k["doc"] == entry["doc"]), "")
                entry = {**entry, "read": again[entry["doc"]]["coverage"],
                         "summary": entry.get("summary") or summary}
            kept.append(entry)
    # Replacing drops every table an earlier setup made; adding, those of the documents given again. A
    # document read again from the agent's knowledge keeps its tables either way (its spreadsheet isn't kept).
    gone = [t for t in earlier_tables if (mode == "replace" or t.get("doc") in given) and t.get("doc") not in again]
    drop = list(dict.fromkeys(t["name"] for t in gone))
    kept_tables = [t for t in earlier_tables if t["name"] not in drop]
    tables, warnings = new_tables(agent_dir, preset, docs, drop, kept_tables)
    all_tables = kept_tables + [_table_summary(t) for t in tables]
    # Lessons beyond the rules: facts (a setup's own, replaced with it) and what exams taught (kept).
    lessons = [x for x in before_lessons if isinstance(x, dict)
               and (mode != "replace" or x.get("taught_by") != "setup")]
    record: dict[str, Any] = {"lessons": lessons}
    for x in agent_lessons.facts_as_lessons(preset["facts"]):
        record, _ = agent_lessons.taught(record, "fact", x["text"], x["source"], taught_by="setup",
                                         importance=x["importance"])
    fact_lessons = [x for x in record["lessons"] if x.get("kind") == "fact"]
    written = write_knowledge(folder, {**merged, "facts": [{"content": x["text"], "source": x.get("source") or {}}
                                                           for x in fact_lessons],
                                       "knowledge": [k for k in preset["knowledge"] if k["doc"] not in given_again]},
                              docs, kept, all_tables)
    knowledge = kept + written
    facts = int(earlier.get("facts") or 0) + len(preset["facts"])
    agent_dir.mkdir(parents=True, exist_ok=True)
    record = {"updated": time.time(), **{g: merged[g] for g in _KEPT_GROUPS}, "skills": skills,
              "knowledge": knowledge, "tables": all_tables, "facts": facts, "lessons": record["lessons"]}
    (agent_dir / _RECORD).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    agent_lessons.write(agent_dir, record)
    if preset_id:
        discard(owner_id, preset_id)
        _JOBS.pop(preset_id, None)
    return {
        "block": rules_block(merged, knowledge, skills, tables=all_tables, facts=len(fact_lessons)),
        "block_all": rules_block(merged, knowledge, skills, tables=all_tables),
        "tables": tables,
        "drop": drop,
        "skills": new_skills,
        "knowledge": written,
        "warnings": warnings,
        "counts": {"rules": len(preset["rules"]), "escalation": len(preset["escalation"]),
                   "templates": len(preset["templates"]), "glossary": len(preset["glossary"]),
                   "intents": len(new_skills), "facts": len(preset["facts"]), "knowledge": len(written),
                   "tables": len(tables), "read_again": len(given_again)},
        "totals": {**{g: len(merged[g]) for g in _KEPT_GROUPS}, "intents": len(skills), "facts": facts,
                   "knowledge": len(knowledge), "tables": len(all_tables)},
    }


async def _wait_ready(port_of: Callable[[], int], token: str, timeout: float) -> int:
    """The agent's port once its REST answers; 0 when it never did."""
    import httpx
    params = {"token": token} if token else {}
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    delay = 0.5
    async with httpx.AsyncClient(timeout=5.0) as client:
        while loop.time() < deadline:
            port = int(port_of() or 0)  # it may have drifted to another port
            if port:
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/api/insights",
                                            params={**params, "limit": 1})
                    if resp.status_code < 500:
                        return port
                except Exception:
                    pass
            await asyncio.sleep(delay)
            delay = min(2.0, delay * 1.4)
    return 0


# Work still on its way to an agent (indexing, tables): a task nobody holds
# can be garbage-collected before it runs.
_PENDING: set[asyncio.Task] = set()


async def ask_to_index(slug: str, port_of: Callable[[], int], token: str, timeout: float = _FACTS_READY_S) -> bool:
    """Ask a running agent to index its knowledge/ now — its semantic memory
    otherwise starts when it first searches, and that search finds the index
    as it was. Never raises; whether it was asked."""
    import httpx
    try:
        port = await _wait_ready(port_of, token, timeout)
        if not port:
            return False
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"http://127.0.0.1:{port}/api/semantic-memory/sync",
                                     params={"token": token} if token else {})
        log.info("agent preset: asked the agent to index its knowledge", agent=slug, status=resp.status_code)
        return resp.status_code < 300
    except Exception:
        log.warning("agent preset: asking the agent to index failed", agent=slug, exc_info=True)
        return False


def schedule_index(slug: str, port_of: Callable[[], int], token: str) -> None:
    try:
        task = asyncio.get_running_loop().create_task(ask_to_index(slug, port_of, token))
    except RuntimeError:
        return
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def give_tables(slug: str, port_of: Callable[[], int], token: str, drop: list[str], tables: list[dict],
                      agent_dir: Path, timeout: float = _FACTS_READY_S) -> int:
    """Give a running agent its owner's spreadsheets as tables once it
    answers, the tables they replace dropped first; how many were made.
    Never raises: what it couldn't be given waits in its folder for its
    next start."""
    dropped: set[str] = set()
    made: set[str] = set()
    try:
        port = await _wait_ready(port_of, token, timeout)
        if port:
            dropped, made = await ptables.post_tables(port, token, drop, tables)
            log.info("agent preset: tables given", agent=slug, made=len(made), of=len(tables), dropped=len(dropped))
        else:
            log.warning("agent preset: the agent never came up, tables left for its next start", agent=slug)
    except Exception:
        log.warning("agent preset: giving tables failed", agent=slug, exc_info=True)
    try:
        await asyncio.to_thread(ptables.forget_pending, agent_dir, dropped | made)
        await asyncio.to_thread(ptables.leave_tables, agent_dir, [n for n in drop if n not in dropped],
                                [t for t in tables if t["name"] not in made])
    except Exception:
        log.warning("agent preset: could not leave the tables for later", agent=slug, exc_info=True)
    return len(made)


# The tables still on their way to each agent: a setup applied again waits
# for the one before, so their drops and rows never cross.
_TABLES_BUSY: dict[str, asyncio.Task] = {}


def schedule_tables(slug: str, port_of: Callable[[], int], token: str, drop: list[str], tables: list[dict],
                    agent_dir: Path) -> None:
    if not drop and not tables:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("agent preset: no loop to give tables on, left for its next start", agent=slug)
        ptables.leave_tables(agent_dir, drop, tables)
        return
    before = _TABLES_BUSY.get(slug)

    async def after_the_last() -> int:
        if before is not None and not before.done() and before.get_loop() is loop:
            await asyncio.wait([before])
        return await give_tables(slug, port_of, token, drop, tables, agent_dir)

    task = loop.create_task(after_the_last())
    _TABLES_BUSY[slug] = task
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)
    task.add_done_callback(lambda t: _TABLES_BUSY.pop(slug, None) if _TABLES_BUSY.get(slug) is t else None)


# ── Apply (to the agent the dialog was opened on) ─────────────────────


class ApplyBody(BaseModel):
    agent: str = Field(min_length=1, max_length=128)
    # The prepared id and the items the person kept, as they edited them.
    preset: dict
    # Added to what an earlier setup gave the agent, or in its place.
    mode: Literal["add", "replace"] = "add"


def knowledge_text(folder: Path, entry: dict) -> str:
    """A document the agent keeps, as one text again: its parts' text (or
    its one file's), without the heads the setup put on them."""
    from captain_claw.tools.knowledge import _body_start

    files = [f for f in entry.get("files") or [] if re.search(r"-part-\d+\.md$", str(f))] or [entry.get("file")]
    bodies: list[str] = []
    for name in files:
        if not (isinstance(name, str) and re.fullmatch(r"[a-z0-9._-]+\.md", name) and (folder / name).is_file()):
            continue
        raw = (folder / name).read_text(encoding="utf-8", errors="replace")
        bodies.append(raw[_body_start(raw):].strip())
    return "\n\n".join(b for b in bodies if b)


_USAGE_OF_LABEL = {label.lower(): key for key, (label, _) in USAGES.items() if label}


class KnowledgeStage(BaseModel):
    docs: list[str] = Field(default_factory=list, max_length=_MAX_FILES)


@router.post("/knowledge/{slug}")
async def stage_knowledge(slug: str, body: KnowledgeStage, user: dict = Depends(get_managed_user)) -> dict:
    """Take documents the agent already keeps — not read, or only partly —
    to read them now, as if they were added: their text from its knowledge
    folder, staged with what of them was read before (only the rest is
    read), their summary and how they are used. Spends no tokens; Apply
    keeps their files and only adds what the read finds."""
    from captain_claw.flight_deck import server as srv
    owner_id = _owner(user)
    entry = _own_process(slug, owner_id)
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    record = await asyncio.to_thread(earlier_setup, agent_dir, workspace)
    known = {e.get("doc"): e for e in record.get("knowledge") or [] if isinstance(e, dict) and e.get("doc")}
    root = owner_root(owner_id)
    out: list[dict] = []
    warnings: list[str] = []
    for name in dict.fromkeys(body.docs):
        e = known.get(name)
        if e is None:
            warnings.append(f"{name}: {entry.get('name') or slug} doesn't keep it any more.")
            continue
        text = await asyncio.to_thread(knowledge_text, workspace / "knowledge", e)
        if not text.strip():
            warnings.append(f"{e.get('title') or name}: its files are gone from the knowledge folder.")
            continue
        state = e.get("read") if isinstance(e.get("read"), dict) else {}
        done = set() if not state else set(state.get("keys") or [])
        left, read, of = await asyncio.to_thread(psel.unread, text, done)
        if state.get("all") or (of and read >= of):
            warnings.append(f"{e.get('title') or name}: all of it is read already.")
            continue
        usage = _USAGE_OF_LABEL.get(str(e.get("use") or "").lower(), "auto")
        staged = await asyncio.to_thread(
            pdocs.stage, root, name, text, "knowledge", slug, None,
            {"tokens": pdocs.tokens_of(left, pdocs.chars_per_token(text)), "read_keys": sorted(done),
             "summary": str(e.get("summary") or "")})
        out.append({**staged, "title": str(e.get("title") or name), "usage": usage,
                    "read": {**_read_state(state or None), "sections": read, "of": of}})
    return {"docs": out, "warnings": warnings}


@router.get("/current/{slug}")
async def current_setup(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """What earlier setups gave one of the caller's agents — for the dialog
    opened on an agent that already has one."""
    from captain_claw.flight_deck import server as srv
    entry = _own_process(slug, _owner(user))
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    record = await asyncio.to_thread(earlier_setup, agent_dir, workspace)
    return {
        "has": bool(record),
        "updated": record.get("updated"),
        **{g: len(record.get(g) or []) for g in _KEPT_GROUPS},
        "skills": [n for n in record.get("skills") or [] if isinstance(n, str)],
        "facts": int(record.get("facts") or 0),
        "knowledge": [{"title": e.get("title") or e.get("doc") or "", "doc": e.get("doc") or "",
                       "parts": int(e.get("parts") or 1), "read": _read_state(e.get("read"))}
                      for e in record.get("knowledge") or [] if isinstance(e, dict)],
        "tables": [{"name": t["name"], "doc": t.get("doc") or "", "sheet": t.get("sheet"),
                    "rows": int(t.get("rows") or 0)}
                   for t in record.get("tables") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)],
    }


async def instruct(slug: str, owner_id: str, block: str, block_all: str) -> tuple[bool, str, str]:
    """The owner's block in the agent's stored instructions, in place of an
    earlier one — the one with every rule that fits (*block_all*) for an
    agent running from before lessons. Returns (stored — without accounts the
    browser keeps the instructions and puts the block in itself —, the
    instructions, the block used)."""
    from captain_claw.flight_deck import server as srv
    caps = await agent_lessons.capabilities(slug) if srv._process_is_alive(slug) else None
    used = block_all if caps is not None and agent_lessons.CAPABILITY not in caps else block
    instructions = ""
    stored = bool(owner_id) and _auth_enabled()
    if stored:
        current = (await srv._owner_agent_instructions(owner_id, "process")).get(slug, "")
        instructions = with_block(current, used)
        if instructions != current:
            await srv._set_owner_agent_instructions(owner_id, "process", slug, instructions, strict=True)
    return stored, instructions, used


def rewrite_lessons(agent_dir: Path, workspace: Path, record: dict) -> dict:
    """An agent's setup record changed by hand (its lessons edited): the
    record, its lessons, knowledge/owner-rules.md written again, and the
    blocks its instructions end with (``block``; ``block_all`` for an agent
    running from before lessons)."""
    record = {**record, "updated": time.time()}
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / _RECORD).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    agent_lessons.write(agent_dir, record)
    preset = normalise({g: record.get(g) or [] for g in _KEPT_GROUPS}, TOTALS)
    facts = [{"content": x["text"], "source": x.get("source") or {}} for x in agent_lessons.from_record(record)
             if x["kind"] == "fact"]
    folder = workspace / "knowledge"
    if preset["rules"] or preset["escalation"] or preset["templates"] or facts:
        folder.mkdir(parents=True, exist_ok=True)
    _write_or_drop(folder / "owner-rules.md", (preset["rules"] or preset["escalation"] or preset["templates"] or facts)
                   and _owner_rules_md({**preset, "facts": facts}))
    knowledge = [k for k in record.get("knowledge") or [] if isinstance(k, dict) and k.get("title")]
    skills = [n for n in record.get("skills") or [] if isinstance(n, str)]
    tables = [t for t in record.get("tables") or [] if isinstance(t, dict) and t.get("name")]
    return {"block": rules_block(preset, knowledge, skills, tables=tables, facts=len(facts)),
            "block_all": rules_block(preset, knowledge, skills, tables=tables)}


async def deliver(slug: str, entry: dict, owner_id: str, agent_dir: Path, done: dict) -> tuple[bool, str, bool]:
    """What a written setup (``seed``) still needs: the owner's block in the
    agent's stored instructions, spreadsheets as datastore tables — through
    its REST while it runs, else at its next start — and its documents
    indexed (its lessons it reads itself). An agent running from before
    lessons gets the block with every rule that fits, as before, until it's
    restarted. Returns (instructions stored, the instructions, running)."""
    from captain_claw.flight_deck import server as srv
    running = srv._process_is_alive(slug)
    stored, instructions, done["block"] = await instruct(slug, owner_id, done["block"], done["block_all"])

    def port_of() -> int:
        return (srv._load_process_registry().get(slug) or {}).get("web_port") or 0

    if done["tables"] or done["drop"]:
        if running:
            schedule_tables(slug, port_of, str(entry.get("web_auth") or ""), done["drop"], done["tables"], agent_dir)
        else:
            await asyncio.to_thread(ptables.leave_tables, agent_dir, done["drop"], done["tables"])
    # Its documents indexed now, not at its first search; one that is off indexes them when it starts.
    if done["knowledge"] and running:
        schedule_index(slug, port_of, str(entry.get("web_auth") or ""))
    return stored, instructions, running


async def add_document(slug: str, owner_id: str, name: str, text: str, raw: bytes = b"", title: str = "",
                       note: str = "", usage: str = "auto") -> dict:
    """One document into an agent's knowledge, as a setup gives one: whole,
    in parts with a section map when it is long, a spreadsheet's sheets as
    datastore tables too, indexed — and not read (its owner reads it for
    rules and skills in Way of working, when they want). The same name again
    takes the earlier copy's place. How it went, for the agent to tell."""
    from captain_claw.flight_deck import server as srv
    entry = srv._load_process_registry().get(slug) or {}
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    root = owner_root(owner_id)
    staged = await asyncio.to_thread(pdocs.stage, root, name, text, "chat", slug,
                                     raw if ptables.is_spreadsheet(name) else None)
    try:
        full = await asyncio.to_thread(pdocs.load_staged, root, staged["id"]) or {}
        had = {e.get("doc") for e in (await asyncio.to_thread(earlier_setup, agent_dir, workspace)).get("knowledge") or []
               if isinstance(e, dict)}
        docs = [{**{k: full.get(k) for k in ("text", "pages", "pages_exact", "truncated", "origin", "agent", "source",
                                             "tables")},
                 "name": name, "read": False, "description": note[:_MAX_DESCRIPTION],
                 "usage": usage if usage in USAGES else "auto"}]
        preset_id = await asyncio.to_thread(save_docs, owner_id, docs, "")
        body = {"id": preset_id, "knowledge": [{"doc": name, "title": title.strip() or Path(name).stem, "summary": note}]}
        done = await asyncio.to_thread(seed, agent_dir, workspace, owner_id, body, "add")
    finally:
        shutil.rmtree(root / "staged" / staged["id"], ignore_errors=True)
    await deliver(slug, entry, owner_id, agent_dir, done)
    written = next(iter(done["knowledge"]), {})
    log.info("agent preset: a document added from a chat", agent=slug, doc=name, chars=len(text),
             parts=written.get("parts"), tables=len(done["tables"]))
    return {"doc": name, "title": written.get("title") or name, "file": written.get("file") or "",
            "parts": int(written.get("parts") or 1), "sections": int(written.get("sections") or 0),
            "map": written.get("map") or "", "chars": len(text), "truncated": bool(full.get("truncated")),
            "tables": [t["name"] for t in done["tables"]], "replaced": name in had, "warnings": done["warnings"]}


@router.post("/apply")
async def apply(body: ApplyBody, user: dict = Depends(get_managed_user)) -> dict:
    """Give one of the caller's process agents the reviewed preset — added to
    what an earlier setup gave it, or in its place: skills and knowledge
    into its folder, the owner's block in its instructions (stored for the
    owner, and returned for the chat to send), its rules, when-to-ask and
    facts as lessons it reads with each message, and the spreadsheets as
    datastore tables once it answers — or, when it is off, at its next
    start. Works on a running agent: no restart."""
    from captain_claw.flight_deck import server as srv
    owner_id = _owner(user)
    slug = body.agent
    entry = _own_process(slug, owner_id)
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    try:
        done = await asyncio.to_thread(seed, agent_dir, workspace, owner_id, body.preset, body.mode)
    except Exception:
        log.warning("agent preset: could not write the setup", agent=slug, exc_info=True)
        raise HTTPException(500, "The setup couldn't be written into the agent's folder.")

    stored, instructions, running = await deliver(slug, entry, owner_id, agent_dir, done)
    log.info("agent preset: applied", agent=slug, mode=body.mode, **done["counts"])
    return {
        "ok": True,
        "mode": body.mode,
        "stored": stored,
        "instructions": instructions,
        "block": done["block"] and f"{_BLOCK_OPEN}\n{done['block']}\n{_BLOCK_CLOSE}",
        "counts": done["counts"],
        "totals": done["totals"],
        "skills": done["skills"],
        "knowledge": [k["title"] for k in done["knowledge"]],
        "parts": sum(k["parts"] for k in done["knowledge"] if k["parts"] > 1),
        "sections": sum(int(k.get("sections") or 0) for k in done["knowledge"]),
        "tables": [t["name"] for t in done["tables"]],
        "warnings": done["warnings"],
        "running": running,
    }
