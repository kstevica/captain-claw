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
with the document and section (page) it came from — and the parts merged.
A document marked "knowledge only" (the default for reference material) isn't
read at all. Nothing reaches the agent yet: the dialog shows the preset for
review, and ``POST /fd/agent-preset/apply`` gives the agent the items the
person kept:

- rules, when-to-ask and templates → a short "Your owner's rules" block after
  its instructions (the full lists in ``knowledge/owner-rules.md``);
- intents → one skill each, in the agent's own skills folder
  (``~/.captain-claw/skills/<name>/SKILL.md`` under its HOME);
- documents → their whole text as Markdown in its workspace's ``knowledge/``
  folder — a long one split into parts small enough for semantic memory to
  index — with an ``INDEX.md`` and the glossary, for this agent only (not
  the owner-wide deep memory);
- facts → insights, through the agent's REST once it is up.

A running agent needs no restart for any of it: it re-reads its skills folder,
indexes its workspace on its next message, and the chat sends it the new
instructions. The documents' text waits server-side under the preset's id for
a day, so Apply doesn't upload them again. The owner's model reads them
(balanced tier first) and the tokens count toward the owner's daily allowance.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.flight_deck import file_library
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.flight_deck.file_library import (
    _auth_enabled,
    _quota_spent,
    _resolve_tier,
    call_model,
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
# The most of each group a preset keeps.
LIMITS = {"rules": 30, "intents": 12, "facts": 40, "escalation": 15, "templates": 8,
          "glossary": 60, "knowledge": 12}

_MAX_FILES = 12
_MAX_INSTRUCTIONS = 20000
_MAX_READ_CHARS = 4_000_000            # the most one setup reads (~1M tokens); all of it is kept anyway
_STEP_MIN, _STEP_MAX = 4000, 100_000   # tokens of material in one step (step_tokens)
_PROMPT_TOKENS = 3000                  # the prompt around a step's material
_PART_BYTES = 200_000                  # a knowledge file: semantic memory skips files over 256 KB
_JOB_KEEP = 3600                       # a finished read's result waits this long
_TIERS = ("balanced", "reason", "fast")  # a careful read beats a quick one here
_MAX_TOKENS = 8000
_RETRY_TOKENS = 16000
_CALL_TIMEOUT = 300.0
_RETRY_NUDGE = "Reply now with the JSON object only: no thinking aloud, no prose."
_TTL = 24 * 3600                       # a prepared preset nobody created waits this long
_BLOCK_CAP = 5000                      # the rules block in the agent's instructions
_MAX_INSTRUCTIONS_TOTAL = 64000        # AgentConfig.fleet_instructions' own cap
_FACTS_READY_S = 90.0
_TAG = "owner-setup"
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
        v = _line(v, limit)
        if v and v not in out:
            out.append(v)
    return out[:count]


def normalise(raw: Any) -> dict:
    """A preset with every group present, each item cleaned, clipped and
    deduplicated — the model's reply and the dialog's edits alike."""
    data = raw if isinstance(raw, dict) else {}

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
        return out[:LIMITS[group]]

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
        "rules": unique([{"text": _line(r.get("text"), 400), "source": _source(r.get("source"))}
                         for r in items("rules")], "text", "rules"),
        "intents": intents[:LIMITS["intents"]],
        "facts": unique(facts, "content", "facts"),
        "escalation": unique([{"text": _line(r.get("text"), 400), "source": _source(r.get("source"))}
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
    return await asyncio.to_thread(pdocs.stage, root, name, text, "upload"), ""


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
    return await asyncio.to_thread(pdocs.stage, root, name, text, "library", agent), ""


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
    """Keep a preset's documents — their whole text — until it is applied."""
    preset_id = secrets.token_hex(12)
    folder = _preset_dir(owner_id, preset_id)
    assert folder is not None
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    meta = {"created": time.time(), "archetype": archetype, "docs": []}
    for n, d in enumerate(docs):
        file = f"{n}.txt"
        (folder / "docs" / file).write_text(d["text"], encoding="utf-8")
        meta["docs"].append({"name": d["name"], "file": file, "chars": len(d["text"]),
                             "pages": d.get("pages") or 0, "read": bool(d.get("read", True)), **note(d)})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return preset_id


def load_docs(owner_id: str, preset_id: str) -> list[dict]:
    """A waiting preset's documents (``name``, ``text``, ``pages``, ``read``,
    the person's ``description`` and ``usage``); empty when it is gone,
    expired, or another person's."""
    folder = _preset_dir(owner_id, preset_id)
    if folder is None or not (folder / "meta.json").is_file():
        return []
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if time.time() - float(meta.get("created") or 0) > _TTL:
            return []
        return [{"name": d["name"], "text": (folder / "docs" / d["file"]).read_text(encoding="utf-8"),
                 "pages": int(d.get("pages") or 0), "read": bool(d.get("read", True)), **note(d)}
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
    less the prompt around it, within 4k–100k tokens. A tier that doesn't
    say its window is taken at 32k (Ollama) or 128k."""
    ctx = int(tier.get("input_ctx") or 0) or (32768 if tier.get("provider") == "ollama" else 128000)
    return max(_STEP_MIN, min(_STEP_MAX, ctx // 2 - _PROMPT_TOKENS))


def _retry_tokens(tier: dict) -> int:
    ctx = int(tier.get("input_ctx") or 0) or (32768 if tier.get("provider") == "ollama" else 128000)
    return min(_RETRY_TOKENS, max(_MAX_TOKENS, ctx - step_tokens(tier) - _PROMPT_TOKENS))


def plan_steps(instructions: str, docs: list[dict], step_tok: int) -> tuple[list[dict], list[str]]:
    """The steps the model reads the material in — each at most *step_tok*
    tokens, small documents sharing one, a long one across many, the
    instructions in the first — and what the person should know about what
    wasn't read. Only documents marked ``read`` are; all of them reach the
    agent's knowledge either way."""
    limit = step_tok * pdocs.CHARS_PER_TOKEN
    budget = _MAX_READ_CHARS
    notes: list[str] = []
    pieces: list[dict] = []
    for d in docs:
        if not d.get("read", True):
            continue
        text = d["text"]
        if budget <= 0:
            notes.append(f"{d['name']}: not read — the documents together are more than one setup reads "
                         "(about a million tokens). It is still kept for the agent to look things up in.")
            continue
        if len(text) > budget:
            notes.append(f"{d['name']}: only the first {budget:,} characters were read (about a million "
                         "tokens in all). The whole document is still kept for the agent.")
            text = text[:budget]
        budget -= len(text)
        parts = pdocs.split_text(text, limit)
        for n, p in enumerate(parts, 1):
            pieces.append({"doc": d, "text": p["text"], "label": p["label"], "part": n, "parts": len(parts)})

    steps: list[dict] = []
    cur: list[dict] = []
    size = len(instructions.encode("utf-8"))
    for piece in pieces:
        piece_size = len(piece["text"].encode("utf-8"))
        if cur and size + piece_size > limit:
            steps.append({"pieces": cur})
            cur, size = [], 0
        cur.append(piece)
        size += piece_size
    if cur or (instructions and not steps):
        steps.append({"pieces": cur})
    for st in steps:
        names = [p["doc"]["name"] + (f", {p['label']}" if p["parts"] > 1 and p["label"] else "")
                 for p in st["pieces"]]
        st["label"] = "; ".join(names) or "your instructions"
    return steps, notes


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


def _doc_part(piece: dict) -> str:
    d = piece["doc"]
    title = f"### {d['name']}"
    if piece["parts"] > 1:
        title += f" — {piece['label'] or 'part'} (part {piece['part']} of {piece['parts']} of this document)"
    head = [title]
    if d.get("description"):
        head.append(f"The owner says: {d['description']}")
    hint = USAGES.get(d.get("usage") or "auto", ("", ""))[1]
    if hint:
        head.append(f"Use it as: {hint}")
    return "\n".join(head) + "\n" + piece["text"]


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
    if n == 1 and instructions:
        parts.append(f"## The owner's instructions\n{instructions}")
    if step["pieces"]:
        parts.append("## The owner's documents\n" + "\n\n".join(_doc_part(p) for p in step["pieces"]))
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
    """One read of a person's material, running in the background."""
    id: str
    owner: str
    state: str = "running"          # running | done | error | cancelled
    phase: str = "reading"          # reading | merging
    step: int = 0
    steps: int = 0
    current: str = ""
    tokens: int = 0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    result: dict | None = None
    task: asyncio.Task | None = None

    def view(self) -> dict:
        out = {"id": self.id, "state": self.state, "phase": self.phase, "step": self.step,
               "steps": self.steps, "current": self.current, "tokens": self.tokens,
               "started": self.started, "finished": self.finished, "error": self.error}
        if self.result is not None:
            out["result"] = self.result
        return out


_JOBS: dict[str, ReadJob] = {}


def _forget_old_jobs(now: float) -> None:
    for jid in [j.id for j in _JOBS.values() if j.finished and now - j.finished > _JOB_KEEP]:
        _JOBS.pop(jid, None)


class _UnreadableError(Exception):
    pass


async def _ask(tier: dict, system: str, prompt: str, owner_id: str, job: ReadJob) -> dict | None:
    """One step's model call — again, with room and a nudge, when the reply
    isn't JSON (a reasoning model that thought past the budget, prose around
    it). None when it never is."""
    for attempt in range(2):
        content, tokens = await call_model(
            tier, system, prompt if attempt == 0 else f"{prompt}\n\n{_RETRY_NUDGE}", owner_id,
            max_tokens=_MAX_TOKENS if attempt == 0 else _retry_tokens(tier), request_timeout=_CALL_TIMEOUT)
        job.tokens += int(tokens or 0)
        got = _parse_part(content)
        if got is not None:
            return got
    return None


async def merge_parts(tier: dict, parts: list[dict], owner_id: str, job: ReadJob, depth: int = 0) -> dict:
    """What every step found, as one preset: the model merges the parts
    (a halving tree when they are too many for one call); without it — or
    when its answer can't be read — they are combined as they are."""
    if len(parts) == 1:
        return parts[0]
    payload = json.dumps(parts, ensure_ascii=False)
    if len(payload) > step_tokens(tier) * pdocs.CHARS_PER_TOKEN:
        if len(parts) > 2 and depth < 6:
            mid = len(parts) // 2
            left = await merge_parts(tier, parts[:mid], owner_id, job, depth + 1)
            right = await merge_parts(tier, parts[mid:], owner_id, job, depth + 1)
            return await merge_parts(tier, [left, right], owner_id, job, depth + 1)
        return combine(parts)
    system = _MERGE_PROMPT_FILE.read_text(encoding="utf-8")
    try:
        got = await _ask(tier, system, f"## The parts\n{payload}", owner_id, job)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("agent preset: merging failed, combined as they are", error=str(exc))
        got = None
    return whole_summaries(got, parts) if got is not None else combine(parts)


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
            preset_id: str) -> dict:
    """The read's result: every document with a knowledge entry — summarised
    by the model, else by the person's own words; one the model invented a
    name for doesn't get in."""
    names = [d["name"] for d in docs]
    known: dict[str, dict] = {}
    for k in preset["knowledge"]:
        name = k["doc"] if k["doc"] in names else next((n for n in names if k["doc"].startswith(n)), "")
        if name and name not in known:
            known[name] = {**k, "doc": name}
    preset["knowledge"] = [dict(known.get(n) or {"doc": n, "title": Path(n).stem[:120], "summary": ""})
                           for n in names]
    for k, d in zip(preset["knowledge"], docs):
        k["summary"] = k["summary"] or d["description"]
    preset["warnings"] = (warnings + preset["warnings"])[:20]
    return {
        "id": preset_id,
        "preset": preset,
        "docs": [{"name": d["name"], "chars": len(d["text"]), "pages": d.get("pages") or 0,
                  "pages_exact": bool(d.get("pages_exact")), "truncated": bool(d.get("truncated")),
                  "origin": d.get("origin") or "upload", "description": d["description"], "usage": d["usage"],
                  "read": bool(d.get("read", True)), **({"agent": d["agent"]} if d.get("agent") else {})}
                 for d in docs],
        "model": model,
        "steps": steps,
        "tokens": tokens,
    }


async def run_read(job: ReadJob, tier: dict | None, arch: dict | None, instructions: str,
                   docs: list[dict], warnings: list[str], owner_id: str) -> None:
    """Read the material step by step, merge what each step found, and leave
    the result on *job* for the dialog to fetch. A step whose answer can't
    be read is reported and skipped; the person's allowance running out
    stops the reading but keeps what was found."""
    try:
        steps, notes = plan_steps(instructions, docs, step_tokens(tier) if tier else _STEP_MAX)
        warnings = warnings + notes
        job.steps = len(steps)
        parts: list[dict] = []
        failed: str = ""
        ran_out = False
        system = _PROMPT_FILE.read_text(encoding="utf-8")
        for n, step in enumerate(steps, 1):
            if n > 1 and await _quota_spent(owner_id):
                warnings.append(f"Today's allowance ran out after part {n - 1} of {len(steps)}, so the "
                                "rest wasn't read. The documents are still kept whole for the agent.")
                ran_out = True
                break
            job.step, job.current = n, step["label"]
            try:
                got = await _ask(tier, system, step_prompt(arch, instructions, step, n, len(steps)),
                                 owner_id, job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("agent preset: a step failed", step=n, model=(tier or {}).get("model"), error=str(exc))
                failed = failed or str(exc)
                got = None
            if got is None:
                warnings.append(f"Part {n} of {len(steps)} ({step['label']}) couldn't be read by the model, "
                                "so nothing was taken from it.")
                continue
            parts.append(got)
        if steps and not parts and not ran_out:
            model = (tier or {}).get("model") or "the model"
            raise _UnreadableError(f"Couldn't read the material with {model}: {failed}" if failed
                              else "The model's answers couldn't be read. Try again, or with less material.")
        if len(parts) > 1:
            job.phase, job.current = "merging", "Putting the parts together"
            preset = combine(parts) if ran_out else await merge_parts(tier, parts, owner_id, job)
        else:
            preset = parts[0] if parts else normalise({})
        job.result = _finish(preset, docs, warnings, str((tier or {}).get("model") or ""), len(steps),
                             job.tokens, job.id)
        job.state = "done"
    except asyncio.CancelledError:
        job.state = "cancelled"
        discard(owner_id, job.id)
        raise
    except Exception as exc:
        if not isinstance(exc, _UnreadableError):
            log.warning("agent preset: reading failed", exc_info=True)
        job.state, job.error = "error", str(exc) or "Reading failed."
        discard(owner_id, job.id)
    finally:
        job.finished = time.time()


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
async def reader(user: dict = Depends(get_managed_user)) -> dict:
    """The model that would read the material, and how much of it one step
    takes — what the dialog estimates steps and tokens from."""
    tier = await _resolve_tier(_owner(user), prefer=_TIERS)
    if not tier:
        return {"model": "", "step_tokens": 0}
    return {"model": str(tier.get("model") or ""), "step_tokens": step_tokens(tier)}


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
        docs.append({**{k: staged.get(k) for k in ("text", "pages", "pages_exact", "truncated", "origin", "agent")},
                     "name": name, "read": choice.read, **note(choice.model_dump())})
    if not instructions and not docs:
        raise HTTPException(400, " ".join(warnings) or "None of the documents are there any more.")

    tier = None
    if instructions or any(d["read"] for d in docs):
        if await _quota_spent(owner_id):
            raise HTTPException(429, "Today's allowance is used up, so the documents can't be read now. "
                                     "Keep them as knowledge only, or try again tomorrow.")
        tier = await _resolve_tier(owner_id, prefer=_TIERS)
        if not tier:
            raise HTTPException(409, "No model is set up to read the documents with. Ask your admin.")
    preset_id = await asyncio.to_thread(save_docs, owner_id, docs, archetype)
    arch = await _archetype(owner_id, archetype)

    now = time.time()
    _forget_old_jobs(now)
    key = owner_key(owner_id)
    # One read at a time per person: a new one replaces one still running.
    for other in [j for j in _JOBS.values() if j.owner == key and j.state == "running"]:
        if other.task is not None:
            other.task.cancel()
    job = ReadJob(id=preset_id, owner=key)
    _JOBS[preset_id] = job
    job.task = asyncio.get_running_loop().create_task(
        run_read(job, tier, arch, instructions, docs, warnings, owner_id))
    return job.view()


def _job(preset_id: str, user: dict) -> ReadJob:
    job = _JOBS.get(str(preset_id or ""))
    if job is None or job.owner != owner_key(_owner(user)):
        raise HTTPException(404, "That reading isn't here any more — Flight Deck may have restarted. "
                                 "Read the material again.")
    return job


@router.get("/prepare/{preset_id}")
async def read_status(preset_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Where a read stands: its step, the tokens spent, and — once done — the
    preset for review."""
    _forget_old_jobs(time.time())
    return _job(preset_id, user).view()


@router.delete("/prepare/{preset_id}")
async def cancel_read(preset_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Stop a read. The step under way is abandoned; its documents are dropped."""
    job = _job(preset_id, user)
    if job.state == "running" and job.task is not None:
        job.task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(job.task), timeout=5)
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
        lines += ["## Rules", *[f"- {r['text']}" for r in preset["rules"]], ""]
    if preset["escalation"]:
        lines += ["## Ask your owner first", *[f"- {r['text']}" for r in preset["escalation"]], ""]
    for t in preset["templates"]:
        lines += [f"## Template: {t['name']}", "", t["body"], ""]
    return "\n".join(lines)


def write_knowledge(folder: Path, preset: dict, docs: list[dict]) -> list[dict]:
    """The kept documents — their whole text — as Markdown, with an index,
    the glossary and the full rules; the documents written (``file``,
    ``title``, ``parts``). A document too long for one file the agent's
    semantic memory indexes is an overview plus parts (``<name>-part-01.md``
    …), each cut at a page or heading."""
    folder.mkdir(parents=True, exist_ok=True)
    doc_of = {d["name"]: d for d in docs}
    taken: set[str] = set()
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
        text = doc["text"].strip()
        parts = pdocs.split_text(text, _PART_BYTES)
        if len(parts) <= 1:
            (folder / f"{stem}.md").write_text("\n".join(head) + "\n---\n\n" + text + "\n", encoding="utf-8")
        else:
            listing: list[str] = []
            for n, part in enumerate(parts, 1):
                name = f"{stem}-part-{n:02d}"
                taken.add(name)
                where = f" ({part['label']})" if part["label"] else ""
                (folder / f"{name}.md").write_text("\n".join([
                    f"# {title} — part {n} of {len(parts)}{where}", "", source,
                    f"The whole document: {stem}.md.", "", "---", "", part["text"].strip(), ""]), encoding="utf-8")
                listing.append(f"- [Part {n}{where}]({name}.md)")
            (folder / f"{stem}.md").write_text("\n".join([
                *head, f"It is long, so it is kept in {len(parts)} parts:", "", *listing, ""]), encoding="utf-8")
        written.append({"file": f"{stem}.md", "title": title, "summary": k["summary"], "use": use,
                        "parts": len(parts)})
    if written:
        (folder / "INDEX.md").write_text("\n".join([
            "# Knowledge", "",
            "Documents your owner gave you. Read the one that fits before you answer from memory.", "",
            *[f"- [{w['title']}]({w['file']})" + (f" — {w['summary']}" if w["summary"] else "")
              + (f" ({w['use'].lower()})" if w["use"] else "")
              + (f" — {w['parts']} parts" if w["parts"] > 1 else "")
              for w in written], ""]), encoding="utf-8")
    if preset["glossary"]:
        (folder / "glossary.md").write_text("\n".join([
            "# Glossary", "", *[f"- **{g['term']}** — {g['meaning']}" for g in preset["glossary"]], ""]),
            encoding="utf-8")
    if preset["rules"] or preset["escalation"] or preset["templates"]:
        (folder / "owner-rules.md").write_text(_owner_rules_md(preset), encoding="utf-8")
    return written


def rules_block(preset: dict, knowledge: list[dict], skills: list[str], cap: int = _BLOCK_CAP) -> str:
    """The short block the new agent's instructions end with — at most
    ``cap`` characters. When-to-ask goes in before the other rules; what
    doesn't fit is pointed at in knowledge/owner-rules.md."""
    head = ["## Your owner's rules", "",
            "Set up by your owner. Where they differ from the general procedure above, "
            "follow these."]
    tail: list[str] = []
    if preset["templates"]:
        tail += ["", "### Templates",
                 "Use these formats when they fit (full text in knowledge/owner-rules.md): "
                 + ", ".join(t["name"] for t in preset["templates"]) + "."]
    if skills:
        tail += ["", "### What you handle",
                 "You have a skill for each of these — read it before you start: "
                 + ", ".join(skills) + "."]
    if knowledge or preset["glossary"]:
        line = "Your workspace's knowledge/ folder holds what your owner gave you"
        if knowledge:
            line += (f" ({len(knowledge)} document{'s' if len(knowledge) != 1 else ''}: "
                     + "; ".join(k["title"] for k in knowledge)[:600] + "; index: knowledge/INDEX.md)")
        if preset["glossary"]:
            line += ", and their terms in knowledge/glossary.md"
        tail += ["", "### Your knowledge", line + ". Look there before you answer from memory."]

    budget = cap - len("\n".join(head + tail)) - 200  # room for headings and the pointer
    kept: dict[str, list[str]] = {"escalation": [], "rules": []}
    dropped = 0
    for group in ("escalation", "rules"):
        for r in preset[group]:
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
    if dropped:
        body += ["", f"({dropped} more rule{'s' if dropped != 1 else ''} in knowledge/owner-rules.md — read it.)"]
    if not body and not tail:
        return ""
    return "\n".join(head + body + tail)[:cap]


def with_block(instructions: str, block: str) -> str:
    """The archetype's instructions followed by the owner's block, within the
    instructions' own cap (the block wins: the general part is clipped)."""
    if not block:
        return instructions
    room = _MAX_INSTRUCTIONS_TOTAL - len(block) - 2
    base = (instructions or "").rstrip()[:max(0, room)]
    return f"{base}\n\n{block}" if base else block


def seed(agent_dir: Path, workspace: Path, owner_id: str, body: Any) -> dict:
    """Write the reviewed preset into an agent's folder. Returns the
    ``block`` its instructions end with, the ``facts`` to give it as
    insights, and what was written: ``skills`` (names) and ``knowledge``
    (documents) — plus how many of each other group it got, in ``counts``.
    The waiting documents are dropped afterwards."""
    data = body if isinstance(body, dict) else {}
    preset = normalise(data)
    preset_id = str(data.get("id") or "")
    docs = load_docs(owner_id, preset_id) if preset_id else []
    if preset["knowledge"] and not docs and preset_id:
        log.warning("agent preset: its documents are gone (expired?)", preset=preset_id)
    skills = write_skills(agent_dir / "data" / "home-config-parent" / ".captain-claw" / "skills",
                          preset["intents"])
    knowledge = write_knowledge(workspace / "knowledge", preset, docs)
    if preset_id:
        discard(owner_id, preset_id)
    return {
        "block": rules_block(preset, knowledge, skills),
        "facts": preset["facts"],
        "skills": skills,
        "knowledge": knowledge,
        "counts": {"rules": len(preset["rules"]), "escalation": len(preset["escalation"]),
                   "templates": len(preset["templates"]), "glossary": len(preset["glossary"]),
                   "intents": len(skills), "facts": len(preset["facts"]), "knowledge": len(knowledge)},
    }


async def _post_facts(port: int, token: str, facts: list[dict]) -> int:
    import httpx
    params = {"token": token} if token else {}
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        given = 0
        for f in facts:
            try:
                resp = await client.post(f"{base}/api/insights", params=params, json={
                    "content": f["content"], "category": f["category"],
                    "importance": f["importance"], "tags": _TAG})
                if resp.status_code < 300:
                    given += 1
            except Exception:
                pass
        return given


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


async def give_facts(slug: str, port_of: Callable[[], int], token: str, facts: list[dict],
                     timeout: float = _FACTS_READY_S) -> int:
    """Give a freshly spawned agent its owner's facts as insights. Never
    raises: a failure is logged and the agent simply starts without them."""
    if not facts:
        return 0
    try:
        port = await _wait_ready(port_of, token, timeout)
        if not port:
            log.warning("agent preset: the agent never came up, facts not given", agent=slug)
            return 0
        given = await _post_facts(port, token, facts)
        log.info("agent preset: facts given", agent=slug, given=given, of=len(facts))
        return given
    except Exception:
        log.warning("agent preset: giving facts failed", agent=slug, exc_info=True)
        return 0


# Facts still on their way to a new agent: a task nobody holds can be
# garbage-collected before it runs.
_PENDING: set[asyncio.Task] = set()


def schedule_facts(slug: str, port_of: Callable[[], int], token: str, facts: list[dict]) -> None:
    if not facts:
        return
    try:
        task = asyncio.get_running_loop().create_task(give_facts(slug, port_of, token, facts))
    except RuntimeError:
        log.warning("agent preset: no loop to give facts on", agent=slug)
        return
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


# ── Apply (to the agent the dialog was opened on) ─────────────────────


class ApplyBody(BaseModel):
    agent: str = Field(min_length=1, max_length=128)
    # The prepared id and the items the person kept, as they edited them.
    preset: dict


@router.post("/apply")
async def apply(body: ApplyBody, user: dict = Depends(get_managed_user)) -> dict:
    """Give one of the caller's process agents the reviewed preset: skills and
    knowledge into its folder, the owner's block after its instructions
    (stored for the owner, and returned for the chat to send), facts as
    insights once it answers. Works on a running agent — no restart."""
    from captain_claw.flight_deck import server as srv
    owner_id = str(user.get("id") or "") if _auth_enabled() else ""
    slug = body.agent
    entry = _own_process(slug, owner_id)
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    try:
        done = await asyncio.to_thread(seed, agent_dir, workspace, owner_id, body.preset)
    except Exception:
        log.warning("agent preset: could not write the setup", agent=slug, exc_info=True)
        raise HTTPException(500, "The setup couldn't be written into the agent's folder.")

    # Without accounts the browser keeps the instructions: it appends the block.
    instructions = ""
    if owner_id and done["block"]:
        current = (await srv._owner_agent_instructions(owner_id, "process")).get(slug, "")
        instructions = with_block(current, done["block"])
        await srv._set_owner_agent_instructions(owner_id, "process", slug, instructions, strict=True)
    if done["facts"]:
        schedule_facts(slug, lambda: (srv._load_process_registry().get(slug) or {}).get("web_port") or 0,
                       str(entry.get("web_auth") or ""), done["facts"])
    log.info("agent preset: applied", agent=slug, **done["counts"])
    return {
        "ok": True,
        "instructions": instructions,
        "block": done["block"],
        "counts": done["counts"],
        "skills": done["skills"],
        "knowledge": [k["title"] for k in done["knowledge"]],
        "parts": sum(k["parts"] for k in done["knowledge"] if k["parts"] > 1),
        "running": srv._process_is_alive(slug),
    }
