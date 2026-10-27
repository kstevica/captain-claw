"""Agent presets — the "Advanced" setup of the kiosk's New agent dialog.

While creating an agent from an archetype, the person can add instructions and
documents (their way of working, policies, FAQs, procedures, price lists,
templates). ``POST /fd/agent-preset/prepare`` reads the documents and has the
owner's model pull out what the new agent must know and do — rules, the
requests it handles (intents, each with its steps), facts, when to ask first,
output templates, a glossary and a summary per document — every item with the
document and section it came from. Nothing reaches an agent yet: the dialog
shows the preset for review, and the spawn takes the items the person kept
(``AgentConfig.preset``) and seeds the new agent before it starts:

- rules, when-to-ask and templates → a short "Your owner's rules" block after
  the archetype's instructions (the full lists in ``knowledge/owner-rules.md``);
- intents → one skill each, in the agent's own skills folder
  (``~/.captain-claw/skills/<name>/SKILL.md`` under its HOME);
- documents → Markdown in its workspace's ``knowledge/`` folder with an
  ``INDEX.md`` and the glossary — semantic memory indexes them, for this
  agent only (not the owner-wide deep memory);
- facts → insights, through the agent's REST once it is up.

The documents' text waits server-side under the preset's id for a day, so
Create doesn't upload them again. The owner's model reads them (balanced
tier first) and the tokens count toward the owner's daily allowance.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

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

GROUPS = ("rules", "intents", "facts", "escalation", "templates", "glossary", "knowledge")
FACT_CATEGORIES = ("contact", "decision", "preference", "fact", "deadline", "project",
                   "workflow", "feedback", "reference")
# The most of each group a preset keeps.
LIMITS = {"rules": 30, "intents": 12, "facts": 40, "escalation": 15, "templates": 8,
          "glossary": 60, "knowledge": 12}

_MAX_FILES = 12
_MAX_FILE_BYTES = 25 * 1024 * 1024    # the dialog's own guard, again here
_MAX_PER_FILE = 40000                  # characters, as the archetype forge reads
_MAX_TOTAL = 160000
_MAX_INSTRUCTIONS = 20000
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


def _read_one(raw: bytes, suffix: str) -> tuple[str | None, str | None]:
    from captain_claw.tools.summarize_files import SummarizeFilesTool
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".txt") as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        return SummarizeFilesTool._read_file_content(tmp_path)
    except Exception as exc:
        return None, str(exc)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def read_documents(files: list[UploadFile]) -> tuple[list[dict], list[str]]:
    """Each uploaded document's text (``name``, ``text``, ``chars``,
    ``truncated``) and what the person should know about the ones that
    didn't come through — said without the model."""
    docs: list[dict] = []
    warnings: list[str] = []
    taken: set[str] = set()
    total = 0
    for uf in [f for f in files or [] if f and f.filename][:_MAX_FILES + 1]:
        name = _display_name(uf.filename, taken)
        if len(docs) >= _MAX_FILES:
            warnings.append(f"Only the first {_MAX_FILES} documents were read.")
            break
        try:
            raw = await uf.read()
        except Exception:
            raw = b""
        if not raw:
            warnings.append(f"{name}: the file is empty.")
            continue
        if len(raw) > _MAX_FILE_BYTES:
            warnings.append(f"{name}: larger than 25 MB, left out.")
            continue
        text, err = await asyncio.to_thread(_read_one, raw, Path(uf.filename).suffix)
        if not (text or "").strip():
            log.info("agent preset: no text in a document", file=name, error=err)
            warnings.append(f"{name}: no text could be read from it (a scanned PDF or a picture?), "
                            "so it was left out.")
            continue
        remaining = _MAX_TOTAL - total
        if remaining <= 0:
            warnings.append(f"{name}: left out — the documents together are too long to read at once.")
            continue
        cut = min(_MAX_PER_FILE, remaining)
        truncated = len(text) > cut
        if truncated:
            warnings.append(f"{name}: only the first {cut:,} characters were read.")
        text = text[:cut]
        total += len(text)
        docs.append({"name": name, "text": text, "chars": len(text), "truncated": truncated})
    return docs, warnings


# ── Waiting presets (the documents between Prepare and Create) ────────


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def _root() -> Path:
    return _data_dir() / ".agent-presets"


def owner_key(owner_id: str) -> str:
    """The folder a person's presets wait in; a deck without sign-in has one."""
    key = re.sub(r"[^A-Za-z0-9_-]", "", str(owner_id or ""))[:64]
    return key if key and _auth_enabled() else "_deck"


def _preset_dir(owner_id: str, preset_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{24}", str(preset_id or "")):
        return None
    return _root() / owner_key(owner_id) / preset_id


def save_docs(owner_id: str, docs: list[dict], archetype: str) -> str:
    preset_id = secrets.token_hex(12)
    folder = _preset_dir(owner_id, preset_id)
    assert folder is not None
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    meta = {"created": time.time(), "archetype": archetype, "docs": []}
    for n, d in enumerate(docs):
        file = f"{n}.txt"
        (folder / "docs" / file).write_text(d["text"], encoding="utf-8")
        meta["docs"].append({"name": d["name"], "file": file, "chars": d["chars"]})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return preset_id


def load_docs(owner_id: str, preset_id: str) -> list[dict]:
    """A waiting preset's documents (``name``, ``text``); empty when it is
    gone, expired, or another person's."""
    folder = _preset_dir(owner_id, preset_id)
    if folder is None or not (folder / "meta.json").is_file():
        return []
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if time.time() - float(meta.get("created") or 0) > _TTL:
            return []
        return [{"name": d["name"], "text": (folder / "docs" / d["file"]).read_text(encoding="utf-8")}
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
    """Drop presets nobody created an agent from within a day."""
    now = time.time() if now is None else now
    gone = 0
    root = _root()
    if not root.is_dir():
        return 0
    for owner in root.iterdir():
        if not owner.is_dir():
            continue
        for folder in owner.iterdir():
            try:
                created = json.loads((folder / "meta.json").read_text(encoding="utf-8")).get("created")
            except Exception:
                created = folder.stat().st_mtime if folder.exists() else now
            if now - float(created or 0) > _TTL:
                shutil.rmtree(folder, ignore_errors=True)
                gone += 1
    return gone


# ── Prepare ───────────────────────────────────────────────────────────


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


def _user_prompt(arch: dict | None, instructions: str, docs: list[dict]) -> str:
    parts: list[str] = []
    if arch:
        parts.append(
            f"## The archetype\nRole: {arch.get('role') or arch.get('id')}\n"
            f"Description: {arch.get('description') or ''}\n\n"
            "Its general procedure (for reference — extract only what the owner's material adds):\n"
            f"{str(arch.get('fleet_instructions') or '')[:3000]}")
    if instructions:
        parts.append(f"## The owner's instructions\n{instructions}")
    if docs:
        parts.append("## The owner's documents\n" + "\n\n".join(
            f"### {d['name']}\n{d['text']}" for d in docs))
    return "\n\n".join(parts)


async def _extract(tier: dict, user_prompt: str, owner_id: str) -> dict:
    system = _PROMPT_FILE.read_text(encoding="utf-8")
    content, _tokens = await call_model(tier, system, user_prompt, owner_id,
                                        max_tokens=_MAX_TOKENS, request_timeout=_CALL_TIMEOUT)
    try:
        return parse_preset(content)
    except (ValueError, json.JSONDecodeError):
        # A reasoning model that thought past the budget, or prose around the
        # JSON: once more, with room and a nudge.
        content, _tokens = await call_model(tier, system, f"{user_prompt}\n\n{_RETRY_NUDGE}", owner_id,
                                            max_tokens=_RETRY_TOKENS, request_timeout=_CALL_TIMEOUT)
        return parse_preset(content)


@router.post("/prepare")
async def prepare(
    archetype: str = Form(""),
    instructions: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    user: dict = Depends(get_managed_user),
) -> dict:
    """Read the person's instructions and documents into a preset for review.
    Persists only the documents' text (under the returned id), never an agent."""
    owner_id = str(user.get("id") or "") if _auth_enabled() else ""
    instructions = (instructions or "").strip()[:_MAX_INSTRUCTIONS]
    if not instructions and not [f for f in files or [] if f and f.filename]:
        raise HTTPException(400, "Add some instructions or at least one document first.")
    if await _quota_spent(owner_id):
        raise HTTPException(429, "Today's allowance is used up, so the documents can't be read now. "
                                 "Create the agent without them, or try again tomorrow.")
    try:
        await asyncio.to_thread(prune)
    except Exception:
        log.debug("agent preset: prune failed", exc_info=True)

    docs, warnings = await read_documents(files)
    if not instructions and not docs:
        raise HTTPException(400, " ".join(warnings) or "None of the documents had text that could be read.")
    tier = await _resolve_tier(owner_id, prefer=_TIERS)
    if not tier:
        raise HTTPException(409, "No model is set up to read the documents with. Ask your admin.")
    arch = await _archetype(owner_id, archetype)
    try:
        preset = await _extract(tier, _user_prompt(arch, instructions, docs), owner_id)
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(502, "The model's answer couldn't be read. Try again, or with fewer documents.")
    except Exception as exc:
        log.warning("agent preset: model call failed", model=tier.get("model"), error=str(exc))
        raise HTTPException(502, f"Couldn't read the documents with {tier.get('model') or 'the model'}: {exc}")

    # Every document the person gave gets a knowledge entry, summary or not;
    # one the model invented a name for doesn't.
    names = [d["name"] for d in docs]
    known = {k["doc"]: k for k in preset["knowledge"] if k["doc"] in names}
    preset["knowledge"] = [known.get(n) or {"doc": n, "title": Path(n).stem[:120], "summary": ""}
                           for n in names]
    preset["warnings"] = (warnings + preset["warnings"])[:20]
    preset_id = await asyncio.to_thread(save_docs, owner_id, docs, archetype)
    return {
        "id": preset_id,
        "preset": preset,
        "docs": [{"name": d["name"], "chars": d["chars"], "truncated": d["truncated"]} for d in docs],
        "model": str(tier.get("model") or ""),
    }


# ── Seeding a new agent (from the spawn, before the agent starts) ─────


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
        lines += [f"Source: {cite} (from your owner, when you were created)", ""]
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
             "Set up by your owner when you were created, from their instructions and documents.", ""]
    if preset["rules"]:
        lines += ["## Rules", *[f"- {r['text']}" for r in preset["rules"]], ""]
    if preset["escalation"]:
        lines += ["## Ask your owner first", *[f"- {r['text']}" for r in preset["escalation"]], ""]
    for t in preset["templates"]:
        lines += [f"## Template: {t['name']}", "", t["body"], ""]
    return "\n".join(lines)


def write_knowledge(folder: Path, preset: dict, docs: list[dict]) -> list[dict]:
    """The kept documents as Markdown, with an index, the glossary and the
    full rules; the documents written (``file``, ``title``)."""
    folder.mkdir(parents=True, exist_ok=True)
    text_of = {d["name"]: d["text"] for d in docs}
    taken: set[str] = set()
    written: list[dict] = []
    for k in preset["knowledge"]:
        text = text_of.get(k["doc"])
        if text is None:
            continue
        file = f"{_file_stem(k['doc'], taken)}.md"
        title = k["title"] or Path(k["doc"]).stem
        head = [f"# {title}", "", f"Source: {k['doc']} — given by your owner when you were created.", ""]
        if k["summary"]:
            head += [k["summary"], ""]
        (folder / file).write_text("\n".join(head) + "\n---\n\n" + text.strip() + "\n", encoding="utf-8")
        written.append({"file": file, "title": title, "summary": k["summary"]})
    if written:
        (folder / "INDEX.md").write_text("\n".join([
            "# Knowledge", "",
            "Documents your owner gave you when you were created. Read the one that fits "
            "before you answer from memory.", "",
            *[f"- [{w['title']}]({w['file']})" + (f" — {w['summary']}" if w["summary"] else "")
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
            "Set up by your owner when you were created. Where they differ from the general "
            "procedure above, follow these."]
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


def seed(agent_dir: Path, workspace: Path, owner_id: str, body: Any) -> tuple[str, list[dict]]:
    """Write the reviewed preset into a new agent's folder, before it starts.
    Returns the block its instructions end with, and the facts to give it as
    insights once it is up. The waiting documents are dropped afterwards."""
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
    return rules_block(preset, knowledge, skills), preset["facts"]


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
