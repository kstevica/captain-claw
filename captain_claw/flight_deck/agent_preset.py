"""Agent presets — "your way of working", set up right after a kiosk agent is created.

The New agent dialog has a "set up its way of working next" box; with it
ticked, the agent is created as usual and the setup dialog opens on it. The
person adds instructions and documents (policies, FAQs, procedures, price
lists, templates) — uploaded, or picked from their File library — and may say
what each document is and how the agent should use it.
``POST /fd/agent-preset/prepare`` reads them and has the owner's model pull out
what the agent must know and do — rules, the requests it handles (intents, each
with its steps), facts, when to ask first, output templates, a glossary and a
summary per document — every item with the document and section it came from.
Nothing reaches the agent yet: the dialog shows the preset for review, and
``POST /fd/agent-preset/apply`` gives the agent the items the person kept:

- rules, when-to-ask and templates → a short "Your owner's rules" block after
  its instructions (the full lists in ``knowledge/owner-rules.md``);
- intents → one skill each, in the agent's own skills folder
  (``~/.captain-claw/skills/<name>/SKILL.md`` under its HOME);
- documents → Markdown in its workspace's ``knowledge/`` folder with an
  ``INDEX.md`` and the glossary — semantic memory indexes them, for this
  agent only (not the owner-wide deep memory);
- facts → insights, through the agent's REST once it is up.

A running agent needs no restart for any of it: it re-reads its skills folder,
indexes its workspace on its next message, and the chat sends it the new
instructions. The documents' text waits server-side under the preset's id for
a day, so Apply doesn't upload them again. The owner's model reads them
(balanced tier first) and the tokens count toward the owner's daily allowance.
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
from pydantic import BaseModel, Field

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


def note(raw: Any) -> dict:
    """The person's note on a document: what it is, and how to use it."""
    d = raw if isinstance(raw, dict) else {}
    usage = str(d.get("usage") or "auto")
    return {"description": _line(d.get("description"), _MAX_DESCRIPTION),
            "usage": usage if usage in USAGES else "auto"}


def _no_text(name: str, err: Any) -> str:
    log.info("agent preset: no text in a document", file=name, error=err)
    return f"{name}: no text could be read from it (a scanned PDF or a picture?), so it was left out."


async def _upload_text(uf: UploadFile, name: str) -> tuple[str, str]:
    """An uploaded document's text, or "" and what to tell the person."""
    try:
        raw = await uf.read()
    except Exception:
        raw = b""
    if not raw:
        return "", f"{name}: the file is empty."
    if len(raw) > _MAX_FILE_BYTES:
        return "", f"{name}: larger than 25 MB, left out."
    text, err = await asyncio.to_thread(_read_one, raw, Path(uf.filename or "").suffix)
    return (text, "") if (text or "").strip() else ("", _no_text(name, err))


def _library_text(path: Path, name: str) -> tuple[str, str]:
    """A document in one of the person's agents' folders: its text, or ""
    and what to tell the person."""
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return "", f"{name}: larger than 25 MB, left out."
        text, err = file_library.read_text(path, path.suffix.lower())
    except Exception as exc:
        text, err = "", str(exc)
    return (text, "") if (text or "").strip() else ("", _no_text(name, err))


async def read_documents(files: list[UploadFile], notes: list | None = None,
                         library: list | None = None, user: dict | None = None,
                         ) -> tuple[list[dict], list[str]]:
    """Each document's text (``name``, ``text``, ``chars``, ``truncated``,
    the person's ``description`` and ``usage``, and where it came from) and
    what the person should know about the ones that didn't come through —
    said without the model. ``notes`` go with ``files`` in order;
    ``library`` is ``[{id, description, usage}]`` of File library files,
    read only from ``user``'s own agents' folders."""
    uploads = [f for f in files or [] if f and f.filename]
    notes = notes if isinstance(notes, list) else []
    picks = [p for p in library or [] if isinstance(p, dict) and p.get("id")]
    sources: list[tuple[str, Any, dict]] = [
        ("upload", uf, note(notes[i] if i < len(notes) else None)) for i, uf in enumerate(uploads)]
    sources += [("library", p, note(p)) for p in picks]

    docs: list[dict] = []
    warnings: list[str] = []
    taken: set[str] = set()
    total = 0
    for origin, src, meta in sources[:_MAX_FILES + 1]:
        if len(docs) >= _MAX_FILES:
            warnings.append(f"Only the first {_MAX_FILES} documents were read.")
            break
        extra: dict = {}
        if origin == "upload":
            name = _display_name(src.filename, taken)
            text, said = await _upload_text(src, name)
        else:
            try:
                row, agents = file_library._owned_row(str(src["id"]), user or {})
                path = file_library._resolve_path(row)
            except HTTPException:
                row, path = None, None
            if row is None or path is None:
                warnings.append("A file picked from your library is no longer there, so it was left out.")
                continue
            name = _display_name(row["filename"], taken)
            extra = {"agent": agents.get(row["agent_slug"], {}).get("name") or row["agent_slug"]}
            text, said = await asyncio.to_thread(_library_text, path, name)
        if said:
            warnings.append(said)
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
        docs.append({"name": name, "text": text, "chars": len(text), "truncated": truncated,
                     "origin": origin, **meta, **extra})
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
        meta["docs"].append({"name": d["name"], "file": file, "chars": d["chars"],
                             **note(d)})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return preset_id


def load_docs(owner_id: str, preset_id: str) -> list[dict]:
    """A waiting preset's documents (``name``, ``text``, the person's
    ``description`` and ``usage``); empty when it is gone, expired, or
    another person's."""
    folder = _preset_dir(owner_id, preset_id)
    if folder is None or not (folder / "meta.json").is_file():
        return []
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if time.time() - float(meta.get("created") or 0) > _TTL:
            return []
        return [{"name": d["name"], "text": (folder / "docs" / d["file"]).read_text(encoding="utf-8"),
                 **note(d)}
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
        parts.append("## The owner's documents\n" + "\n\n".join(_doc_part(d) for d in docs))
    return "\n\n".join(parts)


def _doc_part(d: dict) -> str:
    head = [f"### {d['name']}"]
    if d.get("description"):
        head.append(f"The owner says: {d['description']}")
    hint = USAGES.get(d.get("usage") or "auto", ("", ""))[1]
    if hint:
        head.append(f"Use it as: {hint}")
    return "\n".join(head) + "\n" + d["text"]


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


def _json_list(raw: str) -> list:
    try:
        got = json.loads(raw or "[]")
    except ValueError:
        raise HTTPException(400, "The document notes couldn't be read.")
    return got if isinstance(got, list) else []


def _own_process(slug: str, owner_id: str) -> dict:
    """The caller's own process agent's registry entry — 404 for anyone else's."""
    from captain_claw.flight_deck.server import _verify_process_owner
    return _verify_process_owner(str(slug or ""), owner_id)


@router.post("/prepare")
async def prepare(
    archetype: str = Form(""),
    agent: str = Form(""),
    instructions: str = Form(""),
    notes: str = Form(""),
    library: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    user: dict = Depends(get_managed_user),
) -> dict:
    """Read the person's instructions and documents into a preset for review.
    ``agent``: the agent being set up (its archetype frames the read);
    ``notes``: JSON ``[{description, usage}]``, one per uploaded file in
    order; ``library``: JSON ``[{id, description, usage}]`` of File library
    files. Persists only the documents' text (under the returned id), never
    anything in an agent."""
    owner_id = str(user.get("id") or "") if _auth_enabled() else ""
    instructions = (instructions or "").strip()[:_MAX_INSTRUCTIONS]
    file_notes, picks = _json_list(notes), _json_list(library)
    if not instructions and not picks and not [f for f in files or [] if f and f.filename]:
        raise HTTPException(400, "Add some instructions or at least one document first.")
    if agent:
        archetype = archetype or str(_own_process(agent, owner_id).get("archetype") or "")
    if await _quota_spent(owner_id):
        raise HTTPException(429, "Today's allowance is used up, so the documents can't be read now. "
                                 "Create the agent without them, or try again tomorrow.")
    try:
        await asyncio.to_thread(prune)
    except Exception:
        log.debug("agent preset: prune failed", exc_info=True)

    docs, warnings = await read_documents(files, file_notes, picks, user)
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

    # Every document the person gave gets a knowledge entry — summarised by
    # the model, else by the person's own words; one the model invented a
    # name for doesn't.
    known = {k["doc"]: k for k in preset["knowledge"]}
    preset["knowledge"] = [dict(known.get(d["name"]) or {"doc": d["name"], "title": Path(d["name"]).stem[:120],
                                                         "summary": ""}) for d in docs]
    for k, d in zip(preset["knowledge"], docs):
        k["summary"] = k["summary"] or d["description"]
    preset["warnings"] = (warnings + preset["warnings"])[:20]
    preset_id = await asyncio.to_thread(save_docs, owner_id, docs, archetype)
    return {
        "id": preset_id,
        "preset": preset,
        "docs": [{"name": d["name"], "chars": d["chars"], "truncated": d["truncated"],
                  "origin": d["origin"], "description": d["description"], "usage": d["usage"],
                  **({"agent": d["agent"]} if d.get("agent") else {})} for d in docs],
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
    """The kept documents as Markdown, with an index, the glossary and the
    full rules; the documents written (``file``, ``title``)."""
    folder.mkdir(parents=True, exist_ok=True)
    doc_of = {d["name"]: d for d in docs}
    taken: set[str] = set()
    written: list[dict] = []
    for k in preset["knowledge"]:
        doc = doc_of.get(k["doc"])
        if doc is None:
            continue
        file = f"{_file_stem(k['doc'], taken)}.md"
        title = k["title"] or Path(k["doc"]).stem
        use = USAGES.get(doc.get("usage") or "auto", ("", ""))[0]
        head = [f"# {title}", "", f"Source: {k['doc']} — given to you by your owner.", ""]
        if doc.get("description"):
            head += [f"Your owner says: {doc['description']}", ""]
        if use:
            head += [f"Use it as: {use.lower()}.", ""]
        if k["summary"] and k["summary"] != doc.get("description"):
            head += [k["summary"], ""]
        (folder / file).write_text("\n".join(head) + "\n---\n\n" + doc["text"].strip() + "\n", encoding="utf-8")
        written.append({"file": file, "title": title, "summary": k["summary"], "use": use})
    if written:
        (folder / "INDEX.md").write_text("\n".join([
            "# Knowledge", "",
            "Documents your owner gave you. Read the one that fits before you answer from memory.", "",
            *[f"- [{w['title']}]({w['file']})" + (f" — {w['summary']}" if w["summary"] else "")
              + (f" ({w['use'].lower()})" if w["use"] else "")
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
        "running": srv._process_is_alive(slug),
    }
