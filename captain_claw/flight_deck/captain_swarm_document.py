"""Captain Swarm — the best document for one brief: writing and research.

A brief — "a two-page brief on the EU AI Act for a ten-person design studio",
"research: which open-source vector databases run on a Raspberry Pi, with
sources" — tried several ways. Each attempt is a Bat run in rehearsal: its
workers search, read and write, but nothing outside changes. An attempt ends
with its document in its run's shared folder (``vfs:bat-<id8>/document.md``).

Nobody can measure "good" directly, so a document is scored in parts, each
Captain's own:

- **The checklist.** Made once from the brief when the run starts, with the
  owner's own must-haves first, and pinned (:func:`seal`). A judge on the
  panel's model — never the one the workers run on — grades every attempt
  against it item by item (full, partial, none), twice, for its noise. The
  score is the share covered.
- **The held-out checklist.** Made at the same time from another angle — what
  a demanding expert reader would check — on the panel's model. Attempts and
  the planner never see it, and its items' grades aren't shown to anyone
  planning. A gain must hold on it too.
- **Sources** (when the brief wants them, :func:`check_sources`). Captain
  opens every cited link itself — public addresses only, a real browser when
  a plain fetch can't see the page — and checks that it loads and is about
  what cites it. An attempt whose sources don't hold is out, however well it
  reads. It doesn't check that every claim is right: the reviewer reads for
  made-up facts before an attempt can become the best.

Model judges are not the owner: the page says so and shows the documents. The
best reaches the owner's files only on their click (``POST /runs/{id}/save-document``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import urllib.parse
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import captain_swarm_browser as browser
from captain_claw.flight_deck import captain_swarm_outcome as outcome
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck.captain_swarm_learning import shown_verdict
from captain_claw.logging import get_logger

log = get_logger(__name__)

GRADES = {"full": 1.0, "partial": 0.5, "none": 0.0}
# A researched brief used 7.5–8.5M tokens per attempt on Haiku workers before a
# cap cut it short; twice that leaves room to finish and check sources.
ATTEMPT_TOKENS = 15_000_000
BUDGET = {"tokens": 100_000_000, "hours": 4.0, "max_experiments": 6}
_JUDGE_CHARS = 60_000
_KEEP_CHARS = 200_000
_MIN_WORDS = 80
_MAX_SOURCES = 12
_SOURCES_OK = 0.7
_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]]+")
_STOP = frozenset("about above after again their there these those which while where would could should other "
                  "being because between through during before under within without".split())


def _store():
    return outcome._store()


# ── the objective ────────────────────────────────────────────────────


def document_template(*, task: str, criteria: list[str] | None = None, sources: bool = True, words: int = 0,
                      audience: str = "") -> dict[str, Any]:
    task = " ".join(task.split())
    return {
        "kind": "document",
        "family": "research" if sources else "writing",
        "title": f"Best document: {task[:90]}",
        "goal": task,
        "subject": {"type": "document", "task": task, "criteria": list(criteria or []), "sources": bool(sources),
                    "words": int(words or 0), "audience": audience},
        "metric": {"kind": "rubric", "read": "checklist_covered", "direction": "max", "repeat": 2},
        "guardrails": [],
        "noise": {"min_gain": 0.05},
        "lane": {"kind": "bat", "per_round": 3, "parallel": 3, "attempt_tokens": ATTEMPT_TOKENS},
        "budget": dict(BUDGET),
        "stop": {"target": None, "stall_rounds": 2, "remeasure_every": 99},
    }


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck.captain_swarm_learning import family_slug
    from captain_claw.flight_deck.captain_swarm_objective import (
        ObjectiveError,
        _num,
        attempt_tokens,
        budget_of,
        lanes,
    )
    subj = raw.get("subject") or {}
    task = " ".join(str(subj.get("task") or raw.get("goal") or "").split())
    if len(task) < 15:
        raise ObjectiveError("Say what the document is for, in a sentence or two.")
    criteria = [" ".join(str(c).split())[:240] for c in subj.get("criteria") or [] if str(c).strip()][:12]
    sources = bool(subj.get("sources", True))
    lane = raw.get("lane") or {}
    stop = raw.get("stop") or {}
    parallel, per_round = lanes(lane, default=1, most=outcome.bat_slots(), per_round_most=6)
    return {
        "kind": "document",
        "family": family_slug(raw.get("family")) or ("research" if sources else "writing"),
        "title": str(raw.get("title") or "").strip()[:200] or f"Best document: {task[:90]}",
        "goal": task[:4000],
        "subject": {"type": "document", "task": task[:4000], "criteria": criteria, "sources": sources,
                    "words": int(_num(subj.get("words") or 0, "The length", 0, 20000)),
                    "audience": " ".join(str(subj.get("audience") or "").split())[:200]},
        "metric": {"kind": "rubric", "read": "checklist_covered", "direction": "max",
                   "repeat": int(_num((raw.get("metric") or {}).get("repeat", 2), "Judgings per document", 1, 3))},
        "guardrails": [],
        "noise": {"min_gain": _num((raw.get("noise") or {}).get("min_gain", 0.05), "The smallest gain that counts", 0, 1)},
        "lane": {"kind": "bat", "per_round": per_round, "parallel": parallel,
                 "attempt_tokens": attempt_tokens(lane, ATTEMPT_TOKENS)},
        "budget": budget_of(raw.get("budget"), BUDGET, hours_most=48, experiments_most=30),
        "stop": {"target": None, "stall_rounds": int(_num(stop.get("stall_rounds", 2), "Rounds without a gain", 1, 10)),
                 "remeasure_every": 99},
    }


# ── the checklists (sealed when the run starts) ──────────────────────


_HELD_PROMPT = (
    "A demanding expert reader is about to judge a document written for the brief below — someone who knows the "
    "subject and who the document is for. List the things they would check that a checklist written from the brief "
    "alone might miss: whether it is right, deep enough where it matters, honest about what it doesn't know, and "
    "useful to its reader. Each item one short, checkable sentence. Return ONLY a JSON array of 6–12 strings.\n\n"
    "Brief:\n{brief}"
)


def brief_text(subj: dict) -> str:
    lines = [subj["task"]]
    if subj.get("audience"):
        lines.append(f"Written for: {subj['audience']}.")
    if subj.get("words"):
        lines.append(f"About {subj['words']} words.")
    if subj.get("sources"):
        lines.append("Every factual claim backed by a linked source.")
    if subj.get("criteria"):
        lines.append("It must: " + "; ".join(subj["criteria"]))
    return "\n".join(lines)


def checklist_path(run_id: str) -> Path:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    return loop.sealed_dir(run_id) / "checklist.json"


def checklists(run_id: str) -> dict[str, list[str]]:
    try:
        data = json.loads(checklist_path(run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"open": [], "held": []}
    return {"open": list(data.get("open") or []), "held": list(data.get("held") or [])}


async def seal(run: dict, base: str) -> None:
    """Both checklists, made once and pinned. Idempotent: a re-adopted run keeps its own."""
    from captain_claw.flight_deck import captain_swarm_loop as loop
    from captain_claw.flight_deck import research_rubric
    path = checklist_path(run["id"])
    if path.is_file():
        return
    subj = run["objective"]["subject"]
    brief = brief_text(subj)
    made = research_rubric.parse_rubric(await loop.ask_model(
        run, "Return only what is asked, as JSON.", research_rubric.derive_rubric_prompt(brief),
        label="checklist", temperature=0.2))
    held = research_rubric.parse_rubric(await loop.ask_model(
        run, "Return only what is asked, as JSON.", _HELD_PROMPT.format(brief=brief),
        label="checklist", temperature=0.2, panel=True))
    if not made or not held:
        raise RuntimeError("the checklist couldn't be made from the brief")
    own = [f"(The owner asks) {c}" for c in subj.get("criteria") or []]
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(json.dumps({"open": (own + made)[:30], "held": held[:15]}, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o600)


# ── where a run starts, and the lane ─────────────────────────────────


async def start_point(run: dict) -> str:
    """No document yet."""
    vid = f"v_{secrets.token_hex(6)}"
    await _store().add_version(version_id=vid, run_id=run["id"], content={"text": "", "path": ""})
    return vid


def lane_brief(run: dict, exp: dict, project: str) -> str:
    subj = run["objective"]["subject"]
    lines = [
        f"Brief: {subj['task']}",
        f"Written for: {subj['audience']}" if subj.get("audience") else "",
        f"Length: about {subj['words']} words." if subj.get("words") else "",
        ("It must:\n" + "\n".join(f"- {c}" for c in subj["criteria"])) if subj.get("criteria") else "",
        "",
        f"Do it this way: {exp['hypothesis']}",
        f"Why this way: {exp['why']}" if exp.get("why") else "",
        "",
        "This is a REHEARSAL. Search, read and write as much as you need, but nothing outside changes: sending, "
        "posting, publishing or buying is recorded instead of done. That's expected; don't try to get around it.",
        ("Back every factual claim with a link to the page it comes from, right next to the claim. Captain will open "
         "every link itself: one that doesn't load, or isn't about what it backs, puts the whole document out. Cite "
         "only pages you actually read.") if subj.get("sources") else "",
        f"Write the finished document, in Markdown, to vfs:{project}/document.md: one file holding the whole "
        "document. Notes and drafts go in other files.",
        'End your answer with this JSON block: ```json\n{"document": "document.md", "summary": "<two sentences>"}\n```',
    ]
    return "\n".join(x for x in lines if x is not None).strip()


def read_document(owner: str, bat: dict) -> tuple[str, str]:
    """``(text, where)``: the attempt's document.md, else its largest Markdown or
    text file, else its Bat run's own answer if that is a document."""
    from captain_claw.flight_deck.vfs_routes import _user_root
    folder = _user_root(owner) / f"bat-{bat['id'][:8]}"
    files: list[Path] = []
    if folder.is_dir():
        files = [p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                 and not any(part.startswith(".") for part in p.relative_to(folder).parts)
                 and p.suffix.lower() in (".md", ".markdown", ".txt")]
    chosen = folder / "document.md"
    if not chosen.is_file():
        chosen = max(files, key=lambda p: p.stat().st_size, default=None)
    if chosen is not None and chosen.is_file():
        return chosen.read_text(encoding="utf-8", errors="replace")[:_KEEP_CHARS], str(chosen.relative_to(folder))
    truth = re.sub(r"```json.*?```", "", str(bat.get("truth") or ""), flags=re.DOTALL).strip()
    return (truth[:_KEEP_CHARS], "(its answer)") if len(truth.split()) >= _MIN_WORDS else ("", "")


class DocumentLane(outcome.BatLane):
    """One attempt = one Bat run in rehearsal that writes one document."""

    def task(self, run: dict, exp: dict, bat_id: str) -> str:
        return lane_brief(run, exp, f"bat-{bat_id[:8]}")

    async def poll(self, run: dict, exp: dict) -> dict:
        from captain_claw.flight_deck import bat_loop
        from captain_claw.flight_deck.bat_store import TERMINAL_STATES
        store = bat_loop.get_store()
        b = await store.get_run(exp["child_id"]) if store is not None else None
        if b is None:
            return {"state": "failed", "note": "its Bat run is gone"}
        usd, tokens = float(b.get("cumulative_usd") or 0.0), int(b.get("cumulative_tokens") or 0)
        if b["status"] not in TERMINAL_STATES:
            return {"state": "running", "usd": usd, "tokens": tokens}
        text, where = await asyncio.to_thread(read_document, run["owner_id"], b)
        if not text.strip():
            why = b.get("stopped_reason") or "no document in its folder"
            return {"state": "failed", "usd": usd, "tokens": tokens,
                    "note": f"its Bat run ended ({b['status']}) without a document: {why}"[:300]}
        vid = f"v_{secrets.token_hex(6)}"
        await _store().add_version(version_id=vid, run_id=run["id"], parent=exp["base_sha"],
                                   content={"text": text, "path": where, "bat_run": b["id"],
                                            "words": len(text.split())})
        return {"state": "done", "head_sha": vid, "commits": 1, "dirty": 0, "usd": usd, "tokens": tokens}


# ── screen: proof, enough of a document, sources that hold ───────────


def sources_of(text: str) -> list[str]:
    seen: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?*_`")
        if url not in seen:
            seen.append(url)
    return seen


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zà-ž0-9]{4,}", (text or "").lower()) if w not in _STOP}


def _named(words: set[str]) -> set[str]:
    """Names with a number in them (qwen3, phi4, rpi5, 2b): what a source and
    the claim it backs most often share when the source is data, not prose."""
    return {w for w in words if re.search(r"[a-z]", w) and re.search(r"\d", w)}


def about(page: str, url: str, cited_by: str) -> bool:
    """Is the page — its text and its own address — about what cites it? Two
    shared words, or one shared name with a number in it."""
    words = _content_words(cited_by)
    if not words:
        return True
    path = urllib.parse.urlsplit(url).path.replace("-", " ").replace("_", " ").replace("/", " ")
    shared = words & (_content_words(page) | _content_words(path))
    return len(shared) >= min(2, len(words)) or bool(_named(shared))


def citing_text(text: str, url: str) -> str:
    """The paragraph that cites *url*, without the link itself."""
    for para in re.split(r"\n\s*\n", text or ""):
        if url in para:
            return para.replace(url, " ")
    return ""


async def check_source(url: str, cited_by: str) -> dict:
    """Does the page load as a public page, and is it about what cites it?"""
    text, where, via = await browser.look(url, outcome.fetch_text)
    if not text or len(text) < 200:
        return {"url": url, "ok": False, "via": via, "why": where or "it shows next to nothing"}
    if not about(text, url, cited_by):
        return {"url": url, "ok": False, "via": via, "why": "it loads, but isn't about what cites it"}
    return {"url": url, "ok": True, "via": via, "why": ""}


async def check_sources(text: str) -> list[dict]:
    urls = sources_of(text)[:_MAX_SOURCES]
    return list(await asyncio.gather(*(check_source(u, citing_text(text, u)) for u in urls)))


async def screen(run: dict, exp: dict) -> dict:
    version = await _store().get_version(exp["head_sha"])
    content = (version or {}).get("content") or {}
    text = str(content.get("text") or "")
    subj = run["objective"]["subject"]
    words = len(text.split())
    out = {"changed": [content.get("path") or ""], "diff_stat": f"{words} words", "hits": []}
    if words < _MIN_WORDS:
        return {**out, "ok": False, "reason": "it wrote almost nothing"}
    try:
        found = await rehearsal.leaks(outcome.worker_prefix(exp["child_id"]), since=float(exp["created_at"]) - 5)
    except Exception as exc:  # noqa: BLE001 — no proof is no pass
        return {**out, "ok": False, "reason": f"couldn't check the tool-call register: {exc}"}
    if found:
        what = "; ".join(f"{f['what']} by {f['agent']}" for f in found[:4])
        return {**out, "ok": False, "hits": found,
                "reason": f"its workers did something outside after all ({what}) — the guard has a hole to fix"}
    if subj.get("sources"):
        checks = await check_sources(text)
        await _store().add_version(version_id=f"{exp['head_sha']}_s", run_id=run["id"], parent=exp["head_sha"],
                                   content={"sources": checks})
        held = sum(1 for c in checks if c["ok"])
        out["diff_stat"] = f"{words} words, {held} of {len(checks)} sources hold"
        if not checks:
            return {**out, "ok": False, "reason": "it cites no sources, and the brief asks for them"}
        if held / len(checks) < _SOURCES_OK:
            bad = "; ".join(f"{c['url'][:80]} ({c['why']})" for c in checks if not c["ok"])[:500]
            return {**out, "ok": False,
                    "reason": f"its sources don't hold: {len(checks) - held} of {len(checks)} failed — {bad}"}
    return {**out, "ok": True, "reason": ""}


# ── measure: the checklists, graded by a judge on the panel's model ──


_JUDGE_SYSTEM = (
    "You grade a document against a checklist, item by item. For each numbered item say how fully the document "
    "itself does it: full, partial or none. Grade only what is on the page, not what it promises or what its "
    "headings claim; a heading with nothing under it is none. Return ONLY JSON: "
    '{"grades": [{"item": <number>, "grade": "full|partial|none"}]}'
)


def parse_grades(text: str, n: int) -> list[float] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    rows = (data or {}).get("grades") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    got = [0.0] * n
    seen = 0
    for r in rows:
        try:
            i = int(r.get("item")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        g = str(r.get("grade") or "").strip().lower()
        if 0 <= i < n and g in GRADES:
            got[i] = GRADES[g]
            seen += 1
    return got if seen >= max(1, n // 2) else None


async def grade(run: dict, items: list[str], text: str) -> list[float]:
    """One judge's grades of *text* against *items*; asked again once if unreadable."""
    from captain_claw.flight_deck import captain_swarm_loop as loop
    subj = run["objective"]["subject"]
    user = (f"The brief:\n{brief_text(subj)}\n\nThe checklist:\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
            + f"\n\nThe document:\n{text[:_JUDGE_CHARS]}")
    for _ in range(2):
        got = parse_grades(await loop.ask_model(run, _JUDGE_SYSTEM, user, label="judge", temperature=0.3, panel=True),
                           len(items))
        if got is not None:
            return got
    raise RuntimeError("the judge's grades couldn't be read")


async def measure(run: dict, version_id: str, which: str) -> dict:
    """The share of the checklist the document covers (``dev``: the open one;
    otherwise the held-out one), judged ``repeat`` times."""
    from captain_claw.flight_deck.captain_swarm_score import stats
    version = await _store().get_version(version_id)
    text = str(((version or {}).get("content") or {}).get("text") or "")
    items = checklists(run["id"])["open" if which == "dev" else "held"]
    if not items:
        return {"value": None, "error": "the run has no checklist"}
    if not text.strip():
        return {"value": 0.0, "stats": {"n": 1, "mean": 0.0, "stdev": 0.0, "low": 0.0, "high": 0.0},
                "guard": {}, "per_case": {}, "error": ""}
    repeat = int(run["objective"]["metric"].get("repeat") or 2)
    try:
        runs = await asyncio.gather(*(grade(run, items, text) for _ in range(repeat)))
    except Exception as exc:  # noqa: BLE001
        return {"value": None, "error": str(exc)[:300]}
    s = stats(sum(r) / len(r) for r in runs)
    # Only the open checklist's grades are kept per item: the held-out ones stay sealed.
    per_case = {f"c{i + 1}": round(sum(r[i] for r in runs) / len(runs), 3) for i in range(len(items))} \
        if which == "dev" else {}
    return {"value": s.mean, "stats": s.to_dict(), "guard": {}, "per_case": per_case, "error": ""}


# ── the proposer ─────────────────────────────────────────────────────


_SYSTEM = (
    "You plan attempts at one document for Captain Swarm: a piece of writing or research for a brief. Each attempt "
    "is one way of going about it — the angle, the structure, which sources to read, how deep to go where — tried "
    "by an assistant with web search and a browser. A judge then grades every document against a checklist made "
    "from the brief, a second checklist nobody sees confirms a gain, and Captain opens every source a document "
    "cites. Propose {k} DIFFERENT ways, not rewordings, aimed at what the grades say the best document still "
    "misses. Don't repeat a way the ledger already tried unless you say what is different now. Return ONLY a JSON "
    'array of objects: {{"hypothesis": "<the way, concretely>", "strategy_tag": "<2-4 word kebab-case tag>", '
    '"why": "<what it should do better>"}}. No prose.'
)


def brief(run: dict, ledger: list[dict]) -> str:
    subj = run["objective"]["subject"]
    champ = run.get("champion") or {}
    dev = champ.get("dev") or {}
    items = checklists(run["id"])["open"]
    lines = [f"Brief:\n{brief_text(subj)}"]
    if dev.get("stats") and champ.get("exp"):
        lines.append(f"\nThe best document so far covers {dev['stats']['mean']:.2f} of its checklist.")
    per = dev.get("per_case") or {}
    weak = [(i, items[i], per.get(f"c{i + 1}", 0.0)) for i in range(len(items))
            if per.get(f"c{i + 1}", 0.0) < 1.0] if champ.get("exp") else []
    if weak:
        lines.append("What it still misses or only partly does (grade):")
        lines += [f"- {item} ({g:.1f})" for _, item, g in sorted(weak, key=lambda w: w[2])[:15]]
    elif items and not champ.get("exp"):
        lines.append("Nothing written yet. The checklist it will be graded on:")
        lines += [f"- {item}" for item in items[:25]]
    if ledger:
        lines.append("\nTried so far:")
        for e in ledger[-10:]:
            lines.append(f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:200]}"
                         + (f" → {shown_verdict(e)[:160]}" if e.get("verdict_reason") else ""))
    lines.append(run.get("learned") or "")
    return "\n".join(lines)


async def propose(run: dict, ledger: list[dict], k: int) -> list[dict]:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    text = await loop.ask_ideas(run, _SYSTEM.format(k=k), brief(run, ledger))
    return loop.diverse(loop.parse_proposals(text, k), ledger)


async def review_material(run: dict, exp: dict) -> str:
    version = await _store().get_version(exp["head_sha"])
    text = str(((version or {}).get("content") or {}).get("text") or "")
    if not text.strip():
        raise RuntimeError("it has no document to review")
    srcs = await _store().get_version(f"{exp['head_sha']}_s")
    checks = ((srcs or {}).get("content") or {}).get("sources") or []
    lines = [f"The brief:\n{brief_text(run['objective']['subject'])}"]
    if checks:
        lines.append(f"Captain opened its sources: {sum(1 for c in checks if c['ok'])} of {len(checks)} load and are "
                     "about what cites them.")
    lines.append(f"\nThe document:\n{text[:16000]}" + ("\n… (cut here)" if len(text) > 16000 else ""))
    return "\n".join(lines)


# ── delivery ─────────────────────────────────────────────────────────


async def best_of(run: dict) -> dict:
    """The best document so far, with its source checks ({} before any keep)."""
    champ = run.get("champion") or {}
    if not champ.get("exp"):
        return {}
    version = await _store().get_version(champ["sha"])
    content = (version or {}).get("content") or {}
    srcs = await _store().get_version(f"{champ['sha']}_s")
    return {"text": content.get("text") or "", "path": content.get("path") or "", "bat_run": content.get("bat_run") or "",
            "words": content.get("words") or 0, "sources": ((srcs or {}).get("content") or {}).get("sources") or []}


def save_to_files(owner: str, run: dict, text: str) -> str:
    """The document, written into the owner's own files (vfs:captain-swarm/…)."""
    from captain_claw.flight_deck.vfs_routes import _resolve
    name = re.sub(r"[^a-z0-9]+", "-", run["title"].lower().removeprefix("best document:")).strip("-")[:60] or "document"
    rel = f"{run['id'].removeprefix('cs_')[:8]}-{name}.md"
    target = _resolve(owner, "captain-swarm", rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return f"vfs:captain-swarm/{rel}"
