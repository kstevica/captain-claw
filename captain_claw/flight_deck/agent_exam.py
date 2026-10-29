"""Onboarding exam — does an agent really know the documents it was given?

An agent's owner gives it documents (agent_preset: its knowledge/ folder,
read for rules and skills, or only kept to search). The exam checks what it
does with them, as its users would find out:

- **writing** — sections of its documents are sampled, never the whole of
  them: every document gets a share, each from the parts the setup read and
  the parts it didn't, leaning to sections that tell it what to do
  (agent_preset_select.instructive) and to a focus when one is given. The
  agent's own model writes one question per section with the answer the
  section gives: an ``ask`` ("what's our refund window?") or, from the
  sections with rules, a ``situation`` ("a customer wants a refund on day
  40 — what do you reply?"), which only goes right when the rule reaches it
  without being told to look;
- **asking** — each question goes to the agent through its real chat, a
  fresh private session each, which keeps nothing in its memory and can't
  send or change anything; what it looked up on the way is kept with its
  answer;
- **grading** — each answer is judged against its section, and a wrong one
  gets a cause from what the agent did: it didn't look, it looked in the
  wrong place, it found the section and still got it wrong, or it made
  things up.

An exam is kept per agent (``exams/`` in its folder, outside its workspace)
and can be run again — the same questions, the model it has now — so a new
document, a fact taught, or a model switch shows in the score. Each answer
the exam wanted can be taught to it: the answer as a fact from that section.

    GET    /fd/agent-exam/{slug}                 its exams, the running one, what one costs
    POST   /fd/agent-exam/{slug}                 start one (size, focus) or run one again (again)
    GET    /fd/agent-exam/{slug}/{id}            one exam, with its questions
    DELETE /fd/agent-exam/{slug}/{id}            stop it, or forget it
    POST   /fd/agent-exam/{slug}/{id}/continue   go on with one a restart stopped
    POST   /fd/agent-exam/{slug}/{id}/teach      teach it a question's answer, as a fact
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_idle, agent_lessons
from captain_claw.flight_deck import agent_knowledge as ak
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.flight_deck import agent_preset_outline as poutline
from captain_claw.flight_deck import agent_preset_select as psel
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.flight_deck.file_library import reply_json
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-exam", tags=["agent-exam"])

_QUESTIONS_PROMPT = Path(__file__).parent.parent / "instructions" / "agent_exam_questions_prompt.md"
_GRADE_PROMPT = Path(__file__).parent.parent / "instructions" / "agent_exam_grade_prompt.md"

# How many questions an exam asks: the person picks one.
SIZES = {"quick": 10, "standard": 25, "deep": 60}
DEFAULT_SIZE = "quick"
KINDS = ("ask", "situation")
VERDICTS = ("right", "partly", "wrong", "no_answer")
CAUSES = ("didnt_look", "wrong_place", "got_wrong", "made_up")
_MIN_SECTION_CHARS = 400   # a section with less has nothing to ask about
_SECTION_CHARS = 6000      # of a section, what the writer and the grader see
_WRITE_BATCH = 4           # sections one writing call takes
_GRADE_BATCH = 3           # answers one grading call takes
_SPARE = 0.2               # sections sampled over the size: the writer leaves some out
_READ_SHARE = 0.3          # a document read in part: at least this share of its questions from each side
_SITUATIONS = 0.4          # about this share are situations, from the sections with the most rules
_RULE_MIN = 2.0            # a section this instructive, at least, has a rule to apply
_NEAR = 0.4                # a section next to one picked is this much less likely to be picked too
_ANSWER_CHARS = 6000       # of the agent's reply, what is kept and graded
_LOOKUP_CHARS = 1200       # of a lookup's result, what is kept to show
_MAX_TOKENS = 4000         # a writing or grading call's answer
_ASK_TIMEOUT = 300.0       # a question unanswered this long is stopped
_DRAIN_S = 60.0            # a stopped question: how long its turn is waited out
_KEEP = 20                 # exams kept per agent
_DETAILS_MAX = 8           # details of one reply the grader lists as not in its passage
_DETAIL_NEAR = 600         # … found elsewhere when its figures, names and words are this close
_DETAIL_WORDS = 0.8        # … and at least this share of its words (half, with a figure or a name to go by)
_DETAIL_TRIES = 400        # places a detail's rarest part is looked at, per text
# Tokens one question costs, about, before the agent's own exams say: the writer's share, the
# agent's turn (its instructions, tools and what it looks up), the grader's share.
_COST = {"writing": 1600, "asking": 30000, "grading": 2500}
_POINTS = {"right": 1.0, "partly": 0.5}
_RANK = {"right": 2, "partly": 1, "wrong": 0, "no_answer": 0}   # better or worse than last time


# ── Sampling ──────────────────────────────────────────────────────────


@dataclass
class Source:
    """A document of the agent's knowledge, as the exam samples it."""
    file: str
    title: str
    text: str
    name: str = ""                                # the name a setup gave it under ("" when none did)
    read: set[str] = field(default_factory=set)   # the keys of its sections a setup read
    all_read: bool = False
    known: bool = False                           # read tracking says which sections were read


def candidates(src: Source) -> list[dict]:
    """The sections of *src* worth a question, in order: each with how much
    it tells an agent to do (``score``) and whether a setup read it (None:
    not known). A long one — a whole chapter, as a PDF's often are — is
    offered in windows of what the writer and the grader see (``at``: where
    in the section), so a question can come from any part of it, not only
    its start. Named as the outline names them (``tidy_path``)."""
    out: list[dict] = []
    for s in psel.spans(src.text):
        body = src.text[s["start"]:s["end"]].strip()
        if len(body) < _MIN_SECTION_CHARS or psel._BOILERPLATE_TITLE.search(s["title"].strip()):
            continue
        path, title = poutline.tidy_path(s["path"], s["title"])
        pieces = (psel._cuts(src.text, s["start"], s["end"], _SECTION_CHARS) if len(body) > _SECTION_CHARS
                  else [(s["start"], s["end"])])
        for a, b in pieces:
            text = src.text[a:b].strip()
            if len(text) < _MIN_SECTION_CHARS:
                continue
            out.append({"doc": src.file, "doc_title": src.title, "doc_name": src.name, "key": s["key"],
                        "title": title, "path": path, "start": a, "end": b,
                        "score": psel.instructive(text, s["title"]),
                        "read": (src.all_read or s["key"] in src.read) if src.known else None,
                        **({"at": [a - s["start"], b - s["start"]]} if len(pieces) > 1 else {})})
    return out


def shares(counts: list[int], n: int) -> list[int]:
    """*n* questions shared among documents with *counts* sections to ask
    about: by the square root of their size — a long document gets more, not
    all of them — and each one at least, while there are enough."""
    got = [0] * len(counts)
    weights = [math.sqrt(c) for c in counts]
    total = sum(weights) or 1.0
    left = min(n, sum(counts))
    for i in sorted(range(len(counts)), key=lambda j: -weights[j]):
        if left and counts[i]:
            got[i] += 1
            left -= 1
    while left:
        i = max((j for j in range(len(counts)) if got[j] < counts[j]),
                key=lambda j: n * weights[j] / total - got[j])
        got[i] += 1
        left -= 1
    return got


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for r, i in enumerate(order):
        out[i] = r / (len(values) - 1) if len(values) > 1 else 1.0
    return out


def _draw(cands: list[dict], k: int, rng: random.Random) -> list[dict]:
    """*k* of *cands* by their weight (``w``), a picked one's neighbours made
    less likely, so the questions spread over the document."""
    w = [c["w"] for c in cands]
    picked: list[int] = []
    for _ in range(min(k, len(cands))):
        total = sum(w)
        if total <= 0:
            break
        r = rng.random() * total
        i = 0
        for i, x in enumerate(w):
            r -= x
            if r <= 0 and x > 0:
                break
        picked.append(i)
        w[i] = 0.0
        for j in (i - 1, i + 1):
            if 0 <= j < len(w):
                w[j] *= _NEAR
    return [cands[i] for i in sorted(picked)]


def _pick_in(cands: list[dict], k: int, rng: random.Random) -> list[dict]:
    """*k* sections of one document: from the read and the unread parts
    both, when it was read in part, each side at least ``_READ_SHARE``."""
    read = [c for c in cands if c["read"] is True]
    unread = [c for c in cands if c["read"] is False]
    if k < 2 or not read or not unread:
        return _draw(cands, k, rng)
    least = math.ceil(k * _READ_SHARE)
    k_read = min(len(read), max(least, min(k - least, round(k * len(read) / (len(read) + len(unread))))))
    k_unread = min(len(unread), k - k_read)
    k_read = min(len(read), k - k_unread)
    return sorted(_draw(read, k_read, rng) + _draw(unread, k_unread, rng), key=lambda c: c["start"])


def sample(sources: list[Source], n: int, focus: str = "", seed: int = 0) -> list[dict]:
    """About *n* sections to ask about (a few more: the writer leaves some
    out), every document getting a share — each weighted by how much it
    tells the agent to do and, with a *focus*, how close it is to it — and
    the most instructive of them marked ``situation``."""
    rng = random.Random(seed)
    per_doc = [candidates(s) for s in sources]
    if focus.strip():
        flat = [c for cs in per_doc for c in cs]
        texts = [sources[[s.file for s in sources].index(c["doc"])].text[c["start"]:c["start"] + 800]
                 for c in flat]
        for c, f in zip(flat, psel.focus_scores(texts, focus), strict=True):
            c["focus"] = f
    for cs in per_doc:
        ranks = _ranks([c["score"] for c in cs])
        focus_ranks = _ranks([c.get("focus", 0.0) for c in cs]) if focus.strip() else None
        for i, c in enumerate(cs):
            w = 0.25 + ranks[i]
            if focus_ranks is not None:
                w = (1 - psel.FOCUS_WEIGHT) * w + psel.FOCUS_WEIGHT * (0.1 + 1.5 * focus_ranks[i])
            c["w"] = w
    want = n + max(1, math.ceil(n * _SPARE))
    picks = [p for cs, k in zip(per_doc, shares([len(cs) for cs in per_doc], want), strict=True)
             for p in _pick_in(cs, k, rng)]
    ranked = sorted(picks, key=lambda c: -c["score"])
    room = round(len(picks) * _SITUATIONS)
    for c in picks:
        c["kind"] = "ask"
    for c in ranked[:room]:
        if c["score"] >= _RULE_MIN:
            c["kind"] = "situation"
    for c in picks:
        c.pop("w", None)
        c.pop("focus", None)
    return picks


def _excerpt(text: str, limit: int = _SECTION_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit("\n", 1)[0] + "\n[…]"


def where(q: dict) -> str:
    """A question's section, as a reader finds it: document › parents › section
    (a document whose first heading is its title says it once)."""
    parts = [q["doc_title"], *q.get("path", []), q["title"]]
    return " › ".join(p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1])


def hit_place(q: dict) -> str:
    """A question's section as the knowledge tool's search names a hit's (``Doc › parents › section``)."""
    return " › ".join([q["doc_title"], *q.get("path", []), q["title"]])


# ── Writing the questions ─────────────────────────────────────────────


def writing_prompt(batch: list[dict], text_of: dict[str, str]) -> str:
    blocks = []
    for n, p in enumerate(batch, 1):
        blocks.append(f"## Section {n} — kind: {p['kind']}\nDocument: {p['doc_title']}\nWhere: {where(p)}\n\n"
                      f"{_excerpt(text_of[p['doc']][p['start']:p['end']])}")
    return "\n\n".join(blocks)


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def parse_questions(content: str, batch: list[dict]) -> list[dict]:
    """The writer's questions, each on its section; a section it left out,
    or answered with something unusable, has none."""
    try:
        raw = reply_json(content).get("questions")
    except (ValueError, json.JSONDecodeError):
        return []
    out: list[dict] = []
    taken: set[int] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("section"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(batch) or n in taken:
            continue
        question, answer = _clean(item.get("question"), 1200), _clean(item.get("answer"), 2000)
        if len(question) < 8 or not answer:
            continue
        points = [p for p in (_clean(x, 200) for x in item.get("points") or [] if isinstance(x, (str, int, float)))
                  if p][:4]
        kind = str(item.get("kind") or "").strip().lower()
        p = batch[n - 1]
        taken.add(n)
        out.append({**{k: p[k] for k in ("doc", "doc_title", "doc_name", "key", "title", "path", "read", "at")
                       if k in p},
                    "kind": kind if kind in KINDS else p["kind"],
                    "question": question, "expected": answer, "points": points or [answer[:200]]})
    return out


# ── Grading ───────────────────────────────────────────────────────────


def grading_prompt(batch: list[dict], text_of: dict[str, str | None]) -> str:
    blocks = []
    for n, q in enumerate(batch, 1):
        passage = text_of.get(q["id"]) or "(The section is no longer in the agent's knowledge.)"
        blocks.append(
            f"## Item {n}\nQuestion: {q['question']}\n\nPassage ({where(q)}):\n{_excerpt(passage)}\n\n"
            f"Expected answer: {q['expected']}\nPoints: " + "; ".join(q["points"])
            + f"\n\nReply:\n{_excerpt(q.get('answer') or '(no reply)', _ANSWER_CHARS)}")
    return "\n\n".join(blocks)


def parse_grades(content: str, batch: list[dict]) -> dict[int, dict]:
    """The grader's verdicts by item number (1-based); an item it left out has none."""
    try:
        raw = reply_json(content).get("grades")
    except (ValueError, json.JSONDecodeError):
        return {}
    out: dict[int, dict] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("item"))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict") or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not 1 <= n <= len(batch) or verdict not in VERDICTS:
            continue
        unsupported = [d for d in (_clean(x, 200) for x in item.get("unsupported") or []
                                   if isinstance(x, (str, int, float))) if d][:_DETAILS_MAX]
        out[n] = {"verdict": verdict, "made_up": item.get("made_up") is True or bool(unsupported),
                  "unsupported": unsupported,
                  "missing": [m for m in (_clean(x, 200) for x in item.get("missing") or []
                                          if isinstance(x, (str, int, float))) if m][:4],
                  "why": _clean(item.get("why"), 400)}
    return out


# ── Made up, or from elsewhere in its documents ───────────────────────

_FIGURE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[^\W\d_][\w’'-]*")
_COMMON = frozenset("""a an the and or but nor of in on at to for from by with as is are was were be been its it this
that these those their there than then per each about around roughly approximately over under more less most
least into onto out up down after before between during when while which who whom whose what how why""".split())


def _stem(word: str) -> str:
    return word[:6] if len(word) > 6 else word


def _detail_needs(detail: str) -> tuple[list[str], list[str], list[str]]:
    """What a detail must have nearby to be found: its figures, its names
    (capitalised words but the first), and its words."""
    figures = [f.strip(".,") for f in _FIGURE.findall(detail)]
    words = [w for w in _WORD.findall(detail) if len(w) > 2]
    names = [w.casefold() for w in words[1:] if w[0].isupper() and w.casefold() not in _COMMON]
    plain = [_stem(w.casefold()) for w in words if w.casefold() not in _COMMON and len(w) > 3]
    return figures, names, plain


def _has_figure(window: str, figure: str) -> bool:
    return re.search(rf"(?<![\d.,]){re.escape(figure)}(?!\d|[.,]\d)", window) is not None


def found_in(detail: str, texts: list[str]) -> bool:
    """Whether *detail* (a few words the grader quoted from a reply) is in
    one of *texts* (casefolded): its figures and names all, and most of its
    words (a reply words it its own way: half, beside a figure or a name),
    within _DETAIL_NEAR characters of each other — and, with no figure, no
    long word of it that *texts* never use."""
    figures, names, plain = _detail_needs(detail)
    if not figures and not names and not plain:
        return False
    # Without a figure to go by, a word its documents never use ("emeritus") is what was made up.
    if not figures and any(len(w) >= 6 and not any(w in t for t in texts) for w in plain):
        return False
    share = _DETAIL_WORDS if not figures and not names else _DETAIL_WORDS / 2
    anchor = (figures or sorted(names, key=len, reverse=True) or sorted(plain, key=len, reverse=True))[0]
    pattern = (re.compile(rf"(?<![\d.,]){re.escape(anchor)}(?!\d|[.,]\d)") if figures
               else re.compile(re.escape(anchor)))
    for text in texts:
        for n, m in enumerate(pattern.finditer(text)):
            if n >= _DETAIL_TRIES:
                break
            window = text[max(0, m.start() - _DETAIL_NEAR):m.end() + _DETAIL_NEAR]
            if (all(_has_figure(window, f) for f in figures) and all(x in window for x in names)
                    and sum(1 for w in plain if w in window) >= share * len(plain)):
                return True
    return False


def check_details(details: list[str], corpus: dict[str, list[str]]) -> list[dict]:
    """Each detail the grader found outside the question's passage, with
    where else it is: ``documents`` (anywhere in them), ``lessons`` (a fact
    or rule it was taught), ``looked_up`` (what its lookups returned) — or
    "" when nowhere: made up."""
    return [{"text": d, "found": next((where for where, texts in corpus.items() if found_in(d, texts)), "")}
            for d in details]


def _corpus(srcs: list[Source], lessons: list[str], q: dict) -> dict[str, list[str]]:
    return {"documents": [s.text.casefold() for s in srcs], "lessons": [x.casefold() for x in lessons],
            "looked_up": [str(c.get("result") or "").casefold() for c in q.get("lookups") or []]}


def _lesson_texts(slug: str) -> list[str]:
    """What the agent was taught (agent_lessons): its rules, when-to-ask and facts."""
    from captain_claw.flight_deck import server as srv

    try:
        record = json.loads((srv.DATA_DIR / slug / agent_lessons._RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [x["text"] for x in agent_lessons.from_record(record)] if isinstance(record, dict) else []


# What it looked up: the knowledge tool, or its files read some other way.
_LOOKUP_TOOLS = {"knowledge", "read", "read_file", "grep", "glob", "list_files", "search_files", "memory",
                 "semantic_search", "file_search"}


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def _marks(body: str) -> list[str]:
    """A few lines of a section, distinctive enough that a result holding
    one of them holds the section."""
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if len(re.sub(r"\W", "", ln)) >= 25]
    lines.sort(key=lambda ln: -len(re.findall(r"[^\W\d_]", ln)))
    return [_norm(ln)[:80] for ln in lines[:4]]


def looked(lookups: list[dict]) -> bool:
    """Whether the agent looked anything up in its knowledge before answering."""
    for c in lookups:
        name = str(c.get("tool") or "")
        args = json.dumps(c.get("args") or {}, ensure_ascii=False)
        if name == "knowledge" or (name in _LOOKUP_TOOLS and "knowledge" in args):
            return True
    return False


def reached(q: dict, lookups: list[dict], body: str | None) -> bool:
    """Whether one of the agent's lookups got to the question's section: a
    search hit placed in it, the section opened by its title, or a result
    holding a line of it."""
    place = _norm(hit_place(q))
    marks = _marks(body or "")
    for c in lookups:
        args = c.get("args") or {}
        result = _norm(str(c.get("result") or ""))
        if place and place in result:
            return True
        asked, title = _norm(str(args.get("section") or "").split("›")[-1]), _norm(q["title"])
        if (c.get("tool") == "knowledge" and args.get("action") == "section"
                and asked and (asked == title or title.endswith(asked) or asked.endswith(title))
                and _norm(str(args.get("doc") or "")) in (_norm(q["doc_title"]), _norm(Path(q["doc"]).stem))):
            return True
        if any(m and m in result for m in marks):
            return True
    return False


def settle(grade: dict, details: list[dict]) -> dict:
    """A grade with its details checked (``check_details``): made up only
    when one is found nowhere — the grader sees one passage, and a figure
    from another chapter, or a fact the agent was taught, is its owner's
    too. A right answer that makes something up is only partly right."""
    out = {**grade, "details": details}
    if details:
        out["made_up"] = any(not d["found"] for d in details)
    if out["made_up"] and out["verdict"] == "right":
        out["verdict"] = "partly"
    return out


def cause(q: dict) -> str:
    """Why a question went wrong, from what the agent did — whether it
    ``looked`` and whether that ``found`` its section ("" when it went right)."""
    if q.get("verdict") in (None, "right"):
        return ""
    if q.get("made_up"):
        return "made_up"
    if not q.get("looked"):
        return "didnt_look"
    if not q.get("found"):
        return "wrong_place"
    return "got_wrong"


def summary(questions: list[dict]) -> dict:
    """The score — right 1, partly ½ — overall, by document, kind and
    whether the section was read, and what went wrong how often."""
    graded = [q for q in questions if q.get("verdict")]

    def score(qs: list[dict]) -> dict:
        got = sum(_POINTS.get(q["verdict"], 0.0) for q in qs)
        return {"questions": len(qs), "points": got, "pct": round(100 * got / len(qs)) if qs else None,
                "right": sum(1 for q in qs if q["verdict"] == "right")}

    by_doc: dict[str, list[dict]] = {}
    for q in graded:
        by_doc.setdefault(q["doc"], []).append(q)
    return {
        **score(graded),
        "docs": [{"doc": d, "title": qs[0]["doc_title"], **score(qs)} for d, qs in by_doc.items()],
        "kinds": {k: score([q for q in graded if q["kind"] == k]) for k in KINDS},
        "read": {"read": score([q for q in graded if q.get("read") is True]),
                 "unread": score([q for q in graded if q.get("read") is False])},
        "causes": {c: sum(1 for q in graded if q.get("cause") == c) for c in CAUSES},
        "made_up": sum(1 for q in graded if q.get("made_up")),
        # Asked again: how many went better or worse than last time.
        "changed": {"better": sum(1 for q in graded if q.get("before") in _RANK
                                  and _RANK[q["verdict"]] > _RANK[q["before"]]),
                    "worse": sum(1 for q in graded if q.get("before") in _RANK
                                 and _RANK[q["verdict"]] < _RANK[q["before"]]),
                    "compared": sum(1 for q in graded if q.get("before") in _RANK)},
    }


# ── An exam ───────────────────────────────────────────────────────────


@dataclass
class ExamJob:
    """One exam of one agent: kept in its folder (``exams/<id>.json``) after
    every question, so a restart stops it where it was — ``continue`` goes on."""
    id: str
    agent: str
    owner_id: str = ""
    owner: str = ""                 # owner_key (ap._call paces a person's calls by it)
    agent_name: str = ""
    size: str = DEFAULT_SIZE
    focus: str = ""
    again: str = ""                 # the exam whose questions it asks again
    state: str = "running"          # running | done | stopped | error
    phase: str = "writing"          # writing | asking | grading
    step: int = 0
    steps: int = 0
    model: str = ""                 # who writes, grades and answers: the agent's own model
    via: str = ""                   # agent | tiers (who wrote and graded, when the agent couldn't)
    tokens: int = 0
    cpt: float = pdocs.CHARS_PER_TOKEN
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    waiting: dict | None = None
    questions: list[dict] = field(default_factory=list)
    summary: dict | None = None
    task: asyncio.Task | None = None

    _KEPT = ("id", "agent", "owner_id", "owner", "agent_name", "size", "focus", "again", "state", "phase", "step",
             "steps", "model", "via", "tokens", "started", "finished", "error", "notes", "questions", "summary")

    def view(self, questions: bool = False) -> dict:
        out = {k: getattr(self, k) for k in self._KEPT if k not in ("questions", "owner_id", "owner")}
        out["waiting"] = self.waiting
        if questions:
            out["questions"] = self.questions
        return out

    def data(self) -> dict:
        return {k: getattr(self, k) for k in self._KEPT}


_RUNNING: dict[str, ExamJob] = {}   # agent slug → its exam under way


def _folder(slug: str) -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / slug / "exams"


def save(job: ExamJob) -> None:
    folder = _folder(job.agent)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{job.id}.json.tmp"
    tmp.write_text(json.dumps(job.data(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / f"{job.id}.json")


def _load(slug: str, exam_id: str) -> ExamJob | None:
    if not re.fullmatch(r"[a-z0-9]{6,32}", exam_id or ""):
        return None
    try:
        data = json.loads((_folder(slug) / f"{exam_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    job = ExamJob(id=exam_id, agent=slug)
    for k in ExamJob._KEPT:
        if k in data and k not in ("id", "agent"):
            setattr(job, k, data[k])
    if job.state == "running" and _RUNNING.get(slug) is None:
        job.state = "stopped"   # Flight Deck restarted under it
    return job


def history(slug: str) -> list[ExamJob]:
    """The agent's exams, newest first."""
    folder = _folder(slug)
    jobs = [j for p in folder.glob("*.json") if (j := _load(slug, p.stem)) is not None] if folder.is_dir() else []
    return sorted(jobs, key=lambda j: -float(j.started or 0))


def _prune(slug: str) -> None:
    for old in history(slug)[_KEEP:]:
        (_folder(slug) / f"{old.id}.json").unlink(missing_ok=True)


async def sources(slug: str, workspace: Path) -> list[Source]:
    """The agent's documents with their text, and what a setup read of them."""
    from captain_claw.flight_deck import server as srv
    folder, tool, docs = await ak._docs(workspace)
    by_file, _ = await asyncio.to_thread(ak._entries, srv.DATA_DIR / slug, workspace)
    out: list[Source] = []
    for d in docs:
        await asyncio.to_thread(tool.load, folder, d)
        entry = by_file.get(d.file, {})
        known, keys, everything = ak._read_marks(entry)
        if d.text.strip():
            out.append(Source(file=d.file, title=d.title, text=d.text, name=str(entry.get("doc") or ""),
                              read=keys, all_read=everything, known=known))
    return out


def section_text(srcs: list[Source], q: dict) -> str | None:
    """A question's section (its window, ``at``, of a long one) as the
    agent's knowledge has it now — None when the document or the section is
    gone."""
    src = next((s for s in srcs if s.file == q["doc"]), None)
    if src is None:
        return None
    s = next((s for s in psel.spans(src.text) if s["key"] == q["key"]), None)
    if s is None:
        return None
    at = q.get("at")
    a, b = (int(at[0]), int(at[1])) if isinstance(at, list) and len(at) == 2 else (0, s["end"] - s["start"])
    return src.text[s["start"] + a:min(s["end"], s["start"] + b)].strip()


def cost(n: int, per_question: int = 0) -> int:
    """About what an exam of *n* questions costs, in tokens: as the agent's
    last exam spent a question (*per_question*), else as one usually does."""
    return n * (per_question or sum(_COST.values()))


def spent_per_question(jobs: list[ExamJob]) -> int:
    """What a question cost in the agent's last finished exam (0: none yet)."""
    done = next((j for j in jobs if j.state == "done" and j.questions and j.tokens), None)
    return round(done.tokens / len(done.questions)) if done else 0


# ── Running it ────────────────────────────────────────────────────────


async def _write(job: ExamJob, reader: dict, srcs: list[Source]) -> None:
    """The questions: sampled sections, a writing call for each few."""
    n = SIZES.get(job.size, SIZES[DEFAULT_SIZE])
    seed = int(job.id[:8], 16) if re.fullmatch(r"[0-9a-f]{8}.*", job.id) else 0
    picks = await asyncio.to_thread(sample, srcs, n, job.focus, seed)
    if not picks:
        raise _ExamError("Its documents have no sections long enough to ask about.")
    text_of = {s.file: s.text for s in srcs}
    batches = [picks[i:i + _WRITE_BATCH] for i in range(0, len(picks), _WRITE_BATCH)]
    system = _QUESTIONS_PROMPT.read_text(encoding="utf-8")
    job.phase, job.step, job.steps = "writing", 0, len(batches)
    written: list[dict] = []
    for batch in batches:
        reply = await ap._call(reader, system, writing_prompt(batch, text_of), job.owner_id, job, _MAX_TOKENS)
        got = parse_questions(reply.text, batch)
        if not got and len(reply.text) > 0:
            log.info("agent exam: a writing call gave no questions", agent=job.agent, chars=len(reply.text))
        # The same question twice (repeated text in the documents): asked once.
        seen = {_norm(q["question"]) for q in written} | {_norm(q["expected"]) for q in written}
        for q in got:
            if _norm(q["question"]) not in seen and _norm(q["expected"]) not in seen:
                written.append(q)
                seen |= {_norm(q["question"]), _norm(q["expected"])}
        job.step += 1
    if not written:
        raise _ExamError(f"{job.model or 'Its model'} couldn't write questions from its documents.")
    # The spares sampled for sections left out: drop what's over, evenly.
    rng = random.Random(len(written))
    while len(written) > n:
        written.pop(rng.randrange(len(written)))
    job.questions = [{"id": f"q{i}", **q} for i, q in enumerate(written, 1)]
    save(job)


async def _grade(job: ExamJob, reader: dict, srcs: list[Source]) -> None:
    todo = [q for q in job.questions if "answer" in q and not q.get("verdict")]
    job.phase, job.step, job.steps = "grading", 0, math.ceil(len(todo) / _GRADE_BATCH)
    system = _GRADE_PROMPT.read_text(encoding="utf-8")
    # No reply at all needs no grader: it didn't answer.
    for q in [q for q in todo if not q.get("answer")]:
        q.update(verdict="no_answer", made_up=False, missing=list(q["points"]),
                 why=q.get("error") or "It gave no answer.")
        q["cause"] = cause(q)
    todo = [q for q in todo if q.get("answer")]
    job.steps = math.ceil(len(todo) / _GRADE_BATCH)
    lessons = await asyncio.to_thread(_lesson_texts, job.agent)
    for i in range(0, len(todo), _GRADE_BATCH):
        batch = todo[i:i + _GRADE_BATCH]
        bodies = {q["id"]: section_text(srcs, q) for q in batch}
        reply = await ap._call(reader, system, grading_prompt(batch, bodies), job.owner_id, job, _MAX_TOKENS)
        grades = parse_grades(reply.text, batch)
        for n, q in enumerate(batch, 1):
            g = grades.get(n)
            if g is None:
                continue
            q.update(settle(g, await asyncio.to_thread(check_details, g.pop("unsupported"), _corpus(srcs, lessons, q))))
            q["cause"] = cause(q)
        job.step += 1
        save(job)


async def ask(job: ExamJob, q: dict) -> dict:
    """Ask the agent one question through its real chat — on its exam lane,
    in a fresh session that is deleted after (exam_turn: read-only, keeping
    nothing) — and return what it replied, what it looked up, its tokens
    and how long it took; ``error`` when it didn't answer."""
    import websockets

    from captain_claw import exam_turn
    from captain_claw.flight_deck import server as srv

    slug = job.agent
    if not await ap._agent_up({"agent": slug, "name": job.agent_name}, job):
        raise _ExamError(f"{job.agent_name or slug} isn't running. Switch it on and continue the exam.")
    entry = srv._load_process_registry().get(slug) or {}
    port, token = int(entry.get("web_port") or 0), str(entry.get("web_auth") or "")
    params = f"?lane={exam_turn.LANE}" + (f"&token={token}" if token else "")
    frame = {"type": "chat", "content": q["question"], "no_flow": True, "no_next_steps": True,
             "no_rephrase": True,
             "automation": {"kind": exam_turn.KIND, "mail_write": "deny", "fd_delivers": True,
                            "task": {"id": f"exam-{job.id}-{q['id']}", "name": "Onboarding exam",
                                     "mirror": False}}}
    started = time.monotonic()
    agent_idle.touch(slug)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws{params}", open_timeout=10, close_timeout=5,
                                      max_size=None) as ws:
            while True:  # its welcome, and the lane's replay, before the question
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if msg.get("type") in ("replay_done", "error"):
                    break
            await ws.send(json.dumps(frame))
            try:
                reply = await asyncio.wait_for(_reply(ws), timeout=_ASK_TIMEOUT)
            except TimeoutError:
                # Stopped, and waited out: the next question must not get this one's reply.
                await ws.send(json.dumps({"type": "cancel"}))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(_reply(ws), timeout=_DRAIN_S)
                return {"answer": "", "lookups": [], "seconds": round(time.monotonic() - started),
                        "error": f"No answer in {int(_ASK_TIMEOUT // 60)} minutes."}
    except (OSError, websockets.WebSocketException) as exc:
        raise _ExamError(f"{job.agent_name or slug} stopped answering ({exc}). Continue the exam when it's up.") from exc
    finally:
        agent_idle.touch(slug)
    if reply.get("type") != "chat_message":
        return {"answer": "", "lookups": [], "seconds": round(time.monotonic() - started),
                "error": str(reply.get("message") or "It ended its turn without a reply.")[:500]}
    report = reply.get("exam") if isinstance(reply.get("exam"), dict) else {}
    usage = report.get("usage") if isinstance(report.get("usage"), dict) else {}
    tokens = int(usage.get("total_tokens") or 0) or (
        int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
    lookups = [{"tool": str(c.get("tool") or ""), "args": c.get("args") if isinstance(c.get("args"), dict) else {},
                "ok": c.get("ok") is not False, "result": str(c.get("result") or "")}
               for c in report.get("lookups") or [] if isinstance(c, dict)]
    return {"answer": str(reply.get("content") or "")[:_ANSWER_CHARS], "lookups": lookups, "tokens": tokens,
            "seconds": round(time.monotonic() - started)}


async def _reply(ws: Any) -> dict:
    """The turn's end on the agent's socket: its reply, the error that ended
    it, or its "ready" when it ended with neither."""
    while True:
        msg = json.loads(await ws.recv())
        kind = msg.get("type")
        if (kind == "chat_message" and msg.get("role") == "assistant") or kind == "error" or (
                kind == "status" and msg.get("status") == "ready"):
            return msg


def _kept(lookups: list[dict]) -> list[dict]:
    """The lookups as an exam keeps them: each result's start."""
    return [{**c, "result": c["result"][:_LOOKUP_CHARS]} for c in lookups]


async def _asking(job: ExamJob, srcs: list[Source]) -> None:
    todo = [q for q in job.questions if "answer" not in q]
    job.phase, job.step, job.steps = "asking", len(job.questions) - len(todo), len(job.questions)
    for q in todo:
        got = await ask(job, q)
        job.tokens += int(got.get("tokens") or 0)   # the agent's own turn (it counts it to the owner's day)
        lookups = got.pop("lookups", [])
        # Whether it looked, and found the section, is judged on what it got back
        # in full; the exam keeps each result's start.
        q.update(got, looked=looked(lookups), found=reached(q, lookups, section_text(srcs, q)),
                 lookups=_kept(lookups))
        job.step += 1
        save(job)


async def can_take_exams(job: ExamJob) -> bool:
    """Whether the agent (woken if asleep) knows an exam's turn — one started
    before it could would answer as an ordinary automated turn and keep it."""
    from captain_claw import exam_turn

    if not await ap._agent_up({"agent": job.agent, "name": job.agent_name}, job):
        raise _ExamError(f"{job.agent_name or job.agent} isn't running. Switch it on and try again.")
    caps = await agent_lessons.capabilities(job.agent)
    return caps is not None and exam_turn.CAPABILITY in caps


async def run_exam(job: ExamJob, reader: dict, workspace: Path) -> None:
    """Write the questions (unless it asks an earlier exam's again), ask
    them, grade the answers — saving as it goes."""
    try:
        srcs = await sources(job.agent, workspace)
        if not srcs:
            raise _ExamError("It has no documents to be examined on. Give it some in Way of working.")
        if not await can_take_exams(job):
            raise _ExamError(f"Restart {job.agent_name or job.agent} first: it was started before it could take "
                             "an exam. Then continue the exam.")
        if not job.questions:
            await _write(job, reader, srcs)
        await _asking(job, srcs)
        await _grade(job, reader, srcs)
        ungraded = sum(1 for q in job.questions if not q.get("verdict"))
        if ungraded:
            job.notes.append(f"{ungraded} answer{'s' if ungraded != 1 else ''} couldn't be graded.")
        job.summary = summary(job.questions)
        job.state = "done"
    except asyncio.CancelledError:
        job.state = "stopped"
        raise
    except _ExamError as exc:
        job.state, job.error = "error", str(exc)
    except ap._BusyError:
        job.state, job.error = "error", (f"{job.model or 'Its model'} stayed busy for minutes (a rate limit, or "
                                         "overloaded). What's done is kept: continue it later.")
    except ap._UnreadableError as exc:
        job.state, job.error = "error", str(exc)
    except Exception as exc:
        log.warning("agent exam failed", agent=job.agent, exc_info=True)
        job.state, job.error = "error", f"The exam stopped: {exc}"
    finally:
        job.finished = time.time()
        job.waiting = None
        if job.questions and job.state != "done":
            job.summary = summary(job.questions)
        try:
            save(job)
            _prune(job.agent)
        except OSError:
            log.warning("agent exam: couldn't keep the exam", agent=job.agent, exc_info=True)
        if _RUNNING.get(job.agent) is job:
            _RUNNING.pop(job.agent, None)
        await _tell(job)


class _ExamError(Exception):
    """Why an exam can't go on, for the person."""


async def _tell(job: ExamJob) -> None:
    """The bell when an exam ends. Never raises."""
    if job.state not in ("done", "error"):
        return
    name = job.agent_name or job.agent
    s = job.summary or {}
    if job.state == "done":
        kind, title = "success", f"{name}'s onboarding exam: {s.get('pct')}%"
        body = (f"{s.get('right', 0)} of {s.get('questions', 0)} answers right. "
                "Open it to see what it got wrong and why.")
    else:
        kind, title, body = "error", f"{name}'s onboarding exam stopped", job.error
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.autonomy import _norm_user

        await get_db().add_notification(_norm_user(job.owner_id), kind, title[:200], body[:2000],
                                        ref_type="agent_exam", ref_id=job.agent)
    except Exception:
        log.debug("agent exam: bell notification failed", exc_info=True)


# ── Routes ────────────────────────────────────────────────────────────


def _place(slug: str, user: dict) -> tuple[str, dict, Path]:
    """The caller's own agent: its owner id, registry entry and workspace (404 for anyone else's)."""
    owner_id = ap._owner(user)
    entry, _, workspace = ak._place(slug, user)
    return owner_id, entry, workspace


async def _reader(owner_id: str, slug: str, entry: dict) -> dict:
    """Who writes and grades: the agent's own model — the agent answers, so
    it must be on (asleep is fine: it's woken)."""
    reader, _ = await ap.choose_reader(owner_id, slug)
    if not reader or not reader.get("agent"):
        raise HTTPException(409, f"{entry.get('name') or slug} is switched off. Switch it on first: "
                                 "it answers the exam itself.")
    return reader


def _exam(slug: str, exam_id: str) -> ExamJob:
    running = _RUNNING.get(slug)
    if running is not None and running.id == exam_id:
        return running
    job = _load(slug, exam_id)
    if job is None:
        raise HTTPException(404, "That exam isn't here any more.")
    return job


@router.get("/{slug}")
async def overview(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """The agent's exams (newest first, without their questions), the one
    under way, and what it would be examined on and what that costs."""
    owner_id, entry, workspace = _place(slug, user)
    srcs = await sources(slug, workspace)
    sections = sum(len(candidates(s)) for s in srcs)
    reader = ap.agent_reader(slug, entry) or {}
    running = _RUNNING.get(slug)
    jobs = await asyncio.to_thread(history, slug)
    per_question = spent_per_question(jobs)
    return {
        "agent": slug, "name": str(entry.get("name") or slug),
        "exams": [j.view() for j in jobs if not running or j.id != running.id],
        "running": running.view() if running else None,
        "docs": len(srcs), "sections": sections, "model": str(reader.get("model") or ""),
        "sizes": SIZES, "cost": {k: cost(min(n, sections) or n, per_question) for k, n in SIZES.items()},
    }


class StartBody(BaseModel):
    size: str = Field(DEFAULT_SIZE, max_length=20)
    focus: str = Field("", max_length=300)
    again: str = Field("", max_length=40)    # an earlier exam's id: its questions again


@router.post("/{slug}")
async def start(slug: str, body: StartBody, user: dict = Depends(get_managed_user)) -> dict:
    """Start an exam in the background — new questions, or an earlier
    exam's again — and return it; ``GET /{slug}/{id}`` follows it."""
    owner_id, entry, workspace = _place(slug, user)
    if slug in _RUNNING:
        raise HTTPException(409, "An exam is already under way for this agent.")
    if await ap._quota_spent(owner_id):
        raise HTTPException(429, "Today's allowance is used up. Try the exam again tomorrow.")
    reader = await _reader(owner_id, slug, entry)
    size = body.size if body.size in SIZES else DEFAULT_SIZE
    job = ExamJob(id=secrets.token_hex(8), agent=slug, owner_id=owner_id, owner=ap.owner_key(owner_id),
                  agent_name=str(entry.get("name") or slug), size=size, focus=body.focus.strip(),
                  model=str(reader.get("model") or ""), via="agent")
    if body.again:
        earlier = _exam(slug, body.again)
        if not earlier.questions:
            raise HTTPException(409, "That exam has no questions to ask again.")
        job.size, job.focus = earlier.size, earlier.focus
        job.again = earlier.again or earlier.id
        keep = ("id", "kind", "question", "expected", "points", "doc", "doc_title", "doc_name", "title", "path", "key",
                "read")
        # Each with how it went last time, for "better" or "worse" now.
        job.questions = [{**{k: q[k] for k in keep if k in q}, **({"before": q["verdict"]} if q.get("verdict") else {})}
                         for q in earlier.questions]
    _RUNNING[slug] = job
    save(job)
    job.task = asyncio.get_running_loop().create_task(run_exam(job, reader, workspace))
    return job.view()


@router.get("/{slug}/{exam_id}")
async def get_exam(slug: str, exam_id: str, user: dict = Depends(get_managed_user)) -> dict:
    _place(slug, user)
    return _exam(slug, exam_id).view(questions=True)


@router.delete("/{slug}/{exam_id}")
async def stop_or_forget(slug: str, exam_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Stop the exam under way (what's done is kept), or forget an earlier one."""
    _place(slug, user)
    job = _exam(slug, exam_id)
    if _RUNNING.get(slug) is job and job.task is not None and not job.task.done():
        job.task.cancel()
        return {"ok": True, "stopped": True}
    (_folder(slug) / f"{job.id}.json").unlink(missing_ok=True)
    return {"ok": True, "stopped": False}


@router.post("/{slug}/{exam_id}/continue")
async def resume(slug: str, exam_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Go on with an exam that stopped (a restart, a stop, the model busy):
    the questions not asked yet, then the answers not graded."""
    owner_id, entry, workspace = _place(slug, user)
    if slug in _RUNNING:
        raise HTTPException(409, "An exam is already under way for this agent.")
    job = _exam(slug, exam_id)
    if job.state == "done" or not job.questions:
        raise HTTPException(409, "There's nothing left to do in that exam. Run it again instead.")
    reader = await _reader(owner_id, slug, entry)
    if job.state == "ready":
        # Passed on with its archetype (agent_package): asked for the first time now, by its own model.
        job.started, job.model, job.via = time.time(), str(reader.get("model") or ""), "agent"
    job.state, job.error, job.finished, job.owner_id = "running", "", None, owner_id
    job.owner = ap.owner_key(owner_id)
    _RUNNING[slug] = job
    save(job)
    job.task = asyncio.get_running_loop().create_task(run_exam(job, reader, workspace))
    return job.view()


class TeachBody(BaseModel):
    question: str = Field(..., max_length=20)


@router.post("/{slug}/{exam_id}/teach")
async def teach(slug: str, exam_id: str, body: TeachBody, user: dict = Depends(get_managed_user)) -> dict:
    """Teach the agent the answer a question wanted: a lesson from that
    section (agent_lessons), which reaches it with the messages it bears on
    — from its next one, running or not."""
    from captain_claw.flight_deck import server as srv
    _, _, workspace = _place(slug, user)
    job = _exam(slug, exam_id)
    q = next((q for q in job.questions if q.get("id") == body.question), None)
    if q is None:
        raise HTTPException(404, "That question isn't in the exam.")
    agent_dir = srv.DATA_DIR / slug

    def _add() -> bool:
        # An agent set up before setups kept a record: its record from its files, so nothing of it is lost.
        record = ap.earlier_setup(agent_dir, workspace)
        record, added = agent_lessons.taught(record, "fact", q["expected"], {"doc": q["doc"], "section": q["title"]},
                                             taught_by="exam")
        if added:
            (agent_dir / ap._RECORD).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            agent_lessons.write(agent_dir, record)
        return added

    added = await asyncio.to_thread(_add)
    q["taught"] = True
    if _RUNNING.get(slug) is not job:
        save(job)
    return {"ok": True, "taught": added}
