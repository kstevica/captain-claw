"""Captain Swarm — learning, within a run and across runs.

docs/captain-swarm-plan.md §12, Phase 6.

Within a run: every attempt ends with a one-line lesson (the measured part,
from the loop), and the proposer adds its own reading of what the attempts so
far teach — one ``READING:`` line before its ideas, kept as a lesson.

On a stall: a strategy round. Three reviewers on the panel's model (the one
the workers don't run on), each through its own lens, read the ledger and
name a direction the run hasn't tried; the next rounds' ideas start from
them. Two more rounds after it, still nothing, and the run stops as stalled.

Across runs: lessons are kept per owner and kind of task — the objective's
*family* ("mail-triage", "mrav", "find-price"…). When a run ends, every
attempt that reached a verdict is written down with its strategy tag and
whether it won; readings and strategy directions are written as they come.
The next run of the same family starts with them in its proposer's brief.
Each strategy tag gets a reliability weight — the Bayesian shrink archetypes
use, fails counting double — so a strategy that keeps winning goes into the
first waves, and the proposer is told which ones haven't.

A proposer learns whether held-out cases confirmed a gain, never their
numbers or contents (:func:`shown_verdict`): that holds for this run's ledger
and for every lesson carried to the next.

Delivery: a one-task run's best result can become a playbook on one of the
owner's agents (:func:`playbook_for`, :func:`save_playbook`), only when the
owner clicks, with the text in front of them. Rules reach an agent through
Teach, code through a pull request.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from captain_claw.flight_deck.captain_swarm_store import EXP_FINAL
from captain_claw.logging import get_logger

log = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_SEED = 0.5          # what an untried strategy is worth
_ALPHA = 4.0         # how many runs' worth of belief the seed carries
_CARRY_WON, _CARRY_LOST, _CARRY_NOTES = 6, 6, 4
_PLAYBOOK_TIMEOUT_S = 15.0

# The loop's words for an attempt that beat the best but lost its wave to a better one.
RUNNER_UP = "did better in the same wave"

READING_ASK = (
    "\n\nOne exception to 'no prose': before the JSON, write ONE line that starts with READING: and says, in a "
    "sentence or two, what the attempts so far teach about this task (what works, what doesn't, and why). "
    "Leave it out when nothing has been tried yet."
)

STRATEGY_LENSES = (
    "what the attempts so far have in common — the assumption they all share that may be the wrong one",
    "what nobody has tried at all: a different source, method or framing of the task",
    "what the best result so far still gets wrong, and the most direct way to fix that",
)

_STRATEGY_SYSTEM = (
    "A Captain Swarm run has stalled: rounds of attempts at improving one measured number brought no gain. You "
    "read its ledger through one lens and name ONE new direction for the next attempts — a direction, not a "
    "small tweak of something tried. Never suggest special-casing the measurement or its cases. Return ONLY "
    'JSON: {"direction": "<the direction, concretely, in one or two sentences>", "why": "<one sentence>"}.'
)


# ── families and verdicts ────────────────────────────────────────────


def family_slug(raw: Any) -> str:
    return _SLUG_RE.sub("-", str(raw or "").lower()).strip("-")[:60]


def family_of(obj: dict) -> str:
    """The kind of task an objective is: what its lessons are kept under."""
    return (family_slug(obj.get("family")) or family_slug((obj.get("subject") or {}).get("family"))
            or family_slug(obj.get("kind")) or "code")


_HELD_RE = re.compile(r"held-out [^ ]+ against the best's [^ ;,]+?(: the gain didn't hold)?(?=$|[;,]|, but)")


def shown_verdict(exp: dict) -> str:
    """A verdict as a proposer may read it: whether the held-out cases confirmed
    a gain, never their numbers, and never what a held-out case contains."""
    v = str(exp.get("verdict_reason") or exp.get("status") or "")
    if exp.get("status") == "disqualified" and "held-out" in v:
        return v.split(":", 1)[0]
    return _HELD_RE.sub(lambda m: "the held-out cases didn't confirm it" if m.group(1)
                        else "the held-out cases confirmed it", v)


def outcome_of(exp: dict) -> str:
    """``won`` (kept, or beat the best and lost only to a better one), ``lost``,
    or "" when the lane or the run failed and the idea itself wasn't judged."""
    if exp["status"] == "kept" or (exp["status"] == "discarded" and RUNNER_UP in (exp.get("verdict_reason") or "")):
        return "won"
    if exp["status"] in ("discarded", "disqualified"):
        return "lost"
    return ""


def attempt_text(exp: dict) -> str:
    text = f"{' '.join(exp['hypothesis'].split())[:300]} → {shown_verdict(exp)[:220]}"
    # For a text subject the change itself is the lesson: the rules it added. Never
    # those of a disqualified attempt, which may carry what only held-out cases hold.
    added = (exp.get("change") or {}).get("add") or []
    if added and exp["status"] != "disqualified":
        text += " — added: " + " | ".join(" ".join(str(r).split())[:240] for r in added[:4])
    return text + (f" [{exp['lesson'][:120]}]" if exp.get("lesson") else "")


def weight(won: int, lost: int, seed: float = _SEED, alpha: float = _ALPHA) -> float:
    """A strategy's reliability: its win rate shrunk toward *seed*, every loss
    counting double (``db._reliability_weight``'s rule for archetypes)."""
    den = won + 2.0 * lost + alpha
    return max(0.05, min(0.99, (won + alpha * seed) / den if den > 0 else seed))


# ── writing lessons down ─────────────────────────────────────────────


async def record_run(store: Any, run: dict) -> int:
    """Write down what *run*'s attempts taught, once, when it ends."""
    state = run.get("state") or {}
    if state.get("learned") is not None:
        return 0
    family = family_of(run["objective"])
    n = 0
    for exp in await store.list_experiments(run["id"]):
        if exp["status"] not in EXP_FINAL:
            continue
        await store.add_lesson(owner_id=run["owner_id"], family=family, run_id=run["id"], exp_id=exp["id"],
                               kind="attempt", strategy_tag=exp.get("strategy_tag") or "", outcome=outcome_of(exp),
                               text=attempt_text(exp))
        n += 1
    fresh = await store.get_run(run["id"])
    await store.update_run(run["id"], state={**((fresh or run).get("state") or {}), "learned": n})
    return n


def parse_reading(text: str) -> str:
    m = re.search(r"^\s*\**READING\**:\**\s*(.+)$", text or "", re.MULTILINE | re.IGNORECASE)
    return " ".join(m.group(1).split())[:600] if m else ""


async def note_reading(store: Any, run: dict, text: str) -> str:
    """The proposer's reading of the run so far, kept as a lesson of this run."""
    reading = parse_reading(text)
    if reading:
        await store.add_lesson(owner_id=run["owner_id"], family=family_of(run["objective"]), run_id=run["id"],
                               kind="reading", text=reading)
        await store.append_event(run["id"], "learn", f"the planner's reading: {reading[:300]}")
    return reading


# ── reading lessons back ─────────────────────────────────────────────


async def weights(store: Any, owner: str, family: str) -> dict[str, dict]:
    """Strategy tag → ``{"won", "lost", "weight"}`` over every judged attempt of
    *owner*'s runs of *family*."""
    tally: dict[str, dict] = {}
    for r in await store.list_lessons(owner, family, kinds=("attempt",), limit=5000):
        if r["strategy_tag"] and r["outcome"] in ("won", "lost"):
            t = tally.setdefault(r["strategy_tag"], {"won": 0, "lost": 0})
            t[r["outcome"]] += 1
    return {tag: {**t, "weight": round(weight(t["won"], t["lost"]), 3)} for tag, t in tally.items()}


async def rank(store: Any, run: dict, ideas: list[dict]) -> list[dict]:
    """*ideas* with the strategies that have won most for this kind of task
    first, so they go into the first waves; the planner's order otherwise."""
    w = await weights(store, run["owner_id"], family_of(run["objective"]))
    if not w:
        return ideas
    return sorted(ideas, key=lambda i: -(w.get(i.get("strategy_tag") or "") or {}).get("weight", _SEED))


async def brief_block(store: Any, run: dict) -> str:
    """What the proposer is told beyond this run's own ledger: lessons from
    earlier runs of the same kind of task, which strategies have won there, a
    strategy round's directions, and its own readings of this run."""
    owner, family = run["owner_id"], family_of(run["objective"])
    past = await store.list_lessons(owner, family, other_than=run["id"], limit=400)
    mine = await store.list_lessons(owner, family, run_id=run["id"], kinds=("reading", "strategy"), limit=40)
    w = await weights(store, owner, family)
    won = [r for r in past if r["kind"] == "attempt" and r["outcome"] == "won"][:_CARRY_WON]
    lost = [r for r in past if r["kind"] == "attempt" and r["outcome"] == "lost"][:_CARRY_LOST]
    notes = [r for r in past if r["kind"] in ("reading", "strategy")][:_CARRY_NOTES]
    lines: list[str] = []
    if won or lost or notes:
        lines.append(f"\nFrom earlier runs on this kind of task ({family}), newest first:")
        lines += [f"- worked ({r['strategy_tag'] or '-'}): {r['text']}" for r in won]
        lines += [f"- didn't ({r['strategy_tag'] or '-'}): {r['text']}" for r in lost]
        lines += [f"- noted: {r['text']}" for r in notes]
    if w:
        ranked = sorted(w.items(), key=lambda kv: -kv[1]["weight"])[:12]
        lines.append("Strategies by how often they won on this kind of task (won/tried); reuse a tag when an idea "
                     "is the same strategy: " + ", ".join(f"{t} {v['won']}/{v['won'] + v['lost']}" for t, v in ranked))
    strategy = [r for r in mine if r["kind"] == "strategy"][:len(STRATEGY_LENSES)]
    if strategy:
        lines.append("\nThe run stalled, and a strategy round suggested these directions. Base the next ideas on "
                     "them, unless one was already tried:")
        lines += [f"- {r['text']}" for r in strategy]
    readings = [r for r in mine if r["kind"] == "reading"][:3]
    if readings:
        lines.append("\nYour own readings of this run so far (newest first):")
        lines += [f"- {r['text']}" for r in readings]
    return "\n".join(lines)


# ── the strategy round ───────────────────────────────────────────────


def ledger_text(run: dict, ledger: list[dict]) -> str:
    obj = run["objective"]
    champ = run.get("champion") or {}
    best = ((champ.get("dev") or {}).get("stats") or {}).get("mean")
    lines = [f"Goal: {obj['title']}", obj.get("goal") or "",
             f"The number: {obj['metric']['read']} ({'higher' if obj['metric']['direction'] == 'max' else 'lower'} "
             "is better)."]
    if best is not None and abs(float(best)) < 1e11:
        lines.append(f"The best so far: {float(best):.3f}.")
    lines.append("\nAttempts so far (oldest first):")
    lines += [f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:260]} → {shown_verdict(e)[:200]}"
              for e in ledger[-24:]]
    return "\n".join(x for x in lines if x is not None)


def parse_direction(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(data, dict) or not str(data.get("direction") or "").strip():
        return None
    return {"direction": " ".join(str(data["direction"]).split())[:500],
            "why": " ".join(str(data.get("why") or "").split())[:300]}


async def strategy_round(run: dict, ledger: list[dict]) -> list[dict]:
    """Three readings of a stalled run, one per lens, each naming a new direction."""
    from captain_claw.flight_deck import captain_swarm_loop as loop
    text = ledger_text(run, ledger)

    async def one(lens: str) -> dict | None:
        got = parse_direction(await loop.ask_model(run, _STRATEGY_SYSTEM, f"Your lens: {lens}\n\n{text}",
                                                   label="strategy", temperature=0.7, panel=True))
        return {**got, "lens": lens} if got else None

    results = await asyncio.gather(*(one(lens) for lens in STRATEGY_LENSES), return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


async def keep_directions(store: Any, run: dict, directions: list[dict]) -> None:
    for d in directions:
        await store.add_lesson(owner_id=run["owner_id"], family=family_of(run["objective"]), run_id=run["id"],
                               kind="strategy", text=f"{d['direction']}" + (f" (why: {d['why']})" if d.get("why") else ""))


# ── a playbook from the best result ──────────────────────────────────


def playbook_for(run: dict, ledger: list[dict]) -> dict:
    """A playbook — the do/don't patterns an agent reads for recurring work —
    from a one-task run: the ways that found the best result, and the ways that
    didn't, with why."""
    obj = run["objective"]
    subj = obj["subject"]
    if obj.get("kind") == "document":
        return _document_playbook(run, ledger)
    best = ((run.get("champion") or {}).get("dev") or {}).get("best") or {}
    rank_ = subj.get("rank") or {}
    won = [e for e in ledger if outcome_of(e) == "won"]
    lost = [e for e in ledger if outcome_of(e) == "lost"]
    do = [f"- {' '.join(e['hypothesis'].split())[:400]}" for e in won]
    if best:
        when = time.strftime("%Y-%m-%d", time.localtime(float(run.get("finished_at") or time.time())))
        do.append(f"- The best it found, checked by Captain on {when}: {best.get('title')} — {best['value']:g} "
                  f"{best.get('currency') or ''} at {best.get('url')}. Prices change: check the page again.")
    dont = [f"- {' '.join(e['hypothesis'].split())[:240]} — {shown_verdict(e)[:160]}" for e in lost]
    return {
        "name": f"Captain Swarm: {obj['title'][:90]}",
        "task_type": "web-research",
        "rating": "good",
        "do_pattern": "\n".join(do),
        "dont_pattern": "\n".join(dont),
        "trigger_description": (f"When asked to find the best {rank_.get('label') or 'result'} "
                                f"({'lowest' if rank_.get('direction') == 'min' else 'highest'} wins) for a task like: "
                                f"{subj.get('task', '')[:400]}"),
        "reasoning": (f"Learned by Captain Swarm run {run['id']}: {len(ledger)} attempts in rehearsal, every result "
                      "checked by Captain on the page itself."),
        "tags": f"captain-swarm,{family_of(obj)}",
        "source_session": run["id"],
    }


def _document_playbook(run: dict, ledger: list[dict]) -> dict:
    """How to write or research this kind of thing: the ways that made the best
    documents, and the ways that didn't."""
    obj = run["objective"]
    subj = obj["subject"]
    won = [e for e in ledger if outcome_of(e) == "won"]
    lost = [e for e in ledger if outcome_of(e) == "lost"]
    return {
        "name": f"Captain Swarm: {obj['title'][:90]}",
        "task_type": "web-research" if subj.get("sources") else "document-processing",
        "rating": "good",
        "do_pattern": "\n".join(f"- {' '.join(e['hypothesis'].split())[:400]}" for e in won),
        "dont_pattern": "\n".join(f"- {' '.join(e['hypothesis'].split())[:240]} — {shown_verdict(e)[:160]}" for e in lost),
        "trigger_description": f"When asked to write or research something like: {subj.get('task', '')[:400]}",
        "reasoning": (f"Learned by Captain Swarm run {run['id']}: {len(ledger)} attempts, each graded against a "
                      "checklist made from the brief, with every cited source opened by Captain."),
        "tags": f"captain-swarm,{family_of(obj)}",
        "source_session": run["id"],
    }


async def save_playbook(owner: str, slug: str, playbook: dict) -> dict:
    """Give *playbook* to *owner*'s agent *slug* (its own playbook store)."""
    import httpx

    from captain_claw.flight_deck import agent_preset as ap
    entry = ap._own_process(slug, owner)  # 404 for anyone else's agent
    port = int(entry.get("web_port") or 0)
    if not port:
        raise RuntimeError("that agent isn't running")
    params = {"token": entry["web_auth"]} if entry.get("web_auth") else {}
    async with httpx.AsyncClient(timeout=_PLAYBOOK_TIMEOUT_S) as client:
        r = await client.post(f"http://127.0.0.1:{port}/api/playbooks", params=params, json=playbook)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"the agent refused it: {r.text[:200]}")
    return r.json()
