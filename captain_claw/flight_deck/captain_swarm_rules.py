"""Captain Swarm — real work, first kind: an agent's triage rules, improved against what the owner did.

docs/captain-swarm-plan.md §15, Phase 3a. What varies is a short set of rules
in the owner's voice ("Mail that asks me to choose or confirm something needs
my reply"). Each version is tried on the agent's exam lane — it can only read,
keeps nothing, can't send or change anything — over a practice set frozen from
the owner's own mail (Phase 1), and scored by agreement with what the owner
actually did. Nothing here touches the world: the lane is an edit of a text,
and the measurement is an exam.

A version is tried by putting its rules in front of each question. The agent
itself isn't changed while a run goes on, so an experiment never alters its
real conversations. The best version reaches the agent only when the owner
says so, as lessons (agent_onboarding, ``taught_by: swarm``).

What this kind brings to the loop (:mod:`captain_swarm_loop`): where a run
starts (:func:`start_point`), the seal (:func:`seal`), the edit "lane"
(:class:`RulesLane`), the screen (:func:`screen`), the measurement
(:func:`measure`) and the proposer (:func:`propose`).
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import captain_swarm_practice as practice
from captain_claw.flight_deck.captain_swarm_learning import shown_verdict
from captain_claw.flight_deck.captain_swarm_screen import EVAL_MARKERS
from captain_claw.logging import get_logger

log = get_logger(__name__)

RULE_MAX_CHARS = 300
DEFAULT_MAX_RULES = 12
BUDGET = {"tokens": 60_000_000, "hours": 8.0, "max_experiments": 16}
_MIN_STRING = 10
# A run with more unanswered questions than this measures the agent being away, not its rules.
_MAX_UNANSWERED = 0.2
_WATCH_S = 15.0       # how often a measurement looks for a stop or an empty budget


def _store():
    from captain_claw.flight_deck import captain_swarm_loop as loop
    store = loop.get_store()
    if store is None:
        raise RuntimeError("Captain Swarm's store isn't open")
    return store


def triage_template(*, set_id: str, agent: str, agent_name: str = "", owner_name: str = "",
                    cases: int = 40, repeat: int = 2, holdout_cases: int = 0,
                    set_kind: str = practice.KIND_MAIL_TRIAGE) -> dict[str, Any]:
    who = owner_name or "the owner"
    name = agent_name or agent
    calendar = set_kind == practice.KIND_CALENDAR
    return {
        "kind": "practice",
        "title": (f"{name} answers {who}'s invitations the way {who} would" if calendar
                  else f"{name} triages {who}'s inbox the way {who} would"),
        "goal": (f"Decide, for each calendar invitation, whether {who} accepts it, the way {who} actually did — by "
                 "giving the assistant a few general rules about kinds of invitation." if calendar else
                 f"Decide, for each email, whether it needs {who}'s own reply, the way {who} actually decided — "
                 "by giving the assistant a few general rules about kinds of mail."),
        "subject": {"type": "rules", "agent": agent, "agent_name": name, "owner_name": owner_name,
                    "set_id": set_id, "set_kind": set_kind, "family": practice.KINDS[set_kind]["family"],
                    "start_rules": []},
        "metric": {"kind": "agreement", "read": "f1_accepts" if calendar else "f1_needs_reply", "direction": "max",
                   "cases": cases, "repeat": repeat, "holdout_cases": holdout_cases, "holdout_repeat": repeat},
        "guardrails": [{"name": "tokens per case", "read": "tokens_per_case", "max_increase": 0.3}],
        "noise": {"min_gain": 0.03},
        "lane": {"kind": "rules", "per_round": 2, "parallel": 1, "max_rules": DEFAULT_MAX_RULES},
        "budget": dict(BUDGET),
        "stop": {"target": None, "stall_rounds": 2, "remeasure_every": 4},
    }


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """A complete, checked practice objective. That the set and the agent are the
    owner's own is checked where the owner is known (the routes)."""
    from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError, _num, budget_of, lanes
    subj = raw.get("subject") or {}
    if subj.get("type") != "rules":
        raise ObjectiveError("A practice run improves a set of rules.")
    set_id, agent = str(subj.get("set_id") or ""), str(subj.get("agent") or "")
    if not practice._ID_RE.match(set_id):
        raise ObjectiveError("Choose a practice set.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", agent):
        raise ObjectiveError("Choose one of your agents.")
    max_rules = int(_num((raw.get("lane") or {}).get("max_rules", DEFAULT_MAX_RULES), "Most rules", 1, 30))
    start = [str(r).strip()[:RULE_MAX_CHARS] for r in subj.get("start_rules") or [] if str(r).strip()][:max_rules]
    m = raw.get("metric") or {}
    stop = raw.get("stop") or {}
    parallel, per_round = lanes(raw.get("lane") or {}, default=1, most=8, per_round_most=8)
    set_kind = set_kind_of(subj)
    if set_kind not in practice.KINDS:
        raise ObjectiveError("That isn't a kind of practice set Captain Swarm knows.")
    default_family = practice.KINDS[set_kind]["family"]
    from captain_claw.flight_deck.captain_swarm_learning import family_slug
    return {
        "kind": "practice",
        "family": family_slug(raw.get("family") or subj.get("family")) or default_family,
        "title": str(raw.get("title") or "").strip()[:200] or "Inbox triage",
        "goal": str(raw.get("goal") or "").strip()[:2000],
        "subject": {"type": "rules", "agent": agent, "agent_name": str(subj.get("agent_name") or agent)[:120],
                    "owner_name": str(subj.get("owner_name") or "")[:120], "set_id": set_id, "set_kind": set_kind,
                    "family": str(subj.get("family") or default_family)[:60], "start_rules": start},
        "metric": {"kind": "agreement", "read": "f1_accepts" if set_kind == practice.KIND_CALENDAR else "f1_needs_reply",
                   "direction": "max",
                   "cases": int(_num(m.get("cases", 40), "Cases per measurement", 5, 300)),
                   "repeat": int(_num(m.get("repeat", 2), "Runs per case", 1, 5)),
                   "holdout_cases": int(_num(m.get("holdout_cases", 0), "Held-out cases", 0, 300)),
                   "holdout_repeat": int(_num(m.get("holdout_repeat", m.get("repeat", 2)), "Held-out runs per case", 1, 5))},
        "guardrails": [{"name": "tokens per case", "read": "tokens_per_case",
                        "max_increase": _num(g.get("max_increase", 0.3), "A guardrail's allowed increase", 0, 10)}
                       for g in (raw.get("guardrails") or [{}])[:1]],
        "noise": {"min_gain": _num((raw.get("noise") or {}).get("min_gain", 0.03), "The smallest gain that counts", 0, 1)},
        "lane": {"kind": "rules", "per_round": per_round, "parallel": parallel, "max_rules": max_rules},
        "budget": budget_of(raw.get("budget"), BUDGET),
        "stop": {"target": None if stop.get("target") in (None, "") else _num(stop["target"], "The target", 0, 1),
                 "stall_rounds": int(_num(stop.get("stall_rounds", 2), "Rounds without a gain", 1, 20)),
                 "remeasure_every": int(_num(stop.get("remeasure_every", 4), "Re-measure the best every", 1, 50))},
    }


# ── versions ─────────────────────────────────────────────────────────


def apply_edit(rules: list[str], change: dict[str, Any], max_rules: int = DEFAULT_MAX_RULES) -> list[str]:
    """*rules* after *change*: ``remove`` names rules by their number (1-based) or
    their text, ``add`` brings new ones. Duplicates and empties are dropped and
    the set is cut to *max_rules*."""
    drop: set[int] = set()
    for item in change.get("remove") or []:
        if isinstance(item, int) or str(item).strip().isdigit():
            drop.add(int(item) - 1)
        else:
            text = str(item).strip().casefold()
            drop |= {i for i, r in enumerate(rules) if r.casefold() == text}
    kept = [r for i, r in enumerate(rules) if i not in drop]
    for r in change.get("add") or []:
        r = " ".join(str(r).split())[:RULE_MAX_CHARS]
        if r and r.casefold() not in {k.casefold() for k in kept}:
            kept.append(r)
    return kept[:max_rules]


async def start_point(run: dict) -> str:
    """The run's first version: the rules it was given to start from (often none:
    the agent as it is)."""
    vid = f"v_{secrets.token_hex(6)}"
    await _store().add_version(version_id=vid, run_id=run["id"],
                               content={"rules": list(run["objective"]["subject"].get("start_rules") or [])})
    return vid


async def rules_of(version_id: str) -> list[str]:
    v = await _store().get_version(version_id)
    return list((v or {}).get("content", {}).get("rules") or [])


class RulesLane:
    """An attempt here is an edit of a text: made at once, nothing to wait for,
    nothing to stop or clean up."""

    async def start(self, run: dict, exp: dict, brief: str) -> dict:
        base = await rules_of(exp["base_sha"])
        rules = apply_edit(base, exp.get("change") or {}, int(run["objective"]["lane"].get("max_rules") or DEFAULT_MAX_RULES))
        vid = f"v_{secrets.token_hex(6)}"
        await _store().add_version(version_id=vid, run_id=run["id"], parent=exp["base_sha"], content={"rules": rules})
        return {"child_id": vid, "branch": ""}

    async def poll(self, run: dict, exp: dict) -> dict:
        return {"state": "done", "head_sha": exp["child_id"], "commits": 1, "dirty": 0}

    async def stop(self, run: dict, exp: dict) -> None:
        return None

    async def discard(self, run: dict, exp: dict) -> None:
        return None


async def no_branch(run: dict, version_id: str) -> None:
    return None


# ── seal and screen ──────────────────────────────────────────────────


def sealed_dir(run_id: str) -> Path:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    return loop.sealed_dir(run_id)


def holdout_strings(cases: list[dict]) -> dict[str, str]:
    """What only the held-out cases contain: their senders' or organizers'
    addresses, their subjects or titles, and their ids. A rule carrying one was
    written from a case it was never shown."""
    out: dict[str, str] = {}
    for c in cases:
        where = f"holdout:{c.get('id', '?')}"
        sender = str(c.get("from") or c.get("organizer") or "")
        m = re.search(r"<([^>]+@[^>]+)>", sender)
        addr = (m.group(1) if m else sender).strip().lower()
        if "@" in addr:
            out.setdefault(addr, where)
        subject = " ".join(str(c.get("subject") or c.get("title") or "").split())
        if len(subject) >= _MIN_STRING:
            out.setdefault(subject.lower(), where)
        if c.get("id"):
            out.setdefault(str(c["id"]).lower(), where)
    return out


async def seal(run: dict, base: str) -> None:
    """The practice set is already sealed (Phase 1). What the run adds: what
    only its held-out part contains, minus what the open part shares, for the screen."""
    import asyncio
    import os
    d = sealed_dir(run["id"])
    if (d / "sealed.json").is_file():
        return
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    subj = run["objective"]["subject"]
    hold = await asyncio.to_thread(practice.load_cases, run["owner_id"], subj["set_id"], "holdout")
    opened = await asyncio.to_thread(practice.load_cases, run["owner_id"], subj["set_id"], "open")
    if not hold or not opened:
        raise RuntimeError("the practice set has no cases in one of its parts")
    shared = set(holdout_strings(opened))
    strings = {s: src for s, src in holdout_strings(hold).items() if s not in shared}
    (d / "strings.json").write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
    os.chmod(d / "strings.json", 0o600)
    (d / "sealed.json").write_text(json.dumps({"base": base, "set": subj["set_id"]}), encoding="utf-8")


async def screen(run: dict, exp: dict) -> dict:
    """In bounds and fair: a real change, few short rules, nothing that only the
    held-out mail contains, nothing about the eval itself."""
    new, old = await rules_of(exp["head_sha"]), await rules_of(exp["base_sha"])
    changed = [f"+ {r}" for r in new if r not in old] + [f"- {r}" for r in old if r not in new]
    out = {"changed": changed, "diff_stat": f"{len(new)} rule(s): {sum(1 for c in changed if c[0] == '+')} added, "
                                            f"{sum(1 for c in changed if c[0] == '-')} removed", "hits": []}
    if not changed:
        return {**out, "ok": False, "reason": "it changed nothing"}
    max_rules = int(run["objective"]["lane"].get("max_rules") or DEFAULT_MAX_RULES)
    if len(new) > max_rules or any(len(r) > RULE_MAX_CHARS for r in new):
        return {**out, "ok": False, "reason": f"out of bounds: at most {max_rules} rules of {RULE_MAX_CHARS} characters"}
    try:
        strings = json.loads((sealed_dir(run["id"]) / "strings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        strings = {}
    text = "\n".join(new).lower()
    hits = [{"string": s, "source": src} for s, src in strings.items() if s in text]
    hits += [{"string": m, "source": "eval"} for m in EVAL_MARKERS if m.lower() in text]
    if hits:
        peeked = any(h["source"].startswith("holdout") for h in hits)
        what = "; ".join(repr(h["string"]) for h in hits[:4])
        return {**out, "ok": False, "hits": hits,
                "reason": ("it names held-out mail it was never shown: " if peeked else "it points at the eval: ") + what}
    return {**out, "ok": True, "reason": ""}


# ── measuring ────────────────────────────────────────────────────────


def with_rules(question: str, rules: list[str], owner_name: str = "", set_kind: str = practice.KIND_MAIL_TRIAGE) -> str:
    """The question as one version asks it: its rules first. No rules is the
    agent as it is."""
    if not rules:
        return question
    who = owner_name or "your owner"
    what = "answers invitations" if set_kind == practice.KIND_CALENDAR else "triages mail"
    head = f"How {who} {what} — apply these rules:\n" + "\n".join(f"- {r}" for r in rules)
    return f"{head}\n\n{question}"


def agent_model(slug: str) -> str:
    """The model an agent runs on, as Flight Deck spawned it ("" when unknown)."""
    try:
        from captain_claw.flight_deck import server as srv
        return str((srv._load_process_registry().get(slug) or {}).get("model") or "")
    except Exception:
        return ""


def price_answer(model: str, got: dict) -> tuple[float | None, int]:
    """``(usd, tokens)`` for one answer: its usage on *model* (cache reads and
    writes at their own rates); a bare token count is priced as input. ``usd``
    is None when the model has no known price."""
    from captain_claw.flight_deck import pricing
    usage = got.get("usage") if isinstance(got.get("usage"), dict) else {}
    tokens = int(got.get("tokens") or 0)
    if not usage and tokens:
        usage = {"prompt_tokens": tokens}
    if not usage:
        return 0.0, 0
    cost = pricing.cost_from_usage(model, usage)
    return (float(cost["usd"]) if cost.get("priced") else None), tokens


async def exam_asker(owner_id: str, slug: str, name: str, label: str):
    """An ask over the agent's exam lane — read-only, keeping nothing."""
    from captain_claw.flight_deck import agent_exam as ex
    from captain_claw.flight_deck import agent_preset as ap
    job = ex.ExamJob(id=f"swarm{label}", agent=slug, agent_name=name, owner_id=owner_id if ap._auth_enabled() else "")
    if not await ex.can_take_exams(job):
        raise RuntimeError(f"Restart {name} first: it was started before it could answer practice cases.")
    counter = 0

    async def ask(question: str) -> dict:
        nonlocal counter
        counter += 1
        try:
            return await ex.ask(job, {"id": f"c{counter}", "question": question})
        except ex._ExamError as exc:
            return {"error": str(exc)}
    return ask


def measure_queue(run: dict) -> tuple[str, int]:
    """An agent answers one question at a time: its measurements queue."""
    return f"agent:{run['objective']['subject']['agent']}", 1


async def review_material(run: dict, exp: dict) -> str:
    obj = run["objective"]
    old, new = await rules_of(exp["base_sha"]), await rules_of(exp["head_sha"])
    return "\n".join([f"Goal: {obj['title']}.", obj.get("goal") or "",
                      "Rules before:" if old else "Rules before: none (the assistant decided on its own).",
                      *(f"{i}. {r}" for i, r in enumerate(old, 1)),
                      "Rules now:", *(f"{i}. {r}" for i, r in enumerate(new, 1)),
                      f"The idea behind the change: {exp['hypothesis']}"])


async def measure(run: dict, version_id: str, which: str) -> dict:
    """The agent on the open cases (``dev``), a third of them asked once
    (``slice``, Phase 4's quick look), or the held-out ones, with this version's
    rules: F1 on "needs your reply" across runs, the guardrails, and how often
    each case was right."""
    import asyncio

    from captain_claw.flight_deck import captain_swarm_measure as cm
    obj = run["objective"]
    subj, m = obj["subject"], obj["metric"]
    rules = await rules_of(version_id)
    split = "holdout" if which == "holdout" else "open"
    limit = {"dev": m["cases"], "slice": max(5, m["cases"] // 3)}.get(which, m["holdout_cases"])
    repeat = {"dev": m["repeat"], "slice": 1}.get(which, m["holdout_repeat"])
    cases = cm.pick_cases(await asyncio.to_thread(practice.load_cases, run["owner_id"], subj["set_id"], split), limit)
    if not cases:
        return {"value": None, "error": f"the practice set's {split} part has no cases"}
    ask = await exam_asker(run["owner_id"], subj["agent"], subj.get("agent_name") or subj["agent"],
                           f"{run['id'][-6:]}{version_id[-6:]}")
    store = _store()
    model = agent_model(subj["agent"])
    spent = {"usd": 0.0, "tokens": 0, "unpriced": 0}

    async def ask_with_rules(question: str) -> dict:
        got = await ask(with_rules(question, rules, subj.get("owner_name") or "", set_kind_of(subj)))
        # Every answer is the agent's own model at work: priced on it and counted
        # against the run as it comes, so the budget can stop a measurement midway.
        usd, tokens = price_answer(model, got)
        spent["usd"] += usd or 0.0
        spent["tokens"] += tokens
        spent["unpriced"] += 1 if usd is None and tokens else 0
        await store.add_spend(run["id"], usd or 0.0, tokens)
        return got

    from captain_claw.flight_deck.captain_swarm_objective import budget_spent

    async def stopped() -> bool:
        cur = await store.get_run(run["id"])
        return bool(cur and (cur["status"] not in ("baseline", "running") or budget_spent(cur)))

    flag = {"stop": False}

    async def watch() -> None:
        while not flag["stop"]:
            await asyncio.sleep(_WATCH_S)
            if await stopped():
                flag["stop"] = True

    watcher = asyncio.create_task(watch())
    try:
        result = await cm.run_triage(cases, ask=ask_with_rules, repeat=repeat, kind=set_kind_of(subj),
                                     owner_name=subj.get("owner_name") or "", should_stop=lambda: flag["stop"])
    finally:
        flag["stop"] = True
        watcher.cancel()
    if spent["unpriced"]:
        log.info("captain swarm: answers on a model with no known price", run_id=run["id"], model=model,
                 answers=spent["unpriced"])
    if result["stopped"]:
        cur = await store.get_run(run["id"])
        if cur and budget_spent(cur):
            return {"value": None, "error": f"{budget_spent(cur)} — it ran out mid-measurement"}
        return {"value": None, "error": "the run was stopped"}
    asked = max(1, result["asked"])
    if result["pooled"]["unanswered"] / asked > _MAX_UNANSWERED:
        return {"value": None, "error": f"the agent didn't answer {result['pooled']['unanswered']} of {asked} questions"}
    per_case = {r["id"]: round(sum(1 for p in r["predicted"] if p == r["label"]) / max(1, len(r["predicted"])), 4)
                for r in result["cases"]}
    guard = {g["name"]: result["guardrails"].get(g["read"]) for g in obj.get("guardrails") or []}
    return {"value": result["value"], "stats": result["f1"], "guard": guard, "per_case": per_case,
            "pooled": result["pooled"], "tokens": spent["tokens"], "usd": round(spent["usd"], 6),
            "error": "", "seconds": result["guardrails"]["seconds_per_case"]}


# ── the proposer ─────────────────────────────────────────────────────


_TASKS = {
    practice.KIND_MAIL_TRIAGE: ("triages email. For each email it decides whether {owner} needs to reply personally",
                                "mail (who it's from, what it asks, how it's addressed), never about one particular email"),
    practice.KIND_CALENDAR: ("answers calendar invitations. For each invitation it decides whether {owner} accepts it",
                             "invitation (who organizes it, what it is, when, how long, how many are invited), never "
                             "about one particular event"),
}

_SYSTEM = (
    "You improve how {owner}'s assistant {task}; it is scored against what {owner} actually did. Its behaviour comes "
    "from a short list of rules in {owner}'s voice. Propose {k} DIFFERENT edits to the rules: different ideas, not "
    "rewordings. A rule is one short sentence about a KIND of {kind_of}. Keep the list at {max_rules} rules or fewer. Don't repeat an edit the ledger already "
    "discarded unless you say what is different now. Return ONLY a JSON array of objects: "
    '{{"hypothesis": "<the idea in one line>", "strategy_tag": "<2-4 word kebab-case tag>", '
    '"why": "<which mistakes it should fix>", "add": ["<new rule>"], "remove": [<numbers of current rules to drop>]}}. '
    "No prose."
)


def set_kind_of(subj: dict) -> str:
    return str(subj.get("set_kind") or practice.KIND_MAIL_TRIAGE)


def _case_line(c: dict) -> str:
    if "organizer" in c:  # a calendar invitation
        return (f"- [{'accepted' if c.get('label') else 'declined'}] {c.get('title')}; from {c.get('organizer')}; "
                f"{c.get('start')}, {c.get('duration_min', 0)} min, {c.get('guests', 0)} other guests"
                f"{', repeats' if c.get('recurring') else ''}. {str(c.get('description') or '')[:160]}")
    body = " ".join(str(c.get("body") or "").split())[:220]
    extra = []
    if c.get("cc"):
        extra.append("you were in Cc")
    if c.get("bulk"):
        extra.append("mailing-list/automated headers")
    return (f"- [{'replied' if c.get('label') else 'did not reply'}] From {c.get('from')}; Subject: "
            f"{c.get('subject')}{'; ' + ', '.join(extra) if extra else ''}. {body}")


async def brief(run: dict, ledger: list[dict]) -> str:
    """What the proposer is told: the rules now, the score, and the open mail
    the best version still gets wrong. Never anything held out."""
    import asyncio
    obj = run["objective"]
    champ = run.get("champion") or {}
    rules = await rules_of(champ.get("sha") or "")
    dev = champ.get("dev") or {}
    pooled = dev.get("pooled") or {}
    lines = [f"Goal: {obj['title']}."]
    lines.append("Current rules:" if rules else "Current rules: none yet (the assistant decides on its own).")
    lines += [f"{i}. {r}" for i, r in enumerate(rules, 1)]
    if dev.get("stats"):
        lines.append(f"Score: F1 {dev['stats']['mean']:.3f} ± {dev['stats']['stdev']:.3f} "
                     f"(precision {pooled.get('precision', 0):.2f}, recall {pooled.get('recall', 0):.2f}).")
    per_case = dev.get("per_case") or {}
    wrong = sorted((cid for cid, ok in per_case.items() if ok < 0.5), key=lambda cid: per_case[cid])
    if wrong:
        cases = {c["id"]: c for c in await asyncio.to_thread(practice.load_cases, run["owner_id"],
                                                             obj["subject"]["set_id"], "open")}
        lines.append("\nOpen cases it still gets wrong:")
        lines += [_case_line(cases[cid]) for cid in wrong[:12] if cid in cases]
    if ledger:
        lines.append("\nTried so far (newest last):")
        for e in ledger[-12:]:
            ch = e.get("change") or {}
            lines.append(f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:200]}"
                         + (f" add {ch.get('add')}" if ch.get("add") else "")
                         + (f" → {shown_verdict(e)[:160]}" if e.get("verdict_reason") else ""))
    lines.append(run.get("learned") or "")
    return "\n".join(lines)


def parse(text: str, k: int) -> list[dict]:
    """The edits in a model's reply, tidied; one that changes nothing is dropped."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        add = [" ".join(str(r).split())[:RULE_MAX_CHARS] for r in item.get("add") or [] if str(r).strip()][:4]
        remove = [r for r in item.get("remove") or [] if isinstance(r, (int, str)) and str(r).strip()][:6]
        if not add and not remove:
            continue
        tag = re.sub(r"[^a-z0-9-]+", "-", str(item.get("strategy_tag") or "").lower()).strip("-")[:40]
        out.append({"hypothesis": str(item.get("hypothesis") or "; ".join(add))[:600], "strategy_tag": tag,
                    "why": str(item.get("why") or "")[:600], "files": [], "change": {"add": add, "remove": remove}})
    return out[:k]


async def propose(run: dict, ledger: list[dict], k: int) -> list[dict]:
    """One model call on the owner's tiers, counted against the run."""
    from captain_claw.flight_deck import captain_swarm_loop as loop
    obj = run["objective"]
    who = obj["subject"].get("owner_name") or "the owner"
    task, kind_of = _TASKS[set_kind_of(obj["subject"])]
    system = _SYSTEM.format(owner=who, k=k, task=task.format(owner=who), kind_of=kind_of,
                            max_rules=obj["lane"].get("max_rules") or DEFAULT_MAX_RULES)
    text = await loop.ask_ideas(run, system, await brief(run, ledger))
    return loop.diverse(parse(text, k), ledger)
