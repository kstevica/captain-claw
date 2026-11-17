"""Captain Swarm's ``factory`` kind: a First Mate factory, improved on its own cases.

See docs/first-mate-plan.md §12 (Phase 6, piece 1). An improvement run is a
Captain Swarm run (Kind A, a better way of working) whose subject is one
factory's charter and step outline. Its grants, triggers, budget and "done
when" lines never vary: an improvement can't make a factory reach further.

* **A version** (``cs_versions``) holds ``{charter, steps, worker_mode}``. The
  run starts from the factory's live version.
* **An attempt** is an edit made at once (:class:`FactoryLane`): the idea's new
  charter and/or steps, a new version. Nothing runs to make it.
* **Measuring** a version runs it on the factory's cases, each as one Bat run
  in rehearsal (a Captain Swarm tag: nothing outward happens), and grades each
  result with First Mate's reviewer (``first_mate_cases.grade``, a different
  model from the workers). ``dev`` is the open cases and part of the sealed
  ones; ``holdout`` the rest of the sealed ones, fixed when the run starts
  (:func:`seal`). The value is the average grade; stats are over repeats.
* **The proposer** sees the charter, the steps, the done-when lines, the
  score, and per case only the open cases' titles, inputs, lines and grades.
  A sealed case shows as a number with no words, and held-out numbers only as
  "confirmed" or not (Captain Swarm's ``shown_verdict``).
* **The screen** refuses an edit that changes nothing, outgrows its bounds,
  carries words only a held-out case has, or talks about the eval.

Lessons are kept under the family ``factory-<id>`` (``family_slug``). The best
version reaches the factory only as a new draft the owner approves, which then
passes First Mate's check against the live version (Phase 6, piece 2).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.logging import get_logger

log = get_logger(__name__)

KIND = "factory"
BUDGET = {"tokens": 40_000_000, "hours": 12.0, "max_experiments": 8}
STEPS_MAX = 12
STEP_MAX_CHARS = 300
MIN_CASES = 3
POLL_S = 5.0
CASE_PARALLEL = 2
_MIN_STRING = 12

class _StoppedError(Exception):
    """The run stopped or ran out mid-measurement."""


CASE_NOTE = ("This is a test case. Work only from the input below, as if it had just arrived; don't look up live "
             "mail, calendars or accounts for it. Do the work as the charter asks, and finish with your result.")


def _store():
    from captain_claw.flight_deck import captain_swarm_loop as loop
    s = loop.get_store()
    if s is None:
        raise RuntimeError("Captain Swarm isn't running on this deck")
    return s


def family_of(factory_id: str) -> str:
    from captain_claw.flight_deck.captain_swarm_learning import family_slug
    return family_slug(f"factory-{factory_id}")


# ── the objective ────────────────────────────────────────────────────

def factory_template(factory: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """A raw objective for improving *factory* from its *live* version."""
    rc = live.get("runner_config") or {}
    return {"kind": KIND, "title": f"Improve {factory['title']}"[:200],
            "goal": f"A better way of doing the standing job \"{factory['title']}\": higher grades on its own cases.",
            "subject": {"type": "factory", "factory_id": factory["id"], "title": factory["title"],
                        "owner_id": factory["owner_id"], "version": live["version"], "sha": live["sha"],
                        "start": {"charter": live["charter"], "steps": list(rc.get("steps") or []),
                                  "worker_mode": rc.get("worker_mode", "plain")}},
            "metric": {}, "lane": {}, "budget": {}, "stop": {}}


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """A complete, checked factory objective. That the factory is the owner's own
    and has enough cases is checked where the owner is known (:func:`ready`)."""
    from captain_claw.flight_deck.captain_swarm_objective import (
        ObjectiveError,
        _num,
        budget_of,
        lanes,
    )
    subj = raw.get("subject") or {}
    if subj.get("type") != "factory" or not re.fullmatch(r"fac_[0-9a-f]{6,32}", str(subj.get("factory_id") or "")):
        raise ObjectiveError("An improvement run improves one of your factories.")
    start = subj.get("start") or {}
    charter = str(start.get("charter") or "").strip()
    if not charter:
        raise ObjectiveError("That factory has no charter to improve.")
    m, stop = raw.get("metric") or {}, raw.get("stop") or {}
    parallel, per_round = lanes(raw.get("lane") or {}, default=1, most=4, per_round_most=4)
    return {
        "kind": KIND,
        "family": family_of(subj["factory_id"]),
        "title": str(raw.get("title") or "").strip()[:200] or "Improve a factory",
        "goal": str(raw.get("goal") or "").strip()[:2000],
        "subject": {"type": "factory", "factory_id": subj["factory_id"],
                    "title": str(subj.get("title") or "")[:fms.TITLE_MAX],
                    "owner_id": str(subj.get("owner_id") or "")[:120],
                    "version": int(subj.get("version") or 0), "sha": str(subj.get("sha") or "")[:64],
                    "start": {"charter": charter[:fms.CHARTER_MAX], "steps": _steps(start.get("steps")),
                              "worker_mode": "archetype" if start.get("worker_mode") == "archetype" else "plain"}},
        "metric": {"kind": "grade", "direction": "max",
                   "repeat": int(_num(m.get("repeat", 1), "Runs per case", 1, 3)),
                   "holdout_repeat": int(_num(m.get("holdout_repeat", 1), "Held-out runs per case", 1, 3))},
        "guardrails": [],
        "noise": {"min_gain": _num((raw.get("noise") or {}).get("min_gain", 0.1), "The smallest gain that counts", 0, 1)},
        "lane": {"kind": "factory", "per_round": per_round, "parallel": parallel},
        "budget": budget_of(raw.get("budget"), BUDGET),
        "stop": {"target": None if stop.get("target") in (None, "") else _num(stop["target"], "The target", 0, 1),
                 "stall_rounds": int(_num(stop.get("stall_rounds", 2), "Rounds without a gain", 1, 10)),
                 "remeasure_every": int(_num(stop.get("remeasure_every", 4), "Re-measure the best every", 1, 50))},
    }


def _steps(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(s).split())[:STEP_MAX_CHARS] for s in value if str(s).strip()][:STEPS_MAX]


def split(cases: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """``(dev, holdout)`` case ids: every open case is ``dev``; the sealed ones
    alternate, holdout first, so a factory with sealed cases always holds one out."""
    sealed = sorted((c for c in cases if c.get("sealed", True)), key=lambda c: c["id"])
    dev = [c["id"] for c in cases if not c.get("sealed", True)]
    hold = [c["id"] for i, c in enumerate(sealed) if i % 2 == 0]
    dev += [c["id"] for i, c in enumerate(sealed) if i % 2 == 1]
    return dev, hold


def ready(store: fms.FirstMateStore, owner_id: str, factory_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The owner's factory and its live version, when it can be improved:
    approved, not retired, with at least :data:`MIN_CASES` cases, one held out."""
    from captain_claw.flight_deck import first_mate_cases
    from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
    f = store.get_owned(factory_id, owner_id)
    if f is None:
        raise ObjectiveError("That factory isn't one of yours.")
    live = store.live(factory_id)
    if live is None or f["state"] == "retired":
        raise ObjectiveError("Only an approved, working factory can be improved.")
    dev, hold = split(first_mate_cases.cases(store, factory_id))
    if len(dev) + len(hold) < MIN_CASES or not hold or not dev:
        raise ObjectiveError(f"An improvement run needs at least {MIN_CASES} cases with at least two sealed: "
                             "add some on the factory's Checks tab first.")
    return f, live


# ── versions and the lane ────────────────────────────────────────────

async def start_point(run: dict) -> str:
    vid = f"v_{secrets.token_hex(6)}"
    await _store().add_version(version_id=vid, run_id=run["id"], content=dict(run["objective"]["subject"]["start"]))
    return vid


async def content_of(version_id: str) -> dict[str, Any]:
    v = await _store().get_version(version_id)
    c = (v or {}).get("content") or {}
    return {"charter": str(c.get("charter") or ""), "steps": list(c.get("steps") or []),
            "worker_mode": c.get("worker_mode") or "plain"}


def apply_change(base: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """*base* with an idea's new charter and/or steps."""
    out = dict(base)
    charter = str(change.get("charter") or "").strip()
    if charter:
        out["charter"] = charter[:fms.CHARTER_MAX]
    if isinstance(change.get("steps"), list):
        out["steps"] = _steps(change["steps"])
    return out


class FactoryLane:
    """An attempt is an edit of the charter or steps: made at once, nothing to
    wait for, nothing to stop or clean up."""

    async def start(self, run: dict, exp: dict, brief: str) -> dict:
        new = apply_change(await content_of(exp["base_sha"]), exp.get("change") or {})
        vid = f"v_{secrets.token_hex(6)}"
        await _store().add_version(version_id=vid, run_id=run["id"], parent=exp["base_sha"], content=new)
        return {"child_id": vid, "branch": ""}

    async def poll(self, run: dict, exp: dict) -> dict:
        return {"state": "done", "head_sha": exp["child_id"], "commits": 1, "dirty": 0}

    async def stop(self, run: dict, exp: dict) -> None:
        return None

    async def discard(self, run: dict, exp: dict) -> None:
        return None


async def no_branch(run: dict, version_id: str) -> None:
    return None


async def capacity(run: dict, want: int) -> int:
    return max(1, want)  # an edit takes no slot; measuring waits for Bat's own


# ── seal and screen ──────────────────────────────────────────────────

def sealed_dir(run_id: str) -> Path:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    return loop.sealed_dir(run_id)


def _factory_cases(run: dict) -> list[dict[str, Any]]:
    from captain_claw.flight_deck import first_mate_cases
    return first_mate_cases.cases(fms.get_store(), run["objective"]["subject"]["factory_id"])


def holdout_strings(cases: list[dict[str, Any]]) -> dict[str, str]:
    """What only a held-out case says: its title, its email addresses, and the
    longer lines of its input."""
    out: dict[str, str] = {}
    for c in cases:
        where = f"holdout:{c['id']}"
        title = " ".join(str(c.get("title") or "").split()).lower()
        if len(title) >= _MIN_STRING:
            out.setdefault(title, where)
        for addr in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", str(c.get("input") or "")):
            out.setdefault(addr.lower(), where)
        for line in str(c.get("input") or "").splitlines():
            line = " ".join(line.split()).lower()
            if len(line) >= 24 and not line.startswith(("from:", "to:", "cc:", "subject:", "date:")):
                out.setdefault(line[:120], where)
    return out


async def seal(run: dict, base: str) -> None:
    """Fix the run's split of the factory's cases, and what only the held-out
    ones say, for the screen. Owner-only files beside the run."""
    d = sealed_dir(run["id"])
    if (d / "split.json").is_file():
        return
    cases = await asyncio.to_thread(_factory_cases, run)
    dev, hold = split(cases)
    if not dev or not hold:
        raise RuntimeError("the factory has too few cases to hold any out")
    by_id = {c["id"]: c for c in cases}
    shared = set(holdout_strings([by_id[i] for i in dev]))
    strings = {s: src for s, src in holdout_strings([by_id[i] for i in hold]).items() if s not in shared}
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    for name, data in (("strings.json", strings), ("split.json", {"base": base, "dev": dev, "holdout": hold})):
        (d / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.chmod(d / name, 0o600)


def _split_of(run: dict) -> dict[str, list[str]]:
    try:
        return json.loads((sealed_dir(run["id"]) / "split.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"dev": [], "holdout": []}


async def screen(run: dict, exp: dict) -> dict:
    """In bounds and fair: a real change, a charter of the right size, nothing
    only a held-out case says, nothing about the eval itself."""
    from captain_claw.flight_deck.captain_swarm_screen import EVAL_MARKERS
    new, old = await content_of(exp["head_sha"]), await content_of(exp["base_sha"])
    changed = []
    if new["charter"] != old["charter"]:
        changed.append(f"charter: {len(old['charter'])} → {len(new['charter'])} characters")
    if new["steps"] != old["steps"]:
        changed.append(f"steps: {len(old['steps'])} → {len(new['steps'])}")
    out = {"changed": changed, "diff_stat": "; ".join(changed) or "no change", "hits": []}
    if not changed:
        return {**out, "ok": False, "reason": "it changed nothing"}
    if len(new["charter"]) < 40:
        return {**out, "ok": False, "reason": "out of bounds: the charter is too short to do the job"}
    try:
        strings = json.loads((sealed_dir(run["id"]) / "strings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        strings = {}
    text = (new["charter"] + "\n" + "\n".join(new["steps"])).lower()
    hits = [{"string": s, "source": src} for s, src in strings.items() if s in text]
    hits += [{"string": mk, "source": "eval"} for mk in EVAL_MARKERS if mk.lower() in text]
    if hits:
        peeked = any(h["source"].startswith("holdout") for h in hits)
        what = "; ".join(repr(h["string"][:60]) for h in hits[:3])
        return {**out, "ok": False, "hits": hits,
                "reason": ("it names a held-out case it was never shown: " if peeked else "it points at the eval: ") + what}
    return {**out, "ok": True, "reason": ""}


# ── measuring ────────────────────────────────────────────────────────

def _case_run_config(run: dict, factory: dict, live: dict, content: dict, tag: str) -> dict[str, Any]:
    from captain_claw.flight_deck import first_mate_runs
    from captain_claw.flight_deck.captain_swarm_objective import tokens_left
    version = {**live, "charter": content["charter"],
               "runner_config": {"worker_mode": content["worker_mode"], "steps": content["steps"]}}
    cap = first_mate_runs.token_cap_for(fms.get_store(), factory, version)
    left = tokens_left(run)
    if left is not None:
        cap = max(50_000, min(cap, int(left)))
    cfg = first_mate_runs.bat_config(factory, version, {"id": "", "kind": "run"}, token_cap=cap)
    cfg.update(source="captain_swarm", rehearsal=tag, fm_run="")
    return cfg


async def _one_case(run: dict, factory: dict, live: dict, content: dict, case: dict, open_cases: list[dict],
                    should_stop) -> tuple[float, str, int, float]:
    """One case, one Bat run in rehearsal, graded: ``(score, note, tokens, usd)``."""
    from captain_claw.flight_deck import bat_loop, first_mate_cases, first_mate_runs
    from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
    from captain_claw.flight_deck.bat_store import TERMINAL_STATES
    bat = bat_loop.get_store()
    if bat is None:
        raise RuntimeError("Bat isn't running on this deck")
    version = {**live, "charter": content["charter"]}
    examples = [c for c in open_cases if c["id"] != case["id"]]
    task = first_mate_runs.compose_brief(factory, version, note=CASE_NOTE, rehearsal=True, examples=examples,
                                         context=[{"source": "a test case", "summary": case["title"],
                                                   "body": case["input"]}])
    tag = rehearsal.make_tag(run["id"], f"exp_{secrets.token_hex(6)}")
    bat_id = f"bat_{secrets.token_hex(6)}"
    await bat.create_run(run_id=bat_id, owner_id=run["owner_id"], title=f"Improve {factory['title']}: a case",
                         task=task, config=_case_run_config(run, factory, live, content, tag), origin={},
                         llm_usd_cap=0.0, status="planning")
    try:
        await bat_loop.kick()
    except Exception:  # noqa: BLE001 — it starts on Bat's next tick
        pass
    while True:
        b = await bat.get_run(bat_id)
        if b is None:
            return 0.0, "its Bat run is gone", 0, 0.0
        if b["status"] in TERMINAL_STATES:
            break
        if should_stop():
            await bat_loop.cancel_run(bat, bat_id)
            raise _StoppedError
        await asyncio.sleep(POLL_S)
    tokens, usd = int(b.get("cumulative_tokens") or 0), float(b.get("cumulative_usd") or 0.0)
    if b["status"] != "done":
        return 0.0, f"its run ended {b['status']}", tokens, usd
    score, note = await first_mate_cases.grade(run["owner_id"], case, b.get("truth") or "")
    return score, note, tokens, usd


async def measure(run: dict, version_id: str, which: str) -> dict:
    """The version on the run's ``dev`` or ``holdout`` cases, ``repeat`` times:
    the average grade, stats over repeats, and each ``dev`` case's grade."""
    from captain_claw.flight_deck.captain_swarm_objective import budget_spent
    from captain_claw.flight_deck.captain_swarm_score import stats
    obj = run["objective"]
    subj, m = obj["subject"], obj["metric"]
    fm = fms.get_store()
    factory = fm.get(subj["factory_id"])
    live = fm.live(subj["factory_id"]) if factory else None
    if factory is None or live is None:
        return {"value": None, "error": "the factory is gone or has no live version"}
    by_id = {c["id"]: c for c in await asyncio.to_thread(_factory_cases, run)}
    part = "holdout" if which == "holdout" else "dev"
    ids = [i for i in _split_of(run).get(part, []) if i in by_id]
    if not ids:
        return {"value": None, "error": f"no {part} cases are left"}
    repeat = m["holdout_repeat"] if part == "holdout" else m["repeat"]
    content = await content_of(version_id)
    open_cases = [c for c in by_id.values() if not c.get("sealed", True)]
    store = _store()
    flag = {"stop": False}

    async def watch() -> None:
        while not flag["stop"]:
            await asyncio.sleep(max(POLL_S, 1.0))
            cur = await store.get_run(run["id"])
            if not cur or cur["status"] not in ("baseline", "running") or budget_spent(cur):
                flag["stop"] = True

    gate = asyncio.Semaphore(CASE_PARALLEL)
    spent = {"tokens": 0, "usd": 0.0}

    async def one(case_id: str) -> tuple[str, float, str]:
        async with gate:
            if flag["stop"]:
                raise _StoppedError
            try:
                score, note, tokens, usd = await _one_case(run, factory, live, content, by_id[case_id],
                                                           open_cases, lambda: flag["stop"])
            except Exception:
                flag["stop"] = True  # the others stop too: this measurement can't finish
                raise
            spent["tokens"] += tokens
            spent["usd"] += usd
            await store.add_spend(run["id"], usd, tokens)
            return case_id, score, note

    watcher = asyncio.create_task(watch())
    rounds: list[list[tuple[str, float, str]]] = []
    try:
        for _ in range(repeat):
            rounds.append(list(await asyncio.gather(*(one(i) for i in ids))))
    except _StoppedError:
        cur = await store.get_run(run["id"])
        why = budget_spent(cur) if cur else ""
        return {"value": None, "error": f"{why} — it ran out mid-measurement" if why else "the run was stopped"}
    except RuntimeError as exc:
        return {"value": None, "error": str(exc)[:300]}
    finally:
        flag["stop"] = True
        watcher.cancel()
    means = [sum(s for _, s, _ in r) / len(r) for r in rounds]
    st = stats(means)
    out: dict[str, Any] = {"value": round(st.mean, 4),
                           "stats": {"n": st.n, "mean": st.mean, "stdev": st.stdev, "low": st.low, "high": st.high},
                           "guard": {}, "tokens": spent["tokens"], "usd": round(spent["usd"], 6), "error": ""}
    if part == "dev":
        out["per_case"] = {cid: round(sum(s for i, s, _ in (x for r in rounds for x in r) if i == cid) / len(rounds), 4)
                           for cid in ids}
        out["notes"] = {cid: next((n for i, _, n in rounds[-1] if i == cid), "") for cid in ids}
    return out


# ── the reviewer and the proposer ────────────────────────────────────

async def review_material(run: dict, exp: dict) -> str:
    old, new = await content_of(exp["base_sha"]), await content_of(exp["head_sha"])
    return "\n".join([f"Goal: {run['objective']['title']}.",
                      "Charter before:", old["charter"], "",
                      "Charter now:", new["charter"], "",
                      "Steps before: " + ("; ".join(old["steps"]) or "none"),
                      "Steps now: " + ("; ".join(new["steps"]) or "none"),
                      f"The idea behind the change: {exp['hypothesis']}"])


_SYSTEM = (
    "You improve a standing job that an assistant (a Bat run: workers, a plan, a reviewer) runs again and again for "
    "its owner. Its charter says what one run does; an optional step outline guides the plan. It is graded on test "
    "cases against lines saying what a good result does. Propose {k} DIFFERENT improvements: different ideas, not "
    "rewordings. Each returns the WHOLE new charter (keep every limit the owner set: what it must never do, what it "
    "finishes with; never ask it to do anything outward the old charter didn't) and, if useful, a new step outline "
    "of at most {steps} short steps. Write for kinds of inputs, never for one particular case. Don't repeat an idea "
    "the ledger already discarded unless you say what is different now. Return ONLY a JSON array of objects: "
    '{{"hypothesis": "<the idea in one line>", "strategy_tag": "<2-4 word kebab-case tag>", '
    '"why": "<which mistakes it should fix>", "charter": "<the whole new charter>", "steps": ["<step>"]}}. No prose.'
)


async def brief(run: dict, ledger: list[dict]) -> str:
    """What the proposer is told. Never a sealed case's words, never a held-out number."""
    from captain_claw.flight_deck.captain_swarm_learning import shown_verdict
    obj = run["objective"]
    champ = run.get("champion") or {}
    cur = await content_of(champ.get("sha") or "")
    fm = fms.get_store()
    live = fm.live(obj["subject"]["factory_id"]) or {}
    lines = [f"Goal: {obj['title']}.", "Current charter:", cur["charter"],
             "Current steps: " + ("; ".join(cur["steps"]) or "none (Bat plans on its own)")]
    contract = (live.get("checks") or {}).get("contract") or []
    if contract:
        lines += ["Its done-when lines (fixed):"] + [f"- {c}" for c in contract]
    dev = champ.get("dev") or {}
    if dev.get("stats"):
        lines.append(f"Score on the open part: {dev['stats']['mean']:.2f} (0 to 1).")
    per_case = dev.get("per_case") or {}
    notes = dev.get("notes") or {}
    cases = {c["id"]: c for c in await asyncio.to_thread(_factory_cases, run)}
    shown = sorted(per_case, key=lambda cid: per_case[cid])
    if shown:
        lines.append("\nHow each case went (worst first):")
    for cid in shown[:12]:
        c = cases.get(cid)
        if c is None:
            continue
        if c.get("sealed", True):
            lines.append(f"- a sealed case: {per_case[cid]:.0%}")
            continue
        lines.append(f"- {c['title']}: {per_case[cid]:.0%}. Input: {' '.join(c['input'].split())[:300]}. "
                     f"A good result: {'; '.join(c['expect'])}."
                     + (f" Reviewer: {notes[cid][:160]}" if notes.get(cid) else ""))
    if ledger:
        lines.append("\nTried so far (newest last):")
        for e in ledger[-12:]:
            lines.append(f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:200]}"
                         + (f" → {shown_verdict(e)[:160]}" if e.get("verdict_reason") else ""))
    lines.append(run.get("learned") or "")
    return "\n".join(lines)


def parse(text: str, k: int) -> list[dict]:
    """The ideas in a model's reply; one with neither a charter nor steps is dropped."""
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
        charter = str(item.get("charter") or "").strip()[:fms.CHARTER_MAX]
        steps = _steps(item.get("steps")) if isinstance(item.get("steps"), list) else None
        if len(charter) < 40 and not steps:
            continue
        change: dict[str, Any] = {}
        if len(charter) >= 40:
            change["charter"] = charter
        if steps is not None:
            change["steps"] = steps
        tag = re.sub(r"[^a-z0-9-]+", "-", str(item.get("strategy_tag") or "").lower()).strip("-")[:40]
        out.append({"hypothesis": str(item.get("hypothesis") or "a better charter")[:600], "strategy_tag": tag,
                    "why": str(item.get("why") or "")[:600], "files": [], "change": change})
    return out[:k]


async def propose(run: dict, ledger: list[dict], k: int) -> list[dict]:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    text = await loop.ask_ideas(run, _SYSTEM.format(k=k, steps=STEPS_MAX), await brief(run, ledger))
    return loop.diverse(parse(text, k), ledger)


def parts() -> dict[str, Any]:
    """What this kind brings to Captain Swarm's loop (``captain_swarm_loop.parts_for``)."""
    return {"lane": FactoryLane(), "propose": propose, "measure": measure, "seal": seal, "screen": screen,
            "branch": no_branch, "start": start_point, "capacity": capacity, "queue": lambda run: None,
            "review": review_material, "slice": False, "carry": False}


# ── starting one from First Mate ─────────────────────────────────────

async def start(user: dict[str, Any], factory_id: str, options: dict[str, Any] | None = None,
                *, started_by: str = "the owner, from First Mate") -> dict[str, Any]:
    """An improvement run of the owner's factory, started now (the owner asked
    for it): Captain Swarm's own checks first (admins only, not on the desktop
    app, a model, one run at a time)."""
    from captain_claw.flight_deck import captain_swarm_routes as csr
    await csr.admin_user(user)
    owner = csr._uid(user)
    o = dict(options or {})
    f, live = ready(fms.get_store(), fms.norm_user(owner), factory_id)
    raw = factory_template(f, live)
    for key in ("per_round", "parallel"):
        if o.get(key):
            raw["lane"][key] = o[key]
    for key in ("repeat", "holdout_repeat"):
        if o.get(key):
            raw["metric"][key] = o[key]
    csr._put_budget(raw, o)
    objective = normalize(raw)
    await csr._can_run(owner, objective, user)
    run = await csr._start(owner, objective, status="baseline", started_by=started_by)
    fms.get_store().note(factory_id, "owner", "improvement_started", {"run": run["id"]})
    return run
