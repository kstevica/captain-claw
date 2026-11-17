"""First Mate's cases and checks: a factory measured on inputs it doesn't see coming.

See docs/first-mate-plan.md §7 (Phase 4, piece 2).

**Cases.** Each factory keeps cases: an input (an email, a request, a page:
what a run would be given) and what a good result must do, one line each. A
case is ``sealed`` (the default) or open. All live in the factory's sealed
folder (``FirstMateStore.sealed_dir``), out of any worker's reach; the owner
sees and edits them. Open ones go into the brief as examples (up to three,
never the case a check is running); sealed ones never reach a worker or
anything that writes or improves the brief. They come from the owner, or from a past run that had a fixed input (an
event-triggered run's emails): :func:`case_draft` starts one for the owner to
finish. A run on the live mailbox has no fixed input, so it can't become a case.

**Checks.** A check runs one version on every case, one at a time, each as a
rehearsal (``kind`` ``check``: nothing outside changes, nothing is proposed).
A reviewer on a different model from the workers (Bat's ``_panel_creds``)
grades each result against the case's lines: full, partial or none. No
readable answer counts as nothing. The check's score is the average. Checks
are driven from the supervisor's scan (:func:`drive`).

* **approve**: approving a new version of a factory with cases starts a check
  of that draft first. It goes live only when it scores at least the live
  version's last score (within :data:`TOLERANCE`, the reviewer's noise),
  through ``FirstMateStore.approve`` with the hash it was checked on, so a
  draft changed during the check doesn't go live. Otherwise it stays waiting
  and the owner is told. The very first gate has no earlier score: it passes
  and sets the baseline.
* **weekly**: the live version, once a week while the owner's switch is on. A
  drop of :data:`DRIFT` or more turns the factory amber (``first_mate_score``).
* **manual**: the owner's "Run the checks now".

A check ends ``void`` when its factory can't run any more (retired, over
budget) or its version changed under it.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)

CASES_MAX = 30
EXPECT_MAX = 10
INPUT_MAX = 6000
TOLERANCE = 0.05
DRIFT = 0.1
WEEK = timedelta(days=7)
CASE_NOTE = ("This is a test case. Work only from the input below, as if it had just arrived; don't look up live "
             "mail, calendars or accounts for it. Do the work as the charter asks, and finish with your result.")
_JUDGE = ("You grade one result of a standing job against a checklist, item by item. For each numbered item say how "
          "fully the result itself does it: full, partial or none. Grade only what the result shows, not what it "
          'promises. Return ONLY JSON: {"grades": [{"item": <number>, "grade": "full|partial|none"}]}')
_GRADES = {"full": 1.0, "partial": 0.5, "none": 0.0}


# ── cases (the sealed folder) ──────────────────────────────────────────

def _path(store: fms.FirstMateStore, factory_id: str):
    return store.sealed_dir(factory_id) / "cases.json"


def cases(store: fms.FirstMateStore, factory_id: str) -> list[dict[str, Any]]:
    p = _path(store, factory_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return []
    return [c for c in data.get("cases") or [] if isinstance(c, dict) and c.get("id")]


def _save(store: fms.FirstMateStore, factory_id: str, items: list[dict[str, Any]]) -> None:
    p = _path(store, factory_id)
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cases": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(p)


def _checked(title: Any, input_: Any, expect: Any, sealed: Any) -> dict[str, Any]:
    title = fms._text("title", title, fms.TITLE_MAX, required=True)
    input_ = fms._text("input", input_, INPUT_MAX, required=True)
    lines = fms._str_list("expect", expect, items=EXPECT_MAX)
    if not lines:
        raise FirstMateError("a case needs at least one line saying what a good result does")
    fms._no_secret("the case", " ".join([title, input_, *lines]).split())  # a key pasted anywhere in it
    return {"title": title, "input": input_, "expect": lines, "sealed": fms._bool("sealed", sealed)}


def add_case(store: fms.FirstMateStore, factory_id: str, *, title: Any, input: Any, expect: Any,
             sealed: Any = True, origin: str = "owner", from_run: str = "") -> dict[str, Any]:
    if store.get(factory_id) is None:
        raise FirstMateError("no such factory")
    items = cases(store, factory_id)
    if len(items) >= CASES_MAX:
        raise FirstMateError(f"a factory keeps at most {CASES_MAX} cases")
    case = {"id": "case_" + secrets.token_hex(6), **_checked(title, input, expect, sealed),
            "origin": origin, "from_run": from_run[:40], "created_at": fms._iso(fms._utcnow())}
    _save(store, factory_id, [*items, case])
    store.note(factory_id, "owner", "case_added", {"case": case["id"], "sealed": case["sealed"]})
    return case


def update_case(store: fms.FirstMateStore, factory_id: str, case_id: str, **fields: Any) -> dict[str, Any]:
    items = cases(store, factory_id)
    for n, c in enumerate(items):
        if c["id"] == case_id:
            merged = _checked(fields.get("title", c["title"]), fields.get("input", c["input"]),
                              fields.get("expect", c["expect"]), fields.get("sealed", c["sealed"]))
            items[n] = {**c, **merged}
            _save(store, factory_id, items)
            return items[n]
    raise FirstMateError("no such case")


def delete_case(store: fms.FirstMateStore, factory_id: str, case_id: str) -> None:
    items = cases(store, factory_id)
    left = [c for c in items if c["id"] != case_id]
    if len(left) == len(items):
        raise FirstMateError("no such case")
    _save(store, factory_id, left)
    store.note(factory_id, "owner", "case_deleted", {"case": case_id})


def case_draft(store: fms.FirstMateStore, run: dict[str, Any]) -> dict[str, Any]:
    """A case started from a past run, for the owner to finish: the run's fixed
    input (the events that started it), and its contract as a first guess at
    what a good result does. ``input`` is "" for a run with no fixed input."""
    from captain_claw.flight_deck import first_mate_events
    evs = first_mate_events._events_store()
    parts = []
    for ref in run.get("trigger_ref") or []:
        ev = evs.get_event(str(ref)) if evs is not None else None
        if ev:
            item = first_mate_events.context_of(ev)
            parts.append(f"{item['summary']}\n{item['body']}".strip())
    version = store.version(run["factory_id"], run["version"]) or {}
    return {"title": f"Like the run of {str(run.get('created_at') or '')[:10]}", "input": "\n\n".join(parts)[:INPUT_MAX],
            "expect": list((version.get("checks") or {}).get("contract") or []), "from_run": run["id"],
            "has_input": bool(parts)}


# ── grading ────────────────────────────────────────────────────────────

def parse_grades(text: str, n: int) -> list[float] | None:
    from captain_claw.flight_deck.captain_swarm_document import parse_grades as swarm_parse
    return swarm_parse(text, n)


async def _ask_reviewer(owner: str, system: str, user: str) -> str:
    """One call to the owner's reviewer: the best model the workers don't run on."""
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
    from captain_claw.flight_deck.bat_routes import _panel_creds
    from captain_claw.llm import Message
    tiers, _ = await _load_owner_tiers(get_db(), owner)
    creds = _panel_creds(tiers)[0]
    if not creds:
        raise RuntimeError("no reviewer model: set up a tier set first")
    prov, max_tokens = _provider_call(creds, temperature=0.2, default_max=2048, cap=4096)
    resp = await prov.complete(messages=[Message(role="system", content=system), Message(role="user", content=user)],
                               temperature=0.2, max_tokens=max_tokens)
    return resp.content or ""


async def grade(owner: str, case: dict[str, Any], result: str) -> tuple[float, str]:
    """``(score 0..1, note)``. Unreadable or no reviewer: 0, with why."""
    items = case["expect"]
    user = (f"The input the job got:\n{case['input'][:3000]}\n\nThe checklist:\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
            + f"\n\nThe result:\n{(result or '(nothing)')[:12000]}")
    try:
        for _ in range(2):
            got = parse_grades(await _ask_reviewer(owner, _JUDGE, user), len(items))
            if got is not None:
                return round(sum(got) / len(got), 3), ""
        return 0.0, "the reviewer's grades couldn't be read"
    except Exception as exc:  # noqa: BLE001 — no answer is a fail, never a pass
        return 0.0, f"no reviewer answer: {exc}"[:200]


# ── checks ─────────────────────────────────────────────────────────────

def start_check(store: fms.FirstMateStore, factory_id: str, *, version: int, purpose: str) -> dict[str, Any]:
    """A check of ``version`` on every case. The approve gate's baseline is the
    live version's last score. The approve gate takes the place of a weekly or
    manual check still going."""
    items = cases(store, factory_id)
    f = store.get(factory_id)
    going = store.active_check(factory_id)
    if going is not None and purpose == "approve":
        if going["purpose"] == "approve" and going["version"] == version:
            return going
        store.finish_check(going["id"], "void", note=f"the check of version {version}, waiting for approval, "
                                                     "went first")
    baseline = store.last_score(factory_id, f["version"]) if f and f["version"] else None
    return store.create_check(factory_id, version=version, purpose=purpose, cases=[c["id"] for c in items],
                              baseline=baseline)


async def drive(store: fms.FirstMateStore, now: datetime) -> None:
    """Move every running check on one step. Never raises."""
    for check in store.running_checks():
        try:
            await _step(store, check)
        except Exception:  # noqa: BLE001
            log.warning("first mate: driving a check failed", extra={"check": check.get("id")}, exc_info=True)


async def _step(store: fms.FirstMateStore, check: dict[str, Any]) -> None:
    from captain_claw.flight_deck import first_mate_runs as runs
    f = store.get(check["factory_id"])
    version = store.version(check["factory_id"], check["version"])
    if f is None or version is None or version["sha"] != check["version_sha"]:
        store.finish_check(check["id"], "void", note="its version changed during the check")
        return
    if f["state"] in ("retired", "paused_budget"):
        store.finish_check(check["id"], "void", note=f"the factory is {f['state'].replace('_', ' ')}")
        return
    if check["purpose"] == "approve" and (version["status"] not in fms.OPEN_VERSION
                                          or f["draft_version"] != check["version"]):
        store.finish_check(check["id"], "void", note="the draft isn't waiting for approval any more")
        return
    by_id = {c["id"]: c for c in cases(store, f["id"])}
    results = dict(check["results"])
    for case_id in check["cases"]:
        res = results.get(case_id)
        if res is None:
            if case_id not in by_id:  # deleted since the check began
                store.set_check_result(check["id"], case_id, {"score": None, "note": "the case was deleted"})
                continue
            case = by_id[case_id]
            try:
                run = await runs.start(f["id"], f["owner_id"], trigger="check", kind="check", version=check["version"],
                                       note=CASE_NOTE, refs=[case_id], watch=False,
                                       context=[{"source": "a test case", "summary": case["title"], "body": case["input"]}])
            except FirstMateError as exc:
                if "already has a run going" in str(exc) or "no free slot" in str(exc) or "Bat isn't" in str(exc):
                    return  # wait for the next scan
                store.finish_check(check["id"], "failed", note=str(exc)[:300])
                return
            store.set_check_result(check["id"], case_id, {"run_id": run["id"], "score": None})
            return
        if res.get("score") is not None or res.get("note") == "the case was deleted":
            continue
        run = store.get_run(res.get("run_id", ""))
        if run is None or run["status"] not in fms.TERMINAL_RUN:
            return  # still going
        if run["status"] == "done" and res.get("run_id"):
            score, note = await grade(f["owner_id"], by_id.get(case_id, {"expect": ["a good result"], "input": ""}),
                                      run.get("result", ""))
        else:
            score, note = 0.0, f"its run {run['status']}"
        store.set_check_result(check["id"], case_id, {"run_id": run["id"], "score": score, "note": note})
        return  # one grade per step
    graded = [r["score"] for r in store.get_check(check["id"])["results"].values() if r.get("score") is not None]
    score = round(sum(graded) / len(graded), 3) if graded else 0.0
    await _settle(store, store.get_check(check["id"]), f, score)


async def _settle(store: fms.FirstMateStore, check: dict[str, Any], f: dict[str, Any], score: float) -> None:
    from captain_claw.flight_deck import first_mate_line
    baseline = check.get("baseline")
    if check["purpose"] != "approve":
        previous = store.last_score(f["id"], check["version"])
        store.finish_check(check["id"], "done", score=score)
        if previous is not None and score <= previous - DRIFT:
            await first_mate_line.check_dropped(f, score, previous)
        return
    if baseline is not None and score < baseline - TOLERANCE:
        note = f"It scored {score:.0%} on its cases; the live version scored {baseline:.0%}. It stays a draft."
        store.finish_check(check["id"], "done", score=score, note=note)
        await first_mate_line.check_held(f, check["version"], note)
        return
    try:
        store.approve(f["id"], check["version"], check["version_sha"], f["owner_id"])
    except FirstMateError as exc:
        store.finish_check(check["id"], "void", score=score, note=f"it couldn't go live: {exc}")
        return
    note = (f"It scored {score:.0%} on its cases" + (f" (the live version: {baseline:.0%})" if baseline is not None
                                                     else ", the first score to compare with") + ", and went live.")
    store.finish_check(check["id"], "done", score=score, note=note)
    await first_mate_line.check_passed(store.get(f["id"]), check["version"], note)


def weekly_due(store: fms.FirstMateStore, f: dict[str, Any], now: datetime) -> bool:
    """A running factory with cases, not checked this week, nothing being checked."""
    if f["state"] != "running" or not f["version"] or not cases(store, f["id"]):
        return False
    if store.active_check(f["id"]):
        return False
    last = store.checks(f["id"], limit=1)
    return not last or now - datetime.fromisoformat(last[0]["created_at"]) >= WEEK
