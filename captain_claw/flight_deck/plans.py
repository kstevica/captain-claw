"""Long-horizon planning (#4) — a goal executor.

A *plan* decomposes a goal into ordered steps, each a concrete catalog action (or
a nudge), and drives them to completion across pulses/days. Steps execute through
the SAME rail as everything else (``run_action`` → trust ladder → reverse capture),
so a trusted, reversible step auto-runs while an untrusted one pauses for the user.
On step failure the plan replans (re-decomposes the remainder); on abandon it rolls
back the reversible steps it already did via their captured reverse handles.

Phase 1: the store, LLM decomposition, and ``advance_one`` (run the next step).
Autonomous advance on the heartbeat + replan-on-failure build on this.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

PLAN_STATUSES = ("active", "paused", "done", "failed", "abandoned")
STEP_STATUSES = ("pending", "running", "done", "failed", "skipped")

# How many step failures a plan may absorb (re-decomposing the remainder each
# time) before it gives up — bounds the replan loop on an impossible step.
_MAX_REPLANS = 2


def _norm_user(user_id: str | None) -> str:
    return (user_id or "").strip() or "local"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    import os
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "autonomy.db"
    return Path("~/.captain-claw/autonomy.db").expanduser()


class PlansStore:
    """SQLite-backed plan store (shares autonomy.db; sync sqlite3 + lock)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    goal        TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'active',
                    steps       TEXT NOT NULL DEFAULT '[]',  -- json list of step dicts
                    note        TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_plans_user
                    ON plans(user_id, status, created_at DESC);
                """
            )
            self._c().commit()

    def create_plan(self, user_id: str, goal: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        uid = _norm_user(user_id)
        pid = "plan_" + secrets.token_hex(8)
        now = _utcnow_iso()
        norm_steps = []
        for i, s in enumerate(steps):
            norm_steps.append({
                "idx": i, "title": s.get("title", ""), "kind": s.get("kind", "nudge"),
                "action_id": s.get("action_id", ""), "args": s.get("args") or {},
                "status": "pending", "result": "", "reverse": None,
            })
        with self._lock:
            conn = self._c()
            conn.execute(
                "INSERT INTO plans (id, user_id, goal, status, steps, created_at, updated_at)"
                " VALUES (?, ?, ?, 'active', ?, ?, ?)",
                (pid, uid, goal, json.dumps(norm_steps), now, now),
            )
            conn.commit()
        return self.get_plan(pid) or {}

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        return self._row(r) if r else None

    def list_plans(self, user_id: str, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        limit = max(1, min(200, limit))
        with self._lock:
            if status:
                rows = self._c().execute(
                    "SELECT * FROM plans WHERE user_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
                    (uid, status, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    "SELECT * FROM plans WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (uid, limit),
                ).fetchall()
        return [self._row(r) for r in rows]

    def save_plan(self, plan_id: str, *, steps: list[dict] | None = None,
                  status: str | None = None, note: str | None = None) -> dict[str, Any] | None:
        sets, vals = ["updated_at = ?"], [_utcnow_iso()]
        if steps is not None:
            sets.append("steps = ?"); vals.append(json.dumps(steps))
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if note is not None:
            sets.append("note = ?"); vals.append(note)
        vals.append(plan_id)
        with self._lock:
            conn = self._c()
            conn.execute(f"UPDATE plans SET {', '.join(sets)} WHERE id = ?", vals)
            conn.commit()
        return self.get_plan(plan_id)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["steps"] = json.loads(d.get("steps") or "[]")
        except (ValueError, TypeError):
            d["steps"] = []
        return d


_STORE: PlansStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> PlansStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = PlansStore()
        return _STORE


# ── Decomposition ────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a planner. Break the user's goal into a SHORT ordered list of concrete "
    "steps (1-6) that drive it to completion. Each step is one of:\n"
    '  {"kind":"tool_action","action_id":<catalog id>,"args":{…},"title":short}\n'
    '  {"kind":"nudge","title":short}     — a message/check-in to the user\n'
    '  {"kind":"run_prompt","title":short} — a task the agent does with its tools\n'
    "Prefer catalog actions when one fits; fill required args from the goal (never "
    "invent emails/times/names). Reply with ONLY a JSON array, no prose, starting '['."
)


async def decompose_goal(user_id: str, goal: str) -> list[dict[str, Any]]:
    """One LLM pass (through the user's strongest agent) that turns a goal into
    ordered, catalog-validated steps. Returns [] if it can't plan."""
    goal = (goal or "").strip()
    if not goal:
        return []
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    agent = _strongest_agent(user_id)
    if not agent:
        return []
    from captain_claw.flight_deck.action_catalog import list_catalog, get_action, validate_args
    catalog = [a for a in list_catalog(user_id=user_id) if not a["human_only"]]
    cat_text = "\n".join(f"- {a['id']}: {a['label']} · required {a['required']}" for a in catalog)
    try:
        from captain_claw.games.remote_provider import RemoteLLMProvider
        from captain_claw.llm import Message
        provider = RemoteLLMProvider(host=agent["host"], port=agent["port"],
                                     auth=agent["auth"], name=agent.get("name", ""))
        resp = await provider.complete(
            messages=[
                Message(role="system", content=_PLANNER_SYSTEM + "\n\nAction catalog:\n" + cat_text),
                Message(role="user", content=f"Goal: {goal}"),
            ],
            temperature=0.3, max_tokens=1200,
        )
    except Exception as exc:
        _log.warning("plan decomposition failed: %s", exc)
        return []

    txt = (resp.content or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt).strip()
    data: Any = None
    try:
        data = json.loads(txt)
    except (ValueError, TypeError):
        m = re.search(r"\[.*\]", txt, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except (ValueError, TypeError):
                data = None
    if not isinstance(data, list):
        return []

    steps: list[dict[str, Any]] = []
    for raw in data[:6]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip()
        title = str(raw.get("title") or "").strip()
        if kind not in ("tool_action", "nudge", "run_prompt") or not title:
            continue
        if kind == "tool_action":
            spec = get_action(raw.get("action_id"), user_id)
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            if not spec or spec.get("human_only"):
                continue
            ok, _ = validate_args(spec, args)
            if not ok:
                continue
            steps.append({"title": title[:200], "kind": kind,
                          "action_id": str(raw.get("action_id")), "args": args})
        else:
            steps.append({"title": title[:200], "kind": kind, "action_id": "", "args": {}})
    return steps


# ── Execution ────────────────────────────────────────────────────────

async def _run_step(user_id: str, step: dict[str, Any]) -> dict[str, Any]:
    """Execute one step via the shared rail. Returns {ok, content, reverse?}."""
    kind = step.get("kind")
    if kind == "tool_action":
        from captain_claw.flight_deck.actions import run_action
        res = await run_action(user_id, step.get("action_id", ""), step.get("args") or {})
        reverse = None
        if res.get("ok"):
            from captain_claw.flight_deck import action_catalog
            spec = action_catalog.get_action(step.get("action_id"), user_id)
            reverse = action_catalog.build_reverse(spec, res.get("content", "")) if spec else None
        return {"ok": bool(res.get("ok")), "content": res.get("content") or res.get("error") or "", "reverse": reverse}
    # nudge / run_prompt: deliver an instruction to the strongest agent.
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    from captain_claw.flight_deck.actions import run_tool_on_agent  # noqa: F401  (rail import side-effect)
    agent = _strongest_agent(user_id)
    if not agent:
        return {"ok": False, "content": "no running agent", "reverse": None}
    from captain_claw.flight_deck.basna_routes import _dispatch_one
    prompt = (f"[Plan step] {step.get('title')}" if kind == "nudge"
              else f"[Plan task] {step.get('title')}")
    res = await _dispatch_one(int(agent.get("port") or 0), str(agent.get("auth", "")), prompt, 180.0)
    return {"ok": bool(res.get("ok")), "content": str(res.get("output") or "")[:500], "reverse": None}


async def advance_one(user_id: str, plan_id: str, *, auto: bool = False) -> dict[str, Any]:
    """Run the plan's next pending step. ``auto=True`` only runs a step that's
    trust-eligible to auto-fire; otherwise it pauses for approval. Manual advance
    (auto=False) runs the next step regardless (the user advancing IS approval)."""
    store = get_store()
    plan = store.get_plan(plan_id)
    if not plan or plan.get("user_id") not in (_norm_user(user_id), "local"):
        return {"ok": False, "error": "plan not found"}
    if plan["status"] not in ("active", "paused"):
        return {"ok": False, "error": f"plan is {plan['status']}"}
    steps = plan["steps"]
    idx = next((i for i, s in enumerate(steps) if s["status"] == "pending"), None)
    if idx is None:
        store.save_plan(plan_id, status="done")
        return {"ok": True, "done": True}
    step = steps[idx]

    # Auto mode: only run a step that has earned/granted auto-fire; else pause.
    if auto and step.get("kind") == "tool_action":
        from captain_claw.flight_deck.fd_dispatch import should_auto_dispatch
        from captain_claw.flight_deck.autonomy import resolve_config
        cfg = resolve_config(user_id)
        pseudo = {"kind": "tool_action", "user_id": _norm_user(user_id),
                  "payload": {"action_id": step.get("action_id")}}
        if not should_auto_dispatch(cfg, pseudo):
            store.save_plan(plan_id, status="paused", note=f"awaiting approval: {step['title']}")
            return {"ok": True, "paused": True, "step": step["title"]}
    if auto and step.get("kind") in ("nudge", "run_prompt"):
        # Non-catalog steps aren't auto-run in Phase 1 — pause for the user.
        store.save_plan(plan_id, status="paused", note=f"awaiting approval: {step['title']}")
        return {"ok": True, "paused": True, "step": step["title"]}

    step["status"] = "running"
    store.save_plan(plan_id, steps=steps)
    res = await _run_step(user_id, step)
    step["result"] = str(res.get("content") or "")[:500]
    step["reverse"] = res.get("reverse")
    if res.get("ok"):
        step["status"] = "done"
        remaining = any(s["status"] == "pending" for s in steps)
        store.save_plan(plan_id, steps=steps, status="active" if remaining else "done", note="")
        return {"ok": True, "step": step["title"], "done": not remaining}
    step["status"] = "failed"
    store.save_plan(plan_id, steps=steps)
    # Replan-on-failure (#4 / Theme D): re-decompose what's left rather than
    # failing the whole plan — unless we've already failed too many times (cap
    # avoids an infinite re-plan loop on a step that can't ever succeed).
    failed_count = sum(1 for s in steps if s["status"] == "failed")
    if failed_count <= _MAX_REPLANS:
        done_titles = [s["title"] for s in steps if s["status"] == "done"]
        replan_goal = (
            f"Original goal: {plan['goal']}\n"
            + (f"Already completed: {'; '.join(done_titles)}\n" if done_titles else "")
            + f"The step \"{step['title']}\" just FAILED ({step['result'][:200]}).\n"
            "Produce the remaining steps to still achieve the goal, taking a "
            "different approach for what failed. Omit anything already completed."
        )
        new_steps = await decompose_goal(user_id, replan_goal)
        if new_steps:
            kept = [s for s in steps if s["status"] in ("done", "failed")]
            for ns in new_steps:
                ns["status"] = "pending"
            store.save_plan(plan_id, steps=kept + new_steps, status="active",
                            note=f"replanned after failed step: {step['title']}")
            return {"ok": True, "replanned": True, "step": step["title"],
                    "new_steps": len(new_steps)}
    store.save_plan(plan_id, steps=steps, status="failed", note=f"step failed: {step['title']}")
    return {"ok": False, "error": f"step failed: {step['title']}", "detail": step["result"]}


async def abandon_plan(user_id: str, plan_id: str) -> dict[str, Any]:
    """Abandon a plan and roll back its completed, reversible steps (newest first)."""
    store = get_store()
    plan = store.get_plan(plan_id)
    if not plan or plan.get("user_id") not in (_norm_user(user_id), "local"):
        return {"ok": False, "error": "plan not found"}
    undone = 0
    from captain_claw.flight_deck.actions import run_tool_on_agent
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    agent = _strongest_agent(user_id)
    for step in reversed(plan["steps"]):
        rev = step.get("reverse")
        if step.get("status") == "done" and isinstance(rev, dict) and rev.get("tool") and agent:
            try:
                r = await run_tool_on_agent(agent, rev["tool"], rev.get("args") or {})
                if r.get("ok"):
                    step["status"] = "skipped"
                    undone += 1
            except Exception as exc:
                _log.warning("plan rollback step failed: %s", exc)
    store.save_plan(plan_id, steps=plan["steps"], status="abandoned",
                    note=f"abandoned; rolled back {undone} step(s)")
    return {"ok": True, "rolled_back": undone}
