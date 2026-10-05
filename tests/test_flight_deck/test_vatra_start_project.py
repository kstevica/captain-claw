"""start_vatra must honour the plan-time fields VatraStartRequest declares —
``archetype_ids`` (pinned cast), ``project_id`` (theme + folder),
``reference_folders`` and ``knowledge_session_ids`` — exactly as /route does.

/start decomposes in the background Group 0 pre-phase (plan_vatra_group0 →
_ensure_route) instead of in the request, and that path used to drop all four:
a product BFF (Lupa's commissions / Studio runs) that pinned a house cast or bound
a project got a Lead-selected team with no project context or references.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from captain_claw.flight_deck import basna_routes as br
from captain_claw.flight_deck import vatra_routes as vr

_TIERS = {"reason": {"provider": "openai", "model": "m"}}
_THEME_DESC = "Acme research desk"
_PRIOR = "\n\n## Prior knowledge from earlier runs — build on it\nPRIOR-REPORT-TEXT"


class _DB:
    def __init__(self):
        self.sessions: dict[str, dict] = {
            # A finished prior run used as a knowledge seed.
            "prior-01": {"id": "prior-01", "intent": "earlier", "truth": "Earlier report",
                         "config": json.dumps({"mode": "vatra",
                                               "vfs_project": "prior-run-folder"})},
        }
        self.projects = {"p1": {"id": "p1", "name": "Acme", "description": _THEME_DESC,
                                "instructions": "Cite primary sources.",
                                "vfs_folder": "proj-acme"}}
        self._n = 0

    async def get_all_settings(self, owner_id):
        return {}

    async def create_basna_session(self, user_id, intent, title="", config="{}"):
        self._n += 1
        sid = f"sess{self._n:04d}-new"
        self.sessions[sid] = {"id": sid, "user_id": user_id, "intent": intent,
                              "title": title, "config": config, "status": "", "route": ""}
        return dict(self.sessions[sid])

    async def get_basna_session(self, sid, uid):
        s = self.sessions.get(sid)
        return dict(s) if s else None

    async def update_basna_session(self, sid, uid, **kw):
        self.sessions[sid].update(kw)

    async def delete_basna_session(self, sid, uid):
        return bool(self.sessions.pop(sid, None))

    async def get_basna_project(self, pid, uid):
        return self.projects.get(pid)


def _wire(monkeypatch, db):
    """Stub everything around the real start → Group 0 → _ensure_route path; capture
    what the Lead decomposes on and what the coordination planner is shown."""
    cap: dict = {"build": [], "planner": [], "prior": [], "created": []}

    async def _noop(*a, **k):
        return None

    async def _archetypes(db_, uid):
        return [{"id": "deep-researcher", "role": "Deep Researcher"},
                {"id": "fact-checker", "role": "Fact Checker"},
                {"id": "analyst", "role": "Analyst"}]

    async def _fake_build_plan(db_, uid, intent, max_agents, creds, force_ids=None, **kw):
        cap["build"].append({"intent": intent, "force_ids": force_ids, **kw})
        owners = list(force_ids or ["analyst"])
        subtasks = [{"id": f"s{i}", "title": f"piece {i}", "owner_archetype_id": a,
                     "brief": "b", "depends_on": []} for i, a in enumerate(owners)]
        ctx = "Lead conventions."
        if kw.get("prior_knowledge"):  # _build_plan folds prior knowledge first
            ctx = kw["prior_knowledge"].strip() + "\n\n" + ctx
        return {"mode": "vatra", "domain": "research", "rationale": "r",
                "shared_context": ctx, "subtasks": subtasks,
                "selected": [{"archetype_id": a, "role": "", "why": ""} for a in owners]}

    async def _fake_planner(request, user, sid, *, shared_context, subtasks, **kw):
        cap["planner"].append({"sid": sid, "shared_context": shared_context,
                               "owners": [s["owner_archetype_id"] for s in subtasks]})
        return {"overview": "", "agents": []}

    async def _fake_prior(db_, uid, ids, *, include_board=False, **kw):
        cap["prior"].append({"ids": list(ids), "include_board": include_board})
        return _PRIOR

    class _T:
        def add_done_callback(self, _cb):
            pass

    def _create_task(coro):
        cap["created"].append(coro)
        return _T()

    monkeypatch.setattr(vr, "get_db", lambda: db)
    monkeypatch.setattr(vr, "merged_archetypes", _archetypes)
    monkeypatch.setattr(vr, "_refresh_system_provider_keys", _noop)
    monkeypatch.setattr(vr, "_load_registry", lambda: {})
    monkeypatch.setattr(vr, "_resolve_creds", lambda *a, **k: {})
    for name in ("_progress", "_progress_start", "_progress_done", "_phase"):
        monkeypatch.setattr(vr, name, lambda *a, **k: None)
    monkeypatch.setattr(vr, "_build_plan", _fake_build_plan)
    monkeypatch.setattr(vr, "_run_group0_planner", _fake_planner)
    monkeypatch.setattr(br, "build_prior_knowledge", _fake_prior)
    monkeypatch.setattr(vr.asyncio, "create_task", _create_task)
    return cap


def _body(**kw):
    base = dict(intent="Map the EU battery market", tiers=_TIERS,
                archetype_ids=["deep-researcher", "fact-checker"], project_id="p1",
                reference_folders=["extra-ref"], knowledge_session_ids=["prior-01"],
                knowledge_include_board=True)
    base.update(kw)
    return vr.VatraStartRequest(**base)


async def _start(body, cap):
    req = SimpleNamespace(state=SimpleNamespace(user_id="u1"))
    res = await vr.start_vatra(body, req, {"id": "u1"})
    while cap["created"]:  # run the backgrounded Group 0 pre-phase inline
        await cap["created"].pop(0)
    return res


async def test_start_decomposes_with_pinned_cast_and_project(monkeypatch):
    db = _DB()
    cap = _wire(monkeypatch, db)
    res = await _start(_body(), cap)
    sid = res["session_id"]

    # The Lead decomposed with the pinned cast, against the project theme, seeded
    # with the knowledge runs' preamble (board included as requested).
    (call,) = cap["build"]
    assert call["force_ids"] == ["deep-researcher", "fact-checker"]
    assert call["intent"].startswith("## Project: Acme")
    assert _THEME_DESC in call["intent"]
    assert call["intent"].endswith("Map the EU battery market")
    assert call["prior_knowledge"] == _PRIOR
    assert cap["prior"] == [{"ids": ["prior-01"], "include_board": True}]

    # The persisted plan's shared_context carries the read-only references (project
    # folder + extra folder + the knowledge run's folder) and the prior knowledge…
    sess = db.sessions[sid]
    route = json.loads(sess["route"])
    for folder in ("proj-acme", "extra-ref", "prior-run-folder"):
        assert f"vfs:{folder}/" in route["shared_context"]
    assert "Reference folders (READ-ONLY)" in route["shared_context"]
    assert "PRIOR-REPORT-TEXT" in route["shared_context"]
    # …and the Group 0 planner drafts the coordination plan against that context.
    (plan_call,) = cap["planner"]
    assert plan_call["shared_context"] == route["shared_context"]
    assert plan_call["owners"] == ["deep-researcher", "fact-checker"]
    assert sess["status"] == "awaiting_plan"

    # The project binding + cast are persisted so the UI groups the run, execute
    # folds the theme into shared_context, and a resume re-decomposes on the cast.
    cfg = json.loads(sess["config"])
    assert cfg["project_id"] == "p1"
    assert cfg["project_context"].startswith("## Project: Acme")
    assert cfg["force_ids"] == ["deep-researcher", "fact-checker"]


async def test_start_seeds_the_plan_exactly_like_route(monkeypatch):
    db = _DB()
    cap = _wire(monkeypatch, db)
    routed = await vr.route_vatra(_body(), {"id": "u1"})
    started = await _start(_body(), cap)

    route_call, start_call = cap["build"]
    for key in ("intent", "force_ids", "prior_knowledge"):
        assert start_call[key] == route_call[key], key
    r_route = json.loads(db.sessions[routed["session_id"]]["route"])
    s_route = json.loads(db.sessions[started["session_id"]]["route"])
    assert s_route["shared_context"] == r_route["shared_context"]
    r_cfg = json.loads(db.sessions[routed["session_id"]]["config"])
    s_cfg = json.loads(db.sessions[started["session_id"]]["config"])
    assert (s_cfg["project_id"], s_cfg["project_context"]) == \
        (r_cfg["project_id"], r_cfg["project_context"])


async def test_start_without_seeds_is_unchanged(monkeypatch):
    db = _DB()
    cap = _wire(monkeypatch, db)
    res = await _start(_body(archetype_ids=[], project_id="", reference_folders=[],
                             knowledge_session_ids=[], knowledge_include_board=False), cap)

    (call,) = cap["build"]
    assert call["intent"] == "Map the EU battery market"
    assert call["force_ids"] is None
    assert not call.get("prior_knowledge")
    assert cap["prior"] == []
    route = json.loads(db.sessions[res["session_id"]]["route"])
    assert route["shared_context"] == "Lead conventions."
    cfg = json.loads(db.sessions[res["session_id"]]["config"])
    assert not {"project_id", "project_context", "force_ids"} & set(cfg)
