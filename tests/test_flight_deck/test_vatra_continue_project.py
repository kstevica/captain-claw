"""A continuation round of a project-bound Vatra run must keep the project folder
as a read-only reference.

Round 1 (/route or /start) folds the project's folder into shared_context as a
reference directive, but a continuation re-decomposes from scratch and used to
discard the folder (`_ptheme, _ = _project_context(...)`), so a standing brief
stopped consulting the project corpus after round 1.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck import vatra_routes as vr

_TIERS = {"reason": {"provider": "openai", "model": "m"}}


class _DB:
    def __init__(self):
        self.sessions: dict[str, dict] = {
            "parent01-run": {
                "id": "parent01-run", "intent": "Map the EU battery market",
                "title": "EU batteries", "truth": "# Report\nRound one findings.",
                "analysis": "{}",
                "config": json.dumps({"mode": "vatra", "project_id": "p1",
                                      "project_context": "## Project: Acme\nold theme",
                                      "vfs_project": "vatra-parent01"}),
                "route": json.dumps({"mode": "vatra", "subtasks": [
                    {"id": "s0", "owner_archetype_id": "deep-researcher"},
                    {"id": "s1", "owner_archetype_id": "fact-checker"}]}),
            },
        }
        self.projects = {"p1": {"id": "p1", "name": "Acme", "description": "Acme desk",
                                "instructions": "", "vfs_folder": "proj-acme"}}
        self._n = 0

    async def create_basna_session(self, user_id, intent, title="", config="{}"):
        self._n += 1
        sid = f"child{self._n:03d}-run"
        self.sessions[sid] = {"id": sid, "user_id": user_id, "intent": intent,
                              "title": title, "config": config, "status": "", "route": ""}
        return dict(self.sessions[sid])

    async def get_basna_session(self, sid, uid):
        s = self.sessions.get(sid)
        return dict(s) if s else None

    async def update_basna_session(self, sid, uid, **kw):
        self.sessions[sid].update(kw)

    async def get_basna_project(self, pid, uid):
        return self.projects.get(pid)


async def test_continuation_keeps_project_folder_reference(monkeypatch, tmp_path):
    db = _DB()
    cap: dict = {"build": [], "planner": [], "executed": [], "created": []}

    async def _noop(*a, **k):
        return None

    async def _archetypes(db_, uid):
        return [{"id": "deep-researcher", "role": "Deep Researcher"},
                {"id": "fact-checker", "role": "Fact Checker"}]

    async def _fake_build_plan(db_, uid, intent, max_agents, creds, force_ids=None, **kw):
        cap["build"].append({"intent": intent, "force_ids": force_ids, **kw})
        return {"mode": "vatra", "domain": "research", "rationale": "r",
                "shared_context": "Lead conventions.",
                "subtasks": [{"id": f"s{i}", "title": "t", "owner_archetype_id": a,
                              "brief": "b", "depends_on": []}
                             for i, a in enumerate(force_ids or ["deep-researcher"])],
                "selected": []}

    async def _fake_planner(request, user, sid, *, shared_context, **kw):
        cap["planner"].append(shared_context)
        return {"overview": "", "agents": []}

    async def _fake_execute(body, request, user):
        cap["executed"].append(body.session_id)
        return {"status": "done"}

    class _T:
        def add_done_callback(self, _cb):
            pass

    monkeypatch.setattr(vr, "get_db", lambda: db)
    monkeypatch.setattr(vr, "merged_archetypes", _archetypes)
    monkeypatch.setattr(vr, "_refresh_system_provider_keys", _noop)
    monkeypatch.setattr(vr, "_load_registry", lambda: {})
    monkeypatch.setattr(vr, "_resolve_creds", lambda *a, **k: {})
    for name in ("_progress", "_progress_start", "_progress_done", "_phase"):
        monkeypatch.setattr(vr, name, lambda *a, **k: None)
    monkeypatch.setattr(vr, "_build_plan", _fake_build_plan)
    monkeypatch.setattr(vr, "_run_group0_planner", _fake_planner)
    monkeypatch.setattr(vr, "execute_vatra", _fake_execute)
    monkeypatch.setattr(vr, "_session_files_dir", lambda sid: tmp_path)
    monkeypatch.setattr(vr, "_vfs_manifest", lambda *a, **k: "")
    monkeypatch.setattr(vr.asyncio, "create_task",
                        lambda c: (cap["created"].append(c), _T())[1])

    res = await vr._continue_run("u1", "parent01-run", {"id": "u1"},
                                 instruction="Extend to 2027 forecasts", kind="continue",
                                 tiers=_TIERS, env_vars=None, api_key="")
    while cap["created"]:  # headless Group 0 → auto-approve → execute, inline
        await cap["created"].pop(0)
    sid = res["session_id"]

    # The round's plan points every worker at the project folder (read-only)…
    route = json.loads(db.sessions[sid]["route"])
    assert "vfs:proj-acme/" in route["shared_context"]
    assert "Reference folders (READ-ONLY)" in route["shared_context"]
    # …the Group 0 planner sees it, and the run executes on that persisted plan.
    assert "vfs:proj-acme/" in cap["planner"][0]
    assert cap["executed"] == [sid]
    # The chain stays in the bundle; the continuation keeps its own framing (the
    # theme is folded into shared_context at execute, not prefixed to the Lead's task).
    cfg = json.loads(db.sessions[sid]["config"])
    assert cfg["project_id"] == "p1"
    assert cfg["project_context"].startswith("## Project: Acme")
    (call,) = cap["build"]
    assert call["intent"].startswith("CONTINUE an existing deliverable")


async def test_continuation_without_project_gets_no_reference(monkeypatch, tmp_path):
    db = _DB()
    parent = db.sessions["parent01-run"]
    parent["config"] = json.dumps({"mode": "vatra", "vfs_project": "vatra-parent01"})
    seen: list[str] = []

    async def _noop(*a, **k):
        return None

    async def _archetypes(db_, uid):
        return [{"id": "deep-researcher", "role": "Deep Researcher"}]

    async def _fake_build_plan(db_, uid, intent, max_agents, creds, force_ids=None, **kw):
        return {"mode": "vatra", "domain": "d", "rationale": "", "selected": [],
                "shared_context": "Lead conventions.",
                "subtasks": [{"id": "s0", "title": "t", "owner_archetype_id": "deep-researcher",
                              "brief": "b", "depends_on": []}]}

    async def _fake_planner(request, user, sid, *, shared_context, **kw):
        seen.append(shared_context)
        return {"overview": "", "agents": []}

    async def _fake_execute(body, request, user):
        return {"status": "done"}

    class _T:
        def add_done_callback(self, _cb):
            pass

    created: list = []
    monkeypatch.setattr(vr, "get_db", lambda: db)
    monkeypatch.setattr(vr, "merged_archetypes", _archetypes)
    monkeypatch.setattr(vr, "_refresh_system_provider_keys", _noop)
    monkeypatch.setattr(vr, "_load_registry", lambda: {})
    monkeypatch.setattr(vr, "_resolve_creds", lambda *a, **k: {})
    for name in ("_progress", "_progress_start", "_progress_done", "_phase"):
        monkeypatch.setattr(vr, name, lambda *a, **k: None)
    monkeypatch.setattr(vr, "_build_plan", _fake_build_plan)
    monkeypatch.setattr(vr, "_run_group0_planner", _fake_planner)
    monkeypatch.setattr(vr, "execute_vatra", _fake_execute)
    monkeypatch.setattr(vr, "_session_files_dir", lambda sid: tmp_path)
    monkeypatch.setattr(vr, "_vfs_manifest", lambda *a, **k: "")
    monkeypatch.setattr(vr.asyncio, "create_task", lambda c: (created.append(c), _T())[1])

    await vr._continue_run("u1", "parent01-run", {"id": "u1"}, instruction="More",
                           kind="continue", tiers=_TIERS, env_vars=None, api_key="")
    while created:
        await created.pop(0)
    # A run outside any project bundle plans exactly as before — no reference block.
    assert seen == ["Lead conventions."]
