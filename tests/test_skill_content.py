"""An agent's /api/skills/content: what a skill's SKILL.md says, and a
rewrite of one it keeps itself (its owner's setup's, or its own). The file
is always the one the agent loaded under that name; a built-in skill is
read-only, and a rewrite keeps the skill's name — so it stays the same skill,
on or off as it was. The skill list says which came from the owner's setup."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from captain_claw import config as config_mod
from captain_claw.web import rest_skills

SETUP_SKILL = """---
name: weekly-sales-report
description: "When the owner asks for the weekly sales report"
---

# Weekly sales report

## Steps
1. Pull this week's sales.
2. Send it to the sales director.

Source: handbook.pdf — Reports (from your owner, when they set you up)
"""

OWN_SKILL = """---
name: tidy-notes
description: Tidy meeting notes
---

Group the notes by topic.
"""


class _Req:
    def __init__(self, query=None, body=None):
        self.query = query or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


@pytest.fixture
def agent(tmp_path: Path, monkeypatch):
    managed = tmp_path / "managed"
    for name, text in (("weekly-sales-report", SETUP_SKILL), ("tidy-notes", OWN_SKILL)):
        (managed / name).mkdir(parents=True)
        (managed / name / "SKILL.md").write_text(text)
    bundled = tmp_path / "bundled" / "web-research"
    bundled.mkdir(parents=True)
    (bundled / "SKILL.md").write_text("---\nname: web-research\ndescription: Research the web\n---\nSearch.\n")
    cfg = config_mod.get_config().model_copy(deep=True)
    cfg.skills.managed_dir = str(managed)
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    monkeypatch.setattr("captain_claw.skills._resolve_bundled_skills_dir", lambda: tmp_path / "bundled")
    (tmp_path / "ws").mkdir()
    return types.SimpleNamespace(server=types.SimpleNamespace(agent=types.SimpleNamespace(workspace_base_path=tmp_path / "ws")),
                                 managed=managed)


async def _call(handler, server, req):
    resp = await handler(server, req)
    return resp.status, json.loads(resp.text)


async def test_the_list_says_which_skills_came_from_the_owners_setup(agent):
    status, body = await _call(rest_skills.list_skills, agent.server, _Req())
    by = {s["name"]: s for s in body["skills"]}
    assert by["weekly-sales-report"]["from_owner"] is True and by["weekly-sales-report"]["editable"] is True
    assert by["tidy-notes"]["from_owner"] is False and by["tidy-notes"]["editable"] is True
    assert by["web-research"]["editable"] is False and by["web-research"]["from_owner"] is False


async def test_a_skill_reads_as_its_file(agent):
    status, body = await _call(rest_skills.skill_content, agent.server, _Req({"name": "weekly-sales-report"}))
    assert status == 200 and body["content"] == SETUP_SKILL
    assert body["editable"] is True and body["from_owner"] is True and body["source"] == "captain-managed"
    status, body = await _call(rest_skills.skill_content, agent.server, _Req({"name": "../../etc/passwd"}))
    assert status == 404


async def test_the_owner_corrects_one_it_keeps(agent):
    fixed = SETUP_SKILL.replace("2. Send it to the sales director.", "2. Send it to the sales director by 10:00.")
    status, body = await _call(rest_skills.save_skill_content, agent.server,
                               _Req(body={"name": "weekly-sales-report", "content": fixed}))
    assert status == 200 and body["ok"] is True
    assert (agent.managed / "weekly-sales-report" / "SKILL.md").read_text() == fixed
    assert not list(agent.managed.glob("weekly-sales-report/.*.tmp"))


async def test_a_rewrite_keeps_the_skills_name(agent):
    renamed = OWN_SKILL.replace("name: tidy-notes", "name: something-else")
    status, body = await _call(rest_skills.save_skill_content, agent.server,
                               _Req(body={"name": "tidy-notes", "content": renamed}))
    assert status == 400 and "name: tidy-notes" in body["error"]
    no_frontmatter = "Group the notes by topic, then by date.\n"     # the folder's name stands in: still tidy-notes
    status, _ = await _call(rest_skills.save_skill_content, agent.server,
                            _Req(body={"name": "tidy-notes", "content": no_frontmatter}))
    assert status == 200
    for bad in ("", "   ", "x" * 100_001):
        status, _ = await _call(rest_skills.save_skill_content, agent.server,
                                _Req(body={"name": "tidy-notes", "content": bad}))
        assert status == 400


async def test_a_built_in_skill_is_read_only(agent):
    status, body = await _call(rest_skills.skill_content, agent.server, _Req({"name": "web-research"}))
    assert status == 200 and body["editable"] is False
    status, body = await _call(rest_skills.save_skill_content, agent.server,
                               _Req(body={"name": "web-research", "content": "---\nname: web-research\n---\nHacked.\n"}))
    assert status == 403
    status, _ = await _call(rest_skills.save_skill_content, agent.server,
                            _Req(body={"name": "nope", "content": "x"}))
    assert status == 404
