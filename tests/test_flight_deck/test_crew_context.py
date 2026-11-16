"""Crew context from a person's agents: files an agent made, and a range of what was said
with one, both handed to a worker like attached files. Only ever the person's own."""

from __future__ import annotations

import json
import os

import pytest

from captain_claw.flight_deck import crew_attach, crew_context, crew_store, server

from .test_crew_routes import _approve, _client, _settle, deck  # noqa: F401  (deck is a fixture)

AUTOMATED = "[Automated turn — Scheduled task. Not typed by the person.] Run the report"


@pytest.fixture
def agents(deck, monkeypatch):  # noqa: F811
    home = deck.tmp / "fd-data"
    home.mkdir(exist_ok=True)
    registry = home / ".processes.json"
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", registry)

    def make(slug: str, owner: str) -> object:
        workspace = home / slug / "data" / "workspace"
        workspace.mkdir(parents=True)
        current = json.loads(registry.read_text()) if registry.exists() else {}
        registry.write_text(json.dumps({**current, slug: {"slug": slug, "name": slug.title(), "owner": owner}}))
        return workspace

    return make


# ── Files an agent made ──


async def test_only_their_own_agents_and_an_ownerless_one_only_for_an_admin(deck, agents):  # noqa: F811
    agents("mine", deck.user["id"])
    agents("theirs", deck.other["id"])
    agents("old", "")
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        listed = (await client.get("/fd/crew/agents")).json()["agents"]
        assert [a["id"] for a in listed] == ["mine"]
        assert (await client.get("/fd/crew/agents/theirs/files")).status_code == 404
        assert (await client.get("/fd/crew/agents/old/files")).status_code == 404
        refused = await client.post("/fd/crew/uploads/agent-files", json={"agent": "theirs", "paths": ["a.txt"]})
        assert refused.status_code == 404
    async with _client(deck.admin) as admin:
        assert [a["id"] for a in (await admin.get("/fd/crew/agents")).json()["agents"]] == ["old"]


async def test_it_takes_the_crew_approval(deck, agents):  # noqa: F811
    agents("mine", deck.user["id"])
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/agents")).status_code == 403
        assert (await client.post("/fd/crew/uploads/agent-files", json={"agent": "mine", "paths": ["a"]})).status_code == 403
        assert (await client.get("/fd/crew/conversations")).status_code == 403


async def test_the_list_leaves_out_hidden_heavy_and_linked_things_newest_first(deck, agents):  # noqa: F811
    ws = agents("mine", deck.user["id"])
    (ws / "saved").mkdir()
    (ws / "saved" / "report.md").write_text("old")
    (ws / "notes.txt").write_text("new")
    os.utime(ws / "saved" / "report.md", (1, 1))
    for hidden in (".git", "node_modules", ".code"):
        (ws / hidden).mkdir()
        (ws / hidden / "x.txt").write_text("x")
    (ws / ".env").write_text("KEY=1")
    outside = deck.tmp / "outside.txt"
    outside.write_text("not the agent's")
    (ws / "link.txt").symlink_to(outside)
    (ws / "linked").symlink_to(deck.tmp, target_is_directory=True)
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        got = (await client.get("/fd/crew/agents/mine/files")).json()
    assert [f["path"] for f in got["files"]] == ["notes.txt", "saved/report.md"]
    assert got["more"] is False


async def test_chosen_files_go_with_a_task_like_any_attachment(deck, agents):  # noqa: F811
    ws = agents("mine", deck.user["id"])
    (ws / "saved").mkdir()
    (ws / "saved" / "design.md").write_text("the plan")
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        up = await client.post("/fd/crew/uploads/agent-files", json={"agent": "mine", "paths": ["saved/design.md"]})
        assert up.status_code == 201, up.text
        [meta] = up.json()["files"]
        assert meta["name"] == "design.md" and meta["size"] == 8 and meta["type"] == "text/markdown"
        made = await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "p", "attachments": [meta["id"]]})
        assert made.status_code == 201, made.text
    held = crew_store.task_dir(deck.user["id"], made.json()["id"]) / "attachments" / "design.md"
    assert held.read_text() == "the plan"
    assert (ws / "saved" / "design.md").read_text() == "the plan", "a copy: the agent keeps its own"
    await _settle(deck.user["id"], made.json()["id"])


async def test_nothing_outside_the_workspace_and_nothing_that_looks_like_keys(deck, agents):  # noqa: F811
    ws = agents("mine", deck.user["id"])
    (ws / "ok.txt").write_text("fine")
    (deck.tmp / "fd-data" / "mine" / "config.yaml").write_text("secret: 1")
    outside = deck.tmp / "outside.txt"
    outside.write_text("not the agent's")
    (ws / "link.txt").symlink_to(outside)
    (ws / ".env").write_text("KEY=1")
    (ws / "id_ed25519").write_text("key")
    (ws / "data.db").write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    (ws / ".ssh").mkdir()
    (ws / ".ssh" / "notes.txt").write_text("x")
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        for bad, code in (("../config.yaml", 404), ("../../.processes.json", 404), ("link.txt", 404), ("", 400),
                          (".env", 400), ("id_ed25519", 400), ("data.db", 400), (".ssh/notes.txt", 400),
                          ("missing.txt", 404), (".", 404)):
            resp = await client.post("/fd/crew/uploads/agent-files", json={"agent": "mine", "paths": [bad]})
            assert resp.status_code == code, (bad, resp.text)
        both = await client.post("/fd/crew/uploads/agent-files", json={"agent": "mine", "paths": ["ok.txt", ".env"]})
        assert both.status_code == 400 and ".env" in both.json()["detail"]
    assert not list((crew_store.user_dir(deck.user["id"]) / "uploads").glob("*")), "all or none"


# ── What was said with an agent ──


def _row(role: str, content: str, **meta) -> dict:
    return {"role": role, "content": content, "metadata": json.dumps(meta)}


def _divider(sid: str, title: str = "") -> dict:
    return _row("system", "", topic={"action": "new", "agentSessionId": sid, "title": title})


async def _chat(db, user: dict, lane: str, rows: list[dict], agent: str = "Researcher") -> list[int]:
    await db.upsert_chat_session(lane, user["id"], agent_id=lane.split("::")[0], agent_name=agent)
    return await db.add_chat_messages(lane, user["id"], rows)


async def test_only_the_persons_words_and_the_agents_final_answers(deck):  # noqa: F811
    ids = await _chat(deck.db, deck.user, "proc-researcher", [
        _divider("s1", "Market sizing"),
        _row("user", "How big is the market?"),
        _row("system", "Looking it up…", narration=True),
        _row("tool", "web_search", tool_name="web_search"),
        _row("assistant", "About 2 billion."),
        _row("user", "/clear"),
        _row("user", AUTOMATED),
        _row("assistant", "Daily report: nothing new.", automation="Scheduled task"),
        _row("assistant", "   "),
        _divider("s2"),
        _row("user", "And next year?"),
        _row("assistant", "Growing 10%."),
    ])
    await _chat(deck.db, deck.user, "proc-researcher::AUTO", [_row("assistant", "auto")])
    await _chat(deck.db, deck.other, "proc-other", [_row("user", "their secret")])
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        lanes = (await client.get("/fd/crew/conversations")).json()
        assert [c["id"] for c in lanes["conversations"]] == ["proc-researcher"], "no automation lane, nobody else's"
        assert lanes["conversations"][0]["agent"] == "Researcher"
        sessions = (await client.get("/fd/crew/conversations/sessions", params={"lane": "proc-researcher"})).json()["sessions"]
        assert [s["id"] for s in sessions] == [ids[9], ids[0]], "newest first"
        assert sessions[0]["title"] == "And next year?" and sessions[1]["title"] == "Market sizing"
        assert [s["count"] for s in sessions] == [2, 2], "counted as they'd be sent"
        got = (await client.get("/fd/crew/conversations/messages", params={"lane": "proc-researcher", "session": ids[0]})).json()
        assert [(m["role"], m["text"]) for m in got["messages"]] == [("user", "How big is the market?"), ("assistant", "About 2 billion.")]

        up = await client.post("/fd/crew/uploads/conversation", json={"lane": "proc-researcher", "session": ids[0], "first": ids[1], "last": ids[11]})
        assert up.status_code == 201, up.text
        meta = up.json()
        assert meta["name"].startswith("conversation-Researcher-") and meta["name"].endswith(".md")
        assert meta["type"] == "text/markdown"
    [text] = [p.read_text() for p in (crew_store.user_dir(deck.user["id"]) / "uploads" / meta["id"]).iterdir() if p.suffix == ".md"]
    assert "# Conversation with Researcher: Market sizing" in text
    assert "How big is the market?" in text and "About 2 billion." in text
    for left_out in ("Looking it up", "web_search", "/clear", "Run the report", "Daily report", "And next year?", "Growing"):
        assert left_out not in text, f"{left_out!r}: outside the session, or not a message or a final answer"
    assert text.count("**User**") == 1 and text.count("**Researcher**") == 1


async def test_a_range_within_a_session_and_an_old_log_without_dividers(deck):  # noqa: F811
    ids = await _chat(deck.db, deck.user, "proc-old", [
        _row("user", "one"), _row("assistant", "two"), _row("user", "three"), _row("assistant", "four"),
    ], agent="Old helper")
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        sessions = (await client.get("/fd/crew/conversations/sessions", params={"lane": "proc-old"})).json()["sessions"]
        assert [(s["id"], s["title"], s["count"]) for s in sessions] == [(0, "The whole conversation", 4)]
        up = await client.post("/fd/crew/uploads/conversation", json={"lane": "proc-old", "session": 0, "first": ids[2], "last": ids[1]})
        assert up.status_code == 201, up.text
        empty = await client.post("/fd/crew/uploads/conversation", json={"lane": "proc-old", "session": 0, "first": 10**6, "last": 10**6 + 1})
        assert empty.status_code == 400
    meta = up.json()
    text = (crew_store.user_dir(deck.user["id"]) / "uploads" / meta["id"] / meta["name"]).read_text()
    assert "two" in text and "three" in text and "one" not in text and "four" not in text
    assert meta["name"].startswith("conversation-Old-helper-")


async def test_another_persons_conversation_is_never_reached(deck):  # noqa: F811
    ids = await _chat(deck.db, deck.other, "proc-theirs", [_row("user", "their secret"), _row("assistant", "ok")])
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/conversations/sessions", params={"lane": "proc-theirs"})).status_code == 404
        assert (await client.get("/fd/crew/conversations/messages", params={"lane": "proc-theirs"})).status_code == 404
        resp = await client.post("/fd/crew/uploads/conversation", json={"lane": "proc-theirs", "session": 0, "first": ids[0], "last": ids[1]})
        assert resp.status_code == 404


async def test_a_shared_agents_chat_is_the_members_own(deck):  # noqa: F811
    from captain_claw.flight_deck.chat_routes import _store_id

    lane = "shared:abc123"
    await deck.db.upsert_chat_session(_store_id(lane, deck.user["id"]), deck.user["id"], agent_id=lane, agent_name="Team bot")
    await deck.db.add_chat_messages(_store_id(lane, deck.user["id"]), deck.user["id"], [_row("user", "hi"), _row("assistant", "hello")])
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        lanes = (await client.get("/fd/crew/conversations")).json()["conversations"]
        assert [c["id"] for c in lanes] == [lane], "under the id the page knows, not the stored one"
        got = (await client.get("/fd/crew/conversations/messages", params={"lane": lane, "session": 0})).json()
        assert [m["text"] for m in got["messages"]] == ["hi", "hello"]


def test_the_write_out_names_who_and_when():
    text = crew_context.write_out("Helper", "", [
        {"role": "user", "content": "q", "created_at": "2026-11-16T10:42:00+00:00"},
        {"role": "assistant", "content": "a", "created_at": "2026-11-16T10:43:00+00:00"},
    ])
    assert text.startswith("# Conversation with Helper\n")
    assert "**User** · 2026-11-16 10:42 UTC" in text and "**Helper** · 2026-11-16 10:43 UTC" in text
    assert "2 messages from 2026-11-16 10:42 UTC to 2026-11-16 10:43 UTC" in text


def test_staged_text_is_refused_when_too_big(deck, monkeypatch):  # noqa: F811
    from fastapi import HTTPException

    monkeypatch.setattr(crew_attach, "MAX_FILE_BYTES", 10)
    with pytest.raises(HTTPException) as caught:
        crew_attach.stage_text(deck.user["id"], "big.md", "x" * 11)
    assert caught.value.status_code == 413 and "fewer messages" in caught.value.detail
