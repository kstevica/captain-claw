"""Members' private sessions seen from the owner's side (A1 shared agent).

The owner's agent never adopts a member's session — not from the web
slash commands, the remote (Telegram/WhatsApp/Discord) handler or the local
console — and the owner's session lists, ``#N`` indices and name lookups
count only the owner's own sessions.
"""

from __future__ import annotations

import types

import pytest

from captain_claw.speaker import (
    MEMBER_SESSION_REFUSAL,
    list_owner_sessions,
    member_session_refusal,
    select_owner_session,
)

MEMBER_META = {"speaker_id": "u-ana", "speaker_lane": "A", "speaker_name": "Ana"}


@pytest.fixture
async def sm(tmp_path):
    from captain_claw.session import SessionManager

    manager = SessionManager(tmp_path / "sessions.db")
    yield manager
    await manager.close()


async def _seed(sm):
    """Owner sessions, then newer member sessions (one shares a name)."""
    owner_old = await sm.create_session(name="research")
    owner_new = await sm.create_session(name="owner-latest")
    members = []
    for i in range(3):
        m = await sm.create_session(name=f"spk-ana-{i}", metadata=dict(MEMBER_META))
        m.add_message("user", f"MEMBER PRIVATE {i}")
        await sm.save_session(m)
        members.append(m)
    # A member renamed theirs to the owner's name; it is now the newest.
    members[0].name = "research"
    await sm.save_session(members[0])
    return owner_old, owner_new, members


# ── the helpers ──────────────────────────────────────────────────────


async def test_owner_lists_and_indices_skip_member_sessions(sm):
    owner_old, owner_new, members = await _seed(sm)
    listed = await list_owner_sessions(sm, limit=20)
    assert [s.id for s in listed] == [owner_new.id, owner_old.id]
    assert (await select_owner_session(sm, "#1")).id == owner_new.id
    assert (await select_owner_session(sm, "2")).id == owner_old.id
    assert await select_owner_session(sm, "#3") is None
    # The plain store would have handed the owner a member's session here.
    assert (await sm.select_session("#1")).metadata.get("speaker_id") == "u-ana"


async def test_owner_name_lookup_prefers_the_owners_session(sm):
    owner_old, _, members = await _seed(sm)
    assert (await sm.load_session_by_name("research")).id == members[0].id
    assert (await select_owner_session(sm, "research")).id == owner_old.id


async def test_exact_ids_and_member_only_names_still_resolve_for_the_refusal(sm):
    _, _, members = await _seed(sm)
    by_id = await select_owner_session(sm, members[1].id)
    assert by_id.id == members[1].id
    assert member_session_refusal(by_id) == MEMBER_SESSION_REFUSAL
    by_name = await select_owner_session(sm, "spk-ana-2")
    assert member_session_refusal(by_name) == MEMBER_SESSION_REFUSAL


async def test_owner_lists_page_past_many_member_sessions(sm):
    owner = await sm.create_session(name="owner-only")
    for i in range(45):
        await sm.create_session(name=f"spk-{i}", metadata={"speaker_id": f"u-{i}"})
    assert [s.id for s in await list_owner_sessions(sm, limit=20)] == [owner.id]
    assert (await select_owner_session(sm, "#1")).id == owner.id


# ── remote (Telegram / WhatsApp / Discord) ───────────────────────────


class _UI:
    def __init__(self, command_result: str = ""):
        self.command_result = command_result
        self.errors: list[str] = []
        self.successes: list[str] = []

    def handle_special_command(self, raw_text):
        return self.command_result

    def print_error(self, text):
        self.errors.append(text)

    def print_success(self, text):
        self.successes.append(text)

    def load_monitor_tool_output_from_session(self, messages):
        return None

    def print_session_info(self, session):
        return None

    def print_session_list(self, sessions, current_session_id=None):
        self.listed = list(sessions)


def _owner_agent(sm, session):
    async def _set_last(_sid):
        return True

    sm.set_last_active_session = _set_last
    return types.SimpleNamespace(
        session=session, session_manager=sm,
        refresh_session_runtime_flags=lambda: None,
    )


async def _remote(ctx, raw):
    from captain_claw.remote_command_handler import handle_remote_command

    sent: list[str] = []

    async def _send(text):
        sent.append(text)

    async def _exec(prompt, label):
        raise AssertionError("no prompt should run")

    await handle_remote_command(
        ctx, platform="telegram", raw_text=raw, help_label="Telegram",
        sender_label="owner", send_text=_send, execute_prompt=_exec,
    )
    return sent


@pytest.mark.parametrize("how", ["id", "index", "name"])
async def test_remote_owner_cannot_switch_into_a_member_session(sm, how):
    owner_old, owner_new, members = await _seed(sm)
    start = types.SimpleNamespace(id="start", name="start", metadata={})
    agent = _owner_agent(sm, start)
    selector = {"id": members[1].id, "index": "#1", "name": "research"}[how]
    ctx = types.SimpleNamespace(agent=agent, ui=_UI(f"SESSION_SELECT:{selector}"))
    sent = await _remote(ctx, f"/session switch {selector}")
    if how == "id":
        assert sent == [MEMBER_SESSION_REFUSAL] and agent.session is start
    else:
        # `#1` and the shared name resolve to the owner's own sessions.
        assert agent.session.metadata.get("speaker_id") is None
        assert agent.session.id == (owner_new.id if how == "index" else owner_old.id)


async def test_remote_session_list_shows_only_owner_sessions(sm):
    owner_old, owner_new, _ = await _seed(sm)
    ctx = types.SimpleNamespace(agent=_owner_agent(sm, owner_new), ui=_UI("SESSIONS"))
    sent = await _remote(ctx, "/sessions")
    assert "MEMBER" not in sent[0] and "spk-ana" not in sent[0]
    assert "[1] owner-latest" in sent[0] and "[2] research" in sent[0]


# ── local console ────────────────────────────────────────────────────


async def test_local_owner_cannot_switch_into_a_member_session(sm):
    from captain_claw.local_command_dispatch import dispatch_local_command

    _, owner_new, members = await _seed(sm)
    ui = _UI()
    ctx = types.SimpleNamespace(agent=_owner_agent(sm, owner_new), ui=ui)
    out = await dispatch_local_command(ctx, f"SESSION_SELECT:{members[2].id}", "/session x")
    assert out == "continue"
    assert ui.errors == [MEMBER_SESSION_REFUSAL] and ctx.agent.session is owner_new


async def test_local_owner_cannot_run_a_prompt_in_a_member_session(sm):
    import json

    from captain_claw.local_command_dispatch import dispatch_local_command

    _, owner_new, members = await _seed(sm)
    ui = _UI()
    ctx = types.SimpleNamespace(agent=_owner_agent(sm, owner_new), ui=ui)
    payload = json.dumps({"selector": members[2].id, "prompt": "summarise"})
    out = await dispatch_local_command(ctx, f"SESSION_RUN:{payload}", "/session run x")
    assert out == "continue"
    assert ui.errors == [MEMBER_SESSION_REFUSAL] and ctx.agent.session is owner_new


async def test_local_session_list_shows_only_owner_sessions(sm):
    from captain_claw.local_command_dispatch import dispatch_local_command

    owner_old, owner_new, _ = await _seed(sm)
    ui = _UI()
    ctx = types.SimpleNamespace(agent=_owner_agent(sm, owner_new), ui=ui)
    await dispatch_local_command(ctx, "SESSIONS", "/sessions")
    assert [s.id for s in ui.listed] == [owner_new.id, owner_old.id]


# ── the owner agent's cross-session references ───────────────────────


@pytest.mark.parametrize("ref", ["#1", "research"])
async def test_owner_cross_session_reference_resolves_the_owners_session(sm, ref):
    from captain_claw.agent_context_mixin import AgentContextMixin

    owner_old, owner_new, members = await _seed(sm)
    for s, text in ((owner_old, "OWNER RESEARCH NOTES"), (owner_new, "OWNER LATEST NOTES"),
                    (members[0], "MEMBER PRIVATE ANSWER")):
        s.add_message("assistant", text)
        await sm.save_session(s)
    # members[0] (named "research") is the newest session again.
    members[0].add_message("assistant", "MEMBER PRIVATE ANSWER 2")
    await sm.save_session(members[0])
    agent = types.SimpleNamespace(
        session=types.SimpleNamespace(id="current"), session_manager=sm, memory=None,
        _extract_session_references=lambda q: [ref],
    )
    note = await AgentContextMixin._resolve_cross_session_context(agent, f"see {ref}")
    assert note and "MEMBER PRIVATE" not in note
    assert ("OWNER LATEST NOTES" if ref == "#1" else "OWNER RESEARCH NOTES") in note
