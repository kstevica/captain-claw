""""Why did this happen?" on a tool-call register row: the deck's deciders re-run
on the call's recorded facts, one verdict per layer, and a layer that would
decide differently today says so. No model call."""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck import approvals, auth

from .test_tool_audit_routes import ADMIN, OTHER, OWNER, _calls, _rec, _send, env  # noqa: F401


def _why(client, call_id, path="/fd/tool-audit/calls"):
    r = client.get(f"{path}/{call_id}/why")
    assert r.status_code == 200, r.text
    return r.json()


def _layer(why, key):
    return next(layer for layer in why["layers"] if layer["layer"] == key)


def _policy(**modes):
    asyncio.run(approvals.save_policy(auth.get_db(), modes))


def setup_function():
    approvals.reset_for_tests()


def test_a_plain_call_ran_and_nothing_asks_about_it(env):
    client, *_ = env
    _send(client, _rec(1, tool="read", params={"path": "notes.md"}))
    (c,) = _calls(client)
    why = _why(client, c["id"])
    assert why["summary"] == "It ran: nothing on this deck asks about this kind of call."
    assert [layer["layer"] for layer in why["layers"]] == ["what", "who", "approval", "guard", "outcome"]
    assert _layer(why, "approval")["then"].startswith("Not a call that needs approval")
    assert _layer(why, "who")["then"] == "A turn you started (Web chat)."
    assert _layer(why, "outcome")["then"].startswith("It ran (took")
    assert why["changed"] is False


def test_a_send_that_ran_unasked_would_be_refused_today(env):
    client, *_ = env
    _send(client, _rec(1, tool="google_mail", params={"action": "send", "to": "a@x.test"}))
    (c,) = _calls(client)
    assert _layer(_why(client, c["id"]), "approval")["changed"] is False
    _policy(send="people_only")
    why = _why(client, c["id"])
    layer = _layer(why, "approval")
    assert layer["title"] == "Approval · send a message to someone"
    assert layer["then"] == "No approval was recorded for it."
    assert layer["today"].startswith("Today it would be refused: On this deck agents don't send a message")
    assert layer["changed"] is True and why["changed"] is True
    assert why["summary"].startswith("It ran without asking")


def test_a_refused_send_would_go_ahead_once_the_deck_stops_asking(env):
    client, *_ = env
    _send(client, _rec(1, tool="send_mail", params={"to": "a@x.test"}, outcome="blocked",
                       reason="Approval (send): Owen didn't allow it."))
    (c,) = _calls(client)
    why = _why(client, c["id"])
    assert why["summary"] == "It didn't run: it needed approval and didn't get it."
    layer = _layer(why, "approval")
    assert layer["then"] == "Owen didn't allow it."
    assert layer["today"].startswith("Today it would go ahead without asking")
    assert layer["changed"] is True
    # Asking again today depends on the person: not "different".
    _policy(send="ask")
    layer = _layer(_why(client, c["id"]), "approval")
    assert layer["today"].startswith("Today it would ask first") and layer["changed"] is False


def test_a_scheduled_run_with_nobody_there(env):
    client, *_ = env
    _send(client, _rec(1, tool="send_mail", params={"to": "a@x.test"}, source="scheduled",
                       source_label="Scheduled tasks"))
    (c,) = _calls(client)
    _policy(send="ask")
    why = _why(client, c["id"])
    assert _layer(why, "who")["then"].startswith("It started on its own: Scheduled tasks")
    assert _layer(why, "approval")["today"].startswith("Today it would be refused: Nobody was there")
    assert why["summary"].startswith("It ran on its own")
    # The person always allows it for this agent: it goes ahead today too.
    db = auth.get_db()

    async def owner_row():
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'x', ?, 'user', '2026-01-01', '2026-01-01')",
            (OWNER["id"], OWNER["email"], OWNER["display_name"]))
        await db._db.commit()
        await approvals.set_always(db, OWNER["id"], "helper", "send", True)

    asyncio.run(owner_row())
    layer = _layer(_why(client, c["id"]), "approval")
    assert layer["today"].startswith("Today it would go ahead: Owen always allows") and not layer["changed"]


def test_an_allowed_call_explains_who_allowed_it(env):
    client, *_ = env
    _send(client, _rec(1, tool="send_mail", params={"to": "a@x.test"},
                       reason="Approval (send): Owen allowed it once."))
    (c,) = _calls(client)
    why = _why(client, c["id"])
    assert why["summary"] == "It ran because it was allowed."
    assert _layer(why, "approval")["then"] == "Owen allowed it once."


def test_the_guard_layer(env):
    client, *_ = env
    _send(client, _rec(1, tool="shell", params={"command": "rm -rf /srv/data"}, outcome="blocked",
                       reason="Blast radius: destructive command (approval denied)"))
    _send(client, _rec(2, tool="shell", params={"command": "ls -la"}))
    stopped, plain = sorted(_calls(client), key=lambda c: c["seq"])
    why = _why(client, stopped["id"])
    assert why["summary"] == "It didn't run: the agent's guard stopped it."
    guard = _layer(why, "guard")
    assert guard["then"].startswith("The agent's guard stopped it: Blast radius")
    assert guard["today"].startswith("Today's blast-radius rules match it") and guard["changed"] is False
    assert _layer(_why(client, plain["id"]), "guard")["today"] == "Today's blast-radius rules don't match it."


def test_only_calls_you_may_see_and_admins_see_all(env):
    client, current, *_ = env
    _send(client, _rec(1, tool="read", params={"path": "x"}))
    (c,) = _calls(client)
    current["user"] = OTHER
    assert client.get(f"/fd/tool-audit/calls/{c['id']}/why").status_code == 404
    assert client.get("/fd/tool-audit/calls/9999/why").status_code == 404
    current["user"] = ADMIN  # the admin routes live on the same router
    assert _why(client, c["id"], "/fd/admin/tool-audit/calls")["id"] == c["id"]
    current["user"] = OWNER


def test_a_members_hidden_call_is_judged_by_its_tool_alone(env):
    client, current, store, members = env
    from .test_tool_audit_routes import MEMBER

    r = client.post("/fd/tool-audit/agent/calls",
                    json={"records": [_rec(1, tool="google_mail", params={"action": "send", "to": "m@x.test"},
                                           source="member", source_label="Shared-agent members",
                                           speaker_id=MEMBER["id"], speaker_name="Mia",
                                           speaker_ref="process:helper:0123456789abcdef")]},
                    headers={"X-Agent-Auth": "tok-A"})
    assert r.status_code == 200, r.text
    (c,) = _calls(client)
    assert c["hidden"] is True
    why = _why(client, c["id"])
    assert "judges the tool alone" in _layer(why, "approval")["then"]
    assert "a@x" not in str(why) and "m@x.test" not in str(why)
