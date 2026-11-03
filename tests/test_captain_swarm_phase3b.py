"""Captain Swarm, Phase 3b: the rehearsal guard, intents and the register check,
Bat in rehearsal, one task's best outcome with Captain's own checks, and the
winner carried out once (docs/captain-swarm-plan.md)."""

from __future__ import annotations

import asyncio
import json

import pytest

from captain_claw import rehearsal_turn as rt
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_outcome as outcome
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

TAG = "cs_0123456789ab:exp_abcdef123456"


def run(coro):
    return asyncio.run(coro)


# ── the guard ────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool,args,verdict", [
    ("web_search", {"query": "x"}, "allow"),
    ("web_fetch", {"url": "https://example.com"}, "allow"),
    ("read", {"path": "a.txt"}, "allow"),
    ("write", {"path": "notes.md"}, "allow"),
    ("google_mail", {"action": "search"}, "allow"),
    ("google_mail", {"action": "send", "to": "a@b.c"}, "intent"),
    ("google_mail", {"action": "create_draft"}, "intent"),
    ("google_calendar", {"action": "create_event"}, "intent"),
    ("google_drive", {"action": "upload"}, "intent"),
    ("whatsapp_send_file", {}, "intent"),
    ("consult_peer", {}, "intent"),
    ("payments", {"action": "pay"}, "intent"),
    ("spend", {"action": "authorize"}, "allow"),
    ("spend", {"action": "settle"}, "intent"),
    ("mcp_stripe_create_charge", {}, "intent"),
    ("mcp_notion_search_pages", {}, "allow"),
    ("brand_new_tool", {}, "intent"),
    ("insights", {"action": "add"}, "refuse"),
    ("insights", {"action": "search"}, "allow"),
    ("shell", {"command": "ls"}, "refuse"),
    ("desktop_action", {}, "refuse"),
    ("bat", {"action": "start"}, "refuse"),
    ("captain_swarm", {"action": "start"}, "refuse"),
    ("ask_human", {}, "allow"),
    ("line", {"action": "news"}, "allow"),
])
def test_classify(tool, args, verdict):
    assert rt.classify(tool, args)[0] == verdict


@pytest.mark.parametrize("args,verdict", [
    ({"action": "navigate", "url": "https://shop.example"}, "allow"),
    ({"action": "type", "selector": "input[name=q]", "text": "the book"}, "allow"),
    ({"action": "act", "goal": "find the cheapest offer"}, "allow"),
    ({"action": "click", "text": "Next page", "role": "link"}, "allow"),
    ({"action": "click", "text": "Add to cart"}, "allow"),
    ({"action": "click", "text": "Place order"}, "commit"),
    ({"action": "click", "text": "Proceed to Checkout"}, "commit"),
    ({"action": "click", "text": "Pay now"}, "commit"),
    ({"action": "click", "text": "Confirm booking"}, "commit"),
    ({"action": "click", "selector": "button[type=submit]"}, "commit"),
    ({"action": "click", "selector": ".btn-primary"}, "commit"),          # a button it can't read
    ({"action": "click", "selector": "a.product-link"}, "allow"),
    ({"action": "press_key", "key": "Enter"}, "commit"),
    ({"action": "press_key", "key": "Tab"}, "allow"),
    ({"action": "api_replay", "method": "GET", "endpoint": "/api/x"}, "allow"),
    ({"action": "api_replay", "method": "POST", "endpoint": "/api/order"}, "intent"),
    ({"action": "workflow_run", "workflow_id": "w1"}, "intent"),
    ({"action": "credentials_store"}, "refuse"),
])
def test_browser_commit_points(args, verdict):
    assert rt.classify("browser", args)[0] == verdict


def test_tool_block_records_and_says_not_done(monkeypatch):
    sent = []

    async def fake_record(tag, kind, name, arguments, what, agent):
        sent.append((tag, kind, name, rt.safe_args(arguments), what))
        return True
    monkeypatch.setattr(rt, "_record", fake_record)
    monkeypatch.setenv(rt.ENV, TAG)
    assert run(rt.tool_block("web_search", {"query": "x"}, None)) is None
    msg = run(rt.tool_block("google_mail", {"action": "send", "to": "a@b.c", "password": "x", "_agent": object()}, None))
    assert msg.startswith("Recorded as an intent, not done: google_mail send")
    assert sent[0][0] == TAG and sent[0][1] == "intent" and "password" not in sent[0][3] and "_agent" not in sent[0][3]
    msg = run(rt.tool_block("browser", {"action": "click", "text": "Pay now"}, None))
    assert msg.startswith("Stopped at the commit point") and sent[1][1] == "commit"
    msg = run(rt.tool_block("shell", {"command": "curl -X POST x"}, None))
    assert "isn't available in a rehearsal" in msg and len(sent) == 2
    monkeypatch.setenv(rt.ENV, "not-a-tag")
    assert run(rt.tool_block("google_mail", {"action": "send"}, None)) is None   # no rehearsal at all


def test_the_registry_blocks_a_rehearsal_workers_calls(monkeypatch, tmp_path):
    """The guard sits in the tool registry's own chain: the tool never runs."""
    from captain_claw.tools.registry import Tool, ToolBlockedError, ToolRegistry, ToolResult

    ran = []

    class Mail(Tool):
        name = "google_mail"
        description = "mail"
        parameters = {"type": "object", "properties": {"action": {"type": "string"}}}

        async def execute(self, **kwargs):
            ran.append(kwargs.get("action"))
            return ToolResult(success=True, content="sent")

    async def fake_record(*a):
        return True
    monkeypatch.setattr(rt, "_record", fake_record)
    reg = ToolRegistry(base_path=tmp_path)
    reg.register(Mail())
    monkeypatch.setenv(rt.ENV, TAG)

    async def go():
        try:
            got = await reg.execute("google_mail", {"action": "send"})
        except ToolBlockedError as exc:
            got = str(exc)
        searched = await reg.execute("google_mail", {"action": "search"})
        return got, searched
    blocked, searched = run(go())
    assert "Recorded as an intent" in str(getattr(blocked, "error", blocked) or blocked)
    assert ran == ["search"] and searched.success


# ── intents and the register ─────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    yield s
    loop.set_store(None)
    run(s.close())


def test_record_intent_only_for_the_owners_running_attempt(store):
    async def go():
        await store.create_run(run_id="cs_0123456789ab", owner_id="u", title="t", objective={}, status="running")
        await store.add_experiment(exp_id="exp_abcdef123456", run_id="cs_0123456789ab", round_no=1, n=1, hypothesis="h")
        await store.update_experiment("exp_abcdef123456", status="running")
        ok = await rehearsal.record(store, "u", tag=TAG, kind="commit", tool="browser", what="click 'Pay'", args={"text": "Pay"})
        other = await rehearsal.record(store, "intruder", tag=TAG, kind="intent", tool="x", what="", args={})
        bad = await rehearsal.record(store, "u", tag="cs_x:exp_y", kind="intent", tool="x", what="", args={})
        await store.update_experiment("exp_abcdef123456", status="kept")
        late = await rehearsal.record(store, "u", tag=TAG, kind="intent", tool="x", what="", args={})
        return ok, other, bad, late, await store.list_intents("cs_0123456789ab", "exp_abcdef123456")
    ok, other, bad, late, intents = run(go())
    assert ok["ok"] and not other["ok"] and not bad["ok"] and not late["ok"]
    assert [(i["kind"], i["tool"], i["what"]) for i in intents] == [("commit", "browser", "click 'Pay'")]


def test_register_rows_judged_with_the_guards_rules():
    rows = [
        {"agent_slug": "bat-bat-1a2b-main", "tool": "web_search", "params": "{}", "outcome": "ok"},
        {"agent_slug": "bat-bat-1a2b-main", "tool": "google_mail", "params": json.dumps({"action": "send"}), "outcome": "blocked"},
        {"agent_slug": "bat-bat-1a2b-main", "tool": "google_mail", "params": json.dumps({"action": "send"}), "outcome": "ok"},
        {"agent_slug": "bat-bat-1a2b-main", "tool": "shell", "params": "{}", "outcome": "ok"},
    ]
    found = rehearsal.judge_rows(rows)
    assert [(f["tool"], f["verdict"]) for f in found] == [("google_mail", "intent"), ("shell", "refuse")]
    assert outcome.worker_prefix("bat_1a2b3c4d5e6f") == "bat-bat-1a2b-"


def test_leaks_reads_the_register(tmp_path, monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    from captain_claw.flight_deck.tool_audit_store import ToolAuditStore

    class FakeAudit:
        async def iterate(self, filters):
            assert filters["since"] == 100.0
            for r in [{"agent_slug": "bat-bat-1a2b-s1", "tool": "google_mail", "params": '{"action": "send"}', "outcome": "ok"},
                      {"agent_slug": "someone-else", "tool": "google_mail", "params": '{"action": "send"}', "outcome": "ok"}]:
                yield r
    monkeypatch.setattr(tool_audit_routes, "store", lambda: FakeAudit())
    found = run(rehearsal.leaks("bat-bat-1a2b-", since=100.0))
    assert [f["agent"] for f in found] == ["bat-bat-1a2b-s1"]
    assert ToolAuditStore  # the real store's iterate yields rows, like the fake


# ── Bat in rehearsal ─────────────────────────────────────────────────


def test_bat_rehearsal_skips_the_gate_and_the_delivery(monkeypatch):
    from captain_claw.flight_deck import bat_routes

    class FakeStore:
        async def list_steps(self, run_id):
            return [{"title": "send the email to the client", "status": "pending"}]
    plain = {"id": "bat_1", "status": "planning", "task": "Email the client the offer", "config": {}, "owner_id": "u"}
    rehearse = {**plain, "config": {"rehearsal": TAG}}
    called = []

    async def raise_ask(*a, **k):
        called.append(1)

    async def set_status(*a, **k):
        pass

    async def latest_open_ask(*a):
        return None

    async def set_config(*a):
        pass

    async def append_event(*a, **k):
        pass
    st = FakeStore()
    st.set_status, st.latest_open_ask, st.set_config, st.append_event = set_status, latest_open_ask, set_config, append_event
    monkeypatch.setattr("captain_claw.flight_deck.human_ask.raise_ask", raise_ask)
    assert run(bat_routes._gate_check(st, rehearse)) == "proceed" and not called
    assert run(bat_routes._gate_check(st, plain)) == "wait" and called


# ── Captain's own checks ─────────────────────────────────────────────


def test_parse_candidates():
    text = ('I found these.\n```json\n{"candidates": [{"title": "The Book (paperback)", "url": "https://shop.example/b", '
            '"value": "1,240.50", "currency": "EUR"}, {"title": "no url", "value": 3}, '
            '{"title": "local", "url": "file:///etc/passwd", "value": 1}]}\n```')
    got = outcome.parse_candidates(text)
    assert got == [{"title": "The Book (paperback)", "url": "https://shop.example/b", "value": 1240.5, "currency": "EUR",
                    "details": "", "next_step": ""}]
    assert outcome.parse_candidates("nothing here") == []


def test_number_forms_and_verify():
    assert {"1,240.50", "1.240,50", "1240.50", "1240,50"} <= set(outcome.number_forms(1240.5))
    assert {"412", "412.00", "412,00"} <= set(outcome.number_forms(412.0))
    cons = {"max_value": 500, "must_include": ["paperback"], "currency": "EUR"}
    page = "The Pragmatic Programmer, 20th Anniversary Edition — Paperback — Price: 41,20 € — In stock"
    ok, why = outcome.verify({"title": "The Pragmatic Programmer 20th Anniversary", "value": 41.2}, page, cons, "min")
    assert ok, why
    assert outcome.verify({"title": "The Pragmatic Programmer", "value": 39.0}, page, cons, "min") == \
        (False, "the page doesn't show 39")
    assert not outcome.verify({"title": "A Different Book Entirely", "value": 41.2}, page, cons, "min")[0]
    assert not outcome.verify({"title": "The Pragmatic Programmer", "value": 41.2}, page,
                              {**cons, "must_include": ["hardcover"]}, "min")[0]
    assert not outcome.verify({"title": "The Pragmatic Programmer", "value": 41.2}, page,
                              {**cons, "max_value": 40}, "min")[0]
    assert not outcome.verify({"title": "x", "value": 1}, "", cons, "min")[0]
    # 41.20 must not match inside 141.20
    assert not outcome.verify({"title": "The Pragmatic Programmer", "value": 41.2},
                              "The Pragmatic Programmer paperback 141,20", cons, "min")[0]


def test_page_text_keeps_prices_in_attributes():
    raw = '<html><script>var x=1</script><meta itemprop="price" content="41.20"><h1>Book &amp; more</h1></html>'
    text = outcome.page_text(raw)
    assert "41.20" in text and "Book & more" in text and "var x" not in text


@pytest.mark.parametrize("host,ok", [("127.0.0.1", False), ("10.0.0.5", False), ("169.254.169.254", False),
                                     ("::1", False), ("8.8.8.8", True), ("192.168.1.1", False)])
def test_only_public_addresses(host, ok):
    assert outcome._public(host) is ok


def test_fetch_refuses_private_and_local_addresses():
    for url in ("http://127.0.0.1:25080/fd/admin/provider-keys", "http://localhost/x", "file:///etc/passwd",
                "http://169.254.169.254/latest/meta-data"):
        text, why = run(outcome.fetch_text(url))
        assert text == "" and "not a public web address" in why


# ── the loop, end to end, on one task ────────────────────────────────


class FakeBatLane:
    def __init__(self, store, found):
        self.store, self.found, self.started = store, found, []

    async def start(self, run_, exp, brief):
        self.started.append(exp["id"])
        return {"child_id": f"bat_{len(self.started):012d}", "branch": ""}

    async def poll(self, run_, exp):
        cands = self.found.pop(0)
        if not cands:
            return {"state": "failed", "note": "nothing to check"}
        vid = f"v_fake{len(self.started)}"
        await self.store.add_version(version_id=vid, run_id=run_["id"], content={"candidates": cands})
        return {"state": "done", "head_sha": vid, "commits": len(cands), "usd": 0.5, "tokens": 1000}

    async def stop(self, run_, exp):
        pass

    async def discard(self, run_, exp):
        pass


def test_driver_keeps_the_best_verified_result(store, monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "_POLL_S", 0)
    pages = {"https://a.example/x": "The Book paperback 45,00 €", "https://b.example/y": "The Book paperback 39,90 €",
             "https://c.example/z": "The Book paperback 35,00 €"}

    async def fake_fetch(url):
        return pages.get(url, ""), url
    monkeypatch.setattr(outcome, "fetch_text", fake_fetch)

    async def no_leaks(prefix, since, until=None):
        return []
    monkeypatch.setattr(rehearsal, "leaks", no_leaks)
    lane = FakeBatLane(store, [
        [{"title": "The Book", "url": "https://a.example/x", "value": 45.0}],
        [{"title": "The Book", "url": "https://b.example/y", "value": 39.9},
         {"title": "The Book", "url": "https://c.example/z", "value": 30.0}],   # the page says 35: not borne out
        [],
    ])
    loop.set_lane(lane)
    ideas = [[{"hypothesis": "big shops", "strategy_tag": "a", "files": [], "why": ""},
              {"hypothesis": "used-book sites", "strategy_tag": "b", "files": [], "why": ""}],
             [{"hypothesis": "publisher store", "strategy_tag": "c", "files": [], "why": ""}]]

    async def propose(run_, ledger, k):
        return ideas.pop(0) if ideas else []
    loop.set_proposer(propose)
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book",
                                                     must_include=["paperback"]))

    async def go():
        await store.create_run(run_id="cs_0123456789ab", owner_id="u", title="t", objective=obj, status="baseline")
        status = await loop.Driver(store, "cs_0123456789ab").drive()
        r = await store.get_run("cs_0123456789ab")
        return status, r, await store.list_experiments("cs_0123456789ab")
    try:
        status, r, exps = run(go())
    finally:
        loop.set_lane(None)
        loop.set_proposer(None)
    assert [e["status"] for e in exps] == ["kept", "kept", "failed"]
    assert r["champion"]["dev"]["best"]["url"] == "https://b.example/y" and r["champion"]["dev"]["value"] == 39.9
    checks = r["champion"]["dev"]["checks"]
    assert [c["verified"] for c in checks] == [True, False] and "doesn't show 30" in checks[1]["why"]
    assert round(r["spent_usd"], 2) == 1.0 and status in ("stalled", "stopped")


def test_screen_disqualifies_a_leaky_attempt(store, monkeypatch):
    async def found(prefix, since, until=None):
        return [{"agent": f"{prefix}main", "tool": "google_mail", "what": "google_mail send", "verdict": "intent"}]
    monkeypatch.setattr(rehearsal, "leaks", found)

    async def go():
        await store.add_version(version_id="v_h", run_id="cs_x", content={"candidates": [{"url": "https://a.example", "value": 1}]})
        return await outcome.screen({"id": "cs_x"}, {"head_sha": "v_h", "child_id": "bat_1a2b3c4d5e6f", "created_at": 10})
    got = run(go())
    assert not got["ok"] and "did something outside after all" in got["reason"] and "google_mail send" in got["reason"]


# ── carrying out the winner, once ────────────────────────────────────


def test_carry_out_task_text():
    r = {"objective": {"subject": {"task": "Buy the cheapest copy", "rank": {"label": "price", "direction": "min"}}}}
    best = {"title": "The Book", "value": 39.9, "currency": "EUR", "url": "https://b.example/y",
            "details": "ships in 2 days", "next_step": "click Place order"}
    t = outcome.carry_out_task(r, best, [{"kind": "commit", "what": "click 'Place order'"}])
    assert "once" in t and "https://b.example/y" in t and "click 'Place order'" in t and "ask the owner" in t


def test_carry_out_route_starts_one_bat_run(store, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import bat_loop
    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user
    from captain_claw.flight_deck.bat_store import BatStore

    bat = BatStore(tmp_path / "bat.db")
    monkeypatch.setattr(bat_loop, "get_store", lambda: bat)

    async def kick():
        return 0
    monkeypatch.setattr(bat_loop, "kick", kick)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "role": "user"}
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book"))
    best = {"title": "The Book", "url": "https://b.example/y", "value": 39.9, "currency": "EUR", "details": "", "next_step": ""}

    async def prepare():
        await store.create_run(run_id="cs_0123456789ab", owner_id="u", title="t", objective=obj, status="running")
        await store.update_run("cs_0123456789ab", champion={"sha": "v_c", "exp": "exp_abcdef123456",
                                                           "dev": {"best": best, "value": 39.9}})
    run(prepare())
    with TestClient(app) as c:
        assert c.post("/fd/captain-swarm/runs/cs_0123456789ab/carry-out").status_code == 409     # still running
        run(store.set_status("cs_0123456789ab", "stalled", "done trying"))
        r = c.post("/fd/captain-swarm/runs/cs_0123456789ab/carry-out")
        assert r.status_code == 200, r.text
        bat_id = r.json()["bat_run"]
        assert c.post("/fd/captain-swarm/runs/cs_0123456789ab/carry-out").status_code == 409     # only once
    b = run(bat.get_run(bat_id))
    assert b["status"] == "planning" and "rehearsal" not in (b.get("config") or {})
    assert "https://b.example/y" in b["task"] and "ask the owner" in b["task"]
    run(bat.close())
