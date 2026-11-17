"""First Mate rehearsals (P3, piece 1): a factory run acts on nothing in the world.

Pinned here:

* the guard takes a First Mate run id as its tag, sends its intents to First
  Mate's route, answers in First Mate's words, and holds no site for it;
  Captain Swarm's tags work as before;
* trust decides a run's kind: ``propose`` and ``commit`` rehearse, ``direct``
  acts; a rehearsal's Bat run carries the tag, and its brief says so;
* the intent route keeps an intent only for the caller owner's own rehearsal
  while it goes, within a cap; the agent guard covers it (no remote callers);
* a rehearsal the tool-call register shows reached outside fails, its factory
  pauses, and its owner is told; one that didn't finishes as usual;
* a rehearsal's intents show with its run, and in its Line post.
"""

from __future__ import annotations

import httpx
import pytest

from captain_claw import rehearsal_turn
from captain_claw.flight_deck import bat_loop, events
from captain_claw.flight_deck import first_mate_line as fml
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm

BOB, CAROL = "user-bob", "user-carol"


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": status, "cumulative_tokens": 0,
                             "cumulative_usd": 0.0, "truth": "", "stopped_reason": ""}
        return run_id

    async def append_event(self, run_id, stage, message="", **data):
        return 1

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return [dict(r) for r in self.runs.values() if r["owner_id"] == owner_id][:limit]

    async def set_status(self, run_id, status, **kw):
        self.runs[run_id].update(status=status, **kw)


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    """A rehearsal's leak check reads the tool-call register; a test never opens one."""
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True})
    return s


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    return fake


@pytest.fixture
def owner_is(monkeypatch):
    """Who the intent route resolves the calling agent's owner to."""
    who = {"owner": BOB}
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_resolve_owner", lambda body: who["owner"])
    return who


def _live(store, **kw):
    kw.setdefault("title", "Inbox drafts")
    kw.setdefault("charter", "Draft replies to mail that needs one.")
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


# ── the guard ──────────────────────────────────────────────────────────

def test_the_guard_knows_both_tags(monkeypatch):
    fm_tag, cs_tag = "fmr_0123456789abcdef", "cs_abcdef12:exp_abcdef12"
    assert rehearsal_turn.valid_tag(fm_tag) and rehearsal_turn.valid_tag(cs_tag)
    assert not rehearsal_turn.valid_tag("fmr_") and not rehearsal_turn.valid_tag("bat_123456")
    assert rehearsal_turn.is_first_mate(fm_tag) and not rehearsal_turn.is_first_mate(cs_tag)
    assert rehearsal_turn.route_for(fm_tag) == "first-mate" and rehearsal_turn.route_for(cs_tag) == "captain-swarm"
    monkeypatch.setenv(rehearsal_turn.ENV, fm_tag)
    assert rehearsal_turn.active() == fm_tag
    monkeypatch.setenv(rehearsal_turn.ENV, "not a tag")
    assert rehearsal_turn.active() == ""


async def test_a_first_mate_rehearsal_records_and_holds_no_site(monkeypatch):
    posts: list[tuple[str, dict, str]] = []

    async def fake_post(path, body, agent, tag=""):
        posts.append((path, body, tag))
        return {"ok": True}

    monkeypatch.setattr(rehearsal_turn, "_post", fake_post)
    monkeypatch.setenv(rehearsal_turn.ENV, "fmr_0123456789abcdef")
    answer = await rehearsal_turn.tool_block("google_mail", {"action": "send", "to": "ana@client.example",
                                                            "subject": "Re: invoice", "password": "x"}, None)
    assert answer.startswith("Recorded as an intent, not done") and "Your owner sees" in answer
    [(path, body, tag)] = posts
    assert path == "intent" and tag == "fmr_0123456789abcdef" and body["kind"] == "intent"
    assert body["args"]["to"] == "ana@client.example" and "password" not in body["args"]
    # The browser opens a page with no site check (one run per factory, no crowd).
    assert await rehearsal_turn.tool_block("browser", {"action": "open", "url": "https://shop.example"}, None) is None
    assert len(posts) == 1
    # Captain Swarm's attempt still asks for the site.
    monkeypatch.setenv(rehearsal_turn.ENV, "cs_abcdef12:exp_abcdef12")
    await rehearsal_turn.tool_block("browser", {"action": "open", "url": "https://shop.example"}, None)
    assert posts[-1][0] == "site"


# ── runs ───────────────────────────────────────────────────────────────

async def test_trust_decides_whether_a_run_rehearses(store, bat):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    assert run["kind"] == "rehearsal"
    cfg = bat.runs[run["child_id"]]["config"]
    assert cfg["rehearsal"] == run["id"] and rehearsal_turn.valid_tag(cfg["rehearsal"])
    assert runs.REHEARSAL_NOTE in bat.runs[run["child_id"]]["task"]
    store.finish_run(run["id"], "done")
    g = _live(store, title="Tidy", grants={"organize": True})
    store.set_trust(g["id"], "direct", "owner")
    direct = await runs.start(g["id"], BOB, watch=False)
    assert direct["kind"] == "run" and "rehearsal" not in bat.runs[direct["child_id"]]["config"]
    assert runs.REHEARSAL_NOTE not in bat.runs[direct["child_id"]]["task"]


# ── the intent route ───────────────────────────────────────────────────

def _intent(tag, **kw):
    return routes.AgentIntentReq(tag=tag, kind=kw.pop("kind", "intent"), tool="google_mail",
                                 what="google_mail send", args={"to": "ana@client.example"}, **kw)


async def test_the_route_keeps_an_owners_rehearsal_intents(store, bat, owner_is, monkeypatch):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    got = await routes.agent_intent(_intent(run["id"]))
    assert got["ok"]
    [kept] = store.intents(run["id"])
    assert (kept["kind"], kept["tool"], kept["what"], kept["status"]) == ("intent", "google_mail", "google_mail send", "open")
    assert (await routes.agent_intent(_intent(run["id"], kind="commit")))["ok"]
    assert (await routes.agent_intent(_intent(run["id"], kind="run_it")))["why"] == "not an intent"
    owner_is["owner"] = CAROL
    assert (await routes.agent_intent(_intent(run["id"])))["why"] == "no such rehearsal"
    owner_is["owner"] = BOB
    monkeypatch.setattr(routes, "INTENTS_PER_RUN", 2)
    assert "too many" in (await routes.agent_intent(_intent(run["id"])))["why"]
    store.finish_run(run["id"], "done")
    assert (await routes.agent_intent(_intent(run["id"])))["why"] == "that run is over"


async def test_the_route_refuses_a_run_that_isnt_a_rehearsal(store, bat, owner_is):
    f = _live(store, grants={"organize": True})
    store.set_trust(f["id"], "direct", "owner")
    run = await runs.start(f["id"], BOB, watch=False)
    assert (await routes.agent_intent(_intent(run["id"])))["why"] == "no such rehearsal"
    assert (await routes.agent_intent(_intent("fmr_0000000000000000")))["why"] == "no such rehearsal"


async def test_no_remote_caller_reaches_the_agent_route(monkeypatch):
    from captain_claw.flight_deck import server
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 4001)),
                                 base_url="http://fd.test") as cl:
        r = await cl.post("/fd/first-mate/agent/intent", json={"tag": "fmr_0123456789abcdef"})
    assert r.status_code == 403


# ── the leak check ─────────────────────────────────────────────────────

async def test_a_rehearsal_that_reached_outside_fails_and_pauses(store, bat, monkeypatch):
    from captain_claw.flight_deck import captain_swarm_rehearsal
    seen: list[str] = []

    async def leaks(prefix, since, until=None):
        seen.append(prefix)
        return [{"agent": "bat-bat-1234-worker", "tool": "google_mail", "what": "google_mail send",
                 "verdict": "intent"}]

    monkeypatch.setattr(captain_swarm_rehearsal, "leaks", leaks)
    paused: list[tuple[str, bool]] = []

    async def told(factory, reason, *, needs_you):
        paused.append((reason, needs_you))

    monkeypatch.setattr(fml, "factory_paused", told)
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="done", truth="Drafted 2 replies.")
    done = await runs.refresh(run["id"])
    assert done["status"] == "failed" and done["verdict"] == "leak" and "google_mail send" in done["error"]
    assert seen and seen[0].startswith("bat-")
    g = store.get(f["id"])
    assert g["state"] == "paused" and g["state_reason"] == runs.LEAK_REASON
    assert paused == [(runs.LEAK_REASON, True)]


async def test_a_clean_rehearsal_finishes_as_usual(store, bat, monkeypatch):
    from captain_claw.flight_deck import captain_swarm_rehearsal

    async def no_leaks(prefix, since, until=None):
        return []

    monkeypatch.setattr(captain_swarm_rehearsal, "leaks", no_leaks)
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="done", truth="Drafted 2 replies.")
    assert (await runs.refresh(run["id"]))["status"] == "done"
    assert store.get(f["id"])["state"] == "running"


async def test_the_register_unreadable_leaves_the_run_on_the_guard(store, bat):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="done", truth="ok")
    assert (await runs.refresh(run["id"]))["status"] == "done"  # _no_register: it can't be read


# ── showing intents ────────────────────────────────────────────────────

async def test_intents_show_with_the_run_and_in_its_post(store, bat, owner_is):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    for who in ("Ana", "Ben"):
        await routes.agent_intent(routes.AgentIntentReq(tag=run["id"], kind="intent", tool="google_mail",
                                                        what=f"Reply to {who}", args={}))
    [shown] = routes._with_intents(store, [store.get_run(run["id"])])
    assert [i["what"] for i in shown["intents"]] == ["Reply to Ana", "Reply to Ben"]
    # The words and a short detail made from them, never the raw arguments.
    assert set(shown["intents"][0]) == {"id", "kind", "tool", "what", "status", "detail"}
    text = fml._would(store.intents(run["id"]))
    assert "nothing of it was done" in text and "- Reply to Ben" in text
    assert fml._would([]) == "It needed nothing outside."
