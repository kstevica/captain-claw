"""An agent's "way of working" setup (``/fd/agent-preset``), opened right after it is created.

* a model's reply is cleaned per group (clipped, deduplicated, unknown
  categories fall back, intent names become skill ids), and the whole items of
  a cut-off reply are kept;
* a document with no text is reported without the model; long ones are cut
  at the archetype forge's caps;
* prepare reads with the owner's model, keeps the documents' text under the
  preset's id for that owner only, and gives every document a knowledge entry;
  no input is a 400, no model a 409, a spent allowance a 429;
* the person's note on a document (what it is, how to use it) reaches the
  model and the agent's knowledge; File library picks are read from the
  person's own agents' folders only; the agent being set up frames the read
  with its archetype, and only its owner can name it;
* apply gives an existing agent the reviewed preset: one SKILL.md per intent
  (in the skills folder the agent reads), the documents as Markdown in
  knowledge/ with an index, glossary and the full rules, and a capped rules
  block after its instructions — stored for the owner and returned; another
  person's preset id brings no documents, another person's agent is a 404;
* facts go to the agent as insights once it answers; a failure never raises.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import archetypes as arch_mod
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.skills import _parse_frontmatter

BOB = "user-bob"
CAROL = "user-carol"
INSTR = "fd:process-fleet-instructions"
_ARCH = {"id": "support", "role": "Customer Support", "tier": "balanced",
         "tools": ["read"], "fleet_instructions": "You answer customers kindly."}
_TIER = {"provider": "ollama", "model": "test-model", "base_url": "http://127.0.0.1:1"}

_REPLY = {
    "rules": [{"text": "Always answer in Croatian.", "source": {"doc": "policy.txt", "section": "Tone"}},
              {"text": "always answer in croatian.", "source": {}},
              {"text": "  ", "source": {}}],
    "intents": [{"name": "Refund Request!", "when": "A customer asks for money back.",
                 "steps": ["Check the order date.", "Refund within 14 days."],
                 "examples": ["I want a refund"], "source": {"doc": "policy.txt", "section": "Refunds"}},
                {"name": "refund request", "when": "Second refund flow.", "steps": ["x"]},
                {"name": "", "when": "", "steps": []}],
    "facts": [{"content": "Refund window is 14 days.", "category": "policy", "importance": 42,
               "source": {"doc": "policy.txt"}},
              {"content": "Support email is help@acme.test", "category": "contact", "importance": "x"}],
    "escalation": [{"text": "Ask before refunding more than 200 EUR.", "source": {"doc": "policy.txt"}}],
    "templates": [{"name": "Refund reply", "body": "Dear [name],\n\nYour refund…", "source": {}},
                  {"name": "Empty", "body": ""}],
    "glossary": [{"term": "RMA", "meaning": "Return merchandise authorisation"}],
    "knowledge": [{"doc": "policy.txt", "title": "Refund policy", "summary": "How refunds work."},
                  {"doc": "invented.pdf", "title": "Made up", "summary": "Not uploaded."}],
    "conflicts": ["14 days in one place, 30 in another."],
    "warnings": [],
}


class _FakePopen:
    calls: list[str] = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None):
        _FakePopen.calls.append(Path(cwd).name)
        self.pid = 424242
        if stdout is not None:
            stdout.close()

    def poll(self):
        return None


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck with Bob and Carol; spawns never start a process."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_PORT", "25999")
    monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", _no_docker)
    _FakePopen.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())

    async def fake_merged(_db, _uid):
        return [dict(_ARCH)]

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)

    async def fake_tier(_uid, prefer=()):
        return dict(_TIER)

    monkeypatch.setattr(ap, "_resolve_tier", fake_tier)

    async def not_spent(_uid):
        return False

    monkeypatch.setattr(ap, "_quota_spent", not_spent)
    server._save_process_registry({})
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _auth(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


def _model(monkeypatch, *replies: str) -> list[dict]:
    """The model answers with ``replies`` in turn; the calls it got."""
    calls: list[dict] = []

    async def fake_call(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        calls.append({"user": user, "owner": owner_id, "max_tokens": max_tokens})
        return replies[min(len(calls), len(replies)) - 1], 100

    monkeypatch.setattr(ap, "call_model", fake_call)
    return calls


async def _prepare(uid: str, files: list[tuple[str, bytes]], instructions: str = "Be brief.",
                   **extra: str) -> httpx.Response:
    async with _client() as c:
        return await c.post("/fd/agent-preset/prepare", headers=_auth(uid),
                            data={"archetype": "support", "instructions": instructions, **extra},
                            files=[("files", (n, b, "text/plain")) for n, b in files])


async def _spawn(uid: str, name: str, port: int) -> str:
    async with _client() as c:
        r = await c.post("/fd/spawn-process", headers=_auth(uid), json={
            "name": name, "archetype": "support", "botport_enabled": False,
            "web_enabled": True, "web_port": port})
    assert r.status_code == 200, r.text
    return r.json()["slug"]


# ── Cleaning ──────────────────────────────────────────────────────────


def test_normalise_cleans_every_group():
    p = ap.normalise(_REPLY)
    assert [r["text"] for r in p["rules"]] == ["Always answer in Croatian."]
    assert p["rules"][0]["source"] == {"doc": "policy.txt", "section": "Tone"}
    assert [i["name"] for i in p["intents"]] == ["refund-request", "refund-request-2"]
    assert p["intents"][0]["steps"] == ["Check the order date.", "Refund within 14 days."]
    assert p["facts"][0]["category"] == "fact" and p["facts"][0]["importance"] == 10
    assert p["facts"][1]["importance"] == 5 and p["facts"][1]["source"]["doc"] == "instructions"
    assert [t["name"] for t in p["templates"]] == ["Refund reply"]
    assert "\n" in p["templates"][0]["body"]
    assert p["conflicts"] == ["14 days in one place, 30 in another."]
    assert set(p) == {*ap.GROUPS, "conflicts", "warnings"}


def test_normalise_caps_each_group():
    p = ap.normalise({"rules": [{"text": f"Rule {n}"} for n in range(100)]})
    assert len(p["rules"]) == ap.LIMITS["rules"]


def test_parse_reads_a_fenced_reply():
    p = ap.parse_preset(f"Here you go:\n```json\n{json.dumps(_REPLY)}\n```")
    assert p["escalation"][0]["text"].startswith("Ask before refunding")


def test_parse_keeps_the_whole_items_of_a_cut_off_reply():
    full = json.dumps(_REPLY)
    cut = full[:full.index('"facts"') + 80]   # ends mid-way through the second fact
    p = ap.parse_preset(cut)
    assert [r["text"] for r in p["rules"]] == ["Always answer in Croatian."]
    assert [i["name"] for i in p["intents"]] == ["refund-request", "refund-request-2"]
    assert len(p["facts"]) <= 1


def test_parse_refuses_a_reply_with_nothing():
    with pytest.raises(ValueError):
        ap.parse_preset('{"rules": [], "intents": []}')
    with pytest.raises(ValueError):
        ap.parse_preset("I could not find anything.")


# ── The documents ─────────────────────────────────────────────────────


class _Upload:
    def __init__(self, filename: str, data: bytes):
        self.filename, self._data = filename, data

    async def read(self) -> bytes:
        return self._data


async def test_a_document_without_text_is_reported_without_the_model():
    docs, warnings = await ap.read_documents([_Upload("scan.txt", b"   \n  "), _Upload("ok.txt", b"Hello")])
    assert [d["name"] for d in docs] == ["ok.txt"]
    assert any("scan.txt" in w and "no text" in w for w in warnings)


async def test_long_documents_are_cut_at_the_caps(monkeypatch):
    monkeypatch.setattr(ap, "_MAX_PER_FILE", 10)
    monkeypatch.setattr(ap, "_MAX_TOTAL", 15)
    docs, warnings = await ap.read_documents([
        _Upload("a.txt", b"a" * 50), _Upload("a.txt", b"b" * 50), _Upload("c.txt", b"c" * 50)])
    assert [(d["name"], d["chars"], d["truncated"]) for d in docs] == [("a.txt", 10, True), ("a.txt (2)", 5, True)]
    assert any(w.startswith("c.txt: left out") for w in warnings)


# ── Prepare ───────────────────────────────────────────────────────────


async def test_prepare_reads_with_the_owners_model_and_keeps_the_documents(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    r = await _prepare(BOB, [("policy.txt", b"Refunds within 14 days."), ("empty.txt", b"  "),
                             ("prices.txt", b"Basic plan 10 EUR.")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls[0]["owner"] == BOB and "Customer Support" in calls[0]["user"]
    assert "Refunds within 14 days." in calls[0]["user"] and "Be brief." in calls[0]["user"]
    # Every uploaded document has a knowledge entry; the invented one is gone.
    assert [k["doc"] for k in body["preset"]["knowledge"]] == ["policy.txt", "prices.txt"]
    assert body["preset"]["knowledge"][1] == {"doc": "prices.txt", "title": "prices", "summary": ""}
    assert any("empty.txt" in w for w in body["preset"]["warnings"])
    assert [d["name"] for d in body["docs"]] == ["policy.txt", "prices.txt"]
    assert [d["name"] for d in ap.load_docs(BOB, body["id"])] == ["policy.txt", "prices.txt"]
    assert ap.load_docs(CAROL, body["id"]) == []


async def test_prepare_asks_again_when_the_reply_is_no_good(deck, monkeypatch):
    calls = _model(monkeypatch, "thinking… no JSON yet", json.dumps(_REPLY))
    r = await _prepare(BOB, [("policy.txt", b"Refunds.")])
    assert r.status_code == 200, r.text
    assert len(calls) == 2 and calls[1]["max_tokens"] > calls[0]["max_tokens"]
    assert ap._RETRY_NUDGE in calls[1]["user"]


async def test_prepare_502_when_the_model_never_answers_usefully(deck, monkeypatch):
    _model(monkeypatch, "no", "still no")
    r = await _prepare(BOB, [("policy.txt", b"Refunds.")])
    assert r.status_code == 502


async def test_prepare_needs_something_to_read(deck, monkeypatch):
    _model(monkeypatch, json.dumps(_REPLY))
    assert (await _prepare(BOB, [], instructions="  ")).status_code == 400
    r = await _prepare(BOB, [("scan.txt", b" ")], instructions="")
    assert r.status_code == 400 and "scan.txt" in r.json()["detail"]


async def test_prepare_without_a_model_or_allowance(deck, monkeypatch):
    _model(monkeypatch, json.dumps(_REPLY))

    async def no_tier(_uid, prefer=()):
        return None

    monkeypatch.setattr(ap, "_resolve_tier", no_tier)
    assert (await _prepare(BOB, [("p.txt", b"x")])).status_code == 409

    async def spent(_uid):
        return True

    monkeypatch.setattr(ap, "_quota_spent", spent)
    assert (await _prepare(BOB, [("p.txt", b"x")])).status_code == 429


async def test_prepare_needs_a_session(deck):
    async with _client() as c:
        r = await c.post("/fd/agent-preset/prepare", data={"instructions": "x"})
    assert r.status_code == 401


async def test_the_persons_notes_reach_the_model_and_fill_a_missing_summary(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    notes = [{"description": "Our EU refund policy, 2026", "usage": "rules"},
             {"description": "Price list for the Basic plan", "usage": "bogus"}]
    r = await _prepare(BOB, [("policy.txt", b"Refunds within 14 days."), ("prices.txt", b"Basic 10 EUR.")],
                       notes=json.dumps(notes))
    assert r.status_code == 200, r.text
    body = r.json()
    prompt = calls[0]["user"]
    assert "### policy.txt\nThe owner says: Our EU refund policy, 2026\nUse it as: rules and policies" in prompt
    assert "### prices.txt\nThe owner says: Price list for the Basic plan\nBasic 10 EUR." in prompt
    assert [(d["description"], d["usage"], d["origin"]) for d in body["docs"]] == [
        ("Our EU refund policy, 2026", "rules", "upload"), ("Price list for the Basic plan", "auto", "upload")]
    # The model summarised the policy; the price list has the person's words.
    assert [k["summary"] for k in body["preset"]["knowledge"]] == ["How refunds work.", "Price list for the Basic plan"]
    assert ap.load_docs(BOB, body["id"])[0]["usage"] == "rules"


async def test_library_picks_are_read_from_the_persons_own_agents_only(deck, monkeypatch):
    from captain_claw.flight_deck import file_library as fl
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(server.DATA_DIR / "file_library.db"))
    server._save_process_registry({"bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "owner": BOB}})
    faq = server.DATA_DIR / "bob-agent" / "data" / "workspace" / "faq.md"
    faq.parent.mkdir(parents=True)
    faq.write_text("# FAQ\n\nWe ship within 3 days.\n")
    fl.get_store().sync_agent("bob-agent", fl.walk_workspace("bob-agent"))
    pick = json.dumps([{"id": fl.LibraryStore.file_id("bob-agent", "faq.md"),
                        "description": "Shipping FAQ", "usage": "reference"}])

    calls = _model(monkeypatch, json.dumps(_REPLY))
    r = await _prepare(BOB, [], library=pick)
    assert r.status_code == 200, r.text
    assert "We ship within 3 days." in calls[0]["user"] and "Use it as: reference material" in calls[0]["user"]
    assert r.json()["docs"] == [{"name": "faq.md", "chars": len("# FAQ\n\nWe ship within 3 days."),
                                 "truncated": False, "origin": "library", "description": "Shipping FAQ",
                                 "usage": "reference", "agent": "Bob's agent"}]

    # Carol can't read Bob's file by its id: nothing is read, and she is told.
    r = await _prepare(CAROL, [], instructions="", library=pick)
    assert r.status_code == 400 and "no longer there" in r.json()["detail"]
    assert len(calls) == 1


async def test_the_agent_being_set_up_frames_the_read_and_must_be_ones_own(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24302)
    r = await _prepare(BOB, [("p.txt", b"Refunds.")], archetype="", agent=slug)
    assert r.status_code == 200, r.text
    assert "Customer Support" in calls[0]["user"]
    assert (await _prepare(CAROL, [("p.txt", b"x")], archetype="", agent=slug)).status_code == 404


def test_old_presets_are_pruned(deck):
    old = ap.save_docs(BOB, [{"name": "a.txt", "text": "a", "chars": 1}], "support")
    new = ap.save_docs(BOB, [{"name": "b.txt", "text": "b", "chars": 1}], "support")
    meta = ap._preset_dir(BOB, old) / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "created": time.time() - ap._TTL - 5}))
    assert ap.load_docs(BOB, old) == []
    assert ap.prune() == 1
    assert not ap._preset_dir(BOB, old).exists() and ap._preset_dir(BOB, new).exists()
    assert ap.load_docs(BOB, "../../etc") == [] and ap._preset_dir(BOB, "../x") is None


# ── Seeding ───────────────────────────────────────────────────────────


def test_rules_block_stays_under_its_cap_and_points_at_the_rest():
    preset = ap.normalise({
        "rules": [{"text": f"Rule number {n}: " + "x" * 150} for n in range(30)],
        "escalation": [{"text": "Ask before any payment."}],
        "templates": [{"name": "Offer", "body": "..."}],
        "glossary": [{"term": "RMA", "meaning": "a return"}],
    })
    block = ap.rules_block(preset, [{"file": "p.md", "title": "Policy", "summary": ""}], ["refund-request"])
    assert len(block) <= ap._BLOCK_CAP
    assert block.startswith("## Your owner's rules")
    assert "- Ask before any payment." in block          # when-to-ask goes in first
    assert "more rules in knowledge/owner-rules.md" in block
    assert "refund-request" in block and "knowledge/INDEX.md" in block and "Offer" in block


def test_an_empty_preset_adds_no_block():
    assert ap.rules_block(ap.normalise({}), [], []) == ""
    assert ap.with_block("Base.", "") == "Base."


def test_with_block_keeps_the_instructions_within_their_cap():
    out = ap.with_block("b" * 64000, "## Your owner's rules\n- x")
    assert len(out) <= 64000 and out.endswith("- x")


def test_seed_writes_skills_knowledge_and_the_full_rules(deck, tmp_path):
    preset_id = ap.save_docs(BOB, [{"name": "policy.txt", "text": "Refunds within 14 days.", "chars": 23,
                                    "description": "EU refund policy", "usage": "rules"},
                                   {"name": "dropped.txt", "text": "Not kept.", "chars": 9}], "support")
    reviewed = {**ap.normalise(_REPLY), "id": preset_id}
    reviewed["knowledge"] = [{"doc": "policy.txt", "title": "Refund policy", "summary": "How refunds work."}]
    agent_dir = tmp_path / "agent"
    workspace = agent_dir / "data" / "workspace"
    done = ap.seed(agent_dir, workspace, BOB, reviewed)
    block, facts = done["block"], done["facts"]
    assert done["skills"] == ["refund-request", "refund-request-2"]
    assert done["counts"] == {"rules": 1, "escalation": 1, "templates": 1, "glossary": 1,
                              "intents": 2, "facts": 2, "knowledge": 1}

    skill = agent_dir / "data" / "home-config-parent" / ".captain-claw" / "skills" / "refund-request" / "SKILL.md"
    meta = _parse_frontmatter(skill.read_text())
    assert meta == {"name": "refund-request", "description": "A customer asks for money back."}
    assert "1. Check the order date." in skill.read_text()

    kn = workspace / "knowledge"
    assert sorted(p.name for p in kn.iterdir()) == ["INDEX.md", "glossary.md", "owner-rules.md", "policy.md"]
    assert "Refunds within 14 days." in (kn / "policy.md").read_text()
    assert "Your owner says: EU refund policy" in (kn / "policy.md").read_text()
    assert "Use it as: rules to follow." in (kn / "policy.md").read_text()
    assert "[Refund policy](policy.md) — How refunds work. (rules to follow)" in (kn / "INDEX.md").read_text()
    assert "## Template: Refund reply" in (kn / "owner-rules.md").read_text()
    assert "Always answer in Croatian." in block and "Ask before refunding" in block
    assert [f["content"] for f in facts] == ["Refund window is 14 days.", "Support email is help@acme.test"]
    assert ap.load_docs(BOB, preset_id) == []      # the waiting documents are dropped


def test_another_persons_preset_brings_no_documents(deck, tmp_path):
    preset_id = ap.save_docs(BOB, [{"name": "policy.txt", "text": "Bob's secret.", "chars": 13}], "support")
    reviewed = {**ap.normalise(_REPLY), "id": preset_id,
                "knowledge": [{"doc": "policy.txt", "title": "Policy", "summary": ""}]}
    ap.seed(tmp_path / "agent", tmp_path / "agent" / "ws", CAROL, reviewed)
    assert not (tmp_path / "agent" / "ws" / "knowledge" / "policy.md").exists()
    assert ap.load_docs(BOB, preset_id)            # Bob's is untouched


async def _apply(uid: str, slug: str, preset: dict) -> httpx.Response:
    async with _client() as c:
        return await c.post("/fd/agent-preset/apply", headers=_auth(uid), json={"agent": slug, "preset": preset})


async def test_apply_sets_up_an_existing_agent_and_stores_its_instructions(deck, monkeypatch):
    slug = await _spawn(BOB, "Helpdesk", 24300)
    assert json.loads(await deck.get_setting(BOB, INSTR))[slug] == "You answer customers kindly."
    assert not (server.DATA_DIR / slug / "data" / "workspace" / "knowledge").exists()

    _model(monkeypatch, json.dumps(_REPLY))
    prepared = (await _prepare(BOB, [("policy.txt", b"Refunds within 14 days.")], agent=slug)).json()
    given: list = []
    monkeypatch.setattr(ap, "schedule_facts",
                        lambda slug, port_of, token, facts: given.append((slug, port_of(), token, facts)))
    r = await _apply(BOB, slug, {**prepared["preset"], "id": prepared["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    stored = body["instructions"]
    assert stored.startswith("You answer customers kindly.") and "## Your owner's rules" in stored
    assert stored.endswith(body["block"])
    assert json.loads(await deck.get_setting(BOB, INSTR))[slug] == stored
    assert body["counts"]["intents"] == 2 and body["knowledge"] == ["Refund policy"]
    agent = server.DATA_DIR / slug
    assert (agent / "data" / "workspace" / "knowledge" / "policy.md").is_file()
    assert (agent / "data" / "home-config-parent" / ".captain-claw" / "skills" / "refund-request" / "SKILL.md").is_file()
    entry = server._load_process_registry()[slug]
    assert given == [(slug, 24300, entry["web_auth"], given[0][3])] and len(given[0][3]) == 2
    assert ap.load_docs(BOB, prepared["id"]) == []      # the waiting documents are used up
    assert _FakePopen.calls == [slug]                    # no restart


async def test_apply_refuses_another_persons_agent(deck, monkeypatch):
    slug = await _spawn(BOB, "Helpdesk", 24301)
    r = await _apply(CAROL, slug, {**ap.normalise(_REPLY), "id": ""})
    assert r.status_code == 404
    assert not (server.DATA_DIR / slug / "data" / "workspace" / "knowledge").exists()
    assert json.loads(await deck.get_setting(BOB, INSTR))[slug] == "You answer customers kindly."


# ── Facts ─────────────────────────────────────────────────────────────


async def test_facts_wait_for_the_agent_and_never_raise(monkeypatch):
    posted: list = []

    async def ready(port_of, token, timeout):
        return port_of()

    async def post(port, token, facts):
        posted.append((port, token, len(facts)))
        return len(facts)

    monkeypatch.setattr(ap, "_wait_ready", ready)
    monkeypatch.setattr(ap, "_post_facts", post)
    facts = ap.normalise(_REPLY)["facts"]
    assert await ap.give_facts("a", lambda: 24500, "tok", facts) == 2
    assert posted == [(24500, "tok", 2)]

    async def never(port_of, token, timeout):
        return 0

    monkeypatch.setattr(ap, "_wait_ready", never)
    assert await ap.give_facts("a", lambda: 24500, "tok", facts) == 0

    async def boom(port_of, token, timeout):
        raise RuntimeError("down")

    monkeypatch.setattr(ap, "_wait_ready", boom)
    assert await ap.give_facts("a", lambda: 24500, "tok", facts) == 0
