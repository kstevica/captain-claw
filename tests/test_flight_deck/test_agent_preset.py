"""An agent's "way of working" setup (``/fd/agent-preset``), opened right after it is created.

* a model's reply is cleaned per group (clipped, deduplicated, unknown
  categories fall back, intent names become skill ids), and the whole items of
  a cut-off reply are kept; a part with nothing in it is not a failure;
* a document is read in full when it is added (a Library pick too — not the
  Library's own 10 pages) and staged with its pages and tokens, for its owner
  only, without the model; text is split at pages and headings within a byte
  limit, losing nothing;
* the material is read in steps sized to the model's window — small
  documents share one, a long one spans many, knowledge-only documents are not
  read, more than a setup reads is left out with a note — and the parts are
  merged by the model; a step whose answer can't be read is skipped, none
  readable fails the read, a spent allowance stops it but keeps what was read;
  a read can be stopped, and only its owner follows it;
* the person's note on a document (what it is, how to use it) reaches the
  model and the agent's knowledge; the agent being set up frames the read
  with its archetype, and only its owner can name it;
* apply gives an existing agent the reviewed preset: one SKILL.md per intent
  (in the skills folder the agent reads), every document whole in knowledge/ —
  a long one as an overview plus parts under semantic memory's 256 KB limit —
  with an index, glossary and the full rules, and a capped rules block after
  its instructions — stored for the owner and returned; another person's
  preset id brings no documents, another person's agent is a 404;
* facts go to the agent as insights once it answers; a failure never raises.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_docs as pdocs
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
    """The model answers with ``replies`` in turn (the last one again after
    that); the calls it got."""
    calls: list[dict] = []

    async def fake_call(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        calls.append({"system": system, "user": user, "owner": owner_id, "max_tokens": max_tokens})
        return replies[min(len(calls), len(replies)) - 1], 100

    monkeypatch.setattr(ap, "call_model", fake_call)
    return calls


async def _stage(uid: str, files: list[tuple[str, bytes]] = (), library: list | None = None) -> dict:
    async with _client() as c:
        r = await c.post("/fd/agent-preset/documents", headers=_auth(uid),
                         data={"library": json.dumps(library or [])},
                         files=[("files", (n, b, "text/plain")) for n, b in files] or None)
    assert r.status_code == 200, r.text
    return r.json()


async def _start(uid: str, docs: list[dict], instructions: str = "Be brief.", **extra) -> httpx.Response:
    async with _client() as c:
        return await c.post("/fd/agent-preset/prepare", headers=_auth(uid), json={
            "archetype": "support", "instructions": instructions, "docs": docs, **extra})


async def _wait(uid: str, job_id: str) -> dict:
    async with _client() as c:
        for _ in range(500):
            r = await c.get(f"/fd/agent-preset/prepare/{job_id}", headers=_auth(uid))
            assert r.status_code == 200, r.text
            if r.json()["state"] != "running":
                return r.json()
            await asyncio.sleep(0.01)
    raise AssertionError("the read never finished")


async def _prepare(uid: str, files: list[tuple[str, bytes]], instructions: str = "Be brief.",
                   notes: list[dict] | None = None, **extra) -> dict:
    """Stage *files*, read them (with *notes* per file), and wait: the job."""
    staged = (await _stage(uid, files))["docs"]
    docs = [{"id": d["id"], **((notes or [])[i] if i < len(notes or []) else {})} for i, d in enumerate(staged)]
    r = await _start(uid, docs, instructions, **extra)
    assert r.status_code == 200, r.text
    return await _wait(uid, r.json()["id"])


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


def test_a_part_with_nothing_in_it_is_not_a_failure():
    assert ap._parse_part('{"rules": []}')["rules"] == []
    assert ap._parse_part("I could not find anything.") is None


# ── The documents: read in full, split, staged ────────────────────────


def _pdf_text(pages: int, chars_per_page: int = 2000) -> str:
    """What the PDF extractor makes of a document of *pages* pages."""
    return "\n".join(["# big.pdf", ""] + [f"## Page {n}\n\n" + ("x" * (chars_per_page - 1)) + "\n" for n in range(1, pages + 1)])


def test_split_cuts_at_pages_and_says_which():
    text = _pdf_text(100)
    parts = pdocs.split_text(text, 50_000)
    assert len(parts) > 1 and "".join(p["text"] for p in parts) == text
    assert all(len(p["text"].encode()) <= 50_000 for p in parts)
    assert parts[0]["label"].startswith("pages 1–") and parts[-1]["label"].endswith("–100")
    assert all(p["text"].lstrip().startswith(("#", "## Page")) for p in parts)


def test_split_cuts_a_block_too_big_on_its_own():
    text = "# Title\n\n" + ("word " * 60_000) + "\n\nend"
    parts = pdocs.split_text(text, 20_000)
    assert "".join(p["text"] for p in parts) == text
    assert all(len(p["text"].encode()) <= 20_000 for p in parts)
    multibyte = "č" * 30_000
    assert all(len(p["text"].encode()) <= 10_000 for p in pdocs.split_text(multibyte, 10_000))


def test_pages_are_counted_or_estimated():
    assert pdocs.pages_of(_pdf_text(7)) == (7, True)
    assert pdocs.pages_of("## Slide 1\nA\n## Slide 2\nB") == (2, True)
    assert pdocs.pages_of("y" * 9001) == (4, False)


def test_a_library_file_is_read_in_full(monkeypatch, tmp_path):
    from captain_claw.tools import document_extract as dx
    asked: list = []

    def fake_pdf(path, max_pages):
        asked.append(max_pages)
        return _pdf_text(3), None

    monkeypatch.setattr(dx, "_extract_pdf_markdown", fake_pdf)
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    text, why = pdocs.extract(tmp_path / "a.pdf")
    assert why == "" and "## Page 3" in text and asked == [3000]   # not the Library's 10 pages
    assert pdocs.extract(tmp_path / "x.exe") == ("", "not a kind of document the setup reads")


async def test_documents_are_staged_with_their_size_and_no_model(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    big = _pdf_text(40).encode()
    out = await _stage(BOB, [("policy.md", big), ("empty.txt", b""), ("scan.txt", b"   \n  ")])
    [doc] = out["docs"]
    assert doc["name"] == "policy.md" and doc["pages"] == 40 and doc["pages_exact"] is True
    assert doc["chars"] == len(big.decode()) and doc["tokens"] == -(-doc["chars"] // 4)
    assert any("empty.txt" in w for w in out["warnings"]) and any("scan.txt" in w for w in out["warnings"])
    assert pdocs.load_staged(ap.owner_root(BOB), doc["id"])["text"] == big.decode()
    assert pdocs.load_staged(ap.owner_root(CAROL), doc["id"]) is None
    assert calls == []


async def test_library_picks_are_staged_from_the_persons_own_agents_only(deck, monkeypatch):
    from captain_claw.flight_deck import file_library as fl
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(server.DATA_DIR / "file_library.db"))
    server._save_process_registry({"bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "owner": BOB}})
    faq = server.DATA_DIR / "bob-agent" / "data" / "workspace" / "faq.md"
    faq.parent.mkdir(parents=True)
    faq.write_text("# FAQ\n\nWe ship within 3 days.\n")
    fl.get_store().sync_agent("bob-agent", fl.walk_workspace("bob-agent"))
    file_id = fl.LibraryStore.file_id("bob-agent", "faq.md")

    out = await _stage(BOB, library=[file_id])
    assert [(d["name"], d["origin"], d["agent"]) for d in out["docs"]] == [("faq.md", "library", "Bob's agent")]
    out = await _stage(CAROL, library=[{"id": file_id}])
    assert out["docs"] == [] and "no longer there" in out["warnings"][0]


def test_old_presets_and_staged_documents_are_pruned(deck):
    old = ap.save_docs(BOB, [{"name": "a.txt", "text": "a"}], "support")
    new = ap.save_docs(BOB, [{"name": "b.txt", "text": "b"}], "support")
    meta = ap._preset_dir(BOB, old) / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "created": time.time() - ap._TTL - 5}))
    staged = pdocs.stage(ap.owner_root(BOB), "c.txt", "c", "upload")
    smeta = ap.owner_root(BOB) / "staged" / staged["id"] / "meta.json"
    smeta.write_text(json.dumps({**json.loads(smeta.read_text()), "created": time.time() - pdocs.TTL - 5}))
    assert ap.load_docs(BOB, old) == [] and pdocs.load_staged(ap.owner_root(BOB), staged["id"]) is None
    assert ap.prune() == 2
    assert not ap._preset_dir(BOB, old).exists() and ap._preset_dir(BOB, new).exists()
    assert ap.load_docs(BOB, "../../etc") == [] and ap._preset_dir(BOB, "../x") is None


# ── Planning the steps ────────────────────────────────────────────────


def _doc(name: str, text: str, read: bool = True) -> dict:
    return {"name": name, "text": text, "read": read, "description": "", "usage": "auto"}


def test_steps_share_small_documents_and_split_long_ones():
    long = _pdf_text(100)            # ~200k characters
    steps, notes = ap.plan_steps("Be brief.", [_doc("a.txt", "A" * 100), _doc("b.txt", "B" * 100),
                                               _doc("big.pdf", long), _doc("ref.csv", "R" * 100, read=False)],
                                 step_tok=10_000)      # 40k characters a step
    assert notes == []
    assert [p["doc"]["name"] for p in steps[0]["pieces"]][:2] == ["a.txt", "b.txt"]
    assert all(sum(len(p["text"]) for p in s["pieces"]) <= 40_000 for s in steps)
    assert len(steps) >= 5 and "big.pdf, pages" in steps[-1]["label"]
    assert not any(p["doc"]["name"] == "ref.csv" for s in steps for p in s["pieces"])


def test_instructions_alone_are_one_step_and_too_much_is_not_read(monkeypatch):
    steps, _ = ap.plan_steps("Be brief.", [], step_tok=10_000)
    assert len(steps) == 1 and steps[0]["label"] == "your instructions"
    assert ap.plan_steps("", [_doc("r.csv", "x", read=False)], step_tok=10_000)[0] == []
    monkeypatch.setattr(ap, "_MAX_READ_CHARS", 1000)
    steps, notes = ap.plan_steps("", [_doc("a.txt", "A" * 800), _doc("b.txt", "B" * 800), _doc("c.txt", "C" * 10)],
                                 step_tok=10_000)
    assert sum(len(p["text"]) for s in steps for p in s["pieces"]) == 1000
    assert any(n.startswith("b.txt: only the first 200") for n in notes)
    assert any(n.startswith("c.txt: not read") for n in notes)


def test_a_step_is_sized_from_the_models_window():
    assert ap.step_tokens({"provider": "ollama", "input_ctx": 32768}) == 32768 // 2 - ap._PROMPT_TOKENS
    assert ap.step_tokens({"provider": "ollama"}) == 32768 // 2 - ap._PROMPT_TOKENS
    assert ap.step_tokens({"provider": "anthropic", "input_ctx": 1_000_000}) == 100_000
    assert ap.step_tokens({"provider": "ollama", "input_ctx": 4096}) == 4000


# ── Reading ───────────────────────────────────────────────────────────


async def test_prepare_reads_with_the_owners_model_and_keeps_the_documents(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    job = await _prepare(BOB, [("policy.txt", b"Refunds within 14 days."), ("prices.txt", b"Basic plan 10 EUR.")])
    assert job["state"] == "done", job
    body = job["result"]
    assert len(calls) == 1 and calls[0]["owner"] == BOB and "Customer Support" in calls[0]["user"]
    assert "Refunds within 14 days." in calls[0]["user"] and "Be brief." in calls[0]["user"]
    assert "Reading in parts" not in calls[0]["user"]
    # Every document has a knowledge entry; the invented one is gone.
    assert [k["doc"] for k in body["preset"]["knowledge"]] == ["policy.txt", "prices.txt"]
    assert body["preset"]["knowledge"][1] == {"doc": "prices.txt", "title": "prices", "summary": ""}
    assert [d["name"] for d in body["docs"]] == ["policy.txt", "prices.txt"] and body["steps"] == 1
    assert body["tokens"] == 100 and job["tokens"] == 100
    assert [d["name"] for d in ap.load_docs(BOB, body["id"])] == ["policy.txt", "prices.txt"]
    assert ap.load_docs(CAROL, body["id"]) == []


async def test_the_persons_notes_reach_the_model_and_fill_a_missing_summary(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    notes = [{"description": "Our EU refund policy, 2026", "usage": "rules"},
             {"description": "Price list for the Basic plan", "usage": "bogus"}]
    job = await _prepare(BOB, [("policy.txt", b"Refunds within 14 days."), ("prices.txt", b"Basic 10 EUR.")],
                         notes=notes)
    body = job["result"]
    prompt = calls[0]["user"]
    assert "### policy.txt\nThe owner says: Our EU refund policy, 2026\nUse it as: rules and policies" in prompt
    assert "### prices.txt\nThe owner says: Price list for the Basic plan\nBasic 10 EUR." in prompt
    assert [(d["description"], d["usage"], d["origin"]) for d in body["docs"]] == [
        ("Our EU refund policy, 2026", "rules", "upload"), ("Price list for the Basic plan", "auto", "upload")]
    assert [k["summary"] for k in body["preset"]["knowledge"]] == ["How refunds work.", "Price list for the Basic plan"]
    assert ap.load_docs(BOB, body["id"])[0]["usage"] == "rules"


async def test_a_long_document_is_read_in_steps_and_the_parts_merged(deck, monkeypatch):
    async def small_tier(_uid, prefer=()):
        return {**_TIER, "input_ctx": 8000 * 2 + 2 * ap._PROMPT_TOKENS}   # 8k-token steps

    monkeypatch.setattr(ap, "_resolve_tier", small_tier)
    merged = {**_REPLY, "rules": [{"text": "Merged rule."}],
              "knowledge": [{"doc": "big.md", "title": "The big one", "summary": "All of it."}]}
    part = {"rules": [{"text": "A rule from a part."}], "knowledge": [{"doc": "big.md", "title": "Big", "summary": "Some."}]}
    replies = [json.dumps(part)] * 3 + [json.dumps(merged)]
    calls = _model(monkeypatch, *replies)
    job = await _prepare(BOB, [("big.md", _pdf_text(40).encode())])       # ~80k characters: 3 steps
    assert job["state"] == "done", job
    reads = [c for c in calls if "Merging an agent setup" not in c["system"]]
    merges = [c for c in calls if "Merging an agent setup" in c["system"]]
    assert len(reads) == 3 and len(merges) == 1 and job["steps"] == 3
    assert "This is part 2 of 3" in reads[1]["user"] and "Be brief." not in reads[1]["user"]
    assert "### big.md — pages" in reads[0]["user"] and "(part 1 of 3 of this document)" in reads[0]["user"]
    assert job["result"]["preset"]["rules"] == [{"text": "Merged rule.", "source": {"doc": "instructions", "section": ""}}]
    assert job["result"]["preset"]["knowledge"][0]["title"] == "The big one"
    assert job["result"]["tokens"] == 400


def test_a_merge_that_copied_one_parts_summary_gets_all_of_them():
    parts = [ap.normalise({"knowledge": [{"doc": "h.md", "title": "Handbook", "summary": "Quotes and approvals."}]}),
             ap.normalise({"knowledge": [{"doc": "h.md", "title": "Handbook", "summary": "Monthly close."}]})]
    copied = ap.normalise({"knowledge": [{"doc": "h.md", "title": "Handbook", "summary": "Quotes and approvals."}]})
    assert ap.whole_summaries(copied, parts)["knowledge"][0]["summary"] == "Quotes and approvals. Monthly close."
    rewritten = ap.normalise({"knowledge": [{"doc": "h.md", "title": "Handbook", "summary": "The whole handbook."}]})
    assert ap.whole_summaries(rewritten, parts)["knowledge"][0]["summary"] == "The whole handbook."


async def test_a_step_that_cant_be_read_is_skipped_and_none_readable_fails(deck, monkeypatch):
    async def small_tier(_uid, prefer=()):
        return {**_TIER, "input_ctx": 8000 * 2 + 2 * ap._PROMPT_TOKENS}

    monkeypatch.setattr(ap, "_resolve_tier", small_tier)
    good = json.dumps({"rules": [{"text": "Kept."}]})
    _model(monkeypatch, "no", "still no", good, good, good)   # part 1 fails twice; then the rest
    job = await _prepare(BOB, [("big.md", _pdf_text(40).encode())])
    assert job["state"] == "done"
    assert any(w.startswith("Part 1 of 3") for w in job["result"]["preset"]["warnings"])
    assert [r["text"] for r in job["result"]["preset"]["rules"]] == ["Kept."]

    _model(monkeypatch, "no")
    job = await _prepare(BOB, [("p.txt", b"Refunds.")])
    assert job["state"] == "error" and "couldn't be read" in job["error"]
    assert ap.load_docs(BOB, job["id"]) == []


async def test_running_out_of_allowance_keeps_what_was_read(deck, monkeypatch):
    async def small_tier(_uid, prefer=()):
        return {**_TIER, "input_ctx": 8000 * 2 + 2 * ap._PROMPT_TOKENS}

    monkeypatch.setattr(ap, "_resolve_tier", small_tier)
    calls = _model(monkeypatch, json.dumps({"rules": [{"text": "From part one."}]}))
    checks = {"n": 0}

    async def spent_after_first(_uid):
        checks["n"] += 1
        return checks["n"] > 1        # the check before reading is fine, the one before part 2 isn't

    monkeypatch.setattr(ap, "_quota_spent", spent_after_first)
    job = await _prepare(BOB, [("big.md", _pdf_text(40).encode())])
    assert job["state"] == "done" and len(calls) == 1
    assert any("allowance ran out after part 1 of 3" in w for w in job["result"]["preset"]["warnings"])
    assert job["result"]["preset"]["rules"][0]["text"] == "From part one."


async def test_knowledge_only_documents_cost_nothing(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))

    async def no_tier(_uid, prefer=()):
        return None

    monkeypatch.setattr(ap, "_resolve_tier", no_tier)       # not needed: nothing is read
    staged = (await _stage(BOB, [("prices.csv", b"plan,price\nBasic,10\n")]))["docs"]
    r = await _start(BOB, [{"id": staged[0]["id"], "usage": "reference", "read": False,
                            "description": "Price list"}], instructions="")
    job = await _wait(BOB, r.json()["id"])
    assert job["state"] == "done" and calls == [] and job["result"]["steps"] == 0
    assert job["result"]["preset"]["knowledge"] == [{"doc": "prices.csv", "title": "prices", "summary": "Price list"}]
    assert job["result"]["docs"][0]["read"] is False


async def test_a_read_can_be_stopped_and_only_its_owner_follows_it(deck, monkeypatch):
    started = asyncio.Event()

    async def slow_call(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        started.set()
        await asyncio.sleep(30)
        return json.dumps(_REPLY), 100

    monkeypatch.setattr(ap, "call_model", slow_call)
    staged = (await _stage(BOB, [("p.txt", b"Refunds.")]))["docs"]
    job_id = (await _start(BOB, [{"id": staged[0]["id"]}])).json()["id"]
    await asyncio.wait_for(started.wait(), 5)
    async with _client() as c:
        assert (await c.get(f"/fd/agent-preset/prepare/{job_id}", headers=_auth(CAROL))).status_code == 404
        r = await c.delete(f"/fd/agent-preset/prepare/{job_id}", headers=_auth(BOB))
    assert r.json()["state"] == "cancelled"
    assert ap.load_docs(BOB, job_id) == []


async def test_prepare_needs_something_to_read(deck, monkeypatch):
    _model(monkeypatch, json.dumps(_REPLY))
    assert (await _start(BOB, [], instructions="  ")).status_code == 400
    r = await _start(BOB, [{"id": "0" * 24}], instructions="")
    assert r.status_code == 400 and "gone" in r.json()["detail"]


async def test_prepare_without_a_model_or_allowance(deck, monkeypatch):
    _model(monkeypatch, json.dumps(_REPLY))

    async def no_tier(_uid, prefer=()):
        return None

    monkeypatch.setattr(ap, "_resolve_tier", no_tier)
    assert (await _start(BOB, [])).status_code == 409

    async def spent(_uid):
        return True

    monkeypatch.setattr(ap, "_quota_spent", spent)
    assert (await _start(BOB, [])).status_code == 429


async def test_prepare_needs_a_session(deck):
    async with _client() as c:
        r = await c.post("/fd/agent-preset/prepare", json={"instructions": "x"})
    assert r.status_code == 401


async def test_the_agent_being_set_up_frames_the_read_and_must_be_ones_own(deck, monkeypatch):
    calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24302)
    job = await _prepare(BOB, [("p.txt", b"Refunds.")], archetype="", agent=slug)
    assert job["state"] == "done" and "Customer Support" in calls[0]["user"]
    assert (await _start(CAROL, [], archetype="", agent=slug)).status_code == 404


async def test_the_reader_says_its_model_and_step(deck):
    async with _client() as c:
        r = await c.get("/fd/agent-preset/reader", headers=_auth(BOB))
    assert r.json() == {"model": "test-model", "step_tokens": ap.step_tokens(_TIER)}


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
    prepared = (await _prepare(BOB, [("policy.txt", b"Refunds within 14 days.")], agent=slug))["result"]
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


def test_a_long_document_reaches_the_agent_whole_in_parts_it_can_index(deck, tmp_path):
    text = _pdf_text(400)                                   # ~800 KB
    preset_id = ap.save_docs(BOB, [{"name": "manual.pdf", "text": text, "pages": 400,
                                    "description": "The product manual", "usage": "reference", "read": False}],
                             "support")
    reviewed = {**ap.normalise({}), "id": preset_id,
                "knowledge": [{"doc": "manual.pdf", "title": "Product manual", "summary": ""}]}
    done = ap.seed(tmp_path / "agent", tmp_path / "ws", BOB, reviewed)
    kn = tmp_path / "ws" / "knowledge"
    parts = sorted(kn.glob("manual-part-*.md"))
    assert len(parts) >= 4 and done["knowledge"][0]["parts"] == len(parts)
    assert all(p.stat().st_size <= 262_144 for p in kn.iterdir())     # semantic memory's own limit
    body = "".join(p.read_text().split("---\n\n", 1)[1] for p in parts)
    assert all(f"## Page {n}\n" in body for n in (1, 200, 400))       # every page is there
    overview = (kn / "manual.md").read_text()
    assert f"kept in {len(parts)} parts" in overview and "(manual-part-01.md)" in overview
    assert "Your owner says: The product manual" in overview and "Use it as: reference." in overview
    assert f"— {len(parts)} parts" in (kn / "INDEX.md").read_text()
    assert parts[0].read_text().startswith(f"# Product manual — part 1 of {len(parts)} (pages 1–")


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
