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
    monkeypatch.setenv("FD_SETUP_TOKENS_PER_MIN", "0")       # unpaced; the pacing tests set their own

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
    # The owner's tier reads, unless a test gives the agent a model of its own (_own_model).
    monkeypatch.setattr(ap, "agent_reader", lambda slug, entry: None)
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
    assert ap.step_tokens({"provider": "ollama", "input_ctx": 4096}) == 4000
    # However big the window, a call stays small: 16k of material, under ~20k with its prompt.
    assert ap.step_tokens({"provider": "anthropic", "input_ctx": 1_000_000}) == 16_000
    assert ap.step_tokens({"provider": "anthropic"}) == 16_000


def test_every_call_of_a_long_read_stays_under_20k_tokens():
    big = _doc("manual.md", _pdf_text(120))                 # ~240k characters: many steps
    step_tok = ap.step_tokens({"provider": "anthropic", "input_ctx": 1_000_000})
    steps, _ = ap.plan_steps("Be brief. " * 200, [big], step_tok)
    assert len(steps) > 3
    prompts = [ap.step_prompt(dict(_ARCH), "Be brief. " * 200, s, n, len(steps)) for n, s in enumerate(steps, 1)]
    system = ap._PROMPT_FILE.read_text(encoding="utf-8")
    assert max(len(system) + len(p) for p in prompts) // pdocs.CHARS_PER_TOKEN <= 20_000


def test_a_read_is_paced_unless_its_model_runs_here(monkeypatch):
    monkeypatch.delenv("FD_SETUP_TOKENS_PER_MIN", raising=False)
    assert ap.pace_tpm({"provider": "anthropic", "model": "claude-haiku-4-5"}) == ap._PACE_TPM
    assert ap.pace_tpm({"provider": "ollama", "model": "gemma4:31b-cloud"}) == ap._PACE_TPM
    assert ap.pace_tpm({"provider": "ollama", "model": "glm-5:cloud"}) == ap._PACE_TPM
    # … through the local daemon, as Ollama's cloud models are called:
    assert ap.pace_tpm({"provider": "ollama", "model": "gemma4:31b-cloud",
                        "base_url": "http://localhost:11434"}) == ap._PACE_TPM
    assert ap.pace_tpm({"provider": "ollama", "model": "qwen3:8b"}) == 0              # local Ollama
    assert ap.pace_tpm({"provider": "openai", "model": "m", "base_url": "http://localhost:1234/v1"}) == 0
    monkeypatch.setenv("FD_SETUP_TOKENS_PER_MIN", "10000")
    assert ap.pace_tpm({"provider": "anthropic", "model": "m"}) == 10_000
    monkeypatch.setenv("FD_SETUP_TOKENS_PER_MIN", "0")
    assert ap.pace_tpm({"provider": "anthropic", "model": "m"}) == 0


async def test_calls_are_spaced_by_the_pace_and_the_wait_shows(monkeypatch):
    monkeypatch.setenv("FD_SETUP_TOKENS_PER_MIN", "60000")      # 1,000 tokens a second
    monkeypatch.setattr(ap, "_PACE_GAP", 0.05)
    job = ap.ReadJob(id="j", owner="o")
    cloud = {"provider": "anthropic", "model": "m"}
    start = time.monotonic()
    await ap._pace(cloud, job, 300)                              # the first goes at once …
    assert time.monotonic() - start < 0.1 and job.waiting is None
    second = asyncio.create_task(ap._pace(cloud, job, 300))     # … the next 0.3 s after it
    await asyncio.sleep(0.05)
    assert job.waiting and job.waiting["why"] == "pacing" and job.waiting["until"] > time.time()
    await second
    assert time.monotonic() - start >= 0.25 and job.waiting is None
    local = {"provider": "ollama", "model": "qwen3:8b"}
    start = time.monotonic()
    await ap._pace(local, job, 10**6)                            # a local model never waits
    assert time.monotonic() - start < 0.05


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
    assert r.json() == {"model": "test-model", "step_tokens": ap.step_tokens(_TIER), "pace_tpm": 0,
                        "via": "tiers", "note": ""}


# ── The agent's own model reads ───────────────────────────────────────

_HAIKU = {"provider": "anthropic", "model": "claude-haiku-4-5"}


def _own_model(monkeypatch, slug: str, *, ctx: int = 200000, available: bool = True, **entry) -> None:
    """Give *slug* its own model in its config.yaml (the real agent_reader
    reads it) and say whether it is running or asleep."""
    cfg = server.DATA_DIR / slug / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"model": _HAIKU, "context": {"max_tokens": ctx}}), encoding="utf-8")
    if entry:
        reg = server._load_process_registry()
        reg[slug].update(entry)
        server._save_process_registry(reg)
    monkeypatch.setattr(ap, "agent_reader", _REAL_AGENT_READER)
    monkeypatch.setattr(ap.agent_idle, "is_available", lambda s: available)


_REAL_AGENT_READER = ap.agent_reader


def _agent_model(monkeypatch, *replies, up: bool = True) -> list[dict]:
    """The agent answers (its /api/llm/complete) with *replies* in turn —
    an exception is raised instead of answered; the calls it got."""
    calls: list[dict] = []

    async def fake_up(reader, job):
        return up

    async def fake_complete(reader, system, prompt, owner_id, max_tokens):
        calls.append({"agent": reader["agent"], "model": reader["model"], "user": prompt, "max_tokens": max_tokens})
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply, 50

    monkeypatch.setattr(ap, "_agent_up", fake_up)
    monkeypatch.setattr(ap, "_agent_complete", fake_complete)
    return calls


async def test_the_agent_reads_with_its_own_model(deck, monkeypatch):
    tier_calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24303)
    _own_model(monkeypatch, slug)
    calls = _agent_model(monkeypatch, json.dumps(_REPLY))
    async with _client() as c:
        r = await c.get("/fd/agent-preset/reader", params={"agent": slug}, headers=_auth(BOB))
    assert r.json() == {"model": "claude-haiku-4-5", "via": "agent", "note": "", "pace_tpm": 0,
                        "step_tokens": ap.step_tokens({"provider": "anthropic", "input_ctx": 200000})}

    job = await _prepare(BOB, [("p.txt", b"Refunds within 14 days.")], archetype="", agent=slug)
    assert job["state"] == "done" and job["model"] == "claude-haiku-4-5" and job["via"] == "agent"
    assert job["result"]["model"] == "claude-haiku-4-5" and job["tokens"] == 50
    assert calls and calls[0]["agent"] == slug and "Refunds within 14 days." in calls[0]["user"]
    assert tier_calls == []                     # the owner's model never ran
    async with _client() as c:                  # nor can another person ask about it
        r = await c.get("/fd/agent-preset/reader", params={"agent": slug}, headers=_auth(CAROL))
    assert r.status_code == 404


def test_an_agents_reader_follows_its_config(deck, tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "agent_reader", _REAL_AGENT_READER)
    cfg = server.DATA_DIR / "a" / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("model:\n  provider: ollama\n  model: qwen3:8b\ncontext:\n  max_tokens: 160000\n")
    local = ap.agent_reader("a", {"name": "A", "key_source": "byok"})
    assert local == {"agent": "a", "name": "A", "provider": "ollama", "model": "qwen3:8b", "base_url": "",
                     "input_ctx": 32768, "byok": True}   # a local Ollama window is clamped
    cfg.write_text("model:\n  provider: ollama\n  model: glm-5:cloud\n")
    assert ap.agent_reader("a", {})["input_ctx"] == 0      # unknown: step_tokens' default
    cfg.unlink()
    assert ap.agent_reader("a", {"provider": "anthropic", "model": "m"})["model"] == "m"   # the registry, else
    assert ap.agent_reader("a", {}) is None


async def test_an_agent_thats_off_leaves_the_read_to_the_owners_model(deck, monkeypatch):
    tier_calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24304)
    _own_model(monkeypatch, slug, available=False)
    async with _client() as c:
        r = await c.get("/fd/agent-preset/reader", params={"agent": slug}, headers=_auth(BOB))
    assert r.json()["model"] == "test-model" and r.json()["via"] == "tiers"
    assert "Helpdesk is switched off" in r.json()["note"]

    job = await _prepare(BOB, [("p.txt", b"Refunds.")], archetype="", agent=slug)
    assert job["state"] == "done" and job["via"] == "tiers" and len(tier_calls) == 1
    assert any("Helpdesk is switched off" in w for w in job["result"]["preset"]["warnings"])

    async def no_tier(_uid, prefer=()):
        return None

    monkeypatch.setattr(ap, "_resolve_tier", no_tier)
    r = await _start(BOB, [], "Be brief.", archetype="", agent=slug)
    assert r.status_code == 409 and "Switch it on" in r.json()["detail"]


async def test_an_agent_that_doesnt_answer_hands_over_to_the_owners_model(deck, monkeypatch):
    tier_calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24305)
    _own_model(monkeypatch, slug)
    _agent_model(monkeypatch, json.dumps(_REPLY), up=False)
    job = await _prepare(BOB, [("p.txt", b"Refunds.")], archetype="", agent=slug)
    assert job["state"] == "done" and job["model"] == "test-model" and job["via"] == "tiers"
    assert len(tier_calls) == 1
    assert job["result"]["preset"]["warnings"][0].startswith("Helpdesk didn't answer, so test-model read")


async def test_an_agent_that_stops_answering_mid_read_hands_over(deck, monkeypatch):
    async def small_tier(_uid, prefer=()):
        return {**_TIER, "input_ctx": 8000 * 2 + 2 * ap._PROMPT_TOKENS}

    monkeypatch.setattr(ap, "_resolve_tier", small_tier)
    tier_calls = _model(monkeypatch, json.dumps(_REPLY))
    slug = await _spawn(BOB, "Helpdesk", 24306)
    _own_model(monkeypatch, slug, ctx=8000 * 2 + 2 * ap._PROMPT_TOKENS)
    calls = _agent_model(monkeypatch, json.dumps(_REPLY), ap._AgentGoneError("not answering"))
    job = await _prepare(BOB, [("big.md", _pdf_text(40).encode())], archetype="", agent=slug)
    assert job["state"] == "done" and job["steps"] == 3
    assert len(calls) == 2 and len(tier_calls) >= 2     # part 1 by the agent; the rest by the owner's
    assert any("Helpdesk stopped answering, so test-model read the rest." == w
               for w in job["result"]["preset"]["warnings"])


async def test_an_agents_rest_call_carries_its_model_and_counts_its_tokens(deck, monkeypatch):
    slug = await _spawn(BOB, "Helpdesk", 24307)
    reg = server._load_process_registry()
    reg[slug].update(web_port=24307, web_auth="tok-1", key_source="byok")
    server._save_process_registry(reg)
    monkeypatch.setattr(server, "_process_is_alive", lambda s: True)
    counted: list[tuple] = []

    async def count(owner, tokens, byok):
        counted.append((owner, tokens, byok))

    monkeypatch.setattr(ap, "_count", count)
    seen: list[httpx.Request] = []
    answers = [httpx.Response(200, json={"ok": True, "content": "{}", "usage": {"prompt_tokens": 70,
                                                                                 "completion_tokens": 5}}),
               httpx.Response(500, json={"ok": False, "error": "litellm.RateLimitError: rate_limit_error",
                                         "retry_after": 7}),
               httpx.Response(503, json={"ok": False, "error": "no LLM provider"})]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers[len(seen) - 1]

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    reader = ap.agent_reader(slug, reg[slug]) or {"agent": slug, "name": "Helpdesk", "model": "m"}
    reader = {**reader, "byok": True}

    assert await ap._agent_complete(reader, "sys", "material", BOB, 8000) == ("{}", 75)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/llm/complete" and seen[0].url.params["token"] == "tok-1"
    assert seen[0].url.port == 24307
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "material"}]
    assert body["max_tokens"] == 8000 and body["timeout"] == ap._CALL_TIMEOUT
    assert counted == [(BOB, 75, True)]

    with pytest.raises(ap._ModelError) as busy:
        await ap._agent_complete(reader, "sys", "material", BOB, 8000)
    assert busy.value.wait == 7 and ap.is_busy(busy.value)
    with pytest.raises(ap._AgentGoneError):
        await ap._agent_complete(reader, "sys", "material", BOB, 8000)


# ── A busy model's server is waited out ───────────────────────────────


def _rate_limited(seconds: str) -> Exception:
    """A 429 carrying its response, as LiteLLM's RateLimitError does (``.response``
    with the server's headers) — without importing litellm, which reconfigures
    logging for every test after it."""
    return httpx.HTTPStatusError("Error", request=httpx.Request("POST", "https://api.example"),
                                 response=httpx.Response(429, headers={"retry-after": seconds}))


def test_busy_and_retry_after_are_read_from_the_error():
    from captain_claw.exceptions import LLMAPIError, is_busy, retry_after

    limited = _rate_limited("12")
    wrapped = LLMAPIError("anthropic API call failed: Error", status_code=429)
    wrapped.__context__ = limited            # what LiteLLMProvider's raise inside `except` leaves
    assert is_busy(wrapped) and retry_after(wrapped) == 12.0
    assert is_busy(LLMAPIError("Overloaded", status_code=529))
    assert is_busy("AnthropicException - {\"type\":\"rate_limit_error\"}")
    assert not is_busy(LLMAPIError("invalid x-api-key", status_code=401))
    assert retry_after(ValueError("no response")) is None


def _busy_then(monkeypatch, *replies) -> list[int]:
    """The owner's model: a busy server for every Exception in *replies*,
    else the reply; waits take no time. The calls."""
    calls: list[int] = []

    async def fake_call(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        calls.append(1)
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply, 100

    monkeypatch.setattr(ap, "call_model", fake_call)
    monkeypatch.setattr(ap, "_WAITS", (0, 0, 0))
    return calls


async def test_a_busy_model_is_waited_out(deck, monkeypatch):
    from captain_claw.exceptions import LLMAPIError
    limited = LLMAPIError("anthropic API call failed: litellm.RateLimitError", status_code=429)
    calls = _busy_then(monkeypatch, limited, limited, json.dumps(_REPLY))
    job = await _prepare(BOB, [("p.txt", b"Refunds.")])
    assert job["state"] == "done" and len(calls) == 3 and job["waiting"] is None
    assert job["result"]["preset"]["rules"]


async def test_a_wait_shows_on_the_job_until_it_ends(deck, monkeypatch):
    from captain_claw.exceptions import LLMAPIError
    calls = _busy_then(monkeypatch, LLMAPIError("rate limit", status_code=429), json.dumps(_REPLY))
    monkeypatch.setattr(ap, "_WAITS", (0.3,))
    staged = (await _stage(BOB, [("p.txt", b"Refunds.")]))["docs"]
    r = await _start(BOB, [{"id": staged[0]["id"]}])
    seen = None
    async with _client() as c:
        for _ in range(100):
            view = (await c.get(f"/fd/agent-preset/prepare/{r.json()['id']}", headers=_auth(BOB))).json()
            if view["waiting"]:
                seen = view["waiting"]
                break
            await asyncio.sleep(0.01)
    assert seen and seen["why"] == "busy" and seen["model"] == "test-model" and seen["until"] > time.time() - 1
    assert (await _wait(BOB, r.json()["id"]))["state"] == "done" and len(calls) == 2


async def test_a_model_still_busy_stops_the_read_and_keeps_what_was_read(deck, monkeypatch):
    from captain_claw.exceptions import LLMAPIError

    async def small_tier(_uid, prefer=()):
        return {**_TIER, "input_ctx": 8000 * 2 + 2 * ap._PROMPT_TOKENS}

    monkeypatch.setattr(ap, "_resolve_tier", small_tier)
    limited = LLMAPIError("rate_limit_error", status_code=429)
    calls = _busy_then(monkeypatch, json.dumps(_REPLY), limited)
    job = await _prepare(BOB, [("big.md", _pdf_text(40).encode())])
    assert job["state"] == "done" and job["steps"] == 3
    assert len(calls) == 1 + 1 + len(ap._WAITS)          # part 1, then part 2 tried and waited for
    assert any("test-model was still rate-limited or overloaded after waiting, so parts 2–3 weren't read." in w
               for w in job["result"]["preset"]["warnings"])
    assert job["result"]["preset"]["rules"]               # part 1's findings are kept

    _busy_then(monkeypatch, limited)
    job = await _prepare(BOB, [("p.txt", b"Refunds.")])
    assert job["state"] == "error" and "still rate-limited or overloaded" in job["error"]


async def test_a_refused_call_is_not_waited_for(deck, monkeypatch):
    from captain_claw.exceptions import LLMAPIError
    calls = _busy_then(monkeypatch, LLMAPIError("invalid x-api-key", status_code=401))
    job = await _prepare(BOB, [("p.txt", b"Refunds.")])
    assert job["state"] == "error" and len(calls) == 1 and "invalid x-api-key" in job["error"]


# ── Agent side: /api/llm/complete takes the setup's timeout ───────────


class _TimedProvider:
    request_timeout = None

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.seen: list = []                 # shared by shallow copies

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        from captain_claw.llm import LLMResponse
        self.seen.append(self.request_timeout)
        if self.fail:
            raise self.fail
        return LLMResponse(content="{}", model="m", usage={"total_tokens": 3}, finish_reason="stop")


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


async def test_llm_complete_takes_a_timeout_for_this_call_only():
    import types

    from captain_claw.web_server import WebServer
    shared = _TimedProvider()
    web = WebServer.__new__(WebServer)
    web.agent = types.SimpleNamespace(provider=shared)
    resp = await web._llm_complete(_JsonRequest({"messages": [{"role": "user", "content": "x"}], "timeout": 300}))
    assert resp.status == 200 and shared.seen == [300.0] and shared.request_timeout is None


async def test_llm_complete_says_how_long_a_busy_server_asked_to_wait():
    import types

    from captain_claw.exceptions import LLMAPIError
    from captain_claw.web_server import WebServer
    wrapped = LLMAPIError("anthropic API call failed: litellm.RateLimitError: Error", status_code=429)
    wrapped.__context__ = _rate_limited("9")
    web = WebServer.__new__(WebServer)
    web.agent = types.SimpleNamespace(provider=_TimedProvider(fail=wrapped))
    resp = await web._llm_complete(_JsonRequest({"messages": [{"role": "user", "content": "x"}]}))
    data = json.loads(resp.text)
    assert resp.status == 500 and data["ok"] is False and data["retry_after"] == 9.0
    assert "RateLimitError" in data["error"]


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
    assert len(out) <= 64000 and out.endswith("- x\n" + ap._BLOCK_CLOSE)


def test_a_new_block_replaces_the_one_an_earlier_setup_left():
    once = ap.with_block("You answer customers kindly.", "## Your owner's rules\n\n- Old rule.")
    twice = ap.with_block(once + "\n\nMy own note.", "## Your owner's rules\n\n- New rule.")
    assert twice.count(ap._BLOCK_OPEN) == 1 and "Old rule" not in twice and "New rule" in twice
    assert twice.startswith("You answer customers kindly.\n\nMy own note.")
    assert ap.with_block(once, "") == "You answer customers kindly."
    bare = ("Base.\n\n## Your owner's rules\n\nSet up by your owner. Follow these.\n\n### Rules\n- Old.\n\n"
            "## Later section\nKeep me.")
    assert ap.strip_block(bare) == "Base.\n\n## Later section\nKeep me."


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


async def _apply(uid: str, slug: str, preset: dict, mode: str = "add") -> httpx.Response:
    async with _client() as c:
        return await c.post("/fd/agent-preset/apply", headers=_auth(uid),
                            json={"agent": slug, "preset": preset, "mode": mode})


async def test_apply_sets_up_an_existing_agent_and_stores_its_instructions(deck, monkeypatch):
    slug = await _spawn(BOB, "Helpdesk", 24300)
    assert json.loads(await deck.get_setting(BOB, INSTR))[slug] == "You answer customers kindly."
    assert not (server.DATA_DIR / slug / "data" / "workspace" / "knowledge").exists()

    _model(monkeypatch, json.dumps(_REPLY))
    prepared = (await _prepare(BOB, [("policy.txt", b"Refunds within 14 days.")], agent=slug))["result"]
    given: list = []
    monkeypatch.setattr(ap, "schedule_facts",
                        lambda slug, port_of, token, facts, **kw: given.append((slug, port_of(), token, facts)))
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


# ── An agent that already has a setup ─────────────────────────────────


def _setup(owner, docs, **groups) -> dict:
    """A reviewed preset with *docs* waiting under its id."""
    preset_id = ap.save_docs(owner, docs, "support")
    return {**ap.normalise(groups), "id": preset_id,
            "knowledge": [{"doc": d["name"], "title": d["name"].split(".")[0].title(), "summary": ""} for d in docs]}


def test_a_second_setup_adds_to_the_first(deck, tmp_path):
    agent, ws = tmp_path / "agent", tmp_path / "agent" / "ws"
    agent.mkdir()
    ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "policy.md", "text": "Old policy."},
                                         {"name": "prices.md", "text": "Prices."}],
                                   rules=[{"text": "Old rule."}], glossary=[{"term": "RMA", "meaning": "a return"}],
                                   intents=[{"name": "refund", "when": "Refunds.", "steps": ["a"]}]))
    done = ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "policy.md", "text": "New policy."},
                                                {"name": "faq.md", "text": "FAQ."}],
                                          rules=[{"text": "New rule."}, {"text": "old rule."}],
                                          intents=[{"name": "quote", "when": "Quotes.", "steps": ["b"]}]))
    kn = ws / "knowledge"
    assert (kn / "policy.md").read_text().endswith("New policy.\n")            # given again: replaced
    assert (kn / "prices.md").is_file() and (kn / "faq.md").is_file()
    index = (kn / "INDEX.md").read_text()
    assert all(f"({n}.md)" in index for n in ("policy", "prices", "faq")) and index.count("(policy.md)") == 1
    assert "RMA" in (kn / "glossary.md").read_text()                           # earlier groups stay
    rules = (kn / "owner-rules.md").read_text()
    assert rules.index("New rule.") < rules.index("old rule.") and "Old rule." not in rules   # the newer wording
    skills = agent / "data" / "home-config-parent" / ".captain-claw" / "skills"
    assert (skills / "refund" / "SKILL.md").is_file() and (skills / "quote" / "SKILL.md").is_file()
    assert done["counts"]["rules"] == 2 and done["totals"]["rules"] == 2 and done["totals"]["intents"] == 2
    assert done["totals"]["knowledge"] == 3 and "refund" in done["block"] and "quote" in done["block"]
    record = ap.read_record(agent)
    assert sorted(e["doc"] for e in record["knowledge"]) == ["faq.md", "policy.md", "prices.md"]


def test_a_setup_can_replace_the_one_before(deck, tmp_path):
    agent, ws = tmp_path / "agent", tmp_path / "agent" / "ws"
    agent.mkdir()
    (ws / "knowledge").mkdir(parents=True)
    (ws / "knowledge" / "my-notes.md").write_text("The agent's own file.")
    ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "manual.md", "text": "## Page 1\n" + "x" * 300_000}],
                                   rules=[{"text": "Old rule."}],
                                   intents=[{"name": "refund", "when": "Refunds.", "steps": ["a"]}]))
    assert len(list((ws / "knowledge").glob("manual-part-*.md"))) == 2
    done = ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "faq.md", "text": "FAQ."}], rules=[{"text": "New rule."}]),
                   mode="replace")
    kn = ws / "knowledge"
    assert sorted(p.name for p in kn.iterdir()) == ["INDEX.md", "faq.md", "my-notes.md", "owner-rules.md"]
    assert not (agent / "data" / "home-config-parent" / ".captain-claw" / "skills" / "refund").exists()
    assert "Old rule." not in (kn / "owner-rules.md").read_text()
    assert done["totals"] == {"rules": 1, "escalation": 0, "templates": 0, "glossary": 0, "intents": 0,
                              "facts": 0, "knowledge": 1}


def test_a_setup_from_before_records_is_read_back_from_its_files(deck, tmp_path):
    agent, ws = tmp_path / "agent", tmp_path / "agent" / "ws"
    agent.mkdir()
    ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "manual.md", "text": "## Page 1\n" + "x" * 300_000},
                                         {"name": "faq.md", "text": "FAQ."}],
                                   rules=[{"text": "Old rule."}], escalation=[{"text": "Ask first."}],
                                   templates=[{"name": "Offer", "body": "Dear [name],\n\nOffer."}],
                                   glossary=[{"term": "RMA", "meaning": "a return"}],
                                   intents=[{"name": "refund", "when": "Refunds.", "steps": ["a"]}]))
    (agent / ".owner-setup.json").unlink()                     # as #263 left it: no record
    skills = agent / "data" / "home-config-parent" / ".captain-claw" / "skills"
    (skills / "my-own").mkdir()
    (skills / "my-own" / "SKILL.md").write_text("---\nname: my-own\n---\nNot from a setup.")
    back = ap.earlier_setup(agent, ws)
    assert [r["text"] for r in back["rules"]] == ["Old rule."] and back["escalation"] == [
        {"text": "Ask first.", "source": {"doc": "instructions", "section": ""}}]
    assert back["templates"] == [{"name": "Offer", "body": "Dear [name],\n\nOffer."}]
    assert back["glossary"] == [{"term": "RMA", "meaning": "a return",
                                 "source": {"doc": "instructions", "section": ""}}] and back["skills"] == ["refund"]
    manual = next(e for e in back["knowledge"] if e["file"] == "manual.md")
    assert manual["parts"] == 2 and manual["files"] == ["manual.md", "manual-part-01.md", "manual-part-02.md"]

    done = ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "new.md", "text": "New."}], rules=[{"text": "New rule."}]))
    assert done["totals"]["knowledge"] == 3 and done["totals"]["rules"] == 2 and done["totals"]["intents"] == 1
    assert "(manual.md)" in (ws / "knowledge" / "INDEX.md").read_text()
    ap.seed(agent, ws, BOB, _setup(BOB, [{"name": "only.md", "text": "Only."}]), mode="replace")
    assert not (ws / "knowledge" / "manual-part-01.md").exists() and not (skills / "refund").exists()
    assert (skills / "my-own" / "SKILL.md").is_file()           # not the setup's: left alone


async def test_applying_twice_keeps_one_block_and_says_what_it_has(deck, monkeypatch):
    slug = await _spawn(BOB, "Helpdesk", 24303)
    monkeypatch.setattr(ap, "schedule_facts", lambda *a, **k: None)
    first = _setup(BOB, [{"name": "a.md", "text": "A."}], rules=[{"text": "Rule one."}])
    assert (await _apply(BOB, slug, first)).status_code == 200
    second = _setup(BOB, [{"name": "b.md", "text": "B."}], rules=[{"text": "Rule two."}])
    r = await _apply(BOB, slug, second)
    body = r.json()
    stored = json.loads(await deck.get_setting(BOB, INSTR))[slug]
    assert stored == body["instructions"] and stored.count(ap._BLOCK_OPEN) == 1
    assert "Rule one." in stored and "Rule two." in stored and stored.startswith("You answer customers kindly.")
    assert body["totals"]["knowledge"] == 2 and body["counts"]["knowledge"] == 1 and body["stored"] is True
    async with _client() as c:
        cur = (await c.get(f"/fd/agent-preset/current/{slug}", headers=_auth(BOB))).json()
        assert (await c.get(f"/fd/agent-preset/current/{slug}", headers=_auth(CAROL))).status_code == 404
    assert cur["has"] is True and cur["rules"] == 2 and [k["title"] for k in cur["knowledge"]] == ["A", "B"]

    r = await _apply(BOB, slug, {**_setup(BOB, []), "rules": []} | {"id": ""}, mode="replace")
    assert r.json()["instructions"] == "You answer customers kindly."


async def test_an_agent_that_is_off_gets_its_facts_when_it_next_starts(deck, monkeypatch, tmp_path):
    slug = await _spawn(BOB, "Helpdesk", 24304)
    monkeypatch.setattr(server, "_process_is_alive", lambda s: False)
    scheduled: list = []
    monkeypatch.setattr(ap, "schedule_facts", lambda *a, **k: scheduled.append(a))
    preset = _setup(BOB, [], facts=[{"content": "Refund window is 14 days.", "category": "fact"}])
    assert (await _apply(BOB, slug, preset)).json()["running"] is False
    pending = ap.pending_path(server.DATA_DIR / slug)
    assert scheduled == [] and json.loads(pending.read_text())[0]["content"] == "Refund window is 14 days."

    from captain_claw import insights

    class _Mgr:
        def __init__(self):
            self.added: list = []

        async def add(self, content, category, **kw):
            if content in [a[0] for a in self.added]:
                return None                                    # deduplicated, as the real one does
            self.added.append((content, category, kw.get("tags")))
            return "id"

    ap.leave_facts(server.DATA_DIR / slug, [{"content": "Refund window is 14 days.", "category": "fact",
                                             "importance": 5}])
    mgr = _Mgr()
    assert await insights.import_pending(mgr, pending) == 1
    assert mgr.added == [("Refund window is 14 days.", "fact", "owner-setup")] and not pending.exists()
    assert await insights.import_pending(mgr, pending) == 0


async def test_facts_an_agent_never_took_wait_for_its_next_start(monkeypatch, tmp_path):
    async def never(port_of, token, timeout):
        return 0

    monkeypatch.setattr(ap, "_wait_ready", never)
    facts = ap.normalise(_REPLY)["facts"]
    assert await ap.give_facts("a", lambda: 0, "", facts, agent_dir=tmp_path) == 0
    assert len(json.loads(ap.pending_path(tmp_path).read_text())) == 2


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


# ── Where things came from ────────────────────────────────────────────


def test_rules_and_terms_say_which_document_they_came_from(tmp_path):
    preset = ap.normalise({
        "rules": [{"text": "Refunds within 14 days.", "source": {"doc": "policy.pdf", "section": "Refunds"}},
                  {"text": "Be brief.", "source": {"doc": "instructions"}}],
        "escalation": [{"text": "Ask before discounts over 10%.", "source": {"doc": "prices.xlsx"}}],
        "glossary": [{"term": "RMA", "meaning": "a return", "source": {"doc": "policy.pdf", "section": "p. 3"}}],
    })
    folder, skills = tmp_path / "knowledge", tmp_path / "skills"
    ap.write_knowledge(folder, preset, [])
    rules = (folder / "owner-rules.md").read_text()
    assert "- Refunds within 14 days. (from policy.pdf — Refunds)" in rules
    assert "- Be brief.\n" in rules                               # the owner's own instructions: no "from"
    assert "- Ask before discounts over 10%. (from prices.xlsx)" in rules
    assert "- **RMA** — a return (from policy.pdf — p. 3)" in (folder / "glossary.md").read_text()

    # An agent set up before setups were recorded is read back from these files, sources and all.
    back = ap._record_from_files(skills, folder)
    assert back["rules"] == [{"text": "Refunds within 14 days.", "source": {"doc": "policy.pdf", "section": "Refunds"}},
                             {"text": "Be brief.", "source": {"doc": "instructions", "section": ""}}]
    assert back["escalation"][0]["source"] == {"doc": "prices.xlsx", "section": ""}
    assert back["glossary"] == [{"term": "RMA", "meaning": "a return",
                                 "source": {"doc": "policy.pdf", "section": "p. 3"}}]


def test_a_fact_keeps_its_document_as_the_agents_insight_does():
    assert ap.fact_source({"source": {"doc": "price-list.pdf", "section": "page 4"}}) == {
        "source_file": "price-list.pdf", "source_section": "page 4"}
    assert ap.fact_source({"source": {"doc": "price-list.pdf", "section": ""}}) == {"source_file": "price-list.pdf"}
    assert ap.fact_source({"source": {"doc": "instructions", "section": "x"}}) == {}
    assert ap.fact_source({}) == {}


async def test_facts_reach_the_agent_with_their_document(monkeypatch):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(201, json={"id": "x"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    facts = [{"content": "Shipping is free over 50 EUR.", "category": "fact", "importance": 7,
              "source": {"doc": "price-list.pdf", "section": "page 4"}},
             {"content": "The owner is Ana.", "category": "fact", "importance": 5, "source": {"doc": "instructions"}}]
    assert await ap._post_facts(24500, "tok", facts) == 2
    assert seen[0]["source_file"] == "price-list.pdf" and seen[0]["source_section"] == "page 4"
    assert "source_file" not in seen[1]


async def test_facts_left_for_an_agent_thats_off_keep_their_document(tmp_path):
    from captain_claw import insights
    facts = [{"content": "Shipping is free over 50 EUR.", "category": "fact", "importance": 7,
              "source": {"doc": "price-list.pdf", "section": "page 4"}}]
    ap.leave_facts(tmp_path, facts)
    waiting = json.loads(ap.pending_path(tmp_path).read_text())
    assert waiting[0]["source_file"] == "price-list.pdf"
    mgr = insights.InsightsManager(tmp_path / "insights.db")
    try:
        assert await insights.import_pending(mgr, ap.pending_path(tmp_path)) == 1
        got = (await mgr.list_recent(limit=1))[0]
        assert got["source_file"] == "price-list.pdf" and got["source_section"] == "page 4"
        assert got["source_tool"] == "owner_setup"
    finally:
        await mgr.close()
