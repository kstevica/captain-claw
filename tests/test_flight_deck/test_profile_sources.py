"""The material behind the owner profile, and the researcher that drafts it.

Pinned here:

* ``/fd/profile/sources`` is the signed-in user's own (401 without a session,
  an admin's act-as ignored); a PUT is partial; links are normalised to
  http(s) URLs with a label per network, repeats dropped, bad ones a 400;
* documents are read in full when uploaded and kept on disk under the owner's
  folder; a kind that can't be read is a warning, not a failure; a removed one
  is gone from disk; the index never trusts a stray id;
* the generic settings routes can't read the ``fd:tenant-profile:sources`` key;
* the researcher: refuses with nothing to read from, prefers the team's set
  (its thorough tier) over the person's own, reads documents and pages, skips
  private addresses, searches when a key is among the team's extra keys,
  clips the drafts to the profile caps and keeps the result; a second start
  while one runs is a 409; the result survives in the kept record.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import profile_research as pr
from captain_claw.flight_deck import profile_sources as ps
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.file_library import ModelReply

ADMIN = "user-admin"
BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role, name in ((ADMIN, "admin", "Ada Admin"), (BOB, "user", "Bob Builder")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", name, role))
    await db._db.commit()
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(pr, "_JOBS", {})
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


# ── links and the company ─────────────────────────────────────────────


class TestLinks:
    def test_urls_are_normalised_and_labelled(self):
        assert ps.normalize_url("linkedin.com/in/bob") == "https://linkedin.com/in/bob"
        assert ps.normalize_url("  https://www.github.com/bob?tab=repos ") == "https://www.github.com/bob?tab=repos"
        assert ps.label_for("https://www.linkedin.com/in/bob") == "LinkedIn"
        assert ps.label_for("https://x.com/bob") == "X"
        assert ps.label_for("https://bridges.example/team") == "bridges.example"

    @pytest.mark.parametrize("bad", ["", "   ", "ftp://x.example/a", "javascript:alert(1)", "notaurl", "x" * 501])
    def test_bad_links_say_why(self, bad):
        with pytest.raises(ps.SourceError):
            ps.normalize_url(bad)

    def test_clean_links_drops_repeats_and_caps(self):
        links = ps.clean_links(["github.com/bob", {"url": "https://github.com/bob", "label": "Code"},
                                {"url": "https://bob.example", "label": "  My   site "}], 8)
        assert links == [{"url": "https://github.com/bob", "label": "GitHub"},
                         {"url": "https://bob.example/", "label": "My site"}]
        with pytest.raises(ps.SourceError):
            ps.clean_links([f"https://s{i}.example" for i in range(9)], 8)


class TestSources:
    async def test_needs_a_session_and_ignores_act_as(self, deck):
        async with _client() as c:
            anon = await c.get("/fd/profile/sources")
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Bridges Ltd."})
            got = await c.get("/fd/profile/sources", headers=_hdr(ADMIN, BOB))
        assert anon.status_code == 401
        assert got.status_code == 200 and got.json()["company_name"] == ""
        assert got.json()["name"] == "Ada Admin"

    async def test_empty_to_start_with(self, deck):
        async with _client() as c:
            r = await c.get("/fd/profile/sources", headers=_hdr(BOB))
        body = r.json()
        assert body["name"] == "Bob Builder" and body["email"] == f"{BOB}@x.co"
        assert body["can_rename"] is True
        assert body["links"] == [] and body["company_links"] == [] and body["documents"] == []
        assert body["company_name"] == ""
        assert body["caps"]["links"] == ps.MAX_LINKS and ".pdf" in body["kinds"]

    async def test_put_is_partial_and_normalises(self, deck):
        async with _client() as c:
            r1 = await c.put("/fd/profile/sources", headers=_hdr(BOB), json={
                "links": ["linkedin.com/in/bob", "https://github.com/bob"], "company_name": "  Bridges  Ltd. "})
            r2 = await c.put("/fd/profile/sources", headers=_hdr(BOB), json={
                "company_links": [{"url": "bridges.example", "label": "Site"}]})
        assert r1.status_code == 200, r1.text
        body = r2.json()
        assert body["links"] == [{"url": "https://linkedin.com/in/bob", "label": "LinkedIn"},
                                 {"url": "https://github.com/bob", "label": "GitHub"}]
        assert body["company_name"] == "Bridges Ltd."
        assert body["company_links"] == [{"url": "https://bridges.example/", "label": "Site"}]
        stored = json.loads(await deck.get_setting(BOB, ps.SOURCES_SETTING))
        assert stored["company_name"] == "Bridges Ltd."

    async def test_bad_fields_are_a_400_and_change_nothing(self, deck):
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Kept"})
            bad_link = await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"links": ["ftp://x.example"]})
            bad_name = await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "x" * 200})
            not_obj = await c.put("/fd/profile/sources", headers=_hdr(BOB), json=["x"])
        assert bad_link.status_code == 400 and "http" in bad_link.json()["detail"]
        assert bad_name.status_code == 400 and not_obj.status_code == 400
        assert (await ps.load(deck, BOB))["company_name"] == "Kept"

    async def test_generic_settings_routes_cant_reach_it(self, deck):
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Secret Co."})
            r = await c.put("/fd/settings", headers=_hdr(BOB), json={"settings": {ps.SOURCES_SETTING: "{}"}})
            got = await c.get("/fd/settings", headers=_hdr(BOB))
        assert r.status_code in (400, 403)
        assert "Secret Co." not in got.text
        assert (await ps.load(deck, BOB))["company_name"] == "Secret Co."


# ── documents ─────────────────────────────────────────────────────────


class TestDocuments:
    async def test_upload_reads_keeps_and_removes(self, deck):
        cv = "# Bob Builder\n\nCivil engineer, 12 years of bridges.\n" * 20
        async with _client() as c:
            up = await c.post("/fd/profile/documents", headers=_hdr(BOB), files=[
                ("files", ("cv.md", cv.encode(), "text/markdown")),
                ("files", ("photo.png", b"\x89PNG....", "image/png")),
                ("files", ("empty.txt", b"", "text/plain")),
            ])
            assert up.status_code == 200, up.text
            body = up.json()
            docs = body["documents"]
            assert [d["name"] for d in docs] == ["cv.md"]
            assert docs[0]["chars"] == len(cv.strip()) and docs[0]["pages"] >= 1
            assert len(body["warnings"]) == 2
            assert any("photo.png" in w for w in body["warnings"])
            doc_id = docs[0]["id"]
            folder = server.DATA_DIR / ".profile-sources" / BOB / doc_id
            assert (folder / "text.txt").read_text(encoding="utf-8").startswith("# Bob Builder")
            assert (folder / "original.md").is_file()
            assert ps.document_text(BOB, doc_id).startswith("# Bob Builder")
            # Someone else never sees or removes it.
            other = await c.delete(f"/fd/profile/documents/{doc_id}", headers=_hdr(ADMIN))
            assert other.status_code == 404 and folder.is_dir()
            gone = await c.delete(f"/fd/profile/documents/{doc_id}", headers=_hdr(BOB))
        assert gone.status_code == 200 and gone.json()["documents"] == []
        assert not folder.exists()

    async def test_at_most_ten_documents(self, deck):
        async with _client() as c:
            for n in range(ps.MAX_DOCUMENTS):
                await c.post("/fd/profile/documents", headers=_hdr(BOB),
                             files=[("files", (f"d{n}.txt", b"some text about bob " * 10, "text/plain"))])
            r = await c.post("/fd/profile/documents", headers=_hdr(BOB),
                             files=[("files", ("one-more.txt", b"more text " * 10, "text/plain"))])
        assert len(r.json()["documents"]) == ps.MAX_DOCUMENTS
        assert r.json()["warnings"] and "at most" in r.json()["warnings"][0]

    def test_index_never_trusts_a_stray_id(self):
        parsed = ps.parse(json.dumps({"documents": [{"id": "../../etc", "name": "x"}, {"id": "a" * 24, "name": "ok"}]}))
        assert [d["id"] for d in parsed["documents"]] == ["a" * 24]
        assert ps._doc_dir(BOB, "../x") is None


# ── the researcher ────────────────────────────────────────────────────


_TEAM_SET = {
    "sets": [{"id": "team", "name": "Team", "tiers": {
        "fast": {"provider": "anthropic", "model": "claude-haiku", "api_key": "@system"},
        "thorough": {"provider": "anthropic", "model": "claude-opus", "api_key": "@system", "input_ctx": 200000},
    }, "envVars": [{"key": "BRAVE_API_KEY", "value": "brave-team-key"}]}],
    "defaultSetId": "team",
}


class _Page:
    def __init__(self, status: int, text: str, ctype: str = "text/html; charset=utf-8"):
        self.status_code = status
        self.content = text.encode()
        self.headers = {"content-type": ctype}
        self.encoding = "utf-8"

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]


class _Web:
    """The web as the run sees it: pages by URL, Brave by query."""

    def __init__(self):
        self.pages: dict[str, _Page] = {}
        self.searches: dict[str, list[dict]] = {}
        self.fetched: list[str] = []
        self.queries: list[str] = []
        self.search_headers: list[dict] = []

    async def get(self, url, params=None, headers=None, timeout=None):
        if url.startswith("https://api.search.brave.com/"):
            self.queries.append(params["q"])
            self.search_headers.append(dict(headers or {}))
            results = self.searches.get(params["q"], [])
            return _Page(200, json.dumps({"web": {"results": results}}), "application/json")
        self.fetched.append(url)
        return self.pages.get(url, _Page(404, "nothing here"))

    async def post(self, *a, **k):
        raise AssertionError("not Tavily")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch):
    fake = _Web()
    monkeypatch.setattr(pr, "_http_client", lambda: fake)
    # Every host is public in the tests (no DNS).
    from captain_claw import deck_confine

    monkeypatch.setattr(deck_confine, "url_public", lambda url: "10.0.0.5" not in url and "localhost" not in url)
    return fake


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch):
    """The model, answering with the three texts; records what it was asked."""
    calls: list[dict] = []
    reply = {"about_me": "I am Bob, a civil engineer who builds bridges.",
             "company": "Bridges Ltd. designs footbridges.\n\nIt writes plainly.",
             "instructions": "Answer in English.\nUse metric units.", "notes": ["LinkedIn could not be read."]}

    async def fake_reply(tier, system, user, owner_id, max_tokens=4000, request_timeout=None, images=None):
        calls.append({"tier": tier, "system": system, "user": user, "owner": owner_id})
        return ModelReply(json.dumps(reply), 1234)

    monkeypatch.setattr("captain_claw.flight_deck.file_library.model_reply", fake_reply)
    return {"calls": calls, "reply": reply}


async def _finished(owner: str, timeout: float = 5.0) -> pr.Job:
    job = pr.current(owner)
    assert job is not None
    await asyncio.wait_for(job.task, timeout)
    return job


class TestResearch:
    async def test_refuses_with_nothing_to_read_from(self, deck, web, model):
        async with _client() as c:
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
            idle = await c.get("/fd/profile/research", headers=_hdr(BOB))
        assert r.status_code == 400 and "nothing to research" in r.json()["detail"]
        assert idle.json() == {"state": "idle"}

    async def test_refuses_without_a_model(self, deck, web, model, monkeypatch):
        async def none(db, owner_id):
            return None, "", []

        monkeypatch.setattr(pr, "pick_model", none)
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Bridges Ltd."})
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
        assert r.status_code == 400 and "model" in r.json()["detail"]
        assert pr.current(BOB) is None

    async def test_reads_documents_pages_and_the_web_on_the_team_model(self, deck, web, model):
        await deck.set_system_setting("fd:shared-tier-sets", json.dumps(_TEAM_SET))
        web.pages["https://bob.example/"] = _Page(200, "<html><title>Bob</title><body><h1>Bob Builder</h1><p>" + "I build bridges and teach. " * 20 + "</p></body></html>")
        web.pages["https://linkedin.com/in/bob"] = _Page(999, "sign in")
        web.pages["https://bridges.example/"] = _Page(200, "<html><body><p>" + "Bridges Ltd. makes footbridges for towns. " * 10 + "</p></body></html>")
        web.pages["https://bridges.example/about"] = _Page(200, "<html><body><p>" + "Founded 2014 in Zagreb, 40 people. " * 10 + "</p></body></html>")
        web.pages["https://news.example/bob"] = _Page(200, "<html><body><p>" + "Bob Builder won the bridge prize. " * 10 + "</p></body></html>")
        web.pages["http://10.0.0.5/admin"] = _Page(200, "<html><body><p>" + "secret internal page " * 20 + "</p></body></html>")
        web.searches['"Bob Builder"'] = [
            {"title": "Bob Builder wins prize", "url": "https://news.example/bob", "description": "Bob Builder won"},
            {"title": "Unrelated", "url": "https://other.example/x", "description": "nothing"},
        ]
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={
                "links": ["https://bob.example", "linkedin.com/in/bob", "http://10.0.0.5/admin"],
                "company_name": "Bridges Ltd.", "company_links": ["https://bridges.example"]})
            await c.post("/fd/profile/documents", headers=_hdr(BOB),
                         files=[("files", ("cv.txt", b"Bob Builder, civil engineer. Speaks Croatian and English.", "text/plain"))])
            await c.put("/fd/profile", headers=_hdr(BOB), json={"instructions": "Keep it short."})
            started = await c.post("/fd/profile/research", headers=_hdr(BOB))
            assert started.status_code == 202, started.text
            assert started.json()["state"] == "running" and started.json()["model"] == "claude-opus"
            job = await _finished(BOB)
            got = await c.get("/fd/profile/research", headers=_hdr(BOB))
        assert job.state == "done", job.log
        body = got.json()
        assert body["state"] == "done" and body["model_source"] == "team" and body["tokens"] == 1234
        assert body["drafts"] == {k: v for k, v in model["reply"].items() if k != "notes"}
        assert body["notes"] == ["LinkedIn could not be read."]
        kinds = {(s["kind"], s["name"], s["ok"]) for s in body["sources"]}
        assert ("document", "cv.txt", True) in kinds
        assert ("link", "bob.example", True) in kinds
        assert ("link", "LinkedIn", False) in kinds
        assert ("link", "bridges.example", True) in kinds
        assert ("link", "bridges.example /about", True) in kinds
        assert ("page", "Bob Builder wins prize", True) in kinds or any(s["kind"] == "page" for s in body["sources"])
        # The private address was never fetched; the walled host's reason is said.
        assert "http://10.0.0.5/admin" not in web.fetched
        assert any("private address" in e["message"] for e in body["log"])
        assert any("LinkedIn" in e["message"] and "visitors" in e["message"] for e in body["log"])
        # The search used the team's key and asked about the person and the company.
        assert web.search_headers[0]["X-Subscription-Token"] == "brave-team-key"
        assert '"Bob Builder"' in web.queries and any("Bridges Ltd." in q for q in web.queries)
        # The model saw the material, the current text, and the caps.
        call = model["calls"][0]
        assert call["tier"]["model"] == "claude-opus"
        assert "Document: cv.txt" in call["user"] and "Speaks Croatian" in call["user"]
        assert "Company page: Site" not in call["user"]  # labels come from the host
        assert "Founded 2014" in call["user"] and "bridge prize" in call["user"]
        assert "secret internal page" not in call["user"]
        assert "Keep it short." in call["user"]
        assert "at most 1500 characters" in call["system"] and "at most 4000 characters" in call["system"]
        # Nothing was saved to the profile itself.
        assert (await tp.load_profile(deck, BOB))["about_me"] == ""
        # The result is kept.
        kept = json.loads(await deck.get_setting(BOB, pr.RESULT_SETTING))
        assert kept["state"] == "done" and kept["drafts"]["about_me"].startswith("I am Bob")

    async def test_falls_back_to_own_tiers_and_says_without_search(self, deck, web, model, monkeypatch):
        await deck.set_settings(BOB, {"fd:forge-tiers": json.dumps({"sets": [{"id": "s", "name": "Mine", "tiers": {
            "balanced": {"provider": "openai", "model": "gpt-5", "api_key": "sk-own"}}}], "activeSetId": "s"})})
        async with _client() as c:
            await c.post("/fd/profile/documents", headers=_hdr(BOB),
                         files=[("files", ("bio.txt", b"Bob Builder builds bridges for a living.", "text/plain"))])
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Bridges Ltd."})
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
            assert r.status_code == 202, r.text
            job = await _finished(BOB)
        assert job.state == "done" and job.model_source == "own" and job.model == "gpt-5"
        assert web.queries == []
        assert any("No web search key" in e["message"] for e in job.log)

    async def test_drafts_are_clipped_to_the_caps(self, deck, web, model):
        await deck.set_system_setting("fd:shared-tier-sets", json.dumps(_TEAM_SET))
        model["reply"]["about_me"] = "x" * 3000
        model["reply"]["company"] = "y" * 9000
        async with _client() as c:
            await c.post("/fd/profile/documents", headers=_hdr(BOB),
                         files=[("files", ("bio.txt", b"Bob Builder builds bridges.", "text/plain"))])
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
            assert r.status_code == 202
            job = await _finished(BOB)
        assert job.state == "done"
        assert len(job.drafts["about_me"]) == tp.CAPS["about_me"]
        assert len(job.drafts["company"]) == tp.CAPS["company"]
        assert job.drafts["about_me"].endswith("…")

    async def test_a_model_that_doesnt_answer_in_json_is_an_error_that_is_kept(self, deck, web, monkeypatch):
        await deck.set_system_setting("fd:shared-tier-sets", json.dumps(_TEAM_SET))

        async def prose(*a, **k):
            return ModelReply("Sure! Here is what I found about Bob…", 50)

        monkeypatch.setattr("captain_claw.flight_deck.file_library.model_reply", prose)
        async with _client() as c:
            await c.post("/fd/profile/documents", headers=_hdr(BOB),
                         files=[("files", ("bio.txt", b"Bob Builder builds bridges.", "text/plain"))])
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
            assert r.status_code == 202
            job = await _finished(BOB)
            got = await c.get("/fd/profile/research", headers=_hdr(BOB))
        assert job.state == "error" and "three texts" in job.error
        assert got.json()["state"] == "error" and got.json()["drafts"] is None

    async def test_nothing_readable_is_an_error(self, deck, web, model):
        await deck.set_system_setting("fd:shared-tier-sets", json.dumps(_TEAM_SET))
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"links": ["https://down.example/x"]})
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
            assert r.status_code == 202
            job = await _finished(BOB)
        assert job.state == "error" and "Nothing could be read" in job.error
        assert model["calls"] == []

    async def test_a_second_start_while_one_runs_is_a_409_and_stop_ends_it(self, deck, web, model, monkeypatch):
        await deck.set_system_setting("fd:shared-tier-sets", json.dumps(_TEAM_SET))
        gate = asyncio.Event()

        async def slow(job, *a, **k):
            job.stage = "reading"
            await gate.wait()

        monkeypatch.setattr(pr, "_run", slow)
        async with _client() as c:
            await c.put("/fd/profile/sources", headers=_hdr(BOB), json={"company_name": "Bridges Ltd."})
            first = await c.post("/fd/profile/research", headers=_hdr(BOB))
            again = await c.post("/fd/profile/research", headers=_hdr(BOB))
            polled = await c.get("/fd/profile/research", headers=_hdr(BOB))
            stopped = await c.delete("/fd/profile/research", headers=_hdr(BOB))
            job = pr.current(BOB)
            assert job is not None
            try:
                await asyncio.wait_for(job.task, 5)
            except asyncio.CancelledError:
                pass
            after = await c.get("/fd/profile/research", headers=_hdr(BOB))
            nothing = await c.delete("/fd/profile/research", headers=_hdr(BOB))
        assert first.status_code == 202
        assert again.status_code == 409 and "under way" in again.json()["detail"]
        assert polled.json()["state"] == "running" and polled.json()["stage"] == "reading"
        assert stopped.json() == {"stopped": True}
        assert after.json()["state"] == "cancelled"
        assert nothing.json() == {"stopped": False}
        # A new run may start now.
        gate.set()
        async with _client() as c:
            r = await c.post("/fd/profile/research", headers=_hdr(BOB))
        assert r.status_code == 202

    async def test_a_stale_running_record_reads_as_stopped(self, deck):
        await deck.set_settings(BOB, {pr.RESULT_SETTING: json.dumps({"state": "running", "log": []})})
        async with _client() as c:
            got = await c.get("/fd/profile/research", headers=_hdr(BOB))
        assert got.json()["state"] == "error" and "restarted" in got.json()["error"]

    def test_search_queries(self):
        qs = pr.search_queries("Bob Builder", "Bridges Ltd.", [], [{"url": "https://bridges.example/", "label": "Site"}])
        assert qs == ['"Bob Builder"', '"Bob Builder" Bridges Ltd.', "Bridges Ltd. company", "Bridges Ltd. bridges.example"]
        assert pr.search_queries("", "", [], []) == []

    def test_corpus_fits_the_room_and_keeps_the_trusted_material(self):
        parts = [{"heading": "Document: cv", "text": "cv " * 5000, "limit": 24000},
                 {"heading": "Their page: Site", "text": "page " * 5000, "limit": 10000},
                 {"heading": "Web search: q", "text": "web " * 5000, "limit": 4000}]
        text = pr.corpus({"name": "Bob", "links": [], "company_links": [], "current": {}}, parts, 20000)
        assert len(text) <= 20500
        assert "Document: cv" in text and "Their page: Site" in text
        assert text.index("Document: cv") < text.index("Their page: Site")
