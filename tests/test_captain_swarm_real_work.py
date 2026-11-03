"""Captain Swarm on real work: Captain's own browser for page checks, code
objectives without held-out cases, documents and research, calendar
invitations as a practice set, and the routes behind the page and the kiosk."""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from captain_claw.flight_deck import captain_swarm_browser as browser
from captain_claw.flight_deck import captain_swarm_document as document
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_measure as cm
from captain_claw.flight_deck import captain_swarm_objective as om
from captain_claw.flight_deck import captain_swarm_outcome as outcome
from captain_claw.flight_deck import captain_swarm_practice as practice
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck import captain_swarm_rules as rules
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

RID = "cs_0123456789ab"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    monkeypatch.setattr(loop, "sealed_dir", lambda rid: tmp_path / "sealed" / rid)
    yield s
    loop.set_store(None)
    run(s.close())


# ── Captain's browser ────────────────────────────────────────────────


def test_the_browser_refuses_what_isnt_public(monkeypatch):
    monkeypatch.setenv("FD_CAPTAIN_SWARM_BROWSER", "1")
    for url in ("http://127.0.0.1:25080/", "http://localhost:8765/x", "file:///etc/passwd", "http://10.0.0.1/"):
        text, why = run(browser.rendered_text(url))
        assert text == "" and "not a public web address" in why
    monkeypatch.setenv("FD_CAPTAIN_SWARM_BROWSER", "0")
    assert not browser.enabled()
    assert run(browser.rendered_text("https://example.com/"))[1] == "Captain's browser is off on this deck"


def test_look_tries_a_plain_fetch_first(monkeypatch):
    monkeypatch.setattr(browser, "enabled", lambda: True)
    seen = []

    async def fetch(url):
        return ("plain page" if "plain" in url else ""), url

    async def rendered(url):
        seen.append(url)
        return "rendered page", url
    monkeypatch.setattr(browser, "rendered_text", rendered)
    assert run(browser.look("https://plain.example/", fetch)) == ("plain page", "https://plain.example/", "fetch")
    assert run(browser.look("https://js.example/", fetch)) == ("rendered page", "https://js.example/", "browser")
    assert seen == ["https://js.example/"]


def test_an_outcome_check_looks_again_in_a_browser(store, monkeypatch):
    monkeypatch.setattr(browser, "enabled", lambda: True)
    pages = {"https://a.example/x": "The Book paperback, price loading…", "https://b.example/y": ""}

    async def fetch(url):
        return pages[url], url

    async def rendered(url):
        return ("The Book paperback 39,90 €", url) if "a.example" in url else ("", "the browser couldn't open it")
    monkeypatch.setattr(outcome, "fetch_text", fetch)
    monkeypatch.setattr(browser, "rendered_text", rendered)
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book"))

    async def go():
        await store.create_run(run_id=RID, owner_id="u", title="t", objective=obj, status="running")
        await store.add_version(version_id="v_1", run_id=RID, content={"candidates": [
            {"title": "The Book", "url": "https://a.example/x", "value": 39.9},
            {"title": "The Book", "url": "https://b.example/y", "value": 30.0}]})
        return await outcome.measure(await store.get_run(RID), "v_1", "dev")
    got = run(go())
    a, b = got["checks"]
    assert a["verified"] and a["via"] == "browser" and got["best"]["url"] == "https://a.example/x"
    assert not b["verified"] and "in a browser too" in b["why"]


# ── code without held-out cases ──────────────────────────────────────


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "src" / "x.py").write_text("X = 1\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "b"], check=True)
    return r


def test_a_code_objective_without_held_out_cases(tmp_path, store):
    repo = _repo(tmp_path)
    raw = {"kind": "code", "title": "faster", "subject": {"repo_path": str(repo), "editable": ["src/**"], "protected": []},
           "metric": {"kind": "command", "run": "cd {checkout} && python bench.py", "read": "ms", "direction": "min"},
           "holdout": {}, "lane": {"harness": "claude"}}
    obj = om.normalize(raw, owner_has_shell=True)
    assert obj["holdout"] == {"run": "cd {checkout} && python bench.py", "file": ""} and obj["family"] == "code"
    with pytest.raises(om.ObjectiveError, match="no held-out file"):
        om.normalize({**raw, "holdout": {"run": "x {checkout} {holdout}"}}, owner_has_shell=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run(loop.seal({"id": RID, "objective": obj}, sha))
    sealed = tmp_path / "sealed" / RID
    assert (sealed / "sealed.json").is_file() and not (sealed / "holdout.json").exists()


# ── documents and research ───────────────────────────────────────────


def test_document_objective():
    o = document.normalize(document.document_template(
        task="A brief on the EU AI Act for a small design studio", criteria=["Name the deadlines", " "], words=900))
    assert o["kind"] == "document" and o["family"] == "research" and o["subject"]["criteria"] == ["Name the deadlines"]
    assert o["subject"]["sources"] and o["lane"]["parallel"] == 3 and o["metric"]["repeat"] == 2
    w = document.normalize(document.document_template(task="A short welcome letter for new clients", sources=False))
    assert w["family"] == "writing" and not w["subject"]["sources"]
    from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
    with pytest.raises(ObjectiveError):
        document.normalize(document.document_template(task="too short"))
    assert loop.parts_for("document")["carry"] is False and "document" in loop.REVIEW_QUESTIONS


def test_grades_sources_and_citations():
    assert document.parse_grades('{"grades": [{"item": 1, "grade": "full"}, {"item": 2, "grade": "partial"}, '
                                 '{"item": 3, "grade": "none"}]}', 3) == [1.0, 0.5, 0.0]
    assert document.parse_grades('{"grades": [{"item": 1, "grade": "full"}]}', 4) is None    # too few to trust
    assert document.parse_grades("nope", 2) is None
    text = ("The Act applies from 2026 ([source](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)).\n\n"
            "See also https://artificialintelligenceact.eu/article/50/. And again https://eur-lex.europa.eu/eli/reg/2024/1689/oj")
    assert document.sources_of(text) == ["https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
                                          "https://artificialintelligenceact.eu/article/50/"]
    assert "applies from 2026" in document.citing_text(text, "https://eur-lex.europa.eu/eli/reg/2024/1689/oj")


def test_a_source_must_load_and_be_about_what_cites_it(monkeypatch):
    pages = {"https://ok.example/": "The regulation applies to providers and deployers of artificial intelligence "
                                    "systems placed on the market in the union. " * 5,
             "https://other.example/": "Recipes for summer salads with tomatoes and basil and olive oil. " * 5}

    async def look(url, fetch):
        return pages.get(url, ""), ("" if url in pages else "HTTP 404"), "fetch"
    monkeypatch.setattr(browser, "look", look)
    cited = "Providers of artificial intelligence systems on the union market must label generated content."
    assert run(document.check_source("https://ok.example/", cited))["ok"]
    off = run(document.check_source("https://other.example/", cited))
    assert not off["ok"] and "isn't about what cites it" in off["why"]
    gone = run(document.check_source("https://gone.example/", cited))
    assert not gone["ok"] and "404" in gone["why"]


def _doc_run(store, *, sources=True):
    obj = document.normalize(document.document_template(task="A brief on the EU AI Act for a small design studio",
                                                         sources=sources))

    async def go():
        await store.create_run(run_id=RID, owner_id="u", title=obj["title"], objective=obj, status="running")
        return await store.get_run(RID)
    return run(go())


def test_the_checklists_are_made_once_and_the_held_out_one_stays_sealed(store, monkeypatch):
    calls = []

    async def ask(run_, system, user, **kw):
        calls.append(kw.get("panel"))
        return json.dumps(["Says who the Act applies to", "Names the deadlines"] if not kw.get("panel")
                          else ["Is right about fines", "Says what is unknown"])
    monkeypatch.setattr(loop, "ask_model", ask)
    r = _doc_run(store)
    r["objective"]["subject"]["criteria"] = ["End with a checklist"]
    run(document.seal(r, "v_0"))
    run(document.seal(r, "v_0"))                                   # once: the second call keeps the first
    lists = document.checklists(RID)
    assert [bool(x) for x in calls] == [False, True]                   # the held-out one on the panel's model
    assert lists["open"][0] == "(The owner asks) End with a checklist" and len(lists["open"]) == 3
    assert lists["held"] == ["Is right about fines", "Says what is unknown"]
    assert "Is right about fines" not in document.brief(r, [])          # never shown to the planner


def test_measure_grades_the_open_or_held_out_checklist(store, monkeypatch):
    r = _doc_run(store)
    path = document.checklist_path(RID)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"open": ["a", "b"], "held": ["c", "d", "e"]}))
    asked = []

    async def ask(run_, system, user, **kw):
        asked.append(user)
        n = 2 if "1. a" in user else 3
        return json.dumps({"grades": [{"item": i, "grade": "full" if i == 1 else "partial"} for i in range(1, n + 1)]})
    monkeypatch.setattr(loop, "ask_model", ask)

    async def go():
        await store.add_version(version_id="v_1", run_id=RID, content={"text": "The document " * 100})
        await store.add_version(version_id="v_0", run_id=RID, content={"text": ""})
        return (await document.measure(r, "v_1", "dev"), await document.measure(r, "v_1", "holdout"),
                await document.measure(r, "v_0", "dev"))
    dev, hold, empty = run(go())
    assert dev["value"] == pytest.approx(0.75) and dev["stats"]["n"] == 2 and dev["per_case"] == {"c1": 1.0, "c2": 0.5}
    assert hold["value"] == pytest.approx(2 / 3) and hold["per_case"] == {}         # held-out grades stay sealed
    assert empty["value"] == 0.0 and len(asked) == 4                                  # nothing to grade: no call


def test_screen_wants_a_document_proof_and_sources_that_hold(store, monkeypatch):
    async def no_leaks(prefix, since, until=None):
        return []
    monkeypatch.setattr(rehearsal, "leaks", no_leaks)
    holds = {"https://good.example/": True, "https://bad.example/": False}

    async def check(url, cited):
        return {"url": url, "ok": holds.get(url, False), "via": "fetch", "why": "" if holds.get(url) else "HTTP 404"}
    monkeypatch.setattr(document, "check_source", check)
    r = _doc_run(store)
    body = "Words about the act and what it asks of a studio. " * 30

    async def screen(text):
        vid = f"v_{abs(hash(text)) % 10**8:08d}"
        await store.add_version(version_id=vid, run_id=RID, content={"text": text})
        return await document.screen(r, {"head_sha": vid, "child_id": "bat_000000000001", "created_at": 0})
    short = run(screen("Too short."))
    assert not short["ok"] and "almost nothing" in short["reason"]
    nosrc = run(screen(body))
    assert not nosrc["ok"] and "cites no sources" in nosrc["reason"]
    bad = run(screen(body + "\n\nhttps://good.example/ and https://bad.example/"))
    assert not bad["ok"] and "1 of 2 failed" in bad["reason"]
    good = run(screen(body + "\n\nhttps://good.example/"))
    assert good["ok"] and "1 of 1 sources hold" in good["diff_stat"]


def test_reading_the_attempts_document(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    folder = tmp_path / "vfs" / "u" / "bat-bat_0000"
    folder.mkdir(parents=True)
    (folder / "notes.md").write_text("short notes")
    (folder / "draft.md").write_text("a longer draft " * 50)
    bat = {"id": "bat_000000000001", "truth": ""}
    assert document.read_document("u", bat)[1] == "draft.md"
    (folder / "document.md").write_text("# The document\n" + "body " * 10)
    assert document.read_document("u", bat) == ("# The document\n" + "body " * 10, "document.md")
    empty = {"id": "bat_999999999999", "truth": "Here it is:\n" + "word " * 120 + '```json\n{"document": "x"}\n```'}
    text, where = document.read_document("u", empty)
    assert where == "(its answer)" and "```json" not in text
    assert "vfs:bat-bat_0000/document.md" in document.DocumentLane().task(
        {"objective": document.normalize(document.document_template(task="A brief on the EU AI Act for a studio"))},
        {"hypothesis": "start from the regulation's text", "why": ""}, "bat_000000000001")


def test_a_document_run_end_to_end(store, tmp_path, monkeypatch):
    """The driver with the document kind's own measure and screen, and the routes after it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck import vfs_routes
    from captain_claw.flight_deck.auth import get_current_user

    monkeypatch.setattr(loop, "_POLL_S", 0)
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)

    async def no_leaks(prefix, since, until=None):
        return []
    monkeypatch.setattr(rehearsal, "leaks", no_leaks)
    good = "A clear brief on the AI Act for a design studio, with dates and duties. " * 20
    thin = "Some words about rules. " * 20

    async def ask(run_, system, user, **kw):
        if kw.get("label") == "checklist":
            return json.dumps(["x", "y"])
        full = good[:60] in user
        return json.dumps({"grades": [{"item": i, "grade": "full" if full else "partial"} for i in (1, 2)]})
    monkeypatch.setattr(loop, "ask_model", ask)

    class Lane:
        async def start(self, run_, exp, brief):
            return {"child_id": "bat_000000000001" if "clear" in exp["hypothesis"] else "bat_000000000002", "branch": ""}

        async def poll(self, run_, exp):
            vid = f"v_{exp['id'][-6:]}"
            await store.add_version(version_id=vid, run_id=run_["id"], content={
                "text": good if "clear" in exp["hypothesis"] else thin, "path": "document.md"})
            return {"state": "done", "head_sha": vid, "commits": 1}

        async def stop(self, run_, exp):
            return None

        async def discard(self, run_, exp):
            return None

    async def propose(run_, ledger, k):
        return [] if ledger else [{"hypothesis": "write it clear, with dates", "strategy_tag": "clear", "files": [], "why": ""},
                                  {"hypothesis": "write it thin", "strategy_tag": "thin", "files": [], "why": ""}]

    async def review(run_, exp):
        return {"ok": True, "reason": "answers the brief"}

    async def no_strategy(run_, ledger):
        return []
    for setter, fn in ((loop.set_lane, Lane()), (loop.set_proposer, propose), (loop.set_review, review),
                       (loop.set_strategy, no_strategy)):
        setter(fn)
    obj = document.normalize({**document.document_template(task="A brief on the EU AI Act for a small design studio",
                                                           sources=False), "lane": {"parallel": 2, "per_round": 2}})
    try:
        async def go():
            await store.create_run(run_id=RID, owner_id="u", title=obj["title"], objective=obj, status="baseline")
            return await loop.Driver(store, RID).drive()
        status = run(go())
    finally:
        for setter in (loop.set_lane, loop.set_proposer, loop.set_review, loop.set_strategy):
            setter(None)
        loop._SLOTS.clear()
    r = run(store.get_run(RID))
    exps = run(store.list_experiments(RID))
    assert [e["status"] for e in sorted(exps, key=lambda e: e["n"])] == ["kept", "discarded"]
    assert r["champion"]["dev"]["value"] == 1.0 and status == "stalled"

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "role": "user"}
    with TestClient(app) as c:
        view = c.get(f"/fd/captain-swarm/runs/{RID}").json()["document"]
        assert view["best"]["text"].startswith("A clear brief") and view["held_items"] == 2
        assert [x["grade"] for x in view["checklist"]] == [1.0, 1.0]
        saved = c.post(f"/fd/captain-swarm/runs/{RID}/save-document").json()["saved"]
        pb = c.get(f"/fd/captain-swarm/runs/{RID}/playbook").json()["playbook"]
    assert saved.startswith("vfs:captain-swarm/") and saved.endswith(".md")
    assert (tmp_path / "vfs" / "u" / "captain-swarm" / saved.split("/", 1)[1]).read_text().startswith("A clear brief")
    assert pb["do_pattern"].startswith("- write it clear") and pb["task_type"] == "document-processing"


# ── calendar invitations ─────────────────────────────────────────────


def _event(i, answer="accepted", *, own=False, start="2026-09-01T10:00:00Z", created="2026-08-20T09:00:00Z", series=""):
    return {"id": f"e{i}", "status": "confirmed", "summary": f"Meeting {i}", "created": created,
            "start": {"dateTime": start}, "end": {"dateTime": start.replace("10:00", "10:30")},
            "organizer": {"email": "me@x.example", "self": True} if own else {"email": f"o{i}@x.example", "displayName": f"O {i}"},
            **({"recurringEventId": series} if series else {}),
            "attendees": [{"email": "me@x.example", "self": True, "responseStatus": answer},
                          {"email": f"o{i}@x.example", "displayName": f"O {i}"}]}


def test_invite_case_labels_only_real_answers():
    yes = practice.invite_case(_event(1, "accepted"))
    assert yes["label"] is True and yes["duration_min"] == 30 and yes["organizer"] == "O 1 <o1@x.example>"
    assert yes["guests"] == 1 and yes["others"] == ["O 1"]
    assert practice.invite_case(_event(2, "declined"))["label"] is False
    assert practice.invite_case(_event(3, "needsAction")) is None
    assert practice.invite_case(_event(4, "tentative")) is None
    assert practice.invite_case(_event(5, own=True)) is None
    assert practice.invite_case({**_event(6), "status": "cancelled"}) is None


def test_build_calendar_invites():
    now_ms = 1_790_000_000_000
    day = 86_400_000

    def iso(ms):
        return practice._iso(ms)
    pages = [
        {"items": [_event(i, "accepted" if i % 2 else "declined", created=iso(now_ms - (100 - i * 3) * day),
                          start=iso(now_ms - (99 - i * 3) * day)) for i in range(12)], "nextPageToken": "p2"},
        {"items": [_event(20 + i, "accepted", created=iso(now_ms - (10 - i) * day), start=iso(now_ms - (9 - i) * day),
                          series="weekly") for i in range(3)]
         + [_event(30, own=True), _event(31, "needsAction")]},
    ]

    async def fetch(path, params):
        if path == "/calendars/primary":
            return 200, {"id": "me@x.example"}
        return 200, pages[1] if params.get("pageToken") == "p2" else pages[0]
    built = run(practice.build_calendar_invites(fetch, now_ms=now_ms, window_days=180, holdout_days=30))
    assert built["own"] == ["me@x.example"] and built["skipped"] == {"own": 1, "not answered": 1}
    assert len(built["open"]) == 12 and len(built["holdout"]) == 1          # the weekly series counts once
    assert built["holdout"][0]["recurring"]

    async def no_calendar(path, params):
        return 403, {}
    with pytest.raises(RuntimeError, match="calendar permission"):
        run(practice.build_calendar_invites(no_calendar, now_ms=now_ms))


def test_the_invitation_question_and_its_rules(store, monkeypatch):
    case = practice.invite_case(_event(1))
    q = cm.invite_question(case, "Ana")
    assert "Ana was invited" in q and "Organizer: O 1 <o1@x.example>" in q and "YES (they accept)" in q
    assert cm.question_for("calendar_invites") == (cm.invite_question, "f1_accepts")
    assert cm.question_for("mail_triage")[1] == "f1_needs_reply"
    asked = []

    async def ask(question):
        asked.append(question)
        return {"answer": "YES"}
    got = run(cm.run_triage([case], ask=ask, repeat=1, kind="calendar_invites"))
    assert got["metric"] == "f1_accepts" and "was invited" in asked[0]
    tpl = rules.triage_template(set_id="ci-1", agent="helper", agent_name="Helper", owner_name="Ana",
                                set_kind=practice.KIND_CALENDAR)
    obj = rules.normalize(tpl)
    assert obj["subject"]["set_kind"] == "calendar_invites" and obj["family"] == "calendar-invites"
    assert obj["metric"]["read"] == "f1_accepts" and "answers Ana's invitations" in obj["title"]
    assert rules.with_rules("Q", ["Decline all-day events."], "Ana", "calendar_invites").startswith("How Ana answers invitations")
    assert "o9@x.example" in rules.holdout_strings([practice.invite_case(_event(9))])
    seen = {}

    async def fake_ask_ideas(run_, system, brief):
        seen["system"] = system
        return "[]"
    monkeypatch.setattr(loop, "ask_ideas", fake_ask_ideas)

    async def go():
        await store.create_run(run_id=RID, owner_id="u", title="t", objective=obj, status="running")
        monkeypatch.setattr(practice, "load_cases", lambda owner, set_id, split: [])
        return await rules.propose(await store.get_run(RID), [], 2)
    run(go())
    assert "answers calendar invitations" in seen["system"] and "KIND of invitation" in seen["system"]


def test_routes_offer_documents_code_and_calendar_sets(store, tmp_path, monkeypatch):
    from unittest import mock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    monkeypatch.setattr(practice, "sealed_root", lambda: tmp_path / "practice")

    async def can_run(owner, objective=None, user=None):
        return None
    monkeypatch.setattr(routes, "_can_run", can_run)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "role": "user"}

    async def token(owner):
        return "tok"

    def fetch(client, tok):
        async def f(path, params):
            if path == "/calendars/primary":
                return 200, {"id": "me@x.example"}
            day = 86_400_000
            now = 1_790_000_000_000
            return 200, {"items": [_event(i, "accepted" if i % 2 else "declined",
                                          created=practice._iso(now - (100 - i * 8) * day)) for i in range(13)]}
        return f
    with mock.patch("captain_claw.flight_deck.google_oauth_routes.get_valid_google_access_token", token), \
            mock.patch.object(practice, "calendar_fetch", fetch), \
            mock.patch.object(routes, "_local_models", mock.AsyncMock(return_value=[])), \
            mock.patch.object(routes, "_default_holdout", lambda: ""), \
            mock.patch.object(routes, "_source_repo", lambda: ""), \
            mock.patch.object(routes.time, "time", lambda: 1_790_000_000.0), \
            TestClient(app) as c:
        built = c.post("/fd/captain-swarm/practice/calendar-invites", json={"window_days": 365, "holdout_days": 30})
        assert built.status_code == 200, built.text
        set_id = built.json()["id"]
        for _ in range(50):
            meta = c.get(f"/fd/captain-swarm/practice/{set_id}").json()
            if meta["state"] != "building":
                break
            asyncio.run(asyncio.sleep(0.02))
        assert meta["state"] == "ready", meta
        assert meta["kind"] == "calendar_invites" and set_id.startswith("ci-") and meta["label"] == "you accepted the invitation"
        tpl = c.get("/fd/captain-swarm/templates").json()["templates"]
        assert {"document", "code", "outcome", "triage", "mrav"} <= set(tpl)
        assert tpl["triage"]["sets"][0]["kind"] == "calendar_invites"
        # Every cap a person is offered is in tokens.
        assert all("budget_usd" not in t["defaults"] and "attempt_usd" not in t["defaults"] for t in tpl.values())
        assert tpl["document"]["defaults"]["attempt_tokens"] == document.ATTEMPT_TOKENS
        assert tpl["outcome"]["defaults"]["budget_tokens"] == outcome.BUDGET["tokens"]
        r = c.post("/fd/captain-swarm/runs", json={"template": "document", "options": {
            "task": "A short brief on the EU AI Act for a design studio", "criteria": "Name the deadlines\n- End with a checklist",
            "sources": "0", "family": "Studio Briefs", "attempt_tokens": "8M", "budget_tokens": "40m"}})
        assert r.status_code == 200, r.text
        got = c.get(f"/fd/captain-swarm/runs/{r.json()['id']}").json()
        assert got["kind"] == "document" and got["objective"]["subject"]["criteria"] == ["Name the deadlines", "End with a checklist"]
        assert got["objective"]["subject"]["sources"] is False and got["learning"]["family"] == "studio-briefs"
        assert got["objective"]["lane"]["attempt_tokens"] == 8_000_000 and got["objective"]["budget"]["tokens"] == 40_000_000
        bad = c.post("/fd/captain-swarm/runs", json={"template": "document", "options": {
            "task": "A short brief on the EU AI Act for a design studio", "budget_tokens": "a lot"}})
        assert bad.status_code == 400 and "token budget" in bad.json()["detail"]


def test_a_data_file_can_back_a_claim():
    """Found live: a model's config.json backs its size and context, yet shares no prose with the claim."""
    cfg = ('{"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560, "max_position_embeddings": 40960, '
           '"model_type": "qwen3"}')
    url = "https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json"
    assert document.about(cfg, url, "Qwen3-4B has 4 billion parameters and a 32K context window.")
    assert not document.about("Recipes for summer salads with tomatoes and basil. " * 5, "https://food.example/salads",
                              "Qwen3-4B runs at 6 tokens per second on a Raspberry Pi 5.")
    assert "Gaps, thin parts" in loop.REVIEW_QUESTIONS["document"]
    got = document.normalize(document.document_template(task="A brief on the EU AI Act for a studio"))
    assert got["lane"]["attempt_tokens"] == 15_000_000 and got["budget"]["tokens"] == 100_000_000


def test_caps_are_entered_in_tokens():
    assert om.parse_tokens("80M") == 80_000_000 and om.parse_tokens("500k") == 500_000
    assert om.parse_tokens("1,5m") == 1_500_000 and om.parse_tokens("2_000_000") == 2_000_000
    assert om.parse_tokens(" 3 B ") == 3_000_000_000 and om.parse_tokens("$5") == "$5"
    assert om.tokens_text(24_167_197) == "24.2M" and om.tokens_text(500_000) == "500k" and om.tokens_text(900) == "900"
    # A dollar figure is not a limit: it is dropped, and the defaults stand.
    got = om.budget_of({"usd": 5, "hours": "2"}, outcome.BUDGET)
    assert got == {"tokens": outcome.BUDGET["tokens"], "hours": 2.0, "max_experiments": outcome.BUDGET["max_experiments"]}
    with pytest.raises(om.ObjectiveError, match="token budget"):
        om.budget_of({"tokens": "50"}, outcome.BUDGET)              # below the least a run can do anything with
    with pytest.raises(om.ObjectiveError, match="most tokens one attempt"):
        om.attempt_tokens({"attempt_tokens": "ten"}, outcome.ATTEMPT_TOKENS)
    for norm, tpl in ((outcome.normalize, outcome.outcome_template(task="Find the cheapest copy of The Book")),
                      (rules.normalize, rules.triage_template(set_id="mt-0123456789ab", agent="a"))):
        assert "usd" not in norm(tpl)["budget"] and norm(tpl)["budget"]["tokens"] > 0


def test_the_budget_is_spent_in_tokens_and_old_runs_keep_dollars():
    run_ = {"objective": {"budget": {"tokens": 10_000_000}}, "spent_tokens": 9_999_999, "spent_usd": 99.0}
    assert om.budget_spent(run_) == "" and om.tokens_left(run_) == 1
    run_["spent_tokens"] = 10_000_000
    assert om.budget_spent(run_) == "the token budget is used (10M of 10M)" and om.tokens_left(run_) == 0
    old = {"objective": {"budget": {"usd": 5.0}}, "spent_tokens": 10**9, "spent_usd": 5.0}
    assert om.budget_spent(old).startswith("the budget is spent ($5.00") and om.tokens_left(old) is None
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest copy of The Book"))
    old_attempt = {"objective": {**obj, "budget": {"usd": 5.0, "hours": 4, "max_experiments": 6}}, "spent_usd": 0.0}
    assert outcome.attempt_cap(old_attempt, {})["token_cap"] == outcome.ATTEMPT_TOKENS
    tight = {"objective": obj, "spent_tokens": obj["budget"]["tokens"] - 10}
    assert outcome.attempt_cap(tight, {})["token_cap"] == outcome.ATTEMPT_TOKENS_FLOOR


def test_the_tool_offers_documents_and_tokens():
    from captain_claw.tools.captain_swarm import CaptainSwarmTool
    assert "document" in CaptainSwarmTool.parameters["properties"]["template"]["enum"]
    assert "budget_usd" not in CaptainSwarmTool.description and "budget_tokens" in CaptainSwarmTool.description
