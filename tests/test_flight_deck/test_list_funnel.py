"""The list funnel: a Bat engine that builds a list with code and short model calls.

Pinned here:

* code finds emails (on a site), phones, dates, size bands, states and links in
  text, reads tables (CSV, TSV, XLSX, JSON lines), documents and links files, and
  checks a value against the text it came from;
* the recipe takes the list's file, columns, groups, amounts and allowed values from
  the run's checks, and only what's valid from the model's draft;
* an offline list (a file the owner gave) runs with no web at all: filtered, each row
  judged in batches, every kept value checked against its row, the rest dropped
  with why or kept as maybe;
* a web list searches for its group, reads a size band from a search snippet before
  any page, reads the item's own pages, and keeps a row only when its email is on
  the page it cites and its date is in the window; searches stay within the budget;
* a stopped step goes on from where it was; the merge writes the list and why each
  dropped item went;
* Bat hands funnel steps to the funnel (not to a worker, not to the course check)
  and plans a run the funnel can't build as any Bat run.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, funnel
from captain_claw.flight_deck import funnel_extract as fx
from captain_claw.flight_deck.bat_store import BatStore


def _months_back(n: int) -> str:
    t = date.today()
    y, m = t.year, t.month - n
    while m <= 0:
        y, m = y - 1, m + 12
    return f"{y}-{m:02d}"


# ── code-only facts ────────────────────────────────────────────────────

def test_facts_code_finds():
    t = ("Write to Hello@Acme-Consulting.com or jane@acme-consulting.com (not logo@2x.png, not x@other.example). "
         "Office: 12 Main St, Austin, TX 78701. Call +1 (512) 555-0134. Posted March 3, 2026; launched 2025-11. "
         "Company size 11-50 employees. https://www.acme-consulting.com/about")
    assert fx.emails(t, "acme-consulting.com") == ["hello@acme-consulting.com", "jane@acme-consulting.com"]
    assert "x@other.example" in fx.emails(t)
    assert fx.email_kind("hello@a.example") == "shared" and fx.email_kind("jane@a.example") == "personal"
    assert fx.phones(t) == ["+1 (512) 555-0134"]
    assert [d for d, _ in fx.dates(t)] == ["2026-03-03", "2025-11"]
    assert fx.size_band(t) == (11, 50) and fx.size_band("10,001+ employees") == (10001, 10**9)
    assert fx.us_state(t) == "TX" and fx.us_state("based in New York") == "NY"
    assert fx.links(t) == ["https://www.acme-consulting.com/about"]
    assert fx.domain_of("https://www.Acme.com/x") == "acme.com"
    assert fx.in_text("JANE@acme-consulting.com", t) and not fx.in_text("Dallas", t)
    assert fx.extract("size_band", t) == ["11-50"]
    with pytest.raises(ValueError):
        fx.extract("nope", t)


def test_reading_what_a_list_comes_from(tmp_path):
    (tmp_path / "a.csv").write_text("Name,Email\nAna,ana@a.example\n")
    (tmp_path / "a.tsv").write_text("Name\tEmail\nBo\tbo@b.example\n")
    (tmp_path / "a.jsonl").write_text('{"Name": "Cy", "n": 3.0}\nnot json\n')
    import openpyxl
    wb = openpyxl.Workbook()
    wb.active.append(["Name", "Year"])
    wb.active.append(["Di", 2025])
    wb.save(tmp_path / "a.xlsx")
    assert fx.read_table(tmp_path / "a.csv") == (["Name", "Email"], [{"Name": "Ana", "Email": "ana@a.example"}])
    assert fx.read_table(tmp_path / "a.tsv")[1] == [{"Name": "Bo", "Email": "bo@b.example"}]
    assert fx.read_table(tmp_path / "a.jsonl")[1] == [{"Name": "Cy", "n": "3"}]
    assert fx.read_table(tmp_path / "a.xlsx") == (["Name", "Year"], [{"Name": "Di", "Year": "2025"}])
    (tmp_path / "doc.md").write_text("# Vendors\n\nAcme sells bolts.")
    assert "Acme sells bolts" in fx.read_document(tmp_path / "doc.md")
    assert fx.read_links("1. [A list](https://a.example/list): firms\nsee https://b.example/x") == [
        {"title": "A list", "url": "https://a.example/list", "note": "firms"},
        {"title": "", "url": "https://b.example/x", "note": ""}]
    parts = fx.chunks("para\n\n" * 3000, 6000)
    assert len(parts) > 1 and all(len(p) <= 6000 for p in parts)
    results = fx.search_results("[QUERY: q]\n\n1. Alpha | LinkedIn\n   URL: https://linkedin.com/company/alpha\n"
                                "   Snippet: Alpha | 11-50 employees\n\n2. Beta\n   URL: https://beta.example\n")
    assert [r["url"] for r in results] == ["https://linkedin.com/company/alpha", "https://beta.example"]
    assert results[0]["snippet"] == "Alpha | 11-50 employees"
    assert fx.parse_json('<think>x</think>```json\n{"a": [1]}\n```') == {"a": [1]}


# ── the recipe ─────────────────────────────────────────────────────────

CONTRACT = [{"id": "fm1", "text": "firms.csv: Firm, Website, Segment; 2 per segment", "check": {
    "type": "table", "path": "firms.csv", "columns": ["Firm", "Website", "Segment", "Email", "EmailSourceURL",
                                                       "SignalDate"],
    "min_rows": 4, "min_per": {"Segment": 2}, "allowed": {"Segment": ["x", "y"]}, "unique": ["Website"]}}]


def test_the_recipe_takes_the_checks_over_the_draft():
    base = funnel.from_contract(CONTRACT)
    assert base["output"] == "firms.csv" and base["groups"] == {"x": 2, "y": 2} and base["target"] == 4
    r = funnel.normalize_recipe({
        "item": "firm", "columns": [{"name": "Other", "what": "?"}], "output": "elsewhere.csv",
        "groups": {"z": 9}, "sources": [
            {"kind": "search", "queries": ["x firms", ""], "group": "x", "country": "usa"},
            {"kind": "search", "queries": []}, {"kind": "file", "path": "../etc/passwd"},
            {"kind": "page", "url": "ftp://no"}, {"kind": "links", "path": "extra/links.md", "as": "pages"}],
        "filters": [{"column": "size_band", "op": "size_max", "value": 50}, {"column": "x", "op": "bad"}],
        "gather": [{"kind": "fetch", "url": "{Website}", "paths": ["", "contact"]}],
        "extract": ["emails", "telepathy"],
        "checks": [{"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"},
                   {"column": "Nope", "op": "required"}, {"column": "Segment", "op": "in", "value": ["q"]}],
    }, base)
    assert r["output"] == "firms.csv" and r["columns"][0] == "Firm" and "Other" not in r["columns"]
    assert r["groups"] == {"x": 2, "y": 2} and r["key"] == "Website" and r["site_column"] == "Website"
    assert [s["kind"] for s in r["sources"]] == ["search", "links"]
    assert r["sources"][0]["queries"] == ["x firms"] and r["sources"][0]["country"] == "US"
    assert funnel._country("Croatia") == "" and funnel._country("hr") == "HR"
    assert r["filters"] == [{"column": "size_band", "op": "size_max", "value": 50}]
    assert r["extract"] == ["emails"] and r["decide_batch"] == 1
    assert {"column": "Segment", "op": "in", "value": ["x", "y"]} in r["checks"]  # the checks' values, not "q"
    assert {"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"} in r["checks"]
    assert {"column": "Website", "op": "required"} in r["checks"]
    assert funnel.render("\"{firm}\" site:linkedin.com/company", {"Firm": "Alpha"}) == '"Alpha" site:linkedin.com/company'
    assert funnel.render("{domain} news", {"Website": "https://www.alpha.example/x"}, "Website") == "alpha.example news"
    assert funnel.render("{Email}", {"Firm": "Alpha"}) is None
    assert "Each row is one firm" in funnel.conventions(r) and "appears in the page" in funnel.conventions(r)


# ── a deck to run in ───────────────────────────────────────────────────

class _Reply:
    def __init__(self, content):
        self.content = content


@pytest.fixture
async def deck(tmp_path, monkeypatch):
    """A Bat store, a run folder, a model that answers what the test says, and no
    real search or fetch."""
    store = BatStore(tmp_path / "bat.db")
    await store.init()
    monkeypatch.setattr(bat_loop, "_STORE", store)
    from captain_claw.flight_deck import auth, basna_routes, vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    state: dict = {"recipe": {}, "extract": lambda user: {"items": []}, "decide": lambda items: [],
                   "queries": [], "asked": [], "tools": None}

    class Prov:
        async def complete(self, messages, **_kw):
            system, user = messages[0].content, messages[-1].content
            if system.startswith(funnel.SETUP_SYSTEM[:40]):
                state["asked"].append("setup")
                return _Reply(json.dumps(state["recipe"]))
            if system.startswith("Pull the items out"):
                state["asked"].append("extract")
                return _Reply(json.dumps(state["extract"](user)))
            if system.startswith(funnel.DECIDE_SYSTEM[:40]):
                items = [(int(i), json.loads(f)) for i, f in re.findall(r"## id (\d+)\nfields: (\{.*?\})\n", user)]
                state["asked"].append(f"decide:{len(items)}")
                state.setdefault("decide_prompts", []).append(user)
                return _Reply(json.dumps(state["decide"](items)))
            state["asked"].append("queries")
            return _Reply(json.dumps({"queries": state["queries"]}))

    class Db:
        async def get_user_by_id(self, _id):
            return {"id": "u1", "role": "user"}

    async def tiers(_db, _owner):
        return {"reason": {"provider": "x", "model": "big"}, "fast": {"provider": "x", "model": "small"}}, []

    monkeypatch.setattr(auth, "get_db", lambda: Db())
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(basna_routes, "_effective_key", lambda _p, k, _b=None: k)
    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (Prov(), 1024))

    def no_tools(run, step):
        if state["tools"] is None:
            raise AssertionError("an offline list must not start a search and fetch agent")
        return state["tools"]

    monkeypatch.setattr(funnel, "make_tools", no_tools)
    state["store"] = store
    yield state
    await store.close()


async def _run(store, contract, cfg=None, rid="bat_funnel0001"):
    await store.create_run(run_id=rid, owner_id="u1", title="t", task="Build the list.", status="planning",
                           config={"engine": "funnel", "contract": contract, **(cfg or {})})
    run = await store.get_run(rid)
    folder = bat_routes._run_folder(run)
    (folder / "extra").mkdir(parents=True, exist_ok=True)
    return run, folder


async def _go(store, run, steps):
    """Run the planned steps in order, as Bat's loop would, with the fresh run."""
    out = []
    for s in steps:
        fresh = await store.get_run(run["id"])
        res = await bat_routes._bat_attempt(fresh, s)
        out.append(res)
    return out


# ── an offline list ────────────────────────────────────────────────────

PEOPLE = [{"id": "fm1", "text": "people.csv: Name, Email, Team; at least 3; team a or b; no students", "check": {
    "type": "table", "path": "people.csv", "columns": ["Name", "Email", "Team"], "min_rows": 3,
    "unique": ["Email"], "allowed": {"Team": ["a", "b"]}, "exclude_pattern": "student"}}]


async def test_an_offline_list_needs_no_web(deck):
    run, folder = await _run(deck["store"], PEOPLE)
    (folder / "extra" / "contacts.csv").write_text(
        "Name,Email,Team,Note\nAna,ana@acme.example,a,lead\nBo,bo@beta.example,b,\n"
        "Cy,cy@acme.example,a,student\nAna again,ANA@acme.example,a,dup\nDi,di@delta.example,b,\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}],
                      "decide": "Keep people with a work email.",
                      "checks": [{"column": "Email", "op": "in_evidence"}]}

    def decide(items):
        out = []
        for i, f in items:
            row = {"Name": f["Name"], "Email": f["Email"], "Team": f["Team"]}
            if f["Name"] == "Bo":
                out.append({"id": i, "verdict": "keep", "row": {**row, "Email": "bo@wrong.example"}})
            elif f["Name"] == "Di":
                out.append({"id": i, "verdict": "maybe", "reason": "no team page"})
            else:
                out.append({"id": i, "verdict": "keep", "row": row, "reason": "work email"})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    assert [s["title"] for s in steps] == ["Go through `extra/contacts.csv` for persons and check each",
                                           "Put the list together in people.csv"]
    assert (folder / funnel.RECIPE).is_file() and "Each row is one person" in (folder / "shared" / "conventions.md").read_text()
    work, merged = await _go(deck["store"], run, steps)
    assert deck["asked"] == ["setup", "decide:3"]  # one call for the three rows left after code's filters
    assert work["ok"] and work["output"].startswith("1 kept of 3. Short: its sources ran out.")
    assert "Looked at 4 items: 1 kept, 1 maybe" in work["output"]
    assert (folder / "people.csv").read_text().splitlines() == ["Name,Email,Team", "Ana,ana@acme.example,a"]
    dropped = (folder / "exclusions.csv").read_text()
    assert "matches what's excluded" in dropped and "bo@wrong.example" in dropped and "isn't in what was read" in dropped
    assert "di@delta.example" in (folder / funnel.MAYBE).read_text()
    statuses = [r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)]
    assert sorted(statuses) == ["dropped", "dropped", "kept", "maybe"]  # the duplicate was never looked at twice
    note = (folder / "progress" / "step-1.md").read_text()
    assert "Kept so far: 1 of 3" in note and "1 kept, 1 maybe, 2 dropped" in note
    assert merged["output"].startswith("`people.csv`: 1 rows of 3.")
    # run again (a restart, a Continue): nothing is looked at twice
    again = await bat_routes._bat_attempt(await deck["store"].get_run(run["id"]), steps[0])
    assert "Looked at 0 items" in again["output"] and deck["asked"].count("decide:3") == 1
    assert len(funnel._read_csv(folder / funnel.ROWS)) == 1


async def test_a_document_is_read_for_its_items(deck):
    contract = [{"id": "fm1", "text": "vendors.csv with Vendor, Amount", "check": {
        "type": "table", "path": "vendors.csv", "columns": ["Vendor", "Amount"]}}]
    run, folder = await _run(deck["store"], contract, rid="bat_funnel0002")
    (folder / "extra" / "invoices.md").write_text("Invoice 1: Acme Bolts, total 1,200 EUR.\nInvoice 2: Beta Nuts, 80 EUR.")
    deck["recipe"] = {"item": "vendor", "sources": [{"kind": "document", "path": "extra/invoices.md"}],
                      "checks": [{"column": "Amount", "op": "in_evidence"}]}
    deck["extract"] = lambda user: {"items": [{"name": "Acme Bolts", "fields": {"Vendor": "Acme Bolts", "Amount": "1,200 EUR"}},
                                              {"name": "Beta Nuts", "fields": {"Vendor": "Beta Nuts", "Amount": "80 EUR"}}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep",
                                     "row": {"Vendor": f["Vendor"], "Amount": "999 EUR" if "Beta" in f["Vendor"] else f["Amount"]}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert (folder / "vendors.csv").read_text().splitlines() == ["Vendor,Amount", 'Acme Bolts,"1,200 EUR"']
    assert deck["asked"] == ["setup", "extract", "decide:2"]  # no web, one call to read it, one to judge
    why = {r["key"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert "999 EUR" in why["beta nuts"]  # not in the invoice


# ── a web list ─────────────────────────────────────────────────────────

class _Tools:
    def __init__(self, pages: dict[str, str], results: dict[str, str]):
        self.pages, self.results, self.calls, self.closed = pages, results, [], False

    async def search(self, query, count, country=""):
        self.calls.append(("search", query))
        return True, self.results.get(query, "[RESULTS: 0]\n\nNo results found.")

    async def fetch(self, url):
        self.calls.append(("fetch", url))
        return (True, self.pages[url]) if url in self.pages else (False, "404")

    async def close(self):
        self.closed = True


def _result(title, url, snippet):
    return f"[QUERY: q]\n\n1. {title}\n   URL: {url}\n   Snippet: {snippet}\n"


WEB_RECIPE = {
    "item": "firm", "site_column": "Website", "key": "Website",
    "sources": [{"kind": "search", "queries": ["x firms list"], "group": "x", "country": "US", "count": 20,
                 "about": "small x firms"}],
    "filters": [{"column": "size_band", "op": "size_max", "value": 50}],
    "gather": [{"kind": "search", "query": "\"{Firm}\" site:linkedin.com/company", "count": 5},
               {"kind": "fetch", "url": "{Website}", "paths": ["", "contact"]}],
    "extract": ["emails", "dates"], "decide": "Small x firms with a dated signal and an email on their site.",
    "checks": [{"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"},
               {"column": "SignalDate", "op": "within_months", "value": 24}],
}


def _web_deck(deck, recent):
    deck["recipe"] = WEB_RECIPE
    deck["extract"] = lambda user: {"items": [
        {"name": n, "url": f"https://{n.lower()}.example", "fields": {"Firm": n, "Website": f"https://{n.lower()}.example"}}
        for n in ("Alpha", "Beta", "Gamma")] + [{"name": "A directory", "url": "https://clutch.co/x"}]}
    deck["tools"] = _Tools(
        pages={"https://alpha.example/": f"Alpha. News {recent}-10: we use AI for reports.",
               "https://alpha.example/contact": "Write to hello@alpha.example",
               "https://gamma.example/": "Gamma consulting.", "https://gamma.example/contact": "gamma@gamma.example"},
        results={"x firms list": "list results",
                 '"Alpha" site:linkedin.com/company': _result("Alpha", "https://linkedin.com/company/alpha", "11-50 employees"),
                 '"Beta" site:linkedin.com/company': _result("Beta", "https://linkedin.com/company/beta", "51-200 employees"),
                 '"Gamma" site:linkedin.com/company': _result("Gamma", "https://linkedin.com/company/gamma", "2-10 employees")})

    def decide(items):
        out = []
        for i, f in items:
            if f["Firm"] == "Alpha":
                row = {"Firm": "Alpha", "Website": "https://alpha.example", "Segment": "x", "Email": "hello@alpha.example",
                       "EmailSourceURL": "https://alpha.example/contact", "SignalDate": recent}
            else:  # an email that isn't on the page it cites
                row = {"Firm": "Gamma", "Website": "https://gamma.example", "Segment": "x", "Email": "boss@gamma.example",
                       "EmailSourceURL": "https://gamma.example/contact", "SignalDate": recent}
            out.append({"id": i, "verdict": "keep", "row": row})
        return out

    deck["decide"] = decide


async def test_a_web_list_checks_size_first_and_proves_its_sources(deck):
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_funnel0003")
    recent = _months_back(3)
    _web_deck(deck, recent)
    steps = await funnel.plan(run)
    assert [s["title"] for s in steps] == ["Find and check 2 x", "Put the list together in firms.csv"]
    work, merged = await _go(deck["store"], run, steps)
    tools = deck["tools"]
    fetched = [u for k, u in tools.calls if k == "fetch"]
    assert not any("beta" in u for u in fetched)  # 51-200 on LinkedIn: no page of it read
    assert not any("clutch" in u for k, u in tools.calls)  # a directory isn't an item
    assert tools.closed
    assert deck["asked"].count("decide:1") == 2 and "queries" in deck["asked"]  # still short: asked for more
    rows = funnel._read_csv(folder / "firms.csv")
    assert [r["Firm"] for r in rows] == ["Alpha"] and rows[0]["Email"] == "hello@alpha.example"
    why = {r["key"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS) if r["status"] == "dropped"}
    assert why["beta.example"].startswith("51-200 people")
    assert "boss@gamma.example" in why["gamma.example"] and "isn't in https://gamma.example/contact" in why["gamma.example"]
    assert work["output"].startswith("x: 1 of 2 kept. Short: its sources ran out.")
    queries = json.loads((folder / funnel.QUERIES).read_text())
    assert queries["searches"] == 4 and queries["used"] == ["x firms list"]
    events = [e for e in await deck["store"].list_events(run["id"]) if e["stage"] == "action"]
    kinds = {e["message"] for e in events}
    assert {"web_search", "web_fetch", "decide"} <= kinds
    assert merged["output"].startswith("`firms.csv`: 1 rows of 4. x 1 of 2; y 0 of 2.")


async def test_a_date_outside_its_window_is_dropped(deck):
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_funnel0004")
    _web_deck(deck, _months_back(30))
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    why = {r["key"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert "outside the last 24 months" in why["alpha.example"]


async def test_searches_stay_within_the_budget(deck):
    run, folder = await _run(deck["store"], CONTRACT, cfg={"funnel_search_budget": 2}, rid="bat_funnel0005")
    _web_deck(deck, _months_back(3))
    steps = await funnel.plan(run)
    work, = await _go(deck["store"], run, steps[:1])
    assert sum(1 for k, _ in deck["tools"].calls if k == "search") == 2
    assert json.loads((folder / funnel.QUERIES).read_text())["searches"] == 2
    assert "Searches used in the run: 2 of 2" in work["output"]


# ── Bat's side ─────────────────────────────────────────────────────────

async def test_bat_hands_funnel_steps_to_the_funnel(deck, monkeypatch):
    run = {"id": "bat_x", "owner_id": "u1", "config": {"engine": "funnel", "funnel_steps": {"step-1": {"kind": "work"}}}}
    called = []

    async def attempt(r, s):
        called.append(s["step_key"])
        return {"ok": True, "output": "done", "usd": 0.0, "tokens": 0}

    monkeypatch.setattr(funnel, "attempt", attempt)
    assert (await bat_routes._bat_attempt(run, {"step_key": "step-1"}))["output"] == "done"
    assert called == ["step-1"]
    assert await bat_routes._course_check(run, {"step_key": "step-1"}, 0.0, False) is None
    # a fix round's step is Bat's own; so is any step of a run that isn't the funnel's
    assert funnel.step_kind(run, {"step_key": "fix-1-1"}) is None
    assert funnel.step_kind({"config": {"funnel_steps": {"step-1": {"kind": "work"}}}}, {"step_key": "step-1"}) is None


async def test_a_run_the_funnel_cant_build_is_planned_as_any_bat_run(deck, monkeypatch):
    run, folder = await _run(deck["store"], [{"id": "a", "text": "a clear report", "check": {"type": "judge"}}],
                             rid="bat_funnel0006")
    deck["recipe"] = {"item": "x"}  # no source, no columns
    assert await funnel.plan(run) is None
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any("The list funnel can't run this" in n and "runs as a Bat run instead" in n for n in notes)

    async def nothing(*_a, **_k):
        return None

    for name in ("_ensure_contract", "_shape_contract", "_one_at_a_time", "_models_note"):
        monkeypatch.setattr(bat_routes, name, nothing)

    async def plain(_run):
        return [{"step_key": "step-1", "title": "plain", "seq": 0}]

    monkeypatch.setattr(bat_routes, "_plan_steps", plain)
    assert (await bat_routes._bat_planner(run))[0]["title"] == "plain"

    async def funnel_plan(_run):
        return [{"step_key": "step-1", "title": "funnel", "seq": 0}]

    monkeypatch.setattr(funnel, "plan", funnel_plan)
    assert (await bat_routes._bat_planner(run))[0]["title"] == "funnel"
