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
                   "queries": [], "asked": [], "tools": None, "owner": lambda user: {}}

    class Prov:
        async def complete(self, messages, **_kw):
            system, user = messages[0].content, messages[-1].content
            if system.startswith(funnel.SETUP_SYSTEM[:40]):
                state["asked"].append("setup")
                state.setdefault("setup_prompts", []).append(user)
                r = state["recipe"]
                return _Reply(json.dumps(r() if callable(r) else r))
            if system.startswith("Pull the items out"):
                state["asked"].append("extract")
                return _Reply(json.dumps(state["extract"](user)))
            if system.startswith(funnel.DECIDE_SYSTEM[:40]):
                items = [(int(i), json.loads(f)) for i, f in re.findall(r"## id (\d+)\nfields: (\{.*?\})\n", user)]
                state["asked"].append(f"decide:{len(items)}")
                state.setdefault("decide_prompts", []).append(user)
                return _Reply(json.dumps(state["decide"](items)))
            if system.startswith("An item nearly fits a list"):
                state["asked"].append("look")
                state.setdefault("look_prompts", []).append(user)
                return _Reply(json.dumps(state.get("look", lambda u: {"do": "stop"})(user)))
            if system.startswith("A table is read as items for a list"):
                state["asked"].append("columns")
                return _Reply(json.dumps({"columns": state.get("columns", lambda u: {})(user)}))
            if system.startswith("A list sorts its items into groups"):
                state["asked"].append("group")
                state.setdefault("group_prompts", []).append(user)
                return _Reply(json.dumps({"group": state.get("group", lambda u: "none")(user)}))
            if system.startswith("Before a list is built"):
                state["asked"].append("brief")
                state.setdefault("brief_prompts", []).append(user)
                return _Reply(state.get("brief", ""))
            if system.startswith("Fill the owner's columns"):
                state["asked"].append("owner")
                return _Reply(json.dumps(state["owner"](user)))
            state["asked"].append("queries")
            state.setdefault("query_prompts", []).append(user)
            q = state["queries"]
            return _Reply(json.dumps({"queries": q() if callable(q) else q}))

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
    monkeypatch.setattr(funnel, "MODEL_WAITS", (0.0, 0.0))  # a failed call is tried again at once here
    from captain_claw.flight_deck import jev_routes

    async def no_jev(*_a, **_k):  # no Jev here unless a test connects one
        raise jev_routes.JevNotConnected()

    monkeypatch.setattr(jev_routes, "ask_for_owner", no_jev)
    state["store"] = store
    yield state
    await store.close()


async def _run(store, contract, cfg=None, rid="bat_funnel0001"):
    await store.create_run(run_id=rid, owner_id="u1", title="t", task="Build the list.", status="planning",
                           config={"engine": "funnel", "contract": contract, "funnel_look": "off", **(cfg or {})})
    run = await store.get_run(rid)
    folder = bat_routes._run_folder(run)
    (folder / "extra").mkdir(parents=True, exist_ok=True)
    return run, folder


async def _go(store, run, steps):
    """Run the planned funnel steps in order, as Bat's loop would, with the fresh run.
    The finishing step (an agent's) is left out: its tests run it on their own."""
    out = []
    for s in steps:
        fresh = await store.get_run(run["id"])
        if (funnel.step_kind(fresh, s) or {}).get("kind") == "finish":
            continue
        out.append(await bat_routes._bat_attempt(fresh, s))
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
                                           "Finish what's still short from the nearly-there items",
                                           "Put the list together in people.csv"]
    assert (folder / funnel.RECIPE).is_file() and "Each row is one person" in (folder / "shared" / "conventions.md").read_text()
    work, merged = await _go(deck["store"], run, steps)
    assert deck["asked"] == ["setup", "decide:3"]  # one call for the three rows left after code's filters
    assert work["ok"] and work["output"].startswith("1 kept of 3. Short: its sources ran out.")
    assert "Looked at 4 items: 1 kept, 2 maybe" in work["output"]
    assert (folder / "people.csv").read_text().splitlines() == ["Name,Email,Team", "Ana,ana@acme.example,a"]
    dropped = (folder / "exclusions.md").read_text()
    assert "matches what's excluded" in dropped
    maybe = (folder / funnel.MAYBE).read_text()  # an email it got wrong is left out, never on the list: Bo came close
    assert "di@delta.example" in maybe and "couldn't confirm Email 'bo@wrong.example'" in maybe
    assert "bo@wrong.example" not in json.dumps([m["row"] for m in funnel._read_csv(folder / funnel.MAYBE)])
    statuses = [r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)]
    assert sorted(statuses) == ["dropped", "kept", "maybe", "maybe"]  # the duplicate was never looked at twice
    note = (folder / "progress" / "step-1.md").read_text()
    assert "Kept so far: 1 of 3" in note and "1 kept, 2 maybe, 1 dropped" in note
    assert merged["output"].startswith("`people.csv`: 1 rows of 3.") and "`exclusions.md`: 1 dropped" in merged["output"]
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
    # no web: the owner's document read once before the run, one call to read its items, one to judge
    assert deck["asked"] == ["brief", "setup", "extract", "decide:2"]
    assert funnel.brief_text(folder) == ""  # a brief of the list's own source is not kept beside every item
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
               "https://gamma.example/": "Gamma consulting.", "https://gamma.example/contact": "Write to us with the form."},
        results={"x firms list": "list results",
                 '"Alpha" site:linkedin.com/company': _result("Alpha", "https://linkedin.com/company/alpha", "11-50 employees"),
                 '"Beta" site:linkedin.com/company': _result("Beta", "https://linkedin.com/company/beta", "51-200 employees"),
                 '"Gamma" site:linkedin.com/company': _result("Gamma", "https://linkedin.com/company/gamma", "2-10 employees")})

    def decide(items):
        out = []
        for i, f in items:
            if f.get("Firm") not in ("Alpha", "Gamma"):  # a listing with no site of its own
                out.append({"id": i, "verdict": "drop", "reason": "a directory, not a firm"})
                continue
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
    assert [s["title"] for s in steps] == ["Find and check 2 x", "Finish what's still short from the nearly-there items",
                                           "Put the list together in firms.csv"]
    work, merged = await _go(deck["store"], run, steps)
    tools = deck["tools"]
    fetched = [u for k, u in tools.calls if k == "fetch"]
    assert not any("beta" in u for u in fetched)  # 51-200 on LinkedIn: no page of it read
    assert not any("clutch" in u for k, u in tools.calls)  # a directory isn't an item
    assert tools.closed
    assert deck["asked"].count("decide:1") == 3 and "queries" in deck["asked"]  # still short: asked for more
    rows = funnel._read_csv(folder / "firms.csv")
    assert [r["Firm"] for r in rows] == ["Alpha"] and rows[0]["Email"] == "hello@alpha.example"
    why = {r["key"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert why["beta.example"][0] == "dropped" and why["beta.example"][1].startswith("51-200 people")
    # an email that isn't on the page it cites is left out, never on the list: Gamma came close
    assert why["gamma.example"] == ("maybe", "couldn't confirm Email 'boss@gamma.example' in what was read")
    assert work["output"].startswith("x: 1 of 2 kept. Short: its sources ran out.")
    queries = json.loads((folder / funnel.QUERIES).read_text())
    assert queries["searches"] == 5 and queries["used"] == ["x firms list"]  # 1 list, 3 size bands, 1 site lookup
    events = [e for e in await deck["store"].list_events(run["id"]) if e["stage"] == "action"]
    kinds = {e["message"] for e in events}
    assert {"web_search", "web_fetch", "decide"} <= kinds
    assert merged["output"].startswith("`firms.csv`: 1 rows of 4. x 1 of 2; y 0 of 2.")
    assert (folder / "summary.md").is_file() and json.loads((folder / funnel.COUNTS).read_text())["groups"]["x"]["read"] == 1


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


# ── phase 2: finishing what's short ────────────────────────────────────

def _finish_step(steps):
    return next(s for s in steps if s["title"].startswith("Finish"))


async def test_the_finishing_step_works_only_on_the_nearly_there(deck):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_funnel0010")
    (folder / "extra" / "contacts.csv").write_text(
        "Name,Email,Team\nAna,ana@acme.example,a\nDi,di@delta.example,b\nEd,ed@echo.example,b\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}]}
    deck["decide"] = lambda items: [
        {"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Email": f["Email"], "Team": f["Team"]}}
        if f["Name"] == "Ana" else {"id": i, "verdict": "maybe", "reason": f"no page for {f['Name']}",
                                    "row": {"Team": f["Team"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fresh = await deck["store"].get_run(run["id"])
    prep = funnel.prepare_finish(fresh, _finish_step(steps))
    work = prep["work"]
    assert "Still short: the list 2 more." in work and "2 items came close" in work
    assert "- Di: no page for Di" in work and "don't look for new ones" in work  # a list with no groups
    assert [m["name"] for m in funnel._read_csv(folder / funnel.FINISH_ITEMS)] == ["Di", "Ed"]
    assert (folder / funnel.FINISH).read_text().strip() == "Name,Email,Team"  # the shape its rows take
    # nothing for it: the run says so, or nothing is short, or nothing came close
    off = {**fresh, "config": {**fresh["config"], "funnel_leftovers": "none"}}
    assert "leave a short group as it is" in funnel.prepare_finish(off, _finish_step(steps))["result"]["output"]
    (folder / funnel.MAYBE).unlink()
    assert "Nothing nearly there to finish" in funnel.prepare_finish(fresh, _finish_step(steps))["result"]["output"]
    for _ in range(2):
        funnel._append_csv(folder / funnel.ROWS, ["Name", "Email", "Team", "_group"],
                           {"Name": "N", "Email": f"n{_}@x.example", "Team": "a"})
    assert funnel.prepare_finish(fresh, _finish_step(steps))["result"]["output"] == "Nothing to finish: no group is short."
    # a run whose settings say none plans no finishing step
    run2, _ = await _run(deck["store"], PEOPLE, cfg={"funnel_leftovers": "none"}, rid="bat_other0011")
    (bat_routes._run_folder(run2) / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@acme.example,a\n")
    assert not any(s["title"].startswith("Finish") for s in await funnel.plan(run2))


async def test_bat_runs_the_finishing_step_as_a_small_worker(deck, monkeypatch):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_funnel0012")
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nDi,di@delta.example,b\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "reason": "no page"} for i, _ in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    seen = {}

    class SpawnedError(Exception):
        pass

    async def spawn(stub, user, **kw):
        seen.update(kw)
        raise SpawnedError()

    monkeypatch.setattr(bat_routes, "_spawn_bat_worker", spawn)
    with pytest.raises(SpawnedError):
        await bat_routes._bat_attempt(await deck["store"].get_run(run["id"]), _finish_step(steps))
    assert seen["max_context"] == funnel.AGENT_CONTEXT
    # a fix round's step in a funnel run is small too; in any other run it isn't
    seen.clear()
    with pytest.raises(SpawnedError):
        await bat_routes._bat_attempt(await deck["store"].get_run(run["id"]), {"step_key": "fix-1-1", "title": "Fix", "seq": 9})
    assert seen["max_context"] == funnel.AGENT_CONTEXT
    assert funnel.agent_context({"config": {}}) == 0
    assert funnel.agent_context({"config": {"engine": "funnel", "funnel_agent_context": 32000}}) == 32000
    prompt = bat_routes._build_step_prompt(run, {"step_key": "step-2", "title": "Finish", "work": "# What to work on\nX"}, [])
    assert "# What to work on\nX" in prompt and "# Why this step exists" not in prompt


async def test_the_worker_context_is_capped_never_raised(monkeypatch):
    from captain_claw.flight_deck import server as srv
    got = []

    async def spawn(cfg, request, user):
        got.append(cfg.max_context)
        return type("R", (), {"ok": False, "slug": "s", "message": "no"})()

    monkeypatch.setattr(srv, "spawn_process", spawn)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {})
    monkeypatch.setattr(bat_routes, "_teardown", lambda slugs: None)
    tiers = {"fast": {"provider": "x", "model": "m", "input_ctx": 160000}}
    for cap in (0, 64000, 500000):
        await bat_routes._spawn_bat_worker(None, {"id": "u"}, name="n", tier="fast", tiers=tiers, env_vars=[],
                                           extra_env=[], **({"max_context": cap} if cap else {}))
    assert got == [160000, 64000, 160000]


async def test_the_merge_rechecks_the_finishing_steps_rows(deck):
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_funnel0013")
    recent = _months_back(3)
    _web_deck(deck, recent)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    deck["tools"].pages["https://delta.example/contact"] = "Reach us: team@delta.example"
    deck["tools"].pages["https://eps.example/contact"] = "No email here."
    cols = ["Firm", "Website", "Segment", "Email", "EmailSourceURL", "SignalDate"]
    for row in (
        ["Delta", "https://delta.example", "x", "team@delta.example", "https://delta.example/contact", recent],
        ["Eps", "https://eps.example", "x", "ceo@eps.example", "https://eps.example/contact", recent],
        ["Alpha", "https://alpha.example", "x", "hello@alpha.example", "https://alpha.example/contact", recent],
    ):
        funnel._append_csv(folder / funnel.FINISH, cols, dict(zip(cols, row)))
    merged = await bat_routes._bat_attempt(await deck["store"].get_run(run["id"]), steps[-1])
    assert "The finishing step added 1 (and 1 that failed a check)" in merged["output"]
    rows = funnel._read_csv(folder / "firms.csv")
    assert [r["Firm"] for r in rows] == ["Alpha", "Delta"]  # Alpha once, Delta's email on its page
    stages = {r["key"]: (r["stage"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert stages["delta.example"][0] == "agent"
    assert stages["eps.example"][0] == "agent-check" and "isn't in https://eps.example/contact" in stages["eps.example"][1]
    summary = (folder / "summary.md").read_text()
    assert "| x | 2 | 2 | - |" in summary and "| y | 0 | 2 | 2 |" in summary
    assert "The finishing step added 1 row that passed the same checks" in summary
    assert "Email appears in the page or file it came from (the one in EmailSourceURL)" in summary
    excl = (folder / "exclusions.md").read_text()
    assert "| Beta | x | read |" in excl and "| Eps | x | agent-check |" in excl
    # merged again (a retry): nothing is added twice
    again = await bat_routes._bat_attempt(await deck["store"].get_run(run["id"]), steps[-1])
    assert "finishing step added" not in again["output"]
    assert len(funnel._read_csv(folder / "firms.csv")) == 2


async def test_the_judge_hears_what_code_checked(deck):
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_funnel0014")
    _web_deck(deck, _months_back(3))
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    note = funnel.judge_note(await deck["store"].get_run(run["id"]))
    assert note.startswith("## The list funnel's own checks") and "SignalDate is a date" in note
    assert "It looked at 4 items and kept 1" in note
    assert funnel.judge_note({"config": {}}) == ""


# ── phase 4: what earlier runs checked ─────────────────────────────────

async def test_items_earlier_runs_checked_arent_looked_at_again(deck):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_funnel0020")
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@acme.example,a\nBo,bo@beta.example,b\n")
    (folder / "extra" / funnel.LEDGER_FILE).write_text("key,name,group,outcome,reason\nANA@acme.example,Ana,,kept,on a list\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"},
                                                     {"kind": "file", "path": f"extra/{funnel.LEDGER_FILE}"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Email": f["Email"],
                                                                         "Team": f["Team"]}} for i, f in items]
    steps = await funnel.plan(run)
    assert [s["path"] for s in funnel.load_recipe(folder)["sources"]] == ["extra/contacts.csv"]  # the record isn't one
    work, = await _go(deck["store"], run, steps[:1])
    assert deck["asked"] == ["setup", "decide:1"]
    assert "Skipped 1 that earlier runs already checked." in work["output"]
    assert [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["Bo"]
    assert "Skipped 1 that earlier runs already checked." in (folder / "progress" / "step-1.md").read_text()


# ── thinking off, and a fallback for web lists ─────────────────────────

CHARTER = ("Find small US knowledge firms that are likely to want Captain, and list them in leads.csv. Four "
           "segments, 10 firms in each. A firm counts only if it is headquartered in the US; it has 2–50 employees; "
           "it published something dated in the last 24 months that shows interest in AI; its own website prints "
           "an email address.")
LEADS = [{"id": "fm1", "text": "leads.csv: 15 columns; 10 per Segment", "check": {
    "type": "table", "path": "leads.csv",
    "columns": ["Firm", "Website", "Segment", "State", "Employees", "EmployeesSourceURL", "Email", "EmailType",
                "EmailSourceURL", "SignalDate", "SignalURL", "Signal"],
    "min_rows": 40, "min_per": {"Segment": 10}, "allowed": {"Segment": ["consultancies", "research shops"]},
    "unique": ["Website"]}}]


def test_a_web_list_without_a_models_recipe():
    r = funnel.web_recipe(funnel.from_contract(LEADS), CHARTER)
    assert r["item"] == "knowledge firm" and r["site_column"] == "Website" and r["key"] == "Website"  # from its ask
    assert [s["group"] for s in r["sources"]] == ["consultancies", "research shops"]
    q = r["sources"][0]["queries"]
    assert r["sources"][0]["country"] == "US" and "consultancies directory United States" in q
    assert 'site:linkedin.com/company consultancies United States "2-50 employees"' in q
    assert r["filters"] == [{"column": "size_band", "op": "size_max", "value": 50}]
    assert r["gather"][0] == {"kind": "search", "query": '"{Firm}" site:linkedin.com/company', "count": 5}
    assert r["gather"][1]["url"] == "{Website}" and "contact" in r["gather"][1]["paths"]
    assert {"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"} in r["checks"]
    assert {"column": "SignalDate", "op": "within_months", "value": 24} in r["checks"]
    assert not any(c["column"] in ("Employees", "Signal") and c["op"] == "in_evidence" for c in r["checks"])
    assert "us_state" in r["extract"] and r["decide"].startswith("Find small US knowledge firms")
    assert funnel.web_recipe(funnel.from_contract([]), CHARTER) is None  # no columns, no list


def test_a_few_links_are_reference_not_a_list(tmp_path):
    folder = tmp_path / "run"
    (folder / "extra").mkdir(parents=True)
    links = folder / "extra" / "links-1.md"
    links.write_text("# Links\n\n1. [Web](https://captain.example)\n2. [Repo](https://code.example/captain)\n")
    base = funnel.from_contract(LEADS)
    assert funnel.default_recipe(folder, base) is None
    r = funnel.fallback_recipe(folder, base, CHARTER)
    assert {s["kind"] for s in r["sources"]} == {"search"}
    links.write_text("# Links\n\n" + "".join(f"{i}. [Firm {i}](https://firm{i}.example)\n" for i in range(1, 8)))
    assert funnel.default_recipe(folder, base)["sources"] == [{"kind": "links", "path": "extra/links-1.md",
                                                              "group": "", "as": "items"}]


@pytest.fixture
def calls(deck, monkeypatch):
    """What each model call asked for: its thinking level and its output limit."""
    from captain_claw.flight_deck import basna_routes
    from captain_claw.llm import reasoning
    seen: list[tuple[str, str | None, int]] = []
    inner = basna_routes._provider_call(None)[0]

    class Spy:
        async def complete(self, messages, **kw):
            seen.append((messages[0].content[:24], reasoning.requested_effort(), kw.get("max_tokens")))
            return await inner.complete(messages, **kw)

    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (Spy(), 32768))
    return seen


async def test_the_funnels_calls_dont_think(deck, calls):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_funnel0030")
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@a.example,a\nBo,bo@b.example,b\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "drop", "reason": "no"} for i, _ in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    setup, decide = calls
    assert setup[1] == "off" and setup[2] == funnel.OUT_TOKENS["setup"]
    assert decide[1] == "off" and decide[2] == funnel.OUT_TOKENS["decide"] + funnel.DECIDE_PER_ITEM
    # a run that turns thinking on: its judging calls think, with room for it
    run2, folder2 = await _run(deck["store"], PEOPLE, cfg={"funnel_thinking": "on"}, rid="bat_think0031")
    (folder2 / "extra" / "contacts.csv").write_text("Name,Email,Team\nCy,cy@c.example,a\n")
    calls.clear()
    await _go(deck["store"], run2, (await funnel.plan(run2))[:1])
    assert calls[0][1] == "off" and calls[1][1] is None and calls[1][2] == funnel.THINKING_TOKENS


async def test_setup_falls_back_to_web_searches(deck):
    run, folder = await _run(deck["store"], LEADS, rid="bat_funnel0032")
    run = {**run, "task": CHARTER}
    (folder / "extra" / "links-1.md").write_text("# Links\n\n1. [Web](https://captain.example)\n")
    deck["recipe"] = {"kind": "search", "query": "\"{Firm}\" site:linkedin.com/company"}  # a fragment, no sources
    steps = await funnel.plan(run)
    assert [s["title"] for s in steps][:2] == ["Find and check 10 consultancies", "Find and check 10 research shops"]
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any("set up from web searches built from its checks" in n for n in notes)
    assert deck["asked"] == ["setup", "setup"]  # asked twice, then the fallback


# ── a firm's own site, the charter's rules, and what code tidies ───────

def test_the_charters_value_lists_become_checks():
    charter = ("leads.csv has these columns.\n- Segment is exactly one of the four segment names above.\n"
               "- CaptainFeature is exactly one of: Onboarding, Memory & files, Flows. Pick the one that fits.\n"
               "- Tier is one of gold or silver\n- Nope is one of: a, b")
    got = funnel.charter_checks(charter, ["Segment", "CaptainFeature", "Tier"])
    assert got == [{"column": "CaptainFeature", "op": "in", "value": ["Onboarding", "Memory & files", "Flows"]},
                   {"column": "Tier", "op": "in", "value": ["gold", "silver"]}]


def test_code_tidies_a_models_recipe(tmp_path):
    base = funnel.from_contract(LEADS)
    copied = ["boutique consulting firm directory", "boutique consulting firm site:linkedin.com/company"]
    r = funnel.normalize_recipe({
        "item": "small US knowledge firm (boutique consultancy, research/analysis shop, marketing agency)",
        "sources": [{"kind": "document", "path": "extra/Captain.md"},
                    {"kind": "search", "group": "consultancies", "queries": copied},
                    {"kind": "search", "group": "research shops", "queries": copied}]}, base)
    assert r["item"] == "small US knowledge firm"
    charter = CHARTER + " What Captain is and does is in extra/Captain.md. CaptainFeature is exactly one of: A, B."
    t = funnel.tidy_recipe(r, charter, tmp_path)
    assert [s["kind"] for s in t["sources"]] == ["search", "search"] and t["reference"] == ["extra/Captain.md"]
    assert t["sources"][0]["queries"] == [f"{q} United States" for q in copied]  # names its group: kept, with the place
    assert t["sources"][1]["queries"][0] == "list of research shops United States"  # copied: written for it
    assert {"column": "Signal", "op": "in", "value": ["A", "B"]} not in t["checks"]
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "Captain.md").write_text("# Captain\n\nFeatures: Onboarding, Flows.")
    (tmp_path / "extra" / "links-1.md").write_text("# Links\n\n1. [Web](https://captain.example)\n")
    (tmp_path / "extra" / funnel.LEDGER_FILE).write_text("key\nx.example\n")
    ref = funnel.reference_text(tmp_path, t)
    assert ref.startswith("[extra/Captain.md] # Captain Features: Onboarding, Flows.") and "links-1" not in ref


async def test_a_firm_found_on_linkedin_gets_its_own_site_read(deck):
    contract = [{"id": "fm1", "text": "firms.csv", "check": {
        "type": "table", "path": "firms.csv", "columns": ["Firm", "Website", "Segment", "Email", "EmailSourceURL",
                                                           "CaptainFeature"],
        "min_rows": 2, "min_per": {"Segment": 1}, "allowed": {"Segment": ["x", "y"]}, "unique": ["Website"]}}]
    run, folder = await _run(deck["store"], contract, rid="bat_site00001")
    run = {**run, "task": "Find US firms. CaptainFeature is exactly one of: Onboarding, Flows. Captain is described in extra/Captain.md."}
    (folder / "extra" / "Captain.md").write_text("# Captain\n\nOnboarding teaches agents your documents.")
    deck["recipe"] = {"item": "firm", "site_column": "Website",
                      "sources": [{"kind": "document", "path": "extra/Captain.md"},
                                  {"kind": "search", "group": "x", "country": "US", "queries": ["x firms list"]}],
                      "gather": [{"kind": "fetch", "url": "{Website}", "paths": ["", "contact"]}],
                      "extract": ["emails"],
                      "checks": [{"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"}]}
    deck["extract"] = lambda user: {"items": [{"name": "Delta", "url": "https://www.linkedin.com/company/delta",
                                               "fields": {"Firm": "Delta"}},
                                              {"name": "Eps", "fields": {"Firm": "Eps"}}]}
    deck["tools"] = _Tools(
        pages={"https://delta.example/": "Delta Advisors.", "https://delta.example/contact": "hi@delta.example",
               "https://eps.example/": "Eps.", "https://eps.example/contact": "team@eps.example"},
        results={"x firms list": "list results",
                 '"Delta" firm United States': (
                     "[QUERY: q]\n\n1. Delta | LinkedIn\n   URL: https://www.linkedin.com/company/delta\n   Snippet: -\n\n"
                     "2. Unrelated\n   URL: https://other.example\n   Snippet: -\n\n"
                     "3. Delta Advisors | Home\n   URL: https://delta.example/\n   Snippet: Delta Advisors\n"),
                 '"Eps" firm United States': _result("Eps", "https://eps.example/", "Eps home")})

    def decide(items):
        out = []
        for i, f in items:
            firm = f["Firm"]
            feature = "Onboarding" if firm == "Delta" else "AI Readiness Assessment"  # Eps: not a Captain feature
            out.append({"id": i, "verdict": "keep", "row": {
                "Firm": firm, "Website": f.get("Website", ""), "Segment": "x", "Email": "hi@delta.example" if firm == "Delta"
                else "team@eps.example", "EmailSourceURL": f"{f.get('Website', '')}/contact", "CaptainFeature": feature}})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    recipe = funnel.load_recipe(folder)
    assert [s["kind"] for s in recipe["sources"]] == ["search"] and recipe["reference"] == ["extra/Captain.md"]
    assert {"column": "CaptainFeature", "op": "in", "value": ["Onboarding", "Flows"]} in recipe["checks"]
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": run["task"]}, steps[0])
    fetched = [u for k, u in deck["tools"].calls if k == "fetch"]
    assert "https://delta.example/contact" in fetched and not any("other.example" in u for u in fetched)
    rows = funnel._read_csv(folder / funnel.ROWS)
    assert [(r["Firm"], r["Website"]) for r in rows] == [("Delta", "https://delta.example")]
    why = {r["name"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert "CaptainFeature 'AI Readiness Assessment' isn't one of the allowed values" in why["Eps"]
    prompt = deck["decide_prompts"][0]
    assert "# The goal, in the owner's words" in prompt and "CaptainFeature is exactly one of: Onboarding, Flows" in prompt
    assert "# Reference\n[extra/Captain.md] # Captain Onboarding teaches agents your documents." in prompt


# ── the list's file, the place, and listings from elsewhere ────────────

def test_a_listing_from_elsewhere():
    assert fx.foreign_place("Boutique Growth | Oakville, Ontario | 12 followers") == "Ontario"
    assert fx.foreign_place("NAOS Consulting · Casablanca, Morocco") == "Casablanca"
    for us in ("Acme · Austin, TX 78701", "a firm in New Mexico", "offices in New York and London",
               "Ontario, California office", "based in the UK and United States", "a clothing store"):
        assert fx.foreign_place(us) == "", us


def test_the_lists_file_and_the_place_come_from_the_charter(tmp_path):
    r = funnel.normalize_recipe({"columns": ["Firm", "Website"], "sources": [
        {"kind": "search", "queries": ["boutique consulting firm site:linkedin.com/company", "consultancies in Texas",
                                       "US consulting firms list"]}]})
    assert r["output"] == "results.csv"
    t = funnel.tidy_recipe(r, "Find small US firms and list them in leads.csv. Also write summary.md.", tmp_path)
    assert t["output"] == "leads.csv"
    assert t["sources"][0]["queries"] == ["boutique consulting firm site:linkedin.com/company United States",
                                          "consultancies in Texas", "US consulting firms list"]
    assert t["sources"][0]["country"] == "US"
    named = {**r, "output": "firms.csv"}
    assert funnel.fix_output(named, "list them in leads.csv") is False and named["output"] == "firms.csv"


async def test_a_listing_from_elsewhere_isnt_read(deck):
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_place0001")
    recent = _months_back(3)
    _web_deck(deck, recent)
    deck["extract"] = lambda user: {"items": [
        {"name": "Alpha", "url": "https://alpha.example", "fields": {"Firm": "Alpha", "Website": "https://alpha.example"}},
        {"name": "Maple", "url": "https://maple.example", "fields": {"Firm": "Maple", "Website": "https://maple.example"}}]}
    deck["tools"].results["x firms list"] = (
        "[QUERY: q]\n\n1. Alpha | Consulting\n   URL: https://alpha.example\n   Snippet: Alpha, Austin, TX\n\n"
        "2. Maple Advisory\n   URL: https://maple.example\n   Snippet: Maple Advisory · Toronto, Ontario · 11-50 employees\n")
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fetched = [u for k, u in deck["tools"].calls if k == "fetch"]
    assert not any("maple" in u for u in fetched) and any("alpha" in u for u in fetched)
    why = {r["key"]: (r["stage"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert why["maple.example"] == ("filter", "not in the United States (its listing says Ontario)")


async def test_a_recipe_written_before_gets_the_charters_file(deck):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_named0001")
    run = {**run, "task": "Find people and list them in contacts-out.csv."}
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@a.example,a\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Email": f["Email"],
                                                                         "Team": f["Team"]}} for i, f in items]
    steps = await funnel.plan(run)
    recipe = funnel.load_recipe(folder)
    recipe["output"] = "results.csv"  # as an earlier version wrote it
    (folder / funnel.RECIPE).write_text(json.dumps(recipe))
    fresh = {**(await deck["store"].get_run(run["id"])), "task": run["task"]}
    await bat_routes._bat_attempt(fresh, steps[0])
    await bat_routes._bat_attempt(fresh, steps[-1])
    assert funnel.load_recipe(folder)["output"] == "contacts-out.csv" and (folder / "contacts-out.csv").is_file()


# ── any place, any kind of item ────────────────────────────────────────

def test_which_country_a_charter_means():
    assert fx.target_places("Find small US knowledge firms") == ["US"]
    assert fx.target_places("small Croatian law firms") == ["HR"]
    assert fx.target_places("vegan restaurants in Zagreb") == ["HR"]
    assert fx.target_places("clinics in the UK and Ireland") == ["GB", "IE"]
    assert fx.target_places("German machine makers") == ["DE"]
    assert fx.target_places("Spanish restaurants in London") == []  # a place it doesn't know: no guess
    assert fx.target_places("recipes with lentils") == []
    assert fx.foreign_place("Belgrade, Serbia", ["HR"]) == "Belgrade"
    assert fx.foreign_place("Zagreb, Croatia", ["HR"]) == ""
    assert fx.foreign_place("Toronto, Ontario", ["US", "CA"]) == ""  # either is the list's
    assert fx.place_phrase("US") == "the United States" and fx.place_phrase("HR") == "Croatia"


def test_a_list_of_anything_without_a_models_recipe():
    base = funnel.from_contract([{"id": "fm1", "text": "places.csv", "check": {
        "type": "table", "path": "places.csv", "columns": ["Name", "Website", "Address", "Cuisine", "MenuURL"],
        "min_rows": 20}}])
    r = funnel.web_recipe(base, "Find 20 vegan restaurants in Zagreb that put their menu online, and list them in places.csv.")
    assert r["item"] == "vegan restaurant" and r["output"] == "places.csv"
    q = r["sources"][0]["queries"]
    assert "list of vegan restaurants Croatia" in q and not any("linkedin" in x or "employees" in x for x in q)
    assert r["sources"][0]["country"] == ""  # the search engine takes no Croatia: the words name it
    assert r["filters"] == [] and "size_band" not in r["extract"] and "us_state" not in r["extract"]
    assert funnel.item_from("Collect at least 50 open datasets on air quality, with a link") == "open dataset"


def test_a_croatian_list_keeps_to_croatia(tmp_path):
    r = funnel.normalize_recipe({"columns": ["Name", "Website"], "sources": [
        {"kind": "search", "queries": ["law firms list", "odvjetnička društva Zagreb"]}]})
    t = funnel.tidy_recipe(r, "Find small Croatian law firms with a website.", tmp_path)
    assert t["places"] == ["HR"] and t["sources"][0]["country"] == ""
    assert t["sources"][0]["queries"] == ["law firms list Croatia", "odvjetnička društva Zagreb"]


async def test_a_listing_from_another_country_isnt_read(deck):
    contract = [{"id": "fm1", "text": "firms.csv", "check": {
        "type": "table", "path": "firms.csv", "columns": ["Name", "Website"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, rid="bat_croat0001")
    run = {**run, "task": "Find small Croatian law firms with a website."}
    deck["recipe"] = {"item": "law firm", "site_column": "Website",
                      "sources": [{"kind": "search", "queries": ["law firms list"]}],
                      "gather": [{"kind": "fetch", "url": "{Website}", "paths": [""]}]}
    deck["extract"] = lambda user: {"items": [
        {"name": "Odvjetnik A", "url": "https://a.example", "fields": {"Name": "Odvjetnik A", "Website": "https://a.example"}},
        {"name": "Advokat B", "url": "https://b.example", "fields": {"Name": "Advokat B", "Website": "https://b.example"}}]}
    deck["tools"] = _Tools(pages={"https://a.example/": "Odvjetnik A, Zagreb"}, results={"law firms list Croatia": (
        "[QUERY: q]\n\n1. Odvjetnik A\n   URL: https://a.example\n   Snippet: Zagreb\n\n"
        "2. Advokat B\n   URL: https://b.example\n   Snippet: Advokat B · Belgrade, Serbia\n")})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Website": f["Website"]}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": run["task"]}, steps[0])
    assert not any("b.example" in u for k, u in deck["tools"].calls if k == "fetch")
    why = {r["name"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert why["Advokat B"] == "not in Croatia (its listing says Belgrade)"


def test_a_list_of_linkedin_profiles_keeps_linkedin():
    people = funnel.normalize_recipe({"columns": ["Name", "LinkedIn URL"], "site_column": "LinkedIn URL",
                                      "sources": [{"kind": "search", "queries": ["x"]}]})
    assert "linkedin.com" not in funnel.skip_sites(people)
    firms = funnel.normalize_recipe({"columns": ["Firm", "Website"], "sources": [{"kind": "search", "queries": ["x"]}]})
    assert "linkedin.com" in funnel.skip_sites(firms) and "x.com" in funnel.skip_sites(firms)


# ── filters that would rule out everything ─────────────────────────────

def test_a_filter_that_rules_out_everything_is_dropped():
    base = funnel.from_contract([{"id": "fm6", "text": "x", "check": {
        "type": "table", "path": "leads.csv", "columns": ["Firm", "Segment"], "exclude_pattern": r"^(?!([A-Z]{2})$).*$",
        "min_per": {"Segment": 10}, "allowed": {"Segment": ["marketing/content agencies", "legal/accounting boutiques"]}}}])
    r = funnel.normalize_recipe({"filters": [
        {"column": "Firm", "op": "not_regex", "value": "(?i)(university|government|agency)"},  # agencies are a group
        {"column": "Firm", "op": "not_regex", "value": "(?i)(university|college)"},
        {"column": "size_band", "op": "size_max", "value": 50}]}, base)
    assert r["filters"] == [{"column": "Firm", "op": "not_regex", "value": "(?i)(university|college)"},
                            {"column": "size_band", "op": "size_max", "value": 50}]


def test_what_names_an_item_on_a_page():
    assert funnel.name_words("P&S International Strategies, LLC") == ["international", "strategies"]
    assert funnel.name_words("U.S. Consulting Services, Inc.") == ["consulting", "services", "inc"]
    words = funnel.name_words("Consultancy Services LLC (USA)")
    assert funnel.names_item(words, "https://www.linkedin.com/company/consultancy-services-usa", "Consultancy Services LLC")
    assert not funnel.names_item(words, "https://www.linkedin.com/news", "LinkedIn News")
    assert not funnel.names_item(words, "https://www.example.com/usa", "Top firms in the USA")  # one word of two
    long = funnel.name_words("USBC US Business Consulting and Investments, LLC")
    assert funnel.names_item(long, "https://usbc.example/", "USBC Business Advisors")
    assert not funnel.names_item(long, "https://news.example/", "Business news") and not funnel.names_item([], "u", "t")


async def test_a_size_read_off_a_page_about_something_else_is_no_size(deck):
    """Two firms were dropped as "201-500 people" read off LinkedIn News, a result of
    their own size search that wasn't about them."""
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_sizeof001")
    recent = _months_back(3)
    _web_deck(deck, recent)
    news = "\n2. LinkedIn News\n   URL: https://www.linkedin.com/news\n   Snippet: Big Co (201-500 employees) hires\n"
    deck["tools"].results.update({
        '"Alpha" site:linkedin.com/company': "[QUERY: q]\n" + news,  # nothing of its own: no size, read on
        '"Gamma" site:linkedin.com/company': "[QUERY: q]\n" + news + _result("Gamma", "https://linkedin.com/company/gamma",
                                                                           "2-10 employees").split("\n", 1)[1]})
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    why = {r["key"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert not any(w.startswith("201-500") for w in why.values()), why
    assert why["beta.example"].startswith("51-200 people")  # its own page still counts
    assert "couldn't confirm Email 'boss@gamma.example'" in why["gamma.example"]  # read and judged, not cut on size
    assert [r["Firm"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["Alpha"]


async def test_a_firm_left_with_its_linkedin_page_has_only_that_page_read(deck):
    """No site of its own found, a firm kept its LinkedIn page as its site, and the
    recipe's pages ("news", "contact") were read off LinkedIn's root: linkedin.com/news
    gave it another company's size."""
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_skipsite1")
    deck["recipe"] = {**WEB_RECIPE, "gather": [{"kind": "fetch", "url": "{Website}", "paths": ["", "news", "contact"]}]}
    page = "https://www.linkedin.com/company/zeta-advisors"
    deck["extract"] = lambda user: {"items": [{"name": "Zeta Advisors", "url": page,
                                               "fields": {"Firm": "Zeta Advisors", "Website": page}}]}
    deck["tools"] = _Tools(pages={page: "Zeta Advisors. Company size 11-50 employees.",
                                  "https://www.linkedin.com/news": "Big Co (201-500 employees) is hiring"},
                           results={"x firms list": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "reason": "no site of its own"} for i, _f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert [u for k, u in deck["tools"].calls if k == "fetch"] == [page]
    item = funnel._read_csv(folder / funnel.ITEMS)[0]
    assert item["status"] == "maybe" and "201-500" not in item["reason"]


async def test_a_retried_setup_leaves_no_stale_note(deck, monkeypatch):
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_retry0001")
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@a.example,a\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}]}
    real, calls = fx.parse_json, []

    def flaky(text):
        calls.append(1)
        return None if len(calls) == 1 else real(text)  # the first answer isn't JSON

    monkeypatch.setattr(fx, "parse_json", flaky)
    await funnel.plan(run)
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any(n.startswith("List funnel:") and n.endswith("The recipe is in shared/recipe.json.") for n in notes)


# ── rule tiers: rows on a looser reading when a group runs short ───────

PAPERS = [{"id": "fm1", "text": "papers.csv: Title, Year, Code, Cites; at least 3", "check": {
    "type": "table", "path": "papers.csv", "columns": ["Title", "Year", "Code", "Cites"], "min_rows": 3,
    "unique": ["Title"]}}]
PAPER_RULES = [{"rule": "about graph search", "kind": "must"},
               {"rule": "links its code", "kind": "prefer", "looser": "no code link; Code left empty",
                "columns": ["Code", "Title"]},  # the key never loosens
               {"rule": "cited 10 times or more", "kind": "prefer", "looser": "any citation count, or none found",
                "columns": ["Cites"]},
               {"rule": "peer reviewed", "kind": "prefer"}]  # no looser reading: a must


def test_rules_are_sorted_into_musts_and_soft_ones():
    r = funnel.normalize_recipe({"rules": PAPER_RULES + [{"rule": ""}, "x"]}, funnel.from_contract(PAPERS))
    assert [(x["rule"], x["kind"]) for x in r["rules"]] == [
        ("about graph search", "must"), ("links its code", "prefer"),
        ("cited 10 times or more", "prefer"), ("peer reviewed", "must")]
    assert r["rules"][1]["columns"] == ["Code"] and r["rules"][2]["looser"].startswith("any citation")
    note = funnel.rules_note(r, {"config": {}})
    assert note.startswith("Rules: must — about graph search; peer reviewed.")
    assert "Loosened only when a group runs short — links its code (looser: no code link; Code left empty)" in note
    assert "never loosens" in funnel.rules_note(r, {"config": {"funnel_loosen": "never"}})
    assert funnel.normalize_recipe({}, funnel.from_contract(PAPERS))["rules"] == [] and funnel.rules_note(
        {"rules": []}, {}) == ""


def _papers(deck, folder):
    (folder / "extra" / "papers.csv").write_text(
        "Title,Year,Code,Cites\nAlpha,2023,https://code.example/alpha,40\nBeta,2023,,3\nGamma,2024,,\n"
        "Delta,2023,,2\nEps,2022,,15\n")
    deck["recipe"] = {"item": "paper", "key": "Title", "sources": [{"kind": "file", "path": "extra/papers.csv"}],
                      "decide": "Graph search papers from 2022 on that link their code and are cited 10+ times.",
                      "rules": PAPER_RULES,
                      "checks": [{"column": "Code", "op": "required"}, {"column": "Code", "op": "in_evidence"},
                                 {"column": "Cites", "op": "min", "value": 10}, {"column": "Cites", "op": "required"}]}

    def decide(items):
        out = []
        for i, f in items:
            row = {"Title": f["Title"], "Year": f["Year"], "Code": f.get("Code", ""), "Cites": f.get("Cites", "")}
            if f["Title"] == "Alpha":
                out.append({"id": i, "verdict": "keep", "row": row})
            elif f["Title"] == "Gamma":  # misses both soft rules
                out.append({"id": i, "verdict": "loose", "row": row, "loosened": ["links its code", "cited 10 times"]})
            elif f["Title"] == "Delta":  # a code link it made up: still checked
                out.append({"id": i, "verdict": "loose", "row": {**row, "Code": "https://code.example/made-up"},
                            "loosened": ["cited 10 times or more"]})
            else:  # Beta (code and cites missing, says only the code rule), Eps (no code)
                out.append({"id": i, "verdict": "loose", "row": row, "loosened": ["links its code"]})
        return out

    deck["decide"] = decide


async def test_a_short_list_takes_rows_on_a_looser_reading_after_every_strict_one(deck):
    run, folder = await _run(deck["store"], PAPERS, rid="bat_loose0001")
    _papers(deck, folder)
    steps = await funnel.plan(run)
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any(n.startswith("Rules: must — about graph search") for n in notes)
    work = (await _go(deck["store"], run, steps[:1]))[0]
    assert "# Rule tiers" in deck["decide_prompts"][0] and "looser: no code link" in deck["decide_prompts"][0]
    assert work["output"].startswith("1 kept of 3, 2 more on a looser reading. Looked at 5 items")
    why = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    # a value given must still be in its source: the link it made up is left out, and then its 2 citations fail
    assert why["Delta"] == ("dropped", "check: Cites 2 is under 10")
    assert "made-up" not in (folder / funnel.LOOSE).read_text() + (folder / funnel.ROWS).read_text()
    assert why["Beta"][0] == "dropped" and "Cites 3 is under 10" in why["Beta"][1]  # loosened only the code rule
    assert why["Gamma"][0] == why["Eps"][0] == "loose"
    # the run panel counts them as ready: a group full of loose rows isn't "0 of N"
    got = funnel.counts(folder, funnel.load_recipe(folder))["groups"][""]
    assert (got["on_list"], got["loose_waiting"]) == (1, 2)
    # nothing for the finishing agent: the loose rows fill what's short
    fresh = await deck["store"].get_run(run["id"])
    assert funnel.prepare_finish(fresh, steps[1])["result"]["output"].startswith("Nothing to finish: rows on a looser")
    merged = (await _go(deck["store"], run, steps[2:]))[0]
    rows = (folder / "papers.csv").read_text().splitlines()
    assert rows == ["Title,Year,Code,Cites", "Alpha,2023,https://code.example/alpha,40", "Eps,2022,,15", "Gamma,2024,,"]
    assert "2 meet the soft rules only by their looser reading" in merged["output"]  # Eps first: one rule loosened
    summary = (folder / funnel.SUMMARY).read_text()
    assert "## On a looser reading" in summary and "| Eps |" in summary and "links its code — looser:" in summary
    statuses = {r["name"]: (r["status"], r["stage"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert statuses["Eps"] == statuses["Gamma"] == ("kept", "loose")
    note = funnel.judge_note(await deck["store"].get_run(run["id"]))
    assert "2 rows meet some rules only by their looser reading" in note and "Those rows: Gamma, Eps." in note
    from captain_claw.flight_deck import first_mate_ledger as ledger
    assert {r["name"]: r["outcome"] for r in ledger.rows_from_funnel(folder)}["Gamma"] == "kept"


async def test_a_factory_that_never_loosens_keeps_every_rule(deck):
    run, folder = await _run(deck["store"], PAPERS, {"funnel_loosen": "never"}, rid="bat_loose0002")
    _papers(deck, folder)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert "# Rule tiers" not in deck["decide_prompts"][0]
    assert (folder / "papers.csv").read_text().splitlines() == ["Title,Year,Code,Cites",
                                                                "Alpha,2023,https://code.example/alpha,40"]
    assert {r["status"] for r in funnel._read_csv(folder / funnel.ITEMS) if r["name"] in ("Gamma", "Eps")} == {"maybe"}
    close = {r["Title"]: r for r in funnel._read_csv(folder / "papers-came-close.csv")}
    assert set(close) == {"Beta", "Gamma", "Delta", "Eps"} and close["Eps"]["Cites"] == "15"
    assert close["Delta"]["Code"] == ""  # the link it made up isn't in what was read: not passed on


def test_loose_rows_count_only_once_enough_items_were_looked_at(tmp_path):
    recipe = funnel.normalize_recipe({"rules": PAPER_RULES}, funnel.from_contract(CONTRACT))
    w = funnel.Work({"id": "r", "config": {}}, {"step_key": "step-1"}, recipe, tmp_path, group="x")
    w.kept, w.loose, w.looked_in = {"x": 1}, {"x": 1}, {"x": 29}
    assert not w.full()  # a strict row may still turn up
    w.looked_in["x"] = 30
    assert w.full()
    w.loosen = False
    assert not w.full()


async def test_a_slow_call_skips_what_it_was_for_and_the_step_goes_on(deck):
    import httpx

    from captain_claw.flight_deck import bat_loop
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_slowcal01")
    recent = _months_back(3)
    _web_deck(deck, recent)
    calls = []

    def extract(user):
        calls.append(1)
        if len(calls) <= 3:
            raise httpx.ReadTimeout("slow")  # the first list of results, tried three times: skipped
        return {"items": [{"name": "Alpha", "url": "https://alpha.example",
                           "fields": {"Firm": "Alpha", "Website": "https://alpha.example"}}]}

    deck["extract"] = extract
    deck["queries"] = ["more x firms"]
    deck["tools"].results["more x firms"] = "more results"
    real_fetch = deck["tools"].fetch

    async def fetch(url):
        if "contact" in url and "gamma" in url:
            raise RuntimeError("boom")
        return await real_fetch(url)

    deck["tools"].fetch = fetch
    steps = await funnel.plan(run)
    work = (await _go(deck["store"], run, steps[:1]))[0]
    assert work["ok"] and [r["Firm"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["Alpha"]
    events = [e for e in await deck["store"].list_events(run["id"]) if e["stage"] == "action"]
    assert any("couldn't pull items out of" in (e.get("data") or {}).get("detail", "")
               or "couldn't pull items out of" in str(e) for e in events)
    # a step that still breaks says so in words Bat doesn't take for "too big for one turn"
    assert funnel._plain_error(httpx.ReadTimeout("x")) == "a call got no answer in time"
    assert not bat_loop.size_failure(funnel._plain_error(httpx.ReadTimeout("x")))
    assert not bat_loop.size_failure(funnel._plain_error(RuntimeError("request timed out")))


# ── an item's own site ─────────────────────────────────────────────────

def test_a_site_is_an_items_own_only_when_its_address_names_it():
    own = funnel.site_names_item
    assert own("Darby Consulting", "darbyconsulting.com") and own("NERA Economic Consulting", "nera.com")
    assert own("2 Visions", "2visions.com") and own("Aryo Consulting Group", "aryocg.com")
    assert own("Boston Consulting Group", "bcg.com") and own("Red Cup Agency", "red-cup.co.uk")
    assert own("U.S. Consulting Services, Inc.", "usconsultingservices.com")  # weak words: two of them
    assert not own("U.S. Consulting Services, Inc.", "acmeconsulting.com")
    for article in ("forbes.com", "advisorindex.com", "syncgtm.com", "example.com"):
        assert not own("Darby Consulting", article), article
    assert not own("Darby Consulting", "") and not own("", "darby.com")
    # any language: accents aside, and a name with no Latin letters by the result's title
    assert own("Bistro Čarobnjak", "carobnjak.hr") and own("Kuća Đurđević", "djurdjevic.hr")
    assert own("Kuća Đurđević", "kucadurdevic.hr") and own("Müller", "mueller.de") and own("Müller", "muller.de")
    assert own("Ristorante Nonna Lucía", "nonnalucia.it") and not own("Müller", "schmidt.de")
    assert own("Ресторан Дубрава", "dubrava.rs", "Ресторан Дубрава — почетна")
    assert not own("Ресторан Дубрава", "news.example", "Најбољи ресторани у граду")
    assert funnel.name_words("Bistro Čarobnjak") == ["bistro", "carobnjak"]


async def test_a_page_about_an_item_is_never_taken_for_its_site(deck):
    """Firms got Forbes, a marketplace, a contact-list site or a directory as their own
    site: a result whose title named them. Then no email was "on its own site"."""
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_ownsite01")
    deck["recipe"] = {"item": "US knowledge firm lead", "site_column": "Website",
                      "sources": [{"kind": "search", "group": "x", "country": "US", "queries": ["x firms list"]}],
                      "gather": [{"kind": "fetch", "url": "{Website}", "paths": [""]}], "extract": ["emails"]}
    deck["extract"] = lambda user: {"items": [{"name": "Darby Consulting", "fields": {"Firm": "Darby Consulting"}},
                                              {"name": "Censeo Partners", "fields": {"Firm": "Censeo Partners"}}]}
    forbes = "[QUERY: q]\n\n1. The best boutique firms: Darby Consulting\n   URL: https://www.forbes.com/lists/x\n   Snippet: -\n"
    deck["tools"] = _Tools(
        pages={"https://darbyconsulting.example": "Darby Consulting. hello@darbyconsulting.example",
               "https://www.forbes.com/": "Forbes"},
        results={"x firms list": "list results",
                 '"Darby Consulting" United States': forbes + "\n2. Darby Consulting | Home\n"
                 "   URL: https://darbyconsulting.example/\n   Snippet: -\n",
                 '"Censeo Partners" United States': forbes.replace("Darby Consulting", "Censeo Partners")})
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "reason": "x"} for i, _f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    searched = [q for k, q in deck["tools"].calls if k == "search"]
    assert '"Darby Consulting" United States' in searched  # the item phrase ("US knowledge firm lead") left out
    fetched = [u for k, u in deck["tools"].calls if k == "fetch"]
    assert fetched == ["https://darbyconsulting.example"]  # never forbes.com
    fields = {json.loads(m["fields"])["Firm"]: json.loads(m["fields"]) for m in funnel._read_csv(folder / funnel.MAYBE)}
    assert fields["Darby Consulting"]["Website"] == "https://darbyconsulting.example"
    assert "Website" not in fields["Censeo Partners"]  # no site of its own found: none, not Forbes


# ── searches for a group ───────────────────────────────────────────────

def test_a_slashed_group_is_searched_by_each_part():
    """Groups 2-4 of a run searched "list of research/analysis shops United States":
    the owner's label, slash and all."""
    assert funnel._group_phrases("research/analysis shops", "firm") == [
        "research shops", "research firms", "analysis shops", "analysis firms"]
    assert funnel._group_phrases("marketing/content agencies", "firm") == ["marketing agencies", "content agencies"]
    assert funnel._group_phrases("vegan/vegetarian restaurants", "restaurant") == [
        "vegan restaurants", "vegetarian restaurants"]
    assert funnel._group_phrases("legal/accounting boutiques", "US knowledge firm lead") == [
        "legal boutiques", "accounting boutiques"]  # a long item phrase isn't a noun to borrow
    assert funnel._group_phrases("boutique consultancies", "company") == ["boutique consultancies"]
    qs = funnel._group_queries("research/analysis shops", "United States", None, "firm")
    assert qs[:4] == ["list of research shops United States", "list of research firms United States",
                      "list of analysis shops United States", "list of analysis firms United States"]
    assert not any("/" in q for q in qs) and len(qs) == len(set(qs)) <= 24
    # a model's own searches for a group count when they use a word the setup gave for it
    # ("law firm" for "legal", in any language); no word list of code's guesses for it
    def tidied(words):
        r = funnel.normalize_recipe({"sources": [
            {"kind": "search", "group": "x", "queries": ["boutique law firm directory", "CPA firms list"]},
            {"kind": "search", "group": "y", "queries": ["boutique law firm directory", "CPA firms list"]}]},
            {**funnel.from_contract(CONTRACT), "groups": {"x": 2, "y": 2}})
        r["sources"][0]["group"] = "legal/accounting boutiques"
        r["sources"][1]["group"] = "research/analysis shops"  # a copy of another group's: written for it
        r["group_words"] = words
        return funnel.tidy_recipe(r, "Find firms.")

    r = tidied({"legal/accounting boutiques": ["law firm", "CPA"]})
    assert r["sources"][0]["queries"] == ["boutique law firm directory", "CPA firms list"]
    assert r["sources"][1]["queries"][:2] == ["list of research shops", "list of analysis shops"]  # item: "item"
    assert tidied({})["sources"][0]["queries"][0] == "list of legal boutiques"  # no word for it given
    assert funnel._group_words("Odvjetnički uredi", ["odvjetnik", "pravni ured"]) == ["odvjet", "uredi", "pravni", "ured"]


# ── unshown is not broken ──────────────────────────────────────────────

FIRM_LEADS = [{"id": "fm1", "text": "leads.csv; at least 3", "check": {
    "type": "table", "path": "leads.csv", "min_rows": 3, "unique": ["Firm"],
    "columns": ["Firm", "Website", "State", "Employees", "EmployeesSourceURL", "Email", "EmailSourceURL",
                "SignalDate", "Signal"]}}]
FIRM_LEADS_RULES = [  # as a setup model sorted a real charter: the shown size kept as a must
    {"rule": "US knowledge firm headquartered in the US", "kind": "must"},
    {"rule": "2-50 employees shown by its LinkedIn company page size band or its own team page", "kind": "must"},
    {"rule": "not a university or government body", "kind": "must"},
    {"rule": "own website prints a founder's or partner's email or a shared inbox like info@", "kind": "must"},
    {"rule": "something dated in the last 24 months showing interest in AI", "kind": "prefer",
     "looser": "any mention of AI, undated", "columns": ["SignalDate", "Signal"]}]


def test_only_a_must_about_something_measured_may_go_unshown():
    """A run dropped 45 of 87 firms for "no employee count shown": not found was read as
    broken. Any list: a count, an amount, a date or a rating many items never publish
    may go unshown on a looser reading, and a way to reach the item may be its own
    site's contact page; what an item is and where it is may not."""
    r = funnel.normalize_recipe({"rules": FIRM_LEADS_RULES}, funnel.from_contract(FIRM_LEADS))
    loosen = {x["rule"][:24] for x in funnel.loosenable(r)}
    assert loosen == {"2-50 employees shown by ", "something dated in the l", "own website prints a fou"}
    email = r["rules"][3]
    assert funnel.looser_of(email) == funnel.CONTACT_LOOSER and "never an address guessed" in funnel.CONTACT_LOOSER
    size = r["rules"][1]
    assert size["columns"] == ["Employees", "EmployeesSourceURL"] and not size["protected"]  # inferred from its words
    assert funnel.looser_of(size).startswith("not shown in what was read")
    assert "2-50 employees shown by its LinkedIn" in funnel.rules_note(r, {"config": {}}).split("Loosened")[1]
    for rule, protected in (("rated 4.5 stars or more", False), ("over 1,000 members", False),
                            ("founded before 2015", False), ("revenue under $10M per its filings", False),
                            ("located in Zagreb", True), ("serves vegan food", True), ("peer reviewed", True),
                            ("lists a phone number", False), ("headquartered in Canada", True)):
        got = funnel.normalize_recipe({"rules": [{"rule": rule, "kind": "must"}]},
                                      funnel.from_contract(FIRM_LEADS))["rules"][0]["protected"]
        assert got is protected, rule


async def test_a_fact_a_firm_doesnt_publish_is_no_reason_to_drop_it(deck):
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_unshown01")
    (folder / "extra" / "firms.csv").write_text(
        "Firm,Website,State,Employees,Email,Signal\nA,https://a.example,TX,12,info@a.example,AI pilot\n"
        "B,https://b.example,NY,,info@b.example,AI pilot\nC,https://c.example,CA,9,,AI pilot\n"
        "D,https://d.example,WA,,info@d.example,\nE,https://e.example,OR,120,info@e.example,AI pilot\n")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms, 2-50 people shown, an email on the site, interest in AI.", "rules": FIRM_LEADS_RULES,
                      "checks": [{"column": "Employees", "op": "min", "value": 2},
                                 {"column": "Employees", "op": "max", "value": 50},
                                 {"column": "EmployeesSourceURL", "op": "required"}, {"column": "Email", "op": "required"}]}

    def decide(items):
        out = []
        for i, f in items:
            row = {**f, "EmployeesSourceURL": "https://a.example/team" if f["Firm"] == "A" else ""}
            size = "2-50 employees shown by its LinkedIn company page size band or its own team page"
            if f["Firm"] == "A":
                out.append({"id": i, "verdict": "keep", "row": row})
            elif f["Firm"] == "C":  # no email printed (its contact page is how to reach it), size unshown
                out.append({"id": i, "verdict": "loose", "row": row, "loosened": ["own website prints a founder's", size]})
            elif f["Firm"] == "D":  # size and AI both unshown
                out.append({"id": i, "verdict": "loose", "row": row, "loosened": [size, "something dated"]})
            else:  # B: size unshown; E: says 120, so it's shown to break the rule
                out.append({"id": i, "verdict": "loose", "row": row, "loosened": [size]})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    prompt = deck["decide_prompts"][0]
    assert "Never loosened" in prompt and "own website prints a founder's" in prompt.split("May be met")[1]
    assert "US knowledge firm headquartered in the US" in prompt.split("Never loosened")[1].split("May be met")[0]
    assert "Unshown is not broken" in prompt and "— looser: not shown in what was read" in prompt
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A"][0] == "kept" and got["B"][0] == got["D"][0] == "loose"
    assert got["C"][0] == "loose"  # no address printed, reached by its contact page
    assert got["E"][0] == "dropped" and "Employees 120 is over 50" in got["E"][1]


# ── free text against fixed values ─────────────────────────────────────

def test_a_listings_words_are_never_held_to_a_fixed_value(tmp_path):
    """A run dropped 226 of 264 firms before reading them: "Segment is Boutique consulting
    firm" didn't equal "boutique consultancies" exactly. A list column's fixed values are
    the judging call's to set; the row's check holds them."""
    r = funnel.normalize_recipe({"filters": [
        {"column": "Segment", "op": "in", "value": ["x", "y"]},
        {"column": "segment", "op": "not_in", "value": ["z"]},
        {"column": "Country", "op": "in", "value": ["United States"]},
        {"column": "Employees", "op": "max", "value": 50},
        {"column": "size_band", "op": "size_max", "value": 50}]}, funnel.from_contract(CONTRACT))
    assert [(f["column"], f["op"]) for f in r["filters"]] == [("Country", "in"), ("Employees", "max"),
                                                              ("size_band", "size_max")]
    assert {"column": "Segment", "op": "in", "value": ["x", "y"]} in r["checks"]  # still held on the row
    w = funnel.Work({"id": "r", "config": {}}, {"step_key": "step-1"}, r, tmp_path, group="x")
    item = lambda **f: {"fields": {"name": "A", **f}, "key": "a"}  # noqa: E731
    assert w.passes(item(Country="United States of America")) == ""  # it shares a word with a value it may take
    assert w.passes(item(Country="Canada")) == "Country is Canada"
    assert w.passes(item(Employees="11-50")) == ""  # a range, never "1150"
    assert w.passes(item(Employees="51-200")).startswith("Employees is 51-200")
    assert funnel.number_span("11 to 50 employees") == (11, 50) and funnel.number_span("1,001+")[0] == 1001
    assert funnel.number_span("$2.5M") == (2.5, 2.5) and funnel.number_span("none") is None
    checks = {"checks": [{"column": "Employees", "op": "max", "value": 50}, {"column": "Employees", "op": "min", "value": 2}]}
    assert funnel.check_row(checks, {"Employees": "11-50"}, {}) == ""
    assert funnel.check_row(checks, {"Employees": "51-200"}, {}) == "Employees 51-200 is over 50"
    assert funnel.check_row(checks, {"Employees": "1"}, {}) == "Employees 1 is under 2"


async def test_a_maybe_that_lacks_only_unshown_facts_is_loose(deck):
    """The judging call kept answering maybe for "meets every rule; employee count not
    shown". It now names what it couldn't see, and code decides: loose when each of
    those may go unshown, maybe when a rule that never loosens is among them."""
    assert funnel.DECIDE_LOOSE.startswith(funnel.DECIDE_SYSTEM[:200]) and '"missing"' in funnel.DECIDE_LOOSE
    assert "keep|loose|maybe|drop" in funnel.DECIDE_LOOSE and "loose" not in funnel.DECIDE_SYSTEM
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_missing01")
    (folder / "extra" / "firms.csv").write_text(
        "Firm,Website,State,Email\nA,https://a.example,TX,info@a.example\nB,https://b.example,NY,\n"
        "C,https://c.example,CA,info@c.example\nD,https://d.example,WA,info@d.example\n")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms, 2-50 people shown, an email on the site.", "rules": FIRM_LEADS_RULES,
                      "checks": [{"column": "EmployeesSourceURL", "op": "required"}, {"column": "Email", "op": "required"}]}
    size = "2-50 employees shown by its LinkedIn company page size band or its own team page"

    def decide(items):
        out = []
        for i, f in items:
            if f["Firm"] == "A":  # only the size isn't shown
                out.append({"id": i, "verdict": "maybe", "row": f, "missing": [size]})
            elif f["Firm"] == "B":  # no email either: a contact page will do on a looser reading
                out.append({"id": i, "verdict": "maybe", "row": f, "missing": [size, "own website prints"]})
            elif f["Firm"] == "C":  # names nothing it lacks
                out.append({"id": i, "verdict": "maybe", "row": f, "reason": "unsure"})
            else:  # D: no row to put on the list
                out.append({"id": i, "verdict": "maybe", "missing": [size]})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"A": "loose", "B": "loose", "C": "maybe", "D": "maybe"}
    assert "Employees" in funnel._read_csv(folder / funnel.LOOSE)[0] and funnel._read_csv(folder / funnel.LOOSE)[0]["Firm"] == "A"


async def test_a_drop_must_show_the_fact(deck):
    """A run dropped 32 firms for "no dated AI signal found": a soft rule, and not finding
    a fact isn't breaking it. A drop now names the rule and quotes what was read that
    breaks it; one that quotes nothing read, for rules that may go unshown, isn't a drop."""
    assert '"broken"' in funnel.DECIDE_LOOSE and '"shown"' in funnel.DECIDE_LOOSE
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_dropfact1")
    (folder / "extra" / "firms.csv").write_text(
        "Firm,Website,State,Email,Note\nA,https://a.example,TX,info@a.example,\n"
        "B,https://b.example,NY,info@b.example,Team of 120 people\nC,https://c.example,CA,,\n"
        "D,https://d.example,WA,info@d.example,\nE,https://e.example,OR,info@e.example,\n")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms, 2-50 people, an email on the site, interest in AI.", "rules": FIRM_LEADS_RULES,
                      "checks": [{"column": "Email", "op": "required"}]}
    size = "2-50 employees shown by its LinkedIn company page size band or its own team page"
    ai = "something dated in the last 24 months showing interest in AI"

    def decide(items):
        out = []
        for i, f in items:
            row = {k: f[k] for k in ("Firm", "Website", "State", "Email")}
            if f["Firm"] == "A":  # "not found" quoted as if it were a fact
                out.append({"id": i, "verdict": "drop", "row": row, "broken": [ai], "shown": "no AI signal found"})
            elif f["Firm"] == "B":  # really shown to be too big
                out.append({"id": i, "verdict": "drop", "row": row, "broken": [size], "shown": "Team of 120 people"})
            elif f["Firm"] == "C":  # "no email" with nothing shown: not broken, its contact page will do
                out.append({"id": i, "verdict": "drop", "row": row, "broken": ["own website prints a founder's"]})
            elif f["Firm"] == "D":  # names no rule: can't tell
                out.append({"id": i, "verdict": "drop", "row": row, "reason": "no"})
            else:  # E: nothing quoted at all, and no row to put on the list
                out.append({"id": i, "verdict": "drop", "broken": [ai]})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"A": "loose", "B": "dropped", "C": "loose", "D": "dropped", "E": "maybe"}


async def test_a_firm_reached_by_its_contact_page_never_gets_a_guessed_address(deck):
    """The email rule loosens to a contact page on the firm's own site, the address
    left empty. An address given must still be on the page it cites."""
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_contact01")
    (folder / "extra" / "firms.csv").write_text("Firm,Website,State\nA,https://a.example,TX\nB,https://b.example,NY\n")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms with an email on the site.", "rules": FIRM_LEADS_RULES[3:4],
                      "checks": [{"column": "Email", "op": "required"},
                                 {"column": "Email", "op": "in_evidence", "source_column": "EmailSourceURL"}]}
    rule = "own website prints a founder's"

    def decide(items):
        return [{"id": i, "verdict": "loose", "loosened": [rule],
                 "row": {**f, "EmailSourceURL": f"{f['Website']}/contact",
                         "Email": "guess@b.example" if f["Firm"] == "B" else ""}} for i, f in items]

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A"][0] == "loose"  # reached by its contact page, no address
    # an address not on a page read never passes: left out, B is reached by its contact page too
    assert got["B"][0] == "loose" and "couldn't confirm Email 'guess@b.example'" in got["B"][1]
    assert {r["Firm"]: r["Email"] for r in funnel._read_csv(folder / funnel.LOOSE)}["B"] == ""
    assert "Never loosened" not in funnel.conventions(funnel.load_recipe(folder)).split("## Rules")[0]
    assert "contact page or form" in funnel.conventions(funnel.load_recipe(folder))


async def test_the_finishing_step_may_add_rows_on_a_looser_reading(deck):
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_finloose1")
    (folder / "extra" / "firms.csv").write_text("Firm,Website,State\nA,https://a.example,TX\n")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms.", "rules": FIRM_LEADS_RULES, "target": 3,
                      "checks": [{"column": "Email", "op": "required"}, {"column": "EmployeesSourceURL", "op": "required"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "reason": "x", "missing": ["US knowledge firm"]}
                                    for i, _f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fresh = await deck["store"].get_run(run["id"])
    work = funnel.prepare_finish(fresh, steps[1])["work"]
    assert "_loosened column" in work and "Don't spend long proving" in work
    finish = folder / funnel.FINISH
    assert finish.read_text().splitlines()[0].endswith(",_loosened")
    with finish.open("a", encoding="utf-8") as fh:
        fh.write("Firm A2,https://a2.example,TX,,,,https://a2.example/contact,,,"
                 "own website prints a founder's; 2-50 employees shown\n")
    added, refused, _ = await funnel.check_finish(fresh, steps[1], folder, funnel.load_recipe(folder))
    assert (added, refused) == (1, 0)
    assert [r["Firm"] for r in funnel._read_csv(folder / funnel.LOOSE)] == ["Firm A2"]  # waits with the loose rows
    assert funnel._read_csv(folder / funnel.ROWS) == []


# ── a complete list is done; judgement columns are filled ──────────────

PITCH = [{"id": "fm1", "text": "pitch.csv; at least 2", "check": {
    "type": "table", "path": "pitch.csv", "min_rows": 2, "unique": ["Firm"],
    "columns": ["Firm", "Website", "Email", "EmailType", "CaptainFeature", "CaptainReason"],
    "allowed": {"CaptainFeature": ["Onboarding", "Flows"], "EmailType": ["personal", "shared"]}}}]


def test_columns_about_the_owners_offer_are_judgement_columns():
    r = funnel.normalize_recipe({"rules": [{"rule": "an email on its own site", "kind": "must",
                                            "columns": ["Email", "EmailType"]}],
                                 "reference": ["extra/Captain.md"]}, funnel.from_contract(PITCH))
    ref = "# Captain\nOnboarding teaches agents your documents. Flows run steps."
    assert funnel.judgement_columns(r, ref) == ["CaptainFeature", "CaptainReason"]  # EmailType is the email rule's
    bare = funnel.normalize_recipe({"reference": ["extra/Captain.md"]}, funnel.from_contract(PITCH))
    assert funnel.judgement_columns(bare, ref) == ["CaptainFeature", "CaptainReason"]  # "personal" isn't in it
    assert funnel.judgement_columns(bare, "") == ["CaptainFeature", "CaptainReason"]  # named after the file


async def test_a_row_left_without_its_pitch_is_nearly_there(deck):
    """A run put 40 rows on its list, 33 of them with CaptainFeature and CaptainReason
    empty: a reason "tied to the firm's signal" was skipped when no signal was shown."""
    run, folder = await _run(deck["store"], PITCH, rid="bat_pitch0001")
    (folder / "extra" / "firms.csv").write_text("Firm,Website\nA,https://a.example\nB,https://b.example\n")
    (folder / "extra" / "Captain.md").write_text("# Captain\n\nOnboarding teaches agents your documents.")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "reference": ["extra/Captain.md"], "decide": "Firms.",
                      "checks": [{"column": "CaptainFeature", "op": "in", "value": ["Onboarding", "Flows"]}]}

    def decide(items):
        return [{"id": i, "verdict": "keep", "row": {**f, "CaptainFeature": "Onboarding",
                                                     "CaptainReason": "their guides" if f["Firm"] == "A" else ""}}
                for i, f in items]

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A"][0] == "kept" and got["B"] == ("maybe", "left CaptainReason empty")
    assert "always filled for a row you keep" in funnel.DECIDE_SYSTEM


async def test_a_complete_list_settles_the_review(deck):
    """Bat's reviewers judged a complete funnel list against the charter as written,
    and fix rounds deleted its rows on a looser reading to hunt strict ones."""
    run, folder = await _run(deck["store"], PAPERS, rid="bat_settle001")
    _papers(deck, folder)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    fresh = await deck["store"].get_run(run["id"])
    note = funnel.review_settled(fresh, "- Reviewer: every row needs a code link")
    assert note.startswith("The list is complete and passes every check by code (2 rows on a looser reading")
    summary = (folder / funnel.SUMMARY).read_text()
    assert "## Open points from the review" in summary and "every row needs a code link" in summary
    funnel.review_settled(fresh, "- again")
    assert (folder / funnel.SUMMARY).read_text().count("## Open points from the review") == 1
    assert funnel.review_settled({**fresh, "config": {}}, "x") == ""  # not a funnel run


async def test_a_short_list_leaves_the_review_to_go_on(deck):
    run, folder = await _run(deck["store"], PAPERS, {"funnel_loosen": "never"}, rid="bat_settle002")
    _papers(deck, folder)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert funnel.review_settled(await deck["store"].get_run(run["id"]), "- short") == ""  # 1 of 3: fix rounds may add


# ── what the setup tells code, in any language ─────────────────────────

def test_the_setups_tags_tell_code_what_each_rule_is_about():
    """The rule tiers relied on English words ("employees", "email") to tell a count
    or a way to reach the item from what it is: the setup tags them now, any language."""
    r = funnel.normalize_recipe({
        "rules": [{"rule": "sjedište u Hrvatskoj", "kind": "must", "fact": "place"},
                  {"rule": "odvjetnički ured", "kind": "must", "fact": "identity"},
                  {"rule": "od 2 do 50 zaposlenika", "kind": "must", "fact": "measured"},
                  {"rule": "e-pošta na vlastitoj stranici", "kind": "must", "fact": "contact"},
                  {"rule": "nikad ne pogađaj adresu", "kind": "must", "fact": "safety"},
                  {"rule": "2-50 employees", "kind": "must"}],  # no tag: read from its words
        "columns": [{"name": "Firm"}, {"name": "CaptainReason", "from": "owner"}, {"name": "Pitch", "from": "owner"}],
        "skip_sites": ["https://www.lawyer-directory.example/x", "not a site"]},
        funnel.from_contract(FIRM_LEADS))
    loose = [x["rule"] for x in funnel.loosenable(r)]
    assert loose == ["od 2 do 50 zaposlenika", "e-pošta na vlastitoj stranici", "2-50 employees"]
    assert funnel.looser_of(r["rules"][3]) == funnel.CONTACT_LOOSER
    assert r["skip_sites"] == ["lawyer-directory.example"] and "lawyer-directory.example" in funnel.skip_sites(r)
    assert r["owner_columns"] == []  # not columns of this list
    assert funnel.judgement_columns(funnel.normalize_recipe(
        {"columns": [{"name": "CaptainFeature", "from": "owner"}, {"name": "EmailType"}], "reference": ["extra/Captain.md"]},
        funnel.from_contract(PITCH)), "personal shared Onboarding") == ["CaptainFeature", "CaptainReason"]


def test_a_rule_names_its_columns_by_its_own_words_too():
    """A run made FounderOrPartner required while the email rule named only Email and
    EmailSourceURL: firms reached by their contact page were dropped for no founder."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv", "columns": [
        "Firm", "Website", "FounderOrPartner", "Email", "EmailSourceURL"]}}]
    r = funnel.normalize_recipe({"site_column": "Website", "rules": [
        {"rule": "its own website prints a founder's or partner's email", "kind": "must",
         "columns": ["Email", "EmailSourceURL"]}]}, funnel.from_contract(contract))
    assert r["rules"][0]["columns"] == ["Email", "EmailSourceURL", "FounderOrPartner"]  # never Website, its site


async def test_columns_from_the_owners_material_are_filled_not_left_empty(deck):
    run, folder = await _run(deck["store"], PITCH, rid="bat_ownerf001")
    (folder / "extra" / "firms.csv").write_text("Firm,Website\nA,https://a.example\nB,https://b.example\n")
    (folder / "extra" / "Captain.md").write_text("# Captain\n\nOnboarding teaches agents your documents.")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "reference": ["extra/Captain.md"], "decide": "Firms.",
                      "checks": [{"column": "CaptainFeature", "op": "in", "value": ["Onboarding", "Flows"]}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {**f, "CaptainFeature": "", "CaptainReason": ""}}
                                    for i, f in items]
    deck["owner"] = lambda user: {"1": {"CaptainFeature": "Onboarding", "CaptainReason": "their many guides"}}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A"][0] == "kept"  # filled by the one short call
    assert got["B"] == ("maybe", "left CaptainFeature, CaptainReason empty")  # still empty: nearly there
    assert deck["asked"].count("owner") == 1
    rows = funnel._read_csv(folder / funnel.ROWS)
    assert rows[0]["CaptainReason"] == "their many guides"


async def test_a_rule_that_never_loosens_needs_something_shown(deck):
    """A run put 12 rows on its list with no State, under the never-loosened rule
    "headquartered in the US": nothing shown for a strict rule leaves a row nearly there."""
    run, folder = await _run(deck["store"], FIRM_LEADS, rid="bat_shown0001")
    (folder / "extra" / "firms.csv").write_text("Firm,Website,State\nA,https://a.example,TX\nB,https://b.example,\n")
    rules = [{"rule": "headquartered in the US", "kind": "must", "fact": "place", "columns": ["State"]},
             {"rule": "a founder's email on its own site", "kind": "must", "fact": "contact", "columns": ["Email"]},
             {"rule": "never guess an email", "kind": "must", "fact": "safety", "columns": ["Email"]}]
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "decide": "US firms.", "rules": rules}
    deck["decide"] = lambda items: [{"id": i, "verdict": "loose", "row": f, "loosened": ["a founder's email"]}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A"][0] == "loose"  # its email may go unshown; the safety rule's column doesn't count
    assert got["B"] == ("maybe", "shows nothing for: headquartered in the US (State empty)")
    recipe = funnel.load_recipe(folder)
    assert funnel.unshown_musts(recipe, {"State": "TX"}) == [] and funnel.unshown_musts(recipe, {}) != []


# ── facts read, a stated size, a site abroad, today's date, names ──────

SITES = [{"id": "fm1", "text": "firms.csv; at least 2", "check": {
    "type": "table", "path": "firms.csv", "min_rows": 2, "unique": ["Firm"],
    "columns": ["Firm", "Website", "State", "Email", "EmailType", "EmailSourceURL"],
    "allowed": {"EmailType": ["personal", "shared"]}}}]


def _sites_deck(deck, pages, decide=None):
    deck["recipe"] = {"item": "firm", "key": "Firm", "site_column": "Website",
                      "sources": [{"kind": "search", "group": "", "country": "US", "queries": ["x firms list"]}],
                      "filters": [{"column": "size_band", "op": "size_max", "value": 50}],
                      "gather": [{"kind": "fetch", "url": "{Website}", "paths": ["", "contact"]}],
                      "extract": ["emails"], "decide": "US firms of 2-50 people."}
    names = sorted({u.split("//")[1].split(".")[0] for u in pages})
    deck["extract"] = lambda user: {"items": [{"name": n.title(), "url": f"https://{n}.example",
                                               "fields": {"Firm": n.title(), "Website": f"https://{n}.example"}}
                                              for n in names]}
    deck["tools"] = _Tools(pages=pages, results={"x firms list": "list results"})
    deck["decide"] = decide or (lambda items: [{"id": i, "verdict": "keep", "row": {**f}} for i, f in items])


async def test_facts_read_on_its_own_pages_fill_the_cells_left_empty(deck):
    """11 firms had an email printed on their own page and 9 a state, yet the judging
    call left those cells empty: code fills what it read, from the item's own pages only."""
    run, folder = await _run(deck["store"], SITES, rid="bat_facts0001")
    run = {**run, "task": "Find US firms."}
    _sites_deck(deck, {"https://alpha.example": "Alpha. 500 Main St, Austin, TX 78701",
                       "https://alpha.example/contact": "Write to hello@alpha.example or ann@other.example",
                       "https://beta.example": "Beta in Washington DC", "https://beta.example/contact": "ben@beta.example"})
    steps = await funnel.plan(run)
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": run["task"]}, steps[0])
    rows = {r["Firm"]: r for r in funnel._read_csv(folder / funnel.ROWS)}
    a, b = rows["Alpha"], rows["Beta"]
    assert (a["Email"], a["EmailSourceURL"], a["EmailType"], a["State"]) == (
        "hello@alpha.example", "https://alpha.example/contact", "shared", "TX")
    assert (b["Email"], b["EmailType"], b["State"]) == ("ben@beta.example", "personal", "")  # "Washington DC" is no address


async def test_a_size_its_pages_state_drops_it_and_so_does_a_site_abroad(deck):
    """Kleinberg Kaplan's own about page said "more than 100 lawyers and staff"; a South
    African agency (+27, .co.za) was listed as Californian."""
    run, folder = await _run(deck["store"], SITES, rid="bat_sizeabr01")
    run = {**run, "task": "Find US firms of 2-50 people."}
    _sites_deck(deck, {"https://big.example": "Big LLP has more than 100 lawyers and staff in New York.",
                       "https://big.example/contact": "x",
                       "https://far.example": "Far agency. Call +27 58 303 1234", "https://far.example/contact": "y",
                       "https://good.example": "Good, a team of 12. 9 Elm St, Boise, ID 83702",
                       "https://good.example/contact": "z"})
    steps = await funnel.plan(run)
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": run["task"]}, steps[0])
    why = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert why["Big"][0] == "dropped" and why["Big"][1].startswith("100 people, said on https://big.example")
    assert why["Far"] == ("dropped", "its own site points to South Africa, not the United States")
    assert why["Good"][0] == "kept"
    assert deck["decide_prompts"][0].startswith("# Today\n2")  # the judge knows what day it is


def test_the_charters_date_window_holds_its_date_column():
    r = funnel.normalize_recipe({"checks": [{"column": "SignalDate", "op": "regex", "value": r"^\d{4}"}]},
                                funnel.from_contract([{"id": "a", "text": "x", "check": {
                                    "type": "table", "path": "x.csv", "columns": ["Firm", "SignalDate"]}}]))
    funnel.tidy_recipe(r, "Firms that posted something in the last 2 years.")
    assert {"column": "SignalDate", "op": "within_months", "value": 24} in r["checks"]


def test_a_legal_form_never_makes_two_items_of_one_name():
    """An earlier run dropped "Lux Insights Inc."; the next took "Lux Insights" as new."""
    assert funnel.norm_key("Lux Insights Inc.") == funnel.norm_key("lux insights") == "lux insights"
    assert funnel.norm_key("Acme, LLC") == "acme" and funnel.norm_key("Bistro d.o.o.") == "bistro"
    assert funnel.norm_key("Inc") == "inc" and funnel.norm_key("https://www.acme.example/x") == "acme.example"


async def test_the_owner_fill_reads_what_was_read_about_the_item(deck):
    run, folder = await _run(deck["store"], PITCH, rid="bat_ownread01")
    (folder / "extra" / "firms.csv").write_text("Firm,Website\nA,https://a.example\n")
    (folder / "extra" / "Captain.md").write_text("# Captain\n\nOnboarding teaches agents your documents.")
    deck["recipe"] = {"item": "firm", "key": "Firm", "sources": [{"kind": "file", "path": "extra/firms.csv"}],
                      "reference": ["extra/Captain.md"], "decide": "Firms.",
                      "checks": [{"column": "CaptainFeature", "op": "in", "value": ["Onboarding", "Flows"]}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {**f}} for i, f in items]
    seen = []
    deck["owner"] = lambda user: seen.append(user) or {"1": {"CaptainFeature": "Onboarding", "CaptainReason": "x"}}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert seen and "read: Firm: A Website: https://a.example" in seen[0]


# ── an item that fits another group ────────────────────────────────────

KITCHENS = [{"id": "fm1", "text": "restorani.csv; po 2 u skupini", "check": {
    "type": "table", "path": "restorani.csv", "columns": ["Naziv", "Kuhinja", "Adresa"],
    "min_per": {"Kuhinja": 2}, "allowed": {"Kuhinja": ["hrvatska kuhinja", "talijanska kuhinja"]}}}]


async def test_an_item_that_fits_another_group_is_that_groups_find(deck):
    """The Croatian step dropped an Italian restaurant for not being Croatian, though
    Italian was the next group of the same list."""
    run, folder = await _run(deck["store"], KITCHENS, rid="bat_othergr01")
    (folder / "extra" / "hr.csv").write_text("Naziv,Adresa\nKonoba A,Ilica 1\nPizzeria B,Ilica 2\nTrattoria C,Ilica 3\n"
                                             "Tvornica D,Ilica 4\n")
    (folder / "extra" / "it.csv").write_text("Naziv,Adresa\nOsteria E,Ilica 5\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja",
                      "group_words": {"talijanska kuhinja": ["italian", "talijanski", "pizzeria"],
                                      "hrvatska kuhinja": ["croatian", "hrvatski", "konoba"]},
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"},
                                  {"kind": "file", "path": "extra/it.csv", "group": "talijanska kuhinja"}],
                      "decide": "Restorani u Zagrebu."}

    def decide(items):
        out = []
        for i, f in items:
            n = f["Naziv"]
            if n == "Konoba A":
                out.append({"id": i, "verdict": "keep", "row": {**f, "Kuhinja": "hrvatska kuhinja"}})
            elif n == "Pizzeria B":  # kept, its group named loosely
                out.append({"id": i, "verdict": "keep", "row": {**f, "Kuhinja": "Italian"}})
            elif n == "Trattoria C" and "talijanska kuhinja. An item" in deck["decide_prompts"][-1]:
                out.append({"id": i, "verdict": "keep", "row": {**f, "Kuhinja": "talijanska kuhinja"}})
            elif n == "Trattoria C":  # the Croatian step drops it, naming the other group
                out.append({"id": i, "verdict": "drop", "reason": "Talijanski restoran, ne hrvatski."})
            elif n == "Tvornica D":
                out.append({"id": i, "verdict": "drop", "reason": "Not a restaurant: a factory."})
            else:
                out.append({"id": i, "verdict": "keep", "row": {**f, "Kuhinja": "talijanska kuhinja"}})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:2])
    assert "An item that fits another of these groups instead is that group's find" in deck["decide_prompts"][0]
    rows = {r["Naziv"]: r for r in funnel._read_csv(folder / funnel.ROWS)}
    assert rows["Pizzeria B"]["Kuhinja"] == rows["Pizzeria B"]["_group"] == "talijanska kuhinja"  # named as the list does
    assert rows["Trattoria C"]["_group"] == "talijanska kuhinja"  # handed over, then kept by the Italian step
    items = funnel._read_csv(folder / funnel.ITEMS)
    assert [(r["status"], r["group"]) for r in items if r["name"] == "Trattoria C"] == [
        ("handed", "talijanska kuhinja"), ("kept", "talijanska kuhinja")]
    assert next(r for r in items if r["name"] == "Tvornica D")["status"] == "dropped"  # fits none: dropped


def test_a_value_on_any_page_of_the_items_own_site_is_in_evidence():
    """A restaurant's email on its contact page was refused because the row cited its
    homepage (its Web column)."""
    recipe = {"checks": [{"column": "Email", "op": "in_evidence", "source_column": "Web"}]}
    read = {"https://restac.example": "Dobro došli", "https://restac.example/kontakt": "Pišite na info@restac.example"}
    assert funnel.check_row(recipe, {"Email": "info@restac.example", "Web": "https://restac.example"}, read) == ""
    assert funnel.check_row(recipe, {"Email": "info@restac.example", "Web": "https://www.restac.example/"}, read) == ""
    assert "isn't in" in funnel.check_row(recipe, {"Email": "made@up.example", "Web": "https://restac.example"}, read)
    # another site's pages, or a social site's, are no item's own: the cited page itself
    social = {"https://www.instagram.com/other": "info@restac.example"}
    assert "isn't a page Captain read" in funnel.check_row(
        recipe, {"Email": "info@restac.example", "Web": "https://www.instagram.com/restac"}, social)


def test_groups_named_by_a_word_every_group_shares_are_no_group(tmp_path):
    """"Kuhinja nije točno jedna od tri skupine (navedeno hrvatska, latinoamerička i
    azijska)": a fusion bar dropped in the Croatian step fits the Asian group."""
    recipe = funnel.normalize_recipe({"group_by": "Kuhinja"}, funnel.from_contract([{"id": "a", "text": "x", "check": {
        "type": "table", "path": "r.csv", "columns": ["Naziv", "Kuhinja"], "min_per": {"Kuhinja": 2},
        "allowed": {"Kuhinja": ["hrvatska kuhinja", "talijanska kuhinja", "azijska kuhinja"]}}}]))
    w = funnel.Work({"id": "r", "config": {}}, {"step_key": "step-1"}, recipe, tmp_path, group="hrvatska kuhinja")
    assert w._other_group("Kuhinja nije točno jedna od tri skupine (navedeno hrvatska, latinoamerička i azijska).") \
        == "azijska kuhinja"
    assert w._other_group("Kuhinja nije hrvatska.") == ""  # names no other group
    assert w._other_group("Talijanska i azijska fuzija.") == ""  # two: no guess
    assert w._group_named("Azijska") == "azijska kuhinja" and w._group_named("kuhinja") == ""


def test_a_word_inside_a_groups_phrases_doesnt_name_it(tmp_path):
    """The Italian step handed a hotel's restaurant ("Hotelski restoran") and a fast-food
    place ("Fast food") to the Croatian group: "restoran" and "food" were words of its
    phrases ("restoran hrvatske kuhinje", "traditional croatian food"). And breaking
    another rule of what the item is (not a hotel's) is no reason to hand it over."""
    words = {"hrvatska kuhinja": ["konoba", "restoran hrvatske kuhinje", "traditional croatian food"],
             "talijanska kuhinja": ["trattoria", "italian restaurant"], "azijska kuhinja": ["sushi", "asian restaurant"]}
    recipe = funnel.normalize_recipe({"group_by": "Kuhinja", "group_words": words}, funnel.from_contract([{
        "id": "a", "text": "x", "check": {"type": "table", "path": "r.csv", "columns": ["Naziv", "Kuhinja"],
                                          "min_per": {"Kuhinja": 2}, "allowed": {"Kuhinja": list(words)}}}]))
    w = funnel.Work({"id": "r", "config": {}}, {"step_key": "step-2"}, recipe, tmp_path, group="talijanska kuhinja")
    assert w._other_group("Hotelski restoran (hotel u centru)") == ""
    assert w._other_group("Fast food/zalogajnica") == ""
    assert w._other_group("Tradicionalna konoba s domaćom hranom.") == "hrvatska kuhinja"  # a phrase of its own
    assert w._other_group("Serves traditional Croatian food.") == "hrvatska kuhinja"
    assert w._is_group_rule({"rule": "Kuhinja je točno jedna od: hrvatska, talijanska, azijska"})
    assert not w._is_group_rule({"rule": "Nije hotelski restoran"})


async def test_a_drop_that_broke_another_rule_stays_a_drop(deck):
    """"Rating 4.3 is below 4.5; cuisine is Italian, not Croatian": the Italian step
    would drop it for the rating too."""
    run, folder = await _run(deck["store"], KITCHENS, {"funnel_loosen": "short"}, rid="bat_brokeoth1")
    (folder / "extra" / "hr.csv").write_text("Naziv,Adresa,Ocjena\nTrattoria Uno,Ilica 7,4.3\n")  # the rating it read
    rating = "ocjena 4,5 ili viša"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja",
                      "group_words": {"talijanska kuhinja": ["italian"], "hrvatska kuhinja": ["croatian"]},
                      "rules": [{"rule": "kuhinja je jedna od skupina", "kind": "must", "fact": "identity"},
                                {"rule": rating, "kind": "must", "fact": "measured", "columns": ["Adresa"]}],
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "decide": "Restorani."}
    deck["decide"] = lambda items: [{"id": i, "verdict": "drop", "broken": [rating], "shown": "4.3",
                                     "reason": "Rating 4.3 is below 4.5; cuisine is Italian, not Croatian."}
                                    for i, _f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert [r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)] == ["dropped"]
    assert not (folder / funnel.HANDOVER).exists()


async def test_the_place_rules_address_comes_from_its_own_pages(deck):
    """Seven restaurants came out nearly there with Adresa and Kvart empty, their
    address printed on their own pages."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Web", "Adresa", "Kvart", "Telefon"],
                                                     "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, rid="bat_address01")
    run = {**run, "task": "Pronađi restorane u Zagrebu."}
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web",
                      "sources": [{"kind": "search", "group": "", "queries": ["restorani Zagreb"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": ["", "kontakt"]}],
                      "rules": [{"rule": "Restoran se nalazi u gradu Zagrebu", "kind": "must", "fact": "place",
                                 "columns": ["Adresa", "Kvart"]}], "decide": "Restorani u Zagrebu."}
    deck["extract"] = lambda user: {"items": [{"name": "Konoba Lipa", "url": "https://konoba-lipa.example",
                                               "fields": {"Naziv": "Konoba Lipa", "Web": "https://konoba-lipa.example"}}]}
    deck["tools"] = _Tools(pages={"https://konoba-lipa.example": "Konoba Lipa\nNajbolja domaća kuhinja",
                                  "https://konoba-lipa.example/kontakt": "Kontakt\nLipova 10, 10000 Zagreb\n"
                                                                         "Tel. +385 1 234 5678\nRadno vrijeme 12-23"},
                           results={"restorani Zagreb": "list results"})
    # it named the district but left the address (and phone) empty
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {**f, "Kvart": "Donji grad"}} for i, f in items]
    steps = await funnel.plan(run)
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": run["task"]}, steps[0])
    rows = funnel._read_csv(folder / funnel.ROWS)
    assert rows and rows[0]["Adresa"] == "Lipova 10, 10000 Zagreb" and rows[0]["Kvart"] == "Donji grad"
    assert fx.in_text(rows[0]["Telefon"], "+385 1 234 5678")  # "Telefon" is a phone column too


async def test_what_came_close_is_a_result_of_its_own(deck):
    """Items that nearly made the list sat in the run's working files only: they are a
    result of their own, next to the list, in its columns as far as they were found."""
    run, folder = await _run(deck["store"], KITCHENS, rid="bat_closeby01")
    (folder / "extra" / "hr.csv").write_text("Naziv,Adresa\nKonoba A,Ilica 1\nBistro M,\nKonoba N,Ilica 9\n")
    (folder / "extra" / "it.csv").write_text("Naziv,Adresa\nOsteria E,Ilica 5\nPizzeria F,Ilica 6\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja",
                      "group_words": {"talijanska kuhinja": ["italian", "talijanski"],
                                      "hrvatska kuhinja": ["croatian", "hrvatski", "konoba"]},
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"},
                                  {"kind": "file", "path": "extra/it.csv", "group": "talijanska kuhinja"}],
                      "decide": "Restorani u Zagrebu."}

    def decide(items):
        out = []
        for i, f in items:
            n = f["Naziv"]
            if n == "Bistro M":  # nearly there: its group named loosely, no address found
                out.append({"id": i, "verdict": "maybe", "reason": "Adresa nije pronađena.",
                            "row": {"Naziv": n, "Kuhinja": "Croatian"}})
            elif n == "Pizzeria F":
                out.append({"id": i, "verdict": "maybe", "reason": "Ne zna se radi li za ručak.",
                            "row": {**f, "Kuhinja": "talijanska kuhinja"}})
            elif n == "Konoba N":
                out.append({"id": i, "verdict": "drop", "reason": "Zatvoreno."})
            else:
                out.append({"id": i, "verdict": "keep", "row": {**f, "Kuhinja": "hrvatska kuhinja" if "Konoba" in n
                                                                  else "talijanska kuhinja"}})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    merged = (await _go(deck["store"], run, steps))[-1]
    name = "restorani-came-close.csv"
    assert funnel.came_close_name(funnel.load_recipe(folder)) == name
    close = funnel._read_csv(folder / name)
    assert list(close[0]) == ["Naziv", "Kuhinja", "Adresa", "Why it came close", "Sources"]
    assert [(r["Naziv"], r["Kuhinja"], r["Adresa"]) for r in close] == [
        ("Bistro M", "hrvatska kuhinja", ""), ("Pizzeria F", "talijanska kuhinja", "Ilica 6")]
    assert close[0]["Why it came close"] == "Adresa nije pronađena."  # Konoba N, dropped, isn't there
    assert f"`{name}`: 2 that came close, with why." in merged["output"]
    assert f"- `{name}`: the 2 that came close" in (folder / funnel.SUMMARY).read_text()
    assert "isn't part of the list" in funnel.judge_note(await deck["store"].get_run(run["id"]))
    # a result of its own in the run's results, not one of the other files
    from captain_claw.flight_deck import deliverable_profile as dp
    found = [(2.0, "restorani.csv", folder / "restorani.csv", 1), (1.0, name, folder / name, 1)]
    assert {k: [f[1] for f in v] for k, v in dp.sort_files(found, {"restorani.csv"}).items() if v} == {
        "main": ["restorani.csv"], "close": [name]}
    # a row on a looser reading the list didn't need comes first in its group; one the
    # finishing step added later, or dropped, isn't there
    recipe = funnel.load_recipe(folder)
    funnel._append_csv(folder / funnel.LOOSE, recipe["columns"] + [funnel._GROUP_COL, funnel._LOOSE_COL],
                       {"Naziv": "Konoba L", "Kuhinja": "hrvatska kuhinja", "Adresa": "Ilica 8",
                        funnel._GROUP_COL: "hrvatska kuhinja", funnel._LOOSE_COL: "ocjena 4.5 ili više"})
    funnel._append_csv(folder / funnel.ROWS, recipe["columns"] + [funnel._GROUP_COL],
                       {"Naziv": "Pizzeria F", "Kuhinja": "talijanska kuhinja", "Adresa": "Ilica 6",
                        funnel._GROUP_COL: "talijanska kuhinja"})
    got = funnel.write_came_close(folder, recipe)
    assert [(r["Naziv"], r["Why it came close"]) for r in got] == [
        ("Konoba L", "meets these rules only by their looser reading: ocjena 4.5 ili više"),
        ("Bistro M", "Adresa nije pronađena.")]
    # nothing that came close: no file
    (folder / funnel.LOOSE).unlink()
    (folder / funnel.MAYBE).unlink()
    assert funnel.write_came_close(folder, recipe) == [] and not (folder / name).exists()


def test_an_items_own_site_says_who_it_is():
    assert funnel.site_id("https://www.restac.example/") == "restac.example"
    assert funnel.site_id("restac.example/en/") == "restac.example"  # a language part aside
    assert funnel.site_id("https://restac.example/kontakt") == "restac.example/kontakt"
    assert funnel.site_id("https://instagram.com/restac/") == "instagram.com/restac"  # its page on a shared site
    assert funnel.site_id("https://www.linkedin.com/in/ana-b") == "linkedin.com/in/ana-b"
    assert funnel.site_id("https://instagram.com") == funnel.site_id("https://www.linkedin.com/company/") == ""
    assert funnel.site_id("not a link") == funnel.site_id("") == ""


def test_an_item_found_again_under_another_name_is_the_same_item():
    """One restaurant was on the list twice ("Restac" and "Restac 900"): the same
    website, address, phone and email under two names."""
    recipe = {"columns": ["Naziv", "Adresa", "Web", "Telefon", "Email"], "key": "Naziv", "site_column": "Web"}
    same = funnel.SameItems(recipe)
    same.add({"Naziv": "Restac", "Adresa": "Ilica 1", "Web": "https://restac.example", "Telefon": "+385 1 111 2222",
              "Email": "info@restac.example"})
    same.add({"Naziv": "Ana B", "Web": "https://instagram.com/ana"})
    assert same.of({"Naziv": "Restac 900", "Adresa": "Ilica 1", "Web": "https://www.restac.example/",
                    "Telefon": "01 111 2222"}) == "Restac"
    assert same.of({"Naziv": "Restac (kontakt)", "Web": "https://restac.example/kontakt",
                    "Email": "INFO@restac.example"}) == "Restac"  # another page of its site, the same email
    # another branch of the same chain: another address and phone
    assert same.of({"Naziv": "Restac Maksimir", "Adresa": "Maksimirska 5", "Web": "https://restac.example",
                    "Telefon": "+385 1 999 8888"}) == ""
    assert same.of({"Naziv": "Bob C", "Web": "https://instagram.com/bob"}) == ""  # a shared site: its own page
    # the same address written two ways isn't another branch
    same.add({"Naziv": "Kotač", "Adresa": "Ilička ulica", "Web": "https://kotac.example", "Telefon": "01 555 0101"})
    same.add({"Naziv": "Lipa", "Adresa": "Lipova 10, 10000 Zagreb", "Web": "https://lipa-zg.example"})
    assert same.of({"Naziv": "Restoran Kotač", "Adresa": "Ilička", "Web": "https://kotac.example"}) == "Kotač"
    assert same.of({"Naziv": "Restoran Lipa", "Adresa": "Ul. Lipove 10", "Web": "https://lipa-zg.example/"}) == "Lipa"
    # one row cites a guide's page, not its own site: a word of its name and the same address, or phone
    same.add({"Naziv": "ResLUK (Luk)", "Adresa": "Brežna 140", "Web": "https://guide.example/en/restaurants/luk"})
    assert same.of({"Naziv": "Restoran LUK", "Adresa": "Brežna 140, 10000 Zagreb",
                    "Web": "https://restaurant-luk.example"}) == "ResLUK (Luk)"
    assert same.of({"Naziv": "Kotač bistro", "Web": "https://guide.example/kotac", "Telefon": "+385 1 555 0101"}) == "Kotač"
    assert same.of({"Naziv": "Restoran Mak", "Adresa": "Savska ulica 140", "Web": "https://mak.example"}) == ""
    assert same.of({"Naziv": "Kotač Novi", "Adresa": "Savska 1", "Web": "https://kotac.example"}) == ""  # another street
    assert same.of({"Naziv": "Nowhere", "Web": ""}) == ""
    assert funnel.SameItems({"columns": ["Naziv"], "key": "Naziv"}).of({"Naziv": "x"}) == ""  # no site column


PLACES_WEB = [{"id": "fm1", "text": "mjesta.csv; 3 mjesta", "check": {
    "type": "table", "path": "mjesta.csv", "columns": ["Naziv", "Adresa", "Web", "Telefon"], "min_rows": 3}}]


async def test_the_same_item_under_another_name_isnt_kept_twice(deck):
    run, folder = await _run(deck["store"], PLACES_WEB, rid="bat_sameweb1")
    (folder / "extra" / "m.csv").write_text(
        "Naziv,Adresa,Web,Telefon\nKonoba A,Ilica 1,https://konoba-a.example,+385 1 111 2222\n"
        "Konoba A 900,Ilica 1,https://www.konoba-a.example/,01 111 2222\n"
        "Konoba A Maksimir,Maksimirska 5,https://konoba-a.example,+385 1 999 8888\n")
    deck["recipe"] = {"item": "mjesto", "key": "Naziv", "site_column": "Web",
                      "sources": [{"kind": "file", "path": "extra/m.csv"}], "decide": "Mjesta."}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": f} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert [r["Naziv"] for r in funnel._read_csv(folder / "mjesta.csv")] == ["Konoba A", "Konoba A Maksimir"]
    twin = next(r for r in funnel._read_csv(folder / funnel.ITEMS) if r["name"] == "Konoba A 900")
    assert (twin["status"], twin["stage"]) == ("dropped", "duplicate") and "already looked at as Konoba A" in twin["reason"]
    assert not any("Konoba A 900" in p for p in deck["decide_prompts"])  # caught before a judging call
    # rows already written twice (a run from before) go on the list once
    recipe = funnel.load_recipe(folder)
    funnel._append_csv(folder / funnel.ROWS, recipe["columns"] + [funnel._GROUP_COL],
                       {"Naziv": "Konoba A 900", "Adresa": "Ilica 1", "Web": "https://konoba-a.example"})
    assert [r["Naziv"] for r in funnel.write_list(folder, recipe)] == ["Konoba A", "Konoba A Maksimir"]


def test_a_named_source_and_a_phone_written_another_way_are_in_evidence():
    """A review count was checked against "Google" (the source column held a name, not a
    link), and a phone written "+385 1 ..." wasn't found where the page wrote "01 ..."."""
    recipe = {"checks": [{"column": "Recenzije", "op": "in_evidence", "source_column": "Izvor"},
                         {"column": "Telefon", "op": "in_evidence", "source_column": "Web"}]}
    read = {"https://lipa.example": "Rezervacije 01 555 0202", "https://guide.example/lipa": "1.234 recenzija"}
    row = {"Recenzije": "1.234", "Izvor": "Google", "Telefon": "+385 1 555 0202", "Web": "https://lipa.example"}
    assert funnel.check_row(recipe, row, read) == ""
    assert "isn't in what was read" in funnel.check_row(recipe, {**row, "Recenzije": "999"}, read)
    assert "isn't in https://lipa.example" in funnel.check_row(recipe, {**row, "Telefon": "+385 1 555 0203"}, read)
    assert not fx.in_text("2023-2024", "from 2023 to 2024")


EMPTY_CELLS = [{"id": "fm1", "text": "r.csv", "check": {"type": "table", "path": "r.csv",
                                                       "columns": ["Naziv", "Adresa", "Web", "Preporuka"], "min_rows": 4}}]


async def test_a_cell_left_empty_is_nearly_there_not_a_rule_broken(deck):
    """Twelve restaurants were dropped for "Adresa is empty", "Web is empty" or
    "Preporuka is empty": a fact not found isn't a rule broken."""
    run, folder = await _run(deck["store"], EMPTY_CELLS, rid="bat_emptycel")
    (folder / "extra" / "r.csv").write_text(
        "Naziv,Adresa,Web,Preporuka\nKonoba A,Ilica 1,https://a.example,štrukli\nKonoba B,Ilica 2,https://b.example,\n"
        "Konoba C,,https://c.example,juha\nKonoba D,Ilica 4,,sarma\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web",
                      "sources": [{"kind": "file", "path": "extra/r.csv"}], "decide": "Restorani.",
                      "rules": [{"rule": "Restoran u Zagrebu", "kind": "must", "fact": "place", "columns": ["Adresa"]},
                                {"rule": "Preporuka je jelo s jelovnika", "kind": "prefer", "fact": "other",
                                 "columns": ["Preporuka"], "looser": "prazno kad jelovnik nije objavljen"}],
                      "checks": [{"column": c, "op": "required"} for c in ("Naziv", "Adresa", "Web", "Preporuka")]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": f} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba A"][0] == "kept"
    assert got["Konoba B"] == ("loose", "meets only the looser reading of: Preporuka je jelo s jelovnika")
    assert got["Konoba C"][0] == "maybe"  # its address: the finishing step and the came-close file get it
    assert got["Konoba D"] == ("maybe", "left Web empty")
    assert "dropped" not in {st for st, _ in got.values()}
    maybe = {m["name"]: json.loads(m["row"]) for m in funnel._read_csv(folder / funnel.MAYBE)}
    assert maybe["Konoba D"] == {"Naziv": "Konoba D", "Adresa": "Ilica 4", "Preporuka": "sarma"}


def test_a_size_filter_counts_people_only():
    """"Not a chain with more than 3 locations" became a size filter of 3 (people)."""
    assert "size_max and size_min count people (staff, employees) only" in funnel.SETUP_SYSTEM
    assert "a limit on locations" in funnel.SETUP_SYSTEM


async def test_the_summary_and_the_dropped_items_go_under_the_charters_names(deck):
    """The charter asked for sazetak.md and iskljuceni.md; the run wrote summary.md and
    exclusions.md, and the reviewers said the files were missing."""
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_charnames")
    (folder / "extra" / "people.csv").write_text("Name,Email,Team\nAna,ana@x.example,a\nBo,bo@x.example,c\n")
    deck["recipe"] = {"item": "person", "key": "Email", "sources": [{"kind": "file", "path": "extra/people.csv"}],
                      "decide": "People in team a or b.",
                      "files": {"summary": "sazetak.md", "exclusions": "../iskljuceni.md"}}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": f} for i, f in items]
    steps = await funnel.plan(run)
    merged = (await _go(deck["store"], run, steps))[-1]
    recipe = funnel.load_recipe(folder)
    assert funnel.summary_name(recipe) == "sazetak.md"
    assert funnel.exclusions_name(recipe) == "exclusions.md"  # a name outside the folder isn't taken
    assert (folder / "sazetak.md").is_file() and not (folder / funnel.SUMMARY).exists()
    assert "`sazetak.md`: how it went" in merged["output"]
    bare = funnel.normalize_recipe({"files": {"summary": "people.csv", "exclusions": "notes.pdf"}},
                                   funnel.from_contract(PEOPLE))
    assert funnel.summary_name(bare) == "summary.md" and funnel.exclusions_name(bare) == "exclusions.md"


def test_the_came_close_group_is_written_as_the_list_names_it(tmp_path):
    recipe = funnel.normalize_recipe({"group_by": "Kuhinja"}, funnel.from_contract(KITCHENS))
    funnel._append_csv(tmp_path / funnel.MAYBE, funnel._MAYBE_COLS,
                       {"key": "gyo", "name": "Gyo bar", "group": "talijanska kuhinja",
                        "fields": json.dumps({"Naziv": "Gyo bar", "Kuhinja": "Sushi"}), "why": "no address"})
    got = funnel.write_came_close(tmp_path, recipe)
    assert [(r["Naziv"], r["Kuhinja"]) for r in got] == [("Gyo bar", "talijanska kuhinja")]


PHONES = [{"id": "fm1", "text": "r.csv", "check": {"type": "table", "path": "r.csv", "columns": ["Naziv", "Web", "Telefon"],
                                                  "min_rows": 3}}]


async def test_a_value_that_isnt_borne_out_is_left_out_not_the_item(deck):
    """Two restaurants were dropped because the phone the judging call gave came from an
    Instagram or Facebook page Captain can't read: the value goes, not the item. One
    its own pages print is taken instead of a wrong one."""
    run, folder = await _run(deck["store"], PHONES, rid="bat_unproven")
    (folder / "extra" / "r.csv").write_text("Naziv,Web\nKonoba A,https://konoba-a.example\n"
                                            "Konoba B,https://www.instagram.com/konoba_b/\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/r.csv"}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": [""]}],
                      "rules": [{"rule": "Ima telefon na vlastitoj stranici", "kind": "must", "fact": "contact",
                                 "columns": ["Telefon"]}],
                      "checks": [{"column": "Telefon", "op": "in_evidence", "source_column": "Web"}]}
    deck["tools"] = _Tools(pages={"https://konoba-a.example": "Konoba A\nRezervacije: 01 555 0303"}, results={})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {
        **f, "Telefon": "+385 1 555 9999" if f["Naziv"] == "Konoba A" else "+385 91 000 1111"}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba A"][0] == "kept"  # the phone its own page prints, not the one it was given
    assert fx.in_text(funnel._read_csv(folder / funnel.ROWS)[0]["Telefon"], "01 555 0303")
    assert got["Konoba B"][0] == "loose" and "couldn't confirm Telefon '+385 91 000 1111'" in got["Konoba B"][1]
    assert funnel._read_csv(folder / funnel.LOOSE)[0]["Telefon"] == ""


def test_a_place_in_an_items_own_name_isnt_where_it_is():
    """"Royal India" was dropped as "not in Croatia (its listing says India)"."""
    assert fx.foreign_place("Royal India, authentic food from India", ["HR"], name="Royal India") == ""
    assert fx.foreign_place("Royal India, authentic food from India", ["HR"]) == "India"
    assert fx.foreign_place("Konoba X, Beograd", ["HR"], name="Konoba X") == "Beograd"  # elsewhere still counts
    assert fx.foreign_on_own_pages("Royal India. India in every bite.", "royal-india.example", ["HR"],
                                   name="Royal India") == ""


def test_what_one_row_is_reads_as_a_noun():
    """Notes read "the same a restaurant in Zagreb suitable for a bu as …"."""
    assert funnel._short_item("a restaurant in Zagreb suitable for a business lunch") == "restaurant"
    assert funnel._short_item("The independent bookshop with a cafe") == "independent bookshop"
    assert funnel._short_item("small US knowledge firm") == "small US knowledge firm"
    assert funnel._short_item("an open-source Python library for data validation and parsing") == "open-source Python library"
    assert funnel._short_item("") == "item"


async def test_a_charter_that_accepts_a_short_list_keeps_its_rules_as_written(deck):
    """"8 per group, or the summary explains the shortfall": 18 of 24 rows were on a
    looser reading, and the reviewers said the charter would rather have the shortage."""
    run, folder = await _run(deck["store"], PAPERS, rid="bat_shortok01")
    _papers(deck, folder)
    deck["recipe"] = {**deck["recipe"], "shortage_ok": True}
    steps = await funnel.plan(run)
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any("since the charter accepts a short list with the reason given" in n for n in notes)
    recipe = funnel.load_recipe(folder)
    assert recipe["shortage_ok"] and not funnel.loosen_on(await deck["store"].get_run(run["id"]), recipe)
    assert "every rule is kept as written" in (folder / "shared" / "conventions.md").read_text()
    await _go(deck["store"], run, steps)
    assert "# Rule tiers" not in deck["decide_prompts"][0]
    assert [r["Title"] for r in funnel._read_csv(folder / "papers.csv")] == ["Alpha"]  # no row on a looser reading
    assert {r["Title"] for r in funnel._read_csv(folder / "papers-came-close.csv")} >= {"Gamma", "Eps"}
    summary = (folder / funnel.SUMMARY).read_text()
    assert "## Short" in summary and "The charter accepts a short list with the reason given" in summary
    # short, as the charter allows: the review's points don't send agents to fill it
    settled = funnel.review_settled(await deck["store"].get_run(run["id"]), "- more rows wanted")
    assert settled.startswith("The list is short (the list 2 more), which the charter accepts")
    assert "## Open points from the review" in (folder / funnel.SUMMARY).read_text()
    # a charter that wants it full: still short, fix rounds go on
    assert funnel.normalize_recipe({}, funnel.from_contract(PAPERS))["shortage_ok"] is False


async def test_an_items_link_is_read_as_its_site_and_read_once(deck):
    """13 restaurants came close with no page of their own read: the listing gave their
    site as their link ("url"), the recipe reads "{Web}", so nothing was fetched. And one
    restaurant was judged three times, found again under its domain."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Web", "Telefon"], "min_rows": 3}}]
    run, folder = await _run(deck["store"], contract, rid="bat_urlassite")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web",
                      "sources": [{"kind": "search", "group": "", "queries": ["restorani"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": ["", "kontakt"]}], "decide": "Restorani."}
    deck["extract"] = lambda user: {"items": [
        {"name": "Konoba Lipa", "url": "https://konoba-lipa.example", "fields": {"Naziv": "Konoba Lipa"}},
        {"name": "konoba-lipa.example", "url": "https://www.konoba-lipa.example/", "fields": {"Naziv": "Lipa Zagreb"}}]}
    deck["tools"] = _Tools(pages={"https://konoba-lipa.example": "Konoba Lipa",
                                  "https://konoba-lipa.example/kontakt": "Rezervacije 01 555 0404"},
                           results={"restorani": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {**f}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fetched = [u for k, u in deck["tools"].calls if k == "fetch"]
    assert "https://konoba-lipa.example/kontakt" in fetched  # its own pages, though only its link was given
    rows = funnel._read_csv(folder / funnel.ROWS)
    assert [r["Naziv"] for r in rows] == ["Konoba Lipa"] and fx.in_text(rows[0]["Telefon"], "01 555 0404")
    twin = next(r for r in funnel._read_csv(folder / funnel.ITEMS) if r["key"] == "lipa zagreb")
    assert (twin["status"], twin["stage"]) == ("dropped", "duplicate") and "already looked at as Konoba Lipa" in twin["reason"]
    assert sum("Lipa" in p for p in deck["decide_prompts"]) == 1  # one judging call, not two


def test_a_filter_on_a_column_the_listing_doesnt_give_drops_nothing(tmp_path):
    """Every restaurant was dropped before reading: a setup filter "Adresa matches
    Zagreb" ran on search hits, which give no address."""
    contract = [{"id": "a", "text": "x", "check": {"type": "table", "path": "r.csv", "columns": ["Naziv", "Adresa", "Web"]}}]
    recipe = funnel.normalize_recipe({"filters": [{"column": "Adresa", "op": "regex", "value": "Zagreb"}]},
                                     funnel.from_contract(contract))
    w = funnel.Work({"id": "r", "config": {}}, {"step_key": "step-1"}, recipe, tmp_path)
    assert w.passes({"fields": {"name": "Konoba A", "url": "https://konoba-a.example"}}) == ""  # not known yet
    assert w.passes({"fields": {"name": "Konoba B", "Adresa": "Ilica 5, Zagreb"}}) == ""
    assert w.passes({"fields": {"name": "Konoba C", "Adresa": "Riva 2, Split"}}) == "Adresa doesn't match"


async def test_the_charters_own_alternative_holds_when_nothing_is_loosened(deck):
    """"Ako nema ocjene, procijeni koliko je dobar": the setup made the rating a plain
    must and left that clause out, and with a short list accepted nothing was read
    loosely, so a restaurant with no rating could never be listed."""
    assert "that alternative is part of the rule: write it into" in funnel.SETUP_SYSTEM
    assert "never drop such a clause" in funnel.SETUP_SYSTEM
    run, folder = await _run(deck["store"], PAPERS, rid="bat_charteralt")
    task = "Radovi o pretraživanju grafova. Ako nema citata, procijeni koliko je rad dobar."
    _papers(deck, folder)
    deck["recipe"] = {**deck["recipe"], "shortage_ok": True}
    steps = await funnel.plan({**run, "task": task})
    await bat_routes._bat_attempt({**(await deck["store"].get_run(run["id"])), "task": task}, steps[0])
    prompt = deck["decide_prompts"][0]
    assert "Ako nema citata, procijeni koliko je rad dobar." in prompt
    assert "These words decide" in prompt and "such as what to do when a fact isn't found" in prompt


async def test_a_strict_run_drops_only_what_is_shown_to_break_a_rule(deck):
    """With every rule kept as written, about 8 restaurants were dropped for "No rating,
    contact, or own website found": a fact not found is nearly there, not a rule broken."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Ocjena", "Kuhinja"], "min_rows": 3}}]
    run, folder = await _run(deck["store"], contract, {"funnel_loosen": "never"}, rid="bat_strictdrop")
    (folder / "extra" / "r.csv").write_text("Naziv,Ocjena,Kuhinja\nKonoba A,4.3,\nKonoba B,,\nBistro C,,\n")
    rating, kitchen = "Ocjena 4.5 ili viša", "Kuhinja je hrvatska"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "sources": [{"kind": "file", "path": "extra/r.csv"}],
                      "decide": "Hrvatski restorani s ocjenom 4.5+.",
                      "rules": [{"rule": rating, "kind": "must", "fact": "measured", "columns": ["Ocjena"]},
                                {"rule": kitchen, "kind": "must", "fact": "identity", "columns": ["Kuhinja"]}]}

    def decide(items):
        said = {"Konoba A": ([rating], "4.3", "Ocjena 4.3 je ispod 4.5."),       # shown: dropped
                "Konoba B": ([rating], "", "Nema ocjene ni kontakta."),          # nothing shown: nearly there
                "Bistro C": ([kitchen], "", "Libanonska kuhinja, ne hrvatska.")}  # what it is: dropped
        return [{"id": i, "verdict": "drop", "broken": said[f["Naziv"]][0], "shown": said[f["Naziv"]][1],
                 "reason": said[f["Naziv"]][2], "row": {"Naziv": f["Naziv"]}} for i, f in items]

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"Konoba A": "dropped", "Konoba B": "maybe", "Bistro C": "dropped"}
    assert '"missing": [<rule number>], "broken": [<rule number>]' in funnel.DECIDE_STRICT
    assert "loose" not in funnel.DECIDE_STRICT


def test_a_rule_with_its_own_alternative_is_judged_not_left_nearly_there():
    """The rule read "rating 4 or higher; if no rating, estimated quality judged good
    enough", yet 204 restaurants came close for a rating not found and none was kept on
    the judgement the charter asked for: every judging prompt says how to decide it."""
    for prompt in (funnel.DECIDE_SYSTEM, funnel.DECIDE_LOOSE, funnel.DECIDE_STRICT):
        assert "judge the item on what was read and keep or drop it on that" in prompt
        assert "that fact being missing is never the reason for maybe" in prompt
    assert '"missing": [<rule number>]' in funnel.DECIDE_STRICT and '"loosened"' in funnel.DECIDE_LOOSE  # still differ


async def test_a_stopped_run_still_writes_what_it_found(deck, monkeypatch):
    """A run stopped after 70 minutes had 22 rows only in its working files: no list, no
    summary, no dropped items."""
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_stopwrite")
    (folder / "extra" / "contacts.csv").write_text(
        "Name,Email,Team,Note\nAna,ana@acme.example,a,\nCy,cy@acme.example,a,student\nDi,di@delta.example,b,\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}],
                      "decide": "People with a work email.", "checks": [{"column": "Email", "op": "in_evidence"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep" if f["Name"] == "Ana" else "maybe", "reason": "x",
                                     "row": {"Name": f["Name"], "Email": f["Email"], "Team": f["Team"]}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])  # the work step, then the owner stops it
    assert not (folder / "people.csv").exists()
    note = await funnel.write_on_stop(await deck["store"].get_run(run["id"]), "by its owner")
    assert note.startswith("Stopped (by its owner): what it found so far is written: `people.csv` 1 rows")
    assert (folder / "people.csv").read_text().splitlines() == ["Name,Email,Team", "Ana,ana@acme.example,a"]
    assert (folder / "people-came-close.csv").is_file() and (folder / funnel.EXCLUSIONS).is_file()
    assert "The run was stopped before it finished (by its owner)" in (folder / funnel.SUMMARY).read_text()
    # what isn't a funnel run, or has nothing yet, writes nothing
    assert await funnel.write_on_stop({"id": "x", "owner_id": "u1", "config": {}}, "by its owner") == ""
    # Bat's driver does it when a run stops, not when it only loses its lease
    from captain_claw.flight_deck.bat_loop import BatDriver
    calls = []

    async def fake(r, why):
        calls.append(why)
        return "written"

    monkeypatch.setattr(funnel, "write_on_stop", fake)
    driver = BatDriver(deck["store"])
    await driver._stop(run["id"], "lost_lease", 1)
    await driver._stop(run["id"], "cancelled", 1)
    assert calls == ["by its owner"]
    assert any(e["message"] == "written" for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note")


async def test_a_rules_own_alternative_is_applied_by_code(deck):
    """Even with the alternative in the rule's words and the judging prompt told how to
    use it, the judging call (Sonnet, Haiku) still answered maybe for "no rating read":
    code applies the charter's own alternative, as it applies a looser reading."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Ocjena", "Telefon"], "min_rows": 3}}]
    run, folder = await _run(deck["store"], contract, rid="bat_rulealt01")
    (folder / "extra" / "r.csv").write_text("Naziv,Ocjena,Telefon\nKonoba A,,01 555 0101\nKonoba B,,\nKonoba C,,01 555 0303\n")
    rating, phone = "Ocjena 4 ili viša", "Ima telefon na vlastitoj stranici"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "sources": [{"kind": "file", "path": "extra/r.csv"}],
                      "decide": "Restorani.", "shortage_ok": True,
                      "rules": [{"rule": rating, "kind": "must", "fact": "measured", "columns": ["Ocjena"],
                                 "if_not_shown": "procijenjeno da je dobar"},
                                {"rule": phone, "kind": "must", "fact": "contact", "columns": ["Telefon"]}]}

    def decide(items):
        out = []
        for i, f in items:
            row = {"Naziv": f["Naziv"], "Telefon": f["Telefon"]}
            if f["Naziv"] == "Konoba A":  # only the rating not found
                out.append({"id": i, "verdict": "maybe", "missing": [rating], "reason": "Nema ocjene.", "row": row})
            elif f["Naziv"] == "Konoba B":  # the phone too: still nearly there
                out.append({"id": i, "verdict": "maybe", "missing": [rating, phone], "reason": "Nema ni telefona.",
                            "row": row})
            else:  # a drop that shows nothing
                out.append({"id": i, "verdict": "drop", "broken": [rating], "shown": "", "reason": "Nema ocjene.",
                            "row": row})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    recipe = funnel.load_recipe(folder)
    assert recipe["rules"][0]["if_not_shown"] == "procijenjeno da je dobar"
    assert "if its fact isn't found: procijenjeno da je dobar" in (folder / "shared" / "conventions.md").read_text()
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba A"][0] == "kept" and "met by the charter's own words for a fact not found" in got["Konoba A"][1]
    assert got["Konoba B"][0] == "maybe"
    assert got["Konoba C"][0] == "kept"  # not shown is not broken, and the charter says how to take it
    assert "procijenjeno da je dobar" in got["Konoba C"][1]


CUISINES = [{"id": "fm1", "text": "r.csv; po 2 u skupini", "check": {
    "type": "table", "path": "r.csv", "columns": ["Naziv", "Kuhinja", "Telefon"], "min_per": {"Kuhinja": 2},
    "allowed": {"Kuhinja": ["hrvatska kuhinja", "talijanska kuhinja", "azijska kuhinja"]}}}]


async def test_an_items_group_is_typed_and_jev_decides_it(deck, monkeypatch):
    """A Japanese restaurant was dropped in the Croatian step: its reason named all three
    groups, so code read none. an Italian one's said "Italian-leaning", which no group's words
    matched; a fusion bar's also named a phone not found. The group is now a typed answer, and
    Jev (when connected) decides it."""
    from captain_claw.flight_deck import jev_routes
    asked = []

    async def no_jev(owner, state, questions, model=None, *, source, agent=""):
        asked.append(state["item"])
        raise jev_routes.JevNotConnected()

    monkeypatch.setattr(jev_routes, "ask_for_owner", no_jev)
    run, folder = await _run(deck["store"], CUISINES, {"funnel_loosen": "never"}, rid="bat_typedgrp")
    (folder / "extra" / "hr.csv").write_text("Naziv,Telefon\nSora,01 555 0101\nOsteria Uno,01 555 0202\nBao Bar,\n"
                                             "Hotelski,01 555 0303\nOsteria Due,01 555 0404\n")
    kitchen, phone, hotel, rating = ("Kuhinja je jedna od tri", "Ima telefon na vlastitoj stranici",
                                     "Nije hotelski restoran", "Ocjena 4.5 ili viša")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "rules": [{"rule": kitchen, "kind": "must", "fact": "identity", "columns": ["Kuhinja"]},
                                {"rule": phone, "kind": "must", "fact": "contact", "columns": ["Telefon"]},
                                {"rule": hotel, "kind": "must", "fact": "identity"},
                                {"rule": rating, "kind": "must", "fact": "measured"}]}

    def decide(items):
        said = {"Sora": ([kitchen], "", "azijska kuhinja",
                          "Japanska kuhinja ne pripada nijednoj od tri (hrvatska, talijanska, azijska)."),
                "Osteria Uno": ([kitchen], "", "talijanska kuhinja", "Mediterranean, Italian-leaning; not Croatian."),
                "Bao Bar": ([kitchen, phone], "", "azijska kuhinja", "Fusion/Asian street food; no phone read."),
                "Hotelski": ([hotel], "hotel", "talijanska kuhinja", "A hotel's restaurant."),
                "Osteria Due": ([rating], "4.3", "talijanska kuhinja", "Ocjena 4.3, talijanska kuhinja.")}
        return [{"id": i, "verdict": "drop", "broken": said[f["Naziv"]][0], "shown": said[f["Naziv"]][1],
                 "group": said[f["Naziv"]][2], "reason": said[f["Naziv"]][3], "row": {"Naziv": f["Naziv"]}}
                for i, f in items]

    (folder / "extra" / "hr.csv").write_text((folder / "extra" / "hr.csv").read_text()
                                             .replace("Hotelski,01 555 0303", "Hotelski,hotel 01 555 0303")
                                             .replace("Osteria Due,01 555 0404", "Osteria Due,4.3 01 555 0404"))
    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert 'also carries "group"' in deck["decide_prompts"][0]
    got = {r["name"]: (r["status"], r["group"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Sora"] == ("handed", "azijska kuhinja")  # its typed group, not words of its reason
    assert got["Osteria Uno"] == ("handed", "talijanska kuhinja")
    assert got["Bao Bar"] == ("handed", "azijska kuhinja")  # a phone not found is no reason to keep it from its group
    assert got["Hotelski"][0] == "dropped"  # what it is breaks another rule: out everywhere
    assert got["Osteria Due"][0] == "dropped"  # a 4.3 rating read fails there too
    assert asked == ["Sora"]  # Jev asked only where a drop for its kind goes, and not again once it isn't connected

    # Jev connected and sure: its choice decides; "none" keeps the drop
    async def jev(owner, state, questions, model=None, *, source, agent=""):
        pick = {"Sora": ("talijanska kuhinja", 0.9), "Osteria Uno": ("none", 0.9), "Bao Bar": ("hrvatska kuhinja", 0.3)}
        choice, conf = pick[state["item"]]
        assert set(questions["group"]["options"]) == {"hrvatska kuhinja", "talijanska kuhinja", "azijska kuhinja", "none"}
        return {"answers": {"group": {"type": "choice", "choice": choice, "confidence": conf}}}

    monkeypatch.setattr(jev_routes, "ask_for_owner", jev)
    w = funnel.Work(await deck["store"].get_run(run["id"]), {"step_key": "step-x"}, funnel.load_recipe(folder),
                    folder, group="hrvatska kuhinja")
    v = {"broken": [kitchen], "shown": "", "group": "azijska kuhinja"}
    assert await w._drop_goes_to(v, {"fields": {"name": "Sora"}, "evidence": []}, "x") == "talijanska kuhinja"
    assert await w._drop_goes_to(v, {"fields": {"name": "Osteria Uno"}, "evidence": []}, "x") == ""
    assert await w._drop_goes_to(v, {"fields": {"name": "Bao Bar"}, "evidence": []}, "x") == "azijska kuhinja"  # unsure


async def test_a_near_misses_missing_rules_are_kept(deck):
    run, folder = await _run(deck["store"], CUISINES, {"funnel_loosen": "never"}, rid="bat_missingrl")
    (folder / "extra" / "hr.csv").write_text("Naziv,Telefon\nKonoba A,\n")
    phone = "Ima telefon na vlastitoj stranici"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "rules": [{"rule": phone, "kind": "must", "fact": "contact", "columns": ["Telefon"]}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "missing": [phone], "group": "hrvatska kuhinja",
                                     "reason": "Nema telefona.", "row": {"Naziv": f["Naziv"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert funnel._read_csv(folder / funnel.MAYBE)[0]["missing"] == phone


async def test_rules_are_named_by_number_in_any_language(deck):
    """15 of 16 near misses named no missing rule: the judging call wrote "rating 4+ on
    Google" for "ocjena 4 ili viša na Googleu…", and code found no rule holding those
    words, so the charter's own alternative never applied. Rules are now numbered."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Ocjena", "Telefon"], "min_rows": 3}}]
    run, folder = await _run(deck["store"], contract, rid="bat_numbered1")
    (folder / "extra" / "r.csv").write_text("Naziv,Ocjena,Telefon\nKonoba A,,01 555 0101\nKonoba B,,\nKonoba C,,01 555 0303\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "sources": [{"kind": "file", "path": "extra/r.csv"}],
                      "decide": "Restorani.", "shortage_ok": True,
                      "rules": [{"rule": "ocjena 4 ili viša na Googleu ili TripAdvisoru", "kind": "must",
                                 "fact": "measured", "columns": ["Ocjena"], "if_not_shown": "procijenjeno da je dobar"},
                                {"rule": "telefon za rezervacije na vlastitoj stranici", "kind": "must", "fact": "contact",
                                 "columns": ["Telefon"]}]}
    first = []

    def decide(items):
        out = []
        for i, f in items:
            row = {"Naziv": f["Naziv"], "Telefon": f["Telefon"]}
            if f["Naziv"] == "Konoba A":
                out.append({"id": i, "verdict": "maybe", "missing": [1], "reason": "No rating 4+ on Google.", "row": row})
            elif f["Naziv"] == "Konoba B":
                out.append({"id": i, "verdict": "maybe", "missing": ["R1", "rule 2"], "reason": "No rating, no phone.",
                            "row": row})
            elif not first:  # Konoba C is left out of the first answer
                first.append(1)
            else:
                out.append({"id": i, "verdict": "maybe", "missing": ["1"], "reason": "No rating.", "row": row})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    prompt = deck["decide_prompts"][0]
    assert "# Rules by number" in prompt and "1. ocjena 4 ili viša" in prompt
    assert "— if its fact isn't found: procijenjeno da je dobar" in prompt
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"Konoba A": "kept", "Konoba B": "maybe", "Konoba C": "kept"}  # C: asked again, not dropped
    assert len(deck["decide_prompts"]) == 2 and "Konoba C" in deck["decide_prompts"][1]
    assert funnel._read_csv(folder / funnel.MAYBE)[0]["missing"] == ("ocjena 4 ili viša na Googleu ili TripAdvisoru; "
                                                                      "telefon za rezervacije na vlastitoj stranici")


async def test_a_keep_on_the_charters_alternative_is_checked(deck, monkeypatch):
    """A restaurant was kept on "with no rating, judged good from what was read" though
    the judging call said its cuisine fit none of the groups; and "judged good" was
    taken on trust. Now its typed group must be one of the list's, and Jev, when
    connected, is asked whether the alternative is met."""
    from captain_claw.flight_deck import jev_routes
    run, folder = await _run(deck["store"], CUISINES, {"funnel_loosen": "never"}, rid="bat_altguard1")
    (folder / "extra" / "hr.csv").write_text("Naziv,Telefon\nKonoba A,01 555 0101\nBistro B,01 555 0202\n"
                                             "Konoba C,01 555 0303\nKonoba D,01 555 0404\n")
    rating = "Ocjena 4 ili viša"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "rules": [{"rule": rating, "kind": "must", "fact": "measured", "columns": ["Ocjena"],
                                 "if_not_shown": "procijenjeno da je dobar"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "missing": [1], "reason": "Nema ocjene.",
                                     "group": "none" if f["Naziv"] == "Bistro B" else "hrvatska kuhinja",
                                     "row": {"Naziv": f["Naziv"], "Telefon": f["Telefon"],
                                             "Kuhinja": "hrvatska kuhinja"}} for i, f in items]

    async def jev(owner, state, questions, model=None, *, source, agent=""):
        assert list(questions) == ["r1"] and "procijenjeno da je dobar" in questions["r1"]["instructions"]
        return {"answers": {"r1": {"type": "noul", "noul": {"Konoba C": 0.15, "Konoba D": 0.5}.get(state["item"], 0.9)}}}

    monkeypatch.setattr(jev_routes, "ask_for_owner", jev)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba A"][0] == "kept"  # Jev: yes
    assert got["Bistro B"][0] == "maybe" and "fits none of the list's groups" in got["Bistro B"][1]
    assert got["Konoba C"][0] == "maybe" and "Jev: not met by the charter's own alternative" in got["Konoba C"][1]
    assert got["Konoba D"][0] == "kept"  # Jev unsure: the judging call's maybe stands as "seems to fit"


async def test_a_directory_listing_is_read_alone_and_is_never_the_items_own(deck):
    """40 restaurants' "site" was a city guide's or booking site's page about them: the
    run read that site's own contact, menu and about pages for each (200 fetches), and
    their phone or email could pass for the restaurant's."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Web", "Email"], "min_rows": 4}}]
    run, folder = await _run(deck["store"], contract, rid="bat_dirsites1")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web", "decide": "Restorani.",
                      "sources": [{"kind": "search", "group": "", "queries": ["restorani"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": ["", "kontakt"]}]}
    deck["extract"] = lambda user: {"items": [
        {"name": "Konoba A", "url": "https://guide.example/restoran/grad/konoba-a/", "fields": {"Naziv": "Konoba A"}},
        {"name": "Bistro B", "url": "https://book.example/bistro-b", "fields": {"Naziv": "Bistro B"}},
        {"name": "Bistro C", "url": "https://book.example/bistro-c", "fields": {"Naziv": "Bistro C"}},
        {"name": "Konoba D", "url": "https://konoba-d.example", "fields": {"Naziv": "Konoba D"}}]}
    deck["tools"] = _Tools(pages={
        "https://guide.example/restoran/grad/konoba-a/": "Konoba A, Ilica 1. Pišite: info@guide.example",
        "https://guide.example/kontakt": "Kontakt vodiča: info@guide.example",
        "https://book.example/bistro-b": "Bistro B", "https://book.example/bistro-c": "Bistro C",
        "https://book.example/kontakt": "info@book.example",
        "https://konoba-d.example": "Konoba D", "https://konoba-d.example/kontakt": "Pišite na stol@konoba-d.example"},
        results={"restorani": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {k: v for k, v in f.items() if k == "Naziv"}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fetched = {u for k, u in deck["tools"].calls if k == "fetch"}
    assert "https://guide.example/restoran/grad/konoba-a/" in fetched  # a page deep inside a host: that page alone
    assert "https://guide.example/kontakt" not in fetched
    assert "https://book.example/kontakt" not in fetched  # several items' pages on one host: a booking site
    assert "https://konoba-d.example/kontakt" in fetched  # an own site: its pages
    emails = {r["Naziv"]: r["Email"] for r in funnel._read_csv(folder / funnel.ROWS)}
    assert emails["Konoba A"] == "" and emails["Konoba D"] == "stol@konoba-d.example"  # the guide's isn't its own


FILLED = [{"id": "fm1", "text": "svaki red ima kvart i ocjenu", "severity": "major", "check": {
    "type": "table", "path": "r.csv", "columns": ["Naziv", "Kvart", "Ocjena"], "non_empty": ["Naziv", "Kvart", "Ocjena"]}}]


async def test_the_done_checks_and_the_charters_alternative_agree(deck):
    """A done-check wanted Kvart and Ocjena filled in every row: the funnel didn't ask
    for Kvart, so rows lacking it failed later; and the rows the charter's own
    alternative kept (no rating, judged good) failed for an empty Ocjena. Agents were
    sent to fix it, and the run hit its token cap."""
    rule = {"rule": "ocjena 4 ili viša", "kind": "must", "fact": "measured", "columns": ["Ocjena"],
            "if_not_shown": "procijenjeno da je dobar"}
    r = funnel.normalize_recipe({"rules": [rule], "checks": [{"column": "Ocjena", "op": "required"}]},
                                funnel.from_contract(FILLED))
    req = {c["column"] for c in r["checks"] if c["op"] == "required"}
    assert "Kvart" in req and "Ocjena" not in req  # the done-check's column, but not one the alternative frees
    run, folder = await _run(deck["store"], FILLED, rid="bat_donechk1")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "decide": "Restorani.", "rules": [rule],
                      "sources": [{"kind": "file", "path": "extra/r.csv"}]}
    (folder / "extra" / "r.csv").write_text("Naziv,Kvart,Ocjena\nKonoba A,Centar,\n")
    await funnel.plan(run)
    (folder / "r.csv").write_text("Naziv,Kvart,Ocjena\nKonoba A,Centar,\n")  # kept on "judged good": no rating
    fresh = await deck["store"].get_run(run["id"])
    checks, _, failed = await bat_routes._contract_checks(fresh, False)
    assert not failed and all(c.passed for c in checks)
    (folder / "r.csv").write_text("Naziv,Kvart,Ocjena\nKonoba A,,\n")  # an empty Kvart still fails it
    _, _, failed = await bat_routes._contract_checks(fresh, False)
    assert failed and "Kvart is empty in 1 rows" in failed[0].get("note", "")
    # any other run's done-checks are as they were
    assert funnel.contract_for_checks({"config": {}}, FILLED) == FILLED


async def test_a_pages_footer_is_read(deck):
    """A restaurant came close for "no phone or e-mail on its own site, opening not
    confirmed": its home page has 15,000 characters and its phone, hours and rating are
    in the footer, past the 6,000 Captain kept of a page and the first 1,500 a judging
    call read of it."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Web", "Telefon"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, rid="bat_footer01")
    footer = ("Kontakt Adresa Ilička ulica 76, Zagreb Telefon 01 555 0661 Radno vrijeme Pon–Sub 11–24 "
              "★ 4.7 Google Reviews (755+ recenzija)")
    page = "Konoba Lipa. " + "Naša konoba nudi domaća jela s roštilja i iz krušne peći. " * 260 + footer
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web", "decide": "Restorani.",
                      "sources": [{"kind": "search", "group": "", "queries": ["restorani"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": [""]}]}
    deck["extract"] = lambda user: {"items": [{"name": "Konoba Lipa", "url": "https://konoba-lipa.example",
                                               "fields": {"Naziv": "Konoba Lipa"}}]}
    deck["tools"] = _Tools(pages={"https://konoba-lipa.example": page}, results={"restorani": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Naziv": f["Naziv"]}} for i, f in items]
    assert len(page) > 14_000
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    prompt = deck["decide_prompts"][0]
    assert "Telefon 01 555 0661" in prompt and "Pon–Sub 11–24" in prompt and "4.7 Google Reviews" in prompt  # the footer reaches the judging call
    assert len(prompt) < 30_000  # within the same reading budget
    # code read the whole page: the phone in its footer fills the row
    assert fx.in_text(funnel._read_csv(folder / funnel.ROWS)[0]["Telefon"], "01 555 0661")


async def test_an_item_never_reads_another_items_search_result(deck):
    """Four restaurants were judged on another's facts ("what was read is about Konoba
    Lav"): their links were a guide's pages, a guide's list page on that host was
    taken as their own result, and with none found the whole results page was theirs."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv", "columns": ["Naziv"],
                                                     "min_rows": 3}}]
    run, folder = await _run(deck["store"], contract, rid="bat_borrowed1")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "decide": "Restorani.",
                      "sources": [{"kind": "search", "group": "", "queries": ["konobe"]}]}
    results = ("[QUERY: konobe]\n\n"
               "1. Top 10 konoba in the city - Guide\n   URL: https://guide.example/konobe\n"
               "   Snippet: Konoba Lav: 4.6, open daily 12-23.\n"
               "2. Best places to eat\n   URL: https://other.example/best\n"
               "   Snippet: Konoba Ruža serves fish, 4.5 on its reviews.\n")
    deck["extract"] = lambda user: {"items": [
        {"name": "Konoba Ruža", "url": "https://guide.example/Konoba-Ruza", "fields": {"Naziv": "Konoba Ruža"}},
        {"name": "Konoba Mir", "url": "https://guide.example/Konoba-Mir", "fields": {"Naziv": "Konoba Mir"}}]}
    deck["tools"] = _Tools(pages={}, results={"konobe": results})
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "row": {"Naziv": f["Naziv"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    prompt = "\n".join(deck["decide_prompts"])
    parts = {p.split("\n", 1)[0]: p for p in prompt.split("## id ")[1:]}
    ruza = next(p for p in parts.values() if "Konoba Ruža" in p)
    mir = next(p for p in parts.values() if "Konoba Mir" in p and "Ruža" not in p)
    assert "serves fish" in ruza and "Lav" not in ruza  # its own result, found by its name in a snippet
    assert "Lav" not in mir and "serves fish" not in mir  # none of its own: nothing borrowed


async def test_an_items_pages_are_found_by_its_own_links_not_guessed(deck):
    """Every site had /kontakt, /contact, /jelovnik, /menu and /about fetched, about 400
    guesses in a run, most of them missing; the real pages were at other addresses."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Naziv", "Web", "Telefon"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, rid="bat_sitelinks")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "site_column": "Web", "decide": "Restorani.",
                      "sources": [{"kind": "search", "group": "", "queries": ["restorani"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}",
                                  "paths": ["", "/kontakt", "/contact", "/jelovnik", "/menu", "/about"]}]}
    deck["extract"] = lambda user: {"items": [
        {"name": "Konoba X", "url": "https://konoba-x.example", "fields": {"Naziv": "Konoba X"}},
        {"name": "Bistro Y", "url": "https://bistro-y.example", "fields": {"Naziv": "Bistro Y"}}]}
    home = ("Konoba X. Početna (https://konoba-x.example/) Jelovnik (https://konoba-x.example/nasa-ponuda/jelovnik) "
            "Kontakt (https://konoba-x.example/hr/kontakt-nas) Facebook (https://facebook.com/konobax) "
            "Rezervacije (https://konoba-x.example/#rezervacije) Galerija (https://konoba-x.example/galerija)")
    deck["tools"] = _Tools(pages={"https://konoba-x.example": home,
                                  "https://konoba-x.example/hr/kontakt-nas": "Telefon 01 555 0808",
                                  "https://konoba-x.example/nasa-ponuda/jelovnik": "Štrukli",
                                  "https://bistro-y.example": "Bistro Y, a page read as plain text",
                                  "https://bistro-y.example/kontakt": "Telefon 01 555 0909"},
                           results={"restorani": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Naziv": f["Naziv"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    fetched = [u for k, u in deck["tools"].calls if k == "fetch"]
    x = [u for u in fetched if "konoba-x" in u]
    assert x == ["https://konoba-x.example", "https://konoba-x.example/nasa-ponuda/jelovnik",
                 "https://konoba-x.example/hr/kontakt-nas"]  # its own links the recipe asks for, nothing guessed
    assert not any("facebook" in u or "galerija" in u for u in fetched)
    assert "https://bistro-y.example/kontakt" in fetched  # a page with no links: the recipe's paths, as before
    phones = {r["Naziv"]: r["Telefon"] for r in funnel._read_csv(folder / funnel.ROWS)}
    assert fx.in_text(phones["Konoba X"], "01 555 0808")


async def test_a_narrower_kind_of_a_group_is_that_group(deck, monkeypatch):
    """A restaurant was dropped in the Croatian step as "Dalmatian, not one of the three
    kitchens": a region's own style of a country's kind is that kind."""
    from captain_claw.flight_deck import jev_routes
    run, folder = await _run(deck["store"], CUISINES, {"funnel_loosen": "never"}, rid="bat_narrower1")
    (folder / "extra" / "hr.csv").write_text("Naziv,Telefon\nKonoba D,01 555 0101\nKonoba E,01 555 0202\n"
                                             "Tvornica F,01 555 0303\n")
    kitchen = "Kuhinja je jedna od tri"
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "rules": [{"rule": kitchen, "kind": "must", "fact": "identity", "columns": ["Kuhinja"]}]}
    said = {"Konoba D": ("none", "Dalmatinska kuhinja, nije jedna od tri."),
            "Konoba E": ("hrvatska kuhinja", "Kuhinja nije točno navedena."),
            "Tvornica F": ("none", "Nije restoran.")}
    deck["decide"] = lambda items: [{"id": i, "verdict": "drop", "broken": [1], "shown": "", "group": said[f["Naziv"]][0],
                                     "reason": said[f["Naziv"]][1], "row": {"Naziv": f["Naziv"]}} for i, f in items]

    async def jev(owner, state, questions, model=None, *, source, agent=""):
        pick = {"Konoba D": ("hrvatska kuhinja", 0.9), "Konoba E": ("hrvatska kuhinja", 0.4), "Tvornica F": ("none", 0.9)}
        choice, conf = pick[state["item"]]
        return {"answers": {"group": {"type": "choice", "choice": choice, "confidence": conf}}}

    monkeypatch.setattr(jev_routes, "ask_for_owner", jev)
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert "A narrower or regional kind of a group is that group" in deck["decide_prompts"][0]
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba D"][0] == "maybe" and "its kind is hrvatska kuhinja after all" in got["Konoba D"][1]  # Jev
    assert got["Konoba E"][0] == "maybe"  # Jev unsure: its own typed group says this group
    assert got["Tvornica F"][0] == "dropped"  # fits none


async def test_a_value_that_follows_from_a_fact_may_be_filled_and_reasons_are_whole(deck):
    """A restaurant came close with every fact read but its district, which its street
    address gives; and its reason was cut at 200 characters ("... the missing")."""
    for prompt in (funnel.DECIDE_SYSTEM, funnel.DECIDE_LOOSE, funnel.DECIDE_STRICT):
        assert "the district a street address is in" in prompt and "nothing else is filled unless it is shown" in prompt
    run, folder = await _run(deck["store"], PEOPLE, rid="bat_longwhy01")
    (folder / "extra" / "contacts.csv").write_text("Name,Email,Team\nAna,ana@acme.example,a\n")
    deck["recipe"] = {"item": "person", "sources": [{"kind": "file", "path": "extra/contacts.csv"}], "decide": "People."}
    why = "Shown: the team, the email and the role; " + "every fact the rules need is there, " * 8 + "but the start date."
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "reason": why, "row": {"Name": f["Name"]}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert 200 < len(why) <= 400
    assert funnel._read_csv(folder / funnel.ITEMS)[0]["reason"].endswith("but the start date.")


async def test_a_recorded_run_replays_as_a_test_set(deck, monkeypatch, tmp_path):
    """A change was measured by a live run against the web, different each time: a run
    can record what it searched and read, and another replays it and scores itself."""
    from captain_claw.flight_deck import funnel_eval
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv", "columns": ["Name", "Web", "Phone"],
                                                     "min_rows": 2}}]
    web = {"web_search": {"things": "list results"},
           "web_fetch": {"https://a.example": "A. Call 01 555 0101", "https://b.example": "B. No phone here."}}
    calls = []

    async def call(self, tool, args):
        calls.append(tool)
        key = args.get("query") or args.get("url")
        got = web[tool].get(key)
        return (True, got) if got is not None else (False, "404")

    monkeypatch.setattr(funnel.Tools, "call", call)
    monkeypatch.setattr(funnel, "make_tools", funnel.tools_for)
    recipe = {"item": "thing", "key": "Name", "site_column": "Web", "decide": "Things with a phone.",
              "sources": [{"kind": "search", "group": "", "queries": ["things"]}],
              "gather": [{"kind": "fetch", "url": "{Web}", "paths": [""]}]}
    deck["recipe"] = recipe
    deck["extract"] = lambda user: {"items": [{"name": n, "url": f"https://{n.lower()}.example", "fields": {"Name": n}}
                                              for n in ("A", "B")]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep" if "0101" in json.dumps(f) or f["Name"] == "A" else "maybe",
                                     "row": {"Name": f["Name"]}} for i, f in items]
    run, folder = await _run(deck["store"], contract, {"funnel_record": True}, rid="bat_record01")
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert (folder / funnel.TAPE_OUT / "pages.jsonl").is_file() and "web_fetch" in calls
    made = funnel_eval.make_set(folder, tmp_path / "set")
    assert made == {"recorded": 3, "items": 2}
    expected = funnel._read_csv(tmp_path / "set" / "expected.csv")
    assert [(r["name"], r["verdict"]) for r in expected] == [("A", "keep"), ("B", "close")]
    assert fx.in_text(expected[0]["Phone"], "01 555 0101")

    # replayed: no web, no setup call, the same result; and it scores itself
    calls.clear()
    deck["asked"].clear()
    again, folder2 = await _run(deck["store"], contract, rid="bat_replay01")
    import shutil
    shutil.copytree(tmp_path / "set", folder2 / funnel.TAPE_IN)
    steps = await funnel.plan(again)
    notes = [e["message"] for e in await deck["store"].list_events(again["id"]) if e["stage"] == "note"]
    assert any("replayed from a recording" in n for n in notes) and "setup" not in deck["asked"]
    await _go(deck["store"], again, steps)
    assert calls == []
    report = (folder2 / funnel_eval.REPORT).read_text()
    assert "Verdicts right: 2 of 2 (100%)" in report and "Wrong keeps: 0" in report
    assert "Not on the tape: 0 of 1 searches, 0 of 2 pages" in report
    # a worse funnel shows in the score
    bad = funnel_eval.score(folder2, tmp_path / "set" / "expected.csv")
    assert bad["verdicts_right"] == 2
    (tmp_path / "strict.csv").write_text("name,verdict\nA,drop\nB,keep\nC,keep\n")
    bad = funnel_eval.score(folder2, tmp_path / "strict.csv")
    assert bad["wrong_keeps"] == ["A"] and bad["missed_keeps"] == ["B"] and bad["not_looked_at"] == ["C"]


async def test_a_nearly_there_item_is_looked_into(deck):
    """Most near misses lacked one fact a further page would show, and the judging call
    could only say "maybe". Now the item is looked into, a few steps the model picks,
    and judged again on what that found."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Name", "Web", "Phone"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_look": "on", "funnel_loosen": "never"}, rid="bat_golook01")
    phone = "a phone on its own site"
    deck["recipe"] = {"item": "thing", "key": "Name", "site_column": "Web", "decide": "Things with a phone.",
                      "sources": [{"kind": "search", "group": "", "queries": ["things"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": [""]}],
                      "rules": [{"rule": phone, "kind": "must", "fact": "contact", "columns": ["Phone"]}],
                      "checks": [{"column": "Phone", "op": "in_evidence", "source_column": "Web"}]}
    deck["extract"] = lambda user: {"items": [{"name": n, "url": f"https://{n.lower()}.example", "fields": {"Name": n}}
                                              for n in ("Alpha", "Beta")]}
    deck["tools"] = _Tools(pages={"https://alpha.example": "Alpha. Find us (https://alpha.example/find-us)",
                                  "https://alpha.example/find-us": "Call 01 555 0707",
                                  "https://beta.example": "Beta. Nothing more."},
                           results={"things": "list results"})

    def decide(items):
        out = []
        for i, f in items:
            shown = "01 555 0707" in deck["decide_prompts"][-1] and f["Name"] == "Alpha"
            out.append({"id": i, "verdict": "keep" if shown else "maybe", "missing": [] if shown else [1],
                        "reason": "" if shown else "No phone read.",
                        "row": {"Name": f["Name"], "Web": f["Web"], "Phone": "01 555 0707" if shown else ""}})
        return out

    def look(user):
        if "alpha.example/find-us" in user.split("# Already read")[1].split("#")[0]:
            return {"do": "stop"}
        if "https://alpha.example/find-us" in user:
            return {"do": "fetch", "url": "https://alpha.example/find-us"}
        return {"do": "stop"}

    deck["decide"], deck["look"] = decide, look
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"Alpha": "kept", "Beta": "maybe"}  # Alpha: found by looking; Beta: looked into, nothing there
    assert funnel._read_csv(folder / funnel.ROWS)[0]["Phone"] == "01 555 0707"
    assert "- a phone on its own site" in deck["look_prompts"][0] and "Links seen on what was read" in deck["look_prompts"][0]
    assert deck["asked"].count("look") == 3  # Alpha: fetch, then stop; Beta: stop. Each item once.
    # what each look read is kept with the item, and the summary says what the looks earned
    assert {r["name"]: r["looked"] for r in funnel._read_csv(folder / funnel.ITEMS)} == {"Alpha": "1", "Beta": "0"}
    # no finishing agent for a run that looks as it goes
    fresh = await deck["store"].get_run(run["id"])
    assert funnel.prepare_finish(fresh, steps[1])["result"]["output"].startswith("Nothing to finish: each item that came close")


async def test_a_look_reads_the_owners_files_when_the_run_has_no_web(deck):
    """The same loop for a list built from files: a fact missing from an item's row is
    looked for in the other files the run was given, and no web agent is started."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "labs.csv",
                                                     "columns": ["Lab", "Founded"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_look": "on", "funnel_loosen": "never"}, rid="bat_golook02")
    (folder / "extra" / "labs.csv").write_text("Lab\nBeta Lab\n")
    # keyed by a code the item doesn't carry, so code can't join it: the look has to find it
    (folder / "extra" / "founding.csv").write_text("Code,Lab,Year\nA1,Alpha Lab,2001\nB7,Beta Lab,2019\n")
    deck["recipe"] = {"item": "lab", "key": "Lab", "decide": "Labs with their founding year.",
                      "sources": [{"kind": "file", "path": "extra/labs.csv"}],
                      "rules": [{"rule": "its founding year is known", "kind": "must", "fact": "measured",
                                 "columns": ["Founded"]}],
                      "checks": [{"column": "Founded", "op": "in_evidence"}]}

    def decide(items):
        shown = "Year: 2019" in deck["decide_prompts"][-1]
        return [{"id": i, "verdict": "keep" if shown else "maybe", "missing": [] if shown else [1],
                 "reason": "" if shown else "No founding year.",
                 "row": {"Lab": f["Lab"], "Founded": "2019" if shown else ""}} for i, f in items]

    deck["decide"] = decide
    deck["look"] = lambda user: ({"do": "stop"} if "founding.csv#" in user.split("# Already read")[1].split("# The owner")[0]
                                 else {"do": "read", "path": "extra/founding.csv", "find": "Beta"})
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])  # deck["tools"] is None: a web agent would fail the test
    assert funnel._read_csv(folder / funnel.ROWS)[0]["Founded"] == "2019"
    assert "- extra/founding.csv" in deck["look_prompts"][0] and "search the web" not in deck["look_prompts"][0]
    assert "Year: 2019" in deck["decide_prompts"][-1] and "2001" not in deck["decide_prompts"][-1]  # the lines asked for


async def test_a_search_for_each_item_is_made_when_its_name_came_under_another_key(deck):
    """A per-item search written with the list's key column ("{Name} reviews") was never
    made for an item a search found, since that item carries its name as ``name``: the
    fact was then missing for it and had to be looked for at many times the cost."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "r.csv",
                                                     "columns": ["Name", "Web", "Score"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_loosen": "never"}, rid="bat_keyname01")
    deck["recipe"] = {"item": "thing", "key": "Name", "site_column": "Web", "decide": "Things with a score.",
                      "sources": [{"kind": "search", "group": "", "queries": ["things"]}],
                      "gather": [{"kind": "search", "query": "\"{Name}\" score", "count": 3}]}
    deck["extract"] = lambda user: {"items": [{"name": "Alpha", "url": "https://alpha.example", "fields": {}},
                                              {"name": "Beta", "url": "https://beta.example", "fields": {"Name": "Beta"}}]}
    deck["tools"] = _Tools(pages={}, results={"things": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "missing": [], "reason": "No score.",
                                     "row": {"Name": f.get("Name") or f.get("name")}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    asked = [q for kind, q in deck["tools"].calls if kind == "search"]
    assert '"Alpha" score' in asked  # its name, though no "Name" field came with it
    assert asked.count('"Beta" score') == 1  # a key column that did come is used as it is


async def test_a_kind_said_to_fit_no_group_is_asked_about_alone(deck):
    """Seven items of one group were dropped as "fits none of the groups" (a narrower kind
    of one), with no Jev to say otherwise: the judging call weighs every rule at once
    and reads "one of these groups" literally. The group is asked again, alone."""
    run, folder = await _run(deck["store"], CUISINES, {"funnel_loosen": "never"}, rid="bat_askgroup1")
    (folder / "extra" / "hr.csv").write_text("Naziv,Telefon\nKonoba D,01 555 0101\nTvornica F,01 555 0303\n"
                                             "Konoba G,01 555 0404\n")
    deck["recipe"] = {"item": "restoran", "key": "Naziv", "group_by": "Kuhinja", "decide": "Restorani.",
                      "sources": [{"kind": "file", "path": "extra/hr.csv", "group": "hrvatska kuhinja"}],
                      "rules": [{"rule": "Kuhinja je jedna od tri", "kind": "must", "fact": "identity",
                                 "columns": ["Kuhinja"]}]}
    said = {"Konoba D": "none", "Tvornica F": "none", "Konoba G": "hrvatska kuhinja"}
    deck["decide"] = lambda items: [{"id": i, "verdict": "drop", "broken": [1], "shown": "", "group": said[f["Naziv"]],
                                     "reason": "Nije jedna od tri.", "row": {"Naziv": f["Naziv"]}} for i, f in items]
    deck["group"] = lambda user: "hrvatska kuhinja" if "Konoba D" in user.split("# The item")[1] else "none"
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Konoba D"][0] == "maybe" and "its kind is hrvatska kuhinja after all" in got["Konoba D"][1]
    assert got["Tvornica F"][0] == "dropped"  # asked alone too: still none
    assert got["Konoba G"][0] == "maybe" and deck["asked"].count("group") == 2  # it named its group: not asked
    assert "- hrvatska kuhinja" in deck["group_prompts"][0] and "narrower or regional kind" in funnel.GROUP_SYSTEM


async def test_a_folder_of_many_files_is_one_pattern_and_every_file_is_seen(deck):
    """A list built from a folder of documents: the setup was shown only the first 60
    lines of the folder, had to name a source per file, and a look was offered only the
    first 30 files, so the tables a fact is looked up in (late in the alphabet) never
    came up. Now every file is listed, one pattern stands for many files alike, and a
    look is told of them all."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "labs.csv",
                                                     "columns": ["Lab", "Founded"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, rid="bat_manyfiles1")
    extra = folder / "extra"
    for n in range(70):
        (extra / f"report_{n:03d}.md").write_text(f"# Lab {n}\n\nLab {n} works on things.\n")
    (extra / "zz_founding.csv").write_text("Lab,Year\nLab 1,2001\n")
    listing = funnel._folder_listing(folder)
    assert "- extra/report_069.md" in listing and "- extra/zz_founding.csv" in listing  # past the detailed lines: by name
    recipe = funnel.tidy_recipe(funnel.normalize_recipe(
        {"item": "lab", "key": "Lab", "sources": [{"kind": "document", "path": "extra/report_*"}]},
        funnel.from_contract(contract)), "Labs.", folder)
    paths = [s["path"] for s in recipe["sources"]]
    assert len(paths) == 70 and paths[0] == "extra/report_000.md" and all(s["kind"] == "document" for s in recipe["sources"])
    assert "extra/zz_founding.csv" not in paths  # the pattern's files only
    work = funnel.Work(run, {"step_key": "step-1"}, recipe, folder, "")
    assert "extra/zz_founding.csv" in work._look_files() and len(work._look_files()) == 71
    assert 'one source with a pattern' in funnel.SETUP_SYSTEM


async def test_a_table_named_its_own_way_is_read_and_other_tables_are_joined(deck):
    """A list from the owner's files: a table whose ID column was "Strain" (the list's:
    "StrainID") was passed over without a word, a third of the items with it; and an
    email kept in a directory, a species in a results table, stayed empty, since a look
    only goes for what a rule needs. Now a table's columns are matched to the list's,
    and rows of the owner's other tables about an item join what is read of it."""
    cols = ["StrainID", "ZoneMm", "Species", "Researcher", "Email"]
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "active.csv", "columns": cols,
                                                     "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_loosen": "never"}, rid="bat_tablejoin1")
    extra = folder / "extra"
    (extra / "batch.csv").write_text("Strain,S_aureus_zone_mm,E_coli_zone_mm,Researcher\n"
                                     "NG-0101,22,0,Ana Test\nNG-0102,9,30,Ben Test\n")
    (extra / "sequencing.csv").write_text("Strain,Closest match\nNG101,Bacillus exampli\nNG-0102,Bacillus otheri\n")
    (extra / "staff.csv").write_text("Name,Email\nAna Test,ana@lab.example\nBen Test,ben@lab.example\n")
    (extra / "codes.csv").write_text("Code,Meaning\n22,twenty-two\n9,nine\n")  # a bare number is no key to join by
    deck["recipe"] = {"item": "isolate", "key": "StrainID", "decide": "Active isolates.",
                      "sources": [{"kind": "file", "path": "extra/batch.csv"}],
                      "filters": [{"column": "ZoneMm", "op": "min", "value": 15}]}
    asked = []

    def columns(user):
        asked.append(user)
        return {"S_aureus_zone_mm": "ZoneMm", "E_coli_zone_mm": "ZoneMm", "Nothing": "Species"}

    deck["columns"] = columns
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "reason": "", "row": {
        "StrainID": f["StrainID"], "ZoneMm": f["ZoneMm"], "Researcher": f["Researcher"],
        "Species": "Bacillus exampli", "Email": "ana@lab.example"}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["stage"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"NG-0101": ("kept", "kept"), "NG-0102": ("dropped", "filter")}  # read, and the zone filter found it
    assert len(asked) == 1 and "S_aureus_zone_mm" in asked[0]  # "Strain" by its name; the two zones needed the call
    prompt = deck["decide_prompts"][0]
    assert "Closest match: Bacillus exampli" in prompt  # its row in another table, the ID written another way
    assert "Email: ana@lab.example" in prompt and "ben@lab.example" not in prompt  # by the person its row names
    assert "twenty-two" not in prompt
    row = funnel._read_csv(folder / funnel.ROWS)[0]
    assert (row["Species"], row["Email"]) == ("Bacillus exampli", "ana@lab.example")


async def test_a_table_with_no_column_for_the_key_is_said_so(deck):
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "active.csv",
                                                     "columns": ["StrainID", "ZoneMm"], "min_rows": 1}}]
    run, folder = await _run(deck["store"], contract, rid="bat_tablejoin2")
    (folder / "extra" / "budget.csv").write_text("Month,Spent\nJan,100\nFeb,120\n")
    deck["recipe"] = {"item": "isolate", "key": "StrainID", "decide": "Active isolates.",
                      "sources": [{"kind": "file", "path": "extra/budget.csv"}]}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    events = [json.dumps(e, ensure_ascii=False, default=str) for e in await deck["store"].list_events(run["id"])]
    assert any("couldn't read extra/budget.csv as isolates" in m and "StrainID" in m for m in events)
    assert funnel._read_csv(folder / funnel.ITEMS) == []


async def test_a_filter_never_fires_on_a_number_the_text_doesnt_state(deck):
    """An isolate was ruled out before judging because "the plates were read by one
    person" had been pulled out as 1 replicate; and one whose assay wasn't done yet could
    only be dropped, since the limit code checks was no rule to be found missing."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "active.csv",
                                                     "columns": ["Strain", "Zone", "Plates"], "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_loosen": "never"}, rid="bat_inferred1")
    extra = folder / "extra"
    (extra / "a.md").write_text("# Strain A1\n\nZone 20 mm. The plates were read by one person.\n")
    (extra / "b.md").write_text("# Strain B2\n\nZone 20 mm on 2 plates.\n")
    (extra / "c.md").write_text("# Strain C3\n\nThe assay is still to be done. 4 plates are ready.\n")
    deck["recipe"] = {"item": "strain", "key": "Strain", "decide": "Strains with a zone.",
                      "sources": [{"kind": "document", "path": f"extra/{n}.md"} for n in "abc"],
                      "rules": [{"rule": "the culture is pure", "kind": "must", "fact": "identity", "columns": []}],
                      "filters": [{"column": "Plates", "op": "min", "value": 3}],
                      "checks": [{"column": "Plates", "op": "min", "value": 3}, {"column": "Zone", "op": "min", "value": 15}]}
    pulled = {"A1": {"Strain": "A1", "Zone": "20", "Plates": "1 (read by one person)"},
              "B2": {"Strain": "B2", "Zone": "20", "Plates": "2"}, "C3": {"Strain": "C3", "Plates": "4"}}
    deck["extract"] = lambda user: {"items": [{"name": n, "fields": f} for n, f in pulled.items()
                                              if f"Strain {n}" in user]}

    def decide(items):
        out = []
        for i, f in items:
            if f["Strain"] == "A1":  # the plates aren't counted in what was read
                out.append({"id": i, "verdict": "maybe", "missing": [2], "reason": "Plates not stated.",
                            "row": {"Strain": "A1", "Zone": "20"}})
            else:  # no zone measured: "can't be met", a drop that shows nothing
                out.append({"id": i, "verdict": "drop", "broken": [3], "shown": "", "reason": "No zone was measured.",
                            "row": {"Strain": "C3", "Plates": "4"}})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["stage"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["A1"] == ("maybe", "maybe")  # not filtered on a number the text never gave
    assert got["B2"] == ("dropped", "filter")  # "2 plates" is stated: the filter stands
    assert got["C3"] == ("maybe", "maybe")  # its zone wasn't found, which is not a zone under the limit
    prompt = deck["decide_prompts"][0]
    assert "2. Plates is at least 3" in prompt and "3. Zone is at least 15" in prompt  # the limits, as rules by number
    assert "a state that rules it out" in funnel.SETUP_SYSTEM and "never counted up or guessed" in funnel.EXTRACT_SYSTEM


def test_the_score_leaves_out_items_turned_away_from_a_full_group(tmp_path):
    from captain_claw.flight_deck import funnel_eval
    (tmp_path / "shared" / "funnel").mkdir(parents=True)
    (tmp_path / "shared" / "recipe.json").write_text(json.dumps({"columns": ["Name"], "key": "Name", "sources": []}))
    with (tmp_path / funnel.ITEMS).open("w") as fh:
        fh.write("key,name,group,status,stage,reason\na,A,g,kept,kept,\nb,B,g,dropped,full,g is already full\n")
    (tmp_path / "expected.csv").write_text("name,verdict,group\nA,keep,g\nB,keep,g\n")
    got = funnel_eval.score(tmp_path, tmp_path / "expected.csv")
    assert (got["judged"], got["verdicts_right"], got["missed_keeps"], got["turned_away"]) == (1, 1, [], ["B"])
    assert "Not judged, their group was full: 1 (B)" in funnel_eval.report(got)


def test_a_long_list_gets_the_searches_and_looks_its_length_needs(tmp_path):
    """A list of 500 was held to the 1,000 searches and 60 looks a list of 40 needs."""
    short = {"groups": {}, "target": 40, "sources": [], "rules": [], "checks": [], "columns": ["Name"], "key": "Name"}
    long = {**short, "target": 500}
    grouped = {**short, "target": 0, "groups": {"a": 200, "b": 300}}
    run = {"id": "bat_x", "config": {}}
    assert funnel.search_budget(run, short) == 1000 and funnel.search_budget(run, long) == 6000
    assert funnel.search_budget(run, grouped) == 6000
    assert funnel.search_budget({"id": "bat_x", "config": {"funnel_search_budget": 250}}, long) == 250  # the owner's own
    step = {"step_key": "step-1"}
    assert funnel.Work(run, step, {**short, "target": 20}, tmp_path).look_max == 60  # never under the usual
    assert funnel.Work(run, step, short, tmp_path).look_max == 80
    assert funnel.Work(run, step, long, tmp_path).look_max == 1000
    assert funnel.Work(run, step, grouped, tmp_path, "a").look_max == 400  # this step's group, not the whole list
    assert funnel.Work({"id": "bat_x", "config": {"funnel_look_max": 5}}, step, long, tmp_path).look_max == 5


async def test_a_group_named_for_two_columns_isnt_written_over_one_of_them(deck):
    """A list sorted into "US_company", "UK_company" and so on, with a Type column that
    may only be company or individual: the group's name was written over Type, and the
    list's own check then dropped every good row."""
    cols = ["Name", "Type", "Country"]
    recipe = {"group_by": "Type", "groups": {"US_company": 2, "UK_person": 2}, "columns": cols,
              "checks": [{"column": "Type", "op": "in", "value": ["company", "individual", "person"]}]}
    assert funnel.group_cell(recipe, "US_company", "US_company") == "company"  # the value its name carries
    assert funnel.group_cell(recipe, "US_company", "Individual") == "individual"  # what was written, when allowed
    assert funnel.group_cell(recipe, "UK_person") == "person"
    assert funnel.group_cell({**recipe, "checks": []}, "US_company", "x") == "US_company"  # no allowed list: as before
    plain = {"group_by": "Kind", "groups": {"red": 1}, "checks": [{"column": "Kind", "op": "in", "value": ["red", "blue"]}]}
    assert funnel.group_cell(plain, "red", "Reddish") == "red"  # a group that is a value: written as the list names it

    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv", "columns": cols, "min_rows": 2}}]
    run, folder = await _run(deck["store"], contract, {"funnel_loosen": "never"}, rid="bat_groupcell1")
    (folder / "extra" / "us.csv").write_text("Name,Where\nAcme,US\nBeta,UK\n")
    deck["recipe"] = {"item": "lead", "key": "Name", "group_by": "Type", "decide": "Leads.",
                      "groups": {"US_company": 2, "UK_company": 2},
                      "sources": [{"kind": "file", "path": "extra/us.csv", "group": "US_company"}],
                      "checks": [{"column": "Type", "op": "in", "value": ["company", "individual"]}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "reason": "",
                                     "group": "US_company" if f["Name"] == "Acme" else "UK_company",
                                     "row": {"Name": f["Name"], "Type": "US_company" if f["Name"] == "Acme" else "company",
                                             "Country": f["Where"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    rows = {r["Name"]: r for r in funnel._read_csv(folder / funnel.ROWS)}
    assert rows["Acme"]["Type"] == "company" and rows["Acme"][funnel._GROUP_COL] == "US_company"
    assert rows["Beta"]["Type"] == "company" and rows["Beta"][funnel._GROUP_COL] == "UK_company"  # its own group kept


# ── pages fetched with tables as Markdown rows ────────────────────────────

def test_site_links_label_stops_at_a_table_cell():
    page = ("| Alpha (https://alpha.example/) | Kontakt centar | Web (https://alpha.example/o-nama) |\n"
            "| Beta | Zagreb | Kontakt (https://alpha.example/kontakt) |")
    assert funnel.site_links(page, "https://alpha.example/", ["kontakt"]) == ["https://alpha.example/kontakt"]


def test_a_table_row_is_read_cell_by_cell():
    row = "| Gamma | Ilica 12 · 10000 Zagreb | +385 1 234 5678 |"
    assert funnel.place_line(row, "in Zagreb", ["HR"]) == "10000 Zagreb"
    assert fx.phones(row) == ["+385 1 234 5678"]


def test_place_line_leaves_out_a_label_and_a_heading_mark():
    assert funnel.place_line("- Adresa : Vrtna 20, 10000 Zagreb", "in Zagreb", ["HR"]) == \
        "Vrtna 20, 10000 Zagreb"
    assert funnel.place_line("## Ilica 12, 10000 Zagreb", "in Zagreb", ["HR"]) == "Ilica 12, 10000 Zagreb"
    # a place before the colon is part of the address, not a label
    assert funnel.place_line("Zagreb: Ilica 12", "in Zagreb", ["HR"]) == "Zagreb: Ilica 12"


# ── read first, reach by a contact page, search while short, ask again alone ──

REACH = [{"id": "fm1", "text": "leads.csv", "check": {"type": "table", "path": "leads.csv",
                                                      "columns": ["Name", "Web", "Email"], "min_rows": 9}}]


async def _run_with(store, task, contract, cfg=None, rid="bat_brief01"):
    await store.create_run(run_id=rid, owner_id="u1", title="t", task=task, status="planning",
                           config={"engine": "funnel", "contract": contract, "funnel_look": "off", **(cfg or {})})
    run = await store.get_run(rid)
    folder = bat_routes._run_folder(run)
    (folder / "extra").mkdir(parents=True, exist_ok=True)
    return run, folder


def test_the_links_a_text_names():
    text = "Use counter.example/docs and https://x.example/a. Save leads.csv, write to ana@lab.example, v1.2."
    assert funnel.text_links(text) == ["https://x.example/a", "https://counter.example/docs"]


async def test_what_the_owner_gave_is_read_before_the_run(deck):
    """The owner's documents and the pages their charter links to are read once, before
    anything is searched: the setup, the new searches and every judging call work from
    what they say, not from a link nobody opened."""
    run, folder = await _run_with(deck["store"], "Find labs that would use Counter (counter.example/about). "
                                  "Save leads.csv.", REACH)
    (folder / "extra" / "notes.md").write_text("We sell to labs with fewer than ten people.")
    deck["brief"] = "Counter counts colonies on a plate photo. It is for labs with fewer than ten people."
    deck["tools"] = _Tools(pages={"https://counter.example/about": "Counter counts colonies on a plate photo."},
                           results={"labs": "list results"})
    deck["recipe"] = {"item": "lab", "key": "Name", "decide": "Labs that count colonies.",
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["extract"] = lambda user: {"items": [{"name": "Alpha", "fields": {"Name": "Alpha"}}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    deck["queries"] = ["more labs"]
    steps = await funnel.plan(run)
    assert deck["asked"][:2] == ["brief", "setup"]
    assert "plate photo" in deck["brief_prompts"][0] and "fewer than ten" in deck["brief_prompts"][0]
    assert ("fetch", "https://counter.example/about") in deck["tools"].calls
    assert "It is for labs with fewer than ten people" in deck["setup_prompts"][0]
    assert funnel.brief_text(folder).startswith("Read before the run: extra/notes.md, https://counter.example/about.")
    await _go(deck["store"], run, steps[:1])
    assert "in brief]\nRead before the run" in deck["decide_prompts"][0] and "plate photo" in deck["decide_prompts"][0]
    assert "plate photo" in deck["query_prompts"][0]
    assert deck["asked"].count("brief") == 1 and [c for c in deck["tools"].calls if c[0] == "fetch"] == [
        ("fetch", "https://counter.example/about")]  # read once, not per step


async def test_no_address_but_a_contact_page_is_a_way_to_reach_it(deck):
    """A charter that accepts a short list reads no rule loosely, and every item with no
    published address came close for it. A contact page on the item's own site is a way
    to reach it in every run: its link stands where the address would; with no such
    page the item comes close, never dropped."""
    run, folder = await _run(deck["store"], REACH, rid="bat_reach01")
    rule = "an email published on its own site"
    deck["recipe"] = {"item": "lead", "key": "Email", "site_column": "Web", "decide": "Leads with an email.",
                      "shortage_ok": True, "sources": [{"kind": "search", "group": "", "queries": ["leads"]}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": ["", "contact"]}], "extract": ["emails"],
                      "filters": [{"column": "Email", "op": "regex", "value": ".+@.+"}],
                      "rules": [{"rule": rule, "kind": "must", "fact": "contact", "columns": ["Email"]}],
                      "checks": [{"column": "Email", "op": "required"},
                                 {"column": "Email", "op": "in_evidence", "source_column": "Web"}]}
    deck["extract"] = lambda user: {"items": [
        {"name": n, "url": f"https://{n.lower()}.example", "fields": {"Name": n, "Web": f"https://{n.lower()}.example",
                                                                       "Email": "form only" if n == "Beta" else ""}}
        for n in ("Alpha", "Beta", "Gamma")]}
    deck["tools"] = _Tools(pages={"https://alpha.example": "Alpha. Write to hello@alpha.example",
                                  "https://beta.example": "Beta. Get in touch (https://beta.example/contact)",
                                  "https://beta.example/contact": "Use the form below.",
                                  "https://gamma.example": "Gamma. Nothing more."}, results={"leads": "list results"})

    def decide(items):
        out = []
        for i, f in items:
            got = f["Name"] == "Alpha"
            out.append({"id": i, "verdict": "keep" if got else "maybe", "missing": [] if got else [1],
                        "reason": "" if got else "No email on its site.",
                        "row": {"Name": f["Name"], "Web": f["Web"], "Email": "hello@alpha.example" if got else ""}})
        return out

    deck["decide"] = decide
    steps = await funnel.plan(run)
    recipe = funnel.load_recipe(folder)
    assert recipe["rules"][0]["reach"] == ["Email"] and recipe["rules"][0]["if_not_shown"] == funnel.CONTACT_ALT
    assert not funnel.loosen_on(run, recipe)
    await _go(deck["store"], run, steps)
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Alpha"][0] == "kept" and got["Gamma"][0] == "maybe"
    assert got["Beta"][0] == "kept" and "reached by its contact page" in got["Beta"][1]  # not ruled out by its listing
    rows = {r["Name"]: r["Email"] for r in funnel._read_csv(folder / "leads.csv")}
    assert rows == {"Alpha": "hello@alpha.example", "Beta": "https://beta.example/contact"}
    assert "a contact page or form on its own site" in deck["decide_prompts"][0]
    assert funnel._read_csv(folder / funnel.MAYBE)[0]["missing"] == rule


async def test_a_short_group_keeps_searching_while_searches_find_rows(deck):
    """A step wrote new searches twice and stopped, with most of the run's searches
    unused. Now it writes more while its group is short and the rounds put rows on the
    list, and gives up after two rounds in a row that add none."""
    run, folder = await _run(deck["store"], REACH, rid="bat_rounds01")
    deck["recipe"] = {"item": "lead", "key": "Name", "decide": "Leads.",
                      "sources": [{"kind": "search", "group": "", "queries": ["q0"]}]}
    deck["tools"] = _Tools(pages={}, results={f"q{n}": f"found N{n}" for n in range(4)})
    deck["extract"] = lambda user: {"items": [{"name": m, "fields": {"Name": m}}
                                              for m in re.findall(r"found (N\d)", user)]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    asked = []

    def queries():
        asked.append(1)
        return ["q0", f"q{len(asked)}"]  # one already used, one new

    deck["queries"] = queries
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["N0", "N1", "N2", "N3"]
    state = json.loads((folder / funnel.QUERIES).read_text())
    assert state["rounds"][""] == 5 and state["used"] == ["q0", "q1", "q2", "q3", "q4", "q5"]  # two dry rounds, then stop
    assert "- q3" in deck["query_prompts"][-1]  # the searches it wrote itself are shown as used


async def test_an_empty_answer_for_new_searches_is_asked_for_once_more(deck):
    run, folder = await _run(deck["store"], REACH, rid="bat_rounds02")
    deck["recipe"] = {"item": "lead", "key": "Name", "decide": "Leads.",
                      "sources": [{"kind": "search", "group": "", "queries": ["q0"]}]}
    deck["tools"] = _Tools(pages={}, results={"q0": "found N0", "q1": "found N1"})
    deck["extract"] = lambda user: {"items": [{"name": m, "fields": {"Name": m}}
                                              for m in re.findall(r"found (N\d)", user)]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    calls = []

    def queries():
        calls.append(1)
        return [] if len(calls) == 1 else ["q1"] if len(calls) == 2 else []

    deck["queries"] = queries
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["N0", "N1"]


async def test_an_item_left_out_of_an_answer_is_asked_about_alone(deck):
    """A small model drops items from a batch's answer. Each one left out is asked about
    in a call of its own, not together with the others left out, before it is given up."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "p.csv", "columns": ["Name"]}}]
    run, folder = await _run(deck["store"], contract, rid="bat_alone01")
    (folder / "extra" / "people.csv").write_text("Name\nAna\nBo\nCy\nDi\n")
    deck["recipe"] = {"item": "person", "key": "Name", "decide": "Everyone.", "decide_batch": 4,
                      "sources": [{"kind": "file", "path": "extra/people.csv"}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items
                                    if len(items) == 1 or f["Name"] in ("Ana", "Bo")]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert sorted(r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)) == ["Ana", "Bo", "Cy", "Di"]
    assert [a for a in deck["asked"] if a.startswith("decide")] == ["decide:4", "decide:1", "decide:1"]


def test_a_headcount_is_people_who_work_there():
    assert fx.headcount("more than 100 lawyers and staff") == 100 and fx.headcount("We are a team of 12.") == 12
    assert fx.headcount("Founded in 2019. In 2026 people expect more from a firm.") is None  # a year, not a count
    assert fx.headcount("Our firm has 2,026 employees.") == 2026
    assert fx.headcount("Join 200,000 people who read it every week.") is None  # its readers
    assert fx.headcount("Trusted by 5,000 developers. 40 employees.") == 40


async def test_a_limit_for_one_group_leaves_the_others_alone(deck):
    """A size limit meant for a list's companies ruled out its people too."""
    base = funnel.from_contract(CONTRACT)
    one = {"column": "size_band", "op": "size_max", "value": 50, "groups": ["x"]}
    recipe = funnel.normalize_recipe({"item": "lead", "sources": [{"kind": "search", "queries": ["q"], "group": "x"}],
                                      "filters": [one, {"column": "size_band", "op": "size_min", "value": 2,
                                                        "groups": ["x", "y", "nope"]}]}, base)
    assert recipe["filters"] == [one, {"column": "size_band", "op": "size_min", "value": 2}]  # every group: no scope
    run, folder = await _run(deck["store"], CONTRACT, rid="bat_scope01")
    item, found = {"fields": {"Firm": "Alpha"}}, {"size_band": [("51-200", "a page")]}
    assert funnel.Work(run, {"step_key": "step-1"}, recipe, folder, group="x").passes(item, found).startswith("51-200")
    assert funnel.Work(run, {"step_key": "step-2"}, recipe, folder, group="y").passes(item, found) == ""


async def test_the_runs_search_agent_is_started_once(monkeypatch):
    """Four items read at once each made the first call, and each started the agent: all
    but one failed with "already running" and their items were lost."""
    import asyncio
    tools, started = funnel.Tools({"id": "bat_x", "owner_id": "u1"}, {"step_key": "step-1"}), []

    async def start():
        started.append(1)
        await asyncio.sleep(0.01)
        tools.port = 1

    monkeypatch.setattr(tools, "_start", start)
    await asyncio.gather(*(tools._ensure() for _ in range(4)))
    assert started == [1]


async def test_a_size_is_taken_only_from_a_page_that_is_the_items_own(deck):
    """An item whose only link was an article that mentions it took the article's
    numbers as its size. A page on a site many share counts as the item's own only
    when it names the item."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "l.csv", "columns": ["Name", "Web"]}}]
    run, folder = await _run(deck["store"], contract, rid="bat_ownpage01")
    deck["recipe"] = {"item": "lab", "key": "Name", "site_column": "Web", "decide": "Small labs.",
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}],
                      "filters": [{"column": "size_band", "op": "size_max", "value": 50}],
                      "gather": [{"kind": "fetch", "url": "{Web}", "paths": [""]}]}
    article, profile = "https://news.example/2026/ten-labs-to-watch", "https://hub.example/labs/beta-lab"
    deck["extract"] = lambda user: {"items": [{"name": "Alpha Lab", "url": article, "fields": {"Name": "Alpha Lab", "Web": article}},
                                              {"name": "Beta Lab", "url": profile, "fields": {"Name": "Beta Lab", "Web": profile}}]}
    deck["tools"] = _Tools(pages={article: "Ten labs to watch\n\nOur newsroom has 300 employees. Alpha Lab is one.",
                                  profile: "Beta Lab\n\nBeta Lab has 300 employees."}, results={"labs": "list results"})
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Web": f["Web"]}} for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Alpha Lab"][0] == "kept"  # the newsroom's size isn't its size
    assert got["Beta Lab"] == ("dropped", f"300 people, said on {profile}")


# ── the charter's own groups, searches that name them ─────────────────────

def test_a_short_group_name_is_named_as_a_whole_word():
    """A group named "US" had its searches thrown away as not naming it, and rewritten
    from its name alone: "list of US", "best US"."""
    assert funnel.names_group(["small US consultancies directory"], "US", ["United States", "American"])
    assert funnel.names_group(["business consultancies directory"], "US") is False  # "us" inside a word isn't it
    assert funnel.names_group(["anything"], "") is None
    base = funnel.from_contract([{"id": "fm1", "text": "x", "check": {"type": "table", "path": "l.csv", "columns": ["Name"]}}])
    recipe = funnel.normalize_recipe({"item": "lead", "group_by": "Name", "groups": {"US": 5},
                                      "group_words": {"US": ["United States"]}, "sources": [
        {"kind": "search", "group": "US", "queries": ["US research shops directory"], "about": "small US research shops"},
        {"kind": "search", "group": "US", "queries": ["plate readers"], "about": "US writers on lab tools: blogs"}]}, base)
    tidy = funnel.tidy_recipe(recipe, "Find 5 leads.")
    assert tidy["sources"][0]["queries"] == ["US research shops directory"]  # it names its group: left as written
    assert tidy["sources"][1]["queries"][:2] == ["list of US writers on lab tools", "best US writers on lab tools"]


def test_groups_with_amounts_the_charter_never_states():
    task = "Find 200 US firms, 150 UK firms, 75 US people and 75 UK people."
    assert funnel.groups_not_the_charters({"groups": {"US firms": 200, "UK firms": 150, "US people": 75, "UK people": 75}}, task) == ""
    assert "US 275, UK 225" in funnel.groups_not_the_charters({"groups": {"US": 275, "UK": 225}}, task)
    assert funnel.groups_not_the_charters({"groups": {"a": 0, "b": 0}}, task) == ""  # no amounts: nothing to compare


async def test_the_setup_is_asked_again_when_it_merged_the_charters_groups(deck):
    run, folder = await _run_with(deck["store"], "Find 20 US firms and 5 US people, 10 UK firms and 5 UK people. "
                                  "Save leads.csv.", [{"id": "fm1", "text": "x", "check": {
                                      "type": "table", "path": "leads.csv", "columns": ["Name", "Kind"]}}], rid="bat_groups01")
    merged = {"item": "lead", "key": "Name", "group_by": "Kind", "groups": {"US": 25, "UK": 15},
              "sources": [{"kind": "search", "group": "US", "queries": ["US firms list"]},
                          {"kind": "search", "group": "UK", "queries": ["UK firms list"]}]}
    right = {**merged, "groups": {"US firms": 20, "US people": 5, "UK firms": 10, "UK people": 5},
             "sources": [{"kind": "search", "group": g, "queries": [f"{g} list"]}
                         for g in ("US firms", "US people", "UK firms", "UK people")]}

    answers = [merged, right]  # the first answer merged, the second as the charter has it
    deck["recipe"] = lambda: answers.pop(0)
    steps = await funnel.plan(run)
    assert deck["asked"].count("setup") == 2 and "US 25, UK 15" in deck["setup_prompts"][1]
    assert funnel.load_recipe(folder)["groups"] == {"US firms": 20, "US people": 5, "UK firms": 10, "UK people": 5}
    assert [s["title"] for s in steps][:2] == ["Find and check 20 US firms", "Find and check 5 US people"]


async def test_new_searches_are_written_for_each_of_a_steps_sources_in_turn(deck):
    """A step with two sources wrote its new searches for the first only."""
    run, folder = await _run(deck["store"], REACH, rid="bat_turns01")
    deck["recipe"] = {"item": "lead", "key": "Name", "decide": "Leads.", "sources": [
        {"kind": "search", "group": "", "queries": ["firms"], "about": "small firms"},
        {"kind": "search", "group": "", "queries": ["writers"], "about": "people who write"}]}
    deck["tools"] = _Tools(pages={}, results={})
    deck["extract"] = lambda user: {"items": []}
    n = []

    def queries():
        n.append(1)
        return [f"new {len(n)}"]

    deck["queries"] = queries
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    looked_for = [p.split("\n")[1] for p in deck["query_prompts"]]
    assert looked_for == ["leads: small firms", "leads: people who write"]  # two dry rounds, one for each source


# ── one item's answer, however it is wrapped ──────────────────────────────

def test_a_judging_answer_is_read_however_it_is_wrapped():
    v = {"id": 1, "verdict": "keep", "row": {"Name": "A"}}
    assert funnel.answers_by_id([v], 1) == {1: v}
    assert funnel.answers_by_id(v, 1) == {1: v}                      # the bare object, for one item
    assert funnel.answers_by_id({"items": [v]}, 1) == {1: v}         # an object holding the array
    assert funnel.answers_by_id({"verdict": "drop", "reason": "x"}, 1) == {1: {"verdict": "drop", "reason": "x"}}
    assert funnel.answers_by_id([{**v, "id": 7}], 1) == {1: {**v, "id": 7}}  # one asked: it is that one's
    assert funnel.answers_by_id({"verdict": "keep"}, 3) == {}        # several asked: an id is needed
    assert funnel.answers_by_id([{**v, "id": 9}], 3) == {} and funnel.answers_by_id(None, 1) == {}


async def test_one_items_bare_answer_is_not_left_unanswered(deck, monkeypatch):
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "p.csv", "columns": ["Name"]}}]
    run, folder = await _run(deck["store"], contract, rid="bat_bare01")
    (folder / "extra" / "people.csv").write_text("Name\nAna\n")
    deck["recipe"] = {"item": "person", "key": "Name", "decide": "Everyone.", "decide_batch": 1,
                      "sources": [{"kind": "file", "path": "extra/people.csv"}]}
    deck["decide"] = lambda items: {"verdict": "keep", "row": {"Name": items[0][1]["Name"]}}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    assert [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["Ana"]
    assert [a for a in deck["asked"] if a.startswith("decide")] == ["decide:1"]


def test_a_group_named_in_the_plural_carries_its_cell():
    recipe = {"group_by": "Type", "checks": [{"column": "Type", "op": "in", "value": ["company", "individual"]}]}
    assert funnel.group_cell(recipe, "US companies") == "company"
    assert funnel.group_cell(recipe, "UK individuals") == "individual"
    assert funnel.group_cell(recipe, "US_company") == "company" and funnel.group_cell(recipe, "others", "x") == "x"


# ── a column that joins the item to what the owner offers ──────────────────

async def test_a_why_it_fits_column_is_written_from_the_owners_material(deck):
    """A reason column described the item and never the owner's product: the judging call
    was told nothing about where it comes from, and a rule on its form took it out of
    the owner's columns."""
    run, folder = await _run_with(deck["store"], "Find labs that would use Counter (counter.example). Save leads.csv.",
                                  [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv",
                                                                         "columns": ["Name", "Reason"]}}], rid="bat_fit01")
    deck["brief"] = "Counter counts colonies on a plate photo."
    deck["tools"] = _Tools(pages={"https://counter.example": "Counter counts colonies."}, results={"labs": "r"})
    deck["recipe"] = {"item": "lab", "key": "Name", "decide": "Labs.",
                      "columns": [{"name": "Name"}, {"name": "Reason", "what": "why it would use Counter", "from": "owner"}],
                      "rules": [{"rule": "Reason is one specific sentence", "kind": "must", "columns": ["Reason"]}],
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["extract"] = lambda user: {"items": [{"name": "Alpha", "fields": {"Name": "Alpha"}}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Reason": ""}} for i, f in items]
    deck["owner"] = lambda user: {"1": {"Reason": "Alpha plates by hand; Counter counts colonies from a photo."}}
    steps = await funnel.plan(run)
    recipe = funnel.load_recipe(folder)
    assert funnel.judgement_columns(recipe) == ["Reason"]  # the setup marked it: a rule on its form doesn't undo that
    await _go(deck["store"], run, steps[:1])
    line = next(ln for ln in deck["decide_prompts"][0].splitlines() if ln.startswith("- Reason"))
    assert "why it would use Counter" in line and "names what of the owner's fits" in line
    assert not next(ln for ln in deck["decide_prompts"][0].splitlines() if ln.startswith("- Name")).endswith("any item.")
    assert funnel._read_csv(folder / funnel.ROWS)[0]["Reason"].endswith("Counter counts colonies from a photo.")


async def test_a_contact_page_counts_on_a_site_learnt_while_reading(deck):
    """An item found on a directory has no site in its listing; its site is learnt while
    it is read and named in its row. Its contact page there was read, yet the item came
    close for having no address."""
    run, folder = await _run(deck["store"], REACH, rid="bat_reach02")
    rule = "an email published on its own site"
    deck["recipe"] = {"item": "lead", "key": "Email", "site_column": "Web", "decide": "Leads.", "shortage_ok": True,
                      "sources": [{"kind": "search", "group": "", "queries": ["leads"]}],
                      "gather": [{"kind": "search", "query": "\"{Name}\" contact", "count": 5}],
                      "rules": [{"rule": rule, "kind": "must", "fact": "contact", "columns": ["Email"]}],
                      "checks": [{"column": "Email", "op": "required"}]}
    deck["extract"] = lambda user: {"items": [{"name": n, "fields": {"Name": n}} for n in ("Alpha", "Beta")]}
    deck["tools"] = _Tools(pages={}, results={
        "leads": "list results",
        '"Alpha" contact': _result("Alpha - Contact us", "https://alpha.example/contact", "Use the form to write to Alpha."),
        '"Beta" contact': _result("Beta on a directory", "https://directory.example/firms/beta", "Beta, a firm.")})
    deck["decide"] = lambda items: [{"id": i, "verdict": "maybe", "missing": [1], "reason": "No email; a form only.",
                                     "row": {"Name": f["name"], "Email": "",
                                             "Web": "https://alpha.example" if f["name"] == "Alpha" else "https://beta.example"}}
                                    for i, f in items]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps[:1])
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"Alpha": "kept", "Beta": "maybe"}  # Beta's site is only named: no page of it was read
    assert funnel._read_csv(folder / funnel.ROWS)[0]["Email"] == "https://alpha.example/contact"


# ── an expert agent beside the run (docs/funnel-expert-agent-plan.md, phase 1) ──
# A factory's runs get the owner's pick. A run wakes it, gives it one turn in a session
# of its own before the recipe is written, and reads what it said everywhere the brief
# is read. It writes the new searches. One that can't be reached never stops a run.

from captain_claw.flight_deck import (  # noqa: E402
    agent_idle,
    agent_preset,
    fd_scheduler,
    first_mate_agent,
    first_mate_runs,
    funnel_expert,  # noqa: E402
)
from captain_claw.flight_deck import first_mate_store as fms  # noqa: E402

EXPERT_LIST = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv", "columns": ["Name"], "min_rows": 9}}]
SAID = ("NOTES\nSmall labs are listed by their university's core-facility pages, not in directories.\n\n"
        "DECISIONS\n- Does a two-person lab count? Yes, any lab with a bench.\n- Hospital labs? No.")


@pytest.fixture
def expert(monkeypatch):
    """One agent of the owner's, "Dr. Petri", that answers what the test says."""
    state = {"woken": 0, "turns": [], "asks": [], "released": 0, "ready": True, "says": SAID,
             "queries": lambda: ["core facility directory"]}

    async def ready(slug, timeout=0):
        state["woken"] += 1
        return state["ready"]

    async def capture(*, host, port, auth, prompt, timeout, automation=None, meta=None, identity=None, **_k):
        state["turns"].append({"prompt": prompt, "automation": automation})
        if meta is not None:
            meta["task_session"] = {"id": "sess-1", "name": "x"}
        says = state["says"]
        return (says.pop(0) if says else "") if isinstance(says, list) else says

    class Reply:
        def __init__(self, text):
            self.text, self.tokens = text, 7

    async def complete(reader, system, user, owner, max_tokens, **_k):
        state["asks"].append({"reader": reader, "system": system, "user": user})
        if "You write your columns" in system:
            return Reply(json.dumps(state.get("columns", lambda u: {})(user)))
        if "An item came close to a list" in system:
            return Reply(json.dumps(state.get("second", lambda u: {"verdict": "stay"})(user)))
        return Reply(json.dumps({"queries": state["queries"]()}))

    async def instructions(owner, slug):
        return "Answer as a microbiologist."

    from captain_claw.flight_deck import agent_exam, agent_skills
    from captain_claw.flight_deck import server as srv
    monkeypatch.setattr(first_mate_agent, "mine", lambda owner: [{"slug": "petri", "name": "Dr. Petri"}])
    monkeypatch.setattr(agent_idle, "ready_for_task", ready)
    monkeypatch.setattr(agent_idle, "touch", lambda slug: None)
    monkeypatch.setattr(agent_idle, "release_after_task_soon", lambda slug: state.__setitem__("released", state["released"] + 1))
    monkeypatch.setattr(fd_scheduler, "resolve_agent_by_slug", lambda slug, auth: ("localhost", 9, "t"))
    monkeypatch.setattr(fd_scheduler, "run_prompt_and_capture", capture)
    monkeypatch.setattr(agent_preset, "_agent_complete", complete)
    monkeypatch.setattr(agent_preset, "agent_reader", lambda slug, entry: {"provider": "x", "model": "m"})
    monkeypatch.setattr(agent_skills, "_instructions", instructions)
    monkeypatch.setattr(agent_exam, "_lesson_texts", lambda slug: ["Plates are counted after 48 hours."])
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {"petri": {"name": "Dr. Petri"}})
    return state


def _list(deck):
    deck["recipe"] = {"item": "lab", "key": "Name", "decide": "Labs.",
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["tools"] = _Tools(pages={}, results={"labs": "found N0", "core facility directory": "found N1"})
    deck["extract"] = lambda user: {"items": [{"name": m, "fields": {"Name": m}} for m in ("N0", "N1") if f"found {m}" in user]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    deck["queries"] = ["never asked"]


def test_a_factorys_runs_get_its_expert():
    cfg = fms.check_runner_config("funnel", {"expert": "petri", "search_budget": 50})
    assert cfg["expert"] == "petri" and first_mate_runs.funnel_config({"runner_config": cfg})["funnel_expert"] == "petri"
    assert "funnel_expert" not in first_mate_runs.funnel_config({"runner_config": fms.check_runner_config("funnel", {})})
    with pytest.raises(fms.FirstMateError):
        fms.check_runner_config("bat", {"expert": "petri"})


async def test_the_expert_works_beside_the_run(deck, expert):
    run, folder = await _run(deck["store"], EXPERT_LIST, {"funnel_expert": "petri"}, rid="bat_expert01")
    _list(deck)
    steps = await funnel.plan(run)
    # one turn, as itself, in a session opened for this run, before the recipe is written
    assert deck["asked"][0] == "setup" and len(expert["turns"]) == 1 and expert["woken"] == 1
    auto = expert["turns"][0]["automation"]
    assert auto["kind"] == "colleague" and auto["mail_write"] == "deny"
    assert auto["task"] == {"id": "exp-bat_expert01", "name": "Expert for t", "session_id": "", "mirror": False}
    assert "Build the list." in expert["turns"][0]["prompt"] and "as its owner" in expert["turns"][0]["prompt"]
    assert json.loads((folder / funnel_expert.STATE).read_text())["session"] == "sess-1"
    assert "core-facility pages" in deck["setup_prompts"][0] and "Dr. Petri" in deck["setup_prompts"][0]
    await _go(deck["store"], run, steps)
    assert "core-facility pages" in deck["decide_prompts"][0]  # every judging call reads its notes
    # the new searches are the expert's, on its own model, with what it knows ahead of the question
    ask = expert["asks"][0]
    assert ask["reader"]["agent"] == "petri" and "Answer as a microbiologist." in ask["system"]
    assert "Plates are counted after 48 hours." in ask["system"] and "core-facility pages" in ask["user"]
    assert "queries" not in deck["asked"]
    assert [r["Name"] for r in funnel._read_csv(folder / "leads.csv")] == ["N0", "N1"]
    summary = (folder / "summary.md").read_text()
    assert "## The expert" in summary and "Dr. Petri worked beside this run" in summary
    assert "- Does a two-person lab count? Yes, any lab with a bench." in summary and "- Hospital labs? No." in summary
    assert "It wrote 1 of the run's new searches. They found 1 items, 1 of them on the list." in summary
    assert summary.index("## The expert") < summary.index("## Files")
    assert expert["released"] == 1 and len(expert["turns"]) == 2  # its notes, its check at the end; then let go
    said = [json.loads(ln) for ln in (folder / funnel_expert.LOG).read_text().splitlines()]
    assert [x["what"] for x in said][:2] == ["notes", "searches"]


async def test_a_run_goes_on_without_an_expert_it_cant_reach(deck, expert):
    expert["ready"] = False
    run, folder = await _run(deck["store"], EXPERT_LIST, {"funnel_expert": "petri"}, rid="bat_expert02")
    _list(deck)
    deck["queries"] = ["core facility directory"]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert expert["turns"] == [] and expert["asks"] == [] and "queries" in deck["asked"]  # the run's own model wrote them
    assert [r["Name"] for r in funnel._read_csv(folder / "leads.csv")] == ["N0", "N1"]
    assert "wasn't available (it couldn't be woken or started)" in (folder / "summary.md").read_text()
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert any("isn't available" in n and "goes on without it" in n for n in notes)


async def test_an_agent_that_isnt_the_owners_is_never_asked(deck, expert, monkeypatch):
    monkeypatch.setattr(first_mate_agent, "mine", lambda owner: [])
    run, folder = await _run(deck["store"], EXPERT_LIST, {"funnel_expert": "petri"}, rid="bat_expert03")
    _list(deck)
    await funnel.plan(run)
    assert expert["woken"] == 0 and expert["turns"] == []
    assert json.loads((folder / funnel_expert.STATE).read_text())["why"] == "it isn't one of the owner's agents any more"


async def test_a_run_without_an_expert_is_as_before(deck, expert):
    run, folder = await _run(deck["store"], EXPERT_LIST, rid="bat_expert04")
    _list(deck)
    deck["queries"] = ["core facility directory"]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert expert["woken"] == 0 and not (folder / funnel_expert.STATE).exists()
    assert "## The expert" not in (folder / "summary.md").read_text()


def test_decisions_are_read_from_the_notes(tmp_path):
    (tmp_path / "shared" / "funnel").mkdir(parents=True)
    (tmp_path / funnel_expert.NOTES).write_text("From X:\n\nNOTES\nSomething.\n\n**DECISIONS**\n1. A? Yes.\nNone\n- B? No.\n")
    assert funnel_expert.decisions(tmp_path) == ["A? Yes.", "B? No."]
    (tmp_path / funnel_expert.NOTES).write_text("NOTES\nSomething.\n\nDECISIONS\nNone.\n")
    assert funnel_expert.decisions(tmp_path) == []


async def test_the_expert_writes_the_owners_columns_of_kept_rows(deck, expert):
    """Phase 2. A why-it-fits column was written by the run's small model from one or two
    ideas. With an expert beside the run it is the expert's to write, for rows that are
    being kept only, told what it wrote for the others."""
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv",
                                                     "columns": ["Name", "Reason", "Tier"], "min_rows": 9,
                                                     "allowed": {"Tier": ["a", "b"]}}}]
    run, folder = await _run(deck["store"], contract, {"funnel_expert": "petri"}, rid="bat_expert05")
    deck["recipe"] = {"item": "lab", "key": "Name", "decide": "Labs.", "decide_batch": 1,
                      "columns": [{"name": "Name"}, {"name": "Reason", "what": "why it would use it", "from": "expert"},
                                  {"name": "Tier", "from": "owner"}],
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["tools"] = _Tools(pages={}, results={"labs": "r"})
    deck["extract"] = lambda user: {"items": [{"name": n, "fields": {"Name": n}} for n in ("Alpha", "Beta", "Gamma")]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "drop" if f["Name"] == "Gamma" else "keep", "reason": "x",
                                     "row": {"Name": f["Name"], "Reason": "A lab.", "Tier": "a"}} for i, f in items]
    expert["queries"] = lambda: []

    def columns(user):
        name = "Alpha" if '"Name": "Alpha"' in user else "Beta"
        return {"1": {"Reason": f"{name} plates by hand; the counter reads a photo.", "Tier": "z" if name == "Alpha" else "B"}}

    expert["columns"] = columns
    steps = await funnel.plan(run)
    assert funnel.load_recipe(folder)["owner_columns"] == ["Reason", "Tier"]
    await _go(deck["store"], run, steps)
    rows = {r["Name"]: r for r in funnel._read_csv(folder / "leads.csv")}
    assert rows["Alpha"]["Reason"] == "Alpha plates by hand; the counter reads a photo."
    assert rows["Alpha"]["Tier"] == "a"   # "z" isn't one of its values: the judging call's stands
    assert rows["Beta"]["Tier"] == "b"    # written as the list writes it
    asked = [a for a in expert["asks"] if "# Columns to write" in a["user"]]
    assert len(asked) == 2 and not any("Gamma" in a["user"] for a in asked)  # only rows being kept
    assert "Written for other items lately" not in asked[0]["user"]
    assert "- Reason: Alpha plates by hand" in asked[1]["user"]  # told what it wrote before
    assert "core-facility pages" in asked[0]["user"] and "Answer as a microbiologist." in asked[0]["system"]
    assert "It wrote the owner's columns" in (folder / "summary.md").read_text() and "asked 2 times" in (folder / "summary.md").read_text()


async def test_the_judging_calls_columns_stand_when_the_expert_gives_none(deck, expert):
    contract = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv", "columns": ["Name", "Reason"]}}]
    run, folder = await _run(deck["store"], contract, {"funnel_expert": "petri"}, rid="bat_expert06")
    deck["recipe"] = {"item": "lab", "key": "Name", "decide": "Labs.",
                      "columns": [{"name": "Name"}, {"name": "Reason", "from": "owner"}],
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["tools"] = _Tools(pages={}, results={"labs": "r"})
    deck["extract"] = lambda user: {"items": [{"name": "Alpha", "fields": {"Name": "Alpha"}}]}
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"], "Reason": "Its own."}}
                                    for i, f in items]
    expert["queries"] = lambda: []
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert funnel._read_csv(folder / "leads.csv")[0]["Reason"] == "Its own."


# ── phase 3: the expert on items that come close ──────────────────────────

def _close_list(deck, expert, rules, decide, cfg=None):
    deck["recipe"] = {"item": "lab", "key": "Name", "site_column": "Web", "decide": "Teaching labs.", "decide_batch": 1,
                      "shortage_ok": True, "rules": rules,
                      "sources": [{"kind": "search", "group": "", "queries": ["labs"]}]}
    deck["tools"] = _Tools(pages={}, results={"labs": "r"})
    deck["extract"] = lambda user: {"items": [{"name": n, "fields": {"Name": n, "Web": f"https://{n.lower()}.example"}}
                                              for n in ("Alpha", "Beta", "Gamma")]}
    deck["decide"] = decide
    expert["queries"] = lambda: []


CLOSE = [{"id": "fm1", "text": "x", "check": {"type": "table", "path": "leads.csv", "columns": ["Name", "Web", "City"]}}]


async def test_the_expert_settles_a_near_miss_of_judgement_never_of_fact(deck, expert):
    run, folder = await _run(deck["store"], CLOSE, {"funnel_expert": "petri"}, rid="bat_expert07")
    rules = [{"rule": "it is a teaching lab", "kind": "must", "fact": "identity"},
             {"rule": "its city is shown", "kind": "must", "fact": "place", "columns": ["City"]}]

    def decide(items):
        out = []
        for i, f in items:
            lacks = [2] if f["Name"] == "Gamma" else [1]  # Gamma lacks a fact; the others a judgement
            out.append({"id": i, "verdict": "maybe", "missing": lacks, "reason": "Not sure.",
                        "row": {"Name": f["Name"], "Web": f["Web"], "City": "" if f["Name"] == "Gamma" else "Split"}})
        return out

    _close_list(deck, expert, rules, decide)
    expert["second"] = lambda user: ({"verdict": "keep", "reason": "A university's practical course lab."}
                                     if '"Name": "Alpha"' in user else {"verdict": "drop", "reason": "A contract lab."})
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    got = {r["name"]: (r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got["Alpha"][0] == "kept" and "settled by the run's expert: A university's practical course lab." in got["Alpha"][1]
    assert got["Beta"] == ("dropped", "the run's expert: A contract lab. (Not sure.)")
    assert got["Gamma"][0] == "maybe"  # a missing fact is never the expert's to settle: it wasn't asked
    asked = [a["user"] for a in expert["asks"] if "# What is open" in a["user"]]
    assert len(asked) == 2 and not any("Gamma" in a for a in asked)
    assert "- it is a teaching lab" in asked[0] and "The judging call said: Not sure." in asked[0]
    summary = (folder / "summary.md").read_text()
    assert "asked about 2 items that came close on a question of judgement: 1 it kept, 1 it dropped, 0 it left" in summary


async def test_a_row_the_expert_keeps_still_passes_the_lists_checks(deck, expert):
    run, folder = await _run(deck["store"], CLOSE, {"funnel_expert": "petri"}, rid="bat_expert08")
    rules = [{"rule": "it is a teaching lab", "kind": "must", "fact": "identity"}]
    _close_list(deck, expert, rules, lambda items: [
        {"id": i, "verdict": "maybe", "missing": [1], "reason": "Not sure.",
         "row": {"Name": f["Name"], "Web": f["Web"], "City": "Atlantis" if f["Name"] == "Alpha" else ""}} for i, f in items])
    deck["recipe"]["checks"] = [{"column": "City", "op": "required"}, {"column": "City", "op": "in_evidence"}]
    expert["second"] = lambda user: {"verdict": "keep", "reason": "It is one."}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert got == {"Alpha": "maybe", "Beta": "maybe", "Gamma": "maybe"}  # a city nobody read, or none: not on the list
    assert not funnel._read_csv(folder / "leads.csv")


async def test_the_expert_turns_a_kept_row_down_only_when_switched_on(deck, expert):
    assert fms.check_runner_config("funnel", {"expert": "petri", "expert_veto": True})["expert_veto"] is True
    assert "expert_veto" not in fms.check_runner_config("funnel", {"expert_veto": True})  # no expert: nothing to switch
    assert first_mate_runs.funnel_config({"runner_config": {"expert": "petri", "expert_veto": True}})["funnel_expert_veto"]
    keep = lambda items: [{"id": i, "verdict": "keep", "reason": "Fits.",  # noqa: E731
                           "row": {"Name": f["Name"], "Web": f["Web"], "City": "Split"}} for i, f in items]
    expert["columns"] = lambda user: {"1": {"drop": "A reseller, not a lab."}} if '"Name": "Beta"' in user else {}
    for rid, cfg, kept in (("bat_von00001", {"funnel_expert": "petri", "funnel_expert_veto": True}, ["Alpha", "Gamma"]),
                           ("bat_voff0001", {"funnel_expert": "petri"}, ["Alpha", "Beta", "Gamma"])):
        run, folder = await _run(deck["store"], CLOSE, cfg, rid=rid)  # a run's folder is named by its id's start
        _close_list(deck, expert, [], keep)
        expert["asks"].clear()
        steps = await funnel.plan(run)
        await _go(deck["store"], run, steps)
        assert [r["Name"] for r in funnel._read_csv(folder / "leads.csv")] == kept, [(r["name"], r["status"], r["reason"]) for r in funnel._read_csv(folder / funnel.ITEMS)]
        told = any("you turn down instead" in a["system"] for a in expert["asks"])
        assert told == bool(cfg.get("funnel_expert_veto"))
    got = {r["name"]: r["reason"] for r in funnel._read_csv(
        bat_routes._run_folder(await deck["store"].get_run("bat_von00001")) / funnel.ITEMS)}
    assert got["Beta"] == "turned down by the run's expert: A reseller, not a lab. (Fits.)"


async def test_the_experts_calls_are_capped(deck, expert):
    run, folder = await _run(deck["store"], CLOSE, {"funnel_expert": "petri", "funnel_expert_calls": 1}, rid="bat_expert11")
    rules = [{"rule": "it is a teaching lab", "kind": "must", "fact": "identity"}]
    _close_list(deck, expert, rules, lambda items: [
        {"id": i, "verdict": "maybe", "missing": [1], "reason": "Not sure.",
         "row": {"Name": f["Name"], "Web": f["Web"], "City": "Split"}} for i, f in items])
    expert["second"] = lambda user: {"verdict": "keep", "reason": "It is one."}
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert len(expert["asks"]) == 1  # one call, then the run goes on without asking
    got = {r["name"]: r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)}
    assert sorted(got.values()) == ["kept", "maybe", "maybe"]
    notes = [e["message"] for e in await deck["store"].list_events(run["id"]) if e["stage"] == "note"]
    assert sum("its 1 calls for this run are used" in n for n in notes) == 1


# ── phase 4: the expert's check at the end, and its tokens ────────────────

async def test_the_expert_checks_a_sample_at_the_end_and_changes_no_row(deck, expert):
    run, folder = await _run(deck["store"], CLOSE, {"funnel_expert": "petri"}, rid="bat_xcheck01")
    _close_list(deck, expert, [], lambda items: [
        {"id": i, "verdict": "drop" if f["Name"] == "Gamma" else "keep", "reason": "Fits." if f["Name"] != "Gamma" else "A shop.",
         "row": {"Name": f["Name"], "Web": f["Web"], "City": "Split"}} for i, f in items])
    found = "Beta is a reseller and shouldn't be on the list.\n\nThe searches keep finding shops."
    expert["says"] = [SAID, found]
    steps = await funnel.plan(run)
    await _go(deck["store"], run, steps)
    assert len(expert["turns"]) == 2  # its notes before, its check after
    check = expert["turns"][1]
    assert check["automation"]["task"]["session_id"] == "sess-1"  # the same session of this run
    assert "- Name: Alpha | Web: https://alpha.example | City: Split" in check["prompt"]
    assert "- Gamma: A shop." in check["prompt"] and "(2 shown of 2)" in check["prompt"]
    assert [r["Name"] for r in funnel._read_csv(folder / "leads.csv")] == ["Alpha", "Beta"]  # Beta stays: it changes no row
    summary = (folder / "summary.md").read_text()
    assert "What it found on a sample of the result (it changed no row):" in summary
    assert "> Beta is a reseller and shouldn't be on the list." in summary and "> The searches keep finding shops." in summary
    assert (folder / funnel_expert.CHECK).read_text().startswith("Beta is a reseller")


async def test_the_experts_short_calls_count_toward_the_runs_tokens(deck, expert):
    run, folder = await _run(deck["store"], EXPERT_LIST, {"funnel_expert": "petri"}, rid="bat_xcount01")
    _list(deck)
    steps = await funnel.plan(run)
    before = int((await deck["store"].get_run(run["id"])).get("cumulative_tokens") or 0)
    await _go(deck["store"], run, steps)
    after = int((await deck["store"].get_run(run["id"])).get("cumulative_tokens") or 0)
    assert after - before == 7 * len(expert["asks"]) and expert["asks"]  # the stand-in says 7 tokens a call


# ── a model that fails is not a model that found nothing ──────────────────

def test_an_outage_is_told_from_a_request_the_model_refused():
    import httpx
    assert funnel.outage(httpx.ConnectError("All connection attempts failed"))
    assert funnel.outage(httpx.ReadTimeout("slow")) and funnel.outage(RuntimeError("InternalServerError: 503"))
    assert funnel.outage(RuntimeError("openai streaming callback failed: Connection error."))
    assert not funnel.outage(ValueError("this model's maximum context length is 8192 tokens"))
    assert not funnel.outage(RuntimeError("BadRequestError: 400 invalid schema"))


async def test_a_dead_model_stops_the_step_and_loses_nothing(deck):
    """A local model dropped mid-run. The step read each search's results as "no items",
    dropped what it couldn't judge as unanswered, used up its searches in four minutes
    and went on. Now calls are tried again, and when several in a row still fail the
    step stops: nothing is dropped, the searches it couldn't read aren't used up, and
    its next try carries on."""
    import httpx
    run, folder = await _run(deck["store"], REACH, rid="bat_down0001")
    deck["recipe"] = {"item": "lead", "key": "Name", "decide": "Leads.", "decide_batch": 1,
                      "sources": [{"kind": "search", "group": "", "queries": ["q1", "q2", "q3", "q4"]}]}
    deck["tools"] = _Tools(pages={}, results={f"q{n}": f"found N{n}" for n in range(1, 5)})
    down = {"on": False}

    def extract(user):
        if down["on"]:
            raise httpx.ConnectError("All connection attempts failed")
        return {"items": [{"name": m, "fields": {"Name": m}} for m in re.findall(r"found (N\d)", user)]}

    def decide(items):
        if items[0][1]["Name"] == "N1":
            down["on"] = True  # it answers for the first, then goes down
        if down["on"] and items[0][1]["Name"] != "N1":
            raise httpx.ConnectError("All connection attempts failed")
        return [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]

    deck["extract"], deck["decide"] = extract, decide
    steps = await funnel.plan(run)
    first = (await _go(deck["store"], run, steps[:1]))[0]
    assert not first["ok"] and "the model stopped answering" in first["error"] and "next try picks up" in first["error"]
    items = funnel._read_csv(folder / funnel.ITEMS)
    assert [(r["name"], r["status"]) for r in items] == [("N1", "kept")]  # nothing dropped for a call that failed
    state = json.loads((folder / funnel.QUERIES).read_text())
    assert state["used"] == ["q1"] and state["searches"] == 4  # the three it couldn't read will be made again
    notes = [(e.get("data") or {}).get("detail", "") for e in await deck["store"].list_events(run["id"])]
    assert any("the model stopped answering" in n and "3 searches" in n for n in notes)
    down["on"] = False
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    second = (await _go(deck["store"], run, steps[:1]))[0]
    assert second["ok"] and [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["N1", "N2", "N3", "N4"]


async def test_a_request_the_model_refuses_is_skipped_not_taken_for_an_outage(deck):
    run, folder = await _run(deck["store"], REACH, rid="bat_down0002")
    deck["recipe"] = {"item": "lead", "key": "Name", "decide": "Leads.", "decide_batch": 1,
                      "sources": [{"kind": "search", "group": "", "queries": ["q1", "q2", "q3", "q4", "q5"]}]}
    deck["tools"] = _Tools(pages={}, results={f"q{n}": f"found N{n}" for n in range(1, 6)})
    calls = []

    def extract(user):
        calls.append(1)
        if "found N5" not in user:
            raise ValueError("this model's maximum context length is 8192 tokens")  # four in a row, each once
        return {"items": [{"name": "N5", "fields": {"Name": "N5"}}]}

    deck["extract"] = extract
    deck["decide"] = lambda items: [{"id": i, "verdict": "keep", "row": {"Name": f["Name"]}} for i, f in items]
    steps = await funnel.plan(run)
    got = (await _go(deck["store"], run, steps[:1]))[0]
    assert got["ok"] and [r["Name"] for r in funnel._read_csv(folder / funnel.ROWS)] == ["N5"]
    assert len(calls) == 5  # a refused request isn't tried again
