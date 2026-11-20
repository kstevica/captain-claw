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
    assert "Looked at 4 items: 1 kept, 1 maybe" in work["output"]
    assert (folder / "people.csv").read_text().splitlines() == ["Name,Email,Team", "Ana,ana@acme.example,a"]
    dropped = (folder / "exclusions.md").read_text()
    assert "matches what's excluded" in dropped and "bo@wrong.example" in dropped and "isn't in what was read" in dropped
    assert "di@delta.example" in (folder / funnel.MAYBE).read_text()
    statuses = [r["status"] for r in funnel._read_csv(folder / funnel.ITEMS)]
    assert sorted(statuses) == ["dropped", "dropped", "kept", "maybe"]  # the duplicate was never looked at twice
    note = (folder / "progress" / "step-1.md").read_text()
    assert "Kept so far: 1 of 3" in note and "1 kept, 1 maybe, 2 dropped" in note
    assert merged["output"].startswith("`people.csv`: 1 rows of 3.") and "`exclusions.md`: 2 dropped" in merged["output"]
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
    why = {r["key"]: r["reason"] for r in funnel._read_csv(folder / funnel.ITEMS) if r["status"] == "dropped"}
    assert why["beta.example"].startswith("51-200 people")
    assert "boss@gamma.example" in why["gamma.example"] and "isn't in https://gamma.example/contact" in why["gamma.example"]
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
    assert "isn't in https://gamma.example/contact" in why["gamma.example"]  # read and judged, not cut on size
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
    assert why["Delta"][0] == "dropped" and "made-up" in why["Delta"][1]  # a value given must still be in its source
    assert why["Beta"][0] == "dropped" and "Cites 3 is under 10" in why["Beta"][1]  # loosened only the code rule
    assert why["Gamma"][0] == why["Eps"][0] == "loose"
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
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")  # the first list of results: skipped
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
    # a model's own searches for a group count when they use a word for it ("law" for "legal")
    r = funnel.normalize_recipe({"sources": [
        {"kind": "search", "group": "x", "queries": ["boutique law firm directory", "CPA firms list"]},
        {"kind": "search", "group": "y", "queries": ["boutique law firm directory", "CPA firms list"]}]},
        {**funnel.from_contract(CONTRACT), "groups": {"x": 2, "y": 2}})
    r["sources"][0]["group"] = "legal/accounting boutiques"
    r["sources"][1]["group"] = "research/analysis shops"  # a copy of another group's: written for it
    funnel.tidy_recipe(r, "Find firms.")
    assert r["sources"][0]["queries"] == ["boutique law firm directory", "CPA firms list"]
    assert r["sources"][1]["queries"][:2] == ["list of research shops", "list of analysis shops"]  # item: "item"


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
    assert got["B"][0] == "dropped" and "Email" in got["B"][1]  # an address not on a page read never passes
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
