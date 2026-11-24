"""Web playbooks: the shipped seed keeps the rules, and the rules catch what they should."""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import web_playbooks as wp

ROOT = Path(__file__).resolve().parents[2]


def _book(**over):
    base = {
        "id": "maps.travel_time", "title": "How long from A to B",
        "says": ["How long from Zagreb to Split?", "How far is the airport?"],
        "kind": "look", "where": "any", "sites": ["google.com/maps"],
        "inputs": {"origin": {"label": "From", "kind": "place"}, "destination": {"label": "To", "kind": "place"}},
        "open": "https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}",
        "read": {"duration": "The fastest route's time"},
    }
    base.update(over)
    return base


def _problems(raw) -> list[str]:
    try:
        wp.clean_playbook(raw, "travel")
    except wp.PlaybookError as exc:
        return exc.problems
    return []


def test_the_seed_keeps_every_rule():
    good, problems = wp.check_seed()
    assert problems == []
    assert len(good) >= 300, f"the seed is {len(good)} playbooks; it should be hundreds"


def test_every_category_has_its_file_and_ids_are_unique():
    good, _ = wp.check_seed()
    assert {b["category"] for b in good} == set(wp.CATEGORIES)
    ids = [b["id"] for b in good]
    assert len(ids) == len(set(ids))


def test_act_playbooks_all_stop_and_none_types_a_secret():
    good, _ = wp.check_seed()
    for book in good:
        if book["kind"] == "act":
            assert book["stop"], book["id"]
        assert not any(s.get("secret") for s in book["steps"]), book["id"]


def test_money_playbooks_only_look():
    good, _ = wp.check_seed()
    assert all(b["kind"] == "look" for b in good if b["category"] == "money")


def test_a_good_playbook_comes_back_clean():
    book = wp.clean_playbook(_book(), "travel")
    assert book["inputs"]["origin"] == {"label": "From", "kind": "place", "required": True, "ask": "before"}
    assert book["region"] == "global" and book["stop"] == "" and book["steps"] == []


@pytest.mark.parametrize("over, expect", [
    ({"kind": "act"}, "names its stop"),
    ({"stop": "Book"}, "has no stop"),
    ({"open": "https://www.google.com/maps/dir/?api=1&origin={from_here}"}, "{from_here}"),
    ({"open": "https://www.bing.com/maps?q={origin}"}, "not one of sites"),
    ({"open": "http://www.google.com/maps"}, "https"),
    ({"steps": [{"kind": "type", "target": {"name": "Password"}, "value": "x", "secret": True}]}, "secret"),
    ({"steps": [{"kind": "type", "target": {"name": "Search"}, "value": "{nope}"}]}, "{nope}"),
    ({"steps": [{"kind": "wave"}]}, "kind 'wave'"),
    ({"steps": [{"kind": "key", "key": "F5"}]}, "key must be"),
    ({"read": {}}, "says what to read"),
    ({"says": ["only one"]}, "2–8"),
    ({"id": "Maps travel"}, "site.task"),
    ({"where": "anywhere"}, "where must be"),
    ({"region": "europe"}, "region"),
    ({"inputs": {"mode": {"label": "How", "kind": "choice", "options": ["car"]}}}, "two options"),
    ({"inputs": {"origin": {"label": "From", "kind": "place", "required": False}, "destination": {"label": "To"}}},
     "needs a default"),
    ({"answer": "{duration} via {route}"}, "{route}"),
    ({"needs": ["telepathy"]}, "needs are among"),
    ({"price": 3}, "unknown field 'price'"),
    ({"open": "{origin}"}, "or a url input on its own"),            # a place isn't a link
    ({"open": "https://{site}/login", "sites": [],
      "inputs": {"site": {"label": "Bank", "kind": "url"}}}, "sites needs at least one host"),
])
def test_the_rules_catch(over, expect):
    problems = _problems(_book(**over))
    assert any(expect in p for p in problems), problems


def test_a_site_the_person_names_opens_as_given():
    book = wp.clean_playbook(_book(
        id="web.bank_statements", kind="look", where="chrome", sites=[],
        inputs={"site": {"label": "Bank", "kind": "url"}}, open="{site}",
        steps=[{"kind": "open", "url": "{site}"}], read={"saved": "Statements saved"}), "accounts")
    assert book["open"] == "{site}" and book["sites"] == []


def test_a_duplicate_id_across_files_is_caught(tmp_path):
    one = "category: travel\nplaybooks:\n  - " + _yaml_book()
    (tmp_path / "travel.yaml").write_text(one, encoding="utf-8")
    (tmp_path / "local.yaml").write_text(one.replace("category: travel", "category: local"), encoding="utf-8")
    good, problems = wp.check_seed(tmp_path)
    assert len(good) == 1
    assert any("already used" in p for p in problems)


def _yaml_book() -> str:
    import yaml
    return yaml.safe_dump(_book(), sort_keys=False, allow_unicode=True).replace("\n", "\n    ").rstrip()


def test_vocabulary_matches_recordings():
    from captain_claw.flight_deck import browser_recordings as rec
    assert wp.STEP_KINDS == rec.KINDS
    assert wp.ASK_MODES == rec.ASK_MODES
    assert wp.STEP_KEYS == rec._KEYS


def test_the_seed_ships_with_the_package_and_the_app():
    assert "flight_deck/playbooks/*.yaml" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = (ROOT / "captain_claw.spec").read_text(encoding="utf-8")
    assert '"flight_deck", "playbooks"' in spec


# ── Running one: search, values, the link, the steps, the stop ───────


@pytest.mark.parametrize("ask, first", [
    ("How long from Zagreb to Split by car?", "maps.travel_time"),
    ("where is my amazon parcel", "amazon.track_order"),
    ("track DHL package", "dhl.track"),
    ("cheapest flights to Lisbon in May", "googleflights.cheapest_dates"),
    ("cancel my google play subscription", "googleplay.cancel_subscription"),
    ("book a table for 4 tonight", "opentable.book_table"),
])
def test_search_puts_the_fitting_playbook_first(ask, first):
    assert wp.search(ask)[0]["id"] == first


def test_search_keeps_to_the_persons_country_and_filters():
    assert "hzpp.train_times" in [b["id"] for b in wp.search("train Zagreb Rijeka", region="hr")]
    assert all(b["region"] in ("global", "eu", "gb") for b in wp.search("train times", region="gb", limit=20))
    assert all(b["category"] == "money" for b in wp.search("exchange rate", category="money"))
    assert wp.search("the of my") == []


def test_values_are_checked_and_put_in_shape():
    book = wp.by_id("maps.travel_time")
    got = wp.check_values(book, {"origin": "Zagreb", "destination": "Split, Croatia", "mode": "Transit"})
    assert got["values"] == {"origin": "Zagreb", "destination": "Split, Croatia", "mode": "transit"}
    assert got["problems"] == got["ask_before"] == got["decide"] == got["ask_during"] == []
    got = wp.check_values(book, {"origin": "here", "mode": "boat", "when": "now"})
    assert got["ask_before"] == ["destination"]
    assert any("can't go in a link" in p for p in got["problems"])
    assert any("one of: driving" in p for p in got["problems"])
    assert any("'when' isn't one of its inputs" in p for p in got["problems"])


@pytest.mark.parametrize("kind, raw, value, problem", [
    ("date", "2026-12-01", "2026-12-01", ""), ("date", "1.12.2026", None, "YYYY-MM-DD"),
    ("time", "7:05", "07:05", ""), ("time", "25:00", None, "HH:MM"),
    ("number", "2,5", "2.5", ""), ("number", "3", "3", ""), ("number", "lots", None, "a number"),
    ("url", "mybank.example", "https://mybank.example", ""), ("url", "http://mybank.example", None, "https://"),
    ("url", "javascript:alert(1)", None, "https://"), ("url", "nope", None, "web address"),
    ("flag", "yes", True, ""), ("flag", "maybe", None, "true or false"),
])
def test_each_kind_of_value(kind, raw, value, problem):
    got, said = wp._shape("x", {"label": "X", "kind": kind}, raw)
    assert got == value and problem in said and bool(said) == bool(problem)


def test_the_link_carries_the_values_and_a_named_site_goes_in_as_given():
    book = wp.by_id("maps.travel_time")
    values = wp.check_values(book, {"origin": "Zagreb", "destination": "Split, Croatia"})["values"]
    assert wp.open_link(book, values) == ("https://www.google.com/maps/dir/?api=1&origin=Zagreb"
                                          "&destination=Split%2C%20Croatia&travelmode=driving")
    bank = wp.by_id("web.bank_statements")
    slot = wp._SLOT.fullmatch(bank["open"]).group(1)
    values = wp.check_values(bank, {slot: "online.mybank.example/login?x=1&y=2"})["values"]
    assert wp.open_link(bank, values) == "https://online.mybank.example/login?x=1&y=2"


def _sample(spec: dict) -> object:
    if spec.get("options"):
        return spec["options"][0]
    return {"date": "2026-12-01", "time": "09:30", "number": "2", "url": "https://portal.example",
            "flag": True, "place": "Zagreb"}.get(spec["kind"], "sample")


def test_every_seed_playbook_fills_its_link_and_steps():
    for book in wp.seed():
        given = {n: _sample(s) for n, s in book["inputs"].items() if s["required"] and "default" not in s}
        got = wp.check_values(book, given)
        assert got["problems"] == got["ask_before"] == got["decide"] == [], book["id"]
        assert wp.open_link(book, got["values"]).startswith("https://"), book["id"]
        for line in wp.step_lines(book, got["values"]):
            assert not wp._SLOT.search(line), (book["id"], line)


def test_steps_read_as_instructions_with_the_values_in():
    book = wp.clean_playbook(_book(kind="act", stop="Book", read={}, steps=[
        {"kind": "type", "target": {"role": "combobox", "name": "From"}, "value": "{origin}"},
        {"kind": "click", "target": {"role": "button", "name": "Search"}, "note": "Only once both are in"},
        {"kind": "type", "note": "The date field", "value": "{depart_at}"},
        {"kind": "key", "key": "Enter"},
        {"kind": "do", "text": "Pick the fastest; stop before \"Book\"."},
    ], inputs={"origin": {"label": "From", "kind": "place"}, "destination": {"label": "To", "kind": "place"},
               "depart_at": {"label": "Leaving at", "kind": "time", "required": False}}), "travel")
    lines = wp.step_lines(book, {"origin": "Zagreb"})
    assert lines == ["1. Type “Zagreb” into the field “From”",
                     "2. Click the button “Search” — Only once both are in",
                     "3. Type “‹Leaving at›” into the date field",
                     "4. Press Enter",
                     "5. Pick the fastest; stop before \"Book\"."]


def test_the_stop_is_known_by_its_name_and_the_clicks_before_it_counted():
    assert wp.is_stop("Place order", "Place Order") and wp.is_stop("Check in (2)", "Check in")
    assert wp.is_stop("  Send! ", "Send") and not wp.is_stop("Send feedback", "Send")
    assert not wp.is_stop("Post your reply", "Reply") and not wp.is_stop("", "Send")
    assert wp.stop_clicks_before(wp.by_id("googleplay.cancel_subscription")) == 1   # opens the reason dialog
    assert wp.stop_clicks_before(wp.by_id("jira.create_issue")) == 1                # the top bar's Create
    assert wp.stop_clicks_before(wp.by_id("amazon.start_return")) == 0


def test_the_answer_line_needs_every_part():
    book = wp.by_id("maps.travel_time")
    values = {"origin": "Zagreb", "destination": "Split", "mode": "driving"}
    assert wp.answer_text(book, values, {"duration": "4 h", "route": "A1", "distance": "410 km"}) == \
        "4 h by driving via A1 (410 km)."
    assert wp.answer_text(book, values, {"duration": "4 h", "route": None, "distance": "410 km"}) == ""


# ── The person's recordings, for the agent (browser_hands.py) ────────

RECORDING = {
    "id": "rec_0a1b2c3d4e5f", "name": "Monthly report", "about": "Last month's VAT report from the portal",
    "site": "portal.example", "start_url": "https://portal.example/reports", "updated_at": "2026-11-20T10:00:00Z",
    "steps": [
        {"id": "s1", "kind": "open", "url": "https://portal.example/reports"},
        {"id": "s2", "kind": "click", "target": {"role": "button", "name": "New report"}},
        {"id": "s3", "kind": "type", "target": {"role": "textbox", "name": "Period"}, "value": "2026-10", "ask": "before"},
        {"id": "s4", "kind": "select", "target": {"role": "combobox", "name": "Format"}, "value": "PDF",
         "ask": "before", "options": ["PDF", "CSV"]},
        {"id": "s5", "kind": "type", "target": {"role": "textbox", "name": "Note"}, "ask": "during"},
    ],
}


def test_a_recording_reads_as_a_playbook_with_its_asks_as_inputs():
    from captain_claw.flight_deck import browser_recordings as br

    book = br.as_playbook(RECORDING)
    assert book["id"] == "rec_0a1b2c3d4e5f" and book["kind"] == "recording" and book["where"] == "recording"
    assert book["inputs"]["s3"]["label"] == "Period" and book["inputs"]["s3"]["hint"].endswith("2026-10")
    assert book["inputs"]["s4"] == {"label": "Format", "kind": "choice", "required": True, "ask": "before",
                                    "options": ["PDF", "CSV"], "hint": "When it was recorded: PDF"}
    assert "s5" not in book["inputs"]
    got = wp.check_values(book, {"s3": "2026-11", "s4": "csv"})
    assert got["values"] == {"s3": "2026-11", "s4": "CSV"} and not got["problems"]
    words = br.playbook_words(RECORDING, got["values"])
    text = "\n".join(words["lines"])
    assert "Type “2026-11” into the field “Period”" in text and "Choose “CSV” in" in text
    assert "ask_user" in text and words["more"] == 0


async def test_an_agent_lists_and_reads_only_its_owners_recordings(monkeypatch):
    import httpx
    from fastapi import FastAPI, HTTPException

    from captain_claw.flight_deck import agent_identity
    from captain_claw.flight_deck import browser_hands as bh
    from captain_claw.flight_deck import browser_recordings as br
    from captain_claw.flight_deck.agent_identity import AgentCaller

    def _caller(request, *, what):
        owner, _, slug = request.headers.get("X-Agent-Auth", "").partition(":")
        if not slug:
            raise HTTPException(403, "Unknown agent")
        return AgentCaller(slug=slug, owner=owner)

    monkeypatch.setattr(agent_identity, "require_agent_caller", _caller)
    monkeypatch.setattr(br, "load_all", lambda user: [RECORDING] if user == "alice" else [])
    app = FastAPI()
    app.include_router(bh.agent_router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test") as c:
        alice, bob = {"X-Agent-Auth": "alice:scout"}, {"X-Agent-Auth": "bob:ledger"}
        listed = (await c.get("/fd/browser-hands/agent/recordings", headers=alice)).json()["recordings"]
        assert [b["id"] for b in listed] == ["rec_0a1b2c3d4e5f"]
        assert (await c.get("/fd/browser-hands/agent/recordings", headers=bob)).json() == {"recordings": []}
        assert (await c.get("/fd/browser-hands/agent/recordings/rec_0a1b2c3d4e5f", headers=bob)).status_code == 404
        one = (await c.get("/fd/browser-hands/agent/recordings/rec_0a1b2c3d4e5f", headers=alice)).json()
        assert one["title"] == "Monthly report"
        words = (await c.post("/fd/browser-hands/agent/recordings/rec_0a1b2c3d4e5f/words", headers=alice,
                              json={"inputs": {"s3": "2026-11"}})).json()
        assert any("2026-11" in line for line in words["lines"])
        assert (await c.get("/fd/browser-hands/agent/recordings")).status_code == 403


def test_a_read_is_tried_only_where_the_opened_page_may_answer():
    assert wp.answers_as_opened(wp.by_id("xe.convert_amount"))           # no step before the read
    assert wp.answers_as_opened(wp.by_id("maps.travel_time"))            # its step only applies sometimes
    assert not wp.answers_as_opened(wp.by_id("rottentomatoes.scores"))   # "Open the matching result" first
    assert not wp.answers_as_opened(wp.by_id("googleplay.cancel_subscription"))   # an act never reads first
    looks = [b for b in wp.seed() if b["kind"] == "look"]
    assert 100 < sum(map(wp.answers_as_opened, looks)) < len(looks)


# ── number ranges ────────────────────────────────────────────────────


def test_a_number_input_has_a_range_and_a_value_outside_it_is_the_persons_to_fix():
    base = {"id": "x.nums", "title": "Numbers", "says": ["a", "b"], "kind": "look", "where": "any", "region": "global",
            "sites": ["example.com"], "open": "https://example.com/?n={n}", "read": {"r": "R"}}
    book = wp.clean_playbook({**base, "inputs": {"n": {"label": "N", "kind": "number", "default": 7}}}, "travel")
    assert book["inputs"]["n"]["min"] == 0 and "max" not in book["inputs"]["n"]
    book = wp.clean_playbook({**base, "inputs": {"n": {"label": "N", "kind": "number", "min": 1, "max": 9}}}, "travel")
    assert wp.range_words(book["inputs"]["n"]) == "1 to 9"
    checked = wp.check_values(book, {"n": "-51"})
    assert checked["out_of_range"] == ["n"] and checked["problems"] == ["n (N) is 1 to 9 (got -51)"]
    assert wp.check_values(book, {"n": "4"})["values"] == {"n": "4"}
    for bad, why in (({"kind": "number", "min": 5, "max": 1}, "min is above max"),
                     ({"kind": "text", "min": 1}, "only a number has min and max"),
                     ({"kind": "number", "default": -2}, "out of its range"),
                     ({"kind": "number", "max": "ten"}, "max must be a number")):
        with pytest.raises(wp.PlaybookError, match=why):
            wp.clean_playbook({**base, "inputs": {"n": {"label": "N", **bad}}}, "travel")
    # The shipped ones: nights can't go below 0, people below 1.
    assert wp.by_id("googleflights.cheapest_dates")["inputs"]["nights"]["min"] == 0
    assert wp.by_id("booking.reserve")["inputs"]["adults"]["min"] == 1
    assert wp.by_id("abrp.ev_route_plan")["inputs"]["start_charge"]["max"] == 100
