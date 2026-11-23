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
