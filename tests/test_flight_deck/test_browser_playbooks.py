"""Captain for Chrome — web playbooks in the panel (browser_playbooks.py).

Pinned here:

* the chips for a page: playbooks for its site (and the same brand on another
  country's domain, and a site's part — Google Maps isn't Google Search),
  never ones that only start with a web search, the person's shelf first,
  and their own recordings made there;
* the store: its front (categories, the shelf, popular ones for the person's
  country), search, a whole category, "Mine";
* one playbook in full, and the shelf — per person, kept on the deck;
* a run is the person's own turn to the agent, naming web_playbook with the
  values, with the link and steps written out for an agent without it, its
  stop for an act, a folder with playbook.md; what's missing is refused
  before anything is sent; every route wants the extension's token.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import browser_clip as bc
from captain_claw.flight_deck import browser_playbooks as bp
from captain_claw.flight_deck import browser_recordings as br

from .test_browser_clip import _pair, _until_done, deck  # noqa: F401 — the fixture


@pytest.fixture
async def pdeck(deck, tmp_path, monkeypatch):  # noqa: F811
    from captain_claw.flight_deck import server as srv

    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "fd-data", raising=False)
    app = FastAPI()
    app.include_router(bc.router)
    app.include_router(bp.router)
    app.include_router(br.router)
    app.include_router(bc.owner_router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test")
    try:
        yield {**deck, "client": client, "data": tmp_path / "fd-data"}
    finally:
        await client.aclose()


def _ids(items):
    return [p["id"] for p in items]


@pytest.mark.parametrize("url, has, hasnt", [
    ("https://www.google.com/maps/place/Zagreb", "maps.travel_time", "ir.latest_quarter"),
    ("https://www.google.de/maps/@45.8,15.9,12z", "maps.travel_time", "google.showtimes"),
    ("https://www.google.com/search?q=dune", "google.showtimes", "maps.travel_time"),
    ("https://www.amazon.co.uk/gp/css/order-history", "amazon.track_order", "ebay.search"),
    ("https://play.google.com/store/account/subscriptions", "googleplay.cancel_subscription", "maps.travel_time"),
    ("https://mail.google.com/mail/u/0/", "gmail.waiting_for_reply", "googleplay.subscriptions"),
])
def test_the_chips_are_for_the_site_in_front(url, has, hasnt):
    found = _ids(bp.for_site(url))
    assert has in found and hasnt not in found


def test_no_chips_for_a_page_no_playbook_knows():
    assert bp.for_site("https://example.com/") == [] and bp.for_site("chrome://settings") == []
    assert bp.brand_of("amazon.co.uk") == "amazon" == bp.brand_of("smile.amazon.com")


async def test_the_store_front_search_and_categories(pdeck):
    c = pdeck["client"]
    assert (await c.get("/fd/clip/playbooks")).status_code == 401
    pat = {"Authorization": f"Bearer {await _pair(c, pdeck['jwt'])}"}
    front = (await c.get("/fd/clip/playbooks?region=hr", headers=pat)).json()
    assert front["total"] >= 300 and [x["id"] for x in front["categories"]][:2] == ["travel", "local"]
    assert front["shelf"] == [] and "maps.travel_time" in _ids(front["popular"])
    assert any(p["region"] == "hr" for p in front["popular"]) and all(p["region"] == "hr" for p in front["top"]["regional"])
    row = next(p for p in front["popular"] if p["id"] == "maps.travel_time")
    assert row == {**row, "kind": "look", "where": "any", "site": "google.com/maps", "needs": 2, "mine": False, "runs": 0}

    found = (await c.get("/fd/clip/playbooks/search?q=track+my+dhl+parcel", headers=pat)).json()["items"]
    assert found[0]["id"] == "dhl.track"
    money = (await c.get("/fd/clip/playbooks/search?category=money", headers=pat)).json()["items"]
    assert money and all(p["category"] == "money" for p in money) and len(money) == front["categories"][6]["count"]
    regional = (await c.get("/fd/clip/playbooks/search?category=regional&region=gb", headers=pat)).json()["items"]
    assert regional[0]["region"] == "gb"
    assert (await c.get("/fd/clip/playbooks/search?category=nope", headers=pat)).status_code == 400

    one = (await c.get("/fd/clip/playbooks/googleplay.cancel_subscription", headers=pat)).json()
    assert one["stop"] == "Cancel subscription" and one["inputs"][0]["name"] == "subscription"
    assert one["steps"][0].startswith("1. Find ‹Which subscription›") and one["read"][0]["name"] == "subscription"
    assert (await c.get("/fd/clip/playbooks/no.such", headers=pat)).status_code == 404


async def test_the_shelf_is_the_persons_and_its_chips_come_first(pdeck):
    c = pdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, pdeck['jwt'])}"}
    url = "https://www.amazon.com/gp/your-account/order-history"
    before = _ids((await c.get(f"/fd/clip/playbooks/for?url={url}", headers=pat)).json()["items"])
    assert before[0] == "amazon.track_order"           # an everyday one leads
    assert "amazon.wishlist" not in before[:1]
    on = await c.post("/fd/clip/playbooks/amazon.wishlist/shelf", headers=pat, json={"on": True})
    assert on.json() == {"ok": True, "mine": True, "shelf": ["amazon.wishlist"]}
    chips = (await c.get(f"/fd/clip/playbooks/for?url={url}", headers=pat)).json()
    assert chips["site"] == "amazon.com" and chips["items"][0]["id"] == "amazon.wishlist" and chips["items"][0]["mine"]
    assert len(chips["items"]) <= bp.FOR_SITE_MAX and chips["more"] == len(bp.for_site(url)) - len(chips["items"])
    mine = (await c.get("/fd/clip/playbooks/search?category=mine", headers=pat)).json()["items"]
    assert _ids(mine) == ["amazon.wishlist"]
    kept = json.loads((pdeck["data"] / "browser-playbooks" / "alice.json").read_text())
    assert kept["shelf"] == ["amazon.wishlist"]
    off = await c.post("/fd/clip/playbooks/amazon.wishlist/shelf", headers=pat, json={"on": False})
    assert off.json()["mine"] is False and (await c.get("/fd/clip/playbooks", headers=pat)).json()["shelf"] == []


async def test_the_persons_recordings_on_a_site_are_chips_too(pdeck):
    c = pdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, pdeck['jwt'])}"}
    made = await c.post("/fd/clip/recordings", headers=pat, json={"name": "Monthly report", "steps": [
        {"id": "s1", "kind": "open", "url": "https://portal.example/reports"},
        {"id": "s2", "kind": "click", "target": {"role": "button", "name": "New report"}}]})
    assert made.status_code == 200, made.text
    got = (await c.get("/fd/clip/playbooks/for?url=https://portal.example/reports/2026", headers=pat)).json()
    assert got["items"] == [] and got["recordings"] == [{"id": made.json()["id"], "name": "Monthly report", "steps": 2}]
    assert (await c.get("/fd/clip/playbooks/for?url=https://other.example/", headers=pat)).json()["recordings"] == []


async def test_a_run_is_the_persons_turn_naming_the_playbook(pdeck):
    c = pdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, pdeck['jwt'])}"}
    url = "/fd/clip/playbooks/maps.travel_time/run"
    missing = await c.post(url, headers=pat, json={"agent": "scout", "inputs": {"origin": "Zagreb"}})
    assert missing.status_code == 400 and "To" in missing.json()["detail"]
    bad = await c.post(url, headers=pat, json={"agent": "scout", "inputs": {"origin": "here", "destination": "Split"}})
    assert bad.status_code == 400 and "can't go in a link" in bad.json()["detail"]
    assert (await c.post(url, headers=pat, json={"agent": "nobody", "inputs": {"origin": "A", "destination": "B"}})).status_code == 404
    assert not list(pdeck["clips"].glob("*")) and not pdeck["agent"].frames

    res = await c.post(url, headers=pat, json={"agent": "scout", "inputs": {"origin": "Zagreb", "destination": "Split",
                                                                          "mode": "transit"},
                                              "note": "Before noon", "browser": "Chrome on macOS"})
    assert res.status_code == 200, res.text
    sent = res.json()
    folder = sent["folder"]
    md = (pdeck["clips"] / folder / "playbook.md").read_text()
    assert "Opens https://www.google.com/maps/dir/?api=1&origin=Zagreb&destination=Split&travelmode=transit" in md
    run = await _until_done(c, pat, sent["clip_id"])
    assert run["status"] == "done"
    frame = pdeck["agent"].frames[0]
    assert frame["source"] == "browser" and "automation" not in frame
    text = frame["content"]
    assert text.startswith("▶️ Run the playbook “How long from A to B” for me\n\n- From: Zagreb\n- To: Split")
    assert 'web_playbook(action="run", id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split", "mode": "transit"})' in text
    assert "This time: Before noon" in text and "Without web_playbook: open https://www.google.com/maps/dir/" in text
    assert frame["file_paths"] == [f"vfs:web-clips/{folder}/playbook.md"]
    assert pdeck["admitted"] == [text]
    assert (await c.get("/fd/clip/playbooks/maps.travel_time", headers=pat)).json()["runs"] == 1


async def test_an_act_run_says_who_presses_the_last_button(pdeck):
    c = pdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, pdeck['jwt'])}"}
    res = await c.post("/fd/clip/playbooks/googleplay.cancel_subscription/run", headers=pat,
                       json={"agent": "scout", "inputs": {"subscription": "Duolingo"}, "hands": False})
    assert res.status_code == 200, res.text
    await _until_done(c, pat, res.json()["clip_id"])
    frame = pdeck["agent"].frames[0]
    assert "It stops before “Cancel subscription”" in frame["content"] and "I press “Cancel subscription” myself" in frame["content"]
    assert "in my Chrome (target=\"chrome\")" in frame["content"]
    assert any("Agents can use this browser" in n for n in frame["attachment_notes"])
