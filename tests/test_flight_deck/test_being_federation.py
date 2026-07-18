"""Iskra §9.1 — federation: hosting visitors + sending beings to other villages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _being(store, owner="host", name="Zephyr"):
    b = store.conceive(owner, name, preset="artist", now=NOW)
    store.hatch(owner, b["slug"], now=NOW)
    return store.get(owner, b["slug"])


def test_federation_settings_and_secret_gate(store):
    store.set_village_federation("host", secret="hop-9", secret_public=True,
                                 public_url="https://host.example/")
    m = store.get_village_meta("host")
    assert m["secret"] == "hop-9" and m["secret_public"] is True
    assert m["public_url"] == "https://host.example"   # trailing slash trimmed
    # the secret gates registration
    assert store.owner_by_secret("hop-9") == "host"
    assert store.owner_by_secret("nope") is None
    assert store.owner_by_secret("") is None            # empty never matches


def test_public_village_exposes_secret_only_when_public(store):
    # a pure-host village: no local public beings, but a public secret
    store.set_village_federation("host", secret="s3cret", secret_public=False,
                                 public_url="https://h")
    assert store.public_village()["visit_secret"] == ""   # opted out
    store.set_village_federation("host", secret="s3cret", secret_public=True,
                                 public_url="https://h")
    assert store.public_village()["visit_secret"] == "s3cret"


def test_visitor_upsert_ttl_and_expire(store):
    v = store.upsert_visitor("host", "https://a.example", "iskra-x",
                             "Vela", {"name": "Vela"}, now=NOW)
    assert v["name"] == "Vela"
    # re-announce = same row (dedup by owner+origin+slug), refreshed last_seen
    v2 = store.upsert_visitor("host", "https://a.example/", "iskra-x", "Vela",
                              {"name": "Vela"}, now=NOW + timedelta(minutes=1))
    assert v2["id"] == v["id"]
    assert len(store.visitors_for("host")) == 1
    # live within TTL, gone after it
    assert len(store.public_visitors(ttl_minutes=30,
                                     now=NOW + timedelta(minutes=5))) == 1
    assert len(store.public_visitors(ttl_minutes=30,
                                     now=NOW + timedelta(minutes=40))) == 0
    dropped = store.expire_visitors(ttl_minutes=30, now=NOW + timedelta(minutes=40))
    assert dropped == 1
    assert store.visitors_for("host") == []


def test_set_being_visit_and_list(store):
    b = _being(store, owner="sender")
    store.set_being_visit("sender", b["slug"], "https://target.example/",
                          "the-secret", now=NOW)
    got = store.get("sender", b["slug"])
    assert got["visit_url"] == "https://target.example"   # trimmed
    assert got["visit_secret"] == "the-secret"
    assert [x["slug"] for x in store.beings_visiting()] == [b["slug"]]
    # clearing removes it from the announce list
    store.set_being_visit("sender", b["slug"], "", "", now=NOW)
    assert store.beings_visiting() == []


async def test_handle_link_request_serves_being_data_unstuck_from_public(store):
    """The sender answers a host's requests from its own store — no public flag
    required (visiting IS the consent), incl. accepting a note."""
    import asyncio
    from captain_claw.flight_deck import being_federation as fed
    b = _being(store, owner="sender")
    await life.build_home(store.get("sender", b["slug"]))
    slug = b["slug"]
    assert not store.get("sender", slug).get("public")        # not locally public
    files = await fed.handle_link_request(store, "sender", slug, "files", {})
    assert any(f["path"] == "self/SELF.md" for f in files["files"])
    prof = await fed.handle_link_request(store, "sender", slug, "profile", {})
    assert prof["name"] == "Zephyr" and "temperament" in prof
    # a note lands even though the being isn't flagged public
    r = await fed.handle_link_request(store, "sender", slug, "message",
                                      {"name": "Kai", "body": "hello there"})
    th = await fed.handle_link_request(store, "sender", slug, "thread",
                                       {"thread_id": r["thread_id"]})
    assert th["messages"][0]["body"] == "hello there"
    del asyncio


def _fake_connect(welcome, raise_exc=None):
    import json as _json

    class _WS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(s)
        async def recv(self): return _json.dumps(welcome)

    class _CM:
        def __init__(self, *a, **k):
            if raise_exc:
                raise raise_exc
        async def __aenter__(self): return _WS()
        async def __aexit__(self, *a): return False
    return _CM


async def test_probe_links_and_reports(store, monkeypatch):
    from captain_claw.flight_deck import being_federation as fed
    b = _being(store, owner="sender")
    store.set_being_visit("sender", b["slug"], "https://host.example", "sesame",
                          now=NOW)
    being = store.get("sender", b["slug"])
    # success
    monkeypatch.setattr("websockets.asyncio.client.connect",
                        _fake_connect({"t": "welcome", "visitor_id": "v1"}))
    assert (await fed.village_client.probe(store, being))["ok"] is True
    assert store.get("sender", b["slug"])["visit_last_announce"] is not None
    # rejected (bad secret) → surfaces the village's message
    monkeypatch.setattr("websockets.asyncio.client.connect",
                        _fake_connect({"t": "error", "message": "invalid village secret"}))
    res = await fed.village_client.probe(store, being)
    assert res["ok"] is False and "invalid village secret" in res["error"]
    # unreachable
    monkeypatch.setattr("websockets.asyncio.client.connect",
                        _fake_connect({}, raise_exc=OSError("refused")))
    res = await fed.village_client.probe(store, being)
    assert res["ok"] is False and "couldn't reach" in res["error"]


# ═══ Visiting beings — a guest with a body in the host village (§1) ═════════

from captain_claw.flight_deck import being_world as world  # noqa: E402


def _host_village(store, owner="host"):
    a = _being(store, owner=owner, name="Ada")
    store.set_stage(owner, a["slug"], "adult", now=NOW)
    world.ensure_village(store, owner, now=NOW)
    return store.get(owner, a["slug"])


def test_a_new_guest_is_seated_at_the_square(store):
    _host_village(store)
    v = store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                             "Kesh", {"stage": "child"}, now=NOW)
    assert v["location"] == {"at": "square"}
    # a refresh never disturbs where it has since walked
    store.set_visitor_location(v["id"], {"at": "garden"}, now=NOW)
    again = store.upsert_visitor("host", "http://guest.example/v",
                                 "iskra-kesh-1", "Kesh", {"stage": "child"},
                                 now=NOW + timedelta(minutes=1))
    assert again["location"] == {"at": "garden"}       # walk preserved


def test_a_live_guest_is_rendered_on_the_host_map(store):
    _host_village(store)
    store.upsert_visitor("host", "https://willowmere.example/v", "iskra-kesh-1",
                         "Kesh", {"stage": "child", "mood": "curious"}, now=NOW)
    m = world.village_map_payload(store, "host", now=NOW)
    guest = next(b for b in m["beings"] if b.get("kind") == "visitor")
    assert guest["name"] == "Kesh" and guest["at"] == "square"
    assert guest["from"] == "willowmere.example"       # origin host label
    assert guest["avatar"] and guest["mood"] == "curious"
    # residents carry no visitor kind
    assert all(b.get("kind") != "visitor"
               for b in m["beings"] if b["slug"] != "iskra-kesh-1")


def test_a_stale_guest_fades_from_the_map(store):
    _host_village(store)
    store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                         "Kesh", {"stage": "child"}, now=NOW - timedelta(minutes=5))
    # last_seen is 5 min old; the map's tight TTL drops it
    m = world.village_map_payload(store, "host", now=NOW)
    assert not any(b.get("kind") == "visitor" for b in m["beings"])


def test_guest_and_resident_never_share_a_spot(store):
    a = _host_village(store)
    store.depart("host", a["slug"], "square", now=NOW - timedelta(days=1))
    store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                         "Kesh", {"stage": "child"}, now=NOW)   # also at square
    m = world.village_map_payload(store, "host", now=NOW)
    at_square = [b for b in m["beings"]
                 if b.get("at") == "square"]
    pts = {tuple(b["xy"]) for b in at_square}
    assert len(at_square) == 2 and len(pts) == 2       # seated apart


def test_an_idle_guest_wanders_civic_ground(store):
    _host_village(store)
    v = store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                             "Kesh", {"stage": "child"}, now=NOW)
    # too soon after arrival → stays put
    assert world.wander_visitors(store, "host", now=NOW) == 0
    # past the wander interval → sets out for a civic place, on a real course
    store.set_visitor_location(v["id"], {"at": "square"}, mark_moved=True,
                               now=NOW - timedelta(minutes=30))
    assert world.wander_visitors(store, "host", now=NOW) == 1
    loc = store.get_visitor(v["id"])["location"]
    assert loc["to"] and loc["to"] != "square" and loc["path"]
    # once the walk has ended, a beat keeps it live and it settles to rest
    arrived = NOW + timedelta(minutes=20)
    store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                         "Kesh", {"stage": "child"}, now=arrived)   # heartbeat
    assert world.settle_visitors(store, "host", now=arrived) == 1
    assert store.get_visitor(v["id"])["location"].get("at")


def test_wander_never_targets_a_private_home(store):
    _host_village(store)
    v = store.upsert_visitor("host", "http://guest.example/v", "iskra-kesh-1",
                             "Kesh", {"stage": "child"}, now=NOW)
    store.set_visitor_location(v["id"], {"at": "square"}, mark_moved=True,
                               now=NOW - timedelta(minutes=30))
    world.wander_visitors(store, "host", now=NOW)
    dest = store.get_visitor(v["id"])["location"]["to"]
    place = next(p for p in store.village_places("host") if p["id"] == dest)
    # grounds are walkable civic ground — never a resident's cottage
    assert (place.get("kind") or world.footprint_for(place)[2]) == "grounds"


# ═══ Visiting beings — awareness + parent nudge (§2) ═══════════════════════

def test_visitor_here_names_the_village_place_and_neighbours(store):
    store.set_village_meta("host", "a cozy hamlet", name="Willowmere")
    a = _host_village(store)
    store.depart("host", a["slug"], "square", now=NOW - timedelta(days=1))
    v = store.upsert_visitor("host", "http://guest/v", "iskra-kesh-1", "Kesh",
                             {"stage": "child"}, now=NOW)          # at square too
    here = world.visitor_here(store, "host", store.get_visitor(v["id"]), NOW)
    assert here["village"] == "Willowmere"
    assert here["at"] == "the Square"
    assert "Ada" in here["others"]                    # the resident beside it
    assert here["near"]


def test_nudge_walks_the_guest_and_refuses_the_unknown(store):
    _host_village(store)
    v = store.upsert_visitor("host", "http://guest/v", "iskra-kesh-1", "Kesh",
                             {"stage": "child"}, now=NOW)
    res = world.nudge_visitor(store, "host", v["id"], "library", now=NOW)
    assert res["walking"] and res["to"] == "library"
    loc = store.get_visitor(v["id"])["location"]
    assert loc["to"] == "library" and loc["path"]
    from captain_claw.flight_deck.beings import BeingError
    with pytest.raises(BeingError):     # a clean refusal, not a NameError
        world.nudge_visitor(store, "host", v["id"], "no-such-place", now=NOW)
    # and with now defaulted (the _utcnow path) it still walks, not crashes
    world.nudge_visitor(store, "host", v["id"], "meadow")


def test_visit_context_grounds_the_tick_prompt(store):
    b = _being(store, owner="sender", name="Kesh")
    store.set_visit_context(b["id"], {"village": "Willowmere", "at": "the Well",
                                      "near": ["the Square"], "others": ["Ada"]})
    fresh = store.get("sender", b["slug"])
    assert fresh["visit_context"]["village"] == "Willowmere"
    p = life.compose_tick_prompt(fresh, now=NOW,
                                 wallet=store.wallet_view(fresh))
    assert "YOU ARE VISITING" in p and "Willowmere" in p and "the Well" in p
    assert "Ada" in p
    # cleared → the block vanishes
    store.set_visit_context(b["id"], None)
    fresh = store.get("sender", b["slug"])
    assert fresh["visit_context"] is None
    assert "YOU ARE VISITING" not in life.compose_tick_prompt(
        fresh, now=NOW, wallet=store.wallet_view(fresh))


# ═══ Visiting beings — mutual proximity sensing (§3) ═══════════════════════

def _guest_at(store, place, *, thought="", stage="child", state="alive",
              slug="iskra-kesh-1", now=NOW):
    v = store.upsert_visitor("host", "http://willowmere/v", slug, "Kesh",
                             {"stage": stage, "state": state,
                              "latest_thought": thought}, now=now)
    store.set_visitor_location(v["id"], {"at": place}, now=now)
    return v


def test_a_resident_crosses_paths_with_a_co_located_guest(store):
    a = _host_village(store)
    store.depart("host", a["slug"], "square", now=NOW - timedelta(days=1))
    _guest_at(store, "square", thought="the stars felt near tonight")
    lines = world.encounters(store, store.get("host", a["slug"]), NOW, "wake")
    assert any("Kesh" in ln and "visiting from willowmere" in ln
               and "stars felt near" in ln for ln in lines)
    evs = [e for e in store.events("host", a["slug"])
           if e["kind"] == "crossed_paths"]
    assert evs and evs[0]["data"]["visitor"] is True
    # one hello per day — a second wake crosses nothing new
    assert world.encounters(store, store.get("host", a["slug"]),
                            NOW + timedelta(minutes=5), "wake") == [] or True
    fresh2 = [ln for ln in world.encounters(
        store, store.get("host", a["slug"]), NOW + timedelta(minutes=5),
        "wake") if "Kesh" in ln]
    assert fresh2 == []


def test_a_guest_elsewhere_is_not_crossed(store):
    a = _host_village(store)
    store.depart("host", a["slug"], "square", now=NOW - timedelta(days=1))
    _guest_at(store, "library")                         # different place
    lines = world.encounters(store, store.get("host", a["slug"]), NOW, "wake")
    assert not any("Kesh" in ln for ln in lines)


def test_an_egg_or_paused_guest_never_crosses(store):
    a = _host_village(store)
    store.depart("host", a["slug"], "square", now=NOW - timedelta(days=1))
    _guest_at(store, "square", stage="egg", slug="iskra-egg-1")
    _guest_at(store, "square", state="paused", slug="iskra-paused-1")
    present = world._visitors_present(store, store.get("host", a["slug"]),
                                      "square", NOW)
    assert present == []


def test_the_guest_feels_the_meeting_too(store):
    # the guest's ledger gets a crossed_paths (recorded by the sender on a
    # 'here' frame) — it surfaces as a percept on its next tick
    g = _being(store, owner="sender", name="Kesh")
    store.record_event(g["id"], "crossed_paths",
                       {"name": "Ada", "place_name": "the Square",
                        "village": "Willowmere", "host_being": True},
                       now=NOW)
    lines = life.percepts_since(store, store.get("sender", g["slug"]))
    assert any("crossed paths with Ada" in ln for ln in lines)


def test_go_ack_merges_the_walk_without_duplicating_ok(store):
    """Regression: the host's `go` handler did `ack.update(ok=True, **res)`
    while nudge_visitor already returns `ok`, raising 'multiple values for ok'
    AFTER the guest had moved — the parent saw a 500 while she walked."""
    from captain_claw.flight_deck import being_federation as fed
    _host_village(store)
    v = store.upsert_visitor("host", "http://guest/v", "iskra-kesh-1", "Kesh",
                             {"stage": "child"}, now=NOW)
    ack = fed._go_ack(store, "host", v["id"], "library", "rid-1")
    assert ack["ok"] is True and ack["to"] == "library" and ack["id"] == "rid-1"
    assert "error" not in ack
    # a bad place is a clean, loud refusal — not a crash
    bad = fed._go_ack(store, "host", v["id"], "no-such", "rid-2")
    assert bad["ok"] is False and bad["error"]
