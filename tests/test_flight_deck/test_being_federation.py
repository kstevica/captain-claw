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
