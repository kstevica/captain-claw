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


async def test_announce_one_ships_a_snapshot(store, monkeypatch):
    b = _being(store, owner="sender")
    store.set_being_visit("sender", b["slug"], "https://target.example",
                          "sesame", now=NOW)
    store.set_village_federation("sender", secret="", secret_public=False,
                                 public_url="https://me.example")
    posted: list = []

    class _Resp:
        status_code = 200
        def json(self): return {"ok": True, "visitor_id": "v1"}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None):
            posted.append((url, json)); return _Resp()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    res = await life.announce_one(store, store.get("sender", b["slug"]), now=NOW)
    assert res["ok"] is True
    url, payload = posted[0]
    assert url == "https://target.example/fd/public/village/announce"
    assert payload["secret"] == "sesame"
    assert payload["origin"] == "https://me.example"
    assert payload["slug"] == b["slug"]
    assert payload["profile"]["name"] == "Zephyr"           # full snapshot
    assert "temperament" in payload["profile"]
    # the being is marked announced
    assert store.get("sender", b["slug"])["visit_last_announce"] is not None


async def test_announce_one_refuses_without_public_url(store):
    b = _being(store, owner="sender")
    store.set_being_visit("sender", b["slug"], "https://target.example", "x",
                          now=NOW)
    res = await life.announce_one(store, store.get("sender", b["slug"]), now=NOW)
    assert res["ok"] is False and "public URL" in res["error"]
