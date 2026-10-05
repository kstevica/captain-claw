"""Custom tool-poller sources dedup polled items across polls AND restarts.

The keys used to come from built-in ``hash()`` (salted per process via
PYTHONHASHSEED), so every FD restart re-fired already-seen items; and a row with
no id field (e.g. a tool returning one JSON object) got an empty key, so every
poll ingested it again and forced an arbiter pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import captain_claw.flight_deck.event_sources as es
from captain_claw.flight_deck import actions, autonomy, fd_dispatch
from captain_claw.flight_deck.events import EventsStore

_ROOT = Path(__file__).resolve().parents[2]

# One poll of a custom source "feed" whose tool returns argv[1], at time argv[2],
# into the events DB under FD_DATA_DIR — i.e. one FD process lifetime.
_ONE_POLL = textwrap.dedent("""
    import asyncio, json, sys
    from captain_claw.flight_deck import actions, autonomy, fd_dispatch
    from captain_claw.flight_deck.events import EventsStore
    import captain_claw.flight_deck.event_sources as es

    autonomy.resolve_config = lambda uid: {"custom_sources": [
        {"name": "feed", "tool": "t", "enabled": True}]}
    fd_dispatch._strongest_agent = lambda uid: {"owner": uid}

    async def run_tool(agent, tool, args):
        return {"ok": True, "content": sys.argv[1]}

    actions.run_tool_on_agent = run_tool
    store = EventsStore()
    n = asyncio.run(es._poll_custom_sources("alice", float(sys.argv[2]), store))
    keys = [r[0] for r in store._c().execute(
        "SELECT dedup_key FROM external_events ORDER BY ingested_at").fetchall()]
    print(json.dumps({"ingested": n, "keys": keys}))
""")


def _poll_in_fresh_process(content: str, now: float, data_dir: Path, seed: str) -> dict:
    env = {**os.environ, "FD_DATA_DIR": str(data_dir), "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", _ONE_POLL, content, str(now)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("content", [
    "Build is green; 3 open PRs",                          # unstructured digest
    json.dumps({"status": "green", "open_prs": 3}),        # one JSON object, no id
    json.dumps([{"title": "a"}, {"title": "b"}]),          # id-less rows
])
def test_restart_does_not_refire_already_seen_items(tmp_path, content):
    first = _poll_in_fresh_process(content, 1_000.0, tmp_path, seed="1")
    # Same DB, new process with a different hash seed, past the poll interval.
    second = _poll_in_fresh_process(content, 100_000.0, tmp_path, seed="2")
    assert first["ingested"] >= 1
    assert all(first["keys"])
    assert second["ingested"] == 0
    assert second["keys"] == first["keys"]


# ── in-process: polls against a real EventsStore ───────────────────────


@pytest.fixture()
def feed(tmp_path, monkeypatch):
    """A custom source "feed" whose tool output is ``feed.content``; ``feed.poll()``
    runs one poll (each past the interval) and returns the ingested count."""
    store = EventsStore(tmp_path / "events.db")

    class Feed:
        content = ""
        now = 1_000.0
        source: dict = {"name": "feed", "tool": "t", "enabled": True}

        async def poll(self) -> int:
            self.now += 10_000.0
            return await es._poll_custom_sources("alice", self.now, store)

        def keys(self) -> list[str]:
            return [r[0] for r in store._c().execute(
                "SELECT dedup_key FROM external_events ORDER BY ingested_at").fetchall()]

    f = Feed()

    async def run_tool(agent, tool, args):
        return {"ok": True, "content": f.content}

    monkeypatch.setattr(autonomy, "resolve_config", lambda uid: {"custom_sources": [f.source]})
    monkeypatch.setattr(fd_dispatch, "_strongest_agent", lambda uid: {"owner": uid})
    monkeypatch.setattr(actions, "run_tool_on_agent", run_tool)
    return f


async def test_single_object_result_dedups_across_polls(feed):
    feed.content = json.dumps({"status": "green", "open_prs": 3})
    assert await feed.poll() == 1
    assert await feed.poll() == 0
    assert await feed.poll() == 0
    [key] = feed.keys()
    assert key.startswith("feed:") and key != "feed:"


async def test_single_object_key_ignores_field_order_and_fires_on_change(feed):
    feed.content = json.dumps({"status": "green", "open_prs": 3})
    assert await feed.poll() == 1
    feed.content = json.dumps({"open_prs": 3, "status": "green"})
    assert await feed.poll() == 0
    feed.content = json.dumps({"status": "red", "open_prs": 3})
    assert await feed.poll() == 1


async def test_single_object_with_id_keys_on_the_id(feed):
    feed.content = json.dumps({"id": "run-7", "status": "green"})
    assert await feed.poll() == 1
    feed.content = json.dumps({"id": "run-7", "status": "red"})
    assert await feed.poll() == 0  # same item by id, as for list rows
    assert feed.keys() == ["feed:run-7"]


async def test_list_rows_with_ids_keep_their_keys(feed):
    feed.source = {**feed.source, "id_field": "num"}
    feed.content = json.dumps({"items": [{"num": 1, "t": "a"}, {"num": 2, "t": "b"}]})
    assert await feed.poll() == 2
    assert await feed.poll() == 0
    feed.content = json.dumps({"items": [{"num": 2, "t": "b"}, {"num": 3, "t": "c"}]})
    assert await feed.poll() == 1
    assert feed.keys() == ["feed:1", "feed:2", "feed:3"]


async def test_unstructured_output_dedups_by_content(feed):
    feed.content = "Build is green"
    assert await feed.poll() == 1
    assert await feed.poll() == 0
    feed.content = "Build is red"
    assert await feed.poll() == 1


def test_keys_do_not_come_from_builtin_hash(monkeypatch):
    # hash() is salted per process; a key built from it can't survive a restart.
    def no_hash(_v):
        raise AssertionError("dedup key built from built-in hash()")

    monkeypatch.setattr(es, "hash", no_hash, raising=False)
    keys = {es._dedup_key("feed", "Build is green"),
            es._dedup_key("feed", {"status": "green"})}
    assert len(keys) == 2 and all(k.startswith("feed:") for k in keys)
