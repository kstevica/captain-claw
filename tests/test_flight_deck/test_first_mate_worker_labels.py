"""First Mate's agents are named for their factory and grouped (follow-up).

Pinned here:

* a factory run's Bat worker shows under the factory's name and step, with
  "First Mate · <factory>" as its description, and asks for the First Mate
  group; its slug (bat-<run>-<step>) never changes, since sends and the
  tool-call register are tied to the run by it;
* an improvement run's worker says it's improving; a Bat run that isn't a
  factory's is left alone; a gone agent or factory is no error;
* the process list carries the group.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import first_mate_agent as mate
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck import server as srv


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / ".processes.json"
    path.write_text(json.dumps({
        "bat-bat_e81a-step-1": {"name": "bat-bat_e81a-step-1", "description": "Bat worker", "owner": "user-bob"},
        "bat-bat_e81a-step-2": {"name": "bat-bat_e81a-step-2", "description": "Bat worker", "owner": "user-bob"},
        "bat-bat_77aa-step-1": {"name": "bat-bat_77aa-step-1", "description": "Bat worker", "owner": "user-bob"},
    }))
    monkeypatch.setattr(srv, "PROCESS_REGISTRY_FILE", path)
    return path


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    return s


def _entry(registry, slug):
    return json.loads(registry.read_text())[slug]


async def test_a_factory_worker_is_named_and_grouped(registry, store):
    f = store.create("user-bob", title="Agentic Systems Research", charter="Research agentic systems weekly.")
    await mate.label_worker("bat-bat_e81a-step-1", {"config": {"factory_id": f["id"], "source": "first_mate"}},
                            "step-1")
    e = _entry(registry, "bat-bat_e81a-step-1")
    assert e["name"] == "Agentic Systems Research · step 1"
    assert e["description"] == "First Mate · Agentic Systems Research" and e["group"] == "First Mate"
    assert "bat-bat_e81a-step-1" in json.loads(registry.read_text())  # the slug stays
    await mate.label_worker("bat-bat_e81a-step-2", {"config": {"factory_id": f["id"], "source": "captain_swarm"}},
                            "step-2")
    assert _entry(registry, "bat-bat_e81a-step-2")["name"] == "Agentic Systems Research · improving · step 2"


async def test_other_runs_and_gone_things_are_left_alone(registry, store):
    before = registry.read_text()
    await mate.label_worker("bat-bat_77aa-step-1", {"config": {"source": "bat"}}, "step-1")  # not a factory's
    await mate.label_worker("bat-bat_77aa-step-1", {"config": {"factory_id": "fac_0000000000000000"}}, "step-1")
    await mate.label_worker("bat-gone-step-1", {"config": {"factory_id": "fac_0000000000000000"}}, "step-1")
    assert registry.read_text() == before
    f = store.create("user-bob", title="Inbox replies", charter="Draft replies to client mail.")
    await mate.label_worker("bat-not-there-step-1", {"config": {"factory_id": f["id"]}}, "step-1")
    assert registry.read_text() == before


def test_the_process_list_carries_the_group():
    assert srv.ProcessInfo(slug="x", name="x", status="running", web_port=1).group == ""
    assert srv.ProcessInfo(slug="x", name="x", status="running", web_port=1, group="First Mate").group == "First Mate"
