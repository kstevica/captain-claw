"""A Bat worker lives for one step's turn. One that outlived its turn doesn't come back.

What happened: a deck restart in the middle of a turn left the worker on record, the
next start brought it back as an idle agent, and it sat under First Mate in the
sidebar as "<factory> · Working · 1 worker" after its run was cancelled and its
factory archived, through every restart after.

Pinned here:

* a deck start drops Bat workers it didn't start (stopping any still alive) and
  starts everything else as before; a worker this deck started is left alone;
* when a run ends, every worker still on record for it is removed.
"""

from __future__ import annotations

import os

import pytest

from captain_claw.flight_deck import bat_routes
from captain_claw.flight_deck import server as srv
from captain_claw.flight_deck import proc_compat


@pytest.fixture
def registry(monkeypatch, tmp_path):
    reg: dict[str, dict] = {}
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: reg)
    monkeypatch.setattr(srv, "_save_process_registry", lambda r: None)
    return reg


class _Live:
    def poll(self):
        return None


class _Dead:
    def poll(self):
        return 0


def test_a_deck_start_drops_the_bat_workers_it_did_not_start(registry, monkeypatch, tmp_path):
    registry.update({
        "first-mate": {"name": "First Mate", "web_port": 24001, "role": "standing"},
        "helper": {"name": "Helper", "web_port": 24002},
        # a cancelled run's worker, labelled for its family
        "bat-bat-6c07-main": {"name": "Leads · main", "web_port": 24090, "role": "worker", "run": "bat:bat_6c07aaaa",
                              "factory": "fac_1", "description": "Bat worker"},
        # one that never got its label, still running from the deck before
        "bat-bat-1234-step-1": {"name": "x", "web_port": 24091, "description": "Bat worker", "pid": 424242},
        # this deck's own, mid-turn
        "bat-bat-live-step-2": {"name": "y", "web_port": 24092, "run": "bat:bat_live", "pid": os.getpid()},
    })
    (tmp_path / "bat-bat-6c07-main").mkdir()
    # a deck update stopped this deck's own worker of another run: it's dead, so it goes too
    registry["bat-bat-upd-step-1"] = {"name": "z", "web_port": 24093, "run": "bat:bat_upd"}
    monkeypatch.setattr(srv, "_processes", {"bat-bat-live-step-2": _Live(), "bat-bat-upd-step-1": _Dead()})
    killed, started = [], []
    monkeypatch.setattr(srv, "_kill_pid", lambda pid, timeout=5.0: killed.append(pid))
    real_alive = proc_compat.pid_alive
    monkeypatch.setattr(proc_compat, "pid_alive", lambda pid: pid == 424242 or real_alive(pid))
    monkeypatch.setattr(srv, "_start_registered_process", lambda slug, e: started.append(slug) or True)
    monkeypatch.setenv("FD_REATTACH_STAGGER_S", "0")

    srv._reattach_processes()

    assert sorted(registry) == ["bat-bat-live-step-2", "first-mate", "helper"]
    assert started == ["first-mate", "helper"]
    assert killed == [424242]
    assert not (tmp_path / "bat-bat-6c07-main").exists()


def test_what_counts_as_a_bat_worker():
    assert srv._is_bat_worker("anything", {"run": "bat:bat_1"})
    assert srv._is_bat_worker("bat-bat-1-step-1", {"description": "Bat worker"})
    assert not srv._is_bat_worker("bat-fan", {"description": "My bat facts agent"})
    assert not srv._is_bat_worker("vatra-1", {"run": "vatra:v1"})


def test_a_finished_run_takes_its_leftover_workers_with_it(monkeypatch):
    reg = {
        "bat-bat-r1-step-1": {"run": "bat:bat_r1"},
        "dubina-researcher-bat_r1-s2": {"run": "bat:bat_r1"},
        "bat-bat-r2-step-1": {"run": "bat:bat_r2"},
        "first-mate": {"role": "standing"},
    }
    monkeypatch.setattr(srv, "_load_process_registry", lambda: reg)
    torn: list[list[str]] = []
    monkeypatch.setattr(bat_routes, "_teardown", lambda slugs: torn.append(list(slugs)))
    assert sorted(bat_routes.retire_run_workers("bat_r1")) == ["bat-bat-r1-step-1", "dubina-researcher-bat_r1-s2"]
    assert bat_routes.retire_run_workers("bat_none") == []
    assert torn == [["bat-bat-r1-step-1", "dubina-researcher-bat_r1-s2"]]


async def test_the_finish_hook_removes_them(monkeypatch):
    retired: list[str] = []
    monkeypatch.setattr(bat_routes, "retire_run_workers", lambda rid: retired.append(rid) or [])

    async def no_cost(_run):
        return None

    monkeypatch.setattr(bat_routes, "_log_own_cost", no_cost)
    await bat_routes._bat_on_finish({"id": "bat_r1", "status": "cancelled", "owner_id": "u1",
                                     "config": {"source": "first_mate"}})
    assert retired == ["bat_r1"]
