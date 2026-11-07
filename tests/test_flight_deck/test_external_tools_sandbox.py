# ruff: noqa: F811 — the `data` and `deck` fixtures are imported from test_external_tools
"""Custom tools, Phase 6 on the deck: every run and test step goes through the
launcher with its sandbox rules; an admin can ask for tools to run only in the
sandbox, and then a host without one runs none. (What the sandbox itself
refuses is in test_xtool_sandbox.py, which needs Linux with Landlock.)"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import external_tools as xt

from .test_external_tools import _as_published, _switch_on, data, deck  # noqa: F401


def _flags(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == flag]


def test_a_run_reads_its_version_writes_its_call_and_has_no_network_unless_asked(tmp_path):
    m = xt.parse_manifest({"name": "t", "title": "T", "description": "d", "entry": "main.py", "input": {"type": "object"},
                           "examples": [{"input": {}}]})
    vdir, call = tmp_path / "v", tmp_path / "c"
    cmd = xt._launch(m, vdir / ".deps/bin/python", ["main.py"], cpu_s=10, read_only=[vdir], writable=[call])
    assert str(vdir) in _flags(cmd, "--ro") and _flags(cmd, "--rw") == [str(call)]
    assert "/proc/self" in _flags(cmd, "--ro") and "--no-net" in cmd and "--require-sandbox" not in cmd
    # Nothing writable but the call's folder: not the data folder, not the version.
    assert str(vdir) not in _flags(cmd, "--rw")
    netted = xt.parse_manifest({**m.raw, "network": True})
    cmd = xt._launch(netted, vdir / ".deps/bin/python", ["main.py"], cpu_s=10, read_only=[vdir], writable=[call], require=True)
    assert "--no-net" not in cmd and "--require-sandbox" in cmd
    assert cmd[cmd.index("--") + 1:] == [str(vdir / ".deps/bin/python"), "-E", "-s", "-B", "main.py"]


async def test_only_in_a_sandbox_refuses_where_there_is_none(data, monkeypatch):
    await _as_published("u1", "echo")

    async def required():
        return True

    monkeypatch.setattr(xt, "_require_sandbox", required)
    monkeypatch.setattr(xt, "sandbox_info", lambda: {"os": "darwin", "landlock": 0, "files": False,
                                                     "network": False, "signals": False})
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "echo", "u1", {"text": "hi"})
    assert exc.value.kind == "no_sandbox" and exc.value.status == 503


@pytest.mark.skipif(xt.sandbox_info()["files"], reason="this host has a sandbox")
async def test_the_launcher_refuses_too_when_told_the_host_has_one_it_doesnt(data, monkeypatch):
    """Belt and braces: even if Flight Deck believed in a sandbox here, the
    launcher itself refuses to run the tool without one."""
    await _as_published("u1", "echo")

    async def required():
        return True

    monkeypatch.setattr(xt, "_require_sandbox", required)
    monkeypatch.setattr(xt, "sandbox_info", lambda: {"os": "linux", "landlock": 4, "files": True,
                                                     "network": True, "signals": False})
    res = await xt.run_tool("u1", "echo", "u1", {"text": "hi"})
    assert res["ok"] is False and res["exit_code"] == 126 and "sandbox" in res["stderr"]


def test_the_probe_is_asked_once_and_says_what_the_host_offers(monkeypatch):
    monkeypatch.setattr(xt, "_SANDBOX", None)
    info = xt.sandbox_info()
    assert set(info) == {"os", "landlock", "files", "network", "signals"}
    assert info["files"] == (info["landlock"] >= 1)


async def test_the_admin_switch_and_the_status(deck):
    import httpx
    from fastapi import FastAPI

    from captain_claw.flight_deck import admin_routes
    from captain_claw.flight_deck import external_tools_routes as xr
    from captain_claw.flight_deck.auth import get_current_user

    from .test_external_tools import LOOPBACK

    await _switch_on(deck)
    app = FastAPI()
    app.include_router(xr.router)
    app.include_router(admin_routes.router)
    app.dependency_overrides[get_current_user] = lambda: deck.admin
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test") as c:
        status = (await c.get("/fd/external-tools/status")).json()
        assert status["require_sandbox"] is False and "landlock" in status["sandbox"]
        cfg = (await c.put("/fd/admin/config", json={"external_tools_require_sandbox": True})).json()
        assert cfg["external_tools_require_sandbox"] is True
        assert (await c.get("/fd/external-tools/status")).json()["require_sandbox"] is True
    assert await xt._require_sandbox() is True
