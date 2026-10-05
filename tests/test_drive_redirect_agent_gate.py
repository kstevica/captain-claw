"""The Drive-link redirect only points an agent at google_drive it can call.

web_fetch / web_get / web_fetch_batch and shell curl/wget refuse a Drive file
link (and name the google_drive call) when google_drive can open it. That is
only true for the CALLING agent when its model actually has the tool: a sister
session never registers google_drive, and nano mode cuts it from the tool list.
For those agents the anonymous fetch must run as before, or a link-shared Doc
would be stranded behind a tool they don't have.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import captain_claw.tools.google_drive as gd
from captain_claw.tools.google_drive import agent_offers_google_drive
from captain_claw.tools.shell import ShellTool
from captain_claw.tools.web_fetch import WebFetchTool

FID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
DOC_URL = f"https://docs.google.com/document/d/{FID}/edit?usp=sharing"


class _Tools:
    def __init__(self, names: set[str]):
        self._names = names

    def has_tool(self, name: str) -> bool:
        return name in self._names


def _agent(*, google_drive: bool = True, nano: bool = False) -> SimpleNamespace:
    names = {"google_drive", "web_fetch"} if google_drive else {"web_fetch"}
    return SimpleNamespace(
        tools=_Tools(names), instructions=SimpleNamespace(use_nano=nano),
    )


class _BrokenTools:
    def has_tool(self, name: str) -> bool:
        raise RuntimeError("registry gone")


def test_agent_offers_google_drive():
    assert agent_offers_google_drive(None) is True  # direct call: keep the redirect
    assert agent_offers_google_drive(_agent()) is True
    assert agent_offers_google_drive(_agent(google_drive=False)) is False
    assert agent_offers_google_drive(_agent(nano=True)) is False
    # No instructions object (a bare agent): only registration decides.
    assert agent_offers_google_drive(SimpleNamespace(tools=_Tools({"google_drive"}))) is True
    # A registry that can't answer keeps the redirect rather than guessing no.
    assert agent_offers_google_drive(SimpleNamespace(tools=_BrokenTools())) is True


@pytest.fixture
def drive_links_readable(monkeypatch):
    async def _yes():
        return True

    monkeypatch.setattr(gd, "google_drive_reads_links", _yes)


class _FakeResponse:
    status_code = 200
    headers = {"content-type": "text/html"}
    text = "<html><body><p>" + "Public doc text. " * 60 + "</p></body></html>"

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.urls.append(url)
        return _FakeResponse()


@pytest.mark.parametrize(
    "agent, blocked",
    [
        (None, True),
        (_agent(), True),
        (_agent(google_drive=False), False),  # sister session
        (_agent(nano=True), False),  # nano mode
    ],
)
async def test_web_fetch_redirects_only_agents_with_google_drive(
    drive_links_readable, agent, blocked,
):
    tool = WebFetchTool()
    tool.client = _FakeClient()
    kwargs = {"_agent": agent} if agent is not None else {}
    res = await tool.execute(url=DOC_URL, deep_fetch=False, **kwargs)
    if blocked:
        assert not res.success
        assert f"file_id='{FID}'" in res.error
        assert tool.client.urls == []
    else:
        assert res.success, res.error
        assert tool.client.urls == [DOC_URL]


@pytest.mark.parametrize(
    "agent, readable",
    [
        (None, True),
        (_agent(), True),
        (_agent(google_drive=False), False),
        (_agent(nano=True), False),
    ],
)
async def test_shell_drive_rule_follows_the_calling_agent(
    monkeypatch, drive_links_readable, agent, readable,
):
    seen: dict[str, object] = {}

    def _capture(self, command, *, drive_links_readable=None):
        seen["readable"] = drive_links_readable
        return False, "stop here"  # never actually run curl

    monkeypatch.setattr(ShellTool, "_is_command_safe", _capture)
    kwargs = {"_agent": agent} if agent is not None else {}
    res = await ShellTool().execute(f"curl -sL '{DOC_URL}' -o /dev/null", **kwargs)
    assert not res.success
    assert seen["readable"] is readable
