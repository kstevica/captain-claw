"""Chromium on a bare host: one shared cache, and a plain answer when the
host lacks the system libraries Chromium needs (cohort001, 2026-11)."""

from __future__ import annotations

import sys

import pytest

from captain_claw.tools import browser_session as bs

_LIBS_ERROR = (
    "BrowserType.launch: Target page, context or browser has been closed\n"
    "Browser logs:\n<launching> /srv/x/.cache/ms-playwright/chromium_headless_shell-1248/"
    "chrome-headless-shell-linux64/chrome-headless-shell --disable-field-trial-config\n"
    "chrome-headless-shell: error while loading shared libraries: libnspr4.so: "
    "cannot open shared object file: No such file or directory"
)
_MISSING_BINARY = (
    "BrowserType.launch: Executable doesn't exist at /x/chrome-headless-shell\n"
    "Looks like Playwright was just installed or updated. Please run: playwright install"
)


class _FakeType:
    def __init__(self, *errors: str) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def launch(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return "browser"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(bs, "_libs_missing", None)
    monkeypatch.setattr(bs, "_libs_missing_at", 0.0)
    monkeypatch.setattr(bs, "_install_done", False)
    monkeypatch.setattr(bs, "_install_failed_reason", None)


def test_missing_libraries_are_named():
    assert bs.missing_system_libraries(_LIBS_ERROR) == ["libnspr4.so"]
    ldd = "libatk-1.0.so.0 => not found\nlibnss3.so => not found\nHost system is missing dependencies"
    assert bs.missing_system_libraries(ldd) == ["libatk-1.0.so.0", "libnss3.so"]
    assert bs.missing_system_libraries(_MISSING_BINARY) == []
    assert bs.missing_system_libraries("net::ERR_NAME_NOT_RESOLVED") == []


async def test_missing_libraries_raise_a_plain_answer_and_are_remembered():
    fake = _FakeType(_LIBS_ERROR)
    with pytest.raises(bs.BrowserUnavailable) as caught:
        await bs.launch_chromium(fake)
    msg = str(caught.value)
    assert "libnspr4.so" in msg and "install-deps" in msg
    assert "don't try sudo" in msg and "deep_fetch=false" in msg
    assert caught.value.libraries == ["libnspr4.so"]
    # The next call answers at once, without another slow launch.
    with pytest.raises(bs.BrowserUnavailable):
        await bs.launch_chromium(fake)
    assert fake.calls == 1


async def test_the_answer_expires_so_a_fixed_host_works_again(monkeypatch):
    fake = _FakeType(_LIBS_ERROR)
    with pytest.raises(bs.BrowserUnavailable):
        await bs.launch_chromium(fake)
    monkeypatch.setattr(bs, "_libs_missing_at", bs._libs_missing_at - bs._LIBS_RECHECK_SECONDS - 1)
    assert await bs.launch_chromium(fake) == "browser"


async def test_install_deps_hint_is_not_mistaken_for_a_missing_binary(monkeypatch):
    """Playwright's missing-deps text says `playwright install-deps`; that must
    not start a 700 MB download."""
    async def no_install(**_):
        raise AssertionError("must not install")

    monkeypatch.setattr(bs, "ensure_chromium_installed", no_install)
    fake = _FakeType("Host system is missing dependencies to run browsers.\n"
                     "Please install them with: sudo playwright install-deps\n"
                     "Missing libraries:\n    libnss3.so")
    with pytest.raises(bs.BrowserUnavailable):
        await bs.launch_chromium(fake)


async def test_missing_binary_installs_once_then_launches(monkeypatch):
    installs = []

    async def fake_install(timeout_seconds=300):
        installs.append(timeout_seconds)
        return True, "ok"

    monkeypatch.setattr(bs, "ensure_chromium_installed", fake_install)
    fake = _FakeType(_MISSING_BINARY)
    assert await bs.launch_chromium(fake) == "browser"
    assert installs == [300] and fake.calls == 2


async def test_libraries_missing_after_install_still_raise_plainly(monkeypatch):
    async def fake_install(timeout_seconds=300):
        return True, "ok"

    monkeypatch.setattr(bs, "ensure_chromium_installed", fake_install)
    with pytest.raises(bs.BrowserUnavailable):
        await bs.launch_chromium(_FakeType(_MISSING_BINARY, _LIBS_ERROR))


async def test_no_auto_install_leaves_the_binary_alone():
    with pytest.raises(RuntimeError) as caught:
        await bs.launch_chromium(_FakeType(_MISSING_BINARY), auto_install=False)
    assert not isinstance(caught.value, bs.BrowserUnavailable)


def test_host_browsers_path(monkeypatch, tmp_path):
    assert bs.host_browsers_path({"PLAYWRIGHT_BROWSERS_PATH": "0"}) == "0"
    assert bs.host_browsers_path({"PLAYWRIGHT_BROWSERS_PATH": str(tmp_path / "pw")}) == str(tmp_path / "pw")
    monkeypatch.setattr(sys, "platform", "linux")
    assert bs.host_browsers_path({"HOME": "/home/deck"}) == "/home/deck/.cache/ms-playwright"
    assert bs.host_browsers_path({"HOME": "/home/deck", "XDG_CACHE_HOME": "/c"}) == "/c/ms-playwright"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert bs.host_browsers_path({"HOME": "/Users/d"}) == "/Users/d/Library/Caches/ms-playwright"


def test_agents_share_flight_decks_browser_cache(monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", "/home/deck")
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    env = server._agent_base_env()
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == "/home/deck/.cache/ms-playwright"
    # A non-admin's .env can't move it: PLAYWRIGHT_* is pinned to FD's value.
    assert server._pinned_env_name("PLAYWRIGHT_BROWSERS_PATH")


async def test_web_fetch_falls_back_to_plain_http(monkeypatch):
    from captain_claw.tools import web_fetch as wf

    async def broken(url, *a, **k):
        raise bs.BrowserUnavailable(bs.browser_unavailable_message(["libnss3.so"]), ["libnss3.so"])

    class _Resp:
        status_code = 200
        text = "<html><body><p>Plain text of the page.</p></body></html>"
        content = text.encode()
        headers = {"content-type": "text/html"}

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, url):
            return _Resp()

        async def aclose(self):
            return None

    monkeypatch.setattr(wf, "_deep_fetch", broken)
    tool = wf.WebFetchTool()
    tool.client = _Client()
    result = await tool.execute(url="https://example.com/x", deep_fetch=True)
    assert result.success, result.error
    assert "can't start on this server" in result.content
    assert "Plain text of the page." in result.content


def test_batch_note_says_not_to_install(monkeypatch):
    from captain_claw.tools import web_fetch as wf

    monkeypatch.setattr(bs, "_libs_missing", ["libnss3.so"])
    monkeypatch.setattr(bs, "_libs_missing_at", __import__("time").time())
    note = wf._deep_unavailable_note()
    assert "can't start on this server" in note and "pip install" not in note
