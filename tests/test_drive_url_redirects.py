"""Drive/Docs URLs steer to google_drive — never to the retired gws CLI.

browser refuses Drive file URLs (Forms and published pages excepted);
web_fetch (and web_get / web_fetch_batch) and shell curl/wget refuse a Drive
file URL only while google_drive can open it — Google connected with a scope
that reaches shared links — so public links still fetch otherwise. Every
refusal prints the google_drive call with the id from the URL. The shell
also refuses to run the gws binary at all.
"""

from __future__ import annotations

import os
import time

import pytest

import captain_claw.google_oauth_manager as gom
import captain_claw.session as session_mod
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.tools import shell as shell_mod
from captain_claw.tools.browser import BrowserTool
from captain_claw.tools.google_drive import google_drive_reads_links
from captain_claw.tools.shell import ShellTool, _drive_download_url, _runs_gws
from captain_claw.tools.web_fetch import WebFetchBatchTool, WebFetchTool, WebGetTool

FID = "1AbCdEfGhIjKlMnOpQrStUvWxYz_0123-456789"
DOC_URL = f"https://docs.google.com/document/d/{FID}/edit?usp=sharing"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FID}"
PUBLISHED_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQabcDEF123/pub?output=csv"
PUBLISHED_DOC = "https://docs.google.com/document/d/e/2PACX-1vQabcDEF123/pub"
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSf1234567890abc/viewform"

_DRIVE = "https://www.googleapis.com/auth/drive"
_DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"


def _grant(monkeypatch, *, connected: bool = True, scope: str | None = _DRIVE) -> None:
    """Google connected (or not), with the token's granted *scope*."""
    monkeypatch.setattr(gom, "is_google_connected_cached", lambda *a, **k: connected)
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())

    async def _get_tokens(self):
        if scope is None:
            return None
        return GoogleOAuthTokens(
            access_token="T", refresh_token="", expires_at=time.time() + 3300, scope=scope,
        )

    monkeypatch.setattr(gom.GoogleOAuthManager, "get_tokens", _get_tokens)


@pytest.fixture
def connected(monkeypatch):
    _grant(monkeypatch)


@pytest.fixture
def disconnected(monkeypatch):
    _grant(monkeypatch, connected=False)


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": "text/html"}

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.urls: list[str] = []

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.urls.append(url)
        return _FakeResponse(self.text)


_PAGE = "<html><body><h1>Public doc</h1><p>" + "Readable text. " * 60 + "</p></body></html>"


def _assert_points_at_google_drive(msg: str, *, folder: bool = False) -> None:
    assert "gws" not in msg
    if folder:
        assert f"google_drive(action='list', folder_id='{FID}')" in msg
    else:
        assert f"google_drive(action='read', file_id='{FID}')" in msg
        assert f"google_drive(action='download', file_id='{FID}')" in msg


# ── browser: always blocks, new wording ──────────────────────────────


async def test_browser_navigate_and_open_redirect_to_google_drive():
    tool = BrowserTool()
    res = await tool._navigate(url=DOC_URL)
    assert not res.success
    assert res.error.startswith("Cannot open Google Drive/Docs URLs in the browser")
    _assert_points_at_google_drive(res.error)

    res = await tool._open(url=FOLDER_URL)
    assert not res.success
    _assert_points_at_google_drive(res.error, folder=True)


def test_browser_storage_googleapis_is_not_drive():
    assert not BrowserTool._is_google_drive_url(f"https://storage.googleapis.com/b/{FID}")
    assert BrowserTool._is_google_drive_url(DOC_URL)


def test_browser_opens_forms_and_published_pages():
    """google_drive can't read these, and they open without signing in."""
    for url in (FORM_URL, PUBLISHED_DOC, PUBLISHED_CSV):
        assert not BrowserTool._blocks_drive_url(url), url
    assert BrowserTool._blocks_drive_url(DOC_URL)
    assert BrowserTool._blocks_drive_url(FOLDER_URL)


async def test_browser_refusal_while_connected_does_not_say_connect_google(connected):
    res = await BrowserTool()._navigate(url=DOC_URL)
    assert not res.success
    assert "Google is connected" in res.error


# ── web_fetch / web_get / web_fetch_batch: block only when connected ─


async def test_web_fetch_blocks_drive_url_when_connected(connected):
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=DOC_URL, deep_fetch=False)
    assert not res.success
    _assert_points_at_google_drive(res.error)
    assert tool.client.urls == []


async def test_web_fetch_fetches_drive_url_when_not_connected(disconnected):
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=DOC_URL, deep_fetch=False)
    assert res.success, res.error
    assert "Public doc" in res.content
    assert tool.client.urls == [DOC_URL]


async def test_web_fetch_non_drive_url_unaffected(connected):
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=f"https://storage.googleapis.com/b/{FID}", deep_fetch=False)
    assert res.success, res.error


async def test_web_get_blocks_only_when_connected(monkeypatch):
    tool = WebGetTool()
    tool.client = _FakeClient(_PAGE)
    _grant(monkeypatch)
    res = await tool.execute(url=FOLDER_URL)
    assert not res.success
    _assert_points_at_google_drive(res.error, folder=True)

    _grant(monkeypatch, connected=False)
    res = await tool.execute(url=FOLDER_URL)
    assert res.success, res.error


# Pages google_drive cannot read: published copies (/d/e/<key> is no file
# id) and Forms. They are public by design, so they fetch even when connected.
@pytest.mark.parametrize("url", [PUBLISHED_CSV, PUBLISHED_DOC, FORM_URL,
                                 "https://drive.google.com/drive/my-drive"])
async def test_web_fetch_reads_pages_google_drive_cannot_open(connected, url):
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=url, deep_fetch=False)
    assert res.success, res.error
    assert tool.client.urls == [url]

    getter = WebGetTool()
    getter.client = _FakeClient(_PAGE)
    assert (await getter.execute(url=url)).success


async def test_web_fetch_with_drive_file_scope_only_fetches_anonymously(monkeypatch):
    """drive.file sees only files this app created: a pasted link would 404
    in google_drive, so the anonymous fetch gets its chance."""
    _grant(monkeypatch, scope=f"openid email {_DRIVE_FILE}")
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=DOC_URL, deep_fetch=False)
    assert res.success, res.error
    assert tool.client.urls == [DOC_URL]


@pytest.mark.parametrize(
    "connected,scope,expected",
    [
        (True, _DRIVE, True),
        (True, "https://www.googleapis.com/auth/drive.readonly", True),
        (True, "", True),  # unreported scope: google_drive doesn't refuse it either
        (True, f"openid {_DRIVE_FILE}", False),
        (True, "https://www.googleapis.com/auth/gmail.readonly", False),
        (True, None, False),  # no token after all
        (False, _DRIVE, False),
    ],
)
async def test_google_drive_reads_links(monkeypatch, connected, scope, expected):
    _grant(monkeypatch, connected=connected, scope=scope)
    assert await google_drive_reads_links() is expected


async def test_google_drive_reads_links_false_when_flight_deck_refuses(monkeypatch):
    _grant(monkeypatch)

    async def _refused(self):
        raise gom.FlightDeckRefused(403, "agent not spawned by this deck")

    monkeypatch.setattr(gom.GoogleOAuthManager, "get_tokens", _refused)
    assert await google_drive_reads_links() is False


async def test_refusal_while_connected_does_not_say_connect_google(connected):
    tool = WebFetchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(url=DOC_URL, deep_fetch=False)
    assert "Google is connected" in res.error
    assert "not enabled for this agent" in res.error
    assert "ask the user to connect Google" not in res.error


@pytest.mark.parametrize("deep_fetch", [None, True])
async def test_web_fetch_batch_blocks_drive_in_both_phases(connected, monkeypatch, deep_fetch):
    tool = WebFetchBatchTool()
    tool.client = _FakeClient(_PAGE)
    deep_seen: list[str] = []

    async def _fake_deep(self, deep_urls, outcomes, deep_conc):
        deep_seen.extend(deep_urls)
        for u in deep_urls:
            outcomes[u].ok, outcomes[u].mode, outcomes[u].content = True, "deep", _PAGE
        return True

    monkeypatch.setattr(WebFetchBatchTool, "_deep_phase", _fake_deep)
    res = await tool.execute(urls=[DOC_URL, "https://example.com/a"], deep_fetch=deep_fetch)
    assert res.success
    assert DOC_URL not in tool.client.urls
    assert DOC_URL not in deep_seen
    assert f"google_drive(action='read', file_id='{FID}')" in res.content
    assert "gws" not in res.content


async def test_web_fetch_batch_fetches_drive_when_not_connected(disconnected):
    tool = WebFetchBatchTool()
    tool.client = _FakeClient(_PAGE)
    res = await tool.execute(urls=[DOC_URL])
    assert res.success
    assert tool.client.urls == [DOC_URL]
    assert "google_drive(" not in res.content


# ── shell: curl/wget redirect (connected only) ───────────────────────


def test_shell_curl_drive_blocked_when_connected(connected):
    ok, reason = ShellTool()._is_command_safe(f"curl -L '{DOC_URL}' -o plan.html")
    assert not ok
    assert reason.startswith("Do not use curl/wget")
    _assert_points_at_google_drive(reason)

    ok, reason = ShellTool()._is_command_safe(f"wget {FOLDER_URL}")
    assert not ok
    _assert_points_at_google_drive(reason, folder=True)


def test_shell_curl_drive_allowed_when_not_connected(disconnected):
    ok, reason = ShellTool()._is_command_safe(f"curl -L '{DOC_URL}' -o plan.html")
    assert ok, reason


def test_shell_curl_cloud_storage_never_treated_as_drive(connected):
    ok, reason = ShellTool()._is_command_safe(
        f"curl -O https://storage.googleapis.com/bucket/{FID}.tar.gz"
    )
    assert ok, reason


def test_shell_curl_mentioning_drive_host_without_a_drive_url_runs(connected):
    ok, reason = ShellTool()._is_command_safe(
        "curl 'https://example.com/track?ref=docs.google.com'"
    )
    assert ok, reason


@pytest.mark.parametrize(
    "command",
    [
        f"curl -sL '{DOC_URL}' -o plan.html",
        f"curl -L \\\n  '{DOC_URL}' \\\n  -o plan.html",  # line continuations
        f"sudo curl {DOC_URL}",
        f"echo hi && curl -sSLo out.pdf {DOC_URL}",
        f"timeout 30 wget -q {DOC_URL}",
        f"wget -d {DOC_URL}",  # wget's -d is a flag, not a value option
        f"curl --url {DOC_URL}",
        f"echo $(curl -s {DOC_URL})",
        f"bash -c 'curl {DOC_URL}'",
        f"echo 'a\nb'; curl {DOC_URL}",
        f"curl 'https://drive.google.com/uc?id={FID}&export=download' -o out.pdf",
    ],
)
def test_shell_drive_downloads_are_found(connected, command):
    assert _drive_download_url(command)
    ok, reason = ShellTool()._is_command_safe(command)
    assert not ok
    assert reason.startswith("Do not use curl/wget")


@pytest.mark.parametrize(
    "command",
    [
        # A Drive link in a curl/wget payload, header or referer is no download.
        f"curl -X POST https://hooks.slack.com/services/T/B/x -d '{{\"text\":\"Report ready: {DOC_URL}\"}}'",
        f"curl -s https://api.telegram.org/botX/sendMessage -d text='{DOC_URL}'",
        f"curl -s https://hooks.example.com -d text={DOC_URL}",
        f"curl --json '{{\"url\": \"{DOC_URL}\"}}' https://api.example.com/x",
        f"curl -sH 'Referer: {DOC_URL}' https://example.com",
        f"curl -e {DOC_URL} https://example.com",
        f"wget --post-data 'link={DOC_URL}' https://example.com/hook",
        # ... nor one elsewhere in the command.
        f"curl https://example.com/a.html > a.html; echo '{DOC_URL}' >> sources.md",
        f"git commit -m 'curl fix; see {DOC_URL}'",
        f"pip install pycurl && echo {DOC_URL} > links.txt",
        # Pages google_drive cannot open fetch as before.
        f"curl -sL '{PUBLISHED_CSV}' -o data.csv",
        f"wget '{FORM_URL}'",
    ],
)
def test_shell_curl_that_downloads_no_drive_file_runs(connected, command):
    assert _drive_download_url(command) is None
    ok, reason = ShellTool()._is_command_safe(command)
    assert ok, reason


def test_shell_curl_drive_allowed_with_drive_file_scope_only(connected):
    ok, reason = ShellTool()._is_command_safe(
        f"curl -L '{DOC_URL}' -o plan.html", drive_links_readable=False,
    )
    assert ok, reason


async def test_shell_execute_checks_the_connection_scope(monkeypatch):
    """execute() asks google_drive_reads_links, not just the cached flag."""
    _grant(monkeypatch, connected=False)
    import captain_claw.tools.google_drive as gd

    async def _reads_links():
        return True

    monkeypatch.setattr(gd, "google_drive_reads_links", _reads_links)
    res = await ShellTool().execute(f"curl -sL '{DOC_URL}' -o /dev/null")
    assert not res.success
    assert "Do not use curl/wget" in res.error
    assert f"file_id='{FID}'" in res.error


# ── shell: the gws binary never runs ─────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "gws drive files list",
        "  gws --help",
        "/opt/homebrew/bin/gws auth login",
        "./bin/gws drive files list",
        "ls && gws drive files list",
        "cat ids.txt | gws drive files get",
        "echo $(gws drive files list)",
        "echo `gws auth status`",
        "FOO=1 gws drive",
        "env GOOGLE_WORKSPACE_CLI_TOKEN=x gws drive",
        "sudo -u bob gws drive",
        "nohup gws drive &",
        "time gws",
        "timeout 30 gws drive",
        "bash -c 'gws drive files list'",
        "sh -lc \"cd /tmp && gws x\"",
        "eval 'gws drive'",
        "if true; then gws x; fi",
        "echo start\ngws drive files list",
        "bash <<EOF\ngws drive\nEOF",
        "bash -c 'cd /tmp\ngws x'",
        'eval "ls\ngws x"',
        'echo "a\nb"; gws x',
        'echo "a\nb"\ngws x',
        "ls # don't\ngws x",
        "gws drive 'unclosed",
    ],
)
def test_shell_refuses_running_gws(command):
    assert _runs_gws(command)
    ok, reason = ShellTool()._is_command_safe(command)
    assert not ok
    assert reason == (
        "The Google Workspace CLI (gws) is retired here — use google_drive / "
        "google_calendar / google_mail instead."
    )


@pytest.mark.parametrize(
    "command",
    [
        "cat gws.txt",
        "grep gws file",
        "grep -r 'gws' .",
        "ls ~/gws-backup",
        "echo gws",
        "which gws",
        "command -v gws",
        "brew uninstall gws",
        "python3 gws_script.py",
        "cd gws && ls",
        "mv gws gws.old",
        "git commit -m 'drop gws'",
        "echo \"a; gws\"",
        "echo $(date) gws",
        "rm -rf ~/.config/gws",
        "cat <<EOF > notes.md\ngws drive files list\nEOF",
        "# gws drive",
        # A quoted string spanning lines is one argument, whatever its lines say.
        'git commit -m "Retire the Workspace CLI\ngws is gone; use google_drive"',
        "git commit -m 'Retire gws\n\ngws is gone; google_drive replaces it'",
        'python3 -c "\nimport os\ngws = 1\nprint(gws)\n"',
        'echo "first line\ngws second line"',
        'gh pr create --body "| tool | state |\n| gws | removed |"',
        "echo 'it''s'\nls",
    ],
)
def test_shell_gws_as_argument_is_not_refused(command):
    assert not _runs_gws(command)


def test_shell_multiline_commit_message_mentioning_gws_runs():
    ok, reason = ShellTool()._is_command_safe(
        'git commit -m "Retire the Workspace CLI\n\ngws tool removed; google_drive replaces it"'
    )
    assert ok, reason


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
async def test_scripts_that_shell_out_to_gws_hit_the_stub(tmp_path):
    """A script the command check can't see into still can't reach gws."""
    script = tmp_path / "s.py"
    script.write_text(
        "import subprocess\n"
        "r = subprocess.run(['gws', 'drive'], capture_output=True, text=True)\n"
        "print(r.returncode, r.stderr.strip())\n"
    )
    res = await ShellTool().execute(f"python3 {script}")
    assert res.success, res.error
    assert res.content.startswith("127 The Google Workspace CLI (gws) is retired here")
    stub_dir = shell_mod._gws_shim_dir()
    assert stub_dir and oct(os.stat(stub_dir).st_mode & 0o777) == "0o700"
