"""DriveClient — pagination, retry/backoff, scope acceptance, query escaping.

Pure unit tests against a faked httpx layer. No network, no Google, no tokens.
These cover the exact gaps that made the old google_drive tool unfit for a
filesystem mount: it discarded nextPageToken, never retried a rate limit, and
demanded the full read/write scope.
"""

import httpx
import pytest

from captain_claw.drive_client import (
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotConnected,
    escape_query_value,
)


def _resp(status: int, json_body=None, headers=None, content=b"") -> httpx.Response:
    req = httpx.Request("GET", "https://example.test")
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers or {}, request=req)
    return httpx.Response(status, content=content, headers=headers or {}, request=req)


class _FakeHTTP:
    """Stands in for httpx.AsyncClient — replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.is_closed = False

    async def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(responses, *, scope="https://www.googleapis.com/auth/drive.readonly"):
    async def provider():
        return ("tok", scope)

    c = DriveClient(provider, max_retries=3)
    fake = _FakeHTTP(responses)
    c._client = fake  # type: ignore[assignment]
    c._http = lambda: fake  # type: ignore[method-assign]
    return c, fake


async def _nosleep(_):
    return None


class TestEscaping:
    def test_apostrophe_cannot_break_out_of_the_literal(self):
        assert escape_query_value("a'b") == "a\\'b"

    def test_backslash_is_escaped_first(self):
        assert escape_query_value("a\\b") == "a\\\\b"

    @pytest.mark.asyncio
    async def test_folder_id_is_escaped_in_the_query(self):
        c, fake = _client([_resp(200, {"files": []})])
        await c.list_folder("evil') or ('1'='1", sleep=_nosleep)
        assert "\\'" in fake.calls[0]["params"]["q"]


class TestPagination:
    @pytest.mark.asyncio
    async def test_follows_next_page_token(self):
        c, fake = _client([
            _resp(200, {"files": [{"id": "1", "name": "a", "mimeType": "text/plain"}],
                        "nextPageToken": "P2"}),
            _resp(200, {"files": [{"id": "2", "name": "b", "mimeType": "text/plain"}]}),
        ])
        files, truncated = await c.list_folder("root", sleep=_nosleep)
        assert [f.id for f in files] == ["1", "2"]
        assert truncated is False
        assert fake.calls[1]["params"]["pageToken"] == "P2"

    @pytest.mark.asyncio
    async def test_max_files_truncates_and_flags(self):
        c, fake = _client([
            _resp(200, {"files": [{"id": str(i), "name": str(i), "mimeType": "text/plain"}
                                  for i in range(3)], "nextPageToken": "MORE"}),
        ])
        files, truncated = await c.list_folder("root", max_files=3, sleep=_nosleep)
        assert len(files) == 3
        assert truncated is True  # there was a next page we chose not to fetch


class TestRetry:
    @pytest.mark.asyncio
    async def test_429_is_retried_then_succeeds(self):
        slept: list[float] = []

        async def sleep(d):
            slept.append(d)

        c, fake = _client([
            _resp(429, headers={"Retry-After": "2"}, json_body={"error": {}}),
            _resp(200, {"files": []}),
        ])
        files, _ = await c.list_folder("root", sleep=sleep)
        assert files == []
        assert slept == [2.0]  # honoured Retry-After
        assert len(fake.calls) == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_without_retry_after(self):
        slept: list[float] = []

        async def sleep(d):
            slept.append(d)

        c, fake = _client([
            _resp(503, json_body={"error": {}}),
            _resp(503, json_body={"error": {}}),
            _resp(200, {"files": []}),
        ])
        await c.list_folder("root", sleep=sleep)
        assert slept == [1, 2]  # 2**0, 2**1

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        c, fake = _client([_resp(503, json_body={"error": {}})] * 5)
        with pytest.raises(DriveError):
            await c.list_folder("root", sleep=_nosleep)

    @pytest.mark.asyncio
    async def test_404_is_not_retried(self):
        c, fake = _client([_resp(404, {"error": {"message": "gone"}}),
                           _resp(200, {"files": []})])
        with pytest.raises(DriveError):
            await c.get_metadata("missing", sleep=_nosleep)
        assert len(fake.calls) == 1  # no retry


class TestScope:
    @pytest.mark.asyncio
    async def test_readonly_scope_is_accepted(self):
        c, _ = _client([_resp(200, {"files": []})],
                       scope="https://www.googleapis.com/auth/drive.readonly")
        await c.list_folder("root", sleep=_nosleep)  # must not raise

    @pytest.mark.asyncio
    async def test_drive_file_scope_is_accepted(self):
        c, _ = _client([_resp(200, {"files": []})],
                       scope="openid https://www.googleapis.com/auth/drive.file")
        await c.list_folder("root", sleep=_nosleep)

    @pytest.mark.asyncio
    async def test_no_drive_scope_is_rejected(self):
        c, _ = _client([_resp(200, {"files": []})],
                       scope="openid email https://www.googleapis.com/auth/gmail.readonly")
        with pytest.raises(DriveNotConnected):
            await c.list_folder("root", sleep=_nosleep)

    @pytest.mark.asyncio
    async def test_empty_token_is_not_connected(self):
        async def provider():
            return ("", "")

        c = DriveClient(provider)
        c._http = lambda: _FakeHTTP([])  # type: ignore[method-assign]
        with pytest.raises(DriveNotConnected):
            await c.list_folder("root", sleep=_nosleep)


class TestFetchRouting:
    @pytest.mark.asyncio
    async def test_google_sheet_exports_to_xlsx(self):
        c, fake = _client([_resp(200, content=b"xlsxbytes")])
        f = DriveFile(id="s1", name="Budget", mime_type="application/vnd.google-apps.spreadsheet")
        data, ext = await c.fetch(f, sleep=_nosleep)
        assert data == b"xlsxbytes"
        assert ext == ".xlsx"
        assert "/export" in fake.calls[0]["url"]
        assert "spreadsheetml" in fake.calls[0]["params"]["mimeType"]

    @pytest.mark.asyncio
    async def test_binary_file_downloads_as_media(self):
        c, fake = _client([_resp(200, content=b"%PDF-1.7")])
        f = DriveFile(id="p1", name="report.PDF", mime_type="application/pdf")
        data, ext = await c.fetch(f, sleep=_nosleep)
        assert data == b"%PDF-1.7"
        assert ext == ".pdf"  # lowercased
        assert fake.calls[0]["params"]["alt"] == "media"


class TestDriveFile:
    def test_size_of_google_native_is_none_not_zero(self):
        f = DriveFile.from_api({"id": "x", "name": "Doc",
                                "mimeType": "application/vnd.google-apps.document"})
        assert f.size is None
        assert f.is_google_native and not f.is_folder

    def test_folder_detection(self):
        f = DriveFile.from_api({"id": "d", "name": "F",
                                "mimeType": "application/vnd.google-apps.folder"})
        assert f.is_folder
