"""Drive ids from pasted Drive / Docs / Sheets / Slides URLs."""

from __future__ import annotations

import pytest

from captain_claw.google_ids import (
    drive_id_from_url,
    drive_resource_key,
    google_drive_can_open,
    google_drive_redirect,
    is_drive_folder_url,
    is_google_drive_url,
    is_public_google_page,
)

FID = "1AbCdEfGhIjKlMnOpQrStUvWxYz_0123-456789"


@pytest.mark.parametrize(
    "url",
    [
        f"https://docs.google.com/document/d/{FID}/edit?usp=sharing",
        f"https://docs.google.com/document/d/{FID}/edit#heading=h.abc",
        f"https://docs.google.com/document/u/1/d/{FID}/edit",
        f"https://docs.google.com/spreadsheets/d/{FID}/edit#gid=0",
        f"https://docs.google.com/presentation/d/{FID}/edit#slide=id.p",
        f"https://docs.google.com/forms/d/{FID}/edit",
        f"https://docs.google.com/drawings/d/{FID}/edit",
        f"https://docs.google.com/document/d/{FID}",
        f"https://drive.google.com/file/d/{FID}/view?usp=drive_link",
        f"https://drive.google.com/open?id={FID}",
        f"https://drive.google.com/uc?id={FID}&export=download",
        f"https://drive.google.com/uc?export=download&id={FID}",
        f"https://drive.usercontent.google.com/download?id={FID}&export=download",
        f"https://drive.google.com/drive/folders/{FID}",
        f"https://drive.google.com/drive/folders/{FID}?usp=sharing",
        f"https://drive.google.com/drive/u/0/folders/{FID}",
        f"docs.google.com/document/d/{FID}/edit",  # no scheme
        f"<https://docs.google.com/document/d/{FID}/edit>",
    ],
)
def test_every_url_shape_yields_the_id(url):
    assert drive_id_from_url(url) == FID


def test_url_inside_free_text():
    text = f"please summarise https://docs.google.com/document/d/{FID}/edit, thanks"
    assert drive_id_from_url(text) == FID
    cmd = f"curl -L 'https://drive.google.com/uc?id={FID}&export=download' -o out.pdf"
    assert drive_id_from_url(cmd) == FID
    assert drive_id_from_url(f"wget docs.google.com/document/d/{FID}/export") == FID


@pytest.mark.parametrize("bare", [FID, "0AFoo1234567Uk9PVA", "1AbCdEfGhIj"])
def test_bare_id_passes_through(bare):
    assert drive_id_from_url(bare) == bare


@pytest.mark.parametrize(
    "text",
    [
        "",
        "root",
        "documentation",  # a word, not an id
        "my-report-final",
        "1234567890",
        "/tmp/foo/bar.txt",
        "report.pdf",
        "hello world 1AbC",
        f"https://storage.googleapis.com/bucket/{FID}",  # Cloud Storage
        f"https://example.com/document/d/{FID}/edit",
        "https://docs.google.com/forms/d/e/1FAIpQLSf1234567890abc/viewform",  # published key
        "https://drive.google.com/drive/my-drive",
    ],
)
def test_non_ids_return_none(text):
    assert drive_id_from_url(text) is None


@pytest.mark.parametrize(
    "url,expected",
    [
        (f"https://docs.google.com/document/d/{FID}/edit", True),
        (f"https://drive.google.com/file/d/{FID}/view", True),
        ("https://sheets.google.com/", True),
        (f"docs.google.com/document/d/{FID}", True),
        (f"https://storage.googleapis.com/b/{FID}", False),
        ("https://docs.google.com.evil.example/x", False),
        ("https://evil.example/docs.google.com/x", False),
        ("https://www.google.com/search?q=drive", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_google_drive_url(url, expected):
    assert is_google_drive_url(url) is expected


def test_is_drive_folder_url():
    assert is_drive_folder_url(f"https://drive.google.com/drive/folders/{FID}")
    assert is_drive_folder_url(f"https://drive.google.com/drive/u/3/folders/{FID}?usp=sharing")
    assert is_drive_folder_url(f"https://drive.google.com/embeddedfolderview?id={FID}")
    assert not is_drive_folder_url(f"https://drive.google.com/file/d/{FID}/view")
    assert not is_drive_folder_url(f"https://docs.google.com/document/d/{FID}/edit")


def test_redirect_prints_the_exact_call():
    msg = google_drive_redirect(f"https://docs.google.com/document/d/{FID}/edit", "Blocked.")
    assert msg.startswith("Blocked.")
    assert f"google_drive(action='read', file_id='{FID}')" in msg
    assert f"google_drive(action='info', file_id='{FID}')" in msg
    assert f"google_drive(action='download', file_id='{FID}')" in msg
    assert "gws" not in msg

    folder = google_drive_redirect(f"https://drive.google.com/drive/folders/{FID}", "Blocked.")
    assert f"google_drive(action='list', folder_id='{FID}')" in folder
    assert "action='read'" not in folder

    generic = google_drive_redirect("https://drive.google.com/drive/my-drive", "Blocked.")
    assert "file_id='<id>'" in generic and "folder_id='<id>'" in generic


@pytest.mark.parametrize(
    "url,can_open,public",
    [
        (f"https://docs.google.com/document/d/{FID}/edit", True, False),
        (f"https://docs.google.com/spreadsheets/d/{FID}/export?format=csv", True, False),
        (f"https://drive.google.com/drive/folders/{FID}", True, False),
        (f"https://drive.google.com/uc?id={FID}&export=download", True, False),
        (f"docs.google.com/document/d/{FID}/edit", True, False),  # no scheme
        # Published copies: the /d/e/ key is no file id.
        ("https://docs.google.com/spreadsheets/d/e/2PACX-1vQabc123/pub?output=csv", False, True),
        ("https://docs.google.com/document/d/e/2PACX-1vQabc123/pub", False, True),
        ("https://docs.google.com/presentation/d/e/2PACX-1vQabc123/pub", False, True),
        # Forms have no content google_drive can export, even with an id.
        ("https://docs.google.com/forms/d/e/1FAIpQLSf1234567890abc/viewform", False, True),
        (f"https://docs.google.com/forms/d/{FID}/edit", False, True),
        # Drive, but nothing to pass as an id.
        ("https://drive.google.com/drive/my-drive", False, False),
        ("https://docs.google.com/", False, False),
        (f"https://storage.googleapis.com/bucket/{FID}", False, False),
        (f"https://example.com/forms/d/e/{FID}/viewform", False, False),
    ],
)
def test_google_drive_can_open_and_public_pages(url, can_open, public):
    assert google_drive_can_open(url) is can_open
    assert is_public_google_page(url) is public


def test_resource_key_from_url():
    url = f"https://drive.google.com/file/d/{FID}/view?resourcekey=0-AbC_dEf-123"
    assert drive_resource_key(url) == "0-AbC_dEf-123"
    assert drive_resource_key(f"see {url} please") == "0-AbC_dEf-123"
    assert drive_resource_key(f"https://drive.google.com/file/d/{FID}/view") is None
    assert drive_resource_key(FID) is None
    assert drive_resource_key("https://example.com/?resourcekey=0-abc") is None


def test_redirect_keeps_the_resource_key():
    """An old link-shared file needs its key: the call gets the whole URL."""
    url = f"https://drive.google.com/file/d/{FID}/view?resourcekey=0-AbC_dEf-123"
    msg = google_drive_redirect(url, "Blocked.")
    assert f"google_drive(action='read', file_id='{url}')" in msg
    assert drive_id_from_url(url) == FID


def test_redirect_tail_depends_on_the_connection():
    url = f"https://docs.google.com/document/d/{FID}/edit"
    assert "ask the user to connect Google" in google_drive_redirect(url, "Blocked.")
    connected = google_drive_redirect(url, "Blocked.", connected=True)
    assert "Google is connected" in connected
    assert "not enabled for this agent" in connected
    assert "ask the user to connect Google" not in connected
