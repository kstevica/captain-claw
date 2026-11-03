"""HTML → PDF: WeasyPrint where it loads, Chromium where it can't (a Mac without
Homebrew's Pango, and the desktop app's own Python), and what Chromium may fetch."""

from __future__ import annotations

import pytest

from captain_claw import html_pdf


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(html_pdf, "_weasyprint_error", None)


async def test_weasyprint_when_it_loads(monkeypatch):
    class FakeHTML:
        def __init__(self, string):
            self.string = string

        def write_pdf(self):
            return b"%PDF weasy " + self.string.encode()

    async def no_chromium(html):
        raise AssertionError("Chromium shouldn't be used when WeasyPrint loads")

    monkeypatch.setattr(html_pdf, "_weasyprint_html", lambda: FakeHTML)
    monkeypatch.setattr(html_pdf, "_chromium_pdf", no_chromium)
    assert await html_pdf.html_to_pdf("<p>x</p>") == b"%PDF weasy <p>x</p>"


async def test_chromium_when_weasyprint_cant_load(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pango(name, *args, **kwargs):
        if name == "weasyprint":
            raise OSError("cannot load library 'libpango-1.0-0'")
        return real_import(name, *args, **kwargs)

    async def chromium(html):
        return b"%PDF chromium"

    monkeypatch.setattr(builtins, "__import__", no_pango)
    monkeypatch.setattr(html_pdf, "_chromium_pdf", chromium)
    assert await html_pdf.html_to_pdf("<p>x</p>") == b"%PDF chromium"
    assert "libpango" in html_pdf._weasyprint_error
    # Tried once per process: the second PDF goes straight to Chromium.
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert await html_pdf.html_to_pdf("<p>y</p>") == b"%PDF chromium"


@pytest.mark.parametrize(
    "host, local",
    [
        ("localhost", True),
        ("app.localhost", True),
        ("127.0.0.1", True),
        ("[::1]", True),
        ("10.0.0.5", True),
        ("192.168.178.88", True),
        ("169.254.169.254", True),
        ("0.0.0.0", True),
        ("", True),
        ("example.com", False),
        ("fonts.googleapis.com", False),
        ("8.8.8.8", False),
    ],
)
def test_what_chromium_may_not_fetch(host, local):
    assert html_pdf._local_host(host) is local
