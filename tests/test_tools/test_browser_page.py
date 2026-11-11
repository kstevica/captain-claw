"""Captain for Chrome's page actions in the headless browser (browser_page.py).

Pinned here:

* the page scripts shipped with the package are the extension's own, byte
  for byte (change them in browser-extension/chrome/ and copy them over);
* live (skipped without Playwright's Chromium): in the headless browser,
  read_page/find give refs that click, type, form_input, hover, scroll_to and
  upload_file use; page_text reads the page as Markdown; page_files lists its
  files and download keeps them in saved/downloads/; save_page keeps an
  .mhtml; javascript, zoom, drag, back work; a link that opens a new tab
  moves the work there; a ref from before a navigation says it's gone.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw.config import BrowserToolConfig
from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_page
from captain_claw.tools.browser import BrowserTool

ROOT = Path(__file__).resolve().parents[2]


def test_the_page_scripts_are_the_extensions_own():
    for name in ("hands-page.js", "capture.js"):
        shipped = (browser_page.JS_DIR / name).read_bytes()
        extension = (ROOT / "browser-extension" / "chrome" / name).read_bytes()
        assert shipped == extension, f"captain_claw/tools/browser_page_js/{name} differs from the extension's — copy it over"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "tools/browser_page_js/*.js" in pyproject, "the scripts must ship with the package"


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Test Portal</title>
<style>body{font:15px sans-serif;margin:20px}#box{width:60px;height:60px;background:#c44;position:absolute;left:400px;top:420px}
#drop{width:120px;height:120px;border:2px dashed #444;position:absolute;left:700px;top:400px}.tall{height:1400px}</style></head>
<body><h1>Welcome to the portal</h1><p>Your documents are below.</p>
<form id="f" onsubmit="event.preventDefault(); out.textContent = 'Requested for ' + email.value + ' / ' + kind.value + ' / ' + (agree.checked ? 'agreed' : 'not agreed') + ' / ' + note.value">
<label for="email">Email</label> <input id="email" placeholder="you@example.com">
<label for="kind">Statement</label> <select id="kind"><option value="m">Monthly</option><option value="q">Quarterly</option></select>
<label><input type="checkbox" id="agree"> I agree</label>
<textarea id="note" placeholder="Note"></textarea>
<input type="file" id="up" onchange="upname.textContent = this.files[0] ? this.files[0].name + ' ' + this.files[0].size : ''"><span id="upname"></span>
<button type="submit">Request statement</button></form>
<div id="out"></div>
<div id="hov" onmouseover="this.textContent='Hovered!'" style="padding:6px;border:1px solid #888;display:inline-block">Hover me</div>
<ul><li><a href="/files/report.pdf">Annual report (PDF)</a></li><li><a href="/files/data.csv">Data (CSV)</a></li>
<li><a href="/other.html" target="_blank">Open the other page</a></li><li><a href="/other.html">Go to the other page</a></li></ul>
<div id="box"></div><div id="drop"></div><div id="dragout"></div>
<script>
let dragging = false; const box = document.getElementById('box');
box.addEventListener('mousedown', () => { dragging = true });
document.addEventListener('mousemove', (e) => { if (dragging) { box.style.left = (e.clientX - 30) + 'px'; box.style.top = (e.clientY - 30 + scrollY) + 'px' } });
document.addEventListener('mouseup', (e) => { if (dragging) { dragging = false; const d = document.getElementById('drop').getBoundingClientRect();
  document.getElementById('dragout').textContent = (e.clientX > d.left && e.clientX < d.right && e.clientY > d.top && e.clientY < d.bottom) ? 'Dropped in!' : 'Missed' } });
</script><div class="tall"></div><p id="bottom">The very bottom</p></body></html>"""


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


def _ref(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line and "[ref_" in line:
            return line.split("[", 1)[1].split("]", 1)[0]
    raise AssertionError(f"no ref for {needle!r} in:\n{text[:2000]}")


@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium isn't installed")
async def test_live_the_extensions_page_actions_work_headless(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "files").mkdir(parents=True)
    (site / "index.html").write_text(PAGE)
    (site / "other.html").write_text("<!doctype html><title>Other Page</title><button onclick=\"this.textContent='Clicked other'\">Other button</button>")
    (site / "files" / "report.pdf").write_bytes(b"%PDF-1.4\nfake report\n%%EOF\n")
    (site / "files" / "data.csv").write_text("a,b\n1,2\n")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = BrowserToolConfig(default_wait_seconds=0.1, viewport_width=1280, viewport_height=800)
    monkeypatch.setattr(browser_mod, "get_config", lambda: SimpleNamespace(tools=SimpleNamespace(browser=cfg)))
    monkeypatch.setattr(browser_mod, "_TARGETS", {})
    ws = tmp_path / "ws"
    ws.mkdir()
    kw = {"_runtime_base_path": str(ws), "_session_id": f"page-{tmp_path.name}"}
    tool = BrowserTool()
    base = f"http://127.0.0.1:{port}"

    async def act(action, **args):
        r = await tool.execute(action, **args, **kw)
        assert r.success, f"{action}: {r.error}"
        return r.content

    async def js(code):
        return await act("javascript", code=code)

    async def run():
        await act("navigate", url=f"{base}/index.html")
        tree = await act("read_page")
        assert tree.startswith(browser_page.WEB_CONTENT_NOTE)
        assert 'textbox "Email"' in tree and 'button "Request statement"' in tree and "combobox" in tree
        everything = await act("read_page", filter="all")
        assert "Welcome to the portal" in everything and "heading" in everything
        email = _ref(await act("find", query="email"), "Email")
        assert 'now reads "alice@x.co"' in await act("form_input", ref=email, value="alice@x.co")
        assert 'now reads "alice@x.co"' in await act("form_input", ref=email, value="alice@x.co"), "replaced, not appended"
        assert "Quarterly" in await act("form_input", ref=_ref(tree, "combobox"), value="Quarterly")
        assert "checked" in await act("form_input", ref=_ref(tree, "I agree"), value="true")
        typed = await act("type", ref=_ref(tree, '"Note"'), text="first.draft#1")
        assert 'now reads "first.draft#1"' in typed, typed
        again = await act("type", ref=_ref(tree, '"Note"'), text="Hi there")
        assert "replacing what was there" in again and 'now reads "Hi there"' in again, again
        await act("click", ref=_ref(tree, "Request statement"))
        text = await act("page_text")
        assert "Requested for alice@x.co / q / agreed / Hi there" in text and "# Test Portal" in text

        await act("hover", ref=_ref(await act("find", query="Hover me"), "Hover me"))
        assert await js("document.getElementById('hov').textContent") == "Hovered!"
        assert '"n": 4' in await js("const t = document.title; await new Promise(r => setTimeout(r, 20)); ({title: t, n: 2 + 2})")

        shot = await tool.execute("screenshot", _look_inline=True, **kw)
        assert shot.success and "coordinate=[x, y]" in shot.content
        await act("drag", start_coordinate=[430, 450], coordinate=[760, 460])
        assert await js("document.getElementById('dragout').textContent") == "Dropped in!"
        zoom = await tool.execute("zoom", region=[0, 0, 400, 200], _look_inline=True, **kw)
        assert zoom.success and Path(zoom.images[0]).read_bytes()[:2] == b"\xff\xd8"

        up = ws / "cv.txt"
        up.write_text("hello file")
        await act("upload_file", ref=_ref(tree, "file"), path="cv.txt")
        assert await js("document.getElementById('upname').textContent") == "cv.txt 10"

        assert "page is at" in await act("scroll_to", ref=_ref(everything, "The very bottom"))
        files = await act("page_files")
        assert "report.pdf" in files and "data.csv" in files
        got = await act("download", all=True)
        assert "Downloaded 2 files" in got, got
        folder = ws / "saved" / "downloads" / kw["_session_id"] / "127.0.0.1"
        assert (folder / "report.pdf").read_bytes().startswith(b"%PDF") and (folder / "data.csv").exists()
        saved = await act("save_page")
        assert list(folder.glob("*.mhtml")), saved

        # A link that opens a new tab: the work moves there.
        moved = await act("click", ref=_ref(tree, "Open the other page"))
        assert "new tab" in moved and "Other Page" in moved, moved
        await act("click", role="button", text="Other button")
        assert "Clicked other" in await act("find", query="Clicked other")

        # A ref from before a navigation says it's gone; back works.
        await act("navigate", url=f"{base}/index.html")
        old = _ref(await act("read_page"), "Go to the other page")
        await act("click", ref=old)
        stale = await tool.execute("click", ref=old, **kw)
        assert not stale.success and "isn't on the page any more" in stale.error
        assert "Test Portal" in await act("back")

    try:
        await asyncio.wait_for(run(), 150)
    finally:
        server.shutdown()
        session = browser_mod._BROWSER_SESSIONS.pop(kw["_session_id"], None)
        if session is not None:
            await session.close()

