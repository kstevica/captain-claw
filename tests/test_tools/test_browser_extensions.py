"""Chrome extensions in the agent's headless browser (browser_extensions.py,
BrowserSession's extension mode, the browser tool's extension_* actions).

Pinned here:

* a .crx (v2, v3) gives its zip and its key, and the key gives the id a Web
  Store extension has there; any other extension gets a key of its own so its
  id never changes; Web Store ids are found in addresses;
* unpacking can't leave the folder, follow a link or grow past the cap, and a
  zip holding one top folder is unpacked from inside it;
* the manifest tells the name (also __MSG_…__), the permissions and the sites;
* the policy: off refuses, ask asks the person (yes installs, no keeps
  nothing), ids on the allowlist and "allow" install without asking, a
  user-tier agent on a multi-user deck never installs; in the person's own
  Chrome the extension actions are refused;
* live (skipped without Playwright's Chromium): an installed extension's
  content script runs on pages, its options page opens, a second one installed
  while the browser runs restarts it on the same page, removing one unloads it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import shutil
import socket
import stat
import struct
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw.config import BrowserToolConfig
from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_extensions as bx
from captain_claw.tools.browser import BrowserTool

# ── helpers ───────────────────────────────────────────────────────────


def _manifest(name="Tester", **extra) -> dict:
    return {"manifest_version": 3, "name": name, "version": "1.2.3", **extra}


def _zip(files: dict[str, str | bytes], prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(prefix + name, data)
    return buf.getvalue()


def _key() -> bytes:
    der = bx._new_public_key()
    assert der is not None
    return der


def _pb(field: int, data: bytes) -> bytes:
    def varint(n: int) -> bytes:
        out = b""
        while True:
            b = n & 0x7F
            n >>= 7
            out += bytes([b | (0x80 if n else 0)])
            if not n:
                return out
    return varint((field << 3) | 2) + varint(len(data)) + data


def _crx3(zip_bytes: bytes, key: bytes, *, decoy: bytes | None = None) -> bytes:
    proofs = b""
    if decoy:
        proofs += _pb(2, _pb(1, decoy) + _pb(2, b"sig"))
    proofs += _pb(2, _pb(1, key) + _pb(2, b"sig"))
    header = proofs + _pb(10000, _pb(1, hashlib.sha256(key).digest()[:16]))
    return b"Cr24" + struct.pack("<II", 3, len(header)) + header + zip_bytes


def _crx2(zip_bytes: bytes, key: bytes) -> bytes:
    return b"Cr24" + struct.pack("<III", 2, len(key), 3) + key + b"sig" + zip_bytes


# ── ids and CRX ───────────────────────────────────────────────────────


def test_a_crx_gives_its_zip_and_the_key_of_its_id():
    key, decoy = _key(), _key()
    z = _zip({"manifest.json": json.dumps(_manifest())})
    body, got = bx.parse_crx(_crx3(z, key, decoy=decoy))
    assert body == z and got == key, "the key whose hash is the CRX id, not the first one"
    assert bx.parse_crx(_crx2(z, key)) == (z, key)
    assert bx.parse_crx(z) == (z, None)
    with pytest.raises(bx.ExtensionError):
        bx.parse_crx(b"not an extension")
    ext_id = bx.chrome_id(key)
    assert len(ext_id) == 32 and set(ext_id) <= set("abcdefghijklmnop")


def test_web_store_ids_are_found_in_addresses():
    eid = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    assert bx.webstore_id(eid) == eid
    assert bx.webstore_id(f"https://chromewebstore.google.com/detail/ublock-origin/{eid}?hl=en") == eid
    assert bx.webstore_id(f"https://chrome.google.com/webstore/detail/x/{eid}") == eid
    assert bx.webstore_id(f"https://example.com/{eid}") == ""
    assert bx.webstore_id("not-an-id") == ""


def test_an_extension_keeps_one_id(tmp_path):
    key = _key()
    folder = tmp_path / "a"
    folder.mkdir()
    m = _manifest()
    (folder / "manifest.json").write_text(json.dumps(m))
    assert bx.ensure_key(folder, m, key) == bx.chrome_id(key)
    again = bx.read_manifest(folder)
    assert base64.b64decode(again["key"]) == key
    assert bx.ensure_key(folder, again, None) == bx.chrome_id(key), "a key already there decides"
    other = tmp_path / "b"
    other.mkdir()
    m2 = _manifest()
    (other / "manifest.json").write_text(json.dumps(m2))
    made = bx.ensure_key(other, m2, None)
    assert made == bx.chrome_id(base64.b64decode(bx.read_manifest(other)["key"]))


# ── unpacking ─────────────────────────────────────────────────────────


def test_unpacking_stays_in_its_folder_and_under_the_cap(tmp_path):
    bx.extract_zip(_zip({"manifest.json": "{}", "js/a.js": "1"}, prefix="my-ext-1.0/"), tmp_path / "ok", 10_000)
    assert (tmp_path / "ok" / "manifest.json").exists() and (tmp_path / "ok" / "js" / "a.js").exists()
    for bad in ("../evil.js", "/etc/x", "C:/x.js"):
        with pytest.raises(bx.ExtensionError, match="outside"):
            bx.extract_zip(_zip({"manifest.json": "{}", bad: "x"}), tmp_path / "bad", 10_000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, "/etc/passwd")
    with pytest.raises(bx.ExtensionError, match="link"):
        bx.extract_zip(buf.getvalue(), tmp_path / "link", 10_000)
    with pytest.raises(bx.ExtensionError, match="MB"):
        bx.extract_zip(_zip({"manifest.json": "{}", "big.bin": b"x" * 5000}), tmp_path / "big", 1000)


def test_the_manifest_says_what_it_is_and_can_do(tmp_path):
    (tmp_path / "_locales" / "en").mkdir(parents=True)
    (tmp_path / "_locales" / "en" / "messages.json").write_text(json.dumps(
        {"appName": {"message": "Blocker"}, "appDesc": {"message": "Blocks things"}}))
    (tmp_path / "manifest.json").write_text("""{
      // Chrome reads comments
      "manifest_version": 2, "name": "__MSG_appName__", "description": "__MSG_appDesc__",
      "version": "1.0", "default_locale": "en",
      "permissions": ["storage", "tabs", "https://*/*"],
      "browser_action": {"default_popup": "popup.html"},
      "options_ui": {"page": "opts.html"},
      "content_scripts": [{"matches": ["<all_urls>"], "js": ["c.js"]}]
    }""")
    s = bx.summary(tmp_path, bx.read_manifest(tmp_path))
    assert s["name"] == "Blocker" and s["description"] == "Blocks things"
    assert s["permissions"] == ["storage", "tabs"] and s["sites"] == ["https://*/*"]
    assert s["content_scripts"] == ["<all_urls>"] and s["popup"] == "popup.html" and s["options"] == "opts.html"
    text = bx.describe(s)
    assert "Permissions: storage, tabs" in text and "<all_urls>" in text
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "x"}))
    with pytest.raises(bx.ExtensionError, match="manifest_version"):
        bx.read_manifest(tmp_path)


# ── the tool: policy and routing ──────────────────────────────────────


@pytest.fixture
def tool_env(tmp_path, monkeypatch):
    cfg = BrowserToolConfig(extension_policy="ask", default_wait_seconds=0.1)
    monkeypatch.setattr(browser_mod, "get_config", lambda: SimpleNamespace(tools=SimpleNamespace(browser=cfg)))
    monkeypatch.setattr(browser_mod, "_TARGETS", {})
    monkeypatch.setattr(bx, "deck_refusal", lambda: None)
    asked: list[str] = []
    answer = {"ok": True, "why": ""}

    async def _ask(s, ext_id, source, kwargs):
        asked.append(s["name"])
        return answer["ok"], answer["why"]

    monkeypatch.setattr(bx, "ask_person", _ask)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text(json.dumps(_manifest("Helper", permissions=["storage"])))
    return SimpleNamespace(cfg=cfg, ws=ws, src=src, asked=asked, answer=answer,
                           kw={"_runtime_base_path": str(ws), "_session_id": f"x-{tmp_path.name}"})


async def test_the_policy_decides_who_says_yes(tool_env, monkeypatch):
    tool, kw = BrowserTool(), tool_env.kw
    tool_env.cfg.extension_policy = "off"
    r = await tool.execute("extension_install", path=str(tool_env.src), **kw)
    assert not r.success and "turned off" in r.error
    tool_env.cfg.extension_policy = "ask"
    tool_env.answer.update(ok=False, why="The person said no. Don't install it.")
    r = await tool.execute("extension_install", path=str(tool_env.src), **kw)
    assert not r.success and "said no" in r.error and tool_env.asked == ["Helper"]
    store = bx.ExtensionStore.for_tool(kw)
    assert store.all() == [] and not [p for p in store.root.iterdir() if p.name.startswith(".staging")]
    tool_env.answer.update(ok=True, why="")
    r = await tool.execute("extension_install", path=str(tool_env.src), **kw)
    assert r.success and "Helper 1.2.3" in r.content and "loaded when the browser starts" in r.content
    [entry] = store.all()
    assert (store.root / entry.id / "manifest.json").exists() and len(tool_env.asked) == 2
    # The allowlist and "allow" skip the question.
    tool_env.cfg.extensions_allowed = [entry.id]
    assert (await tool.execute("extension_install", path=str(tool_env.src), **kw)).success
    tool_env.cfg.extensions_allowed = []
    tool_env.cfg.extension_policy = "allow"
    assert (await tool.execute("extension_install", path=str(tool_env.src), **kw)).success
    assert len(tool_env.asked) == 2 and len(store.all()) == 1, "the same extension again replaces it"
    listed = await tool.execute("extension_list", **kw)
    assert entry.id in listed.content and "Helper" in listed.content
    gone = await tool.execute("extension_remove", extension_id="helper", **kw)
    assert gone.success and store.all() == [] and not (store.root / entry.id).exists()
    # A user-tier agent on a multi-user deck never installs one.
    monkeypatch.setattr(bx, "deck_refusal", lambda: "Extensions can't be installed on this deck.")
    r = await tool.execute("extension_install", path=str(tool_env.src), **kw)
    assert not r.success and "this deck" in r.error


async def test_a_web_store_extension_keeps_its_store_id(tool_env, monkeypatch):
    key = _key()
    store_id = bx.chrome_id(key)
    crx = _crx3(_zip({"manifest.json": json.dumps(_manifest("Store One"))}), key)
    fetched: list[str] = []

    async def _download(url, max_bytes):
        fetched.append(url)
        return crx

    monkeypatch.setattr(bx, "_download", _download)
    tool_env.cfg.extension_policy = "allow"
    r = await BrowserTool().execute(
        "extension_install", url=f"https://chromewebstore.google.com/detail/store-one/{store_id}", **tool_env.kw)
    assert r.success and f"id {store_id}" in r.content
    assert "clients2.google.com" in fetched[0] and f"id%3D{store_id}" in fetched[0]
    assert bx.ExtensionStore.for_tool(tool_env.kw).get(store_id).source == f"Chrome Web Store {store_id}"
    bad = await BrowserTool().execute("extension_install", extension_id="nope", **tool_env.kw)
    assert not bad.success and "Chrome Web Store id" in bad.error


async def test_extension_actions_belong_to_the_headless_browser(tool_env):
    tool = BrowserTool()
    r = await tool.execute("extension_list", target="chrome", **tool_env.kw)
    assert not r.success and "installs extensions in their own Chrome themselves" in r.error
    assert tool.timeout_for({"action": "extension_install"}) == 720.0
    assert tool.timeout_for({"action": "click"}) == tool.timeout_seconds
    r = await tool.execute("extension_open", extension_id="missing", **tool_env.kw)
    assert not r.success and "No extension" in r.error



def _deck_zip() -> bytes:
    """Captain for Chrome as Flight Deck hands it out (/fd/clip/extension.zip)."""
    src = bx.captain_checkout()
    assert src is not None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f"captain-for-chrome/{f.relative_to(src).as_posix()}")
    return buf.getvalue()


async def test_captain_for_chrome_comes_from_this_captain(tool_env, monkeypatch):
    from captain_claw import fd_client

    fetched: list[str] = []
    deck = {"base": "http://fd.test", "zip": _deck_zip()}

    async def _download(url, max_bytes):
        fetched.append(url)
        if deck["zip"] is None:
            raise bx.ExtensionError("The download answered 404.")
        return deck["zip"]

    monkeypatch.setattr(bx, "_download", _download)
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: deck["base"])
    tool, kw = BrowserTool(), tool_env.kw
    version = json.loads((bx.captain_checkout() / "manifest.json").read_text())["version"]
    r = await tool.execute("extension_install", extension_id="Captain for Chrome", **kw)
    assert r.success, r.error
    assert f"Installed Captain for Chrome {version} from this Captain" in r.content and "read_page" in r.content
    assert fetched == ["http://fd.test/fd/clip/extension.zip"]
    assert tool_env.asked == [], "Captain's own extension needs no approval"
    store = bx.ExtensionStore.for_tool(kw)
    [entry] = store.all()
    assert entry.source == bx.CAPTAIN_SOURCE and (store.root / entry.id / "hands.js").exists()
    # From the checkout when there's no deck, or the deck has no copy — the same extension, the same id.
    deck["base"] = ""
    assert (await tool.execute("extension_install", extension_id="captain", **kw)).success
    deck["base"], deck["zip"] = "http://fd.test", None
    assert (await tool.execute("extension_install", extension_id="captain", **kw)).success
    assert len(fetched) == 2 and [e.id for e in store.all()] == [entry.id]
    tool_env.cfg.extension_policy = "off"
    r = await tool.execute("extension_install", extension_id="captain", **kw)
    assert not r.success and "turned off" in r.error



async def test_connecting_its_own_browser_needs_the_deck_and_the_persons_yes(tool_env, monkeypatch):
    from captain_claw import fd_client

    tool, kw = BrowserTool(), tool_env.kw
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
    r = await tool.execute("extension_connect", **kw)
    assert not r.success and "needs Flight Deck" in r.error
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "http://fd.test")

    async def _download(url, max_bytes):
        return _deck_zip()

    monkeypatch.setattr(bx, "_download", _download)
    asked: list[dict] = []

    async def _no(kwargs):
        asked.append(kwargs)
        return {"ok": False, "declined": True, "error": "The person said no. Don't connect it unless they ask you to."}

    monkeypatch.setattr(bx, "ask_to_connect", _no)
    r = await tool.execute("extension_connect", **kw)
    store = bx.ExtensionStore.for_tool(kw)
    assert not r.success and "said no" in r.error and len(asked) == 1
    assert store.captain() is not None, "Captain for Chrome is installed first"
    assert store.captain_conn() is None and store.seed() is None
    tool_env.cfg.extension_policy = "off"
    assert "turned off" in (await tool.execute("extension_connect", **kw)).error
    # A connection kept, then Captain removed: the connection goes, and the deck is told.
    store.save_captain_conn({"url": "http://fd.test", "token": "cc_ab_x", "agent": "Scout"})
    seed = store.seed()
    assert seed["ext_id"] == store.captain().id and seed["storage"]["hands"] == {"on": True, "front": False}
    let_go: list[bool] = []

    async def _let_go():
        let_go.append(True)

    monkeypatch.setattr(bx, "let_go_of_own_browser", _let_go)
    assert (await tool.execute("extension_remove", extension_id="captain for chrome", **kw)).success
    assert store.captain_conn() is None and let_go == [True]


# ── live: a real headless Chromium ────────────────────────────────────


def _ext(folder: Path, name: str, marker: str) -> Path:
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "manifest_version": 3, "name": name, "version": "1.0",
        "background": {"service_worker": "sw.js"},
        "action": {"default_popup": "popup.html"}, "options_page": "options.html",
        "content_scripts": [{"matches": ["<all_urls>"], "js": ["cs.js"], "run_at": "document_end"}],
    }))
    (folder / "sw.js").write_text("self.addEventListener('install', () => {})")
    (folder / "cs.js").write_text(
        f"document.documentElement.dataset.{marker} = 'on';"
        f"const d = document.createElement('p'); d.textContent = '{name} was here'; document.body.appendChild(d);")
    (folder / "options.html").write_text(f"<!doctype html><title>{name} options</title><h1>Options</h1>")
    (folder / "popup.html").write_text(f"<!doctype html><title>{name} popup</title><button>Go</button>")
    return folder


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


@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium isn't installed")
async def test_live_extensions_load_open_and_go(tool_env, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "page.html").write_text("<!doctype html><title>Plain page</title><body><h1>Hello</h1></body>")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tool_env.cfg.extension_policy = "allow"
    tool, kw = BrowserTool(), tool_env.kw
    session = None
    try:
        async def run():
            nonlocal session
            first = _ext(tmp_path / "e1", "First Ext", "firstExt")
            r = await tool.execute("extension_install", path=str(first), **kw)
            assert r.success, r.error
            first_id = bx.ExtensionStore.for_tool(kw).get("first ext").id
            page_url = f"http://127.0.0.1:{port}/page.html"
            r = await tool.execute("navigate", url=page_url, **kw)
            assert r.success, r.error
            session = browser_mod._BROWSER_SESSIONS[kw["_session_id"]]
            page = session._page
            await page.wait_for_function("document.documentElement.dataset.firstExt === 'on'", timeout=10000)
            assert "First Ext was here" in await session.get_content()
            # The page actions work in this mode too (the extension's own scripts, isolated from its content script).
            text = await tool.execute("page_text", **kw)
            assert text.success and "Hello" in text.content and "First Ext was here" in text.content, text.error
            workers = [w.url for w in session._context.service_workers]
            for _ in range(50):
                if workers:
                    break
                await asyncio.sleep(0.1)
                workers = [w.url for w in session._context.service_workers]
            assert any(f"chrome-extension://{first_id}/" in u for u in workers), (first_id, workers)

            r = await tool.execute("extension_open", extension_id=first_id, extension_page="options", **kw)
            assert r.success and "First Ext options" in r.content, r.content or r.error
            r = await tool.execute("extension_open", extension_id="first ext", **kw)
            assert r.success and "First Ext popup" in r.content, "popup by default"

            # A second one while the browser runs: it restarts, back on the page it was on.
            await tool.execute("navigate", url=page_url, **kw)
            zipped = tmp_path / "second.zip"
            second = _ext(tmp_path / "e2", "Second Ext", "secondExt")
            with zipfile.ZipFile(zipped, "w") as zf:
                for f in second.iterdir():
                    zf.write(f, f"second-1.0/{f.name}")
            r = await tool.execute("extension_install", path=str(zipped), **kw)
            assert r.success and "restarted" in r.content and page_url in r.content, r.content or r.error
            page = session._page
            await page.wait_for_function(
                "document.documentElement.dataset.firstExt === 'on' && document.documentElement.dataset.secondExt === 'on'",
                timeout=10000)
            listed = (await tool.execute("extension_list", **kw)).content
            assert listed.count("— loaded —") == 2, listed

            # Removing one unloads it.
            r = await tool.execute("extension_remove", extension_id=first_id, **kw)
            assert r.success and "restarted" in r.content, r.content
            page = session._page
            await page.wait_for_function("document.documentElement.dataset.secondExt === 'on'", timeout=10000)
            assert await page.evaluate("document.documentElement.dataset.firstExt") is None
            status = await tool.execute("status", **kw)
            assert "Extensions loaded: 1" in status.content

        await asyncio.wait_for(run(), 150)
    finally:
        server.shutdown()
        if session is not None:
            profiles = list(session._profile_dirs)
            await session.close()
            assert not any(Path(p).exists() for p in profiles), "the temporary profiles go"
        browser_mod._BROWSER_SESSIONS.pop(kw["_session_id"], None)
        shutil.rmtree(tmp_path / "ws", ignore_errors=True)


@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium isn't installed")
async def test_live_captain_for_chrome_runs_in_the_headless_browser(tool_env, monkeypatch):
    from captain_claw import fd_client

    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")   # from the checkout
    tool, kw = BrowserTool(), tool_env.kw
    session = None
    try:
        async def run():
            nonlocal session
            r = await tool.execute("extension_install", extension_id="captain", **kw)
            assert r.success, r.error
            ext_id = bx.ExtensionStore.for_tool(kw).get("captain for chrome").id
            r = await tool.execute("extension_open", extension_id=ext_id, extension_page="sidepanel.html", **kw)
            assert r.success and "Captain" in r.content, r.content or r.error
            session = browser_mod._BROWSER_SESSIONS[kw["_session_id"]]
            for _ in range(50):
                urls = [w.url for w in session._context.service_workers]
                if any(f"chrome-extension://{ext_id}/background.js" == u for u in urls):
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError(f"its worker never started: {urls}")
            worker = next(w for w in session._context.service_workers if ext_id in w.url)
            assert await worker.evaluate("typeof chrome.debugger + ' ' + typeof chrome.tabGroups") == "object object"
            # The panel asks to connect: it isn't paired here.
            await session._page.wait_for_selector("#deckUrl", timeout=10000)

        await asyncio.wait_for(run(), 120)
    finally:
        if session is not None:
            await session.close()
        browser_mod._BROWSER_SESSIONS.pop(kw["_session_id"], None)
