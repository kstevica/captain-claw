"""PR C: the glasses Files viewer never renders a member's file as markup.

``/glasses/view`` runs on the Flight Deck origin and renders ``.md`` with
``marked.parse()`` into ``innerHTML`` (marked does not sanitize). A file a
member of a shared agent added — the agent reports ``created_by`` — must
reach the page flagged, and the page must show it as escaped text (``.html``
too: escaped source, no iframe, no Present). Owner files and files with no
creator (older agents, outside saved/) render exactly as before.

The bridge half runs against a fake agent; the page half lifts ``openFile``
and its helpers out of ``glasses_view.html`` and runs them in Node against a
tiny DOM stub.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response as FastAPIResponse

from captain_claw.flight_deck import glasses_bridge

PAYLOAD = '# Notes\n\nhi <img src=x onerror="fetch(\'/fd/auth/refresh\')">\n'
MEMBER = {"kind": "member", "user_id": "u-mia", "name": "Mia"}
OWNER = {"kind": "owner", "user_id": "", "name": ""}
CH = "pr-c-glasses"


@pytest.fixture
def agent(monkeypatch):
    holder: dict = {"status": 200}

    async def fake_get(ch, path, params=None):
        if path == "/api/files":
            body = holder["listing"]
        elif path == "/api/files/content":
            body = holder["content"]
        else:
            raise AssertionError(path)
        return FastAPIResponse(content=json.dumps(body).encode(), status_code=holder["status"],
                               media_type="application/json")

    monkeypatch.setattr(glasses_bridge, "_agent_get", fake_get)
    monkeypatch.delenv("FD_GLASSES_BRIDGE_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(glasses_bridge.router)
    holder["app"] = app
    yield holder
    glasses_bridge._channels.pop(CH, None)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _entry(name: str, **extra) -> dict:
    return {"logical": f"saved/{name}", "physical": f"/w/saved/{name}", "filename": name,
            "extension": "." + name.rsplit(".", 1)[1], "size": 1, "modified": 1.0,
            "exists": True, **extra}


# ── bridge ───────────────────────────────────────────────────────────


async def test_listing_keeps_the_creator_and_flags_member_files(agent):
    agent["listing"] = [
        _entry("mia.md", created_by={**MEMBER, "user_id": "u-mia"}),
        _entry("own.md", created_by=OWNER),
        _entry("old.html"),                                   # older agent: no creator
        _entry("odd.md", created_by={"kind": "robot", "name": "R"}),
        _entry("str.md", created_by="member"),
    ]
    async with _client(agent["app"]) as c:
        r = await c.get("/glasses/files", params={"c": CH})
    assert r.status_code == 200
    got = {f["filename"]: (f["created_by"], f["member_file"]) for f in r.json()}
    assert got == {
        "mia.md": ({"kind": "member", "name": "Mia"}, True),
        "own.md": ({"kind": "owner", "name": ""}, False),
        "old.html": (None, False),
        "odd.md": ({"kind": "robot", "name": "R"}, True),
        "str.md": ({"kind": "unknown", "name": ""}, True),
    }
    assert all("user_id" not in (f["created_by"] or {}) for f in r.json())


@pytest.mark.parametrize("created_by,flag", [
    (MEMBER, True),
    ({"kind": "robot"}, True),          # an unknown kind fails closed
    ("member", True),
    (OWNER, False),
    (None, False),
    ("absent", False),                  # an older agent sends no created_by
])
async def test_content_flags_member_files(agent, created_by, flag):
    body = {"path": "/w/saved/n.md", "filename": "n.md", "content": PAYLOAD}
    if created_by != "absent":
        body["created_by"] = created_by
    agent["content"] = body
    async with _client(agent["app"]) as c:
        r = await c.get("/glasses/files/content", params={"c": CH, "path": "/w/saved/n.md"})
    assert r.status_code == 200
    data = r.json()
    assert data["member_file"] is flag
    assert data["content"] == PAYLOAD
    assert r.headers.get("cache-control") == glasses_bridge._NO_CACHE["Cache-Control"]


async def test_content_errors_are_relayed_unchanged(agent):
    agent["status"], agent["content"] = 404, {"error": "File not found on disk"}
    async with _client(agent["app"]) as c:
        r = await c.get("/glasses/files/content", params={"c": CH, "path": "/w/saved/n.md"})
    assert (r.status_code, r.json()) == (404, {"error": "File not found on disk"})


# ── page ─────────────────────────────────────────────────────────────


_HARNESS = r"""
const SRC = require('fs').readFileSync(process.argv[1], 'utf8');
const cases = JSON.parse(process.argv[2]);
function lift(start, end) {
  const i = SRC.indexOf(start); if (i < 0) throw new Error('missing ' + start);
  const j = SRC.indexOf(end, i); if (j < 0) throw new Error('missing end of ' + start);
  return SRC.slice(i, j + end.length);
}
const helpers = lift('  function escapeText(src) {', '\n  }\n')
  + lift('  function renderMd(src) {', '\n  }\n')
  + lift('  function renderPlain(src) {', '\n  }\n')
  + lift('  function isMemberFile(f, data) {', '\n  }\n');
const openFile = lift('  async function openFile(f) {', '\n  }\n');
function el(tag) {
  return { tag, children: [], attrs: {}, style: {}, className: '', textContent: '', _html: '',
    hidden: false,
    setAttribute(k, v) { this.attrs[k] = v; },
    appendChild(c) { this.children.push(c); return c; },
    set innerHTML(v) { this._html = v; created.push({ html: v, cls: this.className }); },
    get innerHTML() { return this._html; } };
}
let created = [];
const body = `
  let filesActive = '', filesFrame = null, DATA = null;
  const cq = 'c=x', tq = '', DECK_EMBED = '<!--deck-->', HTML_CLAMP = '<!--clamp-->';
  const window = { marked: { parse: (s) => 'MARKED[' + s + ']' } };
  const marked = window.marked;
  const hasMarked = true;
  const document = { createElement: (t) => { const e = el(t); made.push(e); return e; } };
  const filesContentEl = el('div'), filesPresentEl = el('div');
  function note() {}
  function focusFirst() {}
  async function getJSON() { return DATA; }
  ${helpers}
  ${openFile}
  return async function (f, data) {
    DATA = data; made.length = 0;
    await openFile(f);
    return { frames: made.filter(e => e.tag === 'iframe').map(e => e.srcdoc),
             present: !filesPresentEl.hidden,
             label: made.filter(e => e.className === 'field-k').map(e => e.textContent)[0] };
  };`;
const made = [];
const run = new Function('el', 'made', 'created', body)(el, made, created);
(async () => {
  const out = [];
  for (const [f, data] of cases) {
    created.length = 0;
    const r = await run(f, data);
    out.push({ ...r, html: created.filter(c => c.html).map(c => c.html) });
  }
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""


def _page(cases: list) -> list[dict]:
    page = glasses_bridge._STATIC_DIR / "glasses_view.html"
    proc = subprocess.run(["node", "-e", _HARNESS, str(page), json.dumps(cases)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
HTML = '<html><body><script>parent.x=1</script><h1>DECK</h1></body></html>'


def _f(name: str, **extra) -> dict:
    return {"physical": f"/w/saved/{name}", "filename": name,
            "extension": "." + name.rsplit(".", 1)[1], **extra}


@_needs_node
def test_member_markdown_is_escaped_text_never_marked():
    member, unknown, string, flagged, listed = _page([
        [_f("n.md"), {"content": PAYLOAD, "created_by": MEMBER, "member_file": True}],
        [_f("n.md"), {"content": PAYLOAD, "created_by": {"kind": "robot"}}],
        [_f("n.md"), {"content": PAYLOAD, "created_by": "member"}],
        [_f("n.md"), {"content": PAYLOAD, "member_file": True}],
        [_f("n.md", member_file=True), {"content": PAYLOAD}],
    ])
    for r in (member, unknown, string, flagged, listed):
        assert not any("MARKED[" in h for h in r["html"]), r
        joined = "".join(r["html"])
        assert "<img" not in joined and "&lt;img src=x onerror=&quot;" in joined
        assert r["frames"] == [] and "shown as text" in r["label"]
    assert "added by Mia" in member["label"]
    assert "added by a member" in unknown["label"]


@_needs_node
def test_member_html_is_escaped_source_not_a_frame_nor_presentable():
    (r,) = _page([[_f("d.html"), {"content": HTML, "created_by": MEMBER, "member_file": True}]])
    assert r["frames"] == [] and r["present"] is False
    joined = "".join(r["html"])
    assert "<script" not in joined and "&lt;script&gt;parent.x=1&lt;/script&gt;" in joined


@_needs_node
def test_owner_and_creatorless_files_render_as_before():
    owner_md, old_md, owner_html, old_html = _page([
        [_f("n.md"), {"content": "# hi", "created_by": OWNER, "member_file": False}],
        [_f("n.md"), {"content": "# hi"}],
        [_f("d.html"), {"content": HTML, "created_by": OWNER, "member_file": False}],
        [_f("d.html"), {"content": HTML, "created_by": None}],
    ])
    for r in (owner_md, old_md):
        assert r["html"] == ["MARKED[# hi]"] and r["label"] == "FILE"
    for r in (owner_html, old_html):
        assert r["frames"] == [HTML + "<!--deck--><!--clamp-->"]
        assert r["present"] is True and r["label"] == "FILE"


def test_the_page_keeps_its_one_marked_sink_behind_the_member_branch():
    src = (glasses_bridge._STATIC_DIR / "glasses_view.html").read_text(encoding="utf-8")
    body = src[src.index("  async function openFile(f) {"):]
    body = body[:body.index("\n  }\n")]
    assert body.index("if (memberFile) {") < body.index("renderMd(content)")
    assert re.search(r"filesPresentEl\.hidden = !isHtml \|\| memberFile;", body)
    plain = src[src.index("  function renderPlain(src) {"):]
    assert "marked" not in plain[:plain.index("\n  }\n")]
