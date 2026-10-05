"""Shared-agent UI logic (A1: chat-only shared agent), run in Node.

The Flight Deck frontend has no JS test runner, so — as in
test_gmail_send_ui.py and test_profile_ui.py — these tests lift named
top-level declarations out of the TypeScript sources with the TypeScript
compiler from flight-deck/node_modules, transpile them and run them in a Node
vm, and read a few wiring facts off the parsed tree. They skip when Node or the
frontend dependencies aren't installed.

Pinned here:

* the member socket URL carries the agent ref, the lane and the FD token —
  never the agent's `token`, host or port;
* each Flight Deck close code maps to the message and retry the contract
  promises (part 0 §7 / part 3 §1);
* "is this a Flight Deck–managed worker?" agrees with the server's table,
  and a docker agent's slug is derived the way the server derives it;
* `shared:<ref>` chat ids survive the lane-key round trip;
* the owner note and the member notice carry the deck's host-trust warning
  (and the owner note the restart hint);
* what this browser keeps about a shared chat — queue and plan slices, the
  notice dismissal — is kept per deck user, and leaving drops the slices;
* a bell notice about an agent shared with you leads to it, and the Agent
  Desktop shows "Shared with me" above your own fleet;
* a shared chat never runs the peer_agents handshake, and the shared socket
  never touches the agent's host, port or token.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FD = _ROOT / "flight-deck"
_SHARED = _FD / "src" / "utils" / "sharedAgent.ts"
_MANAGED = _FD / "src" / "utils" / "managedAgents.ts"
_CHAT_STORE = _FD / "src" / "stores" / "chatStore.ts"
_AGENT_CHAT = _FD / "src" / "services" / "agentChat.ts"
_NOTICE = _FD / "src" / "components" / "agents" / "SharedAgentNotice.tsx"
_SHARED_STORE = _FD / "src" / "stores" / "sharedAgentStore.ts"
_NOTIF_STORE = _FD / "src" / "stores" / "notificationStore.ts"
_NOTIF_CENTER = _FD / "src" / "components" / "common" / "NotificationCenter.tsx"
_DESKTOP = _FD / "src" / "pages" / "DesktopPage.tsx"
_SERVER = _ROOT / "captain_claw" / "flight_deck" / "server.py"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

HOST_TRUST_WARNING = (
    "Anyone on this deck who runs their own shell-capable process agent can act as any agent "
    "on this host, including this one."
)

# argv: typescript dir. stdin: {"decls": [{"src", "names"}], "body": js}.
# Runs the named top-level declarations + body in a vm; body sets out (and
# may set done to a promise, awaited before out is printed).
_LIFT = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const kind = (p) => p.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
const declName = (st) => {
  if (st.name) return st.name.text;
  if (ts.isVariableStatement(st)) return st.declarationList.declarations[0].name.getText();
  return null;
};
const parts = [];
for (const d of req.decls) {
  const sf = ts.createSourceFile(d.src, fs.readFileSync(d.src, 'utf8'),
    ts.ScriptTarget.Latest, true, kind(d.src));
  const found = [];
  for (const st of sf.statements) {
    const n = declName(st);
    if (n && d.names.includes(n)) { parts.push(st.getText(sf)); found.push(n); }
  }
  if (found.length !== d.names.length) {
    throw new Error('declarations not found in ' + d.src + ': wanted ' + d.names.join(', '));
  }
}
const js = ts.transpileModule(parts.join('\n') + '\n' + req.body, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const ctx = vm.createContext({ exports: {}, out: null, done: null, URLSearchParams });
vm.runInContext(js, ctx);
// A body may set `done` to a promise (async scenarios); `out` is read after it.
Promise.resolve(ctx.done).then(
  () => process.stdout.write(JSON.stringify(ctx.out)),
  (e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); },
);
"""

# argv: typescript dir, source path. stdin: a function body over (ts, sf, src)
# whose return value is printed as JSON.
_QUERY = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const src = process.argv[2];
const sf = ts.createSourceFile(src, fs.readFileSync(src, 'utf8'),
  ts.ScriptTarget.Latest, true, src.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
const code = fs.readFileSync(0, 'utf8');
process.stdout.write(JSON.stringify(new Function('ts', 'sf', 'src', code)(ts, sf, src)));
"""

_WALK = r"""
const all = [];
(function walk(n) { all.push(n); ts.forEachChild(n, walk); })(sf);
"""


def _lift(decls: list[tuple[Path, list[str]]], body: str):
    req = {"decls": [{"src": str(src), "names": names} for src, names in decls], "body": body}
    proc = subprocess.run(
        ["node", "-e", _LIFT, str(_TS)],
        input=json.dumps(req), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _query(src: Path, code: str):
    proc = subprocess.run(
        ["node", "-e", _QUERY, str(_TS), str(src)],
        input=_WALK + code, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ── sharedWsUrl: ref, lane and FD token only ────────────────────────


def _ws_url(ref: str, lane: str, token: str, protocol: str, host: str) -> str:
    loc = json.dumps({"protocol": protocol, "host": host})
    return _lift([(_SHARED, ["sharedWsUrl"])], (
        f"out = sharedWsUrl({json.dumps(ref)}, {json.dumps(lane)}, {json.dumps(token)}, {loc});"
    ))


def test_shared_ws_url_names_the_agent_by_ref_only():
    ref = "process:helper:0123456789abcdef"
    url = _ws_url(ref, "B", "jwt.with+odd/chars=", "http:", "deck.example:25080")
    parts = urlsplit(url)
    assert parts.scheme == "ws"
    # The deck itself — no agent host or port anywhere in the URL.
    assert parts.netloc == "deck.example:25080"
    assert parts.path == "/fd/agent-ws-shared"
    q = parse_qs(parts.query, keep_blank_values=True)
    assert set(q) == {"ref", "lane", "fd_token"}
    assert "token" not in q
    assert q == {"ref": [ref], "lane": ["B"], "fd_token": ["jwt.with+odd/chars="]}


def test_shared_ws_url_uses_wss_on_https_and_defaults_to_lane_a():
    url = _ws_url("docker:my-agent:fedcba9876543210", "", "t", "https:", "deck.example")
    parts = urlsplit(url)
    assert parts.scheme == "wss"
    assert parts.netloc == "deck.example"
    assert parse_qs(parts.query)["lane"] == ["A"]


# ── sharedCloseInfo: message + retry per close code ─────────────────


def _close_info(code: int, owner: str = "Olga", reason: str = "") -> dict:
    return _lift([(_SHARED, ["GENERIC_REVOKE_REASON", "sharedCloseInfo"])], (
        f"out = sharedCloseInfo({code}, {json.dumps(owner)}, {json.dumps(reason)});"
    ))


def test_close_codes_map_to_retry_kinds():
    assert _close_info(4001)["retry"] == "refresh"
    for code in (4403, 4404, 4400, 4503):
        assert _close_info(code)["retry"] == "none", code
    for code in (4409, 4426, 4429, 4502):
        assert _close_info(code)["retry"] == "button", code
    assert _close_info(4999)["retry"] == "none"


def test_close_messages_name_the_owner():
    assert _close_info(4403)["message"] == "Olga removed your access to this agent"
    assert _close_info(4409)["message"] == "This agent is stopped — ask Olga to start it"
    assert _close_info(4426)["message"] == (
        "This agent needs a restart before it can be shared — ask Olga"
    )
    assert _close_info(4001)["message"] == "Your session expired — sign in again, or Retry"
    assert _close_info(4404)["message"] == "This agent no longer exists"
    assert _close_info(4502)["message"] == "Couldn't reach the agent"
    assert _close_info(4503)["message"] == "Agent sharing is turned off on this Flight Deck"
    assert _close_info(4429)["message"].startswith("Too many open chats with this agent")


def test_close_messages_fall_back_to_the_reason():
    assert _close_info(4400, reason="You own this agent")["message"] == "You own this agent"
    assert _close_info(4400)["message"] == "This shared agent can't be opened"
    assert _close_info(4555, reason="Odd")["message"] == "Odd"
    assert _close_info(4555)["message"] == "Disconnected"


def test_expired_session_banner_does_not_promise_a_reconnect():
    # `_closed` 4001 only arrives once the socket gave up (refresh failed, or
    # refused again before a welcome) — nothing is reconnecting by then.
    info = _close_info(4001, reason="User not found")
    assert "reconnecting" not in info["message"]
    assert "sign in again" in info["message"]
    assert info["retry"] == "refresh"


def test_access_lost_prefers_flight_decks_reason():
    # A member who left in another tab, or an agent that can't be shared any
    # more, must not read as "the owner removed you".
    assert _close_info(4403, reason="You left this shared agent")["message"] == (
        "You left this shared agent"
    )
    respawn = "This agent has no access token; respawn it to share it"
    assert _close_info(4403, reason=respawn)["message"] == respawn
    # The plain revoke (and no reason at all) still names the owner.
    assert _close_info(4403, reason="Access removed")["message"] == (
        "Olga removed your access to this agent"
    )
    assert _close_info(4403, reason="  ")["message"] == "Olga removed your access to this agent"
    assert _close_info(4403, owner="", reason="Access removed")["message"] == (
        "The owner removed your access to this agent"
    )
    for reason in ("You left this shared agent", "Access removed", ""):
        assert _close_info(4403, reason=reason)["retry"] == "none"


def test_generic_revoke_reason_matches_the_server():
    import inspect

    sharing = pytest.importorskip("captain_claw.flight_deck.agent_sharing")
    server = inspect.signature(sharing.close_member_sockets).parameters["reason"].default
    ts_side = _lift([(_SHARED, ["GENERIC_REVOKE_REASON"])], "out = GENERIC_REVOKE_REASON;")
    assert ts_side == server == "Access removed"


# ── managed workers + docker slugs: parity with the server ──────────

# part 1 §4: the table the server's is_managed_agent is tested against.
_MANAGED_TABLE = [
    ("basna-1a2b3c4d-x", "", True),
    ("vatra-0badf00d-writer", "", True),
    ("council-abcdef-y", "", True),
    ("iskra-foo-1a2b", "", True),
    ("dubina-x", "Dubina ephemeral worker", True),
    ("dubina-x", "my own dubina helper", False),
    ("council notes", "", False),
    ("council-notes", "", False),
    ("basna-helper", "", False),
    ("iskra-foo", "", False),
    ("helper", "", False),
]


def _ts_managed(rows) -> list[bool]:
    calls = ", ".join(f"isManagedAgent({json.dumps(s)}, {json.dumps(d)})" for s, d, _ in rows)
    return _lift([(_MANAGED, ["MANAGED_AGENT", "isManagedAgent"])], f"out = [{calls}];")


def test_is_managed_agent_matches_the_contract_table():
    assert _ts_managed(_MANAGED_TABLE) == [want for _, _, want in _MANAGED_TABLE]


def test_is_managed_agent_matches_the_server_when_present():
    sharing = pytest.importorskip("captain_claw.flight_deck.agent_sharing")
    server_side = [sharing.is_managed_agent(s, d) for s, d, _ in _MANAGED_TABLE]
    assert _ts_managed(_MANAGED_TABLE) == server_side


def _server_slug():
    """The server's `_slug`, lifted from server.py's source (no app import)."""
    tree = ast.parse(_SERVER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_slug")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SERVER), "exec"), ns)
    return ns["_slug"]


def test_docker_slug_matches_the_server():
    names = ["My Agent", "  Spaced  ", "Ünïcode Bot", "---x---", "a_b.c", "", "Research-Desk 2"]
    slug = _server_slug()
    calls = ", ".join(f"dockerSlug({json.dumps(n)})" for n in names)
    got = _lift([(_MANAGED, ["dockerSlug"])], f"out = [{calls}];")
    assert got == [slug(n) for n in names]


# ── shared chat ids and lane keys ───────────────────────────────────


def test_shared_container_id_survives_the_lane_key_round_trip():
    out = _lift([
        (_SHARED, ["SHARED_PREFIX", "sharedContainerId"]),
        (_CHAT_STORE, ["LANE_MAIN", "laneKey", "parseLaneKey"]),
    ], (
        "const cid = sharedContainerId('process:x:0123456789abcdef');"
        "out = { cid, b: laneKey(cid, 'B'), a: laneKey(cid, 'A'),"
        " parsedB: parseLaneKey(laneKey(cid, 'B')), parsedA: parseLaneKey(cid) };"
    ))
    assert out["cid"] == "shared:process:x:0123456789abcdef"
    assert out["b"] == "shared:process:x:0123456789abcdef::B"
    assert out["a"] == out["cid"]
    assert out["parsedB"] == {"containerId": out["cid"], "lane": "B"}
    assert out["parsedA"] == {"containerId": out["cid"], "lane": "A"}


# ── what owners and members are told ────────────────────────────────


def test_owner_note_ends_with_the_host_trust_warning():
    note = _lift([(_SHARED, ["OWNER_SHARE_NOTE"])], "out = OWNER_SHARE_NOTE;")
    assert note.startswith("Members chat with this agent in their own private conversations")
    assert "no shell, files, Google, deep memory, MCP servers, scheduled jobs or your fleet" in note
    assert "Member chats use your LLM keys." in note
    # An agent started before sharing was on answers members 4426 until it
    # restarts — the owner is the one who can act on that.
    assert (
        "If this agent was running before sharing was turned on, restart it once so "
        "members can connect." in note
    )
    assert note.endswith("\n\n" + HOST_TRUST_WARNING)


def test_member_notice_names_agent_owner_and_appends_the_warning():
    out = _lift([(_SHARED, ["memberNoticeText"])], (
        f"out = {{ withWarning: memberNoticeText('Helper', 'Olga', {json.dumps(HOST_TRUST_WARNING)}),"
        " without: memberNoticeText('Helper', 'Olga', '') };"
    ))
    assert out["without"].startswith("**Helper belongs to Olga.** ")
    assert "not from Olga or this deck's admins" in out["without"]
    # The knowledge reaches Olga's own chats with THIS agent — not some other
    # agent of hers.
    assert "becomes shared knowledge for everyone using it, including Olga's own chats with it." in (
        out["without"]
    )
    assert "own agent" not in out["without"]
    assert out["without"].endswith(
        "it can only search the web, read public pages and use its shared insights, "
        "playbooks and topics."
    )
    assert "\n" not in out["without"]
    assert out["withWarning"] == out["without"] + "\n\n" + HOST_TRUST_WARNING


# ── wiring: no peer handshake, no agent credentials ─────────────────


def test_shared_welcome_returns_before_the_peer_agents_handshake():
    found = _query(_CHAT_STORE, r"""
      const welcome = all.filter((n) => ts.isCallExpression(n)
          && n.expression.getText(sf) === 'ws.on'
          && n.arguments.length === 2 && n.arguments[0].getText(sf) === "'welcome'");
      if (welcome.length !== 1) return { count: welcome.length };
      const body = welcome[0].arguments[1].body;
      const first = body.statements.find((st) => ts.isIfStatement(st)
          && st.expression.getText(sf) === 'shared');
      const peerAt = body.getText(sf).indexOf("type: 'peer_agents'");
      return {
        count: 1,
        guard: !!first,
        guardReturns: !!first && first.thenStatement.statements.some((st) => ts.isReturnStatement(st)),
        guardBeforePeers: !!first && peerAt > 0
          && (first.getStart(sf) - body.getStart(sf)) < peerAt,
      };
    """)
    assert found == {"count": 1, "guard": True, "guardReturns": True, "guardBeforePeers": True}


def test_shared_socket_never_uses_the_agent_credentials():
    text = _query(_AGENT_CHAT, r"""
      const m = all.find((n) => ts.isMethodDeclaration(n) && n.name.getText(sf) === '_openSharedSocket');
      return m ? m.getText(sf) : null;
    """)
    assert text, "_openSharedSocket not found"
    assert "sharedWsUrl(" in text
    for forbidden in ("this.auth", "this.host", "this.port", "/fd/agent-ws/"):
        assert forbidden not in text, forbidden


# ── the member socket's close policy, driven against a fake WebSocket ──

# Stubs for what AgentChatWS reaches outside itself: the browser WebSocket and
# location, the auth store (token + subscribe), refreshAccessToken and timers.
# Timers are recorded, never fired — a scheduled reconnect shows up there.
_WS_HARNESS = r"""
const sockets = [];
class FakeWS {
  static OPEN = 1;
  constructor(url) { this.url = url; this.readyState = 1; this.sent = []; this.closed = false; sockets.push(this); }
  send(d) { this.sent.push(JSON.parse(d)); }
  close() { this.closed = true; this.readyState = 3; }
}
const WebSocket = FakeWS;
const window = { location: { protocol: 'https:', host: 'deck.example' } };
let token = 'jwt-1';
const listeners = [];
const useAuthStore = {
  getState: () => ({ token }),
  subscribe: (fn) => { listeners.push(fn); return () => { const i = listeners.indexOf(fn); if (i >= 0) listeners.splice(i, 1); }; },
};
function rotate(t) { const prev = { token }; token = t; for (const l of [...listeners]) l({ token }, prev); }
let refreshCalls = 0;
let refreshOk = true;
const refreshAccessToken = () => {
  refreshCalls++;
  if (refreshOk) rotate('jwt-r' + refreshCalls);
  return Promise.resolve(refreshOk);
};
const timers = [];
const setTimeout = (fn, ms) => { timers.push(ms); return timers.length; };
const clearTimeout = () => {};
const flush = async () => { for (let i = 0; i < 6; i++) await null; };
const REF = 'process:helper:0123456789abcdef';
function fresh() {
  sockets.length = 0; timers.length = 0; listeners.length = 0;
  refreshCalls = 0; refreshOk = true; token = 'jwt-1';
  const c = new AgentChatWS('shared:' + REF, '', 0, '', 'B', { sharedRef: REF });
  const ev = [];
  for (const n of ['_connected', '_closed', '_reconnecting']) c.on(n, (d) => ev.push([n, d]));
  return { c, ev };
}
const frame = (s, f) => s.onmessage({ data: JSON.stringify(f) });
"""


def _ws_scenario(body: str):
    return _lift(
        [(_SHARED, ["sharedWsUrl"]), (_AGENT_CHAT, ["AgentChatWS"])],
        _WS_HARNESS + "done = (async () => {" + body + "})();",
    )


def test_member_socket_names_only_the_ref_and_is_usable_on_welcome():
    out = _ws_scenario("""
      const { c, ev } = fresh();
      c.connect();
      const s = sockets[0];
      const beforeWelcome = c.connected;
      frame(s, { type: 'welcome', speaker: { id: 'u', name: 'Ana', owner_name: 'Olga', lane: 'B' } });
      out = { url: s.url, beforeWelcome, after: c.connected, ev: ev.map((e) => e[0]) };
    """)
    parts = urlsplit(out["url"])
    assert (parts.scheme, parts.netloc, parts.path) == ("wss", "deck.example", "/fd/agent-ws-shared")
    assert parse_qs(parts.query) == {
        "ref": ["process:helper:0123456789abcdef"], "lane": ["B"], "fd_token": ["jwt-1"],
    }
    # FD accepts before it checks: open alone isn't "connected" — the welcome is.
    assert out["beforeWelcome"] is False
    assert out["after"] is True
    assert out["ev"] == ["_connected"]


@pytest.mark.parametrize("code", [4403, 4404, 4400, 4409, 4426, 4429, 4502, 4503, 4555])
def test_flight_deck_close_codes_are_final(code):
    out = _ws_scenario(f"""
      const {{ c, ev }} = fresh();
      c.connect();
      frame(sockets[0], {{ type: 'welcome' }});
      frame(sockets[0], {{ type: 'fd_close', code: {code}, reason: 'Access removed' }});
      sockets[0].onclose({{ code: {code}, reason: 'ws reason' }});
      await flush();
      out = {{ sockets: sockets.length, timers: timers.length, refreshCalls, ev, connected: c.connected }};
    """)
    # No backoff loop, no refresh: one socket, nothing scheduled, the reason
    # from FD's fd_close frame (not the transport's) reaches the chat.
    assert out["sockets"] == 1
    assert out["timers"] == 0
    assert out["refreshCalls"] == 0
    assert out["connected"] is False
    assert out["ev"][-1] == ["_closed", {"code": code, "reason": "Access removed"}]
    assert [e for e in out["ev"] if e[0] == "_reconnecting"] == []


def test_network_drop_still_reconnects_with_backoff():
    out = _ws_scenario("""
      const { c, ev } = fresh();
      c.connect();
      frame(sockets[0], { type: 'welcome' });
      sockets[0].onclose({ code: 1006, reason: '' });
      await flush();
      out = { timers: timers.length, ev: ev.map((e) => e[0]) };
    """)
    assert out["timers"] == 1
    assert out["ev"] == ["_connected", "_reconnecting"]


def test_expired_token_refreshes_and_reopens_exactly_once():
    out = _ws_scenario("""
      const { c, ev } = fresh();
      c.connect();
      sockets[0].onclose({ code: 4001, reason: '' });
      await flush();
      const afterFirst = { sockets: sockets.length, refreshCalls, url: sockets[sockets.length - 1].url };
      // Still expired before any welcome: final this time.
      sockets[1].onclose({ code: 4001, reason: '' });
      await flush();
      out = { afterFirst, sockets: sockets.length, refreshCalls, timers: timers.length,
              last: ev[ev.length - 1] };
    """)
    assert out["afterFirst"]["sockets"] == 2
    assert out["afterFirst"]["refreshCalls"] == 1
    # The reopened socket carries the refreshed token.
    assert parse_qs(urlsplit(out["afterFirst"]["url"]).query)["fd_token"] == ["jwt-r1"]
    assert out["sockets"] == 2
    assert out["refreshCalls"] == 1
    assert out["timers"] == 0
    assert out["last"][0] == "_closed" and out["last"][1]["code"] == 4001


def test_expired_token_budget_resets_on_welcome():
    out = _ws_scenario("""
      const { c } = fresh();
      c.connect();
      sockets[0].onclose({ code: 4001, reason: '' });
      await flush();
      frame(sockets[1], { type: 'welcome' });
      sockets[1].onclose({ code: 4001, reason: '' });
      await flush();
      out = { sockets: sockets.length, refreshCalls };
    """)
    assert out == {"sockets": 3, "refreshCalls": 2}


def test_failed_refresh_ends_the_chat():
    out = _ws_scenario("""
      const { c, ev } = fresh();
      refreshOk = false;
      c.connect();
      sockets[0].onclose({ code: 4001, reason: '' });
      await flush();
      out = { sockets: sockets.length, timers: timers.length, last: ev[ev.length - 1] };
    """)
    assert out["sockets"] == 1
    assert out["timers"] == 0
    assert out["last"][0] == "_closed" and out["last"][1]["code"] == 4001


def test_rotated_token_goes_to_flight_deck_until_disconnect():
    out = _ws_scenario("""
      const { c } = fresh();
      c.connect();
      frame(sockets[0], { type: 'welcome' });
      rotate('jwt-2');
      c.disconnect();
      rotate('jwt-3');
      out = { sent: sockets[0].sent, listeners: listeners.length };
    """)
    assert out["sent"] == [{"type": "fd_auth", "fd_token": "jwt-2"}]
    assert out["listeners"] == 0


def test_owner_socket_is_unchanged():
    out = _ws_scenario("""
      sockets.length = 0; timers.length = 0; listeners.length = 0;
      const c = new AgentChatWS('proc-helper', 'localhost', 24001, 'agent-secret', '');
      const ev = [];
      for (const n of ['_connected', '_closed', '_reconnecting']) c.on(n, () => ev.push(n));
      c.connect();
      sockets[0].onopen();
      sockets[0].onclose({ code: 4403, reason: '' });
      await flush();
      out = { url: sockets[0].url, ev, timers: timers.length, listeners: listeners.length };
    """)
    parts = urlsplit(out["url"])
    assert parts.path == "/fd/agent-ws/localhost/24001"
    assert parse_qs(parts.query) == {"token": ["agent-secret"], "fd_token": ["jwt-1"]}
    # Owner chats keep reconnecting whatever the code, and never watch the token.
    assert out["ev"] == ["_connected", "_reconnecting"]
    assert out["timers"] == 1
    assert out["listeners"] == 0


# ── a shared agent's queue doesn't outlive the session in this browser ──

# A Map-backed localStorage and a switchable signed-in user: `user = …` is a
# session change with no teardown in between (expired refresh cookie, an
# admin reset) — the path the sign-out purge never sees.
_STORAGE_HARNESS = r"""
const store = new Map();
const window = { localStorage: {
  get length() { return store.size; },
  key: (i) => [...store.keys()][i] ?? null,
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => { store.set(k, String(v)); },
  removeItem: (k) => { store.delete(k); },
} };
let user = { id: 'u-ana' };
const useAuthStore = { getState: () => ({ user }) };
const R = 'process:x:0123456789abcdef';
const R2 = 'docker:y:fedcba9876543210';
"""

_SLICE_DECLS = [
    (_SHARED, ["SHARED_PREFIX", "sharedContainerId", "sliceUser", "sharedSliceId"]),
    (_CHAT_STORE, [
        "_planLSKey", "savePlanSlice", "loadPlanSlice", "_queueLSKey", "saveQueueSlice",
        "loadQueueSlice", "_sliceId", "purgeSharedSlices", "clearSharedSlices",
        "LANE_MAIN", "LANES", "laneKey",
    ]),
]

_ANA_QUEUE = (
    "{ queue: [{ id: 'q1', content: 'Ana private task', status: 'pending', createdAt: 1 }],"
    " queueAutoMode: true }"
)
_PLAN = "{ planState: null, planningEnabled: true, planLevel: 'enriched', planCardCollapsed: false }"


def test_shared_slices_are_kept_per_deck_user():
    out = _lift(_SLICE_DECLS, _STORAGE_HARNESS + f"""
      const cid = sharedContainerId(R);
      saveQueueSlice(cid, {_ANA_QUEUE});
      saveQueueSlice(laneKey(cid, 'B'), {_ANA_QUEUE});
      savePlanSlice(cid, {_PLAN});
      saveQueueSlice('proc-helper', {_ANA_QUEUE});
      const keys = [...store.keys()].sort();
      user = {{ id: 'u-bob' }};
      const bob = {{ a: loadQueueSlice(cid), b: loadQueueSlice(laneKey(cid, 'B')), plan: loadPlanSlice(cid) }};
      user = {{ id: 'u-ana' }};
      const ana = loadQueueSlice(cid);
      user = null;
      const anon = _queueLSKey(cid);
      out = {{ keys, bob, ana: ana && ana.queue.map((q) => q.content), anon,
              own: _queueLSKey('proc-helper') }};
    """)
    assert out["keys"] == sorted([
        "fd.plan.shared:process:x:0123456789abcdef@u-ana",
        "fd.queue.shared:process:x:0123456789abcdef@u-ana",
        "fd.queue.shared:process:x:0123456789abcdef::B@u-ana",
        # An agent you own keeps its key, user or not.
        "fd.queue.proc-helper",
    ])
    # The next person on this browser starts empty — on every lane.
    assert out["bob"] == {"a": None, "b": None, "plan": None}
    assert out["ana"] == ["Ana private task"]
    assert out["anon"] == "fd.queue.shared:process:x:0123456789abcdef@local"
    assert out["own"] == "fd.queue.proc-helper"


def test_sign_out_purges_shared_slices_only():
    out = _lift(_SLICE_DECLS, _STORAGE_HARNESS + """
      for (const [k, v] of [
        ['fd.queue.shared:process:x:0123456789abcdef@u-ana', '{}'],
        ['fd.queue.shared:process:x:0123456789abcdef::B@u-ana', '{}'],
        ['fd.queue.shared:process:x:0123456789abcdef@u-bob', '{}'],
        ['fd.plan.shared:docker:y:fedcba9876543210@u-ana', '{}'],
        ['fd.queue.proc-helper', '{}'],
        ['fd.plan.proc-helper', '{}'],
        ['fd.queue.plan.shared:process:x:0123456789abcdef', '{}'],
        ['fd.sharedAgentAck.u-ana.process:x:0123456789abcdef', '1'],
      ]) store.set(k, v);
      purgeSharedSlices();
      out = [...store.keys()].sort();
    """)
    assert out == sorted([
        "fd.queue.proc-helper",
        "fd.plan.proc-helper",
        "fd.queue.plan.shared:process:x:0123456789abcdef",
        "fd.sharedAgentAck.u-ana.process:x:0123456789abcdef",
    ])


def test_leaving_a_shared_agent_drops_its_slices():
    out = _lift(_SLICE_DECLS + [(_SHARED_STORE, ["useSharedAgentStore"])], _STORAGE_HARNESS + f"""
      // A declaration, so it is hoisted above the lifted `create(...)` call.
      function create(init) {{
        let state;
        const set = (p) => {{ state = {{ ...state, ...(typeof p === 'function' ? p(state) : p) }}; }};
        const get = () => state;
        state = init(set, get);
        return {{ getState: get }};
      }}
      const calls = [];
      const leaveShare = async (type, ref, owner) => {{ calls.push(['leaveShare', type, ref, owner]); }};
      const getSharedAgents = async () => {{
        calls.push(['fetch', [...store.keys()].sort()]);
        return {{ enabled: true, host_warning: '', agents: [] }};
      }};
      const sessions = new Map([
        ['shared:' + R, {{ containerId: 'shared:' + R }}],
        ['shared:' + R + '::C', {{ containerId: 'shared:' + R }}],
        ['proc-helper', {{ containerId: 'proc-helper' }}],
      ]);
      const useChatStore = {{ getState: () => ({{
        sessions,
        disconnectChat: (k) => {{ calls.push(['disconnect', k]); sessions.delete(k); }},
      }}) }};
      const cid = sharedContainerId(R);
      for (const lane of ['A', 'B', 'C']) saveQueueSlice(laneKey(cid, lane), {_ANA_QUEUE});
      savePlanSlice(cid, {_PLAN});
      saveQueueSlice(sharedContainerId(R2), {_ANA_QUEUE});
      saveQueueSlice('proc-helper', {_ANA_QUEUE});
      user = {{ id: 'u-bob' }};
      saveQueueSlice(cid, {_ANA_QUEUE});
      user = {{ id: 'u-ana' }};
      done = useSharedAgentStore.getState().leave(R, 'u-olga').then(() => {{
        out = {{ calls, left: [...store.keys()].sort() }};
      }});
    """)
    remaining = sorted([
        "fd.queue.shared:docker:y:fedcba9876543210@u-ana",
        "fd.queue.shared:process:x:0123456789abcdef@u-bob",
        "fd.queue.proc-helper",
    ])
    assert out["left"] == remaining
    assert out["calls"] == [
        ["leaveShare", "agent", "process:x:0123456789abcdef", "u-olga"],
        ["disconnect", "shared:process:x:0123456789abcdef"],
        ["disconnect", "shared:process:x:0123456789abcdef::C"],
        # The slices are gone by the time the list refreshes.
        ["fetch", remaining],
    ]


# ── the member notice is acknowledged per deck user ─────────────────


def test_notice_dismissal_is_per_deck_user():
    out = _lift([
        (_SHARED, ["SHARED_ACK_PREFIX", "sliceUser", "sharedAckKey"]),
        (_NOTICE, ["ackKey", "readAck", "writeAck"]),
    ], _STORAGE_HARNESS + """
      const before = readAck(R);
      writeAck(R);
      const ana = readAck(R);
      user = { id: 'u-bob' };       // next member on this kiosk, no teardown
      const bob = readAck(R);
      user = null;
      const anon = readAck(R);
      user = { id: 'u-ana' };
      const otherAgent = readAck(R2);
      const keys = [...store.keys()];
      // Blocked storage: the notice just shows again, nothing throws.
      window.localStorage = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
      writeAck(R);
      out = { before, ana, bob, anon, otherAgent, keys, blocked: readAck(R) };
    """)
    assert out["keys"] == ["fd.sharedAgentAck.u-ana.process:x:0123456789abcdef"]
    assert out["before"] is False
    assert out["ana"] is True
    assert out["bob"] is False
    assert out["anon"] is False
    assert out["otherAgent"] is False
    assert out["blocked"] is False


def test_purge_runs_on_every_sign_out():
    found = _query(_CHAT_STORE, r"""
      const calls = all.filter((n) => ts.isCallExpression(n)
          && n.expression.getText(sf) === 'registerSignOutTeardown');
      return calls.map((c) => c.arguments[0].getText(sf).includes('purgeSharedSlices()'));
    """)
    assert found == [True]


# ── finding an agent someone shared with you ────────────────────────

_SHARED_ROWS = [
    {"agent_ref": "process:helper:0123456789abcdef", "name": "Helper", "slug": "helper",
     "status": "running", "owner_name": "Olga", "owner_email": "olga@example.com"},
    {"agent_ref": "docker:scout:fedcba9876543210", "name": "", "slug": "scout",
     "status": "stopped", "owner_name": "", "owner_email": "olga@example.com"},
]


def _notice_action(ref_type, ref_id, layout="full"):
    return _lift([(_SHARED, ["sharedAgentWhere", "sharedNotificationAction"])], (
        f"out = sharedNotificationAction({json.dumps(ref_type)}, {json.dumps(ref_id)},"
        f" {json.dumps(_SHARED_ROWS)}, {json.dumps(layout)});"
    ))


def test_share_notice_opens_the_chat_of_a_running_agent():
    out = _notice_action("agent", "process:helper:0123456789abcdef")
    assert out == {
        "kind": "chat", "agentRef": "process:helper:0123456789abcdef", "name": "Helper",
        "ownerName": "Olga",
        "hint": "Find it under “Shared with me” on the Agent Desktop. Click to chat with it.",
    }
    simple = _notice_action("agent", "process:helper:0123456789abcdef", "simple")
    assert simple["kind"] == "chat"
    assert simple["hint"].startswith("Find it in your agents list, marked “shared”.")


def test_share_notice_points_at_a_stopped_agent():
    out = _notice_action("agent", "docker:scout:fedcba9876543210")
    assert out["kind"] == "reveal"
    assert out["name"] == "scout"
    assert out["ownerName"] == "olga@example.com"
    assert out["hint"] == (
        "Find it under “Shared with me” on the Agent Desktop. "
        "It's stopped right now — ask olga@example.com to start it."
    )


def test_other_notices_lead_nowhere():
    # Not an agent, an agent no longer shared with you (revoked, left), no ref.
    assert _notice_action("archetype", "process:helper:0123456789abcdef") == {"kind": "none"}
    assert _notice_action("agent", "process:gone:1111111111111111") == {"kind": "none"}
    assert _notice_action("agent", "") == {"kind": "none"}
    assert _notice_action(None, None) == {"kind": "none"}


def test_server_notifications_keep_what_they_are_about():
    out = _lift([(_NOTIF_STORE, ["isServerId", "mapServerType", "useNotificationStore"])], """
      function create(init) {
        let state;
        const set = (p) => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; };
        const get = () => state;
        state = init(set, get);
        return { getState: get };
      }
      const fetchNotifications = async () => ({ unread: 2, items: [
        { id: 'n1', type: 'share', title: 'Olga shared the agent “Helper” with you', body: 'Helper',
          ref_type: 'agent', ref_id: 'process:helper:0123456789abcdef', read: 0,
          created_at: '2026-10-05T10:00:00Z' },
        { id: 'n2', type: 'run', title: 'Run finished', body: '', ref_type: '', ref_id: '',
          read: 1, created_at: '2026-10-05T09:00:00Z' },
      ] });
      done = useNotificationStore.getState().hydrateFromServer().then(() => {
        out = useNotificationStore.getState().notifications.map((n) => [n.id, n.refType ?? null, n.refId ?? null]);
      });
    """)
    assert out == [
        ["n1", "agent", "process:helper:0123456789abcdef"],
        ["n2", None, None],
    ]


def test_bell_acts_on_shared_agent_notices():
    found = _query(_NOTIF_CENTER, r"""
      const fn = (name) => all.find((n) => (ts.isFunctionDeclaration(n) || ts.isVariableDeclaration(n))
          && n.name && n.name.getText(sf) === name);
      const dropdown = fn('NotificationDropdown').getText(sf);
      const item = fn('NotificationItem');
      const rootClick = all.find((n) => ts.isJsxAttribute(n) && n.name.getText(sf) === 'onClick'
          && n.getStart(sf) > item.getStart(sf) && n.getEnd() <= item.getEnd());
      const activate = fn('activate');
      return {
        actionFromRef: dropdown.includes('sharedNotificationAction(n.refType, n.refId, sharedAgents, layout)'),
        refreshesList: dropdown.includes('useSharedAgentStore.getState().fetch()'),
        rootClick: rootClick.getText(sf),
        activate: activate.getText(sf),
        hint: item.getText(sf).includes('{action.hint}'),
      };
    """)
    assert found["actionFromRef"]
    assert found["refreshesList"]
    assert found["hint"]
    # Clicking the item still marks it read, then acts on it.
    assert "onRead()" in found["rootClick"] and "onActivate(action)" in found["rootClick"]
    assert ".openSharedChat(action.agentRef, action.name, action.ownerName)" in found["activate"]
    assert "setView('desktop')" in found["activate"]
    assert "revealSharedSection()" in found["activate"]


def test_reveal_waits_for_the_desktop_to_mount():
    out = _lift([
        (_SHARED, ["SHARED_SECTION_ID"]),
        (_NOTIF_CENTER, ["revealSharedSection"]),
    ], """
      const scrolled = [];
      let mounted = false;
      const looked = [];
      const document = { getElementById: (id) => {
        looked.push(id);
        return mounted ? { scrollIntoView: (o) => scrolled.push(o) } : null;
      } };
      const pending = [];
      const setTimeout = (fn) => { pending.push(fn); };
      revealSharedSection();
      pending.shift()();            // still mounting
      mounted = true;
      pending.shift()();
      out = { looked: [...new Set(looked)], tries: looked.length, scrolled, left: pending.length };
    """)
    assert out == {
        "looked": ["fd-shared-with-me"], "tries": 3,
        "scrolled": [{"behavior": "smooth", "block": "start"}], "left": 0,
    }


def test_desktop_shows_shared_with_me_above_your_own_agents():
    found = _query(_DESKTOP, r"""
      const section = all.find((n) => ts.isJsxAttribute(n) && n.name.getText(sf) === 'id'
          && n.initializer && n.initializer.getText(sf) === '{SHARED_SECTION_ID}');
      const inJsx = (n) => { for (let p = n.parent; p; p = p.parent) if (ts.isJsxExpression(p)) return true; return false; };
      // The first place the page RENDERS the list (not other uses of it).
      const firstCall = (text) => {
        const c = all.find((n) => ts.isCallExpression(n) && n.expression.getText(sf) === text && inJsx(n));
        return c ? c.getStart(sf) : -1;
      };
      return {
        section: section ? section.getStart(sf) : -1,
        cards: firstCall('sharedAgents.map'),
        own: firstCall('unifiedAgents.map'),
        botport: firstCall('instances.map'),
      };
    """)
    assert found["section"] > 0
    assert found["section"] < found["cards"]
    assert found["cards"] < found["botport"] < found["own"]
