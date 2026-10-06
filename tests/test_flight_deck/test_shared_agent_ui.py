"""Shared-agent UI logic (A1: chat-only shared agent; A2: members' own
Google, deep memory and files on process agents), run in Node.

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

A2 (part 3) adds:

* what a member chat can use comes from Flight Deck's `capabilities`, and
  anything but an explicit `true` (a Docker agent, an older deck) is chat-only;
* the member notice tells process-agent members about their own files, deep
  memory and Google (and keeps A1's text, byte for byte, for chat-only
  agents); its dismissal key is bumped so everyone sees it once;
* the member's Google switch: hidden for chat-only agents, a confirm with the
  token-lifetime warning before turning it ON, none for OFF, the store's
  optimistic update rolled back on failure;
* the owner's Share dialog marks members who turned their Google on, and
  the owner's own cards show "shared · N" when Flight Deck reports it.

PR B (context packs) adds: with packs on, the owner's Share note says members
can publish their own context to the agent and that it runs on every turn,
the owner's channels and automations included.
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
_TOGGLE = _FD / "src" / "components" / "agents" / "SharedAgentGoogleToggle.tsx"
_SWITCH_TRACK = _FD / "src" / "components" / "common" / "SwitchTrack.tsx"
_COUNT_BADGE = _FD / "src" / "components" / "agents" / "SharedCountBadge.tsx"
_CHAT_PANEL = _FD / "src" / "components" / "agents" / "ChatPanel.tsx"
_SHARED_CARD = _FD / "src" / "components" / "agents" / "SharedAgentCard.tsx"
_PROCESS_CARD = _FD / "src" / "components" / "agents" / "ProcessCard.tsx"
_CONTAINER_CARD = _FD / "src" / "components" / "agents" / "ContainerCard.tsx"
_SHARE_MODAL = _FD / "src" / "components" / "common" / "ShareModal.tsx"
_SERVER = _ROOT / "captain_claw" / "flight_deck" / "server.py"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # These tests only read sources and run them in a Node vm, but the Node
    # subprocesses inherit this env: never let one see the real ~/.captain-claw.
    monkeypatch.setenv("HOME", str(tmp_path))

# The deck's host-trust warning as Flight Deck sends it (part 0 §9). The UI
# renders whatever FD sends — it is only an argument here.
HOST_TRUST_WARNING = (
    "Anyone on this deck who runs their own shell-capable process agent can act as any agent on "
    "this host, including this one, and can read every user's Flight Deck files at any time. "
    "While a member's message is being answered (up to 20 minutes), they can also use that "
    "member's deep memory and, if the member turned it on, their Google account."
)

# The owner note's own closing paragraph (part 3 §4).
OWNER_HOST_NOTE = (
    "Anyone on this deck who runs their own shell-capable process agent can act as any agent on "
    "this host, including this one, and can read every user's Flight Deck files."
)

# argv: typescript dir. stdin: {"decls": [{"src", "names"}], "body": js, "tsx"}.
# Runs the named top-level declarations + body in a vm; body sets out (and
# may set done to a promise, awaited before out is printed). With "tsx", JSX
# compiles to `React.createElement(...)` — the body stubs `React`.
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
  fileName: req.tsx ? 'lift.tsx' : 'lift.ts',
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
    jsx: ts.JsxEmit.React },
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


def _lift(decls: list[tuple[Path, list[str]]], body: str, *, tsx: bool = False):
    req = {
        "decls": [{"src": str(src), "names": names} for src, names in decls],
        "body": body, "tsx": tsx,
    }
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
    # A2: process-agent members bring THEIR data; Docker members stay chat-only.
    assert "no shell, files, Google, deep memory" not in note
    assert "THEIR Google account" in note
    assert "never yours" in note
    assert "On a Docker agent" in note
    assert "Member chats use your LLM keys." in note
    # An agent started before sharing (or A2) was on can't serve members until
    # it restarts — the owner is the one who can act on that.
    assert (
        "If this agent was running before sharing (or this update) was turned on, restart it "
        "once so members can connect." in note
    )
    assert note.endswith("\n\n" + OWNER_HOST_NOTE)
    assert "read every user's Flight Deck files" in note


def test_member_notice_names_agent_owner_and_appends_the_warning():
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "memberNoticeText"])], (
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
        ['fd.sharedAgentAck.v3.u-ana.process:x:0123456789abcdef', '1'],
      ]) store.set(k, v);
      purgeSharedSlices();
      out = [...store.keys()].sort();
    """)
    assert out == sorted([
        "fd.queue.proc-helper",
        "fd.plan.proc-helper",
        "fd.queue.plan.shared:process:x:0123456789abcdef",
        "fd.sharedAgentAck.v3.u-ana.process:x:0123456789abcdef",
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
    assert out["keys"] == ["fd.sharedAgentAck.v3.u-ana.process:x:0123456789abcdef"]
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


# ══ A2: members' own Google, deep memory and files (part 3) ═══════════

# PR C adds `datastore` (the agent's datastore as a commons, process agents only).
_ALL_CAPS = {"google": True, "deep_memory": True, "files": True, "datastore": True}
_NO_CAPS = {"google": False, "deep_memory": False, "files": False, "datastore": False}


def _a1_notice(agent: str, owner: str) -> str:
    """A1's member notice, byte for byte (chat-only agents keep it)."""
    return (
        f"**{agent} belongs to {owner}.** Your conversations here are private from other members, "
        f"but not from {owner} or this deck's admins. The agent can see the profile you set in "
        "Flight Deck. What it learns from your chats becomes shared knowledge for everyone using "
        f"it, including {owner}'s own chats with it. In shared chats it can only search the web, "
        "read public pages and use its shared insights, playbooks and topics."
    )


def _a2_notice(agent: str, owner: str) -> str:
    """The process-agent member notice (part 3 §4)."""
    return (
        f"**{agent} belongs to {owner}.** Your conversations here are private from other members, "
        f"but not from {owner} or this deck's admins. The agent can see the profile you set in "
        "Flight Deck. What it learns from your chats becomes shared knowledge for everyone using "
        f"it, including {owner}'s own chats with it — and that includes anything it reads from "
        "your mail, calendar, Drive, files or deep memory during a chat. During your chats it "
        "can read, change and delete your own files (your VFS folders) and use your deep memory "
        f"— never {owner}'s — and it uses your Google account, including your Drive folders in "
        "Flight Deck, only if you turn that on below (Drive files you indexed into deep memory "
        "can still turn up in its deep-memory searches with Google off). While it is answering "
        "you (up to 20 minutes), "
        f"{owner}'s agent process acts with those; your VFS folders are files on this computer, "
        f"so {owner}, who controls the agent, can read them at any time. It can't run commands, "
        "use MCP servers or other agents, or schedule anything."
    )


_OWNER_NOTE = (
    "Members chat with this agent in their own private conversations — private from each other, "
    "not from you or the deck's admins. On a process agent, during a member's chat the agent "
    "works with THEIR deep memory, THEIR own files and, only if they turn it on, THEIR Google "
    "account — never yours. On a Docker agent, member chats can only search the web, read public "
    "pages and use the shared insights, playbooks and topics. In member chats it can't use the "
    "shell, your files or accounts, MCP servers, scheduled jobs or your fleet. What it learns "
    "from anyone's chats — yours, and what it reads from members' own mail, files and deep "
    "memory — becomes shared knowledge for everyone using it. Member chats use your LLM keys. "
    "If this agent was running before sharing (or this update) was turned on, restart it once "
    "so members can connect.\n\n" + OWNER_HOST_NOTE
)

_GOOGLE_WARNING = (
    "Let Helper use your Google account (Gmail, Calendar, Drive) during your chats with it?\n\n"
    "It acts as you, through Olga's agent: while one of your messages is being answered (up to "
    "20 minutes), that agent can read your mail, calendar and Drive — including the Drive "
    "folders you added to Flight Deck — and act in them as you: write drafts and, if your Gmail "
    "send settings allow it, send email as you; create, change or delete calendar events; and "
    "upload or overwrite Drive files. A Google access token it gets covers everything you "
    "allowed when you connected Google and stays valid for about an hour, so Olga, who controls "
    "the agent, could keep using your account for up to about an hour and a half after your "
    "message.\n\n"
    "Anything the agent reads from your mail, calendar or Drive can also become shared "
    "knowledge that other members and Olga see.\n\n"
    "Turn this on only if you trust Olga. You can turn it off at any time."
)

_CONNECT_HINT = "Connect your Google account in Connections first — until then this has no effect."


# ── capabilities: only an explicit true counts ──────────────────────


def test_shared_caps_take_only_an_explicit_true():
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "sharedCaps"])], """
      out = {
        none: sharedCaps(undefined),
        empty: sharedCaps({}),
        nulled: sharedCaps({ capabilities: null }),
        partial: sharedCaps({ capabilities: { google: true, files: 'yes' } }),
        truthy: sharedCaps({ capabilities: { google: 'true', deep_memory: 1, files: {}, datastore: 'yes' } }),
        docker: sharedCaps({ runtime: 'docker', capabilities: { google: false, deep_memory: false, files: false, datastore: false } }),
        process: sharedCaps({ capabilities: { google: true, deep_memory: true, files: true, datastore: true } }),
        // An A2-era Flight Deck says nothing about the datastore: not offered.
        a2Deck: sharedCaps({ capabilities: { google: true, deep_memory: true, files: true } }),
        constant: CHAT_ONLY_CAPS,
      };
    """)
    for key in ("none", "empty", "nulled", "truthy", "docker", "constant"):
        assert out[key] == _NO_CAPS, key
    assert out["partial"] == {"google": True, "deep_memory": False, "files": False, "datastore": False}
    assert out["process"] == _ALL_CAPS
    assert out["a2Deck"] == {**_ALL_CAPS, "datastore": False}


# ── the member notice ───────────────────────────────────────────────


def test_member_notice_on_a_process_agent_tells_what_it_uses():
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "memberNoticeText"])], f"""
      const caps = {json.dumps(_ALL_CAPS)};
      out = {{
        withWarning: memberNoticeText('Helper', 'Olga', {json.dumps(HOST_TRUST_WARNING)}, caps),
        without: memberNoticeText('Helper', 'Olga', '', caps),
        defaults: memberNoticeText('  ', '', '', caps),
      }};
    """)
    text = out["without"]
    for want in (
        "your own files", "deep memory", "only if you turn that on",
        "anything it reads from your mail", "can read them at any time", "Olga",
    ):
        assert want in text, want
    assert text == _a2_notice("Helper", "Olga")
    # The deck's warning comes last, verbatim; none → no trailing blank paragraph.
    assert out["withWarning"] == text + "\n\n" + HOST_TRUST_WARNING
    assert "\n" not in text
    assert out["defaults"] == _a2_notice("This agent", "another user")


def test_member_notice_is_a2_when_any_one_capability_is_on():
    # Part 3 §4: the A2 text whenever files OR deep memory OR Google is on —
    # any single one means the member's own data can be in play.
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "memberNoticeText"])], """
      const one = (k) => memberNoticeText('Helper', 'Olga', '',
        { google: false, deep_memory: false, files: false, [k]: true });
      out = { google: one('google'), deep_memory: one('deep_memory'), files: one('files') };
    """)
    for key, text in out.items():
        assert text == _a2_notice("Helper", "Olga"), key


def test_member_notice_stays_a1_for_chat_only_agents():
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "sharedCaps", "memberNoticeText"])], f"""
      const w = {json.dumps(HOST_TRUST_WARNING)};
      out = {{
        three: memberNoticeText('Helper', 'Olga', w),
        chatOnly: memberNoticeText('Helper', 'Olga', w, CHAT_ONLY_CAPS),
        docker: memberNoticeText('Helper', 'Olga', w,
          sharedCaps({{ runtime: 'docker', capabilities: {json.dumps(_NO_CAPS)} }})),
        olderDeck: memberNoticeText('Helper', 'Olga', w, sharedCaps({{}})),
        noRow: memberNoticeText('Helper', 'Olga', w, sharedCaps(undefined)),
        bare: memberNoticeText('Helper', 'Olga', ''),
      }};
    """)
    a1 = _a1_notice("Helper", "Olga")
    assert out["bare"] == a1
    for key in ("three", "chatOnly", "docker", "olderDeck", "noRow"):
        assert out[key] == a1 + "\n\n" + HOST_TRUST_WARNING, key


def test_ack_key_is_bumped_so_everyone_sees_the_a2_notice():
    out = _lift([(_SHARED, ["SHARED_ACK_PREFIX", "sliceUser", "sharedAckKey"])], """
      out = { prefix: SHARED_ACK_PREFIX, key: sharedAckKey('u1', 'process:x:0123456789abcdef'),
              anon: sharedAckKey(null, 'process:x:0123456789abcdef') };
    """)
    assert out["prefix"] == "fd.sharedAgentAck.v3."
    assert out["key"] == "fd.sharedAgentAck.v3.u1.process:x:0123456789abcdef"
    assert out["anon"] == "fd.sharedAgentAck.v3.local.process:x:0123456789abcdef"
    # A dismissal of A1's (or A2's) notice doesn't hide PR C's.
    assert not out["key"].startswith("fd.sharedAgentAck.u1.")
    assert not out["key"].startswith("fd.sharedAgentAck.v2.")


# ── the Google opt-in texts ─────────────────────────────────────────


def test_google_opt_in_label_and_warning():
    out = _lift([(_SHARED, ["agentInSentence", "googleOptInLabel", "googleOptInWarning"])], """
      out = { label: googleOptInLabel('Helper'), warning: googleOptInWarning('Helper', 'Olga'),
              blankLabel: googleOptInLabel(''), blankWarning: googleOptInWarning('', '') };
    """)
    assert out["label"] == "Let Helper use my Google during my chats"
    warning = out["warning"]
    for want in (
        "Olga", "about an hour and a half", "send email as you", "shared knowledge",
        "Turn this on only if you trust Olga",
    ):
        assert want in warning, want
    assert warning == _GOOGLE_WARNING
    assert out["blankLabel"] == "Let this agent use my Google during my chats"
    assert "through another user's agent" in out["blankWarning"]


def test_google_opt_in_hint_until_google_is_connected():
    out = _lift([(_SHARED, ["googleOptInHint"])], """
      out = [googleOptInHint({ google_connected: false }), googleOptInHint({ google_connected: true }),
             googleOptInHint({}), googleOptInHint({ google_enabled: true })];
    """)
    assert out == [_CONNECT_HINT, "", _CONNECT_HINT, _CONNECT_HINT]


# ── what the owner is told ──────────────────────────────────────────


def test_owner_note_is_the_a2_text():
    note = _lift([(_SHARED, ["OWNER_SHARE_NOTE"])], "out = OWNER_SHARE_NOTE;")
    assert note == _OWNER_NOTE


# PR B: with context packs on, the owner is told — before the host-trust
# paragraph — that members can publish their own context to the agent and that
# it runs on every turn, the owner's channels and automations included.
_OWNER_PACKS_NOTE = (
    "Members can also share their own profile, folders and deep memory with this agent. That "
    "is used on every turn, including your channels and automations. You're notified and can "
    "remove any of it under Shared context."
)


def test_owner_note_with_context_packs_tells_the_owner_about_members_packs():
    out = _lift([(_SHARED, ["OWNER_SHARE_NOTE", "ownerShareNote"])], """
      out = { off: ownerShareNote(false), offDocker: ownerShareNote(false, 'docker'),
              process: ownerShareNote(true, 'process'), dflt: ownerShareNote(true),
              docker: ownerShareNote(true, 'docker') };
    """)
    assert out["off"] == _OWNER_NOTE and out["offDocker"] == _OWNER_NOTE
    main = _OWNER_NOTE[: -len("\n\n" + OWNER_HOST_NOTE)]
    assert out["process"] == main + "\n\n" + _OWNER_PACKS_NOTE + "\n\n" + OWNER_HOST_NOTE
    assert out["dflt"] == out["process"]
    # A Docker agent takes members' profiles only.
    assert out["docker"] == out["process"].replace(
        "their own profile, folders and deep memory", "their own profile")
    for note in (out["process"], out["docker"]):
        assert note.endswith("\n\n" + OWNER_HOST_NOTE)


def test_owner_badge_texts():
    out = _lift([(_SHARED, ["ownerGoogleBadgeTitle", "sharedCountLabel", "sharedCountTitle"])], """
      out = { title: ownerGoogleBadgeTitle('Ana'), blank: ownerGoogleBadgeTitle(''),
              label: sharedCountLabel(3),
              titles: [sharedCountTitle(1, 0), sharedCountTitle(3, 0), sharedCountTitle(3, 2)] };
    """)
    assert out["title"] == "Ana lets this agent use their Google account during their chats"
    assert out["blank"] == "This member lets this agent use their Google account during their chats"
    assert out["label"] == "shared · 3"
    assert out["titles"] == [
        "Shared with 1 member", "Shared with 3 members", "Shared with 3 members, 2 with Google on",
    ]


# ── the member's Google switch, rendered against stubs ──────────────

_REF = "process:helper:0123456789abcdef"

_TOGGLE_HARNESS = r"""
const React = { createElement: (type, props, ...children) =>
  ({ type, props: props || {}, children: children.flat(Infinity) }) };
const states = [];
const useState = (init) => [typeof init === 'function' ? init() : init, (v) => states.push(v)];
const Loader2 = 'Loader2';
const confirms = [];
let answer = true;
const window = { confirm: (m) => { confirms.push(m); return answer; } };
const calls = [];
let fail = '';
const useSharedAgentStore = { getState: () => ({ setGoogle: async (ref, on) => {
  calls.push([ref, on]); if (fail) throw new Error(fail);
} }) };
const notes = [];
const useNotificationStore = { getState: () => ({ add: (...a) => notes.push(a) }) };
const find = (n, pred) => {
  if (!n || typeof n !== 'object') return null;
  if (pred(n)) return n;
  for (const c of n.children || []) { const f = find(c, pred); if (f) return f; }
  return null;
};
const text = (n) => (n == null || n === false || n === true) ? ''
  : typeof n === 'object' ? (n.children || []).map(text).join('') : String(n);
const row = (over) => ({
  agent_ref: 'process:helper:0123456789abcdef', runtime: 'process', slug: 'helper', name: 'Helper',
  description: '', status: 'running', owner_id: 'u-olga', owner_name: 'Olga',
  owner_email: 'olga@example.com', shared_at: '',
  capabilities: { google: true, deep_memory: true, files: true },
  google_enabled: false, google_connected: true, ...over,
});
const switchOf = (el) => find(el, (n) => n.props && n.props.role === 'switch');
"""

_TOGGLE_DECLS = [
    (_SHARED, [
        "sharedCaps", "agentInSentence", "googleOptInLabel", "googleOptInWarning", "googleOptInHint",
    ]),
    (_SWITCH_TRACK, ["SwitchTrack"]),
    (_TOGGLE, ["GoogleMark", "SharedAgentGoogleToggle"]),
]


def _toggle(body: str):
    return _lift(_TOGGLE_DECLS, _TOGGLE_HARNESS + "done = (async () => {" + body + "})();", tsx=True)


def test_google_toggle_is_hidden_for_chat_only_agents():
    out = _toggle("""
      out = {
        docker: SharedAgentGoogleToggle({ agent: row({ runtime: 'docker',
          capabilities: { google: false, deep_memory: false, files: false } }) }),
        olderDeck: SharedAgentGoogleToggle({ agent: row({ capabilities: undefined }) }),
        compactNull: SharedAgentGoogleToggle({ agent: row({ capabilities: null }), compact: true }),
        process: !!SharedAgentGoogleToggle({ agent: row({}) }),
      };
    """)
    assert out == {"docker": None, "olderDeck": None, "compactNull": None, "process": True}


def test_google_toggle_on_asks_first_and_off_does_not():
    out = _toggle("""
      // Turning ON, then cancelling the warning: nothing happens.
      answer = false;
      let sw = switchOf(SharedAgentGoogleToggle({ agent: row({}) }));
      const label = sw.props['aria-label'];
      const checkedOff = sw.props['aria-checked'];
      await sw.props.onClick();
      const cancelled = { confirms: confirms.length, calls: calls.length, states: states.length };
      // Confirmed: the store is asked, busy around the call.
      answer = true;
      await sw.props.onClick();
      const turnedOn = { calls: [...calls], states: [...states] };
      // Turning OFF never asks.
      confirms.length = 0; calls.length = 0;
      sw = switchOf(SharedAgentGoogleToggle({ agent: row({ google_enabled: true }) }));
      const checkedOn = sw.props['aria-checked'];
      await sw.props.onClick();
      out = { label, checkedOff, checkedOn, cancelled, turnedOn, offConfirms: confirms.length,
              offCalls: calls };
    """)
    assert out["label"] == "Let Helper use my Google during my chats"
    assert out["checkedOff"] is False and out["checkedOn"] is True
    assert out["cancelled"] == {"confirms": 1, "calls": 0, "states": 0}
    assert out["turnedOn"] == {"calls": [[_REF, True]], "states": [True, False]}
    assert out["offConfirms"] == 0
    assert out["offCalls"] == [[_REF, False]]


def test_google_toggle_confirm_shows_the_warning():
    out = _toggle("""
      answer = false;
      await switchOf(SharedAgentGoogleToggle({ agent: row({}) })).props.onClick();
      await switchOf(SharedAgentGoogleToggle({ agent: row({ name: '', owner_name: '' }) })).props.onClick();
      out = confirms;
    """)
    assert out[0] == _GOOGLE_WARNING
    # Falls back to the slug and the owner's email, as the card does.
    assert out[1].startswith("Let helper use your Google account")
    assert "through olga@example.com's agent" in out[1]


def test_google_toggle_reports_a_failure():
    out = _toggle("""
      fail = 'Not a member';
      await switchOf(SharedAgentGoogleToggle({ agent: row({ google_enabled: true }) })).props.onClick();
      out = { notes, states };
    """)
    assert out["notes"] == [["error", "Could not change Google access", "Not a member"]]
    # Not left spinning.
    assert out["states"] == [True, False]


def test_google_toggle_hint_until_google_is_connected():
    out = _toggle("""
      const full = SharedAgentGoogleToggle({ agent: row({ google_connected: false }) });
      const compact = SharedAgentGoogleToggle({ agent: row({ google_connected: false }), compact: true });
      const connected = SharedAgentGoogleToggle({ agent: row({}) });
      const compactConnected = SharedAgentGoogleToggle({ agent: row({}), compact: true });
      out = {
        full: text(full), compactTitle: compact.props.title, compactText: text(compact),
        connected: text(connected), compactConnectedTitle: compactConnected.props.title ?? null,
        hintClass: (find(full, (n) => n.type === 'p') || { props: {} }).props.className,
      };
    """)
    assert _CONNECT_HINT in out["full"]
    assert out["hintClass"] == "text-[11px] text-zinc-500"
    assert out["compactTitle"] == _CONNECT_HINT
    assert _CONNECT_HINT not in out["compactText"]
    assert _CONNECT_HINT not in out["connected"]
    assert out["compactConnectedTitle"] is None



def test_compact_google_switch_is_one_labelled_pill():
    out = _toggle("""
      const off = SharedAgentGoogleToggle({ agent: row({}), compact: true });
      const on = SharedAgentGoogleToggle({ agent: row({ google_enabled: true }), compact: true });
      const swOff = switchOf(off), swOn = switchOf(on);
      // The pill's own words (its first text span) — what a voice-control user says.
      const words = (sw) => text(find(sw, (n) => n.type === 'span' && typeof n.children[0] === 'string'));
      out = {
        offText: text(swOff), onText: text(swOn),
        offWords: words(swOff), onWords: words(swOn), offLabel: swOff.props['aria-label'],
        offChecked: swOff.props['aria-checked'], onChecked: swOn.props['aria-checked'],
        label: swOn.props['aria-label'], title: swOn.props.title,
        // The track inside the pill shows the same state.
        track: [find(swOff, (n) => n.type === SwitchTrack).props.on,
                find(swOn, (n) => n.type === SwitchTrack).props.on],
        mark: !!find(swOn, (n) => n.type === GoogleMark),
        hintTitle: switchOf(SharedAgentGoogleToggle({ agent: row({ google_connected: false }),
                                                       compact: true })).props.title,
      };
    """)
    # The agent's name lives in the accessible name and tooltip, not the pill;
    # the words don't change with the state (aria-checked and the track do).
    assert out["offText"] == out["onText"] == "Use my GoogleGmail · Calendar · Drive"
    # WCAG 2.5.3: the visible label is part of the accessible name, on and off.
    assert out["offWords"] == out["onWords"] == "Use my Google"
    for words, name in ((out["offWords"], out["offLabel"]), (out["onWords"], out["label"])):
        assert words.lower() in name.lower()
    assert (out["offChecked"], out["onChecked"]) == (False, True)
    assert out["label"] == out["title"] == "Let Helper use my Google during my chats"
    assert out["track"] == [False, True]
    assert out["mark"] is True
    assert out["hintTitle"] == _CONNECT_HINT


def test_full_google_switch_keeps_the_sentence_and_track():
    out = _toggle("""
      const sw = switchOf(SharedAgentGoogleToggle({ agent: row({ google_enabled: true }) }));
      out = { text: text(sw), track: find(sw, (n) => n.type === SwitchTrack).props.on };
    """)
    assert out == {"text": "Let Helper use my Google during my chats", "track": True}


def test_switch_knob_stays_on_its_track():
    # Regression: the knob had no left inset, so inside a <button> it started at
    # the button's centre and the "on" knob slid off the track over the label.
    out = _lift([(_SWITCH_TRACK, ["SwitchTrack"])], _TOGGLE_HARNESS + """
      const cls = (on) => { const t = SwitchTrack({ on }); return [t.props, t.children[0].props.className]; };
      const [offTrack, offKnob] = cls(false), [onTrack, onKnob] = cls(true);
      out = { hidden: offTrack['aria-hidden'], offTrack: offTrack.className, onTrack: onTrack.className,
              offKnob, onKnob };
    """, tsx=True)
    assert out["hidden"] == "true"
    for knob in (out["offKnob"], out["onKnob"]):
        assert "absolute" in knob.split() and "left-0.5" in knob.split()
    # h-4 w-7 track (28px), h-3 w-3 knob (12px): left 2px, on = 2 + 12 → 2px each side.
    assert "translate-x-0" in out["offKnob"].split() and "translate-x-3" in out["onKnob"].split()
    assert "h-4" in out["onTrack"].split() and "w-7" in out["onTrack"].split()
    assert "bg-sky-500" in out["onTrack"].split() and "bg-zinc-600" in out["offTrack"].split()


def test_every_member_switch_uses_the_shared_track():
    quality = _FD / "src" / "components" / "QualityControls.tsx"
    packs = _FD / "src" / "components" / "agents" / "ContextPacksModal.tsx"
    for src in (_TOGGLE, quality, packs):
        body = src.read_text(encoding="utf-8")
        assert "<SwitchTrack on={on} />" in body, src.name
        # No hand-rolled knob left behind.
        assert "translate-x-3.5" not in body and "translate-x-0.5" not in body, src.name


# ── the store: optimistic switch, rolled back on failure; owner counts ──

_STORE_HARNESS = r"""
function create(init) {
  let state;
  const set = (p) => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; };
  const get = () => state;
  state = init(set, get);
  return { getState: get };
}
const A = 'process:helper:0123456789abcdef';
const B = 'docker:scout:fedcba9876543210';
let server = {
  enabled: true, host_warning: 'W',
  agents: [
    { agent_ref: A, name: 'Helper', google_enabled: false, google_connected: true,
      capabilities: { google: true, deep_memory: true, files: true } },
    { agent_ref: B, name: 'Scout', google_enabled: false, google_connected: true,
      capabilities: { google: false, deep_memory: false, files: false } },
  ],
  mine: { 'process:mine:1111111111111111': { members: 2, google: 1 } },
};
const getSharedAgents = async () => JSON.parse(JSON.stringify(server));
const puts = [];
const seen = [];
let fail = '';
const setSharedAgentGoogle = async (ref, on) => {
  seen.push(useSharedAgentStore.getState().agents.find((a) => a.agent_ref === ref).google_enabled);
  puts.push([ref, on]);
  if (fail) throw new Error(fail);
  server.agents = server.agents.map((a) => (a.agent_ref === ref ? { ...a, google_enabled: on } : a));
  return { agent_ref: ref, google_enabled: on };
};
const leaveShare = async () => {};
const sharedContainerId = (r) => 'shared:' + r;
const clearSharedSlices = () => {};
const useChatStore = { getState: () => ({ sessions: new Map(), disconnectChat: () => {} }) };
const st = () => useSharedAgentStore.getState();
const g = (ref) => st().agents.find((a) => a.agent_ref === ref).google_enabled;
"""


def _store(body: str):
    return _lift(
        [(_SHARED_STORE, ["useSharedAgentStore"])],
        _STORE_HARNESS + "done = (async () => {" + body + "})();",
    )


def test_store_set_google_is_optimistic_and_rolls_back():
    out = _store("""
      await st().fetch();
      await st().setGoogle(A, true);
      const on = { seen: [...seen], g: g(A), other: g(B) };
      fail = 'Denied';
      let err = null;
      try { await st().setGoogle(A, false); } catch (e) { err = e.message; }
      out = { on, err, seenOff: seen[seen.length - 1], after: g(A), other: g(B), puts };
    """)
    # The switch flips before the PUT answers, then the list is re-read.
    assert out["on"] == {"seen": [True], "g": True, "other": False}
    # A failed PUT puts the old value back and rethrows for the caller.
    assert out["err"] == "Denied"
    assert out["seenOff"] is False
    assert out["after"] is True
    assert out["other"] is False
    assert out["puts"] == [[_REF, True], [_REF, False]]


def test_store_keeps_owner_counts_and_their_identity():
    out = _store("""
      const initial = st().mine;
      await st().fetch();
      const first = st().mine;
      await st().fetch();
      const stable = st().mine === first;
      server = { ...server, mine: undefined };
      await st().fetch();
      const absent = st().mine;
      server = { ...server, enabled: false, mine: { x: { members: 1, google: 0 } } };
      await st().fetch();
      out = { initial, first, stable, absent, off: st().mine };
    """)
    assert out["initial"] == {}
    assert out["first"] == {"process:mine:1111111111111111": {"members": 2, "google": 1}}
    assert out["stable"] is True
    # An older Flight Deck (no `mine`) or sharing off: no counts, no badge.
    assert out["absent"] == {}
    assert out["off"] == {}


def test_owner_count_badge_shows_only_with_members():
    out = _lift([
        (_SHARED, ["sharedCountLabel", "sharedCountTitle"]),
        (_COUNT_BADGE, ["SharedCountBadge"]),
    ], r"""
      const React = { createElement: (type, props, ...children) =>
        ({ type, props: props || {}, children: children.flat(Infinity) }) };
      const Users = 'Users';
      let state = { mine: {} };
      const useSharedAgentStore = (sel) => sel(state);
      const text = (n) => (n == null || n === false) ? ''
        : typeof n === 'object' ? (n.children || []).map(text).join('') : String(n);
      const none = SharedCountBadge({ agentRef: 'process:mine:1111111111111111' });
      const noRef = SharedCountBadge({});
      state = { mine: { 'process:mine:1111111111111111': { members: 2, google: 1 },
                        'process:zero:2222222222222222': { members: 0, google: 0 } } };
      const two = SharedCountBadge({ agentRef: 'process:mine:1111111111111111' });
      const zero = SharedCountBadge({ agentRef: 'process:zero:2222222222222222' });
      out = { none, noRef, zero, label: text(two), title: two.props.title };
    """, tsx=True)
    assert out["none"] is None and out["noRef"] is None and out["zero"] is None
    assert out["label"] == "shared · 2"
    assert out["title"] == "Shared with 2 members, 1 with Google on"


# ── wiring ──────────────────────────────────────────────────────────

_JSX_HELPERS = r"""
const tag = (n) => (ts.isJsxSelfClosingElement(n) ? n : ts.isJsxElement(n) ? n.openingElement : null);
const tagName = (n) => { const t = tag(n); return t ? t.tagName.getText(sf) : null; };
const attr = (n, name) => {
  const t = tag(n);
  const a = t && t.attributes.properties.find((p) => ts.isJsxAttribute(p) && p.name.getText(sf) === name);
  return a ? (a.initializer ? a.initializer.getText(sf) : true) : null;
};
const guard = (n) => {
  for (let p = n.parent; p; p = p.parent) if (ts.isJsxExpression(p)) return p.expression.getText(sf);
  return null;
};
const named = (name) => all.filter((n) => tagName(n) === name);
"""


def test_chat_panel_wires_caps_notice_and_switch():
    found = _query(_CHAT_PANEL, _JSX_HELPERS + r"""
      const notices = named('SharedAgentNotice');
      const toggles = named('SharedAgentGoogleToggle');
      const decl = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
      const storeCalls = all.filter((n) => ts.isCallExpression(n)
          && n.expression.getText(sf) === 'useSharedAgentStore');
      const closeAt = sf.getFullText().indexOf('sharedClose.message');
      const chip = all.find((n) => ts.isJsxAttribute(n) && n.name.getText(sf) === 'title'
          && n.getText(sf).includes('Your chats are private from other members'));
      return {
        noticeCaps: notices.map((n) => attr(n, 'caps')),
        toggles: toggles.map((n) => ({ agent: attr(n, 'agent'), compact: attr(n, 'compact'), guard: guard(n) })),
        order: notices.length === 1 && toggles.length === 1
          && notices[0].getStart(sf) < toggles[0].getStart(sf) && toggles[0].getStart(sf) < closeAt,
        caps: decl('caps') ? decl('caps').initializer.getText(sf) : null,
        sharedRow: decl('sharedRow') ? decl('sharedRow').initializer.getText(sf) : null,
        storeArgs: storeCalls.map((c) => c.arguments.length),
        chip: chip ? chip.getText(sf) : null,
      };
    """)
    assert found["noticeCaps"] == ["{caps}"]
    assert len(found["toggles"]) == 1
    t = found["toggles"][0]
    assert t["agent"] == "{sharedRow}" and t["compact"] is True
    for cond in ("shared", "sharedRow", "caps.google", "!session.closed"):
        assert cond in t["guard"], cond
    assert found["order"] is True
    assert found["caps"] == "sharedCaps(sharedRow)"
    # Only this chat's row — never a whole-store subscription.
    assert "s.agents.find(" in found["sharedRow"]
    assert found["storeArgs"] and all(n == 1 for n in found["storeArgs"])
    assert "caps.files || caps.deep_memory" in found["chip"]
    assert (
        " During your chats it uses your own files and deep memory, and your Google only if you "
        "turned it on." in found["chip"]
    )


def test_cards_and_share_dialog_wiring():
    card = _query(_SHARED_CARD, _JSX_HELPERS + r"""
      const text = sf.getFullText();
      const desc = text.indexOf('{agent.description}</p>');
      const chat = text.indexOf('openSharedChat(agent.agent_ref');
      return named('SharedAgentGoogleToggle').map((n) => ({ agent: attr(n, 'agent'),
        compact: attr(n, 'compact'), between: desc > 0 && desc < n.getStart(sf) && n.getStart(sf) < chat }));
    """)
    assert card == [{"agent": "{agent}", "compact": None, "between": True}]

    modal = _query(_SHARE_MODAL, _JSX_HELPERS + r"""
      const spans = all.filter((n) => ts.isJsxElement(n) && tagName(n) === 'span'
          && n.children.some((c) => ts.isJsxText(c) && c.getText(sf).trim() === 'Google on'));
      const canChat = sf.getFullText().indexOf('>Can chat<');
      return spans.map((n) => ({ guard: guard(n), title: attr(n, 'title'),
        beforeCanChat: n.getStart(sf) < canChat }));
    """)
    assert len(modal) == 1
    assert "resourceType === 'agent'" in modal[0]["guard"]
    assert "s.google_enabled" in modal[0]["guard"]
    assert modal[0]["title"] == "{ownerGoogleBadgeTitle(s.grantee_name || s.grantee_email)}"
    assert modal[0]["beforeCanChat"] is True

    for src, ref in ((_PROCESS_CARD, "{proc.agent_ref}"), (_CONTAINER_CARD, "{container.agent_ref}")):
        badges = _query(src, _JSX_HELPERS + r"""
          return named('SharedCountBadge').map((n) => attr(n, 'agentRef'));
        """)
        # Compact and expanded views, next to the name.
        assert badges == [ref, ref], src.name
