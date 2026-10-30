"""Taskboard UI logic (the kiosk Taskboard), run in Node.

The Flight Deck frontend has no JS test runner, so — like
test_gmail_autoreply_ui.py — these tests lift the named top-level declarations
out of the TypeScript source with the TypeScript compiler from
flight-deck/node_modules, transpile them and run them in a Node vm. JSX runs
against a stub ``React.createElement`` that returns the element tree, so a
component's words and buttons can be read and its handlers called. They skip
when Node or the frontend dependencies aren't installed.

Covered: the board user is "You" and a deck user's id is never shown; a name
the model read out of the text that repeats a recorded party adds nothing;
the agent couldn't do it / partly done / a new message since done, as notes
and Needs me reasons; and the settings' Connect Jev.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FD = Path(__file__).resolve().parents[2] / "flight-deck"
_SRC = _FD / "src"
_TEXT = _SRC / "components" / "taskboard" / "taskboardText.ts"
_BOARD = _SRC / "components" / "taskboard" / "TaskBoard.tsx"
_SETTINGS = _SRC / "components" / "taskboard" / "TaskboardSettings.tsx"
_LAYOUT = _SRC / "components" / "layout" / "SimpleLayout.tsx"
_TS = _FD / "node_modules" / "typescript"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not _TS.is_dir(),
    reason="needs node and flight-deck/node_modules (npm install)",
)

# argv: typescript dir. stdin: {"parts": [{"src", "names"}], "prelude": js,
# "body": js}. Runs prelude + the named declarations (in source order per
# file) + body in a vm, transpiled as TSX with classic React JSX; body sets
# globalThis.out.
_RUNNER = r"""
const ts = require(process.argv[1]);
const fs = require('fs');
const vm = require('vm');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const parts = [];
for (const part of req.parts) {
  const kind = part.src.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(part.src, fs.readFileSync(part.src, 'utf8'),
    ts.ScriptTarget.Latest, true, kind);
  const found = new Set();
  for (const st of sf.statements) {
    let declared = [];
    if (st.name) declared = [st.name.text];
    else if (ts.isVariableStatement(st)) {
      declared = st.declarationList.declarations.map((d) => d.name.getText(sf));
    }
    if (declared.some((n) => part.names.includes(n))) {
      parts.push(st.getText(sf));
      declared.forEach((n) => found.add(n));
    }
  }
  const missing = part.names.filter((n) => !found.has(n));
  if (missing.length) throw new Error('declarations not found: ' + missing.join(', '));
}
const js = ts.transpileModule(
  (req.prelude || '') + '\n' + parts.join('\n') + '\n' + req.body, {
  fileName: 'lifted.tsx',
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
}).outputText;
const ctx = vm.createContext({ exports: {}, out: null });
vm.runInContext(js, ctx);
process.stdout.write(JSON.stringify(ctx.out));
"""


def _run(parts: list[tuple[Path, list[str]]], body: str, prelude: str = ""):
    proc = subprocess.run(
        ["node", "-e", _RUNNER, str(_TS)],
        input=json.dumps({
            "parts": [{"src": str(src), "names": names} for src, names in parts],
            "prelude": prelude, "body": body,
        }),
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# What each helper needs lifted with it.
_FOLD = ["FOLD_EXTRA", "foldChar", "foldText"]
_PEOPLE = [*_FOLD, "isMe", "partyName", "nameKey", "partyAddress", "distinctParties", "partyFacetLabel"]
_NOTES = ["taskNotes", "answeredInChat"]

_ME = {"id": "fe08502fac164cf78a9de46a7c0d11aa", "name": "Ada Admin"}


def _people(body: str):
    return _run([(_TEXT, _PEOPLE)], body)


# ── You, and never a deck user's id ──


def _party(**kw) -> dict:
    return {"role": "requester", "kind": "name", "address": "", "display": "", "verified": 0, **kw}


def test_the_board_user_is_you_with_no_address_line():
    you = _party(kind="fd_user", address=_ME["id"], display="Ada Admin", verified=1, me=True)
    out = _people(f"""
      const p = {json.dumps(you)};
      out = [partyName(p), partyAddress(p), isMe(p)];
    """)
    assert out == ["You", "", True]


def test_you_without_the_servers_flag_by_the_signed_in_id():
    # A server without `me`: an fd_user party with the signed-in user's id.
    you = _party(kind="fd_user", address=_ME["id"], display="Ada Admin", verified=1)
    other = _party(kind="fd_user", address="0" * 32, display="Bo Member", verified=1)
    out = _people(f"""
      const me = {json.dumps(_ME)};
      out = [partyName({json.dumps(you)}, me), partyName({json.dumps(other)}, me),
             partyName({json.dumps(you)}), isMe({{ ...{json.dumps(you)}, me: false }}, me)];
    """)
    # Without a signed-in id the name stands; an explicit `me: false` wins.
    assert out == ["You", "Bo Member", "Ada Admin", False]


def test_a_deck_users_id_is_never_a_name_nor_an_address():
    raw = _party(kind="fd_user", address="0" * 32, display="0" * 32, verified=1)
    bare = _party(kind="fd_user", address="0" * 32, display="", verified=1)
    fallback = _party(kind="fd_user", address="0" * 32, display="You", verified=1, me=False)
    out = _people(f"""
      const me = {json.dumps(_ME)};
      out = [{json.dumps(raw)}, {json.dumps(bare)}, {json.dumps(fallback)}]
        .map((p) => [partyName(p, me), partyAddress(p, me)]);
    """)
    assert out == [["Someone else", ""]] * 3


def test_addresses_show_only_when_they_add_something():
    email = _party(kind="email", address="ana@example.com", display="Ana Horvat", verified=1)
    same = _party(kind="email", address="ana@example.com", display="ana@example.com", verified=1)
    name = _party(kind="name", address="ada admin", display="Ada Admin")
    phone = _party(kind="waid", address="385911234567", display="+385911234567", verified=1)
    out = _people(f"""
      out = [{json.dumps(email)}, {json.dumps(same)}, {json.dumps(name)}, {json.dumps(phone)}]
        .map((p) => partyAddress(p));
    """)
    # A bare name's address is the name folded: never shown beside it.
    assert out == ["ana@example.com", "", "", "385911234567"]


def test_the_people_filter_names_you_and_hides_deck_user_ids():
    out = _people(f"""
      const me = {json.dumps(_ME)};
      out = [
        partyFacetLabel('fd_user:{_ME["id"]}', 'Ada Admin', me),
        partyFacetLabel('fd_user:{"0" * 32}', '{"0" * 32}', me),
        partyFacetLabel('fd_user:{"0" * 32}', 'Bo Member', me),
        partyFacetLabel('email:ana@example.com', 'Ana Horvat', me),
        partyFacetLabel('name:ada admin', '', me),
      ];
    """)
    assert out == ["You", "Someone else", "Bo Member", "Ana Horvat", "ada admin"]


def test_the_people_filter_says_you_by_the_servers_mark_with_sign_in_off():
    """Sign-in off, no task on the page marks you (no id to go by): the
    value the server marks ``me`` still reads "You", one it marks not you
    is never taken for you, and an unmarked one goes by the id as before."""
    out = _people("""
      const nobody = { id: '', name: '' };
      out = [
        partyFacetLabel('fd_user:local', 'You', nobody, true),
        partyFacetLabel('fd_user:local', 'You', nobody),
        partyFacetLabel('fd_user:local', 'You', { id: 'local', name: '' }, false),
        partyFacetLabel('email:ana@example.com', 'Ana', nobody, false),
      ];
    """)
    assert out == ["You", "Someone else", "Someone else", "Ana"]


def test_the_filters_hand_the_servers_mark_to_the_people_label():
    board = _BOARD.read_text()
    assert "(id, label, f) => partyFacetLabel(id, label, me, f.me)" in board
    parts = (_SRC / "components" / "taskboard" / "taskboardParts.tsx").read_text()
    assert "labelOf?.(f.id, f.label, f)" in parts


# ── One person, listed once ──


def test_a_name_read_from_the_text_that_repeats_a_recorded_party_adds_nothing():
    # The live finding: "From: Ada Admin <id> ✓" and the extraction's "From: Ada Admin".
    parties = [
        _party(kind="fd_user", address=_ME["id"], display="Ada Admin", verified=1, me=True),
        _party(kind="name", address="ada admin", display="Ada Admin"),
        _party(role="mentioned", kind="name", address="dubravkin put", display="Dubravkin Put"),
    ]
    out = _people(f"out = distinctParties({json.dumps(parties)}).map((p) => p.display);")
    assert out == ["Ada Admin", "Dubravkin Put"]


def test_names_match_folded_and_you_go_by_your_own_name_too():
    parties = [
        # The ingest's fallback name when it had none: "You".
        _party(kind="fd_user", address=_ME["id"], display="You", verified=1, me=True),
        _party(role="recipient", kind="email", address="", display="ADA  ADMIN"),
        _party(role="mentioned", kind="name", address="duric", display="Đurić"),
        _party(role="mentioned", kind="name", address="duric", display="Duric"),
    ]
    out = _people(f"""
      out = distinctParties({json.dumps(parties)}, {json.dumps(_ME)}).map((p) => p.display);
    """)
    assert out == ["You", "Đurić"]


def test_recorded_parties_stay_unless_the_same_one_twice():
    parties = [
        _party(kind="email", address="ana@example.com", display="Ana", verified=1),
        _party(role="recipient", kind="email", address="ana@work.example", display="Ana", verified=1),
        _party(kind="email", address="ana@example.com", display="Ana", verified=1),
        # Read from the text before the recorded one it repeats: still nothing.
        _party(role="cc", kind="name", address="ana", display="ana"),
    ]
    out = _people(f"out = distinctParties({json.dumps(parties)}).map((p) => p.role + ' ' + p.address);")
    assert out == ["requester ana@example.com", "recipient ana@work.example"]
    assert _people("out = distinctParties(undefined)") == []


def test_the_rows_people_line_says_you_once():
    parties = [
        _party(kind="fd_user", address=_ME["id"], display="Ada Admin", verified=1, me=True),
        _party(kind="name", address="ada admin", display="Ada Admin"),
        _party(role="doer", kind="name", address="captain", display="Captain"),
        _party(role="recipient", kind="email", address="ana@example.com", display="Ana", verified=1),
    ]
    out = _run([(_TEXT, _PEOPLE), (_BOARD, ["partyLine"])],
               f"out = partyLine({json.dumps(parties)}, {json.dumps(_ME)});")
    assert out == "You, Ana"


# ── The agent couldn't do it, partly done, new since done ──


def _notes(task: dict):
    return _run([(_TEXT, _NOTES)], f"out = taskNotes({json.dumps(task)});")


def test_notes_say_the_agent_declined_with_its_blocker():
    out = _notes({"status": "open", "flags": {
        "agent_declined": True, "answered": "no", "blocker": "  No booking system is connected. "}})
    assert out == [{"id": "agent_declined", "label": "Agent couldn't do it",
                    "detail": "No booking system is connected."}]


def test_notes_partly_done_and_declined_wins_over_it():
    assert _notes({"status": "open", "flags": {"partly_done": True, "blocker": "The PDF is missing"}}) == [
        {"id": "partly_done", "label": "Partly done", "detail": "The PDF is missing"}]
    both = _notes({"status": "open", "flags": {"partly_done": True, "agent_declined": True}})
    assert [n["id"] for n in both] == ["agent_declined"]


def test_a_new_message_since_done_shows_only_on_a_done_task():
    assert _notes({"status": "done", "flags": {"new_since_done": True}}) == [
        {"id": "new_since_done", "label": "New message since done", "detail": ""}]
    assert _notes({"status": "open", "flags": {"new_since_done": True}}) == []
    assert _notes({"status": "done", "flags": None}) == []
    assert _notes({"status": "done"}) == []


def test_answered_in_chat_never_for_a_declined_or_partial_reply():
    out = _run([(_TEXT, _NOTES)], """
      out = [answeredInChat({ answered_in_chat: true }), answeredInChat({ answered_in_chat: true, agent_declined: true }),
             answeredInChat({ answered_in_chat: true, partly_done: true }), answeredInChat(undefined), answeredInChat({})];
    """)
    assert out == [True, False, False, False, False]


def _reason(task: dict):
    base = {"status": "open", "kind": "request", "owes_next": "me", "due_at": None, "proposal": None,
            "links": {}, "flags": {}}
    return _run([(_TEXT, _NOTES), (_BOARD, ["needsReason"])],
                f"out = needsReason({json.dumps({**base, **task})});")


def test_needs_me_reason_for_a_declined_task_carries_the_blocker():
    out = _reason({"flags": {"agent_declined": True, "blocker": "No booking system"}})
    assert out == {"text": "Agent couldn't do it", "more": "No booking system"}
    assert _reason({"flags": {"partly_done": True}}) == {"text": "Partly done", "more": ""}


def test_needs_me_reason_keeps_its_order_around_the_new_notes():
    # A waiting proposal and a failure say so first; a declined task beats
    # "Probably done?" and "Your move"; a plain one is still "Your move".
    assert _reason({"proposal": {"id": "a", "status": "awaiting_approval", "title": ""},
                    "flags": {"agent_declined": True}})["text"] == "Approve the next step"
    assert _reason({"status": "failed", "flags": {"agent_declined": True}})["text"] == "It failed — take a look"
    assert _reason({"flags": {"agent_declined": True, "probably_done": True}})["text"] == "Agent couldn't do it"
    assert _reason({}) == {"text": "Your move", "more": ""}
    assert _reason({"owes_next": "agent"}) is None
    # New since done is the done row's own line, never a Needs me reason.
    assert _reason({"owes_next": "agent", "status": "done", "flags": {"new_since_done": True}}) is None


# ── Connect Jev ──

# A React stub: elements as plain trees ({type, props: {children}}), icons as names.
_REACT = """
const React = { Fragment: 'Fragment', createElement: (type, props, ...children) =>
  ({ type, props: { ...(props || {}), children: children.flat() } }) };
const Plug = 'Plug';
function walk(node, fn) {
  if (!node || typeof node !== 'object') return;
  fn(node);
  for (const c of node.props.children || []) walk(c, fn);
}
function text(node) {
  if (node === null || node === undefined || node === false || node === true) return '';
  if (typeof node !== 'object') return String(node);
  return (node.props.children || []).map(text).join('');
}
"""


def _jev(jev: dict, connect: bool, phone: bool = False):
    return _run([(_SETTINGS, ["JevSection"])], f"""
      let clicked = 0;
      const tree = JevSection({{ jev: {json.dumps(jev)}, phone: {json.dumps(phone)},
                                 onConnect: {'() => { clicked++ }' if connect else 'undefined'} }});
      const buttons = [];
      walk(tree, (n) => {{ if (n.type === 'button') buttons.push(n) }});
      buttons.forEach((b) => b.props.onClick());
      out = {{ text: text(tree), buttons: buttons.map((b) => [text(b), b.props.type]), clicked }};
    """, prelude=_REACT)


def test_not_connected_offers_connect_jev_that_opens_connections():
    out = _jev({"connected": False, "source": ""}, connect=True)
    # A button (not the settings form's submit) that calls the board's handler.
    assert out["buttons"] == [["Connect Jev", "button"]]
    assert out["clicked"] == 1
    assert "Not connected" in out["text"]
    assert "connect Jev in Connections" not in out["text"]


def test_without_a_way_there_the_words_say_where_to_go():
    out = _jev({"connected": False, "source": ""}, connect=False)
    assert out["buttons"] == []
    assert "Not connected — connect Jev in Connections" in out["text"]


def test_connected_shows_whose_key_and_no_button():
    own = _jev({"connected": True, "source": "own"}, connect=True)
    team = _jev({"connected": True, "source": "team"}, connect=True)
    assert own["buttons"] == [] and "Connected with your own key" in own["text"]
    assert "Connected with the team key" in team["text"]


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_the_board_wires_connect_jev_into_both_settings_and_names_the_tab():
    board = _source(_BOARD)
    # The desktop popover and the phone sheet both get it; it closes the
    # settings, then asks for Connections on Jev.
    assert board.count("onConnectJev={connectJev}") == 2
    assert ("const connectJev = onOpenConnections && (() => { setSettingsOpen(false); "
            "onOpenConnections('jev') })") in board
    # A held reply still asks for Google (not the click event as the tab).
    assert "onClick={() => onOpenConnections('google')}" in board
    assert "onClick={onOpenConnections}" not in board


def test_the_kiosk_opens_connections_on_the_tab_the_board_asks_for():
    layout = _source(_LAYOUT)
    assert "const taskboardConnections = (tab: ConnectionsTab = 'google') => {" in layout
    assert "if (locked) { openConnections(tab); return }" in layout
    # The full layout's Connections page opens on the tab it remembers.
    assert "localStorage.setItem('fd:connections-tab', tab)" in layout
