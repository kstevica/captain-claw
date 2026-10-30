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
and Needs me reasons; and the settings' Connect Jev. v2: one People row per
person with every handle, a person's People filter value by their name, a
next step's kind and recipient in words, and the ``?taskboard=`` links — the
one the address opens, the bell line it is taken out of.
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
_STORE = _SRC / "stores" / "taskboardStore.ts"
_BELL = _SRC / "components" / "common" / "NotificationCenter.tsx"
_THREADS = _SRC / "components" / "taskboard" / "TaskThreads.tsx"
_API = _SRC / "services" / "taskboardApi.ts"
_PARTS = _SRC / "components" / "taskboard" / "taskboardParts.tsx"
_NEXT = _SRC / "components" / "taskboard" / "TaskNextStep.tsx"
_UI = _SRC / "stores" / "uiStore.ts"
_APP = _SRC / "App.tsx"
_NOTIFY = _SRC / "services" / "notifications.ts"
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
const ctx = vm.createContext({ exports: {}, out: null, URL });
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


def test_a_hand_over_that_isnt_borne_out_says_so_on_an_open_task():
    note = {"state": "not_done", "note": "It says it sent the email, but no send is on record."}
    assert _notes({"status": "open", "flags": {"proof": note}}) == [
        {"id": "not_proven", "label": "Not proven", "detail": note["note"]}]
    assert _notes({"status": "open", "flags": {"proof": {"state": "failed", "note": "x"}}})[0]["label"] == "Didn't go through"
    assert _notes({"status": "open", "flags": {"proof": {"state": "no_record"}}})[0]["label"] == "Can't check"
    assert _notes({"status": "done", "flags": {"proof": note}}) == []    # the user's word closed it
    for fine in ("proven", "recorded", "answered"):
        assert _notes({"status": "open", "flags": {"proof": {"state": fine}}}) == []
    assert _notes({"status": "open", "flags": {"proof": None}}) == []


def test_each_action_says_what_stands_behind_it():
    out = _run([(_TEXT, ["proofRow"])], """
      out = [proofRow({ outcome: 'ok', seen: 'gmail' }), proofRow({ outcome: 'ok', seen: 'calendar' }),
             proofRow({ outcome: 'ok', seen: '' }), proofRow({ outcome: 'ok', seen: 'gone' }),
             proofRow({ outcome: 'error', reason: 'Email sending is off' }), proofRow({ outcome: 'timeout' })];
    """)
    assert [r["good"] for r in out] == [True, True, True, False, False, False]
    assert [r["detail"] for r in out] == [
        "Gmail has it", "Google Calendar has it", "Recorded by the tool",
        "Recorded, but your Google account doesn't have it", "Email sending is off", "It didn't go through"]


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
    # A reply brought it back: said once the rest has had its say, and never off the user's move.
    assert _reason({"flags": {"came_back": True}}) == {"text": "New reply", "more": ""}
    assert _reason({"flags": {"came_back": True, "probably_done": True}})["text"] == "Probably done?"
    assert _reason({"owes_next": "them", "flags": {"came_back": True}}) is None
    # A hand-over that isn't borne out says so before anything else about the task.
    proof = {"state": "not_done", "note": "No send is on record."}
    assert _reason({"flags": {"proof": proof, "came_back": True, "probably_done": True}}) == {
        "text": "Not proven", "more": "No send is on record."}
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


# ── v2: people, next steps, links ──

_ROWS = [*_PEOPLE, "peopleRows", "HANDLE_CHANNEL", "handleChannel"]
_CHANNEL_NAMES = ["CHANNELS", "DEVICE_KINDS", "channelInfo", "channelLabel"]
# The icons the channel table names, as plain names (no lucide-react in the vm).
_ICONS = "".join(f"const {n} = '{n}';" for n in (
    "AlarmClock Bot Code2 Glasses Inbox Link2 Mail MessageCircle MessagesSquare Plug Send Smartphone Sparkles "
    "Users Watch").split())


def test_one_people_row_per_person_with_every_handle_and_you_once():
    parties = [
        {"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana Horvat", "verified": 1,
         "party_id": "pt_ana", "name": "Ana Horvat"},
        {"role": "requester", "kind": "waid", "address": "385911234567", "display": "Ana", "verified": 1,
         "party_id": "pt_ana", "name": "Ana Horvat"},
        {"role": "recipient", "kind": "email", "address": "ana@x.co", "display": "Ana Horvat", "verified": 1,
         "party_id": "pt_ana", "name": "Ana Horvat"},
        {"role": "cc", "kind": "fd_user", "address": _ME["id"], "display": "Ada Admin", "verified": 1, "me": True},
    ]
    out = _run([(_TEXT, _ROWS)], f"""
      out = peopleRows({json.dumps(parties)}, {json.dumps(_ME)})
        .map((r) => [r.role, r.name, r.me, r.handles.map((h) => [handleChannel(h.kind), h.address])]);
    """)
    assert out == [
        ["requester", "Ana Horvat", False, [["email", "ana@x.co"], ["whatsapp", "385911234567"]]],
        ["recipient", "Ana Horvat", False, [["email", "ana@x.co"]]],
        ["cc", "You", True, []],
    ]


def test_a_persons_people_filter_value_reads_as_their_name():
    out = _people(f"""
      const me = {json.dumps(_ME)};
      out = [partyFacetLabel('party:pt_ana', 'Ana Horvat', me), partyFacetLabel('party:pt_x', '  ', me),
             partyFacetLabel('party:pt_me', 'Ada Admin', me, true),
             partyFacetLabel('waid:385911234567', '+385911234567', me)];
    """)
    assert out == ["Ana Horvat", "Someone", "You", "+385911234567"]


def test_a_next_step_says_what_it_does_and_to_whom():
    out = _run([(_TEXT, [*_CHANNEL_NAMES, "nameKey", *_FOLD, "proposalLabel", "recipientText"])], """
      out = [
        proposalLabel({ kind: 'mail.draft' }), proposalLabel({ kind: 'reply.on_origin', payload: { channel: 'whatsapp' } }),
        proposalLabel({ kind: 'reply.on_origin', payload: { channel: 'device:stopwatch' } }),
        proposalLabel({ kind: 'reply.on_origin' }), proposalLabel({ kind: 'nudge' }), proposalLabel({ kind: 'track' }),
        recipientText({ address: 'ana@x.co', display: 'Ana Horvat', kind: 'email' }),
        recipientText({ address: '385911234567', display: 'Ana Horvat', kind: 'waid' }),
        recipientText({ address: '385911234567', display: '+385911234567', kind: 'waid' }),
        recipientText({ address: '777', display: 'Ana on Telegram', kind: 'telegram' }),
        recipientText({ address: 'ana@x.co', display: '' }),
      ];
    """, prelude=_ICONS)
    assert out == ["Draft email", "Reply on WhatsApp", "Reply on Device · StopWatch", "Reply", "Reminder",
                   "Next step", "Ana Horvat <ana@x.co>", "Ana Horvat (+385911234567)", "+385911234567",
                   "Ana on Telegram", "ana@x.co"]


def test_the_address_opens_one_task_once_and_only_a_task_id():
    prelude = """
      const window = { location: { href: '' }, history: { state: null,
        replaceState(state, title, url) { window.location.href = 'https://deck.example' + url; } } };
    """
    out = _run([(_STORE, ["LINKED", "hasTaskboardLink", "takeTaskboardLink"])], """
      const seen = [];
      for (const href of ['https://deck.example/?taskboard=tb_1a2b&x=1#top', 'https://deck.example/?taskboard=%3Cscript%3E',
                          'https://deck.example/?other=1']) {
        window.location.href = href;
        seen.push([hasTaskboardLink(), takeTaskboardLink(), window.location.href, hasTaskboardLink()]);
      }
      out = seen;
    """, prelude=prelude)
    assert out == [
        [True, "tb_1a2b", "https://deck.example/?x=1#top", False],
        [True, "", "https://deck.example/", False],   # not a task id: taken out, opens nothing
        [False, "", "https://deck.example/?other=1", False],
    ]


def test_a_bell_line_names_its_task_and_loses_its_link_line():
    body = "Ana is waiting for a reply.\n\n[Open in Taskboard](?taskboard=tb_1)"
    out = _run([(_BELL, ["TASK_LINK", "TASK_LINK_LINE"])], f"""
      const text = {json.dumps(body)};
      out = [TASK_LINK.exec(text)?.[1] ?? null, text.replace(TASK_LINK_LINE, ''),
             TASK_LINK.exec('[x](?taskboard=<script>)')?.[1] ?? null,
             'No link here.'.replace(TASK_LINK_LINE, '')];
    """)
    assert out == ["tb_1", "Ana is waiting for a reply.\n", None, "No link here."]


def test_a_same_as_can_be_turned_down_and_a_next_step_is_offered_only_where_the_server_makes_one():
    threads, api, board = _source(_THREADS), _source(_API), _source(_BOARD)
    assert "notSame: (id: string) => api<{ task: TaskItem }>(taskPath(id, '/not_same'), { method: 'POST' })" in api
    assert "/>}Not the same\n" in threads and "onNotSame()" in threads
    assert "onNotSame={onNotSame}" in board and "onNotSame={() => notSame(selected)}" in board
    assert "(task.kind === 'request' || task.kind === 'sent')" in board
    # The bell never shows a dead "Open in Taskboard" link where the board doesn't open.
    assert "const message = n.message.replace(TASK_LINK_LINE, '')" in _source(_BELL)


def test_a_moment_ago_is_just_now_never_in_a_moment():
    """A change made this second (the row's time set from the browser's
    clock) reads "just now" — fmtRelative reads the clock again, so the old
    clamp to now came out "in a moment"."""
    prelude = """
      function fmtRelative(epoch) {
        const secs = Math.round(epoch - Date.now() / 1000)
        const abs = Math.abs(secs)
        if (abs < 45) return secs >= 0 ? 'in a moment' : 'just now'
        return secs >= 0 ? 'in ' + abs + 's' : Math.round(abs / 60) + ' min ago'
      }
    """
    out = _run([(_TEXT, ["agoRelative"])], """
      const now = Date.now() / 1000;
      out = [agoRelative(now), agoRelative(now + 30), agoRelative(now - 10), agoRelative(now - 600)];
    """, prelude=prelude)
    assert out == ["just now", "just now", "just now", "10 min ago"]


def test_a_split_counts_only_what_can_move():
    """A correction (Done, Snooze, a merge's note) stays where it was made:
    one message and a correction is nothing to split."""
    out = _run([(_TEXT, ["movableEvents"])], """
      const e = (kind) => ({ kind });
      out = [movableEvents([e('received'), e('correction')], 2), movableEvents([e('received'), e('reply')], 2),
             movableEvents([e('correction')], 1), movableEvents(undefined), movableEvents([e('received')], 600)];
    """)
    assert out == [1, 2, 0, 0, 600]   # past the latest 500 the timeline holds, they count
    board = _source(_BOARD)
    assert "{picked && e.kind !== 'correction' && (" in board and "movable > 1 && !merged && !splitting" in board
    assert "total={movable}" in board


def test_a_merge_opens_the_task_the_server_says_it_is_part_of():
    board = _source(_BOARD)
    assert "const { task: target } = await taskboardApi.merge(source.id, into)" in board
    assert "res.id === into" not in board


def test_a_board_next_step_is_never_approved_unread_from_the_row_or_the_bell():
    """Its words are a model's: the row and the bell open them, and an edit in
    the open card is never passed over."""
    parts, bell = _source(_PARTS), _source(_BELL)
    assert "onReview={waiting.source === 'taskboard' ? onClick : undefined}" in _source(_BOARD)
    assert "{onReview ? (" in parts
    assert "canApprove={!step || !onOpenTask}" in bell   # where no board opens, the words are shown here instead
    assert "{step.to && <span" in bell and "waiting?.get(n.refId)?.taskboard" in bell
    assert "taskboard: a.source === 'taskboard'" in _source(_NOTIFY)


def test_a_followed_link_shows_the_board_for_this_visit_and_saves_nothing():
    prelude = """
      const saved = [];
      const LAYOUT_MODE_KEY = 'fd:layout-mode', SIMPLE_LEFT_OPEN_KEY = 'l', SIMPLE_RIGHT_OPEN_KEY = 'r',
        SIMPLE_QUEUE_OPEN_KEY = 'q';
      const persist = (k, v) => saved.push([k, v]);
      const loadLayoutMode = () => 'full', loadBool = () => true;
      const create = (fn) => {
        let state;
        const api = { getState: () => state, setState: (p) => { state = { ...state, ...p }; } };
        state = fn(api.setState, api.getState);
        return api;
      };
    """
    out = _run([(_UI, ["layoutChosenSinceHydrate", "layoutVisited", "beginLayoutHydrate", "useUIStore"])], """
      const ui = useUIStore.getState();
      ui.visitLayout('simple');
      const shown = useUIStore.getState().layoutMode;
      const visited = [saved.length, layoutVisited];
      beginLayoutHydrate();   // a sign-in's settings load after the link was followed: still on screen
      const kept = layoutChosenSinceHydrate || layoutVisited;
      useUIStore.getState().setLayoutMode('full');   // their own choice is saved, and ends the visit
      out = [shown, visited, kept, saved, layoutVisited];
    """, prelude=prelude)
    assert out == ["simple", [0, True], True, [["fd:layout-mode", "full"]], False]
    assert "layoutChosenSinceHydrate || layoutVisited ? useUIStore.getState().layoutMode" in _source(_UI)
    app = _source(_APP)
    assert "visitLayout('simple')" in app and "setLayoutMode('simple')" not in app


def test_a_followed_link_opens_the_board_on_any_screen_and_over_an_admins_console():
    """Most links are tapped on a phone, and an admin's default layout is the
    console: both get the board for the visit. Only the apps' console and a
    console address open none (and say so)."""
    app = _source(_APP)
    assert "const [boardVisit] = useState(() => !isAppShell() && hasTaskboardLink())" in app
    # Decided before the console is chosen: a console that mounted would rewrite the address.
    assert app.index("const [boardVisit]") < app.index("const inConsole = ")
    assert "&& !PAYMENTS_LINK && !boardVisit))" in app
    assert "if (isAppShell() || consoleRequested) {" in app
    # The visit's own branch comes before the phone's and the tablet's, after the console's.
    console = app.index("if (inConsole) return <ConsoleLayout />")
    visit = app.index("if (boardVisit && layoutMode === 'simple') {")
    assert console < visit < app.index("// ── Mobile layout ──")
    # Signing in to follow a link doesn't land on the console and lose it.
    land = app[app.index("async function landAdminAfterSignIn"):]
    assert land.index("if (hasTaskboardLink() || useTaskboardStore.getState().open) return") < land.index("openConsole(")


def test_escape_is_for_what_is_on_top_of_the_task():
    """The task's pickers and a split let the key go while the Sources sheet or
    a menu is open over them: it closes that, and ticked events stay ticked."""
    board, threads, nextstep = _source(_BOARD), _source(_THREADS), _source(_NEXT)
    assert "covered={exportOpen || settingsOpen || sourcesOpen || keysOpen}" in board
    assert "if (!splitting || covered) return" in board and "if (!snoozeOpen || covered) return" in board
    assert "if (covered) return" in threads and "if (!open || covered) return" in nextstep
    assert "covered={covered}" in board


def test_suggest_next_step_offers_only_what_the_server_can_make():
    board, nextstep = _source(_BOARD), _source(_NEXT)
    assert "suggestKinds.length > 0" in board and "kinds={suggestKinds}" in board
    assert "KINDS.filter((k) => kinds.includes(k.id)).map(" in nextstep
    assert "next_step_kinds?: NextStepKind[]" in _source(_API)


_KEYS = ["TASK_KEYS", "GO_KEYS", "keyLabel", "boardKey", "stepTo"]


def test_a_key_is_the_boards_only_when_nothing_else_wants_it():
    out = _run([(_TEXT, _KEYS)], """
        const body = { tagName: 'BODY' }
        const inside = { tagName: 'BUTTON' }
        const outside = { tagName: 'BUTTON' }
        const root = { contains: (el) => el === inside }
        const k = (key, target = body, more = {}) => boardKey({ key, target, ...more }, root)
        globalThis.out = {
          letter: k('e'), capital: k('U'), bang: k('!'), onAButton: k('e', inside),
          field: [k('e', { tagName: 'INPUT' }), k('e', { tagName: 'TEXTAREA' }), k('e', { tagName: 'SELECT' }),
                  k('e', { tagName: 'DIV', isContentEditable: true })],
          held: [k('k', body, { metaKey: true }), k('k', body, { ctrlKey: true }), k('e', body, { altKey: true })],
          composing: k('e', body, { isComposing: true }),
          used: k('e', body, { defaultPrevented: true }),
          outside: k('e', outside),
          enter: [k('Enter'), k('Enter', inside)],
          named: [k('Escape'), k('ArrowDown'), k('Tab')],
        }
    """)
    assert (out["letter"], out["capital"], out["bang"], out["onAButton"]) == ("e", "U", "!", "e")
    assert out["field"] == ["", "", "", ""] and out["held"] == ["", "", ""]
    assert out["composing"] == "" and out["used"] == "" and out["outside"] == ""
    assert out["enter"] == ["Enter", ""], "Enter on a button presses that button"
    assert out["named"] == ["", "", ""]


def test_j_and_k_walk_the_list_and_keep_their_place():
    out = _run([(_TEXT, _KEYS)], """
        const ids = ['a', 'b', 'c']
        globalThis.out = {
          next: stepTo(ids, 'a', 0, 1), previous: stepTo(ids, 'b', 1, -1),
          ends: [stepTo(ids, 'c', 2, 1), stepTo(ids, 'a', 0, -1)],
          first: [stepTo(ids, null, -1, 1), stepTo(ids, null, -1, -1)],
          afterClose: [stepTo(ids, null, 1, 1), stepTo(ids, null, 1, -1)],
          leftTheList: [stepTo(ids, 'gone', 1, 1), stepTo(ids, 'gone', 1, -1), stepTo(ids, 'gone', 3, 1), stepTo(ids, 'gone', 0, -1)],
          empty: stepTo([], 'a', 0, 1),
        }
    """)
    assert (out["next"], out["previous"]) == (1, 0)
    assert out["ends"] == [-1, -1]
    assert out["first"] == [0, 0]
    assert out["afterClose"] == [2, 0]
    assert out["leftTheList"] == [1, 0, 2, 0], "what took a done task's place is next"
    assert out["empty"] == -1


def test_the_keys_are_gmails_and_the_buttons_say_them():
    out = _run([(_TEXT, _KEYS)], "globalThis.out = { keys: TASK_KEYS, go: GO_KEYS, labels: [keyLabel('e'), keyLabel('U'), keyLabel('!')] }")
    assert out["keys"]["done"] == "e" and out["keys"]["snooze"] == "b" and out["keys"]["undo"] == "z"
    assert out["keys"]["notTask"] == "!" and out["keys"]["merge"] == "v" and out["keys"]["nextStep"] == "r"
    assert out["go"]["i"] == "needs_me" and out["go"]["a"] == "all" and out["go"]["d"] == "done"
    assert out["labels"] == ["e", "⇧U", "!"]
    board = _source(_BOARD)
    for name in ("reopen", "done", "gotIt", "notTask", "snooze", "rename", "undo", "merge"):
        assert f"<KeyHint k={{TASK_KEYS.{name}}}" in board, name
    assert "wide:inline-block" in _source(_PARTS), "a phone's buttons show no keys"


def test_no_key_approves_a_next_step():
    """The card's Approve stays a click: `r` only reads it, or asks for one that then waits."""
    text, board = _source(_TEXT), _source(_BOARD)
    assert "approve" not in text[text.index("export const TASK_KEYS"):text.index("export const GO_KEYS")].lower()
    start = board.index("keys.current = (key) =>")
    handler = board[start:board.index("pinned.current = false", start)]
    assert "approve" not in handler.lower().replace("approving it stays a click", "")
    assert "resolveProposal" not in handler


def test_moving_away_never_loses_a_split_or_the_snooze_picker():
    board = _source(_BOARD)
    assert "pinned.current = splitting || snoozeOpen" in board
    assert "if (!detailHeld.current) {" in board
    assert "if (over || keysOpen) return" in board


_MD = ["WORD_CHAR", "isSpace", "isWord", "cleanMarkdownLine", "inlinePieces", "markdownPieces"]


def _pieces(text: str) -> list[dict]:
    return _run([(_TEXT, _MD)], f"globalThis.out = markdownPieces({json.dumps(text)})")


def _words(text: str) -> str:
    return "".join(p["text"] for p in _pieces(text))


def test_what_the_screenshots_rows_said_reads_as_text():
    """As a reply on the board really is: its lines run together, cut off where it is kept."""
    got = _pieces("Pročitao sam thread — **Julien ima konkretnu pritužbu.** Odgovor je: > *\"remove the stupid stars\"* Uz to")
    assert "".join(p["text"] for p in got) == ("Pročitao sam thread — Julien ima konkretnu pritužbu. Odgovor je: "
                                               "\"remove the stupid stars\" Uz to")
    assert next(p for p in got if p.get("bold"))["text"] == "Julien ima konkretnu pritužbu."
    assert next(p for p in got if p.get("italic"))["text"] == "\"remove the stupid stars\""


def test_each_mark_reads_as_itself():
    assert _pieces("a **b** c") == [{"text": "a "}, {"text": "b", "bold": True}, {"text": " c"}]
    assert _pieces("*it*")[0] == {"text": "it", "italic": True}
    assert _pieces("_it_")[0] == {"text": "it", "italic": True}
    assert _pieces("`x_y`")[0] == {"text": "x_y", "code": True}
    assert _pieces("~~old~~")[0] == {"text": "old", "strike": True}
    nested = _pieces("**a *b***")
    assert {"text": "b", "bold": True, "italic": True} in nested


def test_text_that_only_looks_like_markdown_stays_as_it_is():
    for text in ("snake_case_name and other_one", "2 * 3 * 4", "2**3", "a > b", "issue #12 and C# too",
                 "__init__", "price: 5*6", "use * as a wildcard", "`not closed"):
        assert _words(text) == text, text


def test_a_mark_clipped_in_half_isnt_shown():
    assert _words("Doslovno: **Roberto Kut") == "Doslovno: Roberto Kut"
    assert _words("kraj rečenice** i dalje") == "kraj rečenice i dalje"


def test_lines_lose_their_marks():
    assert _words("# Title\n> quoted\n- item\n* other\n---\n1. one") == "Title\nquoted\n• item\n• other\n\n1. one"
    assert _words("## Heading in a run of text") == "Heading in a run of text"
    assert _words("prije ### Naslov poslije") == "prije Naslov poslije"


def test_nothing_is_followed_or_loaded_and_html_stays_text():
    assert _words("see [the report](https://evil.example/x) now") == "see the report now"
    assert _words("![tracker](https://evil.example/p.gif)") == "tracker"
    assert _words("[**bold** link](https://x.co \"t\")") == "bold link"
    assert _words("<img src=x onerror=alert(1)>") == "<img src=x onerror=alert(1)>"
    assert "dangerouslySetInnerHTML" not in _source(_PARTS)


def test_the_rows_and_the_task_read_it_the_same():
    board = _source(_BOARD)
    for field in ("task.ask", "task.result_text", "e.summary", "shortOf.detail", "reason.more"):
        assert f"<TaskText text={{{field}}} terms={{terms}} />" in board, field
    assert "<FoldHighlight text={task.ask}" not in board and "<FoldHighlight text={task.result_text}" not in board


def test_a_rows_lines_are_really_clamped():
    """``block`` would win over the clamp's own display: a long result ran to eight lines."""
    board = _source(_BOARD)
    assert "mt-1 text-xs leading-relaxed text-zinc-400 ${task.result_text ? 'line-clamp-1' : 'line-clamp-2'}" in board
    assert "mt-0.5 text-xs leading-relaxed text-zinc-500 ${task.ask ? 'line-clamp-2' : 'line-clamp-3'}" in board


def test_the_bigger_view_is_the_same_pane_with_its_buttons():
    """It only changes the aside's class: moving a pane to another parent would remount it and lose a form."""
    board = _source(_BOARD)
    assert "<aside className={focused ? 'absolute inset-0 z-20 bg-zinc-950' : 'w-[380px]" in board
    assert "if (focused && !selected) setFocused(false)" in board
    assert "if (focusedRef.current) { setFocused(false); return }" in board
    assert "if (key === TASK_KEYS.focus && selected && !phone)" in board
    assert "focused ? 'mx-auto w-full max-w-3xl' : ''" in board, "the buttons come along, under the text"
    assert "reading={focused}" in board


def test_escape_leaves_the_bigger_view_before_the_board():
    board = _source(_BOARD)
    start = board.index("if (keysRef.current) { setKeysOpen(false); return }")
    chain = board[start:board.index("onClose()", start)]
    assert chain.index("focusedRef") < chain.index("selectedRef")


def test_an_email_reads_without_its_tracking_links():
    out = _run([(_TEXT, ["readableEmail"])], """
        const long = 'https://websummit-dot-yamm-track.appspot.com/2M_MmpSl8t2uY_HNaT7MxBb9KeWTDJRV4A_twifOvQH6VCHWMoQem2KHQgTzeJS'
        globalThis.out = {
          bracketed: readableEmail('announced speakers <' + long + '> today'),
          bare: readableEmail('see ' + long + ' now'),
          short: readableEmail('see https://bbcc.hr/ai now'),
          blanks: readableEmail('a\\n\\n\\n\\nb   \\nc'),
          empty: readableEmail(''),
        }
    """)
    assert out["bracketed"] == "announced speakers [websummit-dot-yamm-track.appspot.com] today"
    assert out["bare"] == "see websummit-dot-yamm-track.appspot.com/… now"
    assert out["short"] == "see https://bbcc.hr/ai now"
    assert out["blanks"] == "a\n\nb\nc"
    assert out["empty"] == ""


def test_the_whole_email_is_only_fetched_in_the_bigger_view_and_shown_as_text():
    board = _source(_BOARD)
    assert "{focused && task.links?.gmail_thread_id && <TaskEmail taskId={task.id} terms={terms} />}" in board
    email = (_SRC / "components" / "taskboard" / "TaskEmail.tsx").read_text()
    assert "dangerouslySetInnerHTML" not in email and "<a " not in email and "<img" not in email


def test_next_move_is_the_boards_own_menu_on_a_key():
    board, parts = _source(_BOARD), _source(_PARTS)
    assert 'aria-label="Who moves next"' not in board, "no browser select any more"
    assert "<NextMoveMenu" in board and 'role="listbox"' in parts and 'aria-haspopup="listbox"' in parts
    assert "[TASK_KEYS.move]: [!closed && !failed, () => setMoveOpen(true)]" in board
    out = _run([(_TEXT, _KEYS)], "globalThis.out = TASK_KEYS.move")
    assert out == "m"


def test_with_the_menu_open_the_tasks_other_keys_wait():
    board, parts = _source(_BOARD), _source(_PARTS)
    assert "if (busy || handing || moveOpen) return false" in board
    assert "pinned.current = splitting || snoozeOpen || handing || moveOpen" in board
    start = parts.index("const onKey = (e: KeyboardEvent) =>")
    keys = parts[start:parts.index("const Current =", start)]
    assert "e.stopPropagation()" in keys and "else return" in keys, "a key it doesn't use isn't swallowed"

