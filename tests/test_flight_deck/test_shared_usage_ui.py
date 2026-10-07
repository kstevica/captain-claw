"""PR D UI: the owner's agent can look into how members use it
(`shared_agent_usage`), run in Node like test_shared_agent_ui.py (TypeScript
lifted out of the sources, transpiled and run in a vm; wiring read off the
parsed tree).

Pinned here:

* both new texts, exactly — the member paragraph must equal Flight Deck's
  one-time bell body (`MEMBER_USAGE_PARAGRAPH`) for the same owner, and the
  owner's Share-dialog paragraph keeps the PAUSE_ON_CONTENT clause;
* the member notice gains the paragraph only on an explicit `ownerReads ===
  true`, on every caps variant (chat-only conversations are readable too),
  after the commons paragraph and before the host-trust warning, and stays
  byte-identical otherwise;
* the owner's Share note gains its paragraph on both runtimes, after packs and
  the commons and before the host-trust paragraph, and stays byte-identical
  otherwise;
* the store takes `shared_usage` only on an explicit true with sharing on;
* the chat notice, the notice component and the owner cards are wired to it;
* the notice's dismissal key is bumped (v4): a PR C dismissal no longer hides
  it, and dismissing writes the v4 key only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_flight_deck.test_shared_agent_ui import (  # noqa: F401
    _ALL_CAPS,
    _JSX_HELPERS,
    _NO_CAPS,
    _OWNER_NOTE,
    _OWNER_PACKS_NOTE,
    _STORAGE_HARNESS,
    HOST_TRUST_WARNING,
    OWNER_HOST_NOTE,
    _a1_notice,
    _a2_notice,
    _lift,
    _query,
    pytestmark,
)
from test_flight_deck.test_shared_workspace_ui import (
    _GUARDS,
    OWNER_WORKSPACE_NOTE,
    _workspace_notice,
)

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "flight-deck" / "src"
_SHARED = _SRC / "utils" / "sharedAgent.ts"
_SHARED_SERVICE = _SRC / "services" / "sharedAgents.ts"
_SHARED_STORE = _SRC / "stores" / "sharedAgentStore.ts"
_AGENTS = _SRC / "components" / "agents"
_NOTICE = _AGENTS / "SharedAgentNotice.tsx"
_CHAT_PANEL = _AGENTS / "ChatPanel.tsx"
_PROCESS_CARD = _AGENTS / "ProcessCard.tsx"
_CONTAINER_CARD = _AGENTS / "ContainerCard.tsx"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Only sources are read and run in a Node vm, but the Node subprocesses
    # inherit this env: never let one see the real ~/.captain-claw.
    monkeypatch.setenv("HOME", str(tmp_path))


# ── texts (part 0b §2 / §5, exact) ──────────────────────────────────

# Part 0b §2: Flight Deck's member paragraph (the one-time bell body), with
# its `{Owner}` / `{owner}` slots. ASCII apostrophes, Unicode dashes.
MEMBER_USAGE_PARAGRAPH = (
    "{Owner}'s agent can also look into your use of it when {owner} or this deck's admins ask: that "
    "you use it and how much, what you created or shared on it, and your conversations here, which it "
    "can quote. Anyone {owner} lets talk to the agent can ask it the same — people they connect "
    "through WhatsApp, Telegram, Slack, Discord or the API, {owner}'s other agents and its "
    "automations — and whoever receives those answers may see what it quotes. What it reads this way "
    "isn't automatically turned into shared knowledge for other members. It never opens your Google "
    "account or your private deep memory for this, and it leaves out its replies from the turns in "
    "which it used them for you, though a later reply that repeats what it found there can still be "
    "shown to it.")

# Part 0b §5: the owner's Share-dialog paragraph. "’" is U+2019 throughout
# (like OWNER_WORKSPACE_NOTE); PAUSE_ON_CONTENT is on (user decision J15), so
# the "holds back actions" clause is in. Reworded (integration review) to what
# is enforced: only replies of turns that used a member's Google / private
# deep memory are hidden, and a data-level turn is refused insights, playbooks,
# the datastore and write/edit — not every file write.
OWNER_USAGE_NOTE = (
    "Your agent can also tell you who it is shared with and look into how they use it whenever you "
    "ask — in your Flight Deck chats, your channels and your automations: each member’s activity "
    "and token use, what they created or shared on it, and their private conversations with it. It "
    "leaves out replies from turns where it used a member’s Google account or private deep memory, "
    "though a later reply can repeat what it found there. Members are told. Anyone who can talk to "
    "this agent through your channels, the API or your other agents can ask it the same. A turn that "
    "reads members’ data adds nothing to the shared insights, playbooks or topics, and can’t save to "
    "the datastore or write files with its write and edit tools; after reading their conversations "
    "the agent also holds back actions that send or change things until your next message. Whatever "
    "you then ask it to save to files, the datastore, insights or playbooks is visible to every "
    "member. Treat members’ conversations as untrusted input.")
# A Docker agent's variant: no members' Google / deep memory, no saved/ or
# datastore commons — only insights, playbooks and topics are shared.
OWNER_USAGE_NOTE_DOCKER = (
    "Your agent can also tell you who it is shared with and look into how they use it whenever you "
    "ask — in your Flight Deck chats, your channels and your automations: each member’s activity "
    "and token use, what they shared with it, and their private conversations with it. Members are "
    "told. Anyone who can talk to this agent through your channels, the API or your other agents can "
    "ask it the same. A turn that reads members’ data adds nothing to the shared insights, playbooks "
    "or topics; after reading their conversations the agent also holds back actions that send or "
    "change things until your next message. Whatever you then ask it to save to insights or "
    "playbooks is visible to every member. Treat members’ conversations as untrusted input.")


def _reads(owner: str) -> str:
    """`memberOwnerReadsNotice(owner)` for an already-defaulted owner name."""
    return MEMBER_USAGE_PARAGRAPH.format(Owner=owner[:1].upper() + owner[1:], owner=owner)


def test_member_owner_reads_notice_is_the_bell_text():
    out = _lift([(_SHARED, ["memberOwnerReadsNotice"])], """
      out = { olga: memberOwnerReadsNotice('Olga'), blank: memberOwnerReadsNotice(''),
              spaces: memberOwnerReadsNotice('  '), ana: memberOwnerReadsNotice('ana'),
              padded: memberOwnerReadsNotice('  Olga ') };
    """)
    assert out["olga"] == _reads("Olga")
    assert out["olga"].encode("utf-8") == _reads("Olga").encode("utf-8")
    assert out["padded"] == out["olga"]
    for key in ("blank", "spaces"):
        text = out[key]
        assert text.startswith(
            "The owner's agent can also look into your use of it when the owner or this "
            "deck's admins ask:"), key
        assert "the owner's other agents" in text, key
        assert text == _reads("the owner"), key
    ana = out["ana"]
    assert ana.startswith("Ana's agent")
    for want in ("when ana or this deck's admins ask", "Discord", "can still be shown to it."):
        assert want in ana, want
    assert ana.endswith("can still be shown to it.")
    # Accurate, not an overpromise: only the turns that used them are left out.
    assert "the answers it gave you right after looking" not in ana
    assert "a later reply that repeats what it found there can still be shown" in ana
    for text in out.values():
        assert "\n" not in text
        # ASCII apostrophes only (FD's bell uses them); the two dashes are U+2014.
        assert "’" not in text
        assert text.count("—") == 2


def test_member_owner_reads_notice_matches_flight_decks_bell_when_present():
    fd = pytest.importorskip("captain_claw.flight_deck.shared_usage")
    assert fd.MEMBER_USAGE_PARAGRAPH == MEMBER_USAGE_PARAGRAPH
    out = _lift([(_SHARED, ["memberOwnerReadsNotice"])], """
      out = [memberOwnerReadsNotice('Olga'), memberOwnerReadsNotice('ana'), memberOwnerReadsNotice('')];
    """)
    for owner, ts_side in zip(("Olga", "ana", "the owner"), out):
        assert ts_side == fd.MEMBER_USAGE_PARAGRAPH.format(
            Owner=owner[:1].upper() + owner[1:], owner=owner), owner


def test_owner_usage_note_is_exact():
    out = _lift([(_SHARED, ["OWNER_USAGE_NOTE", "OWNER_USAGE_NOTE_DOCKER"])],
                "out = [OWNER_USAGE_NOTE, OWNER_USAGE_NOTE_DOCKER];")
    note, docker = out
    assert note == OWNER_USAGE_NOTE
    assert note.encode("utf-8") == OWNER_USAGE_NOTE.encode("utf-8")
    assert docker == OWNER_USAGE_NOTE_DOCKER
    assert docker.encode("utf-8") == OWNER_USAGE_NOTE_DOCKER.encode("utf-8")
    for text, quotes in ((note, 5), (docker, 3)):
        assert "visible to every member" in text
        assert "\n" not in text
        assert "'" not in text and text.count("’") == quotes
        # PAUSE_ON_CONTENT (J15) is on: the owner is told about the paused tools.
        assert (
            "after reading their conversations the agent also holds back actions that send or "
            "change things until your next message." in text
        )
    # No overpromise: not "except answers…" and not "can’t save to … files".
    assert "except answers" not in note and "the datastore or files;" not in note
    assert "leaves out replies from turns where it used a member’s Google account" in note
    assert "write files with its write and edit tools" in note
    # Docker: no commons it doesn't have, no members' Google / deep memory.
    for absent in ("Google", "deep memory", "datastore", "files", "created"):
        assert absent not in docker, absent


def test_shared_agent_utils_stay_import_free():
    # The node-lift tests depend on it (part 3 header).
    found = _query(_SHARED, "return all.filter((n) => ts.isImportDeclaration(n)).length;")
    assert found == 0


# ── the member notice ───────────────────────────────────────────────

_NOTICE_DECLS = [(_SHARED, [
    "CHAT_ONLY_CAPS", "sharedCaps", "memberWorkspaceNotice", "memberOwnerReadsNotice",
    "memberNoticeText",
])]


def test_member_notice_without_owner_reads_is_unchanged():
    out = _lift(_NOTICE_DECLS, f"""
      const proc = {json.dumps(_ALL_CAPS)};
      out = {{
        a1: memberNoticeText('Helper', 'Olga', ''),
        a2: memberNoticeText('Helper', 'Olga', '', proc),
        ws: memberNoticeText('Helper', 'Olga', '', proc, true),
        a1False: memberNoticeText('Helper', 'Olga', '', CHAT_ONLY_CAPS, false, false),
        wsFalse: memberNoticeText('Helper', 'Olga', '', proc, true, false),
        warned: memberNoticeText('Helper', 'Olga', {json.dumps(HOST_TRUST_WARNING)}, proc, true),
      }};
    """)
    a2 = _a2_notice("Helper", "Olga")
    ws = a2 + " " + _workspace_notice("Olga")
    assert out["a1"] == out["a1False"] == _a1_notice("Helper", "Olga")
    assert out["a2"] == a2
    assert out["ws"] == out["wsFalse"] == ws
    assert out["warned"] == ws + "\n\n" + HOST_TRUST_WARNING
    for text in out.values():
        assert "look into your use of it" not in text


def test_member_notice_with_owner_reads_adds_the_paragraph_last():
    out = _lift(_NOTICE_DECLS, f"""
      const w = {json.dumps(HOST_TRUST_WARNING)};
      const proc = {json.dumps(_ALL_CAPS)};
      const docker = sharedCaps({{ runtime: 'docker', capabilities: {json.dumps(_NO_CAPS)} }});
      out = {{
        full: memberNoticeText('Helper', 'Olga', w, proc, true, true),
        fullBare: memberNoticeText('Helper', 'Olga', '', proc, true, true),
        noWorkspace: memberNoticeText('Helper', 'Olga', '', proc, false, true),
        chatOnly: memberNoticeText('Helper', 'Olga', '', CHAT_ONLY_CAPS, false, true),
        chatOnlyWarned: memberNoticeText('Helper', 'Olga', w, CHAT_ONLY_CAPS, false, true),
        // The commons flag doesn't reach a chat-only agent; the reads paragraph does.
        docker: memberNoticeText('Helper', 'Olga', '', docker, true, true),
        olderDeck: memberNoticeText('Helper', 'Olga', '', sharedCaps({{}}), true, true),
        nulled: memberNoticeText('Helper', 'Olga', '', null, true, true),
        // Only an explicit true counts.
        yes: memberNoticeText('Helper', 'Olga', '', proc, true, 'yes'),
        one: memberNoticeText('Helper', 'Olga', '', proc, true, 1),
        ws: memberNoticeText('Helper', 'Olga', '', proc, true),
        // The notice's own owner fallback carries into the paragraph.
        noOwner: memberNoticeText('Helper', '', '', proc, false, true),
        spacesOwner: memberNoticeText('  ', '  ', '', CHAT_ONLY_CAPS, false, true),
      }};
    """)
    a2 = _a2_notice("Helper", "Olga")
    a1 = _a1_notice("Helper", "Olga")
    reads = _reads("Olga")
    assert out["full"] == (
        a2 + " " + _workspace_notice("Olga") + " " + reads + "\n\n" + HOST_TRUST_WARNING)
    assert out["fullBare"] == a2 + " " + _workspace_notice("Olga") + " " + reads
    assert out["noWorkspace"] == a2 + " " + reads
    assert out["chatOnly"] == a1 + " " + reads
    assert out["chatOnlyWarned"] == a1 + " " + reads + "\n\n" + HOST_TRUST_WARNING
    for key in ("docker", "olderDeck", "nulled"):
        assert out[key] == a1 + " " + reads, key
    assert out["yes"] == out["one"] == out["ws"]
    assert "look into your use of it" not in out["yes"]
    blank = _reads("another user")
    assert blank.startswith(
        "Another user's agent can also look into your use of it when another user or this "
        "deck's admins ask")
    assert out["noOwner"] == _a2_notice("Helper", "another user") + " " + blank
    assert out["spacesOwner"] == _a1_notice("This agent", "another user") + " " + blank
    for key in ("fullBare", "noWorkspace", "chatOnly", "docker", "noOwner"):
        assert "\n" not in out[key], key


# ── the owner's Share note ──────────────────────────────────────────

_OWNER_DECLS = [(_SHARED, [
    "OWNER_SHARE_NOTE", "OWNER_WORKSPACE_NOTE", "OWNER_USAGE_NOTE", "OWNER_USAGE_NOTE_DOCKER",
    "ownerShareNote",
])]


def test_owner_share_note_with_the_usage_paragraph():
    out = _lift(_OWNER_DECLS, """
      out = {
        off4: ownerShareNote(false, 'process', false, false),
        off3: ownerShareNote(false, 'process', false),
        off1: ownerShareNote(false),
        both3: ownerShareNote(true, 'process', true),
        both4: ownerShareNote(true, 'process', true, false),
        docker2: ownerShareNote(true, 'docker'),
        docker4: ownerShareNote(true, 'docker', false, false),
        usage: ownerShareNote(false, 'process', false, true),
        all: ownerShareNote(true, 'process', true, true),
        dockerUsage: ownerShareNote(true, 'docker', false, true),
        dockerUsageOnly: ownerShareNote(false, 'docker', false, true),
        dockerWsUsage: ownerShareNote(false, 'docker', true, true),
        wsUsage: ownerShareNote(false, 'process', true, true),
        packsUsage: ownerShareNote(true, 'process', false, true),
        truthy: ownerShareNote(false, 'process', false, 'yes'),
        one: ownerShareNote(false, 'process', false, 1),
      };
    """)
    host_tail = "\n\n" + OWNER_HOST_NOTE
    main = _OWNER_NOTE[: -len(host_tail)]
    reworded = main.replace(
        "the shell, your files or accounts",
        "the shell, your accounts or your files outside its saved/ folder")
    docker_packs = _OWNER_PACKS_NOTE.replace(
        "their own profile, folders and deep memory", "their own profile")
    assert docker_packs != _OWNER_PACKS_NOTE
    # Without the flag: byte-identical to PR C.
    assert out["off4"] == out["off3"] == out["off1"] == _OWNER_NOTE
    assert out["both3"] == out["both4"] == (
        reworded + "\n\n" + _OWNER_PACKS_NOTE + "\n\n" + OWNER_WORKSPACE_NOTE + host_tail)
    assert out["docker2"] == out["docker4"] == main + "\n\n" + docker_packs + host_tail
    # With it: the usage paragraph last before the host-trust paragraph.
    assert out["usage"] == main + "\n\n" + OWNER_USAGE_NOTE + host_tail
    assert out["all"] == (
        reworded + "\n\n" + _OWNER_PACKS_NOTE + "\n\n" + OWNER_WORKSPACE_NOTE + "\n\n"
        + OWNER_USAGE_NOTE + host_tail)
    assert out["dockerUsage"] == (
        main + "\n\n" + docker_packs + "\n\n" + OWNER_USAGE_NOTE_DOCKER + host_tail)
    # Docker agents get their own variant, never the commons paragraph.
    assert out["dockerUsageOnly"] == out["dockerWsUsage"] == (
        main + "\n\n" + OWNER_USAGE_NOTE_DOCKER + host_tail)
    assert out["wsUsage"] == reworded + "\n\n" + OWNER_WORKSPACE_NOTE + "\n\n" + OWNER_USAGE_NOTE + host_tail
    assert out["packsUsage"] == main + "\n\n" + _OWNER_PACKS_NOTE + "\n\n" + OWNER_USAGE_NOTE + host_tail
    # Only an explicit true counts.
    assert out["truthy"] == out["one"] == _OWNER_NOTE
    for key, note in out.items():
        assert note.endswith(host_tail), key
        assert note.count(OWNER_USAGE_NOTE) == (
            1 if key in ("usage", "all", "wsUsage", "packsUsage") else 0), key
        assert note.count(OWNER_USAGE_NOTE_DOCKER) == (
            1 if key in ("dockerUsage", "dockerUsageOnly", "dockerWsUsage") else 0), key


# ── the store and the response type ─────────────────────────────────


def test_store_takes_shared_usage_only_on_an_explicit_true():
    out = _lift([(_SHARED_STORE, ["useSharedAgentStore"])], r"""
      function create(init) {
        let state;
        const set = (p) => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; };
        const get = () => state;
        state = init(set, get);
        return { getState: get };
      }
      let server = { enabled: true, host_warning: '', agents: [], shared_usage: true };
      let fail = false;
      const getSharedAgents = async () => { if (fail) throw new Error('x'); return JSON.parse(JSON.stringify(server)); };
      const setSharedAgentGoogle = async () => ({});
      const leaveShare = async () => {};
      const sharedContainerId = (r) => 'shared:' + r;
      const clearSharedSlices = () => {};
      const useChatStore = { getState: () => ({ sessions: new Map(), disconnectChat: () => {} }) };
      const st = () => useSharedAgentStore.getState();
      done = (async () => {
        const res = { initial: st().sharedUsage };
        await st().fetch(); res.on = st().sharedUsage;
        // Its own flag: the commons flag stays off.
        res.workspace = st().memberWorkspace;
        fail = true; await st().fetch(); res.blip = st().sharedUsage; fail = false;
        server = { ...server, shared_usage: 'true' }; await st().fetch(); res.truthy = st().sharedUsage;
        server = { ...server, shared_usage: 1 }; await st().fetch(); res.one = st().sharedUsage;
        server = { ...server, shared_usage: undefined }; await st().fetch(); res.olderDeck = st().sharedUsage;
        server = { ...server, shared_usage: true, enabled: false }; await st().fetch(); res.sharingOff = st().sharedUsage;
        server = { ...server, shared_usage: true, enabled: true }; await st().fetch(); res.back = st().sharedUsage;
        out = res;
      })();
    """)
    assert out == {
        "initial": False, "on": True, "workspace": False, "blip": True, "truthy": False,
        "one": False, "olderDeck": False, "sharingOff": False, "back": True,
    }


def test_shared_agents_response_declares_shared_usage():
    found = _query(_SHARED_SERVICE, r"""
      const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'SharedAgentsResponse');
      const m = i.members.find((x) => x.name && x.name.getText(sf) === 'shared_usage');
      return m ? { optional: !!m.questionToken, type: m.type.getText(sf) } : null;
    """)
    assert found == {"optional": True, "type": "boolean"}


# ── wiring ──────────────────────────────────────────────────────────


def test_chat_notice_component_and_owner_cards_wiring():
    chat = _query(_CHAT_PANEL, _JSX_HELPERS + _GUARDS + r"""
      const notices = named('SharedAgentNotice');
      return { reads: notices.map((n) => attr(n, 'ownerReads')), usage: init('sharedUsage'),
               guards: notices.map((n) => guards(n)[0] || '') };
    """)
    assert chat["reads"] == ["{sharedUsage}"]
    assert chat["usage"] == "useSharedAgentStore((s) => s.sharedUsage)"
    assert chat["guards"][0].startswith("shared &&")

    notice = _query(_NOTICE, r"""
      const call = all.find((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'memberNoticeText');
      const fn = all.find((n) => ts.isFunctionDeclaration(n) && n.name && n.name.text === 'SharedAgentNotice');
      const param = fn.parameters[0];
      const el = param.name.elements.find((e) => e.name.getText(sf) === 'ownerReads');
      const member = param.type.members.find((m) => m.name && m.name.getText(sf) === 'ownerReads');
      return { args: call.arguments.map((a) => a.getText(sf)),
               dflt: el && el.initializer ? el.initializer.getText(sf) : null,
               type: member ? { optional: !!member.questionToken, type: member.type.getText(sf) } : null };
    """)
    assert notice == {
        "args": ["agentName", "ownerName", "hostWarning", "caps", "workspace", "ownerReads"],
        "dflt": "false",
        "type": {"optional": True, "type": "boolean"},
    }

    for src, note in (
        (_PROCESS_CARD, "{ownerShareNote(packsEnabled, 'process', workspaceEnabled, usageEnabled)}"),
        (_CONTAINER_CARD, "{ownerShareNote(packsEnabled, 'docker', false, usageEnabled)}"),
    ):
        card = _query(src, _JSX_HELPERS + _GUARDS + r"""
          return { notes: named('ShareModal').map((n) => attr(n, 'note')), usage: init('usageEnabled') };
        """)
        assert card == {"notes": [note], "usage": "useSharedAgentStore((s) => s.sharedUsage)"}, src.name


# ── the notice's dismissal (v4) ─────────────────────────────────────

_RENDER_HARNESS = r"""
const React = { createElement: (type, props, ...children) =>
  ({ type, props: props || {}, children: children.flat(Infinity) }) };
const useState = (init) => [typeof init === 'function' ? init() : init, () => {}];
const Info = 'Info';
const X = 'X';
const find = (n, pred) => {
  if (!n || typeof n !== 'object') return null;
  if (pred(n)) return n;
  for (const c of n.children || []) { const f = find(c, pred); if (f) return f; }
  return null;
};
const text = (n) => (n == null || n === false || n === true) ? ''
  : typeof n === 'object' ? (n.children || []).map(text).join('') : String(n);
"""

_RENDER_DECLS = [
    (_SHARED, [
        "SHARED_ACK_PREFIX", "sliceUser", "sharedAckKey", "CHAT_ONLY_CAPS",
        "memberWorkspaceNotice", "memberOwnerReadsNotice", "memberNoticeText",
    ]),
    (_NOTICE, ["ackKey", "readAck", "writeAck", "withBold", "SharedAgentNotice"]),
]


def test_an_older_dismissal_no_longer_hides_the_notice():
    out = _lift(_RENDER_DECLS, _STORAGE_HARNESS + _RENDER_HARNESS + f"""
      const proc = {json.dumps(_ALL_CAPS)};
      // Ana dismissed PR C's notice (and A1's) on this browser.
      store.set('fd.sharedAgentAck.v3.u-ana.' + R, '1');
      store.set('fd.sharedAgentAck.v2.u-ana.' + R, '1');
      const props = {{ agentRef: R, agentName: 'Helper', ownerName: 'Olga', hostWarning: '',
                       caps: proc, workspace: true, ownerReads: true }};
      const first = SharedAgentNotice(props);
      const body = text(first);
      const button = find(first, (n) => n.type === 'button');
      button.props.onClick();
      const keys = [...store.keys()].sort();
      const again = SharedAgentNotice(props);
      // Without the flag the paragraph is absent (another agent: not dismissed).
      const plain = text(SharedAgentNotice({{ agentRef: R2, agentName: 'Helper', ownerName: 'Olga',
                                              hostWarning: '', caps: proc, workspace: true }}));
      out = {{ shown: first !== null, body, keys, again, plain }};
    """, tsx=True)
    a2 = _a2_notice("Helper", "Olga").replace("**", "")
    assert out["shown"] is True
    assert out["body"] == a2 + " " + _workspace_notice("Olga") + " " + _reads("Olga")
    # Dismissing writes the v4 key only; older keys are left as they were.
    assert out["keys"] == sorted([
        "fd.sharedAgentAck.v2.u-ana.process:x:0123456789abcdef",
        "fd.sharedAgentAck.v3.u-ana.process:x:0123456789abcdef",
        "fd.sharedAgentAck.v4.u-ana.process:x:0123456789abcdef",
    ])
    assert out["again"] is None
    assert out["plain"] == a2 + " " + _workspace_notice("Olga")
