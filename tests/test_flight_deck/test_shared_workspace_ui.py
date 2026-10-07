"""PR C UI: a shared PROCESS agent's saved files and datastore as a commons,
run in Node like test_shared_agent_ui.py (TypeScript lifted out of the
sources, transpiled and run in a vm; wiring read off the parsed tree).

Pinned here:

* every new text, exactly — including the owner's commons paragraph, which
  Flight Deck also sends as a one-time bell and so must not drift;
* the member notice gains the commons paragraph only where the agent's files
  are on AND the deck serves members' panels; the owner's Share note gains
  its paragraph (process agents only) before the host-trust paragraph, and
  stays byte-identical otherwise; the notice's dismissal key is bumped (v3);
* creator badges: "You" / "<owner> (owner)" / the member's name for members,
  "✎ <member>" on member items only for the owner; upload checks; which panels
  show (`workspaceVisible`); a member's file reaches the viewer by its id,
  never a host path;
* the member service calls only Flight Deck's member routes, by `ref`, with
  no host, port or agent token; the store takes `member_workspace` only on an
  explicit true;
* Flight Deck's HTML previews no longer run same-origin, remote images in
  someone else's markdown aren't fetched, and a member's HTML is never
  offered as a deck;
* the right column, the member card, the chat notice and the owner card are
  wired to all of the above behind the flag.
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
    HOST_TRUST_WARNING,
    OWNER_HOST_NOTE,
    _a2_notice,
    _lift,
    _query,
    pytestmark,
)

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "flight-deck" / "src"
_WS_UTILS = _SRC / "utils" / "sharedWorkspace.ts"
_SHARED = _SRC / "utils" / "sharedAgent.ts"
_WS_SERVICE = _SRC / "services" / "sharedWorkspace.ts"
_SHARED_SERVICE = _SRC / "services" / "sharedAgents.ts"
_FILE_TRANSFER = _SRC / "services" / "fileTransfer.ts"
_SHARED_STORE = _SRC / "stores" / "sharedAgentStore.ts"
_AGENTS = _SRC / "components" / "agents"
_BADGE = _AGENTS / "CreatorBadge.tsx"
_FILES_PANEL = _AGENTS / "SharedFilesPanel.tsx"
_DATA_PANEL = _AGENTS / "SharedDatastorePanel.tsx"
_FILE_VIEWER = _AGENTS / "FileViewer.tsx"
_AGENT_FILES = _AGENTS / "AgentFilesPanel.tsx"
_FILE_BROWSER = _AGENTS / "FileBrowser.tsx"
_AGENT_DATA = _AGENTS / "AgentDatastorePanel.tsx"
_DS_BROWSER = _AGENTS / "DatastoreBrowser.tsx"
_SHARED_CARD = _AGENTS / "SharedAgentCard.tsx"
_NOTICE = _AGENTS / "SharedAgentNotice.tsx"
_CHAT_PANEL = _AGENTS / "ChatPanel.tsx"
_PROCESS_CARD = _AGENTS / "ProcessCard.tsx"
_CONTAINER_CARD = _AGENTS / "ContainerCard.tsx"
_SIMPLE = _SRC / "components" / "layout" / "SimpleLayout.tsx"
_VFS_VIEWER = _SRC / "components" / "vfs" / "VFSFileViewer.tsx"
_FOLDERS_PAGE = _SRC / "pages" / "AgentFoldersPage.tsx"
_APP_CODE = _SRC / "pages" / "AppCodePage.tsx"
_PIN_STORE = _SRC / "stores" / "pinnedFilesStore.ts"
_PINNED = _SRC / "components" / "common" / "PinnedFiles.tsx"
_COUNCIL_FILES = _SRC / "components" / "council" / "CouncilFileBrowser.tsx"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Only sources are read and run in a Node vm, but the Node subprocesses
    # inherit this env: never let one see the real ~/.captain-claw.
    monkeypatch.setenv("HOME", str(tmp_path))


# ── texts (part 3 §3, exact) ────────────────────────────────────────

TEXTS = {
    "MEMBER_FILES_NOTE": (
        "Everyone who uses this agent can open these files. You can delete only the ones you added."
    ),
    "MEMBER_DATA_NOTE": (
        "Everyone who uses this agent can see these tables. To add or change data, ask the agent "
        "in chat — you can change only what you added."
    ),
    "CHAT_ONLY_TEXT": "A shared agent is chat only — its files and datastore stay with its owner.",
    "UPLOAD_LABEL": "Upload",
    "UPLOAD_TITLE": (
        "Add a file to your own folder on this agent — everyone who uses it can open it"
    ),
    "NO_SHARED_FILES": "No files yet.",
    "NO_SHARED_TABLES": "No tables yet.",
    "TRUNCATED_NOTE": "Showing the newest 2,000 files.",
    "CREATED_BY_COLUMN": "Created by",
    "FILES_BUTTON": "Files",
    "DATA_BUTTON": "Data",
}

_WS_HELPERS = [
    "creatorLabel", "ownerBadgeLabel", "ownerBadgeTitle", "deleteConfirmText", "uploadError",
    "workspaceVisible", "isMemberCreated", "toViewerFile",
]
_ALL_WS = [(_WS_UTILS, list(TEXTS) + _WS_HELPERS)]

# Part 0c §1: the owner-commons text, byte for byte. "agent’s" is U+2019;
# every other apostrophe is ASCII. FD sends the same string as a bell.
OWNER_WORKSPACE_NOTE = (
    "Members can also open everything in this agent’s saved/ folder — including what is "
    "already there: files you uploaded in your own chats, screenshots and browser captures, "
    "script outputs, the scripts and tools it saved for you (check them for passwords or "
    "keys), and what its channels and automations saved, such as WhatsApp or email "
    "attachments — and see every table and row in its datastore. They can add their own "
    "files, tables and rows, shown with their name, and change or delete only what they "
    "added; you can change or delete all of it. Its other files (workspace, output/, "
    "workflows/) stay yours. Your agent reads what members add, so treat it as untrusted "
    "input, especially in automations."
)


def _workspace_notice(owner: str) -> str:
    """Part 3 §3 `memberWorkspaceNotice`."""
    return (
        "Files saved in your chats here (this agent’s saved/ folder) and its data tables are "
        f"shared with everyone who uses this agent: {owner} and every other member can open "
        "what you or the agent save there — including anything it saves from your mail, Drive "
        "or deep memory — and you can open theirs. Only whoever created a file, table or row, "
        f"and {owner}, can change or delete it, and everyone sees who created it."
    )


def test_every_workspace_text_is_exact():
    out = _lift(_ALL_WS, "out = {" + ", ".join(TEXTS) + "};")
    assert out == TEXTS


def test_owner_workspace_note_is_the_bell_text_byte_for_byte():
    note = _lift([(_SHARED, ["OWNER_WORKSPACE_NOTE"])], "out = OWNER_WORKSPACE_NOTE;")
    assert note == OWNER_WORKSPACE_NOTE
    assert note.encode("utf-8") == OWNER_WORKSPACE_NOTE.encode("utf-8")
    # The one typographic apostrophe; "\n"-free (FD's bell body is one line).
    assert note.count("\u2019") == 1 and "agent\u2019s saved/" in note
    assert "\n" not in note
    for named in ("uploaded in your own chats", "screenshots and browser captures",
                  "script outputs", "scripts and tools", "channels and automations"):
        assert named in note, named


# ── the member notice ───────────────────────────────────────────────


def test_member_workspace_notice():
    out = _lift([(_SHARED, ["memberWorkspaceNotice"])], """
      out = { olga: memberWorkspaceNotice('Olga'), blank: memberWorkspaceNotice(''),
              spaces: memberWorkspaceNotice('   '), padded: memberWorkspaceNotice('  Olga ') };
    """)
    assert out["olga"] == _workspace_notice("Olga")
    assert out["blank"] == _workspace_notice("the owner")
    assert out["spaces"] == _workspace_notice("the owner")
    assert out["padded"] == _workspace_notice("Olga")


_NOTICE_DECLS = [(_SHARED, ["CHAT_ONLY_CAPS", "sharedCaps", "memberWorkspaceNotice", "memberNoticeText"])]


def test_member_notice_adds_the_commons_paragraph_only_with_the_flag_and_files():
    out = _lift(_NOTICE_DECLS, f"""
      const w = {json.dumps(HOST_TRUST_WARNING)};
      const proc = {json.dumps(_ALL_CAPS)};
      const docker = sharedCaps({{ runtime: 'docker', capabilities: {json.dumps(_NO_CAPS)} }});
      out = {{
        four: memberNoticeText('Helper', 'Olga', w, proc),
        on: memberNoticeText('Helper', 'Olga', w, proc, true),
        off: memberNoticeText('Helper', 'Olga', w, proc, false),
        bare: memberNoticeText('Helper', 'Olga', '', proc, true),
        dockerOn: memberNoticeText('Helper', 'Olga', w, docker, true),
        dockerFour: memberNoticeText('Helper', 'Olga', w, docker),
        // Files off but deep memory on (not a deck we ship, but the rule is files).
        noFiles: memberNoticeText('Helper', 'Olga', w, {{ ...proc, files: false }}, true),
        noFilesFour: memberNoticeText('Helper', 'Olga', w, {{ ...proc, files: false }}),
        olderDeck: memberNoticeText('Helper', 'Olga', w, sharedCaps({{}}), true),
        noOwner: memberNoticeText('Helper', '', '', proc, true),
        nulled: memberNoticeText('Helper', 'Olga', '', null, true),
      }};
    """)
    a2 = _a2_notice("Helper", "Olga")
    assert out["four"] == a2 + "\n\n" + HOST_TRUST_WARNING
    assert out["on"] == a2 + " " + _workspace_notice("Olga") + "\n\n" + HOST_TRUST_WARNING
    assert out["off"] == out["four"]
    assert out["bare"] == a2 + " " + _workspace_notice("Olga")
    assert out["dockerOn"] == out["dockerFour"]
    assert "saved/" not in out["dockerOn"]
    assert out["noFiles"] == out["noFilesFour"]
    assert "saved/" not in out["olderDeck"]
    # The notice's own owner fallback carries into the commons paragraph.
    assert out["noOwner"] == (
        _a2_notice("Helper", "another user") + " " + _workspace_notice("another user"))
    assert "saved/" not in out["nulled"]


def test_ack_prefix_is_v4():
    out = _lift([(_SHARED, ["SHARED_ACK_PREFIX", "sliceUser", "sharedAckKey"])], """
      out = { prefix: SHARED_ACK_PREFIX, key: sharedAckKey('u1', 'process:x:0123456789abcdef') };
    """)
    assert out == {"prefix": "fd.sharedAgentAck.v4.",
                   "key": "fd.sharedAgentAck.v4.u1.process:x:0123456789abcdef"}


def test_shared_caps_carry_the_datastore():
    out = _lift([(_SHARED, ["CHAT_ONLY_CAPS", "sharedCaps"])], """
      out = {
        on: sharedCaps({ capabilities: { datastore: true } }),
        truthy: sharedCaps({ capabilities: { datastore: 1 } }),
        a2: sharedCaps({ capabilities: { google: true, deep_memory: true, files: true } }),
        constant: CHAT_ONLY_CAPS,
      };
    """)
    assert out["on"] == {**_NO_CAPS, "datastore": True}
    assert out["truthy"] == _NO_CAPS
    assert out["a2"] == {**_ALL_CAPS, "datastore": False}
    assert out["constant"] == _NO_CAPS


# ── the owner's Share note ──────────────────────────────────────────


def test_owner_share_note_with_the_commons():
    out = _lift([(_SHARED, ["OWNER_SHARE_NOTE", "OWNER_WORKSPACE_NOTE", "ownerShareNote"])], """
      out = {
        base: OWNER_SHARE_NOTE,
        plain: ownerShareNote(false, 'process', false),
        ws: ownerShareNote(false, 'process', true),
        both: ownerShareNote(true, 'process', true),
        dockerBoth: ownerShareNote(true, 'docker', true),
        docker2: ownerShareNote(true, 'docker'),
        dockerWsOnly: ownerShareNote(false, 'docker', true),
        packs: ownerShareNote(true, 'process', false),
        packs2: ownerShareNote(true, 'process'),
        truthy: ownerShareNote(false, 'process', 'yes'),
      };
    """)
    assert out["base"] == _OWNER_NOTE
    assert out["plain"] == _OWNER_NOTE
    host_tail = "\n\n" + OWNER_HOST_NOTE
    main = _OWNER_NOTE[: -len(host_tail)]
    reworded = main.replace(
        "the shell, your files or accounts",
        "the shell, your accounts or your files outside its saved/ folder")
    assert reworded != main
    ws = out["ws"]
    assert ws == reworded + "\n\n" + OWNER_WORKSPACE_NOTE + host_tail
    assert OWNER_WORKSPACE_NOTE in ws
    assert "your files outside its saved/ folder" in ws
    assert "your files or accounts" not in ws
    assert ws.endswith(host_tail)
    # Packs first, then the commons, then the host-trust paragraph.
    assert out["both"] == (
        reworded + "\n\n" + _OWNER_PACKS_NOTE + "\n\n" + OWNER_WORKSPACE_NOTE + host_tail)
    # Docker members stay chat-only: no commons paragraph, no rewording.
    assert out["dockerBoth"] == out["docker2"]
    assert out["dockerWsOnly"] == _OWNER_NOTE
    # Packs only: byte-identical to PR B.
    assert out["packs"] == out["packs2"] == main + "\n\n" + _OWNER_PACKS_NOTE + host_tail
    assert out["truthy"] == _OWNER_NOTE


# ── helpers ─────────────────────────────────────────────────────────


def test_creator_labels_and_owner_badges():
    out = _lift(_ALL_WS, """
      const me = { kind: 'me', name: '' };
      const owner = { kind: 'owner', name: 'Olga' };
      const ownerBlank = { kind: 'owner', name: '' };
      const ana = { kind: 'member', name: ' Ana ' };
      const blank = { kind: 'member', name: '  ' };
      out = {
        labels: [creatorLabel(me), creatorLabel(owner), creatorLabel(ownerBlank),
                 creatorLabel(ownerBlank, 'Olga'), creatorLabel(owner, 'Someone'),
                 creatorLabel(ana), creatorLabel(blank), creatorLabel(null), creatorLabel(undefined)],
        badge: [ownerBadgeLabel(ana), ownerBadgeLabel(blank), ownerBadgeLabel(owner),
                ownerBadgeLabel(me), ownerBadgeLabel(null)],
        title: [ownerBadgeTitle(ana), ownerBadgeTitle(blank), ownerBadgeTitle(owner),
                ownerBadgeTitle(undefined)],
        member: [isMemberCreated(ana), isMemberCreated(owner), isMemberCreated(me),
                 isMemberCreated(null), isMemberCreated(undefined)],
        del: deleteConfirmText('report.pdf'),
      };
    """)
    assert out["labels"] == [
        "You", "Olga (owner)", "Owner (owner)", "Olga (owner)", "Olga (owner)",
        "Ana", "A member", "", "",
    ]
    assert out["badge"] == ["✎ Ana", "✎ a member", "", "", ""]
    assert out["title"] == [
        "Added by Ana (a member of this shared agent)",
        "Added by a member (a member of this shared agent)", "", "",
    ]
    assert out["member"] == [True, False, False, False, False]
    assert out["del"] == "Delete “report.pdf”? Everyone who uses this agent loses it."


_UPLOAD = {"max_bytes": 26214400, "extensions": [".csv", ".pdf", ".png", ".txt"]}


def test_upload_error():
    out = _lift(_ALL_WS, f"""
      const up = {json.dumps(_UPLOAD)};
      const e = (name, size) => uploadError({{ name, size }}, up);
      out = {{
        zip: e('a.zip', 10), upper: e('REPORT.PDF', 10), ok: e('notes.txt', 1),
        big: e('a.pdf', 26214401), max: e('a.pdf', 26214400), noDot: e('README', 1),
        trailing: e('a.', 1), html: e('page.html', 1), svg: e('x.svg', 1),
        double: e('a.tar.csv', 1), hidden: e('.csv', 1),
        bigZip: e('a.zip', 26214401),
        mb: uploadError({{ name: 'a.pdf', size: 3 * 1048576 }}, {{ max_bytes: 2 * 1048576 + 5, extensions: ['.pdf'] }}),
      }};
    """)
    no = "That kind of file can’t be uploaded here"
    assert out["zip"] == no and out["noDot"] == no and out["trailing"] == no
    assert out["html"] == no and out["svg"] == no
    assert out["upper"] == "" and out["ok"] == "" and out["max"] == "" and out["double"] == ""
    assert out["hidden"] == ""
    assert out["big"] == "That file is too large (25 MB at most)"
    # The kind is checked first.
    assert out["bigZip"] == no
    assert out["mb"] == "That file is too large (2 MB at most)"


def test_workspace_visible_needs_the_flag_and_the_cap():
    out = _lift(_ALL_WS, """
      const all = { files: true, datastore: true };
      out = {
        on: workspaceVisible(true, all),
        off: workspaceVisible(false, all),
        filesOnly: workspaceVisible(true, { files: true, datastore: false }),
        dataOnly: workspaceVisible(true, { files: false, datastore: true }),
        none: workspaceVisible(true, { files: false, datastore: false }),
        truthy: workspaceVisible('yes', { files: 1, datastore: 'true' }),
      };
    """)
    assert out == {
        "on": {"files": True, "datastore": True},
        "off": {"files": False, "datastore": False},
        "filesOnly": {"files": True, "datastore": False},
        "dataOnly": {"files": False, "datastore": True},
        "none": {"files": False, "datastore": False},
        "truthy": {"files": False, "datastore": False},
    }


def test_to_viewer_file_names_the_file_by_its_id():
    out = _lift(_ALL_WS, """
      out = toViewerFile({ id: 'downloads/s1/a.md', path: 'saved/downloads/s1/a.md',
        filename: 'a.md', extension: '.md', size: 3, modified: 7, mime_type: 'text/markdown',
        is_text: true, created_by: { kind: 'member', name: 'Ana' }, can_delete: false });
    """)
    assert out == {
        "logical": "saved/downloads/s1/a.md", "physical": "downloads/s1/a.md",
        "filename": "a.md", "extension": ".md", "exists": True, "size": 3, "modified": 7,
        "mime_type": "text/markdown", "is_text": True, "source": "shared",
        "created_by": {"kind": "member", "name": "Ana"},
    }


def test_creator_badge_renders_per_mode():
    out = _lift([
        (_WS_UTILS, ["CREATED_BY_COLUMN", "creatorLabel", "isMemberCreated", "ownerBadgeLabel",
                     "ownerBadgeTitle"]),
        (_BADGE, ["MEMBER_TINTS", "CreatorBadge"]),
    ], r"""
      const React = { createElement: (type, props, ...children) =>
        ({ type, props: props || {}, children: children.flat(Infinity) }) };
      const text = (n) => (n == null || n === false) ? ''
        : typeof n === 'object' ? (n.children || []).map(text).join('') : String(n);
      const r = (props) => { const el = CreatorBadge(props); return el && { text: text(el), title: el.props.title, cls: el.props.className }; };
      const ana = { kind: 'member', name: 'Ana' };
      out = {
        mMe: r({ mode: 'member', creator: { kind: 'me', name: '' } }),
        mOwner: r({ mode: 'member', creator: { kind: 'owner', name: '' }, ownerName: 'Olga' }),
        mAna: r({ mode: 'member', creator: ana }),
        mNone: r({ mode: 'member', creator: null }),
        oAna: r({ mode: 'owner', creator: ana }),
        oOwner: r({ mode: 'owner', creator: { kind: 'owner', name: '' } }),
        oNone: r({ mode: 'owner' }),
      };
    """, tsx=True)
    assert out["mMe"]["text"] == "You" and "violet" in out["mMe"]["cls"]
    assert out["mOwner"]["text"] == "Olga (owner)" and "sky" in out["mOwner"]["cls"]
    assert out["mAna"]["text"] == "Ana" and "zinc" in out["mAna"]["cls"]
    assert out["mNone"] is None
    assert out["oAna"] == {
        "text": "✎ Ana", "title": "Added by Ana (a member of this shared agent)",
        "cls": out["oAna"]["cls"],
    }
    assert "amber" in out["oAna"]["cls"]
    # The owner's own and legacy items look as before.
    assert out["oOwner"] is None and out["oNone"] is None


def test_untrusted_markdown_drops_remote_image_sources_only():
    out = _lift([(_FILE_VIEWER, ["untrustedUrlTransform"])], r"""
      const defaultUrlTransform = (u) => 'T:' + u;
      const t = untrustedUrlTransform;
      out = {
        https: t('https://x.example/p.png', 'src'), http: t('HTTP://x/p.png', 'src'),
        proto: t('//x.example/p.png', 'src'), space: t('  //x/p.png', 'src'),
        back: t('/\\x.example/p.png', 'src'), tab: t('/\t/x/p.png', 'src'),
        rel: t('img/a.png', 'src'), root: t('/a.png', 'src'),
        link: t('https://x.example/', 'href'),
        // Browsers strip every leading C0 control or space: `<\x01//host>`
        // is a valid markdown destination that loads from `host`.
        ctl: t('\x01//x.example/p.png', 'src'), nul: t('\x00\x1f https://x/p.png', 'src'),
      };
    """)
    for key in ("https", "http", "proto", "space", "back", "tab", "ctl", "nul"):
        assert out[key] == "", key
    assert out["rel"] == "T:img/a.png" and out["root"] == "T:/a.png"
    # Links still render (through the default transform).
    assert out["link"] == "T:https://x.example/"


# ── service + store ─────────────────────────────────────────────────

_REF = "process:helper:0123456789abcdef"


def test_service_requests_name_the_agent_by_ref_only():
    out = _lift([(_WS_SERVICE, [
        "FD_BASE", "listSharedFiles", "sharedFileUrl", "uploadSharedFile", "deleteSharedFile",
        "listSharedTables",
    ])], f"""
      const R = {json.dumps(_REF)};
      const seen = [];
      const fdFetch = async (path, init) => {{
        seen.push([path, init ? {{ method: init.method,
          body: typeof init.body === 'string' ? JSON.parse(init.body) : init.body && init.body.parts }} : null]);
        return {{}};
      }};
      class FormData {{ constructor() {{ this.parts = []; }} append(k, v) {{ this.parts.push([k, v]); }} }}
      let auth = {{ token: 'jwt a/b', authEnabled: true }};
      const useAuthStore = {{ getState: () => auth }};
      done = (async () => {{
        await listSharedFiles(R);
        await uploadSharedFile(R, 'FILE');
        await uploadSharedFile(R, 'FILE', 'B');
        await deleteSharedFile(R, 'downloads/s 1/a.pdf');
        await listSharedTables(R);
        const view = sharedFileUrl(R, 'downloads/s 1/a&b.pdf', 'view');
        const dl = sharedFileUrl(R, 'x.csv', 'download');
        auth = {{ token: '', authEnabled: false }};
        const anon = sharedFileUrl(R, 'x.csv', 'view');
        out = {{ seen, view, dl, anon }};
      }})();
    """)
    enc = "process%3Ahelper%3A0123456789abcdef"
    assert out["seen"] == [
        [f"/shared-agents/files?ref={enc}", None],
        [f"/shared-agents/files/upload?ref={enc}&lane=A", {"method": "POST", "body": [["file", "FILE"]]}],
        [f"/shared-agents/files/upload?ref={enc}&lane=B", {"method": "POST", "body": [["file", "FILE"]]}],
        [f"/shared-agents/files/delete?ref={enc}",
         {"method": "POST", "body": {"id": "downloads/s 1/a.pdf"}}],
        [f"/shared-agents/datastore/tables?ref={enc}", None],
    ]
    assert out["view"] == (
        f"/fd/shared-agents/files/view?ref={enc}&id=downloads%2Fs%201%2Fa%26b.pdf&fd_token=jwt%20a%2Fb")
    assert out["dl"] == f"/fd/shared-agents/files/download?ref={enc}&id=x.csv&fd_token=jwt%20a%2Fb"
    assert out["anon"] == f"/fd/shared-agents/files/view?ref={enc}&id=x.csv"


def test_service_has_only_member_routes_and_no_host_path():
    found = _query(_WS_SERVICE, r"""
      const lit = (n) => {
        if (ts.isStringLiteral(n) || ts.isNoSubstitutionTemplateLiteral(n)) return n.text;
        if (ts.isTemplateExpression(n)) return n.head.text + n.templateSpans.map((s) => '${' + s.expression.getText(sf) + '}' + s.literal.text).join('');
        return null;
      };
      // (A template's middle/tail parts aren't string literals: no doubles.)
      const literals = all.map(lit).filter((s) => s !== null);
      const idents = all.filter((n) => ts.isIdentifier(n)).map((n) => n.text);
      const props = all.filter((n) => ts.isInterfaceDeclaration(n))
        .flatMap((i) => i.members.map((m) => m.name ? m.name.getText(sf) : ''));
      return { literals, idents, props };
    """)
    routes = [s for s in found["literals"] if "/shared-agents" in s]
    paths = {s[s.index("/shared-agents"):].split("?")[0] for s in routes}
    assert paths == {
        "/shared-agents/files", "/shared-agents/files/${mode}", "/shared-agents/files/upload",
        "/shared-agents/files/delete", "/shared-agents/datastore/tables",
    }
    for s in routes:
        assert "?ref=" in s, s
    for s in found["literals"]:
        assert "/agent-" not in s, s
        assert "token=" not in s.replace("fd_token=", ""), s
    for banned in ("host", "port", "auth_token", "web_auth"):
        assert banned not in found["idents"], banned
        assert banned not in found["props"], banned


def test_store_takes_member_workspace_only_on_an_explicit_true():
    out = _lift([(_SHARED_STORE, ["useSharedAgentStore"])], r"""
      function create(init) {
        let state;
        const set = (p) => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; };
        const get = () => state;
        state = init(set, get);
        return { getState: get };
      }
      let server = { enabled: true, host_warning: '', agents: [], member_workspace: true };
      let fail = false;
      const getSharedAgents = async () => { if (fail) throw new Error('x'); return JSON.parse(JSON.stringify(server)); };
      const setSharedAgentGoogle = async () => ({});
      const leaveShare = async () => {};
      const sharedContainerId = (r) => 'shared:' + r;
      const clearSharedSlices = () => {};
      const useChatStore = { getState: () => ({ sessions: new Map(), disconnectChat: () => {} }) };
      const st = () => useSharedAgentStore.getState();
      done = (async () => {
        const res = { initial: st().memberWorkspace };
        await st().fetch(); res.on = st().memberWorkspace;
        fail = true; await st().fetch(); res.blip = st().memberWorkspace; fail = false;
        server = { ...server, member_workspace: 'true' }; await st().fetch(); res.truthy = st().memberWorkspace;
        server = { ...server, member_workspace: undefined }; await st().fetch(); res.olderDeck = st().memberWorkspace;
        server = { ...server, member_workspace: true, enabled: false }; await st().fetch(); res.sharingOff = st().memberWorkspace;
        out = res;
      })();
    """)
    assert out == {
        "initial": False, "on": True, "blip": True, "truthy": False, "olderDeck": False,
        "sharingOff": False,
    }


def test_types_declare_the_new_fields():
    resp = _query(_SHARED_SERVICE, r"""
      const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'SharedAgentsResponse');
      const m = i.members.find((x) => x.name && x.name.getText(sf) === 'member_workspace');
      return m ? { optional: !!m.questionToken, type: m.type.getText(sf) } : null;
    """)
    assert resp == {"optional": True, "type": "boolean"}
    agent_file = _query(_FILE_TRANSFER, r"""
      const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'AgentFile');
      const m = i.members.find((x) => x.name && x.name.getText(sf) === 'created_by');
      const imp = all.find((n) => ts.isImportDeclaration(n) && n.moduleSpecifier.text === '../utils/sharedWorkspace');
      return { field: m ? { optional: !!m.questionToken, type: m.type.getText(sf) } : null,
               typeOnly: !!(imp && imp.importClause && imp.importClause.isTypeOnly) };
    """)
    assert agent_file == {"field": {"optional": True, "type": "Creator | null"}, "typeOnly": True}


# ── active content ──────────────────────────────────────────────────


def test_previews_never_run_same_origin():
    for src in (_FILE_VIEWER, _VFS_VIEWER, _FOLDERS_PAGE):
        text = src.read_text(encoding="utf-8")
        assert "allow-same-origin" not in text, src.name
        frames = _query(src, _JSX_HELPERS + r"""
          return named('iframe').filter((n) => attr(n, 'srcDoc')).map((n) => attr(n, 'sandbox'));
        """)
        # The file viewer picks its sandbox per file (see the next test).
        want = "{htmlFrame(content, untrusted).sandbox}" if src == _FILE_VIEWER else '"allow-scripts"'
        assert frames == [want], src.name
    # Only the app runner keeps it, on purpose (documented there).
    keep = sorted(p.relative_to(_SRC).as_posix() for p in _SRC.rglob("*.ts*")
                  if "allow-same-origin" in p.read_text(encoding="utf-8", errors="replace"))
    assert keep == ["pages/AppCodePage.tsx"]


_INERT_CSP = (
    """<meta http-equiv="Content-Security-Policy" content="default-src 'none'; """
    """style-src 'unsafe-inline'; img-src data:">"""
)


def test_someone_elses_html_renders_inert():
    out = _lift([(_FILE_VIEWER, ["INERT_HTML_CSP", "htmlFrame"])], r"""
      const page = '<!doctype html><img src="https://x.example/p.png"><script>alert(1)</script>';
      out = { theirs: htmlFrame(page, true), mine: htmlFrame(page, false), csp: INERT_HTML_CSP };
    """, tsx=True)
    page = '<!doctype html><img src="https://x.example/p.png"><script>alert(1)</script>'
    assert out["csp"] == _INERT_CSP
    # Someone else's: no scripts at all, and nothing loads from anywhere.
    assert out["theirs"] == {"srcDoc": _INERT_CSP + page, "sandbox": ""}
    # The viewer's own (the owner's, in the owner's panels): as before.
    assert out["mine"] == {"srcDoc": page, "sandbox": "allow-scripts"}
    found = _query(_FILE_VIEWER, _JSX_HELPERS + r"""
      return named('iframe').map((n) => ({ srcDoc: attr(n, 'srcDoc'), sandbox: attr(n, 'sandbox') }));
    """)
    assert found == [{"srcDoc": "{htmlFrame(content, untrusted).srcDoc}",
                      "sandbox": "{htmlFrame(content, untrusted).sandbox}"}]


def test_next_file_never_shows_the_last_ones_content_under_its_trust():
    # The real FileViewer, rendered by a tiny hooks runtime: a render that sets
    # state re-runs at once (as React does); effects run only when asked.
    out = _lift([
        (_FILE_TRANSFER, ["getFileTypeGroup"]),
        (_WS_UTILS, ["isMemberCreated"]),
        (_FILE_VIEWER, ["EDITABLE_GROUPS", "untrustedUrlTransform", "INERT_HTML_CSP",
                        "htmlFrame", "FileViewer"]),
    ], r"""
      const React = { createElement: (t, p, ...c) => ({ t, p: p || {}, c }), Fragment: 'Fragment' };
      const Icon = 'Icon';
      const X = Icon, Download = Icon, Loader2 = Icon, AlertCircle = Icon, Maximize2 = Icon,
        Minimize2 = Icon, ChevronLeft = Icon, ChevronRight = Icon, Copy = Icon, Check = Icon,
        Pencil = Icon, Save = Icon, Markdown = 'Markdown', CodeEditor = 'CodeEditor';
      const remarkGfm = null, defaultUrlTransform = (u) => u;
      const formatSize = () => '', saveFileContent = async () => {};
      const getViewUrl = (h, p, path) => 'view:' + path, getDownloadUrl = (h, p, path) => 'dl:' + path;
      const window = { addEventListener() {}, removeEventListener() {} };
      const BODY = { '/w/saved/theirs.html': '<b>THEIRS</b>', '/w/saved/mine.html': '<b>MINE</b>' };
      const fetch = (u) => Promise.resolve({ ok: true, text: async () => BODY[u.slice(5)] });
      let hooks = [], i = 0, again = false, effects = [];
      function useState(v) { const k = i++; if (!(k in hooks)) hooks[k] = v;
        return [hooks[k], (x) => { hooks[k] = typeof x === 'function' ? x(hooks[k]) : x; again = true; }]; }
      function useRef(v) { const k = i++; if (!(k in hooks)) hooks[k] = { current: v }; return hooks[k]; }
      function useCallback(f) { i++; return f; }
      function useEffect(f) { i++; effects.push(f); }
      function render(props) {
        let tree, n = 0;
        do { again = false; i = 0; effects = []; tree = FileViewer(props); } while (again && ++n < 10);
        return tree;
      }
      const settle = async () => { for (let n = 0; n < 20; n++) await Promise.resolve(); };
      const frames = (t) => !t || typeof t !== 'object' ? []
        : Array.isArray(t) ? t.flatMap(frames)
        : [...(t.t === 'iframe' ? [{ srcDoc: t.p.srcDoc, sandbox: t.p.sandbox }] : []), ...t.c.flatMap(frames)];
      const file = (name, kind) => ({ logical: 'saved/' + name, physical: '/w/saved/' + name, filename: name,
        extension: '.html', exists: true, size: 1, modified: 0, mime_type: 'text/html', is_text: true,
        source: 'workspace', created_by: { kind, name: kind } });
      const theirs = { file: file('theirs.html', 'member'), onClose() {} };
      const mine = { file: file('mine.html', 'owner'), onClose() {} };
      done = (async () => {
        render(theirs); effects.forEach((f) => f());
        await settle();
        const loaded = frames(render(theirs));
        // Next → the owner's own HTML: the member's text must not run there.
        const switching = frames(render(mine));
        effects.forEach((f) => f());
        await settle();
        out = { loaded, switching, after: frames(render(mine)) };
      })();
    """, tsx=True)
    assert out["loaded"] == [{"srcDoc": _INERT_CSP + "<b>THEIRS</b>", "sandbox": ""}]
    assert out["switching"] == []
    assert out["after"] == [{"srcDoc": "<b>MINE</b>", "sandbox": "allow-scripts"}]


def test_file_viewer_takes_urls_read_only_and_guards_markdown():
    found = _query(_FILE_VIEWER, _JSX_HELPERS + r"""
      const props = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'FileViewerProps');
      const member = (k) => { const m = props.members.find((x) => x.name && x.name.getText(sf) === k);
        return m ? { optional: !!m.questionToken, type: m.type.getText(sf) } : null; };
      const decl = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
      // For each <Markdown>, the condition of the conditional it sits in, and on which side.
      const side = (n) => {
        for (let c = n, p = n.parent; p; c = p, p = p.parent) {
          if (ts.isConditionalExpression(p)) return { cond: p.condition.getText(sf), whenTrue: p.whenTrue === c || p.whenTrue.pos <= c.pos && c.end <= p.whenTrue.end };
        }
        return null;
      };
      return {
        urls: member('urls'), readOnly: member('readOnly'), host: member('host'),
        untrusted: decl('untrusted') ? decl('untrusted').initializer.getText(sf) : null,
        viewUrl: decl('viewUrl').initializer.getText(sf),
        downloadUrl: decl('downloadUrl').initializer.getText(sf),
        editable: decl('editable').initializer.getText(sf),
        autoEdit: decl('autoEditRef').initializer.getText(sf),
        markdown: named('Markdown').map((n) => ({ transform: attr(n, 'urlTransform'), side: side(n) })),
      };
    """)
    assert found["urls"] == {"optional": True, "type": "{ view: string; download: string }"}
    assert found["readOnly"] == {"optional": True, "type": "boolean"}
    assert found["host"]["optional"] is True
    assert found["untrusted"] == (
        "file.source === 'shared' ? file.created_by?.kind !== 'me' : isMemberCreated(file.created_by)")
    assert found["viewUrl"].startswith("urls ? urls.view :")
    assert found["downloadUrl"].startswith("urls ? urls.download :")
    assert found["editable"].startswith("!readOnly &&")
    assert "!readOnly" in found["autoEdit"]
    md = found["markdown"]
    assert len(md) == 2
    with_t = [m for m in md if m["transform"]]
    without = [m for m in md if not m["transform"]]
    assert with_t == [{"transform": "{untrustedUrlTransform}",
                       "side": {"cond": "untrusted", "whenTrue": True}}]
    assert without == [{"transform": None, "side": {"cond": "untrusted", "whenTrue": False}}]


def test_member_html_is_never_offered_as_a_deck():
    for src in (_AGENT_FILES, _FILE_BROWSER):
        found = _query(src, _JSX_HELPERS + r"""
          const decl = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
          return {
            deckable: decl('isDeckable').initializer.getText(sf),
            badges: named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator') })),
          };
        """)
        assert "!isMemberCreated(f.created_by)" in found["deckable"], src.name
        assert found["badges"] == [{"mode": '"owner"', "creator": "{f.created_by}"}], src.name
    tables = _query(_AGENT_DATA, _JSX_HELPERS + r"""
      return named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator') }));
    """)
    assert tables == [{"mode": '"owner"', "creator": "{t.created_by}"}]


def test_pins_keep_the_creator():
    # The pin type carries where the file was listed and who created it…
    fields = _query(_PIN_STORE, r"""
      const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'PinnedFile');
      const m = (k) => { const x = i.members.find((y) => y.name && y.name.getText(sf) === k);
        return x ? { optional: !!x.questionToken, type: x.type.getText(sf) } : null; };
      return { source: m('source'), created_by: m('created_by') };
    """)
    assert fields == {"source": {"optional": True, "type": "string"},
                      "created_by": {"optional": True, "type": "Creator | null"}}
    # …every place that pins copies both…
    for src in (_AGENT_FILES, _FILE_BROWSER, _COUNCIL_FILES):
        calls = _query(src, r"""
          return all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'pinFile')
            .map((c) => c.arguments[0].properties.map((p) => p.getText(sf)));
        """)
        assert len(calls) == 1, src.name
        assert {"source: f.source", "created_by: f.created_by"} <= set(calls[0]), src.name
    # …and the pinned panel hands them to the viewer and badges a member's pin.
    wiring = _query(_PINNED, _JSX_HELPERS + r"""
      return {
        file: named('FileViewer').map((n) => attr(n, 'file')),
        badge: named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator') })),
      };
    """)
    assert wiring == {"file": ["{pinViewerFile(viewingFile.file)}"],
                      "badge": [{"mode": '"owner"', "creator": "{pin.created_by}"}]}
    # The viewer's own rule decides what is someone else's.
    untrusted = _query(_FILE_VIEWER, r"""
      const d = all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === 'untrusted');
      return d.initializer.getText(sf);
    """)
    out = _lift([(_WS_UTILS, ["isMemberCreated"]), (_PINNED, ["pinViewerFile"])], f"""
      const untrusted = new Function('file', 'isMemberCreated', 'return ' + {json.dumps(untrusted)});
      const base = {{ id: 'p', agentId: 'a', agentName: 'A', host: 'h', port: 1, auth: 't',
        filename: 'x.html', extension: '.html', physical: '/w/saved/x.html', logical: 'saved/x.html',
        size: 1, mime_type: 'text/html', pinnedAt: '', tags: [] }};
      const run = (extra) => {{ const f = pinViewerFile({{ ...base, ...extra }});
        return {{ source: f.source, created_by: f.created_by, untrusted: untrusted(f, isMemberCreated) }}; }};
      out = {{
        member: run({{ source: 'workspace', created_by: {{ kind: 'member', name: 'Ana' }} }}),
        owner: run({{ source: 'workspace', created_by: {{ kind: 'owner', name: 'Olga' }} }}),
        legacy: run({{}}),
        sharedNoCreator: run({{ source: 'shared' }}),
      }};
    """, tsx=True)
    assert out["member"] == {"source": "workspace", "created_by": {"kind": "member", "name": "Ana"},
                             "untrusted": True}
    assert out["owner"]["untrusted"] is False
    # A pin from before member files opens as it always did…
    assert out["legacy"] == {"source": "", "created_by": None, "untrusted": False}
    # …but one from a member's context with no creator is someone else's.
    assert out["sharedNoCreator"]["untrusted"] is True


# ── wiring ──────────────────────────────────────────────────────────

_GUARDS = r"""
const guards = (n) => {
  const out = [];
  for (let p = n.parent; p; p = p.parent) if (ts.isJsxExpression(p) && p.expression) out.push(p.expression.getText(sf));
  return out;
};
const decl = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
const init = (name) => (decl(name) && decl(name).initializer ? decl(name).initializer.getText(sf) : null);
"""


def test_datastore_browser_member_mode():
    found = _query(_DS_BROWSER, _JSX_HELPERS + _GUARDS + r"""
      const effects = all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'useEffect'
          && n.arguments[0].getText(sf).includes('fetchTables()'));
      // No URL piece (template or string concatenation) that names sharedRef
      // also names host/port/auth.
      const mixes = all.filter((n) => ts.isTemplateExpression(n)
          || (ts.isBinaryExpression(n) && n.operatorToken.kind === ts.SyntaxKind.PlusToken))
        .filter((n) => { const ids = []; (function w(x) { if (ts.isIdentifier(x)) ids.push(x.text); ts.forEachChild(x, w); })(n);
          return ids.includes('sharedRef') && (ids.includes('host') || ids.includes('port') || ids.includes('auth')); })
        .map((n) => n.getText(sf));
      const exp = all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === 'url'
          && n.initializer.getText(sf).includes('/export?format='));
      return {
        deps: effects.map((e) => e.arguments[1].getText(sf)),
        base: init('base'), tokenQs: init('tokenQs'), mixes,
        exportUrl: exp ? exp.initializer.getText(sf) : null,
        showCreator: init('showCreator'), rowColumns: init('rowColumns'),
        heads: all.filter((n) => ts.isJsxExpression(n) && n.expression && n.expression.getText(sf) === 'CREATED_BY_COLUMN')
          .map((n) => guards(n)),
        badges: named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator'), guards: guards(n) })),
        props: (() => { const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'DatastoreBrowserProps');
          return i.members.map((m) => m.name.getText(sf)); })(),
      };
    """)
    assert found["deps"] == ["[host, port, vfsProject, sharedRef]"]
    assert "'/shared-agents/datastore'" in found["base"]
    assert found["base"].index("isShared") < found["base"].index("'/shared-agents/datastore'")
    assert "'&ref=' + encodeURIComponent(sharedRef as string)" in found["tokenQs"]
    assert found["mixes"] == []
    assert found["exportUrl"] == (
        "`/fd${base}/tables/${encodeURIComponent(selectedTable)}/export?format=${format}${tokenQs}`")
    # Owner mode: the column only when some row is a member's.
    assert found["showCreator"].startswith("!!rows && (isShared ||")
    assert "_creator?.kind === 'member'" in found["showCreator"]
    assert "c !== '_creator'" in found["rowColumns"]
    assert len(found["heads"]) == 1 and found["heads"][0][0].startswith("showCreator && (")
    assert {"sharedRef", "ownerName"} <= set(found["props"])
    modes = [(b["mode"], b["creator"]) for b in found["badges"]]
    assert modes == [
        ("{sharedRef ? 'member' : 'owner'}", "{table.created_by}"),
        ("{isShared ? 'member' : 'owner'}", "{row._creator}"),
    ]
    assert found["badges"][1]["guards"][0].startswith("showCreator && (")


def test_files_panel_is_read_and_own_delete_only():
    text = _FILES_PANEL.read_text(encoding="utf-8")
    for banned in ("usePinnedFilesStore", "transferFile", "/deck/view", "saveFileContent",
                   "getViewUrl", "getDownloadUrl", "listAgentFiles"):
        assert banned not in text, banned
    found = _query(_FILES_PANEL, _JSX_HELPERS + _GUARDS + r"""
      const fn = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
      const viewer = named('FileViewer')[0];
      const btns = all.filter((n) => ts.isJsxSelfClosingElement(n) && tagName(n) === 'IconBtn')
        .map((n) => ({ title: attr(n, 'title'), guards: guards(n) }));
      const pick = fn('handlePick').initializer.getText(sf);
      return {
        viewer: viewer && {
          file: attr(viewer, 'file'), urls: attr(viewer, 'urls'), readOnly: attr(viewer, 'readOnly'),
          host: attr(viewer, 'host'), port: attr(viewer, 'port'), auth: attr(viewer, 'auth'),
          startInEdit: attr(viewer, 'startInEdit'),
        },
        btns,
        pickOrder: pick.indexOf('uploadError(') >= 0 && pick.indexOf('uploadError(') < pick.indexOf('uploadSharedFile('),
        pickLane: pick.includes("uploadSharedFile(agentRef, file, 'A')"),
        view: fn('handleView').initializer.getText(sf),
        del: fn('handleDelete').initializer.getText(sf),
        accept: attr(all.find((n) => tagName(n) === 'input' && attr(n, 'type') === '"file"'), 'accept'),
        texts: all.filter((n) => ts.isJsxExpression(n) && n.expression && ts.isIdentifier(n.expression)
          && /^[A-Z_]+$/.test(n.expression.text)).map((n) => n.expression.text),
        badge: named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator') })),
      };
    """)
    v = found["viewer"]
    assert v["file"] == "{toViewerFile(viewing)}"
    assert "sharedFileUrl(agentRef, viewing.id, 'view')" in v["urls"]
    assert "sharedFileUrl(agentRef, viewing.id, 'download')" in v["urls"]
    assert v["readOnly"] is True
    assert v["host"] is None and v["port"] is None and v["auth"] is None and v["startInEdit"] is None
    titles = {b["title"]: b["guards"] for b in found["btns"]}
    assert set(titles) == {'"View"', '"Download"', '"Delete"'}
    assert any("f.can_delete" in g for g in titles['"Delete"'])
    assert any("isViewable(toViewerFile(f))" in g for g in titles['"View"'])
    assert found["pickOrder"] is True and found["pickLane"] is True
    assert "sharedFileUrl(agentRef, f.id, 'view')" in found["view"] and "'pdf'" in found["view"]
    assert "confirm(deleteConfirmText(f.filename))" in found["del"]
    assert found["del"].index("confirm(") < found["del"].index("deleteSharedFile(")
    assert "upload.extensions.join(',')" in found["accept"]
    for t in ("UPLOAD_LABEL", "MEMBER_FILES_NOTE", "TRUNCATED_NOTE"):
        assert t in found["texts"], t
    assert found["badge"] == [{"mode": '"member"', "creator": "{f.created_by}"}]


def test_datastore_panel_is_read_only():
    found = _query(_DATA_PANEL, _JSX_HELPERS + r"""
      const b = named('DatastoreBrowser')[0];
      return {
        attrs: b && Object.fromEntries(['sharedRef', 'agentName', 'ownerName', 'initialTable', 'host', 'port', 'auth']
          .map((k) => [k, attr(b, k)])),
        note: all.some((n) => ts.isJsxExpression(n) && n.expression && n.expression.getText(sf) === 'MEMBER_DATA_NOTE'),
        empty: sf.getFullText().includes('NO_SHARED_TABLES'),
        badge: named('CreatorBadge').map((n) => ({ mode: attr(n, 'mode'), creator: attr(n, 'creator') })),
      };
    """)
    assert found["attrs"] == {
        "sharedRef": "{agentRef}", "agentName": "{agentName}", "ownerName": "{ownerName}",
        "initialTable": "{openTable}", "host": None, "port": None, "auth": None,
    }
    assert found["note"] is True and found["empty"] is True
    assert found["badge"] == [{"mode": '"member"', "creator": "{t.created_by}"}]


def test_context_column_shows_member_panels_only_behind_the_flag():
    found = _query(_SIMPLE, _JSX_HELPERS + _GUARDS + r"""
      const col = all.find((n) => ts.isFunctionDeclaration(n) && n.name && n.name.text === 'ContextColumn');
      const inCol = (n) => n.pos >= col.pos && n.end <= col.end;
      const panels = all.filter(inCol).filter((n) => ['SharedFilesPanel', 'SharedDatastorePanel'].includes(tagName(n)));
      const chatOnly = all.filter(inCol).filter((n) => ts.isJsxExpression(n) && n.expression && n.expression.getText(sf) === 'CHAT_ONLY_TEXT');
      return {
        vis: init('vis'), memberWorkspace: init('memberWorkspace'), sharedRow: init('sharedRow'),
        panels: panels.map((n) => ({ tag: tagName(n), guards: guards(n), ref: attr(n, 'agentRef'),
          owner: attr(n, 'ownerName'), key: attr(n, 'key') })),
        chatOnly: chatOnly.map((n) => guards(n)),
        literal: col.getText(sf).includes('A shared agent is chat only'),
      };
    """)
    assert found["vis"] == "workspaceVisible(memberWorkspace, sharedCaps(sharedRow))"
    assert found["memberWorkspace"] == "useSharedAgentStore((s) => s.memberWorkspace)"
    assert "s.agents.find(" in found["sharedRow"] and "sharedContainerId(a.agent_ref) === agentId" in found["sharedRow"]
    tags = sorted(p["tag"] for p in found["panels"])
    assert tags == ["SharedDatastorePanel", "SharedDatastorePanel", "SharedFilesPanel", "SharedFilesPanel"]
    for p in found["panels"]:
        outer = p["guards"][-1]
        assert "shared" in outer and "sharedRow" in outer and "(vis.files || vis.datastore)" in outer
        assert any("vis." in g for g in p["guards"][:-1]) or p["guards"][0] == outer
        assert p["ref"] == "{sharedRow.agent_ref}"
        assert p["owner"] == "{sharedRow.owner_name || sharedRow.owner_email}"
        assert p["key"] == "{agentId}"
    files_guards = [p["guards"][0] for p in found["panels"] if p["tag"] == "SharedFilesPanel"]
    data_guards = [p["guards"][0] for p in found["panels"] if p["tag"] == "SharedDatastorePanel"]
    assert all("vis.files" in g for g in files_guards)
    assert all("vis.datastore" in g or "vis.files ?" in g for g in data_guards)
    # The chat-only text stays for everything else that is shared.
    assert len(found["chatOnly"]) == 1 and found["literal"] is False


def test_member_card_buttons_behind_the_flag():
    found = _query(_SHARED_CARD, _JSX_HELPERS + _GUARDS + r"""
      const button = (label) => all.filter((n) => ts.isJsxElement(n) && tagName(n) === 'button'
        && n.children.some((c) => ts.isJsxExpression(c) && c.expression && c.expression.getText(sf) === label))
        .map((n) => ({ guards: guards(n).map((g) => g.split(' &&')[0]), disabled: attr(n, 'disabled'),
          icon: n.children.map((c) => tagName(c)).filter(Boolean) }));
      const dlg = named('SharedFilesDialog').map((n) => ({ guards: guards(n), ref: attr(n, 'agentRef') }));
      const ds = named('DatastoreBrowser').map((n) => ({ guards: guards(n), ref: attr(n, 'sharedRef'),
        owner: attr(n, 'ownerName'), host: attr(n, 'host') }));
      const inner = named('SharedFilesPanel').map((n) => attr(n, 'agentRef'));
      return { vis: init('vis'), files: button('FILES_BUTTON'), data: button('DATA_BUTTON'), dlg, ds, inner,
        mw: init('memberWorkspace') };
    """)
    assert found["vis"] == "workspaceVisible(memberWorkspace, sharedCaps(agent))"
    assert found["mw"] == "useSharedAgentStore((s) => s.memberWorkspace)"
    assert found["files"] == [{"guards": ["vis.files"], "disabled": "{!running}", "icon": ["FolderOpen"]}]
    assert found["data"] == [{"guards": ["vis.datastore"], "disabled": "{!running}", "icon": ["Database"]}]
    assert len(found["dlg"]) == 1 and "vis.files" in found["dlg"][0]["guards"][0]
    assert found["dlg"][0]["ref"] == "{agent.agent_ref}"
    assert len(found["ds"]) == 1 and "vis.datastore" in found["ds"][0]["guards"][0]
    assert found["ds"][0]["ref"] == "{agent.agent_ref}" and found["ds"][0]["host"] is None
    assert found["ds"][0]["owner"] == "{owner}"
    assert found["inner"] == ["{agentRef}"]


def test_chat_notice_and_owner_card_wiring():
    chat = _query(_CHAT_PANEL, _JSX_HELPERS + _GUARDS + r"""
      return { notice: named('SharedAgentNotice').map((n) => attr(n, 'workspace')), mw: init('memberWorkspace') };
    """)
    assert chat == {"notice": ["{memberWorkspace}"], "mw": "useSharedAgentStore((s) => s.memberWorkspace)"}
    notice = _query(_NOTICE, r"""
      const call = all.find((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'memberNoticeText');
      const fn = all.find((n) => ts.isFunctionDeclaration(n) && n.name && n.name.text === 'SharedAgentNotice');
      const param = fn.parameters[0].name.elements.find((e) => e.name.getText(sf) === 'workspace');
      return { args: call.arguments.map((a) => a.getText(sf)), dflt: param && param.initializer ? param.initializer.getText(sf) : null };
    """)
    assert notice == {"args": ["agentName", "ownerName", "hostWarning", "caps", "workspace", "ownerReads"],
                      "dflt": "false"}
    proc = _query(_PROCESS_CARD, _JSX_HELPERS + _GUARDS + r"""
      return { notes: named('ShareModal').map((n) => attr(n, 'note')), ws: init('workspaceEnabled') };
    """)
    assert proc == {"notes": ["{ownerShareNote(packsEnabled, 'process', workspaceEnabled, usageEnabled)}"],
                    "ws": "useSharedAgentStore((s) => s.memberWorkspace)"}
    docker = _query(_CONTAINER_CARD, _JSX_HELPERS + r"""
      return named('ShareModal').map((n) => attr(n, 'note'));
    """)
    assert docker == ["{ownerShareNote(packsEnabled, 'docker', false, usageEnabled)}"]
