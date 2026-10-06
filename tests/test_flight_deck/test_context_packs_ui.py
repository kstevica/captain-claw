"""Context packs UI (PR B, part 3), run in Node.

As in test_shared_agent_ui.py: named top-level declarations are lifted out of
the TypeScript sources with the TypeScript compiler from flight-deck/node_modules,
transpiled and run in a Node vm (the dialogs against a tiny hooks runtime and
stubbed services), and a few wiring facts are read off the parsed tree. Skips
when Node or the frontend dependencies aren't installed.

Pinned here:

* every UI text of part 3 §3, byte for byte — incl. the (r3) channels note
  that every publishing text carries (packs reach the owner's channels and
  automations too);
* the pure helpers: disclosure per role, confirm per kind (the folder and its
  vfs:@ name; ALL of the deep memory), summaries, publisher names cleaned as FD
  cleans them with the role as a separate badge (no "(owner)" spoof, only my
  own rows say You), tag picking, folder names, the agent-capability note per
  role (and "couldn't check" for a running agent);
* the dialog: asks before publishing (never before stopping), builds the folder
  name from the publisher's fixed prefix, offers only existing tags, lets the
  owner remove a member's pack after a confirm, shows the outdated / not-running
  notes, keeps its data and shows FD's reason on failure, closes on Esc;
* the Profile card lists my packs everywhere, marks inactive ones, stops one —
  and with sharing off lists them as paused (still stoppable) instead of
  hiding them; the page warns above About me / My company while my profile is
  shared on agents;
* wiring: the service calls only the four routes and has no host-path field,
  the store sets `contextPacks` only from an explicit `true`, every entry
  point (owner cards, member card, member chat bar) sits behind it, and the
  owner cards' Share note tells the owner about members' packs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_flight_deck.test_shared_agent_ui import (  # noqa: F401
    _JSX_HELPERS,
    _lift,
    _query,
    pytestmark,
)

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "flight-deck" / "src"
_UTILS = _SRC / "utils" / "contextPacks.ts"
_SERVICE = _SRC / "services" / "contextPacks.ts"
_SHARED_SERVICE = _SRC / "services" / "sharedAgents.ts"
_SHARED_STORE = _SRC / "stores" / "sharedAgentStore.ts"
_MODAL = _SRC / "components" / "agents" / "ContextPacksModal.tsx"
_MY_PACKS = _SRC / "components" / "profile" / "MyContextPacks.tsx"
_PROCESS_CARD = _SRC / "components" / "agents" / "ProcessCard.tsx"
_CONTAINER_CARD = _SRC / "components" / "agents" / "ContainerCard.tsx"
_SHARED_CARD = _SRC / "components" / "agents" / "SharedAgentCard.tsx"
_CHAT_PANEL = _SRC / "components" / "agents" / "ChatPanel.tsx"
_PROFILE_PAGE = _SRC / "pages" / "ProfilePage.tsx"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Only sources are read and run in a Node vm, but the Node subprocesses
    # inherit this env: never let one see the real ~/.captain-claw.
    monkeypatch.setenv("HOME", str(tmp_path))


# Part 3 §3, exact.
TEXTS = {
    "PACKS_TITLE": "Share with this agent’s people",
    "PACKS_BUTTON": "Shared context",
    "PACKS_MENU_LABEL": "Shared context…",
    "PROFILE_PACK_LABEL": "My profile (about me, company)",
    "PROFILE_PACK_HINT": (
        "Shares the About me and Company from your Profile — never your standing "
        "preferences."
    ),
    "VFS_PACK_LABEL": "Folders (read-only)",
    "VFS_PACK_HINT": (
        "The agent can read the folder’s files (not hidden ones) but can’t change, move "
        "or delete anything. Folders linked from elsewhere and Google Drive folders can’t "
        "be shared. The agent reaches a folder as vfs:@<name>; the name starts with your "
        "own prefix and stays yours on this agent. Stop sharing a folder before you "
        "delete it."
    ),
    "DEEP_PACK_LABEL": "My deep memory",
    "DEEP_PACK_HINT": (
        "The agent’s deep-memory searches also cover your deep memory — all of it, "
        "including documents you indexed from your files or Google Drive, or only the "
        "entries that carry the tags you pick. Files and Drive documents Flight Deck "
        "indexed for you carry no tags, so a tag choice leaves them out; files your agents "
        "indexed carry your agents’ tags, so picking one of those tags includes them. "
        "Nobody else can add to it or delete from it."
    ),
    "TAGS_LABEL": "Only entries tagged (optional)",
    "NO_TAGS_NOTE": "Your deep memory has no tags, so it can only be shared whole.",
    "DOCKER_PACKS_NOTE": "On a Docker agent only your profile can be shared.",
    "AGENT_OUTDATED_NOTE": (
        "This agent runs an older version that ignores shared context. Ask its owner to "
        "restart it (a Docker agent needs its image rebuilt or pulled first). Until then, "
        "vfs:@ names typed in its chats open the person’s own folder of that name, not "
        "the shared one."
    ),
    "AGENT_OUTDATED_OWNER_NOTE": (
        "This agent runs an older version that ignores shared context. Restart this agent "
        "(Docker: rebuild or pull its image first) so it uses shared context. Until then, "
        "vfs:@ names typed in its chats open the person’s own folder of that name, not "
        "the shared one."
    ),
    "AGENT_NOT_RUNNING_NOTE": (
        "This agent isn’t running, so its version can’t be checked. Shared context takes "
        "effect once it runs a current version."
    ),
    "AGENT_UNCHECKED_NOTE": (
        "Couldn’t check this agent’s version. If it runs an older one, it ignores shared "
        "context until it’s restarted, and vfs:@ names typed in its chats open the "
        "person’s own folder of that name, not the shared one."
    ),
    "REVOKE_NOTE": (
        "Stopping takes effect on the agent’s next step for folders and deep memory, and "
        "from its next message for profiles (a reply already running can keep a profile "
        "for up to about 10 minutes). It doesn’t make the agent unlearn what it already "
        "read."
    ),
    "ACTIVE_PACKS_HEADING": "Shared on this agent",
    "NO_PACKS_TEXT": "Nothing is shared on this agent yet.",
    "MY_PACKS_HEADING": "Shared with agents’ people",
    "MY_PACKS_EMPTY": (
        "You don’t share anything with an agent’s people yet. Open an agent (yours or one "
        "shared with you) and choose “Shared context”."
    ),
    "INACTIVE_HINT": (
        "Inactive: you no longer have access to that agent, it changed owner, or the "
        "folder was deleted or replaced."
    ),
    "PACKS_PAUSED_NOTE": (
        "Paused (sharing is off on this deck) — they come back when it’s turned on."
    ),
    "MY_PACKS_ID": "fd-my-context-packs",
    "ALIAS_ERROR": (
        "Use lowercase letters, digits or dashes — the whole name at most 40 characters"
    ),
    "CHANNELS_NOTE": (
        "People who reach this agent through its channels (WhatsApp and glasses, Telegram, "
        "Slack, Discord, the API), its automations and the owner’s other agents — and anyone "
        "it emails or messages — may see what you share."
    ),
}
CHANNELS_NOTE = TEXTS["CHANNELS_NOTE"]

_TAIL = (
    "Your name and whether you’re a member or the owner are shown next to it. Anything the "
    "agent reads from it becomes shared knowledge: it can come up later in any of the agent’s "
    "chats and channels, and it stays after you stop sharing — stopping doesn’t make the agent "
    "unlearn."
)
# The owner also learns what members can publish to their agent (process: all
# three kinds; Docker: profile only).
_OWNER_MEMBERS = (
    " Members can also share their own profile, folders and deep memory with this agent. That "
    "is used on every turn, including your channels and automations. You’re notified and can "
    "remove any of it under “Shared on this agent” below."
)
_CONFIRM_TAIL = (
    "\n\n" + CHANNELS_NOTE
    + " Your name is shown next to it, and what the agent reads from it can become shared "
    "knowledge."
)

_HELPERS = [
    "packsDisclosure", "derivedAlias", "packConfirmText", "packSummary", "cleanPublisherName",
    "publisherName", "packRoleBadge", "packOwnerTag", "packOwnerLabel", "removePackConfirm",
    "profilePackAgents", "profileSharedNote", "toggleTag", "fullAlias", "aliasSuffixError",
    "capabilityNote", "capabilityWarns",
]
_ALL_UTILS = [(_UTILS, list(TEXTS) + _HELPERS)]


def _utils(body: str):
    return _lift(_ALL_UTILS, body)


# ── texts ───────────────────────────────────────────────────────────


def test_every_text_is_exact():
    out = _utils("out = {" + ", ".join(TEXTS) + "};")
    assert out == TEXTS


def test_disclosure_for_a_member_names_the_owner_and_the_channels():
    out = _utils("""
      out = {
        member: packsDisclosure('Helper', 'Olga', 'member'),
        owner: packsDisclosure('Helper', 'Olga', 'owner'),
        blank: packsDisclosure('  ', '', 'member'),
        blankOwner: packsDisclosure('', '', 'owner'),
        dockerOwner: packsDisclosure('Helper', 'Olga', 'owner', 'docker'),
        dockerMember: packsDisclosure('Helper', 'Olga', 'member', 'docker'),
      };
    """)
    member = out["member"]
    assert member == (
        "What you share here is available to everyone who uses Helper: Olga, everywhere the "
        "agent answers or works for them (their Flight Deck chats, its channels and its "
        "automations), and every member, in their own chats. " + CHANNELS_NOTE
        + " Who can reach it there is up to Olga. " + _TAIL
    )
    for part in ("Olga", "every member", CHANNELS_NOTE, "WhatsApp", "glasses", "Telegram",
                 "Slack", "Discord", "the API", "automations", "the owner’s other agents",
                 "anyone it emails or messages", "may see what you share", "up to Olga",
                 "Your name", "unlearn"):
        assert part in member, part

    owner = out["owner"]
    assert owner == (
        "What you share here is available to everyone who uses Helper: you, everywhere the "
        "agent answers or works for you (your Flight Deck chats, its channels and its "
        "automations), and every member, in their own chats. " + CHANNELS_NOTE
        + " Who can reach it there is up to you. " + _TAIL + _OWNER_MEMBERS
    )
    for part in ("you, everywhere the agent answers", CHANNELS_NOTE, "up to you",
                 "including your channels and automations", "remove any of it"):
        assert part in owner, part
    # Only the owner is told about members' packs; a Docker agent takes profiles only.
    assert "Members can also share" not in member
    assert out["dockerOwner"] == owner.replace(
        "their own profile, folders and deep memory", "their own profile")
    assert out["dockerMember"] == member

    # r2's Flight-Deck-only wording is gone (r3: packs reach channels and automations).
    for text in (member, owner):
        for old in ("isn’t used", "main lane (A)", "in Flight Deck:"):
            assert old not in text, old

    assert out["blank"].startswith("What you share here is available to everyone who uses "
                                   "this agent: its owner, everywhere")
    assert "up to its owner. " in out["blank"]
    assert out["blankOwner"].startswith("What you share here is available to everyone who uses "
                                        "this agent: you,")


def test_confirm_text_per_kind_repeats_the_channels_note():
    out = _utils("""
      out = {
        profile: packConfirmText('profile', 'Helper'),
        vfs: packConfirmText('vfs', ' Helper ', [], { project: 'notes', alias: 'ana-my-notes', prefix: 'ana' }),
        vfsDerived: packConfirmText('vfs', 'Helper', [], { project: 'Q3 Plans', alias: '', prefix: 'ana' }),
        vfsBare: packConfirmText('vfs', 'Helper'),
        deep: packConfirmText('deep_memory', 'Helper'),
        deepTags: packConfirmText('deep_memory', 'Helper', ['finance', 'domain:legal']),
        other: packConfirmText('mcp', ''),
      };
    """)
    assert out["profile"] == (
        "Share your About me and Company with everyone who uses Helper?" + _CONFIRM_TAIL
    )
    # A folder names itself and its vfs:@ name (FD's derived one when none is typed).
    assert out["vfs"] == (
        "Share your folder “notes” (as vfs:@ana-my-notes), read-only, with everyone who uses "
        "Helper?" + _CONFIRM_TAIL
    )
    assert out["vfsDerived"] == (
        "Share your folder “Q3 Plans” (as vfs:@ana-q3-plans, or the next free name), read-only, "
        "with everyone who uses Helper?" + _CONFIRM_TAIL
    )
    assert out["vfsBare"] == "Share this folder, read-only, with everyone who uses Helper?" + _CONFIRM_TAIL
    # No tags: the whole pool, Drive and file documents included — said so.
    assert out["deep"] == (
        "Let everyone who uses Helper search ALL of your deep memory, including documents "
        "indexed from your files and Google Drive?" + _CONFIRM_TAIL
    )
    assert out["deepTags"] == (
        "Let everyone who uses Helper search your deep memory (only entries tagged finance, "
        "domain:legal)?" + _CONFIRM_TAIL
    )
    assert out["other"] == "Share this with everyone who uses this agent?" + _CONFIRM_TAIL
    for text in out.values():
        _, rest = text.split("\n\n", 1)
        assert rest.startswith(CHANNELS_NOTE)
        assert text.endswith("shared knowledge.")


def test_pack_summary_and_owner_label():
    out = _utils("""
      out = {
        profile: packSummary({ kind: 'profile' }),
        vfs: packSummary({ kind: 'vfs', project: 'Q3 plans', alias: 'ana-q3-plans' }),
        deepAll: packSummary({ kind: 'deep_memory', tags: [] }),
        deepNone: packSummary({ kind: 'deep_memory' }),
        deepTags: packSummary({ kind: 'deep_memory', tags: ['finance', 'domain:x'] }),
        other: packSummary({ kind: 'mcp' }),
        mine: packOwnerLabel({ mine: true, owner_name: 'Ana', role: 'member' }),
        member: packOwnerLabel({ mine: false, owner_name: ' Ana ', role: 'member' }),
        blank: packOwnerLabel({ mine: false, owner_name: '  ', role: 'member' }),
        absent: packOwnerLabel({}),
        owner: packOwnerLabel({ mine: false, owner_name: 'Olga', role: 'owner' }),
        spoof: packOwnerLabel({ mine: false, owner_name: 'Olga (Owner)', role: 'member' }),
        spoofOnly: packOwnerLabel({ mine: false, owner_name: '(owner)', role: 'member' }),
        tagged: packOwnerLabel({ mine: false, owner_name: 'Ana', role: 'member', owner_tag: '3f9a' }),
        badTag: packOwnerLabel({ mine: false, owner_name: 'Ana', role: 'member', owner_tag: 'x) (owner' }),
        you: packOwnerLabel({ mine: false, owner_name: 'You', role: 'member' }),
        remove: removePackConfirm({ mine: false, owner_name: 'Ana', role: 'member', owner_tag: '#3f9a',
                                    kind: 'vfs', project: 'notes', alias: 'ana-notes' }),
      };
    """)
    assert out == {
        "profile": "Profile (about me, company)",
        "vfs": "Folder “Q3 plans” as vfs:@ana-q3-plans",
        "deepAll": "Deep memory (all of it)",
        "deepNone": "Deep memory (all of it)",
        "deepTags": "Deep memory (tags: finance, domain:x)",
        "other": "Shared context",
        "mine": "You",
        # The role comes from FD's `role` only, on every row, after the quoted name.
        "member": "“Ana” (member)",
        "blank": "Someone (member)",
        "absent": "Someone (member)",
        "owner": "“Olga” (owner)",
        "spoof": "“Olga Owner” (member)",
        "spoofOnly": "“owner” (member)",
        "tagged": "“Ana” (member, #3f9a)",
        "badTag": "“Ana” (member)",
        # Only my own rows say You; a member named "You" stays quoted.
        "you": "“You” (member)",
        "remove": "Remove Folder “notes” as vfs:@ana-notes, shared by “Ana” (member, #3f9a), from this agent?",
    }


def test_publisher_names_are_cleaned_like_fd_and_never_carry_the_role():
    # Display names are free text: none can come out looking like FD's role tag.
    names = {
        "nested": "Olga (ow(owner)ner)",
        "zeroWidth": "Olga (own\u200ber)",
        "fullWidth": "Olga\uff08owner\uff09",
        "cyrillic": "Olga (\u043ewner)",
        "brackets": "[Olga] <owner> {x}",
        "controls": "Ol\u0000ga\n\tAna\u202e",
        "dashes": "Ana <!-- x --> Kova\u010d",
        "keeps": "Ana-Marie O'Neil Jr.",
        "you": "You",
    }
    out = _utils(
        "const names = " + json.dumps(names) + ";"
        "const res = {};"
        "for (const [k, v] of Object.entries(names)) res[k] = {"
        "  clean: cleanPublisherName(v),"
        "  shown: publisherName({ mine: false, owner_name: v }),"
        "  label: packOwnerLabel({ mine: false, owner_name: v, role: 'member' }) };"
        "res.mine = publisherName({ mine: true, owner_name: 'Olga' });"
        "res.badges = [packRoleBadge('owner'), packRoleBadge('member'), packRoleBadge('(owner)'), packRoleBadge(undefined)];"
        "res.tags = [packOwnerTag({ owner_tag: '3f9a' }), packOwnerTag({ owner_tag: '#3F9A' }),"
        "  packOwnerTag({}), packOwnerTag({ owner_tag: 'owner' })];"
        "out = res;"
    )
    assert out["nested"]["clean"] == "Olga ow owner ner"
    assert out["zeroWidth"]["clean"] == "Olga owner"
    assert out["fullWidth"]["clean"] == "Olga owner"
    assert out["cyrillic"]["clean"] == "Olga \u043ewner"
    assert out["brackets"]["clean"] == "Olga owner x"
    assert out["controls"]["clean"] == "Olga Ana"
    assert out["dashes"]["clean"] == "Ana x Kova\u010d"
    assert out["keeps"]["clean"] == "Ana-Marie O'Neil Jr."
    for k, v in names.items():
        label = out[k]["label"]
        assert label.endswith(" (member)"), (k, label)
        # The only "(owner)" / "(member)" is FD's, at the very end.
        assert "(owner)" not in label, (k, label)
        assert label.count("(") == 1 and label.count(")") == 1, (k, label)
        assert out[k]["shown"] == "“" + out[k]["clean"] + "”"
    assert out["you"]["shown"] == "“You”" and out["mine"] == "You"
    assert out["badges"] == ["owner", "member", "member", "member"]
    assert out["tags"] == ["#3f9a", "#3F9A", "", ""]


def test_profile_shared_note_names_the_agents():
    out = _utils("""
      const row = (o) => ({ kind: 'profile', active: true, agent_name: 'Helper', ...o });
      out = {
        none: profileSharedNote(profilePackAgents([])),
        folderOnly: profileSharedNote(profilePackAgents([row({ kind: 'vfs' })])),
        one: profileSharedNote(profilePackAgents([row({}), row({}), row({ kind: 'deep_memory', agent_name: 'X' })])),
        two: profileSharedNote(profilePackAgents([row({}), row({ agent_name: 'Scout' })])),
        three: profileSharedNote(['Helper', 'Scout', 'Ops']),
        inactive: profilePackAgents([row({ active: false }), row({ agent_name: 'Scout' })]),
        blankName: profilePackAgents([row({ agent_name: '' })]),
      };
    """)
    assert out["none"] == "" and out["folderOnly"] == ""
    assert out["one"] == (
        "About me and My company are also shared with everyone who uses Helper — including "
        "their channels and automations. Saving updates what they see."
    )
    assert "everyone who uses Helper and Scout — including" in out["two"]
    assert "everyone who uses Helper, Scout and Ops — including" in out["three"]
    assert out["inactive"] == ["Scout"]
    assert out["blankName"] == ["an agent"]


def test_toggle_tag_adds_removes_and_caps():
    out = _utils("""
      const ten = Array.from({ length: 10 }, (_, i) => 't' + i);
      const two = ['a', 'b'];
      out = {
        add: toggleTag(['a'], 'b'),
        remove: toggleTag(['a', 'b'], 'a'),
        eleventh: toggleTag(ten, 'x'),
        eleventhSame: toggleTag(ten, 'x') === ten,
        removeAtCap: toggleTag(ten, 't3').length,
        smallCap: toggleTag(two, 'c', 2),
        untouched: two,
      };
    """)
    assert out["add"] == ["a", "b"]
    assert out["remove"] == ["b"]
    assert out["eleventh"] == [f"t{i}" for i in range(10)]
    assert out["eleventhSame"] is True
    assert out["removeAtCap"] == 9
    assert out["smallCap"] == ["a", "b"]
    assert out["untouched"] == ["a", "b"]


def test_folder_names_and_capability_note():
    out = _utils("""
      out = {
        emptySuffix: fullAlias('ana', ''),
        blankSuffix: fullAlias('ana', '   '),
        trimmed: fullAlias('ana', ' notes '),
        errEmpty: aliasSuffixError('ana', ''),
        errOk: aliasSuffixError('ana', 'notes'),
        errDash: aliasSuffixError('ana', 'q3-plans-2'),
        errUpper: aliasSuffixError('ana', 'Notes'),
        errUnderscore: aliasSuffixError('ana', 'a_b'),
        errLong: aliasSuffixError('ana', 'x'.repeat(40)),
        errFits: aliasSuffixError('ana', 'x'.repeat(36)),
        errTooLong: aliasSuffixError('ana', 'x'.repeat(37)),
        capTrue: capabilityNote(true),
        capFalse: capabilityNote(false),
        capNull: capabilityNote(null),
        capUndef: capabilityNote(undefined),
        capOwner: capabilityNote(false, 'owner', true, 'Olga'),
        capMember: capabilityNote(false, 'member', true, 'Olga'),
        capMemberDollar: capabilityNote(false, 'member', true, '$& Co'),
        capUnchecked: capabilityNote(null, 'member', true, 'Olga'),
        capUncheckedOwner: capabilityNote(null, 'owner', true),
        capStopped: capabilityNote(null, 'owner', false),
        capOldDeck: capabilityNote(null, 'member', undefined),
        capTrueRunning: capabilityNote(true, 'owner', true),
        warns: [capabilityWarns(false, false), capabilityWarns(null, true), capabilityWarns(null, false),
                capabilityWarns(undefined, undefined), capabilityWarns(true, true)],
        derived: [derivedAlias('ana', 'Q3 Plans'), derivedAlias('ana', 'Kovač / notes!'),
                  derivedAlias('ana', '!!!'), derivedAlias('ana', 'x'.repeat(50))],
      };
    """)
    assert out["emptySuffix"] == "" and out["blankSuffix"] == ""
    assert out["trimmed"] == "ana-notes"
    assert out["errEmpty"] == "" and out["errOk"] == "" and out["errDash"] == ""
    for key in ("errUpper", "errUnderscore", "errLong", "errTooLong"):
        assert out[key] == TEXTS["ALIAS_ERROR"], key
    # "ana-" + 36 = 40 characters: the longest allowed name.
    assert out["errFits"] == ""
    assert out["capTrue"] == ""
    assert out["capFalse"] == TEXTS["AGENT_OUTDATED_NOTE"]
    assert out["capNull"] == TEXTS["AGENT_NOT_RUNNING_NOTE"]
    assert out["capUndef"] == TEXTS["AGENT_NOT_RUNNING_NOTE"]
    # The owner restarts it themselves; a member asks the owner by name.
    assert out["capOwner"] == TEXTS["AGENT_OUTDATED_OWNER_NOTE"]
    assert "Ask its owner" not in out["capOwner"]
    assert out["capMember"] == TEXTS["AGENT_OUTDATED_NOTE"].replace("Ask its owner", "Ask Olga")
    assert out["capMemberDollar"] == TEXTS["AGENT_OUTDATED_NOTE"].replace("Ask its owner", "Ask $& Co")
    # Running, but the version check failed: not "isn't running".
    assert out["capUnchecked"] == TEXTS["AGENT_UNCHECKED_NOTE"]
    assert out["capUncheckedOwner"] == TEXTS["AGENT_UNCHECKED_NOTE"]
    assert out["capStopped"] == TEXTS["AGENT_NOT_RUNNING_NOTE"]
    assert out["capOldDeck"] == TEXTS["AGENT_NOT_RUNNING_NOTE"]
    assert out["capTrueRunning"] == ""
    assert out["warns"] == [True, True, False, False, False]
    # FD's derive_alias, mirrored for the confirm.
    assert out["derived"] == ["ana-q3-plans", "ana-kovac-notes", "ana-folder", "ana-" + "x" * 36]


# ── the dialog, rendered against a tiny hooks runtime ───────────────

_REF = "process:helper:0123456789abcdef"

# A minimal React: hooks keyed by call order, effects run after a render when
# their deps change, `settle()` re-renders while state keeps changing. Function
# components met inside the tree (the Switch) are expanded in place; icons are
# strings. `find`/`findAll`/`text` walk the produced tree.
_RUNTIME = r"""
const React = {
  Fragment: 'Fragment',
  createElement: (type, props, ...children) => {
    const kids = children.flat(Infinity);
    if (typeof type === 'function') return type({ ...(props || {}), children: kids });
    return { type, props: props || {}, children: kids };
  },
};
const X = 'X', Layers = 'Layers', Loader2 = 'Loader2', Trash2 = 'Trash2';
let slots = [], idx = 0, pending = [], dirty = false, tree = null, renderFn = null;
const same = (a, b) => !!a && !!b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
function useState(init) {
  const i = idx++;
  if (!(i in slots)) slots[i] = typeof init === 'function' ? init() : init;
  return [slots[i], (v) => { slots[i] = typeof v === 'function' ? v(slots[i]) : v; dirty = true; }];
}
function useCallback(fn, deps) {
  const i = idx++;
  if (slots[i] && same(slots[i].deps, deps)) return slots[i].fn;
  slots[i] = { fn, deps };
  return fn;
}
function useEffect(fn, deps) {
  const i = idx++;
  const prev = slots[i];
  if (!prev || !same(prev.deps, deps)) {
    const slot = { deps, cleanup: null };
    slots[i] = slot;
    pending.push(() => {
      if (prev && typeof prev.cleanup === 'function') prev.cleanup();
      slot.cleanup = fn();
    });
  }
}
function render() {
  idx = 0; dirty = false; pending = [];
  tree = renderFn();
  const fx = pending; pending = [];
  for (const f of fx) f();
}
async function settle() {
  for (let k = 0; k < 200; k++) { await null; if (dirty) render(); }
}
function mount(fn) {
  for (const s of slots) if (s && typeof s.cleanup === 'function') s.cleanup();
  slots = []; renderFn = fn; render(); return settle();
}
const createPortal = (el, where) => (where === document.body ? el : null);
const keyListeners = [];
const document = { body: { tag: 'body' },
  addEventListener: (t, h) => { if (t === 'keydown') keyListeners.push(h); },
  removeEventListener: (t, h) => { const i = keyListeners.indexOf(h); if (i >= 0) keyListeners.splice(i, 1); } };
const confirms = [];
let answer = true;
const window = { confirm: (m) => { confirms.push(m); return answer; } };
const isNode = (n) => n && typeof n === 'object' && 'type' in n;
const findAll = (n, pred, acc = []) => {
  if (!isNode(n)) return acc;
  if (pred(n)) acc.push(n);
  for (const c of n.children || []) findAll(c, pred, acc);
  return acc;
};
const find = (n, pred) => findAll(n, pred)[0] || null;
const text = (n) => (n == null || n === false || n === true) ? ''
  : isNode(n) ? (n.children || []).map(text).join('') : String(n);
const byText = (type, t) => find(tree, (n) => n.type === type && text(n).includes(t));
const switchNamed = (label) => find(tree, (n) => n.props.role === 'switch' && n.props['aria-label'] === label);
"""

_MODAL_HARNESS = _RUNTIME + r"""
const REF = 'process:helper:0123456789abcdef';
let hostWarning = '';
const useSharedAgentStore = (sel) => sel({ hostWarning });
let server = null;
const calls = [];
let failCreate = '', failGet = '';
let gate = null;
const row = (over) => ({ id: 'p' + Math.random().toString(16).slice(2, 10), kind: 'profile',
  owner_id: '', owner_name: 'Ana', role: 'member', mine: true, project: '', alias: '', tags: [],
  created_at: '', can_remove: true, active: true, ...over });
const base = (over) => ({
  agent_ref: REF, agent_name: 'Helper', runtime: 'process', role: 'member', owner_name: 'Olga',
  kinds: ['profile', 'vfs', 'deep_memory'], packs: [], mine: [],
  eligible_projects: ['notes', 'plans'], agent_supports_packs: true, alias_prefix: 'ana',
  deep_memory_tags: [{ tag: 'finance', count: 12 }, { tag: 'domain:legal', count: 3 },
                     { tag: 'misc', count: 1 }],
  limits: { max_packs: 32, max_vfs_per_owner: 5, max_tags: 10 }, ...over });
const getAgentPacks = async (ref) => {
  calls.push(['get', ref]);
  if (failGet) throw new Error(failGet);
  return JSON.parse(JSON.stringify(server));
};
const createPack = async (body) => {
  calls.push(['create', body]);
  if (gate) await gate;
  if (failCreate) throw new Error(failCreate);
  const p = row({ kind: body.kind, project: body.project || '',
    alias: body.kind === 'vfs' ? (body.alias || 'ana-' + body.project) : '', tags: body.tags || [] });
  server = { ...server, mine: [...server.mine, p], packs: [...server.packs, p] };
  return { pack: p };
};
const deletePack = async (id) => {
  calls.push(['delete', id]);
  server = { ...server, mine: server.mine.filter((p) => p.id !== id),
             packs: server.packs.filter((p) => p.id !== id) };
  return { ok: true };
};
let closed = 0;
const open = (data, name = 'Helper') => {
  server = data;
  const props = { agentRef: REF, agentName: name, onClose: () => { closed++; } };
  return mount(() => ContextPacksModal(props));
};
const click = async (n) => { n.props.onClick({ stopPropagation() {} }); await settle(); };
const typeInto = async (n, value) => { n.props.onChange({ target: { value } }); await settle(); };
const creates = () => calls.filter((c) => c[0] === 'create').map((c) => c[1]);
const deletes = () => calls.filter((c) => c[0] === 'delete').map((c) => c[1]);
const gets = () => calls.filter((c) => c[0] === 'get').length;
"""

_MODAL_DECLS = _ALL_UTILS + [
    (_MODAL, ["sectionLabel", "Publisher", "errorText", "Switch", "ContextPacksModal"]),
]


def _modal(body: str):
    return _lift(_MODAL_DECLS, _MODAL_HARNESS + "done = (async () => {" + body + "})();", tsx=True)


def test_modal_shows_the_disclosure_and_who_shares_what():
    out = _modal("""
      hostWarning = 'HOST WARNING';
      await open(base({ packs: [
        row({ id: 'a1', mine: false, owner_name: 'Olga', role: 'owner', can_remove: false }),
        row({ id: 'a2', kind: 'vfs', project: 'notes', alias: 'ana-notes', mine: true }),
      ] }));
      const all = text(tree);
      out = {
        all, gets: gets(),
        title: text(find(tree, (n) => n.type === 'div' && n.props.className === 'text-sm font-semibold text-zinc-100')),
        removers: findAll(tree, (n) => n.type === 'button'
          && ['Stop sharing', 'Remove from this agent'].includes(n.props.title)).length,
      };
    """)
    all_ = out["all"]
    assert out["gets"] == 1
    assert out["title"] == TEXTS["PACKS_TITLE"]
    assert ("What you share here is available to everyone who uses Helper: Olga," in all_)
    assert CHANNELS_NOTE in all_
    assert TEXTS["ACTIVE_PACKS_HEADING"] in all_
    # Name, then FD's role badge (on every row), then what is shared.
    assert "“Olga” owner · Profile (about me, company)" in all_
    assert "You member · Folder “notes” as vfs:@ana-notes" in all_
    # Only my own row can be removed here (Olga's isn't removable by a member).
    assert out["removers"] == 1
    assert all_.index(TEXTS["REVOKE_NOTE"]) < all_.index("HOST WARNING")
    # Agent understands packs: no capability note.
    assert TEXTS["AGENT_OUTDATED_NOTE"] not in all_
    assert TEXTS["AGENT_NOT_RUNNING_NOTE"] not in all_


def test_modal_rows_badge_the_role_and_never_trust_the_name():
    out = _modal("""
      await open(base({ role: 'owner', owner_name: 'Olga', packs: [
        row({ id: 'o', mine: true, owner_name: 'Olga', role: 'owner' }),
        row({ id: 's1', mine: false, owner_name: 'Olga (ow(owner)ner)', role: 'member' }),
        row({ id: 's2', mine: false, owner_name: 'Olga\uff08own\u200ber\uff09', role: 'member' }),
        row({ id: 'y', mine: false, owner_name: 'You', role: 'member' }),
        row({ id: 'a1', mine: false, owner_name: 'Ana', role: 'member', owner_tag: '3f9a' }),
        row({ id: 'a2', mine: false, owner_name: 'Ana', role: 'member', owner_tag: 'b71c' }),
      ] }));
      // The "Shared on this agent" rows, each: label span → name, badge, (tag), " · ".
      const rows = findAll(tree, (n) => n.type === 'div'
          && String(n.props.className).includes('px-2.5 py-1.5')).map((r) => {
        const label = r.children.find((c) => isNode(c) && c.type === 'span');
        const spans = findAll(label, (n) => n.type === 'span').slice(1);
        return { text: text(label), name: text(spans[0]), badge: text(spans[1]),
                 tag: String(spans[2].props.className).includes('font-mono') ? text(spans[2]) : '' };
      });
      out = rows;
    """)
    assert [r["name"] for r in out] == [
        "You", "“Olga ow owner ner”", "“Olga owner”", "“You”", "“Ana”", "“Ana”",
    ]
    assert [r["badge"] for r in out] == ["owner", "member", "member", "member", "member", "member"]
    # FD's collision tag tells the two Anas apart.
    assert [r["tag"] for r in out] == ["", "", "", "", "#3f9a", "#b71c"]
    # No member's row reads like the owner's.
    for r in out[1:]:
        assert "(owner)" not in r["text"] and "owner ·" not in r["text"], r


def test_modal_capability_notes():
    out = _modal("""
      const res = {};
      for (const [k, v, over] of [['old', false, {}], ['down', null, {}],
                                  ['ownerOld', false, { role: 'owner' }],
                                  ['unchecked', null, { agent_running: true }]]) {
        await open(base({ agent_supports_packs: v, ...over }));
        const amber = find(tree, (n) => n.type === 'div' && String(n.props.className || '').includes('bg-amber-500/10'));
        res[k] = { all: text(tree), amber: amber ? text(amber) : null,
                   canShare: !switchNamed('My profile (about me, company)').props.disabled };
      }
      out = res;
    """)
    # A member asks the owner by name; the owner restarts it themselves.
    assert out["old"]["amber"] == TEXTS["AGENT_OUTDATED_NOTE"].replace("Ask its owner", "Ask Olga")
    assert out["ownerOld"]["amber"] == TEXTS["AGENT_OUTDATED_OWNER_NOTE"]
    assert out["down"]["amber"] is None
    assert TEXTS["AGENT_NOT_RUNNING_NOTE"] in out["down"]["all"]
    # Running but unchecked: a warning, and never "isn't running".
    assert out["unchecked"]["amber"] == TEXTS["AGENT_UNCHECKED_NOTE"]
    assert TEXTS["AGENT_NOT_RUNNING_NOTE"] not in out["unchecked"]["all"]
    # Publishing stays possible either way (takes effect after the restart).
    for k in ("old", "down", "ownerOld", "unchecked"):
        assert out[k]["canShare"] is True, k


def test_modal_profile_switch_asks_before_publishing_only():
    out = _modal("""
      await open(base({}));
      let sw = switchNamed('My profile (about me, company)');
      const off = sw.props['aria-checked'];
      answer = false;
      await click(sw);
      const cancelled = { creates: creates().length, confirms: confirms.length };
      answer = true;
      await click(switchNamed('My profile (about me, company)'));
      sw = switchNamed('My profile (about me, company)');
      const on = sw.props['aria-checked'];
      confirms.length = 0;
      await click(sw);
      out = { off, on, cancelled, creates: creates(), deletes: deletes().length,
              offConfirms: confirms.length, gets: gets(),
              after: switchNamed('My profile (about me, company)').props['aria-checked'] };
    """)
    assert out["off"] is False and out["on"] is True and out["after"] is False
    assert out["cancelled"] == {"creates": 0, "confirms": 1}
    assert out["creates"] == [{"agent_ref": _REF, "kind": "profile"}]
    assert out["deletes"] == 1
    assert out["offConfirms"] == 0
    # Mount + a reload after each create/delete.
    assert out["gets"] == 3


def test_modal_profile_confirm_text():
    out = _modal("""
      answer = false;
      await open(base({ agent_name: '' }), 'Fallback name');
      await click(switchNamed('My profile (about me, company)'));
      out = confirms;
    """)
    assert out == [
        "Share your About me and Company with everyone who uses Fallback name?" + _CONFIRM_TAIL
    ]


def test_modal_shares_a_folder_under_the_publishers_prefix():
    out = _modal("""
      await open(base({}));
      const input = find(tree, (n) => n.type === 'input' && n.props.placeholder === 'name (optional)');
      const prefix = text(find(tree, (n) => n.type === 'span' && text(n).startsWith('vfs:@')));
      await typeInto(input, 'Notes');
      const bad = { err: text(tree).includes(TEXTS_ALIAS_ERROR), disabled: byText('button', 'Share').props.disabled };
      await typeInto(find(tree, (n) => n.type === 'input' && n.props.placeholder === 'name (optional)'), 'my-notes');
      const sel = find(tree, (n) => n.type === 'select');
      await typeInto(sel, 'plans');
      await click(byText('button', 'Share'));
      const first = creates()[0];
      const afterFirst = { options: findAll(find(tree, (n) => n.type === 'select'), (n) => n.type === 'option').map(text),
                           suffix: find(tree, (n) => n.type === 'input' && n.props.placeholder === 'name (optional)').props.value };
      // Empty suffix → '' (Flight Deck derives the name).
      await click(byText('button', 'Share'));
      out = { prefix, bad, first, second: creates()[1], afterFirst, confirms,
              all: text(tree) };
    """.replace("TEXTS_ALIAS_ERROR", json.dumps(TEXTS["ALIAS_ERROR"])))
    assert out["prefix"] == "vfs:@ana-"
    assert out["bad"] == {"err": True, "disabled": True}
    assert out["first"] == {"agent_ref": _REF, "kind": "vfs", "project": "plans", "alias": "ana-my-notes"}
    assert out["afterFirst"] == {"options": ["notes"], "suffix": ""}
    assert out["second"] == {"agent_ref": _REF, "kind": "vfs", "project": "notes", "alias": ""}
    # Each confirm names the folder picked and its vfs:@ name (FD derives one
    # when none is typed), so the default first folder can't go out unnoticed.
    assert out["confirms"] == [
        "Share your folder “plans” (as vfs:@ana-my-notes), read-only, with everyone who uses "
        "Helper?" + _CONFIRM_TAIL,
        "Share your folder “notes” (as vfs:@ana-notes, or the next free name), read-only, with "
        "everyone who uses Helper?" + _CONFIRM_TAIL,
    ]
    # Both shared: each shows its summary and a "Stop sharing" button; nothing left to add.
    assert "Folder “plans” as vfs:@ana-my-notes" in out["all"]
    assert "No folder of yours can be shared." not in out["all"]


def test_modal_folder_limits_and_no_eligible_folder():
    out = _modal("""
      await open(base({ eligible_projects: [] }));
      const none = text(tree);
      const full = [1, 2].map((i) => row({ id: 'v' + i, kind: 'vfs', project: 'f' + i, alias: 'ana-f' + i }));
      await open(base({ eligible_projects: ['f1', 'f2', 'f3'], mine: full,
                        limits: { max_packs: 32, max_vfs_per_owner: 2, max_tags: 10 } }));
      const capped = { select: !!find(tree, (n) => n.type === 'select'), all: text(tree) };
      const inactive = row({ id: 'v9', kind: 'vfs', project: 'gone', alias: 'ana-gone', active: false });
      await open(base({ mine: [inactive] }));
      out = { none, capped, inactive: text(tree) };
    """)
    assert "No folder of yours can be shared." in out["none"]
    assert out["capped"]["select"] is False
    assert "No folder of yours can be shared." not in out["capped"]["all"]
    assert TEXTS["INACTIVE_HINT"] in out["inactive"]


def test_modal_deep_memory_offers_only_existing_tags():
    out = _modal("""
      await open(base({ limits: { max_packs: 32, max_vfs_per_owner: 5, max_tags: 2 } }));
      const chips = () => findAll(tree, (n) => n.type === 'label').map((l) => ({ text: text(l),
        box: find(l, (n) => n.type === 'input') }));
      const labels = chips().map((c) => c.text);
      const freeText = findAll(tree, (n) => n.type === 'input' && n.props.type !== 'checkbox')
        .map((n) => n.props.placeholder);
      for (const t of ['finance', 'domain:legal']) {
        await (async () => { chips().find((c) => c.text.startsWith(t + ' ')).box.props.onChange(); await settle(); })();
      }
      const third = chips().find((c) => c.text.startsWith('misc ')).box.props.disabled;
      answer = false;
      await click(switchNamed('My deep memory'));
      answer = true;
      await click(switchNamed('My deep memory'));
      const on = { checked: switchNamed('My deep memory').props['aria-checked'], all: text(tree),
                   chips: chips().length };
      await click(byText('button', 'Stop sharing'));
      out = { labels, freeText, third, confirms, creates: creates(), on, deletes: deletes().length,
              offAfter: switchNamed('My deep memory').props['aria-checked'] };
    """)
    assert out["labels"] == ["finance (12)", "domain:legal (3)", "misc (1)"]
    # The only free-text input is the folder name — no typed tags.
    assert out["freeText"] == ["name (optional)"]
    assert out["third"] is True
    assert out["confirms"] == [
        "Let everyone who uses Helper search your deep memory (only entries tagged finance, "
        "domain:legal)?" + _CONFIRM_TAIL
    ] * 2
    assert out["creates"] == [
        {"agent_ref": _REF, "kind": "deep_memory", "tags": ["finance", "domain:legal"]}
    ]
    assert out["on"]["checked"] is True
    assert "Deep memory (tags: finance, domain:legal)" in out["on"]["all"]
    assert out["on"]["chips"] == 0
    assert out["deletes"] == 1
    assert out["offAfter"] is False


def test_modal_deep_memory_without_tags_and_docker():
    out = _modal("""
      await open(base({ deep_memory_tags: [] }));
      const noTags = text(tree);
      await click(switchNamed('My deep memory'));
      const sent = creates()[0];
      await open(base({ runtime: 'docker', kinds: ['profile'], eligible_projects: [],
                        deep_memory_tags: [] }));
      out = { noTags, sent, docker: text(tree), switches: findAll(tree, (n) => n.props.role === 'switch')
        .map((n) => n.props['aria-label']), confirm: confirms[0] };
    """)
    assert TEXTS["NO_TAGS_NOTE"] in out["noTags"]
    assert TEXTS["TAGS_LABEL"] not in out["noTags"]
    assert out["sent"] == {"agent_ref": _REF, "kind": "deep_memory", "tags": []}
    assert out["confirm"] == (
        "Let everyone who uses Helper search ALL of your deep memory, including documents "
        "indexed from your files and Google Drive?" + _CONFIRM_TAIL
    )
    assert TEXTS["DOCKER_PACKS_NOTE"] in out["docker"]
    assert TEXTS["VFS_PACK_LABEL"] not in out["docker"]
    assert out["switches"] == [TEXTS["PROFILE_PACK_LABEL"]]


def test_modal_owner_removes_a_members_pack_after_a_confirm():
    out = _modal("""
      const theirs = row({ id: 'm1', mine: false, owner_name: 'Ana', role: 'member', can_remove: true,
                           kind: 'vfs', project: 'notes', alias: 'ana-notes' });
      const own = row({ id: 'o1', mine: true, owner_name: 'Olga', role: 'owner', can_remove: true });
      await open(base({ role: 'owner', packs: [theirs, own], mine: [own] }));
      const theirsBtn = findAll(tree, (n) => n.type === 'button' && n.props.title === 'Remove from this agent');
      answer = false;
      await click(theirsBtn[0]);
      const kept = deletes().length;
      answer = true;
      await click(findAll(tree, (n) => n.type === 'button' && n.props.title === 'Remove from this agent')[0]);
      const confirmsBefore = confirms.length;
      await click(findAll(tree, (n) => n.type === 'button' && n.props.title === 'Stop sharing')[0]);
      out = { theirs: theirsBtn.length, kept, deletes: deletes(), confirms, confirmsBefore,
              disclosure: text(tree) };
    """)
    assert out["theirs"] == 1
    assert out["kept"] == 0
    assert out["confirms"] == [
        "Remove Folder “notes” as vfs:@ana-notes, shared by “Ana” (member), from this agent?",
    ] * 2
    # My own row: no confirm.
    assert out["confirmsBefore"] == 2
    assert out["deletes"] == ["m1", "o1"]
    assert "available to everyone who uses Helper: you, everywhere" in out["disclosure"]
    # The owner is told members' packs run on their channels and automations too.
    assert _OWNER_MEMBERS.strip() in out["disclosure"]


def test_modal_keeps_its_data_and_shows_fds_reason_on_failure():
    out = _modal("""
      await open(base({}));
      failCreate = 'You already share that with this agent';
      await click(switchNamed('My profile (about me, company)'));
      const afterFail = { all: text(tree), disabled: switchNamed('My profile (about me, company)').props.disabled };
      failGet = 'Agent not found';
      failCreate = '';
      await click(switchNamed('My profile (about me, company)'));
      out = { afterFail, afterGetFail: text(tree) };
    """)
    assert "You already share that with this agent" in out["afterFail"]["all"]
    assert out["afterFail"]["disabled"] is False
    assert TEXTS["PROFILE_PACK_LABEL"] in out["afterFail"]["all"]
    # A failed reload keeps the last data and shows FD's reason.
    assert "Agent not found" in out["afterGetFail"]
    assert TEXTS["ACTIVE_PACKS_HEADING"] in out["afterGetFail"]


def test_modal_disables_controls_while_a_request_runs():
    out = _modal("""
      await open(base({}));
      let release;
      gate = new Promise((r) => { release = r; });
      switchNamed('My profile (about me, company)').props.onClick();
      await settle();
      const during = {
        profile: switchNamed('My profile (about me, company)').props.disabled,
        deep: switchNamed('My deep memory').props.disabled,
        share: byText('button', 'Share').props.disabled,
        spinners: findAll(tree, (n) => n.type === 'Loader2').length,
      };
      release();
      await settle();
      out = { during, after: {
        profile: switchNamed('My profile (about me, company)').props.disabled,
        deep: switchNamed('My deep memory').props.disabled,
        on: switchNamed('My profile (about me, company)').props['aria-checked'],
        spinners: findAll(tree, (n) => n.type === 'Loader2').length,
      } };
    """)
    assert out["during"] == {"profile": True, "deep": True, "share": True, "spinners": 1}
    assert out["after"] == {"profile": False, "deep": False, "on": True, "spinners": 0}


def test_modal_closes_on_escape_and_backdrop():
    out = _modal("""
      await open(base({}));
      for (const h of [...keyListeners]) h({ key: 'Enter' });
      const enter = closed;
      for (const h of [...keyListeners]) h({ key: 'Escape' });
      const esc = closed;
      tree.props.onClick();
      const inner = tree.children.find((c) => isNode(c));
      let stopped = false;
      inner.props.onClick({ stopPropagation: () => { stopped = true; } });
      out = { enter, esc, backdrop: closed, stopped };
    """)
    assert out == {"enter": 0, "esc": 1, "backdrop": 2, "stopped": True}


# ── the Profile card ────────────────────────────────────────────────

_MY_HARNESS = _RUNTIME + r"""
let mine = [];
const calls = [];
let missing = false;
const getMyPacks = async () => {
  calls.push('get');
  if (missing) throw new Error('Not Found');
  return { packs: JSON.parse(JSON.stringify(mine)) };
};
const deletePack = async (id) => { calls.push('delete:' + id); mine = mine.filter((p) => p.id !== id); return { ok: true }; };
"""


def test_profile_card_lists_marks_and_stops_my_packs():
    out = _lift(_ALL_UTILS + [(_MY_PACKS, ["MyContextPacks"])], _MY_HARNESS + r"""
      done = (async () => {
        await mount(() => MyContextPacks());
        const empty = text(tree);
        mine = [
          { id: 'a', kind: 'profile', agent_ref: 'r1', agent_name: 'Helper', active: true, tags: [] },
          { id: 'b', kind: 'vfs', project: 'notes', alias: 'ana-notes', agent_ref: 'r2',
            agent_name: 'Scout', active: false, tags: [] },
        ];
        await mount(() => MyContextPacks());
        const listed = text(tree);
        const rows = findAll(tree, (n) => n.type === 'button' && text(n).includes('Stop sharing'));
        await (async () => { rows[1].props.onClick(); await settle(); })();
        out = { empty, listed, calls, after: text(tree) };
      })();
    """, tsx=True)
    assert TEXTS["MY_PACKS_HEADING"] in out["empty"]
    assert TEXTS["MY_PACKS_EMPTY"] in out["empty"]
    listed = out["listed"]
    assert "Profile (about me, company) · Helper" in listed
    assert "Folder “notes” as vfs:@ana-notes · Scout" in listed
    # Only the inactive row carries the hint.
    assert listed.count(TEXTS["INACTIVE_HINT"]) == 1
    assert listed.index(TEXTS["INACTIVE_HINT"]) > listed.index("Scout")
    assert out["calls"] == ["get", "get", "delete:b", "get"]
    assert "Scout" not in out["after"] and TEXTS["INACTIVE_HINT"] not in out["after"]
    # Sharing on: no paused marks.
    assert TEXTS["PACKS_PAUSED_NOTE"] not in listed and "paused" not in listed


def test_profile_card_with_sharing_off_lists_paused_packs_and_still_stops_them():
    out = _lift(_ALL_UTILS + [(_MY_PACKS, ["MyContextPacks"])], _MY_HARNESS + r"""
      done = (async () => {
        const heard = [];
        const props = { sharingOff: true, showEmpty: false, onPacks: (l) => heard.push(l.map((p) => p.id)) };
        // Nothing shared: no card at all.
        await mount(() => MyContextPacks(props));
        const empty = tree;
        // A deck without the route: no card either.
        missing = true;
        await mount(() => MyContextPacks(props));
        const noRoute = tree;
        missing = false;
        // FD keeps the packs with sharing off; none is active.
        mine = [
          { id: 'a', kind: 'profile', agent_ref: 'r1', agent_name: 'Helper', active: false, tags: [] },
          { id: 'b', kind: 'vfs', project: 'notes', alias: 'ana-notes', agent_ref: 'r2',
            agent_name: 'Scout', active: false, tags: [] },
        ];
        await mount(() => MyContextPacks(props));
        const listed = text(tree);
        const id = tree.props.id;
        const chips = findAll(tree, (n) => n.type === 'span' && text(n) === 'paused').length;
        const stops = findAll(tree, (n) => n.type === 'button' && text(n).includes('Stop sharing'));
        await (async () => { stops[0].props.onClick(); await settle(); })();
        const after = text(tree);
        mine = [];
        await (async () => { findAll(tree, (n) => n.type === 'button')[0].props.onClick(); await settle(); })();
        out = { empty, noRoute, listed, id, chips, after, gone: tree, calls, heard };
      })();
    """, tsx=True)
    assert out["empty"] is None and out["noRoute"] is None
    listed = out["listed"]
    assert TEXTS["MY_PACKS_HEADING"] in listed
    assert listed.count(TEXTS["PACKS_PAUSED_NOTE"]) == 1
    assert out["chips"] == 2
    # Paused, not inactive: the inactive reasons would be wrong with sharing off.
    assert TEXTS["INACTIVE_HINT"] not in listed
    assert "Profile (about me, company) · Helper" in listed
    assert out["id"] == TEXTS["MY_PACKS_ID"]
    assert "Helper" not in out["after"] and "Scout" in out["after"]
    # The last one stopped: the card goes away.
    assert out["gone"] is None
    assert out["calls"] == ["get", "get", "get", "delete:a", "get", "delete:b", "get"]
    # The page hears every list loaded (its About me note).
    assert out["heard"] == [[], ["a", "b"], ["b"], []]


# ── wiring ──────────────────────────────────────────────────────────


def test_service_calls_only_the_pack_routes_and_has_no_host_path():
    found = _query(_SERVICE, r"""
      const paths = all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'fdFetch')
        .map((c) => {
          const a = c.arguments[0];
          if (ts.isStringLiteral(a)) return a.text;
          if (ts.isTemplateExpression(a)) return a.head.text + a.templateSpans.map((s) => '${…}' + s.literal.text).join('');
          return '?';
        })
        .map((p) => p.split('?')[0]);
      const props = all.filter((n) => ts.isInterfaceDeclaration(n))
        .flatMap((i) => i.members.map((m) => m.name ? m.name.getText(sf) : ''));
      const exported = all.filter((n) => (ts.isVariableStatement(n) || ts.isFunctionDeclaration(n))
          && (n.modifiers || []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword))
        .map((n) => ts.isVariableStatement(n) ? n.declarationList.declarations[0].name.getText(sf) : n.name.getText(sf));
      return { paths, props, exported };
    """)
    assert sorted(found["paths"]) == sorted(
        ["/context-packs", "/context-packs", "/context-packs/${…}", "/context-packs/mine"]
    )
    assert set(found["paths"]) == {"/context-packs", "/context-packs/mine", "/context-packs/${…}"}
    for banned in ("root", "path", "host", "port", "token", "resource_key"):
        assert banned not in found["props"], banned
    assert set(found["exported"]) == {"getAgentPacks", "createPack", "deletePack", "getMyPacks"}


def test_service_requests():
    out = _lift([(_SERVICE, ["getAgentPacks", "createPack", "deletePack", "getMyPacks"])], r"""
      const seen = [];
      const fdFetch = async (path, init) => { seen.push([path, init ? { method: init.method, body: init.body ? JSON.parse(init.body) : null } : null]); return {}; };
      done = (async () => {
        await getAgentPacks('process:a b:0123456789abcdef');
        await createPack({ agent_ref: 'r', kind: 'vfs', project: 'notes', alias: '' });
        await deletePack('0123456789abcdef0123456789abcdef');
        await getMyPacks();
        out = seen;
      })();
    """)
    assert out == [
        ["/context-packs?agent_ref=process%3Aa%20b%3A0123456789abcdef", None],
        ["/context-packs", {"method": "POST",
                            "body": {"agent_ref": "r", "kind": "vfs", "project": "notes", "alias": ""}}],
        ["/context-packs/0123456789abcdef0123456789abcdef", {"method": "DELETE", "body": None}],
        ["/context-packs/mine", None],
    ]


def test_store_offers_packs_only_on_an_explicit_true():
    out = _lift([(_SHARED_STORE, ["useSharedAgentStore"])], r"""
      function create(init) {
        let state;
        const set = (p) => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; };
        const get = () => state;
        state = init(set, get);
        return { getState: get };
      }
      let server = { enabled: true, host_warning: '', agents: [], context_packs: true };
      let fail = false;
      const getSharedAgents = async () => { if (fail) throw new Error('x'); return JSON.parse(JSON.stringify(server)); };
      const setSharedAgentGoogle = async () => ({});
      const leaveShare = async () => {};
      const sharedContainerId = (r) => 'shared:' + r;
      const clearSharedSlices = () => {};
      const useChatStore = { getState: () => ({ sessions: new Map(), disconnectChat: () => {} }) };
      const st = () => useSharedAgentStore.getState();
      done = (async () => {
        const res = { initial: st().contextPacks };
        await st().fetch(); res.on = st().contextPacks;
        fail = true; await st().fetch(); res.blip = st().contextPacks; fail = false;
        server = { ...server, context_packs: 'true' }; await st().fetch(); res.truthy = st().contextPacks;
        server = { ...server, context_packs: undefined }; await st().fetch(); res.olderDeck = st().contextPacks;
        server = { ...server, context_packs: true, enabled: false }; await st().fetch(); res.sharingOff = st().contextPacks;
        out = res;
      })();
    """)
    assert out == {
        "initial": False, "on": True, "blip": True, "truthy": False, "olderDeck": False,
        "sharingOff": False,
    }


def test_shared_agents_response_declares_context_packs():
    found = _query(_SHARED_SERVICE, r"""
      const i = all.find((n) => ts.isInterfaceDeclaration(n) && n.name.text === 'SharedAgentsResponse');
      const m = i.members.find((x) => x.name && x.name.getText(sf) === 'context_packs');
      return m ? { optional: !!m.questionToken, type: m.type.getText(sf) } : null;
    """)
    assert found == {"optional": True, "type": "boolean"}


_STORE_FLAG = r"""
const decl = (name) => all.find((n) => ts.isVariableDeclaration(n) && n.name.getText(sf) === name);
const init = (name) => (decl(name) && decl(name).initializer ? decl(name).initializer.getText(sf) : null);
"""


@pytest.mark.parametrize("src, runtime", [(_PROCESS_CARD, "process"), (_CONTAINER_CARD, "docker")],
                         ids=["process", "container"])
def test_owner_cards_offer_shared_context_behind_the_flag(src, runtime):
    found = _query(src, _JSX_HELPERS + _STORE_FLAG + r"""
      const items = all.filter((n) => ts.isObjectLiteralExpression(n)
        && n.properties.some((p) => p.name && p.name.getText(sf) === 'label'
          && p.initializer && p.initializer.getText(sf) === 'PACKS_MENU_LABEL'));
      const prop = (o, k) => { const p = o.properties.find((x) => x.name && x.name.getText(sf) === k); return p ? p.initializer.getText(sf) : null; };
      const labels = all.filter((n) => ts.isPropertyAssignment(n) && n.name.getText(sf) === 'label')
        .map((p) => p.initializer.getText(sf));
      const modals = named('ContextPacksModal');
      return {
        items: items.map((o) => ({ show: prop(o, 'show'), onClick: prop(o, 'onClick'), icon: prop(o, 'icon') })),
        afterShare: labels.indexOf('PACKS_MENU_LABEL') === labels.indexOf("'Share…'") + 1,
        packsEnabled: init('packsEnabled'), canPacks: init('canPacks'),
        modals: modals.map((n) => ({ guard: guard(n), ref: attr(n, 'agentRef'), name: attr(n, 'agentName') })),
        // The Share dialog's note tells the owner about members' packs when packs are on.
        shareNotes: named('ShareModal').map((n) => attr(n, 'note')),
        fragmentNextToShare: modals.length > 0 && (() => {
          const frag = decl('configModal');
          return !!frag && frag.getStart(sf) < modals[0].getStart(sf) && modals[0].getEnd() <= frag.getEnd()
            && frag.getText(sf).includes('<ShareModal');
        })(),
      };
    """)
    assert found["items"] == [{"show": "canPacks", "onClick": "onPacks", "icon": "Layers"}]
    assert found["afterShare"] is True
    assert found["packsEnabled"] == "useSharedAgentStore((s) => s.contextPacks)"
    assert found["canPacks"] == "canShare && packsEnabled"
    assert len(found["modals"]) == 1
    m = found["modals"][0]
    assert "showPacks" in m["guard"] and "canPacks" in m["guard"]
    assert m["ref"] in ("{proc.agent_ref!}", "{container.agent_ref!}")
    assert m["name"] == "{agentName}"
    assert found["fragmentNextToShare"] is True
    assert found["shareNotes"] == [f"{{ownerShareNote(packsEnabled, '{runtime}')}}"]


def test_member_card_button_and_modal_behind_the_flag():
    found = _query(_SHARED_CARD, _JSX_HELPERS + _STORE_FLAG + r"""
      const modals = named('ContextPacksModal');
      const buttons = all.filter((n) => ts.isJsxElement(n) && tagName(n) === 'button'
        && n.children.some((c) => ts.isJsxExpression(c) && c.expression && c.expression.getText(sf) === 'PACKS_BUTTON'));
      return {
        flag: init('contextPacks'),
        modals: modals.map((n) => ({ guard: guard(n), ref: attr(n, 'agentRef'), name: attr(n, 'agentName') })),
        buttons: buttons.map((n) => ({ guard: guard(n), icon: n.children.some((c) => tagName(c) === 'Layers') })),
      };
    """)
    assert found["flag"] == "useSharedAgentStore((s) => s.contextPacks)"
    assert len(found["modals"]) == 1
    assert "contextPacks" in found["modals"][0]["guard"]
    assert found["modals"][0]["ref"] == "{agent.agent_ref}"
    assert found["modals"][0]["name"] == "{name}"
    assert len(found["buttons"]) == 1
    assert found["buttons"][0]["icon"] is True
    assert found["buttons"][0]["guard"].startswith("contextPacks &&")


def test_member_chat_bar_and_modal_inside_the_shared_branch():
    found = _query(_CHAT_PANEL, _JSX_HELPERS + _STORE_FLAG + r"""
      const modals = named('ContextPacksModal');
      const toggles = named('SharedAgentGoogleToggle');
      const buttons = all.filter((n) => ts.isJsxElement(n) && tagName(n) === 'button'
        && n.children.some((c) => ts.isJsxExpression(c) && c.expression && c.expression.getText(sf) === 'PACKS_BUTTON'));
      // The bar: the JSX expression that holds the Google switch.
      let bar = null;
      for (let p = toggles[0].parent; p; p = p.parent) if (ts.isJsxExpression(p)) { bar = p.expression; break; }
      const barCond = bar ? bar.getText(sf).split('&& (\n')[0] : null;
      const inBar = (n) => !!bar && bar.getStart(sf) <= n.getStart(sf) && n.getEnd() <= bar.getEnd();
      // The effect that closes the dialog: its deps.
      const resets = all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === 'useEffect'
          && n.arguments[0].getText(sf).includes('setShowPacks(false)'))
        .map((c) => (c.arguments[1] ? c.arguments[1].getText(sf) : null));
      return {
        flag: init('packsEnabled'),
        live: init('sharedLive'),
        resets,
        barCond,
        modals: modals.map((n) => ({ guard: guard(n), ref: attr(n, 'agentRef'), name: attr(n, 'agentName'), inBar: inBar(n) })),
        buttons: buttons.map((n) => ({ guard: guard(n), inBar: inBar(n) })),
      };
    """)
    assert found["flag"] == "useSharedAgentStore((s) => s.contextPacks)"
    assert found["barCond"].strip() == (
        "shared && sharedRow && !session.closed && (caps.google || packsEnabled)"
    )
    assert len(found["buttons"]) == 1
    assert found["buttons"][0]["inBar"] is True
    assert found["buttons"][0]["guard"].startswith("packsEnabled &&")
    assert len(found["modals"]) == 1
    m = found["modals"][0]
    assert m["guard"].startswith("shared && packsEnabled && showPacks")
    assert m["ref"] == "{shared.agentRef}" and m["name"] == "{session.containerName}"
    # The dialog closes — and stays closed — when the chat switches agent, ends,
    # or the agent is no longer shared with me (left / revoked / sharing off).
    assert found["live"] == "!!sharedRow && !session?.closed"
    assert "sharedLive" in m["guard"]
    assert found["resets"] == ["[sharedRef, sharedLive]"]


def test_profile_page_card_last_and_the_about_me_note():
    found = _query(_PROFILE_PAGE, _JSX_HELPERS + _STORE_FLAG + r"""
      const cards = named('MyContextPacks');
      const text = sf.getFullText();
      const aboutMe = all.find((n) => tagName(n) === 'CappedTextarea' && attr(n, 'label') === '"About me"');
      const notes = all.filter((n) => ts.isJsxExpression(n) && n.expression
          && n.expression.getText(sf).startsWith('sharedNote &&'));
      const links = all.filter((n) => tagName(n) === 'button' && attr(n, 'onClick') === '{showMyPacks}');
      return {
        flag: init('contextPacks'),
        loaded: init('sharedLoaded'),
        note: init('sharedNote'),
        scroll: init('showMyPacks'),
        cards: cards.map((n) => ({ guard: guard(n), sharingOff: attr(n, 'sharingOff'),
          showEmpty: attr(n, 'showEmpty'), onPacks: attr(n, 'onPacks'),
          afterPreview: n.getStart(sf) > text.indexOf('What your agents receive') })),
        notes: notes.map((n) => ({ beforeAboutMe: !!aboutMe && n.getEnd() <= aboutMe.getStart(sf),
          amber: n.getText(sf).includes('bg-amber-500/10') })),
        links: links.map((n) => ({ inNote: notes.some((x) => x.getStart(sf) <= n.getStart(sf) && n.getEnd() <= x.getEnd()),
          text: n.getText(sf).includes('MY_PACKS_HEADING') })),
      };
    """)
    assert found["flag"] == "useSharedAgentStore((s) => s.contextPacks)"
    assert found["loaded"] == "useSharedAgentStore((s) => s.loaded)"
    # The card is no longer hidden with sharing off: paused rows, still stoppable.
    assert len(found["cards"]) == 1
    card = found["cards"][0]
    assert card["guard"].startswith("data && caps && (")
    assert card["sharingOff"] == "{sharedLoaded && !contextPacks}"
    assert card["showEmpty"] == "{contextPacks}"
    assert card["onPacks"] == "{setMyPacks}"
    assert card["afterPreview"] is True
    # About me / My company: an amber note while my profile is shared on agents
    # (only with packs on), above the fields, linking to the card.
    assert found["note"] == "contextPacks ? profileSharedNote(profilePackAgents(myPacks)) : ''"
    assert found["notes"] == [{"beforeAboutMe": True, "amber": True}]
    assert found["links"] == [{"inNote": True, "text": True}]
    assert "getElementById(MY_PACKS_ID)" in found["scroll"]


def test_modal_wiring_uses_the_helpers():
    found = _query(_MODAL, _JSX_HELPERS + r"""
      const callee = (name) => all.filter((n) => ts.isCallExpression(n) && n.expression.getText(sf) === name).length;
      const inputs = named('input').map((n) => ({ type: attr(n, 'type'), value: attr(n, 'value') }));
      const textareas = named('textarea').length;
      return { cap: callee('capabilityNote'), alias: callee('fullAlias'), err: callee('aliasSuffixError'),
               toggle: callee('toggleTag'), disclosure: callee('packsDisclosure'),
               confirm: callee('window.confirm'), portal: callee('createPortal'), inputs, textareas };
    """)
    assert found["cap"] >= 1 and found["alias"] >= 1 and found["err"] >= 1
    assert found["toggle"] >= 1 and found["disclosure"] == 1
    assert found["portal"] == 1
    # Asks before: profile on, sharing a folder, deep memory on, removing someone else's pack.
    assert found["confirm"] == 4
    # One free-text input (the folder name suffix); tags are checkboxes only.
    free = [i for i in found["inputs"] if i["type"] != '"checkbox"']
    assert free == [{"type": None, "value": "{suffix}"}]
    assert found["textareas"] == 0
