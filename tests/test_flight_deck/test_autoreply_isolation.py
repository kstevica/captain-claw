"""Gmail topic auto-replies stay out of Autonomous Work (the PR #56 pin).

PR #56 made email proposal-only for the arbiter and its dispatch: Autonomous
Work never drafts or sends mail on its own. The topic auto-reply engine is a
separate, explicitly opted-in path, and these source pins keep the two apart:

* nothing in the autonomy stack (arbiter, dispatch, catalog, autonomy, its
  routes, consciousness, the event-source adapters) mentions the engine;
* the engine's modules never dispatch, propose or record autonomy actions,
  never grant ``mail_write: allow``, never import the autonomy stack, never
  write a Gmail draft, and reach an agent only through ``run_tool``;
* the dispatch marker and the agent's automation kinds are exactly what they
  were — no ``auto_reply`` kind was added.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from captain_claw import mail_authority
from captain_claw.flight_deck import fd_dispatch
from captain_claw.flight_deck import gmail_autoreply_writer as writer

FD = Path(__file__).resolve().parents[2] / "captain_claw" / "flight_deck"

AUTONOMY_SOURCES = sorted(
    [FD / name for name in ("arbiter.py", "fd_dispatch.py", "action_catalog.py", "autonomy.py",
                            "autonomy_routes.py", "consciousness.py")]
    + list(FD.glob("event_sources*.py")))
AUTOREPLY_SOURCES = sorted(FD.glob("gmail_autoreply*.py"))

# The autonomy stack the engine must never import.
FORBIDDEN_MODULES = {"arbiter", "fd_dispatch", "action_catalog", "autonomy", "autonomy_routes",
                     "consciousness", "event_sources", "event_sources_google"}


def _ids(paths):
    return [p.name for p in paths]


def test_the_sources_exist():
    names = {p.name for p in AUTOREPLY_SOURCES}
    assert {"gmail_autoreply.py", "gmail_autoreply_config.py", "gmail_autoreply_judges.py",
            "gmail_autoreply_writer.py"} <= names
    assert {p.name for p in AUTONOMY_SOURCES} >= {"arbiter.py", "event_sources.py",
                                                  "event_sources_google.py"}
    assert all(p.is_file() for p in AUTONOMY_SOURCES)


@pytest.mark.parametrize("path", AUTONOMY_SOURCES, ids=_ids(AUTONOMY_SOURCES))
def test_the_autonomy_stack_never_mentions_the_engine(path):
    assert "gmail_autoreply" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", AUTOREPLY_SOURCES, ids=_ids(AUTOREPLY_SOURCES))
def test_the_engine_never_acts_through_autonomy(path):
    text = path.read_text(encoding="utf-8")
    for needle in ("dispatch_action", "should_auto_dispatch", "add_action"):
        assert needle not in text, needle
    # No allow grant, in any quoting or spacing.
    assert not re.search(r"""["']mail_write["']\s*:\s*["']allow["']""", text)
    assert "mail_write=\"allow\"" not in text and "mail_write='allow'" not in text


@pytest.mark.parametrize("path", AUTOREPLY_SOURCES, ids=_ids(AUTOREPLY_SOURCES))
def test_the_engine_never_imports_the_autonomy_stack(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.rsplit(".", 1)[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.rsplit(".", 1)[-1])
            imported.update(a.name for a in node.names)
    assert not imported & FORBIDDEN_MODULES, imported & FORBIDDEN_MODULES


@pytest.mark.parametrize("path", AUTOREPLY_SOURCES, ids=_ids(AUTOREPLY_SOURCES))
def test_the_engine_only_ever_takes_mail_off_the_event_spine(path):
    """The spine is the arbiter's intake. The engine only claims an email it
    took on (``claim_handled``, so the arbiter never proposes a reply of its
    own next to the auto-reply) and gives one back that it did NOT handle
    (``release_claim``). It never ingests, surfaces or reads events."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    store_calls = {node.func.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Call)
                   and isinstance(node.func.value.func, ast.Attribute)
                   and node.func.value.func.attr == "get_store"}
    assert store_calls <= {"claim_handled", "release_claim"}, store_calls
    text = path.read_text(encoding="utf-8")
    for needle in ("add_event", "list_new", "ingest"):
        assert needle not in text, needle


def test_the_engine_claims_the_spine_before_the_reply_is_written():
    """Provisionally, once the matched email passed the thread check and
    before the writer runs — not only after the send minutes later (the
    arbiter would have surfaced the email by then). And every worked row is
    settled after (claim kept when handled, released when not)."""
    tree = ast.parse((FD / "gmail_autoreply.py").read_text(encoding="utf-8"))
    after_match = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef) and node.name == "_after_match")
    lines: dict[str, int] = {}
    for node in ast.walk(after_match):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            lines.setdefault(name, node.lineno)
    assert lines["_takeover"] < lines["_claim_spine"] < lines["compose_reply"]
    guarded = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "_guarded")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_settle_spine" for n in ast.walk(guarded))


@pytest.mark.parametrize("path", AUTOREPLY_SOURCES, ids=_ids(AUTOREPLY_SOURCES))
def test_the_engine_never_writes_a_gmail_draft(path):
    text = path.read_text(encoding="utf-8")
    assert "create_draft" not in text and "update_draft" not in text
    assert not re.search(r"""["']/drafts""", text), "no Gmail drafts endpoint"


@pytest.mark.parametrize("path", AUTOREPLY_SOURCES, ids=_ids(AUTOREPLY_SOURCES))
def test_the_engine_reaches_an_agent_only_through_run_tool(path):
    """Of flight_deck.actions only ``run_tool_on_agent`` (one read-only tool,
    deny marker) — never ``run_action`` / ``undo_action`` (autonomy's)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "actions"}
    assert used <= {"run_tool_on_agent"}, used


def test_the_dispatch_marker_is_unchanged():
    assert fd_dispatch._AUTONOMY_AUTOMATION == {
        "kind": "autonomy", "job_text": "", "mail_write": "deny"}


def test_the_automation_kinds_are_unchanged():
    assert mail_authority.AUTOMATION_KINDS == frozenset({
        "autonomy", "autonomy_tool", "plan", "fd_scheduler", "flow", "flow_tool", "cron",
        "peer", "peer_relay", "sister", "botport", "fd_worker", "being", "mcp_task", "bat",
        "link", "exam", "unknown",
    })
    assert not any("reply" in kind for kind in mail_authority.AUTOMATION_KINDS)
    assert set(mail_authority.KIND_LABELS) == set(mail_authority.AUTOMATION_KINDS)


def test_the_lookup_marker_is_the_agents_own_deny_default():
    assert writer.LOOKUP_MARKER == {"kind": "autonomy_tool", "job_text": "",
                                    "mail_write": "deny"}
    bound = mail_authority.from_wire(dict(writer.LOOKUP_MARKER), default_kind="autonomy_tool")
    assert (bound.kind, bound.mail_write) == ("autonomy_tool", "deny")
