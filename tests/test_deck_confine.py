"""Tenant confinement on a multi-user Flight Deck (deck_confine + FD's pinning).

Every process agent runs as the deck's OS user, so on a sign-in deck an owner
call's path arguments stay in the agent's own folder, its workspace and its
owner's VFS root; a non-admin's agent runs no code on the host and follows no
link to a folder outside its owner's VFS root; Flight Deck pins all of it after
the agent's own .env and env_vars. Unset (sign-in off) changes nothing.

Every test runs with HOME and FD_DATA_DIR in a tmp dir.
"""

from __future__ import annotations

import json
import re
import sqlite3
import types
from pathlib import Path

import pytest

from captain_claw import deck_confine
from captain_claw.config import get_config
from captain_claw.exceptions import ToolBlockedError
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

_ENV_CLEARED = (
    "CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "CLAW_VFS_PROJECT", "CLAW_VFS_SCOPE",
    "CLAW_CONFINE", "CLAW_CONFINE_HOME", "CLAW_CONFINE_WORKSPACE", "FD_AGENT_CONFINE",
    "FD_URL", "FD_AGENT_SHARED_SECRET",
)


class _Probe(Tool):
    """Records the arguments it was called with."""

    def __init__(self, name, props=("url",)):
        self.name = name
        self.description = name
        self.parameters = {"type": "object",
                           "properties": {p: {"type": "string"} for p in props}, "required": []}
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, content="ok")


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """A deck's data dir: this agent (owner `owner`), another tenant's agent
    and VFS root, the deck's own files, and a folder outside it all."""
    from captain_claw.tools.glob import GlobTool
    from captain_claw.tools.grep import GrepTool
    from captain_claw.tools.read import ReadTool
    from captain_claw.tools.vfs import VfsTool
    from captain_claw.tools.write import WriteTool

    home_dir = tmp_path / "home"
    (home_dir / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    for var in _ENV_CLEARED:
        monkeypatch.delenv(var, raising=False)
    fd = (tmp_path / "fd-data").resolve()
    monkeypatch.setenv("FD_DATA_DIR", str(fd))
    monkeypatch.setenv("CLAW_VFS_USER", "owner")
    monkeypatch.setattr(get_config().tools.read, "extra_dirs", [])

    agent = fd / "agent-a"
    ws = agent / "data" / "workspace"
    (ws / "saved").mkdir(parents=True)
    (ws / "notes.md").write_text("MY NOTES alpha\n")
    (agent / ".env").write_text("OPENAI_API_KEY=team-key\n")
    (agent / "data" / "home-config").mkdir(parents=True)
    (agent / "data" / "home-config" / "config.yaml").write_text("model: x\n")
    (agent / "data" / "sessions").mkdir()
    (agent / "data" / "sessions" / "notes.txt").write_text("my own session notes\n")

    other = fd / "agent-b" / "data" / "workspace"
    other.mkdir(parents=True)
    (other / "deal.md").write_text("OTHER TENANT alpha\n")
    (fd / "flight-deck.db").write_text("tokens alpha\n")

    own_vfs = fd / "vfs" / "owner"
    (own_vfs / "proj").mkdir(parents=True)
    (own_vfs / "proj" / "mine.md").write_text("MY VFS alpha\n")
    victim_vfs = fd / "vfs" / "victim"
    (victim_vfs / "proj").mkdir(parents=True)
    (victim_vfs / "proj" / "theirs.md").write_text("VICTIM VFS alpha\n")

    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / "ext.md").write_text("EXTERNAL alpha\n")
    (own_vfs / ".vfs-links.json").write_text(json.dumps({"ext": {"path": str(outside), "mode": "rw"}}))
    monkeypatch.chdir(agent)    # where Flight Deck starts it

    reg = ToolRegistry(base_path=ws)
    for tool in (ReadTool(), WriteTool(), GlobTool(), GrepTool(), VfsTool()):
        reg.register(tool)
    probes = {"shell": _Probe("shell", ("command",)), "browser": _Probe("browser"),
              "hosting": _Probe("hosting", ("action", "kind", "name", "project", "start_cmd")),
              "direct_api": _Probe("direct_api", ("action", "url"))}
    for p in probes.values():
        reg.register(p)

    def confine(tier: str | None) -> None:
        if tier is None:
            for var in ("CLAW_CONFINE", "CLAW_CONFINE_HOME", "CLAW_CONFINE_WORKSPACE"):
                monkeypatch.delenv(var, raising=False)
            return
        monkeypatch.setenv("CLAW_CONFINE", tier)
        monkeypatch.setenv("CLAW_CONFINE_HOME", str(agent))
        monkeypatch.setenv("CLAW_CONFINE_WORKSPACE", str(ws))

    confine("user")
    return types.SimpleNamespace(
        reg=reg, fd=fd, agent=agent, ws=ws, other=other, own_vfs=own_vfs,
        victim_vfs=victim_vfs, outside=outside, probes=probes, confine=confine,
        tmp=tmp_path.resolve(),
    )


async def call(d, name, args):
    return await d.reg.execute(name, dict(args), runtime_base_path=d.ws)


async def refused(d, name, args) -> str:
    with pytest.raises(ToolBlockedError) as info:
        await call(d, name, args)
    reason = info.value.reason
    for marker in (str(d.tmp), str(d.fd), "/private/", "/Users/"):
        assert marker not in reason, (marker, reason)
    return reason


# ── off: a sign-in-off deck (one person) changes nothing ─────────────


async def test_unconfined_agent_reads_anywhere(deck):
    deck.confine(None)
    res = await call(deck, "read", {"path": str(deck.other / "deal.md")})
    assert "OTHER TENANT" in res.content
    res = await call(deck, "shell", {"command": "echo hi"})
    assert res.success


# ── files: own roots only ────────────────────────────────────────────


@pytest.mark.parametrize("tier", ["user", "admin"])
async def test_own_files_are_readable(deck, tier):
    deck.confine(tier)
    assert "MY NOTES" in (await call(deck, "read", {"path": "notes.md"})).content
    assert "MY VFS" in (await call(deck, "read", {"path": "vfs:proj/mine.md"})).content
    assert "MY VFS" in (await call(deck, "read", {"path": str(deck.own_vfs / "proj" / "mine.md")})).content
    assert "session notes" in (await call(deck, "read", {
        "path": str(deck.agent / "data" / "sessions" / "notes.txt")})).content


@pytest.mark.parametrize("tier", ["user", "admin"])
async def test_other_tenants_and_the_deck_are_refused(deck, tier):
    deck.confine(tier)
    targets = [deck.other / "deal.md", deck.fd / "flight-deck.db",
               deck.victim_vfs / "proj" / "theirs.md", Path("/etc/passwd")]
    if tier == "user":
        targets.append(deck.outside / "ext.md")     # an admin's agent follows its links
    for target in targets:
        reason = await refused(deck, "read", {"path": str(target)})
        assert reason.startswith(deck_confine.PATH_REFUSED), reason
    await refused(deck, "read", {"path": "../../../agent-b/data/workspace/deal.md"})
    await refused(deck, "read", {"path": "vfs:proj/../../victim/proj/theirs.md"})


async def test_the_agents_env_file_is_not_readable(deck):
    await refused(deck, "read", {"path": str(deck.agent / ".env")})
    await refused(deck, "read", {"path": "../../.env"})


async def test_a_symlink_out_of_the_workspace_is_refused(deck):
    (deck.ws / "sneaky.md").symlink_to(deck.other / "deal.md")
    await refused(deck, "read", {"path": "sneaky.md"})


async def test_writes_stay_in_the_workspace_and_vfs(deck):
    assert (await call(deck, "write", {"path": "out.md", "content": "x"})).success
    assert (await call(deck, "write", {"path": "vfs:proj/new.md", "content": "x"})).success
    await refused(deck, "write", {"path": str(deck.agent / ".env"), "content": "PYTHONPATH=x"})
    reason = await refused(deck, "write", {
        "path": str(deck.agent / "data" / "home-config" / "config.yaml"), "content": "x"})
    assert "read-only" in reason
    await refused(deck, "write", {"path": str(deck.own_vfs / ".vfs-links.json"), "content": "{}"})
    await refused(deck, "write", {"path": str(deck.other / "deal.md"), "content": "x"})
    assert (deck.agent / ".env").read_text() == "OPENAI_API_KEY=team-key\n"


async def test_glob_and_grep_stay_home(deck):
    await refused(deck, "glob", {"pattern": str(deck.fd) + "/**/*.md"})
    await refused(deck, "glob", {"pattern": "*.md", "root": str(deck.other)})
    await refused(deck, "grep", {"pattern": "alpha", "path": str(deck.fd)})
    # A directory symlink inside the workspace: ** follows it, results don't.
    (deck.ws / "elsewhere").symlink_to(deck.other, target_is_directory=True)
    res = await call(deck, "glob", {"pattern": "**/*.md"})
    assert "notes.md" in res.content and "deal.md" not in res.content
    res = await call(deck, "grep", {"pattern": "alpha", "path": "."})
    assert "MY NOTES" in res.content and "OTHER TENANT" not in res.content
    res = await call(deck, "glob", {"pattern": "vfs:**/*.md"})
    assert "mine.md" in res.content and "theirs.md" not in res.content


async def test_browser_takes_web_addresses_only(deck):
    for url in ("file:///etc/passwd", "FILE:" + str(deck.fd / "flight-deck.db"),
                "view-source:file:///etc/passwd", "chrome://settings"):
        reason = await refused(deck, "browser", {"url": url})
        assert "web address" in reason
    for url in ("https://example.com", "http://example.com/x", "example.com", "example.com:8080/x",
                "about:blank"):
        assert (await call(deck, "browser", {"url": url})).success, url


# ── links: a non-admin's agent follows only mounts in its own root ───


async def test_user_agent_ignores_links_outside_the_vfs_root(deck):
    from captain_claw import vfs

    assert vfs.project_root("ext") == (deck.own_vfs / "ext").resolve()
    try:
        res = await call(deck, "read", {"path": "vfs:ext/ext.md"})
    except Exception as exc:
        assert "EXTERNAL" not in str(exc)
    else:
        assert "EXTERNAL" not in (res.content or "")


async def test_admin_agent_follows_its_links(deck):
    from captain_claw import vfs

    deck.confine("admin")
    assert vfs.project_root("ext") == deck.outside
    assert "EXTERNAL" in (await call(deck, "read", {"path": "vfs:ext/ext.md"})).content
    assert "EXTERNAL" in (await call(deck, "read", {"path": str(deck.outside / "ext.md")})).content


def test_read_folders_are_an_admins(deck, monkeypatch):
    assert deck_confine.read_folders(["/srv/docs"]) == []
    deck.confine("admin")
    assert deck_confine.read_folders(["/srv/docs"]) == ["/srv/docs"]


# ── code: a non-admin's agent runs none ──────────────────────────────


async def test_user_agent_runs_no_code(deck):
    reason = await refused(deck, "shell", {"command": "cat /etc/passwd"})
    assert reason == deck_confine.EXEC_REFUSED
    assert "shell" not in deck.reg.list_tools()
    assert not deck.probes["shell"].calls


async def test_admin_agent_keeps_its_shell(deck):
    deck.confine("admin")
    assert "shell" in deck.reg.list_tools()
    assert (await call(deck, "shell", {"command": "echo hi"})).success


def test_user_agent_installs_no_skill_dependencies(deck):
    from captain_claw.skills import install_skill_dependencies

    res = install_skill_dependencies("anything", deck.ws, get_config())
    assert not res.ok and res.message == deck_confine.EXEC_REFUSED


# ── hosting: a non-admin's agent publishes static sites, not apps ────


async def test_user_agent_publishes_static_not_apps(deck):
    assert (await call(deck, "hosting", {"action": "publish", "kind": "static",
                                         "name": "site", "project": "saved"})).success
    for args in ({"action": "publish", "kind": "app", "name": "srv",
                  "project": "saved", "start_cmd": "node server.js"},
                 {"action": "start", "name": "srv"}):
        with pytest.raises(ToolBlockedError) as info:
            await call(deck, "hosting", args)
        assert info.value.reason == deck_confine.HOSTING_APP_REFUSED
    assert not deck.probes["hosting"].calls or all(
        c.get("kind") != "app" for c in deck.probes["hosting"].calls)


async def test_admin_agent_hosts_apps(deck):
    deck.confine("admin")
    assert (await call(deck, "hosting", {"action": "publish", "kind": "app", "name": "srv",
                                         "project": "saved", "start_cmd": "node server.js"})).success


# ── a relative path: ruled from the workspace, looked up from the agent's folder too ──

_BESIDE = ("../agent-b/data/workspace/deal.md", "x/../../agent-b/data/workspace/deal.md", "../flight-deck.db",
           "../vfs/victim/proj/theirs.md", ".env", "../agent-a/.env")


@pytest.mark.parametrize("tier", ["user", "admin"])
async def test_a_read_stays_in_from_the_agents_folder_too(deck, tier):
    deck.confine(tier)
    for path in (*_BESIDE, str(deck.ws / ".env")):
        await refused(deck, "read", {"path": path})


async def test_a_read_beside_the_decks_data_is_refused_too(deck):
    await refused(deck, "read", {"path": "../../outside/ext.md"})


@pytest.mark.parametrize("name,field", [
    ("public_link", "path"), ("google_mail", "attachments"), ("microsoft_mail", "attachments"),
    ("send_mail", "attachments"), ("linked_system", "files"), ("watch_background", "path"), ("jev", "state_file"),
    ("summarize_files", "folder"), ("google_drive", "local_path"), ("microsoft_onedrive", "local_path"),
])
def test_every_tool_that_looks_there_is_ruled_the_same(deck, name, field):
    for value in _BESIDE:
        assert deck_confine.check_tool_paths(name, {field: value}, deck.ws), value


async def test_own_files_are_still_found_from_either_folder(deck):
    (deck.agent / "only-here.md").write_text("FROM THE AGENTS FOLDER\n")
    assert "MY NOTES" in (await call(deck, "read", {"path": "notes.md"})).content
    assert "FROM THE AGENTS FOLDER" in (await call(deck, "read", {"path": "only-here.md"})).content
    assert "my own session notes" in (await call(deck, "read", {"path": "../sessions/notes.txt"})).content
    # A repository's own .env, in Code mode, isn't the agent's.
    (deck.ws / ".env").write_text("REPO=1\n")
    assert "REPO=1" in (await call(deck, "read", {"path": ".env"})).content
    # What isn't anywhere is just not found.
    missing = await call(deck, "read", {"path": "nothing.md"})
    assert not missing.success and "not found" in missing.error.lower()


async def test_a_drive_upload_looks_in_the_workspace_first(deck, monkeypatch):
    from captain_claw.tools import microsoft_onedrive
    from captain_claw.tools.google_drive import GoogleDriveTool
    from captain_claw.tools.microsoft_onedrive import MicrosoftOneDriveTool

    (deck.ws / ".env").write_text("REPO=1\n")
    runtime = {"_runtime_base_path": str(deck.ws)}
    assert Path(GoogleDriveTool._workspace_first(".env", runtime)).read_text() == "REPO=1\n"
    assert GoogleDriveTool._workspace_first("only-here.md", runtime) == "only-here.md"
    assert GoogleDriveTool._workspace_first(str(deck.ws / "notes.md"), runtime) == str(deck.ws / "notes.md")

    async def tokens(*_a, **_k):
        return types.SimpleNamespace(access_token="t", scope="")

    async def graph(*_a, **_k):
        return "t", frozenset()

    monkeypatch.setattr(microsoft_onedrive, "graph_token", graph)
    for tool in (GoogleDriveTool(), MicrosoftOneDriveTool()):
        seen = []

        async def upload(token, local_path="", **kwargs):
            seen.append(local_path)
            return ToolResult(success=True, content="ok")

        tool._get_tokens = tokens
        tool._action_upload = upload
        await tool.execute(action="upload", local_path=".env", _runtime_base_path=str(deck.ws))
        assert Path(seen[0]).read_text() == "REPO=1\n", tool.name


# ── payments: its source is a file on prepare, and how it's paid on purchase ──


@pytest.fixture
def prepared(deck, monkeypatch):
    """The real payments tool, under a Flight Deck that takes what it is sent.
    What it was sent."""
    from captain_claw import payments_client
    from captain_claw.tools.payments import PaymentsTool

    monkeypatch.setenv("FD_URL", "http://fd.test")
    sent: list[dict] = []

    async def fake_call(method, path, body=None):
        sent.append(body or {})
        return {"intent": {"id": "pay_1"}}, None

    monkeypatch.setattr(payments_client, "call", fake_call)
    deck.reg.register(PaymentsTool())
    return sent


@pytest.mark.parametrize("tier", ["user", "admin"])
async def test_an_invoice_file_has_to_be_this_agents(deck, prepared, tier):
    deck.confine(tier)
    for source in (str(deck.other / "deal.md"), "../../../agent-b/data/workspace/deal.md",
                   str(deck.victim_vfs / "proj" / "theirs.md"), "vfs:proj/../../victim/proj/theirs.md", *_BESIDE):
        await refused(deck, "payments", {"action": "prepare", "source": source})
    assert prepared == []


async def test_this_agents_own_invoice_is_still_read(deck, prepared):
    import base64

    (deck.ws / "saved" / "invoice.pdf").write_bytes(b"INVOICE")
    for source, text in (("notes.md", b"MY NOTES"), (str(deck.ws / "notes.md"), b"MY NOTES"),
                         ("vfs:proj/mine.md", b"MY VFS"), ("invoice.pdf", b"INVOICE")):
        assert (await call(deck, "payments", {"action": "prepare", "source": source})).success, source
        assert text in base64.b64decode(prepared[-1]["source_b64"])


def test_a_source_is_a_file_only_on_prepare(deck):
    check = deck_confine.check_tool_paths
    theirs = str(deck.other / "deal.md")
    assert check("payments", {"action": " Prepare ", "source": theirs}, deck.ws)
    for action in ("purchase", "list", ""):
        assert check("payments", {"action": action, "source": "revolut_card"}, deck.ws) is None
        assert check("payments", {"action": action, "source": theirs}, deck.ws) is None


# ── web tools: a non-admin's agent can't aim them at the server ──────


async def test_user_agent_web_tools_refuse_private_addresses(deck):
    for url in ("http://127.0.0.1:8765/fd/code/agent/list", "http://localhost/x",
                "http://169.254.169.254/latest/meta-data", "http://10.0.0.5:2019/config",
                "http://[::1]/"):
        reason = await refused(deck, "direct_api", {"action": "call", "url": url})
        assert reason == deck_confine.PRIVATE_URL_REFUSED
    assert (await call(deck, "direct_api",
                       {"action": "call", "url": "https://api.example.com/v1"})).success


async def test_admin_agent_web_tools_reach_anywhere(deck):
    deck.confine("admin")
    assert (await call(deck, "direct_api",
                       {"action": "call", "url": "http://127.0.0.1:8765/x"})).success


def test_url_public_classifies_addresses():
    pub = deck_confine.url_public
    assert pub("https://example.com") and pub("https://8.8.8.8")
    assert pub("") and pub("not a url")                       # nothing to resolve → allowed
    for bad in ("http://127.0.0.1/x", "http://localhost", "http://10.1.2.3:9",
                "http://169.254.169.254/", "http://[::1]/", "http://192.168.1.1/"):
        assert not pub(bad), bad


def test_read_or_write_follows_the_action():
    kind = deck_confine._kind
    assert kind("docx", "path", "docx", {"action": "read"}) == "read"
    assert kind("docx", "path", "docx", {"action": "create"}) == "write"
    assert kind("docx", "path", "docx", {"action": "edit"}) == "write"
    assert kind("docx", "path", "docx", {"action": "edit", "output_path": "x.docx"}) == "read"
    assert deck_confine.PATH_RULES["xlsx"][0] == ("path", "docx"), "a workbook goes the way a Word file goes"
    assert kind("xlsx", "path", "docx", {"action": "check"}) == "read"
    assert kind("datastore", "file_path", "datastore", {"action": "export"}) == "write"
    assert kind("datastore", "file_path", "datastore", {"action": "import_file"}) == "read"


# ── knowledge add: the file goes to Flight Deck, into the agent's documents ──


@pytest.fixture
def kept(deck, monkeypatch):
    """The real knowledge tool, under a Flight Deck that keeps whatever it is
    sent. What it was sent."""
    import httpx

    from captain_claw.tools.knowledge import KnowledgeTool

    monkeypatch.setenv("FD_URL", "http://fd.test")
    sent: list[bytes] = []
    real = httpx.AsyncClient

    def handler(request):
        sent.append(request.content)
        return httpx.Response(200, json={"doc": "doc.md", "title": "doc", "chars": 1})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    deck.reg.register(KnowledgeTool())
    return sent


@pytest.mark.parametrize("tier", ["user", "admin"])
async def test_knowledge_add_refuses_what_isnt_this_agents(deck, kept, tier):
    deck.confine(tier)
    (deck.ws / "alias.md").symlink_to(deck.other / "deal.md")
    for path in (str(deck.other / "deal.md"), "../../../agent-b/data/workspace/deal.md", "alias.md",
                 str(deck.victim_vfs / "proj" / "theirs.md"), "vfs:proj/../../victim/proj/theirs.md",
                 str(deck.agent / ".env")):
        await refused(deck, "knowledge", {"action": "add", "path": path})
    assert kept == []
    # Nor does the answer tell which of the server's files exist.
    there = await refused(deck, "knowledge", {"action": "add", "path": str(deck.fd / "flight-deck.db")})
    assert there == await refused(deck, "knowledge", {"action": "add", "path": str(deck.fd / "nothing.db")})


async def test_knowledge_add_keeps_this_agents_own_files(deck, kept):
    upload = deck.ws / "saved" / "media" / "s1" / "quote-20261031-101010-abc123.md"
    upload.parent.mkdir(parents=True)
    upload.write_text("FROM THE CHAT\n")
    for path, text in (("notes.md", b"MY NOTES"), (str(upload), b"FROM THE CHAT"), ("vfs:proj/mine.md", b"MY VFS")):
        assert (await call(deck, "knowledge", {"action": "add", "path": path})).success, path
        assert text in kept[-1]
    # What a picture said, with no file to read.
    assert (await call(deck, "knowledge", {"action": "add", "text": "Total 42", "name": "receipt.md"})).success


async def test_knowledge_add_elsewhere_is_for_a_deck_without_tenants(deck, kept):
    path = str(deck.other / "deal.md")
    await refused(deck, "knowledge", {"action": "add", "path": path})
    deck.confine(None)
    assert (await call(deck, "knowledge", {"action": "add", "path": path})).success
    assert b"OTHER TENANT" in kept[-1]


# ── coverage: every path-like argument of every tool is ruled ────────

_PATHY = re.compile(r"(path|file|dir|folder|root|glob|pattern|dest|output|^out$|"
                    r"attachment|image|^a$|^b$|template|url|source)")
# Not host paths: text to search for, ids, names, recipients, playbook text…
_NOT_PATHS = {
    "grep": {"pattern"}, "playbooks": {"do_pattern", "dont_pattern"},
    "google_drive": {"file_id", "folder_id"}, "microsoft_onedrive": {"file_id", "folder_id"},
    "google_mail": {"attachment"}, "microsoft_mail": {"attachment"},
    "intentions": {"match"}, "browser": {"api_id", "scroll_direction"},
    "image_gen": {"prompt"}, "insights": {"source", "source_section"},
    "reasoning_effort": {"source"}, "typesense": {"source"},
}


def _tool_classes() -> dict[str, type]:
    import importlib
    import inspect
    import pkgutil

    import captain_claw.tools as pkg

    out: dict[str, type] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            module = importlib.import_module(f"captain_claw.tools.{mod.name}")
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Tool) and cls is not Tool and cls.__module__ == module.__name__:
                params = getattr(cls, "parameters", None)
                if isinstance(params, dict) and getattr(cls, "name", ""):
                    out[cls.name] = cls
    return out


def test_every_path_like_argument_is_ruled():
    classes = _tool_classes()
    assert {"read", "write", "glob", "grep", "shell", "docx", "public_link"} <= set(classes)
    missing = []
    for name, cls in sorted(classes.items()):
        if name in deck_confine.EXEC_TOOLS:
            continue
        ruled = {field for field, _ in deck_confine.PATH_RULES.get(name, ())}
        free = deck_confine.NON_PATH_PARAMS.get(name, frozenset()) | _NOT_PATHS.get(name, set())
        for key in (cls.parameters.get("properties") or {}):
            if _PATHY.search(key) and key not in ruled and key not in free:
                missing.append(f"{name}.{key}")
    assert not missing, missing


# ── Flight Deck pins it (server._pin_confinement) ────────────────────


@pytest.fixture
def fd_server(deck, monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "DATA_DIR", deck.fd)
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    con = sqlite3.connect(deck.fd / "flight-deck.db.real")
    con.execute("CREATE TABLE users (id TEXT PRIMARY KEY, role TEXT)")
    con.executemany("INSERT INTO users VALUES (?, ?)", [("owner", "user"), ("boss", "admin")])
    con.commit()
    con.close()
    (deck.fd / "flight-deck.db").unlink()
    (deck.fd / "flight-deck.db.real").rename(deck.fd / "flight-deck.db")
    return server


def _spawn_env(server, deck, owner, user_env, workspace=""):
    base = {"PATH": "/usr/bin", "OPENAI_API_KEY": "deck", "FD_DATA_DIR": str(deck.fd)}
    env = {**base, **user_env}
    server._pin_confinement(env, base, agent_dir=deck.agent, owner_id=owner, workspace=workspace)
    return env


def test_non_admin_gets_user_and_loses_code_loading_env(fd_server, deck):
    env = _spawn_env(fd_server, deck, "owner", {
        "PYTHONPATH": str(deck.ws), "LD_PRELOAD": "x.so", "PATH": str(deck.ws),
        "GIT_SSH_COMMAND": "sh x", "CLAW_VFS_ROOT": "/", "FD_DATA_DIR": "/elsewhere",
        "https_proxy": "http://evil", "CLAW_CONFINE": "admin", "MY_TOKEN": "keep",
        "CLAW_VFS_SCOPE": "home,commons",
    })
    assert env["CLAW_CONFINE"] == "user"
    assert env["CLAW_CONFINE_HOME"] == str(deck.agent)
    assert env["CLAW_CONFINE_WORKSPACE"] == str(deck.agent / "data" / "workspace")
    for gone in ("PYTHONPATH", "LD_PRELOAD", "GIT_SSH_COMMAND", "CLAW_VFS_ROOT", "https_proxy"):
        assert gone not in env, gone
    assert env["PATH"] == "/usr/bin" and env["FD_DATA_DIR"] == str(deck.fd)
    assert env["MY_TOKEN"] == "keep"
    assert env["CLAW_VFS_SCOPE"] == "home,commons"     # only ever narrows


def test_non_admin_loses_the_other_loaders_and_routes_too(fd_server, deck):
    """Config/code a tool loads from a path (git via XDG, OpenSSL engines,
    Playwright's driver, Java agents, pip/uv indexes), a command a library
    runs (BROWSER, EDITOR, SSH_ASKPASS …), a resolver alias or a certificate
    authority of the owner's own."""
    named = {
        "XDG_CONFIG_HOME": str(deck.ws), "OPENSSL_CONF": "x.cnf", "PLAYWRIGHT_DRIVER_PATH": "x",
        "PLAYWRIGHT_BROWSERS_PATH": "x", "JAVA_TOOL_OPTIONS": "-javaagent:x", "_JAVA_OPTIONS": "x",
        "PIP_INDEX_URL": "http://evil", "UV_INDEX_URL": "http://evil", "BROWSER": "sh x",
        "EDITOR": "sh x", "SSH_ASKPASS": "x", "ZDOTDIR": "x", "LESSOPEN": "|x", "CURL_HOME": "x",
        "GCONV_PATH": "x", "HOSTALIASES": "x", "SSL_CERT_FILE": "ca.pem", "ssl_cert_dir": "x",
        "REQUESTS_CA_BUNDLE": "ca.pem", "CURL_CA_BUNDLE": "ca.pem", "SSLKEYLOGFILE": "k",
    }
    env = _spawn_env(fd_server, deck, "owner", {**named, "BRAVE_API_KEY": "keep", "MY_URL": "keep"})
    for gone in named:
        assert gone not in env, gone
    assert env["BRAVE_API_KEY"] == env["MY_URL"] == "keep"
    assert _spawn_env(fd_server, deck, "boss", named)["OPENSSL_CONF"] == "x.cnf"   # an admin's own


def test_admin_keeps_its_env(fd_server, deck):
    env = _spawn_env(fd_server, deck, "boss", {"PYTHONPATH": "/opt/lib"})
    assert env["CLAW_CONFINE"] == "admin" and env["PYTHONPATH"] == "/opt/lib"


def test_unknown_owner_is_a_user(fd_server, deck):
    assert _spawn_env(fd_server, deck, "", {})["CLAW_CONFINE"] == "user"
    assert _spawn_env(fd_server, deck, "nobody", {})["CLAW_CONFINE"] == "user"


def test_a_non_admins_workspace_must_be_its_own(fd_server, deck):
    default = str(deck.agent / "data" / "workspace")
    repo = deck.fd / "vfs" / "owner" / "proj"
    assert _spawn_env(fd_server, deck, "owner", {}, workspace=str(repo))["CLAW_CONFINE_WORKSPACE"] == str(repo)
    for ws in (str(deck.other), str(deck.victim_vfs / "proj"), "/etc"):
        assert _spawn_env(fd_server, deck, "owner", {}, workspace=ws)["CLAW_CONFINE_WORKSPACE"] == default
    assert _spawn_env(fd_server, deck, "boss", {}, workspace="/srv/repo")["CLAW_CONFINE_WORKSPACE"] == "/srv/repo"


def test_sign_in_off_or_switched_off_confines_nothing(fd_server, deck, monkeypatch):
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    env = _spawn_env(fd_server, deck, "owner", {"CLAW_CONFINE": "admin", "PYTHONPATH": "/x"})
    assert "CLAW_CONFINE" not in env and env["PYTHONPATH"] == "/x"
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AGENT_CONFINE", "off")
    assert "CLAW_CONFINE" not in _spawn_env(fd_server, deck, "owner", {})


# ── linking a folder of the server is an admin's ─────────────────────


def test_linking_a_server_folder_is_an_admins():
    from fastapi import HTTPException

    from captain_claw.flight_deck.vfs_routes import _require_link_admin

    with pytest.raises(HTTPException) as info:
        _require_link_admin({"id": "u1", "role": "user"})
    assert info.value.status_code == 403
    _require_link_admin({"id": "a1", "role": "admin"})
