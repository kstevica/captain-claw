"""Confinement for an agent on a multi-user Flight Deck (the agent side).

Flight Deck runs every process agent as the same OS user as itself, so
without this an agent's file tools reach the whole host: other tenants'
agent folders and VFS roots, the deck's database (everyone's Google tokens),
its ``.env``. On a sign-in deck Flight Deck pins these, after the agent's own
``.env`` and env_vars, so neither the agent nor its owner can change them
(``flight_deck/server.py: _pin_confinement``):

* ``CLAW_CONFINE`` — ``user`` (the owner isn't an admin) or ``admin``. Unset
  (sign-in off: one person's deck, the desktop app) changes nothing here.
* ``CLAW_CONFINE_HOME`` — the agent's own folder (``<FD_DATA_DIR>/<slug>``).
* ``CLAW_CONFINE_WORKSPACE`` — its workspace (Code mode's repo, else
  ``<home>/data/workspace``).

What an owner call's path arguments may name (a shared-agent member's calls
keep speaker.py's own, narrower rules):

* read: the agent's folder (not its ``.env``: a team key may be in it), its
  workspace, its owner's VFS root (Drive / OneDrive mounts live there) and
  the call's shared folders (pack_access); for ``admin`` also the owner's
  linked folders and configured read folders (``tools.read.extra_dirs``).
* write: the workspace and the VFS root — never the agent's control files
  (``.env``, ``data/home-config*``: Flight Deck re-reads them at restart),
  the VFS registry files or a Drive mount's tree; for ``admin`` also linked
  folders.

``user`` also loses the tools that run code or drive the host
(:data:`EXEC_TOOLS`) — one shell call reads whatever these rules refuse — and
its owner's links to folders outside their VFS root (``vfs.project_root``).
Until agents run in an OS sandbox an ``admin`` agent's shell still reaches
the whole host: this stops tenants and prompt-injected file calls, not an
admin's own shell.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

CONFINE_ENV = "CLAW_CONFINE"
HOME_ENV = "CLAW_CONFINE_HOME"
WORKSPACE_ENV = "CLAW_CONFINE_WORKSPACE"
TIERS = ("user", "admin")

# Run code or drive the host computer. A `user` agent never gets them.
EXEC_TOOLS = frozenset({
    "shell", "terminal", "scripts", "app_runner", "code", "termux", "bat",
    "desktop_action", "screen_capture", "pinchtab",
})

EXEC_REFUSED = (
    "Not available on this deck: running commands and driving the computer are "
    "reserved for admins' agents. Use read/write/glob and the other tools instead."
)
HOSTING_APP_REFUSED = (
    "Not available on this deck: hosting a runnable app (kind='app') starts a "
    "process on the server and is reserved for admins' agents. You can still "
    "publish a static site with kind='static'."
)
PRIVATE_URL_REFUSED = (
    "Not allowed on this deck: that address is on the server itself or a "
    "private network. Use a public http(s) address."
)
PATH_REFUSED = "Not allowed on this deck: "
_OUTSIDE = (" is outside this agent's own files (its workspace, its folder and "
            "your VFS folders)")

# Under the VFS root: Flight Deck's bookkeeping and the Drive mounts' trees.
_VFS_WRITE_REFUSED = frozenset({".vfs-links.json", ".vfs-meta.jsonl",
                                ".drive-manifest.json", ".drive-cache", ".drive"})
_WEB_SCHEMES = frozenset({"http", "https", "about"})
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_HOST_PORT_RE = re.compile(r"^[A-Za-z0-9.\-]+:\d")
_GLOB_CHARS = ("*", "?", "[")

# Every path-like argument of an owner call, per tool: (field, kind).
# kind — "read" / "write": a host path (absolute, ~, workspace-relative or
# vfs:); "glob": a pattern whose fixed prefix is read; "rel": relative to the
# tool's own folder, so never absolute and never ".."; "vfs": the vfs tool's
# vfs-space path; "url": a web address (no file:// and the like). docx and
# datastore pick read or write by action (:func:`_kind`). Tools in EXEC_TOOLS
# have none: `user` can't call them, and an admin's shell reaches the host.
PATH_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "read": (("path", "read"),),
    "write": (("path", "write"),),
    "edit": (("path", "write"),),
    "glob": (("pattern", "glob"), ("root", "read")),
    "grep": (("path", "read"), ("glob", "rel")),
    "pdf_extract": (("path", "read"),),
    "docx_extract": (("path", "read"),),
    "xlsx_extract": (("path", "read"),),
    "pptx_extract": (("path", "read"),),
    "docx": (("path", "docx"), ("output_path", "write"), ("template", "read")),
    "audio": (("path", "read"), ("output_path", "write")),
    "clipboard": (("path", "read"),),
    "codemap": (("path", "rel"),),
    "researchmap": (("path", "rel"),),
    "cv": (("path", "read"), ("paths", "read"), ("a", "read"), ("b", "read"),
           ("image", "read"), ("template", "read"), ("out", "write")),
    "datastore": (("file_path", "datastore"),),
    "flight_deck": (("file", "read"),),
    "google_drive": (("local_path", "read"), ("output_path", "write")),
    "google_mail": (("attachments", "read"),),
    "microsoft_mail": (("attachments", "read"),),
    "microsoft_onedrive": (("local_path", "read"), ("output_path", "write")),
    "send_mail": (("attachments", "read"),),
    "image_gen": (("output_path", "write"),),
    "visualize": (("output_path", "write"),),
    "image_ocr": (("path", "read"),),
    "image_vision": (("path", "read"),),
    "video_vision": (("path", "read"),),
    "linked_system": (("files", "read"),),
    "pocket_tts": (("output_path", "write"),),
    "public_link": (("path", "read"),),
    "summarize_files": (("folder", "read"), ("pattern", "rel")),
    "twitter": (("image_path", "read"),),
    "typesense": (("file_path", "read"),),
    "vatra": (("path", "read"),),
    "watch_background": (("path", "read"),),
    "jev": (("state_file", "read"),),
    "knowledge": (("path", "read"),),
    "whatsapp_send_file": (("path", "read"),),
    "vfs": (("path", "vfs"), ("to", "vfs")),
    "hosting": (("subdir", "rel"),),
    "browser": (("url", "url"),),
}

# Schema properties that look like paths but aren't (the test that keeps
# PATH_RULES complete reads this).
NON_PATH_PARAMS: dict[str, frozenset[str]] = {
    # which attachment of a message / draft (a number or a name); a mail folder
    "google_mail": frozenset({"attachment", "remove_attachments"}),
    "microsoft_mail": frozenset({"attachment", "remove_attachments", "folder"}),
    "microsoft_onedrive": frozenset({"path"}),       # a path inside OneDrive
    "whatsapp_send_file": frozenset({"filename"}),   # a name it looks up in saved/
    "payments": frozenset({"merchant_url"}),
    "app_runner": frozenset({"path"}),               # an HTTP route of the app
    "web_fetch": frozenset({"url"}), "web_get": frozenset({"url"}),
    "web_fetch_batch": frozenset({"urls"}),           # httpx: web schemes only
    "direct_api": frozenset({"url"}), "apis": frozenset({"base_url"}),
}

# Tool arguments that carry an OUTBOUND address (not a file path). For a
# `user`-tier agent these must not point back at the server or a private
# network — otherwise the agent reaches Flight Deck's own loopback routes or a
# neighbouring service instead of the public web. (admin agents, and decks
# with sign-in off, keep the normal behaviour.)
URL_PARAMS: dict[str, tuple[str, ...]] = {
    "browser": ("url",),
    "direct_api": ("url",),
    "apis": ("base_url",),
    "web_fetch": ("url",),
    "web_get": ("url",),
    "web_fetch_batch": ("urls",),
}


def tier() -> str:
    value = (os.environ.get(CONFINE_ENV) or "").strip().lower()
    return value if value in TIERS else ""


def active() -> bool:
    return bool(tier())


def exec_blocked(name: str) -> bool:
    return tier() == "user" and name in EXEC_TOOLS


def hosting_app_blocked(name: str, arguments: dict | None) -> bool:
    """A `user`-tier agent may publish a static site but not a runnable app: an
    app runs its start command as a process on the shared host. Static hosting,
    and every other hosting action (list/stop/status/unpublish), stay open."""
    if tier() != "user" or name != "hosting":
        return False
    args = arguments or {}
    action = str(args.get("action") or "").strip().lower()
    if action == "start":
        return True
    if action == "publish":
        return str(args.get("kind") or "static").strip().lower() == "app"
    return False


def external_links_allowed() -> bool:
    """Whether the owner's links to folders outside their VFS root resolve."""
    return tier() != "user"


def read_folders(dirs: list[str]) -> list[str]:
    """The configured read folders (tools.read.extra_dirs) this agent may use:
    none for ``user`` — the owner can change the config, not this deck's walls."""
    return [] if tier() == "user" else list(dirs or [])


def _env_path(key: str) -> Path | None:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _within(p: Path, root: Path | None) -> bool:
    if root is None:
        return False
    from captain_claw.vfs import path_within

    return path_within(p, root)


def _vfs_root() -> Path | None:
    try:
        from captain_claw import vfs

        return vfs.user_root()
    except (PermissionError, OSError, RuntimeError, ValueError):
        return None


def _external_roots(cls: str) -> list[Path]:
    """``admin`` only: the owner's linked folders (read and write — a link's
    own read-only flag is the vfs layer's) and, for reads, the configured
    read folders."""
    if tier() != "admin":
        return []
    out: list[Path] = []
    try:
        from captain_claw import vfs

        root = vfs.user_root()
        for name in vfs.read_links_at(root):
            tgt = vfs.safe_link_target_at(root, name)
            if tgt is not None:
                out.append(tgt.resolve())
    except (PermissionError, OSError, RuntimeError, ValueError):
        pass
    if cls == "read":
        try:
            from captain_claw.config import get_config

            for d in get_config().tools.read.extra_dirs or []:
                out.append(Path(d).expanduser().resolve())
        except Exception:
            pass
    return out


def _roots(cls: str) -> tuple[Path | None, Path | None, Path | None, list[Path]]:
    """(home, workspace, VFS root, external roots) for one check or one batch."""
    return _env_path(HOME_ENV), _env_path(WORKSPACE_ENV), _vfs_root(), _external_roots(cls)


def allowed(real: Path, cls: str, roots: tuple | None = None) -> bool:
    """Whether a resolved path is this agent's to read (``cls="read"``) or
    write. Never raises."""
    try:
        home, workspace, vfs_root, external = roots if roots is not None else _roots(cls)
        if cls == "read":
            from captain_claw import pack_access

            if pack_access.pack_of_path(real) is not None:
                return pack_access.result_ok(real)
            if _within(real, workspace):
                return True
            if _within(real, home):
                return real != home / ".env"
        elif _within(real, workspace):
            return True
        if _within(real, vfs_root):
            if cls == "read":
                return True
            parts = real.relative_to(vfs_root).parts if real != vfs_root else ()
            return not any(part in _VFS_WRITE_REFUSED for part in parts)
        return any(_within(real, r) for r in external)
    except Exception:
        return False


def filter_results(paths: list[str]) -> list[str]:
    """For glob/grep results: all of them when confinement is off, else only
    the files this agent may read (realpath — ``**`` follows symlinks)."""
    if not active():
        return paths
    roots = _roots("read")
    out = []
    for p in paths:
        try:
            if allowed(Path(p).resolve(), "read", roots):
                out.append(p)
        except Exception:
            pass
    return out


def url_ok(value: str) -> bool:
    """A web address (or a bare host): never file://, view-source:, chrome:…"""
    text = str(value or "").strip()
    m = _SCHEME_RE.match(text)
    if not m or m.group(1).lower() in _WEB_SCHEMES:
        return True
    return bool(_HOST_PORT_RE.match(text))      # "example.com:8080/x"


def _url_host(value: str) -> str | None:
    """The host of a URL or a bare ``host[:port][/path]``, or None when there
    is nothing to resolve (e.g. a path-only value)."""
    text = str(value or "").strip()
    if not text:
        return None
    from urllib.parse import urlsplit

    # A scheme-less "example.com:8080/x" needs a leading "//" for urlsplit to
    # read it as an authority rather than a scheme.
    ref = text if _SCHEME_RE.match(text) else "//" + text
    try:
        return urlsplit(ref, scheme="http").hostname or None
    except ValueError:
        return None


def _ip_is_local(ip: str) -> bool:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return bool(addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def url_public(value: str) -> bool:
    """Whether a URL's host is a public address — the gate for a ``user``-tier
    agent's outbound web tools. A literal private / loopback / link-local IP
    fails; so does ``localhost``, and any hostname that resolves to such an
    address. A name that can't be resolved right now is allowed (DNS may be
    down; the tool's own request then fails on its own)."""
    host = _url_host(value)
    if host is None:
        return True
    import ipaddress

    try:
        ipaddress.ip_address(host)
        return not _ip_is_local(host)           # a literal IP in the host
    except ValueError:
        pass
    low = host.lower().rstrip(".")
    if low == "localhost" or low.endswith(".localhost"):
        return False
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    return not any(_ip_is_local(ai[4][0]) for ai in infos)


def _resolve(value: str, base: Path) -> Path | None:
    from captain_claw import vfs

    if vfs.is_vfs_path(value):
        return vfs.resolve_vfs_path(value)
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _glob_prefix(pattern: str, base: Path) -> Path:
    fixed: list[str] = []
    for part in Path(pattern).expanduser().parts:
        if any(c in part for c in _GLOB_CHARS):
            break
        fixed.append(part)
    p = Path(*fixed) if fixed else Path(".")
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _kind(name: str, field: str, kind: str, arguments: dict) -> str:
    action = str(arguments.get("action") or "").strip().lower()
    if kind == "docx":
        # create writes `path`; edit/accept/reject change it in place unless
        # they write output_path; the rest only read it.
        if action == "create":
            return "write"
        if action in ("edit", "accept", "reject") and not arguments.get("output_path"):
            return "write"
        return "read"
    if kind == "datastore":
        return "write" if action == "export" else "read"
    return kind


def _check_value(field: str, kind: str, value: str, arguments: dict, base: Path) -> str | None:
    if "\x00" in value:
        return PATH_REFUSED + f"{field} must be a plain text path"
    if kind == "url":
        return None if url_ok(value) else PATH_REFUSED + f"{field} must be a web address (http or https)"
    if kind == "rel":
        from captain_claw import vfs

        p = Path(value)
        if vfs.is_vfs_path(value) or p.is_absolute() or value.startswith("~") or ".." in p.parts:
            return PATH_REFUSED + f"{field} must stay inside the folder it is relative to"
        return None
    if kind == "vfs":
        from captain_claw import vfs

        real = vfs.resolve_vfs_path(value if vfs.is_vfs_path(value) else f"vfs:{value}")
        if real is None:
            return PATH_REFUSED + f"{field} leaves your VFS folders"
        return None if allowed(real, "write" if arguments.get("action") in (
            "mkdir", "mv", "rm", "write") else "read") else PATH_REFUSED + field + _OUTSIDE
    if kind == "glob":
        from captain_claw import vfs

        if vfs.is_vfs_path(value):
            return None     # resolved under the VFS root; results are filtered
        root = arguments.get("root")
        start = base
        if isinstance(root, str) and root.strip():
            start = Path(root).expanduser()
            if not start.is_absolute():
                start = base / start
        real = _glob_prefix(value, start)
        return None if allowed(real, "read") else PATH_REFUSED + field + _OUTSIDE
    real = _resolve(value, base)
    if real is None:
        return PATH_REFUSED + f"{field} leaves your VFS folders"
    if allowed(real, kind):
        return None
    if kind == "write" and allowed(real, "read"):
        return PATH_REFUSED + f"{field} is read-only for this agent"
    return PATH_REFUSED + field + _OUTSIDE


def check_tool_paths(name: str, arguments: dict | None, runtime_base: Any) -> str | None:
    """The refusal for an owner call whose path arguments leave this agent's
    roots, else None. Messages never contain host paths. Run it in the tool's
    context, after pack_access.prepare_call, so shared folders resolve."""
    if not active():
        return None
    args = arguments or {}
    # Outbound-address tools: a `user`-tier agent may not aim them at the server
    # or a private network. Checked before the path rules, since most of these
    # tools carry no file path (they aren't in PATH_RULES).
    if tier() == "user":
        for field in URL_PARAMS.get(name, ()):
            value = args.get(field)
            if value is None or value == "":
                continue
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                if isinstance(item, str) and item and not url_public(item):
                    return PRIVATE_URL_REFUSED
    rules = PATH_RULES.get(name)
    if not rules:
        return None
    try:
        base = Path(runtime_base).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return PATH_REFUSED + "this agent has no workspace"
    try:
        for field, kind in rules:
            value = args.get(field)
            if value is None or value == "":
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if item is None or item == "":
                    continue
                if not isinstance(item, str):
                    return PATH_REFUSED + f"{field} must be a plain text path"
                refusal = _check_value(field, _kind(name, field, kind, args), item, args, base)
                if refusal:
                    return refusal
    except (PermissionError, OSError, RuntimeError, ValueError):
        return PATH_REFUSED + "that path can't be checked"
    return None
