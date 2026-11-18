"""In-turn context governor: big tool results spill to files, pages are de-duplicated.

A Flight Deck worker (Vatra) does its whole piece inside ONE long turn, so its
tool results pile up in the request until the per-call packer drops them. With
the governor on (``CLAW_CTX_GOVERNOR=1``, or ``context.turn_governor``):

* **Spill at birth.** A content tool's result larger than ~10% of the agent's
  real context window is written in full to a file, and the model gets the head
  plus a pointer it can ``read`` (offset/limit) or ``grep``. In a run with a
  bound shared folder the file lands in ``vfs:<project>/sources/`` (outside the
  worker's home, so it survives teardown and teammates can read it); otherwise
  in the agent's ``saved/downloads/<session>/context/``.
* **Page de-duplication.** web_fetch saves every page it fetches to the run's
  ``sources/`` under a name derived from the URL, so a teammate (or the same
  worker) fetching that URL again gets the saved copy's pointer and a short
  preview instead of the whole page a second time.

Everything here is a no-op with the governor off.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

GOVERNOR_ENV = "CLAW_CTX_GOVERNOR"
# How much one content tool may extract for the spill to see (the inline size is
# decided after, from the agent's window) — a safety bound, not a budget.
SPILL_SOURCE_MAX = 1_000_000
# Content tools whose large results spill. `read` never does: its file is
# already on disk and it pages with offset/limit.
SPILL_TOOLS = frozenset({
    "web_fetch", "web_get",
    "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
})  # web_fetch_batch shapes its own governed output (one share per URL)
POINTER_MARK = "Full text saved to "
DEDUPE_MARK = "Already fetched in this run"
# Share of the agent's real window one inline result may take, in real tokens.
_INLINE_SHARE = 0.10
_INLINE_MIN_TOKENS = 4_000
_INLINE_MAX_TOKENS = 24_000
_MIN_REQUESTED_CHARS = 4_000
# A saved page older than this is fetched again rather than reused.
_DEDUPE_MAX_AGE_S = 24 * 3600
_DEDUPE_PREVIEW_CHARS = 4_000
# A saved page this small is cheaper to fetch again than to trust (a bot
# challenge, a consent wall, a JS shell); it is neither shared nor reused.
_MIN_SHARED_CHARS = 1_500
_CHALLENGE_HINTS = (
    "just a moment", "enable javascript", "access denied", "captcha",
    "are you a robot", "verify you are human", "request blocked", "cf-chl",
)
_WORKER_MARKERS = ("CLAW_VATRA_WORKER", "CLAW_BASNA_WORKER")
_HEADER_RE = re.compile(
    r"<!-- fetched (?P<at>\d+)(?: by (?P<by>[^·]+?))? · (?P<chars>\d+) chars"
    r"(?: · (?P<tool>[a-z_]+))?(?: · status (?P<status>\d+))?")
# The first line of a governed result (a spilled preview, a de-duplicated page).
_GOVERNED_LINE_RE = re.compile(
    r"^\[(?:[a-z_]+ .*? · \d+ chars · " + re.escape(POINTER_MARK) + r"|" + re.escape(DEDUPE_MARK) + r")")


def governor_enabled() -> bool:
    """Whether the in-turn context governor is on for this process."""
    raw = os.environ.get(GOVERNOR_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    try:
        return bool(getattr(get_config().context, "turn_governor", False))
    except Exception:  # noqa: BLE001 — sizing must never break a call
        return False


def ctx_ratio(agent: Any = None) -> float:
    """The agent's calibrated real/estimated token ratio (1.0 when unknown)."""
    fn = getattr(agent, "_ctx_ratio", None)
    if callable(fn):
        try:
            return max(1.0, float(fn()))
        except Exception:  # noqa: BLE001
            return 1.0
    return 1.0


def real_context_budget(agent: Any = None) -> int:
    """The agent's real context window in tokens: ``context.max_tokens``, cut to
    the provider's ``num_ctx`` or a window learned from an overflow error."""
    try:
        tokens = int(get_config().context.max_tokens or 0)
    except Exception:  # noqa: BLE001
        tokens = 0
    learned = getattr(agent, "_learned_window", None)
    for value in (
        getattr(getattr(agent, "provider", None), "num_ctx", 0),
        learned() if callable(learned) else 0,
    ):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < (tokens or value + 1):
            tokens = value
    return max(0, tokens)


def inline_cap_chars(agent: Any = None) -> int:
    """Characters one tool result may keep inline: ~10% of the real window, in
    real tokens (so a model whose tokenizer reads numbers denser gets fewer)."""
    budget = real_context_budget(agent) or 160_000
    cap_tokens = min(_INLINE_MAX_TOKENS, max(_INLINE_MIN_TOKENS, int(budget * _INLINE_SHARE)))
    return max(_MIN_REQUESTED_CHARS, int(cap_tokens * 4 / ctx_ratio(agent)))


def _slug(text: str, limit: int = 80) -> str:
    s = re.sub(r"^https?://", "", text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s[:limit] or "source"


def _sha8(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]


def source_file_name(tool: str, key: str) -> str:
    """Deterministic file name for a saved source. A page web_fetch read is
    named the way R10's corpus names it (``<slug>-<sha8>.md``) — that namespace
    is the run's page dedupe, and only web_fetch writes it. Everything else is
    prefixed with its tool."""
    if tool == "web_fetch" and str(key).lower().startswith(("http://", "https://")):
        return f"{_slug(key)}-{_sha8(key)}.md"
    base = os.path.basename(str(key).rstrip("/\\")) or str(key)
    return f"{tool}-{_slug(base, 60)}-{_sha8(str(key))}.md"


def notes_path(project: str = "", subtask: str = "", session: str = "") -> str:
    """A worker's running notes file. In a Vatra run it sits in the run's shared
    folder as ``vfs:<project>/notes/<sid8>-<subtask>-notes.md`` — the session
    prefix keeps a continuation round (same folder, same s1/s2 ids) from
    appending to an earlier round's notes, and the suffix keeps it from sharing a
    basename with a declared deliverable. A Bat step's worker keeps them there
    too (its step's id), so they outlive the worker: a retry of the step finds
    them. Elsewhere ``notes.md``."""
    project = (project or os.environ.get("CLAW_VFS_PROJECT", "")).strip()
    subtask = (subtask or os.environ.get("CLAW_VATRA_SUBTASK", "")
               or os.environ.get("CLAW_BAT_SUBTASK", "")).strip()
    session = (session or os.environ.get("CLAW_VATRA_SESSION", "")
               or os.environ.get("CLAW_BAT_SESSION", "")).strip()
    if project and subtask:
        prefix = f"{_slug(session, 8)}-" if session else ""
        return f"vfs:{project}/notes/{prefix}{_slug(subtask, 60)}-notes.md"
    return "notes.md"


_GOVERNOR_NOTES_RE = re.compile(r"(?:^|/)notes/[^/]+-notes\.md$", re.IGNORECASE)


def is_governor_notes_path(path: Any) -> bool:
    """Whether *path* has the shape :func:`notes_path` gives a Vatra worker
    (``…/notes/<sid8>-<subtask>-notes.md``) — scratch, never a deliverable. Kept
    tight on purpose: a real ``notes/`` package or a ``notes.md`` deliverable
    is not matched."""
    return bool(_GOVERNOR_NOTES_RE.search(str(path or "").strip().replace("\\", "/")))


def run_sources_dir() -> tuple[Path, str] | None:
    """``(directory, vfs pointer prefix)`` of the run's shared ``sources/`` —
    only for a Flight Deck worker bound to a run folder."""
    if not os.environ.get("CLAW_VFS_PROJECT", "").strip():
        return None
    if not any(os.environ.get(marker) for marker in _WORKER_MARKERS):
        return None
    try:
        from captain_claw import vfs

        if vfs.project_is_readonly():
            return None
        directory = vfs.project_root(create=True) / "sources"
        directory.mkdir(parents=True, exist_ok=True)
        return directory, f"vfs:{vfs.default_project()}/sources"
    except Exception as e:  # noqa: BLE001 — fall back to the private spill dir
        log.debug("run sources dir unavailable", error=str(e))
        return None


def _local_spill_dir(agent: Any) -> tuple[Path, str]:
    # Hidden, and outside saved/: a Flight Deck run sweeps a worker's workspace
    # files into its deliverables (vatra _capture_generated skips `.context`).
    base = getattr(agent, "workspace_base_path", None) or os.getcwd()
    session = getattr(getattr(agent, "session", None), "id", None) or "session"
    rel = Path(".context") / str(session)
    directory = Path(base) / rel
    directory.mkdir(parents=True, exist_ok=True)
    return directory, str(rel)


def _owner_label(agent: Any) -> str:
    owner = os.environ.get("CLAW_VATRA_OWNER", "").strip()
    if owner:
        return owner
    return str(getattr(agent, "agent_name", "") or getattr(agent, "name", "") or "an agent")[:60]


def shareable_page(content: str, status: int | None = None) -> bool:
    """Whether a fetched page is worth sharing as the run's copy: a success,
    not tiny, not a bot-challenge or consent wall."""
    if status is not None and int(status) >= 400:
        return False
    text = content or ""
    if len(text) < _MIN_SHARED_CHARS:
        return False
    head = text[:3000].lower()
    return not (len(text) < 8_000 and any(h in head for h in _CHALLENGE_HINTS))


def save_source(agent: Any, tool: str, key: str, content: str, *, shared: bool = True,
                status: int | None = None) -> str | None:
    """Write *content* to the run's ``sources/`` (else the agent's private spill
    dir) and return the pointer the model can read; None when nothing could be
    written."""
    if not content:
        return None
    target = run_sources_dir() if shared else None
    tmp_name = None
    try:
        directory, prefix = target if target else _local_spill_dir(agent)
        name = source_file_name(tool, key)
        status_part = f" · status {int(status)}" if status is not None else ""
        header = (f"# Source: {key}\n\n<!-- fetched {int(time.time())} by "
                  f"{_owner_label(agent)} · {len(content)} chars · {tool}{status_part} -->\n\n")
        path = directory / name
        # A unique temp file: teammates may save the same page at once.
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(header + content)
        os.replace(tmp_name, path)
        tmp_name = None
        return f"{prefix}/{name}"
    except Exception as e:  # noqa: BLE001 — a failed spill keeps the inline text
        log.warning("context spill write failed", tool=tool, error=str(e))
        return None
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def is_governed_output(content: str) -> bool:
    """Whether a tool result is already a governed view (spilled or deduped)."""
    return bool(_GOVERNED_LINE_RE.match((content or "")[:2000]))


def spill_view(tool: str, key: str, content: str, pointer: str, cap: int) -> str:
    """The inline result for a saved source: a pointer line, then the head."""
    total = len(content)
    if total <= cap:
        return f"[{tool} {key} · {total} chars · {POINTER_MARK}{pointer}]\n\n{content}"
    return (
        f"[{tool} {key} · {total} chars · {POINTER_MARK}{pointer} — read it with `read` "
        "(offset/limit) or `grep` for anything past this preview]\n\n"
        f"{content[:cap]}\n\n... [preview only — {total - cap} more chars in {pointer}]"
    )


def _key_for(tool: str, arguments: dict[str, Any] | None) -> str:
    args = arguments or {}
    if tool == "web_fetch_batch":
        urls = args.get("urls") or []
        return f"batch of {len(urls)} URLs" if isinstance(urls, list) else "batch"
    return str(args.get("url") or args.get("path") or args.get("file_id") or tool)


def spill_tool_result(agent: Any, tool: str, arguments: dict[str, Any] | None,
                      content: str) -> str:
    """Spill a large content-tool result to a file; return what the model sees."""
    if not governor_enabled() or tool not in SPILL_TOOLS or not isinstance(content, str):
        return content
    if is_governed_output(content) or content.lstrip().startswith("[Full text saved to "):
        return content  # the tool already saved it (web_fetch / R10 corpus)
    cap = inline_cap_chars(agent)
    requested = (arguments or {}).get("max_chars")
    try:
        if requested is not None and int(requested) < cap:
            cap = max(_MIN_REQUESTED_CHARS, int(requested))
    except (TypeError, ValueError):
        pass
    if len(content) <= cap:
        return content
    key = _key_for(tool, arguments)
    # Private: only web_fetch's own governed path writes the run's shared copy.
    pointer = save_source(agent, tool, key, content, shared=False)
    if not pointer:
        return (content[:cap] + f"\n\n... [{len(content) - cap} more chars cut — the full "
                "result could not be saved; narrow the request to see them]")
    log.info("Tool result spilled", tool=tool, chars=len(content), inline=cap, pointer=pointer)
    return spill_view(tool, key, content, pointer, cap)


# ── page de-duplication ──────────────────────────────────────────────


def find_fetched(url: str) -> dict[str, Any] | None:
    """The run's saved copy of *url*, if any: {path, pointer, by, at, chars}."""
    found = run_sources_dir()
    if not found or not str(url).lower().startswith(("http://", "https://")):
        return None
    directory, prefix = found
    name = source_file_name("web_fetch", url)
    path = directory / name
    try:
        if not path.is_file():
            return None
        head = path.read_text(encoding="utf-8", errors="replace")[:600]
    except OSError:
        return None
    match = _HEADER_RE.search(head)
    at = int(match.group("at")) if match else int(path.stat().st_mtime)
    if time.time() - at > _DEDUPE_MAX_AGE_S:
        return None
    if match and match.group("tool") and match.group("tool") != "web_fetch":
        return None          # another tool's spill can't stand in for the page
    if match and match.group("status") and int(match.group("status")) >= 400:
        return None
    if match and int(match.group("chars")) < _MIN_SHARED_CHARS:
        return None          # cheaper to fetch again than to trust
    return {
        "path": path, "pointer": f"{prefix}/{name}",
        "by": (match.group("by") or "").strip() if match else "",
        "at": at, "chars": int(match.group("chars")) if match else 0,
    }


def _body_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    _, sep, body = text.partition(" -->\n\n")
    return body if sep else text


def dedupe_view(url: str, found: dict[str, Any], agent: Any = None) -> str:
    """What web_fetch returns for a page the run already fetched."""
    try:
        body = _body_of(found["path"])
    except OSError:
        body = ""
    preview = min(_DEDUPE_PREVIEW_CHARS, inline_cap_chars(agent))
    who = f" by {found['by']}" if found.get("by") else ""
    chars = found.get("chars") or len(body)
    when = time.strftime("%H:%M UTC", time.gmtime(found.get("at") or time.time()))
    out = (f"[{DEDUPE_MARK}{who} at {when} — {POINTER_MARK}{found['pointer']} ({chars} chars). "
           "The preview below is its start; read the file with `read` (offset/limit) or `grep` "
           f"for the rest instead of fetching the page again.]\n\n[URL: {url}]\n\n{body[:preview]}")
    if len(body) > preview:
        out += f"\n\n... [preview only — {len(body) - preview} more chars in {found['pointer']}]"
    return out
