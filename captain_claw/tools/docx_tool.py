"""docx tool: Word documents made and changed properly, without scripts.

The work happens in :mod:`captain_claw.docx_ops`; this module decides where
files are read from and written to, keeps a backup before changing a file in
place, and reports what changed in a form the model can check at a glance.

Where an edit lands: a file in the agent's own ``saved/`` area (outside
``downloads/`` and the working folders) or in a writable VFS project is
changed in place, with a backup. Anything else — a user's upload in
``saved/downloads/``, a read-only Drive folder, a file elsewhere on disk — is
left untouched and the redlined copy goes to ``saved/output/<session>/`` under
the same name, which is also where chat bridges pick deliverables up.

A shared agent's member (speaker.py) works on their own files only: the path
table there confines ``path`` / ``output_path`` / ``template``; here a change
lands in place only when the member may change that file (else their redlined
copy goes to their own folder), every file written must pass
:func:`speaker.member_may_write`, pictures named in Markdown must be ones the
member can read, no backup is kept, results never show host paths, and the
changes carry the member's name.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from captain_claw import docx_ops as D
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool
from captain_claw.vfs import is_vfs_path, project_is_readonly, resolve_vfs_path, split_scheme

log = get_logger(__name__)

ACTIONS = ("create", "read", "edit", "changes", "comments", "accept", "reject")
_WORD_SUFFIXES = {".docx", ".docm"}
# A document name the model gave another extension: report.doc → report.docx.
_OTHER_TEXT_SUFFIXES = {".doc", ".md", ".txt", ".rtf", ".odt", ".html", ".pdf"}
_NOT_IN_PLACE = ("downloads", "scripts", "tools", "skills")
_PREVIEW_CHARS = 400


def docx_file_name(path: str) -> str:
    """*path* with the extension the saved file gets: report.doc → report.docx,
    "Offer" → "Offer.docx" (speaker.py checks a member's path under this name)."""
    suffix = Path(path).suffix.lower()
    if suffix in _WORD_SUFFIXES:
        return path
    if suffix in _OTHER_TEXT_SUFFIXES:
        return path[: -len(suffix)] + ".docx"
    return path + ".docx"


def _member() -> Any:
    from captain_claw import speaker

    return speaker.current()


def _author(kwargs: dict[str, Any]) -> str:
    """The name on tracked changes and comments: a member's own (they can't
    pick one; speaker.py drops `author`), else the one asked for or the agent's."""
    member = _member()
    if member is not None:
        return f"{member.display_name or 'Member'} (via {_default_author()})"
    return str(kwargs.get("author") or "").strip() or _default_author()


def _show(path: Path) -> str:
    """How a file is named in results: a member sees saved/… or vfs:…, never a host path."""
    if _member() is None:
        return str(path)
    from captain_claw.tools.edit import EditTool

    return EditTool._member_display(path)


def _scrub(text: str) -> str:
    """An error for a member without host paths."""
    from captain_claw import speaker

    roots = speaker.current_roots() if _member() is not None else None
    if roots is None:
        return text
    for root, shown in ((roots.vfs_root, "vfs:"), (roots.saved_base, "saved"),
                        (roots.runtime_base, ".")):
        if root is not None:
            text = text.replace(str(root), shown)
    return text


def _writable(target: Path) -> Path:
    """*target* when this call may write it (always for the owner)."""
    from captain_claw import speaker

    if _member() is not None and not speaker.member_may_write(target):
        raise D.DocxError(speaker.PATH_REFUSED_PREFIX + "you can only save in your own folders, "
                          "and change only files you created — pass output_path "
                          "(a new file name, or vfs:<project>/<name>.docx).")
    return target


def _default_author() -> str:
    try:
        from captain_claw.personality import load_personality

        name = (load_personality().name or "").strip()
        return name or "Captain"
    except Exception:
        return "Captain"


def _edits_arg(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise D.DocxError(f"`edits` is not valid JSON: {exc}") from None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise D.DocxError("`edits` must be a non-empty list of edit objects, each with an `op`.")
    return raw


def _ids_arg(raw: Any) -> list[str] | None:
    if raw in (None, "", []):
        return None
    if isinstance(raw, str):
        raw = [p for p in raw.replace(",", " ").split() if p]
    if not isinstance(raw, list):
        raw = [raw]
    return [str(x).strip().lstrip("#") for x in raw if str(x).strip()]


class DocxTool(Tool):
    """Create, read and edit Word documents with tracked changes."""

    name = "docx"
    timeout_seconds = 120.0
    description = (
        "Word documents (.docx): create, read and edit them properly — never write "
        "python-docx scripts or touch the XML yourself. "
        "create: a new .docx from Markdown (# headings, paragraphs, - and 1. lists, "
        "| tables |, > quotes, ``` code, ![caption](saved picture path), [TOC], "
        "\\pagebreak, **bold** *italic* [link](url)); optional template (.docx/.dotx "
        "whose styles, page setup, header and footer to use), page_numbers, fonts. "
        "read: numbered paragraphs, tracked changes inline as {+inserted+} [-deleted-], "
        "comments as ⟦anchor⟧cN — read before you edit. "
        "edit: `edits` = a list of changes, ALWAYS recorded as Word tracked changes "
        "under your name so the reviewer accepts or rejects each one in Word: "
        "replace (find → replace_with; only the words that differ are marked), "
        "delete (find, or whole paragraphs: paragraph [+ to]), insert (text after/before "
        "existing text), insert_paragraphs (Markdown content after/before paragraph N), "
        "format (bold/italic/underline/strike/highlight/color/size on find or paragraph), "
        "set_style (paragraph [+ to] → style), comment (text on find or paragraph). "
        "Track Changes is switched on in the saved file. A user's upload is never "
        "changed: the redlined copy is saved under saved/output with the same name. "
        "changes / comments: list them with ids. accept / reject: resolve tracked "
        "changes (all, ids, or by_author) — only when the user asks for it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS),
                       "description": "What to do."},
            "path": {
                "type": "string",
                "description": (
                    "The .docx file. For create, where to save it (a name like "
                    "'Offer.docx' lands in saved/; vfs:<project>/… for the shared folders)."
                ),
            },
            "content": {"type": "string",
                        "description": "create: the whole document as Markdown."},
            "edits": {
                "type": "array",
                "description": (
                    "edit: changes applied in order, all or nothing. Paragraph numbers "
                    "are those `read` showed (before this call); `find` is exact text "
                    "inside ONE paragraph as it reads now (unique unless occurrence/all)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": list(D.OPS)},
                        "find": {"type": "string",
                                 "description": "Exact existing text (replace/delete/format/comment)."},
                        "replace_with": {"type": "string", "description": "replace: the new text."},
                        "text": {"type": "string",
                                 "description": "insert: text to add; comment: the comment."},
                        "after": {"type": "string",
                                  "description": "insert: existing text to add after; "
                                                 "insert_paragraphs: paragraph number."},
                        "before": {"type": "string",
                                   "description": "insert: existing text to add before; "
                                                  "insert_paragraphs: paragraph number."},
                        "content": {"type": "string",
                                    "description": "insert_paragraphs: Markdown for the new paragraphs."},
                        "paragraph": {"type": "integer",
                                      "description": "Paragraph number: limits `find` to it, or is the "
                                                     "target of delete/format/set_style/comment."},
                        "to": {"type": "integer",
                               "description": "delete/set_style: last paragraph of a range."},
                        "occurrence": {"type": "integer",
                                       "description": "Which match when `find` occurs more than once (1 = first)."},
                        "all": {"type": "boolean", "description": "Apply to every match of `find`."},
                        "style": {"type": "string",
                                  "description": "set_style / insert_paragraphs: paragraph style name, e.g. 'Heading 2'."},
                        "bold": {"type": "boolean"},
                        "italic": {"type": "boolean"},
                        "underline": {"type": "boolean"},
                        "strike": {"type": "boolean"},
                        "highlight": {"type": "string",
                                      "description": "yellow, green, cyan, … or '' to remove."},
                        "color": {"type": "string", "description": "Hex colour, e.g. C00000."},
                        "size": {"type": "number", "description": "Font size in points."},
                    },
                    "required": ["op"],
                },
            },
            "output_path": {
                "type": "string",
                "description": "edit/accept/reject: save the result here instead (a new file).",
            },
            "author": {
                "type": "string",
                "description": "Name on tracked changes and comments (default: your own name). "
                               "Use the user's name only when they ask for that.",
            },
            "view": {"type": "string", "enum": ["markup", "final", "original"],
                     "description": "read: markup (default) shows changes; final as if all accepted; "
                                    "original as if all rejected."},
            "offset": {"type": "integer", "description": "read: first paragraph number to show."},
            "limit": {"type": "integer", "description": "read: how many paragraphs (default 200)."},
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "accept/reject: change ids from `changes`."},
            "by_author": {"type": "string", "description": "accept/reject: only this author's changes."},
            "all": {"type": "boolean", "description": "accept/reject: every tracked change."},
            "template": {"type": "string", "description": "create: a .docx/.dotx to take styles and layout from."},
            "title": {"type": "string", "description": "create: document title (file properties)."},
            "page_numbers": {"type": "boolean", "description": "create: 'Page X of Y' in the footer."},
            "font": {"type": "string", "description": "create: body font, e.g. Calibri."},
            "font_size": {"type": "number", "description": "create: body font size in points."},
            "heading_font": {"type": "string", "description": "create: heading font."},
            "heading_color": {"type": "string", "description": "create: heading colour, hex."},
            "overwrite": {"type": "boolean", "description": "create: replace an existing file."},
        },
        "required": ["action", "path"],
    }

    async def execute(self, action: str, path: str, **kwargs: Any) -> ToolResult:
        action = str(action or "").strip().lower()
        if action not in ACTIONS:
            return ToolResult(success=False, error=f"Unknown action {action!r}; use one of: {', '.join(ACTIONS)}.")
        try:
            if action == "create":
                return await self._create(path, kwargs)
            source = await self._source(path, kwargs, read_only=action in ("read", "changes", "comments"))
            if action == "read":
                return await asyncio.to_thread(self._read, source, kwargs)
            if action == "changes":
                return await asyncio.to_thread(self._changes, source)
            if action == "comments":
                return await asyncio.to_thread(self._comments, source)
            return await self._change(action, path, source, kwargs)
        except D.DocxError as exc:
            return ToolResult(success=False, error=_scrub(str(exc)))
        except Exception as exc:
            log.error("docx tool failed", action=action, path=path, error=str(exc))
            return ToolResult(success=False, error=_scrub(f"{type(exc).__name__}: {exc}"))

    # ── Paths ──────────────────────────────────────────────────────

    async def _source(self, path: str, kwargs: dict[str, Any], *, template: bool = False,
                      read_only: bool = False) -> Path:
        """An existing file: vfs- and Drive-aware like the extract tools, then
        the edit tool's fallbacks, then where `create` would have put it."""
        from captain_claw.tools.document_extract import _resolve_readable_file
        from captain_claw.tools.edit import EditTool, backup_of, backup_refused

        found, error = await _resolve_readable_file(path, kwargs.get("_runtime_base_path"))
        if found is None:
            found = EditTool._resolve_path(path, kwargs)
        if found is None and not is_vfs_path(path):
            candidate = WriteTool.resolve_local_target(path, kwargs)
            found = candidate if candidate.is_file() else None
        if found is None:
            raise D.DocxError(error or f"File not found: {path}")
        allowed = _WORD_SUFFIXES | ({".dotx", ".dotm"} if template else set())
        backup = backup_of(found)
        if backup and backup[1] in _WORD_SUFFIXES:
            # A backup the edits left: read as the document it copies, never changed.
            if read_only or template:
                return found
            raise D.DocxError(backup_refused(found, backup[0], "docx"))
        if found.suffix.lower() not in allowed:
            hint = " Convert a legacy .doc to .docx first." if found.suffix.lower() == ".doc" else ""
            raise D.DocxError(f"Not a .docx file: {found.name}.{hint}")
        return found

    @staticmethod
    def _target(path: str, kwargs: dict[str, Any]) -> Path:
        """Where a new file named *path* is written (create, output_path)."""
        if is_vfs_path(path):
            project = split_scheme(path)[0]
            if project_is_readonly(project):
                raise D.DocxError(f"'{project}' is a read-only folder — save somewhere else.")
            target = resolve_vfs_path(docx_file_name(path), create_parents=True)
            if target is None:
                raise D.DocxError(f"Invalid vfs path (escapes your folder): {path}")
        else:
            target = WriteTool.resolve_local_target(docx_file_name(path), kwargs)
        _writable(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _in_place(source: Path, kwargs: dict[str, Any]) -> bool:
        """May an edit change *source* itself? Only the agent's own files."""
        real = source.resolve()
        saved_root = WriteTool._resolve_saved_root(kwargs).resolve()
        try:
            rel = real.relative_to(saved_root)
            return bool(rel.parts) and rel.parts[0] not in _NOT_IN_PLACE \
                and not any(part.startswith(".") for part in rel.parts)
        except ValueError:
            pass
        try:
            from captain_claw.vfs import to_display, user_root

            real.relative_to(Path(user_root()).resolve())
            shown = to_display(real)
            return not project_is_readonly(split_scheme(shown)[0])
        except Exception:
            pass
        base = kwargs.get("_runtime_base_path")
        if os.environ.get("CLAW_WRITE_DIRECT") and base is not None:
            try:
                real.relative_to(Path(base).expanduser().resolve())
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def _member_may(source: Path) -> bool:
        from captain_claw import speaker

        return speaker.member_may_write(source)

    def _image_root(self, kwargs: dict[str, Any]):
        base = kwargs.get("_runtime_base_path")
        saved_root = WriteTool._resolve_saved_root(kwargs)

        def resolve(src: str) -> Path | None:
            raw = Path(src).expanduser()
            if is_vfs_path(src):
                vfs_target = resolve_vfs_path(src)
                options = [vfs_target] if vfs_target is not None else []
            elif raw.is_absolute():
                options = [raw]
            else:
                parts = list(raw.parts)
                if parts and parts[0] == "saved":
                    parts = parts[1:]
                options = [Path(base) / raw if base is not None else raw]
                if parts:
                    options.append(saved_root.joinpath(*parts))
            found = next((o for o in options if o.is_file()), None)
            if found is not None and _member() is not None:
                from captain_claw import speaker

                # A member's document may only carry pictures they can read.
                return found if speaker.path_allowed(found) else None
            return found

        return resolve

    # ── Saving ─────────────────────────────────────────────────────

    @staticmethod
    def _save(doc: Any, target: Path) -> int:
        """Write atomically, then reopen to be sure Word-side readers can."""
        from captain_claw import saved_attribution

        buffer = io.BytesIO()
        doc.save(buffer)
        data = buffer.getvalue()
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            D.open_document(tmp)  # the file must open again before it replaces anything
            if target.exists():
                try:
                    os.chmod(tmp, target.stat().st_mode & 0o777)
                except OSError:
                    pass
            prior = saved_attribution.prior_creator(target)
            os.replace(tmp, target)
            saved_attribution.note_write(target, prior)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        from captain_claw.vfs import record_author

        record_author(target)
        return len(data)

    @staticmethod
    def _register(logical: str, target: Path, kwargs: dict[str, Any]) -> None:
        registry = kwargs.get("_file_registry")
        if registry is None:
            return
        try:
            registry.register(logical_path=logical, physical_path=str(target),
                              task_id=str(kwargs.get("_task_id", "")))
        except Exception:
            pass

    # ── Actions ────────────────────────────────────────────────────

    async def _create(self, path: str, kwargs: dict[str, Any]) -> ToolResult:
        content = kwargs.get("content")
        if not isinstance(content, str) or not content.strip():
            raise D.DocxError("create needs `content`: the document as Markdown.")
        target = self._target(path, kwargs)
        if target.exists() and not kwargs.get("overwrite"):
            raise D.DocxError(
                f"{target.name} already exists. To change it, use action=edit (tracked changes); "
                "to replace it with a new document, pass overwrite=true.")
        template = None
        if kwargs.get("template"):
            template = await self._source(str(kwargs["template"]), kwargs, template=True)
        look = D.Look(
            font=str(kwargs.get("font") or ""),
            font_size=float(kwargs.get("font_size") or 0),
            heading_font=str(kwargs.get("heading_font") or ""),
            heading_color=str(kwargs.get("heading_color") or ""),
        )
        author = _author(kwargs)

        def build() -> tuple[int, list[str], dict[str, Any]]:
            doc, notes = D.build_document(
                content, template=template, title=str(kwargs.get("title") or ""),
                author=author, look=look, page_numbers=bool(kwargs.get("page_numbers")),
                image_root=self._image_root(kwargs))
            size = self._save(doc, target)
            return size, notes, D.summary(D.open_document(target))

        size, notes, info = await asyncio.to_thread(build)
        self._register(path, target, kwargs)
        lines = [f"Created {_show(target)} ({size:,} bytes, {info['paragraphs']} paragraphs, "
                 f"{info['tables']} tables)."]
        lines += [f"Note: {n}" for n in notes]
        return ToolResult(
            success=True, content="\n".join(lines),
            system_hint=("The document is saved and complete. To change it later use "
                         "docx edit (tracked changes) or create with overwrite=true for a "
                         "full rewrite — never a script."))

    def _read(self, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        doc = D.open_document(source)
        view = str(kwargs.get("view") or "markup").lower()
        if view not in ("markup", "final", "original"):
            view = "markup"
        offset = int(kwargs.get("offset") or 1)
        limit = max(1, min(int(kwargs.get("limit") or 200), 1000))
        body, next_at = D.render(doc, view=view, offset=offset, limit=limit)
        info = D.summary(doc)
        authors = ", ".join(f"{a}: {n}" for a, n in info["by_author"].items())
        head = [
            f"{source.name} — {info['paragraphs']} paragraphs, {info['tables']} tables, "
            f"{info['sections']} section(s). Empty paragraphs are not shown; numbers stay.",
            f"Track Changes: {'ON' if info['tracking'] else 'off'} · {info['changes']} tracked "
            f"change(s){f' ({authors})' if authors else ''} · {info['comments']} comment(s)",
        ]
        if view == "markup":
            head.append("Markup: {+inserted+} [-deleted-] ¶+ new paragraph ¶− paragraph "
                        "break removed ⟦…⟧cN commented text.")
        else:
            head.append(f"View: {view} (as if every change were "
                        f"{'accepted' if view == 'final' else 'rejected'}).")
        head += D.headers_and_footers(doc)
        tail = f"\n\n… more: read again with offset={next_at}." if next_at else ""
        return ToolResult(success=True, content="\n".join(head) + "\n\n" + body + tail)

    def _changes(self, source: Path) -> ToolResult:
        doc = D.open_document(source)
        changes = D.list_changes(doc)
        if not changes:
            return ToolResult(success=True, content=f"{source.name} has no tracked changes.")
        lines = [f"{source.name} — {len(changes)} tracked change(s):"]
        for c in changes[:300]:
            text = (c.text[:160] + "…") if len(c.text) > 160 else c.text
            when = c.date.replace("T", " ").rstrip("Z")[:16]
            lines.append(f"#{c.id} [{c.paragraph}] {c.kind} by {c.author or '?'} {when}"
                         + (f": {text!r}" if text else ""))
        if len(changes) > 300:
            lines.append(f"… and {len(changes) - 300} more.")
        return ToolResult(success=True, content="\n".join(lines))

    def _comments(self, source: Path) -> ToolResult:
        doc = D.open_document(source)
        comments = D.list_comments(doc)
        if not comments:
            return ToolResult(success=True, content=f"{source.name} has no comments.")
        lines = [f"{source.name} — {len(comments)} comment(s):"]
        for c in comments:
            anchor = (c.anchor[:100] + "…") if len(c.anchor) > 100 else c.anchor
            lines.append(f"c{c.id} [{c.paragraph}] {c.author} {c.date}: {c.text}"
                         + (f"  (on {anchor!r})" if anchor else ""))
        return ToolResult(success=True, content="\n".join(lines))

    async def _change(self, action: str, path: str, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        from captain_claw.tools.edit import EditTool

        output_path = str(kwargs.get("output_path") or "").strip()
        if output_path:
            target = self._target(output_path, kwargs)
        elif self._in_place(source, kwargs) and (_member() is None or self._member_may(source)):
            target = source
        else:
            # saved/output/<session>/<same name> (or the run's / repo's folder).
            target = WriteTool.resolve_local_target(f"output/{source.name}", kwargs)
            if target.exists() and target.resolve() != source.resolve():
                raise D.DocxError(
                    f"A redlined copy of {source.name} already exists at {_show(target)}. To "
                    "add to that redline, run this on that path; to start a separate copy "
                    "from the original, pass output_path.")
            _writable(target)
            target.parent.mkdir(parents=True, exist_ok=True)
        author = _author(kwargs)

        if action == "edit":
            edits = _edits_arg(kwargs.get("edits"))

            def run() -> tuple[D.Reviser, D.EditReport, Any]:
                doc = D.open_document(source)
                rv = D.Reviser.for_document(doc, author)
                report = D.apply_edits(doc, edits, rv, image_root=self._image_root(kwargs))
                D.set_tracking(doc, True)
                return rv, report, doc
        else:
            ids = _ids_arg(kwargs.get("ids"))
            who = str(kwargs.get("by_author") or "").strip()
            if not ids and not who and not kwargs.get("all"):
                raise D.DocxError(f"{action}: say which changes — ids, by_author, or all=true.")

            def run() -> tuple[dict[str, int], None, Any]:
                doc = D.open_document(source)
                if ids:
                    known = {c.id for c in D.list_changes(doc)}
                    missing = [i for i in ids if i not in known]
                    if missing:
                        raise D.DocxError(f"No tracked change with id {', '.join(missing)} — "
                                          "list them with action=changes.")
                done = D.resolve(doc, accept=action == "accept", ids=ids, author=who)
                return done, None, doc

        result, report, doc = await asyncio.to_thread(run)
        backup = ""
        if target == source and _member() is None:
            # Never for a member: backups share one folder per file name (edit.py).
            backup = await asyncio.to_thread(EditTool()._create_backup, source, kwargs) or ""
        await asyncio.to_thread(self._save, doc, target)
        if target != source:
            self._register(path, target, kwargs)

        where = (f"Saved in place: {_show(target)}" if target == source
                 else f"Saved: {_show(target)} (the original {source.name} is unchanged)")
        if action == "edit":
            counts = {k: v for k, v in result.counts.items() if v}
            summary = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()) or "no changes"
            lines = [f"Edited {target.name} as tracked changes by {result.author}: {summary}. "
                     "Track Changes is on.", where]
            if backup:
                lines.append(f"Backup: {backup}")
            lines += [f"Note: {n}" for n in report.notes]
            lines.append("Changed paragraphs now read:")
            numbering = {p: i for i, p in enumerate(D.paragraphs(doc), 1)}
            label = D.labeler(doc)
            seen: set[int] = set()
            for el in report.touched:
                paras = [el] if el.tag == D.P else list(el.iter(D.P))
                for p in paras:
                    n = numbering.get(p)
                    if n is None or n in seen:
                        continue
                    seen.add(n)
                    text = D.render_paragraph(p, "markup")
                    text = (text[:_PREVIEW_CHARS] + "…") if len(text) > _PREVIEW_CHARS else text
                    lines.append(f"[{n}] {label(p)}{text}")
            hint = ("The changes are saved. Don't re-read the whole document to check — "
                    "the lines above are the result. Give the user the saved file.")
        else:
            verb = "Accepted" if action == "accept" else "Rejected"
            count = next(iter(result.values()))
            left = len(D.list_changes(doc))
            lines = [f"{verb} {count} tracked change(s) in {target.name}; {left} left.", where]
            if backup:
                lines.append(f"Backup: {backup}")
            hint = None
        return ToolResult(success=True, content="\n".join(lines), system_hint=hint)
