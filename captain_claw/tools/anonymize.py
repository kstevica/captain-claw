"""`anonymize` tool — take private data out of a file, a page or a text.

Flight Deck finds names, dates of birth, bank accounts, emails, addresses,
phone numbers, ID numbers, companies … and replaces each with a token
(``⟦PERSON_1⟧``, ``⟦IBAN_2⟧``) that means nothing outside Captain. The real
values stay in an encrypted vault in Flight Deck that belongs to the person;
only they can see it (Flight Deck → the vault), and ``restore`` writes a file
with the real values put back — for them, never into this agent's context.

**The original never passes through the model.** For ``path`` and ``url`` this
tool reads the file or page itself (Captain code, not the model) and sends the
content straight to Flight Deck; only the anonymized result comes back. Text
given inline (``text``) has already been seen by the model — the tool still
works, and says so.

Plan and the rest of the design: docs/anonymization-plan.md.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)

INLINE_MAX = 8000
PREVIEW_CHARS = 3000
MAX_FILE_BYTES = 25 * 1024 * 1024

_TEXT_SUFFIXES = {".txt", ".text", ".log", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".eml", ".rtf", ".srt", ".vtt"}
_FORMAT_BY_SUFFIX = {".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
                     ".md": "md", ".markdown": "md"}
_EXTRACT_SUFFIXES = {".docx", ".xlsx", ".pptx", ".pdf", ".html", ".htm"}


def _session_folder(kwargs: dict[str, Any]) -> str:
    sid = re.sub(r"[^A-Za-z0-9_\-]+", "-", str(kwargs.get("_session_id") or "")).strip("-")
    return sid[:48] or "shared"


def default_vault(kwargs: dict[str, Any]) -> str:
    """One vault per conversation, so two conversations are never linked
    unless the agent names a case."""
    sid = str(kwargs.get("_session_id") or "").strip()
    return f"chat-{sid[:40]}" if sid else "default"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


class AnonymizeTool(Tool):
    name = "anonymize"
    timeout_seconds = 240.0
    description = (
        "Take PRIVATE DATA out of a file, web page or text: people's names, dates of birth, bank accounts "
        "(IBAN), card numbers, emails, phone numbers, street addresses, OIB/JMBG/passport/tax numbers, IPs, "
        "profile links, plates, company names. Each is replaced by a token like ⟦PERSON_1⟧ or ⟦IBAN_2⟧ that "
        "means nothing outside Captain; the same person gets the same token in every document of the same "
        "vault. The real values are kept encrypted in Flight Deck for the person — you never see them.\n"
        "Use it when the user asks to anonymize, pseudonymize, redact or de-identify something, to remove "
        "personal data / PII / GDPR data, or before you send a document's content somewhere it shouldn't "
        "carry names and numbers (HR: anonimiziraj, ukloni osobne podatke).\n"
        "action='anonymize' — give ONE of: `path` (a file: txt, md, csv, tsv, json, jsonl, html, docx, xlsx, "
        "pptx, pdf — office and PDF files come back as markdown), `url` (a page), or `text`. Prefer path/url: "
        "the tool reads them itself, so the original never reaches your model; `text` you have already seen. "
        "A file or page result is saved as a new file (path in the result). `vault` names the case (default: "
        "this conversation's); use the same vault for documents that belong together. mode='one_way' keeps "
        "no vault (real anonymization — nothing can be put back). `keep`: values to leave as they are (the "
        "user's own company, public figures). `extra`: terms to always replace, e.g. "
        "[{\"text\": \"Project Falcon\", \"kind\": \"PROJECT\"}] or a name the tool missed.\n"
        "action='restore' — put the real values back into an anonymized file (`path`) or `text` from the "
        "same `vault`; the result is written to a file for the user and is NOT shown to you. Never read "
        "a restored file back (that sends the real values to your model) — give the user its path or share "
        "it. action='vaults' lists the person's vaults (names and counts only).\n"
        "Work with the tokens as they are: keep them exactly (⟦PERSON_1⟧) in anything you write, so restore "
        "can put the values back. Names with nothing around them that marks them as names (a bare "
        "'we met Marko') can be missed — check the result and pass misses in `extra`."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["anonymize", "restore", "vaults"],
                "description": "anonymize — take private data out; restore — put it back into a file for "
                               "the user; vaults — list the person's vaults.",
            },
            "path": {
                "type": "string",
                "description": "A file: a path as you know it (workspace, saved/…, vfs:<project>/…). Read "
                               "by the tool, never shown to you.",
            },
            "url": {"type": "string", "description": "anonymize: a web page to fetch and anonymize."},
            "text": {"type": "string", "description": "Inline text (you have already seen it — prefer path/url)."},
            "vault": {
                "type": "string",
                "description": "The case this belongs to (any short name). Default: this conversation's vault. "
                               "restore needs the vault the text was anonymized into.",
            },
            "mode": {
                "type": "string",
                "enum": ["reversible", "one_way"],
                "description": "reversible (default) keeps the real values in the vault; one_way keeps nothing.",
            },
            "keep": {
                "type": "array", "items": {"type": "string"},
                "description": "anonymize: values never to replace.",
            },
            "extra": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "kind": {"type": "string"}},
                    "required": ["text"],
                },
                "description": "anonymize: terms to always replace; kind like PERSON, ORG, PROJECT (default PERSON).",
            },
            "output_path": {
                "type": "string",
                "description": "Where to write the result file (default: saved/anonymized/… or saved/restored/…).",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        from captain_claw.fd_client import flight_deck_base, flight_deck_headers

        base = flight_deck_base()
        headers = flight_deck_headers()
        if not base or not headers.get("X-Agent-Auth"):
            return ToolResult(success=False, error=(
                "Anonymization runs in Flight Deck: this agent wasn't started by a Flight Deck, so it can't "
                "anonymize anything. Don't paste the data elsewhere instead — tell the user."))
        try:
            if action == "anonymize":
                return await self._anonymize(base, headers, kwargs)
            if action == "restore":
                return await self._restore(base, headers, kwargs)
            if action == "vaults":
                return await self._vaults(base, headers)
            return ToolResult(success=False, error=f"Unknown action '{action}' — use anonymize, restore or vaults.")
        except httpx.HTTPError as exc:
            log.warning("anonymize: Flight Deck unreachable", action=action, error=type(exc).__name__)
            return ToolResult(success=False, error=f"Couldn't reach Flight Deck: {type(exc).__name__}")

    # ── sources ─────────────────────────────────────────────────────

    def _read_file(self, value: str, kwargs: dict[str, Any]) -> tuple[str, str, Path] | str:
        """(content, format, resolved path) — or why not."""
        from captain_claw import mail_attachments
        from captain_claw.tools import document_extract as dx

        resolved = mail_attachments.resolve_path(value, kwargs)
        if resolved is None:
            return f"No file '{value}' found. Pass the path you know it by (workspace, saved/…, vfs:…)."
        why = mail_attachments.credential_reason(resolved, kwargs)
        if why:
            return f"Not read: {why}. Anonymization is for personal data, not keys."
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            return f"Can't read {resolved.name}: {type(exc).__name__}"
        if size > MAX_FILE_BYTES:
            return f"{resolved.name} is over {MAX_FILE_BYTES // (1024 * 1024)} MB — split it first."
        suffix = resolved.suffix.lower()
        if suffix in _EXTRACT_SUFFIXES:
            try:
                if suffix == ".docx":
                    content = dx._extract_docx_markdown(resolved)
                elif suffix == ".xlsx":
                    content = dx._extract_xlsx_markdown(resolved, 100000)
                elif suffix == ".pptx":
                    content = dx._extract_pptx_markdown(resolved, 2000)
                elif suffix == ".pdf":
                    content, err = dx._extract_pdf_markdown(resolved, 5000)
                    if err or content is None:
                        return f"Can't read {resolved.name}: {err or 'no text in it'}"
                else:
                    from captain_claw.tools.web_fetch import _extract_readable_text

                    content = _extract_readable_text(_decode(resolved.read_bytes()))
            except Exception as exc:  # noqa: BLE001 — a broken document is the user's to hear about
                return f"Can't read {resolved.name}: {type(exc).__name__}"
            return content, "md", resolved
        data = resolved.read_bytes()
        if b"\x00" in data[:4096]:
            return f"{resolved.name} isn't a text or document file this tool can read."
        fmt = _FORMAT_BY_SUFFIX.get(suffix, "text")
        if suffix not in _FORMAT_BY_SUFFIX and suffix not in _TEXT_SUFFIXES and suffix:
            fmt = "text"
        return _decode(data), fmt, resolved

    @staticmethod
    async def _fetch(url: str) -> tuple[str, str] | str:
        from captain_claw.tools.web_fetch import _extract_readable_text, _make_http_client

        if not re.match(r"^https?://", url, re.I):
            return "url must start with http:// or https://"
        from captain_claw import deck_confine

        async with _make_http_client() as client:
            resp = await client.get(url)
        if deck_confine.tier() == "user" and not deck_confine.url_public(str(resp.url)):
            # The address was checked before the call; a redirect must not
            # land on this server or a private network either.
            return "That page redirects to a private address — not fetched."
        if resp.status_code >= 400:
            return f"The page answered {resp.status_code}."
        ctype = resp.headers.get("content-type", "")
        if "html" in ctype or resp.text.lstrip()[:1] == "<":
            return _extract_readable_text(resp.text, str(resp.url)), "md"
        if "json" in ctype:
            return resp.text, "json"
        if "csv" in ctype:
            return resp.text, "csv"
        return resp.text, "text"

    # ── actions ─────────────────────────────────────────────────────

    async def _anonymize(self, base: str, headers: dict, kwargs: dict) -> ToolResult:
        given = [k for k in ("path", "url", "text") if str(kwargs.get(k) or "").strip()]
        if len(given) != 1:
            return ToolResult(success=False, error="Give exactly one of `path`, `url` or `text`.")
        source = given[0]
        stem, out_suffix = "", ".txt"
        if source == "path":
            got = self._read_file(str(kwargs["path"]).strip(), kwargs)
            if isinstance(got, str):
                return ToolResult(success=False, error=got)
            content, fmt, resolved = got
            stem = resolved.stem
            out_suffix = {"md": ".md", "csv": ".csv", "tsv": ".tsv", "json": ".json", "jsonl": ".jsonl"}.get(
                fmt, resolved.suffix if resolved.suffix.lower() in _TEXT_SUFFIXES else ".txt")
            if resolved.suffix.lower() == ".md":
                out_suffix = ".md"
        elif source == "url":
            got_url = await self._fetch(str(kwargs["url"]).strip())
            if isinstance(got_url, str):
                return ToolResult(success=False, error=got_url)
            content, fmt = got_url
            stem = "page"
            out_suffix = {"md": ".md", "json": ".json", "csv": ".csv"}.get(fmt, ".txt")
        else:
            content, fmt = str(kwargs["text"]), "text"

        vault = str(kwargs.get("vault") or "").strip() or default_vault(kwargs)
        body: dict[str, Any] = {
            "content": content, "format": fmt, "vault": vault, "mode": kwargs.get("mode") or "reversible",
            "source": source, "filename": stem,
        }
        for key in ("keep", "extra"):
            if kwargs.get(key):
                body[key] = kwargs[key]
        data, err = await self._post(base, headers, "/fd/anonymize/agent/run", body)
        if err:
            return ToolResult(success=False, error=f"Not anonymized: {err}")

        output = str(data.get("output") or "")
        saved = ""
        if source != "text" or len(output) > INLINE_MAX:
            name = (data.get("filename") or "anonymized") + ".anonymized" + out_suffix
            target, werr = self._target(kwargs, "anonymized", name)
            if werr:
                return ToolResult(success=False, error=werr)
            self._write(target, output, kwargs)
            saved = str(target)

        lines = [self._headline(data)]
        lines.extend(self._entity_lines(data))
        if data.get("already_exposed"):
            lines.append("Note: this text came from your context, so your model had already seen the original. "
                         "Next time pass `path` or `url` and the tool reads it without you seeing it.")
        if saved:
            lines.append(f"Saved: {saved}")
        if len(output) <= INLINE_MAX:
            lines += ["", "--- anonymized ---", output]
        else:
            lines += ["", f"--- first {PREVIEW_CHARS} characters (the whole result is in the file) ---",
                      output[:PREVIEW_CHARS]]
        return ToolResult(success=True, content="\n".join(lines))

    async def _restore(self, base: str, headers: dict, kwargs: dict) -> ToolResult:
        from captain_claw import mail_authority

        authority = mail_authority.current()
        if authority.mode == "automated":
            return ToolResult(success=False, error=(
                f"Not restored: this is an automated turn ({authority.kind}). Real values are put back only "
                "when a person asks — the user can restore from Flight Deck."))
        given = [k for k in ("path", "text") if str(kwargs.get(k) or "").strip()]
        if len(given) != 1:
            return ToolResult(success=False, error="Give exactly one of `path` (an anonymized file) or `text`.")
        stem, suffix, fmt = "restored", ".txt", "text"
        if given[0] == "path":
            got = self._read_file(str(kwargs["path"]).strip(), kwargs)
            if isinstance(got, str):
                return ToolResult(success=False, error=got)
            content, fmt, resolved = got
            stem = re.sub(r"\.anonymized$", "", resolved.stem)
            suffix = resolved.suffix or ".txt"
        else:
            content = str(kwargs["text"])
        vault = str(kwargs.get("vault") or "").strip() or default_vault(kwargs)
        data, err = await self._post(base, headers, "/fd/anonymize/agent/restore",
                                     {"content": content, "format": fmt, "vault": vault})
        if err:
            return ToolResult(success=False, error=f"Not restored: {err}")
        target, werr = self._target(kwargs, "restored", f"{stem}.restored{suffix}")
        if werr:
            return ToolResult(success=False, error=werr)
        self._write(target, str(data.get("output") or ""), kwargs)
        unknown = data.get("unknown") or []
        lines = [
            f"DONE — put back {data.get('restored', 0)} value(s) from vault \"{(data.get('vault') or {}).get('name', vault)}\". "
            f"The restored file is for the user: {target}",
            "Its content is NOT shown to you, and don't read it back (that would send the real values to your "
            "model). Give the user the path, or share the file if they ask.",
        ]
        if unknown:
            lines.append(f"Not in this vault (left as they are): {', '.join(unknown[:20])}"
                         + (" …" if len(unknown) > 20 else "") + " — maybe another vault?")
        return ToolResult(success=True, content="\n".join(lines))

    async def _vaults(self, base: str, headers: dict) -> ToolResult:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{base}/fd/anonymize/agent/vaults", headers=headers)
        if resp.status_code != 200:
            return ToolResult(success=False, error=self._detail(resp))
        vaults = (resp.json() or {}).get("vaults") or []
        if not vaults:
            return ToolResult(success=True, content="The person has no anonymization vaults yet.")
        lines = []
        for v in vaults:
            kinds = ", ".join(f"{n} {k}" for k, n in sorted((v.get("kinds") or {}).items()))
            used = time.strftime("%Y-%m-%d", time.gmtime(v.get("last_used_at") or 0))
            lines.append(f"• {v.get('name')} — {v.get('entity_count', 0)} values ({kinds or 'empty'}) · last used {used}")
        return ToolResult(success=True, content="\n".join(lines))

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _headline(data: dict) -> str:
        summary = data.get("summary") or {}
        what = ", ".join(f"{n} {k.lower().replace('_', ' ')}" for k, n in sorted(summary.items())) or "nothing found"
        vault = data.get("vault")
        where = f"vault \"{vault['name']}\"" if vault else "one-way (no vault kept)"
        fixed = (data.get("verify") or {}).get("fixed") or 0
        check = "checked" + (f", {fixed} more caught on the second look" if fixed else "")
        return f"Anonymized — {what}; {where}; {check}."

    @staticmethod
    def _entity_lines(data: dict) -> list[str]:
        ents = data.get("entities") or []
        if not ents:
            return []
        shown = ents[:60]
        line = "Tokens: " + "; ".join(
            f"{e['token']} {e['kind'].lower()} ×{e['count']}{' (new)' if e.get('new') else ''}" for e in shown)
        return [line + (f"; … {len(ents) - len(shown)} more" if len(ents) > len(shown) else "")]

    @staticmethod
    def _target(kwargs: dict, folder: str, name: str) -> tuple[Path, str]:
        name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("_") or "result.txt"
        explicit = str(kwargs.get("output_path") or "").strip()
        if explicit:
            from captain_claw.vfs import is_vfs_path, resolve_vfs_path

            if is_vfs_path(explicit):
                target = resolve_vfs_path(explicit)
                if target is None:
                    return Path(), f"Can't write to {explicit}."
                return Path(target), ""
            p = Path(explicit).expanduser()
            if not p.is_absolute():
                root = kwargs.get("_runtime_base_path") or Path.cwd()
                p = Path(root) / p
            return p, ""
        saved = kwargs.get("_saved_base_path")
        root = Path(saved) if saved else Path(kwargs.get("_runtime_base_path") or Path.cwd()) / "saved"
        target = root / folder / _session_folder(kwargs) / name
        n = 2
        while target.exists():
            target = target.with_name(f"{Path(name).stem}-{n}{Path(name).suffix}")
            n += 1
        return target, ""

    @staticmethod
    def _write(target: Path, text: str, kwargs: dict) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        registry = kwargs.get("_file_registry")
        if registry is not None:
            try:
                registry.register(logical_path=str(target), physical_path=str(target),
                                  task_id=str(kwargs.get("_task_id", "")))
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        if isinstance(detail, str) and detail:
            return detail
        return f"Flight Deck answered {resp.status_code}"

    async def _post(self, base: str, headers: dict, path: str, body: dict) -> tuple[dict, str]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds - 10) as client:
            resp = await client.post(f"{base}{path}", json=body, headers=headers)
        if resp.status_code == 200:
            return resp.json(), ""
        if resp.status_code == 404 and path.endswith("/run"):
            return {}, "this Flight Deck has no anonymization yet (update it)"
        return {}, self._detail(resp)
