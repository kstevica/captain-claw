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
FOLDER_WAIT_SECONDS = 200  # under the tool timeout; a folder job keeps going on Flight Deck after this
FOLDER_POLL_SECONDS = 2.0



def _member_auth() -> tuple[dict[str, str], dict[str, str]]:
    """(headers, params) a shared agent's member call adds — ``({}, {})``
    for the owner. Raises when a member's turn has no grant."""
    from captain_claw import speaker

    return speaker.grant_headers(), speaker.grant_params()


def _shown(path: Path) -> str:
    """A path as the person knows it (``vfs:…`` inside their VFS)."""
    try:
        from captain_claw.vfs import to_display

        shown = to_display(path)
        return str(shown) if shown else str(path)
    except Exception:  # noqa: BLE001
        return str(path)


def _session_folder(kwargs: dict[str, Any]) -> str:
    sid = re.sub(r"[^A-Za-z0-9_\-]+", "-", str(kwargs.get("_session_id") or "")).strip("-")
    return sid[:48] or "shared"


def default_vault(kwargs: dict[str, Any]) -> str:
    """One vault per conversation, so two conversations are never linked
    unless the agent names a case — or, while the person's privacy gate
    applies, the gate's vault, so its tokens and these mean the same."""
    from captain_claw import anonymize_gate_client

    agent = kwargs.get("_agent")
    if agent is not None and anonymize_gate_client.cached_active(agent):
        from captain_claw.anonymize.gate import GATE_VAULT

        return GATE_VAULT
    sid = str(kwargs.get("_session_id") or "").strip()
    return f"chat-{sid[:40]}" if sid else "default"


class AnonymizeTool(Tool):
    name = "anonymize"
    # A trusted model that looks too reads a long file chunk by chunk: on a
    # local model that is minutes (anonymize_gate_client.CALL_TIMEOUT).
    timeout_seconds = 1800.0
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
        "can put the values back.\n"
        "Lanes: code always looks (patterns, checksums, labels); when the deck has a TRUSTED model (local, or "
        "one the admin vouches for) it also looks — it catches bare names ('we met Marko') and descriptions "
        "that point to one person. Leave `lane` out to get both when available. If the result says the "
        "trusted model didn't run, bare names can be missed: check the result and pass misses in `extra`."
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
                               "by the tool, never shown to you. Or a VFS folder (vfs:<project>/<folder>/): every "
                               "file in it gets an anonymized copy in a new folder beside it.",
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
            "generalize": {
                "type": "boolean",
                "description": "anonymize with mode='one_way' only: make details coarser instead of tokens — date "
                               "of birth → year, address → city, ages and money amounts → ranges (data shared for "
                               "good, where an exact age or salary would point to one person).",
            },
            "lane": {
                "type": "string",
                "enum": ["deterministic", "llm", "both"],
                "description": "anonymize: who looks — deterministic (code only), llm (the trusted model only) or "
                               "both. Leave out: both when the deck has a trusted model, else code only.",
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
        # A shared agent's member: every call carries the turn's grant, so
        # Flight Deck files their data in THEIR vaults (never the owner's).
        try:
            _member_auth()
        except Exception:  # noqa: BLE001 — SpeakerGrantMissing / identity lost
            return ToolResult(success=False, error=(
                "Anonymization is only available while answering this member's own message."))
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

    def _read_bytes(self, value: str, kwargs: dict[str, Any]) -> tuple[bytes, Path] | str:
        """(the file's bytes, its resolved path) — or why not. Flight Deck
        decides what it can read (text, csv, json, docx, xlsx, pptx, pdf, html)."""
        from captain_claw import mail_attachments

        resolved = mail_attachments.resolve_path(value, kwargs)
        if resolved is None:
            return f"No file '{value}' found. Pass the path you know it by (workspace, saved/…, vfs:…)."
        why = mail_attachments.credential_reason(resolved, kwargs)
        if why:
            return f"Not read: {why}. Anonymization is for personal data, not keys."
        try:
            size = resolved.stat().st_size
            if size > MAX_FILE_BYTES:
                return f"{resolved.name} is over {MAX_FILE_BYTES // (1024 * 1024)} MB — split it first."
            return resolved.read_bytes(), resolved
        except OSError as exc:
            return f"Can't read {resolved.name}: {type(exc).__name__}"

    @staticmethod
    async def _fetch(url: str) -> tuple[str, str] | str:
        from captain_claw.tools.web_fetch import _extract_readable_text, _make_http_client

        if not re.match(r"^https?://", url, re.I):
            return "url must start with http:// or https://"
        from captain_claw import deck_confine, speaker

        if speaker.current() is not None:
            # A member's fetch: public addresses only, checked at every hop,
            # and a capped body — web_fetch's own member path.
            from captain_claw.tools.web_fetch import _member_fetch

            try:
                text, status, _truncated = await _member_fetch(url)
            except Exception as exc:  # noqa: BLE001 — refused address, network, HTTP error
                return f"Couldn't fetch that page on a shared agent ({type(exc).__name__})."
            if status >= 400:
                return f"The page answered {status}."
            if text.lstrip()[:1] == "<":
                return _extract_readable_text(text, url), "md"
            return text, "text"

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
        if source == "path":
            return await self._anonymize_file(base, headers, kwargs)
        stem, out_suffix = "", ".txt"
        if source == "url":
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
        for key in ("keep", "extra", "lane", "generalize"):
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
            lines.append(f"Saved: {_shown(Path(saved))}")
        if len(output) <= INLINE_MAX:
            lines += ["", "--- anonymized ---", output]
        else:
            lines += ["", f"--- first {PREVIEW_CHARS} characters (the whole result is in the file) ---",
                      output[:PREVIEW_CHARS]]
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _vfs_folder(value: str) -> tuple[str, str] | None:
        """(project, path) when *value* names a VFS folder, else None."""
        from captain_claw.vfs import is_vfs_path, resolve_vfs_path, split_scheme

        if not is_vfs_path(value):
            return None
        try:
            if not resolve_vfs_path(value).is_dir():
                return None
        except (PermissionError, ValueError, OSError):
            return None
        project, rel = split_scheme(value)
        return project, rel.strip("/")

    async def _anonymize_folder(self, base: str, headers: dict, kwargs: dict, project: str,
                                path: str) -> ToolResult:
        """Every file in a VFS folder, on Flight Deck (a job): the copies go in a
        new folder beside it (or the person's Anonymized folder when it's
        read-only), names in their paths tokenized too. Waits for the job."""
        import asyncio

        body = {**self._body_options(kwargs), "project": project, "path": path}
        job, err = await self._post(base, headers, "/fd/anonymize/agent/vfs/folder", body)
        if err:
            return ToolResult(success=False, error=f"Not anonymized: {err}")
        extra, params = _member_auth()
        deadline = time.monotonic() + FOLDER_WAIT_SECONDS
        async with httpx.AsyncClient(timeout=30.0) as client:
            while job.get("running") and time.monotonic() < deadline:
                await asyncio.sleep(FOLDER_POLL_SECONDS)
                resp = await client.get(f"{base}/fd/anonymize/agent/vfs/job", headers={**headers, **extra},
                                        params=params)
                if resp.status_code == 200:
                    job = resp.json() or job
        where = f"vfs:{project}/{path}".rstrip("/")
        if job.get("running"):
            return ToolResult(success=True, content=(
                f"Still anonymizing {where} on Flight Deck ({job.get('done', 0)} of {job.get('total', 0)} files so "
                "far). It carries on by itself; the copies appear in a new folder beside it when it's done."))
        dest = job.get("dest") or {}
        lines = [f"Anonymized {job.get('done', 0)} of {job.get('total', 0)} files in {where}."]
        if dest.get("project"):
            place = "/".join(p for p in (dest["project"], dest.get("path") or "") if p)
            lines.append(f"Copies: vfs:{place}/" + ("" if job.get("beside") else
                                                    " (in the person's own Anonymized folder — the source is read-only)"))
        kinds = ", ".join(f"{n} {k}" for k, n in sorted((job.get("entities") or {}).items()))
        if kinds:
            lines.append(f"Replaced: {kinds}.")
        if job.get("vault"):
            lines.append(f"Vault: {job['vault']}")
        if job.get("failed_count"):
            names = ", ".join(f["path"] for f in (job.get("failed") or [])[:5])
            lines.append(f"Couldn't anonymize {job['failed_count']} (no copy made): {names}")
        if job.get("skipped_count"):
            names = ", ".join(f["path"] for f in (job.get("skipped") or [])[:5])
            lines.append(f"Left out {job['skipped_count']} (pictures, archives, other kinds): {names}")
        if job.get("message"):
            lines.append(str(job["message"]))
        return ToolResult(success=bool(job.get("done")) or not job.get("total"), content="\n".join(lines))

    def _body_options(self, kwargs: dict) -> dict[str, Any]:
        body: dict[str, Any] = {"vault": str(kwargs.get("vault") or "").strip() or default_vault(kwargs),
                                "mode": kwargs.get("mode") or "reversible"}
        for key in ("keep", "extra", "lane", "generalize"):
            if kwargs.get(key):
                body[key] = kwargs[key]
        return body

    async def _anonymize_file(self, base: str, headers: dict, kwargs: dict) -> ToolResult:
        """The file as it is goes to Flight Deck and comes back as the same
        kind of file (Word, Excel, PowerPoint keep their formatting; PDF and
        web pages come back as markdown)."""
        import base64

        folder = self._vfs_folder(str(kwargs["path"]).strip())
        if folder is not None:
            return await self._anonymize_folder(base, headers, kwargs, *folder)
        got = self._read_bytes(str(kwargs["path"]).strip(), kwargs)
        if isinstance(got, str):
            return ToolResult(success=False, error=got)
        blob, resolved = got
        body = {**self._body_options(kwargs), "filename": resolved.name,
                "data": base64.b64encode(blob).decode("ascii"), "source": "file"}
        data, err = await self._post(base, headers, "/fd/anonymize/agent/file", body)
        if err:
            return ToolResult(success=False, error=f"Not anonymized: {err}")
        target, werr = self._target(kwargs, "anonymized", str(data.get("filename") or "document.anonymized"))
        if werr:
            return ToolResult(success=False, error=werr)
        self._write_bytes(target, base64.b64decode(data.get("data") or ""), kwargs)
        preview = str(data.get("preview") or "")
        lines = [self._headline(data), *self._entity_lines(data), f"Saved: {_shown(target)}"]
        lines += [f"Note: {w}" for w in data.get("warnings") or []]
        if preview:
            shown = preview[:INLINE_MAX] if len(preview) <= INLINE_MAX else preview[:PREVIEW_CHARS]
            label = "--- its text, anonymized ---" if len(preview) <= INLINE_MAX else (
                f"--- the first {PREVIEW_CHARS} characters of its text (the whole result is in the file) ---")
            lines += ["", label, shown]
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
        vault = str(kwargs.get("vault") or "").strip() or default_vault(kwargs)
        if given[0] == "path":
            import base64

            got = self._read_bytes(str(kwargs["path"]).strip(), kwargs)
            if isinstance(got, str):
                return ToolResult(success=False, error=got)
            blob, resolved = got
            data, err = await self._post(base, headers, "/fd/anonymize/agent/file/restore", {
                "filename": resolved.name, "data": base64.b64encode(blob).decode("ascii"), "vault": vault})
            if err:
                return ToolResult(success=False, error=f"Not restored: {err}")
            restored_bytes = base64.b64decode(data.get("data") or "")
            name = str(data.get("filename") or "document.restored")
        else:
            data, err = await self._post(base, headers, "/fd/anonymize/agent/restore",
                                         {"content": str(kwargs["text"]), "format": "text", "vault": vault})
            if err:
                return ToolResult(success=False, error=f"Not restored: {err}")
            restored_bytes = str(data.get("output") or "").encode("utf-8")
            name = "text.restored.txt"
        target, werr = self._target(kwargs, "restored", name)
        if werr:
            return ToolResult(success=False, error=werr)
        self._write_bytes(target, restored_bytes, kwargs)
        unknown = data.get("unknown") or []
        lines = [
            f"DONE — put back {data.get('restored', 0)} value(s) from vault \"{(data.get('vault') or {}).get('name', vault)}\". "
            f"The restored file is for the user: {_shown(target)}",
            "Its content is NOT shown to you, and don't read it back (that would send the real values to your "
            "model). Give the user the path, or share the file if they ask.",
        ]
        if unknown:
            lines.append(f"Not in this vault (left as they are): {', '.join(unknown[:20])}"
                         + (" …" if len(unknown) > 20 else "") + " — maybe another vault?")
        return ToolResult(success=True, content="\n".join(lines))

    async def _vaults(self, base: str, headers: dict) -> ToolResult:
        extra, params = _member_auth()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{base}/fd/anonymize/agent/vaults", headers={**headers, **extra}, params=params)
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
        lanes = data.get("lanes") or {}
        llm = lanes.get("llm")
        if llm == "ok":
            who = f"code + trusted model {lanes.get('llm_model', '')} ({lanes.get('llm_terms', 0)} more found by it)"
        elif llm == "failed":
            who = ("code only — the trusted model FAILED, so bare names and descriptions may be left: "
                   "tell the user, check the result")
        elif lanes.get("deterministic") == "off":
            who = "trusted model only"
        else:
            who = "code only (no trusted model on this deck — bare names can be missed)"
        return f"Anonymized — {what}; {where}; {who}; {check}."

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
        from captain_claw import speaker

        if speaker.current() is not None:
            # A member's results go to THEIR own folder, not the agent's
            # saved/ — a commons every member reads (a restored file holds
            # real values).
            from captain_claw.vfs import default_project, resolve_vfs_path

            try:
                resolved = resolve_vfs_path(f"vfs:{default_project()}/anonymize/{folder}/{name}")
            except PermissionError:
                resolved = None
            if resolved is None:
                return Path(), speaker.FILES_UNAVAILABLE_MESSAGE
            target = Path(resolved)
        else:
            saved = kwargs.get("_saved_base_path")
            root = Path(saved) if saved else Path(kwargs.get("_runtime_base_path") or Path.cwd()) / "saved"
            target = root / folder / _session_folder(kwargs) / name
        n = 2
        while target.exists():
            target = target.with_name(f"{Path(name).stem}-{n}{Path(name).suffix}")
            n += 1
        return target, ""

    @classmethod
    def _write(cls, target: Path, text: str, kwargs: dict) -> None:
        cls._write_bytes(target, text.encode("utf-8"), kwargs)

    @staticmethod
    def _write_bytes(target: Path, blob: bytes, kwargs: dict) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
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
        extra, params = _member_auth()
        async with httpx.AsyncClient(timeout=self.timeout_seconds - 10) as client:
            resp = await client.post(f"{base}{path}", json=body, headers={**headers, **extra}, params=params)
        if resp.status_code == 200:
            return resp.json(), ""
        if resp.status_code == 404 and path.endswith("/run"):
            return {}, "this Flight Deck has no anonymization yet (update it)"
        return {}, self._detail(resp)
