"""Anonymization routes — ``/fd/anonymize/*``.

The engine (:mod:`captain_claw.anonymize`) runs here, next to the vaults
(:mod:`anonymize_vault`), so agents and the person get the same tokens for the
same case and the real values never leave Flight Deck except back to the
person who owns them.

Agent routes (an agent's ``X-Agent-Auth``; the vault's person is the agent's
owner, or — on a shared agent's member turn — the member, never the owner):

    POST /agent/run       {content, format, vault, mode, kinds, keep, extra, filename, source}
    POST /agent/restore   {content, format, vault} → the restored text, for the tool to write
                          to a file (the tool never hands it to the model)
    GET  /agent/vaults    names and counts — never values

Person routes (signed in):

    GET    /kinds
    POST   /run, /restore           same bodies as the agent's
    GET    /vaults                  the person's vaults
    GET    /vaults/{vault}          with every real value (logged as a view)
    PATCH  /vaults/{vault}          {name?, expires_at? (unix seconds or null)}
    DELETE /vaults/{vault}
    GET    /vaults/{vault}/events

Request bodies are read by hand: a malformed body must never reach FastAPI's
validation handler, which logs and echoes the raw input — private data.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from captain_claw.anonymize import engine
from captain_claw.anonymize import formats
from captain_claw.anonymize import llm_lane
from captain_claw.anonymize import usage
from captain_claw.anonymize.detectors import KINDS
from captain_claw.flight_deck import anonymize_gate as gate_settings
from captain_claw.flight_deck import anonymize_ner as ner
from captain_claw.flight_deck import anonymize_trusted as trusted
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import _LOCAL_USER, _fd_auth_enabled, get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Where a long anonymization stands, for a caller that shows it (a data room's
# run): set it around anonymize_blob; it gets {"step": "reading" | "model" |
# "replacing" | "saving" | "writing", "chunk", "chunks"}, and the engine's
# passes while replacing. Nothing private goes through it.
PROGRESS: ContextVar[Callable[[dict[str, Any]], Awaitable[None]] | None] = ContextVar("anonymize_progress",
                                                                                      default=None)


async def _progress(**event: Any) -> None:
    hook = PROGRESS.get()
    if hook is None:
        return
    try:
        await hook(event)
    except Exception:  # noqa: BLE001 — showing progress never stops the work
        pass


# What the engine's pieces of text are, by the document they came from.
_UNITS = {"xlsx": "cells", "docx": "paragraphs", "pptx": "text boxes"}


def _engine_progress(fmt: str) -> Callable[[dict[str, Any]], None]:
    """A plain function the engine calls from its worker thread: it hands each
    tick to the event loop, which reports it through PROGRESS."""
    loop = asyncio.get_running_loop()
    unit = _UNITS.get(fmt, "parts")
    hook = PROGRESS.get()

    def tick(event: dict[str, Any]) -> None:
        async def send() -> None:
            if hook is not None:
                try:
                    await hook({"step": "replacing", "unit": unit, **event})
                except Exception:  # noqa: BLE001
                    pass

        loop.call_soon_threadsafe(lambda: loop.create_task(send()))

    return tick

router = APIRouter(prefix="/fd/anonymize", tags=["anonymize"])

MAX_CHARS = 2_000_000
MAX_SEGMENTS = 200_000
MAX_TERMS = 500
MAX_PINNED = 2000
_CUSTOM_KIND_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,30}$")

KIND_LABELS = {
    "PERSON": "People's names", "ORG": "Companies and organisations", "EMAIL": "Email addresses",
    "PHONE": "Phone numbers", "ADDRESS": "Street addresses", "DOB": "Dates of birth",
    "IBAN": "Bank accounts (IBAN)", "CARD": "Payment card numbers", "CVV": "Card security codes",
    "CARD_EXPIRY": "Card expiry dates", "OIB": "OIB", "JMBG": "JMBG", "ID_NUMBER": "ID, passport, tax numbers",
    "SSN": "US social security numbers", "IP": "IP addresses", "PROFILE_URL": "Social profile links",
    "PLATE": "Licence plates", "DETAIL": "Descriptions that point to one person (trusted model only)",
}
LANES = ("deterministic", "llm", "both")


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


def _str_list(value: Any, what: str, max_len: int = 300) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail=f"{what} must be a list")
    out = [str(v).strip() for v in value if str(v or "").strip()]
    if len(out) > MAX_TERMS or any(len(v) > max_len for v in out):
        raise HTTPException(status_code=422, detail=f"{what}: at most {MAX_TERMS} values of {max_len} characters")
    return out


def _options(data: dict[str, Any]) -> engine.Options:
    kinds_in = _str_list(data.get("kinds"), "kinds", 40)
    unknown = [k for k in kinds_in if k.upper() not in llm_lane.LLM_KINDS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown kinds: {', '.join(unknown)} (known: {', '.join(KINDS)})")
    extra: list[tuple[str, str]] = []
    raw_extra = data.get("extra")
    if raw_extra is not None:
        if not isinstance(raw_extra, list) or len(raw_extra) > MAX_TERMS:
            raise HTTPException(status_code=422, detail=f"extra must be a list of at most {MAX_TERMS} items")
        for item in raw_extra:
            if isinstance(item, str):
                text, kind = item, "PERSON"
            elif isinstance(item, dict):
                text, kind = str(item.get("text") or ""), str(item.get("kind") or "PERSON")
            else:
                continue
            kind = kind.strip().upper()
            if not _CUSTOM_KIND_RE.match(kind):
                raise HTTPException(status_code=422, detail="an extra term's kind is upper-case letters, digits and _")
            if text.strip() and len(text) <= 300:
                extra.append((text.strip(), kind))
    kinds = frozenset(k.upper() for k in kinds_in) | frozenset(k for _t, k in extra) if kinds_in else None
    cc = str(data.get("default_country_code") or "385").lstrip("+")
    if not re.fullmatch(r"\d{1,3}", cc):
        raise HTTPException(status_code=422, detail="default_country_code is 1–3 digits")
    gen_in = data.get("generalize")
    if gen_in is True:
        generalize = frozenset(engine.GENERALIZE_KINDS)
    elif isinstance(gen_in, list):
        generalize = frozenset(str(k).upper() for k in gen_in)
        if generalize - set(engine.GENERALIZE_KINDS):
            raise HTTPException(status_code=422, detail=f"generalize takes: {', '.join(engine.GENERALIZE_KINDS)}")
    elif gen_in in (None, False):
        generalize = frozenset()
    else:
        raise HTTPException(status_code=422, detail="generalize is true or a list of kinds")
    if generalize and str(data.get("mode") or "reversible") != "one_way":
        raise HTTPException(status_code=422, detail=(
            "generalize needs mode 'one_way': coarser values (a year, a city, a range) can't be put back"))
    # pinned: [{"text", "token"}] — a data room's leftovers, each given the
    # token its real value already has in the vault.
    pinned: list[tuple[str, str]] = []
    raw_pinned = data.get("pinned")
    if raw_pinned is not None:
        if not isinstance(raw_pinned, list) or len(raw_pinned) > MAX_PINNED:
            raise HTTPException(status_code=422, detail=f"pinned must be a list of at most {MAX_PINNED} items")
        for item in raw_pinned:
            text = str(item.get("text") or "").strip() if isinstance(item, dict) else ""
            token = str(item.get("token") or "") if isinstance(item, dict) else ""
            if text and len(text) <= 300 and engine.TOKEN_RE.fullmatch(token):
                pinned.append((text, token))
    # one_word_names=False: code never takes a lone first name or surname for a
    # known person (a data room's final sweep: it only looks for the vault's values).
    return engine.Options(kinds=kinds, keep=tuple(_str_list(data.get("keep"), "keep")), extra=tuple(extra),
                          default_cc=cc, generalize=generalize, one_word_names=data.get("one_word_names") is not False,
                          pinned=tuple(pinned))


def _content(data: dict[str, Any]) -> tuple[str, str]:
    content = data.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=422, detail="content (text) is required")
    if len(content) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"content is over {MAX_CHARS:,} characters — split it")
    fmt = str(data.get("format") or "text").lower()
    return content, fmt if fmt in formats.TEXT_FORMATS else "text"


def _prepare(content: str, fmt: str) -> formats.Prepared:
    try:
        prepared = formats.prepare(content, fmt)
    except formats.FormatError as exc:
        raise HTTPException(status_code=422, detail=f"{fmt}: {exc}") from None
    if len(prepared.segments) > MAX_SEGMENTS:
        raise HTTPException(status_code=413, detail=f"over {MAX_SEGMENTS:,} cells / values — split it")
    return prepared


def _store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, fd_crypto.CryptoUnavailable):
        return HTTPException(status_code=503, detail=f"Vaults can't be used: {exc}. mode 'one_way' still works.")
    return HTTPException(status_code=422, detail=str(exc))


# ── the work, shared by both route sets ─────────────────────────────

MAX_FILE_BYTES = 25 * 1024 * 1024
_TEXT_EXT = {".txt": "text", ".text": "text", ".log": "text", ".md": "md", ".markdown": "md", ".csv": "csv",
             ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".xml": "text",
             ".yaml": "text", ".yml": "text", ".eml": "text", ".ini": "text", ".srt": "text", ".vtt": "text"}
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")  # read with local OCR
FILE_KINDS = sorted({*_TEXT_EXT, ".html", ".htm", ".pdf", ".docx", ".xlsx", ".pptx", *_IMAGE_EXT})
PICTURE_MODES = ("keep", "strip")


@dataclasses.dataclass
class _Doc:
    """What gets anonymized: its segments, and how to write it back from a
    run's outputs and edits (text, or a file's bytes)."""
    format: str
    segments: list[engine.Segment]
    rebuild: Any  # (outputs, edits) -> str | bytes
    warnings: list[str] = dataclasses.field(default_factory=list)
    out_ext: str = ""


def _text_doc(content: str, fmt: str, *, out_ext: str = "", warnings: list[str] | None = None) -> _Doc:
    prepared = _prepare(content, fmt)
    return _Doc(prepared.format, prepared.segments, lambda outs, _eds: prepared.rebuild(outs),
                list(warnings or []), out_ext)


def _file_input(data: dict[str, Any]) -> tuple[str, bytes]:
    import base64
    import binascii

    filename = str(data.get("filename") or "").strip().replace("\\", "/").rsplit("/", 1)[-1][:255]
    raw = data.get("data")
    if not filename or not isinstance(raw, str):
        raise HTTPException(status_code=422, detail="filename and data (the file, base64) are required")
    if len(raw) > MAX_FILE_BYTES * 4 // 3 + 16:
        raise HTTPException(status_code=413, detail=f"the file is over {MAX_FILE_BYTES // (1024 * 1024)} MB")
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="data isn't valid base64") from None
    return filename, blob


async def _file_doc(filename: str, blob: bytes, *, scrub: bool, pictures: str = "keep") -> _Doc:
    from pathlib import Path

    from captain_claw.anonymize import office, scans

    ext = Path(filename).suffix.lower()
    kind = office.kind_of(filename)
    if kind:
        reader = await asyncio.to_thread(scans.picture_reader) if scrub and pictures == "strip" else None
        try:
            prep = await asyncio.to_thread(office.prepare, blob, kind, scrub=scrub, pictures=pictures,
                                           picture_reader=reader)
        except office.OfficeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if len(prep.segments) > MAX_SEGMENTS:
            raise HTTPException(status_code=413, detail=f"over {MAX_SEGMENTS:,} pieces of text — split it")
        warnings = list(prep.warnings)
        if scrub and pictures == "strip" and reader is None:
            warnings.append("No OCR engine is installed, so text in the pictures wasn't read before they "
                            "were removed.")
        return _Doc(kind, prep.segments, lambda _outs, eds: prep.rebuild(eds), warnings, ext)
    if ext == ".pdf":
        text, notes = await asyncio.to_thread(_pdf_text, blob)
        return _text_doc(text, "md", out_ext=".md", warnings=[
            "A PDF comes back as markdown text — its layout and pictures aren't kept (real PDF redaction isn't built).",
            *notes])
    if ext in _IMAGE_EXT:
        try:
            text = await asyncio.to_thread(scans.image_text, blob, ext)
        except scans.ScanError as exc:
            raise HTTPException(status_code=422, detail=f"Can't read the picture: {exc}") from None
        return _text_doc(text or "", "md", out_ext=".md", warnings=[
            "A picture comes back as the text read from it with OCR (local, no model), as markdown."])
    if ext in (".html", ".htm"):
        from captain_claw.tools.web_fetch import _extract_readable_text

        return _text_doc(_extract_readable_text(formats.decode_text(blob)), "md", out_ext=".md",
                         warnings=["A web page comes back as markdown text."])
    if ext in _TEXT_EXT:
        if b"\x00" in blob[:4096]:
            raise HTTPException(status_code=422, detail=f"{filename} isn't a text file")
        return _text_doc(formats.decode_text(blob), _TEXT_EXT[ext], out_ext=ext)
    raise HTTPException(status_code=415, detail=(
        f"Can't anonymize {ext or 'files without an extension'} — text, csv, json, docx, xlsx, pptx, pdf, html or a picture (png, jpg, tif …)."))


def _pdf_text(blob: bytes) -> tuple[str, list[str]]:
    """A PDF as markdown. Pages without a usable text layer (a scan) are read
    with local OCR when an engine is installed; the rest keep their text."""
    import tempfile
    from pathlib import Path

    from captain_claw.anonymize import scans
    from captain_claw.tools import document_extract as dx

    ocr = scans.available()
    if ocr and scans.pdf_needs_ocr(blob):
        try:
            text, pages = scans.pdf_text(blob)
        except scans.ScanError as exc:
            raise HTTPException(status_code=422, detail=f"Can't read the scanned PDF: {exc}") from None
        return text, [f"{pages} scanned page(s) were read with OCR (local, no model)."] if pages else []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "document.pdf"
        path.write_bytes(blob)
        content, err = dx._extract_pdf_markdown(path, 5000)
    if err or content is None:
        hint = "" if ocr else " (scanned pages need an OCR engine — none is installed)"
        raise HTTPException(status_code=422, detail=f"Can't read the PDF: {err or 'no text in it'}{hint}")
    return content, []


# ── token history ───────────────────────────────────────────────────


def _short(value: Any, limit: int = 120) -> str:
    return re.sub(r"[\x00-\x1f]+", " ", str(value or "")).strip()[:limit]


def _uses(entities: list[engine.Entity], texts: list[str], data: dict[str, Any],
          trusted_label: str = "") -> list[av.TokenUse]:
    """What a run did to its tokens: "added" for each new one (the finder
    that matched it), "out" for each one in the text, with the context of its
    first appearance in the output and the model it goes to (the agent's,
    for the privacy gate)."""
    contexts = usage.token_contexts(texts, {e.token for e in entities})
    to_model = _short(data.get("to_model"))
    out: list[av.TokenUse] = []
    for e in entities:
        count, context = contexts.get(e.token, (e.count, ""))
        if e.new:
            out.append(av.TokenUse(e.token, "added", 1, context, e.found_by, e.derived_from,
                                   trusted_label if e.found_by == "extra" else ""))
        out.append(av.TokenUse(e.token, "out", count or e.count or 1, context, model=to_model))
    return out


async def _record(row: dict[str, Any], uses: list[av.TokenUse], *, channel: str, actor: str,
                  data: dict[str, Any]) -> None:
    try:
        await av.store().record_uses(row, uses, channel=channel, actor=actor, label=_short(data.get("purpose")))
    except Exception as exc:  # noqa: BLE001 — the history never fails the work
        log.warning("Token history not recorded", error=type(exc).__name__)


def _lane(data: dict[str, Any], model: trusted.TrustedModel | None) -> str:
    lane = str(data.get("lane") or "").strip().lower() or ("both" if model else "deterministic")
    if lane not in LANES:
        raise HTTPException(status_code=422, detail="lane is 'deterministic', 'llm' or 'both'")
    if lane != "deterministic" and model is None:
        raise HTTPException(status_code=409, detail=(
            "No trusted model is set up for anonymization, so only the deterministic lane can run. "
            "An admin sets one (a local model, or a cloud one they attest) — or pass lane='deterministic'."))
    return lane


async def _anonymize(person: str, doc: _Doc, data: dict[str, Any], *, actor: str, source: str,
                     stem: str, one_way_state: engine.VaultState | None = None) -> dict[str, Any]:
    """Run *doc* through the lanes and the engine into the person's vault (or
    none, one_way). Returns the response fields every route shares, plus
    ``_result`` (the RunResult) for the caller to write the output.
    *one_way_state*: for one_way, an in-memory state several runs share (one
    job over a folder), so a person keeps one token across its files."""
    options = _options(data)
    mode = str(data.get("mode") or "reversible")
    if mode not in ("reversible", "one_way"):
        raise HTTPException(status_code=422, detail="mode is 'reversible' or 'one_way'")
    store = av.store()
    vault_row: dict[str, Any] | None = None
    model = await trusted.usable(get_db())
    lane = _lane(data, model)
    lanes: dict[str, Any] = {"deterministic": "off" if lane == "llm" else "ok",
                             "llm": "off" if lane == "deterministic" else "ok"}
    try:
        if mode == "reversible":
            vault_row = await store.find_or_create(person, str(data.get("vault") or "default"), created_by=actor)
        if lane != "deterministic":
            assert model is not None
            started = time.monotonic()
            try:
                found = await _llm_terms(doc.segments, options, model)
            except llm_lane.LaneFailed as exc:
                log.warning("Anonymization LLM lane failed", model=model.label, actor=actor, reason=str(exc))
                if lane == "llm" or model.on_failure != "deterministic":
                    raise HTTPException(status_code=502, detail=(
                        f"The trusted model ({model.label}) failed: {exc}. Nothing was returned — retry, or "
                        "pass lane='deterministic' to anonymize with code alone.")) from None
                lanes["llm"] = "failed"
                lanes["llm_error"] = str(exc)
            else:
                # With the trusted model looking, code doesn't take one word for a known person.
                options = dataclasses.replace(options, extra=options.extra + tuple(found.terms),
                                              deterministic=lane != "llm", one_word_names=False)
                lanes.update({"llm_model": model.label, "llm_terms": len(found.terms),
                              "llm_dropped": found.dropped, "llm_chunks": found.chunks,
                              "llm_seconds": round(time.monotonic() - started, 1)})
                if found.tokens and model.basis != "local":
                    from captain_claw.flight_deck import usage_quota

                    await usage_quota.record_tokens(get_db(), person, found.tokens)

        await _progress(step="replacing", pieces=len(doc.segments), unit=_UNITS.get(doc.format, "parts"))

        def work(state: engine.VaultState) -> tuple[engine.RunResult, str]:
            result = engine.run(doc.segments, state, options)
            if options.generalize:
                result = engine.generalize(doc.segments, result, state, options.generalize)
            name = engine.anonymize_filename(stem, state, options) if stem else ""
            return result, name

        # The engine's own progress (round, pass, n of N), from its worker thread
        # to whoever listens here — only when someone does.
        engine_token = engine.PROGRESS.set(_engine_progress(doc.format) if PROGRESS.get() is not None else None)
        # Cancelling this coroutine (a stopped room) stops the worker thread too.
        stop_event = threading.Event()
        cancel_token = engine.CANCEL.set(stop_event)
        try:
            if vault_row is None:
                result, out_name = await asyncio.to_thread(work, one_way_state if one_way_state is not None
                                                           else engine.VaultState())
            else:
                async with store.lock(vault_row["id"]):
                    state = await store.load_state(vault_row)
                    result, out_name = await asyncio.to_thread(work, state)
                    await _progress(step="saving")
                    await store.save_run(vault_row, state)
        except asyncio.CancelledError:
            stop_event.set()
            raise
        finally:
            engine.CANCEL.reset(cancel_token)
            engine.PROGRESS.reset(engine_token)
    except engine.VerifyFailed as exc:
        log.warning("Anonymization refused its own output", kinds=exc.kinds, actor=actor)
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (av.VaultError, fd_crypto.CryptoUnavailable) as exc:
        raise _store_error(exc) from None

    counts = engine.summary(result.used)
    if vault_row is not None:
        await _record(vault_row, _uses(result.used, list(result.outputs), data,
                                       str(lanes.get("llm_model") or "") if lanes.get("llm") == "ok" else ""),
                      channel=source or "text", actor=actor, data=data)
        await store.log(person, vault_row["id"], actor, "anonymize", {
            "format": doc.format, "source": source, "entities": counts,
            "new": sum(1 for e in result.used if e.new), "fixed": result.fixed, "lane": lane,
            "llm": lanes.get("llm"), "llm_terms": lanes.get("llm_terms", 0),
        })
    return {
        "vault": {"id": vault_row["id"], "name": vault_row["name"]} if vault_row else None,
        "mode": mode,
        "format": doc.format,
        "filename": out_name,
        "entities": [{"token": e.token, "kind": e.kind, "count": e.count, "new": e.new} for e in result.used],
        "summary": counts,
        "verify": {"ok": True, "rounds": result.rounds, "fixed": result.fixed},
        "lane": lane,
        "lanes": lanes,
        "generalized": sorted(options.generalize),
        "name_model": ner._state["model"] if detectors_ner_on() else "",
        "warnings": list(doc.warnings),
        "already_exposed": source == "text",
        "_result": result,
    }


async def run_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    content, fmt = _content(data)
    doc = _text_doc(content, fmt)
    out = await _anonymize(person, doc, data, actor=actor, source=str(data.get("source") or ""),
                           stem=str(data.get("filename") or "")[:255])
    result = out.pop("_result")
    out["output"] = doc.rebuild(result.outputs, result.edits)
    return out


def _preview(texts: list[str], limit: int = 20000) -> str:
    joined = "\n".join(t for t in texts if t.strip())
    return joined[:limit]


async def anonymize_blob(person: str, filename: str, blob: bytes, data: dict[str, Any], *, actor: str,
                         source: str, one_way_state: engine.VaultState | None = None) -> tuple[bytes, dict[str, Any]]:
    """A file's bytes in, the anonymized file's bytes out (pdf / html as
    markdown), plus the response fields; ``filename`` is the new file's name
    (its own name tokenized, then ``.anonymized``)."""
    from pathlib import Path

    if len(blob) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"the file is over {MAX_FILE_BYTES // (1024 * 1024)} MB")
    pictures = str(data.get("pictures") or "keep")
    if pictures not in PICTURE_MODES:
        raise HTTPException(status_code=422, detail=f"pictures must be one of {', '.join(PICTURE_MODES)}")
    await _progress(step="reading")
    doc = await _file_doc(filename, blob, scrub=True, pictures=pictures)
    out = await _anonymize(person, doc, data, actor=actor, source=source, stem=Path(filename).stem,
                           one_way_state=one_way_state)
    result = out.pop("_result")
    await _progress(step="writing")
    written = await asyncio.to_thread(doc.rebuild, result.outputs, result.edits)
    blob_out = written if isinstance(written, bytes) else written.encode("utf-8")
    out["filename"] = f"{out['filename'] or 'document'}.anonymized{doc.out_ext or Path(filename).suffix.lower()}"
    out["preview"] = _preview(result.outputs)
    return blob_out, out


async def anonymize_names(person: str, names: list[str], data: dict[str, Any], *, actor: str,
                          one_way_state: engine.VaultState | None = None) -> list[str]:
    """Folder names tokenized like file names, into the same vault (for
    one_way, *one_way_state* or nothing), so a copied folder tree doesn't
    carry names in its paths. Call it after the files ran: what they taught
    the vault is what finds a bare name ("Ivan Horvat/")."""
    options = _options(data)
    store = av.store()

    def work(state: engine.VaultState) -> list[str]:
        return [engine.anonymize_filename(n, state, options) or n for n in names]

    try:
        if str(data.get("mode") or "reversible") == "one_way":
            return await asyncio.to_thread(work, one_way_state if one_way_state is not None
                                           else engine.VaultState())
        row = await store.find_or_create(person, str(data.get("vault") or "default"), created_by=actor)
        async with store.lock(row["id"]):
            state = await store.load_state(row)
            out = await asyncio.to_thread(work, state)
            await store.save_run(row, state)
    except (av.VaultError, fd_crypto.CryptoUnavailable) as exc:
        raise _store_error(exc) from None
    await _record(row, _uses([e for e in state.entities if e.count], out, data), channel="names", actor=actor,
                  data=data)
    return out


async def file_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    """A whole file in, the same kind of file out (pdf / html as markdown)."""
    import base64
    import mimetypes

    filename, blob = _file_input(data)
    blob_out, out = await anonymize_blob(person, filename, blob, data, actor=actor,
                                         source=str(data.get("source") or "file"))
    name = out["filename"]
    out.update({
        "data": base64.b64encode(blob_out).decode("ascii"),
        "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
        "size": len(blob_out),
    })
    return out


async def _llm_terms(segments: list[engine.Segment], options: engine.Options,
                     model: trusted.TrustedModel) -> llm_lane.LaneResult:
    """What the trusted model finds. It reads the text as it is, in both lanes:
    shown code's output instead, it couldn't see a name code had guessed wrong
    ("Ivan Horvat" as two other people's tokens) — and names are its call."""
    texts = [s.text for s in segments]
    joined = "\n".join(t for t in texts if t.strip())

    async def chunk_done(done: int, total: int) -> None:
        await _progress(step="model", chunk=done, chunks=total)

    return await llm_lane.detect(joined, trusted.completer(model), kinds=options.kinds, on_progress=chunk_done)


async def _vault_state(person: str, data: dict[str, Any]) -> tuple[dict[str, Any], engine.VaultState]:
    name = str(data.get("vault") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="vault is required — the one the text was anonymized into")
    store = av.store()
    try:
        row = await store.get(person, name)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No vault '{name}'")
        return row, await store.load_state(row)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None


async def _restore_doc(person: str, doc: _Doc, data: dict[str, Any], *, actor: str) -> tuple[Any, dict[str, Any]]:
    row, state = await _vault_state(person, data)
    total = 0
    unknown: list[str] = []
    outs: list[str] = []
    all_edits: list[list[tuple[int, int, str]]] = []
    for seg in doc.segments:
        edits, n, unk = engine.restore_edits(seg.text, state)
        outs.append(engine.apply_edits(seg.text, edits))
        all_edits.append(edits)
        total += n
        unknown.extend(u for u in unk if u not in unknown)
    written = await asyncio.to_thread(doc.rebuild, outs, all_edits)
    store = av.store()
    to_model = _short(data.get("to_model"))
    back = usage.token_contexts([seg.text for seg in doc.segments], set(state.by_token))
    await _record(row, [av.TokenUse(t, "back", n, c, model=to_model) for t, (n, c) in back.items()],
                  channel=_short(data.get("source"), 60) or "restore", actor=actor, data=data)
    await store.touch(row)
    await store.log(person, row["id"], actor, "restore", {"restored": total, "unknown": len(unknown),
                                                          "format": doc.format})
    return written, {"vault": {"id": row["id"], "name": row["name"]}, "restored": total, "unknown": unknown}


async def restore_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    content, fmt = _content(data)
    written, out = await _restore_doc(person, _text_doc(content, fmt), data, actor=actor)
    out["output"] = written
    return out


async def restore_file_for(person: str, data: dict[str, Any], *, actor: str) -> dict[str, Any]:
    import base64
    import mimetypes
    from pathlib import Path

    filename, blob = _file_input(data)
    doc = await _file_doc(filename, blob, scrub=False)
    written, out = await _restore_doc(person, doc, data, actor=actor)
    blob_out = written if isinstance(written, bytes) else written.encode("utf-8")
    stem = re.sub(r"\.anonymized$", "", Path(filename).stem)
    name = f"{stem}.restored{doc.out_ext or Path(filename).suffix.lower()}"
    out.update({"filename": name, "data": base64.b64encode(blob_out).decode("ascii"),
                "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                "size": len(blob_out)})
    return out


# ── agent routes ────────────────────────────────────────────────────


def _agent_slug(request: Request) -> str:
    try:
        from captain_claw.flight_deck import server as srv

        matched, _owner, slug = srv._find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
        return str(slug) if matched and slug else ""
    except Exception:
        return ""


async def _agent_person(request: Request) -> tuple[str, str]:
    """``(person, actor)``: whose vaults this agent call uses — the member on a
    shared agent's member turn, else the agent's owner — resolved from this
    deck's own records, never from the agent's word."""
    from captain_claw.flight_deck import speaker_grants

    acting = await speaker_grants.acting_member(request)
    if acting is not None:
        return acting.user_id, f"agent:{acting.slug} (member)"
    if _fd_auth_enabled():
        from captain_claw.flight_deck import google_oauth_routes as _g

        _g._authorize_agent_call(request)
        owner = await _g._agent_owner(request, service="Anonymization")
        return owner, f"agent:{_agent_slug(request)}"
    from captain_claw.flight_deck.agent_identity import require_agent_caller

    caller = require_agent_caller(request, what="Anonymization")
    return _LOCAL_USER["id"], f"agent:{_agent_slug(request) or caller.slug}"


@router.post("/agent/run")
async def anonymize_agent_run(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await run_for(person, await _body(request), actor=actor)


@router.post("/agent/restore")
async def anonymize_agent_restore(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await restore_for(person, await _body(request), actor=actor)


@router.post("/agent/file")
async def anonymize_agent_file(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await file_for(person, await _body(request), actor=actor)


@router.post("/agent/file/restore")
async def anonymize_agent_file_restore(request: Request) -> dict[str, Any]:
    person, actor = await _agent_person(request)
    return await restore_file_for(person, await _body(request), actor=actor)


@router.get("/agent/status")
async def anonymize_agent_status(request: Request) -> dict[str, Any]:
    await _agent_person(request)
    return await _status()


@router.post("/agent/gate")
async def anonymize_agent_gate(request: Request) -> dict[str, Any]:
    """Does the person's privacy gate apply to this agent, on the model it
    runs on now? (The agent says which model; the person's setting and the
    deck's trust rules decide.)"""
    person, _actor = await _agent_person(request)
    data = await _body(request)
    return await gate_settings.decide(person, str(data.get("provider") or "")[:100],
                                      str(data.get("model") or "")[:200], str(data.get("base_url") or "")[:500])


@router.get("/agent/vaults")
async def anonymize_agent_vaults(request: Request) -> dict[str, Any]:
    person, _actor = await _agent_person(request)
    return {"vaults": await av.store().list_vaults(person)}


# ── person routes ───────────────────────────────────────────────────


@router.get("/kinds")
async def anonymize_kinds(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {"kinds": [{"kind": k, "label": KIND_LABELS.get(k, k)} for k in KINDS], "formats": list(formats.TEXT_FORMATS),
            "file_kinds": FILE_KINDS, "max_file_mb": MAX_FILE_BYTES // (1024 * 1024)}


def detectors_ner_on() -> bool:
    from captain_claw.anonymize import detectors

    return detectors._ner_backend is not None


async def _status() -> dict[str, Any]:
    model = await trusted.usable(get_db())
    return {"llm": bool(model), "model": model.label if model else "",
            "default_lane": "both" if model else "deterministic",
            "name_model": ner._state["model"] if detectors_ner_on() else ""}


@router.get("/status")
async def anonymize_status(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _status()


@router.post("/run")
async def anonymize_run(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await run_for(user["id"], await _body(request), actor="person")


@router.post("/file")
async def anonymize_file(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await file_for(user["id"], await _body(request), actor="person")


@router.post("/file/restore")
async def anonymize_file_restore(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await restore_file_for(user["id"], await _body(request), actor="person")


@router.post("/restore")
async def anonymize_restore(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await restore_for(user["id"], await _body(request), actor="person")


@router.get("/vaults")
async def anonymize_vaults(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {"vaults": await av.store().list_vaults(user["id"])}


async def _own(user: dict, vault: str) -> dict[str, Any]:
    try:
        row = await av.store().get(user["id"], vault)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    if row is None:
        raise HTTPException(status_code=404, detail="No such vault")
    return row


@router.get("/vaults/{vault}")
async def anonymize_vault_detail(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    store = av.store()
    try:
        entries = await store.entries(row)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    await store.log(user["id"], row["id"], "person", "view", {"entities": len(entries)})
    return {"vault": av._public(row), "entries": entries}


@router.patch("/vaults/{vault}")
async def anonymize_vault_update(vault: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    data = await _body(request)
    expires: float | None | str = "keep"
    if "expires_at" in data:
        value = data["expires_at"]
        if value is not None and not isinstance(value, (int, float)):
            raise HTTPException(status_code=422, detail="expires_at is unix seconds, or null for never")
        expires = None if value is None else float(value)
    try:
        updated = await av.store().update(user["id"], row["id"], name=data.get("name"), expires_at=expires)
    except av.VaultError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if updated is None:
        raise HTTPException(status_code=404, detail="No such vault")
    await av.store().log(user["id"], row["id"], "person", "update",
                         {k: True for k in ("name", "expires_at") if k in data})
    return {"vault": av._public(updated)}


@router.delete("/vaults/{vault}")
async def anonymize_vault_delete(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    await av.store().delete(user["id"], row["id"], actor="person")
    return {"deleted": row["id"]}


@router.post("/vaults/{vault}/reset")
async def anonymize_vault_reset(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Empty the vault (values and their history), keep it and its token
    numbers — the Chat vault starts over without old tokens ever meaning
    something new."""
    row = await _own(user, vault)
    reset = await av.store().reset(user["id"], row["id"], actor="person")
    if reset is None:
        raise HTTPException(status_code=404, detail="No such vault")
    return {"vault": av._public(reset)}


@router.get("/vaults/{vault}/events")
async def anonymize_vault_events(vault: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    row = await _own(user, vault)
    return {"events": await av.store().events(user["id"], row["id"])}


@router.get("/vaults/{vault}/tokens/{token}")
async def anonymize_token_history(vault: str, token: str, limit: int = av.HISTORY_PAGE, before: int | None = None,
                                  user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """One token's history: what added it, and when it went out and came back
    (contexts as they were sent — tokens, no real values), newest first, a
    page at a time (*before*: the oldest line's ``id`` of the page shown)."""
    m = engine.TOKEN_RE.fullmatch(token.strip())
    if m is None:
        raise HTTPException(status_code=422, detail="token looks like ⟦PERSON_1⟧")
    row = await _own(user, vault)
    try:
        return await av.store().token_history(row, engine.make_token(m.group(1) or m.group(3),
                                                                     int(m.group(2) or m.group(4))),
                                              limit=limit, before=before)
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None


# ── admin: the trusted model ────────────────────────────────────────


@router.get("/admin/trusted-model")
async def anonymize_trusted_get(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    m = await trusted.configured(get_db())
    return {"model": m.public() if m else None, "providers": list(trusted.PROVIDERS)}


@router.put("/admin/trusted-model")
async def anonymize_trusted_put(request: Request, admin: dict = Depends(require_admin)) -> dict[str, Any]:
    try:
        m = await trusted.save(get_db(), await _body(request), admin=str(admin.get("id") or ""))
    except trusted.TrustError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except fd_crypto.CryptoUnavailable as exc:
        raise _store_error(exc) from None
    log.info("Anonymization trusted model set", model=m.label, basis=m.basis, admin=admin.get("id"))
    return {"model": m.public()}


@router.delete("/admin/trusted-model")
async def anonymize_trusted_delete(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    await trusted.clear(get_db())
    m = await trusted.configured(get_db())  # the environment's, if any
    return {"model": m.public() if m else None}


@router.post("/admin/trusted-model/test")
async def anonymize_trusted_test(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    """Run the built-in sample cases (synthetic people — the same ones the
    tests hold every lane to) through the code alone and through code + the
    trusted model, and say what each left behind."""
    m = await trusted.configured(get_db())
    if m is None:
        raise HTTPException(status_code=404, detail="No trusted model is set")
    if not m.usable:
        raise HTTPException(status_code=409, detail="The trusted model is switched off or not trusted")
    out = []
    for case in engine_cases():
        prepared = formats.prepare(case["text"], "text")
        det, _r = await asyncio.to_thread(engine.anonymize_text, case["text"], engine.VaultState())
        started = time.monotonic()
        try:
            found = await _llm_terms(prepared.segments, engine.Options(), m)
            opts = engine.Options(extra=tuple(found.terms), one_word_names=False)
            both, _r = await asyncio.to_thread(engine.anonymize_text, case["text"], engine.VaultState(), opts)
            error = ""
        except llm_lane.LaneFailed as exc:
            both, error = "", str(exc)
        out.append({
            "id": case["id"], "lanes": case["lanes"], "seconds": round(time.monotonic() - started, 1),
            "code_missed": [v for v in case["must_hide"] if v in det],
            "both_missed": [v for v in case["must_hide"] if v in both] if not error else None,
            "both_over_hid": [v for v in case["must_keep"] if v not in both] if not error else None,
            "error": error,
        })
    return {"model": m.label, "cases": out}


def engine_cases() -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    return json.loads((Path(engine.__file__).parent / "cases.json").read_text(encoding="utf-8"))["cases"]


# ── the privacy gate (the person's setting) ─────────────────────────


@router.get("/gate")
async def anonymize_gate_get(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.anonymize import gate as G

    settings = await gate_settings.load(user["id"])
    return {"gate": settings.public(), "vault": G.GATE_VAULT,
            "groups": [{"id": k, "label": G.GROUP_LABELS[k]} for k in G.GROUPS],
            "model_available": bool(await trusted.usable(get_db()))}


@router.put("/gate")
async def anonymize_gate_put(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    try:
        settings = await gate_settings.save(user["id"], await _body(request))
    except gate_settings.GateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    log.info("Privacy gate set", user=user["id"], enabled=settings.enabled, groups=settings.groups, when=settings.when)
    return {"gate": settings.public()}


@router.post("/reveal")
async def anonymize_reveal(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Real values for the person reading a reply (the chat's "Show real
    names"): the gate's vault unless another is named."""
    from captain_claw.anonymize import gate as G

    data = await _body(request)
    data.setdefault("vault", G.GATE_VAULT)
    data["format"] = "text"
    return await restore_for(user["id"], data, actor="person (chat)")


# ── admin: the name-recognition pack ────────────────────────────────


@router.get("/admin/ner")
async def anonymize_ner_get(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return ner.status(await ner.load_config(get_db()))


@router.put("/admin/ner")
async def anonymize_ner_put(request: Request, _admin: dict = Depends(require_admin)) -> dict[str, Any]:
    data = await _body(request)
    try:
        config = await ner.save_config(get_db(), bool(data.get("enabled")), str(data.get("model") or ner.DEFAULT_MODEL))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    ner.apply(config)  # loads it if it is already downloaded; never downloads by itself
    return ner.status(config)


@router.post("/admin/ner/download")
async def anonymize_ner_download(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    config = await ner.load_config(get_db())
    if not config["enabled"]:
        raise HTTPException(status_code=409, detail="Turn name recognition on first")
    if not ner.runtime_available():
        raise HTTPException(status_code=409, detail=(
            "This deck has no transformers / torch, so it can't run the model "
            "(pip install transformers torch, then restart Flight Deck)"))
    ner.apply(config, download=True)
    return ner.status(config)
