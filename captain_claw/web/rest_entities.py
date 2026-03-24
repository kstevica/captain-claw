"""REST handlers for entity CRUD (Todos, Contacts, Scripts, APIs)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


# ── Todos ────────────────────────────────────────────────────────────


async def list_todos(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/todos — list todo items."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    sm = server.agent.session_manager
    status_filter = request.query.get("status")
    responsible_filter = request.query.get("responsible")
    session_filter = request.query.get("session_id")
    items = await sm.list_todos(
        limit=200,
        status_filter=status_filter or None,
        responsible_filter=responsible_filter or None,
        session_filter=session_filter or None,
    )
    return web.json_response(
        [item.to_dict() for item in items],
        dumps=_JSON_DUMPS,
    )


async def create_todo(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/todos — create a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    content = str(body.get("content", "")).strip()
    if not content:
        return web.json_response({"error": "content is required"}, status=400)
    sm = server.agent.session_manager
    item = await sm.create_todo(
        content=content,
        responsible=str(body.get("responsible", "human")).strip() or "human",
        priority=str(body.get("priority", "normal")).strip() or "normal",
        source_session=str(body.get("source_session", "")).strip() or None,
        target_session=str(body.get("target_session", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
    )
    return web.json_response(item.to_dict(), status=201)


async def update_todo(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/todos/{id} — update a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    todo_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    sm = server.agent.session_manager
    kwargs: dict[str, Any] = {}
    for field in ("content", "status", "responsible", "priority", "target_session", "tags"):
        if field in body:
            kwargs[field] = str(body[field]).strip() if body[field] is not None else None
    ok = await sm.update_todo(todo_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Todo not found"}, status=404)
    item = await sm.load_todo(todo_id)
    return web.json_response(item.to_dict() if item else {"ok": True})


async def delete_todo(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/todos/{id} — delete a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    todo_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.delete_todo(todo_id)
    if not ok:
        return web.json_response({"error": "Todo not found"}, status=404)
    return web.json_response({"ok": True})


# ── Contacts ─────────────────────────────────────────────────────────


async def list_contacts(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts — list contacts."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    sm = server.agent.session_manager
    items = await sm.list_contacts(limit=200)
    return web.json_response(
        [c.to_dict() for c in items],
        dumps=_JSON_DUMPS,
    )


async def search_contacts(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts/search?q= — search contacts."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    sm = server.agent.session_manager
    items = await sm.search_contacts(query, limit=50)
    return web.json_response(
        [c.to_dict() for c in items],
        dumps=_JSON_DUMPS,
    )


async def create_contact(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/contacts — create a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    sm = server.agent.session_manager
    item = await sm.create_contact(
        name=name,
        description=str(body.get("description", "")).strip() or None,
        position=str(body.get("position", "")).strip() or None,
        organization=str(body.get("organization", "")).strip() or None,
        relation=str(body.get("relation", "")).strip() or None,
        email=str(body.get("email", "")).strip() or None,
        phone=str(body.get("phone", "")).strip() or None,
        importance=int(body.get("importance", 1)),
        tags=str(body.get("tags", "")).strip() or None,
        notes=str(body.get("notes", "")).strip() or None,
        privacy_tier=str(body.get("privacy_tier", "normal")).strip() or "normal",
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_contact(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts/{id} — get a single contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    item = await sm.load_contact(contact_id)
    if not item:
        return web.json_response({"error": "Contact not found"}, status=404)
    return web.json_response(
        item.to_dict(),
        dumps=_JSON_DUMPS,
    )


async def update_contact(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/contacts/{id} — update a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    sm = server.agent.session_manager
    kwargs: dict[str, Any] = {}
    for field in ("name", "description", "position", "organization", "relation",
                   "email", "phone", "tags", "notes", "privacy_tier"):
        if field in body:
            kwargs[field] = str(body[field]).strip() if body[field] is not None else None
    if "importance" in body:
        kwargs["importance"] = max(1, min(10, int(body["importance"])))
        kwargs["importance_pinned"] = True
    ok = await sm.update_contact(contact_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Contact not found"}, status=404)
    item = await sm.load_contact(contact_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_contact(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/contacts/{id} — delete a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.delete_contact(contact_id)
    if not ok:
        return web.json_response({"error": "Contact not found"}, status=404)
    return web.json_response({"ok": True})


async def preview_contacts_import(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/contacts/import/preview — parse CSV/vCard headers + sample rows for mapping UI.

    Returns JSON with:
    - headers: list of column names from the file
    - sample: first 3 data rows as dicts
    - autoMap: suggested mapping from our fields to CSV columns
    - format: "csv" or "vcard"
    """
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "Multipart body required"}, status=400)

        file_field = None
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                file_field = field
                break

        if file_field is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        original_name = file_field.filename or "contacts.csv"
        ext = Path(original_name).suffix.lower()

        chunks: list[bytes] = []
        while True:
            chunk = await file_field.read_chunk(8192)
            if not chunk:
                break
            chunks.append(chunk)
        file_bytes = b"".join(chunks)
        if not file_bytes:
            return web.json_response({"error": "Empty file"}, status=400)

        if ext == ".vcf":
            cards = _parse_vcards(file_bytes.decode("utf-8", errors="replace"))
            # Build a tabular preview from vCard data.
            headers = ["FN", "N", "EMAIL", "TEL", "ORG", "TITLE", "NOTE"]
            sample: list[dict[str, str]] = []
            for card in cards[:3]:
                sample.append({
                    "FN": card.get("fn", ""),
                    "N": card.get("n", ""),
                    "EMAIL": ", ".join(card.get("emails", [])),
                    "TEL": ", ".join(card.get("phones", [])),
                    "ORG": card.get("org", ""),
                    "TITLE": card.get("title", ""),
                    "NOTE": card.get("note", ""),
                })
            auto_map = {
                "name": "FN", "email": "EMAIL", "phone": "TEL",
                "organization": "ORG", "position": "TITLE", "notes": "NOTE",
            }
            return web.json_response({
                "headers": headers,
                "sample": sample,
                "autoMap": auto_map,
                "totalRows": len(cards),
                "format": "vcard",
                "filename": original_name,
            })

        elif ext == ".csv":
            text = file_bytes.decode("utf-8-sig")
            csv_reader = csv.DictReader(io.StringIO(text))
            headers = csv_reader.fieldnames or []
            sample = []
            row_count = 0
            for row in csv_reader:
                row_count += 1
                if len(sample) < 3:
                    sample.append({h: str(row.get(h, "") or "").strip() for h in headers})

            # Auto-detect best mapping.
            auto_map = _auto_map_csv_columns(headers)

            return web.json_response({
                "headers": list(headers),
                "sample": sample,
                "autoMap": auto_map,
                "totalRows": row_count,
                "format": "csv",
                "filename": original_name,
            })

        else:
            return web.json_response(
                {"error": f"Unsupported format '{ext}'. Use .csv or .vcf."},
                status=400,
            )

    except web.HTTPException:
        raise
    except Exception as exc:
        return web.json_response({"error": f"Preview failed: {exc}"}, status=500)


# Our contact fields and known CSV column aliases for auto-mapping.
_FIELD_ALIASES: dict[str, list[str]] = {
    "name": ["Name", "File As", "Full Name", "Display Name"],
    "first_name": ["First Name", "Given Name"],
    "last_name": ["Last Name", "Family Name"],
    "middle_name": ["Middle Name", "Additional Name"],
    "email": [
        "E-mail 1 - Value", "E-mail 2 - Value", "E-mail 3 - Value",
        "Email", "Email Address", "E-mail Address",
    ],
    "phone": [
        "Phone 1 - Value", "Phone 2 - Value", "Phone 3 - Value",
        "Phone", "Mobile Phone", "Primary Phone",
    ],
    "organization": ["Organization Name", "Organization 1 - Name", "Company", "Company Name"],
    "position": ["Organization Title", "Organization 1 - Title", "Job Title", "Title"],
    "notes": ["Notes", "Note"],
    "tags": ["Labels", "Group Membership", "Categories", "Groups"],
    "description": ["Nickname", "Occupation"],
}


def _auto_map_csv_columns(headers: list[str]) -> dict[str, str]:
    """Given CSV headers, return best-guess mapping {our_field: csv_column}."""
    header_lower = {h.lower().strip(): h for h in headers}
    result: dict[str, str] = {}

    for our_field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in header_lower:
                result[our_field] = header_lower[alias.lower()]
                break

    return result


async def import_contacts(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/contacts/import — bulk import contacts from Google Contacts CSV or vCard file.

    Accepts multipart form with:
    - ``file`` field containing a .csv or .vcf file
    - optional ``mapping`` field: JSON string with column mapping {our_field: csv_column}

    Returns JSON summary with imported count and skipped entries.
    """
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)

    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "Multipart body required"}, status=400)

        file_bytes = b""
        original_name = "contacts.csv"
        mapping_json = ""
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                original_name = field.filename or "contacts.csv"
                chunks: list[bytes] = []
                while True:
                    chunk = await field.read_chunk(8192)
                    if not chunk:
                        break
                    chunks.append(chunk)
                file_bytes = b"".join(chunks)
            elif field.name == "mapping":
                mapping_json = (await field.read()).decode("utf-8", errors="replace")

        if not file_bytes:
            return web.json_response({"error": "No file or empty file"}, status=400)

        ext = Path(original_name).suffix.lower()

        # Parse user-supplied mapping if provided.
        mapping: dict[str, str] = {}
        if mapping_json:
            try:
                mapping = json.loads(mapping_json)
            except Exception:
                pass

        sm = server.agent.session_manager

        if ext == ".vcf":
            result = await _import_vcard(sm, file_bytes)
        elif ext == ".csv":
            result = await _import_google_csv(sm, file_bytes, mapping=mapping)
        else:
            return web.json_response(
                {"error": f"Unsupported format '{ext}'. Use .csv (Google Contacts export) or .vcf (vCard)."},
                status=400,
            )

        return web.json_response(result)

    except web.HTTPException:
        raise
    except Exception as exc:
        return web.json_response({"error": f"Import failed: {exc}"}, status=500)


def _csv_get(row: dict[str, str], *keys: str) -> str:
    """Return the first non-empty value found for any of the given column names."""
    for key in keys:
        val = str(row.get(key, "") or "").strip()
        if val:
            return val
    return ""


async def _import_google_csv(
    sm: Any,
    file_bytes: bytes,
    *,
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse contacts CSV and create contact entries using column mapping.

    *mapping* is {our_field: csv_column_name}.  When provided the import
    uses the user's explicit mapping.  Otherwise it falls back to auto-detection.
    """
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []

    # Resolve mapping: user-supplied > auto-detected.
    m = mapping if mapping else _auto_map_csv_columns(headers)

    def _mg(row: dict[str, str], field: str) -> str:
        """Get value for a mapped field."""
        col = m.get(field, "")
        if not col:
            return ""
        return str(row.get(col, "") or "").strip()

    def _mg_multi(row: dict[str, str], field: str) -> list[str]:
        """Get values for a field that may map to multiple numbered columns (email, phone)."""
        col = m.get(field, "")
        if not col:
            return []
        vals: list[str] = []
        # If column exists directly, use it.
        direct = str(row.get(col, "") or "").strip()
        if direct:
            vals.append(direct)
        # Also scan for numbered variants (E-mail 2 - Value, Phone 2 - Value, etc.).
        # Detect the pattern from the mapped column name.
        for variant_num in range(2, 5):
            if "1" in col:
                variant_col = col.replace("1", str(variant_num))
                variant_val = str(row.get(variant_col, "") or "").strip()
                if variant_val:
                    vals.append(variant_val)
        return vals

    imported = 0
    skipped = 0
    errors: list[str] = []

    for row_num, row in enumerate(reader, start=2):
        # Build name from mapped fields.
        first = _mg(row, "first_name")
        middle = _mg(row, "middle_name")
        last = _mg(row, "last_name")
        full_name = _mg(row, "name")

        name_parts = [p for p in (first, middle, last) if p]
        name = " ".join(name_parts) if name_parts else full_name

        # Fallbacks when no name fields mapped or all empty.
        if not name:
            name = _mg(row, "organization")
        if not name:
            emails_fb = _mg_multi(row, "email")
            name = emails_fb[0] if emails_fb else ""
        if not name:
            phones_fb = _mg_multi(row, "phone")
            name = phones_fb[0] if phones_fb else ""
        if not name:
            skipped += 1
            continue

        # Collect emails and phones (with numbered variants).
        emails = _mg_multi(row, "email")
        email = ", ".join(emails) if emails else None

        phones_raw = _mg_multi(row, "phone")
        # Try to append type/label if available.
        phones: list[str] = []
        phone_col = m.get("phone", "")
        for i, pval in enumerate(phones_raw):
            col_num = str(i + 1)
            label_col = phone_col.replace("Value", "Label").replace("1", col_num) if "1" in phone_col else ""
            if not label_col:
                label_col = phone_col.replace("Value", "Type").replace("1", col_num) if "1" in phone_col else ""
            plabel = str(row.get(label_col, "") or "").strip() if label_col else ""
            phones.append(f"{pval} ({plabel})" if plabel else pval)
        phone = ", ".join(phones) if phones else None

        organization = _mg(row, "organization") or None
        position = _mg(row, "position") or None
        notes = _mg(row, "notes") or None
        description = _mg(row, "description") or None

        # Tags from Labels/Groups.
        groups = _mg(row, "tags")
        tags = None
        if groups:
            parts = [g.strip() for g in groups.split(":::")]
            parts = [g for g in parts if g and g != "* myContacts"]
            if parts:
                tags = ", ".join(parts)

        try:
            await sm.create_contact(
                name=name,
                email=email,
                phone=phone,
                organization=organization,
                position=position,
                notes=notes,
                tags=tags,
                description=description,
                source_session="google-contacts-import",
            )
            imported += 1
        except Exception as e:
            skipped += 1
            errors.append(f"Row {row_num} ({name}): {e}")

    result: dict[str, Any] = {
        "imported": imported,
        "skipped": skipped,
        "format": "google-csv",
    }
    if errors:
        result["errors"] = errors[:20]
    return result


async def _import_vcard(sm: Any, file_bytes: bytes) -> dict[str, Any]:
    """Parse vCard (.vcf) file and create contact entries.

    Supports vCard 2.1, 3.0, and 4.0.  One .vcf can contain multiple
    contacts separated by BEGIN:VCARD / END:VCARD blocks.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    cards = _parse_vcards(text)

    imported = 0
    skipped = 0
    errors: list[str] = []

    for idx, card in enumerate(cards, start=1):
        name = card.get("fn", "").strip()
        if not name:
            # Build from N field: family;given;middle;prefix;suffix
            n_parts = card.get("n", "").split(";")
            given = n_parts[1].strip() if len(n_parts) > 1 else ""
            family = n_parts[0].strip() if n_parts else ""
            name = f"{given} {family}".strip()
        if not name:
            skipped += 1
            continue

        email = ", ".join(card.get("emails", [])) or None
        phone = ", ".join(card.get("phones", [])) or None
        organization = card.get("org", "").split(";")[0].strip() or None
        position = card.get("title", "").strip() or None
        notes = card.get("note", "").strip() or None

        try:
            await sm.create_contact(
                name=name,
                email=email,
                phone=phone,
                organization=organization,
                position=position,
                notes=notes,
                source_session="vcard-import",
            )
            imported += 1
        except Exception as e:
            skipped += 1
            errors.append(f"Card {idx} ({name}): {e}")

    result: dict[str, Any] = {
        "imported": imported,
        "skipped": skipped,
        "format": "vcard",
    }
    if errors:
        result["errors"] = errors[:20]
    return result


def _parse_vcards(text: str) -> list[dict[str, Any]]:
    """Lightweight vCard parser.  Returns a list of dicts, one per card."""
    cards: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.upper() == "BEGIN:VCARD":
            current = {"emails": [], "phones": []}
            continue
        if line.upper() == "END:VCARD":
            if current:
                cards.append(current)
            current = None
            continue
        if current is None:
            continue

        # Split property name from value.
        if ":" not in line:
            continue
        prop_part, _, value = line.partition(":")
        # Handle parameters (e.g., TEL;TYPE=CELL:+123).
        prop_name = prop_part.split(";")[0].upper()

        if prop_name == "FN":
            current["fn"] = value.strip()
        elif prop_name == "N":
            current["n"] = value.strip()
        elif prop_name == "ORG":
            current["org"] = value.strip()
        elif prop_name == "TITLE":
            current["title"] = value.strip()
        elif prop_name == "NOTE":
            current["note"] = value.strip()
        elif prop_name == "EMAIL":
            val = value.strip()
            if val:
                current["emails"].append(val)
        elif prop_name == "TEL":
            val = value.strip()
            if val:
                current["phones"].append(val)

    return cards


# ── Scripts ──────────────────────────────────────────────────────────


async def list_scripts(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    items = await server.agent.session_manager.list_scripts(limit=200)
    return web.json_response(
        [s.to_dict() for s in items],
        dumps=_JSON_DUMPS,
    )


async def search_scripts(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    items = await server.agent.session_manager.search_scripts(query, limit=50)
    return web.json_response(
        [s.to_dict() for s in items],
        dumps=_JSON_DUMPS,
    )


async def create_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    file_path = str(body.get("file_path", "")).strip()
    if not name or not file_path:
        return web.json_response({"error": "name and file_path are required"}, status=400)
    item = await server.agent.session_manager.create_script(
        name=name, file_path=file_path,
        description=str(body.get("description", "")).strip() or None,
        purpose=str(body.get("purpose", "")).strip() or None,
        language=str(body.get("language", "")).strip() or None,
        created_reason=str(body.get("created_reason", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    item = await server.agent.session_manager.load_script(script_id)
    if not item:
        return web.json_response({"error": "Script not found"}, status=404)
    return web.json_response(
        item.to_dict(), dumps=_JSON_DUMPS,
    )


async def update_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    kwargs: dict[str, Any] = {}
    for fld in ("name", "file_path", "description", "purpose", "language",
                 "created_reason", "tags"):
        if fld in body:
            kwargs[fld] = str(body[fld]).strip() if body[fld] is not None else None
    ok = await server.agent.session_manager.update_script(script_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Script not found"}, status=404)
    item = await server.agent.session_manager.load_script(script_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    ok = await server.agent.session_manager.delete_script(script_id)
    if not ok:
        return web.json_response({"error": "Script not found"}, status=404)
    return web.json_response({"ok": True})


# ── APIs ─────────────────────────────────────────────────────────────


async def list_apis(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    items = await server.agent.session_manager.list_apis(limit=200)
    return web.json_response(
        [a.to_dict() for a in items],
        dumps=_JSON_DUMPS,
    )


async def search_apis(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    items = await server.agent.session_manager.search_apis(query, limit=50)
    return web.json_response(
        [a.to_dict() for a in items],
        dumps=_JSON_DUMPS,
    )


async def create_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    base_url = str(body.get("base_url", "")).strip()
    if not name or not base_url:
        return web.json_response({"error": "name and base_url are required"}, status=400)
    item = await server.agent.session_manager.create_api(
        name=name, base_url=base_url,
        endpoints=str(body.get("endpoints", "")).strip() or None,
        auth_type=str(body.get("auth_type", "")).strip() or None,
        credentials=str(body.get("credentials", "")).strip() or None,
        description=str(body.get("description", "")).strip() or None,
        purpose=str(body.get("purpose", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    item = await server.agent.session_manager.load_api(api_id)
    if not item:
        return web.json_response({"error": "API not found"}, status=404)
    return web.json_response(
        item.to_dict(), dumps=_JSON_DUMPS,
    )


async def update_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    kwargs: dict[str, Any] = {}
    for fld in ("name", "base_url", "endpoints", "auth_type", "credentials",
                 "description", "purpose", "tags"):
        if fld in body:
            kwargs[fld] = str(body[fld]).strip() if body[fld] is not None else None
    ok = await server.agent.session_manager.update_api(api_id, **kwargs)
    if not ok:
        return web.json_response({"error": "API not found"}, status=404)
    item = await server.agent.session_manager.load_api(api_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    ok = await server.agent.session_manager.delete_api(api_id)
    if not ok:
        return web.json_response({"error": "API not found"}, status=404)
    return web.json_response({"ok": True})
