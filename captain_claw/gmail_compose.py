"""Gmail message building + send-policy helpers shared by the agent tool and FD.

Pure helpers (no Flight Deck imports) used by both sides of a Gmail send:

* :class:`~captain_claw.tools.google_mail.GoogleMailTool` — drafts, and sends
  directly in local (standalone) mode;
* ``captain_claw.flight_deck.gmail_send_routes`` — the Flight Deck send gate,
  which re-checks the owner's policy, sends with the owner's token and audits.

Recipient allowlists accept two entry forms:

* an exact address — ``alice@example.com``;
* a domain — ``@example.com`` (or bare ``example.com``, normalized to
  ``@example.com``) matching addresses AT that domain exactly. Subdomains are
  NOT included: ``@example.com`` does not allow ``bob@mail.example.com`` —
  list ``@mail.example.com`` separately. An allowlist is a guard against an
  injected "send this to x@…", so it never widens implicitly.
"""

from __future__ import annotations

import base64
import email.policy
import email.utils
import hashlib
import html as html_module
import json
import re
from email.headerregistry import Address
from email.message import EmailMessage
from typing import Any

import httpx

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"

_G = "https://www.googleapis.com/auth/"
# users.messages.send accepts any of these. gmail.compose (what decks grant
# for drafts) already allows sending — scopes are not the send gate, policy is.
SEND_SCOPES = (
    _G + "gmail.send",
    _G + "gmail.compose",
    _G + "gmail.modify",
    "https://mail.google.com/",
)
# users.drafts.send (and drafts.get) — gmail.send alone can't touch drafts.
DRAFT_SEND_SCOPES = (
    _G + "gmail.compose",
    _G + "gmail.modify",
    "https://mail.google.com/",
)

# Per message, across To + Cc + Bcc.
MAX_RECIPIENTS = 20

# Set by Flight Deck on a 403 that means "sending is turned off" (the user's
# policy, or the deck's FD_GMAIL_SEND kill switch) — the tool then points the
# agent at create_draft. Other 403s (unknown agent, allowlist) don't carry it.
SEND_REFUSED_HEADER = "X-FD-Gmail-Send"

# Set by Flight Deck (value ``unknown``) when the send call to Gmail failed in
# a way that may still have delivered (a 5xx, or no answer after the request
# went out) — the tool then says "may or may not have been sent", not "not sent".
SEND_OUTCOME_HEADER = "X-FD-Gmail-Send-Outcome"


# ---------------------------------------------------------------------------
# Raw RFC-822 message builder
# ---------------------------------------------------------------------------


def build_raw_message(
    to: str = "",
    cc: str = "",
    bcc: str = "",
    subject: str = "",
    body: str = "",
    html_body: str = "",
    in_reply_to: str = "",
    references: str = "",
) -> str:
    """Build a base64url-encoded RFC-822 message for the Gmail API.

    Construct with the SMTP policy from the start so every header
    fold and every line ending is RFC 5322-compliant CRLF. Building
    with the default compat32 policy and only applying SMTP at
    serialization time has been observed to produce drafts whose
    body Gmail's web UI fails to render in threaded reply drafts.
    """
    msg = EmailMessage(policy=email.policy.SMTP)
    if to:
        msg["To"] = to
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc
    if subject:
        msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    plain = body or ""

    # Always attach an HTML alternative. Gmail's web compose UI is
    # HTML-first — for threaded reply drafts specifically, it only
    # reliably renders the ``text/html`` part; drafts with just a
    # ``text/plain`` part have been observed to show an empty body
    # in the Gmail UI even though the raw message has the text.
    effective_html = html_body
    if not effective_html:
        effective_html = text_to_html(plain)
    if not plain and html_body:
        plain = html_to_text(html_body)

    msg.set_content(plain or " ")
    msg.add_alternative(effective_html, subtype="html")

    raw_bytes = msg.as_bytes()
    return base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")


def text_to_html(text: str) -> str:
    """Convert a plain text body to a simple HTML equivalent.

    Newlines become ``<br>`` and characters are HTML-escaped. This
    is intentionally minimal — Gmail will reformat it when the user
    opens the draft anyway, we just need a non-empty HTML part so
    the compose UI renders the body correctly.
    """
    if not text:
        return "<div></div>"
    escaped = html_module.escape(text)
    return "<div>" + escaped.replace("\n", "<br>") + "</div>"


def html_to_text(html_str: str) -> str:
    """Best-effort HTML to plain text conversion."""
    text = html_str
    # Replace common block tags with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<(hr)\s*/?>", "\n---\n", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html_module.unescape(text)
    # Collapse whitespace
    lines = [line.rstrip() for line in text.splitlines()]
    # Remove excessive blank lines
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Reply threading
# ---------------------------------------------------------------------------


async def fetch_reply_context(
    client: httpx.AsyncClient, token: str, message_id: str,
) -> dict[str, str]:
    """What a reply to *message_id* needs to thread under it in Gmail.

    Returns ``thread_id``, ``in_reply_to`` / ``references`` (the RFC 5322
    threading headers), ``reply_to_default`` (the recipient when the caller
    sets none; not reply-all) and ``subject_default`` (``Re: <original>``,
    unless it already starts with ``re:``). Raises
    :class:`httpx.HTTPStatusError` on an API error — callers map it.

    ``reply_to_default`` is the original's Reply-To, else From — except when
    the account owner wrote the original (labelled ``SENT`` or ``DRAFT``, e.g.
    a thread's "newest message" is the owner's own reply): then it is the
    original's To, as Gmail's Reply does, so a follow-up never goes back to
    the owner instead of the person it was meant for.
    """
    resp = await client.get(
        f"{GMAIL_API}/users/me/messages/{message_id}",
        params={
            "format": "metadata",
            "metadataHeaders": [
                "From", "To", "Cc", "Subject",
                "Message-ID", "References", "Reply-To",
            ],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    orig = resp.json()

    orig_headers: dict[str, str] = {}
    for h in orig.get("payload", {}).get("headers", []):
        name = (h.get("name") or "").lower()
        orig_headers[name] = h.get("value", "") or ""

    orig_msg_id = orig_headers.get("message-id", "")
    orig_refs = orig_headers.get("references", "")
    orig_subject = orig_headers.get("subject", "")

    in_reply_to = ""
    references = ""
    if orig_msg_id:
        in_reply_to = orig_msg_id
        references = (orig_refs + " " + orig_msg_id).strip() if orig_refs else orig_msg_id

    subject_default = ""
    if orig_subject:
        if orig_subject.lower().startswith("re:"):
            subject_default = orig_subject
        else:
            subject_default = f"Re: {orig_subject}"

    if {"SENT", "DRAFT"} & set(orig.get("labelIds") or []):
        reply_to_default = orig_headers.get("to", "")
    else:
        reply_to_default = orig_headers.get("reply-to") or orig_headers.get("from", "")

    return {
        "thread_id": orig.get("threadId", "") or "",
        "in_reply_to": in_reply_to,
        "references": references,
        "reply_to_default": reply_to_default,
        "subject_default": subject_default,
    }


# ---------------------------------------------------------------------------
# Recipients
# ---------------------------------------------------------------------------

# Deliberately plain: one @, a dotted domain, no whitespace. Header parsing is
# email.utils' job; this only rejects what parsed into a non-address ("Bob").
_ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DOMAIN_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$"
)


def _field_pairs(field: str) -> list[tuple[str, str]]:
    """``(name, address)`` pairs of one header value, blanks dropped. A value
    email.utils can't parse comes back as no pairs (newer Pythons parse
    strictly and return ``('', '')`` for the whole value)."""
    if not (field or "").strip():
        return []
    return [(n, a.strip()) for n, a in email.utils.getaddresses([field]) if a.strip()]


def parse_addresses(*fields: str) -> list[str]:
    """Every address in the header values *fields* (To, Cc, Bcc …):
    lowercased, deduplicated, in order, empties dropped."""
    out: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for _name, addr in _field_pairs(field):
            a = addr.lower()
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def _usable_address(addr: str) -> bool:
    """A plain ASCII addr-spec the message builder can put on the wire as is.
    Non-ASCII (``josé@…``, ``info@münchen.de``) can't be encoded into a header
    without changing the address — an IDN domain must be written ``xn--…``."""
    if not addr.isascii() or not _ADDRESS_RE.match(addr):
        return False
    try:
        Address(addr_spec=addr)
    except Exception:
        return False
    return True


def invalid_recipients(*fields: str) -> list[str]:
    """What in *fields* is not a usable recipient: a header value that parses
    to nothing (``a@b.c; d@e.f``, an injected line break …), an entry that
    isn't an address (``Bob``) or a non-ASCII address. Empty when everything
    is fine."""
    bad: list[str] = []
    for field in fields:
        if not (field or "").strip():
            continue
        pairs = _field_pairs(field)
        if not pairs:
            bad.append(field.strip()[:120])
            continue
        bad.extend(a for _n, a in pairs if not _usable_address(a))
    return bad


def invalid_recipients_message(bad: list[str]) -> str:
    """The refusal for :func:`invalid_recipients`' *bad* entries."""
    hint = "Separate addresses with commas."
    if any(not b.isascii() for b in bad):
        hint = ("Addresses must be plain ASCII (write an internationalized domain "
                "in its xn-- form); separate addresses with commas.")
    return f"Invalid recipient(s): {', '.join(bad)}. {hint} Nothing was sent."


def format_recipients(field: str) -> str:
    """Re-render header value *field* from the addresses it parses to.

    A send puts THIS on the wire, so the recipients the allowlist checked are
    exactly the ones Gmail delivers to — no gap between how this module and
    Gmail would read an odd header. Display names stay readable (UTF-8,
    quoted where needed): the message builder RFC 2047-encodes them on the
    wire, and the same string is what the user sees in results and history.
    Raises :class:`ValueError` for an address :func:`invalid_recipients`
    would have refused.
    """
    out: list[str] = []
    for name, addr in _field_pairs(field):
        if not _usable_address(addr):
            raise ValueError(f"not a usable address: {addr[:120]}")
        out.append(str(Address(display_name=name, addr_spec=addr)))
    return ", ".join(out)


def normalize_allowlist(entries: Any) -> list[str]:
    """Trimmed, lowercased, deduplicated allowlist (order kept); domains come
    back as ``@domain``. Raises :class:`ValueError` naming every invalid entry
    (or when *entries* isn't a list of strings)."""
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError("allowed_recipients must be a list of addresses or domains")
    out: list[str] = []
    seen: set[str] = set()
    bad: list[str] = []
    for raw in entries:
        if not isinstance(raw, str):
            bad.append(repr(raw))
            continue
        e = raw.strip().lower()
        if not e:
            continue
        if e.startswith("@"):
            norm = e if _DOMAIN_RE.match(e[1:]) else ""
        elif "@" in e:
            norm = e if _ADDRESS_RE.match(e) else ""
        else:
            norm = "@" + e if _DOMAIN_RE.match(e) else ""
        if not norm:
            bad.append(raw.strip())
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    if bad:
        raise ValueError(
            "Invalid allowed recipient(s): " + ", ".join(bad[:10])
            + (" …" if len(bad) > 10 else "")
            + ". Use an address (alice@example.com) or a domain (@example.com)."
        )
    return out


def recipients_allowed(addresses: list[str], allowlist: list[str]) -> list[str]:
    """The DISALLOWED subset of *addresses* under the normalized *allowlist*
    (see the module docstring for entry forms). An empty allowlist allows
    anyone, so it returns ``[]``."""
    if not allowlist:
        return []
    allowed = set(allowlist)
    out: list[str] = []
    for addr in addresses:
        a = (addr or "").strip().lower()
        domain = a.rsplit("@", 1)[-1] if "@" in a else ""
        if a in allowed or (domain and "@" + domain in allowed):
            continue
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------


def content_hash(
    to: str = "",
    cc: str = "",
    bcc: str = "",
    subject: str = "",
    body: str = "",
    html_body: str = "",
    reply_to: str = "",
) -> str:
    """sha256 hex of a message's normalized content — the same recipients
    (any order/case/display name), subject (whitespace/case) and body
    (line endings/trailing spaces) hash alike, so a retried send is caught.
    *reply_to* (the Gmail id of the message a reply answers) is part of it: the
    same short reply to two different emails is two emails, not a retry."""
    def _body(text: str) -> str:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return "\n".join(line.rstrip() for line in lines).strip()

    parts = [
        ",".join(sorted(parse_addresses(to))),
        ",".join(sorted(parse_addresses(cc))),
        ",".join(sorted(parse_addresses(bcc))),
        " ".join((subject or "").split()).lower(),
        _body(body),
        _body(html_body),
    ]
    if (reply_to or "").strip():
        parts.append("reply:" + reply_to.strip())
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()
