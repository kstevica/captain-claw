"""Gmail message building + send-policy helpers shared by the agent tool and FD.

Pure helpers (no Flight Deck imports) used by both sides of a Gmail send:

* :class:`~captain_claw.tools.google_mail.GoogleMailTool` — drafts, and sends
  directly in local (standalone) mode;
* ``captain_claw.flight_deck.gmail_send_routes`` — the Flight Deck send gate,
  which re-checks the owner's policy, sends with the owner's token and audits.

The duplicate checks live here too: :func:`content_hash` (Flight Deck's
exact-retry check) and :func:`find_repeats` (the same email — recipient and
subject — already in Drafts or Sent), whose refusal wording
(:func:`repeat_refusal`) both sides share.

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
import difflib
import email.policy
import email.utils
import hashlib
import html as html_module
import json
import logging
import re
from email.headerregistry import Address
from email.message import EmailMessage
from typing import Any

import httpx

_log = logging.getLogger(__name__)

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
    sets none; not reply-all), ``subject_default`` (``Re: <original>``,
    unless it already starts with ``re:``) and ``internal_date`` (the
    original's Gmail ``internalDate``, epoch ms as text — what
    :func:`find_repeats` compares a thread's sent replies against). Raises
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
        "internal_date": str(orig.get("internalDate") or ""),
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


# ---------------------------------------------------------------------------
# Repeat check — the same email already drafted or sent
# ---------------------------------------------------------------------------
#
# content_hash above catches a byte-identical retry of a Flight Deck send. This
# catches the near-identical one — a second draft of an email that is already
# in Drafts, a re-send of one already in Sent (from any session, cron run or
# the user's own Gmail) — by recipient + subject, with the scopes agents
# already hold: drafts.list / drafts.get take gmail.compose or gmail.readonly;
# messages.list / threads.get need gmail.readonly.

# Reply / forward prefixes: English re / fw / fwd, German aw (Antwort), Nordic
# sv (svar), Croatian odg (odgovor) — optionally numbered ("Re[2]:").
_SUBJECT_PREFIX_RE = re.compile(r"^\s*(?:re|fwd?|aw|sv|odg)\s*(?:\[\d+\])?\s*:\s*", re.IGNORECASE)

# difflib ratio at or above which two normalized subjects are the same email.
SUBJECT_SIMILARITY = 0.85

# Drafts / Sent candidates one check reads (one metadata GET each).
REPEAT_CANDIDATES = 10

# Recipients that go into the Drafts / Sent search query.
_QUERY_ADDRESSES = 10

# In every repeat refusal — the tool's and Flight Deck's — so callers (the
# autonomous action rail) can tell "already done" from a failure.
REPEAT_MARK = "— repeat of "


def normalize_subject(subject: str) -> str:
    """*subject* without its reply / forward prefixes (repeatedly: ``Re: Fwd:
    RE: x`` → ``x``), case-folded, whitespace collapsed."""
    s = subject or ""
    while True:
        stripped = _SUBJECT_PREFIX_RE.sub("", s, count=1)
        if stripped == s:
            break
        s = stripped
    return " ".join(s.split()).casefold()


_DIGITS_RE = re.compile(r"\d+")


def subjects_match(a: str, b: str) -> bool:
    """The same email by subject: equal after :func:`normalize_subject`, or a
    difflib ratio of at least :data:`SUBJECT_SIMILARITY` with the same numbers
    in both — "Invoice 1041" / "Invoice 1042", "Q3 report" / "Q4 report" or
    two dated weekly reports are different emails, however alike the text.
    Two empty subjects are not a match (nothing to compare)."""
    na, nb = normalize_subject(a), normalize_subject(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if _DIGITS_RE.findall(na) != _DIGITS_RE.findall(nb):
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= SUBJECT_SIMILARITY


def short_date(value: str) -> str:
    """An RFC 5322 Date header as ``YYYY-MM-DD HH:MM`` (as written — the
    sender's zone), else the first 20 characters of whatever it is."""
    try:
        return email.utils.parsedate_to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return (value or "")[:20]


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _candidate(message: dict[str, Any], kind: str, draft_id: str = "") -> dict[str, str]:
    """An existing Gmail message (*kind* ``draft`` / ``sent``) as a match."""
    headers: dict[str, str] = {}
    for h in (message.get("payload") or {}).get("headers", []):
        name = (h.get("name") or "").lower()
        if name in ("to", "cc", "bcc", "subject", "date"):
            headers[name] = h.get("value", "") or ""
    return {
        "kind": kind,
        "draft_id": draft_id,
        "message_id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "bcc": headers.get("bcc", ""),
        "subject": headers.get("subject", ""),
        "date": short_date(headers["date"]) if headers.get("date") else "",
    }


async def _get_json(
    client: httpx.AsyncClient, path: str, params: dict[str, Any], headers: dict[str, str],
) -> dict[str, Any]:
    resp = await client.get(f"{GMAIL_API}/users/me{path}", params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


async def _thread_repeats(
    client: httpx.AsyncClient, headers: dict[str, str], thread_id: str, after_ms: int,
) -> list[dict[str, str]]:
    """A reply's repeats: a DRAFT in the thread, or a SENT message newer than
    the email being answered (skipped when its date is unknown — the order
    can't be told)."""
    try:
        thread = await _get_json(client, f"/threads/{thread_id}", {"format": "metadata"}, headers)
    except Exception as exc:
        _log.debug("Repeat check: thread %s unreadable: %s", thread_id, exc)
        return []
    drafts: list[dict[str, str]] = []
    sent: list[dict[str, str]] = []
    for message in thread.get("messages") or []:
        labels = set(message.get("labelIds") or [])
        if "DRAFT" in labels:
            drafts.append(_candidate(message, "draft"))
        elif "SENT" in labels and after_ms and _as_int(message.get("internalDate")) > after_ms:
            sent.append(_candidate(message, "sent"))
    if drafts:
        # threads.get has no draft ids — drafts.list maps message id → Draft ID.
        try:
            listing = await _get_json(client, "/drafts", {"maxResults": 100}, headers)
            ids = {
                str((stub.get("message") or {}).get("id") or ""): str(stub.get("id") or "")
                for stub in listing.get("drafts") or []
            }
            for d in drafts:
                d["draft_id"] = ids.get(d["message_id"], "")
        except Exception as exc:
            _log.debug("Repeat check: drafts list failed: %s", exc)
    return drafts + sent


async def find_repeats(
    client: httpx.AsyncClient,
    token: str,
    *,
    to: str = "",
    cc: str = "",
    bcc: str = "",
    subject: str = "",
    thread_id: str = "",
    after_ms: int | str = 0,
    days: int = 14,
    limit: int = REPEAT_CANDIDATES,
) -> list[dict[str, str]]:
    """Emails already drafted or sent that a new one would repeat.

    * A reply (*thread_id* — the thread it goes into): the thread holds a
      DRAFT, or a SENT message newer than the email being answered
      (*after_ms*, its ``internalDate``), to a recipient of this one.
    * Otherwise: a draft (any age), or an email in Sent from the last *days*
      days, with a recipient in common (To/Cc/Bcc) and the same subject
      (:func:`subjects_match`). An email without a subject or a To/Cc
      recipient isn't checked.

    Each match is a dict: ``kind`` (``draft`` / ``sent``), ``draft_id`` (drafts;
    may be empty), ``message_id``, ``thread_id``, ``to``, ``cc``, ``bcc``,
    ``subject``, ``date``. Drafts come first. *days* 0 turns the check off.

    Never raises — every lookup fails open: a grant with gmail.compose but no
    gmail.readonly gets a 403 on the Sent search and still has Drafts checked.
    """
    if _as_int(days) <= 0:
        return []

    def _header_text(value: Any) -> Any:
        # A model may pass recipients as a list (the message builder takes one).
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))
        return value

    to, cc, bcc = _header_text(to), _header_text(cc), _header_text(bcc)
    try:
        mine = set(parse_addresses(to, cc, bcc))
        query_addrs = parse_addresses(to, cc)[:_QUERY_ADDRESSES]
        has_subject = bool(normalize_subject(subject))
    except Exception as exc:  # not header text (a list, None …) — nothing to compare
        _log.debug("Repeat check skipped: %s", exc)
        return []
    headers = {"Authorization": f"Bearer {token}"}

    def _shares_recipient(candidate: dict[str, str]) -> bool:
        try:
            theirs = set(parse_addresses(candidate["to"], candidate["cc"], candidate["bcc"]))
        except Exception:  # an unparseable header on their side — fail open
            return False
        return bool(mine & theirs)

    if thread_id:
        # Different recipients are not repeats — but a thread draft with no
        # recipient yet (or a reply with none) still is one.
        return [
            c for c in await _thread_repeats(client, headers, thread_id, _as_int(after_ms))
            if not mine or not (c["to"] or c["cc"] or c["bcc"]) or _shares_recipient(c)
        ]

    if not has_subject or not query_addrs:
        return []
    # Gmail's to: matches only the To header — ask for Cc too, as one OR group.
    who = "{" + " ".join(f"{op}:{a}" for a in query_addrs for op in ("to", "cc")) + "}"

    def _same(candidate: dict[str, str]) -> bool:
        return _shares_recipient(candidate) and subjects_match(subject, candidate["subject"])

    matches: list[dict[str, str]] = []
    try:
        listing = await _get_json(client, "/drafts", {"q": who, "maxResults": limit}, headers)
        for stub in (listing.get("drafts") or [])[:limit]:
            draft_id = str(stub.get("id") or "")
            if not draft_id:
                continue
            try:
                draft = await _get_json(client, f"/drafts/{draft_id}", {"format": "metadata"}, headers)
            except Exception as exc:
                _log.debug("Repeat check: draft %s unreadable: %s", draft_id, exc)
                continue
            candidate = _candidate(draft.get("message") or {}, "draft", draft_id)
            if _same(candidate):
                matches.append(candidate)
    except Exception as exc:
        _log.debug("Repeat check: drafts search failed: %s", exc)

    try:
        listing = await _get_json(
            client, "/messages",
            {"q": f"in:sent {who} newer_than:{_as_int(days)}d", "maxResults": limit}, headers,
        )
        for stub in (listing.get("messages") or [])[:limit]:
            message_id = str(stub.get("id") or "")
            if not message_id:
                continue
            try:
                message = await _get_json(client, f"/messages/{message_id}", {
                    "format": "metadata",
                    "metadataHeaders": ["To", "Cc", "Bcc", "Subject", "Date"],
                }, headers)
            except Exception as exc:
                _log.debug("Repeat check: message %s unreadable: %s", message_id, exc)
                continue
            candidate = _candidate(message, "sent")
            if _same(candidate):
                matches.append(candidate)
    except Exception as exc:
        # A 403 here is a compose-only grant — Drafts above were still checked.
        _log.debug("Repeat check: Sent search failed: %s", exc)
    return matches


def describe_repeat(match: dict[str, str]) -> str:
    """One existing email as a repeat refusal names it."""
    what = f'"{match.get("subject") or "(no subject)"}" to {match.get("to") or "?"}'
    if match.get("kind") == "draft":
        if match.get("draft_id"):
            return f"Draft ID {match['draft_id']} ({what})"
        return (f"a draft in this thread, Message ID {match.get('message_id') or '?'} "
                f"({what}; list_drafts shows its Draft ID)")
    return (f"the email sent on {match.get('date') or '?'}, Message ID "
            f"{match.get('message_id') or '?'} ({what})")


def repeat_refusal(matches: list[dict[str, str]], *, lead: str) -> str:
    """The refusal for :func:`find_repeats`' *matches* — what exists, what to do
    instead, and the one way past it. *lead* says what didn't happen ("Not
    created", "Not sent"). Always contains :data:`REPEAT_MARK`."""
    first = matches[0]
    more = f" and {len(matches) - 1} more like it" if len(matches) > 1 else ""
    draft_id = next(
        (m["draft_id"] for m in matches if m.get("kind") == "draft" and m.get("draft_id")), "",
    )
    text = (
        f"{lead} {REPEAT_MARK}{describe_repeat(first)}{more}. This email is already "
        "drafted or sent, so no new copy was made. Tell the user, with that Draft ID or "
        "sent date."
    )
    if draft_id:
        text += (
            f" To send that draft use google_mail action=send_draft draft_id={draft_id}; "
            f"to change it use action=update_draft draft_id={draft_id}."
        )
    return text + (
        " Don't retry. Only if the user explicitly asks for another, separate copy, "
        "call again with allow_repeat=true."
    )


def is_repeat_refusal(text: str) -> bool:
    """Whether a tool / Flight Deck error is a :func:`repeat_refusal` — the email
    already existed, so nothing failed."""
    return REPEAT_MARK in (text or "")
