"""IMAP + SMTP email with a username and password — the core Flight Deck runs.

The third mail connection next to Google (Gmail API) and Microsoft (Graph):
any mailbox that speaks IMAP for reading and SMTP for sending — a company
mail server, Fastmail, iCloud, Yahoo, Zoho, GMX, a hosting provider's mail, a
Gmail account with an app password, Proton Mail through its Bridge.

A Flight Deck user can connect **several** of these inboxes. Each is an
:class:`Account` with its own short ``id`` (``work``, ``support``, …), and that
id starts every reference this module hands out, so an agent can never act on
one inbox with an id it got from another:

* message / draft ID — ``<account>:<uidvalidity>:<uid>:<folder>``. The folder
  is last (it may contain ``:``); UIDVALIDITY makes a stale id fail instead of
  reaching a different message after the server renumbered the folder.
* thread ID — ``<account>:<root Message-ID>``: the first ``References`` entry,
  else ``In-Reply-To``, else the message's own Message-ID. Threads are found by
  header search (``Message-ID`` / ``References`` / ``In-Reply-To``), so they
  work on every server, not only those with the THREAD extension.

Everything here is synchronous (``imaplib`` / ``smtplib`` from the standard
library — no new dependency) and takes the account with its password: Flight
Deck keeps the password encrypted (``fd_crypto``) and calls these functions in
a worker thread. The password never reaches an agent.

Security choices:

* SSL/TLS or STARTTLS, with certificate checks. Plain text, and TLS without a
  certificate check, only to a loopback host (a local bridge such as Proton
  Mail Bridge, whose certificate is self-signed).
* A host that resolves to a link-local or cloud-metadata address is refused —
  a mail setting must not become a way to probe the deck's own network.
* Nothing is ever deleted except a draft this module replaces or sends; moves
  never go to Trash or Junk.
"""

from __future__ import annotations

import base64
import email
import email.header
import email.headerregistry
import email.policy
import email.utils
import imaplib
import ipaddress
import re
import smtplib
import socket
import ssl
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import Message
from typing import Any

from captain_claw import gmail_compose, mail_attachments

TIMEOUT = 30.0
MAX_MESSAGE_BYTES = 30 * 1024 * 1024
MAX_BODY_CHARS = 30_000
MAX_THREAD_BODY_CHARS = 4_000
MAX_RESULTS = 50
MAX_REFS = 100
PREVIEW_BYTES = 16_384
PREVIEW_CHARS = 240
SECURITY = ("ssl", "starttls", "none")
DEFAULT_PORTS = {("imap", "ssl"): 993, ("imap", "starttls"): 143, ("imap", "none"): 143,
                 ("smtp", "ssl"): 465, ("smtp", "starttls"): 587, ("smtp", "none"): 25}
ACCOUNT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class MailError(Exception):
    """A failure with a message written for the person / the model.
    *status* is the HTTP status Flight Deck answers with."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class SendOutcomeUnknown(MailError):
    """The message went to the SMTP server and the answer never came back:
    it may have been delivered. Never retried blindly."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 504)


# ── accounts ────────────────────────────────────────────────────────


@dataclass
class Account:
    """One connected inbox. ``smtp_host`` "" = receive-only (no sending)."""

    id: str
    email: str
    username: str
    password: str
    imap_host: str
    imap_port: int
    imap_security: str = "ssl"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_security: str = "ssl"
    # A separate SMTP login (a relay service, a provider with other SMTP
    # credentials); empty = the IMAP login.
    smtp_username: str = ""
    smtp_password: str = ""
    label: str = ""
    display_name: str = ""
    # Put a copy of every send into the Sent folder. Off for providers that
    # file SMTP sends themselves (Gmail, Outlook.com, Yahoo…), or it doubles.
    save_sent: bool = True

    @property
    def can_send(self) -> bool:
        return bool(self.smtp_host)

    @property
    def name(self) -> str:
        return self.label or self.email

    def from_header(self) -> str:
        if self.display_name:
            return str(email.headerregistry.Address(display_name=self.display_name, addr_spec=self.email))
        return self.email

    def smtp_login(self) -> tuple[str, str]:
        if self.smtp_username:
            return self.smtp_username, self.smtp_password
        return self.username, self.password


_HOST_RE = re.compile(
    r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def clean_host(value: Any, what: str) -> str:
    """A host name or IP literal, lower-cased; ValueError otherwise."""
    host = str(value or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        raise ValueError(f"{what} is required")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if "://" in host or "/" in host:
        raise ValueError(f"{what} must be a host name like imap.example.com, not a URL")
    if not _HOST_RE.match(host):
        raise ValueError(f"{what} isn't a valid host name: {host[:80]}")
    return host


def is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _port(value: Any, default: int, what: str) -> int:
    if value in (None, ""):
        return default
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number from 1 to 65535") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{what} must be a number from 1 to 65535")
    return port


def _security(value: Any, host: str, what: str) -> str:
    sec = str(value or "ssl").strip().lower().replace("/", "").replace("-", "")
    sec = {"ssltls": "ssl", "tls": "ssl", "plain": "none", "starttls": "starttls"}.get(sec, sec)
    if sec not in SECURITY:
        raise ValueError(f"{what} must be ssl, starttls or none")
    if sec == "none" and not is_loopback(host):
        raise ValueError(
            f"{what}: an unencrypted connection would send the password in clear text — "
            "only allowed to a server on this machine (localhost). Use SSL/TLS or STARTTLS."
        )
    return sec


def account_from_settings(data: dict[str, Any], *, account_id: str, password: str,
                          smtp_password: str = "") -> Account:
    """Check user-entered settings and build the :class:`Account`
    (ValueError with a readable message)."""
    addr = str(data.get("email") or "").strip()
    if not _EMAIL_RE.match(addr) or not addr.isascii():
        raise ValueError("email must be a plain email address (name@example.com)")
    username = str(data.get("username") or "").strip() or addr
    if len(username) > 320:
        raise ValueError("username is too long")
    if not password:
        raise ValueError("password is required")
    imap_host = clean_host(data.get("imap_host"), "IMAP server")
    imap_sec = _security(data.get("imap_security"), imap_host, "IMAP security")
    imap_port = _port(data.get("imap_port"), DEFAULT_PORTS[("imap", imap_sec)], "IMAP port")
    smtp_host = ""
    smtp_sec, smtp_port = "ssl", 465
    if str(data.get("smtp_host") or "").strip():
        smtp_host = clean_host(data.get("smtp_host"), "SMTP server")
        smtp_sec = _security(data.get("smtp_security"), smtp_host, "SMTP security")
        smtp_port = _port(data.get("smtp_port"), DEFAULT_PORTS[("smtp", smtp_sec)], "SMTP port")
    smtp_user = str(data.get("smtp_username") or "").strip()
    if smtp_user and not smtp_password:
        raise ValueError("a separate SMTP username needs its SMTP password")
    label = " ".join(str(data.get("label") or "").split())[:60]
    display = " ".join(str(data.get("display_name") or "").split())[:120]
    save_sent = data.get("save_sent")
    return Account(
        id=account_id, email=addr, username=username, password=password,
        imap_host=imap_host, imap_port=imap_port, imap_security=imap_sec,
        smtp_host=smtp_host, smtp_port=smtp_port, smtp_security=smtp_sec,
        smtp_username=smtp_user, smtp_password=smtp_password if smtp_user else "",
        label=label, display_name=display,
        save_sent=True if save_sent is None else bool(save_sent),
    )


# ── provider presets ────────────────────────────────────────────────

# domain(s) → settings + a hint shown beside the form. "save_sent": False for
# providers that put SMTP sends into Sent themselves.
_PRESETS: list[dict[str, Any]] = [
    {"name": "Gmail", "domains": ["gmail.com", "googlemail.com"],
     "imap_host": "imap.gmail.com", "smtp_host": "smtp.gmail.com", "smtp_port": 587, "smtp_security": "starttls", "save_sent": False,
     "hint": "Gmail needs an app password (Google Account → Security → 2-Step Verification "
             "→ App passwords). The Google tab connects Gmail without a password."},
    {"name": "iCloud Mail", "domains": ["icloud.com", "me.com", "mac.com"],
     "imap_host": "imap.mail.me.com", "smtp_host": "smtp.mail.me.com", "smtp_port": 587,
     "smtp_security": "starttls", "username_local_part": True,
     "hint": "iCloud needs an app-specific password (appleid.apple.com → Sign-In and "
             "Security). The username is the part before @icloud.com."},
    {"name": "Yahoo Mail", "domains": ["yahoo.com", "ymail.com", "rocketmail.com", "yahoo.co.uk",
                                       "yahoo.de", "yahoo.fr", "yahoo.it", "yahoo.es"],
     "imap_host": "imap.mail.yahoo.com", "smtp_host": "smtp.mail.yahoo.com", "smtp_port": 587, "smtp_security": "starttls", "save_sent": False,
     "hint": "Yahoo needs an app password (Account security → Generate app password)."},
    {"name": "AOL Mail", "domains": ["aol.com"],
     "imap_host": "imap.aol.com", "smtp_host": "smtp.aol.com", "smtp_port": 587, "smtp_security": "starttls", "save_sent": False,
     "hint": "AOL needs an app password (Account security → Generate app password)."},
    {"name": "Fastmail", "domains": ["fastmail.com", "fastmail.fm", "messagingengine.com"],
     "imap_host": "imap.fastmail.com", "smtp_host": "smtp.fastmail.com", "smtp_port": 587, "smtp_security": "starttls",
     "hint": "Fastmail needs an app password (Settings → Privacy & Security → App passwords)."},
    {"name": "Zoho Mail", "domains": ["zoho.com", "zohomail.com", "zoho.eu"],
     "imap_host": "imap.zoho.com", "smtp_host": "smtp.zoho.com", "smtp_port": 587, "smtp_security": "starttls",
     "hint": "Turn IMAP on in Zoho Mail settings; with two-factor sign-in use an app password. "
             "Zoho EU accounts use imap.zoho.eu / smtp.zoho.eu."},
    {"name": "GMX", "domains": ["gmx.com", "gmx.net", "gmx.de", "gmx.at", "gmx.ch"],
     "imap_host": "imap.gmx.net", "smtp_host": "mail.gmx.net", "smtp_port": 587,
     "smtp_security": "starttls", "hint": "Turn on IMAP access in GMX settings first."},
    {"name": "WEB.DE", "domains": ["web.de"],
     "imap_host": "imap.web.de", "smtp_host": "smtp.web.de", "smtp_port": 587,
     "smtp_security": "starttls", "hint": "Turn on IMAP access in WEB.DE settings first."},
    {"name": "mail.com", "domains": ["mail.com", "email.com", "usa.com"],
     "imap_host": "imap.mail.com", "smtp_host": "smtp.mail.com", "smtp_port": 587,
     "smtp_security": "starttls"},
    {"name": "Yandex Mail", "domains": ["yandex.com", "yandex.ru", "ya.ru"],
     "imap_host": "imap.yandex.com", "smtp_host": "smtp.yandex.com",
     "hint": "Yandex needs an app password and IMAP turned on in mail settings."},
    {"name": "Mail.ru", "domains": ["mail.ru", "inbox.ru", "list.ru", "bk.ru"],
     "imap_host": "imap.mail.ru", "smtp_host": "smtp.mail.ru",
     "hint": "Mail.ru needs a password for external apps."},
    {"name": "Proton Mail (Bridge)", "domains": ["proton.me", "protonmail.com", "pm.me", "protonmail.ch"],
     "imap_host": "127.0.0.1", "imap_port": 1143, "imap_security": "starttls",
     "smtp_host": "127.0.0.1", "smtp_port": 1025, "smtp_security": "starttls", "save_sent": False,
     "hint": "Proton Mail works through Proton Mail Bridge running on the same machine as this "
             "Flight Deck; use the username and password Bridge shows (not your Proton password)."},
    {"name": "Hrvatski Telekom", "domains": ["t-com.hr", "htnet.hr"],
     "imap_host": "imap.t-com.hr", "smtp_host": "smtp.t-com.hr"},
    {"name": "Iskon", "domains": ["iskon.hr"],
     "imap_host": "mail.iskon.hr", "smtp_host": "mail.iskon.hr"},
    {"name": "Outlook.com / Microsoft 365", "domains": ["outlook.com", "hotmail.com", "live.com",
                                                       "msn.com", "outlook.de", "hotmail.de"],
     "imap_host": "outlook.office365.com", "smtp_host": "smtp.office365.com", "smtp_port": 587,
     "smtp_security": "starttls", "save_sent": False, "microsoft": True,
     "hint": "Microsoft turned off password sign-in for Outlook.com and most Microsoft 365 "
             "mailboxes — connect it in the Microsoft tab instead."},
]


def _preset_for(domain: str) -> dict[str, Any] | None:
    for p in _PRESETS:
        if domain in p["domains"]:
            return p
    return None


def suggest_settings(address: str) -> dict[str, Any]:
    """Settings to start the form from, for *address*: a built-in preset,
    else the usual ``imap.<domain>`` / ``smtp.<domain>`` guess
    (``source`` says which). Flight Deck may improve on a guess with
    Thunderbird's provider database (:func:`parse_autoconfig`)."""
    addr = str(address or "").strip().lower()
    domain = addr.rsplit("@", 1)[-1] if "@" in addr else addr
    p = _preset_for(domain)
    if p:
        local = addr.split("@", 1)[0] if "@" in addr else ""
        return {
            "source": "preset", "provider": p["name"],
            "username": local if p.get("username_local_part") else addr,
            "imap_host": p["imap_host"], "imap_port": p.get("imap_port", 993),
            "imap_security": p.get("imap_security", "ssl"),
            "smtp_host": p["smtp_host"], "smtp_port": p.get("smtp_port", 465),
            "smtp_security": p.get("smtp_security", "ssl"),
            "save_sent": p.get("save_sent", True), "hint": p.get("hint", ""),
            "microsoft": bool(p.get("microsoft")),
        }
    return {
        "source": "guess", "provider": "", "username": addr,
        "imap_host": f"imap.{domain}" if domain else "", "imap_port": 993, "imap_security": "ssl",
        "smtp_host": f"smtp.{domain}" if domain else "", "smtp_port": 465, "smtp_security": "ssl",
        "save_sent": True, "hint": "", "microsoft": False,
    }


def parse_autoconfig(xml_text: str, address: str) -> dict[str, Any] | None:
    """Settings from a Thunderbird autoconfig document (the ISPDB format), or
    None when it has no usable IMAP server. Placeholders %EMAILADDRESS%,
    %EMAILLOCALPART% and %EMAILDOMAIN% are filled in from *address*."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    local, _, domain = address.partition("@")

    def fill(text: str | None) -> str:
        return (text or "").replace("%EMAILADDRESS%", address).replace(
            "%EMAILLOCALPART%", local).replace("%EMAILDOMAIN%", domain).strip()

    def server(kind: str) -> dict[str, Any] | None:
        best = None
        for node in root.iter("incomingServer" if kind == "imap" else "outgoingServer"):
            if (node.get("type") or "").lower() != kind:
                continue
            sock = (node.findtext("socketType") or "").upper()
            sec = {"SSL": "ssl", "STARTTLS": "starttls"}.get(sock)
            if not sec:
                continue  # plain text: never offered
            entry = {"host": fill(node.findtext("hostname")).lower(),
                     "port": int(fill(node.findtext("port")) or 0) or None,
                     "security": sec, "username": fill(node.findtext("username"))}
            if entry["host"] and (best is None or (sec == "ssl" and best["security"] != "ssl")):
                best = entry
        return best

    imap, smtp = server("imap"), server("smtp")
    if not imap:
        return None
    provider = ""
    for node in root.iter("displayName"):
        provider = (node.text or "").strip()
        break
    out = {
        "source": "autoconfig", "provider": provider, "username": imap["username"] or address,
        "imap_host": imap["host"], "imap_port": imap["port"] or DEFAULT_PORTS[("imap", imap["security"])],
        "imap_security": imap["security"], "smtp_host": "", "smtp_port": 465,
        "smtp_security": "ssl", "save_sent": True, "hint": "", "microsoft": False,
    }
    if smtp:
        out.update(smtp_host=smtp["host"],
                   smtp_port=smtp["port"] or DEFAULT_PORTS[("smtp", smtp["security"])],
                   smtp_security=smtp["security"])
    if out["imap_host"].endswith(("office365.com", "outlook.com")):
        out["microsoft"] = True
        out["hint"] = _preset_for("outlook.com")["hint"]
    return out


# ── connections ─────────────────────────────────────────────────────

_BLOCKED_NETS = [ipaddress.ip_network(n) for n in (
    "169.254.0.0/16", "fe80::/10", "100.100.100.200/32", "fd00:ec2::254/128", "0.0.0.0/8",
)]


def _check_target(host: str, port: int) -> None:
    """Refuse a host that resolves to link-local / metadata space."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise MailError(f"Can't find the server {host} — check the server name.", 502) from None
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        except ValueError:
            continue
        if any(ip in net for net in _BLOCKED_NETS):
            raise MailError(f"{host} resolves to {ip}, which is not a mail server address.", 400)


# How long a setup test waits for a TCP connection before calling a port
# unreachable (every address of the host at once). The real connection after it
# keeps the longer TIMEOUT.
PROBE_TIMEOUT = 8.0


def reachable(host: str, port: int, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether any address of *host*:*port* takes a TCP connection within
    *timeout* — all addresses tried at once, so a host whose IPv6 (or one
    address) is black-holed doesn't cost a timeout per address."""
    import selectors

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    sel = selectors.DefaultSelector()
    socks: list[socket.socket] = []
    try:
        for family, kind, proto, _name, addr in infos:
            try:
                sock = socket.socket(family, kind, proto)
            except OSError:
                continue
            sock.setblocking(False)
            socks.append(sock)
            err = sock.connect_ex(addr)
            if err == 0:
                return True
            sel.register(sock, selectors.EVENT_WRITE)
        deadline = time.monotonic() + timeout
        pending = len(sel.get_map())
        while pending:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            for key, _ev in sel.select(left):
                sock = key.fileobj
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0:  # type: ignore[union-attr]
                    return True
                sel.unregister(sock)
                pending -= 1
        return False
    finally:
        sel.close()
        for sock in socks:
            try:
                sock.close()
            except OSError:
                pass


def _tls(host: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if is_loopback(host):
        # A local bridge (Proton Mail Bridge …) uses a self-signed certificate;
        # the traffic never leaves this machine.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _net_error(exc: BaseException, what: str, host: str, port: int) -> MailError:
    if isinstance(exc, MailError):
        return exc
    if isinstance(exc, ssl.SSLCertVerificationError):
        return MailError(f"{what} {host}: its TLS certificate isn't trusted ({exc.verify_message}).")
    if isinstance(exc, ssl.SSLError):
        return MailError(
            f"{what} {host}:{port}: the TLS handshake failed — check the security setting "
            "(SSL/TLS on 993/465, STARTTLS on 143/587) and the port.")
    if isinstance(exc, socket.gaierror):
        return MailError(f"Can't find the server {host} — check the server name.")
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return MailError(f"{what} {host}:{port} didn't answer in time — check the server and port.")
    if isinstance(exc, ConnectionRefusedError):
        return MailError(f"{what} {host}:{port} refused the connection — check the port.")
    if isinstance(exc, OSError):
        return MailError(f"{what} {host}:{port}: {exc.strerror or exc}")
    return MailError(f"{what} {host}: {exc}")


def _server_text(exc: BaseException) -> str:
    text = exc.args[0] if exc.args else str(exc)
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return " ".join(str(text).split())[:300]


# ── modified UTF-7 (RFC 3501 §5.1.3) ────────────────────────────────


def utf7_encode(name: str) -> str:
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            raw = "".join(buf).encode("utf-16-be")
            out.append("&" + base64.b64encode(raw).decode("ascii").rstrip("=").replace("/", ",") + "-")
            buf.clear()

    for ch in name:
        if 0x20 <= ord(ch) <= 0x7E:
            flush()
            out.append("&-" if ch == "&" else ch)
        else:
            buf.append(ch)
    flush()
    return "".join(out)


def utf7_decode(name: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch != "&":
            out.append(ch)
            i += 1
            continue
        end = name.find("-", i)
        if end < 0:
            out.append(name[i:])
            break
        chunk = name[i + 1:end]
        if not chunk:
            out.append("&")
        else:
            b64 = chunk.replace(",", "/")
            b64 += "=" * (-len(b64) % 4)
            try:
                raw = base64.b64decode(b64, validate=True)
                if not raw or len(raw) % 2:
                    raise ValueError("not UTF-16")
                out.append(raw.decode("utf-16-be"))
            except Exception:
                out.append(name[i:end + 1])  # unreadable: keep it as the server wrote it
        i = end + 1
    return "".join(out)


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ── folders ─────────────────────────────────────────────────────────

_ROLE_FLAGS = {"\\sent": "sent", "\\drafts": "drafts", "\\trash": "trash", "\\junk": "junk",
               "\\archive": "archive", "\\all": "all", "\\flagged": "flagged"}
_ROLE_NAMES = {
    "sent": ("sent", "sent items", "sent messages", "sent mail", "sent-mail", "gesendet",
             "gesendete elemente", "gesendete objekte", "poslano", "poslana pošta", "poslane",
             "envoyés", "éléments envoyés", "enviados", "inviata", "posta inviata", "verzonden"),
    "drafts": ("drafts", "draft", "entwürfe", "skice", "nacrti", "brouillons", "borradores",
               "bozze", "concepten"),
    "trash": ("trash", "deleted", "deleted items", "deleted messages", "bin", "papierkorb",
              "gelöschte elemente", "smeće", "otpad", "corbeille", "papelera", "cestino",
              "prullenbak"),
    "junk": ("junk", "spam", "junk e-mail", "junk email", "bulk mail", "neželjena pošta",
             "courrier indésirable", "correo no deseado", "posta indesiderata"),
    "archive": ("archive", "archives", "archiv", "arhiva", "arhiv", "archivio", "archivo"),
}
# What a person / model may call a role folder.
_ROLE_WORDS = {
    "inbox": "inbox", "sent": "sent", "sent items": "sent", "sent mail": "sent", "outbox": "sent",
    "drafts": "drafts", "draft": "drafts", "trash": "trash", "deleted": "trash",
    "deleted items": "trash", "bin": "trash", "junk": "junk", "spam": "junk",
    "archive": "archive", "all mail": "all", "starred": "flagged", "flagged": "flagged",
}
# A move never goes here, and organizing a thread leaves its messages here alone.
NO_MOVE_ROLES = ("trash", "junk", "drafts", "sent")

_LIST_RE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?P<delim>"(?:\\.|[^"\\])*"|NIL)\s*(?P<name>.*)$', re.S)


@dataclass
class Folder:
    name: str      # decoded, as shown
    raw: str       # as the server knows it (modified UTF-7)
    delim: str
    flags: frozenset[str]
    role: str = ""

    @property
    def selectable(self) -> bool:
        return not ({"\\noselect", "\\nonexistent"} & self.flags)


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return re.sub(r"\\(.)", r"\1", token[1:-1])
    return token


def parse_list(data: Iterable[Any]) -> list[Folder]:
    """LIST response lines (as imaplib returns them) → folders with roles."""
    folders: list[Folder] = []
    for item in data or []:
        if item is None:
            continue
        literal = None
        if isinstance(item, tuple):
            line = item[0].decode("utf-8", "replace")
            literal = item[1].decode("utf-8", "replace")
        else:
            line = item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
        m = _LIST_RE.match(line.strip())
        if not m:
            continue
        flags = frozenset(f.lower() for f in m.group("flags").split())
        delim = m.group("delim")
        delim = "" if delim == "NIL" else _unquote(delim)
        raw = literal if literal is not None else _unquote(m.group("name"))
        if re.fullmatch(r"\{\d+\}", raw.strip()):
            continue
        role = next((_ROLE_FLAGS[f] for f in flags if f in _ROLE_FLAGS), "")
        if raw.upper() == "INBOX":
            role, raw = "inbox", "INBOX"
        folders.append(Folder(utf7_decode(raw), raw, delim, flags, role))
    # Roles by name for servers without SPECIAL-USE — only a role no folder
    # claims by flag, and only an exact (last-part) name.
    taken = {f.role for f in folders if f.role}
    for role, names in _ROLE_NAMES.items():
        if role in taken:
            continue
        for f in sorted(folders, key=lambda f: f.name.count(f.delim or "/")):
            leaf = f.name.rsplit(f.delim, 1)[-1] if f.delim else f.name
            if not f.role and f.selectable and leaf.casefold() in names:
                f.role = role
                break
    return folders


# ── message references ──────────────────────────────────────────────


def make_ref(account_id: str, uidvalidity: int, uid: int, folder: str) -> str:
    return f"{account_id}:{uidvalidity}:{uid}:{folder}"


@dataclass(frozen=True)
class Ref:
    account: str
    uidvalidity: int
    uid: int
    folder: str

    def __str__(self) -> str:
        return make_ref(self.account, self.uidvalidity, self.uid, self.folder)


def parse_ref(value: Any, what: str = "message_id") -> Ref:
    text = str(value or "").strip()
    parts = text.split(":", 3)
    if len(parts) != 4 or not parts[1].isdigit() or not parts[2].isdigit() or not parts[3]:
        raise MailError(
            f"{what} {text[:120]!r} isn't an ID this tool printed (they look like "
            "'work:1700000000:4521:INBOX'). Find the email with list_messages or search first.", 400)
    return Ref(parts[0], int(parts[1]), int(parts[2]), parts[3])


def account_of(value: Any) -> str:
    """The account id an id (message, draft or thread) starts with ("" if none)."""
    text = str(value or "").strip()
    head = text.split(":", 1)[0] if ":" in text else ""
    return head if ACCOUNT_ID_RE.match(head) else ""


_MSGID_RE = re.compile(r"<[^<>\s]+>")


def msgids(value: str) -> list[str]:
    return _MSGID_RE.findall(value or "")


def thread_root(message_id: str, in_reply_to: str, references: str) -> str:
    refs = msgids(references)
    if refs:
        return refs[0]
    irt = msgids(in_reply_to)
    if irt:
        return irt[0]
    own = msgids(message_id)
    return own[0] if own else ""


def parse_thread_id(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    acct, _, root = text.partition(":")
    ids = msgids(root)
    if not acct or not ids:
        raise MailError(
            f"thread_id {text[:120]!r} isn't a Thread ID this tool printed (they look like "
            "'work:<abc@example.com>').", 400)
    return acct, ids[0]


# ── header / body helpers ───────────────────────────────────────────


def decode_header(value: Any) -> str:
    if value is None:
        return ""
    try:
        text = str(email.header.make_header(email.header.decode_header(str(value))))
    except Exception:
        text = str(value)
    return " ".join(text.split())


def _short_date(date_header: str, internal: str = "") -> str:
    for source in (date_header, internal):
        if not source:
            continue
        try:
            dt = email.utils.parsedate_to_datetime(source)
            return dt.strftime("%Y-%m-%d %H:%M %z").strip()
        except Exception:
            continue
    return (date_header or internal or "")[:25]


def _sort_key(date_header: str, internal: str) -> float:
    for source in (date_header, internal):
        try:
            return email.utils.parsedate_to_datetime(source).timestamp()
        except Exception:
            continue
    return 0.0


def body_text(msg: Message) -> tuple[str, bool]:
    """The readable text of *msg* (plain preferred, else HTML as text) and
    whether it came from HTML."""
    plain = html = None
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disp = (part.get("Content-Disposition") or "").lower()
        if disp.startswith("attachment"):
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True) or b""
        except Exception:
            payload = b""
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, "replace")
        except LookupError:
            text = payload.decode("utf-8", "replace")
        if ctype == "text/plain" and plain is None:
            plain = text
        elif ctype == "text/html" and html is None:
            html = text
    if plain and plain.strip():
        return plain.replace("\r\n", "\n").strip(), False
    if html:
        return gmail_compose.html_to_text(html).strip(), True
    return "", False


def _preview(raw: bytes) -> str:
    try:
        msg = email.message_from_bytes(raw)
        text, _ = body_text(msg)
    except Exception:
        return ""
    text = " ".join(text.split())
    return text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")


# ── FETCH parsing ───────────────────────────────────────────────────

_NEW_MSG_RE = re.compile(rb"^\s*\d+\s+\(")


def _fetch_items(data: Iterable[Any]) -> list[tuple[str, bytes | None]]:
    """imaplib FETCH data → ``[(metadata text, literal bytes or None)]``, one
    per message (one literal per message requested)."""
    out: list[list[Any]] = []
    for item in data or []:
        if item is None:
            continue
        if isinstance(item, tuple):
            out.append([item[0].decode("latin-1"), item[1]])
        elif isinstance(item, bytes):
            if _NEW_MSG_RE.match(item):
                out.append([item.decode("latin-1"), None])
            elif out:
                out[-1][0] += item.decode("latin-1")
    return [(m, lit) for m, lit in out]


def _meta(meta: str) -> dict[str, Any]:
    uid = re.search(r"\bUID (\d+)", meta)
    flags = re.search(r"\bFLAGS \(([^)]*)\)", meta)
    internal = re.search(r'\bINTERNALDATE "([^"]+)"', meta)
    size = re.search(r"\bRFC822\.SIZE (\d+)", meta)
    return {
        "uid": int(uid.group(1)) if uid else 0,
        "flags": frozenset(f.lower() for f in (flags.group(1).split() if flags else [])),
        "internal": internal.group(1) if internal else "",
        "size": int(size.group(1)) if size else 0,
    }


_SUMMARY_FIELDS = "DATE FROM TO CC REPLY-TO SUBJECT MESSAGE-ID IN-REPLY-TO REFERENCES CONTENT-TYPE"


# ── search query → IMAP SEARCH ──────────────────────────────────────

_QUERY_RE = re.compile(r'(-?)([a-z_]+):("([^"]*)"|\S+)|(-?)"([^"]*)"|(\S+)', re.I)


def _imap_date(value: str) -> str:
    v = value.strip().replace("/", "-")
    try:
        d = datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise MailError(f"Dates in a search look like 2026-01-31 (got {value!r}).", 400) from None
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}"


def _ago(value: str) -> str:
    m = re.fullmatch(r"(\d+)([dwmy])", value.strip().lower())
    if not m:
        raise MailError(f"newer_than / older_than take 7d, 2w, 3m or 1y (got {value!r}).", 400)
    n, unit = int(m.group(1)), m.group(2)
    days = n * {"d": 1, "w": 7, "m": 30, "y": 365}[unit]
    d = datetime.now(UTC) - timedelta(days=days)
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}"


def _size(value: str) -> str:
    m = re.fullmatch(r"(\d+)([kmg]?)b?", value.strip().lower())
    if not m:
        raise MailError(f"larger / smaller take 500k, 2m … (got {value!r}).", 400)
    return str(int(m.group(1)) * {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[m.group(2)])


def translate_query(query: str) -> list[tuple[bool, str, str | None]]:
    """A Gmail-like query (``from:ana subject:"Q3 report" since:2026-01-01
    is:unread has:attachment -from:noreply invoice``) as
    ``[(negated, IMAP key, value or None)]``."""
    terms: list[tuple[bool, str, str | None]] = []
    for m in _QUERY_RE.finditer(query or ""):
        if m.group(2):
            neg = bool(m.group(1))
            key = m.group(2).lower()
            val = m.group(4) if m.group(4) is not None else m.group(3)
            simple = {"from": "FROM", "to": "TO", "cc": "CC", "bcc": "BCC", "subject": "SUBJECT",
                      "body": "BODY", "text": "TEXT"}
            if key in simple:
                terms.append((neg, simple[key], val))
            elif key in ("since", "after"):
                terms.append((neg, "SINCE", _imap_date(val)))
            elif key == "before":
                terms.append((neg, "BEFORE", _imap_date(val)))
            elif key == "on":
                terms.append((neg, "ON", _imap_date(val)))
            elif key == "newer_than":
                terms.append((neg, "SINCE", _ago(val)))
            elif key == "older_than":
                terms.append((neg, "BEFORE", _ago(val)))
            elif key == "larger":
                terms.append((neg, "LARGER", _size(val)))
            elif key == "smaller":
                terms.append((neg, "SMALLER", _size(val)))
            elif key == "is":
                flag = {"unread": "UNSEEN", "unseen": "UNSEEN", "read": "SEEN", "seen": "SEEN",
                        "flagged": "FLAGGED", "starred": "FLAGGED", "unflagged": "UNFLAGGED",
                        "answered": "ANSWERED", "replied": "ANSWERED", "unanswered": "UNANSWERED",
                        "draft": "DRAFT"}.get(val.lower())
                if not flag:
                    raise MailError(f"Unknown is:{val} — use is:unread, is:read, is:flagged or is:answered.", 400)
                terms.append((neg, flag, None))
            elif key == "has" and val.lower().startswith("attachment"):
                terms.append((neg, "HEADER", "Content-Type\x00multipart/mixed"))
            elif key in ("header",) and "=" in val:
                name, _, hv = val.partition("=")
                terms.append((neg, "HEADER", f"{name}\x00{hv}"))
            else:
                # Not a known key: the whole token is text to find.
                terms.append((neg, "TEXT", m.group(0).lstrip("-")))
        elif m.group(6) is not None:
            terms.append((bool(m.group(5)), "TEXT", m.group(6)))
        elif m.group(7):
            word = m.group(7)
            neg = word.startswith("-") and len(word) > 1
            terms.append((neg, "TEXT", word[1:] if neg else word))
    return terms


def search_args(terms: list[tuple[bool, str, str | None]]) -> tuple[list[Any], bytes | None]:
    """IMAP SEARCH arguments for *terms*, and the literal (non-ASCII text goes
    as a literal — one per command, so it is the last criterion; any further
    non-ASCII value is sent as UTF-8 in quotes, which most servers accept)."""
    if not terms:
        return ["ALL"], None
    ordered = sorted(terms, key=lambda t: bool(t[2] and not t[2].isascii()))
    literal: bytes | None = None
    args: list[Any] = []
    utf8 = any(t[2] and not t[2].isascii() for t in terms)
    if utf8:
        args += ["CHARSET", "UTF-8"]
    for i, (neg, key, val) in enumerate(ordered):
        if neg:
            args.append("NOT")
        if key == "HEADER" and val is not None:
            name, _, hv = val.partition("\x00")
            args += ["HEADER", name, _quote(hv) if hv.isascii() else _quote(hv).encode("utf-8")]
            continue
        args.append(key)
        if val is None:
            continue
        if key in ("SINCE", "BEFORE", "ON", "LARGER", "SMALLER"):
            args.append(val)
        elif val.isascii():
            args.append(_quote(val))
        elif literal is None and i == len(ordered) - 1:
            literal = val.encode("utf-8")
        else:
            args.append(_quote(val).encode("utf-8"))
    return args, literal


# ── the session ─────────────────────────────────────────────────────


class Session:
    """One logged-in IMAP connection to *account*. Use as a context manager."""

    def __init__(self, account: Account) -> None:
        self.account = account
        self.conn: imaplib.IMAP4 | None = None
        self.caps: frozenset[str] = frozenset()
        self._folders: list[Folder] | None = None
        self._selected: tuple[str, bool] | None = None
        self.uidvalidity = 0
        self.exists = 0
        self._prefix: str | None = None

    # connection ------------------------------------------------------

    def __enter__(self) -> Session:
        a = self.account
        _check_target(a.imap_host, a.imap_port)
        try:
            if a.imap_security == "ssl":
                conn: imaplib.IMAP4 = imaplib.IMAP4_SSL(
                    a.imap_host, a.imap_port, ssl_context=_tls(a.imap_host), timeout=TIMEOUT)
            else:
                conn = imaplib.IMAP4(a.imap_host, a.imap_port, timeout=TIMEOUT)
                if a.imap_security == "starttls":
                    if "STARTTLS" not in conn.capabilities:
                        conn.shutdown()
                        raise MailError(
                            f"IMAP server {a.imap_host}:{a.imap_port} doesn't offer STARTTLS — "
                            "try SSL/TLS on port 993.")
                    conn.starttls(ssl_context=_tls(a.imap_host))
        except Exception as exc:
            raise _net_error(exc, "IMAP server", a.imap_host, a.imap_port) from None
        self.conn = conn
        try:
            self._login()
            typ, dat = conn.capability()
            if typ == "OK" and dat and dat[0]:
                self.caps = frozenset(dat[0].decode("ascii", "replace").upper().split())
            else:
                self.caps = frozenset(c.upper() for c in conn.capabilities)
        except Exception:
            self.close()
            raise
        return self

    def _login(self) -> None:
        a, conn = self.account, self.conn
        assert conn is not None
        try:
            if a.username.isascii() and a.password.isascii():
                conn.login(a.username, a.password)
            elif "AUTH=PLAIN" in conn.capabilities:
                creds = ("\0" + a.username + "\0" + a.password).encode("utf-8")
                conn.authenticate("PLAIN", lambda _challenge: creds)
            else:
                raise MailError("This IMAP server can't take a password with non-ASCII characters.", 400)
        except imaplib.IMAP4.error as exc:
            text = _server_text(exc)
            if re.search(r"application-specific password|app password", text, re.I):
                raise MailError(
                    f"{a.username} needs an app password, not the account's normal password — the "
                    f"provider said: {text}. Make one in the account's security settings (for Gmail: "
                    "Google Account → Security → 2-Step Verification → App passwords) and use it "
                    "here.", 401) from None
            raise MailError(
                f"The IMAP server refused the login for {a.username}: {text}. Check "
                "the username and password — many providers need an app password.", 401) from None
        except (OSError, imaplib.IMAP4.abort) as exc:
            raise _net_error(exc, "IMAP server", a.imap_host, a.imap_port) from None

    def close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is None:
            return
        try:
            conn.logout()
        except Exception:
            try:
                conn.shutdown()
            except Exception:
                pass

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _cmd(self, what: str, fn: Callable[..., tuple[str, list[Any]]], *args: Any) -> list[Any]:
        try:
            typ, dat = fn(*args)
        except imaplib.IMAP4.abort as exc:
            raise MailError(f"The IMAP server dropped the connection during {what}: {_server_text(exc)}") from None
        except imaplib.IMAP4.error as exc:
            raise MailError(f"The IMAP server refused {what}: {_server_text(exc)}") from None
        except OSError as exc:
            a = self.account
            raise _net_error(exc, "IMAP server", a.imap_host, a.imap_port) from None
        if typ != "OK":
            text = b" ".join(d for d in dat if isinstance(d, bytes)).decode("utf-8", "replace")
            raise MailError(f"The IMAP server refused {what}: {' '.join(text.split())[:300]}")
        return dat

    # folders ---------------------------------------------------------

    def folders(self) -> list[Folder]:
        if self._folders is None:
            assert self.conn is not None
            dat = self._cmd("listing folders", self.conn.list, '""', "*")
            self._folders = parse_list(dat)
            if not any(f.role == "inbox" for f in self._folders):
                self._folders.insert(0, Folder("INBOX", "INBOX", "/", frozenset(), "inbox"))
        return self._folders

    def role(self, role: str) -> Folder | None:
        """The folder with *role*. When several claim it (a server whose
        config adds a second \\Sent …) the pick must not follow the LIST
        order, which servers don't keep stable: the shallowest, then by name."""
        found = [f for f in self.folders() if f.role == role and f.selectable]
        if not found:
            return None
        return min(found, key=lambda f: (f.name.count(f.delim) if f.delim else 0, f.name.casefold()))

    def folder(self, spec: Any) -> Folder:
        """A folder by name / path / role word ('sent', 'drafts' …)."""
        text = str(spec or "").strip()
        if not text or text.casefold() == "inbox":
            return self.role("inbox") or self.folders()[0]
        fs = [f for f in self.folders() if f.selectable]
        for f in fs:
            if f.name == text or f.raw == text:
                return f
        low = text.casefold()
        for f in fs:
            if f.name.casefold() == low:
                return f
        role = _ROLE_WORDS.get(low)
        if role:
            found = self.role(role)
            if found:
                return found
        # 'Parent/Child' typed with '/' on a '.'-delimited server, or the leaf alone.
        for f in fs:
            if f.delim and f.name.replace(f.delim, "/").casefold() == low:
                return f
        leaves = [f for f in fs if f.delim and f.name.rsplit(f.delim, 1)[-1].casefold() == low]
        if len(leaves) == 1:
            return leaves[0]
        names = ", ".join(f.name for f in fs[:40])
        raise MailError(f"No folder {text!r} in {self.account.name}. Folders: {names}", 404)

    def namespace_prefix(self) -> tuple[str, str]:
        """The personal namespace ``(prefix, delimiter)`` — 'INBOX.' on
        Courier / Cyrus style servers, '' elsewhere."""
        if self._prefix is None:
            prefix, delim = "", ""
            if "NAMESPACE" in self.caps and self.conn is not None:
                try:
                    typ, dat = self.conn.namespace()
                    text = dat[0].decode("utf-8", "replace") if typ == "OK" and dat and dat[0] else ""
                    m = re.match(r'\(\("((?:\\.|[^"\\])*)"\s+(?:"((?:\\.|[^"\\])*)"|NIL)\)', text)
                    if m:
                        prefix, delim = _unquote(f'"{m.group(1)}"'), _unquote(f'"{m.group(2) or ""}"')
                except Exception:
                    pass
            if not delim:
                delim = next((f.delim for f in self.folders() if f.delim), "/")
            self._prefix = prefix
            self._delim = delim
        return self._prefix, self._delim

    def create_folder(self, name: str) -> tuple[Folder, bool]:
        """*name* ('Parent/Child' makes it inside Parent) — ``(folder, created)``."""
        parts = [p.strip() for p in str(name or "").strip().strip("/").split("/")]
        if not parts or not all(parts):
            raise MailError("A folder name is required ('Receipts', or 'Clients/Acme').", 400)
        try:
            existing = self.folder(name)
            return existing, False
        except MailError:
            pass
        prefix, delim = self.namespace_prefix()
        if parts[0].casefold() == "inbox":
            inbox = self.role("inbox")
            parts[0] = inbox.raw if inbox else "INBOX"
            full = delim.join(parts)
        else:
            full = delim.join(parts)
            if prefix and not full.startswith(prefix):
                full = prefix + full
        raw = utf7_encode(full)
        assert self.conn is not None
        # Parents first (most servers create them, some don't).
        segs = raw.split(delim) if delim else [raw]
        for i in range(1, len(segs)):
            parent = delim.join(segs[:i])
            if not any(f.raw == parent for f in self.folders()):
                try:
                    self.conn.create(_quote(parent))
                except Exception:
                    pass
        self._cmd(f"creating folder {name!r}", self.conn.create, _quote(raw))
        try:
            self.conn.subscribe(_quote(raw))
        except Exception:
            pass
        self._folders = None
        return self.folder(utf7_decode(raw)), True

    # selecting / searching / fetching --------------------------------

    def select(self, folder: Folder, *, readonly: bool = True) -> None:
        if self._selected == (folder.raw, readonly):
            return
        assert self.conn is not None
        dat = self._cmd(f"opening {folder.name}", self.conn.select, _quote(folder.raw), readonly)
        try:
            self.exists = int(dat[0] or 0)
        except (TypeError, ValueError):
            self.exists = 0
        uv = self.conn.response("UIDVALIDITY")[1]
        try:
            self.uidvalidity = int(uv[0]) if uv and uv[0] else 0
        except (TypeError, ValueError):
            self.uidvalidity = 0
        self._selected = (folder.raw, readonly)

    def open_ref(self, ref: Ref, *, readonly: bool = True) -> Folder:
        if ref.account != self.account.id:
            raise MailError(f"That ID belongs to inbox {ref.account!r}, not {self.account.id!r}.", 400)
        folder = self.folder(ref.folder)
        self.select(folder, readonly=readonly)
        if ref.uidvalidity and self.uidvalidity and ref.uidvalidity != self.uidvalidity:
            raise MailError(
                f"The folder {folder.name} was renumbered by the server since that ID was made — "
                "find the email again with list_messages or search.", 404)
        return folder

    def search(self, args: list[Any], literal: bytes | None = None) -> list[int]:
        assert self.conn is not None
        if literal is not None:
            self.conn.literal = literal
        dat = self._cmd("the search", self.conn.uid, "SEARCH", *args)
        out: list[int] = []
        for chunk in dat:
            if isinstance(chunk, bytes):
                out.extend(int(x) for x in chunk.split() if x.isdigit())
        return sorted(set(out))

    def fetch_summaries(self, uids: list[int], *, preview: bool = False) -> list[dict[str, Any]]:
        if not uids:
            return []
        assert self.conn is not None
        uidset = ",".join(str(u) for u in uids)
        dat = self._cmd(
            "reading the message list", self.conn.uid, "FETCH", uidset,
            f"(UID FLAGS INTERNALDATE RFC822.SIZE BODY.PEEK[HEADER.FIELDS ({_SUMMARY_FIELDS})])")
        previews: dict[int, str] = {}
        if preview:
            pdat = self._cmd("reading previews", self.conn.uid, "FETCH", uidset,
                             f"(UID BODY.PEEK[]<0.{PREVIEW_BYTES}>)")
            for meta, lit in _fetch_items(pdat):
                if lit is not None:
                    previews[_meta(meta)["uid"]] = _preview(lit)
        folder_name = next((f.name for f in self.folders() if self._selected and f.raw == self._selected[0]), "")
        out = []
        for meta, lit in _fetch_items(dat):
            m = _meta(meta)
            if not m["uid"]:
                continue
            hdr = email.message_from_bytes(lit or b"")
            s = summary_from(hdr, m, self.account.id, self.uidvalidity, folder_name)
            if preview:
                s["preview"] = previews.get(m["uid"], "")
            out.append(s)
        out.sort(key=lambda s: s["ts"], reverse=True)
        return out

    def fetch_raw(self, uid: int) -> tuple[bytes, dict[str, Any]]:
        assert self.conn is not None
        dat = self._cmd("reading the message", self.conn.uid, "FETCH", str(uid), "(UID RFC822.SIZE FLAGS INTERNALDATE)")
        items = _fetch_items(dat)
        if not items:
            raise MailError("Message not found — it may have been moved or deleted. Search for it again.", 404)
        meta = _meta(items[0][0])
        if meta["size"] > MAX_MESSAGE_BYTES:
            raise MailError(
                f"This email is {mail_attachments.human_size(meta['size'])} — too big to open here "
                f"(limit {mail_attachments.human_size(MAX_MESSAGE_BYTES)}).", 413)
        dat = self._cmd("reading the message", self.conn.uid, "FETCH", str(uid), "(UID BODY.PEEK[])")
        for m, lit in _fetch_items(dat):
            if lit is not None:
                return lit, meta
        raise MailError("Message not found — it may have been moved or deleted. Search for it again.", 404)

    def append(self, folder: Folder, data: bytes, flags: str = "(\\Seen)") -> int:
        """Store *data* in *folder*; its UID when the server says (UIDPLUS), else 0."""
        assert self.conn is not None
        dat = self._cmd(f"saving to {folder.name}", self.conn.append, _quote(folder.raw), flags,
                        imaplib.Time2Internaldate(time.time()), data)
        text = b" ".join(d for d in dat if isinstance(d, bytes)).decode("ascii", "replace")
        m = re.search(r"APPENDUID (\d+) (\d+)", text)
        self._selected = None
        return int(m.group(2)) if m else 0

    def store(self, uids: list[int], op: str, flags: str) -> None:
        assert self.conn is not None
        self._cmd("changing flags", self.conn.uid, "STORE", ",".join(map(str, uids)), op, flags)

    def expunge(self, uids: list[int]) -> None:
        assert self.conn is not None
        if "UIDPLUS" in self.caps:
            self._cmd("removing", self.conn.uid, "EXPUNGE", ",".join(map(str, uids)))
        else:
            self._cmd("removing", self.conn.expunge)

    def move(self, uids: list[int], dest: Folder) -> dict[int, int]:
        """Move *uids* from the selected folder to *dest*; ``old → new UID``
        where the server says (COPYUID)."""
        assert self.conn is not None
        uidset = ",".join(map(str, uids))
        if "MOVE" in self.caps:
            dat = self._cmd(f"moving to {dest.name}", self.conn.uid, "MOVE", uidset, _quote(dest.raw))
        else:
            dat = self._cmd(f"copying to {dest.name}", self.conn.uid, "COPY", uidset, _quote(dest.raw))
            self.store(uids, "+FLAGS.SILENT", "(\\Deleted)")
            self.expunge(uids)
        texts = [d.decode("ascii", "replace") for d in dat if isinstance(d, bytes)]
        code = self.conn.response("COPYUID")[1]
        texts += [c.decode("ascii", "replace") if isinstance(c, bytes) else str(c) for c in (code or []) if c]
        for text in texts:
            m = re.search(r"(?:COPYUID )?(\d+) ([\d:,]+) ([\d:,]+)", text)
            if m:
                src, dst = _uidset(m.group(2)), _uidset(m.group(3))
                if len(src) == len(dst):
                    return dict(zip(src, dst))
        return {}

    def find_by_msgid(self, folder: Folder, message_id: str) -> int:
        self.select(folder)
        uids = self.search(["HEADER", "Message-ID", _quote(message_id)])
        return uids[-1] if uids else 0


def _uidset(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":", 1)
            lo, hi = sorted((int(a), int(b)))
            out.extend(range(lo, hi + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def summary_from(hdr: Message, meta: dict[str, Any], account_id: str, uidvalidity: int,
                 folder: str) -> dict[str, Any]:
    mid = decode_header(hdr.get("Message-ID"))
    irt = decode_header(hdr.get("In-Reply-To"))
    refs = decode_header(hdr.get("References"))
    root = thread_root(mid, irt, refs)
    ctype = (hdr.get("Content-Type") or "").lower()
    date = decode_header(hdr.get("Date"))
    return {
        "id": make_ref(account_id, uidvalidity, meta["uid"], folder),
        "account": account_id,
        "folder": folder,
        "thread_id": f"{account_id}:{root}" if root else "",
        "message_id": mid,
        "from": decode_header(hdr.get("From")),
        "to": decode_header(hdr.get("To")),
        "cc": decode_header(hdr.get("Cc")),
        "reply_to": decode_header(hdr.get("Reply-To")),
        "subject": decode_header(hdr.get("Subject")),
        "date": _short_date(date, meta.get("internal", "")),
        "unread": "\\seen" not in meta["flags"],
        "flagged": "\\flagged" in meta["flags"],
        "answered": "\\answered" in meta["flags"],
        "draft": "\\draft" in meta["flags"],
        "has_attachments": "multipart/mixed" in ctype,
        "size": meta.get("size", 0),
        "ts": _sort_key(date, meta.get("internal", "")),
    }


def _strip_private(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*items* as answered (``ts`` stays: Flight Deck merges inboxes by it)."""
    return items


# ── read actions ────────────────────────────────────────────────────


def list_folders(account: Account) -> list[dict[str, Any]]:
    with Session(account) as s:
        return [{"name": f.name, "role": f.role, "selectable": f.selectable} for f in s.folders()]


def list_messages(account: Account, *, folder: str = "", unread_only: bool = False,
                  max_results: int = 10, include_body: bool = False) -> dict[str, Any]:
    with Session(account) as s:
        f = s.folder(folder)
        s.select(f)
        if unread_only:
            uids = s.search(["UNSEEN"])
        else:
            uids = s.search(["ALL"]) if s.exists else []
        total = len(uids)
        items = s.fetch_summaries(uids[-max_results:], preview=include_body)
        return {"folder": f.name, "total": total, "messages": _strip_private(items[:max_results])}


def _search_folders(s: Session, folder: str) -> list[Folder]:
    if str(folder or "").strip().casefold() in ("all", "*", "everywhere"):
        all_mail = s.role("all")
        if all_mail:
            return [all_mail]
        skip = {"trash", "junk"}
        return [f for f in s.folders() if f.selectable and f.role not in skip][:30]
    return [s.folder(folder)]


def search(account: Account, *, query: str, folder: str = "", max_results: int = 10,
           include_body: bool = False) -> dict[str, Any]:
    args, literal = search_args(translate_query(query))
    with Session(account) as s:
        found: list[dict[str, Any]] = []
        total = 0
        folders = _search_folders(s, folder)
        for f in folders:
            s.select(f)
            if not s.exists:
                continue
            uids = s.search(list(args), literal)
            total += len(uids)
            found += s.fetch_summaries(uids[-max_results:], preview=include_body)
        found.sort(key=lambda m: m["ts"], reverse=True)
        where = folders[0].name if len(folders) == 1 else f"{len(folders)} folders"
        return {"folder": where, "total": total, "messages": _strip_private(found[:max_results])}


def _attachment_list(msg: Message) -> list[dict[str, Any]]:
    return [
        {"index": i, "name": fp.name, "size": fp.size, "inline": fp.inline,
         "mime_type": fp.part.get_content_type()}
        for i, fp in enumerate(mail_attachments.message_files(msg), 1)
    ]


def _detail(raw: bytes, meta: dict[str, Any], s: Session, folder: Folder, *,
            max_chars: int = MAX_BODY_CHARS) -> dict[str, Any]:
    msg = mail_attachments.parse_message(raw)
    hdr = email.message_from_bytes(raw)
    out = summary_from(hdr, meta, s.account.id, s.uidvalidity, folder.name)
    text, from_html = body_text(msg)
    out["bcc"] = decode_header(hdr.get("Bcc"))
    out["body"] = text[:max_chars]
    out["body_truncated"] = len(text) > max_chars
    out["body_from_html"] = from_html
    out["attachments"] = _attachment_list(msg)
    out["list_unsubscribe"] = decode_header(hdr.get("List-Unsubscribe"))[:300]
    return out


def read_message(account: Account, ref: Ref) -> dict[str, Any]:
    with Session(account) as s:
        folder = s.open_ref(ref)
        raw, meta = s.fetch_raw(ref.uid)
        meta["uid"] = ref.uid
        return _detail(raw, meta, s, folder)


def _thread_folders(s: Session, *, include_drafts: bool = True) -> list[Folder]:
    all_mail = s.role("all")
    roles = ["inbox", "sent", "archive"] + (["drafts"] if include_drafts else [])
    out = [all_mail] if all_mail else []
    for role in roles:
        f = s.role(role)
        if f and f not in out:
            out.append(f)
    if all_mail and include_drafts:
        d = s.role("drafts")
        if d and d not in out:
            out.append(d)
    return out


def _thread_uids(s: Session, root: str) -> list[int]:
    q = _quote(root)
    return s.search(["OR", "OR", "HEADER", "Message-ID", q, "HEADER", "References", q,
                     "HEADER", "In-Reply-To", q])


# get_thread looks in at most this many folders (an \\All folder, where the
# server has one, holds the whole conversation by itself).
_THREAD_MAX_FOLDERS = 40


def _every_folder(s: Session) -> list[Folder]:
    """Where a whole conversation can be: the \\All folder (Gmail) plus
    Drafts, else every selectable folder but Trash and Junk — people file
    replies into folders of their own."""
    all_mail = s.role("all")
    if all_mail:
        return _thread_folders(s)
    first = [f for f in (s.role(r) for r in ("inbox", "sent", "drafts", "archive")) if f]
    rest = [f for f in s.folders() if f.selectable and f.role not in ("trash", "junk") and f not in first]
    return (first + rest)[:_THREAD_MAX_FOLDERS]


def get_thread(account: Account, root: str, *, max_messages: int = 30) -> dict[str, Any]:
    with Session(account) as s:
        seen: set[str] = set()
        messages: list[dict[str, Any]] = []
        for f in _every_folder(s):
            s.select(f)
            if not s.exists:
                continue
            for uid in _thread_uids(s, root)[-max_messages:]:
                raw, meta = s.fetch_raw(uid)
                meta["uid"] = uid
                d = _detail(raw, meta, s, f, max_chars=MAX_THREAD_BODY_CHARS)
                key = d["message_id"] or d["id"]
                if key in seen:
                    continue
                seen.add(key)
                d["ts"] = _sort_key(decode_header(email.message_from_bytes(raw).get("Date")), meta["internal"])
                messages.append(d)
        messages.sort(key=lambda m: m["ts"])
        if not messages:
            raise MailError("No messages found for that Thread ID.", 404)
        return {"thread_id": f"{account.id}:{root}", "messages": _strip_private(messages[-max_messages:])}


def get_attachments(account: Account, ref: Ref, which: str) -> list[mail_attachments.Attachment]:
    """The file parts *which* names ('1', a name, or 'all')."""
    with Session(account) as s:
        s.open_ref(ref)
        raw, _meta = s.fetch_raw(ref.uid)
    msg = mail_attachments.parse_message(raw)
    files = mail_attachments.message_files(msg)
    if not files:
        raise MailError("This email has no attachments.", 404)
    want = str(which or "").strip()
    if not want:
        raise MailError("Which attachment? Pass attachment='1' (its number from read_message), its name, or 'all'.", 400)
    if want.casefold() == "all":
        chosen = files
    elif want.isdigit() and 1 <= int(want) <= len(files):
        chosen = [files[int(want) - 1]]
    else:
        chosen = [f for f in files if f.name.casefold() == want.casefold()] or \
                 [f for f in files if want.casefold() in f.name.casefold()][:1]
        if not chosen:
            names = ", ".join(f"{i}. {f.name}" for i, f in enumerate(files, 1))
            raise MailError(f"No attachment {want!r}. This email has: {names}", 404)
    out = []
    for fp in chosen:
        if fp.part.get_content_maintype() == "message":
            data = fp.part.as_bytes()
            payload = fp.part.get_payload()
            if isinstance(payload, list) and payload:
                data = payload[0].as_bytes()
        else:
            data = fp.part.get_payload(decode=True) or b""
        out.append(mail_attachments.Attachment(fp.name, fp.part.get_content_type(), data))
    return out


# ── composing ───────────────────────────────────────────────────────


@dataclass
class ReplyContext:
    message_id: str = ""
    references: str = ""
    to_default: str = ""
    cc_all: str = ""
    subject_default: str = ""
    date: str = ""
    ref: Ref | None = None


def reply_context(s: Session, ref: Ref, *, reply_all: bool) -> ReplyContext:
    s.open_ref(ref)
    raw, _meta = s.fetch_raw(ref.uid)
    hdr = email.message_from_bytes(raw)
    mid = msgids(decode_header(hdr.get("Message-ID")))
    mid_s = mid[0] if mid else ""
    refs = " ".join(msgids(decode_header(hdr.get("References"))) + ([mid_s] if mid_s else []))
    subject = decode_header(hdr.get("Subject"))
    if not re.match(r"^\s*re\s*:", subject, re.I):
        subject = f"Re: {subject}".strip()
    own = s.account.email.lower()
    sender = decode_header(hdr.get("Reply-To")) or decode_header(hdr.get("From"))
    to_hdr = decode_header(hdr.get("To"))
    if own in gmail_compose.parse_addresses(decode_header(hdr.get("From"))):
        # Our own sent email: reply to its recipients.
        to_default = to_hdr
    else:
        to_default = sender
    cc = ""
    if reply_all:
        exclude = set(gmail_compose.parse_addresses(to_default)) | {own}
        others = [
            (n, a) for n, a in email.utils.getaddresses([to_hdr, decode_header(hdr.get("Cc"))])
            if a and a.lower() not in exclude
        ]
        cc = ", ".join(email.utils.formataddr((n, a)) for n, a in others)
    return ReplyContext(mid_s, refs, to_default, cc, subject, decode_header(hdr.get("Date")), ref)


def build_message(account: Account, *, to: str, cc: str, bcc: str, subject: str, body: str,
                  html_body: str, in_reply_to: str = "", references: str = "",
                  attachments: list[mail_attachments.Attachment] | None = None) -> tuple[bytes, str]:
    """``(RFC 822 bytes, Message-ID)`` from *account*."""
    domain = account.email.rsplit("@", 1)[-1]
    mid = email.utils.make_msgid(domain=domain)
    data = gmail_compose.build_message_bytes(
        to=to, cc=cc, bcc=bcc, subject=subject, body=body, html_body=html_body,
        in_reply_to=in_reply_to, references=references, attachments=attachments,
        from_addr=account.from_header(), date=email.utils.formatdate(localtime=True), message_id=mid,
    )
    return data, mid


def without_bcc(data: bytes) -> bytes:
    msg = mail_attachments.parse_message(data)
    if msg.get("Bcc") is None:
        return data
    del msg["Bcc"]
    return msg.as_bytes()


def drafts_folder(s: Session) -> Folder:
    found = s.role("drafts")
    if found:
        return found
    folder, _created = s.create_folder("Drafts")
    folder.role = "drafts"
    return folder


def save_draft(s: Session, data: bytes, message_id: str) -> str:
    f = drafts_folder(s)
    uid = s.append(f, data, "(\\Draft \\Seen)")
    s.select(f)
    if not uid:
        uid = s.find_by_msgid(f, message_id)
    return make_ref(s.account.id, s.uidvalidity, uid, f.name) if uid else ""


def delete_message(s: Session, ref: Ref) -> None:
    s.open_ref(ref, readonly=False)
    s.store([ref.uid], "+FLAGS.SILENT", "(\\Deleted)")
    s.expunge([ref.uid])


def list_drafts(account: Account, *, query: str = "", max_results: int = 10) -> dict[str, Any]:
    with Session(account) as s:
        f = s.role("drafts")
        if not f:
            return {"folder": "Drafts", "total": 0, "messages": []}
        s.select(f)
        if not s.exists:
            return {"folder": f.name, "total": 0, "messages": []}
        args, literal = search_args(_draft_terms(query))
        uids = s.search(args, literal)
        items = s.fetch_summaries(uids[-max_results:], preview=True)
        return {"folder": f.name, "total": len(uids), "messages": _strip_private(items[:max_results])}


def _draft_terms(query: str) -> list[tuple[bool, str, str | None]]:
    q = (query or "").strip()
    if not q:
        return []
    if ":" in q:
        return translate_query(q)
    return [(False, "SUBJECT", q)]


# ── the repeat check ────────────────────────────────────────────────


def find_repeats(s: Session, *, to: str, cc: str, bcc: str, subject: str,
                 reply_msgid: str = "", days: int = 14) -> list[dict[str, str]]:
    """The same email already in Drafts (any age) or Sent (the last *days*):
    for a reply, one answering the same message; else the same recipient and
    a matching subject (:func:`gmail_compose.subjects_match`)."""
    addrs = gmail_compose.parse_addresses(to, cc, bcc)
    if not reply_msgid and (not addrs or not subject.strip()):
        return []
    matches: list[dict[str, str]] = []
    places = [("draft", s.role("drafts"), None)]
    if days > 0:
        since = datetime.now(UTC) - timedelta(days=days)
        places.append(("sent", s.role("sent"), f"{since.day:02d}-{_MONTHS[since.month - 1]}-{since.year}"))
    for kind, folder, since_s in places:
        if folder is None:
            continue
        try:
            s.select(folder)
            if not s.exists:
                continue
            base = ["SINCE", since_s] if since_s else []
            if reply_msgid:
                uids = s.search(base + ["HEADER", "In-Reply-To", _quote(reply_msgid)])
            else:
                uids_set: set[int] = set()
                for addr in addrs[:3]:
                    uids_set.update(s.search(base + ["TO", _quote(addr)]))
                uids = sorted(uids_set)
            for m in s.fetch_summaries(uids[-10:]):
                if not reply_msgid:
                    theirs = set(gmail_compose.parse_addresses(m["to"], m["cc"]))
                    if not theirs & set(addrs) or not gmail_compose.subjects_match(subject, m["subject"]):
                        continue
                matches.append({
                    "kind": kind, "draft_id": m["id"] if kind == "draft" else "",
                    "message_id": m["id"], "to": m["to"], "subject": m["subject"], "date": m["date"],
                })
        except MailError:
            continue
    return matches


def repeat_refusal(matches: list[dict[str, str]], *, lead: str, reply: bool = False) -> str:
    """Like :func:`gmail_compose.repeat_refusal`, with imap_mail's actions."""
    first = matches[0]
    more = f" and {len(matches) - 1} more like it" if len(matches) > 1 else ""
    draft_id = next((m["draft_id"] for m in matches if m.get("kind") == "draft" and m.get("draft_id")), "")
    text = (f"{lead} {gmail_compose.REPEAT_MARK}{gmail_compose.describe_repeat(first)}{more}. "
            "This email is already drafted or sent, so no new copy was made. Tell the user, "
            "with that Draft ID or sent date.")
    if draft_id:
        text += (f" To send that draft use imap_mail action=send_draft draft_id={draft_id}; "
                 f"to change it use action=update_draft draft_id={draft_id}.")
    return text + gmail_compose.repeat_way_past(reply=reply)


# ── SMTP ────────────────────────────────────────────────────────────


def _smtp_connect(account: Account) -> smtplib.SMTP:
    a = account
    if not a.smtp_host:
        raise MailError(f"{a.name} is set up to receive only — it has no SMTP server for sending.", 400)
    _check_target(a.smtp_host, a.smtp_port)
    try:
        if a.smtp_security == "ssl":
            conn: smtplib.SMTP = smtplib.SMTP_SSL(a.smtp_host, a.smtp_port, timeout=TIMEOUT,
                                                  context=_tls(a.smtp_host))
        else:
            conn = smtplib.SMTP(a.smtp_host, a.smtp_port, timeout=TIMEOUT)
            conn.ehlo()
            if a.smtp_security == "starttls":
                if not conn.has_extn("starttls"):
                    conn.close()
                    raise MailError(f"SMTP server {a.smtp_host}:{a.smtp_port} doesn't offer STARTTLS — "
                                    "try SSL/TLS on port 465.")
                conn.starttls(context=_tls(a.smtp_host))
                conn.ehlo()
    except Exception as exc:
        raise _net_error(exc, "SMTP server", a.smtp_host, a.smtp_port) from None
    user, password = a.smtp_login()
    if user:
        try:
            conn.login(user, password)
        except smtplib.SMTPAuthenticationError as exc:
            _close(conn)
            text = exc.smtp_error.decode("utf-8", "replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
            raise MailError(f"The SMTP server refused the login for {user}: {' '.join(text.split())[:300]}. "
                            "Check the username and password (or app password).", 401) from None
        except smtplib.SMTPNotSupportedError:
            _close(conn)
            raise MailError(f"SMTP server {a.smtp_host} doesn't accept a login on this connection — "
                            "check the security setting.") from None
        except UnicodeEncodeError:
            _close(conn)
            raise MailError("This SMTP server can't take a password with non-ASCII characters.", 400) from None
        except (smtplib.SMTPException, OSError) as exc:
            _close(conn)
            raise _net_error(exc, "SMTP server", a.smtp_host, a.smtp_port) from None
    return conn


def _close(conn: smtplib.SMTP) -> None:
    try:
        conn.quit()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def smtp_deliver(account: Account, recipients: list[str], data: bytes) -> str:
    """Send *data* to *recipients* (envelope) from *account*'s address; the
    server's answer. :class:`MailError` = not sent; :class:`SendOutcomeUnknown`
    = handed over but never confirmed."""
    conn = _smtp_connect(account)
    try:
        try:
            code, resp = conn.mail(account.email)
            if code != 250:
                raise MailError(f"The SMTP server refused the sender {account.email}: "
                                f"{resp.decode('utf-8', 'replace')[:200]}. Nothing was sent.")
            refused = []
            for r in recipients:
                code, resp = conn.rcpt(r)
                if code not in (250, 251):
                    refused.append(f"{r} ({code} {resp.decode('utf-8', 'replace')[:120]})")
            if refused:
                conn.rset()
                raise MailError("The SMTP server refused: " + "; ".join(refused) + ". Nothing was sent.", 400)
        except MailError:
            raise
        except (smtplib.SMTPException, OSError) as exc:
            raise MailError(f"SMTP error before sending ({_server_text(exc)}). Nothing was sent.") from None
        try:
            code, resp = conn.data(data)
        except smtplib.SMTPDataError as exc:
            text = exc.smtp_error.decode("utf-8", "replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
            raise MailError(f"The SMTP server refused the email ({exc.smtp_code} {text[:200]}). Nothing was sent.") from None
        except (smtplib.SMTPServerDisconnected, OSError, smtplib.SMTPException) as exc:
            raise SendOutcomeUnknown(
                f"The SMTP server took the email but never confirmed it ({type(exc).__name__}) — it "
                "may have been sent.") from None
        return resp.decode("utf-8", "replace")[:200]
    finally:
        _close(conn)


def test_smtp(account: Account) -> None:
    _close(_smtp_connect(account))


# The other standard way in when the typed one can't be reached: implicit TLS
# and STARTTLS ports (many hosts block outgoing 465 and 25, few block 587).
_ALTERNATIVES = {
    "imap": [(993, "ssl"), (143, "starttls")],
    "smtp": [(587, "starttls"), (465, "ssl")],
}


def _unreachable(kind: str, host: str, port: int) -> str:
    what = "IMAP" if kind == "imap" else "SMTP"
    return (f"{what} server {host}:{port} can't be reached from this Flight Deck (no answer within "
            f"{PROBE_TIMEOUT:.0f} s). The server or port may be wrong, or this deck's hosting "
            "provider blocks the port — many block outgoing 465 and 25.")


def _login_check(kind: str, account: Account) -> dict[str, Any]:
    if kind == "imap":
        with Session(account) as s:
            fs = s.folders()
            return {"ok": True, "folders": len(fs), "roles": sorted({f.role for f in fs if f.role})}
    test_smtp(account)
    return {"ok": True}


def _with(account: Account, kind: str, port: int, security: str) -> Account:
    import dataclasses

    if kind == "imap":
        return dataclasses.replace(account, imap_port=port, imap_security=security)
    return dataclasses.replace(account, smtp_port=port, smtp_security=security)


def _check_server(kind: str, account: Account) -> dict[str, Any]:
    """Probe, then log in, to one server. When the typed port can't be
    reached, the other standard port is tried too: ``alternative`` says how
    that went (``{"port", "security", "ok", "error"?}``)."""
    host = account.imap_host if kind == "imap" else account.smtp_host
    port = account.imap_port if kind == "imap" else account.smtp_port
    try:
        _check_target(host, port)
    except MailError as exc:
        return {"ok": False, "error": exc.message}
    # The typed port and the other standard ones are probed at the same time,
    # so a blocked port costs one PROBE_TIMEOUT, not one per port.
    alts = [] if is_loopback(host) else [
        (p, sec) for p, sec in _ALTERNATIVES[kind] if p != port]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1 + len(alts)) as pool:
        probes = {p: pool.submit(reachable, host, p) for p in [port] + [a[0] for a in alts]}
        answers = {p: f.result() for p, f in probes.items()}
    if not answers[port]:
        out: dict[str, Any] = {"ok": False, "error": _unreachable(kind, host, port), "unreachable": True}
        for alt_port, alt_sec in alts:
            if not answers.get(alt_port):
                continue
            alt: dict[str, Any] = {"port": alt_port, "security": alt_sec}
            try:
                alt.update(_login_check(kind, _with(account, kind, alt_port, alt_sec)))
            except MailError as exc:
                alt.update(ok=False, error=exc.message)
            out["alternative"] = alt
            label = "SSL/TLS" if alt_sec == "ssl" else "STARTTLS"
            out["error"] += (f" Port {alt_port} with {label} does answer"
                             + (" and works — use that." if alt.get("ok") else "."))
            break
        return out
    try:
        return _login_check(kind, account)
    except MailError as exc:
        return {"ok": False, "error": exc.message}


def test_account(account: Account) -> dict[str, Any]:
    """Log in to IMAP and SMTP (when set) at the same time — nothing is
    sent or changed. A port that doesn't answer fails within
    :data:`PROBE_TIMEOUT`, with the other standard port tried for it."""
    from concurrent.futures import ThreadPoolExecutor

    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        imap = pool.submit(_check_server, "imap", account)
        smtp = pool.submit(_check_server, "smtp", account) if account.smtp_host else None
        out["imap"] = imap.result()
        out["smtp"] = smtp.result() if smtp else {"ok": True, "skipped": True}
    out["ok"] = bool(out["imap"]["ok"] and out["smtp"]["ok"])
    return out


def apply_alternatives(account: Account, result: dict[str, Any]) -> tuple[Account, list[str]]:
    """*account* with each server switched to the working ``alternative``
    the test found for it (the typed port unreachable, the other one logged
    in), and a line per switch for the person."""
    notes: list[str] = []
    for kind in ("imap", "smtp"):
        r = result.get(kind) or {}
        alt = r.get("alternative") or {}
        if r.get("ok") or not alt.get("ok"):
            continue
        old_port = account.imap_port if kind == "imap" else account.smtp_port
        account = _with(account, kind, int(alt["port"]), str(alt["security"]))
        label = "SSL/TLS" if alt["security"] == "ssl" else "STARTTLS"
        notes.append(f"{kind.upper()}: port {old_port} doesn't answer from this Flight Deck, so port "
                     f"{alt['port']} with {label} is used.")
        result[kind] = {k: v for k, v in alt.items() if k not in ("port", "security")}
    result["ok"] = bool(result["imap"].get("ok") and result["smtp"].get("ok"))
    return account, notes


def after_send(account: Account, data: bytes, *, answered: Ref | None,
               draft: Ref | None) -> list[str]:
    """What follows a confirmed send: the copy in Sent (unless the provider
    files it), \\Answered on the email it replies to, the sent draft removed.
    Each is best effort — the email is out; the notes say what didn't work."""
    notes: list[str] = []
    try:
        with Session(account) as s:
            if account.save_sent:
                try:
                    sent = s.role("sent")
                    if sent is None:
                        sent, _ = s.create_folder("Sent")
                    s.append(sent, data, "(\\Seen)")
                except MailError as exc:
                    notes.append(f"the copy in Sent wasn't saved ({exc.message})")
            if answered is not None:
                try:
                    s.open_ref(answered, readonly=False)
                    s.store([answered.uid], "+FLAGS.SILENT", "(\\Answered)")
                except MailError:
                    pass
            if draft is not None:
                try:
                    delete_message(s, draft)
                except MailError as exc:
                    notes.append(f"the draft is still in Drafts ({exc.message})")
    except MailError as exc:
        notes.append(f"the mailbox couldn't be updated after sending ({exc.message})")
    return notes


# ── organizing ──────────────────────────────────────────────────────


def _group(refs: list[Ref]) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for r in refs:
        out.setdefault((r.folder, r.uidvalidity), []).append(r.uid)
    return out


def expand_threads(s: Session, roots: list[str]) -> list[Ref]:
    """The Inbox messages of each thread (organizing a conversation acts on
    what is in the Inbox; Sent, Drafts, Trash and Junk stay as they are)."""
    inbox = s.role("inbox") or s.folder("INBOX")
    s.select(inbox)
    out: list[Ref] = []
    for root in roots:
        for uid in _thread_uids(s, root):
            out.append(Ref(s.account.id, s.uidvalidity, uid, inbox.name))
    return out


def archive_folder(s: Session) -> Folder:
    for role in ("archive", "all"):
        f = s.role(role)
        if f:
            return f
    folder, _ = s.create_folder("Archive")
    return folder


def move(account: Account, refs: list[Ref], roots: list[str], dest_spec: str, *,
         archive: bool = False) -> dict[str, Any]:
    with Session(account) as s:
        dest = archive_folder(s) if archive else s.folder(dest_spec)
        if not archive and dest.role in NO_MOVE_ROLES:
            raise MailError(f"Not moved: {dest.name} is the {dest.role} folder — a move never goes to "
                            "Trash, Junk, Drafts or Sent. Nothing was changed.", 400)
        targets = list(refs) + (expand_threads(s, roots) if roots else [])
        pairs: list[tuple[str, int]] = []   # (old id, new UID or 0)
        skipped: list[str] = []
        for (folder_name, uv), uids in _group(targets).items():
            src = s.folder(folder_name)
            if src.raw == dest.raw:
                skipped += [make_ref(account.id, uv, u, folder_name) for u in uids]
                continue
            s.select(src, readonly=False)
            if uv and s.uidvalidity and uv != s.uidvalidity:
                skipped += [make_ref(account.id, uv, u, folder_name) for u in uids]
                continue
            uids = sorted(set(uids))
            mapping = s.move(uids, dest)
            pairs += [(make_ref(account.id, uv, u, folder_name), mapping.get(u, 0)) for u in uids]
        dest_uv = 0
        if any(new for _old, new in pairs):
            s.select(dest)
            dest_uv = s.uidvalidity
        moved = [(old, make_ref(account.id, dest_uv, new, dest.name) if new else "") for old, new in pairs]
        return {"destination": dest.name, "moved": moved, "skipped": skipped}


def set_seen(account: Account, refs: list[Ref], roots: list[str], seen: bool) -> dict[str, Any]:
    with Session(account) as s:
        targets = list(refs) + (expand_threads(s, roots) if roots else [])
        done: list[str] = []
        for (folder_name, uv), uids in _group(targets).items():
            s.select(s.folder(folder_name), readonly=False)
            if uv and s.uidvalidity and uv != s.uidvalidity:
                continue
            s.store(sorted(set(uids)), "+FLAGS.SILENT" if seen else "-FLAGS.SILENT", "(\\Seen)")
            done += [make_ref(account.id, uv, u, folder_name) for u in sorted(set(uids))]
        return {"changed": done}


def create_folder(account: Account, name: str) -> dict[str, Any]:
    with Session(account) as s:
        folder, created = s.create_folder(name)
        return {"folder": folder.name, "created": created}

