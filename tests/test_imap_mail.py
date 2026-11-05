"""captain_claw.imap_mail — the IMAP/SMTP core Flight Deck runs for the
Email (IMAP) connection. No network: the protocol pieces are tested on the
shapes imaplib hands back, SMTP through a scripted connection.

(tests/test_imap_mail_live.py runs the same code against a real server when
CAPTAIN_CLAW_IMAP_LIVE is set.)
"""

import email
import email.policy
import smtplib
import socket

import pytest

from captain_claw import gmail_compose
from captain_claw import imap_mail as m
from captain_claw.mail_attachments import Attachment


def _acct(**over):
    data = {"email": "ana@firma.hr", "imap_host": "imap.firma.hr", "smtp_host": "smtp.firma.hr"}
    data.update(over)
    return m.account_from_settings(data, account_id="work", password="pw")


# ── modified UTF-7 ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,encoded", [
    ("INBOX", "INBOX"),
    ("Računi", "Ra&AQ0-uni"),
    ("Entwürfe", "Entw&APw-rfe"),
    ("Tom & Jerry", "Tom &- Jerry"),
    ("Smeće/Šibenik", "Sme&AQc-e/&AWA-ibenik"),
    ("日本語", "&ZeVnLIqe-"),
])
def test_utf7_known_values(name, encoded):
    assert m.utf7_encode(name) == encoded
    assert m.utf7_decode(encoded) == name


def test_utf7_survives_emoji_and_bad_input():
    assert m.utf7_decode(m.utf7_encode("Projekti 🚀 2026")) == "Projekti 🚀 2026"
    assert m.utf7_decode("broken&@@@-x") == "broken&@@@-x"
    assert m.utf7_decode("no-end&AQ0") == "no-end&AQ0"


# ── LIST parsing and folder roles ───────────────────────────────────


LIST_DATA = [
    b'(\\HasNoChildren) "/" "INBOX"',
    b'(\\HasNoChildren \\Sent) "/" "Poslano"',
    b'(\\HasNoChildren \\Drafts) "/" Skice',
    b'(\\HasChildren \\Noselect) "/" "[Gmail]"',
    b'(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"',
    (b'(\\HasNoChildren) "/" {10}', b'Sme&AQc-e/x'),
    b'(\\HasNoChildren) "/" "Klijenti/Ra&AQ0-uni"',
    None,
]


def test_parse_list_roles_from_flags_and_literals():
    fs = {f.name: f for f in m.parse_list(LIST_DATA)}
    assert fs["INBOX"].role == "inbox"
    assert fs["Poslano"].role == "sent"
    assert fs["Skice"].role == "drafts"
    assert fs["[Gmail]/All Mail"].role == "all"
    assert not fs["[Gmail]"].selectable
    assert "Smeće/x" in fs and fs["Smeće/x"].raw == "Sme&AQc-e/x"
    assert fs["Klijenti/Računi"].raw == "Klijenti/Ra&AQ0-uni"


def test_parse_list_roles_by_name_without_special_use():
    fs = {f.name: f for f in m.parse_list([
        b'() "." INBOX', b'() "." INBOX.Sent', b'() "." "INBOX.Deleted Messages"',
        b'() "." INBOX.Archive', b'() "." "INBOX.Junk"', b'() "." INBOX.Drafts',
        b'() "." "INBOX.Projects.Sent"',
    ])}
    assert fs["INBOX.Sent"].role == "sent"
    assert fs["INBOX.Deleted Messages"].role == "trash"
    assert fs["INBOX.Archive"].role == "archive"
    assert fs["INBOX.Junk"].role == "junk"
    assert fs["INBOX.Drafts"].role == "drafts"
    # A deeper folder of the same name doesn't steal the role.
    assert fs["INBOX.Projects.Sent"].role == ""


def test_name_fallback_never_overrides_a_flag():
    fs = {f.name: f for f in m.parse_list([
        b'(\\Sent) "/" "Gesendet"', b'() "/" "Sent"', b'() "/" INBOX'])}
    assert fs["Gesendet"].role == "sent"
    assert fs["Sent"].role == ""


def _session_with(folders):
    s = m.Session(_acct())
    s._folders = folders
    return s


def test_role_pick_is_stable_whatever_the_list_order():
    a = m.parse_list([b'(\\Drafts) "/" Skice', b'(\\Drafts) "/" Drafts', b'() "/" INBOX'])
    b = list(reversed(a))
    assert _session_with(a).role("drafts").name == _session_with(b).role("drafts").name == "Drafts"


def test_folder_lookup_by_name_role_word_path_and_leaf():
    s = _session_with(m.parse_list([
        b'() "." INBOX', b'(\\Sent) "." "Sent Items"', b'() "." "Clients"', b'() "." "Clients.Acme"']))
    assert s.folder("").name == "INBOX"
    assert s.folder("sent").name == "Sent Items"
    assert s.folder("SENT ITEMS").name == "Sent Items"
    assert s.folder("Clients/Acme").name == "Clients.Acme"
    assert s.folder("acme").name == "Clients.Acme"
    with pytest.raises(m.MailError) as exc:
        s.folder("Nope")
    assert exc.value.status == 404 and "Clients.Acme" in exc.value.message


# ── search queries ──────────────────────────────────────────────────


def test_translate_query_keys():
    terms = m.translate_query('from:ana subject:"Q3 report" since:2026-01-31 is:unread has:attachment '
                              '-from:noreply invoice newer_than:7d larger:2m')
    keys = [(neg, key) for neg, key, _ in terms]
    assert (False, "FROM") in keys and (False, "SUBJECT") in keys and (False, "SINCE") in keys
    assert (False, "UNSEEN") in keys and (True, "FROM") in keys and (False, "TEXT") in keys
    assert (False, "LARGER") in keys and (False, "HEADER") in keys
    assert ("SUBJECT", "Q3 report") in [(k, v) for _, k, v in terms]
    assert ("SINCE", "31-Jan-2026") in [(k, v) for _, k, v in terms]
    assert ("LARGER", str(2 * 1024 * 1024)) in [(k, v) for _, k, v in terms]


@pytest.mark.parametrize("query", ["since:31.01.2026", "is:important", "newer_than:soon", "larger:big"])
def test_translate_query_refuses_unreadable_values(query):
    with pytest.raises(m.MailError) as exc:
        m.translate_query(query)
    assert exc.value.status == 400


def test_search_args_ascii_and_literal():
    args, lit = m.search_args(m.translate_query("from:ana is:unread"))
    assert args == ["FROM", '"ana"', "UNSEEN"] and lit is None
    args, lit = m.search_args(m.translate_query("račun from:ana"))
    assert args[:2] == ["CHARSET", "UTF-8"]
    # The non-ASCII value goes last, as the literal.
    assert args[-1] == "TEXT" and lit == "račun".encode()
    args, lit = m.search_args(m.translate_query("-subject:škola"))
    assert args[-2:] == ["NOT", "SUBJECT"] and lit == "škola".encode()
    assert m.search_args([]) == (["ALL"], None)


def test_search_args_quotes_dangerous_text():
    args, _ = m.search_args(m.translate_query('subject:"a\\"b" text:x)'))
    assert '"a\\\\\\"b"' in args or '"a\\"b"' not in args  # quotes escaped, never raw


# ── IDs ─────────────────────────────────────────────────────────────


def test_refs_round_trip_with_colons_in_folder():
    ref = m.parse_ref("work:1700000000:4521:INBOX.Clients: Acme")
    assert (ref.account, ref.uidvalidity, ref.uid, ref.folder) == ("work", 1700000000, 4521, "INBOX.Clients: Acme")
    assert str(ref) == "work:1700000000:4521:INBOX.Clients: Acme"
    assert m.account_of("work:1:2:INBOX") == "work"
    assert m.account_of("<not-an-id@x>") == ""


@pytest.mark.parametrize("bad", ["", "4521", "work:abc:1:INBOX", "work:1:2:", "work:1:2"])
def test_parse_ref_refuses(bad):
    with pytest.raises(m.MailError):
        m.parse_ref(bad)


def test_thread_ids():
    assert m.thread_root("<c@x>", "<b@x>", "<a@x> <b@x>") == "<a@x>"
    assert m.thread_root("<c@x>", "<b@x>", "") == "<b@x>"
    assert m.thread_root("<c@x>", "", "") == "<c@x>"
    assert m.parse_thread_id("support:<a@x>") == ("support", "<a@x>")
    with pytest.raises(m.MailError):
        m.parse_thread_id("support:nothing")


# ── account settings ────────────────────────────────────────────────


def test_account_defaults_ports_from_security():
    a = _acct(imap_security="starttls", smtp_security="starttls")
    assert (a.imap_port, a.smtp_port) == (143, 587)
    a = _acct()
    assert (a.imap_port, a.smtp_port, a.username) == (993, 465, "ana@firma.hr")


@pytest.mark.parametrize("over,needle", [
    ({"imap_security": "none"}, "clear text"),
    ({"imap_host": "https://imap.firma.hr"}, "not a URL"),
    ({"imap_host": "imap firma"}, "valid host"),
    ({"imap_port": 70000}, "65535"),
    ({"email": "not-an-address"}, "email"),
    ({"smtp_username": "relay-user"}, "SMTP password"),
])
def test_account_settings_refused(over, needle):
    with pytest.raises(ValueError) as exc:
        _acct(**over)
    assert needle in str(exc.value)


def test_plain_text_only_to_this_machine():
    a = _acct(imap_host="127.0.0.1", imap_security="none", smtp_host="localhost", smtp_security="none")
    assert a.imap_security == a.smtp_security == "none"


def test_receive_only_and_separate_smtp_login():
    a = _acct(smtp_host="")
    assert not a.can_send
    b = m.account_from_settings({"email": "a@b.co", "imap_host": "imap.b.co", "smtp_host": "smtp.relay.io",
                                 "smtp_username": "apikey"}, account_id="x", password="imap", smtp_password="relay")
    assert b.smtp_login() == ("apikey", "relay")
    assert _acct().smtp_login() == ("ana@firma.hr", "pw")


def test_password_required():
    with pytest.raises(ValueError):
        m.account_from_settings({"email": "a@b.co", "imap_host": "imap.b.co"}, account_id="x", password="")


# ── presets and autoconfig ──────────────────────────────────────────


def test_presets():
    g = m.suggest_settings("me@gmail.com")
    assert g["source"] == "preset" and g["imap_host"] == "imap.gmail.com" and g["save_sent"] is False
    i = m.suggest_settings("me@icloud.com")
    assert i["username"] == "me" and i["smtp_security"] == "starttls"
    o = m.suggest_settings("me@outlook.com")
    assert o["microsoft"] and "Microsoft tab" in o["hint"]
    x = m.suggest_settings("ivana@firma.hr")
    assert x["source"] == "guess" and x["imap_host"] == "imap.firma.hr"


AUTOCONFIG = """<?xml version="1.0"?>
<clientConfig version="1.1"><emailProvider id="example.org">
  <displayName>Example Mail</displayName>
  <incomingServer type="imap"><hostname>imap.example.org</hostname><port>143</port>
    <socketType>STARTTLS</socketType><username>%EMAILLOCALPART%</username></incomingServer>
  <incomingServer type="imap"><hostname>imap.example.org</hostname><port>993</port>
    <socketType>SSL</socketType><username>%EMAILADDRESS%</username></incomingServer>
  <incomingServer type="pop3"><hostname>pop.example.org</hostname><port>995</port>
    <socketType>SSL</socketType></incomingServer>
  <outgoingServer type="smtp"><hostname>smtp.%EMAILDOMAIN%</hostname><port>25</port>
    <socketType>plain</socketType></outgoingServer>
  <outgoingServer type="smtp"><hostname>smtp.%EMAILDOMAIN%</hostname><port>587</port>
    <socketType>STARTTLS</socketType></outgoingServer>
</emailProvider></clientConfig>"""


def test_parse_autoconfig_prefers_ssl_skips_plain():
    s = m.parse_autoconfig(AUTOCONFIG, "ana@example.org")
    assert s["provider"] == "Example Mail"
    assert (s["imap_host"], s["imap_port"], s["imap_security"], s["username"]) == \
        ("imap.example.org", 993, "ssl", "ana@example.org")
    assert (s["smtp_host"], s["smtp_port"], s["smtp_security"]) == ("smtp.example.org", 587, "starttls")
    assert m.parse_autoconfig("<not xml", "a@b.c") is None
    assert m.parse_autoconfig("<clientConfig/>", "a@b.c") is None


# ── FETCH responses ─────────────────────────────────────────────────


def test_fetch_items_literal_then_trailer_and_gmail_order():
    data = [
        (b'1 (UID 7 FLAGS (\\Seen) INTERNALDATE "05-Nov-2026 09:00:00 +0100" RFC822.SIZE 900 '
         b'BODY[HEADER.FIELDS (SUBJECT)] {20}', b"Subject: First\r\n\r\n"),
        b")",
        (b"2 (UID 9 BODY[HEADER.FIELDS (SUBJECT)] {21}", b"Subject: Second\r\n\r\n"),
        b" FLAGS (\\Flagged) RFC822.SIZE 50)",
        b"3 (UID 11 FLAGS () BODY[HEADER.FIELDS (SUBJECT)] NIL)",
    ]
    items = m._fetch_items(data)
    metas = [m._meta(meta) for meta, _ in items]
    assert [x["uid"] for x in metas] == [7, 9, 11]
    assert "\\seen" in metas[0]["flags"] and "\\flagged" in metas[1]["flags"]
    assert metas[0]["size"] == 900 and metas[1]["size"] == 50
    assert items[2][1] is None


def test_summary_decodes_headers_and_threads():
    hdr = email.message_from_bytes(
        b"From: =?utf-8?q?Ivana_Kova=C4=8D?= <ivana@k.hr>\r\nSubject: =?utf-8?b?UG9udWRh?=\r\n"
        b"Date: Wed, 05 Nov 2026 09:00:00 +0100\r\nMessage-ID: <c@x>\r\nReferences: <a@x> <b@x>\r\n"
        b"Content-Type: multipart/mixed; boundary=x\r\n\r\n")
    s = m.summary_from(hdr, {"uid": 4, "flags": frozenset(), "internal": "", "size": 1}, "work", 17, "INBOX")
    assert s["from"] == "Ivana Kovač <ivana@k.hr>" and s["subject"] == "Ponuda"
    assert s["id"] == "work:17:4:INBOX" and s["thread_id"] == "work:<a@x>"
    assert s["unread"] and s["has_attachments"] and s["date"].startswith("2026-11-05 09:00")


def test_body_text_prefers_plain_and_falls_back_to_html():
    plain = email.message_from_bytes(b"Content-Type: text/plain; charset=utf-8\r\n\r\nHello \xc4\x8d")
    assert m.body_text(plain) == ("Hello č", False)
    html = email.message_from_bytes(b"Content-Type: text/html\r\n\r\n<p>Hi <b>there</b></p>")
    text, from_html = m.body_text(html)
    assert from_html and "Hi" in text and "<b>" not in text


# ── connections ─────────────────────────────────────────────────────


def test_link_local_and_metadata_hosts_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 993))])
    with pytest.raises(m.MailError) as exc:
        m._check_target("imap.evil.test", 993)
    assert exc.value.status == 400


def test_loopback_skips_certificate_check_only_there():
    assert m._tls("127.0.0.1").verify_mode.name == "CERT_NONE"
    assert m._tls("imap.firma.hr").check_hostname is True


# ── composing ───────────────────────────────────────────────────────


def test_build_message_has_sender_date_and_id_and_bcc_is_stripped_for_the_wire():
    a = _acct(display_name="Ana Horvat")
    data, mid = m.build_message(a, to="bob@x.co", cc="", bcc="eve@x.co", subject="Račun",
                                body="Hi", html_body="", attachments=[Attachment("a.txt", "text/plain", b"x")])
    msg = email.message_from_bytes(data, policy=email.policy.default)
    assert msg["From"] == "Ana Horvat <ana@firma.hr>"
    assert msg["Message-ID"] == mid and mid.endswith("@firma.hr>")
    assert msg["Date"] and msg["Bcc"] == "eve@x.co" and msg["Subject"] == "Račun"
    wire = email.message_from_bytes(m.without_bcc(data), policy=email.policy.default)
    assert wire["Bcc"] is None and wire["Message-ID"] == mid


def test_gmail_builder_unchanged_without_the_new_headers():
    raw = gmail_compose.build_message_bytes(to="a@b.co", subject="x", body="y")
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    assert msg["From"] is None and msg["Message-ID"] is None and msg["Date"] is None


def test_repeat_refusal_names_imap_actions():
    text = m.repeat_refusal([{"kind": "draft", "draft_id": "work:1:2:Drafts", "to": "a@b.co",
                              "subject": "Hi", "message_id": "work:1:2:Drafts"}], lead="Not created")
    assert gmail_compose.REPEAT_MARK in text and "imap_mail action=send_draft draft_id=work:1:2:Drafts" in text
    assert "google_mail" not in text


# ── SMTP outcomes ───────────────────────────────────────────────────


class _FakeSMTP:
    def __init__(self, *, rcpt=None, data_exc=None):
        self.rcpt_codes = rcpt or {}
        self.data_exc = data_exc
        self.calls = []

    def mail(self, sender):
        self.calls.append(("mail", sender))
        return 250, b"ok"

    def rcpt(self, r):
        self.calls.append(("rcpt", r))
        return self.rcpt_codes.get(r, (250, b"ok"))

    def rset(self):
        self.calls.append(("rset",))

    def data(self, payload):
        self.calls.append(("data", len(payload)))
        if self.data_exc:
            raise self.data_exc
        return 250, b"2.0.0 queued as ABC"

    def quit(self):
        self.calls.append(("quit",))


def test_smtp_deliver_ok(monkeypatch):
    fake = _FakeSMTP()
    monkeypatch.setattr(m, "_smtp_connect", lambda a: fake)
    assert "queued" in m.smtp_deliver(_acct(), ["bob@x.co"], b"Subject: x\r\n\r\nhi\r\n")
    assert ("mail", "ana@firma.hr") in fake.calls and fake.calls[-1] == ("quit",)


def test_smtp_refused_recipient_sends_nothing(monkeypatch):
    fake = _FakeSMTP(rcpt={"eve@x.co": (550, b"no such user")})
    monkeypatch.setattr(m, "_smtp_connect", lambda a: fake)
    with pytest.raises(m.MailError) as exc:
        m.smtp_deliver(_acct(), ["bob@x.co", "eve@x.co"], b"x")
    assert not isinstance(exc.value, m.SendOutcomeUnknown)
    assert "eve@x.co" in exc.value.message and not any(c[0] == "data" for c in fake.calls)


def test_smtp_data_refused_is_not_sent(monkeypatch):
    monkeypatch.setattr(m, "_smtp_connect", lambda a: _FakeSMTP(data_exc=smtplib.SMTPDataError(554, b"spam")))
    with pytest.raises(m.MailError) as exc:
        m.smtp_deliver(_acct(), ["bob@x.co"], b"x")
    assert not isinstance(exc.value, m.SendOutcomeUnknown) and "Nothing was sent" in exc.value.message


@pytest.mark.parametrize("exc", [smtplib.SMTPServerDisconnected("gone"), TimeoutError("slow")])
def test_smtp_lost_after_data_is_unknown(monkeypatch, exc):
    monkeypatch.setattr(m, "_smtp_connect", lambda a: _FakeSMTP(data_exc=exc))
    with pytest.raises(m.SendOutcomeUnknown) as err:
        m.smtp_deliver(_acct(), ["bob@x.co"], b"x")
    assert err.value.status == 504 and "may have been sent" in err.value.message


def test_receive_only_account_cannot_connect_smtp():
    with pytest.raises(m.MailError) as exc:
        m._smtp_connect(_acct(smtp_host=""))
    assert exc.value.status == 400 and "receive only" in exc.value.message


# ── setup test: fast probe, other standard port, both servers at once ──


def test_reachable_open_closed_and_black_holed():
    import time as _t

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert m.reachable("127.0.0.1", port, timeout=2)
    finally:
        srv.close()
    assert not m.reachable("127.0.0.1", port, timeout=2)  # closed now: refused, at once
    t = _t.monotonic()
    assert not m.reachable("10.255.255.1", 465, timeout=1.0)  # no answer at all
    assert _t.monotonic() - t < 2.5


def _gmail_like():
    return m.account_from_settings({"email": "me@gmail.com", "imap_host": "imap.gmail.com",
                                    "smtp_host": "smtp.gmail.com", "smtp_port": 465,
                                    "smtp_security": "ssl"}, account_id="g", password="app-pw")


def test_blocked_465_finds_587(monkeypatch):
    monkeypatch.setattr(m, "_check_target", lambda h, p: None)
    monkeypatch.setattr(m, "reachable", lambda h, p, timeout=m.PROBE_TIMEOUT: (h, p) != ("smtp.gmail.com", 465))
    tried = []

    def login(kind, acct):
        tried.append((kind, acct.smtp_port if kind == "smtp" else acct.imap_port))
        return {"ok": True}

    monkeypatch.setattr(m, "_login_check", login)
    res = m.test_account(_gmail_like())
    assert res["imap"]["ok"] and not res["smtp"]["ok"] and res["smtp"]["unreachable"]
    assert res["smtp"]["alternative"] == {"port": 587, "security": "starttls", "ok": True}
    assert "can't be reached" in res["smtp"]["error"] and "587 with STARTTLS" in res["smtp"]["error"]
    assert ("smtp", 465) not in tried and ("smtp", 587) in tried
    acct, notes = m.apply_alternatives(_gmail_like(), res)
    assert (acct.smtp_port, acct.smtp_security) == (587, "starttls")
    assert res["ok"] and notes and "port 465 doesn't answer" in notes[0]


def test_alternative_that_answers_but_refuses_the_login_is_not_applied(monkeypatch):
    monkeypatch.setattr(m, "_check_target", lambda h, p: None)
    monkeypatch.setattr(m, "reachable", lambda h, p, timeout=m.PROBE_TIMEOUT: p != 465)

    def login(kind, acct):
        if kind == "smtp":
            raise m.MailError("needs an app password", 401)
        return {"ok": True}

    monkeypatch.setattr(m, "_login_check", login)
    res = m.test_account(_gmail_like())
    assert res["smtp"]["alternative"]["ok"] is False
    acct, notes = m.apply_alternatives(_gmail_like(), res)
    assert acct.smtp_port == 465 and notes == [] and not res["ok"]


def test_both_servers_checked_at_once(monkeypatch):
    import time as _t

    monkeypatch.setattr(m, "_check_target", lambda h, p: None)
    monkeypatch.setattr(m, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(m, "_login_check", lambda kind, acct: (_t.sleep(0.4), {"ok": True})[1])
    t = _t.monotonic()
    assert m.test_account(_gmail_like())["ok"]
    assert _t.monotonic() - t < 0.75


def test_presets_prefer_587_for_smtp():
    for addr in ("a@gmail.com", "a@yahoo.com", "a@aol.com", "a@fastmail.com", "a@zoho.com"):
        s = m.suggest_settings(addr)
        assert (s["smtp_port"], s["smtp_security"]) == (587, "starttls"), addr


def test_app_password_refusal_in_plain_words(monkeypatch):
    import imaplib

    class Conn:
        capabilities = ("IMAP4REV1",)

        def login(self, u, p):
            raise imaplib.IMAP4.error(b"[ALERT] Application-specific password required: "
                                      b"https://support.google.com/accounts/answer/185833 (Failure)")

    s = m.Session(_gmail_like())
    s.conn = Conn()
    with pytest.raises(m.MailError) as exc:
        s._login()
    assert exc.value.status == 401 and "needs an app password" in exc.value.message
    assert "App passwords" in exc.value.message
