"""Sites that block the agent — and asking the person to lend their own Chrome.

Some sites refuse automated visitors: a Cloudflare / DataDome / PerimeterX /
Imperva / Akamai bot check, a bare 403, or a connection dropped mid-handshake.
The headless browser on the server and ``web_fetch`` both look automated, and
the agent's own Captain for Chrome (``extension_connect``) runs in that same
server browser, so all of them get the same wall. The person's own Chrome —
Captain for Chrome on their computer, signed in as them — usually gets
through.

So the ``browser`` tool and ``web_fetch`` report what blocked them here, per
session and site:

* one of them blocked → its result says to try the other once;
* both blocked → the result tells the agent to ask the person, with
  ``ask_user``, whether it may use their Chrome — and the question fits what
  Flight Deck knows (``GET /fd/browser-hands/agent/status``): their Chrome is
  connected for agents (ask may I), Captain for Chrome is set up but not open
  to agents (ask them to turn on "Agents can use this browser"), or it isn't
  set up at all (recommend installing it, and how). If they stopped this agent
  in the extension, it doesn't ask.

Nothing here acts on its own: the agent asks, and only on a yes goes on with
``browser(target="chrome")``. A shared agent's member turn gets none of this
(the owner's Chrome is never theirs to ask for), and without Flight Deck
there's nobody's Chrome to offer.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

BROWSER = "browser"
FETCH = "fetch"
KEEP_S = 2 * 3600.0           # a site's record lasts this long in a session
MAX_RECORDS = 500
SHORT_PAGE_CHARS = 1500       # a bot wall is a short page; long ones are real pages
STATUS_TIMEOUT_S = 6.0

# HTTP answers that mean "not for you" rather than "not here": a sign-in, a
# refusal, rate limiting, LinkedIn's 999. A bare 503 is usually just a site
# being down, so it counts only with a bot check's marks on the page.
BLOCK_STATUSES = {401: "it asks for a sign-in", 403: "it refuses automated visitors",
                  429: "it rate-limits automated visitors", 999: "it refuses automated visitors"}

# Page titles bot checks use, whole.
_WALL_TITLES = frozenset({
    "just a moment", "just a moment...", "one moment, please...", "please wait...", "please wait",
    "attention required!", "attention required! | cloudflare", "access denied", "forbidden", "403 forbidden",
    "error 403", "robot check", "security check", "human verification", "are you a robot?",
    "are you a robot", "bot verification", "captcha", "captcha challenge", "request rejected", "blocked",
    "pardon our interruption", "access to this page has been denied", "access to this page has been denied.",
    "you have been blocked", "sorry, you have been blocked", "checking your browser",
    "verify you are human", "please verify you are a human", "client challenge", "too many requests",
    "429 too many requests", "ddos-guard", "verifying...", "verifying you are human",
})
# …and phrases in them distinctive enough to count anywhere in a title.
_WALL_TITLE_BITS = (
    "| cloudflare", "attention required", "just a moment", "checking your browser", "verify you are human",
    "are you a robot", "ddos-guard", "pardon our interruption", "access to this page has been denied",
    "you have been blocked", "checking if the site connection is secure",
)
# Marks of a bot check in a SHORT page's text or markup, with the vendor. Only
# a challenge's own marks: the scripts these vendors put on ordinary pages
# (Cloudflare's jsd, Imperva's _Incapsula_Resource, DataDome's and
# PerimeterX's tags) are no sign of a wall.
_WALL_MARKS: tuple[tuple[str, str], ...] = (
    ("enable javascript and cookies to continue", "Cloudflare"),
    ("checking if the site connection is secure", "Cloudflare"),
    ("needs to review the security of your connection", "Cloudflare"),
    ("performance & security by cloudflare", "Cloudflare"),
    ("why have i been blocked", "Cloudflare"),
    ("cf-chl-", "Cloudflare"),
    ("cf_chl_opt", "Cloudflare"),
    ("captcha-delivery.com", "DataDome"),
    ("px-captcha", "PerimeterX"),
    ("press & hold", "PerimeterX"),
    ("press and hold", "PerimeterX"),
    ("incapsula incident", "Imperva"),
    ("pardon our interruption", "Imperva"),
    ("the requested url was rejected", "F5"),
    ("sec-if-cpt", "Akamai"),
    ("ddos-guard", "DDoS-Guard"),
    ("checking your browser before accessing", ""),
    ("verify you are human", ""),
    ("verifying you are human", ""),
    ("please verify you are a human", ""),
    ("please complete the security check", ""),
    ("sorry, you have been blocked", ""),
    ("you have been blocked", ""),
    ("unusual traffic from your computer", ""),
    ("enter the characters you see below", ""),
    ("are you a robot", ""),
    ("not a robot", ""),
    ("you don't have permission to access", ""),
    ("access denied", ""),
)
# Browser errors that are how some walls answer a headless browser
# (Akamai drops HTTP/2 mid-stream, others reset or send nothing).
_NET_WALLS = ("err_http2_protocol_error", "err_connection_reset", "err_empty_response",
              "err_connection_closed", "err_too_many_redirects", "err_blocked_by_response")
_NOT_A_WALL = ("err_name_not_resolved", "name or service not known", "nodename nor servname",
               "getaddrinfo failed", "err_internet_disconnected", "err_address_unreachable",
               "err_cert_", "certificate verify failed", "err_invalid_url", "err_file_not_found",
               "err_aborted")

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

# (session, site) → what blocked each way, when, and whether the agent was
# already told to ask.
_SEEN: dict[tuple[str, str], dict[str, Any]] = {}


# ── telling a wall from a page ───────────────────────────────────────


def title_of(html: str) -> str:
    m = _TITLE_RE.search(html or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _vendor(text: str) -> str:
    low = text.lower()
    for mark, vendor in _WALL_MARKS:
        if vendor and mark in low:
            return vendor
    return "Cloudflare" if "cloudflare" in low or "ray id:" in low else ""


def block_reason(status: int | None = None, title: str = "", text: str = "", html: str = "") -> str:
    """Why this answer looks like a bot wall rather than the page — ``""`` when
    it looks like the page (or just a missing one: a 404 isn't a wall)."""
    t = re.sub(r"\s+", " ", str(title or "")).strip().lower()
    if not t and html:
        t = title_of(html).lower()
    body = str(text or "").strip()
    short = len(body) < SHORT_PAGE_CHARS
    walled = t in _WALL_TITLES or any(bit in t for bit in _WALL_TITLE_BITS)
    seen = ""
    if not walled and short:
        hay = (body[:SHORT_PAGE_CHARS] + "\n" + str(html or "")[:40_000]).lower()
        seen = next((mark for mark, _v in _WALL_MARKS if mark in hay), "")
        walled = bool(seen)
    code = int(status or 0)
    if walled:
        vendor = _vendor(" ".join((t, body[:SHORT_PAGE_CHARS], str(html or "")[:40_000])))
        why = f"a bot check ({vendor})" if vendor else "a bot check"
        if t and not seen:
            why += f", page titled “{str(title or title_of(html)).strip()[:80]}”"
        return why + (f", HTTP {code}" if code >= 400 else "")
    if code in BLOCK_STATUSES:
        return f"{BLOCK_STATUSES[code]} (HTTP {code})"
    return ""


def net_reason(message: str) -> str:
    """A browser or HTTP error that's how walls answer — ``""`` for one that
    isn't (no such site, no network, a bad certificate)."""
    low = str(message or "").lower()
    if any(m in low for m in _NOT_A_WALL):
        return ""
    for m in _NET_WALLS:
        if m in low:
            return f"the site dropped the connection ({m.upper()})"
    if "timeout" in low and "exceeded" in low:
        return "the page never finished loading"
    return ""


def fetch_error_reason(exc: BaseException) -> tuple[str, bool]:
    """``(why, strong)`` for a fetch that raised — strong when it's clearly a
    refusal (a 401/403/429), weak when the site just didn't answer; ``("", _)``
    when it's not about blocking (a 404, no such site)."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code if exc.response is not None else 0
        why = block_reason(status=code, html=_safe_text(exc.response))
        return why, bool(why)
    text = f"{type(exc).__name__}: {exc}"
    why = net_reason(text)
    if why:
        return why, False
    if isinstance(exc, (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError)):
        if any(m in text.lower() for m in _NOT_A_WALL):
            return "", False
        return f"no answer from the site ({type(exc).__name__})", False
    return "", False


def _safe_text(resp: Any) -> str:
    try:
        return str(resp.text or "")[:40_000]
    except Exception:  # noqa: BLE001
        return ""


# ── what each session saw ────────────────────────────────────────────


def site_of(url: str) -> str:
    try:
        host = (urlparse(str(url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _session(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("_session_id") or "").strip() or "default"


def _member_turn(kwargs: dict[str, Any]) -> bool:
    try:
        from captain_claw import speaker

        return speaker.principal_for(kwargs.get("_agent")) is not None
    except Exception:  # noqa: BLE001
        return False


def _prune(now: float) -> None:
    for key in [k for k, v in _SEEN.items() if now - v.get("at", 0) > KEEP_S]:
        _SEEN.pop(key, None)
    while len(_SEEN) > MAX_RECORDS:
        _SEEN.pop(min(_SEEN, key=lambda k: _SEEN[k].get("at", 0)), None)


def forget(kwargs: dict[str, Any], url: str, via: str = "") -> None:
    """*via* got the page after all (or the person's Chrome did, ``via=""``)."""
    key = (_session(kwargs), site_of(url))
    rec = _SEEN.get(key)
    if rec is None:
        return
    if not via:
        _SEEN.pop(key, None)
        return
    rec.pop(via, None)
    if not (rec.get(BROWSER) or rec.get(FETCH)):
        _SEEN.pop(key, None)


def recent_block(kwargs: dict[str, Any]) -> str:
    """The site this session was last blocked on, if it was lately (``""`` if none)."""
    sid = _session(kwargs)
    now = time.time()
    hits = [(v.get("at", 0), site) for (s, site), v in _SEEN.items()
            if s == sid and now - v.get("at", 0) <= KEEP_S and (v.get(BROWSER) or v.get(FETCH))]
    return max(hits)[1] if hits else ""


async def chrome_status() -> dict[str, Any] | None:
    """What Flight Deck knows about the person's Chrome for this agent, or
    ``None`` (no Flight Deck, or it didn't say)."""
    from captain_claw.fd_client import flight_deck_base, flight_deck_headers

    base = flight_deck_base()
    if not base:
        return None
    try:
        async with httpx.AsyncClient(timeout=STATUS_TIMEOUT_S) as client:
            resp = await client.get(f"{base}/fd/browser-hands/agent/status", headers=flight_deck_headers())
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.info("Chrome status unavailable", error=str(exc))
        return None
    return data if isinstance(data, dict) else None


async def note_failure(kwargs: dict[str, Any], url: str, via: str, why: str, *, strong: bool = True) -> str:
    """Remember that *via* (``"browser"`` / ``"fetch"``) couldn't get *url* —
    and the note for the agent: try the other way, or (both failed) ask the
    person for their Chrome. ``""`` when there's nothing to say."""
    site = site_of(url)
    if not site or not why or via not in (BROWSER, FETCH):
        return ""
    if _member_turn(kwargs):
        return ""
    now = time.time()
    _prune(now)
    key = (_session(kwargs), site)
    rec = _SEEN.setdefault(key, {})
    rec[via] = {"why": why, "strong": bool(strong)}
    rec["at"] = now
    rec["url"] = str(url).strip()
    other = FETCH if via == BROWSER else BROWSER
    seen_other = rec.get(other)
    if not seen_other:
        if not strong:
            return ""
        if via == BROWSER:
            return ("\n\n--- Blocked ---\n"
                    f"{site} blocks your headless browser ({why}). Try a plain fetch once: "
                    f"web_fetch(url=\"{rec['url']}\", deep_fetch=false). If that can't get it either, the person's "
                    "own Chrome can — you'll be told how to ask them.")
        return ("\n\n--- Blocked ---\n"
                f"web_fetch couldn't get {site} ({why}). Try the browser once: "
                f"browser(action=\"navigate\", url=\"{rec['url']}\"). If it's blocked there too, the person's own "
                "Chrome can — you'll be told how to ask them.")
    if not (strong or seen_other.get("strong")):
        return ""   # two shrugs (timeouts) — maybe the site is just down
    if rec.get("hinted"):
        return ("\n\n--- Blocked ---\n"
                f"{site} still blocks you ({why}). If you haven't yet, ask the person (ask_user) once whether you "
                "may use their own Chrome (browser target=\"chrome\"); if they said no, don't ask again — say you "
                "couldn't get it.")
    status = await chrome_status()
    from captain_claw.fd_client import flight_deck_base

    if status is None and not flight_deck_base():
        return ""   # nobody's Chrome to offer without Flight Deck
    rec["hinted"] = True
    browser_why = (rec.get(BROWSER) or {}).get("why", "")
    fetch_why = (rec.get(FETCH) or {}).get("why", "")
    return ask_chrome_note(site, rec["url"], browser_why, fetch_why, status)


def ask_chrome_note(site: str, url: str, browser_why: str, fetch_why: str, status: dict[str, Any] | None) -> str:
    """Both ways are blocked: how to ask the person for their Chrome, fitting
    what Flight Deck said about it."""
    head = ("\n\n--- Blocked: ask to use their Chrome ---\n"
            f"{site} blocks your headless browser ({browser_why}) and web_fetch couldn't get it either "
            f"({fetch_why}). Your server browsers — extension_connect's too — get the same wall; the person's own "
            "Chrome (Captain for Chrome on their computer, signed in as them) usually gets through.\n")
    go = f"Yes → browser(action=\"navigate\", url=\"{url}\", target=\"chrome\"), then page_text / read_page there."
    tail = ("\nNo / not now → don't ask again for this site; say you couldn't get it and why. If ask_user can't "
            "reach them (a chat channel, or a turn that started on its own), ask in your reply instead.")
    question = f"{site} blocks my browser on the server."
    if status and status.get("stopped"):
        return head + ("They stopped you using their Chrome in Captain for Chrome, so don't use it or ask: tell "
                       f"them {site} blocks you, and that letting you use their Chrome again would get it.")
    if status and status.get("person_connected", status.get("connected") and not status.get("own_connected")):
        return head + (
            "Their Chrome is connected for agents. Ask first, once:\n"
            f"ask_user(question=\"{question} May I open it in your Chrome (Captain for Chrome) instead?\", "
            "confirm=true, help=\"I'll work in a tab of my own there; you can watch and stop me.\")\n" + go + tail)
    if status and status.get("extension_set_up"):
        return head + (
            "Captain for Chrome is set up in their browser but isn't open to agents right now (Chrome is closed, or "
            "its Settings → \"Agents can use this browser\" is off). Ask, once:\n"
            f"ask_user(question=\"{question} Can I use your Chrome for it? Open Captain for Chrome → Settings, turn on "
            "“Agents can use this browser”, then answer here.\", choices=[\"It's on — go ahead\", \"Not now\"])\n"
            + go.replace("Yes →", "Go ahead →") + tail)
    if status and status.get("extension_set_up") is False:
        return head + (
            "They don't have Captain for Chrome yet — recommend it. Ask, once:\n"
            f"ask_user(question=\"{question} With Captain for Chrome in your Chrome I could open it there, signed in "
            "as you. Want to set it up?\", choices=[\"It's set up — go ahead\", \"Not now\"], help=\"In Flight Deck: "
            "Connections → Browser → Download the extension and unzip it; in chrome://extensions turn on Developer "
            "mode, Load unpacked and pick the folder; pair it with this deck, then in its Settings turn on “Agents "
            "can use this browser”.\")\n" + go.replace("Yes →", "Go ahead →") + tail)
    return head + (
        "Ask, once:\n"
        f"ask_user(question=\"{question} May I use your Chrome (Captain for Chrome) for it?\", "
        "choices=[\"Yes — it's on\", \"Not now\"], help=\"Not installed yet? Flight Deck → Connections → Browser shows "
        "how; then its Settings → “Agents can use this browser”.\")\n" + go + tail)


def chrome_missing_note(kwargs: dict[str, Any]) -> str:
    """Said when ``target="chrome"`` finds no Chrome right after a site
    blocked this session — instead of pointing back at the headless browser."""
    site = recent_block(kwargs)
    if not site:
        return ""
    return (f"Their Chrome still isn't connected for agents, and {site} blocks your headless browser (your own "
            "Chrome via extension_connect too), so only theirs helps here. Ask them (ask_user) to open Captain for "
            "Chrome → Settings and turn on “Agents can use this browser” — or, if they don't have it, to set it up "
            "from Flight Deck → Connections → Browser — then try again. If they'd rather not, say you couldn't get "
            "the page and why.")


def own_browser_note(kwargs: dict[str, Any]) -> str:
    """Said when a ``target="chrome"`` act ran in the agent's own server
    browser while a site blocks this session."""
    site = recent_block(kwargs)
    if not site:
        return ""
    return (f"\n(That ran in your own server browser, which {site} blocks — the person's Chrome isn't connected. "
            "Ask them (ask_user) to turn on “Agents can use this browser” in Captain for Chrome if you need it.)")
