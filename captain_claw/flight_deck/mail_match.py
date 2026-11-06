"""Which of an owner's agents an email is for — pure, no I/O (``agent_mail``).

An email is an agent's when (strongest first):

1. it came to the agent's **own inbox** (the caller says so);
2. To / Cc / Delivered-To has one of the agent's **addresses** (its inbox's
   address, an alias that lands in the owner's inbox like
   ``you+researcher@gmail.com``);
3. the **subject** names the agent — one of the names it answers to, as a word
   ("Researcher: Q3 numbers", "[Researcher] …", "For Researcher");
4. the **body** addresses it explicitly: a greeting ("Hi Researcher", "Dear
   Researcher", "Bok Istraživaču"), the name opening a line ("Researcher, …"),
   ``@Researcher``, "for / attn / za Researcher", or "Researcher, please / could
   you / can you / molim …".

Within one tier the earliest mention wins, then the longest name. Names are
matched as whole words, accents and case folded.

:func:`automated_reason` is the gate before any of it: bulk, list, auto-replies
and no-reply senders are nobody's work.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

BODY_HEAD = 2000          # how much of the body is read for an address to the agent
_NOREPLY_RE = re.compile(
    r"^(?:no[-_.]?reply|do[-_.]?not[-_.]?reply|donotreply|mailer[-_.]daemon|postmaster|bounces?|"
    r"notifications?|notify|alerts?|newsletter|news|marketing|info-noreply)(?:[-_.+].*)?$")
_GREETINGS = (
    "hi", "hello", "hey", "dear", "hiya", "greetings", "good morning", "good afternoon", "good evening",
    "bok", "pozdrav", "zdravo", "poštovani", "postovani", "poštovana", "postovana", "dragi", "draga",
    "hallo", "hola", "ciao", "salut", "bonjour",
)
_ASKS = ("please", "pls", "could you", "can you", "would you", "will you", "kindly", "molim", "možeš", "mozes",
         "možete", "mozete", "trebam", "need you")
_FOR = ("for", "attn", "attn:", "attention", "attention:", "fao", "za", "to:")


@dataclass(frozen=True)
class Agent:
    slug: str
    names: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mail:
    subject: str = ""
    body: str = ""
    to: tuple[str, ...] = ()
    cc: tuple[str, ...] = ()
    delivered_to: tuple[str, ...] = ()
    sender: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    slug: str
    why: str      # "own_inbox" | "address" | "subject" | "body"
    name: str = ""  # the name or address that matched


def fold(text: str) -> str:
    """Lower-cased, accents off, ``đ`` → ``d`` (as the board folds)."""
    text = str(text or "").replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


def _addr(value: str) -> str:
    return str(value or "").strip().strip("<>").lower()


def automated_reason(mail: Mail) -> str:
    """Why this email is nobody's work ("" when it may be someone's): an
    auto-reply, a list or bulk mail, a no-reply sender."""
    h = {str(k).lower(): str(v or "") for k, v in (mail.headers or {}).items()}
    auto = h.get("auto-submitted", "").strip().lower()
    if auto and auto != "no":
        return "an automatic email"
    if h.get("precedence", "").strip().lower() in ("bulk", "list", "junk", "auto_reply"):
        return "bulk mail"
    if h.get("list-id") or h.get("list-unsubscribe") or h.get("list-post"):
        return "a mailing list or newsletter"
    if h.get("x-autoreply") or h.get("x-autorespond"):
        return "an automatic email"
    local = _addr(mail.sender).split("@", 1)[0]
    if not local:
        return "no sender"
    if _NOREPLY_RE.match(local):
        return "a no-reply sender"
    return ""


def _name_pattern(name: str) -> re.Pattern | None:
    folded = fold(" ".join(str(name or "").split()))
    if len(folded) < 2:
        return None
    return re.compile(r"(?<![\w@.])" + re.escape(folded).replace(r"\ ", r"\s+") + r"(?![\w])")


def _first(text: str, agents: Iterable[Agent]) -> tuple[int, int, Agent, str] | None:
    """The earliest whole-word mention of any agent's name in *text*:
    ``(position, -length, agent, name)``."""
    best = None
    for agent in agents:
        for name in agent.names:
            pat = _name_pattern(name)
            if pat is None:
                continue
            m = pat.search(text)
            if m:
                cand = (m.start(), -len(m.group(0)), agent, name)
                if best is None or cand[:2] < best[:2]:
                    best = cand
    return best


def _addressed(body: str, agent: Agent) -> tuple[int, str] | None:
    """Where the body addresses *agent* explicitly, the earliest: ``(pos, name)``."""
    best = None
    for name in agent.names:
        folded = fold(" ".join(str(name or "").split()))
        if len(folded) < 2:
            continue
        n = re.escape(folded).replace(r"\ ", r"\s+")
        greet = "|".join(re.escape(fold(g)).replace(r"\ ", r"\s+") for g in _GREETINGS)
        asks = "|".join(re.escape(fold(a)).replace(r"\ ", r"\s+") for a in _ASKS)
        fors = "|".join(re.escape(f) for f in _FOR)
        patterns = (
            rf"(?m)^[ \t>]*(?:{greet})[ \t,]+{n}(?![\w])",          # Hi Researcher
            rf"(?m)^[ \t]*{n}[ \t]*[,:!—–-]",                     # Researcher, …
            rf"(?<![\w.])@{n}(?![\w])",                           # @Researcher
            rf"(?<![\w])(?:this\s+is\s+)?(?:{fors})\s+{n}(?![\w])",  # for Researcher, attn Researcher, za …
            rf"(?<![\w]){n},?\s+(?:{asks})(?![\w])",              # Researcher, please …
        )
        for p in patterns:
            m = re.search(p, body)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), name)
    return best


def match(mail: Mail, agents: list[Agent], *, own_inbox: str = "") -> Match | None:
    """The agent this email is for, or None. *own_inbox* is the slug whose own
    inbox it came to (that is the answer, whatever it says)."""
    if own_inbox:
        return Match(own_inbox, "own_inbox")
    rcpts = [_addr(a) for a in (*mail.to, *mail.cc, *mail.delivered_to) if a]
    for addr in rcpts:
        for agent in agents:
            if addr in {_addr(a) for a in agent.addresses}:
                return Match(agent.slug, "address", addr)
    hit = _first(fold(mail.subject), agents)
    if hit:
        return Match(hit[2].slug, "subject", hit[3])
    body = fold(str(mail.body or "")[:BODY_HEAD])
    best = None
    for agent in agents:
        found = _addressed(body, agent)
        if found and (best is None or found[0] < best[0]):
            best = (found[0], agent, found[1])
    if best:
        return Match(best[1].slug, "body", best[2])
    return None


def why_words(m: Match, *, where: str, own_email: str = "") -> str:
    """How the agent is told the email came to it."""
    if m.why == "own_inbox":
        return f"in your own inbox{f' ({own_email})' if own_email else ''}"
    if m.why == "address":
        return f"to your address {m.name}, in {where}"
    if m.why == "subject":
        return f"in {where}, with your name in the subject"
    return f"in {where}, addressed to you in its body"
